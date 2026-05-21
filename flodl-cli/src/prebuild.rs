//! Pre-flight build for cluster commands.
//!
//! Heterogeneous-rig pain: source lives on a shared mount (NFS /
//! virtiofs / S3-FUSE) so editing on the controller is visible to
//! remote hosts, but each host needs its own libtorch-linked binary
//! (cu128 for a Blackwell host, cu126-pt27 for a Pascal host). Without
//! pre-flight, the first `fdl @cluster <cmd>` after an edit hits stale
//! remote binaries — confusing runtime errors or worse, silent wrong
//! behaviour.
//!
//! This module runs `cargo build` LOCALLY on the controller, once per
//! remote host, into per-host `target/cluster/<host>/` directories
//! with the right libtorch bind-mounted. The shared mount delivers the
//! resulting binary to the remote, which execs it directly (no cargo,
//! no rust toolchain on remote). Per-host target dirs isolate cargo's
//! fingerprint cache so a libtorch swap on one host doesn't invalidate
//! anyone else's incremental build.
//!
//! Convention: command name == binary name. `fdl @cluster ddp-bench`
//! builds `--bin ddp-bench`. Features derive from the host's libtorch
//! `.arch` (cuda=12.x → `--features cuda`, cuda=none → no features).
//!
//! Builds run in parallel across hosts (per-host target dirs ⇒ zero
//! contention). First failure aborts the rest; remaining builds finish
//! their current cargo invocation but their stderr is collected and
//! surfaced together so the user sees every host's diagnostic.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};

use crate::config::{ClusterConfig, ClusterHost};
use crate::libtorch::detect::LibtorchInfo;

/// Env var carrying the per-host pre-flight build envelope (a JSON
/// map) from fdl-cli's prebuild phase to flodl's launcher. The
/// launcher reads it on the controller side just before fan-out and
/// substitutes the direct-binary form for each host whose entry is
/// present.
///
/// Map shape: `{ "<host-name>": { "bin": "<relative path under
/// host.path>", "ld_library_path": "<absolute LD_LIBRARY_PATH>" }, ...
/// }`. Hosts absent from the map fall back to the launcher's existing
/// `fdl <cmd>` re-entry on the remote.
pub const ENV_PREBUILD_PER_HOST: &str = "FLODL_PREBUILD_PER_HOST";

/// Run pre-flight builds for every remote host in `cluster`. The
/// controller itself is skipped — its build is handled by the normal
/// dispatch path (`cargo run` in Docker against the local `.active`).
///
/// `cmd_name` is both the fdl command and the cargo `--bin` target.
/// `controller_host` is the local hostname (skipped from the remotes).
///
/// Each per-host build runs in a Docker compose service (`cuda` when
/// the host's libtorch advertises a CUDA version in `.arch`, `dev`
/// otherwise). The build's env is overridden so:
///   - `LIBTORCH_HOST_PATH` points at the resolved host libtorch dir
///   - `CARGO_TARGET_DIR` points at `target/cluster/<host>/`
///
/// Returns `Ok(())` on universal success. Returns `Err(combined_msg)`
/// listing every host that failed (with its stderr tail) on any
/// failure. Builds running when a failure surfaces complete to natural
/// stopping — cargo's per-crate granularity means cancelling mid-build
/// would leave the per-host target dir in a half-baked state.
pub fn prebuild_remotes(
    project_root: &Path,
    cluster: &ClusterConfig,
    cmd_name: &str,
    controller_host: &str,
) -> Result<(), String> {
    let remotes: Vec<&ClusterHost> = cluster
        .hosts
        .iter()
        .filter(|h| h.name != controller_host)
        .collect();
    if remotes.is_empty() {
        return Ok(());
    }

    eprintln!(
        "fdl: pre-flight build for {} remote host(s): {}",
        remotes.len(),
        remotes.iter().map(|h| h.name.as_str()).collect::<Vec<_>>().join(", "),
    );

    let project_root = Arc::new(project_root.to_path_buf());
    let cmd_name = Arc::new(cmd_name.to_string());
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let envelope: Arc<Mutex<BTreeMap<String, PerHostEnvelope>>> =
        Arc::new(Mutex::new(BTreeMap::new()));

    let mut handles = Vec::with_capacity(remotes.len());
    for host in remotes {
        let host = host.clone();
        let project_root = Arc::clone(&project_root);
        let cmd_name = Arc::clone(&cmd_name);
        let errors = Arc::clone(&errors);
        let envelope = Arc::clone(&envelope);
        handles.push(thread::spawn(move || {
            match prebuild_one_host(&project_root, &host, &cmd_name) {
                Ok(env_entry) => {
                    eprintln!("fdl: pre-flight OK ({})", host.name);
                    envelope.lock().unwrap().insert(host.name.clone(), env_entry);
                }
                Err(e) => {
                    eprintln!("fdl: pre-flight FAILED ({}): {}", host.name, e);
                    errors.lock().unwrap().push(format!("{}: {}", host.name, e));
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    let errs = Arc::try_unwrap(errors)
        .map_err(|_| "internal: error collector still has outstanding refs".to_string())?
        .into_inner()
        .map_err(|e| format!("internal: errors mutex poisoned: {e}"))?;
    if !errs.is_empty() {
        return Err(format!(
            "pre-flight build failed on {} host(s):\n  {}",
            errs.len(),
            errs.join("\n  "),
        ));
    }

    // Emit the per-host envelope so the flodl launcher's remote
    // dispatch can substitute the direct-binary form for each host
    // (skipping the `fdl <cmd>` re-entry — no cargo on remote).
    let env_map = Arc::try_unwrap(envelope)
        .map_err(|_| "internal: envelope still has outstanding refs".to_string())?
        .into_inner()
        .map_err(|e| format!("internal: envelope mutex poisoned: {e}"))?;
    let json = serde_json::to_string(&env_map)
        .map_err(|e| format!("internal: serialize prebuild envelope: {e}"))?;
    // SAFETY: main has not spawned threads at this point in dispatch.
    unsafe { std::env::set_var(ENV_PREBUILD_PER_HOST, json); }
    Ok(())
}

/// Per-host pre-flight build artifact descriptor — exactly what the
/// launcher needs to substitute the remote dispatch with a direct
/// binary exec. Mirrors `flodl::distributed::launcher::PerHostPrebuild`
/// on the consumer side; the two structs share an on-the-wire JSON
/// schema but are independent types because the crates can't share
/// declarations without a circular dep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerHostEnvelope {
    /// Path to the compiled binary, relative to the host's project
    /// checkout (`host.path`). e.g.
    /// `target/cluster/flodl-pascal/release/ddp-bench`.
    pub bin: String,
    /// Absolute path the launcher should set as `LD_LIBRARY_PATH` so
    /// the binary finds its libtorch at runtime. e.g.
    /// `/home/me/rdl/libtorch/builds/sm61-sm120/lib`. The launcher may
    /// append host-specific extras (e.g. `:/usr/local/lib` for bare-
    /// metal libnccl) via `host.env: { LD_LIBRARY_PATH: ... }`.
    pub ld_library_path: String,
}

/// Build `cmd_name` for one host. Picks docker service + cargo
/// features from the host's libtorch metadata. Returns a
/// [`PerHostEnvelope`] describing where the resulting binary lives
/// (so the launcher can substitute it on the remote-dispatch path)
/// and what `LD_LIBRARY_PATH` the remote should set.
fn prebuild_one_host(
    project_root: &Path,
    host: &ClusterHost,
    cmd_name: &str,
) -> Result<PerHostEnvelope, String> {
    let libtorch_path = host.libtorch_path.as_ref().ok_or_else(|| {
        format!(
            "host {:?} has no `libtorch_path:` set in cluster.yml — \
             pre-flight build needs one to resolve the host's libtorch",
            host.name,
        )
    })?;
    let (info, host_path) = crate::run::resolve_libtorch_at(Path::new(libtorch_path))
        .ok_or_else(|| {
            format!(
                "host {:?}: `libtorch_path: {libtorch_path}` did not resolve \
                 to a valid libtorch (pointer file, libtorch root with .active, \
                 or direct variant dir with lib/)",
                host.name,
            )
        })?;
    let (features_arg, docker_svc) = features_and_service(&info);
    let target_dir = format!("target/cluster/{}", host.name);

    let build_cmd = if features_arg.is_empty() {
        format!(
            "cargo build --release --bin {bin}",
            bin = posix_quote(cmd_name),
        )
    } else {
        format!(
            "cargo build --release --features {feat} --bin {bin}",
            feat = posix_quote(features_arg),
            bin = posix_quote(cmd_name),
        )
    };

    let docker_cmd = format!(
        "docker compose run --rm {svc} bash -c {inner}",
        svc = docker_svc,
        inner = posix_quote(&build_cmd),
    );

    let mut cmd = Command::new("sh");
    cmd.args(["-c", &docker_cmd])
        .current_dir(project_root)
        .env("LIBTORCH_HOST_PATH", &host_path)
        .env(
            "LIBTORCH_CPU_PATH",
            "./libtorch/precompiled/cpu",
        )
        .env("CARGO_TARGET_DIR", &target_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::null());

    if let Some(cuda_version) = &info.cuda_version {
        if cuda_version != "none" {
            let normalised = if cuda_version.matches('.').count() < 2 {
                format!("{cuda_version}.0")
            } else {
                cuda_version.clone()
            };
            let cuda_tag = normalised
                .splitn(3, '.')
                .take(2)
                .collect::<Vec<_>>()
                .join(".");
            cmd.env("CUDA_VERSION", &normalised);
            cmd.env("CUDA_TAG", &cuda_tag);
        }
    }

    let status = cmd
        .status()
        .map_err(|e| format!("spawn `{docker_cmd}`: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cargo build exited {} (libtorch={host_path}, target={target_dir}, \
             features={feat})",
            status.code().unwrap_or(-1),
            feat = if features_arg.is_empty() { "(none)" } else { features_arg },
        ));
    }
    Ok(PerHostEnvelope {
        bin: format!("{target_dir}/release/{cmd_name}"),
        ld_library_path: format!("{host_path}/lib"),
    })
}

/// Pick cargo features + docker compose service from the host's
/// libtorch `.arch` metadata. `cuda=12.x` → (`cuda`, `cuda`); anything
/// else → (`""`, `dev`).
fn features_and_service(info: &LibtorchInfo) -> (&'static str, &'static str) {
    match info.cuda_version.as_deref() {
        Some(v) if v != "none" => ("cuda", "cuda"),
        _ => ("", "dev"),
    }
}

/// Single-quote a string for `sh -c` so embedded spaces/quotes don't
/// break the outer shell parse. Local copy to avoid `pub(crate)`
/// promotion of `run::posix_quote`.
fn posix_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = s.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ',' | '=' | ':')
    });
    if safe {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_and_service_cuda_present_picks_cuda() {
        let info = LibtorchInfo {
            path: "precompiled/cu128".into(),
            torch_version: Some("2.10.0".into()),
            cuda_version: Some("12.8".into()),
            archs: Some("8.0".into()),
            source: Some("precompiled".into()),
        };
        assert_eq!(features_and_service(&info), ("cuda", "cuda"));
    }

    #[test]
    fn features_and_service_cuda_none_picks_dev() {
        let info = LibtorchInfo {
            path: "precompiled/cpu".into(),
            torch_version: Some("2.10.0".into()),
            cuda_version: Some("none".into()),
            archs: None,
            source: Some("precompiled".into()),
        };
        assert_eq!(features_and_service(&info), ("", "dev"));
    }

    #[test]
    fn features_and_service_no_arch_defaults_to_dev() {
        let info = LibtorchInfo {
            path: "unknown".into(),
            torch_version: None,
            cuda_version: None,
            archs: None,
            source: None,
        };
        assert_eq!(features_and_service(&info), ("", "dev"));
    }

    #[test]
    fn posix_quote_round_trips_safe_strings() {
        assert_eq!(posix_quote("ddp-bench"), "ddp-bench");
        assert_eq!(posix_quote("target/cluster/exa"), "target/cluster/exa");
        assert_eq!(posix_quote(""), "''");
    }

    #[test]
    fn posix_quote_wraps_unsafe_strings() {
        assert_eq!(posix_quote("a b"), "'a b'");
        assert_eq!(posix_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn envelope_serializes_to_stable_json() {
        let mut env = BTreeMap::new();
        env.insert(
            "host-b".to_string(),
            PerHostEnvelope {
                bin: "target/cluster/host-b/release/bench".into(),
                ld_library_path: "/opt/lt-b/lib".into(),
            },
        );
        env.insert(
            "host-a".to_string(),
            PerHostEnvelope {
                bin: "target/cluster/host-a/release/bench".into(),
                ld_library_path: "/opt/lt-a/lib".into(),
            },
        );
        let json = serde_json::to_string(&env).unwrap();
        // BTreeMap iterates in sorted key order ⇒ stable JSON output
        // regardless of insertion order.
        assert_eq!(
            json,
            r#"{"host-a":{"bin":"target/cluster/host-a/release/bench","ld_library_path":"/opt/lt-a/lib"},"host-b":{"bin":"target/cluster/host-b/release/bench","ld_library_path":"/opt/lt-b/lib"}}"#,
        );
    }

    #[test]
    fn envelope_round_trips_through_serde() {
        let mut env = BTreeMap::new();
        env.insert(
            "h1".to_string(),
            PerHostEnvelope {
                bin: "t/c/h1/release/x".into(),
                ld_library_path: "/opt/lt/lib".into(),
            },
        );
        let json = serde_json::to_string(&env).unwrap();
        let back: BTreeMap<String, PerHostEnvelope> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
        let e = back.get("h1").unwrap();
        assert_eq!(e.bin, "t/c/h1/release/x");
        assert_eq!(e.ld_library_path, "/opt/lt/lib");
    }
}
