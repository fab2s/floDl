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
    cmd_cwd: &Path,
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

    // Whether the controller runs builds inside Docker. Sourced from
    // the controller's own `docker:` field in cluster.yml (if listed)
    // or `None` when absent — native-Rust controllers (no docker
    // installed) get the bare cargo invocation. Falls back to None
    // when the controller isn't listed in cluster.hosts (e.g.
    // orchestrator-only mode); the bare path is the safe default.
    let controller_docker_svc: Option<String> = cluster
        .hosts
        .iter()
        .find(|h| h.name == controller_host)
        .and_then(|h| h.docker.clone());

    eprintln!(
        "fdl: pre-flight build for {} remote host(s): {}",
        remotes.len(),
        remotes.iter().map(|h| h.name.as_str()).collect::<Vec<_>>().join(", "),
    );

    // Controller's view of the shared project root. Falls back to the
    // controller host's own `path:` when `cluster.controller_path` is
    // unset (homogeneous-mount rigs, the common case).
    let controller_path: std::path::PathBuf = cluster
        .controller_path
        .as_deref()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            cluster
                .hosts
                .iter()
                .find(|h| h.name == controller_host)
                .map(|h| std::path::PathBuf::from(&h.path))
        })
        .unwrap_or_else(|| project_root.to_path_buf());

    let project_root = Arc::new(project_root.to_path_buf());
    let cmd_cwd = Arc::new(cmd_cwd.to_path_buf());
    let cmd_name = Arc::new(cmd_name.to_string());
    let controller_path = Arc::new(controller_path);
    let controller_docker_svc = Arc::new(controller_docker_svc);
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let envelope: Arc<Mutex<BTreeMap<String, PerHostEnvelope>>> =
        Arc::new(Mutex::new(BTreeMap::new()));

    let mut handles = Vec::with_capacity(remotes.len());
    for host in remotes {
        let host = host.clone();
        let project_root = Arc::clone(&project_root);
        let cmd_cwd = Arc::clone(&cmd_cwd);
        let cmd_name = Arc::clone(&cmd_name);
        let controller_path = Arc::clone(&controller_path);
        let controller_docker_svc = Arc::clone(&controller_docker_svc);
        let errors = Arc::clone(&errors);
        let envelope = Arc::clone(&envelope);
        handles.push(thread::spawn(move || {
            match prebuild_one_host(
                &project_root, &cmd_cwd, &controller_path,
                &host, &cmd_name,
                controller_docker_svc.as_deref(),
            ) {
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
    /// Subdirectory under the host's project checkout to `cd` into
    /// before exec — the relative offset of the command's filesystem
    /// cwd from `project_root`. Mirrors the cwd the controller-side
    /// build used (e.g. `ddp-bench` for `fdl ddp-bench`). Empty string
    /// means execute from `host.path` directly. Relative-path defaults
    /// the binary expects (e.g. `--data-dir data`, `--output runs/`)
    /// only resolve correctly when the remote cwd matches.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cwd_subpath: String,
}

/// Build `cmd_name` for one host. Picks docker service + cargo
/// features from the host's libtorch metadata. Returns a
/// [`PerHostEnvelope`] describing where the resulting binary lives
/// (so the launcher can substitute it on the remote-dispatch path)
/// and what `LD_LIBRARY_PATH` the remote should set.
///
/// `controller_path` is the controller's view of the shared project
/// root. The libtorch convention says the variant lives at
/// `<controller_path>/libtorch/<host.arch>` for the build (controller
/// view) and `<host.path>/libtorch/<host.arch>` for the runtime
/// (remote view). Both point at the same physical libtorch via the
/// shared mount; the two paths differ only when controller and remote
/// see the project at different filesystem locations.
fn prebuild_one_host(
    project_root: &Path,
    cmd_cwd: &Path,
    controller_path: &Path,
    host: &ClusterHost,
    cmd_name: &str,
    controller_docker_svc: Option<&str>,
) -> Result<PerHostEnvelope, String> {
    let arch = host.arch.as_ref().ok_or_else(|| {
        format!(
            "host {:?} has no `arch:` set in cluster.yml — \
             pre-flight build needs the libtorch variant subpath \
             (e.g. `arch: precompiled/cu128` or `arch: builds/sm61-sm120`)",
            host.name,
        )
    })?;
    // Controller-side libtorch variant dir, resolved via convention.
    let controller_variant_dir = controller_path.join("libtorch").join(arch);
    if !controller_variant_dir.join("lib").is_dir() {
        return Err(format!(
            "host {:?}: controller-side libtorch at `{}` (resolved from \
             `<controller_path>/libtorch/<arch>`) does not look like a \
             valid libtorch install (missing `lib/`?)",
            host.name,
            controller_variant_dir.display(),
        ));
    }
    let host_path = controller_variant_dir.display().to_string();
    // Derive features + docker service from the YAML-declared `arch:`
    // basename — single source of truth, no `.arch` metadata file
    // required. `cpu` is the only non-CUDA variant by convention; every
    // other basename (`cuNN`, `sm<NN>-sm<NN>`, etc.) is a GPU build.
    let (features_arg, feature_docker_svc) = features_and_service_from_arch(arch);
    let cuda_version_for_image = cuda_version_from_arch(arch);
    let target_dir_relative = format!("target/cluster/{}", host.name);

    // Two execution modes — docker-backed (controller has `docker:`
    // set in cluster.yml) or native cargo on the host filesystem.
    //
    // Docker mode: the project root mounts at `/workspace`; cwd +
    // CARGO_TARGET_DIR are in the `/workspace/...` namespace. The
    // service to use is the controller's `docker:` value when
    // present (it owns the toolchain), falling back to the libtorch-
    // derived `cuda` / `dev` choice (matches the existing
    // `fdl cuda-build` / `fdl build` split).
    //
    // Native mode: cwd is the cmd's filesystem cwd, CARGO_TARGET_DIR
    // is the same project-root-relative path on the host, and
    // LIBTORCH_PATH is set directly on the cargo process (no Docker
    // bind-mount indirection).
    let (sh_cmd, cwd_for_spawn, extra_envs): (String, &Path, Vec<(&str, String)>) =
        if let Some(_svc) = controller_docker_svc {
            // Docker-backed build.
            let target_dir_in_container = format!("/workspace/{target_dir_relative}");
            let sub_path = cmd_cwd
                .strip_prefix(project_root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let cwd_in_container = if sub_path.is_empty() {
                "/workspace".to_string()
            } else {
                format!("/workspace/{sub_path}")
            };
            let build_cmd = if features_arg.is_empty() {
                format!(
                    "cd {cwd} && CARGO_TARGET_DIR={tgt} cargo build --release --bin {bin}",
                    cwd = posix_quote(&cwd_in_container),
                    tgt = posix_quote(&target_dir_in_container),
                    bin = posix_quote(cmd_name),
                )
            } else {
                format!(
                    "cd {cwd} && CARGO_TARGET_DIR={tgt} cargo build --release --features {feat} --bin {bin}",
                    cwd = posix_quote(&cwd_in_container),
                    tgt = posix_quote(&target_dir_in_container),
                    feat = posix_quote(features_arg),
                    bin = posix_quote(cmd_name),
                )
            };
            let svc = feature_docker_svc;
            let docker_cmd = format!(
                "docker compose run --rm {svc} bash -c {inner}",
                svc = svc,
                inner = posix_quote(&build_cmd),
            );
            (docker_cmd, project_root, vec![
                ("LIBTORCH_HOST_PATH", host_path.clone()),
                ("LIBTORCH_CPU_PATH", "./libtorch/precompiled/cpu".into()),
            ])
        } else {
            // Native build (no docker on controller).
            let target_dir_abs = project_root.join(&target_dir_relative);
            let bash_cmd = if features_arg.is_empty() {
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
            (bash_cmd, cmd_cwd, vec![
                ("LIBTORCH_PATH", host_path.clone()),
                (
                    "CARGO_TARGET_DIR",
                    target_dir_abs.to_string_lossy().into_owned(),
                ),
            ])
        };

    let mut cmd = Command::new("sh");
    cmd.args(["-c", &sh_cmd])
        .current_dir(cwd_for_spawn)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::null());
    for (k, v) in &extra_envs {
        cmd.env(k, v);
    }

    if let Some(cuda_version) = &cuda_version_for_image {
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

    let status = cmd
        .status()
        .map_err(|e| format!("spawn `{sh_cmd}`: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cargo build exited {} (libtorch={host_path}, target={target_dir_relative}, \
             features={feat})",
            status.code().unwrap_or(-1),
            feat = if features_arg.is_empty() { "(none)" } else { features_arg },
        ));
    }
    // Runtime LD_LIBRARY_PATH uses the REMOTE-side view: the rank
    // exec's the binary on the remote, where libtorch is at
    // `<host.path>/libtorch/<arch>/lib` per the convention.
    let runtime_lib = format!(
        "{path}/libtorch/{arch}/lib",
        path = host.path.trim_end_matches('/'),
    );
    let _ = host_path; // controller-side path used only for the build above
    // cwd_subpath: the cmd's filesystem cwd relative to project_root.
    // For `fdl ddp-bench` invoked from the repo, cmd_cwd is
    // `<repo>/ddp-bench`, so subpath is `ddp-bench`. The remote
    // launcher uses this to cd into the matching subdir before exec.
    let cwd_subpath = cmd_cwd
        .strip_prefix(project_root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(PerHostEnvelope {
        bin: format!("{target_dir_relative}/release/{cmd_name}"),
        ld_library_path: runtime_lib,
        cwd_subpath,
    })
}

/// Pick cargo features + docker compose service from the host's
/// libtorch `.arch` metadata. `cuda=12.x` → (`cuda`, `cuda`); anything
/// else → (`""`, `dev`).
/// Derive `(cargo --features arg, docker-compose service name)` from
/// the YAML `arch:` path basename. The yml `arch:` IS the single
/// source of truth (no `.arch` metadata file required) — `cpu` is the
/// only non-CUDA convention; everything else is a GPU variant.
fn features_and_service_from_arch(arch: &str) -> (&'static str, &'static str) {
    let basename = std::path::Path::new(arch)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if basename == "cpu" {
        ("", "dev")
    } else {
        ("cuda", "cuda")
    }
}

/// Extract a CUDA major.minor string from a `precompiled/cuNN` arch
/// path basename (e.g. `cu128` → `"12.8"`). Returns `None` for source
/// builds (`builds/sm…`) where the arch alone does not encode a CUDA
/// version — the caller falls back to the `CUDA_VERSION` env var (or
/// docker-compose's own default) for the toolkit image tag.
fn cuda_version_from_arch(arch: &str) -> Option<String> {
    let basename = std::path::Path::new(arch)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let rest = basename.strip_prefix("cu")?;
    if rest.len() < 2 || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let major = &rest[..rest.len() - 1];
    let minor = &rest[rest.len() - 1..];
    Some(format!("{major}.{minor}"))
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
    fn features_and_service_precompiled_cuda_picks_cuda() {
        assert_eq!(
            features_and_service_from_arch("precompiled/cu128"),
            ("cuda", "cuda")
        );
    }

    #[test]
    fn features_and_service_precompiled_cpu_picks_dev() {
        assert_eq!(
            features_and_service_from_arch("precompiled/cpu"),
            ("", "dev")
        );
    }

    #[test]
    fn features_and_service_source_build_picks_cuda() {
        // Source builds under `builds/<gpu-arch>` are CUDA by
        // convention; only `cpu` basename is non-CUDA.
        assert_eq!(
            features_and_service_from_arch("builds/sm61-sm120"),
            ("cuda", "cuda")
        );
        assert_eq!(
            features_and_service_from_arch("builds/sm80"),
            ("cuda", "cuda")
        );
    }

    #[test]
    fn cuda_version_from_arch_extracts_precompiled_version() {
        assert_eq!(cuda_version_from_arch("precompiled/cu128"), Some("12.8".into()));
        assert_eq!(cuda_version_from_arch("precompiled/cu126"), Some("12.6".into()));
        assert_eq!(cuda_version_from_arch("precompiled/cu118"), Some("11.8".into()));
    }

    #[test]
    fn cuda_version_from_arch_none_for_source_builds_and_cpu() {
        assert_eq!(cuda_version_from_arch("builds/sm61-sm120"), None);
        assert_eq!(cuda_version_from_arch("builds/sm80"), None);
        assert_eq!(cuda_version_from_arch("precompiled/cpu"), None);
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
                cwd_subpath: String::new(),
            },
        );
        env.insert(
            "host-a".to_string(),
            PerHostEnvelope {
                bin: "target/cluster/host-a/release/bench".into(),
                ld_library_path: "/opt/lt-a/lib".into(),
                cwd_subpath: String::new(),
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
                cwd_subpath: "ddp-bench".into(),
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
