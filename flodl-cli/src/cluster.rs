//! Cluster-mode env preparation.
//!
//! Process-per-rank model: flodl owns fan-out and controller
//! orchestration (`flodl::distributed::launcher::run_launcher`). fdl-cli's
//! job here is purely to ship the parsed cluster topology to the launcher
//! via env vars, then let the normal `RunScript` / `ExecCommand` dispatch
//! invoke the user binary. The user binary's
//! `flodl::distributed::launcher::dispatch` reads the env, detects
//! launcher role, and fans out (ssh for remote hosts, fork+exec for local
//! hosts). All log fan-in + ClusterController + exit-code propagation happen on
//! the flodl side.
//!
//! ```text
//! fdl @cluster train
//!   ↓ fdl-cli parses fdl.yml + fdl.cluster.yml overlay
//!   ↓ fdl-cli calls prepare_cluster_env: sets FLODL_FULL_CLUSTER_JSON,
//!     FLODL_FDL_CMD, FDL_ENV on its own process env
//!   ↓ fdl-cli falls through to normal RunScript / ExecCommand path
//!   ↓ resolved command (e.g. `cargo run --release --bin my-trainer`) runs
//!   ↓ my-trainer inherits env, flodl::launcher::dispatch detects Launcher
//!   ↓ launcher fans out: ssh per remote host, fork+exec per local rank
//!   ↓ each rank child has FLODL_CLUSTER_JSON + FLODL_LOCAL_RANK set
//!   ↓ rank-side flodl::launcher::dispatch returns Role::Rank, training runs
//! ```
//!
//! Recursion guard: the launcher's ssh fan-out invokes `fdl <cmd>` on the
//! remote, which re-enters fdl-cli with `FLODL_CLUSTER_JSON` set (not
//! `FLODL_FULL_CLUSTER_JSON`). [`should_dispatch`](crate::cluster::should_dispatch)
//! returns `false` in that case so the remote fdl-cli skips cluster setup
//! and just runs the user binary normally — the user binary's launcher
//! dispatch then detects `Role::Rank` (because `FLODL_LOCAL_RANK` is also
//! set).

use std::path::Path;
use std::process::Command;

use crate::config::{self, ClusterConfig, ProjectConfig};

/// Env var name carrying the *full* multi-host topology (hex-encoded
/// JSON of [`ClusterConfig`]). Set by fdl-cli on its own process env so
/// the spawned user binary inherits it and detects launcher role.
/// Mirrors `flodl::distributed::launcher::ENV_FULL_CLUSTER_JSON`.
pub const ENV_FULL_CLUSTER_JSON: &str = "FLODL_FULL_CLUSTER_JSON";

/// Env var name carrying the original fdl command name (e.g. `train`).
/// Read by the launcher when it needs to invoke `fdl <cmd>` over ssh
/// on remote hosts. Mirrors `flodl::distributed::launcher::ENV_FDL_CMD`.
pub const ENV_FDL_CMD: &str = "FLODL_FDL_CMD";

/// Env var name picking the overlay env name (e.g. `cluster`). Set by
/// fdl-cli at first-arg parsing time; propagated through to remote
/// hosts by the launcher so they see the same overlay-merged view.
pub const ENV_FDL_ENV: &str = "FDL_ENV";

/// Env var name carrying the slim per-rank envelope. Set by the
/// launcher (not fdl-cli) on each rank child. Kept here so the
/// recursion guard can reference it by name. Mirrors
/// `flodl::distributed::cluster::ENV_CLUSTER_JSON`.
pub const ENV_CLUSTER_JSON: &str = "FLODL_CLUSTER_JSON";

/// Pre-resolved `name:ip` pairs (space-separated) for every cluster
/// host, written by [`prepare_cluster_env`] using the controller's NSS
/// resolution. Consumed by run/prebuild/schema-cache when they build
/// `docker compose run --rm` commands: each pair is injected as a
/// `--add-host name:ip` flag so the containerized launcher can SSH
/// into cluster hosts without depending on the container's own
/// resolver (which lacks `libnss-libvirt` etc.).
pub const ENV_CLUSTER_EXTRA_HOSTS: &str = "FLODL_CLUSTER_EXTRA_HOSTS";

/// Controller's OS user name (resolved on the host by fdl-cli, before
/// any docker spawn). The launcher in the container reads it as the
/// default `ssh -l` target when the per-host `ssh_user:` is unset.
/// Bridges the container-vs-host user mismatch (containers ship a
/// stock `ubuntu` UID-1000 user, but `ubuntu@<remote>` is rarely the
/// account the user actually uses on cluster hosts).
pub const ENV_HOST_USER: &str = "FLODL_HOST_USER";

/// Env var name overriding the OS hostname for cluster lookups.
/// Mirrors `flodl::distributed::cluster::ENV_HOST_OVERRIDE`.
pub const ENV_HOST_OVERRIDE: &str = "FLODL_HOST_NAME";

/// Env var name picking this rank's local-rank index within its host.
/// Set by the launcher on rank children. Mirrors
/// `flodl::distributed::cluster::ENV_LOCAL_RANK`.
pub const ENV_LOCAL_RANK: &str = "FLODL_LOCAL_RANK";

/// Top-level cluster-dispatch decision.
///
/// Returns `false` when `FLODL_CLUSTER_JSON` is set — that signals we're
/// a recursive fdl invocation on a remote host that the launcher's ssh
/// fan-out reached, and we should fall through to normal dispatch.
/// Otherwise delegates to [`config::cluster_dispatch_enabled`].
pub fn should_dispatch(project: &ProjectConfig, chain: &[Option<bool>]) -> bool {
    if is_recursive_invocation() {
        return false;
    }
    config::cluster_dispatch_enabled(project, chain)
}

/// Whether this fdl invocation is itself a spawned child of a launcher's
/// ssh fan-out (`FLODL_CLUSTER_JSON` already set in env). Used as the
/// recursion guard everywhere cluster dispatch is evaluated.
pub fn is_recursive_invocation() -> bool {
    std::env::var_os(ENV_CLUSTER_JSON).is_some()
}

/// Prepare the env vars needed for the user binary's flodl launcher to
/// detect launcher role and fan out. Caller continues to normal
/// dispatch (`RunScript` / `ExecCommand`); the spawned subprocess
/// inherits these env vars and the launcher takes over.
///
/// `overlay_env` is the overlay name from `fdl @<env>` (e.g.
/// `Some("cluster")`); propagated to remote hosts via the launcher so
/// they see the same overlay-merged `commands:` resolution.
///
/// Returns `Err` if the cluster config is invalid or JSON serialization
/// fails — surfaces the error before the user binary even starts.
pub fn prepare_cluster_env(
    cluster: &ClusterConfig,
    overlay_env: Option<&str>,
    cmd: &str,
) -> Result<(), String> {
    cluster.validate()?;
    // Pre-resolve `master_addr` on the controller (where NSS knows
    // names declared in `/etc/hosts`, `libnss-libvirt`, mDNS, etc.)
    // and ship the resolved IP in the envelope to remote ranks. Remote
    // VMs that don't share the controller's NSS view (a Pascal VM on
    // libvirt's virbr0 has no plugin to resolve "exa") then connect
    // by numeric IP without needing their own resolver to know cluster
    // hostnames. If resolution fails on the controller, ship the
    // original string and let the remote try its own NSS as a last
    // resort.
    let mut shippable = cluster.clone();
    if let Some(ip) = resolve_host_to_ip(&shippable.master_addr) {
        shippable.master_addr = ip;
    }
    let json = shippable.canonical_json()?;
    let hex = hex_encode(json.as_bytes());
    let extra_hosts = resolve_cluster_extra_hosts(cluster);

    // SAFETY: main() has not spawned threads at this point in the
    // dispatch flow (mirrors gpus::apply_cuda_visible_devices's
    // invariant; documented in main.rs).
    unsafe {
        std::env::set_var(ENV_FULL_CLUSTER_JSON, &hex);
        std::env::set_var(ENV_FDL_CMD, cmd);
        std::env::set_var(ENV_HOST_USER, resolve_local_user());
        if !extra_hosts.is_empty() {
            std::env::set_var(ENV_CLUSTER_EXTRA_HOSTS, extra_hosts.join(" "));
        }
        if let Some(e) = overlay_env {
            if !e.trim().is_empty() {
                std::env::set_var(ENV_FDL_ENV, e);
            }
        }
    }
    Ok(())
}

/// Resolve each cluster host's `name` to an IP via the controller's
/// NSS (which on Linux includes static `/etc/hosts`, `libnss-libvirt`,
/// `libnss-mdns`, and DNS — anything `getaddrinfo` knows about).
/// Returns `Vec<"name:ip">` strings suitable for `--add-host`
/// injection into `docker compose run`.
///
/// Hosts that fail to resolve are skipped with a stderr warning; the
/// caller still gets the partial list (better-than-nothing semantics
/// for the launcher inside the container).
fn resolve_cluster_extra_hosts(cluster: &ClusterConfig) -> Vec<String> {
    cluster
        .hosts
        .iter()
        .filter_map(|h| resolve_host_to_ip(&h.name).map(|ip| format!("{}:{ip}", h.name)))
        .collect()
}

/// Resolve a hostname to an IP string via `getaddrinfo`. Returns
/// `None` (with a stderr warning) when resolution fails on the
/// controller — caller decides whether to fall back to shipping the
/// original string or hard-fail.
fn resolve_host_to_ip(host: &str) -> Option<String> {
    use std::net::ToSocketAddrs;
    // Already a numeric address? Skip the lookup, return as-is.
    if host.parse::<std::net::IpAddr>().is_ok() {
        return Some(host.to_string());
    }
    match (host, 0u16).to_socket_addrs() {
        Ok(mut iter) => iter.next().map(|sa| sa.ip().to_string()),
        Err(e) => {
            eprintln!(
                "fdl: warning: host {host:?} did not resolve on controller: {e} \
                 (remote ranks will retry via their own NSS — fix host-side \
                 resolution if they also fail)"
            );
            None
        }
    }
}

/// Write a temporary docker-compose overlay (under `project_root`)
/// that populates `extra_hosts:` for the cluster-capable services
/// (`cuda`, `dev`, `bench`) from the controller-resolved cluster
/// hosts in [`ENV_CLUSTER_EXTRA_HOSTS`], then return the `-f` flag
/// sequence to splice in front of `docker compose run`.
///
/// `docker compose run` itself does not accept `--add-host` (that's
/// `docker run` only), but compose merges multiple `-f` files, so the
/// overlay extends the base config without mutating it.
///
/// Returns the empty string (and writes nothing) when not in cluster
/// mode — non-cluster runs keep their existing `docker compose run`
/// invocation unchanged.
pub fn cluster_compose_overlay_arg(project_root: &Path) -> String {
    let raw = match std::env::var(ENV_CLUSTER_EXTRA_HOSTS) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let pairs: Vec<&str> = raw.split_whitespace().filter(|p| !p.is_empty()).collect();
    if pairs.is_empty() {
        return String::new();
    }

    let mut entries = String::new();
    for pair in &pairs {
        entries.push_str("      - \"");
        entries.push_str(pair);
        entries.push_str("\"\n");
    }

    // extra_hosts is per-service in compose; apply to every
    // cluster-capable service so the same override file works
    // regardless of which one the dispatch lands on.
    let overlay = format!(
        "# Generated by fdl-cli (cluster mode) — DO NOT EDIT BY HAND.\n\
         # Regenerated on every `fdl @cluster ...` invocation.\n\
         services:\n\
         \x20\x20cuda:\n\
         \x20\x20\x20\x20extra_hosts:\n{entries}\
         \x20\x20dev:\n\
         \x20\x20\x20\x20extra_hosts:\n{entries}\
         \x20\x20bench:\n\
         \x20\x20\x20\x20extra_hosts:\n{entries}",
    );

    let overlay_path = project_root.join(".fdl-cluster-overlay.yml");
    if let Err(e) = std::fs::write(&overlay_path, overlay) {
        eprintln!(
            "fdl: warning: failed to write cluster compose overlay at {:?}: {e} \
             (continuing without --add-host injection — remote hostnames \
             may not resolve inside the container)",
            overlay_path
        );
        return String::new();
    }

    // base docker-compose.yml first, then our overlay second, so the
    // overlay's extra_hosts merges into the base service definitions.
    format!(" -f docker-compose.yml -f {}", overlay_path.display())
}

/// Hex-encode raw bytes (lowercase, no separators). Companion to the
/// library's `hex_decode` in `flodl::distributed::cluster`. Kept here
/// so `prepare_cluster_env` doesn't pull in a flodl runtime dep.
pub fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(TABLE[(b >> 4) as usize] as char);
        s.push(TABLE[(b & 0x0F) as usize] as char);
    }
    s
}

/// Resolve the controller's OS user name. Used to pre-populate
/// [`ENV_HOST_USER`] before docker spawn, so the launcher inside the
/// container can default `ssh -l <user>` to the host's identity.
/// Falls through `USER` then `whoami` then `"unknown-user"`.
pub fn resolve_local_user() -> String {
    if let Ok(s) = std::env::var("USER") {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    Command::new("whoami")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown-user".to_string())
}

/// Resolve the local OS hostname. Used by `gpus::synthesize_local_cluster`
/// (the `--gpus` single-host shorthand) and by `prebuild` to skip the
/// controller from the remote-host fan-out. Test/override seam via
/// [`ENV_HOST_OVERRIDE`]; falls back to the `hostname(1)` command.
pub fn resolve_local_hostname() -> String {
    if let Ok(s) = std::env::var(ENV_HOST_OVERRIDE) {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown-host".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Env-mutating tests serialize via this mutex per
    /// `feedback_env_mutating_tests_mutex`. `should_dispatch` reads
    /// `FLODL_CLUSTER_JSON` and `prepare_cluster_env` sets several
    /// vars; both classes are guarded by the same lock.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn should_dispatch_returns_false_when_cluster_json_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized via ENV_MUTEX above.
        unsafe {
            std::env::set_var(ENV_CLUSTER_JSON, "deadbeef");
        }
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  x: { cluster: true, run: \"echo hi\" }
";
        let project: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            !should_dispatch(&project, &[Some(true)]),
            "recursion guard: must return false when FLODL_CLUSTER_JSON is set"
        );
        unsafe {
            std::env::remove_var(ENV_CLUSTER_JSON);
        }
    }

    #[test]
    fn should_dispatch_delegates_when_env_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::remove_var(ENV_CLUSTER_JSON);
        }
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  x: { run: \"echo hi\" }
";
        let project: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!should_dispatch(&project, &[None]));
        assert!(should_dispatch(&project, &[Some(true)]));
    }

    #[test]
    fn hex_encode_matches_library() {
        // Well-known mappings; library's flodl::distributed::cluster::hex_decode
        // is the round-trip partner.
        assert_eq!(hex_encode(b""), "");
        assert_eq!(hex_encode(&[0x00]), "00");
        assert_eq!(hex_encode(&[0xff]), "ff");
        assert_eq!(hex_encode(&[0x0f, 0xa0]), "0fa0");
        assert_eq!(hex_encode(b"hi"), "6869");
    }

    #[test]
    fn prepare_cluster_env_sets_required_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Clear env first so we observe what prepare_cluster_env sets.
        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_FDL_CMD);
            std::env::remove_var(ENV_FDL_ENV);
        }
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  train: { cluster: true, run: \"true\" }
";
        let project: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        let cluster = project.cluster.as_ref().unwrap();
        prepare_cluster_env(cluster, Some("cluster"), "train").expect("prepare OK");

        assert!(!std::env::var(ENV_FULL_CLUSTER_JSON).unwrap().is_empty());
        assert_eq!(std::env::var(ENV_FDL_CMD).unwrap(), "train");
        assert_eq!(std::env::var(ENV_FDL_ENV).unwrap(), "cluster");

        // Verify the full envelope round-trips back to the canonical JSON.
        let hex = std::env::var(ENV_FULL_CLUSTER_JSON).unwrap();
        // Decode and parse it as JSON.
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));

        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_FDL_CMD);
            std::env::remove_var(ENV_FDL_ENV);
        }
    }

    #[test]
    fn prepare_cluster_env_skips_fdl_env_when_blank() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::remove_var(ENV_FDL_ENV);
        }
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  train: { cluster: true, run: \"true\" }
";
        let project: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        let cluster = project.cluster.as_ref().unwrap();
        // None overlay → no FDL_ENV var set.
        prepare_cluster_env(cluster, None, "train").unwrap();
        assert!(std::env::var_os(ENV_FDL_ENV).is_none());

        // Empty overlay → also no FDL_ENV var.
        prepare_cluster_env(cluster, Some("   "), "train").unwrap();
        assert!(std::env::var_os(ENV_FDL_ENV).is_none());

        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_FDL_CMD);
        }
    }

    #[test]
    fn prepare_cluster_env_validates_cluster() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Empty master_addr → validate() fails → prepare_cluster_env errors.
        let cluster = ClusterConfig {
            master_port: 29500,
            ..Default::default()
        };
        let err = prepare_cluster_env(&cluster, None, "train").unwrap_err();
        assert!(err.contains("master_addr"), "got: {err}");
    }
}
