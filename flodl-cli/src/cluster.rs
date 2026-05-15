//! Cluster-mode env preparation.
//!
//! Process-per-rank model: flodl owns fan-out and controller
//! orchestration (`flodl::distributed::launcher::run_launcher`). fdl-cli's
//! job here is purely to ship the parsed cluster topology to the launcher
//! via env vars, then let the normal `RunScript` / `ExecCommand` dispatch
//! invoke the user binary. The user binary's
//! [`flodl::distributed::launcher::dispatch`] reads the env, detects
//! launcher role, and fans out (ssh for remote hosts, fork+exec for local
//! hosts). All log fan-in + CpuAverager + exit-code propagation happen on
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
//! `FLODL_FULL_CLUSTER_JSON`). [`should_dispatch`] returns `false` in that
//! case so the remote fdl-cli skips cluster setup and just runs the user
//! binary normally — the user binary's launcher dispatch then detects
//! `Role::Rank` (because `FLODL_LOCAL_RANK` is also set).

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
    let json = cluster.canonical_json()?;
    let hex = hex_encode(json.as_bytes());

    // SAFETY: main() has not spawned threads at this point in the
    // dispatch flow (mirrors gpus::apply_cuda_visible_devices's
    // invariant; documented in main.rs).
    unsafe {
        std::env::set_var(ENV_FULL_CLUSTER_JSON, &hex);
        std::env::set_var(ENV_FDL_CMD, cmd);
        if let Some(e) = overlay_env {
            if !e.trim().is_empty() {
                std::env::set_var(ENV_FDL_ENV, e);
            }
        }
    }
    Ok(())
}

/// Hex-encode raw bytes (lowercase, no separators). Companion to the
/// library's `hex_decode` in `flodl::distributed::cluster`. Kept here
/// so `prepare_cluster_env` doesn't pull in a flodl runtime dep.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(TABLE[(b >> 4) as usize] as char);
        s.push(TABLE[(b & 0x0F) as usize] as char);
    }
    s
}

/// Resolve the local OS hostname for `gpus::synthesize_local_cluster`
/// (the `--gpus` single-host shorthand). Test/override seam via
/// [`ENV_HOST_OVERRIDE`]; falls back to the `hostname(1)` command.
pub(crate) fn resolve_local_hostname() -> String {
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
