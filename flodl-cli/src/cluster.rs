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
///
/// On success returns a `Vec<String>` of non-fatal resolution warnings
/// (one entry per host whose NSS lookup failed or yielded only loopback
/// addresses). The cluster-dispatch site in `main.rs` is the one that
/// chooses to print them. Tests that exercise this function for its
/// env-setting behavior simply ignore the returned Vec.
pub fn prepare_cluster_env(
    cluster: &ClusterConfig,
    overlay_env: Option<&str>,
    cmd: &str,
) -> Result<Vec<String>, String> {
    cluster.validate()?;
    let mut warnings: Vec<String> = Vec::new();
    // Pre-resolve `controller.host` on the controller (where NSS knows
    // names declared in `/etc/hosts`, `libnss-libvirt`, mDNS, etc.)
    // and ship the resolved IP in the envelope to remote ranks. Remote
    // VMs that don't share the controller's NSS view (a Pascal VM on
    // libvirt's virbr0 has no plugin to resolve "exa") then connect
    // by numeric IP without needing their own resolver to know cluster
    // hostnames. If resolution fails on the controller, ship the
    // original string and let the remote try its own NSS as a last
    // resort.
    let mut shippable = cluster.clone();
    let (controller_ip, controller_warning) =
        resolve_host_to_ip(&shippable.controller.host);
    if let Some(ip) = controller_ip {
        shippable.controller.host = ip;
    }
    if let Some(w) = controller_warning {
        warnings.push(w);
    }
    let json = shippable.canonical_json()?;
    let hex = hex_encode(json.as_bytes());
    let (extra_hosts, host_warnings) = resolve_cluster_extra_hosts(cluster);
    warnings.extend(host_warnings);

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
    Ok(warnings)
}

/// Resolve each cluster worker's `host` to an IP via the controller's
/// NSS (which on Linux includes static `/etc/hosts`, `libnss-libvirt`,
/// `libnss-mdns`, and DNS — anything `getaddrinfo` knows about).
/// Returns `(Vec<"host:ip">, Vec<warning>)`: the first list is suitable
/// for `--add-host` injection into `docker compose run`; the second is
/// human-readable warnings the cluster-dispatch site can surface to the
/// user. Workers that fail to resolve are skipped from the `host:ip`
/// list (better-than-nothing semantics for the launcher inside the
/// container — the unresolved host will retry via its own NSS).
fn resolve_cluster_extra_hosts(cluster: &ClusterConfig) -> (Vec<String>, Vec<String>) {
    let mut hosts = Vec::new();
    let mut warnings = Vec::new();
    for w in &cluster.workers {
        let (ip, warning) = resolve_host_to_ip(&w.host);
        if let Some(ip) = ip {
            hosts.push(format!("{}:{ip}", w.host));
        }
        if let Some(w) = warning {
            warnings.push(w);
        }
    }
    (hosts, warnings)
}

/// Resolve a hostname to an IP string via `getaddrinfo`. Returns
/// `(Option<ip>, Option<warning>)` — both are independently optional so
/// the caller can ship the resolved IP AND surface the warning, or
/// either alone. The function itself is silent; warnings are returned
/// for the cluster-dispatch site to emit (or ignore in non-dispatch
/// contexts like unit tests that exercise the env-setting paths).
///
/// Prefers a non-loopback address when `getaddrinfo` returns several
/// candidates. Debian/Ubuntu install `/etc/hosts` with a
/// `127.0.1.1 <hostname>` line by default, which `getaddrinfo` returns
/// FIRST — that IP works for the local host but is unreachable from
/// any peer (a libvirt VM, another rig). Skipping loopback in the
/// iterator picks the bridge / LAN address remote ranks can actually
/// dial. If ONLY loopback resolves, return it WITH a warning string
/// (likely misconfig — better to surface than to silently ship an
/// unreachable IP).
fn resolve_host_to_ip(host: &str) -> (Option<String>, Option<String>) {
    use std::net::ToSocketAddrs;
    // Already a numeric address? Skip the lookup, return as-is.
    if host.parse::<std::net::IpAddr>().is_ok() {
        return (Some(host.to_string()), None);
    }
    match (host, 0u16).to_socket_addrs() {
        Ok(iter) => {
            let (ip, only_loopback) = select_preferred_ip(iter.map(|sa| sa.ip()));
            let warning = match (&ip, only_loopback) {
                (Some(ip), true) => Some(format!(
                    "host {host:?} only resolves to loopback {ip} on the controller \
                     — remote ranks will fail to connect. Set `master_addr` (or the \
                     host's `name`) in fdl.cluster.yml to a non-loopback IP reachable \
                     from peer nodes (e.g. the libvirt bridge IP 192.168.122.1 for \
                     virbr0)."
                )),
                _ => None,
            };
            (ip, warning)
        }
        Err(e) => (
            None,
            Some(format!(
                "host {host:?} did not resolve on controller: {e} (remote ranks \
                 will retry via their own NSS — fix host-side resolution if they \
                 also fail)"
            )),
        ),
    }
}

/// Pick the best IP from a `getaddrinfo` iterator: first non-loopback
/// wins; if every candidate is loopback, return the first loopback
/// with `only_loopback=true` so the caller can warn. Pure function,
/// no NSS dependency — exists so the selection rule is unit-testable.
fn select_preferred_ip<I: IntoIterator<Item = std::net::IpAddr>>(
    iter: I,
) -> (Option<String>, bool) {
    let mut loopback_fallback: Option<String> = None;
    for ip in iter {
        if !ip.is_loopback() {
            return (Some(ip.to_string()), false);
        }
        loopback_fallback.get_or_insert_with(|| ip.to_string());
    }
    let only_loopback = loopback_fallback.is_some();
    (loopback_fallback, only_loopback)
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
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
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
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
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
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
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
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
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
    fn select_preferred_ip_prefers_non_loopback() {
        use std::net::IpAddr;
        // The Debian/Ubuntu /etc/hosts shape we have to handle:
        // 127.0.1.1 comes back FIRST, the routable LAN/bridge IP second.
        let ips: Vec<IpAddr> = vec![
            "127.0.1.1".parse().unwrap(),
            "192.168.122.1".parse().unwrap(),
        ];
        let (ip, only_loopback) = select_preferred_ip(ips);
        assert_eq!(ip.as_deref(), Some("192.168.122.1"));
        assert!(!only_loopback);
    }

    #[test]
    fn select_preferred_ip_falls_back_to_loopback_with_flag() {
        use std::net::IpAddr;
        // Misconfig case: only loopback resolves. Return it so the
        // caller still has SOMETHING, but flip the flag so the caller
        // warns.
        let ips: Vec<IpAddr> = vec!["127.0.1.1".parse().unwrap(), "::1".parse().unwrap()];
        let (ip, only_loopback) = select_preferred_ip(ips);
        assert_eq!(ip.as_deref(), Some("127.0.1.1"));
        assert!(only_loopback);
    }

    #[test]
    fn select_preferred_ip_empty_iterator() {
        let (ip, only_loopback) = select_preferred_ip(std::iter::empty());
        assert!(ip.is_none());
        assert!(!only_loopback);
    }

    #[test]
    fn select_preferred_ip_skips_ipv6_loopback() {
        use std::net::IpAddr;
        // IPv6 loopback (::1) must be skipped just like 127.x.
        let ips: Vec<IpAddr> = vec!["::1".parse().unwrap(), "10.0.0.5".parse().unwrap()];
        let (ip, only_loopback) = select_preferred_ip(ips);
        assert_eq!(ip.as_deref(), Some("10.0.0.5"));
        assert!(!only_loopback);
    }

    #[test]
    fn prepare_cluster_env_validates_cluster() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // Empty controller.host → validate() fails → prepare_cluster_env errors.
        let cluster = ClusterConfig {
            controller: crate::config::ClusterController {
                host: String::new(),
                port: 1337,
                path: String::new(),
                nccl_socket_ifname: None,
                docker: None,
                arch: None,
                data_path: None,
            },
            workers: Vec::new(),
            env: std::collections::BTreeMap::new(),
        };
        let err = prepare_cluster_env(&cluster, None, "train").unwrap_err();
        assert!(err.contains("controller.host"), "got: {err}");
    }
}
