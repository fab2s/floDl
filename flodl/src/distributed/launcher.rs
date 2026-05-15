//! Cluster launcher: role detection + fan-out + controller orchestration.
//!
//! Slots transparently into [`Trainer::setup`] and friends on cluster-mode
//! startup. Each user-binary invocation routes to one of three roles:
//!
//! - **Launcher**: the parent process that fdl-cli execs after parsing
//!   `fdl.yml`. Reads the full cluster topology, spawns one child per rank
//!   (fork/exec for local hosts, ssh for remote), starts the controller
//!   thread (TCP byte router for CPU averaging + log fan-in), waits for
//!   every child to exit, then exits itself.
//!
//! - **Rank**: a spawned child running the user's training code. Inherits
//!   the slim per-host envelope and the rank-slot env var injected by the
//!   launcher; existing [`Trainer::setup`] cluster-path logic handles the
//!   rest (rendezvous, `Ddp::wrap`, training loop).
//!
//! - **Single-device**: no cluster envelope in env. Caller continues with
//!   today's single-device path. Bit-identical to pre-cluster behavior.
//!
//! # Wire protocol (env vars)
//!
//! Two env vars distinguish the launcher and rank roles. The names are
//! deliberately namespaced so a fdl-cli invocation in a cluster context
//! never sets them both at the same time:
//!
//! - [`ENV_FULL_CLUSTER_JSON`] (`FLODL_FULL_CLUSTER_JSON`): hex-encoded
//!   JSON of the *full* cluster topology (all hosts + ranks + devices).
//!   Set by fdl-cli when invoking the user binary as the launcher. The
//!   launcher reads it once to drive fan-out; never propagated to rank
//!   children.
//!
//! - [`crate::distributed::cluster::ENV_CLUSTER_JSON`]
//!   (`FLODL_CLUSTER_JSON`): hex-encoded slim per-host envelope, mirroring
//!   the existing rank-side wire format. Set by the launcher (not fdl-cli)
//!   when spawning each rank child. Read by [`LocalCluster::from_env`].
//!
//! - [`crate::distributed::cluster::ENV_LOCAL_RANK`] (`FLODL_LOCAL_RANK`):
//!   integer index into the slim envelope's `host.ranks`. Set by the
//!   launcher when spawning each rank child. Read by
//!   [`crate::distributed::cluster::LocalCluster::my_rank`].
//!
//! Role detection table:
//!
//! | `FLODL_FULL_CLUSTER_JSON` | `FLODL_CLUSTER_JSON` | `FLODL_LOCAL_RANK` | Role |
//! |---|---|---|---|
//! | unset | unset | unset | [`Role::SingleDevice`] |
//! | unset | set | set | [`Role::Rank`] |
//! | set | unset | unset | [`Role::Launcher`] (dispatch fans out) |
//! | other combinations | | | loud error |
//!
//! # Design notes
//!
//! The "two-env-var" wire protocol is the smallest additive change to
//! today's setup. Slim per-rank envelopes stay the same shape on the
//! rank side, so [`LocalCluster::from_env`] needs no change. The new
//! launcher-side parser ([`FullCluster::from_env`]) consumes the full
//! topology in a separate path. A future cleanup could unify both
//! shapes; for 4b the additive form keeps the blast radius small.
//!
//! [`Trainer::setup`]: crate::distributed::Trainer::setup
//! [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env

use std::env;

use crate::tensor::{Result, TensorError};

/// Environment variable carrying the *full* cluster topology to the
/// launcher process. Set by fdl-cli; consumed only by [`dispatch`]. Not
/// propagated to rank children (each child gets a slim per-host envelope
/// instead via `FLODL_CLUSTER_JSON`).
pub const ENV_FULL_CLUSTER_JSON: &str = "FLODL_FULL_CLUSTER_JSON";

/// Role this process plays in the cluster, decided by [`dispatch`].
///
/// Returned to the caller so it can either continue with training
/// (`Rank`/`SingleDevice`) or unwind cleanly (`LauncherDone` — the
/// launcher has already finished fan-out + controller wait).
#[derive(Debug, PartialEq, Eq)]
pub enum Role {
    /// No cluster envelope in env. Continue with today's single-device
    /// training path.
    SingleDevice,
    /// This process is a rank. Continue with cluster-mode training
    /// (`Trainer::setup` will read the slim envelope and rendezvous).
    Rank,
    /// This process was the launcher. Fan-out completed, all ranks
    /// finished, controller shut down cleanly. Caller should propagate
    /// and exit the program (process is done).
    LauncherDone,
}

/// Detect role from env and run launcher orchestration if applicable.
///
/// Called by [`Trainer::setup`] and the other entry points at the very
/// top of the cluster-init path. Three outcomes:
///
/// - [`Role::SingleDevice`] — env has no cluster markers. Caller proceeds.
/// - [`Role::Rank`] — env identifies this process as a rank. Caller
///   proceeds (existing cluster-path code handles rendezvous, etc.).
/// - [`Role::LauncherDone`] — env identified this process as the
///   launcher; this call ran fan-out + waited for ranks. Caller should
///   propagate up and exit (`std::process::exit(0)` or equivalent).
///
/// Loud error on inconsistent env (e.g. both full-cluster and rank-slot
/// vars set — silently winning one over the other costs hours of
/// debugging on a misconfigured rig).
///
/// [`Trainer::setup`]: crate::distributed::Trainer::setup
pub fn dispatch() -> Result<Role> {
    let full_set = env::var_os(ENV_FULL_CLUSTER_JSON).is_some();
    let slim_set = env::var_os(crate::distributed::cluster::ENV_CLUSTER_JSON).is_some();
    let slot_set = env::var_os(crate::distributed::cluster::ENV_LOCAL_RANK).is_some();

    match (full_set, slim_set, slot_set) {
        (false, false, false) => Ok(Role::SingleDevice),
        (false, true, true) => Ok(Role::Rank),
        (true, false, false) => {
            run_launcher()?;
            Ok(Role::LauncherDone)
        }
        // Any other combination is a misconfiguration. Loud error with
        // every bit named so the operator can see what's off.
        _ => Err(TensorError::new(&format!(
            "cluster launcher: inconsistent env (FLODL_FULL_CLUSTER_JSON={}, \
             FLODL_CLUSTER_JSON={}, FLODL_LOCAL_RANK={}). \
             Expected: all-unset (single-device), slim+slot only (rank), \
             or full only (launcher).",
            on_off(full_set),
            on_off(slim_set),
            on_off(slot_set),
        ))),
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "set" } else { "unset" }
}

/// Launcher-mode orchestration. Read full topology, spawn ranks, start
/// controller, wait, shut down. Returns when every rank child has exited.
///
/// **Status (4b.B.1):** scaffolding. Reads + parses the full topology
/// envelope. Fan-out (ssh / fork-exec), controller (TCP byte router), and
/// log fan-in land in subsequent slices (4b.B.2 — `controller.rs`,
/// 4b.B.3 — `cpu_reduce.rs`, 4b.B.4 — Trainer wire-in). Until those
/// land, this function returns an error indicating the unimplemented path.
fn run_launcher() -> Result<()> {
    let _full = FullCluster::from_env()?;
    Err(TensorError::new(
        "cluster launcher: fan-out + controller not yet implemented \
         (4b.B.2/3/4). Full topology parsed; fdl @cluster should still go \
         through the legacy flodl-cli/src/cluster.rs path during the 4b \
         transition.",
    ))
}

// ---------------------------------------------------------------------------
// FullCluster: launcher-side parser for the multi-host topology.
// ---------------------------------------------------------------------------

/// Full cluster topology as seen by the launcher process.
///
/// Mirrors flodl-cli's `ClusterConfig` shape; lives on the flodl side so
/// the framework owns cluster orchestration end-to-end. The slim
/// per-rank envelopes parsed by [`LocalCluster`] are derived from this
/// view at fan-out time.
///
/// Like [`LocalCluster`], the rank-side `local_devices: "all"` shorthand
/// is resolved at parse time when applicable (host-side resolution
/// happens later, after envelope ship — see [`crate::distributed::cluster`]
/// for the slim path).
///
/// [`LocalCluster`]: crate::distributed::cluster::LocalCluster
#[derive(Debug, Clone)]
pub struct FullCluster {
    pub master_addr: String,
    pub master_port: u16,
    pub hosts: Vec<FullHost>,
}

/// One host's entry in the full topology, launcher-side.
///
/// Differs from [`HostBlock`] by carrying `ssh:` (launcher-only field
/// stripped from slim envelopes) and the unresolved `local_devices:
/// "all"` form (which is only resolved on the host that will use it).
///
/// [`HostBlock`]: crate::distributed::cluster::HostBlock
#[derive(Debug, Clone)]
pub struct FullHost {
    pub name: String,
    pub ranks: Vec<usize>,
    /// Either an explicit list of CUDA indices or `None` for the `"all"`
    /// shorthand (resolved at startup on the host that owns this entry).
    pub local_devices: Option<Vec<u8>>,
    pub nccl_socket_ifname: String,
    pub path: String,
    pub libtorch_path: Option<String>,
    /// SSH target for remote dispatch. `None` means the host runs on
    /// the same machine as the launcher (fork/exec path, no ssh).
    pub ssh: Option<String>,
}

impl FullCluster {
    /// Read + parse the full topology from [`ENV_FULL_CLUSTER_JSON`].
    ///
    /// Loud errors on missing var, hex/JSON decode failure, or schema
    /// violations. The launcher-only path; not relevant on rank children.
    pub fn from_env() -> Result<Self> {
        let raw = env::var(ENV_FULL_CLUSTER_JSON).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: reading {ENV_FULL_CLUSTER_JSON} failed: {e}"
            ))
        })?;
        let bytes = crate::distributed::cluster::hex_decode(raw.trim()).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: {ENV_FULL_CLUSTER_JSON} hex-decode failed: {e}"
            ))
        })?;
        let val: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: {ENV_FULL_CLUSTER_JSON} JSON parse failed: {e}"
            ))
        })?;
        Self::from_value(&val)
    }

    /// Parse from a pre-decoded JSON value. Test entry point + future
    /// programmatic callers.
    pub fn from_value(val: &serde_json::Value) -> Result<Self> {
        let obj = val.as_object().ok_or_else(|| {
            TensorError::new("cluster launcher: top-level JSON must be an object")
        })?;

        let master_addr = obj
            .get("master_addr")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensorError::new("cluster launcher: master_addr (string) required"))?
            .to_string();
        if master_addr.trim().is_empty() {
            return Err(TensorError::new(
                "cluster launcher: master_addr must be non-empty",
            ));
        }

        let master_port_u64 = obj
            .get("master_port")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| TensorError::new("cluster launcher: master_port (u16) required"))?;
        let master_port = u16::try_from(master_port_u64).map_err(|_| {
            TensorError::new(&format!(
                "cluster launcher: master_port must fit in u16 (got {master_port_u64})"
            ))
        })?;

        let hosts_val = obj
            .get("hosts")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorError::new("cluster launcher: hosts (array) required"))?;
        if hosts_val.is_empty() {
            return Err(TensorError::new(
                "cluster launcher: hosts must be non-empty",
            ));
        }

        let hosts: Vec<FullHost> = hosts_val
            .iter()
            .enumerate()
            .map(|(i, h)| parse_full_host(h, i))
            .collect::<Result<_>>()?;

        // Cross-host rank check: union must be exactly 0..world_size.
        let mut all: Vec<usize> = hosts.iter().flat_map(|h| h.ranks.iter().copied()).collect();
        let ws = all.len();
        all.sort_unstable();
        let expected: Vec<usize> = (0..ws).collect();
        if all != expected {
            return Err(TensorError::new(&format!(
                "cluster launcher: ranks across hosts must be exactly 0..{ws} \
                 with no duplicates or gaps, got sorted-unique sequence {all:?}"
            )));
        }

        Ok(FullCluster {
            master_addr,
            master_port,
            hosts,
        })
    }

    /// Total ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.hosts.iter().map(|h| h.ranks.len()).sum()
    }

    /// Whether the cluster spans more than one physical host.
    pub fn spans_multiple_hosts(&self) -> bool {
        self.hosts.len() > 1
    }
}

fn parse_full_host(v: &serde_json::Value, i: usize) -> Result<FullHost> {
    let obj = v.as_object().ok_or_else(|| {
        TensorError::new(&format!("cluster launcher: hosts[{i}] must be an object"))
    })?;

    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: hosts[{i}].name (string) required"
            ))
        })?
        .to_string();
    if name.trim().is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: hosts[{i}].name must be non-empty"
        )));
    }

    let ranks_arr = obj
        .get("ranks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: hosts[{i}] ({name:?}): ranks (array) required"
            ))
        })?;
    if ranks_arr.is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: hosts[{i}] ({name:?}): ranks must be non-empty"
        )));
    }
    let ranks: Vec<usize> = ranks_arr
        .iter()
        .enumerate()
        .map(|(j, e)| {
            let n = e.as_u64().ok_or_else(|| {
                TensorError::new(&format!(
                    "cluster launcher: hosts[{i}].ranks[{j}]: non-integer entry"
                ))
            })?;
            usize::try_from(n).map_err(|_| {
                TensorError::new(&format!(
                    "cluster launcher: hosts[{i}].ranks[{j}]: value {n} out of range"
                ))
            })
        })
        .collect::<Result<_>>()?;

    let local_devices = match obj.get("local_devices") {
        None => {
            return Err(TensorError::new(&format!(
                "cluster launcher: hosts[{i}] ({name:?}): local_devices required"
            )));
        }
        Some(serde_json::Value::String(s)) if s == "all" => None,
        Some(serde_json::Value::String(s)) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: hosts[{i}] ({name:?}): local_devices: \
                 expected \"all\" or array, got string {s:?}"
            )));
        }
        Some(serde_json::Value::Array(arr)) => {
            let v: Vec<u8> = arr
                .iter()
                .enumerate()
                .map(|(j, e)| {
                    let n = e.as_u64().ok_or_else(|| {
                        TensorError::new(&format!(
                            "cluster launcher: hosts[{i}].local_devices[{j}]: \
                             non-integer entry"
                        ))
                    })?;
                    u8::try_from(n).map_err(|_| {
                        TensorError::new(&format!(
                            "cluster launcher: hosts[{i}].local_devices[{j}]: \
                             value {n} does not fit in u8"
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            if v.len() != ranks.len() {
                return Err(TensorError::new(&format!(
                    "cluster launcher: hosts[{i}] ({name:?}): ranks ({}) and \
                     local_devices ({}) length mismatch",
                    ranks.len(),
                    v.len()
                )));
            }
            Some(v)
        }
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: hosts[{i}] ({name:?}): local_devices: \
                 expected \"all\" or array, got {other}"
            )));
        }
    };

    let nccl_socket_ifname = obj
        .get("nccl_socket_ifname")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: hosts[{i}] ({name:?}): nccl_socket_ifname (string) required"
            ))
        })?
        .to_string();

    let path = obj
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: hosts[{i}] ({name:?}): path (string) required"
            ))
        })?
        .to_string();
    if path.trim().is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: hosts[{i}] ({name:?}): path must be non-empty"
        )));
    }

    let libtorch_path = obj
        .get("libtorch_path")
        .and_then(|v| v.as_str())
        .map(String::from);

    let ssh = obj
        .get("ssh")
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(FullHost {
        name,
        ranks,
        local_devices,
        nccl_socket_ifname,
        path,
        libtorch_path,
        ssh,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn canonical_full_json() -> serde_json::Value {
        json!({
            "master_addr": "192.168.122.1",
            "master_port": 29500,
            "hosts": [
                {
                    "name": "master-host",
                    "ranks": [0],
                    "local_devices": [0],
                    "nccl_socket_ifname": "virbr0",
                    "path": "/opt/flodl",
                    "libtorch_path": "/data/ssd/flodl/libtorch"
                },
                {
                    "name": "worker-host",
                    "ssh": "worker-host",
                    "ranks": [1, 2],
                    "local_devices": "all",
                    "nccl_socket_ifname": "enp1s0",
                    "path": "/srv/flodl"
                }
            ]
        })
    }

    #[test]
    fn parses_full_topology() {
        let c = FullCluster::from_value(&canonical_full_json()).unwrap();
        assert_eq!(c.master_addr, "192.168.122.1");
        assert_eq!(c.master_port, 29500);
        assert_eq!(c.world_size(), 3);
        assert!(c.spans_multiple_hosts());

        assert_eq!(c.hosts.len(), 2);
        assert_eq!(c.hosts[0].name, "master-host");
        assert_eq!(c.hosts[0].ranks, vec![0]);
        assert_eq!(c.hosts[0].local_devices, Some(vec![0]));
        assert_eq!(c.hosts[0].ssh, None);

        assert_eq!(c.hosts[1].name, "worker-host");
        assert_eq!(c.hosts[1].ranks, vec![1, 2]);
        // "all" stays unresolved at launcher-parse time; each host resolves
        // its own at startup.
        assert_eq!(c.hosts[1].local_devices, None);
        assert_eq!(c.hosts[1].ssh.as_deref(), Some("worker-host"));
    }

    #[test]
    fn rejects_empty_hosts() {
        let mut v = canonical_full_json();
        v["hosts"] = json!([]);
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("hosts must be non-empty"), "got: {err}");
    }

    #[test]
    fn rejects_rank_gap_across_hosts() {
        let mut v = canonical_full_json();
        v["hosts"][1]["ranks"] = json!([2, 3]); // gap: 0 + (2,3) misses rank 1
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(
            err.to_string().contains("duplicates or gaps"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_duplicate_ranks() {
        let mut v = canonical_full_json();
        v["hosts"][1]["ranks"] = json!([0, 1]); // collides with master-host's [0]
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(
            err.to_string().contains("duplicates or gaps"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_local_devices_length_mismatch_for_explicit() {
        let mut v = canonical_full_json();
        v["hosts"][1]["local_devices"] = json!([0]); // ranks: [1, 2] needs 2 devices
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn accepts_local_devices_all_at_launcher_parse_time() {
        // "all" stays symbolic; resolution is deferred to startup on the
        // host that ends up parsing the slim envelope.
        let mut v = canonical_full_json();
        v["hosts"][0]["local_devices"] = json!("all");
        let c = FullCluster::from_value(&v).unwrap();
        assert_eq!(c.hosts[0].local_devices, None);
    }

    #[test]
    fn rejects_unknown_local_devices_string() {
        let mut v = canonical_full_json();
        v["hosts"][0]["local_devices"] = json!("every");
        let err = FullCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("every"),
            "got: {msg}"
        );
    }

    #[test]
    fn rejects_master_port_overflow() {
        let mut v = canonical_full_json();
        v["master_port"] = json!(100_000);
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("u16"), "got: {err}");
    }
}
