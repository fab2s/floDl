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
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::thread;

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

/// Launcher-mode orchestration. Read full topology, spawn ranks, wait
/// for them, return when every rank child has exited.
///
/// **4b.B.4a scope: local-only fan-out.** Spawns rank processes for
/// the host whose name matches this launcher's hostname (fork-exec of
/// `current_exe`). Errors loudly when the topology lists any other host
/// (ssh fan-out lands in 4b.B.4b).
///
/// **Not yet wired:** during the 4b transition, today's
/// `flodl-cli/src/cluster.rs` still does its own N-child fan-out, so
/// `dispatch()` never reaches launcher role in practice — the FLODL_LOCAL_RANK
/// env var fdl-cli sets pushes role detection straight to `Rank`. 4b.C
/// flips fdl-cli to spawn a single launcher child instead, at which point
/// this function gets exercised end-to-end.
fn run_launcher() -> Result<()> {
    let full = FullCluster::from_env()?;
    let me = crate::distributed::cluster::resolve_hostname()?;

    // Reject ssh-requiring topologies in 4b.B.4a.
    for host in &full.hosts {
        if host.name != me && host.ssh.is_some() {
            return Err(TensorError::new(&format!(
                "cluster launcher: ssh fan-out to remote host {:?} not yet \
                 implemented (4b.B.4b). Local-only topologies work in this \
                 build.",
                host.name
            )));
        }
        if host.name != me && host.ssh.is_none() {
            return Err(TensorError::new(&format!(
                "cluster launcher: host {:?} declared but missing ssh: field \
                 and hostname doesn't match this launcher ({me:?}). Either \
                 set ssh: for remote hosts (deferred to 4b.B.4b) or restrict \
                 the topology to this host.",
                host.name
            )));
        }
    }

    // Find our host's slice.
    let my_host = full.hosts.iter().find(|h| h.name == me).ok_or_else(|| {
        TensorError::new(&format!(
            "cluster launcher: this host's name {me:?} doesn't appear in \
             cluster.hosts. Controller-without-rank orchestrator-only mode \
             isn't supported in 4b.B.4a (no remote hosts to spawn). Add this \
             host to cluster.hosts or run on a host that's already listed."
        ))
    })?;

    let exe = env::current_exe().map_err(|e| {
        TensorError::new(&format!(
            "cluster launcher: current_exe() failed: {e}"
        ))
    })?;
    let user_args: Vec<String> = env::args().skip(1).collect();

    // Spawn one child per rank on this host.
    let mut children: Vec<(usize, std::process::Child, Vec<thread::JoinHandle<()>>)> =
        Vec::with_capacity(my_host.ranks.len());
    for local_rank in 0..my_host.ranks.len() {
        let envelope = build_slim_envelope_for(&full, my_host);
        let envelope_hex = crate::distributed::cluster::hex_encode(
            serde_json::to_string(&envelope)
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster launcher: serialize slim envelope failed: {e}"
                    ))
                })?
                .as_bytes(),
        );

        let mut cmd = Command::new(&exe);
        cmd.args(&user_args)
            .env(
                crate::distributed::cluster::ENV_CLUSTER_JSON,
                &envelope_hex,
            )
            .env(
                crate::distributed::cluster::ENV_LOCAL_RANK,
                local_rank.to_string(),
            )
            // FLODL_FULL_CLUSTER_JSON is launcher-only — strip it from
            // children so they detect Role::Rank, not Role::Launcher.
            .env_remove(ENV_FULL_CLUSTER_JSON)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: spawn rank {local_rank} of {} failed: {e}",
                my_host.name
            ))
        })?;

        // Forward stdout/stderr line-by-line with a rank prefix so the
        // controller's terminal shows interleaved-but-attributable output.
        let global_rank = my_host.ranks[local_rank];
        let prefix = format!("[{}:r{global_rank}] ", my_host.name);
        let mut forwarders = Vec::with_capacity(2);
        if let Some(out) = child.stdout.take() {
            let prefix_clone = prefix.clone();
            forwarders.push(thread::spawn(move || {
                forward_lines(out, prefix_clone, false);
            }));
        }
        if let Some(err) = child.stderr.take() {
            let prefix_clone = prefix.clone();
            forwarders.push(thread::spawn(move || {
                forward_lines(err, prefix_clone, true);
            }));
        }
        children.push((local_rank, child, forwarders));
    }

    // Wait for every child + join forwarders. Propagate non-zero exit
    // codes as a loud error.
    let mut any_failure: Option<TensorError> = None;
    for (local_rank, mut child, forwarders) in children {
        let status = child.wait().map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: wait on rank {local_rank} failed: {e}"
            ))
        })?;
        for f in forwarders {
            let _ = f.join();
        }
        if !status.success() {
            let code = status.code().unwrap_or(-1);
            let msg = format!(
                "cluster launcher: rank {local_rank} of {} exited with status {code}",
                my_host.name
            );
            // Keep the first failure; continue waiting on the rest so
            // we don't leave zombies if multiple ranks die.
            if any_failure.is_none() {
                any_failure = Some(TensorError::new(&msg));
            } else {
                eprintln!("{msg}");
            }
        }
    }
    if let Some(err) = any_failure {
        return Err(err);
    }
    Ok(())
}

/// Build the slim per-host envelope JSON the rank process consumes via
/// [`LocalCluster::from_env`]. Mirrors the shape fdl-cli emits today; the
/// duplication will go away once 4b.C completes.
///
/// [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env
fn build_slim_envelope_for(full: &FullCluster, host: &FullHost) -> serde_json::Value {
    use serde_json::Value;
    let mut host_obj = serde_json::Map::new();
    host_obj.insert("name".into(), Value::String(host.name.clone()));
    host_obj.insert(
        "ranks".into(),
        Value::Array(host.ranks.iter().map(|r| Value::from(*r)).collect()),
    );
    host_obj.insert(
        "local_devices".into(),
        match &host.local_devices {
            None => Value::String("all".into()),
            Some(v) => Value::Array(v.iter().map(|d| Value::from(*d)).collect()),
        },
    );
    host_obj.insert(
        "nccl_socket_ifname".into(),
        Value::String(host.nccl_socket_ifname.clone()),
    );
    host_obj.insert("path".into(), Value::String(host.path.clone()));
    if let Some(p) = &host.libtorch_path {
        host_obj.insert("libtorch_path".into(), Value::String(p.clone()));
    }

    let mut envelope = serde_json::Map::new();
    envelope.insert(
        "master_addr".into(),
        Value::String(full.master_addr.clone()),
    );
    envelope.insert("master_port".into(), Value::from(full.master_port));
    envelope.insert("world_size".into(), Value::from(full.world_size()));
    envelope.insert("num_hosts".into(), Value::from(full.hosts.len()));
    envelope.insert("host".into(), Value::Object(host_obj));
    Value::Object(envelope)
}

/// Forward a child stream line-by-line with a prefix, mirroring
/// fdl-cli's launcher behavior. `to_stderr=true` routes to stderr (per
/// `feedback_docker_stdout_buffering` for debug-level output), else
/// stdout.
fn forward_lines<R: std::io::Read>(stream: R, prefix: String, to_stderr: bool) {
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        match line {
            Ok(l) => {
                if to_stderr {
                    eprintln!("{prefix}{l}");
                } else {
                    println!("{prefix}{l}");
                }
            }
            Err(_) => break,
        }
    }
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

    #[test]
    fn slim_envelope_strips_ssh_carries_metadata() {
        // Direct test of the build_slim_envelope_for helper: the slim
        // shape must round-trip through LocalCluster::from_env on the
        // rank side, so it has to match that parser's expectations
        // (master_addr/master_port/world_size/num_hosts/host with no
        // ssh field).
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let worker = full.hosts.iter().find(|h| h.name == "worker-host").unwrap();
        let env = build_slim_envelope_for(&full, worker);

        assert_eq!(env["master_addr"], "192.168.122.1");
        assert_eq!(env["master_port"], 29500);
        assert_eq!(env["world_size"], 3);
        assert_eq!(env["num_hosts"], 2);
        assert_eq!(env["host"]["name"], "worker-host");
        assert_eq!(env["host"]["ranks"], serde_json::json!([1, 2]));
        assert_eq!(env["host"]["local_devices"], serde_json::json!("all"));
        assert_eq!(env["host"]["nccl_socket_ifname"], "enp1s0");
        // ssh: stripped (launcher-only field; slim envelope is rank-side).
        assert!(env["host"].get("ssh").is_none(), "ssh must be stripped");
    }

    #[test]
    fn slim_envelope_emits_explicit_local_devices_when_present() {
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let master = full.hosts.iter().find(|h| h.name == "master-host").unwrap();
        let env = build_slim_envelope_for(&full, master);
        assert_eq!(env["host"]["local_devices"], serde_json::json!([0]));
    }

    #[test]
    fn slim_envelope_round_trips_through_local_cluster_parser() {
        // Smoke test: the slim envelope built by the launcher must parse
        // cleanly via the rank-side LocalCluster::from_value. Same wire
        // contract, validated end-to-end.
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let master = full.hosts.iter().find(|h| h.name == "master-host").unwrap();
        let env = build_slim_envelope_for(&full, master);
        let parsed = crate::distributed::cluster::LocalCluster::from_value(&env)
            .expect("slim envelope must parse via LocalCluster::from_value");
        assert_eq!(parsed.world_size(), 3);
        assert_eq!(parsed.master_addr, "192.168.122.1");
        assert_eq!(parsed.host.name, "master-host");
        assert_eq!(parsed.host.ranks, vec![0]);
        assert_eq!(parsed.host.local_devices, vec![0]);
    }
}
