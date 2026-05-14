//! Cluster topology for multi-host DDP.
//!
//! A [`Cluster`] is the canonical JSON form of the `cluster:` block from
//! `fdl.yml` (or its overlay). `fdl-cli` deep-merges the YAML, canonicalizes
//! to JSON, and ships the file to each host; the library reads it via
//! [`Cluster::from_json`].
//!
//! Each rank is named explicitly: rank assignment survives YAML reorderings
//! and resumed-training restarts. Hostname self-lookup picks the right host
//! block at startup (`FLODL_HOST_NAME` env var overrides for test rigs).
//!
//! Use [`Cluster::rendezvous`] to bootstrap the NCCL communicator across
//! hosts -- it returns a [`TcpRendezvous`](super::TcpRendezvous) carrying the
//! shared NCCL unique ID, this host's local ranks, and CUDA devices.

use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::log;
use crate::{Result, TensorError};

use super::{NcclUniqueId, TcpRendezvous};

/// Environment variable that overrides the OS hostname for cluster lookup.
///
/// Useful for production test rigs where the OS hostname doesn't match the
/// `cluster.hosts[].name` entry (e.g. a VM whose libvirt-assigned hostname
/// drifts from the deployment label).
pub const ENV_HOST_OVERRIDE: &str = "FLODL_HOST_NAME";

thread_local! {
    /// Per-thread hostname override used by integration tests that spawn
    /// multiple "host" threads in one process. Higher priority than the
    /// env var because cargo tests cannot set distinct env values per thread.
    static THREAD_HOSTNAME_OVERRIDE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Set the per-thread hostname override seen by [`Cluster::this_host`].
///
/// Test-only seam. Production code should set [`ENV_HOST_OVERRIDE`] or rely
/// on the OS `hostname` command. Calling with `None` clears the override.
#[cfg(test)]
pub(crate) fn set_thread_hostname_override(name: Option<&str>) {
    THREAD_HOSTNAME_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = name.map(String::from);
    });
}

/// Canonical cluster topology, parsed from the JSON config file shipped by
/// `fdl-cli` to each host.
///
/// Fields are public to keep the type a transparent value: the canonical
/// constructor is [`Cluster::from_json`], which validates rank assignment
/// before returning. Mutating fields after construction bypasses validation;
/// don't.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Hostname or IP where the NCCL bootstrap rendezvous listens.
    ///
    /// Must be reachable by every non-master host. Typically the address of
    /// whichever host owns rank 0.
    pub master_addr: String,

    /// TCP port for the rendezvous handshake.
    ///
    /// Port `master_port + 1` is reserved for the dashboard side-channel
    /// (see the multi-host DDP design doc).
    pub master_port: u16,

    /// One entry per physical host in the cluster.
    pub hosts: Vec<HostBlock>,
}

/// Per-host topology entry.
///
/// `ranks` and `local_devices` are positionally paired: `ranks[i]` runs on
/// CUDA device `local_devices[i]`. Validation enforces equal length.
#[derive(Debug, Clone)]
pub struct HostBlock {
    /// Hostname as reported by the `hostname` command on this machine, or
    /// the value of `FLODL_HOST_NAME` if set.
    pub name: String,

    /// Global ranks owned by this host. Must be a subset of `0..world_size`.
    pub ranks: Vec<usize>,

    /// CUDA device indices (`0..num_visible_gpus`) backing each rank.
    /// Paired by position with `ranks`.
    pub local_devices: Vec<u8>,

    /// Network interface NCCL should bind to (e.g. `virbr0`, `enp1s0`).
    /// Surfaces in `NCCL_SOCKET_IFNAME` -- loud error if unset when the
    /// cluster spans multiple hosts.
    pub nccl_socket_ifname: String,

    /// Path to the libtorch install for `fdl-cli` to bind-mount into the
    /// Docker container on this host. Hint for the launcher only; the
    /// library does not consume this field.
    pub libtorch_path: Option<String>,
}

impl Cluster {
    /// Parse a canonical cluster config JSON file.
    ///
    /// Validates that ranks form `0..world_size` exactly (no duplicates,
    /// no gaps) and that every host's `ranks.len() == local_devices.len()`.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| {
            TensorError::new(&format!(
                "cluster: failed to open {}: {}",
                path.display(),
                e
            ))
        })?;
        let val: Value = serde_json::from_reader(BufReader::new(file)).map_err(|e| {
            TensorError::new(&format!(
                "cluster: failed to parse {} as JSON: {}",
                path.display(),
                e
            ))
        })?;
        Self::from_value(&val)
    }

    /// Parse from an already-deserialized JSON value. Useful for tests.
    pub fn from_value(val: &Value) -> Result<Self> {
        let obj = val.as_object().ok_or_else(|| {
            TensorError::new("cluster: top-level JSON must be an object")
        })?;

        let master_addr = obj
            .get("master_addr")
            .and_then(Value::as_str)
            .ok_or_else(|| TensorError::new("cluster: master_addr (string) required"))?
            .to_string();

        let master_port_u64 = obj
            .get("master_port")
            .and_then(Value::as_u64)
            .ok_or_else(|| TensorError::new("cluster: master_port (u16) required"))?;
        let master_port = u16::try_from(master_port_u64).map_err(|_| {
            TensorError::new(&format!(
                "cluster: master_port must fit in u16 (got {master_port_u64})"
            ))
        })?;

        let hosts_arr = obj
            .get("hosts")
            .and_then(Value::as_array)
            .ok_or_else(|| TensorError::new("cluster: hosts (array) required"))?;

        if hosts_arr.is_empty() {
            return Err(TensorError::new("cluster: hosts must be non-empty"));
        }

        let mut hosts = Vec::with_capacity(hosts_arr.len());
        for (i, h) in hosts_arr.iter().enumerate() {
            hosts.push(parse_host(h, i)?);
        }

        validate_rank_assignment(&hosts)?;

        Ok(Cluster {
            master_addr,
            master_port,
            hosts,
        })
    }

    /// Total number of ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.hosts.iter().map(|h| h.ranks.len()).sum()
    }

    /// Identify this host within the cluster.
    ///
    /// Resolves the local hostname (overridable via [`ENV_HOST_OVERRIDE`])
    /// and looks it up in `self.hosts`. Loud error listing all known host
    /// names if not found -- silent fallthrough on a mistyped hostname would
    /// be hours of debugging.
    ///
    /// **Side effect**: on success, registers the host name with the logger
    /// via [`crate::log::set_node_label`]. This must happen before worker
    /// threads spawn; subsequent calls are no-ops (the label is a `OnceLock`).
    pub fn this_host(&self) -> Result<&HostBlock> {
        let name = resolve_hostname()?;
        let host = self.hosts.iter().find(|h| h.name == name).ok_or_else(|| {
            let known: Vec<&str> = self.hosts.iter().map(|h| h.name.as_str()).collect();
            TensorError::new(&format!(
                "cluster: hostname {name:?} not in cluster.hosts {known:?} \
                 (set {ENV_HOST_OVERRIDE} to override)"
            ))
        })?;
        log::set_node_label(&host.name);
        Ok(host)
    }

    /// Whether this host owns rank 0 (the rendezvous master).
    pub fn is_master_host(&self) -> Result<bool> {
        Ok(self.this_host()?.ranks.contains(&0))
    }

    /// Whether the cluster spans more than one physical host.
    pub fn spans_multiple_hosts(&self) -> bool {
        self.hosts.len() > 1
    }

    /// Bootstrap the NCCL communicator across hosts.
    ///
    /// Master (rank-0 host) generates an [`NcclUniqueId`](super::NcclUniqueId),
    /// binds [`master_port`](Self::master_port), and distributes the ID to
    /// every other host. The 32-byte `dataset_signature` is exchanged at the
    /// same time -- loud error on mismatch, since silent fan-out into
    /// different data shards is the worst class of bug.
    ///
    /// Side effect: calls [`Self::this_host`], which registers the node
    /// label with the logger.
    pub fn rendezvous(&self, dataset_signature: [u8; 32]) -> Result<TcpRendezvous> {
        TcpRendezvous::establish(self, dataset_signature, NcclUniqueId::new)
    }
}

fn parse_host(v: &Value, idx: usize) -> Result<HostBlock> {
    let obj = v.as_object().ok_or_else(|| {
        TensorError::new(&format!("cluster.hosts[{idx}]: must be an object"))
    })?;

    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            TensorError::new(&format!("cluster.hosts[{idx}].name (string) required"))
        })?
        .to_string();

    let ranks = parse_usize_array(obj.get("ranks"), &format!("cluster.hosts[{idx}].ranks"))?;
    if ranks.is_empty() {
        return Err(TensorError::new(&format!(
            "cluster.hosts[{idx}] ({name:?}): ranks must be non-empty"
        )));
    }

    let devs_u64 = parse_u64_array(
        obj.get("local_devices"),
        &format!("cluster.hosts[{idx}].local_devices"),
    )?;
    let local_devices: Vec<u8> = devs_u64
        .into_iter()
        .map(|d| {
            u8::try_from(d).map_err(|_| {
                TensorError::new(&format!(
                    "cluster.hosts[{idx}].local_devices: value {d} does not fit in u8"
                ))
            })
        })
        .collect::<Result<_>>()?;

    if ranks.len() != local_devices.len() {
        return Err(TensorError::new(&format!(
            "cluster.hosts[{idx}] ({name:?}): ranks ({}) and local_devices ({}) length mismatch",
            ranks.len(),
            local_devices.len()
        )));
    }

    let nccl_socket_ifname = obj
        .get("nccl_socket_ifname")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster.hosts[{idx}] ({name:?}): nccl_socket_ifname (string) required"
            ))
        })?
        .to_string();

    let libtorch_path = obj
        .get("libtorch_path")
        .and_then(Value::as_str)
        .map(String::from);

    Ok(HostBlock {
        name,
        ranks,
        local_devices,
        nccl_socket_ifname,
        libtorch_path,
    })
}

fn parse_usize_array(v: Option<&Value>, label: &str) -> Result<Vec<usize>> {
    let arr = v
        .and_then(Value::as_array)
        .ok_or_else(|| TensorError::new(&format!("{label} (array) required")))?;
    arr.iter()
        .map(|e| {
            let n = e
                .as_u64()
                .ok_or_else(|| TensorError::new(&format!("{label}: non-integer entry")))?;
            usize::try_from(n).map_err(|_| {
                TensorError::new(&format!("{label}: value {n} does not fit in usize"))
            })
        })
        .collect()
}

fn parse_u64_array(v: Option<&Value>, label: &str) -> Result<Vec<u64>> {
    let arr = v
        .and_then(Value::as_array)
        .ok_or_else(|| TensorError::new(&format!("{label} (array) required")))?;
    arr.iter()
        .map(|e| {
            e.as_u64()
                .ok_or_else(|| TensorError::new(&format!("{label}: non-integer entry")))
        })
        .collect()
}

fn validate_rank_assignment(hosts: &[HostBlock]) -> Result<()> {
    let mut all: Vec<usize> = hosts.iter().flat_map(|h| h.ranks.iter().copied()).collect();
    let ws = all.len();
    if ws == 0 {
        return Err(TensorError::new(
            "cluster: total rank count is zero (every host has empty ranks)",
        ));
    }
    all.sort_unstable();
    let expected: Vec<usize> = (0..ws).collect();
    if all != expected {
        return Err(TensorError::new(&format!(
            "cluster: ranks across hosts must be exactly 0..{ws} with no duplicates or gaps, \
             got sorted-unique sequence {all:?}"
        )));
    }
    Ok(())
}

fn resolve_hostname() -> Result<String> {
    if let Some(s) = THREAD_HOSTNAME_OVERRIDE.with(|c| c.borrow().clone()) {
        return Ok(s);
    }
    if let Ok(s) = env::var(ENV_HOST_OVERRIDE) {
        let s = s.trim();
        if !s.is_empty() {
            return Ok(s.to_string());
        }
    }
    let out = Command::new("hostname").output().map_err(|e| {
        TensorError::new(&format!(
            "cluster: `hostname` command failed: {e} \
             (set {ENV_HOST_OVERRIDE} to override)"
        ))
    })?;
    if !out.status.success() {
        return Err(TensorError::new(
            "cluster: `hostname` command exited non-zero",
        ));
    }
    let s = String::from_utf8(out.stdout).map_err(|e| {
        TensorError::new(&format!("cluster: hostname output not UTF-8: {e}"))
    })?;
    Ok(s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn three_rank_two_host() -> Value {
        json!({
            "master_addr": "192.168.122.1",
            "master_port": 29500,
            "hosts": [
                { "name": "fab2s", "ranks": [0], "local_devices": [0],
                  "nccl_socket_ifname": "virbr0",
                  "libtorch_path": "/data/ssd/flodl/libtorch" },
                { "name": "flodl-pascal", "ranks": [1, 2], "local_devices": [0, 1],
                  "nccl_socket_ifname": "enp1s0",
                  "libtorch_path": "/mnt/flodl/libtorch" }
            ]
        })
    }

    #[test]
    fn parses_canonical_topology() {
        let c = Cluster::from_value(&three_rank_two_host()).expect("parse");
        assert_eq!(c.master_addr, "192.168.122.1");
        assert_eq!(c.master_port, 29500);
        assert_eq!(c.world_size(), 3);
        assert_eq!(c.hosts.len(), 2);
        assert!(c.spans_multiple_hosts());

        let pascal = &c.hosts[1];
        assert_eq!(pascal.name, "flodl-pascal");
        assert_eq!(pascal.ranks, vec![1, 2]);
        assert_eq!(pascal.local_devices, vec![0, 1]);
        assert_eq!(pascal.nccl_socket_ifname, "enp1s0");
        assert_eq!(pascal.libtorch_path.as_deref(), Some("/mnt/flodl/libtorch"));
    }

    #[test]
    fn rejects_duplicate_ranks() {
        let mut v = three_rank_two_host();
        v["hosts"][1]["ranks"] = json!([1, 1]);
        let err = Cluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("duplicates or gaps"), "got: {err}");
    }

    #[test]
    fn rejects_rank_gap() {
        let mut v = three_rank_two_host();
        v["hosts"][1]["ranks"] = json!([2, 3]); // 1 missing
        let err = Cluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("duplicates or gaps"), "got: {err}");
    }

    #[test]
    fn rejects_ranks_devices_len_mismatch() {
        let mut v = three_rank_two_host();
        v["hosts"][1]["ranks"] = json!([1]);
        v["hosts"][1]["local_devices"] = json!([0, 1]);
        let err = Cluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn rejects_empty_host_list() {
        let v = json!({
            "master_addr": "127.0.0.1",
            "master_port": 29500,
            "hosts": []
        });
        let err = Cluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("hosts must be non-empty"), "got: {err}");
    }

    #[test]
    fn rejects_master_port_overflow() {
        let mut v = three_rank_two_host();
        v["master_port"] = json!(100_000); // > u16::MAX
        let err = Cluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("u16"), "got: {err}");
    }

    #[test]
    fn this_host_uses_env_override() {
        let v = three_rank_two_host();
        let c = Cluster::from_value(&v).unwrap();
        // SAFETY: serial sets/unsets a process-wide env var; tests in this
        // module rely on env_mutex below to avoid races.
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "flodl-pascal");
        }
        let h = c.this_host().expect("lookup");
        assert_eq!(h.name, "flodl-pascal");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
    }

    #[test]
    fn this_host_loud_error_on_unknown() {
        let v = three_rank_two_host();
        let c = Cluster::from_value(&v).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "not-in-cluster");
        }
        let err = c.this_host().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains("not-in-cluster"), "got: {msg}");
        assert!(msg.contains("fab2s") && msg.contains("flodl-pascal"), "got: {msg}");
        assert!(msg.contains(ENV_HOST_OVERRIDE), "got: {msg}");
    }

    #[test]
    fn thread_local_override_beats_env_var() {
        let v = three_rank_two_host();
        let c = Cluster::from_value(&v).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "fab2s");
        }
        // Thread-local takes precedence; the env var pointing at "fab2s"
        // should be ignored when the thread-local says "flodl-pascal".
        set_thread_hostname_override(Some("flodl-pascal"));
        let h = c.this_host().expect("lookup");
        assert_eq!(h.name, "flodl-pascal");
        set_thread_hostname_override(None);
        // With thread-local cleared, the env var wins.
        let h2 = c.this_host().expect("lookup via env");
        assert_eq!(h2.name, "fab2s");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
    }

    // Env-mutating tests need a module-level Mutex; "unique env var name" is
    // not enough when multiple tests touch the same var.
    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());
}
