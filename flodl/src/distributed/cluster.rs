//! Per-node cluster view for multi-host DDP.
//!
//! A [`LocalCluster`] is the per-node slice of the cluster topology, shipped
//! by `fdl-cli` to each host via the `FLODL_CLUSTER_JSON` environment
//! variable (hex-encoded JSON). It carries:
//!
//! - Master coordinates (`master_addr`, `master_port`) so non-master ranks
//!   know where to phone home.
//! - World metadata (`world_size`, `num_hosts`) needed by NCCL bootstrap.
//! - This host's slice (`host`) -- its ranks, CUDA devices, NCCL socket
//!   interface, project path, libtorch path.
//!
//! The library never sees the full cross-host topology; that lives in
//! `fdl-cli` and stays on the controller. The slim envelope is roughly
//! 250 bytes for a 3-host setup, comfortably below `ARG_MAX` even for
//! pathological cluster sizes.
//!
//! Use [`LocalCluster::from_env`] at startup -- absent env var returns
//! `Ok(None)` (single-host mode). [`LocalCluster::rendezvous`] bootstraps
//! the NCCL communicator, returning a
//! [`TcpRendezvous`](super::TcpRendezvous) with this host's local ranks,
//! CUDA devices, and the shared NCCL unique ID.

use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::log;
use crate::tensor::Device;
use crate::{Result, TensorError};

use super::{NcclUniqueId, TcpRendezvous};

/// Environment variable carrying the hex-encoded JSON envelope.
///
/// `fdl-cli`'s launcher sets this when invoking the remote command per host;
/// the library reads it via [`LocalCluster::from_env`]. Presence of this var
/// is also the recursion guard -- a remote `fdl <cmd>` invocation seeing it
/// skips its own cluster-dispatch branch.
pub const ENV_CLUSTER_JSON: &str = "FLODL_CLUSTER_JSON";

/// Environment variable that overrides the OS hostname for cluster lookup.
///
/// Useful for production test rigs where the OS hostname doesn't match the
/// `cluster.hosts[].name` entry (e.g. a VM whose libvirt-assigned hostname
/// drifts from the deployment label).
pub const ENV_HOST_OVERRIDE: &str = "FLODL_HOST_NAME";

/// Environment variable picking this process's local-rank index within its
/// host. Indexes into `cluster.host.ranks` / `cluster.host.local_devices`
/// (positionally paired). Mirrors torchrun's `LOCAL_RANK`.
///
/// In the process-per-rank model, each spawned child has exactly one entry
/// here. The launcher (`flodl-cli/src/cluster.rs`) injects a distinct value
/// per child. The library uses it via [`LocalCluster::my_rank`] to pick the
/// global rank and CUDA device for this process out of the envelope.
pub const ENV_LOCAL_RANK: &str = "FLODL_LOCAL_RANK";

thread_local! {
    /// Per-thread hostname override used by integration tests that spawn
    /// multiple "host" threads in one process. Higher priority than the
    /// env var because cargo tests cannot set distinct env values per thread.
    static THREAD_HOSTNAME_OVERRIDE: RefCell<Option<String>> = const { RefCell::new(None) };

    /// Per-thread local-rank override used by integration tests that spawn
    /// multiple rank threads in one process (each thread is one rank).
    /// Higher priority than [`ENV_LOCAL_RANK`] because cargo tests cannot
    /// set distinct env values per thread. Parallel to
    /// [`THREAD_HOSTNAME_OVERRIDE`].
    static THREAD_LOCAL_RANK_OVERRIDE: RefCell<Option<usize>> = const { RefCell::new(None) };
}

/// Set the per-thread hostname override seen by [`LocalCluster::this_host`].
///
/// Test-only seam. Production code should set [`ENV_HOST_OVERRIDE`] or rely
/// on the OS `hostname` command. Calling with `None` clears the override.
#[cfg(test)]
pub(crate) fn set_thread_hostname_override(name: Option<&str>) {
    THREAD_HOSTNAME_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = name.map(String::from);
    });
}

/// Set the per-thread local-rank override seen by [`LocalCluster::my_rank`]
/// (via [`local_rank_index_from_env`]).
///
/// Test-only seam. Production code sets [`ENV_LOCAL_RANK`] (the fdl-cli
/// launcher injects this per spawned child). Calling with `None` clears the
/// override. Parallel to [`set_thread_hostname_override`].
#[cfg(test)]
pub(crate) fn set_thread_local_rank_override(idx: Option<usize>) {
    THREAD_LOCAL_RANK_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = idx;
    });
}

/// Per-node view of the cluster, shipped by `fdl-cli` via [`ENV_CLUSTER_JSON`].
///
/// Fields are public for transparency; the canonical constructor is
/// [`LocalCluster::from_env`] (production) or [`LocalCluster::from_json`] /
/// [`LocalCluster::from_value`] (tests / standalone scripts).
#[derive(Debug, Clone)]
pub struct LocalCluster {
    /// Hostname or IP where the NCCL bootstrap rendezvous listens. Must be
    /// reachable by every non-master host. Typically the address of the host
    /// owning rank 0.
    pub master_addr: String,

    /// TCP port for the rendezvous handshake. Port `master_port + 1` is
    /// reserved for the future dashboard side-channel.
    pub master_port: u16,

    /// Total number of ranks across the cluster.
    pub world_size: usize,

    /// Number of physical hosts in the cluster. The master accepts
    /// `num_hosts - 1` incoming TCP connections during rendezvous.
    pub num_hosts: usize,

    /// This host's slice of the topology.
    pub host: HostBlock,
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

    /// Project checkout path on this host. `fdl-cli` cd's here before
    /// invoking the remote command. Surfaces in logs for "which checkout
    /// am I running from?" diagnostics; the library does not otherwise
    /// consume this field.
    pub path: String,

    /// Path to the libtorch install for `fdl-cli` to bind-mount into the
    /// Docker container on this host. Hint for the launcher only; the
    /// library does not consume this field.
    pub libtorch_path: Option<String>,
}

impl LocalCluster {
    /// Read the per-node envelope from [`ENV_CLUSTER_JSON`].
    ///
    /// Returns `Ok(None)` when the env var is absent -- single-host mode is
    /// the default and not an error. Returns `Err` on malformed hex / JSON /
    /// invalid topology (loud errors over silent fallback).
    pub fn from_env() -> Result<Option<Self>> {
        let raw = match env::var(ENV_CLUSTER_JSON) {
            Ok(s) => s,
            Err(env::VarError::NotPresent) => return Ok(None),
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "cluster: reading {ENV_CLUSTER_JSON} failed: {e}"
                )));
            }
        };
        let bytes = hex_decode(raw.trim()).map_err(|e| {
            TensorError::new(&format!(
                "cluster: {ENV_CLUSTER_JSON} hex-decode failed: {e}"
            ))
        })?;
        let val: Value = serde_json::from_slice(&bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster: {ENV_CLUSTER_JSON} JSON parse failed: {e}"
            ))
        })?;
        Self::from_value(&val).map(Some)
    }

    /// Parse a slim per-node envelope from a JSON file.
    ///
    /// Production reads via [`Self::from_env`]; this entry point exists for
    /// tests and any standalone driver that wants to persist envelopes to
    /// disk.
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

    /// Parse from an already-deserialized JSON value. Validates structure.
    pub fn from_value(val: &Value) -> Result<Self> {
        let obj = val
            .as_object()
            .ok_or_else(|| TensorError::new("cluster: top-level JSON must be an object"))?;

        let master_addr = obj
            .get("master_addr")
            .and_then(Value::as_str)
            .ok_or_else(|| TensorError::new("cluster: master_addr (string) required"))?
            .to_string();
        if master_addr.trim().is_empty() {
            return Err(TensorError::new("cluster: master_addr must be non-empty"));
        }

        let master_port_u64 = obj
            .get("master_port")
            .and_then(Value::as_u64)
            .ok_or_else(|| TensorError::new("cluster: master_port (u16) required"))?;
        let master_port = u16::try_from(master_port_u64).map_err(|_| {
            TensorError::new(&format!(
                "cluster: master_port must fit in u16 (got {master_port_u64})"
            ))
        })?;

        let world_size = obj
            .get("world_size")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| TensorError::new("cluster: world_size (usize) required"))?;
        if world_size == 0 {
            return Err(TensorError::new("cluster: world_size must be > 0"));
        }

        let num_hosts = obj
            .get("num_hosts")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| TensorError::new("cluster: num_hosts (usize) required"))?;
        if num_hosts == 0 {
            return Err(TensorError::new("cluster: num_hosts must be > 0"));
        }
        if num_hosts > world_size {
            return Err(TensorError::new(&format!(
                "cluster: num_hosts ({num_hosts}) cannot exceed world_size ({world_size})"
            )));
        }

        let host_val = obj
            .get("host")
            .ok_or_else(|| TensorError::new("cluster: host (object) required"))?;
        let host = parse_host(host_val)?;

        for &r in &host.ranks {
            if r >= world_size {
                return Err(TensorError::new(&format!(
                    "cluster.host ({:?}): rank {r} out of bounds for world_size {world_size}",
                    host.name
                )));
            }
        }

        Ok(LocalCluster {
            master_addr,
            master_port,
            world_size,
            num_hosts,
            host,
        })
    }

    /// Total number of ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Consistency check: resolved hostname must match the envelope's
    /// `host.name`. If they mismatch, the launcher shipped this envelope to
    /// the wrong host -- loud error.
    ///
    /// **Side effect**: on success, registers the host name with the logger
    /// via [`crate::log::set_node_label`]. This must happen before worker
    /// threads spawn; subsequent calls are no-ops (the label is a `OnceLock`).
    pub fn this_host(&self) -> Result<&HostBlock> {
        let name = resolve_hostname()?;
        if name != self.host.name {
            return Err(TensorError::new(&format!(
                "cluster: resolved hostname {name:?} does not match envelope's \
                 host.name {:?} -- the launcher shipped this envelope to the \
                 wrong host (set {ENV_HOST_OVERRIDE} to override for test rigs)",
                self.host.name
            )));
        }
        log::set_node_label(&self.host.name);
        Ok(&self.host)
    }

    /// Whether this host owns rank 0 (the rendezvous master).
    pub fn is_master_host(&self) -> Result<bool> {
        Ok(self.host.ranks.contains(&0))
    }

    /// Pick this process's `(global_rank, device)` out of the envelope.
    ///
    /// Reads [`ENV_LOCAL_RANK`] and indexes into `this_host().ranks` /
    /// `this_host().local_devices` (positionally paired). In the
    /// process-per-rank model, each spawned child owns exactly one slot here.
    ///
    /// Loud errors:
    /// - [`ENV_LOCAL_RANK`] unset (cluster mode requires every process to
    ///   know its own slot; the launcher injects this per child)
    /// - value does not parse as `usize`
    /// - value is out of bounds vs `this_host().ranks.len()`
    ///
    /// Side effect: calls [`Self::this_host`], which validates the hostname
    /// matches the envelope and registers the node label with the logger.
    pub fn my_rank(&self) -> Result<(usize, Device)> {
        let host = self.this_host()?;
        let idx = local_rank_index_from_env(host.ranks.len(), &host.name)?;
        Ok((host.ranks[idx], Device::CUDA(host.local_devices[idx])))
    }

    /// Whether the cluster spans more than one physical host.
    pub fn spans_multiple_hosts(&self) -> bool {
        self.num_hosts > 1
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

fn parse_host(v: &Value) -> Result<HostBlock> {
    let obj = v
        .as_object()
        .ok_or_else(|| TensorError::new("cluster.host: must be an object"))?;

    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| TensorError::new("cluster.host.name (string) required"))?
        .to_string();

    let ranks = parse_usize_array(obj.get("ranks"), "cluster.host.ranks")?;
    if ranks.is_empty() {
        return Err(TensorError::new(&format!(
            "cluster.host ({name:?}): ranks must be non-empty"
        )));
    }

    let local_devices = parse_local_devices(
        obj.get("local_devices"),
        &name,
        ranks.len(),
    )?;

    let nccl_socket_ifname = obj
        .get("nccl_socket_ifname")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster.host ({name:?}): nccl_socket_ifname (string) required"
            ))
        })?
        .to_string();

    let path = obj
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster.host ({name:?}): path (string) required"
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
        path,
        libtorch_path,
    })
}

/// Parse the `local_devices` field of a host entry, accepting either:
///
/// - An explicit array of CUDA device indices, paired positionally with
///   `ranks` (length must match).
/// - The string `"all"`, resolved here via [`crate::tensor::cuda_device_count`]
///   to indices `0..ranks_len`. The host must have at least `ranks_len`
///   visible CUDA devices, otherwise loud error.
///
/// Symmetric for controller and remote nodes. Each host resolves its own
/// `"all"` at envelope-parse time, using the GPU count visible to its
/// running process (CUDA_VISIBLE_DEVICES applies).
fn parse_local_devices(v: Option<&Value>, host_name: &str, ranks_len: usize) -> Result<Vec<u8>> {
    let v = v.ok_or_else(|| {
        TensorError::new("cluster.host.local_devices: required ([..] or \"all\")")
    })?;

    if let Some(s) = v.as_str() {
        if s != "all" {
            return Err(TensorError::new(&format!(
                "cluster.host.local_devices: expected \"all\" or array, got string {s:?}"
            )));
        }
        let available = crate::tensor::cuda_device_count();
        if available < 0 {
            return Err(TensorError::new(
                "cluster.host.local_devices: \"all\" requires CUDA support; \
                 cuda_device_count() returned a negative value",
            ));
        }
        let available = available as usize;
        if available < ranks_len {
            return Err(TensorError::new(&format!(
                "cluster.host ({host_name:?}): local_devices: \"all\" \
                 resolved to {available} visible CUDA device(s), but \
                 ranks.len() = {ranks_len} requires at least that many. \
                 Check CUDA_VISIBLE_DEVICES and host GPU inventory."
            )));
        }
        return Ok((0..ranks_len as u8).collect());
    }

    let devs_u64 = parse_u64_array(Some(v), "cluster.host.local_devices")?;
    let local_devices: Vec<u8> = devs_u64
        .into_iter()
        .map(|d| {
            u8::try_from(d).map_err(|_| {
                TensorError::new(&format!(
                    "cluster.host.local_devices: value {d} does not fit in u8"
                ))
            })
        })
        .collect::<Result<_>>()?;

    if ranks_len != local_devices.len() {
        return Err(TensorError::new(&format!(
            "cluster.host ({host_name:?}): ranks ({}) and local_devices ({}) length mismatch",
            ranks_len,
            local_devices.len()
        )));
    }
    Ok(local_devices)
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

/// Read and validate the local-rank index.
///
/// Priority: thread-local override (test seam, [`set_thread_local_rank_override`])
/// first, then [`ENV_LOCAL_RANK`]. `local_count` is `this_host().ranks.len()`;
/// `host_name` surfaces in error messages to disambiguate which host the
/// launcher targeted. Loud errors on env-unset (when no thread override),
/// unparseable, or out-of-bounds.
fn local_rank_index_from_env(local_count: usize, host_name: &str) -> Result<usize> {
    let idx = if let Some(i) = THREAD_LOCAL_RANK_OVERRIDE.with(|c| *c.borrow()) {
        i
    } else {
        let raw = env::var(ENV_LOCAL_RANK).map_err(|_| {
            TensorError::new(&format!(
                "cluster: {ENV_LOCAL_RANK} not set; in cluster mode each process \
                 must own exactly one local rank. The fdl-cli launcher injects \
                 this env var per spawned child -- if you are running cluster \
                 code without the launcher, set it manually."
            ))
        })?;
        let trimmed = raw.trim();
        trimmed.parse::<usize>().map_err(|e| {
            TensorError::new(&format!(
                "cluster: {ENV_LOCAL_RANK}={trimmed:?} is not a valid usize: {e}"
            ))
        })?
    };
    if idx >= local_count {
        return Err(TensorError::new(&format!(
            "cluster: {ENV_LOCAL_RANK}={idx} out of bounds for host {host_name:?} \
             (host owns {local_count} local rank(s); valid indexes are \
             0..{local_count})"
        )));
    }
    Ok(idx)
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

/// Decode a hex string (any case, no separators) to raw bytes.
///
/// Used by [`LocalCluster::from_env`]; also exposed for the matching encoder
/// in test setup. Zero-dep: hand-rolled to keep `flodl` free of `hex` crate.
pub(crate) fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err(format!("odd-length hex string ({} chars)", s.len()));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> std::result::Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(10 + b - b'a'),
        b'A'..=b'F' => Ok(10 + b - b'A'),
        _ => Err(format!("invalid hex character {:?}", b as char)),
    }
}

/// Hex-encode raw bytes. Companion to [`hex_decode`]; used by tests and by
/// the matching encoder in `fdl-cli`'s launcher.
#[cfg(test)]
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(TABLE[(b >> 4) as usize] as char);
        s.push(TABLE[(b & 0x0F) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn worker_envelope() -> Value {
        json!({
            "master_addr": "192.168.122.1",
            "master_port": 29500,
            "world_size": 3,
            "num_hosts": 2,
            "host": {
                "name": "worker-host",
                "ranks": [1, 2],
                "local_devices": [0, 1],
                "nccl_socket_ifname": "enp1s0",
                "path": "/srv/flodl",
                "libtorch_path": "/mnt/flodl/libtorch"
            }
        })
    }

    fn master_envelope() -> Value {
        json!({
            "master_addr": "192.168.122.1",
            "master_port": 29500,
            "world_size": 3,
            "num_hosts": 2,
            "host": {
                "name": "master-host",
                "ranks": [0],
                "local_devices": [0],
                "nccl_socket_ifname": "virbr0",
                "path": "/opt/flodl",
                "libtorch_path": "/data/ssd/flodl/libtorch"
            }
        })
    }

    #[test]
    fn parses_canonical_envelope() {
        let c = LocalCluster::from_value(&worker_envelope()).expect("parse");
        assert_eq!(c.master_addr, "192.168.122.1");
        assert_eq!(c.master_port, 29500);
        assert_eq!(c.world_size(), 3);
        assert_eq!(c.num_hosts, 2);
        assert!(c.spans_multiple_hosts());

        assert_eq!(c.host.name, "worker-host");
        assert_eq!(c.host.ranks, vec![1, 2]);
        assert_eq!(c.host.local_devices, vec![0, 1]);
        assert_eq!(c.host.nccl_socket_ifname, "enp1s0");
        assert_eq!(c.host.path, "/srv/flodl");
        assert_eq!(c.host.libtorch_path.as_deref(), Some("/mnt/flodl/libtorch"));
    }

    #[test]
    fn rejects_local_devices_all_when_cuda_devices_insufficient() {
        // `local_devices: "all"` resolves on the host that receives the
        // envelope, using cuda_device_count(). With ranks of length > visible
        // CUDA devices (always true in CPU test mode, where count == 0), the
        // resolver must error loudly mentioning the count.
        let mut v = worker_envelope();
        v["host"]["local_devices"] = json!("all");
        let err = LocalCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("\"all\""),
            "expected loud error mentioning local_devices: all, got: {msg}"
        );
    }

    #[test]
    fn parses_local_devices_all_when_cuda_sufficient() {
        // Success path: only meaningful when CUDA is available with enough
        // devices for the rank count. Self-skip in CPU mode.
        let avail = crate::tensor::cuda_device_count();
        if avail < 1 {
            eprintln!("cuda_device_count() = {avail}; skipping all-shorthand success test");
            return;
        }
        // Build an envelope with ranks_len = 1 (always satisfiable when at
        // least one CUDA device is visible).
        let mut v = master_envelope();
        v["host"]["local_devices"] = json!("all");
        let c = LocalCluster::from_value(&v).expect("parse with local_devices: all");
        // Resolved indices are 0..ranks_len; ranks_len = 1 in master_envelope.
        assert_eq!(c.host.local_devices, vec![0u8]);
    }

    #[test]
    fn rejects_local_devices_unknown_string() {
        let mut v = worker_envelope();
        v["host"]["local_devices"] = json!("every");
        let err = LocalCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("every"),
            "expected loud error mentioning the bad value, got: {msg}"
        );
    }

    #[test]
    fn rejects_missing_path() {
        let mut v = worker_envelope();
        v["host"].as_object_mut().unwrap().remove("path");
        let err = LocalCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("path"), "got: {msg}");
        assert!(msg.contains("worker-host"), "got: {msg}");
    }

    #[test]
    fn rejects_ranks_devices_len_mismatch() {
        let mut v = worker_envelope();
        v["host"]["ranks"] = json!([1]);
        v["host"]["local_devices"] = json!([0, 1]);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn rejects_rank_out_of_bounds() {
        let mut v = worker_envelope();
        v["host"]["ranks"] = json!([1, 5]);
        v["host"]["local_devices"] = json!([0, 1]);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("out of bounds"), "got: {err}");
    }

    #[test]
    fn rejects_zero_world_size() {
        let mut v = worker_envelope();
        v["world_size"] = json!(0);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }

    #[test]
    fn rejects_num_hosts_exceeding_world_size() {
        let mut v = worker_envelope();
        v["num_hosts"] = json!(99);
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("num_hosts"), "got: {err}");
    }

    #[test]
    fn rejects_master_port_overflow() {
        let mut v = worker_envelope();
        v["master_port"] = json!(100_000); // > u16::MAX
        let err = LocalCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("u16"), "got: {err}");
    }

    #[test]
    fn is_master_host_works() {
        let worker = LocalCluster::from_value(&worker_envelope()).unwrap();
        let master = LocalCluster::from_value(&master_envelope()).unwrap();
        assert!(!worker.is_master_host().unwrap());
        assert!(master.is_master_host().unwrap());
    }

    #[test]
    fn this_host_matches_envelope() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: ENV_MUTEX serializes env-mutating tests in this module.
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
        }
        let h = c.this_host().expect("hostname matches");
        assert_eq!(h.name, "worker-host");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
    }

    #[test]
    fn this_host_loud_error_on_mismatch() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "wrong-host");
        }
        let err = c.this_host().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains("wrong-host"), "got: {msg}");
        assert!(msg.contains("worker-host"), "got: {msg}");
        assert!(msg.contains(ENV_HOST_OVERRIDE), "got: {msg}");
    }

    #[test]
    fn thread_local_override_beats_env_var() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "wrong-host");
        }
        // Thread-local takes precedence; even though env says "wrong-host",
        // the thread-local says "worker-host" which matches the envelope.
        set_thread_hostname_override(Some("worker-host"));
        let h = c.this_host().expect("thread-local wins");
        assert_eq!(h.name, "worker-host");
        set_thread_hostname_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::remove_var(ENV_CLUSTER_JSON);
        }
        assert!(LocalCluster::from_env().unwrap().is_none());
    }

    #[test]
    fn from_env_round_trips_hex() {
        let v = worker_envelope();
        let bytes = serde_json::to_vec(&v).unwrap();
        let hex = hex_encode(&bytes);

        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_CLUSTER_JSON, &hex);
        }
        let c = LocalCluster::from_env().expect("decode ok").expect("Some");
        unsafe {
            env::remove_var(ENV_CLUSTER_JSON);
        }
        assert_eq!(c.host.name, "worker-host");
        assert_eq!(c.world_size, 3);
        assert_eq!(c.num_hosts, 2);
    }

    #[test]
    fn from_env_rejects_bad_hex() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_CLUSTER_JSON, "not-valid-hex-zz");
        }
        let err = LocalCluster::from_env().unwrap_err();
        unsafe {
            env::remove_var(ENV_CLUSTER_JSON);
        }
        assert!(err.to_string().contains("hex-decode"), "got: {err}");
    }

    #[test]
    fn hex_round_trip() {
        let data = b"\x00\x0fhello\xff\xab";
        let h = hex_encode(data);
        assert_eq!(h, "000f68656c6c6fffab");
        let back = hex_decode(&h).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn hex_decode_uppercase() {
        let back = hex_decode("FF0A").unwrap();
        assert_eq!(back, vec![0xFF, 0x0A]);
    }

    #[test]
    fn hex_decode_rejects_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn my_rank_picks_first_slot() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::set_var(ENV_LOCAL_RANK, "0");
        }
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        // worker_envelope: ranks=[1,2], local_devices=[0,1]. Index 0 -> (1, CUDA(0)).
        assert_eq!(global_rank, 1);
        assert_eq!(device, Device::CUDA(0));
    }

    #[test]
    fn my_rank_picks_second_slot() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::set_var(ENV_LOCAL_RANK, "1");
        }
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        // worker_envelope: index 1 -> (2, CUDA(1)).
        assert_eq!(global_rank, 2);
        assert_eq!(device, Device::CUDA(1));
    }

    #[test]
    fn my_rank_loud_error_when_env_unset() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::remove_var(ENV_LOCAL_RANK);
        }
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains(ENV_LOCAL_RANK), "got: {msg}");
        assert!(msg.contains("not set"), "got: {msg}");
    }

    #[test]
    fn my_rank_loud_error_on_unparseable_value() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::set_var(ENV_LOCAL_RANK, "not-a-number");
        }
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        let msg = err.to_string();
        assert!(msg.contains(ENV_LOCAL_RANK), "got: {msg}");
        assert!(msg.contains("not-a-number"), "got: {msg}");
    }

    #[test]
    fn my_rank_loud_error_on_oob_index() {
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::set_var(ENV_LOCAL_RANK, "5"); // host owns 2 ranks (indexes 0,1)
        }
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        let msg = err.to_string();
        assert!(msg.contains("out of bounds"), "got: {msg}");
        assert!(msg.contains("worker-host"), "got: {msg}");
        // The error names the valid range so the user can fix the launcher.
        assert!(msg.contains("0..2"), "got: {msg}");
    }

    #[test]
    fn my_rank_accepts_whitespace_padded_value() {
        // The launcher emits unquoted numeric, but defending against
        // user-set values with stray whitespace is cheap.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::set_var(ENV_LOCAL_RANK, "  1  ");
        }
        let (gr, _) = c.my_rank().expect("trimmed parse ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        assert_eq!(gr, 2);
    }

    #[test]
    fn my_rank_thread_local_override_beats_env() {
        // Threaded multi-rank tests set distinct thread-local rank overrides
        // per thread; env vars are process-wide and would conflict. The
        // thread-local must win.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::set_var(ENV_LOCAL_RANK, "0"); // env says 0, override says 1
        }
        set_thread_local_rank_override(Some(1));
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        set_thread_local_rank_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        // worker_envelope: index 1 -> (2, CUDA(1)). Override wins over env=0.
        assert_eq!(global_rank, 2);
        assert_eq!(device, Device::CUDA(1));
    }

    #[test]
    fn my_rank_thread_local_override_works_without_env() {
        // In threaded tests, env var is typically unset; the override is the
        // sole source of the local-rank index.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::remove_var(ENV_LOCAL_RANK);
        }
        set_thread_local_rank_override(Some(0));
        let (global_rank, device) = c.my_rank().expect("my_rank ok");
        set_thread_local_rank_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        assert_eq!(global_rank, 1);
        assert_eq!(device, Device::CUDA(0));
    }

    #[test]
    fn my_rank_thread_local_override_clears_back_to_env() {
        // After clearing the override (passing None), the env path takes
        // over -- and absent env should produce the canonical loud error.
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::remove_var(ENV_LOCAL_RANK);
        }
        set_thread_local_rank_override(Some(0));
        let _ = c.my_rank().expect("override path ok");
        set_thread_local_rank_override(None);
        let err = c.my_rank().unwrap_err();
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        // No override, no env -> loud error mentioning the env var.
        assert!(err.to_string().contains(ENV_LOCAL_RANK), "got: {err}");
        assert!(err.to_string().contains("not set"), "got: {err}");
    }

    #[test]
    fn my_rank_thread_local_override_oob_still_bounds_checked() {
        // The bounds check runs regardless of source: an out-of-bounds
        // override index produces the same loud error as an out-of-bounds
        // env value. Catches test bugs (wrong index passed to the helper).
        let c = LocalCluster::from_value(&worker_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "worker-host");
            env::remove_var(ENV_LOCAL_RANK);
        }
        set_thread_local_rank_override(Some(99)); // host owns 2 ranks
        let err = c.my_rank().unwrap_err();
        set_thread_local_rank_override(None);
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
        }
        let msg = err.to_string();
        assert!(msg.contains("out of bounds"), "got: {msg}");
        assert!(msg.contains("0..2"), "got: {msg}");
    }

    #[test]
    fn my_rank_single_rank_host() {
        // master_envelope: ranks=[0], local_devices=[0]. Single-rank host.
        // FLODL_LOCAL_RANK=0 still must be set per the explicit-overlay
        // contract -- launcher always injects it.
        let c = LocalCluster::from_value(&master_envelope()).unwrap();
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            env::set_var(ENV_HOST_OVERRIDE, "master-host");
            env::set_var(ENV_LOCAL_RANK, "0");
        }
        let (global_rank, device) = c.my_rank().expect("single-rank ok");
        unsafe {
            env::remove_var(ENV_HOST_OVERRIDE);
            env::remove_var(ENV_LOCAL_RANK);
        }
        assert_eq!(global_rank, 0);
        assert_eq!(device, Device::CUDA(0));
    }

    // Env-mutating tests need a module-level Mutex; "unique env var name" is
    // not enough when multiple tests touch the same var.
    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());
}
