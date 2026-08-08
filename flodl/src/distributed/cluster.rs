//! Per-node cluster view for multi-host DDP.
//!
//! A [`LocalCluster`] is the per-node slice of the cluster topology, shipped
//! by `fdl-cli` to each host via the `FLODL_INTERNAL_CLUSTER_JSON` environment
//! variable (hex-encoded JSON). It carries:
//!
//! - Controller coordinates (`controller.host`, `controller.port`) so every
//!   rank knows where to dial in to the orchestrator.
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
//! the NCCL communicator, returning a `TcpRendezvous` with this
//! host's local ranks, CUDA devices, and the shared NCCL unique ID.

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

use super::NcclUniqueId;
use super::rendezvous::TcpRendezvous;
use super::wire::SessionSalt;

/// Environment variable carrying the hex-encoded JSON envelope.
///
/// `fdl-cli`'s launcher sets this when invoking the remote command per host;
/// the library reads it via [`LocalCluster::from_env`]. Presence of this var
/// is also the recursion guard -- a remote `fdl <cmd>` invocation seeing it
/// skips its own cluster-dispatch branch.
pub const ENV_CLUSTER_JSON: &str = "FLODL_INTERNAL_CLUSTER_JSON";

/// Environment variable that overrides the OS hostname for cluster lookup.
///
/// Useful for production test rigs where the OS hostname doesn't match the
/// `cluster.workers[].host` entry (e.g. a VM whose libvirt-assigned hostname
/// drifts from the deployment label).
pub const ENV_HOST_OVERRIDE: &str = "FLODL_HOST_NAME";

/// Environment variable picking this process's local-rank index within its
/// host. Indexes into `cluster.worker.ranks` / `cluster.worker.local_devices`
/// (positionally paired). Mirrors torchrun's `LOCAL_RANK`.
///
/// In the process-per-rank model, each spawned child has exactly one entry
/// here. The launcher (`flodl-cli/src/cluster.rs`) injects a distinct value
/// per child. The library uses it via [`LocalCluster::my_rank`] to pick the
/// global rank and CUDA device for this process out of the envelope.
pub const ENV_LOCAL_RANK: &str = "FLODL_INTERNAL_LOCAL_RANK";

/// True if `key` must not appear in a user-supplied cluster/host `env`
/// map (cluster-yml `env:` blocks or [`crate::distributed::ClusterBuilder::env`]
/// / [`crate::distributed::HostBuilder::env`]).
///
/// The launcher splices the per-rank command environment as a shell
/// assignment prefix and applies user env *after* its own built-ins, so
/// a user key that collides with a built-in would win (shell last-wins)
/// and silently break device mapping, rank identity, or the HMAC
/// envelope. Two classes are reserved:
///
/// - Anything with the loud `FLODL_INTERNAL_` prefix: launcher-private
///   transport / identity vars a user never sets. The prefix makes this
///   self-documenting and future-proof (new internals are covered
///   automatically).
/// - A small set of names that are user-facing *elsewhere* but reserved
///   inside a cluster env-map because the launcher sets them per-rank or
///   they select global behavior: `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` (per-rank
///   device pin + the enumeration order it was derived under; cannot be
///   renamed — they are the vendor/driver contract),
///   [`ENV_HOST_OVERRIDE`] (`FLODL_HOST_NAME`, a legit standalone
///   override but launcher-set per rank here), and `FDL_ENV` (the
///   overlay selector; a per-rank override would split-brain the
///   config).
///
/// User-facing knobs like `FLODL_VERBOSITY` and `FLODL_DASHBOARD_BIND`
/// are deliberately NOT reserved — they carry no `FLODL_INTERNAL_`
/// prefix and are safe to set per cluster/host.
pub fn is_reserved_cluster_env_key(key: &str) -> bool {
    key.starts_with("FLODL_INTERNAL_")
        || key == "CUDA_VISIBLE_DEVICES"
        || key == "CUDA_DEVICE_ORDER"
        // The AMD masks outrank CUDA_VISIBLE_DEVICES for HIP (first one
        // set wins), so an env-block value would silently defeat the
        // launcher's per-rank pin — the same reservation, other vendor.
        || key == "HIP_VISIBLE_DEVICES"
        || key == "ROCR_VISIBLE_DEVICES"
        || key == "GPU_DEVICE_ORDINAL"
        || key == ENV_HOST_OVERRIDE
        || key == crate::distributed::launcher::ENV_FDL_ENV
}

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
    /// Controller's rendezvous bind point. Every worker dials
    /// `controller.host:controller.port` for the NCCL bootstrap.
    pub controller: ControllerBlock,

    /// Total number of ranks across the cluster.
    pub world_size: usize,

    /// Number of physical workers in the cluster. The controller accepts
    /// `num_workers` incoming TCP connections during rendezvous (the
    /// controller is NOT itself a rank).
    pub num_workers: usize,

    /// This worker's slice of the topology.
    pub worker: WorkerBlock,

    /// 128-bit session salt generated by the launcher and shipped to
    /// every rank's envelope. Used as the HMAC key for the cross-process
    /// control channel and the `RoundFrame` data channel; see
    /// `wire` for the wire-protocol details.
    ///
    /// Absent / zero in single-host non-cluster mode (no cross-process
    /// channels to authenticate).
    ///
    pub salt: SessionSalt,

    /// Controller wants per-rank resource samples attached to metrics
    /// reports. Set by the launcher when its harness carries a
    /// [`crate::monitor::Timeline`] (the samples persist host-qualified
    /// into `timeline.json`); a rank-side `monitor.serve(port)` enables
    /// the same sampling independently of this flag. Absent = `false` —
    /// headless runs pay no NVML poller.
    pub rank_resources: bool,
}

/// Controller bind coordinates, shipped per rank inside the envelope.
#[derive(Debug, Clone)]
pub struct ControllerBlock {
    /// Hostname or IP where the controller's single accepting port
    /// listens. Must be reachable by every worker.
    pub host: String,

    /// The controller's single TCP port: NCCL rendezvous, CPU-reduce
    /// data, and coordinator control all accept here, routed by each
    /// connection's channel-select magic (see `port_mux`). Only the
    /// host-local rank↔relay loopback channels use derived offsets
    /// (`port + 4`/`+5`), which never leave the worker host.
    pub port: u16,
}

/// Per-worker topology entry.
///
/// `ranks` and `local_devices` are positionally paired: `ranks[i]` runs
/// on CUDA device `local_devices[i]`. Validation enforces equal length.
#[derive(Debug, Clone)]
pub struct WorkerBlock {
    /// Hostname / identifier (was `WorkerBlock.name`). As reported by the
    /// `hostname` command on this machine, or the value of
    /// `FLODL_HOST_NAME` if set.
    pub host: String,

    /// Global ranks owned by this worker. Must be a subset of
    /// `0..world_size`.
    pub ranks: Vec<usize>,

    /// CUDA device indices (`0..num_visible_gpus`) backing each rank.
    /// Paired by position with `ranks`.
    pub local_devices: Vec<u8>,

    /// Network interface NCCL should bind to (e.g. `virbr0`, `enp1s0`).
    /// Surfaces in `NCCL_SOCKET_IFNAME` -- loud error if unset when the
    /// cluster spans multiple workers.
    pub nccl_socket_ifname: String,

    /// Project checkout path on this worker. `fdl-cli` cd's here before
    /// invoking the remote command. Surfaces in logs for "which
    /// checkout am I running from?" diagnostics; the library does not
    /// otherwise consume this field.
    pub path: String,

    /// libtorch variant subpath under `<path>/libtorch/` on this
    /// worker. The runtime libtorch lives at `<path>/libtorch/<arch>/`
    /// by convention. Hint for the launcher only; the library does not
    /// consume this field.
    pub arch: Option<String>,

    /// Dataset source root on this worker: the directory its ranks READ
    /// training data from. May be a shared mount visible to every host
    /// or a node-local directory, and the library does not distinguish
    /// them: it forwards the path, the training binary reads it.
    ///
    /// `None` when the host did not declare `data_path:`. Only an
    /// EXPLICIT declaration travels, so the convention default never
    /// arrives as a path that may not exist. Reach it through
    /// [`LocalCluster::data_path`] rather than this field, so
    /// single-host runs take the same code path.
    pub data_path: Option<String>,

    /// Integrated-GPU host-RAM share for this host: the fraction of
    /// `MemTotal` the GPU aperture claims (same knob as
    /// `DataLoaderBuilder::gpu_ram_share`; discrete GPUs ignore it).
    /// Already resolved by the controller (per-host declaration over
    /// cluster-scope default) and overwritten by a walk-in's own
    /// `join.gpu_ram_share:` at envelope localization. The trainer
    /// fills `DdpRunConfig::gpu_ram_share` from it only when the
    /// binary's config left it `None` — explicit code keeps the last
    /// word, like a passed `--data-dir` does for `data_path`.
    pub gpu_ram_share: Option<f64>,
}

/// This process's dataset source root, or `None` when nothing declared
/// one.
///
/// The one call a training binary needs to honour a cluster's
/// `data_path:` without knowing whether it is running under a cluster
/// at all. `None` covers both "no cluster envelope" (a solo run) and
/// "cluster, but this host declared no source root", which want the
/// same answer: keep whatever default the binary defines.
///
/// Returns `Err` only on a malformed envelope, which is the same error
/// the rank would hit moments later during rendezvous. Propagate it
/// rather than treating a corrupt envelope as an absent one.
///
/// ```no_run
/// # fn main() -> flodl::tensor::Result<()> {
/// let data_dir = match flodl::distributed::cluster_data_path()? {
///     Some(p) => p,
///     None => std::path::PathBuf::from("data"),
/// };
/// # let _ = data_dir;
/// # Ok(())
/// # }
/// ```
pub fn cluster_data_path() -> Result<Option<std::path::PathBuf>> {
    Ok(LocalCluster::from_env()?.and_then(|c| c.data_path().map(std::path::PathBuf::from)))
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

        let controller_val = obj
            .get("controller")
            .and_then(Value::as_object)
            .ok_or_else(|| TensorError::new("cluster: controller (object) required"))?;
        let controller_host = controller_val
            .get("host")
            .and_then(Value::as_str)
            .ok_or_else(|| TensorError::new("cluster: controller.host (string) required"))?
            .to_string();
        if controller_host.trim().is_empty() {
            return Err(TensorError::new(
                "cluster: controller.host must be non-empty",
            ));
        }
        let controller_port_u64 = controller_val
            .get("port")
            .and_then(Value::as_u64)
            .ok_or_else(|| TensorError::new("cluster: controller.port (u16) required"))?;
        let controller_port = u16::try_from(controller_port_u64).map_err(|_| {
            TensorError::new(&format!(
                "cluster: controller.port must fit in u16 (got {controller_port_u64})"
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

        let num_workers = obj
            .get("num_workers")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| TensorError::new("cluster: num_workers (usize) required"))?;
        if num_workers == 0 {
            return Err(TensorError::new("cluster: num_workers must be > 0"));
        }
        if num_workers > world_size {
            return Err(TensorError::new(&format!(
                "cluster: num_workers ({num_workers}) cannot exceed world_size ({world_size})"
            )));
        }

        let worker_val = obj
            .get("worker")
            .ok_or_else(|| TensorError::new("cluster: worker (object) required"))?;
        let worker = parse_worker(worker_val)?;

        for &r in &worker.ranks {
            if r >= world_size {
                return Err(TensorError::new(&format!(
                    "cluster.worker ({:?}): rank {r} out of bounds for world_size {world_size}",
                    worker.host
                )));
            }
        }

        // Session salt is optional in the envelope: the launcher fills
        // it in for cluster runs, but it is absent for envelopes built
        // manually (tests, single-host code paths). Default to all
        // zeros in that case -- equivalent to the prior "no salt"
        // behavior, and HMACs against an all-zero key still authenticate
        // intra-cluster as long as every participant agrees.
        let salt = match obj.get("salt").and_then(Value::as_str) {
            Some(s) => super::wire::salt_from_hex(s)
                .map_err(|e| TensorError::new(&format!("cluster: salt: {e}")))?,
            None => [0u8; super::wire::SESSION_SALT_BYTES],
        };

        // Optional resource-sampling opt-in; envelopes predating the
        // field (or built manually) default to off.
        let rank_resources = obj
            .get("rank_resources")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        Ok(LocalCluster {
            controller: ControllerBlock {
                host: controller_host,
                port: controller_port,
            },
            world_size,
            num_workers,
            worker,
            salt,
            rank_resources,
        })
    }

    /// Total number of ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// This worker's dataset source root, when its host declared one.
    ///
    /// Unlike [`Self::this_worker`] this does not verify the hostname:
    /// the envelope carries exactly one worker block and it is this
    /// process's own, so reading a path out of it needs no identity
    /// check and stays usable before the logger label is set.
    pub fn data_path(&self) -> Option<&str> {
        self.worker.data_path.as_deref()
    }

    /// This host's integrated-GPU RAM share, when anything declared one
    /// (per-host entry, cluster-scope default, or a walk-in's own join
    /// config — resolved in that reverse order before the envelope
    /// reached this process). See [`WorkerBlock::gpu_ram_share`].
    pub fn gpu_ram_share(&self) -> Option<f64> {
        self.worker.gpu_ram_share
    }

    /// Consistency check: resolved hostname must match the envelope's
    /// `worker.host`. If they mismatch, the launcher shipped this envelope to
    /// the wrong host -- loud error.
    ///
    /// **Side effect**: on success, registers the host name with the logger
    /// via [`crate::log::set_node_label`]. This must happen before worker
    /// threads spawn; subsequent calls are no-ops (the label is a `OnceLock`).
    pub fn this_worker(&self) -> Result<&WorkerBlock> {
        let name = resolve_hostname()?;
        if name != self.worker.host {
            return Err(TensorError::new(&format!(
                "cluster: resolved hostname {name:?} does not match envelope's \
                 worker.host {:?} -- the launcher shipped this envelope to the \
                 wrong host (set {ENV_HOST_OVERRIDE} to override for test rigs)",
                self.worker.host
            )));
        }
        log::set_node_label(&self.worker.host);
        Ok(&self.worker)
    }

    /// Pick this process's `(global_rank, device)` out of the envelope.
    ///
    /// Reads [`ENV_LOCAL_RANK`] and indexes into `this_worker().ranks` /
    /// `this_worker().local_devices` (positionally paired). In the
    /// process-per-rank model, each spawned child owns exactly one slot here.
    ///
    /// Loud errors:
    /// - [`ENV_LOCAL_RANK`] unset (cluster mode requires every process to
    ///   know its own slot; the launcher injects this per child)
    /// - value does not parse as `usize`
    /// - value is out of bounds vs `this_worker().ranks.len()`
    ///
    /// Side effect: calls [`Self::this_worker`], which validates the
    /// hostname matches the envelope and registers the node label with
    /// the logger.
    pub fn my_rank(&self) -> Result<(usize, Device)> {
        let worker = self.this_worker()?;
        let idx = local_rank_index_from_env(worker.ranks.len(), &worker.host)?;
        // When the launcher per-child scopes the rank via
        // `CUDA_VISIBLE_DEVICES=<phys>` (the standard torchrun-style
        // multi-process-CUDA recipe), the child sees only one GPU, and
        // libtorch addresses it as `CUDA(0)` regardless of the
        // physical index. Returning the envelope's
        // `local_devices[idx]` (the physical index) would point at a
        // device the child can't see and yield `cudaErrorInvalidDevice`
        // (or worse, sticky `cudaErrorNoKernelImageForDevice` when
        // first allocation triggers module load on the wrong CC).
        // Detect the single-value form (`CUDA_VISIBLE_DEVICES=N`) and
        // return `CUDA(0)`; fall back to the envelope when unset or
        // when multi-value (in which case the child sees the same
        // physical layout as the cluster spec).
        //
        // The variable consulted is the one the child's runtime actually
        // reads: HIP prefers its own masks over CUDA_VISIBLE_DEVICES
        // (the first one SET wins), so on a ROCm build checking only the
        // CUDA spelling would miss the mask that scoped the child.
        let mask_vars: &[&str] = match crate::sys::build_vendor() {
            Some(flodl_hw::GpuVendor::Amd) => &[
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
            ],
            _ => &["CUDA_VISIBLE_DEVICES"],
        };
        if let Some(visible) = mask_vars.iter().find_map(|k| std::env::var(k).ok())
            && !visible.is_empty()
            && !visible.contains(',')
        {
            return Ok((worker.ranks[idx], Device::CUDA(0)));
        }
        Ok((worker.ranks[idx], Device::CUDA(worker.local_devices[idx])))
    }

    /// Whether the cluster spans more than one physical host.
    pub fn spans_multiple_workers(&self) -> bool {
        self.num_workers > 1
    }

    /// Bootstrap the NCCL communicator across hosts.
    ///
    /// Master (rank-0 host) generates an [`NcclUniqueId`], binds
    /// [`controller.port`](ControllerBlock::port), and distributes the
    /// ID to every other host. The 32-byte `dataset_signature` is
    /// exchanged at the same time -- loud error on mismatch, since
    /// silent fan-out into different data shards is the worst class of
    /// bug.
    ///
    /// Side effect: calls [`Self::this_worker`], which registers the
    /// node label with the logger.
    pub fn rendezvous(&self, dataset_signature: [u8; 32]) -> Result<TcpRendezvous> {
        TcpRendezvous::establish(self, dataset_signature, NcclUniqueId::new)
    }
}

fn parse_worker(v: &Value) -> Result<WorkerBlock> {
    let obj = v
        .as_object()
        .ok_or_else(|| TensorError::new("cluster.worker: must be an object"))?;

    let name = obj
        .get("host")
        .and_then(Value::as_str)
        .ok_or_else(|| TensorError::new("cluster.worker.host (string) required"))?
        .to_string();

    let ranks = parse_usize_array(obj.get("ranks"), "cluster.worker.ranks")?;
    if ranks.is_empty() {
        return Err(TensorError::new(&format!(
            "cluster.worker ({name:?}): ranks must be non-empty"
        )));
    }

    let local_devices = parse_local_devices(obj.get("local_devices"), &name, ranks.len())?;

    let nccl_socket_ifname = obj
        .get("nccl_socket_ifname")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster.worker ({name:?}): nccl_socket_ifname (string) required"
            ))
        })?
        .to_string();

    let path = obj
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster.worker ({name:?}): path (string) required"
            ))
        })?
        .to_string();

    let arch = obj.get("arch").and_then(Value::as_str).map(String::from);

    let data_path = obj
        .get("data_path")
        .and_then(Value::as_str)
        .map(String::from);

    // Emitted by the controller as a plain JSON number; anything else
    // (or out of range) means a corrupt envelope, not user input — the
    // yml-side validation already happened at topology parse.
    let gpu_ram_share = match obj.get("gpu_ram_share") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
            TensorError::new(&format!(
                "cluster.worker ({name:?}): gpu_ram_share must be a number, got {v}"
            ))
        })?),
    };

    Ok(WorkerBlock {
        host: name,
        ranks,
        local_devices,
        nccl_socket_ifname,
        path,
        arch,
        data_path,
        gpu_ram_share,
    })
}

/// Parse the `local_devices` field of a host entry, accepting either:
///
/// - An explicit array of CUDA device indices, paired positionally with
///   `ranks` (length must match).
/// - The string `"all"`, resolved here via [`crate::tensor::gpu_device_count`]
///   to indices `0..ranks_len`. The host must have at least `ranks_len`
///   visible CUDA devices, otherwise loud error.
///
/// Symmetric for controller and remote nodes. Each host resolves its own
/// `"all"` at envelope-parse time, using the GPU count visible to its
/// running process (CUDA_VISIBLE_DEVICES applies).
fn parse_local_devices(v: Option<&Value>, host_name: &str, ranks_len: usize) -> Result<Vec<u8>> {
    let v = v.ok_or_else(|| {
        TensorError::new("cluster.worker.local_devices: required ([..] or \"all\")")
    })?;

    if let Some(s) = v.as_str() {
        if s != "all" {
            return Err(TensorError::new(&format!(
                "cluster.worker.local_devices: expected \"all\" or array, got string {s:?}"
            )));
        }
        let available = crate::tensor::gpu_device_count();
        if available < 0 {
            return Err(TensorError::new(
                "cluster.worker.local_devices: \"all\" requires CUDA support; \
                 gpu_device_count() returned a negative value",
            ));
        }
        let available = available as usize;
        if available < ranks_len {
            return Err(TensorError::new(&format!(
                "cluster.worker ({host_name:?}): local_devices: \"all\" \
                 resolved to {available} visible CUDA device(s), but \
                 ranks.len() = {ranks_len} requires at least that many. \
                 Check CUDA_VISIBLE_DEVICES and host GPU inventory."
            )));
        }
        return Ok((0..ranks_len as u8).collect());
    }

    let devs_u64 = parse_u64_array(Some(v), "cluster.worker.local_devices")?;
    let local_devices: Vec<u8> = devs_u64
        .into_iter()
        .map(|d| {
            u8::try_from(d).map_err(|_| {
                TensorError::new(&format!(
                    "cluster.worker.local_devices: value {d} does not fit in u8"
                ))
            })
        })
        .collect::<Result<_>>()?;

    if ranks_len != local_devices.len() {
        return Err(TensorError::new(&format!(
            "cluster.worker ({host_name:?}): ranks ({}) and local_devices ({}) length mismatch",
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
            usize::try_from(n)
                .map_err(|_| TensorError::new(&format!("{label}: value {n} does not fit in usize")))
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
/// Priority: thread-local override (test seam, via the private
/// `set_thread_local_rank_override`) first, then [`ENV_LOCAL_RANK`].
/// `local_count` is `this_worker().ranks.len()`;
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

/// Resolve this host's name as the cluster code sees it.
///
/// Test override > [`ENV_HOST_OVERRIDE`] env var > `hostname(1)`
/// command. Trimmed; non-empty guaranteed on success. Used by
/// [`LocalCluster::this_worker`] to match against the envelope and by
/// the launcher to find the matching entry in
/// [`crate::distributed::launcher::FullCluster::workers`].
pub(crate) fn resolve_hostname() -> Result<String> {
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
        return Err(TensorError::new(&format!(
            "cluster: `hostname` command exited non-zero \
             (set {ENV_HOST_OVERRIDE} to override)"
        )));
    }
    let s = String::from_utf8(out.stdout).map_err(|e| {
        TensorError::new(&format!(
            "cluster: hostname output not UTF-8: {e} \
             (set {ENV_HOST_OVERRIDE} to override)"
        ))
    })?;
    Ok(s.trim().to_string())
}

/// Decode a hex string (any case, no separators) to raw bytes.
///
/// Used by [`LocalCluster::from_env`]; also exposed for the matching encoder
/// in test setup. Zero-dep: hand-rolled to keep `flodl` free of `hex` crate.
pub(crate) fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
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

/// Hex-encode raw bytes. Companion to [`hex_decode`]; used by the
/// launcher when building child-process env vars and by test setup
/// that constructs envelopes inline.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(TABLE[(b >> 4) as usize] as char);
        s.push(TABLE[(b & 0x0F) as usize] as char);
    }
    s
}

/// Shared mutex used by all in-crate tests that mutate process env vars
/// touched by [`LocalCluster`] (e.g. [`ENV_CLUSTER_JSON`],
/// [`ENV_LOCAL_RANK`], [`ENV_HOST_OVERRIDE`]). Two test modules in the
/// same crate (e.g. `cluster::tests` + `monitor::tests`) can both touch
/// these vars; without a single shared lock they race in the parallel
/// test harness even when each test uses a unique name internally,
/// because env-var visibility is process-wide.
#[cfg(test)]
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
#[path = "cluster_tests.rs"]
mod tests;
