//! `fdl probe` — check host readiness for training.
//!
//! Single-host (default): probes the local box for GPU + libtorch +
//! shared-data path + NCCL. Cluster context (env overlay): each
//! configured host is probed via SSH and the per-host status is
//! aggregated into one report. See [`run`].
//!
//! Design notes
//! ============
//! flodl assumes shared storage is available to every node at the
//! same logical path (NAS / SMB / virtiofs / S3-FUSE / SSHFS). The
//! probe is the gate that confirms each host can see it BEFORE
//! training fans out, instead of discovering it mid-AllReduce when a
//! checkpoint write hangs on a stale mount. The convention default
//! ([`crate::config::DEFAULT_DATA_PATH`]) applies when a host does
//! not declare `data_path:` in `fdl.cluster.yml`.
//!
//! Probe is intentionally thin — it reuses
//! [`crate::libtorch::detect`] and [`crate::util::system`] for the
//! existing detection logic and adds shared-mount + NCCL discovery
//! on top. The format is `diagnose`-style (text by default, `--json`
//! emits machine-readable output for `fdl deploy` / CI to consume).

use std::path::PathBuf;

use crate::context::Context;
use crate::libtorch::detect::LibtorchInfo;
use crate::util::system::GpuInfo;

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

/// Run the probe.
///
/// **Single-host** (no active env overlay): probes the local box and
/// emits one report. `--data-path` overrides config; `--skip-mount`
/// short-circuits the shared-data check.
///
/// **Cluster** (`fdl @cluster probe` / `FDL_ENV=cluster`): loads
/// `fdl.<env>.yml`'s `cluster.workers:` list. For each host: if it's
/// the local host, probes in-process; otherwise SSHes to it and runs
/// `<worker.path>/target/release/fdl probe --json` remotely. Per-host
/// JSON is parsed back into [`ProbeReport`] and aggregated.
///
/// Exit code: `0` when every probed host is green; `1` when any
/// host raised issues.
pub fn run(
    json: bool,
    skip_mount: bool,
    data_path_override: Option<PathBuf>,
    libtorch_path_override: Option<PathBuf>,
    via_docker: Option<String>,
) -> i32 {
    let ctx = Context::resolve();
    // Cluster fan-out only applies when no explicit libtorch override
    // is passed — overrides are how the *remote* probe is invoked, so
    // we must NOT recurse back into cluster mode on the remote side.
    if libtorch_path_override.is_none()
        && let Ok(env_name) = std::env::var("FDL_ENV")
        && let Some(cluster) = load_cluster_for_env(&ctx, &env_name)
    {
        return run_cluster(&cluster, json, skip_mount);
    }
    // Single-host (local OR remote-being-probed). When `--data-path` is
    // passed explicitly, treat a missing path as an error; when absent
    // (falling back to DEFAULT_DATA_PATH), treat it as a warning.
    let data_path_explicit = data_path_override.is_some();
    let report = probe_local(
        &ctx,
        skip_mount,
        data_path_override,
        libtorch_path_override,
        via_docker,
        data_path_explicit,
    );
    if json {
        print_json(&report);
    } else {
        print_report(&report);
    }
    if report.green() { 0 } else { 1 }
}

// ---------------------------------------------------------------------------
// Report structs
// ---------------------------------------------------------------------------

/// Top-level probe verdict for one host. `green()` is the aggregate
/// gate.
///
/// `issues` are blocking errors (exit non-zero); `warnings` are advisory
/// (exit zero, surfaced in the report). The split matters because
/// "/flodl/data missing" on a single-host rig that doesn't use shared
/// storage is informational, while a worker host that declared an
/// explicit `data_path:` in cluster.yml and can't see it is broken.
pub struct ProbeReport {
    pub host: String,
    pub gpus: Vec<GpuInfo>,
    pub libtorch: LibtorchStatus,
    pub data_path: DataPathStatus,
    pub nccl: NcclStatus,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

impl ProbeReport {
    /// `true` when no issues were collected — every checked component
    /// passed (warnings do NOT flip this). Callers may still want to
    /// inspect individual statuses for diagnostic detail; the exit
    /// code follows this flag.
    pub fn green(&self) -> bool {
        self.issues.is_empty()
    }
}

/// libtorch directory + arch metadata + per-GPU compatibility verdict.
pub struct LibtorchStatus {
    /// Parsed `.arch` metadata (if libtorch is present + readable).
    pub info: Option<LibtorchInfo>,
    /// `lib/` subdirectory present (cheap "is this a libtorch dir?"
    /// check that doesn't require parsing).
    pub valid_dir: bool,
    /// Per-GPU `(gpu_index, archs_cover_this_gpu)`. Empty when libtorch
    /// is missing.
    pub archs_match: Vec<(u8, bool)>,
}

/// Shared-data path visibility + filesystem-type detection.
pub struct DataPathStatus {
    pub path: PathBuf,
    pub exists: bool,
    pub readable: bool,
    /// Underlying filesystem type from `/proc/mounts` (e.g. `virtiofs`,
    /// `nfs4`, `cifs`, `fuse.sshfs`, `ext4`). `None` when the path is
    /// not mounted (falls inside the parent FS) or when /proc/mounts
    /// is unavailable.
    pub fs_type: Option<String>,
    /// `true` when the check was explicitly bypassed via
    /// `--skip-mount`; the path/exists fields are unset (`PathBuf::new`
    /// + false) in that case.
    pub skipped: bool,
}

/// NCCL discovery result. NCCL is loaded dynamically by libtorch, so
/// the probe just hunts for `libnccl.so*` on the usual library paths
/// — unless [`Self::via_docker`] is set, in which case NCCL ships
/// inside the container image and the host scan is skipped.
pub struct NcclStatus {
    /// First `libnccl.so*` found, if any. Used in the report to show
    /// the user which install will be picked up.
    pub library_path: Option<PathBuf>,
    /// All discovered `libnccl.so*` paths (informational; multiple
    /// versions in different prefixes is a misconfiguration source).
    pub all_found: Vec<PathBuf>,
    /// Docker compose service that owns NCCL on this host. When set,
    /// the probe records "via Docker image `<svc>`" instead of scanning
    /// the host filesystem. `None` means the host runs flodl natively
    /// and NCCL must live on it.
    pub via_docker: Option<String>,
}

// ---------------------------------------------------------------------------
// Single-host probe
// ---------------------------------------------------------------------------

/// Probe the local host. `data_path_override` (from `--data-path` CLI
/// flag) overrides config; `skip_mount` short-circuits the shared-data
/// check (useful for single-host setups without a shared FS
/// configured); `libtorch_path_override` (from `--libtorch-path`)
/// points at a libtorch install outside the project tree (used by
/// cluster-mode remote probes where libtorch lives on a dedicated
/// share like `/mnt/libtorch`). `via_docker` (from `--docker <svc>` or
/// the cluster.yml host's `docker:` field) tells the probe NCCL ships
/// inside a container image, so host-level NCCL scanning is replaced
/// by an informational "via Docker image `<svc>`" line.
///
/// `data_path_explicit`: when `true`, a missing shared-data path is an
/// ERROR (the user/cluster.yml promised it); when `false`, it's a
/// WARNING (the convention default was used). Internal flag — callers
/// must derive it from "did the caller pass an explicit data_path".
mod checks;
mod cluster;
mod report;

pub use checks::probe_local;
// Crate-internal, so re-exported at that visibility: `prepare` asks the
// same questions about a mountpoint that the probe does.
pub(crate) use checks::{detect_fs_type, mounted_at};
use cluster::{load_cluster_for_env, run_cluster};
use report::{print_json, print_report};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
