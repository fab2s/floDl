//! Dashboard sink: the controller-side endpoint for rank-emitted
//! dashboard wire frames.
//!
//! Post controller-active refactor, the live training dashboard runs on
//! the launcher process (not on rank 0). Ranks emit
//! `TimingMsgWire::Dashboard*` frames at startup
//! ([`crate::distributed::wire::TimingMsgWire::DashboardRegister`],
//! [`crate::distributed::wire::TimingMsgWire::DashboardSetSvg`],
//! [`crate::distributed::wire::TimingMsgWire::DashboardSetMetadata`],
//! [`crate::distributed::wire::TimingMsgWire::DashboardSetHardware`])
//! plus per-epoch resource samples piggy-backed on
//! [`crate::distributed::wire::MetricsMsgWire::resources`]. The
//! [`crate::distributed::cluster_coordinator::ClusterCoordinator`]
//! forwards every such frame to an optional [`DashboardSink`], whose
//! concrete implementation in
//! [`crate::distributed::launcher`] wraps the HTTP `DashboardServer`.
//!
//! Kept as a trait (not a concrete struct) so the coord stays decoupled
//! from the dashboard server's HTTP/SSE machinery, and so coord-side
//! unit tests can stub the sink without a real bind.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::distributed::launcher::FullCluster;
use crate::distributed::wire::ResourceSampleWire;
use crate::monitor::{EpochRecord, GpuMetrics, Monitor, ResourceSample};

/// Controller-side endpoint for rank-emitted dashboard wire frames.
///
/// All methods are infallible — the sink's job is to forward to the
/// dashboard server's shared state, never to fail the coord. Methods
/// take `&self` so the sink can be cheaply shared (typically wrapped in
/// `Arc`).
pub trait DashboardSink: Send + Sync {
    /// Rank requested the dashboard be bound on `port`. First-arrival
    /// wins; subsequent registrations validate match (warn on mismatch,
    /// don't re-bind).
    fn register_port(&self, rank: usize, port: u16);

    /// Rank shipped the graph SVG. First non-empty arrival wins; the
    /// SVG is identical across ranks so subsequent are dropped.
    fn set_svg(
        &self,
        rank: usize,
        svg: String,
        label: Option<String>,
        hash: Option<String>,
    );

    /// Rank shipped the dashboard metadata blob (hyperparameters,
    /// config). Last-write-wins; the launcher dashboard serves whatever
    /// is currently set.
    fn set_metadata(&self, rank: usize, json: String);

    /// Rank shipped its hardware summary string. Per-rank — the
    /// dashboard renders one tab per `rank` labelled
    /// `host:lr=<local_rank> gr=<global_rank>` (the launcher resolves
    /// host + local_rank from its [`crate::distributed::launcher::FullCluster`]
    /// world map).
    fn set_hardware(&self, rank: usize, summary: String);

    /// Per-epoch resource sample for `rank`. Carried as the
    /// `resources` field on [`crate::distributed::wire::MetricsMsgWire`]
    /// alongside the existing metric report.
    fn push_resource_sample(&self, rank: usize, sample: ResourceSampleWire);

    /// Aggregated [`crate::distributed::ddp_run::EpochMetrics`] for the
    /// current epoch, ready for the dashboard's main tab.
    fn push_epoch_metrics(
        &self,
        metrics: &crate::distributed::ddp_run::EpochMetrics,
    );
}

/// Concrete [`DashboardSink`] that owns a launcher-hosted [`Monitor`]
/// + per-rank state.
///
/// The HTTP server is bound lazily on the first
/// [`Self::register_port`] call (the rank's `monitor.serve(port)`
/// triggers the registration over the wire). Subsequent registrations
/// validate the port matches and warn on mismatch; ranks all run the
/// same user binary so the port is identical in practice.
///
/// Per-rank resource samples and hardware summaries are stored
/// indexed by global rank. The dashboard's main tab renders the
/// controller-aggregated [`crate::distributed::ddp_run::EpochMetrics`]
/// time series; per-rank tabs render each rank's resource snapshots
/// labelled `host:lr=<local_rank> gr=<global_rank>` (resolved from the
/// launcher's [`FullCluster`] world map).
pub struct ClusterDashboardSink {
    monitor: Mutex<Monitor>,
    cluster: Arc<FullCluster>,
    /// Hostname the dashboard URL prints to the launcher's stderr. Resolved
    /// at construction (typically [`crate::distributed::cluster::resolve_hostname`]).
    controller_host: String,
    /// Bound port (0 = not yet bound).
    bound_port: Mutex<u16>,
    /// `true` once an SVG has been installed; subsequent set_svg calls
    /// are dropped (every rank ships the same SVG).
    svg_installed: Mutex<bool>,
    /// Per-rank hardware summary strings, indexed by global rank.
    per_rank_hardware: Mutex<Vec<Option<String>>>,
    /// Per-rank latest resource sample, indexed by global rank.
    per_rank_resources: Mutex<Vec<Option<ResourceSample>>>,
    /// Wall-clock start (used to compute ETA in pushed epoch records).
    start_time: Instant,
}

impl ClusterDashboardSink {
    /// Build a dashboard sink for the given cluster topology. The
    /// dashboard URL printed on bind is `http://{controller_host}:{port}`;
    /// `controller_host` is typically the launcher's resolved hostname
    /// (e.g. from `cluster::resolve_hostname`).
    ///
    /// `total_epochs` mirrors the [`Monitor::new`] argument; it sets
    /// the dashboard header's "epoch N/total" frame and the ETA
    /// denominator. Pass the user's `num_epochs`.
    pub fn new(
        cluster: Arc<FullCluster>,
        controller_host: String,
        total_epochs: usize,
    ) -> Self {
        let world_size = cluster.world_size();
        let mut mon = Monitor::new(total_epochs);
        // The launcher always serves the dashboard; in-process gating
        // (single-process detect_is_primary path) does not apply here.
        // is_primary is true by default on Monitor::new; the cluster
        // sink keeps it that way.
        mon.silent_summary();
        ClusterDashboardSink {
            monitor: Mutex::new(mon),
            cluster,
            controller_host,
            bound_port: Mutex::new(0),
            svg_installed: Mutex::new(false),
            per_rank_hardware: Mutex::new(vec![None; world_size]),
            per_rank_resources: Mutex::new(vec![None; world_size]),
            start_time: Instant::now(),
        }
    }

    /// Resolve `(host, local_rank)` for a global rank from the world
    /// map. Returns `None` if `rank` is out of range.
    fn resolve_rank(&self, rank: usize) -> Option<(String, usize)> {
        for worker in &self.cluster.workers {
            if let Some(local) = worker.ranks.iter().position(|&r| r == rank) {
                return Some((worker.host.clone(), local));
            }
        }
        None
    }

    /// Render the aggregated hardware string from per-rank entries:
    /// `host:lr=<lr> gr=<gr>: <summary> | …`.
    fn render_aggregated_hardware(&self) -> String {
        let map = self.per_rank_hardware.lock().unwrap();
        let mut parts: Vec<String> = Vec::with_capacity(map.len());
        for (rank, entry) in map.iter().enumerate() {
            if let Some(summary) = entry {
                let label = match self.resolve_rank(rank) {
                    Some((host, lr)) => format!("{host}:lr={lr} gr={rank}"),
                    None => format!("gr={rank}"),
                };
                parts.push(format!("{label}: {summary}"));
            }
        }
        parts.join(" | ")
    }
}

impl DashboardSink for ClusterDashboardSink {
    fn register_port(&self, rank: usize, port: u16) {
        let mut bound = self.bound_port.lock().unwrap();
        if *bound == port {
            return; // idempotent re-registration
        }
        if *bound != 0 {
            eprintln!(
                "cluster dashboard: rank {rank} requested port {port}; \
                 already bound to {} (ignoring)",
                *bound,
            );
            return;
        }
        // First registration — bind the server.
        let mut mon = self.monitor.lock().unwrap();
        match mon.serve(port) {
            Ok(()) => {
                *bound = port;
                eprintln!(
                    "cluster dashboard: http://{}:{}",
                    self.controller_host, port,
                );
            }
            Err(e) => {
                eprintln!(
                    "cluster dashboard: bind port {port} failed: {e}",
                );
            }
        }
    }

    fn set_svg(
        &self,
        _rank: usize,
        svg: String,
        label: Option<String>,
        hash: Option<String>,
    ) {
        let mut installed = self.svg_installed.lock().unwrap();
        if *installed {
            return; // first-arrival wins; SVG is identical across ranks
        }
        let mut mon = self.monitor.lock().unwrap();
        mon.set_svg(&svg);
        mon.set_identity(label.as_deref(), hash.as_deref());
        *installed = true;
    }

    fn set_metadata(&self, _rank: usize, json: String) {
        let value: serde_json::Value = match serde_json::from_str(&json) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "cluster dashboard: discarding malformed metadata JSON: {e}"
                );
                return;
            }
        };
        let mut mon = self.monitor.lock().unwrap();
        mon.set_metadata(value);
    }

    fn set_hardware(&self, rank: usize, summary: String) {
        {
            let mut map = self.per_rank_hardware.lock().unwrap();
            if rank < map.len() {
                map[rank] = Some(summary);
            }
        }
        let aggregated = self.render_aggregated_hardware();
        let mut mon = self.monitor.lock().unwrap();
        mon.set_hardware(aggregated);
    }

    fn push_resource_sample(&self, rank: usize, sample: ResourceSampleWire) {
        let sample: ResourceSample = sample.into();
        let mut map = self.per_rank_resources.lock().unwrap();
        if rank < map.len() {
            map[rank] = Some(sample);
        }
    }

    fn push_epoch_metrics(
        &self,
        metrics: &crate::distributed::ddp_run::EpochMetrics,
    ) {
        // Build a flat `(name, value)` metric vec including avg_loss +
        // every aggregated scalar. Sorted scalar keys give deterministic
        // ordering matching the rank-side log path.
        let mut flat: Vec<(String, f64)> =
            Vec::with_capacity(1 + metrics.scalars.len());
        flat.push(("loss".to_string(), metrics.avg_loss));
        let mut keys: Vec<&String> = metrics.scalars.keys().collect();
        keys.sort();
        for k in keys {
            flat.push((k.clone(), metrics.scalars[k]));
        }

        // Per-rank GPU tabs from EpochMetrics' aggregated arrays.
        let mut gpu_metrics: Vec<GpuMetrics> = Vec::with_capacity(
            metrics.device_indices.len(),
        );
        for (i, &dev) in metrics.device_indices.iter().enumerate() {
            gpu_metrics.push(GpuMetrics {
                device_index: dev,
                throughput: metrics
                    .per_rank_throughput
                    .get(i)
                    .copied()
                    .unwrap_or(0.0),
                chunk_ratio: metrics
                    .per_rank_batch_share
                    .get(i)
                    .copied()
                    .unwrap_or(0.0),
                shard_size: 0, // not tracked per-epoch in cluster mode
            });
        }

        // Synthesize a cluster-wide ResourceSample for the dashboard
        // header / per-rank tabs from the per-rank pushes. `gpus` is
        // built one entry per rank using the rank's primary GPU
        // snapshot (each rank's CUDA_VISIBLE_DEVICES is scoped to one
        // device by the launcher, so each rank's first GPU is its
        // active one). Top-level fields (cpu/ram) come from the most
        // recent rank report — the launcher host's own resources are
        // not sampled here.
        let resources = {
            let map = self.per_rank_resources.lock().unwrap();
            let mut combined = ResourceSample::default();
            for (rank, sample_opt) in map.iter().enumerate() {
                let Some(sample) = sample_opt else { continue };
                // Last-write wins for the scalar fields (best effort).
                if sample.cpu_percent.is_some() {
                    combined.cpu_percent = sample.cpu_percent;
                }
                if sample.ram_used_bytes.is_some() {
                    combined.ram_used_bytes = sample.ram_used_bytes;
                }
                if sample.ram_total_bytes.is_some() {
                    combined.ram_total_bytes = sample.ram_total_bytes;
                }
                if sample.gpu_util_percent.is_some() {
                    combined.gpu_util_percent = sample.gpu_util_percent;
                }
                if sample.vram_total_bytes.is_some() {
                    combined.vram_total_bytes = sample.vram_total_bytes;
                }
                if sample.vram_allocated_bytes.is_some() {
                    combined.vram_allocated_bytes = sample.vram_allocated_bytes;
                }
                if sample.aggregate_rank.is_some() {
                    combined.aggregate_rank = sample.aggregate_rank;
                }
                // Take this rank's primary GPU snapshot as a tab entry,
                // prefixing the device name with `host:lr=<lr> gr=<gr>`
                // so the dashboard JS renders cluster-aware tab labels.
                if let Some(mut gpu) = sample.gpus.first().cloned() {
                    let label_prefix = match self.resolve_rank(rank) {
                        Some((host, lr)) => {
                            format!("{host}:lr={lr} gr={rank} ")
                        }
                        None => format!("gr={rank} "),
                    };
                    gpu.name = format!("{label_prefix}{}", gpu.name);
                    combined.gpus.push(gpu);
                }
            }
            combined
        };

        let duration_secs = self.start_time.elapsed().as_secs_f64();
        let record = EpochRecord {
            epoch: metrics.epoch,
            duration_secs: metrics.epoch_ms / 1000.0,
            metrics: flat,
            resources,
            gpu_metrics,
        };
        let _ = duration_secs; // reserved for future ETA recalibration
        let mut mon = self.monitor.lock().unwrap();
        mon.log_epoch_record(record);
    }
}
