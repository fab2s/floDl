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

    /// Signal end-of-training to the dashboard so the SSE `complete`
    /// event fires (browser stops the elapsed counter, switches the
    /// status dot to "done"). Called by the launcher after every rank
    /// child exits — symmetric to single-process [`Monitor::finish`]
    /// in the rank-side path. Default = no-op (test stubs need not
    /// implement). Idempotent.
    fn shutdown(&self) {}
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

    /// Render the aggregated hardware string grouped by host:
    /// `host: <cpu> | gr=N lr=M: <gpu> | gr=K lr=L: <gpu> | other_host: <cpu> | …`.
    ///
    /// Each rank's `summary` is `"<cpu> | <my_gpu>"` (trimmed by
    /// `cluster_worker::trim_hardware_to_assigned` before emit). For
    /// each host we dedup the CPU from the first arriving rank and
    /// list every rank's GPU with a `gr=N lr=M` label. Host order
    /// follows `FullCluster.workers`.
    fn render_aggregated_hardware(&self) -> String {
        let map = self.per_rank_hardware.lock().unwrap();
        let mut host_blocks: Vec<String> = Vec::with_capacity(self.cluster.workers.len());
        for worker in &self.cluster.workers {
            let mut cpu_emitted = false;
            let mut block = format!("{}:", worker.host);
            for (local_rank, &global_rank) in worker.ranks.iter().enumerate() {
                let Some(summary) = map.get(global_rank).and_then(|s| s.as_ref()) else {
                    continue;
                };
                let mut parts = summary.split(" | ");
                let cpu = parts.next().unwrap_or("");
                let gpu = parts.next();
                if !cpu_emitted && !cpu.is_empty() {
                    block.push(' ');
                    block.push_str(cpu);
                    cpu_emitted = true;
                }
                if let Some(gpu) = gpu {
                    block.push_str(&format!(" | gr={global_rank} lr={local_rank}: {gpu}"));
                }
            }
            // Skip hosts with no rank reports yet (block only has
            // `host:` — no CPU appended, no GPUs); avoid emitting an
            // orphan "exa:" segment before any rank on that host has
            // pushed its hardware.
            if cpu_emitted {
                host_blocks.push(block);
            }
        }
        host_blocks.join(" | ")
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
        // First registration — bind the server. The internal-only
        // `serve_local_unconditional` bypasses `Monitor::serve`'s
        // cluster / launcher gating (which would correctly skip the
        // bind on the launcher process — but the sink wants the
        // bind, that's the whole point of the controller-active
        // refactor).
        let mut mon = self.monitor.lock().unwrap();
        match mon.serve_local_unconditional(port) {
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

    fn shutdown(&self) {
        let mut mon = self.monitor.lock().unwrap();
        mon.shutdown_dashboard_server();
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
        // Home tab + per-rank tabs from the per-rank pushes.
        //
        // Aggregation rules:
        // - CPU% and RAM are per-HOST (ranks on the same host share
        //   the same /proc/stat + /proc/meminfo), so dedup by host
        //   first to avoid double-counting. Then:
        //     * cpu_percent  = mean across hosts (cluster compute load)
        //     * ram_used     = sum across hosts (cluster total in use)
        //     * ram_total    = sum across hosts (cluster physical RAM)
        // - GPU util is per-RANK (each rank reports its assigned GPU):
        //     * gpu_util_percent = mean across all ranks (cluster GPU load)
        //     * vram_allocated   = sum across ranks (cluster VRAM in use)
        //     * vram_total       = sum across ranks (cluster physical VRAM)
        // - `gpus` is built one entry per rank with `device_index`
        //   rewritten to the global rank (rank-local sampler returns
        //   `device_index=0` uniformly when CUDA_VISIBLE_DEVICES is
        //   scoped, or duplicates when not — both collapse under the
        //   dashboard's `gpuSeries[g.dev]` key otherwise). Name is
        //   prefixed `host:lr=<lr> gr=<gr> ` so the JS tab labels
        //   stay cluster-aware.
        let resources = {
            let map = self.per_rank_resources.lock().unwrap();
            let mut combined = ResourceSample::default();

            // Per-host dedup for CPU/RAM. Walk topology order (stable);
            // first rank on each host that has a sample contributes.
            let mut cpu_sum = 0.0_f64;
            let mut cpu_count = 0u32;
            let mut ram_used_sum = 0u64;
            let mut ram_total_sum = 0u64;
            for worker in &self.cluster.workers {
                for &global_rank in &worker.ranks {
                    let Some(Some(sample)) = map.get(global_rank) else { continue };
                    // Take first non-empty rank per host, then break.
                    if let Some(cpu) = sample.cpu_percent {
                        cpu_sum += cpu as f64;
                        cpu_count += 1;
                    }
                    if let Some(used) = sample.ram_used_bytes {
                        ram_used_sum = ram_used_sum.saturating_add(used);
                    }
                    if let Some(total) = sample.ram_total_bytes {
                        ram_total_sum = ram_total_sum.saturating_add(total);
                    }
                    break;
                }
            }
            if cpu_count > 0 {
                combined.cpu_percent = Some((cpu_sum / cpu_count as f64) as f32);
            }
            if ram_used_sum > 0 {
                combined.ram_used_bytes = Some(ram_used_sum);
            }
            if ram_total_sum > 0 {
                combined.ram_total_bytes = Some(ram_total_sum);
            }

            // Per-rank GPU aggregation + per-rank tab entries.
            let mut gpu_util_sum = 0.0_f64;
            let mut gpu_util_count = 0u32;
            let mut vram_alloc_sum = 0u64;
            let mut vram_total_sum = 0u64;
            for (rank, sample_opt) in map.iter().enumerate() {
                let Some(sample) = sample_opt else { continue };
                if let Some(u) = sample.gpu_util_percent {
                    gpu_util_sum += u as f64;
                    gpu_util_count += 1;
                }
                if let Some(a) = sample.vram_allocated_bytes {
                    vram_alloc_sum = vram_alloc_sum.saturating_add(a);
                }
                if let Some(t) = sample.vram_total_bytes {
                    vram_total_sum = vram_total_sum.saturating_add(t);
                }
                if let Some(mut gpu) = sample.gpus.first().cloned() {
                    let label_prefix = match self.resolve_rank(rank) {
                        Some((host, lr)) => format!("{host}:lr={lr} gr={rank} "),
                        None => format!("gr={rank} "),
                    };
                    gpu.device_index = rank as u8;
                    gpu.name = format!("{label_prefix}{}", gpu.name);
                    combined.gpus.push(gpu);
                }
            }
            if gpu_util_count > 0 {
                combined.gpu_util_percent = Some((gpu_util_sum / gpu_util_count as f64) as f32);
            }
            if vram_alloc_sum > 0 {
                combined.vram_allocated_bytes = Some(vram_alloc_sum);
            }
            if vram_total_sum > 0 {
                combined.vram_total_bytes = Some(vram_total_sum);
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
