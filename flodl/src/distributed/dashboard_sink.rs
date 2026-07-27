//! Dashboard sink: the controller-side endpoint for rank-emitted
//! dashboard wire frames.
//!
//! Post controller-active refactor, the live training dashboard runs on
//! the launcher process (not on rank 0). Ranks emit
//! `TimingMsgWire::Dashboard*` frames at startup
//! (`DashboardRegister`,
//! `DashboardSetSvg`,
//! `DashboardSetMetadata`,
//! `DashboardSetHardware`)
//! plus per-epoch resource samples piggy-backed on
//! `resources`. The
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
    /// `resources` field on `MetricsMsgWire`
    /// alongside the existing metric report.
    fn push_resource_sample(&self, rank: usize, sample: ResourceSampleWire);

    /// Aggregated [`crate::distributed::ddp_run::EpochMetrics`] for the
    /// current epoch, ready for the dashboard's main tab.
    fn push_epoch_metrics(
        &self,
        metrics: &crate::distributed::ddp_run::EpochMetrics,
    );

    /// Sub-epoch monitor records for ONE reduce window: the flat,
    /// path-keyed JSONL records of the window's record tree (root first,
    /// then each host / rank node). Emitted only when the
    /// `reports_per_epoch` cadence fires, so the rate is user-bounded and
    /// independent of the reduce rate.
    ///
    /// Complements [`Self::push_epoch_metrics`] rather than replacing it:
    /// the per-epoch feed is unchanged, this one fills the gap *between*
    /// epochs (a one-epoch LLM run has exactly one epoch point).
    /// Default = no-op, so test stubs and sinks that only render epochs
    /// need not implement it.
    fn push_window_records(&self, records: Vec<serde_json::Value>) {
        let _ = records;
    }

    /// Alert-lane records (`kind: "event"`) — rank loss, divergence drift, a
    /// dropped control broadcast. Same path-keyed stream as
    /// [`Self::push_window_records`] (so they persist and ship to log sinks
    /// unchanged), but a separate feed for the portal: an alert is an
    /// interruption, not a data point on a curve.
    ///
    /// Already collapsed and capped by
    /// [`crate::monitor::event_lane::EventLane`] upstream, so the call rate
    /// is bounded no matter how bad the run gets. Default = no-op.
    fn push_events(&self, records: Vec<serde_json::Value>) {
        let _ = records;
    }

    /// Signal end-of-training to the dashboard so the SSE `complete`
    /// event fires (browser stops the elapsed counter, switches the
    /// status dot to "done"). Called by the launcher after every rank
    /// child exits — symmetric to single-process [`Monitor::finish`]
    /// in the rank-side path. Default = no-op (test stubs need not
    /// implement). Idempotent.
    fn shutdown(&self) {}
}

/// Wall-clock milliseconds since the epoch — the axis every record shares.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Trim vendor boilerplate off a GPU model so a legend entry stays short
/// (`NVIDIA GeForce RTX 5060 Ti` → `RTX 5060 Ti`). Mirrors the dashboard's own
/// `shortGpuName`.
fn short_gpu_name(name: &str) -> String {
    let mut s = name.trim();
    for prefix in ["NVIDIA ", "GeForce "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
        }
    }
    s.to_string()
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
    /// Flat records of the most recent sub-epoch window tree (root first).
    /// Held so the portal can serve the current tree by path; streaming
    /// them live is a later slice.
    latest_window_records: Mutex<Vec<serde_json::Value>>,
    /// The run's alert feed, oldest first, bounded by
    /// [`crate::monitor::event_lane::MAX_EVENTS`]. The coordinator's lane
    /// bounds each `(class, path)`; this bounds the accumulation across the
    /// whole run so a long unhealthy run cannot grow the sink without limit.
    recent_events: Mutex<Vec<serde_json::Value>>,
    /// Optional append-only JSONL persistence for the record stream. The
    /// same records that go live also land on disk when the user opted in
    /// (`record_log_dir`); `None` keeps the stream live-only.
    record_log: Option<Arc<crate::monitor::record_log::RecordLog>>,
    /// Where to write the self-contained dashboard archive at teardown, or
    /// `None` for live-only. Sink-driven because on a cluster run this is the
    /// only `Monitor` holding the epochs and the record plane.
    dashboard_html: Option<String>,
    /// Non-core roll-up declarations for user scalars. Applied when building
    /// the epoch record tree (the only cadence carrying user scalars) and
    /// published once as the stream's `meta` record so consumers roll up the
    /// same way the controller did.
    scalar_reductions: crate::monitor::record::Reductions,
    /// Whether the `meta` record has been published yet. It is emitted lazily
    /// with the first record push rather than at construction, so it carries a
    /// timestamp from the same clock as the records it describes.
    meta_published: Mutex<bool>,
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
            latest_window_records: Mutex::new(Vec::new()),
            recent_events: Mutex::new(Vec::new()),
            record_log: None,
            dashboard_html: None,
            scalar_reductions: crate::monitor::record::Reductions::new(),
            meta_published: Mutex::new(false),
            start_time: Instant::now(),
        }
    }

    /// Attach (or clear) the append-only record-stream persistence. The
    /// launcher builds the log from `record_log_dir`; `None` leaves the
    /// stream live-only.
    pub fn with_record_log(
        mut self,
        log: Option<Arc<crate::monitor::record_log::RecordLog>>,
    ) -> Self {
        self.record_log = log;
        self
    }

    /// Write a self-contained dashboard archive to `path` at teardown, or
    /// `None` for live-only.
    pub fn with_dashboard_html(mut self, path: Option<String>) -> Self {
        self.dashboard_html = path;
        self
    }

    /// Pin the saved archive's theme. `None` leaves it to the reader's OS.
    pub fn with_dashboard_theme(self, theme: Option<String>) -> Self {
        if let Some(t) = theme {
            self.monitor.lock().unwrap().set_archive_theme(&t);
        }
        self
    }

    /// Install the user-scalar roll-up declarations from the run config.
    pub fn with_scalar_reductions(
        mut self,
        reductions: crate::monitor::record::Reductions,
    ) -> Self {
        self.scalar_reductions = reductions;
        self
    }

    /// Publish the `meta` record once, ahead of the first data record.
    ///
    /// The record plane keeps it as a standing declaration and replays it into
    /// every SSE subscriber's preamble, so a viewer can never interpret a
    /// record without first knowing how it rolls up. Emitting it lazily (rather
    /// than at construction) keeps it on the same clock as the records, and
    /// keeps it out of runs that never push one.
    /// The `meta` record, the first time only; `None` on every later call.
    ///
    /// Split from the push so the once-only semantics are observable without a
    /// bound server (`Monitor::push_records` is a no-op when none is attached,
    /// so a test asserting through it would pass vacuously).
    fn take_meta_record(&self, ts: u64) -> Option<serde_json::Value> {
        let mut done = self.meta_published.lock().unwrap();
        if *done {
            return None;
        }
        *done = true;
        Some(crate::monitor::record::meta_record(&self.scalar_reductions, ts))
    }

    /// Publish the declarations to the live record plane, once per run.
    ///
    /// Deliberately **not** written to the record log: a `meta` record carries
    /// no `path` (it describes the whole stream, not a node), and
    /// `RecordLog::append` skips pathless records, so persisting it here would
    /// be a silent no-op. The saved archive bakes the declarations explicitly
    /// instead.
    fn publish_meta_once(&self, ts: u64) {
        if let Some(meta) = self.take_meta_record(ts) {
            self.monitor.lock().unwrap().push_records(vec![meta]);
        }
    }

    /// Per-rank host names indexed by global rank, as the record tree's path
    /// shaping expects.
    fn record_hosts(&self) -> Vec<String> {
        let mut hosts = vec![String::new(); self.cluster.world_size()];
        for worker in &self.cluster.workers {
            for &r in &worker.ranks {
                if let Some(slot) = hosts.get_mut(r) {
                    *slot = worker.host.clone();
                }
            }
        }
        hosts
    }

    /// Per-rank resource + legend extras for the epoch record tree, read off
    /// the samples the ranks piggy-back on their metrics reports.
    ///
    /// These are sampled once per epoch, which is exactly this record's
    /// cadence — so the values are fresh here, where smearing them across every
    /// sub-epoch window report would have invented data between samples.
    fn record_extras(&self) -> Vec<crate::monitor::record::RankExtras> {
        let map = self.per_rank_resources.lock().unwrap();
        map.iter()
            .map(|s| {
                let Some(sample) = s else {
                    return crate::monitor::record::RankExtras::default();
                };
                crate::monitor::record::RankExtras {
                    res: crate::monitor::record::Res {
                        // absent≠zero: an unsampled field stays None and is
                        // excluded from the rollup, never averaged in as 0.
                        gpu_util: sample.gpu_util_percent.map(|v| v as f64),
                        vram_alloc: sample.vram_allocated_bytes.map(|v| v as f64),
                        vram_total: sample.vram_total_bytes.map(|v| v as f64),
                    },
                    label: sample.gpus.first().map(|g| short_gpu_name(&g.name)),
                }
            })
            .collect()
    }

    /// Build and ship the epoch-boundary record tree to the live record plane
    /// and the persisted log — the same two sinks the window reports use.
    fn push_epoch_records(&self, metrics: &crate::distributed::ddp_run::EpochMetrics) {
        let hosts = self.record_hosts();
        let extras = self.record_extras();
        // The epoch cadence is the ONLY one carrying user scalars
        // (`WindowRankStat` has no scalars field), so this is the one place a
        // non-core reduction can change an answer. Undeclared keys still fall
        // through to `Mean`.
        let tree = crate::monitor::record::NodeRecord::from_epoch_metrics(
            metrics,
            Some(&hosts),
            &self.scalar_reductions,
            &extras,
        );
        let ts = now_ms();
        self.publish_meta_once(ts);
        // No `tick`: an epoch record closes an epoch, it does not belong to a
        // sub-epoch window. `ts` is the axis both cadences share.
        let records = tree.flat_records(ts, None, Some(metrics.epoch));
        self.monitor.lock().unwrap().push_records(records.clone());
        if let Some(log) = &self.record_log {
            log.append(&records);
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
                if crate::monitor::dashboard_bind_is_loopback() {
                    // Loopback default: the URL below only works ON the
                    // controller. Point remote viewers at an SSH tunnel rather
                    // than implying network reachability the bind doesn't have.
                    eprintln!(
                        "cluster dashboard: http://localhost:{port} (bound on the \
                         controller, loopback). View from another machine via \
                         `ssh -L {port}:localhost:{port} {}`, or set \
                         FLODL_DASHBOARD_BIND=0.0.0.0 to expose it (no auth).",
                        self.controller_host,
                    );
                } else {
                    eprintln!(
                        "cluster dashboard: http://{}:{}",
                        self.controller_host, port,
                    );
                }
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
        // Flush the record stream before the dashboard goes down, so the
        // final window's records are on disk for the post-run explorer.
        if let Some(log) = &self.record_log {
            log.flush();
        }
        let mut mon = self.monitor.lock().unwrap();
        // Server down FIRST: it drains the record-push channel, so the archive
        // below sees the whole run including the final window rather than
        // whatever happened to be applied already.
        mon.shutdown_dashboard_server();
        if let Some(path) = &self.dashboard_html {
            mon.write_archive_now(path);
        }
    }

    fn push_window_records(&self, records: Vec<serde_json::Value>) {
        // One verbose line for the root node — the sub-epoch curve as it
        // happens, and what rig validation greps for. Full records are
        // retained for the portal.
        if crate::log::enabled(crate::log::Verbosity::Verbose)
            && let Some(root) = records.first()
        {
            let m = &root["metrics"];
            crate::verbose!(
                "  ddp: window report epoch={} tick={} loss={} tput={} work={}",
                root["epoch"],
                root["tick"],
                m["loss"],
                m["throughput"],
                root["work"],
            );
        }
        // Window records precede the first epoch record, so the declaration
        // has to be published from here too or a viewer that connects during
        // epoch 0 interprets records without it. Fall back to the local clock
        // rather than skipping when a record carries no `ts` — a missing
        // timestamp must not silently cost the whole stream its declarations.
        let ts = records
            .first()
            .and_then(|r| r["ts"].as_u64())
            .unwrap_or_else(now_ms);
        self.publish_meta_once(ts);
        // One producer, three sinks: the same records go to the live record
        // plane (path-scoped SSE), to disk, and to the latest-window slot.
        self.monitor.lock().unwrap().push_records(records.clone());
        if let Some(log) = &self.record_log {
            log.append(&records);
        }
        *self.latest_window_records.lock().unwrap() = records;
    }

    fn push_events(&self, records: Vec<serde_json::Value>) {
        // No log line here — the coordinator already printed one per alert
        // (it does so with or without a sink attached, so a headless cluster
        // run stays as loud as a dashboard one).
        //
        // Same stream as the node records, live and on disk alike: the record
        // plane scopes by `path` and the log routes by `path`, so an alert
        // reaches its origin node's viewer and lands in its origin node's
        // file, interleaved with that node's metrics.
        self.monitor.lock().unwrap().push_records(records.clone());
        if let Some(log) = &self.record_log {
            log.append(&records);
        }
        let mut feed = self.recent_events.lock().unwrap();
        feed.extend(records);
        let overflow = feed.len().saturating_sub(crate::monitor::event_lane::MAX_EVENTS);
        if overflow > 0 {
            feed.drain(0..overflow);
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

        // The epoch-boundary half of the record stream. This is the ONLY feed
        // carrying user scalars and the resource sample, so without it a level
        // view has framework metrics and nothing else. Same tree, same paths as
        // the sub-epoch window reports — they interleave per level.
        self.push_epoch_records(metrics);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::ClusterBuilder;
    use crate::monitor::record_log::{DEFAULT_MAX_LOG_BYTES, RecordLog};
    use serde_json::json;

    /// Unique temp dir per test, removed on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir()
                .join(format!("flodl-sinklog-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn cluster() -> Arc<FullCluster> {
        Arc::new(
            ClusterBuilder::new()
                .controller("127.0.0.1")
                .port(29500)
                .done()
                .host("exa")
                .ranks([0])
                .devices([0])
                .nccl_socket_ifname("lo")
                .path("/opt/flodl")
                .done()
                .build()
                .expect("test cluster builds"),
        )
    }

    fn window(tick: u64) -> Vec<serde_json::Value> {
        vec![
            json!({"v":1,"kind":"node","path":"root","tick":tick,"metrics":{"loss":0.5}}),
            json!({"v":1,"kind":"node","path":"root/rank0","tick":tick,"metrics":{"loss":0.5}}),
        ]
    }

    /// With a record log attached, pushed window records land on disk in
    /// the node tree — the live stream and the history share one producer.
    #[test]
    fn window_records_persist_to_the_node_tree() {
        let d = TempDir::new("persist");
        let log = Arc::new(RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES));
        let sink = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1)
            .with_record_log(Some(Arc::clone(&log)));

        sink.push_window_records(window(1));
        sink.push_window_records(window(2));
        sink.shutdown(); // flushes

        assert!(d.0.join("root.log").is_file());
        assert!(d.0.join("root/rank0.log").is_file());
        // Tail-read resume sees both windows, newest last.
        let tail = log.tail("root", 10);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[1]["tick"], 2);
    }

    /// A declared roll-up must actually change the cross-rank answer, and the
    /// undeclared default must stay `Mean`.
    ///
    /// This is the defect the declaration exists to fix: both `build_tree` call
    /// sites used to pass an empty map, so a **count** was averaged across
    /// ranks while the portal's legend asserted `(mean)` — wrong, and stated
    /// rather than merely implied.
    #[test]
    fn a_declared_reduction_changes_the_rollup_and_mean_stays_the_default() {
        use crate::monitor::record::{Leaf, Reduction, Reductions, Res, build_tree};

        let leaves: Vec<Leaf> = [("rank0", 10.0), ("rank1", 30.0)]
            .iter()
            .map(|(name, v)| Leaf {
                path: vec![name.to_string()],
                work: 1.0,
                metrics: [
                    ("tokens_seen".to_string(), *v),
                    ("undeclared".to_string(), *v),
                ]
                .into_iter()
                .collect(),
                res: Res::default(),
                device: None,
                alive: true,
                label: None,
            })
            .collect();

        let mut declared = Reductions::new();
        declared.insert("tokens_seen".to_string(), Reduction::Sum);

        let with = build_tree(&leaves, &declared);
        let without = build_tree(&leaves, &Reductions::new());

        // Declared Sum: the count adds up.
        assert_eq!(with.metrics.get("tokens_seen"), Some(&40.0));
        // Undeclared: still the work-weighted mean, in the same tree.
        assert_eq!(with.metrics.get("undeclared"), Some(&20.0));
        // And without the declaration the count is silently averaged — the bug.
        assert_eq!(without.metrics.get("tokens_seen"), Some(&20.0));
    }

    /// The declarations must reach consumers, which means a `meta` record ahead
    /// of any data record. Nothing emitted one before this change, so
    /// `reductionOf` in the page always fell through to `'mean'`.
    #[test]
    fn the_sink_publishes_the_reduction_declarations_exactly_once() {
        use crate::monitor::record::Reduction;

        let mut declared = crate::monitor::record::Reductions::new();
        declared.insert("tokens_seen".to_string(), Reduction::Sum);
        let sink = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1)
            .with_scalar_reductions(declared);

        // First record push claims the declaration...
        sink.push_window_records(window(1));
        // ...and it is not re-emitted on any later push, at either cadence.
        sink.push_window_records(window(2));
        assert!(
            sink.take_meta_record(1).is_none(),
            "meta must be published once per run, not once per push",
        );

        // The declaration itself, as a consumer receives it.
        let fresh = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1)
            .with_scalar_reductions({
                let mut r = crate::monitor::record::Reductions::new();
                r.insert("tokens_seen".to_string(), Reduction::Sum);
                r
            });
        let meta = fresh.take_meta_record(7).expect("first call yields the meta");
        assert_eq!(meta["kind"], "meta");
        assert_eq!(meta["ts"], 7);
        assert_eq!(meta["reductions"]["tokens_seen"], "sum");
    }

    /// A `meta` record carries no `path`, and `RecordLog::append` skips
    /// pathless records — so it must not be handed to the log at all rather
    /// than being appended and silently dropped.
    #[test]
    fn meta_is_not_written_to_the_record_log() {
        let d = TempDir::new("meta-nolog");
        let log = Arc::new(RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES));
        let sink = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1)
            .with_record_log(Some(Arc::clone(&log)))
            .with_scalar_reductions(crate::monitor::record::Reductions::new());

        sink.push_window_records(window(1));
        sink.shutdown();

        // Only the node record for this path; no stray pathless line.
        let tail = log.tail("root", 10);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0]["kind"], "node");
    }

    /// Alerts share the node tree with metrics: an event lands in its origin
    /// node's file, interleaved with that node's `node` records, because the
    /// log routes on `path` alone.
    #[test]
    fn events_persist_beside_the_metrics_of_their_origin_node() {
        let d = TempDir::new("events");
        let log = Arc::new(RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES));
        let sink = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1)
            .with_record_log(Some(Arc::clone(&log)));

        sink.push_window_records(window(1));
        sink.push_events(vec![json!({
            "v": 1, "ts": 7, "sev": "critical", "path": "root/rank0",
            "kind": "event", "class": "rank_lost", "detail": "died", "count": 1,
        })]);
        sink.shutdown();

        let tail = log.tail("root/rank0", 10);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0]["kind"], "node");
        assert_eq!(tail[1]["kind"], "event");
        assert_eq!(tail[1]["class"], "rank_lost");
        // The root node's own log is untouched by a rank-scoped alert.
        assert_eq!(log.tail("root", 10).len(), 1);
    }

    /// The sink's alert feed is bounded: a long unhealthy run cannot grow it
    /// past the lane's cap, and the newest alerts are the ones kept.
    #[test]
    fn the_event_feed_is_bounded_newest_wins() {
        let sink = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1);
        let n = crate::monitor::event_lane::MAX_EVENTS;
        for i in 0..(n + 10) {
            sink.push_events(vec![json!({
                "v": 1, "ts": i, "path": "root", "kind": "event",
                "class": "drift", "detail": "d", "count": 1,
            })]);
        }
        let feed = sink.recent_events.lock().unwrap();
        assert_eq!(feed.len(), n);
        assert_eq!(feed[n - 1]["ts"], n + 9);
    }

    /// Two hosts, two ranks — so the epoch tree gets a host tier and the
    /// per-host paths are exercised.
    fn cluster2() -> Arc<FullCluster> {
        Arc::new(
            ClusterBuilder::new()
                .controller("127.0.0.1")
                .port(29501)
                .done()
                .host("exa")
                .ranks([0])
                .devices([0])
                .nccl_socket_ifname("lo")
                .path("/opt/flodl")
                .done()
                .host("pascal")
                .ranks([1])
                .devices([0])
                .nccl_socket_ifname("lo")
                .path("/opt/flodl")
                .done()
                .build()
                .expect("2-host test cluster builds"),
        )
    }

    fn epoch_metrics2() -> crate::distributed::ddp_run::EpochMetrics {
        let mut scalars = std::collections::HashMap::new();
        scalars.insert("eval_acc".to_string(), 0.87);
        crate::distributed::ddp_run::EpochMetrics {
            epoch: 2,
            scalars,
            per_rank: vec![Default::default(), Default::default()],
            avg_loss: 0.25,
            epoch_ms: 500.0,
            per_rank_throughput: vec![12.0, 4.0],
            per_rank_batch_share: vec![0.75, 0.25],
            per_rank_share_complete_ms: vec![480.0, 490.0],
            per_rank_compute_only_ms: vec![400.0, 410.0],
            per_rank_data_starve_ms: vec![10.0, 60.0],
            device_indices: vec![0, 0],
        }
    }

    fn sample_with_gpu(util: f32, alloc: u64, name: &str) -> ResourceSampleWire {
        let mut s = ResourceSample {
            gpu_util_percent: Some(util),
            vram_allocated_bytes: Some(alloc),
            vram_total_bytes: Some(8_000_000_000),
            ..Default::default()
        };
        s.gpus.push(crate::monitor::GpuSnapshot { name: name.to_string(), ..Default::default() });
        s.into()
    }

    /// The per-epoch feed is the ONLY source of user scalars and the resource
    /// sample, so it must also become records — otherwise a level view has
    /// framework metrics and nothing else.
    #[test]
    fn epoch_metrics_also_become_records_on_both_sinks() {
        let d = TempDir::new("epochrec");
        let log = Arc::new(RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES));
        let sink = ClusterDashboardSink::new(cluster2(), "exa".to_string(), 4)
            .with_record_log(Some(Arc::clone(&log)));

        sink.push_resource_sample(0, sample_with_gpu(90.0, 5_000_000_000, "NVIDIA GeForce RTX 5060 Ti"));
        sink.push_resource_sample(1, sample_with_gpu(50.0, 1_000_000_000, "NVIDIA GeForce GTX 1060"));
        sink.push_epoch_metrics(&epoch_metrics2());
        sink.shutdown();

        // Same path shaping as the window feed: two hosts => a host tier.
        let root = log.tail("root", 10);
        assert_eq!(root.len(), 1, "one epoch record at root");
        let r = &root[0];
        assert_eq!(r["epoch_complete"], true, "marked as an epoch boundary");
        assert!(r.get("tick").is_none(), "an epoch record has no window index");
        assert_eq!(r["epoch"], 2);
        // avg_loss injected (EpochMetrics has no per-rank loss at all) and the
        // root-only user scalar carried.
        assert_eq!(r["metrics"]["loss"], 0.25);
        assert_eq!(r["metrics"]["eval_acc"], 0.87);
        // throughput sums; gpu_util is the work-weighted mean 90*.75+50*.25=80.
        assert_eq!(r["metrics"]["throughput"], 16.0);
        assert!((r["res"]["gpu_util"].as_f64().unwrap() - 80.0).abs() < 1e-9);
        assert_eq!(r["res"]["vram_alloc"], 6_000_000_000.0);

        // The leaf carries its own device + a short legend label.
        let leaf = log.tail("root/exa/rank0", 10);
        assert_eq!(leaf.len(), 1);
        assert_eq!(leaf[0]["label"], "RTX 5060 Ti", "vendor prefixes trimmed");
        assert_eq!(leaf[0]["res"]["gpu_util"], 90.0);
        assert!(log.tail("root/pascal/rank1", 10)[0]["label"] == "GTX 1060");
    }

    /// An unsampled rank must contribute no `res` at all rather than zeros —
    /// absent≠zero survives the sink hop.
    #[test]
    fn an_unsampled_rank_contributes_no_res() {
        let d = TempDir::new("nores");
        let log = Arc::new(RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES));
        let sink = ClusterDashboardSink::new(cluster2(), "exa".to_string(), 4)
            .with_record_log(Some(Arc::clone(&log)));
        sink.push_epoch_metrics(&epoch_metrics2());
        sink.shutdown();
        let r = &log.tail("root", 10)[0];
        assert!(r.get("res").is_none(), "no resource fields anywhere: {r}");
        assert!(log.tail("root/exa/rank0", 10)[0].get("label").is_none());
    }

    #[test]
    fn short_gpu_name_trims_vendor_prefixes() {
        assert_eq!(short_gpu_name("NVIDIA GeForce RTX 5060 Ti"), "RTX 5060 Ti");
        assert_eq!(short_gpu_name("NVIDIA GP106"), "GP106");
        assert_eq!(short_gpu_name("  Radeon RX 7900  "), "Radeon RX 7900");
    }

    /// Without a record log the sink is live-only — no files, no panic.
    #[test]
    fn no_record_log_means_live_only() {
        let d = TempDir::new("liveonly");
        let sink = ClusterDashboardSink::new(cluster(), "exa".to_string(), 1);
        sink.push_window_records(window(1));
        sink.shutdown();
        assert!(!d.0.join("root.log").exists());
        // The latest window is still held in memory for the portal.
        assert_eq!(sink.latest_window_records.lock().unwrap().len(), 2);
    }
}
