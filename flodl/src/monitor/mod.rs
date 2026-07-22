//! Training monitor with human-readable ETA, resource tracking, and live dashboard.
//!
//! The monitor prints a one-line summary per epoch and optionally serves a live
//! web dashboard with charts, resource graphs, and metric logs.
//!
//! ```ignore
//! use flodl::Monitor;
//!
//! let mut monitor = Monitor::new(num_epochs);
//! monitor.serve(3000)?;        // live dashboard at http://localhost:3000
//! monitor.watch(&model);       // graph SVG in dashboard
//!
//! for epoch in 0..num_epochs {
//!     let t = std::time::Instant::now();
//!     // ... training steps ...
//!     model.record_scalar("loss", loss_val);
//!     model.record_scalar("lr", current_lr);
//!
//!     model.flush(&[]);
//!     monitor.log(epoch, t.elapsed(), &model);
//! }
//!
//! monitor.finish();
//! ```

pub mod format;
pub mod resources;
pub mod timeline;
mod server;
pub(crate) use server::dashboard_bind_is_loopback;

use std::fmt::Write;
use std::time::{Duration, Instant};

use crate::graph::Graph;

pub use format::{format_eta, format_bytes, format_metric};
pub use resources::{ResourceSample, ResourceSampler, GpuSnapshot};
pub use timeline::{Timeline, TimelineBroadcast, TimelineEvent, EventKind, TimelineSample, GpuTimelineSample, TimelineSummary};

/// DDP metrics for a single GPU (throughput, batch split, shard size).
#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    /// CUDA device index.
    pub device_index: u8,
    /// EMA throughput in samples/ms.
    pub throughput: f64,
    /// Fraction of the batch assigned to this device (0.0-1.0).
    pub chunk_ratio: f64,
    /// Number of samples in this device's shard last batch.
    pub shard_size: i64,
}

/// Recorded snapshot of a single training epoch: timing, metrics, and resource usage.
#[derive(Clone)]
pub struct EpochRecord {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Wall-clock duration of this epoch in seconds.
    pub duration_secs: f64,
    /// Named metric values recorded during this epoch (e.g., `("loss", 0.42)`).
    pub metrics: Vec<(String, f64)>,
    /// System resource snapshot taken at the end of this epoch.
    pub resources: ResourceSample,
    /// Per-GPU DDP metrics (empty for single-GPU training).
    pub gpu_metrics: Vec<GpuMetrics>,
}

/// Trait for values accepted by [`Monitor::log()`] as the `metrics` argument.
///
/// This lets `log` accept plain `&[("loss", val)]` slices, a `&Graph` reference
/// (which pulls the latest observation epoch), or a `(&Graph, &[...])` tuple
/// that appends extra metrics to the graph's own.
///
/// For multi-GPU / cluster training, log against
/// [`crate::distributed::EpochMetrics`] — that impl carries the
/// per-rank view aggregated by the coordinator and surfaces it
/// through [`Self::gpu_metrics`]. Graph-backed sources are
/// single-device by construction so their `gpu_metrics` returns
/// empty.
pub trait Metrics {
    /// Convert into owned `(name, value)` pairs for recording.
    fn into_metrics(self) -> Vec<(String, f64)>;
    /// Per-rank [`GpuMetrics`] for cluster / multi-GPU training.
    /// Default: empty. The `&EpochMetrics` impl populates this from
    /// the coordinator's aggregated per-rank view; Graph-backed
    /// sources stay empty because a Graph reference only ever names
    /// a single device.
    fn gpu_metrics(&self) -> Vec<GpuMetrics> { Vec::new() }
}

/// Plain metric slice: `&[("loss", val)]`.
impl<'a> Metrics for &'a [(&'a str, f64)] {
    fn into_metrics(self) -> Vec<(String, f64)> {
        self.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
}

/// Plain metric array literal: `&[("loss", val), ("lr", lr)]`.
impl<const N: usize> Metrics for &[(&str, f64); N] {
    fn into_metrics(self) -> Vec<(String, f64)> {
        self.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
}

/// Per-GPU metrics surfaced from a `&Graph` source. In cluster mode
/// the framework populates the graph's aggregated-metrics slot via a
/// coord broadcast (see
/// [`crate::distributed::wire::ControlMsgWire::EpochAggregated`]);
/// this function reads that slot to produce the per-GPU tabs the
/// dashboard renders. Returns empty in single-GPU runs (no per-rank
/// dimension) and pre-first-aggregation cluster runs.
fn graph_gpu_metrics(graph: &Graph) -> Vec<GpuMetrics> {
    graph
        .aggregated_gpu_tabs()
        .into_iter()
        .map(|(device_index, throughput, chunk_ratio)| GpuMetrics {
            device_index,
            throughput,
            chunk_ratio,
            shard_size: 0,
        })
        .collect()
}

/// Graph only: `&model` -- reads latest epoch history.
impl Metrics for &Graph {
    fn into_metrics(self) -> Vec<(String, f64)> {
        self.latest_metrics()
    }
    fn gpu_metrics(&self) -> Vec<GpuMetrics> {
        graph_gpu_metrics(self)
    }
}

/// Graph + extras tuple: `(&model, &[("lr", lr)])`.
impl<'a> Metrics for (&'a Graph, &'a [(&'a str, f64)]) {
    fn into_metrics(self) -> Vec<(String, f64)> {
        let (graph, extra) = self;
        let mut m = graph.latest_metrics();
        m.extend(extra.iter().map(|(k, v)| (k.to_string(), *v)));
        m
    }
    fn gpu_metrics(&self) -> Vec<GpuMetrics> {
        graph_gpu_metrics(self.0)
    }
}

/// Graph + extras array literal: `(&model, &[("lr", lr)])`.
impl<'a, const N: usize> Metrics for (&'a Graph, &'a [(&'a str, f64); N]) {
    fn into_metrics(self) -> Vec<(String, f64)> {
        let (graph, extra) = self;
        let mut m = graph.latest_metrics();
        m.extend(extra.iter().map(|(k, v)| (k.to_string(), *v)));
        m
    }
    fn gpu_metrics(&self) -> Vec<GpuMetrics> {
        graph_gpu_metrics(self.0)
    }
}

/// DDP builder epoch metrics: feeds `record_scalar` data and per-GPU
/// throughput/batch share into the Monitor.
///
/// ```ignore
/// while let Some(m) = handle.next_metrics() {
///     monitor.log(m.epoch, Duration::from_millis(m.epoch_ms as u64), &m);
/// }
/// ```
impl Metrics for &crate::distributed::EpochMetrics {
    fn into_metrics(self) -> Vec<(String, f64)> {
        let mut out = Vec::with_capacity(self.scalars.len() + 1);
        out.push(("loss".to_string(), self.avg_loss));
        // Deterministic order: sort by key
        let mut keys: Vec<&String> = self.scalars.keys().collect();
        keys.sort();
        for k in keys {
            out.push((k.clone(), self.scalars[k]));
        }
        out
    }

    fn gpu_metrics(&self) -> Vec<GpuMetrics> {
        self.device_indices.iter().enumerate().map(|(i, &dev)| {
            GpuMetrics {
                device_index: dev,
                throughput: self.per_rank_throughput.get(i).copied().unwrap_or(0.0),
                chunk_ratio: self.per_rank_batch_share.get(i).copied().unwrap_or(0.0),
                shard_size: 0, // not tracked per-epoch in builder mode
            }
        }).collect()
    }
}

/// Training monitor with ETA, resource tracking, and optional live dashboard.
pub struct Monitor {
    total_epochs: usize,
    epochs: Vec<EpochRecord>,
    start_time: Instant,
    sampler: ResourceSampler,
    server: Option<server::DashboardServer>,
    save_html: Option<String>,
    svg_snapshot: Option<String>,
    metadata: Option<serde_json::Value>,
    graph_label: Option<String>,
    graph_hash: Option<String>,
    hardware: String,
    /// `true` when this rank should serve the dashboard + persist
    /// log entries (single-GPU runs, cluster rank 0). `false` on
    /// non-primary cluster ranks — their `serve`, `log`, and
    /// `save_html` calls no-op so the user can keep one `Monitor`
    /// construction at the top of their training loop, running the
    /// same code on every rank, and get exactly one dashboard
    /// rendering the global cross-rank view (via the user's
    /// `Graph::latest_metrics()` reading from the coord-broadcast
    /// aggregated slot — see
    /// [`crate::distributed::wire::ControlMsgWire::EpochAggregated`]).
    is_primary: bool,
    /// Suppress the `"training complete in …"` terminal summary
    /// emitted by [`Self::finish`]. Wrappers (e.g. ddp-bench's harness
    /// owns a richer `done: loss=…, syncs=…, idle=…` summary) opt
    /// into this so the terminal doesn't show two near-identical
    /// end-of-run lines. HTML archive + dashboard side effects are
    /// unaffected. Default `false`.
    silent_summary: bool,
}

impl Monitor {
    /// Create a new monitor for `total_epochs` epochs.
    ///
    /// In cluster mode (process-per-rank), only rank 0 fully
    /// activates the monitor (dashboard server, epoch records, HTML
    /// export); other ranks construct a no-op monitor so the
    /// user-facing training-loop code stays identical across single-
    /// GPU and cluster runs. The user calls `Monitor::new` /
    /// `monitor.serve` / `monitor.log` exactly once, and the
    /// framework routes the visible side effects to the primary
    /// rank only.
    ///
    /// Never initializes the CUDA runtime: GPU identity comes from
    /// nvidia-smi and live metrics from NVML, so constructing a
    /// monitor before [`Trainer::run`](crate::distributed::Trainer::run)
    /// is safe and does not violate the no-CUDA-before-`Trainer::run`
    /// rule (this same code runs in the fan-out launcher process).
    pub fn new(total_epochs: usize) -> Self {
        let is_primary = Self::detect_is_primary();
        let hardware = crate::tensor::hardware_summary();
        // Stash the rank's hardware string for the cluster_worker to
        // emit at startup. No-op cost when not in cluster mode (the
        // launcher's dashboard sink never receives the frame, so the
        // string is just held in a static Mutex until process exit).
        if Self::in_cluster_mode() {
            crate::distributed::cluster_dashboard_emit::stash_hardware(
                hardware.clone(),
            );
        }
        Self {
            total_epochs,
            epochs: Vec::with_capacity(total_epochs),
            start_time: Instant::now(),
            sampler: ResourceSampler::new(),
            server: None,
            save_html: None,
            svg_snapshot: None,
            metadata: None,
            graph_label: None,
            graph_hash: None,
            hardware,
            is_primary,
            silent_summary: false,
        }
    }

    /// `true` when this process is a cluster rank child. Cluster
    /// ranks defer the dashboard's HTTP bind to the controller and
    /// instead stash their intent into
    /// [`crate::distributed::cluster_dashboard_emit`] for the
    /// cluster_worker to forward over the wire.
    ///
    /// Distinct from [`Self::in_launcher_process`]: the launcher
    /// process has `FLODL_INTERNAL_FULL_CLUSTER_JSON` (full topology) set but
    /// NOT `FLODL_INTERNAL_CLUSTER_JSON` (per-rank envelope); ranks have the
    /// per-rank envelope. Single-process / `Ddp::wrap`-thread has
    /// neither.
    fn in_cluster_mode() -> bool {
        matches!(
            crate::distributed::LocalCluster::from_env(),
            Ok(Some(_))
        )
    }

    /// `true` when this process is the launcher trampoline. The
    /// launcher hosts the dashboard via
    /// [`crate::distributed::ClusterDashboardSink`] — the user's
    /// Monitor on this process should NOT bind locally (or the sink
    /// would fight it for the port). Read together with
    /// [`Self::in_cluster_mode`]: launcher and rank are mutually
    /// exclusive (per the `(FLODL_INTERNAL_FULL_CLUSTER_JSON, FLODL_INTERNAL_CLUSTER_JSON)`
    /// table in `launcher.rs`).
    fn in_launcher_process() -> bool {
        std::env::var_os(
            crate::distributed::launcher::ENV_FULL_CLUSTER_JSON,
        )
        .is_some()
    }

    /// Suppress the terminal `"training complete in …"` line emitted
    /// from [`Self::finish`]. Useful for wrappers that own a richer
    /// end-of-run line (e.g. ddp-bench's harness `done:` line). HTML
    /// archive saves + dashboard pushes are unaffected.
    pub fn silent_summary(&mut self) -> &mut Self {
        self.silent_summary = true;
        self
    }

    /// Decide whether this process's Monitor records history /
    /// prints the per-epoch terminal line.
    ///
    /// - Single-process / `Ddp::wrap`-thread / launcher: `true`. The
    ///   Monitor is fully active.
    /// - Cluster rank child: `true` for rank 0, `false` for others.
    ///   The launcher hosts the dashboard (per controller-active
    ///   refactor), so the rank's Monitor server-side is always
    ///   inert in cluster mode; the `is_primary` gate now exists only
    ///   to deduplicate per-rank terminal output. Rank 0 prints the
    ///   one-line summary; other ranks no-op so the `[host:rN]`-
    ///   prefixed forwarder shows one line per epoch from the cohort,
    ///   not N.
    fn detect_is_primary() -> bool {
        match crate::distributed::LocalCluster::from_env() {
            Ok(Some(cluster)) => match cluster.my_rank() {
                Ok((rank, _)) => rank == 0,
                Err(_) => true,
            },
            _ => true,
        }
    }

    /// `true` when this monitor will serve the dashboard / persist
    /// records. Test-friendly accessor.
    pub fn is_primary(&self) -> bool {
        self.is_primary
    }

    /// Start a live dashboard HTTP server on the given port.
    ///
    /// The dashboard is accessible at `http://localhost:{port}` and updates
    /// in real time as training progresses.
    pub fn serve(&mut self, port: u16) -> std::io::Result<()> {
        if Self::in_launcher_process() {
            // Launcher trampoline: the user's Monitor.serve here runs
            // before `Trainer::run` dispatches into
            // `run_launcher_with_config`. The launcher's
            // `ClusterDashboardSink` (constructed inside that call)
            // owns the dashboard server; the user's Monitor must NOT
            // also bind or the two will race for the port. Skip
            // silently — the sink prints `cluster dashboard: …` once
            // a rank's DashboardRegister arrives.
            return Ok(());
        }
        if Self::in_cluster_mode() {
            // Cluster-rank mode: the launcher hosts the dashboard at
            // `controllerHost:port`. Stash the port; the cluster_worker
            // emits a `DashboardRegister` frame at startup so the
            // launcher's sink binds the server. Don't bind locally —
            // the rank's process can crash without taking down the
            // dashboard with it.
            crate::distributed::cluster_dashboard_emit::stash_port(port);
            return Ok(());
        }
        if !self.is_primary {
            // Belt-and-braces: in non-cluster single-process layouts
            // is_primary is always true, so this is unreachable; kept
            // as a guard against future Monitor wiring that flips it.
            return Ok(());
        }
        self.bind_dashboard_locally(port)?;
        crate::msg!("  dashboard: http://localhost:{}", port);
        Ok(())
    }

    /// Force a local HTTP bind on `port`, bypassing the cluster /
    /// launcher gating in [`Self::serve`]. Used by the launcher-side
    /// [`crate::distributed::ClusterDashboardSink`] whose Monitor
    /// lives inside the launcher trampoline process (where
    /// `Self::serve` would otherwise no-op). Does not print the
    /// `dashboard: …` line — the sink prints `cluster dashboard: …`
    /// with the controller-host URL.
    pub(crate) fn serve_local_unconditional(
        &mut self,
        port: u16,
    ) -> std::io::Result<()> {
        self.bind_dashboard_locally(port)
    }

    /// Shut the dashboard's HTTP server down + emit the SSE `complete`
    /// event so connected browsers stop the elapsed counter and flip
    /// to the "done" status. Symmetric to what
    /// [`Self::finish`] does at end-of-training in the rank-side path;
    /// used by [`crate::distributed::DashboardSink::shutdown`]
    /// when the launcher tears down after every rank child has exited.
    /// Idempotent — calling on a never-bound Monitor is a no-op.
    pub(crate) fn shutdown_dashboard_server(&mut self) {
        if let Some(ref mut srv) = self.server {
            srv.shutdown();
        }
    }

    /// Shared bind path for [`Self::serve`] and
    /// [`Self::serve_local_unconditional`]. Performs the TCP bind +
    /// initial header / gpu_init injection; the calling surface
    /// decides whether to print a URL line.
    fn bind_dashboard_locally(&mut self, port: u16) -> std::io::Result<()> {
        let srv = server::DashboardServer::start(port)?;
        srv.set_hardware(self.hardware.clone());

        // Sample GPU hardware for immediate tab init (before epoch 1).
        // Skip in the launcher process: the launcher host doesn't
        // necessarily own a GPU, and even if it does the per-rank
        // tabs come from rank-emitted Dashboard frames anyway. The
        // sink's first push_resource_sample populates the tabs.
        if !Self::in_launcher_process() {
            let init_sample = self.sampler.sample();
            if init_sample.gpus.len() >= 2 {
                srv.set_gpu_init(Self::gpu_init_json(&init_sample.gpus));
            }
        }

        self.server = Some(srv);
        Ok(())
    }

    /// Save a self-contained HTML dashboard archive when `finish()` is called.
    ///
    /// The archive contains all epoch data, resource metrics, and the graph
    /// SVG baked into a single file — no server needed, just open it in a browser.
    ///
    /// This is the Monitor's export — for a simpler static chart from the
    /// graph's observation system, see [`Graph::plot_html()`](crate::graph::Graph::plot_html).
    ///
    /// ```ignore
    /// monitor.save_html("training_report.html");
    /// ```
    pub fn save_html(&mut self, path: &str) {
        self.save_html = Some(path.to_string());
    }

    /// Attach arbitrary JSON metadata (hyperparameters, config, etc.)
    /// that will be included in the live dashboard and HTML archive.
    pub fn set_metadata(&mut self, meta: serde_json::Value) {
        if Self::in_cluster_mode() {
            crate::distributed::cluster_dashboard_emit::stash_metadata(
                meta.to_string(),
            );
        }
        if let Some(ref srv) = self.server {
            srv.set_metadata(meta.to_string());
        }
        self.metadata = Some(meta);
    }

    /// Display the graph architecture in the dashboard (and HTML archive).
    ///
    /// Generates an SVG from the graph. Requires Graphviz (`dot`) to be
    /// installed. Silently does nothing if SVG generation fails.
    pub fn watch(&mut self, graph: &Graph) {
        self.capture_graph_identity(graph);
        if let Ok(svg_bytes) = graph.svg(None) {
            self.set_svg(&String::from_utf8_lossy(&svg_bytes));
        }
    }

    /// Display the graph architecture with profiling heat map.
    ///
    /// Uses the most recent profiling data from the graph. Call
    /// `graph.enable_profiling()` and run at least one forward pass before
    /// calling this, otherwise falls back to the plain graph SVG.
    pub fn watch_profiled(&mut self, graph: &Graph) {
        self.capture_graph_identity(graph);
        // Try profiled SVG first, fall back to plain
        if let Ok(svg_bytes) = graph.svg_with_profile(None) {
            self.set_svg(&String::from_utf8_lossy(&svg_bytes));
        } else if let Ok(svg_bytes) = graph.svg(None) {
            self.set_svg(&String::from_utf8_lossy(&svg_bytes));
        }
    }

    /// Set a raw SVG string for display in the dashboard and HTML archive.
    pub fn set_svg(&mut self, svg: &str) {
        self.svg_snapshot = Some(svg.to_string());
        if Self::in_cluster_mode() {
            crate::distributed::cluster_dashboard_emit::stash_svg(
                svg.to_string(),
                self.graph_label.clone(),
                self.graph_hash.clone(),
            );
        }
        if let Some(ref srv) = self.server {
            srv.set_svg(svg.to_string());
        }
    }

    /// Replace the hardware-summary string displayed in the dashboard
    /// header. The default value is captured at [`Monitor::new`] from
    /// the running process via [`crate::tensor::hardware_summary`]; the
    /// cluster path uses this setter to install a multi-rank summary
    /// composed from per-rank hardware strings the launcher receives
    /// over the wire.
    pub fn set_hardware(&mut self, hardware: impl Into<String>) {
        self.hardware = hardware.into();
        if let Some(ref srv) = self.server {
            srv.set_hardware(self.hardware.clone());
        }
    }

    /// Push a pre-built [`EpochRecord`] through the same pipeline as
    /// [`Self::log`] — minus the local resource sample and the
    /// terminal one-liner. Used by the launcher's cluster dashboard
    /// sink, which builds records from controller-aggregated
    /// [`crate::distributed::ddp_run::EpochMetrics`] plus per-rank
    /// resource samples received over the wire.
    ///
    /// Drives the same JSON encoding as `log()` (`epoch_to_json`), so
    /// the dashboard HTML / JS sees identical frame shapes whether
    /// driven by the single-process Monitor or the launcher sink.
    /// Honors `is_primary` for symmetry with `log()`; non-primary calls
    /// no-op.
    pub fn log_epoch_record(&mut self, record: EpochRecord) {
        if !self.is_primary {
            return;
        }
        let epoch = record.epoch;
        self.epochs.push(record);
        if let Some(ref srv) = self.server {
            srv.push_epoch(self.epoch_to_json(epoch));
        }
    }

    /// Set the graph label and structural hash for the dashboard header.
    ///
    /// This is the standalone equivalent of what [`watch()`](Self::watch) does
    /// via `capture_graph_identity`. Use when you have the identity strings
    /// but not a `&Graph` reference (e.g. from `DdpHandle::setup_monitor()`).
    pub fn set_identity(&mut self, label: Option<&str>, hash: Option<&str>) {
        self.graph_label = label.map(|s| s.to_string());
        self.graph_hash = hash.map(|s| s.to_string());
        if let Some(ref srv) = self.server {
            srv.set_label_hash(
                self.graph_label.clone(),
                self.graph_hash.clone(),
            );
        }
    }

    fn capture_graph_identity(&mut self, graph: &Graph) {
        self.graph_label = graph.label().map(|s| s.to_string());
        self.graph_hash = Some(graph.structural_hash().to_string());
        if let Some(ref srv) = self.server {
            srv.set_label_hash(
                self.graph_label.clone(),
                self.graph_hash.clone(),
            );
        }
        self.capture_param_info(graph);
    }

    /// Build parameter summary and merge into metadata.
    fn capture_param_info(&mut self, graph: &Graph) {
        use crate::nn::Module;

        let params = graph.parameters();
        let total: i64 = params.iter()
            .map(|p| p.variable.shape().iter().product::<i64>())
            .sum();
        let trainable: i64 = params.iter()
            .filter(|p| !p.is_frozen())
            .map(|p| p.variable.shape().iter().product::<i64>())
            .sum();
        let frozen = total - trainable;

        let param_info = serde_json::json!({
            "parameters": {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
            }
        });

        // Merge into existing metadata (user-set fields take precedence)
        let merged = match &self.metadata {
            Some(existing) => {
                if let (serde_json::Value::Object(mut base), serde_json::Value::Object(extra)) =
                    (param_info.clone(), existing.clone())
                {
                    base.extend(extra);
                    serde_json::Value::Object(base)
                } else {
                    existing.clone()
                }
            }
            None => param_info,
        };

        if let Some(ref srv) = self.server {
            srv.set_metadata(merged.to_string());
        }
        self.metadata = Some(merged);
    }

    /// Log an epoch's results. Prints a one-line summary and pushes data
    /// to the dashboard if active.
    ///
    /// `epoch` is zero-based. `duration` is the wall-clock time for this epoch.
    ///
    /// The `metrics` argument accepts several forms:
    ///
    /// ```ignore
    /// // Plain metrics:
    /// monitor.log(epoch, t.elapsed(), &[("loss", val), ("lr", lr)]);
    ///
    /// // Graph observation (reads latest epoch history):
    /// monitor.log(epoch, t.elapsed(), &model);
    ///
    /// // Graph + extras (graph metrics first, then extras):
    /// monitor.log(epoch, t.elapsed(), (&model, &[("lr", lr)]));
    /// ```
    ///
    /// When using a graph, call [`Graph::flush()`] first so the epoch
    /// history is up to date. `log` does **not** flush — this keeps
    /// observation and monitoring decoupled.
    pub fn log(&mut self, epoch: usize, duration: Duration, metrics: impl Metrics) {
        if !self.is_primary {
            // Non-primary cluster ranks no-op so user code stays
            // identical: only rank 0 records / prints / pushes to
            // the dashboard. The aggregated view this `log` would
            // surface is identical across ranks anyway (the coord's
            // broadcast lands on every rank), so dropping non-primary
            // calls loses no information.
            return;
        }
        let gpu_metrics = metrics.gpu_metrics();
        let metrics = metrics.into_metrics();
        let duration_secs = duration.as_secs_f64();
        let resources = self.sampler.sample();

        let record = EpochRecord {
            epoch,
            duration_secs,
            metrics: metrics.clone(),
            resources: resources.clone(),
            gpu_metrics: gpu_metrics.clone(),
        };
        self.epochs.push(record);

        // --- Terminal output ---
        let mut line = String::with_capacity(256);
        let epoch_display = epoch + 1;
        let width = digit_count(self.total_epochs);
        let _ = write!(line, "  epoch {:>w$}/{}", epoch_display, self.total_epochs, w = width);

        for (name, val) in &metrics {
            let _ = write!(line, "  {}={}", name, format_metric(*val));
        }

        let _ = write!(line, "  [{}",format_eta(duration_secs));

        // ETA from the recent-epoch pace: mean of the last ≤5 epoch
        // durations, NOT the global elapsed/epochs average. The global
        // average is anchored at monitor start, so epoch 1 absorbs the
        // whole startup (data load, NCCL init, ElChe calibration) and the
        // ETA begins inflated, then melts for the rest of the run; it also
        // lags real pace changes (ElChe's schedule converging speeds epochs
        // up mid-run) by averaging them with ancient history. A short
        // sliding window tracks the actual current pace. `self.epochs` was
        // pushed above, so the window is never empty.
        if epoch_display < self.total_epochs {
            let k = self.epochs.len().min(5);
            let recent: f64 = self.epochs[self.epochs.len() - k..]
                .iter()
                .map(|r| r.duration_secs)
                .sum::<f64>()
                / k as f64;
            let remaining = recent * (self.total_epochs - epoch_display) as f64;
            let _ = write!(line, "  ETA {}", format_eta(remaining));
        }
        line.push(']');

        // Resource summary (compact). The VRAM/util numbers come from a
        // randomly-sampled rank (see ResourceSample::aggregate_rank); the
        // label exposes which rank, so a reader can correlate across
        // epochs as the sample drifts.
        let res = &resources;
        if let Some(alloc) = res.vram_allocated_bytes {
            let spill = match res.vram_total_bytes {
                Some(total) if alloc > total => alloc - total,
                _ => 0,
            };
            let label = match res.aggregate_rank {
                Some(idx) => format!("VRAM[cuda{idx}]"),
                None => String::from("VRAM"),
            };
            let _ = write!(
                line,
                "  {}: {} / {}",
                label,
                format_bytes(alloc),
                format_bytes(spill),
            );
        }
        // Label the util sample: a bare `(14%)` after the ETA bracket reads
        // as run progress. Same sampled rank as the VRAM figures, so it gets
        // the same bracket style.
        if let Some(gpu) = res.gpu_util_percent {
            let label = match res.aggregate_rank {
                Some(idx) => format!("gpu[cuda{idx}]"),
                None => String::from("gpu"),
            };
            let _ = write!(line, "  {label} {:.0}%", gpu);
        }

        crate::msg!("{}", line);

        // --- Dashboard push ---
        if let Some(ref srv) = self.server {
            srv.push_epoch(self.epoch_to_json(epoch));
        }
    }

    /// Signal training is complete. Prints a summary line.
    ///
    /// If `save_html` was called, writes the dashboard archive to disk.
    pub fn finish(&mut self) {
        self.finish_inner();
    }

    /// Signal training is complete and update the graph SVG with profiling data.
    ///
    /// If the graph has profiling enabled, the final SVG shows a timing heat
    /// map from the last forward pass — representative of steady-state
    /// performance. This SVG is pushed to the live dashboard and baked into
    /// the HTML archive.
    ///
    /// ```ignore
    /// model.enable_profiling();
    /// // ... training loop ...
    /// monitor.finish_with(&model);
    /// ```
    pub fn finish_with(&mut self, graph: &Graph) {
        // Try profiled SVG, fall back to plain
        if let Ok(svg_bytes) = graph.svg_with_profile(None) {
            self.set_svg(&String::from_utf8_lossy(&svg_bytes));
        } else if let Ok(svg_bytes) = graph.svg(None) {
            self.set_svg(&String::from_utf8_lossy(&svg_bytes));
        } else {
            eprintln!("  warning: could not generate graph SVG (is graphviz installed?)");
        }
        self.finish_inner();
    }

    fn finish_inner(&mut self) {
        if !self.is_primary {
            // Non-primary cluster ranks have no records / no server /
            // no HTML to write. Skip the summary print + export so
            // user-level `monitor.finish()` is a clean no-op there.
            return;
        }
        if !self.silent_summary {
            let total_time = self.start_time.elapsed().as_secs_f64();
            let mut line = format!("  training complete in {}", format_eta(total_time));

            if let Some(last) = self.epochs.last() {
                for (name, val) in &last.metrics {
                    let _ = write!(line, "  | {}: {}", name, format_metric(*val));
                }
            }

            crate::msg!("{}", line);
        }

        // Save HTML archive
        if let Some(ref path) = self.save_html {
            match self.build_archive() {
                Ok(html) => {
                    if let Err(e) = std::fs::write(path, html) {
                        eprintln!("  warning: failed to save dashboard archive: {}", e);
                    } else {
                        crate::msg!("  saved: {}", path);
                    }
                }
                Err(e) => eprintln!("  warning: failed to build dashboard archive: {}", e),
            }
        }

        if let Some(ref mut srv) = self.server {
            srv.shutdown();
        }
    }

    /// Return all recorded epoch data, ordered by epoch index.
    pub fn history(&self) -> &[EpochRecord] {
        &self.epochs
    }

    /// Write a human-readable training log to a text file.
    ///
    /// Each line has the format: `epoch N/T  metric=value  [duration]`.
    /// A final `# total: ...` line gives the overall wall-clock time.
    pub fn write_log(&self, path: &str) -> std::io::Result<()> {
        let mut b = String::with_capacity(4096);
        let _ = writeln!(b, "# flodl training log");
        let width = digit_count(self.total_epochs);

        for record in &self.epochs {
            let _ = write!(b, "epoch {:>w$}/{}", record.epoch + 1, self.total_epochs, w = width);
            for (name, val) in &record.metrics {
                let _ = write!(b, "  {}={}", name, format_metric(*val));
            }
            let _ = write!(b, "  [{}]", format_eta(record.duration_secs));
            b.push('\n');
        }

        if !self.epochs.is_empty() {
            let total = self.start_time.elapsed().as_secs_f64();
            let _ = writeln!(b, "# total: {}", format_eta(total));
        }

        std::fs::write(path, b)
    }

    /// Export epoch data to CSV for analysis in external tools.
    ///
    /// Columns: `epoch`, `duration_s`, one column per metric name, then
    /// `cpu_pct`, `ram_used`, `gpu_pct`, `vram_alloc`, `vram_spill`. Metric names are
    /// taken from the first epoch's metrics.
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        if self.epochs.is_empty() {
            return Ok(());
        }

        let metric_names: Vec<&str> = self.epochs[0]
            .metrics
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();

        let mut b = String::with_capacity(4096);
        b.push_str("epoch,duration_s");
        for name in &metric_names {
            b.push(',');
            b.push_str(name);
        }
        b.push_str(",cpu_pct,ram_used,gpu_pct,vram_alloc,vram_spill\n");

        for record in &self.epochs {
            let _ = write!(b, "{},{:.3}", record.epoch + 1, record.duration_secs);
            for (_, val) in &record.metrics {
                let _ = write!(b, ",{:.8}", val);
            }
            let spill = match (record.resources.vram_allocated_bytes, record.resources.vram_total_bytes) {
                (Some(alloc), Some(total)) if alloc > total => (alloc - total).to_string(),
                _ => String::new(),
            };
            let _ = write!(
                b,
                ",{},{},{},{},{}",
                record.resources.cpu_percent.map_or("".to_string(), |v| format!("{:.1}", v)),
                record.resources.ram_used_bytes.map_or("".to_string(), |v| v.to_string()),
                record.resources.gpu_util_percent.map_or("".to_string(), |v| format!("{:.1}", v)),
                record.resources.vram_allocated_bytes.map_or("".to_string(), |v| v.to_string()),
                spill,
            );
            b.push('\n');
        }

        std::fs::write(path, b)
    }

    /// Build a self-contained HTML archive with all epoch data baked in.
    ///
    /// The dashboard template checks for `ARCHIVE_DATA` on load — if present
    /// it replays from the baked data instead of connecting to SSE.
    fn build_archive(&self) -> std::result::Result<String, std::fmt::Error> {
        // Serialize all epochs to JSON array
        let mut data_json = String::from("[");
        for (i, record) in self.epochs.iter().enumerate() {
            if i > 0 { data_json.push(','); }
            let _ = write!(data_json, "{}", self.epoch_record_to_json(record));
        }
        data_json.push(']');

        // SVG as a JS template literal (backtick / ${ escaping is
        // template-literal safety; the </script> neutralization is applied
        // once to the whole assembled block below).
        let svg_js = match &self.svg_snapshot {
            Some(svg) => {
                let escaped = svg
                    .replace('\\', "\\\\")
                    .replace('`', "\\`")
                    .replace("${", "\\${");
                format!("`{}`", escaped)
            }
            None => "null".to_string(),
        };

        // Label, hash, and metadata for archive
        let label_js = match &self.graph_label {
            Some(l) => format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let hash_js = match &self.graph_hash {
            Some(h) => format!("\"{}\"", h),
            None => "null".to_string(),
        };
        let meta_js = match &self.metadata {
            Some(v) => v.to_string(),
            None => "null".to_string(),
        };

        let total_time = self.start_time.elapsed().as_secs_f64();

        let hw_js = format!("\"{}\"", self.hardware.replace('\\', "\\\\").replace('"', "\\\""));

        // GPU init from first epoch's resource data
        let gpu_init_js = self.epochs.first()
            .filter(|e| e.resources.gpus.len() >= 2)
            .map(|e| Self::gpu_init_json(&e.resources.gpus))
            .unwrap_or_else(|| "null".to_string());

        // Inject archive constants before the main <script> tag. Neutralize
        // </script> once across the whole assembled body (a value in any
        // constant — data, svg, label, hash, metadata, hardware — could
        // otherwise close the tag early; the HTML parser ignores JS quoting).
        let archive_consts = format!(
            "\nconst ARCHIVE_DATA={};\nconst ARCHIVE_SVG={};\nconst ARCHIVE_COMPLETE=\"Complete ({})\";\nconst ARCHIVE_LABEL={};\nconst ARCHIVE_HASH={};\nconst ARCHIVE_META={};\nconst ARCHIVE_HARDWARE={};\nconst ARCHIVE_GPU_INIT={};\n",
            data_json,
            svg_js,
            format_eta(total_time),
            label_js,
            hash_js,
            meta_js,
            hw_js,
            gpu_init_js,
        );
        let archive_block = format!("<script>{}</script>", neutralize_script_close(&archive_consts));

        let template = include_str!("dashboard.html");
        let html = template
            .replace("<title>floDl Training Dashboard</title>",
                     "<title>floDl Training Report</title>")
            .replace("<script>", &format!("{}\n<script>", archive_block));

        Ok(html)
    }

    /// Write a resource block to a JSON buffer.
    fn write_resources(b: &mut String, res: &ResourceSample) {
        b.push_str(",\"resources\":{");
        let mut first = true;
        if let Some(cpu) = res.cpu_percent
            && cpu.is_finite()
        {
            let _ = write!(b, "\"cpu\":{:.1}", cpu);
            first = false;
        }
        if let (Some(used), Some(total)) = (res.ram_used_bytes, res.ram_total_bytes) {
            if !first { b.push(','); }
            let _ = write!(b, "\"ram_used\":{},\"ram_total\":{}", used, total);
            first = false;
        }
        if let Some(gpu) = res.gpu_util_percent
            && gpu.is_finite()
        {
            if !first { b.push(','); }
            let _ = write!(b, "\"gpu\":{:.1}", gpu);
            first = false;
        }
        if let Some(alloc) = res.vram_allocated_bytes {
            if !first { b.push(','); }
            let _ = write!(b, "\"vram_alloc\":{}", alloc);
            if let Some(total) = res.vram_total_bytes {
                let _ = write!(b, ",\"vram_total\":{}", total);
            }
        }
        b.push('}');
    }

    /// Write per-GPU data (hardware + DDP metrics) to a JSON buffer.
    fn write_gpus(b: &mut String, res: &ResourceSample, ddp: &[GpuMetrics]) {
        if res.gpus.is_empty() && ddp.is_empty() {
            return;
        }
        b.push_str(",\"gpus\":[");
        let hw = &res.gpus;
        let n = hw.len().max(ddp.len());
        for i in 0..n {
            if i > 0 { b.push(','); }
            b.push('{');
            let mut first = true;
            // Hardware data from GpuSnapshot
            if let Some(gpu) = hw.get(i) {
                let _ = write!(b, "\"dev\":{}", gpu.device_index);
                first = false;
                if !gpu.name.is_empty() {
                    let _ = write!(b, ",\"name\":\"{}\"", gpu.name);
                }
                if let Some(util) = gpu.util_percent {
                    let _ = write!(b, ",\"util\":{:.1}", util);
                }
                if let Some(alloc) = gpu.vram_allocated_bytes {
                    let _ = write!(b, ",\"vram_alloc\":{}", alloc);
                }
                if let Some(total) = gpu.vram_total_bytes {
                    let _ = write!(b, ",\"vram_total\":{}", total);
                }
            }
            // DDP metrics from GpuMetrics
            if let Some(m) = ddp.get(i) {
                if first {
                    let _ = write!(b, "\"dev\":{}", m.device_index);
                }
                let _ = write!(b, ",\"throughput\":{:.4}", m.throughput);
                let _ = write!(b, ",\"chunk\":{:.4}", m.chunk_ratio);
                let _ = write!(b, ",\"shard\":{}", m.shard_size);
            }
            b.push('}');
        }
        b.push(']');
    }

    /// Serialize GPU hardware info as a JSON array for dashboard init.
    /// Minimal format: `[{"dev":0,"name":"...","vram_total":...}, ...]`
    fn gpu_init_json(gpus: &[resources::GpuSnapshot]) -> String {
        use std::fmt::Write;
        let mut b = String::from("[");
        for (i, gpu) in gpus.iter().enumerate() {
            if i > 0 { b.push(','); }
            b.push('{');
            let _ = write!(b, "\"dev\":{}", gpu.device_index);
            if !gpu.name.is_empty() {
                let _ = write!(b, ",\"name\":\"{}\"", gpu.name);
            }
            if let Some(total) = gpu.vram_total_bytes {
                let _ = write!(b, ",\"vram_total\":{}", total);
            }
            b.push('}');
        }
        b.push(']');
        b
    }

    /// Write metric values to a JSON buffer, replacing NaN/Infinity with null.
    fn write_metrics(b: &mut String, metrics: &[(String, f64)]) {
        b.push_str(",\"metrics\":{");
        for (i, (name, val)) in metrics.iter().enumerate() {
            if i > 0 { b.push(','); }
            if val.is_finite() {
                let _ = write!(b, "\"{}\":{:.8}", name, val);
            } else {
                let _ = write!(b, "\"{}\":null", name);
            }
        }
        b.push('}');
    }

    /// Serialize an epoch record to JSON from a stored record.
    fn epoch_record_to_json(&self, record: &EpochRecord) -> String {
        self.epoch_json(record, record.epoch + 1, None)
    }

    /// Serialize the latest epoch record to JSON (no serde), with a live ETA.
    fn epoch_to_json(&self, epoch: usize) -> String {
        let record = &self.epochs[self.epochs.len() - 1];
        let epoch_display = epoch + 1;
        // ETA only while epochs remain (not on the final epoch).
        let eta = if epoch_display < self.total_epochs {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let per_epoch = elapsed / epoch_display as f64;
            Some(per_epoch * (self.total_epochs - epoch_display) as f64)
        } else {
            None
        };
        self.epoch_json(record, epoch_display, eta)
    }

    /// Shared epoch-record JSON body: `{ epoch, total, duration [, eta] +
    /// metrics + resources + gpus }` (no serde). `eta` (seconds) is emitted
    /// only when present and finite — the sole difference between the
    /// stored-record and live-latest serializers.
    fn epoch_json(&self, record: &EpochRecord, epoch_display: usize, eta: Option<f64>) -> String {
        let mut b = String::with_capacity(512);
        b.push('{');
        let _ = write!(
            b,
            "\"epoch\":{},\"total\":{},\"duration\":{:.4}",
            epoch_display,
            self.total_epochs,
            record.duration_secs,
        );

        if let Some(remaining) = eta
            && remaining.is_finite()
        {
            let _ = write!(b, ",\"eta\":{:.1}", remaining);
        }

        Self::write_metrics(&mut b, &record.metrics);
        Self::write_resources(&mut b, &record.resources);
        Self::write_gpus(&mut b, &record.resources, &record.gpu_metrics);

        b.push('}');
        b
    }
}

/// Number of digits needed to display a number.
fn digit_count(n: usize) -> usize {
    if n == 0 { return 1; }
    ((n as f64).log10().floor() as usize) + 1
}

/// Neutralize `</script>` in data destined for an inline `<script>` block.
///
/// The HTML parser scans for `</script` literally, ignorant of JS string
/// or template-literal quoting, so a data value containing `</script>`
/// closes the tag early even inside `"..."` or `` `...` ``. This must run
/// on the whole assembled script body (every injected constant), not
/// per-value. `<\/script` is transparent everywhere it can land: JSON
/// (`\/` decodes to `/`), JS string, and template literal all render it as
/// `</script`. Both dashboard emitters — the live server's `serve_html`
/// and the static-report archive block — route their injected constants
/// through this one function so the escape set can't drift between them.
pub(crate) fn neutralize_script_close(body: &str) -> String {
    body.replace("</script", "<\\/script")
        .replace("</SCRIPT", "<\\/SCRIPT")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_basic() {
        let mut monitor = Monitor::new(10);
        monitor.log(0, Duration::from_millis(100), &[("loss", 1.5)]);
        monitor.log(1, Duration::from_millis(90), &[("loss", 1.2)]);
        assert_eq!(monitor.history().len(), 2);
        assert_eq!(monitor.history()[1].epoch, 1);
    }

    #[test]
    fn test_neutralize_script_close() {
        assert_eq!(neutralize_script_close("a</script>b"), "a<\\/script>b");
        assert_eq!(neutralize_script_close("x</SCRIPT>y"), "x<\\/SCRIPT>y");
        assert_eq!(neutralize_script_close("safe data"), "safe data");
    }

    #[test]
    fn test_archive_html_neutralizes_script_close_in_data() {
        // A label or metadata value containing </script> must not break out
        // of the injected <script> block. Before the fix the label/hash/
        // hardware constants were embedded without </script> neutralization.
        let mut monitor = Monitor::new(10);
        monitor.set_identity(Some("evil</script><script>alert(1)</script>"), None);
        monitor.set_metadata(serde_json::json!({
            "note": "meta</script><img src=x onerror=alert(2)>"
        }));
        monitor.log(0, Duration::from_millis(100), &[("loss", 1.0)]);

        let html = monitor.build_archive().unwrap();

        // The malicious payloads survive only in neutralized form.
        assert!(html.contains("evil<\\/script><script>alert(1)<\\/script>"),
            "label </script> not neutralized");
        assert!(html.contains("meta<\\/script>"), "metadata </script> not neutralized");
        // No raw breakout: the ONLY </script> occurrences are structural
        // closing tags, never immediately preceded by our payload text.
        assert!(!html.contains("evil</script>"), "raw label breakout present");
        assert!(!html.contains("meta</script>"), "raw metadata breakout present");
    }

    #[test]
    fn test_log_with_graph() {
        use crate::*;

        let dev = crate::tensor::test_device();
        let model = FlowBuilder::from(Linear::on_device(2, 4, dev).unwrap())
            .through(Linear::on_device(4, 2, dev).unwrap())
            .tag("output")
            .build()
            .unwrap();

        let mut monitor = Monitor::new(5);

        // Record + flush (user's responsibility)
        model.record_scalar("loss", 1.5);
        model.record_scalar("loss", 1.3);
        model.flush(&[]);

        // Graph + extras via tuple
        monitor.log(0, Duration::from_millis(50), (&model, &[("lr", 0.01)]));

        assert_eq!(monitor.history().len(), 1);
        let metrics = &monitor.history()[0].metrics;
        assert!(metrics.iter().any(|(k, _)| k == "loss"), "missing graph metric 'loss'");
        assert!(metrics.iter().any(|(k, _)| k == "lr"), "missing extra metric 'lr'");

        // loss should be the mean of 1.5 and 1.3
        let loss = metrics.iter().find(|(k, _)| k == "loss").unwrap().1;
        assert!((loss - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_log_graph_only() {
        use crate::*;

        let dev = crate::tensor::test_device();
        let model = FlowBuilder::from(Linear::on_device(2, 4, dev).unwrap())
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let mut monitor = Monitor::new(5);

        model.record_scalar("loss", 2.0);
        model.flush(&[]);

        // Graph only, no extras
        monitor.log(0, Duration::from_millis(50), &model);

        let metrics = &monitor.history()[0].metrics;
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].0, "loss");
        assert!((metrics[0].1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_digit_count() {
        assert_eq!(digit_count(0), 1);
        assert_eq!(digit_count(9), 1);
        assert_eq!(digit_count(10), 2);
        assert_eq!(digit_count(100), 3);
        assert_eq!(digit_count(999), 3);
    }

    #[test]
    fn test_watch_captures_label_hash() {
        use crate::*;

        let dev = crate::tensor::test_device();
        let model = FlowBuilder::from(Linear::on_device(2, 4, dev).unwrap())
            .label("test-model")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let mut monitor = Monitor::new(5);
        monitor.watch(&model);

        assert_eq!(monitor.graph_label.as_deref(), Some("test-model"));
        assert!(monitor.graph_hash.is_some());
        assert_eq!(monitor.graph_hash.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn test_build_archive_with_metadata() {
        use crate::*;

        let dev = crate::tensor::test_device();
        let model = FlowBuilder::from(Linear::on_device(2, 4, dev).unwrap())
            .label("meta-test")
            .through(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();

        let mut monitor = Monitor::new(5);
        monitor.watch(&model);
        monitor.set_metadata(serde_json::json!({
            "lr": 0.001,
            "batch_size": 32
        }));
        monitor.log(0, Duration::from_millis(50), &[("loss", 1.0)]);

        let html = monitor.build_archive().unwrap();
        assert!(html.contains("ARCHIVE_LABEL"));
        assert!(html.contains("ARCHIVE_HASH"));
        assert!(html.contains("ARCHIVE_META"));
        assert!(html.contains("meta-test"));
        assert!(html.contains("batch_size"));
    }

    // -----------------------------------------------------------------
    // is_primary cluster-detection: shares `cluster::ENV_MUTEX` with
    // other env-mutating tests in the crate (cluster::tests touches
    // the same vars). Uses `set_thread_local_rank_override` to avoid
    // touching `ENV_LOCAL_RANK` from a multi-test runner.
    // -----------------------------------------------------------------

    #[test]
    fn is_primary_defaults_true_when_no_cluster_env() {
        let _guard = crate::distributed::cluster::ENV_MUTEX.lock().unwrap();
        // Safety: holding the env-mutex serialises every in-crate test
        // that touches ENV_CLUSTER_JSON; no concurrent reader can race.
        unsafe {
            std::env::remove_var(crate::distributed::cluster::ENV_CLUSTER_JSON);
        }
        let monitor = Monitor::new(1);
        assert!(
            monitor.is_primary(),
            "no cluster envelope -> single-host mode -> primary",
        );
    }

    #[test]
    fn is_primary_true_for_cluster_rank_zero() {
        let envelope = serde_json::json!({
            "controller": { "host": "127.0.0.1", "port": 29500 },
            "world_size": 1,
            "num_workers": 1,
            "worker": {
                "host": "master",
                "ranks": [0],
                "local_devices": [0],
                "nccl_socket_ifname": "lo",
                "path": "/tmp",
                "arch": null,
            }
        });
        let hex = crate::distributed::cluster::hex_encode(
            &serde_json::to_vec(&envelope).unwrap(),
        );
        let _guard = crate::distributed::cluster::ENV_MUTEX.lock().unwrap();
        crate::distributed::cluster::set_thread_local_rank_override(Some(0));
        crate::distributed::cluster::set_thread_hostname_override(Some("master"));
        unsafe {
            std::env::set_var(
                crate::distributed::cluster::ENV_CLUSTER_JSON,
                &hex,
            );
        }
        let is_primary = Monitor::new(1).is_primary();
        // Clean up before asserting so a failing test doesn't leak
        // env into siblings that share the mutex.
        unsafe {
            std::env::remove_var(crate::distributed::cluster::ENV_CLUSTER_JSON);
        }
        crate::distributed::cluster::set_thread_local_rank_override(None);
        crate::distributed::cluster::set_thread_hostname_override(None);
        assert!(
            is_primary,
            "host owns rank 0 -> Monitor is the primary (dashboard) rank",
        );
    }

    #[test]
    fn is_primary_false_for_cluster_rank_nonzero() {
        // Worker host owns rank 1 only. Local-index 0 of this host
        // resolves to global rank 1 (per `LocalCluster::my_rank`).
        let envelope = serde_json::json!({
            "controller": { "host": "127.0.0.1", "port": 29500 },
            "world_size": 2,
            "num_workers": 2,
            "worker": {
                "host": "worker",
                "ranks": [1],
                "local_devices": [0],
                "nccl_socket_ifname": "lo",
                "path": "/tmp",
                "arch": null,
            }
        });
        let hex = crate::distributed::cluster::hex_encode(
            &serde_json::to_vec(&envelope).unwrap(),
        );
        let _guard = crate::distributed::cluster::ENV_MUTEX.lock().unwrap();
        crate::distributed::cluster::set_thread_local_rank_override(Some(0));
        crate::distributed::cluster::set_thread_hostname_override(Some("worker"));
        unsafe {
            std::env::set_var(
                crate::distributed::cluster::ENV_CLUSTER_JSON,
                &hex,
            );
        }
        let is_primary = Monitor::new(1).is_primary();
        unsafe {
            std::env::remove_var(crate::distributed::cluster::ENV_CLUSTER_JSON);
        }
        crate::distributed::cluster::set_thread_local_rank_override(None);
        crate::distributed::cluster::set_thread_hostname_override(None);
        assert!(
            !is_primary,
            "host does not own rank 0 -> Monitor must no-op on serve/log/finish",
        );
    }
}
