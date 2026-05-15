//! Training monitor with human-readable ETA, resource tracking, and live dashboard.
//!
//! The monitor prints a one-line summary per epoch and optionally serves a live
//! web dashboard with charts, resource graphs, and metric logs.
//!
//! ```ignore
//! use flodl::monitor::Monitor;
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
pub trait Metrics {
    /// Convert into owned `(name, value)` pairs for recording.
    fn into_metrics(self) -> Vec<(String, f64)>;
    /// Per-GPU DDP metrics. Default: empty (single-GPU or non-graph sources).
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

/// Build per-GPU DDP metrics from a Graph.
///
/// In process-per-rank cluster mode, each process only sees its own rank;
/// per-rank telemetry surfacing across ranks needs the cross-process
/// observability protocol (deferred). Returns empty for now.
fn graph_gpu_metrics(_graph: &Graph) -> Vec<GpuMetrics> {
    Vec::new()
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
}

impl Monitor {
    /// Create a new monitor for `total_epochs` epochs.
    pub fn new(total_epochs: usize) -> Self {
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
            hardware: crate::tensor::hardware_summary(),
        }
    }

    /// Start a live dashboard HTTP server on the given port.
    ///
    /// The dashboard is accessible at `http://localhost:{port}` and updates
    /// in real time as training progresses.
    pub fn serve(&mut self, port: u16) -> std::io::Result<()> {
        let srv = server::DashboardServer::start(port)?;
        crate::msg!("  dashboard: http://localhost:{}", port);
        srv.set_hardware(self.hardware.clone());

        // Sample GPU hardware for immediate tab init (before epoch 1)
        let init_sample = self.sampler.sample();
        if init_sample.gpus.len() >= 2 {
            srv.set_gpu_init(Self::gpu_init_json(&init_sample.gpus));
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
        if let Some(ref srv) = self.server {
            srv.set_svg(svg.to_string());
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

        // ETA
        if epoch_display < self.total_epochs {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let per_epoch = elapsed / epoch_display as f64;
            let remaining = per_epoch * (self.total_epochs - epoch_display) as f64;
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
        if let Some(gpu) = res.gpu_util_percent {
            let _ = write!(line, " ({:.0}%)", gpu);
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
        let total_time = self.start_time.elapsed().as_secs_f64();
        let mut line = format!("  training complete in {}", format_eta(total_time));

        if let Some(last) = self.epochs.last() {
            for (name, val) in &last.metrics {
                let _ = write!(line, "  | {}: {}", name, format_metric(*val));
            }
        }

        crate::msg!("{}", line);

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
        // Prevent HTML parser from seeing </script> in embedded JSON
        let data_json = data_json
            .replace("</script", "<\\/script")
            .replace("</SCRIPT", "<\\/SCRIPT");

        // SVG as a JS string literal (template literal for safe escaping)
        let svg_js = match &self.svg_snapshot {
            Some(svg) => {
                let escaped = svg
                    .replace('\\', "\\\\")
                    .replace('`', "\\`")
                    .replace("${", "\\${")
                    .replace("</script", "<\\/script")
                    .replace("</SCRIPT", "<\\/SCRIPT");
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
            Some(v) => {
                v.to_string()
                    .replace("</script", "<\\/script")
                    .replace("</SCRIPT", "<\\/SCRIPT")
            }
            None => "null".to_string(),
        };

        let total_time = self.start_time.elapsed().as_secs_f64();

        let hw_js = format!("\"{}\"", self.hardware.replace('\\', "\\\\").replace('"', "\\\""));

        // GPU init from first epoch's resource data
        let gpu_init_js = self.epochs.first()
            .filter(|e| e.resources.gpus.len() >= 2)
            .map(|e| Self::gpu_init_json(&e.resources.gpus))
            .unwrap_or_else(|| "null".to_string());

        // Inject archive constants before the main <script> tag
        let archive_block = format!(
            "<script>\nconst ARCHIVE_DATA={};\nconst ARCHIVE_SVG={};\nconst ARCHIVE_COMPLETE=\"Complete ({})\";\nconst ARCHIVE_LABEL={};\nconst ARCHIVE_HASH={};\nconst ARCHIVE_META={};\nconst ARCHIVE_HARDWARE={};\nconst ARCHIVE_GPU_INIT={};\n</script>",
            data_json,
            svg_js,
            format_eta(total_time),
            label_js,
            hash_js,
            meta_js,
            hw_js,
            gpu_init_js,
        );

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
        let epoch_display = record.epoch + 1;

        let mut b = String::with_capacity(512);
        b.push('{');
        let _ = write!(
            b,
            "\"epoch\":{},\"total\":{},\"duration\":{:.4}",
            epoch_display,
            self.total_epochs,
            record.duration_secs,
        );

        Self::write_metrics(&mut b, &record.metrics);
        Self::write_resources(&mut b, &record.resources);
        Self::write_gpus(&mut b, &record.resources, &record.gpu_metrics);

        b.push('}');
        b
    }

    /// Serialize an epoch record to JSON (no serde).
    fn epoch_to_json(&self, epoch: usize) -> String {
        let record = &self.epochs[self.epochs.len() - 1];

        let mut b = String::with_capacity(512);
        b.push('{');
        let _ = write!(
            b,
            "\"epoch\":{},\"total\":{},\"duration\":{:.4}",
            epoch + 1,
            self.total_epochs,
            record.duration_secs,
        );

        // ETA
        let epoch_display = epoch + 1;
        if epoch_display < self.total_epochs && epoch_display > 0 {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let per_epoch = elapsed / epoch_display as f64;
            let remaining = per_epoch * (self.total_epochs - epoch_display) as f64;
            if remaining.is_finite() {
                let _ = write!(b, ",\"eta\":{:.1}", remaining);
            }
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
}
