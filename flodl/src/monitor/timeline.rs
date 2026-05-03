//! High-frequency system timeline profiler for training diagnostics.
//!
//! Captures CPU, RAM, and per-GPU metrics at configurable intervals (default 100ms),
//! plus training events (sync, epoch boundaries, anchor changes, throttle). Detects
//! idle gaps and produces swimlane visualizations for debugging DDP behavior.
//!
//! ```ignore
//! use flodl::monitor::Timeline;
//!
//! let tl = Timeline::new(100); // 100ms polling
//! tl.start();
//!
//! // ... training with event injection ...
//! tl.event(EventKind::EpochStart { epoch: 0 });
//! // ... training ...
//! tl.event(EventKind::EpochEnd { epoch: 0, loss: 0.42, lr: 0.001 });
//!
//! tl.stop();
//! tl.save_html("timeline.html")?;
//! ```

use std::fmt::Write as FmtWrite;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::resources::{read_cpu_times, read_meminfo, CpuTimes};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-GPU snapshot within a timeline sample.
#[derive(Debug, Clone)]
pub struct GpuTimelineSample {
    /// CUDA device index.
    pub device: u8,
    /// GPU compute utilization (0-100%).
    pub compute_util: u8,
    /// Physical VRAM used (NVML, bytes).
    pub vram_used_bytes: u64,
    /// CUDA caching allocator bytes.
    pub vram_allocated_bytes: u64,
    /// Total physical VRAM (bytes).
    pub vram_total_bytes: u64,
}

/// A single timeline sample capturing full system state.
#[derive(Debug, Clone)]
pub struct TimelineSample {
    /// Milliseconds since timeline start.
    pub elapsed_ms: u64,
    /// CPU utilization (0-100%).
    pub cpu_util: f32,
    /// System RAM used (bytes).
    pub ram_used_bytes: u64,
    /// System RAM total (bytes).
    pub ram_total_bytes: u64,
    /// Per-GPU snapshots.
    pub gpus: Vec<GpuTimelineSample>,
}

/// A timestamped training event.
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Milliseconds since timeline start.
    pub elapsed_ms: u64,
    /// Event type.
    pub kind: EventKind,
}

/// Training event types captured by the timeline.
#[derive(Debug, Clone)]
pub enum EventKind {
    /// Worker started processing an epoch.
    EpochStart { epoch: usize },
    /// Worker finished an epoch. `lr` is the optimizer's current learning
    /// rate at end-of-epoch; under step-decay schedules this is the LR that
    /// was active throughout the epoch, under continuous schedules it is the
    /// last value the scheduler wrote.
    EpochEnd { epoch: usize, loss: f64, lr: f64 },
    /// AllReduce or parameter sync started.
    SyncStart,
    /// AllReduce or parameter sync completed.
    SyncEnd { duration_ms: f64 },
    /// CPU averaging started (coordinator collecting snapshots).
    CpuAvgStart,
    /// CPU averaging completed.
    CpuAvgEnd { duration_ms: f64 },
    /// El Che anchor value changed.
    AnchorChanged { from: usize, to: usize },
    /// Worker was throttled (max_batch_diff exceeded).
    Throttle { rank: usize },
    /// Auto-detected GPU idle gap (post-processing).
    Idle { device: u8, duration_ms: f64 },
    /// User-defined event.
    Custom { label: String },
    /// MSF passive observation: per-AllReduce divergence + lambda sample.
    ///
    /// Emitted at every `ConvergenceGuard::observe_lambda` call. `d_raw` is
    /// the max normalized delta across ranks; `lambda_raw`/`lambda_ema` are
    /// the across-event Lyapunov proxy `(1/k) * log(D_t / D_{t-1})` and its
    /// EMA. `None` on the first event in a fresh estimator or when below the
    /// noise floor. See `docs/design/msf-cadence-control.md`.
    Divergence {
        d_raw: f64,
        lambda_raw: Option<f64>,
        lambda_ema: Option<f64>,
        k_used: usize,
        k_max: usize,
        step: usize,
        /// Per-rank `||pre - post|| / ||post||`. Length = world_size.
        deltas: Vec<f64>,
        /// L2 norm of the post-AllReduce consensus weights `||W̄_t||`.
        /// `None` when not computed (NCCL v1 path skips this). Carries the
        /// longitudinal meta-oscillator state: tracking `||W̄_t||` across
        /// events gives consensus magnitude trajectory between syncs.
        post_norm: Option<f64>,
        /// Per-rank pre-AllReduce L2 norm `||W_i||`. `None` when not
        /// computed. With `post_norm` and `deltas` this enables the
        /// cosine-similarity / magnitude-shift decomposition (MSF/SWA
        /// directional vs magnitude split).
        pre_norms: Option<Vec<f64>>,
        /// In-flight epoch at the time of this event (= `last_aggregated_epoch
        /// + 1`, or 0 before the first epoch aggregates). `None` for
        /// timelines emitted before this field was added; consumers fall
        /// back to `EpochEnd` timestamp lookup.
        epoch: Option<usize>,
    },
    /// Guard-specific diagnostic values for the current AllReduce event.
    /// Emitted by the coordinator after each `report()` call to a
    /// pluggable [`crate::distributed::ddp_run::ConvergenceGuard`]. The key
    /// set depends on the active guard (e.g. MsfGuard emits
    /// `lambda_raw` / `lambda_ema`; TrendGuard emits `d_history_last`;
    /// NoGuard emits nothing). Old timelines lack this event entirely;
    /// consumers should treat absence as "no diagnostics available".
    GuardTelemetry {
        epoch: usize,
        step: usize,
        values: Vec<(String, f64)>,
    },
    /// MSF passive observation: per-epoch divergence + lambda aggregates.
    ///
    /// Emitted at `on_epoch_aggregated`. Aggregates over all `Divergence`
    /// events in this epoch plus a snapshot of the last sample. The lambda
    /// estimator state is NOT reset across epochs — `prev_d` carries forward.
    DivergenceEpoch {
        epoch: usize,
        sync_count: usize,
        d_min: f64,
        d_max: f64,
        d_mean: f64,
        lambda_min: Option<f64>,
        lambda_max: Option<f64>,
        lambda_mean: Option<f64>,
        lambda_ema_at_epoch_end: Option<f64>,
        d_at_epoch_end: f64,
        k_at_epoch_end: usize,
    },
}

/// Aggregate statistics from a timeline.
#[derive(Debug, Clone)]
pub struct TimelineSummary {
    /// Total duration in milliseconds.
    pub total_ms: u64,
    /// Number of samples collected.
    pub sample_count: usize,
    /// Number of events recorded.
    pub event_count: usize,
    /// Per-GPU idle percentage (fraction of samples with compute_util below threshold).
    pub gpu_idle_pct: Vec<f64>,
    /// Number of sync events (SyncStart count).
    pub sync_count: usize,
    /// Average sync duration in ms (from SyncEnd events).
    pub avg_sync_ms: f64,
    /// Number of CPU averaging events.
    pub cpu_avg_count: usize,
    /// Average CPU averaging duration in ms.
    pub avg_cpu_avg_ms: f64,
    /// Number of anchor changes.
    pub anchor_change_count: usize,
    /// Number of throttle events.
    pub throttle_count: usize,
}

/// A batch of samples and events sent to live subscribers at the broadcast interval.
#[derive(Debug, Clone)]
pub struct TimelineBroadcast {
    /// Samples collected since the last broadcast.
    pub samples: Vec<TimelineSample>,
    /// Events injected since the last broadcast.
    pub events: Vec<TimelineEvent>,
}

/// High-frequency system profiler for training diagnostics.
///
/// Captures CPU, RAM, and per-GPU metrics at configurable intervals plus
/// training events. Thread-safe: wrap in `Arc` and share across coordinator
/// and worker threads.
///
/// Polling and broadcasting are decoupled: samples are collected at
/// `poll_interval_ms` (default 100ms) for full-resolution post-hoc analysis,
/// while live subscribers receive batched updates at `broadcast_interval_ms`
/// (default 1000ms) to keep network and rendering overhead low.
pub struct Timeline {
    start: Instant,
    poll_interval_ms: u64,
    broadcast_interval_ms: u64,
    samples: Mutex<Vec<TimelineSample>>,
    events: Mutex<Vec<TimelineEvent>>,
    stop_flag: AtomicBool,
    poller_handle: Mutex<Option<JoinHandle<()>>>,
    /// Live subscribers receive batched updates at the broadcast interval.
    /// Cleaned up on send failure (subscriber dropped).
    subscribers: Mutex<Vec<mpsc::Sender<TimelineBroadcast>>>,
    /// Pending samples accumulated since last broadcast (only accessed by poll thread).
    /// Stored here rather than as a poll_loop local so subscribe() can document the contract.
    pending_samples: Mutex<Vec<TimelineSample>>,
    /// Pending events accumulated since last broadcast.
    pending_events: Mutex<Vec<TimelineEvent>>,
}

impl Timeline {
    /// Create a new timeline with the given poll interval (milliseconds).
    ///
    /// Returns an `Arc<Timeline>` since it is always shared across threads.
    /// Call `start()` to begin background sampling.
    ///
    /// Broadcast interval defaults to 1000ms (10x the typical 100ms poll).
    pub fn new(poll_interval_ms: u64) -> Arc<Self> {
        Arc::new(Self {
            start: Instant::now(),
            poll_interval_ms,
            broadcast_interval_ms: 1000,
            samples: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            stop_flag: AtomicBool::new(false),
            poller_handle: Mutex::new(None),
            subscribers: Mutex::new(Vec::new()),
            pending_samples: Mutex::new(Vec::new()),
            pending_events: Mutex::new(Vec::new()),
        })
    }

    /// Create a new timeline with explicit poll and broadcast intervals.
    ///
    /// `poll_interval_ms`: how often to sample system metrics (default 100ms).
    /// `broadcast_interval_ms`: how often to send batched updates to
    /// subscribers (default 1000ms). Subscribers receive all samples collected
    /// since the last broadcast, keeping network overhead low while retaining
    /// full-resolution data for post-hoc analysis.
    pub fn with_intervals(poll_interval_ms: u64, broadcast_interval_ms: u64) -> Arc<Self> {
        Arc::new(Self {
            start: Instant::now(),
            poll_interval_ms,
            broadcast_interval_ms,
            samples: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            stop_flag: AtomicBool::new(false),
            poller_handle: Mutex::new(None),
            subscribers: Mutex::new(Vec::new()),
            pending_samples: Mutex::new(Vec::new()),
            pending_events: Mutex::new(Vec::new()),
        })
    }

    /// Subscribe to live batched updates.
    ///
    /// Returns a receiver that yields [`TimelineBroadcast`] batches at the
    /// configured broadcast interval. The receiver is disconnected when the
    /// timeline is stopped or dropped.
    ///
    /// Multiple subscribers are supported. Failed sends (dropped receiver)
    /// are silently cleaned up.
    pub fn subscribe(&self) -> mpsc::Receiver<TimelineBroadcast> {
        let (tx, rx) = mpsc::channel();
        self.subscribers.lock().unwrap().push(tx);
        rx
    }

    /// Start background polling. Idempotent: does nothing if already running.
    pub fn start(self: &Arc<Self>) {
        let mut handle = self.poller_handle.lock().unwrap();
        if handle.is_some() {
            return; // already running
        }

        self.stop_flag.store(false, Ordering::SeqCst);
        let tl = Arc::clone(self);
        *handle = Some(thread::spawn(move || tl.poll_loop()));
    }

    /// Stop background polling and join the thread.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        let handle = self.poller_handle.lock().unwrap().take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }

    /// Inject a training event with the current timestamp.
    pub fn event(&self, kind: EventKind) {
        let elapsed_ms = self.start.elapsed().as_millis() as u64;
        let evt = TimelineEvent { elapsed_ms, kind };
        self.events.lock().unwrap().push(evt.clone());
        self.pending_events.lock().unwrap().push(evt);
    }

    /// Current elapsed milliseconds since timeline creation.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Detect idle gaps for a device: consecutive samples where `compute_util < threshold_pct`
    /// lasting at least `min_ms` milliseconds.
    ///
    /// Returns `(start_ms, end_ms)` pairs.
    pub fn idle_gaps(&self, device: u8, threshold_pct: u8, min_ms: u64) -> Vec<(u64, u64)> {
        let samples = self.samples.lock().unwrap();
        let mut gaps = Vec::new();
        let mut gap_start: Option<u64> = None;

        for s in samples.iter() {
            let util = s
                .gpus
                .iter()
                .find(|g| g.device == device)
                .map(|g| g.compute_util)
                .unwrap_or(100);

            if util < threshold_pct {
                if gap_start.is_none() {
                    gap_start = Some(s.elapsed_ms);
                }
            } else if let Some(start) = gap_start.take() {
                let duration = s.elapsed_ms.saturating_sub(start);
                if duration >= min_ms {
                    gaps.push((start, s.elapsed_ms));
                }
            }
        }

        // Close trailing gap
        if let Some(start) = gap_start {
            if let Some(last) = samples.last() {
                let duration = last.elapsed_ms.saturating_sub(start);
                if duration >= min_ms {
                    gaps.push((start, last.elapsed_ms));
                }
            }
        }

        gaps
    }

    /// Compute aggregate statistics from the timeline.
    pub fn summary(&self) -> TimelineSummary {
        let samples = self.samples.lock().unwrap();
        let events = self.events.lock().unwrap();

        let total_ms = samples.last().map(|s| s.elapsed_ms).unwrap_or(0);
        let sample_count = samples.len();

        // Per-GPU idle percentage (compute_util < 5%)
        let n_gpus = samples.first().map(|s| s.gpus.len()).unwrap_or(0);
        let mut gpu_idle_pct = vec![0.0; n_gpus];
        if sample_count > 0 {
            for s in samples.iter() {
                for (gi, g) in s.gpus.iter().enumerate() {
                    if g.compute_util < 5 {
                        gpu_idle_pct[gi] += 1.0;
                    }
                }
            }
            for v in &mut gpu_idle_pct {
                *v = *v / sample_count as f64 * 100.0;
            }
        }

        let mut sync_count = 0usize;
        let mut sync_total_ms = 0.0f64;
        let mut sync_end_count = 0usize;
        let mut cpu_avg_count = 0usize;
        let mut cpu_avg_total_ms = 0.0f64;
        let mut cpu_avg_end_count = 0usize;
        let mut anchor_change_count = 0usize;
        let mut throttle_count = 0usize;

        for e in events.iter() {
            match &e.kind {
                EventKind::SyncStart => sync_count += 1,
                EventKind::SyncEnd { duration_ms } => {
                    sync_total_ms += duration_ms;
                    sync_end_count += 1;
                }
                EventKind::CpuAvgStart => cpu_avg_count += 1,
                EventKind::CpuAvgEnd { duration_ms } => {
                    cpu_avg_total_ms += duration_ms;
                    cpu_avg_end_count += 1;
                }
                EventKind::AnchorChanged { .. } => anchor_change_count += 1,
                EventKind::Throttle { .. } => throttle_count += 1,
                _ => {}
            }
        }

        TimelineSummary {
            total_ms,
            sample_count,
            event_count: events.len(),
            gpu_idle_pct,
            sync_count,
            avg_sync_ms: if sync_end_count > 0 {
                sync_total_ms / sync_end_count as f64
            } else {
                0.0
            },
            cpu_avg_count,
            avg_cpu_avg_ms: if cpu_avg_end_count > 0 {
                cpu_avg_total_ms / cpu_avg_end_count as f64
            } else {
                0.0
            },
            anchor_change_count,
            throttle_count,
        }
    }

    /// Take ownership of samples and events, consuming the stored data.
    /// After this call, the internal vectors are empty.
    pub fn drain(&self) -> (Vec<TimelineSample>, Vec<TimelineEvent>) {
        let mut samples = self.samples.lock().unwrap();
        let mut events = self.events.lock().unwrap();
        let s = std::mem::take(&mut *samples);
        let e = std::mem::take(&mut *events);
        (s, e)
    }

    /// Number of samples collected so far.
    pub fn sample_count(&self) -> usize {
        self.samples.lock().unwrap().len()
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    /// Save timeline as JSON.
    pub fn save_json(&self, path: &str) -> io::Result<()> {
        let samples = self.samples.lock().unwrap();
        let events = self.events.lock().unwrap();

        let mut out = String::with_capacity(samples.len() * 120 + events.len() * 80);
        out.push_str("{\n\"samples\":[\n");
        write_samples_json(&mut out, &samples);
        out.push_str("],\n\"events\":[\n");
        write_events_json(&mut out, &events);
        out.push_str("]\n}\n");

        let mut f = std::fs::File::create(path)?;
        f.write_all(out.as_bytes())
    }

    /// Save timeline as CSV.
    pub fn save_csv(&self, path: &str) -> io::Result<()> {
        let samples = self.samples.lock().unwrap();

        let n_gpus = samples.first().map(|s| s.gpus.len()).unwrap_or(0);

        let mut out = String::with_capacity(samples.len() * 80);
        // Header
        out.push_str("elapsed_ms,cpu_util,ram_used,ram_total");
        for i in 0..n_gpus {
            let _ = write!(
                out,
                ",gpu{i}_util,gpu{i}_vram_alloc,gpu{i}_vram_used,gpu{i}_vram_total"
            );
        }
        out.push('\n');

        for s in samples.iter() {
            let _ = write!(
                out,
                "{},{:.1},{},{}",
                s.elapsed_ms, s.cpu_util, s.ram_used_bytes, s.ram_total_bytes,
            );
            for g in &s.gpus {
                let _ = write!(
                    out,
                    ",{},{},{},{}",
                    g.compute_util, g.vram_allocated_bytes, g.vram_used_bytes, g.vram_total_bytes,
                );
            }
            out.push('\n');
        }

        let mut f = std::fs::File::create(path)?;
        f.write_all(out.as_bytes())
    }

    /// Save timeline as a self-contained HTML visualization.
    pub fn save_html(&self, path: &str) -> io::Result<()> {
        let samples = self.samples.lock().unwrap();
        let events = self.events.lock().unwrap();

        let template = include_str!("timeline.html");

        // Build data injection block
        let mut samples_json = String::with_capacity(samples.len() * 100);
        write_samples_json(&mut samples_json, &samples);

        let mut events_json = String::with_capacity(events.len() * 80);
        write_events_json(&mut events_json, &events);

        let inject = format!(
            "<script>\nconst TIMELINE_SAMPLES=[{}];\nconst TIMELINE_EVENTS=[{}];\n</script>\n",
            samples_json, events_json,
        );

        let html = template.replacen("<!-- TIMELINE_DATA -->", &inject, 1);

        let mut f = std::fs::File::create(path)?;
        f.write_all(html.as_bytes())
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    fn poll_loop(&self) {
        let interval = Duration::from_millis(self.poll_interval_ms);
        let broadcast_interval = Duration::from_millis(self.broadcast_interval_ms);
        let mut prev_cpu: Option<CpuTimes> = None;
        let mut last_broadcast = Instant::now();

        let n_gpus = {
            #[cfg(feature = "cuda")]
            {
                crate::tensor::cuda_device_count().max(0) as usize
            }
            #[cfg(not(feature = "cuda"))]
            {
                0usize
            }
        };

        while !self.stop_flag.load(Ordering::SeqCst) {
            let elapsed_ms = self.start.elapsed().as_millis() as u64;

            // CPU utilization (delta)
            let cur_cpu = read_cpu_times();
            let cpu_util = match (&prev_cpu, &cur_cpu) {
                (Some(prev), Some(cur)) => {
                    let dt = cur.total.saturating_sub(prev.total);
                    let di = cur.idle.saturating_sub(prev.idle);
                    if dt > 0 {
                        (dt.saturating_sub(di)) as f32 / dt as f32 * 100.0
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            prev_cpu = cur_cpu;

            // RAM
            let (ram_used, ram_total) = read_meminfo().unwrap_or((0, 0));

            // Per-GPU
            let mut gpus = Vec::with_capacity(n_gpus);
            for i in 0..n_gpus {
                #[cfg(feature = "cuda")]
                let (compute_util, vram_used, vram_alloc, vram_total) = {
                    let idx = i as i32;
                    let util = crate::tensor::cuda_utilization_idx(idx)
                        .map(|u| u as u8)
                        .unwrap_or(0);
                    let (used, total) = crate::tensor::cuda_memory_info_idx(idx).unwrap_or((0, 0));
                    let alloc = crate::tensor::cuda_allocated_bytes_idx(idx).unwrap_or(0);
                    (util, used, alloc, total)
                };
                #[cfg(not(feature = "cuda"))]
                let (compute_util, vram_used, vram_alloc, vram_total) = (0u8, 0u64, 0u64, 0u64);

                gpus.push(GpuTimelineSample {
                    device: i as u8,
                    compute_util,
                    vram_used_bytes: vram_used,
                    vram_allocated_bytes: vram_alloc,
                    vram_total_bytes: vram_total,
                });
            }

            let sample = TimelineSample {
                elapsed_ms,
                cpu_util,
                ram_used_bytes: ram_used,
                ram_total_bytes: ram_total,
                gpus,
            };

            // Store in full-resolution archive
            self.samples.lock().unwrap().push(sample.clone());
            // Buffer for next broadcast
            self.pending_samples.lock().unwrap().push(sample);

            // Broadcast to subscribers at the slower interval
            if last_broadcast.elapsed() >= broadcast_interval {
                self.flush_broadcast();
                last_broadcast = Instant::now();
            }

            // Sleep in small increments to check stop flag
            let wake = Instant::now() + interval;
            while Instant::now() < wake {
                if self.stop_flag.load(Ordering::SeqCst) {
                    // Final broadcast before exit
                    self.flush_broadcast();
                    return;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    /// Send pending samples and events to all subscribers, then clear the buffers.
    fn flush_broadcast(&self) {
        let samples = std::mem::take(&mut *self.pending_samples.lock().unwrap());
        let events = std::mem::take(&mut *self.pending_events.lock().unwrap());

        if samples.is_empty() && events.is_empty() {
            return;
        }

        let batch = TimelineBroadcast { samples, events };
        let mut subs = self.subscribers.lock().unwrap();
        subs.retain(|tx| tx.send(batch.clone()).is_ok());
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.poller_handle.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

impl std::fmt::Debug for Timeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timeline")
            .field("poll_interval_ms", &self.poll_interval_ms)
            .field("broadcast_interval_ms", &self.broadcast_interval_ms)
            .field("samples", &self.samples.lock().unwrap().len())
            .field("events", &self.events.lock().unwrap().len())
            .field("running", &!self.stop_flag.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// JSON helpers (manual, no serde -- matches monitor pattern)
// ---------------------------------------------------------------------------

fn write_samples_json(out: &mut String, samples: &[TimelineSample]) {
    for (i, s) in samples.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        let _ = write!(
            out,
            "{{\"t\":{},\"cpu\":{:.1},\"ram\":[{},{}],\"gpus\":[",
            s.elapsed_ms, s.cpu_util, s.ram_used_bytes, s.ram_total_bytes,
        );
        for (gi, g) in s.gpus.iter().enumerate() {
            if gi > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"d\":{},\"u\":{},\"vu\":{},\"va\":{},\"vt\":{}}}",
                g.device,
                g.compute_util,
                g.vram_used_bytes,
                g.vram_allocated_bytes,
                g.vram_total_bytes,
            );
        }
        out.push_str("]}");
    }
}

fn write_events_json(out: &mut String, events: &[TimelineEvent]) {
    for (i, e) in events.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        let _ = write!(out, "{{\"t\":{},", e.elapsed_ms);
        match &e.kind {
            EventKind::EpochStart { epoch } => {
                let _ = write!(out, "\"k\":\"epoch_start\",\"epoch\":{epoch}");
            }
            EventKind::EpochEnd { epoch, loss, lr } => {
                let _ = write!(
                    out,
                    "\"k\":\"epoch_end\",\"epoch\":{epoch},\"loss\":{loss:.6},\"lr\":{lr:.6e}"
                );
            }
            EventKind::SyncStart => {
                out.push_str("\"k\":\"sync_start\"");
            }
            EventKind::SyncEnd { duration_ms } => {
                let _ = write!(out, "\"k\":\"sync_end\",\"ms\":{duration_ms:.3}");
            }
            EventKind::CpuAvgStart => {
                out.push_str("\"k\":\"cpu_avg_start\"");
            }
            EventKind::CpuAvgEnd { duration_ms } => {
                let _ = write!(out, "\"k\":\"cpu_avg_end\",\"ms\":{duration_ms:.3}");
            }
            EventKind::AnchorChanged { from, to } => {
                let _ = write!(out, "\"k\":\"anchor\",\"from\":{from},\"to\":{to}");
            }
            EventKind::Throttle { rank } => {
                let _ = write!(out, "\"k\":\"throttle\",\"rank\":{rank}");
            }
            EventKind::Idle {
                device,
                duration_ms,
            } => {
                let _ = write!(
                    out,
                    "\"k\":\"idle\",\"dev\":{device},\"ms\":{duration_ms:.1}"
                );
            }
            EventKind::Custom { label } => {
                // Escape quotes in label
                let escaped = label.replace('\\', "\\\\").replace('"', "\\\"");
                let _ = write!(out, "\"k\":\"custom\",\"label\":\"{escaped}\"");
            }
            EventKind::Divergence {
                d_raw,
                lambda_raw,
                lambda_ema,
                k_used,
                k_max,
                step,
                deltas,
                post_norm,
                pre_norms,
                epoch,
            } => {
                let _ = write!(
                    out,
                    "\"k\":\"div\",\"d\":{d_raw:.6e},\"k_used\":{k_used},\"k_max\":{k_max},\"step\":{step}"
                );
                if let Some(ep) = epoch {
                    let _ = write!(out, ",\"epoch\":{ep}");
                }
                if let Some(l) = lambda_raw {
                    let _ = write!(out, ",\"lambda\":{l:.6e}");
                }
                if let Some(l) = lambda_ema {
                    let _ = write!(out, ",\"lambda_ema\":{l:.6e}");
                }
                if let Some(p) = post_norm {
                    let _ = write!(out, ",\"post_norm\":{p:.6e}");
                }
                if let Some(prs) = pre_norms {
                    out.push_str(",\"pre_norms\":[");
                    for (i, p) in prs.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        let _ = write!(out, "{p:.6e}");
                    }
                    out.push(']');
                }
                out.push_str(",\"deltas\":[");
                for (i, d) in deltas.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "{d:.6e}");
                }
                out.push(']');
            }
            EventKind::GuardTelemetry { epoch, step, values } => {
                let _ = write!(
                    out,
                    "\"k\":\"guard_telemetry\",\"epoch\":{epoch},\"step\":{step},\"values\":{{",
                );
                for (i, (k, v)) in values.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    let escaped = k.replace('\\', "\\\\").replace('"', "\\\"");
                    let _ = write!(out, "\"{escaped}\":{v:.6e}");
                }
                out.push('}');
            }
            EventKind::DivergenceEpoch {
                epoch,
                sync_count,
                d_min,
                d_max,
                d_mean,
                lambda_min,
                lambda_max,
                lambda_mean,
                lambda_ema_at_epoch_end,
                d_at_epoch_end,
                k_at_epoch_end,
            } => {
                let _ = write!(
                    out,
                    "\"k\":\"div_epoch\",\"epoch\":{epoch},\"syncs\":{sync_count},\
                     \"d_min\":{d_min:.6e},\"d_max\":{d_max:.6e},\"d_mean\":{d_mean:.6e},\
                     \"d_end\":{d_at_epoch_end:.6e},\"k_end\":{k_at_epoch_end}"
                );
                if let Some(l) = lambda_min {
                    let _ = write!(out, ",\"lambda_min\":{l:.6e}");
                }
                if let Some(l) = lambda_max {
                    let _ = write!(out, ",\"lambda_max\":{l:.6e}");
                }
                if let Some(l) = lambda_mean {
                    let _ = write!(out, ",\"lambda_mean\":{l:.6e}");
                }
                if let Some(l) = lambda_ema_at_epoch_end {
                    let _ = write!(out, ",\"lambda_ema_end\":{l:.6e}");
                }
            }
        }
        out.push('}');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_create_and_event() {
        let tl = Timeline::new(100);
        tl.event(EventKind::EpochStart { epoch: 0 });
        tl.event(EventKind::SyncStart);
        tl.event(EventKind::SyncEnd { duration_ms: 1.5 });
        tl.event(EventKind::EpochEnd {
            epoch: 0,
            loss: 0.42,
            lr: 0.001,
        });

        let events = tl.events.lock().unwrap();
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0].kind, EventKind::EpochStart { epoch: 0 }));
    }

    #[test]
    fn test_idle_gaps() {
        let tl = Timeline::new(100);
        // Manually inject samples
        {
            let mut samples = tl.samples.lock().unwrap();
            for i in 0..20 {
                let util = if (5..15).contains(&i) { 2 } else { 80 };
                samples.push(TimelineSample {
                    elapsed_ms: i * 100,
                    cpu_util: 50.0,
                    ram_used_bytes: 1_000_000,
                    ram_total_bytes: 8_000_000,
                    gpus: vec![GpuTimelineSample {
                        device: 0,
                        compute_util: util,
                        vram_used_bytes: 0,
                        vram_allocated_bytes: 0,
                        vram_total_bytes: 0,
                    }],
                });
            }
        }

        let gaps = tl.idle_gaps(0, 5, 500);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], (500, 1500)); // samples 5-14 (ms 500-1400), ends at sample 15 (ms 1500)
    }

    #[test]
    fn test_summary() {
        let tl = Timeline::new(100);
        tl.event(EventKind::SyncStart);
        tl.event(EventKind::SyncEnd { duration_ms: 2.0 });
        tl.event(EventKind::SyncStart);
        tl.event(EventKind::SyncEnd { duration_ms: 4.0 });
        tl.event(EventKind::AnchorChanged { from: 10, to: 12 });
        tl.event(EventKind::Throttle { rank: 1 });

        let s = tl.summary();
        assert_eq!(s.sync_count, 2);
        assert!((s.avg_sync_ms - 3.0).abs() < 0.01);
        assert_eq!(s.anchor_change_count, 1);
        assert_eq!(s.throttle_count, 1);
    }

    #[test]
    fn test_json_export() {
        let tl = Timeline::new(100);
        {
            let mut samples = tl.samples.lock().unwrap();
            samples.push(TimelineSample {
                elapsed_ms: 0,
                cpu_util: 45.0,
                ram_used_bytes: 4_000_000_000,
                ram_total_bytes: 8_000_000_000,
                gpus: vec![GpuTimelineSample {
                    device: 0,
                    compute_util: 82,
                    vram_used_bytes: 2_000_000_000,
                    vram_allocated_bytes: 1_800_000_000,
                    vram_total_bytes: 8_000_000_000,
                }],
            });
        }
        tl.event(EventKind::SyncStart);

        // Just verify it doesn't panic
        let mut buf = String::new();
        let samples = tl.samples.lock().unwrap();
        let events = tl.events.lock().unwrap();
        write_samples_json(&mut buf, &samples);
        assert!(buf.contains("\"t\":0"));
        assert!(buf.contains("\"u\":82"));

        let mut buf2 = String::new();
        write_events_json(&mut buf2, &events);
        assert!(buf2.contains("\"sync_start\""));
    }

    #[test]
    fn test_subscribe_receives_batches() {
        // Use a short broadcast interval so we can test without sleeping long
        let tl = Timeline::with_intervals(50, 200);
        let rx = tl.subscribe();

        // Inject events before starting (should be included in first broadcast)
        tl.event(EventKind::EpochStart { epoch: 0 });

        tl.start();
        // Wait enough for at least one broadcast cycle
        std::thread::sleep(Duration::from_millis(350));
        tl.stop();

        // Should have received at least one broadcast batch
        let mut total_samples = 0;
        let mut total_events = 0;
        while let Ok(batch) = rx.try_recv() {
            total_samples += batch.samples.len();
            total_events += batch.events.len();
        }

        // Bar is "the broadcast wiring works end-to-end", not throughput.
        // Loaded CI hosts (WSL2/Docker, concurrent CUDA jobs) can starve
        // the poll thread enough that only one sample lands in the
        // 350ms window; that still proves subscribe/poll/broadcast
        // round-tripped a sample.
        assert!(total_samples >= 1, "expected samples, got {total_samples}");
        // The epoch event should have been broadcast
        assert!(total_events >= 1, "expected events, got {total_events}");
    }

    #[test]
    fn test_with_intervals() {
        let tl = Timeline::with_intervals(50, 500);
        assert_eq!(tl.poll_interval_ms, 50);
        assert_eq!(tl.broadcast_interval_ms, 500);
    }
}
