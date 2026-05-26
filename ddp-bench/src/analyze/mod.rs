//! Post-hoc run analysis: reads `training.log` for convergence data (loss,
//! eval, epoch timing) and optionally `timeline.json` for GPU utilization,
//! idle gap detection, and sync/ElChe instrumentation.

use std::path::Path;

// ---------------------------------------------------------------------------
// Timeline data (mirrors flodl::monitor::Timeline JSON format)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GpuSample {
    #[allow(dead_code)]
    pub device: u8,
    pub util: u8,
    /// CUDA caching allocator bytes (from "va" field).
    pub vram_allocated: u64,
    /// Physical VRAM used bytes (from "vu" field).
    #[allow(dead_code)]
    pub vram_used: u64,
    /// Total physical VRAM bytes (from "vt" field).
    pub vram_total: u64,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub t: u64,
    pub gpus: Vec<GpuSample>,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub t: u64,
    pub kind: EventKind,
}

#[derive(Debug, Clone)]
pub enum EventKind {
    EpochStart { epoch: usize },
    EpochEnd { epoch: usize, loss: f64, #[allow(dead_code)] lr: f64 },
    SyncStart,
    SyncEnd { ms: f64 },
    CpuAvgStart,
    CpuAvgEnd { ms: f64 },
    Anchor { #[allow(dead_code)] from: usize, #[allow(dead_code)] to: usize },
    Throttle { #[allow(dead_code)] rank: usize },
    /// LR-aware meta-controller nudged the anchor down (raw factor,
    /// pre/post anchor). The cycle's NET anchor delta still surfaces
    /// via `Anchor` from `finish_averaging_*`; `MetaNudge` isolates the
    /// meta's contribution.
    #[allow(dead_code)]
    MetaNudge { factor: f64, from: usize, to: usize },
    /// MSF per-AllReduce sample (passive observation, no behavior effect).
    /// Currently we only count these for the summary; per-event detail is
    /// kept on the JSON for downstream analysis tools.
    #[allow(dead_code)]
    Divergence {
        d_raw: f64,
        lambda_raw: Option<f64>,
        lambda_ema: Option<f64>,
        k_used: usize,
        k_max: usize,
        step: usize,
        deltas: Vec<f64>,
        /// L2 norm of the post-AllReduce consensus weights `||W̄||`. `None`
        /// for timelines emitted before the post_norm wiring landed.
        post_norm: Option<f64>,
        /// Per-rank pre-AllReduce L2 norm `||W_i||`. `None` for timelines
        /// emitted before the pre_norm wiring landed; combined with `deltas`
        /// and `post_norm` enables the cosine-similarity / magnitude-shift
        /// decomposition (MSF/SWA directional vs magnitude split).
        pre_norms: Option<Vec<f64>>,
        /// In-flight epoch at the time of this event. `None` for timelines
        /// emitted before the field was added; consumers fall back to
        /// `EpochEnd` timestamp lookup.
        epoch: Option<usize>,
    },
    /// MSF per-epoch aggregate snapshot.
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

/// Loaded timeline data for one run.
pub struct Timeline {
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
}

/// A detected GPU idle gap.
#[derive(Debug, Clone)]
pub struct IdleGap {
    pub device: u8,
    pub start_ms: u64,
    #[allow(dead_code)]
    pub end_ms: u64,
    pub duration_ms: u64,
    pub cause: IdleCause,
}

/// Classification of what caused an idle gap.
#[derive(Debug, Clone)]
pub enum IdleCause {
    /// Near an epoch boundary (epoch_end within window).
    EpochBoundary { epoch: usize },
    /// Overlaps with a sync event.
    Sync,
    /// Overlaps with CPU averaging.
    CpuAveraging,
    /// At the very start or end of training.
    Startup,
    /// No nearby event explains it.
    Unexplained,
}

impl std::fmt::Display for IdleCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdleCause::EpochBoundary { epoch } => write!(f, "epoch-boundary({})", epoch),
            IdleCause::Sync => write!(f, "sync"),
            IdleCause::CpuAveraging => write!(f, "cpu-avg"),
            IdleCause::Startup => write!(f, "startup"),
            IdleCause::Unexplained => write!(f, "unexplained"),
        }
    }
}

/// Per-GPU VRAM statistics.
#[derive(Debug, Clone, Default)]
pub struct VramStats {
    #[allow(dead_code)]
    pub device: u8,
    /// Peak VRAM allocated (bytes) during the run.
    pub peak_allocated: u64,
    /// Mean VRAM allocated (bytes) during the run.
    pub mean_allocated: u64,
    /// Total VRAM on this device (bytes).
    pub total: u64,
}

/// Per-epoch convergence data.
#[derive(Debug, Clone)]
pub struct EpochData {
    #[allow(dead_code)]
    pub epoch: usize,
    /// Loss at end of this epoch.
    pub loss: f64,
    /// Eval metric (accuracy, perplexity, etc.) if available.
    #[allow(dead_code)]
    pub eval: Option<f64>,
    /// Wall-clock span for this epoch (ms).
    #[allow(dead_code)]
    pub wall_ms: f64,
}

/// Aggregate analysis of a single run.
#[derive(Debug, Clone)]
pub struct RunAnalysis {
    pub model: String,
    pub mode: String,
    pub total_ms: u64,
    #[allow(dead_code)]
    pub n_epochs: usize,
    pub final_loss: f64,
    /// Final eval metric (from `final eval=X.XXXX` or last per-epoch eval).
    pub final_eval: Option<f64>,
    /// Per-epoch convergence trajectory.
    pub epoch_data: Vec<EpochData>,
    /// Per-GPU active percentage.
    pub gpu_active_pct: Vec<f64>,
    /// Sync event count.
    pub sync_count: usize,
    /// Average sync duration (ms).
    pub avg_sync_ms: f64,
    /// Total sync time (ms).
    #[allow(dead_code)]
    pub total_sync_ms: f64,
    /// CPU averaging count and average.
    pub cpu_avg_count: usize,
    pub avg_cpu_avg_ms: f64,
    /// Anchor changes.
    pub anchor_changes: usize,
    /// Throttle events.
    pub throttle_count: usize,
    /// All detected idle gaps (multi-second focus).
    pub idle_gaps: Vec<IdleGap>,
    /// Total idle time per GPU by cause (ms).
    pub idle_by_cause: Vec<IdleByCause>,
    /// Per-GPU VRAM statistics.
    pub vram_stats: Vec<VramStats>,
    /// Total epoch overlap time (ms). Nonzero when streaming epochs overlap.
    pub epoch_overlap_ms: f64,
    /// Sync intervals: time between consecutive SyncEnd events (ms).
    pub sync_intervals: Vec<f64>,
    /// Training-only wall time (ms). Set for `run_baseline_solo` runs from
    /// the `# train_only:` log footer; lets the speedup table compare DDP
    /// against solo's training-only wall time, excluding solo's per-epoch
    /// eval cost (DDP only pays one final eval).
    pub train_only_ms: Option<u64>,
    /// Per-rank averages across the run (from `per-rank:` log lines).
    /// Empty for solo and single-rank runs.
    pub per_rank_avg: Vec<PerRankAvg>,
    /// MSF passive observation data (lambda_hat, per-epoch aggregates,
    /// phase-transition candidates). Empty for runs predating MSF logging
    /// or for modes that produce no AllReduce events (Solo, Sync without
    /// divergence reports).
    pub msf: MsfAnalysis,
}

mod timeline;
mod log;
mod fit;
mod msf;

pub use timeline::load_timeline;
pub use log::{parse_training_log, apply_training_log};

use msf::{build_msf_analysis, MsfAnalysis};

/// Per-rank stats averaged across the run.
#[derive(Debug, Clone)]
pub struct PerRankAvg {
    pub rank: usize,
    pub device: u8,
    /// Mean batch_share across observed epochs (0..1).
    pub batch_share: f64,
    /// Mean throughput in samples/ms.
    pub throughput: f64,
}

/// Total idle time for one GPU broken down by cause.
#[derive(Debug, Clone, Default)]
pub struct IdleByCause {
    pub device: u8,
    pub epoch_boundary_ms: f64,
    pub sync_ms: f64,
    pub cpu_avg_ms: f64,
    pub startup_ms: f64,
    pub unexplained_ms: f64,
    pub total_ms: f64,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Minimum idle gap duration to report (ms).
const MIN_GAP_MS: u64 = 500;

/// Window around an idle gap to search for correlated events (ms).
const CORRELATION_WINDOW_MS: u64 = 500;

/// Analyze a loaded timeline.
pub fn analyze(model: &str, mode: &str, tl: &Timeline) -> RunAnalysis {
    let total_ms = tl.samples.last().map(|s| s.t).unwrap_or(0);
    let n_gpus = tl.samples.first().map(|s| s.gpus.len()).unwrap_or(0);

    // GPU active %
    let sample_count = tl.samples.len();
    let mut gpu_active_pct = vec![0.0; n_gpus];
    if sample_count > 0 {
        for s in &tl.samples {
            for (i, g) in s.gpus.iter().enumerate() {
                if g.util >= 5 {
                    gpu_active_pct[i] += 1.0;
                }
            }
        }
        for v in &mut gpu_active_pct {
            *v = *v / sample_count as f64 * 100.0;
        }
    }

    // VRAM statistics per GPU
    let mut vram_stats: Vec<VramStats> = (0..n_gpus)
        .map(|i| VramStats { device: i as u8, ..Default::default() })
        .collect();
    if sample_count > 0 {
        let mut vram_sums: Vec<u64> = vec![0; n_gpus];
        for s in &tl.samples {
            for (i, g) in s.gpus.iter().enumerate() {
                if g.vram_allocated > vram_stats[i].peak_allocated {
                    vram_stats[i].peak_allocated = g.vram_allocated;
                }
                vram_sums[i] += g.vram_allocated;
                if g.vram_total > 0 {
                    vram_stats[i].total = g.vram_total;
                }
            }
        }
        for (vs, sum) in vram_stats.iter_mut().zip(vram_sums.iter()) {
            vs.mean_allocated = sum / sample_count as u64;
        }
    }

    // Sync stats
    let mut sync_count = 0usize;
    let mut sync_total_ms = 0.0f64;
    let mut cpu_avg_count = 0usize;
    let mut cpu_avg_total_ms = 0.0f64;
    let mut anchor_changes = 0usize;
    let mut throttle_count = 0usize;

    // Track sync intervals (time between consecutive SyncEnd events)
    let mut sync_end_times: Vec<u64> = Vec::new();

    for e in &tl.events {
        match &e.kind {
            EventKind::SyncStart => sync_count += 1,
            EventKind::SyncEnd { ms } => {
                sync_total_ms += ms;
                sync_end_times.push(e.t);
            }
            EventKind::CpuAvgStart => cpu_avg_count += 1,
            EventKind::CpuAvgEnd { ms } => cpu_avg_total_ms += ms,
            EventKind::Anchor { .. } => anchor_changes += 1,
            EventKind::Throttle { .. } => throttle_count += 1,
            _ => {}
        }
    }

    let sync_intervals: Vec<f64> = sync_end_times.windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();

    // Epoch info: collect all start/end events per epoch
    let mut epoch_ends: Vec<(usize, f64, u64)> = Vec::new(); // (epoch, loss, t)
    let mut epoch_starts: Vec<(usize, u64)> = Vec::new();
    for e in &tl.events {
        match &e.kind {
            EventKind::EpochEnd { epoch, loss, .. } => epoch_ends.push((*epoch, *loss, e.t)),
            EventKind::EpochStart { epoch } => epoch_starts.push((*epoch, e.t)),
            _ => {}
        }
    }

    let final_loss = epoch_ends.last().map(|(_, l, _)| *l).unwrap_or(0.0);

    // Per-epoch data: wall time + loss trajectory
    let max_epoch = epoch_ends.iter().map(|(e, _, _)| *e).max().unwrap_or(0);
    let n_epochs = if epoch_ends.is_empty() { 0 } else { max_epoch + 1 };
    let mut epoch_data = Vec::with_capacity(n_epochs);

    // Collect epoch spans for overlap detection
    let mut epoch_spans: Vec<(u64, u64)> = Vec::with_capacity(n_epochs); // (min_start, max_end)

    for ep in 0..n_epochs {
        let starts: Vec<u64> = epoch_starts.iter()
            .filter(|(e, _)| *e == ep)
            .map(|(_, t)| *t)
            .collect();
        let ends: Vec<u64> = epoch_ends.iter()
            .filter(|(e, _, _)| *e == ep)
            .map(|(_, _, t)| *t)
            .collect();
        // Loss: use the last EpochEnd for this epoch (most complete)
        let loss = epoch_ends.iter()
            .rfind(|(e, _, _)| *e == ep)
            .map(|(_, l, _)| *l)
            .unwrap_or(0.0);

        let wall_ms = match (starts.iter().min(), ends.iter().max()) {
            (Some(&s), Some(&e)) => {
                epoch_spans.push((s, e));
                (e - s) as f64
            }
            _ => {
                epoch_spans.push((0, 0));
                0.0
            }
        };

        epoch_data.push(EpochData { epoch: ep, loss, eval: None, wall_ms });
    }

    // Epoch overlap: sum of overlapping time between consecutive epoch spans
    let mut epoch_overlap_ms = 0.0f64;
    for pair in epoch_spans.windows(2) {
        let (_, prev_end) = pair[0];
        let (next_start, _) = pair[1];
        if prev_end > next_start {
            epoch_overlap_ms += (prev_end - next_start) as f64;
        }
    }

    // Idle gap detection per GPU
    let mut all_gaps: Vec<IdleGap> = Vec::new();
    let mut idle_by_cause: Vec<IdleByCause> = (0..n_gpus as u8)
        .map(|d| IdleByCause { device: d, ..Default::default() })
        .collect();

    // First training event timestamp (skip startup idle)
    let first_training_t = tl.events.first().map(|e| e.t).unwrap_or(0);

    for (gpu_idx, idle) in idle_by_cause.iter_mut().enumerate() {
        let device = gpu_idx as u8;
        let mut gap_start: Option<u64> = None;

        for s in &tl.samples {
            let util = s.gpus.get(gpu_idx).map(|g| g.util).unwrap_or(100);

            if util < 5 {
                if gap_start.is_none() {
                    gap_start = Some(s.t);
                }
            } else if let Some(start) = gap_start.take() {
                let duration = s.t.saturating_sub(start);
                if duration >= MIN_GAP_MS {
                    let cause = classify_gap(start, s.t, first_training_t, &tl.events);
                    accumulate_cause(idle, &cause, duration as f64);
                    all_gaps.push(IdleGap {
                        device,
                        start_ms: start,
                        end_ms: s.t,
                        duration_ms: duration,
                        cause,
                    });
                }
            }
        }

        // Trailing gap
        if let Some(start) = gap_start
            && let Some(last) = tl.samples.last()
        {
            let duration = last.t.saturating_sub(start);
            if duration >= MIN_GAP_MS {
                let cause = classify_gap(start, last.t, first_training_t, &tl.events);
                accumulate_cause(idle, &cause, duration as f64);
                all_gaps.push(IdleGap {
                    device,
                    start_ms: start,
                    end_ms: last.t,
                    duration_ms: duration,
                    cause,
                });
            }
        }

        // Compute total
        idle.total_ms = idle.epoch_boundary_ms
            + idle.sync_ms
            + idle.cpu_avg_ms
            + idle.startup_ms
            + idle.unexplained_ms;
    }

    let msf = build_msf_analysis(&tl.events);

    RunAnalysis {
        model: model.to_string(),
        mode: mode.to_string(),
        total_ms,
        n_epochs,
        final_loss,
        final_eval: None,
        epoch_data,
        gpu_active_pct,
        sync_count,
        avg_sync_ms: if sync_count > 0 { sync_total_ms / sync_count as f64 } else { 0.0 },
        total_sync_ms: sync_total_ms,
        cpu_avg_count,
        avg_cpu_avg_ms: if cpu_avg_count > 0 { cpu_avg_total_ms / cpu_avg_count as f64 } else { 0.0 },
        anchor_changes,
        throttle_count,
        idle_gaps: all_gaps,
        idle_by_cause,
        vram_stats,
        epoch_overlap_ms,
        sync_intervals,
        train_only_ms: None,
        per_rank_avg: Vec::new(),
        msf,
    }
}

/// Create a minimal RunAnalysis with no timeline data.
/// Training log data is applied afterwards via `apply_training_log`.
pub fn empty_analysis(model: &str, mode: &str) -> RunAnalysis {
    RunAnalysis {
        model: model.to_string(),
        mode: mode.to_string(),
        total_ms: 0,
        n_epochs: 0,
        final_loss: 0.0,
        final_eval: None,
        epoch_data: Vec::new(),
        gpu_active_pct: Vec::new(),
        sync_count: 0,
        avg_sync_ms: 0.0,
        total_sync_ms: 0.0,
        cpu_avg_count: 0,
        avg_cpu_avg_ms: 0.0,
        anchor_changes: 0,
        throttle_count: 0,
        idle_gaps: Vec::new(),
        idle_by_cause: Vec::new(),
        vram_stats: Vec::new(),
        epoch_overlap_ms: 0.0,
        sync_intervals: Vec::new(),
        train_only_ms: None,
        per_rank_avg: Vec::new(),
        msf: MsfAnalysis::default(),
    }
}

/// Classify an idle gap by the nearest event.
fn classify_gap(start: u64, end: u64, first_training_t: u64, events: &[Event]) -> IdleCause {
    // Startup: gap starts before first training event
    if start <= first_training_t {
        return IdleCause::Startup;
    }

    let window_start = start.saturating_sub(CORRELATION_WINDOW_MS);
    let window_end = end + CORRELATION_WINDOW_MS;

    // Check for epoch boundaries first (most interesting)
    for e in events {
        if e.t < window_start || e.t > window_end {
            continue;
        }
        if let EventKind::EpochEnd { epoch, .. } = &e.kind {
            return IdleCause::EpochBoundary { epoch: *epoch };
        }
    }

    // Check for CPU averaging overlap
    for e in events {
        if e.t < window_start || e.t > window_end {
            continue;
        }
        if matches!(e.kind, EventKind::CpuAvgStart | EventKind::CpuAvgEnd { .. }) {
            return IdleCause::CpuAveraging;
        }
    }

    // Check for sync overlap
    for e in events {
        if e.t < window_start || e.t > window_end {
            continue;
        }
        if matches!(e.kind, EventKind::SyncStart | EventKind::SyncEnd { .. }) {
            return IdleCause::Sync;
        }
    }

    IdleCause::Unexplained
}

fn accumulate_cause(by_cause: &mut IdleByCause, cause: &IdleCause, ms: f64) {
    match cause {
        IdleCause::EpochBoundary { .. } => by_cause.epoch_boundary_ms += ms,
        IdleCause::Sync => by_cause.sync_ms += ms,
        IdleCause::CpuAveraging => by_cause.cpu_avg_ms += ms,
        IdleCause::Startup => by_cause.startup_ms += ms,
        IdleCause::Unexplained => by_cause.unexplained_ms += ms,
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover available runs in the output directory.
/// Returns (model, mode) pairs sorted by model then mode.
/// A run is valid if it has a `training.log` (required for loss data).
pub fn discover_runs(output_dir: &str) -> Vec<(String, String)> {
    let mut runs = Vec::new();
    let base = Path::new(output_dir);
    if !base.is_dir() {
        return runs;
    }

    if let Ok(models) = std::fs::read_dir(base) {
        for model_entry in models.flatten() {
            if !model_entry.path().is_dir() {
                continue;
            }
            let model = model_entry.file_name().to_string_lossy().to_string();
            if let Ok(modes) = std::fs::read_dir(model_entry.path()) {
                for mode_entry in modes.flatten() {
                    let log_path = mode_entry.path().join("training.log");
                    if log_path.exists() {
                        let mode = mode_entry.file_name().to_string_lossy().to_string();
                        runs.push((model.clone(), mode));
                    }
                }
            }
        }
    }

    runs.sort();
    runs
}
