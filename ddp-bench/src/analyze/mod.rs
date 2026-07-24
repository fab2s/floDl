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

/// Per-GPU slice of a rank-reported sample. `Option` mirrors the JSON:
/// the producer omits what the rank did not sample — absent ≠ zero.
#[derive(Debug, Clone)]
pub struct RankGpuSample {
    /// Physical device index ON THE RANK'S HOST (collides across hosts;
    /// only meaningful together with the parent sample's `host`).
    pub device: u8,
    pub util: Option<f64>,
    /// The rank process's CUDA caching allocator bytes ("va").
    pub vram_allocated: Option<u64>,
    pub vram_total: Option<u64>,
}

/// A rank-reported resource sample from the timeline's `rank_samples`
/// array (cluster runs). Rides the metrics wire at reduce-window
/// cadence — sparse against the local poller's `samples`.
#[derive(Debug, Clone)]
pub struct RankSample {
    #[allow(dead_code)]
    pub t: u64,
    pub rank: usize,
    /// World-map host name; empty = depositor had no topology.
    pub host: String,
    /// Host-level CPU/RAM (parsed for completeness; not rendered yet).
    #[allow(dead_code)]
    pub cpu: Option<f64>,
    #[allow(dead_code)]
    pub ram_used: Option<u64>,
    #[allow(dead_code)]
    pub ram_total: Option<u64>,
    pub gpus: Vec<RankGpuSample>,
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
}

/// Loaded timeline data for one run.
pub struct Timeline {
    /// Controller host name (world-map name, stamped by the cluster
    /// launcher). `None` on solo runs and files predating the stamp.
    pub host: Option<String>,
    pub samples: Vec<Sample>,
    pub events: Vec<Event>,
    /// Rank-reported host-qualified samples (cluster runs; empty otherwise).
    pub rank_samples: Vec<RankSample>,
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
    /// Per-GPU active percentage, parallel to [`Self::gpu_devices`].
    pub gpu_active_pct: Vec<f64>,
    /// Host-physical device ids the timeline sampled (sorted). Parallel to
    /// `gpu_active_pct`; report columns label by these ids, never by slot.
    pub gpu_devices: Vec<u8>,
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
    /// Controller host name from the timeline stamp. `None` on solo
    /// runs and timelines predating the stamp — then no rank-sample
    /// host can be identified as "already covered by the local poller".
    pub controller_host: Option<String>,
    /// Per-(rank, device) aggregates over the rank-reported sparse
    /// samples. Empty outside cluster runs.
    pub rank_res: Vec<RankResStats>,
}

/// One rank's GPU resource aggregate across the run, from the sparse
/// rank-reported samples (~one per reduce window — direction, not
/// precision; `n_samples` says how thin the evidence is).
#[derive(Debug, Clone)]
pub struct RankResStats {
    pub rank: usize,
    /// Host the rank runs on (world-map name; empty = unknown topology).
    pub host: String,
    /// Physical device index on that host.
    pub device: u8,
    /// Mean compute utilization over util-carrying samples. NVML util
    /// is a ~1s rolling mean, so this — not a >=5% active indicator,
    /// which saturates at sparse cadence — is the sparse duty-cycle
    /// estimator.
    pub mean_util: Option<f64>,
    /// Peak / mean CUDA-allocator bytes over va-carrying samples. In
    /// multi-process runs this is the only true allocator reading —
    /// the controller's local poller sees its own (empty) allocator.
    pub peak_allocated: Option<u64>,
    pub mean_allocated: Option<u64>,
    /// Total VRAM on the device (last seen).
    pub vram_total: Option<u64>,
    /// Rank samples carrying a GPU slice for this device.
    pub n_samples: usize,
}

mod timeline;
pub mod log;

pub use timeline::load_timeline;
pub use log::{parse_training_log, apply_training_log};

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

    // Physical device ids present in the samples. The sampler records the
    // host-physical index (`d` field), and the array holds only the GPUs
    // this process monitors — indexing stats by array POSITION misattributes
    // a run on device 1 to "GPU0" (the historical solo-1 bug). Everything
    // below keys by device id via this sorted union.
    let mut gpu_devices: Vec<u8> = Vec::new();
    for s in &tl.samples {
        for g in &s.gpus {
            if !gpu_devices.contains(&g.device) {
                gpu_devices.push(g.device);
            }
        }
    }
    gpu_devices.sort_unstable();
    let n_gpus = gpu_devices.len();
    let slot_of = |device: u8| gpu_devices.iter().position(|d| *d == device);

    // GPU active %
    let sample_count = tl.samples.len();
    let mut gpu_active_pct = vec![0.0; n_gpus];
    if sample_count > 0 {
        for s in &tl.samples {
            for g in &s.gpus {
                if g.util >= 5
                    && let Some(i) = slot_of(g.device)
                {
                    gpu_active_pct[i] += 1.0;
                }
            }
        }
        for v in &mut gpu_active_pct {
            *v = *v / sample_count as f64 * 100.0;
        }
    }

    // VRAM statistics per GPU
    let mut vram_stats: Vec<VramStats> = gpu_devices.iter()
        .map(|d| VramStats { device: *d, ..Default::default() })
        .collect();
    if sample_count > 0 {
        let mut vram_sums: Vec<u64> = vec![0; n_gpus];
        for s in &tl.samples {
            for g in &s.gpus {
                let Some(i) = slot_of(g.device) else { continue };
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
    let mut idle_by_cause: Vec<IdleByCause> = gpu_devices.iter()
        .map(|d| IdleByCause { device: *d, ..Default::default() })
        .collect();

    // First training event timestamp (skip startup idle)
    let first_training_t = tl.events.first().map(|e| e.t).unwrap_or(0);

    for idle in idle_by_cause.iter_mut() {
        let device = idle.device;
        let mut gap_start: Option<u64> = None;

        for s in &tl.samples {
            let util = s.gpus.iter()
                .find(|g| g.device == device)
                .map(|g| g.util)
                .unwrap_or(100);

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

    let rank_res = aggregate_rank_samples(&tl.rank_samples);

    RunAnalysis {
        model: model.to_string(),
        mode: mode.to_string(),
        total_ms,
        n_epochs,
        final_loss,
        final_eval: None,
        epoch_data,
        gpu_active_pct,
        gpu_devices,
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
        controller_host: tl.host.clone(),
        rank_res,
    }
}

/// Aggregate the rank-reported sparse samples per (rank, device).
/// Only `Some` fields count — an unsampled tick must not drag a mean
/// toward zero or fake an idle GPU.
fn aggregate_rank_samples(samples: &[RankSample]) -> Vec<RankResStats> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Acc {
        host: String,
        n: usize,
        util_n: usize,
        util_sum: f64,
        va_n: u64,
        va_sum: u64,
        va_peak: u64,
        vt: Option<u64>,
    }

    let mut acc: BTreeMap<(usize, u8), Acc> = BTreeMap::new();
    for s in samples {
        for g in &s.gpus {
            let e = acc.entry((s.rank, g.device)).or_default();
            e.host = s.host.clone();
            e.n += 1;
            if let Some(u) = g.util {
                e.util_n += 1;
                e.util_sum += u;
            }
            if let Some(va) = g.vram_allocated {
                e.va_n += 1;
                e.va_sum += va;
                e.va_peak = e.va_peak.max(va);
            }
            if let Some(vt) = g.vram_total {
                e.vt = Some(vt);
            }
        }
    }

    acc.into_iter()
        .map(|((rank, device), a)| RankResStats {
            rank,
            host: a.host,
            device,
            mean_util: (a.util_n > 0).then(|| a.util_sum / a.util_n as f64),
            peak_allocated: (a.va_n > 0).then_some(a.va_peak),
            mean_allocated: (a.va_n > 0).then(|| a.va_sum / a.va_n),
            vram_total: a.vt,
            n_samples: a.n,
        })
        .collect()
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
        gpu_devices: Vec::new(),
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
        controller_host: None,
        rank_res: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(device: u8, util: Option<f64>, va: Option<u64>, vt: Option<u64>) -> RankGpuSample {
        RankGpuSample { device, util, vram_allocated: va, vram_total: vt }
    }

    fn rs(rank: usize, host: &str, gpus: Vec<RankGpuSample>) -> RankSample {
        RankSample {
            t: 0,
            rank,
            host: host.to_string(),
            cpu: None,
            ram_used: None,
            ram_total: None,
            gpus,
        }
    }

    /// Aggregation keys by (rank, device), averages only over the ticks
    /// that actually carried a value, and tracks the allocator peak.
    #[test]
    fn aggregate_keys_and_means() {
        let samples = vec![
            rs(0, "exa", vec![gpu(0, Some(80.0), Some(1000), Some(16000))]),
            rs(0, "exa", vec![gpu(0, Some(40.0), Some(2000), Some(16000))]),
            rs(1, "pascal", vec![gpu(1, Some(90.0), Some(3000), Some(6000))]),
        ];
        let out = aggregate_rank_samples(&samples);
        assert_eq!(out.len(), 2);

        let r0 = out.iter().find(|s| s.rank == 0).unwrap();
        assert_eq!(r0.host, "exa");
        assert_eq!(r0.device, 0);
        assert_eq!(r0.n_samples, 2);
        assert_eq!(r0.mean_util, Some(60.0));
        assert_eq!(r0.peak_allocated, Some(2000));
        assert_eq!(r0.mean_allocated, Some(1500));

        let r1 = out.iter().find(|s| s.rank == 1).unwrap();
        assert_eq!(r1.device, 1);
        assert_eq!(r1.mean_util, Some(90.0));
    }

    /// A missing field must never read as zero: an unsampled tick drops
    /// out of the mean instead of dragging it toward 0. `None` for a
    /// field that was never present on any tick.
    #[test]
    fn aggregate_absent_is_not_zero() {
        let samples = vec![
            // util present, va absent
            rs(0, "exa", vec![gpu(0, Some(50.0), None, None)]),
            // util absent, va present
            rs(0, "exa", vec![gpu(0, None, Some(4000), Some(16000))]),
        ];
        let out = aggregate_rank_samples(&samples);
        let r0 = &out[0];
        assert_eq!(r0.n_samples, 2);
        // mean_util averages the single util-carrying tick, not /2.
        assert_eq!(r0.mean_util, Some(50.0));
        // va from the single va-carrying tick.
        assert_eq!(r0.peak_allocated, Some(4000));
        assert_eq!(r0.mean_allocated, Some(4000));
        assert_eq!(r0.vram_total, Some(16000));
    }

    /// Empty rank_samples aggregates to nothing (solo / pre-cluster).
    #[test]
    fn aggregate_empty() {
        assert!(aggregate_rank_samples(&[]).is_empty());
    }
}
