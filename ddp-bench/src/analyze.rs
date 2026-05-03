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
        /// L2 norm of the post-AllReduce consensus weights. Only emitted by
        /// CPU averaging path (NCCL v1 doesn't compute it).
        post_norm: Option<f64>,
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

/// Per-epoch MSF aggregate (mirrors EventKind::DivergenceEpoch).
#[derive(Debug, Clone)]
pub struct MsfEpoch {
    pub epoch: usize,
    pub sync_count: usize,
    pub d_min: f64,
    pub d_max: f64,
    pub d_mean: f64,
    pub d_at_epoch_end: f64,
    pub k_at_epoch_end: usize,
    pub lambda_min: Option<f64>,
    pub lambda_max: Option<f64>,
    pub lambda_mean: Option<f64>,
    pub lambda_ema_at_epoch_end: Option<f64>,
    /// Learning rate at end of this epoch (from the matching EpochEnd event).
    /// `None` for runs predating per-event LR logging.
    pub lr: Option<f64>,
}

/// A detected phase-transition candidate (heuristic threshold).
#[derive(Debug, Clone)]
pub struct MsfPhaseCandidate {
    pub epoch: usize,
    /// `lambda_min` at the candidate event (most-negative single sample).
    pub lambda_min: f64,
    /// `d_at_epoch_end` (where the system landed after the transition).
    pub d_end: f64,
    /// Ratio `d_at_epoch_end / d_at_previous_epoch_end` (smaller = bigger collapse).
    pub d_ratio: f64,
}

/// Per-rank D distribution + per-rank lambda estimates.
///
/// Rank 0 is conventionally the fast GPU under heterogeneous dispatch (gets
/// the largest batch_share). Backend-dependent: NCCL exposes per-rank-step
/// asymmetry in D_t (rank 0 wins max-D race ~57% of events on heterogeneous
/// 3-GPU rigs), CPU averaging hides it (~33% per rank).
#[derive(Debug, Clone)]
pub struct MsfPerRank {
    pub rank: usize,
    pub n: usize,
    pub d_mean: f64,
    pub d_sd: f64,
    pub d_min: f64,
    pub d_max: f64,
    /// Fraction of events where this rank had the highest delta across ranks.
    /// Uniform = 1/world_size. Higher = this rank dominates the max-D race.
    pub win_pct: f64,
    pub lambda_mean: f64,
    pub lambda_sd: f64,
}

/// Comparison of the existing convergence guard's "3 consecutive D rises
/// above threshold" rule vs an MSF-style guard firing on sustained positive
/// `λ_ema` (R5 in the design doc).
///
/// Both guards are simulated post-hoc against the per-epoch `div_epoch`
/// series. The comparison answers: "would an MSF-based guard fire at the
/// same epochs the current heuristic does?" and "do they catch different
/// regime transitions?"
#[derive(Debug, Clone, Default)]
pub struct MsfGuardComparison {
    /// Epochs where the current guard's "3 rises above threshold" rule fires.
    pub current_fires: Vec<usize>,
    /// Epochs where the MSF guard's "λ_ema sustained > λ_threshold" rule fires.
    pub msf_fires: Vec<usize>,
    /// Epochs where both rules fire (intersection).
    pub both: Vec<usize>,
    /// Epochs where only the current guard fires.
    pub current_only: Vec<usize>,
    /// Epochs where only the MSF guard fires.
    pub msf_only: Vec<usize>,
}

/// Longitudinal meta-oscillator velocity stats: per-event consensus
/// magnitude motion `|Δ||W̄|||/||W̄||_prev` aggregated across the run.
///
/// Independent of D_t (transversal): tracks LR schedule + gradient size, not
/// inter-rank synchronization. Phase-transition signal complementary to λ̂.
#[derive(Debug, Clone, Default)]
pub struct MsfLongitudinal {
    /// Number of events with both prev_post_norm and post_norm available.
    pub n: usize,
    /// `||W̄||` summary statistics across the run.
    pub post_norm_min: f64,
    pub post_norm_max: f64,
    pub post_norm_mean: f64,
    /// Per-event velocity `|Δ||W̄|||/||W̄||_prev` summary statistics.
    #[allow(dead_code)]
    pub velocity_min: f64,
    pub velocity_max: f64,
    pub velocity_mean: f64,
    pub velocity_sd: f64,
}

/// R1 informal-test result: log(d_max) vs cumulative step linear fit within
/// a single LR window (auto-detected from EpochEnd LR transitions).
///
/// Slope is in units of ln(D)/step. R² is the coefficient of determination.
/// R1 predicts log(D_t) approximately linear in step within stable phases;
/// high R² supports R1, low R² either falsifies or indicates noise-dominated
/// equilibria where the marginal-stability prediction (slope ≈ 0) is correct
/// but the variance can't be fit against.
#[derive(Debug, Clone)]
pub struct MsfLrWindowFit {
    pub lr: f64,
    pub epoch_start: usize,
    pub epoch_end: usize,
    pub n_events: usize,
    pub step_min: usize,
    pub step_max: usize,
    pub slope_per_step: f64,
    pub r2: f64,
}

/// Aggregate MSF analysis for a single run.
#[derive(Debug, Clone, Default)]
pub struct MsfAnalysis {
    /// Number of `Divergence` (per-AllReduce) events seen.
    pub div_event_count: usize,
    /// Per-epoch aggregates, in epoch order.
    pub epochs: Vec<MsfEpoch>,
    /// Heuristic phase-transition candidates: epochs where `lambda_min` is
    /// strongly negative AND `d_end / prev_d_end` shows a sharp collapse.
    pub phase_candidates: Vec<MsfPhaseCandidate>,
    /// Per-rank D distribution stats. Empty for runs without per-rank deltas.
    pub per_rank: Vec<MsfPerRank>,
    /// Pairwise Pearson correlation of D trajectories: list of `((i, j), r)`
    /// for `i < j`. Values consistently > 0.99 across modes empirically —
    /// supports the meta-oscillator framing (ranks are coupled, not
    /// independent oscillators).
    pub rank_correlations: Vec<((usize, usize), f64)>,
    /// R1 informal linear fits per auto-detected LR window. Empty when no
    /// EpochEnd events carry LR (runs predating per-event LR logging).
    pub lr_window_fits: Vec<MsfLrWindowFit>,
    /// Longitudinal meta-velocity (consensus magnitude motion). `None` when
    /// no `Divergence` event carries `post_norm` (runs predating post_norm
    /// wiring, or backends that don't compute it).
    pub longitudinal: Option<MsfLongitudinal>,
    /// Guard simulator comparison: current guard fires vs MSF-style guard
    /// fires on the same `div_epoch` series.
    pub guard_comparison: MsfGuardComparison,
}

impl MsfAnalysis {
    /// Whether any MSF data was captured for this run.
    pub fn has_data(&self) -> bool {
        !self.epochs.is_empty()
    }
}

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

/// Load a timeline from a JSON file.
pub fn load_timeline(path: &Path) -> Result<Timeline, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let val: serde_json::Value = serde_json::from_str(&data)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?;

    let samples = parse_samples(&val["samples"])?;
    let events = parse_events(&val["events"])?;

    Ok(Timeline { samples, events })
}

fn parse_samples(val: &serde_json::Value) -> Result<Vec<Sample>, String> {
    let arr = val.as_array().ok_or("samples is not an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let t = item["t"].as_u64().unwrap_or(0);
        let gpus = if let Some(gpu_arr) = item["gpus"].as_array() {
            gpu_arr
                .iter()
                .map(|g| GpuSample {
                    device: g["d"].as_u64().unwrap_or(0) as u8,
                    util: g["u"].as_u64().unwrap_or(0) as u8,
                    vram_allocated: g["va"].as_u64().unwrap_or(0),
                    vram_used: g["vu"].as_u64().unwrap_or(0),
                    vram_total: g["vt"].as_u64().unwrap_or(0),
                })
                .collect()
        } else {
            Vec::new()
        };
        out.push(Sample { t, gpus });
    }
    Ok(out)
}

fn parse_events(val: &serde_json::Value) -> Result<Vec<Event>, String> {
    let arr = val.as_array().ok_or("events is not an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let t = item["t"].as_u64().unwrap_or(0);
        let kind = match item["k"].as_str().unwrap_or("") {
            "epoch_start" => EventKind::EpochStart {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
            },
            "epoch_end" => EventKind::EpochEnd {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
                loss: item["loss"].as_f64().unwrap_or(0.0),
                lr: item["lr"].as_f64().unwrap_or(f64::NAN),
            },
            "sync_start" => EventKind::SyncStart,
            "sync_end" => EventKind::SyncEnd {
                ms: item["ms"].as_f64().unwrap_or(0.0),
            },
            "cpu_avg_start" => EventKind::CpuAvgStart,
            "cpu_avg_end" => EventKind::CpuAvgEnd {
                ms: item["ms"].as_f64().unwrap_or(0.0),
            },
            "anchor" => EventKind::Anchor {
                from: item["from"].as_u64().unwrap_or(0) as usize,
                to: item["to"].as_u64().unwrap_or(0) as usize,
            },
            "throttle" => EventKind::Throttle {
                rank: item["rank"].as_u64().unwrap_or(0) as usize,
            },
            "div" => {
                let deltas = item["deltas"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|v| v.as_f64().unwrap_or(0.0))
                            .collect::<Vec<f64>>()
                    })
                    .unwrap_or_default();
                EventKind::Divergence {
                    d_raw: item["d"].as_f64().unwrap_or(0.0),
                    lambda_raw: item["lambda"].as_f64(),
                    lambda_ema: item["lambda_ema"].as_f64(),
                    k_used: item["k_used"].as_u64().unwrap_or(0) as usize,
                    k_max: item["k_max"].as_u64().unwrap_or(0) as usize,
                    step: item["step"].as_u64().unwrap_or(0) as usize,
                    deltas,
                    post_norm: item["post_norm"].as_f64(),
                }
            }
            "div_epoch" => EventKind::DivergenceEpoch {
                epoch: item["epoch"].as_u64().unwrap_or(0) as usize,
                sync_count: item["syncs"].as_u64().unwrap_or(0) as usize,
                d_min: item["d_min"].as_f64().unwrap_or(0.0),
                d_max: item["d_max"].as_f64().unwrap_or(0.0),
                d_mean: item["d_mean"].as_f64().unwrap_or(0.0),
                lambda_min: item["lambda_min"].as_f64(),
                lambda_max: item["lambda_max"].as_f64(),
                lambda_mean: item["lambda_mean"].as_f64(),
                lambda_ema_at_epoch_end: item["lambda_ema_end"].as_f64(),
                d_at_epoch_end: item["d_end"].as_f64().unwrap_or(0.0),
                k_at_epoch_end: item["k_end"].as_u64().unwrap_or(0) as usize,
            },
            _ => continue, // skip unknown
        };
        out.push(Event { t, kind });
    }
    Ok(out)
}

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

/// LR change above this fraction starts a new auto-detected window.
/// Step-decays jump 10x and trigger cleanly; cosine schedules accumulate
/// into ~5%-step buckets which is acceptable resolution for analysis.
const LR_WINDOW_CHANGE_FRAC: f64 = 0.05;

/// Compute log(d_max) vs cumulative step OLS within a (start_epoch, end_epoch)
/// range. Filters non-positive d_max (log undefined). Returns `None` if too
/// few finite points (n < 5).
fn fit_lr_window(
    events: &[Event],
    epoch_step_ranges: &[(usize, usize, usize)], // (epoch, step_first, step_last)
    start_ep: usize,
    end_ep: usize,
) -> Option<(usize, usize, usize, f64, f64)> {
    let mut step_lo = usize::MAX;
    let mut step_hi = 0usize;
    for (ep, lo, hi) in epoch_step_ranges {
        if *ep >= start_ep && *ep <= end_ep {
            step_lo = step_lo.min(*lo);
            step_hi = step_hi.max(*hi);
        }
    }
    if step_lo > step_hi {
        return None;
    }
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for e in events {
        if let EventKind::Divergence { d_raw, step, .. } = &e.kind
            && *step >= step_lo
            && *step <= step_hi
            && *d_raw > 1e-12
        {
            xs.push(*step as f64);
            ys.push(d_raw.ln());
        }
    }
    let n = xs.len();
    if n < 5 {
        return None;
    }
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for k in 0..n {
        let dx = xs[k] - mx;
        let dy = ys[k] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let r2 = if syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        1.0
    };
    Some((n, step_lo, step_hi, slope, r2))
}

/// Threshold for the current convergence guard simulator. Matches the
/// production default in `flodl::distributed::ddp_run::ConvergenceGuard`.
/// Used in the post-hoc guard-comparison analysis.
const GUARD_CURRENT_D_THRESHOLD: f64 = 0.01;
/// MSF-style guard: λ_ema must exceed this for `GUARD_MSF_CONSECUTIVE`
/// consecutive epochs to fire. λ_ema > 0 means transversal deviation is
/// growing cycle-over-cycle (R5 innovation interpretation).
const GUARD_MSF_LAMBDA_THRESHOLD: f64 = 1.0e-3;
/// Number of consecutive epochs above `GUARD_MSF_LAMBDA_THRESHOLD` before
/// the MSF guard fires.
const GUARD_MSF_CONSECUTIVE: usize = 3;

fn simulate_guard_comparison(epochs: &[MsfEpoch]) -> MsfGuardComparison {
    // Current guard: 3 consecutive D_max rises, last > threshold.
    let mut current_fires: Vec<usize> = Vec::new();
    if epochs.len() >= 3 {
        for i in 2..epochs.len() {
            let d_now = epochs[i].d_max;
            let d_prev = epochs[i - 1].d_max;
            let d_prev2 = epochs[i - 2].d_max;
            if d_now > d_prev && d_prev > d_prev2 && d_now > GUARD_CURRENT_D_THRESHOLD {
                current_fires.push(epochs[i].epoch);
            }
        }
    }
    // MSF guard: λ_ema_at_epoch_end sustained above threshold.
    let mut msf_fires: Vec<usize> = Vec::new();
    let mut streak = 0usize;
    for me in epochs {
        let above = me
            .lambda_ema_at_epoch_end
            .map(|v| v > GUARD_MSF_LAMBDA_THRESHOLD)
            .unwrap_or(false);
        if above {
            streak += 1;
            if streak >= GUARD_MSF_CONSECUTIVE {
                msf_fires.push(me.epoch);
                streak = 0;
            }
        } else {
            streak = 0;
        }
    }
    let cur_set: std::collections::HashSet<usize> = current_fires.iter().copied().collect();
    let msf_set: std::collections::HashSet<usize> = msf_fires.iter().copied().collect();
    let mut both: Vec<usize> = cur_set.intersection(&msf_set).copied().collect();
    let mut current_only: Vec<usize> = cur_set.difference(&msf_set).copied().collect();
    let mut msf_only: Vec<usize> = msf_set.difference(&cur_set).copied().collect();
    both.sort_unstable();
    current_only.sort_unstable();
    msf_only.sort_unstable();
    MsfGuardComparison {
        current_fires,
        msf_fires,
        both,
        current_only,
        msf_only,
    }
}

/// Heuristic threshold for phase-transition candidate detection.
///
/// Marks an epoch as a candidate when its `lambda_min` is more negative than
/// this AND the per-epoch end-D collapses by at least a factor of 3 vs the
/// previous epoch's end-D. Tuned against the 200-epoch ResNet-20 sweep where
/// LR-drop epochs (100, 150) show `lambda_min` around -2e-2 to -5e-2.
const PHASE_LAMBDA_THRESHOLD: f64 = -1.0e-2;
/// Minimum collapse ratio (`d_end / prev_d_end < 1/3`) to flag as a candidate.
const PHASE_D_COLLAPSE_RATIO: f64 = 1.0 / 3.0;

fn build_msf_analysis(events: &[Event]) -> MsfAnalysis {
    let mut div_event_count = 0usize;
    let mut epochs: Vec<MsfEpoch> = Vec::new();
    // Per-rank tracking: walk div events in step order.
    // (rank index -> list of d at each event), step list per event.
    let mut per_rank_d: Vec<Vec<f64>> = Vec::new();
    let mut per_rank_step: Vec<Vec<usize>> = Vec::new();
    let mut win_counts: Vec<usize> = Vec::new();
    // Map epoch index -> LR at end of that epoch (from EpochEnd events).
    let mut epoch_lr: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for e in events {
        if let EventKind::EpochEnd { epoch, lr, .. } = &e.kind
            && lr.is_finite()
        {
            epoch_lr.insert(*epoch, *lr);
        }
    }

    for e in events {
        match &e.kind {
            EventKind::Divergence { deltas, step, .. } => {
                div_event_count += 1;
                // Initialize per-rank vectors once we see world_size.
                if per_rank_d.is_empty() && !deltas.is_empty() {
                    per_rank_d = vec![Vec::new(); deltas.len()];
                    per_rank_step = vec![Vec::new(); deltas.len()];
                    win_counts = vec![0; deltas.len()];
                }
                if deltas.len() == per_rank_d.len() {
                    for (r, d) in deltas.iter().enumerate() {
                        per_rank_d[r].push(*d);
                        per_rank_step[r].push(*step);
                    }
                    // Win = rank with max d this event.
                    if let Some((max_r, _)) = deltas
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    {
                        win_counts[max_r] += 1;
                    }
                }
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
                epochs.push(MsfEpoch {
                    epoch: *epoch,
                    sync_count: *sync_count,
                    d_min: *d_min,
                    d_max: *d_max,
                    d_mean: *d_mean,
                    d_at_epoch_end: *d_at_epoch_end,
                    k_at_epoch_end: *k_at_epoch_end,
                    lambda_min: *lambda_min,
                    lambda_max: *lambda_max,
                    lambda_mean: *lambda_mean,
                    lambda_ema_at_epoch_end: *lambda_ema_at_epoch_end,
                    lr: epoch_lr.get(epoch).copied(),
                });
            }
            _ => {}
        }
    }

    // Per-rank summary stats + per-rank lambda from consecutive event ratios.
    let world_size = per_rank_d.len();
    let total_wins: usize = win_counts.iter().sum();
    let mut per_rank: Vec<MsfPerRank> = Vec::with_capacity(world_size);
    for r in 0..world_size {
        let ds = &per_rank_d[r];
        let steps = &per_rank_step[r];
        let n = ds.len();
        if n == 0 {
            continue;
        }
        let d_mean = ds.iter().sum::<f64>() / n as f64;
        let d_sd = (ds.iter().map(|x| (x - d_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let d_min = ds.iter().copied().fold(f64::INFINITY, f64::min);
        let d_max = ds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let win_pct = if total_wins > 0 {
            win_counts[r] as f64 / total_wins as f64 * 100.0
        } else {
            0.0
        };
        // Per-rank lambda from consecutive d ratios.
        let mut lambdas: Vec<f64> = Vec::with_capacity(n);
        for i in 1..n {
            if ds[i - 1] > 1e-8 && ds[i] > 1e-8 {
                let k_diff = steps[i].saturating_sub(steps[i - 1]).max(1);
                lambdas.push((ds[i] / ds[i - 1]).ln() / k_diff as f64);
            }
        }
        let (lambda_mean, lambda_sd) = if lambdas.is_empty() {
            (0.0, 0.0)
        } else {
            let m = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            let s = (lambdas.iter().map(|x| (x - m).powi(2)).sum::<f64>() / lambdas.len() as f64)
                .sqrt();
            (m, s)
        };
        per_rank.push(MsfPerRank {
            rank: r,
            n,
            d_mean,
            d_sd,
            d_min,
            d_max,
            win_pct,
            lambda_mean,
            lambda_sd,
        });
    }

    // Pairwise Pearson correlation of per-rank D trajectories.
    let mut rank_correlations: Vec<((usize, usize), f64)> = Vec::new();
    for i in 0..world_size {
        for j in (i + 1)..world_size {
            let xs = &per_rank_d[i];
            let ys = &per_rank_d[j];
            let n = xs.len().min(ys.len());
            if n < 2 {
                continue;
            }
            let mx = xs.iter().take(n).sum::<f64>() / n as f64;
            let my = ys.iter().take(n).sum::<f64>() / n as f64;
            let mut sxx = 0.0;
            let mut syy = 0.0;
            let mut sxy = 0.0;
            for k in 0..n {
                let dx = xs[k] - mx;
                let dy = ys[k] - my;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
            if sxx > 0.0 && syy > 0.0 {
                rank_correlations.push(((i, j), sxy / (sxx * syy).sqrt()));
            }
        }
    }

    let mut phase_candidates: Vec<MsfPhaseCandidate> = Vec::new();
    for i in 1..epochs.len() {
        let curr = &epochs[i];
        let prev = &epochs[i - 1];
        let Some(lmin) = curr.lambda_min else { continue };
        if lmin >= PHASE_LAMBDA_THRESHOLD {
            continue;
        }
        if prev.d_at_epoch_end <= 0.0 {
            continue;
        }
        let ratio = curr.d_at_epoch_end / prev.d_at_epoch_end;
        if ratio < PHASE_D_COLLAPSE_RATIO {
            phase_candidates.push(MsfPhaseCandidate {
                epoch: curr.epoch,
                lambda_min: lmin,
                d_end: curr.d_at_epoch_end,
                d_ratio: ratio,
            });
        }
    }

    // Per-epoch step ranges: walk div events in chronological order, assign
    // each to the first containing-epoch by div_epoch event timestamps.
    // (epoch -> (min step, max step))
    let mut epoch_step_min: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut epoch_step_max: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut div_with_t: Vec<(u64, &Event)> = Vec::new();
    let mut epoch_end_t: Vec<(u64, usize)> = Vec::new();
    for ev in events {
        match &ev.kind {
            EventKind::Divergence { .. } => div_with_t.push((ev.t, ev)),
            EventKind::DivergenceEpoch { epoch, .. } => epoch_end_t.push((ev.t, *epoch)),
            _ => {}
        }
    }
    div_with_t.sort_by_key(|x| x.0);
    epoch_end_t.sort_by_key(|x| x.0);
    let mut ep_idx = 0usize;
    for (t, ev) in &div_with_t {
        while ep_idx < epoch_end_t.len() && epoch_end_t[ep_idx].0 < *t {
            ep_idx += 1;
        }
        let cur_epoch = if ep_idx < epoch_end_t.len() {
            epoch_end_t[ep_idx].1
        } else {
            continue;
        };
        if let EventKind::Divergence { step, .. } = &ev.kind {
            epoch_step_min
                .entry(cur_epoch)
                .and_modify(|s| *s = (*s).min(*step))
                .or_insert(*step);
            epoch_step_max
                .entry(cur_epoch)
                .and_modify(|s| *s = (*s).max(*step))
                .or_insert(*step);
        }
    }
    let mut epoch_step_ranges: Vec<(usize, usize, usize)> = epoch_step_min
        .iter()
        .filter_map(|(ep, lo)| epoch_step_max.get(ep).map(|hi| (*ep, *lo, *hi)))
        .collect();
    epoch_step_ranges.sort_by_key(|x| x.0);

    // Detect LR windows from MsfEpoch.lr transitions.
    let lr_window_fits: Vec<MsfLrWindowFit> = if epochs.iter().any(|e| e.lr.is_some()) {
        let mut windows: Vec<(f64, usize, usize)> = Vec::new();
        let mut cur: Option<(f64, usize, usize)> = None;
        for me in &epochs {
            if let Some(lr) = me.lr {
                match cur {
                    None => cur = Some((lr, me.epoch, me.epoch)),
                    Some((cur_lr, start, _)) => {
                        let frac = if cur_lr.abs() > 1e-12 {
                            (lr - cur_lr).abs() / cur_lr.abs()
                        } else {
                            f64::INFINITY
                        };
                        if frac > LR_WINDOW_CHANGE_FRAC {
                            windows.push((cur_lr, start, me.epoch.saturating_sub(1)));
                            cur = Some((lr, me.epoch, me.epoch));
                        } else {
                            cur = Some((cur_lr, start, me.epoch));
                        }
                    }
                }
            }
        }
        if let Some((lr, start, end)) = cur {
            windows.push((lr, start, end));
        }
        windows
            .into_iter()
            .filter_map(|(lr, start, end)| {
                fit_lr_window(events, &epoch_step_ranges, start, end).map(
                    |(n, s_lo, s_hi, slope, r2)| MsfLrWindowFit {
                        lr,
                        epoch_start: start,
                        epoch_end: end,
                        n_events: n,
                        step_min: s_lo,
                        step_max: s_hi,
                        slope_per_step: slope,
                        r2,
                    },
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    // Longitudinal meta-velocity: walk div events in chronological order,
    // compute |Δ post_norm| / post_norm_prev. Only available when post_norm
    // is logged (cpu modes always; nccl modes after post_norm wiring).
    let mut velocities: Vec<f64> = Vec::new();
    let mut post_norms: Vec<f64> = Vec::new();
    let mut prev_pn: Option<f64> = None;
    for (_, ev) in &div_with_t {
        if let EventKind::Divergence { post_norm, .. } = &ev.kind
            && let Some(pn) = post_norm
            && pn.is_finite()
            && *pn > 0.0
        {
            post_norms.push(*pn);
            if let Some(prev) = prev_pn
                && prev > 0.0
            {
                velocities.push((pn - prev).abs() / prev);
            }
            prev_pn = Some(*pn);
        } else {
            // Lost a sample (no post_norm) — break the velocity chain so
            // we don't compare across non-contiguous events.
            prev_pn = None;
        }
    }
    let longitudinal = if post_norms.is_empty() {
        None
    } else {
        let pn_min = post_norms.iter().copied().fold(f64::INFINITY, f64::min);
        let pn_max = post_norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pn_mean = post_norms.iter().sum::<f64>() / post_norms.len() as f64;
        let (v_min, v_max, v_mean, v_sd) = if velocities.is_empty() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let mn = velocities.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = velocities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = velocities.iter().sum::<f64>() / velocities.len() as f64;
            let var = velocities.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / velocities.len() as f64;
            (mn, mx, mean, var.sqrt())
        };
        Some(MsfLongitudinal {
            n: velocities.len(),
            post_norm_min: pn_min,
            post_norm_max: pn_max,
            post_norm_mean: pn_mean,
            velocity_min: v_min,
            velocity_max: v_max,
            velocity_mean: v_mean,
            velocity_sd: v_sd,
        })
    };

    let guard_comparison = simulate_guard_comparison(&epochs);

    MsfAnalysis {
        div_event_count,
        epochs,
        phase_candidates,
        per_rank,
        rank_correlations,
        lr_window_fits,
        longitudinal,
        guard_comparison,
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
// Training log parser
// ---------------------------------------------------------------------------

/// Parsed training log data.
pub struct TrainingLog {
    pub epochs: Vec<LogEpoch>,
    /// Standalone `final eval=X.XXXX` line (modes that eval once after training).
    pub final_eval: Option<f64>,
    /// Total wall time from `# total:` footer (ms).
    pub total_ms: Option<f64>,
    /// Training-only wall time from `# train_only:` summary (ms). Set by
    /// `run_baseline_solo` so the report can compare DDP wall time against
    /// solo's training-only time, excluding the per-epoch eval cost solo
    /// pays but DDP only pays once at the end.
    pub train_only_ms: Option<f64>,
    /// GPU header lines (e.g. `gpu0: NVIDIA GeForce RTX 5060 Ti (15GB, sm_120)`).
    pub gpu_info: Vec<String>,
}

/// One epoch line from the training log.
pub struct LogEpoch {
    pub epoch: usize,
    pub loss: f64,
    pub eval: Option<f64>,
    /// Training-only wall time for the epoch (ms). Parsed from `train=Xs`
    /// (new format) or `time=Xs` (legacy, where the value was already
    /// training-only).
    pub time_ms: f64,
    /// Per-rank breakdown from the `per-rank:` line that follows multi-rank
    /// epoch lines. Empty for solo and single-rank runs.
    pub per_rank: Vec<RankSnapshot>,
}

/// Per-rank stats for one epoch.
#[derive(Debug, Clone)]
pub struct RankSnapshot {
    pub rank: usize,
    pub device: u8,
    /// Fraction of batches this rank consumed (0..1, sums to ~1 across ranks).
    pub batch_share: f64,
    /// Throughput in samples/ms.
    pub throughput: f64,
}

/// Parse a `training.log` file.
///
/// Format:
/// ```text
/// # gpu0: ...
/// epoch 0: loss=0.311125, eval=0.9732, time=2.2s
/// epoch 1: loss=0.131376, time=2.3s
/// final eval=0.9732
/// # total: 12.7s (0m 13s)
/// ```
pub fn parse_training_log(path: &Path) -> Result<TrainingLog, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

    let mut epochs = Vec::new();
    let mut final_eval = None;
    let mut total_ms = None;
    let mut train_only_ms = None;
    let mut gpu_info = Vec::new();

    for line in data.lines() {
        let line = line.trim();

        // # gpu0: NVIDIA GeForce RTX 5060 Ti (15GB, sm_120)
        if let Some(rest) = line.strip_prefix("# gpu") {
            if rest.contains(':') {
                gpu_info.push(format!("gpu{rest}"));
            }
            continue;
        }

        // epoch N: loss=X.XXXXXX[, eval=X.XXXX][, train_acc=X.XXXX][, train=X.Xs|time=X.Xs][, eval_time=X.Xs]
        if let Some(rest) = line.strip_prefix("epoch ") {
            if let Some((epoch_str, kv_part)) = rest.split_once(": ") {
                let epoch: usize = epoch_str.parse().unwrap_or(0);
                let mut loss = 0.0;
                let mut eval = None;
                let mut time_ms = 0.0;

                for kv in kv_part.split(", ") {
                    if let Some(v) = kv.strip_prefix("loss=") {
                        loss = v.parse().unwrap_or(0.0);
                    } else if let Some(v) = kv.strip_prefix("eval=")
                        .or_else(|| kv.strip_prefix("metric="))
                    {
                        eval = Some(v.parse().unwrap_or(0.0));
                    } else if let Some(v) = kv.strip_prefix("train=")
                        .or_else(|| kv.strip_prefix("time="))
                    {
                        // `train=` is the new explicit form (training-only);
                        // `time=` is the legacy column (also training-only,
                        // since epoch_ms was always measured before eval).
                        if let Some(ms) = v.strip_suffix("ms") {
                            time_ms = ms.parse::<f64>().unwrap_or(0.0);
                        } else if let Some(secs) = v.strip_suffix('s') {
                            time_ms = secs.parse::<f64>().unwrap_or(0.0) * 1000.0;
                        }
                    }
                    // eval_time is parsed but not stored on LogEpoch yet; the
                    // total is in the `# train_only: Xs (eval: Ys)` summary.
                }

                epochs.push(LogEpoch { epoch, loss, eval, time_ms, per_rank: Vec::new() });
            }
        }
        // per-rank: rank0[cuda0,share=0.3447,tput=56.88] rank1[cuda1,share=...]
        else if let Some(rest) = line.strip_prefix("per-rank:") {
            let snapshots = parse_per_rank_line(rest);
            if let Some(last) = epochs.last_mut() {
                last.per_rank = snapshots;
            }
        }
        // final eval=X.XXXX
        else if let Some(v) = line.strip_prefix("final eval=") {
            final_eval = Some(v.parse().unwrap_or(0.0));
        }
        // # train_only: 13.6s (eval: 0.6s)
        else if let Some(rest) = line.strip_prefix("# train_only: ")
            && let Some(secs_str) = rest.split('s').next()
        {
            train_only_ms = secs_str.trim().parse::<f64>().ok().map(|s| s * 1000.0);
        }
        // # total: 12.7s (0m 13s)
        else if let Some(rest) = line.strip_prefix("# total: ")
            && let Some(secs_str) = rest.split('s').next()
        {
            total_ms = secs_str.trim().parse::<f64>().ok().map(|s| s * 1000.0);
        }
    }

    Ok(TrainingLog { epochs, final_eval, total_ms, train_only_ms, gpu_info })
}

/// Parse the body of a `per-rank:` line into [`RankSnapshot`]s.
///
/// Format (from `harness::run_unified`):
/// ` rank0[cuda0,share=0.3447,tput=56.88] rank1[cuda1,share=0.3533,tput=57.35]`
fn parse_per_rank_line(rest: &str) -> Vec<RankSnapshot> {
    let mut out = Vec::new();
    for token in rest.split_whitespace() {
        // token: rankN[cudaD,share=X,tput=Y]
        let Some(open) = token.find('[') else { continue };
        let Some(close_idx) = token.find(']') else { continue };
        let rank_str = &token[..open];
        let body = &token[open + 1..close_idx];
        let Some(rank_num) = rank_str.strip_prefix("rank") else { continue };
        let Ok(rank) = rank_num.parse::<usize>() else { continue };

        let mut device: u8 = 0;
        let mut batch_share = 0.0;
        let mut throughput = 0.0;
        for kv in body.split(',') {
            if let Some(v) = kv.strip_prefix("cuda") {
                device = v.parse().unwrap_or(0);
            } else if let Some(v) = kv.strip_prefix("share=") {
                batch_share = v.parse().unwrap_or(0.0);
            } else if let Some(v) = kv.strip_prefix("tput=") {
                throughput = v.parse().unwrap_or(0.0);
            }
        }
        out.push(RankSnapshot { rank, device, batch_share, throughput });
    }
    out
}

/// Apply training log data to a RunAnalysis, overriding timeline-derived
/// loss/eval/epoch data with the authoritative log values.
pub fn apply_training_log(analysis: &mut RunAnalysis, log: &TrainingLog) {
    if log.epochs.is_empty() {
        return;
    }

    // Override epoch data
    analysis.epoch_data = log.epochs.iter().map(|e| EpochData {
        epoch: e.epoch,
        loss: e.loss,
        eval: e.eval,
        wall_ms: e.time_ms,
    }).collect();
    analysis.n_epochs = analysis.epoch_data.len();

    // Final loss from last epoch
    analysis.final_loss = log.epochs.last().map(|e| e.loss).unwrap_or(0.0);

    // Final eval: standalone line wins, otherwise last per-epoch eval
    analysis.final_eval = log.final_eval.or_else(|| {
        log.epochs.iter().rev().find_map(|e| e.eval)
    });

    // Total time from log footer (if timeline had no samples)
    if analysis.total_ms == 0
        && let Some(ms) = log.total_ms
    {
        analysis.total_ms = ms as u64;
    }

    // Training-only time (set by run_baseline_solo via `# train_only:` line).
    analysis.train_only_ms = log.train_only_ms.map(|ms| ms as u64);

    // Per-rank averages: collect snapshots across epochs, average per rank.
    if log.epochs.iter().any(|e| !e.per_rank.is_empty()) {
        use std::collections::BTreeMap;
        let mut acc: BTreeMap<usize, (u8, f64, f64, usize)> = BTreeMap::new();
        for ep in &log.epochs {
            for snap in &ep.per_rank {
                let entry = acc.entry(snap.rank).or_insert((snap.device, 0.0, 0.0, 0));
                entry.0 = snap.device;
                entry.1 += snap.batch_share;
                entry.2 += snap.throughput;
                entry.3 += 1;
            }
        }
        analysis.per_rank_avg = acc.into_iter().map(|(rank, (device, share_sum, tput_sum, n))| {
            let n_f = n as f64;
            PerRankAvg {
                rank,
                device,
                batch_share: if n > 0 { share_sum / n_f } else { 0.0 },
                throughput: if n > 0 { tput_sum / n_f } else { 0.0 },
            }
        }).collect();
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
