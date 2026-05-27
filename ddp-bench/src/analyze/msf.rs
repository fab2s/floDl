//! MSF (Mean Squared Field) divergence analysis — types + per-event recomputation pipeline.

use super::fit::{fit_lr_window, LR_WINDOW_CHANGE_FRAC};
use super::{EpochData, Event, EventKind};

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
/// `λ_ema`.
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

/// log(D) vs cumulative step linear fit within a single LR window
/// (auto-detected from EpochEnd LR transitions).
///
/// Slope is in units of ln(D)/step. R² is the coefficient of determination.
/// The marginal-stability prediction is that log(D_t) is approximately
/// linear in step within stable phases; high R² supports the prediction,
/// low R² either falsifies it or indicates a noise-dominated equilibrium
/// where slope ≈ 0 holds but the variance can't be fit against.
///
/// Two bases are reported: `D_max` (per-event maximum across ranks; the legacy
/// metric — sensitive to per-rank step asymmetry, especially under NCCL) and
/// `D_mean` (per-event mean across ranks; the meta-oscillator amplitude —
/// averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so
/// the mean and max trace the same underlying process up to scale, but the
/// mean is less noisy). `D_mean` fields are `None` for runs predating per-rank
/// `deltas` capture.
///
/// `transient_skipped` is the number of leading epochs trimmed from the
/// window before fitting (0 = full window). A non-zero value indicates a
/// post-transient sub-window: typically emitted alongside the full-window
/// fit for the first LR window to separate the initialization transient
/// from the stable-LR steady state.
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
    pub slope_per_step_dmean: Option<f64>,
    pub r2_dmean: Option<f64>,
    /// Per-epoch aggregation of `ln(D_mean)` (intra-epoch log-mean) vs the
    /// epoch's mean cumulative step. Denoises intra-epoch SGD variance — the
    /// dominant remaining noise source after the cross-rank D_mean swap.
    pub slope_per_step_epoch_dmean: Option<f64>,
    pub r2_epoch_dmean: Option<f64>,
    pub n_epoch_points: Option<usize>,
    /// OLS of `ln(D_mean)` vs `k_used` (per-event cycle length: steps since
    /// last sync). Tests the alternative R1 framing where the relevant
    /// "drift clock" restarts at each AllReduce — sync is a reset, so within
    /// a fixed-LR window the natural axis is steps-since-sync, not
    /// cumulative training step. `None` when no events have non-empty
    /// `deltas`, or when k_used has zero variance across the window.
    pub slope_by_k_used_dmean: Option<f64>,
    pub r2_by_k_used_dmean: Option<f64>,
    /// Range of k_used observed in the window (controller-determined; varies
    /// per event when ElChe's convergence guard truncates cycles early).
    pub k_used_min: Option<usize>,
    pub k_used_max: Option<usize>,
    /// Per-rank by-k OLS: bottom-scale consistency check on the meta-
    /// oscillator framing. `slope_by_k_per_rank[i]` is the within-cycle
    /// Lyapunov estimate computed from rank i's `D_i` trajectory alone.
    /// Under the meta-oscillator framing (cross-rank Pearson r > 0.99),
    /// per-rank slopes should match the meta-D_mean slope within seed-to-
    /// seed sd; divergence between per-rank and meta slopes indicates the
    /// framing is breaking down and per-rank treatment is required.
    /// Empty when `deltas` are unavailable or rank count varies across
    /// events.
    pub slope_by_k_per_rank: Vec<f64>,
    pub r2_by_k_per_rank: Vec<f64>,
    pub transient_skipped: usize,
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
    /// Per-LR-window linear fits of log(D) vs cumulative step. Empty when
    /// no EpochEnd events carry LR (runs predating per-event LR logging).
    pub lr_window_fits: Vec<MsfLrWindowFit>,
    /// Longitudinal meta-velocity (consensus magnitude motion). `None` when
    /// no `Divergence` event carries `post_norm` (runs predating post_norm
    /// wiring, or backends that don't compute it).
    pub longitudinal: Option<MsfLongitudinal>,
    /// Guard simulator comparison: current guard fires vs MSF-style guard
    /// fires on per-event recomputed lambda. Both sides operate at the
    /// per-event temporal grain matching production.
    pub guard_comparison: MsfGuardComparison,
    /// Threshold sweep for the MSF guard. Each row: `(threshold, sustain,
    /// fires, epochs_covered)`. Lets the reader judge sensitivity without
    /// committing to a single magic-number threshold.
    pub msf_threshold_sweep: Vec<MsfThresholdSweepRow>,
    /// Predictive-value stats for the MSF kill criterion (does `λ̂` carry
    /// forward-looking signal on divergence and eval). `None` when too few
    /// events for stable Pearson estimates.
    pub predictive: Option<MsfPredictive>,
    /// Per-event recomputed λ̂ trajectory. Populated by `build_msf_analysis`;
    /// consumed by `apply_training_log` to fill `predictive` once per-epoch
    /// eval is available. Kept on the struct so we don't re-walk events.
    pub recomputed: Vec<RecomputedLambda>,
    /// Stratified predictive: Pearson(λ_raw_t, ln(D_{t+1})) restricted to
    /// events inside each LR window. Steady-state events dilute the
    /// run-global correlation; per-window numbers surface where the
    /// signal actually lives (warmup, post-LR-drop transient).
    pub predictive_by_lr_window: Vec<MsfPredictiveByLrWindow>,
}

/// Per-LR-window predictive correlation row.
#[derive(Debug, Clone)]
pub struct MsfPredictiveByLrWindow {
    pub lr: f64,
    pub epoch_start: usize,
    pub epoch_end: usize,
    pub n_pairs: usize,
    pub r: Option<f64>,
}

/// One row of the MSF-guard threshold sweep.
#[derive(Debug, Clone)]
pub struct MsfThresholdSweepRow {
    pub threshold: f64,
    pub sustain: usize,
    pub fires: usize,
    /// Distinct epochs touched by at least one fire.
    pub epochs_covered: usize,
}

/// MSF kill-criterion predictive correlations.
///
/// All correlations are Pearson and scale-invariant under the `k_used` ↔
/// `k_max` rescale, so values reported here transfer cleanly between the
/// pre-correction and post-correction λ̂ formulae. The point of these
/// numbers is not to validate the *magnitude* of λ̂ but to validate that
/// it carries forward-looking signal.
#[derive(Debug, Clone)]
pub struct MsfPredictive {
    /// Number of (λ̂_t, log D_{t+1}) pairs with both finite.
    pub n_lambda_to_next_logd: usize,
    /// Pearson(λ_raw_t, ln(D_{t+1})). Negative correlation is the design
    /// doc's expected sign: high λ̂ (transversal stretching) → larger
    /// next-event divergence... wait — the *correct* sign depends on
    /// whether D collapses or stretches over the window the rate covers.
    /// Reporting raw correlation; interpretation goes in the report prose.
    pub lambda_to_next_logd_r: Option<f64>,
    /// Number of `(λ_mean_per_epoch, eval[ep])` pairs with both finite.
    pub n_lambda_to_eval: usize,
    /// Pearson(epoch-mean λ̂, eval at end of same epoch).
    pub lambda_mean_to_eval_r: Option<f64>,
    /// Pearson(λ_ema_at_epoch_end, eval at end of same epoch).
    pub lambda_ema_to_eval_r: Option<f64>,
}

impl MsfAnalysis {
    /// Whether any MSF data was captured for this run.
    pub fn has_data(&self) -> bool {
        !self.epochs.is_empty()
    }
}

/// Threshold for the current convergence guard simulator. Matches the
/// production default in `flodl::distributed::ddp_run::ConvergenceGuard`.
const GUARD_CURRENT_D_THRESHOLD: f64 = 0.01;
/// Default centre threshold used in the headline guard-comparison row;
/// the threshold sweep table covers a grid around this point.
const GUARD_MSF_LAMBDA_THRESHOLD: f64 = 1.0e-3;
/// Default sustain length for the headline MSF guard row.
const GUARD_MSF_CONSECUTIVE: usize = 3;
/// MSF-guard sweep grids.
const MSF_THRESHOLD_GRID: &[f64] = &[0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1];
const MSF_SUSTAIN_GRID: &[usize] = &[3, 5, 10];

/// Per-event recomputed lambda sample. Cached on `MsfAnalysis` so the
/// predictive-value step (which runs after `apply_training_log` joins
/// per-epoch eval) can reuse the canonical pipeline without re-walking
/// the event list.
#[derive(Debug, Clone, Copy)]
pub struct RecomputedLambda {
    pub epoch: usize,
    /// Cumulative training step at this event. Kept for downstream
    /// scatter/diagnostic plots; not currently consumed by the report
    /// tables, which aggregate by epoch.
    #[allow(dead_code)]
    pub step: usize,
    pub d_raw: f64,
    pub lambda_raw: Option<f64>,
    pub lambda_ema: Option<f64>,
}

/// In-analyzer port of `flodl::distributed::ddp_run::convergence::LambdaEstimator`.
///
/// Mirrors the post-corrections semantics: rate denominator `k_max`,
/// full-reset on noise-floor `d_raw`, Adam-style bias-corrected EMA from
/// `ema_raw=0.0`. We rebuild λ̂ here so analyze.rs is a single canonical
/// pipeline — old timelines (which emit `lambda_raw = log(D/prev) / k_used`)
/// produce the same downstream tables as new ones (`/ k_max`).
struct LambdaEstimator {
    prev_d: Option<f64>,
    ema_raw: f64,
    ema_t: u32,
    alpha: f64,
    noise_floor: f64,
}

impl LambdaEstimator {
    fn new() -> Self {
        Self { prev_d: None, ema_raw: 0.0, ema_t: 0, alpha: 0.9, noise_floor: 1e-8 }
    }

    fn observe(&mut self, d_raw: f64, k_max: usize) -> (Option<f64>, Option<f64>) {
        let lambda_raw = match self.prev_d {
            Some(prev) if prev > self.noise_floor && d_raw > self.noise_floor && k_max > 0 => {
                Some((d_raw / prev).ln() / k_max as f64)
            }
            _ => None,
        };
        if let Some(l) = lambda_raw {
            self.ema_raw = self.alpha * self.ema_raw + (1.0 - self.alpha) * l;
            self.ema_t = self.ema_t.saturating_add(1);
        }
        self.prev_d = if d_raw > self.noise_floor { Some(d_raw) } else { None };
        let lambda_ema = if self.ema_t == 0 {
            None
        } else {
            let denom = 1.0 - self.alpha.powi(self.ema_t as i32);
            Some(if denom > 0.0 { self.ema_raw / denom } else { self.ema_raw })
        };
        (lambda_raw, lambda_ema)
    }
}

/// Per-event recomputed-λ̂ trajectory in t-order.
fn recompute_lambdas(events: &[Event]) -> Vec<RecomputedLambda> {
    // Walk div events in event-list order (already t-sorted at load).
    let mut out: Vec<RecomputedLambda> = Vec::new();
    let mut est = LambdaEstimator::new();
    // For old timelines without an `epoch` field, we t-lookup via the
    // sorted EpochEnd timestamps: an event at time t belongs to the
    // first epoch whose EpochEnd has t' >= t.
    let mut epoch_ends: Vec<(u64, usize)> = events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::EpochEnd { epoch, .. } => Some((e.t, *epoch)),
            _ => None,
        })
        .collect();
    epoch_ends.sort_by_key(|x| x.0);
    let mut ep_idx = 0usize;
    for ev in events {
        let EventKind::Divergence { d_raw, k_max, step, epoch, .. } = &ev.kind else { continue };
        let (lambda_raw, lambda_ema) = est.observe(*d_raw, *k_max);
        let cur_epoch = if let Some(e) = epoch {
            *e
        } else {
            while ep_idx < epoch_ends.len() && epoch_ends[ep_idx].0 < ev.t {
                ep_idx += 1;
            }
            if ep_idx < epoch_ends.len() { epoch_ends[ep_idx].1 } else { continue }
        };
        out.push(RecomputedLambda {
            epoch: cur_epoch,
            step: *step,
            d_raw: *d_raw,
            lambda_raw,
            lambda_ema,
        });
    }
    out
}

/// Per-event simulator: ConvergenceGuard::check_trend over a 5-element
/// ring of `d_raw`. Fires whenever the last 3 samples are strictly rising
/// AND the latest exceeds `threshold`. Returns the epoch numbers of the
/// firing events (de-duplicated, sorted).
fn simulate_current_guard_per_event(rec: &[RecomputedLambda], threshold: f64) -> Vec<usize> {
    let mut fires: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    if rec.len() < 3 {
        return Vec::new();
    }
    for i in 2..rec.len() {
        let a = rec[i - 2].d_raw;
        let b = rec[i - 1].d_raw;
        let c = rec[i].d_raw;
        if c > b && b > a && c > threshold {
            fires.insert(rec[i].epoch);
        }
    }
    fires.into_iter().collect()
}

/// Per-event simulator: MSF-style. Fires whenever `lambda_ema` has been
/// strictly above `threshold` for `sustain` consecutive events. Streak
/// resets after each fire. Returns firing epoch numbers (de-duplicated,
/// sorted).
fn simulate_msf_guard_per_event(
    rec: &[RecomputedLambda],
    threshold: f64,
    sustain: usize,
) -> Vec<usize> {
    let mut fires: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut streak = 0usize;
    for r in rec {
        let above = r.lambda_ema.map(|v| v > threshold).unwrap_or(false);
        if above {
            streak += 1;
            if streak >= sustain {
                fires.insert(r.epoch);
                streak = 0;
            }
        } else {
            streak = 0;
        }
    }
    fires.into_iter().collect()
}

fn simulate_guard_comparison(rec: &[RecomputedLambda]) -> MsfGuardComparison {
    let current_fires =
        simulate_current_guard_per_event(rec, GUARD_CURRENT_D_THRESHOLD);
    let msf_fires =
        simulate_msf_guard_per_event(rec, GUARD_MSF_LAMBDA_THRESHOLD, GUARD_MSF_CONSECUTIVE);
    let cur_set: std::collections::HashSet<usize> = current_fires.iter().copied().collect();
    let msf_set: std::collections::HashSet<usize> = msf_fires.iter().copied().collect();
    let mut both: Vec<usize> = cur_set.intersection(&msf_set).copied().collect();
    let mut current_only: Vec<usize> = cur_set.difference(&msf_set).copied().collect();
    let mut msf_only: Vec<usize> = msf_set.difference(&cur_set).copied().collect();
    both.sort_unstable();
    current_only.sort_unstable();
    msf_only.sort_unstable();
    MsfGuardComparison { current_fires, msf_fires, both, current_only, msf_only }
}

fn build_msf_threshold_sweep(rec: &[RecomputedLambda]) -> Vec<MsfThresholdSweepRow> {
    let mut rows = Vec::with_capacity(MSF_THRESHOLD_GRID.len() * MSF_SUSTAIN_GRID.len());
    for &threshold in MSF_THRESHOLD_GRID {
        for &sustain in MSF_SUSTAIN_GRID {
            let fires = simulate_msf_guard_per_event(rec, threshold, sustain);
            let epochs_covered: std::collections::HashSet<usize> =
                fires.iter().copied().collect();
            rows.push(MsfThresholdSweepRow {
                threshold,
                sustain,
                fires: fires.len(),
                epochs_covered: epochs_covered.len(),
            });
        }
    }
    rows
}

fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return None;
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
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx * syy).sqrt())
}

/// Predictive-value correlations: λ̂_t → next D, λ̂ aggregates → eval.
pub(super) fn build_predictive(
    rec: &[RecomputedLambda],
    epochs: &[MsfEpoch],
    epoch_data: &[EpochData],
) -> Option<MsfPredictive> {
    if rec.len() < 4 {
        return None;
    }
    // Within-event: pair λ_raw[t] with ln(D[t+1]).
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..rec.len() - 1 {
        let Some(l) = rec[i].lambda_raw else { continue };
        let d_next = rec[i + 1].d_raw;
        if d_next <= 0.0 || !d_next.is_finite() || !l.is_finite() {
            continue;
        }
        xs.push(l);
        ys.push(d_next.ln());
    }
    let lambda_to_next_logd_r = pearson(&xs, &ys);
    let n_lambda_to_next_logd = xs.len();

    // Per-epoch: pair recomputed lambda aggregates with eval. Use eval
    // from `EpochData` (training log) joined by epoch index.
    let eval_by_epoch: std::collections::HashMap<usize, f64> = epoch_data
        .iter()
        .filter_map(|e| e.eval.map(|v| (e.epoch, v)))
        .collect();
    let mut x_mean = Vec::new();
    let mut y_mean = Vec::new();
    let mut x_ema = Vec::new();
    let mut y_ema = Vec::new();
    for me in epochs {
        let Some(eval) = eval_by_epoch.get(&me.epoch).copied() else { continue };
        if let Some(lm) = me.lambda_mean
            && lm.is_finite()
        {
            x_mean.push(lm);
            y_mean.push(eval);
        }
        if let Some(le) = me.lambda_ema_at_epoch_end
            && le.is_finite()
        {
            x_ema.push(le);
            y_ema.push(eval);
        }
    }
    let n_lambda_to_eval = x_mean.len();
    let lambda_mean_to_eval_r = pearson(&x_mean, &y_mean);
    let lambda_ema_to_eval_r = pearson(&x_ema, &y_ema);
    Some(MsfPredictive {
        n_lambda_to_next_logd,
        lambda_to_next_logd_r,
        n_lambda_to_eval,
        lambda_mean_to_eval_r,
        lambda_ema_to_eval_r,
    })
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

pub(super) fn build_msf_analysis(events: &[Event]) -> MsfAnalysis {
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
            .enumerate()
            .flat_map(|(idx, (lr, start, end))| {
                let is_first = idx == 0;
                fit_lr_window(events, &epoch_step_ranges, start, end, is_first)
                    .into_iter()
                    .map(move |mut f| {
                        f.lr = lr;
                        f
                    })
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

    // Canonical λ̂ pipeline: re-run estimation in t-order using the
    // post-corrections semantics (k_max denominator, full-reset on
    // noise-floor, bias-corrected EMA). Old timelines that emitted
    // lambda computed with k_used become equivalent to new ones once
    // analyze.rs takes over the math.
    let recomputed = recompute_lambdas(events);

    // Override per-epoch lambda aggregates with values derived from the
    // recomputed per-event series so MsfEpoch is self-consistent under
    // the new pipeline. Per-epoch d aggregates (d_min/d_max/d_mean/
    // d_at_epoch_end) are pure observations and don't need recompute.
    let mut by_epoch: std::collections::HashMap<usize, Vec<&RecomputedLambda>> =
        std::collections::HashMap::new();
    for r in &recomputed {
        by_epoch.entry(r.epoch).or_default().push(r);
    }
    for me in &mut epochs {
        let Some(rs) = by_epoch.get(&me.epoch) else { continue };
        let lambdas: Vec<f64> = rs.iter().filter_map(|r| r.lambda_raw).collect();
        if lambdas.is_empty() {
            me.lambda_min = None;
            me.lambda_max = None;
            me.lambda_mean = None;
        } else {
            let mn = lambdas.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = lambdas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            me.lambda_min = Some(mn);
            me.lambda_max = Some(mx);
            me.lambda_mean = Some(mean);
        }
        me.lambda_ema_at_epoch_end = rs.last().and_then(|r| r.lambda_ema);
    }

    let guard_comparison = simulate_guard_comparison(&recomputed);
    let msf_threshold_sweep = build_msf_threshold_sweep(&recomputed);

    // Stratified predictive: r(λ_raw_t, ln(D_{t+1})) per LR window. Only
    // pairs WHERE BOTH ENDPOINTS are in the same window are counted, so a
    // pair straddling a phase transition doesn't contribute (otherwise we
    // pick up the LR-drop collapse as artefactual signal).
    let predictive_by_lr_window: Vec<MsfPredictiveByLrWindow> = lr_window_fits
        .iter()
        .filter(|w| w.transient_skipped == 0)
        .map(|w| {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for i in 0..recomputed.len().saturating_sub(1) {
                let cur = &recomputed[i];
                let nxt = &recomputed[i + 1];
                if cur.epoch < w.epoch_start || cur.epoch > w.epoch_end {
                    continue;
                }
                if nxt.epoch < w.epoch_start || nxt.epoch > w.epoch_end {
                    continue;
                }
                let Some(l) = cur.lambda_raw else { continue };
                if !l.is_finite() || nxt.d_raw <= 0.0 || !nxt.d_raw.is_finite() {
                    continue;
                }
                xs.push(l);
                ys.push(nxt.d_raw.ln());
            }
            MsfPredictiveByLrWindow {
                lr: w.lr,
                epoch_start: w.epoch_start,
                epoch_end: w.epoch_end,
                n_pairs: xs.len(),
                r: pearson(&xs, &ys),
            }
        })
        .collect();

    MsfAnalysis {
        div_event_count,
        epochs,
        phase_candidates,
        per_rank,
        rank_correlations,
        lr_window_fits,
        longitudinal,
        guard_comparison,
        msf_threshold_sweep,
        predictive: None, // filled by apply_training_log once eval is joined
        recomputed,
        predictive_by_lr_window,
    }
}
