//! Unified convergence guard for DDP parameter averaging.
//!
//! Monitors weight-space divergence across replicas and recommends cadence
//! adjustments. Replaces the old dual system (loss-based for NCCL, norm-based
//! for CPU) with a single guard that both paths feed into.
//!
//! The primary signal is `||params_before_sync - params_after_sync|| / ||params_after_sync||`,
//! which measures each replica's L2 distance from the consensus (average).
//! Three norms (pre, post, diff) give a full decomposition into magnitude
//! shift and directional coherence for free.

use std::collections::VecDeque;

use super::ApplyPolicy;

// ---------------------------------------------------------------------------
// Divergence report
// ---------------------------------------------------------------------------

/// Divergence measurements from a single sync event.
///
/// The primary signal is `deltas` -- per-rank normalized L2 distance from the
/// replica's pre-sync params to the post-sync average. This captures both
/// magnitude drift and directional divergence.
///
/// When `pre_norms` and `post_norm` are available, the decomposition into
/// magnitude shift and cosine similarity is free via:
/// `cos = (||pre||^2 + ||post||^2 - (delta * ||post||)^2) / (2 * ||pre|| * ||post||)`
pub struct DivergenceReport {
    /// Per-rank normalized delta: `||pre - post|| / ||post||`.
    pub deltas: Vec<f64>,
    /// Per-rank pre-sync param L2 norm. `None` when not computed (NCCL v1).
    pub pre_norms: Option<Vec<f64>>,
    /// Post-sync param L2 norm (same across ranks after AllReduce).
    /// `None` when not computed (NCCL v1).
    pub post_norm: Option<f64>,
}

impl DivergenceReport {
    /// Max normalized delta (`||pre-post|| / ||post||`) across ranks.
    /// This is the primary cadence signal: worst-case replica drift.
    pub fn max_relative_delta(&self) -> f64 {
        self.deltas.iter().copied().fold(0.0_f64, f64::max)
    }

    /// Per-rank cosine similarity between pre-sync and post-sync params.
    ///
    /// Computed via the identity:
    /// `cos = (||pre||^2 + ||post||^2 - ||pre-post||^2) / (2 * ||pre|| * ||post||)`
    ///
    /// Returns `None` if `pre_norms` or `post_norm` are not available.
    /// Values near 1.0 = directionally aligned, near 0.0 = orthogonal.
    pub fn cosine_similarities(&self) -> Option<Vec<f64>> {
        let pre_norms = self.pre_norms.as_ref()?;
        let post_norm = self.post_norm?;
        if post_norm < 1e-10 {
            return None;
        }
        Some(
            self.deltas
                .iter()
                .zip(pre_norms)
                .map(|(&delta, &pre_norm)| {
                    if pre_norm < 1e-10 {
                        return 0.0;
                    }
                    // ||pre - post||^2 = (delta * ||post||)^2 since delta is normalized
                    let diff_sq = (delta * post_norm).powi(2);
                    let pre_sq = pre_norm.powi(2);
                    let post_sq = post_norm.powi(2);
                    ((pre_sq + post_sq - diff_sq) / (2.0 * pre_norm * post_norm)).clamp(-1.0, 1.0)
                })
                .collect(),
        )
    }

    /// Per-rank magnitude shift: `||pre|| - ||post||`.
    ///
    /// Positive = replica had larger weights than consensus (possible overfitting).
    /// Negative = replica had smaller weights.
    /// Returns `None` if `pre_norms` or `post_norm` are not available.
    pub fn magnitude_shifts(&self) -> Option<Vec<f64>> {
        let pre_norms = self.pre_norms.as_ref()?;
        let post_norm = self.post_norm?;
        Some(pre_norms.iter().map(|&pre| pre - post_norm).collect())
    }
}

// ---------------------------------------------------------------------------
// Convergence action
// ---------------------------------------------------------------------------

/// Recommended cadence adjustment from the convergence guard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceAction {
    /// No divergence concern. Cadence can grow normally.
    Stable,
    /// Divergence trending up. Suppress anchor/overshoot growth (hold current values).
    SuppressGrowth,
    /// High sustained divergence. Nudge anchor down by `factor` (0.0-1.0).
    /// Factor 0.5 = halve the anchor. Same as the existing CPU path behavior.
    NudgeDown { factor: f64 },
}

// ---------------------------------------------------------------------------
// MSF lambda_hat estimator (Phase 1: passive observation)
// ---------------------------------------------------------------------------

/// Per-event sample of the across-event transversal Lyapunov proxy.
///
/// `lambda_raw_t = (1/k_max) * log(D_t / D_{t-1})` where `k_max` is the
/// per-rank step count of the slowest rank between consecutive AllReduces
/// (i.e. the wallclock cadence interval). See
/// `docs/design/msf-cadence-control.md`. Using `k_max` (rather than
/// `sum(steps_since_avg)`) puts the rate on the wallclock-time axis the
/// MSF framework expects; cross-world-size comparison stays meaningful.
/// `None` on the first event or when either operand is below the noise
/// floor (R4 clamp triggers a full estimator reset).
#[derive(Debug, Clone)]
pub struct LambdaSample {
    /// Max normalized delta this event (= `report.max_relative_delta()`).
    pub d_raw: f64,
    /// Across-event proxy `(1/k_max) * log(D_t / D_{t-1})`. `None` on first
    /// event after a reset or below noise floor.
    pub lambda_raw: Option<f64>,
    /// Bias-corrected EMA of `lambda_raw` (C2 in the design doc, default
    /// `alpha = 0.9`). Adam-style `EMA / (1 - alpha^t)` removes the
    /// startup-toward-zero bias for the first ~10 events — exactly the
    /// warmup phase where lambda is most informative.
    pub lambda_ema: Option<f64>,
    /// Cadence interval just completed expressed as total batches consumed
    /// across all ranks (`sum(steps_since_avg)`). Kept for accounting only
    /// — NOT used as the lambda denominator.
    pub k_used: usize,
    /// `max(steps_since_avg)` — wallclock cadence interval per rank.
    /// Used as the lambda denominator (per-rank time axis).
    pub k_max: usize,
}

/// Estimator for the across-event lambda proxy.
///
/// Holds the previous `D_t`, the raw EMA state, and an event counter for
/// Adam-style bias correction. Updated at every AllReduce via
/// [`ConvergenceGuard::observe_lambda`]. Reset across mode/run boundaries
/// (epoch boundaries do NOT reset the estimator: lambda continuity matters
/// across epochs; only mode transitions or full re-init reset).
pub struct LambdaEstimator {
    prev_d: Option<f64>,
    /// Raw EMA — uncorrected, starts at 0.0. Bias correction
    /// `1 / (1 - alpha^t)` is applied at sample time. Adam-style: requires
    /// the EMA to start from 0 so the formula recovers `l_1` exactly on
    /// the first update.
    ema_raw: f64,
    /// Number of `lambda_raw` updates since last reset. `0` means no
    /// observations yet → `lambda_ema` is `None`.
    ema_t: u32,
    alpha: f64,
    noise_floor: f64,
}

impl Default for LambdaEstimator {
    fn default() -> Self {
        Self {
            prev_d: None,
            ema_raw: 0.0,
            ema_t: 0,
            alpha: 0.9,
            noise_floor: 1e-8,
        }
    }
}

impl LambdaEstimator {
    /// Observe a new `D_t` and return the resulting [`LambdaSample`].
    ///
    /// `k_max` (per-rank steps since last AllReduce) is the rate denominator;
    /// `k_used` (sum across ranks) is carried for accounting.
    ///
    /// Below-noise-floor `d_raw` triggers a full estimator reset: `prev_d`
    /// cleared, EMA preserved (state from above-floor events still useful
    /// once the system climbs back). Half-skipping (keep `prev_d`, skip
    /// sample) breaks the time axis on resume, since the next `lambda_raw`
    /// would be computed against the wrong elapsed-time baseline.
    pub fn observe(&mut self, d_raw: f64, k_used: usize, k_max: usize) -> LambdaSample {
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
        if d_raw > self.noise_floor {
            self.prev_d = Some(d_raw);
        } else {
            self.prev_d = None;
        }
        let lambda_ema = if self.ema_t == 0 {
            None
        } else {
            let denom = 1.0 - self.alpha.powi(self.ema_t as i32);
            Some(if denom > 0.0 {
                self.ema_raw / denom
            } else {
                self.ema_raw
            })
        };
        LambdaSample {
            d_raw,
            lambda_raw,
            lambda_ema,
            k_used,
            k_max,
        }
    }

    /// Full reset: clears `prev_d`, EMA, and event counter. Use across
    /// mode/run boundaries.
    pub fn reset(&mut self) {
        self.prev_d = None;
        self.ema_raw = 0.0;
        self.ema_t = 0;
    }
}

// ---------------------------------------------------------------------------
// Per-epoch accumulator
// ---------------------------------------------------------------------------

/// Aggregated divergence + lambda statistics over a single epoch.
///
/// Updated at every AllReduce within the epoch via
/// [`ConvergenceGuard::observe_lambda`]. Drained at epoch boundaries via
/// [`ConvergenceGuard::take_epoch_snapshot`], which resets the accumulator
/// (but not the underlying [`LambdaEstimator`] — lambda continuity carries
/// across epochs).
#[derive(Debug, Clone, Default)]
pub struct EpochSnapshot {
    /// Number of AllReduce events in this epoch.
    pub count: usize,
    /// Number of events that produced a finite `lambda_raw` (excludes the
    /// first event in a fresh estimator and noise-floor-clamped events).
    pub lambda_count: usize,
    pub d_min: f64,
    pub d_max: f64,
    pub d_sum: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub lambda_sum: f64,
    /// Last sample observed in this epoch (snapshot at epoch end).
    pub last_sample: Option<LambdaSample>,
}

impl EpochSnapshot {
    fn new() -> Self {
        Self {
            d_min: f64::INFINITY,
            d_max: f64::NEG_INFINITY,
            lambda_min: f64::INFINITY,
            lambda_max: f64::NEG_INFINITY,
            ..Default::default()
        }
    }

    fn update(&mut self, sample: &LambdaSample) {
        self.count += 1;
        self.d_sum += sample.d_raw;
        if sample.d_raw < self.d_min {
            self.d_min = sample.d_raw;
        }
        if sample.d_raw > self.d_max {
            self.d_max = sample.d_raw;
        }
        if let Some(l) = sample.lambda_raw {
            self.lambda_count += 1;
            self.lambda_sum += l;
            if l < self.lambda_min {
                self.lambda_min = l;
            }
            if l > self.lambda_max {
                self.lambda_max = l;
            }
        }
        self.last_sample = Some(sample.clone());
    }

    /// Mean `D_t` across this epoch's events. Zero if no events.
    pub fn d_mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.d_sum / self.count as f64
        }
    }

    /// Mean `lambda_raw` across this epoch's finite-lambda events. `None` if
    /// no event in this epoch produced a finite `lambda_raw`.
    pub fn lambda_mean(&self) -> Option<f64> {
        if self.lambda_count == 0 {
            None
        } else {
            Some(self.lambda_sum / self.lambda_count as f64)
        }
    }
}

// ---------------------------------------------------------------------------
// Convergence guard
// ---------------------------------------------------------------------------

/// Unified convergence guard for both NCCL and CPU averaging paths.
///
/// Owns the divergence ring buffer and mode-specific reaction logic.
/// The coordinator feeds [`DivergenceReport`]s after each sync event;
/// the guard returns a [`ConvergenceAction`] that the coordinator applies
/// to ElChe's anchor and overshoot controls.
///
/// Also owns the [`LambdaEstimator`] and per-epoch accumulator for MSF
/// passive observation (Phase 1). These are pure measurement: they observe
/// the same quantities the trend rule already computes, with no behavioral
/// effect on cadence.
///
/// The ring buffer persists across sync intervals (required for trend
/// detection). Per-interval accumulators (`nccl_sync_divergence`) live
/// in the coordinator and are reset there.
pub struct ConvergenceGuard {
    policy: ApplyPolicy,
    enabled: bool,
    threshold: f64,
    /// Ring buffer of `max_relative_delta` from recent sync intervals (up to 5).
    history: VecDeque<f64>,
    /// MSF lambda estimator (passive: log-only, no controller effect).
    lambda: LambdaEstimator,
    /// Per-epoch aggregates over [`LambdaSample`]s.
    epoch: EpochSnapshot,
}


impl ConvergenceGuard {
    /// Create a new guard for the given training policy.
    ///
    /// `threshold` controls sensitivity. Set high (e.g. 1.0) for log-only mode
    /// during calibration. A value of ~2x steady-state divergence is typical.
    pub fn new(policy: ApplyPolicy, enabled: bool, threshold: f64) -> Self {
        Self {
            policy,
            enabled,
            threshold,
            history: VecDeque::with_capacity(6),
            lambda: LambdaEstimator::default(),
            epoch: EpochSnapshot::new(),
        }
    }

    /// Observe a divergence event for MSF passive logging.
    ///
    /// Updates the [`LambdaEstimator`] and the per-epoch accumulator. Returns
    /// the resulting [`LambdaSample`] for downstream emission. Call this in
    /// addition to (or after) [`ConvergenceGuard::report`] at every AllReduce.
    pub fn observe_lambda(&mut self, d_raw: f64, k_used: usize, k_max: usize) -> LambdaSample {
        let sample = self.lambda.observe(d_raw, k_used, k_max);
        self.epoch.update(&sample);
        sample
    }

    /// Drain the per-epoch accumulator and return the snapshot.
    ///
    /// Resets the accumulator. Does NOT reset the underlying
    /// [`LambdaEstimator`]: lambda continuity carries across epoch
    /// boundaries (epochs are not theoretical-framework boundaries).
    pub fn take_epoch_snapshot(&mut self) -> EpochSnapshot {
        std::mem::replace(&mut self.epoch, EpochSnapshot::new())
    }

    /// Full reset of the lambda estimator (mode/run boundary).
    ///
    /// Clears `prev_d` and EMA. The trend ring buffer (`history`) is not
    /// touched here; it is owned by the existing reset path.
    pub fn reset_lambda(&mut self) {
        self.lambda.reset();
        self.epoch = EpochSnapshot::new();
    }

    /// Feed a divergence report and get a cadence action.
    ///
    /// Mode-specific logic:
    /// - `Sync`: always `Stable` (sync every batch, divergence near-zero by construction).
    ///   In practice, the worker skips divergence computation entirely for Sync mode.
    /// - `Cadence`: trend detection (3 consecutive rising values above threshold).
    /// - `Async`: same as Cadence for v1 (spike detection deferred to v2).
    pub fn report(&mut self, report: &DivergenceReport) -> ConvergenceAction {
        if !self.enabled {
            return ConvergenceAction::Stable;
        }

        let divergence = report.max_relative_delta();

        // Push to ring buffer (keep last 5 intervals).
        if self.history.len() >= 5 {
            self.history.pop_front();
        }
        self.history.push_back(divergence);

        match self.policy {
            ApplyPolicy::Sync => ConvergenceAction::Stable,
            ApplyPolicy::Cadence | ApplyPolicy::Async => self.check_trend(),
        }
    }

    /// Trend detection: 3 consecutive rising values, latest above threshold.
    ///
    /// Filters out single spikes and noise. Only sustained drift suppresses
    /// growth. Returns `SuppressGrowth` (not `NudgeDown`) because aggressive
    /// anchor reduction on trends alone can kill convergence. The overhead
    /// auto-tune in ElChe handles loosening.
    fn check_trend(&self) -> ConvergenceAction {
        if self.history.len() < 3 {
            return ConvergenceAction::Stable;
        }

        let len = self.history.len();
        let rising = self.history[len - 1] > self.history[len - 2]
            && self.history[len - 2] > self.history[len - 3]
            && self.history[len - 1] > self.threshold;

        if rising {
            crate::verbose!(
                "  ddp: weight-space divergence trending up | history={:.4?} | suppressing growth",
                Vec::from(self.history.clone()),
            );
            ConvergenceAction::SuppressGrowth
        } else {
            ConvergenceAction::Stable
        }
    }

    /// Direct access to divergence history for testing and monitoring.
    pub fn history(&self) -> &VecDeque<f64> {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_report(deltas: &[f64]) -> DivergenceReport {
        DivergenceReport {
            deltas: deltas.to_vec(),
            pre_norms: None,
            post_norm: None,
        }
    }

    fn make_full_report(
        deltas: &[f64],
        pre_norms: &[f64],
        post_norm: f64,
    ) -> DivergenceReport {
        DivergenceReport {
            deltas: deltas.to_vec(),
            pre_norms: Some(pre_norms.to_vec()),
            post_norm: Some(post_norm),
        }
    }

    // --- DivergenceReport ---

    #[test]
    fn max_relative_delta_picks_worst_rank() {
        let r = make_report(&[0.01, 0.05, 0.03]);
        assert!((r.max_relative_delta() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn max_relative_delta_empty_is_zero() {
        let r = make_report(&[]);
        assert_eq!(r.max_relative_delta(), 0.0);
    }

    #[test]
    fn cosine_similarities_none_when_missing_norms() {
        let r = make_report(&[0.01, 0.02]);
        assert!(r.cosine_similarities().is_none());
        assert!(r.magnitude_shifts().is_none());
    }

    #[test]
    fn cosine_similarities_correct_for_small_delta() {
        // Small delta -> pre and post nearly identical -> cos near 1.0
        let r = make_full_report(&[0.001], &[10.0], 10.0);
        let cos = r.cosine_similarities().unwrap();
        assert!(cos[0] > 0.999, "expected cos near 1.0, got {}", cos[0]);
    }

    #[test]
    fn magnitude_shifts_correct() {
        let r = make_full_report(&[0.01, 0.02], &[10.5, 9.8], 10.0);
        let shifts = r.magnitude_shifts().unwrap();
        assert!((shifts[0] - 0.5).abs() < 1e-10);
        assert!((shifts[1] - (-0.2)).abs() < 1e-10);
    }

    // --- ConvergenceGuard ---

    #[test]
    fn sync_always_stable() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Sync, true, 0.01);
        for _ in 0..10 {
            let action = g.report(&make_report(&[0.5, 0.5]));
            assert_eq!(action, ConvergenceAction::Stable);
        }
    }

    #[test]
    fn cadence_trend_suppresses_growth() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        // 3 rising values above threshold
        assert_eq!(g.report(&make_report(&[0.02])), ConvergenceAction::Stable); // only 1
        assert_eq!(g.report(&make_report(&[0.03])), ConvergenceAction::Stable); // only 2
        assert_eq!(
            g.report(&make_report(&[0.04])),
            ConvergenceAction::SuppressGrowth
        ); // 3 rising
    }

    #[test]
    fn cadence_non_rising_is_stable() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        g.report(&make_report(&[0.05]));
        g.report(&make_report(&[0.04])); // dropped
        assert_eq!(
            g.report(&make_report(&[0.06])),
            ConvergenceAction::Stable
        ); // not 3 consecutive rising
    }

    #[test]
    fn below_threshold_is_stable() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.10);
        // Rising but below threshold
        g.report(&make_report(&[0.01]));
        g.report(&make_report(&[0.02]));
        assert_eq!(
            g.report(&make_report(&[0.03])),
            ConvergenceAction::Stable
        );
    }

    #[test]
    fn disabled_always_stable() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, false, 0.01);
        g.report(&make_report(&[0.1]));
        g.report(&make_report(&[0.2]));
        assert_eq!(
            g.report(&make_report(&[0.3])),
            ConvergenceAction::Stable
        );
    }

    #[test]
    fn history_capped_at_5() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        for i in 0..10 {
            g.report(&make_report(&[i as f64 * 0.01]));
        }
        assert_eq!(g.history().len(), 5);
    }

    // --- LambdaEstimator ---

    #[test]
    fn lambda_first_event_is_none() {
        let mut e = LambdaEstimator::default();
        let s = e.observe(0.05, 8, 4);
        assert!(s.lambda_raw.is_none());
        assert!(s.lambda_ema.is_none());
        assert!((s.d_raw - 0.05).abs() < 1e-12);
        assert_eq!(s.k_used, 8);
        assert_eq!(s.k_max, 4);
    }

    #[test]
    fn lambda_growth_positive_uses_k_max() {
        let mut e = LambdaEstimator::default();
        e.observe(0.01, 8, 4);
        let s = e.observe(0.02, 8, 4);
        // Denominator is k_max (per-rank steps), NOT k_used (sum across ranks).
        // (1/4) * ln(2) ~= 0.1733.
        let expected = std::f64::consts::LN_2 / 4.0;
        let got = s.lambda_raw.expect("second event must produce lambda");
        assert!((got - expected).abs() < 1e-10);
        assert!(got > 0.0);
    }

    #[test]
    fn lambda_decay_negative_uses_k_max() {
        let mut e = LambdaEstimator::default();
        e.observe(0.04, 8, 4);
        let s = e.observe(0.02, 8, 4);
        // (1/k_max) * ln(0.5) = -ln(2)/4 ~= -0.1733
        let expected = -std::f64::consts::LN_2 / 4.0;
        let got = s.lambda_raw.expect("second event must produce lambda");
        assert!((got - expected).abs() < 1e-10);
        assert!(got < 0.0);
    }

    #[test]
    fn lambda_k_max_unchanged_by_world_size() {
        // Two estimators see the same per-rank trajectory but different
        // world sizes. With k_max as the denominator, they must agree;
        // the previous k_used=sum convention would have given a 2x split.
        let mut e_two = LambdaEstimator::default();
        e_two.observe(0.01, 8, 4); // 2 ranks, 4 steps each
        let s_two = e_two.observe(0.02, 8, 4);
        let mut e_three = LambdaEstimator::default();
        e_three.observe(0.01, 12, 4); // 3 ranks, 4 steps each
        let s_three = e_three.observe(0.02, 12, 4);
        let l_two = s_two.lambda_raw.unwrap();
        let l_three = s_three.lambda_raw.unwrap();
        assert!((l_two - l_three).abs() < 1e-12, "{l_two} vs {l_three}");
    }

    #[test]
    fn lambda_noise_floor_resets_estimator() {
        let mut e = LambdaEstimator::default();
        e.observe(0.01, 8, 4);
        // d_raw below noise floor -> full reset: lambda_raw None, prev_d cleared.
        let s = e.observe(1e-12, 8, 4);
        assert!(s.lambda_raw.is_none());
        // Next event treats this as the first post-reset sample: lambda_raw None.
        // Half-skipping (keeping prev_d=0.01) would have produced ln(2)/k_max
        // off the wrong baseline — that contract was wrong on the time axis.
        let s2 = e.observe(0.02, 8, 4);
        assert!(s2.lambda_raw.is_none());
        // Subsequent event resumes lambda computation against the post-reset prev_d.
        let s3 = e.observe(0.04, 8, 4);
        let expected = std::f64::consts::LN_2 / 4.0;
        assert!((s3.lambda_raw.unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn lambda_ema_bias_corrected_at_warmup() {
        // First lambda_raw of e.g. 1.0 should produce a bias-corrected EMA
        // also at 1.0 (not 1.0 * 0.1 = 0.1, the raw EMA value). Adam-style
        // 1/(1-alpha^t) correction removes the startup-toward-zero pull.
        let mut e = LambdaEstimator::default();
        e.observe(1.0, 1, 1);
        let s = e.observe(std::f64::consts::E, 1, 1);
        let lambda = s.lambda_raw.unwrap();
        let ema = s.lambda_ema.unwrap();
        // Both should be ~1.0; bias-corrected EMA matches lambda_raw exactly
        // on the first update.
        assert!((lambda - 1.0).abs() < 1e-10, "lambda={lambda}");
        assert!((ema - 1.0).abs() < 1e-10, "ema={ema}");
    }

    #[test]
    fn lambda_ema_smooths_over_constant_input() {
        let mut e = LambdaEstimator::default();
        // Feed a sequence where consecutive ratios are constant = e^1.
        // lambda_raw = 1.0 every step at k_max=1; bias-corrected EMA must
        // sit at 1.0 every step (constant input -> no bias to correct away).
        let mut prev = 1.0_f64;
        for _ in 0..200 {
            let next = prev * std::f64::consts::E;
            e.observe(next, 1, 1);
            prev = next;
        }
        // Raw EMA also converges to 1.0 since constant input has no startup bias.
        assert!((e.ema_raw - 1.0).abs() < 1e-3, "raw_ema = {}", e.ema_raw);
    }

    #[test]
    fn lambda_zero_d_does_not_crash() {
        let mut e = LambdaEstimator::default();
        let s1 = e.observe(0.0, 8, 4);
        assert!(s1.lambda_raw.is_none());
        let s2 = e.observe(0.0, 8, 4);
        assert!(s2.lambda_raw.is_none());
    }

    // --- EpochSnapshot ---

    #[test]
    fn epoch_snapshot_aggregates() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        g.observe_lambda(0.04, 8, 4);
        g.observe_lambda(0.02, 8, 4);
        g.observe_lambda(0.03, 8, 4);
        let snap = g.take_epoch_snapshot();
        assert_eq!(snap.count, 3);
        assert!((snap.d_min - 0.02).abs() < 1e-12);
        assert!((snap.d_max - 0.04).abs() < 1e-12);
        assert!((snap.d_mean() - 0.03).abs() < 1e-12);
        assert_eq!(snap.lambda_count, 2); // first event has no lambda
        assert!(snap.lambda_mean().is_some());
        assert!(snap.last_sample.is_some());
    }

    #[test]
    fn epoch_snapshot_resets_accumulator_not_prev_d() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        g.observe_lambda(0.01, 8, 4);
        g.observe_lambda(0.02, 8, 4);
        let snap1 = g.take_epoch_snapshot();
        assert_eq!(snap1.count, 2);
        // After snapshot, accumulator empty but prev_d still 0.02.
        // Next event computes lambda against 0.02, so doubling -> +ln(2)/k_max.
        let s = g.observe_lambda(0.04, 8, 4);
        let expected = std::f64::consts::LN_2 / 4.0;
        assert!((s.lambda_raw.unwrap() - expected).abs() < 1e-10);
        let snap2 = g.take_epoch_snapshot();
        assert_eq!(snap2.count, 1);
        assert_eq!(snap2.lambda_count, 1);
    }

    #[test]
    fn reset_lambda_clears_prev_d() {
        let mut g = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        g.observe_lambda(0.01, 8, 4);
        g.observe_lambda(0.02, 8, 4);
        g.reset_lambda();
        // After reset, next event has no prev_d -> lambda_raw None
        let s = g.observe_lambda(0.04, 8, 4);
        assert!(s.lambda_raw.is_none());
    }

    // --- ConvergenceGuard (existing tests continued) ---

    #[test]
    fn async_same_as_cadence_v1() {
        let mut gc = ConvergenceGuard::new(ApplyPolicy::Cadence, true, 0.01);
        let mut ga = ConvergenceGuard::new(ApplyPolicy::Async, true, 0.01);
        let reports: Vec<DivergenceReport> =
            vec![make_report(&[0.02]), make_report(&[0.03]), make_report(&[0.04])];
        for r in &reports {
            let ac = gc.report(r);
            let aa = ga.report(r);
            assert_eq!(ac, aa);
        }
    }
}
