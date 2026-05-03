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
/// `lambda_raw_t = (1/k) * log(D_t / D_{t-1})` from the MSF cadence-control
/// design (`docs/design/msf-cadence-control.md`). `None` on the first event
/// or when either operand is below the noise floor (R4 clamp).
#[derive(Debug, Clone)]
pub struct LambdaSample {
    /// Max normalized delta this event (= `report.max_relative_delta()`).
    pub d_raw: f64,
    /// Across-event proxy `(1/k) * log(D_t / D_{t-1})`. `None` on first event
    /// or below noise floor.
    pub lambda_raw: Option<f64>,
    /// EMA-smoothed `lambda_raw` (C2 in the design doc, default `alpha = 0.9`).
    pub lambda_ema: Option<f64>,
    /// Cadence interval just completed: `sum(steps_since_avg)` across ranks.
    pub k_used: usize,
    /// `max(steps_since_avg)` — useful under heterogeneous DDP where ranks
    /// drift apart. Equal to `k_used / world_size` only for homogeneous Sync.
    pub k_max: usize,
}

/// Estimator for the across-event lambda proxy.
///
/// Holds the previous `D_t` and the EMA state. Updated at every AllReduce
/// via [`ConvergenceGuard::observe_lambda`]. Reset across mode/run boundaries
/// (epoch boundaries do NOT reset the estimator: lambda continuity matters
/// across epochs; only mode transitions or full re-init reset).
pub struct LambdaEstimator {
    prev_d: Option<f64>,
    ema: Option<f64>,
    alpha: f64,
    noise_floor: f64,
}

impl Default for LambdaEstimator {
    fn default() -> Self {
        Self {
            prev_d: None,
            ema: None,
            alpha: 0.9,
            noise_floor: 1e-8,
        }
    }
}

impl LambdaEstimator {
    /// Observe a new `D_t` and return the resulting [`LambdaSample`].
    pub fn observe(&mut self, d_raw: f64, k_used: usize, k_max: usize) -> LambdaSample {
        let lambda_raw = match self.prev_d {
            Some(prev) if prev > self.noise_floor && d_raw > self.noise_floor && k_used > 0 => {
                Some((d_raw / prev).ln() / k_used as f64)
            }
            _ => None,
        };
        if let Some(l) = lambda_raw {
            self.ema = Some(match self.ema {
                Some(e) => self.alpha * e + (1.0 - self.alpha) * l,
                None => l,
            });
        }
        if d_raw > self.noise_floor {
            self.prev_d = Some(d_raw);
        }
        LambdaSample {
            d_raw,
            lambda_raw,
            lambda_ema: self.ema,
            k_used,
            k_max,
        }
    }

    /// Full reset: clears `prev_d` and EMA. Use across mode/run boundaries.
    pub fn reset(&mut self) {
        self.prev_d = None;
        self.ema = None;
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
    fn lambda_growth_positive() {
        let mut e = LambdaEstimator::default();
        e.observe(0.01, 8, 4);
        let s = e.observe(0.02, 8, 4);
        // (1/8) * ln(2) ~= 0.0866
        let expected = std::f64::consts::LN_2 / 8.0;
        let got = s.lambda_raw.expect("second event must produce lambda");
        assert!((got - expected).abs() < 1e-10);
        assert!(got > 0.0);
    }

    #[test]
    fn lambda_decay_negative() {
        let mut e = LambdaEstimator::default();
        e.observe(0.04, 8, 4);
        let s = e.observe(0.02, 8, 4);
        // (1/8) * ln(0.5) = -ln(2)/8 ~= -0.0866
        let expected = -std::f64::consts::LN_2 / 8.0;
        let got = s.lambda_raw.expect("second event must produce lambda");
        assert!((got - expected).abs() < 1e-10);
        assert!(got < 0.0);
    }

    #[test]
    fn lambda_noise_floor_clamps() {
        let mut e = LambdaEstimator::default();
        e.observe(0.01, 8, 4);
        // d_raw below noise floor -> lambda_raw None and prev_d unchanged
        let s = e.observe(1e-12, 8, 4);
        assert!(s.lambda_raw.is_none());
        // Next event uses prev_d = 0.01 (the last valid one), not 1e-12
        let s2 = e.observe(0.02, 8, 4);
        let expected = std::f64::consts::LN_2 / 8.0;
        assert!((s2.lambda_raw.unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn lambda_ema_smooths_over_constant_input() {
        let mut e = LambdaEstimator::default();
        // After many constant +1 events, EMA should approach the input.
        e.observe(1.0, 1, 1); // first: lambda None, ema None
        for _ in 0..200 {
            // d goes 1, e, e^2, ... -> lambda_raw = 1.0/k = 1.0 each step
            // Use prev=1.0, d=e^1, then prev=e, d=e^2, etc.
        }
        // Simpler: feed a sequence where consecutive ratios are constant=e^1.
        let mut prev = 1.0_f64;
        for _ in 0..200 {
            let next = prev * std::f64::consts::E;
            e.observe(next, 1, 1);
            prev = next;
        }
        let ema = e.ema.expect("ema must be Some after many events");
        // lambda_raw is 1.0 every step, EMA must converge to 1.0
        assert!((ema - 1.0).abs() < 1e-3, "ema = {ema}");
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
        // Next event computes lambda against 0.02, so doubling -> +ln(2)/8.
        let s = g.observe_lambda(0.04, 8, 4);
        let expected = std::f64::consts::LN_2 / 8.0;
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
