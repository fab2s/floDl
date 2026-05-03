//! Pluggable convergence guards for DDP parameter averaging.
//!
//! A guard observes the per-AllReduce divergence trajectory and recommends
//! cadence adjustments through a [`ConvergenceAction`]. Three concrete impls
//! ship out of the box:
//!
//! - [`NoGuard`]: passive baseline, always [`ConvergenceAction::Stable`]. Use
//!   to collect an unconditioned trajectory for fair guard comparison.
//! - [`TrendGuard`]: production default. Three-rises-above-threshold rule on
//!   the per-rank `||pre - post|| / ||post||` ring buffer. Returns
//!   [`ConvergenceAction::SuppressGrowth`] on persistent rising drift.
//! - [`MsfGuard`]: rate-based detector built on the across-event MSF proxy
//!   `λ_ema = EMA((1/k_max) * log(D_t / D_{t-1}))`. Soft + hard thresholds:
//!   sustained `λ_ema > suppress_threshold` → `SuppressGrowth`; sustained
//!   `λ_ema > nudge_threshold` → [`ConvergenceAction::NudgeDown`].
//!
//! All three implement [`ConvergenceGuard`]. The Trainer accepts any
//! `impl ConvergenceGuard + 'static` via `DdpBuilder::convergence_guard`;
//! the coordinator boxes it internally.
//!
//! See `docs/design/msf-cadence-control.md` for the theoretical framing
//! and the soft/hard threshold derivation.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Divergence report
// ---------------------------------------------------------------------------

/// Divergence measurements from a single sync event.
///
/// The primary signal is `deltas` — per-rank normalized L2 distance from the
/// replica's pre-sync params to the post-sync average. This captures both
/// magnitude drift and directional divergence.
///
/// When `pre_norms` and `post_norm` are available, the decomposition into
/// magnitude shift and cosine similarity is free via:
/// `cos = (||pre||^2 + ||post||^2 - (delta * ||post||)^2) / (2 * ||pre|| * ||post||)`
pub struct DivergenceReport {
    /// Per-rank normalized delta: `||pre - post|| / ||post||`.
    pub deltas: Vec<f64>,
    /// Per-rank pre-sync param L2 norm. `None` when not computed.
    pub pre_norms: Option<Vec<f64>>,
    /// Post-sync param L2 norm (same across ranks after AllReduce).
    /// `None` when not computed.
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
    /// Positive = replica had larger weights than consensus.
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
    /// Factor 0.5 = halve the anchor.
    NudgeDown { factor: f64 },
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Pluggable convergence-monitoring strategy.
///
/// Called by the coordinator after every AllReduce with a [`DivergenceReport`]
/// plus the cadence interval (`k_used` = total batches across ranks since
/// last AllReduce, `k_max` = max per-rank steps = wallclock interval). The
/// guard returns a [`ConvergenceAction`] applied to ElChe's anchor and
/// overshoot controls.
///
/// Implementations are owned per-run and are not thread-shared (the
/// coordinator runs `report` on a single thread). `Send` is required so the
/// guard can be moved into the DDP runtime; `Sync` is a future-proofing bound
/// for cross-thread observation.
pub trait ConvergenceGuard: Send + Sync {
    /// Observe a divergence event and return the recommended action.
    fn report(
        &mut self,
        report: &DivergenceReport,
        k_used: usize,
        k_max: usize,
    ) -> ConvergenceAction;

    /// Guard-specific diagnostic values for the current observation. Default
    /// empty. Used to populate the `EventKind::GuardTelemetry` timeline event
    /// after each `report` call.
    fn telemetry(&self) -> Vec<(&'static str, f64)> {
        Vec::new()
    }

    /// Reset guard state (clear history / estimator). Called at mode/run
    /// boundaries. Default no-op for stateless guards.
    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// MSF lambda estimator (used by MsfGuard)
// ---------------------------------------------------------------------------

/// Per-event sample of the across-event transversal Lyapunov proxy.
///
/// `lambda_raw_t = (1/k_max) * log(D_t / D_{t-1})` where `k_max` is the
/// per-rank step count of the slowest rank between consecutive AllReduces
/// (i.e. the wallclock cadence interval). See
/// `docs/design/msf-cadence-control.md`.
#[derive(Debug, Clone)]
pub struct LambdaSample {
    /// Max normalized delta this event (= `report.max_relative_delta()`).
    pub d_raw: f64,
    /// Across-event proxy `(1/k_max) * log(D_t / D_{t-1})`. `None` on first
    /// event after a reset or below noise floor.
    pub lambda_raw: Option<f64>,
    /// Bias-corrected EMA of `lambda_raw` (Adam-style `EMA / (1 - alpha^t)`).
    pub lambda_ema: Option<f64>,
    /// Cadence interval just completed expressed as total batches across ranks.
    /// Kept for accounting only — NOT used as the lambda denominator.
    pub k_used: usize,
    /// `max(steps_since_avg)` — wallclock cadence interval per rank.
    /// Used as the lambda denominator (per-rank time axis).
    pub k_max: usize,
}

/// Estimator for the across-event lambda proxy.
///
/// Holds the previous `D_t`, the raw EMA state, and an event counter for
/// Adam-style bias correction. Reset across mode/run boundaries (epoch
/// boundaries do NOT reset: lambda continuity matters across epochs).
pub struct LambdaEstimator {
    prev_d: Option<f64>,
    /// Raw EMA — uncorrected, starts at 0.0.
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
    /// Construct with a custom alpha. `noise_floor` keeps its default 1e-8.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            ..Default::default()
        }
    }

    /// Observe a new `D_t` and return the resulting [`LambdaSample`].
    ///
    /// `k_max` (per-rank steps since last AllReduce) is the rate denominator;
    /// `k_used` (sum across ranks) is carried for accounting.
    ///
    /// Below-noise-floor `d_raw` triggers a full estimator reset.
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
        self.prev_d = if d_raw > self.noise_floor {
            Some(d_raw)
        } else {
            None
        };
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

    /// Full reset.
    pub fn reset(&mut self) {
        self.prev_d = None;
        self.ema_raw = 0.0;
        self.ema_t = 0;
    }
}

// ---------------------------------------------------------------------------
// NoGuard
// ---------------------------------------------------------------------------

/// Always-stable guard. Use as the unconditioned baseline trajectory
/// generator: lets ElChe's overhead-tune drive cadence with no convergence
/// intervention. The right control-experiment partner for any other guard.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoGuard;

impl ConvergenceGuard for NoGuard {
    fn report(&mut self, _: &DivergenceReport, _: usize, _: usize) -> ConvergenceAction {
        ConvergenceAction::Stable
    }
}

// ---------------------------------------------------------------------------
// TrendGuard (production default)
// ---------------------------------------------------------------------------

/// Production guard: 3-consecutive-rises-above-threshold rule on
/// `max_relative_delta` history. Suppresses anchor growth on persistent
/// rising drift. Does not currently issue [`ConvergenceAction::NudgeDown`]
/// — adding that is a separate decision (the trend signal is too noisy on
/// its own to drive aggressive anchor reduction).
pub struct TrendGuard {
    threshold: f64,
    /// Ring buffer of `max_relative_delta` from recent sync intervals (up to 5).
    history: VecDeque<f64>,
}

impl Default for TrendGuard {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl TrendGuard {
    /// Build with the given divergence threshold.
    ///
    /// Set high (e.g. 1.0) for log-only mode during calibration.
    /// `~2x steady-state divergence` is typical for active operation.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            history: VecDeque::with_capacity(6),
        }
    }

    /// Builder: set the divergence threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Direct access to divergence history for testing and monitoring.
    pub fn history(&self) -> &VecDeque<f64> {
        &self.history
    }

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
}

impl ConvergenceGuard for TrendGuard {
    fn report(&mut self, report: &DivergenceReport, _: usize, _: usize) -> ConvergenceAction {
        let divergence = report.max_relative_delta();
        if self.history.len() >= 5 {
            self.history.pop_front();
        }
        self.history.push_back(divergence);
        self.check_trend()
    }

    fn telemetry(&self) -> Vec<(&'static str, f64)> {
        // Latest observation only — mirror the production signal in the
        // timeline so dashboards can plot the trend buffer's head.
        match self.history.back() {
            Some(&d) => vec![("d_history_last", d)],
            None => Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// MsfGuard
// ---------------------------------------------------------------------------

/// MSF-based rate detector with soft (suppress) + hard (nudge) thresholds.
///
/// Tracks `λ_ema` (bias-corrected EMA of `(1/k_max) * log(D_t / D_{t-1})`)
/// and escalates:
///
/// - `λ_ema > suppress_threshold` for `suppress_sustain` consecutive events
///   → [`ConvergenceAction::SuppressGrowth`]
/// - `λ_ema > nudge_threshold` for `nudge_sustain` consecutive events
///   → [`ConvergenceAction::NudgeDown`] with `nudge_factor`
///
/// Streaks are independent: nudge fires the first time its own threshold has
/// been sustained (it doesn't require suppress to have fired first). Either
/// fire resets only its own streak, so the guard can fall back from nudge to
/// suppress as `λ_ema` decays.
///
/// Disable nudge by setting `nudge_threshold = f64::INFINITY` (or via
/// [`MsfGuard::without_nudge`]).
pub struct MsfGuard {
    estimator: LambdaEstimator,
    suppress_threshold: f64,
    suppress_sustain: usize,
    nudge_threshold: f64,
    nudge_sustain: usize,
    nudge_factor: f64,
    suppress_streak: usize,
    nudge_streak: usize,
    /// Latest sample (cached for telemetry).
    last_sample: Option<LambdaSample>,
}

impl Default for MsfGuard {
    fn default() -> Self {
        Self {
            estimator: LambdaEstimator::default(),
            suppress_threshold: 1.0e-3,
            suppress_sustain: 3,
            nudge_threshold: 1.0e-2,
            nudge_sustain: 3,
            nudge_factor: 0.5,
            suppress_streak: 0,
            nudge_streak: 0,
            last_sample: None,
        }
    }
}

impl MsfGuard {
    /// Builder: set EMA smoothing coefficient (0.0-1.0). Default 0.9.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.estimator = LambdaEstimator::with_alpha(alpha);
        self
    }

    /// Builder: configure the soft (SuppressGrowth) trigger.
    pub fn with_suppress(mut self, threshold: f64, sustain: usize) -> Self {
        self.suppress_threshold = threshold;
        self.suppress_sustain = sustain;
        self
    }

    /// Builder: configure the hard (NudgeDown) trigger.
    ///
    /// `factor` is the ElChe anchor reduction multiplier (0.5 = halve).
    pub fn with_nudge(mut self, threshold: f64, sustain: usize, factor: f64) -> Self {
        self.nudge_threshold = threshold;
        self.nudge_sustain = sustain;
        self.nudge_factor = factor;
        self
    }

    /// Builder: disable the NudgeDown trigger entirely (only SuppressGrowth fires).
    pub fn without_nudge(mut self) -> Self {
        self.nudge_threshold = f64::INFINITY;
        self
    }

    /// Last observation, primarily for testing.
    pub fn last_sample(&self) -> Option<&LambdaSample> {
        self.last_sample.as_ref()
    }
}

impl ConvergenceGuard for MsfGuard {
    fn report(
        &mut self,
        report: &DivergenceReport,
        k_used: usize,
        k_max: usize,
    ) -> ConvergenceAction {
        let d_raw = report.max_relative_delta();
        let sample = self.estimator.observe(d_raw, k_used, k_max);
        let lambda_ema = sample.lambda_ema;
        self.last_sample = Some(sample);

        // Streak update. Each threshold has its own counter so the guard can
        // de-escalate naturally as λ_ema decays past nudge but stays above
        // suppress.
        let lema = lambda_ema.unwrap_or(f64::NEG_INFINITY);
        if lema > self.suppress_threshold {
            self.suppress_streak += 1;
        } else {
            self.suppress_streak = 0;
        }
        if lema > self.nudge_threshold {
            self.nudge_streak += 1;
        } else {
            self.nudge_streak = 0;
        }

        // Hard trigger wins over soft when both fire on the same event.
        if self.nudge_streak >= self.nudge_sustain && self.nudge_threshold.is_finite() {
            self.nudge_streak = 0;
            crate::verbose!(
                "  ddp: msf λ_ema={:.4e} sustained > nudge_threshold {:.4e} | nudging anchor down by {:.2}",
                lema, self.nudge_threshold, self.nudge_factor,
            );
            return ConvergenceAction::NudgeDown {
                factor: self.nudge_factor,
            };
        }
        if self.suppress_streak >= self.suppress_sustain {
            self.suppress_streak = 0;
            crate::verbose!(
                "  ddp: msf λ_ema={:.4e} sustained > suppress_threshold {:.4e} | suppressing growth",
                lema, self.suppress_threshold,
            );
            return ConvergenceAction::SuppressGrowth;
        }
        ConvergenceAction::Stable
    }

    fn telemetry(&self) -> Vec<(&'static str, f64)> {
        let s = match &self.last_sample {
            Some(s) => s,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(2);
        if let Some(l) = s.lambda_raw {
            out.push(("lambda_raw", l));
        }
        if let Some(l) = s.lambda_ema {
            out.push(("lambda_ema", l));
        }
        out
    }

    fn reset(&mut self) {
        self.estimator.reset();
        self.suppress_streak = 0;
        self.nudge_streak = 0;
        self.last_sample = None;
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

    fn make_full_report(deltas: &[f64], pre_norms: &[f64], post_norm: f64) -> DivergenceReport {
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

    // --- NoGuard ---

    #[test]
    fn no_guard_is_always_stable() {
        let mut g = NoGuard;
        for _ in 0..10 {
            assert_eq!(
                g.report(&make_report(&[0.5, 0.5]), 8, 4),
                ConvergenceAction::Stable
            );
        }
    }

    #[test]
    fn no_guard_emits_no_telemetry() {
        let g = NoGuard;
        assert!(g.telemetry().is_empty());
    }

    // --- TrendGuard ---

    #[test]
    fn trend_default_threshold_is_0_01() {
        let g = TrendGuard::default();
        // Behavioural check: 3 rising values just above 0.01 should fire.
        let mut g = g;
        g.report(&make_report(&[0.011]), 8, 4);
        g.report(&make_report(&[0.012]), 8, 4);
        assert_eq!(
            g.report(&make_report(&[0.013]), 8, 4),
            ConvergenceAction::SuppressGrowth
        );
    }

    #[test]
    fn trend_3_rises_above_threshold_suppress() {
        let mut g = TrendGuard::new(0.01);
        assert_eq!(
            g.report(&make_report(&[0.02]), 8, 4),
            ConvergenceAction::Stable
        );
        assert_eq!(
            g.report(&make_report(&[0.03]), 8, 4),
            ConvergenceAction::Stable
        );
        assert_eq!(
            g.report(&make_report(&[0.04]), 8, 4),
            ConvergenceAction::SuppressGrowth
        );
    }

    #[test]
    fn trend_non_rising_is_stable() {
        let mut g = TrendGuard::new(0.01);
        g.report(&make_report(&[0.05]), 8, 4);
        g.report(&make_report(&[0.04]), 8, 4); // dropped
        assert_eq!(
            g.report(&make_report(&[0.06]), 8, 4),
            ConvergenceAction::Stable
        );
    }

    #[test]
    fn trend_below_threshold_is_stable() {
        let mut g = TrendGuard::new(0.10);
        g.report(&make_report(&[0.01]), 8, 4);
        g.report(&make_report(&[0.02]), 8, 4);
        assert_eq!(
            g.report(&make_report(&[0.03]), 8, 4),
            ConvergenceAction::Stable
        );
    }

    #[test]
    fn trend_history_capped_at_5() {
        let mut g = TrendGuard::new(0.01);
        for i in 0..10 {
            g.report(&make_report(&[i as f64 * 0.01]), 8, 4);
        }
        assert_eq!(g.history().len(), 5);
    }

    #[test]
    fn trend_reset_clears_history() {
        let mut g = TrendGuard::new(0.01);
        for _ in 0..5 {
            g.report(&make_report(&[0.05]), 8, 4);
        }
        g.reset();
        assert!(g.history().is_empty());
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
        let expected = std::f64::consts::LN_2 / 4.0;
        let got = s.lambda_raw.expect("second event must produce lambda");
        assert!((got - expected).abs() < 1e-10);
    }

    #[test]
    fn lambda_decay_negative_uses_k_max() {
        let mut e = LambdaEstimator::default();
        e.observe(0.04, 8, 4);
        let s = e.observe(0.02, 8, 4);
        let expected = -std::f64::consts::LN_2 / 4.0;
        let got = s.lambda_raw.expect("second event must produce lambda");
        assert!((got - expected).abs() < 1e-10);
    }

    #[test]
    fn lambda_k_max_unchanged_by_world_size() {
        let mut e_two = LambdaEstimator::default();
        e_two.observe(0.01, 8, 4);
        let s_two = e_two.observe(0.02, 8, 4);
        let mut e_three = LambdaEstimator::default();
        e_three.observe(0.01, 12, 4);
        let s_three = e_three.observe(0.02, 12, 4);
        let l_two = s_two.lambda_raw.unwrap();
        let l_three = s_three.lambda_raw.unwrap();
        assert!((l_two - l_three).abs() < 1e-12, "{l_two} vs {l_three}");
    }

    #[test]
    fn lambda_noise_floor_resets_estimator() {
        let mut e = LambdaEstimator::default();
        e.observe(0.01, 8, 4);
        let s = e.observe(1e-12, 8, 4);
        assert!(s.lambda_raw.is_none());
        let s2 = e.observe(0.02, 8, 4);
        assert!(s2.lambda_raw.is_none());
        let s3 = e.observe(0.04, 8, 4);
        let expected = std::f64::consts::LN_2 / 4.0;
        assert!((s3.lambda_raw.unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn lambda_ema_bias_corrected_at_warmup() {
        let mut e = LambdaEstimator::default();
        e.observe(1.0, 1, 1);
        let s = e.observe(std::f64::consts::E, 1, 1);
        let lambda = s.lambda_raw.unwrap();
        let ema = s.lambda_ema.unwrap();
        assert!((lambda - 1.0).abs() < 1e-10, "lambda={lambda}");
        assert!((ema - 1.0).abs() < 1e-10, "ema={ema}");
    }

    #[test]
    fn lambda_ema_smooths_over_constant_input() {
        let mut e = LambdaEstimator::default();
        let mut prev = 1.0_f64;
        for _ in 0..200 {
            let next = prev * std::f64::consts::E;
            e.observe(next, 1, 1);
            prev = next;
        }
        assert!((e.ema_raw - 1.0).abs() < 1e-3, "raw_ema = {}", e.ema_raw);
    }

    // --- MsfGuard ---

    #[test]
    fn msf_default_starts_stable() {
        let mut g = MsfGuard::default();
        let s = g.report(&make_report(&[0.01]), 8, 4);
        assert_eq!(s, ConvergenceAction::Stable);
    }

    #[test]
    fn msf_suppress_fires_after_sustain() {
        // Feed an exponentially-rising D so λ_ema climbs quickly and stays
        // above suppress_threshold for ≥ suppress_sustain events.
        let mut g = MsfGuard::default()
            .with_suppress(1.0e-3, 3)
            .with_nudge(f64::INFINITY, 3, 0.5); // disable nudge
        // Drive λ_raw ≈ ln(2)/4 ≈ 0.173 every event so EMA quickly clears 1e-3.
        let mut prev = 1.0e-4;
        let mut fired = false;
        for _ in 0..20 {
            let next = prev * 2.0;
            let action = g.report(&make_report(&[next]), 8, 4);
            if matches!(action, ConvergenceAction::SuppressGrowth) {
                fired = true;
                break;
            }
            prev = next;
        }
        assert!(fired, "expected SuppressGrowth within 20 events");
    }

    #[test]
    fn msf_nudge_fires_on_sustained_steeper_growth() {
        // Same shape but nudge_threshold is also 1e-3 → fires immediately
        // once λ_ema clears the bar (default suppress 1e-3 fires too, but
        // nudge wins when both trigger on the same event).
        let mut g = MsfGuard::default()
            .with_suppress(1.0e-3, 3)
            .with_nudge(1.0e-3, 3, 0.5);
        let mut prev = 1.0e-4;
        let mut nudge_fires = 0;
        for _ in 0..20 {
            let next = prev * 2.0;
            if matches!(
                g.report(&make_report(&[next]), 8, 4),
                ConvergenceAction::NudgeDown { .. }
            ) {
                nudge_fires += 1;
            }
            prev = next;
        }
        assert!(nudge_fires >= 1, "expected NudgeDown to fire at least once");
    }

    #[test]
    fn msf_decaying_lambda_does_not_fire() {
        // Feed a halving D every event → λ_raw ≈ -ln(2)/4 < 0; neither trigger
        // should ever fire.
        let mut g = MsfGuard::default();
        let mut prev = 0.5;
        for _ in 0..30 {
            let next = prev * 0.5;
            let action = g.report(&make_report(&[next]), 8, 4);
            assert_eq!(action, ConvergenceAction::Stable);
            prev = next;
        }
    }

    #[test]
    fn msf_without_nudge_disables_hard_trigger() {
        let mut g = MsfGuard::default()
            .with_suppress(1.0e-3, 3)
            .without_nudge();
        let mut prev = 1.0e-4;
        let mut nudge_fires = 0;
        for _ in 0..20 {
            let next = prev * 2.0;
            if matches!(
                g.report(&make_report(&[next]), 8, 4),
                ConvergenceAction::NudgeDown { .. }
            ) {
                nudge_fires += 1;
            }
            prev = next;
        }
        assert_eq!(nudge_fires, 0);
    }

    #[test]
    fn msf_telemetry_carries_lambda_after_first_observation() {
        let mut g = MsfGuard::default();
        g.report(&make_report(&[0.01]), 8, 4);
        g.report(&make_report(&[0.02]), 8, 4);
        let t = g.telemetry();
        assert!(t.iter().any(|(k, _)| *k == "lambda_raw"));
        assert!(t.iter().any(|(k, _)| *k == "lambda_ema"));
    }

    #[test]
    fn msf_reset_clears_state() {
        let mut g = MsfGuard::default();
        g.report(&make_report(&[0.01]), 8, 4);
        g.report(&make_report(&[0.02]), 8, 4);
        g.reset();
        let s = g.report(&make_report(&[0.04]), 8, 4);
        assert_eq!(s, ConvergenceAction::Stable);
        // After reset the next event has no prev_d, so lambda_raw is None
        // and there's no telemetry to publish until a second observation.
        assert!(g.last_sample().unwrap().lambda_raw.is_none());
    }
}
