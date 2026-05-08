//! LR-aware meta-controller above ElChe.
//!
//! Watches the LR trajectory, anchor trend, and convergence guard verdicts
//! each averaging cycle. Reactively dispatches `nudge_anchor_down` on sharp
//! LR drops or sustained divergence patterns. ElChe's natural overhead
//! auto-tune handles recovery; the meta layer only adds reactive corrections
//! that the natural feedback can't absorb fast enough.
//!
//! # Two watchers, one actuator
//!
//! - **LR-cliff watcher** — single-cycle `|ΔLR / LR_prev|` exceeds threshold.
//!   Precautionary: anticipates the divergence that a sharp LR drop would
//!   cause, before it shows up in the signal.
//! - **Convergence watcher** — trailing K guard verdicts sustained
//!   `NudgeDown` or `SuppressGrowth`. Corrective: measures actual divergence
//!   already evident in the signal.
//!
//! Both watchers feed the same actuator: [`crate::distributed::ElChe::nudge_anchor_down`].
//! The effective nudge factor is dampened or amplified by `anchor_trend` so
//! the meta-controller doesn't pile on while ElChe is already correcting,
//! and pushes harder when ElChe's overhead auto-tune is fighting back.
//!
//! # Phases as hints, not gates
//!
//! The existing controller [`crate::distributed::Phase`] machine
//! (Probe / Warmup / Stable / Mature) is consulted to modulate per-phase
//! `base_factor` and the convergence watcher's sustain count `K`, but does
//! not gate firing. Probe is the only excluded phase (no calibration data,
//! all signals are noise).
//!
//! # Stages
//!
//! - **Stage 1 (this commit)**: plumbing only. [`LrEventMeta::observe`]
//!   records to windows; emits no actions. Verifies the wiring is in place
//!   without behavior change.
//! - **Stage 2**: detector logic (LR cliff, convergence pattern, anchor_trend
//!   dampening), unit-tested in isolation.
//! - **Stage 3**: activation — actions dispatched to coordinator's
//!   `nudge_anchor_down` path. Default off, opt-in via
//!   [`crate::distributed::ddp_run::DdpRunConfig::with_meta_controller`].

use std::collections::VecDeque;

use crate::distributed::Phase;
use crate::distributed::ddp_run::convergence::ConvergenceAction;

/// Static configuration for the meta-controller.
///
/// Sensible defaults from the 2026-05-09 design lock; override via builder
/// methods at the [`crate::distributed::ddp_run::DdpRunConfig`] level once
/// CLI plumbing lands in Stage 3.
#[derive(Debug, Clone, Copy)]
pub struct LrEventMetaConfig {
    /// Capacity of the LR trajectory window. Stage 1: not consulted; Stage 2
    /// uses the most recent two samples to compute the single-cycle delta.
    pub lr_window_cap: usize,
    /// Capacity of the anchor trajectory window. Stage 2 uses this for the
    /// `anchor_trend` (`(current − mean) / mean`) computation.
    pub anchor_window_cap: usize,
    /// Capacity of the convergence guard verdict window. Must be at least
    /// `max(sustain_k_for_phase)` so the longest-sustain phase has room.
    pub guard_window_cap: usize,
    /// Single-cycle `|ΔLR / LR_prev|` threshold for LR-cliff detection.
    /// 0.3 catches MultiStepLR's typical 5×/10× drops; cosine annealing's
    /// per-cycle delta stays well below.
    pub sharp_drop_threshold: f64,
    /// Lower clamp for the effective nudge factor after `anchor_trend`
    /// dampening / amplification.
    pub effective_factor_min: f64,
    /// Upper clamp for the effective nudge factor after `anchor_trend`
    /// dampening / amplification.
    pub effective_factor_max: f64,
}

impl Default for LrEventMetaConfig {
    fn default() -> Self {
        Self {
            lr_window_cap: 3,
            anchor_window_cap: 5,
            guard_window_cap: 5,
            sharp_drop_threshold: 0.3,
            effective_factor_min: 0.1,
            effective_factor_max: 0.95,
        }
    }
}

/// Recommended action emitted by [`LrEventMeta::observe`].
///
/// The `factor` field is the final value to pass to
/// [`crate::distributed::ElChe::nudge_anchor_down`]: it already incorporates
/// the per-phase base, `anchor_trend` dampening / amplification, and the
/// configured clamps. When both watchers fire in the same cycle the factor
/// is the multiplicative compound of the two effective factors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetaAction {
    /// No action this cycle.
    Noop,
    /// Apply `el_che.nudge_anchor_down(factor)` with the supplied factor.
    NudgeDown { factor: f64 },
}

/// LR-aware meta-controller above ElChe.
///
/// Records observations into windows; computes / emits actions in later
/// stages. Single-threaded ownership (the coordinator owns one instance and
/// drives it on its own thread); no locks.
pub struct LrEventMeta {
    config: LrEventMetaConfig,
    lr_window: VecDeque<f64>,
    anchor_window: VecDeque<usize>,
    guard_window: VecDeque<ConvergenceAction>,
}

impl LrEventMeta {
    /// Create a new meta-controller with the supplied configuration.
    pub fn new(config: LrEventMetaConfig) -> Self {
        let lr_window = VecDeque::with_capacity(config.lr_window_cap);
        let anchor_window = VecDeque::with_capacity(config.anchor_window_cap);
        let guard_window = VecDeque::with_capacity(config.guard_window_cap);
        Self { config, lr_window, anchor_window, guard_window }
    }

    /// Create with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(LrEventMetaConfig::default())
    }

    /// Record one averaging-cycle observation, run the two watchers, and
    /// return the combined recommendation.
    ///
    /// Returns [`MetaAction::Noop`] when neither watcher fires (or `phase` is
    /// [`Phase::Probe`] — no calibration data to act on). Returns
    /// [`MetaAction::NudgeDown`] with a factor that already incorporates the
    /// per-phase base, `anchor_trend` dampening / amplification, and the
    /// configured clamps. Same-cycle double-fire compounds multiplicatively.
    pub fn observe(
        &mut self,
        lr: f64,
        anchor: usize,
        verdict: ConvergenceAction,
        phase: Phase,
    ) -> MetaAction {
        push_capped(&mut self.lr_window, lr, self.config.lr_window_cap);
        push_capped(&mut self.anchor_window, anchor, self.config.anchor_window_cap);
        push_capped(&mut self.guard_window, verdict, self.config.guard_window_cap);

        // Probe: no calibration, all signals noise. Don't act.
        if phase < Phase::Warmup {
            return MetaAction::Noop;
        }

        let trend = self.anchor_trend();

        // LR-cliff watcher: precautionary, fires on single-cycle sharp drop.
        // Base factor is fixed at LR_CLIFF_BASE_FACTOR (matches the existing
        // 0.5 semantics of `nudge_anchor_down` for divergence correction).
        let lr_cliff_factor = if self.sharp_drop_detected() {
            Some(self.effective_factor(LR_CLIFF_BASE_FACTOR, trend))
        } else {
            None
        };

        // Convergence watcher: corrective, fires on trailing-K sustained
        // NudgeDown / SuppressGrowth verdicts. Phase-modulated K and base
        // factor (echoes the original 2026-05-01 emergency-factor pattern).
        let conv_factor = if self.convergence_pattern_fires(phase) {
            Some(self.effective_factor(base_factor_for(phase), trend))
        } else {
            None
        };

        match (lr_cliff_factor, conv_factor) {
            (None, None) => MetaAction::Noop,
            (Some(f), None) | (None, Some(f)) => MetaAction::NudgeDown { factor: f },
            (Some(f1), Some(f2)) => {
                let combined = (f1 * f2)
                    .clamp(self.config.effective_factor_min, self.config.effective_factor_max);
                MetaAction::NudgeDown { factor: combined }
            }
        }
    }

    /// Continuous anchor stability signal: `(current − mean) / mean` over
    /// the recent anchor window. Returns 0 when the window is too small or
    /// the mean is non-positive.
    ///
    /// - Negative: anchor is below recent mean (already nudged down, in
    ///   recovery). Dampens fresh nudges.
    /// - Positive: anchor is above recent mean (overhead auto-tune climbing).
    ///   Amplifies fresh nudges so they aren't immediately undone.
    /// - Zero: stable.
    pub fn anchor_trend(&self) -> f64 {
        if self.anchor_window.len() < 2 {
            return 0.0;
        }
        let n = self.anchor_window.len() as f64;
        let mean: f64 = self.anchor_window.iter().map(|&a| a as f64).sum::<f64>() / n;
        if mean <= 0.0 {
            return 0.0;
        }
        let current = *self.anchor_window.back().unwrap() as f64;
        (current - mean) / mean
    }

    /// Binary diagnostic: anchor has not moved much over the recent window.
    /// True iff `|anchor_trend| < SETTLED_EPSILON`. Not load-bearing in
    /// control logic; useful for telemetry and sweep analysis.
    pub fn is_settled(&self) -> bool {
        self.anchor_trend().abs() < SETTLED_EPSILON
    }

    /// LR-cliff detector. Fires on a single-cycle drop greater than
    /// `config.sharp_drop_threshold`. Rising LR (warmup) and gentle decay
    /// (cosine annealing) both correctly leave it silent.
    fn sharp_drop_detected(&self) -> bool {
        if self.lr_window.len() < 2 {
            return false;
        }
        let mut iter = self.lr_window.iter().rev();
        let lr_now = *iter.next().unwrap();
        let lr_prev = *iter.next().unwrap();
        if lr_prev <= 0.0 || lr_now >= lr_prev {
            return false;
        }
        let drop_ratio = (lr_prev - lr_now) / lr_prev;
        drop_ratio > self.config.sharp_drop_threshold
    }

    /// Convergence-pattern detector. Fires when the trailing K guard
    /// verdicts are all `NudgeDown` or `SuppressGrowth`, where K is
    /// phase-modulated. Single isolated noisy verdicts do not trip it.
    fn convergence_pattern_fires(&self, phase: Phase) -> bool {
        let k = sustain_k_for(phase);
        if k == 0 || self.guard_window.len() < k {
            return false;
        }
        self.guard_window
            .iter()
            .rev()
            .take(k)
            .all(|v| {
                matches!(
                    v,
                    ConvergenceAction::NudgeDown { .. } | ConvergenceAction::SuppressGrowth,
                )
            })
    }

    /// Compute the effective nudge factor: `base − (1 − base) · anchor_trend`,
    /// clamped to the configured `[min, max]` range. Symmetric around
    /// `trend = 0`: rising trend amplifies (smaller factor → larger
    /// reduction), falling trend dampens (factor closer to 1 → smaller
    /// reduction).
    fn effective_factor(&self, base: f64, trend: f64) -> f64 {
        let raw = base - (1.0 - base) * trend;
        raw.clamp(self.config.effective_factor_min, self.config.effective_factor_max)
    }

    /// Read-only access to the recorded LR window (oldest → newest).
    pub fn lr_window(&self) -> &VecDeque<f64> {
        &self.lr_window
    }

    /// Read-only access to the recorded anchor window (oldest → newest).
    pub fn anchor_window(&self) -> &VecDeque<usize> {
        &self.anchor_window
    }

    /// Read-only access to the recorded guard verdict window
    /// (oldest → newest).
    pub fn guard_window(&self) -> &VecDeque<ConvergenceAction> {
        &self.guard_window
    }

    /// Static configuration in use.
    pub fn config(&self) -> &LrEventMetaConfig {
        &self.config
    }
}

/// Base nudge factor for the LR-cliff watcher. Matches the long-standing
/// `nudge_anchor_down(0.5)` semantics used by the divergence-correction path
/// in the coordinator: halve the anchor on a hard signal.
const LR_CLIFF_BASE_FACTOR: f64 = 0.5;

/// Threshold below which `|anchor_trend|` qualifies as "settled" for the
/// binary diagnostic. Loose enough to ignore the dead-zone chatter from
/// `recompute_batch_counts`; tight enough to flag genuine drift.
const SETTLED_EPSILON: f64 = 0.10;

/// Per-phase base nudge factor for the convergence watcher.
///
/// Echoes the 2026-05-01 emergency-factor pattern (Warmup 2.0×, Stable
/// 1.7×, Mature 1.5×): aggressive when low-trust, gentle when high-trust.
/// Probe is excluded at the call site; the value here is a defensive
/// fallback that should never be reached in normal operation.
fn base_factor_for(phase: Phase) -> f64 {
    match phase {
        Phase::Probe | Phase::Warmup => 0.3,
        Phase::Stable => 0.5,
        Phase::Mature => 0.7,
    }
}

/// Per-phase sustain count `K` for the convergence watcher: the trailing K
/// verdicts must all be `NudgeDown` or `SuppressGrowth` for the pattern to
/// fire. Cautious phases require longer sustain (more evidence); confident
/// phases react sooner.
fn sustain_k_for(phase: Phase) -> usize {
    match phase {
        Phase::Probe | Phase::Warmup => 5,
        Phase::Stable => 3,
        Phase::Mature => 2,
    }
}

/// FIFO push with fixed capacity. Drops the oldest sample when full so the
/// most recent `cap` observations are retained.
fn push_capped<T>(buf: &mut VecDeque<T>, val: T, cap: usize) {
    if cap == 0 {
        return;
    }
    while buf.len() >= cap {
        buf.pop_front();
    }
    buf.push_back(val);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: feed a sequence of `(lr, anchor, verdict)` observations at
    /// `phase`, returning the final `MetaAction` from the last call.
    fn run_sequence(
        meta: &mut LrEventMeta,
        phase: Phase,
        seq: &[(f64, usize, ConvergenceAction)],
    ) -> MetaAction {
        let mut last = MetaAction::Noop;
        for &(lr, anchor, v) in seq {
            last = meta.observe(lr, anchor, v, phase);
        }
        last
    }

    // -----------------------------------------------------------------------
    // Storage / windowing (carries over from Stage 1)
    // -----------------------------------------------------------------------

    #[test]
    fn single_observation_emits_noop() {
        let mut meta = LrEventMeta::with_default_config();
        let action = meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        // Single sample: no LR delta available, no convergence pattern.
        assert_eq!(action, MetaAction::Noop);
    }

    #[test]
    fn observe_records_into_windows_with_cap() {
        let cfg = LrEventMetaConfig {
            lr_window_cap: 3,
            anchor_window_cap: 3,
            guard_window_cap: 3,
            ..LrEventMetaConfig::default()
        };
        let mut meta = LrEventMeta::new(cfg);
        for i in 0..5 {
            meta.observe(
                0.1 * (i as f64),
                100 + i,
                ConvergenceAction::Stable,
                Phase::Stable,
            );
        }
        assert_eq!(meta.lr_window().len(), 3);
        assert_eq!(meta.anchor_window().len(), 3);
        assert_eq!(meta.guard_window().len(), 3);
        assert!((meta.lr_window()[0] - 0.2).abs() < 1e-9);
        assert_eq!(*meta.anchor_window().back().unwrap(), 104);
    }

    #[test]
    fn push_capped_zero_capacity_is_noop() {
        let mut buf: VecDeque<i32> = VecDeque::new();
        push_capped(&mut buf, 1, 0);
        assert!(buf.is_empty());
    }

    // -----------------------------------------------------------------------
    // Phase gating
    // -----------------------------------------------------------------------

    #[test]
    fn probe_phase_silences_both_watchers() {
        let mut meta = LrEventMeta::with_default_config();
        // Build a state that would otherwise fire LR cliff: 0.1 → 0.01 (90% drop).
        meta.observe(0.1, 100, ConvergenceAction::NudgeDown { factor: 0.5 }, Phase::Probe);
        let action = meta.observe(
            0.01,
            100,
            ConvergenceAction::NudgeDown { factor: 0.5 },
            Phase::Probe,
        );
        assert_eq!(action, MetaAction::Noop, "Probe must never act");
    }

    // -----------------------------------------------------------------------
    // LR-cliff watcher
    // -----------------------------------------------------------------------

    #[test]
    fn lr_cliff_fires_on_multistep_drop() {
        let mut meta = LrEventMeta::with_default_config();
        // Sharp 10x drop in a single cycle (typical MultiStepLR).
        meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        let action = meta.observe(0.01, 100, ConvergenceAction::Stable, Phase::Stable);
        match action {
            MetaAction::NudgeDown { factor } => {
                // anchor_trend = 0 (constant 100 → mean 100 → trend 0),
                // base_factor = LR_CLIFF_BASE_FACTOR = 0.5, no dampening.
                assert!((factor - 0.5).abs() < 1e-9, "expected 0.5, got {factor}");
            }
            MetaAction::Noop => panic!("expected NudgeDown on 90% drop"),
        }
    }

    #[test]
    fn lr_cliff_silent_on_smooth_decay() {
        let mut meta = LrEventMeta::with_default_config();
        // Cosine-style smooth decay: tiny per-cycle deltas well below threshold.
        let mut lr = 0.1;
        let mut last = MetaAction::Noop;
        for _ in 0..10 {
            last = meta.observe(lr, 100, ConvergenceAction::Stable, Phase::Stable);
            lr *= 0.98; // 2% drop per cycle, well under 30% threshold
        }
        assert_eq!(last, MetaAction::Noop, "smooth decay must not fire LR cliff");
    }

    #[test]
    fn lr_cliff_silent_on_lr_rise() {
        let mut meta = LrEventMeta::with_default_config();
        // Warmup: rising LR. Even large rises should not fire (only drops do).
        meta.observe(0.001, 100, ConvergenceAction::Stable, Phase::Stable);
        let action = meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        assert_eq!(action, MetaAction::Noop, "rising LR must not fire LR cliff");
    }

    #[test]
    fn lr_cliff_silent_with_too_short_window() {
        let mut meta = LrEventMeta::with_default_config();
        // First observation: only one LR sample, no delta available.
        let action = meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        assert_eq!(action, MetaAction::Noop);
    }

    #[test]
    fn lr_cliff_threshold_at_boundary() {
        let cfg = LrEventMetaConfig {
            sharp_drop_threshold: 0.3,
            ..LrEventMetaConfig::default()
        };
        let mut meta = LrEventMeta::new(cfg);
        // Drop of 25%: under threshold, must not fire.
        meta.observe(1.0, 100, ConvergenceAction::Stable, Phase::Stable);
        let action = meta.observe(0.75, 100, ConvergenceAction::Stable, Phase::Stable);
        assert_eq!(action, MetaAction::Noop, "25% drop is under threshold");

        // Drop of 35%: over threshold, must fire.
        let mut meta = LrEventMeta::new(cfg);
        meta.observe(1.0, 100, ConvergenceAction::Stable, Phase::Stable);
        let action = meta.observe(0.65, 100, ConvergenceAction::Stable, Phase::Stable);
        assert!(matches!(action, MetaAction::NudgeDown { .. }), "35% drop over threshold");
    }

    // -----------------------------------------------------------------------
    // Convergence-pattern watcher
    // -----------------------------------------------------------------------

    #[test]
    fn convergence_fires_after_k_consecutive_at_stable() {
        let mut meta = LrEventMeta::with_default_config();
        // Stable phase: K=3 sustained verdicts required.
        let nudge = ConvergenceAction::NudgeDown { factor: 0.5 };
        run_sequence(
            &mut meta,
            Phase::Stable,
            &[(0.1, 100, nudge), (0.1, 100, nudge)],
        );
        let action = meta.observe(0.1, 100, nudge, Phase::Stable);
        match action {
            MetaAction::NudgeDown { factor } => {
                // anchor_trend = 0 (constant), base_factor at Stable = 0.5.
                assert!((factor - 0.5).abs() < 1e-9, "expected 0.5, got {factor}");
            }
            MetaAction::Noop => panic!("3 consecutive NudgeDown at Stable should fire"),
        }
    }

    #[test]
    fn convergence_silent_on_isolated_nudge() {
        let mut meta = LrEventMeta::with_default_config();
        let nudge = ConvergenceAction::NudgeDown { factor: 0.5 };
        run_sequence(
            &mut meta,
            Phase::Stable,
            &[
                (0.1, 100, ConvergenceAction::Stable),
                (0.1, 100, nudge),
                (0.1, 100, ConvergenceAction::Stable),
            ],
        );
        let action = meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        assert_eq!(action, MetaAction::Noop, "isolated NudgeDown must not fire");
    }

    #[test]
    fn convergence_silent_on_two_at_stable_needs_three() {
        let mut meta = LrEventMeta::with_default_config();
        let nudge = ConvergenceAction::NudgeDown { factor: 0.5 };
        meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        meta.observe(0.1, 100, nudge, Phase::Stable);
        let action = meta.observe(0.1, 100, nudge, Phase::Stable);
        assert_eq!(action, MetaAction::Noop, "Stable phase needs K=3, only 2 sustained");
    }

    #[test]
    fn convergence_at_mature_needs_only_two() {
        let mut meta = LrEventMeta::with_default_config();
        let sg = ConvergenceAction::SuppressGrowth;
        meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Mature);
        meta.observe(0.1, 100, sg, Phase::Mature);
        let action = meta.observe(0.1, 100, sg, Phase::Mature);
        match action {
            MetaAction::NudgeDown { factor } => {
                // Mature base_factor = 0.7, trend = 0.
                assert!((factor - 0.7).abs() < 1e-9);
            }
            MetaAction::Noop => panic!("Mature K=2 with 2 sustained must fire"),
        }
    }

    #[test]
    fn convergence_mixed_sustained_kinds_fire() {
        let mut meta = LrEventMeta::with_default_config();
        // Mix of NudgeDown and SuppressGrowth — both qualify.
        let nudge = ConvergenceAction::NudgeDown { factor: 0.5 };
        let sg = ConvergenceAction::SuppressGrowth;
        meta.observe(0.1, 100, sg, Phase::Stable);
        meta.observe(0.1, 100, nudge, Phase::Stable);
        let action = meta.observe(0.1, 100, sg, Phase::Stable);
        assert!(matches!(action, MetaAction::NudgeDown { .. }));
    }

    // -----------------------------------------------------------------------
    // anchor_trend
    // -----------------------------------------------------------------------

    #[test]
    fn anchor_trend_rising() {
        let mut meta = LrEventMeta::with_default_config();
        for &a in &[100, 110, 120, 130, 140] {
            meta.observe(0.1, a, ConvergenceAction::Stable, Phase::Stable);
        }
        let trend = meta.anchor_trend();
        assert!(trend > 0.0, "rising anchor must have positive trend, got {trend}");
    }

    #[test]
    fn anchor_trend_falling() {
        let mut meta = LrEventMeta::with_default_config();
        for &a in &[100, 80, 60, 40, 20] {
            meta.observe(0.1, a, ConvergenceAction::Stable, Phase::Stable);
        }
        let trend = meta.anchor_trend();
        assert!(trend < 0.0, "falling anchor must have negative trend, got {trend}");
    }

    #[test]
    fn anchor_trend_stable_near_zero() {
        let mut meta = LrEventMeta::with_default_config();
        for _ in 0..5 {
            meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        }
        let trend = meta.anchor_trend();
        assert!(trend.abs() < 1e-9, "constant anchor trend must be 0, got {trend}");
    }

    #[test]
    fn anchor_trend_short_window_is_zero() {
        let mut meta = LrEventMeta::with_default_config();
        meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        // Only 1 sample → trend should be 0.
        assert_eq!(meta.anchor_trend(), 0.0);
    }

    #[test]
    fn is_settled_within_epsilon() {
        let mut meta = LrEventMeta::with_default_config();
        for _ in 0..5 {
            meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        }
        assert!(meta.is_settled());

        let mut meta = LrEventMeta::with_default_config();
        for &a in &[100, 50, 150, 50, 150] {
            meta.observe(0.1, a, ConvergenceAction::Stable, Phase::Stable);
        }
        // Highly variable anchor should not be settled.
        assert!(!meta.is_settled() || meta.anchor_trend().abs() < SETTLED_EPSILON);
    }

    // -----------------------------------------------------------------------
    // effective_factor symmetry / clamps
    // -----------------------------------------------------------------------

    #[test]
    fn effective_factor_symmetric_around_zero_trend() {
        let meta = LrEventMeta::with_default_config();
        let base = 0.5;
        // Trend 0 → base.
        assert!((meta.effective_factor(base, 0.0) - 0.5).abs() < 1e-9);
        // Trend −0.5 → 0.75 (dampened, less aggressive).
        assert!((meta.effective_factor(base, -0.5) - 0.75).abs() < 1e-9);
        // Trend +0.5 → 0.25 (amplified, more aggressive).
        assert!((meta.effective_factor(base, 0.5) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn effective_factor_clamps_at_extremes() {
        let cfg = LrEventMetaConfig {
            effective_factor_min: 0.1,
            effective_factor_max: 0.95,
            ..LrEventMetaConfig::default()
        };
        let meta = LrEventMeta::new(cfg);
        // Extreme positive trend → clamped at 0.1 (below would go negative).
        let f = meta.effective_factor(0.5, 5.0);
        assert!((f - 0.1).abs() < 1e-9, "expected clamp to 0.1, got {f}");
        // Extreme negative trend → clamped at 0.95.
        let f = meta.effective_factor(0.5, -5.0);
        assert!((f - 0.95).abs() < 1e-9, "expected clamp to 0.95, got {f}");
    }

    // -----------------------------------------------------------------------
    // Compound (both watchers fire same cycle)
    // -----------------------------------------------------------------------

    #[test]
    fn both_watchers_compound_factor_at_mature() {
        let mut meta = LrEventMeta::with_default_config();
        // Mature: K=2 sustained verdicts. Build up sustain pattern with
        // constant LR + anchor, then fire LR cliff on the third call.
        let nudge = ConvergenceAction::NudgeDown { factor: 0.5 };
        meta.observe(0.1, 100, nudge, Phase::Mature);
        meta.observe(0.1, 100, nudge, Phase::Mature);
        // Now: convergence pattern has 2 NudgeDown in trailing window (K=2 fires).
        // LR drops 90% on this call → LR cliff also fires.
        let action = meta.observe(0.01, 100, nudge, Phase::Mature);
        match action {
            MetaAction::NudgeDown { factor } => {
                // LR cliff: base 0.5, trend 0 → 0.5
                // Convergence at Mature: base 0.7, trend 0 → 0.7
                // Compound: 0.5 * 0.7 = 0.35
                assert!((factor - 0.35).abs() < 1e-9, "expected compound 0.35, got {factor}");
            }
            MetaAction::Noop => panic!("both watchers should fire and compound"),
        }
    }

    // -----------------------------------------------------------------------
    // Per-phase factor / sustain modulation
    // -----------------------------------------------------------------------

    #[test]
    fn per_phase_base_factors_match_design() {
        assert!((base_factor_for(Phase::Warmup) - 0.3).abs() < 1e-9);
        assert!((base_factor_for(Phase::Stable) - 0.5).abs() < 1e-9);
        assert!((base_factor_for(Phase::Mature) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn per_phase_sustain_k_match_design() {
        assert_eq!(sustain_k_for(Phase::Warmup), 5);
        assert_eq!(sustain_k_for(Phase::Stable), 3);
        assert_eq!(sustain_k_for(Phase::Mature), 2);
    }

    // -----------------------------------------------------------------------
    // Trend dampening / amplification (sequential cliff → recovery scenario)
    // -----------------------------------------------------------------------

    #[test]
    fn trend_dampens_subsequent_nudge_after_cliff() {
        let mut meta = LrEventMeta::with_default_config();
        // Build anchor history at 100, then anchor drops to 50 (e.g. after a
        // prior nudge). LR also drops sharply on this cycle: cliff fires, but
        // anchor_trend is now strongly negative → dampened factor.
        for _ in 0..4 {
            meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
        }
        // Now anchor at 50: mean over window (100,100,100,100,50) = 90; trend
        // = (50-90)/90 ≈ -0.444. LR drops 90%.
        let action = meta.observe(0.01, 50, ConvergenceAction::Stable, Phase::Stable);
        match action {
            MetaAction::NudgeDown { factor } => {
                // base = 0.5, trend ≈ -0.444 → effective = 0.5 - 0.5 * (-0.444) ≈ 0.722.
                // Falling-trend regime: factor closer to 1 = less aggressive.
                assert!(factor > 0.5, "falling trend must dampen factor, got {factor}");
                assert!(factor < 0.95, "factor should not be fully clamped, got {factor}");
            }
            MetaAction::Noop => panic!("LR cliff still fires under dampening"),
        }
    }

    #[test]
    fn trend_amplifies_nudge_during_relax_up_climb() {
        let mut meta = LrEventMeta::with_default_config();
        // Anchor climbing (overhead auto-tune relaxing up).
        for &a in &[100, 110, 120, 130, 140] {
            meta.observe(0.1, a, ConvergenceAction::Stable, Phase::Stable);
        }
        // Now LR drops sharply. anchor_trend > 0 → effective_factor < base.
        let action = meta.observe(0.01, 140, ConvergenceAction::Stable, Phase::Stable);
        match action {
            MetaAction::NudgeDown { factor } => {
                assert!(factor < 0.5, "rising trend must amplify nudge, got {factor}");
                assert!(factor > 0.1, "factor should not be fully clamped, got {factor}");
            }
            MetaAction::Noop => panic!("LR cliff fires regardless of trend"),
        }
    }
}
