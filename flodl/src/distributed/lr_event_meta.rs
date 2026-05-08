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
/// Stage 1 always emits [`MetaAction::Noop`]. Stages 2–3 will emit
/// [`MetaAction::NudgeDown`] when the watchers fire; the factor field
/// already accounts for the per-phase base, the `anchor_trend` dampening,
/// and the configured clamps.
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

    /// Record one averaging-cycle observation. Stage 1: pure storage, always
    /// returns [`MetaAction::Noop`]. The `phase` argument is recorded for
    /// future stages; today only Probe→non-Probe matters for any logic.
    pub fn observe(
        &mut self,
        lr: f64,
        anchor: usize,
        verdict: ConvergenceAction,
        _phase: Phase,
    ) -> MetaAction {
        push_capped(&mut self.lr_window, lr, self.config.lr_window_cap);
        push_capped(&mut self.anchor_window, anchor, self.config.anchor_window_cap);
        push_capped(&mut self.guard_window, verdict, self.config.guard_window_cap);
        MetaAction::Noop
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

    #[test]
    fn observe_is_inert_in_stage_1() {
        let mut meta = LrEventMeta::with_default_config();
        let action = meta.observe(0.1, 100, ConvergenceAction::Stable, Phase::Stable);
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
        // Front of lr_window should be 0.2 (oldest after capping 5 inserts to 3).
        assert!((meta.lr_window()[0] - 0.2).abs() < 1e-9);
        // Back of anchor_window is the most recent insert (104).
        assert_eq!(*meta.anchor_window().back().unwrap(), 104);
    }

    #[test]
    fn push_capped_zero_capacity_is_noop() {
        let mut buf: VecDeque<i32> = VecDeque::new();
        push_capped(&mut buf, 1, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn varied_verdicts_record_in_order() {
        let mut meta = LrEventMeta::with_default_config();
        meta.observe(0.1, 10, ConvergenceAction::Stable, Phase::Probe);
        meta.observe(0.1, 10, ConvergenceAction::SuppressGrowth, Phase::Warmup);
        meta.observe(0.1, 10, ConvergenceAction::NudgeDown { factor: 0.5 }, Phase::Stable);
        let verdicts: Vec<_> = meta.guard_window().iter().copied().collect();
        assert!(matches!(verdicts[0], ConvergenceAction::Stable));
        assert!(matches!(verdicts[1], ConvergenceAction::SuppressGrowth));
        assert!(matches!(
            verdicts[2],
            ConvergenceAction::NudgeDown { factor } if (factor - 0.5).abs() < 1e-9
        ));
    }
}
