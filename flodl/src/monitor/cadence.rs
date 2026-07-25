//! Report cadence — *when* the monitor emits a per-window record
//! (`.design/monitoring-portal-b3.md`, "Reporting cadence").
//!
//! Metrics historically emit once per **epoch**, which is useless for
//! single-epoch LLM training (one report for the whole run). The tick instead
//! rides the **sync/reduce-window boundary**, throttled to a target
//! `reports_per_epoch` (x): a report fires at a sync boundary each time
//! cumulative in-epoch realized work crosses the next `k · (epoch_work / x)`
//! fraction.
//!
//! This is a *read-only observer of the existing step clock* — it never
//! introduces a second clock (coverage and sync must stay on one clock). The
//! caller advances it only at sync boundaries with the cumulative in-epoch work
//! it already tracks (`global_step` since epoch start, samples-proportional
//! under a uniform batch size), so a report never lands mid-window.
//!
//! Behavior falls out cleanly: early small windows accumulate across several
//! syncs before a crossing (a report spans several syncs), settling toward one
//! report per window as windows grow; a window large enough to cross several
//! thresholds at once still fires **once** (there is only one window of data),
//! advancing past the skipped thresholds. Single-epoch training degenerates to
//! "x reports over the whole run" with no special-casing.
//!
//! PR1b scope: this pure scheduler + [`report_interval`] helper, unit-tested.
//! Wiring it into `finish_averaging_head` and emitting the per-window record
//! tree is the next slice (needs the hot-path accumulation of the per-batch
//! loss that the coordinator currently drops).

/// Steps between reports for `reports_per_epoch` reports per epoch, given the
/// dataset — a **pre-training** estimate (data is known ahead), so the report
/// interval is knowable before launch. Returns steps per report
/// (`epoch_work / x`), or `f64::INFINITY` when reporting is disabled or the
/// inputs are degenerate.
pub fn report_interval(total_samples: usize, batch_size: usize, reports_per_epoch: usize) -> f64 {
    if batch_size == 0 || reports_per_epoch == 0 {
        return f64::INFINITY;
    }
    let steps_per_epoch = (total_samples / batch_size) as f64;
    steps_per_epoch / reports_per_epoch as f64
}

/// Decides, at each sync boundary, whether to emit a monitor report — firing at
/// most `reports_per_epoch` times per epoch on `epoch_work / x` work-fraction
/// crossings. Work is any linear unit consistent across calls (steps or
/// samples; the decision is scale-invariant).
#[derive(Debug, Clone)]
pub struct ReportScheduler {
    /// Target reports per epoch (`x`). `0` disables reporting.
    reports_per_epoch: usize,
    /// Linear work per epoch (steps or samples per epoch); must be `> 0` to fire.
    epoch_work: f64,
    /// Reports already fired in the current epoch.
    fired: usize,
}

impl ReportScheduler {
    /// A scheduler targeting `reports_per_epoch` (x) reports across one epoch's
    /// worth of `epoch_work` (steps- or samples-per-epoch).
    pub fn new(reports_per_epoch: usize, epoch_work: f64) -> Self {
        Self {
            reports_per_epoch,
            epoch_work,
            fired: 0,
        }
    }

    /// Work between consecutive report thresholds (`epoch_work / x`).
    fn interval(&self) -> f64 {
        self.epoch_work / self.reports_per_epoch as f64
    }

    /// Call **at a sync boundary** with `in_epoch_work` = cumulative linear work
    /// since the current epoch started. Returns `true` iff a new report
    /// threshold was crossed (emit a report now). A single call advances past
    /// every threshold it crossed but returns `true` once — one window yields at
    /// most one report.
    pub fn on_sync(&mut self, in_epoch_work: f64) -> bool {
        if self.reports_per_epoch == 0 || in_epoch_work <= 0.0 {
            return false;
        }
        // A positive, finite threshold interval is required to fire; this also
        // rejects a zero / NaN `epoch_work` without a negated comparison.
        let interval = self.interval();
        if !interval.is_finite() || interval <= 0.0 {
            return false;
        }
        // How many thresholds the cumulative work has reached, capped at x
        // (in-epoch work can slightly exceed epoch_work under overshoot).
        let reached = (in_epoch_work / interval).floor() as usize;
        let due = reached.min(self.reports_per_epoch);
        if due > self.fired {
            self.fired = due;
            true
        } else {
            false
        }
    }

    /// Reset for a new epoch. Call at an epoch boundary; the caller separately
    /// emits the epoch-boundary report (carrying the `epoch_complete` marker).
    pub fn reset_epoch(&mut self) {
        self.fired = 0;
    }

    /// Reports fired so far in the current epoch (diagnostic).
    pub fn fired_this_epoch(&self) -> usize {
        self.fired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_helper_matches_epoch_work_over_x() {
        // 1000 samples / batch 10 = 100 steps/epoch; 4 reports => every 25 steps.
        assert_eq!(report_interval(1000, 10, 4), 25.0);
        // Disabled / degenerate inputs.
        assert_eq!(report_interval(1000, 10, 0), f64::INFINITY);
        assert_eq!(report_interval(1000, 0, 4), f64::INFINITY);
    }

    #[test]
    fn fires_on_each_threshold_crossing() {
        // epoch_work 100, x 4 => thresholds at 25, 50, 75, 100.
        let mut s = ReportScheduler::new(4, 100.0);
        assert!(!s.on_sync(10.0)); // below 25
        assert!(!s.on_sync(24.0));
        assert!(s.on_sync(25.0)); // hit 25 -> report 1
        assert!(!s.on_sync(40.0)); // between 25 and 50
        assert!(s.on_sync(55.0)); // crossed 50 -> report 2
        assert!(s.on_sync(80.0)); // crossed 75 -> report 3
        assert!(s.on_sync(100.0)); // crossed 100 -> report 4
        assert!(!s.on_sync(120.0)); // capped at x=4, no more this epoch
        assert_eq!(s.fired_this_epoch(), 4);
    }

    #[test]
    fn report_only_at_sync_boundaries() {
        // The scheduler is only consulted at sync boundaries, so a report can
        // only land at one. Small 3-step windows: the report for the 25-step
        // threshold fires at the FIRST sync at/after 25 (step 27 here), never
        // mid-window.
        let mut s = ReportScheduler::new(4, 100.0);
        let mut fired_at = Vec::new();
        let mut work = 0.0;
        for _ in 0..9 {
            work += 3.0; // 3, 6, ... 27
            if s.on_sync(work) {
                fired_at.push(work);
            }
        }
        assert_eq!(fired_at, vec![27.0]); // snapped to the sync that crossed 25
    }

    #[test]
    fn big_window_crossing_many_thresholds_fires_once() {
        // A window so large it jumps past several thresholds fires ONE report
        // (one window = one datum), advancing past the skipped thresholds.
        let mut s = ReportScheduler::new(10, 100.0); // thresholds every 10
        assert!(s.on_sync(35.0)); // crossed 10,20,30 at once -> one report
        assert_eq!(s.fired_this_epoch(), 3); // advanced past the skipped ones
        assert!(!s.on_sync(38.0)); // still within [30,40)
        assert!(s.on_sync(42.0)); // crossed 40 -> next report
    }

    #[test]
    fn epoch_reset_restarts_thresholds() {
        let mut s = ReportScheduler::new(2, 100.0); // thresholds at 50, 100
        assert!(s.on_sync(60.0));
        assert!(s.on_sync(100.0));
        assert_eq!(s.fired_this_epoch(), 2);
        s.reset_epoch();
        assert_eq!(s.fired_this_epoch(), 0);
        assert!(s.on_sync(50.0)); // fresh epoch fires again
    }

    #[test]
    fn single_epoch_degenerate_spreads_x_over_the_run() {
        // Single-epoch LLM: epoch_work = whole-run steps; x reports over the run.
        // No epoch reset ever happens; the scheduler still fires exactly x times.
        let mut s = ReportScheduler::new(5, 10_000.0); // every 2000 steps
        let mut fires = 0;
        for step in (0..=10_000).step_by(500) {
            if s.on_sync(step as f64) {
                fires += 1;
            }
        }
        assert_eq!(fires, 5);
    }

    #[test]
    fn step_to_sample_proxy_is_scale_invariant() {
        // Uniform batch size => steps proportional to samples. Thresholding
        // steps and thresholding samples (×batch_size) yield identical fire
        // decisions, so the step proxy is exact.
        let bs = 32.0;
        let mut in_steps = ReportScheduler::new(4, 100.0);
        let mut in_samples = ReportScheduler::new(4, 100.0 * bs);
        for step in (0..=100).step_by(7) {
            let a = in_steps.on_sync(step as f64);
            let b = in_samples.on_sync(step as f64 * bs);
            assert_eq!(a, b, "at step {step}");
        }
        assert_eq!(in_steps.fired_this_epoch(), in_samples.fired_this_epoch());
    }

    #[test]
    fn disabled_and_degenerate_never_fire() {
        assert!(!ReportScheduler::new(0, 100.0).on_sync(50.0)); // x = 0
        assert!(!ReportScheduler::new(4, 0.0).on_sync(50.0)); // epoch_work 0
        assert!(!ReportScheduler::new(4, 100.0).on_sync(0.0)); // no work yet
    }
}
