//! Per-rank, per-window bookkeeping: the coordinator's view of what
//! each rank did since the last reduce.
//!
//! One [`WindowLedger`] instance lives on the coordinator and absorbs
//! what used to be five loose per-rank vectors. It records the raw
//! events (a batch delivered, a callback's wall-time to exclude) and
//! owns the per-rank derivations built on them (marginal delivered
//! rate, first-batch fill excess, per-batch wall). Cohort *policy* —
//! when the window fires, which feed ElChe schedules on — stays with
//! the coordinator and reads through this ledger.
//!
//! # Authority
//!
//! The ledger is **advisory scheduling state**, one of the two
//! authorities the realized-work design keeps separate (see
//! [`crate::distributed::realized_work`]): it drives *when* windows
//! fire and *how* work is allocated, on the single step clock. The
//! ground-truth divisor of the reduce is NOT read from here — it is
//! the mass computed rank-side at snapshot time and carried with each
//! contribution. A rank the coordinator believes did `n` steps may
//! realize a different count at its snapshot; the reduce is exact
//! either way.
//!
//! # The delivered / fill split
//!
//! [`WindowLedger::record_batch`] implements the report-at-sync
//! delivered feed's marginal rule: the window's FIRST batch carries the
//! per-window fill (control transit, plan pickup, prefetch spin-up,
//! first-batch unpipelined H2D) and is captured separately
//! ([`WindowLedger::fill_excess_ms`] derives the excess); batches
//! `2..n` accumulate into the marginal delivered rate so the fixed fill
//! never pollutes the per-batch cost ElChe schedules on.

/// Per-rank accumulators for the current reduce window. See the module
/// docs for the authority note and the delivered / fill split.
#[derive(Debug)]
pub(crate) struct WindowLedger {
    /// Steps (batches) completed since the last averaging cycle.
    steps: Vec<usize>,
    /// Wall-clock ms accumulated since the last averaging cycle. Sum of
    /// per-batch `Batch.batch_ms` (= `train_step` time) — COMPUTE ONLY.
    /// The feed for the Sync policy and the per-batch UID-generator
    /// tiebreak; superseded by the delivered accumulators for the
    /// progressive (Cadence / Async) feed.
    wall_ms: Vec<f64>,
    /// Rank-reported DELIVERED wall (`batch_ms + data_ms`) accumulated
    /// continuously from each `Batch`, marginal (first batch skipped),
    /// with its matched batch count — the progressive ElChe timing
    /// feed. Present at sync by construction (no completion-frame
    /// race).
    delivered_ms: Vec<f64>,
    delivered_batches: Vec<usize>,
    /// Delivered cost of the window's FIRST batch — the one the
    /// marginal accumulators deliberately skip. Its excess over the
    /// marginal rate is the per-window fill.
    first_batch_ms: Vec<f64>,
    /// Sum of the per-batch training loss reported this window, with its
    /// own matched count — the sub-epoch loss feed for the monitor
    /// record stream. Deliberately NOT divided by `steps`: step counts
    /// reset backend-specifically ([`WindowLedger::reset_steps`]) while
    /// this pair resets with the timing accumulators, so a shared
    /// denominator could divide by an already-reset count.
    loss_sum: Vec<f64>,
    loss_count: Vec<usize>,
}

impl WindowLedger {
    pub(crate) fn new(world_size: usize) -> Self {
        WindowLedger {
            steps: vec![0; world_size],
            wall_ms: vec![0.0; world_size],
            delivered_ms: vec![0.0; world_size],
            delivered_batches: vec![0; world_size],
            first_batch_ms: vec![0.0; world_size],
            loss_sum: vec![0.0; world_size],
            loss_count: vec![0; world_size],
        }
    }

    // --- write path -----------------------------------------------------

    /// Record one delivered batch for `rank`: bump the step count and
    /// compute wall, and route the delivered cost (`batch_ms + data_ms`)
    /// per the marginal rule — the window's first batch into the fill
    /// slot, batches `2..n` into the marginal delivered accumulators.
    /// Ignores an out-of-range rank (malformed frame; callers validate).
    pub(crate) fn record_batch(&mut self, rank: usize, batch_ms: f64, data_ms: f64) {
        if rank >= self.steps.len() {
            return;
        }
        self.steps[rank] = self.steps[rank].saturating_add(1);
        self.wall_ms[rank] += batch_ms;
        if self.steps[rank] > 1 {
            self.delivered_ms[rank] += batch_ms + data_ms;
            self.delivered_batches[rank] += 1;
        } else {
            self.first_batch_ms[rank] = batch_ms + data_ms;
        }
    }

    /// Record `rank`'s per-batch training loss for the sub-epoch monitor
    /// feed. Split from [`WindowLedger::record_batch`] because it is a
    /// monitoring accumulator with its own reset clock (see the field
    /// docs), not part of the delivered / fill scheduling machinery.
    /// Ignores an out-of-range rank and a non-finite loss (a diverged
    /// batch must not poison the window mean).
    pub(crate) fn record_batch_loss(&mut self, rank: usize, loss: f64) {
        if rank >= self.loss_sum.len() || !loss.is_finite() {
            return;
        }
        self.loss_sum[rank] += loss;
        self.loss_count[rank] += 1;
    }

    /// Mean training loss `rank` reported this window, or `None` when it
    /// reported no batch. `None` is **absent, not zero** — the monitor
    /// record stream omits an unmeasured metric rather than averaging in
    /// a false zero.
    pub(crate) fn mean_loss(&self, rank: usize) -> Option<f64> {
        let n = *self.loss_count.get(rank)?;
        if n == 0 {
            return None;
        }
        Some(self.loss_sum[rank] / n as f64)
    }

    /// Exclude a callback's wall-time (checkpoint / eval / epoch_fn)
    /// from `rank`'s compute AND delivered accumulators, so ElChe's
    /// rebalancer does not read callback cost as compute slowness in
    /// either balancer denominator. Clamped at 0 to absorb EWMA noise
    /// and fp drift. No-op on an out-of-range rank.
    pub(crate) fn absorb_callback_cost(&mut self, rank: usize, elapsed_ms: f64) {
        if let Some(w) = self.wall_ms.get_mut(rank) {
            *w = (*w - elapsed_ms).max(0.0);
        }
        if let Some(d) = self.delivered_ms.get_mut(rank) {
            *d = (*d - elapsed_ms).max(0.0);
        }
    }

    // --- steps ----------------------------------------------------------

    /// Steps `rank` completed this window. Panics on an out-of-range
    /// rank, like the direct indexing it replaces.
    pub(crate) fn steps(&self, rank: usize) -> usize {
        self.steps[rank]
    }

    /// Per-rank step counts for the whole cohort.
    pub(crate) fn steps_all(&self) -> &[usize] {
        &self.steps
    }

    /// Smallest per-rank step count (0 for an empty cohort).
    pub(crate) fn min_steps(&self) -> usize {
        self.steps.iter().copied().min().unwrap_or(0)
    }

    /// Largest per-rank step count (0 for an empty cohort).
    pub(crate) fn max_steps(&self) -> usize {
        self.steps.iter().copied().max().unwrap_or(0)
    }

    /// Total steps across the cohort this window.
    pub(crate) fn total_steps(&self) -> usize {
        self.steps.iter().sum()
    }

    // --- compute wall ---------------------------------------------------

    /// Compute-only wall ms `rank` accumulated this window.
    pub(crate) fn wall_ms(&self, rank: usize) -> f64 {
        self.wall_ms[rank]
    }

    /// Per-rank compute wall for the whole cohort.
    pub(crate) fn wall_ms_all(&self) -> &[f64] {
        &self.wall_ms
    }

    /// Average per-batch compute wall (ms) for `rank` this window.
    /// `f64::INFINITY` when the rank has no batches yet (cold start),
    /// so it sorts LAST in "fastest" pickers — un-calibrated ranks
    /// shouldn't win capacity elections. Defensive on out-of-range
    /// ranks (treated as no-batches).
    pub(crate) fn per_batch_wall_ms(&self, rank: usize) -> f64 {
        let steps = self.steps.get(rank).copied().unwrap_or(0);
        if steps == 0 {
            return f64::INFINITY;
        }
        let wall = self.wall_ms.get(rank).copied().unwrap_or(0.0);
        wall / steps as f64
    }

    // --- delivered feed -------------------------------------------------

    /// Whether `rank` has a usable delivered sample this window
    /// (nonzero marginal batches AND ms). A single-batch window leaves
    /// the marginal accumulators empty (its only batch was the fill
    /// batch), so this is false there by design.
    pub(crate) fn has_delivered_sample(&self, rank: usize) -> bool {
        self.delivered_batches[rank] > 0 && self.delivered_ms[rank] > 0.0
    }

    /// Marginal delivered ms `rank` accumulated this window.
    pub(crate) fn delivered_ms(&self, rank: usize) -> f64 {
        self.delivered_ms[rank]
    }

    /// Per-rank marginal delivered ms for the whole cohort.
    pub(crate) fn delivered_ms_all(&self) -> &[f64] {
        &self.delivered_ms
    }

    /// Marginal delivered batch count for `rank` this window.
    pub(crate) fn delivered_batches(&self, rank: usize) -> usize {
        self.delivered_batches[rank]
    }

    /// Per-rank marginal delivered batch counts for the whole cohort.
    pub(crate) fn delivered_batches_all(&self) -> &[usize] {
        &self.delivered_batches
    }

    /// Per-window FILL for `rank`: the excess of the window's first
    /// batch over the steady-state marginal rate — the amortizable
    /// per-window cost (control transit, plan pickup, prefetch
    /// spin-up) the marginal feed excludes. `0.0` when the window has
    /// no marginal sample (cold start / single-batch window).
    pub(crate) fn fill_excess_ms(&self, rank: usize) -> f64 {
        if !self.has_delivered_sample(rank) {
            return 0.0;
        }
        let marginal = self.delivered_ms[rank] / self.delivered_batches[rank] as f64;
        (self.first_batch_ms[rank] - marginal).max(0.0)
    }

    // --- window resets --------------------------------------------------

    /// Reset the timing accumulators (compute wall, delivered, first-
    /// batch fill) for a new window. Step counts are NOT touched —
    /// their reset placement is backend-specific (see
    /// [`WindowLedger::reset_steps`]).
    pub(crate) fn reset_timing(&mut self) {
        self.wall_ms.fill(0.0);
        self.delivered_ms.fill(0.0);
        self.delivered_batches.fill(0);
        self.first_batch_ms.fill(0.0);
        self.loss_sum.fill(0.0);
        self.loss_count.fill(0);
    }

    /// Reset the per-rank step counts for a new window. Split from
    /// [`WindowLedger::reset_timing`] because the CPU path must reset
    /// steps BEFORE the atomic-dispatch fold (chunk sizing reads them)
    /// while the timing reset rides the shared finish tail.
    pub(crate) fn reset_steps(&mut self) {
        self.steps.fill(0);
    }

    // --- test support ---------------------------------------------------

    /// Test-only mutator for the step count. Mirrors the direct field
    /// writes the pre-ledger test helpers used.
    #[cfg(test)]
    pub(crate) fn set_steps_for_test(&mut self, rank: usize, n: usize) {
        self.steps[rank] = n;
    }

    /// Test-only mutator for the compute wall accumulator.
    #[cfg(test)]
    pub(crate) fn set_wall_ms_for_test(&mut self, rank: usize, ms: f64) {
        self.wall_ms[rank] = ms;
    }

    /// Test-only mutator for the marginal delivered credit pair. Lets
    /// predicate tests seed decoupled step/delivered combinations that
    /// [`WindowLedger::record_batch`] (which couples them) cannot
    /// produce.
    #[cfg(test)]
    pub(crate) fn set_delivered_for_test(&mut self, rank: usize, ms: f64, batches: usize) {
        self.delivered_ms[rank] = ms;
        self.delivered_batches[rank] = batches;
    }
}

#[cfg(test)]
mod tests {
    use super::WindowLedger;

    #[test]
    fn first_batch_fills_then_marginal_accumulates() {
        let mut l = WindowLedger::new(2);
        // First batch: fill slot, no marginal sample.
        l.record_batch(0, 10.0, 5.0);
        assert_eq!(l.steps(0), 1);
        assert_eq!(l.wall_ms(0), 10.0);
        assert!(!l.has_delivered_sample(0));
        assert_eq!(l.fill_excess_ms(0), 0.0); // no marginal yet
        // Batches 2..n: marginal accumulates, fill untouched.
        l.record_batch(0, 8.0, 2.0);
        l.record_batch(0, 8.0, 2.0);
        assert_eq!(l.steps(0), 3);
        assert!(l.has_delivered_sample(0));
        assert_eq!(l.delivered_ms(0), 20.0);
        assert_eq!(l.delivered_batches(0), 2);
        // fill excess = first (15) - marginal (10) = 5.
        assert!((l.fill_excess_ms(0) - 5.0).abs() < 1e-12);
        // Untouched rank stays zeroed.
        assert_eq!(l.steps(1), 0);
    }

    #[test]
    fn fill_excess_clamps_at_zero() {
        let mut l = WindowLedger::new(1);
        l.record_batch(0, 1.0, 0.0); // first: 1ms
        l.record_batch(0, 8.0, 2.0); // marginal: 10ms > first
        assert_eq!(l.fill_excess_ms(0), 0.0);
    }

    #[test]
    fn absorb_callback_cost_clamps_and_ignores_oob() {
        let mut l = WindowLedger::new(1);
        l.record_batch(0, 10.0, 0.0);
        l.record_batch(0, 10.0, 5.0);
        l.absorb_callback_cost(0, 12.0);
        assert_eq!(l.wall_ms(0), 8.0);
        assert_eq!(l.delivered_ms(0), 3.0);
        l.absorb_callback_cost(0, 100.0); // clamp, not underflow
        assert_eq!(l.wall_ms(0), 0.0);
        assert_eq!(l.delivered_ms(0), 0.0);
        l.absorb_callback_cost(7, 1.0); // out of range: no-op
    }

    #[test]
    fn record_batch_ignores_oob_rank() {
        let mut l = WindowLedger::new(1);
        l.record_batch(3, 1.0, 1.0);
        assert_eq!(l.total_steps(), 0);
    }

    #[test]
    fn resets_are_split_steps_vs_timing() {
        let mut l = WindowLedger::new(2);
        l.record_batch(0, 10.0, 5.0);
        l.record_batch(0, 10.0, 5.0);
        l.record_batch(1, 4.0, 1.0);
        l.reset_timing();
        // Timing gone, steps preserved (backend-specific placement).
        assert_eq!(l.wall_ms(0), 0.0);
        assert!(!l.has_delivered_sample(0));
        assert_eq!(l.fill_excess_ms(0), 0.0);
        assert_eq!(l.steps(0), 2);
        l.reset_steps();
        assert_eq!(l.steps(0), 0);
        assert_eq!(l.steps(1), 0);
    }

    #[test]
    fn cohort_step_stats() {
        let mut l = WindowLedger::new(3);
        l.record_batch(0, 1.0, 0.0);
        l.record_batch(0, 1.0, 0.0);
        l.record_batch(2, 1.0, 0.0);
        assert_eq!(l.min_steps(), 0);
        assert_eq!(l.max_steps(), 2);
        assert_eq!(l.total_steps(), 3);
    }

    #[test]
    fn mean_loss_is_absent_until_a_batch_reports() {
        let mut l = WindowLedger::new(2);
        // Absent, NOT zero, before any batch.
        assert_eq!(l.mean_loss(0), None);
        l.record_batch_loss(0, 0.4);
        l.record_batch_loss(0, 0.6);
        assert!((l.mean_loss(0).unwrap() - 0.5).abs() < 1e-12);
        // A rank that reported nothing stays absent.
        assert_eq!(l.mean_loss(1), None);
        // Out-of-range rank: no panic, no record.
        l.record_batch_loss(9, 1.0);
        assert_eq!(l.mean_loss(9), None);
    }

    #[test]
    fn non_finite_loss_never_poisons_the_window_mean() {
        let mut l = WindowLedger::new(1);
        l.record_batch_loss(0, 0.5);
        l.record_batch_loss(0, f64::NAN);
        l.record_batch_loss(0, f64::INFINITY);
        // Only the finite sample counts.
        assert_eq!(l.mean_loss(0), Some(0.5));
    }

    #[test]
    fn loss_resets_with_the_window_timing() {
        let mut l = WindowLedger::new(1);
        l.record_batch_loss(0, 2.0);
        assert_eq!(l.mean_loss(0), Some(2.0));
        l.reset_timing();
        // Absent again — a new window starts unmeasured, not at zero.
        assert_eq!(l.mean_loss(0), None);
        // And the count reset with it, so the next window's mean is its own.
        l.record_batch_loss(0, 3.0);
        assert_eq!(l.mean_loss(0), Some(3.0));
    }

    #[test]
    fn loss_count_is_independent_of_step_resets() {
        // The loss denominator must NOT be `steps`: the CPU path resets
        // steps before the dispatch fold, which would divide the window's
        // loss by an already-zeroed count.
        let mut l = WindowLedger::new(1);
        l.record_batch(0, 1.0, 0.0);
        l.record_batch_loss(0, 4.0);
        l.record_batch(0, 1.0, 0.0);
        l.record_batch_loss(0, 6.0);
        l.reset_steps(); // backend-specific, mid-window
        assert_eq!(l.steps(0), 0);
        assert_eq!(l.mean_loss(0), Some(5.0)); // unaffected
    }

    #[test]
    fn per_batch_wall_is_infinite_cold() {
        let mut l = WindowLedger::new(1);
        assert_eq!(l.per_batch_wall_ms(0), f64::INFINITY);
        assert_eq!(l.per_batch_wall_ms(9), f64::INFINITY); // defensive oob
        l.record_batch(0, 6.0, 0.0);
        l.record_batch(0, 4.0, 0.0);
        assert!((l.per_batch_wall_ms(0) - 5.0).abs() < 1e-12);
    }
}
