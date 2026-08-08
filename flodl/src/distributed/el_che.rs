//! El Che: heterogeneous DDP cadence strategy.
//!
//! The column marches at the slowest one's pace. The slow device
//! anchors the cadence (`anchor` batches per sync step), the fast
//! ones range ahead doing more work, and everyone rejoins at AllReduce.
//! No one waits, no one idles.
//!
//! After each sync step, call [`report_timing`](ElChe::report_timing)
//! with measured wall times and AllReduce overhead. El Che refines
//! batch ratios and auto-tunes the anchor count to keep AllReduce overhead
//! below a configurable target (default 10%).
//!
//! # Example
//!
//! ```ignore
//! let ddp = Ddp::wrap(&[&model0, &model1], &devices)?;
//! let mut cadence = ElChe::new(2, 10);
//!
//! loop {
//!     let start_events = record_start_events(&devices)?;
//!     for rank in 0..2 {
//!         for _ in 0..cadence.batches(rank) {
//!             forward_backward(rank)?;
//!         }
//!     }
//!     let wall_ms = measure_elapsed(&start_events)?;
//!
//!     let sync_start = Instant::now();
//!     ddp.weighted_all_reduce_gradients(cadence.batch_counts())?;
//!     let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;
//!
//!     cadence.report_timing(&wall_ms, cadence.batch_counts(), sync_ms);
//! }
//! ```

/// Cohort band: a rank is in the slow-cohort (election-eligible) when its
/// smoothed ms is within `(1 - COHORT_BAND)` of the slowest. Excludes
/// clearly-fast GPUs from anchor candidacy by *evidence*, not by oracle —
/// once timing converges, an RTX with materially lower ms_per_batch is not
/// a candidate at all. Spec prior carries the same job during cold start.
const COHORT_BAND: f64 = 0.15;

/// Dominance margin: within the cohort, only swap to a challenger when its
/// smoothed ms exceeds the current anchor's smoothed ms by ≥ this fraction.
/// Sticky-with-margin replaces the prior single tie-band; near-identical
/// ranks (e.g. two same-model GPUs) won't churn on noise.
const DOMINANCE_MARGIN: f64 = 0.10;

/// Trust window capacity: per-rank ring buffer of recent ms_per_batch
/// readings. Replaces the prior EMA + adaptive-α scheme. Mean across
/// the window is the smoothed signal for both election (cohort threshold,
/// dominance margin) and batch-count proportions.
const TRUST_WINDOW_CAP: usize = 5;

/// Upper clamp on the per-rank speed ratio `slow_ms / ms`. No real rig
/// exceeds ~10x; a legitimate-but-degenerate tiny sample (sub-ms wall in
/// a 1-sample trust window) would otherwise turn into a 1e4x ratio and a
/// garbage 1e5-batch schedule on paths without `max_batch_diff` /
/// `max_total_batches`.
const MAX_SPEED_RATIO: f64 = 64.0;

/// Sliding-window capacity for `batch_counts` snapshots. Source for
/// `recent_batch_share`, the smoothed view of cadence per-rank allocation
/// reported as the per-epoch batch-share metric. ~10 syncs balances
/// responsiveness (catches genuine cadence shifts within an epoch) and
/// stability (one noisy sync doesn't move the metric).
const BATCH_COUNTS_WINDOW_CAP: usize = 10;

/// Consecutive `Stable` convergence verdicts required to re-arm window-
/// pressure growth after the guard returned anything else. The asymmetry
/// (disable on the first non-`Stable`, re-arm only after a clean streak) is
/// the margin to the convergence cliff: the controller settles below the
/// boundary instead of limit-cycling against it. See [`ElChe::growth_enabled`].
const GROWTH_REARM_STABLE: usize = 5;

/// Cap on a single window-pressure grow step (multiplicative). The proposal
/// scale is `overhead_fraction / overhead_target`, which can be large on the
/// first big-overhead reading (the historical 10→22 single-shot jump);
/// capping at ×2/cycle rides the steep part of the amortization curve in a
/// few cycles without overshooting the knee on one noisy sample.
const GROWTH_STEP_CAP: f64 = 2.0;

/// Signal margin required for a window-pressure proposal to fire during
/// `Warmup`, as a multiple of `overhead_target`. Set to [`GROWTH_STEP_CAP`]
/// on purpose: clearing it means `scale = overhead/target ≥ cap`, so every
/// early proposal is already cap-clamped — acting before the trust window
/// fills is allowed exactly where noise cannot change the decision. A
/// sync-swamped cohort (tiny model, expensive reduce: overhead near 1.0)
/// starts amortizing on its second window instead of its sixth, while
/// borderline pressure near the knee — where jittery early readings could
/// flip the sign — keeps the full five-calibration patience.
const WARMUP_GROWTH_MARGIN: f64 = GROWTH_STEP_CAP;

// Anchor swaps are gated by `Phase::Stable` (≥5 calibrations). Stable starts
// at the 6th `report_timing` call, by which point each rank has a full
// 5-sample trust window AND the noisiest first sample (kernel JIT, cuBLAS
// plan caching, NCCL buffer allocation — costs that ride disproportionately
// on the newer/larger GPU's first few syncs) has rolled out. This keeps the
// initial-pick lock long enough to weather cold-start measurement skew.

/// FIFO ring buffer of f64 samples with a fixed capacity. Used per-rank to
/// hold the most recent `TRUST_WINDOW_CAP` ms_per_batch readings; mean
/// over the buffer is the smoothed signal consumed by election and
/// batch-count proportioning.
#[derive(Debug, Clone)]
struct RingBuffer {
    samples: Vec<f64>,
    capacity: usize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, value: f64) {
        if self.samples.len() >= self.capacity {
            self.samples.remove(0);
        }
        self.samples.push(value);
    }

    /// Mean of samples currently in the buffer. Returns 0.0 when empty,
    /// preserving the "no data yet" sentinel used by callers.
    fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Lifecycle phase of the cadence balancer. Probe = no calibrations yet,
/// Warmup = first few calibrations (the anchor pin is held; growth may act
/// from the second calibration if it clears the margin), Stable = normal
/// operation including anchor election and overhead auto-tune at the real
/// target, Mature = long-running steady state. Phase ordering is monotonic
/// and supports `>=` comparisons for gating logic.
///
/// Mature gates the same ElChe actions as Stable; the difference lives in
/// the LR-aware meta-controller, which reads the phase to pick a gentler
/// nudge factor and a shorter sustain count as trust accumulates (see
/// `lr_event_meta::base_factor_for` / `sustain_k_for`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Phase {
    /// Initial fixed-size measurement period before any averaging-driven
    /// calibration. Same code path as the legacy "uncalibrated" branch;
    /// exits to `Warmup` on the first successful `report_timing` call.
    Probe,
    /// First few calibrations after Probe — anchor should change rarely.
    Warmup,
    /// Normal operation with hysteresis on anchor changes.
    Stable,
    /// Long-running steady-state with full failure-state machinery.
    Mature,
}

/// Source-agnostic anchor verdict: what a convergence assessment asks
/// ElChe to do with its window, regardless of who produced it (the
/// convergence guard, the LR-aware meta-controller, or any future
/// detector). ElChe applies the verdict via [`ElChe::apply_verdict`]
/// and never learns the source; arbitration between competing verdict
/// producers is the caller's, in one place.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AnchorVerdict {
    /// Convergence is clean: commit any pending window-pressure grow
    /// proposal and count toward the growth re-arm latch. With
    /// `relax_up`, additionally drift the anchor one batch toward
    /// `max_anchor` (the async-mode opt-in; see
    /// [`ElChe::relax_anchor_up`] for the drift-cap rules).
    Stable {
        /// Also relax the anchor upward by one (async opt-in).
        relax_up: bool,
    },
    /// Divergence trending up: drop the pending grow proposal and latch
    /// growth off until consecutive `Stable` verdicts re-arm it.
    SuppressGrowth,
    /// Sustained divergence (or a proactive prediction, e.g. an LR
    /// cliff): shrink the anchor multiplicatively by `factor`
    /// (0.5 = halve), drop the pending grow proposal, latch growth off.
    NudgeDown {
        /// Multiplicative anchor shrink factor, clamped to `[0.1, 1.0]`.
        factor: f64,
    },
}

/// One reduce window's observations, the event-shaped timing feed for
/// [`ElChe::report_window`]. Built by the coordinator from its window
/// ledger once per averaging cycle; owned values because it is a
/// message, not a view.
///
/// Carries BOTH timing scales so the scale-selection policy (the
/// mixed-scale inversion guard) lives in ElChe, next to the relative
/// allocation model it protects:
///
/// - compute-only `(wall_ms, steps)` — always coherent across ranks;
/// - marginal delivered `(delivered_ms, delivered_batches)` — carries
///   data + transport cost; a rank without a delivered sample this
///   window has `(0.0, 0)`.
///
/// `delivered_coherent` is the caller's attestation that the delivered
/// scale is safe to use at all: the mode rides the delivered feed AND
/// every alive mover has a delivered sample this window. The caller
/// owns that predicate because it owns membership (dead ranks) and the
/// ledger; ElChe owns what to do with it — feeding a PARTIAL delivered
/// set would compare incomparable scales and invert the allocation
/// (see [`WindowReport::select_feed`]).
#[derive(Debug, Clone)]
pub struct WindowReport {
    /// Per-rank compute-only wall (ms) this window.
    pub wall_ms: Vec<f64>,
    /// Per-rank step counts this window (matched divisor for `wall_ms`).
    pub steps: Vec<usize>,
    /// Per-rank marginal delivered wall (ms) this window.
    pub delivered_ms: Vec<f64>,
    /// Per-rank marginal delivered batch counts (matched divisor for
    /// `delivered_ms`).
    pub delivered_batches: Vec<usize>,
    /// Per-rank first-batch fill excess (ms) — the window-pressure
    /// growth signal.
    pub fill_ms: Vec<f64>,
    /// The delivered scale is coherent this window (mode supports it
    /// AND every alive mover has a delivered sample).
    pub delivered_coherent: bool,
    /// Duration (ms) of the sync that closed the window.
    pub sync_ms: f64,
}

impl WindowReport {
    /// The `(ms, batches)` pair ElChe schedules on this window — the
    /// scale-selection policy. ElChe's allocation is RELATIVE, so the
    /// per-rank scale must be uniform within a window: when the
    /// delivered scale is not coherent, EVERY rank feeds the compute
    /// scale (a uniformly compute-fed window is coherent; the next
    /// window returns to delivered). When coherent, ranks with a
    /// delivered sample feed delivered cost; ranks without one are
    /// non-movers by the caller's attestation and fall back to their
    /// (empty) compute pair, contributing `(0, 0)` on either scale.
    pub fn select_feed(&self) -> (Vec<f64>, Vec<usize>) {
        if !self.delivered_coherent {
            return (self.wall_ms.clone(), self.steps.clone());
        }
        let mut ms = Vec::with_capacity(self.wall_ms.len());
        let mut batches = Vec::with_capacity(self.wall_ms.len());
        for r in 0..self.wall_ms.len() {
            let has_sample = self.delivered_batches[r] > 0 && self.delivered_ms[r] > 0.0;
            if has_sample {
                ms.push(self.delivered_ms[r]);
                batches.push(self.delivered_batches[r]);
            } else {
                ms.push(self.wall_ms[r]);
                batches.push(self.steps[r]);
            }
        }
        (ms, batches)
    }
}

pub struct ElChe {
    world_size: usize,
    /// Anchor batch count (slow device processes this many per step).
    anchor: usize,
    /// Per-device batch counts for the current cadence step.
    batch_counts: Vec<usize>,
    /// Per-rank trust window of recent ms_per_batch readings. Mean over the
    /// window is the smoothed signal for election + batch-count proportions.
    /// Replaces the prior EMA + adaptive-α scheme; window-mean gives O(K)
    /// memory, uniform weighting, and survives single-reading outliers
    /// without per-call α math.
    ms_per_batch_window: Vec<RingBuffer>,
    /// Per-rank counter of consecutive zero/invalid wall_ms reports. When a
    /// rank misses `TRUST_WINDOW_CAP` reports in a row, its window is cleared
    /// so smoothing can react fast on recovery (death exception).
    consecutive_zero_reports: Vec<usize>,
    /// Sliding window of recent batch_counts snapshots, captured at the end
    /// of each `report_timing` after `recompute_batch_counts` settles. Source
    /// for `recent_batch_share` — exposes the cadence's actual per-rank
    /// allocation as an observation, not a separate prediction. Capped at
    /// `BATCH_COUNTS_WINDOW_CAP`.
    batch_counts_window: std::collections::VecDeque<Vec<usize>>,
    /// Whether at least one real measurement has been taken.
    calibrated: bool,
    /// Window-pressure target: max per-window FIXED overhead (reduce + fill)
    /// as a fraction of the bottleneck rank's window wall. The anchor grows
    /// to keep it below this; see `propose_anchor`.
    overhead_target: f64,
    /// Minimum anchor (never below initial value).
    min_anchor: usize,
    /// Maximum anchor (gradient staleness limit).
    max_anchor: usize,
    /// Upper bound on the total reduce window (`sum(batch_counts)`), set
    /// by the coordinator to the epoch's batch count. The overhead
    /// auto-tune may grow the per-rank schedule to amortize sync cost,
    /// but a reduce window must fit within one epoch — otherwise a single
    /// window spans multiple dataset passes, collapsing the sync rate and
    /// breaking the "controller knows exactly how many steps per reduce
    /// and per epoch" invariant. `None` = unbounded (threaded default;
    /// only the cluster coordinator sets it).
    max_total_batches: Option<usize>,
    /// True when the last `recompute_batch_counts` had to scale the
    /// schedule down to fit `max_total_batches` (the window≤epoch cap).
    /// While binding, anchor GROWTH proposals are suppressed: the
    /// delivered window cannot actually grow, so a Grow would only
    /// ratchet the anchor toward `max_anchor` while every
    /// anchor-derived quantity (telemetry, nudge arithmetic) quietly
    /// detaches from the schedule being run.
    window_cap_binding: bool,
    /// Maximum allowed batch difference between fastest and slowest worker.
    /// When set, workers that exceed this lead are throttled until the
    /// slowest catches up. `Some(0)` = strict lockstep (sync DDP behavior).
    max_batch_diff: Option<usize>,
    /// Current lifecycle phase. Starts at `Probe`, progresses on calibration.
    phase: Phase,
    /// Currently elected slow-anchor rank (None until first calibration).
    /// Replaces the implicit `argmax(ms_per_batch)` pick: stickiness +
    /// deterministic tiebreak prevents flap when two ranks are within
    /// `TIE_BAND` of each other in measured speed.
    anchor_rank: Option<usize>,
    /// Number of successful `report_timing` calls (each one a calibration).
    /// Drives phase transitions Warmup→Stable→Mature.
    calibration_count: u64,
    /// Per-rank ms of wall-time to subtract from the next batch-count
    /// recompute, consumed once then auto-cleared. Set by the
    /// coordinator before the LAST sync cycle of an epoch that fires a
    /// user callback (`eval_fn` / `epoch_fn` / `checkpoint_fn`) so the
    /// firing rank's quota shrinks just enough to absorb the callback
    /// wall-time inside its compute slack instead of bloating the
    /// sync-barrier wait. Zero entries are no-ops.
    ///
    /// Consumed in `recompute_batch_counts`: rank `r`'s
    /// computed target drops by `ceil(slack_ms[r] / smoothed_ms[r])`
    /// (clamped at 1) on the next recompute. The vector is zeroed in
    /// the same call so the effect lands exactly once per
    /// `apply_callback_slack` invocation.
    pending_callback_slack_ms: Vec<f64>,
    /// Pending anchor change from the last `report_timing`'s
    /// `overhead_target` auto-tune, awaiting a [`ConvergenceGuard`]
    /// verdict to commit or veto. `None` outside of `Phase::Stable+`,
    /// inside the 5% dead-zone, or when no `report_timing` has fired
    /// yet. The convergence-guard verdict drives one of:
    ///
    /// - [`Self::commit_proposed_anchor`] (Stable): apply the proposal
    ///   regardless of direction.
    /// - [`Self::veto_proposed_growth`] (SuppressGrowth): drop the
    ///   pending grow and latch growth off until re-armed — divergence
    ///   is rising and growth would make it worse (proposals are
    ///   grow-only by design; see [`ProposedAnchor`]).
    /// - [`Self::discard_proposed_anchor`] (NudgeDown): drop the
    ///   proposal; the nudge supersedes it directly on the current
    ///   anchor.
    ///
    /// Transient between `report_timing` and the verdict apply — never
    /// serialized into [`crate::distributed::ElCheState`].
    ///
    /// [`ConvergenceGuard`]: super::ddp_run::ConvergenceGuard
    proposed_anchor: Option<ProposedAnchor>,
    /// Per-rank amortizable per-window FILL (ms), set by the coordinator
    /// before `report_timing` and consumed once by the next
    /// `propose_anchor`. The fill is the window's first-batch excess over
    /// the steady-state (marginal) rate — control transit, plan pickup,
    /// prefetch spin-up, first-batch unpipelined H2D — the cost the
    /// marginal-anchor allocation feed deliberately excludes (batch 1
    /// skipped). It is the window-pressure signal: a fixed per-window cost
    /// that amortizes as the window grows. `0.0` (unset, e.g. before
    /// any coordinator report) makes window-pressure fall back to the
    /// reduce-overhead term alone. Indexed by the elected anchor rank in
    /// `propose_anchor`. Zeroed in `report_timing` after the proposal so a
    /// stale fill cannot drive a later cycle.
    pending_window_fill_ms: Vec<f64>,
    /// Growth-enable latch for the window-pressure controller. Set `false`
    /// the instant the convergence guard returns anything other than
    /// `Stable` (`SuppressGrowth` early-warning OR `NudgeDown`), and
    /// re-armed to `true` only after `GROWTH_REARM_STABLE` consecutive
    /// `Stable` verdicts. This is the margin to the convergence cliff: the
    /// controller backs off the instant divergence trends up and refuses to
    /// poke the boundary again until convergence is robustly clean, rather
    /// than re-attempting growth every cycle. Starts `true` (nothing seen
    /// yet). Driven by `commit_proposed_anchor` / `veto_proposed_growth` /
    /// `discard_proposed_anchor`.
    growth_enabled: bool,
    /// Count of consecutive `Stable` guard verdicts since the last non-Stable
    /// one; re-arms `growth_enabled` at `GROWTH_REARM_STABLE`.
    consecutive_stable: usize,
    /// Whether window-pressure anchor growth applies at all under the active
    /// reduce policy. Growth amortizes per-window fixed cost across a cadence
    /// window; **Sync has no window** (it reduces every step, gated at
    /// `steps >= 1`), so growth there is meaningless and would only inflate the
    /// telemetry anchor and — worse — the checkpointed `ElCheState.anchor`,
    /// mis-seeding a later Cadence resume. Set `false` for Sync (see
    /// [`Self::with_window_growth_applicable`]); `true` for Cadence/Async.
    /// Config-derived, NOT trajectory state: [`Self::restore_from_state`]
    /// leaves it untouched.
    window_growth_applicable: bool,
}

/// A staged window-pressure grow proposal awaiting a convergence-guard
/// verdict. See [`ElChe::proposed_anchor`].
///
/// Grow-only by design: the window-pressure controller never shrinks the
/// anchor for "fresher gradients". Empirically, in the regime this targets
/// (`H_max` far above one epoch) fewer syncs is harmless to convergence, so
/// shrinking only re-creates the small-window overhead it exists to remove.
/// The single downward force is the convergence guard's `NudgeDown`
/// ([`ElChe::nudge_anchor_down`]), and the epoch cap
/// ([`ElChe::set_max_total_batches`]) is the hard ceiling.
#[derive(Debug, Clone, Copy)]
enum ProposedAnchor {
    /// Per-window fixed overhead (reduce + fill) exceeds `overhead_target`
    /// as a fraction of window wall: grow the anchor to amortize it over
    /// more local batches. Committed by `Stable`; dropped by
    /// `SuppressGrowth` / `NudgeDown`.
    Grow(usize),
}

impl ElChe {
    /// Create a new sync cadence.
    ///
    /// `world_size`: number of devices (must be >= 2).
    /// `anchor`: initial batch count for the slow device per sync step.
    ///
    /// The first step uses equal counts (`anchor` for every device).
    /// After [`report_timing`](ElChe::report_timing), ratios adapt
    /// to measured throughput.
    pub fn new(world_size: usize, anchor: usize) -> Self {
        assert!(world_size >= 2, "El Che requires at least 2 devices");
        assert!(anchor >= 1, "anchor must be >= 1");
        ElChe {
            world_size,
            anchor,
            batch_counts: vec![anchor; world_size],
            ms_per_batch_window: (0..world_size)
                .map(|_| RingBuffer::new(TRUST_WINDOW_CAP))
                .collect(),
            consecutive_zero_reports: vec![0; world_size],
            batch_counts_window: std::collections::VecDeque::with_capacity(BATCH_COUNTS_WINDOW_CAP),
            calibrated: false,
            overhead_target: 0.05,
            min_anchor: anchor,
            max_anchor: 1000,
            max_total_batches: None,
            window_cap_binding: false,
            max_batch_diff: None,
            phase: Phase::Probe,
            anchor_rank: None,
            calibration_count: 0,
            pending_callback_slack_ms: vec![0.0; world_size],
            proposed_anchor: None,
            pending_window_fill_ms: vec![0.0; world_size],
            growth_enabled: true,
            consecutive_stable: 0,
            // Default applicable; the coordinator flips this off for Sync.
            window_growth_applicable: true,
        }
    }

    /// Current lifecycle phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Currently elected slow-anchor rank (None until first calibration).
    pub fn anchor_rank(&self) -> Option<usize> {
        self.anchor_rank
    }

    /// Snapshot this ElChe's trajectory state for checkpoint persistence.
    ///
    /// Captures the fields a resume API needs to restore cadence
    /// behavior without re-calibrating from scratch: anchor, elected
    /// anchor rank, per-rank smoothed `ms_per_batch` (trust-window
    /// mean), phase, calibration count. User-set knobs
    /// (`overhead_target`, `min_anchor`, `max_anchor`, `max_batch_diff`)
    /// are NOT captured — they come from the user's `DdpRunConfig` at
    /// controller construction on resume, so a re-bind to different
    /// knobs is supported.
    ///
    /// Called by [`ClusterCoordinator`] when broadcasting
    /// `ShutdownWithSave`; the produced state is written to
    /// `<save_path>.meta.json` via [`CheckpointMeta::with_elche_state`].
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`CheckpointMeta::with_elche_state`]:
    ///     crate::distributed::CheckpointMeta::with_elche_state
    pub fn to_state(&self) -> crate::distributed::ElCheState {
        let smoothed_ms_per_batch: Vec<f64> =
            (0..self.world_size).map(|r| self.smoothed_ms(r)).collect();
        crate::distributed::ElCheState {
            anchor: self.anchor,
            anchor_rank: self.anchor_rank,
            smoothed_ms_per_batch,
            phase: self.phase,
            calibration_count: self.calibration_count,
            // `trend_history` is convergence-guard state, not ElChe
            // state — the coordinator populates it from
            // `convergence_guard.trend_history()` after this call (see
            // `ClusterCoordinator::dispatch_shutdown_with_save`).
            trend_history: None,
        }
    }

    /// Restore the dynamic trajectory fields from a previously-captured
    /// [`crate::distributed::ElCheState`] snapshot. Inverse of
    /// [`Self::to_state`].
    ///
    /// Restored: `anchor`, `anchor_rank`, `phase`, `calibration_count`,
    /// `calibrated` (true when any rank had a positive smoothed reading),
    /// and the per-rank trust window seeded with the saved
    /// `smoothed_ms_per_batch`. The user-set knobs (`overhead_target`,
    /// `min_anchor`, `max_anchor`, `max_batch_diff`) are left as-is on
    /// `self` — the caller already configured those from the user's
    /// `DdpRunConfig` at construction time, and resume by design
    /// supports re-binding to different knobs.
    ///
    /// `state.world_size` must match `self.world_size`; the saved
    /// `smoothed_ms_per_batch` length is the authoritative check. A
    /// mismatch surfaces loudly so callers don't silently resume a
    /// 3-rank snapshot into a 2-rank cluster (a config-coherence bug).
    ///
    /// Window seeding is lossy by design: the snapshot only carries the
    /// trust-window mean, not the raw samples. We seed each rank's
    /// window with one sample equal to the mean. The first few
    /// post-resume `report_timing` calls re-populate raw samples and
    /// the smoothed signal converges back to actual conditions within
    /// `TRUST_WINDOW_CAP` calibrations.
    pub fn restore_from_state(
        &mut self,
        state: &crate::distributed::ElCheState,
    ) -> crate::tensor::Result<()> {
        if state.smoothed_ms_per_batch.len() != self.world_size {
            return Err(crate::tensor::TensorError::new(&format!(
                "ElChe::restore_from_state: snapshot world_size {} != \
                 current world_size {}; resume must use the same world \
                 size as the saved run",
                state.smoothed_ms_per_batch.len(),
                self.world_size,
            )));
        }
        self.anchor = state.anchor;
        self.anchor_rank = state.anchor_rank;
        self.phase = state.phase;
        self.calibration_count = state.calibration_count;
        for (rank, &smoothed) in state.smoothed_ms_per_batch.iter().enumerate() {
            self.ms_per_batch_window[rank].clear();
            if smoothed > 0.0 {
                self.ms_per_batch_window[rank].push(smoothed);
            }
        }
        // `calibrated` is true iff any rank had a real reading — matches
        // the post-`report_timing` invariant the snapshot was taken under.
        self.calibrated = state.smoothed_ms_per_batch.iter().any(|&v| v > 0.0);
        Ok(())
    }

    /// Smoothed ms_per_batch for `rank` — mean over the trust window.
    /// 0.0 when window is empty (rank hasn't produced a positive reading yet).
    fn smoothed_ms(&self, rank: usize) -> f64 {
        self.ms_per_batch_window
            .get(rank)
            .map(|w| w.mean())
            .unwrap_or(0.0)
    }

    /// Slow-cohort: ranks whose smoothed ms is within `(1 - COHORT_BAND)`
    /// of the slowest. Implements "fast GPU never anchor" by evidence —
    /// once timing converges, a clearly-faster rank falls outside the band
    /// and is excluded from anchor candidacy. Returns empty when no rank has
    /// a positive smoothed reading yet.
    fn slow_cohort(&self) -> Vec<usize> {
        let max_ms = (0..self.world_size)
            .map(|r| self.smoothed_ms(r))
            .fold(0.0_f64, f64::max);
        if max_ms <= 0.0 {
            return Vec::new();
        }
        let threshold = max_ms * (1.0 - COHORT_BAND);
        (0..self.world_size)
            .filter(|&r| self.smoothed_ms(r) >= threshold)
            .collect()
    }

    /// Elect the slow-anchor rank: cohort filter + within-cohort sticky-with-
    /// margin. A rank is a candidate only if its smoothed ms is in the slow
    /// cohort (within `COHORT_BAND` of slowest). Within the cohort, the
    /// current anchor is kept unless a challenger's smoothed ms exceeds it
    /// by ≥ `DOMINANCE_MARGIN`. Lowest-index tiebreak when no current anchor
    /// is in the cohort.
    fn elect_anchor(&self) -> Option<usize> {
        let cohort = self.slow_cohort();
        if cohort.is_empty() {
            return None;
        }
        if cohort.len() == 1 {
            return Some(cohort[0]);
        }
        if let Some(c) = self.anchor_rank
            && cohort.contains(&c)
        {
            let cur = self.smoothed_ms(c);
            let challenger = cohort
                .iter()
                .copied()
                .filter(|&r| r != c)
                .map(|r| (r, self.smoothed_ms(r)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((other, other_ms)) = challenger
                && other_ms > cur * (1.0 + DOMINANCE_MARGIN)
            {
                return Some(other);
            }
            return Some(c);
        }
        cohort.into_iter().min()
    }

    /// `ms_per_batch` of the elected anchor (smoothed). 0.0 if not yet elected.
    fn slow_ms(&self) -> f64 {
        self.anchor_rank.map(|r| self.smoothed_ms(r)).unwrap_or(0.0)
    }

    /// Set the target per-window FIXED overhead (reduce + fill) as a
    /// fraction of the bottleneck rank's window wall.
    ///
    /// Default: 0.05 (5%). The anchor auto-tunes upward to keep this
    /// per-window overhead below the target — fewer, larger windows amortize
    /// the fixed cost. Lower values = fewer syncs = larger window = more
    /// gradient staleness (bounded by the convergence guard and the epoch
    /// cap). Clamped to `[0.01, 0.50]`.
    pub fn with_overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = target.clamp(0.01, 0.50);
        self
    }

    /// Enable or disable window-pressure anchor growth for the active reduce
    /// policy.
    ///
    /// Growth amortizes per-window fixed cost (reduce + fill) across a cadence
    /// window, so it is only meaningful when reduces are *windowed*
    /// (Cadence / Async). Under Sync the reduce fires every step, so there is
    /// no window to amortize; leaving growth on there would inflate the
    /// telemetry anchor and the checkpointed `ElCheState.anchor`, mis-seeding a
    /// later Cadence resume. The coordinator calls this with `false` in Sync
    /// mode and `true` otherwise. Default (unset): `true`.
    pub fn with_window_growth_applicable(mut self, applicable: bool) -> Self {
        self.window_growth_applicable = applicable;
        self
    }

    /// Set the maximum anchor count (gradient staleness limit).
    ///
    /// Default: 1000. Higher values allow fewer syncs but accumulate more
    /// batches of gradient before averaging. Set to 1 to sync after every
    /// slow-device batch (minimal accumulation, traditional DDP cadence).
    /// The overhead auto-tune typically settles well below this cap; the
    /// default exists primarily as a safety net against runaway growth.
    pub fn with_max_anchor(mut self, max: usize) -> Self {
        self.max_anchor = max.max(1);
        // Ensure min_anchor doesn't exceed max_anchor
        if self.min_anchor > self.max_anchor {
            self.min_anchor = self.max_anchor;
            self.anchor = self.anchor.clamp(self.min_anchor, self.max_anchor);
        }
        self
    }

    /// Cap the total reduce window (`sum(batch_counts)`) at `max_total`
    /// batches — set by the coordinator to the epoch's batch count so a
    /// reduce window can never grow past one dataset pass. The overhead
    /// auto-tune still grows the schedule to amortize sync cost, but
    /// `recompute_batch_counts` scales the per-rank counts down
    /// proportionally if their sum would exceed this bound. `None`
    /// (default) leaves the window unbounded.
    pub fn set_max_total_batches(&mut self, max_total: usize) {
        self.max_total_batches = if max_total == 0 {
            None
        } else {
            Some(max_total)
        };
    }

    /// Set the minimum anchor count (overhead auto-tune floor).
    ///
    /// Default: equals the initial anchor (so the auto-tune cannot shrink
    /// below the start value). Set explicitly to force the auto-tune above
    /// its natural overhead-equilibrium setpoint, or together with
    /// [`Self::with_max_anchor`] (same value) to pin the anchor at a fixed
    /// cadence.
    ///
    /// Note: only the overhead auto-tune's shrink path honors this floor.
    /// The convergence guard's `NudgeDown`, which routes through
    /// [`Self::nudge_anchor_down`], bypasses it (treated as a stronger
    /// signal than overhead). For hard pinning, also disable the
    /// convergence guard at the
    /// [`crate::distributed::ddp_run::DdpRunConfig`] level.
    pub fn with_min_anchor(mut self, min: usize) -> Self {
        self.min_anchor = min.max(1);
        // Ensure min_anchor doesn't exceed max_anchor
        if self.min_anchor > self.max_anchor {
            self.min_anchor = self.max_anchor;
        }
        // Lift the current anchor up to the new floor if needed
        if self.anchor < self.min_anchor {
            self.anchor = self.min_anchor;
        }
        self
    }

    /// Set the maximum batch difference between fastest and slowest worker.
    ///
    /// When the fastest worker leads the slowest by more than this many
    /// batches, it is throttled (paused) until the gap closes. This prevents
    /// catastrophic divergence with large batches or extreme speed ratios.
    ///
    /// - `None` (default): no limit, workers run freely.
    /// - `Some(0)`: strict lockstep, equivalent to synchronous DDP.
    /// - `Some(n)`: fast workers may lead by at most `n` batches.
    pub fn with_max_batch_diff(mut self, max: usize) -> Self {
        self.max_batch_diff = Some(max);
        self
    }

    /// Current max batch diff setting.
    pub fn max_batch_diff(&self) -> Option<usize> {
        self.max_batch_diff
    }

    /// Set initial speed estimate before the first timing measurement.
    ///
    /// `slow_rank`: which device is slowest (receives `anchor` batches).
    /// `ratio`: how many times faster the fastest device is (e.g., 3.0
    /// means the fast GPU processes ~3x more batches per unit time).
    ///
    /// Default (without this call): all devices start equal (`anchor`
    /// batches each). After the first [`report_timing`](ElChe::report_timing),
    /// actual measurements replace this estimate, so even a wrong guess
    /// self-corrects in one step.
    ///
    /// ```ignore
    /// // RTX 5060 Ti (rank 0) is ~2.3x faster than GTX 1060 (rank 1)
    /// let che = ElChe::new(2, 10).with_speed_ratio(1, 2.3);
    /// // → rank 0: 23 batches, rank 1: 10 batches
    /// ```
    pub fn with_speed_ratio(mut self, slow_rank: usize, ratio: f64) -> Self {
        assert!(
            slow_rank < self.world_size,
            "slow_rank ({slow_rank}) out of bounds for world_size ({})",
            self.world_size,
        );
        let ratio = ratio.max(1.0);
        for rank in 0..self.world_size {
            if rank == slow_rank {
                self.batch_counts[rank] = self.anchor;
            } else {
                self.batch_counts[rank] = (self.anchor as f64 * ratio).round().max(1.0) as usize;
            }
        }
        // The user is asserting `slow_rank` is the slowest device; record it
        // as the initial anchor so cold-start logic doesn't hand the role to
        // rank 0 by default. Subsequent `report_timing` calls may still
        // re-elect once enough timing data accumulates.
        self.anchor_rank = Some(slow_rank);
        self
    }

    /// Pin the initial slow-anchor rank without committing to a speed ratio.
    ///
    /// Used by the coordinator when the user supplies `partition_ratios`
    /// (smallest ratio = slow rank), or by `with_device_indices` after a
    /// spec-prior pick. The pin is "soft" — it only sets the cold-start
    /// anchor; once enough calibrations accumulate (Phase::Stable),
    /// `elect_anchor` may move the anchor based on measured timing.
    pub fn with_initial_anchor(mut self, slow_rank: usize) -> Self {
        assert!(
            slow_rank < self.world_size,
            "slow_rank ({slow_rank}) out of bounds for world_size ({})",
            self.world_size,
        );
        self.anchor_rank = Some(slow_rank);
        self
    }

    /// Auto-detect the cold-start anchor from device hardware specs.
    ///
    /// Queries each CUDA device's compute capability and total VRAM, scores
    /// them as `sm_major*100 + sm_minor*10 + vram_gb`, and picks the rank
    /// with the lowest score (slowest by spec). Skips silently if any
    /// device-property query fails (no CUDA, invalid index) or if an
    /// initial anchor was already pinned (e.g. via `with_speed_ratio` or
    /// `with_initial_anchor`) — explicit user knowledge outranks the prior.
    ///
    /// `device_indices` must be ordered by rank: `device_indices[r]` is the
    /// CUDA device index for DDP rank `r`.
    pub fn with_device_indices(mut self, device_indices: &[i32]) -> Self {
        if self.anchor_rank.is_some() {
            return self;
        }
        if device_indices.len() != self.world_size {
            return self;
        }
        if let Some(slow) = spec_prior::slowest_rank(device_indices) {
            self.anchor_rank = Some(slow);
        }
        self
    }

    /// Batch count for the given device rank in the current cadence step.
    pub fn batches(&self, rank: usize) -> usize {
        self.batch_counts[rank]
    }

    /// Per-device batch counts (for `Ddp::weighted_all_reduce_gradients`).
    pub fn batch_counts(&self) -> &[usize] {
        &self.batch_counts
    }

    /// Stage per-rank callback wall-time (ms) to absorb on the next
    /// `recompute_batch_counts` call. The coord sets this just
    /// before the last sync cycle of an epoch that fires a user
    /// callback on a known rank, so the firing rank's quota for that
    /// cycle drops by `ceil(slack_ms / smoothed_ms_per_batch)` batches
    /// — leaving compute slack to run the callback without bloating
    /// the barrier wait.
    ///
    /// Inputs:
    /// - `slack_ms`: length-`world_size` vector. Index `r` is the
    ///   callback budget for rank `r` (zero = no slack, the typical
    ///   case for non-firing ranks).
    ///
    /// Silently no-ops when `slack_ms.len() != self.world_size` to
    /// match the rest of the ElChe builder/setter shape (callers
    /// constructed off-by-one inputs would otherwise crash a running
    /// training cluster, not what we want; recompute-without-slack is
    /// a safe fallback).
    ///
    /// The slack is consumed exactly once per
    /// `recompute_batch_counts` call: after the per-rank
    /// targets are computed with the slack subtracted, the pending
    /// vector is zeroed. The caller can re-set the vector before each
    /// recompute, or leave it zeroed for cycles where no callback
    /// fires.
    pub fn apply_callback_slack(&mut self, slack_ms: &[f64]) {
        if slack_ms.len() != self.world_size {
            return;
        }
        self.pending_callback_slack_ms.clone_from_slice(slack_ms);
    }

    /// Read the currently-staged callback slack (ms per rank). Returns
    /// all-zero by default. Test/diagnostic accessor; production code
    /// goes through [`Self::apply_callback_slack`].
    pub fn pending_callback_slack_ms(&self) -> &[f64] {
        &self.pending_callback_slack_ms
    }

    /// Stage the per-rank per-window FILL (ms) for the window-pressure
    /// controller, consumed once by the next `propose_anchor`. The fill is
    /// the window's first-batch excess over the steady-state (marginal)
    /// rate — the amortizable per-window fixed cost the marginal-anchor
    /// allocation feed excludes. The coordinator computes it from the
    /// per-window timing and sets it before [`Self::report_timing`]; left
    /// unset (all-zero), window-pressure falls back to the reduce-overhead
    /// term alone.
    ///
    /// Silently no-ops on a length mismatch (matches the rest of the
    /// builder/setter shape: a caller off-by-one must not crash a running
    /// cluster; growing on the reduce term alone is a safe fallback).
    pub fn set_window_fill_ms(&mut self, fill_ms: &[f64]) {
        if fill_ms.len() != self.world_size {
            return;
        }
        self.pending_window_fill_ms.clone_from_slice(fill_ms);
    }

    /// Read the currently-staged per-window fill (ms per rank). Returns
    /// all-zero by default. Test/diagnostic accessor.
    pub fn pending_window_fill_ms(&self) -> &[f64] {
        &self.pending_window_fill_ms
    }

    /// Whether window-pressure growth is currently armed (the latch). Test/
    /// diagnostic accessor; the latch is driven by the guard verdict through
    /// [`Self::commit_proposed_anchor`] / [`Self::veto_proposed_growth`] /
    /// [`Self::discard_proposed_anchor`].
    pub fn growth_enabled(&self) -> bool {
        self.growth_enabled
    }

    /// Total batches across all devices for this cadence step.
    pub fn total_batches(&self) -> usize {
        self.batch_counts.iter().sum()
    }

    /// Current anchor batch count (slow device batches per step).
    pub fn anchor(&self) -> usize {
        self.anchor
    }

    /// Target wall time (ms) for one sync interval.
    ///
    /// Returns `anchor * slowest_ms_per_batch`, the intended wall-clock
    /// duration between AllReduce events. Both GPUs should accumulate
    /// this much compute time before syncing. Returns 0 if not yet
    /// calibrated (no timing data).
    pub fn anchor_wall_ms(&self) -> f64 {
        if !self.calibrated {
            return 0.0;
        }
        self.anchor as f64 * self.slow_ms()
    }

    /// Reduce the anchor by `factor` (e.g. 0.5 = halve).
    ///
    /// One-directional correction for parameter divergence: tightens sync
    /// cadence when replicas drift apart. Does NOT loosen; ElChe's overhead
    /// auto-tune handles upward adjustment.
    ///
    /// Bypasses `min_anchor` (clamped to 1) because divergence is a stronger
    /// signal than the overhead floor. The overhead auto-tune will recover
    /// the anchor upward once divergence subsides.
    pub fn nudge_anchor_down(&mut self, factor: f64) {
        // clamp() propagates NaN; a NaN factor (user-configured guard
        // math gone wrong) would collapse the anchor to 1 silently.
        if !factor.is_finite() {
            return;
        }
        let new = (self.anchor as f64 * factor.clamp(0.1, 1.0)).ceil() as usize;
        self.anchor = new.max(1).min(self.anchor);
        let slow_ms = self.slow_ms();
        if slow_ms > 0.0 {
            self.recompute_batch_counts(slow_ms);
        }
    }

    /// Relax the anchor upward by 1 batch on stable convergence.
    ///
    /// Symmetric upward path to [`Self::nudge_anchor_down`]: lets async-mode anchor
    /// drift toward `max_anchor` over time as long as the convergence guard
    /// reports `Stable`, amortizing AllReduce barrier cost over more local
    /// SGD steps. Pairs with the downward `NudgeDown` path so the control
    /// loop has both directions.
    ///
    /// Honors the user-defined `max_batch_diff` cap when set: refuses to
    /// relax if the projected per-rank batch_counts spread at `anchor + 1`
    /// would exceed `max_batch_diff`. With ratio R between fastest and
    /// slowest rank and cap M, anchor is bounded by `M / (R - 1)` — e.g.
    /// for ratio 3 and `max_batch_diff = 100`, anchor caps at 50 (yielding
    /// `[50, 150]`, diff exactly 100).
    ///
    /// No-op when already at `max_anchor`, or when no calibrated
    /// `ms_per_batch` exists yet (Probe phase).
    pub fn relax_anchor_up(&mut self) {
        if self.anchor >= self.max_anchor {
            return;
        }
        // Honor user-defined drift cap.
        if let Some(max_diff) = self.max_batch_diff {
            let smoothed: Vec<f64> = (0..self.world_size).map(|r| self.smoothed_ms(r)).collect();
            let max_ms = smoothed.iter().copied().fold(0.0_f64, f64::max);
            let min_ms = smoothed
                .iter()
                .copied()
                .filter(|&m| m > 0.0)
                .fold(f64::MAX, f64::min);
            if max_ms > 0.0 && min_ms.is_finite() && min_ms > 0.0 {
                let new_anchor = self.anchor + 1;
                let projected_fast =
                    (new_anchor as f64 * max_ms / min_ms).round().max(1.0) as usize;
                if projected_fast.saturating_sub(new_anchor) > max_diff {
                    return;
                }
            }
        }
        self.anchor += 1;
        let slow_ms = self.slow_ms();
        if slow_ms > 0.0 {
            self.recompute_batch_counts(slow_ms);
        }
    }

    /// Commit any pending window-pressure grow proposal from the last
    /// `report_timing`. Called on `ConvergenceAction::Stable` — the guard
    /// saw no divergence concern, so the grow applies.
    ///
    /// Also advances the growth-enable latch: each `Stable` verdict counts
    /// toward re-arming growth (`GROWTH_REARM_STABLE` consecutive clean
    /// verdicts), so growth that was latched off by a prior `SuppressGrowth`
    /// / `NudgeDown` only resumes once convergence is robustly clean again.
    ///
    /// Pairs with [`Self::veto_proposed_growth`] and
    /// [`Self::discard_proposed_anchor`] to make the convergence guard
    /// authoritative over `overhead_target`. No-op on the anchor when no
    /// proposal is pending (Probe/Warmup phase, overhead below target, or no
    /// `report_timing` call between guard verdicts).
    pub fn commit_proposed_anchor(&mut self) {
        self.consecutive_stable = self.consecutive_stable.saturating_add(1);
        if self.consecutive_stable >= GROWTH_REARM_STABLE {
            self.growth_enabled = true;
        }
        if let Some(ProposedAnchor::Grow(n)) = self.proposed_anchor.take() {
            self.anchor = n;
            let slow_ms = self.slow_ms();
            if slow_ms > 0.0 {
                self.recompute_batch_counts(slow_ms);
            }
        }
    }

    /// Drop the pending grow proposal and latch growth OFF. Called on
    /// `ConvergenceAction::SuppressGrowth` — the guard saw divergence
    /// trending up, so growth would make it worse. Growth stays disabled
    /// until `GROWTH_REARM_STABLE` consecutive `Stable` verdicts re-arm it
    /// (the margin to the cliff: don't poke the boundary again until
    /// convergence is robustly clean).
    pub fn veto_proposed_growth(&mut self) {
        self.proposed_anchor = None;
        self.consecutive_stable = 0;
        self.growth_enabled = false;
    }

    /// Drop any pending grow proposal and latch growth OFF. Called on
    /// `ConvergenceAction::NudgeDown` — the nudge ([`Self::nudge_anchor_down`])
    /// shrinks the anchor directly; growth is disabled and re-arms only
    /// after `GROWTH_REARM_STABLE` consecutive `Stable` verdicts.
    pub fn discard_proposed_anchor(&mut self) {
        self.proposed_anchor = None;
        self.consecutive_stable = 0;
        self.growth_enabled = false;
    }

    /// Apply a source-agnostic [`AnchorVerdict`] — the ONE verdict seam.
    ///
    /// Every verdict producer (convergence guard, meta-controller,
    /// future detectors) funnels through here; ElChe never learns the
    /// source. The fine-grained anchor methods
    /// ([`Self::commit_proposed_anchor`], [`Self::veto_proposed_growth`],
    /// [`Self::discard_proposed_anchor`], [`Self::nudge_anchor_down`],
    /// [`Self::relax_anchor_up`]) remain available as building blocks,
    /// but orchestration code should speak verdicts.
    ///
    /// `NudgeDown` both discards the pending grow proposal AND nudges:
    /// a proposal staged this cycle was computed from the pre-nudge
    /// anchor, so a later `Stable` verdict committing it would silently
    /// overwrite the nudge (the discard-then-nudge order is canonical;
    /// the two operations touch disjoint state, so producers that
    /// historically nudged first are unaffected).
    pub fn apply_verdict(&mut self, verdict: AnchorVerdict) {
        match verdict {
            AnchorVerdict::Stable { relax_up } => {
                self.commit_proposed_anchor();
                if relax_up {
                    self.relax_anchor_up();
                }
            }
            AnchorVerdict::SuppressGrowth => {
                self.veto_proposed_growth();
            }
            AnchorVerdict::NudgeDown { factor } => {
                self.discard_proposed_anchor();
                self.nudge_anchor_down(factor);
            }
        }
    }

    /// Whether at least one timing measurement has been reported.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Whether a speed hint was applied (batch_counts are non-uniform).
    ///
    /// Used by the coordinator to decide if epoch 0 should use
    /// throughput-proportional partitions before calibration.
    pub fn has_speed_hint(&self) -> bool {
        self.batch_counts.windows(2).any(|w| w[0] != w[1])
    }

    /// Per-device smoothed milliseconds per batch (mean over trust window).
    /// Returns a fresh `Vec` rather than a slice because the smoothed values
    /// are computed from internal ring buffers; callers store or iterate the
    /// vec directly.
    pub fn ms_per_batch(&self) -> Vec<f64> {
        (0..self.world_size).map(|r| self.smoothed_ms(r)).collect()
    }

    /// Per-rank smoothed ms-per-batch as an `Option<f64>` — `None` when
    /// the trust window for that rank is empty (no positive reading
    /// has landed yet), `Some(ms)` when a calibrated value is
    /// available. Distinguishes "uncalibrated" from a legitimate
    /// `0.0` better than [`Self::ms_per_batch`], which collapses both
    /// into 0.0. Used by `ClusterCoordinator::resolve_fastest_role`
    /// to pick the live rank with the lowest calibrated ms-per-batch
    /// (Fastest policy).
    pub fn smoothed_ms_per_batch(&self, rank: usize) -> Option<f64> {
        self.ms_per_batch_window
            .get(rank)
            .filter(|w| !w.is_empty())
            .map(|w| w.mean())
    }

    /// Ingest one reduce window's observations — the event-shaped feed.
    ///
    /// Stages the window-pressure fill, selects the timing scale via
    /// [`WindowReport::select_feed`] (the mixed-scale inversion guard),
    /// and feeds [`Self::report_timing`] when the window carries any
    /// signal (an all-zero feed — e.g. a fully-idle window — reports
    /// nothing, so no spurious zero-ms sample poisons the trust
    /// windows). This is the coordinator's one timing entry point; the
    /// lower-level [`Self::set_window_fill_ms`] + [`Self::report_timing`]
    /// pair remains for callers that assemble their own feed.
    pub fn report_window(&mut self, report: &WindowReport) {
        self.set_window_fill_ms(&report.fill_ms);
        let (ms, batches) = report.select_feed();
        if ms.iter().any(|&m| m > 0.0) {
            self.report_timing(&ms, &batches, report.sync_ms);
        }
    }

    /// Report timing after a cadence step completes.
    ///
    /// `wall_ms[rank]`: wall-clock time for all batches on that device (ms).
    /// `actual_batches[rank]`: number of batches each rank actually processed
    /// since the last sync (i.e., `steps_since_avg`). In Cadence mode the fast
    /// GPU may process more batches than its intended `batch_counts` while
    /// waiting for the slow GPU to reach the trigger threshold. Using the
    /// intended count as divisor would inflate the fast GPU's ms_per_batch,
    /// inverting the throughput ratio.
    /// `sync_ms`: AllReduce overhead for this step (ms).
    ///
    /// Updates batch ratios based on measured throughput. If AllReduce
    /// overhead exceeds the target, anchor auto-tunes upward.
    pub fn report_timing(&mut self, wall_ms: &[f64], actual_batches: &[usize], sync_ms: f64) {
        assert_eq!(
            wall_ms.len(),
            self.world_size,
            "wall_ms length must match world_size",
        );
        assert_eq!(
            actual_batches.len(),
            self.world_size,
            "actual_batches length must match world_size",
        );

        // Push each rank's reading into its trust window. A zero/invalid
        // wall_ms increments the death-exception counter; if a rank misses
        // `TRUST_WINDOW_CAP` reports in a row, its window is cleared so
        // smoothing reacts fast on recovery.
        for (rank, &wall) in wall_ms.iter().enumerate() {
            let n = actual_batches.get(rank).copied().unwrap_or(0);
            if n > 0 && wall > 0.0 && wall.is_finite() {
                let new_ms = wall / n as f64;
                self.ms_per_batch_window[rank].push(new_ms);
                self.consecutive_zero_reports[rank] = 0;
            } else {
                self.consecutive_zero_reports[rank] += 1;
                if self.consecutive_zero_reports[rank] >= TRUST_WINDOW_CAP {
                    self.ms_per_batch_window[rank].clear();
                }
            }
        }

        // Election is allowed in two cases: to set the initial pick when
        // none exists, and during steady state. Probe + Warmup hold the
        // pin (from spec prior, partition_ratios, with_speed_ratio, or the
        // rank-0 fallback set on the first report) so cold-start noise on
        // the larger/newer GPU can't transiently flip the anchor away from
        // the actual slow rank. The cohort filter inside `elect_anchor`
        // then does the load-bearing work of excluding clearly-fast ranks.
        let allow_election = self.anchor_rank.is_none() || self.phase >= Phase::Stable;
        if allow_election && let Some(elected) = self.elect_anchor() {
            self.anchor_rank = Some(elected);
        }

        // No anchor yet (no positive readings on any rank) — bail.
        let anchor_rank = match self.anchor_rank {
            Some(r) => r,
            None => return,
        };
        let mut slow_ms = self.smoothed_ms(anchor_rank);
        if slow_ms <= 0.0 {
            // PINNED-ANCHOR DEAD-WINDOW ESCAPE. A pinned anchor (spec
            // prior / partition_ratios / with_initial_anchor) that never
            // produces a valid reading would otherwise freeze the
            // controller in Warmup forever: this bail runs BEFORE
            // `calibration_count += 1`, and re-election requires
            // `Phase::Stable`, which requires calibration. Once the rank
            // has missed a full trust window of reports, stop trusting
            // the pin and elect from the ranks that DO have data.
            if self.consecutive_zero_reports[anchor_rank] >= TRUST_WINDOW_CAP
                && let Some(elected) = self.elect_anchor()
            {
                self.anchor_rank = Some(elected);
                slow_ms = self.smoothed_ms(elected);
            }
            if slow_ms <= 0.0 {
                return;
            }
        }

        // Window-pressure auto-tune: propose anchor growth from the
        // bottleneck rank's per-window fixed overhead (reduce + fill). The
        // proposal is NOT applied here — the caller's convergence-guard
        // verdict drives one of [`Self::commit_proposed_anchor`] (Stable),
        // [`Self::veto_proposed_growth`] (SuppressGrowth — drops the grow),
        // or [`Self::discard_proposed_anchor`] (NudgeDown — drops the grow;
        // the nudge then shrinks directly). Makes the convergence guard
        // authoritative over `overhead_target`: rising divergence vetoes
        // growth before it lands, and latches growth off until convergence
        // is robustly clean again.
        //
        // Gated on the SECOND calibration rather than `Phase::Stable`. The
        // noise hazard this gate used to own — the historical pre-cap 10→22
        // single-shot jump on one sparse reading — is owned by
        // [`GROWTH_STEP_CAP`] now (×2/cycle at most), every step still needs
        // the convergence guard's blessing to land, and the Shrink
        // hysteresis unwinds an overshoot. Election keeps the full `Stable`
        // gate (cold-start skew can crown the wrong anchor; see
        // `allow_election` above) — growth is different in kind: its signal
        // is dominated by the measured reduce+fill wall, orders of magnitude
        // above per-batch noise whenever it matters, and cold-start skew
        // inflates the anchor's marginal, which UNDER-proposes — the error
        // direction is the safe one. Holding growth through the full trust
        // window instead cost one full reduce per window across every
        // cluster run's first five windows: the exact overhead this term
        // exists to amortize, paid at the run phase where windows are
        // smallest and the ratio is worst (measured 78% overhead ×5 windows
        // on the 3-rank OLMo rig, 2026-07-28). Two disciplines keep the
        // early path honest: the first report never proposes (its sync_ms
        // can carry cohort-formation cost, and one sample is a reading, not
        // a trend), and a warmup proposal must clear the target by
        // [`WARMUP_GROWTH_MARGIN`] — borderline pressure waits for the full
        // trust window exactly as before.
        self.proposed_anchor = None;
        if self.phase >= Phase::Stable
            || (self.phase == Phase::Warmup && self.calibration_count >= 1)
        {
            self.proposed_anchor = self.propose_anchor(sync_ms);
        }
        // The fill is a one-window signal: consume it so a stale value can't
        // drive a later cycle (mirrors the callback-slack one-shot).
        for f in &mut self.pending_window_fill_ms {
            *f = 0.0;
        }

        // Recompute batch counts from current (pre-proposal) anchor. The
        // commit/veto path recomputes again if the proposal lands.
        self.recompute_batch_counts(slow_ms);
        // Snapshot the post-recompute batch_counts so `recent_batch_share`
        // can report the cadence's actual per-rank allocation as a smoothed
        // observation. Cap is FIFO, oldest dropped on overflow.
        if self.batch_counts_window.len() >= BATCH_COUNTS_WINDOW_CAP {
            self.batch_counts_window.pop_front();
        }
        self.batch_counts_window
            .push_back(self.batch_counts.clone());
        self.calibrated = true;
        self.calibration_count += 1;
        crate::verbose!(
            "  ddp-diag: ms_per_batch={:?} batch_counts={:?} anchor_rank={:?} anchor={}",
            self.ms_per_batch()
                .iter()
                .map(|m| (m * 10.0).round() / 10.0)
                .collect::<Vec<_>>(),
            self.batch_counts,
            self.anchor_rank,
            self.anchor,
        );
        self.advance_phase();
    }

    /// Smoothed per-rank batch share as an observation of recent cadence.
    ///
    /// Averages the last `BATCH_COUNTS_WINDOW_CAP` `batch_counts` snapshots
    /// (each captured at the end of `report_timing`) and normalizes to sum
    /// to 1.0. This is the metric source for per-epoch `share` reporting:
    /// it answers "what fraction of work did the balancer assign each rank,
    /// recently?" rather than "what fraction of samples did each rank
    /// happen to consume?" — those agree in steady state but diverge under
    /// progressive dispatch's tail-balance equalization near epoch end.
    ///
    /// Falls back to the current `batch_counts` ratio when no snapshots
    /// have been captured yet (no `report_timing` calls), and to equal
    /// shares if both are degenerate.
    pub fn recent_batch_share(&self) -> Vec<f64> {
        if self.batch_counts_window.is_empty() {
            let total: usize = self.batch_counts.iter().sum();
            if total == 0 {
                return vec![1.0 / self.world_size as f64; self.world_size];
            }
            return self
                .batch_counts
                .iter()
                .map(|&c| c as f64 / total as f64)
                .collect();
        }
        let mut sums = vec![0.0_f64; self.world_size];
        let mut total = 0.0_f64;
        for snap in &self.batch_counts_window {
            for (r, &c) in snap.iter().enumerate() {
                sums[r] += c as f64;
                total += c as f64;
            }
        }
        if total <= 0.0 {
            return vec![1.0 / self.world_size as f64; self.world_size];
        }
        sums.into_iter().map(|s| s / total).collect()
    }

    /// Phase transition rules. Probe→Warmup at first calibration; Warmup→Stable
    /// at 5 calibrations; Stable→Mature at 20. Inside ElChe, phases gate anchor
    /// election, window growth and the auto-tune's fire threshold; the LR-aware
    /// meta-controller additionally reads the phase for per-phase parameter
    /// tightening (nudge factor and sustain count).
    fn advance_phase(&mut self) {
        let next = match self.phase {
            Phase::Probe => Phase::Warmup,
            Phase::Warmup if self.calibration_count >= 5 => Phase::Stable,
            Phase::Stable if self.calibration_count >= 20 => Phase::Mature,
            p => p,
        };
        if next != self.phase {
            crate::verbose!(
                "  ddp: ElChe phase {:?} -> {:?} (calibration #{}, anchor=rank {})",
                self.phase,
                next,
                self.calibration_count,
                self.anchor_rank.map(|r| r as i64).unwrap_or(-1),
            );
            self.phase = next;
        }
    }

    /// Clamp batch counts to a maximum total, preserving proportions.
    ///
    /// Returns a new batch-count vector. Use near epoch boundaries to
    /// avoid consuming more batches than remain.
    pub fn clamp_total(&self, max_total: usize) -> Vec<usize> {
        let current_total = self.total_batches();
        if current_total <= max_total {
            return self.batch_counts.clone();
        }
        let scale = max_total as f64 / current_total as f64;
        let mut clamped: Vec<usize> = self
            .batch_counts
            .iter()
            .map(|&n| (n as f64 * scale).floor().max(1.0) as usize)
            .collect();
        // Distribute remainder to stay exactly at max_total.
        let sum: usize = clamped.iter().sum();
        let mut remainder = max_total.saturating_sub(sum);
        for c in &mut clamped {
            if remainder == 0 {
                break;
            }
            *c += 1;
            remainder -= 1;
        }
        clamped
    }

    /// Window-pressure grow proposal (pure decision logic, extracted so the
    /// math is unit-testable in isolation; nothing here can trigger a
    /// reduce). Grow-only — see [`ProposedAnchor`].
    ///
    /// The signal is the bottleneck (anchor) rank's per-window FIXED
    /// overhead as a fraction of its window wall:
    ///
    /// ```text
    /// overhead = (reduce_ms + fill_b) / (anchor·marginal_b + reduce_ms + fill_b)
    /// ```
    ///
    /// where `marginal_b` is the anchor rank's steady-state ms/batch and
    /// `fill_b` its first-batch fill (`pending_window_fill_ms`). Both
    /// `reduce_ms` and `fill_b` are fixed per-window costs, so the fraction
    /// falls as the window (`anchor·marginal_b`) grows — a diminishing-
    /// returns signal that settles at the knee where amortization is spent,
    /// NOT at an absolute utilization target (the achievable utilization
    /// ceiling is rig-dependent — heterogeneity / granularity — and an
    /// absolute target would chase an unreachable floor). The anchor rank is
    /// used because it never waits on allocation imbalance, so its residual
    /// overhead is purely the amortizable per-window cost, cleanly separated
    /// from ElChe's share-allocation job.
    ///
    /// This replaces the prior `reduce_ms / compute` rule, which saw only
    /// the reduce: cheap on NCCL it never tripped (anchor stalled tiny);
    /// expensive on CPU it over-grew. Folding the (backend-independent) fill
    /// into the numerator gives a consistent operating point on both.
    ///
    /// Suppressed when growth is latched off (the guard saw rising
    /// divergence — see [`Self::growth_enabled`]) or the window≤epoch cap is
    /// binding (a larger anchor delivers nothing at the cap and only
    /// detaches the anchor from the schedule).
    fn propose_anchor(&self, sync_ms: f64) -> Option<ProposedAnchor> {
        if !self.window_growth_applicable {
            // Sync policy: reduces fire every step, so there is no window to
            // amortize. Never propose growth — it would only pollute the
            // telemetry anchor and the checkpointed ElCheState.
            return None;
        }
        if !self.growth_enabled {
            return None;
        }
        if self.window_cap_binding {
            crate::verbose!(
                "  ddp: window-pressure growth suppressed — window cap binding \
                 (epoch is the ceiling at this size)"
            );
            return None;
        }
        let b = self.anchor_rank?;
        let marginal_b = self.smoothed_ms(b);
        if marginal_b <= 0.0 {
            return None;
        }
        let fill_b = self
            .pending_window_fill_ms
            .get(b)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let reduce_ms = sync_ms.max(0.0);
        let window_compute = self.anchor as f64 * marginal_b;
        let fixed = reduce_ms + fill_b;
        let denom = window_compute + fixed;
        if denom <= 0.0 || fixed <= 0.0 {
            return None;
        }
        let overhead = fixed / denom;
        // Before `Stable`, only an unambiguous signal may act (see
        // [`WARMUP_GROWTH_MARGIN`]); the scale below still divides by the
        // real target, so a warmup proposal that clears the margin is
        // cap-clamped by construction.
        let fire_at = if self.phase >= Phase::Stable {
            self.overhead_target
        } else {
            self.overhead_target * WARMUP_GROWTH_MARGIN
        };
        if overhead <= fire_at {
            return None;
        }
        let scale = (overhead / self.overhead_target).min(GROWTH_STEP_CAP);
        let new_anchor = (self.anchor as f64 * scale).ceil() as usize;
        let clamped = new_anchor.clamp(self.min_anchor, self.max_anchor);
        if clamped > self.anchor {
            return Some(ProposedAnchor::Grow(clamped));
        }
        None
    }

    /// Recompute batch counts: slow device gets `anchor`, faster devices
    /// get proportionally more based on their ms_per_batch.
    ///
    /// Applies a dead zone: a rank's count only changes when the new value
    /// differs from the current by more than 5%. Trust-window smoothing
    /// already filters per-call noise, so the dead zone only needs to
    /// suppress 1-batch chatter; sized at 5% to capture genuine
    /// within-cohort speed differences (e.g. two near-identical 1060s
    /// where one runs ~7-8% slower) that a 10% gate would mask.
    fn recompute_batch_counts(&mut self, slow_ms: f64) {
        for rank in 0..self.world_size {
            let ms = self.smoothed_ms(rank);
            let target_no_slack = if ms <= 0.0 || (ms - slow_ms).abs() < 1e-6 {
                self.anchor
            } else {
                let ratio = (slow_ms / ms).min(MAX_SPEED_RATIO);
                (self.anchor as f64 * ratio).round().max(1.0) as usize
            };

            // Callback slack: when the coord staged ms of wall-time for
            // this rank to absorb (eval_fn / epoch_fn / checkpoint_fn
            // firing on this rank for the upcoming cycle), subtract the
            // equivalent batch count so the rank finishes its quota
            // early and runs the callback in the freed slack instead of
            // bloating the sync-barrier wait. Skipped when slack is
            // zero (the typical case) or when the rank has no
            // calibrated ms-per-batch reading. Clamps target at 1 so
            // even a heavy callback doesn't starve the rank of training
            // work.
            let slack_ms = self.pending_callback_slack_ms[rank];
            let slack_batches = if slack_ms > 0.0 && ms > 0.0 {
                (slack_ms / ms).ceil() as usize
            } else {
                0
            };
            let target = target_no_slack.saturating_sub(slack_batches).max(1);

            let current = self.batch_counts[rank];
            let diff = (target as f64 - current as f64).abs();
            // Dead zone: only update if change exceeds 5% of current count.
            // Always update on first calibration (current == anchor for all).
            // Slack-driven changes bypass the dead zone — the slack vector
            // is set explicitly per-cycle by the coord, so silently
            // ignoring it because the delta is small would defeat the
            // point.
            let slack_active = slack_batches > 0;
            if diff > current as f64 * 0.05 || !self.calibrated || slack_active {
                // Clamp per-update change to max_batch_diff (if set).
                // Without this, a sudden speed change (thermal throttle, power
                // limit) can cause the batch count to jump far beyond the
                // intended limit in a single update, and the reactive throttle
                // in check_throttle() only catches it one tick later.
                let clamped = match self.max_batch_diff {
                    Some(max) if self.calibrated => {
                        if target > current {
                            current.saturating_add(max).min(target)
                        } else {
                            current.saturating_sub(max).max(target).max(1)
                        }
                    }
                    _ => target,
                };
                self.batch_counts[rank] = clamped;
            }
        }
        // Window cap: a reduce window must fit within one epoch. The
        // overhead auto-tune can grow the per-rank counts to amortize an
        // expensive sync, but if their sum exceeds the epoch's batch
        // count the window would span multiple dataset passes — syncs
        // collapse to <1/epoch and the schedule no longer "knows how many
        // steps per epoch". Scale the counts down proportionally (keeping
        // the speed-derived ratio) so `sum(batch_counts) <= max_total`.
        // No-op when unset or already within bound.
        //
        // DEGENERATE CASE — the per-rank floor of 1 deliberately WINS over
        // the cap: when `world_size > max_total` (more ranks than batches
        // per epoch) the floored sum still exceeds `max_total`. This cannot
        // alter training results: these counts are RATIOS the dispatcher
        // apportions, not a contract — `final_window_alloc`
        // (epoch_dispatch.rs) allocates exactly `pool.remaining()` by
        // largest-remainder whenever the pool holds less than the schedule
        // claims (which in this degenerate case is every window), so some
        // ranks legitimately receive 0. A 0-step rank ships a TRUE-count,
        // weight-0 frame that the consensus excludes (see
        // `snapshot_params`'s batch_count contract), and the reduce divides
        // by accepted mass, never by claims. Keeping the floor here keeps
        // every rank's quota alive for the next window's rebalance.
        self.window_cap_binding = false;
        if let Some(max_total) = self.max_total_batches {
            let total: usize = self.batch_counts.iter().sum();
            if total > max_total && max_total > 0 {
                self.window_cap_binding = true;
                let scale = max_total as f64 / total as f64;
                for c in &mut self.batch_counts {
                    *c = ((*c as f64) * scale).floor().max(1.0) as usize;
                }
                // Hand any rounding remainder to the fastest rank (largest
                // count) so the cohort still uses the full window budget.
                let used: usize = self.batch_counts.iter().sum();
                if let Some(rem) = max_total.checked_sub(used)
                    && rem > 0
                    && let Some(fastest) =
                        (0..self.world_size).max_by_key(|&r| self.batch_counts[r])
                {
                    self.batch_counts[fastest] += rem;
                }
            }
        }
        // Slack is consumed exactly once per recompute. Zeroing here
        // (rather than letting the caller manage) avoids the
        // double-application bug where two back-to-back recomputes both
        // subtract the same slack.
        for s in &mut self.pending_callback_slack_ms {
            *s = 0.0;
        }
    }
}

/// Cold-start anchor selection from device hardware specs.
///
/// Combines arch generation (`sm_86` → 86, `gfx1030` → 1030) and total
/// VRAM in GB into a single ordinal score per rank. Higher score = better
/// spec = faster GPU (likely). The slowest rank by score is the cold-start
/// anchor pick. Generation dominates; VRAM tiebreaks within the same arch
/// generation. Single-vendor cohorts only, by construction.
///
/// Returns `None` if any device-property query fails — the caller falls
/// back to the rank-0 default (or whatever the existing logic produces).
mod spec_prior {
    /// Ordinal "spec score" for a CUDA device. Higher = better spec.
    /// Returns `None` when device-property queries fail (e.g. CUDA absent).
    ///
    /// Uses [`crate::sys::detect_gpus`] (an out-of-process vendor probe:
    /// `nvidia-smi`, or the KFD topology on ROCm) instead of
    /// libtorch's `cuda_compute_capability` / `gpu_memory_info_idx` so
    /// this can run on the controller's main thread without violating
    /// the "no CUDA touch before fan-out" invariant.
    fn score(device_index: i32, gpus: &[crate::sys::GpuInfo]) -> Option<f64> {
        let gpu = gpus.iter().find(|g| g.index as i32 == device_index)?;
        let vram_gb = gpu.vram_bytes() as f64 / 1_073_741_824.0;
        // `generation()` orders hardware within a vendor (sm_86 -> 86,
        // gfx1030 -> 1030). The ×10 keeps VRAM as a tiebreak inside one
        // generation rather than letting a big-VRAM older card outrank a
        // newer one. Callers guarantee a single vendor; see below.
        Some((gpu.arch.generation() as f64) * 10.0 + vram_gb)
    }

    /// Rank with the lowest spec score across `device_indices`. Lowest-rank
    /// tiebreak when two ranks score equal. Returns `None` when any device
    /// query fails — caller falls back to current behavior.
    ///
    /// Also returns `None` for a **mixed-vendor** cohort. Arch generation
    /// numbers are only meaningful within a vendor (`GpuArch::generation`
    /// says so), so ranking an NVIDIA card against an AMD one by this
    /// score would be arbitrary, and arbitrary is worse than absent here:
    /// the caller's fallback is a defensible default, whereas a wrong
    /// anchor pick makes every rank wait on it.
    pub(super) fn slowest_rank(device_indices: &[i32]) -> Option<usize> {
        let gpus = crate::sys::detect_gpus();
        let cohort: Vec<&crate::sys::GpuInfo> = device_indices
            .iter()
            .filter_map(|&idx| gpus.iter().find(|g| g.index as i32 == idx))
            .collect();
        if cohort.windows(2).any(|w| w[0].vendor != w[1].vendor) {
            return None;
        }
        let scores: Option<Vec<(usize, f64)>> = device_indices
            .iter()
            .enumerate()
            .map(|(rank, &idx)| score(idx, &gpus).map(|s| (rank, s)))
            .collect();
        let scores = scores?;
        scores
            .into_iter()
            .min_by(|(ra, a), (rb, b)| {
                a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(ra.cmp(rb))
            })
            .map(|(rank, _)| rank)
    }
}

#[cfg(test)]
mod meta_nudge_tests {
    use super::*;

    // H12 regression: when the meta-controller nudges the anchor down and
    // a grow proposal was staged this cycle (from the PRE-nudge anchor),
    // the nudge must survive a subsequent guard-`Stable`
    // `commit_proposed_anchor`. The meta path (`observe_meta`) achieves
    // this by calling `discard_proposed_anchor` right after the nudge —
    // exactly what the guard's own NudgeDown branch does.
    #[test]
    fn nudge_then_discard_survives_a_later_commit() {
        let mut el = ElChe::new(2, 10);
        // Stage a grow proposal as report_timing would (private access).
        el.proposed_anchor = Some(ProposedAnchor::Grow(20));
        el.nudge_anchor_down(0.5); // meta LR-cliff nudge: 10 -> 5
        assert_eq!(el.anchor(), 5);
        el.discard_proposed_anchor(); // meta path invalidates the stale grow
        el.commit_proposed_anchor(); // guard verdict Stable, later in the cycle
        assert_eq!(
            el.anchor(),
            5,
            "meta nudge must survive: discard drops the pre-nudge grow so \
             commit has nothing to apply"
        );
    }

    // The mirror image — this is the BUG the fix removes. Without the
    // discard, commit applies the pre-nudge grow target and clobbers the
    // nudge. Kept as executable documentation of why the discard is
    // load-bearing (NOT the desired behavior).
    #[test]
    fn nudge_without_discard_is_clobbered_by_commit() {
        let mut el = ElChe::new(2, 10);
        el.proposed_anchor = Some(ProposedAnchor::Grow(20));
        el.nudge_anchor_down(0.5); // 10 -> 5
        assert_eq!(el.anchor(), 5);
        el.commit_proposed_anchor(); // no discard: the grow wins
        assert_eq!(
            el.anchor(),
            20,
            "documents the H12 bug: an un-discarded grow overwrites the nudge"
        );
    }
}

#[cfg(test)]
mod verdict_seam_tests {
    use super::*;

    // The seam must reproduce the guard branch exactly: NudgeDown both
    // discards the staged (pre-nudge) grow proposal and nudges, so a
    // later Stable commit has nothing stale to apply (H12 through the
    // seam instead of the fine-grained methods).
    #[test]
    fn verdict_nudge_down_discards_and_nudges() {
        let mut el = ElChe::new(2, 10);
        el.proposed_anchor = Some(ProposedAnchor::Grow(20));
        el.apply_verdict(AnchorVerdict::NudgeDown { factor: 0.5 });
        assert_eq!(el.anchor(), 5);
        el.apply_verdict(AnchorVerdict::Stable { relax_up: false });
        assert_eq!(
            el.anchor(),
            5,
            "the verdict's built-in discard must survive a later Stable commit"
        );
        assert!(!el.growth_enabled(), "NudgeDown latches growth off");
    }

    #[test]
    fn verdict_stable_commits_pending_grow() {
        let mut el = ElChe::new(2, 10);
        el.proposed_anchor = Some(ProposedAnchor::Grow(20));
        el.apply_verdict(AnchorVerdict::Stable { relax_up: false });
        assert_eq!(el.anchor(), 20);
    }

    #[test]
    fn verdict_stable_relax_up_drifts_anchor() {
        let mut el = ElChe::new(2, 10);
        el.apply_verdict(AnchorVerdict::Stable { relax_up: true });
        assert_eq!(el.anchor(), 11, "relax_up drifts +1 toward max_anchor");
        let mut el = ElChe::new(2, 10);
        el.apply_verdict(AnchorVerdict::Stable { relax_up: false });
        assert_eq!(el.anchor(), 10, "without relax_up the anchor holds");
    }

    #[test]
    fn verdict_suppress_growth_drops_proposal_and_latches() {
        let mut el = ElChe::new(2, 10);
        el.proposed_anchor = Some(ProposedAnchor::Grow(20));
        el.apply_verdict(AnchorVerdict::SuppressGrowth);
        assert_eq!(el.anchor(), 10, "SuppressGrowth holds the anchor");
        assert!(!el.growth_enabled());
        el.apply_verdict(AnchorVerdict::Stable { relax_up: false });
        assert_eq!(el.anchor(), 10, "the vetoed proposal is gone for good");
    }
}

#[cfg(test)]
mod report_window_tests {
    use super::*;

    fn report(
        wall: &[f64],
        steps: &[usize],
        dms: &[f64],
        dbatches: &[usize],
        coherent: bool,
    ) -> WindowReport {
        WindowReport {
            wall_ms: wall.to_vec(),
            steps: steps.to_vec(),
            delivered_ms: dms.to_vec(),
            delivered_batches: dbatches.to_vec(),
            fill_ms: vec![0.0; wall.len()],
            delivered_coherent: coherent,
            sync_ms: 1.0,
        }
    }

    #[test]
    fn coherent_report_feeds_the_delivered_scale() {
        let mut el = ElChe::new(2, 4);
        el.report_window(&report(
            &[40.0, 100.0],
            &[4, 4],
            &[80.0, 220.0],
            &[4, 4],
            true,
        ));
        assert!(el.is_calibrated());
        // 80/4 = 20 (delivered), not 40/4 = 10 (compute).
        assert!((el.smoothed_ms_per_batch(0).unwrap() - 20.0).abs() < 1e-9);
        assert!((el.smoothed_ms_per_batch(1).unwrap() - 55.0).abs() < 1e-9);
    }

    #[test]
    fn incoherent_report_feeds_the_compute_scale() {
        let mut el = ElChe::new(2, 4);
        el.report_window(&report(
            &[40.0, 100.0],
            &[4, 4],
            &[80.0, 0.0],
            &[4, 0],
            false,
        ));
        assert!((el.smoothed_ms_per_batch(0).unwrap() - 10.0).abs() < 1e-9);
        assert!((el.smoothed_ms_per_batch(1).unwrap() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn all_zero_window_reports_nothing() {
        let mut el = ElChe::new(2, 4);
        el.report_window(&report(&[0.0, 0.0], &[0, 0], &[0.0, 0.0], &[0, 0], true));
        assert!(
            !el.is_calibrated(),
            "a fully-idle window must not poison the trust windows"
        );
    }
}

#[cfg(test)]
mod window_growth_policy_tests {
    use super::*;

    // M2: window-pressure anchor growth must be exempt under Sync. Sync
    // reduces every step (no window to amortize), so growth there only
    // pollutes the telemetry anchor and the checkpointed ElCheState.anchor —
    // mis-seeding a later Cadence resume. Build a state that WOULD grow under a
    // windowed policy, then assert the Sync flag suppresses it.
    #[test]
    fn sync_policy_exempt_from_window_pressure_growth() {
        let mut el = ElChe::new(2, 10);
        el.anchor_rank = Some(0);
        // smoothed_ms(0) = 1.0 → window_compute = anchor(10) * 1 = 10.
        el.ms_per_batch_window[0].push(1.0);
        // reduce_ms = 10 → fixed = 10, overhead = 10/(10+10) = 0.5 ≫ 0.05 target.

        // Windowed policy (default applicable): growth fires.
        assert!(
            matches!(el.propose_anchor(10.0), Some(ProposedAnchor::Grow(n)) if n > 10),
            "windowed policy should propose growth in this high-overhead state"
        );

        // Sync exemption: the identical state proposes nothing.
        let el_sync = el.with_window_growth_applicable(false);
        assert!(
            el_sync.propose_anchor(10.0).is_none(),
            "Sync must never propose window-pressure growth"
        );
    }

    // Growth proposals start at the SECOND calibration, not at `Stable`:
    // waiting the full trust window paid one whole reduce per window across
    // every run's first five windows — at exactly the run phase where the
    // window is smallest and the overhead ratio worst. GROWTH_STEP_CAP,
    // the guard verdict, and the Shrink hysteresis own the noise hazard the
    // Stable gate used to double-cover.
    #[test]
    fn warmup_proposes_growth_from_the_second_calibration() {
        let mut el = ElChe::new(2, 10);
        // Report #1: high overhead, but a first reading is not a trend —
        // its sync can carry cohort-formation cost. No proposal.
        let bc = el.batch_counts().to_vec();
        el.report_timing(&[1000.0, 1000.0], &bc, 5000.0);
        assert_eq!(el.phase, Phase::Warmup);
        assert!(
            el.proposed_anchor.is_none(),
            "the first report must never propose growth"
        );

        // Report #2: still deep in Warmup (Stable needs 5), overhead
        // 5000/(10·100 + 5000) = 0.83 ≫ target. The proposal fires now and
        // is capped at ×2 like any other cycle.
        let bc = el.batch_counts().to_vec();
        el.report_timing(&[1000.0, 1000.0], &bc, 5000.0);
        assert_eq!(el.phase, Phase::Warmup);
        assert!(
            matches!(el.proposed_anchor, Some(ProposedAnchor::Grow(20))),
            "second calibration must propose capped growth, got {:?}",
            el.proposed_anchor,
        );

        // And it still lands only through the guard-verdict pipeline.
        el.commit_proposed_anchor();
        assert_eq!(el.anchor(), 20);
    }

    // The guard stays authoritative during Warmup exactly as in Stable: a
    // veto drops the early proposal without touching the anchor.
    #[test]
    fn warmup_growth_still_dies_on_a_guard_veto() {
        let mut el = ElChe::new(2, 10);
        let bc = el.batch_counts().to_vec();
        el.report_timing(&[1000.0, 1000.0], &bc, 5000.0);
        let bc = el.batch_counts().to_vec();
        el.report_timing(&[1000.0, 1000.0], &bc, 5000.0);
        assert!(matches!(el.proposed_anchor, Some(ProposedAnchor::Grow(_))));

        el.veto_proposed_growth();
        assert_eq!(el.anchor(), 10, "a vetoed warmup proposal must not land");
    }

    // Borderline pressure keeps the old patience: an overhead above target
    // but under WARMUP_GROWTH_MARGIN × target must not act during Warmup —
    // that is the regime where jittery early readings could flip the sign,
    // and the trust-window wait is exactly the right answer there. The same
    // reading fires once Stable is reached.
    #[test]
    fn borderline_warmup_pressure_waits_for_stable() {
        let mut el = ElChe::new(2, 10).with_overhead_target(0.10);
        // window_compute = 10·100 = 1000ms; sync 150 → overhead
        // 150/1150 ≈ 0.13: above target 0.10, below the 0.20 warmup bar.
        // Proposals are evaluated BEFORE the calibration count advances the
        // phase, so all five warmup reports (and not just four) see the
        // warmup bar — the first Stable-bar evaluation is the sixth report,
        // exactly where the old `Stable`-only gate first proposed.
        for i in 0..5 {
            let bc = el.batch_counts().to_vec();
            el.report_timing(&[1000.0, 1000.0], &bc, 150.0);
            assert!(
                el.proposed_anchor.is_none(),
                "borderline pressure proposed during warmup (report #{})",
                i + 1,
            );
        }
        assert_eq!(el.phase, Phase::Stable);
        // Sixth report evaluates under the plain target: now it proposes.
        let bc = el.batch_counts().to_vec();
        el.report_timing(&[1000.0, 1000.0], &bc, 150.0);
        assert!(
            matches!(el.proposed_anchor, Some(ProposedAnchor::Grow(_))),
            "the same pressure must fire once Stable, got {:?}",
            el.proposed_anchor,
        );
    }
}
