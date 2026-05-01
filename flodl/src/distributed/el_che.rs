/// El Che: heterogeneous DDP cadence strategy.
///
/// The column marches at the slowest one's pace. The slow device
/// anchors the cadence (`anchor` batches per sync step), the fast
/// ones range ahead doing more work, and everyone rejoins at AllReduce.
/// No one waits, no one idles.
///
/// After each sync step, call [`report_timing`](ElChe::report_timing)
/// with measured wall times and AllReduce overhead. El Che refines
/// batch ratios and auto-tunes the anchor count to keep AllReduce overhead
/// below a configurable target (default 10%).
///
/// # Example
///
/// ```ignore
/// let ddp = Ddp::wrap(&[&model0, &model1], &devices)?;
/// let mut cadence = ElChe::new(2, 10);
///
/// loop {
///     let start_events = record_start_events(&devices)?;
///     for rank in 0..2 {
///         for _ in 0..cadence.batches(rank) {
///             forward_backward(rank)?;
///         }
///     }
///     let wall_ms = measure_elapsed(&start_events)?;
///
///     let sync_start = Instant::now();
///     ddp.weighted_all_reduce_gradients(cadence.batch_counts())?;
///     let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;
///
///     cadence.report_timing(&wall_ms, cadence.batch_counts(), sync_ms);
/// }
/// ```
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
        Self { samples: Vec::with_capacity(capacity), capacity }
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

    fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Lifecycle phase of the cadence balancer. Probe = no calibrations yet,
/// Warmup = first few calibrations (election allowed but anchor stays sticky
/// until `MIN_REPORTS_BEFORE_SWAP`), Stable = normal operation including
/// overhead auto-tune, Mature = long-running steady state. Phase ordering
/// is monotonic and supports `>=` comparisons for gating logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    /// Whether at least one real measurement has been taken.
    calibrated: bool,
    /// Target: max AllReduce overhead as fraction of compute time.
    overhead_target: f64,
    /// Minimum anchor (never below initial value).
    min_anchor: usize,
    /// Maximum anchor (gradient staleness limit).
    max_anchor: usize,
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
            calibrated: false,
            overhead_target: 0.10,
            min_anchor: anchor,
            max_anchor: 200,
            max_batch_diff: None,
            phase: Phase::Probe,
            anchor_rank: None,
            calibration_count: 0,
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
        if let Some(c) = self.anchor_rank {
            if cohort.contains(&c) {
                let cur = self.smoothed_ms(c);
                let challenger = cohort
                    .iter()
                    .copied()
                    .filter(|&r| r != c)
                    .map(|r| (r, self.smoothed_ms(r)))
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some((other, other_ms)) = challenger {
                    if other_ms > cur * (1.0 + DOMINANCE_MARGIN) {
                        return Some(other);
                    }
                }
                return Some(c);
            }
        }
        cohort.into_iter().min()
    }

    /// `ms_per_batch` of the elected anchor (smoothed). 0.0 if not yet elected.
    fn slow_ms(&self) -> f64 {
        self.anchor_rank
            .map(|r| self.smoothed_ms(r))
            .unwrap_or(0.0)
    }

    /// Set the target AllReduce overhead as a fraction of compute time.
    ///
    /// Default: 0.10 (10%). The anchor auto-tunes upward to keep overhead
    /// below this target. Lower values = fewer syncs = more gradient
    /// staleness.
    pub fn with_overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = target.clamp(0.01, 0.50);
        self
    }

    /// Set the maximum anchor count (gradient staleness limit).
    ///
    /// Default: 200. Higher values allow fewer syncs but accumulate more
    /// batches of gradient before averaging. Set to 1 to sync after every
    /// slow-device batch (minimal accumulation, traditional DDP cadence).
    pub fn with_max_anchor(mut self, max: usize) -> Self {
        self.max_anchor = max.max(1);
        // Ensure min_anchor doesn't exceed max_anchor
        if self.min_anchor > self.max_anchor {
            self.min_anchor = self.max_anchor;
            self.anchor = self.anchor.clamp(self.min_anchor, self.max_anchor);
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
                self.batch_counts[rank] =
                    (self.anchor as f64 * ratio).round().max(1.0) as usize;
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
    /// anchor; once `MIN_REPORTS_BEFORE_SWAP` calibrations accumulate,
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
        let new = (self.anchor as f64 * factor.clamp(0.1, 1.0)).ceil() as usize;
        self.anchor = new.max(1).min(self.anchor);
        let slow_ms = self.slow_ms();
        if slow_ms > 0.0 {
            self.recompute_batch_counts(slow_ms);
        }
    }

    /// Relax the anchor upward by 1 batch on stable convergence.
    ///
    /// Symmetric upward path to [`nudge_anchor_down`]: lets async-mode anchor
    /// drift toward [`max_anchor`] over time as long as the convergence guard
    /// reports `Stable`, amortizing AllReduce barrier cost over more local
    /// SGD steps. Closes a half-implemented loop where the convergence guard
    /// previously had a downward path (`NudgeDown`) but no upward path —
    /// `max_overshoot` grew on stable but anchor stayed stuck at `min_anchor`.
    ///
    /// Honors the user-defined `max_batch_diff` cap when set: refuses to
    /// relax if the projected per-rank batch_counts spread at `anchor + 1`
    /// would exceed `max_batch_diff`. With ratio R between fastest and
    /// slowest rank and cap M, anchor is bounded by `M / (R - 1)` — e.g.
    /// for ratio 3 and `max_batch_diff = 100`, anchor caps at 50 (yielding
    /// `[50, 150]`, diff exactly 100).
    ///
    /// No-op when already at [`max_anchor`], or when no calibrated
    /// `ms_per_batch` exists yet (Probe phase).
    pub fn relax_anchor_up(&mut self) {
        if self.anchor >= self.max_anchor {
            return;
        }
        // Honor user-defined drift cap.
        if let Some(max_diff) = self.max_batch_diff {
            let smoothed: Vec<f64> = (0..self.world_size)
                .map(|r| self.smoothed_ms(r))
                .collect();
            let max_ms = smoothed.iter().copied().fold(0.0_f64, f64::max);
            let min_ms = smoothed.iter().copied()
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
        let allow_election =
            self.anchor_rank.is_none() || self.phase >= Phase::Stable;
        if allow_election {
            if let Some(elected) = self.elect_anchor() {
                self.anchor_rank = Some(elected);
            }
        }

        // No anchor yet (no positive readings on any rank) — bail.
        let anchor_rank = match self.anchor_rank {
            Some(r) => r,
            None => return,
        };
        let slow_ms = self.smoothed_ms(anchor_rank);
        if slow_ms <= 0.0 {
            return;
        }

        // Overhead auto-tune: scales the anchor up when AllReduce overhead
        // exceeds the target, decays it slowly when overhead drops well
        // below half the target. Gated to `Phase::Stable+` because the
        // multiplicative `scale = overhead / target` compounds noise
        // dramatically on sparse early readings (the historical 10→22
        // anchor jump on the first measurement). Held off until each rank
        // has a full trust window of evidence.
        if self.phase >= Phase::Stable {
            let compute_ms = wall_ms.iter().copied().fold(0.0_f64, f64::max);
            if compute_ms > 0.0 && sync_ms > 0.0 {
                let overhead = sync_ms / compute_ms;
                if overhead > self.overhead_target {
                    let scale = overhead / self.overhead_target;
                    let new_anchor =
                        (self.anchor as f64 * scale).ceil() as usize;
                    self.anchor =
                        new_anchor.clamp(self.min_anchor, self.max_anchor);
                } else if overhead < self.overhead_target * 0.5
                          && self.anchor > self.min_anchor {
                    self.anchor -= 1;
                }
            }
        }

        // Recompute batch counts from (possibly updated) anchor.
        self.recompute_batch_counts(slow_ms);
        self.calibrated = true;
        self.calibration_count += 1;
        crate::verbose!(
            "  ddp-diag: ms_per_batch={:?} batch_counts={:?} anchor_rank={:?} anchor={}",
            self.ms_per_batch().iter().map(|m| (m * 10.0).round() / 10.0).collect::<Vec<_>>(),
            self.batch_counts,
            self.anchor_rank,
            self.anchor,
        );
        self.advance_phase();
    }

    /// Phase transition rules. Probe→Warmup at first calibration; Warmup→Stable
    /// at 5 calibrations; Stable→Mature at 20. Per-phase parameter tightening
    /// (locked anchor in Warmup, stricter Mature thresholds) lands in PR 2-3 —
    /// PR 1 only tracks transitions and logs them.
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
                self.phase, next, self.calibration_count,
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
            let target = if ms <= 0.0 || (ms - slow_ms).abs() < 1e-6 {
                self.anchor
            } else {
                let ratio = slow_ms / ms;
                (self.anchor as f64 * ratio).round().max(1.0) as usize
            };

            let current = self.batch_counts[rank];
            let diff = (target as f64 - current as f64).abs();
            // Dead zone: only update if change exceeds 5% of current count.
            // Always update on first calibration (current == anchor for all).
            if diff > current as f64 * 0.05 || !self.calibrated {
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
    }
}

/// Cold-start anchor selection from device hardware specs.
///
/// Combines compute capability (sm_major × 100 + sm_minor × 10) and total
/// VRAM in GB into a single ordinal score per rank. Higher score = better
/// spec = faster GPU (likely). The slowest rank by score is the cold-start
/// anchor pick. Compute capability dominates; VRAM tiebreaks within the
/// same arch generation.
///
/// Returns `None` if any device-property query fails — the caller falls
/// back to the rank-0 default (or whatever the existing logic produces).
mod spec_prior {
    /// Ordinal "spec score" for a CUDA device. Higher = better spec.
    /// Returns `None` when device-property queries fail (e.g. CUDA absent).
    fn score(device_index: i32) -> Option<f64> {
        let (sm_major, sm_minor) =
            crate::tensor::cuda_compute_capability(device_index)?;
        let (_free, total) =
            crate::tensor::cuda_memory_info_idx(device_index).ok()?;
        let vram_gb = total as f64 / 1_073_741_824.0;
        Some((sm_major as f64) * 100.0 + (sm_minor as f64) * 10.0 + vram_gb)
    }

    /// Rank with the lowest spec score across `device_indices`. Lowest-rank
    /// tiebreak when two ranks score equal. Returns `None` when any device
    /// query fails — caller falls back to current behavior.
    pub(super) fn slowest_rank(device_indices: &[i32]) -> Option<usize> {
        let scores: Option<Vec<(usize, f64)>> = device_indices
            .iter()
            .enumerate()
            .map(|(rank, &idx)| score(idx).map(|s| (rank, s)))
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

