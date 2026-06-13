//! Coordinator: lightweight scheduling thread for DDP run mode.

use std::sync::mpsc;

use crate::tensor::{Device, Result};

use std::collections::{BTreeMap, HashMap};

use super::{
    ApplyPolicy, AverageBackend, TimingMsg, MetricsMsg,
    ParamSnapshot, ControlMsg, EpochPlan, TrainedState,
};

mod cpu_avg;

// Re-exported at `pub(super)` so `super::coordinator::ChunkPool` works in
// `ddp_run::tests`. ChunkPool lives at `crate::distributed::chunk_pool`
// (shared between the threaded coordinator and `ClusterCoordinator`).
pub(super) use crate::distributed::chunk_pool::ChunkPool;
use cpu_avg::CpuAvgState;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Lightweight scheduling coordinator for DDP run mode.
///
/// NOT an optimizer. Each GPU runs its own Adam. The coordinator:
/// 1. Collects timing from workers (for ElChe throughput ratios)
/// 2. Triggers periodic parameter averaging (NCCL or CPU path)
/// 3. Monitors divergence to correct ElChe's anchor (tighten-only)
/// 4. Rebalances data partitions after ElChe calibrates
///
/// All fields are Send. Runs on a dedicated CPU thread.
pub struct Coordinator {
    // Channels
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<MetricsMsg>,
    /// Only used with [`AverageBackend::Cpu`].
    param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Dedicated receivers for final snapshots from each worker.
    final_param_rxs: Vec<mpsc::Receiver<ParamSnapshot>>,
    control_txs: Vec<mpsc::Sender<ControlMsg>>,

    // Configuration
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    total_samples: usize,

    // Scheduling
    pub(super) el_che: crate::distributed::ddp::ElChe,
    version: u64,
    /// Per-rank steps since last averaging.
    pub(super) steps_since_avg: Vec<usize>,
    /// Cumulative total batches across all GPUs. Updated at each sync:
    /// `global_step += sum(steps_since_avg)`. Broadcast to workers so they
    /// can compute per-batch LR as `scheduler.lr(global_step + local_offset)`.
    pub(super) global_step: usize,

    /// Has ElChe been calibrated (first timing report received)?
    calibrated: bool,

    /// Number of workers still actively training. Decremented when the
    /// coordinator drains a [`TimingMsg::Exiting`] message. Single-writer
    /// (coordinator thread only), no race with NCCL collectives.
    pub(super) active_count: usize,

    // Timing accumulation for ElChe
    /// Accumulated wall-clock ms per rank since last averaging.
    /// Fed to ElChe::report_timing at each averaging event.
    pub(super) wall_ms_accum: Vec<f64>,
    /// Most recent batch_ms per rank (for display/monitoring).
    last_batch_ms: Vec<f64>,
    /// Most recent CPU averaging time (ms). Fed to ElChe as sync_ms so
    /// the overhead auto-tune works for the CPU backend too.
    last_avg_ms: f64,
    /// Per-rank throttle state. Prevents sending duplicate Throttle messages
    /// to workers that are already blocked. Reset after averaging.
    throttled: Vec<bool>,

    // Checkpointing
    /// Number of averaging events completed.
    avg_count: usize,
    /// Save a checkpoint every N global epochs. None = disabled.
    pub(super) checkpoint_every: Option<usize>,

    // Non-blocking CPU averaging
    /// State machine for CPU averaging (Idle/Collecting/Computing).
    avg_state: CpuAvgState,
    /// Timeout for snapshot collection (seconds).
    snapshot_timeout_secs: u64,
    /// Number of CPU averaging rounds aborted due to timeout.
    abort_count: usize,
    /// Consecutive soft-abort counter (reset whenever a collection round
    /// completes). Drives the hard cap that turns an endless
    /// abort-retry livelock — a rank that never responds while staying
    /// "alive" — into a loud error instead of a silent forever-loop.
    consecutive_aborts: usize,

    // Epoch metrics aggregation
    /// Channel to send aggregated epoch metrics to DdpHandle.
    epoch_metrics_tx: Option<mpsc::Sender<super::EpochMetrics>>,
    /// Buffer for collecting per-rank metrics before aggregation.
    epoch_buffer: HashMap<usize, Vec<MetricsMsg>>,
    /// CUDA device index per rank (for EpochMetrics GPU data).
    device_indices: Vec<u8>,
    /// Optional host-side per-epoch callback. Called on this thread once per
    /// epoch with the aggregated metrics, before pushing to `epoch_metrics_tx`.
    /// Errors are logged; training continues.
    metrics_fn: Option<super::MetricsFn>,

    // Global epoch management
    /// Total number of epochs to train.
    num_epochs: usize,
    /// What epoch each rank is currently working on (last dispatched).
    pub(super) rank_epoch: Vec<usize>,
    /// True if rank finished its epoch but is blocked by lookahead (Auto mode).
    rank_waiting: Vec<bool>,
    /// Last globally-aggregated epoch (all ranks reported).
    /// None = no epoch aggregated yet.
    pub(super) last_aggregated_epoch: Option<usize>,
    /// User-specified partition ratios (disables auto-rebalancing).
    partition_ratios: Option<Vec<f64>>,
    /// Cached epoch plans: computed once per epoch, consistent across ranks.
    epoch_plan_cache: HashMap<usize, Vec<EpochPlan>>,

    // Progressive chunk dispatch
    /// Whether progressive dispatch is enabled.
    progressive: bool,
    /// Active chunk pools keyed by epoch. Multiple pools may be active when
    /// fast GPUs stream ahead into the next epoch's data.
    pub(super) chunk_pools: BTreeMap<usize, ChunkPool>,
    /// Floor for chunk size (in batches). Default: 4.
    min_chunk_batches: usize,
    /// Batch size (samples per batch), needed for chunk sizing.
    batch_size: usize,

    // Streaming epoch overshoot
    /// Maximum batches past ElChe's planned sync count any GPU may execute.
    /// Gates cross-epoch dispatch in progressive mode.
    pub(super) max_overshoot: usize,
    /// True when max_overshoot is auto-tuned (not user-set).
    pub(super) overshoot_auto: bool,
    /// Initial value for reset on convergence degradation.
    pub(super) overshoot_initial: usize,
    /// Absolute ceiling on max_overshoot (safety valve, applied after auto-tune).
    overshoot_ceiling: usize,
    /// Allow ElChe to grow the anchor on Stable convergence verdicts.
    /// When false, `relax_anchor_up()` is suppressed so the anchor only
    /// changes via the overhead-based auto-tune in `el_che.report_timing`.
    pub(super) elche_relax_up: bool,

    // Divergence monitoring (per-sync-interval, reset after averaging)
    /// Per-rank cumulative loss since last sync (monitoring/logging only,
    /// not used for cadence decisions).
    loss_accum: Vec<f64>,
    /// Per-rank batch count contributing to loss_accum since last sync.
    loss_count: Vec<usize>,
    /// Per-rank weight-space divergence from the most recent AllReduce.
    /// Set via `sync_divergence` in TimingMsg ack. Reset after averaging.
    nccl_sync_divergence: Vec<Option<f64>>,
    /// Per-rank pre-AllReduce L2 norm `||params_before||_i` from the most
    /// recent AllReduce. Set via `pre_norm` in SyncAck (workers compute it
    /// alongside divergence in the same loop). Reset after averaging.
    nccl_sync_pre_norm: Vec<Option<f64>>,
    /// Post-AllReduce consensus L2 norm `||params_after||`. Identical across
    /// ranks by AllReduce construction, so a single scalar suffices. Set
    /// from the first rank's SyncAck; subsequent rank acks are ignored
    /// (debug_assert checks consistency). Reset after averaging.
    nccl_sync_post_norm: Option<f64>,
    /// Pluggable convergence-monitoring strategy. Wired in by the
    /// builder; defaults to `TrendGuard::default()` when not set.
    convergence_guard: Box<dyn super::convergence::ConvergenceGuard>,
    /// Per-epoch d-aggregator (replaces the old EpochSnapshot owned by the
    /// guard). Lambda aggregates moved out — analyze.rs recomputes them
    /// from per-event observables now that the guard pipeline is plural.
    epoch_d_min: f64,
    epoch_d_max: f64,
    epoch_d_sum: f64,
    epoch_d_count: usize,
    epoch_last_d: f64,
    epoch_last_k_max: usize,

    // Timeline profiling
    /// Optional high-frequency system timeline for event injection.
    timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,
    /// Instant when the last NCCL sync started (for duration measurement).
    nccl_sync_start: Option<std::time::Instant>,
    /// Wall-time duration (ms) of the most-recent completed NCCL sync.
    /// Captured in `process_timing_msg` when the last rank acks. Fed to
    /// `ElChe::report_timing` as `sync_ms` so the anchor auto-tune block
    /// fires on the NCCL backend (the CPU-avg path uses `last_avg_ms`).
    last_nccl_sync_ms: f64,

    // LR-aware meta-controller
    /// Optional meta-controller above ElChe. `None` when
    /// [`super::DdpRunConfig::with_meta_controller`] is `false` (default).
    pub(super) lr_event_meta: Option<crate::distributed::lr_event_meta::LrEventMeta>,
    /// Most recent LR observed per rank from
    /// [`super::TimingMsg::LrUpdate`]. Indexed by rank. `None` until the
    /// first message from that rank arrives.
    pub(super) last_lr_per_rank: Vec<Option<f64>>,

    // NCCL sync acknowledgment
    /// Per-rank: last worker `step_count` seen in a TimingMsg.
    /// Monotonically increasing (workers never reset `local_step`).
    last_step_count: Vec<usize>,
    /// Per-rank: `last_step_count` snapshot at the time SyncNow was sent.
    /// A rank is acknowledged when its `step_count` exceeds this threshold.
    nccl_sync_step: Vec<usize>,
    /// Per-rank: true once a post-sync timing message has arrived.
    /// Without this gate, stale timing from pre-sync batches refills
    /// `steps_since_avg` and floods AllReduce calls, deadlocking GPU streams.
    nccl_ack: Vec<bool>,
}

/// Builder for configuring a [`Coordinator`].
pub struct CoordinatorBuilder {
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<MetricsMsg>,
    param_rx: mpsc::Receiver<ParamSnapshot>,
    final_param_rxs: Vec<mpsc::Receiver<ParamSnapshot>>,
    control_txs: Vec<mpsc::Sender<ControlMsg>>,
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    total_samples: usize,
    el_che: crate::distributed::ddp::ElChe,
    divergence_threshold: f64,
    divergence_guard: bool,
    /// Pluggable convergence guard. When set, takes precedence over the
    /// legacy `(divergence_guard, divergence_threshold)` pair, which become
    /// `TrendGuard` configuration only when no explicit guard is supplied.
    convergence_guard_override: Option<Box<dyn super::convergence::ConvergenceGuard>>,
    checkpoint_every: Option<usize>,
    snapshot_timeout_secs: u64,
    epoch_metrics_tx: Option<mpsc::Sender<super::EpochMetrics>>,
    device_indices: Vec<u8>,
    num_epochs: usize,
    partition_ratios: Option<Vec<f64>>,
    progressive: bool,
    batch_size: usize,
    timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,
    /// User-set max overshoot, or None for auto.
    max_overshoot: Option<usize>,
    /// Absolute ceiling on max_overshoot (safety valve). Default: 15.
    overshoot_ceiling: usize,
    /// Allow anchor relax-up on Stable convergence. Default: false.
    elche_relax_up: bool,
    metrics_fn: Option<super::MetricsFn>,
    /// Enable the LR-aware meta-controller. Default: false (off; opt-in
    /// until validation sweep). See
    /// [`crate::distributed::lr_event_meta`] for the design.
    meta_controller: bool,
}

impl CoordinatorBuilder {
    /// Enable or disable progressive chunk dispatch.
    /// Default: true for Cadence/Async, false for Sync.
    pub fn progressive(mut self, enabled: bool) -> Self {
        self.progressive = enabled;
        self
    }

    /// Set the batch size (needed for chunk sizing in progressive mode).
    pub fn batch_size(mut self, bs: usize) -> Self {
        self.batch_size = bs;
        self
    }

    /// Attach a system timeline for event injection.
    pub fn timeline(mut self, tl: Option<std::sync::Arc<crate::monitor::Timeline>>) -> Self {
        self.timeline = tl;
        self
    }

    /// Set the divergence threshold for the trend guardrail.
    /// Default: 0.05 (5% relative loss divergence between ranks).
    pub fn divergence_threshold(mut self, threshold: f64) -> Self {
        self.divergence_threshold = threshold;
        self
    }

    /// Disable the divergence guardrail entirely.
    /// ElChe's overhead auto-tune handles cadence on its own; the guardrail
    /// is an optional safety net that suppresses anchor growth when replicas
    /// drift apart. Disable when you know your workload is stable or prefer
    /// full control via ElChe's parameters.
    /// Install a fully-configured convergence guard. Takes precedence over
    /// the legacy `divergence_threshold` / `no_divergence_guard` settings.
    pub fn convergence_guard(
        mut self,
        guard: Box<dyn super::convergence::ConvergenceGuard>,
    ) -> Self {
        self.convergence_guard_override = Some(guard);
        self
    }

    pub fn no_divergence_guard(mut self) -> Self {
        self.divergence_guard = false;
        self
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    /// Default: 0.10 (10%). Lower = more frequent sync, higher = less overhead.
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.el_che = self.el_che.with_overhead_target(target);
        self
    }

    /// Set the maximum anchor (max batches between AllReduce).
    /// Default: 1000. Controls gradient staleness bound.
    pub fn max_anchor(mut self, max: usize) -> Self {
        self.el_che = self.el_che.with_max_anchor(max);
        self
    }

    /// Set the checkpoint interval (global epochs between checkpoints).
    pub fn checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = Some(n);
        self
    }

    /// Set the timeout for CPU averaging snapshot collection (seconds).
    pub fn snapshot_timeout_secs(mut self, secs: u64) -> Self {
        self.snapshot_timeout_secs = secs;
        self
    }

    /// Set the channel for forwarding aggregated epoch metrics to the main thread.
    pub fn epoch_metrics_tx(mut self, tx: mpsc::Sender<super::EpochMetrics>) -> Self {
        self.epoch_metrics_tx = Some(tx);
        self
    }

    /// Set the host-side per-epoch metrics callback.
    ///
    /// Called on the coordinator thread once per epoch, before pushing to the
    /// `epoch_metrics_tx` queue. Errors are logged to stderr; training continues.
    pub fn metrics_fn(mut self, f: super::MetricsFn) -> Self {
        self.metrics_fn = Some(f);
        self
    }

    /// Set the CUDA device indices (one per rank).
    pub fn device_indices(mut self, indices: Vec<u8>) -> Self {
        self.device_indices = indices;
        self
    }

    /// Set the total number of epochs to train.
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    /// Set explicit per-rank partition ratios.
    pub fn partition_ratios(mut self, ratios: Option<Vec<f64>>) -> Self {
        self.partition_ratios = ratios;
        self
    }

    /// Set the maximum overshoot past the planned sync point.
    /// `None` = auto-tuned. `Some(0)` = disable cross-epoch streaming.
    pub fn max_overshoot(mut self, max: Option<usize>) -> Self {
        self.max_overshoot = max;
        self
    }

    /// Set the absolute ceiling on max_overshoot (safety valve).
    /// Default: 15. Applied after all auto-tune logic.
    pub fn overshoot_ceiling(mut self, ceiling: usize) -> Self {
        self.overshoot_ceiling = ceiling;
        self
    }

    /// Allow or suppress ElChe's anchor relax-up on Stable convergence.
    /// Default: false (off). Opt in to grow the anchor on `Stable` verdicts;
    /// when false the anchor only changes via the overhead-based auto-tune.
    pub fn elche_relax_up(mut self, enabled: bool) -> Self {
        self.elche_relax_up = enabled;
        self
    }

    /// Enable the LR-aware meta-controller above ElChe. Default: false.
    ///
    /// When enabled, a [`crate::distributed::lr_event_meta::LrEventMeta`]
    /// is constructed and held by the coordinator. The coordinator forwards
    /// per-cycle LR + guard verdicts and dispatches any returned action
    /// to ElChe's `nudge_anchor_down` path.
    pub fn meta_controller(mut self, enabled: bool) -> Self {
        self.meta_controller = enabled;
        self
    }

    /// Build the coordinator.
    pub fn build(self) -> Coordinator {
        let total_batches = self.total_samples / self.batch_size.max(1);
        let overshoot_auto = self.max_overshoot.is_none();
        let overshoot_initial = match self.max_overshoot {
            Some(n) => n,
            None => (total_batches / 100).clamp(2, 5),
        };

        Coordinator {
            timing_rx: self.timing_rx,
            metrics_rx: self.metrics_rx,
            param_rx: self.param_rx,
            final_param_rxs: self.final_param_rxs,
            control_txs: self.control_txs,
            policy: self.policy,
            backend: self.backend,
            world_size: self.world_size,
            total_samples: self.total_samples,
            el_che: self.el_che,
            version: 0,
            steps_since_avg: vec![0; self.world_size],
            global_step: 0,
            calibrated: false,
            active_count: self.world_size,
            wall_ms_accum: vec![0.0; self.world_size],
            last_batch_ms: vec![0.0; self.world_size],
            last_avg_ms: 0.0,
            last_nccl_sync_ms: 0.0,
            throttled: vec![false; self.world_size],
            avg_count: 0,
            checkpoint_every: self.checkpoint_every,
            avg_state: CpuAvgState::Idle,
            snapshot_timeout_secs: self.snapshot_timeout_secs,
            abort_count: 0,
            consecutive_aborts: 0,
            epoch_metrics_tx: self.epoch_metrics_tx,
            epoch_buffer: HashMap::new(),
            device_indices: self.device_indices,
            num_epochs: self.num_epochs,
            rank_epoch: vec![0; self.world_size],
            rank_waiting: vec![false; self.world_size],
            last_aggregated_epoch: None,
            partition_ratios: self.partition_ratios,
            epoch_plan_cache: HashMap::new(),
            progressive: self.progressive,
            chunk_pools: BTreeMap::new(),
            min_chunk_batches: 4,
            batch_size: self.batch_size.max(1),
            max_overshoot: overshoot_initial,
            overshoot_auto,
            overshoot_initial,
            overshoot_ceiling: self.overshoot_ceiling,
            elche_relax_up: self.elche_relax_up,
            loss_accum: vec![0.0; self.world_size],
            loss_count: vec![0; self.world_size],
            nccl_sync_divergence: vec![None; self.world_size],
            nccl_sync_pre_norm: vec![None; self.world_size],
            nccl_sync_post_norm: None,
            convergence_guard: self.convergence_guard_override.unwrap_or_else(|| {
                if self.divergence_guard {
                    Box::new(super::convergence::TrendGuard::new(self.divergence_threshold))
                } else {
                    Box::new(super::convergence::NoGuard)
                }
            }),
            epoch_d_min: f64::INFINITY,
            epoch_d_max: f64::NEG_INFINITY,
            epoch_d_sum: 0.0,
            epoch_d_count: 0,
            epoch_last_d: 0.0,
            epoch_last_k_max: 0,
            timeline: self.timeline,
            nccl_sync_start: None,
            lr_event_meta: if self.meta_controller {
                Some(crate::distributed::lr_event_meta::LrEventMeta::with_default_config())
            } else {
                None
            },
            last_lr_per_rank: vec![None; self.world_size],
            last_step_count: vec![0; self.world_size],
            nccl_sync_step: vec![0; self.world_size],
            nccl_ack: vec![true; self.world_size],
            metrics_fn: self.metrics_fn,
        }
    }
}

impl Coordinator {
    /// Create a coordinator builder.
    #[allow(clippy::too_many_arguments)]
    pub fn builder(
        timing_rx: mpsc::Receiver<TimingMsg>,
        metrics_rx: mpsc::Receiver<MetricsMsg>,
        param_rx: mpsc::Receiver<ParamSnapshot>,
        final_param_rxs: Vec<mpsc::Receiver<ParamSnapshot>>,
        control_txs: Vec<mpsc::Sender<ControlMsg>>,
        policy: ApplyPolicy,
        backend: AverageBackend,
        world_size: usize,
        total_samples: usize,
        el_che: crate::distributed::ddp::ElChe,
    ) -> CoordinatorBuilder {
        CoordinatorBuilder {
            timing_rx,
            metrics_rx,
            param_rx,
            final_param_rxs,
            control_txs,
            policy,
            backend,
            world_size,
            total_samples,
            el_che,
            divergence_threshold: 0.05,
            divergence_guard: true,
            convergence_guard_override: None,
            checkpoint_every: None,
            snapshot_timeout_secs: 5,
            epoch_metrics_tx: None,
            device_indices: (0..world_size as u8).collect(),
            num_epochs: 1,
            partition_ratios: None,
            progressive: !matches!(policy, ApplyPolicy::Sync),
            batch_size: 1,
            timeline: None,
            max_overshoot: None,
            overshoot_ceiling: 15,
            elche_relax_up: false,
            metrics_fn: None,
            meta_controller: false,
        }
    }

    /// Feed an averaging-cycle observation to the LR-aware meta-controller
    /// (when enabled) and dispatch any returned action to ElChe.
    ///
    /// No-op when:
    /// - meta is disabled (`lr_event_meta` is `None`),
    /// - no rank has reported its LR yet (cold-start, ≤ 1 cycle).
    ///
    /// On [`crate::distributed::lr_event_meta::MetaAction::NudgeDown`], calls
    /// [`crate::distributed::ElChe::nudge_anchor_down`] with the
    /// trend-dampened factor and logs the transition. ElChe's overhead
    /// auto-tune handles the natural relax-back over subsequent cycles.
    pub(super) fn observe_meta(
        &mut self,
        verdict: super::convergence::ConvergenceAction,
    ) {
        let Some(lr) = self.last_lr_per_rank.iter().copied().find_map(|x| x) else {
            return;
        };
        let anchor = self.el_che.anchor();
        let phase = self.el_che.phase();
        let action = match self.lr_event_meta.as_mut() {
            Some(meta) => meta.observe(lr, anchor, verdict, phase),
            None => return,
        };
        if let crate::distributed::lr_event_meta::MetaAction::NudgeDown { factor } = action {
            let old = self.el_che.anchor();
            self.el_che.nudge_anchor_down(factor);
            let new = self.el_che.anchor();
            // Don't emit AnchorChanged here: the post-match block in
            // finish_averaging_{nccl,cpu} captures the cycle's net anchor
            // change and emits a single event covering both the meta nudge
            // and any guard-driven adjustment that follows. MetaNudge
            // isolates the meta's contribution with the raw factor.
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::MetaNudge {
                    factor,
                    from: old,
                    to: new,
                });
            }
            crate::verbose!(
                "  ddp: meta-controller nudge factor={:.3} anchor {} -> {}",
                factor, old, new,
            );
        }
    }

    /// Current model version (bumped after each averaging).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Whether ElChe has been calibrated.
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Per-rank steps since last averaging.
    pub fn steps_since_avg(&self) -> &[usize] {
        &self.steps_since_avg
    }

    /// Whether CPU averaging is currently in progress (Collecting or Computing).
    pub fn is_cpu_averaging(&self) -> bool {
        !matches!(self.avg_state, CpuAvgState::Idle)
    }

    /// Number of successful averaging events completed.
    pub fn avg_count(&self) -> usize {
        self.avg_count
    }

    /// Number of CPU averaging rounds aborted due to timeout.
    pub fn abort_count(&self) -> usize {
        self.abort_count
    }

    /// Most recent per-rank batch time (ms).
    pub fn last_batch_ms(&self) -> &[f64] {
        &self.last_batch_ms
    }

    /// Most recent CPU averaging time (ms). Zero for NCCL backend.
    pub fn last_avg_ms(&self) -> f64 {
        self.last_avg_ms
    }

    /// Whether all epochs have been aggregated (training is complete).
    pub fn all_epochs_done(&self) -> bool {
        self.last_aggregated_epoch.is_some_and(|e| e + 1 >= self.num_epochs)
    }

    /// Emit a periodic state dump (gated by `-vv` verbosity).
    pub fn debug_state_dump(&self, tick: u64) {
        if !crate::log::enabled(crate::log::Verbosity::Debug) {
            return;
        }
        let pools: Vec<_> = self.chunk_pools.iter()
            .map(|(e, p)| {
                let inf: Vec<_> = (0..self.world_size)
                    .map(|r| format!("{}:{}/{}", r, p.completed[r], p.dispatched[r]))
                    .collect();
                format!("e{}(cur={}/{} [{}])", e, p.cursor, p.total_samples, inf.join(" "))
            })
            .collect();
        let wall_rounded: Vec<_> = self.wall_ms_accum.iter()
            .map(|w| (w * 10.0).round() / 10.0)
            .collect();
        crate::debug!(
            "  ddp-state: tick={} steps={:?} wall={:.0?} throttled={:?} \
             nccl_ack={:?} rank_epoch={:?} active={} last_agg={:?} avg#={} \
             pools=[{}]",
            tick, self.steps_since_avg, wall_rounded,
            self.throttled, self.nccl_ack,
            self.rank_epoch, self.active_count,
            self.last_aggregated_epoch, self.avg_count,
            pools.join(", "),
        );
    }

    // -----------------------------------------------------------------------
    // Global epoch management
    // -----------------------------------------------------------------------

    /// Compute partition sizes per rank based on policy and throughput.
    pub(super) fn compute_partition_sizes(&self) -> Vec<usize> {
        if let Some(ratios) = &self.partition_ratios {
            return ratio_to_sizes(ratios, self.total_samples);
        }
        match self.policy {
            ApplyPolicy::Sync => {
                equal_sizes(self.world_size, self.total_samples)
            }
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                if self.el_che.is_calibrated() || self.el_che.has_speed_hint() {
                    throughput_sizes(&self.el_che, self.total_samples)
                } else {
                    equal_sizes(self.world_size, self.total_samples)
                }
            }
        }
    }

    /// Get (or lazily compute) the epoch plans for a given epoch.
    ///
    /// Partition sizes are computed once per epoch and cached, ensuring
    /// consistent offsets across all ranks even when dispatched at different times.
    fn plans_for_epoch(&mut self, epoch: usize) -> Vec<EpochPlan> {
        if let Some(plans) = self.epoch_plan_cache.get(&epoch) {
            return plans.clone();
        }
        let sizes = self.compute_partition_sizes();
        let mut plans = Vec::with_capacity(self.world_size);
        let mut offset = 0;
        for &size in &sizes {
            plans.push(EpochPlan { epoch, partition_offset: offset, partition_size: size });
            offset += size;
        }
        crate::verbose!("  ddp: epoch {epoch} | partitions {sizes:?}");
        self.epoch_plan_cache.insert(epoch, plans.clone());
        plans
    }

    /// Send StartEpoch to all ranks (used for epoch 0 and Sync/Cadence dispatch).
    ///
    /// In progressive mode, delegates to `start_epoch_progressive`.
    pub fn send_all_plans(&mut self, epoch: usize) {
        if self.progressive {
            self.start_epoch_progressive(epoch);
            return;
        }
        let plans = self.plans_for_epoch(epoch);
        for (rank, plan) in plans.into_iter().enumerate() {
            self.rank_epoch[rank] = epoch;
            self.rank_waiting[rank] = false;
            let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(plan));
        }
    }

    // -----------------------------------------------------------------------
    // Progressive chunk dispatch
    // -----------------------------------------------------------------------

    /// Start a new epoch in progressive mode: create a chunk pool and
    /// dispatch initial chunks to all ranks.
    fn start_epoch_progressive(&mut self, epoch: usize) {
        // Align pool total to batch boundary. Sub-batch remainders can't form
        // a full batch, so they're dropped (standard DataLoader behaviour).
        // Without this, is_epoch_done never fires when total % batch_size != 0.
        let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
        let pool = ChunkPool::new(epoch, batch_total, self.world_size);
        self.chunk_pools.insert(epoch, pool);

        let sizes: Vec<usize> = (0..self.world_size)
            .map(|r| self.compute_chunk_batches(r, epoch))
            .collect();
        crate::verbose!(
            "  ddp: epoch {epoch} progressive | initial chunks (batches) {sizes:?}"
        );
        for (rank, &batch_count) in sizes.iter().enumerate() {
            self.dispatch_next_chunk_with_batches(rank, epoch, batch_count);
        }
    }

    /// Dispatch the next chunk to a rank from the active pool.
    ///
    /// Computes chunk size based on calibration state, takes from the pool,
    /// and sends a `StartEpoch` plan. Does nothing if the pool is exhausted.
    fn dispatch_next_chunk(&mut self, rank: usize) {
        let epoch = self.rank_epoch[rank];
        // Try current epoch's pool first
        if self.chunk_pools.get(&epoch).is_some_and(|p| p.remaining() > 0) {
            let batches = self.compute_chunk_batches(rank, epoch);
            let remaining = self.chunk_pools.get(&epoch).map_or(0, |p| p.remaining());
            crate::verbose!(
                "  ddp: chunk -> rank {rank} | {batches} batches | {remaining} samples left"
            );
            self.dispatch_next_chunk_with_batches(rank, epoch, batches);
            return;
        }

        // Current pool exhausted for this rank. Try cross-epoch streaming.
        // Skip past already-aggregated epochs: their pools were removed
        // during try_aggregate_epochs_progressive. Re-creating them here
        // would produce an orphan pool that blocks all future aggregation
        // (BTreeMap iteration breaks at the first incomplete pool).
        let first_live = self.last_aggregated_epoch
            .map_or(0, |agg| agg + 1);
        let next_epoch = (epoch + 1).max(first_live);
        if next_epoch >= self.num_epochs {
            return;
        }

        // Overshoot gate (Async only, both backends): don't dispatch if
        // rank has exceeded its planned batch count by more than
        // max_overshoot since the last sync. Only applies when streaming
        // AHEAD of a not-yet-aggregated epoch. If the rank's current
        // epoch is already aggregated, this is a normal transition.
        //
        // Sync runs averaging every batch so steps_since_avg never drifts;
        // Cadence uses AllReduce as its sole coordination layer (per
        // `feedback_nccl_no_overshoot_throttle`). Only Async accumulates
        // cross-cycle drift that needs a batch-scale bound. A gated NCCL
        // rank in `wait_for_epoch_plan` still processes the next SyncNow
        // via `dispatch_control`, and `finish_averaging_nccl` re-dispatches
        // idle ranks once `steps_since_avg` is reset.
        if matches!(self.policy, ApplyPolicy::Async) {
            let current_aggregated = self.last_aggregated_epoch
                .is_some_and(|agg| epoch <= agg);
            if !current_aggregated {
                let planned = self.el_che.batch_counts().get(rank).copied().unwrap_or(0);
                if planned > 0 && self.steps_since_avg[rank] >= planned + self.max_overshoot {
                    crate::debug!(
                        "  ddp: overshoot gate BLOCKED rank {rank} | steps={} planned={} overshoot={} | wall_ms={:?}",
                        self.steps_since_avg[rank], planned, self.max_overshoot, self.wall_ms_accum,
                    );
                    return; // At overshoot limit, wait for next AllReduce
                }
            }
        }

        // Create next epoch's pool on-demand
        if !self.chunk_pools.contains_key(&next_epoch) {
            let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
            self.chunk_pools.insert(
                next_epoch,
                ChunkPool::new(next_epoch, batch_total, self.world_size),
            );
            crate::verbose!("  ddp: streaming -> epoch {next_epoch} pool created");
        }

        let batches = self.compute_chunk_batches(rank, next_epoch);
        let remaining = self.chunk_pools.get(&next_epoch).map_or(0, |p| p.remaining());
        crate::verbose!(
            "  ddp: chunk -> rank {rank} | {batches} batches | {remaining} samples left (epoch {next_epoch})"
        );
        self.dispatch_next_chunk_with_batches(rank, next_epoch, batches);
    }

    fn dispatch_next_chunk_with_batches(&mut self, rank: usize, epoch: usize, batches: usize) {
        let samples = batches * self.batch_size;
        if samples == 0 {
            return;
        }
        let (offset, actual_size) = match self.chunk_pools.get_mut(&epoch) {
            Some(pool) => match pool.take_chunk(samples, rank) {
                Some(v) => v,
                None => return,
            },
            None => return,
        };
        self.rank_epoch[rank] = epoch;
        self.rank_waiting[rank] = false;
        let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(EpochPlan {
            epoch,
            partition_offset: offset,
            partition_size: actual_size,
        }));
    }

    /// Compute how many batches the next chunk for `rank` should contain.
    fn compute_chunk_batches(&self, rank: usize, epoch: usize) -> usize {
        let pool = match self.chunk_pools.get(&epoch) {
            Some(pool) => pool,
            None => return 0,
        };
        let remaining_samples = pool.remaining();
        let remaining_batches = remaining_samples / self.batch_size;
        if remaining_batches == 0 {
            return 0;
        }

        if !self.el_che.is_calibrated() && !self.el_che.has_speed_hint() {
            // Probe: small equal chunks for fast calibration.
            // ~10% of total per rank, min 4 batches. Enough for 5-6 averaging
            // events at anchor=10, giving ElChe's EMA time to stabilize.
            let probe = (self.total_samples / (self.world_size * 10 * self.batch_size)).max(4);
            return probe.min(remaining_batches);
        }

        // Calibrated: proportional to throughput
        let counts = self.el_che.batch_counts();
        let total_counts: usize = counts.iter().sum();
        if total_counts == 0 {
            return remaining_batches.min(self.min_chunk_batches);
        }
        let ratio = counts[rank] as f64 / total_counts as f64;
        let mut target = (remaining_batches as f64 * ratio).ceil() as usize;

        // Tail-balance: when remaining work won't fill a full round of
        // chunks, size this chunk to finish when the slowest in-flight
        // rank finishes, preventing fast-GPU idle at epoch end.
        // Works in samples (not batches) to avoid truncation from
        // non-batch-aligned tail chunks.
        if remaining_batches < target * self.world_size {
            let my_ms = self.last_batch_ms[rank];
            if my_ms > 0.0 {
                let ms_per_sample = my_ms / self.batch_size as f64;
                let max_other_ms = (0..self.world_size)
                    .filter(|&r| r != rank)
                    .map(|r| {
                        let in_flight = pool.in_flight(r);
                        let r_ms = if self.last_batch_ms[r] > 0.0 {
                            self.last_batch_ms[r] / self.batch_size as f64
                        } else {
                            ms_per_sample
                        };
                        in_flight as f64 * r_ms
                    })
                    .fold(0.0_f64, f64::max);

                // Only tail-balance when the slowest rank has more than
                // one batch worth of wall-time left — below that the
                // overhead of a smaller chunk isn't worth it.
                if max_other_ms > self.last_batch_ms[rank] {
                    let fill = (max_other_ms / ms_per_sample).ceil() as usize;
                    let fill_batches = fill.div_ceil(self.batch_size);
                    target = target.min(fill_batches);
                }
            }
        }

        target.max(self.min_chunk_batches).min(remaining_batches)
    }

    /// Send StartEpoch to a single rank (Auto per-rank dispatch).
    fn send_rank_plan(&mut self, rank: usize, epoch: usize) {
        let plans = self.plans_for_epoch(epoch);
        if let Some(plan) = plans.into_iter().nth(rank) {
            self.rank_epoch[rank] = epoch;
            self.rank_waiting[rank] = false;
            let _ = self.control_txs[rank].send(ControlMsg::StartEpoch(plan));
        }
    }

    /// Called per-message when a rank's MetricsMsg arrives (epoch done for that rank).
    ///
    /// In Auto mode, immediately dispatches the next epoch if within lookahead.
    fn on_rank_done(&mut self, rank: usize, finished_epoch: usize) {
        if !matches!(self.policy, ApplyPolicy::Async) {
            return;
        }
        let next = finished_epoch + 1;
        if next >= self.num_epochs {
            return;
        }
        let within_lookahead = match self.last_aggregated_epoch {
            // Before any aggregation: allow epoch 0 and 1.
            // Epoch 0 was sent at startup; epoch 1 is the first lookahead.
            None => next <= 1,
            Some(agg) => next.saturating_sub(agg) <= 1,
        };
        if within_lookahead {
            self.send_rank_plan(rank, next);
        } else {
            self.rank_waiting[rank] = true;
        }
    }

    /// Called when all ranks have reported for an epoch (aggregation complete).
    ///
    /// Dispatches next epoch or sends Shutdown based on policy.
    pub(super) fn on_epoch_aggregated(&mut self, epoch: usize) {
        self.last_aggregated_epoch = Some(epoch);
        self.epoch_plan_cache.remove(&epoch);

        // Drain per-epoch d-aggregator. Lambda fields are intentionally
        // None going forward — pluggable guards mean per-epoch lambda
        // aggregation belongs to analyze.rs (it recomputes from per-event
        // observables) rather than to a guard-specific snapshot. Emit only
        // when at least one AllReduce happened (Sync mode with no
        // divergence reports yields count=0).
        let snap = self.take_epoch_d_summary();
        if snap.count > 0
            && let Some(ref tl) = self.timeline
        {
            tl.event(crate::monitor::EventKind::DivergenceEpoch {
                epoch,
                sync_count: snap.count,
                d_min: snap.d_min,
                d_max: snap.d_max,
                d_mean: snap.d_mean(),
                lambda_min: None,
                lambda_max: None,
                lambda_mean: None,
                lambda_ema_at_epoch_end: None,
                d_at_epoch_end: snap.d_at_epoch_end,
                k_at_epoch_end: snap.k_at_epoch_end,
            });
        }

        // Checkpoint on global epoch boundaries (1-based for file naming).
        // Threaded DDP sends only to rank 0's channel, so target_rank=0 is
        // both correct semantically (rank 0 is the only recipient) and
        // matches the v1 cluster-mode invariant (worker no-ops unless
        // target_rank == self.rank).
        if let Some(every) = self.checkpoint_every {
            if every > 0 && (epoch + 1) % every == 0 {
                if let Some(tx) = self.control_txs.first() {
                    let _ = tx.send(ControlMsg::Checkpoint {
                        version: (epoch + 1) as u64,
                        target_rank: 0,
                    });
                }
            }
        }

        let next_global = epoch + 1;
        if next_global >= self.num_epochs {
            // All epochs done: tell all workers to exit.
            for tx in &self.control_txs {
                let _ = tx.send(ControlMsg::Shutdown);
            }
            return;
        }

        if self.progressive {
            // Streaming epochs: pools are created on-demand by dispatch_next_chunk.
            // Re-dispatch to idle ranks (no in-flight chunks) that may be waiting
            // for work after exhausting their previous pool.
            for rank in 0..self.world_size {
                let has_inflight = self.chunk_pools.values()
                    .any(|p| p.in_flight(rank) > 0);
                if !has_inflight {
                    self.dispatch_next_chunk(rank);
                }
            }
            return;
        }

        match self.policy {
            ApplyPolicy::Sync | ApplyPolicy::Cadence => {
                self.send_all_plans(next_global);
            }
            ApplyPolicy::Async => {
                // Legacy per-rank dispatch already happened in on_rank_done.
                // Unblock ranks that were waiting due to lookahead.
                for rank in 0..self.world_size {
                    if self.rank_waiting[rank] {
                        let next = self.rank_epoch[rank] + 1;
                        if next < self.num_epochs {
                            self.send_rank_plan(rank, next);
                        }
                    }
                }
            }
        }
    }

    /// Process a single timing message. Shared by [`Self::drain_timing`] and
    /// [`Self::drain_timing_blocking`].
    fn process_timing_msg(&mut self, msg: TimingMsg) {
        match msg {
            TimingMsg::Batch { rank, batch_ms, step_count, param_norm, batch_loss, sync_divergence } => {
                self.steps_since_avg[rank] = self.steps_since_avg[rank].saturating_add(1);
                self.wall_ms_accum[rank] += batch_ms;
                self.last_step_count[rank] = self.last_step_count[rank].max(step_count);
                self.last_batch_ms[rank] = batch_ms;
                // Accumulate loss for monitoring (not used for cadence decisions).
                if batch_loss > 0.0 {
                    self.loss_accum[rank] += batch_loss;
                    self.loss_count[rank] += 1;
                }
                // Capture weight-space divergence from post-sync ack.
                if let Some(div) = sync_divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                let _ = param_norm; // retained in TimingMsg for monitoring
                // Ack NCCL sync when the worker's step_count exceeds the
                // snapshot at trigger time (proves the worker processed the
                // SyncNow and completed the AllReduce before this batch).
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsg::SyncAck { rank, step_count, divergence, post_norm, pre_norm } => {
                // Post-SyncNow ack: update step count for nccl_ack + capture
                // divergence + pre/post_norm, but do NOT increment
                // steps_since_avg. Treating this as a batch inflates
                // global_step by one per sync per rank, firing LR schedulers
                // early.
                self.last_step_count[rank] = self.last_step_count[rank].max(step_count);
                if let Some(div) = divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if let Some(p) = pre_norm {
                    self.nccl_sync_pre_norm[rank] = Some(p);
                }
                if let Some(p) = post_norm {
                    // Post-AllReduce identical across ranks; first rank wins,
                    // subsequent acks must agree to within fp tolerance.
                    match self.nccl_sync_post_norm {
                        None => self.nccl_sync_post_norm = Some(p),
                        Some(prev) => debug_assert!(
                            (prev - p).abs() <= 1e-6 * prev.abs().max(1.0),
                            "post_norm rank-disagreement: prev={prev} new={p} (rank {rank})"
                        ),
                    }
                }
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsg::Exiting { .. } => {
                self.active_count = self.active_count.saturating_sub(1);
            }
            TimingMsg::LrUpdate { rank, lr } => {
                if rank < self.last_lr_per_rank.len() {
                    self.last_lr_per_rank[rank] = Some(lr);
                }
            }
            // Cluster-mode only; OLD threaded coordinator never sees
            // these (workers emit them from the cluster-mode heartbeat
            // thread + CPU param bridge + NCCL re-rendezvous helper).
            TimingMsg::Heartbeat { .. } => {}
            TimingMsg::SnapshotReady { .. } => {}
            TimingMsg::NewNcclIdGenerated { .. } => {}
            TimingMsg::EvalResult { .. } => {}
            // Threaded DDP does not retry checkpoints (single-process,
            // shared address space, no failure-on-write expected
            // beyond what `eprintln!` already surfaces). Cluster mode
            // owns the retry policy; the threaded path drops the
            // frame intentionally.
            TimingMsg::CheckpointResult { rank: _, version, elapsed_ms: _, error } => {
                if let Some(e) = error {
                    eprintln!(
                        "ddp (threaded): checkpoint v{version} reported \
                         failure: {e}"
                    );
                }
            }
            // Threaded DDP does not exercise the cluster_coordinator
            // EWMA / time-exclusion path. Frame is dropped intentionally.
            TimingMsg::EpochFnElapsed { .. } => {}
        }
    }

    /// If every rank has now acknowledged the in-flight NCCL sync, record
    /// its wall-time duration (sent SyncNow → all ranks past the AllReduce).
    /// The next call to `finish_averaging_nccl` will feed this as `sync_ms`
    /// to `ElChe::report_timing`, so the anchor auto-tune block actually
    /// fires on NCCL backend (was always 0 before — `last_avg_ms` is only
    /// populated by the CPU averaging path).
    fn capture_nccl_sync_elapsed_if_complete(&mut self) {
        if self.nccl_ack.iter().all(|&a| a) {
            if let Some(start) = self.nccl_sync_start.take() {
                self.last_nccl_sync_ms =
                    start.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    /// Process all pending timing messages (non-blocking drain).
    ///
    /// Updates per-rank step counts and accumulates wall-clock time for ElChe.
    /// When a worker sends [`TimingMsg::Exiting`], decrements `active_count`
    /// so [`should_average`](Self::should_average) stops triggering collectives.
    pub fn drain_timing(&mut self) {
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
    }

    /// Block until a timing message arrives or `timeout` elapses, then drain
    /// all remaining messages non-blocking.
    ///
    /// Returns `false` if the channel is disconnected (all senders dropped),
    /// meaning all workers have exited. The caller should break its loop.
    pub fn drain_timing_blocking(&mut self, timeout: std::time::Duration) -> bool {
        match self.timing_rx.recv_timeout(timeout) {
            Ok(msg) => self.process_timing_msg(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => return true,
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
        // Drain remaining messages non-blocking
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
        true
    }

    /// Process all pending metrics messages (non-blocking drain).
    ///
    /// Returns collected metrics for logging/monitoring. Also buffers
    /// per-rank messages and aggregates into [`EpochMetrics`](super::EpochMetrics)
    /// when all active ranks have reported for the same epoch.
    pub fn drain_metrics(&mut self) -> Vec<MetricsMsg> {
        let mut msgs = Vec::new();
        while let Ok(msg) = self.metrics_rx.try_recv() {
            if self.progressive {
                // Progressive: route completion to the correct pool by epoch
                if let Some(pool) = self.chunk_pools.get_mut(&msg.epoch) {
                    pool.mark_completed(msg.rank, msg.samples_processed);
                }
                crate::debug!(
                    "  ddp: metrics rank {} epoch {} done | samples={} | pools={:?}",
                    msg.rank, msg.epoch, msg.samples_processed,
                    self.chunk_pools.keys().collect::<Vec<_>>(),
                );
                self.epoch_buffer.entry(msg.epoch).or_default().push(msg.clone());
                // Dispatch next chunk to this rank (if pool has work)
                self.dispatch_next_chunk(msg.rank);
            } else {
                // Legacy: per-rank Auto dispatch before aggregation
                self.on_rank_done(msg.rank, msg.epoch);
                self.epoch_buffer.entry(msg.epoch).or_default().push(msg.clone());
            }
            msgs.push(msg);
        }
        // Global aggregation + dispatch
        self.try_aggregate_epochs();
        msgs
    }

    /// Check if any buffered epoch has reports from all active ranks.
    /// If so, aggregate, send metrics, and trigger epoch transitions.
    fn try_aggregate_epochs(&mut self) {
        if self.progressive {
            self.try_aggregate_epochs_progressive();
        } else {
            self.try_aggregate_epochs_legacy();
        }
    }

    /// Legacy aggregation: one MetricsMsg per rank per epoch.
    fn try_aggregate_epochs_legacy(&mut self) {
        let expected = self.active_count;
        let mut complete: Vec<usize> = self.epoch_buffer.iter()
            .filter(|(_, msgs)| msgs.len() >= expected)
            .map(|(epoch, _)| *epoch)
            .collect();
        complete.sort_unstable();
        for epoch in complete {
            if let Some(msgs) = self.epoch_buffer.remove(&epoch) {
                let bc_share = self.el_che.recent_batch_share();
                let metrics = aggregate_epoch_metrics(epoch, &msgs, &self.device_indices, &bc_share);
                if let Some(f) = &self.metrics_fn {
                    if let Err(e) = f(&metrics) {
                        eprintln!("  ddp: metrics_fn returned error (epoch {epoch}): {e}");
                    }
                }
                if let Some(tx) = &self.epoch_metrics_tx {
                    let _ = tx.send(metrics);
                }
                self.on_epoch_aggregated(epoch);
            }
        }
    }

    /// Progressive aggregation: check all pools, fire global event for completed ones.
    ///
    /// Only aggregates epoch N if no earlier epoch pool is still active.
    /// This prevents a fast GPU from streaming ahead and triggering Shutdown
    /// for the final epoch while the slow GPU is still processing earlier work.
    fn try_aggregate_epochs_progressive(&mut self) {
        // Collect completed epochs in order, stopping at first incomplete pool.
        // BTreeMap iterates in ascending key order, so if epoch 1's pool isn't
        // done, epoch 2 won't aggregate even if its pool is done.
        let mut completed: Vec<(usize, f64)> = Vec::new();
        for (&epoch, pool) in &self.chunk_pools {
            if pool.is_epoch_done() {
                completed.push((epoch, pool.epoch_elapsed_ms()));
            } else {
                break; // Earlier epoch not done: can't aggregate anything after it
            }
        }

        for (epoch, epoch_ms) in completed {
            self.chunk_pools.remove(&epoch);

            if let Some(msgs) = self.epoch_buffer.remove(&epoch) {
                let bc_share = self.el_che.recent_batch_share();
                let mut metrics = aggregate_epoch_metrics(epoch, &msgs, &self.device_indices, &bc_share);
                metrics.epoch_ms = epoch_ms;
                if let Some(f) = &self.metrics_fn {
                    if let Err(e) = f(&metrics) {
                        eprintln!("  ddp: metrics_fn returned error (epoch {epoch}): {e}");
                    }
                }
                if let Some(tx) = &self.epoch_metrics_tx {
                    let _ = tx.send(metrics);
                }
                crate::verbose!(
                    "  ddp: epoch {epoch} progressive complete | {:.0}ms",
                    epoch_ms,
                );
                self.on_epoch_aggregated(epoch);
            }
        }
    }

    /// Check if averaging should be triggered based on the current policy.
    pub fn should_average(&self) -> bool {
        // Don't re-trigger while a CPU averaging cycle is in progress.
        if !matches!(self.avg_state, CpuAvgState::Idle) {
            return false;
        }
        // Don't re-trigger NCCL averaging until all ranks have acknowledged
        // the previous SyncNow (sent at least one timing message since).
        if matches!(self.backend, AverageBackend::Nccl)
            && !self.nccl_ack.iter().all(|&a| a)
        {
            return false;
        }
        // Training complete: workers received Shutdown, skip stale averaging.
        if self.all_epochs_done() {
            return false;
        }
        // Collectives require all ranks. If any worker has exited,
        // skip averaging to prevent NCCL deadlock or channel disconnect.
        if self.active_count < self.world_size {
            return false;
        }
        // All ranks must have trained at least one batch since the last
        // sync. A rank at 0 steps is setting up a new chunk (blocked in
        // prefetch or batch loading) or idle in wait_for_epoch_plan.
        // Sending SyncNow to it would deadlock: the NCCL collective
        // blocks the participating rank's GPU while the zero-step rank
        // can't call AllReduce until its batch setup completes.
        if self.steps_since_avg.contains(&0) {
            return false;
        }
        match self.policy {
            ApplyPolicy::Sync => {
                self.steps_since_avg.iter().all(|&s| s >= 1)
            }
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                // Count-based trigger: fire when each rank completes its
                // scheduled `batch_counts[r]`. Timing feeds `batch_counts`
                // through `ElChe::recompute_batch_counts` so the next
                // cycle's schedule lands closer to the estimated wall
                // time, but it does NOT gate firing. Gating on
                // `anchor * smoothed_slow_ms` is structurally fragile:
                // the target derives from samples that only land when
                // the gate fires, so an upward spike in `smoothed_slow_ms`
                // (cold-start warmup, thermal throttle, GPU contention,
                // mid-run lazy init) can lock the target above achievable
                // wall time and deadlock the cohort indefinitely.
                let counts = self.el_che.batch_counts();
                self.steps_since_avg.iter().enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
        }
    }

    /// Throttle workers that have run too far ahead of the slowest rank.
    ///
    /// Sends [`ControlMsg::Throttle`] to any worker whose `steps_since_avg`
    /// exceeds the slowest rank's by more than `max_batch_diff`. The worker
    /// blocks until the next real command (averaging or shutdown).
    ///
    /// Tracks which ranks are already throttled to avoid sending duplicate
    /// Throttle messages (which would nest blocking loops in the worker).
    pub fn check_throttle(&mut self) {
        // NCCL cadence uses AllReduce as its coordination mechanism.
        // Throttle is an async/CPU concept: it blocks the fast worker waiting
        // for SyncNow, but if the slow worker is idle (between epochs),
        // should_average never fires and the throttled worker deadlocks.
        if matches!(self.backend, AverageBackend::Nccl) {
            return;
        }

        let max_diff = match self.el_che.max_batch_diff() {
            Some(d) => d,
            None => return,
        };

        if self.active_count < self.world_size {
            return; // some worker exited, don't throttle
        }

        let min_steps = self.steps_since_avg.iter().copied().min().unwrap_or(0);

        for (rank, &steps) in self.steps_since_avg.iter().enumerate() {
            let should_throttle = steps > min_steps + max_diff;
            if should_throttle && !self.throttled[rank] {
                let _ = self.control_txs[rank].send(ControlMsg::Throttle);
                self.throttled[rank] = true;
                if let Some(ref tl) = self.timeline {
                    tl.event(crate::monitor::EventKind::Throttle { rank });
                }
            }
        }
    }

    /// Run one tick of the coordinator loop.
    ///
    /// Drains timing/metrics, throttles fast workers, checks averaging.
    /// Returns collected metrics (if any) for external logging.
    pub fn tick(&mut self) -> Result<Vec<MetricsMsg>> {
        self.drain_timing();
        self.check_throttle();
        self.poll_cpu_averaging()?;
        let metrics = self.drain_metrics();

        if self.should_average() {
            // Final drain to catch last-second Exiting messages before
            // sending SyncNow (prevents AllReduce with a dead worker).
            self.drain_timing();
            if self.should_average() {
                self.trigger_averaging()?;
            }
        }

        Ok(metrics)
    }

    /// Collect final parameter snapshots from all workers after the main loop exits.
    ///
    /// Blocks on each dedicated `final_param_rx` channel with a timeout.
    /// Returns a [`TrainedState`] averaged from whatever snapshots arrived
    /// (partial failure: survivors' params are returned). Returns `None` if
    /// zero snapshots were collected.
    pub fn collect_final_state(&self) -> Option<TrainedState> {
        let timeout = std::time::Duration::from_secs(10);
        let mut snapshots = Vec::new();
        for (rank, rx) in self.final_param_rxs.iter().enumerate() {
            match rx.recv_timeout(timeout) {
                Ok(snap) => snapshots.push(snap),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    crate::verbose!("  ddp: timeout waiting for final snapshot from rank {rank}");
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    crate::verbose!("  ddp: rank {rank} channel disconnected (worker errored)");
                }
            }
        }
        if snapshots.is_empty() {
            return None;
        }
        // Average the final snapshots (reuse the existing averaging logic)
        match Self::average_params(&snapshots, self.version) {
            Ok(averaged) => Some(TrainedState {
                params: averaged.params,
                buffers: averaged.buffers,
            }),
            Err(_) => {
                // Fallback: return the first snapshot's tensors on CPU
                let snap = &snapshots[0];
                let params = snap.params.iter()
                    .filter_map(|t| t.to_device(Device::CPU).ok())
                    .collect();
                let buffers = snap.buffers.iter()
                    .filter_map(|t| t.to_device(Device::CPU).ok())
                    .collect();
                Some(TrainedState { params, buffers })
            }
        }
    }

    /// Send Shutdown to all workers so they can exit their
    /// `drain_until_shutdown` loop.
    pub fn shutdown_workers(&self) {
        for tx in &self.control_txs {
            let _ = tx.send(ControlMsg::Shutdown);
        }
    }
}

// ---------------------------------------------------------------------------
// Partition sizing helpers
// ---------------------------------------------------------------------------

/// Equal partition sizes with remainder distributed to the first ranks.
pub(crate) fn equal_sizes(world_size: usize, total: usize) -> Vec<usize> {
    let base = total / world_size;
    let remainder = total % world_size;
    (0..world_size)
        .map(|r| base + if r < remainder { 1 } else { 0 })
        .collect()
}

/// Throughput-proportional partition sizes from ElChe ms_per_batch.
///
/// Faster ranks (lower ms/batch) get more samples. Remainder distributed
/// to the fastest ranks.
pub(crate) fn throughput_sizes(
    el_che: &crate::distributed::ddp::ElChe,
    total: usize,
) -> Vec<usize> {
    let ms = el_che.ms_per_batch();
    // Inverse of ms_per_batch = throughput (batches/ms). Guard against zero.
    let throughputs: Vec<f64> = ms.iter().map(|&m| 1.0 / m.max(0.001)).collect();
    let total_tp: f64 = throughputs.iter().sum();
    if total_tp <= 0.0 {
        return equal_sizes(ms.len(), total);
    }
    let mut sizes: Vec<usize> = throughputs.iter()
        .map(|t| ((t / total_tp) * total as f64).floor() as usize)
        .collect();
    // Distribute remainder to fastest ranks (highest throughput first).
    let assigned: usize = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);
    if remaining > 0 {
        // Sort rank indices by throughput descending.
        let mut rank_order: Vec<usize> = (0..ms.len()).collect();
        rank_order.sort_by(|&a, &b| throughputs[b].partial_cmp(&throughputs[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &rank in &rank_order {
            if remaining == 0 { break; }
            sizes[rank] += 1;
            remaining -= 1;
        }
    }
    sizes
}

/// Convert user-specified ratios to absolute partition sizes.
///
/// Ratios are normalized to sum to 1.0. Remainder distributed to the
/// ranks with the largest ratios.
pub(crate) fn ratio_to_sizes(ratios: &[f64], total: usize) -> Vec<usize> {
    let sum: f64 = ratios.iter().sum();
    let norm: Vec<f64> = if sum > 0.0 {
        ratios.iter().map(|r| r / sum).collect()
    } else {
        vec![1.0 / ratios.len() as f64; ratios.len()]
    };
    let mut sizes: Vec<usize> = norm.iter()
        .map(|r| (r * total as f64).floor() as usize)
        .collect();
    let assigned: usize = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);
    if remaining > 0 {
        let mut rank_order: Vec<usize> = (0..ratios.len()).collect();
        rank_order.sort_by(|&a, &b| norm[b].partial_cmp(&norm[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &rank in &rank_order {
            if remaining == 0 { break; }
            sizes[rank] += 1;
            remaining -= 1;
        }
    }
    sizes
}

/// Aggregate per-rank `MetricsMsg` into a single `EpochMetrics`.
///
/// Loss and scalars are averaged weighted by batch count (proportional
/// to each rank's contribution). Epoch time is the max across ranks.
///
/// In progressive dispatch mode, each rank sends one [`MetricsMsg`] per
/// chunk (not per epoch), so there may be many more messages than ranks.
/// This function aggregates by rank first so the output always has
/// exactly `world_size` entries per vector.
/// `bc_share` is the smoothed per-rank batch-share view from the balancer
/// (i.e. `el_che.recent_batch_share()` averaged over the last few sync
/// snapshots). It replaces the prior "samples-consumed / total-samples"
/// share, which conflated cadence allocation with progressive dispatch
/// tail-balance dynamics. Caller is expected to pass `world_size` entries
/// summing to ~1.0; degenerate input falls back to equal shares.
pub(crate) fn aggregate_epoch_metrics(
    epoch: usize,
    msgs: &[MetricsMsg],
    device_indices: &[u8],
    bc_share: &[f64],
) -> super::EpochMetrics {
    let world_size = device_indices.len();

    // --- Step 1: Aggregate per-chunk messages by rank ---
    let mut rank_batches: Vec<usize> = vec![0; world_size];
    let mut rank_samples: Vec<usize> = vec![0; world_size];
    let mut rank_loss_sum: Vec<f64> = vec![0.0; world_size];
    let mut rank_time_ms: Vec<f64> = vec![0.0; world_size];
    let mut rank_share_complete_ms: Vec<f64> = vec![0.0; world_size];
    let mut rank_compute_only_ms: Vec<f64> = vec![0.0; world_size];
    let mut rank_data_starve_ms: Vec<f64> = vec![0.0; world_size];
    // Per-rank scalar accumulators: (sum, count) per key
    let mut rank_scalars: Vec<std::collections::HashMap<String, (f64, usize)>> =
        (0..world_size).map(|_| std::collections::HashMap::new()).collect();

    for m in msgs {
        let r = m.rank.min(world_size - 1);
        rank_batches[r] += m.batches_processed;
        rank_samples[r] += m.samples_processed;
        rank_loss_sum[r] += m.avg_loss * m.batches_processed as f64;
        // Max time across chunks (sequential within a rank). Mirrors
        // existing epoch_ms aggregation: chunk messages report monotonic
        // cumulative-from-epoch-start times in progressive dispatch, so the
        // largest is the rank's epoch total. The new fields use the same
        // convention so the worker can populate them consistently.
        rank_time_ms[r] = rank_time_ms[r].max(m.epoch_ms);
        rank_share_complete_ms[r] = rank_share_complete_ms[r].max(m.share_complete_ms);
        rank_compute_only_ms[r] = rank_compute_only_ms[r].max(m.compute_only_ms);
        rank_data_starve_ms[r] = rank_data_starve_ms[r].max(m.data_starve_ms);
        for (k, (sum, count)) in &m.scalars {
            let entry = rank_scalars[r].entry(k.clone()).or_insert((0.0, 0));
            entry.0 += sum;
            entry.1 += count;
        }
    }

    // --- Step 2: Compute aggregated metrics ---
    let total_batches: usize = rank_batches.iter().sum();

    // Batch-weighted average loss
    let avg_loss = if total_batches > 0 {
        rank_loss_sum.iter().sum::<f64>() / total_batches as f64
    } else {
        0.0
    };

    // Max epoch_ms across ranks
    let epoch_ms = rank_time_ms.iter().copied().fold(0.0_f64, f64::max);

    // Per-rank scalar means (each rank's sum/count)
    let per_rank: Vec<std::collections::HashMap<String, f64>> = rank_scalars
        .iter()
        .map(|scalars| {
            scalars
                .iter()
                .map(|(k, (sum, count))| {
                    (k.clone(), if *count > 0 { sum / *count as f64 } else { 0.0 })
                })
                .collect()
        })
        .collect();

    // Weighted-average scalars across ranks
    let mut scalars: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut weights: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (r, rank_sc) in rank_scalars.iter().enumerate() {
        let w = rank_batches[r] as f64;
        for (k, (sum, count)) in rank_sc {
            if *count > 0 {
                let mean = sum / *count as f64;
                *scalars.entry(k.clone()).or_default() += mean * w;
                *weights.entry(k.clone()).or_default() += w;
            }
        }
    }
    for (k, v) in &mut scalars {
        if let Some(w) = weights.get(k) {
            if *w > 0.0 {
                *v /= *w;
            }
        }
    }

    // Per-rank throughput (samples/ms) and batch share.
    //
    // Throughput uses share_complete_ms as the denominator, NOT epoch_ms.
    // epoch_ms includes any post-completion idle the fast rank spends
    // waiting at the sync barrier for slower ranks; dividing by it produces
    // an inverted tput signal (the fast rank looks slow because it idles
    // more), which feeds the balancer a signal that says "give the fast
    // rank less work" — exactly backwards. share_complete_ms = compute +
    // data-pipeline wait, measured per rank, excludes peer-induced idle,
    // so the resulting tput tracks the rank's actual capacity.
    //
    // Falls back to epoch_ms when share_complete_ms wasn't populated (legacy
    // call sites or test fixtures using the old MetricsMsg shape).
    let total_samples: usize = rank_samples.iter().sum();
    let per_rank_throughput: Vec<f64> = (0..world_size).map(|r| {
        let denom = if rank_share_complete_ms[r] > 0.0 {
            rank_share_complete_ms[r]
        } else {
            rank_time_ms[r]
        };
        if denom > 0.0 { rank_samples[r] as f64 / denom } else { 0.0 }
    }).collect();
    // Per-rank batch share comes from the balancer's smoothed view of its
    // own recent batch_counts allocation (`el_che.recent_batch_share()`),
    // not from samples consumed. Under progressive dispatch the latter is
    // equalized by tail-balance and obscures the cadence's actual ratios.
    // Degenerate input (wrong length, all zeros) falls back to equal shares.
    let per_rank_batch_share: Vec<f64> = if bc_share.len() == world_size {
        let sum: f64 = bc_share.iter().sum();
        if sum > 0.0 {
            bc_share.to_vec()
        } else {
            vec![1.0 / world_size as f64; world_size]
        }
    } else {
        vec![1.0 / world_size as f64; world_size]
    };
    let _ = total_samples; // retained above for per_rank_throughput

    super::EpochMetrics {
        epoch, scalars, per_rank, avg_loss, epoch_ms,
        per_rank_throughput, per_rank_batch_share,
        per_rank_share_complete_ms: rank_share_complete_ms,
        per_rank_compute_only_ms: rank_compute_only_ms,
        per_rank_data_starve_ms: rank_data_starve_ms,
        device_indices: device_indices.to_vec(),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
