//! Coordinator: lightweight scheduling thread for DDP run mode.

use std::sync::mpsc;

use crate::tensor::{Device, Result};

use std::collections::{BTreeMap, HashMap};

use super::{
    ApplyPolicy, AverageBackend, TimingMsg, MetricsMsg,
    ParamSnapshot, ControlMsg, EpochPlan, TrainedState,
};

mod chunk_pool;
mod cpu_avg;

// Re-exported at `pub(super)` so `super::coordinator::ChunkPool` works in
// `ddp_run::tests`. ChunkPool is `pub` inside its (private) module — the
// effective visibility is set by this re-export, not by chunk_pool.rs.
pub(super) use chunk_pool::ChunkPool;
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
    /// (el_che.rs:292-308) actually fires on NCCL backend. Was previously
    /// always 0 because `last_avg_ms` is only populated by the CPU avg path.
    last_nccl_sync_ms: f64,

    // LR-aware meta-controller (Stage 2: observed but emission not yet
    // dispatched to nudge_anchor_down — that lands in Stage 3.)
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
    /// is constructed and held by the coordinator. Stage 1 of the rollout
    /// keeps the meta dormant (field present, no observations); later
    /// stages wire LR + verdict forwarding and action dispatch.
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
            // and any guard-driven adjustment that follows.
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::Custom {
                    label: format!("meta-nudge factor={factor:.3} anchor={old}->{new}"),
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

        // Overshoot gate: don't dispatch if rank has exceeded its planned
        // batch count by more than max_overshoot since the last sync.
        // Only applies when streaming AHEAD of a not-yet-aggregated epoch.
        // If the rank's current epoch is already aggregated, this is a normal
        // transition (all ranks completed), not overshoot.
        //
        // Skip for NCCL backend: overshoot is an async/CPU concept. NCCL
        // cadence uses AllReduce as its sole coordination mechanism; blocking
        // the fast GPU here forces it into wait_for_epoch_plan where it can't
        // send timing messages, leaving nccl_ack permanently false and
        // deadlocking should_average + check_throttle.
        if !matches!(self.backend, AverageBackend::Nccl) {
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
        if let Some(every) = self.checkpoint_every {
            if every > 0 && (epoch + 1) % every == 0 {
                if let Some(tx) = self.control_txs.first() {
                    let _ = tx.send(ControlMsg::Checkpoint { version: (epoch + 1) as u64 });
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
            // thread + CPU param bridge respectively).
            TimingMsg::Heartbeat { .. } => {}
            TimingMsg::SnapshotReady { .. } => {}
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
            ApplyPolicy::Cadence => {
                // Wall-time trigger: fire when the slowest rank has
                // accumulated enough compute time since the last sync.
                // This avoids estimation errors from batch-count prediction
                // (EMA lag, dead zone, rounding) that cause the fast GPU
                // to idle at AllReduce boundaries.
                let target = self.el_che.anchor_wall_ms();
                if target > 0.0 {
                    let min_wall = self.wall_ms_accum.iter()
                        .copied()
                        .fold(f64::MAX, f64::min);
                    return min_wall >= target;
                }
                // Fallback (uncalibrated): batch-count trigger with equal
                // counts until first timing measurement arrives.
                let counts = self.el_che.batch_counts();
                self.steps_since_avg.iter().enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
            ApplyPolicy::Async => {
                // Batch-count trigger: proportional to throughput.
                // Async benefits from overshooting — the divergence between
                // replicas provides implicit regularization (Local SGD).
                // Wall-time matching kills this by constraining the fast GPU.
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
pub(super) fn aggregate_epoch_metrics(
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
mod tests {
    use super::*;
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // Helper: create a Coordinator without CUDA or real GPU threads.
    // -----------------------------------------------------------------------

    fn make_test_coordinator(
        world_size: usize,
        policy: ApplyPolicy,
        total_samples: usize,
    ) -> Coordinator {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        for _ in 0..world_size {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, _crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(world_size, 10);
        Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            policy, AverageBackend::Nccl, world_size, total_samples, el_che,
        )
        .num_epochs(10)
        .build()
    }

    /// Variant that returns the timing sender so tests can inject TimingMsg.
    fn make_test_coordinator_with_channels(
        world_size: usize,
        policy: ApplyPolicy,
        total_samples: usize,
    ) -> (
        Coordinator,
        mpsc::Sender<TimingMsg>,
        Vec<mpsc::Receiver<ControlMsg>>,
    ) {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        let mut control_rxs = Vec::new();
        for _ in 0..world_size {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
            control_rxs.push(crx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(world_size, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            policy, AverageBackend::Nccl, world_size, total_samples, el_che,
        )
        .num_epochs(10)
        .build();
        (coord, timing_tx, control_rxs)
    }

    // -----------------------------------------------------------------------
    // compute_partition_sizes
    // -----------------------------------------------------------------------

    #[test]
    fn partition_sync_equal_sizes() {
        let coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        let sizes = coord.compute_partition_sizes();
        assert_eq!(sizes, vec![500, 500]);
    }

    #[test]
    fn partition_sync_uneven_remainder() {
        let coord = make_test_coordinator(3, ApplyPolicy::Sync, 100);
        let sizes = coord.compute_partition_sizes();
        // 100 / 3 = 33 remainder 1. First rank gets the extra.
        assert_eq!(sizes, vec![34, 33, 33]);
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn partition_cadence_uncalibrated_falls_back_to_equal() {
        // Before calibration, Cadence mode uses equal sizes.
        let coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        assert!(!coord.el_che.is_calibrated());
        assert!(!coord.el_che.has_speed_hint());
        let sizes = coord.compute_partition_sizes();
        assert_eq!(sizes, vec![500, 500]);
    }

    #[test]
    fn partition_cadence_calibrated_proportional() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Simulate calibration: rank 0 is 2x faster (5ms/batch vs 10ms/batch).
        // report_timing needs wall_ms and actual_batches.
        coord.el_che.report_timing(&[50.0, 100.0], &[10, 10], 5.0);
        assert!(coord.el_che.is_calibrated());
        let sizes = coord.compute_partition_sizes();
        // Rank 0 has throughput 1/5=0.2, rank 1 has throughput 1/10=0.1.
        // Proportions: 0.2/0.3 = 0.667, 0.1/0.3 = 0.333.
        // ~667 and ~333.
        assert!(sizes[0] > sizes[1], "fast rank should get more: {:?}", sizes);
        assert_eq!(sizes.iter().sum::<usize>(), 1000);
    }

    #[test]
    fn partition_with_user_ratios() {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        for _ in 0..3 {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, _crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(3, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            ApplyPolicy::Cadence, AverageBackend::Nccl, 3, 900, el_che,
        )
        .partition_ratios(Some(vec![3.0, 2.0, 1.0]))
        .num_epochs(5)
        .build();

        let sizes = coord.compute_partition_sizes();
        // Ratios 3:2:1 normalized -> 0.5, 0.333, 0.167. On 900 samples:
        // floor: 450, 300, 150 = 900. No remainder.
        assert_eq!(sizes, vec![450, 300, 150]);
        assert_eq!(sizes.iter().sum::<usize>(), 900);
    }

    // -----------------------------------------------------------------------
    // Free function edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn equal_sizes_single_worker() {
        let sizes = equal_sizes(1, 100);
        assert_eq!(sizes, vec![100]);
    }

    #[test]
    fn ratio_to_sizes_zero_sum() {
        // All zero ratios: falls back to equal.
        let sizes = ratio_to_sizes(&[0.0, 0.0], 100);
        assert_eq!(sizes, vec![50, 50]);
    }

    // -----------------------------------------------------------------------
    // should_average
    // -----------------------------------------------------------------------

    #[test]
    fn should_average_sync_triggers_when_all_have_one_step() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        // Initially no steps: should not average.
        assert!(!coord.should_average());

        // Only rank 0 has a step.
        coord.steps_since_avg[0] = 1;
        assert!(!coord.should_average());

        // Both ranks have a step: should trigger.
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_cadence_wall_time_trigger() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Calibrate ElChe so anchor_wall_ms() returns > 0.
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        assert!(coord.el_che.is_calibrated());
        let target = coord.el_che.anchor_wall_ms();
        assert!(target > 0.0, "anchor_wall_ms should be positive after calibration");

        // Satisfy preconditions: steps > 0 and nccl_ack = true.
        coord.steps_since_avg = vec![1, 1];
        coord.nccl_ack = vec![true, true];

        // Not enough wall time accumulated.
        coord.wall_ms_accum[0] = target * 0.5;
        coord.wall_ms_accum[1] = target * 0.5;
        assert!(!coord.should_average());

        // Both ranks reach the target.
        coord.wall_ms_accum[0] = target;
        coord.wall_ms_accum[1] = target;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_cadence_min_wall_governs() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        let target = coord.el_che.anchor_wall_ms();

        // Rank 0 has enough, rank 1 does not: min < target.
        coord.wall_ms_accum[0] = target * 2.0;
        coord.wall_ms_accum[1] = target * 0.3;
        assert!(!coord.should_average());
    }

    #[test]
    fn should_average_async_batch_count_trigger() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        // Uncalibrated ElChe: batch_counts are [10, 10] (equal, anchor=10).
        let counts = coord.el_che.batch_counts().to_vec();
        assert_eq!(counts, vec![10, 10]);

        // Rank 0 at 10, rank 1 at 9: not enough.
        coord.steps_since_avg[0] = 10;
        coord.steps_since_avg[1] = 9;
        assert!(!coord.should_average());

        // Both at target.
        coord.steps_since_avg[1] = 10;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_blocked_when_worker_exited() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // One worker exits.
        coord.active_count = 1;
        assert!(!coord.should_average(), "must not average with fewer active workers");
    }

    #[test]
    fn should_average_blocked_when_all_epochs_done() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // Mark all epochs aggregated (num_epochs=10, so epoch 9 is the last).
        coord.last_aggregated_epoch = Some(9);
        assert!(coord.all_epochs_done());
        assert!(!coord.should_average());
    }

    #[test]
    fn should_average_blocked_during_cpu_averaging() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // Simulate CPU averaging in progress.
        coord.avg_state = CpuAvgState::Collecting {
            snapshots: Vec::new(),
            received: vec![false; 2],
            deadline: Instant::now() + std::time::Duration::from_secs(5),
            start: Instant::now(),
            steps_snapshot: vec![0; 2],
            wall_ms_snapshot: vec![0.0; 2],
        };
        assert!(!coord.should_average());
    }

    // -----------------------------------------------------------------------
    // process_timing_msg
    // -----------------------------------------------------------------------

    #[test]
    fn process_timing_msg_batch_accumulates() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        assert_eq!(coord.steps_since_avg[0], 0);
        assert_eq!(coord.wall_ms_accum[0], 0.0);

        coord.process_timing_msg(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None });
        assert_eq!(coord.steps_since_avg[0], 1);
        assert!((coord.wall_ms_accum[0] - 10.0).abs() < 1e-9);
        assert!((coord.last_batch_ms[0] - 10.0).abs() < 1e-9);

        // Second message accumulates.
        coord.process_timing_msg(TimingMsg::Batch { rank: 0, batch_ms: 15.0, step_count: 2, param_norm: None, batch_loss: 0.1, sync_divergence: None });
        assert_eq!(coord.steps_since_avg[0], 2);
        assert!((coord.wall_ms_accum[0] - 25.0).abs() < 1e-9);
        assert!((coord.last_batch_ms[0] - 15.0).abs() < 1e-9);

        // Rank 1 is independent.
        assert_eq!(coord.steps_since_avg[1], 0);
        assert_eq!(coord.wall_ms_accum[1], 0.0);
    }

    #[test]
    fn process_timing_msg_exiting_decrements_active() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        assert_eq!(coord.active_count, 2);

        coord.process_timing_msg(TimingMsg::Exiting { rank: 0 });
        assert_eq!(coord.active_count, 1);

        // Saturating: double exit won't underflow.
        coord.process_timing_msg(TimingMsg::Exiting { rank: 0 });
        assert_eq!(coord.active_count, 0);
    }

    // -----------------------------------------------------------------------
    // on_rank_done / on_epoch_aggregated
    // -----------------------------------------------------------------------

    #[test]
    fn on_rank_done_async_dispatches_within_lookahead() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        // Disable progressive so on_rank_done does the per-rank dispatch.
        coord.progressive = false;

        // Rank 0 finishes epoch 0. Next is epoch 1. No aggregation yet,
        // so lookahead allows epoch 0 and 1 (next <= 1).
        coord.on_rank_done(0, 0);
        // Should have sent StartEpoch(1) to rank 0.
        let msg = control_rxs[0].try_recv();
        assert!(
            matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 1),
            "expected StartEpoch(1)"
        );
    }

    #[test]
    fn on_rank_done_async_blocks_beyond_lookahead() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;

        // Rank 0 finishes epoch 1 but no aggregation yet.
        // next = 2, within_lookahead = (2 <= 1) = false. Should NOT dispatch.
        coord.on_rank_done(0, 1);
        assert!(
            control_rxs[0].try_recv().is_err(),
            "should NOT dispatch epoch 2 beyond lookahead"
        );
        assert!(coord.rank_waiting[0], "rank should be marked waiting");
    }

    #[test]
    fn on_rank_done_sync_is_noop() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 1000);
        // Sync policy: on_rank_done does nothing (early return for non-Async).
        coord.on_rank_done(0, 0);
        assert!(control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn on_rank_done_last_epoch_is_noop() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;
        // num_epochs = 10, so finishing epoch 9 means next = 10 >= 10.
        coord.on_rank_done(0, 9);
        assert!(control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn on_epoch_aggregated_sends_shutdown_after_last_epoch() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 1000);
        // Aggregate the last epoch (num_epochs=10, so epoch 9).
        coord.on_epoch_aggregated(9);
        assert!(coord.all_epochs_done());
        // Both ranks should receive Shutdown.
        for rx in &control_rxs {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::Shutdown)),
                "expected Shutdown"
            );
        }
    }

    #[test]
    fn on_epoch_aggregated_dispatches_next_sync() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 100);
        // Disable progressive for Sync.
        coord.progressive = false;

        // Aggregate epoch 0: should dispatch epoch 1 to both ranks.
        coord.on_epoch_aggregated(0);
        assert_eq!(coord.last_aggregated_epoch, Some(0));
        for (rank, rx) in control_rxs.iter().enumerate() {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 1),
                "rank {rank}: expected StartEpoch(1)"
            );
        }
    }

    #[test]
    fn on_epoch_aggregated_unblocks_waiting_ranks_async() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;

        // Simulate: rank 0 finished epoch 1 and is waiting (beyond lookahead).
        coord.rank_epoch[0] = 1;
        coord.rank_waiting[0] = true;
        // Rank 1 is also stuck.
        coord.rank_epoch[1] = 1;
        coord.rank_waiting[1] = true;

        // Aggregate epoch 0: should unblock waiting ranks, dispatching epoch 2.
        coord.on_epoch_aggregated(0);
        for (rank, rx) in control_rxs.iter().enumerate() {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 2),
                "rank {rank}: expected StartEpoch(2)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Coordinator state accessors
    // -----------------------------------------------------------------------

    #[test]
    fn all_epochs_done_boundary() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 100);
        // num_epochs = 10. Not done until epoch 9 is aggregated.
        assert!(!coord.all_epochs_done());
        coord.last_aggregated_epoch = Some(8);
        assert!(!coord.all_epochs_done());
        coord.last_aggregated_epoch = Some(9);
        assert!(coord.all_epochs_done());
    }

    #[test]
    fn finish_averaging_nccl_resets_counters() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 5;
        coord.steps_since_avg[1] = 3;
        coord.wall_ms_accum[0] = 50.0;
        coord.wall_ms_accum[1] = 30.0;
        coord.throttled[0] = true;

        coord.finish_averaging_nccl();

        assert_eq!(coord.steps_since_avg, vec![0, 0]);
        assert_eq!(coord.wall_ms_accum, vec![0.0, 0.0]);
        assert!(!coord.throttled[0]);
        assert_eq!(coord.version, 1);
        assert_eq!(coord.avg_count, 1);
    }

    #[test]
    fn finish_averaging_cpu_subtracts_snapshots() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        // Simulate: at trigger time, steps were [5, 3], wall was [50, 30].
        // During averaging, rank 0 did 2 more batches (20ms), rank 1 did 1 (10ms).
        coord.steps_since_avg = vec![7, 4];
        coord.wall_ms_accum = vec![70.0, 40.0];

        let steps_snap = vec![5, 3];
        let wall_snap = vec![50.0, 30.0];
        coord.finish_averaging_cpu(12.5, &steps_snap, &wall_snap, None);

        // Residual = current - snapshot.
        assert_eq!(coord.steps_since_avg, vec![2, 1]);
        assert!((coord.wall_ms_accum[0] - 20.0).abs() < 1e-9);
        assert!((coord.wall_ms_accum[1] - 10.0).abs() < 1e-9);
        assert_eq!(coord.version, 1);
        assert_eq!(coord.avg_count, 1);
    }

    // -----------------------------------------------------------------------
    // NCCL divergence detection (trend-based, gentle guardrail)
    // -----------------------------------------------------------------------

    /// Helper: simulate one averaging interval with given weight-space divergence.
    fn run_interval_with_divergence(coord: &mut Coordinator, div0: f64, div1: f64) {
        coord.nccl_sync_divergence = vec![Some(div0), Some(div1)];
        coord.wall_ms_accum = vec![100.0, 200.0];
        coord.steps_since_avg = vec![10, 10];
        coord.finish_averaging_nccl();
    }

    #[test]
    fn divergence_trend_suppresses_overshoot_growth() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        let overshoot_before = coord.max_overshoot;

        // 3 intervals with rising weight-space divergence.
        // Interval 1: max_relative_delta = 0.05
        run_interval_with_divergence(&mut coord, 0.03, 0.05);
        // Interval 2: max_relative_delta = 0.13
        run_interval_with_divergence(&mut coord, 0.08, 0.13);
        // Interval 3: max_relative_delta = 0.23
        run_interval_with_divergence(&mut coord, 0.15, 0.23);

        // Rising trend detected -> 3rd interval suppresses growth.
        // Overshoot should have grown for intervals 1-2 (no trend yet)
        // but NOT for interval 3.
        let cap = (coord.total_samples / coord.batch_size.max(1) / 50).clamp(3, 10);
        let expected = (overshoot_before + 2).min(cap).min(coord.overshoot_ceiling);
        assert_eq!(coord.max_overshoot, expected,
            "3rd interval with rising trend should suppress growth");
    }

    #[test]
    fn divergence_no_trend_allows_overshoot_growth() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        let overshoot_before = coord.max_overshoot;

        // 3 intervals with flat low divergence (no trend).
        for _ in 0..3 {
            run_interval_with_divergence(&mut coord, 0.005, 0.005);
        }

        let cap = (coord.total_samples / coord.batch_size.max(1) / 50).clamp(3, 10);
        let expected = (overshoot_before + 3).min(cap).min(coord.overshoot_ceiling);
        assert_eq!(coord.max_overshoot, expected,
            "flat low divergence should allow normal overshoot growth");
    }

    #[test]
    fn divergence_single_spike_does_not_suppress() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        let overshoot_before = coord.max_overshoot;

        // Single high-divergence interval (no trend, need 3 for rising).
        run_interval_with_divergence(&mut coord, 0.5, 0.8);

        assert!(coord.max_overshoot > overshoot_before,
            "single measurement should not suppress growth");
    }

    #[test]
    fn divergence_never_shrinks_anchor() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        let anchor_before = coord.el_che.anchor();

        // 5 intervals with rising weight-space divergence.
        for i in 0..5 {
            let div = 0.05 + i as f64 * 0.03; // rising
            run_interval_with_divergence(&mut coord, div * 0.8, div);
        }

        // Anchor must NEVER decrease from divergence detection.
        // ElChe's overhead auto-tune may change it, but nudge_anchor_down
        // is not called from the divergence path.
        assert!(coord.el_che.anchor() >= anchor_before,
            "divergence must never shrink anchor: was {anchor_before}, now {}",
            coord.el_che.anchor());
    }

    #[test]
    fn divergence_accumulators_reset_each_interval() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.loss_accum = vec![5.0, 3.0];
        coord.loss_count = vec![100, 50];
        coord.nccl_sync_divergence = vec![Some(0.05), Some(0.03)];
        coord.wall_ms_accum = vec![100.0, 200.0];
        coord.steps_since_avg = vec![100, 50];

        coord.finish_averaging_nccl();

        assert_eq!(coord.loss_accum, vec![0.0, 0.0]);
        assert_eq!(coord.loss_count, vec![0, 0]);
        // nccl_sync_divergence is reset after averaging.
        assert_eq!(coord.nccl_sync_divergence, vec![None, None]);
    }

    // `divergence_history_capped_at_5` removed: the ring-buffer cap is
    // now an implementation detail of the concrete guard. Coverage moved
    // to `convergence::tests::trend_history_capped_at_5`.

    #[test]
    fn divergence_overshoot_ceiling_caps() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        coord.overshoot_ceiling = 3;
        coord.max_overshoot = 5;

        coord.wall_ms_accum = vec![100.0, 200.0];
        coord.steps_since_avg = vec![10, 10];
        coord.finish_averaging_nccl();

        assert!(coord.max_overshoot <= 3,
            "overshoot {} should be capped at ceiling 3", coord.max_overshoot);
    }

    #[test]
    fn process_timing_msg_accumulates_loss_and_norm() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        assert_eq!(coord.loss_accum[0], 0.0);
        assert_eq!(coord.loss_count[0], 0);

        coord.process_timing_msg(TimingMsg::Batch {
            rank: 0, batch_ms: 10.0, step_count: 1, param_norm: Some(5.5), batch_loss: 0.3, sync_divergence: None,
        });
        assert!((coord.loss_accum[0] - 0.3).abs() < 1e-9);
        assert_eq!(coord.loss_count[0], 1);

        coord.process_timing_msg(TimingMsg::Batch {
            rank: 0, batch_ms: 10.0, step_count: 2, param_norm: None, batch_loss: 0.2, sync_divergence: None,
        });
        assert!((coord.loss_accum[0] - 0.5).abs() < 1e-9);
        assert_eq!(coord.loss_count[0], 2);

        // SyncNow ack (batch_loss=0.0) is skipped.
        coord.process_timing_msg(TimingMsg::Batch {
            rank: 0, batch_ms: 0.0, step_count: 3, param_norm: None, batch_loss: 0.0, sync_divergence: None,
        });
        assert_eq!(coord.loss_count[0], 2); // unchanged

        assert_eq!(coord.loss_count[1], 0); // rank 1 independent
    }

    // -----------------------------------------------------------------------
    // SyncAck handling (regression guard: do not inflate steps_since_avg)
    // -----------------------------------------------------------------------

    /// SyncAck must NOT increment steps_since_avg, otherwise every NCCL sync
    /// inflates global_step by one batch per rank and the LR scheduler fires
    /// early. Real bug found 2026-04-13: with ~27 syncs/epoch * 2 ranks the
    /// inflation was 6.9% and a MultiStepLR milestone at total/2 fired at
    /// epoch 94 instead of 100.
    #[test]
    fn sync_ack_does_not_inflate_steps_since_avg() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);

        // Simulate 10 real batches on each rank.
        for step in 1..=10 {
            for rank in 0..2 {
                coord.process_timing_msg(TimingMsg::Batch {
                    rank, batch_ms: 5.0, step_count: step,
                    param_norm: None, batch_loss: 0.1, sync_divergence: None,
                });
            }
        }
        assert_eq!(coord.steps_since_avg, vec![10, 10]);
        let wall_before: Vec<f64> = coord.wall_ms_accum.clone();

        // Inject a SyncAck for each rank (post-NCCL ack).
        for rank in 0..2 {
            coord.process_timing_msg(TimingMsg::SyncAck {
                rank, step_count: 11, divergence: Some(0.01),
                post_norm: Some(1.0), pre_norm: Some(1.01),
            });
        }

        // The killer assertion: SyncAck must leave steps_since_avg untouched.
        assert_eq!(coord.steps_since_avg, vec![10, 10],
            "SyncAck must not be counted as a real batch");
        // Wall-time accumulator also untouched (it's per-batch wall time).
        assert_eq!(coord.wall_ms_accum, wall_before);
        // But last_step_count must reflect the post-sync step.
        assert_eq!(coord.last_step_count, vec![11, 11]);
        // Divergence captured.
        assert_eq!(coord.nccl_sync_divergence[0], Some(0.01));
        assert_eq!(coord.nccl_sync_divergence[1], Some(0.01));
    }

    /// SyncAck must satisfy nccl_ack so the next averaging cycle isn't blocked.
    #[test]
    fn sync_ack_satisfies_nccl_ack() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Snapshot the step counters at sync trigger time (rank0=5, rank1=5).
        coord.nccl_sync_step = vec![5, 5];
        coord.nccl_ack = vec![false, false];

        // Worker reports SyncAck with step_count=6 (one past snapshot).
        for rank in 0..2 {
            coord.process_timing_msg(TimingMsg::SyncAck {
                rank, step_count: 6, divergence: None,
                post_norm: None, pre_norm: None,
            });
        }
        assert_eq!(coord.nccl_ack, vec![true, true],
            "SyncAck with step_count > snapshot must flip nccl_ack");
    }
}
