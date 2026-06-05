//! Test-only constructors and accessors for
//! [`super::ClusterCoordinator`]. All methods are gated `#[cfg(test)]`
//! (or `pub(crate)` test-flavoured accessors used from `tests.rs`).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use crate::distributed::ddp_run::ApplyPolicy;
use crate::distributed::wire::{SessionSalt, TimingMsgWire};

use super::{
    ClusterCoordinator, ClusterCoordinatorConfig, CpuAvgState, NcclRendezvousPending,
    initial_callback_role,
};

impl ClusterCoordinator {
    /// `true` after [`Self::dispatch_shutdown_with_save`] has fired
    /// (max_failure threshold breached or backend hard limit hit).
    /// Test-only accessor; the flag is internal state.
    #[cfg(test)]
    pub(crate) fn shutdown_with_save_dispatched(&self) -> bool {
        self.shutdown_with_save_dispatched
    }

    /// Test-only peek at the current rendezvous-pending generator
    /// rank. `None` when no rendezvous is in flight (steady-state OR
    /// after exhaustion → ShutdownWithSave). Drives the retry-path
    /// tests that need to observe generator-rank transitions across
    /// successive ticks.
    #[cfg(test)]
    pub(crate) fn rendezvous_pending_generator(&self) -> Option<usize> {
        self.nccl_rendezvous_pending
            .as_ref()
            .map(|p| p.generator_rank)
    }

    /// Test-only peek at the list of generators already tried in the
    /// current rendezvous. Empty on the first attempt; grows by one
    /// each time [`Self::check_rendezvous_timeout`] retries.
    #[cfg(test)]
    pub(crate) fn rendezvous_tried_generators(&self) -> Vec<usize> {
        self.nccl_rendezvous_pending
            .as_ref()
            .map(|p| p.tried_generators.clone())
            .unwrap_or_default()
    }

    /// Test-only seam: install a synthetic pending rendezvous so the
    /// retry path can be unit-tested without a live NCCL setup.
    /// `initiated_offset_secs` shifts `initiated_at` into the past so
    /// `check_rendezvous_timeout` trips immediately on the next tick
    /// without sleeping.
    #[cfg(test)]
    pub(crate) fn test_seed_rendezvous_pending(
        &mut self,
        generator_rank: usize,
        survivors_ordered: Vec<usize>,
        initiated_offset_secs: u64,
    ) {
        let initiated_at = Instant::now()
            .checked_sub(Duration::from_secs(initiated_offset_secs))
            .unwrap_or_else(Instant::now);
        self.nccl_rendezvous_pending = Some(NcclRendezvousPending {
            generator_rank,
            survivors_ordered,
            initiated_at,
            tried_generators: Vec::new(),
        });
    }

    /// Test-only peek at the most-recent per-rank LR snapshot. `None`
    /// for ranks that have not yet sent a [`TimingMsgWire::LrUpdate`].
    #[cfg(test)]
    pub(crate) fn last_lr_per_rank_for_test(&self) -> &[Option<f64>] {
        &self.last_lr_per_rank
    }

    /// Test-only peek at whether the LR-aware meta-controller is
    /// active. Returns `true` when
    /// [`ClusterCoordinatorConfig::meta_controller`] was set.
    #[cfg(test)]
    pub(crate) fn meta_controller_enabled_for_test(&self) -> bool {
        self.lr_event_meta.is_some()
    }

    /// Test-only peek at the per-rank observed sync-lag history (ms).
    /// See [`Self::last_observed_sync_lag_ms`] for the caveat about
    /// barrier correlation.
    #[cfg(test)]
    pub(crate) fn last_observed_sync_lag_ms_for_test(&self) -> &[Option<f64>] {
        &self.last_observed_sync_lag_ms
    }

    /// Build a headless ClusterCoordinator for unit-testing internal
    /// state-machine logic without spinning up TCP listeners or
    /// reader threads. `control_streams` and `reader_handles` are
    /// empty — calls into [`Self::send_control`] return a benign
    /// `TensorError` instead of panicking, so the
    /// retry-redispatch path in
    /// [`Self::handle_checkpoint_result`] surfaces as a log line
    /// rather than crashing the test. Test fixtures that need to
    /// drive the full wire path should use `spawn_coord` /
    /// `start_from_listener` instead.
    #[cfg(test)]
    pub(crate) fn for_test(mut config: ClusterCoordinatorConfig) -> Self {
        let world_size = config.world_size;
        let salt: SessionSalt =
            [0u8; crate::distributed::wire::SESSION_SALT_BYTES];
        let (_timing_tx, timing_rx) = mpsc::channel::<TimingMsgWire>();
        let (_metrics_tx, metrics_rx) =
            mpsc::channel::<crate::distributed::wire::MetricsMsgWire>();
        let el_che = std::mem::replace(
            &mut config.el_che,
            crate::distributed::ddp::ElChe::new(world_size.max(1), 1),
        );
        let calibrated = config.start_elche_state.is_some()
            && el_che.is_calibrated();
        ClusterCoordinator {
            policy: config.policy,
            backend: config.backend,
            world_size,
            overshoot_initial: config.overshoot_initial,
            overshoot_ceiling: config.overshoot_ceiling,
            overshoot_auto: config.overshoot_auto,
            elche_relax_up: config.elche_relax_up,
            el_che,
            convergence_guard: config.convergence_guard,
            version: 0,
            avg_count: config.start_avg_count,
            global_step: config.start_global_step,
            calibrated,
            active_count: world_size,
            max_overshoot: config.overshoot_initial,
            steps_since_avg: vec![0; world_size],
            wall_ms_accum: vec![0.0; world_size],
            delivered_span_start: vec![None; world_size],
            delivered_span_crossed: vec![false; world_size],
            delivered_ms_accum: vec![0.0; world_size],
            delivered_batches_accum: vec![0; world_size],
            last_batch_ms: vec![0.0; world_size],
            last_step_count: vec![0; world_size],
            nccl_sync_step: vec![0; world_size],
            nccl_ack: vec![true; world_size],
            nccl_sync_divergence: vec![None; world_size],
            nccl_sync_pre_norm: vec![None; world_size],
            nccl_sync_post_norm: None,
            throttled: vec![false; world_size],
            dispatch_hold_logged: vec![false; world_size],
            last_nccl_sync_ms: 0.0,
            nccl_sync_start: None,
            epoch_d_min: f64::INFINITY,
            epoch_d_max: f64::NEG_INFINITY,
            epoch_d_sum: 0.0,
            epoch_d_count: 0,
            epoch_last_d: 0.0,
            epoch_last_k_max: 0,
            lr_event_meta: if config.meta_controller {
                Some(crate::distributed::lr_event_meta::LrEventMeta::with_default_config())
            } else {
                None
            },
            last_lr_per_rank: vec![None; world_size],
            cpu_avg_state: CpuAvgState::Idle,
            prof_enabled: crate::log::enabled(crate::log::Verbosity::Debug),
            stall_last_global_step: 0,
            stall_since: None,
            stall_last_dump: None,
            dead_ranks: config.dead_ranks,
            heartbeat_timeout_secs: config.heartbeat_timeout_secs,
            rendezvous_timeout_secs: config.rendezvous_timeout_secs,
            last_heartbeat: vec![Instant::now(); world_size],
            last_step_count_at_epoch_start: vec![0; world_size],
            nccl_rendezvous_pending: None,
            local_ranks: config.local_ranks.clone(),
            max_failure: config.max_failure,
            epoch_callback_policy: config.epoch_callback_policy,
            checkpoint_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            eval_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_callback_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_role_dirty: true,
            checkpoint_tried_ranks: std::collections::HashMap::new(),
            last_checkpoint_elapsed_ms_ewma: None,
            last_eval_elapsed_ms_ewma: None,
            last_epoch_fn_elapsed_ms_ewma: None,
            save_path: config.save_path.clone(),
            checkpoint_every: config.checkpoint_every,
            shutdown_with_save_dispatched: false,
            last_observed_sync_lag_ms: vec![None; world_size],
            last_observed_upload_ms: vec![None; world_size],
            rank_epoch: vec![0; world_size],
            last_aggregated_epoch: None,
            last_dispatched_epoch: None,
            shutdown_initiated: false,
            final_eval_dispatched: false,
            epoch_plan_cache: std::collections::HashMap::new(),
            total_samples: config.total_samples,
            batch_size: config.batch_size.max(1),
            num_epochs: config.num_epochs,
            partition_ratios: config.partition_ratios,
            timing_rx,
            metrics_rx,
            metrics_buffer: std::collections::BTreeMap::new(),
            chunk_pools: std::collections::BTreeMap::new(),
            progressive: config.progressive.unwrap_or(
                !matches!(config.policy, ApplyPolicy::Sync),
            ),
            min_chunk_batches: 4,
            metrics_fn: config.metrics_fn.clone(),
            metrics_sink_tx: config.metrics_sink_tx.clone(),
            eval_result_fn: config.eval_result_fn.clone(),
            eval_every_epochs: config.eval_every_epochs,
            metrics_device_indices: (0..world_size as u8).collect(),
            control_streams: Vec::new(),
            rank_to_conn: Vec::new(),
            reader_handles: Vec::new(),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            bound_port: 0,
            salt,
            timeline: config.timeline.clone(),
            sync_start: None,
            cpu_avg_start: None,
            dashboard_sink: config.dashboard_sink.clone(),
        }
    }

    /// Test-only mutator for `wall_ms_accum[rank]`. Used by
    /// `checkpoint_time_excluded_from_wall_ms_accum` to set a known
    /// starting state before invoking `handle_checkpoint_result`.
    #[cfg(test)]
    pub(crate) fn set_wall_ms_accum_for_test(&mut self, rank: usize, ms: f64) {
        self.wall_ms_accum[rank] = ms;
    }

    /// Test-only accessor for `wall_ms_accum[rank]`.
    #[cfg(test)]
    pub(crate) fn wall_ms_accum_for_test(&self, rank: usize) -> f64 {
        self.wall_ms_accum[rank]
    }

    /// Test-only accessor for `heartbeat_timeout_secs`.
    #[cfg(test)]
    pub(crate) fn heartbeat_timeout_secs(&self) -> u64 {
        self.heartbeat_timeout_secs
    }

    /// Test-only mutator: force a rank's `last_heartbeat` to an
    /// arbitrary `Instant`. Used by
    /// `checkpoint_role_failover_on_rank_death` to age out a rank.
    #[cfg(test)]
    pub(crate) fn set_last_heartbeat_for_test(
        &mut self,
        rank: usize,
        when: Instant,
    ) {
        self.last_heartbeat[rank] = when;
    }

    /// Test-only wrapper around the private `check_dead_ranks` so
    /// tests can drive the heartbeat-stale path directly without
    /// spinning up the tick loop.
    #[cfg(test)]
    pub(crate) fn check_dead_ranks_for_test(&mut self) {
        self.check_dead_ranks();
    }

    /// Test-only wrappers exposing the private Fastest-resolution
    /// helpers to the test module. The role accessors mirror the
    /// `checkpoint_role` public accessor for the other two roles
    /// (intentionally test-only since the public API surfaces are
    /// covered by the role-bearing wire messages workers see).
    #[cfg(test)]
    pub(crate) fn resolve_fastest_role_for_test(&self) -> usize {
        self.resolve_fastest_role()
    }
    #[cfg(test)]
    pub(crate) fn re_resolve_callback_roles_on_death_for_test(
        &mut self,
        dead_rank: usize,
    ) {
        self.re_resolve_callback_roles_on_death(dead_rank);
    }
    #[cfg(test)]
    pub(crate) fn set_callback_roles_for_test(
        &mut self,
        checkpoint: usize,
        eval: usize,
        epoch_cb: usize,
    ) {
        self.checkpoint_role = checkpoint;
        self.eval_role = eval;
        self.epoch_callback_role = epoch_cb;
    }
    #[cfg(test)]
    pub(crate) fn eval_role_for_test(&self) -> usize {
        self.eval_role
    }
    #[cfg(test)]
    pub(crate) fn epoch_callback_role_for_test(&self) -> usize {
        self.epoch_callback_role
    }
    #[cfg(test)]
    pub(crate) fn epoch_role_dirty_for_test(&self) -> bool {
        self.epoch_role_dirty
    }

    /// Test-only: install a `ChunkPool` for the given epoch so the
    /// `maybe_apply_callback_slack_for_next_cycle` path can read a
    /// known `remaining()`. Production code creates pools in
    /// `dispatch_epoch`; tests skip that scaffold.
    #[cfg(test)]
    pub(crate) fn install_chunk_pool_for_test(
        &mut self,
        epoch: usize,
        total_samples: usize,
    ) {
        self.chunk_pools.insert(
            epoch,
            crate::distributed::chunk_pool::ChunkPool::new(
                epoch,
                total_samples,
                self.world_size,
            ),
        );
    }

    /// Test-only: chunk size the dispatcher would hand `rank` in `epoch`.
    #[cfg(test)]
    pub(crate) fn compute_chunk_batches_for_test(&self, rank: usize, epoch: usize) -> usize {
        self.compute_chunk_batches(rank, epoch)
    }

    /// Test-only: the end-of-training final-consensus-reduce decision.
    #[cfg(test)]
    pub(crate) fn needs_final_consensus_reduce_for_test(&self) -> bool {
        self.needs_final_consensus_reduce()
    }

    /// Test-only: drive `rank_epoch[rank]` directly. Production sets
    /// it inside `dispatch_next_chunk_with_batches`.
    #[cfg(test)]
    pub(crate) fn set_rank_epoch_for_test(&mut self, rank: usize, epoch: usize) {
        self.rank_epoch[rank] = epoch;
    }

    /// Test-only wrapper around the private producer-side slack
    /// staging so tests can verify its effect on
    /// `el_che.pending_callback_slack_ms()` without driving an entire
    /// `finish_averaging_*` cycle.
    #[cfg(test)]
    pub(crate) fn maybe_apply_callback_slack_for_test(&mut self) {
        self.maybe_apply_callback_slack_for_next_cycle();
    }

    /// Test-only mutable accessor for the embedded `ElChe`. Used by
    /// slack-producer tests to drive `report_timing` calibrate the
    /// el_che before invoking the producer.
    #[cfg(test)]
    pub(crate) fn el_che_mut_for_test(
        &mut self,
    ) -> &mut crate::distributed::ddp::ElChe {
        &mut self.el_che
    }

    /// Test-only accessor for the embedded `ElChe`. Used by
    /// slack-producer tests to verify the pending slack vector was set.
    #[cfg(test)]
    pub(crate) fn el_che_for_test(&self) -> &crate::distributed::ddp::ElChe {
        &self.el_che
    }

    /// Test-only mutator for `steps_since_avg[rank]`. Mirrors
    /// [`Self::set_wall_ms_accum_for_test`] for gate-firing assertions
    /// that need a known per-rank step count without driving the full
    /// timing-message path.
    #[cfg(test)]
    pub(crate) fn set_steps_since_avg_for_test(&mut self, rank: usize, n: usize) {
        self.steps_since_avg[rank] = n;
    }

    /// Force every rank's `nccl_ack` to `acked`. Used to prove the CPU
    /// re-arm path is independent of `nccl_ack` (the NCCL-only token).
    pub(crate) fn set_all_nccl_ack_for_test(&mut self, acked: bool) {
        for a in &mut self.nccl_ack {
            *a = acked;
        }
    }

    /// Put the CPU averaging state machine into `Pending` (a cycle in
    /// flight). Used to assert the `cpu_avg_state` re-arm gate.
    pub(crate) fn set_cpu_avg_pending_for_test(&mut self) {
        self.cpu_avg_state = CpuAvgState::Pending;
    }
}
