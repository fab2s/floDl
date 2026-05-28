//! Epoch dispatch and progressive chunk-pool scheduling for
//! [`super::ClusterCoordinator`].

use crate::distributed::ddp_run::ApplyPolicy;
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::{Result, TensorError};

use super::ClusterCoordinator;

impl ClusterCoordinator {
    /// Compute per-rank partition sizes for one epoch.
    ///
    /// Priority order:
    /// 1. Explicit `partition_ratios` from the config (test rigs,
    ///    user override).
    /// 2. ElChe throughput-derived sizes once calibrated (or once a
    ///    `with_speed_hint` is set in the config).
    /// 3. Equal sizes (fallback at startup before ElChe has
    ///    observations).
    pub(super) fn compute_partition_sizes(&self) -> Vec<usize> {
        if let Some(ratios) = &self.partition_ratios {
            return crate::distributed::ddp_run::ratio_to_sizes(
                ratios,
                self.total_samples,
            );
        }
        match self.policy {
            ApplyPolicy::Sync => crate::distributed::ddp_run::equal_sizes(
                self.world_size,
                self.total_samples,
            ),
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                if self.el_che.is_calibrated() || self.el_che.has_speed_hint() {
                    crate::distributed::ddp_run::throughput_sizes(
                        &self.el_che,
                        self.total_samples,
                    )
                } else {
                    crate::distributed::ddp_run::equal_sizes(
                        self.world_size,
                        self.total_samples,
                    )
                }
            }
        }
    }

    /// Get (or lazily compute + cache) the per-rank plans for `epoch`.
    ///
    /// Caching guarantees every rank receives consistent
    /// `(partition_offset, partition_size)` even when [`Self::dispatch_epoch`]
    /// gets called twice for the same epoch (the second call returns
    /// the cached plans). Verbatim port of OLD
    /// `Coordinator::plans_for_epoch`.
    pub(super) fn plans_for_epoch(
        &mut self,
        epoch: usize,
    ) -> Vec<crate::distributed::wire::EpochPlanWire> {
        use crate::distributed::wire::EpochPlanWire;
        if let Some(plans) = self.epoch_plan_cache.get(&epoch) {
            return plans.clone();
        }
        let sizes = self.compute_partition_sizes();
        let mut plans: Vec<EpochPlanWire> = Vec::with_capacity(self.world_size);
        let mut offset: u64 = 0;
        for &size in &sizes {
            plans.push(EpochPlanWire {
                epoch: epoch as u64,
                partition_offset: offset,
                partition_size: size as u64,
            });
            offset += size as u64;
        }
        self.epoch_plan_cache.insert(epoch, plans.clone());
        plans
    }

    /// Broadcast `StartEpoch(plan)` to every connected rank, updating
    /// `rank_epoch[r]` to `epoch` for each rank as it goes out.
    ///
    /// In **non-progressive** mode, sends one `StartEpoch` per rank
    /// carrying that rank's full per-epoch partition; returns the
    /// plans dispatched. In **progressive** mode (set on the coord
    /// config) creates a
    /// [`crate::distributed::chunk_pool::ChunkPool`] for the epoch
    /// and dispatches the first chunk to every rank; subsequent
    /// chunks are dispatched from `drain_metrics_and_aggregate` as
    /// ranks report chunk completion. The returned `Vec` reflects only
    /// the FIRST chunk per rank in progressive mode (callers should
    /// consume it as such).
    pub fn dispatch_epoch(
        &mut self,
        epoch: usize,
    ) -> Result<Vec<crate::distributed::wire::EpochPlanWire>> {
        if self.total_samples == 0 {
            return Err(TensorError::new(
                "cluster_coordinator: dispatch_epoch requires total_samples > 0; \
                 set ClusterCoordinatorConfig::total_samples before constructing.",
            ));
        }
        // Push the current epoch-callback role to workers BEFORE
        // StartEpoch arrives, so the worker's autonomous epoch_fn
        // fire-check sees a definite role on the first transition.
        // No-op on subsequent calls until `epoch_role_dirty` flips
        // (Fastest re-resolve on rank death).
        self.broadcast_epoch_callback_role_if_dirty()?;
        // User checkpoint cadence: when entering epoch N (N > 0) and
        // `N % checkpoint_every == 0`, broadcast a `Checkpoint(N)` frame
        // before `StartEpoch`. Workers fire `checkpoint_fn(N, &model)`
        // on the rank selected by [`EpochCallbackPolicy`]; others have
        // `checkpoint_fn = None` and treat the frame as a no-op. The
        // version reflects "model state at the end of epoch N-1", which
        // matches the threaded-path semantic `(epoch + 1) % every == 0`
        // (where the `+1` is the same off-by-one as treating epoch as
        // a 0-indexed counter).
        if epoch > 0 {
            if let Some(every) = self.checkpoint_every {
                if every > 0 && epoch % every == 0 {
                    // Targeted dispatch: the coord's `checkpoint_role`
                    // is the sticky assignee; the worker no-ops unless
                    // `target_rank == self.rank`. Stays addressed to
                    // the SAME live rank across checkpoints until that
                    // rank fails or dies, at which point the
                    // controller fails over.
                    let target = self.checkpoint_role;
                    let msg = ControlMsgWire::Checkpoint {
                        version: epoch as u64,
                        target_rank: target as u64,
                    };
                    self.send_control(target, &msg)?;
                }
            }
            // Eval cadence: dispatch `ExecuteEvalCallback` to the
            // current `eval_role` when the boundary aligns with
            // `eval_every_epochs`. Targeted (parallels the
            // `Checkpoint` dispatch above): the role is sticky across
            // cadences, re-resolved only on rank death when policy
            // is `Fastest`.
            if let Some(every) = self.eval_every_epochs {
                if every > 0 && epoch % every == 0 {
                    // schedule_id derived from epoch for now (one eval
                    // per cadence); a richer scheduler would mint a
                    // monotonic counter to disambiguate concurrent
                    // dispatches.
                    let target = self.eval_role;
                    let msg = ControlMsgWire::ExecuteEvalCallback {
                        schedule_id: epoch as u64,
                        epoch: epoch as u64,
                        target_rank: target as u64,
                    };
                    self.send_control(target, &msg)?;
                }
            }
        }
        if self.progressive {
            return self.start_epoch_progressive(epoch);
        }
        let plans = self.plans_for_epoch(epoch);
        for (rank, plan) in plans.iter().enumerate() {
            let msg = ControlMsgWire::StartEpoch(plan.clone());
            self.send_control(rank, &msg)?;
            self.rank_epoch[rank] = epoch;
            // Snapshot per-rank monotonic batch counter so a future
            // dead-rank declaration can compute how many of this
            // epoch's samples rank `r` had already processed (and
            // therefore how many remain to redistribute via
            // `ExtendPartition`).
            self.last_step_count_at_epoch_start[rank] = self.last_step_count[rank];
        }
        Ok(plans)
    }

    /// Start a new epoch in progressive mode: create a
    /// [`crate::distributed::chunk_pool::ChunkPool`] and dispatch the
    /// first chunk to every rank. Returns the per-rank
    /// [`crate::distributed::wire::EpochPlanWire`] of those first chunks
    /// so callers can pair the call with rank-side acknowledgments in
    /// tests. Subsequent chunks are dispatched from
    /// `drain_metrics_and_aggregate` on receipt of each rank's per-chunk
    /// MetricsMsg.
    ///
    /// Aligns the pool total to a batch boundary. Sub-batch remainders
    /// can't form a full batch and are dropped (standard DataLoader
    /// behaviour) — without this `is_epoch_done` never fires when
    /// `total_samples % batch_size != 0`.
    pub(super) fn start_epoch_progressive(
        &mut self,
        epoch: usize,
    ) -> Result<Vec<crate::distributed::wire::EpochPlanWire>> {
        let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
        self.chunk_pools.insert(
            epoch,
            crate::distributed::chunk_pool::ChunkPool::new(
                epoch,
                batch_total,
                self.world_size,
            ),
        );
        let sizes: Vec<usize> = (0..self.world_size)
            .map(|r| self.compute_chunk_batches(r, epoch))
            .collect();
        crate::verbose!(
            "  ddp: epoch {epoch} progressive | initial chunks (batches) {sizes:?}"
        );
        let mut plans: Vec<crate::distributed::wire::EpochPlanWire> =
            Vec::with_capacity(self.world_size);
        for (rank, &batch_count) in sizes.iter().enumerate() {
            if let Some(plan) = self.dispatch_next_chunk_with_batches(
                rank, epoch, batch_count,
            )? {
                plans.push(plan);
            } else {
                // Rank received no work (e.g. world_size > batch_total).
                // Push an empty plan so the returned Vec has world_size
                // entries (callers / tests expect that shape).
                plans.push(crate::distributed::wire::EpochPlanWire {
                    epoch: epoch as u64,
                    partition_offset: 0,
                    partition_size: 0,
                });
            }
            self.last_step_count_at_epoch_start[rank] = self.last_step_count[rank];
        }
        Ok(plans)
    }

    /// Dispatch the next chunk to `rank` from the active pool. Called
    /// after a rank reports a chunk-complete MetricsMsg in progressive
    /// mode.
    ///
    /// Tries the rank's current epoch's pool first. If exhausted,
    /// streams ahead into the next epoch's pool (subject to the
    /// overshoot gate, which fires in `ApplyPolicy::Async` only —
    /// `Sync` reduces every batch so drift can't build, and `Cadence`
    /// uses the next AllReduce as its sole coordination layer per
    /// `feedback_nccl_no_overshoot_throttle`). Gated ranks are kicked
    /// back into motion by [`Self::wake_idle_ranks_in_progressive`]
    /// after the next averaging cycle.
    pub(super) fn dispatch_next_chunk(&mut self, rank: usize) {
        let epoch = self.rank_epoch[rank];
        if self.chunk_pools.get(&epoch).is_some_and(|p| p.remaining() > 0) {
            let batches = self.compute_chunk_batches(rank, epoch);
            if let Err(e) = self.dispatch_next_chunk_with_batches(
                rank, epoch, batches,
            ) {
                crate::verbose!(
                    "  ddp: dispatch_next_chunk(rank={rank}, epoch={epoch}) error: {e}"
                );
            }
            return;
        }

        // Current pool exhausted for this rank: stream ahead. Skip
        // past already-aggregated epochs (their pools were removed by
        // `drain_metrics_and_aggregate`); re-creating them here would
        // produce an orphan pool that blocks all future aggregation
        // (BTreeMap walk stops at the first incomplete pool).
        let first_live = self.last_aggregated_epoch.map_or(0, |agg| agg + 1);
        let next_epoch = (epoch + 1).max(first_live);
        if next_epoch >= self.num_epochs {
            return;
        }

        // Overshoot gate (Async only, both backends): don't dispatch
        // into a future epoch when the rank has streamed too far past
        // its planned batch count since the last averaging. The next
        // AllReduce / Update resets `steps_since_avg` inside
        // `finish_averaging_*` and `wake_idle_ranks_in_progressive`
        // re-dispatches the gated rank; a fast NCCL rank sitting in
        // `wait_for_epoch_plan` still processes `SyncNow` via
        // `dispatch_control` (the SyncAck flips `nccl_ack` back to
        // true), so the gate is deadlock-free on NCCL too.
        if matches!(self.policy, ApplyPolicy::Async) {
            let current_aggregated = self.last_aggregated_epoch
                .is_some_and(|agg| epoch <= agg);
            if !current_aggregated {
                let planned = self.el_che.batch_counts().get(rank).copied().unwrap_or(0);
                if planned > 0
                    && self.steps_since_avg[rank] >= planned + self.max_overshoot
                {
                    crate::debug!(
                        "  ddp: overshoot gate BLOCKED rank {rank} | steps={} planned={} overshoot={}",
                        self.steps_since_avg[rank], planned, self.max_overshoot,
                    );
                    return;
                }
            }
        }

        if !self.chunk_pools.contains_key(&next_epoch) {
            let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
            self.chunk_pools.insert(
                next_epoch,
                crate::distributed::chunk_pool::ChunkPool::new(
                    next_epoch,
                    batch_total,
                    self.world_size,
                ),
            );
            crate::verbose!("  ddp: streaming -> epoch {next_epoch} pool created");
        }
        let batches = self.compute_chunk_batches(rank, next_epoch);
        if let Err(e) = self.dispatch_next_chunk_with_batches(
            rank, next_epoch, batches,
        ) {
            crate::verbose!(
                "  ddp: dispatch_next_chunk(rank={rank}, next_epoch={next_epoch}) error: {e}"
            );
        }
    }

    /// Take `batches * batch_size` samples from the epoch's pool and
    /// dispatch a `StartEpoch` carrying the chunk slice. Returns the
    /// dispatched plan wire (for test callers) or `None` if the pool
    /// is exhausted / batches == 0.
    pub(super) fn dispatch_next_chunk_with_batches(
        &mut self,
        rank: usize,
        epoch: usize,
        batches: usize,
    ) -> Result<Option<crate::distributed::wire::EpochPlanWire>> {
        let samples = batches * self.batch_size;
        if samples == 0 {
            return Ok(None);
        }
        let (offset, actual_size) = match self.chunk_pools.get_mut(&epoch) {
            Some(pool) => match pool.take_chunk(samples, rank) {
                Some(v) => v,
                None => return Ok(None),
            },
            None => return Ok(None),
        };
        self.rank_epoch[rank] = epoch;
        let plan = crate::distributed::wire::EpochPlanWire {
            epoch: epoch as u64,
            partition_offset: offset as u64,
            partition_size: actual_size as u64,
        };
        let msg = ControlMsgWire::StartEpoch(plan.clone());
        self.send_control(rank, &msg)?;
        Ok(Some(plan))
    }

    /// Compute how many batches the next chunk for `rank` in `epoch`
    /// should contain. Cold-start (pre-calibration) uses a small probe
    /// chunk (~10% of dataset per rank, floored at 4 batches) so
    /// ElChe gets enough averaging events to stabilise quickly.
    /// Post-calibration uses throughput-proportional sizing with a
    /// `min_chunk_batches` floor.
    pub(super) fn compute_chunk_batches(&self, rank: usize, epoch: usize) -> usize {
        let pool = match self.chunk_pools.get(&epoch) {
            Some(p) => p,
            None => return 0,
        };
        let remaining_batches = pool.remaining() / self.batch_size;
        if remaining_batches == 0 {
            return 0;
        }
        if !self.el_che.is_calibrated() && !self.el_che.has_speed_hint() {
            // Probe: small equal chunks for fast calibration (~10%
            // per rank, min 4 batches).
            let probe = (self.total_samples
                / (self.world_size * 10 * self.batch_size))
                .max(4);
            return probe.min(remaining_batches);
        }
        // Calibrated: proportional to ElChe's throughput-derived batch counts.
        let counts = self.el_che.batch_counts();
        let total_counts: usize = counts.iter().sum();
        if total_counts == 0 {
            return remaining_batches.min(self.min_chunk_batches);
        }
        let ratio = counts.get(rank).copied().unwrap_or(0) as f64
            / total_counts as f64;
        let target = (remaining_batches as f64 * ratio).ceil() as usize;
        target.max(self.min_chunk_batches).min(remaining_batches)
    }

    /// Wake any progressive rank stalled in `wait_for_epoch_plan` after
    /// being held by the overshoot gate (or otherwise finished its last
    /// chunk with no in-flight work). Called from the tail of
    /// [`Self::finish_averaging_nccl`] / [`Self::finish_averaging_cpu`]
    /// once `steps_since_avg` has been reset — without this kick,
    /// nothing drives `dispatch_next_chunk` for the stalled rank (no
    /// MetricsMsg arrives from a rank sitting in `wait_for_epoch_plan`).
    ///
    /// Mirrors threaded `coordinator/cpu_avg.rs:376-387` / `:544-553`,
    /// and the post-epoch hook at the tail of
    /// `try_advance_or_shutdown_after_aggregate`. The
    /// dispatched `StartEpoch` queues after the just-broadcast
    /// `SyncNow` / `Update` in each rank's control stream, so a fast
    /// NCCL rank in `wait_for_epoch_plan` processes the collective
    /// first and then takes the new chunk via `pending_plan`.
    pub(super) fn wake_idle_ranks_in_progressive(&mut self) {
        if !self.progressive {
            return;
        }
        for rank in 0..self.world_size {
            if self.is_dead(rank) {
                continue;
            }
            let has_inflight = self.chunk_pools.values()
                .any(|p| p.in_flight(rank) > 0);
            if !has_inflight {
                self.dispatch_next_chunk(rank);
            }
        }
    }
}
