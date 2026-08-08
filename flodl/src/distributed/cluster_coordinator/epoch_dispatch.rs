//! Epoch dispatch and progressive chunk-pool scheduling for
//! [`super::ClusterCoordinator`].

use crate::distributed::ddp_run::ApplyPolicy;
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::{Result, TensorError};

use super::{ClusterCoordinator, FinalWindowPlan};

/// One rank's pre-composed post-reduce `Update` payload: the folded
/// next-window chunk (Cadence atomic dispatch; `None` otherwise) plus
/// the rollback token to un-take it if the send fails. Composed by
/// [`ClusterCoordinator::compose_window_plans`] so dispatch policy
/// lives here, not in the transport file that ships the frames.
pub(super) struct PlannedUpdate {
    pub(super) rank: usize,
    pub(super) next_plan: Option<crate::distributed::wire::EpochPlanWire>,
    /// `rank_epoch[rank]` before the take — restored on rollback.
    pub(super) prev_epoch: usize,
}

impl ClusterCoordinator {
    /// Picks covered by one epoch: the whole pick space when
    /// `epoch_splits == 1`, otherwise this epoch's slice of a data pass.
    ///
    /// The single place the coordinator turns an epoch index into a
    /// length. Partition sizing, the chunk pool and the staging
    /// advisories all go through it, which is why the load-bearing
    /// `window <= epoch` cap keeps holding once an epoch is a slice: the
    /// bound simply follows the slice down.
    pub(super) fn epoch_samples(&self, epoch: usize) -> usize {
        crate::rng::epoch_split_span(epoch, self.epoch_splits.max(1), self.total_samples).1
    }

    /// Compute per-rank partition sizes for one epoch.
    ///
    /// Priority order:
    /// 1. Explicit `partition_ratios` from the config (test rigs,
    ///    user override).
    /// 2. ElChe throughput-derived sizes once calibrated (or once a
    ///    `with_speed_hint` is set in the config).
    /// 3. Equal sizes (fallback at startup before ElChe has
    ///    observations).
    pub(super) fn compute_partition_sizes(&self, epoch: usize) -> Vec<usize> {
        let epoch_samples = self.epoch_samples(epoch);
        if let Some(ratios) = &self.partition_ratios {
            return crate::distributed::ddp_run::ratio_to_sizes(ratios, epoch_samples);
        }
        match self.policy {
            ApplyPolicy::Sync => crate::distributed::ddp_run::equal_sizes(
                self.world_size,
                epoch_samples,
            ),
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                if self.el_che.is_calibrated() || self.el_che.has_speed_hint() {
                    crate::distributed::ddp_run::throughput_sizes(
                        &self.el_che,
                        epoch_samples,
                    )
                } else {
                    crate::distributed::ddp_run::equal_sizes(
                        self.world_size,
                        epoch_samples,
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
        let sizes = self.compute_partition_sizes(epoch);
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
    /// `ChunkPool` for the epoch
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
        // (Fastest re-resolve on rank death). Best-effort: a rank that
        // missed the role frame has a broken connection and is reaped by
        // heartbeat staleness; blocking the whole epoch on it would park
        // the healthy cohort.
        if let Err(e) = self.broadcast_epoch_callback_role_if_dirty() {
            crate::verbose!("  ddp: epoch-callback role broadcast incomplete: {e}");
        }
        // User checkpoint cadence: when entering epoch N (N > 0) and
        // `N % checkpoint_every == 0`, broadcast a `Checkpoint(N)` frame
        // before `StartEpoch`. Workers fire `checkpoint_fn(N, &model)`
        // on the rank selected by [`EpochCallbackPolicy`]; others have
        // `checkpoint_fn = None` and treat the frame as a no-op. The
        // version reflects "model state at the end of epoch N-1", which
        // matches the `(epoch + 1) % every == 0` checkpoint-cadence
        // semantic (where the `+1` is the same off-by-one as treating
        // epoch as a 0-indexed counter).
        if epoch > 0 {
            // Checkpoint fires on the cadence OR on a folded user intent
            // (`Worker::request_checkpoint`) — the intent is a request the
            // controller services at this coherent boundary, cleared once
            // folded.
            let checkpoint_by_cadence = self
                .checkpoint_every
                .is_some_and(|every| every > 0 && epoch.is_multiple_of(every));
            if checkpoint_by_cadence || self.pending_checkpoint_intent {
                self.pending_checkpoint_intent = false;
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
                // Best-effort: a missed checkpoint is a gap in the
                // checkpoint series, not a reason to halt training
                // (the role failover machinery re-targets on death).
                if let Err(e) = self.send_control(target, &msg) {
                    eprintln!(
                        "flodl ddp: checkpoint dispatch to rank {target} \
                         failed at epoch {epoch}: {e}"
                    );
                }
            }
            // Eval fires on the cadence (`eval_every_epochs`) OR on a folded
            // user intent (`Worker::request_eval`). Targeted to the current
            // `eval_role` (parallels the `Checkpoint` dispatch above): the role
            // is sticky across cadences, re-resolved only on rank death when
            // policy is `Fastest`.
            let eval_by_cadence = self
                .eval_every_epochs
                .is_some_and(|every| every > 0 && epoch.is_multiple_of(every));
            if eval_by_cadence || self.pending_eval_intent {
                self.pending_eval_intent = false;
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
                // Best-effort, same rationale as the checkpoint
                // dispatch above.
                if let Err(e) = self.send_control(target, &msg) {
                    eprintln!(
                        "flodl ddp: eval dispatch to rank {target} \
                         failed at epoch {epoch}: {e}"
                    );
                }
            }
        }
        if self.progressive {
            return self.start_epoch_progressive(epoch);
        }
        let plans = self.plans_for_epoch(epoch);
        for (rank, plan) in plans.iter().enumerate() {
            if self.is_dead(rank) {
                continue;
            }
            let msg = ControlMsgWire::StartEpoch(plan.clone());
            // Best-effort per rank: a fail-fast `?` here left every rank
            // after the broken connection without a StartEpoch — parked in
            // `wait_for_epoch_plan` for an epoch the coordinator believed
            // dispatched. The failed rank's connection is broken; heartbeat
            // staleness reaps it and `ExtendPartition` redistributes its
            // share.
            if let Err(e) = self.send_control(rank, &msg) {
                eprintln!(
                    "flodl ddp: StartEpoch({epoch}) send to rank {rank} \
                     failed: {e}; continuing with remaining ranks"
                );
                continue;
            }
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
    /// `ChunkPool` and dispatch the
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
        let batch_total = (self.epoch_samples(epoch) / self.batch_size) * self.batch_size;
        let span_sizes = self.reservation_span_sizes(batch_total);
        self.chunk_pools.insert(
            epoch,
            crate::distributed::chunk_pool::ChunkPool::new_with_spans(
                epoch,
                batch_total,
                &span_sizes,
            ),
        );
        // A tiny epoch (whole dataset < one window + crumb) is its own final
        // window: plan it before sizing so the per-rank sizes are coherent.
        self.refresh_final_window_plan(epoch);

        // Reservation advisories: each rank's deterministic run-stream
        // (this epoch's spans + the predicted next epoch's) for its
        // background stager. Best-effort — advisory frames never gate
        // dispatch, and a rank without a stager ignores them.
        self.emit_stage_advisories(epoch);

        let sizes: Vec<usize> = (0..self.world_size)
            .map(|r| self.compute_chunk_batches(r, epoch))
            .collect();
        crate::verbose!(
            "  ddp: epoch {} progressive | initial chunks (batches) {sizes:?}",
            crate::distributed::ddp_run::epoch_label(epoch, self.epoch_splits)
        );
        let mut plans: Vec<crate::distributed::wire::EpochPlanWire> =
            Vec::with_capacity(self.world_size);
        for (rank, &batch_count) in sizes.iter().enumerate() {
            // Best-effort per rank, same rationale as the non-progressive
            // loop above: a fail-fast `?` here escalated one broken
            // connection at kickoff into a fatal launcher Err — coordinator
            // thread gone, every healthy rank watchdog-fired 30s later.
            // The failed take is already rolled back inside
            // `dispatch_next_chunk_with_batches`; heartbeat staleness reaps
            // the rank and its span is redistributed.
            let plan = match self.dispatch_next_chunk_with_batches(
                rank, epoch, batch_count,
            ) {
                Ok(plan) => plan,
                Err(e) => {
                    eprintln!(
                        "flodl ddp: first chunk of epoch {epoch} send to \
                         rank {rank} failed: {e}; continuing with \
                         remaining ranks"
                    );
                    None
                }
            };
            if let Some(plan) = plan {
                plans.push(plan);
            } else {
                // Rank received no work (e.g. world_size > batch_total) or
                // its send failed. Push an empty plan so the returned Vec
                // has world_size entries (callers / tests expect that
                // shape).
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
        // Plan the epoch's final reduce window before sizing (no-op until the
        // pool is within one window of empty). The first rank dispatched this
        // window sees the window-start remainder and pins the split.
        self.refresh_final_window_plan(epoch);

        // ONE-CHUNK-IN-FLIGHT INVARIANT (barrier-paced: Sync/Cadence). The
        // worker's `pending_plan` is a single slot (a second `StartEpoch`
        // silently overwrites the first), so a rank must never have two
        // chunks outstanding at once. Without this guard, atomic-dispatch
        // races the completion path: `finish_averaging_cpu` folds the next
        // chunk (in_flight > 0) AND resets `steps_since_avg` to 0, then
        // the just-finished pre-reduce chunk's `MetricsMsg` arrives and
        // `dispatch_next_chunk` sees `steps == 0 < budget` (reduce barrier
        // below passes) and dispatches a SECOND chunk — the worker drops
        // one, its samples stay `dispatched`-but-never-`completed`, so
        // `in_flight` sticks, `is_epoch_done` never fires, and the epoch
        // wedges. Mirrors the in-flight guard `wake_idle_ranks_in_progressive`
        // already applies. Keyed on the PACING policy, NOT the backend: the
        // single-slot `pending_plan` is the same for NCCL and CPU workers, so
        // the invariant holds for NCCL `Cadence` too. cpu-async is exempt: it
        // intentionally overruns via `max_overshoot` (bounded lookahead).
        if self.policy.is_barrier_paced()
            && self.chunk_pools.values().any(|p| p.in_flight(rank) > 0)
        {
            if !self.dispatch_hold_logged[rank] {
                self.dispatch_hold_logged[rank] = true;
                crate::debug!(
                    "  ddp: in-flight HOLD rank {rank} | already has an outstanding chunk"
                );
            }
            return;
        }

        // REDUCE BARRIER. The controller is the single scheduler for the
        // "one logical GPU partitioned into heterogeneous per-rank step
        // counts" model: it never hands out a step that crosses a
        // barrier. A rank that has produced its full step budget since
        // the last reduce gets nothing more until the reduce resets
        // `steps_since_avg` (then `finish_averaging_cpu` /
        // `wake_idle_ranks_in_progressive` / the post-aggregate hook
        // re-dispatch it). Budget = `counts[rank]` for Sync/Cadence
        // (hard); `counts[rank] + max_overshoot` for cpu-async — the one
        // mode allowed to overrun, bounded for now by the single
        // `max_overshoot` knob (a future `max_overshoot_epoch` may split
        // the reduce and epoch allowances). Applies to BOTH backends:
        // the reduce is coordinator-triggered, so NCCL needs this
        // software barrier exactly like CPU (see `reduce_step_budget`).
        let reduce_budget = self.reduce_step_budget(rank);
        if reduce_budget > 0 && self.window.steps(rank) >= reduce_budget {
            // Log once per HOLD episode (deduped via `dispatch_hold_logged`,
            // cleared at the reduce reset) — this branch re-fires every
            // dispatch attempt, so an unguarded log floods at ~150k lines/s.
            if !self.dispatch_hold_logged[rank] {
                self.dispatch_hold_logged[rank] = true;
                crate::debug!(
                    "  ddp: reduce barrier HOLD rank {rank} | steps={} budget={}",
                    self.window.steps(rank), reduce_budget,
                );
            }
            return;
        }

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

        // EPOCH BARRIER (barrier-paced: Sync/Cadence). `epoch >= first_live`
        // means this rank sits at the active epoch's edge and the next chunk
        // would cross into a not-yet-aggregated epoch. Sync/Cadence forbid the
        // crossing outright — the next epoch is dispatched by
        // `try_advance_or_shutdown_after_aggregate` once the current
        // epoch's reduces complete. cpu-async is allowed to cross,
        // bounded by the same `max_overshoot` budget already enforced by
        // the reduce-barrier check above (so reaching here means async is
        // within budget). A rank merely catching up to the active epoch
        // (`epoch < first_live`) is not crossing a barrier and proceeds.
        // Keyed on the PACING policy, NOT the backend: in cadence every rank
        // is EXACTLY one epoch (the reduce IS the epoch-coherence mechanism),
        // so NCCL `Cadence` must hold this barrier too — the reduce is
        // coordinator-triggered, so the collective does not pace the fast
        // rank, and exempting NCCL let it stream across every epoch and wedge.
        if self.policy.is_barrier_paced() && epoch >= first_live {
            // Deduped like the reduce-barrier HOLD above: at every epoch
            // tail this branch re-fires per dispatch attempt for each held
            // rank — unguarded it flooded a 200ep -vvv run with 43M lines
            // (99.99% of the log) and stole coordinator tick CPU.
            if !self.dispatch_hold_logged[rank] {
                self.dispatch_hold_logged[rank] = true;
                crate::debug!(
                    "  ddp: epoch barrier HOLD rank {rank} | epoch={epoch} first_live={first_live}"
                );
            }
            return;
        }

        if !self.chunk_pools.contains_key(&next_epoch) {
            let batch_total =
                (self.epoch_samples(next_epoch) / self.batch_size) * self.batch_size;
            let span_sizes = self.reservation_span_sizes(batch_total);
            self.chunk_pools.insert(
                next_epoch,
                crate::distributed::chunk_pool::ChunkPool::new_with_spans(
                    next_epoch,
                    batch_total,
                    &span_sizes,
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
        let prev_epoch = self.rank_epoch[rank];
        let Some(plan) = self.take_next_chunk_plan(rank, epoch, batches) else {
            return Ok(None);
        };
        let msg = ControlMsgWire::StartEpoch(plan.clone());
        if let Err(e) = self.send_control(rank, &msg) {
            // TRANSACTIONAL ROLLBACK: the take mutated the pool (and the
            // rank's epoch bookkeeping) before the send. Left
            // in place after a failed send, the taken samples would stay
            // dispatched-but-never-completed — a ghost in-flight chunk
            // that permanently wedges `is_epoch_done` and the reduce gate
            // off ONE transient write error. Undo everything the take did,
            // then surface the error (the rank's connection is broken; if
            // it stays broken, heartbeat staleness declares it dead and
            // `forfeit` keeps the epoch moving).
            self.rollback_chunk_take(rank, epoch, &plan, prev_epoch);
            return Err(e);
        }
        Ok(Some(plan))
    }

    /// Undo a [`Self::take_next_chunk_plan`] whose dispatch could not be
    /// delivered: the pool take and `rank_epoch` are restored to their
    /// pre-take state.
    pub(super) fn rollback_chunk_take(
        &mut self,
        rank: usize,
        epoch: usize,
        plan: &crate::distributed::wire::EpochPlanWire,
        prev_epoch: usize,
    ) {
        if let Some(pool) = self.chunk_pools.get_mut(&epoch) {
            pool.rollback_take(
                rank,
                plan.partition_offset as usize,
                plan.partition_size as usize,
            );
        }
        self.rank_epoch[rank] = prev_epoch;
    }

    /// Take `batches * batch_size` samples from `epoch`'s pool for
    /// `rank`, advance `rank_epoch`, and build the `EpochPlanWire` —
    /// **without sending anything**. Returns `None` if `batches == 0`
    /// or the pool is exhausted.
    ///
    /// Two callers ship the resulting plan differently:
    /// [`Self::dispatch_next_chunk_with_batches`] wraps it in a
    /// `StartEpoch` control frame; the atomic-dispatch path
    /// ([`Self::compose_window_plans`], consumed by
    /// `finish_averaging_cpu`) folds it into the post-reduce `Update`
    /// frame so the rank starts its next window without a separate
    /// control round-trip.
    pub(super) fn take_next_chunk_plan(
        &mut self,
        rank: usize,
        epoch: usize,
        batches: usize,
    ) -> Option<crate::distributed::wire::EpochPlanWire> {
        let samples = batches * self.batch_size;
        if samples == 0 {
            return None;
        }
        let (offset, actual_size) =
            self.chunk_pools.get_mut(&epoch)?.take_chunk(samples, rank)?;
        self.rank_epoch[rank] = epoch;
        Some(crate::distributed::wire::EpochPlanWire {
            epoch: epoch as u64,
            partition_offset: offset as u64,
            partition_size: actual_size as u64,
        })
    }

    /// Pre-compose the post-reduce dispatch plans for every live rank —
    /// the dispatch-policy half of the CPU finalize, computed HERE so
    /// the transport file (`cycle_cpu.rs`) only ships frames and never
    /// composes plans (audit I2). Each entry carries its own rollback
    /// token ([`PlannedUpdate::prev_epoch`]) so a failed `Update` send
    /// un-takes exactly that rank's chunk via
    /// [`Self::rollback_planned_update`].
    ///
    /// Resets the per-window step counters FIRST: the fold sizes the
    /// next chunk via `compute_chunk_batches`, whose
    /// `cap_to_reduce_budget` reads `steps_since_avg`. With the reset
    /// deferred until after the fold (the old order), the cap saw the
    /// JUST-CLOSED window's full step count, `budget_remaining`
    /// bottomed out at the `.max(1)` floor, and every epoch tail folded
    /// a 1-batch chunk — off-schedule, and a fill-cost-inflated 1-batch
    /// sample in the delivered feed. The reduce that brought us here IS
    /// the window boundary; the new window's budget is fresh.
    ///
    /// Only [`ApplyPolicy::Cadence`] folds chunks (atomic dispatch);
    /// other policies get `next_plan: None` entries so the `Update`
    /// fan-out still reaches every live rank. Dead ranks get no entry
    /// at all: their controller-side stream is shut, and folding a
    /// chunk for them would ghost it.
    ///
    /// Per-rank takes are independent (per-rank reservation spans), so
    /// composing all plans before any send is equivalent to the old
    /// interleaved take-send loop — same 0..world_size take order,
    /// same per-rank rollback on failure.
    pub(super) fn compose_window_plans(&mut self) -> Vec<PlannedUpdate> {
        self.window.reset_steps();
        let fold = matches!(self.policy, ApplyPolicy::Cadence);
        let mut planned = Vec::with_capacity(self.world_size);
        for rank in 0..self.world_size {
            if self.is_dead(rank) {
                continue;
            }
            let prev_epoch = self.rank_epoch[rank];
            let next_plan = if fold {
                self.fold_next_chunk_for_rank(rank)
            } else {
                None
            };
            planned.push(PlannedUpdate { rank, next_plan, prev_epoch });
        }
        planned
    }

    /// Undo one [`PlannedUpdate`] whose `Update` frame could not be
    /// delivered: the folded chunk (if any) is returned to the pool and
    /// the rank's epoch bookkeeping restored. No-op for a plan-less
    /// entry.
    pub(super) fn rollback_planned_update(&mut self, planned: &PlannedUpdate) {
        if let Some(plan) = &planned.next_plan {
            self.rollback_chunk_take(
                planned.rank,
                plan.epoch as usize,
                plan,
                planned.prev_epoch,
            );
        }
    }

    /// atomic-dispatch: compute + take `rank`'s next reduce-window chunk
    /// to fold into the post-reduce `Update` frame, or `None` to defer
    /// to the existing dispatch path.
    ///
    /// **Intra-epoch only**: returns a chunk only when the rank's
    /// *current* epoch pool still has work. At an epoch boundary
    /// (`remaining() == 0`) it returns `None` so the epoch-advance path
    /// (`try_advance_or_shutdown_after_aggregate`) dispatches the next
    /// epoch's first chunk, keeping atomic-dispatch out of the
    /// epoch-aggregation flow. No reduce-barrier check is applied: the
    /// reduce that just completed makes a fresh window available for
    /// every rank by construction (`steps_since_avg` is reset to 0 in
    /// the same `finish_averaging_cpu`). Chunk size is the next cycle's
    /// schedule (`compute_chunk_batches` reads the post-`report_timing`
    /// `batch_counts`).
    pub(super) fn fold_next_chunk_for_rank(
        &mut self,
        rank: usize,
    ) -> Option<crate::distributed::wire::EpochPlanWire> {
        if self.is_dead(rank) {
            return None;
        }
        let epoch = self.rank_epoch[rank];
        if self.chunk_pools.get(&epoch).is_none_or(|p| p.remaining() == 0) {
            return None;
        }
        self.refresh_final_window_plan(epoch);
        let batches = self.compute_chunk_batches(rank, epoch);
        self.take_next_chunk_plan(rank, epoch, batches)
    }

    /// Resume kickoff for a coverage-granular checkpoint: reconstruct the
    /// recorded in-progress epoch pools and dispatch only the uncovered
    /// remainder, instead of a fresh full epoch. Returns `Ok(true)` when it
    /// handled the kickoff (caller skips `dispatch_epoch`); `Ok(false)` when
    /// there is no coverage to resume from (fresh run, or a non-progressive /
    /// Sync run that has no chunk pools) so the caller falls back to
    /// `dispatch_epoch(start_epoch)`.
    ///
    /// Errors loudly if the recorded `seed` differs from the run's `seed`: a
    /// different shuffle re-randomizes the index space, so the recorded
    /// uncovered offsets would map to different samples and resume would both
    /// repeat covered data and skip uncovered data. The whole contract rests
    /// on reconstructing the SAME permutation (see
    /// [`crate::distributed::ddp_run::SHUFFLE_BASE_SEED`]).
    pub fn resume_progressive_from_coverage(&mut self) -> Result<bool> {
        let Some(coverage) = self.start_coverage.take() else {
            return Ok(false);
        };
        if !self.progressive {
            // Coverage-granular resume is a progressive-mode (Cadence/Async)
            // feature; a Sync run has no chunk pools to reconstruct.
            return Ok(false);
        }
        if coverage.seed != self.seed {
            return Err(TensorError::new(&format!(
                "cluster_coordinator: resume coverage seed {} != run seed {} — the \
                 epoch permutation would differ, so recorded uncovered offsets map \
                 to different samples (resume would repeat covered data and skip \
                 uncovered data). Resume with the same seed.",
                coverage.seed, self.seed,
            )));
        }
        if coverage.epoch_splits != self.epoch_splits {
            return Err(TensorError::new(&format!(
                "cluster_coordinator: resume coverage epoch_splits {} != run \
                 epoch_splits {} — an epoch covers a different slice of the pass, so \
                 the recorded uncovered offsets map to different samples exactly as a \
                 changed seed would (resume would repeat covered data and skip \
                 uncovered data). Resume with the same epoch_splits.",
                coverage.epoch_splits, self.epoch_splits,
            )));
        }
        if coverage.per_epoch.is_empty() {
            // Nothing was in progress at the snapshot (saved exactly on a clean
            // epoch boundary): fall back to fresh dispatch of `start_epoch`.
            return Ok(false);
        }
        // Reconstruct each in-progress epoch's pool to its recorded holes.
        for ec in &coverage.per_epoch {
            let pool = crate::distributed::chunk_pool::ChunkPool::from_coverage(
                ec.epoch,
                ec.total_samples,
                self.world_size,
                &ec.uncovered_ranges,
            );
            self.chunk_pools.insert(ec.epoch, pool);
        }
        // The lowest recorded epoch is the active one; everything before it is
        // fully covered (aggregated). Anchor `last_aggregated_epoch` there so
        // the streaming `first_live` math (`last_aggregated + 1`) lets ranks
        // advance past the resumed epochs but not re-enter completed ones.
        let lowest = coverage
            .per_epoch
            .iter()
            .map(|ec| ec.epoch)
            .min()
            .expect("non-empty per_epoch checked above");
        self.last_aggregated_epoch = lowest.checked_sub(1);
        crate::verbose!(
            "  ddp: resume from coverage | epochs {:?} reconstructed, active epoch {lowest}",
            coverage.per_epoch.iter().map(|e| e.epoch).collect::<Vec<_>>(),
        );
        // Dispatch the first chunk of the active epoch to every live rank; the
        // streaming path (drain_metrics / wake_idle_ranks_in_progressive) takes
        // over from there, including advancing into the next recorded epoch.
        for rank in 0..self.world_size {
            if self.is_dead(rank) {
                continue;
            }
            self.rank_epoch[rank] = lowest;
            self.last_step_count_at_epoch_start[rank] = self.last_step_count[rank];
            self.dispatch_next_chunk(rank);
        }
        Ok(true)
    }

    /// Snapshot data coverage across every in-progress epoch pool into a
    /// [`crate::distributed::CoverageBlock`] for a checkpoint. Called at the
    /// reduce that takes the checkpoint, so the recorded coverage is exactly
    /// "what the just-forged consensus has trained against" — in-flight chunks
    /// are reported uncovered (see
    /// [`crate::distributed::chunk_pool::ChunkPool::uncovered_ranges`]) and
    /// re-dispatched as first-coverage on resume. Records the run's `seed` so
    /// resume can verify the same permutation space.
    pub(super) fn snapshot_coverage(&self) -> crate::distributed::CoverageBlock {
        use crate::distributed::{CoverageBlock, EpochCoverage};
        let per_epoch = self
            .chunk_pools
            .iter()
            .map(|(&epoch, pool)| EpochCoverage {
                epoch,
                total_samples: pool.total_samples,
                uncovered_ranges: pool.uncovered_ranges(),
            })
            .collect();
        CoverageBlock {
            seed: self.seed,
            epoch_splits: self.epoch_splits,
            batch_size: self.batch_size,
            per_epoch,
        }
    }

    /// Refresh [`Self::final_window_plan`] for `epoch` (barrier-paced only).
    ///
    /// A no-op unless the pool has dropped to within one window of empty
    /// (`remaining < Σ batch_counts + world_size`), in which case the WHOLE
    /// remainder is the epoch's final reduce window. The plan is computed
    /// ONCE — the first dispatch call of that window, when the pool still
    /// holds the window-start remainder — and cached keyed by epoch, so the
    /// per-rank, pool-draining dispatch loop (CPU fold loop / NCCL wake loop,
    /// both size one rank then take from the shared pool) reads a consistent
    /// split instead of shearing it into degenerate scraps. Cleared (set
    /// `None`) outside the final-window regime, so normal windows fall through
    /// to per-rank [`Self::compute_chunk_batches`] sizing. Idempotent within a
    /// window; recomputed when `epoch` advances.
    pub(super) fn refresh_final_window_plan(&mut self, epoch: usize) {
        if !self.policy.is_barrier_paced() {
            self.final_window_plan = None;
            return;
        }
        let Some(pool) = self.chunk_pools.get(&epoch) else {
            self.final_window_plan = None;
            return;
        };
        let rem = pool.remaining() / self.batch_size;
        let counts = self.el_che.batch_counts();
        let total: usize = (0..self.world_size)
            .filter(|&r| !self.is_dead(r))
            .map(|r| counts.get(r).copied().unwrap_or(0))
            .sum();
        // Final window iff the remainder fits in one schedule plus a
        // sub-cohort crumb (`< Σcounts + world_size`). Outside that, a normal
        // window dispatches `Σcounts` and leaves `>= world_size` behind, so
        // the per-rank path handles it.
        if rem == 0 || total == 0 || rem >= total + self.world_size {
            self.final_window_plan = None;
            return;
        }
        // Already planned for this epoch AND the schedule it was computed
        // from still holds: keep the window-start split (the live `rem` has
        // since drained as ranks took their slots). A `batch_counts` change
        // between pin and fire — a mid-epoch anchor-growth commit, a nudge,
        // an election — makes the pin STALE: dispatch would keep serving
        // slots sized for a schedule the firing gate no longer checks
        // against (the 2026-07-29 tail-crumb deadlock). Re-pin from the
        // live remainder + live counts instead; allocations are over
        // REMAINING batches, so already-served slots are never re-counted,
        // and the reduce barrier still holds over-quota ranks whatever
        // their fresh slot says.
        if matches!(
            &self.final_window_plan,
            Some(plan) if plan.epoch == epoch && plan.pinned_counts == counts
        ) {
            return;
        }
        let alive: Vec<bool> =
            (0..self.world_size).map(|r| !self.is_dead(r)).collect();
        let alloc = final_window_alloc(rem, counts, &alive);
        crate::debug!(
            "  ddp: final-window plan pinned | epoch={epoch} rem={rem} \
             counts={counts:?} alloc={alloc:?}",
        );
        self.final_window_plan = Some(FinalWindowPlan {
            epoch,
            alloc,
            pinned_counts: counts.to_vec(),
        });
    }

    /// Compute how many batches the next chunk for `rank` in `epoch`
    /// should contain. Cold-start (pre-calibration) uses a small probe
    /// chunk (~10% of dataset per rank, floored at 4 batches) so
    /// ElChe gets enough averaging events to stabilise quickly.
    /// Post-calibration uses throughput-proportional sizing with a
    /// `min_chunk_batches` floor.
    /// Per-rank reservation span sizes (samples, batch-aligned, summing
    /// to `batch_total`) from ElChe's throughput ratios — equal split
    /// until calibrated. The spans partition the epoch permutation so
    /// each rank's upcoming data is deterministic for the whole epoch
    /// (what the staging layer prefetches from); the pool's tail-steal
    /// truing absorbs ratio drift, so a span is a reservation, not a
    /// contract.
    pub(super) fn reservation_span_sizes(&self, batch_total: usize) -> Vec<usize> {
        let total_batches = batch_total / self.batch_size.max(1);
        let counts = self.el_che.batch_counts();
        let total_counts: usize = counts.iter().sum();
        let calibrated = self.el_che.is_calibrated() || self.el_che.has_speed_hint();
        let mut batches: Vec<usize> = if calibrated && total_counts > 0 {
            counts
                .iter()
                .map(|&c| total_batches * c / total_counts)
                .collect()
        } else {
            vec![total_batches / self.world_size.max(1); self.world_size]
        };
        // Hand the rounding remainder out round-robin so sizes sum exactly.
        let assigned: usize = batches.iter().sum();
        let n = batches.len().max(1);
        for i in 0..total_batches.saturating_sub(assigned) {
            batches[i % n] += 1;
        }
        batches.iter().map(|&b| b * self.batch_size).collect()
    }

    /// Emit every alive rank's `StageAdvisory`. Called at progressive
    /// epoch start and re-emitted at reduce boundaries (from the
    /// post-reduce wake path), so truing drift and ratio changes reach
    /// the stagers on the window clock — reservation state changes
    /// never get their own timer. Latest frame wins on the worker.
    pub(super) fn emit_stage_advisories(&mut self, epoch: usize) {
        for rank in 0..self.world_size {
            if self.is_dead(rank) {
                continue;
            }
            let mut segments: Vec<(u64, Vec<(u64, u64)>)> = Vec::new();
            let current = self.advisory_spans_for_rank(epoch, rank);
            if !current.is_empty() {
                segments.push((
                    epoch as u64,
                    current.iter().map(|&(o, s)| (o as u64, s as u64)).collect(),
                ));
            }
            let predicted = self.predicted_epoch_spans(epoch + 1, rank);
            if !predicted.is_empty() {
                segments.push((
                    (epoch + 1) as u64,
                    predicted.iter().map(|&(o, s)| (o as u64, s as u64)).collect(),
                ));
            }
            if segments.is_empty() {
                continue;
            }
            let msg = crate::distributed::wire::ControlMsgWire::StageAdvisory {
                counts: self
                    .el_che
                    .batch_counts()
                    .iter()
                    .map(|&c| c as u64)
                    .collect(),
                segments,
            };
            let _ = self.send_control(rank, &msg);
        }
    }

    /// Predicted reservation spans for a FUTURE epoch whose pool does
    /// not exist yet: the same ratio table the pool will be built from,
    /// laid out over that epoch's permutation. The prediction can drift
    /// from the eventual table if ratios move before the epoch starts —
    /// margin-covered, and the advisory refresh at each reduce keeps
    /// the head accurate. Empty past the run's last epoch.
    pub(super) fn predicted_epoch_spans(
        &self,
        next_epoch: usize,
        rank: usize,
    ) -> Vec<(usize, usize)> {
        if next_epoch >= self.num_epochs() || self.batch_size == 0 {
            return Vec::new();
        }
        let batch_total =
            (self.epoch_samples(next_epoch) / self.batch_size) * self.batch_size;
        if batch_total == 0 {
            return Vec::new();
        }
        let sizes = self.reservation_span_sizes(batch_total);
        let mut starts = Vec::with_capacity(sizes.len());
        let mut at = 0usize;
        for &s in &sizes {
            starts.push(at);
            at += s;
        }
        let counts = self.el_che.batch_counts();
        let mut spans = Vec::new();
        if sizes[rank] > 0 {
            spans.push((starts[rank], sizes[rank]));
        }
        for r in 0..self.world_size {
            if r == rank || sizes[r] == 0 {
                continue;
            }
            let margin = counts.get(r).copied().unwrap_or(0).max(1) * self.batch_size;
            let m = margin.min(sizes[r]);
            spans.push((starts[r] + sizes[r] - m, m));
        }
        spans
    }

    /// Certainty-ordered advisory spans for `rank`'s stager: its own
    /// reserved span first (deterministic for the whole epoch), then
    /// every other span's tail — the truing margins, one reduce window
    /// each, where the boundary can move under throughput drift. Spans
    /// may overlap across ranks (margins are staged by several ranks
    /// on purpose); allocation stays exclusive in the pool.
    pub(super) fn advisory_spans_for_rank(
        &self,
        epoch: usize,
        rank: usize,
    ) -> Vec<(usize, usize)> {
        let Some(pool) = self.chunk_pools.get(&epoch) else {
            return Vec::new();
        };
        let mut spans = Vec::new();
        let own = pool.reservation(rank);
        if own.1 > 0 {
            spans.push(own);
        }
        let counts = self.el_che.batch_counts();
        for r in 0..self.world_size {
            if r == rank {
                continue;
            }
            let (start, len) = pool.reservation(r);
            if len == 0 {
                continue;
            }
            let margin = counts.get(r).copied().unwrap_or(0).max(1) * self.batch_size;
            let m = margin.min(len);
            spans.push((start + len - m, m));
        }
        spans
    }

    pub(super) fn compute_chunk_batches(&self, rank: usize, epoch: usize) -> usize {
        let pool = match self.chunk_pools.get(&epoch) {
            Some(p) => p,
            None => return 0,
        };
        let remaining_batches = pool.remaining() / self.batch_size;
        if remaining_batches == 0 {
            return 0;
        }

        // FINAL-WINDOW PLAN (barrier-paced). Once the pool drops to within one
        // window of empty, `refresh_final_window_plan` has dispatched the whole
        // remainder as a single coherent window sized to avoid any lone-1 step
        // (the delivered-feed fallback trigger). Serve each rank its
        // pre-computed slot and bypass the per-rank sizing below entirely
        // (including `cap_to_reduce_budget` — the plan IS the final word, and
        // its fold crumb deliberately runs one batch past the reduce budget on
        // a slow rank). See `docs/design/epoch-tail-allocation.md`.
        if let Some(plan) = self.final_window_plan.as_ref()
            && plan.epoch == epoch {
                return plan.alloc.get(rank).copied().unwrap_or(0);
            }

        // EDGE SCHEDULE (epoch / run tail). When less than a full window of
        // work remains (`remaining < Σ batch_counts`), EVERY progressive
        // mode dispatches its share-capped integer slice from the shared
        // pool: `batch_counts[rank].min(remaining)`. Pure integer min — no
        // `min_chunk` floor, no fractional split — so it stays EXACT (the
        // pool drains to exactly 0, 100% of steps dispatched; the divide in
        // the averaging weight is separate weight-space math). Ranks drain
        // the pool in dispatch order; whoever finds it empty gets 0 and is
        // excluded from the weighted average (sum-and-count). Capping each
        // rank at its full-window share means no rank runs more than one
        // normal window of un-synced steps before the final reduce, so the
        // final average never over-weights a single over-driven rank.
        // Cadence's schedule-exact path below already does exactly this;
        // this extends it to the cpu-async / NCCL proportional path, which
        // would otherwise floor slow ranks up to `min_chunk` and split
        // proportionally rather than capping at the share.
        let total_counts: usize = self.el_che.batch_counts().iter().sum();
        if total_counts > 0 && remaining_batches < total_counts {
            let share = self.el_che.batch_counts().get(rank).copied().unwrap_or(0);
            return self.cap_to_reduce_budget(rank, share.min(remaining_batches));
        }

        // SCHEDULE-EXACT dispatch (barrier-paced: Sync/Cadence): one chunk ==
        // one reduce window == the rank's scheduled `counts[rank]` (capped to
        // the pool residual). This is what holds the single step clock:
        // the pool drains at exactly the reduce rate, so coverage (epoch)
        // and synchronization (reduce) can never decouple into two racing
        // clocks. Pre-calibration `counts` is the equal-split anchor
        // (`ElChe::new` seeds `[anchor; world_size]`), so the warmup
        // window is just an equal schedule — no probe special-case. Keyed on
        // the PACING policy, NOT the backend: NCCL `Cadence` needs the
        // window-sized chunk too, else it gets a whole-epoch chunk and the
        // reduce can only fire once per epoch (serializing the cohort).
        // cpu-async (bounded lookahead / streaming) falls through to the
        // throughput-proportional sizing below.
        if self.policy.is_barrier_paced() {
            let window = self.el_che.batch_counts().get(rank).copied().unwrap_or(0);
            if window > 0 {
                return window.min(remaining_batches);
            }
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
        let sized = target.max(self.min_chunk_batches).min(remaining_batches);
        // REDUCE BARRIER cap: never dispatch past the rank's remaining
        // step budget before the next reduce, so the chunk lands exactly
        // on the reduce boundary instead of overshooting it (this is what
        // keeps the `min_chunk_batches` floor from punching through a
        // small cadence window). `reduce_step_budget` is 0 for NCCL /
        // pre-calibration, leaving sizing untouched there. The caller
        // (`dispatch_next_chunk`) already withholds when the budget is
        // fully spent, so `budget_remaining >= 1` whenever we get here.
        self.cap_to_reduce_budget(rank, sized)
    }

    /// Cap a proposed chunk size at the rank's remaining reduce-step budget
    /// so a chunk never crosses a reduce barrier (it lands exactly on the
    /// boundary). The `.max(1)` keeps `budget_remaining` positive, but the
    /// outer `min` never inflates a smaller `sized` — a 0-share rank stays
    /// 0. No-op when `reduce_step_budget` is 0 (NCCL / pre-calibration).
    fn cap_to_reduce_budget(&self, rank: usize, sized: usize) -> usize {
        let budget = self.reduce_step_budget(rank);
        if budget > 0 {
            let budget_remaining =
                budget.saturating_sub(self.window.steps(rank)).max(1);
            sized.min(budget_remaining)
        } else {
            sized
        }
    }

    /// Per-rank step budget between reduces — the reduce hard barrier in
    /// the "one logical GPU, heterogeneous per-rank step counts" model.
    ///
    /// `counts[rank]` for Sync/Cadence (hard); `+ max_overshoot` for
    /// cpu-async (the one mode allowed to overrun, bounded for now by the
    /// single `max_overshoot` knob). Returns 0 — meaning "no software
    /// barrier, size/dispatch normally" — only when `batch_counts[rank]` is 0
    /// (a 0-share edge rank).
    ///
    /// Keyed on the PACING policy, NOT the backend. The reduce in cadence is
    /// coordinator-triggered (the worker syncs on `SyncNow`, not autonomously
    /// at its window edge), so the NCCL collective does NOT pace the fast
    /// rank — it needs the same software barrier as CPU. Returning 0 for NCCL
    /// (the old "its collective blocks ranks intrinsically" assumption) let
    /// the fast rank stream past its window across every epoch and wedge the
    /// cohort. There is no `is_calibrated` guard: pre-calibration
    /// `batch_counts` is the equal-split anchor schedule (`[anchor;
    /// world_size]`), a valid window, so the reduce barrier holds from the
    /// very first window ("the probe is no different from the rest of
    /// training").
    fn reduce_step_budget(&self, rank: usize) -> usize {
        let base = self.el_che.batch_counts().get(rank).copied().unwrap_or(0);
        if base == 0 {
            return 0;
        }
        if matches!(self.policy, ApplyPolicy::Async) {
            base + self.max_overshoot
        } else {
            base
        }
    }

    /// Wake any progressive rank stalled in `wait_for_epoch_plan` after
    /// being held by the overshoot gate (or otherwise finished its last
    /// chunk with no in-flight work). Called from the tail of
    /// [`Self::finish_averaging_nccl`] / [`Self::finish_averaging_cpu`]
    /// once `steps_since_avg` has been reset — without this kick,
    /// nothing drives `dispatch_next_chunk` for the stalled rank (no
    /// MetricsMsg arrives from a rank sitting in `wait_for_epoch_plan`).
    ///
    /// Runs from the post-epoch hook at the tail of
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
        // Reduce boundary = the reservation clock: refresh every rank's
        // stage advisory so truing drift and ratio changes reach the
        // stagers without a timer of their own.
        if let Some(&epoch) = self.chunk_pools.keys().next_back() {
            self.emit_stage_advisories(epoch);
        }
    }
}

/// Index of the alive rank with the largest scheduled `counts` (ties → lowest
/// index). The fast rank: never padded by the fold crumb, because its
/// between-sync step count is the convergence-sensitive quantity. `None` when
/// no rank is alive.
fn fastest_alive_rank(counts: &[usize], alive: &[bool]) -> Option<usize> {
    (0..counts.len())
        .filter(|&r| alive[r])
        .max_by_key(|&r| (counts.get(r).copied().unwrap_or(0), usize::MAX - r))
}

/// Repair lone-1 allocations in place: any alive rank holding exactly one
/// batch has it moved onto the smallest alive peer that already holds `>= 1`
/// (which becomes `>= 2`); the orphan drops to 0. Never moves onto a 0 — that
/// would just relocate the lone 1. If no `>= 1` peer exists (the remainder is
/// too small for any pair), the lone 1 is left as is and that window's feed
/// falls back to the compute scale — irreducible and benign.
fn consolidate_lone_ones(alloc: &mut [usize], alive: &[bool]) {
    while let Some(orphan) =
        (0..alloc.len()).find(|&r| alive[r] && alloc[r] == 1)
    {
        let peer = (0..alloc.len())
            .filter(|&p| p != orphan && alive[p] && alloc[p] >= 1)
            .min_by_key(|&p| alloc[p]);
        match peer {
            Some(p) => {
                alloc[p] += 1;
                alloc[orphan] = 0;
            }
            None => break,
        }
    }
}

/// Allocate an epoch's FINAL reduce window: the entire remaining `rem` batches
/// in one window, sized so no participating rank ends at exactly 1 step (the
/// delivered-feed marginal-skip fallback trigger). `counts` is the live ElChe
/// schedule, `alive[r]` is false for a dead rank (allocated 0). The returned
/// vector sums to `rem` exactly (coverage is exact — this only reshapes how
/// the remainder is split). See `docs/design/epoch-tail-allocation.md`.
///
/// Two regimes by `rem` vs `Σcounts` (over alive ranks):
/// - `rem >= Σcounts` (FOLD BAND, `rem = Σcounts + R`, `R < world_size`):
///   each alive rank takes its full scheduled share, then the `R`-batch crumb
///   goes one-per-rank to the slowest alive non-fast ranks. The fast rank is
///   never padded.
/// - `rem < Σcounts` (PROPORTIONAL SUB-WINDOW): largest-remainder split of
///   `rem` across alive shares, then [`consolidate_lone_ones`].
fn final_window_alloc(rem: usize, counts: &[usize], alive: &[bool]) -> Vec<usize> {
    let world = counts.len();
    let mut alloc = vec![0usize; world];
    if rem == 0 {
        return alloc;
    }
    let total: usize = (0..world)
        .filter(|&r| alive[r])
        .map(|r| counts[r])
        .sum();
    if total == 0 {
        return alloc;
    }

    if rem >= total {
        // FOLD BAND: full schedule + a sub-cohort crumb, dispatched together.
        for r in 0..world {
            if alive[r] {
                alloc[r] = counts[r];
            }
        }
        let mut crumb = rem - total;
        if crumb > 0 {
            let fast = fastest_alive_rank(counts, alive);
            let mut order: Vec<usize> = (0..world)
                .filter(|&r| alive[r] && Some(r) != fast)
                .collect();
            // Slowest first so the crumb lands on the ranks with the most
            // headroom; round-robin if it exceeds the slot count (only when
            // dead ranks shrink the cohort below the crumb size).
            order.sort_by_key(|&r| counts[r]);
            if order.is_empty() {
                // Single alive rank: it takes the whole remainder.
                if let Some(f) = fast {
                    alloc[f] += crumb;
                }
            } else {
                let mut i = 0;
                while crumb > 0 {
                    alloc[order[i % order.len()]] += 1;
                    crumb -= 1;
                    i += 1;
                }
            }
        }
        return alloc;
    }

    // PROPORTIONAL SUB-WINDOW: largest-remainder apportionment of `rem`.
    let mut frac: Vec<(usize, f64)> = Vec::with_capacity(world);
    let mut placed = 0usize;
    for r in 0..world {
        if !alive[r] {
            continue;
        }
        let exact = rem as f64 * counts[r] as f64 / total as f64;
        let floor = exact.floor() as usize;
        alloc[r] = floor;
        placed += floor;
        frac.push((r, exact - floor as f64));
    }
    frac.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut leftover = rem - placed;
    let mut i = 0;
    while leftover > 0 && !frac.is_empty() {
        alloc[frac[i % frac.len()].0] += 1;
        leftover -= 1;
        i += 1;
    }
    consolidate_lone_ones(&mut alloc, alive);
    alloc
}

#[cfg(test)]
mod final_window_alloc_tests {
    use super::{consolidate_lone_ones, fastest_alive_rank, final_window_alloc};

    fn no_lone_one(alloc: &[usize]) -> bool {
        alloc.iter().all(|&n| n != 1)
    }

    #[test]
    fn proportional_split_sums_and_has_no_lone_one() {
        // The observed rig tail: remaining 72, schedule [71, 18, 15].
        let alloc = final_window_alloc(72, &[71, 18, 15], &[true, true, true]);
        assert_eq!(alloc.iter().sum::<usize>(), 72, "coverage exact");
        assert!(no_lone_one(&alloc), "no rank at exactly 1: {alloc:?}");
        // Fast rank still dominant, slow ranks get a real (>=2) slice.
        assert!(alloc[0] > alloc[1] && alloc[0] > alloc[2], "{alloc:?}");
    }

    #[test]
    fn tiny_proportional_windows_consolidate() {
        // Remainders that would orphan a slow rank under a raw split.
        for rem in 3..=20usize {
            let alloc = final_window_alloc(rem, &[71, 18, 15], &[true, true, true]);
            assert_eq!(alloc.iter().sum::<usize>(), rem, "rem={rem} {alloc:?}");
            assert!(no_lone_one(&alloc), "rem={rem} lone-1: {alloc:?}");
        }
    }

    #[test]
    fn fold_band_pads_slow_ranks_not_the_fastest() {
        // rem = Σcounts + 2: two crumb batches, both to slow ranks.
        let counts = [71, 18, 15];
        let alloc = final_window_alloc(106, &counts, &[true, true, true]);
        assert_eq!(alloc.iter().sum::<usize>(), 106);
        assert_eq!(alloc[0], 71, "fast rank never padded");
        assert_eq!(alloc[1] + alloc[2], 35, "crumb landed on the slow ranks");
    }

    #[test]
    fn dead_rank_gets_nothing() {
        let alloc = final_window_alloc(40, &[71, 18, 15], &[true, false, true]);
        assert_eq!(alloc[1], 0, "dead rank allocated nothing: {alloc:?}");
        assert_eq!(alloc.iter().sum::<usize>(), 40);
        assert!(no_lone_one(&alloc), "{alloc:?}");
    }

    #[test]
    fn irreducible_lone_one_is_left_for_the_fallback() {
        // rem = 1, nowhere to consolidate: accept the single lone 1.
        let alloc = final_window_alloc(1, &[71, 18, 15], &[true, true, true]);
        assert_eq!(alloc.iter().sum::<usize>(), 1);
    }

    #[test]
    fn consolidate_examples() {
        let alive = [true, true, true];
        let mut a = [1, 1, 0];
        consolidate_lone_ones(&mut a, &alive);
        assert!(a.iter().all(|&n| n != 1) && a.iter().sum::<usize>() == 2, "{a:?}");
        let mut b = [2, 1, 1];
        consolidate_lone_ones(&mut b, &alive);
        assert!(b.iter().all(|&n| n != 1) && b.iter().sum::<usize>() == 4, "{b:?}");
    }

    #[test]
    fn fastest_rank_skips_dead() {
        assert_eq!(fastest_alive_rank(&[71, 18, 15], &[false, true, true]), Some(1));
        assert_eq!(fastest_alive_rank(&[71, 18, 15], &[true, true, true]), Some(0));
    }
}
