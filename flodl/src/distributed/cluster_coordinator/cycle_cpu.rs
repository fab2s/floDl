//! CPU averaging-cycle transport mechanics for
//! [`super::ClusterCoordinator`]: arming (`RequestParams` + the
//! non-progressive Throttle barrier), the `CpuAvgState` machine
//! (`Pending` → finalize → `Idle`), and its stall backstop.
//! Split-impl sibling of `averaging.rs`, which keeps the policy hooks
//! (`trigger_averaging` dispatch, `finish_averaging_head` feedback
//! into ElChe + the guard).
//!
//! Constraints on everything in this file:
//! - **No interior wait may go heartbeat-silent** (see
//!   `cycle_nccl.rs`; the CPU machine is poll-driven from `tick`, so
//!   it inherits the heartbeat for free — keep it that way).
//! - **The cycle owns no cadence state.** The schedule is the single
//!   step clock; this file executes windows, it never decides them.

use std::time::{Duration, Instant};

use crate::distributed::ddp_run::ApplyPolicy;
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::Result;

use super::{ClusterCoordinator, CpuAvgState};

impl ClusterCoordinator {
    /// CPU arm of [`Self::trigger_averaging`]: the transport mechanics
    /// of one CPU cycle — `RequestParams` broadcast, the
    /// non-progressive Throttle barrier, re-arm slot resets, and
    /// opening the `Pending` window that `poll_cpu_averaging`
    /// finalizes.
    pub(super) fn arm_cpu_cycle(&mut self) -> Result<()> {

            self.nccl_sync_start = Some(Instant::now());
            // Best-effort (see the SyncNow twin above): a rank whose
            // RequestParams never arrived has a broken connection;
            // heartbeat staleness declares it dead and
            // `poll_cpu_averaging`'s is_dead exception completes the
            // cycle without it.
            if let Err(e) = self.broadcast_control(&ControlMsgWire::RequestParams) {
                eprintln!(
                    "flodl ddp: RequestParams broadcast incomplete ({e}); \
                     relying on dead-rank recovery"
                );
            }
            // Hard barrier for the non-progressive (Sync) path:
            // after snapshotting, the fast rank blocks until the
            // averaged `Update` lands, recreating NCCL's
            // AllReduce-blocking semantics on the CPU backend.
            // Without it the fast rank keeps training through the
            // (TCP-latency) averaging window and laps the slow
            // ranks, so the average degrades and convergence drops
            // off NCCL parity. Released by the bridge's
            // `ControlMsg::Update(avg)` (the worker's `Throttle`
            // handler still services `RequestParams` while blocked,
            // so the round that releases it can still complete).
            //
            // The gate is `!progressive`, not the policy enum: the
            // Throttle is needed exactly when dispatch cannot starve
            // the rank of work during the averaging window.
            // Progressive Cadence dispatches `chunk == counts ==
            // window`, so the reduce barrier in `dispatch_next_chunk`
            // already withholds further work and the fast rank idles
            // in `wait_for_epoch_plan` (still servicing
            // `RequestParams`) — the Throttle would be redundant
            // there. Non-progressive Sync has no inter-window
            // dispatch to starve (whole epoch trained in one inner
            // loop), so the Throttle IS its barrier. Async is always
            // progressive and bounds lookahead via the overshoot
            // budget instead.
            if !self.progressive {
                if let Err(e) = self.broadcast_control(&ControlMsgWire::Throttle) {
                    crate::verbose!("  ddp: Throttle broadcast incomplete: {e}");
                }
                for t in &mut self.throttled {
                    *t = true;
                }
            }
            for rank in 0..self.world_size {
                self.nccl_sync_step[rank] = self.last_step_count[rank];
                self.nccl_ack[rank] = false;
            }
            // Reset per-rank upload markers so the new cycle's
            // measurements aren't read against a stale prior cycle.
            // NCCL path skips this — there's no SnapshotReady on
            // the in-place collective so the slots stay None
            // throughout.
            for slot in &mut self.last_observed_upload_ms {
                *slot = None;
            }
            // CpuAvgStart: open the Pending window. Mirrors threaded
            // `coordinator/cpu_avg.rs:634`. Closed in
            // `finish_averaging_cpu` via the `cpu_avg_start` field
            // so MSF / dashboard see the same `CpuAvgEnd
            // { duration_ms }` payload shape on cluster runs.
            //
            // DELIBERATE SEMANTICS: this clock starts at the TRIGGER, so
            // the derived sync_ms is the controller-perspective cost of
            // the whole rendezvous — including each rank draining its
            // in-flight batch (~1 batch, a stable additive term) and the
            // snapshot transport from far ranks (the dominant term on a
            // slow link, and exactly what the anchor must amortize). Do
            // NOT narrow this to collect-complete → scatter-done: that
            // would hide transport cost and under-grow the anchor for
            // distant ranks.
            self.cpu_avg_start = Some(Instant::now());
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::CpuAvgStart);
            }
            // Defer `finish_averaging_cpu` until every rank's
            // bridge SyncAck has populated `nccl_sync_divergence`
            // (otherwise the guard reads all-Nones → zero, breaking
            // divergence-driven cadence control on cycle 1).
            // `poll_cpu_averaging` (called from `tick`) finalizes.
            //
            // No deadline: dropping a CPU averaging cycle is a
            // correctness violation for Local SGD (per-rank drift
            // accumulates super-linearly across missed rendezvous
            // points). Liveness is a SEPARATE concern handled by
            // the heartbeat fault detector; slow-but-alive ranks
            // are absorbed by ElChe's per-rank wall /
            // `batch_counts` rebalance on the next cycle.
            self.cpu_avg_state = CpuAvgState::Pending;
        Ok(())
    }

    /// Drive the CPU averaging state machine one tick. No-op when
    /// [`CpuAvgState::Idle`].
    ///
    /// In [`CpuAvgState::Pending`]: if every ALIVE rank's bridge
    /// `SyncAck` has landed (i.e. `nccl_sync_divergence[r].is_some()`),
    /// runs [`Self::finish_averaging_cpu`] and returns to `Idle`.
    /// Dead ranks (per [`Self::dead_ranks`]) count as "acked" because
    /// their bridge SyncAck will never arrive — the controller has
    /// already released the in-flight AllReduce with surviving ranks
    /// only, so finalizing here is correct.
    ///
    /// The gate is on `nccl_sync_divergence`, not `nccl_ack`. `nccl_ack`
    /// can flip from an in-flight `Batch` whose `step_count` exceeds
    /// `nccl_sync_step` (set at trigger time) — that path is correct
    /// for the NCCL backend (no separate bridge; the post-AllReduce
    /// Batch IS the sync evidence) but not for the CPU backend, where
    /// the bridge's `SyncAck` is the only signal that the AllReduce
    /// round-trip actually finished. Gating on `nccl_sync_divergence`
    /// ensures the next `finish_averaging_cpu` reads real per-rank
    /// divergence rather than the all-Nones sentinel.
    pub(super) fn poll_cpu_averaging(&mut self) -> Result<()> {
        if !matches!(self.cpu_avg_state, CpuAvgState::Pending) {
            return Ok(());
        }
        let all_alive_acked = (0..self.world_size).all(|r| {
            if self.is_dead(r) {
                return true;
            }
            self.nccl_sync_divergence[r].is_some()
        });
        if all_alive_acked {
            // Finalize FIRST, then flip to Idle — and flip even when the
            // finalize errored (re-running it would double-fold chunks and
            // double-bump the version; the error already surfaced). The
            // old order (Idle before finalize) let a finalize error leave
            // half the cohort updated with the state machine claiming the
            // cycle was over.
            let result = self.finish_averaging_cpu();
            self.cpu_avg_state = CpuAvgState::Idle;
            return result;
        }
        // HARD CEILING on Pending. The no-deadline stance is correct for
        // SLOW cycles (dropping a reduce is a Local-SGD correctness
        // violation, and dead ranks are the heartbeat detector's job) —
        // but an alive-but-wedged cohort otherwise parks here forever
        // with heartbeats flowing (the quiet-wedge signature: coord tick
        // alive, workers parked). Past the ceiling this is a scheduler
        // wedge, not a slow reduce: dump the gate state and escalate to
        // save-and-shutdown so an overnight run ends as a diagnosed
        // checkpoint instead of a silent hang. The ceiling is generous —
        // 10x the heartbeat timeout (default 300s) dwarfs any observed
        // CPU reduce — so a healthy rig can never hit it.
        if let Some(start) = self.cpu_avg_start {
            let ceiling_secs = self.heartbeat_timeout_secs.saturating_mul(10).max(300);
            if start.elapsed() > Duration::from_secs(ceiling_secs) {
                eprintln!(
                    "flodl ddp: CPU averaging cycle stalled past its {ceiling_secs}s \
                     ceiling with the cohort alive — escalating to ShutdownWithSave"
                );
                self.dump_stall_state(start.elapsed().as_secs_f64());
                self.cpu_avg_state = CpuAvgState::Idle;
                if let Err(e) = self.dispatch_shutdown_with_save(
                    crate::distributed::SaveReason::ReduceStall,
                ) {
                    crate::verbose!(
                        "  ddp: ShutdownWithSave after reduce stall failed: {e}"
                    );
                }
            }
        }
        Ok(())
    }

    pub(super) fn finish_averaging_cpu(&mut self) -> Result<()> {
        self.finish_averaging_head();

        let fold = matches!(self.policy, ApplyPolicy::Cadence);
        // Reset the per-window step counters BEFORE the fold: the fold
        // sizes the next chunk via `compute_chunk_batches`, whose
        // `cap_to_reduce_budget` reads `steps_since_avg`. With the reset
        // deferred until after the fold (the old order), the cap saw the
        // JUST-CLOSED window's full step count, `budget_remaining`
        // bottomed out at the `.max(1)` floor, and every epoch tail folded
        // a 1-batch chunk — off-schedule, and a fill-cost-inflated 1-batch
        // sample in the delivered feed. The reduce that brought us here IS
        // the window boundary; the new window's budget is fresh.
        self.window.reset_steps();
        for rank in 0..self.world_size {
            // Dead ranks get nothing: their controller-side stream is shut,
            // and folding a chunk for them would ghost it.
            if self.is_dead(rank) {
                continue;
            }
            let prev_epoch = self.rank_epoch[rank];
            let next_plan = if fold {
                self.fold_next_chunk_for_rank(rank)
            } else {
                None
            };
            let send_result = self.send_control(
                rank,
                &ControlMsgWire::Update {
                    version: self.version,
                    next_plan: next_plan.clone(),
                },
            );
            if let Err(e) = send_result {
                // BEST-EFFORT finalize: one broken connection must not
                // abort the cycle mid-fan-out (it left the cohort half
                // updated and killed the coordinator thread via `?`). Roll
                // back this rank's folded chunk so it isn't ghosted, log
                // loudly, and keep finalizing the others; heartbeat
                // staleness reaps the broken rank.
                if let Some(plan) = next_plan {
                    let epoch = plan.epoch as usize;
                    self.rollback_chunk_take(rank, epoch, &plan, prev_epoch);
                }
                eprintln!(
                    "flodl ddp: Update send to rank {rank} failed during CPU \
                     averaging finalize ({e}); continuing with remaining ranks"
                );
                self.note_lost_broadcast("Update", 1);
            }
        }
        // SetGlobalStep is still broadcast so workers can update the
        // per-batch LR scheduler base. Same as the NCCL path. Best-effort:
        // a failed rank misses one LR base update, which the next cycle's
        // broadcast corrects (or the rank is reaped as dead).
        if let Err(e) = self.broadcast_control(&ControlMsgWire::SetGlobalStep {
            global_step: self.global_step as u64,
        }) {
            crate::verbose!("  ddp: SetGlobalStep broadcast incomplete: {e}");
        }

        self.finish_averaging_tail();
        // CpuAvgEnd: close the Pending window opened in trigger_averaging.
        // duration_ms is the bridge round-trip time (RequestParams
        // broadcast -> all alive SyncAcks landed). Distinct from the
        // outer SyncEnd which also covers the post-finalize work.
        if let Some(start) = self.cpu_avg_start.take() {
            if let Some(ref tl) = self.timeline {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                tl.event(crate::monitor::EventKind::CpuAvgEnd { duration_ms });
            }
        }
        self.finish_pending_checkpoint_meta();
        self.emit_sync_end();
        Ok(())
    }
}
