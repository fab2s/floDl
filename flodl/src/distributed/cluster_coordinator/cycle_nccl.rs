//! NCCL averaging-cycle transport mechanics for
//! [`super::ClusterCoordinator`]: arming (`SyncNow` + the
//! window-completion wait), stall backstops, and the elastic
//! re-rendezvous path. Split-impl sibling of `averaging.rs`, which
//! keeps the policy hooks (`trigger_averaging` dispatch,
//! `finish_averaging_head` feedback into ElChe + the guard).
//!
//! Constraints on everything in this file:
//! - **No interior wait may go heartbeat-silent** — any blocking loop
//!   must keep calling `emit_coord_heartbeat` or every healthy rank's
//!   coord-liveness watchdog fires and the cohort self-destructs.
//! - **The cycle owns no cadence state.** The schedule (ElChe's
//!   `batch_counts`) is the single step clock; this file executes
//!   windows, it never decides them.

use std::time::{Duration, Instant};

use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::{Result, TensorError};

use super::{ClusterCoordinator, NcclRendezvousPending};

impl ClusterCoordinator {
    /// NCCL arm of [`Self::trigger_averaging`]: the transport mechanics
    /// of one NCCL cycle — the deterministic window-completion wait
    /// (which MUST keep the coord→rank heartbeat beating; see the
    /// beacon note inside), the `SyncNow` broadcast, re-arm slot
    /// resets, and the inline finish.
    pub(super) fn arm_nccl_cycle(&mut self) -> Result<()> {

            // TRANSPORT-AWARE FEED: pull the window's chunk-completion
            // metrics into the delivered accounting BEFORE the inline
            // `finish_averaging_nccl` consumes the window report. The gate
            // fired because every rank hit its window — their completion
            // frames are already queued (per-connection FIFO: the final
            // Batch report and the MetricsMsg ride the same relay
            // stream), just not drained this tick (`tick` drains metrics
            // AFTER the trigger). Without this, the delivered spans are
            // stale/partial at feed time and NCCL had to fall back to a
            // compute-only feed — blind to data/transport cost, leaving
            // the fast rank ~45% idle at the barrier on x1-link rigs.
            // Aggregation is NOT run here (pool removal would break the
            // quiesced-tail ordering); it stays in `tick`.
            //
            // DETERMINISTIC WINDOW-COMPLETION WAIT (progressive Cadence
            // only — Sync feeds compute-wall and has no window chunks to
            // wait for). At gate-fire every mover has COMPLETED its
            // window chunk (the gate requires `steps >= count`, or
            // quiesced with nothing in flight), so each mover's
            // completion frame is ALREADY SENT — queued or on the wire,
            // trailing the final Batch report that fired the gate by
            // microseconds on the same relay stream. Wait for the frames
            // instead of guessing a settle time, keeping the heartbeat /
            // dead-rank detector live so a rank that dies mid-wait is
            // declared dead, drops out of the movers set via `is_dead`,
            // and the predicate completes (a frame cannot be silently
            // lost while its rank stays alive: TCP lost frame == broken
            // connection == heartbeats stop == dead-rank fires).
            //
            // TERMINATION: bounded by relay latency when healthy; by
            // `heartbeat_timeout_secs` (+2s slack) under rank failure;
            // and when NO failure detector exists (`dead_ranks` ledger
            // unset — headless coordinators, non-elastic runs, unit
            // tests that never send completion frames) a missing frame
            // cannot be attributed to death, so the wait is capped SHORT
            // and the report's all-or-none falls back to a coherent
            // compute-scale window. The cohort is barrier-parked while
            // we wait — the only thing delayed is the reduce itself.
            if self.progressive && matches!(self.policy, ApplyPolicy::Cadence) {
                let ceiling = if self.dead_ranks.is_some() {
                    Duration::from_secs(
                        self.heartbeat_timeout_secs.saturating_add(2),
                    )
                } else {
                    Duration::from_millis(100)
                };
                let wait_start = Instant::now();
                let mut slow_wait_logged = false;
                loop {
                    self.drain_metrics();
                    if self.movers_delivered_complete() {
                        break;
                    }
                    if wait_start.elapsed() >= ceiling {
                        crate::verbose!(
                            "  ddp: window-completion wait hit its \
                             ceiling ({:?}) — feeding compute-scale \
                             this window (all-or-none fallback)",
                            ceiling,
                        );
                        break;
                    }
                    self.drain_timing();
                    self.check_dead_ranks();
                    // Keep the OUTBOUND liveness beacon alive too. This
                    // wait's ceiling deliberately outlasts rank-death
                    // attribution (heartbeat_timeout + 2s), but the
                    // coord→rank heartbeat is normally emitted from
                    // tick() — which this loop blocks. Without this,
                    // every HEALTHY rank's coord-liveness watchdog fires
                    // at heartbeat_timeout (the beacon has been silent
                    // longer than the deadline, since the ceiling
                    // exceeds it by design) and the whole cohort
                    // self-destructs mid-wait — observed as an
                    // intermittent cadence wedge whenever a mover's
                    // completion frame arrived late. Throttled to 1s
                    // internally, so per-iteration is cheap.
                    self.emit_coord_heartbeat();
                    if !slow_wait_logged
                        && wait_start.elapsed().as_secs_f64() > 1.0
                    {
                        slow_wait_logged = true;
                        crate::verbose!(
                            "  ddp: window-completion wait >1s — a \
                             mover's completion frame is late \
                             (watching heartbeats)",
                        );
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            self.cycle.arm(Instant::now(), &self.last_step_count);
            // Best-effort: a rank whose send failed has a broken
            // connection — its heartbeats ride the same stream, so
            // dead-rank detection fires shortly and the NCCL
            // abort/rebuild path recovers the collective. Aborting the
            // trigger here would kill the coordinator thread instead.
            if let Err(e) = self.broadcast_control(&ControlMsgWire::SyncNow) {
                eprintln!(
                    "flodl ddp: SyncNow broadcast incomplete ({e}); \
                     relying on dead-rank recovery"
                );
            }
            self.finish_averaging_nccl()?;
        Ok(())
    }

    /// NCCL twin of [`Self::poll_cpu_averaging`]'s hard ceiling. The CPU
    /// backend parks in `CpuAvgState::Pending` between `RequestParams` and
    /// the bridge SyncAcks, so its wedge backstop lives there. The NCCL
    /// backend has no such state — `finish_averaging_nccl` runs inline at
    /// trigger and the cohort re-arms only when every rank reports a
    /// post-`SyncNow` `Batch`/`SyncAck` (setting the cycle's ack slot). A
    /// rank wedged in the in-place collective (NCCL deadlock, or a peer
    /// parked at the barrier whose heartbeat thread still ticks so
    /// dead-rank detection never fires) leaves the cycle's `started_at`
    /// armed and its acks incomplete forever — the same quiet-wedge
    /// signature, with no equivalent backstop until now.
    ///
    /// Past the same generous ceiling (10x the heartbeat timeout, ≥300s,
    /// which dwarfs any real NCCL AllReduce) with the cohort alive but not
    /// fully acked, escalate to save-and-shutdown so an overnight run ends
    /// as a diagnosed checkpoint rather than a silent hang. Disarms
    /// `started_at` so the escalation fires once. Healthy cycles take
    /// `started_at` via the all-acked elapsed capture
    /// (`AvgCycleState::capture_sync_elapsed_if_complete`) long before
    /// the ceiling, so this never trips on a sound rig.
    pub(super) fn poll_nccl_reduce_stall(&mut self) -> Result<()> {
        // 10x the heartbeat timeout (default 300s) dwarfs any real NCCL
        // AllReduce, so a healthy rig can never hit it. Mirrors
        // `poll_cpu_averaging`'s ceiling.
        let ceiling = Duration::from_secs(
            self.heartbeat_timeout_secs.saturating_mul(10).max(300),
        );
        self.poll_nccl_reduce_stall_with(ceiling)
    }

    /// Inner body of [`Self::poll_nccl_reduce_stall`], parameterized by
    /// `ceiling` so tests can exercise the escalation without waiting the
    /// production 300s+ floor.
    pub(super) fn poll_nccl_reduce_stall_with(&mut self, ceiling: Duration) -> Result<()> {
        if !matches!(self.backend, AverageBackend::Nccl) {
            return Ok(());
        }
        // Only meaningful while a sync is in flight: the cycle's
        // `started_at` is set at `SyncNow` broadcast and taken once
        // every alive rank acks.
        let Some(start) = self.cycle.started_at else {
            return Ok(());
        };
        // Re-arm pending, not stalled: the all-acked transition is the
        // capture path's job, not an escalation.
        if self.cycle.all_alive_acked(|r| self.is_dead(r)) {
            return Ok(());
        }
        if start.elapsed() > ceiling {
            eprintln!(
                "flodl ddp: NCCL averaging cycle stalled past its {}s ceiling \
                 with the cohort alive but not fully acked — escalating to \
                 ShutdownWithSave",
                ceiling.as_secs(),
            );
            self.dump_stall_state(start.elapsed().as_secs_f64());
            self.cycle.started_at = None;
            if let Err(e) = self
                .dispatch_shutdown_with_save(crate::distributed::SaveReason::ReduceStall)
            {
                crate::verbose!(
                    "  ddp: ShutdownWithSave after NCCL reduce stall failed: {e}"
                );
            }
        }
        Ok(())
    }

    /// Whether the last NCCL averaging cycle has fully settled: every
    /// alive rank's post-collective `SyncAck` has landed (or no sync was
    /// ever in flight). A rank's ack is sent AFTER its
    /// `sync_now_nccl` stream-synchronize returns, so all-alive-acked
    /// means every kernel of the collective has retired on every rank —
    /// the only state in which it is safe to let any rank exit and tear
    /// down its NCCL resources.
    ///
    /// This is the coordinator-confirmed-exit gate: `Shutdown` must not
    /// be broadcast while this is false. NCCL kernel completion is not
    /// globally synchronized (the LL small-message protocol can retire
    /// one rank's kernel while its peers' kernels of the SAME collective
    /// are still in flight), so an early-finishing rank that receives
    /// `Shutdown`, exits its process, and destroys its comm strands the
    /// peers in `synchronize()` at 100% GPU forever. Trivially true on
    /// the CPU backend — its equivalent settle gate is
    /// `cpu_avg_state == Idle`, enforced at the top of
    /// `try_advance_or_shutdown_after_aggregate`.
    ///
    /// Uses the alive-acked form (`is_dead(r) || acked[r]`), not the
    /// raw `started_at.is_none()`: dead ranks never ack, so the
    /// capture path can leave `started_at` armed after a death even
    /// though the surviving cohort has fully settled. A rank that exits
    /// WITHOUT acking keeps this false by design — the stranded cohort
    /// is then ended by `poll_nccl_reduce_stall`'s ceiling escalation,
    /// not by a Shutdown that no parked rank could read anyway.
    pub(super) fn nccl_sync_settled(&self) -> bool {
        if !matches!(self.backend, AverageBackend::Nccl) {
            return true;
        }
        self.cycle.started_at.is_none()
            || self.cycle.all_alive_acked(|r| self.is_dead(r))
    }

    /// NCCL-backend re-rendezvous initiation. Called from
    /// [`Self::check_dead_ranks`] once a rank has been declared dead
    /// on the NCCL path. No-op on CPU backend (the controller-side
    /// release handles CPU AllReduces). No-op when a rendezvous is
    /// already pending — additional deaths during the wait will be
    /// rolled into the same rendezvous when it completes
    /// (`broadcast_new_nccl_session` reads the *current* alive set,
    /// not the snapshot from rendezvous-initiation time).
    pub(super) fn initiate_nccl_rendezvous_if_needed(&mut self) -> Result<()> {
        if !matches!(self.backend, AverageBackend::Nccl) {
            return Ok(());
        }
        if self.nccl_rendezvous_pending.is_some() {
            return Ok(());
        }
        let survivors_ordered: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_dead(*r))
            .collect();
        if survivors_ordered.len() < 2 {
            crate::verbose!(
                "  ddp: NCCL rendezvous skipped — fewer than 2 survivors \
                 ({} alive of {})",
                survivors_ordered.len(),
                self.world_size,
            );
            return Ok(());
        }
        let generator_rank = self.pick_uid_generator(&survivors_ordered);
        self.send_control(generator_rank, &ControlMsgWire::RequestNewNcclId)?;
        self.nccl_rendezvous_pending = Some(NcclRendezvousPending {
            generator_rank,
            survivors_ordered,
            initiated_at: Instant::now(),
            tried_generators: Vec::new(),
        });
        Ok(())
    }

    /// Retry an in-flight NCCL re-rendezvous when the chosen generator
    /// has died or has not responded within
    /// [`super::NCCL_RENDEZVOUS_TIMEOUT_SECS`]. Picks the next candidate from
    /// the rendezvous's `survivors_ordered` after filtering out dead
    /// ranks and already-tried generators. When the candidate pool is
    /// exhausted, falls back to
    /// [`Self::dispatch_shutdown_with_save`] so the cluster doesn't
    /// hang forever on a dead-on-arrival generator chain.
    ///
    /// No-op when no rendezvous is pending, when the backend isn't
    /// NCCL, or when the generator is still alive and inside its
    /// timeout window.
    pub(super) fn check_rendezvous_timeout(&mut self) {
        if !matches!(self.backend, AverageBackend::Nccl) {
            return;
        }
        let timeout = Duration::from_secs(self.rendezvous_timeout_secs);
        let Some(pending) = self.nccl_rendezvous_pending.as_ref() else {
            return;
        };
        let generator_dead = self.is_dead(pending.generator_rank);
        let timed_out = pending.initiated_at.elapsed() > timeout;
        if !generator_dead && !timed_out {
            return;
        }

        let previous_generator = pending.generator_rank;
        let survivors = pending.survivors_ordered.clone();
        let mut tried = pending.tried_generators.clone();
        tried.push(previous_generator);

        // Filter: alive AND not previously tried. Preserve
        // `survivors_ordered`'s order (ascending global rank) so the
        // retry sequence is deterministic.
        let next: Option<usize> = survivors
            .iter()
            .copied()
            .find(|r| !self.is_dead(*r) && !tried.contains(r));

        crate::verbose!(
            "  ddp: NCCL rendezvous retry — previous generator {} {} (elapsed={:?}); \
             {} candidates remain",
            previous_generator,
            if generator_dead { "DIED" } else { "TIMED OUT" },
            pending.initiated_at.elapsed(),
            survivors
                .iter()
                .filter(|r| !self.is_dead(**r) && !tried.contains(r))
                .count(),
        );

        match next {
            Some(new_generator) => {
                // Send before mutating state so a send failure leaves
                // the previous pending entry intact for another retry
                // on the next tick.
                if let Err(e) = self
                    .send_control(new_generator, &ControlMsgWire::RequestNewNcclId)
                {
                    crate::verbose!(
                        "  ddp: NCCL rendezvous retry send to rank {} failed: {} \
                         (will try again next tick)",
                        new_generator,
                        e,
                    );
                    return;
                }
                if let Some(pending) = self.nccl_rendezvous_pending.as_mut() {
                    pending.generator_rank = new_generator;
                    pending.initiated_at = Instant::now();
                    pending.tried_generators = tried;
                }
            }
            None => {
                // Exhausted the candidate pool: every survivor at
                // initiation time has been asked and either died or
                // timed out. Clear the pending state and fall back to
                // ShutdownWithSave so survivors persist state instead
                // of hanging on an un-completable rendezvous.
                crate::verbose!(
                    "  ddp: NCCL rendezvous candidate pool exhausted; \
                     dispatching ShutdownWithSave"
                );
                self.nccl_rendezvous_pending = None;
                if let Some(reason) = self.unrecoverable_reason().or(Some(
                    crate::distributed::SaveReason::SingleSurvivor,
                ))
                    && let Err(e) = self.dispatch_shutdown_with_save(reason) {
                        crate::verbose!(
                            "  ddp: ShutdownWithSave after rendezvous exhaustion failed: {}",
                            e,
                        );
                    }
            }
        }
    }

    /// Pick the rank that should generate the next NCCL unique-id.
    ///
    /// Tier 1: the lowest-numbered SURVIVING rank that's in
    /// [`Self::local_ranks`] (co-located with the coord process,
    /// same-process latency, lowest correlated-failure risk).
    ///
    /// Tier 2: the surviving rank with the smallest observed
    /// per-batch wall from the window ledger (NOT
    /// barrier-correlated — clean per-rank capacity proxy). Ties
    /// break by lowest global rank. When no rank has timing
    /// history yet, this collapses to "lowest surviving global rank"
    /// (deterministic).
    pub(super) fn pick_uid_generator(&self, survivors_ordered: &[usize]) -> usize {
        // Tier 1: prefer a local survivor.
        if let Some(&local) = self
            .local_ranks
            .iter()
            .filter(|r| !self.is_dead(**r))
            .min()
        {
            return local;
        }
        // Tier 2: fastest network survivor (per-batch wall time,
        // tiebreak by global rank). `f64::partial_cmp` returns None
        // on NaN; treat as Equal so the rank tiebreak applies.
        survivors_ordered
            .iter()
            .copied()
            .min_by(|&a, &b| {
                let ta = self.per_rank_ms_per_batch(a);
                let tb = self.per_rank_ms_per_batch(b);
                ta.partial_cmp(&tb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            })
            .unwrap_or(survivors_ordered[0])
    }

    /// Average per-batch wall-time (ms) for rank `r` from the current
    /// cadence-interval accumulators. Returns `f64::INFINITY` when
    /// the rank has no batches yet (cold start) so it sorts LAST in
    /// the "fastest" picker — un-calibrated ranks shouldn't be
    /// preferred as UID generators.
    pub(super) fn per_rank_ms_per_batch(&self, r: usize) -> f64 {
        self.window.per_batch_wall_ms(r)
    }

    /// Broadcast `NewNcclSession` to every survivor. Called from the
    /// `NewNcclIdGenerated` arm of `process_timing_msg` after the
    /// generator rank ships back its freshly-generated UID. Each
    /// survivor receives a frame whose `new_rank` reflects its
    /// position among survivors ordered by ascending global rank.
    pub(super) fn broadcast_new_nccl_session(&mut self, uid_bytes: Vec<u8>) -> Result<()> {
        // Recompute the survivor set from the *current* `is_dead`
        // ledger — additional deaths during the rendezvous wait must
        // be reflected in the broadcast so we don't ship a
        // NewNcclSession to an already-dead rank.
        let survivors: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_dead(*r))
            .collect();
        if survivors.len() < 2 {
            return Err(TensorError::new(
                "cluster_coordinator: NewNcclSession broadcast aborted; \
                 fewer than 2 surviving ranks (NCCL requires world_size >= 2)",
            ));
        }
        let new_world_size = survivors.len() as u64;
        for (new_rank, &global_rank) in survivors.iter().enumerate() {
            let msg = ControlMsgWire::NewNcclSession {
                uid_bytes: uid_bytes.clone(),
                new_rank: new_rank as u64,
                new_world_size,
            };
            self.send_control(global_rank, &msg)?;
        }
        Ok(())
    }

    pub(super) fn finish_averaging_nccl(&mut self) -> Result<()> {
        self.finish_averaging_head();

        // Best-effort (see the CPU twin): a failed rank misses one LR base
        // update; the broken connection is reaped by heartbeat staleness.
        if let Err(e) = self.broadcast_control(&ControlMsgWire::SetGlobalStep {
            global_step: self.global_step as u64,
        }) {
            crate::verbose!("  ddp: SetGlobalStep broadcast incomplete: {e}");
        }

        self.window.reset_steps();

        self.finish_averaging_tail();
        self.finish_pending_checkpoint_meta();
        self.emit_sync_end();
        Ok(())
    }
}
