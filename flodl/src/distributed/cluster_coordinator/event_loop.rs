//! Coordinator event-loop core for [`super::ClusterCoordinator`]:
//! `tick`, the timing-message switchboard, the per-cycle metrics
//! aggregator, throttle dispatch, and the post-aggregate advance /
//! shutdown decision.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::{ControlMsgWire, TimingMsgWire};
use crate::tensor::Result;

use super::{ClusterCoordinator, RunPhase};

/// Coord→rank liveness-beacon cadence. Matches the rank→coord
/// `HEARTBEAT_CADENCE_MS` (1s): frequent enough that a rank's default 30s
/// coord-liveness deadline sees ~30 beacons before it could false-trip, cheap
/// enough that the per-tick broadcast overhead is negligible.
const COORD_HEARTBEAT_CADENCE: Duration = Duration::from_secs(1);

impl ClusterCoordinator {
    /// Process a single timing message. Ported literally from OLD
    /// `Coordinator::process_timing_msg`, modulo the field-name `rank`
    /// being `u64` on the wire vs `usize` in-process.
    pub(super) fn process_timing_msg(&mut self, msg: TimingMsgWire) {
        // Liveness tick: every frame from a rank counts as proof-of-
        // life. Updates `last_heartbeat[rank]` before any per-message
        // work so even malformed-rank frames (rejected below) at least
        // refresh the slot. `check_dead_ranks` later compares against
        // wall-clock.
        let rank_for_liveness: Option<usize> = match &msg {
            TimingMsgWire::Batch { rank, .. }
            | TimingMsgWire::SyncAck { rank, .. }
            | TimingMsgWire::Exiting { rank }
            | TimingMsgWire::LrUpdate { rank, .. }
            | TimingMsgWire::Intent { rank, .. }
            | TimingMsgWire::Heartbeat { rank, .. }
            | TimingMsgWire::SnapshotReady { rank }
            | TimingMsgWire::NewNcclIdGenerated { rank, .. }
            | TimingMsgWire::EvalResult { rank, .. }
            | TimingMsgWire::CheckpointResult { rank, .. }
            | TimingMsgWire::EpochFnElapsed { rank, .. }
            | TimingMsgWire::DashboardRegister { rank, .. }
            | TimingMsgWire::DashboardSetSvg { rank, .. }
            | TimingMsgWire::DashboardSetMetadata { rank, .. }
            | TimingMsgWire::DashboardSetHardware { rank, .. }
            | TimingMsgWire::ResourceSample { rank, .. } => Some(*rank as usize),
        };
        if let Some(r) = rank_for_liveness {
            if r < self.last_heartbeat.len() {
                self.last_heartbeat[r] = Instant::now();
            }
        }
        match msg {
            TimingMsgWire::Batch {
                rank,
                batch_ms,
                data_ms,
                step_count,
                param_norm,
                batch_loss,
                sync_divergence,
            } => {
                let rank = rank as usize;
                let step_count = step_count as usize;
                if rank >= self.world_size {
                    return; // ignore malformed; tests will fail loudly
                }
                // REPORT-AT-SYNC delivered feed: the ledger accumulates
                // steps + compute wall continuously and routes the
                // delivered cost per the marginal rule (window's FIRST
                // batch into the fill slot, batches 2..n into the
                // marginal rate) — present at the reduce by construction
                // (no completion-frame race). The window report and the
                // window-pressure fill consume this.
                self.window.record_batch(rank, batch_ms, data_ms);
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                self.last_batch_ms[rank] = batch_ms;
                // Sub-epoch loss feed: accumulate the per-batch loss into
                // the window so the monitor record emitted at the reduce
                // boundary carries a real loss curve between epochs (an
                // epoch-only curve is a single point for 1-epoch LLM runs).
                // Monitoring-only — no scheduling authority reads it.
                self.window.record_batch_loss(rank, batch_loss);
                let _ = param_norm;
                self.cycle.note_batch_ack(rank, step_count, sync_divergence);
            }
            TimingMsgWire::SyncAck {
                rank,
                step_count,
                divergence,
                post_norm,
                pre_norm,
            } => {
                let rank = rank as usize;
                let step_count = step_count as usize;
                if rank >= self.world_size {
                    return;
                }
                // The cycle machine knows whether a SyncAck's
                // `step_count` is meaningful: only the NCCL path folds
                // it (for re-arm and global-step tracking); the CPU
                // bridge's SyncAck carries none, and folding it would
                // poison `last_step_count` — see
                // `AvgCycleState::sync_ack_step_meaningful` for the
                // full story (that bug happened once).
                if self.cycle.sync_ack_step_meaningful() {
                    self.last_step_count[rank] =
                        self.last_step_count[rank].max(step_count);
                }
                // Divergence / norm evidence, the ack gate, and the
                // per-rank sync-lag capture all live on the cycle.
                self.cycle
                    .note_sync_ack(rank, step_count, divergence, pre_norm, post_norm);
            }
            TimingMsgWire::Exiting { rank } => {
                // Clean-exit latch: decrement exactly once, and remember the
                // exit so the heartbeat staleness scan doesn't declare this
                // rank dead 30s later (it stops heartbeating after Exiting)
                // and decrement again — the double count inflated
                // `dead_count` into spurious MaxFailureExceeded /
                // SingleSurvivor escalations during teardown.
                let rank = rank as usize;
                if rank < self.world_size && !self.exited[rank] {
                    self.exited[rank] = true;
                    self.active_count = self.active_count.saturating_sub(1);
                }
            }
            TimingMsgWire::LrUpdate { rank, lr } => {
                let rank = rank as usize;
                if rank < self.last_lr_per_rank.len() {
                    self.last_lr_per_rank[rank] = Some(lr);
                }
                // The meta-controller is consulted on every averaging
                // cycle via `observe_meta`; per-message work is just
                // recording the latest LR.
            }
            TimingMsgWire::Intent { rank, kind } => {
                // Cooperative-tier user intent (request, not command). Set the
                // cohort-wide pending flag; `dispatch_epoch` folds it into the
                // role-elected eval / checkpoint dispatch at the next epoch
                // boundary (never mid-window), on the elected rank — so the
                // requesting `rank` only matters for the liveness tick above.
                let _ = rank;
                match kind {
                    crate::distributed::wire::IntentKind::EvalNow => {
                        self.pending_eval_intent = true;
                    }
                    crate::distributed::wire::IntentKind::CheckpointNow => {
                        self.pending_checkpoint_intent = true;
                    }
                }
            }
            TimingMsgWire::Heartbeat { .. } => {
                // Liveness slot already refreshed above; nothing
                // further to do per-frame. `check_dead_ranks` reads
                // last_heartbeat each tick.
            }
            TimingMsgWire::SnapshotReady { rank } => {
                // Capture honest per-rank upload latency: T(now) -
                // T(RequestParams broadcast). The rank emitted this
                // BEFORE entering the AllReduce barrier (see param
                // bridge in cluster_worker.rs), so the measurement is
                // clean of slowest-rank barrier contamination — the
                // exact "honest per-rank capacity" signal flagged on
                // the barrier-correlated sync-lag slot as "planned upload-
                // completion marker".
                //
                // The cycle's `started_at` is the broadcast anchor; if
                // the cycle has already finalized (all SyncAcks in,
                // the elapsed capture took the anchor), this frame is
                // a late-arriving straggler and is dropped — the
                // upload slot keeps its prior value or None.
                self.cycle.note_snapshot_ready(rank as usize);
            }
            TimingMsgWire::NewNcclIdGenerated { rank, uid_bytes } => {
                let rank = rank as usize;
                if let Some(state) = self.nccl_rendezvous_pending.take() {
                    if state.generator_rank != rank {
                        crate::verbose!(
                            "  ddp: dropping NewNcclIdGenerated from rank {} \
                             (expected from generator rank {})",
                            rank,
                            state.generator_rank,
                        );
                        // Put back so we keep waiting for the real generator.
                        self.nccl_rendezvous_pending = Some(state);
                        return;
                    }
                    // Broadcast the new uid to each surviving rank
                    // with its position-in-shrunken-cohort. Survivors
                    // are ordered by ascending global rank.
                    if let Err(e) =
                        self.broadcast_new_nccl_session(uid_bytes)
                    {
                        crate::verbose!(
                            "  ddp: NewNcclSession broadcast failed: {} \
                             (NCCL elastic membership will not recover \
                             from this round of deaths; cluster may hang)",
                            e,
                        );
                    }
                } else {
                    crate::verbose!(
                        "  ddp: dropping unexpected NewNcclIdGenerated \
                         from rank {} (no rendezvous pending)",
                        rank,
                    );
                }
            }
            TimingMsgWire::EvalResult {
                rank,
                schedule_id: _,
                epoch,
                metric,
                elapsed_ms,
                error,
            } => {
                self.handle_eval_result(
                    rank as usize,
                    epoch as usize,
                    metric,
                    elapsed_ms,
                    error,
                );
            }
            TimingMsgWire::CheckpointResult {
                rank,
                version,
                elapsed_ms,
                error,
            } => {
                self.handle_checkpoint_result(
                    rank as usize,
                    version,
                    elapsed_ms,
                    error,
                );
            }
            TimingMsgWire::EpochFnElapsed {
                rank,
                epoch: _,
                elapsed_ms,
            } => {
                self.handle_epoch_fn_elapsed(rank as usize, elapsed_ms);
            }
            // Dashboard-channel frames: forwarded to the controller-side
            // DashboardSink when configured (the launcher hosts the
            // dashboard server post controller-active refactor). Sink
            // absent ⇒ silently dropped: the coord has no use for these
            // outside the launcher dashboard.
            TimingMsgWire::DashboardRegister { rank, port } => {
                if let Some(ref sink) = self.dashboard_sink {
                    sink.register_port(rank as usize, port);
                }
            }
            TimingMsgWire::DashboardSetSvg { rank, svg, label, hash } => {
                if let Some(ref sink) = self.dashboard_sink {
                    sink.set_svg(rank as usize, svg, label, hash);
                }
            }
            TimingMsgWire::DashboardSetMetadata { rank, json } => {
                if let Some(ref sink) = self.dashboard_sink {
                    sink.set_metadata(rank as usize, json);
                }
            }
            TimingMsgWire::DashboardSetHardware { rank, summary } => {
                if let Some(ref sink) = self.dashboard_sink {
                    sink.set_hardware(rank as usize, summary);
                }
            }
            TimingMsgWire::ResourceSample { rank, sample } => {
                let rank = rank as usize;
                if rank < self.world_size {
                    self.absorb_resource_sample(rank, sample);
                }
            }
        }
    }

    /// Absorb one rank's resource sample, from whichever cadence delivered it
    /// (the per-epoch `MetricsMsgWire` piggy-back or the sub-epoch
    /// `TimingMsgWire::ResourceSample` frame). One place so the two paths
    /// cannot drift: the timeline persists it host-qualified, the dashboard
    /// sink renders per-rank tabs, and the coordinator retains it for the next
    /// window record.
    pub(super) fn absorb_resource_sample(
        &mut self,
        rank: usize,
        wire_sample: crate::distributed::wire::ResourceSampleWire,
    ) {
        if let Some(tl) = &self.timeline {
            let host = self.rank_hosts.get(rank).map(String::as_str).unwrap_or("");
            let sample: crate::monitor::ResourceSample = wire_sample.clone().into();
            tl.rank_sample(rank, host, &sample);
        }
        // Retain for the window-report builder. Marked fresh so a window
        // record carries a sample only ONCE — repeating the last value on
        // every report would smear one reading across the whole epoch, which
        // is exactly what the sub-epoch cadence exists to avoid.
        if let Some(slot) = self.latest_res.get_mut(rank) {
            *slot = Some((
                crate::monitor::record::Res {
                    gpu_util: wire_sample.gpu_util_percent.map(|v| v as f64),
                    vram_alloc: wire_sample.vram_allocated_bytes.map(|v| v as f64),
                    vram_total: wire_sample.vram_total_bytes.map(|v| v as f64),
                },
                true,
            ));
        }
        if let Some(sink) = &self.dashboard_sink {
            sink.push_resource_sample(rank, wire_sample);
        }
    }

    /// Drain every pending timing message non-blocking.
    pub fn drain_timing(&mut self) {
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
    }

    /// Block up to `timeout` for the first timing message, then drain
    /// the rest non-blocking. Returns `false` when every reader thread
    /// has exited (all senders dropped) so the caller can break its
    /// loop.
    pub fn drain_timing_blocking(&mut self, timeout: Duration) -> bool {
        match self.timing_rx.recv_timeout(timeout) {
            Ok(msg) => self.process_timing_msg(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => return true,
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
        true
    }

    /// Stall watchdog (debug instrumentation). `global_step` advances
    /// only at `finish_averaging_*`; if it doesn't move for
    /// [`super::STALL_DUMP_SECS`] while ranks are alive, dump the
    /// `should_average` gate inputs once so the tight-window cadence
    /// wedge is captured rather than guessed. Re-arms (dumps again) on
    /// the next stall after progress resumes.
    pub(super) fn maybe_dump_stall(&mut self) {
        // ALWAYS ON (not gated on -vvv): the dump is the only diagnostic a
        // production wedge leaves behind, it costs nothing while training
        // progresses, and it emits at most once per interval. The interval
        // is tiered: tight (STALL_DUMP_SECS) under -vvv for active
        // debugging; generous (STALL_DUMP_PROD_SECS) otherwise so
        // legitimately slow reduce windows (epoch-sized cadence windows on
        // slow rigs, long eval callbacks) don't spam a healthy run.
        let interval = if self.prof_enabled {
            super::STALL_DUMP_SECS
        } else {
            super::STALL_DUMP_PROD_SECS
        };
        if self.global_step != self.stall_last_global_step {
            self.stall_last_global_step = self.global_step;
            self.stall_since = Some(Instant::now());
            self.stall_last_dump = None;
            return;
        }
        if self.active_count == 0 {
            return;
        }
        let since = *self.stall_since.get_or_insert_with(Instant::now);
        if since.elapsed() < Duration::from_secs(interval) {
            return;
        }
        // Re-dump every interval while stalled so one repro shows
        // whether ranks are still progressing (last_step_count moving) or
        // frozen.
        let due = self
            .stall_last_dump
            .is_none_or(|t| t.elapsed() >= Duration::from_secs(interval));
        if due {
            self.dump_stall_state(since.elapsed().as_secs_f64());
            self.stall_last_dump = Some(Instant::now());
        }
    }

    /// Dump the `should_average` gate state for a stalled cohort:
    /// per-rank `steps_since_avg` vs the scheduled `batch_counts`
    /// window, current epoch, and each chunk pool's residual + in-flight
    /// count. The predicted tight-window wedge looks like: epoch pool
    /// `remaining=0`, one rank `0 < steps < window` (took the epoch's
    /// short final chunk), the rest `steps=0` (drained, held at the
    /// epoch barrier) — so `should_average` can never fire.
    pub(super) fn dump_stall_state(&self, stalled_secs: f64) {
        let counts = self.el_che.batch_counts();
        eprintln!(
            "[stall-watch] STALL {:.0}s no reduce | cycle={:?} \
             active={}/{} last_agg_epoch={:?} avg_count={} global_step={} \
             lost_broadcasts={}",
            stalled_secs,
            self.cycle.machine,
            self.active_count,
            self.world_size,
            self.last_aggregated_epoch,
            self.avg_count,
            self.global_step,
            self.lost_broadcasts,
        );
        for r in 0..self.world_size {
            let steps = self.window.steps(r);
            let window = counts.get(r).copied().unwrap_or(0);
            let gate = if self.is_dead(r) {
                "dead"
            } else if steps == 0 {
                "ZERO (blocks gate)"
            } else if steps < window {
                "BELOW-WINDOW (blocks gate)"
            } else {
                "ready"
            };
            eprintln!(
                "  rank {r}: epoch={} steps_since_avg={} window(counts)={} \
                 last_step_count={} -> {gate}",
                self.rank_epoch[r], steps, window, self.last_step_count[r],
            );
        }
        for (epoch, pool) in &self.chunk_pools {
            let inflight: Vec<usize> =
                (0..self.world_size).map(|r| pool.in_flight(r)).collect();
            eprintln!(
                "  pool epoch={epoch}: remaining={} in_flight={inflight:?}",
                pool.remaining(),
            );
        }
    }

    /// Check whether an averaging cycle should be triggered now.
    ///
    /// Re-arm (the "previous cycle has settled" gate) is backend-split:
    ///
    /// - **NCCL**: the cycle's ack slot — set true once rank `r`
    ///   reports a `Batch`/`SyncAck` whose `step_count` exceeds the
    ///   trigger-time snapshot, proving it processed the previous
    ///   `SyncNow`.
    /// - **CPU**: the machine's phase back at `Idle` — the deferred
    ///   `poll_cpu_averaging` only returns there after it has finalized
    ///   the previous cycle. The CPU path does NOT consult the ack
    ///   slots: the bridge's `SyncAck` carries no meaningful
    ///   `step_count`, so gating CPU on the acks (and faking a large
    ///   `step_count` to satisfy them) poisoned `last_step_count` and
    ///   permanently wedged the re-arm gate after the warmup window.
    pub fn should_average(&self) -> bool {
        // CPU re-arm: don't re-trigger while a cycle is still in flight.
        if self.cycle.cpu_pending() {
            return false;
        }
        // active_count must be > 0 — if every rank is dead, training
        // is over (caller's responsibility to detect that separately).
        if self.active_count == 0 {
            return false;
        }
        // Count-based gate: fire once every alive rank has completed its
        // scheduled `batch_counts[r]` (Sync: any step). Timing is a
        // measurement that feeds `batch_counts` via
        // `ElChe::recompute_batch_counts`, NOT a firing condition — a
        // wall-time gate self-reinforces into deadlock (the target derives
        // from samples that only land when the gate fires).
        //
        // EXCEPTION — quiesced tail ranks: the edge schedule can hand a rank
        // fewer (or zero) steps in the final window of an epoch, after which
        // it can step no more this epoch (no in-flight chunk + its pool
        // drained). Such a rank is treated as satisfied (excluded from the
        // firing condition) — it still joins the collective on
        // `RequestParams`, contributing weight 0 via sum-and-count. Without
        // this, a zero-step tail rank blocks the gate forever while the
        // movers sit HELD at the reduce barrier (`steps_since_avg` never
        // resets) — a wedge. `any_mover` keeps a fully-idle window (no rank
        // stepped) from firing a degenerate empty reduce.
        let counts = self.el_che.batch_counts();
        let mut any_mover = false;
        for (r, &count) in counts.iter().enumerate() {
            // Skip dead ranks: they won't ack / step / accumulate wall_ms.
            if self.is_dead(r) {
                continue;
            }
            // NCCL re-arm only; CPU re-arms via the Pending phase above.
            if matches!(self.backend, AverageBackend::Nccl) && !self.cycle.acked[r] {
                return false;
            }
            let steps = self.window.steps(r);
            if steps > 0 {
                any_mover = true;
            }
            let hit_count = match self.policy {
                ApplyPolicy::Sync => steps >= 1,
                ApplyPolicy::Cadence | ApplyPolicy::Async => steps >= count,
            };
            // ...OR a quiesced tail rank (progressive only — Sync has no
            // chunk pool and trains a whole epoch per loop, so a sub-count
            // rank there means "still training", never quiesced).
            if !(hit_count || self.progressive && self.is_rank_quiesced(r)) {
                return false;
            }
        }
        any_mover
    }

    /// A rank can take no more steps in its current epoch: no chunk is in
    /// flight for it anywhere AND its epoch's pool is drained. Used by
    /// [`Self::should_average`] to exclude edge-schedule zero/short tail
    /// ranks from the firing gate (so they cannot block the reduce while
    /// the movers sit at the barrier). The rank's epoch pool must EXIST and
    /// be drained — a missing pool is NOT quiesced (we can't conclude the
    /// rank is done; `should_average` runs before `drain_metrics_and_aggregate`
    /// removes a finished epoch's pool, so at a real tail the drained pool is
    /// still present).
    fn is_rank_quiesced(&self, r: usize) -> bool {
        let no_in_flight = !self.chunk_pools.values().any(|p| p.in_flight(r) > 0);
        let epoch = self.rank_epoch[r];
        let pool_drained = self
            .chunk_pools
            .get(&epoch)
            .is_some_and(|p| p.remaining() == 0);
        no_in_flight && pool_drained
    }

    /// Throttle fast workers. NCCL backend is a no-op (the collective
    /// itself coordinates pacing); CPU backend defers to ElChe's
    /// rebalancer instead of explicit throttle frames.
    pub fn check_throttle(&mut self) -> Result<()> {
        if matches!(self.backend, AverageBackend::Nccl) {
            return Ok(());
        }
        let max_diff = match self.el_che.max_batch_diff() {
            Some(d) => d,
            None => return Ok(()),
        };
        if self.active_count < self.world_size {
            return Ok(());
        }
        let min_steps = self.window.min_steps();
        // Snapshot to avoid borrow-conflict on self.control_streams in send.
        let mut to_throttle: Vec<usize> = Vec::new();
        for (rank, &steps) in self.window.steps_all().iter().enumerate() {
            let should = steps > min_steps + max_diff;
            if should && !self.cycle.is_throttled(rank) {
                to_throttle.push(rank);
            }
        }
        for rank in to_throttle {
            self.send_control(rank, &ControlMsgWire::Throttle)?;
            self.cycle.set_throttled(rank);
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::Throttle { rank });
            }
        }
        Ok(())
    }

    /// Broadcast a [`ControlMsgWire::CoordHeartbeat`] to every rank, throttled
    /// to [`COORD_HEARTBEAT_CADENCE`]. Fires independently of training traffic
    /// so a rank can tell a legitimately-silent coordinator (mid-compute) from
    /// a wedged-open one.
    ///
    /// Deliberately infallible: the result of `broadcast_control` is logged at
    /// verbose and dropped. Propagating it would let a single transient send
    /// error abort the whole tick loop (`Err(e) => break` at the call site),
    /// and a genuinely unreachable rank is already handled by heartbeat
    /// staleness. Headless test coordinators (no control streams) short-circuit
    /// before attempting any send.
    pub(super) fn emit_coord_heartbeat(&mut self) {
        if self.control_streams.is_empty() {
            return; // headless coord (test fixtures) — nothing to beacon to
        }
        let due = self
            .last_coord_heartbeat
            .is_none_or(|t| t.elapsed() >= COORD_HEARTBEAT_CADENCE);
        if !due {
            return;
        }
        if let Err(e) = self.broadcast_control(&ControlMsgWire::CoordHeartbeat) {
            crate::verbose!("  ddp: coord heartbeat broadcast (best-effort): {e}");
        }
        self.last_coord_heartbeat = Some(Instant::now());
    }

    /// One coordinator tick: drain incoming timing, throttle fast
    /// workers, and trigger averaging when due. Mirrors OLD
    /// `Coordinator::tick`. Returns `false` when every reader thread
    /// has exited so the caller can break its loop.
    pub fn tick(&mut self) -> Result<bool> {
        self.drain_timing();
        // Coord→rank liveness beacon (throttled to ~1s). Best-effort and
        // error-swallowing on purpose: a transient send failure must never
        // abort the tick loop, and a wedged rank's failure is already reaped
        // by the staleness scan. The reverse-direction twin of the rank→coord
        // heartbeat, letting each rank detect a wedged-open coordinator.
        self.emit_coord_heartbeat();
        // Heartbeat-driven dead-rank detection. Must run BEFORE
        // poll_cpu_averaging / should_average so the rest of this
        // tick already sees the updated active membership; a rank
        // declared dead this tick won't gate the cycle's finalize.
        // No-op when elastic membership isn't configured.
        self.check_dead_ranks();
        // Cascading-death + slow-generator guard: an in-flight NCCL
        // rendezvous whose generator died (or stopped responding) would
        // hang the cohort indefinitely. Retries from the next survivor
        // candidate, or falls back to ShutdownWithSave when exhausted.
        // Runs independently of the dead-ranks ledger so a synthetic
        // pending state (test seam) is also exercised; production-side
        // pending only ever comes from `initiate_nccl_rendezvous_if_needed`
        // which already requires the ledger.
        self.check_rendezvous_timeout();
        self.check_throttle()?;
        // CPU-backend async finalize: if a cycle's `RequestParams` was
        // broadcast in a prior tick and all bridge SyncAcks have now
        // arrived (alive ranks only), finalize it here.
        // No-op on NCCL backend (state stays `Idle`).
        self.poll_cpu_averaging()?;
        // NCCL twin: escalate an alive-but-wedged in-flight collective
        // (SyncNow broadcast, acks never completing) past its
        // ceiling. No-op on CPU backend and whenever no sync is in flight.
        self.poll_nccl_reduce_stall()?;
        if self.should_average() {
            self.trigger_averaging()?;
        }
        // Stall watchdog (debug): capture the should_average gate state
        // if no reduce has fired for a while (the tight-window cadence
        // wedge). No-op until the threshold trips.
        self.maybe_dump_stall();
        // The mpsc returns Disconnected when every cloned sender has
        // dropped (every reader thread has exited). drain_timing alone
        // can't see that — try_recv just returns Empty if there's no
        // current message and the channel is healthy. Probe explicitly.
        //
        // The reader-handle check uses `is_finished()` (not just
        // `is_some()`) because handles are never taken during the tick
        // loop — they only get taken in `shutdown()` / `Drop`. So
        // `is_some()` alone reduces the alive check to
        // `active_count > 0`, and if a rank exits without the coord
        // receiving its `Exiting` frame (TCP RST during teardown, or
        // any other lossy close), `active_count` never decrements and
        // the coord runs forever — hanging the bench's metrics_rx.
        // `is_finished()` reflects the reader thread's actual exit:
        // when the worker closes its stream, the reader sees EOF and
        // returns, and the coord then shuts down regardless of whether
        // Exiting was received.
        let any_reader_running = self.reader_handles.iter().any(|h| {
            h.as_ref().is_some_and(|j| !j.is_finished())
        });
        let alive = self.active_count > 0 && any_reader_running;
        // Drain metrics + try to aggregate completed epochs every tick.
        // Cheap: most ticks see an empty channel; on tick where every
        // alive rank has reported the same epoch, one `EpochMetrics`
        // is built and `metrics_fn` fires.
        self.drain_metrics_and_aggregate();
        // Post-aggregate epoch transition: dispatch the next epoch's
        // `StartEpoch` plan (non-progressive, non-Async), or broadcast
        // `Shutdown` when the final epoch has aggregated. Deferred
        // until any pending CPU averaging cycle has finalized so the
        // bridge SyncAck round-trip can complete (see method docs).
        self.try_advance_or_shutdown_after_aggregate();
        Ok(alive)
    }

    /// Drain pending [`crate::distributed::wire::MetricsMsgWire`]
    /// frames from the reader threads. In non-progressive mode,
    /// aggregate once every alive rank has reported for the same
    /// epoch. In progressive mode, accumulate per-chunk reports per
    /// epoch + dispatch the next chunk to the reporting rank, then
    /// aggregate when the epoch's `ChunkPool::is_epoch_done()` fires
    /// (and only in ascending epoch order — a fast rank streaming
    /// ahead can't aggregate while earlier epochs still have
    /// in-flight chunks on the slow rank).
    ///
    /// In all paths: build [`crate::distributed::ddp_run::EpochMetrics`]
    /// and fire the user-supplied `metrics_fn` + `metrics_sink_tx`.
    /// Per-epoch buffers are dropped on aggregation. Late frames from
    /// a dead rank are ignored.
    pub(super) fn drain_metrics_and_aggregate(&mut self) {
        self.drain_metrics();
        self.aggregate_ready_epochs();
    }

    /// Drain pending metrics frames WITHOUT aggregating epochs: per-message
    /// bookkeeping only (pool `mark_completed`, metrics buffering,
    /// progressive next-chunk dispatch).
    ///
    /// Split out so `trigger_averaging`'s NCCL arm can pull the CURRENT
    /// window's completions into the delivered accounting BEFORE its inline
    /// `finish_averaging_nccl` consumes the window report — the staleness that
    /// previously forced NCCL onto the compute-only feed (allocation blind
    /// to the x1-link data/transport cost → fast rank idle ~45% at the
    /// barrier). Aggregation — and its pool removal — stays in
    /// [`Self::aggregate_ready_epochs`], preserving the
    /// `should_average`-sees-drained-pool ordering that
    /// [`Self::is_rank_quiesced`] depends on. Safe to call repeatedly per
    /// tick (`try_recv` drain); dispatch attempts for held ranks are no-ops
    /// (the reduce / epoch barriers hold them).
    pub(super) fn drain_metrics(&mut self) {
        // Track ranks that received a chunk-complete report so we can
        // dispatch the next chunk after the borrow on `chunk_pools` /
        // `metrics_buffer` is released.
        let mut progressive_completions: Vec<(usize, usize)> = Vec::new();
        while let Ok(wire) = self.metrics_rx.try_recv() {
            let rank = wire.rank as usize;
            if rank >= self.world_size {
                continue;
            }
            // Forward the rank's resource sample (if present) before
            // consuming `wire`. The sample piggy-backs on MetricsMsgWire
            // so we get it for free here. Two independent consumers:
            // the dashboard sink renders per-rank hardware tabs, and
            // the timeline persists the sample host-qualified into
            // `timeline.json`'s `rank_samples` (the only record of
            // remote hosts' GPU/VRAM activity — the harness's local
            // poller cannot see them).
            if let Some(wire_sample) = wire.resources.clone() {
                self.absorb_resource_sample(rank, wire_sample);
            }
            let msg = crate::distributed::ddp_run::MetricsMsg {
                rank,
                epoch: wire.epoch as usize,
                avg_loss: wire.avg_loss,
                batches_processed: wire.batches_processed as usize,
                epoch_ms: wire.epoch_ms,
                samples_processed: wire.samples_processed as usize,
                share_complete_ms: wire.share_complete_ms,
                compute_only_ms: wire.compute_only_ms,
                data_starve_ms: wire.data_starve_ms,
                scalars: wire.scalars
                    .into_iter()
                    .map(|(k, (sum, count))| (k, (sum, count as usize)))
                    .collect(),
            };
            if self.progressive {
                if let Some(pool) = self.chunk_pools.get_mut(&msg.epoch) {
                    pool.mark_completed(rank, msg.samples_processed);
                }
                progressive_completions.push((rank, msg.epoch));
            }
            // Growth bound: entries drain in `aggregate_ready_epochs`,
            // whose readiness recomputes `alive` per pass. A rank that
            // stops reporting either stops heartbeating too (declared
            // dead -> the epochs it was blocking aggregate without it)
            // or is stuck mid-epoch (produces no further epochs, so the
            // buffer stops growing). Metrics ride the same TCP control
            // stream as liveness — silent loss without disconnect is
            // not a failure mode this buffer needs to defend against.
            self.metrics_buffer
                .entry(wire.epoch)
                .or_default()
                .push(msg);
        }

        // Progressive: dispatch the next chunk to every rank that just
        // reported a chunk completion. Done after the drain loop so
        // we're not borrowing chunk_pools / metrics_buffer.
        if self.progressive {
            for (rank, _epoch) in progressive_completions {
                self.dispatch_next_chunk(rank);
            }
        }
    }

    /// Aggregate any epochs whose metrics are complete — the second half of
    /// [`Self::drain_metrics_and_aggregate`]: readiness resolution,
    /// `EpochMetrics` build + `metrics_fn`/sink fire, buffer + pool removal,
    /// and the post-aggregate dispatch hooks.
    pub(super) fn aggregate_ready_epochs(&mut self) {
        // Resolve readiness per dispatch mode.
        let alive: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_dead(*r))
            .collect();
        let ready_epochs: Vec<u64> = if self.progressive {
            // BTreeMap order: walk chunk_pools in ascending epoch
            // order, collecting done ones, STOPPING at the first
            // not-done (so a fast rank streaming ahead can't aggregate
            // before slower ranks finish earlier epochs).
            let mut ready: Vec<u64> = Vec::new();
            for (&epoch, pool) in &self.chunk_pools {
                if pool.is_epoch_done() {
                    ready.push(epoch as u64);
                } else {
                    break;
                }
            }
            ready
        } else {
            // Non-progressive: every alive rank emits exactly one
            // MetricsMsg per epoch. Epoch is ready when each alive
            // rank appears at least once in the Vec.
            self.metrics_buffer
                .iter()
                .filter_map(|(&epoch, msgs)| {
                    if alive.iter().all(|&r| msgs.iter().any(|m| m.rank == r)) {
                        Some(epoch)
                    } else {
                        None
                    }
                })
                .collect()
        };

        for epoch_key in ready_epochs {
            let msgs = match self.metrics_buffer.remove(&epoch_key) {
                Some(v) => v,
                None => continue,
            };
            // Pool-derived epoch wall-time wins in progressive mode
            // (the pool's `epoch_start` is the only authority); in
            // non-progressive the worker-reported max stands.
            let epoch_ms_override = if self.progressive {
                self.chunk_pools.remove(&(epoch_key as usize))
                    .map(|p| p.epoch_elapsed_ms())
            } else {
                None
            };
            // bc_share: ElChe's smoothed per-rank batch allocation
            // (the cadence's actual partition). For Sync mode ElChe is
            // not driving partitions so this collapses to equal share,
            // which is the right answer for that policy. For
            // Cadence / Async, this surfaces the real per-rank share —
            // matching what the dashboard / `EpochMetrics.per_rank_batch_share`
            // consumer expects to see when partition balancing is on.
            // `recent_batch_share` falls back to equal split when no
            // timing snapshots have landed yet (cold-start epoch 0),
            // so it never produces NaN.
            let bc_share = self.el_che.recent_batch_share();
            let mut metrics = crate::distributed::ddp_run::aggregate_epoch_metrics(
                epoch_key as usize,
                &msgs,
                &self.metrics_device_indices,
                &bc_share,
            );
            if let Some(ms) = epoch_ms_override {
                metrics.epoch_ms = ms;
            }
            self.last_aggregated_epoch = Some(epoch_key as usize);
            // Drain the per-epoch d-aggregator and emit `DivergenceEpoch`
            // when at least one AllReduce contributed a sample. Lambda
            // fields are intentionally None — `ddp-bench/src/analyze/msf.rs`
            // recomputes guard-specific λ̂ from the per-event Divergence
            // observables.
            let snap = self.take_epoch_d_summary();
            if snap.count > 0
                && let Some(ref tl) = self.timeline
            {
                tl.event(crate::monitor::EventKind::DivergenceEpoch {
                    epoch: epoch_key as usize,
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
            if let Some(ref f) = self.metrics_fn {
                if let Err(e) = f(&metrics) {
                    eprintln!(
                        "cluster_coordinator: metrics_fn returned error (epoch {epoch_key}): {e}"
                    );
                }
            }
            // Broadcast the aggregated view back to every rank so the
            // cooperative (`into_worker`) user loop's
            // `monitor.log(&model)` sees the cross-rank picture
            // (global scalars + per-rank GPU tabs). The
            // `Trainer::builder().run()` path already had this via the
            // sink tx; this broadcast gives the same UX to cooperative
            // users in process-per-rank cluster runs. Broadcast
            // failures are non-fatal — a rank's stream may have
            // already closed during shutdown; surface as verbose.
            let wire_metrics: crate::distributed::wire::EpochMetricsWire =
                metrics.clone().into();
            if let Err(e) = self.broadcast_control(
                &ControlMsgWire::EpochAggregated(wire_metrics),
            ) {
                crate::verbose!(
                    "  ddp: EpochAggregated broadcast (epoch {epoch_key}) failed: {}",
                    e,
                );
            }
            // Forward to the controller-side dashboard sink (when the
            // launcher hosts a dashboard). Same aggregated value the
            // user's `metrics_fn` / `metrics_sink_tx` receive — the
            // dashboard surfaces it as the main-tab time series.
            if let Some(ref sink) = self.dashboard_sink {
                sink.push_epoch_metrics(&metrics);
            }
            if let Some(ref tx) = self.metrics_sink_tx {
                // Sink receiver dropped is benign — handle was
                // dropped before training finished. Don't surface
                // as an error.
                let _ = tx.send(metrics);
            }
        }

        // Epoch transition is handled by `try_advance_or_shutdown_after_aggregate`
        // which is invoked from `tick()` after `poll_cpu_averaging` has had
        // a chance to drive a still-pending CPU averaging cycle to Idle.
        // Calling it here directly would race with bridge SyncAcks: the
        // worker batch loop is async in cluster mode (Batch send and
        // MetricsMsg are not serialized against RequestParams/SyncAck),
        // so MetricsMsg can land while the CPU cycle is still Pending and
        // shutting workers down at that point would drop the in-flight
        // cycle.
    }

    /// Post-aggregation epoch transition. Once an epoch has aggregated
    /// AND any in-flight averaging cycle has finalized
    /// (CPU phase back at `Idle`), either dispatch the next epoch
    /// (non-progressive, non-Async) or broadcast `Shutdown` (final
    /// epoch). Idempotent: tracks `last_dispatched_epoch` so repeated
    /// ticks past the final aggregate don't re-broadcast `Shutdown`,
    /// and a still-pending CPU cycle simply defers the call until the
    /// next tick once the cycle finalizes.
    ///
    /// Split from `drain_metrics_and_aggregate` because the cluster
    /// path is async: workers can post-send `MetricsMsg` while the
    /// previous batch's bridge SyncAck is still in transit.
    /// Whether end-of-training must force one final consensus reduce
    /// before shutdown: at least two alive ranks (a lone survivor's weights
    /// are already canonical, and NCCL needs `world_size >= 2`) AND some
    /// alive rank still carries un-reduced trailing steps from the edge
    /// schedule (`steps_since_avg > 0`). When false, the cohort is already
    /// coherent (the last regular reduce landed on the boundary).
    pub(super) fn needs_final_consensus_reduce(&self) -> bool {
        self.active_count >= 2
            && (0..self.world_size)
                .any(|r| !self.is_dead(r) && self.window.steps(r) > 0)
    }

    pub(super) fn try_advance_or_shutdown_after_aggregate(&mut self) {
        if self.run_phase == RunPhase::ShutdownInitiated {
            return;
        }
        let Some(latest) = self.last_aggregated_epoch else {
            return;
        };
        // Wait for any in-flight CPU averaging cycle to finalize before
        // either dispatching a new epoch (which would race with the
        // pending SyncAck round-trip) or shutting workers down (which
        // would drop the cycle and leave the divergence guard with
        // all-Nones — see `end_to_end_sync_cpu_smoke` regression).
        if self.cycle.cpu_pending() {
            return;
        }
        let next = latest + 1;
        if next >= self.num_epochs {
            // COHERENT FINAL MODEL. The edge schedule can leave trailing
            // steps that never filled a full reduce window, so
            // `should_average` never fired for them — un-reduced drift that
            // would make the saved/evaled model depend on which rank's copy
            // we happen to read. Before shutting down, force ONE final
            // reduce so every rank converges to a single consensus that
            // accounts for ALL steps (the weighted sum-and-count excludes
            // ranks that did 0 tail steps). Same coherence the final
            // checkpoint + the single eval need.
            //
            // Safe against the teardown deadlock: an idle rank sits in
            // `wait_for_epoch_plan` SERVICING control messages (SyncNow /
            // RequestParams) until `Shutdown`, so the collective completes
            // with full participation; mpsc is FIFO so the reduce frame is
            // dequeued before `Shutdown`, and `sync_now_nccl` synchronizes
            // before returning. The CPU-phase-Idle gate above plus
            // the per-tick `try_advance` let the async CPU reduce finalize
            // (which resets `steps_since_avg`) before this re-enters and
            // shuts down; NCCL's finish is inline so its reset lands at once.
            // Skip for a lone survivor (< 2 alive): a single rank's weights
            // ARE the consensus, and NCCL needs world_size >= 2.
            if self.needs_final_consensus_reduce() {
                match self.trigger_averaging() {
                    Ok(()) => return,
                    Err(e) => {
                        // Don't hang the run on a final-reduce failure:
                        // log and fall through to shutdown with whatever
                        // each rank currently holds.
                        crate::verbose!(
                            "  ddp: final consensus reduce failed, \
                             shutting down without it: {e}",
                        );
                    }
                }
            }
            // SINGLE CANONICAL EVAL. Every rank now holds the coherent
            // consensus (the final reduce just landed, or there were no
            // trailing steps), so dispatch ONE eval to the controller-chosen
            // rank (`EpochCallbackPolicy::Fastest` by default) — the final
            // metric is measured once on the canonical model, not redundantly
            // on every rank. The scalar flows back via
            // `TimingMsg::EvalResult` → `eval_result_fn`; mpsc is FIFO so the
            // rank evals before it processes the `Shutdown` sent next tick,
            // and the coordinator's teardown ticks drain the result. Only
            // when an `eval_result_fn` is wired and the chosen rank is alive.
            if self.eval_result_fn.is_some()
                && self.run_phase == RunPhase::Training
                && self.eval_role < self.world_size
                && !self.is_dead(self.eval_role)
            {
                self.run_phase = RunPhase::FinalEvalDispatched;
                let msg = ControlMsgWire::ExecuteEvalCallback {
                    schedule_id: u64::MAX, // sentinel: the final canonical eval
                    epoch: self.num_epochs as u64,
                    target_rank: self.eval_role as u64,
                };
                if let Err(e) = self.send_control(self.eval_role, &msg) {
                    crate::verbose!("  ddp: final eval dispatch failed: {e}");
                }
                // Give the chosen rank a tick to eval + ship EvalResult
                // before Shutdown goes out (next tick, flag set → shutdown).
                return;
            }
            // COORDINATOR-CONFIRMED EXIT: never broadcast `Shutdown`
            // while an NCCL sync is still in flight. The final consensus
            // reduce (or the last regular window's reduce) can have some
            // ranks' collective kernels still running when another rank
            // has already acked — NCCL's LL small-message protocol does
            // not synchronize kernel completion across ranks. A rank
            // that takes `Shutdown` at that point exits its process and
            // destroys its NCCL comm mid-collective, stranding every
            // peer in `synchronize()` at 100% GPU forever (observed as
            // the ~1/6 end-of-run cadence wedge on the 3-rank rig). The
            // per-rank control channel is FIFO, so gating the broadcast
            // on all-alive-acked guarantees every rank has fully retired
            // the collective before any rank can see `Shutdown`.
            // Idempotent retry: this method runs every tick, so the
            // broadcast fires on the first settled tick; a sync that
            // never settles (rank wedged/lost mid-collective) is ended
            // by `poll_nccl_reduce_stall`'s ceiling escalation instead.
            if !self.nccl_sync_settled() {
                return;
            }
            self.run_phase = RunPhase::ShutdownInitiated;
            if let Err(e) = self.shutdown_workers() {
                crate::verbose!(
                    "  ddp: shutdown_workers after final aggregate failed: {}",
                    e,
                );
            }
        } else if self.progressive {
            // Streaming-epoch re-dispatch: any alive rank that hit the
            // overshoot gate (or otherwise finished its last chunk and
            // is sitting in `wait_for_epoch_plan`) has no MetricsMsg
            // arriving to drive dispatch_next_chunk via
            // drain_metrics_and_aggregate. The overshoot reset happens
            // at averaging — but reset alone doesn't kick a stalled
            // rank back into motion. After every epoch aggregate, walk
            // alive ranks and dispatch_next_chunk to any with no
            // in-flight chunks across any pool. Without this, multi-epoch progressive
            // runs (cpu-cadence, cpu-async, nccl-cadence)
            // wedge after epoch 0 once the calibrated `batch_counts`
            // pulls the fast rank past its planned + max_overshoot
            // budget.
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
        } else if !matches!(self.policy, ApplyPolicy::Async)
            && self.last_dispatched_epoch.is_none_or(|d| d < next)
        {
            // Latch ON SUCCESS only. Latching before the dispatch turned
            // any dispatch error into a permanent wedge: the idempotence
            // guard suppressed every retry while the cohort parked in
            // `wait_for_epoch_plan` with live heartbeats. Per-rank send
            // failures inside `dispatch_epoch` are best-effort (logged,
            // dead-rank machinery reaps them), so an `Err` here is a hard
            // config/state problem — still worth retrying on the next
            // aggregate tick rather than silently never dispatching again.
            match self.dispatch_epoch(next) {
                Ok(_) => self.last_dispatched_epoch = Some(next),
                Err(e) => {
                    eprintln!(
                        "flodl ddp: dispatch_epoch({next}) after aggregate \
                         failed: {e}; will retry"
                    );
                }
            }
        }
    }
}
