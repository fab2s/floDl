//! Coordinator event-loop core for [`super::ClusterCoordinator`]:
//! `tick`, the timing-message switchboard, the per-cycle metrics
//! aggregator, throttle dispatch, and the post-aggregate advance /
//! shutdown decision.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::{ControlMsgWire, TimingMsgWire};
use crate::tensor::Result;

use super::{ClusterCoordinator, CpuAvgState};

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
            | TimingMsgWire::Heartbeat { rank, .. }
            | TimingMsgWire::SnapshotReady { rank }
            | TimingMsgWire::NewNcclIdGenerated { rank, .. }
            | TimingMsgWire::EvalResult { rank, .. }
            | TimingMsgWire::CheckpointResult { rank, .. }
            | TimingMsgWire::EpochFnElapsed { rank, .. }
            | TimingMsgWire::DashboardRegister { rank, .. }
            | TimingMsgWire::DashboardSetSvg { rank, .. }
            | TimingMsgWire::DashboardSetMetadata { rank, .. }
            | TimingMsgWire::DashboardSetHardware { rank, .. } => Some(*rank as usize),
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
                self.steps_since_avg[rank] =
                    self.steps_since_avg[rank].saturating_add(1);
                self.wall_ms_accum[rank] += batch_ms;
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                self.last_batch_ms[rank] = batch_ms;
                let _ = batch_loss; // monitoring only in this slice
                let _ = param_norm;
                if let Some(div) = sync_divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    self.capture_nccl_sync_elapsed_if_complete();
                }
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
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                if let Some(div) = divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if let Some(p) = pre_norm {
                    self.nccl_sync_pre_norm[rank] = Some(p);
                }
                if let Some(p) = post_norm {
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
                    // Per-rank sync lag (wall time from
                    // `RequestParams` / `SyncNow` broadcast to this
                    // rank's SyncAck). Captured BEFORE
                    // `capture_nccl_sync_elapsed_if_complete` takes
                    // `nccl_sync_start` on the all-acked transition.
                    // Feeds the adaptive CPU deadline computed in the
                    // NEXT `trigger_averaging`.
                    if let Some(start) = self.nccl_sync_start {
                        if rank < self.last_observed_sync_lag_ms.len() {
                            self.last_observed_sync_lag_ms[rank] =
                                Some(start.elapsed().as_secs_f64() * 1000.0);
                        }
                    }
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsgWire::Exiting { rank: _ } => {
                self.active_count = self.active_count.saturating_sub(1);
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
                // `last_observed_sync_lag_ms` as "planned upload-
                // completion marker".
                //
                // `nccl_sync_start` is the broadcast anchor; if the
                // cycle has already finalized (all SyncAcks in,
                // `capture_nccl_sync_elapsed_if_complete` took the
                // anchor), this frame is a late-arriving stragger
                // and we drop it — `last_observed_upload_ms[rank]`
                // keeps the prior value or None.
                let rank = rank as usize;
                if rank < self.last_observed_upload_ms.len() {
                    if let Some(start) = self.nccl_sync_start {
                        self.last_observed_upload_ms[rank] =
                            Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
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
    /// loop. Mirrors OLD `Coordinator::drain_timing_blocking`.
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

    /// Check whether an averaging cycle should be triggered now. Ported
    /// literally from OLD `Coordinator::should_average`.
    ///
    /// `nccl_ack` is named for the NCCL path's SyncAck mechanism but
    /// serves both backends in the new TCP model: workers send a
    /// `TimingMsg::SyncAck` after every averaging round (regardless of
    /// backend) so the coordinator can gate re-triggering until the
    /// previous round has settled.
    pub fn should_average(&self) -> bool {
        // Every gate skips dead ranks: they won't ack, won't step, and
        // won't accumulate wall_ms. Treating them as "satisfied" lets
        // the surviving cohort keep training.
        for r in 0..self.world_size {
            if self.is_dead(r) {
                continue;
            }
            if !self.nccl_ack[r] {
                return false;
            }
            if self.steps_since_avg[r] == 0 {
                return false;
            }
        }
        // active_count must be > 0 — if every rank is dead, training
        // is over (caller's responsibility to detect that separately).
        if self.active_count == 0 {
            return false;
        }
        match self.policy {
            ApplyPolicy::Sync => (0..self.world_size)
                .filter(|r| !self.is_dead(*r))
                .all(|r| self.steps_since_avg[r] >= 1),
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                // Count-based gate: fire when each rank has completed its
                // scheduled `batch_counts[r]`. The phenomenological
                // invariant is "training progresses by scheduled steps" —
                // timing is a measurement that feeds `batch_counts` via
                // `ElChe::recompute_batch_counts`, NOT a firing condition.
                // A wall-time gate (`min_wall >= anchor * smoothed_slow_ms`)
                // is structurally fragile: the target is derived from
                // samples that only land when the gate fires, so any
                // upward spike in `smoothed_slow_ms` (cold-start warmup,
                // thermal throttle, GPU contention, mid-run lazy init)
                // can lock the target above achievable wall time and
                // deadlock the cohort. Count-based gating sidesteps that
                // loop entirely.
                let counts = self.el_che.batch_counts();
                (0..self.world_size)
                    .filter(|r| !self.is_dead(*r))
                    .all(|r| self.steps_since_avg[r] >= counts[r])
            }
        }
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
        let min_steps = self.steps_since_avg.iter().copied().min().unwrap_or(0);
        // Snapshot to avoid borrow-conflict on self.control_streams in send.
        let mut to_throttle: Vec<usize> = Vec::new();
        for (rank, &steps) in self.steps_since_avg.iter().enumerate() {
            let should = steps > min_steps + max_diff;
            if should && !self.throttled[rank] {
                to_throttle.push(rank);
            }
        }
        for rank in to_throttle {
            self.send_control(rank, &ControlMsgWire::Throttle)?;
            self.throttled[rank] = true;
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::Throttle { rank });
            }
        }
        Ok(())
    }

    /// One coordinator tick: drain incoming timing, throttle fast
    /// workers, and trigger averaging when due. Mirrors OLD
    /// `Coordinator::tick`. Returns `false` when every reader thread
    /// has exited so the caller can break its loop.
    pub fn tick(&mut self) -> Result<bool> {
        self.drain_timing();
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
        if self.should_average() {
            self.trigger_averaging()?;
        }
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
        // Track ranks that received a chunk-complete report so we can
        // dispatch the next chunk after the borrow on `chunk_pools` /
        // `metrics_buffer` is released.
        let mut progressive_completions: Vec<(usize, usize)> = Vec::new();
        while let Ok(wire) = self.metrics_rx.try_recv() {
            let rank = wire.rank as usize;
            if rank >= self.world_size {
                continue;
            }
            // Forward the rank's resource sample (if present) to the
            // dashboard sink before consuming `wire`. The sample piggy-
            // backs on MetricsMsgWire so we get it for free here; the
            // sink renders per-rank hardware tabs.
            if let (Some(sink), Some(sample)) = (&self.dashboard_sink, wire.resources.clone()) {
                sink.push_resource_sample(rank, sample);
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

        // Resolve readiness per dispatch mode.
        let alive: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_rank_dead(*r))
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
            // observables. Mirrors threaded `coordinator/mod.rs:934-951`.
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
            // user-owned `Trainer::setup` training loop's
            // `monitor.log(&model)` sees the cross-rank picture
            // (global scalars + per-rank GPU tabs). The
            // `Trainer::builder` path already had this via the sink
            // tx; this broadcast gives the same UX to setup-mode
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
        // so MetricsMsg can land while `cpu_avg_state == Pending` and
        // shutting workers down at that point would drop the in-flight
        // cycle.
    }

    /// Post-aggregation epoch transition. Once an epoch has aggregated
    /// AND any in-flight averaging cycle has finalized
    /// (`cpu_avg_state == Idle`), either dispatch the next epoch
    /// (non-progressive, non-Async) or broadcast `Shutdown` (final
    /// epoch). Idempotent: tracks `last_dispatched_epoch` so repeated
    /// ticks past the final aggregate don't re-broadcast `Shutdown`,
    /// and a still-pending CPU cycle simply defers the call until the
    /// next tick once the cycle finalizes.
    ///
    /// Mirrors threaded `Coordinator::on_epoch_aggregated` (see
    /// `ddp_run/coordinator/mod.rs:924`) but split from
    /// `drain_metrics_and_aggregate` because the cluster path is
    /// async: workers can post-send `MetricsMsg` while the previous
    /// batch's bridge SyncAck is still in transit.
    pub(super) fn try_advance_or_shutdown_after_aggregate(&mut self) {
        if self.shutdown_initiated {
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
        if !matches!(self.cpu_avg_state, CpuAvgState::Idle) {
            return;
        }
        let next = latest + 1;
        if next >= self.num_epochs {
            self.shutdown_initiated = true;
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
            // in-flight chunks across any pool. Mirrors threaded
            // `Coordinator::on_epoch_aggregated` (ddp_run/coordinator/
            // mod.rs:978-988). Without this, multi-epoch progressive
            // runs (cpu-cadence, cpu-async, nccl-cadence, nccl-async)
            // wedge after epoch 0 once the calibrated `batch_counts`
            // pulls the fast rank past its planned + max_overshoot
            // budget.
            for rank in 0..self.world_size {
                if self.is_rank_dead(rank) {
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
            self.last_dispatched_epoch = Some(next);
            if let Err(e) = self.dispatch_epoch(next) {
                crate::verbose!(
                    "  ddp: dispatch_epoch({}) after aggregate failed: {}",
                    next,
                    e,
                );
            }
        }
    }
}
