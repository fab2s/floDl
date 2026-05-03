//! CPU averaging state machine.
//!
//! Instead of blocking the coordinator thread during snapshot collection and
//! averaging computation, the CPU path operates as a multi-tick state machine:
//!
//! ```text
//! Idle -> Collecting -> Computing -> Idle
//!          (try_recv      (thread
//!           per tick)      join)
//! ```
//!
//! This keeps [`Coordinator::check_throttle`] running every tick, even during
//! averaging. Counter snapshots taken at trigger time allow correct timing
//! attribution: batches during the averaging window carry into the next period.

use std::time::Instant;

use crate::tensor::{Device, Result, Tensor, TensorError};

use super::super::{
    ApplyPolicy, AverageBackend, ControlMsg, ParamSnapshot, AveragedParams,
    convergence,
};
use super::Coordinator;

pub(super) enum CpuAvgState {
    /// No averaging in progress. `should_average()` may trigger a new cycle.
    Idle,
    /// Waiting for worker snapshots. `try_recv` on `param_rx` each tick.
    Collecting {
        snapshots: Vec<ParamSnapshot>,
        received: Vec<bool>,
        deadline: Instant,
        start: Instant,
        /// `steps_since_avg` at trigger time (for subtract-not-zero).
        steps_snapshot: Vec<usize>,
        /// `wall_ms_accum` at trigger time (for subtract-not-zero).
        wall_ms_snapshot: Vec<f64>,
    },
    /// `average_params` + divergence check running on a background thread.
    Computing {
        handle: std::thread::JoinHandle<Result<CpuAvgResult>>,
        start: Instant,
        steps_snapshot: Vec<usize>,
        wall_ms_snapshot: Vec<f64>,
    },
}

/// Result from the CPU averaging compute thread.
pub(super) struct CpuAvgResult {
    averaged: AveragedParams,
    /// Weight-space divergence report (for convergence guard).
    /// None if all norms are zero.
    divergence: Option<convergence::DivergenceReport>,
}

impl Coordinator {
    /// Trigger parameter averaging based on the configured backend.
    ///
    /// For NCCL: sends `SyncNow` to all workers, then runs the common tail
    /// (ElChe report, version bump, counter reset) synchronously.
    ///
    /// For CPU: sends `RequestParams`, snapshots the current counters, and
    /// enters the `CpuAvgState::Collecting` state. The actual collection
    /// and averaging happen over subsequent [`Self::poll_cpu_averaging`] ticks,
    /// keeping [`Self::check_throttle`] active throughout.
    pub fn trigger_averaging(&mut self) -> Result<()> {
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::SyncStart);
        }
        match self.backend {
            AverageBackend::Nccl => {
                self.nccl_sync_start = Some(Instant::now());
                for tx in &self.control_txs {
                    let _ = tx.send(ControlMsg::SyncNow);
                }
                // Snapshot each rank's last seen worker step_count and
                // mark unacknowledged. should_average() won't fire again
                // until every rank sends a timing message with step_count
                // above this snapshot, proving it processed the SyncNow.
                for rank in 0..self.world_size {
                    self.nccl_sync_step[rank] = self.last_step_count[rank];
                    self.nccl_ack[rank] = false;
                }
                self.finish_averaging_nccl();
            }
            AverageBackend::Cpu => {
                // Snapshot counters at trigger time. Batches that arrive
                // during collection/computing carry into the next period.
                let steps_snapshot = self.steps_since_avg.clone();
                let wall_ms_snapshot = self.wall_ms_accum.clone();

                // Send RequestParams to all workers. In Sync mode, also
                // send Throttle so workers block after snapshotting (same
                // semantics as NCCL AllReduce blocking). Without this,
                // workers keep training with diverging params while the
                // CPU averaging state machine runs, then the Update
                // overwrites those steps, wasting compute and corrupting
                // the optimizer's momentum/variance state.
                let sync_throttle = matches!(self.policy, ApplyPolicy::Sync);
                for (rank, tx) in self.control_txs.iter().enumerate() {
                    let _ = tx.send(ControlMsg::RequestParams);
                    if sync_throttle {
                        let _ = tx.send(ControlMsg::Throttle);
                        self.throttled[rank] = true;
                    }
                }

                let timeout_secs = self.snapshot_timeout_secs;
                self.avg_state = CpuAvgState::Collecting {
                    snapshots: Vec::with_capacity(self.world_size),
                    received: vec![false; self.world_size],
                    deadline: Instant::now()
                        + std::time::Duration::from_secs(timeout_secs),
                    start: Instant::now(),
                    steps_snapshot,
                    wall_ms_snapshot,
                };
                // Return immediately; poll_cpu_averaging drives the rest.
            }
        }
        Ok(())
    }

    /// Common tail for NCCL averaging: report to ElChe, check divergence,
    /// bump version, zero counters.
    ///
    /// NCCL workers block in AllReduce so no new batches arrive during the
    /// collective; zeroing is correct.
    ///
    /// Weight-space divergence guard feeds into cadence control via
    /// [`convergence::ConvergenceGuard`]. The guard is mode-aware: Sync skips entirely,
    /// Cadence/Async use trend detection on `||pre-post|| / ||post||`.
    pub(in super::super) fn finish_averaging_nccl(&mut self) {
        let old_anchor = self.el_che.anchor();
        // Use the wall-time elapsed of the PREVIOUS NCCL sync (captured in
        // process_timing_msg when the last rank acked) as `sync_ms`. Without
        // this, NCCL's `last_avg_ms` stayed at 0 and the anchor auto-tune
        // block in el_che.rs:292-308 silently skipped. Result: anchor pinned
        // at min, syncs too frequent, AllReduce barrier wait dominates.
        let prev_sync_ms = self.last_nccl_sync_ms;
        self.last_nccl_sync_ms = 0.0;
        if self.wall_ms_accum.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(&self.wall_ms_accum, &self.steps_since_avg, prev_sync_ms);
            self.last_avg_ms = 0.0;
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        // Feed weight-space divergence (from previous sync's acks) to the guard.
        // Invariant: should_average() requires all nccl_ack == true before next
        // trigger, so all nccl_sync_divergence[rank] are populated by now.
        // Post-norm is identical across ranks post-AllReduce; we keep a single
        // scalar populated by the first rank's ack (debug_assert in the
        // SyncAck handler enforces inter-rank agreement).
        let nccl_pre_norms: Option<Vec<f64>> =
            if self.nccl_sync_pre_norm.iter().all(|p| p.is_some()) {
                Some(self.nccl_sync_pre_norm.iter().map(|p| p.unwrap()).collect())
            } else {
                None
            };
        let report = convergence::DivergenceReport {
            deltas: self.nccl_sync_divergence.iter()
                .map(|d| d.unwrap_or(0.0))
                .collect(),
            pre_norms: nccl_pre_norms,
            post_norm: self.nccl_sync_post_norm,
        };
        let action = self.convergence_guard.report(&report);

        self.version += 1;
        self.avg_count += 1;

        // Apply convergence action. Overshoot is an Async-only concept.
        match action {
            convergence::ConvergenceAction::Stable => {
                if self.policy == ApplyPolicy::Async {
                    if self.overshoot_auto {
                        let cap = (self.total_samples / self.batch_size.max(1) / 50).clamp(3, 10);
                        self.max_overshoot = (self.max_overshoot + 1).min(cap);
                    }
                    // Symmetric upward path for anchor: relax cadence on stable
                    // convergence (capped by max_batch_diff and max_anchor).
                    // Without this, anchor stays pinned at min_anchor forever
                    // because the overhead-based auto-tune lives in a dead
                    // zone for low-overhead workloads. Suppressed when the
                    // user disables relax-up (e.g. to isolate share-allocation
                    // dynamics from the periodic anchor cycle).
                    if self.elche_relax_up {
                        self.el_che.relax_anchor_up();
                    }
                }
            }
            convergence::ConvergenceAction::SuppressGrowth => {
                // Hold current anchor and overshoot, don't grow.
            }
            convergence::ConvergenceAction::NudgeDown { factor } => {
                self.el_che.nudge_anchor_down(factor);
                if self.overshoot_auto && self.policy == ApplyPolicy::Async {
                    self.max_overshoot = self.overshoot_initial;
                }
            }
        }
        // Absolute ceiling (safety valve, applied after all auto-tune logic).
        if self.policy == ApplyPolicy::Async {
            self.max_overshoot = self.max_overshoot.min(self.overshoot_ceiling);
        }

        // Timeline: sync duration and anchor changes. Uses the previous
        // sync's measured duration (captured when last rank acked). The
        // earlier code took `nccl_sync_start.elapsed()` here but that runs
        // synchronously inside trigger_averaging immediately after sending
        // SyncNow, before any rank can complete AllReduce — so the elapsed
        // was always ~0ms.
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::SyncEnd { duration_ms: prev_sync_ms });
            let new_anchor = self.el_che.anchor();
            if new_anchor != old_anchor {
                tl.event(crate::monitor::EventKind::AnchorChanged {
                    from: old_anchor,
                    to: new_anchor,
                });
            }
        }

        // Advance global step by total batches from all GPUs in this cycle.
        let cycle_batches: usize = self.steps_since_avg.iter().sum();
        let k_max = self.steps_since_avg.iter().copied().max().unwrap_or(0);
        self.global_step += cycle_batches;

        // MSF passive observation: lambda_hat sample for this AllReduce.
        // Reads quantities already in scope (D_t, k); no behavior change.
        let d_raw = report.max_relative_delta();
        let lambda_sample = self.convergence_guard.observe_lambda(d_raw, cycle_batches, k_max);
        let in_flight_epoch = self.last_aggregated_epoch.map(|e| e + 1).unwrap_or(0);
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::Divergence {
                d_raw,
                lambda_raw: lambda_sample.lambda_raw,
                lambda_ema: lambda_sample.lambda_ema,
                k_used: cycle_batches,
                k_max,
                step: self.global_step,
                deltas: report.deltas.clone(),
                post_norm: report.post_norm,
                pre_norms: report.pre_norms.clone(),
                epoch: Some(in_flight_epoch),
            });
        }

        // Broadcast new global step to all workers so they can compute
        // per-batch LR as scheduler.lr(global_step + local_offset).
        for tx in &self.control_txs {
            let _ = tx.send(ControlMsg::SetGlobalStep(self.global_step));
        }

        crate::verbose!(
            "  ddp: NCCL averaging #{} complete (v{}, global_step={})",
            self.avg_count, self.version, self.global_step
        );

        for s in &mut self.steps_since_avg {
            *s = 0;
        }
        for a in &mut self.wall_ms_accum {
            *a = 0.0;
        }
        for t in &mut self.throttled {
            *t = false;
        }
        // Reset per-interval accumulators.
        for l in &mut self.loss_accum {
            *l = 0.0;
        }
        for c in &mut self.loss_count {
            *c = 0;
        }
        for d in &mut self.nccl_sync_divergence {
            *d = None;
        }
        for p in &mut self.nccl_sync_pre_norm {
            *p = None;
        }
        self.nccl_sync_post_norm = None;

        // Re-dispatch to ranks that are idle (no in-flight chunks in any pool)
        // and may have been waiting at the overshoot gate. Now that
        // steps_since_avg is reset, the gate is open.
        if self.progressive {
            for rank in 0..self.world_size {
                let has_inflight = self.chunk_pools.values()
                    .any(|p| p.in_flight(rank) > 0);
                if !has_inflight {
                    self.dispatch_next_chunk(rank);
                }
            }
        }
    }

    // check_nccl_divergence removed: replaced by ConvergenceGuard in finish_averaging_nccl.

    /// Common tail for CPU averaging: report snapshot counters to ElChe,
    /// apply divergence correction, subtract snapshots from current counters
    /// (preserve during-averaging batches).
    pub(in super::super) fn finish_averaging_cpu(
        &mut self,
        avg_ms: f64,
        steps_snapshot: &[usize],
        wall_ms_snapshot: &[f64],
        divergence_report: Option<convergence::DivergenceReport>,
    ) {
        self.last_avg_ms = avg_ms;
        let old_anchor = self.el_che.anchor();

        // Report the snapshot values to ElChe (accurate for the period
        // that triggered averaging, not inflated by during-averaging batches).
        if wall_ms_snapshot.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(wall_ms_snapshot, steps_snapshot, self.last_avg_ms);
            self.last_avg_ms = 0.0;
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        // Feed divergence to the unified convergence guard.
        let action = if let Some(report) = divergence_report.as_ref() {
            self.convergence_guard.report(report)
        } else {
            convergence::ConvergenceAction::Stable
        };

        match action {
            convergence::ConvergenceAction::Stable => {
                if self.policy == ApplyPolicy::Async {
                    if self.overshoot_auto {
                        let cap = (self.total_samples / self.batch_size.max(1) / 50).clamp(3, 10);
                        self.max_overshoot = (self.max_overshoot + 1).min(cap);
                    }
                    if self.elche_relax_up {
                        self.el_che.relax_anchor_up();
                    }
                }
            }
            convergence::ConvergenceAction::SuppressGrowth => {}
            convergence::ConvergenceAction::NudgeDown { factor } => {
                self.el_che.nudge_anchor_down(factor);
                crate::verbose!(
                    "  ddp: weight-space divergence nudge, anchor {} -> {}",
                    old_anchor, self.el_che.anchor()
                );
                if self.overshoot_auto && self.policy == ApplyPolicy::Async {
                    self.max_overshoot = self.overshoot_initial;
                }
            }
        }
        if self.policy == ApplyPolicy::Async {
            self.max_overshoot = self.max_overshoot.min(self.overshoot_ceiling);
        }

        self.version += 1;
        self.avg_count += 1;

        // Timeline: CPU averaging completion and anchor changes
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::CpuAvgEnd { duration_ms: avg_ms });
            tl.event(crate::monitor::EventKind::SyncEnd { duration_ms: avg_ms });
            let new_anchor = self.el_che.anchor();
            if new_anchor != old_anchor {
                tl.event(crate::monitor::EventKind::AnchorChanged {
                    from: old_anchor,
                    to: new_anchor,
                });
            }
        }

        // Advance global step by total batches from all GPUs at trigger time.
        // Use snapshot (not current) counters -- batches during the averaging
        // window belong to the next cycle.
        let cycle_batches: usize = steps_snapshot.iter().sum();
        let k_max = steps_snapshot.iter().copied().max().unwrap_or(0);
        self.global_step += cycle_batches;

        // MSF passive observation: lambda_hat sample for this AllReduce.
        // Only emit when a divergence report was actually produced (Sync mode
        // skips divergence computation entirely).
        if let Some(ref report) = divergence_report {
            let d_raw = report.max_relative_delta();
            let lambda_sample =
                self.convergence_guard.observe_lambda(d_raw, cycle_batches, k_max);
            let in_flight_epoch = self.last_aggregated_epoch.map(|e| e + 1).unwrap_or(0);
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::Divergence {
                    d_raw,
                    lambda_raw: lambda_sample.lambda_raw,
                    lambda_ema: lambda_sample.lambda_ema,
                    k_used: cycle_batches,
                    k_max,
                    step: self.global_step,
                    deltas: report.deltas.clone(),
                    post_norm: report.post_norm,
                    pre_norms: report.pre_norms.clone(),
                    epoch: Some(in_flight_epoch),
                });
            }
        }

        // Broadcast new global step to all workers.
        for tx in &self.control_txs {
            let _ = tx.send(ControlMsg::SetGlobalStep(self.global_step));
        }

        crate::verbose!(
            "  ddp: CPU averaging #{} complete (v{}, {:.1}ms, global_step={})",
            self.avg_count, self.version, avg_ms, self.global_step
        );

        // Subtract snapshot from current counters. Residual = batches
        // that happened during the averaging window, carried forward.
        for (i, s) in self.steps_since_avg.iter_mut().enumerate() {
            *s = s.saturating_sub(steps_snapshot[i]);
        }
        for (i, a) in self.wall_ms_accum.iter_mut().enumerate() {
            *a = (*a - wall_ms_snapshot[i]).max(0.0);
        }
        for t in &mut self.throttled {
            *t = false;
        }

        // Re-dispatch to idle ranks that may have been waiting at the overshoot gate.
        if self.progressive {
            for rank in 0..self.world_size {
                let has_inflight = self.chunk_pools.values()
                    .any(|p| p.in_flight(rank) > 0);
                if !has_inflight {
                    self.dispatch_next_chunk(rank);
                }
            }
        }
    }

    /// Abort an in-progress CPU averaging cycle and return to Idle.
    ///
    /// Drains stale snapshots from `param_rx` so the next cycle starts clean.
    fn abort_cpu_averaging(&mut self) {
        self.avg_state = CpuAvgState::Idle;
        // Drain any in-flight snapshots from the aborted round.
        while self.param_rx.try_recv().is_ok() {}
    }

    /// Cleanly shut down any in-progress CPU averaging before the coordinator exits.
    ///
    /// If in Computing state, joins the background thread (waits for it to finish)
    /// so no detached thread holds GPU resources when the coordinator returns.
    /// If in Collecting state, drains stale snapshots.
    pub fn drain_avg_state(&mut self) {
        let state = std::mem::replace(&mut self.avg_state, CpuAvgState::Idle);
        match state {
            CpuAvgState::Idle => {}
            CpuAvgState::Collecting { snapshots, .. } => {
                crate::verbose!(
                    "  ddp: discarding in-progress CPU averaging \
                     (Collecting, {}/{} snapshots received)",
                    snapshots.len(), self.world_size
                );
                while self.param_rx.try_recv().is_ok() {}
            }
            CpuAvgState::Computing { handle, .. } => {
                // Join the compute thread so no detached thread holds GPU resources.
                let result = handle.join();
                let status = match &result {
                    Ok(Ok(_)) => "completed result discarded",
                    Ok(Err(e)) => {
                        eprintln!("  ddp: CPU averaging compute error: {e}");
                        "errored"
                    }
                    Err(_) => "panicked",
                };
                crate::verbose!(
                    "  ddp: discarding in-progress CPU averaging (Computing, {status})"
                );
                // Drain any snapshots it might have sent before we discard the result.
                while self.param_rx.try_recv().is_ok() {}
            }
        }
    }

    /// Drive the CPU averaging state machine one tick.
    ///
    /// Called every coordinator loop iteration. Handles three states:
    ///
    /// - **Idle**: no-op.
    /// - **Collecting**: `try_recv` on `param_rx` for pending snapshots.
    ///   Transitions to Computing when all ranks have responded, or soft-aborts
    ///   on timeout (drains stale snapshots, returns to Idle, logs warning).
    /// - **Computing**: checks if the background thread has finished. When done,
    ///   sends `Update` to all workers and runs `finish_averaging_cpu`.
    ///
    /// Returns `Ok(())` on normal progress (including soft abort).
    /// Returns `Err` only on unrecoverable errors (compute thread panic).
    pub fn poll_cpu_averaging(&mut self) -> Result<()> {
        // Take ownership of the state to avoid borrow issues.
        let state = std::mem::replace(&mut self.avg_state, CpuAvgState::Idle);

        match state {
            CpuAvgState::Idle => {
                self.avg_state = CpuAvgState::Idle;
            }
            CpuAvgState::Collecting {
                mut snapshots,
                mut received,
                deadline,
                start,
                steps_snapshot,
                wall_ms_snapshot,
            } => {
                // Drain all available snapshots (non-blocking).
                while let Ok(snap) = self.param_rx.try_recv() {
                    if snap.rank < self.world_size && !received[snap.rank] {
                        received[snap.rank] = true;
                        snapshots.push(snap);
                    }
                    // Ignore duplicates or out-of-range ranks.
                }

                if snapshots.len() >= self.world_size {
                    // All snapshots collected. Spawn compute thread.
                    if let Some(ref tl) = self.timeline {
                        tl.event(crate::monitor::EventKind::CpuAvgStart);
                    }
                    let version = self.version + 1;

                    let handle = std::thread::Builder::new()
                        .name("cpu-avg-compute".into())
                        .spawn(move || {
                            Self::compute_average_and_divergence(
                                snapshots, version,
                            )
                        })
                        .map_err(|e| TensorError::new(
                            &format!("failed to spawn CPU averaging thread: {e}")
                        ))?;

                    self.avg_state = CpuAvgState::Computing {
                        handle,
                        start,
                        steps_snapshot,
                        wall_ms_snapshot,
                    };
                } else if Instant::now() >= deadline {
                    // Timeout: soft abort.
                    let missing: Vec<usize> = received.iter().enumerate()
                        .filter(|(_, got)| !**got)
                        .map(|(r, _)| r)
                        .collect();
                    self.abort_count += 1;
                    crate::verbose!(
                        "  ddp: CPU averaging timeout, missing ranks: {missing:?} \
                         (abort #{}, will retry)", self.abort_count
                    );
                    self.abort_cpu_averaging();
                } else {
                    // Still waiting. Put state back.
                    self.avg_state = CpuAvgState::Collecting {
                        snapshots,
                        received,
                        deadline,
                        start,
                        steps_snapshot,
                        wall_ms_snapshot,
                    };
                }
            }
            CpuAvgState::Computing {
                handle,
                start,
                steps_snapshot,
                wall_ms_snapshot,
            } => {
                if handle.is_finished() {
                    let result = handle.join()
                        .map_err(|_| TensorError::new(
                            "CPU averaging compute thread panicked"
                        ))??;

                    let avg_ms = start.elapsed().as_secs_f64() * 1000.0;

                    // Send averaged params to all workers.
                    for (rank, tx) in self.control_txs.iter().enumerate() {
                        if tx.send(ControlMsg::Update(result.averaged.clone())).is_err() {
                            eprintln!(
                                "  ddp: failed to deliver Update to rank {rank} \
                                 (worker channel dead)"
                            );
                        }
                    }

                    self.finish_averaging_cpu(
                        avg_ms,
                        &steps_snapshot,
                        &wall_ms_snapshot,
                        result.divergence,
                    );
                    // avg_state is already Idle from the mem::replace.
                } else {
                    // Still computing. Put state back.
                    self.avg_state = CpuAvgState::Computing {
                        handle,
                        start,
                        steps_snapshot,
                        wall_ms_snapshot,
                    };
                }
            }
        }

        Ok(())
    }

    /// Compute weighted average of parameter snapshots.
    ///
    /// Weight each rank's contribution by its `batch_count` (number of batches
    /// since last averaging). This ensures faster GPUs contribute more.
    /// Buffers (e.g. BatchNorm running stats) are averaged with equal weight.
    pub(in super::super) fn average_params(snapshots: &[ParamSnapshot], version: u64) -> Result<AveragedParams> {
        if snapshots.is_empty() {
            return Err(TensorError::new("average_params: no snapshots"));
        }

        let n_params = snapshots[0].params.len();
        let n_buffers = snapshots[0].buffers.len();

        // Validate all snapshots have the same structure.
        for (i, snap) in snapshots.iter().enumerate() {
            if snap.params.len() != n_params {
                return Err(TensorError::new(&format!(
                    "average_params: rank {} has {} params, expected {}",
                    i, snap.params.len(), n_params
                )));
            }
            if snap.buffers.len() != n_buffers {
                return Err(TensorError::new(&format!(
                    "average_params: rank {} has {} buffers, expected {}",
                    i, snap.buffers.len(), n_buffers
                )));
            }
        }

        let total_batches: usize = snapshots.iter().map(|s| s.batch_count.max(1)).sum();

        // Weighted average of parameters on CPU (snapshots may be on different devices)
        let mut avg_params = Vec::with_capacity(n_params);
        for pi in 0..n_params {
            let first = snapshots[0].params[pi].to_device(Device::CPU)?;
            let acc = Tensor::zeros_like(&first)?;
            for snap in snapshots {
                let weight = snap.batch_count.max(1) as f64 / total_batches as f64;
                let cpu_param = snap.params[pi].to_device(Device::CPU)?;
                let scaled = cpu_param.mul_scalar(weight)?;
                acc.add_(&scaled)?;
            }
            avg_params.push(acc);
        }

        // Average buffers (equal weight, e.g. BatchNorm running mean/var).
        let inv_n = 1.0 / snapshots.len() as f64;
        let mut avg_buffers = Vec::with_capacity(n_buffers);
        for bi in 0..n_buffers {
            let first = snapshots[0].buffers[bi].to_device(Device::CPU)?;
            let acc = Tensor::zeros_like(&first)?;
            for snap in snapshots {
                let cpu_buf = snap.buffers[bi].to_device(Device::CPU)?;
                let scaled = cpu_buf.mul_scalar(inv_n)?;
                acc.add_(&scaled)?;
            }
            avg_buffers.push(acc);
        }

        Ok(AveragedParams {
            params: avg_params,
            buffers: avg_buffers,
            version,
        })
    }

    /// Compute weighted average and divergence on a background thread.
    ///
    /// Static method: captures only the data it needs, no `&self` borrow.
    /// Returns `CpuAvgResult` with the averaged params and the relative
    /// divergence across replicas (for anchor correction by ElChe).
    fn compute_average_and_divergence(
        snapshots: Vec<ParamSnapshot>,
        version: u64,
    ) -> Result<CpuAvgResult> {
        // Normalize all snapshot tensors to CPU once up front. In production,
        // worker.snapshot_params() already copies to CPU so these are no-ops.
        // This ensures correctness when snapshot tensors are on GPU (tests,
        // or future callers that skip the worker D2H copy).
        let snapshots: Vec<ParamSnapshot> = snapshots.into_iter()
            .map(|snap| Ok(ParamSnapshot {
                rank: snap.rank,
                params: snap.params.into_iter()
                    .map(|t| t.to_device(Device::CPU))
                    .collect::<Result<_>>()?,
                buffers: snap.buffers.into_iter()
                    .map(|t| t.to_device(Device::CPU))
                    .collect::<Result<_>>()?,
                batch_count: snap.batch_count,
            }))
            .collect::<Result<_>>()?;

        // average_params also calls to_device(CPU) internally, but since
        // we already normalized above, those are no-ops -- no redundant copies.
        let averaged = Self::average_params(&snapshots, version)?;

        // Compute ||average|| for normalization.
        let mut avg_norm_sq = 0.0f64;
        for p in &averaged.params {
            let n: f64 = p.norm()?.item()?;
            avg_norm_sq += n * n;
        }
        let avg_norm = avg_norm_sq.sqrt();

        let divergence = if avg_norm < 1e-10 {
            None
        } else {
            // Compute per-rank: ||snapshot - average|| / ||average||
            // and per-rank pre-sync norm: ||snapshot||.
            let mut deltas = Vec::with_capacity(snapshots.len());
            let mut pre_norms = Vec::with_capacity(snapshots.len());
            for snap in &snapshots {
                let mut diff_sq = 0.0f64;
                let mut pre_sq = 0.0f64;
                for (sp, ap) in snap.params.iter().zip(&averaged.params) {
                    let diff = sp.sub(ap)?;
                    let dn: f64 = diff.norm()?.item()?;
                    diff_sq += dn * dn;
                    let pn: f64 = sp.norm()?.item()?;
                    pre_sq += pn * pn;
                }
                deltas.push(diff_sq.sqrt() / avg_norm);
                pre_norms.push(pre_sq.sqrt());
            }
            Some(convergence::DivergenceReport {
                deltas,
                pre_norms: Some(pre_norms),
                post_norm: Some(avg_norm),
            })
        };

        Ok(CpuAvgResult { averaged, divergence })
    }
}
