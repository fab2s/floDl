//! Control-message dispatch: `handle_control` (non-blocking drain), `drain_until_shutdown`, and the central `dispatch_control` state machine.


use crate::nn::Module;
use crate::tensor::{Result, TensorError};

use super::super::{
    ControlMsg, TimingMsg, make_partition,
};
use super::GpuWorker;

impl<M: Module> GpuWorker<M> {
    /// Process pending control messages (non-blocking).
    ///
    /// Returns `true` if a Shutdown was received.
    pub fn handle_control(&mut self) -> Result<bool> {
        while let Ok(msg) = self.control_rx.try_recv() {
            if self.dispatch_control(msg)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Block on control messages until Shutdown or channel disconnect.
    ///
    /// Called after training is done and `report_exiting()` has been sent.
    /// Skips NCCL collectives (SyncNow): since this worker has reported
    /// Exiting, the coordinator may not send SyncNow to our peers, but
    /// if it was already in-flight, calling AllReduce here would deadlock
    /// if the peer has also exited or errored.
    pub fn drain_until_shutdown(&mut self) {
        while let Ok(msg) = self.control_rx.recv() {
            match msg {
                ControlMsg::SyncNow => {
                    // Skip: peer may be dead, AllReduce would deadlock.
                    // The coordinator will stop triggering collectives
                    // once it processes our Exiting message.
                }
                ControlMsg::Shutdown => break,
                other => {
                    if self.dispatch_control(other).unwrap_or(true) {
                        break;
                    }
                }
            }
        }
    }

    /// Handle a single control message. Returns `true` on Shutdown.
    pub(super) fn dispatch_control(&mut self, msg: ControlMsg) -> Result<bool> {
        match msg {
            ControlMsg::RequestParams => {
                let _ = self.param_tx.send(self.snapshot_params());
            }
            ControlMsg::Update(avg) => {
                self.load_averaged(&avg)?;
                self.steps_since_avg = 0;
            }
            ControlMsg::SyncNow => {
                crate::debug!("  ddp-worker: rank {} SyncNow (step={}, epoch={})", self.rank, self.local_step, self.current_epoch);
                let (divergence, post_norm, pre_norm) = self.sync_now_nccl()?;
                crate::debug!("  ddp-worker: rank {} SyncNow done", self.rank);
                self.steps_since_avg = 0;
                // Bump local_step and send a dedicated SyncAck so the
                // coordinator's nccl_ack mechanism sees step_count > snapshot.
                // Without this, a SyncNow processed in wait_for_epoch_plan
                // (no batches to train afterward) leaves nccl_ack permanently
                // false, blocking all future should_average() calls.
                //
                // SyncAck is used instead of TimingMsg::Batch so the
                // coordinator doesn't count this as a real batch -- that would
                // inflate steps_since_avg (and thus global_step) by one per
                // sync per rank, firing the LR scheduler early.
                self.local_step += 1;
                let _ = self.timing_tx.send(TimingMsg::SyncAck {
                    rank: self.rank,
                    step_count: self.local_step,
                    divergence,
                    post_norm,
                    pre_norm,
                });
            }
            ControlMsg::StartEpoch(plan) => {
                self.pending_plan = Some(plan);
            }
            ControlMsg::DeclareDead { .. }
            | ControlMsg::NewNcclSession { .. }
            | ControlMsg::RequestNewNcclId => {
                // Cluster-mode elastic-membership signals. The
                // cluster_worker layer intercepts these in its
                // inbound bridge (updates a local DeadRanks ledger,
                // stages a pending NCCL session, or generates a
                // fresh UID and replies through the timing
                // channel); they should not reach the inner
                // GpuWorker. If one slips through (e.g. via test
                // wiring), drop silently — the OLD threaded path
                // has no comm-replacement surface to act on it.
            }
            ControlMsg::ShutdownWithSave { reason } => {
                // Cluster-mode unrecoverable-failure persistence.
                // Write the bundle to `save_path` (rank 0 is the
                // canonical writer for the model + meta; all ranks
                // attempt the optimizer save since per-rank momentum
                // buffers differ — rank 0's `.optim` is the canonical
                // one to load from, but persisting per-rank files
                // makes a future "average optimizer state on resume"
                // path tractable without re-instrumenting).
                //
                // Errors during save log loud and don't block exit:
                // we'd rather surface a disk-full / permission error
                // than deadlock the cluster on shutdown.
                if let Some(stem) = self.save_path.clone() {
                    self.write_checkpoint_bundle(&stem, reason);
                } else {
                    crate::verbose!(
                        "  ddp-worker: rank {} ShutdownWithSave received \
                         but save_path is unset; exiting without saving",
                        self.rank,
                    );
                }
                return Ok(true);
            }
            ControlMsg::ExtendPartition {
                partition_offset,
                partition_size,
            } => {
                // Resolve the new slice through `make_partition` keyed
                // on the SAME (current_epoch, base_seed) the worker
                // used for its StartEpoch, so the appended indices
                // align with the rest of the cluster's view of the
                // permutation. Append in-place; both the sync and the
                // prefetch loop in `run_epoch_plan` re-check
                // `partition.len()` each iteration so the extension is
                // processed before declaring the epoch complete. The
                // prefetch path also gets the newly-completable
                // batches submitted to its load queue here so the
                // background worker has work to feed the consumer.
                let extra = make_partition(
                    partition_offset,
                    partition_size,
                    self.dataset.len(),
                    self.current_epoch,
                    self.base_seed,
                );
                let old_batches = self.partition.len() / self.batch_size;
                self.partition.extend(extra);
                let new_batches = self.partition.len() / self.batch_size;
                if let Some(ref pw) = self.prefetch {
                    for batch_idx in old_batches..new_batches {
                        let start = batch_idx * self.batch_size;
                        let end = start + self.batch_size;
                        pw.load_batch(self.partition[start..end].to_vec());
                    }
                }
            }
            ControlMsg::Throttle => {
                // Worker is ahead of the slowest rank: block until averaging
                // completes (SyncNow/Update) or Shutdown. Intermediate messages
                // (RequestParams, StartEpoch) are handled but don't release
                // the throttle. Duplicate Throttle messages are ignored.
                loop {
                    match self.control_rx.recv() {
                        Ok(ControlMsg::Throttle) => continue, // already throttled
                        Ok(msg) => {
                            let releases = matches!(
                                &msg,
                                ControlMsg::SyncNow
                                    | ControlMsg::Update(_)
                                    | ControlMsg::Shutdown
                            );
                            let shutdown = self.dispatch_control(msg)?;
                            if shutdown || releases {
                                return Ok(shutdown);
                            }
                        }
                        Err(_) => return Ok(true), // channel dead
                    }
                }
            }
            ControlMsg::SetGlobalStep(step) => {
                self.global_step = step;
            }
            ControlMsg::Checkpoint { version, target_rank } => {
                // Targeted: only the rank named by the coord runs.
                // Other ranks silently ignore the frame (in cluster
                // mode the broadcast is already targeted by the
                // coord; in threaded DDP the coord sends only to
                // rank 0's channel — both paths converge on
                // `target_rank == self.rank` being the only run gate).
                // Worker never decides retry / abort — it reports
                // and lets the coord decide (see `#29` design).
                if target_rank != self.rank {
                    return Ok(false);
                }
                let start = std::time::Instant::now();
                let err = match self.checkpoint_fn.as_ref() {
                    Some(f) => f(version, &self.model).err().map(|e| e.to_string()),
                    None => Some(format!(
                        "checkpoint dispatched to rank {} but checkpoint_fn \
                         is None (config bug or stale role assignment)",
                        self.rank
                    )),
                };
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let _ = self.timing_tx.send(TimingMsg::CheckpointResult {
                    rank: self.rank,
                    version,
                    elapsed_ms,
                    error: err,
                });
            }
            ControlMsg::ExecuteEvalCallback { schedule_id, epoch, target_rank } => {
                // Targeted: only the rank named by the coord runs.
                // Mirrors the `Checkpoint` arm — worker never decides
                // whether it is the evaluator. With #28b's all-Some
                // cluster-mode policy, every rank has `eval_fn`
                // available so coord-driven role rotation works
                // without loud errors.
                if target_rank != self.rank {
                    return Ok(false);
                }
                // Flip the model into eval mode for BN/Dropout/etc.
                // correctness, run the user closure against the
                // held-out dataset, then restore train mode. The
                // scalar metric (or error) flows back to the
                // controller via `TimingMsg::EvalResult`; the
                // controller's `eval_result_fn` fires on receipt.
                //
                // `elapsed_ms` is measured around the closure (eval
                // + train-mode flip) so the coord can time-exclude
                // it from `wall_ms_accum[rank]` — symmetric with the
                // checkpoint path.
                if let Some(ref f) = self.eval_fn {
                    let start = std::time::Instant::now();
                    let result = match self.eval_dataset.as_ref() {
                        Some(ds) => {
                            self.model.eval();
                            let r = f(&self.model, ds.as_ref());
                            self.model.train();
                            r
                        }
                        None => Err(TensorError::new(
                            "ddp: eval_fn set without eval_dataset; \
                             attach a held-out dataset via \
                             DdpBuilder::eval_dataset(...)",
                        )),
                    };
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let (metric, error) = match result {
                        Ok(m) => (m, None),
                        Err(e) => (f64::NAN, Some(e.to_string())),
                    };
                    let _ = self.timing_tx.send(TimingMsg::EvalResult {
                        rank: self.rank,
                        schedule_id,
                        epoch,
                        metric,
                        elapsed_ms,
                        error,
                    });
                }
            }
            ControlMsg::SetEpochCallbackRole { rank } => {
                // Controller resolved (or re-resolved) the rank that
                // should fire `epoch_fn` at each epoch transition.
                // Worker just stores it — the autonomous fire-check
                // in the cluster worker's main loop reads this.
                self.epoch_callback_role = Some(rank);
            }
            ControlMsg::Shutdown => return Ok(true),
            ControlMsg::EpochAggregated(metrics) => {
                // Coord-pushed cross-rank aggregated view. Stash the
                // latest snapshot under `aggregated_metrics` so the
                // user's `Graph` (sharing the same Arc<Mutex<...>>
                // via setup-mode wiring) surfaces the global view
                // under `latest_metrics()` and `graph_gpu_metrics()`.
                // Cluster-builder runs that drive the training loop
                // inside the framework's closure can also reach this
                // via `GpuWorker::aggregated_metrics()`.
                if let Ok(mut slot) = self.aggregated_metrics.lock() {
                    *slot = Some(metrics);
                }
            }
        }
        Ok(false)
    }
}
