//! Control-message dispatch: `handle_control` (non-blocking drain), `drain_until_shutdown`, and the central `dispatch_control` state machine.

use crate::nn::Module;
use crate::tensor::Result;

use super::super::{ControlMsg, TimingMsg, make_partition, pick_space};
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
    pub(crate) fn dispatch_control(&mut self, msg: ControlMsg) -> Result<bool> {
        // Instrumentation: count processed control messages to test
        // whether cpu mode carries more per-cycle control traffic.
        if self.prof_enabled {
            self.ctrl_msgs_handled += 1;
        }
        match msg {
            ControlMsg::RequestParams => {
                // Record what this frame ships (see `steps_at_snapshot` on
                // the struct): the matching `Update` subtracts exactly this,
                // so overshoot steps taken during the averaging round-trip
                // keep their mass credit for the next frame.
                self.steps_at_snapshot = self.steps_since_avg;
                // The retained consensus may alias the pinned snapshot
                // staging (decode-into); the D2H below clobbers those
                // bytes, so invalidate BEFORE snapshotting. The round's
                // `Update` re-retains.
                self.last_consensus = None;
                // Instrumentation (gated): time the GPU→CPU readout — the
                // per-window snapshot the CPU averaging path pays to
                // publish weights for the reduce.
                if self.prof_enabled {
                    let t = std::time::Instant::now();
                    let snap = self.snapshot_params();
                    self.snapshot_ns_total += t.elapsed().as_nanos();
                    self.snapshot_count += 1;
                    let _ = self.param_tx.send(snap);
                } else {
                    let snap = self.snapshot_params();
                    let _ = self.param_tx.send(snap);
                }
            }
            ControlMsg::Update(avg) => {
                self.load_averaged(&avg)?;
                // Subtract the shipped count, don't zero: steps taken since
                // the snapshot (cpu-async overshoot) survive the EASGD blend
                // in `load_averaged` and must ride the next frame's mass.
                // Marker reset so a spurious second Update subtracts 0.
                self.steps_since_avg = self.steps_since_avg.saturating_sub(self.steps_at_snapshot);
                self.steps_at_snapshot = 0;
                // Consensus consumers gate on realized rounds only: an
                // all-idle round's payload is the rank's own state, not a
                // consensus. An armed eval survives unrealized rounds and
                // fires at the next realized one.
                if avg.realized {
                    if let Some((schedule_id, epoch)) = self.pending_consensus_eval.take() {
                        self.consensus_eval_at_reduce(&avg, schedule_id, epoch);
                    }
                    // Retain for the final canonical eval (see
                    // `last_consensus` on the struct for the validity
                    // window).
                    self.last_consensus = Some(avg);
                }
            }
            ControlMsg::StageAdvisory { counts, segments } => {
                // Purely advisory: forward to the background stager
                // (latest wins there). Never blocks, never fails the
                // control loop.
                if let Some(stager) = &self.stager {
                    stager.advise(super::stager::StageAdvisory { counts, segments });
                }
            }
            ControlMsg::SyncNow => {
                crate::debug!(
                    "  ddp-worker: rank {} SyncNow (step={}, epoch={})",
                    self.rank,
                    self.local_step,
                    self.current_epoch
                );
                let (divergence, post_norm, pre_norm) = self.sync_now_nccl()?;
                crate::debug!("  ddp-worker: rank {} SyncNow done", self.rank);
                // NCCL sync is synchronous — nothing can step between the
                // collective and this line; zero is exact here.
                self.steps_since_avg = 0;
                self.steps_at_snapshot = 0;
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
                // One-chunk-in-flight, worker side: two dispatch sources
                // converge here (coord `StartEpoch` frames and the inbound
                // bridge's `StartEpoch` synthesized from `Update.next_plan`).
                // A duplicate dispatch overwriting an unconsumed plan drops a
                // chunk on the floor — the coordinator's `in_flight` for it
                // never completes and the reduce gate wedges. Make the
                // violation loud instead of a silent overnight hang; keep the
                // NEWER plan (the coordinator's in_flight tracks the newest
                // dispatch, so the older one is the stranded chunk either way).
                if let Some(old) = &self.pending_plan {
                    eprintln!(
                        "flodl ddp: rank {} StartEpoch overwrites unconsumed plan \
                         (old epoch {} offset {} size {}; new epoch {} offset {} size {}) \
                         — one-chunk-in-flight violated, a chunk was dropped",
                        self.rank,
                        old.epoch,
                        old.partition_offset,
                        old.partition_size,
                        plan.epoch,
                        plan.partition_offset,
                        plan.partition_size,
                    );
                    debug_assert!(false, "StartEpoch overwrote an unconsumed pending_plan");
                }
                self.pending_plan = Some(plan);
            }
            ControlMsg::DeclareDead | ControlMsg::NewNcclSession | ControlMsg::RequestNewNcclId => {
                // Cluster-mode elastic-membership signals. The
                // cluster_worker layer intercepts these in its
                // inbound bridge (updates a local DeadRanks ledger,
                // stages a pending NCCL session, or generates a
                // fresh UID and replies through the timing
                // channel); they should not reach the inner
                // GpuWorker. If one slips through (e.g. via test
                // wiring), drop silently — the inner GpuWorker has no
                // comm-replacement surface to act on it.
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
                    pick_space(self.dataset.len(), self.augment),
                    self.current_epoch,
                    self.epoch_splits,
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
                                ControlMsg::SyncNow | ControlMsg::Update(_) | ControlMsg::Shutdown
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
            ControlMsg::Checkpoint {
                version,
                target_rank,
            } => {
                // Targeted: only the rank named by the coord runs.
                // Other ranks silently ignore the frame (in cluster
                // mode the broadcast is already targeted by the
                // coord; in threaded DDP the coord sends only to
                // rank 0's channel — both paths converge on
                // `target_rank == self.rank` being the only run gate).
                // Worker never decides retry / abort; it reports
                // and lets the coord decide.
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
            ControlMsg::ExecuteEvalCallback {
                schedule_id,
                epoch,
                target_rank,
                adopt_consensus,
            } => {
                // Targeted: only the rank named by the coord runs.
                // Mirrors the `Checkpoint` arm; worker never decides
                // whether it is the evaluator. Every rank has
                // `eval_fn` available in cluster mode so coord-driven
                // role rotation works without loud errors.
                if target_rank != self.rank {
                    return Ok(false);
                }
                if self.eval_fn.is_none() {
                    return Ok(false);
                }
                // `adopt_consensus` (final canonical eval, CPU): an EASGD
                // rank's live model is a blend — overwrite it with the
                // retained last realized consensus before scoring. Without
                // `easgd_alpha` the post-writeback model already IS the
                // consensus, so the flag is a no-op. No restore: this only
                // fires at the run's settled end.
                //
                // `elapsed_ms` covers the whole sequence (adopt copies +
                // eval + mode flips) so the coord's time exclusion absorbs
                // exactly what stalled the rank — symmetric with the
                // checkpoint path.
                let start = std::time::Instant::now();
                let (metric, error) = if adopt_consensus && self.easgd_alpha.is_some() {
                    let retained = self
                        .last_consensus
                        .as_ref()
                        .map(|a| (a.params.clone(), a.buffers.clone()));
                    match retained {
                        Some((params, buffers)) => {
                            match self.adopt_consensus_for_final_eval(&params, &buffers) {
                                Ok(()) => self.eval_metric(),
                                Err(e) => (
                                    f64::NAN,
                                    Some(format!("final consensus adopt failed before eval: {e}")),
                                ),
                            }
                        }
                        None => (
                            f64::NAN,
                            Some(
                                "final consensus eval requested but no realized \
                                 consensus is retained on this rank"
                                    .to_string(),
                            ),
                        ),
                    }
                } else {
                    self.eval_metric()
                };
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let _ = self.timing_tx.send(TimingMsg::EvalResult {
                    rank: self.rank,
                    schedule_id,
                    epoch,
                    metric,
                    elapsed_ms,
                    error,
                });
            }
            ControlMsg::ArmConsensusEval {
                schedule_id,
                epoch,
                target_rank,
            } => {
                // Targeted like `ExecuteEvalCallback`. Just arms — the
                // next realized `Update` fires the consensus eval (see
                // the `Update` arm above). A re-arm before the previous
                // fired supersedes it (latest wins; the superseded eval
                // is a missed cadence, not an error).
                if target_rank == self.rank {
                    self.pending_consensus_eval = Some((schedule_id, epoch));
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
                // Cooperative tier: forward the full per-epoch series to the
                // user's drain (armed only there; None everywhere else, so no
                // accumulation in managed / setup mode). Clone before the slot
                // write consumes `metrics`. A dropped receiver is benign (the
                // user stopped polling); this never blocks the control drain.
                if let Some(tx) = &self.metrics_stream_tx {
                    let _ = tx.send((*metrics).clone());
                }
                if let Ok(mut slot) = self.aggregated_metrics.lock() {
                    *slot = Some(*metrics);
                }
            }
            ControlMsg::EvalBroadcast { epoch, metric } => {
                // Cooperative tier: forward the controller-elected eval to the
                // user's drain (armed only there; None elsewhere). Sibling of
                // the EpochAggregated forward above; a dropped receiver is
                // benign and never blocks the control drain.
                if let Some(tx) = &self.eval_stream_tx {
                    let _ = tx.send((epoch, metric));
                }
            }
            ControlMsg::SaveConsensusModel { target_rank } => {
                // NCCL consensus checkpoint: the elected rank writes its
                // CURRENT model — which holds the just-completed in-place
                // weighted-AllReduce consensus, params work-weighted and f32
                // buffers mover-averaged (no EASGD blend on the NCCL path) —
                // to `<save_path>.fdl`. Targeted: only the named rank runs.
                // No `.optim`, no shutdown; best-effort (mirrors the CPU
                // forge's detached write), so no result frame. The coord's
                // `.meta.json` is the resume index.
                if target_rank != self.rank {
                    return Ok(false);
                }
                match self.save_path.clone() {
                    Some(stem) => {
                        self.write_model_to_fdl(&stem);
                        // Outer-optimizer momentum rides the same elected-rank
                        // write: this rank's replicated momentum -> `<stem>.outer.fdl`.
                        // No-op for a stateless outer optimizer / no outer
                        // optimizer (OuterAvg writes no artifact).
                        self.write_outer_momentum_to_fdl(&stem);
                    }
                    None => eprintln!(
                        "ddp-worker: rank {} SaveConsensusModel received but \
                         save_path is unset; consensus .fdl not written",
                        self.rank,
                    ),
                }
            }
        }
        Ok(false)
    }

    /// Run the user's `eval_fn` against `eval_dataset` on the current
    /// model and return `(metric, error)` — the `TimingMsg::EvalResult`
    /// payload halves. Flips the model into eval mode for
    /// BN/Dropout/etc. correctness and restores train mode after; eval
    /// forwards run on ONE elected rank, so graph profiling is paused
    /// around the closure (letting them feed the accumulator would tilt
    /// exactly this rank's per-node means). Callers gate on
    /// `eval_fn.is_some()` and own the elapsed measurement + result
    /// send, so consensus-adopt copies land inside the same
    /// time-excluded window.
    pub(super) fn eval_metric(&mut self) -> (f64, Option<String>) {
        let Some(ref f) = self.eval_fn else {
            return (
                f64::NAN,
                Some("eval dispatched to a rank with no eval_fn (caller gate bug)".to_string()),
            );
        };
        match self.eval_dataset.as_ref() {
            Some(ds) => {
                let profiling_paused = self.pause_graph_profiling();
                self.model.eval();
                let r = f(&self.model, ds.as_ref());
                self.model.train();
                if profiling_paused {
                    self.resume_graph_profiling();
                }
                match r {
                    Ok(m) => (m, None),
                    Err(e) => (f64::NAN, Some(e.to_string())),
                }
            }
            None => (
                f64::NAN,
                Some(
                    "ddp: eval_fn set without eval_dataset; attach a held-out \
                     dataset via DdpBuilder::eval_dataset(...)"
                        .to_string(),
                ),
            ),
        }
    }

    /// Fire an armed consensus eval at a realized reduce: score THIS
    /// round's consensus, then leave the model exactly where the normal
    /// writeback would have. Runs inside the `Update` arm, so no
    /// training step can interleave.
    ///
    /// - `easgd_alpha == None`: the writeback already overwrote the
    ///   model with the consensus — eval in place.
    /// - `easgd_alpha == Some(_)`: the model holds the post-reduce
    ///   blend; swap-score-restore (`scored_consensus_swap`) stashes the
    ///   blend, adopts `avg.params`, evals, and restores the blend
    ///   VERBATIM — bit-identical to a round with no eval.
    ///
    /// Best-effort like every callback dispatch: errors ride the
    /// `EvalResult.error` field instead of failing the control loop.
    fn consensus_eval_at_reduce(
        &mut self,
        avg: &super::super::AveragedParams,
        schedule_id: u64,
        epoch: u64,
    ) {
        if self.eval_fn.is_none() {
            // Parity with the wire dispatch: a rank without eval_fn
            // silently no-ops (role rotation without loud errors).
            return;
        }
        let start = std::time::Instant::now();
        let (metric, error) = match self.easgd_alpha {
            None => {
                // The writeback's H2D is async on the comm stream; eval
                // forwards are not a train step, so `sync_before_forward`
                // never covers them — fence here.
                self.fence_comm_stream();
                self.eval_metric()
            }
            Some(_) => match self.scored_consensus_swap(avg) {
                Ok(me) => me,
                Err(e) => (f64::NAN, Some(format!("consensus eval swap failed: {e}"))),
            },
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
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
