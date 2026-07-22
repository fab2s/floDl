//! Cooperative execution tier: the [`Worker`] handle.
//!
//! The "missing middle" between the managed tier (`Trainer::builder().run()`,
//! framework owns the loop) and the bypass primitive (`Ddp::wrap`, user owns
//! everything). The **user owns the training loop body** while the
//! `ClusterCoordinator` stays authoritative over cadence, partition, averaging,
//! role-election, elastic membership, and checkpoint orchestration — so the
//! trained model matches the managed tier.
//!
//! ```ignore
//! let mut w = Trainer::builder(mf, of, tf).into_worker()?;
//! while let Some(_plan) = w.next_plan()? {
//!     while let Some(batch) = w.next_batch()? {   // owned batch, borrows nothing
//!         let loss = train_step(w.model(), &batch)?; // user forward + loss
//!         loss.backward()?;
//!         if w.step(&loss)?.shutdown { break; }      // framework-owned tail + control
//!     }
//! }
//! let state = w.finish()?;
//! ```
//!
//! `next_plan` yields the next unit of work the controller dispatches — under
//! progressive cadence (the default for Cadence/Async) that is one *chunk* of
//! an epoch, not the whole epoch, so the returned [`EpochPlan`]'s `.epoch` can
//! repeat across calls (see [`Worker::next_plan`]). The inner loop drains that
//! unit's batches.
//!
//! The reduce is never decided here: it happens as a side effect of the
//! control drain inside `step` (`after_step` -> `handle_control`) when the
//! coordinator's `SyncNow` frame lands, at the ElChe cadence. `step` never
//! names a cadence and never calls the collective directly, which is what keeps
//! the single-step-clock and determinism invariants intact.

use crate::autograd::Variable;
use crate::distributed::cluster_worker::ClusterWorker;
use crate::nn::Module;
use crate::tensor::cuda_stream::StreamGuard;
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::worker::{EpochState, GpuWorker};
use super::{EpochMetrics, EpochPlan, TrainedState};

/// Outcome of a cooperative training [`Worker::step`].
#[derive(Debug, Clone, Copy)]
pub struct StepOutcome {
    /// The controller asked the cohort to shut down (a `Shutdown` frame was
    /// drained during this step's control processing). The user's loop should
    /// stop; the next `next_plan` returns `None`.
    pub shutdown: bool,
}

/// The cooperative-tier training handle. The user drives the epoch/batch loop;
/// the framework owns the per-step bookkeeping, the coordinator-driven reduce,
/// timing reports, control draining, elastic membership, and role election.
///
/// Presents the cohort as one logical trainer ("the collective as a whole"):
/// there is no per-rank / per-device surface, because in data-parallel DDP the
/// ranks are replicas and single-rank side tasks (eval / checkpoint / logging)
/// are handled by the controller's role election, not by user code.
pub struct Worker<M: Module> {
    inner: WorkerInner<M>,
    /// Current epoch cursor; `None` between epochs (before the first
    /// `next_batch` of an epoch and after the shard is drained).
    epoch_state: Option<EpochState>,
    /// Plan handed out by the last `next_plan`, consumed lazily by the first
    /// `next_batch` of the epoch (which runs `begin_epoch`).
    pending_plan: Option<EpochPlan>,
    /// Stamped at the end of `next_batch` (after `sync_before_forward`) and read
    /// at `step` completion, so the compute window matches the managed
    /// `train_step`'s timing (forward + backward + optimizer tail).
    compute_start: Option<std::time::Instant>,
    /// Held across the user's forward + backward (from `next_batch` delivery
    /// through `step`) so the gradient kernels run on `compute_stream`, matching
    /// `train_step`'s internal guard. Without it the AccumulateGrad nodes (also
    /// pinned to `compute_stream`) see a stream mismatch and libtorch warns.
    /// Owned (does not borrow the worker), so it can live in the handle.
    active_guard: Option<StreamGuard>,
    /// The controller signalled shutdown (seen in `next_batch`'s wait or in
    /// `step`'s control drain); `next_plan` returns `None` from here on.
    shutdown: bool,
    /// Set by `finish` so the Drop guard knows the run ended cleanly. Without
    /// it, dropping a cluster rank's `Worker` mid-run (user loop errored /
    /// panicked) writes a death record and exits to unblock peers.
    finished: bool,
    /// Forensic context for the Drop guard: `Some` on a cluster rank, `None` on
    /// a single device (no peers to unblock, nothing to record).
    forensics: Option<RankForensics>,
    /// Full per-epoch metrics stream, armed only on the cluster path. Fed by
    /// the worker's `dispatch_control` (during `step` / `next_plan`) with every
    /// coordinator-broadcast aggregated epoch — the same series the managed
    /// [`DdpHandle::poll_metrics`](super::DdpHandle::poll_metrics) receives.
    /// Drained non-blocking via [`Self::poll_metrics`]. `None` on the
    /// single-device path (no coordinator aggregates).
    metrics_rx: Option<std::sync::mpsc::Receiver<EpochMetrics>>,
    /// Controller-elected eval stream, armed only on the cluster path. Fed by
    /// the worker's `dispatch_control` with every `EvalBroadcast` frame — the
    /// eval the controller ran on the rank IT elected (Fastest), not a
    /// hardcoded one. Drained non-blocking via [`Self::poll_eval`]. `None` on
    /// the single-device path (no controller).
    eval_rx: Option<std::sync::mpsc::Receiver<(usize, f64)>>,
}

/// Death-record context carried by a cluster-rank `Worker` for its Drop guard.
struct RankForensics {
    save_path: Option<String>,
    global_rank: usize,
    world_size: usize,
}

enum WorkerInner<M: Module> {
    /// Single-device fallback: no coordinator. Synthesizes full-dataset epoch
    /// plans locally and returns `TrainedState` from its own final snapshot.
    Single {
        worker: GpuWorker<M>,
        num_epochs: usize,
        total_samples: usize,
        next_epoch: usize,
    },
    /// Coordinator-driven rank (multi-GPU / cluster). The `ClusterWorker` owns
    /// the bridge threads; the plan cursor (`wait_for_epoch_plan` +
    /// `fire_epoch_callback`) and the teardown come from its step-wise methods.
    /// Constructed by `DdpHandle::run_cluster_rank_worker` via `into_worker`.
    Cluster(ClusterWorker<M>),
}

impl<M: Module + 'static> Worker<M> {
    /// Wrap a bare [`GpuWorker`] as a single-device cooperative worker.
    /// No coordinator; epoch plans are synthesized locally over the whole
    /// dataset for `num_epochs`.
    pub(crate) fn single(
        worker: GpuWorker<M>,
        num_epochs: usize,
        total_samples: usize,
    ) -> Self {
        Worker {
            inner: WorkerInner::Single {
                worker,
                num_epochs,
                total_samples,
                next_epoch: 0,
            },
            epoch_state: None,
            pending_plan: None,
            compute_start: None,
            active_guard: None,
            shutdown: false,
            finished: false,
            forensics: None, // single device: no peers to unblock
            metrics_rx: None, // no coordinator aggregates on a single device
            eval_rx: None,    // no controller eval on a single device
        }
    }

    /// Wrap a coordinator-connected [`ClusterWorker`] as a cooperative rank.
    /// The forensic context (`save_path` / rank / world size) arms the Drop
    /// guard so an un-`finish()`ed drop writes a death record and exits to
    /// unblock peers.
    pub(crate) fn cluster(
        mut cluster: ClusterWorker<M>,
        save_path: Option<String>,
        global_rank: usize,
        world_size: usize,
    ) -> Self {
        // Arm the full metrics stream before the user runs their first
        // `next_plan`: dispatch (which fills the stream) only happens on this
        // thread, so no aggregated frame can be missed between here and the
        // first drain point.
        let metrics_rx = cluster.inner_mut().enable_metrics_stream();
        let eval_rx = cluster.inner_mut().enable_eval_stream();
        Worker {
            inner: WorkerInner::Cluster(cluster),
            epoch_state: None,
            pending_plan: None,
            compute_start: None,
            active_guard: None,
            shutdown: false,
            finished: false,
            forensics: Some(RankForensics {
                save_path,
                global_rank,
                world_size,
            }),
            metrics_rx: Some(metrics_rx),
            eval_rx: Some(eval_rx),
        }
    }

    fn worker_mut(&mut self) -> &mut GpuWorker<M> {
        match &mut self.inner {
            WorkerInner::Single { worker, .. } => worker,
            WorkerInner::Cluster(cw) => cw.inner_mut(),
        }
    }

    fn worker_ref(&self) -> &GpuWorker<M> {
        match &self.inner {
            WorkerInner::Single { worker, .. } => worker,
            WorkerInner::Cluster(cw) => cw.inner(),
        }
    }

    /// The model, for the user's forward + loss. No rank / device surface —
    /// the cohort is presented as one logical trainer.
    pub fn model(&self) -> &M {
        self.worker_ref().model()
    }

    /// The controller's most recent cross-rank aggregated [`EpochMetrics`], or
    /// `None` before the first epoch has been aggregated. This is the
    /// collective-as-a-whole monitoring surface (never per-rank locals);
    /// `None` on the single-device path (no coordinator aggregates).
    pub fn epoch_metrics(&self) -> Option<EpochMetrics> {
        self.worker_ref()
            .aggregated_metrics()
            .lock()
            .ok()
            .and_then(|g| (*g).clone())
    }

    /// Drain every aggregated [`EpochMetrics`] the controller has broadcast
    /// since the last call, oldest first (non-blocking; empty `Vec` when
    /// nothing new). This is the full per-epoch series — the cooperative
    /// counterpart of the managed [`DdpHandle::poll_metrics`], fed the same
    /// coordinator-aggregated values — so a `Monitor` sees every epoch, not
    /// just the latest (which [`Self::epoch_metrics`] gives).
    ///
    /// **Non-blocking by construction.** The worker thread is the only thing
    /// that advances control (frames are dispatched inside `step` / `next_plan`
    /// on *this* thread), so there is no separate pump to wait on — a blocking
    /// "next" would deadlock. Call it after each `step` / at epoch boundaries
    /// and feed what it returns to your monitor. Always empty on the
    /// single-device path (no controller aggregates).
    ///
    /// [`DdpHandle::poll_metrics`]: super::DdpHandle::poll_metrics
    pub fn poll_metrics(&self) -> Vec<EpochMetrics> {
        match &self.metrics_rx {
            Some(rx) => {
                let mut out = Vec::new();
                while let Ok(m) = rx.try_recv() {
                    out.push(m);
                }
                out
            }
            None => Vec::new(),
        }
    }

    /// Drain the controller-elected eval results broadcast since the last
    /// call, oldest first as `(epoch, metric)` (non-blocking; empty when
    /// nothing new). The eval ran on the rank the controller picked (Fastest
    /// by default) on the coherent consensus model — the whole point of the
    /// cooperative tier: you write one loop and the collective's single-rank
    /// side tasks are placed for you, not pinned to a hardcoded rank.
    ///
    /// The final canonical eval lands here during the loop's terminal
    /// `next_plan` (the controller sends it just before `Shutdown`), so poll
    /// once more after the loop to catch it. Requesting evals mid-run
    /// ([`Self::request_eval`]) surfaces them here per epoch. Same
    /// non-blocking contract as [`Self::poll_metrics`]; always empty on the
    /// single-device path (no controller).
    pub fn poll_eval(&self) -> Vec<(usize, f64)> {
        match &self.eval_rx {
            Some(rx) => {
                let mut out = Vec::new();
                while let Ok(e) = rx.try_recv() {
                    out.push(e);
                }
                out
            }
            None => Vec::new(),
        }
    }

    /// Ask the controller to run the eval callback at its next coherent
    /// occasion (a **request, not a command**): the intent flows to the
    /// controller, which folds it into the role-elected `ExecuteEvalCallback`
    /// dispatch at the next epoch boundary — on the rank its policy elects, not
    /// necessarily this one. The user expresses intent; the controller decides
    /// *when* and *which rank*, preserving the collective's coherence.
    ///
    /// Fire-and-forget. **No-op on the single-device path** (no controller to
    /// service it — drive eval directly in your loop there).
    pub fn request_eval(&self) {
        self.worker_ref()
            .report_intent(crate::distributed::wire::IntentKind::EvalNow);
    }

    /// Ask the controller to checkpoint at its next coherent occasion (a
    /// request, not a command; see [`Self::request_eval`]). Folds into the
    /// role-elected `Checkpoint` dispatch at the next epoch boundary.
    /// Fire-and-forget; no-op on the single-device path.
    pub fn request_checkpoint(&self) {
        self.worker_ref()
            .report_intent(crate::distributed::wire::IntentKind::CheckpointNow);
    }

    /// Advance to the next unit of work the controller dispatches, or `None`
    /// when training is over (all epochs done, or the controller signalled
    /// shutdown). The returned [`EpochPlan`] carries `.epoch` (which epoch this
    /// unit belongs to) and its partition span.
    ///
    /// **Granularity is not one-per-epoch under progressive dispatch** (the
    /// default for Cadence / Async): the controller splits an epoch into
    /// several chunks and dispatches them as separate plans, so this yields
    /// once per *chunk* and the same `.epoch` repeats across consecutive calls.
    /// Under `Sync` it is one plan per epoch. To run per-epoch logic in your
    /// loop, gate on `.epoch` changing rather than on each `next_plan` call
    /// (the framework already fires the registered role-elected `epoch_fn` once
    /// per epoch transition for you).
    ///
    /// On the cluster path this blocks in `wait_for_epoch_plan` (draining
    /// control so a `SyncNow` cannot deadlock a peer).
    pub fn next_plan(&mut self) -> Result<Option<EpochPlan>> {
        if self.shutdown {
            return Ok(None);
        }
        // Release any lingering per-batch guard (restore default stream).
        self.active_guard = None;
        // Defensive: if a caller advanced without draining the prior shard,
        // close it now so its coverage is reported (a well-formed loop drains
        // via `next_batch` returning `None`, which already ran `end_epoch`).
        if let Some(mut st) = self.epoch_state.take() {
            if !st.shutdown() {
                self.worker_mut().end_epoch(&mut st)?;
            }
        }
        self.pending_plan = None;
        self.compute_start = None;

        let plan = match &mut self.inner {
            WorkerInner::Single {
                worker,
                num_epochs,
                total_samples,
                next_epoch,
            } => {
                if *next_epoch >= *num_epochs {
                    None
                } else {
                    let epoch = *next_epoch;
                    *next_epoch += 1;
                    // begin_epoch (run lazily by next_batch) sets current_epoch;
                    // the synth plan covers the whole dataset (no partitioning
                    // on a single device).
                    let _ = worker;
                    Some(EpochPlan {
                        epoch,
                        partition_offset: 0,
                        partition_size: *total_samples,
                    })
                }
            }
            WorkerInner::Cluster(cw) => match cw.inner_mut().wait_for_epoch_plan()? {
                Some(plan) => {
                    cw.fire_epoch_callback(plan.epoch);
                    Some(plan)
                }
                None => None,
            },
        };

        if plan.is_none() {
            // Cluster: shutdown / disconnect. Single: all epochs done.
            self.shutdown = matches!(self.inner, WorkerInner::Cluster(_));
        }
        self.pending_plan = plan.clone();
        Ok(plan)
    }

    /// Yield the next device-ready, transformed batch of the current epoch, or
    /// `None` at the shard end. Runs the per-epoch setup lazily on the first
    /// call (partition / activation-peak / prefetch-depth / VRAM-pool install /
    /// channel open + submit). Drains control while waiting on the prefetch
    /// path. The batch is owned and borrows nothing, so the user can run
    /// forward + backward against it while still calling `&mut self` methods.
    ///
    /// On the shard-end `None` this fires `end_epoch` (coverage accounting); on
    /// a shutdown seen mid-shard it sets the shutdown flag and skips the
    /// report, matching the managed tier.
    pub fn next_batch(&mut self) -> Result<Option<Vec<Tensor>>> {
        // Drop any guard from a prior batch (a well-formed loop already dropped
        // it in `step`; this restores the default stream before re-arming so a
        // misused double `next_batch` cannot stack guards non-LIFO).
        self.active_guard = None;
        // Lazy per-epoch setup on the first call of the epoch.
        if self.epoch_state.is_none() {
            let plan = match self.pending_plan.take() {
                Some(plan) => plan,
                // No live plan (next_batch called before next_plan, or after
                // the shard drained). Nothing to yield.
                None => return Ok(None),
            };
            let st = self.worker_mut().begin_epoch(&plan)?;
            self.epoch_state = Some(st);
        }

        let mut st = self
            .epoch_state
            .take()
            .expect("epoch_state present after lazy setup");
        match self.worker_mut().next_batch_inner(&mut st) {
            Ok(Some(batch)) => {
                // Open the compute-timing window and pin the user's forward +
                // backward to compute_stream, exactly as the managed train_step
                // does: sync_before_forward, then the guard, then the timer.
                self.worker_mut().sync_before_forward()?;
                self.active_guard = self.worker_ref().compute_stream_guard();
                self.compute_start = Some(std::time::Instant::now());
                self.epoch_state = Some(st);
                Ok(Some(batch))
            }
            Ok(None) => {
                // Shard drained, or shutdown seen while waiting.
                if st.shutdown() {
                    self.shutdown = true; // skip end_epoch (matches managed)
                } else {
                    self.worker_mut().end_epoch(&mut st)?;
                }
                // Cursor cleared (st dropped); next_plan starts the next one.
                Ok(None)
            }
            Err(e) => {
                self.epoch_state = Some(st);
                Err(e)
            }
        }
    }

    /// Run the framework-owned tail of a training step for the batch last
    /// yielded by [`Self::next_batch`]: gradient clip, LR schedule, optimizer
    /// step, `zero_grad`, step counters, the per-batch timing report, and the
    /// control drain (where a coordinator `SyncNow` drives the work-weighted
    /// reduce). The caller must have already run the model forward producing
    /// `loss` and called `loss.backward()`.
    ///
    /// `loss` supplies the scalar value for the timing report and the epoch
    /// loss average (read via `.item()`; unchanged by the preceding backward).
    pub fn step(&mut self, loss: &Variable) -> Result<StepOutcome> {
        let mut st = self.epoch_state.take().ok_or_else(|| {
            TensorError::new(
                "Worker::step called with no batch in flight; call next_batch() first",
            )
        })?;

        let loss_val: f64 = loss.data().item()?;
        // Framework-owned tail (clip / LR / opt.step / zero_grad / counters).
        self.worker_mut().optimizer_step_and_bookkeep()?;
        // Close the compute window (forward + backward + tail), matching the
        // managed train_step window.
        let ms = self
            .compute_start
            .take()
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        // report_timing + activation-peak calibration + param-norm + control
        // drain (the SyncNow-driven reduce rides this drain).
        self.worker_mut().after_step(&mut st, loss_val, ms)?;

        let shutdown = st.shutdown();
        if shutdown {
            self.shutdown = true;
        }
        self.epoch_state = Some(st);
        // The user's forward + backward for this batch is complete; release the
        // compute_stream guard (restores the default stream).
        self.active_guard = None;
        Ok(StepOutcome { shutdown })
    }

    /// End training and return the final params + buffers (on CPU) for
    /// inference. Closes any epoch left in flight, then runs the rank teardown
    /// (cluster) or captures the final snapshot (single-device).
    pub fn finish(mut self) -> Result<TrainedState> {
        // Release any lingering per-batch guard before the final accounting.
        self.active_guard = None;
        if let Some(mut st) = self.epoch_state.take() {
            if !st.shutdown() {
                self.worker_mut().end_epoch(&mut st)?;
            }
        }

        // Borrow (never move) self.inner — `Worker` has a Drop impl, so moving
        // a field out is illegal; teardown / snapshot both take `&mut`.
        let state = match &mut self.inner {
            WorkerInner::Single { worker, .. } => {
                let snap = worker.snapshot_params();
                TrainedState {
                    params: snap
                        .params
                        .iter()
                        .map(|t| t.to_device(Device::CPU))
                        .collect::<Result<Vec<_>>>()?,
                    buffers: snap
                        .buffers
                        .iter()
                        .map(|t| t.to_device(Device::CPU))
                        .collect::<Result<Vec<_>>>()?,
                }
            }
            WorkerInner::Cluster(cluster) => {
                // Snapshot tensors already land on CPU (snapshot_params'
                // contract). `None` (worker errored before send_final_snapshot)
                // falls back to an empty state, matching the via_coord path.
                // `finish()` reaching this point IS the clean completion
                // (error paths drop with `finished == false` and exit
                // through the death record instead).
                let final_snapshot = cluster.teardown(true);
                final_snapshot
                    .map(|snap| TrainedState {
                        params: snap.params,
                        buffers: snap.buffers,
                    })
                    .unwrap_or(TrainedState {
                        params: Vec::new(),
                        buffers: Vec::new(),
                    })
            }
        };

        // Disarm the Drop guard ONLY now — every `?` above has cleared, so this
        // is a fully clean finish. If `finish()` had errored above (e.g.
        // `end_epoch`'s report send failed, or the snapshot D2H failed), `self`
        // drops with `finished == false`: a cluster rank then writes its death
        // record + `exit(1)` to unblock peers (matching the managed path's
        // teardown-error handling), while a single-device `Worker` (no
        // forensics) just returns the `Err`.
        self.finished = true;
        Ok(state)
    }
}

impl<M: Module> Drop for Worker<M> {
    /// Blocked-peer-hang protection for the cooperative cluster tier. A rank
    /// whose `Worker` is dropped WITHOUT `finish()` (the user's loop returned
    /// `Err` and propagated, or panicked) is a self-inflicted rank death: write
    /// the forensic record and exit non-zero so the launcher's supervisor
    /// SIGTERMs the peers — mirroring the managed rank entry's `catch_unwind` +
    /// `clean_process_exit(1)`. A clean `finish()` disarms this; the single-device
    /// path has no `forensics` (no peers), so it is a plain no-op.
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let Some(f) = self.forensics.take() else {
            return; // single-device: nothing to record, no peers to unblock
        };
        let panicking = std::thread::panicking();
        let reason = if panicking {
            "cooperative Worker dropped during a panic (user loop panicked)".to_string()
        } else {
            "cooperative Worker dropped without finish() (user loop returned Err)".to_string()
        };
        if let Some(stem) = &f.save_path {
            let record = crate::distributed::RankDeathRecord::new(
                f.global_rank,
                f.world_size,
                reason.clone(),
            );
            let path =
                crate::distributed::CheckpointBundle::rank_death_path(stem, f.global_rank);
            match record.write_to_file(&path) {
                Ok(()) => eprintln!(
                    "flodl cluster rank: wrote death record to {}",
                    path.display()
                ),
                Err(werr) => eprintln!(
                    "flodl cluster rank: failed to write death record to {}: {werr}",
                    path.display()
                ),
            }
        }
        if panicking {
            // Let the unwind continue to terminate the process (exit 101); the
            // panic hook already printed the payload. An unwind drops the
            // libtorch objects properly, so it must NOT be short-circuited
            // with an exit call here.
            eprintln!("flodl cluster rank: {reason}");
        } else {
            eprintln!("flodl cluster rank: {reason}; exiting to unblock peers");
            // clean_process_exit, NOT process::exit: this stack holds a live
            // model/optimizer that never unwinds, and libtorch's static
            // destructors GP-fault over it — turning this deliberate exit(1)
            // into a SIGSEGV/139 that masks the real cause (observed live on
            // the rig). The death record above is already durable (fs::write).
            crate::distributed::ddp_run::clean_process_exit(1);
        }
    }
}
