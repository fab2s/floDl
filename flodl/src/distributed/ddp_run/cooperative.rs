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
//! while let Some(_epoch) = w.next_epoch()? {
//!     while let Some(batch) = w.next_batch()? {   // owned batch, borrows nothing
//!         let loss = train_step(w.model(), &batch)?; // user forward + loss
//!         loss.backward()?;
//!         if w.step(&loss)?.shutdown { break; }      // framework-owned tail + control
//!     }
//! }
//! let state = w.finish()?;
//! ```
//!
//! The reduce is never decided here: it happens as a side effect of the
//! control drain inside `step` (`after_step` -> `handle_control`) when the
//! coordinator's `SyncNow` frame lands, at the ElChe cadence. `step` never
//! names a cadence and never calls the collective directly, which is what keeps
//! the single-step-clock and determinism invariants intact.
//!
//! Construction (`into_worker`) is wired in the following PR; this module is
//! the API surface over the already-built worker.

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
    /// stop; the next `next_epoch` returns `None`.
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
    /// Plan handed out by the last `next_epoch`, consumed lazily by the first
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
    /// `step`'s control drain); `next_epoch` returns `None` from here on.
    shutdown: bool,
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
    ///
    /// Constructed by `DdpHandle::into_worker` (wired in the next PR); the
    /// method arms below are the cooperative API over it, validated by the
    /// live 2-GPU equivalence test.
    #[allow(dead_code)]
    Cluster(ClusterWorker<M>),
}

impl<M: Module + 'static> Worker<M> {
    /// Wrap a bare [`GpuWorker`] as a single-device cooperative worker.
    /// No coordinator; epoch plans are synthesized locally over the whole
    /// dataset for `num_epochs`.
    // `allow(dead_code)`: exercised by the cooperative unit tests now, and by
    // `DdpHandle::into_worker` in the next PR (non-test builds have no caller
    // until then).
    #[allow(dead_code)]
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
        }
    }

    /// Wrap a coordinator-connected [`ClusterWorker`] as a cooperative rank.
    /// Constructed by `DdpHandle::into_worker` (wired in the next PR).
    #[allow(dead_code)]
    pub(crate) fn cluster(cluster: ClusterWorker<M>) -> Self {
        Worker {
            inner: WorkerInner::Cluster(cluster),
            epoch_state: None,
            pending_plan: None,
            compute_start: None,
            active_guard: None,
            shutdown: false,
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

    /// Advance to the next epoch. `Some(plan)` opens an epoch (the plan is the
    /// coordinator's — or, single-device, the whole dataset); `None` ends the
    /// training loop (all epochs done, or the controller signalled shutdown).
    ///
    /// On the cluster path this blocks in `wait_for_epoch_plan` (draining
    /// control so a `SyncNow` cannot deadlock a peer) and fires the
    /// role-elected `epoch_fn` on the epoch transition.
    pub fn next_epoch(&mut self) -> Result<Option<EpochPlan>> {
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
                // No live epoch (next_batch called before next_epoch, or after
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
                // Cursor cleared (st dropped); next_epoch starts the next one.
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

        match self.inner {
            WorkerInner::Single { mut worker, .. } => {
                let snap = worker.snapshot_params();
                Ok(TrainedState {
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
                })
            }
            WorkerInner::Cluster(mut cluster) => {
                // Snapshot tensors already land on CPU (snapshot_params'
                // contract). `None` (worker errored before send_final_snapshot)
                // falls back to an empty state, matching the via_coord path.
                let final_snapshot = cluster.teardown();
                Ok(final_snapshot
                    .map(|snap| TrainedState {
                        params: snap.params,
                        buffers: snap.buffers,
                    })
                    .unwrap_or(TrainedState {
                        params: Vec::new(),
                        buffers: Vec::new(),
                    }))
            }
        }
    }
}
