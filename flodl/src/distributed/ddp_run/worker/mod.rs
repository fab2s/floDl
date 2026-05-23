//! GPU worker: thread-local training loop bound to a single device.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::nn::buffer::Buffer;
use crate::distributed::cuda_event::CudaEvent;
use crate::distributed::cuda_stream::CudaStream;
use crate::distributed::nccl::{NcclAbortHandle, NcclRankComm};
use crate::nn::{Module, Optimizer};
use crate::tensor::{Device, Tensor};

use super::{
    CheckpointFn, EpochMetrics, EvalFn, TimingMsg, MetricsMsg,
    ParamSnapshot, ControlMsg, EpochPlan,
};


mod constructor;
mod control;
mod epoch_plan;
mod reporting;
mod self_driven;
mod sync;

// ---------------------------------------------------------------------------
// GpuWorker
// ---------------------------------------------------------------------------

/// A training worker bound to a single GPU device.
///
/// Generic over the model type `M` so training closures can access concrete
/// model methods (graph tags, traces, etc.) beyond the [`Module`] trait.
///
/// NOT Send: contains `Rc<RefCell<...>>` types (Variable, Buffer, Module).
/// Must be constructed *inside* the spawned thread from Send ingredients.
///
/// Each worker runs an independent training loop with its own optimizer.
/// Communication with the coordinator is via channels (all messages are Send).
pub struct GpuWorker<M: Module> {
    // -- Thread-local model state (non-Send) --
    model: M,
    optimizer: Box<dyn Optimizer>,
    /// Ordered parameter variables for gradient extraction and param loading.
    pub(super) param_vars: Vec<Variable>,
    /// Ordered buffers for buffer synchronization.
    buffer_list: Vec<Buffer>,

    // -- Worker identity --
    rank: usize,
    /// World size — populated from [`super::WorkerConfig::world_size`]. Read by
    /// the self-driven cluster-rank loop to compute its data-partition
    /// slice; not used by the coordinator-driven path (the coordinator
    /// owns world topology there).
    world_size: usize,
    device: Device,

    // -- CUDA streams for overlap (None on CPU) --
    compute_stream: Option<CudaStream>,
    comm_stream: Option<CudaStream>,
    /// Recorded on comm_stream after param copy/AllReduce.
    /// compute_stream waits on this before each forward.
    copy_done: Option<CudaEvent>,
    /// Pending H2D wait flag for the cpu-avg path. Set in `load_averaged`
    /// when params are copy_(non_blocking) on `comm_stream`, cleared in
    /// `sync_before_forward` after host-synchronizing the comm stream.
    /// Moves the post-Update H2D wait OUTSIDE `train_step`'s timing window —
    /// otherwise the implicit GPU sync at `loss.data().item()?` propagates
    /// the queued `wait_event` into `batch_ms` and pollutes ElChe's
    /// throughput signal mode-asymmetrically (cpu-avg only; NCCL path
    /// host-synchronizes inside `sync_now_nccl` already, so no flag set there).
    pending_param_h2d: bool,
    /// Most recent host-side H2D wait inside `sync_before_forward` (ms).
    /// Diagnostic only — not fed back into the controller. Useful for
    /// verifying the pollution removal under different rig topologies
    /// (e.g. PCIe x8 vs chipset x2 lines on heterogeneous boards).
    last_h2d_wait_ms: f64,

    // -- NCCL per-rank communicator (None for CPU averaging or CPU device) --
    nccl_comm: Option<NcclRankComm>,
    /// Abort handle for the current NCCL comm. Cloned from `nccl_comm`
    /// at construction time (and re-cloned by [`Self::replace_nccl_comm`]
    /// after a rebuild). The cluster-mode NCCL watchdog thread holds
    /// its own clone of this `Arc` and calls `abort()` when the local
    /// `DeadRanks` ledger registers a newly-dead peer. The main thread
    /// uses [`NcclAbortHandle::is_aborted`] to distinguish "our abort"
    /// from other NCCL errors in [`Self::sync_now_nccl`].
    nccl_abort_handle: Option<Arc<NcclAbortHandle>>,
    /// Cluster-mode NCCL session mailbox. Populated by the
    /// cluster_worker inbound bridge on each `NewNcclSession` arrival;
    /// drained here by `sync_now_nccl` post-abort to rebuild the comm.
    /// `None` outside cluster mode (standalone single-process NCCL has
    /// no rendezvous channel).
    nccl_session_mailbox: Option<
        Arc<Mutex<Option<crate::distributed::cluster_worker::PendingNcclSession>>>,
    >,
    /// Cluster-mode local dead-rank ledger (a clone of the
    /// `cluster_worker`'s ledger). Polled by `wait_for_nccl_session` to
    /// detect the lone-survivor case (NCCL needs `world_size >= 2`); in
    /// that case the rendezvous is impossible and we bail fast instead
    /// of waiting 60s for a `NewNcclSession` that will never arrive.
    /// `None` outside cluster mode.
    local_dead_ranks: Option<Arc<crate::distributed::controller::DeadRanks>>,

    // -- Channels --
    timing_tx: mpsc::Sender<TimingMsg>,
    metrics_tx: mpsc::Sender<MetricsMsg>,
    /// Used only with AverageBackend::Cpu.
    param_tx: mpsc::Sender<ParamSnapshot>,
    /// Dedicated channel for the final snapshot (avoids race with CPU averaging param_tx).
    final_param_tx: mpsc::Sender<ParamSnapshot>,
    control_rx: mpsc::Receiver<ControlMsg>,

    // -- Data iteration --
    dataset: Arc<dyn BatchDataSet>,
    /// Sample indices for this worker's current partition.
    pub(super) partition: Vec<usize>,
    batch_size: usize,
    base_seed: u64,

    // -- Training state --
    local_step: usize,
    /// Batches since last averaging (for snapshot weighting).
    steps_since_avg: usize,
    current_version: u64,
    pub(super) current_epoch: usize,
    /// Queued epoch plan from coordinator (set if StartEpoch arrives during run_epoch_plan).
    pub(super) pending_plan: Option<EpochPlan>,
    /// Cumulative total batches across all GPUs at last sync.
    /// Updated by `SetGlobalStep` from the coordinator after averaging.
    /// Workers compute per-batch LR as `scheduler.lr(global_step + steps_since_avg)`.
    global_step: usize,
    /// Per-batch LR scheduler. When set, the worker adjusts the optimizer's
    /// learning rate before each `optimizer.step()`.
    scheduler: Option<Arc<dyn crate::nn::Scheduler>>,
    /// DDP linear-scaling factor (Goyal et al., 2017). Applied multiplicatively
    /// to the scheduler's output each batch, so schedulers see the scaling too.
    /// When no scheduler is attached, the scaling is baked into the optimizer
    /// once at startup via [`Self::scale_lr`]. Default: 1.0 (no scaling).
    lr_scale: f64,
    /// Most recent aggregated [`EpochMetrics`] broadcast from the coord
    /// (via [`crate::distributed::wire::ControlMsgWire::EpochAggregated`]).
    /// Shared with the user's `Graph` (when running setup-mode training)
    /// so `Graph::latest_metrics()` and `graph_gpu_metrics()` surface
    /// the global cross-rank view to user code without the user needing
    /// to think about ranks. `None` until the first aggregation lands
    /// (cold-start, single-GPU runs that never trigger coord-side
    /// aggregation).
    aggregated_metrics: Arc<Mutex<Option<EpochMetrics>>>,

    // -- Checkpoint --
    /// Called on rank 0 after averaging events. Log-and-continue on error.
    pub(super) checkpoint_fn: Option<CheckpointFn<M>>,
    /// Sticky rank designated by the controller to fire `epoch_fn` at
    /// epoch transitions. `None` until the coord broadcasts the first
    /// [`ControlMsg::SetEpochCallbackRole`]; while `None`, the
    /// autonomous epoch-transition fire in the cluster worker's main
    /// loop is gated off (no rank fires until the controller has
    /// chosen). Coord resolves the value at startup
    /// ([`crate::distributed::ddp_run::EpochCallbackPolicy::Rank`]) or
    /// runtime ([`EpochCallbackPolicy::Fastest`]) and updates this
    /// state via wire-pushed
    /// [`crate::distributed::wire::ControlMsgWire::SetEpochCallbackRole`].
    ///
    /// [`EpochCallbackPolicy::Fastest`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Fastest
    pub(super) epoch_callback_role: Option<usize>,
    /// User-supplied eval callback. Fires from [`Self::handle_control`]
    /// on [`ControlMsg::ExecuteEvalCallback`] receipt, only on the rank
    /// chosen by [`crate::distributed::ddp_run::EpochCallbackPolicy`]
    /// (other ranks see `None`). The framework flips
    /// [`Module::eval`]/[`Module::train`] around the closure and ships
    /// the scalar result back to the controller via
    /// [`TimingMsg::EvalResult`].
    pub(super) eval_fn: Option<EvalFn<M>>,
    /// Held-out eval dataset paired with `eval_fn`. Required when
    /// `eval_fn` is set (the framework loud-errors if absent at eval
    /// firing time). `Arc<dyn BatchDataSet>` matches the training
    /// `dataset` shape; user iterates batches inside the closure.
    pub(super) eval_dataset: Option<Arc<dyn BatchDataSet>>,
    /// Checkpoint-bundle stem for the cluster save-on-unrecoverable-
    /// failure flow. Populated from [`super::WorkerConfig::save_path`]. When
    /// set, the worker writes a `<save_path>.fdl` / `<save_path>.optim`
    /// / `<save_path>.meta.json` bundle on
    /// [`ControlMsg::ShutdownWithSave`] receipt; when unset, the
    /// shutdown still happens but no save attempt is made.
    save_path: Option<String>,

    // -- Async prefetch (VRAM gauge) --
    /// Background prefetch worker for async H2D transfers (None on CPU).
    prefetch: Option<crate::data::prefetch::PrefetchWorker>,
    /// Bytes per sample (for VRAM gauge depth calculation).
    per_sample_bytes: usize,
    /// Measured activation peak (activations + gradients) from training.
    /// Used as a reserve in the VRAM gauge so prefetch doesn't fill
    /// memory that forward/backward will need. Zero = not yet measured;
    /// first chunk runs sync to calibrate.
    activation_peak_bytes: usize,
    /// Maximum gradient norm for clipping (None = no clipping).
    max_grad_norm: Option<f64>,
    /// EASGD elastic averaging weight (0, 1]. `None` = full overwrite
    /// (current behavior; uses fast non-blocking copy_). When `Some(α)`,
    /// `load_averaged` blends `W_local := (1-α)·W_local + α·W_avg` instead
    /// of overwriting. Reference: Zhang, Choromanska, LeCun, NeurIPS 2015.
    easgd_alpha: Option<f64>,
    /// Optional system timeline for event injection.
    timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,

    /// Scratch buffers for pre-sync parameter snapshot (weight-space divergence).
    /// Allocated once at worker creation. `None` when policy == Sync (divergence
    /// is near-zero by construction, no point measuring) or no NCCL comm.
    pre_sync_scratch: Option<Vec<Tensor>>,

    /// Strong references to each parameter's AccumulateGrad node, created
    /// under `StreamGuard(compute_stream)` during worker init. Keeping
    /// these alive pins the nodes' streams to `compute_stream` across
    /// the worker's lifetime; without this, the nodes are GCed between
    /// iterations and re-created on the autograd engine's default stream,
    /// triggering libtorch's "AccumulateGrad stream does not match" warning.
    ///
    /// DO NOT REMOVE: never read at runtime, existence is the point. The
    /// `_` prefix signals intentional liveness-only ownership. Dropping
    /// this field at any point before worker teardown re-introduces the
    /// stream-mismatch bug on the next backward().
    _grad_accumulators: Vec<crate::tensor::GradAccumulatorHandle>,
}

/// Channels bundle returned by [`GpuWorker::channels`] for wiring into the coordinator.
pub struct WorkerChannels {
    /// Receives timing reports from this worker.
    pub timing_rx: mpsc::Receiver<TimingMsg>,
    /// Receives epoch-end metrics from this worker.
    pub metrics_rx: mpsc::Receiver<MetricsMsg>,
    /// Receives parameter snapshots from this worker (CPU averaging path).
    pub param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Receives the final parameter snapshot from this worker (sent before exit).
    pub final_param_rx: mpsc::Receiver<ParamSnapshot>,
    /// Sends control messages to this worker.
    pub control_tx: mpsc::Sender<ControlMsg>,
}

/// Worker-side channel endpoints for passing into [`GpuWorker::new`].
#[allow(clippy::type_complexity)]
pub type WorkerEndpoints = (
    mpsc::Sender<TimingMsg>,
    mpsc::Sender<MetricsMsg>,
    mpsc::Sender<ParamSnapshot>,
    mpsc::Sender<ParamSnapshot>,  // final_param_tx
    mpsc::Receiver<ControlMsg>,
);

impl<M: Module> GpuWorker<M> {
    /// This worker's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The rank designated by the controller to fire `epoch_fn` at
    /// each epoch transition. `None` until the coord has resolved
    /// [`crate::distributed::ddp_run::EpochCallbackPolicy`] and
    /// broadcast [`ControlMsg::SetEpochCallbackRole`]; while `None`
    /// the autonomous fire is gated off. The cluster worker's main
    /// loop consults this via the public accessor.
    pub fn epoch_callback_role(&self) -> Option<usize> {
        self.epoch_callback_role
    }

    /// This worker's device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Current local step count.
    pub fn local_step(&self) -> usize {
        self.local_step
    }

    /// Current model version (updated after loading averaged params).
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Current epoch number (0-based).
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Clone of this worker's current NCCL abort handle, if any.
    ///
    /// The cluster-mode watchdog thread holds one of these and calls
    /// [`NcclAbortHandle::abort`] when a peer rank dies; the main
    /// thread checks [`NcclAbortHandle::is_aborted`] in the NCCL
    /// `sync_now_nccl` error path to distinguish "our abort" from
    /// other failures. Returns `None` for CPU averaging / non-cluster
    /// setups where no NCCL comm exists.
    pub fn nccl_abort_handle(&self) -> Option<Arc<NcclAbortHandle>> {
        self.nccl_abort_handle.clone()
    }

    /// Attach the cluster-mode NCCL session mailbox. Called by the
    /// cluster_worker layer between construction and the main loop.
    /// When set, `sync_now_nccl`'s abort-retry path reads new comm
    /// bytes from this slot (populated by the inbound bridge on each
    /// `NewNcclSession` frame from the coord). No-op for standalone
    /// single-process NCCL setups.
    pub(crate) fn attach_nccl_session_mailbox(
        &mut self,
        mailbox: Arc<Mutex<Option<crate::distributed::cluster_worker::PendingNcclSession>>>,
    ) {
        self.nccl_session_mailbox = Some(mailbox);
    }

    /// Attach the cluster-mode local dead-rank ledger. Polled by
    /// [`Self::wait_for_nccl_session`] to detect the lone-survivor case
    /// (NCCL needs `world_size >= 2`); in that case the rendezvous is
    /// impossible and we bail fast. No-op for standalone NCCL setups.
    pub(crate) fn attach_local_dead_ranks(
        &mut self,
        dead_ranks: Arc<crate::distributed::controller::DeadRanks>,
    ) {
        self.local_dead_ranks = Some(dead_ranks);
    }

    /// Replace this worker's NCCL communicator with `new_comm` after a
    /// cluster-mode re-rendezvous (peer rank died, surviving cohort
    /// formed a fresh comm).
    ///
    /// `self.rank` and `self.world_size` are STABLE global identity —
    /// they index per-rank coord-side state (ElChe batch counts,
    /// wall-ms accumulators, divergence vectors, partition computation,
    /// `TimingMsg::*.rank` fields) and must not change across a
    /// rebuild. The NCCL comm tracks its own internal `rank` /
    /// `world_size` (= position in the shrunken cohort / cohort size)
    /// for collective dispatch; the worker never reads those.
    pub fn replace_nccl_comm(&mut self, new_comm: NcclRankComm) {
        self.nccl_abort_handle = Some(new_comm.abort_handle());
        self.nccl_comm = Some(new_comm);
    }

    /// Set the learning rate on this worker's optimizer.
    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    /// Current learning rate on this worker's optimizer. Reflects the most
    /// recent value written by either [`Self::set_lr`], the attached
    /// scheduler (in `train_step`), or [`Self::scale_lr`].
    pub fn current_lr(&self) -> f64 {
        self.optimizer.lr()
    }

    /// Scale the learning rate by a factor (for DDP linear scaling rule).
    ///
    /// Applies the scaling to the optimizer immediately. Has no effect on
    /// subsequent schedulers: use [`Self::set_lr_scale`] for a factor that
    /// persists across scheduler updates.
    pub fn scale_lr(&mut self, factor: f64) {
        self.optimizer.scale_lr(factor);
    }

    /// Set the DDP linear-scaling factor without touching the optimizer's
    /// current LR. Applied multiplicatively to the attached scheduler's
    /// output on every batch, so the scaling survives per-batch LR updates.
    pub fn set_lr_scale(&mut self, scale: f64) {
        self.lr_scale = scale;
    }

    /// Attach a per-batch LR scheduler.
    ///
    /// When set, the worker computes
    /// `scheduler.lr(global_step + steps_since_avg) * lr_scale` before each
    /// optimizer step, ensuring the LR tracks global training progress and
    /// honors the DDP linear-scaling rule.
    pub fn set_scheduler(&mut self, sched: Arc<dyn crate::nn::Scheduler>) {
        self.scheduler = Some(sched);
    }

    /// A reference to the concrete model.
    pub fn model(&self) -> &M {
        &self.model
    }
}
