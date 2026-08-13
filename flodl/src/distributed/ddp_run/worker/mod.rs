//! GPU worker: thread-local training loop bound to a single device.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::nccl::{NcclAbortHandle, NcclRankComm};
use crate::nn::buffer::Buffer;
use crate::nn::{Module, Optimizer};
use crate::tensor::cuda_event::GpuEvent;
use crate::tensor::cuda_stream::GpuStream;
use crate::tensor::{Device, Tensor};

use super::{
    AveragedParams, CheckpointFn, ControlMsg, EpochMetrics, EpochPlan, EvalFn, MetricsMsg,
    ParamSnapshot, TimingMsg,
};

mod constructor;
mod control;
mod epoch_plan;
mod reporting;
mod stager;
mod sync;

/// Per-epoch training cursor, driven step-wise by the cooperative
/// [`Worker`](crate::distributed::ddp_run::Worker) tier.
pub(crate) use epoch_plan::EpochState;
/// Crate-internal export for the NCCL test suite (`nccl_tests.rs`
/// exercises the fused weighted collective directly on a 2-GPU comm;
/// the suite compiles CPU-side too and self-skips at runtime, so this
/// is gated on `test` alone, not the cuda feature).
#[cfg(test)]
pub(crate) use sync::weighted_allreduce_nccl;

// ---------------------------------------------------------------------------
// GpuWorker
// ---------------------------------------------------------------------------

/// Shared cell holding the CURRENT NCCL abort handle for a rank.
///
/// Written by [`GpuWorker::replace_nccl_comm`] on every comm rebuild;
/// read by the cluster-mode NCCL watchdog on every firing, so the
/// watchdog always aborts the live comm across cascading peer deaths.
pub(crate) type NcclAbortSlot = Arc<Mutex<Option<Arc<NcclAbortHandle>>>>;

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
    compute_stream: Option<GpuStream>,
    comm_stream: Option<GpuStream>,
    /// Recorded on comm_stream after param copy/AllReduce.
    /// compute_stream waits on this before each forward.
    copy_done: Option<GpuEvent>,
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
    /// Instrumentation: instant of the most recent `load_averaged`
    /// (averaged params applied). The cluster run loop reads it to split
    /// the between-chunk wait into "blocked until Update arrived"
    /// (reduce + slow-rank compute wait) vs "Update → next StartEpoch"
    /// (post-reduce dispatch wait — should be ~0 if atomic-dispatch is
    /// landing).
    last_update_at: Option<std::time::Instant>,
    /// Instrumentation: cumulative GPU←CPU writeback wait (the post-Update
    /// H2D sync in `sync_before_forward`, i.e. "update their weights"
    /// cost the CPU path pays and NCCL's in-place collective doesn't).
    h2d_wait_ms_total: f64,
    /// Instrumentation gate: cached `crate::log::enabled(Verbosity::Debug)`
    /// (i.e. `-vvv`), read once at construction. Guards ALL profiling
    /// collection + reporting below so the prof path costs nothing (and
    /// emits nothing) at normal verbosity.
    prof_enabled: bool,
    /// Instrumentation: cumulative time + count of `snapshot_params`
    /// calls driven by `RequestParams` (the GPU→CPU *readout* the CPU
    /// averaging path pays each reduce window). This is the suspected
    /// cpu-cadence floor: a synchronous, per-param `to_device(CPU)` on
    /// the compute thread, worst on slow-PCIe ranks. Timed at the
    /// `dispatch_control` call site (`&mut self`); the final-snapshot
    /// readout (`reporting::send_final_snapshot`) is excluded since it's
    /// teardown, not per-window.
    snapshot_ns_total: u128,
    snapshot_count: u64,
    /// Instrumentation: run-level sums of the per-chunk `compute_ms` and
    /// `data_starve_ms` the balancer feeds into `share_complete_ms`
    /// (= compute + data). The cluster run loop's `run_epoch` total minus
    /// these two is the "other (ctrl/sync/transport)" overhead ElChe does
    /// NOT see — the suspected cpu-vs-nccl gap (per-batch report_timing
    /// TCP writes + handle_control). Plus a count of control messages
    /// processed, to test "more messages in cpu mode".
    compute_ms_run_total: f64,
    data_ms_run_total: f64,
    ctrl_msgs_handled: u64,

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
    /// Cluster-mode shared abort slot. The NCCL watchdog re-reads the
    /// CURRENT handle from here on every firing and
    /// [`Self::replace_nccl_comm`] refreshes it on every rebuild, so
    /// cascading peer deaths always abort the live comm (a handle
    /// captured at spawn time goes stale at the first rebuild).
    /// `None` outside cluster mode.
    nccl_abort_slot: Option<NcclAbortSlot>,
    /// Cluster-mode NCCL session mailbox. Populated by the
    /// cluster_worker inbound bridge on each `NewNcclSession` arrival;
    /// drained here by `sync_now_nccl` post-abort to rebuild the comm.
    /// `None` outside cluster mode (standalone single-process NCCL has
    /// no rendezvous channel).
    nccl_session_mailbox:
        Option<Arc<Mutex<Option<crate::distributed::nccl_session::PendingNcclSession>>>>,
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
    /// Slices per data pass (see [`super::WorkerConfig::epoch_splits`]).
    /// `1` means an epoch is a full pass; above that, `epoch` counts
    /// events and this rank resolves its slice of the pass permutation.
    epoch_splits: usize,
    /// Augmentation multiplicity: the partition is a PICK stream and a
    /// pick decodes as `(pick / augment, pick % augment)`.
    augment: usize,
    /// Delivery transform, keyed per pick (see
    /// [`crate::data::TransformFn`]). Applied after device transfer,
    /// before the training step.
    transform: Option<crate::data::TransformFn>,

    // -- Training state --
    local_step: usize,
    /// Count of NCCL sync cycles this rank has processed. Diagnostic-only
    /// (`-vvv` collective-step logging in `weighted_allreduce_nccl`): every
    /// rank receives the coordinator's `SyncNow` broadcasts in order, so this
    /// counter is the SAME across ranks for the same reduce — a lagging value
    /// on one rank pins a cohort desync.
    nccl_sync_seq: usize,
    /// Batches since last averaging (for snapshot weighting).
    steps_since_avg: usize,
    /// `steps_since_avg` captured when the last `RequestParams` snapshot
    /// shipped — the step count whose work that frame carried. On `Update`
    /// the counter subtracts THIS instead of zeroing: under cpu-async the
    /// worker keeps training through the round-trip and the incoming
    /// consensus is EASGD-blended, so the overshoot steps' work survives in
    /// the params — their mass must ride the NEXT frame. Pre-snapshot steps
    /// count fully despite sitting on a previous blend, so full credit for
    /// the overshoot is the consistent accounting (zeroing was the
    /// inconsistency). Under sync/cadence no steps land between snapshot and
    /// Update, so subtract and zero coincide.
    steps_at_snapshot: usize,
    /// Consensus allocation-weighting exponent γ (1.0 = plain
    /// work-weighting). The NCCL weighted reduce folds it into the
    /// PreMulSum factor (`nᵢ^γ / Σn^γ`); the CPU path applies it in the
    /// cluster-worker bridge's frame weighting. See
    /// [`crate::distributed::ElCheConfig::gamma`].
    gamma: f64,
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
    /// Cooperative-tier full metrics stream. `None` in managed / setup mode
    /// (no accumulation, zero cost). When armed by
    /// [`Self::enable_metrics_stream`] (only on the cooperative `Worker`
    /// cluster path), every `EpochAggregated` frame is *also* forwarded here,
    /// not just folded into the latest-only `aggregated_metrics` slot — so the
    /// user's [`crate::distributed::Worker::poll_metrics`] drains the same
    /// per-epoch series the managed [`crate::distributed::DdpHandle::poll_metrics`]
    /// gets from the coordinator's launcher-side sink. Fed from
    /// `dispatch_control`, so it populates from both `step` and `next_plan`.
    metrics_stream_tx: Option<mpsc::Sender<EpochMetrics>>,
    /// Cooperative-tier eval stream, sibling of [`Self::metrics_stream_tx`].
    /// `None` in managed / setup mode. When armed by
    /// [`Self::enable_eval_stream`] (cooperative `Worker` cluster path),
    /// every `EvalBroadcast` frame (`(epoch, metric)`) is forwarded here, so
    /// [`crate::distributed::Worker::poll_eval`] surfaces the controller-
    /// elected eval without a launcher. Fed from `dispatch_control`.
    eval_stream_tx: Option<mpsc::Sender<(usize, f64)>>,

    // -- Checkpoint --
    /// Called on rank 0 after averaging events. Log-and-continue on error.
    pub(super) checkpoint_fn: Option<CheckpointFn<M>>,
    /// Sticky rank designated by the controller to fire `epoch_fn` at
    /// epoch transitions. `None` until the coord broadcasts the first
    /// `ControlMsg::SetEpochCallbackRole`; while `None`, the
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
    /// on `ControlMsg::ExecuteEvalCallback` receipt, only on the rank
    /// chosen by [`crate::distributed::ddp_run::EpochCallbackPolicy`]
    /// (other ranks see `None`). The framework flips
    /// [`Module::eval`]/[`Module::train`] around the closure and ships
    /// the scalar result back to the controller via
    /// `TimingMsg::EvalResult`.
    pub(super) eval_fn: Option<EvalFn<M>>,
    /// Held-out eval dataset paired with `eval_fn`. Required when
    /// `eval_fn` is set (the framework loud-errors if absent at eval
    /// firing time). `Arc<dyn BatchDataSet>` matches the training
    /// `dataset` shape; user iterates batches inside the closure.
    pub(super) eval_dataset: Option<Arc<dyn BatchDataSet>>,
    /// Armed consensus eval (`ControlMsg::ArmConsensusEval`): fires at
    /// the next REALIZED `ControlMsg::Update`, scoring that round's
    /// consensus. `(schedule_id, epoch)` ride back on the
    /// `TimingMsg::EvalResult`. Survives unrealized (all-idle) rounds;
    /// evaporates at shutdown if no realized round follows (best-effort,
    /// like every callback dispatch).
    pub(super) pending_consensus_eval: Option<(u64, u64)>,
    /// The last realized averaging round's consensus, retained for the
    /// final canonical eval (`ExecuteEvalCallback { adopt_consensus }`).
    /// Shallow clones — under decode-into they alias the pinned snapshot
    /// staging, so the retained tensors are only valid until the next
    /// snapshot D2H clobbers those bytes: `RequestParams` clears this
    /// BEFORE snapshotting, and a realized `Update` re-retains. At the
    /// natural end the settle sequencing guarantees the last realized
    /// round is retained when the final eval frame arrives.
    pub(super) last_consensus: Option<AveragedParams>,
    /// Checkpoint-bundle stem for the cluster save-on-unrecoverable-
    /// failure flow. Populated from [`super::WorkerConfig::save_path`]. When
    /// set, the worker writes a `<save_path>.fdl` / `<save_path>.optim`
    /// / `<save_path>.meta.json` bundle on
    /// `ControlMsg::ShutdownWithSave` receipt; when unset, the
    /// shutdown still happens but no save attempt is made.
    save_path: Option<String>,

    // -- Async prefetch (VRAM gauge) --
    /// Background prefetch worker for async H2D transfers (None on CPU).
    prefetch: Option<crate::data::prefetch::PrefetchWorker>,
    /// Background reservation stager: warms the sample-keyed staging
    /// tier (shared with `dataset`, which is the read-through wrapper)
    /// from coordinator `StageAdvisory` frames. Dormant until the
    /// first advisory.
    stager: Option<stager::StagerHandle>,
    /// Bytes per sample (for VRAM gauge depth calculation).
    per_sample_bytes: usize,
    /// VRAM share for this worker's data plane (see
    /// [`super::WorkerConfig::vram_max_usage`]); feeds the per-plan
    /// prefetch-depth sizing.
    vram_max_usage: f64,
    /// Host-RAM share for this worker's data plane; feeds the per-plan
    /// reader-ring sizing (same knob the stager budgets from).
    ram_max_usage: f64,
    /// GPU share of host RAM on an integrated target; corrects the
    /// staging budget for a unified-memory part (see
    /// [`super::DdpRunConfig::gpu_ram_share`]).
    gpu_ram_share: Option<f64>,
    /// Measured activation peak (activations + gradients) from training.
    /// Used as a reserve in the VRAM gauge so prefetch doesn't fill
    /// memory that forward/backward will need. Zero = not yet measured;
    /// first chunk runs sync to calibrate.
    activation_peak_bytes: usize,
    /// One-shot latch: the device sample pool's budget install has been
    /// signalled to the prefetch worker (first post-calibration plan
    /// boundary, when the activation peak makes the VRAM probe honest).
    vram_pool_budget_sent: bool,
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
    /// Companion scratch for the f32 buffers riding the NCCL sync
    /// (mover-averaged running stats). Only used by the abort-recovery
    /// retry in `sync_now_nccl` — an aborted buffer collective can leave
    /// running stats partially premultiplied, and unlike params there is
    /// no divergence readout to co-fund the allocation, so this exists
    /// purely so retries restart from clean buffer state. `None` when no
    /// NCCL comm or the model has no f32 buffers.
    pre_sync_buffer_scratch: Option<Vec<Tensor>>,

    /// Outer optimizer applied to the consensus on the NCCL path, replicated
    /// per rank. After the in-place AllReduce leaves the work-weighted
    /// consensus on every rank's params, each rank runs
    /// `outer_step(prev_global, consensus)` on its own GPU copies. Identical
    /// inputs + a deterministic op give every rank the same new global and
    /// momentum update, so the cohort stays in lock-step with NO extra
    /// collective. `None` = no outer optimizer (today's plain averaging).
    /// Built once per rank from the user's `.outer_optimizer(..)` factory.
    outer_optimizer: Option<Box<dyn crate::distributed::OuterOptimizer>>,
    /// Per-rank `prev_global` anchor for the NCCL outer step: the global this
    /// rank adopted at the end of the previous reduce window. `None` on the
    /// first window (the consensus is used as the anchor, so the outer
    /// gradient is zero). A param-sized GPU buffer, allocated lazily; with the
    /// outer optimizer's replicated momentum this is the +2 param-sized GPU
    /// buffers/rank of the DiLoCo/SlowMo footprint.
    outer_prev_global: Option<Vec<Tensor>>,

    /// Persistent pinned (page-locked) host staging buffers for the
    /// GPU->CPU parameter / buffer readout in [`Self::snapshot_params`]
    /// (CPU averaging path). Lazily allocated on the first snapshot (one
    /// per param / buffer, matching shape+dtype) and REUSED every reduce
    /// window. Pinned allocation is expensive and non-swappable, hence the
    /// reuse; the batched async D2H into these buffers collapses the old
    /// per-param synchronous `to_device(CPU)` (N serialized device syncs,
    /// the cpu-cadence idle floor on slow-PCIe ranks) down to a single
    /// `synchronize()`.
    ///
    /// SINGLE-CONSUMER INVARIANT: the returned [`ParamSnapshot`] shares
    /// storage with these buffers, so each reduce window must FULLY consume
    /// its snapshot before the next `snapshot_params` overwrites them. The
    /// coordinator's CPU-averaging cadence guarantees this: it issues
    /// one `RequestParams` per cycle and the worker re-snapshots only
    /// after the resulting `Update` round-trips back (the `CpuAvgPhase`
    /// Idle -> Pending -> Idle cycle), so reuse never aliases an
    /// in-flight snapshot.
    ///
    /// On the barrier-paced path the same window admits one more
    /// writer and reader THROUGH this shared storage: the reduce
    /// client decodes the consensus reply back INTO these buffers
    /// (`CpuReduceClient::set_decode_into_request` — the snapshot bytes
    /// are dead once the streamed encode has read them; a bf16-wire
    /// reply lands verbatim in the bf16 staging), and `load_averaged`'s
    /// async H2D then reads them; the next window's `snapshot_params`
    /// entry fence retires that H2D before the D2H overwrite. Same
    /// invariant, zero extra locked memory.
    /// Empty on CPU device / non-CPU-averaging setups (the readout falls
    /// back to a per-tensor passthrough).
    ///
    /// With [`WorkerConfig::bf16_wire`] the PARAM staging allocates as
    /// bfloat16 (the D2H `copy_` casts on the source device): half the
    /// pinned RAM and half the PCIe bytes, matching the bf16 frames the
    /// reduce bridge builds from the snapshot. Buffer staging is exempt
    /// (see `snapshot_pinned_buffers`), and the end-of-training snapshot
    /// bypasses this staging entirely (`snapshot_params_exact`).
    ///
    /// [`WorkerConfig::bf16_wire`]: crate::distributed::ddp_run::WorkerConfig::bf16_wire
    snapshot_pinned_params: Vec<Tensor>,
    /// Companion pinned staging buffers for non-learnable buffers
    /// (BatchNorm running stats etc.). Same lazy-alloc + reuse +
    /// single-consumer contract as `snapshot_pinned_params`, but ALWAYS
    /// at the buffer's native dtype even under `bf16_wire`: the reduce
    /// bridge selects reduce-eligible buffers by `dtype() == Float32`,
    /// so a bf16-staged running stat would silently fall out of the
    /// sync (the NCCL-buffers bug class); the wire cast still halves
    /// their frames, and they are KB-scale regardless.
    snapshot_pinned_buffers: Vec<Tensor>,
    /// Stage the param snapshot in bfloat16 (see
    /// [`WorkerConfig::bf16_wire`](crate::distributed::ddp_run::WorkerConfig::bf16_wire)).
    bf16_wire: bool,
    /// Whether the pinned-readout failure has been reported. The fallback
    /// is silent-correct but slow; log the regression exactly once.
    pinned_fallback_logged: bool,

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
///
/// In the single-host fallback the receivers and `control_tx` are *held*
/// (not drained) so the worker's senders and `control_rx` stay connected
/// for the whole run; only `metrics_rx` is consumed, because there is no
/// coordinator to drain the rest. The multi-GPU path wires its channels
/// inline and reads the receivers through `Coordinator::builder`. Several
/// fields are therefore liveness-held rather than read in the non-test
/// build (and exercised directly by the worker tests).
#[allow(dead_code)]
pub(crate) struct WorkerChannels {
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
pub(crate) type WorkerEndpoints = (
    mpsc::Sender<TimingMsg>,
    mpsc::Sender<MetricsMsg>,
    mpsc::Sender<ParamSnapshot>,
    mpsc::Sender<ParamSnapshot>, // final_param_tx
    mpsc::Receiver<ControlMsg>,
);

impl<M: Module> GpuWorker<M> {
    /// This worker's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Instrumentation: instant of the most recent `load_averaged`
    /// (averaged params applied). `None` until the first sync.
    pub fn last_update_at(&self) -> Option<std::time::Instant> {
        self.last_update_at
    }

    /// Instrumentation: cumulative GPU←CPU writeback (post-Update H2D
    /// sync) wait in ms — the "update their weights" cost of the CPU
    /// averaging path.
    pub fn h2d_wait_ms_total(&self) -> f64 {
        self.h2d_wait_ms_total
    }

    /// Instrumentation: cumulative GPU→CPU snapshot readout
    /// (`snapshot_params` via `RequestParams`) time in ms and call
    /// count. The per-window cost the CPU averaging path pays to publish
    /// weights for the reduce; the suspected cpu-cadence idle floor.
    pub fn snapshot_readout_ms_total(&self) -> f64 {
        self.snapshot_ns_total as f64 / 1e6
    }

    /// Instrumentation: number of per-window `snapshot_params` readouts.
    pub fn snapshot_readout_count(&self) -> u64 {
        self.snapshot_count
    }

    /// Instrumentation gate (`-vvv`): whether profiling collection +
    /// reporting is active. Read by the cluster run loop to gate its
    /// own timing and the `[worker-prof]` teardown line.
    pub fn prof_enabled(&self) -> bool {
        self.prof_enabled
    }

    /// Instrumentation: run-level compute / data sums (ms) the balancer
    /// uses as `share_complete_ms`, plus the control-message count.
    /// `run_epoch - (compute + data)` is the ctrl/sync/transport overhead
    /// ElChe doesn't see.
    pub fn compute_ms_run_total(&self) -> f64 {
        self.compute_ms_run_total
    }

    /// Instrumentation: run-level data-starve (prefetch/load wait) ms.
    pub fn data_ms_run_total(&self) -> f64 {
        self.data_ms_run_total
    }

    /// Instrumentation: count of control messages processed by
    /// `dispatch_control` (tests "more messages in cpu mode").
    pub fn ctrl_msgs_handled(&self) -> u64 {
        self.ctrl_msgs_handled
    }

    /// The rank designated by the controller to fire `epoch_fn` at
    /// each epoch transition. `None` until the coord has resolved
    /// [`crate::distributed::ddp_run::EpochCallbackPolicy`] and
    /// broadcast `ControlMsg::SetEpochCallbackRole`; while `None`
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

    /// Test-only window into the mass counter (see `steps_at_snapshot`).
    #[cfg(test)]
    pub(crate) fn steps_since_avg(&self) -> usize {
        self.steps_since_avg
    }

    /// Test-only setter simulating trained batches (the production
    /// increment lives in the epoch-plan loop).
    #[cfg(test)]
    pub(crate) fn set_steps_since_avg(&mut self, n: usize) {
        self.steps_since_avg = n;
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
        mailbox: Arc<Mutex<Option<crate::distributed::nccl_session::PendingNcclSession>>>,
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

    /// Attach the cluster-mode shared NCCL abort slot (see
    /// [`NcclAbortSlot`]). [`Self::replace_nccl_comm`] refreshes it on
    /// every rebuild so the watchdog always holds the live comm's
    /// handle. No-op for standalone NCCL setups.
    pub(crate) fn attach_nccl_abort_slot(&mut self, slot: NcclAbortSlot) {
        self.nccl_abort_slot = Some(slot);
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
        let handle = new_comm.abort_handle();
        // Re-arm the cluster-mode watchdog: it re-reads this slot on
        // every firing, so the NEXT peer death aborts the rebuilt comm
        // instead of no-op'ing on the old handle's tripped flag.
        if let Some(slot) = &self.nccl_abort_slot {
            *slot.lock().expect("nccl abort slot poisoned") = Some(Arc::clone(&handle));
        }
        self.nccl_abort_handle = Some(handle);
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

    /// Suspend graph profiling around a non-training forward (eval,
    /// epoch callbacks), returning whether it was active. Those fire on
    /// ONE elected rank, so letting them feed the accumulator would
    /// tilt exactly that rank's per-node means. Pair with
    /// [`resume_graph_profiling`](Self::resume_graph_profiling) when
    /// this returns true.
    pub(crate) fn pause_graph_profiling(&self) -> bool {
        use crate::graph::GraphExt;
        (&self.model as &dyn Module)
            .as_graph()
            .is_some_and(|g| g.pause_profiling())
    }

    /// Re-activate graph profiling after
    /// [`pause_graph_profiling`](Self::pause_graph_profiling).
    pub(crate) fn resume_graph_profiling(&self) {
        use crate::graph::GraphExt;
        if let Some(g) = (&self.model as &dyn Module).as_graph() {
            g.resume_profiling();
        }
    }

    /// Ship the accumulated graph timings to the coordinator (one
    /// frame, clean teardown only). Best-effort: no graph, no samples,
    /// or a dropped channel all mean no frame, never an error: the
    /// heat map is optional UX, not a training invariant.
    pub(crate) fn emit_graph_profile(&self) {
        use crate::graph::GraphExt;
        let Some(stats) = (&self.model as &dyn Module)
            .as_graph()
            .and_then(|g| g.profile_stats())
        else {
            return;
        };
        let gpu_model = match self.device {
            Device::CUDA(idx) => crate::tensor::gpu_device_name_idx(i32::from(idx))
                .unwrap_or_else(|| "gpu".to_string()),
            Device::CPU => "cpu".to_string(),
        };
        let profile = crate::distributed::wire::GraphProfileWire {
            hash: stats.structural_hash,
            gpu_model,
            source: stats.source.label().to_string(),
            samples: stats.samples as u64,
            total_min_ms: stats.total_min.as_secs_f64() * 1000.0,
            total_mean_ms: stats.total_mean.as_secs_f64() * 1000.0,
            nodes: stats
                .nodes
                .into_iter()
                .map(|n| crate::distributed::wire::GraphNodeTimingWire {
                    id: n.id,
                    level: n.level as u32,
                    min_ms: n.min.as_secs_f64() * 1000.0,
                    mean_ms: n.mean.as_secs_f64() * 1000.0,
                })
                .collect(),
        };
        let _ = self.timing_tx.send(TimingMsg::GraphProfile {
            rank: self.rank,
            profile,
        });
    }
}
