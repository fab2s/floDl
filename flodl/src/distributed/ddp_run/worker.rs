//! GPU worker: thread-local training loop bound to a single device.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use crate::autograd::{Variable, NoGradGuard};
use crate::data::BatchDataSet;
use crate::nn::buffer::Buffer;
use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
use crate::distributed::cuda_stream::{CudaStream, StreamGuard};
use crate::distributed::nccl::{NcclRankComm, ReduceOp};
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::{
    CheckpointFn, WorkerConfig, TimingMsg, MetricsMsg,
    ParamSnapshot, AveragedParams, ControlMsg, EpochPlan, make_partition,
};

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

    // -- Checkpoint --
    /// Called on rank 0 after averaging events. Log-and-continue on error.
    pub(super) checkpoint_fn: Option<CheckpointFn<M>>,

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
    /// Create the channel pairs for one worker.
    ///
    /// Returns (worker-side senders/receiver, coordinator-side receivers/sender).
    /// Call this on the main thread, then pass the worker-side halves into
    /// [`GpuWorker::new`] inside the spawned thread.
    pub fn channels() -> (WorkerEndpoints, WorkerChannels) {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (metrics_tx, metrics_rx) = mpsc::channel();
        let (param_tx, param_rx) = mpsc::channel();
        let (final_param_tx, final_param_rx) = mpsc::channel();
        let (control_tx, control_rx) = mpsc::channel();
        (
            (timing_tx, metrics_tx, param_tx, final_param_tx, control_rx),
            WorkerChannels { timing_rx, metrics_rx, param_rx, final_param_rx, control_tx },
        )
    }

    /// Build a GpuWorker inside a spawned thread.
    ///
    /// `model_factory` creates the model on `config.device` (thread-local, Rc-based).
    /// `optim_factory` creates the optimizer for the model's parameters.
    /// `initial_params`/`initial_buffers` from `WorkerConfig` are copied into the
    /// model's Variables to synchronize all workers to the same starting state.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F, G, O>(
        config: &WorkerConfig,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        nccl_comm: Option<NcclRankComm>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        timing_tx: mpsc::Sender<TimingMsg>,
        metrics_tx: mpsc::Sender<MetricsMsg>,
        param_tx: mpsc::Sender<ParamSnapshot>,
        final_param_tx: mpsc::Sender<ParamSnapshot>,
        control_rx: mpsc::Receiver<ControlMsg>,
    ) -> Result<Self>
    where
        F: FnOnce(Device) -> Result<M>,
        G: FnOnce(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        // Create CUDA streams first (before model construction) so model
        // parameters are allocated on the same stream used by subsequent
        // forward/backward passes. Without this, AccumulateGrad nodes end
        // up on the default stream while gradients arrive on compute_stream,
        // triggering libtorch's "stream does not match" warning and breaking
        // CUDA graph capture.
        let (compute_stream, comm_stream, copy_done) = if config.device.is_cuda() {
            let cs = CudaStream::new(config.device, false)?;
            let ms = CudaStream::new(config.device, false)?;
            let ev = CudaEvent::new(CudaEventFlags::DisableTiming)?;
            // Record initial event so first wait_event is a no-op
            ev.record_on(&ms)?;
            (Some(cs), Some(ms), Some(ev))
        } else {
            (None, None, None)
        };

        // Build the model under compute_stream so every leaf tensor
        // (parameters, buffers) and the AccumulateGrad nodes created at
        // first backward belong to the training stream.
        let model = {
            let _guard = compute_stream.as_ref().map(StreamGuard::new);
            model_factory(config.device)?
        };
        let params = model.parameters();
        let buffers = model.buffers();

        // Copy initial params into model variables on compute_stream
        // (no_grad: leaf tensors with requires_grad).
        if params.len() != config.initial_params.len() {
            return Err(TensorError::new(&format!(
                "GpuWorker rank {}: model has {} params but config has {}",
                config.rank, params.len(), config.initial_params.len()
            )));
        }
        {
            let _guard = compute_stream.as_ref().map(StreamGuard::new);
            let _no_grad = NoGradGuard::new();
            for (p, src) in params.iter().zip(&config.initial_params) {
                p.variable.data().copy_(src, false)?;
            }
        }

        // Copy initial buffers into model buffers on compute_stream.
        if buffers.len() != config.initial_buffers.len() {
            return Err(TensorError::new(&format!(
                "GpuWorker rank {}: model has {} buffers but config has {}",
                config.rank, buffers.len(), config.initial_buffers.len()
            )));
        }
        {
            let _guard = compute_stream.as_ref().map(StreamGuard::new);
            for (b, src) in buffers.iter().zip(&config.initial_buffers) {
                b.get().copy_(src, false)?;
            }
        }

        // Eagerly materialize each parameter's AccumulateGrad node under
        // compute_stream and hold a strong reference so it survives
        // between iterations. The node captures the current CUDA stream
        // at construction time into its input_metadata. If the node is
        // GCed and re-created on the autograd engine's worker thread
        // (whose current stream is the device default), libtorch fires
        // the "AccumulateGrad stream does not match" warning on every
        // DDP run that uses a non-default training stream.
        let grad_accumulators: Vec<crate::tensor::GradAccumulatorHandle> = {
            let _guard = compute_stream.as_ref().map(StreamGuard::new);
            let mut handles = Vec::with_capacity(params.len());
            for p in &params {
                if let Some(h) = p.variable.ensure_grad_accumulator()? {
                    handles.push(h);
                }
            }
            handles
        };

        // Create optimizer for this replica's parameters on compute_stream
        // so optimizer state tensors (momentum, Adam moments, ...) are
        // allocated on the same stream as the gradients that will update them.
        let optimizer = {
            let _guard = compute_stream.as_ref().map(StreamGuard::new);
            optim_factory(&params)
        };

        // Extract variable handles (for snapshot/load)
        let param_vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        let buffer_list = buffers;

        // Create prefetch worker for async H2D (VRAM gauge).
        // Cap depth at 512 to avoid huge channel allocations when
        // batch_bytes is tiny (e.g. toy test datasets).
        // Depth 0 = skip prefetch entirely (sync fallback for tight VRAM).
        // Skip entirely when the dataset fits in a single batch (nothing to
        // prefetch ahead of).
        //
        // Note: activation_reserve=0 here because we haven't measured the
        // training activation peak yet. The first run_epoch_plan() will
        // force depth=0 (sync) to calibrate, then adjust on subsequent chunks.
        let total_batches = dataset.len() / config.batch_size.max(1);
        let (prefetch, per_sample_bytes) = if config.device.is_cuda() && total_batches > 1 {
            let sample = dataset.get_batch(&[0])?;
            let psb: usize = sample.iter().map(|t| t.nbytes()).sum();
            drop(sample);
            let depth = crate::data::prefetch_depth_from_vram(
                psb, config.batch_size, config.device, 0.90, 0,
            ).min(512);
            // Reset peak stats so first run_epoch_plan gets a clean baseline.
            crate::tensor::cuda_reset_peak_stats_idx(config.device.index() as i32);
            if depth > 0 {
                let pw = crate::data::prefetch::PrefetchWorker::new(
                    Arc::clone(&dataset), config.device, depth,
                );
                (Some(pw), psb)
            } else {
                (None, psb)
            }
        } else {
            (None, 0)
        };

        // Allocate scratch buffers for weight-space divergence measurement.
        // Skip for Sync mode (AllReduce every batch, divergence near-zero).
        let pre_sync_scratch = if nccl_comm.is_some() && config.policy != super::ApplyPolicy::Sync {
            let scratch: Result<Vec<Tensor>> = param_vars.iter()
                .map(|v| Tensor::zeros_like(&v.data()))
                .collect();
            scratch.ok()
        } else {
            None
        };

        Ok(GpuWorker {
            model,
            optimizer: Box::new(optimizer),
            param_vars,
            buffer_list,
            rank: config.rank,
            device: config.device,
            compute_stream,
            comm_stream,
            copy_done,
            pending_param_h2d: false,
            last_h2d_wait_ms: 0.0,
            nccl_comm,
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
            control_rx,
            dataset,
            partition: Vec::new(), // filled by first StartEpoch from coordinator
            batch_size: config.batch_size,
            base_seed: config.seed,
            local_step: 0,
            steps_since_avg: 0,
            current_version: 0,
            current_epoch: 0,
            pending_plan: None,
            global_step: 0,
            scheduler: None,
            lr_scale: 1.0,
            checkpoint_fn,
            prefetch,
            per_sample_bytes,
            activation_peak_bytes: 0,
            max_grad_norm: config.max_grad_norm,
            easgd_alpha: config.easgd_alpha,
            timeline: config.timeline.clone(),
            pre_sync_scratch,
            _grad_accumulators: grad_accumulators,
        })
    }

    /// This worker's rank.
    pub fn rank(&self) -> usize {
        self.rank
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

    /// Extract current parameter values as a [`ParamSnapshot`].
    ///
    /// Tensors are copied to CPU so that the coordinator's compute thread
    /// never needs CUDA access (avoiding slow CUDA context init on the
    /// compute thread, which can deadlock with `drain_avg_state`).
    ///
    /// Synchronizes comm_stream before reading, so Update + RequestParams
    /// processed in the same `handle_control()` call cannot read mid-copy data.
    pub fn snapshot_params(&self) -> ParamSnapshot {
        // Wait for any pending load_averaged() non-blocking copy to finish.
        if let Some(stream) = &self.comm_stream {
            let _ = stream.synchronize();
        }
        let params = self.param_vars.iter()
            .map(|v| {
                let t = v.data();
                if t.device() == Device::CPU { t } else { t.to_device(Device::CPU).unwrap_or(t) }
            })
            .collect();
        let buffers = self.buffer_list.iter()
            .map(|b| {
                let t = b.get();
                if t.device() == Device::CPU { t } else { t.to_device(Device::CPU).unwrap_or(t) }
            })
            .collect();
        ParamSnapshot {
            rank: self.rank,
            params,
            buffers,
            batch_count: self.steps_since_avg.max(1),
        }
    }

    /// Load averaged parameters from the coordinator (CPU averaging path).
    ///
    /// Uses `copy_(non_blocking=true)` on the comm stream for GPU overlap.
    /// Records a `CudaEvent` so the compute stream waits before the next forward.
    pub fn load_averaged(&mut self, update: &AveragedParams) -> Result<()> {
        if update.params.len() != self.param_vars.len() {
            return Err(TensorError::new(&format!(
                "load_averaged: expected {} params, got {}",
                self.param_vars.len(), update.params.len()
            )));
        }

        let non_blocking = self.comm_stream.is_some();
        // Set comm stream as current if available (copy_ respects current stream)
        let _guard = self.comm_stream.as_ref().map(StreamGuard::new);

        // no_grad: parameters are leaf tensors with requires_grad=true
        //
        // Two paths: full overwrite (default, fast — single non-blocking
        // copy_ per param) and EASGD elastic blend (opt-in, when
        // `easgd_alpha` is set — stages averaged params on GPU, then
        // applies a batched in-place lerp `var := var + α(avg − var)`
        // = `(1−α)·var + α·avg` across all params via one CUDA kernel
        // launch). Buffers (BatchNorm running stats etc.) always overwrite
        // — blending them is undefined under the EASGD framework.
        {
            let _no_grad = NoGradGuard::new();
            match self.easgd_alpha {
                None => {
                    for (var, src) in self.param_vars.iter().zip(&update.params) {
                        var.data().copy_(src, non_blocking)?;
                    }
                }
                Some(alpha) => {
                    let mut avg_staged: Vec<crate::tensor::Tensor> =
                        Vec::with_capacity(update.params.len());
                    let mut dst_handles: Vec<crate::tensor::Tensor> =
                        Vec::with_capacity(update.params.len());
                    for (var, src) in self.param_vars.iter().zip(&update.params) {
                        let dst = var.data();
                        // zeros_like allocates a same-shape/dtype/device
                        // tensor; we immediately overwrite via copy_ so the
                        // initial zeroing is unused (no `empty_like` in flodl).
                        let avg_gpu = crate::tensor::Tensor::zeros_like(&dst)?;
                        avg_gpu.copy_(src, non_blocking)?;
                        avg_staged.push(avg_gpu);
                        dst_handles.push(dst);
                    }
                    crate::tensor::Tensor::foreach_lerp_scalar_(
                        &dst_handles, &avg_staged, alpha,
                    )?;
                }
            }
        }
        for (buf, src) in self.buffer_list.iter().zip(&update.buffers) {
            buf.get().copy_(src, non_blocking)?;
        }

        // Record event on comm_stream so compute_stream can wait
        if let (Some(ev), Some(stream)) = (&self.copy_done, &self.comm_stream) {
            ev.record_on(stream)?;
        }
        // Mark H2D pending so the next `sync_before_forward` host-syncs the
        // comm stream BEFORE `train_step`'s timing window opens. Without
        // this, the wait_event-based path leaks the H2D wait into
        // `batch_ms` via the implicit GPU sync at `loss.item()`.
        self.pending_param_h2d = true;

        self.current_version = update.version;
        Ok(())
    }

    /// Perform in-place NCCL AllReduce(Avg) on this rank's parameters.
    ///
    /// All ranks must process SyncNow concurrently for the collective to complete.
    /// Runs on `comm_stream` and records `copy_done` so the compute stream waits
    /// before the next forward.
    ///
    /// Performs the in-place NCCL AllReduce(Avg) on this rank's parameters
    /// and returns the divergence triple `(divergence, post_norm, pre_norm)`:
    /// - `divergence = ||pre - post|| / ||post||` (this rank's transversal
    ///   deviation from the post-AllReduce consensus),
    /// - `post_norm = ||post||` (the L2 norm of the consensus weights after
    ///   AllReduce; identical across ranks by construction),
    /// - `pre_norm = ||W_i||` (this rank's pre-AllReduce L2 norm; per-rank).
    ///
    /// All three are `None` together when scratch buffers are absent
    /// (Sync mode or no NCCL comm). With all three available the coordinator
    /// gets the cosine-similarity / magnitude-shift decomposition for free
    /// (MSF/SWA directional vs magnitude split) plus the longitudinal
    /// meta-oscillator state.
    fn sync_now_nccl(&self) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
        let _diag_start = Instant::now();
        let comm = match &self.nccl_comm {
            Some(c) => c,
            None => return Ok((None, None, None)),
        };

        let param_tensors: Vec<_> = self.param_vars.iter().map(|v| v.data()).collect();
        let param_refs: Vec<&Tensor> = param_tensors.iter().collect();

        // Snapshot pre-sync params into scratch buffers for divergence measurement.
        if let Some(ref scratch) = self.pre_sync_scratch {
            let _guard = self.comm_stream.as_ref().map(StreamGuard::new);
            for (dst, src) in scratch.iter().zip(&param_tensors) {
                dst.copy_(src, true)?; // non-blocking on comm_stream
            }
        }

        if let Some(stream) = &self.comm_stream {
            // AllReduce on comm_stream (in-place averaging)
            let nccl_start = Instant::now();
            comm.all_reduce_on_stream(&param_refs, ReduceOp::Avg, stream)?;
            // HOST-synchronize: block until AllReduce completes.
            stream.synchronize()?;
            let nccl_ms = nccl_start.elapsed().as_secs_f64() * 1000.0;

            // Compute weight-space divergence: ||pre - post|| / ||post||
            let divg_start = Instant::now();
            let divergence = if let Some(ref scratch) = self.pre_sync_scratch {
                // scratch = pre (from copy above). Compute pre-norm BEFORE
                // mutating scratch (the next foreach_add_list_ overwrites it
                // in place to scratch = pre - post).
                let pre_norm_tensors = Tensor::foreach_norm(scratch, 2.0)?;
                let mut pre_sq = 0.0f64;
                for n in &pre_norm_tensors {
                    let v: f64 = n.item()?;
                    pre_sq += v * v;
                }
                let pre_norm = pre_sq.sqrt();

                // scratch[i] += (-1) * param_tensors[i]  ->  scratch[i] = pre[i] - post[i]
                Tensor::foreach_add_list_(scratch, &param_tensors, -1.0)?;

                let diff_norms = Tensor::foreach_norm(scratch, 2.0)?;
                let post_norms = Tensor::foreach_norm(&param_tensors, 2.0)?;

                let mut diff_sq = 0.0f64;
                for n in &diff_norms {
                    let v: f64 = n.item()?;
                    diff_sq += v * v;
                }
                let mut post_sq = 0.0f64;
                for n in &post_norms {
                    let v: f64 = n.item()?;
                    post_sq += v * v;
                }

                let post_norm = post_sq.sqrt();
                let div = if post_norm > 1e-10 {
                    diff_sq.sqrt() / post_norm
                } else {
                    0.0
                };

                crate::verbose!(
                    "  ddp-worker: rank {} sync divergence={:.6} (||delta||={:.4}, ||pre||={:.4}, ||post||={:.4})",
                    self.rank, div, diff_sq.sqrt(), pre_norm, post_norm,
                );
                (Some(div), Some(post_norm), Some(pre_norm))
            } else {
                (None, None, None)
            };

            // Record event so compute_stream waits before next forward
            if let Some(ev) = &self.copy_done {
                ev.record_on(stream)?;
            }
            let divg_ms = divg_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = _diag_start.elapsed().as_secs_f64() * 1000.0;
            crate::verbose!(
                "  ddp-sync-diag: rank {} sync_total={:.1}ms (nccl={:.1}ms divg={:.1}ms)",
                self.rank, total_ms, nccl_ms, divg_ms,
            );

            Ok(divergence)
        } else {
            comm.all_reduce(&param_refs, ReduceOp::Avg)?;
            Ok((None, None, None))
        }
    }

    /// Host-synchronize the comm stream so any pending cpu-avg H2D copy
    /// completes BEFORE `train_step`'s timing window opens.
    ///
    /// Must be called before each forward pass. The previous implementation
    /// queued a `compute_stream.wait_event(copy_done)` here — a stream-level
    /// dependency that returned to the host immediately. The catch: the
    /// implicit GPU sync at `loss.data().item()?` inside the timing window
    /// then propagated the H2D wait into the measured `batch_ms`, polluting
    /// ElChe's throughput signal asymmetrically (cpu-avg only — NCCL's
    /// `sync_now_nccl` host-synchronizes internally, so there's nothing to
    /// wait for here on that path).
    ///
    /// The host-sync moves that wait outside the timing window. Total wall
    /// time is identical (we pay the H2D wait either way); only the clock
    /// that measures it changes. The `pending_param_h2d` flag avoids
    /// per-batch synchronize overhead on batches that don't follow an Update.
    /// No-op on CPU (no streams).
    fn sync_before_forward(&mut self) -> Result<()> {
        if self.pending_param_h2d
            && let Some(stream) = &self.comm_stream
        {
            let t = Instant::now();
            stream.synchronize()?;
            self.last_h2d_wait_ms = t.elapsed().as_secs_f64() * 1000.0;
            self.pending_param_h2d = false;
        }
        Ok(())
    }

    /// Run one forward + backward + optimizer step.
    ///
    /// `train_fn` receives a reference to the concrete model `M` and the batch
    /// tensors, and must return the scalar loss [`Variable`]. The worker handles
    /// stream sync, backward, optimizer step, and zero_grad.
    ///
    /// Returns `(loss_value, wall_ms)`.
    pub fn train_step(
        &mut self,
        batch: &[Tensor],
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<(f64, f64)> {
        self.sync_before_forward()?;

        // Pin all CUDA work for this call to compute_stream. The
        // AccumulateGrad nodes for this worker's parameters are pinned to
        // compute_stream in GpuWorker::new (see `_grad_accumulators`). The
        // gradient-producing kernels invoked here must arrive on the same
        // stream or libtorch fires the input_buffer.cpp:240
        // "AccumulateGrad node's stream does not match" warning.
        // dispatch_next_chunk already wraps the chunk loop in StreamGuard,
        // so this nests harmlessly there. Direct callers (custom training
        // loops, tests) need it here for the same guarantee.
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        let start = Instant::now();

        // User-provided forward + loss computation
        let loss = train_fn(&self.model, batch)?;
        let loss_val: f64 = loss.data().item()?;

        // Backward
        loss.backward()?;

        // Per-worker gradient clipping (before optimizer step).
        if let Some(max_norm) = self.max_grad_norm {
            let params: Vec<Tensor> = self.model.parameters()
                .iter()
                .filter(|p| p.variable.grad().is_some())
                .map(|p| p.variable.data())
                .collect();
            if !params.is_empty() {
                Tensor::clip_grad_norm_fused(&params, max_norm)?;
            }
        }

        // Per-batch LR: scheduler tracks global progress.
        // global_step = total batches at last sync, steps_since_avg = local
        // batches since then. The LR reflects this worker's real position
        // in the global schedule, multiplied by the DDP linear-scaling factor
        // (1.0 when lr_scale_ratio == 0 or world_size == 1).
        if let Some(ref sched) = self.scheduler {
            let base = sched.lr(self.global_step + self.steps_since_avg);
            self.optimizer.set_lr(base * self.lr_scale);
        }

        // Optimizer step (GPU-local Adam, ~0.1ms fused kernel)
        self.optimizer.step()?;
        self.optimizer.zero_grad();

        self.local_step += 1;
        self.steps_since_avg += 1;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((loss_val, elapsed_ms))
    }

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
    fn dispatch_control(&mut self, msg: ControlMsg) -> Result<bool> {
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
            ControlMsg::Checkpoint { version } => {
                if let Some(ref f) = self.checkpoint_fn {
                    if let Err(e) = f(version, &self.model) {
                        eprintln!("  ddp: checkpoint failed (v{version}): {e}");
                    }
                }
            }
            ControlMsg::Shutdown => return Ok(true),
        }
        Ok(false)
    }

    /// Send a timing report to the coordinator.
    pub fn report_timing(
        &self,
        batch_ms: f64,
        param_norm: Option<f64>,
        batch_loss: f64,
        sync_divergence: Option<f64>,
    ) -> Result<()> {
        self.timing_tx.send(TimingMsg::Batch {
            rank: self.rank,
            batch_ms,
            step_count: self.local_step,
            param_norm,
            batch_loss,
            sync_divergence,
        }).map_err(|_| TensorError::new("timing channel disconnected"))
    }

    /// Compute the L2 norm of all model parameters.
    ///
    /// Uses `Tensor::foreach_norm` for a single batched CUDA kernel instead
    /// of per-parameter norm calls. Returns the global L2 norm (sqrt of sum
    /// of squared per-tensor norms). Used for NCCL divergence detection.
    fn compute_param_norm(&self) -> Result<f64> {
        let data: Vec<Tensor> = self.param_vars.iter().map(|v| v.data()).collect();
        if data.is_empty() {
            return Ok(0.0);
        }
        let norms = Tensor::foreach_norm(&data, 2.0)?;
        let mut total_sq = 0.0f64;
        for n in &norms {
            let val: f64 = n.item()?;
            total_sq += val * val;
        }
        Ok(total_sq.sqrt())
    }

    /// Send the final parameter snapshot on the dedicated channel before exiting.
    ///
    /// This uses `final_param_tx` (not `param_tx`) to avoid racing with
    /// CPU averaging snapshot collection on the same channel.
    pub fn send_final_snapshot(&self) {
        let _ = self.final_param_tx.send(self.snapshot_params());
    }

    /// Abort the NCCL communicator, unblocking any stuck collective.
    ///
    /// Must be called before [`Self::send_final_snapshot`] when the training loop
    /// exits due to shutdown. A pending AllReduce on `comm_stream` (from a
    /// SyncNow whose peer died) would block `to_device(CPU)` in snapshot_params
    /// because the CUDA default stream synchronizes with all other streams.
    pub fn abort_nccl(&mut self) {
        if let Some(comm) = self.nccl_comm.take() {
            let _ = comm.abort_handle().abort();
        }
    }

    /// Notify the coordinator that this worker is about to exit.
    ///
    /// Must be called before the thread terminates so the coordinator
    /// stops including this rank in NCCL collectives.
    pub fn report_exiting(&self) {
        let _ = self.timing_tx.send(TimingMsg::Exiting { rank: self.rank });
    }

    /// Send epoch-end metrics to the coordinator.
    ///
    /// Drains the thread-local scalar accumulator populated by
    /// [`record_scalar()`](super::record_scalar) calls during this epoch.
    pub fn report_epoch(
        &self,
        avg_loss: f64,
        batches: usize,
        epoch_ms: f64,
        share_complete_ms: f64,
        compute_only_ms: f64,
        data_starve_ms: f64,
    ) -> Result<()> {
        let scalars = super::drain_scalars();
        self.metrics_tx.send(MetricsMsg {
            rank: self.rank,
            epoch: self.current_epoch,
            avg_loss,
            batches_processed: batches,
            epoch_ms,
            samples_processed: batches * self.batch_size,
            share_complete_ms,
            compute_only_ms,
            data_starve_ms,
            scalars,
        }).map_err(|_| TensorError::new("metrics channel disconnected"))
    }

    /// Block until the coordinator sends a StartEpoch or Shutdown.
    ///
    /// Handles intermediate control messages (SyncNow, RequestParams, etc.)
    /// to prevent NCCL deadlock while waiting between epochs.
    /// Returns `Some(plan)` for the next epoch, or `None` on Shutdown/disconnect.
    pub fn wait_for_epoch_plan(&mut self) -> Result<Option<EpochPlan>> {
        crate::debug!("  ddp-worker: rank {} waiting for plan (step={})", self.rank, self.local_step);
        let wait_start = Instant::now();
        loop {
            // Check if a plan was queued by dispatch_control (e.g. StartEpoch
            // arrived during Throttle handler). Must be checked each iteration,
            // not just at entry, because dispatch_control may set it mid-loop.
            if let Some(plan) = self.pending_plan.take() {
                let waited = wait_start.elapsed().as_secs_f64() * 1000.0;
                crate::verbose!("  ddp-dispatch-diag: rank {} waited {:.0}ms (pending plan)", self.rank, waited);
                crate::debug!("  ddp-worker: rank {} got plan (pending) epoch={}", self.rank, plan.epoch);
                return Ok(Some(plan));
            }
            match self.control_rx.recv() {
                Ok(ControlMsg::StartEpoch(plan)) => {
                    let waited = wait_start.elapsed().as_secs_f64() * 1000.0;
                    crate::verbose!("  ddp-dispatch-diag: rank {} waited {:.0}ms for StartEpoch", self.rank, waited);
                    crate::debug!("  ddp-worker: rank {} got plan epoch={}", self.rank, plan.epoch);
                    return Ok(Some(plan));
                }
                Ok(ControlMsg::Shutdown) => return Ok(None),
                Ok(msg) => {
                    crate::debug!("  ddp-worker: rank {} wait_for_plan got {:?}", self.rank,
                        match &msg {
                            ControlMsg::SyncNow => "SyncNow",
                            ControlMsg::Throttle => "Throttle",
                            ControlMsg::RequestParams => "RequestParams",
                            ControlMsg::Update(_) => "Update",
                            ControlMsg::SetGlobalStep(_) => "SetGlobalStep",
                            ControlMsg::Checkpoint { .. } => "Checkpoint",
                            ControlMsg::Shutdown => "Shutdown",
                            ControlMsg::StartEpoch(_) => "StartEpoch",
                        }
                    );
                    if self.dispatch_control(msg)? {
                        return Ok(None); // Shutdown consumed by handler (e.g. Throttle)
                    }
                }
                Err(_) => return Ok(None), // disconnected
            }
        }
    }

    /// Process one partition (or chunk) from the coordinator's plan.
    ///
    /// Generates sample indices from the plan's offset and size using the
    /// same deterministic shuffle as all other ranks. Reports metrics at
    /// the end so the coordinator can track completion.
    ///
    /// On CUDA, batches are prefetched asynchronously via a background
    /// worker thread with a VRAM-sized buffer (gauge model). On CPU,
    /// batches are loaded synchronously.
    ///
    /// Returns `true` if a Shutdown was received mid-plan.
    pub fn run_epoch_plan(
        &mut self,
        plan: &EpochPlan,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<bool> {
        self.current_epoch = plan.epoch;
        self.partition = make_partition(
            plan.partition_offset, plan.partition_size,
            self.dataset.len(), plan.epoch, self.base_seed,
        );

        let num_batches = self.partition.len() / self.batch_size;
        if num_batches == 0 {
            // Still report so coordinator gets the "done" signal.
            let _ = self.report_epoch(0.0, 0, 0.0, 0.0, 0.0, 0.0);
            return Ok(false);
        }

        // ALL CUDA work must avoid the default stream and device-wide sync.
        // The CUDA default stream implicitly synchronizes with every other
        // stream, and cuda_synchronize waits for ALL streams on the device.
        // If a SyncNow triggered AllReduce on comm_stream (via the other rank)
        // while this rank touches the default stream or calls device sync,
        // it blocks waiting for comm_stream which waits for this rank -> deadlock.
        //
        // Solution: use compute_stream for all ops, sync compute_stream only.
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        // NOTE: cuda_empty_cache() was here to defragment VRAM between chunks,
        // but it internally does a device-wide sync that deadlocks with pending
        // NCCL AllReduce on comm_stream. Removed: the caching allocator handles
        // fragmentation adequately without explicit cache flushes.

        // Update activation peak from the previous chunk's high-water mark.
        // Uses max() so the budget never grows beyond the worst observed peak.
        // Sync compute_stream only (NOT device-wide cuda_synchronize which
        // would block on comm_stream's pending AllReduce -> deadlock).
        if self.device.is_cuda() && self.activation_peak_bytes > 0 {
            let idx = self.device.index() as i32;
            if let Some(ref stream) = self.compute_stream {
                let _ = stream.synchronize();
            }
            if let Ok(peak) = crate::tensor::cuda_peak_active_bytes_idx(idx) {
                if let Ok(baseline) = crate::tensor::cuda_active_bytes_idx(idx) {
                    let overhead = (peak as usize).saturating_sub(baseline as usize);
                    let batch_bytes = self.per_sample_bytes * self.batch_size;
                    let activation = overhead.saturating_sub(batch_bytes);
                    self.activation_peak_bytes = self.activation_peak_bytes.max(activation);
                }
            }
            crate::tensor::cuda_reset_peak_stats_idx(idx);
        }

        // Recalculate prefetch depth at each plan boundary (VRAM may vary).
        // Cap at num_batches: no point buffering more than the chunk contains.
        // Depth 0 means VRAM is too tight for any prefetch buffer.
        //
        // If activation peak hasn't been measured yet, force depth=0 (sync
        // fallback) so the first chunk can calibrate safely.
        let use_prefetch = if let Some(ref mut pw) = self.prefetch {
            if self.activation_peak_bytes == 0 && self.device.is_cuda() {
                pw.set_prefetch_depth(0);
                false
            } else {
                let vram_depth = crate::data::prefetch_depth_from_vram(
                    self.per_sample_bytes, self.batch_size, self.device, 0.90,
                    self.activation_peak_bytes,
                );
                let depth = vram_depth.min(num_batches);
                pw.set_prefetch_depth(depth);
                depth > 0
            }
        } else {
            false
        };

        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::EpochStart { epoch: plan.epoch });
        }
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        // Per-chunk timing accumulators populated by both prefetch and sync
        // paths. Read at chunk end to populate MetricsMsg fields and feed
        // the balancer with an honest tput signal.
        let mut compute_ms_total = 0.0_f64;
        let mut data_starve_ms_total = 0.0_f64;

        if use_prefetch {
            // CUDA async path: prefetch with VRAM gauge.
            let prefetch = self.prefetch.as_ref().unwrap();
            // start_distributed_epoch creates a fresh bounded channel whose
            // capacity equals the prefetch depth (VRAM budget). The prefetch
            // thread fills it; SyncSender blocks when VRAM is full.
            let batch_rx = prefetch.start_distributed_epoch();

            // Submit all batch indices for async H2D transfer
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                prefetch.load_batch(self.partition[start..end].to_vec());
            }

            // Consume prefetched batches as they become ready
            let mut batch_done = 0usize;
            let chunk_diag_start = Instant::now();
            let mut prefetch_wait_diag = std::time::Duration::ZERO;
            let mut compute_ms_diag = 0.0_f64;
            for _ in 0..num_batches {
                // Interleave control message processing with prefetch waiting.
                // SyncNow can arrive at any time; if we block on batch_rx.recv()
                // the peer enters AllReduce waiting for us -> deadlock.
                // Use recv_timeout to periodically check for control messages.
                if self.handle_control()? {
                    return Ok(true);
                }
                let wait_start = Instant::now();
                let prefetched = loop {
                    match batch_rx.recv_timeout(std::time::Duration::from_millis(10)) {
                        Ok(batch) => break batch
                            .map_err(|e| TensorError::new(&format!("prefetch error: {e}")))?,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if self.handle_control()? {
                                return Ok(true);
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return Err(TensorError::new("prefetch channel closed"));
                        }
                    }
                };
                prefetch_wait_diag += wait_start.elapsed();

                // Ensure compute stream waits for async H2D copy to finish
                #[cfg(feature = "cuda")]
                if let Some(ref event) = prefetched.ready_event {
                    if let Some(ref stream) = self.compute_stream {
                        stream.wait_event(event)?;
                    }
                }

                let (loss, ms) = self.train_step(&prefetched.tensors, train_fn)?;
                compute_ms_diag += ms;
                batch_done += 1;
                total_loss += loss;
                let norm = if self.steps_since_avg % 10 == 0 {
                    self.compute_param_norm().ok()
                } else {
                    None
                };
                let _ = self.report_timing(ms, norm, loss, None);
                if self.handle_control()? {
                    return Ok(true); // Shutdown
                }
            }
            let chunk_total_ms = chunk_diag_start.elapsed().as_secs_f64() * 1000.0;
            let prefetch_ms = prefetch_wait_diag.as_secs_f64() * 1000.0;
            let other_ms = chunk_total_ms - prefetch_ms - compute_ms_diag;
            crate::verbose!(
                "  ddp-worker-diag: rank {} chunk={} batches | total={:.0}ms compute={:.0}ms prefetch_wait={:.0}ms other(sync/ctrl)={:.0}ms",
                self.rank, batch_done, chunk_total_ms, compute_ms_diag, prefetch_ms, other_ms,
            );
            compute_ms_total = compute_ms_diag;
            data_starve_ms_total = prefetch_ms;
            crate::debug!("  ddp-worker: rank {} epoch {} chunk done ({} batches)", self.rank, plan.epoch, batch_done);
        } else {
            // Sync path: load one batch at a time, move to device if needed.
            // Used for CPU devices, or CUDA when VRAM is too tight for prefetch.
            let measuring_peak = self.activation_peak_bytes == 0 && self.device.is_cuda();

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let data_start = Instant::now();
                let batch = self.dataset.get_batch(indices)?;

                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch.into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };
                data_starve_ms_total += data_start.elapsed().as_secs_f64() * 1000.0;

                let (loss, ms) = self.train_step(&batch, train_fn)?;
                compute_ms_total += ms;
                total_loss += loss;

                // After first batch: measure activation peak from CUDA stats.
                // The peak includes model + batch + activations + gradients.
                // Subtract baseline (model/optimizer/NCCL) and one batch to
                // isolate the activation + gradient overhead. This is the
                // reserve that prefetch_depth_from_vram must account for.
                if measuring_peak && batch_idx == 0 {
                    if let Some(ref stream) = self.compute_stream {
                        let _ = stream.synchronize();
                    }
                    let idx = self.device.index() as i32;
                    if let Ok(peak) = crate::tensor::cuda_peak_active_bytes_idx(idx) {
                        if let Ok(current) = crate::tensor::cuda_active_bytes_idx(idx) {
                            let overhead = (peak as usize).saturating_sub(current as usize);
                            let batch_bytes = self.per_sample_bytes * self.batch_size;
                            self.activation_peak_bytes = overhead.saturating_sub(batch_bytes);
                        }
                    }
                    // Reset for ongoing monitoring in subsequent chunks.
                    crate::tensor::cuda_reset_peak_stats_idx(idx);
                }

                let norm = if self.steps_since_avg % 10 == 0 {
                    self.compute_param_norm().ok()
                } else {
                    None
                };
                let _ = self.report_timing(ms, norm, loss, None);
                if self.handle_control()? {
                    return Ok(true); // Shutdown
                }
            }
        }

        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
        // Honest balancer denominator: time the rank spent on its assigned
        // work (compute + data wait), excluding any post-completion idle
        // waiting at a sync barrier. epoch_ms includes that idle on the
        // fast rank, which inverts the tput signal the balancer reads.
        // share_complete_ms is computed from the rank's own pipeline times
        // (compute_ms_total + data_starve_ms_total), so it tracks the
        // rank's actual capacity, not how long it idles for peers.
        let share_complete_ms = compute_ms_total + data_starve_ms_total;
        let avg_loss = total_loss / num_batches as f64;
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::EpochEnd {
                epoch: plan.epoch,
                loss: avg_loss,
                lr: self.optimizer.lr(),
            });
        }
        let _ = self.report_epoch(
            avg_loss, num_batches, epoch_ms,
            share_complete_ms, compute_ms_total, data_starve_ms_total,
        );

        Ok(false)
    }
}
