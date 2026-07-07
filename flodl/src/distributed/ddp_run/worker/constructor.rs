//! Worker construction: `channels()` factory + `new()` thread-local builder.

use std::sync::{Arc, Mutex, mpsc};

use crate::autograd::{NoGradGuard, Variable};
use crate::data::BatchDataSet;
use crate::distributed::cuda_event::{CudaEvent, CudaEventFlags};
use crate::distributed::cuda_stream::{CudaStream, StreamGuard};
use crate::distributed::nccl::NcclRankComm;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::super::{
    ApplyPolicy, CheckpointFn, ControlMsg, EvalFn,
    MetricsMsg, ParamSnapshot, TimingMsg, WorkerConfig,
};
use super::{GpuWorker, WorkerChannels, WorkerEndpoints};

impl<M: Module> GpuWorker<M> {
    /// Create the channel pairs for one worker.
    ///
    /// Returns (worker-side senders/receiver, coordinator-side receivers/sender).
    /// Call this on the main thread, then pass the worker-side halves into
    /// [`GpuWorker::new`] inside the spawned thread.
    pub(crate) fn channels() -> (WorkerEndpoints, WorkerChannels) {
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
    pub(crate) fn new<F, G, O>(
        config: &WorkerConfig,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        nccl_comm: Option<NcclRankComm>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
        timing_tx: mpsc::Sender<TimingMsg>,
        metrics_tx: mpsc::Sender<MetricsMsg>,
        param_tx: mpsc::Sender<ParamSnapshot>,
        final_param_tx: mpsc::Sender<ParamSnapshot>,
        control_rx: mpsc::Receiver<ControlMsg>,
        outer_optimizer: Option<Box<dyn crate::distributed::OuterOptimizer>>,
    ) -> Result<Self>
    where
        F: FnOnce(Device) -> Result<M>,
        G: FnOnce(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        // Set the per-thread log prefix so every flodl log line from this
        // worker carries its identity. Thread-based DDP (`Ddp::wrap`: one
        // process hosts every rank) shows `[rN]`. Process-per-rank children
        // are spawned by the cluster launcher, which ALREADY line-prefixes
        // their stdout/stderr with `[host:dev:rN]` (see
        // `launcher::forward_lines`) -- and that wrap also tags raw lines
        // (libtorch warnings, bench `final eval=` / `done:` prints) the
        // in-process logger never sees. Setting the same prefix in-process
        // would double it on flodl-log-macro lines, so skip it there,
        // detected by the launcher's per-rank env marker.
        let local_dev = match config.device {
            Device::CUDA(d) => d,
            _ => 0,
        };
        if std::env::var(crate::distributed::cluster::ENV_LOCAL_RANK).is_err() {
            crate::log::set_thread_device(local_dev, Some(config.rank));
        }

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

        // Allocate scratch buffers for weight-space divergence
        // measurement AND for cluster-mode NCCL abort recovery (the
        // retry path in `sync_now_nccl` restores params from this
        // scratch after a peer-death abort). Allocated whenever an
        // NCCL comm is attached — the divergence value is near-zero in
        // Sync mode but the recovery path needs the buffer regardless,
        // so paying the alloc once is simpler than threading a
        // `cluster_mode` flag through the constructor.
        let pre_sync_scratch = if nccl_comm.is_some() {
            let scratch: Result<Vec<Tensor>> = param_vars.iter()
                .map(|v| Tensor::zeros_like(&v.data()))
                .collect();
            scratch.ok()
        } else {
            None
        };

        // Adopt the model's shared aggregated-metrics slot (Graph
        // exposes one; other Modules default to None and get a
        // private slot). Captured BEFORE the `model` move into
        // `GpuWorker` below so the slot lookup still has a reference
        // to read from.
        let aggregated_slot = model
            .aggregated_metrics_slot()
            .unwrap_or_else(|| Arc::new(Mutex::new(None)));
        Ok(GpuWorker {
            model,
            optimizer: Box::new(optimizer),
            param_vars,
            buffer_list,
            rank: config.rank,
            world_size: config.world_size,
            device: config.device,
            epoch_callback_role: None,
            compute_stream,
            comm_stream,
            copy_done,
            pending_param_h2d: false,
            last_h2d_wait_ms: 0.0,
            last_update_at: None,
            h2d_wait_ms_total: 0.0,
            prof_enabled: crate::log::enabled(crate::log::Verbosity::Debug),
            snapshot_ns_total: 0,
            snapshot_count: 0,
            snapshot_pinned_params: Vec::new(),
            snapshot_pinned_buffers: Vec::new(),
            pinned_fallback_logged: false,
            compute_ms_run_total: 0.0,
            data_ms_run_total: 0.0,
            ctrl_msgs_handled: 0,
            nccl_abort_handle: nccl_comm.as_ref().map(|c| c.abort_handle()),
            nccl_abort_slot: None,
            nccl_comm,
            nccl_session_mailbox: None,
            local_dead_ranks: None,
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
            nccl_sync_seq: 0,
            steps_since_avg: 0,
            steps_at_snapshot: 0,
            current_version: 0,
            current_epoch: 0,
            pending_plan: None,
            global_step: 0,
            scheduler: None,
            lr_scale: 1.0,
            aggregated_metrics: aggregated_slot,
            checkpoint_fn,
            eval_fn,
            eval_dataset,
            save_path: config.save_path.clone(),
            prefetch,
            per_sample_bytes,
            activation_peak_bytes: 0,
            max_grad_norm: config.max_grad_norm,
            // EASGD elastic blending is an Async-only concept: Sync and
            // Cadence MUST full-overwrite to the consensus each window. Gate
            // structurally on the worker's policy here -- the single point
            // every worker (threaded, single-host, cluster) is built through
            // -- so a stray `easgd_alpha` from ANY upstream config path can
            // never blend a non-async worker. The config value alone is too
            // weak a guard for this invariant, and the only honored value is
            // already mode-defaulted to `Some` for CpuAsync.
            easgd_alpha: if matches!(config.policy, ApplyPolicy::Async) {
                config.easgd_alpha
            } else {
                None
            },
            timeline: config.timeline.clone(),
            pre_sync_scratch,
            outer_optimizer,
            outer_prev_global: None,
            _grad_accumulators: grad_accumulators,
        })
    }
}
