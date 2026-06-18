//! Single-host fallback path: no coordinator, no worker threads, no
//! parameter averaging.
//!
//! Used when fewer than 2 CUDA devices are visible. The training loop
//! runs synchronously on the main thread and the same per-epoch metrics
//! / checkpoint / eval callbacks fire as multi-GPU runs, so the
//! [`DdpHandle`] surface is identical for single- and multi-GPU
//! invocations.
//!
//! [`DdpHandle`]: super::DdpHandle

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::ddp_run::worker::GpuWorker;
use crate::distributed::ddp_run::{
    self, ApplyPolicy, CheckpointFn, EpochFn, EvalFn, EvalResultFn, MetricsFn, TrainedState,
    WorkerConfig,
};
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor};

use super::DdpHandle;

impl DdpHandle {
    /// Single-GPU fallback: run training on the main thread.
    ///
    /// No coordinator, no worker threads, no parameter averaging.
    /// Same training loop as multi-GPU workers. Synchronous: returns the
    /// `DdpHandle` only after all epochs complete, with the per-epoch
    /// `EpochMetrics` already queued for [`DdpHandle::next_metrics`] and
    /// the optional `metrics_fn` already fired for each epoch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_single<F, M, G, O, T>(
        model_factory: &F,
        optim_factory: &G,
        train_fn: &T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        device: Device,
        checkpoint_fn: Option<CheckpointFn<M>>,
        checkpoint_every: Option<usize>,
        epoch_fn: Option<EpochFn<M>>,
        metrics_fn: Option<MetricsFn>,
        max_grad_norm: Option<f64>,
        scheduler: Option<Arc<dyn crate::nn::Scheduler>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
        eval_every_epochs: Option<usize>,
        eval_result_fn: Option<EvalResultFn>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable>,
    {
        crate::verbose!("  ddp: single device ({device:?}) | no coordination");

        let total_samples = dataset.len();
        let tmp_model = model_factory(device)?;
        let initial_params: Vec<Tensor> = tmp_model.parameters().iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = tmp_model.buffers().iter()
            .map(|b| b.get())
            .collect();
        let graph_ref = tmp_model.as_graph();
        let architecture_svg = graph_ref
            .and_then(|g| g.svg(None).ok())
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        let graph_label = graph_ref.and_then(|g| g.label().map(|s| s.to_string()));
        let graph_hash = graph_ref.map(|g| g.structural_hash().to_string());
        drop(tmp_model);

        let training_meta = Some(serde_json::json!({
            "gpus": 1,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "mode": "single-gpu fallback",
        }));

        let config = WorkerConfig {
            rank: 0,
            world_size: 1,
            device,
            initial_params,
            initial_buffers,
            total_samples,
            batch_size,
            seed: crate::distributed::ddp_run::SHUFFLE_BASE_SEED,
            max_grad_norm,
            // Single-GPU fallback never goes through the cpu-async load_averaged
            // path, so EASGD alpha is irrelevant here. None keeps the
            // current-behavior copy_ path in case the code path changes.
            easgd_alpha: None,
            // Single-GPU fallback does no averaging; gamma is irrelevant. 1.0
            // (plain work-weighting) is the neutral default.
            gamma: 1.0,
            timeline: None,
            policy: ApplyPolicy::Sync, // single-GPU fallback: no divergence measurement
            save_path: None,
        };

        // Keep the worker channels: `run_epoch_plan` calls `worker.report_epoch`
        // which sends a `MetricsMsg` on metrics_tx. Draining metrics_rx per
        // epoch lets us aggregate into `EpochMetrics`, fire `metrics_fn`, and
        // push to the handle's metrics queue — same surface as multi-GPU.
        let (worker_endpoints, worker_channels) = GpuWorker::<M>::channels();
        let (timing_tx, metrics_tx, param_tx, final_param_tx, control_rx) = worker_endpoints;

        let mut worker = GpuWorker::new(
            &config,
            model_factory,
            optim_factory,
            dataset,
            None, // no NCCL for single-GPU
            checkpoint_fn.clone(),
            eval_fn.clone(),
            eval_dataset.clone(),
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
            control_rx,
        )?;

        // Attach per-batch LR scheduler.
        if let Some(sched) = scheduler {
            worker.set_scheduler(sched);
        }

        // Epoch-metrics channel for the returned DdpHandle, mirroring multi-GPU.
        let (epoch_metrics_tx, epoch_metrics_rx) = std::sync::mpsc::channel::<ddp_run::EpochMetrics>();
        let device_index: u8 = match device {
            Device::CUDA(idx) => idx,
            _ => 0,
        };
        let device_indices = vec![device_index];

        // Train directly on this thread (no coordinator, local epoch management)
        for epoch in 0..num_epochs {
            // Set current_epoch before epoch_fn so
            // worker.current_epoch() is correct inside the callback.
            worker.current_epoch = epoch;
            if let Some(ref f) = epoch_fn {
                f(epoch, &mut worker);
            }
            let plan = ddp_run::EpochPlan {
                epoch,
                partition_offset: 0,
                partition_size: total_samples,
            };
            worker.run_epoch_plan(&plan, train_fn)?;

            // Drain MetricsMsg(s) emitted by report_epoch this iteration.
            // Single-GPU is non-progressive, so exactly one msg per epoch
            // (or zero if num_batches == 0; report_epoch always sends).
            let mut msgs: Vec<ddp_run::MetricsMsg> = Vec::new();
            while let Ok(m) = worker_channels.metrics_rx.try_recv() {
                msgs.push(m);
            }
            if !msgs.is_empty() {
                // Single-GPU fast path: only one rank, so the cadence-share
                // is trivially [1.0]. No balancer involved.
                let bc_share = vec![1.0_f64];
                let metrics = ddp_run::coordinator::aggregate_epoch_metrics(
                    epoch, &msgs, &device_indices, &bc_share,
                );
                if let Some(f) = &metrics_fn {
                    if let Err(e) = f(&metrics) {
                        eprintln!("  ddp: metrics_fn returned error (epoch {epoch}): {e}");
                    }
                }
                let _ = epoch_metrics_tx.send(metrics);
            }

            // Single-GPU checkpoint: version = epoch number (monotonic)
            if let (Some(every), Some(f)) = (checkpoint_every, &checkpoint_fn) {
                if every > 0 && (epoch + 1) % every == 0 {
                    if let Err(e) = f((epoch + 1) as u64, worker.model()) {
                        eprintln!("  ddp: checkpoint failed (epoch {}): {e}", epoch + 1);
                    }
                }
            }

            // Single-GPU eval cadence: mirrors the cluster controller's
            // dispatch. Fire after epoch N at (N+1) % every == 0 so the
            // semantic matches "evaluate the model at end of this epoch".
            // The framework flips train/eval mode; user supplies the
            // batch iteration inside the closure.
            if let (Some(every), Some(efn), Some(ds)) =
                (eval_every_epochs, &eval_fn, &eval_dataset)
            {
                if every > 0 && (epoch + 1) % every == 0 {
                    worker.model().eval();
                    let result = efn(worker.model(), ds.as_ref());
                    worker.model().train();
                    match result {
                        Ok(metric) => {
                            if let Some(rf) = &eval_result_fn {
                                if let Err(e) = rf(epoch + 1, metric) {
                                    eprintln!(
                                        "  ddp: eval_result_fn returned error (epoch {}): {e}",
                                        epoch + 1,
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "  ddp: eval_fn returned error (epoch {}): {e}",
                                epoch + 1,
                            );
                        }
                    }
                }
            }
        }
        // Drop the sender so next_metrics() returns None after the queue drains.
        drop(epoch_metrics_tx);

        // Capture final state before dropping the worker
        let snap = worker.snapshot_params();
        let final_state = TrainedState {
            params: snap.params.iter()
                .map(|t| t.to_device(Device::CPU))
                .collect::<Result<Vec<_>>>()?,
            buffers: snap.buffers.iter()
                .map(|t| t.to_device(Device::CPU))
                .collect::<Result<Vec<_>>>()?,
        };

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(true)),
            nccl_abort_handles: Vec::new(),
            final_state: Some(final_state),
            metrics_rx: Some(epoch_metrics_rx),
            launcher_driver: None,
            architecture_svg,
            graph_label,
            graph_hash,
            training_meta,
        })
    }
}
