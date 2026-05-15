//! DDP run-mode orchestrator: spawns GPU worker threads and a coordinator thread.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::nccl;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::{
    ApplyPolicy, AverageBackend, DdpRunConfig, CheckpointFn, EpochFn,
    TrainedState, TimingMsg, WorkerConfig,
};
use super::worker::GpuWorker;
use super::coordinator::Coordinator;

// ---------------------------------------------------------------------------
// DDP run-mode orchestrator
// ---------------------------------------------------------------------------

/// DDP run-mode handle: spawns GPU worker threads and a coordinator thread.
///
/// Each GPU runs its own training loop with a local optimizer. The coordinator
/// triggers periodic parameter averaging based on [`ApplyPolicy`] and
/// [`AverageBackend`]. Workers self-manage their epochs.
///
/// Use [`Trainer::builder()`](crate::distributed::Trainer::builder) for the full configuration API.
///
/// # Quick start
///
/// ```ignore
/// use flodl::*;
///
/// let handle = Trainer::builder(model_factory, optim_factory, train_fn)
///     .dataset(dataset)
///     .batch_size(32)
///     .num_epochs(10)
///     .run()?;                // non-blocking: spawns threads, returns immediately
///
/// let state = handle.join()?;   // blocks until training completes
/// // state.params / state.buffers contain the averaged trained tensors (CPU)
/// ```
///
/// # Recommended configurations
///
/// **Homogeneous GPUs (same model, same VRAM):**
/// `Sync` + `Nccl`. Equivalent to PyTorch DDP. Simplest, best convergence.
///
/// **Heterogeneous GPUs (mixed generations, different VRAM):**
/// `Cadence` + `Nccl`. ElChe assigns more batches to the fast GPU. Consider
/// [`with_max_batch_diff`](DdpRunConfig::with_max_batch_diff) as a safety guard.
///
/// **Maximum throughput (large models, expensive batches):**
/// `Async` + `Nccl`. Auto-tunes averaging interval. Monitor loss curves.
///
/// **Debugging / no NCCL:**
/// Any policy + `Cpu`. Works everywhere, logs averaging time for comparison.
///
/// # Single-GPU fallback
///
/// With fewer than 2 CUDA devices, training runs on the main thread with no
/// coordinator or averaging. The API is identical; [`join`](Self::join) returns
/// a [`TrainedState`] in both cases.
pub struct DdpHandle {
    worker_handles: Vec<std::thread::JoinHandle<Result<()>>>,
    coordinator_handle: Option<std::thread::JoinHandle<Result<TrainedState>>>,
    devices: Vec<Device>,
    shutdown: Arc<AtomicBool>,
    /// Abort handles for NCCL communicators. Calling abort unblocks any
    /// worker stuck in an NCCL collective (e.g. AllReduce for a dead rank).
    nccl_abort_handles: Vec<Arc<nccl::NcclAbortHandle>>,
    /// For single-GPU mode: final state captured inline during run_single().
    final_state: Option<TrainedState>,
    /// Receiver for aggregated epoch metrics from the coordinator.
    metrics_rx: Option<mpsc::Receiver<super::EpochMetrics>>,
    /// Graph architecture SVG captured from the model (if it implements as_graph).
    architecture_svg: Option<String>,
    /// Graph label (from as_graph().label()).
    graph_label: Option<String>,
    /// Structural hash (from as_graph().structural_hash()).
    graph_hash: Option<String>,
    /// Training config snapshot for monitor metadata.
    training_meta: Option<serde_json::Value>,
}

impl DdpHandle {
    /// Detect GPUs, spawn worker threads and coordinator thread with default config.
    ///
    /// Prefer [`Trainer::builder()`](crate::distributed::Trainer::builder) as the primary entry point.
    #[allow(clippy::too_many_arguments)]
    #[deprecated(since = "0.3.0", note = "Use Trainer::builder() instead")]
    pub fn auto<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        #[allow(deprecated)]
        Self::auto_with(
            model_factory, optim_factory, train_fn,
            dataset, batch_size, num_epochs,
            policy, backend, DdpRunConfig::new(),
        )
    }

    /// Detect GPUs, spawn worker threads and coordinator thread.
    ///
    /// Prefer [`Trainer::builder()`](crate::distributed::Trainer::builder) as the primary entry point.
    #[allow(clippy::too_many_arguments)]
    #[deprecated(since = "0.3.0", note = "Use Trainer::builder() instead")]
    pub fn auto_with<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
        config: DdpRunConfig,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        Self::launch(
            model_factory, optim_factory, train_fn,
            dataset, batch_size, num_epochs,
            policy, backend, config, None, None, None, None, None,
        )
    }

    /// Internal launcher shared by `auto_with` and the builder.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn launch<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
        config: DdpRunConfig,
        checkpoint_fn: Option<CheckpointFn<M>>,
        epoch_fn: Option<EpochFn<M>>,
        metrics_fn: Option<super::MetricsFn>,
        scheduler_fn: Option<Box<dyn Fn(usize) -> Arc<dyn crate::nn::Scheduler> + Send + Sync>>,
        convergence_guard: Option<Box<dyn super::ConvergenceGuard>>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        // Cluster-mode detection: under the process-per-rank model,
        // Trainer::builder runs inside each rank process — one device per
        // process, no in-process N-thread coordinator. Dispatches to the
        // matching cluster-rank inline-loop entry by (policy, backend).
        // Slices land one combo at a time across 4b.D.1a.{ii, iii, iv} +
        // 4b.D.1b; each combo has its own loud-error pointer until lit.
        if let Some(cluster) = crate::distributed::cluster::LocalCluster::from_env()? {
            return match (policy, backend) {
                (ApplyPolicy::Sync, AverageBackend::Nccl) => {
                    Self::run_cluster_rank_sync_nccl(
                        cluster,
                        model_factory,
                        optim_factory,
                        train_fn,
                        dataset,
                        batch_size,
                        num_epochs,
                        config,
                    )
                }
                (ApplyPolicy::Cadence, AverageBackend::Nccl)
                | (ApplyPolicy::Async, AverageBackend::Nccl) => {
                    // Under NCCL backend, Cadence and Async share the same
                    // algorithm: overshoot is the only OLD-coordinator
                    // distinction, and it's an async/CPU concept (no-op for
                    // NCCL). Both policies route through the Cadence helper;
                    // the helper carries policy in WorkerConfig for the
                    // worker's pre_sync_scratch / metadata-emitting paths
                    // that branch on it.
                    Self::run_cluster_rank_cadence_nccl(
                        cluster,
                        policy,
                        model_factory,
                        optim_factory,
                        train_fn,
                        dataset,
                        batch_size,
                        num_epochs,
                        config,
                        convergence_guard,
                    )
                }
                _ => Err(crate::tensor::TensorError::new(&format!(
                    "Trainer::builder cluster mode: ApplyPolicy::Sync (4b.D.1a.ii) \
                     + Cadence / Async (4b.D.1a.iii / iv) on AverageBackend::Nccl \
                     are implemented. Cpu backend (all policies) lands in \
                     4b.D.1b. Requested: {policy:?} + {backend:?}. Use \
                     Trainer::setup / setup_with for cluster-aware training \
                     in the meantime (user owns \
                     the loop).",
                ))),
            };
        }

        let devices = crate::tensor::usable_cuda_devices();

        // Single-GPU fallback: run on main thread, no coordinator.
        if devices.len() < 2 {
            let dev = devices.first().copied().unwrap_or(Device::CPU);
            let scheduler = scheduler_fn.map(|f| f(1));
            return Self::run_single(
                &model_factory, &optim_factory, &train_fn,
                dataset, batch_size, num_epochs, dev,
                checkpoint_fn.as_ref().cloned(),
                config.checkpoint_every,
                epoch_fn,
                metrics_fn,
                config.max_grad_norm,
                scheduler,
            );
        }

        // Print device summary (same style as Trainer::setup)
        Self::print_summary(&devices, &policy, &backend);

        // Step 1: Create temp model on device[0] to extract initial params
        let tmp_model = model_factory(devices[0])?;
        let initial_params: Vec<Tensor> = tmp_model.parameters().iter()
            .map(|p| p.variable.data().to_device(Device::CPU).and_then(|t| t.pin_memory()))
            .collect::<Result<Vec<_>>>()?;
        let initial_buffers: Vec<Tensor> = tmp_model.buffers().iter()
            .map(|b| b.get().to_device(Device::CPU).and_then(|t| t.pin_memory()))
            .collect::<Result<Vec<_>>>()?;
        // Capture graph identity before dropping (for monitor/dashboard)
        let graph_ref = tmp_model.as_graph();
        let architecture_svg = graph_ref
            .and_then(|g| g.svg(None).ok())
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
        let graph_label = graph_ref.and_then(|g| g.label().map(|s| s.to_string()));
        let graph_hash = graph_ref.map(|g| g.structural_hash().to_string());
        drop(tmp_model);

        let world_size = devices.len();
        let total_samples = dataset.len();

        // LR scaling: factor = 1.0 + (world_size - 1) * ratio.
        // Compensates for the LR schedule advancing faster when global_step
        // tracks all GPUs' batches. (Goyal et al., 2017 linear scaling rule.)
        let lr_scale_factor = if world_size > 1 && config.lr_scale_ratio > 0.0 {
            let factor = 1.0 + (world_size as f64 - 1.0) * config.lr_scale_ratio;
            crate::verbose!(
                "  ddp: LR scaled by {factor:.2}x (ratio={:.2}, world_size={world_size}). \
                 Adjust with .lr_scale_ratio()",
                config.lr_scale_ratio,
            );
            factor
        } else {
            1.0
        };

        // Build training config snapshot for monitor metadata
        let progressive = config.progressive_dispatch
            .unwrap_or(!matches!(policy, ApplyPolicy::Sync));
        let training_meta = Some(Self::build_training_meta(
            &devices, &policy, &backend, batch_size, num_epochs,
            total_samples, progressive, &config,
        ));

        // Step 2: Create channels
        let (timing_tx_main, timing_rx) = mpsc::channel();
        let (metrics_tx_main, metrics_rx) = mpsc::channel();
        let (param_tx_main, param_rx) = mpsc::channel();

        let mut coord_control_txs = Vec::new();
        let mut worker_control_rxs = Vec::new();
        let mut worker_final_txs = Vec::new();
        let mut coord_final_rxs = Vec::new();
        for _ in 0..world_size {
            let (tx, rx) = mpsc::channel();
            coord_control_txs.push(tx);
            worker_control_rxs.push(rx);
            let (ftx, frx) = mpsc::channel();
            worker_final_txs.push(ftx);
            coord_final_rxs.push(frx);
        }

        // Step 2b: Init NCCL comms from main thread, then split into per-rank comms.
        // CRITICAL: ncclCommInitRank from worker threads corrupts CUDA context on
        // heterogeneous GPUs. Always use NcclComms::new() + split() instead.
        // See NcclRankComm and NcclComms::split docs for details.
        let (mut rank_comms, nccl_abort_handles): (Vec<Option<_>>, Vec<_>) =
            if backend == AverageBackend::Nccl {
                let group = nccl::NcclComms::new(&devices)?;
                let comms = group.split()?;
                let aborts = comms.iter().map(|c| c.abort_handle()).collect();
                (comms.into_iter().map(Some).collect(), aborts)
            } else {
                ((0..world_size).map(|_| None).collect(), Vec::new())
            };

        // Step 3: Create ElChe with config knobs
        let anchor = config.anchor.unwrap_or(10);
        let mut el_che = crate::distributed::ddp::ElChe::new(world_size, anchor);
        if let Some(target) = config.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }
        if let Some(min) = config.min_anchor {
            el_che = el_che.with_min_anchor(min);
        }
        if let Some(diff) = config.max_batch_diff {
            el_che = el_che.with_max_batch_diff(diff);
        }

        // Cold-start anchor: precedence is partition_ratios > spec prior >
        // rank-0 fallback. When the user supplied per-rank ratios, the
        // smallest ratio is the slow rank by user assertion. Otherwise we
        // ask the GPUs themselves (compute capability + VRAM) which one is
        // most likely the slowest. Either way, the pick is "soft" — once
        // timing data accumulates, election may move the anchor.
        if let Some(ratios) = config.partition_ratios.as_ref() {
            if ratios.len() == world_size {
                if let Some((slow_rank, _)) = ratios
                    .iter()
                    .enumerate()
                    .min_by(|(ra, a), (rb, b)| {
                        a.partial_cmp(b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(ra.cmp(rb))
                    })
                {
                    el_che = el_che.with_initial_anchor(slow_rank);
                }
            }
        } else {
            let cuda_indices: Vec<i32> = devices.iter().filter_map(|d| match d {
                Device::CUDA(idx) => Some(*idx as i32),
                _ => None,
            }).collect();
            if cuda_indices.len() == world_size {
                el_che = el_che.with_device_indices(&cuda_indices);
            }
        }

        // Step 3b: Create epoch metrics channel (coordinator -> main thread)
        let (epoch_metrics_tx, epoch_metrics_rx) = mpsc::channel();

        // Device indices for coordinator GPU metrics
        let coord_device_indices: Vec<u8> = devices.iter().map(|d| match d {
            Device::CUDA(idx) => *idx,
            _ => 0,
        }).collect();

        // Step 4: Spawn coordinator thread
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_coord = shutdown.clone();
        let div_threshold = config.divergence_threshold;
        let no_div_guard = config.no_divergence_guard;
        let ckpt_every = config.checkpoint_every;
        let snap_timeout = config.snapshot_timeout_secs;
        let partition_ratios = config.partition_ratios.clone();
        let max_grad_norm = config.max_grad_norm;
        let timeline = config.timeline.clone();
        let coord_timeline = timeline.clone();
        let coord_batch_size = batch_size;
        let seed: u64 = 42;

        let coordinator_handle = std::thread::Builder::new()
            .name("ddp-coordinator".into())
            .spawn(move || -> Result<TrainedState> {
                let mut builder = Coordinator::builder(
                    timing_rx, metrics_rx, param_rx,
                    coord_final_rxs,
                    coord_control_txs,
                    policy, backend,
                    world_size, total_samples, el_che,
                )
                .snapshot_timeout_secs(snap_timeout)
                .epoch_metrics_tx(epoch_metrics_tx)
                .device_indices(coord_device_indices)
                .num_epochs(num_epochs)
                .partition_ratios(partition_ratios)
                .progressive(progressive)
                .batch_size(coord_batch_size)
                .timeline(coord_timeline.clone())
                .max_overshoot(config.max_overshoot)
                .elche_relax_up(config.elche_relax_up)
                .meta_controller(config.meta_controller);
                if let Some(mf) = metrics_fn {
                    builder = builder.metrics_fn(mf);
                }
                // Pluggable guard takes precedence; legacy
                // divergence_threshold/no_divergence_guard still flow into
                // the default TrendGuard configuration when no explicit
                // guard is supplied.
                if let Some(g) = convergence_guard {
                    builder = builder.convergence_guard(g);
                } else {
                    if let Some(dt) = div_threshold {
                        builder = builder.divergence_threshold(dt);
                    }
                    if no_div_guard {
                        builder = builder.no_divergence_guard();
                    }
                }
                if let Some(n) = ckpt_every {
                    builder = builder.checkpoint_every(n);
                }
                let mut coord = builder.build();

                // Send first epoch plans to all workers.
                // Uses speed_hint partition sizes if available.
                coord.send_all_plans(0);

                let poll_timeout = std::time::Duration::from_micros(100);
                let mut loop_tick: u64 = 0;
                let mut last_state_dump = std::time::Instant::now();
                let loop_err = loop {
                    loop_tick += 1;
                    if shutdown_coord.load(Ordering::Relaxed) {
                        crate::verbose!("  ddp: coordinator exit: shutdown flag set (worker error?)");
                        break None;
                    }
                    if !coord.drain_timing_blocking(poll_timeout) {
                        crate::verbose!("  ddp: coordinator exit: all timing channels disconnected");
                        break None;
                    }
                    if coord.active_count == 0 {
                        crate::verbose!("  ddp: coordinator exit: all workers exited");
                        break None;
                    }
                    // on_epoch_aggregated sends Shutdown when last epoch completes.
                    // Workers exit, channels disconnect, drain_timing_blocking returns false.
                    if coord.all_epochs_done() {
                        break None;
                    }

                    // Periodic state dump (every 2s) for deadlock diagnosis.
                    if last_state_dump.elapsed().as_secs() >= 2 {
                        last_state_dump = std::time::Instant::now();
                        coord.debug_state_dump(loop_tick);
                    }

                    coord.check_throttle();
                    if let Err(e) = coord.poll_cpu_averaging() {
                        shutdown_coord.store(true, Ordering::Relaxed);
                        break Some(e);
                    }
                    // drain_metrics -> on_rank_done (Auto per-rank dispatch)
                    //               -> try_aggregate_epochs -> on_epoch_aggregated
                    //                  (Sync/Cadence broadcast or Auto unblock)
                    for m in coord.drain_metrics() {
                        crate::verbose!(
                            "  ddp: rank {} epoch {} | loss={:.4} batches={} time={:.0}ms",
                            m.rank, m.epoch, m.avg_loss, m.batches_processed, m.epoch_ms
                        );
                    }
                    if coord.should_average() {
                        coord.drain_timing();
                        if coord.should_average() {
                            if let Err(e) = coord.trigger_averaging() {
                                shutdown_coord.store(true, Ordering::Relaxed);
                                break Some(e);
                            }
                        }
                    }
                };

                // Ensure any in-progress CPU averaging is fully cleaned up
                // before we return. This joins the compute thread (if any)
                // so no detached thread holds GPU resources that could
                // interfere with subsequent NCCL init.
                coord.drain_avg_state();

                // Tell workers to exit their drain_until_shutdown loop.
                // Workers that finished training are blocked on recv()
                // waiting for this signal before dropping their NcclRankComm.
                coord.shutdown_workers();

                // collect_final_state uses recv_timeout (blocking) so
                // no sleep is needed: it waits for each worker's snapshot.
                match coord.collect_final_state() {
                    Some(state) => Ok(state),
                    None => match loop_err {
                        Some(e) => Err(e),
                        None => Err(TensorError::new(
                            "coordinator: no final snapshots received from workers"
                        )),
                    },
                }
            })
            .map_err(|e| TensorError::new(&format!("failed to spawn coordinator: {e}")))?;

        // Step 5: Spawn GPU worker threads
        let scheduler = scheduler_fn.map(|f| f(world_size));
        let model_factory = Arc::new(model_factory);
        let optim_factory = Arc::new(optim_factory);
        let train_fn = Arc::new(train_fn);
        // checkpoint_fn is already Arc, clone for each worker
        let mut worker_handles = Vec::new();

        for (rank, control_rx) in worker_control_rxs.into_iter().enumerate() {
            let device = devices[rank];
            let mf = model_factory.clone();
            let of = optim_factory.clone();
            let tf = train_fn.clone();
            let ds = dataset.clone();
            let params = initial_params.clone();
            let buffers = initial_buffers.clone();
            let t_tx = timing_tx_main.clone();
            let t_tx_err = timing_tx_main.clone();
            let m_tx = metrics_tx_main.clone();
            let p_tx = param_tx_main.clone();
            let fp_tx = worker_final_txs.remove(0);
            let ckpt_fn = checkpoint_fn.clone();
            let epoch_fn_w = epoch_fn.clone();
            let scheduler_w = scheduler.clone();
            let shutdown_w = shutdown.clone();

            let worker_nccl = rank_comms[rank].take();
            let worker_tl = timeline.clone();
            let lr_scale = lr_scale_factor;
            let config = WorkerConfig {
                rank,
                world_size,
                device,
                initial_params: params,
                initial_buffers: buffers,
                total_samples,
                batch_size,
                seed,
                max_grad_norm,
                easgd_alpha: config.easgd_alpha,
                timeline: worker_tl,
                policy,
            };

            let handle = std::thread::Builder::new()
                .name(format!("ddp-gpu-{rank}"))
                .spawn(move || {
                    // Set CUDA device for this thread
                    if let Device::CUDA(idx) = device {
                        crate::tensor::set_current_cuda_device(idx);
                    }

                    // Inner closure so we can always run cleanup on exit.
                    let result = (|| -> Result<()> {
                        // Build worker inside this thread (model + optimizer are
                        // Rc-based, thread-local). NCCL comm was pre-initialized
                        // on the main thread via NcclComms::split() to avoid
                        // per-thread ncclCommInitRank CUDA context corruption.
                        let mut worker = GpuWorker::new(
                            &config,
                            |dev| (*mf)(dev),
                            |params| (*of)(params),
                            ds,
                            worker_nccl,
                            ckpt_fn,
                            t_tx,
                            m_tx,
                            p_tx,
                            fp_tx,
                            control_rx,
                        )?;

                        // Apply linear LR scaling for DDP.
                        //
                        // With a scheduler attached, we store the factor so
                        // it can be applied multiplicatively to the
                        // scheduler's output every batch. Without a
                        // scheduler, we scale the optimizer's LR once at
                        // startup and leave it alone.
                        if lr_scale > 1.0 {
                            if scheduler_w.is_some() {
                                worker.set_lr_scale(lr_scale);
                            } else {
                                worker.scale_lr(lr_scale);
                            }
                        }

                        // Attach per-batch LR scheduler (global step tracking).
                        if let Some(ref sched) = scheduler_w {
                            worker.set_scheduler(Arc::clone(sched));
                        }

                        // Training loop: coordinator-driven epochs.
                        // Workers are mode-agnostic: they wait for a plan,
                        // fire epoch_fn, process the partition, and report.
                        // In progressive mode, multiple plans may arrive for
                        // the same epoch (chunks); the epoch_fn guard ensures
                        // it only fires once per epoch transition.
                        worker.current_epoch = usize::MAX; // sentinel for first epoch_fn
                        loop {
                            if shutdown_w.load(Ordering::Relaxed) {
                                break;
                            }
                            let plan = match worker.wait_for_epoch_plan()? {
                                Some(p) => p,
                                None => break, // Shutdown or disconnect
                            };
                            // Only fire epoch_fn on epoch transitions (not per-chunk).
                            // The usize::MAX sentinel ensures epoch 0 triggers it.
                            if plan.epoch != worker.current_epoch {
                                worker.current_epoch = plan.epoch;
                                if let Some(ref f) = epoch_fn_w {
                                    f(plan.epoch, &mut worker);
                                }
                            }
                            if worker.run_epoch_plan(&plan, &*tf)? {
                                break; // Shutdown received mid-epoch
                            }
                        }

                        // Abort NCCL comm before snapshot: a pending AllReduce
                        // from a SyncNow whose peer died would block to_device(CPU)
                        // because the CUDA default stream waits for all streams.
                        worker.abort_nccl();

                        // Send final snapshot on the dedicated channel before exiting.
                        // Uses final_param_tx (not param_tx) to avoid racing with
                        // CPU averaging snapshot collection.
                        worker.send_final_snapshot();
                        worker.report_exiting();

                        // Handle remaining control messages until Shutdown.
                        // SyncNow is skipped (NCCL comm already aborted).
                        worker.drain_until_shutdown();
                        Ok(())
                    })();

                    if let Err(ref e) = result {
                        eprintln!("  ddp: worker {rank} error: {e}");
                        // Ensure coordinator knows this rank is gone even on
                        // error (prevents NCCL deadlock on surviving workers).
                        // Send directly on the raw channel since the worker
                        // may not exist (e.g. GpuWorker::new failed).
                        let _ = t_tx_err.send(TimingMsg::Exiting { rank });
                        // Signal all siblings to stop so they don't block in
                        // an NCCL collective waiting for this dead rank.
                        shutdown_w.store(true, Ordering::Relaxed);
                    }

                    result
                })
                .map_err(|e| TensorError::new(&format!("failed to spawn worker {rank}: {e}")))?;

            worker_handles.push(handle);
        }

        // Drop the main thread's clones so coordinator sees channel disconnect
        // when all workers are done
        drop(timing_tx_main);
        drop(metrics_tx_main);
        drop(param_tx_main);

        Ok(DdpHandle {
            worker_handles,
            coordinator_handle: Some(coordinator_handle),
            devices: devices.to_vec(),
            shutdown,
            nccl_abort_handles,
            final_state: None,
            metrics_rx: Some(epoch_metrics_rx),
            architecture_svg,
            graph_label,
            graph_hash,
            training_meta,
        })
    }

    /// Single-GPU fallback: run training on the main thread.
    ///
    /// No coordinator, no worker threads, no parameter averaging.
    /// Same training loop as multi-GPU workers. Synchronous: returns the
    /// `DdpHandle` only after all epochs complete, with the per-epoch
    /// `EpochMetrics` already queued for [`DdpHandle::next_metrics`] and
    /// the optional `metrics_fn` already fired for each epoch.
    #[allow(clippy::too_many_arguments)]
    fn run_single<F, M, G, O, T>(
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
        metrics_fn: Option<super::MetricsFn>,
        max_grad_norm: Option<f64>,
        scheduler: Option<Arc<dyn crate::nn::Scheduler>>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable>,
    {
        use std::sync::atomic::AtomicBool;

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
            seed: 42,
            max_grad_norm,
            // Single-GPU fallback never goes through the cpu-async load_averaged
            // path, so EASGD alpha is irrelevant here. None keeps the
            // current-behavior copy_ path in case the code path changes.
            easgd_alpha: None,
            timeline: None,
            policy: ApplyPolicy::Sync, // single-GPU fallback: no divergence measurement
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
        let (epoch_metrics_tx, epoch_metrics_rx) = std::sync::mpsc::channel::<super::EpochMetrics>();
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
            let plan = super::EpochPlan {
                epoch,
                partition_offset: 0,
                partition_size: total_samples,
            };
            worker.run_epoch_plan(&plan, train_fn)?;

            // Drain MetricsMsg(s) emitted by report_epoch this iteration.
            // Single-GPU is non-progressive, so exactly one msg per epoch
            // (or zero if num_batches == 0; report_epoch always sends).
            let mut msgs: Vec<super::MetricsMsg> = Vec::new();
            while let Ok(m) = worker_channels.metrics_rx.try_recv() {
                msgs.push(m);
            }
            if !msgs.is_empty() {
                // Single-GPU fast path: only one rank, so the cadence-share
                // is trivially [1.0]. No balancer involved.
                let bc_share = vec![1.0_f64];
                let metrics = super::coordinator::aggregate_epoch_metrics(
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
            architecture_svg,
            graph_label,
            graph_hash,
            training_meta,
        })
    }

    /// Cluster-rank entry point for `ApplyPolicy::Sync + AverageBackend::Nccl`.
    ///
    /// This rank reads its slot from [`LocalCluster::from_env`], does the
    /// NCCL rendezvous, builds the model on its assigned device, syncs
    /// initial parameters from rank 0 via NCCL broadcast, and spawns a
    /// single training thread that runs
    /// [`GpuWorker::run_self_driven_sync_nccl`] — no in-process N-thread
    /// coordinator, no mpsc orchestration.
    ///
    /// **Behavior preserved (vs the old threaded coordinator):**
    /// - Same `train_step` (forward + backward + optional grad clipping +
    ///   scheduler + optimizer step + CUDA stream pinning)
    /// - Same `sync_now_nccl` (in-place AllReduce + divergence measurement)
    /// - Same dataset shuffle (deterministic global permutation via
    ///   [`make_partition`])
    /// - Same initial-state sync (rank 0 broadcasts params + buffers to
    ///   every rank)
    ///
    /// **Not yet wired in this slice (4b.D.1a.ii):** progressive chunk
    /// dispatch (sync data loading only), per-epoch metrics aggregation
    /// (`metrics_rx` is `None`), `epoch_fn` / `metrics_fn` / `scheduler` /
    /// `checkpoint_every` callbacks (silently ignored — future slices wire
    /// them back in one at a time). The compile gate on the public
    /// `DdpBuilder` API hides this gap: callers can still chain the
    /// fluent methods; they just produce no-ops until the supporting
    /// machinery lands.
    ///
    /// [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env
    /// [`GpuWorker::run_self_driven_sync_nccl`]: crate::distributed::ddp_run::GpuWorker::run_self_driven_sync_nccl
    /// [`make_partition`]: crate::distributed::ddp_run::make_partition
    #[allow(clippy::too_many_arguments)]
    fn run_cluster_rank_sync_nccl<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        config: DdpRunConfig,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::AtomicBool;
        use crate::distributed::nccl::NcclRankComm;

        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        let total_samples = dataset.len();

        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} (Sync+Nccl)"
        );

        let training_meta = Some(serde_json::json!({
            "mode": "cluster-rank Sync+Nccl",
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
        }));

        // All model/optimizer/CUDA work happens inside the spawned
        // training thread — M (containing Rc<RefCell<…>>) isn't Send, so
        // we can't build a GpuWorker on the main thread and move it in.
        // The thread does: rendezvous → init NcclRankComm → build model
        // → broadcast initial state from rank 0 → build GpuWorker → run
        // self-driven Sync loop → snapshot final state.
        //
        // dataset_signature isn't yet plumbed through DdpBuilder /
        // DdpRunConfig (it lives on DdpConfig for Trainer::setup_with).
        // Default [0u8; 32] — every rank trivially agrees; cluster
        // rendezvous accepts. Future slice exposes it on the builder
        // for opt-in shard-divergence detection.
        let dataset_sig = [0u8; 32];
        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;

        let coordinator_handle = std::thread::spawn(move || -> Result<TrainedState> {
            let rdv = cluster.rendezvous(dataset_sig)?;
            let nccl_comm = NcclRankComm::init_rank(global_rank, world_size, rdv.unique_id())?;

            // Build tmp model, broadcast initial state from rank 0, then
            // pin to CPU for the WorkerConfig (GpuWorker::new re-creates
            // the model and copies the pinned params back to GPU).
            let tmp_model = model_factory(device)?;
            let initial_params_gpu: Vec<Tensor> = tmp_model
                .parameters()
                .iter()
                .map(|p| p.variable.data())
                .collect();
            let initial_buffers_gpu: Vec<Tensor> = tmp_model
                .buffers()
                .iter()
                .map(|b| b.get())
                .collect();
            if !initial_params_gpu.is_empty() {
                let refs: Vec<&Tensor> = initial_params_gpu.iter().collect();
                nccl_comm.broadcast(&refs, 0)?;
            }
            if !initial_buffers_gpu.is_empty() {
                let refs: Vec<&Tensor> = initial_buffers_gpu.iter().collect();
                nccl_comm.broadcast(&refs, 0)?;
            }
            let initial_params: Vec<Tensor> = initial_params_gpu
                .iter()
                .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
                .collect::<Result<Vec<_>>>()?;
            let initial_buffers: Vec<Tensor> = initial_buffers_gpu
                .iter()
                .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
                .collect::<Result<Vec<_>>>()?;
            drop(tmp_model);

            let worker_config = WorkerConfig {
                rank: global_rank,
                world_size,
                device,
                initial_params,
                initial_buffers,
                total_samples,
                batch_size,
                seed: 42,
                max_grad_norm,
                easgd_alpha: None,
                timeline: timeline_for_thread,
                policy: ApplyPolicy::Sync,
            };

            // Worker channels: nothing drains them in cluster-rank mode
            // (no coordinator). Held inside the closure so the worker's
            // internal sends silently buffer. Future slice wires
            // metrics_rx back into DdpHandle.
            let (worker_endpoints, _worker_channels) = GpuWorker::<M>::channels();
            let (timing_tx, metrics_tx, param_tx, final_param_tx, control_rx) =
                worker_endpoints;

            let mut worker = GpuWorker::new(
                &worker_config,
                model_factory,
                optim_factory,
                dataset,
                Some(nccl_comm),
                None,
                timing_tx,
                metrics_tx,
                param_tx,
                final_param_tx,
                control_rx,
            )?;

            worker.run_self_driven_sync_nccl(num_epochs, &train_fn)?;
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
        });

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: Some(coordinator_handle),
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
            nccl_abort_handles: Vec::new(), // abort plumbing deferred — handle
                                            // lives inside the thread's NcclRankComm
            final_state: None,
            metrics_rx: None, // wired in a follow-up slice
            architecture_svg: None, // model lives inside thread; metadata deferred
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }

    /// Cluster-rank entry point for `ApplyPolicy::Cadence` /
    /// `ApplyPolicy::Async` + `AverageBackend::Nccl`.
    ///
    /// Under NCCL backend, Cadence and Async share the same algorithm:
    /// overshoot machinery (the only OLD-coordinator distinction) is an
    /// async/CPU concept (`feedback_overshoot_async_only` /
    /// `feedback_nccl_no_overshoot_throttle`) — irrelevant for NCCL. Both
    /// policies route through this helper; [`WorkerConfig::policy`]
    /// carries the policy enum for the worker's per-policy bookkeeping.
    ///
    /// Drives [`GpuWorker::run_self_driven_cadence_nccl`] which runs:
    /// - Per-batch `train_step` (Local SGD: each rank advances independently)
    /// - At ElChe-driven K boundary: cross-rank timing AllReduce →
    ///   parameter AllReduce-Avg with weight-space divergence → cross-rank
    ///   AllReduce-gather of `(divergence, pre_norm)` →
    ///   [`ElChe::report_timing`] → `convergence_guard.report(...)` →
    ///   [`ConvergenceAction`] applied to ElChe
    ///   ([`nudge_anchor_down`](super::super::ddp::ElChe::nudge_anchor_down)
    ///   on NudgeDown, [`relax_anchor_up`](super::super::ddp::ElChe::relax_anchor_up)
    ///   on Stable+flag).
    ///
    /// **Comm ownership:** the [`NcclRankComm`] is built inside the spawned
    /// thread (rendezvous lives inside, since `M` isn't `Send`) and handed
    /// to [`Ddp::from_comm`] after the initial-state broadcast. The
    /// [`GpuWorker`] is constructed with `nccl_comm = None`; the cadence
    /// loop drives all NCCL collectives through the `Ddp` handle
    /// (`average_params_with_divergence`, `all_reduce_per_rank_f64`).
    ///
    /// **ElChe construction** mirrors the orchestrator's main path
    /// (`anchor` / `max_anchor` / `min_anchor` / `overhead_target` /
    /// `max_batch_diff` from [`DdpRunConfig`]; cold-start anchor pick).
    ///
    /// **ConvergenceGuard construction** mirrors the OLD
    /// `Coordinator::builder` recipe: user-supplied override > NoGuard
    /// when `no_divergence_guard` set > `TrendGuard::new(divergence_threshold
    /// or 0.05)`. Each rank builds its own guard from the same scalars,
    /// so they stay in lockstep across ranks (deterministic input →
    /// deterministic verdict).
    ///
    /// **Still deferred (carried forward through 4b.D.1b):** progressive
    /// chunk dispatch, per-epoch metrics aggregation, `epoch_fn` /
    /// `metrics_fn` / `scheduler_fn` / `checkpoint_every` callbacks,
    /// LR-aware meta-controller (needs scheduler_fn flow), Timeline
    /// events for `Divergence` / `SyncEnd` / `AnchorChanged` /
    /// `GuardTelemetry`.
    ///
    /// [`GpuWorker::run_self_driven_cadence_nccl`]:
    ///     crate::distributed::ddp_run::GpuWorker::run_self_driven_cadence_nccl
    /// [`Ddp::from_comm`]: crate::distributed::Ddp::from_comm
    /// [`ElChe::report_timing`]: crate::distributed::ElChe::report_timing
    /// [`ConvergenceAction`]: super::convergence::ConvergenceAction
    #[allow(clippy::too_many_arguments)]
    fn run_cluster_rank_cadence_nccl<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        policy: ApplyPolicy,
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        config: DdpRunConfig,
        convergence_guard: Option<Box<dyn super::convergence::ConvergenceGuard>>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::AtomicBool;
        use crate::distributed::nccl::NcclRankComm;
        use crate::distributed::ddp::{Ddp, ElChe};

        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        let total_samples = dataset.len();

        let policy_label = match policy {
            ApplyPolicy::Sync => "Sync",
            ApplyPolicy::Cadence => "Cadence",
            ApplyPolicy::Async => "Async",
        };
        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} \
             ({policy_label}+Nccl)"
        );

        let training_meta = Some(serde_json::json!({
            "mode": format!("cluster-rank {policy_label}+Nccl"),
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
        }));

        // ElChe construction snapshot — mirrors orchestrator main path's
        // Step 3 (`crate::distributed::ddp::ElChe::new(world_size, anchor)`
        // + with_overhead_target / with_max_anchor / with_min_anchor /
        // with_max_batch_diff / cold-start pick). Hoisted out of the
        // thread closure so the inputs are Send; the ElChe is rebuilt
        // inside the thread (ElChe is `Clone`-free, simpler to recompute
        // than to plumb).
        let anchor = config.anchor.unwrap_or(10);
        let overhead_target = config.overhead_target;
        let max_anchor = config.max_anchor;
        let min_anchor = config.min_anchor;
        let max_batch_diff = config.max_batch_diff;
        let partition_ratios = config.partition_ratios.clone();
        let elche_relax_up = config.elche_relax_up;
        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;

        // ConvergenceGuard recipe (mirrors Coordinator::builder default):
        // user override wins, then NoGuard when no_divergence_guard set,
        // else TrendGuard::new(divergence_threshold or 0.05).
        let divergence_threshold = config.divergence_threshold.unwrap_or(0.05);
        let no_divergence_guard = config.no_divergence_guard;

        // CUDA device indices for ElChe cold-start prior (slow-rank pick by
        // compute capability + VRAM). Empty when this rank isn't on CUDA;
        // ElChe falls back to rank 0 as the slow rank in that case.
        let cuda_idx_for_thread: Option<i32> = match device {
            Device::CUDA(idx) => Some(idx as i32),
            _ => None,
        };

        let dataset_sig = [0u8; 32];

        let coordinator_handle = std::thread::spawn(move || -> Result<TrainedState> {
            let rdv = cluster.rendezvous(dataset_sig)?;
            let nccl_comm = NcclRankComm::init_rank(global_rank, world_size, rdv.unique_id())?;

            // Build tmp model, broadcast initial state from rank 0 via the
            // raw comm (Ddp::sync_params would do this but Ddp's params
            // borrow into model fields; doing it before Ddp::from_comm is
            // simpler and avoids re-grabbing tensor handles).
            let tmp_model = model_factory(device)?;
            let initial_params_gpu: Vec<Tensor> = tmp_model
                .parameters()
                .iter()
                .map(|p| p.variable.data())
                .collect();
            let initial_buffers_gpu: Vec<Tensor> = tmp_model
                .buffers()
                .iter()
                .map(|b| b.get())
                .collect();
            if !initial_params_gpu.is_empty() {
                let refs: Vec<&Tensor> = initial_params_gpu.iter().collect();
                nccl_comm.broadcast(&refs, 0)?;
            }
            if !initial_buffers_gpu.is_empty() {
                let refs: Vec<&Tensor> = initial_buffers_gpu.iter().collect();
                nccl_comm.broadcast(&refs, 0)?;
            }
            let initial_params: Vec<Tensor> = initial_params_gpu
                .iter()
                .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
                .collect::<Result<Vec<_>>>()?;
            let initial_buffers: Vec<Tensor> = initial_buffers_gpu
                .iter()
                .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
                .collect::<Result<Vec<_>>>()?;

            // Hand comm ownership to Ddp; the cadence loop drives all
            // collectives through `ddp` (average_params,
            // all_reduce_per_rank_f64). Borrows tmp_model only for the
            // parameter/buffer-list extraction; tmp_model is dropped right
            // after so the rebuilt model inside GpuWorker owns the live
            // Variables.
            let ddp = Ddp::from_comm(nccl_comm, &tmp_model, device)?;
            drop(tmp_model);

            // Build ElChe per the orchestrator main-path recipe. Cold-start
            // anchor pick: partition_ratios (smallest = slow) > device-
            // indices prior. Every rank sees the same inputs, so every
            // ElChe instance starts identically.
            let mut el_che = ElChe::new(world_size, anchor);
            if let Some(target) = overhead_target {
                el_che = el_che.with_overhead_target(target);
            }
            if let Some(max) = max_anchor {
                el_che = el_che.with_max_anchor(max);
            }
            if let Some(min) = min_anchor {
                el_che = el_che.with_min_anchor(min);
            }
            if let Some(diff) = max_batch_diff {
                el_che = el_che.with_max_batch_diff(diff);
            }
            if let Some(ratios) = partition_ratios.as_ref() {
                if ratios.len() == world_size {
                    if let Some((slow_rank, _)) = ratios
                        .iter()
                        .enumerate()
                        .min_by(|(ra, a), (rb, b)| {
                            a.partial_cmp(b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(ra.cmp(rb))
                        })
                    {
                        el_che = el_che.with_initial_anchor(slow_rank);
                    }
                }
            } else if let Some(idx) = cuda_idx_for_thread {
                // Single-rank device-indices prior: every rank only knows
                // its own device. The orchestrator main path gathered all
                // ranks' devices before calling with_device_indices, but
                // in cluster-rank mode we'd need a cross-rank AllReduce to
                // reproduce that — deferred. Passing the single-element
                // slice is a no-op for the prior (with_device_indices
                // requires len == world_size).
                let _ = idx; // intentionally unused: see comment above
            }

            // Partition sizes per orchestrator main-path policy: explicit
            // partition_ratios > throughput_sizes if calibrated > equal.
            // First-epoch ElChe is uncalibrated, so this defaults to
            // equal_sizes — matches old coordinator behavior (Cadence
            // recomputes per epoch from current ElChe state, but the
            // inline loop currently uses a single partition for all
            // epochs; per-epoch rebalance is deferred along with the
            // metrics/callback wiring).
            let partition_sizes: Vec<usize> = if let Some(ratios) = partition_ratios.as_ref() {
                super::coordinator::ratio_to_sizes(ratios, total_samples)
            } else if el_che.is_calibrated() || el_che.has_speed_hint() {
                super::coordinator::throughput_sizes(&el_che, total_samples)
            } else {
                super::coordinator::equal_sizes(world_size, total_samples)
            };

            let worker_config = WorkerConfig {
                rank: global_rank,
                world_size,
                device,
                initial_params,
                initial_buffers,
                total_samples,
                batch_size,
                seed: 42,
                max_grad_norm,
                easgd_alpha: None,
                timeline: timeline_for_thread,
                policy,
            };

            // Worker channels: nothing drains them in cluster-rank mode.
            // Held inside the closure so the worker's internal sends
            // silently buffer until the channel is dropped.
            let (worker_endpoints, _worker_channels) = GpuWorker::<M>::channels();
            let (timing_tx, metrics_tx, param_tx, final_param_tx, control_rx) =
                worker_endpoints;

            // GpuWorker built with `nccl_comm = None`: this slice routes
            // every NCCL op through `ddp` above (including the divergence
            // measurement). Symmetric with the Sync slice's choice.
            let mut worker = GpuWorker::new(
                &worker_config,
                model_factory,
                optim_factory,
                dataset,
                None,
                None,
                timing_tx,
                metrics_tx,
                param_tx,
                final_param_tx,
                control_rx,
            )?;

            // Build the per-rank ConvergenceGuard. User override consumed
            // here (each rank-process has its own DdpBuilder + config, so
            // the user's Box moves into this rank's closure cleanly).
            let mut guard: Box<dyn super::convergence::ConvergenceGuard> =
                match convergence_guard {
                    Some(g) => g,
                    None => {
                        if no_divergence_guard {
                            Box::new(super::convergence::NoGuard)
                        } else {
                            Box::new(super::convergence::TrendGuard::new(
                                divergence_threshold,
                            ))
                        }
                    }
                };

            // Pre-sync scratch allocated once for the full training session.
            let scratch = ddp.make_divergence_scratch()?;

            worker.run_self_driven_cadence_nccl(
                &ddp,
                &mut el_che,
                guard.as_mut(),
                &scratch,
                &partition_sizes,
                elche_relax_up,
                num_epochs,
                &train_fn,
            )?;
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
        });

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: Some(coordinator_handle),
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
            nccl_abort_handles: Vec::new(),
            final_state: None,
            metrics_rx: None,
            architecture_svg: None,
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }

    /// Number of GPUs in this DDP group.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Devices in use.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Graph architecture SVG, if the model implements [`Module::as_graph()`].
    ///
    /// Captured automatically from the model factory at launch time.
    /// Pass to [`Monitor::set_svg()`](crate::monitor::Monitor::set_svg) to
    /// display the graph in the dashboard:
    ///
    /// ```ignore
    /// if let Some(svg) = handle.architecture_svg() {
    ///     monitor.set_svg(svg);
    /// }
    /// ```
    pub fn architecture_svg(&self) -> Option<&str> {
        self.architecture_svg.as_deref()
    }

    /// Wire this handle's graph identity, architecture SVG, and training
    /// config into a [`Monitor`](crate::monitor::Monitor).
    ///
    /// Call once after [`run()`](super::DdpBuilder::run), before the metrics
    /// loop. This is the DDP equivalent of calling `monitor.watch(&graph)`
    /// in single-GPU training.
    ///
    /// ```ignore
    /// let handle = Trainer::builder(factory, optim, train_fn)
    ///     .dataset(ds).batch_size(32).num_epochs(10)
    ///     .run()?;
    /// handle.setup_monitor(&mut monitor);
    ///
    /// while let Some(m) = handle.next_metrics() {
    ///     monitor.log(m.epoch, Duration::from_millis(m.epoch_ms as u64), &m);
    /// }
    /// monitor.finish();
    /// ```
    pub fn setup_monitor(&self, monitor: &mut crate::monitor::Monitor) {
        if let Some(svg) = &self.architecture_svg {
            monitor.set_svg(svg);
        }
        monitor.set_identity(
            self.graph_label.as_deref(),
            self.graph_hash.as_deref(),
        );
        if let Some(meta) = &self.training_meta {
            monitor.set_metadata(meta.clone());
        }
    }

    /// Non-blocking: drain all available aggregated epoch metrics.
    ///
    /// Returns an empty `Vec` when nothing is currently queued. In multi-GPU
    /// mode, metrics arrive per epoch as the coordinator aggregates ranks.
    /// In single-GPU mode, `run_single` is synchronous, so by the time
    /// callers can poll, all per-epoch metrics are already in the queue
    /// (and `metrics_fn`, if registered, has already fired for each).
    pub fn poll_metrics(&self) -> Vec<super::EpochMetrics> {
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

    /// Blocking: wait for the next epoch's aggregated metrics.
    ///
    /// Returns `None` when training ends (sender dropped). In multi-GPU
    /// mode this blocks per epoch as the coordinator aggregates ranks; in
    /// the single-GPU fallback the queue is fully populated by the time
    /// `run()` returns, so calls return queued metrics non-blocking, then
    /// `None`.
    pub fn next_metrics(&self) -> Option<super::EpochMetrics> {
        self.metrics_rx.as_ref().and_then(|rx| rx.recv().ok())
    }

    /// Abort all NCCL communicators, unblocking any stuck collective ops.
    ///
    /// Called on error/shutdown to ensure no worker thread hangs forever
    /// in an AllReduce waiting for a dead rank.
    fn abort_nccl(&self) {
        for h in &self.nccl_abort_handles {
            let _ = h.abort();
        }
    }

    /// Wait for all training to complete and return the trained state.
    ///
    /// Workers run their `num_epochs` and exit naturally. Each sends a final
    /// parameter snapshot before terminating. The coordinator collects and
    /// averages these into a [`TrainedState`] (CPU tensors).
    ///
    /// For single-GPU mode, the state was captured inline during training.
    ///
    /// On partial failure (some workers died), returns the average of
    /// surviving workers' final snapshots. Returns an error only if
    /// all workers failed.
    pub fn join(mut self) -> Result<TrainedState> {
        // Single-GPU: state was captured in run_single()
        if let Some(state) = self.final_state.take() {
            return Ok(state);
        }

        // Join ALL workers, even if some fail. A failed worker already
        // set shutdown=true (see the error path in the spawn closure),
        // but we set it again on first error to cover panics.
        let mut first_err: Option<TensorError> = None;
        let handles: Vec<_> = self.worker_handles.drain(..).collect();

        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.shutdown.store(true, Ordering::Relaxed);
                    self.abort_nccl();
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    self.shutdown.store(true, Ordering::Relaxed);
                    self.abort_nccl();
                    if first_err.is_none() {
                        first_err = Some(TensorError::new("worker thread panicked"));
                    }
                }
            }
        }

        // All workers done (or failed). Shut down coordinator.
        self.shutdown.store(true, Ordering::Relaxed);

        if let Some(h) = self.coordinator_handle.take() {
            match h.join() {
                Ok(Ok(state)) => {
                    // Coordinator succeeded, but if a worker errored, warn.
                    // Return the state (partial training is still useful) but
                    // log the worker error so it's not silently swallowed.
                    if let Some(ref e) = first_err {
                        eprintln!("  ddp: WARNING: training state recovered but worker error occurred: {e}");
                    }
                    return Ok(state);
                }
                Ok(Err(e)) if first_err.is_none() => first_err = Some(e),
                Err(_) if first_err.is_none() => {
                    first_err = Some(TensorError::new("coordinator thread panicked"));
                }
                _ => {}
            }
        }

        Err(first_err.unwrap_or_else(|| TensorError::new("join: no trained state available")))
    }

    /// Print device summary to stderr (same style as Trainer::setup).
    fn print_summary(devices: &[Device], policy: &ApplyPolicy, backend: &AverageBackend) {
        use crate::tensor::{cuda_device_name_idx, cuda_memory_info_idx};
        use crate::monitor::format_bytes;

        let mut parts = Vec::with_capacity(devices.len());
        let mut names = Vec::with_capacity(devices.len());

        for &dev in devices {
            if let Device::CUDA(idx) = dev {
                let raw_name = cuda_device_name_idx(idx as i32)
                    .unwrap_or_else(|| format!("CUDA({})", idx));
                let short = raw_name
                    .strip_prefix("NVIDIA ")
                    .unwrap_or(&raw_name)
                    .to_string();
                let vram = cuda_memory_info_idx(idx as i32)
                    .ok()
                    .map(|(_, total)| format!(" ({})", format_bytes(total)))
                    .unwrap_or_default();
                parts.push(format!("{}{}", short, vram));
                names.push(raw_name);
            }
        }

        let heterogeneous = names.windows(2).any(|w| w[0] != w[1]);
        let mode = if heterogeneous { "heterogeneous" } else { "homogeneous" };
        let policy_str = match policy {
            ApplyPolicy::Sync => "sync",
            ApplyPolicy::Cadence => "cadence",
            ApplyPolicy::Async => "async",
        };
        let backend_str = match backend {
            AverageBackend::Nccl => "nccl",
            AverageBackend::Cpu => "cpu",
        };

        crate::verbose!(
            "  ddp: {} GPUs ({}) | {} | policy={} backend={}",
            devices.len(), mode, parts.join(" | "), policy_str, backend_str,
        );
    }

    /// Build a training config snapshot as JSON for monitor metadata.
    #[allow(clippy::too_many_arguments)]
    fn build_training_meta(
        devices: &[Device],
        policy: &ApplyPolicy,
        backend: &AverageBackend,
        batch_size: usize,
        num_epochs: usize,
        total_samples: usize,
        progressive: bool,
        config: &DdpRunConfig,
    ) -> serde_json::Value {
        use crate::tensor::cuda_device_name_idx;

        let gpu_names: Vec<String> = devices.iter().map(|d| {
            if let Device::CUDA(idx) = d {
                cuda_device_name_idx(*idx as i32)
                    .unwrap_or_else(|| format!("CUDA({})", idx))
            } else {
                format!("{d:?}")
            }
        }).collect();

        let policy_str = match policy {
            ApplyPolicy::Sync => "sync",
            ApplyPolicy::Cadence => "cadence",
            ApplyPolicy::Async => "async",
        };
        let backend_str = match backend {
            AverageBackend::Nccl => "nccl",
            AverageBackend::Cpu => "cpu",
        };

        let mut meta = serde_json::json!({
            "gpus": devices.len(),
            "gpu_names": gpu_names,
            "policy": policy_str,
            "backend": backend_str,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "progressive_dispatch": progressive,
        });

        if let Some(anchor) = config.anchor {
            meta["anchor"] = serde_json::json!(anchor);
        }
        if let Some(target) = config.overhead_target {
            meta["overhead_target"] = serde_json::json!(target);
        }
        if let Some(max) = config.max_anchor {
            meta["max_anchor"] = serde_json::json!(max);
        }
        if let Some(min) = config.min_anchor {
            meta["min_anchor"] = serde_json::json!(min);
        }
        if let Some(diff) = config.max_batch_diff {
            meta["max_batch_diff"] = serde_json::json!(diff);
        }
        if let Some(overshoot) = config.max_overshoot {
            meta["max_overshoot"] = serde_json::json!(overshoot);
        }
        if let Some(dt) = config.divergence_threshold {
            meta["divergence_threshold"] = serde_json::json!(dt);
        }

        meta
    }
}

// ---------------------------------------------------------------------------
// DdpBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and launching framework-managed DDP training.
///
/// Created via [`Trainer::builder()`](crate::distributed::Trainer::builder). Required fields must be set before
/// calling [`run`](Self::run); missing fields produce a clear panic message.
///
/// # Example
///
/// ```ignore
/// use flodl::*;
///
/// let handle = Trainer::builder(
///     |dev| model_factory(dev),
///     |params| Adam::new(params, 0.001),
///     |model, batch| { /* return loss Variable */ },
/// )
/// .dataset(dataset)
/// .batch_size(32)
/// .num_epochs(10)
/// .policy(ApplyPolicy::Cadence)
/// .backend(AverageBackend::Nccl)
/// .run()?;
///
/// let state = handle.join()?; // blocks until training completes
/// ```
#[allow(clippy::type_complexity)]
pub struct DdpBuilder<F, M, G, O, T>
where
    F: Fn(Device) -> Result<M> + Send + Sync + 'static,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
    O: Optimizer + 'static,
    T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
{
    model_factory: F,
    optim_factory: G,
    train_fn: T,
    dataset: Option<Arc<dyn BatchDataSet>>,
    batch_size: Option<usize>,
    num_epochs: Option<usize>,
    policy: ApplyPolicy,
    backend: AverageBackend,
    config: DdpRunConfig,
    checkpoint_fn: Option<CheckpointFn<M>>,
    epoch_fn: Option<EpochFn<M>>,
    metrics_fn: Option<super::MetricsFn>,
    /// Factory receives `world_size`, returns the scheduler.
    scheduler_fn: Option<Box<dyn Fn(usize) -> Arc<dyn crate::nn::Scheduler> + Send + Sync>>,
    /// Pluggable convergence guard. When set, takes precedence over the
    /// legacy `divergence_threshold` / `no_divergence_guard` fields on
    /// [`DdpRunConfig`]. Boxed because trait-object guards aren't `Clone`.
    convergence_guard: Option<Box<dyn super::ConvergenceGuard>>,
    _phantom: PhantomData<(M, O)>,
}

impl<F, M, G, O, T> DdpBuilder<F, M, G, O, T>
where
    F: Fn(Device) -> Result<M> + Send + Sync + 'static,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
    O: Optimizer + 'static,
    T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
{
    /// Set the training dataset (required).
    pub fn dataset(mut self, dataset: Arc<dyn BatchDataSet>) -> Self {
        self.dataset = Some(dataset);
        self
    }

    /// Set the batch size (required).
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Set the number of epochs (required).
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = Some(n);
        self
    }

    /// Set the averaging policy. Default: [`ApplyPolicy::Cadence`].
    pub fn policy(mut self, policy: ApplyPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the averaging backend. Default: [`AverageBackend::Nccl`].
    pub fn backend(mut self, backend: AverageBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.config = self.config.with_overhead_target(target);
        self
    }

    /// Set the maximum anchor count.
    pub fn max_anchor(mut self, max: usize) -> Self {
        self.config = self.config.with_max_anchor(max);
        self
    }

    /// Set the minimum anchor count (auto-tune floor).
    ///
    /// Combined with `max_anchor(min)` (same value) plus
    /// [`Self::convergence_guard`] = `NoGuard` and
    /// [`Self::no_divergence_guard`], pins the anchor at a fixed cadence.
    pub fn min_anchor(mut self, min: usize) -> Self {
        self.config = self.config.with_min_anchor(min);
        self
    }

    /// Set the initial anchor count.
    pub fn anchor(mut self, anchor: usize) -> Self {
        self.config = self.config.with_anchor(anchor);
        self
    }

    /// Set the divergence threshold for the trend guardrail.
    pub fn divergence_threshold(mut self, threshold: f64) -> Self {
        self.config = self.config.with_divergence_threshold(threshold);
        self
    }

    /// Disable the divergence guardrail. ElChe's overhead auto-tune
    /// handles cadence alone. Use when you know your workload is stable.
    ///
    /// Equivalent to `.convergence_guard(NoGuard)` but kept for backward
    /// compatibility with the older boolean-flag API.
    pub fn no_divergence_guard(mut self) -> Self {
        self.config = self.config.with_no_divergence_guard();
        self
    }

    /// Install a custom convergence guard.
    ///
    /// When set, takes precedence over the legacy `divergence_threshold` and
    /// `no_divergence_guard` settings. Three concrete impls ship in flodl:
    /// [`super::NoGuard`], [`super::TrendGuard`] (production default), and
    /// [`super::MsfGuard`] (rate-based detector with soft+hard thresholds).
    ///
    /// ```text
    /// .convergence_guard(
    ///     MsfGuard::default()
    ///         .with_suppress(1e-3, 3)
    ///         .with_nudge(1e-2, 3, 0.5),
    /// )
    /// ```
    pub fn convergence_guard<C>(mut self, guard: C) -> Self
    where
        C: super::ConvergenceGuard + 'static,
    {
        self.convergence_guard = Some(Box::new(guard));
        self
    }

    /// Set the maximum batch lead of fastest over slowest worker.
    /// `0` = strict lockstep.
    pub fn max_batch_diff(mut self, max: usize) -> Self {
        self.config = self.config.with_max_batch_diff(max);
        self
    }

    /// Set explicit per-rank partition ratios, e.g. `[0.55, 0.225, 0.225]`
    /// for a fast rank plus two slower ranks.
    ///
    /// **Static fixed splits — does not auto-rebalance.** When set, the
    /// coordinator dispatches each epoch's batches in proportion to these
    /// ratios and skips ElChe's throughput-based rebalancer. Length must
    /// match the auto-detected `world_size` and values must sum to ~1.0.
    ///
    /// **Currently honored in `Sync` policy only.** The `Cadence` and
    /// `Async` policies use progressive dispatch driven by ElChe; they
    /// do not consult `partition_ratios`. For dynamic heterogeneous
    /// scheduling under those policies, ElChe's auto-calibration is
    /// the intended path (see `speed_hint` for an initial seed).
    pub fn partition_ratios(mut self, ratios: &[f64]) -> Self {
        self.config = self.config.with_partition_ratios(ratios);
        self
    }

    /// Set the maximum overshoot past the planned sync point.
    ///
    /// Controls how far a fast GPU can stream past its planned batch count
    /// into the next epoch's data. Default: auto-tuned from convergence.
    pub fn max_overshoot(mut self, max: usize) -> Self {
        self.config = self.config.with_max_overshoot(max);
        self
    }

    /// Allow or suppress ElChe's anchor relax-up on stable convergence.
    ///
    /// Default: `false` (opt-in). When `true`, each `Stable` convergence
    /// verdict triggers `el_che.relax_anchor_up()` to grow the anchor toward
    /// `max_anchor`. Opt in when measuring the relax-up regime; the default
    /// keeps the anchor under overhead-based auto-tune alone.
    pub fn elche_relax_up(mut self, enabled: bool) -> Self {
        self.config = self.config.with_elche_relax_up(enabled);
        self
    }

    /// Enable EASGD elastic averaging on the cpu-async path with weight α.
    /// `α` must be in `(0, 1]`. See [`DdpRunConfig::with_easgd_alpha`].
    pub fn easgd_alpha(mut self, alpha: f64) -> Self {
        self.config = self.config.with_easgd_alpha(alpha);
        self
    }

    /// Enable the LR-aware meta-controller above ElChe. Default: `false`.
    ///
    /// When enabled, the coordinator constructs a
    /// [`crate::distributed::lr_event_meta::LrEventMeta`] that observes the
    /// LR trajectory, anchor trend, and convergence guard verdicts each
    /// averaging cycle. Sharp LR drops or sustained divergence patterns
    /// trigger reactive `nudge_anchor_down` calls; ElChe's overhead
    /// auto-tune handles recovery. Off by default until validation sweep.
    pub fn meta_controller(mut self, enabled: bool) -> Self {
        self.config = self.config.with_meta_controller(enabled);
        self
    }

    /// Save a checkpoint every N global epochs.
    pub fn checkpoint_every(mut self, n: usize) -> Self {
        self.config = self.config.with_checkpoint_every(n);
        self
    }

    /// Enable or disable progressive chunk dispatch.
    ///
    /// When enabled, the coordinator streams work in small chunks instead of
    /// sending full epoch partitions, adapting to throughput continuously.
    /// Default: auto (true for Cadence/Async, false for Sync).
    pub fn progressive_dispatch(mut self, enabled: bool) -> Self {
        self.config = self.config.with_progressive_dispatch(enabled);
        self
    }

    /// Set maximum gradient norm for per-worker clipping.
    ///
    /// Each worker clips accumulated gradients (L2 norm) after backward
    /// and before the optimizer step. Same knob as `DdpConfig::max_grad_norm`
    /// for the setup/El Che path.
    pub fn max_grad_norm(mut self, max_norm: f64) -> Self {
        self.config = self.config.with_max_grad_norm(max_norm);
        self
    }

    /// Attach a high-frequency system timeline for profiling DDP behavior.
    ///
    /// The coordinator and workers inject training events (sync, epoch,
    /// anchor changes, throttle) into the timeline for post-run analysis.
    pub fn timeline(mut self, tl: std::sync::Arc<crate::monitor::Timeline>) -> Self {
        self.config = self.config.with_timeline(tl);
        self
    }

    /// Set the LR scaling ratio for multi-GPU training.
    ///
    /// Formula: `lr_factor = 1.0 + (world_size - 1) * ratio`.
    ///
    /// - `1.0` (default): full linear scaling (Goyal et al., 2017).
    /// - `0.0`: no scaling.
    /// - `0.5`: half linear scaling.
    pub fn lr_scale_ratio(mut self, ratio: f64) -> Self {
        self.config = self.config.with_lr_scale_ratio(ratio);
        self
    }

    /// Set the checkpoint function called on rank 0 after averaging.
    ///
    /// Receives `(version, &model)`. Errors are logged but do not stop training.
    ///
    /// For Graph models: `.checkpoint_fn(|ver, g| g.save_checkpoint(&format!("ckpt_v{ver}.fdl")))`
    pub fn checkpoint_fn<C>(mut self, f: C) -> Self
    where
        C: Fn(u64, &M) -> Result<()> + Send + Sync + 'static,
    {
        self.checkpoint_fn = Some(Arc::new(f));
        self
    }

    /// Attach a per-batch LR scheduler factory.
    ///
    /// The factory receives `world_size` so user-defined schedulers can
    /// account for multi-GPU training (e.g. scale warmup duration).
    ///
    /// Each worker adjusts its optimizer's LR before every `optimizer.step()`:
    ///
    /// ```text
    /// lr = scheduler.lr(global_step + steps_since_last_sync)
    /// ```
    ///
    /// At each sync point, the coordinator broadcasts the updated global step
    /// so all workers track the same schedule.
    pub fn scheduler<S>(mut self, factory: S) -> Self
    where
        S: Fn(usize) -> Arc<dyn crate::nn::Scheduler> + Send + Sync + 'static,
    {
        self.scheduler_fn = Some(Box::new(factory));
        self
    }

    /// Set an epoch callback called at the start of each epoch inside each worker thread.
    ///
    /// Receives `(epoch, &mut GpuWorker<M>)`. Runs before [`run_epoch_plan`](GpuWorker::run_epoch_plan),
    /// so [`current_epoch()`](GpuWorker::current_epoch) is already correct.
    ///
    /// Typical uses: noise curricula, dynamic loss weights.
    /// For LR scheduling, prefer [`.scheduler()`](Self::scheduler) which
    /// provides per-batch granularity with global step tracking.
    ///
    /// ```text
    /// .epoch_fn(move |epoch, worker| {
    ///     // custom per-epoch logic
    /// })
    /// ```
    pub fn epoch_fn<E>(mut self, f: E) -> Self
    where
        E: Fn(usize, &mut GpuWorker<M>) + Send + Sync + 'static,
    {
        self.epoch_fn = Some(Arc::new(f));
        self
    }

    /// Set a host-side per-epoch metrics callback.
    ///
    /// Called once per epoch with the aggregated [`super::EpochMetrics`]:
    /// on the coordinator thread for multi-GPU, on the main thread for the
    /// single-GPU fallback. Errors are logged to stderr; training continues.
    /// The same metric is also pushed to the [`DdpHandle::next_metrics`]
    /// queue, so this composes with explicit polling rather than replacing it.
    ///
    /// Use this to keep the chained `Trainer::builder(...).run()?.join()?`
    /// shape observable without a manual polling loop:
    ///
    /// ```ignore
    /// Trainer::builder(model_factory, optim_factory, train_step)
    ///     .dataset(dataset).batch_size(32).num_epochs(N)
    ///     .metrics_fn(move |m| {
    ///         println!("epoch {}: loss={:.4}", m.epoch, m.avg_loss);
    ///         Ok(())
    ///     })
    ///     .run()?
    ///     .join()?;
    /// ```
    ///
    /// Works identically on 1-or-N GPUs: the single-GPU fallback path is
    /// synchronous, so `next_metrics()` returns all queued metrics
    /// back-to-back after `run()` returns; the callback itself fires
    /// per-epoch as training progresses, the same as multi-GPU.
    pub fn metrics_fn<E>(mut self, f: E) -> Self
    where
        E: Fn(&super::EpochMetrics) -> Result<()> + Send + Sync + 'static,
    {
        self.metrics_fn = Some(Arc::new(f));
        self
    }

    /// Launch training. Non-blocking: spawns threads and returns immediately.
    ///
    /// Call [`DdpHandle::join`] to block until training completes and retrieve
    /// the trained parameters and buffers.
    ///
    /// # Panics
    ///
    /// Panics if `dataset`, `batch_size`, or `num_epochs` were not set.
    pub fn run(self) -> Result<DdpHandle> {
        let dataset = self.dataset.expect("DdpBuilder: dataset is required");
        let batch_size = self.batch_size.expect("DdpBuilder: batch_size is required");
        let num_epochs = self.num_epochs.expect("DdpBuilder: num_epochs is required");

        DdpHandle::launch(
            self.model_factory,
            self.optim_factory,
            self.train_fn,
            dataset,
            batch_size,
            num_epochs,
            self.policy,
            self.backend,
            self.config,
            self.checkpoint_fn,
            self.epoch_fn,
            self.metrics_fn,
            self.scheduler_fn,
            self.convergence_guard,
        )
    }
}

impl DdpHandle {
    /// Create a builder for configuring framework-managed DDP training.
    ///
    /// Prefer [`Trainer::builder()`](crate::distributed::Trainer::builder) as the primary entry point.
    /// This method exists for backward compatibility.
    #[deprecated(since = "0.3.0", note = "Use Trainer::builder() instead")]
    pub fn builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        Self::new_builder(model_factory, optim_factory, train_fn)
    }

    /// Internal builder constructor, called by [`Trainer::builder()`](crate::distributed::Trainer::builder).
    pub(crate) fn new_builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        DdpBuilder {
            model_factory,
            optim_factory,
            train_fn,
            dataset: None,
            batch_size: None,
            num_epochs: None,
            policy: ApplyPolicy::Cadence,
            backend: AverageBackend::Nccl,
            config: DdpRunConfig::new(),
            checkpoint_fn: None,
            epoch_fn: None,
            metrics_fn: None,
            scheduler_fn: None,
            convergence_guard: None,
            _phantom: PhantomData,
        }
    }
}

impl Drop for DdpHandle {
    fn drop(&mut self) {
        // Signal shutdown if not already joined
        self.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
        // Abort NCCL comms so workers stuck in collectives can unblock.
        self.abort_nccl();
    }
}
