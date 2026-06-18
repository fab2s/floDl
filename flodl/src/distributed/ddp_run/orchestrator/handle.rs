//! `DdpHandle`: handle returned by `Trainer::builder().run()` / `Trainer::setup()`.
//!
//! Owns the worker join handles, the coordinator thread, and the shutdown
//! signal. Provides `join`, `poll_metrics`, `next_metrics`, plus the
//! `launch` constructor called from the builder.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::nccl;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use crate::distributed::ddp_run::{
    ApplyPolicy, AverageBackend, CheckpointFn, ConvergenceGuard, DdpRunConfig, EpochFn,
    EpochMetrics, EvalFn, EvalResultFn, MetricsFn, SchedulerFn, TimingMsg, TrainedState,
    WorkerConfig,
};
use crate::distributed::ddp_run::coordinator::Coordinator;
use crate::distributed::ddp_run::worker::GpuWorker;
use super::coord_config::build_coord_config_from_builder;

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
    pub(super) worker_handles: Vec<std::thread::JoinHandle<Result<()>>>,
    pub(super) coordinator_handle: Option<std::thread::JoinHandle<Result<TrainedState>>>,
    pub(super) devices: Vec<Device>,
    pub(super) shutdown: Arc<AtomicBool>,
    /// Abort handles for NCCL communicators. Calling abort unblocks any
    /// worker stuck in an NCCL collective (e.g. AllReduce for a dead rank).
    pub(super) nccl_abort_handles: Vec<Arc<nccl::NcclAbortHandle>>,
    /// For single-GPU mode: final state captured inline during run_single().
    pub(super) final_state: Option<TrainedState>,
    /// Receiver for aggregated epoch metrics from the coordinator.
    pub(super) metrics_rx: Option<mpsc::Receiver<EpochMetrics>>,
    /// Launcher driver thread (cluster mode, Role::Launcher). The
    /// trampoline spawns `run_launcher_with_config` here instead of
    /// running it inline + exiting; [`Self::join`] awaits it so user
    /// code can poll [`Self::next_metrics`] between `run()` and
    /// `join()`. `None` outside launcher mode.
    pub(super) launcher_driver: Option<std::thread::JoinHandle<Result<()>>>,
    /// Graph architecture SVG captured from the model (if it implements as_graph).
    pub(super) architecture_svg: Option<String>,
    /// Graph label (from as_graph().label()).
    pub(super) graph_label: Option<String>,
    /// Structural hash (from as_graph().structural_hash()).
    pub(super) graph_hash: Option<String>,
    /// Training config snapshot for monitor metadata.
    pub(super) training_meta: Option<serde_json::Value>,
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
            None, None, None, None,
        )
    }

    /// Internal launcher shared by `auto_with` and the builder.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn launch<F, M, G, O, T>(
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
        metrics_fn: Option<MetricsFn>,
        scheduler_fn: Option<SchedulerFn>,
        convergence_guard: Option<Box<dyn ConvergenceGuard>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
        eval_result_fn: Option<EvalResultFn>,
        outer_optimizer_factory: Option<
            crate::distributed::outer_optimizer::OuterOptimizerFactory,
        >,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        // Auto-promote: when this process would otherwise enter
        // `Role::SingleDevice` (no cluster envelope in env) but
        // 2+ visible GPUs are present, synthesize a localhost cluster
        // and set `FLODL_FULL_CLUSTER_JSON` so the dispatch below
        // returns `Role::Launcher`. This makes `Trainer::run` /
        // `Trainer::builder().run()` "just work" for single-host
        // multi-GPU without any cluster yml or programmatic config —
        // matching the UX of the legacy in-process threaded path
        // while running on the canonical process-per-rank path.
        //
        // Skips auto-promote when:
        //   - Any cluster envelope env var is already set (user opted
        //     in via fdl-cli overlay or `cfg.cluster`).
        //   - `detect_gpus()` returns <2 (no DDP to do; single-device
        //     path will run).
        //   - Compiled with `cfg(test)` (flodl's own test builds; tests
        //     that exercise multi-GPU should use `Ddp::wrap` for the
        //     thread-based path. External crates depending on flodl
        //     always see auto-promote in production builds, including
        //     `cargo run` and release binaries.)
        //
        // `detect_gpus()` respects `CUDA_VISIBLE_DEVICES`, so production
        // callers that want to scope down also have that lever.
        #[cfg(not(test))]
        {
            use crate::distributed::launcher::ENV_FULL_CLUSTER_JSON;
            use crate::distributed::cluster::{ENV_CLUSTER_JSON, ENV_LOCAL_RANK};
            let env_pristine = std::env::var_os(ENV_FULL_CLUSTER_JSON).is_none()
                && std::env::var_os(ENV_CLUSTER_JSON).is_none()
                && std::env::var_os(ENV_LOCAL_RANK).is_none();
            if env_pristine {
                let gpus = crate::sys::detect_gpus();
                if gpus.len() >= 2 {
                    match crate::distributed::ClusterBuilder::all_local_gpus() {
                        Ok(full) => {
                            let hex = crate::distributed::cluster::hex_encode(
                                full.to_json().to_string().as_bytes(),
                            );
                            // SAFETY: DdpHandle::launch is called from
                            // main() before any user-spawned threads;
                            // matches the invariant documented for
                            // fdl-cli's `prepare_cluster_env`.
                            unsafe {
                                std::env::set_var(ENV_FULL_CLUSTER_JSON, hex);
                            }
                        }
                        Err(e) => {
                            // `all_local_gpus` errors when no GPUs are
                            // visible — but we just checked >=2. The
                            // only realistic failure is `hostname(1)`
                            // failing; surface it loudly rather than
                            // silently falling back to single-device.
                            return Err(crate::tensor::TensorError::new(&format!(
                                "auto-promote multi-GPU failed: {e}"
                            )));
                        }
                    }
                }
            }
        }

        // Launcher trampoline. In launcher mode this process is the
        // fan-out orchestrator — no training body to run here. Build
        // the controller-scope config from the user's `DdpRunConfig`
        // and `convergence_guard` (native trait object, same process),
        // hand it to `run_launcher_with_config`, exit when ranks
        // finish. The `Box<dyn ConvergenceGuard>` flows straight into
        // the spawned controller thread on the same host — the
        // cross-process gap that previously forced env-var workarounds
        // is dissolved.
        match crate::distributed::launcher::dispatch()? {
            crate::distributed::launcher::Role::Launcher => {
                let full = crate::distributed::launcher::FullCluster::from_env()?;
                let world_size = full.world_size();
                // Sink for aggregated EpochMetrics. The coord pushes
                // each completed epoch's metrics here; the user's
                // `DdpHandle::next_metrics()` polls them off. Wired
                // alongside `metrics_fn` (both fire on aggregation).
                let (sink_tx, sink_rx) =
                    mpsc::channel::<EpochMetrics>();
                let mut coord_config = build_coord_config_from_builder(
                    policy,
                    backend,
                    &config,
                    convergence_guard,
                    metrics_fn,
                    eval_result_fn,
                    world_size,
                    dataset.len(),
                    batch_size,
                    num_epochs,
                )?;
                coord_config = coord_config.metrics_sink_tx(sink_tx);
                // Capture the static model schema (param/buffer names) for the
                // controller-side consensus-checkpoint writer. Built on CPU in
                // the launcher process — reads names only, touches no CUDA
                // context (honors the "no CUDA before training" launcher
                // invariant). Best-effort: a factory that fails here just
                // leaves the schema unset (consensus checkpoints degrade to
                // meta-only); it does not abort the launch.
                match model_factory(Device::CPU) {
                    Ok(probe) => {
                        coord_config = coord_config.model_schema(
                            crate::distributed::ModelSchema::from_module(&probe),
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "cluster launcher: model schema capture failed \
                             (consensus checkpoints will be meta-only): {e}"
                        );
                    }
                }
                // Spawn the launcher driver on a dedicated thread.
                // Previously this called `run_launcher_with_config`
                // inline then `process::exit(0)` — user's main() never
                // returned past `.run()`. Returning a `DdpHandle` here
                // lets user code poll metrics + call `.join()` to
                // await completion.
                // Build the controller's outer optimizer from the factory
                // (CPU backend: the step runs once at the controller in this
                // launcher process). `None` leaves the reduce stream
                // untouched. Constructed here, off any CUDA context, so it
                // honors the "no CUDA before training" launcher invariant.
                let outer_optimizer =
                    outer_optimizer_factory.as_ref().map(|f| f());
                let driver = std::thread::Builder::new()
                    .name("flodl-launcher-driver".to_string())
                    .spawn(move || {
                        crate::distributed::launcher::run_launcher_with_config(
                            full,
                            Some(coord_config),
                            outer_optimizer,
                        )
                    })
                    .map_err(|e| {
                        crate::tensor::TensorError::new(&format!(
                            "spawn launcher driver thread: {e}"
                        ))
                    })?;
                return Ok(DdpHandle {
                    worker_handles: Vec::new(),
                    coordinator_handle: None,
                    devices: Vec::new(),
                    shutdown: Arc::new(AtomicBool::new(false)),
                    nccl_abort_handles: Vec::new(),
                    final_state: None,
                    metrics_rx: Some(sink_rx),
                    launcher_driver: Some(driver),
                    architecture_svg: None,
                    graph_label: None,
                    graph_hash: None,
                    training_meta: None,
                });
            }
            crate::distributed::launcher::Role::Relay => {
                // This process is a per-host transport relay (spawned by
                // the launcher). Run the byte-router until its local ranks
                // finish, then exit — it never trains.
                match crate::distributed::launcher::run_relay() {
                    Ok(()) => std::process::exit(0),
                    Err(e) => {
                        eprintln!("cluster relay: {e}");
                        std::process::exit(1);
                    }
                }
            }
            crate::distributed::launcher::Role::Rank
            | crate::distributed::launcher::Role::SingleDevice => {}
        }

        // Cluster-mode detection: under the process-per-rank model,
        // Trainer::builder runs inside each rank process — one device per
        // process, no in-process N-thread coordinator. Dispatches to the
        // matching cluster-rank entry by (policy, backend). All entries
        // route through `ClusterCoordinator` (singleton-ElChe-on-controller).
        // `save_path` is optional: when set, the cluster persists a
        // bundle on unrecoverable failure; when unset, the run executes
        // normally and just skips save activity (legitimate for tests
        // and inference-style usage).
        //
        // Pre-rendezvous failures (envelope parse, GPU placement, NCCL
        // init before coordinator handshake) are fatal at the rank-process
        // level: the rank has no path to recovery because
        // `ClusterCoordinator` only sees ranks that registered, and a
        // `Result::Err` returned to the user binary risks being swallowed
        // as a per-combo failure (e.g. ddp-bench's `run_combo` Err arm
        // prints and continues). Peers blocked at NCCL init then hang the
        // launcher. Exit non-zero from here instead so the launcher's
        // `supervise_children` SIGTERMs survivors. Post-rendezvous
        // failures stay under `ClusterCoordinator`'s `max_fail` policy
        // (heartbeat staleness drops dead ranks; cohort continues if
        // under threshold) and never reach this site.
        //
        // `LocalCluster::from_env` returns `Ok(None)` ONLY when
        // `FLODL_CLUSTER_JSON` is unset, so any `Err` here unambiguously
        // identifies a cluster-rank context with a fatal parse failure.
        match crate::distributed::cluster::LocalCluster::from_env() {
            Ok(Some(cluster)) => {
                // One cluster session per process (see launcher's
                // `claim_cluster_entry`): the rank-side bridges and the
                // coordinator they dial are equally per-session.
                if let Err(e) =
                    crate::distributed::launcher::claim_cluster_entry("rank")
                {
                    eprintln!("flodl cluster rank: {e}");
                    std::process::exit(1);
                }
                let dispatch_result: Result<Self> = Self::run_cluster_rank_via_coord(
                    cluster,
                    policy,
                    backend,
                    model_factory,
                    optim_factory,
                    train_fn,
                    dataset,
                    batch_size,
                    num_epochs,
                    config,
                    scheduler_fn,
                    epoch_fn,
                    checkpoint_fn,
                    eval_fn,
                    eval_dataset,
                );
                return match dispatch_result {
                    Ok(h) => Ok(h),
                    Err(e) => {
                        eprintln!(
                            "flodl cluster rank: pre-rendezvous setup failed: {e}"
                        );
                        std::process::exit(1);
                    }
                };
            }
            Ok(None) => {
                // Not a cluster rank (FLODL_CLUSTER_JSON unset). Fall
                // through to the single-host path below.
            }
            Err(e) => {
                eprintln!(
                    "flodl cluster rank: envelope parse failed: {e}"
                );
                std::process::exit(1);
            }
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
                eval_fn,
                eval_dataset,
                config.eval_every_epochs,
                eval_result_fn,
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
        let anchor = config.elche.anchor;
        let mut el_che = crate::distributed::ddp::ElChe::new(world_size, anchor);
        if let Some(target) = config.elche.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.elche.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }
        if let Some(min) = config.elche.min_anchor {
            el_che = el_che.with_min_anchor(min);
        }
        if let Some(diff) = config.elche.max_batch_diff {
            el_che = el_che.with_max_batch_diff(diff);
        }

        // Cold-start anchor: precedence is partition_ratios > spec prior >
        // rank-0 fallback. When the user supplied per-rank ratios, the
        // smallest ratio is the slow rank by user assertion. Otherwise we
        // ask the GPUs themselves (compute capability + VRAM) which one is
        // most likely the slowest. Either way, the pick is "soft" — once
        // timing data accumulates, election may move the anchor.
        if let Some(ratios) = config.elche.partition_ratios.as_ref() {
            // Explicit user input: a length mismatch is a config error,
            // not a silent skip (the old behavior dropped the slow-rank
            // seed AND let the runtime partitioner consume a wrong-length
            // vec).
            if ratios.len() != world_size {
                return Err(crate::tensor::TensorError::new(&format!(
                    "partition_ratios has {} entries but world_size is \
                     {world_size}; provide one ratio per rank",
                    ratios.len(),
                )));
            }
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
        let div_threshold = config.elche.divergence_threshold;
        let no_div_guard = config.elche.no_divergence_guard;
        let ckpt_every = config.checkpoint_every;
        let snap_timeout = config.snapshot_timeout_secs;
        let partition_ratios = config.elche.partition_ratios.clone();
        let max_grad_norm = config.max_grad_norm;
        let timeline = config.timeline.clone();
        let coord_timeline = timeline.clone();
        let coord_batch_size = batch_size;
        let seed: u64 = crate::distributed::ddp_run::SHUFFLE_BASE_SEED;

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
                .max_overshoot(config.elche.max_overshoot)
                .elche_relax_up(config.elche.relax_up)
                .meta_controller(config.elche.meta_controller);
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
                easgd_alpha: config.elche.easgd_alpha,
                gamma: config.elche.gamma,
                timeline: worker_tl,
                policy,
                save_path: None,
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
                            None, // eval_fn: threaded path, not yet wired
                            None, // eval_dataset
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
            launcher_driver: None,
            architecture_svg,
            graph_label,
            graph_hash,
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

    /// Blocking: wait for the next epoch's aggregated metrics.
    ///
    /// Returns `None` when training ends (sender dropped). In multi-GPU
    /// mode this blocks per epoch as the coordinator aggregates ranks; in
    /// the single-GPU fallback the queue is fully populated by the time
    /// `run()` returns, so calls return queued metrics non-blocking, then
    /// `None`.
    pub fn next_metrics(&self) -> Option<EpochMetrics> {
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
    ///
    /// **Launcher-mode caveat** (cluster runs and the multi-GPU
    /// auto-promote path): the launcher process never trains — ranks are
    /// separate processes — so the returned `TrainedState` is **empty**
    /// (`params` / `buffers` are zero-length). Cross-process final-state
    /// egress is a planned follow-up; until it lands, retrieve the final
    /// model from the checkpoint bundle (`save_path` +
    /// `checkpoint_every`, or the `ShutdownWithSave` bundle). A warning
    /// is logged when an empty launcher-mode state is returned so the
    /// gap is visible at runtime, not just in docs.
    pub fn join(mut self) -> Result<TrainedState> {
        // Single-GPU: state was captured in run_single()
        if let Some(state) = self.final_state.take() {
            return Ok(state);
        }

        // Cluster mode launcher: wait on the launcher driver thread.
        // The driver runs `run_launcher_with_config` which spawns
        // ranks, drives the ClusterCoordinator until completion, and
        // tears down. The launcher process holds no rank state — final
        // params live in the rank subprocesses where each rank's own
        // `DdpHandle::join` returns them via the via_coord coordinator
        // thread; cross-process snapshot egress to the launcher is a
        // follow-up. Return an empty `TrainedState` here so the
        // launcher's `.join()` still completes cleanly.
        if let Some(driver) = self.launcher_driver.take() {
            return match driver.join() {
                Ok(Ok(())) => {
                    // See the launcher-mode caveat on `join`'s docs:
                    // surfacing this at runtime keeps the transparent-DDP
                    // gap loud until cross-process state egress lands.
                    eprintln!(
                        "flodl ddp: join() on the launcher returns an EMPTY \
                         TrainedState (ranks are separate processes); use the \
                         checkpoint bundle for the final model"
                    );
                    Ok(TrainedState {
                        params: Vec::new(),
                        buffers: Vec::new(),
                    })
                }
                Ok(Err(e)) => Err(e),
                Err(_) => Err(TensorError::new(
                    "join: launcher driver thread panicked",
                )),
            };
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
    ///
    /// Uses [`crate::sys::detect_gpus`] (nvidia-smi based) instead of
    /// libtorch's `cuda_device_name_idx` / `cuda_memory_info_idx` so it
    /// can run on the controller's main thread without violating the
    /// "no CUDA touch before fan-out" invariant.
    fn print_summary(devices: &[Device], policy: &ApplyPolicy, backend: &AverageBackend) {
        use crate::monitor::format_bytes;

        let gpus = crate::sys::detect_gpus();
        let mut parts = Vec::with_capacity(devices.len());
        let mut names = Vec::with_capacity(devices.len());

        for &dev in devices {
            if let Device::CUDA(idx) = dev {
                let gpu = gpus.iter().find(|g| g.index == idx);
                let raw_name = gpu
                    .map(|g| g.name.clone())
                    .unwrap_or_else(|| format!("CUDA({})", idx));
                let short = gpu
                    .map(|g| g.short_name())
                    .unwrap_or_else(|| raw_name.clone());
                let vram = gpu
                    .map(|g| format!(" ({})", format_bytes(g.vram_bytes())))
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

        meta["anchor"] = serde_json::json!(config.elche.anchor);
        if let Some(target) = config.elche.overhead_target {
            meta["overhead_target"] = serde_json::json!(target);
        }
        if let Some(max) = config.elche.max_anchor {
            meta["max_anchor"] = serde_json::json!(max);
        }
        if let Some(min) = config.elche.min_anchor {
            meta["min_anchor"] = serde_json::json!(min);
        }
        if let Some(diff) = config.elche.max_batch_diff {
            meta["max_batch_diff"] = serde_json::json!(diff);
        }
        if let Some(overshoot) = config.elche.max_overshoot {
            meta["max_overshoot"] = serde_json::json!(overshoot);
        }
        if let Some(dt) = config.elche.divergence_threshold {
            meta["divergence_threshold"] = serde_json::json!(dt);
        }

        meta
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
