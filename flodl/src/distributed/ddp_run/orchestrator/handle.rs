//! `DdpHandle`: handle returned by `Trainer::builder().run()` / `Trainer::setup()`.
//!
//! Holds, depending on topology, the launcher driver thread (cluster /
//! multi-GPU auto-promote) or the inline single-GPU state. Provides `join`,
//! `poll_metrics`, `next_metrics`, plus the `launch` constructor called from
//! the builder.

use std::sync::mpsc;
use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use crate::distributed::ddp_run::{
    ApplyPolicy, AverageBackend, CheckpointFn, ConvergenceGuard, DdpRunConfig, EpochFn,
    EpochMetrics, EvalFn, EvalResultFn, MetricsFn, SchedulerFn, TrainedState,
};
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
///
/// # Dropping without joining
///
/// [`join`](Self::join) is the intended terminal call. In launcher / cluster
/// mode, dropping the handle without joining detaches the launcher driver
/// thread: **training continues to completion in the background** rather than
/// being torn down — an accidental drop never cancels a long run. For an
/// explicit cooperative cancel, call [`Self::shutdown`]. In single-GPU mode
/// training has already finished by the time the handle exists, so a drop is
/// a plain no-op.
pub struct DdpHandle {
    pub(super) devices: Vec<Device>,
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
    /// Cooperative-shutdown flag shared with the launcher's coordinator /
    /// rendezvous threads. [`Self::shutdown`] raises it; the launcher's
    /// accept polls and tick loop observe it within one poll interval.
    /// `None` outside launcher mode.
    pub(super) launcher_abort: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
    /// Internal launcher shared by the builder (`DdpBuilder::run`).
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
        // Auto-promote: when this process would otherwise enter
        // `Role::SingleDevice` (no cluster envelope in env) but
        // 2+ visible GPUs are present, synthesize a localhost cluster
        // and set `FLODL_INTERNAL_FULL_CLUSTER_JSON` so the dispatch below
        // returns `Role::Launcher`. This makes `Trainer::run` /
        // `Trainer::builder().run()` "just work" for single-host
        // multi-GPU without any cluster yml or programmatic config —
        // matching the UX of the legacy in-process threaded path
        // while running on the canonical process-per-rank path.
        //
        // Skips auto-promote when:
        //   - Any cluster-role env var is already set (fdl-cli overlay,
        //     programmatic `cfg.cluster`, or this process IS a spawned
        //     rank / relay child re-entering the user binary).
        //   - `detect_gpus()` returns <2 (no DDP to do; single-device
        //     path will run).
        //   - Compiled with `cfg(test)` (flodl's own test builds; tests
        //     that exercise multi-GPU use `Ddp::wrap` — manual per-rank
        //     DDP. External crates depending on flodl always see
        //     auto-promote in production builds, including `cargo run`
        //     and release binaries.)
        //
        // `detect_gpus()` respects `CUDA_VISIBLE_DEVICES`, so production
        // callers that want to scope down also have that lever.
        #[cfg(not(test))]
        {
            use crate::distributed::launcher::ENV_FULL_CLUSTER_JSON;
            // Role-env gate shared with the programmatic-cluster
            // promotion in `DdpBuilder::run`: covers ALL role vars,
            // including the relay's. A relay child spawned on this
            // very host sees >=2 GPUs too — re-promoting inside it
            // poisoned its env and killed the cohort at dispatch
            // ("inconsistent env").
            if crate::distributed::launcher::role_env_pristine() {
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
                // One-time reminder: in a multi-host cluster the
                // checkpoint/resume bundle is split across hosts and each
                // piece is written on its writer's host, so the path must
                // land on shared storage or resume breaks. Fires here (once,
                // on the controller, before fan-out) only for genuine 2+-host
                // clusters; single-box multi-GPU never trips it.
                if let Some(msg) = checkpoint_shared_storage_warning(
                    full.spans_multiple_workers(),
                    config.save_path.as_deref(),
                    config.resume_from.as_deref(),
                ) {
                    eprintln!("{msg}");
                }
                // Sink for aggregated EpochMetrics. The coord pushes
                // each completed epoch's metrics here; the user's
                // `DdpHandle::next_metrics()` polls them off. Wired
                // alongside `metrics_fn` (both fire on aggregation).
                let (sink_tx, sink_rx) =
                    mpsc::channel::<EpochMetrics>();
                // Capture the static model schema (param/buffer names) for the
                // controller-side consensus-checkpoint writer. Built on CPU in
                // the launcher process — reads names only, touches no CUDA
                // context (honors the "no CUDA before training" launcher
                // invariant). Best-effort: a factory that fails here just
                // leaves the schema unset (consensus checkpoints degrade to
                // meta-only); it does not abort the launch.
                let mut model_schema: Option<crate::distributed::ModelSchema> = None;
                match model_factory(Device::CPU) {
                    Ok(probe) => {
                        model_schema =
                            Some(crate::distributed::ModelSchema::from_module(&probe));
                        // Model-derived frame ceiling: the same CPU probe
                        // yields the exact wire footprint, replacing the
                        // 1 GiB default reject-threshold on every
                        // length-prefixed read in this process (the
                        // controller lives here). The relay children get
                        // it via `RelaySpec`; ranks derive the identical
                        // value from the model they build. Set BEFORE the
                        // driver spawns anything that reads frames.
                        let params: Vec<crate::tensor::Tensor> = probe
                            .parameters()
                            .iter()
                            .map(|p| p.variable.data())
                            .collect();
                        let buffers: Vec<crate::tensor::Tensor> =
                            probe.buffers().iter().map(|b| b.get()).collect();
                        let wire_bytes =
                            crate::distributed::wire::tensors_wire_bytes(&params)
                                + crate::distributed::wire::tensors_wire_bytes(&buffers);
                        crate::distributed::wire::set_frame_ceiling(
                            crate::distributed::wire::derive_frame_ceiling(wire_bytes),
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
                let mut outer_optimizer =
                    outer_optimizer_factory.as_ref().map(|f| f());
                // Resume: re-seed the outer optimizer's momentum from
                // `<stem>.outer.fdl` (the outer momentum lives controller-side,
                // so the launcher restores it, not the ranks). Skipped when not
                // resuming, when the sidecar is absent (a fresh / OuterAvg
                // run), or for a stateless variant (load is a no-op). A throwaway
                // CPU probe model supplies the parameter shapes — same "no CUDA
                // before training" CPU-only probe as the schema capture above.
                if let (Some(opt), Some(stem)) =
                    (outer_optimizer.as_mut(), config.resume_from.as_ref())
                {
                    let outer_path = crate::distributed::CheckpointBundle::model_path(stem)
                        .with_extension("outer.fdl");
                    if outer_path.exists() {
                        match model_factory(Device::CPU) {
                            Ok(probe) => {
                                let loaded = outer_path
                                    .to_str()
                                    .ok_or_else(|| {
                                        crate::tensor::TensorError::new(
                                            "resume: non-utf8 outer-momentum path",
                                        )
                                    })
                                    .and_then(|p| {
                                        crate::distributed::load_outer_momentum(&probe, p)
                                    })
                                    .and_then(|m| opt.load_checkpoint_state(m));
                                match loaded {
                                    Ok(()) => eprintln!(
                                        "  resume: loaded outer-optimizer momentum from {}",
                                        outer_path.display()
                                    ),
                                    Err(e) => eprintln!(
                                        "  resume: outer-momentum load from {} failed \
                                         ({e}); outer optimizer starts from zero momentum",
                                        outer_path.display()
                                    ),
                                }
                            }
                            Err(e) => eprintln!(
                                "  resume: probe model for outer-momentum shapes failed \
                                 ({e}); outer optimizer starts from zero momentum"
                            ),
                        }
                    }
                }
                // Controller-scope coordinator wiring. The config is
                // built by a FACTORY at world-formation time (the join
                // window decides the world size, not the config file),
                // so everything it needs is captured here and sized
                // there.
                // Schedule space: picks (samples × augment views) — the
                // coordinator's whole ledger runs in this space.
                let dataset_len = dataset.len() * config.augment.max(1);
                let coord_spec = crate::distributed::launcher::CoordSpec {
                    backend,
                    config_factory: Box::new(move |world_size| {
                        let mut coord_config = build_coord_config_from_builder(
                            policy,
                            backend,
                            &config,
                            convergence_guard,
                            metrics_fn,
                            eval_result_fn,
                            world_size,
                            dataset_len,
                            batch_size,
                            num_epochs,
                        )?;
                        coord_config = coord_config.metrics_sink_tx(sink_tx);
                        if let Some(schema) = model_schema {
                            coord_config = coord_config.model_schema(schema);
                        }
                        Ok(coord_config)
                    }),
                };
                // Cooperative-shutdown flag shared with the launcher's
                // infrastructure threads; DdpHandle::shutdown raises it.
                let launcher_abort =
                    Arc::new(std::sync::atomic::AtomicBool::new(false));
                let abort_for_driver = Arc::clone(&launcher_abort);
                let driver = std::thread::Builder::new()
                    .name("flodl-launcher-driver".to_string())
                    .spawn(move || {
                        crate::distributed::launcher::run_launcher_with_config(
                            full,
                            Some(coord_spec),
                            outer_optimizer,
                            abort_for_driver,
                        )
                    })
                    .map_err(|e| {
                        crate::tensor::TensorError::new(&format!(
                            "spawn launcher driver thread: {e}"
                        ))
                    })?;
                return Ok(DdpHandle {
                    devices: Vec::new(),
                    final_state: None,
                    metrics_rx: Some(sink_rx),
                    launcher_driver: Some(driver),
                    launcher_abort: Some(launcher_abort),
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
            crate::distributed::launcher::Role::Agent => {
                // This process is a per-host worker agent (dial-in
                // membership): join the controller's window, spawn +
                // supervise this host's relay and rank children, exit.
                // It never trains.
                match crate::distributed::launcher::run_agent() {
                    Ok(()) => std::process::exit(0),
                    Err(e) => {
                        eprintln!("cluster agent: {e}");
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
        // `FLODL_INTERNAL_CLUSTER_JSON` is unset, so any `Err` here unambiguously
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
                    outer_optimizer_factory,
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
                // Not a cluster rank (FLODL_INTERNAL_CLUSTER_JSON unset). Fall
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
                config.vram_pool,
                config.vram_max_usage,
                config.ram_max_usage,
                config.augment,
                config.transform.clone(),
                scheduler,
                eval_fn,
                eval_dataset,
                config.eval_every_epochs,
                eval_result_fn,
            );
        }

        // Reaching here means 2+ visible CUDA devices with no cluster
        // envelope and no launcher role. In production (`cfg(not(test))`)
        // that combination auto-promotes to the process-per-rank path at
        // the top of this function, so this is unreachable. It is reachable
        // only in flodl's own `cfg(test)` builds, where auto-promote is
        // gated off; multi-GPU testing there uses `Ddp::wrap` (thread-based)
        // or the cluster-worker substrate. The in-process thread-per-GPU
        // training engine that once lived here has been removed.
        Err(crate::tensor::TensorError::new(
            "in-process multi-GPU training has been removed: on 2+ GPUs use \
             Trainer::run / Trainer::builder().run() (process-per-rank \
             auto-promote), or Ddp::wrap for manual thread-based DDP",
        ))
    }

    /// Number of GPUs in this DDP group.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Devices in use.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Graph architecture SVG, if the model presents a graph through
    /// [`Module::as_any`] (see [`crate::graph::GraphExt::as_graph`]).
    ///
    /// [`Module::as_any`]: crate::nn::Module::as_any
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

    /// Cooperatively cancel a launcher-mode run and wait for teardown.
    ///
    /// Raises the shared abort flag observed by the launcher's
    /// coordinator accept/tick loops and the rendezvous accept poll,
    /// then joins the launcher driver like [`Self::join`]. Spawned rank
    /// children observe the coordinator going away (control-stream EOF /
    /// coord-liveness deadline) and self-terminate; the launcher's child
    /// supervision reaps them before the driver returns.
    ///
    /// This is the EXPLICIT cancel — dropping the handle deliberately
    /// does NOT cancel (see "Dropping without joining" on
    /// [`DdpHandle`]): an accidental drop must never kill a long run.
    ///
    /// Single-GPU mode: training already completed inline; this is
    /// equivalent to [`Self::join`].
    pub fn shutdown(mut self) -> Result<TrainedState> {
        if let Some(flag) = self.launcher_abort.take() {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        self.join()
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

        // Neither single-GPU state nor a launcher driver: the only handle
        // shape that reached this point came from the removed in-process
        // engine, which no longer produces a `DdpHandle`. Unreachable in
        // practice; surface loudly rather than hang.
        Err(TensorError::new("join: no trained state available"))
    }

}

/// Build the one-time multi-host reminder that checkpoint/resume paths
/// must live on shared storage, or `None` when it does not apply.
///
/// In a cluster the checkpoint/resume bytes are split across hosts and
/// each write lands on whatever host the writer runs on: the elected
/// `checkpoint_role` rank (and its failover targets) runs the user's
/// `checkpoint_fn` on its own host; on `ShutdownWithSave` each surviving
/// worker writes `<stem>.fdl`/`.optim` on its own host while the
/// controller writes `<stem>.meta.json` on ITS host; and `resume_from`
/// reads that meta on the controller. So a `save_path`/`resume_from`
/// that is not on shared storage scatters the bundle and breaks resume
/// -- the classic way in is a run that works single-host and silently
/// breaks on scale-up when the elected rank happens to be remote.
///
/// This is a reminder, not a hard error: we cannot verify a path is
/// shared from its string alone (a relative path under the shared
/// project mount is fine; an absolute path on local disk is not), and we
/// cannot force the user's `checkpoint_fn` to write shared. Single-worker
/// clusters (single-box multi-GPU auto-promote) never see it -- every
/// piece lands on the one host's disk, so there is nothing to share.
fn checkpoint_shared_storage_warning(
    spans_multiple_workers: bool,
    save_path: Option<&str>,
    resume_from: Option<&str>,
) -> Option<String> {
    if !spans_multiple_workers {
        return None;
    }
    // Name whichever path(s) are set; if both are set and differ, name
    // both so the operator can eyeball each.
    let stem = match (save_path, resume_from) {
        (Some(s), Some(r)) if s != r => {
            format!("save_path {s:?} / resume_from {r:?}")
        }
        (Some(s), _) => format!("{s:?}"),
        (_, Some(r)) => format!("{r:?}"),
        (None, None) => return None,
    };
    Some(format!(
        "flodl cluster: checkpoint/resume path {stem} must resolve to SHARED \
         storage visible to every host. In a multi-host cluster the elected \
         checkpoint rank and each worker's save-on-failure write on their OWN \
         host, while the controller writes the `.meta.json` sidecar and reads \
         it back on resume on ITS host -- a host-local path scatters the \
         bundle and breaks resume. If the path already resolves to shared \
         storage (NAS / virtiofs / SSHFS / the shared project mount) on all \
         hosts, ignore this."
    ))
}

#[cfg(test)]
mod tests {
    use super::checkpoint_shared_storage_warning;

    #[test]
    fn no_warning_on_single_worker_cluster() {
        // Single-box multi-GPU auto-promote: everything lands on one
        // host's disk, so a save_path is fine.
        assert!(
            checkpoint_shared_storage_warning(false, Some("runs/ckpt"), None).is_none()
        );
    }

    #[test]
    fn no_warning_when_no_checkpoint_path_set() {
        assert!(checkpoint_shared_storage_warning(true, None, None).is_none());
    }

    #[test]
    fn warns_multi_host_with_save_path() {
        let msg = checkpoint_shared_storage_warning(true, Some("runs/ckpt"), None)
            .expect("multi-host + save_path must warn");
        assert!(msg.contains("runs/ckpt"));
        assert!(msg.contains("SHARED"));
    }

    #[test]
    fn warns_multi_host_with_resume_only() {
        let msg = checkpoint_shared_storage_warning(true, None, Some("runs/ckpt"))
            .expect("multi-host + resume_from must warn");
        assert!(msg.contains("runs/ckpt"));
    }

    #[test]
    fn names_both_paths_when_they_differ() {
        let msg =
            checkpoint_shared_storage_warning(true, Some("out/save"), Some("in/resume"))
                .expect("must warn");
        assert!(msg.contains("out/save"));
        assert!(msg.contains("in/resume"));
    }
}
