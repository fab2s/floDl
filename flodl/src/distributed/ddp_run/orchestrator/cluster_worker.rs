//! Cluster worker-side rank entry points (the four `via_coord` paths).
//!
//! Each of the four functions is invoked synchronously on the rank's
//! main thread when the launcher trampoline resolves a cluster role:
//! one per `(ApplyPolicy, AverageBackend)` combination. They build a
//! `GpuWorker`, connect a [`ClusterWorker`] to the coord-side
//! [`ClusterCoordinator`] (NCCL or CPU-averaging backend), and drive
//! the training loop to completion.
//!
//! Helpers (`parse_or_resolve_socket_addr`, `rank_fires_callbacks`) live
//! here because they are exclusively used by these four functions.
//!
//! [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
//! [`ClusterWorker`]: crate::distributed::cluster_worker::ClusterWorker

use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::ddp_run::{
    convergence, ApplyPolicy, CheckpointFn, DdpRunConfig, EpochCallbackPolicy,
    EpochFn, EvalFn, SchedulerFn, TrainedState, WorkerConfig,
};
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::DdpHandle;

/// Parse `host:port`, falling back to DNS / `/etc/hosts` resolution.
///
/// `SocketAddr::from_str` only accepts numeric IPs. In cluster mode the
/// user's `controller_addr` is often a short hostname (e.g. `exa`) that
/// resolves through the host's `/etc/hosts` under `network_mode: host`.
fn parse_or_resolve_socket_addr(addr: &str) -> Result<std::net::SocketAddr> {
    use std::net::ToSocketAddrs;
    if let Ok(s) = addr.parse::<std::net::SocketAddr>() {
        return Ok(s);
    }
    let mut iter = addr.to_socket_addrs().map_err(|e| {
        TensorError::new(&format!("ddp: resolve addr '{addr}': {e}"))
    })?;
    iter.next().ok_or_else(|| {
        TensorError::new(&format!("ddp: resolve addr '{addr}': no addresses returned"))
    })
}

/// Decide whether this cluster-mode rank should receive a copy of the
/// user-supplied per-epoch closures (`epoch_fn` / `checkpoint_fn` /
/// `eval_fn`) at GpuWorker construction time.
///
/// Under controller-driven role assignment the answer is **always
/// `true` in cluster mode**: every rank holds the closure compiled
/// in, and the coord's runtime-pushed role state (sticky
/// `checkpoint_role`, `eval_role`, `epoch_callback_role` on
/// `ClusterCoordinator`) gates actual execution per-message
/// (`Checkpoint` / `ExecuteEvalCallback` carry `target_rank`;
/// `epoch_fn` reads `GpuWorker::epoch_callback_role()` at each epoch
/// transition).
///
/// Without all-Some, role rotation on rank death (or `Fastest`
/// re-resolve) would land a `Checkpoint` / `ExecuteEvalCallback`
/// frame on a worker whose `*_fn = None`, loud-erroring with
/// "dispatched to rank X but fn is None". All-Some makes coord-driven
/// rotation work as designed.
///
/// Validates the policy itself loud-errors on out-of-bounds
/// `Rank(n)`. `Fastest` is fully supported.
fn rank_fires_callbacks(
    policy: EpochCallbackPolicy,
    _global_rank: usize,
    world_size: usize,
) -> Result<bool> {
    match policy {
        EpochCallbackPolicy::Rank(n) => {
            if n >= world_size {
                return Err(crate::tensor::TensorError::new(&format!(
                    "EpochCallbackPolicy::Rank({n}) out of bounds (world_size={world_size}). \
                     Pick a rank in 0..{world_size}."
                )));
            }
            // Every rank holds the closure; coord's targeted dispatch
            // + `epoch_callback_role` wire-pushed state gates which
            // rank actually fires per-event.
            Ok(true)
        }
        EpochCallbackPolicy::Fastest => Ok(true),
    }
}

impl DdpHandle {
    /// Cluster-rank entry point for `ApplyPolicy::Sync + AverageBackend::Nccl`
    /// driven by a [`ClusterCoordinator`] (elastic-membership-aware).
    ///
    /// Routes through [`ClusterWorker`] talking to a coord on
    /// `controller_addr:controller_port + 3` over TCP control frames. The
    /// `pre_sync_scratch` buffers are allocated unconditionally for
    /// NCCL workers (see `GpuWorker::new`) so the abort-retry path in
    /// `sync_now_nccl` can restore params after a peer-death abort
    /// and re-AllReduce on the survivor cohort.
    ///
    /// `save_path` on [`DdpRunConfig`] is optional. When set, the
    /// controller writes `<save_path>.meta.json` and workers write
    /// `<save_path>.fdl` / `.optim` on unrecoverable failure (via
    /// `ShutdownWithSave`). When unset, the run executes normally
    /// and just skips save activity (legitimate for tests and
    /// inference-style usage).
    ///
    /// Final params + buffers ARE returned via [`TrainedState`] on
    /// this path: the inner GpuWorker's end-of-training
    /// `send_final_snapshot` is captured in
    /// [`crate::distributed::cluster_worker::ClusterWorker::run_until_shutdown`]
    /// and ferried into a `TrainedState` here. The
    /// `ShutdownWithSave`-written bundle remains the resume vehicle
    /// for the unrecoverable-failure case (rank crashed before the
    /// snapshot path ran).
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterWorker`]: crate::distributed::cluster_worker::ClusterWorker
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_cluster_rank_sync_nccl_via_coord<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        config: DdpRunConfig,
        scheduler_fn: Option<SchedulerFn>,
        epoch_fn: Option<EpochFn<M>>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
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

        // `save_path` is optional. When set, the cluster persists a
        // bundle on unrecoverable failure (and on the worker side at
        // checkpoint events). When unset, the run executes normally
        // and just skips all save activity — legitimate for tests and
        // inference-style runs.
        let save_path = config.save_path.clone();

        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        let total_samples = dataset.len();

        // Resolve callback policy: only the chosen rank gets `epoch_fn`;
        // others see `None`, so the fire-site in
        // `ClusterWorker::run_until_shutdown` is a cheap no-op.
        let fires_callbacks =
            rank_fires_callbacks(config.epoch_callback_policy, global_rank, world_size)?;
        let epoch_fn_for_thread = if fires_callbacks { epoch_fn } else { None };
        let checkpoint_fn_for_thread =
            if fires_callbacks { checkpoint_fn } else { None };
        let eval_fn_for_thread = if fires_callbacks { eval_fn } else { None };
        let eval_dataset_for_thread = if fires_callbacks { eval_dataset } else { None };

        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} \
             (Sync+Nccl via_coord, save_path={save_path:?})"
        );

        // Coord control channel at controller_port + 3. controller_port is
        // already used by the NCCL rendezvous; +2 is the
        // ClusterController (CPU averaging, unused in NCCL mode but
        // still bound by the launcher).
        let coord_port = cluster.controller.port.saturating_add(3);
        let coord_addr_str = format!("{}:{coord_port}", cluster.controller.host);
        let coord_addr = parse_or_resolve_socket_addr(&coord_addr_str)?;
        let session_salt = cluster.salt;
        let dataset_sig = [0u8; 32];

        let training_meta = Some(serde_json::json!({
            "mode": "cluster-rank Sync+Nccl via_coord",
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "coord_addr": coord_addr_str,
            "save_path": save_path,
        }));

        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;
        let save_path_for_thread = save_path.clone();

        // Pin device + init NCCL on the rank process's main thread.
        // `ncclCommInitRank` MUST run on a thread that already owns
        // the CUDA context — calling it from a freshly spawned thread
        // corrupts the CUDA context on heterogeneous GPUs. Matches
        // the documented "NCCL init-on-main + split() pattern
        // required" (CLAUDE.md) and the legacy `NcclComms::new+split`
        // discipline at orchestrator.rs:765-777. The spawn closure
        // below re-pins the device thread-locally and uses the comm
        // moved in via closure capture.
        #[cfg(feature = "cuda")]
        if let crate::tensor::Device::CUDA(idx) = device {
            crate::tensor::set_current_cuda_device(idx);
        }
        let rdv = cluster.rendezvous(dataset_sig)?;
        let nccl_comm =
            NcclRankComm::init_rank(global_rank, world_size, rdv.unique_id())?;
        drop(rdv);

        // Run the rank body synchronously on the rank process's main
        // thread. Process-per-rank model: each rank is its own process
        // with one GPU, so there's no in-process N-worker coordination
        // — the legacy `std::thread::spawn` here was vestigial shape
        // from the threaded Coordinator path. Keeping NCCL init AND
        // collectives on the same thread also avoids the per-thread
        // CUDA-context-inheritance issue observed with NCCL 2.27.5 +
        // precompiled cu128 libtorch + sm_120 on Blackwell.
        //
        // Wrap in a fast-exit-on-Err guard: `Err` escaping here means
        // the rank can't continue. Returning Err to the caller's
        // run-loop risks being swallowed (e.g. ddp-bench's `run_combo`
        // Err arm prints and continues — the launcher's child-
        // supervision then sees exit status 0 and blocked peers hang
        // forever). Process-exit non-zero so the launcher SIGTERMs
        // local peers; for remote ranks the coord's heartbeat-
        // staleness detector drops them post-registration
        // (`max_failure` applies) or the SSH client's broken
        // connection propagates pre-registration.
        let worker_result: Result<TrainedState> = (move || -> Result<TrainedState> {
            // Build tmp model, broadcast initial state, pin to CPU
            // (GpuWorker::new re-creates the model and copies params
            // back to GPU on the worker thread).
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
                save_path: save_path_for_thread,
            };

            // ClusterWorker connects to the coord, sets up the
            // inbound/outbound/heartbeat/watchdog bridge threads, and
            // attaches the session mailbox into the inner GpuWorker.
            // `cpu_client = None` — NCCL backend doesn't use the CPU
            // reduce client for averaging. `epoch_fn_for_thread` is
            // `Some(...)` only on the rank chosen by
            // `EpochCallbackPolicy`; other ranks see `None`.
            let mut cluster_worker =
                crate::distributed::cluster_worker::ClusterWorker::connect_and_build(
                    coord_addr,
                    None,
                    global_rank as u32,
                    session_salt,
                    worker_config,
                    model_factory,
                    optim_factory,
                    dataset,
                    Some(nccl_comm),
                    checkpoint_fn_for_thread,
                    epoch_fn_for_thread,
                    eval_fn_for_thread,
                    eval_dataset_for_thread,
                )?;

            // Per-batch LR scheduler: stateless pure function attached
            // to the worker. Per-batch invocation reads
            // `global_step + steps_since_avg`, where `global_step` is
            // controller-broadcast via `SetGlobalStep` after every
            // averaging cycle. Workers stay in lockstep without an
            // explicit per-LR broadcast.
            if let Some(f) = scheduler_fn {
                cluster_worker.inner_mut().set_scheduler(f(world_size));
            }

            let final_snapshot = cluster_worker.run_until_shutdown(train_fn)?;

            // Final snapshot captured from the inner GpuWorker before
            // teardown. `None` means snapshot_params didn't run (worker
            // errored before send_final_snapshot or the channel
            // disconnected); fall back to an empty TrainedState so
            // callers can still consume the bundle written via
            // `ShutdownWithSave`. Tensors land on CPU per
            // `snapshot_params`'s contract.
            Ok(final_snapshot
                .map(|snap| TrainedState {
                    params: snap.params,
                    buffers: snap.buffers,
                })
                .unwrap_or(TrainedState {
                    params: Vec::new(),
                    buffers: Vec::new(),
                }))
        })();
        let final_state = match worker_result {
            Ok(state) => state,
            Err(e) => {
                eprintln!("flodl cluster rank: rank failed: {e}");
                std::process::exit(1);
            }
        };

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
            nccl_abort_handles: Vec::new(),
            final_state: Some(final_state),
            metrics_rx: None,
            launcher_driver: None,
            architecture_svg: None,
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }

    /// Cluster-rank entry point for `ApplyPolicy::Cadence` /
    /// `ApplyPolicy::Async` + `AverageBackend::Nccl` driven by a
    /// [`ClusterCoordinator`] (elastic-membership + persistence aware).
    ///
    /// Mirrors [`run_cluster_rank_sync_nccl_via_coord`](Self::run_cluster_rank_sync_nccl_via_coord)
    /// — the worker side is identical between Sync and Cadence under the
    /// via-coord routing because the
    /// [`ClusterCoordinator`](crate::distributed::cluster_coordinator::ClusterCoordinator)
    /// owns ElChe + ConvergenceGuard for all three policies (see
    /// [`ClusterCoordinator::trigger_averaging`] for the cadence broadcast
    /// and [`ClusterCoordinator::finish_averaging_nccl`] for the guard /
    /// `report_timing` / `nudge_anchor_down` / `relax_anchor_up`
    /// pipeline). The worker just trains batches and responds to coord-
    /// issued `SyncNow` via `handle_control` → `sync_now_nccl`.
    ///
    /// Cadence and Async NCCL collapse to the same entry: overshoot
    /// machinery (the only old-coordinator Cadence/Async distinction)
    /// is an async/CPU concept (see `feedback_overshoot_async_only` and
    /// `feedback_nccl_no_overshoot_throttle`). [`WorkerConfig::policy`]
    /// carries the user's chosen policy for log lines + future-policy
    /// metadata but does not branch the algorithm.
    ///
    /// `save_path` on [`DdpRunConfig`] is REQUIRED — the cluster
    /// save-on-unrecoverable-failure flow needs a destination. Loud
    /// error at startup if unset.
    ///
    /// # Controller-scope config
    ///
    /// `policy`, `convergence_guard`, and the ElChe knobs on
    /// [`DdpRunConfig`] are controller-scope configuration. The
    /// controller is a singleton scheduler that lives in the launcher
    /// process, decoupled from any rank, so trait objects
    /// (`Box<dyn ConvergenceGuard>`) constructed in the rank-side
    /// `main()` are threaded through this entry only for API symmetry
    /// with the single-host path. The authoritative controller-side
    /// install happens at the launcher boundary, where the same
    /// `DdpRunConfig` is visible to both controller and rank dispatch.
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterCoordinator::trigger_averaging`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator::trigger_averaging
    /// [`ClusterCoordinator::finish_averaging_nccl`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_cluster_rank_cadence_nccl_via_coord<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        policy: ApplyPolicy,
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        config: DdpRunConfig,
        convergence_guard: Option<Box<dyn convergence::ConvergenceGuard>>,
        scheduler_fn: Option<SchedulerFn>,
        epoch_fn: Option<EpochFn<M>>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
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

        // save_path is required: the via_coord path' save-on-failure
        // flow needs a destination.
        // `save_path` optional: persistence on unrecoverable failure
        // is opt-in. Unset = run normally, skip saves.
        let save_path = config.save_path.clone();

        // The controller owns ElChe + guard. `convergence_guard` is
        // threaded through here so the call site stays honest (the
        // builder accepts it; this entry receives it), but the
        // controller-side install is deferred to the launcher-
        // trampoline slice — see this method's doc comment.
        let _ = convergence_guard;

        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        let total_samples = dataset.len();

        let fires_callbacks =
            rank_fires_callbacks(config.epoch_callback_policy, global_rank, world_size)?;
        let epoch_fn_for_thread = if fires_callbacks { epoch_fn } else { None };
        let checkpoint_fn_for_thread =
            if fires_callbacks { checkpoint_fn } else { None };
        let eval_fn_for_thread = if fires_callbacks { eval_fn } else { None };
        let eval_dataset_for_thread = if fires_callbacks { eval_dataset } else { None };

        let policy_label = match policy {
            ApplyPolicy::Sync => "Sync",
            ApplyPolicy::Cadence => "Cadence",
            ApplyPolicy::Async => "Async",
        };
        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} \
             ({policy_label}+Nccl via_coord, save_path={save_path:?})"
        );

        // Coord control channel at controller_port + 3 (same convention as
        // the Sync via_coord entry).
        let coord_port = cluster.controller.port.saturating_add(3);
        let coord_addr_str = format!("{}:{coord_port}", cluster.controller.host);
        let coord_addr = parse_or_resolve_socket_addr(&coord_addr_str)?;
        let session_salt = cluster.salt;
        let dataset_sig = [0u8; 32];

        let training_meta = Some(serde_json::json!({
            "mode": format!("cluster-rank {policy_label}+Nccl via_coord"),
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "coord_addr": coord_addr_str,
            "save_path": save_path,
        }));

        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;
        let easgd_alpha = config.easgd_alpha;
        let save_path_for_thread = save_path.clone();

        // Pin device + init NCCL on the rank process's main thread.
        // See `run_cluster_rank_sync_nccl_via_coord` above for the
        // full rationale (NCCL init-on-main + split() pattern; spawn
        // closure re-pins device thread-locally and uses moved-in
        // comm; CUDA module warm-up before NCCL init for sm_120
        // precompiled-cu128 stacks).
        #[cfg(feature = "cuda")]
        if let crate::tensor::Device::CUDA(idx) = device {
            crate::tensor::set_current_cuda_device(idx);
        }
        let rdv = cluster.rendezvous(dataset_sig)?;
        let nccl_comm =
            NcclRankComm::init_rank(global_rank, world_size, rdv.unique_id())?;
        drop(rdv);

        // Run synchronously on rank's main thread. See
        // `run_cluster_rank_sync_nccl_via_coord` for the full rationale
        // (process-per-rank model: no in-rank workers to coordinate,
        // legacy spawn was vestigial).
        let worker_result: Result<TrainedState> = (move || -> Result<TrainedState> {
            // Build tmp model, broadcast initial state, pin to CPU.
            // GpuWorker::new re-creates the model + copies params back to
            // GPU on the worker thread.
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
                easgd_alpha,
                timeline: timeline_for_thread,
                policy,
                save_path: save_path_for_thread,
            };

            // ClusterWorker bridges set up heartbeat + NCCL watchdog +
            // inbound (DeclareDead / NewNcclSession / ShutdownWithSave)
            // / outbound timing. The worker's `handle_control` responds
            // to coord-issued `SyncNow` via `sync_now_nccl` — identical
            // semantics to the Sync via_coord path. `cpu_client = None`
            // because NCCL handles all collectives.
            let mut cluster_worker =
                crate::distributed::cluster_worker::ClusterWorker::connect_and_build(
                    coord_addr,
                    None,
                    global_rank as u32,
                    session_salt,
                    worker_config,
                    model_factory,
                    optim_factory,
                    dataset,
                    Some(nccl_comm),
                    checkpoint_fn_for_thread,
                    epoch_fn_for_thread,
                    eval_fn_for_thread,
                    eval_dataset_for_thread,
                )?;

            if let Some(f) = scheduler_fn {
                cluster_worker.inner_mut().set_scheduler(f(world_size));
            }

            let final_snapshot = cluster_worker.run_until_shutdown(train_fn)?;

            // Final snapshot captured from the inner GpuWorker before
            // teardown. `None` falls back to an empty TrainedState; the
            // ShutdownWithSave bundle remains the canonical resume path
            // for the unrecoverable-failure case.
            Ok(final_snapshot
                .map(|snap| TrainedState {
                    params: snap.params,
                    buffers: snap.buffers,
                })
                .unwrap_or(TrainedState {
                    params: Vec::new(),
                    buffers: Vec::new(),
                }))
        })();
        let final_state = match worker_result {
            Ok(state) => state,
            Err(e) => {
                eprintln!("flodl cluster rank: rank failed: {e}");
                std::process::exit(1);
            }
        };

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
            nccl_abort_handles: Vec::new(),
            final_state: Some(final_state),
            metrics_rx: None,
            launcher_driver: None,
            architecture_svg: None,
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }

    /// Cluster-rank entry point for `ApplyPolicy::Sync + AverageBackend::Cpu`
    /// driven by a [`ClusterCoordinator`] (singleton-ElChe-on-controller,
    /// elastic-membership-aware).
    ///
    /// CPU-averaging counterpart of
    /// [`run_cluster_rank_sync_nccl_via_coord`](Self::run_cluster_rank_sync_nccl_via_coord).
    /// The worker connects to the controller for CPU averaging
    /// (`controller_port + 2`) and to the coordinator for control frames
    /// (`controller_port + 3`). The coordinator owns ElChe, ConvergenceGuard,
    /// per-rank divergence aggregation, anchor adjustment, and
    /// `max_overshoot` auto-tune via
    /// [`ClusterCoordinator::finish_averaging_cpu`].
    ///
    /// Initial params are broadcast from rank 0 via
    /// [`CpuReduceClient::broadcast_from_root`] BEFORE the client is
    /// handed to [`ClusterWorker::connect_and_build`]. The controller's
    /// accept loop is one-shot, so the same client must serve both the
    /// initial broadcast and the per-cycle averaging that follows.
    ///
    /// `save_path` on [`DdpRunConfig`] is REQUIRED — the cluster
    /// save-on-unrecoverable-failure path needs a destination. Loud
    /// error at startup if unset.
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterCoordinator::finish_averaging_cpu`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterWorker::connect_and_build`]:
    ///     crate::distributed::cluster_worker::ClusterWorker::connect_and_build
    /// [`CpuReduceClient::broadcast_from_root`]:
    ///     crate::distributed::CpuReduceClient::broadcast_from_root
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_cluster_rank_sync_cpu_via_coord<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        config: DdpRunConfig,
        scheduler_fn: Option<SchedulerFn>,
        epoch_fn: Option<EpochFn<M>>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::AtomicBool;
        use crate::distributed::cpu_reduce::CpuReduceClient;

        // `save_path` optional: persistence on unrecoverable failure
        // is opt-in. Unset = run normally, skip saves.
        let save_path = config.save_path.clone();

        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        let total_samples = dataset.len();

        let fires_callbacks =
            rank_fires_callbacks(config.epoch_callback_policy, global_rank, world_size)?;
        let epoch_fn_for_thread = if fires_callbacks { epoch_fn } else { None };
        let checkpoint_fn_for_thread =
            if fires_callbacks { checkpoint_fn } else { None };
        let eval_fn_for_thread = if fires_callbacks { eval_fn } else { None };
        let eval_dataset_for_thread = if fires_callbacks { eval_dataset } else { None };

        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} \
             (Sync+Cpu via_coord, save_path={save_path:?})"
        );

        let controller_port = cluster.controller.port.saturating_add(2);
        let controller_addr_str = format!("{}:{controller_port}", cluster.controller.host);
        let controller_addr = parse_or_resolve_socket_addr(&controller_addr_str)?;
        let coord_port = cluster.controller.port.saturating_add(3);
        let coord_addr_str = format!("{}:{coord_port}", cluster.controller.host);
        let coord_addr = parse_or_resolve_socket_addr(&coord_addr_str)?;
        let session_salt = cluster.salt;

        let training_meta = Some(serde_json::json!({
            "mode": "cluster-rank Sync+Cpu via_coord",
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "controller_addr": controller_addr_str,
            "coord_addr": coord_addr_str,
            "save_path": save_path,
        }));

        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;
        let save_path_for_thread = save_path.clone();

        // Run synchronously on rank's main thread. See
        // `run_cluster_rank_sync_nccl_via_coord` for the full rationale
        // (process-per-rank model, no in-rank coordination, legacy
        // spawn was vestigial).
        let worker_result: Result<TrainedState> = (move || -> Result<TrainedState> {
            // Connect to the ClusterController for CPU averaging. Same
            // client serves initial broadcast and the per-cycle reduce
            // loop (controller's accept is one-shot).
            let mut cpu_client = CpuReduceClient::connect(
                controller_addr,
                global_rank as u32,
                world_size as u32,
                session_salt,
            )?;

            // Build tmp model, broadcast initial params from rank 0 via
            // the avg-trick so all ranks start with identical weights
            // even when the user's factory isn't deterministic.
            let tmp_model = model_factory(device)?;
            let initial_params_local: Vec<Tensor> = tmp_model
                .parameters()
                .iter()
                .map(|p| p.variable.data())
                .collect();
            let initial_buffers_local: Vec<Tensor> = tmp_model
                .buffers()
                .iter()
                .map(|b| b.get())
                .collect();

            if !initial_params_local.is_empty() {
                let refs: Vec<&Tensor> = initial_params_local.iter().collect();
                let broadcast = cpu_client.broadcast_from_root(&refs, 0)?;
                // copy_ into the parameter's underlying tensor (a leaf
                // with requires_grad=True) must go through a no_grad
                // guard. Without it, libtorch's autograd validates:
                // "a leaf Variable that requires grad is being used in
                // an in-place operation" and aborts. Mirrors PyTorch's
                // `with torch.no_grad(): dst.copy_(src)` convention for
                // bootstrap-time parameter sync.
                crate::autograd::no_grad(|| -> crate::tensor::Result<()> {
                    for (dst, src) in initial_params_local.iter().zip(&broadcast) {
                        dst.copy_(src, false)?;
                    }
                    Ok(())
                })?;
            }

            let initial_params: Vec<Tensor> = initial_params_local
                .iter()
                .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
                .collect::<Result<Vec<_>>>()?;
            let initial_buffers: Vec<Tensor> = initial_buffers_local
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
                save_path: save_path_for_thread,
            };

            // ClusterWorker takes ownership of cpu_client for the
            // per-cycle CPU averaging path. The coordinator drives
            // averaging via control frames; the param bridge ships
            // ParamSnapshots back through `cpu_client` and synthesises
            // `TimingMsg::SyncAck` (with divergence/pre_norm/post_norm)
            // for the coord's `finish_averaging_cpu` pipeline.
            let mut cluster_worker =
                crate::distributed::cluster_worker::ClusterWorker::connect_and_build(
                    coord_addr,
                    Some(cpu_client),
                    global_rank as u32,
                    session_salt,
                    worker_config,
                    model_factory,
                    optim_factory,
                    dataset,
                    None,
                    checkpoint_fn_for_thread,
                    epoch_fn_for_thread,
                    eval_fn_for_thread,
                    eval_dataset_for_thread,
                )?;

            if let Some(f) = scheduler_fn {
                cluster_worker.inner_mut().set_scheduler(f(world_size));
            }

            let final_snapshot = cluster_worker.run_until_shutdown(train_fn)?;

            // Final snapshot captured from the inner GpuWorker before
            // teardown. `None` falls back to an empty TrainedState; the
            // ShutdownWithSave bundle remains the canonical resume path
            // for the unrecoverable-failure case.
            Ok(final_snapshot
                .map(|snap| TrainedState {
                    params: snap.params,
                    buffers: snap.buffers,
                })
                .unwrap_or(TrainedState {
                    params: Vec::new(),
                    buffers: Vec::new(),
                }))
        })();
        let final_state = match worker_result {
            Ok(state) => state,
            Err(e) => {
                eprintln!("flodl cluster rank: rank failed: {e}");
                std::process::exit(1);
            }
        };

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
            nccl_abort_handles: Vec::new(),
            final_state: Some(final_state),
            metrics_rx: None,
            launcher_driver: None,
            architecture_svg: None,
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }

    /// Cluster-rank entry point for `ApplyPolicy::Cadence` /
    /// `ApplyPolicy::Async` + `AverageBackend::Cpu` driven by a
    /// [`ClusterCoordinator`] (singleton-ElChe-on-controller).
    ///
    /// CPU counterpart of
    /// [`run_cluster_rank_cadence_nccl_via_coord`](Self::run_cluster_rank_cadence_nccl_via_coord).
    /// Under via_coord routing the worker is policy-agnostic: it
    /// responds to coord-issued averaging triggers and EASGD-blends the
    /// returned tensors. The Cadence-vs-Async distinction lives on the
    /// controller (cadence broadcast in
    /// [`ClusterCoordinator::trigger_averaging`] + overshoot / anchor /
    /// guard pipeline in
    /// [`ClusterCoordinator::finish_averaging_cpu`]).
    ///
    /// Initial params are broadcast from rank 0 via
    /// [`CpuReduceClient::broadcast_from_root`] BEFORE the client is
    /// handed to [`ClusterWorker::connect_and_build`] — the same client
    /// must serve both the initial broadcast and the per-cycle
    /// averaging (controller accept loop is one-shot).
    ///
    /// `save_path` on [`DdpRunConfig`] is REQUIRED.
    ///
    /// `convergence_guard` is threaded through for API symmetry with
    /// the NCCL via_coord entry; the controller-side install is
    /// deferred to the launcher-trampoline pass (see
    /// [`run_cluster_rank_cadence_nccl_via_coord`](Self::run_cluster_rank_cadence_nccl_via_coord)
    /// for the rationale).
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterCoordinator::trigger_averaging`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator::trigger_averaging
    /// [`ClusterCoordinator::finish_averaging_cpu`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterWorker::connect_and_build`]:
    ///     crate::distributed::cluster_worker::ClusterWorker::connect_and_build
    /// [`CpuReduceClient::broadcast_from_root`]:
    ///     crate::distributed::CpuReduceClient::broadcast_from_root
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_cluster_rank_cadence_cpu_via_coord<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        policy: ApplyPolicy,
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        num_epochs: usize,
        config: DdpRunConfig,
        convergence_guard: Option<Box<dyn convergence::ConvergenceGuard>>,
        scheduler_fn: Option<SchedulerFn>,
        epoch_fn: Option<EpochFn<M>>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        use std::sync::atomic::AtomicBool;
        use crate::distributed::cpu_reduce::CpuReduceClient;

        // `save_path` optional: persistence on unrecoverable failure
        // is opt-in. Unset = run normally, skip saves.
        let save_path = config.save_path.clone();

        // Threaded through for API symmetry; controller-side install is
        // the trampoline pass's job. See the NCCL via_coord doc.
        let _ = convergence_guard;

        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        let total_samples = dataset.len();

        let fires_callbacks =
            rank_fires_callbacks(config.epoch_callback_policy, global_rank, world_size)?;
        let epoch_fn_for_thread = if fires_callbacks { epoch_fn } else { None };
        let checkpoint_fn_for_thread =
            if fires_callbacks { checkpoint_fn } else { None };
        let eval_fn_for_thread = if fires_callbacks { eval_fn } else { None };
        let eval_dataset_for_thread = if fires_callbacks { eval_dataset } else { None };

        let policy_label = match policy {
            ApplyPolicy::Sync => "Sync",
            ApplyPolicy::Cadence => "Cadence",
            ApplyPolicy::Async => "Async",
        };
        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} \
             ({policy_label}+Cpu via_coord, save_path={save_path:?})"
        );

        let controller_port = cluster.controller.port.saturating_add(2);
        let controller_addr_str = format!("{}:{controller_port}", cluster.controller.host);
        let controller_addr = parse_or_resolve_socket_addr(&controller_addr_str)?;
        let coord_port = cluster.controller.port.saturating_add(3);
        let coord_addr_str = format!("{}:{coord_port}", cluster.controller.host);
        let coord_addr = parse_or_resolve_socket_addr(&coord_addr_str)?;
        let session_salt = cluster.salt;

        let training_meta = Some(serde_json::json!({
            "mode": format!("cluster-rank {policy_label}+Cpu via_coord"),
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "controller_addr": controller_addr_str,
            "coord_addr": coord_addr_str,
            "save_path": save_path,
        }));

        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;
        let easgd_alpha = config.easgd_alpha;
        let save_path_for_thread = save_path.clone();

        // Run synchronously on rank's main thread. See
        // `run_cluster_rank_sync_nccl_via_coord` for the full rationale.
        let worker_result: Result<TrainedState> = (move || -> Result<TrainedState> {
            let mut cpu_client = CpuReduceClient::connect(
                controller_addr,
                global_rank as u32,
                world_size as u32,
                session_salt,
            )?;

            let tmp_model = model_factory(device)?;
            let initial_params_local: Vec<Tensor> = tmp_model
                .parameters()
                .iter()
                .map(|p| p.variable.data())
                .collect();
            let initial_buffers_local: Vec<Tensor> = tmp_model
                .buffers()
                .iter()
                .map(|b| b.get())
                .collect();

            // Initial broadcast load-bearing for K>>1 cadence and EASGD
            // (which smooths but does not erase divergent initial state).
            if !initial_params_local.is_empty() {
                let refs: Vec<&Tensor> = initial_params_local.iter().collect();
                let broadcast = cpu_client.broadcast_from_root(&refs, 0)?;
                // copy_ into the parameter's underlying tensor (a leaf
                // with requires_grad=True) must go through a no_grad
                // guard. Without it, libtorch's autograd validates:
                // "a leaf Variable that requires grad is being used in
                // an in-place operation" and aborts. Mirrors PyTorch's
                // `with torch.no_grad(): dst.copy_(src)` convention for
                // bootstrap-time parameter sync.
                crate::autograd::no_grad(|| -> crate::tensor::Result<()> {
                    for (dst, src) in initial_params_local.iter().zip(&broadcast) {
                        dst.copy_(src, false)?;
                    }
                    Ok(())
                })?;
            }

            let initial_params: Vec<Tensor> = initial_params_local
                .iter()
                .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
                .collect::<Result<Vec<_>>>()?;
            let initial_buffers: Vec<Tensor> = initial_buffers_local
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
                easgd_alpha,
                timeline: timeline_for_thread,
                policy,
                save_path: save_path_for_thread,
            };

            let mut cluster_worker =
                crate::distributed::cluster_worker::ClusterWorker::connect_and_build(
                    coord_addr,
                    Some(cpu_client),
                    global_rank as u32,
                    session_salt,
                    worker_config,
                    model_factory,
                    optim_factory,
                    dataset,
                    None,
                    checkpoint_fn_for_thread,
                    epoch_fn_for_thread,
                    eval_fn_for_thread,
                    eval_dataset_for_thread,
                )?;

            if let Some(f) = scheduler_fn {
                cluster_worker.inner_mut().set_scheduler(f(world_size));
            }

            let final_snapshot = cluster_worker.run_until_shutdown(train_fn)?;

            // Final snapshot captured from the inner GpuWorker before
            // teardown. `None` falls back to an empty TrainedState.
            Ok(final_snapshot
                .map(|snap| TrainedState {
                    params: snap.params,
                    buffers: snap.buffers,
                })
                .unwrap_or(TrainedState {
                    params: Vec::new(),
                    buffers: Vec::new(),
                }))
        })();
        let final_state = match worker_result {
            Ok(state) => state,
            Err(e) => {
                eprintln!("flodl cluster rank: rank failed: {e}");
                std::process::exit(1);
            }
        };

        Ok(DdpHandle {
            worker_handles: Vec::new(),
            coordinator_handle: None,
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
            nccl_abort_handles: Vec::new(),
            final_state: Some(final_state),
            metrics_rx: None,
            launcher_driver: None,
            architecture_svg: None,
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }
}
