//! Cluster worker-side rank entry point (the `via_coord` path).
//!
//! [`DdpHandle::run_cluster_rank_via_coord`] is invoked synchronously on
//! the rank's main thread when the launcher trampoline resolves a
//! cluster role. Parameterized by `(ApplyPolicy, AverageBackend)` — the
//! training engine, transport shell, and run loop are identical across
//! all combinations; only the reduce-backend bootstrap differs (NCCL
//! rendezvous + comm init vs CPU reduce client), so that is the one
//! `match backend` in the body. It builds a `GpuWorker`, connects a
//! [`ClusterWorker`] to the coord-side [`ClusterCoordinator`], and
//! drives the training loop to completion.
//!
//! Helpers (`parse_or_resolve_socket_addr`, `rank_fires_callbacks`) live
//! here because they are exclusively used by this entry.
//!
//! [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
//! [`ClusterWorker`]: crate::distributed::cluster_worker::ClusterWorker

use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::ddp_run::{
    ApplyPolicy, AverageBackend, CheckpointFn, DdpRunConfig, EpochCallbackPolicy,
    EpochFn, EvalFn, SchedulerFn, TrainedState, WorkerConfig,
};
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{DType, Device, Result, Tensor, TensorError};

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
///
/// **Memory trade-off (deliberate):** because the answer is always
/// `true`, every rank carries the `epoch_fn` / `checkpoint_fn` /
/// `eval_fn` closures and whatever state they capture, even though only
/// the current role rank executes them. This is the cost of elastic role
/// rotation — when the role rank dies (or `Fastest` re-resolves), the
/// coord can hand the role to any survivor, which only works if every
/// survivor already holds the closure. Dropping the unused copies would
/// re-break rotation, so the cost is intrinsic, not an oversight. Keep
/// captured state lean (e.g. `Arc` shared handles, not cloned datasets)
/// if a callback closure is heavy.
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
    /// Cluster-rank entry point, parameterized by
    /// `(ApplyPolicy, AverageBackend)` — every cluster rank runs this,
    /// driven by a [`ClusterCoordinator`] (elastic-membership-aware).
    ///
    /// Routes through [`ClusterWorker`] talking to the coord via the
    /// host-local relay's control loopback over TCP control frames; the
    /// CPU backend additionally dials the relay's data loopback for the
    /// reduce channel. The `pre_sync_scratch` buffers are allocated
    /// unconditionally for NCCL workers (see `GpuWorker::new`) so the
    /// abort-retry path in `sync_now_nccl` can restore params after a
    /// peer-death abort and re-AllReduce on the survivor cohort.
    ///
    /// `save_path` on [`DdpRunConfig`] is optional. When set, the
    /// controller writes `<save_path>.meta.json` and workers write
    /// `<save_path>.fdl` / `.optim` on unrecoverable failure (via
    /// `ShutdownWithSave`). When unset, the run executes normally and
    /// just skips save activity (legitimate for tests and
    /// inference-style usage).
    ///
    /// The controller owns ElChe + the convergence guard; the rank side
    /// installs neither (the launcher trampoline builds them into the
    /// `ClusterCoordinatorConfig`).
    ///
    /// **NCCL init-on-main**: `ncclCommInitRank` MUST run on a thread
    /// that already owns the CUDA context — calling it from a freshly
    /// spawned thread corrupts the CUDA context on heterogeneous GPUs.
    /// The whole body runs synchronously on the rank process's main
    /// thread (process-per-rank model: no in-rank workers to
    /// coordinate), which also keeps NCCL init AND collectives on one
    /// thread, avoiding the per-thread CUDA-context-inheritance issue
    /// observed with NCCL 2.27.5 + precompiled cu128 libtorch + sm_120
    /// on Blackwell.
    ///
    /// Final params + buffers ARE returned via [`TrainedState`]: the
    /// inner GpuWorker's end-of-training `send_final_snapshot` is
    /// captured in
    /// [`crate::distributed::cluster_worker::ClusterWorker::run_until_shutdown`]
    /// and ferried into a `TrainedState` here. The
    /// `ShutdownWithSave`-written bundle remains the resume vehicle for
    /// the unrecoverable-failure case (rank crashed before the snapshot
    /// path ran).
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterWorker`]: crate::distributed::cluster_worker::ClusterWorker
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_cluster_rank_via_coord<F, M, G, O, T>(
        cluster: crate::distributed::cluster::LocalCluster,
        policy: ApplyPolicy,
        backend: AverageBackend,
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
        use std::sync::atomic::AtomicBool;

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

        let policy_label = match policy {
            ApplyPolicy::Sync => "Sync",
            ApplyPolicy::Cadence => "Cadence",
            ApplyPolicy::Async => "Async",
        };
        let backend_label = match backend {
            AverageBackend::Nccl => "Nccl",
            AverageBackend::Cpu => "Cpu",
        };
        crate::verbose!(
            "  ddp: cluster rank {global_rank}/{world_size} on {device:?} \
             ({policy_label}+{backend_label} via_coord, save_path={save_path:?})"
        );

        // Ranks reach the coordinator through their host-local relay's
        // control loopback (+5), not the controller's control port (+3)
        // directly. The relay forwards each rank's frames upstream.
        let coord_port = cluster
            .controller
            .port
            .saturating_add(crate::distributed::relay::RELAY_CONTROL_LOOPBACK_OFFSET);
        let coord_addr_str = format!("127.0.0.1:{coord_port}");
        let coord_addr = parse_or_resolve_socket_addr(&coord_addr_str)?;
        // CPU backend only: the reduce data channel rides the relay's
        // data loopback (+4); the relay forwards each rank's reduce
        // buffer upstream to the controller (+2).
        let controller_addr_str = match backend {
            AverageBackend::Cpu => {
                let controller_port = cluster.controller.port.saturating_add(
                    crate::distributed::relay::RELAY_DATA_LOOPBACK_OFFSET,
                );
                Some(format!("127.0.0.1:{controller_port}"))
            }
            AverageBackend::Nccl => None,
        };
        let session_salt = cluster.salt;
        let dataset_sig = [0u8; 32];

        let mut meta = serde_json::json!({
            "mode": format!("cluster-rank {policy_label}+{backend_label} via_coord"),
            "global_rank": global_rank,
            "world_size": world_size,
            "device": format!("{device:?}"),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "total_samples": total_samples,
            "coord_addr": coord_addr_str,
            "save_path": save_path,
        });
        if let Some(ref addr) = controller_addr_str {
            meta["controller_addr"] = serde_json::json!(addr);
        }
        let training_meta = Some(meta);

        let timeline_for_thread = config.timeline.clone();
        let max_grad_norm = config.max_grad_norm;
        // EASGD blending is gated to Async at the single authoritative
        // point, `GpuWorker::new` (every worker path funnels through it),
        // which forces `None` for any non-async worker. Pass the configured
        // value through unchanged.
        let easgd_alpha = config.elche.easgd_alpha;
        let save_path_for_thread = save_path.clone();

        // Run the rank body synchronously on the rank process's main
        // thread (see the init-on-main doc above).
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
            // Backend bootstrap — the ONE divergence between the rank
            // entries. NCCL: pin the device + init the comm on this
            // (main) thread (see the init-on-main doc above). CPU: dial
            // the relay's data loopback for the reduce channel. Both
            // produce the (nccl_comm, cpu_client) pair `ClusterWorker`
            // consumes.
            #[cfg(feature = "cuda")]
            if matches!(backend, AverageBackend::Nccl) {
                if let crate::tensor::Device::CUDA(idx) = device {
                    crate::tensor::set_current_cuda_device(idx);
                }
            }
            let nccl_comm = match backend {
                AverageBackend::Nccl => {
                    let rdv = cluster.rendezvous(dataset_sig)?;
                    let comm = crate::distributed::nccl::NcclRankComm::init_rank(
                        global_rank,
                        world_size,
                        rdv.unique_id(),
                    )?;
                    drop(rdv);
                    Some(comm)
                }
                AverageBackend::Cpu => None,
            };
            let mut cpu_client = match &controller_addr_str {
                Some(addr) => {
                    Some(crate::distributed::cpu_reduce::CpuReduceClient::connect(
                        parse_or_resolve_socket_addr(addr)?,
                        global_rank as u32,
                        world_size as u32,
                        session_salt,
                    )?)
                }
                None => None,
            };
            // Build tmp model, broadcast initial state, pin to CPU.
            // GpuWorker::new re-creates the model + copies params back
            // to GPU on the worker thread. The initial broadcast is
            // load-bearing for K>>1 cadence and EASGD (which smooths but
            // does not erase divergent initial state).
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
            match (&nccl_comm, &mut cpu_client) {
                (Some(comm), _) => {
                    // NCCL: in-place broadcast from rank 0.
                    if !initial_params_local.is_empty() {
                        let refs: Vec<&Tensor> = initial_params_local.iter().collect();
                        comm.broadcast(&refs, 0)?;
                    }
                    if !initial_buffers_local.is_empty() {
                        let refs: Vec<&Tensor> = initial_buffers_local.iter().collect();
                        comm.broadcast(&refs, 0)?;
                    }
                }
                (None, Some(client)) => {
                    // CPU: round-trip through the reduce channel, then
                    // copy_ back in. The copy_ into a requires_grad leaf
                    // must run under no_grad — libtorch otherwise aborts
                    // with "a leaf Variable that requires grad is being
                    // used in an in-place operation". Mirrors PyTorch's
                    // `with torch.no_grad(): dst.copy_(src)` convention
                    // for bootstrap-time parameter sync.
                    if !initial_params_local.is_empty() {
                        let refs: Vec<&Tensor> = initial_params_local.iter().collect();
                        let broadcast = client.broadcast_from_root(&refs, 0)?;
                        crate::autograd::no_grad(|| -> crate::tensor::Result<()> {
                            for (dst, src) in initial_params_local.iter().zip(&broadcast) {
                                dst.copy_(src, false)?;
                            }
                            Ok(())
                        })?;
                    }
                    // Buffers too — the NCCL path broadcasts both, and a
                    // rank-sensitive buffer init (e.g. randomly seeded
                    // running stats) would otherwise diverge silently. The
                    // CPU reduce transport is f32-only, so only f32 buffers
                    // ride the channel; non-f32 buffers (e.g. BatchNorm's
                    // i64 `num_batches_tracked`) are deterministic counters
                    // initialized identically on every rank, so leaving
                    // them at their factory value is correct, not a dropped
                    // sync. All ranks build the same model, so the f32
                    // subset matches in count/order across ranks and the
                    // collective stays balanced.
                    let f32_buffers: Vec<&Tensor> = initial_buffers_local
                        .iter()
                        .filter(|b| b.dtype() == DType::Float32)
                        .collect();
                    if !f32_buffers.is_empty() {
                        let broadcast = client.broadcast_from_root(&f32_buffers, 0)?;
                        crate::autograd::no_grad(|| -> crate::tensor::Result<()> {
                            for (dst, src) in f32_buffers.iter().zip(&broadcast) {
                                dst.copy_(src, false)?;
                            }
                            Ok(())
                        })?;
                    }
                }
                (None, None) => {
                    return Err(TensorError::new(
                        "run_cluster_rank_via_coord: neither NCCL comm nor CPU \
                         reduce client was constructed (backend bootstrap bug)",
                    ));
                }
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
                // On resume, read the shuffle seed from the checkpoint meta so
                // this rank reproduces the recorded epoch permutation exactly;
                // a fresh run falls back to SHUFFLE_BASE_SEED. The coordinator
                // reads the same meta, so the cohort stays consistent without a
                // broadcast.
                seed: crate::distributed::ddp_run::resolve_shuffle_seed(
                    config.resume_from.as_deref(),
                )?,
                max_grad_norm,
                easgd_alpha,
                gamma: config.elche.gamma,
                timeline: timeline_for_thread,
                policy,
                save_path: save_path_for_thread,
            };

            // ClusterWorker bridges set up heartbeat + NCCL watchdog +
            // inbound (DeclareDead / NewNcclSession / ShutdownWithSave)
            // / outbound timing. NCCL handles its collectives in-place
            // (`cpu_client = None`); the CPU backend reduces through the
            // client's data channel. `epoch_fn_for_thread` is gated by
            // the coord's runtime role push.
            let mut cluster_worker =
                crate::distributed::cluster_worker::ClusterWorker::connect_and_build(
                    coord_addr,
                    cpu_client,
                    global_rank as u32,
                    session_salt,
                    worker_config,
                    model_factory,
                    optim_factory,
                    dataset,
                    nccl_comm,
                    checkpoint_fn_for_thread,
                    epoch_fn_for_thread,
                    eval_fn_for_thread,
                    eval_dataset_for_thread,
                    outer_optimizer_factory,
                )?;

            if let Some(f) = scheduler_fn {
                cluster_worker.inner_mut().set_scheduler(f(world_size));
            }

            // Resume: re-seed this rank's replicated outer-optimizer momentum
            // from `<stem>.outer.fdl` (NCCL path; the momentum lives per-rank
            // on GPU, so every rank loads it, mirroring the model's replicated
            // resume). No-op when not resuming / no outer optimizer / sidecar
            // absent. Errors are logged + ignored (resume from zero momentum is
            // a safe fallback, not a hard failure).
            if let Some(stem) = config.resume_from.as_ref() {
                if let Err(e) = cluster_worker.inner_mut().resume_outer_momentum(stem) {
                    eprintln!(
                        "cluster_worker: rank {global_rank} outer-momentum resume \
                         failed ({e}); starting from zero momentum"
                    );
                }
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
                // Last-gasp forensic record so a postmortem can tell this
                // self-inflicted crash apart from a controller-driven
                // ShutdownWithSave (which writes a CheckpointMeta). Best-
                // effort and only when a save_path exists; a write failure
                // is logged, never allowed to mask the original error.
                if let Some(ref stem) = save_path {
                    let record = crate::distributed::RankDeathRecord::new(
                        global_rank,
                        world_size,
                        e.to_string(),
                    );
                    let path = crate::distributed::CheckpointBundle::rank_death_path(
                        stem,
                        global_rank,
                    );
                    match record.write_to_file(&path) {
                        Ok(()) => eprintln!(
                            "flodl cluster rank: wrote death record to {}",
                            path.display()
                        ),
                        Err(werr) => eprintln!(
                            "flodl cluster rank: failed to write death record \
                             to {}: {werr}",
                            path.display()
                        ),
                    }
                }
                std::process::exit(1);
            }
        };

        Ok(DdpHandle {
            devices: vec![device],
            shutdown: Arc::new(AtomicBool::new(false)),
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
