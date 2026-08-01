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
    ApplyPolicy, AverageBackend, DdpRunConfig, EpochCallbackPolicy,
    RankCallbacks, SchedulerFn, TrainedState, Worker, WorkerConfig,
};
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{DType, Device, Result, Tensor, TensorError};

use super::DdpHandle;

/// Parse `host:port`, falling back to DNS / `/etc/hosts` resolution.
///
/// `SocketAddr::from_str` only accepts numeric IPs. In cluster mode the
/// controller's `host:` is often a short hostname (e.g. `exa`) that
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
/// Last-gasp forensic record so a postmortem can tell a self-inflicted rank
/// crash (Err / panic / cooperative `Worker` dropped without `finish()`) apart
/// from a controller-driven `ShutdownWithSave` (which writes a `CheckpointMeta`).
/// Best-effort and only when a `save_path` exists; a write failure is logged,
/// never allowed to mask the original cause. Shared by the managed rank entry
/// (`run_cluster_rank_via_coord`), the cooperative setup-failure path
/// (`run_cluster_rank_worker`), and the cooperative `Worker`'s Drop guard.
pub(crate) fn write_rank_death_record(
    save_path: Option<&str>,
    global_rank: usize,
    world_size: usize,
    reason: String,
) {
    let Some(stem) = save_path else {
        return;
    };
    let record =
        crate::distributed::RankDeathRecord::new(global_rank, world_size, reason);
    let path = crate::distributed::CheckpointBundle::rank_death_path(stem, global_rank);
    match record.write_to_file(&path) {
        Ok(()) => eprintln!(
            "flodl cluster rank: wrote death record to {}",
            path.display()
        ),
        Err(werr) => eprintln!(
            "flodl cluster rank: failed to write death record to {}: {werr}",
            path.display()
        ),
    }
}

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
        rank_callbacks: RankCallbacks<M>,
    ) -> Result<Self>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        // `save_path` optional: persistence on unrecoverable failure is opt-in.
        let save_path = config.save_path.clone();
        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();
        // Schedule space: picks (samples × augment views).
        let total_samples = dataset.len() * config.augment.max(1);

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

        // `training_meta` records the relay loopback addresses (control +5,
        // data +4) for monitor metadata. Recompute the strings here — pure
        // functions of the controller port — since `build_cluster_worker` owns
        // the sockets it actually dials.
        let coord_addr_str = format!(
            "127.0.0.1:{}",
            cluster.controller.port.saturating_add(
                crate::distributed::relay::RELAY_CONTROL_LOOPBACK_OFFSET,
            )
        );
        let controller_addr_str = match backend {
            AverageBackend::Cpu => Some(format!(
                "127.0.0.1:{}",
                cluster.controller.port.saturating_add(
                    crate::distributed::relay::RELAY_DATA_LOOPBACK_OFFSET,
                )
            )),
            AverageBackend::Nccl => None,
        };
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

        // Run the rank body synchronously on the rank process's main thread
        // (NCCL init-on-main; process-per-rank has no in-rank workers). Wrap in
        // catch_unwind so a panic still writes the forensic death record — a
        // bare panic would unwind to exit 101 with no `.death.json`,
        // indistinguishable in a postmortem from a clean exit. AssertUnwindSafe
        // is sound: on a panic the captured state is never reused.
        let worker_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            move || -> Result<TrainedState> {
                // Build + connect the worker (shared with the cooperative
                // entry), then self-drive it to shutdown.
                let cluster_worker = Self::build_cluster_worker(
                    cluster,
                    global_rank,
                    device,
                    world_size,
                    policy,
                    backend,
                    model_factory,
                    optim_factory,
                    dataset,
                    batch_size,
                    config,
                    scheduler_fn,
                    rank_callbacks,
                )?;

                let final_snapshot = cluster_worker.run_until_shutdown(train_fn)?;

                // Final snapshot captured before teardown; `None` (worker
                // errored before send_final_snapshot) falls back to an empty
                // TrainedState so callers can still consume the ShutdownWithSave
                // bundle. Tensors land on CPU per snapshot_params' contract.
                Ok(final_snapshot
                    .map(|snap| TrainedState {
                        params: snap.params,
                        buffers: snap.buffers,
                    })
                    .unwrap_or(TrainedState {
                        params: Vec::new(),
                        buffers: Vec::new(),
                    }))
            },
        ));
        // Last-gasp forensic record (see `write_rank_death_record`): tells a
        // self-inflicted crash apart from a controller-driven ShutdownWithSave.
        // Exit non-zero on Err so the launcher SIGTERMs local peers (a returned
        // Err risks being swallowed by a caller's run-loop, hanging blocked
        // peers).
        let write_death_record =
            |reason: String| write_rank_death_record(save_path.as_deref(), global_rank, world_size, reason);
        let final_state = match worker_outcome {
            Ok(Ok(state)) => state,
            Ok(Err(e)) => {
                eprintln!("flodl cluster rank: rank failed: {e}");
                write_death_record(e.to_string());
                // clean_process_exit, NOT process::exit: the worker's model /
                // comm state is still live on this never-unwound stack, and
                // libtorch's static destructors GP-fault over it — the
                // supervisor then reports "signal 11" instead of this
                // deliberate exit 1 (observed live on the rig when peers died
                // during a formation broadcast).
                super::clean_process_exit(1);
            }
            Err(panic) => {
                // The default panic hook has already printed the message +
                // backtrace; record the death, then resume the unwind so the
                // panic still surfaces (exit 101) for the launcher's
                // supervisor and any postmortem tooling.
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic payload".to_string());
                eprintln!("flodl cluster rank: rank panicked: {msg}");
                write_death_record(format!("panic: {msg}"));
                std::panic::resume_unwind(panic);
            }
        };

        Ok(DdpHandle {
            devices: vec![device],
            final_state: Some(final_state),
            metrics_rx: None,
            launcher_driver: None,
            launcher_abort: None,
            architecture_svg: None,
            graph_label: None,
            graph_hash: None,
            training_meta,
        })
    }

    /// Cooperative cluster-rank entry: [`Self::build_cluster_worker`] (the
    /// bootstrap shared with the managed [`Self::run_cluster_rank_via_coord`] —
    /// backend init, initial-state broadcast, bridges, scheduler, outer-momentum
    /// resume) then wrap the connected worker in a [`Worker`] the user drives,
    /// instead of self-driving `run_until_shutdown`. No `train_fn` (the user
    /// supplies forward+loss in their loop); the reduce still rides the
    /// coordinator's `SyncNow` drained inside `Worker::step`.
    ///
    /// **Failure model.** A pre-rendezvous setup failure writes the forensic
    /// death record here and returns `Err` (the caller exits non-zero to
    /// unblock peers). A *training-loop* failure is caught by the `Worker`'s
    /// Drop guard (un-`finish()`ed drop → death record + exit), so the
    /// blocked-peer-hang protection the managed path gets from its
    /// `catch_unwind` is preserved across the user-owned loop.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_cluster_rank_worker<F, M, G, O>(
        cluster: crate::distributed::cluster::LocalCluster,
        policy: ApplyPolicy,
        backend: AverageBackend,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        config: DdpRunConfig,
        scheduler_fn: Option<SchedulerFn>,
        rank_callbacks: RankCallbacks<M>,
    ) -> Result<Worker<M>>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
    {
        let save_path = config.save_path.clone();
        let (global_rank, device) = cluster.my_rank()?;
        let world_size = cluster.world_size();

        // The shared bootstrap can fail before the `Worker` (and its Drop
        // guard) exists, so write the forensic record here on Err — a setup
        // failure is then as traceable as a managed-path failure; the caller
        // exits non-zero to unblock peers.
        match Self::build_cluster_worker(
            cluster,
            global_rank,
            device,
            world_size,
            policy,
            backend,
            model_factory,
            optim_factory,
            dataset,
            batch_size,
            config,
            scheduler_fn,
            rank_callbacks,
        ) {
            Ok(cluster_worker) => Ok(Worker::cluster(
                cluster_worker,
                save_path,
                global_rank,
                world_size,
            )),
            Err(e) => {
                write_rank_death_record(
                    save_path.as_deref(), global_rank, world_size, e.to_string(),
                );
                Err(e)
            }
        }
    }

    /// Shared cluster-rank bootstrap for both the managed
    /// ([`Self::run_cluster_rank_via_coord`]) and cooperative
    /// ([`Self::run_cluster_rank_worker`]) entries — everything up through
    /// `connect_and_build`: backend init (NCCL init-on-main / CPU reduce
    /// client), initial-state broadcast from rank 0, `WorkerConfig`, the bridge
    /// setup, scheduler attach, and outer-momentum resume. Returns the
    /// connected [`ClusterWorker`](crate::distributed::cluster_worker::ClusterWorker);
    /// the caller decides how to drive it (self-driven `run_until_shutdown` vs
    /// the user-owned cooperative loop). `(global_rank, device, world_size)`
    /// are precomputed by the caller — both need them independently (managed:
    /// `training_meta`; cooperative: the Drop-guard forensics).
    ///
    /// This is the single source of the rank bootstrap; the two entries differ
    /// ONLY in their wrappers (meta + `catch_unwind` + `run_until_shutdown` +
    /// `TrainedState` vs on-Err death record + `Worker::cluster`).
    #[allow(clippy::too_many_arguments)]
    fn build_cluster_worker<F, M, G, O>(
        cluster: crate::distributed::cluster::LocalCluster,
        global_rank: usize,
        device: Device,
        world_size: usize,
        policy: ApplyPolicy,
        backend: AverageBackend,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        batch_size: usize,
        config: DdpRunConfig,
        scheduler_fn: Option<SchedulerFn>,
        rank_callbacks: RankCallbacks<M>,
    ) -> Result<crate::distributed::cluster_worker::ClusterWorker<M>>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
    {
        let RankCallbacks {
            checkpoint_fn,
            epoch_fn,
            eval_fn,
            eval_dataset,
            outer_optimizer_factory,
        } = rank_callbacks;
        let save_path = config.save_path.clone();
        // Schedule space: picks (samples × augment views).
        let total_samples = dataset.len() * config.augment.max(1);

        // Controller-driven role assignment: every rank holds the callback
        // closures; the coord's runtime role push gates who fires. Validates
        // the policy (loud on an out-of-bounds `Rank(n)`).
        let fires_callbacks = rank_fires_callbacks(
            config.epoch_callback_policy, global_rank, world_size,
        )?;
        let epoch_fn = if fires_callbacks { epoch_fn } else { None };
        let checkpoint_fn = if fires_callbacks { checkpoint_fn } else { None };
        let eval_fn = if fires_callbacks { eval_fn } else { None };
        let eval_dataset = if fires_callbacks { eval_dataset } else { None };

        // Ranks reach the coordinator through their host-local relay's control
        // loopback (+5); the CPU backend's reduce channel rides the relay's
        // data loopback (+4).
        let coord_port = cluster.controller.port.saturating_add(
            crate::distributed::relay::RELAY_CONTROL_LOOPBACK_OFFSET,
        );
        let coord_addr =
            parse_or_resolve_socket_addr(&format!("127.0.0.1:{coord_port}"))?;
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

        // Backend bootstrap (the one divergence): NCCL inits the comm on this
        // (main) thread — `ncclCommInitRank` from a freshly spawned thread
        // corrupts the CUDA context on heterogeneous GPUs; CPU dials the relay
        // data loopback for the reduce channel.
        #[cfg(feature = "gpu")]
        if matches!(backend, AverageBackend::Nccl) {
            if let crate::tensor::Device::CUDA(idx) = device {
                crate::tensor::set_current_cuda_device(idx);
            }
        }
        let nccl_comm = match backend {
            AverageBackend::Nccl => {
                let rdv = cluster.rendezvous(dataset_sig)?;
                let comm = crate::distributed::nccl::NcclRankComm::init_rank(
                    global_rank, world_size, rdv.unique_id(),
                )?;
                drop(rdv);
                Some(comm)
            }
            AverageBackend::Cpu => None,
        };
        let mut cpu_client = match &controller_addr_str {
            Some(addr) => {
                let mut client = crate::distributed::cpu_reduce::CpuReduceClient::connect(
                    parse_or_resolve_socket_addr(addr)?,
                    global_rank as u32,
                    world_size as u32,
                    session_salt,
                )?;
                // Model frames ride bf16 when configured (Control frames
                // — including the initial broadcast below — stay f32).
                client.set_bf16_wire(config.elche.bf16_wire);
                // Pinned decode is armed below, once the model exists and
                // its staging size is known (the RAM gate needs it).
                Some(client)
            }
            None => None,
        };

        // Build tmp model, broadcast initial state from rank 0, pin to CPU.
        // The initial broadcast is load-bearing for K>>1 cadence and EASGD
        // (which smooth but do not erase divergent initial state).
        let tmp_model = model_factory(device)?;
        let initial_params_local: Vec<Tensor> = tmp_model
            .parameters().iter().map(|p| p.variable.data()).collect();
        crate::distributed::ddp_run::ensure_trainable_params(
            initial_params_local.len(), "ddp: cluster rank",
        )?;
        let initial_buffers_local: Vec<Tensor> = tmp_model
            .buffers().iter().map(|b| b.get()).collect();
        // Model-derived frame ceiling for this rank's length-prefixed readers,
        // installed BEFORE the first framed read (the bootstrap consensus).
        {
            use crate::distributed::wire;
            let wire_bytes = wire::tensors_wire_bytes(&initial_params_local)
                + wire::tensors_wire_bytes(&initial_buffers_local);
            wire::set_frame_ceiling(wire::derive_frame_ceiling(wire_bytes));
        }
        // Barrier-paced CUDA consumers decode Model replies back into
        // the pinned SNAPSHOT staging they were streamed from
        // (`set_decode_into_request`), making `load_averaged`'s
        // `copy_(non_blocking)` a true async H2D (a pageable source
        // degrades to a synchronous bounce copy). The staging's bytes
        // are dead once the streamed encode has read them, so this
        // locks ZERO marginal memory and needs no RAM gate (the retired
        // slot-staging predecessor's locked second model copy is what
        // thrashed the RAM-tight rig host — +6-8% epoch wall of
        // reduce-window swap; see `pinned_decode_affordable`'s
        // calibration note). Under `bf16_wire` the reply lands verbatim
        // in the bf16 staging and the writeback stages through a
        // device-side bounce (see `h2d_copy_via_bounce`), halving the
        // H2D bytes as well. Async always keeps fresh-alloc decode: its
        // control channel has two producers with only per-producer
        // FIFO, so a second Update can be live while the first is
        // unapplied — a single staging set cannot survive that (see
        // `set_decode_into_request`'s single-consumer contract).
        if let Some(client) = &mut cpu_client
            && device.is_cuda()
            && policy.is_barrier_paced()
        {
            client.set_decode_into_request(true);
        }
        match (&nccl_comm, &mut cpu_client) {
            (Some(comm), _) => {
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
                // CPU: round-trip through the reduce channel, copy_ back under
                // no_grad (in-place on a requires_grad leaf otherwise aborts).
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
                // f32 buffers only (CPU reduce transport is f32-only; non-f32
                // buffers e.g. BatchNorm's i64 num_batches_tracked are
                // deterministic counters initialized identically on every rank).
                let f32_buffers: Vec<&Tensor> = initial_buffers_local
                    .iter().filter(|b| b.dtype() == DType::Float32).collect();
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
                    "build_cluster_worker: neither NCCL comm nor CPU reduce \
                     client was constructed (backend bootstrap bug)",
                ));
            }
        }

        let initial_params: Vec<Tensor> = initial_params_local.iter()
            .map(|t| t.to_device(Device::CPU).and_then(|t| t.pin_memory()))
            .collect::<Result<Vec<_>>>()?;
        let initial_buffers: Vec<Tensor> = initial_buffers_local.iter()
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
            augment: config.augment.max(1),
            transform: config.transform.clone(),
            // On resume, reproduce the recorded epoch permutation exactly; the
            // coordinator reads the same meta, so the cohort stays consistent
            // without a broadcast.
            seed: crate::distributed::ddp_run::resolve_shuffle_seed(
                config.resume_from.as_deref(),
            )?,
            max_grad_norm: config.max_grad_norm,
            vram_pool: config.vram_pool,
            vram_max_usage: config.vram_max_usage,
            ram_max_usage: config.ram_max_usage,
            sample_cache: config.sample_cache,
            disk_stage_gb: config.disk_stage_gb,
            disk_stage_dir: config.disk_stage_dir.clone(),
            // EASGD blending is gated to Async at the single authoritative
            // point (`GpuWorker::new`); pass the configured value through.
            easgd_alpha: config.elche.easgd_alpha,
            gamma: config.elche.gamma,
            bf16_wire: config.elche.bf16_wire,
            timeline: config.timeline.clone(),
            policy,
            save_path,
            // Mirror the coord's staleness threshold so both liveness
            // directions keep one notion of "gone"; an explicit user
            // heartbeat_timeout_secs wins unscaled.
            coord_liveness_timeout_secs: config.heartbeat_timeout_secs.unwrap_or_else(
                || crate::distributed::wire::scaled_deadline_secs(
                    crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
                ),
            ),
        };

        // ClusterWorker bridges set up heartbeat + NCCL watchdog + inbound
        // (DeclareDead / NewNcclSession / ShutdownWithSave) + outbound timing.
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
                RankCallbacks {
                    checkpoint_fn,
                    epoch_fn,
                    eval_fn,
                    eval_dataset,
                    outer_optimizer_factory,
                },
            )?;

        if let Some(f) = scheduler_fn {
            cluster_worker.inner_mut().set_scheduler(f(world_size));
        }
        // Resume: re-seed this rank's replicated outer-optimizer momentum. No-op
        // when not resuming / no outer optimizer / sidecar absent; errors are
        // logged + ignored (resume from zero momentum is a safe fallback).
        if let Some(stem) = config.resume_from.as_ref() {
            if let Err(e) = cluster_worker.inner_mut().resume_outer_momentum(stem) {
                eprintln!(
                    "cluster_worker: rank {global_rank} outer-momentum resume \
                     failed ({e}); starting from zero momentum"
                );
            }
        }
        Ok(cluster_worker)
    }
}
