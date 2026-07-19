//! Cluster worker: TCP-driven wrapper around the
//! [`crate::distributed::ddp_run::GpuWorker`].
//!
//! Reuses every GpuWorker method unchanged (`train_step`,
//! `sync_now_nccl`, `load_averaged`, `run_epoch_plan`,
//! `wait_for_epoch_plan`, `report_timing`, `report_epoch`,
//! `snapshot_params`, EASGD blend, prefetch + DataLoader integration,
//! etc.). The ClusterWorker layers on top of those exactly two
//! changes:
//!
//! 1. **Connect + handshake to the cluster coordinator over TCP**,
//!    matching the handshake bytes defined by
//!    [`cluster_coordinator`](crate::distributed::cluster_coordinator).
//! 2. **Bridge the GpuWorker's mpsc channels to TCP** via two background
//!    threads (one inbound, one outbound). The inner GpuWorker still
//!    sees mpsc senders/receivers; the bridges translate to and from
//!    `ControlFrame`s on the wire.
//!
//! # Architecture
//!
//! ```text
//! rank process:
//!   ClusterWorker::connect_and_build(...)
//!     ├── TcpStream::connect(coord_addr)
//!     ├── handshake (24-byte rank → coord, 16-byte ack, both HMAC-keyed)
//!     ├── mpsc::channel() x5 (timing/metrics/param/final_param/control)
//!     ├── GpuWorker::new(... mpsc-end ...) — unchanged
//!     ├── spawn TCP→control bridge (decode ControlFrame → push ControlMsg)
//!     └── spawn timing→TCP bridge (drain TimingMsg → encode ControlFrame)
//!
//! ClusterWorker::run_until_shutdown(train_fn)
//!     loop:
//!       inner.wait_for_epoch_plan()       (blocks on control_rx)
//!       inner.run_epoch_plan(&plan, train_fn)
//!     send_final_snapshot + report_exiting
//!     join bridges
//! ```
//!
//! # Wire surface
//!
//! `ControlMsgWire` variants `SyncNow`, `Throttle`, `SetGlobalStep`,
//! `StartEpoch`, `Shutdown`, `Checkpoint`, `ExecuteEvalCallback`,
//! `ExtendPartition`, `DeclareDead`, `NewNcclSession`,
//! `RequestNewNcclId`, `Update`, `RequestParams`, and
//! `ShutdownWithSave` all flow through `control_wire_to_msg` into
//! the in-process `ControlMsg` channel (some are intercepted at the
//! inbound bridge for elastic-membership / NCCL-rebuild handling
//! rather than forwarded to the inner `GpuWorker`). `TimingMsgWire`
//! variants flow rank→coord in the opposite direction; the param
//! bridge synthesises a real `ControlMsg::Update(AveragedParams)` on
//! the CPU averaging path via `CpuReduceClient`.
//!
//! # Tests
//!
//! - CPU structural test: ClusterWorker handshakes with a real
//!   [`crate::distributed::cluster_coordinator::ClusterCoordinator`],
//!   runs a trivial CPU model + dataset through
//!   one Sync averaging cycle, exits cleanly.
//! - `#[ignore = "cuda"]` end-to-end NCCL smoke test: two ranks share a
//!   NcclRankComm, do real AllReduce(Avg) on their parameters after a
//!   few batches, verify weights converge to consensus. Runs via
//!   `fdl cuda-test-nccl` on a multi-GPU rig.

use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::wire::{read_handshake_ack, write_handshake_rank};
use crate::distributed::ddp_run::{
    ControlMsg, EpochFn, EpochPlan, GpuWorker, RankCallbacks, TimingMsg, WorkerConfig,
};
use crate::distributed::nccl::NcclRankComm;
use crate::distributed::relay::mux::{try_read_len_framed, write_len_framed, LenFramedRead};
use crate::distributed::wire::{
    ControlFrame, ControlMsgWire, MsgKind, SessionSalt,
};
use crate::distributed::wire_convert::{
    control_wire_to_msg, metrics_msg_to_wire, timing_msg_to_wire,
};
use crate::distributed::nccl_session::PendingNcclSession;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

// ---------------------------------------------------------------------------
// ClusterWorker
// ---------------------------------------------------------------------------

/// TCP-driven training worker. Wraps an inner [`GpuWorker`] with
/// bridge threads that translate between the GpuWorker's mpsc channels
/// and the control-channel `ControlFrame` wire protocol.
///
/// NOT Send (the inner [`GpuWorker`] holds `Rc<RefCell<...>>`).
/// Construct and run on the same thread.
pub struct ClusterWorker<M: Module> {
    inner: Option<GpuWorker<M>>,
    /// Background bridge thread handles. Joined on
    /// [`Self::run_until_shutdown`] cleanup.
    bridges: Vec<JoinHandle<()>>,
    /// Cooperative shutdown for bridge threads. Flipped during
    /// `run_until_shutdown` teardown.
    shutdown_flag: Arc<AtomicBool>,
    /// Per-worker dead-rank ledger. Updated by the inbound bridge when
    /// `ControlMsgWire::DeclareDead` arrives; polled by the NCCL
    /// watchdog thread to trigger `NcclAbortHandle::abort` on the
    /// local comm. Distinct from the coord-side `Arc<DeadRanks>`
    /// because workers in different processes can't share Arcs with
    /// the coord — each worker holds its own copy and the coord
    /// drives state via the wire.
    #[allow(dead_code)]
    local_dead_ranks: Arc<crate::distributed::controller::DeadRanks>,
    /// Pending NCCL session mailbox. Inbound bridge stores the most
    /// recent `NewNcclSession` payload from the coord; the main-thread
    /// comm-rebuild path inside `sync_now_nccl::wait_for_nccl_session`
    /// takes from this slot to rebuild the comm on the survivor
    /// cohort after a peer-death abort.
    #[allow(dead_code)]
    nccl_session_mailbox: Arc<std::sync::Mutex<Option<PendingNcclSession>>>,
    /// User-supplied per-epoch callback. Populated only on the rank
    /// chosen by [`crate::distributed::ddp_run::EpochCallbackPolicy`];
    /// non-chosen ranks receive `None` so they skip the firing cost.
    /// Invoked from [`Self::run_until_shutdown`] between
    /// `wait_for_epoch_plan` and `run_epoch_plan` on each epoch
    /// transition.
    epoch_fn: Option<EpochFn<M>>,
    /// Receiver for the final parameter snapshot the inner GpuWorker
    /// emits via [`crate::distributed::ddp_run::GpuWorker::send_final_snapshot`]
    /// at end-of-training. Taken in `run_until_shutdown` and drained
    /// after the snapshot send so the returned [`crate::distributed::ddp_run::ParamSnapshot`]
    /// can be ferried up the via_coord stack into a `TrainedState`.
    /// Replaces the prior discard bridge — the receiver is now drained
    /// on the calling thread instead of consumed by a background thread.
    final_param_rx: Option<mpsc::Receiver<crate::distributed::ddp_run::ParamSnapshot>>,
}

impl<M: Module + 'static> ClusterWorker<M> {
    /// Connect to the cluster coordinator at `coord_addr`, complete the
    /// handshake (validated with the shared `salt`), construct an inner
    /// [`GpuWorker`] with the provided model/optimizer/dataset/NCCL
    /// communicator, and spawn the mpsc↔TCP bridge threads.
    ///
    /// On error any partially-set-up resources are cleaned up (stream
    /// dropped, mpsc channels dropped, no leaked threads).
    ///
    /// All `Send` ingredients must be passed in; the closures run on
    /// the spawning thread because `GpuWorker<M>` is not `Send`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn connect_and_build<F, G, O>(
        coord_addr: SocketAddr,
        cpu_client: Option<crate::distributed::cpu_reduce::CpuReduceClient>,
        rank_id: u32,
        salt: SessionSalt,
        config: WorkerConfig,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        nccl_comm: Option<NcclRankComm>,
        rank_callbacks: RankCallbacks<M>,
    ) -> Result<Self>
    where
        F: FnOnce(Device) -> Result<M>,
        G: FnOnce(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        let RankCallbacks {
            checkpoint_fn,
            epoch_fn,
            eval_fn,
            eval_dataset,
            outer_optimizer_factory,
        } = rank_callbacks;
        if rank_id as usize >= config.world_size {
            return Err(TensorError::new(&format!(
                "cluster_worker: rank_id {rank_id} >= world_size {}",
                config.world_size,
            )));
        }

        // Ranks dial their host-local relay's control loopback. The relay
        // process may bind a beat after the rank starts (launcher spawns
        // both), so retry briefly rather than fail on the first refusal.
        let stream = crate::distributed::wire::connect_with_retry(coord_addr, "cluster_worker coord")?;
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| {
                TensorError::new(&format!("cluster_worker: set_read_timeout: {e}"))
            })?;
        // Write-stall ceiling (fd-level, covers every cloned handle): a
        // wedged relay errors the outbound bridge and the heartbeat
        // emitter instead of parking them — the bridges then unwind and
        // the rank exits rather than hanging silently.
        stream
            .set_write_timeout(Some(crate::distributed::wire::write_stall_timeout()))
            .map_err(|e| {
                TensorError::new(&format!("cluster_worker: set_write_timeout: {e}"))
            })?;

        // Two independent stream handles so the inbound reader and the
        // outbound writer can sit on different threads without
        // contending on a single OS file descriptor.
        let mut handshake_stream = stream;
        write_handshake_rank(
            &mut handshake_stream,
            rank_id,
            config.world_size as u32,
            &salt,
        )?;
        read_handshake_ack(&mut handshake_stream, &salt)?;
        // Clear the handshake timeout; per-frame waits can run long.
        handshake_stream
            .set_read_timeout(Some(Duration::from_millis(250)))
            .map_err(|e| {
                TensorError::new(&format!("cluster_worker: set_read_timeout: {e}"))
            })?;

        let read_stream = handshake_stream;
        let mut write_stream = read_stream.try_clone().map_err(|e| {
            TensorError::new(&format!(
                "cluster_worker: stream try_clone for bridge split: {e}"
            ))
        })?;
        // NOTE: do NOT touch read_timeout on the write half. try_clone
        // dups the fd and SO_RCVTIMEO lives on the SHARED socket, so a
        // `set_read_timeout(None)` here silently clears the 250ms timeout
        // set on the read half above — the inbound loop's shutdown-flag
        // poll then never fires while idle and bridge teardown hangs on
        // the coordinator closing the socket. Writes never read; they use
        // TCP send-buffer back-pressure, not timeouts.
        // Single write handle, single outbound thread: both `timing_rx`
        // and `metrics_rx` drain through one drainer that owns the
        // socket. Earlier revisions split this into two bridges
        // (timing + metrics) sharing the socket via a second try_clone,
        // on the assumption that per-handle TCP write atomicity would
        // serialize the frames. That assumption is wrong — multi-byte
        // write() syscalls from different threads against the same
        // kernel socket can interleave bytes mid-frame, and a frame
        // whose header + payload come from two different sources fails
        // HMAC verification on the reader, exits the coord's reader
        // thread, and silently kills the rank (heartbeat then goes
        // stale at 30s and the rank is declared dead). Surfaced first
        // on cpu-async (progressive dispatch → frequent MetricsMsg
        // collides with per-batch timing frames). Single-writer
        // discipline is the structural fix.

        // mpsc quintet — the worker-side senders flow into GpuWorker,
        // the coord-side ends stay with the bridges. Clone the senders
        // that bridges need access to (timing_tx for the SyncAck
        // emitted after a CPU-averaging round, control_tx for the
        // param bridge's synthesized ControlMsg::Update).
        let (timing_tx, timing_rx) = mpsc::channel::<TimingMsg>();
        let timing_tx_for_param_bridge = timing_tx.clone();
        let timing_tx_for_heartbeat = timing_tx.clone();
        let timing_tx_for_inbound = timing_tx.clone();
        let (metrics_tx, metrics_rx) = mpsc::channel::<crate::distributed::ddp_run::MetricsMsg>();
        let (param_tx, param_rx) =
            mpsc::channel::<crate::distributed::ddp_run::ParamSnapshot>();
        let (final_param_tx, final_param_rx) =
            mpsc::channel::<crate::distributed::ddp_run::ParamSnapshot>();
        let (control_tx, control_rx) = mpsc::channel::<ControlMsg>();
        let control_tx_for_param_bridge = control_tx.clone();

        // CpuReduceClient on the data channel, used by the param
        // bridge below when AverageBackend::Cpu is in play. `None` when
        // the worker is in NCCL-only mode. Caller-built so the same
        // client can be used for an initial broadcast on the spawning
        // thread before being handed off here (the controller's accept
        // loop is one-shot — a connect/disconnect/reconnect dance would
        // fail handshake).

        // Worker-local dead-rank ledger + NCCL session mailbox.
        // Constructed BEFORE inner so the NCCL watchdog thread (below)
        // can take an Arc clone. The inbound bridge populates both
        // when the coord broadcasts `DeclareDead` / `NewNcclSession`.
        let local_dead_ranks =
            crate::distributed::controller::DeadRanks::new(config.world_size);
        let nccl_session_mailbox: Arc<std::sync::Mutex<Option<PendingNcclSession>>> =
            Arc::new(std::sync::Mutex::new(None));

        // Build this rank's replicated outer optimizer from the per-site
        // factory (once per rank). `None` = plain averaging.
        let outer_optimizer = outer_optimizer_factory.as_ref().map(|f| f());
        let mut inner = GpuWorker::<M>::new(
            &config,
            model_factory,
            optim_factory,
            dataset,
            nccl_comm,
            checkpoint_fn,
            eval_fn,
            eval_dataset,
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
            control_rx,
            outer_optimizer,
        )?;

        // Attach the cluster-mode NCCL session mailbox so the inner's
        // `sync_now_nccl` retry path can read new comm bytes after a
        // peer-death abort. The inbound bridge populates the slot on
        // each `NewNcclSession` arrival from the coord.
        inner.attach_nccl_session_mailbox(Arc::clone(&nccl_session_mailbox));
        // Attach the local dead-rank ledger so the inner's
        // `wait_for_nccl_session` can short-circuit when this rank is
        // the lone NCCL survivor (no peer to rendezvous with). Without
        // this, the worker waits 60s for a `NewNcclSession` that the
        // coord will never send.
        inner.attach_local_dead_ranks(Arc::clone(&local_dead_ranks));

        // NCCL abort slot: a shared cell the watchdog re-reads on every
        // poll and `GpuWorker::replace_nccl_comm` refreshes on every
        // rebuild, so cascading deaths (a second peer dying after the
        // cohort already rebuilt once) always abort the LIVE comm — a
        // captured handle would go stale at the first rebuild. Cluster
        // mode without an NCCL comm (CPU averaging) leaves it `None`
        // and skips the watchdog entirely.
        let nccl_abort_slot: crate::distributed::ddp_run::NcclAbortSlot =
            Arc::new(std::sync::Mutex::new(inner.nccl_abort_handle()));
        inner.attach_nccl_abort_slot(Arc::clone(&nccl_abort_slot));

        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let mut bridges: Vec<JoinHandle<()>> = Vec::new();

        // Inbound bridge: TCP ControlFrame → ControlMsg → control_tx.
        // Intercepts the elastic-membership control frames (DeclareDead,
        // NewNcclSession, RequestNewNcclId) and routes them to the
        // local ledger / mailbox / UID reply path rather than the inner
        // GpuWorker — the inner is typically blocked in an in-flight
        // NCCL collective and can't service control messages until the
        // watchdog aborts the comm.
        let salt_in = salt;
        let shutdown_in = Arc::clone(&shutdown_flag);
        let rank_in = config.rank;
        let coord_liveness_timeout_in = config.coord_liveness_timeout_secs;
        let mut read_stream_for_bridge = read_stream;
        let dead_for_inbound = Arc::clone(&local_dead_ranks);
        let mailbox_for_inbound = Arc::clone(&nccl_session_mailbox);
        bridges.push(
            thread::Builder::new()
                .name(format!("flodl-worker-inbound:r{rank_in}"))
                .spawn(move || {
                    inbound_loop(
                        rank_in,
                        &mut read_stream_for_bridge,
                        &salt_in,
                        &shutdown_in,
                        &control_tx,
                        &dead_for_inbound,
                        &mailbox_for_inbound,
                        &timing_tx_for_inbound,
                        coord_liveness_timeout_in,
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn inbound bridge for rank {rank_in}: {e}"
                    ))
                })?,
        );

        // Outbound bridge: drains both `timing_rx` (per-batch Timing
        // frames + heartbeats + SyncAcks) and `metrics_rx` (per-chunk
        // / per-epoch Metrics frames) and writes each as a
        // ControlFrame to the coordinator. Single thread, single
        // socket handle → frame writes are serialized by construction;
        // no concurrent-write race on the kernel socket.
        let salt_out = salt;
        let shutdown_out = Arc::clone(&shutdown_flag);
        let rank_out = config.rank;
        bridges.push(
            thread::Builder::new()
                .name(format!("flodl-worker-outbound:r{rank_out}"))
                .spawn(move || {
                    outbound_loop(
                        rank_out,
                        &mut write_stream,
                        &salt_out,
                        &shutdown_out,
                        timing_rx,
                        metrics_rx,
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn outbound bridge for rank {rank_out}: {e}"
                    ))
                })?,
        );
        // Param bridge: receives ParamSnapshot from the inner GpuWorker
        // (triggered by ControlMsg::RequestParams), runs an all-reduce
        // round-trip through the data channel via CpuReduceClient, and
        // synthesizes a real ControlMsg::Update(AveragedParams) back to
        // the inner so it can call load_averaged unchanged.
        //
        // When data_addr was not provided, this stays a discard bridge
        // (NCCL-only worker layout — the inner never emits ParamSnapshot
        // in that mode either, so the receiver simply idles).
        let rank_for_bridge = rank_id as u64;
        let gamma_for_bridge = config.gamma;
        bridges.push(
            thread::Builder::new()
                .name(format!("flodl-worker-param-bridge:r{rank_out}"))
                .spawn(move || {
                    param_bridge_loop(
                        rank_for_bridge,
                        param_rx,
                        cpu_client,
                        control_tx_for_param_bridge,
                        timing_tx_for_param_bridge,
                        gamma_for_bridge,
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn param bridge: {e}"
                    ))
                })?,
        );
        // `final_param_rx` is parked on `ClusterWorker` (instead of being
        // drained by a background bridge) so `run_until_shutdown` can
        // ferry the inner's end-of-training `send_final_snapshot()` back
        // up the call stack into a real `TrainedState` rather than
        // silently discarding it.
        let final_param_rx_for_handle = Some(final_param_rx);
        // Heartbeat thread: fires at HEARTBEAT_CADENCE_MS so the coord
        // can distinguish "rank alive but blocked at the AllReduce
        // barrier" from "rank dead." The thread is independent of the
        // training loop, so a wedged inner GpuWorker still produces
        // heartbeats (training will stall but cluster doesn't think
        // the rank is dead — operations signal). Stops on shutdown_flag.
        let shutdown_for_hb = Arc::clone(&shutdown_flag);
        let rank_for_hb = rank_id as usize;
        bridges.push(
            thread::Builder::new()
                .name(format!("flodl-worker-heartbeat:r{rank_out}"))
                .spawn(move || {
                    heartbeat_loop(
                        rank_for_hb,
                        timing_tx_for_heartbeat,
                        shutdown_for_hb,
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn heartbeat thread: {e}"
                    ))
                })?,
        );

        // NCCL watchdog: poll the local ledger for newly-dead peers and
        // abort the in-flight NCCL collective so the main thread's
        // `sync_now_nccl` can break out of the AllReduce barrier and
        // rebuild the comm on the survivor cohort. Only spawned when an
        // NCCL comm exists — CPU averaging doesn't need this (the
        // coord's shared DeadRanks ledger shuts down the controller
        // stream which already releases the blocked AllReduce read).
        //
        // The watchdog reads the CURRENT handle from the shared slot on
        // every poll, so a comm rebuild re-arms it automatically: death A
        // aborts comm 1, the survivors rebuild comm 2 (refreshing the
        // slot), death B aborts comm 2. Per-handle `aborted` flags keep
        // each abort idempotent per comm lifetime.
        let spawn_watchdog = nccl_abort_slot
            .lock()
            .expect("nccl abort slot poisoned")
            .is_some();
        if spawn_watchdog {
            let shutdown_for_wd = Arc::clone(&shutdown_flag);
            let dead_for_wd = Arc::clone(&local_dead_ranks);
            let slot_for_wd = Arc::clone(&nccl_abort_slot);
            let rank_for_wd = rank_id as usize;
            bridges.push(
                thread::Builder::new()
                    .name(format!("flodl-worker-nccl-watchdog:r{rank_out}"))
                    .spawn(move || {
                        nccl_watchdog_loop(
                            rank_for_wd,
                            slot_for_wd,
                            dead_for_wd,
                            shutdown_for_wd,
                        );
                    })
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster_worker: spawn NCCL watchdog: {e}"
                        ))
                    })?,
            );
        }

        Ok(ClusterWorker {
            inner: Some(inner),
            bridges,
            shutdown_flag,
            local_dead_ranks,
            nccl_session_mailbox,
            epoch_fn,
            final_param_rx: final_param_rx_for_handle,
        })
    }

    /// Borrow the inner [`GpuWorker`] for direct method calls (rank,
    /// device, scheduler attachment, etc.). Used by callers that need
    /// to configure the worker between construction and the main loop.
    pub fn inner(&self) -> &GpuWorker<M> {
        self.inner
            .as_ref()
            .expect("inner GpuWorker present until run_until_shutdown drops it")
    }

    /// Mutable borrow of the inner [`GpuWorker`].
    pub fn inner_mut(&mut self) -> &mut GpuWorker<M> {
        self.inner
            .as_mut()
            .expect("inner GpuWorker present until run_until_shutdown drops it")
    }

    /// Drive the worker's main loop until the coordinator sends
    /// Shutdown or the control channel disconnects. Mirrors the OLD
    /// coordinator-driven `run_epoch_plan` cycle:
    ///
    /// ```text
    /// loop {
    ///   plan = wait_for_epoch_plan()   // blocks on control_rx
    ///   if shutdown { break }
    ///   shutdown = run_epoch_plan(&plan, train_fn)
    ///   if shutdown { break }
    /// }
    /// abort_nccl + send_final_snapshot + report_exiting
    /// drain bridges and exit
    /// ```
    ///
    /// On exit, the inner GpuWorker is dropped (causing the timing /
    /// metrics / param channel senders to disconnect), the shutdown
    /// flag is flipped to signal the bridges, and all bridge threads
    /// are joined.
    ///
    /// Returns the rank's end-of-training [`crate::distributed::ddp_run::ParamSnapshot`]
    /// when one was successfully received from the inner GpuWorker before
    /// teardown; `None` when the inner errored out before reaching
    /// `send_final_snapshot` (best-effort: snapshot is opportunistic, not
    /// load-bearing for the shutdown path). The via_coord callers in
    /// `orchestrator.rs` convert it into a `TrainedState`.
    pub fn run_until_shutdown<T>(
        mut self,
        train_fn: T,
    ) -> Result<Option<crate::distributed::ddp_run::ParamSnapshot>>
    where
        T: Fn(&M, &[Tensor]) -> Result<Variable>,
    {
        // Inner is set in connect_and_build; only `run_until_shutdown`
        // takes it out. Unwrap is safe here.
        let mut inner = self
            .inner
            .take()
            .expect("inner GpuWorker present at run_until_shutdown");

        // Controller-driven role assignment: every cluster worker can
        // have `epoch_fn = Some(...)` regardless of policy. The runtime
        // gate is `inner.epoch_callback_role() == Some(inner.rank())`,
        // set by the coord's wire-pushed `ControlMsg::SetEpochCallbackRole`.
        // Workers without the role skip the fire; on
        // `EpochCallbackPolicy::Fastest` re-resolve (e.g. after rank
        // death), the coord broadcasts a fresh role and the worker
        // picks it up before the next epoch boundary.
        // Move epoch_fn out of `self` so the loop body can borrow it
        // without colliding with `self.bridges` teardown below.
        let epoch_fn = self.epoch_fn.take();
        // `usize::MAX` sentinel so the first plan (epoch 0) always
        // triggers a fire-check.
        let mut last_epoch_fired: usize = usize::MAX;

        // Instrumentation (gated on `-vvv` via `inner.prof_enabled()`):
        // split each rank's wall into time in `run_epoch_plan` (compute +
        // data + in-chunk control) vs blocked in `wait_for_epoch_plan`
        // (reduce-barrier / next-dispatch wait). Teardown stderr summary
        // separates "compute" from "waiting at the barrier" to locate the
        // cpu-cadence idle. All collection below is `if prof`-guarded.
        let prof = inner.prof_enabled();
        let mut wait_ns: u128 = 0;
        let mut run_ns: u128 = 0;
        // Split the between-chunk wait at the moment averaged params land
        // (`load_averaged`): pre-Update = blocked waiting for the reduce
        // to complete (which can't trigger until the slow ranks finish
        // their compute window); post-Update = the dispatch wait from
        // weights-applied to the next StartEpoch (atomic-dispatch should
        // drive this to ~0). Tests whether the cpu idle is slow-rank /
        // reduce wait vs a dispatch round-trip.
        let mut wait_pre_update_ns: u128 = 0;
        let mut wait_post_update_ns: u128 = 0;

        let exit_clean = (|| -> Result<bool> {
            loop {
                let prev_update = if prof { inner.last_update_at() } else { None };
                let w0 = std::time::Instant::now();
                let plan = inner.wait_for_epoch_plan()?;
                if prof {
                    let w_elapsed = w0.elapsed().as_nanos();
                    wait_ns += w_elapsed;
                    // If an Update landed during this wait, split at it.
                    match inner.last_update_at() {
                        Some(u) if Some(u) != prev_update && u >= w0 => {
                            let pre = u.duration_since(w0).as_nanos();
                            wait_pre_update_ns += pre;
                            wait_post_update_ns += w_elapsed.saturating_sub(pre);
                        }
                        _ => wait_pre_update_ns += w_elapsed,
                    }
                }
                match plan {
                    Some(plan) => {
                        // Fire `epoch_fn` once per epoch transition (not
                        // per-chunk in progressive dispatch). Sync-aligned
                        // by construction: `StartEpoch` arrives after the
                        // controller's `finish_averaging_*` completes the
                        // prior cycle's bookkeeping.
                        if plan.epoch != last_epoch_fired {
                            last_epoch_fired = plan.epoch;
                            let is_role = inner.epoch_callback_role()
                                == Some(inner.rank());
                            if is_role {
                                if let Some(ref f) = epoch_fn {
                                    let start = std::time::Instant::now();
                                    f(plan.epoch, &mut inner);
                                    let elapsed_ms =
                                        start.elapsed().as_secs_f64() * 1000.0;
                                    inner.report_epoch_fn_elapsed(
                                        plan.epoch,
                                        elapsed_ms,
                                    );
                                }
                            }
                        }
                        let r0 = std::time::Instant::now();
                        let shutdown = inner.run_epoch_plan(&plan, &train_fn)?;
                        if prof {
                            run_ns += r0.elapsed().as_nanos();
                        }
                        if shutdown {
                            return Ok(true);
                        }
                    }
                    None => return Ok(true),
                }
            }
        })();

        // Per-rank compute-vs-barrier-wait summary (stderr, `-vvv` only),
        // with the wait split into pre-Update (reduce + slow-rank wait)
        // vs post-Update (dispatch wait) + the GPU←CPU writeback cost.
        if prof {
            let run_s = run_ns as f64 / 1e9;
            let wait_s = wait_ns as f64 / 1e9;
            let pre_s = wait_pre_update_ns as f64 / 1e9;
            let post_s = wait_post_update_ns as f64 / 1e9;
            let h2d_s = inner.h2d_wait_ms_total() / 1e3;
            // GPU→CPU snapshot readout (the per-window weight publish);
            // counted inside run_epoch, so its share of run_s names how
            // much of "compute" is actually the synchronous D2H readout.
            let snap_s = inner.snapshot_readout_ms_total() / 1e3;
            let snap_n = inner.snapshot_readout_count();
            let snap_per = if snap_n > 0 { snap_s / snap_n as f64 } else { 0.0 };
            // run_epoch split: compute + data are what ElChe's
            // share_complete_ms sees; `other` (= run_epoch - compute -
            // data) is the ctrl/sync/transport overhead it does NOT —
            // the suspected cpu-vs-nccl allocation-blind gap.
            let compute_s = inner.compute_ms_run_total() / 1e3;
            let data_s = inner.data_ms_run_total() / 1e3;
            let other_s = (run_s - compute_s - data_s).max(0.0);
            let ctrl_msgs = inner.ctrl_msgs_handled();
            let tot = (run_s + wait_s).max(1e-9);
            eprintln!(
                "[worker-prof] rank={} run_epoch={:.1}s ({:.0}%) wait={:.1}s ({:.0}%) \
                 | run-split: compute={:.1}s data={:.1}s other(ctrl/sync)={:.1}s \
                 | wait-split: pre_update(reduce+slowrank)={:.1}s post_update(dispatch)={:.2}s \
                 | h2d_writeback={:.2}s | snapshot_readout={:.1}s ({} calls, {:.0}ms/call) \
                 | ctrl_msgs={}",
                inner.rank(),
                run_s,
                100.0 * run_s / tot,
                wait_s,
                100.0 * wait_s / tot,
                compute_s,
                data_s,
                other_s,
                pre_s,
                post_s,
                h2d_s,
                snap_s,
                snap_n,
                snap_per * 1e3,
                ctrl_msgs,
            );
        }

        // On error exit (e.g. lone NCCL survivor bailing out of
        // `wait_for_nccl_session`), the coord may have queued
        // `Shutdown` / `ShutdownWithSave` frames in `control_rx` that
        // never reached `handle_control` because the main loop already
        // returned. Drain those now so the rank-side checkpoint bundle
        // gets written before we exit (the controller-side
        // `.meta.json` was already written by `dispatch_shutdown_with_save`).
        // Clean-exit paths see no queued shutdown and this is a no-op.
        let _ = inner.drain_pending_shutdown();

        // Abort NCCL before the snapshot: a pending AllReduce from a
        // SyncNow whose peer died would block snapshot_params' stream
        // synchronize forever (the error-exit path can get here with a
        // collective still in flight and no DeclareDead ever arriving —
        // e.g. the coord died with the peer). Idempotent with the
        // watchdog's abort.
        inner.abort_nccl();

        // Even on error, try to gracefully report exit + drop senders
        // so the coordinator side cleans up. send_final_snapshot uses
        // the dedicated final_param channel; the receiver now lives on
        // `self.final_param_rx` (no background discard bridge), so the
        // send + receive happen sequentially on this thread.
        // report_exiting goes through the outbound bridge.
        inner.send_final_snapshot();
        inner.report_exiting();

        // Drain the final snapshot before dropping inner (otherwise the
        // mpsc Sender disconnects and the recv() races with the channel
        // emptying). `try_recv` first to pick up the just-sent value;
        // fall back to a short recv_timeout to catch any rare scheduler
        // delay between send_final_snapshot and the receiver becoming
        // ready. Best-effort: an error from snapshot_params (e.g. CUDA
        // pinned-memory failure inside send_final_snapshot) leaves the
        // channel empty and we surface `None` up to the caller.
        let final_snapshot = self.final_param_rx.take().and_then(|rx| {
            match rx.try_recv() {
                Ok(snap) => Some(snap),
                Err(mpsc::TryRecvError::Empty) => rx
                    .recv_timeout(std::time::Duration::from_millis(500))
                    .ok(),
                Err(mpsc::TryRecvError::Disconnected) => None,
            }
        });

        // Drop inner → all mpsc::Sender clones held by the inner
        // disconnect → bridges see Disconnected on their Receivers and
        // exit naturally. The shutdown_flag is a belt-and-suspenders
        // signal for the inbound bridge (it has no inner-side sender
        // to disconnect; it reads from TCP and sends INTO control_tx,
        // which we drop here too via the inner).
        drop(inner);
        self.shutdown_flag.store(true, Ordering::SeqCst);
        for handle in self.bridges.drain(..) {
            let _ = handle.join();
        }

        exit_clean.map(|_| final_snapshot)
    }
}

impl<M: Module> Drop for ClusterWorker<M> {
    fn drop(&mut self) {
        // Best-effort if run_until_shutdown wasn't called. The inner
        // GpuWorker is dropped here; bridges then see disconnect.
        self.shutdown_flag.store(true, Ordering::SeqCst);
        self.inner.take();
        for handle in self.bridges.drain(..) {
            let _ = handle.join();
        }
    }
}


mod bridges;
mod param_bridge;
// Re-exported only for the CPU-reduce integration test that drives
// `sumcount_reduce` directly; production callers are inside param_bridge.
#[cfg(test)]
pub(crate) use param_bridge::sumcount_reduce;
use bridges::*;
use param_bridge::*;

#[cfg(test)]
#[path = "../cluster_worker_tests.rs"]
mod tests;
