//! Cluster worker: TCP-driven wrapper around the OLD threaded
//! [`crate::distributed::ddp_run::GpuWorker`].
//!
//! Reuses every OLD GpuWorker method unchanged (`train_step`,
//! `sync_now_nccl`, `load_averaged`, `run_epoch_plan`,
//! `wait_for_epoch_plan`, `report_timing`, `report_epoch`,
//! `snapshot_params`, EASGD blend, prefetch + DataLoader integration,
//! etc.). The ClusterWorker layers on top of those exactly two
//! changes:
//!
//! 1. **Connect + handshake to the cluster coordinator over TCP**,
//!    matching the handshake bytes defined by
//!    [`cluster_coordinator`](crate::distributed::cluster_coordinator).
//! 2. **Bridge the OLD mpsc channels to TCP** via two background
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
    CheckpointFn, ControlMsg, EpochFn, EpochPlan, EvalFn, GpuWorker, TimingMsg, WorkerConfig,
};
use crate::distributed::nccl::NcclRankComm;
use crate::distributed::relay::mux::{try_read_len_framed, write_len_framed, LenFramedRead};
use crate::distributed::wire::{
    ControlFrame, ControlMsgWire, MsgKind, SessionSalt,
    TimingMsgWire,
};
#[cfg(test)]
use crate::distributed::wire::EpochPlanWire;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

// ---------------------------------------------------------------------------
// Bridge helpers (mpsc ↔ TCP)
// ---------------------------------------------------------------------------

/// Convert an in-process `TimingMsg` into the bincode-serializable
/// [`TimingMsgWire`] for transit over the TCP control channel.
fn timing_msg_to_wire(msg: TimingMsg) -> TimingMsgWire {
    match msg {
        TimingMsg::Batch {
            rank,
            batch_ms,
            data_ms,
            step_count,
            param_norm,
            batch_loss,
            sync_divergence,
        } => TimingMsgWire::Batch {
            rank: rank as u64,
            batch_ms,
            data_ms,
            step_count: step_count as u64,
            param_norm,
            batch_loss,
            sync_divergence,
        },
        TimingMsg::SyncAck {
            rank,
            step_count,
            divergence,
            post_norm,
            pre_norm,
        } => TimingMsgWire::SyncAck {
            rank: rank as u64,
            step_count: step_count as u64,
            divergence,
            post_norm,
            pre_norm,
        },
        TimingMsg::Exiting { rank } => TimingMsgWire::Exiting {
            rank: rank as u64,
        },
        TimingMsg::LrUpdate { rank, lr } => TimingMsgWire::LrUpdate {
            rank: rank as u64,
            lr,
        },
        TimingMsg::Heartbeat { rank, step_count } => TimingMsgWire::Heartbeat {
            rank: rank as u64,
            step_count: step_count as u64,
        },
        TimingMsg::SnapshotReady { rank } => TimingMsgWire::SnapshotReady {
            rank: rank as u64,
        },
        TimingMsg::NewNcclIdGenerated { rank, uid_bytes } => {
            TimingMsgWire::NewNcclIdGenerated {
                rank: rank as u64,
                uid_bytes,
            }
        }
        TimingMsg::EvalResult {
            rank,
            schedule_id,
            epoch,
            metric,
            elapsed_ms,
            error,
        } => TimingMsgWire::EvalResult {
            rank: rank as u64,
            schedule_id,
            epoch,
            metric,
            elapsed_ms,
            error,
        },
        TimingMsg::CheckpointResult {
            rank,
            version,
            elapsed_ms,
            error,
        } => TimingMsgWire::CheckpointResult {
            rank: rank as u64,
            version,
            elapsed_ms,
            error,
        },
        TimingMsg::EpochFnElapsed {
            rank,
            epoch,
            elapsed_ms,
        } => TimingMsgWire::EpochFnElapsed {
            rank: rank as u64,
            epoch: epoch as u64,
            elapsed_ms,
        },
    }
}

/// Convert an inbound [`ControlMsgWire`] from the coordinator into an
/// optional in-process `ControlMsg` for [`GpuWorker::dispatch_control`].
///
/// Returns `Ok(None)` for wire variants that don't need in-process
/// dispatch:
///
/// - `ControlMsgWire::Update { version, next_plan }`: the wire-side
///   notification that the averaging cycle is complete. The real
///   in-process `ControlMsg::Update(AveragedParams)` flows through the
///   param bridge (where the param bridge synthesizes one with the
///   actual averaged tensors from the data channel), so the wire-Update
///   is informational here. Its atomic-dispatch `next_plan` (when
///   `Some`) is consumed at the inbound-bridge call site, which
///   synthesises a `StartEpoch` for the inner; it is not handled by
///   this function.
///
/// All other wire variants map 1:1.
fn control_wire_to_msg(wire: ControlMsgWire) -> Result<Option<ControlMsg>> {
    match wire {
        ControlMsgWire::RequestParams => Ok(Some(ControlMsg::RequestParams)),
        // Informational on its own; the param bridge drives the real
        // `ControlMsg::Update(AveragedParams)`. The atomic-dispatch
        // `next_plan` is handled at the inbound-bridge call site (it
        // synthesises a `StartEpoch` there), so it never reaches here in
        // production; ignored for the rare direct callers (tests).
        ControlMsgWire::Update { .. } => Ok(None),
        ControlMsgWire::SyncNow => Ok(Some(ControlMsg::SyncNow)),
        ControlMsgWire::StartEpoch(plan) => Ok(Some(ControlMsg::StartEpoch(EpochPlan {
            epoch: plan.epoch as usize,
            partition_offset: plan.partition_offset as usize,
            partition_size: plan.partition_size as usize,
        }))),
        ControlMsgWire::ExtendPartition {
            partition_offset,
            partition_size,
        } => Ok(Some(ControlMsg::ExtendPartition {
            partition_offset: partition_offset as usize,
            partition_size: partition_size as usize,
        })),
        ControlMsgWire::DeclareDead { .. } => Ok(Some(ControlMsg::DeclareDead)),
        ControlMsgWire::NewNcclSession { .. } => Ok(Some(ControlMsg::NewNcclSession)),
        ControlMsgWire::RequestNewNcclId => Ok(Some(ControlMsg::RequestNewNcclId)),
        ControlMsgWire::StageAdvisory { counts, segments } => {
            Ok(Some(ControlMsg::StageAdvisory {
                counts: counts.into_iter().map(|c| c as usize).collect(),
                segments: segments
                    .into_iter()
                    .map(|(epoch, spans)| {
                        (
                            epoch as usize,
                            spans
                                .into_iter()
                                .map(|(o, s)| (o as usize, s as usize))
                                .collect(),
                        )
                    })
                    .collect(),
            }))
        }
        ControlMsgWire::Throttle => Ok(Some(ControlMsg::Throttle)),
        ControlMsgWire::SetGlobalStep { global_step } => {
            Ok(Some(ControlMsg::SetGlobalStep(global_step as usize)))
        }
        ControlMsgWire::Checkpoint { version, target_rank } => {
            // `u64::MAX` is reserved for v2 controller-as-checkpointer
            // (CPU-async mode where the coord holds the canonical
            // averaged tensors). In v1 the coord must never dispatch
            // it; if a buggy/future coord does, surface loudly so we
            // don't silently fall through to "no-op for every rank".
            if target_rank == u64::MAX {
                return Err(crate::tensor::TensorError::new(
                    "cluster_worker: Checkpoint target_rank=u64::MAX is reserved \
                     for controller-as-checkpointer (v2); v1 must dispatch to a \
                     worker rank ID",
                ));
            }
            Ok(Some(ControlMsg::Checkpoint {
                version,
                target_rank: target_rank as usize,
            }))
        }
        ControlMsgWire::ExecuteEvalCallback {
            schedule_id,
            epoch,
            target_rank,
        } => {
            if target_rank == u64::MAX {
                return Err(crate::tensor::TensorError::new(
                    "cluster_worker: ExecuteEvalCallback target_rank=u64::MAX \
                     is reserved (controller-as-evaluator, future); v1 must \
                     dispatch to a worker rank ID",
                ));
            }
            Ok(Some(ControlMsg::ExecuteEvalCallback {
                schedule_id,
                epoch,
                target_rank: target_rank as usize,
            }))
        }
        ControlMsgWire::SetEpochCallbackRole { rank } => {
            Ok(Some(ControlMsg::SetEpochCallbackRole {
                rank: rank as usize,
            }))
        }
        ControlMsgWire::Shutdown => Ok(Some(ControlMsg::Shutdown)),
        ControlMsgWire::ShutdownWithSave { reason } => {
            // Forward-compat: unknown reason byte falls back to
            // GracefulShutdown so a newer coord doesn't crash older
            // workers. The save still happens; only the recorded
            // reason loses fidelity.
            let reason = crate::distributed::checkpoint_meta::SaveReason::from_u8(reason)
                .unwrap_or(crate::distributed::checkpoint_meta::SaveReason::GracefulShutdown);
            Ok(Some(ControlMsg::ShutdownWithSave { reason }))
        }
        ControlMsgWire::EpochAggregated(metrics_wire) => {
            Ok(Some(ControlMsg::EpochAggregated(metrics_wire.into())))
        }
        ControlMsgWire::SaveConsensusModel { target_rank } => {
            if target_rank == u64::MAX {
                return Err(crate::tensor::TensorError::new(
                    "cluster_worker: SaveConsensusModel target_rank=u64::MAX is \
                     reserved; the coordinator must dispatch to a worker rank ID",
                ));
            }
            Ok(Some(ControlMsg::SaveConsensusModel {
                target_rank: target_rank as usize,
            }))
        }
        // Pure liveness beacon: the inbound bridge intercepts it before this
        // point (resetting its coord-liveness deadline) and never forwards it
        // to the inner worker. Reached only by direct callers (tests); no
        // inner dispatch, like `Update { .. }`.
        ControlMsgWire::CoordHeartbeat => Ok(None),
    }
}

/// Inverse of [`control_wire_to_msg`], used only for diagnostic
/// echoing in tests (workers don't normally send ControlMsg outbound).
#[cfg(test)]
fn _epoch_plan_to_wire(plan: EpochPlan) -> EpochPlanWire {
    EpochPlanWire {
        epoch: plan.epoch as u64,
        partition_offset: plan.partition_offset as u64,
        partition_size: plan.partition_size as u64,
    }
}

// ---------------------------------------------------------------------------
// ClusterWorker
// ---------------------------------------------------------------------------

/// TCP-driven training worker. Wraps an inner [`GpuWorker`] with
/// bridge threads that translate between the OLD mpsc channels and
/// the new control-channel `ControlFrame` wire protocol.
///
/// Mailbox slot for the most-recent coord-broadcast NCCL session.
/// Updated by the inbound bridge on each `NewNcclSession` arrival;
/// consumed by the main thread (post-comm-abort) when rebuilding the
/// NCCL comm. Slot semantics: latest write wins; old values are
/// silently overwritten on each new session.
///
/// Fields are populated by the inbound bridge but currently unread;
/// the consumer is the sync_now_nccl retry path in a follow-on slice.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PendingNcclSession {
    pub uid_bytes: Vec<u8>,
    pub new_rank: usize,
    pub new_world_size: usize,
}

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
    /// transition (matches the threaded path's fire-point).
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
    pub fn connect_and_build<F, G, O>(
        coord_addr: SocketAddr,
        cpu_client: Option<crate::distributed::cpu_reduce::CpuReduceClient>,
        rank_id: u32,
        salt: SessionSalt,
        config: WorkerConfig,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        nccl_comm: Option<NcclRankComm>,
        checkpoint_fn: Option<CheckpointFn<M>>,
        epoch_fn: Option<EpochFn<M>>,
        eval_fn: Option<EvalFn<M>>,
        eval_dataset: Option<Arc<dyn BatchDataSet>>,
        outer_optimizer_factory: Option<
            crate::distributed::outer_optimizer::OuterOptimizerFactory,
        >,
    ) -> Result<Self>
    where
        F: FnOnce(Device) -> Result<M>,
        G: FnOnce(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
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
        // triggers a fire-check — mirrors the threaded path's behavior.
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
        // e.g. the coord died with the peer). Mirrors the threaded
        // path's exit sequence; idempotent with the watchdog's abort.
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

// ---------------------------------------------------------------------------
// Bridge thread bodies
// ---------------------------------------------------------------------------

/// TCP → control_tx bridge: read `ControlFrame`s, decode the
/// payload, push into the in-process control channel.
///
/// Elastic-membership frames intercepted here (NOT forwarded to the
/// inner GpuWorker):
/// - `DeclareDead { rank }` → `local_dead_ranks.declare_dead(rank)`,
///   which the NCCL watchdog observes and uses to abort the in-flight
///   collective.
/// - `NewNcclSession { uid_bytes, new_rank, new_world_size }` →
///   `mailbox.replace(Some(PendingNcclSession { … }))`. The main
///   thread reads this slot after its NCCL collective errors out
///   (post-abort) to rebuild the comm.
/// - `RequestNewNcclId` → call `NcclUniqueId::new()` to generate fresh
///   bytes locally and ship them back to the coord via the timing
///   channel as `TimingMsg::NewNcclIdGenerated`. Generation happens
///   here (not on the coord) because the coord process may not link
///   libnccl while workers always do.
///
/// All other frames fall through to `control_wire_to_msg` and the
/// inner control channel as before.
#[allow(clippy::too_many_arguments)]
fn inbound_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    control_tx: &mpsc::Sender<ControlMsg>,
    local_dead_ranks: &Arc<crate::distributed::controller::DeadRanks>,
    nccl_session_mailbox: &Arc<std::sync::Mutex<Option<PendingNcclSession>>>,
    timing_tx: &mpsc::Sender<TimingMsg>,
    coord_liveness_timeout_secs: u64,
) {
    // ESCAPE HATCH: any abnormal exit of this bridge (coordinator or relay
    // death, frame corruption, EOF without a prior Shutdown frame) must wake
    // the inner GpuWorker — WHEREVER it is parked:
    //
    // - Parked in a blocking `control_rx.recv()` (`wait_for_epoch_plan` /
    //   a barrier wait): injecting Shutdown breaks the cycle — the inner
    //   exits its run loop, `run_until_shutdown` drops it, the param
    //   channel disconnects and every bridge unwinds.
    // - Parked INSIDE an NCCL collective / stream synchronize: the control
    //   channel is never read there, so Shutdown alone leaves the rank a
    //   zombie at 100% CPU (its heartbeats only ever told the dead
    //   coordinator). The controller is this rank's only window on the
    //   world, so losing the link means every peer is unreachable:
    //   declare them all dead in the local ledger — the NCCL watchdog
    //   observes the ledger within its poll tick and aborts the comm,
    //   the collective errors out, and the lone-survivor bail exits the
    //   rank with a death record instead of zombifying.
    //
    // On a CLEAN coordinator shutdown the real Shutdown frame arrives
    // BEFORE the link drops; `clean_shutdown_seen` suppresses the peer
    // poison on the subsequent EOF (the main thread may legitimately
    // still be draining the final coherent reduce), and the duplicate
    // Shutdown injection stays a harmless no-op.
    let inject_shutdown = || {
        let _ = control_tx.send(ControlMsg::Shutdown);
    };
    let poison_peers = || {
        for r in 0..local_dead_ranks.world_size() {
            if r != rank {
                local_dead_ranks.declare_dead(r);
            }
        }
    };
    let mut clean_shutdown_seen = false;
    // Coord-liveness deadline: if no inbound frame (CoordHeartbeat OR real
    // traffic) arrives within this window, the coordinator is presumed
    // wedged-open (alive TCP, frozen userspace — the SIGSTOP / deadlock case
    // that never produces EOF or error) and we bail like a hard link drop.
    // Reuses `heartbeat_timeout_secs` so both liveness directions share one
    // wall-clock timescale. Reset below on every successful frame decode; the
    // handshake ACK we just read seeds the clock at loop entry.
    let coord_liveness_deadline = Duration::from_secs(coord_liveness_timeout_secs);
    let mut last_inbound = std::time::Instant::now();
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        // The worker reaches the coord through its host relay, which
        // forwards length-framed opaque ControlFrame blobs on the loopback
        // leg. Read the blob, parse the frame, then dispatch as before.
        match try_read_len_framed(stream) {
            Ok(LenFramedRead::Blob(blob)) => {
                let frame = match ControlFrame::read_from(&mut blob.as_slice(), salt) {
                    Ok(Some(f)) => f,
                    Ok(None) => {
                        eprintln!("cluster_worker: inbound r{rank} truncated ControlFrame");
                        if !clean_shutdown_seen {
                            poison_peers();
                        }
                        inject_shutdown();
                        return;
                    }
                    Err(e) => {
                        eprintln!(
                            "cluster_worker: inbound r{rank} ControlFrame parse: {e}"
                        );
                        if !clean_shutdown_seen {
                            poison_peers();
                        }
                        inject_shutdown();
                        return;
                    }
                };
                // Any coherent frame is proof the coord is alive: reset the
                // liveness deadline (covers CoordHeartbeat AND real traffic).
                last_inbound = std::time::Instant::now();
                match frame.kind {
                MsgKind::Control => match frame.decode::<ControlMsgWire>() {
                    Ok(wire) => match wire {
                        // Pure liveness beacon: absorbed here (the deadline
                        // was just reset above), never forwarded to the inner.
                        ControlMsgWire::CoordHeartbeat => {}
                        // Elastic-membership interception (does NOT
                        // forward to the inner GpuWorker).
                        ControlMsgWire::DeclareDead { rank: dead_r } => {
                            local_dead_ranks.declare_dead(dead_r as usize);
                        }
                        ControlMsgWire::NewNcclSession {
                            uid_bytes,
                            new_rank,
                            new_world_size,
                        } => {
                            let pending = PendingNcclSession {
                                uid_bytes,
                                new_rank: new_rank as usize,
                                new_world_size: new_world_size as usize,
                            };
                            if let Ok(mut slot) = nccl_session_mailbox.lock() {
                                *slot = Some(pending);
                            }
                        }
                        ControlMsgWire::RequestNewNcclId => {
                            match crate::distributed::nccl::NcclUniqueId::new() {
                                Ok(uid) => {
                                    let uid_bytes = uid.as_bytes().to_vec();
                                    let _ = timing_tx.send(
                                        TimingMsg::NewNcclIdGenerated {
                                            rank,
                                            uid_bytes,
                                        },
                                    );
                                }
                                Err(e) => {
                                    eprintln!(
                                        "cluster_worker: inbound r{rank} \
                                         NcclUniqueId::new failed: {e}"
                                    );
                                }
                            }
                        }
                        // atomic-dispatch: the post-reduce Update may
                        // carry the rank's next reduce-window chunk. The
                        // wire-Update itself is informational (the param
                        // bridge synthesises the real
                        // `ControlMsg::Update(AveragedParams)`); when a
                        // `next_plan` rides along, synthesise a
                        // `StartEpoch` so the inner starts the next window
                        // without a separate coord round-trip. Ordering is
                        // safe: the param bridge's `Update(avg)` was sent
                        // (same control channel) before its SyncAck, and
                        // the coord only emits this frame after that ack,
                        // so the inner dequeues `Update(avg)` before this
                        // `StartEpoch` (mpsc FIFO).
                        ControlMsgWire::Update { next_plan, .. } => {
                            if let Some(plan) = next_plan {
                                let msg = ControlMsg::StartEpoch(EpochPlan {
                                    epoch: plan.epoch as usize,
                                    partition_offset: plan.partition_offset as usize,
                                    partition_size: plan.partition_size as usize,
                                });
                                if control_tx.send(msg).is_err() {
                                    // Inner GpuWorker dropped its receiver.
                                    return;
                                }
                            }
                        }
                        // Everything else: existing path through
                        // control_wire_to_msg → inner control_tx.
                        other => match control_wire_to_msg(other) {
                            Ok(Some(msg)) => {
                                if matches!(msg, ControlMsg::Shutdown) {
                                    // Clean teardown announced: the EOF
                                    // that follows is expected — do not
                                    // poison the peer ledger for it.
                                    clean_shutdown_seen = true;
                                }
                                if control_tx.send(msg).is_err() {
                                    // Inner GpuWorker dropped its receiver.
                                    return;
                                }
                            }
                            Ok(None) => {
                                // Wire-side notification with no in-process
                                // dispatch (e.g. Update{version} —
                                // informational; the param bridge handles
                                // the real ControlMsg::Update(AveragedParams).)
                            }
                            Err(e) => {
                                eprintln!(
                                    "cluster_worker: inbound r{rank} control_wire_to_msg: {e}"
                                );
                                if !clean_shutdown_seen {
                                    poison_peers();
                                }
                                inject_shutdown();
                                return;
                            }
                        },
                    },
                    Err(e) => {
                        eprintln!(
                            "cluster_worker: inbound r{rank} decode ControlMsgWire: {e}"
                        );
                        if !clean_shutdown_seen {
                            poison_peers();
                        }
                        inject_shutdown();
                        return;
                    }
                },
                other => {
                    // The control channel only carries Control frames
                    // in the coord→rank direction. Drop everything
                    // else with a diagnostic.
                    eprintln!(
                        "cluster_worker: inbound r{rank} unexpected MsgKind {other:?} \
                         on coord→rank channel; dropping"
                    );
                }
                }
            }
            Ok(LenFramedRead::WouldBlock) => {
                // Wedged-open coordinator: alive socket, no frames. Neither
                // EOF nor error ever fires, so the read-timeout poll alone
                // would loop here forever. If the coord has been silent past
                // the liveness deadline, treat it exactly like a hard link
                // drop — poison peers (NCCL watchdog aborts the collective)
                // and inject Shutdown so the rank exits with a death record.
                if last_inbound.elapsed() >= coord_liveness_deadline {
                    eprintln!(
                        "cluster_worker: inbound r{rank} coordinator silent for \
                         {coord_liveness_timeout_secs}s (presumed wedged); \
                         declaring peers dead and shutting down"
                    );
                    if !clean_shutdown_seen {
                        poison_peers();
                    }
                    inject_shutdown();
                    return;
                }
                continue;
            }
            Ok(LenFramedRead::Eof) => {
                if !clean_shutdown_seen {
                    poison_peers();
                }
                inject_shutdown();
                return;
            }
            Err(e) => {
                // Exit-time broken-pipe / EOF is the common case here:
                // the coord closed its end during shutdown. Downgrade
                // to verbose so steady-state logs stay clean.
                crate::verbose!("cluster_worker: inbound r{rank} wire error: {e}");
                if !clean_shutdown_seen {
                    poison_peers();
                }
                inject_shutdown();
                return;
            }
        }
    }
}

/// timing_rx → TCP bridge: drain in-process timing reports, encode
/// each as a `ControlFrame` and write to the coordinator.
/// Heartbeat cadence (ms). Fast enough that the coord's default 30s
/// staleness threshold catches a wedged rank within ~30 heartbeats,
/// slow enough that the per-cycle frame overhead is negligible.
const HEARTBEAT_CADENCE_MS: u64 = 1_000;

/// Worker-side heartbeat emitter. Fires a `TimingMsg::Heartbeat`
/// every [`HEARTBEAT_CADENCE_MS`] until `shutdown` is signalled or the
/// `timing_tx` channel closes (inner GpuWorker dropped). The heartbeat
/// flows through the outbound bridge alongside Batch / SyncAck / etc.,
/// so the coord receives liveness signal even while the inner is
/// blocked at the AllReduce barrier — distinguishing "alive at
/// barrier" from "dead."
fn heartbeat_loop(
    rank: usize,
    timing_tx: mpsc::Sender<TimingMsg>,
    shutdown: Arc<AtomicBool>,
) {
    let mut step_count: usize = 0;
    while !shutdown.load(Ordering::SeqCst) {
        step_count = step_count.saturating_add(1);
        if timing_tx
            .send(TimingMsg::Heartbeat { rank, step_count })
            .is_err()
        {
            // Inner GpuWorker dropped → channel closed → exit.
            return;
        }
        thread::sleep(Duration::from_millis(HEARTBEAT_CADENCE_MS));
    }
}

/// Poll-cadence for the NCCL watchdog. 100ms keeps detection latency
/// low (a death registered by the inbound bridge is acted on within
/// this window) without burning CPU on the polling loop.
const NCCL_WATCHDOG_POLL_MS: u64 = 100;

/// NCCL watchdog thread body.
///
/// Polls `local_dead_ranks.dead_count()` and calls
/// [`abort`](crate::distributed::nccl::NcclAbortHandle::abort) on the
/// CURRENT handle each time the count
/// increases. The abort unblocks the main thread's in-flight NCCL
/// collective with an Err so the main thread can rebuild the comm on
/// the surviving cohort.
///
/// The handle is re-read from the shared slot on every firing:
/// `GpuWorker::replace_nccl_comm` refreshes the slot at each rebuild,
/// so cascading deaths always abort the live comm (a captured handle
/// would go stale after the first rebuild — its `aborted` flag already
/// tripped — leaving a second death with no abort path: permanent
/// hang). `abort()` stays idempotent per handle via that flag.
fn nccl_watchdog_loop(
    rank: usize,
    abort_slot: crate::distributed::ddp_run::NcclAbortSlot,
    local_dead_ranks: Arc<crate::distributed::controller::DeadRanks>,
    shutdown: Arc<AtomicBool>,
) {
    let mut last_dead_count = 0usize;
    while !shutdown.load(Ordering::SeqCst) {
        let now_dead = local_dead_ranks.dead_count();
        if now_dead > last_dead_count {
            crate::verbose!(
                "  cluster_worker: rank {} NCCL watchdog: dead_count {} -> {}, \
                 aborting NCCL comm",
                rank,
                last_dead_count,
                now_dead,
            );
            let handle = abort_slot
                .lock()
                .expect("nccl abort slot poisoned")
                .clone();
            match handle {
                Some(h) => {
                    if let Err(e) = h.abort() {
                        eprintln!(
                            "cluster_worker: rank {} NCCL watchdog abort error: {}",
                            rank, e,
                        );
                    }
                }
                None => {
                    // Slot emptied (comm torn down for exit); nothing to
                    // abort.
                }
            }
            last_dead_count = now_dead;
        }
        thread::sleep(Duration::from_millis(NCCL_WATCHDOG_POLL_MS));
    }
}

fn outbound_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    timing_rx: mpsc::Receiver<TimingMsg>,
    metrics_rx: mpsc::Receiver<crate::distributed::ddp_run::MetricsMsg>,
) {
    // Drain the rank-side dashboard intent stashed by the user's
    // Monitor calls (`monitor.serve`, `.watch`, `.set_metadata`,
    // captured hardware string). When a dashboard port has been
    // requested, emit the matching `TimingMsgWire::Dashboard*` frames
    // so the launcher's `ClusterDashboardSink` binds the HTTP server
    // and seeds its header / per-rank tabs. Construct a local
    // `ResourceSampler` only when the user opted in — sampling costs
    // a /proc/stat parse + the NVML poller thread, neither worth
    // paying for headless runs.
    let pending = crate::distributed::cluster_dashboard_emit::drain();
    // Per-rank assigned CUDA device. On hosts where
    // `CUDA_VISIBLE_DEVICES` is scoped per rank (`cuda_device_count()
    // == 1`) the sampler returns a single GPU and the filter is a
    // no-op. On hosts where multiple physical GPUs are visible to
    // every rank (Pascal-via-VFIO observed: r1 uses cuda 0, r2 uses
    // cuda 1, both processes see both devices) the sampler returns
    // two snapshots — only ONE belongs to this rank's worker. Without
    // filtering, the dashboard sink would take `.first()` and report
    // the WRONG device's allocator stats (zero, since this process
    // never allocated there). Pull the assigned device index here so
    // `write_metrics` can strip foreign-device entries before shipping,
    // and so `emit_dashboard_setup` can trim the rank's hardware
    // string to its own GPU (the launcher's sink then groups per
    // host and lists per-rank GPU labels without dupes).
    let assigned_device_idx: Option<u8> =
        crate::distributed::LocalCluster::from_env()
            .ok()
            .flatten()
            .and_then(|c| c.my_rank().ok())
            .and_then(|(_, dev)| match dev {
                crate::tensor::Device::CUDA(idx) => Some(idx),
                _ => None,
            });
    let resource_sampler: Option<std::sync::Mutex<crate::monitor::ResourceSampler>> =
        if pending.port.is_some() {
            emit_dashboard_setup(stream, salt, rank, &pending, assigned_device_idx);
            Some(std::sync::Mutex::new(
                crate::monitor::ResourceSampler::new(),
            ))
        } else {
            None
        };
    // recv_timeout so we can periodically check the shutdown flag and
    // service the lower-frequency metrics channel between timing
    // frames. Single thread = serial writes on `stream`; no socket-
    // share race.
    loop {
        if shutdown.load(Ordering::SeqCst) {
            // Drain any final messages so a SyncAck, Exiting, or
            // final-epoch MetricsMsg doesn't get lost on exit.
            while let Ok(msg) = timing_rx.try_recv() {
                let _ = write_timing(stream, salt, msg);
            }
            while let Ok(msg) = metrics_rx.try_recv() {
                let _ = write_metrics(stream, salt, msg, resource_sampler.as_ref(), assigned_device_idx);
            }
            return;
        }
        // Metrics first: lower frequency (per-chunk / per-epoch) and
        // latency-sensitive for dashboard surfacing. Cheap when empty
        // (try_recv returns immediately).
        match metrics_rx.try_recv() {
            Ok(msg) => {
                if let Err(e) = write_metrics(stream, salt, msg, resource_sampler.as_ref(), assigned_device_idx) {
                    crate::verbose!(
                        "cluster_worker: outbound r{rank} metrics write error: {e}"
                    );
                    return;
                }
                continue;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                // Metrics sender dropped; timing channel may still be
                // alive (e.g. heartbeats during teardown). Fall through
                // to timing drain — timing's Disconnected arm exits.
            }
        }
        match timing_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(msg) => {
                if let Err(e) = write_timing(stream, salt, msg) {
                    // Exit-time BrokenPipe is the common case: coord
                    // dropped its end during shutdown. Downgrade so it
                    // doesn't drown steady-state logs.
                    crate::verbose!("cluster_worker: outbound r{rank} write error: {e}");
                    return;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Inner GpuWorker dropped → drain just in case (no-op
                // since Disconnected means buffer empty) and exit.
                return;
            }
        }
    }
}


/// Serialize a `ControlFrame` and write it length-delimited to the
/// worker's host relay, which forwards the opaque blob upstream to the
/// coordinator. Control-channel mirror of the data channel's framing.
fn write_framed_control<W: std::io::Write>(stream: &mut W, frame: &ControlFrame) -> Result<()> {
    let mut buf = Vec::new();
    frame.write_to(&mut buf)?;
    write_len_framed(stream, &buf)
}

fn write_timing(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: TimingMsg,
) -> Result<()> {
    let wire = timing_msg_to_wire(msg);
    let frame = ControlFrame::encode(salt, MsgKind::Timing, &wire)?;
    write_framed_control(stream, &frame)
}

fn write_metrics(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: crate::distributed::ddp_run::MetricsMsg,
    resource_sampler: Option<&std::sync::Mutex<crate::monitor::ResourceSampler>>,
    assigned_device_idx: Option<u8>,
) -> Result<()> {
    let mut wire = metrics_msg_to_wire(msg);
    if let Some(sampler) = resource_sampler {
        // Mutex held briefly — sampler::sample reads /proc/stat +
        // copies the GPU poller's accumulator. No collective; cheap.
        let mut s = sampler.lock().unwrap();
        let mut sample = s.sample();
        // When CUDA_VISIBLE_DEVICES isn't scoped per rank, the sampler
        // returns one snapshot per physical device — but only the
        // rank's assigned device carries this process's allocator
        // stats. Strip foreign-device entries so the dashboard sink's
        // `gpus.first()` lands on the correct GPU.
        if let Some(target) = assigned_device_idx {
            sample.gpus.retain(|g| g.device_index == target);
        }
        wire.resources = Some(sample.into());
    }
    let frame = ControlFrame::encode(salt, MsgKind::Metrics, &wire)?;
    write_framed_control(stream, &frame)
}

/// Emit the rank-side dashboard setup sequence — `DashboardRegister`
/// gated on `port`, plus `DashboardSetSvg` / `DashboardSetMetadata` /
/// `DashboardSetHardware` whenever the stash holds a value. Called
/// once at outbound-loop startup after the user's harness has had a
/// chance to populate the stash through `monitor.serve` /
/// `monitor.watch` / `monitor.set_metadata` and `Monitor::new`'s
/// hardware capture. Errors are logged verbosely but never abort the
/// rank — the dashboard is optional UX, not a training invariant.
fn emit_dashboard_setup(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    rank: usize,
    pending: &crate::distributed::cluster_dashboard_emit::PendingDashboardConfig,
    assigned_device_idx: Option<u8>,
) {
    use crate::distributed::wire::TimingMsgWire;
    let rank_u64 = rank as u64;
    let mut emit = |msg: TimingMsgWire| {
        if let Err(e) = write_timing_wire(stream, salt, &msg) {
            crate::verbose!(
                "cluster_worker: outbound r{rank} dashboard emit failed: {e}",
            );
        }
    };
    if let Some(port) = pending.port {
        emit(TimingMsgWire::DashboardRegister { rank: rank_u64, port });
    }
    if let Some(ref svg) = pending.svg {
        emit(TimingMsgWire::DashboardSetSvg {
            rank: rank_u64,
            svg: svg.clone(),
            label: pending.label.clone(),
            hash: pending.hash.clone(),
        });
    }
    if let Some(ref json) = pending.metadata_json {
        emit(TimingMsgWire::DashboardSetMetadata {
            rank: rank_u64,
            json: json.clone(),
        });
    }
    if let Some(ref hw) = pending.hardware {
        // `tensor::hardware_summary` returns `CPU | gpu0 | gpu1 | …`
        // — every visible GPU. In cluster mode each rank only USES one
        // GPU (its assigned device); listing the others puffs the
        // launcher's header and visually repeats hardware across ranks
        // on the same host. Trim to `CPU | <my_gpu>` so the sink's
        // per-host grouping can render: `host: cpu | gr=N lr=M: gpu |
        // gr=K lr=L: gpu | other_host: …`.
        let trimmed = trim_hardware_to_assigned(hw, assigned_device_idx);
        emit(TimingMsgWire::DashboardSetHardware {
            rank: rank_u64,
            summary: trimmed,
        });
    }
}

/// Split `full` on `" | "` and keep `[0]` (CPU) + the GPU at
/// `assigned_device_idx` if present. Returns the original string
/// untouched when no assigned device is known (single-process / CPU
/// builds) or when the segment count doesn't match the expected
/// `cpu | gpu0 | gpu1 | …` shape (e.g. NVML returned no GPU names).
fn trim_hardware_to_assigned(
    full: &str,
    assigned_device_idx: Option<u8>,
) -> String {
    let Some(target) = assigned_device_idx else {
        return full.to_string();
    };
    let parts: Vec<&str> = full.split(" | ").collect();
    if parts.len() < 2 {
        return full.to_string();
    }
    let cpu = parts[0];
    // GPUs are positionally indexed: parts[1] = device 0, parts[2] =
    // device 1, etc. Use `target + 1` to index into the GPU portion.
    let gpu_idx = target as usize + 1;
    match parts.get(gpu_idx) {
        Some(gpu) => format!("{cpu} | {gpu}"),
        None => cpu.to_string(),
    }
}

/// Write an already-built [`TimingMsgWire`] directly. The non-wire
/// `write_timing` takes an in-process `TimingMsg`; the dashboard emit
/// path skips that intermediate and serializes the wire form directly
/// (no in-process `TimingMsg` variant exists for these — they're a
/// pure wire concern).
fn write_timing_wire(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: &crate::distributed::wire::TimingMsgWire,
) -> Result<()> {
    let frame = ControlFrame::encode(salt, MsgKind::Timing, msg)?;
    write_framed_control(stream, &frame)
}

/// Convert in-process [`crate::distributed::ddp_run::MetricsMsg`]
/// into wire-compatible [`crate::distributed::wire::MetricsMsgWire`]
/// for transit over the metrics-channel TCP frame.
fn metrics_msg_to_wire(msg: crate::distributed::ddp_run::MetricsMsg)
    -> crate::distributed::wire::MetricsMsgWire
{
    crate::distributed::wire::MetricsMsgWire {
        rank: msg.rank as u64,
        epoch: msg.epoch as u64,
        avg_loss: msg.avg_loss,
        batches_processed: msg.batches_processed as u64,
        epoch_ms: msg.epoch_ms,
        samples_processed: msg.samples_processed as u64,
        share_complete_ms: msg.share_complete_ms,
        compute_only_ms: msg.compute_only_ms,
        data_starve_ms: msg.data_starve_ms,
        scalars: msg.scalars
            .into_iter()
            .map(|(k, (sum, count))| (k, (sum, count as u64)))
            .collect(),
        // Populated by the dashboard-aware emit path when the launcher
        // hosts a dashboard; the plain wire conversion leaves it None
        // and lets the worker layer (which holds the ResourceSampler)
        // attach a sample before writing.
        resources: None,
    }
}

/// CPU-averaging param bridge: receives
/// [`ParamSnapshot`](crate::distributed::ddp_run::ParamSnapshot)s from the
/// inner [`GpuWorker`] (triggered by `RequestParams`), runs an
/// all-reduce round-trip through the data channel via
/// [`crate::distributed::cpu_reduce::CpuReduceClient`], and feeds the
/// averaged tensors back to the inner as `ControlMsg::Update`. Also
/// emits `TimingMsg::SyncAck` on the timing channel so the
/// coordinator's `nccl_ack` gate releases. The SyncAck carries the
/// weight-space divergence triple (`||pre - post|| / ||post||`,
/// `pre_norm`, `post_norm`) so the coord's
/// [`ConvergenceGuard`](crate::distributed::ddp_run::convergence::ConvergenceGuard)
/// sees real signal on the CPU averaging path.
///
/// When `cpu_client` is `None`, the bridge degrades to a discard
/// drainer (NCCL-only worker layout — the inner never emits
/// ParamSnapshot in that mode either, so the channel idles).
fn param_bridge_loop(
    rank: u64,
    param_rx: mpsc::Receiver<crate::distributed::ddp_run::ParamSnapshot>,
    cpu_client: Option<crate::distributed::cpu_reduce::CpuReduceClient>,
    control_tx: mpsc::Sender<ControlMsg>,
    timing_tx: mpsc::Sender<TimingMsg>,
    gamma: f64,
) {
    use crate::distributed::ddp_run::{AveragedParams, ParamSnapshot};
    let Some(mut client) = cpu_client else {
        // Discard mode (NCCL-only worker).
        while param_rx.recv().is_ok() {}
        return;
    };
    // ESCAPE HATCH (same contract as the inbound bridge): a mid-round error
    // here means this rank can no longer participate in CPU averaging — the
    // inner GpuWorker would otherwise wait at the reduce barrier for an
    // `Update` that will never come. Wake it with Shutdown so the rank exits
    // (checkpoint drain + final snapshot + Exiting) instead of parking
    // forever; the coordinator's heartbeat staleness then reaps it.
    let inject_shutdown = || {
        let _ = control_tx.send(ControlMsg::Shutdown);
    };
    // Monotonic local version counter; bumped per round so the
    // synthesized AveragedParams.version increases consistently.
    let mut version: u64 = 0;
    // Pre-sync scratch for weight-space divergence math. Allocated
    // lazily on the first ParamSnapshot (shapes match the inner
    // GpuWorker's param tensors; reused unchanged across rounds).
    let mut pre_scratch: Option<Vec<Tensor>> = None;

    while let Ok(snapshot) = param_rx.recv() {
        let ParamSnapshot {
            rank: snap_rank,
            params,
            buffers,
            batch_count: n_i,
        } = snapshot;
        debug_assert_eq!(
            snap_rank as u64, rank,
            "param bridge: snapshot.rank mismatch with bridge rank"
        );

        // One-time scratch allocation matched to the snapshot shapes.
        if pre_scratch.is_none() {
            let allocated: Result<Vec<Tensor>> =
                params.iter().map(Tensor::zeros_like).collect();
            match allocated {
                Ok(s) => pre_scratch = Some(s),
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} scratch alloc: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            }
        }
        let scratch = pre_scratch.as_ref().expect("scratch just allocated");

        // Capture pre-sync params into scratch (deep copy; scratch
        // never shares storage with snapshot.params, so the math
        // stays correct regardless of device or ApplyPolicy).
        let mut copy_failed = false;
        for (dst, src) in scratch.iter().zip(params.iter()) {
            if let Err(e) = dst.copy_(src, false) {
                eprintln!(
                    "cluster_worker: param bridge r{rank} pre_scratch copy_: {e}"
                );
                copy_failed = true;
                break;
            }
        }
        if copy_failed {
            inject_shutdown();
            return;
        }

        // Emit SnapshotReady BEFORE entering the AllReduce barrier so
        // the coord's per-rank capacity signal (T_ready - T_request)
        // measures snapshot + upload only, NOT polluted by slowest-
        // rank barrier wait. Failure to send is non-fatal — channel
        // closed means the coord-side bridge is gone, and the next
        // op will surface the real error.
        let _ = timing_tx.send(TimingMsg::SnapshotReady {
            rank: rank as usize,
        });

        // Realized-work weighted AllReduce. Each rank scales its
        // contribution by the work it did since the last sync (params:
        // batch_count `n_i` raised to gamma; buffers: a 0/1 mover
        // indicator) and ships the mass IN THE SAME FRAME — the
        // controller divides the summed contributions ONCE by the mass
        // of exactly the frames it accepted, so the divisor can never
        // disagree with the sum and the scattered frame IS the
        // consensus (also what the forge writes and the outer optimizer
        // steps). Sum stays associative, so it composes with a future
        // per-host partial sum (the relay sum-and-count) without
        // averaging-of-averages. A rank that did 0 steps still holds
        // the previous consensus, so it contributes zero mass — but it
        // STILL joins every collective, so no rank stalls. f32 only in
        // v1; CpuReduceClient surfaces a loud error otherwise.
        //
        // The count gather is the all-idle skip signal (and schedule
        // telemetry): every rank learns Σ n_i and makes the identical
        // enter-or-skip decision. The authoritative divisor is NOT
        // taken from here — it rides each reduce frame.
        let world = client.world_size() as usize;
        let mut counts = vec![0.0f64; world];
        if (rank as usize) < world {
            counts[rank as usize] = n_i as f64;
        }
        if let Err(e) = client.all_reduce_per_rank_f64(&mut counts) {
            eprintln!("cluster_worker: param bridge r{rank} count gather: {e}");
            inject_shutdown();
            return;
        }
        let total_n: f64 = counts.iter().sum();

        // All-idle (no rank moved since the last sync — every snapshot
        // already equals the consensus): skip the reduce, leave params /
        // buffers unchanged. Every rank sees the same gathered `total_n ==
        // 0`, so the skip is collective-consistent (no rank left waiting in
        // an all_reduce its peers did not call).
        // Gamma allocation-weighting: rank k is weighted nₖ^γ (γ=1.0 = plain
        // work-weighting, byte-identical to pre-gamma; γ<1 compresses the
        // fast/over-allocated rank's dominance, γ=0 = unweighted average,
        // γ<0 = per-step-equal). Idle ranks (nₖ=0) contribute zero mass for
        // any γ (the idle guard in `realized_work`). No local normalizer:
        // the divisor rides the frames — the controller divides once by
        // the mass it accepted. Only the params consensus is
        // gamma-weighted; buffers stay equal-weighted among movers.
        let my_w = crate::distributed::realized_work::gamma_mass(n_i as f64, gamma);
        let avg_params = if total_n == 0.0 {
            params.clone()
        } else {
            match sumcount_reduce(&mut client, &params, my_w) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} all_reduce params: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            }
        };

        // Weight-space divergence (||pre - post|| / ||post||, plus
        // pre_norm / post_norm) computed before the buffer reduce so
        // a later buffer error path can't mask the params triple.
        let (divergence, post_norm, pre_norm) =
            match compute_divergence(scratch, &avg_params) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} divergence: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            };

        // Buffers (BatchNorm running stats etc.): equal weight among the
        // ranks that moved (idle excluded via the 0/1 indicator); the
        // controller divides by the accepted mover count.
        //
        // Only the f32 subset rides the reduce — the CPU transport is
        // f32-only, and non-f32 buffers (integer counters and the like)
        // are deterministic values updated identically on every rank,
        // so passing the local value through unchanged is correct, not
        // a dropped sync (mirrors the bootstrap broadcast's filter in
        // `rank_entry`). The merge is positional, so the consumer's
        // zip against the live buffer list stays aligned; every rank
        // builds the same model, so the subset matches in count/order
        // across ranks and the collective stays balanced.
        let f32_buffer_idx: Vec<usize> = buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.dtype() == crate::tensor::DType::Float32)
            .map(|(i, _)| i)
            .collect();
        let avg_buffers = if f32_buffer_idx.is_empty() || total_n == 0.0 {
            buffers.clone()
        } else {
            let subset: Vec<Tensor> =
                f32_buffer_idx.iter().map(|&i| buffers[i].clone()).collect();
            let my_indicator = crate::distributed::realized_work::mover_mass(n_i as f64);
            match sumcount_reduce(&mut client, &subset, my_indicator) {
                Ok(reduced) => {
                    let mut merged = buffers.clone();
                    for (k, &i) in f32_buffer_idx.iter().enumerate() {
                        merged[i] = reduced[k].clone();
                    }
                    merged
                }
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} all_reduce buffers: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            }
        };
        version += 1;
        let avg = AveragedParams {
            params: avg_params,
            buffers: avg_buffers,
            version,
        };
        if control_tx.send(ControlMsg::Update(avg)).is_err() {
            // Inner GpuWorker dropped its receiver; tear down.
            return;
        }
        // Ack the coordinator. CPU re-arm runs off the coord's
        // `cpu_avg_state` machine (finalized by `poll_cpu_averaging` once
        // every rank's divergence has landed), NOT off `step_count`, so
        // this ack carries no synthetic step. A real step_count isn't
        // available here anyway — the inner GpuWorker doesn't bump
        // `local_step` on `RequestParams`. Sending 0 keeps the coord's
        // `last_step_count` clean (it ignores CPU-path step_counts). The
        // previous `usize::MAX / 2` sentinel poisoned `last_step_count`,
        // wedging the NCCL-style re-arm gate after a few cycles.
        let _ = timing_tx.send(TimingMsg::SyncAck {
            rank: rank as usize,
            step_count: 0,
            divergence: Some(divergence),
            post_norm,
            pre_norm,
        });
    }
    // Channel closed (clean training end): emit the accumulated reduce
    // profile so the cpu-cadence reduce floor can be attributed to
    // serialize / wire / deserialize. One line per rank, on stderr.
    client.log_profile_summary();
}

/// Sum-and-count weighted AllReduce over the CPU data channel: returns
/// `Σ_r (w_r · T_r) / total_weight` per tensor, computed as a plain Sum
/// followed by a SINGLE divide.
///
/// `CpuReduceClient::all_reduce_tensors` implements the avg-trick (the
/// controller sums every rank's contribution then divides by
/// `world_size`), so this pre-multiplies each rank's scaled contribution
/// by `world_size` to recover a plain Sum, reduces, then divides once by
/// `total_weight`. Deferring the divide to a single final step (rather
/// than forming `w_r/total` before the reduce) keeps the operation
/// associative: a future per-host partial sum can fold local ranks into
/// one `(Σ w·T, Σ w)` pair and the root still divides exactly once — no
/// averaging-of-averages. A zero-weight rank contributes a zeroed tensor
/// but still joins the collective, so the cohort never stalls.
/// Realized-work weighted reduce: ship `my_weight`-scaled tensors with
/// the mass riding the SAME frame, so the controller's divisor is the
/// summed mass of exactly the contributions it accepted into the round
/// — the sum and its divisor can never disagree, whatever the cohort
/// did between rounds. The controller divides ONCE and scatters the
/// consensus, which is therefore also what the checkpoint forge writes
/// and what the outer optimizer steps.
///
/// Returns the adopted tensors: the scattered consensus, or shallow
/// clones of `tensors` when the round realized no work (returned mass
/// `0.0`, e.g. every accepted contributor was idle) — the caller keeps
/// its local state, mirroring the all-idle collective skip.
pub(crate) fn sumcount_reduce(
    client: &mut crate::distributed::cpu_reduce::CpuReduceClient,
    tensors: &[Tensor],
    my_weight: f64,
) -> Result<Vec<Tensor>> {
    let scaled: Vec<Tensor> = tensors
        .iter()
        .map(|t| t.mul_scalar(my_weight))
        .collect::<Result<_>>()?;
    let refs: Vec<&Tensor> = scaled.iter().collect();
    let (consensus, realized) = client.all_reduce_weighted(
        &refs,
        crate::distributed::controller::RoundKind::Model,
        my_weight,
    )?;
    if !crate::distributed::realized_work::is_realized(realized) {
        crate::debug!(
            "cluster_worker: reduce round realized no work (mass 0); keeping local state"
        );
        return Ok(tensors.to_vec());
    }
    Ok(consensus)
}

/// Compute the weight-space divergence triple
/// `(||pre - post|| / ||post||, post_norm, pre_norm)`.
///
/// Mirrors
/// [`CpuReduceClient::average_params_with_divergence`](
/// crate::distributed::cpu_reduce::CpuReduceClient::average_params_with_divergence
/// ) but accepts pre and post as separate slices — the param bridge
/// keeps `post` (averaged) in a freshly returned vector rather than
/// mutating snapshot tensors in place, so the math stays correct
/// regardless of whether snapshot tensors share storage with the
/// inner GpuWorker's live params (true on CPU device, false on
/// CUDA + to_device(CPU) hop).
///
/// **Mutates `pre` in place** (subtracts `post`); the caller treats
/// `pre` as scratch that is overwritten by each round.
fn compute_divergence(
    pre: &[Tensor],
    post: &[Tensor],
) -> Result<(f64, Option<f64>, Option<f64>)> {
    if pre.is_empty() {
        return Ok((0.0, None, None));
    }
    if pre.len() != post.len() {
        return Err(TensorError::new(&format!(
            "compute_divergence: pre.len() ({}) must equal post.len() ({})",
            pre.len(),
            post.len(),
        )));
    }

    // pre_norm BEFORE the foreach_add_list_ subtracts post from scratch.
    let pre_norm_tensors = Tensor::foreach_norm(pre, 2.0)?;
    let mut pre_sq = 0.0f64;
    for n in &pre_norm_tensors {
        let v: f64 = n.item()?;
        pre_sq += v * v;
    }
    let pre_norm = pre_sq.sqrt();

    // scratch[i] += -1 * post[i]  →  scratch[i] = pre - post.
    Tensor::foreach_add_list_(pre, post, -1.0)?;
    let diff_norms = Tensor::foreach_norm(pre, 2.0)?;
    let post_norms = Tensor::foreach_norm(post, 2.0)?;

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
    let divergence = if post_norm > 1e-10 {
        diff_sq.sqrt() / post_norm
    } else {
        0.0
    };

    Ok((divergence, Some(post_norm), Some(pre_norm)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "cluster_worker_tests.rs"]
mod tests;
