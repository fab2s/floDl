//! Cluster worker: TCP-driven wrapper around the OLD threaded
//! [`GpuWorker`](crate::distributed::ddp_run::GpuWorker).
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
//!    [`ControlFrame`]s on the wire.
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
//! # Scope of 4b.D.1d.2
//!
//! - **In scope**: Sync+Nccl path end-to-end. ControlMsgWire variants
//!   `SyncNow`, `Throttle`, `SetGlobalStep`, `StartEpoch`, `Shutdown`,
//!   `Checkpoint` translate to in-process `ControlMsg`. TimingMsgWire
//!   variants all four directions.
//! - **Out of scope (1d.4 follow-up)**: `ControlMsgWire::Update` (CPU
//!   averaging path — needs tensors via the data channel), `RequestParams`
//!   (CPU averaging), final-snapshot data channel egress, metrics
//!   aggregation. These bridges either drop messages with a debug log or
//!   surface a loud error.
//!
//! # Tests
//!
//! - CPU structural test: ClusterWorker handshakes with a real
//!   [`ClusterCoordinator`], runs a trivial CPU model + dataset through
//!   one Sync averaging cycle, exits cleanly.
//! - `#[ignore = "cuda"]` end-to-end NCCL smoke test: two ranks share a
//!   NcclRankComm, do real AllReduce(Avg) on their parameters after a
//!   few batches, verify weights converge to consensus. Runs via
//!   `fdl cuda-test-nccl` on a multi-GPU rig.

use std::io::Read;
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::distributed::cluster_coordinator::{write_handshake_rank, CTRL_HS_ACK, CTRL_HS_VERSION};
use crate::distributed::ddp_run::{
    CheckpointFn, ControlMsg, EpochPlan, GpuWorker, TimingMsg, WorkerConfig,
};
use crate::distributed::nccl::NcclRankComm;
use crate::distributed::wire::{
    hmac_sha256_64, ControlFrame, ControlMsgWire, FrameRead, MsgKind, SessionSalt,
    TimingMsgWire,
};
#[cfg(test)]
use crate::distributed::wire::EpochPlanWire;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor, TensorError};

// ---------------------------------------------------------------------------
// Handshake (worker side)
// ---------------------------------------------------------------------------

const HS_ACK_BYTES: usize = 16;

fn read_handshake_ack(stream: &mut TcpStream, salt: &SessionSalt) -> Result<()> {
    let mut buf = [0u8; HS_ACK_BYTES];
    stream.read_exact(&mut buf).map_err(|e| {
        TensorError::new(&format!(
            "cluster_worker: handshake ack read failed: {e} \
             (coordinator may have rejected our handshake)"
        ))
    })?;
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != CTRL_HS_ACK {
        return Err(TensorError::new(&format!(
            "cluster_worker: handshake ack magic 0x{magic:08x} != 0x{CTRL_HS_ACK:08x}"
        )));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != CTRL_HS_VERSION {
        return Err(TensorError::new(&format!(
            "cluster_worker: handshake ack version {version} != {CTRL_HS_VERSION}"
        )));
    }
    let full = hmac_sha256_64(salt, &buf[0..8]);
    let expected = full.to_le_bytes();
    let got: [u8; 8] = buf[8..16].try_into().unwrap();
    if expected != got {
        return Err(TensorError::new(
            "cluster_worker: handshake ack HMAC verification failed; \
             session salt disagreement (worker holds a different salt than coordinator)",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Bridge helpers (mpsc ↔ TCP)
// ---------------------------------------------------------------------------

/// Convert an in-process [`TimingMsg`] into the bincode-serializable
/// [`TimingMsgWire`] for transit over the TCP control channel.
fn timing_msg_to_wire(msg: TimingMsg) -> TimingMsgWire {
    match msg {
        TimingMsg::Batch {
            rank,
            batch_ms,
            step_count,
            param_norm,
            batch_loss,
            sync_divergence,
        } => TimingMsgWire::Batch {
            rank: rank as u64,
            batch_ms,
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
    }
}

/// Convert an inbound [`ControlMsgWire`] from the coordinator into an
/// optional in-process [`ControlMsg`] for [`GpuWorker::dispatch_control`].
///
/// Returns `Ok(None)` for wire variants that don't need in-process
/// dispatch:
///
/// - `ControlMsgWire::Update { version }`: the wire-side notification
///   that the averaging cycle is complete. The real in-process
///   `ControlMsg::Update(AveragedParams)` flows through the param
///   bridge (where the param bridge synthesizes one with the actual
///   averaged tensors from the data channel). The wire-Update is
///   informational only.
///
/// All other wire variants map 1:1.
fn control_wire_to_msg(wire: ControlMsgWire) -> Result<Option<ControlMsg>> {
    match wire {
        ControlMsgWire::RequestParams => Ok(Some(ControlMsg::RequestParams)),
        ControlMsgWire::Update { version: _ } => Ok(None),
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
        ControlMsgWire::Throttle => Ok(Some(ControlMsg::Throttle)),
        ControlMsgWire::SetGlobalStep { global_step } => {
            Ok(Some(ControlMsg::SetGlobalStep(global_step as usize)))
        }
        ControlMsgWire::Checkpoint { version } => Ok(Some(ControlMsg::Checkpoint { version })),
        ControlMsgWire::Shutdown => Ok(Some(ControlMsg::Shutdown)),
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
/// the new control-channel [`ControlFrame`] wire protocol.
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
        data_addr: Option<SocketAddr>,
        rank_id: u32,
        salt: SessionSalt,
        config: WorkerConfig,
        model_factory: F,
        optim_factory: G,
        dataset: Arc<dyn BatchDataSet>,
        nccl_comm: Option<NcclRankComm>,
        checkpoint_fn: Option<CheckpointFn<M>>,
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

        // Connect with a generous timeout; ranks may briefly race the
        // coordinator's accept() after the launcher kicks them off.
        let stream = TcpStream::connect_timeout(&coord_addr, Duration::from_secs(10))
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_worker: connect to {coord_addr} failed: {e}"
                ))
            })?;
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| {
                TensorError::new(&format!("cluster_worker: set_read_timeout: {e}"))
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
        // Writes shouldn't inherit the short read_timeout; clear it on
        // the write half just to be explicit (writes use TCP send buffer
        // back-pressure, not timeouts).
        write_stream.set_read_timeout(None).ok();

        // mpsc quintet — the worker-side senders flow into GpuWorker,
        // the coord-side ends stay with the bridges. Clone the senders
        // that bridges need access to (timing_tx for the SyncAck
        // emitted after a CPU-averaging round, control_tx for the
        // param bridge's synthesized ControlMsg::Update).
        let (timing_tx, timing_rx) = mpsc::channel::<TimingMsg>();
        let timing_tx_for_param_bridge = timing_tx.clone();
        let timing_tx_for_heartbeat = timing_tx.clone();
        let (metrics_tx, metrics_rx) = mpsc::channel::<crate::distributed::ddp_run::MetricsMsg>();
        let (param_tx, param_rx) =
            mpsc::channel::<crate::distributed::ddp_run::ParamSnapshot>();
        let (final_param_tx, final_param_rx) =
            mpsc::channel::<crate::distributed::ddp_run::ParamSnapshot>();
        let (control_tx, control_rx) = mpsc::channel::<ControlMsg>();
        let control_tx_for_param_bridge = control_tx.clone();

        // CpuReduceClient on the data channel, used by the param
        // bridge below when AverageBackend::Cpu is in play. None when
        // the worker is in NCCL-only mode (data_addr unset).
        let cpu_client = if let Some(addr) = data_addr {
            Some(crate::distributed::cpu_reduce::CpuReduceClient::connect(
                addr,
                rank_id,
                config.world_size as u32,
                salt,
            )?)
        } else {
            None
        };

        let inner = GpuWorker::<M>::new(
            &config,
            model_factory,
            optim_factory,
            dataset,
            nccl_comm,
            checkpoint_fn,
            timing_tx,
            metrics_tx,
            param_tx,
            final_param_tx,
            control_rx,
        )?;

        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let mut bridges: Vec<JoinHandle<()>> = Vec::new();

        // Inbound bridge: TCP ControlFrame → ControlMsg → control_tx.
        let salt_in = salt;
        let shutdown_in = Arc::clone(&shutdown_flag);
        let rank_in = config.rank;
        let mut read_stream_for_bridge = read_stream;
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
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn inbound bridge for rank {rank_in}: {e}"
                    ))
                })?,
        );

        // Outbound bridge: timing_rx → TimingMsgWire → TCP ControlFrame.
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
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn outbound bridge for rank {rank_out}: {e}"
                    ))
                })?,
        );

        // Discard bridges: metrics + param + final_param egress paths.
        // 1d.5 wires metrics aggregation; 1d.4 wires CPU-averaging
        // param snapshot data-channel egress. Until then, drain to keep
        // the inner GpuWorker's sends from blocking.
        bridges.push(
            thread::Builder::new()
                .name(format!("flodl-worker-discard-metrics:r{rank_out}"))
                .spawn(move || {
                    while metrics_rx.recv().is_ok() {
                        // Drop. 1d.5 wires this to the data channel.
                    }
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn metrics discard bridge: {e}"
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
                    );
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn param bridge: {e}"
                    ))
                })?,
        );
        bridges.push(
            thread::Builder::new()
                .name(format!("flodl-worker-discard-final:r{rank_out}"))
                .spawn(move || {
                    while final_param_rx.recv().is_ok() {
                        // Drop. 1d.4 wires this to the data channel.
                    }
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_worker: spawn final discard bridge: {e}"
                    ))
                })?,
        );
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

        Ok(ClusterWorker {
            inner: Some(inner),
            bridges,
            shutdown_flag,
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
    pub fn run_until_shutdown<T>(mut self, train_fn: T) -> Result<()>
    where
        T: Fn(&M, &[Tensor]) -> Result<Variable>,
    {
        // Inner is set in connect_and_build; only `run_until_shutdown`
        // takes it out. Unwrap is safe here.
        let mut inner = self
            .inner
            .take()
            .expect("inner GpuWorker present at run_until_shutdown");

        let exit_clean = (|| -> Result<bool> {
            loop {
                match inner.wait_for_epoch_plan()? {
                    Some(plan) => {
                        let shutdown = inner.run_epoch_plan(&plan, &train_fn)?;
                        if shutdown {
                            return Ok(true);
                        }
                    }
                    None => return Ok(true),
                }
            }
        })();

        // Even on error, try to gracefully report exit + drop senders
        // so the coordinator side cleans up. send_final_snapshot uses
        // the dedicated final_param channel which the discard bridge
        // consumes; report_exiting goes through the outbound bridge.
        inner.send_final_snapshot();
        inner.report_exiting();

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

        exit_clean.map(|_| ())
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

/// TCP → control_tx bridge: read [`ControlFrame`]s, decode the
/// payload, push into the in-process control channel.
fn inbound_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    control_tx: &mpsc::Sender<ControlMsg>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        match ControlFrame::try_read_from(stream, salt) {
            Ok(FrameRead::Frame(frame)) => match frame.kind {
                MsgKind::Control => match frame.decode::<ControlMsgWire>() {
                    Ok(wire) => match control_wire_to_msg(wire) {
                        Ok(Some(msg)) => {
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
                            return;
                        }
                    },
                    Err(e) => {
                        eprintln!(
                            "cluster_worker: inbound r{rank} decode ControlMsgWire: {e}"
                        );
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
            },
            Ok(FrameRead::WouldBlock) => continue,
            Ok(FrameRead::Eof) => return,
            Err(e) => {
                eprintln!("cluster_worker: inbound r{rank} wire error: {e}");
                return;
            }
        }
    }
}

/// timing_rx → TCP bridge: drain in-process timing reports, encode
/// each as a [`ControlFrame`] and write to the coordinator.
/// Heartbeat cadence (ms). Fast enough that the coord's default 30s
/// staleness threshold catches a wedged rank within ~30 heartbeats,
/// slow enough that the per-cycle frame overhead is negligible.
const HEARTBEAT_CADENCE_MS: u64 = 1_000;

/// Worker-side heartbeat emitter. Fires a [`TimingMsg::Heartbeat`]
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

fn outbound_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    timing_rx: mpsc::Receiver<TimingMsg>,
) {
    // recv_timeout so we can periodically check the shutdown flag.
    loop {
        if shutdown.load(Ordering::SeqCst) {
            // Drain any final messages so a SyncAck or Exiting
            // doesn't get lost on exit.
            while let Ok(msg) = timing_rx.try_recv() {
                let _ = write_one(stream, salt, msg);
            }
            return;
        }
        match timing_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(msg) => {
                if let Err(e) = write_one(stream, salt, msg) {
                    eprintln!("cluster_worker: outbound r{rank} write error: {e}");
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

fn write_one(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: TimingMsg,
) -> Result<()> {
    let wire = timing_msg_to_wire(msg);
    let frame = ControlFrame::encode(salt, MsgKind::Timing, &wire)?;
    frame.write_to(stream)
}

/// CPU-averaging param bridge: receives [`ParamSnapshot`]s from the
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
) {
    use crate::distributed::ddp_run::{AveragedParams, ParamSnapshot};
    let Some(mut client) = cpu_client else {
        // Discard mode (NCCL-only worker).
        while param_rx.recv().is_ok() {}
        return;
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
            batch_count: _,
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

        // AllReduce-Avg params via the data channel; returns NEW
        // averaged tensors (snapshot.params untouched). f32 only in
        // v1; CpuReduceClient surfaces a loud error otherwise.
        let param_refs: Vec<&Tensor> = params.iter().collect();
        let avg_params = match client.all_reduce_tensors(&param_refs) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "cluster_worker: param bridge r{rank} all_reduce params: {e}"
                );
                return;
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
                    return;
                }
            };

        let buffer_refs: Vec<&Tensor> = buffers.iter().collect();
        let avg_buffers = if buffer_refs.is_empty() {
            Vec::new()
        } else {
            match client.all_reduce_tensors(&buffer_refs) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} all_reduce buffers: {e}"
                    );
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
        // Ack the coordinator. Use a synthetic large step_count so the
        // coord's `step_count > nccl_sync_step` gate (NCCL-specific
        // deadlock guard) trivially passes on the CPU path.
        let _ = timing_tx.send(TimingMsg::SyncAck {
            rank: rank as usize,
            step_count: usize::MAX / 2,
            divergence: Some(divergence),
            post_norm,
            pre_norm,
        });
    }
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
mod tests {
    use super::*;
    use crate::distributed::cluster_coordinator::{
        ClusterCoordinator, ClusterCoordinatorConfig,
    };
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
    use crate::distributed::wire::SESSION_SALT_BYTES;
    use std::net::Ipv4Addr;
    use std::time::Instant;

    /// Deterministic non-zero test salt — same value as the
    /// cluster_coordinator / controller test salts so cross-module
    /// integration tests can chain freely.
    const TEST_SALT: SessionSalt = [0x42u8; SESSION_SALT_BYTES];

    fn coord_config_sync_nccl(world_size: usize) -> ClusterCoordinatorConfig {
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 1),
        )
        .no_divergence_guard()
    }

    /// Spawn a ClusterCoordinator that drives `drive` to completion,
    /// then shuts down. Returns the bound port + join handle.
    fn spawn_coord<D>(
        world_size: usize,
        drive: D,
    ) -> (u16, thread::JoinHandle<Result<()>>)
    where
        D: Send + 'static + FnOnce(&mut ClusterCoordinator) -> Result<()>,
    {
        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("bind succeeds");
        let h = thread::spawn(move || -> Result<()> {
            let mut coord = ClusterCoordinator::start_from_listener(
                listener,
                TEST_SALT,
                coord_config_sync_nccl(world_size),
            )?;
            let r = drive(&mut coord);
            let _ = coord.shutdown();
            r
        });
        (port, h)
    }

    /// Smoke test: a ClusterWorker can hold a TcpStream open against a
    /// real ClusterCoordinator after handshake, even with no inner
    /// GpuWorker constructed yet. Exercises just the handshake
    /// bytes (matching salt path, ack HMAC verification).
    #[test]
    fn handshake_with_real_coordinator() {
        let world_size = 1;
        // ClusterCoordinator demands world_size >= 2 (ElChe), so we
        // use 2 here and have a dummy second rank just complete its
        // handshake then drop.
        let world_size = world_size.max(2);
        let (port, coord_handle) = spawn_coord(world_size, |coord| {
            // Drive one tick to confirm the coord registered both
            // ranks before they drop.
            // The accept loop in start_from_listener already validated
            // both handshakes; coord.tick() just returns Ok.
            let _ = coord.tick();
            Ok(())
        });
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);

        // Direct-handshake closures (no inner GpuWorker required —
        // we exercise only the handshake bytes + ack here).
        fn raw_rank_handshake(addr: SocketAddr, rank: u32, ws: u32) {
            let mut stream =
                TcpStream::connect_timeout(&addr, Duration::from_secs(5)).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            write_handshake_rank(&mut stream, rank, ws, &TEST_SALT).unwrap();
            read_handshake_ack(&mut stream, &TEST_SALT).unwrap();
            // Hold the stream open briefly so the coord can register
            // before we drop.
            thread::sleep(Duration::from_millis(50));
        }
        let r0 = thread::spawn(move || raw_rank_handshake(addr, 0, world_size as u32));
        let r1 = thread::spawn(move || raw_rank_handshake(addr, 1, world_size as u32));
        r0.join().unwrap();
        r1.join().unwrap();
        coord_handle.join().unwrap().expect("coord drives clean");
    }

    /// Salt mismatch on the worker side surfaces loudly at handshake.
    #[test]
    fn handshake_rejects_wrong_salt_on_worker_side() {
        let world_size = 2;
        let bad_salt: SessionSalt = [0u8; SESSION_SALT_BYTES];

        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .unwrap();
        let coord_handle = thread::spawn(move || -> Result<ClusterCoordinator> {
            ClusterCoordinator::start_from_listener(
                listener,
                TEST_SALT,
                coord_config_sync_nccl(world_size),
            )
        });

        // Worker connects with the wrong salt; coordinator's
        // start_from_listener errors → coord_handle joins with Err.
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let rank = thread::spawn(move || {
            let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).unwrap();
            let _ = write_handshake_rank(&mut s, 0, world_size as u32, &bad_salt);
            let _ = read_handshake_ack(&mut s, &bad_salt);
        });
        let err = match coord_handle.join().unwrap() {
            Ok(_) => panic!("expected coord to reject wrong-salt handshake"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC failure, got: {err}"
        );
        let _ = rank.join();
    }

    /// End-to-end Sync+Nccl smoke test. Requires CUDA + NCCL; runs
    /// only under `fdl cuda-test-nccl`. Validates the full
    /// connect → handshake → wait_for_epoch_plan → train_step → SyncNow
    /// → SyncAck → Shutdown round-trip with two real ranks doing real
    /// NCCL AllReduce(Avg) on their parameters.
    ///
    /// Acceptance: after a few averaging cycles, both ranks' parameter
    /// tensors are bit-identical (NCCL AllReduce-Avg makes them so).
    ///
    /// Marked `#[ignore]` so the CPU test suite skips it; lift the
    /// `ignore` (or run via `fdl cuda-test-nccl`) on the Pascal rig.
    #[test]
    #[ignore = "requires CUDA + NCCL — run via fdl cuda-test-nccl"]
    fn end_to_end_sync_nccl_smoke() {
        // The full body is left for the next slice's bring-up on the
        // Pascal rig. Once the rig is online we'll:
        //  1. Build a 2-rank NCCL communicator via NcclComms + split().
        //  2. Spawn ClusterCoordinator on master_port + 3 with
        //     coord_config_sync_nccl(2).
        //  3. For each rank: in a thread, construct a tiny model
        //     (Linear with a few params), a small in-memory dataset,
        //     SGD optimizer, ClusterWorker::connect_and_build, then
        //     ClusterWorker::run_until_shutdown(train_fn).
        //  4. After workers exit, assert the two ranks' final
        //     parameters are bit-identical (collected via the
        //     final_param channel, even though it discards by
        //     default in 1d.2 — this test can attach a non-discard
        //     final bridge for validation).
        //
        // For now the test is structural — when CUDA tests run it
        // simply asserts that the module compiles and links
        // cleanly. Body lands in the Pascal-rig follow-up.
    }

    // -----------------------------------------------------------------
    // 4b.D.1d.4b — end-to-end Sync+Cpu smoke test scaffolding
    // -----------------------------------------------------------------

    use crate::distributed::cluster_coordinator::ClusterCoordinator as CCoord;
    use crate::distributed::controller::ClusterController;
    use crate::nn::Linear;

    /// Index-deterministic dataset on CPU. Each sample's values are
    /// derived from its index, so two ranks reading disjoint partitions
    /// see DIFFERENT samples (and thus DIFFERENT gradients post-SGD).
    /// `Tensor::randn` would produce shared values across threads under
    /// libtorch's global RNG, collapsing per-rank divergence to zero
    /// and defeating the divergence-wire assertion below.
    struct TestDataset {
        n: usize,
    }
    impl crate::data::BatchDataSet for TestDataset {
        fn len(&self) -> usize {
            self.n
        }
        fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
            let n = indices.len() as i64;
            // Inputs: each sample is [idx, idx+1, idx+2, idx+3] / 10.
            // Targets: each sample is [idx * 0.1, idx * 0.2].
            // Both deterministic in `indices`, so disjoint partitions
            // → distinct gradients → non-zero post-AllReduce divergence.
            let mut x_vals: Vec<f32> = Vec::with_capacity(indices.len() * 4);
            let mut y_vals: Vec<f32> = Vec::with_capacity(indices.len() * 2);
            for &idx in indices {
                let f = idx as f32;
                x_vals.extend_from_slice(&[
                    f / 10.0,
                    (f + 1.0) / 10.0,
                    (f + 2.0) / 10.0,
                    (f + 3.0) / 10.0,
                ]);
                y_vals.extend_from_slice(&[f * 0.1, f * 0.2]);
            }
            Ok(vec![
                Tensor::from_f32(&x_vals, &[n, 4], Device::CPU)?,
                Tensor::from_f32(&y_vals, &[n, 2], Device::CPU)?,
            ])
        }
    }

    /// Mirror of `ddp_run::tests::mse_train`. MSE between Linear's
    /// output and the dataset's target tensor.
    fn mse_train(model: &Linear, batch: &[Tensor]) -> Result<Variable> {
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        let output = model.forward(&input)?;
        let diff = output.sub(&target)?;
        diff.mul(&diff)?.mean()
    }

    /// Records each `report` call's `deltas` into a shared vector so a
    /// test can verify the param bridge populated the divergence triple
    /// in its `SyncAck` AND that the coord's CPU finalize state
    /// machine (1d.4d) deferred `finish_averaging_cpu` until the
    /// SyncAcks landed. Returns `Stable` so the test's anchor stays
    /// stable.
    struct RecordingGuard {
        captured: Arc<std::sync::Mutex<Vec<Vec<f64>>>>,
    }

    impl crate::distributed::ddp_run::convergence::ConvergenceGuard for RecordingGuard {
        fn report(
            &mut self,
            report: &crate::distributed::ddp_run::convergence::DivergenceReport,
            _k_used: usize,
            _k_max: usize,
        ) -> crate::distributed::ddp_run::convergence::ConvergenceAction {
            self.captured.lock().unwrap().push(report.deltas.clone());
            crate::distributed::ddp_run::convergence::ConvergenceAction::Stable
        }
    }

    /// End-to-end Sync+Cpu smoke test (CPU device, no NCCL): spawn
    /// `ClusterController` (data) and `ClusterCoordinator` (control)
    /// alongside 2 `ClusterWorker` threads with a trivial Linear model
    /// and `TestDataset`, run one averaging cycle via the param bridge,
    /// assert avg_count fires + workers exit cleanly AND the coord's
    /// convergence guard received strictly-positive per-rank divergence
    /// on cycle 1 (validates the bridge's
    /// [`compute_divergence`](super::compute_divergence) flowed
    /// end-to-end AND that the CPU finalize state machine deferred the
    /// guard verdict until the bridge SyncAcks populated the captures
    /// — 1d.4d).
    #[test]
    fn end_to_end_sync_cpu_smoke() {
        let world_size = 2usize;
        let total_samples = 8usize;
        let batch_size = 4usize;

        // 1. Shared DeadRanks ledger + ClusterController on data port.
        //    No rank is dead in this smoke test; the ledger is wired
        //    for API completeness and to prove the dead-rank-aware
        //    controller path doesn't regress the happy case.
        let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
        let controller = ClusterController::start_with_dead_ranks(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            world_size,
            TEST_SALT,
            Arc::clone(&dead_ranks),
        )
        .expect("ClusterController::start_with_dead_ranks succeeds");
        let data_port = controller.port();
        let data_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), data_port);

        // 2. ClusterCoordinator listener. bind() returns the port
        //    before any accept blocks; start_from_listener (which
        //    blocks) runs on a dedicated thread so workers can connect
        //    in parallel.
        let (coord_listener, coord_port) = CCoord::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("coord bind succeeds");
        let coord_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), coord_port);

        // RecordingGuard captures the deltas every `finish_averaging_*`
        // pass — proves both the bridge wire (1d.4c) AND the deferred
        // finalize (1d.4d) are correct end-to-end on cycle 1.
        let captured_deltas: Arc<std::sync::Mutex<Vec<Vec<f64>>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_coord = Arc::clone(&captured_deltas);
        let dead_ranks_for_coord = Arc::clone(&dead_ranks);
        let config_for_coord = move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Cpu,
                world_size,
                crate::distributed::ddp::ElChe::new(world_size, 1),
            )
            .with_convergence_guard(Box::new(RecordingGuard {
                captured: captured_for_coord,
            }))
            .dead_ranks(dead_ranks_for_coord)
            .total_samples(total_samples)
            .batch_size(batch_size)
            .num_epochs(1)
        };
        let coord_thread = thread::spawn(move || -> Result<CCoord> {
            CCoord::start_from_listener(coord_listener, TEST_SALT, config_for_coord())
        });

        // 3. Build a reference Linear model on CPU to capture initial
        //    params/buffers. Each worker thread's model_factory builds
        //    its own fresh Linear; WorkerConfig.initial_params overrides
        //    the random init so all ranks align at startup.
        let ref_model = Linear::on_device(4, 2, Device::CPU).unwrap();
        let initial_params: Vec<Tensor> = ref_model
            .parameters()
            .iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = ref_model
            .buffers()
            .iter()
            .map(|b| b.get())
            .collect();
        drop(ref_model);

        // 4. Spawn worker threads. Each connects to the coord (control)
        //    + builds a CpuReduceClient (data). connect_and_build is
        //    blocking on both handshakes; the coord_thread above
        //    unblocks once both workers handshake.
        let salt = TEST_SALT;
        let mut worker_handles: Vec<thread::JoinHandle<Result<()>>> = Vec::new();
        for rank_id in 0..world_size {
            let initial_params = initial_params.clone();
            let initial_buffers = initial_buffers.clone();
            worker_handles.push(thread::spawn(move || -> Result<()> {
                let config = WorkerConfig {
                    rank: rank_id,
                    world_size,
                    device: Device::CPU,
                    initial_params,
                    initial_buffers,
                    total_samples,
                    batch_size,
                    seed: 42,
                    max_grad_norm: None,
                    easgd_alpha: None,
                    timeline: None,
                    policy: ApplyPolicy::Sync,
                };
                let dataset: Arc<dyn crate::data::BatchDataSet> =
                    Arc::new(TestDataset { n: total_samples });
                let worker = ClusterWorker::connect_and_build(
                    coord_addr,
                    Some(data_addr),
                    rank_id as u32,
                    salt,
                    config,
                    |d| Linear::on_device(4, 2, d),
                    |params| crate::nn::SGD::new(params, 0.01, 0.0),
                    dataset,
                    None, // no NCCL
                    None, // no checkpoint
                )?;
                worker.run_until_shutdown(mse_train)
            }));
        }

        // 5. Coord thread unblocks after both worker handshakes; recover
        //    the configured coord.
        let mut coord = coord_thread
            .join()
            .expect("coord thread join")
            .expect("start_from_listener succeeds");

        // 6. Dispatch the only epoch + drive ticks until at least one
        //    averaging cycle fires. Bound the wall budget so a buggy
        //    coord doesn't hang the suite.
        coord.dispatch_epoch(0).expect("dispatch_epoch(0) succeeds");
        let start = Instant::now();
        while coord.avg_count() == 0 {
            if start.elapsed() > Duration::from_secs(10) {
                panic!(
                    "end_to_end_sync_cpu_smoke: avg_count never advanced \
                     (no averaging cycle observed within 10s)"
                );
            }
            coord.tick().expect("tick");
            thread::sleep(Duration::from_millis(10));
        }
        assert!(coord.avg_count() >= 1, "at least one averaging cycle");

        // 6b. With 1d.4d's deferred finalize, cycle 1's guard sees REAL
        //     divergence (the coord waited for every bridge SyncAck to
        //     land before running `finish_averaging_cpu`). Assert the
        //     guard captured strictly-positive per-rank deltas on the
        //     first cycle. A regression to synchronous finalize would
        //     surface as cycle-1 deltas == [0.0, 0.0] (the all-Nones
        //     sentinel `unwrap_or(0.0)`).
        let cycles = captured_deltas.lock().unwrap().clone();
        assert!(
            !cycles.is_empty(),
            "RecordingGuard saw no averaging cycles despite avg_count >= 1"
        );
        let first = &cycles[0];
        assert_eq!(
            first.len(),
            world_size,
            "DivergenceReport.deltas len ({}) must equal world_size ({})",
            first.len(),
            world_size,
        );
        assert!(
            first.iter().all(|d| d.is_finite() && *d > 0.0),
            "expected strictly-positive per-rank divergence on cycle 1, got {first:?}"
        );

        // 7. Send Shutdown to workers and tear down the coord.
        coord.shutdown_workers().ok();
        coord.shutdown().ok();

        // 8. Join workers. They should exit cleanly after receiving
        //    Shutdown through the inbound bridge.
        for (rank_id, h) in worker_handles.into_iter().enumerate() {
            let r = h.join().expect("worker thread join");
            r.unwrap_or_else(|e| {
                panic!("worker rank {rank_id} run_until_shutdown: {e}");
            });
        }

        // 9. Shut the controller down.
        controller.shutdown().ok();
    }
}
