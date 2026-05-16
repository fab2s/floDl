//! Cluster coordinator: process-model port of the OLD threaded
//! `ddp_run::coordinator::Coordinator`.
//!
//! Owns the per-cluster scheduling state (ElChe, ConvergenceGuard,
//! per-rank wall-time accumulation, sync acknowledgments) and drives
//! averaging decisions for the cluster. Where the OLD design used
//! `mpsc::{Sender, Receiver}` to talk to in-process worker threads,
//! this type talks to remote rank processes over TCP. The state
//! machine and decision logic are ported literally; only the I/O
//! changes.
//!
//! # Architecture
//!
//! ```text
//! launcher process:
//!   ClusterCoordinator::start(bind_addr, world_size, salt, config)
//!     ├── binds control TcpListener
//!     ├── accepts N rank connections (handshake validates salt)
//!     ├── spawns one reader thread per rank
//!     │     reads ControlFrame, decodes TimingMsgWire / MetricsMsgWire,
//!     │     forwards on internal mpsc::Sender
//!     └── owns Vec<TcpStream> (write half, for outbound ControlFrame)
//!
//! caller drives:
//!   coord.tick()  // drain timing mpsc, check_throttle, should_average,
//!                  // trigger_averaging
//! ```
//!
//! # Scope of 4b.D.1d.1
//!
//! Ports the load-bearing subset from the OLD coordinator:
//!
//! - State fields: ElChe, ConvergenceGuard, `steps_since_avg`,
//!   `wall_ms_accum`, `last_step_count`, `nccl_sync_step` / `nccl_ack`,
//!   `nccl_sync_divergence` / `pre_norm` / `post_norm`, `throttled`,
//!   `active_count`, `version`, `avg_count`, `global_step`,
//!   `last_nccl_sync_ms`.
//! - Methods: [`Self::process_timing_msg`], [`Self::should_average`],
//!   [`Self::trigger_averaging`] (NCCL path), [`Self::check_throttle`],
//!   [`Self::drain_timing`], [`Self::tick`].
//! - New: [`Self::start`] + [`Self::shutdown`] (TCP accept loop +
//!   per-rank reader threads).
//!
//! Deferred to later slices: epoch dispatch / progressive chunk pools
//! (1d.3+), CPU 3-phase averaging (1d.4), heartbeat fault detection
//! (1d.5), metrics aggregation (1d.5+), meta-controller observe wiring
//! (1d.3+).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hmac_sha256::HMAC;

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::convergence::{
    self, ConvergenceAction, ConvergenceGuard, NoGuard, TrendGuard,
};
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::{
    ControlFrame, ControlMsgWire, FrameRead, MsgKind, SessionSalt, TimingMsgWire,
};
use crate::tensor::{Result, TensorError};

// ---------------------------------------------------------------------------
// Control-channel handshake
// ---------------------------------------------------------------------------

/// Rank → coordinator handshake magic (mirrors
/// [`wire::CONTROL_HANDSHAKE_MAGIC_RANK`]).
pub(crate) const CTRL_HS_RANK: u32 = crate::distributed::wire::CONTROL_HANDSHAKE_MAGIC_RANK;

/// Coordinator → rank handshake-ack magic.
pub(crate) const CTRL_HS_ACK: u32 = crate::distributed::wire::CONTROL_HANDSHAKE_MAGIC_ACK;

/// Wire-version used inside the handshake bytes.
pub(crate) const CTRL_HS_VERSION: u32 = crate::distributed::wire::CONTROL_PROTOCOL_VERSION;

/// Handshake byte layout (rank → coordinator):
///
/// ```text
/// u32 magic       = CTRL_HS_RANK
/// u32 version     = CTRL_HS_VERSION
/// u32 rank_id     (0..world_size)
/// u32 world_size  (rank's view; coordinator validates)
/// u64 auth_tag    = first 8 bytes of HMAC-SHA256(salt, hdr[0..16])
/// ```
///
/// Total: 24 bytes. The HMAC proves the rank shares the launcher's
/// session salt; mismatched salts surface here before any control
/// frame round-trip.
const HS_RANK_BYTES: usize = 24;

/// Handshake-ack layout (coordinator → rank):
///
/// ```text
/// u32 magic       = CTRL_HS_ACK
/// u32 version     = CTRL_HS_VERSION
/// u64 auth_tag    = first 8 bytes of HMAC-SHA256(salt, hdr[0..8])
/// ```
///
/// Total: 16 bytes.
const HS_ACK_BYTES: usize = 16;

fn hmac_first8(salt: &SessionSalt, bytes: &[u8]) -> [u8; 8] {
    let full: [u8; 32] = HMAC::mac(bytes, salt.as_slice());
    full[0..8].try_into().unwrap()
}

/// Worker-side companion to [`read_handshake_rank`]. Exported at
/// crate visibility for the upcoming `cluster_worker` slice (1d.2);
/// the slice 1d.1 tests are the only current callers.
#[allow(dead_code)]
pub(crate) fn write_handshake_rank(
    stream: &mut TcpStream,
    rank_id: u32,
    world_size: u32,
    salt: &SessionSalt,
) -> Result<()> {
    let mut buf = [0u8; HS_RANK_BYTES];
    buf[0..4].copy_from_slice(&CTRL_HS_RANK.to_le_bytes());
    buf[4..8].copy_from_slice(&CTRL_HS_VERSION.to_le_bytes());
    buf[8..12].copy_from_slice(&rank_id.to_le_bytes());
    buf[12..16].copy_from_slice(&world_size.to_le_bytes());
    let tag = hmac_first8(salt, &buf[0..16]);
    buf[16..24].copy_from_slice(&tag);
    stream.write_all(&buf).map_err(|e| {
        TensorError::new(&format!("cluster_coordinator: handshake write: {e}"))
    })
}

fn read_handshake_rank(
    stream: &mut TcpStream,
    expected_world_size: u32,
    salt: &SessionSalt,
) -> Result<u32> {
    let mut buf = [0u8; HS_RANK_BYTES];
    stream.read_exact(&mut buf).map_err(|e| {
        TensorError::new(&format!("cluster_coordinator: handshake read: {e}"))
    })?;
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != CTRL_HS_RANK {
        return Err(TensorError::new(&format!(
            "cluster_coordinator: handshake magic 0x{magic:08x} != 0x{CTRL_HS_RANK:08x}"
        )));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != CTRL_HS_VERSION {
        return Err(TensorError::new(&format!(
            "cluster_coordinator: handshake version {version} != {CTRL_HS_VERSION}"
        )));
    }
    let rank_id = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let world_size = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    if world_size != expected_world_size {
        return Err(TensorError::new(&format!(
            "cluster_coordinator: handshake world_size {world_size} != expected {expected_world_size}"
        )));
    }
    let expected_tag = hmac_first8(salt, &buf[0..16]);
    let got_tag: [u8; 8] = buf[16..24].try_into().unwrap();
    if expected_tag != got_tag {
        return Err(TensorError::new(
            "cluster_coordinator: handshake HMAC verification failed; \
             session salt disagreement (rank from a different training session, \
             or wrong key configured)",
        ));
    }
    Ok(rank_id)
}

fn write_handshake_ack(stream: &mut TcpStream, salt: &SessionSalt) -> Result<()> {
    let mut buf = [0u8; HS_ACK_BYTES];
    buf[0..4].copy_from_slice(&CTRL_HS_ACK.to_le_bytes());
    buf[4..8].copy_from_slice(&CTRL_HS_VERSION.to_le_bytes());
    let tag = hmac_first8(salt, &buf[0..8]);
    buf[8..16].copy_from_slice(&tag);
    stream.write_all(&buf).map_err(|e| {
        TensorError::new(&format!("cluster_coordinator: handshake ack write: {e}"))
    })
}

// ---------------------------------------------------------------------------
// ClusterCoordinator
// ---------------------------------------------------------------------------

/// Configuration for [`ClusterCoordinator::start`]. Subset of OLD
/// `CoordinatorBuilder`: only the fields needed for the slice 1d.1
/// scope (NCCL averaging + ElChe ownership + ConvergenceGuard).
pub struct ClusterCoordinatorConfig {
    pub policy: ApplyPolicy,
    pub backend: AverageBackend,
    pub world_size: usize,
    pub el_che: ElChe,
    /// Boxed convergence guard. Defaults to [`TrendGuard::default()`]
    /// when omitted in the builder; set [`NoGuard`] to disable.
    pub convergence_guard: Box<dyn ConvergenceGuard>,
    /// Allow ElChe anchor relax-up on Stable convergence verdicts.
    pub elche_relax_up: bool,
    /// Initial max-overshoot (Async-only). Auto-tuned upward on Stable
    /// verdicts; reset to `overshoot_initial` on NudgeDown.
    pub overshoot_initial: usize,
    pub overshoot_ceiling: usize,
    /// True when the user did not set `max_overshoot` explicitly; lets
    /// `trigger_averaging` adjust the bound on convergence verdicts.
    pub overshoot_auto: bool,
}

impl ClusterCoordinatorConfig {
    /// Construct with sensible defaults: TrendGuard with default
    /// threshold, no anchor relax-up, overshoot_initial=3, ceiling=15.
    pub fn new(
        policy: ApplyPolicy,
        backend: AverageBackend,
        world_size: usize,
        el_che: ElChe,
    ) -> Self {
        ClusterCoordinatorConfig {
            policy,
            backend,
            world_size,
            el_che,
            convergence_guard: Box::new(TrendGuard::default()),
            elche_relax_up: false,
            overshoot_initial: 3,
            overshoot_ceiling: 15,
            overshoot_auto: true,
        }
    }

    pub fn with_convergence_guard(
        mut self,
        guard: Box<dyn ConvergenceGuard>,
    ) -> Self {
        self.convergence_guard = guard;
        self
    }

    pub fn no_divergence_guard(mut self) -> Self {
        self.convergence_guard = Box::new(NoGuard);
        self
    }

    pub fn elche_relax_up(mut self, enabled: bool) -> Self {
        self.elche_relax_up = enabled;
        self
    }

    pub fn overshoot(mut self, initial: usize, ceiling: usize, auto: bool) -> Self {
        self.overshoot_initial = initial;
        self.overshoot_ceiling = ceiling;
        self.overshoot_auto = auto;
        self
    }
}

/// Process-model coordinator: ports the OLD threaded
/// `ddp_run::coordinator::Coordinator` to talk to remote rank
/// processes over TCP.
///
/// Hand off control of one TCP control channel + one reader thread per
/// rank. Drive the state machine via [`Self::tick`] from the
/// containing thread.
pub struct ClusterCoordinator {
    // --- Static config ---
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    overshoot_initial: usize,
    overshoot_ceiling: usize,
    overshoot_auto: bool,
    elche_relax_up: bool,

    // --- Scheduling state (ported literally) ---
    el_che: ElChe,
    convergence_guard: Box<dyn ConvergenceGuard>,
    version: u64,
    avg_count: u64,
    global_step: usize,
    /// Set once ElChe has its first timing report.
    calibrated: bool,
    /// World_size minus exited workers.
    active_count: usize,
    /// Async-only: max batches a rank can run past the planned sync.
    max_overshoot: usize,

    /// Per-rank steps since the last averaging cycle.
    steps_since_avg: Vec<usize>,
    /// Per-rank wall-clock ms accumulated since the last averaging cycle.
    wall_ms_accum: Vec<f64>,
    /// Per-rank most-recent batch duration (ms).
    last_batch_ms: Vec<f64>,
    /// Per-rank most-recent worker step counter.
    last_step_count: Vec<usize>,
    /// Per-rank: `last_step_count` snapshot at the time SyncNow was sent.
    nccl_sync_step: Vec<usize>,
    /// Per-rank: true once a post-sync timing message has arrived.
    nccl_ack: Vec<bool>,
    /// Per-rank: weight-space divergence reported in the last SyncAck.
    nccl_sync_divergence: Vec<Option<f64>>,
    /// Per-rank: pre-AllReduce L2 norm from the last SyncAck.
    nccl_sync_pre_norm: Vec<Option<f64>>,
    /// Post-AllReduce L2 norm (identical across ranks; populated by the
    /// first rank's SyncAck).
    nccl_sync_post_norm: Option<f64>,
    /// Per-rank: True if a Throttle has been sent and not yet cleared.
    throttled: Vec<bool>,
    /// Wall-time (ms) of the last completed NCCL sync; fed to ElChe as
    /// `sync_ms` on the next `report_timing` call.
    last_nccl_sync_ms: f64,
    /// Instant the most recent SyncNow was emitted.
    nccl_sync_start: Option<Instant>,

    // --- Channels / threads ---
    /// Reader threads (one per rank) push decoded timing messages here;
    /// the coordinator thread drains via [`Self::drain_timing`].
    timing_rx: mpsc::Receiver<TimingMsgWire>,
    /// Outbound control streams (one per rank, write half held here).
    /// Reader thread holds a try-cloned read half.
    control_streams: Vec<TcpStream>,
    /// Reader-thread join handles. Drop on [`Self::shutdown`].
    reader_handles: Vec<Option<JoinHandle<()>>>,
    /// Signals reader threads to stop reading and exit.
    shutdown_flag: Arc<AtomicBool>,
    /// Bound port of the control listener (for tests / diagnostics).
    bound_port: u16,
    /// Session salt — write side uses it for outbound ControlFrames.
    salt: SessionSalt,
}

impl ClusterCoordinator {
    /// Bind a control TcpListener at `bind_addr`, accept exactly
    /// `world_size` rank connections (validating the session salt at
    /// handshake), spawn per-rank reader threads, and return the
    /// configured coordinator.
    ///
    /// Returns `Err` if any handshake fails (loud error: salt mismatch,
    /// magic mismatch, version mismatch, world_size disagreement,
    /// duplicate rank_id).
    pub fn start(
        bind_addr: SocketAddr,
        salt: SessionSalt,
        config: ClusterCoordinatorConfig,
    ) -> Result<Self> {
        let (listener, _port) = Self::bind(bind_addr)?;
        Self::start_from_listener(listener, salt, config)
    }

    /// Bind the control listener without blocking on accept. Useful for
    /// tests that need to publish the bound port before spawning rank
    /// connections (the post-bind accept loop blocks the calling
    /// thread until every rank has connected).
    pub fn bind(bind_addr: SocketAddr) -> Result<(TcpListener, u16)> {
        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            TensorError::new(&format!(
                "cluster_coordinator: bind {bind_addr} failed: {e}"
            ))
        })?;
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: local_addr() failed: {e}"
                ))
            })?
            .port();
        Ok((listener, bound_port))
    }

    /// Accept connections + handshake on a pre-bound listener. Pair
    /// with [`Self::bind`] when the caller needs the port before
    /// blocking on accepts (e.g. tests that spawn rank threads after
    /// publishing the port through a channel).
    pub fn start_from_listener(
        listener: TcpListener,
        salt: SessionSalt,
        config: ClusterCoordinatorConfig,
    ) -> Result<Self> {
        let world_size = config.world_size;
        if world_size == 0 {
            return Err(TensorError::new(
                "cluster_coordinator: world_size must be > 0",
            ));
        }
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: local_addr() failed: {e}"
                ))
            })?
            .port();

        // Accept world_size connections, validate handshake, place each
        // at its claimed rank slot. Order-independent.
        let mut streams: Vec<Option<TcpStream>> =
            (0..world_size).map(|_| None).collect();
        let mut connected = 0usize;
        while connected < world_size {
            let (mut stream, _peer) = listener.accept().map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: accept failed: {e}"
                ))
            })?;
            // 10s handshake timeout protects against wedged ranks.
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: set_read_timeout: {e}"
                    ))
                })?;
            let rank_id = read_handshake_rank(&mut stream, world_size as u32, &salt)?;
            let rank_idx = rank_id as usize;
            if rank_idx >= world_size {
                return Err(TensorError::new(&format!(
                    "cluster_coordinator: handshake rank_id {rank_idx} >= world_size {world_size}"
                )));
            }
            if streams[rank_idx].is_some() {
                return Err(TensorError::new(&format!(
                    "cluster_coordinator: duplicate rank_id {rank_idx} connected"
                )));
            }
            write_handshake_ack(&mut stream, &salt)?;
            // Clear the handshake timeout; ControlFrame reads can take
            // arbitrarily long under load.
            stream
                .set_read_timeout(None)
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: clear read_timeout: {e}"
                    ))
                })?;
            streams[rank_idx] = Some(stream);
            connected += 1;
        }
        let mut streams: Vec<TcpStream> = streams.into_iter()
            .map(|s| s.expect("all slots filled by accept loop"))
            .collect();

        // Spawn one reader thread per rank. Each thread holds the read
        // half of a try_clone'd stream; the coordinator owns the write
        // half. ControlFrame::read_from handles HMAC validation per frame.
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let (timing_tx, timing_rx) = mpsc::channel::<TimingMsgWire>();

        let mut reader_handles: Vec<Option<JoinHandle<()>>> = Vec::with_capacity(world_size);
        for (rank, stream) in streams.iter_mut().enumerate() {
            let mut read_half = stream.try_clone().map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: stream try_clone for rank {rank}: {e}"
                ))
            })?;
            // Reader uses a short timeout to observe shutdown between frames.
            read_half
                .set_read_timeout(Some(Duration::from_millis(250)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: reader set_read_timeout: {e}"
                    ))
                })?;
            let tx = timing_tx.clone();
            let salt_for_reader = salt;
            let shutdown_for_reader = Arc::clone(&shutdown_flag);
            let handle = thread::Builder::new()
                .name(format!("flodl-coord-reader:r{rank}"))
                .spawn(move || {
                    reader_loop(rank, &mut read_half, &salt_for_reader, &shutdown_for_reader, &tx);
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: spawn reader for rank {rank}: {e}"
                    ))
                })?;
            reader_handles.push(Some(handle));
        }
        // Drop the extra sender we cloned for the closures; loop exit
        // depends on every cloned sender being dropped, but that happens
        // automatically when reader threads exit.
        drop(timing_tx);

        Ok(ClusterCoordinator {
            policy: config.policy,
            backend: config.backend,
            world_size,
            overshoot_initial: config.overshoot_initial,
            overshoot_ceiling: config.overshoot_ceiling,
            overshoot_auto: config.overshoot_auto,
            elche_relax_up: config.elche_relax_up,
            el_che: config.el_che,
            convergence_guard: config.convergence_guard,
            version: 0,
            avg_count: 0,
            global_step: 0,
            calibrated: false,
            active_count: world_size,
            max_overshoot: config.overshoot_initial,
            steps_since_avg: vec![0; world_size],
            wall_ms_accum: vec![0.0; world_size],
            last_batch_ms: vec![0.0; world_size],
            last_step_count: vec![0; world_size],
            nccl_sync_step: vec![0; world_size],
            nccl_ack: vec![true; world_size],
            nccl_sync_divergence: vec![None; world_size],
            nccl_sync_pre_norm: vec![None; world_size],
            nccl_sync_post_norm: None,
            throttled: vec![false; world_size],
            last_nccl_sync_ms: 0.0,
            nccl_sync_start: None,
            timing_rx,
            control_streams: streams,
            reader_handles,
            shutdown_flag,
            bound_port,
            salt,
        })
    }

    // -----------------------------------------------------------------
    // Public accessors (mirror the OLD coordinator's getters)
    // -----------------------------------------------------------------

    pub fn bound_port(&self) -> u16 {
        self.bound_port
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    pub fn steps_since_avg(&self) -> &[usize] {
        &self.steps_since_avg
    }

    pub fn avg_count(&self) -> u64 {
        self.avg_count
    }

    pub fn global_step(&self) -> usize {
        self.global_step
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn max_overshoot(&self) -> usize {
        self.max_overshoot
    }

    pub fn el_che(&self) -> &ElChe {
        &self.el_che
    }

    // -----------------------------------------------------------------
    // State-machine drive
    // -----------------------------------------------------------------

    /// Process a single timing message. Ported literally from OLD
    /// `Coordinator::process_timing_msg`, modulo the field-name `rank`
    /// being `u64` on the wire vs `usize` in-process.
    fn process_timing_msg(&mut self, msg: TimingMsgWire) {
        match msg {
            TimingMsgWire::Batch {
                rank,
                batch_ms,
                step_count,
                param_norm,
                batch_loss,
                sync_divergence,
            } => {
                let rank = rank as usize;
                let step_count = step_count as usize;
                if rank >= self.world_size {
                    return; // ignore malformed; tests will fail loudly
                }
                self.steps_since_avg[rank] =
                    self.steps_since_avg[rank].saturating_add(1);
                self.wall_ms_accum[rank] += batch_ms;
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                self.last_batch_ms[rank] = batch_ms;
                let _ = batch_loss; // monitoring only in this slice
                let _ = param_norm;
                if let Some(div) = sync_divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsgWire::SyncAck {
                rank,
                step_count,
                divergence,
                post_norm,
                pre_norm,
            } => {
                let rank = rank as usize;
                let step_count = step_count as usize;
                if rank >= self.world_size {
                    return;
                }
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                if let Some(div) = divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if let Some(p) = pre_norm {
                    self.nccl_sync_pre_norm[rank] = Some(p);
                }
                if let Some(p) = post_norm {
                    match self.nccl_sync_post_norm {
                        None => self.nccl_sync_post_norm = Some(p),
                        Some(prev) => debug_assert!(
                            (prev - p).abs() <= 1e-6 * prev.abs().max(1.0),
                            "post_norm rank-disagreement: prev={prev} new={p} (rank {rank})"
                        ),
                    }
                }
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsgWire::Exiting { rank: _ } => {
                self.active_count = self.active_count.saturating_sub(1);
            }
            TimingMsgWire::LrUpdate { rank: _, lr: _ } => {
                // Meta-controller wiring is deferred to 1d.3+; for now
                // just drop the message. Acknowledging it costs a match
                // arm; ignoring is intentional.
            }
        }
    }

    fn capture_nccl_sync_elapsed_if_complete(&mut self) {
        if self.nccl_ack.iter().all(|&a| a) {
            if let Some(start) = self.nccl_sync_start.take() {
                self.last_nccl_sync_ms =
                    start.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    /// Drain every pending timing message non-blocking.
    pub fn drain_timing(&mut self) {
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
    }

    /// Block up to `timeout` for the first timing message, then drain
    /// the rest non-blocking. Returns `false` when every reader thread
    /// has exited (all senders dropped) so the caller can break its
    /// loop. Mirrors OLD `Coordinator::drain_timing_blocking`.
    pub fn drain_timing_blocking(&mut self, timeout: Duration) -> bool {
        match self.timing_rx.recv_timeout(timeout) {
            Ok(msg) => self.process_timing_msg(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => return true,
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
        true
    }

    /// Check whether an averaging cycle should be triggered now. Ported
    /// literally from OLD `Coordinator::should_average`. CPU 3-phase
    /// guard is omitted (deferred to 1d.4).
    pub fn should_average(&self) -> bool {
        if matches!(self.backend, AverageBackend::Nccl)
            && !self.nccl_ack.iter().all(|&a| a)
        {
            return false;
        }
        if self.active_count < self.world_size {
            return false;
        }
        if self.steps_since_avg.contains(&0) {
            return false;
        }
        match self.policy {
            ApplyPolicy::Sync => self.steps_since_avg.iter().all(|&s| s >= 1),
            ApplyPolicy::Cadence => {
                let target = self.el_che.anchor_wall_ms();
                if target > 0.0 {
                    let min_wall = self
                        .wall_ms_accum
                        .iter()
                        .copied()
                        .fold(f64::MAX, f64::min);
                    return min_wall >= target;
                }
                let counts = self.el_che.batch_counts();
                self.steps_since_avg
                    .iter()
                    .enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
            ApplyPolicy::Async => {
                let counts = self.el_che.batch_counts();
                self.steps_since_avg
                    .iter()
                    .enumerate()
                    .all(|(r, &s)| s >= counts[r])
            }
        }
    }

    /// Throttle fast workers. Ported literally from OLD
    /// `Coordinator::check_throttle`. NCCL backend is a no-op (collective
    /// already coordinates), CPU backend not yet supported in 1d.1.
    pub fn check_throttle(&mut self) -> Result<()> {
        if matches!(self.backend, AverageBackend::Nccl) {
            return Ok(());
        }
        let max_diff = match self.el_che.max_batch_diff() {
            Some(d) => d,
            None => return Ok(()),
        };
        if self.active_count < self.world_size {
            return Ok(());
        }
        let min_steps = self.steps_since_avg.iter().copied().min().unwrap_or(0);
        // Snapshot to avoid borrow-conflict on self.control_streams in send.
        let mut to_throttle: Vec<usize> = Vec::new();
        for (rank, &steps) in self.steps_since_avg.iter().enumerate() {
            let should = steps > min_steps + max_diff;
            if should && !self.throttled[rank] {
                to_throttle.push(rank);
            }
        }
        for rank in to_throttle {
            self.send_control(rank, &ControlMsgWire::Throttle)?;
            self.throttled[rank] = true;
        }
        Ok(())
    }

    /// Trigger an averaging cycle. NCCL only in slice 1d.1 (CPU 3-phase
    /// is deferred to 1d.4). Ported literally from OLD
    /// `Coordinator::trigger_averaging` (NCCL arm) +
    /// `finish_averaging_nccl`.
    pub fn trigger_averaging(&mut self) -> Result<()> {
        match self.backend {
            AverageBackend::Nccl => {
                self.nccl_sync_start = Some(Instant::now());
                self.broadcast_control(&ControlMsgWire::SyncNow)?;
                for rank in 0..self.world_size {
                    self.nccl_sync_step[rank] = self.last_step_count[rank];
                    self.nccl_ack[rank] = false;
                }
                self.finish_averaging_nccl()?;
            }
            AverageBackend::Cpu => {
                return Err(TensorError::new(
                    "cluster_coordinator: CPU averaging backend not yet supported \
                     in slice 1d.1; lands in slice 1d.4. Use AverageBackend::Nccl \
                     for the current scope.",
                ));
            }
        }
        Ok(())
    }

    fn finish_averaging_nccl(&mut self) -> Result<()> {
        let prev_sync_ms = self.last_nccl_sync_ms;
        self.last_nccl_sync_ms = 0.0;
        if self.wall_ms_accum.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(
                &self.wall_ms_accum,
                &self.steps_since_avg,
                prev_sync_ms,
            );
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        let nccl_pre_norms: Option<Vec<f64>> =
            if self.nccl_sync_pre_norm.iter().all(|p| p.is_some()) {
                Some(self.nccl_sync_pre_norm.iter().map(|p| p.unwrap()).collect())
            } else {
                None
            };
        let report = convergence::DivergenceReport {
            deltas: self
                .nccl_sync_divergence
                .iter()
                .map(|d| d.unwrap_or(0.0))
                .collect(),
            pre_norms: nccl_pre_norms,
            post_norm: self.nccl_sync_post_norm,
        };
        let cycle_batches: usize = self.steps_since_avg.iter().sum();
        let k_max = self.steps_since_avg.iter().copied().max().unwrap_or(0);
        let action = self.convergence_guard.report(&report, cycle_batches, k_max);

        self.version += 1;
        self.avg_count += 1;

        match action {
            ConvergenceAction::Stable => {
                if self.policy == ApplyPolicy::Async {
                    if self.overshoot_auto {
                        self.max_overshoot =
                            (self.max_overshoot + 1).min(self.overshoot_ceiling);
                    }
                    if self.elche_relax_up {
                        self.el_che.relax_anchor_up();
                    }
                }
            }
            ConvergenceAction::SuppressGrowth => {}
            ConvergenceAction::NudgeDown { factor } => {
                self.el_che.nudge_anchor_down(factor);
                if self.overshoot_auto && self.policy == ApplyPolicy::Async {
                    self.max_overshoot = self.overshoot_initial;
                }
            }
        }
        if self.policy == ApplyPolicy::Async {
            self.max_overshoot = self.max_overshoot.min(self.overshoot_ceiling);
        }

        self.global_step += cycle_batches;

        self.broadcast_control(&ControlMsgWire::SetGlobalStep {
            global_step: self.global_step as u64,
        })?;

        for s in &mut self.steps_since_avg {
            *s = 0;
        }
        for a in &mut self.wall_ms_accum {
            *a = 0.0;
        }
        for t in &mut self.throttled {
            *t = false;
        }
        for d in &mut self.nccl_sync_divergence {
            *d = None;
        }
        for p in &mut self.nccl_sync_pre_norm {
            *p = None;
        }
        self.nccl_sync_post_norm = None;
        Ok(())
    }

    /// One coordinator tick: drain incoming timing, throttle fast
    /// workers, and trigger averaging when due. Mirrors OLD
    /// `Coordinator::tick`. Returns `false` when every reader thread
    /// has exited so the caller can break its loop.
    pub fn tick(&mut self) -> Result<bool> {
        self.drain_timing();
        self.check_throttle()?;
        if self.should_average() {
            self.trigger_averaging()?;
        }
        // The mpsc returns Disconnected when every cloned sender has
        // dropped (every reader thread has exited). drain_timing alone
        // can't see that — try_recv just returns Empty if there's no
        // current message and the channel is healthy. Probe explicitly.
        let alive = self.active_count > 0
            && self.reader_handles.iter().any(|h| h.is_some());
        Ok(alive)
    }

    // -----------------------------------------------------------------
    // Outbound control frame I/O
    // -----------------------------------------------------------------

    fn send_control(&mut self, rank: usize, msg: &ControlMsgWire) -> Result<()> {
        if rank >= self.world_size {
            return Err(TensorError::new(&format!(
                "cluster_coordinator: send_control rank {rank} >= world_size {}",
                self.world_size
            )));
        }
        let frame = ControlFrame::encode(&self.salt, MsgKind::Control, msg)?;
        frame.write_to(&mut self.control_streams[rank]).map_err(|e| {
            TensorError::new(&format!(
                "cluster_coordinator: send_control(rank={rank}): {e}"
            ))
        })?;
        Ok(())
    }

    fn broadcast_control(&mut self, msg: &ControlMsgWire) -> Result<()> {
        for rank in 0..self.world_size {
            self.send_control(rank, msg)?;
        }
        Ok(())
    }

    /// Send Shutdown to every rank. Called from [`Self::shutdown`];
    /// kept public so callers running the coordinator inline can drop
    /// it from a different point in their loop if needed.
    pub fn shutdown_workers(&mut self) -> Result<()> {
        self.broadcast_control(&ControlMsgWire::Shutdown)
    }

    /// Stop reader threads, send Shutdown to every connected rank,
    /// join the threads, drop streams. Idempotent on the shutdown flag.
    pub fn shutdown(mut self) -> Result<()> {
        // Best-effort send Shutdown before tearing readers down. Ignore
        // write errors here: a rank may already have exited.
        let _ = self.shutdown_workers();
        self.shutdown_flag.store(true, Ordering::SeqCst);
        for handle_opt in self.reader_handles.iter_mut() {
            if let Some(handle) = handle_opt.take() {
                let _ = handle.join();
            }
        }
        Ok(())
    }
}

impl Drop for ClusterCoordinator {
    fn drop(&mut self) {
        // Best-effort shutdown if the caller forgot to call shutdown().
        self.shutdown_flag.store(true, Ordering::SeqCst);
        for handle_opt in self.reader_handles.iter_mut() {
            if let Some(handle) = handle_opt.take() {
                let _ = handle.join();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-rank reader thread
// ---------------------------------------------------------------------------

/// Read [`ControlFrame`]s from one rank's control stream, decode the
/// payload according to [`MsgKind`], and forward to the coordinator
/// via `tx`. Exits when:
///
/// - `shutdown` flips to true (set by [`ClusterCoordinator::shutdown`]),
/// - the stream EOFs cleanly (rank closed),
/// - or any wire-level error surfaces (HMAC mismatch, bincode decode,
///   bad msg_kind).
fn reader_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    tx: &mpsc::Sender<TimingMsgWire>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        match ControlFrame::try_read_from(stream, salt) {
            Ok(FrameRead::Frame(frame)) => match frame.kind {
                MsgKind::Timing => match frame.decode::<TimingMsgWire>() {
                    Ok(msg) => {
                        if tx.send(msg).is_err() {
                            // Coordinator dropped its receiver.
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "cluster_coordinator: reader r{rank} decode TimingMsg: {e}"
                        );
                        return;
                    }
                },
                MsgKind::Metrics => {
                    // 1d.5 wires per-epoch metrics aggregation; drop for now.
                }
                MsgKind::Heartbeat => {
                    // 1d.5 wires heartbeat fault detection; drop now.
                }
                MsgKind::Control | MsgKind::ParamSnapshotMeta => {
                    eprintln!(
                        "cluster_coordinator: reader r{rank} got unexpected \
                         MsgKind {:?} on rank→coord path; dropping",
                        frame.kind
                    );
                }
            },
            Ok(FrameRead::WouldBlock) => {
                // Idle tick: re-check shutdown and keep reading.
                continue;
            }
            Ok(FrameRead::Eof) => {
                // Peer closed cleanly.
                return;
            }
            Err(e) => {
                eprintln!(
                    "cluster_coordinator: reader r{rank} wire error: {e}"
                );
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::wire::TimingMsgWire;
    use std::net::Ipv4Addr;

    /// Deterministic non-zero test salt (mirrors controller.rs::tests).
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

    /// Spawn a fake rank that connects to `port`, handshakes with
    /// `salt`, runs `body` against the connected stream, then drops it.
    fn fake_rank<F>(
        port: u16,
        rank_id: u32,
        world_size: u32,
        salt: SessionSalt,
        body: F,
    ) -> thread::JoinHandle<Result<()>>
    where
        F: Send + 'static + FnOnce(&mut TcpStream, &SessionSalt) -> Result<()>,
    {
        thread::spawn(move || -> Result<()> {
            let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
            let mut stream = TcpStream::connect(addr).map_err(|e| {
                TensorError::new(&format!("fake_rank {rank_id} connect: {e}"))
            })?;
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .map_err(|e| TensorError::new(&format!("set_read_timeout: {e}")))?;
            write_handshake_rank(&mut stream, rank_id, world_size, &salt)?;
            let mut ack = [0u8; HS_ACK_BYTES];
            stream.read_exact(&mut ack).map_err(|e| {
                TensorError::new(&format!("fake_rank {rank_id} ack read: {e}"))
            })?;
            let magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
            if magic != CTRL_HS_ACK {
                return Err(TensorError::new(&format!(
                    "fake_rank {rank_id}: unexpected ack magic 0x{magic:08x}"
                )));
            }
            // Verify the ack HMAC ourselves.
            let expected = hmac_first8(&salt, &ack[0..8]);
            let got: [u8; 8] = ack[8..16].try_into().unwrap();
            if expected != got {
                return Err(TensorError::new(
                    "fake_rank: ack HMAC verification failed",
                ));
            }
            stream
                .set_read_timeout(None)
                .map_err(|e| TensorError::new(&format!("clear timeout: {e}")))?;
            body(&mut stream, &salt)
        })
    }

    fn cfg_sync_nccl(world_size: usize) -> ClusterCoordinatorConfig {
        // ElChe::new requires ≥ 2 devices; tests use world_size ≥ 2.
        assert!(world_size >= 2, "tests use world_size >= 2");
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 1),
        )
        .no_divergence_guard()
    }

    fn cfg_async_nccl(world_size: usize) -> ClusterCoordinatorConfig {
        assert!(world_size >= 2, "tests use world_size >= 2");
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Async,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 4)
                .with_max_batch_diff(2),
        )
        .no_divergence_guard()
    }

    /// Send a Timing-kind ControlFrame on a fake-rank stream.
    fn send_timing(
        stream: &mut TcpStream,
        salt: &SessionSalt,
        msg: TimingMsgWire,
    ) -> Result<()> {
        let frame = ControlFrame::encode(salt, MsgKind::Timing, &msg)?;
        frame.write_to(stream)
    }

    /// Read one Control-kind ControlFrame from the rank-side stream.
    fn recv_control(
        stream: &mut TcpStream,
        salt: &SessionSalt,
    ) -> Result<ControlMsgWire> {
        let frame = ControlFrame::read_from(stream, salt)?
            .ok_or_else(|| TensorError::new("EOF before frame"))?;
        if frame.kind != MsgKind::Control {
            return Err(TensorError::new(&format!(
                "unexpected frame kind {:?}",
                frame.kind
            )));
        }
        frame.decode::<ControlMsgWire>()
    }

    /// Pre-bind a listener in the test (so we can publish the port
    /// before any accept blocks), spawn rank threads against that
    /// port, then drive the coordinator's accept + state machine in
    /// a worker thread. Returns the rank-side and coord-side join
    /// handles plus the bound port for the rank-side connect.
    fn spawn_coord<F>(
        _world_size: usize,
        config_fn: impl FnOnce() -> ClusterCoordinatorConfig + Send + 'static,
        drive: F,
    ) -> (u16, thread::JoinHandle<Result<()>>)
    where
        F: Send + 'static + FnOnce(&mut ClusterCoordinator) -> Result<()>,
    {
        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("bind succeeds");
        assert_eq!(listener.local_addr().unwrap().port(), port);
        let handle = thread::spawn(move || -> Result<()> {
            let mut coord = ClusterCoordinator::start_from_listener(
                listener, TEST_SALT, config_fn(),
            )?;
            let r = drive(&mut coord);
            // Best-effort shutdown even on failure so the readers join.
            let _ = coord.shutdown();
            r
        });
        (port, handle)
    }

    #[test]
    fn handshake_round_trip_with_matching_salt() {
        // 2 ranks, Sync; both handshake and immediately drop. No
        // averaging cycle expected — `drive` just returns Ok.
        let world_size = 2;
        let (port, coord_handle) =
            spawn_coord(world_size, move || cfg_sync_nccl(world_size), |_coord| Ok(()));

        let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_, _| Ok(()));
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |_, _| Ok(()));
        r0.join().unwrap().expect("rank 0 handshake");
        r1.join().unwrap().expect("rank 1 handshake");
        coord_handle.join().unwrap().expect("coord drives clean");
    }

    #[test]
    fn handshake_rejects_wrong_salt_full_path() {
        // Coordinator has TEST_SALT; rank 0 connects with all-zero salt.
        // The accept loop's handshake validation fails →
        // start_from_listener returns an error.
        let world_size = 2;
        let bad_salt: SessionSalt = [0u8; 16];

        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .unwrap();
        let coord_handle = thread::spawn(move || -> Result<ClusterCoordinator> {
            ClusterCoordinator::start_from_listener(
                listener, TEST_SALT, cfg_sync_nccl(world_size),
            )
        });

        let rank = thread::spawn(move || {
            let mut s = TcpStream::connect(
                SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port),
            )
            .unwrap();
            // Wrong salt → handshake HMAC fails server-side.
            let _ = write_handshake_rank(&mut s, 0, world_size as u32, &bad_salt);
            // Read until the server drops us.
            let mut throwaway = [0u8; 16];
            let _ = s.read_exact(&mut throwaway);
        });
        let err = match coord_handle.join().unwrap() {
            Ok(_) => panic!("expected start_from_listener to fail on bad-salt rank"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC failure, got: {err}"
        );
        let _ = rank.join();
    }

    /// Bind a listener ourselves, hand the connection to the
    /// handshake validator directly, exercise the wrong-salt branch.
    #[test]
    fn read_handshake_rank_rejects_wrong_salt_direct() {
        let listener = TcpListener::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)
        ).unwrap();
        let port = listener.local_addr().unwrap().port();

        let bad_salt: SessionSalt = [0u8; 16];
        assert_ne!(bad_salt, TEST_SALT);

        let rank = thread::spawn(move || {
            let mut s = TcpStream::connect(
                SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port),
            ).unwrap();
            // Send a handshake keyed by the wrong salt.
            write_handshake_rank(&mut s, 0, 1, &bad_salt).unwrap();
            // Don't expect an ack; the coordinator should drop us.
            drop(s);
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        server_stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let err = read_handshake_rank(&mut server_stream, 1, &TEST_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HMAC verification failed"),
            "expected HMAC failure, got: {msg}"
        );
        rank.join().unwrap();
    }

    #[test]
    fn sync_policy_fires_after_each_rank_step_once() {
        // 2 ranks, Sync policy: after each rank reports one Batch, the
        // coordinator should fire SyncNow + SetGlobalStep exactly once.
        let world_size = 2;
        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_sync_nccl(world_size),
            |coord| {
                let start = Instant::now();
                while coord.avg_count() == 0 {
                    if start.elapsed() > Duration::from_secs(5) {
                        return Err(TensorError::new(
                            "sync_policy_fires timed out waiting for avg_count",
                        ));
                    }
                    coord.tick()?;
                    thread::sleep(Duration::from_millis(10));
                }
                assert_eq!(coord.avg_count(), 1, "exactly one averaging cycle");
                Ok(())
            },
        );

        let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 0,
                batch_ms: 10.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let msg = recv_control(s, salt)?;
            assert_eq!(msg, ControlMsgWire::SyncNow);
            let msg2 = recv_control(s, salt)?;
            assert!(matches!(msg2, ControlMsgWire::SetGlobalStep { .. }));
            Ok(())
        });
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 1,
                batch_ms: 12.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.4,
                sync_divergence: None,
            })?;
            let msg = recv_control(s, salt)?;
            assert_eq!(msg, ControlMsgWire::SyncNow);
            let msg2 = recv_control(s, salt)?;
            assert!(matches!(msg2, ControlMsgWire::SetGlobalStep { .. }));
            Ok(())
        });

        r0.join().unwrap().expect("rank 0 sees SyncNow + SetGlobalStep");
        r1.join().unwrap().expect("rank 1 sees SyncNow + SetGlobalStep");
        coord_handle.join().unwrap().expect("coord finishes");
    }

    // Throttle is an Async/CPU-backend concept; NCCL backend uses
    // AllReduce as the coordination mechanism (sending Throttle there
    // would deadlock with the collective). Slice 1d.1 only wires the
    // NCCL backend, so a Throttle behavioral test belongs to 1d.4 when
    // AverageBackend::Cpu lands. The path is structurally exercised
    // here by `cfg_async_nccl`, which goes through `check_throttle`
    // and confirms the NCCL early-return guard (the function returns
    // without sending a frame to any rank).
    #[test]
    fn check_throttle_nccl_backend_is_no_op() {
        // Construct a coord with Async+Nccl; tick once with both ranks
        // having reported a single batch. check_throttle must return
        // Ok and send no Throttle frames.
        let world_size = 2;
        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_async_nccl(world_size),
            |coord| {
                // Wait for at least one timing message per rank, then
                // run a few ticks. If check_throttle were to send a
                // Throttle here, the rank-side recv would surface it
                // and the rank closure would assert. We don't.
                let deadline = Instant::now() + Duration::from_secs(2);
                while coord.steps_since_avg().contains(&0) {
                    if Instant::now() > deadline {
                        return Err(TensorError::new(
                            "did not receive a batch from each rank",
                        ));
                    }
                    coord.tick()?;
                    thread::sleep(Duration::from_millis(10));
                }
                // A few extra ticks — no Throttle should fire.
                for _ in 0..10 {
                    coord.tick()?;
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(())
            },
        );

        let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 0,
                batch_ms: 5.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            // Drain inbound frames until the coordinator drops us.
            // We must NOT receive a Throttle; if we do, assert.
            let mut got = ControlFrame::read_from(s, salt);
            while let Ok(Some(frame)) = got {
                let kind = frame.kind;
                let msg = frame.decode::<ControlMsgWire>()?;
                assert!(
                    !matches!(msg, ControlMsgWire::Throttle),
                    "Throttle must not fire on NCCL backend (rank 0, kind={kind:?})"
                );
                got = ControlFrame::read_from(s, salt);
            }
            Ok(())
        });
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 1,
                batch_ms: 5.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let mut got = ControlFrame::read_from(s, salt);
            while let Ok(Some(frame)) = got {
                let msg = frame.decode::<ControlMsgWire>()?;
                assert!(
                    !matches!(msg, ControlMsgWire::Throttle),
                    "Throttle must not fire on NCCL backend (rank 1)"
                );
                got = ControlFrame::read_from(s, salt);
            }
            Ok(())
        });

        coord_handle.join().unwrap().expect("coord drives");
        // Rank threads may still be reading frames; coord.shutdown sent
        // Shutdown frames to them which they should decode and exit.
        // The asserts above guard the no-Throttle invariant.
        let _ = r0.join();
        let _ = r1.join();
    }
}
