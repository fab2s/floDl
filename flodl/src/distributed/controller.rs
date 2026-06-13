//! Cluster controller: TCP byte router for the star-topology
//! cross-process gradient sum that powers [`AverageBackend::Cpu`].
//!
//! [`ClusterController`] owns the data channel for CPU averaging
//! under the process-model design. ElChe scheduling + worker control
//! live on the companion
//! [`crate::distributed::cluster_coordinator::ClusterCoordinator`]
//! (control channel). The data path here is a star-topology byte
//! router; the control path is connection-per-rank.
//!
//! Architecture (star, not collective): every rank ships a `RoundFrame`
//! containing this round's tensors to a single TCP listener on the
//! launcher process. The launcher accumulates the per-tensor sum on its
//! CPU, divides by `world_size`, and writes the averaged `RoundFrame`
//! back to each rank. No NCCL. Genuinely async from NCCL's perspective:
//! ranks' GPUs keep computing while their CPUs push/pull bytes.
//!
//! Future swap-in: a gloo-backed all-reduce can replace the serial
//! summation when the single-CPU bottleneck shows up (~8+ ranks for
//! typical gradient sizes). The wire protocol is intentionally simple
//! enough that the swap touches only the inner reduce loop, not the
//! protocol or rank-side client.
//!
//! # Wire protocol (little-endian, no compression)
//!
//! Handshake (rank → controller, exactly once per connection):
//! ```text
//! u32 magic         = 0xF10D_17C0
//! u32 protocol_ver  = 1
//! u32 rank_id       (this rank's global rank, 0..world_size)
//! u32 world_size    (rank's view; controller validates against its own)
//! ```
//!
//! Handshake ack (controller → rank):
//! ```text
//! u32 magic        = 0xF10D_17C1
//! u32 protocol_ver = 1
//! ```
//!
//! RoundFrame (rank → controller, then controller → rank, identical
//! shape both directions):
//! ```text
//! u32 magic       = 0xF10D_17F1
//! u32 num_tensors
//! for each tensor:
//!   u8  dtype   (0 = f32; v1 only)
//!   u8  ndim
//!   u32 dim_0, dim_1, ..., dim_{ndim-1}
//!   u64 nbytes
//!   <nbytes> raw bytes (native byte order)
//! u64 auth_tag    = first 8 bytes of HMAC-SHA256(session_salt, frame_body)
//!                   (mismatched salts surface as a loud HMAC verification
//!                   error on the first round-trip)
//! ```
//!
//! Tensor data is native byte order. Cross-arch clusters (x86 + ARM)
//! would need a canonicalization step; out of scope for v1 (homogeneous
//! arch is the common case and the only one our test rig exercises).
//!
//! # State machine
//!
//! ```text
//! Idle → Accepting (collect N connections + N handshakes)
//!      → Reducing  (per-round: recv N frames, sum, scatter)
//!      → Shutdown  (any rank disconnects cleanly, or shutdown signal)
//! ```
//!
//! [`AverageBackend::Cpu`]: crate::distributed::AverageBackend::Cpu

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use hmac_sha256::HMAC;

use crate::distributed::relay::mux::{MuxRead, MuxRecord, RelayControlMsg};
use crate::distributed::wire::SessionSalt;
use crate::tensor::{Result, TensorError};

pub(crate) const HANDSHAKE_MAGIC_RANK: u32 = 0xF10D_17C0;
pub(crate) const HANDSHAKE_MAGIC_CONTROLLER_ACK: u32 = 0xF10D_17C1;
pub(crate) const ROUND_FRAME_MAGIC: u32 = 0xF10D_17F1;

/// Wire-protocol version for the CPU-averaging data channel.
///
/// Every [`RoundFrame`] body is followed by an 8-byte HMAC-SHA256
/// footer keyed by the session salt; a session-salt disagreement
/// surfaces as an `HMAC verification failed` error on the first
/// round-trip.
pub(crate) const PROTOCOL_VERSION: u32 = 1;

/// dtype tag for f32 in the wire protocol. Only dtype supported.
pub const DTYPE_F32: u8 = 0;

/// Shared dead-rank ledger. Set by the cluster coordinator when it
/// declares a rank dead (stale heartbeat). Read by the controller's
/// reduce loop to skip the rank's contribution in the current and future
/// rounds, and by the coord-side `should_average` /
/// `poll_cpu_averaging` gates to exclude dead ranks from quorum
/// counting.
///
/// Under the per-host relay transport, the controller no longer owns a
/// stream per rank, so there is nothing to shut down to wake it. The
/// reduce loop polls this ledger (its round-wait uses a timeout), so a
/// coord-declared death is observed on the next poll tick; rank death
/// also arrives directly as a relay
/// [`crate::distributed::relay::mux::RelayControlMsg::RankExit`].
#[derive(Debug)]
pub struct DeadRanks {
    flags: Vec<AtomicBool>,
}

impl DeadRanks {
    /// Create a fresh dead-rank ledger sized for `world_size`. All
    /// ranks start alive.
    pub fn new(world_size: usize) -> Arc<Self> {
        Arc::new(Self {
            flags: (0..world_size).map(|_| AtomicBool::new(false)).collect(),
        })
    }

    /// Declare `rank` permanently dead for the rest of this run.
    /// Idempotent flag set. No-op if `rank >= world_size`. The
    /// controller's reduce loop picks this up on its next round-wait
    /// poll tick.
    pub fn declare_dead(&self, rank: usize) {
        if let Some(flag) = self.flags.get(rank) {
            flag.store(true, Ordering::SeqCst);
        }
    }

    /// Check if `rank` is dead.
    pub fn is_dead(&self, rank: usize) -> bool {
        self.flags
            .get(rank)
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// Count of dead ranks.
    pub fn dead_count(&self) -> usize {
        self.flags
            .iter()
            .filter(|f| f.load(Ordering::SeqCst))
            .count()
    }

    /// World size the ledger was sized for.
    pub fn world_size(&self) -> usize {
        self.flags.len()
    }
}

/// Background CPU-averager. Owns a [`TcpListener`] bound to the
/// controller's address and a worker thread that runs the accept +
/// reduce loop.
///
/// Constructed via [`ClusterController::start`] (or
/// [`ClusterController::start_with_dead_ranks`] to share a dead-rank
/// ledger with the coordinator); clean shutdown via
/// [`ClusterController::shutdown`] (signals the worker, then joins).
#[derive(Debug)]
pub struct ClusterController {
    bound_port: u16,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<()>>>,
}

impl ClusterController {
    /// Bind a TCP listener at `bind_addr` and spawn the reduce thread.
    ///
    /// The thread blocks waiting for exactly `world_size` rank
    /// connections, validates each one's handshake, then runs the
    /// reduce loop until ranks disconnect or [`Self::shutdown`] is
    /// called.
    ///
    /// `salt` is the 128-bit session salt shipped via the cluster
    /// envelope. Every [`RoundFrame`] body is authenticated with an
    /// HMAC-SHA256 footer keyed by this value; a rank-side mismatch
    /// surfaces loudly on the first round-trip. Use
    /// `[0u8; SESSION_SALT_BYTES]` for in-process tests that pair this
    /// directly with a matching `CpuReduceClient`.
    ///
    /// Use `127.0.0.1:0` for tests (kernel-assigned port; read back via
    /// [`Self::port`]). Use the cluster's `controller_addr:controller_port+2`
    /// in production.
    // Production callers share a DeadRanks ledger (start_with_dead_ranks);
    // this standalone form is exercised by the controller tests.
    #[allow(dead_code)]
    pub fn start(
        bind_addr: SocketAddr,
        world_size: usize,
        salt: SessionSalt,
    ) -> Result<Self> {
        // Standalone constructor: world is fixed at startup and no
        // elastic-membership path. Equivalent to passing a private
        // ledger that nobody else can declare into.
        let dead_ranks = DeadRanks::new(world_size);
        Self::start_with_dead_ranks(bind_addr, world_size, salt, dead_ranks)
    }

    /// Like [`Self::start`] but shares the dead-rank ledger with the
    /// coordinator. When the coord declares a rank dead, the
    /// controller's reduce thread skips its contribution and divides by
    /// the surviving-rank count instead of `world_size`. Use the
    /// [`DeadRanks`] returned by [`DeadRanks::new`] (or pass the same
    /// Arc clone to both this constructor and the
    /// [`crate::distributed::cluster_coordinator::ClusterCoordinator`]
    /// via its config).
    pub fn start_with_dead_ranks(
        bind_addr: SocketAddr,
        world_size: usize,
        salt: SessionSalt,
        dead_ranks: Arc<DeadRanks>,
    ) -> Result<Self> {
        if world_size == 0 {
            return Err(TensorError::new(
                "cluster_controller: world_size must be > 0",
            ));
        }
        if dead_ranks.world_size() != world_size {
            return Err(TensorError::new(&format!(
                "cluster_controller: dead_ranks world_size ({}) must match \
                 controller world_size ({})",
                dead_ranks.world_size(),
                world_size,
            )));
        }
        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: bind {bind_addr} failed: {e}"
            ))
        })?;
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: local_addr() failed: {e}"
                ))
            })?
            .port();
        // Short accept timeout so the worker thread can observe the
        // shutdown flag between connections without blocking forever.
        listener
            .set_nonblocking(false)
            .map_err(|e| TensorError::new(&format!("cluster_controller: set_nonblocking: {e}")))?;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_cloned = Arc::clone(&shutdown);
        let handle = thread::Builder::new()
            .name(format!("flodl-cluster-controller:{bound_port}"))
            .spawn(move || {
                run_reduce_thread(listener, world_size, salt, shutdown_cloned, dead_ranks)
            })
            .map_err(|e| {
                TensorError::new(&format!("cluster_controller: spawn worker failed: {e}"))
            })?;

        Ok(ClusterController {
            bound_port,
            shutdown,
            handle: Some(handle),
        })
    }

    /// Bound TCP port. With `bind_addr.port() == 0`, returns the
    /// kernel-assigned port (test entry point); otherwise the requested
    /// port.
    pub fn port(&self) -> u16 {
        self.bound_port
    }

    /// Signal the reduce thread to stop, then join it. Idempotent.
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            return h
                .join()
                .map_err(|_| TensorError::new("cluster_controller: worker panicked"))?;
        }
        Ok(())
    }
}

impl Drop for ClusterController {
    fn drop(&mut self) {
        // Best-effort shutdown if the caller didn't explicitly call
        // shutdown(). Joins are blocking, which is fine — Drop runs at
        // process or scope exit; we want the worker out cleanly.
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Reduce-thread worker
// ---------------------------------------------------------------------------

/// Reduce-thread read poll cadence: per-host reader threads block in
/// `try_read_from` with this timeout so they re-check the shutdown flag
/// on idle ticks, and the reduce loop's round-wait re-evaluates dead
/// ranks (which can be declared externally by the coordinator with no
/// notify) on the same cadence.
const REDUCE_POLL: Duration = Duration::from_millis(100);

fn run_reduce_thread(
    listener: TcpListener,
    world_size: usize,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
    dead_ranks: Arc<DeadRanks>,
) -> Result<()> {
    listener
        .set_nonblocking(true)
        .map_err(|e| TensorError::new(&format!("cluster_controller: set_nonblocking: {e}")))?;

    let slots = Arc::new(ReduceSlots::new(world_size));
    // Sole-writer half per relay connection (the reduce loop writes
    // replies); the matching read half is owned by a per-connection
    // reader thread. `rank_conn[rank]` indexes the connection carrying
    // that rank.
    let mut conn_writes: Vec<TcpStream> = Vec::new();
    let mut rank_conn: Vec<Option<usize>> = (0..world_size).map(|_| None).collect();
    let mut reader_threads: Vec<JoinHandle<()>> = Vec::new();
    let mut covered = 0usize;

    // Phase 1: accept per-host relay connections. Each announces the ranks
    // it carries via a `RelayHello`; accept until every global rank is
    // covered exactly once.
    while covered < world_size {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                let _ = stream.set_nodelay(true);
                // Relay handshake (blocking read — relays send Hello
                // immediately on connect).
                let ranks = match MuxRecord::read_from(&mut stream, &salt)? {
                    Some(MuxRecord::Control(RelayControlMsg::Hello { host, ranks })) => {
                        crate::verbose!(
                            "  cluster_controller: relay '{host}' carries ranks {ranks:?}"
                        );
                        ranks
                    }
                    Some(other) => {
                        return Err(TensorError::new(&format!(
                            "cluster_controller: expected relay Hello, got {other:?}"
                        )));
                    }
                    None => {
                        return Err(TensorError::new(
                            "cluster_controller: relay closed connection before Hello",
                        ));
                    }
                };
                let conn_idx = conn_writes.len();
                let mut conn_ranks: Vec<usize> = Vec::with_capacity(ranks.len());
                for r in &ranks {
                    let r = *r as usize;
                    if r >= world_size {
                        return Err(TensorError::new(&format!(
                            "cluster_controller: relay announced rank {r} >= world_size {world_size}"
                        )));
                    }
                    if rank_conn[r].is_some() {
                        return Err(TensorError::new(&format!(
                            "cluster_controller: rank {r} announced by two relays"
                        )));
                    }
                    rank_conn[r] = Some(conn_idx);
                    conn_ranks.push(r);
                }
                MuxRecord::control(RelayControlMsg::HelloAck).write_to(&mut stream, &salt)?;

                let read_half = stream.try_clone().map_err(|e| {
                    TensorError::new(&format!("cluster_controller: relay try_clone: {e}"))
                })?;
                read_half
                    .set_read_timeout(Some(REDUCE_POLL))
                    .map_err(|e| {
                        TensorError::new(&format!("cluster_controller: set_read_timeout: {e}"))
                    })?;
                conn_writes.push(stream);
                covered += conn_ranks.len();

                let slots_c = Arc::clone(&slots);
                let dead_c = Arc::clone(&dead_ranks);
                let shutdown_c = Arc::clone(&shutdown);
                let t = thread::Builder::new()
                    .name(format!("flodl-controller-relay{conn_idx}"))
                    .spawn(move || {
                        reduce_reader(read_half, conn_ranks, slots_c, dead_c, shutdown_c, salt)
                    })
                    .map_err(|e| {
                        TensorError::new(&format!("cluster_controller: spawn reader: {e}"))
                    })?;
                reader_threads.push(t);
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "cluster_controller: accept failed: {e}"
                )));
            }
        }
    }
    let _ = listener.set_nonblocking(false);

    // Phase 2: reduce loop. Wait until every alive rank has deposited this
    // round's frame, average (skipping dead ranks, dividing by the alive
    // count), and scatter the averaged frame back tagged per rank down its
    // owning relay connection. The reduce loop is the SOLE writer of every
    // relay connection. Terminates when an alive rank's data connection
    // closes (clean training-end exit), every rank is dead, or shutdown is
    // signalled.
    let outcome = loop {
        if shutdown.load(Ordering::SeqCst) {
            break Ok(());
        }
        match slots.wait_for_round(&dead_ranks, &shutdown, REDUCE_POLL) {
            RoundOutcome::Frames(frames) => {
                if let Err(e) =
                    average_and_scatter(&frames, &mut conn_writes, &rank_conn, &dead_ranks, &salt)
                {
                    break Err(e);
                }
            }
            RoundOutcome::Shutdown => break Ok(()),
            RoundOutcome::Error(e) => break Err(e),
        }
    };

    // Tear down: signal the reader threads and join them so the
    // connections close cleanly before this thread returns.
    shutdown.store(true, Ordering::SeqCst);
    for t in reader_threads {
        let _ = t.join();
    }
    outcome
}

// ---------------------------------------------------------------------------
// Per-host demux: reader threads + round-collection slots
// ---------------------------------------------------------------------------

/// Shared per-round frame collection, fed by the per-connection reader
/// threads and drained by the reduce loop. One slot per rank.
struct ReduceSlots {
    inner: Mutex<SlotsInner>,
    cv: Condvar,
}

struct SlotsInner {
    /// This round's frame per rank (`None` until the rank's reader
    /// deposits it; taken by the reduce loop once all alive ranks present).
    frames: Vec<Option<RoundFrame>>,
    /// A reader observed an alive rank's data connection close (clean
    /// training-end exit, or a relay/host drop) — the reduce loop should
    /// terminate cleanly. Mirrors the pre-relay "alive-rank EOF → clean
    /// shutdown" semantics.
    shutdown: bool,
    /// A reader hit a hard wire error on an alive rank's frame. Surfaced
    /// from the reduce loop as the thread's `Err`.
    error: Option<TensorError>,
}

/// Outcome of one [`ReduceSlots::wait_for_round`] call.
enum RoundOutcome {
    /// Every alive rank's frame for this round (dead ranks are `None`).
    Frames(Vec<Option<RoundFrame>>),
    /// Clean shutdown requested (alive-rank exit, all ranks dead, or the
    /// external shutdown flag).
    Shutdown,
    /// A reader surfaced a hard wire error.
    Error(TensorError),
}

impl ReduceSlots {
    fn new(world_size: usize) -> Self {
        ReduceSlots {
            inner: Mutex::new(SlotsInner {
                frames: (0..world_size).map(|_| None).collect(),
                shutdown: false,
                error: None,
            }),
            cv: Condvar::new(),
        }
    }

    /// A reader deposits `rank`'s frame for the current round.
    fn deposit(&self, rank: usize, frame: RoundFrame) {
        let mut inner = self.inner.lock().unwrap();
        if rank < inner.frames.len() {
            inner.frames[rank] = Some(frame);
        }
        self.cv.notify_all();
    }

    /// Request a clean shutdown of the reduce loop.
    fn request_shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.shutdown = true;
        self.cv.notify_all();
    }

    /// Record the first hard wire error; surfaced by the reduce loop.
    fn set_error(&self, err: TensorError) {
        let mut inner = self.inner.lock().unwrap();
        if inner.error.is_none() {
            inner.error = Some(err);
        }
        self.cv.notify_all();
    }

    /// Block until every alive rank has deposited a frame, then take them
    /// (leaving dead ranks `None`). Re-evaluates dead ranks every `poll`
    /// so a coord-declared death (which carries no notify) is observed.
    fn wait_for_round(
        &self,
        dead: &DeadRanks,
        external_shutdown: &AtomicBool,
        poll: Duration,
    ) -> RoundOutcome {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if external_shutdown.load(Ordering::SeqCst) || inner.shutdown {
                return RoundOutcome::Shutdown;
            }
            if let Some(e) = inner.error.take() {
                return RoundOutcome::Error(e);
            }
            let ws = inner.frames.len();
            let alive: Vec<usize> = (0..ws).filter(|r| !dead.is_dead(*r)).collect();
            if alive.is_empty() {
                // Every rank dead/done → nothing left to reduce.
                return RoundOutcome::Shutdown;
            }
            if alive.iter().all(|r| inner.frames[*r].is_some()) {
                let mut out: Vec<Option<RoundFrame>> = Vec::with_capacity(ws);
                for r in 0..ws {
                    if dead.is_dead(r) {
                        inner.frames[r] = None;
                        out.push(None);
                    } else {
                        out.push(inner.frames[r].take());
                    }
                }
                return RoundOutcome::Frames(out);
            }
            let (guard, _timeout) = self.cv.wait_timeout(inner, poll).unwrap();
            inner = guard;
        }
    }
}

/// Per-connection reader: demux `Data{rank}` records into the reduce
/// slots, surface `RankExit` / EOF as clean shutdown for still-alive
/// ranks, and parse the opaque RoundFrame payload from memory.
fn reduce_reader(
    mut read: TcpStream,
    ranks: Vec<usize>,
    slots: Arc<ReduceSlots>,
    dead_ranks: Arc<DeadRanks>,
    shutdown: Arc<AtomicBool>,
    salt: SessionSalt,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        match MuxRecord::try_read_from(&mut read, &salt) {
            Ok(MuxRead::Record(MuxRecord::Data { rank, payload })) => {
                let r = rank as usize;
                if dead_ranks.is_dead(r) {
                    continue; // late frame from a dead rank — drop
                }
                let mut slice = &payload[..];
                match read_round_frame(&mut slice, &salt) {
                    Ok(Some(frame)) => slots.deposit(r, frame),
                    Ok(None) => {
                        slots.set_error(TensorError::new(&format!(
                            "cluster_controller: truncated RoundFrame payload for rank {r}"
                        )));
                        return;
                    }
                    Err(e) => {
                        slots.set_error(e);
                        return;
                    }
                }
            }
            Ok(MuxRead::Record(MuxRecord::Control(RelayControlMsg::RankExit { rank }))) => {
                // Alive rank's data connection closed: clean training-end
                // exit (or an undetected failure). Mirrors the pre-relay
                // "alive-rank EOF → clean shutdown" semantics. A RankExit
                // for an already-dead rank is expected post-failure cleanup.
                if !dead_ranks.is_dead(rank as usize) {
                    slots.request_shutdown();
                }
            }
            Ok(MuxRead::Record(MuxRecord::Control(_))) => {
                // Hello/HelloAck only occur at startup; ignore mid-stream.
            }
            Ok(MuxRead::WouldBlock) => {}
            Ok(MuxRead::Eof) => {
                // Relay connection closed. If any of this host's ranks were
                // still alive, treat as their exit (clean shutdown).
                if ranks.iter().any(|r| !dead_ranks.is_dead(*r)) {
                    slots.request_shutdown();
                }
                return;
            }
            Err(e) => {
                if ranks.iter().any(|r| !dead_ranks.is_dead(*r)) {
                    slots.set_error(e);
                }
                return;
            }
        }
    }
}

/// Average this round's alive-rank frames and scatter the result back,
/// tagged per rank, down each rank's owning relay connection. The reduce
/// loop is the sole writer of every connection.
fn average_and_scatter(
    frames: &[Option<RoundFrame>],
    conn_writes: &mut [TcpStream],
    rank_conn: &[Option<usize>],
    dead_ranks: &DeadRanks,
    salt: &SessionSalt,
) -> Result<()> {
    let averaged = reduce_average_alive(frames)?;
    // The averaged frame is identical for every rank; serialize once and
    // forward the same bytes tagged per rank.
    let mut buf: Vec<u8> = Vec::new();
    write_round_frame(&mut buf, &averaged, salt)?;
    for (rank, conn) in rank_conn.iter().enumerate() {
        if dead_ranks.is_dead(rank) {
            continue;
        }
        let Some(ci) = conn else {
            continue;
        };
        MuxRecord::data(rank as u32, buf.clone()).write_to(&mut conn_writes[*ci], salt)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------

/// Read and validate the rank-side data-channel handshake, returning the
/// announced `rank_id`. Exposed at crate visibility so the per-host relay
/// ([`crate::distributed::relay`]) can terminate the handshake toward its
/// local ranks exactly as the controller does.
pub(crate) fn read_handshake(stream: &mut TcpStream, expected_world_size: usize) -> Result<usize> {
    let mut buf = [0u8; 16];
    stream.read_exact(&mut buf).map_err(|e| {
        TensorError::new(&format!("cluster_controller: handshake read failed: {e}"))
    })?;
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != HANDSHAKE_MAGIC_RANK {
        return Err(TensorError::new(&format!(
            "cluster_controller: handshake magic 0x{magic:08x} != 0x{HANDSHAKE_MAGIC_RANK:08x}"
        )));
    }
    let proto_ver = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if proto_ver != PROTOCOL_VERSION {
        return Err(TensorError::new(&format!(
            "cluster_controller: handshake protocol_version {proto_ver} != {PROTOCOL_VERSION}"
        )));
    }
    let rank_id = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let rank_world_size = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    if rank_world_size != expected_world_size {
        return Err(TensorError::new(&format!(
            "cluster_controller: handshake world_size {rank_world_size} != expected {expected_world_size}"
        )));
    }
    Ok(rank_id)
}

/// Write the controller-side data-channel handshake ack. Exposed at
/// crate visibility for the per-host relay (see [`read_handshake`]).
pub(crate) fn write_handshake_ack(stream: &mut TcpStream) -> Result<()> {
    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&HANDSHAKE_MAGIC_CONTROLLER_ACK.to_le_bytes());
    buf[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    stream.write_all(&buf).map_err(|e| {
        TensorError::new(&format!("cluster_controller: handshake ack write failed: {e}"))
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// RoundFrame
// ---------------------------------------------------------------------------

/// A round's payload: a list of tensors with shape + dtype + data.
///
/// Identical shape sent rank→controller and controller→rank. v1 only
/// supports `DTYPE_F32`; controller errors loudly on other dtypes.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundFrame {
    pub tensors: Vec<TensorPayload>,
}

/// One tensor inside a [`RoundFrame`].
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPayload {
    /// Wire dtype tag (see [`DTYPE_F32`]).
    pub dtype: u8,
    /// Tensor shape.
    pub shape: Vec<u32>,
    /// Raw tensor bytes (native byte order).
    pub bytes: Vec<u8>,
}

impl TensorPayload {
    /// Number of element-slots in the tensor (product of shape dims).
    pub fn numel(&self) -> usize {
        self.shape.iter().map(|d| *d as usize).product()
    }
}

/// Read a RoundFrame from a single rank's stream. Returns `Ok(None)` on
/// clean EOF (rank closed its end normally — signals shutdown).
///
/// Reads the existing v1 frame body byte-for-byte, then reads the 8-byte
/// HMAC-SHA256 footer (`PROTOCOL_VERSION = 2`) and authenticates the
/// body against `salt`. Mismatched salts surface here on the very first
/// round-trip with a clear, loud error.
///
/// `pub(crate)` so the rank-side client in `cpu_reduce` can share the
/// wire format without duplication.
pub(crate) fn read_round_frame<R: Read>(
    stream: &mut R,
    salt: &SessionSalt,
) -> Result<Option<RoundFrame>> {
    let mut mac = HMAC::new(salt.as_slice());

    let mut hdr = [0u8; 8];
    match stream.read_exact(&mut hdr) {
        Ok(()) => {}
        Err(e) if matches!(e.kind(), ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset) => {
            return Ok(None);
        }
        Err(e) => {
            return Err(TensorError::new(&format!(
                "cluster_controller: frame header read failed: {e}"
            )));
        }
    }
    mac.update(hdr);
    let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    if magic != ROUND_FRAME_MAGIC {
        return Err(TensorError::new(&format!(
            "cluster_controller: frame magic 0x{magic:08x} != 0x{ROUND_FRAME_MAGIC:08x}"
        )));
    }
    let num_tensors = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;

    let mut tensors = Vec::with_capacity(num_tensors);
    for ti in 0..num_tensors {
        let mut meta = [0u8; 2];
        stream.read_exact(&mut meta).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] meta read failed: {e}"
            ))
        })?;
        mac.update(meta);
        let dtype = meta[0];
        let ndim = meta[1] as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut d = [0u8; 4];
            stream.read_exact(&mut d).map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] shape read failed: {e}"
                ))
            })?;
            mac.update(d);
            shape.push(u32::from_le_bytes(d));
        }
        let mut nb = [0u8; 8];
        stream.read_exact(&mut nb).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] nbytes read failed: {e}"
            ))
        })?;
        mac.update(nb);
        let nbytes = u64::from_le_bytes(nb) as usize;
        // Incremental allocation: nbytes is unauthenticated until the
        // trailing MAC verifies; never pre-allocate a hostile length.
        let bytes = crate::distributed::wire::read_exact_incremental(stream, nbytes)
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] data read failed: {e}"
                ))
            })?;
        mac.update(&bytes);
        tensors.push(TensorPayload {
            dtype,
            shape,
            bytes,
        });
    }

    // HMAC-SHA256-64 footer: 8 bytes, little-endian, equal to the first
    // 8 bytes of HMAC-SHA256(salt, body). Backwards-incompatible vs
    // PROTOCOL_VERSION = 1 (which had no footer).
    let mut footer = [0u8; 8];
    stream.read_exact(&mut footer).map_err(|e| {
        TensorError::new(&format!(
            "cluster_controller: frame HMAC footer read failed: {e} \
             (sender at PROTOCOL_VERSION < 2, or stream truncated mid-frame)"
        ))
    })?;
    let received = u64::from_le_bytes(footer);
    let computed_full: [u8; 32] = mac.finalize();
    let computed = u64::from_le_bytes(computed_full[0..8].try_into().unwrap());
    if computed != received {
        return Err(TensorError::new(&format!(
            "cluster_controller: RoundFrame HMAC verification failed (computed \
             0x{computed:016x}, wire carried 0x{received:016x}); session salt \
             disagreement, tampered frame, or payload corruption"
        )));
    }
    Ok(Some(RoundFrame { tensors }))
}

/// Write a RoundFrame to a stream, appending the 8-byte HMAC-SHA256
/// footer keyed by `salt`. `pub(crate)` companion to
/// [`read_round_frame`]; shared by the rank-side client.
pub(crate) fn write_round_frame<W: Write>(
    stream: &mut W,
    frame: &RoundFrame,
    salt: &SessionSalt,
) -> Result<()> {
    let mut mac = HMAC::new(salt.as_slice());

    let mut hdr = [0u8; 8];
    hdr[0..4].copy_from_slice(&ROUND_FRAME_MAGIC.to_le_bytes());
    hdr[4..8].copy_from_slice(&(frame.tensors.len() as u32).to_le_bytes());
    stream.write_all(&hdr).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame header write failed: {e}"))
    })?;
    mac.update(hdr);
    for (ti, t) in frame.tensors.iter().enumerate() {
        let meta = [t.dtype, t.shape.len() as u8];
        stream.write_all(&meta).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] meta write failed: {e}"
            ))
        })?;
        mac.update(meta);
        for d in &t.shape {
            let d_bytes = d.to_le_bytes();
            stream.write_all(&d_bytes).map_err(|e| {
                TensorError::new(&format!(
                    "cluster_controller: tensor[{ti}] shape write failed: {e}"
                ))
            })?;
            mac.update(d_bytes);
        }
        let nb_bytes = (t.bytes.len() as u64).to_le_bytes();
        stream.write_all(&nb_bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] nbytes write failed: {e}"
            ))
        })?;
        mac.update(nb_bytes);
        stream.write_all(&t.bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] data write failed: {e}"
            ))
        })?;
        mac.update(&t.bytes);
    }

    // 8-byte HMAC-SHA256-64 footer, keyed by salt.
    let computed_full: [u8; 32] = mac.finalize();
    let mut footer = [0u8; 8];
    footer.copy_from_slice(&computed_full[0..8]);
    stream.write_all(&footer).map_err(|e| {
        TensorError::new(&format!("cluster_controller: frame HMAC footer write failed: {e}"))
    })?;
    stream
        .flush()
        .map_err(|e| TensorError::new(&format!("cluster_controller: frame flush failed: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Reduction (CPU sum + divide by world_size)
// ---------------------------------------------------------------------------

/// Average per-rank frames into a single frame, skipping dead ranks.
///
/// `frames[i] = None` means rank `i` is dead and didn't contribute;
/// `Some(frame)` means rank `i` is alive. The divisor is the
/// alive-count (number of `Some`), not `frames.len()` — matching the
/// avg-trick semantics over the surviving cohort.
///
/// Validates that every alive rank's frames have identical schema
/// (same number of tensors, same dtype per tensor, same shape per
/// tensor). Returns the element-wise mean.
///
/// v1 supports only [`DTYPE_F32`]; loud error on other dtypes (so a
/// future user wiring f16 here gets a clear pointer at where to add
/// support, instead of silent garbage from byte-level summation).
fn reduce_average_alive(frames: &[Option<RoundFrame>]) -> Result<RoundFrame> {
    let alive: Vec<&RoundFrame> = frames.iter().filter_map(|f| f.as_ref()).collect();
    if alive.is_empty() {
        return Err(TensorError::new(
            "cluster_controller: reduce_average_alive called with no alive ranks \
             (all participants dead — caller should not have reached this point)",
        ));
    }
    let n = alive.len();
    let ref_frame = alive[0];
    // Adapter so the existing schema-validation + reduce code below
    // can keep using its original variable names.
    let frames: &[&RoundFrame] = &alive;
    // Schema validation.
    for (i, f) in frames.iter().enumerate().skip(1) {
        if f.tensors.len() != ref_frame.tensors.len() {
            return Err(TensorError::new(&format!(
                "cluster_controller: rank {i} sent {} tensors; rank 0 sent {}",
                f.tensors.len(),
                ref_frame.tensors.len()
            )));
        }
        for (ti, (a, b)) in ref_frame.tensors.iter().zip(f.tensors.iter()).enumerate() {
            if a.dtype != b.dtype {
                return Err(TensorError::new(&format!(
                    "cluster_controller: rank {i} tensor[{ti}] dtype {} != rank 0 dtype {}",
                    b.dtype, a.dtype
                )));
            }
            if a.shape != b.shape {
                return Err(TensorError::new(&format!(
                    "cluster_controller: rank {i} tensor[{ti}] shape {:?} != rank 0 shape {:?}",
                    b.shape, a.shape
                )));
            }
            if a.bytes.len() != b.bytes.len() {
                return Err(TensorError::new(&format!(
                    "cluster_controller: rank {i} tensor[{ti}] nbytes {} != rank 0 nbytes {}",
                    b.bytes.len(),
                    a.bytes.len()
                )));
            }
        }
    }

    // Reduce per tensor.
    let mut out_tensors = Vec::with_capacity(ref_frame.tensors.len());
    for ti in 0..ref_frame.tensors.len() {
        let dtype = ref_frame.tensors[ti].dtype;
        if dtype != DTYPE_F32 {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] dtype {dtype} not supported in v1 \
                 (only DTYPE_F32 = 0 supported). Add other dtypes in controller.rs::reduce_average."
            )));
        }
        let shape = ref_frame.tensors[ti].shape.clone();
        let numel = ref_frame.tensors[ti].numel();
        if numel * std::mem::size_of::<f32>() != ref_frame.tensors[ti].bytes.len() {
            return Err(TensorError::new(&format!(
                "cluster_controller: tensor[{ti}] shape {shape:?} numel*sizeof(f32) {} != nbytes {}",
                numel * std::mem::size_of::<f32>(),
                ref_frame.tensors[ti].bytes.len()
            )));
        }
        let mut accum: Vec<f32> = vec![0.0; numel];
        for f in frames.iter() {
            let view = bytes_as_f32(&f.tensors[ti].bytes)?;
            for (a, x) in accum.iter_mut().zip(view.iter()) {
                *a += *x;
            }
        }
        let inv = 1.0_f32 / (n as f32);
        for a in &mut accum {
            *a *= inv;
        }
        out_tensors.push(TensorPayload {
            dtype: DTYPE_F32,
            shape,
            bytes: f32_to_bytes(&accum),
        });
    }
    Ok(RoundFrame {
        tensors: out_tensors,
    })
}

fn bytes_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(TensorError::new(&format!(
            "cluster_controller: f32 byte count {} not divisible by 4",
            bytes.len()
        )));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = [0u8; 4];
        b.copy_from_slice(&bytes[i * 4..(i + 1) * 4]);
        out.push(f32::from_le_bytes(b));
    }
    Ok(out)
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "controller_tests.rs"]
mod tests;
