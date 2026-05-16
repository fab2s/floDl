//! Cluster controller: TCP byte router for the star-topology
//! cross-process gradient sum that powers [`AverageBackend::Cpu`].
//!
//! Slice 4b.D.1d.0 renamed the underlying type from `CpuAverager` to
//! [`ClusterController`] to reflect its broader role under the
//! process-model port -- it carries the data channel today and will
//! own ElChe scheduling + worker control in the following slices. The
//! data path stays the same star-topology byte router.
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
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use hmac_sha256::HMAC;

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
/// reduce thread to skip the rank's stream in the current and future
/// rounds, and by the coord-side `should_average` /
/// `poll_cpu_averaging` gates to exclude dead ranks from quorum
/// counting.
///
/// The struct also holds per-rank shutdown handles registered by the
/// controller after accept. `declare_dead` shuts down the dead rank's
/// stream, which wakes the controller's reduce thread out of any
/// pending read (so the cycle can release with survivors-only data).
#[derive(Debug)]
pub struct DeadRanks {
    flags: Vec<AtomicBool>,
    /// Per-rank shutdown handles for the controller-side stream.
    /// `Some` after accept registers them; `None` until accept or after
    /// `declare_dead` consumed (takes) the handle to invoke shutdown.
    stream_handles: Mutex<Vec<Option<TcpStream>>>,
}

impl DeadRanks {
    /// Create a fresh dead-rank ledger sized for `world_size`. All
    /// ranks start alive.
    pub fn new(world_size: usize) -> Arc<Self> {
        Arc::new(Self {
            flags: (0..world_size).map(|_| AtomicBool::new(false)).collect(),
            stream_handles: Mutex::new((0..world_size).map(|_| None).collect()),
        })
    }

    /// Declare `rank` permanently dead for the rest of this run.
    /// Idempotent. Sets the rank's flag, then shuts down its
    /// controller-side stream so the reduce thread unblocks from any
    /// pending read on that rank. No-op if `rank >= world_size`.
    pub fn declare_dead(&self, rank: usize) {
        if rank >= self.flags.len() {
            return;
        }
        let was_already_dead = self.flags[rank].swap(true, Ordering::SeqCst);
        if was_already_dead {
            return;
        }
        if let Ok(mut handles) = self.stream_handles.lock() {
            if let Some(slot) = handles.get_mut(rank) {
                if let Some(stream) = slot.take() {
                    let _ = stream.shutdown(Shutdown::Both);
                }
            }
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

    /// Controller registers a stream handle for `rank` after the
    /// accept-side handshake completes. The handle is a `try_clone` of
    /// the rank's stream — shutting it down affects the underlying OS
    /// file descriptor, waking any pending read on the original stream
    /// owned by the reduce thread.
    pub(crate) fn register_stream_handle(&self, rank: usize, handle: TcpStream) {
        if let Ok(mut handles) = self.stream_handles.lock() {
            if let Some(slot) = handles.get_mut(rank) {
                *slot = Some(handle);
            }
        }
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
    /// directly with a matching [`CpuReduceClient`].
    ///
    /// Use `127.0.0.1:0` for tests (kernel-assigned port; read back via
    /// [`Self::port`]). Use the cluster's `master_addr:master_port+2`
    /// in production.
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
    let mut streams: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();
    let mut connected = 0usize;

    // Phase 1: accept exactly `world_size` connections and validate the
    // handshake on each. Connections may arrive in any order; the
    // handshake's rank_id places each stream at the right slot.
    while connected < world_size {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                stream
                    .set_read_timeout(Some(Duration::from_millis(500)))
                    .map_err(|e| TensorError::new(&format!("cluster_controller: set_read_timeout: {e}")))?;
                let rank_id = read_handshake(&mut stream, world_size)?;
                if rank_id >= world_size {
                    return Err(TensorError::new(&format!(
                        "cluster_controller: handshake rank_id {rank_id} >= world_size {world_size}"
                    )));
                }
                if streams[rank_id].is_some() {
                    return Err(TensorError::new(&format!(
                        "cluster_controller: duplicate rank_id {rank_id} connected"
                    )));
                }
                write_handshake_ack(&mut stream)?;
                // Switch to blocking reads with no timeout for the
                // long-running reduce loop. Timeouts here would make
                // legitimately slow rounds look like failures.
                stream
                    .set_read_timeout(None)
                    .map_err(|e| TensorError::new(&format!("cluster_controller: set_read_timeout(None): {e}")))?;
                // Register a try_clone with the dead-rank ledger so
                // the coord can wake the reduce thread out of a
                // pending read on this rank when declaring it dead.
                // The cloned handle shares the OS file descriptor —
                // shutdown on either half affects both.
                if let Ok(handle) = stream.try_clone() {
                    dead_ranks.register_stream_handle(rank_id, handle);
                }
                streams[rank_id] = Some(stream);
                connected += 1;
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
    // All connected — drop nonblocking on the listener now that no more
    // accepts are expected.
    let _ = listener.set_nonblocking(false);
    let mut streams: Vec<TcpStream> = streams.into_iter().map(|s| s.unwrap()).collect();

    // Phase 2: reduce loop. Each round reads a RoundFrame from every
    // ALIVE rank, sums the per-tensor data, divides by the alive count,
    // writes the averaged frame back to alive ranks only. Terminates
    // when an alive rank disconnects cleanly (EOF on read while NOT
    // declared dead) or when shutdown is signalled.
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match read_round_from_all(&mut streams, &salt, &dead_ranks)? {
            Some(frames) => {
                let averaged = reduce_average_alive(&frames)?;
                write_round_to_all(&mut streams, &averaged, &salt, &dead_ranks)?;
            }
            None => return Ok(()), // an ALIVE rank EOFed → clean shutdown
        }
    }
}

// ---------------------------------------------------------------------------
// Handshake
// ---------------------------------------------------------------------------

fn read_handshake(stream: &mut TcpStream, expected_world_size: usize) -> Result<usize> {
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

fn write_handshake_ack(stream: &mut TcpStream) -> Result<()> {
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
pub(crate) fn read_round_frame(
    stream: &mut TcpStream,
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
        let mut bytes = vec![0u8; nbytes];
        stream.read_exact(&mut bytes).map_err(|e| {
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

/// Read a frame from every alive rank. Returns `Ok(Some(frames))` with
/// per-rank optional frames (None for dead ranks; Some for alive).
/// Returns `Ok(None)` only if an ALIVE rank EOFs (signals shutdown).
///
/// EOF on a rank whose `dead_ranks` flag is already set is treated as
/// expected (the coord shut down its stream to release this cycle) and
/// the rank is silently skipped. A read error on a dead rank similarly
/// folds into the skip path.
fn read_round_from_all(
    streams: &mut [TcpStream],
    salt: &SessionSalt,
    dead_ranks: &Arc<DeadRanks>,
) -> Result<Option<Vec<Option<RoundFrame>>>> {
    let mut frames: Vec<Option<RoundFrame>> = Vec::with_capacity(streams.len());
    for (rank, s) in streams.iter_mut().enumerate() {
        if dead_ranks.is_dead(rank) {
            frames.push(None);
            continue;
        }
        match read_round_frame(s, salt) {
            Ok(Some(f)) => frames.push(Some(f)),
            Ok(None) => {
                // Rank EOF'd. If it was just declared dead (race
                // between our `is_dead` check above and the coord's
                // shutdown), treat as expected skip.
                if dead_ranks.is_dead(rank) {
                    frames.push(None);
                    continue;
                }
                return Ok(None);
            }
            Err(e) => {
                // A read error on a freshly-declared-dead rank is the
                // expected wakeup from `dead_ranks.declare_dead`'s
                // stream shutdown. Treat as a skip; only propagate
                // errors when the rank wasn't declared dead.
                if dead_ranks.is_dead(rank) {
                    frames.push(None);
                    continue;
                }
                return Err(e);
            }
        }
    }
    Ok(Some(frames))
}

/// Write a RoundFrame to a stream, appending the 8-byte HMAC-SHA256
/// footer keyed by `salt`. `pub(crate)` companion to
/// [`read_round_frame`]; shared by the rank-side client.
pub(crate) fn write_round_frame(
    stream: &mut TcpStream,
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

fn write_round_to_all(
    streams: &mut [TcpStream],
    frame: &RoundFrame,
    salt: &SessionSalt,
    dead_ranks: &Arc<DeadRanks>,
) -> Result<()> {
    for (rank, s) in streams.iter_mut().enumerate() {
        if dead_ranks.is_dead(rank) {
            // Dead rank's stream may have been shut down by the coord;
            // even if it isn't, the rank isn't going to consume.
            continue;
        }
        write_round_frame(s, frame, salt)?;
    }
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
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::Ipv4Addr;
    use std::sync::mpsc;

    /// Deterministic non-zero test salt: exercises the HMAC path (zero
    /// salt is degenerate enough that an accidental "skip the HMAC"
    /// regression could silently still produce all-zero footers and
    /// "pass" — a non-zero salt catches that).
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

    /// Fake rank client: connects to the controller, does the handshake,
    /// and runs `n_rounds` of (send_frame → recv_averaged_frame).
    /// Returns the vector of received averaged frames.
    fn fake_rank(
        port: u16,
        rank_id: u32,
        world_size: u32,
        salt: SessionSalt,
        send_frames: Vec<RoundFrame>,
    ) -> Result<Vec<RoundFrame>> {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let mut stream = TcpStream::connect(addr).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: connect: {e}"))
        })?;

        // Handshake send
        let mut h = [0u8; 16];
        h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        h[8..12].copy_from_slice(&rank_id.to_le_bytes());
        h[12..16].copy_from_slice(&world_size.to_le_bytes());
        stream.write_all(&h).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: handshake write: {e}"))
        })?;
        let mut ack = [0u8; 8];
        stream.read_exact(&mut ack).map_err(|e| {
            TensorError::new(&format!("fake_rank {rank_id}: ack read: {e}"))
        })?;
        let ack_magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
        assert_eq!(ack_magic, HANDSHAKE_MAGIC_CONTROLLER_ACK);

        let mut received = Vec::with_capacity(send_frames.len());
        for f in send_frames {
            write_round_frame(&mut stream, &f, &salt)?;
            let r = read_round_frame(&mut stream, &salt)?
                .ok_or_else(|| TensorError::new("fake_rank: EOF before averaged frame"))?;
            received.push(r);
        }
        // Drop stream → clean EOF to controller, signals shutdown.
        Ok(received)
    }

    fn one_tensor_frame(data: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![TensorPayload {
                dtype: DTYPE_F32,
                shape: vec![data.len() as u32],
                bytes: f32_to_bytes(data),
            }],
        }
    }

    fn two_tensor_frame(a: &[f32], b: &[f32]) -> RoundFrame {
        RoundFrame {
            tensors: vec![
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![a.len() as u32],
                    bytes: f32_to_bytes(a),
                },
                TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![b.len() as u32],
                    bytes: f32_to_bytes(b),
                },
            ],
        }
    }

    #[test]
    fn two_rank_average_one_round() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let t0 = thread::spawn(move || {
            let r = fake_rank(port, 0, 2, TEST_SALT, vec![one_tensor_frame(&[1.0, 2.0, 3.0])]);
            tx0.send(r).unwrap();
        });
        let t1 = thread::spawn(move || {
            let r = fake_rank(port, 1, 2, TEST_SALT, vec![one_tensor_frame(&[3.0, 4.0, 5.0])]);
            tx1.send(r).unwrap();
        });

        let r0 = rx0.recv().unwrap().unwrap();
        let r1 = rx1.recv().unwrap().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        avg.shutdown().unwrap();

        // Average of (1,2,3) and (3,4,5) = (2,3,4)
        let expected = bytes_as_f32(&r0[0].tensors[0].bytes).unwrap();
        assert_eq!(expected, vec![2.0, 3.0, 4.0]);
        // Both ranks receive the same averaged frame.
        assert_eq!(r0, r1);
    }

    #[test]
    fn three_rank_average_multi_round_multi_tensor() {
        // Three ranks, two rounds each, each round carries two tensors.
        // Exercises:
        //   - multi-rank star summation (3 ranks)
        //   - multi-round reduce loop (2 rounds)
        //   - multi-tensor frames (2 tensors per frame)
        //   - clean shutdown on rank EOF
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            3,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        // Round-1 inputs: ranks 0/1/2 contribute (0,10), (10,20), (20,30) for tensor 0
        //                          and (1,1), (2,2), (3,3) for tensor 1.
        // Round-2 inputs: same shape, halved values (just to vary the data).
        let r0_frames = vec![
            two_tensor_frame(&[0.0, 10.0], &[1.0, 1.0]),
            two_tensor_frame(&[0.0, 5.0], &[0.5, 0.5]),
        ];
        let r1_frames = vec![
            two_tensor_frame(&[10.0, 20.0], &[2.0, 2.0]),
            two_tensor_frame(&[5.0, 10.0], &[1.0, 1.0]),
        ];
        let r2_frames = vec![
            two_tensor_frame(&[20.0, 30.0], &[3.0, 3.0]),
            two_tensor_frame(&[10.0, 15.0], &[1.5, 1.5]),
        ];

        let (tx0, rx0) = mpsc::channel();
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let t0 = thread::spawn(move || tx0.send(fake_rank(port, 0, 3, TEST_SALT, r0_frames)).unwrap());
        let t1 = thread::spawn(move || tx1.send(fake_rank(port, 1, 3, TEST_SALT, r1_frames)).unwrap());
        let t2 = thread::spawn(move || tx2.send(fake_rank(port, 2, 3, TEST_SALT, r2_frames)).unwrap());

        let r0 = rx0.recv().unwrap().unwrap();
        let r1 = rx1.recv().unwrap().unwrap();
        let r2 = rx2.recv().unwrap().unwrap();
        t0.join().unwrap();
        t1.join().unwrap();
        t2.join().unwrap();
        avg.shutdown().unwrap();

        // Each rank received exactly 2 averaged frames.
        assert_eq!(r0.len(), 2, "rank 0 should receive 2 averaged frames");
        assert_eq!(r1.len(), 2);
        assert_eq!(r2.len(), 2);

        // Round 1 averages: tensor 0 = (10, 20), tensor 1 = (2, 2)
        let r1_t0 = bytes_as_f32(&r0[0].tensors[0].bytes).unwrap();
        let r1_t1 = bytes_as_f32(&r0[0].tensors[1].bytes).unwrap();
        assert_eq!(r1_t0, vec![10.0, 20.0]);
        assert_eq!(r1_t1, vec![2.0, 2.0]);

        // Round 2 averages: tensor 0 = (5, 10), tensor 1 = (1, 1)
        let r2_t0 = bytes_as_f32(&r0[1].tensors[0].bytes).unwrap();
        let r2_t1 = bytes_as_f32(&r0[1].tensors[1].bytes).unwrap();
        assert_eq!(r2_t0, vec![5.0, 10.0]);
        assert_eq!(r2_t1, vec![1.0, 1.0]);

        // All three ranks see bit-identical averaged frames.
        assert_eq!(r0, r1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn rejects_wrong_handshake_magic() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        let mut s = TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        let mut bad = [0u8; 16];
        bad[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        s.write_all(&bad).unwrap();
        // The controller should reject and drop us; read_exact on ack
        // would return EOF or error. We accept either outcome.
        let mut ack = [0u8; 8];
        let _ = s.read_exact(&mut ack);
        drop(s);
        // The reduce thread terminates with an error; shutdown still
        // joins cleanly.
        let _ = avg.shutdown(); // err is OK here
    }

    #[test]
    fn rejects_world_size_mismatch() {
        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            2,
            TEST_SALT,
        )
        .unwrap();
        let port = avg.port();

        // Rank claims world_size = 3 but controller is configured for 2.
        let mut s = TcpStream::connect(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)).unwrap();
        let mut h = [0u8; 16];
        h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
        h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        h[8..12].copy_from_slice(&0u32.to_le_bytes());
        h[12..16].copy_from_slice(&3u32.to_le_bytes());
        s.write_all(&h).unwrap();
        // Server drops us.
        let mut ack = [0u8; 8];
        let _ = s.read_exact(&mut ack);
        drop(s);
        let _ = avg.shutdown();
    }

    #[test]
    fn rejects_non_f32_dtype_in_reduce() {
        // Pure unit test of reduce_average_alive without TCP wiring.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7, // bogus dtype
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: 7,
                    shape: vec![2],
                    bytes: vec![0; 8],
                }],
            }),
        ];
        let err = reduce_average_alive(&frames).unwrap_err();
        assert!(
            err.to_string().contains("dtype 7"),
            "expected dtype-7-not-supported, got: {err}"
        );
    }

    #[test]
    fn rejects_shape_mismatch_across_ranks() {
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[1.0, 2.0]),
                }],
            }),
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![3],
                    bytes: f32_to_bytes(&[1.0, 2.0, 3.0]),
                }],
            }),
        ];
        let err = reduce_average_alive(&frames).unwrap_err();
        assert!(err.to_string().contains("shape"), "got: {err}");
    }

    #[test]
    fn reduce_average_alive_skips_none_entries_and_divides_by_alive_count() {
        // 3-rank world, rank 1 dead (None). Mean over alive = (rank0 + rank2) / 2.
        let frames = vec![
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[2.0, 4.0]),
                }],
            }),
            None, // rank 1 dead
            Some(RoundFrame {
                tensors: vec![TensorPayload {
                    dtype: DTYPE_F32,
                    shape: vec![2],
                    bytes: f32_to_bytes(&[6.0, 8.0]),
                }],
            }),
        ];
        let out = reduce_average_alive(&frames).unwrap();
        let avg = bytes_as_f32(&out.tensors[0].bytes).unwrap();
        // (2 + 6) / 2 = 4.0; (4 + 8) / 2 = 6.0
        assert!((avg[0] - 4.0).abs() < 1e-6, "got {avg:?}");
        assert!((avg[1] - 6.0).abs() < 1e-6, "got {avg:?}");
    }

    #[test]
    fn reduce_average_alive_rejects_all_dead() {
        let frames: Vec<Option<RoundFrame>> = vec![None, None];
        let err = reduce_average_alive(&frames).unwrap_err();
        assert!(
            err.to_string().contains("no alive ranks"),
            "got: {err}"
        );
    }

    #[test]
    fn averager_zero_world_size_errors() {
        let err = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            0,
            TEST_SALT,
        )
        .unwrap_err();
        assert!(err.to_string().contains("world_size"), "got: {err}");
    }

    /// Cross-session safety: a rank using a salt the controller doesn't
    /// share must fail the first RoundFrame's HMAC check loudly. This
    /// is the load-bearing test that proves the salt is wired through
    /// both directions.
    #[test]
    fn rejects_round_frame_with_wrong_salt() {
        use crate::distributed::wire::SESSION_SALT_BYTES;
        let controller_salt = TEST_SALT;
        // The "rogue" salt: same length, different bytes.
        let rogue_salt: SessionSalt = [0xAAu8; SESSION_SALT_BYTES];
        assert_ne!(controller_salt, rogue_salt);

        let avg = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            1,
            controller_salt,
        )
        .unwrap();
        let port = avg.port();

        // Single-rank handshake (so the controller proceeds to the
        // reduce loop), then send one RoundFrame keyed with the wrong
        // salt. The controller's read_round_frame must error on the
        // HMAC footer.
        let send_res = thread::spawn(move || -> Result<()> {
            let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
            let mut stream = TcpStream::connect(addr).unwrap();

            // Handshake (the salt does not participate in handshake bytes;
            // any rank with matching world_size + magic + version connects).
            let mut h = [0u8; 16];
            h[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
            h[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
            h[8..12].copy_from_slice(&0u32.to_le_bytes());
            h[12..16].copy_from_slice(&1u32.to_le_bytes());
            stream.write_all(&h).unwrap();
            let mut ack = [0u8; 8];
            stream.read_exact(&mut ack).unwrap();

            // Now send a frame keyed by the rogue salt. The controller's
            // HMAC over the body will not match → the reduce thread
            // errors out and shuts down.
            let frame = one_tensor_frame(&[1.0, 2.0, 3.0]);
            write_round_frame(&mut stream, &frame, &rogue_salt)?;
            Ok(())
        });
        let _ = send_res.join().unwrap();

        // Drain the controller's status to confirm the loud error path
        // ran. `shutdown()` joins the thread and propagates its Result.
        let err = avg.shutdown().expect_err(
            "controller's reduce thread must propagate a HMAC verification error",
        );
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC verification failure, got: {err}"
        );
    }
}
