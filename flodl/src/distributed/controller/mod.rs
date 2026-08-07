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
//! CPU, divides ONCE by the summed realized-work mass of exactly the
//! frames it accepted into the round, and writes the consensus
//! `RoundFrame` back to each rank. No NCCL. Genuinely async from NCCL's perspective:
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
//! u32 protocol_ver  = 2
//! u32 rank_id       (this rank's global rank, 0..world_size)
//! u32 world_size    (rank's view; controller validates against its own)
//! ```
//!
//! Handshake ack (controller → rank):
//! ```text
//! u32 magic        = 0xF10D_17C1
//! u32 protocol_ver = 2
//! ```
//!
//! RoundFrame (rank → controller, then controller → rank, identical
//! shape both directions):
//! ```text
//! u32 magic       = 0xF10D_17F1
//! u32 num_tensors
//! u8  round_kind  (0 = Model, 1 = Control)
//! f64 weight      (realized-work mass; see RoundFrame::weight)
//! for each tensor:
//!   u8  dtype   (0 = f32, 1 = bf16)
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
///
/// v3 added the wire zero-elision: a payload may declare `nbytes = 0`
/// while its shape has elements, meaning all zeros of the declared
/// dtype/shape (see [`TensorPayload::bytes`]). A v2 reader folds such
/// a frame into a loud schema error rather than misreading it, but the
/// version bump moves the mixed-version failure to the handshake —
/// first contact, named cause — instead of the first idle window.
pub(crate) const PROTOCOL_VERSION: u32 = 3;

/// dtype tag for f32 in the wire protocol — the default and the only
/// dtype [`RoundKind::Control`] traffic ever rides (count gathers carry
/// integers bf16 cannot represent exactly above 256, and the formation
/// broadcast must hand every rank byte-exact initial state).
pub const DTYPE_F32: u8 = 0;

/// dtype tag for bfloat16 in the wire protocol. Opt-in for
/// [`RoundKind::Model`] frames via
/// [`ElCheConfig::bf16_wire`](crate::distributed::ElCheConfig::bf16_wire):
/// halves every param-plane frame (pinned snapshots, relay fold traffic,
/// WAN payloads). All arithmetic on the plane — the relay fold, the
/// controller's sum, the divide-once normalization, the outer optimizer
/// — still ACCUMULATES IN F32; bf16 exists only at the wire/buffer
/// boundary (encode on write, decode on read).
pub const DTYPE_BF16: u8 = 1;

/// Ceiling on a [`RoundFrame`]'s claimed tensor count — unauthenticated
/// until the MAC verifies, so bounded before the read loop trusts it.
/// Generous: one entry per model param/buffer; the largest realistic
/// models sit in the thousands.
pub(crate) const MAX_ROUND_FRAME_TENSORS: usize = 65_536;



mod dead_ranks;
mod round_frame;
pub use dead_ranks::DeadRanks;
pub use round_frame::{RoundFrame, RoundKind, TensorPayload};
pub(crate) use round_frame::{read_round_frame, write_round_frame};
// Production summing now runs either incrementally at the relay fold
// ([`crate::distributed::relay`]'s HostFold, same monoid) or inside
// `reduce_realized_work`; the direct entry remains the reference the
// fold tests compare against.
#[cfg(test)]
pub(crate) use round_frame::sum_frames;
pub(crate) use round_frame::{f32_slice_to_payload_bytes, payload_to_f32};
pub(crate) use round_frame::{
    accumulate_payload_into, payload_element_size, read_round_frame_streamed,
    round_frame_wire_len, scale_payload_bytes, write_round_frame_streamed, PayloadPart,
};
use round_frame::reduce_realized_work;
// Byte-codec helpers used only by the controller round-frame tests.
#[cfg(test)]
use round_frame::{bytes_as_f32, f32_to_bytes};

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
    /// [`Self::port`]). Production goes through
    /// [`Self::start_from_source`] with a leg of the single-port mux.
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
        Self::start_with_dead_ranks(bind_addr, world_size, salt, dead_ranks, None, None)
    }

    /// Like [`Self::start`] but shares the dead-rank ledger with the
    /// coordinator. When the coord declares a rank dead, the
    /// controller's reduce thread stops waiting for its frames; the
    /// realized-work reduce only ever sums (and normalizes by) the
    /// frames it actually accepted, so a death needs no divisor
    /// adjustment. Use the
    /// [`DeadRanks`] returned by [`DeadRanks::new`] (or pass the same
    /// Arc clone to both this constructor and the
    /// [`crate::distributed::cluster_coordinator::ClusterCoordinator`]
    /// via its config).
    pub fn start_with_dead_ranks(
        bind_addr: SocketAddr,
        world_size: usize,
        salt: SessionSalt,
        dead_ranks: Arc<DeadRanks>,
        forge: Option<Arc<crate::distributed::CheckpointForge>>,
        outer_optimizer: Option<Box<dyn crate::distributed::OuterOptimizer>>,
    ) -> Result<Self> {
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
        let source = crate::distributed::port_mux::StreamSource::from_listener(
            listener,
            "cluster_controller",
        )?;
        Self::start_from_source(
            source, bound_port, world_size, salt, dead_ranks, forge,
            outer_optimizer,
        )
    }

    /// Like [`Self::start_with_dead_ranks`] but accepting connections
    /// from a pre-built [`StreamSource`] — the production entry, handed
    /// the data leg of the launcher's single-port mux. `bound_port` is
    /// carried for diagnostics ([`Self::port`]).
    ///
    /// [`StreamSource`]: crate::distributed::port_mux::StreamSource
    pub(crate) fn start_from_source(
        source: crate::distributed::port_mux::StreamSource,
        bound_port: u16,
        world_size: usize,
        salt: SessionSalt,
        dead_ranks: Arc<DeadRanks>,
        forge: Option<Arc<crate::distributed::CheckpointForge>>,
        outer_optimizer: Option<Box<dyn crate::distributed::OuterOptimizer>>,
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
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_cloned = Arc::clone(&shutdown);
        let handle = thread::Builder::new()
            .name(format!("flodl-cluster-controller:{bound_port}"))
            .spawn(move || {
                run_reduce_thread(
                    source, world_size, salt, shutdown_cloned, dead_ranks, forge,
                    outer_optimizer,
                )
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
    source: crate::distributed::port_mux::StreamSource,
    world_size: usize,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
    dead_ranks: Arc<DeadRanks>,
    forge: Option<Arc<crate::distributed::CheckpointForge>>,
    outer_optimizer: Option<Box<dyn crate::distributed::OuterOptimizer>>,
) -> Result<()> {
    // Outer optimizer applied to the consensus before scatter (and before
    // the forge tap, so the checkpoint captures the stepped global). `None`
    // leaves the reduce stream byte-for-byte as the plain weighted average.
    let mut outer_stepper = outer_optimizer
        .map(crate::distributed::outer_optimizer::OuterStepper::new);

    let slots = Arc::new(ReduceSlots::new());
    // Sole-writer half per relay connection (the reduce loop writes
    // replies); the matching read half is owned by a per-connection
    // reader thread. `rank_conn[rank]` indexes the connection carrying
    // that rank; `conn_ranks[conn]` is the inverse map.
    let mut conn_writes: Vec<TcpStream> = Vec::new();
    let mut rank_conn: Vec<Option<usize>> = (0..world_size).map(|_| None).collect();
    let mut all_conn_ranks: Vec<Vec<usize>> = Vec::new();
    let mut reader_threads: Vec<JoinHandle<()>> = Vec::new();
    let mut covered = 0usize;

    // Phase 1: accept per-host relay connections. Each announces the ranks
    // it carries via a `RelayHello`; accept until every global rank is
    // covered exactly once.
    while covered < world_size {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match source.try_accept("cluster_controller") {
            Ok(None) => {
                thread::sleep(Duration::from_millis(20));
            }
            Ok(Some(mut stream)) => {
                let _ = stream.set_nodelay(true);
                // Write-stall ceiling (fd-level): a wedged relay errors
                // the reduce thread's scatter instead of parking it; the
                // elastic scatter below then declares that connection's
                // ranks dead and continues with the survivors.
                stream
                    .set_write_timeout(Some(crate::distributed::wire::write_stall_timeout()))
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster_controller: set_write_timeout: {e}"
                        ))
                    })?;
                // 10s handshake timeout (mirrors the coordinator's
                // formation guard): the mux peek only guarantees the 4
                // magic bytes arrived, so a relay that wedges between
                // magic and Hello must error this loop — a blocking
                // read here outlives the shutdown flag and hangs the
                // Drop join. The post-handshake read half re-arms its
                // own timeout below.
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster_controller: set_read_timeout: {e}"
                        ))
                    })?;
                // Channel-select magic, then the relay handshake
                // (relays send both immediately on connect).
                crate::distributed::wire::expect_channel_magic(
                    &mut stream,
                    crate::distributed::wire::CHANNEL_MAGIC_DATA,
                    "cluster_controller",
                )?;
                let ranks = match MuxRecord::read_from(&mut stream, &salt)? {
                    // Model signatures ride the control channel's Hello
                    // only; the data channel's bare handshake carries
                    // none and the coordinator already compared them.
                    Some(MuxRecord::Control(RelayControlMsg::Hello {
                        host,
                        ranks,
                        model_sigs: _,
                    })) => {
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
                let registered = slots.register_conn(conn_ranks.clone());
                debug_assert_eq!(registered, conn_idx);
                all_conn_ranks.push(conn_ranks.clone());

                let slots_c = Arc::clone(&slots);
                let dead_c = Arc::clone(&dead_ranks);
                let shutdown_c = Arc::clone(&shutdown);
                let t = thread::Builder::new()
                    .name(format!("flodl-controller-relay{conn_idx}"))
                    .spawn(move || {
                        reduce_reader(
                            read_half, conn_idx, conn_ranks, slots_c, dead_c, shutdown_c,
                            salt,
                        )
                    })
                    .map_err(|e| {
                        TensorError::new(&format!("cluster_controller: spawn reader: {e}"))
                    })?;
                reader_threads.push(t);
            }
            Err(e) => return Err(e),
        }
    }
    drop(source);

    // Phase 2: reduce loop. Wait until every expected host connection has
    // deposited its folded frame, reduce (dividing once by the accepted
    // mass), and scatter ONE consensus Broadcast down each surviving
    // connection — the relay fans it out to its local ranks. The reduce
    // loop is the SOLE writer of every relay connection. Terminates when
    // an alive rank's data connection closes (clean training-end exit),
    // every rank is dead, or shutdown is signalled.
    //
    // Dead-diff forwarding: deaths declared cluster-side (coordinator
    // heartbeat staleness, scatter failure on another host) must reach
    // the owning relay's fold barrier, or a wedged-but-connected local
    // rank parks that host's fold forever. Forwarded from the wait's
    // poll hook — the only moment the reduce loop is otherwise idle and
    // still the sole writer of the connections.
    let mut forwarded_dead: Vec<bool> = vec![false; world_size];
    let outcome = loop {
        if shutdown.load(Ordering::SeqCst) {
            break Ok(());
        }
        let round = {
            let conn_writes = &mut conn_writes;
            let forwarded_dead = &mut forwarded_dead;
            slots.wait_for_round(&dead_ranks, &shutdown, REDUCE_POLL, || {
                forward_dead_diffs(
                    &dead_ranks,
                    forwarded_dead,
                    &rank_conn,
                    conn_writes,
                    &salt,
                );
            })
        };
        match round {
            RoundOutcome::Frames(frames) => {
                if let Err(e) = average_and_scatter(
                    &frames,
                    &mut conn_writes,
                    &all_conn_ranks,
                    &dead_ranks,
                    &salt,
                    forge.as_deref(),
                    outer_stepper.as_mut(),
                ) {
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
/// threads and drained by the reduce loop. One slot per CONNECTION
/// (host relay): each relay folds its local ranks' contributions into a
/// single [`MuxRecord::HostFrame`] per round, so the controller
/// accounts per host, not per rank.
///
/// [`MuxRecord::HostFrame`]: crate::distributed::relay::mux::MuxRecord::HostFrame
struct ReduceSlots {
    inner: Mutex<SlotsInner>,
    cv: Condvar,
}

struct SlotsInner {
    /// This round's folded frame per connection (`None` until the
    /// connection's reader deposits it; taken by the reduce loop once
    /// every expected connection is present).
    frames: Vec<Option<RoundFrame>>,
    /// Global ranks carried by each connection (from its `RelayHello`).
    /// A connection is EXPECTED in a round while any of its ranks is
    /// alive.
    conn_ranks: Vec<Vec<usize>>,
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
    /// Every expected connection's folded frame for this round
    /// (connections whose ranks are all dead are `None`).
    Frames(Vec<Option<RoundFrame>>),
    /// Clean shutdown requested (alive-rank exit, all ranks dead, or the
    /// external shutdown flag).
    Shutdown,
    /// A reader surfaced a hard wire error.
    Error(TensorError),
}

impl ReduceSlots {
    fn new() -> Self {
        ReduceSlots {
            inner: Mutex::new(SlotsInner {
                frames: Vec::new(),
                conn_ranks: Vec::new(),
                shutdown: false,
                error: None,
            }),
            cv: Condvar::new(),
        }
    }

    /// Register a relay connection carrying `ranks`; returns its slot
    /// index. Called from the accept phase as relays hello in.
    fn register_conn(&self, ranks: Vec<usize>) -> usize {
        let mut inner = self.inner.lock().unwrap();
        inner.frames.push(None);
        inner.conn_ranks.push(ranks);
        inner.frames.len() - 1
    }

    /// A reader deposits its connection's folded frame for the current
    /// round.
    fn deposit(&self, conn: usize, frame: RoundFrame) {
        let mut inner = self.inner.lock().unwrap();
        if conn < inner.frames.len() {
            inner.frames[conn] = Some(frame);
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

    /// Block until every expected connection (one with ≥1 alive rank)
    /// has deposited its folded frame, then take them (leaving
    /// fully-dead connections `None`). Re-evaluates dead ranks every
    /// `poll` so a coord-declared death (which carries no notify) is
    /// observed. `on_poll` runs on every wake (deposit, shutdown, or
    /// poll tick) OUTSIDE the slots lock — the reduce loop uses it to
    /// forward newly-observed deaths to the owning relays, which must
    /// happen while this wait is parked (a relay whose local rank was
    /// declared dead cluster-side would otherwise hold its fold — and
    /// this wait — forever).
    fn wait_for_round(
        &self,
        dead: &DeadRanks,
        external_shutdown: &AtomicBool,
        poll: Duration,
        mut on_poll: impl FnMut(),
    ) -> RoundOutcome {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if external_shutdown.load(Ordering::SeqCst) || inner.shutdown {
                return RoundOutcome::Shutdown;
            }
            if let Some(e) = inner.error.take() {
                return RoundOutcome::Error(e);
            }
            let n_conns = inner.frames.len();
            let expected: Vec<usize> = (0..n_conns)
                .filter(|c| inner.conn_ranks[*c].iter().any(|r| !dead.is_dead(*r)))
                .collect();
            if expected.is_empty() {
                // Every rank dead/done → nothing left to reduce.
                return RoundOutcome::Shutdown;
            }
            if expected.iter().all(|c| inner.frames[*c].is_some()) {
                let mut out: Vec<Option<RoundFrame>> = Vec::with_capacity(n_conns);
                for c in 0..n_conns {
                    if expected.contains(&c) {
                        out.push(inner.frames[c].take());
                    } else {
                        // Stale frame from a connection whose ranks all
                        // died — clear it so it can't leak into a later
                        // round.
                        inner.frames[c] = None;
                        out.push(None);
                    }
                }
                return RoundOutcome::Frames(out);
            }
            drop(inner);
            on_poll();
            inner = self.inner.lock().unwrap();
            let (guard, _timeout) = self.cv.wait_timeout(inner, poll).unwrap();
            inner = guard;
        }
    }
}

/// Per-connection reader: deposit the relay's folded `HostFrame` into
/// this connection's reduce slot, surface `RankExit` / EOF as clean
/// shutdown for still-alive ranks, and parse the RoundFrame payload
/// from memory.
fn reduce_reader(
    mut read: TcpStream,
    conn_idx: usize,
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
            Ok(MuxRead::Record(MuxRecord::HostFrame { payload })) => {
                if ranks.iter().all(|r| dead_ranks.is_dead(*r)) {
                    continue; // late frame from a fully-dead host — drop
                }
                let mut slice = &payload[..];
                match read_round_frame(&mut slice, &salt) {
                    Ok(Some(frame)) => slots.deposit(conn_idx, frame),
                    Ok(None) => {
                        slots.set_error(TensorError::new(&format!(
                            "cluster_controller: truncated HostFrame payload on \
                             connection {conn_idx} (ranks {ranks:?})"
                        )));
                        return;
                    }
                    Err(e) => {
                        slots.set_error(e);
                        return;
                    }
                }
            }
            Ok(MuxRead::Record(MuxRecord::Data { rank, .. })) => {
                // Per-rank data records no longer exist on the data
                // channel — the relay folds. Receiving one means a
                // stale relay build is talking to this controller.
                slots.set_error(TensorError::new(&format!(
                    "cluster_controller: per-rank Data record (rank {rank}) on the \
                     data channel; the relay is expected to fold local frames into \
                     a HostFrame — mixed relay/controller builds?"
                )));
                return;
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
            Ok(MuxRead::Record(MuxRecord::Broadcast { .. })) => {
                // Down-leg record; a relay never sends one upstream.
                eprintln!(
                    "cluster_controller: unexpected Broadcast record from relay \
                     {conn_idx}; dropping"
                );
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

/// Forward newly-observed rank deaths to their owning relay connection
/// as [`RelayControlMsg::DeclareDead`] so the relay drops the rank from
/// its fold barrier. Best-effort per connection: a failed write means
/// the connection itself is dying (elastic scatter / EOF handling own
/// that); `forwarded` keeps each death forwarded exactly once.
fn forward_dead_diffs(
    dead_ranks: &DeadRanks,
    forwarded: &mut [bool],
    rank_conn: &[Option<usize>],
    conn_writes: &mut [TcpStream],
    salt: &SessionSalt,
) {
    for (rank, fwd) in forwarded.iter_mut().enumerate() {
        if *fwd || !dead_ranks.is_dead(rank) {
            continue;
        }
        *fwd = true;
        let Some(ci) = rank_conn.get(rank).copied().flatten() else {
            continue;
        };
        if let Err(e) = MuxRecord::control(RelayControlMsg::DeclareDead {
            rank: rank as u32,
        })
        .write_to(&mut conn_writes[ci], salt)
        {
            crate::verbose!(
                "  cluster_controller: DeclareDead({rank}) forward to relay \
                 {ci} failed ({e}); connection presumed dying",
            );
        }
    }
}

/// Reduce this round's folded host frames and scatter ONE consensus
/// `Broadcast` down each surviving connection — the relay fans it out
/// to its local alive ranks. The reduce loop is the sole writer of
/// every connection.
fn average_and_scatter(
    frames: &[Option<RoundFrame>],
    conn_writes: &mut [TcpStream],
    conn_ranks: &[Vec<usize>],
    dead_ranks: &DeadRanks,
    salt: &SessionSalt,
    forge: Option<&crate::distributed::CheckpointForge>,
    outer_stepper: Option<&mut crate::distributed::outer_optimizer::OuterStepper>,
) -> Result<()> {
    let averaged = reduce_realized_work(frames)?;
    // OUTER STEP: transform the averaged consensus into the new global
    // BEFORE scatter (ranks adopt the stepped global) and BEFORE the forge
    // tap below (the checkpoint captures the stepped global). Applies to the
    // parameters frame only; Control / buffers frames pass through. `None`
    // (no outer optimizer configured) leaves `averaged` exactly as the
    // weighted average — the byte-for-byte pre-outer-optimizer path.
    // Outer-momentum capture rides alongside the step: only when a checkpoint
    // is armed (rare) and the variant carries state. Serialized synchronously
    // here (owned bytes) so the forge's detached writer never races the next
    // window's momentum update. `None` for OuterAvg / unarmed.
    let mut outer_momentum: Option<Vec<TensorPayload>> = None;
    let averaged = match outer_stepper {
        Some(stepper) => {
            let stepped = stepper.process_frame(averaged)?;
            if stepped.kind == RoundKind::Model
                && forge.is_some_and(|f| f.is_armed())
                && let Some(m) = stepper.checkpoint_state()
            {
                let refs: Vec<&crate::tensor::Tensor> = m.iter().collect();
                // Controller-local payloads (never hit the wire): stay
                // f32 so the `<stem>.outer.fdl` momentum is exact
                // whatever the model wire dtype.
                outer_momentum = Some(
                    crate::distributed::cpu_reduce::tensors_to_round_frame(&refs, DTYPE_F32)?
                        .tensors,
                );
            }
            stepped
        }
        None => averaged,
    };
    // The consensus frame is identical for every rank; serialize once, wrap
    // once, and forward the same record to every surviving connection —
    // `write_to` borrows, so cloning the model-sized buffer per connection
    // was pure alloc+memcpy churn (n_conns × frame bytes per window).
    let mut buf: Vec<u8> = Vec::new();
    write_round_frame(&mut buf, &averaged, salt)?;
    let record = MuxRecord::broadcast(buf);
    // ELASTIC SCATTER: a write failure (including a zero-progress stall
    // tripping the socket's write timeout) marks that CONNECTION broken
    // and declares its ranks dead, and the scatter continues to the
    // surviving connections — one wedged host degrades membership
    // instead of killing the run. `wait_for_round` recomputes the
    // expected set every poll, and the realized-work reduce is exact
    // over whatever cohort remains; if every connection breaks, the
    // next round-wait sees an empty expected set and shuts down.
    for (ci, ranks) in conn_ranks.iter().enumerate() {
        if ranks.iter().all(|r| dead_ranks.is_dead(*r)) {
            continue; // fully-dead host — no one to deliver to
        }
        if let Err(e) = record.write_to(&mut conn_writes[ci], salt) {
            eprintln!(
                "cluster_controller: consensus broadcast to relay {ci} \
                 (ranks {ranks:?}) failed ({e}); declaring its ranks dead and \
                 continuing with survivors"
            );
            for r in ranks {
                dead_ranks.declare_dead(*r);
            }
        }
    }
    // FORGE TAP: scatter ranks first (they resume training ASAP), then — if the
    // coordinator armed a checkpoint — hand this reduce's averaged consensus to
    // the forge. `averaged` is unused after the scatter above, so it is *moved*
    // (no clone) into the accumulator; once a cycle's model reduces (params then
    // buffers) have all arrived, the forge spawns a detached `.fdl` writer. Only
    // `Model` reduces feed the forge — the per-rank count-gather (`Control`)
    // scatters normally but is never part of the model. No-op when unarmed; the
    // reduce loop never blocks on the disk write.
    if let Some(f) = forge {
        // Zero-mass guard: a round that realized no work scatters a
        // meaningless (elided-zeros) frame that every rank answers by
        // keeping its local state — the checkpoint is a consensus
        // consumer too, so it must skip the same way rather than
        // accumulate zeros into the `.fdl`.
        if averaged.kind == RoundKind::Model
            && crate::distributed::realized_work::is_realized(averaged.weight)
        {
            // Stash this cycle's outer momentum (if captured above) so the
            // forge writes `<stem>.outer.fdl` alongside `<stem>.fdl` when the
            // accumulation completes.
            if let Some(m) = outer_momentum {
                f.stash_outer_momentum(m);
            }
            f.accumulate(averaged);
        }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../controller_tests.rs"]
mod tests;
