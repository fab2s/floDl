//! Per-host relay agent: byte-router on the control channel, fold
//! station on the data channel.
//!
//! One [`RelayChannel`] runs per {data, control} channel per host. It
//! terminates each local rank's handshake on a loopback listener exactly
//! as the controller would, then multiplexes the host's traffic over a
//! single upstream connection to the real controller, emitting
//! [`RelayControlMsg::RankExit`] when a local rank disconnects.
//!
//! # Topology
//!
//! ```text
//!   rank r0 --loopback--\                          /-- one upstream --\
//!   rank r1 --loopback---> RelayChannel (host H) --|   connection      |-- controller
//!   rank rk --loopback--/    mux up / demux down    \-- (per channel) -/
//! ```
//!
//! # Thread model (per channel)
//!
//! - **one rank-reader thread per local rank**: reads length-framed
//!   blobs off the rank's loopback socket and pushes work to the
//!   outbound queue. On EOF / error it pushes `RankExit{rank}` and
//!   exits.
//! - **one outbound-writer thread**: the *sole* writer of the upstream
//!   connection (single-writer discipline — see
//!   `feedback_no_locks_hot_path` / the historical two-writer HMAC race).
//!   Drains the outbound queue, writes each [`MuxRecord`] upstream.
//! - **one upstream-reader thread**: reads [`MuxRecord`]s from the
//!   controller and writes to the rank loopback sockets. It is the sole
//!   writer of each rank socket (single-writer per rank). On upstream
//!   EOF it flags shutdown (controller gone → tear down the host).
//!
//! # Control channel: forward. Data channel: fold.
//!
//! On the **control channel** the relay never parses a forwarded
//! payload: each blob crosses as a rank-tagged [`MuxRecord::Data`].
//!
//! On the **data channel** the relay is the first fold tier of the
//! realized-work reduce. Rank readers parse and HMAC-verify each local
//! [`RoundFrame`]; the depositor that completes the round (every local
//! *alive* rank present) sums them element-wise — masses too — via
//! [`controller::sum_frames`] (the fold NEVER divides; the controller
//! divides exactly once) and ships ONE re-signed
//! [`MuxRecord::HostFrame`] upstream. The controller's consensus comes
//! back as ONE [`MuxRecord::Broadcast`], fanned out to every local
//! alive rank. Local liveness is absorbed here: a rank EOF folds the
//! remainder, and controller-side deaths arrive as
//! [`RelayControlMsg::DeclareDead`] so a wedged-but-connected rank
//! cannot park the host's fold (no fold deadline — dropping a reduce
//! round silently is a correctness violation for Local SGD).
//!
//! [`MuxRecord`]: super::mux::MuxRecord
//! [`MuxRecord::Data`]: super::mux::MuxRecord::Data
//! [`MuxRecord::HostFrame`]: super::mux::MuxRecord::HostFrame
//! [`MuxRecord::Broadcast`]: super::mux::MuxRecord::Broadcast
//! [`RelayControlMsg::RankExit`]: super::mux::RelayControlMsg::RankExit
//! [`RelayControlMsg::DeclareDead`]: super::mux::RelayControlMsg::DeclareDead
//! [`RoundFrame`]: crate::distributed::controller::RoundFrame
//! [`controller::sum_frames`]: crate::distributed::controller::sum_frames

use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::distributed::controller::{self, RoundFrame};
use crate::distributed::wire::SessionSalt;
use crate::tensor::{Result, TensorError};

use super::mux::{
    try_read_len_framed, write_len_framed, LenFramedRead, MuxRead, MuxRecord, RelayControlMsg,
};

/// Poll cadence for the relay's reader/writer threads. Read timeouts are
/// set to this so each thread re-checks its shutdown flag on idle ticks
/// without busy-spinning.
const POLL_TIMEOUT: Duration = Duration::from_millis(100);


// ---------------------------------------------------------------------------
// Channel kind + handshake termination
// ---------------------------------------------------------------------------

/// Which control protocol the relay terminates toward its local ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    /// CPU-averaging data channel
    /// ([`crate::distributed::controller`]). Bare handshake.
    Data,
    /// Coordinator control channel
    /// ([`crate::distributed::cluster_coordinator`]). Salt-authenticated
    /// handshake.
    Control,
}

impl ChannelKind {
    /// Channel-select magic this channel's upstream dial opens with
    /// (routes the connection through the controller's single-port mux).
    fn channel_magic(self) -> u32 {
        match self {
            ChannelKind::Data => crate::distributed::wire::CHANNEL_MAGIC_DATA,
            ChannelKind::Control => crate::distributed::wire::CHANNEL_MAGIC_CONTROL,
        }
    }

    /// Terminate one rank's handshake on `stream` exactly as the
    /// controller would, returning the announced `rank_id`. The
    /// handshake is a fixed-size exchange (no length framing); the
    /// length-framed blob protocol begins only after this returns.
    fn terminate_handshake(
        self,
        stream: &mut TcpStream,
        world_size: usize,
        salt: &SessionSalt,
    ) -> Result<u32> {
        match self {
            ChannelKind::Data => {
                let rank =
                    crate::distributed::controller::read_handshake(stream, world_size)?;
                crate::distributed::controller::write_handshake_ack(stream)?;
                Ok(rank as u32)
            }
            ChannelKind::Control => {
                let rank = crate::distributed::cluster_coordinator::read_handshake_rank(
                    stream,
                    world_size as u32,
                    salt,
                )?;
                crate::distributed::cluster_coordinator::write_handshake_ack(stream, salt)?;
                Ok(rank)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RelayChannel handle
// ---------------------------------------------------------------------------

/// A running relay transport channel for one host. Owns the mux threads;
/// [`Self::shutdown`] (or drop) signals them and joins.
pub struct RelayChannel {
    shutdown: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl RelayChannel {
    /// Bind the loopback listener for this channel. Returns the listener
    /// (hand to [`Self::start`]) and the bound port (so ranks can be
    /// pointed at it; pass `127.0.0.1:0` in tests for a kernel-assigned
    /// port).
    pub fn bind(loopback_bind: SocketAddr) -> Result<(TcpListener, u16)> {
        let listener = TcpListener::bind(loopback_bind).map_err(|e| {
            TensorError::new(&format!("relay: loopback bind {loopback_bind} failed: {e}"))
        })?;
        let port = listener
            .local_addr()
            .map_err(|e| TensorError::new(&format!("relay: local_addr failed: {e}")))?
            .port();
        Ok((listener, port))
    }

    /// Accept the host's local ranks on `listener` (terminating each
    /// handshake), connect upstream to the controller, send the
    /// [`RelayControlMsg::Hello`] handshake, then start the mux threads.
    ///
    /// Blocks until all `ranks.len()` local ranks have connected and the
    /// upstream `HelloAck` is received. `world_size` is the cluster-wide
    /// rank count (validated in the per-rank handshake).
    pub fn start(
        listener: TcpListener,
        kind: ChannelKind,
        upstream_addr: SocketAddr,
        host: String,
        mut ranks: Vec<u32>,
        world_size: usize,
        salt: SessionSalt,
    ) -> Result<Self> {
        ranks.sort_unstable();

        // Phase 1: accept + terminate each local rank's handshake.
        let expected: HashSet<u32> = ranks.iter().copied().collect();
        let mut rank_streams: Vec<(u32, TcpStream)> = Vec::with_capacity(ranks.len());
        while rank_streams.len() < ranks.len() {
            let (mut stream, _peer) = listener.accept().map_err(|e| {
                TensorError::new(&format!("relay: loopback accept failed: {e}"))
            })?;
            let _ = stream.set_nodelay(true);
            // Write-stall ceiling (fd-level): a wedged rank must error
            // the upstream_reader's demux write instead of parking it
            // (its per-write failures are already tolerated).
            let _ = stream
                .set_write_timeout(Some(crate::distributed::wire::write_stall_timeout()));
            let rank = kind.terminate_handshake(&mut stream, world_size, &salt)?;
            if !expected.contains(&rank) {
                return Err(TensorError::new(&format!(
                    "relay: rank {rank} connected but is not in this host's rank set {ranks:?}"
                )));
            }
            if rank_streams.iter().any(|(r, _)| *r == rank) {
                return Err(TensorError::new(&format!(
                    "relay: duplicate rank {rank} connected on loopback"
                )));
            }
            rank_streams.push((rank, stream));
        }

        // Phase 2: connect upstream + relay handshake.
        let mut upstream = crate::distributed::wire::connect_with_retry(
            upstream_addr,
            "relay upstream",
        )?;
        let _ = upstream.set_nodelay(true);
        // Dialer-side cleartext guard: a public controller address means
        // this host's frames cross an uncontrolled network unencrypted.
        if let Ok(peer) = upstream.peer_addr() {
            crate::distributed::wire::warn_cleartext_public_peer("relay upstream", peer);
        }
        // Write-stall ceiling: a wedged controller errors outbound_writer,
        // which flags relay shutdown — reachable teardown instead of a
        // parked writer holding the host hostage.
        let _ = upstream
            .set_write_timeout(Some(crate::distributed::wire::write_stall_timeout()));
        // Channel-select magic first: the controller's single-port mux
        // routes on it, and the owning subsystem validates it before the
        // Hello.
        crate::distributed::wire::write_channel_magic(
            &mut upstream,
            kind.channel_magic(),
        )?;
        MuxRecord::control(RelayControlMsg::Hello {
            host,
            ranks: ranks.clone(),
        })
        .write_to(&mut upstream, &salt)?;
        match MuxRecord::read_from(&mut upstream, &salt)? {
            Some(MuxRecord::Control(RelayControlMsg::HelloAck)) => {}
            Some(other) => {
                return Err(TensorError::new(&format!(
                    "relay: expected HelloAck from controller, got {other:?}"
                )));
            }
            None => {
                return Err(TensorError::new(
                    "relay: controller closed connection before HelloAck",
                ));
            }
        }

        // Phase 3: spawn the mux threads.
        let shutdown = Arc::new(AtomicBool::new(false));
        let threads = spawn_mux(kind, rank_streams, upstream, salt, Arc::clone(&shutdown))?;
        Ok(RelayChannel { shutdown, threads })
    }

    /// Signal the mux threads to stop and join them. Idempotent.
    // Production relays exit via `join` (natural completion); the forced
    // teardown is exercised by the relay fault tests.
    #[allow(dead_code)]
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        Ok(())
    }

    /// Block until the mux threads exit on their own — i.e. until every
    /// local rank has disconnected (the relay self-shuts when its last
    /// rank exits) or the upstream connection closes. Unlike
    /// [`Self::shutdown`], this does NOT force an early stop; it waits for
    /// natural completion (training finished). Used by the relay process
    /// to stay up for the whole run, then exit cleanly.
    pub fn join(mut self) -> Result<()> {
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        Ok(())
    }
}

impl Drop for RelayChannel {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Upstream connect with retry
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Data-channel fold station
// ---------------------------------------------------------------------------

/// Host-local mirror of the controller's round collection: one slot per
/// local rank, plus the local dead-set. The mutation that completes the
/// round (a deposit, or a death that removes the last missing rank)
/// takes the frames and performs the fold — no dedicated fold thread,
/// no condvar, no deadline.
struct FoldCtx {
    inner: Mutex<FoldInner>,
    /// Global rank ids served by this host (the fold barrier's roster).
    local_ranks: Vec<u32>,
    salt: SessionSalt,
    /// Fold instrumentation: rounds folded, Σ bytes deposited by local
    /// ranks, Σ bytes shipped upstream. Summarized once (Drop) so the
    /// K→1 uplink reduction is observable on the rig.
    rounds: AtomicU64,
    bytes_in: AtomicU64,
    bytes_up: AtomicU64,
}

struct FoldInner {
    /// This round's parsed frame per local rank (`None` until
    /// deposited; taken by the completing fold).
    frames: HashMap<u32, RoundFrame>,
    /// Local ranks out of the fold barrier: loopback EOF (observed
    /// here) or controller-declared death ([`RelayControlMsg::DeclareDead`]).
    dead: HashSet<u32>,
}

impl FoldCtx {
    fn new(local_ranks: Vec<u32>, salt: SessionSalt) -> Self {
        FoldCtx {
            inner: Mutex::new(FoldInner {
                frames: HashMap::with_capacity(local_ranks.len()),
                dead: HashSet::new(),
            }),
            local_ranks,
            salt,
            rounds: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_up: AtomicU64::new(0),
        }
    }

    /// Deposit `rank`'s frame for the current round; fold + ship if this
    /// completes it. Returns `Err` on protocol violations that must tear
    /// the channel down (double deposit — the rank↔relay leg is a strict
    /// ping-pong, so a second frame before the round folded means a
    /// desynced stream).
    fn deposit(
        &self,
        rank: u32,
        frame: RoundFrame,
        frame_bytes: u64,
        tx: &mpsc::SyncSender<MuxRecord>,
    ) -> Result<()> {
        let taken = {
            let mut inner = self.inner.lock().expect("relay fold lock poisoned");
            if inner.dead.contains(&rank) {
                // Late frame from a rank already out of the barrier
                // (mirrors the controller's late-frame drop).
                return Ok(());
            }
            if inner.frames.insert(rank, frame).is_some() {
                return Err(TensorError::new(&format!(
                    "relay fold: rank {rank} deposited twice in one round \
                     (rank↔relay ping-pong violated; stream desynced)"
                )));
            }
            self.bytes_in.fetch_add(frame_bytes, Ordering::Relaxed);
            self.take_if_complete(&mut inner)
        };
        self.fold_and_ship(taken, tx)
    }

    /// Remove `rank` from the fold barrier (loopback EOF or
    /// controller-declared death); fold + ship if the round is now
    /// complete without it. Idempotent.
    fn mark_dead(&self, rank: u32, tx: &mpsc::SyncSender<MuxRecord>) -> Result<()> {
        let taken = {
            let mut inner = self.inner.lock().expect("relay fold lock poisoned");
            inner.dead.insert(rank);
            // A frame it already deposited this round stays in — the
            // rank realized that work; the controller-side accept
            // ledger is what decides acceptance across hosts.
            self.take_if_complete(&mut inner)
        };
        self.fold_and_ship(taken, tx)
    }

    /// Under the lock: if every local alive rank has deposited (and at
    /// least one frame is present), take the round's frames.
    fn take_if_complete(&self, inner: &mut FoldInner) -> Option<Vec<RoundFrame>> {
        let complete = self
            .local_ranks
            .iter()
            .all(|r| inner.dead.contains(r) || inner.frames.contains_key(r));
        if !complete || inner.frames.is_empty() {
            return None;
        }
        Some(inner.frames.drain().map(|(_, f)| f).collect())
    }

    /// Outside the lock: sum the taken frames, re-sign, ship upstream.
    fn fold_and_ship(
        &self,
        taken: Option<Vec<RoundFrame>>,
        tx: &mpsc::SyncSender<MuxRecord>,
    ) -> Result<()> {
        let Some(frames) = taken else {
            return Ok(());
        };
        let refs: Vec<&RoundFrame> = frames.iter().collect();
        let folded = controller::sum_frames(&refs)?;
        let mut buf = Vec::new();
        controller::write_round_frame(&mut buf, &folded, &self.salt)?;
        self.rounds.fetch_add(1, Ordering::Relaxed);
        self.bytes_up.fetch_add(buf.len() as u64, Ordering::Relaxed);
        tx.send(MuxRecord::host_frame(buf)).map_err(|_| {
            TensorError::new("relay fold: outbound writer gone before fold shipped")
        })
    }

    /// Fan the controller's consensus out to every local alive rank.
    /// Per-write failures are tolerated exactly like the byte-router
    /// path (a rank socket already gone means no one to deliver to).
    fn fan_out(&self, payload: &[u8], rank_writes: &mut HashMap<u32, TcpStream>) {
        let dead: Vec<u32> = {
            let inner = self.inner.lock().expect("relay fold lock poisoned");
            inner.dead.iter().copied().collect()
        };
        for rank in &self.local_ranks {
            if dead.contains(rank) {
                continue;
            }
            if let Some(w) = rank_writes.get_mut(rank) {
                let _ = write_len_framed(w, payload);
            }
        }
    }
}

impl Drop for FoldCtx {
    fn drop(&mut self) {
        let rounds = self.rounds.load(Ordering::Relaxed);
        if rounds == 0 {
            return;
        }
        let bytes_in = self.bytes_in.load(Ordering::Relaxed) as f64 / 1e6;
        let bytes_up = self.bytes_up.load(Ordering::Relaxed) as f64 / 1e6;
        let ratio = if bytes_up > 0.0 { bytes_in / bytes_up } else { 0.0 };
        eprintln!(
            "[relay-fold-prof] ranks={} rounds={rounds} | local={bytes_in:.2}MB \
             uplink={bytes_up:.2}MB fold-ratio={ratio:.2}x",
            self.local_ranks.len(),
        );
    }
}

// ---------------------------------------------------------------------------
// Mux core
// ---------------------------------------------------------------------------

/// Start the bidirectional mux over established (post-handshake) streams.
///
/// Spawns: one rank-reader per local rank, one outbound writer (sole
/// upstream writer), one upstream reader (sole writer of each rank
/// socket). Returns the join handles; the caller owns shutdown via the
/// shared `shutdown` flag.
///
/// Shutdown converges from three sources, all landing on the same flag:
/// explicit `shutdown` store, upstream EOF (controller gone), or the last
/// local rank exiting (host has no ranks left to serve).
fn spawn_mux(
    kind: ChannelKind,
    rank_streams: Vec<(u32, TcpStream)>,
    upstream: TcpStream,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
) -> Result<Vec<JoinHandle<()>>> {
    // Data channel: the fold station shared by every rank reader (they
    // deposit) and the upstream reader (deaths + consensus fan-out).
    // Control channel: pure byte-router, no fold context.
    let fold: Option<Arc<FoldCtx>> = match kind {
        ChannelKind::Data => Some(Arc::new(FoldCtx::new(
            rank_streams.iter().map(|(r, _)| *r).collect(),
            salt,
        ))),
        ChannelKind::Control => None,
    };
    // BOUNDED: per-rank reader threads pump full param-snapshot blobs
    // into this queue while a single blocking writer drains it to the
    // (possibly slow) upstream link. Unbounded, a slow upstream grew the
    // queue by O(ranks x model bytes) per reduce window; the bound makes
    // a full queue block the rank readers — re-creating the natural TCP
    // backpressure the mux removed. Sized in records, not bytes: each
    // rank has at most a handful of frames in flight per window.
    let (tx, rx) = mpsc::sync_channel::<MuxRecord>(64);
    let active = Arc::new(AtomicUsize::new(rank_streams.len()));

    // Upstream: one read clone (upstream-reader) + the original for the
    // sole outbound writer.
    let up_read = upstream
        .try_clone()
        .map_err(|e| TensorError::new(&format!("relay: upstream try_clone failed: {e}")))?;
    up_read
        .set_read_timeout(Some(POLL_TIMEOUT))
        .map_err(|e| TensorError::new(&format!("relay: upstream set_read_timeout: {e}")))?;
    let up_write = upstream;

    // Per rank: a read half (rank-reader) + a write half handed to the
    // upstream-reader (sole writer of each rank socket).
    let mut rank_writes: HashMap<u32, TcpStream> = HashMap::with_capacity(rank_streams.len());
    let mut rank_reads: Vec<(u32, TcpStream)> = Vec::with_capacity(rank_streams.len());
    for (rank, stream) in rank_streams {
        let write_half = stream
            .try_clone()
            .map_err(|e| TensorError::new(&format!("relay: rank {rank} try_clone failed: {e}")))?;
        stream
            .set_read_timeout(Some(POLL_TIMEOUT))
            .map_err(|e| TensorError::new(&format!("relay: rank {rank} set_read_timeout: {e}")))?;
        rank_writes.insert(rank, write_half);
        rank_reads.push((rank, stream));
    }

    let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(rank_reads.len() + 2);

    // Outbound writer (sole upstream writer).
    {
        let shutdown = Arc::clone(&shutdown);
        threads.push(
            thread::Builder::new()
                .name("flodl-relay-out".into())
                .spawn(move || outbound_writer(up_write, rx, salt, shutdown))
                .map_err(|e| TensorError::new(&format!("relay: spawn outbound writer: {e}")))?,
        );
    }

    // Upstream reader (sole writer of each rank socket). On the data
    // channel it also holds a queue sender: a `DeclareDead` it receives
    // can complete the round and ship the fold from this thread.
    {
        let shutdown = Arc::clone(&shutdown);
        let fold = fold.clone();
        let tx = tx.clone();
        threads.push(
            thread::Builder::new()
                .name("flodl-relay-up".into())
                .spawn(move || upstream_reader(up_read, rank_writes, salt, shutdown, fold, tx))
                .map_err(|e| TensorError::new(&format!("relay: spawn upstream reader: {e}")))?,
        );
    }

    // Rank readers.
    for (rank, stream) in rank_reads {
        let tx = tx.clone();
        let shutdown = Arc::clone(&shutdown);
        let active = Arc::clone(&active);
        let fold = fold.clone();
        threads.push(
            thread::Builder::new()
                .name(format!("flodl-relay-r{rank}"))
                .spawn(move || rank_reader(rank, stream, tx, shutdown, active, fold))
                .map_err(|e| TensorError::new(&format!("relay: spawn rank {rank} reader: {e}")))?,
        );
    }
    // Drop the template sender so the outbound writer's rx disconnects
    // once every rank-reader has dropped its clone (all ranks gone).
    drop(tx);

    Ok(threads)
}

/// Rank-reader: loopback rank socket → outbound queue.
///
/// Control channel (`fold` = None): wraps each length-framed blob as
/// `Data{rank, blob}` untouched. Data channel (`fold` = Some): parses +
/// HMAC-verifies the blob as a [`RoundFrame`] and deposits it into the
/// host fold — the deposit that completes the round ships the folded
/// `HostFrame` from this thread.
///
/// Emits `RankExit{rank}` on EOF/error then exits (on the data channel
/// the rank also leaves the fold barrier, which may complete the round
/// for the survivors). The last reader to exit flags shutdown.
fn rank_reader(
    rank: u32,
    mut stream: TcpStream,
    tx: mpsc::SyncSender<MuxRecord>,
    shutdown: Arc<AtomicBool>,
    active: Arc<AtomicUsize>,
    fold: Option<Arc<FoldCtx>>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match try_read_len_framed(&mut stream) {
            Ok(LenFramedRead::Blob(blob)) => match &fold {
                None => {
                    if tx.send(MuxRecord::data(rank, blob)).is_err() {
                        break; // outbound writer gone
                    }
                }
                Some(ctx) => {
                    // Parse + verify (the blob's end-to-end HMAC is
                    // checked by read_round_frame; the relay holds the
                    // session salt). Any failure here is a desynced or
                    // corrupt local stream — tear the channel down
                    // loudly rather than fold garbage.
                    let parsed = controller::read_round_frame(
                        &mut blob.as_slice(),
                        &ctx.salt,
                    );
                    let deposit = match parsed {
                        Ok(Some(frame)) => {
                            ctx.deposit(rank, frame, blob.len() as u64, &tx)
                        }
                        Ok(None) => Err(TensorError::new(&format!(
                            "relay fold: truncated RoundFrame from rank {rank}"
                        ))),
                        Err(e) => Err(e),
                    };
                    if let Err(e) = deposit {
                        eprintln!("relay fold: rank {rank}: {e}");
                        shutdown.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            },
            Ok(LenFramedRead::WouldBlock) => {
                // idle tick — loop re-checks shutdown
            }
            Ok(LenFramedRead::Eof) => {
                if let Some(ctx) = &fold {
                    // Leave the fold barrier; this may complete the
                    // round for the surviving local ranks.
                    if let Err(e) = ctx.mark_dead(rank, &tx) {
                        eprintln!("relay fold: rank {rank} exit-fold: {e}");
                        shutdown.store(true, Ordering::SeqCst);
                    }
                }
                let _ = tx.send(MuxRecord::control(RelayControlMsg::RankExit { rank }));
                break;
            }
            Err(_) => {
                // Treat any read error as the rank being gone. RankExit is
                // idempotent on the controller (declare_dead), so a
                // spurious one during teardown is harmless.
                if let Some(ctx) = &fold {
                    if let Err(e) = ctx.mark_dead(rank, &tx) {
                        eprintln!("relay fold: rank {rank} error-fold: {e}");
                        shutdown.store(true, Ordering::SeqCst);
                    }
                }
                let _ = tx.send(MuxRecord::control(RelayControlMsg::RankExit { rank }));
                break;
            }
        }
    }
    if active.fetch_sub(1, Ordering::SeqCst) == 1 {
        // Last local rank gone — nothing left to relay for this host.
        shutdown.store(true, Ordering::SeqCst);
    }
}

/// Outbound writer: the SOLE writer of the upstream connection. Drains
/// the queue, writes each record. recv_timeout lets it observe the
/// shutdown flag and the all-senders-dropped (Disconnected) condition.
fn outbound_writer(
    mut up_write: TcpStream,
    rx: mpsc::Receiver<MuxRecord>,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        match rx.recv_timeout(POLL_TIMEOUT) {
            Ok(rec) => {
                if rec.write_to(&mut up_write, &salt).is_err() {
                    shutdown.store(true, Ordering::SeqCst);
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Upstream reader: controller → loopback ranks. The SOLE writer of each
/// rank socket. Demuxes `Data{rank, blob}` to the owning rank; fans
/// `Broadcast` consensus frames out to every local alive rank (data
/// channel); applies `DeclareDead` to the fold barrier; flags shutdown
/// on upstream EOF (controller gone).
fn upstream_reader(
    mut up_read: TcpStream,
    mut rank_writes: HashMap<u32, TcpStream>,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
    fold: Option<Arc<FoldCtx>>,
    tx: mpsc::SyncSender<MuxRecord>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match MuxRecord::try_read_from(&mut up_read, &salt) {
            Ok(MuxRead::Record(MuxRecord::Data { rank, payload })) => {
                if let Some(w) = rank_writes.get_mut(&rank) {
                    // Rank socket may already be gone (rank exited); a
                    // failed write just means there's no one to deliver
                    // to. Drop it — the rank's RankExit is already in
                    // flight / delivered upstream.
                    let _ = write_len_framed(w, &payload);
                }
                // Unknown rank: controller addressed a rank not on this
                // host. Drop silently — a misroute would have failed the
                // mux HMAC, so this only happens on a benign post-exit
                // race.
            }
            Ok(MuxRead::Record(MuxRecord::Broadcast { payload })) => {
                match &fold {
                    Some(ctx) => ctx.fan_out(&payload, &mut rank_writes),
                    None => {
                        // Broadcast is a data-channel record; on the
                        // control channel it means a desynced peer.
                        eprintln!(
                            "relay: Broadcast record on the control channel; \
                             dropping (desynced controller?)"
                        );
                    }
                }
            }
            Ok(MuxRead::Record(MuxRecord::Control(
                RelayControlMsg::DeclareDead { rank },
            ))) => {
                if let Some(ctx) = &fold {
                    // Controller-side death (heartbeat staleness, a
                    // broken host elsewhere): drop the rank from the
                    // fold barrier so it cannot park the host's fold.
                    // May complete the round — the fold ships from
                    // this thread via `tx`.
                    if let Err(e) = ctx.mark_dead(rank, &tx) {
                        eprintln!("relay fold: DeclareDead({rank}): {e}");
                        shutdown.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
            Ok(MuxRead::Record(MuxRecord::Control(_))) => {
                // Hello/HelloAck only occur at startup; ignore mid-stream.
            }
            Ok(MuxRead::Record(MuxRecord::HostFrame { .. })) => {
                // Up-leg record; a controller never sends one. Drop with
                // a diagnostic.
                eprintln!("relay: unexpected HostFrame from controller; dropping");
            }
            Ok(MuxRead::WouldBlock) => {
                // idle tick
            }
            Ok(MuxRead::Eof) => {
                shutdown.store(true, Ordering::SeqCst);
                break;
            }
            Err(_) => {
                shutdown.store(true, Ordering::SeqCst);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
