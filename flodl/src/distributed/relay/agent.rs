//! Per-host relay agent: the byte-router (transport only, v1).
//!
//! One [`RelayChannel`] runs per {data, control} channel per host. It
//! terminates each local rank's handshake on a loopback listener exactly
//! as the controller would, then multiplexes every local rank's frames
//! over a single upstream connection to the real controller — tagging
//! each forwarded frame with its rank ([`MuxRecord::Data`]) and emitting
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
//!   blobs off the rank's loopback socket, wraps each as
//!   `MuxRecord::Data{rank, blob}`, and pushes it to the outbound queue.
//!   On EOF / error it pushes `RankExit{rank}` and exits.
//! - **one outbound-writer thread**: the *sole* writer of the upstream
//!   connection (single-writer discipline — see
//!   `feedback_no_locks_hot_path` / the historical two-writer HMAC race).
//!   Drains the outbound queue, writes each [`MuxRecord`] upstream.
//! - **one upstream-reader thread**: reads [`MuxRecord`]s from the
//!   controller, and for each `Data{rank, blob}` writes the length-framed
//!   blob to that rank's loopback socket. It is the sole writer of each
//!   rank socket (single-writer per rank). On upstream EOF it flags
//!   shutdown (controller gone → tear down the host).
//!
//! The relay FORWARDS; it never parses a forwarded payload. Averaging
//! math + snapshot path are untouched; this is pure transport.
//!
//! [`MuxRecord`]: super::mux::MuxRecord
//! [`MuxRecord::Data`]: super::mux::MuxRecord::Data
//! [`RelayControlMsg::RankExit`]: super::mux::RelayControlMsg::RankExit

use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

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
    /// CPU-averaging data channel (`controller_port + 2`,
    /// [`crate::distributed::controller`]). Bare handshake.
    Data,
    /// Coordinator control channel (`controller_port + 3`,
    /// [`crate::distributed::cluster_coordinator`]). Salt-authenticated
    /// handshake.
    Control,
}

impl ChannelKind {
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
                .set_write_timeout(Some(crate::distributed::wire::WRITE_STALL_TIMEOUT));
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
        // Write-stall ceiling: a wedged controller errors outbound_writer,
        // which flags relay shutdown — reachable teardown instead of a
        // parked writer holding the host hostage.
        let _ = upstream
            .set_write_timeout(Some(crate::distributed::wire::WRITE_STALL_TIMEOUT));
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
        let threads = spawn_mux(rank_streams, upstream, salt, Arc::clone(&shutdown))?;
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
    rank_streams: Vec<(u32, TcpStream)>,
    upstream: TcpStream,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
) -> Result<Vec<JoinHandle<()>>> {
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

    // Upstream reader (sole writer of each rank socket).
    {
        let shutdown = Arc::clone(&shutdown);
        threads.push(
            thread::Builder::new()
                .name("flodl-relay-up".into())
                .spawn(move || upstream_reader(up_read, rank_writes, salt, shutdown))
                .map_err(|e| TensorError::new(&format!("relay: spawn upstream reader: {e}")))?,
        );
    }

    // Rank readers.
    for (rank, stream) in rank_reads {
        let tx = tx.clone();
        let shutdown = Arc::clone(&shutdown);
        let active = Arc::clone(&active);
        threads.push(
            thread::Builder::new()
                .name(format!("flodl-relay-r{rank}"))
                .spawn(move || rank_reader(rank, stream, tx, shutdown, active))
                .map_err(|e| TensorError::new(&format!("relay: spawn rank {rank} reader: {e}")))?,
        );
    }
    // Drop the template sender so the outbound writer's rx disconnects
    // once every rank-reader has dropped its clone (all ranks gone).
    drop(tx);

    Ok(threads)
}

/// Rank-reader: loopback rank socket → outbound queue. Wraps each
/// length-framed blob as `Data{rank, blob}`; emits `RankExit{rank}` on
/// EOF/error then exits. The last reader to exit flags shutdown.
fn rank_reader(
    rank: u32,
    mut stream: TcpStream,
    tx: mpsc::SyncSender<MuxRecord>,
    shutdown: Arc<AtomicBool>,
    active: Arc<AtomicUsize>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match try_read_len_framed(&mut stream) {
            Ok(LenFramedRead::Blob(blob)) => {
                if tx.send(MuxRecord::data(rank, blob)).is_err() {
                    break; // outbound writer gone
                }
            }
            Ok(LenFramedRead::WouldBlock) => {
                // idle tick — loop re-checks shutdown
            }
            Ok(LenFramedRead::Eof) => {
                let _ = tx.send(MuxRecord::control(RelayControlMsg::RankExit { rank }));
                break;
            }
            Err(_) => {
                // Treat any read error as the rank being gone. RankExit is
                // idempotent on the controller (declare_dead), so a
                // spurious one during teardown is harmless.
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
/// rank socket. Demuxes `Data{rank, blob}` to the owning rank; flags
/// shutdown on upstream EOF (controller gone).
fn upstream_reader(
    mut up_read: TcpStream,
    mut rank_writes: HashMap<u32, TcpStream>,
    salt: SessionSalt,
    shutdown: Arc<AtomicBool>,
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
            Ok(MuxRead::Record(MuxRecord::Control(_))) => {
                // No downstream relay-control signals defined in v1 beyond
                // the HelloAck already consumed during startup. Ignore.
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
