//! Single-port accept mux for the controller host.
//!
//! Every cross-host channel that used to own a controller-side port
//! (NCCL rendezvous at `port`, CPU-reduce data at `port+2`, coordinator
//! control at `port+3`) now shares **one accepting port**. Dialers open
//! with a 4-byte channel-select magic (see
//! [`wire`](crate::distributed::wire) — `CHANNEL_MAGIC_*`); the
//! dispatcher thread here peeks that magic without consuming it and
//! hands the connection to the owning subsystem through a channel. The
//! subsystem then consumes and validates the magic as its first read,
//! so a directly-bound listener (tests, loopback) and a muxed source
//! see byte-identical streams.
//!
//! Routing is intentionally dumb: no salt, no frame parsing, no
//! handshake ownership. A connection that never produces 4 bytes within
//! the peek deadline, sends an unknown magic, or targets a channel
//! whose subsystem is not running (receiver dropped) is dropped loudly
//! and the dispatcher moves on — a hostile or misconfigured peer
//! condemns its own connection, never the run. Honest dialers write the
//! magic immediately after `connect`, so sequential dispatch never
//! stalls behind a legitimate peer.
//!
//! One channel is not flodl wire at all: a plain HTTP request's leading
//! `"GET "` / `"POST"` bytes route to the status responder
//! (`distributed::status`), which serves the run's membership state as
//! `state.json` and the operator start switch (`POST /start`) on this
//! same port. For that leg the consumer does NOT strip a magic — the
//! four bytes are part of the request line it reads.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::tensor::{Result, TensorError};

use super::wire::{
    CHANNEL_MAGIC_CONTROL, CHANNEL_MAGIC_DATA, CHANNEL_MAGIC_HTTP_GET, CHANNEL_MAGIC_HTTP_POST,
    CHANNEL_MAGIC_JOIN, CHANNEL_MAGIC_RENDEZVOUS,
};

/// Poll cadence of the dispatcher's non-blocking accept loop.
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// Wall-clock budget for a freshly accepted connection to produce its
/// 4 magic bytes (scaled by `FLODL_NET_TIMEOUT_SCALE`). Honest dialers
/// write them immediately after `connect`; only a wedged or hostile
/// peer runs this out.
const PEEK_TIMEOUT_SECS: u64 = 5;

/// One accepted-stream hand-off queue per muxed channel. Returned by
/// [`PortMux::start`]; wrap each receiver in
/// [`StreamSource::Mux`] and hand it to the owning subsystem. Dropping
/// an unused receiver is the correct way to declare "this channel has
/// no subsystem this run" — the dispatcher drops (resets) connections
/// routed to a closed channel, so a stray dialer fails fast instead of
/// queueing forever.
pub(crate) struct MuxAccept {
    pub rendezvous: Receiver<TcpStream>,
    pub data: Receiver<TcpStream>,
    pub control: Receiver<TcpStream>,
    pub join: Receiver<TcpStream>,
    /// Plain HTTP requests (`fdl status` / `fdl start`, curl, a
    /// browser) — a request line's leading `"GET "` / `"POST"` routes
    /// like any channel magic. Unlike the flodl channels the consumer
    /// does NOT strip a magic: the four bytes are part of the request
    /// it reads.
    pub status: Receiver<TcpStream>,
}

/// Owns the single accepting listener and its dispatcher thread.
/// Shutdown is Drop-based (flag + join), mirroring `ClusterController`.
pub(crate) struct PortMux {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    bound_port: u16,
}

impl PortMux {
    /// Start the dispatcher on a pre-bound listener (the caller owns
    /// binding and its double-run/bind-failure diagnostics). `abort` is
    /// the launcher's failure flag: the accept loop re-checks it every
    /// poll so the failure path can join this thread promptly.
    pub fn start(listener: TcpListener, abort: Arc<AtomicBool>) -> Result<(Self, MuxAccept)> {
        let bound_port = listener
            .local_addr()
            .map_err(|e| TensorError::new(&format!("port_mux: local_addr() failed: {e}")))?
            .port();
        listener
            .set_nonblocking(true)
            .map_err(|e| TensorError::new(&format!("port_mux: set_nonblocking failed: {e}")))?;

        let (rdv_tx, rdv_rx) = channel();
        let (data_tx, data_rx) = channel();
        let (ctrl_tx, ctrl_rx) = channel();
        let (join_tx, join_rx) = channel();
        let (status_tx, status_rx) = channel();

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_c = Arc::clone(&shutdown);
        let handle = thread::Builder::new()
            .name(format!("flodl-port-mux:{bound_port}"))
            .spawn(move || {
                dispatch_loop(
                    listener, rdv_tx, data_tx, ctrl_tx, join_tx, status_tx, shutdown_c, abort,
                );
            })
            .map_err(|e| TensorError::new(&format!("port_mux: spawn dispatcher failed: {e}")))?;

        Ok((
            PortMux {
                shutdown,
                handle: Some(handle),
                bound_port,
            },
            MuxAccept {
                rendezvous: rdv_rx,
                data: data_rx,
                control: ctrl_rx,
                join: join_rx,
                status: status_rx,
            },
        ))
    }

    /// Bound TCP port (kernel-assigned when the listener was bound to
    /// port 0 — test entry point).
    pub fn port(&self) -> u16 {
        self.bound_port
    }
}

impl Drop for PortMux {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_loop(
    listener: TcpListener,
    rdv_tx: Sender<TcpStream>,
    data_tx: Sender<TcpStream>,
    ctrl_tx: Sender<TcpStream>,
    join_tx: Sender<TcpStream>,
    status_tx: Sender<TcpStream>,
    shutdown: Arc<AtomicBool>,
    abort: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) || abort.load(Ordering::SeqCst) {
            return;
        }
        let (stream, peer) = match listener.accept() {
            Ok(pair) => pair,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL);
                continue;
            }
            Err(e) => {
                // Listener-level failure (fd exhaustion, teardown race):
                // loud and terminal for the dispatcher. Subsystems see
                // their sources disconnect and surface their own errors.
                eprintln!("port_mux: accept failed: {e}");
                return;
            }
        };
        dispatch_one(
            stream, peer, &rdv_tx, &data_tx, &ctrl_tx, &join_tx, &status_tx,
        );
    }
}

/// Peek the channel magic and route the stream. Failures drop the
/// connection (loudly), never the dispatcher.
#[allow(clippy::too_many_arguments)]
fn dispatch_one(
    stream: TcpStream,
    peer: SocketAddr,
    rdv_tx: &Sender<TcpStream>,
    data_tx: &Sender<TcpStream>,
    ctrl_tx: &Sender<TcpStream>,
    join_tx: &Sender<TcpStream>,
    status_tx: &Sender<TcpStream>,
) {
    // Controller-side cleartext guard: a public peer on the mux port
    // means training frames cross an uncontrolled network unencrypted.
    crate::distributed::wire::warn_cleartext_public_peer("cluster controller", peer);
    // Accepted socket inherits the listener's non-blocking flag; flip it
    // back so the peek deadline below is honored via SO_RCVTIMEO.
    if stream.set_nonblocking(false).is_err() {
        return;
    }
    let deadline_secs = crate::distributed::wire::scaled_deadline_secs(PEEK_TIMEOUT_SECS);
    if stream
        .set_read_timeout(Some(Duration::from_secs(deadline_secs)))
        .is_err()
    {
        return;
    }

    let magic = match peek_magic(&stream, Duration::from_secs(deadline_secs)) {
        Ok(m) => m,
        Err(why) => {
            eprintln!(
                "port_mux: dropping connection from {peer} ({why}); \
                 continuing to accept"
            );
            return;
        }
    };

    // Hand the stream over with a pristine timeout config — each
    // subsystem sets its own.
    if stream.set_read_timeout(None).is_err() {
        return;
    }
    let (tx, name) = match magic {
        CHANNEL_MAGIC_RENDEZVOUS => (rdv_tx, "rendezvous"),
        CHANNEL_MAGIC_DATA => (data_tx, "data"),
        CHANNEL_MAGIC_CONTROL => (ctrl_tx, "control"),
        CHANNEL_MAGIC_JOIN => (join_tx, "join"),
        CHANNEL_MAGIC_HTTP_GET | CHANNEL_MAGIC_HTTP_POST => (status_tx, "status"),
        other => {
            eprintln!(
                "port_mux: dropping connection from {peer} (unknown channel \
                 magic 0x{other:08x}); continuing to accept"
            );
            return;
        }
    };
    if tx.send(stream).is_err() {
        // Receiver dropped: no subsystem owns this channel in this run
        // (e.g. rendezvous on a CPU-backend run). Dropping the stream
        // here resets the dialer, which fails fast with a loud error on
        // its side instead of hanging on a never-served connection.
        eprintln!(
            "port_mux: dropping connection from {peer} (no {name}-channel \
             subsystem running); continuing to accept"
        );
    }
}

/// Peek until 4 bytes are available (without consuming them), the peer
/// hangs up, or `deadline` elapses. The blocking `peek` returns as soon
/// as ≥1 byte is readable, so a magic split across TCP segments loops
/// here rather than failing.
fn peek_magic(stream: &TcpStream, deadline: Duration) -> std::result::Result<u32, String> {
    let start = Instant::now();
    let mut buf = [0u8; 4];
    loop {
        match stream.peek(&mut buf) {
            Ok(0) => return Err("closed before channel magic".to_string()),
            Ok(n) if n >= 4 => return Ok(u32::from_le_bytes(buf)),
            Ok(_) => {
                if start.elapsed() > deadline {
                    return Err("channel magic incomplete within deadline".to_string());
                }
                thread::sleep(Duration::from_millis(5));
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return Err("no channel magic within deadline".to_string());
            }
            Err(e) => return Err(format!("peek failed: {e}")),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamSource: uniform accept front for direct listeners and mux legs
// ---------------------------------------------------------------------------

/// Where a subsystem's accept loop gets its connections: a directly
/// bound [`TcpListener`] (tests, single-channel setups) or one leg of
/// the [`PortMux`]. Both yield streams whose first unread bytes are the
/// channel magic — the subsystem consumes and validates it either way.
pub(crate) enum StreamSource {
    Listener(TcpListener),
    Mux(Receiver<TcpStream>),
}

impl StreamSource {
    /// Wrap a directly bound listener (flips it non-blocking so
    /// [`Self::try_accept`] can poll).
    pub fn from_listener(listener: TcpListener, what: &str) -> Result<Self> {
        listener
            .set_nonblocking(true)
            .map_err(|e| TensorError::new(&format!("{what}: set_nonblocking failed: {e}")))?;
        Ok(StreamSource::Listener(listener))
    }

    /// Non-blocking accept: `Ok(Some)` with a blocking-mode stream,
    /// `Ok(None)` when nothing is pending (callers poll-sleep), `Err`
    /// when the source is gone (listener error / mux dispatcher exited).
    pub fn try_accept(&self, what: &str) -> Result<Option<TcpStream>> {
        match self {
            StreamSource::Listener(listener) => match listener.accept() {
                Ok((stream, _peer)) => {
                    // Accepted socket may inherit non-blocking; flip it
                    // back so per-stream read/write timeouts are honored.
                    stream.set_nonblocking(false).map_err(|e| {
                        TensorError::new(&format!("{what}: set_nonblocking(false) failed: {e}"))
                    })?;
                    Ok(Some(stream))
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
                Err(e) => Err(TensorError::new(&format!("{what}: accept failed: {e}"))),
            },
            StreamSource::Mux(rx) => match rx.try_recv() {
                Ok(stream) => Ok(Some(stream)),
                Err(TryRecvError::Empty) => Ok(None),
                Err(TryRecvError::Disconnected) => Err(TensorError::new(&format!(
                    "{what}: port mux dispatcher exited"
                ))),
            },
        }
    }
}

#[cfg(test)]
#[path = "port_mux_tests.rs"]
mod tests;
