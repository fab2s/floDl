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
//! realized-work reduce. Rank readers parse, HMAC-verify, and fold each
//! local frame INCREMENTALLY into the round's running f32 sums (the
//! `sum_frames` monoid computed one deposit at a time — element-wise
//! sum, masses too; the fold NEVER divides, the controller divides
//! exactly once); the depositor that completes the round (every local
//! *alive* rank merged) re-encodes the sums and ships ONE re-signed
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

use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::distributed::controller::{self, RoundKind, TensorPayload};
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
                let rank = crate::distributed::wire::read_handshake_rank(
                    stream,
                    world_size as u32,
                    salt,
                )?;
                crate::distributed::wire::write_handshake_ack(stream, salt)?;
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

/// Host-local mirror of the controller's round collection: a running
/// incremental fold plus the local dead-set. Each deposit merges into
/// the fold as it arrives; the mutation that completes the round (a
/// deposit, or a death that removes the last missing rank) takes the
/// fold and ships it — no dedicated fold thread, no condvar, no
/// deadline.
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
    /// This round's running fold (`None` until the first deposit;
    /// taken by the completing fold).
    fold: Option<HostFold>,
    /// Local ranks out of the fold barrier: loopback EOF (observed
    /// here) or controller-declared death ([`RelayControlMsg::DeclareDead`]).
    dead: HashSet<u32>,
}

/// One round's incremental host fold: the running f32 sums every local
/// deposit merges into, plus the frame schema the first deposit pinned.
///
/// This IS `sum_frames` computed one contribution at a time (same f32
/// accumulation, same schema validation, mass summed, NEVER divides) —
/// the associative monoid makes deposit order irrelevant. Holding the
/// f32 image instead of the deposited frames caps the fold's residency
/// at one f32 model copy regardless of how many local ranks feed it.
///
/// MAC-BEFORE-USE NOTE: a deposit merges payload bytes into the shared
/// sums BEFORE its frame's HMAC footer is reached (the parse streams).
/// That is sound ONLY because every deposit error — HMAC, schema,
/// truncation — tears the whole data channel down (`rank_reader` flags
/// shutdown; the round never ships): no code path continues with a
/// polluted accumulator. Any future error-tolerant deposit path must
/// buffer per-deposit and merge after verification instead.
struct HostFold {
    kind: RoundKind,
    /// Σ realized-work mass of the deposits (never divided here — the
    /// divide-once law belongs to the controller).
    weight: f64,
    /// The fold's tensor state — seed until a second deposit arrives.
    payloads: FoldPayloads,
    /// Ranks whose contribution is already merged (the double-deposit
    /// detector the frames map used to provide).
    deposited: HashSet<u32>,
}

/// The fold's tensor state, staged to keep residency at the wire dtype
/// for as long as possible.
///
/// The round's FIRST deposit is held verbatim as its wire payloads
/// (`Seed`) — on a bf16 wire that is HALF the f32 image, and it is what
/// the barrier hold phase (seconds long, while later ranks compute) has
/// resident. The f32 sums only materialize when a second NON-ELIDED
/// deposit arrives (`Sums`) — elided deposits (wire zero-elision) merge
/// nothing, so they never force the promotion. A single-rank host
/// therefore never builds the f32 image at all: its fold ships the seed
/// verbatim — byte-identical to decode + re-encode, since the bf16
/// codec round-trips its own output exactly — which makes the relay
/// tier nearly free on 1-GPU-per-node topologies (the cloud shape). The
/// same seed-verbatim ship carries an all-elided round upstream still
/// elided, whatever the host's rank count.
enum FoldPayloads {
    /// First deposit, verbatim wire payloads (dtype + shape double as
    /// the schema later deposits are validated against).
    Seed(Vec<TensorPayload>),
    /// Two or more deposits merged: per-tensor running sums, always
    /// f32 whatever the wire dtype.
    Sums {
        schema: Vec<FoldSchema>,
        sums: Vec<Vec<f32>>,
    },
}

/// Wire identity of one tensor slot in the fold schema.
struct FoldSchema {
    dtype: u8,
    shape: Vec<u32>,
}

/// Loud schema mismatch shared by the promote and accumulate paths.
fn fold_schema_check(
    rank: u32,
    ti: usize,
    got_dtype: u8,
    got_shape: &[u32],
    want_dtype: u8,
    want_shape: &[u32],
) -> Result<()> {
    if got_dtype != want_dtype {
        return Err(TensorError::new(&format!(
            "relay fold: rank {rank} tensor[{ti}] dtype {got_dtype} != the round's \
             dtype {want_dtype} (cohort mixing bf16_wire settings, or desynced \
             rounds)"
        )));
    }
    if got_shape != want_shape {
        return Err(TensorError::new(&format!(
            "relay fold: rank {rank} tensor[{ti}] shape {got_shape:?} != the \
             round's shape {want_shape:?} (desynced rounds)"
        )));
    }
    Ok(())
}

impl HostFold {
    /// Merge one rank's frame blob into the running fold, seeding it on
    /// the round's first deposit. The blob is parsed payload-by-payload
    /// ([`controller::read_round_frame_streamed`]); each payload's bytes
    /// are decoded into the f32 sums and freed before the next one is
    /// read — the deposited frame never exists as a `RoundFrame`.
    ///
    /// Elided payloads (wire zero-elision — an idle rank's zero-mass
    /// contribution, a formation-broadcast zeros frame) are zeros by
    /// schema and merge nothing: they ride the schema checks alone, and
    /// seed→sums promotion is LAZY (it fires at the round's first
    /// non-elided post-seed payload). A round whose every deposit is
    /// elided therefore keeps its seed verbatim and ships still elided
    /// — the elision composes through the fold tier instead of
    /// rematerializing a model of zero bytes.
    ///
    /// Loud error on any schema disagreement with the pinned first
    /// deposit — tensor count, dtype, shape, byte length, or
    /// [`RoundKind`] — same contract as `sum_frames` (a mismatch means
    /// desynced rounds, e.g. a cohort mixing `bf16_wire` settings).
    fn accumulate(
        fold: &mut Option<HostFold>,
        rank: u32,
        blob: &[u8],
        salt: &SessionSalt,
    ) -> Result<()> {
        let Some(f) = fold else {
            // Round's first deposit: hold it verbatim as the seed (wire
            // dtype residency; also pins the schema).
            let mut payloads: Vec<TensorPayload> = Vec::new();
            let hdr = controller::read_round_frame_streamed(
                &mut &blob[..],
                salt,
                &mut |_, p| {
                    payloads.push(p);
                    Ok(())
                },
            )?;
            let Some((kind, weight)) = hdr else {
                return Err(TensorError::new(&format!(
                    "relay fold: truncated RoundFrame from rank {rank}"
                )));
            };
            *fold = Some(HostFold {
                kind,
                weight,
                payloads: FoldPayloads::Seed(payloads),
                deposited: HashSet::from([rank]),
            });
            return Ok(());
        };

        // Second and later deposits stream payload-by-payload and are
        // never materialized. Seed→sums promotion is lazy (see the doc
        // above): it fires at the first non-elided post-seed payload,
        // and an elided seed promotes to zeros of its declared shape
        // (`payload_to_f32` sizes from numel) when a full deposit does
        // arrive.
        let payloads = &mut f.payloads;
        let expected = match &*payloads {
            FoldPayloads::Seed(seed) => seed.len(),
            FoldPayloads::Sums { schema, .. } => schema.len(),
        };
        let mut seen = 0usize;
        let hdr = controller::read_round_frame_streamed(
            &mut &blob[..],
            salt,
            &mut |ti, p| {
                // Schema check against whichever form the fold holds;
                // the borrow ends with this block so promotion below
                // can take the payloads mutably.
                {
                    let (want_dtype, want_shape): (u8, &[u32]) = match &*payloads {
                        FoldPayloads::Seed(seed) => match seed.get(ti) {
                            Some(s) => (s.dtype, &s.shape),
                            None => {
                                return Err(TensorError::new(&format!(
                                    "relay fold: rank {rank} frame carries more than \
                                     {expected} tensors (schema pinned by the round's \
                                     first deposit)"
                                )));
                            }
                        },
                        FoldPayloads::Sums { schema, .. } => match schema.get(ti) {
                            Some(sc) => (sc.dtype, &sc.shape),
                            None => {
                                return Err(TensorError::new(&format!(
                                    "relay fold: rank {rank} frame carries more than \
                                     {expected} tensors (schema pinned by the round's \
                                     first deposit)"
                                )));
                            }
                        },
                    };
                    fold_schema_check(rank, ti, p.dtype, &p.shape, want_dtype, want_shape)?;
                }
                seen = ti + 1;
                if p.is_elided() {
                    // Zeros by schema — nothing to merge, and the seed
                    // (whatever its form) stays exactly as it was.
                    return Ok(());
                }
                if let FoldPayloads::Seed(seed) = payloads {
                    let mut schema: Vec<FoldSchema> = Vec::with_capacity(seed.len());
                    let mut sums: Vec<Vec<f32>> = Vec::with_capacity(seed.len());
                    for sp in seed.iter_mut() {
                        sums.push(controller::payload_to_f32(sp).map_err(|e| {
                            TensorError::new(&format!("relay fold: seed promotion: {e}"))
                        })?);
                        schema.push(FoldSchema {
                            dtype: sp.dtype,
                            shape: std::mem::take(&mut sp.shape),
                        });
                        // Drain the seed as the f32 image grows instead
                        // of holding both whole.
                        sp.bytes = Vec::new();
                    }
                    *payloads = FoldPayloads::Sums { schema, sums };
                }
                let FoldPayloads::Sums { sums, .. } = payloads else {
                    unreachable!("promoted just above");
                };
                let sum = sums
                    .get_mut(ti)
                    .expect("bounds established by the schema match above");
                // Byte-length agreement is enforced inside the
                // accumulate (payload bytes vs sums numel).
                controller::accumulate_payload_into(&p, sum).map_err(|e| {
                    TensorError::new(&format!("relay fold: rank {rank} tensor[{ti}]: {e}"))
                })?;
                Ok(())
            },
        )?;
        let Some((kind, weight)) = hdr else {
            return Err(TensorError::new(&format!(
                "relay fold: truncated RoundFrame from rank {rank}"
            )));
        };
        if kind != f.kind {
            return Err(TensorError::new(&format!(
                "relay fold: rank {rank} frame kind {kind:?} != the round's kind \
                 {:?} (desynced reduce rounds)",
                f.kind
            )));
        }
        if seen != expected {
            return Err(TensorError::new(&format!(
                "relay fold: rank {rank} frame carries {seen} tensors; the round's \
                 first deposit carried {expected}"
            )));
        }
        f.weight += weight;
        f.deposited.insert(rank);
        Ok(())
    }
}

impl FoldCtx {
    fn new(local_ranks: Vec<u32>, salt: SessionSalt) -> Self {
        FoldCtx {
            inner: Mutex::new(FoldInner {
                fold: None,
                dead: HashSet::new(),
            }),
            local_ranks,
            salt,
            rounds: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_up: AtomicU64::new(0),
        }
    }

    /// Deposit `rank`'s frame blob for the current round, folding it
    /// INCREMENTALLY into the running f32 sums (the frame is parsed
    /// payload-by-payload straight out of the blob and never
    /// materialized — with N model-sized local frames per round, the
    /// old hold-then-sum kept N of them until the barrier; the
    /// accumulator holds ONE f32 image regardless of N). Ships the
    /// re-encoded fold if this deposit completes the round.
    ///
    /// Returns `Err` on protocol violations that must tear the channel
    /// down (double deposit — the rank↔relay leg is a strict ping-pong,
    /// so a second frame before the round folded means a desynced
    /// stream — plus every schema/parse/HMAC failure from the
    /// incremental fold, see [`HostFold`]).
    fn deposit(
        &self,
        rank: u32,
        blob: &[u8],
        tx: &mpsc::SyncSender<MuxRecord>,
    ) -> Result<()> {
        let taken = {
            let mut inner = self.inner.lock().expect("relay fold lock poisoned");
            if inner.dead.contains(&rank) {
                // Late frame from a rank already out of the barrier
                // (mirrors the controller's late-frame drop).
                return Ok(());
            }
            if inner
                .fold
                .as_ref()
                .is_some_and(|f| f.deposited.contains(&rank))
            {
                return Err(TensorError::new(&format!(
                    "relay fold: rank {rank} deposited twice in one round \
                     (rank↔relay ping-pong violated; stream desynced)"
                )));
            }
            HostFold::accumulate(&mut inner.fold, rank, blob, &self.salt)?;
            self.bytes_in.fetch_add(blob.len() as u64, Ordering::Relaxed);
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
            // A contribution it already deposited this round stays in —
            // it is already merged into the running sums; the rank
            // realized that work, and the controller-side accept ledger
            // is what decides acceptance across hosts.
            self.take_if_complete(&mut inner)
        };
        self.fold_and_ship(taken, tx)
    }

    /// Under the lock: if every local alive rank has deposited (and at
    /// least one contribution is in), take the round's fold.
    fn take_if_complete(&self, inner: &mut FoldInner) -> Option<HostFold> {
        let complete = self.local_ranks.iter().all(|r| {
            inner.dead.contains(r)
                || inner
                    .fold
                    .as_ref()
                    .is_some_and(|f| f.deposited.contains(r))
        });
        if !complete {
            return None;
        }
        inner.fold.take()
    }

    /// Outside the lock: re-encode the fold as ONE HostFrame payload
    /// and ship it upstream. A single-deposit round (`Seed`) ships its
    /// wire payloads verbatim — byte-identical to decode + re-encode,
    /// the codec round-trips its own output exactly — so a 1-rank host
    /// never touches f32. A merged round (`Sums`) streams straight from
    /// the f32 accumulator into the payload buffer, draining each
    /// tensor as it is encoded — no folded `RoundFrame` object, no
    /// separate serialize pass (the old path materialized the folded
    /// frame AND its serialization on top of the held rank frames).
    fn fold_and_ship(
        &self,
        taken: Option<HostFold>,
        tx: &mpsc::SyncSender<MuxRecord>,
    ) -> Result<()> {
        let Some(fold) = taken else {
            return Ok(());
        };
        let HostFold {
            kind,
            weight,
            payloads,
            deposited: _,
        } = fold;
        let mut buf: Vec<u8>;
        match payloads {
            FoldPayloads::Seed(seed) => {
                let parts: Vec<controller::PayloadPart<'_>> = seed
                    .iter()
                    .map(|p| controller::PayloadPart {
                        dtype: p.dtype,
                        shape: &p.shape,
                        nbytes: p.bytes.len() as u64,
                    })
                    .collect();
                buf = Vec::with_capacity(
                    controller::round_frame_wire_len(&parts) as usize,
                );
                controller::write_round_frame_streamed(
                    &mut buf,
                    kind,
                    weight,
                    &parts,
                    &self.salt,
                    &mut |ti, tee| {
                        use std::io::Write;
                        tee.write_all(&seed[ti].bytes)
                            .map_err(|e| TensorError::new(&e.to_string()))
                    },
                )?;
                drop(seed);
            }
            FoldPayloads::Sums { schema, mut sums } => {
                let parts: Vec<controller::PayloadPart<'_>> = schema
                    .iter()
                    .zip(sums.iter())
                    .map(|(s, sum)| {
                        Ok(controller::PayloadPart {
                            dtype: s.dtype,
                            shape: &s.shape,
                            nbytes: (sum.len()
                                * controller::payload_element_size(s.dtype)?)
                                as u64,
                        })
                    })
                    .collect::<Result<_>>()?;
                buf = Vec::with_capacity(
                    controller::round_frame_wire_len(&parts) as usize,
                );
                controller::write_round_frame_streamed(
                    &mut buf,
                    kind,
                    weight,
                    &parts,
                    &self.salt,
                    &mut |ti, tee| {
                        use std::io::Write;
                        // Drain: this tensor's sums are freed as soon as
                        // its wire bytes exist, so the accumulator
                        // shrinks while the payload buffer grows instead
                        // of coexisting whole.
                        let sum = std::mem::take(&mut sums[ti]);
                        let bytes = controller::f32_slice_to_payload_bytes(
                            &sum,
                            schema[ti].dtype,
                        )?;
                        drop(sum);
                        tee.write_all(&bytes)
                            .map_err(|e| TensorError::new(&e.to_string()))
                    },
                )?;
            }
        }
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
/// HMAC-verifies the blob as a
/// [`RoundFrame`](crate::distributed::controller::RoundFrame) and folds
/// it incrementally into the host fold — the deposit that completes the
/// round ships the re-encoded `HostFrame` from this thread.
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
                    // Parse + verify + fold in one pass (the blob's
                    // end-to-end HMAC is checked by the streamed parse;
                    // the relay holds the session salt). Any failure
                    // here is a desynced or corrupt local stream — tear
                    // the channel down loudly rather than fold garbage
                    // (which is also what keeps the merge-before-verify
                    // accumulation sound, see [`HostFold`]).
                    if let Err(e) = ctx.deposit(rank, &blob, &tx) {
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
