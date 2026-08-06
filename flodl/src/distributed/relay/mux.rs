//! Relay multiplexing wire format for the per-host transport tier.
//!
//! The relay (one per host) collapses the cluster's
//! one-TCP-connection-per-rank topology into one connection per host
//! per channel. Toward its local ranks the relay looks exactly like the
//! controller does today (same [`RoundFrame`] / [`ControlFrame`]
//! protocol); upstream toward the real controller it speaks the
//! *muxed* protocol defined here.
//!
//! # Two legs, two framings
//!
//! ```text
//!   rank  --[len-framed opaque blob]-->  relay  --[MuxRecord]-->  controller
//!         <--[len-framed opaque blob]--         <--[MuxRecord]--
//! ```
//!
//! - **rank ↔ relay (loopback):** each frame the rank would have sent
//!   to the controller is length-delimited with a bare 4-byte prefix
//!   ([`write_len_framed`] / [`read_len_framed`]). The blob is the
//!   existing [`RoundFrame`] / [`ControlFrame`] bytes verbatim — already
//!   HMAC-authed, so the loopback prefix carries no auth of its own.
//!
//! - **relay ↔ controller (network):** frames are wrapped in
//!   [`MuxRecord`]s so the single per-host connection can carry the
//!   host's traffic. The mux header (including the routing-sensitive
//!   `rank` field where present) is HMAC-authed with the session salt,
//!   mirroring the rest of the cluster wire protocol — a flipped `rank`
//!   would misroute, so the tag must be tamper-evident.
//!
//! # Control channel: pure transport. Data channel: fold.
//!
//! On the **control channel** the relay FORWARDS: each rank's
//! [`ControlFrame`] crosses untouched as a rank-tagged
//! [`MuxRecord::Data`], and the payload keeps its end-to-end HMAC.
//!
//! On the **data channel** the relay FOLDS: it parses and verifies its
//! local ranks' [`RoundFrame`]s, sums them element-wise (masses too),
//! and sends ONE re-signed [`MuxRecord::HostFrame`] upstream per reduce
//! round; the controller answers with ONE [`MuxRecord::Broadcast`]
//! consensus that the relay fans out to every local rank. K local ranks
//! therefore cost 1× the model bytes on the host uplink in each
//! direction instead of K×. The relay is a trusted aggregation point on
//! this leg — it holds the session salt (it terminates the rank
//! handshakes with it), so frame authenticity inside the host is
//! salt-scoped, not per-hop.
//!
//! [`RoundFrame`]: crate::distributed::controller::RoundFrame
//! [`ControlFrame`]: crate::distributed::wire::ControlFrame

use std::io::{ErrorKind, Read, Write};

use serde::{Deserialize, Serialize};

use crate::distributed::wire::{hmac_sha256_64, SessionSalt};
use crate::tensor::{Result, TensorError};

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// Magic prefix on every [`MuxRecord`] header (relay ↔ controller leg).
pub const MUX_RECORD_MAGIC: u32 = 0xF10D_17D0;

/// Wire version of the relay-mux protocol. Independent of the control-
/// and data-channel protocol versions. Bump on any breaking change to
/// the [`MuxRecord`] header or to [`RelayControlMsg`].
pub const MUX_PROTOCOL_VERSION: u32 = 1;

/// Fixed mux header size: magic(4) + version(4) + kind(1) + rank(4) +
/// payload_len(4) + auth_tag(8).
const MUX_HEADER_LEN: usize = 25;

/// Record kind: a forwarded opaque frame blob tagged with its rank.
const REC_DATA: u8 = 0x01;
/// Record kind: a relay-level control signal ([`RelayControlMsg`]).
const REC_CONTROL: u8 = 0x02;
/// Record kind: a relay-folded host frame (data channel, relay →
/// controller). One per host per reduce round; no rank tag.
const REC_HOST_FRAME: u8 = 0x03;
/// Record kind: a host-wide consensus frame (data channel, controller →
/// relay). The relay fans it out to every local alive rank; no rank tag.
const REC_BROADCAST: u8 = 0x04;

// ---------------------------------------------------------------------------
// Bincode helpers (local; mirror wire.rs's private pair)
// ---------------------------------------------------------------------------

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, bincode_config())
        .map_err(|e| TensorError::new(&format!("relay_mux: bincode encode failed: {e}")))
}

fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T> {
    let (v, _used) = bincode::serde::decode_from_slice(bytes, bincode_config())
        .map_err(|e| TensorError::new(&format!("relay_mux: bincode decode failed: {e}")))?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Relay-level control signals
// ---------------------------------------------------------------------------

/// Relay-level control messages exchanged over the per-host connection,
/// distinct from the forwarded data-plane frames. Carried inside a
/// [`MuxRecord::Control`] (bincode payload), HMAC-authed like every mux
/// record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelayControlMsg {
    /// Relay → controller upstream handshake (one per channel, first
    /// record on the connection). Announces which global ranks this
    /// host's connection carries so the controller can map
    /// connection → ranks (for demux and for declaring all of a host's
    /// ranks dead when the connection drops).
    Hello {
        /// Relay host name (diagnostic; controller logs it).
        host: String,
        /// Global ranks carried by this connection (the host's local
        /// rank set, in ascending order).
        ranks: Vec<u32>,
        /// Per-rank model signatures, aligned with `ranks`. Empty on
        /// the data channel (its bare handshake carries none); the
        /// coordinator compares only what arrives non-empty.
        model_sigs: Vec<[u8; 32]>,
    },
    /// Controller → relay handshake acknowledgement.
    HelloAck,
    /// Relay → controller: a local rank's loopback connection closed
    /// (clean exit or crash — the relay does not distinguish in v1). The
    /// controller declares the rank dead so its reduce barrier releases
    /// instead of waiting forever. Host death is signalled implicitly by
    /// the whole per-host connection EOFing, not by this message.
    RankExit { rank: u32 },
    /// Controller → relay (data channel): `rank` has been declared dead
    /// cluster-side (coordinator heartbeat staleness, scatter failure on
    /// another host, ...). The relay drops the rank from its fold
    /// barrier so a wedged-but-connected local rank cannot park the
    /// host's fold forever — the exact mirror of the controller's own
    /// `wait_for_round` re-evaluating [`DeadRanks`] every poll. Local
    /// EOFs the relay observes itself; this covers the deaths only the
    /// controller can see.
    ///
    /// [`DeadRanks`]: crate::distributed::controller::DeadRanks
    DeclareDead { rank: u32 },
}

// ---------------------------------------------------------------------------
// MuxRecord (relay ↔ controller leg)
// ---------------------------------------------------------------------------

/// One framed record on the relay ↔ controller connection.
///
/// Either a [`Self::Data`] blob tagged with its originating rank, or a
/// relay-level [`Self::Control`] signal. Written by [`Self::write_to`],
/// read by [`Self::read_from`] / [`Self::try_read_from`].
#[derive(Debug, Clone, PartialEq)]
pub enum MuxRecord {
    /// A forwarded opaque frame for `rank` — the verbatim
    /// [`crate::distributed::wire::ControlFrame`] bytes (control
    /// channel), never parsed by the relay.
    Data { rank: u32, payload: Vec<u8> },
    /// A relay-level control signal.
    Control(RelayControlMsg),
    /// Data channel, relay → controller: the host's folded
    /// [`crate::distributed::controller::RoundFrame`] for one reduce
    /// round — the element-wise sum of every accepted local rank's
    /// contribution, mass summed, re-signed by the relay. One per host
    /// per round; carries no rank tag.
    HostFrame { payload: Vec<u8> },
    /// Data channel, controller → relay: the round's consensus
    /// [`crate::distributed::controller::RoundFrame`], identical for
    /// every rank. The relay writes it to each local alive rank's
    /// loopback socket; carries no rank tag.
    Broadcast { payload: Vec<u8> },
}

/// Hard ceiling on any length-prefixed payload (mux records and
/// len-framed blobs): [`crate::distributed::wire::frame_ceiling`] — the
/// model-derived session bound when the process has installed one
/// (launcher probe / `RelaySpec` / rank model), a 1 GiB default
/// otherwise. The length field is UNAUTHENTICATED until the trailing
/// MAC verifies, so a hostile or corrupt peer can claim up to
/// `u32::MAX` and force the reader to buffer it before rejection —
/// incremental allocation makes the attacker pay the bandwidth, the
/// ceiling bounds the memory.
use crate::distributed::wire::frame_ceiling;

impl MuxRecord {
    /// Tag an opaque frame blob with its originating rank.
    pub fn data(rank: u32, payload: Vec<u8>) -> Self {
        MuxRecord::Data { rank, payload }
    }

    /// Wrap a relay-level control signal.
    pub fn control(msg: RelayControlMsg) -> Self {
        MuxRecord::Control(msg)
    }

    /// Wrap a relay-folded host frame (data channel, up-leg).
    pub fn host_frame(payload: Vec<u8>) -> Self {
        MuxRecord::HostFrame { payload }
    }

    /// Wrap a host-wide consensus frame (data channel, down-leg).
    pub fn broadcast(payload: Vec<u8>) -> Self {
        MuxRecord::Broadcast { payload }
    }

    /// `(record_kind, rank, payload_bytes)` for the wire header.
    fn parts(&self) -> Result<(u8, u32, std::borrow::Cow<'_, [u8]>)> {
        match self {
            MuxRecord::Data { rank, payload } => {
                Ok((REC_DATA, *rank, std::borrow::Cow::Borrowed(payload)))
            }
            MuxRecord::Control(msg) => {
                Ok((REC_CONTROL, 0, std::borrow::Cow::Owned(encode(msg)?)))
            }
            MuxRecord::HostFrame { payload } => {
                Ok((REC_HOST_FRAME, 0, std::borrow::Cow::Borrowed(payload)))
            }
            MuxRecord::Broadcast { payload } => {
                Ok((REC_BROADCAST, 0, std::borrow::Cow::Borrowed(payload)))
            }
        }
    }

    /// Serialize the full mux header + payload to the writer.
    ///
    /// The `auth_tag` authenticates the routing-sensitive header bytes
    /// (magic, version, kind, rank, payload_len) together with the
    /// payload, keyed by the session salt.
    pub fn write_to<W: Write>(&self, w: &mut W, salt: &SessionSalt) -> Result<()> {
        let (kind, rank, payload) = self.parts()?;
        let payload_len = u32::try_from(payload.len()).map_err(|_| {
            TensorError::new(&format!(
                "relay_mux: payload too large: {} bytes (max {})",
                payload.len(),
                u32::MAX
            ))
        })?;
        let mut hdr = [0u8; MUX_HEADER_LEN];
        hdr[0..4].copy_from_slice(&MUX_RECORD_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&MUX_PROTOCOL_VERSION.to_le_bytes());
        hdr[8] = kind;
        hdr[9..13].copy_from_slice(&rank.to_le_bytes());
        hdr[13..17].copy_from_slice(&payload_len.to_le_bytes());
        // auth_tag over header[0..17] (everything but the tag slot) plus
        // the payload bytes.
        let auth_tag = hmac_sha256_64_2(salt, &hdr[0..17], &payload);
        hdr[17..25].copy_from_slice(&auth_tag.to_le_bytes());
        match self {
            // Data-plane frames (host fold up, consensus broadcast down)
            // are model-sized: the atomic [hdr‖payload] copy below would
            // be a whole extra model image per ship, and a payload that
            // large never lands atomically on the wire anyway. Split
            // writes are safe here because every mux reader commits to
            // the full record once the first header byte arrives
            // ([`Self::try_read_from`]'s idle gate + `fill_committed`),
            // so a mid-record read timeout can no longer desync the
            // stream.
            MuxRecord::HostFrame { .. } | MuxRecord::Broadcast { .. } => {
                w.write_all(&hdr).map_err(|e| {
                    TensorError::new(&format!("relay_mux: record header write failed: {e}"))
                })?;
                w.write_all(&payload).map_err(|e| {
                    TensorError::new(&format!("relay_mux: record write failed: {e}"))
                })?;
            }
            // Small control-plane records keep the single atomic write:
            // a reader on a timeout'd socket must never see a header
            // without its payload. Two separate write_all calls open a
            // window where the writer is preempted mid-frame and the
            // reader's read_exact(payload) times out having consumed
            // partial bytes, desyncing the stream — the reader-side
            // commit protects against it too, but on loopback one
            // buffer → one write → the frame lands atomically, and
            // these frames are bytes-cheap to copy.
            MuxRecord::Data { .. } | MuxRecord::Control(_) => {
                let mut frame = Vec::with_capacity(MUX_HEADER_LEN + payload.len());
                frame.extend_from_slice(&hdr);
                frame.extend_from_slice(&payload);
                w.write_all(&frame).map_err(|e| {
                    TensorError::new(&format!("relay_mux: record write failed: {e}"))
                })?;
            }
        }
        Ok(())
    }

    /// Parse a record, validating magic + version + `auth_tag`. Returns
    /// `Ok(None)` on clean EOF (peer closed the connection).
    pub fn read_from<R: Read>(r: &mut R, salt: &SessionSalt) -> Result<Option<Self>> {
        let mut hdr = [0u8; MUX_HEADER_LEN];
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset
                ) =>
            {
                return Ok(None);
            }
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "relay_mux: header read failed: {e}"
                )));
            }
        }
        Self::finish_read(hdr, r, salt).map(Some)
    }

    /// Like [`Self::read_from`] but distinguishes "no data right now"
    /// (read timeout / non-blocking) from clean EOF and from wire errors,
    /// for poll loops that re-check a shutdown flag on idle ticks. Once a
    /// header byte is consumed the method commits to the full record.
    pub fn try_read_from<R: Read>(r: &mut R, salt: &SessionSalt) -> Result<MuxRead> {
        let mut hdr = [0u8; MUX_HEADER_LEN];
        // Idle gate: read only the FIRST byte under the socket's read
        // timeout. WouldBlock/TimedOut here means no frame is in progress,
        // so the caller can poll its shutdown flag and try again.
        match read_idle_gate(r)? {
            IdleGate::Idle => return Ok(MuxRead::WouldBlock),
            IdleGate::Eof => return Ok(MuxRead::Eof),
            IdleGate::Byte(b) => hdr[0] = b,
        }
        // Committed: a frame is in flight. Read the rest of the header and
        // the payload ignoring read timeouts — a partial read must never be
        // abandoned (that desyncs the stream).
        fill_committed(r, &mut hdr[1..])?;
        Self::finish_read(hdr, r, salt).map(MuxRead::Record)
    }

    fn finish_read<R: Read>(hdr: [u8; MUX_HEADER_LEN], r: &mut R, salt: &SessionSalt) -> Result<Self> {
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != MUX_RECORD_MAGIC {
            return Err(TensorError::new(&format!(
                "relay_mux: record magic 0x{magic:08x} != 0x{MUX_RECORD_MAGIC:08x}"
            )));
        }
        let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if version != MUX_PROTOCOL_VERSION {
            return Err(TensorError::new(&format!(
                "relay_mux: record version {version} != {MUX_PROTOCOL_VERSION}"
            )));
        }
        let kind = hdr[8];
        let rank = u32::from_le_bytes(hdr[9..13].try_into().unwrap());
        let payload_len = u32::from_le_bytes(hdr[13..17].try_into().unwrap()) as usize;
        let auth_tag = u64::from_le_bytes(hdr[17..25].try_into().unwrap());
        let ceiling = frame_ceiling();
        if payload_len > ceiling {
            return Err(TensorError::new(&format!(
                "relay_mux: record payload_len {payload_len} exceeds the frame \
                 ceiling {ceiling} (kind=0x{kind:02x}, rank={rank}); corrupt or \
                 hostile peer, or a model that has outgrown the frame ceiling"
            )));
        }

        // Committed fill: the header is already in hand, so the payload
        // must complete even if a read timeout fires mid-frame (a partial
        // read abandoned here would desync the stream). Incremental
        // allocation: the length is unauthenticated until the MAC check
        // below, so never trust it for one big up-front allocation.
        let payload = fill_committed_incremental(r, payload_len)?;

        let actual = hmac_sha256_64_2(salt, &hdr[0..17], &payload);
        if actual != auth_tag {
            return Err(TensorError::new(&format!(
                "relay_mux: HMAC verification failed (computed 0x{actual:016x}, \
                 wire carried 0x{auth_tag:016x}); session salt disagreement, \
                 tampered record, or corruption (kind=0x{kind:02x}, rank={rank}, \
                 len={payload_len})"
            )));
        }

        match kind {
            REC_DATA => Ok(MuxRecord::Data { rank, payload }),
            REC_CONTROL => Ok(MuxRecord::Control(decode(&payload)?)),
            REC_HOST_FRAME => Ok(MuxRecord::HostFrame { payload }),
            REC_BROADCAST => Ok(MuxRecord::Broadcast { payload }),
            other => Err(TensorError::new(&format!(
                "relay_mux: unknown record kind 0x{other:02x}"
            ))),
        }
    }
}

/// Outcome of a single [`MuxRecord::try_read_from`] call.
#[derive(Debug)]
pub enum MuxRead {
    /// A record was decoded and HMAC-verified.
    Record(MuxRecord),
    /// No record available within the reader's timeout window. Keep
    /// polling.
    WouldBlock,
    /// Peer closed the connection cleanly. No more records will arrive.
    Eof,
}

// ---------------------------------------------------------------------------
// Length-framed opaque blobs (rank ↔ relay loopback leg)
// ---------------------------------------------------------------------------

/// Write an opaque frame blob length-delimited as `[u32 len][bytes]`.
///
/// Used on the rank ↔ relay loopback leg, where the blob is the existing
/// [`RoundFrame`] / [`ControlFrame`] bytes (already HMAC-authed
/// end-to-end). The length prefix lets the relay forward opaque bytes
/// without parsing the frame.
///
/// [`RoundFrame`]: crate::distributed::controller::RoundFrame
/// [`ControlFrame`]: crate::distributed::wire::ControlFrame
pub fn write_len_framed<W: Write>(w: &mut W, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        TensorError::new(&format!(
            "relay_mux: len-framed blob too large: {} bytes (max {})",
            bytes.len(),
            u32::MAX
        ))
    })?;
    // Single atomic write (prefix + body) so a reader on a timeout'd
    // socket never sees the length prefix without its body — see the
    // rationale in `MuxRecord::write_to`.
    let mut framed = Vec::with_capacity(4 + bytes.len());
    framed.extend_from_slice(&len.to_le_bytes());
    framed.extend_from_slice(bytes);
    w.write_all(&framed)
        .map_err(|e| TensorError::new(&format!("relay_mux: len-framed write failed: {e}")))?;
    Ok(())
}

/// Write ONLY the 4-byte length prefix of a len-framed blob, for senders
/// that stream the body directly to the socket instead of materializing
/// it ([`crate::distributed::cpu_reduce`]'s model-frame path — the body
/// is hundreds of MB there, and this is what lets it never exist as a
/// contiguous buffer on the sender).
///
/// Splitting prefix and body across write calls is safe against the
/// reader-desync class that forced [`write_len_framed`]'s single atomic
/// write: [`try_read_len_framed`]'s idle gate only applies to the FIRST
/// byte — once that byte lands the reader commits and reads through
/// timeouts ([`fill_committed`]-style) until the body completes, exactly
/// as it must for large bodies that never arrive atomically anyway.
/// Keep small poll-loop frames on [`write_len_framed`].
///
/// Validates `len` against u32 (the prefix width) and the frame ceiling
/// — the reader enforces the ceiling anyway, but failing HERE names the
/// sender instead of tearing an opaque stream down at the relay.
pub fn write_len_prefix<W: Write>(w: &mut W, len: u64) -> Result<()> {
    let ceiling = frame_ceiling();
    if len > ceiling as u64 {
        return Err(TensorError::new(&format!(
            "relay_mux: len-framed blob length {len} exceeds the frame ceiling \
             {ceiling}; model has outgrown the frame ceiling (see \
             wire::frame_ceiling)"
        )));
    }
    let len = u32::try_from(len).map_err(|_| {
        TensorError::new(&format!(
            "relay_mux: len-framed blob too large: {len} bytes (max {})",
            u32::MAX
        ))
    })?;
    w.write_all(&len.to_le_bytes())
        .map_err(|e| TensorError::new(&format!("relay_mux: len prefix write failed: {e}")))
}

/// Read ONLY the 4-byte length prefix of a len-framed blob (`Ok(None)`
/// on clean EOF before it), for receivers that parse the body straight
/// off the stream instead of buffering it — the read-side companion of
/// [`write_len_prefix`], with the same rationale: the model-frame body
/// is hundreds of MB, and this is what lets it never exist as a
/// contiguous buffer on the receiver. Callers should bound the body
/// parse with [`Read::take`]`(len)` and verify the parser consumed
/// exactly `len` bytes, so a prefix/body disagreement surfaces as a
/// loud named error instead of a desynced stream.
pub fn read_len_prefix<R: Read>(r: &mut R) -> Result<Option<usize>> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e)
            if matches!(
                e.kind(),
                ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset
            ) =>
        {
            return Ok(None);
        }
        Err(e) => {
            return Err(TensorError::new(&format!(
                "relay_mux: len prefix read failed: {e}"
            )));
        }
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    let ceiling = frame_ceiling();
    if len > ceiling {
        return Err(TensorError::new(&format!(
            "relay_mux: len-framed blob length {len} exceeds the frame ceiling \
             {ceiling}; corrupt or hostile peer"
        )));
    }
    Ok(Some(len))
}

/// Read a length-delimited opaque blob. Returns `Ok(None)` on clean EOF
/// (peer closed before the next length prefix).
///
/// No production caller since the A4b streaming reads (the rank leg
/// parses bodies straight off the stream via [`read_len_prefix`], the
/// relay poll loops use [`try_read_len_framed`]); retained as the
/// materialized reader the relay/mux test simulators drive rank sides
/// with.
#[cfg_attr(not(test), allow(dead_code))]
pub fn read_len_framed<R: Read>(r: &mut R) -> Result<Option<Vec<u8>>> {
    let Some(len) = read_len_prefix(r)? else {
        return Ok(None);
    };
    let body = crate::distributed::wire::read_exact_incremental(r, len)
        .map_err(|e| TensorError::new(&format!("relay_mux: len-framed body read failed: {e}")))?;
    Ok(Some(body))
}

/// Like [`read_len_framed`] but distinguishes idle (read timeout /
/// non-blocking) from clean EOF, for poll loops. Once the length prefix
/// is consumed the method commits to reading the full body.
pub fn try_read_len_framed<R: Read>(r: &mut R) -> Result<LenFramedRead> {
    let mut len_buf = [0u8; 4];
    // Idle gate: only the first prefix byte is read under the timeout.
    match read_idle_gate(r)? {
        IdleGate::Idle => return Ok(LenFramedRead::WouldBlock),
        IdleGate::Eof => return Ok(LenFramedRead::Eof),
        IdleGate::Byte(b) => len_buf[0] = b,
    }
    // Committed: finish the prefix + body ignoring read timeouts.
    fill_committed(r, &mut len_buf[1..])?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let ceiling = frame_ceiling();
    if len > ceiling {
        return Err(TensorError::new(&format!(
            "relay_mux: len-framed blob length {len} exceeds the frame ceiling \
             {ceiling}; corrupt or hostile peer"
        )));
    }
    let body = fill_committed_incremental(r, len)?;
    Ok(LenFramedRead::Blob(body))
}

/// Outcome of a single [`try_read_len_framed`] call.
#[derive(Debug)]
pub enum LenFramedRead {
    /// A complete blob was read.
    Blob(Vec<u8>),
    /// No data within the reader's timeout window. Keep polling.
    WouldBlock,
    /// Peer closed the stream cleanly.
    Eof,
}

// ---------------------------------------------------------------------------
// Frame-atomic reads over a timeout'd socket
// ---------------------------------------------------------------------------

/// Outcome of reading the first byte of a (possible) frame under the
/// socket's read timeout.
enum IdleGate {
    /// No data within the timeout window — no frame in progress.
    Idle,
    /// Peer closed cleanly before any byte of a new frame.
    Eof,
    /// First byte of a frame; the reader is now committed.
    Byte(u8),
}

/// Read exactly one byte under the socket's read timeout, mapping
/// timeout/non-blocking to [`IdleGate::Idle`] and clean close to
/// [`IdleGate::Eof`]. This is the ONLY timeout-sensitive read in a frame;
/// everything after the first byte is read committed (see
/// [`fill_committed`]).
fn read_idle_gate<R: Read>(r: &mut R) -> Result<IdleGate> {
    let mut b = [0u8; 1];
    loop {
        match r.read(&mut b) {
            Ok(0) => return Ok(IdleGate::Eof),
            Ok(_) => return Ok(IdleGate::Byte(b[0])),
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e)
                if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) =>
            {
                return Ok(IdleGate::Idle);
            }
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset
                ) =>
            {
                return Ok(IdleGate::Eof);
            }
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "relay_mux: idle-gate read failed: {e}"
                )));
            }
        }
    }
}

/// Fill `buf` completely, treating WouldBlock / TimedOut / Interrupted as
/// "keep waiting" — the caller is already committed to a frame in flight,
/// so a read timeout must NOT abandon the partial frame (that desyncs the
/// stream; the single hardest bug in this transport). The socket's read
/// timeout merely paces the retry loop. A clean close mid-frame
/// (`read` → 0) is a hard error: a peer that vanished mid-frame.
/// Wall-clock budget for a committed (mid-frame) read to make ANY
/// progress. A peer that vanishes without FIN/RST (host power loss,
/// network partition — no TCP keepalive is configured) leaves the
/// socket in eternal WouldBlock; without this deadline the reader
/// thread wedges forever and `RelayChannel::shutdown`/`Drop` then hang
/// in `join()`. Mid-frame + no bytes for this long = the peer is gone;
/// erroring out is the correct semantic. Generous: any live peer
/// trickles at least one byte well within it.
const COMMITTED_READ_STARVATION_SECS: u64 = 60;

fn fill_committed<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    let mut filled = 0;
    let mut last_progress = std::time::Instant::now();
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(TensorError::new(
                    "relay_mux: peer closed mid-frame (committed read hit EOF)",
                ));
            }
            Ok(n) => {
                filled += n;
                last_progress = std::time::Instant::now();
            }
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                ) =>
            {
                // Frame still arriving; the socket's read timeout paces
                // this loop (no busy-spin). The starvation deadline only
                // fires when NO bytes arrive at all for the whole budget.
                if last_progress.elapsed().as_secs() >= COMMITTED_READ_STARVATION_SECS {
                    return Err(TensorError::new(&format!(
                        "relay_mux: committed read starved mid-frame for \
                         {COMMITTED_READ_STARVATION_SECS}s ({filled}/{} bytes); \
                         peer presumed gone",
                        buf.len(),
                    )));
                }
                continue;
            }
            Err(e) => {
                return Err(TensorError::new(&format!(
                    "relay_mux: committed read failed: {e}"
                )));
            }
        }
    }
    Ok(())
}

/// Committed read of `len` bytes with INCREMENTAL allocation: the buffer
/// grows in chunks as bytes actually arrive, so an unauthenticated
/// (garbage/hostile) length field can only make us allocate what the
/// peer really sends — never a multi-GiB up-front `vec![0; len]`.
fn fill_committed_incremental<R: Read>(r: &mut R, len: usize) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    while buf.len() < len {
        let chunk = (len - buf.len())
            .min(crate::distributed::wire::READ_CHUNK);
        let old_len = buf.len();
        buf.resize(old_len + chunk, 0);
        fill_committed(r, &mut buf[old_len..])?;
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// HMAC over two byte regions (header-without-tag || payload)
// ---------------------------------------------------------------------------

/// HMAC-SHA256-64 over the concatenation of two byte regions, keyed by
/// `salt`. Lets the mux header authenticate its routing fields together
/// with the payload without an intermediate allocation in the common
/// (empty/borrowed payload) case.
fn hmac_sha256_64_2(salt: &SessionSalt, a: &[u8], b: &[u8]) -> u64 {
    // hmac_sha256_64 takes a single slice; build the MAC input once.
    // Header is 17 bytes; payloads here are control blobs or already-
    // framed opaque data, so a single concat is acceptable and keeps the
    // auth helper identical to the rest of the wire layer.
    let mut buf = Vec::with_capacity(a.len() + b.len());
    buf.extend_from_slice(a);
    buf.extend_from_slice(b);
    hmac_sha256_64(salt, &buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "mux_tests.rs"]
mod tests;
