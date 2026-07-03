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
//!   HMAC-authed end-to-end, so the loopback prefix carries no auth of
//!   its own. The relay never parses the blob; it forwards opaque bytes.
//!
//! - **relay ↔ controller (network):** each blob is wrapped in a
//!   [`MuxRecord`] that tags it with its originating `rank` so the
//!   single per-host connection can carry every local rank's frames. The
//!   mux header (including the routing-sensitive `rank` field) is
//!   HMAC-authed with the session salt, mirroring the rest of the
//!   cluster wire protocol — a flipped `rank` would misroute, so the tag
//!   must be tamper-evident. The wrapped payload keeps its own
//!   end-to-end HMAC, so the relay cannot tamper with tensor/control
//!   bytes undetected.
//!
//! # Pure transport (v1)
//!
//! The relay FORWARDS; it does not aggregate. Rank R's full reduce
//! buffer crosses the wire untouched and the controller still does the
//! flat sum over all ranks. The v1 win is connection count (N → 1 per
//! host) and "address a node, not N GPUs", NOT bytes. Sum-and-count (the
//! N×-fewer-bytes wire reduction) is a separate later layer that will
//! live in the relay and is the only place that would parse the payload.
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
    },
    /// Controller → relay handshake acknowledgement.
    HelloAck,
    /// Relay → controller: a local rank's loopback connection closed
    /// (clean exit or crash — the relay does not distinguish in v1). The
    /// controller declares the rank dead so its reduce barrier releases
    /// instead of waiting forever. Host death is signalled implicitly by
    /// the whole per-host connection EOFing, not by this message.
    RankExit { rank: u32 },
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
    /// [`crate::distributed::controller::RoundFrame`] (data channel) or
    /// [`crate::distributed::wire::ControlFrame`] (control channel)
    /// bytes, never parsed by the relay.
    Data { rank: u32, payload: Vec<u8> },
    /// A relay-level control signal.
    Control(RelayControlMsg),
}

/// Hard ceiling on any length-prefixed payload (mux records and
/// len-framed blobs). The length field is UNAUTHENTICATED until the
/// trailing MAC verifies, so a hostile or corrupt peer can claim up to
/// `u32::MAX` and force the reader to buffer it before rejection —
/// incremental allocation makes the attacker pay the bandwidth, this
/// cap bounds the memory. Sized with generous headroom over a full
/// model params `RoundFrame` on the data channel; a legitimate model
/// outgrowing it fails loudly here, naming this constant. A follow-up
/// derives the exact bound from the model via the rendezvous handshake
/// instead of a constant.
pub(crate) const MAX_MUX_PAYLOAD: usize = 1 << 30; // 1 GiB

impl MuxRecord {
    /// Tag an opaque frame blob with its originating rank.
    pub fn data(rank: u32, payload: Vec<u8>) -> Self {
        MuxRecord::Data { rank, payload }
    }

    /// Wrap a relay-level control signal.
    pub fn control(msg: RelayControlMsg) -> Self {
        MuxRecord::Control(msg)
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
        // Single atomic write: a reader on a timeout'd socket must never
        // see a header without its payload. Two separate write_all calls
        // open a window where the writer is preempted mid-frame and the
        // reader's read_exact(payload) times out having consumed partial
        // bytes, desyncing the stream. One buffer → one write → the frame
        // lands (and on loopback becomes readable) atomically.
        let mut frame = Vec::with_capacity(MUX_HEADER_LEN + payload.len());
        frame.extend_from_slice(&hdr);
        frame.extend_from_slice(&payload);
        w.write_all(&frame)
            .map_err(|e| TensorError::new(&format!("relay_mux: record write failed: {e}")))?;
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
        if payload_len > MAX_MUX_PAYLOAD {
            return Err(TensorError::new(&format!(
                "relay_mux: record payload_len {payload_len} exceeds MAX_MUX_PAYLOAD \
                 {MAX_MUX_PAYLOAD} (kind=0x{kind:02x}, rank={rank}); corrupt or \
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

/// Read a length-delimited opaque blob. Returns `Ok(None)` on clean EOF
/// (peer closed before the next length prefix).
pub fn read_len_framed<R: Read>(r: &mut R) -> Result<Option<Vec<u8>>> {
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
    if len > MAX_MUX_PAYLOAD {
        return Err(TensorError::new(&format!(
            "relay_mux: len-framed blob length {len} exceeds MAX_MUX_PAYLOAD \
             {MAX_MUX_PAYLOAD}; corrupt or hostile peer"
        )));
    }
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
    if len > MAX_MUX_PAYLOAD {
        return Err(TensorError::new(&format!(
            "relay_mux: len-framed blob length {len} exceeds MAX_MUX_PAYLOAD \
             {MAX_MUX_PAYLOAD}; corrupt or hostile peer"
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
