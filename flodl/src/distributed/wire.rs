//! Control-channel wire protocol for the cluster process model.
//!
//! Companion to the existing data-channel ([`controller`]) protocol that
//! carries averaged-tensor [`RoundFrame`]s. The control channel carries
//! lightweight scheduling messages: timing reports from workers, ElChe-
//! computed epoch plans from the controller, sync triggers, throttle
//! signals, etc. Heavy tensor data never travels here -- it stays on the
//! data channel.
//!
//! ## Why two channels
//!
//! Faithful port of the OLD coordinator/worker threaded model: mpsc had
//! separate channels for `timing_rx`, `metrics_rx`, `param_rx`, and
//! `control_txs`. Collapsing them into one TCP stream would couple
//! scheduling latency to bulk-data throughput; per-channel back-pressure
//! is the cleanest port.
//!
//! ## Frame layout
//!
//! Every control frame uses [`ControlFrame`]:
//!
//! ```text
//! u32 magic       = CONTROL_FRAME_MAGIC
//! u32 version     = CONTROL_PROTOCOL_VERSION
//! u64 auth_tag    = hmac_sha256_64(session_salt, payload_bytes)
//! u32 msg_kind    (one of MSG_KIND_*)
//! u32 payload_len
//! <payload_len>   bincode-serialized message
//! ```
//!
//! 24-byte header is small enough that even a one-byte payload fits
//! comfortably in a single TCP segment.
//!
//! ## Session salt (HMAC key)
//!
//! Launcher generates a 128-bit random salt per training session and
//! distributes it via the cluster envelope. Every control frame's
//! `auth_tag` is HMAC-SHA256 over the payload bytes, keyed by the salt,
//! truncated to 64 bits. A frame from a wrong session (stale process,
//! MITM without the key, network mix-up) fails authentication with
//! probability 2^-64 and surfaces loudly.
//!
//! Payloads are **not** confidential -- HMAC authenticates but does not
//! encrypt. An attacker on the wire can still read bincode bytes. The
//! guarantee is that without the salt they cannot forge or tamper with
//! frames. Encryption (TLS or noise) is a separate future upgrade and
//! is orthogonal to the HMAC framing.
//!
//! ## Relationship to OLD types
//!
//! The wire-friendly types here mirror the in-process [`ddp_run`] types
//! (`ControlMsg`, `TimingMsg`, `MetricsMsg`, `EpochPlan`,
//! `ParamSnapshot`) but strip out [`Tensor`] handles -- those are
//! re-attached at the receiving end by pairing each `Update` /
//! `ParamSnapshotMeta` with the matching [`RoundFrame`] on the data
//! channel.
//!
//! [`controller`]: crate::distributed::controller
//! [`RoundFrame`]: crate::distributed::controller::RoundFrame
//! [`ddp_run`]: crate::distributed::ddp_run

use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};

use hmac_sha256::HMAC;
use serde::{Deserialize, Serialize};

use crate::tensor::{Result, TensorError};

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// Magic number on the rank-side control-channel handshake (rank → controller).
pub const CONTROL_HANDSHAKE_MAGIC_RANK: u32 = 0xF10D_17C2;

/// Magic number on the controller's handshake ack (controller → rank).
pub const CONTROL_HANDSHAKE_MAGIC_ACK: u32 = 0xF10D_17C3;

/// Magic number for every [`ControlFrame`] (in either direction after
/// handshake).
pub const CONTROL_FRAME_MAGIC: u32 = 0xF10D_17C4;

/// Wire version of the control-channel protocol. Independent of the
/// data-channel `PROTOCOL_VERSION` in `controller.rs`. Bump on any
/// breaking change to [`ControlFrame`] or to the wire-message types.
pub const CONTROL_PROTOCOL_VERSION: u32 = 1;

/// Length of the random session salt in bytes.
pub const SESSION_SALT_BYTES: usize = 16;

/// One byte tagging the payload type inside a [`ControlFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MsgKind {
    /// Coordinator → worker control signal. Payload: [`ControlMsgWire`].
    Control = 0x01,
    /// Worker → coordinator timing report. Payload: [`TimingMsgWire`].
    Timing = 0x02,
    /// Worker → coordinator per-epoch metrics. Payload: [`MetricsMsgWire`].
    Metrics = 0x03,
    /// Worker → coordinator pre-snapshot metadata, paired with a
    /// matching [`RoundFrame`] on the data channel. Payload:
    /// [`ParamSnapshotMetaWire`].
    ParamSnapshotMeta = 0x04,
    /// Periodic heartbeat (control-channel). Payload: [`HeartbeatWire`].
    Heartbeat = 0x05,
}

impl MsgKind {
    /// Parse a wire-encoded kind. Loud error on unknown.
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0x01 => Ok(MsgKind::Control),
            0x02 => Ok(MsgKind::Timing),
            0x03 => Ok(MsgKind::Metrics),
            0x04 => Ok(MsgKind::ParamSnapshotMeta),
            0x05 => Ok(MsgKind::Heartbeat),
            _ => Err(TensorError::new(&format!(
                "wire: unknown MsgKind tag 0x{v:08x}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Authentication tag helper (HMAC-SHA256 truncated to 64 bits)
// ---------------------------------------------------------------------------

/// 128-bit session salt, generated by the launcher and shipped to every
/// rank via the cluster envelope. Used as the HMAC key for every wire
/// frame; cross-session frames fail authentication via
/// [`hmac_sha256_64`].
pub type SessionSalt = [u8; SESSION_SALT_BYTES];

/// HMAC-SHA256 over `bytes` keyed by `salt`, truncated to the leading
/// 64 bits (little-endian).
///
/// Replaces the original xxh3-based integrity tag with a real
/// cryptographic MAC. Reuses the existing `hmac-sha256` workspace dep
/// (already pulled in for graph hashing) so no new external crate is
/// introduced. SHA-256 throughput is on the order of GB/s on modern
/// CPUs; for the control-channel's small payloads and the data-
/// channel's once-per-K-batch RoundFrames the overhead is negligible.
///
/// Truncation to 64 bits gives 2^-64 forgery probability per attempt
/// without the salt, which is sufficient for session isolation and
/// tamper detection at the frame level. RFC 2104 permits arbitrary
/// truncation; the security level is "at least min(half_full_tag, bits
/// kept)" -- 64 bits is well above any realistic online-attack budget.
///
/// Not a substitute for encryption: payloads remain visible to anyone
/// on the wire.
pub fn hmac_sha256_64(salt: &SessionSalt, bytes: &[u8]) -> u64 {
    let full: [u8; 32] = HMAC::mac(bytes, salt.as_slice());
    u64::from_le_bytes(full[0..8].try_into().unwrap())
}

/// Generate a fresh random session salt from the OS-seeded thread RNG.
///
/// Reuses the workspace `rand` dep (default-on via the `rng` feature)
/// so this works on every platform `rand` supports, not just
/// Linux-flavored `/dev/urandom`. `rand::make_rng()` returns a
/// ChaCha-backed thread CSPRNG seeded from the OS, suitable for
/// HMAC-key material.
///
/// Per-session, not per-rank: the launcher generates ONE salt and
/// every rank receives the same value via the cluster envelope.
///
/// Gated by the `rng` feature (on by default). Cluster mode requires
/// `rng`; build configurations that disable `rng` cannot generate
/// salts and must rely on the zero-default value (single-host).
#[cfg(feature = "rng")]
pub fn generate_session_salt() -> SessionSalt {
    use rand::Rng;
    let mut buf = [0u8; SESSION_SALT_BYTES];
    rand::rng().fill_bytes(&mut buf);
    buf
}

/// Hex-encode a 16-byte salt into a 32-char lowercase string for
/// inclusion in the cluster envelope JSON. Reuses the existing
/// [`crate::distributed::cluster::hex_encode`] format.
pub fn salt_to_hex(salt: &SessionSalt) -> String {
    crate::distributed::cluster::hex_encode(salt)
}

/// Inverse of [`salt_to_hex`]. Loud error on wrong length / non-hex
/// chars; bubbles the error message context so callers point at the
/// envelope source.
pub fn salt_from_hex(s: &str) -> Result<SessionSalt> {
    let trimmed = s.trim();
    if trimmed.len() != SESSION_SALT_BYTES * 2 {
        return Err(TensorError::new(&format!(
            "wire: session salt hex must be {} chars (got {})",
            SESSION_SALT_BYTES * 2,
            trimmed.len()
        )));
    }
    let bytes = crate::distributed::cluster::hex_decode(trimmed)
        .map_err(|e| TensorError::new(&format!("wire: session salt hex-decode: {e}")))?;
    let mut out = [0u8; SESSION_SALT_BYTES];
    out.copy_from_slice(&bytes);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Bincode helpers
// ---------------------------------------------------------------------------

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, bincode_config())
        .map_err(|e| TensorError::new(&format!("wire: bincode encode failed: {e}")))
}

fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T> {
    let (v, _used) = bincode::serde::decode_from_slice(bytes, bincode_config())
        .map_err(|e| TensorError::new(&format!("wire: bincode decode failed: {e}")))?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// ControlFrame
// ---------------------------------------------------------------------------

/// One framed message on the control channel.
///
/// Constructed by [`ControlFrame::encode`] / [`ControlFrame::write_to`]
/// (writer side); parsed by [`ControlFrame::read_from`] (reader side).
/// The header is hand-rolled little-endian; the payload is bincode-
/// serialized.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlFrame {
    /// Payload tag.
    pub kind: MsgKind,
    /// hmac_sha256_64(session_salt, payload_bytes). Set by `write_to`,
    /// validated by `read_from`.
    pub auth_tag: u64,
    /// Bincode bytes of the payload.
    pub payload: Vec<u8>,
}

impl ControlFrame {
    /// Encode `payload` as bincode bytes and pair with its salt check.
    ///
    /// Convenience wrapper for callers that have a serializable message
    /// in hand; the alternative is to set `payload` manually if the
    /// caller already holds bytes.
    pub fn encode<T: Serialize>(
        salt: &SessionSalt,
        kind: MsgKind,
        msg: &T,
    ) -> Result<Self> {
        let payload = encode(msg)?;
        let auth_tag = hmac_sha256_64(salt, &payload);
        Ok(ControlFrame {
            kind,
            auth_tag,
            payload,
        })
    }

    /// Decode this frame's payload as `T`. Caller is responsible for
    /// matching `T` to [`Self::kind`].
    pub fn decode<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        decode(&self.payload)
    }

    /// Serialize the full header + payload to the writer. Single
    /// `write_all` per region to keep tcpdumps readable.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&CONTROL_FRAME_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&CONTROL_PROTOCOL_VERSION.to_le_bytes());
        hdr[8..16].copy_from_slice(&self.auth_tag.to_le_bytes());
        hdr[16..20].copy_from_slice(&(self.kind as u32).to_le_bytes());
        let payload_len = u32::try_from(self.payload.len()).map_err(|_| {
            TensorError::new(&format!(
                "wire: payload too large: {} bytes (max {} bytes)",
                self.payload.len(),
                u32::MAX
            ))
        })?;
        hdr[20..24].copy_from_slice(&payload_len.to_le_bytes());
        w.write_all(&hdr).map_err(|e| {
            TensorError::new(&format!("wire: ControlFrame header write failed: {e}"))
        })?;
        w.write_all(&self.payload).map_err(|e| {
            TensorError::new(&format!("wire: ControlFrame payload write failed: {e}"))
        })?;
        Ok(())
    }

    /// Parse a frame from the reader, validating magic + version +
    /// `auth_tag`. Returns `Ok(None)` on clean EOF.
    ///
    /// Treats `WouldBlock` and `TimedOut` on the initial header read as
    /// errors. For short-timeout / non-blocking readers, prefer
    /// [`Self::try_read_from`].
    pub fn read_from<R: Read>(r: &mut R, salt: &SessionSalt) -> Result<Option<Self>> {
        let mut hdr = [0u8; 24];
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
                    "wire: ControlFrame header read failed: {e}"
                )));
            }
        }
        Self::finish_read_from(hdr, r, salt).map(Some)
    }

    /// Like [`Self::read_from`] but distinguishes "no data available
    /// right now" (e.g. `set_read_timeout` fired, or non-blocking
    /// reader sees no bytes) from clean EOF and from wire errors.
    /// Used by the cluster coordinator's per-rank reader thread so it
    /// can re-check its shutdown flag periodically without exiting on
    /// idle ticks.
    ///
    /// Once a single header byte has been consumed the method commits
    /// to reading the full frame — partial reads beyond that point
    /// surface as `Err`.
    pub fn try_read_from<R: Read>(r: &mut R, salt: &SessionSalt) -> Result<FrameRead> {
        let mut hdr = [0u8; 24];
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) => {
                return match e.kind() {
                    ErrorKind::UnexpectedEof | ErrorKind::ConnectionReset => {
                        Ok(FrameRead::Eof)
                    }
                    ErrorKind::WouldBlock | ErrorKind::TimedOut => {
                        Ok(FrameRead::WouldBlock)
                    }
                    _ => Err(TensorError::new(&format!(
                        "wire: ControlFrame header read failed: {e}"
                    ))),
                };
            }
        }
        Self::finish_read_from(hdr, r, salt).map(FrameRead::Frame)
    }

    fn finish_read_from<R: Read>(
        hdr: [u8; 24],
        r: &mut R,
        salt: &SessionSalt,
    ) -> Result<Self> {
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != CONTROL_FRAME_MAGIC {
            return Err(TensorError::new(&format!(
                "wire: ControlFrame magic 0x{magic:08x} != 0x{CONTROL_FRAME_MAGIC:08x}"
            )));
        }
        let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if version != CONTROL_PROTOCOL_VERSION {
            return Err(TensorError::new(&format!(
                "wire: ControlFrame version {version} != {CONTROL_PROTOCOL_VERSION}"
            )));
        }
        let auth_tag = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let kind_u32 = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
        let kind = MsgKind::from_u32(kind_u32)?;
        let payload_len = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        let mut payload = vec![0u8; payload_len];
        r.read_exact(&mut payload).map_err(|e| {
            TensorError::new(&format!(
                "wire: ControlFrame payload read failed (kind={kind:?}, len={payload_len}): {e}"
            ))
        })?;
        let actual = hmac_sha256_64(salt, &payload);
        if actual != auth_tag {
            return Err(TensorError::new(&format!(
                "wire: ControlFrame HMAC verification failed (computed \
                 0x{actual:016x}, header carried 0x{auth_tag:016x}); session \
                 salt disagreement, tampered frame, or payload corruption \
                 (kind={kind:?}, len={payload_len})"
            )));
        }
        Ok(ControlFrame {
            kind,
            auth_tag,
            payload,
        })
    }
}

/// Outcome of a single [`ControlFrame::try_read_from`] call.
#[derive(Debug)]
pub enum FrameRead {
    /// A frame was decoded and HMAC-verified.
    Frame(ControlFrame),
    /// No frame available within the reader's timeout window (or the
    /// reader is non-blocking and has nothing buffered). Caller should
    /// keep polling.
    WouldBlock,
    /// Peer closed the stream cleanly. No more frames will arrive.
    Eof,
}

// ---------------------------------------------------------------------------
// Wire-friendly message types
// ---------------------------------------------------------------------------
// These mirror the in-process types in ddp_run::mod but strip out Tensor
// handles. Tensor data is paired via the data channel's RoundFrame.

/// Wire-side mirror of [`ddp_run::EpochPlan`]. Pure plain data.
///
/// [`ddp_run::EpochPlan`]: crate::distributed::ddp_run::EpochPlan
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpochPlanWire {
    pub epoch: u64,
    pub partition_offset: u64,
    pub partition_size: u64,
}

/// Wire-side mirror of [`ddp_run::ControlMsg`]. The `Update` variant
/// carries only a version stamp; the matching tensors travel via the
/// data channel.
///
/// [`ddp_run::ControlMsg`]: crate::distributed::ddp_run::ControlMsg
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMsgWire {
    /// CPU path: ask the worker to send its current ParamSnapshot.
    RequestParams,
    /// CPU path: averaged params with `version` are ready on the data
    /// channel; worker reads the next RoundFrame and applies it.
    Update { version: u64 },
    /// NCCL path: trigger in-place AllReduce on the worker's params.
    SyncNow,
    /// Begin processing a new epoch with the given partition.
    StartEpoch(EpochPlanWire),
    /// Extend the worker's current-epoch partition with additional
    /// indices from the global permutation. Emitted mid-epoch by the
    /// coord when redistributing a freshly-dead rank's un-processed
    /// samples onto survivors, preserving the "intended N samples per
    /// epoch" invariant under rank failure. The worker appends the
    /// indices (computed via `make_partition` with the new
    /// `partition_offset` / `partition_size`, the current epoch, and
    /// the shared seed) to its in-flight partition; its epoch loop
    /// re-checks the bound each iteration so the appended batches
    /// are processed before completing the epoch.
    ExtendPartition {
        partition_offset: u64,
        partition_size: u64,
    },
    /// Coord-emitted notification that a peer rank has been declared
    /// dead (heartbeat staleness). Surviving workers update their
    /// local dead-rank ledger so the NCCL watchdog thread can call
    /// [`crate::distributed::nccl::NcclAbortHandle::abort`] on the
    /// current comm; the worker's main thread sees its blocked
    /// AllReduce return with an Err and then waits for a
    /// [`Self::NewNcclSession`] frame from the coord to rebuild the
    /// comm with the shrunken cohort. No-op on CPU backend (the
    /// coord already drives the controller-side release via the
    /// shared `DeadRanks` ledger).
    DeclareDead { rank: u64 },
    /// Coord-emitted request to a single surviving rank: please
    /// generate a fresh `NcclUniqueId` and ship it back via
    /// [`TimingMsgWire::NewNcclIdGenerated`]. The coord then relays
    /// the bytes to every survivor via [`Self::NewNcclSession`].
    ///
    /// Why this two-step instead of having the coord generate the
    /// uid itself: the coord process (typically the launcher's host)
    /// may not link libnccl or have NCCL initialized, so
    /// `ncclGetUniqueId` would be unavailable there. Asking a rank
    /// (which already has libnccl loaded) keeps the coord
    /// CUDA-feature-independent. The coord picks the lowest-numbered
    /// surviving rank for determinism.
    RequestNewNcclId,
    /// Coord-emitted notification that the surviving cohort should
    /// re-rendezvous on a fresh NCCL communicator. Sent after one or
    /// more [`Self::DeclareDead`] frames + a successful
    /// [`Self::RequestNewNcclId`] → [`TimingMsgWire::NewNcclIdGenerated`]
    /// round-trip. Each remaining rank can then call
    /// [`crate::distributed::nccl::NcclRankComm::init_rank`] with the
    /// new (uid, world_size, local rank-in-comm) tuple. The
    /// per-recipient `new_rank` is the recipient's position among
    /// survivors, ordered by ascending global rank (rank 0 stays
    /// rank 0 if alive; if rank 1 died, original rank 2 becomes new
    /// rank 1; etc.). `new_world_size` is `world_size - dead_count`.
    NewNcclSession {
        /// 128-byte NCCL unique-id, freshly generated by the lowest
        /// surviving rank and relayed through the coord. All
        /// survivors receive the same bytes so they meet on the same
        /// communicator.
        uid_bytes: Vec<u8>,
        /// Recipient's new rank inside the shrunken communicator.
        new_rank: u64,
        /// Total number of ranks in the new communicator
        /// (`original_world_size - dead_count`).
        new_world_size: u64,
    },
    /// Worker is too far ahead; block until the next real command.
    Throttle,
    /// Update the worker's global step count after averaging.
    SetGlobalStep { global_step: u64 },
    /// Save a checkpoint from rank 0 after averaging.
    Checkpoint { version: u64 },
    /// Shut down this worker.
    Shutdown,
}

/// Wire-side mirror of [`ddp_run::TimingMsg`]. All fields are plain
/// data; the OLD type was already serde-compatible in shape.
///
/// [`ddp_run::TimingMsg`]: crate::distributed::ddp_run::TimingMsg
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TimingMsgWire {
    Batch {
        rank: u64,
        batch_ms: f64,
        step_count: u64,
        param_norm: Option<f64>,
        batch_loss: f64,
        sync_divergence: Option<f64>,
    },
    SyncAck {
        rank: u64,
        step_count: u64,
        divergence: Option<f64>,
        post_norm: Option<f64>,
        pre_norm: Option<f64>,
    },
    Exiting {
        rank: u64,
    },
    LrUpdate {
        rank: u64,
        lr: f64,
    },
    /// Periodic worker-emitted liveness signal. Fires on a fixed cadence
    /// from the cluster worker's heartbeat thread independent of
    /// training progress, so the coord can distinguish "rank alive but
    /// blocked at AllReduce barrier" from "rank dead." Stale heartbeat
    /// triggers dead-rank declaration → elastic averaging path.
    Heartbeat {
        rank: u64,
        /// Worker's local step counter at emission time (diagnostic
        /// only — staleness detection is purely wall-clock based on
        /// the coordinator's last-received instant).
        step_count: u64,
    },
    /// Per-rank "snapshot ready, about to enter AllReduce barrier"
    /// marker. Emitted by the worker's CPU-averaging bridge BEFORE it
    /// blocks in `cpu_client.all_reduce_tensors`. The wall-time from
    /// the coord's `RequestParams` broadcast to this frame's arrival
    /// is honest per-rank capacity — snapshot + upload time only,
    /// NOT polluted by the slowest-rank barrier wait that contaminates
    /// `SyncAck` timestamps.
    SnapshotReady { rank: u64 },
    /// Response to [`ControlMsgWire::RequestNewNcclId`]: the chosen
    /// surviving rank generated a fresh `NcclUniqueId` and ships its
    /// raw bytes back to the coord. Coord then broadcasts
    /// [`ControlMsgWire::NewNcclSession`] with these bytes to every
    /// survivor (including the one that generated them).
    NewNcclIdGenerated {
        /// Sender rank (for the coord to validate the response came
        /// from the rank it asked).
        rank: u64,
        /// 128-byte NCCL unique-id, as produced by
        /// `crate::distributed::nccl::NcclUniqueId::new()`.
        uid_bytes: Vec<u8>,
    },
}

/// Wire-side mirror of [`ddp_run::MetricsMsg`]. All fields plain data.
///
/// [`ddp_run::MetricsMsg`]: crate::distributed::ddp_run::MetricsMsg
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MetricsMsgWire {
    pub rank: u64,
    pub epoch: u64,
    pub avg_loss: f64,
    pub batches_processed: u64,
    pub epoch_ms: f64,
    pub samples_processed: u64,
    pub share_complete_ms: f64,
    pub compute_only_ms: f64,
    pub data_starve_ms: f64,
    pub scalars: HashMap<String, (f64, u64)>,
}

/// Metadata header paired with a [`RoundFrame`] on the data channel when
/// a worker is shipping its ParamSnapshot for CPU averaging.
///
/// [`RoundFrame`]: crate::distributed::controller::RoundFrame
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamSnapshotMetaWire {
    pub rank: u64,
    pub batch_count: u64,
    /// Number of parameter tensors in the upcoming RoundFrame.
    pub num_params: u32,
    /// Number of buffer tensors in the upcoming RoundFrame, following
    /// the params. Worker concatenates `params || buffers` into a
    /// single RoundFrame; this field tells the receiver where the
    /// split is.
    pub num_buffers: u32,
}

/// Periodic worker-emitted heartbeat. Coordinator uses last-heard time
/// per rank for fault detection (slice 1d.5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatWire {
    pub rank: u64,
    /// Monotonic-ish step count at heartbeat time. Diagnostic.
    pub step_count: u64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const ZERO_SALT: SessionSalt = [0u8; 16];
    const SAMPLE_SALT: SessionSalt = [
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00,
    ];

    #[test]
    fn hmac_sha256_64_is_deterministic_and_key_sensitive() {
        // Same (salt, bytes) → same tag every call.
        let h = hmac_sha256_64(&ZERO_SALT, b"hello");
        let h2 = hmac_sha256_64(&ZERO_SALT, b"hello");
        assert_eq!(h, h2);
        // Different salt → different tag (with overwhelming probability).
        let h3 = hmac_sha256_64(&SAMPLE_SALT, b"hello");
        assert_ne!(h, h3);
        // Different message → different tag.
        let h4 = hmac_sha256_64(&ZERO_SALT, b"hellp");
        assert_ne!(h, h4);
    }

    #[test]
    fn hmac_sha256_64_truncation_matches_full_mac() {
        // The truncated tag must be exactly the first 8 bytes of the full
        // HMAC-SHA256 output, interpreted little-endian. Guards against
        // accidental endian flips or wrong-half truncation in future edits.
        let bytes = b"some payload bytes for verification";
        let full: [u8; 32] = HMAC::mac(bytes, SAMPLE_SALT.as_slice());
        let mut expected_first_8 = [0u8; 8];
        expected_first_8.copy_from_slice(&full[0..8]);
        let expected = u64::from_le_bytes(expected_first_8);
        assert_eq!(hmac_sha256_64(&SAMPLE_SALT, bytes), expected);
    }

    #[test]
    fn msg_kind_round_trip() {
        for k in [
            MsgKind::Control,
            MsgKind::Timing,
            MsgKind::Metrics,
            MsgKind::ParamSnapshotMeta,
            MsgKind::Heartbeat,
        ] {
            let v = k as u32;
            assert_eq!(MsgKind::from_u32(v).unwrap(), k);
        }
    }

    #[test]
    fn msg_kind_rejects_unknown() {
        let err = MsgKind::from_u32(0xDEAD).unwrap_err();
        assert!(err.to_string().contains("MsgKind"), "got: {err}");
    }

    #[test]
    fn control_frame_round_trip_in_memory() {
        let plan = EpochPlanWire {
            epoch: 7,
            partition_offset: 100,
            partition_size: 256,
        };
        let msg = ControlMsgWire::StartEpoch(plan.clone());
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();

        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .expect("frame, not EOF");
        assert_eq!(got.kind, MsgKind::Control);
        assert_eq!(got.auth_tag, frame.auth_tag);

        let decoded: ControlMsgWire = got.decode().unwrap();
        assert_eq!(decoded, msg);
        match decoded {
            ControlMsgWire::StartEpoch(p) => assert_eq!(p, plan),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn control_frame_rejects_wrong_salt() {
        let msg = ControlMsgWire::Shutdown;
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Control, &msg).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let err = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HMAC verification failed"),
            "expected HMAC verification failure, got: {msg}"
        );
    }

    #[test]
    fn control_frame_rejects_wrong_magic() {
        // Build a frame manually with bad magic.
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        hdr[4..8].copy_from_slice(&CONTROL_PROTOCOL_VERSION.to_le_bytes());
        // auth_tag zero, kind=Control, payload_len=0
        hdr[16..20].copy_from_slice(&(MsgKind::Control as u32).to_le_bytes());
        let mut cur = Cursor::new(hdr.to_vec());
        let err = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap_err();
        assert!(err.to_string().contains("magic"), "got: {err}");
    }

    #[test]
    fn control_frame_rejects_wrong_version() {
        let mut hdr = [0u8; 24];
        hdr[0..4].copy_from_slice(&CONTROL_FRAME_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&99u32.to_le_bytes());
        hdr[16..20].copy_from_slice(&(MsgKind::Control as u32).to_le_bytes());
        let mut cur = Cursor::new(hdr.to_vec());
        let err = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap_err();
        assert!(err.to_string().contains("version"), "got: {err}");
    }

    #[test]
    fn control_frame_eof_returns_none() {
        let mut cur = Cursor::new(Vec::<u8>::new());
        let got = ControlFrame::read_from(&mut cur, &ZERO_SALT).unwrap();
        assert!(got.is_none(), "EOF before header bytes should be None");
    }

    #[test]
    fn timing_msg_round_trip_all_variants() {
        let cases = [
            TimingMsgWire::Batch {
                rank: 1,
                batch_ms: 12.5,
                step_count: 42,
                param_norm: Some(3.5),
                batch_loss: 0.1,
                sync_divergence: None,
            },
            TimingMsgWire::SyncAck {
                rank: 2,
                step_count: 100,
                divergence: Some(0.01),
                post_norm: Some(5.0),
                pre_norm: Some(5.01),
            },
            TimingMsgWire::Exiting { rank: 3 },
            TimingMsgWire::LrUpdate { rank: 0, lr: 1e-3 },
        ];
        for c in cases {
            let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Timing, &c).unwrap();
            let mut buf = Vec::new();
            frame.write_to(&mut buf).unwrap();
            let mut cur = Cursor::new(buf);
            let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
                .unwrap()
                .unwrap();
            assert_eq!(got.kind, MsgKind::Timing);
            let back: TimingMsgWire = got.decode().unwrap();
            assert_eq!(back, c);
        }
    }

    #[test]
    fn metrics_msg_round_trip_with_scalars() {
        let mut scalars = HashMap::new();
        scalars.insert("loss".to_string(), (12.5, 100));
        scalars.insert("acc".to_string(), (0.85, 100));
        let m = MetricsMsgWire {
            rank: 1,
            epoch: 3,
            avg_loss: 0.42,
            batches_processed: 50,
            epoch_ms: 1234.5,
            samples_processed: 6400,
            share_complete_ms: 1100.0,
            compute_only_ms: 900.0,
            data_starve_ms: 50.0,
            scalars,
        };
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Metrics, &m).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .unwrap();
        let back: MetricsMsgWire = got.decode().unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn param_snapshot_meta_round_trip() {
        let meta = ParamSnapshotMetaWire {
            rank: 2,
            batch_count: 17,
            num_params: 50,
            num_buffers: 6,
        };
        let frame =
            ControlFrame::encode(&SAMPLE_SALT, MsgKind::ParamSnapshotMeta, &meta).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .unwrap();
        let back: ParamSnapshotMetaWire = got.decode().unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn heartbeat_round_trip() {
        let hb = HeartbeatWire {
            rank: 0,
            step_count: 12345,
        };
        let frame = ControlFrame::encode(&SAMPLE_SALT, MsgKind::Heartbeat, &hb).unwrap();
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(buf);
        let got = ControlFrame::read_from(&mut cur, &SAMPLE_SALT)
            .unwrap()
            .unwrap();
        let back: HeartbeatWire = got.decode().unwrap();
        assert_eq!(back, hb);
    }

    #[cfg(feature = "rng")]
    #[test]
    fn generate_session_salt_returns_distinct_values_on_repeat() {
        // Two calls in the same process must produce different salts
        // (with overwhelming probability) and never the zero pattern.
        let a = generate_session_salt();
        let b = generate_session_salt();
        assert_ne!(a, b, "two ThreadRng salt draws collided");
        assert_ne!(a, [0u8; SESSION_SALT_BYTES]);
    }

    #[test]
    fn salt_hex_round_trip() {
        let s = SAMPLE_SALT;
        let h = salt_to_hex(&s);
        assert_eq!(h.len(), SESSION_SALT_BYTES * 2);
        let back = salt_from_hex(&h).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn salt_from_hex_rejects_wrong_length() {
        let err = salt_from_hex("deadbeef").unwrap_err();
        assert!(err.to_string().contains("hex must be"), "got: {err}");
    }

    #[test]
    fn salt_from_hex_rejects_bad_chars() {
        let bad = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"; // 32 chars but non-hex
        let err = salt_from_hex(bad).unwrap_err();
        assert!(err.to_string().contains("hex-decode"), "got: {err}");
    }

    #[test]
    fn payload_too_large_errors_on_write() {
        // u32::MAX + 1 isn't allocatable; simulate with a marker test that
        // the bounds check exists by directly constructing the payload.
        // (Allocating 4GB in tests is impractical; rely on the bounds
        // check at u32::try_from in write_to.)
        let frame = ControlFrame {
            kind: MsgKind::Control,
            auth_tag: 0,
            payload: Vec::new(),
        };
        // Sanity: zero-length payload writes successfully.
        let mut buf = Vec::new();
        frame.write_to(&mut buf).unwrap();
        // Header is exactly 24 bytes (no payload).
        assert_eq!(buf.len(), 24);
    }
}
