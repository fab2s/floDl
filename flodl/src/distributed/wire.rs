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
//! Scheduling latency must stay decoupled from bulk-data throughput;
//! collapsing them into one TCP stream would couple the two. The control
//! channel carries timing, scheduling, and sync messages; the data
//! channel carries averaged-tensor frames. Per-channel back-pressure
//! falls out naturally.
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
//! `auth_tag` is HMAC-SHA256 over `kind || payload_len || payload`
//! (header fields inside the envelope), keyed by the salt,
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
//! [`Tensor`]: crate::tensor::Tensor
//! [`ddp_run`]: crate::distributed::ddp_run
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
pub const CONTROL_PROTOCOL_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Channel-select magics (single-port mux)
// ---------------------------------------------------------------------------
//
// Every cross-host dial toward the controller opens with one of these
// 4-byte LE magics, making the connection self-describing: the
// controller's single-port dispatcher
// ([`PortMux`](crate::distributed::port_mux::PortMux)) peeks the magic
// to route the connection, and the owning subsystem then consumes and
// validates it as the first read on the accepted stream. The magic is
// deliberately UNAUTHENTICATED (it precedes any HMAC'd frame): routing
// is not security-sensitive — a spoofed magic only lands the connection
// at a subsystem whose frame authentication rejects it, exactly as a
// wrong port number did when each channel had its own listener.
//
// Rank↔relay loopback channels keep their dedicated per-host ports and
// do NOT carry a channel magic — they never leave the host.

/// NCCL bootstrap rendezvous (rank → controller).
pub(crate) const CHANNEL_MAGIC_RENDEZVOUS: u32 = 0xF10D_17E0;

/// CPU-reduce data channel (relay → controller).
pub(crate) const CHANNEL_MAGIC_DATA: u32 = 0xF10D_17E1;

/// Coordinator control channel (relay → controller).
pub(crate) const CHANNEL_MAGIC_CONTROL: u32 = 0xF10D_17E2;

/// Membership join channel (worker agent → controller). Unlike the
/// other channels this one can legitimately be dialed by a peer that
/// does not hold the session salt yet — see
/// [`membership`](crate::distributed::membership) for how admission
/// keys its frames.
pub(crate) const CHANNEL_MAGIC_JOIN: u32 = 0xF10D_17E3;

/// ASCII `"GET "` — the first four bytes of an HTTP GET request line.
/// Not a flodl-minted magic: routing it lets plain HTTP clients
/// (`fdl status`, curl, a browser) read the run's `state.json` from
/// the same controller port every other channel shares. The status
/// responder is read-only and serves membership metadata only — its
/// trust follows bind scope exactly like join admission (a loopback
/// bind is reachable only through sshd). See
/// `distributed::status` for the responder.
pub(crate) const CHANNEL_MAGIC_HTTP_GET: u32 = u32::from_le_bytes(*b"GET ");

// ---------------------------------------------------------------------------
// Cleartext guard
// ---------------------------------------------------------------------------

/// Whether `ip` stays inside a controlled / non-public network scope:
/// loopback, RFC1918 private, link-local, or RFC6598 shared address
/// space (CGNAT — also the WireGuard/Tailscale overlay range, where the
/// transport is already encrypted). IPv6: loopback, link-local
/// (`fe80::/10`), unique-local (`fc00::/7`); IPv4-mapped addresses are
/// classified by their inner IPv4.
pub(crate) fn is_private_or_local(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                // RFC 6598 shared address space (100.64.0.0/10).
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64)
        }
        std::net::IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_private_or_local(std::net::IpAddr::V4(v4));
            }
            v6.is_loopback()
                // Link-local fe80::/10.
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // Unique-local fc00::/7.
                || (v6.segments()[0] & 0xfe00) == 0xfc00
        }
    }
}

/// Loud warning (never an error) when a cleartext channel touches a
/// public peer: flodl's documented contract is a controlled / private
/// network — frames are HMAC-authenticated but NOT encrypted, so params
/// and gradients cross such a link readable. Tunnels are the supported
/// way out (`tunnel: true` per host, or any encrypted overlay).
/// Explicit selectors error; conventions warn — this is the convention
/// side. Warned once per distinct peer IP per process so a chatty
/// channel doesn't flood the log.
pub(crate) fn warn_cleartext_public_peer(what: &str, peer: std::net::SocketAddr) {
    if is_private_or_local(peer.ip()) {
        return;
    }
    static WARNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<std::net::IpAddr>>> =
        std::sync::OnceLock::new();
    let warned = WARNED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut set) = warned.lock() {
        if !set.insert(peer.ip()) {
            return;
        }
    }
    eprintln!(
        "flodl: WARNING: {what} peer {peer} is outside any private network \
         range and this channel is CLEARTEXT (frames are HMAC-authenticated, \
         not encrypted) — params and gradients cross this link readable. \
         flodl's documented contract is a controlled/private network; for \
         anything else route the traffic through an SSH tunnel \
         (`tunnel: true` on the worker in cluster.yml) or an encrypted \
         overlay (WireGuard / VPN)."
    );
}

/// Write the channel-select magic. First bytes on every cross-host
/// dial, immediately after `connect`.
pub(crate) fn write_channel_magic<W: Write>(w: &mut W, magic: u32) -> Result<()> {
    w.write_all(&magic.to_le_bytes()).map_err(|e| {
        TensorError::new(&format!("wire: writing channel magic failed: {e}"))
    })
}

/// Consume the 4-byte channel-select magic from an accepted stream and
/// require it to be `expected`. `what` names the accepting subsystem
/// for the error message.
pub(crate) fn expect_channel_magic<R: Read>(
    r: &mut R,
    expected: u32,
    what: &str,
) -> Result<()> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| {
        TensorError::new(&format!("{what}: reading channel magic failed: {e}"))
    })?;
    let got = u32::from_le_bytes(buf);
    if got != expected {
        return Err(TensorError::new(&format!(
            "{what}: channel magic 0x{got:08x} != 0x{expected:08x} \
             (connection routed to the wrong channel?)"
        )));
    }
    Ok(())
}

/// Hard cap on a ControlFrame payload. Control frames carry bincode
/// messages in the bytes-to-KB range (the largest is an NCCL UID or a
/// scalars map); anything bigger is a corrupt or hostile length field.
/// Rejecting BEFORE allocation closes the unauthenticated-length
/// memory-DoS: a single garbage header on an exposed port could
/// otherwise demand a multi-GiB allocation up front.
pub const MAX_CONTROL_PAYLOAD: usize = 16 * 1024 * 1024;

/// Incremental-allocation chunk for length-prefixed reads whose
/// legitimate payloads can be large (tensor data). The buffer grows as
/// bytes actually arrive instead of trusting the unauthenticated length
/// for one big up-front allocation.
pub(crate) const READ_CHUNK: usize = 64 * 1024 * 1024;

/// Default hard ceiling on any length-prefixed data-plane payload (mux
/// records, len-framed blobs, RoundFrame tensor totals) when no
/// model-derived bound has been installed. Length fields are
/// UNAUTHENTICATED until the trailing MAC verifies, so a hostile or
/// corrupt peer can claim up to `u32::MAX`; incremental allocation makes
/// the attacker pay the bandwidth, this cap bounds the memory.
pub(crate) const DEFAULT_FRAME_CEILING: usize = 1 << 30; // 1 GiB

/// Floor for the model-derived frame ceiling: even a tiny model keeps a
/// generous reject-threshold so bookkeeping frames, header slack, and
/// future dtype growth never brush the bound.
const FRAME_CEILING_FLOOR: usize = 64 * 1024 * 1024; // 64 MiB

/// Session frame ceiling, installed once per process by
/// [`set_frame_ceiling`]. `None` until then → [`DEFAULT_FRAME_CEILING`].
static FRAME_CEILING: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// The active hard ceiling for length-prefixed data-plane payloads.
///
/// Model-derived when the process has installed one (launcher: from its
/// CPU probe; relay: from its `RelaySpec`; rank: from the model it
/// builds), [`DEFAULT_FRAME_CEILING`] otherwise. Purely a local
/// reject-threshold — no cross-process agreement is required, so a
/// process still on the default merely rejects later than its peers.
pub(crate) fn frame_ceiling() -> usize {
    *FRAME_CEILING.get().unwrap_or(&DEFAULT_FRAME_CEILING)
}

/// Install the session frame ceiling. First caller wins (the value is
/// derived deterministically from the model in every process, so a
/// second call carries the same number); zero is ignored (unset relay
/// spec field from a build that predates it).
pub(crate) fn set_frame_ceiling(bytes: usize) {
    if bytes == 0 {
        return;
    }
    if FRAME_CEILING.set(bytes).is_ok() {
        crate::verbose!("  wire: frame ceiling set to {bytes} bytes (model-derived)");
    }
}

/// Derive the frame ceiling from a model's wire footprint (Σ param +
/// buffer bytes): ×2 margin for header slack / dtype growth / count
/// gathers, floored at 64 MiB. The ×2 also covers the relay fold's
/// `HostFrame`, which sums element-wise and never outgrows one rank's
/// frame.
pub(crate) fn derive_frame_ceiling(model_wire_bytes: usize) -> usize {
    model_wire_bytes
        .saturating_mul(2)
        .max(FRAME_CEILING_FLOOR)
}

/// Σ bytes of `tensors` as they ride a `RoundFrame` (numel × element
/// size; header slack is absorbed by [`derive_frame_ceiling`]'s margin).
pub(crate) fn tensors_wire_bytes(tensors: &[crate::tensor::Tensor]) -> usize {
    tensors
        .iter()
        .map(|t| t.numel().max(0) as usize * t.dtype().element_size())
        .sum()
}

/// One shared TCP connect budget for every cluster dial (rank ->
/// rendezvous, relay -> controller, worker -> coordinator, reduce
/// client -> controller): ~30s of 500ms attempts. The four call sites
/// used to carry four hand-rolled copies with DISAGREEING budgets
/// (10s relay vs 30s rendezvous vs 5s worker) — a slow controller
/// start then killed the relay first and the diagnostics blamed the
/// wrong tier. One budget, ordered by definition.
pub(crate) const CONNECT_ATTEMPTS: u32 = 60;
/// Pause between [`CONNECT_ATTEMPTS`].
pub(crate) const CONNECT_BACKOFF: std::time::Duration =
    std::time::Duration::from_millis(500);
/// Zero-progress write-stall ceiling for every cluster socket.
///
/// `SO_SNDTIMEO` applies per `write()` call and `write_all` loops over
/// partial writes, so a slow-but-draining link never trips (each
/// successful partial write starts a fresh window); only a peer whose
/// receive path made no progress for this long — a wedged process or a
/// silently-dead link TCP cannot detect — turns the blocking write into
/// an error. Matches the heartbeat-staleness default so both liveness
/// axes agree on what "gone" means. Socket options are fd-level, so one
/// call at socket setup covers every cloned handle.
pub(crate) const WRITE_STALL_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);

/// Env var scaling every cluster network deadline together.
///
/// The wire budgets above (connect, write-stall) and their siblings
/// (coord heartbeat staleness, rank coord-liveness, CPU reduce read
/// deadline) are LAN-tuned defaults that also define ONE coherent
/// notion of "gone" — a peer silent past ~30s. On a slow link (WAN /
/// NAT hub-and-spoke — a declared target of the CPU controller path)
/// that notion must stretch *uniformly*: scaling only one budget lets
/// a slower axis declare a peer dead while a faster one still waits.
/// So a single multiplier scales the whole set: `>1` for slow
/// networks (`3` ≈ "gone" at 90s), `<1` for test rigs that want fast
/// failure detection (floor 0.1 keeps every deadline above the 1s
/// coord-heartbeat cadence, which deliberately does NOT scale — extra
/// beacons on a slow link are harmless, only deadlines matter).
///
/// Read once per process (cached). The launcher forwards it to remote
/// rank/relay children automatically (like `FLODL_VERBOSITY`), so
/// setting it where `fdl` runs covers the whole cluster coherently —
/// that is the canonical route. (A `cluster.env:` entry would reach
/// rank children but not the controller-side coordinator: a split
/// notion of "gone". The fan-out path validates it early and loudly;
/// this reader warns-once and falls back to 1.0 so a library-only
/// consumer with a bad value degrades to defaults instead of
/// panicking mid-run.)
pub(crate) const ENV_NET_TIMEOUT_SCALE: &str = "FLODL_NET_TIMEOUT_SCALE";

/// Parse a `FLODL_NET_TIMEOUT_SCALE` value. Pure — unit-tested apart
/// from the cached env reader. `None` (unset) is scale 1.0. Set values
/// must be finite and ≥ 0.1.
pub(crate) fn parse_net_timeout_scale(raw: Option<&str>) -> std::result::Result<f64, String> {
    let Some(raw) = raw else { return Ok(1.0) };
    let trimmed = raw.trim();
    let parsed: f64 = trimmed.parse().map_err(|_| {
        format!(
            "{ENV_NET_TIMEOUT_SCALE}={trimmed:?} is not a number; expected a \
             scale factor ≥ 0.1 (e.g. 3 for a slow WAN link, 0.5 for a \
             fast-failure test rig)"
        )
    })?;
    if !parsed.is_finite() || parsed < 0.1 {
        return Err(format!(
            "{ENV_NET_TIMEOUT_SCALE}={trimmed} is out of range; expected a \
             finite scale factor ≥ 0.1 (0.1 keeps every deadline above the \
             1s heartbeat cadence)"
        ));
    }
    Ok(parsed)
}

/// The process-wide network-timeout scale (cached on first read).
/// Invalid values warn once and fall back to 1.0 — the cluster fan-out
/// path has already validated loudly by the time library code runs.
pub(crate) fn net_timeout_scale() -> f64 {
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *SCALE.get_or_init(|| {
        let raw = std::env::var(ENV_NET_TIMEOUT_SCALE).ok();
        match parse_net_timeout_scale(raw.as_deref()) {
            Ok(s) => s,
            Err(msg) => {
                eprintln!("flodl: {msg}; using default scale 1.0");
                1.0
            }
        }
    })
}

/// [`CONNECT_ATTEMPTS`] scaled by [`net_timeout_scale`] (backoff pause
/// stays fixed; the attempt count carries the scale).
pub(crate) fn connect_attempts() -> u32 {
    ((CONNECT_ATTEMPTS as f64 * net_timeout_scale()).ceil() as u32).max(1)
}

/// [`WRITE_STALL_TIMEOUT`] scaled by [`net_timeout_scale`].
pub(crate) fn write_stall_timeout() -> std::time::Duration {
    WRITE_STALL_TIMEOUT.mul_f64(net_timeout_scale())
}

/// Scale a whole-second deadline default by [`net_timeout_scale`],
/// flooring at 1s. Used where the deadline is carried as `u64` seconds
/// (coord heartbeat staleness, rank coord-liveness, reduce read).
pub(crate) fn scaled_deadline_secs(base_secs: u64) -> u64 {
    ((base_secs as f64 * net_timeout_scale()).ceil() as u64).max(1)
}

/// Join a host and port into a dial string, bracketing IPv6 literals.
///
/// A bare IPv6 address (`fe80::1`) concatenated as `host:port` yields the
/// ambiguous `fe80::1:1337`, which [`std::net::ToSocketAddrs`] rejects —
/// the port-suffix form requires brackets: `[fe80::1]:1337`. Hostnames
/// and IPv4 addresses (which never contain `:`) pass through unchanged,
/// and an already-bracketed host is left as-is. The `(host, port)` tuple
/// form of `ToSocketAddrs` avoids this, but a pre-formatted string is
/// needed where the dial target must also `Display` (see
/// [`connect_with_retry`]'s bounds).
pub(crate) fn join_host_port(host: &str, port: u16) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// TCP connect with the shared cluster retry budget. `what` names the
/// dial for the error message (e.g. "relay upstream", "rendezvous").
pub(crate) fn connect_with_retry<A>(
    addr: A,
    what: &str,
) -> Result<std::net::TcpStream>
where
    A: std::net::ToSocketAddrs + std::fmt::Display + Copy,
{
    let attempts = connect_attempts();
    let mut last_err: Option<std::io::Error> = None;
    for _ in 0..attempts {
        match std::net::TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last_err = Some(e);
                std::thread::sleep(CONNECT_BACKOFF);
            }
        }
    }
    Err(TensorError::new(&format!(
        "{what}: connect to {addr} failed after {attempts} attempts \
         (~{}s): {}",
        attempts as u64 * CONNECT_BACKOFF.as_millis() as u64 / 1000,
        last_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "no error captured".into()),
    )))
}

/// Read exactly `len` bytes, growing the buffer in [`READ_CHUNK`] steps
/// so a garbage/hostile length field can only make us allocate as much
/// as the peer actually sends.
pub(crate) fn read_exact_incremental<R: Read>(
    r: &mut R,
    len: usize,
) -> std::io::Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    while buf.len() < len {
        let chunk = (len - buf.len()).min(READ_CHUNK);
        let old_len = buf.len();
        buf.resize(old_len + chunk, 0);
        r.read_exact(&mut buf[old_len..])?;
    }
    Ok(buf)
}

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
    /// Orphan / reserved for wire-format stability. Was worker →
    /// coordinator pre-snapshot metadata (`ParamSnapshotMetaWire`, now
    /// deleted); the coordinator no-ops this tag on receipt. The enum
    /// value is kept so the protocol byte layout stays stable.
    ParamSnapshotMeta = 0x04,
    /// Orphan / reserved for wire-format stability. Live heartbeats now
    /// flow through [`TimingMsgWire::Heartbeat`] over [`Self::Timing`]
    /// (the dedicated `HeartbeatWire` struct was deleted); the
    /// coordinator no-ops this tag on receipt. The enum value is kept so
    /// the protocol byte layout stays stable.
    Heartbeat = 0x05,
    /// Bootstrap rendezvous frame: worker → controller hello, controller →
    /// worker role assignment, and either-direction NCCL unique-id
    /// transport. Payload: [`RendezvousMsgWire`].
    Rendezvous = 0x06,
    /// Membership join frame: worker agent ↔ controller on the join
    /// channel (hello / accept / reject / world-formed / rank-exited /
    /// abort). Payload: [`JoinMsgWire`].
    Join = 0x07,
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
            0x06 => Ok(MsgKind::Rendezvous),
            0x07 => Ok(MsgKind::Join),
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
///
/// The salt-keyed MAC is therefore also a SESSION TOKEN, not just an
/// integrity check: frames from an adjacent training session (same rig,
/// reused ports, stale process) carry a different salt and fail
/// authentication — cross-run signal isolation is a guarantee, not an
/// accident of port allocation.
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

/// Frame MAC for [`ControlFrame`]: HMAC over `kind || payload_len ||
/// payload` (all little-endian), keyed by the session salt. Covering the
/// header fields means a flipped `kind` (a captured `Shutdown` replayed
/// as `SyncNow`-shaped junk) or a forged length no longer authenticates —
/// the payload-only MAC left both outside the envelope. Mirrors
/// `hmac_sha256_64_2`'s header-coverage discipline on the mux layer.
fn frame_mac(salt: &SessionSalt, kind: MsgKind, payload: &[u8]) -> u64 {
    let mut macd = Vec::with_capacity(8 + payload.len());
    macd.extend_from_slice(&(kind as u32).to_le_bytes());
    macd.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    macd.extend_from_slice(payload);
    hmac_sha256_64(salt, &macd)
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
/// inclusion in the cluster envelope JSON. Reuses the same hex format
/// the cluster module uses for envelope encoding.
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
        let auth_tag = frame_mac(salt, kind, &payload);
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
    /// `try_read_from`.
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
        if payload_len > MAX_CONTROL_PAYLOAD {
            return Err(TensorError::new(&format!(
                "wire: ControlFrame payload_len {payload_len} exceeds \
                 MAX_CONTROL_PAYLOAD {MAX_CONTROL_PAYLOAD} (kind={kind:?}); \
                 rejecting before allocation"
            )));
        }
        let mut payload = vec![0u8; payload_len];
        r.read_exact(&mut payload).map_err(|e| {
            TensorError::new(&format!(
                "wire: ControlFrame payload read failed (kind={kind:?}, len={payload_len}): {e}"
            ))
        })?;
        let actual = frame_mac(salt, kind, &payload);
        // Timing-safe by construction: a u64 equality is one compare, no
        // early-exit byte scan to probe. The computed tag in the error
        // text below never crosses the wire — MAC failures are dropped
        // without a reply on every accept path (a reject that echoed the
        // valid tag for the attacker's payload would hand them a forgery).
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
/// Note: `PartialEq` only (not `Eq`) because the `EpochAggregated`
/// variant carries float fields via [`EpochMetricsWire`]. All test
/// asserts use `assert_eq!` / `assert_ne!` / `matches!` which need
/// only `PartialEq`.
///
/// [`ddp_run::ControlMsg`]: crate::distributed::ddp_run::ControlMsg
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ControlMsgWire {
    /// CPU path: ask the worker to send its current ParamSnapshot.
    RequestParams,
    /// CPU path: averaged params with `version` are ready on the data
    /// channel; worker reads the next RoundFrame and applies it.
    ///
    /// `next_plan` is the atomic-dispatch payload: when `Some`, it is
    /// the rank's next reduce-window chunk, folded into this same frame
    /// so the worker applies the averaged params **and** starts the next
    /// window without a separate `StartEpoch` round-trip from the coord
    /// (clawing back the post-reduce control RTT on the CPU path). The
    /// inbound bridge synthesises a `ControlMsg::StartEpoch(next_plan)`
    /// into the worker's control channel, ordered *after* the param
    /// bridge's `ControlMsg::Update(avg)` (same mpsc, FIFO), so averaged
    /// params are always applied before the next window begins. `None`
    /// at an epoch boundary (or when the reduce barrier holds the rank):
    /// the existing epoch-advance / `wake_idle_ranks_in_progressive`
    /// path dispatches the next chunk as before.
    Update {
        version: u64,
        next_plan: Option<EpochPlanWire>,
    },
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
    /// Coord-emitted directive to persist a checkpoint bundle for the
    /// given `version` (epoch index at the cadence boundary). Targeted:
    /// only the rank whose `rank == target_rank` runs its
    /// `checkpoint_fn`; every other rank receiving this frame no-ops.
    /// The coord owns the role assignment (sticky `checkpoint_role`
    /// with failover on rank death or `CheckpointResult.error`); the
    /// worker never decides whether it is the checkpointer.
    ///
    /// `target_rank` semantics:
    /// - `0..world_size` → the rank ID that should execute. Other
    ///   ranks receiving the frame silently ignore it.
    /// - `u64::MAX` → reserved for "controller executes" (CPU-async
    ///   mode where the controller already holds the canonical
    ///   averaged tensors post `finish_averaging_cpu`); dispatching it
    ///   today emits a loud error.
    ///
    /// Execution result flows back via
    /// [`TimingMsgWire::CheckpointResult`] (success or failure) so the
    /// controller can retry on a different live rank when the
    /// assigned rank reports an error.
    Checkpoint { version: u64, target_rank: u64 },
    /// Coord-emitted directive to run the user's [`EvalFn`] against
    /// `eval_dataset` on `target_rank`. Targeted (parallels
    /// [`Self::Checkpoint`]): only the rank whose `rank == target_rank`
    /// executes; every other rank receiving this frame no-ops. The
    /// coord owns the role assignment (`eval_role`, resolved per
    /// [`EpochCallbackPolicy`]); the worker never decides.
    ///
    /// `target_rank == u64::MAX` is reserved for "controller executes"
    /// (a future CPU-async variant); the current implementation rejects
    /// it via the same loud error as [`Self::Checkpoint`].
    ///
    /// Result flows back via [`TimingMsgWire::EvalResult`] with the
    /// same `schedule_id`.
    ///
    /// [`EvalFn`]: crate::distributed::ddp_run::EvalFn
    /// [`EpochCallbackPolicy`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy
    ExecuteEvalCallback {
        schedule_id: u64,
        epoch: u64,
        target_rank: u64,
    },
    /// Coord-emitted notification that the rank designated to fire the
    /// user-supplied `epoch_fn` has been resolved (or re-resolved on
    /// rank death). Broadcast to every worker; each worker updates
    /// its local `epoch_callback_role` state and fires `epoch_fn`
    /// only on epoch transitions where `epoch_callback_role ==
    /// self.rank`.
    ///
    /// Unlike `Checkpoint` / `ExecuteEvalCallback`, the `epoch_fn` is
    /// not coord-dispatched — it fires autonomously inside the
    /// worker's main loop at every epoch transition. The role
    /// assignment must therefore live in worker state, not be encoded
    /// per-message.
    ///
    /// Used for [`EpochCallbackPolicy::Fastest`] runtime resolution
    /// (ElChe-derived) and also fires for `Rank(n)` at startup so the
    /// worker has a definite role before the first epoch transition.
    ///
    /// [`EpochCallbackPolicy::Fastest`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Fastest
    SetEpochCallbackRole { rank: u64 },
    /// Shut down this worker.
    Shutdown,
    /// Coord-emitted directive to persist a checkpoint bundle (model
    /// params + buffers + optimizer state + meta JSON) to the
    /// configured `save_path` and then exit. Sent when a cluster run
    /// is unrecoverable: the `max_failure` threshold was breached, or
    /// (in NCCL mode) the surviving cohort dropped below 2 ranks and
    /// no new comm can be formed. Workers consult
    /// [`crate::distributed::CheckpointBundle`] for bundle paths.
    ///
    /// `reason` is the wire-byte encoding of
    /// [`crate::distributed::SaveReason`]; decoded via
    /// [`crate::distributed::SaveReason::from_u8`]. Unknown bytes are
    /// treated as [`crate::distributed::SaveReason::GracefulShutdown`]
    /// by the receiver (forward-compatible fallback).
    ShutdownWithSave { reason: u8 },
    /// Aggregated per-epoch metrics broadcast from coord to every
    /// rank after `drain_metrics_and_aggregate` has built an
    /// [`EpochMetricsWire`] from all alive ranks' per-rank reports.
    ///
    /// Lets each rank's local `Graph` surface the GLOBAL aggregated
    /// view (user-defined scalars + per-rank GPU tabs) under
    /// `latest_metrics()` / `graph_gpu_metrics()`. The framework-
    /// managed `Trainer::builder` path already had this view via
    /// `DdpHandle::next_metrics()`; this broadcast gives the same
    /// view to the user-owned `Trainer::setup` training loop in
    /// process-per-rank cluster mode. User code stays identical:
    /// `monitor.log(epoch, dur, &model)` sees the aggregated view
    /// regardless of single-GPU / local-multi-GPU / cluster.
    EpochAggregated(EpochMetricsWire),
    /// NCCL consensus checkpoint: tell the elected rank to write its CURRENT
    /// model (params + buffers) to `<save_path>.fdl` as the resumable consensus
    /// — distinct from [`Self::Checkpoint`] (which fires the user `checkpoint_fn`)
    /// and from [`Self::ShutdownWithSave`] (which also writes `.optim` and
    /// exits). Dispatched by the coordinator at `finish_averaging_nccl`, AFTER
    /// the in-place AllReduce-Avg, so the rank's `self.model` holds the pure
    /// consensus (no EASGD blend on the NCCL path). The CPU path does NOT use
    /// this — its consensus is forged controller-side
    /// ([`crate::distributed::CheckpointForge`]). Targeted send; the worker
    /// no-ops unless `target_rank == self.rank`. No result frame (best-effort,
    /// mirrors the CPU forge's detached write); the `.meta.json` written
    /// coord-side is the resume index.
    SaveConsensusModel { target_rank: u64 },
    /// Coord→rank liveness beacon. Broadcast to every rank on a fixed ~1s
    /// cadence, independent of training traffic, so a rank can distinguish
    /// "coordinator alive but legitimately silent during my compute window"
    /// from "coordinator wedged / gone." The rank's inbound bridge resets its
    /// coord-liveness deadline on ANY inbound frame (this beacon or real
    /// traffic) and, if no frame arrives within `heartbeat_timeout_secs`,
    /// declares the coord dead — poisoning its peer ledger and injecting a
    /// local Shutdown so the rank exits with a death record instead of
    /// spinning forever on `WouldBlock` against a wedged-open socket.
    ///
    /// This is the reverse-direction twin of [`TimingMsgWire::Heartbeat`]
    /// (rank→coord): both directions now have an independent liveness signal
    /// on the same wall-clock timescale. Purely informational — the inbound
    /// bridge intercepts it and never forwards it to the inner worker.
    CoordHeartbeat,
    /// The rank's data-reservation view: its upcoming run-stream as
    /// `segments` — `(epoch, spans)` in walk order, each segment's
    /// `(offset, size)` spans into that epoch's global permutation in
    /// certainty order (the rank's own reserved span first, then the
    /// tails of the other ranks' spans: the truing margins, whose final
    /// owner is uncertain). Cross-epoch segments let the stager walk
    /// into the next epoch while this one trains — the data plane is
    /// epoch-blind, an epoch is just where the order function switches.
    ///
    /// `counts` is the current reduce-window schedule (batch counts per
    /// rank): the consumption-rate snapshot a worker uses to split its
    /// host's RAM budget consumption-proportionally among co-hosted
    /// ranks (equal lookahead TIME, not equal bytes).
    ///
    /// Purely advisory for the worker's background stager: staging may
    /// overlap across ranks near the boundaries, allocation
    /// (`StartEpoch` chunks) never does, so stale or over-staged data
    /// needs no invalidation — only allocated work executes. Emitted at
    /// progressive epoch start and refreshed at reduce boundaries
    /// (reservation state changes ride the window clock). Latest frame
    /// wins; workers without a stager ignore it.
    StageAdvisory {
        counts: Vec<u64>,
        segments: Vec<(u64, Vec<(u64, u64)>)>,
    },
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
        /// Per-batch DATA wall (ms): prefetch/H2D stall for this batch
        /// (prefetch path) or dataset fetch+to-device (sync path). Paired
        /// with `batch_ms` (compute-only) so the coordinator can accumulate
        /// a DELIVERED feed (`batch_ms + data_ms`) continuously — present at
        /// sync by construction, unlike the completion-frame-closed span.
        #[serde(default)]
        data_ms: f64,
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
    /// Worker → coord eval result. Carries the scalar metric returned
    /// by the user's [`crate::distributed::ddp_run::EvalFn`] (or an
    /// error string when the closure failed). Result-bearing per
    /// `feedback_loud_errors_over_silent.md`: success path carries
    /// `error = None`; failures carry the error message and `metric =
    /// 0.0`.
    ///
    /// `elapsed_ms` is the wall-time the eval closure took on the rank.
    /// Mirrors [`Self::CheckpointResult`] for symmetry: the coord
    /// subtracts it from `wall_ms_accum[rank]` so ElChe does not
    /// mis-attribute eval cost as compute slowness, and feeds it into
    /// `last_eval_elapsed_ms_ewma` for callback-aware partition
    /// scheduling.
    EvalResult {
        rank: u64,
        schedule_id: u64,
        epoch: u64,
        metric: f64,
        elapsed_ms: f64,
        error: Option<String>,
    },
    /// Result of a `checkpoint_fn` invocation by `rank` for the given
    /// `version`. Parallels [`Self::EvalResult`] for the checkpoint
    /// task: workers never decide on retry; they always report
    /// (success or failure) and let the controller pick the next
    /// action. `elapsed_ms` is the wall-time the closure took (used
    /// by the coord to (a) subtract from `wall_ms_accum[rank]` so
    /// ElChe does not mis-attribute checkpoint cost as training
    /// slowness, and (b) feed a `last_checkpoint_elapsed_ms_ewma`
    /// reserved for v2 rendezvous-aware scheduling). Success carries
    /// `error = None`; failure carries the closure's `TensorError`
    /// rendered as a String (`feedback_loud_errors_over_silent.md`).
    CheckpointResult {
        rank: u64,
        version: u64,
        elapsed_ms: f64,
        error: Option<String>,
    },
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
    /// Worker → coord notice: this rank just finished firing the
    /// user-supplied `epoch_fn` for the given `epoch`. `elapsed_ms` is
    /// the wall-time the closure took. Symmetric counterpart to
    /// [`Self::EvalResult`] / [`Self::CheckpointResult`] for the
    /// autonomously-fired `epoch_fn` path (the worker fires `epoch_fn`
    /// inside its main loop on epoch boundaries; there is no
    /// coord-dispatched directive, only this post-fire report).
    ///
    /// The coord subtracts `elapsed_ms` from `wall_ms_accum[rank]` so
    /// ElChe does not mis-attribute epoch_fn cost as compute slowness,
    /// and feeds it into `last_epoch_fn_elapsed_ms_ewma` for
    /// callback-aware partition scheduling.
    EpochFnElapsed {
        rank: u64,
        epoch: u64,
        elapsed_ms: f64,
    },
    /// Rank → controller dashboard registration. Sent at worker startup
    /// when the user's harness has called [`crate::monitor::Monitor::serve`]
    /// before launching training. The launcher's coord forwards this to
    /// the dashboard sink, which binds the HTTP server (idempotent across
    /// ranks — all ranks register the same port, first wins, subsequent
    /// registrations validate match).
    DashboardRegister {
        rank: u64,
        /// HTTP port the controller should bind for the dashboard.
        port: u16,
    },
    /// Rank → controller dashboard graph SVG. Sent at worker startup
    /// when the user's harness has called
    /// [`crate::monitor::Monitor::watch`] before launching training.
    /// Every rank ships the same SVG (graph is identical across ranks);
    /// the launcher's dashboard caches the first arrival and ignores
    /// subsequent.
    DashboardSetSvg {
        rank: u64,
        svg: String,
        label: Option<String>,
        hash: Option<String>,
    },
    /// Rank → controller dashboard metadata blob (hyperparameters,
    /// config, etc.) Sent at worker startup. Carries the JSON value
    /// pre-serialized by the rank (avoids dragging `serde_json::Value`
    /// into the bincode-serialized wire surface).
    DashboardSetMetadata {
        rank: u64,
        json: String,
    },
    /// Rank → controller per-rank hardware summary string. Sent at
    /// worker startup. The launcher renders these as per-rank tabs
    /// labelled `host:lr=<local_rank> gr=<global_rank>`; the
    /// `host`/`local_rank` are resolved from the launcher's
    /// [`FullCluster`] world map keyed by `rank` (global).
    ///
    /// [`FullCluster`]: crate::distributed::launcher::FullCluster
    DashboardSetHardware {
        rank: u64,
        summary: String,
    },
}

/// Per-GPU snapshot wire mirror of [`crate::monitor::resources::GpuSnapshot`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct GpuSnapshotWire {
    pub device_index: u8,
    pub name: String,
    pub util_percent: Option<f32>,
    pub vram_allocated_bytes: Option<u64>,
    pub vram_total_bytes: Option<u64>,
}

/// Per-rank resource sample wire mirror of
/// [`crate::monitor::resources::ResourceSample`]. Carried as an optional
/// field on every [`MetricsMsgWire`] so the launcher's dashboard can
/// render per-rank hardware tabs without paying for a separate
/// `MsgKind` round-trip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ResourceSampleWire {
    pub cpu_percent: Option<f32>,
    pub ram_used_bytes: Option<u64>,
    pub ram_total_bytes: Option<u64>,
    pub gpu_util_percent: Option<f32>,
    pub vram_total_bytes: Option<u64>,
    pub vram_allocated_bytes: Option<u64>,
    pub aggregate_rank: Option<u8>,
    pub gpus: Vec<GpuSnapshotWire>,
}

impl From<crate::monitor::GpuSnapshot> for GpuSnapshotWire {
    fn from(g: crate::monitor::GpuSnapshot) -> Self {
        GpuSnapshotWire {
            device_index: g.device_index,
            name: g.name,
            util_percent: g.util_percent,
            vram_allocated_bytes: g.vram_allocated_bytes,
            vram_total_bytes: g.vram_total_bytes,
        }
    }
}

impl From<GpuSnapshotWire> for crate::monitor::GpuSnapshot {
    fn from(w: GpuSnapshotWire) -> Self {
        crate::monitor::GpuSnapshot {
            device_index: w.device_index,
            name: w.name,
            util_percent: w.util_percent,
            vram_allocated_bytes: w.vram_allocated_bytes,
            vram_total_bytes: w.vram_total_bytes,
        }
    }
}

impl From<crate::monitor::ResourceSample> for ResourceSampleWire {
    fn from(s: crate::monitor::ResourceSample) -> Self {
        ResourceSampleWire {
            cpu_percent: s.cpu_percent,
            ram_used_bytes: s.ram_used_bytes,
            ram_total_bytes: s.ram_total_bytes,
            gpu_util_percent: s.gpu_util_percent,
            vram_total_bytes: s.vram_total_bytes,
            vram_allocated_bytes: s.vram_allocated_bytes,
            aggregate_rank: s.aggregate_rank,
            gpus: s.gpus.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<ResourceSampleWire> for crate::monitor::ResourceSample {
    fn from(w: ResourceSampleWire) -> Self {
        crate::monitor::ResourceSample {
            cpu_percent: w.cpu_percent,
            ram_used_bytes: w.ram_used_bytes,
            ram_total_bytes: w.ram_total_bytes,
            gpu_util_percent: w.gpu_util_percent,
            vram_total_bytes: w.vram_total_bytes,
            vram_allocated_bytes: w.vram_allocated_bytes,
            aggregate_rank: w.aggregate_rank,
            gpus: w.gpus.into_iter().map(Into::into).collect(),
        }
    }
}

/// Wire-side mirror of [`ddp_run::MetricsMsg`]. All fields plain data.
///
/// `resources` is `Option<>` because not every per-epoch report needs a
/// resource sample (the worker only populates it when the dashboard has
/// been requested, and progressive-chunk reports leave it empty). When
/// `Some(_)`, the launcher's dashboard renders per-rank hardware tabs
/// for the originating rank.
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
    #[serde(default)]
    pub resources: Option<ResourceSampleWire>,
}

/// Wire-side mirror of [`ddp_run::EpochMetrics`]. Carries the
/// aggregated cross-rank view the coord builds in
/// [`ClusterCoordinator::drain_metrics_and_aggregate`] back to every
/// rank via [`ControlMsgWire::EpochAggregated`], so each rank's
/// `Graph` can surface the global metric view + per-rank GPU tabs to
/// user code without the user needing to think about ranks.
///
/// Field shapes mirror [`crate::distributed::ddp_run::EpochMetrics`]
/// exactly; `usize` widens to `u64` on the wire for stability across
/// 32/64-bit hosts.
///
/// [`ddp_run::EpochMetrics`]: crate::distributed::ddp_run::EpochMetrics
/// [`ClusterCoordinator::drain_metrics_and_aggregate`]:
///     crate::distributed::cluster_coordinator::ClusterCoordinator
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EpochMetricsWire {
    pub epoch: u64,
    pub scalars: HashMap<String, f64>,
    pub per_rank: Vec<HashMap<String, f64>>,
    pub avg_loss: f64,
    pub epoch_ms: f64,
    pub per_rank_throughput: Vec<f64>,
    pub per_rank_batch_share: Vec<f64>,
    pub per_rank_share_complete_ms: Vec<f64>,
    pub per_rank_compute_only_ms: Vec<f64>,
    pub per_rank_data_starve_ms: Vec<f64>,
    pub device_indices: Vec<u8>,
}



/// Per-rank role assignment for the bootstrap rendezvous.
///
/// The controller (orchestrator on the launcher host) decides which rank
/// generates the NCCL unique ID via `ncclGetUniqueId`. The controller
/// itself cannot make that call — its process may not link libnccl, and
/// even when it does, the controller's role is strictly orchestration.
/// Same constraint as elastic-resize ([`ControlMsgWire::RequestNewNcclId`]).
///
/// Default policy: the first rank of the local-host worker if any, else
/// `workers[0].ranks[0]`. Future: routable via [`EpochCallbackPolicy`]
/// once timing data exists (cannot apply at bootstrap — no data yet).
///
/// [`EpochCallbackPolicy`]: crate::distributed::ddp_run::EpochCallbackPolicy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendezvousRole {
    /// You generate `NcclUniqueId::new()` and send it back to the controller.
    Generate,
    /// Wait — the controller will send you the unique ID after collecting
    /// it from the designated generator.
    Wait,
}

/// Wire-side bootstrap rendezvous message. Carried inside a
/// [`ControlFrame`] tagged with [`MsgKind::Rendezvous`]; HMAC-signed
/// with the session salt like every other control frame.
///
/// Three-message protocol per worker connection (all controller-driven):
///
/// 1. worker → controller: [`Self::Hello`] (dataset-sig check + identity)
/// 2. controller → worker: [`Self::Role`] (Generate or Wait)
/// 3. UID transport:
///    - if Generate: worker → controller: [`Self::Uid`]
///    - if Wait:     controller → worker: [`Self::Uid`] (broadcast after collection)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendezvousMsgWire {
    /// Worker introduces itself to the controller after dialing in.
    ///
    /// The `dataset_sig` lets the controller verify all ranks agree on
    /// the same dataset shard layout — silent divergence across ranks is
    /// the worst class of bug. The `global_rank` is the value the
    /// controller assigned at probe time (lives in the cluster
    /// envelope); the worker echoes it so the controller can index the
    /// accepted stream by rank for the subsequent [`Self::Role`] target.
    Hello {
        /// 32-byte signature of the dataset shard configuration. Same
        /// value across every rank when shards are consistent.
        dataset_sig: [u8; 32],
        /// Rank the controller assigned via worker-order × device-count
        /// probe. Read from `FLODL_INTERNAL_LOCAL_RANK` × envelope `worker.ranks`.
        global_rank: u32,
        /// Worker host name (diagnostic; controller logs it in error
        /// messages).
        host_name: String,
    },
    /// Controller's role assignment to a worker. Sent after every Hello
    /// has been validated.
    Role(RendezvousRole),
    /// NCCL unique-id bytes. Generator rank writes this to the
    /// controller after receiving [`RendezvousRole::Generate`]; the
    /// controller broadcasts it to every Wait rank.
    Uid {
        /// Raw 128-byte `NcclUniqueId` value.
        uid_bytes: Vec<u8>,
    },
}

/// Wire-side membership message. Carried inside a [`ControlFrame`]
/// tagged with [`MsgKind::Join`] on the join channel
/// ([`CHANNEL_MAGIC_JOIN`]).
///
/// The join channel is the one channel a peer may dial WITHOUT holding
/// the session salt yet (open admission hands the salt out in the
/// reply), so its pre-admission frames — [`Self::Hello`],
/// [`Self::Accept`], [`Self::Reject`] — are keyed by the trust mode:
/// the session salt when it is pre-shared (fan-out rig mode; a hello
/// keyed otherwise fails authentication and is rejected), or an
/// all-zeros key in open-admission mode, where the MAC still enforces
/// protocol conformance + integrity but authentication comes from the
/// bind scope (loopback behind sshd) instead. Everything after
/// admission — [`Self::WorldFormed`], [`Self::RankExited`],
/// [`Self::Abort`] — is keyed with the session salt, binding the
/// connection to the admitted identity.
///
/// Protocol version rides in every [`ControlFrame`] header
/// (`CONTROL_PROTOCOL_VERSION`), so a version-skewed worker is rejected
/// at the frame layer before its hello is even decoded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinMsgWire {
    /// Worker agent → controller: request to join the forming world.
    Hello {
        /// Worker host name (member identity; duplicate joins under the
        /// same name are rejected).
        host: String,
        /// Physical CUDA device ids this worker will run, one per rank
        /// (already resolved on the worker — `"all"` shorthands never
        /// cross the wire). The controller carries these into the
        /// host's envelope so rank↔device pinning survives the dial-in
        /// path unchanged.
        local_devices: Vec<u8>,
        /// GPU inventory, one label per device (informational: logged
        /// and surfaced in the membership state, not validated).
        gpus: Vec<String>,
        /// libtorch variant label (informational, same as `gpus`).
        libtorch: String,
        /// 32-byte dataset signature; must match the controller's own.
        /// A stale worker from a previous run is rejected at the door,
        /// not discovered mid-epoch.
        dataset_sig: [u8; 32],
    },
    /// Controller → worker agent: admitted. Carries the assigned global
    /// rank ids (admission order — contiguous by construction).
    Accept {
        /// Global rank ids assigned to this worker's local GPUs.
        ranks: Vec<u32>,
        /// Session salt (hex) in open-admission mode, where the joiner
        /// does not hold it yet. `None` when the salt was pre-shared —
        /// re-sending it would put the secret on the wire for nothing.
        salt_hex: Option<String>,
        /// Seconds (already scaled) the worker should be prepared to
        /// wait for [`Self::WorldFormed`] — the remaining hard-cap
        /// budget at admission time, so the worker's read deadline
        /// self-describes instead of guessing the controller's window
        /// config.
        formation_wait_secs: u64,
    },
    /// Controller → worker agent: join refused (version skew, dataset
    /// mismatch, duplicate host, capacity abuse). The connection is
    /// closed after this frame.
    Reject {
        /// Human-readable refusal reason, logged verbatim on the worker.
        reason: String,
    },
    /// Controller → worker agent: the world is formed. Ships the exact
    /// artifacts the fan-out launcher builds for a managed host — the
    /// slim per-host envelope and the relay spec, both hex-encoded JSON
    /// — so the agent spawns its relay + rank children verbatim and
    /// everything downstream of world formation stays unchanged.
    WorldFormed {
        /// Hex-encoded slim per-host envelope JSON (the
        /// `FLODL_INTERNAL_CLUSTER_JSON` payload for rank children).
        envelope_hex: String,
        /// Hex-encoded `RelaySpec` JSON (the `FLODL_INTERNAL_RELAY_JSON`
        /// payload for the host's relay child). `None` when the run has
        /// no coordinator (legacy NCCL routing) — ranks dial the
        /// controller directly and no relay is spawned.
        relay_spec_hex: Option<String>,
    },
    /// Worker agent → controller: one of this host's rank children
    /// exited. Non-zero exits feed the controller's elastic dead-rank
    /// machinery with per-rank granularity (the join connection's EOF
    /// only signals whole-host death).
    RankExited {
        /// Global rank id of the exited child.
        rank: u32,
        /// Process exit code (negative when killed by a signal).
        code: i32,
    },
    /// Controller → worker agent: the run is over before (or instead
    /// of) world formation — window failed quorum, or the launcher
    /// aborted. The agent tears down and exits.
    Abort {
        /// Human-readable cause, logged verbatim on the worker.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
