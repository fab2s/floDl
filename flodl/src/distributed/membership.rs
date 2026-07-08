//! Dial-in membership: the join window that forms a training world.
//!
//! Workers join; the controller admits. A worker agent dials the
//! controller's single mux port with [`CHANNEL_MAGIC_JOIN`], sends a
//! [`JoinMsgWire::Hello`], and is validated immediately — version skew
//! (frame layer), dataset signature, duplicate host, capacity abuse are
//! all rejected loudly at the door. Admitted workers get their global
//! rank ids in admission order, which keeps the rank space contiguous
//! by construction: the first joiner with `k` GPUs holds `0..k`, the
//! next `m`-GPU joiner holds `k..k+m`, and `world_size` is simply the
//! total when the window closes. No holes, no compaction.
//!
//! # Window semantics (quorum knobs)
//!
//! The world is formed once, at start, governed by [`JoinConfig`]:
//!
//! - **`min_rank_start`** — the quorum, counted in ranks. The run
//!   cannot start below it.
//! - **`join_timeout_secs`** — the join window. Reaching quorum early
//!   does NOT close the window: late workers within it are still
//!   admitted. More capacity is never refused while the door is open.
//! - **`target_ranks`** — optional early close the moment it is
//!   reached. When the launcher started exactly N ranks' worth of
//!   workers, the run starts the moment all N are in.
//! - **`max_join_timeout_secs`** — the hard cap. Between the window and
//!   the cap the controller waits for quorum only: the first moment
//!   quorum is met in that range, the world forms. Past the cap with
//!   quorum unmet, the run fails loudly.
//!
//! Both timeouts scale with `FLODL_NET_TIMEOUT_SCALE` like the rest of
//! the deadline set.
//!
//! # Trust: how join frames are keyed
//!
//! The join channel is the one channel a peer may dial without holding
//! the session salt (that is the point — open admission hands the salt
//! out in the [`JoinMsgWire::Accept`] reply). Pre-admission frames are
//! keyed by the trust mode (see [`join_frame_key`]):
//!
//! - **pre-shared** (fan-out rig mode): the session salt. A hello keyed
//!   with anything else fails frame authentication and is rejected —
//!   same guarantee as every other channel.
//! - **open admission** (loopback bind behind sshd, or explicit
//!   `open_admission: true`): an all-zeros key. The MAC still enforces
//!   protocol conformance and integrity, but authentication comes from
//!   the bind scope — reachability through sshd proves possession of an
//!   authorized SSH key.
//!
//! Post-admission frames ([`JoinMsgWire::WorldFormed`] and later) are
//! always keyed with the session salt, binding the connection to the
//! admitted identity.
//!
//! [`CHANNEL_MAGIC_JOIN`]: crate::distributed::wire::CHANNEL_MAGIC_JOIN

use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::distributed::port_mux::StreamSource;
use crate::distributed::wire::{
    CHANNEL_MAGIC_JOIN, ControlFrame, JoinMsgWire, MsgKind, SESSION_SALT_BYTES,
    SessionSalt, expect_channel_magic, salt_to_hex, scaled_deadline_secs,
};
use crate::tensor::{Result, TensorError};

/// Poll cadence of the join window's accept loop.
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// Per-connection budget for the channel magic + hello frame (scaled by
/// `FLODL_NET_TIMEOUT_SCALE`). Honest agents send both immediately
/// after connect.
const JOIN_HANDSHAKE_TIMEOUT_SECS: u64 = 10;

/// Cap on rejected join attempts before the window itself fails: a
/// scanner or a misconfigured fleet hammering the port must not be able
/// to spin the accept loop forever. Mirrors the rendezvous reject cap.
const MAX_REJECTED_JOINS: usize = 1024;

/// Cap on the rank count a single hello may claim. A real host brings a
/// handful of GPUs; an absurd claim is a hostile or corrupt hello and
/// is rejected before it can inflate every per-rank structure
/// downstream.
const MAX_RANKS_PER_JOIN: usize = 256;

/// Quorum knobs governing the join window. See the module docs for the
/// window semantics; [`Self::validate`] enforces the cross-field
/// invariants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinConfig {
    /// Minimum ranks (GPUs) required to start — the quorum.
    pub min_rank_start: usize,
    /// Join window length in seconds (default 300). Quorum reached
    /// early does not close it.
    pub join_timeout_secs: u64,
    /// Optional early close: the window closes the moment this many
    /// ranks are in. Unset means the full window runs.
    pub target_ranks: Option<usize>,
    /// Hard cap in seconds (default 600). Quorum still unmet when it
    /// expires fails the run loudly.
    pub max_join_timeout_secs: u64,
    /// Accept joins without pre-shared-salt authentication. Sound on a
    /// loopback bind (reachability is the authentication); anywhere
    /// else it must be an explicit, loudly-warned choice.
    pub open_admission: bool,
}

impl Default for JoinConfig {
    fn default() -> Self {
        JoinConfig {
            min_rank_start: 1,
            join_timeout_secs: 300,
            target_ranks: None,
            max_join_timeout_secs: 600,
            open_admission: false,
        }
    }
}

impl JoinConfig {
    /// Cross-field validation, loud on violation. Called by
    /// [`MembershipLedger::new`] so no window can run on an
    /// inconsistent config.
    pub fn validate(&self) -> Result<()> {
        if self.min_rank_start == 0 {
            return Err(TensorError::new(
                "cluster join: min_rank_start must be >= 1 (a world of zero \
                 ranks cannot train)",
            ));
        }
        if let Some(target) = self.target_ranks {
            if target < self.min_rank_start {
                return Err(TensorError::new(&format!(
                    "cluster join: target_ranks ({target}) must be >= \
                     min_rank_start ({}) — the early-close target cannot sit \
                     below the quorum",
                    self.min_rank_start,
                )));
            }
        }
        if self.max_join_timeout_secs < self.join_timeout_secs {
            return Err(TensorError::new(&format!(
                "cluster join: max_join_timeout ({}s) must be >= join_timeout \
                 ({}s) — the hard cap cannot expire before the window",
                self.max_join_timeout_secs, self.join_timeout_secs,
            )));
        }
        Ok(())
    }
}

/// The HMAC key for pre-admission join frames: the session salt when it
/// is pre-shared, an all-zeros key under open admission (see the module
/// docs for why that is sound there and only there).
pub(crate) fn join_frame_key(open_admission: bool, salt: &SessionSalt) -> SessionSalt {
    if open_admission {
        [0u8; SESSION_SALT_BYTES]
    } else {
        *salt
    }
}

/// Resolve whether this window runs open admission, per the trust
/// model: a loopback bind makes open admission sound (the only path to
/// the port is through sshd, so reachability proves possession of an
/// authorized SSH key); anywhere else it takes the explicit
/// `open_admission: true` knob, which is honored with a loud warning —
/// an open join port on a LAN/WAN lets any network neighbor
/// *participate in* (poison) training. Explicit selectors error,
/// conventions warn: the knob is explicit consent, so this warns.
pub(crate) fn resolve_open_admission(config: &JoinConfig, bind_is_loopback: bool) -> bool {
    if bind_is_loopback {
        return true;
    }
    if config.open_admission {
        eprintln!(
            "flodl: WARNING: open_admission is enabled on a NON-loopback \
             controller bind — any peer that can reach the join port can \
             join (and therefore influence) this training run. Sound only \
             on a fully trusted network segment; prefer tunneled workers \
             (`tunnel: true`), which flip the bind to loopback and make \
             reachability itself the authentication."
        );
        return true;
    }
    false
}

/// Lifecycle phase of a cluster run, from the join window through
/// training. Serialized into the membership state for observability
/// (`state.json`, `fdl status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterPhase {
    /// Join window open, collecting workers.
    Waiting,
    /// Window closed with quorum; infrastructure starting.
    Forming,
    /// Ranks are training.
    Training,
    /// Run finished cleanly.
    Done,
    /// Run failed (quorum unmet, or a fatal error after formation).
    Failed,
}

/// One admitted worker, as the ledger records it.
#[derive(Debug, Clone)]
pub struct JoinedMember {
    /// Worker host name (unique across the world).
    pub host: String,
    /// Global rank ids assigned at admission (contiguous slice of the
    /// rank space).
    pub ranks: Vec<usize>,
    /// Physical CUDA device ids backing those ranks, in rank order —
    /// resolved on the worker, carried here so the launcher can
    /// synthesize the host's envelope with correct rank↔device pinning.
    pub local_devices: Vec<u8>,
    /// GPU inventory labels from the hello (informational).
    pub gpus: Vec<String>,
    /// libtorch variant label from the hello (informational).
    pub libtorch: String,
    /// Seconds after window open when this worker joined.
    pub joined_at_secs: u64,
}

/// Serializable membership snapshot — the one state source behind the
/// stderr log lines, `state.json`, and `fdl status`.
#[derive(Debug, Clone, Serialize)]
pub struct MembershipSnapshot {
    /// Current lifecycle phase.
    pub phase: ClusterPhase,
    /// Ranks admitted so far.
    pub joined_ranks: usize,
    /// Hosts admitted so far.
    pub joined_hosts: usize,
    /// Quorum knob echo.
    pub min_rank_start: usize,
    /// Early-close knob echo.
    pub target_ranks: Option<usize>,
    /// Seconds until the join window closes (`None` once it has).
    pub window_remaining_secs: Option<u64>,
    /// Seconds until the hard cap (`None` once expired).
    pub cap_remaining_secs: Option<u64>,
    /// Per-host membership.
    pub members: Vec<MemberSnapshot>,
}

/// Per-host slice of [`MembershipSnapshot`].
#[derive(Debug, Clone, Serialize)]
pub struct MemberSnapshot {
    /// Worker host name.
    pub host: String,
    /// Assigned global rank ids.
    pub ranks: Vec<usize>,
    /// Physical CUDA device ids backing those ranks.
    pub local_devices: Vec<u8>,
    /// GPU inventory labels.
    pub gpus: Vec<String>,
    /// libtorch variant label.
    pub libtorch: String,
    /// Seconds after window open when the host joined.
    pub joined_at_secs: u64,
}

/// What the window should do right now, given who has joined and how
/// much time has passed. Produced by [`MembershipLedger::verdict`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WindowVerdict {
    /// Keep accepting.
    Open,
    /// Close the window and form the world. Carries the human reason
    /// for the formation log line.
    Formed(&'static str),
    /// Fail the run loudly.
    Failed(String),
}

/// Pure membership state machine: admission, rank assignment, window
/// verdicts, snapshots. Owns no I/O — [`run_join_window`] drives it
/// against real connections, tests drive it directly.
#[derive(Debug)]
pub(crate) struct MembershipLedger {
    config: JoinConfig,
    /// Dataset signature every hello must match. `None` lets the first
    /// admitted worker set the reference (mirrors the rendezvous
    /// behavior); the launcher passes its own signature when it has
    /// one.
    expected_dataset_sig: Option<[u8; 32]>,
    members: Vec<JoinedMember>,
    next_rank: usize,
}

impl MembershipLedger {
    /// Validate `config` and open an empty ledger.
    pub fn new(
        config: JoinConfig,
        expected_dataset_sig: Option<[u8; 32]>,
    ) -> Result<Self> {
        config.validate()?;
        Ok(MembershipLedger {
            config,
            expected_dataset_sig,
            members: Vec::new(),
            next_rank: 0,
        })
    }

    /// Ranks admitted so far.
    pub fn joined_ranks(&self) -> usize {
        self.next_rank
    }

    /// Admit a hello: validate it, assign the next contiguous rank ids,
    /// record the member. `Err` carries the reject reason sent back to
    /// the worker (the window itself stays open — a bad joiner condemns
    /// its own attempt, never the run).
    pub fn admit(
        &mut self,
        host: &str,
        local_devices: Vec<u8>,
        gpus: Vec<String>,
        libtorch: String,
        dataset_sig: [u8; 32],
        elapsed: Duration,
    ) -> std::result::Result<Vec<usize>, String> {
        if host.trim().is_empty() {
            return Err("host name must be non-empty".to_string());
        }
        let rank_count = local_devices.len();
        if rank_count == 0 {
            return Err("local_devices must be non-empty (a worker with no \
                 ranks cannot train)"
                .to_string());
        }
        if rank_count > MAX_RANKS_PER_JOIN {
            return Err(format!(
                "{rank_count} local devices exceeds the per-worker cap \
                 {MAX_RANKS_PER_JOIN}"
            ));
        }
        {
            let mut seen = [false; 256];
            for d in &local_devices {
                if std::mem::replace(&mut seen[*d as usize], true) {
                    return Err(format!(
                        "duplicate local device {d} — each rank must pin a \
                         distinct physical GPU"
                    ));
                }
            }
        }
        if self.members.iter().any(|m| m.host == host) {
            return Err(format!(
                "host {host:?} already joined this run (duplicate join — a \
                 stale worker from a previous launch, or two workers sharing \
                 one host name)"
            ));
        }
        match self.expected_dataset_sig {
            None => self.expected_dataset_sig = Some(dataset_sig),
            Some(ref expected) if expected != &dataset_sig => {
                return Err(format!(
                    "dataset signature mismatch (worker {}…, run {}…) — the \
                     worker was built against a different dataset shard \
                     layout",
                    hex_prefix(&dataset_sig),
                    hex_prefix(expected),
                ));
            }
            Some(_) => {}
        }
        let ranks: Vec<usize> = (self.next_rank..self.next_rank + rank_count).collect();
        self.next_rank += rank_count;
        self.members.push(JoinedMember {
            host: host.to_string(),
            ranks: ranks.clone(),
            local_devices,
            gpus,
            libtorch,
            joined_at_secs: elapsed.as_secs(),
        });
        Ok(ranks)
    }

    /// Roll back the most recent admission — used when the accept reply
    /// cannot be written (the worker died mid-join), so its rank ids
    /// return to the pool and the space stays contiguous. Loud error if
    /// `host` is not the latest member (retraction is only sound at the
    /// tail).
    pub fn retract_last(&mut self, host: &str) -> Result<()> {
        match self.members.last() {
            Some(m) if m.host == host => {
                let m = self.members.pop().expect("last() was Some");
                self.next_rank -= m.ranks.len();
                Ok(())
            }
            _ => Err(TensorError::new(&format!(
                "cluster join: retract_last({host:?}) is not the most recent \
                 admission — rank ids are assigned contiguously and only the \
                 tail can be returned"
            ))),
        }
    }

    /// Window decision for the current membership at `elapsed` time
    /// since window open. `window` / `cap` are the already-scaled
    /// durations for `join_timeout` / `max_join_timeout`.
    pub fn verdict(
        &self,
        elapsed: Duration,
        window: Duration,
        cap: Duration,
    ) -> WindowVerdict {
        let joined = self.next_rank;
        // Early close on target: the launcher started exactly this much
        // capacity, no reason to wait out the window.
        if let Some(target) = self.config.target_ranks {
            if joined >= target {
                return WindowVerdict::Formed("target ranks reached");
            }
        }
        // Inside the window the door stays open no matter what: quorum
        // reached early does not refuse later capacity.
        if elapsed < window {
            return WindowVerdict::Open;
        }
        if joined >= self.config.min_rank_start {
            return WindowVerdict::Formed("join window closed with quorum");
        }
        // Grace range: window expired below quorum, keep waiting up to
        // the hard cap. The settled-window semantics no longer apply —
        // the first moment quorum is met here, the world forms (the
        // quorum check above fires on the next poll).
        if elapsed < cap {
            return WindowVerdict::Open;
        }
        WindowVerdict::Failed(format!(
            "quorum not met: {joined}/{} ranks joined within the \
             max_join_timeout hard cap ({}s scaled to {}s)",
            self.config.min_rank_start,
            self.config.max_join_timeout_secs,
            cap.as_secs(),
        ))
    }

    /// Observability snapshot at `elapsed` time since window open.
    pub fn snapshot(&self, phase: ClusterPhase, elapsed: Duration) -> MembershipSnapshot {
        let window = Duration::from_secs(scaled_deadline_secs(self.config.join_timeout_secs));
        let cap = Duration::from_secs(scaled_deadline_secs(self.config.max_join_timeout_secs));
        let remaining = |limit: Duration| -> Option<u64> {
            limit.checked_sub(elapsed).map(|d| d.as_secs())
        };
        MembershipSnapshot {
            phase,
            joined_ranks: self.next_rank,
            joined_hosts: self.members.len(),
            min_rank_start: self.config.min_rank_start,
            target_ranks: self.config.target_ranks,
            window_remaining_secs: remaining(window),
            cap_remaining_secs: remaining(cap),
            members: self
                .members
                .iter()
                .map(|m| MemberSnapshot {
                    host: m.host.clone(),
                    ranks: m.ranks.clone(),
                    local_devices: m.local_devices.clone(),
                    gpus: m.gpus.clone(),
                    libtorch: m.libtorch.clone(),
                    joined_at_secs: m.joined_at_secs,
                })
                .collect(),
        }
    }

    /// Consume the ledger into its member list (window closed).
    fn into_members(self) -> Vec<JoinedMember> {
        self.members
    }
}

/// Leading 4 bytes of a signature as hex, for reject messages that
/// should identify without dumping 64 chars.
fn hex_prefix(sig: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(8);
    for b in &sig[..4] {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// One admitted worker with its live join connection. The connection
/// stays open past formation as the host control link: the launcher
/// sends [`JoinMsgWire::WorldFormed`] / [`JoinMsgWire::Abort`] down it,
/// the agent reports [`JoinMsgWire::RankExited`] up it, and EOF means
/// the whole host died.
#[derive(Debug)]
pub(crate) struct AdmittedWorker {
    pub member: JoinedMember,
    pub stream: TcpStream,
}

/// Product of a successful join window.
#[derive(Debug)]
pub(crate) struct FormedWorld {
    /// Admitted workers in admission order (rank ids ascend with the
    /// index).
    pub workers: Vec<AdmittedWorker>,
    /// Total ranks — frozen the moment the window closed.
    pub world_size: usize,
    /// Membership state at formation ([`ClusterPhase::Forming`]) — the
    /// seed of the run-long observability state; the launcher advances
    /// its phase through training and completion.
    pub snapshot: MembershipSnapshot,
}

/// Debug-stream rendering of the membership state — the same JSON the
/// `state.json` endpoint serves, so the log and the HTTP surface can
/// never disagree.
pub(crate) fn log_state(snapshot: &MembershipSnapshot) {
    if crate::log::enabled(crate::log::Verbosity::Debug) {
        if let Ok(js) = serde_json::to_string(snapshot) {
            crate::debug!("cluster membership: state {js}");
        }
    }
}

/// Run the join window: accept connections from `source` (the mux's
/// join leg), admit hellos against a fresh [`MembershipLedger`], and
/// return the formed world when the window closes with quorum.
///
/// `pre_shared_salt` selects the trust mode: `true` requires every
/// hello to authenticate with the session salt; `false` (open
/// admission) accepts zero-keyed hellos and hands the salt out in the
/// accept reply. The caller decides based on bind scope + the
/// `open_admission` knob — this function just executes the choice.
///
/// Every join, reject, and transition emits a stderr log line; `abort`
/// stops the window promptly (launcher failure path).
pub(crate) fn run_join_window(
    source: &StreamSource,
    config: &JoinConfig,
    salt: &SessionSalt,
    pre_shared_salt: bool,
    expected_dataset_sig: Option<[u8; 32]>,
    abort: &AtomicBool,
) -> Result<FormedWorld> {
    let mut ledger = MembershipLedger::new(config.clone(), expected_dataset_sig)?;
    let join_key = join_frame_key(!pre_shared_salt, salt);
    let window = Duration::from_secs(scaled_deadline_secs(config.join_timeout_secs));
    let cap = Duration::from_secs(scaled_deadline_secs(config.max_join_timeout_secs));
    let started = Instant::now();
    let mut admitted: Vec<AdmittedWorker> = Vec::new();
    let mut rejected = 0usize;

    eprintln!(
        "cluster join: window open (quorum {} ranks, target {}, window {}s, \
         cap {}s, admission: {})",
        config.min_rank_start,
        config
            .target_ranks
            .map(|t| t.to_string())
            .unwrap_or_else(|| "none".to_string()),
        window.as_secs(),
        cap.as_secs(),
        if pre_shared_salt { "pre-shared salt" } else { "open" },
    );

    loop {
        let elapsed = started.elapsed();
        match ledger.verdict(elapsed, window, cap) {
            WindowVerdict::Open => {}
            WindowVerdict::Formed(reason) => {
                let world_size = ledger.joined_ranks();
                let snapshot = ledger.snapshot(ClusterPhase::Forming, elapsed);
                let members = ledger.into_members();
                eprintln!(
                    "cluster join: world formed — {world_size} ranks across \
                     {} host(s) after {}s ({reason})",
                    members.len(),
                    elapsed.as_secs(),
                );
                log_state(&snapshot);
                debug_assert_eq!(admitted.len(), members.len());
                return Ok(FormedWorld { workers: admitted, world_size, snapshot });
            }
            WindowVerdict::Failed(why) => {
                let msg = format!("cluster join: FAILED — {why}");
                eprintln!("{msg}");
                abort_admitted(&mut admitted, salt, &why);
                return Err(TensorError::new(&msg));
            }
        }
        if abort.load(Ordering::SeqCst) {
            let why = "launcher aborted before the world formed".to_string();
            abort_admitted(&mut admitted, salt, &why);
            return Err(TensorError::new(&format!("cluster join: {why}")));
        }

        let mut stream = match source.try_accept("cluster join")? {
            Some(s) => s,
            None => {
                std::thread::sleep(ACCEPT_POLL);
                continue;
            }
        };

        // Handshake budget: magic + hello must arrive promptly; a
        // wedged dialer only condemns its own attempt.
        let _ = stream.set_nodelay(true);
        let handshake =
            Duration::from_secs(scaled_deadline_secs(JOIN_HANDSHAKE_TIMEOUT_SECS));
        if stream.set_read_timeout(Some(handshake)).is_err()
            || stream
                .set_write_timeout(Some(crate::distributed::wire::write_stall_timeout()))
                .is_err()
        {
            rejected += 1;
            check_reject_cap(rejected, &ledger)?;
            continue;
        }

        match handle_join_dial(
            &mut stream,
            &mut ledger,
            &join_key,
            salt,
            pre_shared_salt,
            started,
            cap,
        ) {
            Ok(member) => {
                let snap_ranks = ledger.joined_ranks();
                eprintln!(
                    "cluster join: host {:?} joined with ranks {:?} ({} GPU(s), \
                     libtorch {:?}) — {snap_ranks} rank(s) in{}",
                    member.host,
                    member.ranks,
                    member.gpus.len(),
                    member.libtorch,
                    match config.target_ranks {
                        Some(t) => format!(", target {t}"),
                        None => format!(", quorum {}", config.min_rank_start),
                    },
                );
                // Structured twin of the log line: the same snapshot
                // that backs `state.json`, on the debug stream.
                log_state(&ledger.snapshot(ClusterPhase::Waiting, started.elapsed()));
                admitted.push(AdmittedWorker { member, stream });
            }
            Err(why) => {
                eprintln!("cluster join: rejected a join attempt: {why}");
                rejected += 1;
                check_reject_cap(rejected, &ledger)?;
            }
        }
    }
}

/// Reject-cap guard: past [`MAX_REJECTED_JOINS`] the window itself
/// fails loudly — the port is being hammered and honest workers can no
/// longer be told apart from the noise.
fn check_reject_cap(rejected: usize, ledger: &MembershipLedger) -> Result<()> {
    if rejected > MAX_REJECTED_JOINS {
        return Err(TensorError::new(&format!(
            "cluster join: aborting after {rejected} rejected join attempts \
             ({} ranks admitted) — the join port is being hammered by \
             something that is not a flodl worker",
            ledger.joined_ranks(),
        )));
    }
    Ok(())
}

/// One dialer's handshake: consume the channel magic, read + decode the
/// hello, admit it, write the accept reply. `Err` is the reject reason
/// (already sent to the peer when possible); the stream is dropped by
/// the caller on `Err`.
#[allow(clippy::too_many_arguments)]
fn handle_join_dial(
    stream: &mut TcpStream,
    ledger: &mut MembershipLedger,
    join_key: &SessionSalt,
    salt: &SessionSalt,
    pre_shared_salt: bool,
    started: Instant,
    cap: Duration,
) -> std::result::Result<JoinedMember, String> {
    expect_channel_magic(stream, CHANNEL_MAGIC_JOIN, "cluster join")
        .map_err(|e| e.to_string())?;
    let frame = match ControlFrame::read_from(stream, join_key) {
        Ok(Some(f)) => f,
        Ok(None) => return Err("connection closed before hello".to_string()),
        // MAC failure lands here too: under pre-shared admission that is
        // exactly a peer without the salt.
        Err(e) => return Err(e.to_string()),
    };
    if frame.kind != MsgKind::Join {
        let why = format!("expected a Join frame, got {:?}", frame.kind);
        reject(stream, join_key, &why);
        return Err(why);
    }
    let msg: JoinMsgWire = match frame.decode() {
        Ok(m) => m,
        Err(e) => {
            let why = format!("hello decode failed: {e}");
            reject(stream, join_key, &why);
            return Err(why);
        }
    };
    let JoinMsgWire::Hello { host, local_devices, gpus, libtorch, dataset_sig } = msg
    else {
        let why = "first join-channel message must be Hello".to_string();
        reject(stream, join_key, &why);
        return Err(why);
    };
    let ranks = match ledger.admit(
        &host,
        local_devices,
        gpus,
        libtorch,
        dataset_sig,
        started.elapsed(),
    ) {
        Ok(r) => r,
        Err(why) => {
            reject(stream, join_key, &why);
            return Err(format!("host {host:?}: {why}"));
        }
    };
    let accept = JoinMsgWire::Accept {
        ranks: ranks.iter().map(|r| *r as u32).collect(),
        // Open admission is precisely the mode where the joiner has no
        // salt yet; pre-shared mode never re-sends the secret.
        salt_hex: (!pre_shared_salt).then(|| salt_to_hex(salt)),
        // What is left of the hard cap is exactly how long this worker
        // may have to wait for WorldFormed.
        formation_wait_secs: cap.saturating_sub(started.elapsed()).as_secs(),
    };
    let write = ControlFrame::encode(join_key, MsgKind::Join, &accept)
        .and_then(|f| f.write_to(stream));
    if let Err(e) = write {
        // The worker died between hello and accept: return its rank ids
        // to the pool so the space stays contiguous.
        let _ = ledger.retract_last(&host);
        return Err(format!(
            "host {host:?} admitted but the accept reply failed ({e}); \
             admission rolled back"
        ));
    }
    let member = ledger
        .members
        .last()
        .cloned()
        .expect("admit() just pushed this member");
    Ok(member)
}

/// Best-effort reject frame; the peer may already be gone.
fn reject(stream: &mut TcpStream, join_key: &SessionSalt, reason: &str) {
    let msg = JoinMsgWire::Reject { reason: reason.to_string() };
    let _ = ControlFrame::encode(join_key, MsgKind::Join, &msg)
        .and_then(|f| f.write_to(stream));
}

/// Best-effort abort to every admitted worker (window failed / launcher
/// abort). Keyed with the session salt — these workers are admitted, so
/// they hold it.
fn abort_admitted(admitted: &mut [AdmittedWorker], salt: &SessionSalt, reason: &str) {
    let msg = JoinMsgWire::Abort { reason: reason.to_string() };
    for w in admitted.iter_mut() {
        let _ = ControlFrame::encode(salt, MsgKind::Join, &msg)
            .and_then(|f| f.write_to(&mut w.stream));
    }
}

#[cfg(test)]
#[path = "membership_tests.rs"]
mod tests;
