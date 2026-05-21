//! Cluster coordinator: process-model port of the OLD threaded
//! `ddp_run::coordinator::Coordinator`.
//!
//! Owns the per-cluster scheduling state (ElChe, ConvergenceGuard,
//! per-rank wall-time accumulation, sync acknowledgments) and drives
//! averaging decisions for the cluster. Where the OLD design used
//! `mpsc::{Sender, Receiver}` to talk to in-process worker threads,
//! this type talks to remote rank processes over TCP. The state
//! machine and decision logic are ported literally; only the I/O
//! changes.
//!
//! # Architecture
//!
//! ```text
//! launcher process:
//!   ClusterCoordinator::start(bind_addr, world_size, salt, config)
//!     ├── binds control TcpListener
//!     ├── accepts N rank connections (handshake validates salt)
//!     ├── spawns one reader thread per rank
//!     │     reads ControlFrame, decodes TimingMsgWire / MetricsMsgWire,
//!     │     forwards on internal mpsc::Sender
//!     └── owns Vec<TcpStream> (write half, for outbound ControlFrame)
//!
//! caller drives:
//!   coord.tick()  // drain timing mpsc, check_throttle, should_average,
//!                  // trigger_averaging
//! ```
//!
//! # Responsibilities
//!
//! - Owns per-cluster scheduling state: ElChe, [`ConvergenceGuard`],
//!   `steps_since_avg`, `wall_ms_accum`, `last_step_count`,
//!   `nccl_sync_step` / `nccl_ack`, `nccl_sync_divergence` /
//!   `pre_norm` / `post_norm`, `throttled`, `active_count`,
//!   `version`, `avg_count`, `global_step`, `last_nccl_sync_ms`.
//! - Drives averaging decisions: [`ClusterCoordinator::should_average`],
//!   [`ClusterCoordinator::trigger_averaging`] (NCCL),
//!   [`ClusterCoordinator::check_throttle`],
//!   [`ClusterCoordinator::drain_timing`], [`ClusterCoordinator::tick`].
//! - Owns TCP lifecycle: [`ClusterCoordinator::start`] +
//!   [`ClusterCoordinator::shutdown`] (accept loop + per-rank reader
//!   threads).
//! - Drives epoch dispatch (with progressive chunk-pool support),
//!   CPU 3-phase averaging, heartbeat fault detection, metrics
//!   aggregation, and meta-controller observe wiring.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hmac_sha256::HMAC;

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::convergence::{
    self, ConvergenceAction, ConvergenceGuard, NoGuard, TrendGuard,
};
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::{
    ControlFrame, ControlMsgWire, FrameRead, MsgKind, SessionSalt, TimingMsgWire,
};
use crate::tensor::{Result, TensorError};

// ---------------------------------------------------------------------------
// Control-channel handshake
// ---------------------------------------------------------------------------

/// Rank → coordinator handshake magic (mirrors
/// [`crate::distributed::wire::CONTROL_HANDSHAKE_MAGIC_RANK`]).
pub(crate) const CTRL_HS_RANK: u32 = crate::distributed::wire::CONTROL_HANDSHAKE_MAGIC_RANK;

/// Coordinator → rank handshake-ack magic.
pub(crate) const CTRL_HS_ACK: u32 = crate::distributed::wire::CONTROL_HANDSHAKE_MAGIC_ACK;

/// Wire-version used inside the handshake bytes.
pub(crate) const CTRL_HS_VERSION: u32 = crate::distributed::wire::CONTROL_PROTOCOL_VERSION;

/// Handshake byte layout (rank → coordinator):
///
/// ```text
/// u32 magic       = CTRL_HS_RANK
/// u32 version     = CTRL_HS_VERSION
/// u32 rank_id     (0..world_size)
/// u32 world_size  (rank's view; coordinator validates)
/// u64 auth_tag    = first 8 bytes of HMAC-SHA256(salt, hdr[0..16])
/// ```
///
/// Total: 24 bytes. The HMAC proves the rank shares the launcher's
/// session salt; mismatched salts surface here before any control
/// frame round-trip.
const HS_RANK_BYTES: usize = 24;

/// Handshake-ack layout (coordinator → rank):
///
/// ```text
/// u32 magic       = CTRL_HS_ACK
/// u32 version     = CTRL_HS_VERSION
/// u64 auth_tag    = first 8 bytes of HMAC-SHA256(salt, hdr[0..8])
/// ```
///
/// Total: 16 bytes.
const HS_ACK_BYTES: usize = 16;

fn hmac_first8(salt: &SessionSalt, bytes: &[u8]) -> [u8; 8] {
    let full: [u8; 32] = HMAC::mac(bytes, salt.as_slice());
    full[0..8].try_into().unwrap()
}

/// Worker-side companion to [`read_handshake_rank`]. Exported at
/// crate visibility for use by [`crate::distributed::cluster_worker`].
#[allow(dead_code)]
pub(crate) fn write_handshake_rank(
    stream: &mut TcpStream,
    rank_id: u32,
    world_size: u32,
    salt: &SessionSalt,
) -> Result<()> {
    let mut buf = [0u8; HS_RANK_BYTES];
    buf[0..4].copy_from_slice(&CTRL_HS_RANK.to_le_bytes());
    buf[4..8].copy_from_slice(&CTRL_HS_VERSION.to_le_bytes());
    buf[8..12].copy_from_slice(&rank_id.to_le_bytes());
    buf[12..16].copy_from_slice(&world_size.to_le_bytes());
    let tag = hmac_first8(salt, &buf[0..16]);
    buf[16..24].copy_from_slice(&tag);
    stream.write_all(&buf).map_err(|e| {
        TensorError::new(&format!("cluster_coordinator: handshake write: {e}"))
    })
}

fn read_handshake_rank(
    stream: &mut TcpStream,
    expected_world_size: u32,
    salt: &SessionSalt,
) -> Result<u32> {
    let mut buf = [0u8; HS_RANK_BYTES];
    stream.read_exact(&mut buf).map_err(|e| {
        TensorError::new(&format!("cluster_coordinator: handshake read: {e}"))
    })?;
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != CTRL_HS_RANK {
        return Err(TensorError::new(&format!(
            "cluster_coordinator: handshake magic 0x{magic:08x} != 0x{CTRL_HS_RANK:08x}"
        )));
    }
    let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if version != CTRL_HS_VERSION {
        return Err(TensorError::new(&format!(
            "cluster_coordinator: handshake version {version} != {CTRL_HS_VERSION}"
        )));
    }
    let rank_id = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let world_size = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    if world_size != expected_world_size {
        return Err(TensorError::new(&format!(
            "cluster_coordinator: handshake world_size {world_size} != expected {expected_world_size}"
        )));
    }
    let expected_tag = hmac_first8(salt, &buf[0..16]);
    let got_tag: [u8; 8] = buf[16..24].try_into().unwrap();
    if expected_tag != got_tag {
        return Err(TensorError::new(
            "cluster_coordinator: handshake HMAC verification failed; \
             session salt disagreement (rank from a different training session, \
             or wrong key configured)",
        ));
    }
    Ok(rank_id)
}

fn write_handshake_ack(stream: &mut TcpStream, salt: &SessionSalt) -> Result<()> {
    let mut buf = [0u8; HS_ACK_BYTES];
    buf[0..4].copy_from_slice(&CTRL_HS_ACK.to_le_bytes());
    buf[4..8].copy_from_slice(&CTRL_HS_VERSION.to_le_bytes());
    let tag = hmac_first8(salt, &buf[0..8]);
    buf[8..16].copy_from_slice(&tag);
    stream.write_all(&buf).map_err(|e| {
        TensorError::new(&format!("cluster_coordinator: handshake ack write: {e}"))
    })
}

/// Initial value for the coord's three role-rank fields
/// (`checkpoint_role`, `eval_role`, `epoch_callback_role`) at startup.
/// Resolves [`crate::distributed::ddp_run::EpochCallbackPolicy`] using
/// only the information available at construction time:
/// - `Rank(n)` → `n` (clamped to `0..world_size`).
/// - `Fastest` → 0 (lowest rank as an uncalibrated default; the coord
///   re-resolves to the actual smoothed-ms_per_batch winner once the
///   first ElChe sample lands, and on every subsequent rank death).
///
/// World-size 0 is treated as a config bug elsewhere; this helper
/// returns 0 in that case rather than panicking — `start_from_listener`
/// gates on `world_size >= 1` before reaching here in production.
fn initial_callback_role(
    policy: crate::distributed::ddp_run::EpochCallbackPolicy,
    world_size: usize,
) -> usize {
    match policy {
        crate::distributed::ddp_run::EpochCallbackPolicy::Rank(n) => {
            if world_size == 0 { 0 } else { n.min(world_size - 1) }
        }
        crate::distributed::ddp_run::EpochCallbackPolicy::Fastest => 0,
    }
}

// ---------------------------------------------------------------------------
// ClusterCoordinator
// ---------------------------------------------------------------------------

/// Configuration for [`ClusterCoordinator::start`]. Carries the
/// fields the controller needs to drive NCCL/CPU averaging, ElChe
/// ownership, ConvergenceGuard, dispatch, and the user-callback
/// surface ([`epoch_fn`], [`checkpoint_fn`], [`metrics_fn`],
/// [`eval_fn`], etc.).
///
/// [`epoch_fn`]: crate::distributed::ddp_run::EpochFn
/// [`checkpoint_fn`]: crate::distributed::ddp_run::CheckpointFn
/// [`metrics_fn`]: crate::distributed::ddp_run::MetricsFn
/// [`eval_fn`]: crate::distributed::ddp_run::EvalFn
pub struct ClusterCoordinatorConfig {
    pub policy: ApplyPolicy,
    pub backend: AverageBackend,
    pub world_size: usize,
    pub el_che: ElChe,
    /// Boxed convergence guard. Defaults to [`TrendGuard::default()`]
    /// when omitted in the builder; set [`NoGuard`] to disable.
    pub convergence_guard: Box<dyn ConvergenceGuard>,
    /// Allow ElChe anchor relax-up on Stable convergence verdicts.
    pub elche_relax_up: bool,
    /// Initial max-overshoot (Async-only). Auto-tuned upward on Stable
    /// verdicts; reset to `overshoot_initial` on NudgeDown.
    pub overshoot_initial: usize,
    pub overshoot_ceiling: usize,
    /// True when the user did not set `max_overshoot` explicitly; lets
    /// `trigger_averaging` adjust the bound on convergence verdicts.
    pub overshoot_auto: bool,

    /// Total samples in the dataset; basis for partition sizing in
    /// [`ClusterCoordinator::dispatch_epoch`]. Default 0 (caller must
    /// set via [`Self::total_samples`] before dispatching epochs).
    pub total_samples: usize,
    /// Batch size; consumed by the progressive chunk-pool dispatch
    /// path to size per-chunk batches.
    pub batch_size: usize,
    /// Total number of epochs to train; informs `dispatch_epoch`'s
    /// out-of-range guard. Default 0 = unbounded (caller controls).
    pub num_epochs: usize,
    /// User-specified per-rank partition ratios. When `Some`, takes
    /// precedence over ElChe-derived throughput sizing. Length must
    /// equal `world_size` if set.
    pub partition_ratios: Option<Vec<f64>>,
    /// Enable the LR-aware meta-controller above ElChe. Default: false
    /// (off; opt-in). When enabled, a
    /// [`crate::distributed::lr_event_meta::LrEventMeta`] is constructed
    /// and held by the coordinator; per-rank LR updates from
    /// [`TimingMsgWire::LrUpdate`] populate `last_lr_per_rank`, and the
    /// meta is consulted after every averaging-cycle guard verdict
    /// (see the coord's private `observe_meta` hook).
    /// `MetaAction::NudgeDown` dispatches to
    /// [`crate::distributed::ddp::ElChe::nudge_anchor_down`].
    pub meta_controller: bool,

    /// Shared dead-rank ledger (with the cluster controller). When the
    /// coord declares a rank dead (stale heartbeat), it sets the
    /// ledger flag and shuts down the rank's controller-side stream
    /// so any in-flight AllReduce releases with surviving ranks only.
    /// Pass the same Arc clone to
    /// [`crate::distributed::controller::ClusterController::start_with_dead_ranks`].
    /// `None` = elastic-membership disabled.
    pub dead_ranks: Option<Arc<crate::distributed::controller::DeadRanks>>,

    /// Heartbeat staleness threshold (seconds). If a rank's last
    /// TimingMsg-frame arrival is older than this, the coord declares
    /// the rank dead. Default 30s. Ignored when `dead_ranks` is None.
    pub heartbeat_timeout_secs: u64,

    /// Wall-budget (seconds) for an in-flight NCCL re-rendezvous to
    /// complete. Default 5s. On expiry, the coord retries from the
    /// next survivor in the rendezvous's `survivors_ordered`
    /// (excluding already-tried + now-dead ranks). When the candidate
    /// pool is exhausted the fallback is `dispatch_shutdown_with_save`.
    /// Tunable so tests can trigger the retry in well under 1s instead
    /// of waiting the production default.
    pub rendezvous_timeout_secs: u64,

    /// Global ranks running on the same host as the coordinator
    /// process (the launcher's host in production, the test
    /// process in tests). NCCL re-rendezvous prefers picking the
    /// UID generator from this set when any are alive — same-host
    /// rank is same-process latency for the UID hand-off and has
    /// lower correlated-failure risk than a network peer. Defaults
    /// to empty (no preference; picker falls back to fastest
    /// surviving network rank by `wall_ms_accum / steps_since_avg`,
    /// breaking ties by lowest global rank).
    pub local_ranks: Vec<usize>,

    /// Threshold for declaring a cluster run unrecoverable.
    ///
    /// When the count of dead ranks reaches this limit, the coord
    /// broadcasts a save-and-shutdown signal to all survivors so they
    /// can persist model + optimizer + meta state to disk before
    /// exiting (rather than hanging indefinitely waiting for a
    /// re-rendezvous that cannot complete).
    ///
    /// `None` = no user-configured threshold; backend hard limits
    /// still apply (NCCL: lone survivor triggers, since NCCL requires
    /// `world_size >= 2`; CPU: all-dead triggers). Defaults to `None`.
    pub max_failure: Option<crate::distributed::max_failure::MaxFailureThreshold>,

    /// Checkpoint bundle stem for unrecoverable-failure persistence.
    ///
    /// When the coord broadcasts `ShutdownWithSave`, workers write a
    /// bundle (`.fdl` model, `.optim` optimizer state, `.meta.json`
    /// trajectory) with this stem. See
    /// [`crate::distributed::CheckpointBundle`] for path derivation.
    /// `None` in standalone-coord tests; production cluster builders
    /// require this to be set (loud error at `builder.run()` time).
    pub save_path: Option<String>,

    /// Cadence for `epoch_fn`-equivalent user checkpoint callback. When
    /// set, the coord emits
    /// [`crate::distributed::wire::ControlMsgWire::Checkpoint`] every
    /// `checkpoint_every` epochs (right before dispatching the next
    /// epoch plan). Workers handle this on the rank chosen by
    /// [`crate::distributed::ddp_run::EpochCallbackPolicy`]; others see
    /// no-op. `None` or `0` = disabled.
    pub checkpoint_every: Option<usize>,

    /// User-supplied per-epoch metrics callback. Fires on the
    /// coordinator after [`crate::distributed::wire::MetricsMsgWire`]
    /// frames from every alive rank have been aggregated into
    /// [`crate::distributed::ddp_run::EpochMetrics`]. `None` = no
    /// callback wired; aggregation still happens (used internally by
    /// the elastic-balancer + future polling surfaces).
    pub metrics_fn: Option<crate::distributed::ddp_run::MetricsFn>,

    /// Optional sink for aggregated
    /// [`crate::distributed::ddp_run::EpochMetrics`]. Populated by the
    /// launcher trampoline so the user's
    /// [`crate::distributed::DdpHandle::next_metrics`] polling loop
    /// drains aggregates as they're produced. Fed in parallel with
    /// `metrics_fn` — both callbacks/sinks receive the same value.
    /// `None` skips the sink emit.
    pub metrics_sink_tx: Option<mpsc::Sender<crate::distributed::ddp_run::EpochMetrics>>,

    /// User-supplied eval-result callback (controller-side). Fires
    /// when the chosen rank's eval pass returns a scalar metric over
    /// the wire. `None` = no callback wired; result is logged to
    /// stderr instead.
    pub eval_result_fn: Option<crate::distributed::ddp_run::EvalResultFn>,

    /// Eval cadence (in epochs). `Some(n)` triggers an eval dispatch
    /// every `n` epochs from `dispatch_epoch`. `None` or `0` disables.
    pub eval_every_epochs: Option<usize>,

    /// Which rank should fire user-supplied per-epoch callbacks
    /// (`epoch_fn` / `checkpoint_fn` / `eval_fn`). Default
    /// [`EpochCallbackPolicy::Rank(0)`]; setting
    /// [`EpochCallbackPolicy::Fastest`] makes the coord runtime-
    /// resolve the role from ElChe's per-rank smoothed throughput,
    /// with sticky retention across cadences and re-resolution only
    /// on rank death. See `#28b` design notes.
    ///
    /// [`EpochCallbackPolicy::Rank(0)`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Rank
    /// [`EpochCallbackPolicy::Fastest`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Fastest
    pub epoch_callback_policy: crate::distributed::ddp_run::EpochCallbackPolicy,

    /// Progressive dispatch toggle. `None` = auto (true for
    /// `Cadence` / `Async`, false for `Sync`). `Some(true)` or
    /// `Some(false)` is the explicit user override from
    /// [`super::ddp_run::DdpRunConfig::progressive_dispatch`]. When
    /// enabled, the coordinator streams work in small chunks
    /// adapting to measured throughput instead of dispatching one
    /// full per-rank partition per epoch.
    pub progressive: Option<bool>,

    /// Resume kickoff: starting epoch for the launcher's initial
    /// `dispatch_epoch(start_epoch)` call. Default `0` (fresh run).
    /// Set to `meta.epoch` from a loaded
    /// [`crate::distributed::CheckpointMeta`] sidecar to resume at the
    /// epoch a prior run was saved at. Carried separately from
    /// `start_elche_state` because the launcher reads it before
    /// `ClusterCoordinator::start` consumes the config.
    pub start_epoch: usize,

    /// Resume kickoff: initial `global_step` value. Default `0`. The
    /// scheduler's batch-position offset; preserved across resume so
    /// LR-warmup curves and decay schedules pick up where they left off.
    pub start_global_step: usize,

    /// Resume kickoff: initial `avg_count` value. Default `0`. The sync
    /// round counter; preserved across resume for telemetry continuity.
    /// Not load-bearing for trajectory math.
    pub start_avg_count: u64,

    /// Resume kickoff: optional [`crate::distributed::ElCheState`]
    /// snapshot from a prior run's `.meta.json`. When `Some`, the coord
    /// constructor calls `ElChe::restore_from_state(...)` on the
    /// user-built `el_che` after applying user knobs, layering saved
    /// trajectory state (anchor, anchor_rank, phase, calibration_count,
    /// trust-window seed) on top of the user's config. `None` =
    /// fresh-start (no ElChe state to restore).
    pub start_elche_state: Option<crate::distributed::ElCheState>,
}

impl ClusterCoordinatorConfig {
    /// Construct with sensible defaults: TrendGuard with default
    /// threshold, no anchor relax-up, overshoot_initial=3, ceiling=15.
    pub fn new(
        policy: ApplyPolicy,
        backend: AverageBackend,
        world_size: usize,
        el_che: ElChe,
    ) -> Self {
        ClusterCoordinatorConfig {
            policy,
            backend,
            world_size,
            el_che,
            convergence_guard: Box::new(TrendGuard::default()),
            elche_relax_up: false,
            overshoot_initial: 3,
            overshoot_ceiling: 15,
            overshoot_auto: true,
            total_samples: 0,
            batch_size: 1,
            num_epochs: 0,
            partition_ratios: None,
            meta_controller: false,
            dead_ranks: None,
            heartbeat_timeout_secs: 30,
            rendezvous_timeout_secs: NCCL_RENDEZVOUS_TIMEOUT_SECS,
            local_ranks: Vec::new(),
            max_failure: None,
            save_path: None,
            checkpoint_every: None,
            metrics_fn: None,
            metrics_sink_tx: None,
            eval_result_fn: None,
            eval_every_epochs: None,
            epoch_callback_policy:
                crate::distributed::ddp_run::EpochCallbackPolicy::default(),
            progressive: None,
            start_epoch: 0,
            start_global_step: 0,
            start_avg_count: 0,
            start_elche_state: None,
        }
    }

    /// Resume builder: stamp the loaded
    /// [`crate::distributed::CheckpointMeta`] trajectory (epoch +
    /// global_step + sync_round + optional ElChe state) onto this
    /// config. The launcher reads `start_epoch` to drive its kickoff
    /// `dispatch_epoch(start_epoch)`; the coord constructor consumes
    /// the rest. `trend_history` inside the ElCheState is consumed by
    /// the convergence-guard build path in the orchestrator (the coord
    /// itself doesn't carry guard state — the boxed guard is already
    /// rebuilt with restored history before reaching this config).
    pub fn resume_from_meta(
        mut self,
        meta: &crate::distributed::CheckpointMeta,
    ) -> Self {
        self.start_epoch = meta.epoch;
        self.start_global_step = meta.global_step;
        self.start_avg_count = meta.sync_round;
        self.start_elche_state = meta.elche_state.clone();
        self
    }

    /// Attach the user-supplied eval-result callback (controller-side).
    pub fn eval_result_fn(
        mut self,
        f: crate::distributed::ddp_run::EvalResultFn,
    ) -> Self {
        self.eval_result_fn = Some(f);
        self
    }

    /// Eval cadence in epochs. `0` disables.
    pub fn eval_every_epochs(mut self, n: usize) -> Self {
        self.eval_every_epochs = if n == 0 { None } else { Some(n) };
        self
    }

    /// Set the [`EpochCallbackPolicy`] that decides which rank fires
    /// per-epoch user callbacks (`epoch_fn`, `checkpoint_fn`,
    /// `eval_fn`). Default [`EpochCallbackPolicy::Rank(0)`].
    ///
    /// [`EpochCallbackPolicy`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy
    /// [`EpochCallbackPolicy::Rank(0)`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy::Rank
    pub fn epoch_callback_policy(
        mut self,
        policy: crate::distributed::ddp_run::EpochCallbackPolicy,
    ) -> Self {
        self.epoch_callback_policy = policy;
        self
    }

    /// Override progressive-dispatch mode. `None` (default) follows
    /// the auto rule: true for `Cadence` / `Async`, false for `Sync`.
    /// `Some(true)` / `Some(false)` is an explicit override. See
    /// [`Self::progressive`].
    pub fn progressive(mut self, enabled: bool) -> Self {
        self.progressive = Some(enabled);
        self
    }

    /// Attach the user-supplied per-epoch metrics callback.
    pub fn metrics_fn(mut self, f: crate::distributed::ddp_run::MetricsFn) -> Self {
        self.metrics_fn = Some(f);
        self
    }

    /// Attach an `EpochMetrics` sink. The coord clones each aggregated
    /// metric into both this sink and the optional `metrics_fn`
    /// callback. Used by the launcher trampoline to wire the user's
    /// `DdpHandle::next_metrics()` polling loop.
    pub fn metrics_sink_tx(
        mut self,
        tx: mpsc::Sender<crate::distributed::ddp_run::EpochMetrics>,
    ) -> Self {
        self.metrics_sink_tx = Some(tx);
        self
    }

    /// Cadence (in epochs) for user-supplied `checkpoint_fn` invocation.
    /// `None` or `0` disables. See [`Self::checkpoint_every`].
    pub fn checkpoint_every(mut self, every: usize) -> Self {
        self.checkpoint_every = if every == 0 { None } else { Some(every) };
        self
    }

    pub fn total_samples(mut self, n: usize) -> Self {
        self.total_samples = n;
        self
    }

    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    pub fn partition_ratios(mut self, ratios: Option<Vec<f64>>) -> Self {
        self.partition_ratios = ratios;
        self
    }

    pub fn with_convergence_guard(
        mut self,
        guard: Box<dyn ConvergenceGuard>,
    ) -> Self {
        self.convergence_guard = guard;
        self
    }

    pub fn no_divergence_guard(mut self) -> Self {
        self.convergence_guard = Box::new(NoGuard);
        self
    }

    pub fn elche_relax_up(mut self, enabled: bool) -> Self {
        self.elche_relax_up = enabled;
        self
    }

    pub fn overshoot(mut self, initial: usize, ceiling: usize, auto: bool) -> Self {
        self.overshoot_initial = initial;
        self.overshoot_ceiling = ceiling;
        self.overshoot_auto = auto;
        self
    }

    /// Enable the LR-aware meta-controller above ElChe. Default: false.
    ///
    /// When enabled, a [`crate::distributed::lr_event_meta::LrEventMeta`]
    /// is constructed by [`ClusterCoordinator::start_from_listener`] and
    /// observed after every averaging-cycle guard verdict. Ports the
    /// OLD `CoordinatorBuilder::meta_controller` opt-in (see
    /// [`crate::distributed::lr_event_meta`] for the design and rollout
    /// stages).
    pub fn meta_controller(mut self, enabled: bool) -> Self {
        self.meta_controller = enabled;
        self
    }

    /// Share a dead-rank ledger with the cluster controller. Required
    /// for elastic-membership (rank-death-survives-the-run) semantics.
    /// Pass the same `Arc<DeadRanks>` to
    /// [`crate::distributed::controller::ClusterController::start_with_dead_ranks`].
    pub fn dead_ranks(
        mut self,
        ledger: Arc<crate::distributed::controller::DeadRanks>,
    ) -> Self {
        self.dead_ranks = Some(ledger);
        self
    }

    /// Override the heartbeat staleness threshold. Default 30s.
    pub fn heartbeat_timeout_secs(mut self, secs: u64) -> Self {
        self.heartbeat_timeout_secs = secs;
        self
    }

    /// Override the NCCL re-rendezvous wall-budget. See
    /// [`Self::rendezvous_timeout_secs`].
    pub fn rendezvous_timeout_secs(mut self, secs: u64) -> Self {
        self.rendezvous_timeout_secs = secs;
        self
    }

    /// Mark ranks running on the same host as the coord process.
    /// The NCCL re-rendezvous picker prefers these when picking a
    /// UID generator (local same-process latency + lower correlated
    /// failure risk). Used by the launcher to pin its local GPU
    /// rank(s); standalone-coord tests typically leave this empty.
    pub fn local_ranks(mut self, ranks: Vec<usize>) -> Self {
        self.local_ranks = ranks;
        self
    }

    /// Set the unrecoverable-failure threshold. When the dead-rank
    /// count reaches this limit, the coord broadcasts a save-and-
    /// shutdown signal to survivors so they persist state before exit.
    ///
    /// Backend hard limits apply regardless (NCCL needs `world_size >= 2`
    /// for a comm; CPU needs at least 1 survivor).
    pub fn max_failure(
        mut self,
        threshold: crate::distributed::max_failure::MaxFailureThreshold,
    ) -> Self {
        self.max_failure = Some(threshold);
        self
    }

    /// Set the checkpoint bundle stem for unrecoverable-failure
    /// persistence. Workers consult [`crate::distributed::CheckpointBundle`]
    /// to derive `<save_path>.fdl`, `<save_path>.optim`,
    /// `<save_path>.meta.json` on `ShutdownWithSave`.
    pub fn save_path(mut self, path: impl Into<String>) -> Self {
        self.save_path = Some(path.into());
        self
    }
}

/// CPU-backend averaging state machine. Restores cycle-1 guard-verdict
/// correctness on the CPU path: the worker bridges compute the
/// AllReduce + weight-space divergence and emit
/// [`crate::distributed::wire::TimingMsgWire::SyncAck`] with the
/// divergence triple; the coordinator now waits for all of those
/// SyncAcks (via this state) before calling
/// [`ClusterCoordinator::finish_averaging_cpu`], so the guard sees real
/// divergence instead of the all-Nones sentinel that
/// `unwrap_or(0.0)` collapses to zero.
///
/// **Wait policy:** the coordinator waits **indefinitely** for every
/// rank's SyncAck. Dropping a CPU averaging cycle is a correctness
/// violation for Local SGD (per-rank drift accumulates super-linearly
/// across missed rendezvous points), so the only safe response to a
/// stalled rank is to keep waiting. **Liveness detection lives outside
/// the averaging path**: heartbeats feed the coordinator independently
/// and surface dead ranks as fatal training errors. Slow (but live)
/// ranks are absorbed by ElChe on the next cycle, which rebalances
/// [`crate::distributed::ddp::ElChe::batch_counts`] from the observed
/// wall-time. Confirmed rank death triggers elastic averaging via
/// rendezvous rebuild on the shrunken survivor cohort; the dead rank's
/// remaining partition is resharded onto survivors via
/// [`ControlMsgWire::ExtendPartition`].
///
/// NCCL backend keeps the synchronous trigger → finish pattern (OLD
/// `Coordinator::finish_averaging_nccl` parity).
#[derive(Debug)]
enum CpuAvgState {
    /// No averaging cycle in flight.
    Idle,
    /// `trigger_averaging` broadcast `RequestParams` and is waiting for
    /// every rank's bridge SyncAck to populate `nccl_sync_divergence` /
    /// `nccl_sync_pre_norm` / `nccl_sync_post_norm`.
    /// `poll_cpu_averaging` transitions back to `Idle` once `nccl_ack`
    /// is all-true (finalizes the cycle). No deadline — see the
    /// type-level docstring above for the rationale.
    Pending,
}

/// Hard wall-budget for an in-flight NCCL re-rendezvous to complete.
/// On expiry, [`ClusterCoordinator::check_rendezvous_timeout`] retries
/// from the next survivor in `survivors_ordered` (excluding now-dead +
/// already-tried generators); when the candidate pool is exhausted the
/// coord falls back to `dispatch_shutdown_with_save`.
///
/// 5s sits well above any realistic `ncclGetUniqueId` + TCP round-trip
/// (sub-second in practice) while staying well below the 30s heartbeat
/// timeout that catches the "generator silently dies" case via a
/// different path.
const NCCL_RENDEZVOUS_TIMEOUT_SECS: u64 = 5;

/// In-flight NCCL re-rendezvous bookkeeping. Created when the coord
/// declares one or more dead ranks on the NCCL path; cleared when the
/// chosen generator rank ships back a fresh `NcclUniqueId` and the
/// coord broadcasts [`crate::distributed::wire::ControlMsgWire::NewNcclSession`]
/// to every survivor.
#[derive(Debug)]
struct NcclRendezvousPending {
    /// Rank we sent `RequestNewNcclId` to. The coord drops any
    /// `NewNcclIdGenerated` whose `rank` field doesn't match.
    generator_rank: usize,
    /// Survivor ranks ordered by ascending global index, captured at
    /// initiation time. Consumed by
    /// [`ClusterCoordinator::check_rendezvous_timeout`] when the chosen
    /// generator dies or stops responding: the retry path picks the
    /// next candidate from this set after filtering out dead ranks
    /// and `tried_generators`.
    survivors_ordered: Vec<usize>,
    /// When the current `RequestNewNcclId` went out. Refreshed every
    /// retry. `check_rendezvous_timeout` triggers a retry when
    /// `initiated_at.elapsed() > NCCL_RENDEZVOUS_TIMEOUT_SECS`.
    initiated_at: Instant,
    /// Ranks we already asked and that did NOT respond within the
    /// timeout (or that died before responding). Excluded from the
    /// candidate pool on subsequent retries so a single slow rank
    /// doesn't loop.
    tried_generators: Vec<usize>,
}

/// Process-model coordinator: ports the OLD threaded
/// `ddp_run::coordinator::Coordinator` to talk to remote rank
/// processes over TCP.
///
/// Hand off control of one TCP control channel + one reader thread per
/// rank. Drive the state machine via [`Self::tick`] from the
/// containing thread.
pub struct ClusterCoordinator {
    // --- Static config ---
    policy: ApplyPolicy,
    backend: AverageBackend,
    world_size: usize,
    overshoot_initial: usize,
    overshoot_ceiling: usize,
    overshoot_auto: bool,
    elche_relax_up: bool,

    // --- Scheduling state (ported literally) ---
    el_che: ElChe,
    convergence_guard: Box<dyn ConvergenceGuard>,
    version: u64,
    avg_count: u64,
    global_step: usize,
    /// Set once ElChe has its first timing report.
    calibrated: bool,
    /// World_size minus exited workers.
    active_count: usize,
    /// Async-only: max batches a rank can run past the planned sync.
    max_overshoot: usize,

    /// Per-rank steps since the last averaging cycle.
    steps_since_avg: Vec<usize>,
    /// Per-rank wall-clock ms accumulated since the last averaging cycle.
    wall_ms_accum: Vec<f64>,
    /// Per-rank most-recent batch duration (ms).
    last_batch_ms: Vec<f64>,
    /// Per-rank most-recent worker step counter.
    last_step_count: Vec<usize>,
    /// Per-rank: `last_step_count` snapshot at the time SyncNow was sent.
    nccl_sync_step: Vec<usize>,
    /// Per-rank: true once a post-sync timing message has arrived.
    nccl_ack: Vec<bool>,
    /// Per-rank: weight-space divergence reported in the last SyncAck.
    nccl_sync_divergence: Vec<Option<f64>>,
    /// Per-rank: pre-AllReduce L2 norm from the last SyncAck.
    nccl_sync_pre_norm: Vec<Option<f64>>,
    /// Post-AllReduce L2 norm (identical across ranks; populated by the
    /// first rank's SyncAck).
    nccl_sync_post_norm: Option<f64>,
    /// Per-rank: True if a Throttle has been sent and not yet cleared.
    throttled: Vec<bool>,
    /// Wall-time (ms) of the last completed NCCL sync; fed to ElChe as
    /// `sync_ms` on the next `report_timing` call.
    last_nccl_sync_ms: f64,
    /// Instant the most recent SyncNow was emitted.
    nccl_sync_start: Option<Instant>,

    /// LR-aware meta-controller above ElChe. `None` when
    /// [`ClusterCoordinatorConfig::meta_controller`] is `false` (default).
    /// Observed after every averaging-cycle guard verdict via
    /// [`Self::observe_meta`]; dispatches
    /// [`crate::distributed::lr_event_meta::MetaAction::NudgeDown`] to
    /// [`crate::distributed::ddp::ElChe::nudge_anchor_down`].
    lr_event_meta: Option<crate::distributed::lr_event_meta::LrEventMeta>,
    /// Most recent learning rate observed per rank via
    /// [`TimingMsgWire::LrUpdate`]. Indexed by rank. `None` until the
    /// first LR update from that rank arrives.
    last_lr_per_rank: Vec<Option<f64>>,

    /// CPU-backend averaging state machine. See [`CpuAvgState`].
    cpu_avg_state: CpuAvgState,
    /// Shared dead-rank ledger with the controller. `None` when
    /// elastic membership is disabled (legacy / NCCL-only setups).
    /// Mutating side: [`Self::check_dead_ranks`] calls `declare_dead`
    /// on heartbeat staleness; reading side: should_average,
    /// poll_cpu_averaging, and other gates skip dead ranks.
    dead_ranks: Option<Arc<crate::distributed::controller::DeadRanks>>,
    /// Heartbeat staleness threshold copied from config.
    heartbeat_timeout_secs: u64,
    /// NCCL re-rendezvous wall-budget copied from config. Consumed by
    /// [`Self::check_rendezvous_timeout`].
    rendezvous_timeout_secs: u64,
    /// Per-rank wall-clock of the last TimingMsg frame received from
    /// that rank (any TimingMsgWire variant counts as a liveness
    /// signal — Batch, SyncAck, Heartbeat, SnapshotReady, LrUpdate).
    /// Initialized to `Instant::now()` at coord start; updated by
    /// [`Self::process_timing_msg`] before any other per-message work.
    /// Drives [`Self::check_dead_ranks`].
    last_heartbeat: Vec<Instant>,
    /// Per-rank snapshot of `last_step_count[r]` at the moment
    /// [`Self::dispatch_epoch`] emitted `StartEpoch` to rank `r`.
    /// Computing `last_step_count[r] - last_step_count_at_epoch_start[r]`
    /// at dead-rank-declaration time yields the number of batches rank
    /// `r` has processed in the current epoch, which translates to a
    /// sample-count offset into its partition for the un-processed
    /// remainder to redistribute to survivors via `ExtendPartition`.
    /// Initialized to 0; refreshed every `dispatch_epoch`.
    last_step_count_at_epoch_start: Vec<usize>,
    /// In-flight NCCL re-rendezvous (NCCL backend only). `Some` while
    /// the coord is awaiting a fresh `NcclUniqueId` from the chosen
    /// generator rank; cleared once the new session has been
    /// broadcast to survivors. CPU backend always leaves this `None`.
    nccl_rendezvous_pending: Option<NcclRendezvousPending>,
    /// Ranks co-located with the coord process. NCCL re-rendezvous
    /// UID-generator picker prefers these (Tier 1) over fastest
    /// network rank (Tier 2). Sourced from
    /// [`ClusterCoordinatorConfig::local_ranks`].
    local_ranks: Vec<usize>,

    /// Unrecoverable-failure threshold; copied from
    /// [`ClusterCoordinatorConfig::max_failure`].
    max_failure: Option<crate::distributed::max_failure::MaxFailureThreshold>,

    /// [`EpochCallbackPolicy`] copied from config; drives Fastest
    /// resolution (vs. static `Rank(n)` resolution at startup).
    ///
    /// [`EpochCallbackPolicy`]:
    ///     crate::distributed::ddp_run::EpochCallbackPolicy
    epoch_callback_policy: crate::distributed::ddp_run::EpochCallbackPolicy,
    /// Sticky assignee for `ControlMsgWire::Checkpoint`. Initial value
    /// resolved from `epoch_callback_policy` at construction
    /// (`Rank(n)` → n; `Fastest` → lowest live rank as a uncalibrated
    /// default; ElChe smoothed values take over once the run reaches
    /// the first averaging cycle). Failover on (a) the role's
    /// `CheckpointResult.error` or (b) the role's rank death. v1
    /// dispatches every checkpoint to this rank (rather than
    /// broadcasting + filtering rank-side) so the worker never has to
    /// decide whether it is the checkpointer.
    checkpoint_role: usize,
    /// Sticky assignee for `ControlMsgWire::ExecuteEvalCallback`.
    /// Same resolution + failover semantics as `checkpoint_role`.
    eval_role: usize,
    /// Sticky assignee for `epoch_fn` (worker fires autonomously at
    /// each epoch transition; this is pushed to workers via
    /// `ControlMsgWire::SetEpochCallbackRole` rather than per-event
    /// dispatch because there is no per-epoch event to attach).
    epoch_callback_role: usize,
    /// `true` when the epoch_callback_role value has not yet been
    /// broadcast to workers since the last change. Set on
    /// construction + on every Fastest re-resolution. Cleared after
    /// the next successful broadcast.
    epoch_role_dirty: bool,
    /// Per-version set of ranks that have already attempted +
    /// reported failure for that version. Used to (a) pick the next
    /// untried live rank on retry, and (b) detect exhaustion (all
    /// live ranks tried + failed → give up on this version; existing
    /// `MaxFailureThreshold` governs the longer-term run health).
    checkpoint_tried_ranks:
        std::collections::HashMap<u64, std::collections::HashSet<usize>>,
    /// EWMA of recent successful checkpoint wall-times (ms). Reserved
    /// for v2 rendezvous-aware scheduling (controller aligns
    /// checkpoint dispatch with AllReduce barriers so the cost
    /// overlaps with idle barrier-wait). v1 just records.
    last_checkpoint_elapsed_ms_ewma: Option<f64>,
    /// EWMA of recent successful `eval_fn` wall-times (ms). Consumed by
    /// ElChe's last-batch slack reservation: the firing rank's share
    /// of the trailing batches is reduced so the eval pass absorbs
    /// into the rank's idle slack instead of overrunning the next
    /// sync barrier. Updated on every successful eval result. None
    /// until the first eval reports.
    last_eval_elapsed_ms_ewma: Option<f64>,
    /// EWMA of recent `epoch_fn` wall-times (ms) on the role rank.
    /// Mirrors [`Self::last_eval_elapsed_ms_ewma`] for the
    /// autonomously-fired `epoch_fn` path. Used by the same last-batch
    /// slack reservation mechanism.
    last_epoch_fn_elapsed_ms_ewma: Option<f64>,

    /// `true` once [`Self::dispatch_shutdown_with_save`] has broadcast
    /// `ShutdownWithSave`. Guards against re-broadcasting on subsequent
    /// `check_dead_ranks` ticks — once survivors are persisting state,
    /// repeat broadcasts are noise.
    shutdown_with_save_dispatched: bool,
    /// Per-rank wall-time (ms) from the most recent averaging cycle's
    /// `RequestParams` broadcast to that rank's SyncAck arrival.
    /// Populated by [`Self::process_timing_msg`]'s SyncAck arm.
    ///
    /// **Caveat: barrier-correlated, NOT per-rank capacity.** A rank's
    /// bridge blocks inside the AllReduce barrier until every rank
    /// arrives, so individual lag values converge toward the slowest
    /// rank's contribution. Use this as a *cycle latency* indicator
    /// only — do NOT feed it into ElChe or partition-balancing logic
    /// as a per-rank throughput proxy. Honest per-rank capacity
    /// comes from `wall_ms_accum` / `steps_since_avg` (already excludes
    /// the barrier wait) and [`Self::last_observed_upload_ms`] (the
    /// pre-barrier snapshot+upload marker).
    last_observed_sync_lag_ms: Vec<Option<f64>>,

    /// Per-rank wall-time (ms) from `RequestParams` broadcast to that
    /// rank's [`crate::distributed::wire::TimingMsgWire::SnapshotReady`]
    /// arrival — captured by the SnapshotReady handler in
    /// [`Self::process_timing_msg`].
    ///
    /// Honest per-rank capacity signal: the rank emits SnapshotReady
    /// AFTER snapshot+upload but BEFORE entering the AllReduce barrier
    /// (see the param bridge in `cluster_worker.rs`), so this lag is
    /// not contaminated by the slowest-rank barrier wait that pollutes
    /// `last_observed_sync_lag_ms`.
    ///
    /// Cleared (per-rank slot reset to `None`) at the start of every
    /// new cycle by [`Self::trigger_averaging`] so the values reflect
    /// the in-flight cycle only. `None` means the rank never reported
    /// SnapshotReady this cycle (NCCL backend — there's no
    /// snapshot+upload step — or a dead rank that exited mid-cycle).
    ///
    /// Currently exposed via [`Self::last_observed_upload_ms`] for
    /// telemetry + planned ElChe consumption; the rebalancer is not
    /// yet wired to read this.
    last_observed_upload_ms: Vec<Option<f64>>,

    /// Checkpoint bundle stem for the controller-side `.meta.json`
    /// write on `ShutdownWithSave`. Mirrors
    /// [`ClusterCoordinatorConfig::save_path`] — populated at
    /// construction, never mutated. `None` skips the meta write (and
    /// the worker side skips its own bundle write since
    /// `save_path` flows through `WorkerConfig` from the rank-side
    /// `DdpRunConfig` independently).
    save_path: Option<String>,

    /// Cadence (in epochs) for user-supplied `checkpoint_fn`. When
    /// `Some(n)` and `n > 0`, [`Self::dispatch_epoch`] emits
    /// [`crate::distributed::wire::ControlMsgWire::Checkpoint`] right
    /// before broadcasting the next `StartEpoch` whenever the epoch
    /// boundary lines up with `n`. Workers receive the frame; only the
    /// rank chosen by [`crate::distributed::ddp_run::EpochCallbackPolicy`]
    /// has `checkpoint_fn = Some(...)`, others no-op.
    checkpoint_every: Option<usize>,

    // --- Epoch dispatch ---
    /// Per-rank current epoch (last StartEpoch dispatched).
    rank_epoch: Vec<usize>,
    /// Last globally-aggregated epoch index (all ranks reported).
    /// `None` until the first aggregation.
    last_aggregated_epoch: Option<usize>,
    /// Last epoch index for which `dispatch_epoch` was driven by the
    /// post-aggregate advance hook (see
    /// [`Self::try_advance_or_shutdown_after_aggregate`]). Used to
    /// keep the hook idempotent across ticks — without this, every
    /// tick after aggregation would re-dispatch the next epoch.
    /// `None` for fresh runs; the initial `dispatch_epoch(0)` kickoff
    /// is the launcher's responsibility and does not set this field.
    last_dispatched_epoch: Option<usize>,
    /// Set once [`Self::shutdown_workers`] has been broadcast from the
    /// post-aggregate hook so the broadcast does not fire on every
    /// subsequent tick before the readers observe stream close.
    shutdown_initiated: bool,
    /// Cached epoch plans: computed once per epoch, consistent across
    /// ranks regardless of when the StartEpoch frame goes out.
    epoch_plan_cache: std::collections::HashMap<usize, Vec<crate::distributed::wire::EpochPlanWire>>,
    /// Total samples in the dataset; basis for partition computation.
    total_samples: usize,
    /// Batch size; consumed by the progressive chunk-pool dispatch
    /// path to size per-chunk batches.
    batch_size: usize,
    /// Total number of epochs the trainer asked for.
    num_epochs: usize,
    /// User-specified per-rank partition ratios, if any.
    partition_ratios: Option<Vec<f64>>,

    // --- Channels / threads ---
    /// Reader threads (one per rank) push decoded timing messages here;
    /// the coordinator thread drains via [`Self::drain_timing`].
    timing_rx: mpsc::Receiver<TimingMsgWire>,
    /// Reader threads also push decoded per-epoch metrics here;
    /// the coordinator thread drains via
    /// [`Self::drain_metrics_and_aggregate`] and aggregates into
    /// [`crate::distributed::ddp_run::EpochMetrics`] once each epoch
    /// has reports from every alive rank.
    metrics_rx: mpsc::Receiver<crate::distributed::wire::MetricsMsgWire>,
    /// Per-epoch buffer of arrived MetricsMsg reports. Keyed by epoch.
    /// In non-progressive dispatch each alive rank emits exactly one
    /// message per epoch; in progressive dispatch each rank emits one
    /// per chunk (so many messages per rank per epoch). Aggregation
    /// fires once the readiness condition is met (every alive rank has
    /// reported at least once for non-progressive, or the epoch's
    /// `ChunkPool::is_epoch_done()` returns true for progressive).
    /// `BTreeMap` rather than `HashMap` so progressive aggregation
    /// walks epochs in ascending order, matching the threaded
    /// coordinator's ordering invariant.
    metrics_buffer: std::collections::BTreeMap<
        u64,
        Vec<crate::distributed::ddp_run::MetricsMsg>,
    >,
    /// Per-epoch progressive chunk pools. Created on `dispatch_epoch`
    /// when [`Self::progressive`] is set, drained by
    /// `drain_metrics_and_aggregate` on epoch completion. `BTreeMap`
    /// keeps cross-epoch streaming in ascending order so a fast rank
    /// streaming ahead doesn't aggregate before slower ranks finish
    /// earlier epochs.
    chunk_pools: std::collections::BTreeMap<usize, crate::distributed::chunk_pool::ChunkPool>,
    /// When true, dispatch streams work in small chunks (one
    /// `StartEpoch` per chunk) adapting to measured throughput. When
    /// false, dispatch sends the full per-rank partition once per
    /// epoch (the legacy behaviour). Derived at construction time
    /// from
    /// [`ClusterCoordinatorConfig::progressive`] /
    /// [`super::ddp_run::DdpRunConfig::progressive_dispatch`] /
    /// [`ApplyPolicy`] (auto: true for Cadence/Async, false for Sync).
    progressive: bool,
    /// Minimum chunk size in batches. After calibration, the
    /// throughput-proportional chunk sizer floors at this value so a
    /// rank doesn't get a one-batch chunk that pays per-chunk overhead
    /// without amortising it.
    min_chunk_batches: usize,
    /// User-supplied per-epoch metrics callback (controller-side). Fires
    /// after each successful aggregation. `None` = no callback wired.
    metrics_fn: Option<crate::distributed::ddp_run::MetricsFn>,
    /// Sink for aggregated `EpochMetrics` consumed by
    /// `DdpHandle::next_metrics()`. Populated alongside `metrics_fn`;
    /// both fire on each aggregation. `None` skips the sink emit
    /// (handle either doesn't poll or isn't owned by this process).
    metrics_sink_tx: Option<mpsc::Sender<crate::distributed::ddp_run::EpochMetrics>>,
    /// User-supplied eval-result callback. Fires in
    /// `process_timing_msg` when a `TimingMsgWire::EvalResult` arrives.
    /// `None` = no callback; error path still logs to stderr.
    eval_result_fn: Option<crate::distributed::ddp_run::EvalResultFn>,
    /// Eval cadence (in epochs). `Some(n)` triggers
    /// `ControlMsgWire::ExecuteEvalCallback` every `n` epochs from
    /// `dispatch_epoch`. `None` = disabled.
    eval_every_epochs: Option<usize>,
    /// CUDA device indices per rank, captured at construction for the
    /// `EpochMetrics::device_indices` field. Currently `0..world_size`
    /// (one device per rank, assigned by global_rank); the
    /// `EpochCallbackPolicy` design assumes process-per-rank with one
    /// device per process.
    metrics_device_indices: Vec<u8>,
    /// Outbound control streams (one per rank, write half held here).
    /// Reader thread holds a try-cloned read half.
    control_streams: Vec<TcpStream>,
    /// Reader-thread join handles. Drop on [`Self::shutdown`].
    reader_handles: Vec<Option<JoinHandle<()>>>,
    /// Signals reader threads to stop reading and exit.
    shutdown_flag: Arc<AtomicBool>,
    /// Bound port of the control listener (for tests / diagnostics).
    bound_port: u16,
    /// Session salt — write side uses it for outbound ControlFrames.
    salt: SessionSalt,
}

impl ClusterCoordinator {
    /// Bind a control TcpListener at `bind_addr`, accept exactly
    /// `world_size` rank connections (validating the session salt at
    /// handshake), spawn per-rank reader threads, and return the
    /// configured coordinator.
    ///
    /// Returns `Err` if any handshake fails (loud error: salt mismatch,
    /// magic mismatch, version mismatch, world_size disagreement,
    /// duplicate rank_id).
    pub fn start(
        bind_addr: SocketAddr,
        salt: SessionSalt,
        config: ClusterCoordinatorConfig,
    ) -> Result<Self> {
        let (listener, _port) = Self::bind(bind_addr)?;
        Self::start_from_listener(listener, salt, config)
    }

    /// Bind the control listener without blocking on accept. Useful for
    /// tests that need to publish the bound port before spawning rank
    /// connections (the post-bind accept loop blocks the calling
    /// thread until every rank has connected).
    pub fn bind(bind_addr: SocketAddr) -> Result<(TcpListener, u16)> {
        let listener = TcpListener::bind(bind_addr).map_err(|e| {
            TensorError::new(&format!(
                "cluster_coordinator: bind {bind_addr} failed: {e}"
            ))
        })?;
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: local_addr() failed: {e}"
                ))
            })?
            .port();
        Ok((listener, bound_port))
    }

    /// Accept connections + handshake on a pre-bound listener. Pair
    /// with [`Self::bind`] when the caller needs the port before
    /// blocking on accepts (e.g. tests that spawn rank threads after
    /// publishing the port through a channel).
    pub fn start_from_listener(
        listener: TcpListener,
        salt: SessionSalt,
        config: ClusterCoordinatorConfig,
    ) -> Result<Self> {
        let world_size = config.world_size;
        if world_size == 0 {
            return Err(TensorError::new(
                "cluster_coordinator: world_size must be > 0",
            ));
        }
        let bound_port = listener
            .local_addr()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: local_addr() failed: {e}"
                ))
            })?
            .port();

        // Accept world_size connections, validate handshake, place each
        // at its claimed rank slot. Order-independent.
        let mut streams: Vec<Option<TcpStream>> =
            (0..world_size).map(|_| None).collect();
        let mut connected = 0usize;
        while connected < world_size {
            let (mut stream, _peer) = listener.accept().map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: accept failed: {e}"
                ))
            })?;
            // 10s handshake timeout protects against wedged ranks.
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: set_read_timeout: {e}"
                    ))
                })?;
            let rank_id = read_handshake_rank(&mut stream, world_size as u32, &salt)?;
            let rank_idx = rank_id as usize;
            if rank_idx >= world_size {
                return Err(TensorError::new(&format!(
                    "cluster_coordinator: handshake rank_id {rank_idx} >= world_size {world_size}"
                )));
            }
            if streams[rank_idx].is_some() {
                return Err(TensorError::new(&format!(
                    "cluster_coordinator: duplicate rank_id {rank_idx} connected"
                )));
            }
            write_handshake_ack(&mut stream, &salt)?;
            // Clear the handshake timeout; ControlFrame reads can take
            // arbitrarily long under load.
            stream
                .set_read_timeout(None)
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: clear read_timeout: {e}"
                    ))
                })?;
            streams[rank_idx] = Some(stream);
            connected += 1;
        }
        let mut streams: Vec<TcpStream> = streams.into_iter()
            .map(|s| s.expect("all slots filled by accept loop"))
            .collect();

        // Spawn one reader thread per rank. Each thread holds the read
        // half of a try_clone'd stream; the coordinator owns the write
        // half. ControlFrame::read_from handles HMAC validation per frame.
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let (timing_tx, timing_rx) = mpsc::channel::<TimingMsgWire>();
        let (metrics_tx, metrics_rx) =
            mpsc::channel::<crate::distributed::wire::MetricsMsgWire>();

        let mut reader_handles: Vec<Option<JoinHandle<()>>> = Vec::with_capacity(world_size);
        for (rank, stream) in streams.iter_mut().enumerate() {
            let mut read_half = stream.try_clone().map_err(|e| {
                TensorError::new(&format!(
                    "cluster_coordinator: stream try_clone for rank {rank}: {e}"
                ))
            })?;
            // Reader uses a short timeout to observe shutdown between frames.
            read_half
                .set_read_timeout(Some(Duration::from_millis(250)))
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: reader set_read_timeout: {e}"
                    ))
                })?;
            let tx = timing_tx.clone();
            let mtx = metrics_tx.clone();
            let salt_for_reader = salt;
            let shutdown_for_reader = Arc::clone(&shutdown_flag);
            let handle = thread::Builder::new()
                .name(format!("flodl-coord-reader:r{rank}"))
                .spawn(move || {
                    reader_loop(rank, &mut read_half, &salt_for_reader, &shutdown_for_reader, &tx, &mtx);
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster_coordinator: spawn reader for rank {rank}: {e}"
                    ))
                })?;
            reader_handles.push(Some(handle));
        }
        // Drop the extra senders we cloned for the closures; loop exit
        // depends on every cloned sender being dropped, but that happens
        // automatically when reader threads exit.
        drop(timing_tx);
        drop(metrics_tx);

        // Resume: layer saved trajectory state on top of the user-built
        // ElChe (which carries the user's knobs from this run's
        // DdpRunConfig). When `start_elche_state` is None, the ElChe
        // stays fresh.
        let mut el_che = config.el_che;
        if let Some(ref state) = config.start_elche_state {
            el_che.restore_from_state(state)?;
        }
        // `calibrated` mirrors the post-restore ElChe state: true when
        // any rank has a positive smoothed reading. Matches the
        // invariant the snapshot was taken under.
        let calibrated = config.start_elche_state.is_some()
            && el_che.is_calibrated();
        Ok(ClusterCoordinator {
            policy: config.policy,
            backend: config.backend,
            world_size,
            overshoot_initial: config.overshoot_initial,
            overshoot_ceiling: config.overshoot_ceiling,
            overshoot_auto: config.overshoot_auto,
            elche_relax_up: config.elche_relax_up,
            el_che,
            convergence_guard: config.convergence_guard,
            version: 0,
            avg_count: config.start_avg_count,
            global_step: config.start_global_step,
            calibrated,
            active_count: world_size,
            max_overshoot: config.overshoot_initial,
            steps_since_avg: vec![0; world_size],
            wall_ms_accum: vec![0.0; world_size],
            last_batch_ms: vec![0.0; world_size],
            last_step_count: vec![0; world_size],
            nccl_sync_step: vec![0; world_size],
            nccl_ack: vec![true; world_size],
            nccl_sync_divergence: vec![None; world_size],
            nccl_sync_pre_norm: vec![None; world_size],
            nccl_sync_post_norm: None,
            throttled: vec![false; world_size],
            last_nccl_sync_ms: 0.0,
            nccl_sync_start: None,
            lr_event_meta: if config.meta_controller {
                Some(crate::distributed::lr_event_meta::LrEventMeta::with_default_config())
            } else {
                None
            },
            last_lr_per_rank: vec![None; world_size],
            cpu_avg_state: CpuAvgState::Idle,
            dead_ranks: config.dead_ranks,
            heartbeat_timeout_secs: config.heartbeat_timeout_secs,
            rendezvous_timeout_secs: config.rendezvous_timeout_secs,
            last_heartbeat: vec![Instant::now(); world_size],
            last_step_count_at_epoch_start: vec![0; world_size],
            nccl_rendezvous_pending: None,
            local_ranks: config.local_ranks.clone(),
            max_failure: config.max_failure,
            epoch_callback_policy: config.epoch_callback_policy,
            checkpoint_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            eval_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_callback_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_role_dirty: true,
            checkpoint_tried_ranks: std::collections::HashMap::new(),
            last_checkpoint_elapsed_ms_ewma: None,
            last_eval_elapsed_ms_ewma: None,
            last_epoch_fn_elapsed_ms_ewma: None,
            save_path: config.save_path.clone(),
            checkpoint_every: config.checkpoint_every,
            shutdown_with_save_dispatched: false,
            last_observed_sync_lag_ms: vec![None; world_size],
            last_observed_upload_ms: vec![None; world_size],
            rank_epoch: vec![0; world_size],
            last_aggregated_epoch: None,
            last_dispatched_epoch: None,
            shutdown_initiated: false,
            epoch_plan_cache: std::collections::HashMap::new(),
            total_samples: config.total_samples,
            batch_size: config.batch_size.max(1),
            num_epochs: config.num_epochs,
            partition_ratios: config.partition_ratios,
            timing_rx,
            metrics_rx,
            metrics_buffer: std::collections::BTreeMap::new(),
            chunk_pools: std::collections::BTreeMap::new(),
            // Resolve progressive: explicit override wins, otherwise
            // auto-on for Cadence/Async, off for Sync (matches the
            // threaded coordinator's default).
            progressive: config.progressive.unwrap_or(
                !matches!(config.policy, ApplyPolicy::Sync),
            ),
            // Floor for proportional chunk sizing after calibration —
            // matches the threaded coordinator's default.
            min_chunk_batches: 4,
            metrics_fn: config.metrics_fn.clone(),
            metrics_sink_tx: config.metrics_sink_tx.clone(),
            eval_result_fn: config.eval_result_fn.clone(),
            eval_every_epochs: config.eval_every_epochs,
            metrics_device_indices: (0..world_size as u8).collect(),
            control_streams: streams,
            reader_handles,
            shutdown_flag,
            bound_port,
            salt,
        })
    }

    // -----------------------------------------------------------------
    // Public accessors (mirror the OLD coordinator's getters)
    // -----------------------------------------------------------------

    pub fn bound_port(&self) -> u16 {
        self.bound_port
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    pub fn steps_since_avg(&self) -> &[usize] {
        &self.steps_since_avg
    }

    pub fn avg_count(&self) -> u64 {
        self.avg_count
    }

    pub fn global_step(&self) -> usize {
        self.global_step
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// `true` after [`Self::dispatch_shutdown_with_save`] has fired
    /// (max_failure threshold breached or backend hard limit hit).
    /// Test-only accessor; the flag is internal state.
    #[cfg(test)]
    pub(crate) fn shutdown_with_save_dispatched(&self) -> bool {
        self.shutdown_with_save_dispatched
    }

    /// Test-only peek at the current rendezvous-pending generator
    /// rank. `None` when no rendezvous is in flight (steady-state OR
    /// after exhaustion → ShutdownWithSave). Drives the retry-path
    /// tests that need to observe generator-rank transitions across
    /// successive ticks.
    #[cfg(test)]
    pub(crate) fn rendezvous_pending_generator(&self) -> Option<usize> {
        self.nccl_rendezvous_pending
            .as_ref()
            .map(|p| p.generator_rank)
    }

    /// Test-only peek at the list of generators already tried in the
    /// current rendezvous. Empty on the first attempt; grows by one
    /// each time [`Self::check_rendezvous_timeout`] retries.
    #[cfg(test)]
    pub(crate) fn rendezvous_tried_generators(&self) -> Vec<usize> {
        self.nccl_rendezvous_pending
            .as_ref()
            .map(|p| p.tried_generators.clone())
            .unwrap_or_default()
    }

    /// Test-only seam: install a synthetic pending rendezvous so the
    /// retry path can be unit-tested without a live NCCL setup.
    /// `initiated_offset_secs` shifts `initiated_at` into the past so
    /// `check_rendezvous_timeout` trips immediately on the next tick
    /// without sleeping.
    #[cfg(test)]
    pub(crate) fn test_seed_rendezvous_pending(
        &mut self,
        generator_rank: usize,
        survivors_ordered: Vec<usize>,
        initiated_offset_secs: u64,
    ) {
        let initiated_at = Instant::now()
            .checked_sub(Duration::from_secs(initiated_offset_secs))
            .unwrap_or_else(Instant::now);
        self.nccl_rendezvous_pending = Some(NcclRendezvousPending {
            generator_rank,
            survivors_ordered,
            initiated_at,
            tried_generators: Vec::new(),
        });
    }

    pub fn max_overshoot(&self) -> usize {
        self.max_overshoot
    }

    pub fn el_che(&self) -> &ElChe {
        &self.el_che
    }

    /// Test-only peek at the most-recent per-rank LR snapshot. Mirrors
    /// the OLD coordinator's `last_lr_per_rank`. `None` for ranks that
    /// have not yet sent a [`TimingMsgWire::LrUpdate`].
    #[cfg(test)]
    pub(crate) fn last_lr_per_rank_for_test(&self) -> &[Option<f64>] {
        &self.last_lr_per_rank
    }

    /// Test-only peek at whether the LR-aware meta-controller is
    /// active. Returns `true` when
    /// [`ClusterCoordinatorConfig::meta_controller`] was set.
    #[cfg(test)]
    pub(crate) fn meta_controller_enabled_for_test(&self) -> bool {
        self.lr_event_meta.is_some()
    }

    /// Test-only peek at the per-rank observed sync-lag history (ms).
    /// See [`Self::last_observed_sync_lag_ms`] for the caveat about
    /// barrier correlation.
    #[cfg(test)]
    pub(crate) fn last_observed_sync_lag_ms_for_test(&self) -> &[Option<f64>] {
        &self.last_observed_sync_lag_ms
    }

    /// Per-rank wall-time (ms) from `RequestParams` broadcast to that
    /// rank's
    /// [`crate::distributed::wire::TimingMsgWire::SnapshotReady`].
    /// Honest per-rank capacity signal — measured BEFORE the AllReduce
    /// barrier so it's clean of slowest-rank contamination (unlike
    /// the internal `last_observed_sync_lag_ms` field which records
    /// the post-barrier round-trip). Suitable as a per-rank
    /// upload-throughput proxy for ElChe's partition rebalancer.
    ///
    /// `None` for ranks that haven't reported SnapshotReady this cycle
    /// (NCCL backend has no SnapshotReady, dead ranks, or
    /// late-arriving stragglers). Reset at the start of every
    /// averaging cycle.
    pub fn last_observed_upload_ms(&self) -> &[Option<f64>] {
        &self.last_observed_upload_ms
    }

    /// Feed an averaging-cycle observation to the LR-aware meta-controller
    /// (when enabled) and dispatch any returned action to ElChe.
    ///
    /// Ported from OLD `Coordinator::observe_meta` (`coordinator/mod.rs`);
    /// matches behavior including `MetaAction::NudgeDown` dispatch via
    /// [`crate::distributed::ddp::ElChe::nudge_anchor_down`] (OLD's Stage
    /// 3 — already landed per `project_06_controller_arc.md`).
    ///
    /// No-op when:
    /// - the meta-controller is disabled (`lr_event_meta` is `None`),
    /// - no rank has reported its LR yet (cold-start, ≤ 1 cycle).
    ///
    /// On [`crate::distributed::lr_event_meta::MetaAction::NudgeDown`]
    /// the resulting anchor change is captured by the cycle's net
    /// `AnchorChanged` event in the caller (we don't emit a duplicate
    /// event here — only a `verbose!` log line, mirroring OLD).
    fn observe_meta(
        &mut self,
        verdict: crate::distributed::ddp_run::convergence::ConvergenceAction,
    ) {
        let Some(lr) = self.last_lr_per_rank.iter().copied().find_map(|x| x) else {
            return;
        };
        let anchor = self.el_che.anchor();
        let phase = self.el_che.phase();
        let action = match self.lr_event_meta.as_mut() {
            Some(meta) => meta.observe(lr, anchor, verdict, phase),
            None => return,
        };
        if let crate::distributed::lr_event_meta::MetaAction::NudgeDown { factor } = action {
            let old = self.el_che.anchor();
            self.el_che.nudge_anchor_down(factor);
            let new = self.el_che.anchor();
            crate::verbose!(
                "  ddp: meta-controller nudge factor={:.3} anchor {} -> {}",
                factor, old, new,
            );
        }
    }

    // -----------------------------------------------------------------
    // State-machine drive
    // -----------------------------------------------------------------

    /// Process a single timing message. Ported literally from OLD
    /// `Coordinator::process_timing_msg`, modulo the field-name `rank`
    /// being `u64` on the wire vs `usize` in-process.
    fn process_timing_msg(&mut self, msg: TimingMsgWire) {
        // Liveness tick: every frame from a rank counts as proof-of-
        // life. Updates `last_heartbeat[rank]` before any per-message
        // work so even malformed-rank frames (rejected below) at least
        // refresh the slot. `check_dead_ranks` later compares against
        // wall-clock.
        let rank_for_liveness: Option<usize> = match &msg {
            TimingMsgWire::Batch { rank, .. }
            | TimingMsgWire::SyncAck { rank, .. }
            | TimingMsgWire::Exiting { rank }
            | TimingMsgWire::LrUpdate { rank, .. }
            | TimingMsgWire::Heartbeat { rank, .. }
            | TimingMsgWire::SnapshotReady { rank }
            | TimingMsgWire::NewNcclIdGenerated { rank, .. }
            | TimingMsgWire::EvalResult { rank, .. }
            | TimingMsgWire::CheckpointResult { rank, .. }
            | TimingMsgWire::EpochFnElapsed { rank, .. } => Some(*rank as usize),
        };
        if let Some(r) = rank_for_liveness {
            if r < self.last_heartbeat.len() {
                self.last_heartbeat[r] = Instant::now();
            }
        }
        match msg {
            TimingMsgWire::Batch {
                rank,
                batch_ms,
                step_count,
                param_norm,
                batch_loss,
                sync_divergence,
            } => {
                let rank = rank as usize;
                let step_count = step_count as usize;
                if rank >= self.world_size {
                    return; // ignore malformed; tests will fail loudly
                }
                self.steps_since_avg[rank] =
                    self.steps_since_avg[rank].saturating_add(1);
                self.wall_ms_accum[rank] += batch_ms;
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                self.last_batch_ms[rank] = batch_ms;
                let _ = batch_loss; // monitoring only in this slice
                let _ = param_norm;
                if let Some(div) = sync_divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsgWire::SyncAck {
                rank,
                step_count,
                divergence,
                post_norm,
                pre_norm,
            } => {
                let rank = rank as usize;
                let step_count = step_count as usize;
                if rank >= self.world_size {
                    return;
                }
                self.last_step_count[rank] =
                    self.last_step_count[rank].max(step_count);
                if let Some(div) = divergence {
                    self.nccl_sync_divergence[rank] = Some(div);
                }
                if let Some(p) = pre_norm {
                    self.nccl_sync_pre_norm[rank] = Some(p);
                }
                if let Some(p) = post_norm {
                    match self.nccl_sync_post_norm {
                        None => self.nccl_sync_post_norm = Some(p),
                        Some(prev) => debug_assert!(
                            (prev - p).abs() <= 1e-6 * prev.abs().max(1.0),
                            "post_norm rank-disagreement: prev={prev} new={p} (rank {rank})"
                        ),
                    }
                }
                if rank < self.nccl_ack.len()
                    && !self.nccl_ack[rank]
                    && step_count > self.nccl_sync_step[rank]
                {
                    self.nccl_ack[rank] = true;
                    // Per-rank sync lag (wall time from
                    // `RequestParams` / `SyncNow` broadcast to this
                    // rank's SyncAck). Captured BEFORE
                    // `capture_nccl_sync_elapsed_if_complete` takes
                    // `nccl_sync_start` on the all-acked transition.
                    // Feeds the adaptive CPU deadline computed in the
                    // NEXT `trigger_averaging`.
                    if let Some(start) = self.nccl_sync_start {
                        if rank < self.last_observed_sync_lag_ms.len() {
                            self.last_observed_sync_lag_ms[rank] =
                                Some(start.elapsed().as_secs_f64() * 1000.0);
                        }
                    }
                    self.capture_nccl_sync_elapsed_if_complete();
                }
            }
            TimingMsgWire::Exiting { rank: _ } => {
                self.active_count = self.active_count.saturating_sub(1);
            }
            TimingMsgWire::LrUpdate { rank, lr } => {
                let rank = rank as usize;
                if rank < self.last_lr_per_rank.len() {
                    self.last_lr_per_rank[rank] = Some(lr);
                }
                // The meta-controller is consulted on every averaging
                // cycle via `observe_meta`; per-message work is just
                // recording the latest LR.
            }
            TimingMsgWire::Heartbeat { .. } => {
                // Liveness slot already refreshed above; nothing
                // further to do per-frame. `check_dead_ranks` reads
                // last_heartbeat each tick.
            }
            TimingMsgWire::SnapshotReady { rank } => {
                // Capture honest per-rank upload latency: T(now) -
                // T(RequestParams broadcast). The rank emitted this
                // BEFORE entering the AllReduce barrier (see param
                // bridge in cluster_worker.rs), so the measurement is
                // clean of slowest-rank barrier contamination — the
                // exact "honest per-rank capacity" signal flagged on
                // `last_observed_sync_lag_ms` as "planned upload-
                // completion marker".
                //
                // `nccl_sync_start` is the broadcast anchor; if the
                // cycle has already finalized (all SyncAcks in,
                // `capture_nccl_sync_elapsed_if_complete` took the
                // anchor), this frame is a late-arriving stragger
                // and we drop it — `last_observed_upload_ms[rank]`
                // keeps the prior value or None.
                let rank = rank as usize;
                if rank < self.last_observed_upload_ms.len() {
                    if let Some(start) = self.nccl_sync_start {
                        self.last_observed_upload_ms[rank] =
                            Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
            TimingMsgWire::NewNcclIdGenerated { rank, uid_bytes } => {
                let rank = rank as usize;
                if let Some(state) = self.nccl_rendezvous_pending.take() {
                    if state.generator_rank != rank {
                        crate::verbose!(
                            "  ddp: dropping NewNcclIdGenerated from rank {} \
                             (expected from generator rank {})",
                            rank,
                            state.generator_rank,
                        );
                        // Put back so we keep waiting for the real generator.
                        self.nccl_rendezvous_pending = Some(state);
                        return;
                    }
                    // Broadcast the new uid to each surviving rank
                    // with its position-in-shrunken-cohort. Survivors
                    // are ordered by ascending global rank.
                    if let Err(e) =
                        self.broadcast_new_nccl_session(uid_bytes)
                    {
                        crate::verbose!(
                            "  ddp: NewNcclSession broadcast failed: {} \
                             (NCCL elastic membership will not recover \
                             from this round of deaths; cluster may hang)",
                            e,
                        );
                    }
                } else {
                    crate::verbose!(
                        "  ddp: dropping unexpected NewNcclIdGenerated \
                         from rank {} (no rendezvous pending)",
                        rank,
                    );
                }
            }
            TimingMsgWire::EvalResult {
                rank,
                schedule_id: _,
                epoch,
                metric,
                elapsed_ms,
                error,
            } => {
                self.handle_eval_result(
                    rank as usize,
                    epoch as usize,
                    metric,
                    elapsed_ms,
                    error,
                );
            }
            TimingMsgWire::CheckpointResult {
                rank,
                version,
                elapsed_ms,
                error,
            } => {
                self.handle_checkpoint_result(
                    rank as usize,
                    version,
                    elapsed_ms,
                    error,
                );
            }
            TimingMsgWire::EpochFnElapsed {
                rank,
                epoch: _,
                elapsed_ms,
            } => {
                self.handle_epoch_fn_elapsed(rank as usize, elapsed_ms);
            }
        }
    }

    /// Handle a `TimingMsgWire::CheckpointResult` from a worker
    /// (S4 fleshes this out: time exclusion + role failover + retry
    /// across live untried ranks + EWMA update).
    ///
    /// S2 lands the stub so the wire propagation compiles; the
    /// post-S4 behavior is documented at the call site.
    /// Resolve the current "fastest" rank — the live rank with the
    /// lowest smoothed ms-per-batch reading from ElChe. Returns
    /// `usize::MAX` only when every rank is dead (caller should treat
    /// as no-op). Fallback when ElChe is uncalibrated (no sample yet
    /// from any rank): lowest-index live rank.
    ///
    /// Sticky semantics: callers retain the previously-resolved value
    /// across cadences and consult this method only on (a) initial
    /// resolution and (b) re-resolution after a role rank dies. ElChe
    /// drift between resolutions does not bounce the role around — by
    /// design, since checkpoint / eval / epoch_fn want a stable
    /// assignee (callbacks may stash thread-local state).
    fn resolve_fastest_role(&self) -> usize {
        let mut best: Option<(usize, f64)> = None;
        for r in 0..self.world_size {
            if self.is_dead(r) {
                continue;
            }
            let smoothed = self.el_che.smoothed_ms_per_batch(r);
            if let Some(ms) = smoothed {
                match best {
                    None => best = Some((r, ms)),
                    Some((_, prev)) if ms < prev => best = Some((r, ms)),
                    _ => {}
                }
            }
        }
        if let Some((r, _)) = best {
            return r;
        }
        // No ElChe samples yet: fall back to lowest live rank.
        for r in 0..self.world_size {
            if !self.is_dead(r) {
                return r;
            }
        }
        usize::MAX
    }

    /// Re-resolve all three role-rank fields against the current live-
    /// rank set + ElChe state, marking the epoch role dirty if it
    /// changed so the next dispatch broadcasts the update. Called on
    /// rank death + after the first calibrated ElChe sample. No-op for
    /// `Rank(n)` policy — the static rank stays put even after death
    /// (the rank-targeted dispatch to a dead rank will fail loudly
    /// rather than silently re-route, matching the user's "controller
    /// decides" principle).
    fn re_resolve_callback_roles_on_death(&mut self, dead_rank: usize) {
        if !matches!(
            self.epoch_callback_policy,
            crate::distributed::ddp_run::EpochCallbackPolicy::Fastest
        ) {
            return;
        }
        let prev_epoch = self.epoch_callback_role;
        if dead_rank == self.checkpoint_role {
            self.checkpoint_role = self.resolve_fastest_role();
        }
        if dead_rank == self.eval_role {
            self.eval_role = self.resolve_fastest_role();
        }
        if dead_rank == self.epoch_callback_role {
            self.epoch_callback_role = self.resolve_fastest_role();
        }
        if self.epoch_callback_role != prev_epoch
            && self.epoch_callback_role != usize::MAX
        {
            self.epoch_role_dirty = true;
        }
    }

    /// Broadcast `SetEpochCallbackRole { rank }` to every live worker
    /// if `epoch_role_dirty`. Clears the flag on successful broadcast.
    /// Called at the top of `dispatch_epoch` so workers always have
    /// a definite role before they receive their first `StartEpoch`.
    fn broadcast_epoch_callback_role_if_dirty(&mut self) -> Result<()> {
        if !self.epoch_role_dirty {
            return Ok(());
        }
        if self.epoch_callback_role == usize::MAX {
            // No live rank to designate; defer until at least one is
            // alive. (Unlikely in practice — world_size >= 1 invariant.)
            return Ok(());
        }
        let msg = ControlMsgWire::SetEpochCallbackRole {
            rank: self.epoch_callback_role as u64,
        };
        self.broadcast_control(&msg)?;
        self.epoch_role_dirty = false;
        Ok(())
    }

    fn handle_checkpoint_result(
        &mut self,
        rank: usize,
        version: u64,
        elapsed_ms: f64,
        error: Option<String>,
    ) {
        // Time exclusion: subtract checkpoint elapsed from this rank's
        // wall_ms_accum so ElChe's rebalancer does not see checkpoint
        // cost as compute slowness. Clamp at 0 to handle EWMA noise.
        if rank < self.wall_ms_accum.len() {
            self.wall_ms_accum[rank] =
                (self.wall_ms_accum[rank] - elapsed_ms).max(0.0);
        }
        match error {
            None => {
                // Success path: update the EWMA (alpha=0.3 — same
                // shape the rest of the framework uses for recent-
                // value smoothing), clear this version's tried set.
                let alpha = 0.3_f64;
                self.last_checkpoint_elapsed_ms_ewma =
                    Some(match self.last_checkpoint_elapsed_ms_ewma {
                        Some(prev) => alpha * elapsed_ms + (1.0 - alpha) * prev,
                        None => elapsed_ms,
                    });
                self.checkpoint_tried_ranks.remove(&version);
                self.checkpoint_role = rank;
                crate::verbose!(
                    "  ddp: checkpoint v{version} succeeded on rank {rank} \
                     ({elapsed_ms:.1} ms)",
                );
            }
            Some(err_msg) => {
                eprintln!(
                    "cluster_coordinator: checkpoint v{version} failed on \
                     rank {rank}: {err_msg}"
                );
                // Record this rank as tried, then release the mut-
                // borrow before calling `is_dead` (which needs &self).
                self.checkpoint_tried_ranks
                    .entry(version)
                    .or_default()
                    .insert(rank);
                let tried_snapshot: std::collections::HashSet<usize> = self
                    .checkpoint_tried_ranks
                    .get(&version)
                    .cloned()
                    .unwrap_or_default();
                let next = (0..self.world_size).find(|&r| {
                    r != rank && !self.is_dead(r) && !tried_snapshot.contains(&r)
                });
                match next {
                    Some(r) => {
                        self.checkpoint_role = r;
                        let msg = ControlMsgWire::Checkpoint {
                            version,
                            target_rank: r as u64,
                        };
                        if let Err(e) = self.send_control(r, &msg) {
                            eprintln!(
                                "cluster_coordinator: checkpoint v{version} \
                                 retry-dispatch to rank {r} failed: {e}"
                            );
                        }
                    }
                    None => {
                        eprintln!(
                            "cluster_coordinator: checkpoint v{version} \
                             exhausted all live ranks; giving up (existing \
                             MaxFailureThreshold continues to govern run \
                             health). tried={tried_snapshot:?}"
                        );
                        self.checkpoint_tried_ranks.remove(&version);
                    }
                }
            }
        }
    }

    /// Process an eval result from a worker. Mirrors
    /// [`Self::handle_checkpoint_result`] for the eval callback: fires
    /// the user's `eval_result_fn` (success path), time-excludes the
    /// closure's wall-time from `wall_ms_accum[rank]` so ElChe does not
    /// see eval cost as compute slowness, and updates
    /// `last_eval_elapsed_ms_ewma` for callback-aware partition
    /// scheduling. Unlike checkpoint, eval has no retry path — failed
    /// evals are logged and training continues, matching `metrics_fn`'s
    /// SkipAndContinue default. Time exclusion + EWMA fire regardless
    /// of success / failure: the wall-time was spent either way.
    fn handle_eval_result(
        &mut self,
        rank: usize,
        epoch: usize,
        metric: f64,
        elapsed_ms: f64,
        error: Option<String>,
    ) {
        // Time exclusion (parallel to checkpoint): subtract from this
        // rank's wall_ms_accum so ElChe's rebalancer does not interpret
        // eval cost as compute slowness. Clamp at 0 to absorb fp drift.
        if rank < self.wall_ms_accum.len() {
            self.wall_ms_accum[rank] =
                (self.wall_ms_accum[rank] - elapsed_ms).max(0.0);
        }
        // EWMA blend (alpha=0.3, same as checkpoint). Fires on every
        // report regardless of error: the closure wall-time is honest
        // even when the metric is not.
        let alpha = 0.3_f64;
        self.last_eval_elapsed_ms_ewma =
            Some(match self.last_eval_elapsed_ms_ewma {
                Some(prev) => alpha * elapsed_ms + (1.0 - alpha) * prev,
                None => elapsed_ms,
            });
        // User-facing dispatch: fire `eval_result_fn` on success; log
        // and continue on failure. Errors from the closure are logged
        // and training continues, matching `metrics_fn`'s
        // SkipAndContinue default.
        if let Some(err_msg) = error {
            eprintln!(
                "cluster_coordinator: eval_fn returned error (epoch {epoch}): {err_msg}"
            );
        } else if let Some(ref f) = self.eval_result_fn {
            if let Err(e) = f(epoch, metric) {
                eprintln!(
                    "cluster_coordinator: eval_result_fn returned error (epoch {epoch}): {e}"
                );
            }
        }
    }

    /// Process an `epoch_fn` post-fire report from a worker. Mirrors
    /// [`Self::handle_eval_result`] minus the user-facing dispatch:
    /// `epoch_fn` has no return value, the worker fires it
    /// autonomously, and the only coord-side bookkeeping is time
    /// exclusion + EWMA.
    fn handle_epoch_fn_elapsed(&mut self, rank: usize, elapsed_ms: f64) {
        if rank < self.wall_ms_accum.len() {
            self.wall_ms_accum[rank] =
                (self.wall_ms_accum[rank] - elapsed_ms).max(0.0);
        }
        let alpha = 0.3_f64;
        self.last_epoch_fn_elapsed_ms_ewma =
            Some(match self.last_epoch_fn_elapsed_ms_ewma {
                Some(prev) => alpha * elapsed_ms + (1.0 - alpha) * prev,
                None => elapsed_ms,
            });
    }

    /// Detect "next cycle is the last cycle of the current epoch" and,
    /// when true, stage per-rank callback wall-time on ElChe so the
    /// recompute in [`Self::finish_averaging_nccl`] /
    /// [`Self::finish_averaging_cpu`] shrinks the firing rank's quota
    /// for the next chunk. Workers absorb the callback inside the
    /// freed compute slack instead of bloating the AllReduce barrier
    /// wait.
    ///
    /// Conditions checked:
    /// - There's a pool for the current epoch and it's not empty.
    /// - The pool's remaining batches fit in one more cycle (≤ sum of
    ///   ElChe's `batch_counts`).
    /// - At least one callback fires on the upcoming epoch boundary:
    ///   * `epoch_fn` always fires on `epoch_callback_role`.
    ///   * `checkpoint_fn` fires on `checkpoint_role` if
    ///     `(current_epoch + 1) % checkpoint_every == 0`.
    ///   * `eval_fn` fires on `eval_role` if
    ///     `(current_epoch + 1) % eval_every_epochs == 0`.
    /// - The total slack per rank passes the
    ///   `max(0.05 * anchor_wall_ms, 100 ms)` guard.
    ///
    /// Silent no-op when any precondition fails. Slack is consumed
    /// exactly once on the next ElChe recompute (see
    /// [`ElChe::apply_callback_slack`]).
    fn maybe_apply_callback_slack_for_next_cycle(&mut self) {
        // Need a calibrated ElChe to translate ms → batches; pre-
        // calibration the partition is uniform anyway.
        if !self.el_che.is_calibrated() {
            return;
        }
        // Find the in-flight epoch (rank_epoch[0] — all ranks are sync-
        // aligned at finish_averaging time; any rank's view works).
        let epoch = self.rank_epoch.first().copied().unwrap_or(0);
        let remaining_batches = match self.chunk_pools.get(&epoch) {
            Some(pool) if pool.remaining() >= self.batch_size => {
                pool.remaining() / self.batch_size
            }
            _ => return,
        };
        let total_counts: usize = self.el_che.batch_counts().iter().sum();
        if total_counts == 0 || remaining_batches > total_counts {
            // Not the last cycle of the epoch yet.
            return;
        }
        // Compute per-rank callback wall-time for the upcoming epoch
        // boundary (epoch → epoch+1).
        let next_epoch = epoch.saturating_add(1);
        let mut slack_ms = vec![0.0_f64; self.world_size];
        // epoch_fn fires every epoch transition on epoch_callback_role.
        if let Some(ewma) = self.last_epoch_fn_elapsed_ms_ewma {
            let role = self.epoch_callback_role;
            if role < self.world_size {
                slack_ms[role] += ewma;
            }
        }
        // checkpoint_fn cadence: same `epoch > 0 && epoch % every == 0`
        // shape the dispatch site uses.
        if let Some(every) = self.checkpoint_every {
            if every > 0 && next_epoch > 0 && next_epoch % every == 0 {
                if let Some(ewma) = self.last_checkpoint_elapsed_ms_ewma {
                    let role = self.checkpoint_role;
                    if role < self.world_size {
                        slack_ms[role] += ewma;
                    }
                }
            }
        }
        // eval_fn cadence: mirror of checkpoint cadence.
        if let Some(every) = self.eval_every_epochs {
            if every > 0 && next_epoch > 0 && next_epoch % every == 0 {
                if let Some(ewma) = self.last_eval_elapsed_ms_ewma {
                    let role = self.eval_role;
                    if role < self.world_size {
                        slack_ms[role] += ewma;
                    }
                }
            }
        }
        // Guard: drop sub-threshold per-rank entries. Both an absolute
        // floor (100 ms — below noise on any realistic sync cycle) and
        // a relative floor (5 % of anchor wall-time — sub-noise on
        // long cycles regardless of absolute scale). Keep the larger
        // of the two so neither domain (small models on fast hardware,
        // large models on slow hardware) sees the wrong threshold.
        let cycle_ms = self.el_che.anchor_wall_ms();
        let threshold = (0.05 * cycle_ms).max(100.0);
        let mut any_meaningful = false;
        for s in slack_ms.iter_mut() {
            if *s < threshold {
                *s = 0.0;
            } else {
                any_meaningful = true;
            }
        }
        if !any_meaningful {
            return;
        }
        self.el_che.apply_callback_slack(&slack_ms);
    }

    fn capture_nccl_sync_elapsed_if_complete(&mut self) {
        if self.nccl_ack.iter().all(|&a| a) {
            if let Some(start) = self.nccl_sync_start.take() {
                self.last_nccl_sync_ms =
                    start.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    /// Drain every pending timing message non-blocking.
    pub fn drain_timing(&mut self) {
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
    }

    /// Block up to `timeout` for the first timing message, then drain
    /// the rest non-blocking. Returns `false` when every reader thread
    /// has exited (all senders dropped) so the caller can break its
    /// loop. Mirrors OLD `Coordinator::drain_timing_blocking`.
    pub fn drain_timing_blocking(&mut self, timeout: Duration) -> bool {
        match self.timing_rx.recv_timeout(timeout) {
            Ok(msg) => self.process_timing_msg(msg),
            Err(mpsc::RecvTimeoutError::Timeout) => return true,
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
        while let Ok(msg) = self.timing_rx.try_recv() {
            self.process_timing_msg(msg);
        }
        true
    }

    /// Check whether an averaging cycle should be triggered now. Ported
    /// literally from OLD `Coordinator::should_average`.
    ///
    /// `nccl_ack` is named for the NCCL path's SyncAck mechanism but
    /// serves both backends in the new TCP model: workers send a
    /// `TimingMsg::SyncAck` after every averaging round (regardless of
    /// backend) so the coordinator can gate re-triggering until the
    /// previous round has settled.
    pub fn should_average(&self) -> bool {
        // Every gate skips dead ranks: they won't ack, won't step, and
        // won't accumulate wall_ms. Treating them as "satisfied" lets
        // the surviving cohort keep training.
        for r in 0..self.world_size {
            if self.is_dead(r) {
                continue;
            }
            if !self.nccl_ack[r] {
                return false;
            }
            if self.steps_since_avg[r] == 0 {
                return false;
            }
        }
        // active_count must be > 0 — if every rank is dead, training
        // is over (caller's responsibility to detect that separately).
        if self.active_count == 0 {
            return false;
        }
        match self.policy {
            ApplyPolicy::Sync => (0..self.world_size)
                .filter(|r| !self.is_dead(*r))
                .all(|r| self.steps_since_avg[r] >= 1),
            ApplyPolicy::Cadence => {
                let target = self.el_che.anchor_wall_ms();
                if target > 0.0 {
                    let min_wall = (0..self.world_size)
                        .filter(|r| !self.is_dead(*r))
                        .map(|r| self.wall_ms_accum[r])
                        .fold(f64::MAX, f64::min);
                    return min_wall >= target;
                }
                let counts = self.el_che.batch_counts();
                (0..self.world_size)
                    .filter(|r| !self.is_dead(*r))
                    .all(|r| self.steps_since_avg[r] >= counts[r])
            }
            ApplyPolicy::Async => {
                let counts = self.el_che.batch_counts();
                (0..self.world_size)
                    .filter(|r| !self.is_dead(*r))
                    .all(|r| self.steps_since_avg[r] >= counts[r])
            }
        }
    }

    /// Throttle fast workers. NCCL backend is a no-op (the collective
    /// itself coordinates pacing); CPU backend defers to ElChe's
    /// rebalancer instead of explicit throttle frames.
    pub fn check_throttle(&mut self) -> Result<()> {
        if matches!(self.backend, AverageBackend::Nccl) {
            return Ok(());
        }
        let max_diff = match self.el_che.max_batch_diff() {
            Some(d) => d,
            None => return Ok(()),
        };
        if self.active_count < self.world_size {
            return Ok(());
        }
        let min_steps = self.steps_since_avg.iter().copied().min().unwrap_or(0);
        // Snapshot to avoid borrow-conflict on self.control_streams in send.
        let mut to_throttle: Vec<usize> = Vec::new();
        for (rank, &steps) in self.steps_since_avg.iter().enumerate() {
            let should = steps > min_steps + max_diff;
            if should && !self.throttled[rank] {
                to_throttle.push(rank);
            }
        }
        for rank in to_throttle {
            self.send_control(rank, &ControlMsgWire::Throttle)?;
            self.throttled[rank] = true;
        }
        Ok(())
    }

    /// Trigger an averaging cycle. Dispatches to the backend-specific
    /// trigger message + finish hook. Mirrors OLD
    /// `Coordinator::trigger_averaging`.
    ///
    /// - NCCL: broadcast `SyncNow`; finish_averaging_nccl runs
    ///   convergence inline using last-round divergence data + emits
    ///   `SetGlobalStep`.
    /// - CPU: broadcast `RequestParams`; finish_averaging_cpu mirrors
    ///   the NCCL flow but emits `Update{version}` as the lifecycle
    ///   barrier. Workers receive averaged tensors via the data
    ///   channel ([`crate::distributed::cpu_reduce::CpuReduceClient`])
    ///   between RequestParams and the next round.
    pub fn trigger_averaging(&mut self) -> Result<()> {
        match self.backend {
            AverageBackend::Nccl => {
                self.nccl_sync_start = Some(Instant::now());
                self.broadcast_control(&ControlMsgWire::SyncNow)?;
                for rank in 0..self.world_size {
                    self.nccl_sync_step[rank] = self.last_step_count[rank];
                    self.nccl_ack[rank] = false;
                }
                self.finish_averaging_nccl()?;
            }
            AverageBackend::Cpu => {
                self.nccl_sync_start = Some(Instant::now());
                self.broadcast_control(&ControlMsgWire::RequestParams)?;
                for rank in 0..self.world_size {
                    self.nccl_sync_step[rank] = self.last_step_count[rank];
                    self.nccl_ack[rank] = false;
                }
                // Reset per-rank upload markers so the new cycle's
                // measurements aren't read against a stale prior cycle.
                // NCCL path skips this — there's no SnapshotReady on
                // the in-place collective so the slots stay None
                // throughout.
                for slot in &mut self.last_observed_upload_ms {
                    *slot = None;
                }
                // Defer `finish_averaging_cpu` until every rank's
                // bridge SyncAck has populated `nccl_sync_divergence`
                // (otherwise the guard reads all-Nones → zero, breaking
                // divergence-driven cadence control on cycle 1).
                // `poll_cpu_averaging` (called from `tick`) finalizes.
                //
                // No deadline: dropping a CPU averaging cycle is a
                // correctness violation for Local SGD (per-rank drift
                // accumulates super-linearly across missed rendezvous
                // points). Liveness is a SEPARATE concern handled by
                // the heartbeat fault detector; slow-but-alive ranks
                // are absorbed by ElChe's per-rank `wall_ms_accum` /
                // `batch_counts` rebalance on the next cycle.
                self.cpu_avg_state = CpuAvgState::Pending;
            }
        }
        Ok(())
    }

    /// Drive the CPU averaging state machine one tick. No-op when
    /// [`CpuAvgState::Idle`].
    ///
    /// In [`CpuAvgState::Pending`]: if every ALIVE rank's bridge
    /// `SyncAck` has landed (i.e. `nccl_sync_divergence[r].is_some()`),
    /// runs [`Self::finish_averaging_cpu`] and returns to `Idle`.
    /// Dead ranks (per [`Self::dead_ranks`]) count as "acked" because
    /// their bridge SyncAck will never arrive — the controller has
    /// already released the in-flight AllReduce with surviving ranks
    /// only, so finalizing here is correct.
    ///
    /// The gate is on `nccl_sync_divergence`, not `nccl_ack`. `nccl_ack`
    /// can flip from an in-flight `Batch` whose `step_count` exceeds
    /// `nccl_sync_step` (set at trigger time) — that path is correct
    /// for the NCCL backend (no separate bridge; the post-AllReduce
    /// Batch IS the sync evidence) but not for the CPU backend, where
    /// the bridge's `SyncAck` is the only signal that the AllReduce
    /// round-trip actually finished. Gating on `nccl_sync_divergence`
    /// ensures the next `finish_averaging_cpu` reads real per-rank
    /// divergence rather than the all-Nones sentinel.
    fn poll_cpu_averaging(&mut self) -> Result<()> {
        if !matches!(self.cpu_avg_state, CpuAvgState::Pending) {
            return Ok(());
        }
        let all_alive_acked = (0..self.world_size).all(|r| {
            if self.is_dead(r) {
                return true;
            }
            self.nccl_sync_divergence[r].is_some()
        });
        if all_alive_acked {
            self.cpu_avg_state = CpuAvgState::Idle;
            return self.finish_averaging_cpu();
        }
        Ok(())
    }

    /// Scan `last_heartbeat` for stale entries. For each rank whose
    /// most-recent frame arrival exceeds `heartbeat_timeout_secs` and
    /// is not already dead, declare it dead via the shared
    /// [`crate::distributed::controller::DeadRanks`] ledger.
    ///
    /// Declaring a rank dead:
    /// - Sets the rank's flag (shared with controller).
    /// - Shuts down the rank's controller-side stream, waking any
    ///   in-flight AllReduce so it releases with surviving ranks.
    /// - Decrements `active_count` so subsequent `should_average`
    ///   gates use the smaller quorum.
    ///
    /// No-op when `dead_ranks` is `None` (elastic membership not
    /// configured — rank death is permanently blocking).
    fn check_dead_ranks(&mut self) {
        let Some(ledger) = self.dead_ranks.as_ref().cloned() else {
            return;
        };
        let now = Instant::now();
        let threshold = Duration::from_secs(self.heartbeat_timeout_secs);
        let mut any_newly_dead = false;
        for r in 0..self.world_size {
            if ledger.is_dead(r) {
                continue;
            }
            if now.duration_since(self.last_heartbeat[r]) > threshold {
                crate::verbose!(
                    "  ddp: heartbeat stale on rank {} (>{}s), declaring dead",
                    r,
                    self.heartbeat_timeout_secs,
                );
                // Compute the dead rank's un-processed remainder
                // BEFORE flipping `active_count` (the survivor count
                // used by the redistribution formula reads the
                // pre-decrement value plus the to-die rank).
                let remainder_plan = self.compute_dead_rank_remainder(r);
                ledger.declare_dead(r);
                self.active_count = self.active_count.saturating_sub(1);
                self.last_heartbeat[r] = now;
                any_newly_dead = true;
                // Callback-role failover. For `Rank(n)` policy the
                // role stays put — a static rank that died will surface
                // as a loud send_control error at the next dispatch
                // (matches the "controller decides" principle: no
                // silent re-routing of a user-pinned rank). For
                // `Fastest` policy, re-resolve all three roles
                // against the new live set + ElChe smoothed values.
                match self.epoch_callback_policy {
                    crate::distributed::ddp_run::EpochCallbackPolicy::Rank(_) => {
                        // Legacy #29 failover: if Rank(n) policy and
                        // the dead rank happens to be the checkpoint
                        // role, fall over to lowest live as a best-
                        // effort. Eval/epoch roles stay pinned.
                        if r == self.checkpoint_role {
                            if let Some(next) =
                                (0..self.world_size).find(|&i| i != r && !ledger.is_dead(i))
                            {
                                self.checkpoint_role = next;
                                crate::verbose!(
                                    "  ddp: checkpoint_role failover {} -> {} \
                                     (prior role declared dead)",
                                    r,
                                    next,
                                );
                            }
                        }
                    }
                    crate::distributed::ddp_run::EpochCallbackPolicy::Fastest => {
                        self.re_resolve_callback_roles_on_death(r);
                        crate::verbose!(
                            "  ddp: Fastest re-resolve after rank {} death \
                             — checkpoint={}, eval={}, epoch_fn={}",
                            r,
                            self.checkpoint_role,
                            self.eval_role,
                            self.epoch_callback_role,
                        );
                    }
                }
                // NCCL backend: notify every surviving worker so they
                // can update their LOCAL dead-rank ledgers and the
                // NCCL watchdog can abort the in-flight collective.
                // CPU backend doesn't need this — the controller-side
                // stream shutdown via the shared `DeadRanks` ledger
                // already releases its blocked AllReduce read.
                if matches!(self.backend, AverageBackend::Nccl) {
                    if let Err(e) = self.broadcast_control(
                        &ControlMsgWire::DeclareDead { rank: r as u64 },
                    ) {
                        crate::verbose!(
                            "  ddp: DeclareDead broadcast for rank {} failed: {}",
                            r,
                            e,
                        );
                    }
                }
                if let Some((remainder_offset, remainder_size)) = remainder_plan {
                    if let Err(e) = self.redistribute_dead_rank_partition(
                        r,
                        remainder_offset,
                        remainder_size,
                    ) {
                        crate::verbose!(
                            "  ddp: ExtendPartition dispatch for dead rank {} \
                             remainder failed: {} (samples will roll into \
                             next epoch's reshuffle)",
                            r,
                            e,
                        );
                    }
                }
            }
        }
        // After processing all deaths this tick, decide whether the
        // cluster is recoverable. If user-configured max_failure was
        // breached, or the backend's hard limit is hit (NCCL needs
        // world_size>=2; CPU needs at least 1 survivor), broadcast
        // ShutdownWithSave so survivors persist state before exiting.
        // This MUST come before initiate_nccl_rendezvous_if_needed —
        // the rendezvous path would silently early-exit at <2
        // survivors, leaving the lone survivor blocked indefinitely.
        if any_newly_dead {
            if let Some(reason) = self.unrecoverable_reason() {
                if let Err(e) = self.dispatch_shutdown_with_save(reason) {
                    crate::verbose!(
                        "  ddp: ShutdownWithSave broadcast failed: {}",
                        e,
                    );
                }
            } else if let Err(e) = self.initiate_nccl_rendezvous_if_needed() {
                crate::verbose!(
                    "  ddp: NCCL rendezvous initiation failed: {}",
                    e,
                );
            }
        }
    }

    /// Compute the un-processed `(partition_offset, partition_size)`
    /// inside dead rank `r`'s current-epoch partition. Returns `None`
    /// when there's nothing to redistribute (rank already finished its
    /// partition, partition_size was zero, or `epoch_plan_cache` has
    /// no entry for this rank's epoch).
    fn compute_dead_rank_remainder(&self, r: usize) -> Option<(u64, u64)> {
        let epoch = self.rank_epoch[r];
        let plans = self.epoch_plan_cache.get(&epoch)?;
        let plan = plans.get(r)?;
        let processed_batches = self
            .last_step_count[r]
            .saturating_sub(self.last_step_count_at_epoch_start[r]);
        let processed_samples = (processed_batches * self.batch_size) as u64;
        if processed_samples >= plan.partition_size {
            return None;
        }
        let remainder_offset = plan.partition_offset + processed_samples;
        let remainder_size = plan.partition_size - processed_samples;
        Some((remainder_offset, remainder_size))
    }

    /// Slice the dead rank's un-processed remainder across surviving
    /// ranks and emit an [`crate::distributed::wire::ControlMsgWire::ExtendPartition`]
    /// frame to each. Currently splits equally; ElChe-weighted
    /// distribution (using `partition_ratios` or throughput-derived
    /// sizes) is a refinement landing alongside SnapshotReady →
    /// ElChe consumer in a future slice. Per-rank slice sizes that
    /// don't divide evenly distribute the remainder one sample at a
    /// time to the first ranks.
    ///
    /// `dead_rank` itself is skipped. `world_size - active_count` may
    /// already include `dead_rank` if the caller decremented
    /// `active_count` before calling this method — that's fine
    /// because the filter below uses the live `is_dead` ledger which
    /// the caller already flipped.
    fn redistribute_dead_rank_partition(
        &mut self,
        dead_rank: usize,
        remainder_offset: u64,
        remainder_size: u64,
    ) -> Result<()> {
        if remainder_size == 0 {
            return Ok(());
        }
        let survivors: Vec<usize> = (0..self.world_size)
            .filter(|r| *r != dead_rank && !self.is_dead(*r))
            .collect();
        if survivors.is_empty() {
            return Err(TensorError::new(
                "cluster_coordinator: redistribute called with no surviving ranks",
            ));
        }
        let n = survivors.len() as u64;
        let per_size = remainder_size / n;
        let leftover = remainder_size % n;
        let mut cursor = remainder_offset;
        for (i, rank) in survivors.iter().enumerate() {
            let extra = if (i as u64) < leftover { 1 } else { 0 };
            let slice_size = per_size + extra;
            if slice_size == 0 {
                continue;
            }
            let msg = ControlMsgWire::ExtendPartition {
                partition_offset: cursor,
                partition_size: slice_size,
            };
            self.send_control(*rank, &msg)?;
            cursor += slice_size;
        }
        crate::verbose!(
            "  ddp: redistributed dead rank {}'s {} un-processed samples \
             across {} survivors",
            dead_rank,
            remainder_size,
            survivors.len(),
        );
        Ok(())
    }

    /// True iff `rank` is known dead via the shared ledger. Returns
    /// false when no ledger is configured.
    fn is_dead(&self, rank: usize) -> bool {
        self.dead_ranks
            .as_ref()
            .map(|d| d.is_dead(rank))
            .unwrap_or(false)
    }

    /// Determine whether the cluster's current state is unrecoverable
    /// and what [`crate::distributed::SaveReason`] should be recorded.
    ///
    /// Returns `None` either when the state is fine OR when a save +
    /// shutdown has already been dispatched (the flag prevents repeat
    /// broadcasts on subsequent ticks).
    ///
    /// Ordering: user-configured `max_failure` is checked first so that
    /// a configured threshold takes precedence over the backend's hard
    /// limit (a user with `MaxFailureThreshold::Absolute(1)` on an NCCL
    /// cluster gets `MaxFailureExceeded`, not `SingleSurvivor`).
    fn unrecoverable_reason(&self) -> Option<crate::distributed::SaveReason> {
        if self.shutdown_with_save_dispatched {
            return None;
        }
        let dead_count = self.world_size.saturating_sub(self.active_count);
        if let Some(threshold) = self.max_failure {
            if dead_count >= threshold.limit_for(self.world_size) {
                return Some(crate::distributed::SaveReason::MaxFailureExceeded);
            }
        }
        match self.backend {
            AverageBackend::Nccl if self.active_count < 2 => {
                // NCCL requires world_size >= 2 to form a comm; the
                // lone survivor cannot continue.
                Some(crate::distributed::SaveReason::SingleSurvivor)
            }
            AverageBackend::Cpu if self.active_count == 0 => {
                Some(crate::distributed::SaveReason::AllRanksLost)
            }
            _ => None,
        }
    }

    /// Broadcast `ShutdownWithSave` to all surviving ranks so they
    /// persist `.fdl` (model) + `.optim` (per-rank optimizer) files to
    /// the configured `save_path`. The controller writes the
    /// `.meta.json` sidecar itself before broadcasting — only the
    /// controller has the live ElChe trajectory + the cluster-wide
    /// epoch/step/sync-round counters, so the meta is its job. Workers
    /// own the model bytes (their GPU memory) and per-rank optimizer
    /// state, so those stay rank-side.
    ///
    /// After broadcasting, mark the flag so we don't re-broadcast on
    /// subsequent `check_dead_ranks` ticks. Broadcast goes to ALL
    /// ranks the wire-side knows about; dead ranks have already shut
    /// down their stream and the send is a no-op (matches the pattern
    /// used by `broadcast_control` for `DeclareDead`).
    fn dispatch_shutdown_with_save(
        &mut self,
        reason: crate::distributed::SaveReason,
    ) -> Result<()> {
        crate::verbose!(
            "  ddp: unrecoverable cluster state ({:?}); broadcasting \
             ShutdownWithSave (active={}/{})",
            reason,
            self.active_count,
            self.world_size,
        );

        // Controller-side meta.json write. Only fires when save_path is
        // configured (no destination = no meta). Errors log loud but
        // don't block the broadcast — losing the meta sidecar is bad
        // but not as bad as hanging the cluster on an unrecoverable
        // failure.
        if let Some(ref stem) = self.save_path {
            let meta_path =
                crate::distributed::CheckpointBundle::meta_path(stem);
            // Cluster-wide epoch: take the max across all known ranks.
            // Each rank's `rank_epoch[r]` reflects the last StartEpoch
            // dispatched to that rank, so max is the highest epoch any
            // rank reached.
            let epoch = self.rank_epoch.iter().copied().max().unwrap_or(0);
            // Stitch the guard's divergence ring buffer into the ElChe
            // state snapshot. `to_state` defaults `trend_history` to
            // None because ElChe doesn't own the guard; we own both
            // sides here so we can finish the picture.
            let mut elche_state = self.el_che.to_state();
            elche_state.trend_history = self.convergence_guard.trend_history();
            let meta = crate::distributed::CheckpointMeta::new(
                epoch,
                self.global_step,
                self.avg_count,
                self.world_size,
                reason,
            )
            .with_elche_state(elche_state);
            if let Err(e) = meta.write_to_file(&meta_path) {
                eprintln!(
                    "  ddp: controller meta write to {} failed: {e}",
                    meta_path.display(),
                );
            }
        }

        let msg = ControlMsgWire::ShutdownWithSave {
            reason: reason.to_u8(),
        };
        self.broadcast_control(&msg)?;
        self.shutdown_with_save_dispatched = true;
        Ok(())
    }

    /// NCCL-backend re-rendezvous initiation. Called from
    /// [`Self::check_dead_ranks`] once a rank has been declared dead
    /// on the NCCL path. No-op on CPU backend (the controller-side
    /// release handles CPU AllReduces). No-op when a rendezvous is
    /// already pending — additional deaths during the wait will be
    /// rolled into the same rendezvous when it completes
    /// (`broadcast_new_nccl_session` reads the *current* alive set,
    /// not the snapshot from rendezvous-initiation time).
    fn initiate_nccl_rendezvous_if_needed(&mut self) -> Result<()> {
        if !matches!(self.backend, AverageBackend::Nccl) {
            return Ok(());
        }
        if self.nccl_rendezvous_pending.is_some() {
            return Ok(());
        }
        let survivors_ordered: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_dead(*r))
            .collect();
        if survivors_ordered.len() < 2 {
            crate::verbose!(
                "  ddp: NCCL rendezvous skipped — fewer than 2 survivors \
                 ({} alive of {})",
                survivors_ordered.len(),
                self.world_size,
            );
            return Ok(());
        }
        let generator_rank = self.pick_uid_generator(&survivors_ordered);
        self.send_control(generator_rank, &ControlMsgWire::RequestNewNcclId)?;
        self.nccl_rendezvous_pending = Some(NcclRendezvousPending {
            generator_rank,
            survivors_ordered,
            initiated_at: Instant::now(),
            tried_generators: Vec::new(),
        });
        Ok(())
    }

    /// Retry an in-flight NCCL re-rendezvous when the chosen generator
    /// has died or has not responded within
    /// [`NCCL_RENDEZVOUS_TIMEOUT_SECS`]. Picks the next candidate from
    /// the rendezvous's `survivors_ordered` after filtering out dead
    /// ranks and already-tried generators. When the candidate pool is
    /// exhausted, falls back to
    /// [`Self::dispatch_shutdown_with_save`] so the cluster doesn't
    /// hang forever on a dead-on-arrival generator chain.
    ///
    /// No-op when no rendezvous is pending, when the backend isn't
    /// NCCL, or when the generator is still alive and inside its
    /// timeout window.
    fn check_rendezvous_timeout(&mut self) {
        if !matches!(self.backend, AverageBackend::Nccl) {
            return;
        }
        let timeout = Duration::from_secs(self.rendezvous_timeout_secs);
        let Some(pending) = self.nccl_rendezvous_pending.as_ref() else {
            return;
        };
        let generator_dead = self.is_dead(pending.generator_rank);
        let timed_out = pending.initiated_at.elapsed() > timeout;
        if !generator_dead && !timed_out {
            return;
        }

        let previous_generator = pending.generator_rank;
        let survivors = pending.survivors_ordered.clone();
        let mut tried = pending.tried_generators.clone();
        tried.push(previous_generator);

        // Filter: alive AND not previously tried. Preserve
        // `survivors_ordered`'s order (ascending global rank) so the
        // retry sequence is deterministic.
        let next: Option<usize> = survivors
            .iter()
            .copied()
            .find(|r| !self.is_dead(*r) && !tried.contains(r));

        crate::verbose!(
            "  ddp: NCCL rendezvous retry — previous generator {} {} (elapsed={:?}); \
             {} candidates remain",
            previous_generator,
            if generator_dead { "DIED" } else { "TIMED OUT" },
            pending.initiated_at.elapsed(),
            survivors
                .iter()
                .filter(|r| !self.is_dead(**r) && !tried.contains(r))
                .count(),
        );

        match next {
            Some(new_generator) => {
                // Send before mutating state so a send failure leaves
                // the previous pending entry intact for another retry
                // on the next tick.
                if let Err(e) = self
                    .send_control(new_generator, &ControlMsgWire::RequestNewNcclId)
                {
                    crate::verbose!(
                        "  ddp: NCCL rendezvous retry send to rank {} failed: {} \
                         (will try again next tick)",
                        new_generator,
                        e,
                    );
                    return;
                }
                if let Some(pending) = self.nccl_rendezvous_pending.as_mut() {
                    pending.generator_rank = new_generator;
                    pending.initiated_at = Instant::now();
                    pending.tried_generators = tried;
                }
            }
            None => {
                // Exhausted the candidate pool: every survivor at
                // initiation time has been asked and either died or
                // timed out. Clear the pending state and fall back to
                // ShutdownWithSave so survivors persist state instead
                // of hanging on an un-completable rendezvous.
                crate::verbose!(
                    "  ddp: NCCL rendezvous candidate pool exhausted; \
                     dispatching ShutdownWithSave"
                );
                self.nccl_rendezvous_pending = None;
                if let Some(reason) = self.unrecoverable_reason().or(Some(
                    crate::distributed::SaveReason::SingleSurvivor,
                )) {
                    if let Err(e) = self.dispatch_shutdown_with_save(reason) {
                        crate::verbose!(
                            "  ddp: ShutdownWithSave after rendezvous exhaustion failed: {}",
                            e,
                        );
                    }
                }
            }
        }
    }

    /// Pick the rank that should generate the next NCCL unique-id.
    ///
    /// Tier 1: the lowest-numbered SURVIVING rank that's in
    /// [`Self::local_ranks`] (co-located with the coord process,
    /// same-process latency, lowest correlated-failure risk).
    ///
    /// Tier 2: the surviving rank with the smallest observed
    /// `wall_ms_accum / steps_since_avg` (per-batch wall, NOT
    /// barrier-correlated — clean per-rank capacity proxy). Ties
    /// break by lowest global rank. When no rank has timing
    /// history yet, this collapses to "lowest surviving global rank"
    /// (deterministic).
    fn pick_uid_generator(&self, survivors_ordered: &[usize]) -> usize {
        // Tier 1: prefer a local survivor.
        if let Some(&local) = self
            .local_ranks
            .iter()
            .filter(|r| !self.is_dead(**r))
            .min()
        {
            return local;
        }
        // Tier 2: fastest network survivor (per-batch wall time,
        // tiebreak by global rank). `f64::partial_cmp` returns None
        // on NaN; treat as Equal so the rank tiebreak applies.
        survivors_ordered
            .iter()
            .copied()
            .min_by(|&a, &b| {
                let ta = self.per_rank_ms_per_batch(a);
                let tb = self.per_rank_ms_per_batch(b);
                ta.partial_cmp(&tb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            })
            .unwrap_or(survivors_ordered[0])
    }

    /// Average per-batch wall-time (ms) for rank `r` from the current
    /// cadence-interval accumulators. Returns `f64::INFINITY` when
    /// the rank has no batches yet (cold start) so it sorts LAST in
    /// the "fastest" picker — un-calibrated ranks shouldn't be
    /// preferred as UID generators.
    fn per_rank_ms_per_batch(&self, r: usize) -> f64 {
        let steps = self.steps_since_avg.get(r).copied().unwrap_or(0);
        if steps == 0 {
            return f64::INFINITY;
        }
        let wall = self.wall_ms_accum.get(r).copied().unwrap_or(0.0);
        wall / steps as f64
    }

    /// Broadcast `NewNcclSession` to every survivor. Called from the
    /// `NewNcclIdGenerated` arm of `process_timing_msg` after the
    /// generator rank ships back its freshly-generated UID. Each
    /// survivor receives a frame whose `new_rank` reflects its
    /// position among survivors ordered by ascending global rank.
    fn broadcast_new_nccl_session(&mut self, uid_bytes: Vec<u8>) -> Result<()> {
        // Recompute the survivor set from the *current* `is_dead`
        // ledger — additional deaths during the rendezvous wait must
        // be reflected in the broadcast so we don't ship a
        // NewNcclSession to an already-dead rank.
        let survivors: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_dead(*r))
            .collect();
        if survivors.len() < 2 {
            return Err(TensorError::new(
                "cluster_coordinator: NewNcclSession broadcast aborted; \
                 fewer than 2 surviving ranks (NCCL requires world_size >= 2)",
            ));
        }
        let new_world_size = survivors.len() as u64;
        for (new_rank, &global_rank) in survivors.iter().enumerate() {
            let msg = ControlMsgWire::NewNcclSession {
                uid_bytes: uid_bytes.clone(),
                new_rank: new_rank as u64,
                new_world_size,
            };
            self.send_control(global_rank, &msg)?;
        }
        Ok(())
    }

    fn finish_averaging_nccl(&mut self) -> Result<()> {
        let prev_sync_ms = self.last_nccl_sync_ms;
        self.last_nccl_sync_ms = 0.0;
        // Stage per-rank callback slack BEFORE report_timing so the
        // recompute inside ElChe applies it to the next cycle's
        // batch_counts (when the next cycle is the LAST cycle of the
        // current epoch).
        self.maybe_apply_callback_slack_for_next_cycle();
        if self.wall_ms_accum.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(
                &self.wall_ms_accum,
                &self.steps_since_avg,
                prev_sync_ms,
            );
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        let nccl_pre_norms: Option<Vec<f64>> =
            if self.nccl_sync_pre_norm.iter().all(|p| p.is_some()) {
                Some(self.nccl_sync_pre_norm.iter().map(|p| p.unwrap()).collect())
            } else {
                None
            };
        let report = convergence::DivergenceReport {
            deltas: self
                .nccl_sync_divergence
                .iter()
                .map(|d| d.unwrap_or(0.0))
                .collect(),
            pre_norms: nccl_pre_norms,
            post_norm: self.nccl_sync_post_norm,
        };
        let cycle_batches: usize = self.steps_since_avg.iter().sum();
        let k_max = self.steps_since_avg.iter().copied().max().unwrap_or(0);
        let action = self.convergence_guard.report(&report, cycle_batches, k_max);

        // LR-aware meta-controller (OLD `observe_meta` parity): consult
        // the meta after the guard verdict; a `NudgeDown` MetaAction
        // dispatches to `el_che.nudge_anchor_down` and composes
        // multiplicatively with the guard's own anchor adjustment
        // below.
        self.observe_meta(action);

        self.version += 1;
        self.avg_count += 1;

        match action {
            ConvergenceAction::Stable => {
                if self.policy == ApplyPolicy::Async {
                    if self.overshoot_auto {
                        self.max_overshoot =
                            (self.max_overshoot + 1).min(self.overshoot_ceiling);
                    }
                    if self.elche_relax_up {
                        self.el_che.relax_anchor_up();
                    }
                }
            }
            ConvergenceAction::SuppressGrowth => {}
            ConvergenceAction::NudgeDown { factor } => {
                self.el_che.nudge_anchor_down(factor);
                if self.overshoot_auto && self.policy == ApplyPolicy::Async {
                    self.max_overshoot = self.overshoot_initial;
                }
            }
        }
        if self.policy == ApplyPolicy::Async {
            self.max_overshoot = self.max_overshoot.min(self.overshoot_ceiling);
        }

        self.global_step += cycle_batches;

        self.broadcast_control(&ControlMsgWire::SetGlobalStep {
            global_step: self.global_step as u64,
        })?;

        for s in &mut self.steps_since_avg {
            *s = 0;
        }
        for a in &mut self.wall_ms_accum {
            *a = 0.0;
        }
        for t in &mut self.throttled {
            *t = false;
        }
        for d in &mut self.nccl_sync_divergence {
            *d = None;
        }
        for p in &mut self.nccl_sync_pre_norm {
            *p = None;
        }
        self.nccl_sync_post_norm = None;
        Ok(())
    }

    /// CPU-backend counterpart to [`Self::finish_averaging_nccl`].
    ///
    /// Differs from the NCCL path in one place: the lifecycle barrier
    /// emitted to workers is `ControlMsgWire::Update { version }` (the
    /// workers received their averaged tensors via the data channel
    /// already; this notification bumps their `current_version` and
    /// resets `steps_since_avg`).
    ///
    /// Everything else (ElChe report, convergence guard verdict,
    /// overshoot / anchor tuning, counter reset) mirrors NCCL.
    fn finish_averaging_cpu(&mut self) -> Result<()> {
        let prev_sync_ms = self.last_nccl_sync_ms;
        self.last_nccl_sync_ms = 0.0;
        // Mirror of `finish_averaging_nccl`: stage slack before
        // `report_timing` so the next-cycle batch_counts shrink on the
        // callback-firing rank.
        self.maybe_apply_callback_slack_for_next_cycle();
        if self.wall_ms_accum.iter().any(|&ms| ms > 0.0) {
            self.el_che.report_timing(
                &self.wall_ms_accum,
                &self.steps_since_avg,
                prev_sync_ms,
            );
            if !self.calibrated && self.el_che.is_calibrated() {
                self.calibrated = true;
            }
        }

        let pre_norms: Option<Vec<f64>> =
            if self.nccl_sync_pre_norm.iter().all(|p| p.is_some()) {
                Some(self.nccl_sync_pre_norm.iter().map(|p| p.unwrap()).collect())
            } else {
                None
            };
        let report = convergence::DivergenceReport {
            deltas: self
                .nccl_sync_divergence
                .iter()
                .map(|d| d.unwrap_or(0.0))
                .collect(),
            pre_norms,
            post_norm: self.nccl_sync_post_norm,
        };
        let cycle_batches: usize = self.steps_since_avg.iter().sum();
        let k_max = self.steps_since_avg.iter().copied().max().unwrap_or(0);
        let action = self.convergence_guard.report(&report, cycle_batches, k_max);

        // LR-aware meta-controller (OLD `observe_meta` parity); see
        // [`Self::finish_averaging_nccl`] for the rationale.
        self.observe_meta(action);

        self.version += 1;
        self.avg_count += 1;

        match action {
            ConvergenceAction::Stable => {
                if self.policy == ApplyPolicy::Async {
                    if self.overshoot_auto {
                        self.max_overshoot =
                            (self.max_overshoot + 1).min(self.overshoot_ceiling);
                    }
                    if self.elche_relax_up {
                        self.el_che.relax_anchor_up();
                    }
                }
            }
            ConvergenceAction::SuppressGrowth => {}
            ConvergenceAction::NudgeDown { factor } => {
                self.el_che.nudge_anchor_down(factor);
                if self.overshoot_auto && self.policy == ApplyPolicy::Async {
                    self.max_overshoot = self.overshoot_initial;
                }
            }
        }
        if self.policy == ApplyPolicy::Async {
            self.max_overshoot = self.max_overshoot.min(self.overshoot_ceiling);
        }

        self.global_step += cycle_batches;

        // CPU lifecycle barrier: workers received averaged tensors on
        // the data channel; this Update notification tells them to
        // bump `current_version` and reset `steps_since_avg`.
        self.broadcast_control(&ControlMsgWire::Update {
            version: self.version,
        })?;
        // SetGlobalStep is still broadcast so workers can update the
        // per-batch LR scheduler base. Same as the NCCL path.
        self.broadcast_control(&ControlMsgWire::SetGlobalStep {
            global_step: self.global_step as u64,
        })?;

        for s in &mut self.steps_since_avg {
            *s = 0;
        }
        for a in &mut self.wall_ms_accum {
            *a = 0.0;
        }
        for t in &mut self.throttled {
            *t = false;
        }
        for d in &mut self.nccl_sync_divergence {
            *d = None;
        }
        for p in &mut self.nccl_sync_pre_norm {
            *p = None;
        }
        self.nccl_sync_post_norm = None;
        Ok(())
    }

    /// One coordinator tick: drain incoming timing, throttle fast
    /// workers, and trigger averaging when due. Mirrors OLD
    /// `Coordinator::tick`. Returns `false` when every reader thread
    /// has exited so the caller can break its loop.
    pub fn tick(&mut self) -> Result<bool> {
        self.drain_timing();
        // Heartbeat-driven dead-rank detection. Must run BEFORE
        // poll_cpu_averaging / should_average so the rest of this
        // tick already sees the updated active membership; a rank
        // declared dead this tick won't gate the cycle's finalize.
        // No-op when elastic membership isn't configured.
        self.check_dead_ranks();
        // Cascading-death + slow-generator guard: an in-flight NCCL
        // rendezvous whose generator died (or stopped responding) would
        // hang the cohort indefinitely. Retries from the next survivor
        // candidate, or falls back to ShutdownWithSave when exhausted.
        // Runs independently of the dead-ranks ledger so a synthetic
        // pending state (test seam) is also exercised; production-side
        // pending only ever comes from `initiate_nccl_rendezvous_if_needed`
        // which already requires the ledger.
        self.check_rendezvous_timeout();
        self.check_throttle()?;
        // CPU-backend async finalize: if a cycle's `RequestParams` was
        // broadcast in a prior tick and all bridge SyncAcks have now
        // arrived (alive ranks only), finalize it here.
        // No-op on NCCL backend (state stays `Idle`).
        self.poll_cpu_averaging()?;
        if self.should_average() {
            self.trigger_averaging()?;
        }
        // The mpsc returns Disconnected when every cloned sender has
        // dropped (every reader thread has exited). drain_timing alone
        // can't see that — try_recv just returns Empty if there's no
        // current message and the channel is healthy. Probe explicitly.
        //
        // The reader-handle check uses `is_finished()` (not just
        // `is_some()`) because handles are never taken during the tick
        // loop — they only get taken in `shutdown()` / `Drop`. So
        // `is_some()` alone reduces the alive check to
        // `active_count > 0`, and if a rank exits without the coord
        // receiving its `Exiting` frame (TCP RST during teardown, or
        // any other lossy close), `active_count` never decrements and
        // the coord runs forever — hanging the bench's metrics_rx.
        // `is_finished()` reflects the reader thread's actual exit:
        // when the worker closes its stream, the reader sees EOF and
        // returns, and the coord then shuts down regardless of whether
        // Exiting was received.
        let any_reader_running = self.reader_handles.iter().any(|h| {
            h.as_ref().is_some_and(|j| !j.is_finished())
        });
        let alive = self.active_count > 0 && any_reader_running;
        // Drain metrics + try to aggregate completed epochs every tick.
        // Cheap: most ticks see an empty channel; on tick where every
        // alive rank has reported the same epoch, one `EpochMetrics`
        // is built and `metrics_fn` fires.
        self.drain_metrics_and_aggregate();
        // Post-aggregate epoch transition: dispatch the next epoch's
        // `StartEpoch` plan (non-progressive, non-Async), or broadcast
        // `Shutdown` when the final epoch has aggregated. Deferred
        // until any pending CPU averaging cycle has finalized so the
        // bridge SyncAck round-trip can complete (see method docs).
        self.try_advance_or_shutdown_after_aggregate();
        Ok(alive)
    }

    /// Drain pending [`crate::distributed::wire::MetricsMsgWire`]
    /// frames from the reader threads. In non-progressive mode,
    /// aggregate once every alive rank has reported for the same
    /// epoch. In progressive mode, accumulate per-chunk reports per
    /// epoch + dispatch the next chunk to the reporting rank, then
    /// aggregate when the epoch's `ChunkPool::is_epoch_done()` fires
    /// (and only in ascending epoch order — a fast rank streaming
    /// ahead can't aggregate while earlier epochs still have
    /// in-flight chunks on the slow rank).
    ///
    /// In all paths: build [`crate::distributed::ddp_run::EpochMetrics`]
    /// and fire the user-supplied `metrics_fn` + `metrics_sink_tx`.
    /// Per-epoch buffers are dropped on aggregation. Late frames from
    /// a dead rank are ignored.
    fn drain_metrics_and_aggregate(&mut self) {
        // Track ranks that received a chunk-complete report so we can
        // dispatch the next chunk after the borrow on `chunk_pools` /
        // `metrics_buffer` is released.
        let mut progressive_completions: Vec<(usize, usize)> = Vec::new();
        while let Ok(wire) = self.metrics_rx.try_recv() {
            let rank = wire.rank as usize;
            if rank >= self.world_size {
                continue;
            }
            let msg = crate::distributed::ddp_run::MetricsMsg {
                rank,
                epoch: wire.epoch as usize,
                avg_loss: wire.avg_loss,
                batches_processed: wire.batches_processed as usize,
                epoch_ms: wire.epoch_ms,
                samples_processed: wire.samples_processed as usize,
                share_complete_ms: wire.share_complete_ms,
                compute_only_ms: wire.compute_only_ms,
                data_starve_ms: wire.data_starve_ms,
                scalars: wire.scalars
                    .into_iter()
                    .map(|(k, (sum, count))| (k, (sum, count as usize)))
                    .collect(),
            };
            if self.progressive {
                if let Some(pool) = self.chunk_pools.get_mut(&msg.epoch) {
                    pool.mark_completed(rank, msg.samples_processed);
                }
                progressive_completions.push((rank, msg.epoch));
            }
            self.metrics_buffer
                .entry(wire.epoch)
                .or_default()
                .push(msg);
        }

        // Progressive: dispatch the next chunk to every rank that just
        // reported a chunk completion. Done after the drain loop so
        // we're not borrowing chunk_pools / metrics_buffer.
        if self.progressive {
            for (rank, _epoch) in progressive_completions {
                self.dispatch_next_chunk(rank);
            }
        }

        // Resolve readiness per dispatch mode.
        let alive: Vec<usize> = (0..self.world_size)
            .filter(|r| !self.is_rank_dead(*r))
            .collect();
        let ready_epochs: Vec<u64> = if self.progressive {
            // BTreeMap order: walk chunk_pools in ascending epoch
            // order, collecting done ones, STOPPING at the first
            // not-done (so a fast rank streaming ahead can't aggregate
            // before slower ranks finish earlier epochs).
            let mut ready: Vec<u64> = Vec::new();
            for (&epoch, pool) in &self.chunk_pools {
                if pool.is_epoch_done() {
                    ready.push(epoch as u64);
                } else {
                    break;
                }
            }
            ready
        } else {
            // Non-progressive: every alive rank emits exactly one
            // MetricsMsg per epoch. Epoch is ready when each alive
            // rank appears at least once in the Vec.
            self.metrics_buffer
                .iter()
                .filter_map(|(&epoch, msgs)| {
                    if alive.iter().all(|&r| msgs.iter().any(|m| m.rank == r)) {
                        Some(epoch)
                    } else {
                        None
                    }
                })
                .collect()
        };

        for epoch_key in ready_epochs {
            let msgs = match self.metrics_buffer.remove(&epoch_key) {
                Some(v) => v,
                None => continue,
            };
            // Pool-derived epoch wall-time wins in progressive mode
            // (the pool's `epoch_start` is the only authority); in
            // non-progressive the worker-reported max stands.
            let epoch_ms_override = if self.progressive {
                self.chunk_pools.remove(&(epoch_key as usize))
                    .map(|p| p.epoch_elapsed_ms())
            } else {
                None
            };
            // bc_share: equal share across alive ranks. Future
            // refinement could use `el_che.recent_batch_share()`
            // when calibrated; equal is fine for the aggregate
            // surfacing and never produces NaN.
            let bc_share = vec![1.0_f64 / alive.len().max(1) as f64; self.world_size];
            let mut metrics = crate::distributed::ddp_run::aggregate_epoch_metrics(
                epoch_key as usize,
                &msgs,
                &self.metrics_device_indices,
                &bc_share,
            );
            if let Some(ms) = epoch_ms_override {
                metrics.epoch_ms = ms;
            }
            self.last_aggregated_epoch = Some(epoch_key as usize);
            if let Some(ref f) = self.metrics_fn {
                if let Err(e) = f(&metrics) {
                    eprintln!(
                        "cluster_coordinator: metrics_fn returned error (epoch {epoch_key}): {e}"
                    );
                }
            }
            // Broadcast the aggregated view back to every rank so the
            // user-owned `Trainer::setup` training loop's
            // `monitor.log(&model)` sees the cross-rank picture
            // (global scalars + per-rank GPU tabs). The
            // `Trainer::builder` path already had this via the sink
            // tx; this broadcast gives the same UX to setup-mode
            // users in process-per-rank cluster runs. Broadcast
            // failures are non-fatal — a rank's stream may have
            // already closed during shutdown; surface as verbose.
            let wire_metrics: crate::distributed::wire::EpochMetricsWire =
                metrics.clone().into();
            if let Err(e) = self.broadcast_control(
                &ControlMsgWire::EpochAggregated(wire_metrics),
            ) {
                crate::verbose!(
                    "  ddp: EpochAggregated broadcast (epoch {epoch_key}) failed: {}",
                    e,
                );
            }
            if let Some(ref tx) = self.metrics_sink_tx {
                // Sink receiver dropped is benign — handle was
                // dropped before training finished. Don't surface
                // as an error.
                let _ = tx.send(metrics);
            }
        }

        // Epoch transition is handled by `try_advance_or_shutdown_after_aggregate`
        // which is invoked from `tick()` after `poll_cpu_averaging` has had
        // a chance to drive a still-pending CPU averaging cycle to Idle.
        // Calling it here directly would race with bridge SyncAcks: the
        // worker batch loop is async in cluster mode (Batch send and
        // MetricsMsg are not serialized against RequestParams/SyncAck),
        // so MetricsMsg can land while `cpu_avg_state == Pending` and
        // shutting workers down at that point would drop the in-flight
        // cycle.
    }

    /// Post-aggregation epoch transition. Once an epoch has aggregated
    /// AND any in-flight averaging cycle has finalized
    /// (`cpu_avg_state == Idle`), either dispatch the next epoch
    /// (non-progressive, non-Async) or broadcast `Shutdown` (final
    /// epoch). Idempotent: tracks `last_dispatched_epoch` so repeated
    /// ticks past the final aggregate don't re-broadcast `Shutdown`,
    /// and a still-pending CPU cycle simply defers the call until the
    /// next tick once the cycle finalizes.
    ///
    /// Mirrors threaded `Coordinator::on_epoch_aggregated` (see
    /// `ddp_run/coordinator/mod.rs:924`) but split from
    /// `drain_metrics_and_aggregate` because the cluster path is
    /// async: workers can post-send `MetricsMsg` while the previous
    /// batch's bridge SyncAck is still in transit.
    fn try_advance_or_shutdown_after_aggregate(&mut self) {
        if self.shutdown_initiated {
            return;
        }
        let Some(latest) = self.last_aggregated_epoch else {
            return;
        };
        // Wait for any in-flight CPU averaging cycle to finalize before
        // either dispatching a new epoch (which would race with the
        // pending SyncAck round-trip) or shutting workers down (which
        // would drop the cycle and leave the divergence guard with
        // all-Nones — see `end_to_end_sync_cpu_smoke` regression).
        if !matches!(self.cpu_avg_state, CpuAvgState::Idle) {
            return;
        }
        let next = latest + 1;
        if next >= self.num_epochs {
            self.shutdown_initiated = true;
            if let Err(e) = self.shutdown_workers() {
                crate::verbose!(
                    "  ddp: shutdown_workers after final aggregate failed: {}",
                    e,
                );
            }
        } else if !self.progressive
            && !matches!(self.policy, ApplyPolicy::Async)
            && self.last_dispatched_epoch.is_none_or(|d| d < next)
        {
            self.last_dispatched_epoch = Some(next);
            if let Err(e) = self.dispatch_epoch(next) {
                crate::verbose!(
                    "  ddp: dispatch_epoch({}) after aggregate failed: {}",
                    next,
                    e,
                );
            }
        }
    }

    /// True when `rank` is in the shared dead-rank ledger. `false` when
    /// elastic membership isn't configured (`dead_ranks` is `None`).
    fn is_rank_dead(&self, rank: usize) -> bool {
        self.dead_ranks
            .as_ref()
            .map(|d| d.is_dead(rank))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------
    // Epoch dispatch (ports OLD coordinator/mod.rs compute_partition_sizes
    // + plans_for_epoch + send_all_plans, with ControlMsgWire frames
    // replacing the OLD mpsc::Sender<ControlMsg>.)
    // -----------------------------------------------------------------

    /// Compute per-rank partition sizes for one epoch.
    ///
    /// Priority order:
    /// 1. Explicit `partition_ratios` from the config (test rigs,
    ///    user override).
    /// 2. ElChe throughput-derived sizes once calibrated (or once a
    ///    `with_speed_hint` is set in the config).
    /// 3. Equal sizes (fallback at startup before ElChe has
    ///    observations).
    ///
    /// Verbatim port of OLD `Coordinator::compute_partition_sizes`.
    fn compute_partition_sizes(&self) -> Vec<usize> {
        if let Some(ratios) = &self.partition_ratios {
            return crate::distributed::ddp_run::ratio_to_sizes(
                ratios,
                self.total_samples,
            );
        }
        match self.policy {
            ApplyPolicy::Sync => crate::distributed::ddp_run::equal_sizes(
                self.world_size,
                self.total_samples,
            ),
            ApplyPolicy::Cadence | ApplyPolicy::Async => {
                if self.el_che.is_calibrated() || self.el_che.has_speed_hint() {
                    crate::distributed::ddp_run::throughput_sizes(
                        &self.el_che,
                        self.total_samples,
                    )
                } else {
                    crate::distributed::ddp_run::equal_sizes(
                        self.world_size,
                        self.total_samples,
                    )
                }
            }
        }
    }

    /// Get (or lazily compute + cache) the per-rank plans for `epoch`.
    ///
    /// Caching guarantees every rank receives consistent
    /// `(partition_offset, partition_size)` even when [`Self::dispatch_epoch`]
    /// gets called twice for the same epoch (the second call returns
    /// the cached plans). Verbatim port of OLD
    /// `Coordinator::plans_for_epoch`.
    fn plans_for_epoch(
        &mut self,
        epoch: usize,
    ) -> Vec<crate::distributed::wire::EpochPlanWire> {
        use crate::distributed::wire::EpochPlanWire;
        if let Some(plans) = self.epoch_plan_cache.get(&epoch) {
            return plans.clone();
        }
        let sizes = self.compute_partition_sizes();
        let mut plans: Vec<EpochPlanWire> = Vec::with_capacity(self.world_size);
        let mut offset: u64 = 0;
        for &size in &sizes {
            plans.push(EpochPlanWire {
                epoch: epoch as u64,
                partition_offset: offset,
                partition_size: size as u64,
            });
            offset += size as u64;
        }
        self.epoch_plan_cache.insert(epoch, plans.clone());
        plans
    }

    /// Broadcast `StartEpoch(plan)` to every connected rank, updating
    /// `rank_epoch[r]` to `epoch` for each rank as it goes out.
    ///
    /// In **non-progressive** mode, sends one `StartEpoch` per rank
    /// carrying that rank's full per-epoch partition; returns the
    /// plans dispatched. In **progressive** mode (set on the coord
    /// config) creates a
    /// [`crate::distributed::chunk_pool::ChunkPool`] for the epoch
    /// and dispatches the first chunk to every rank; subsequent
    /// chunks are dispatched from `drain_metrics_and_aggregate` as
    /// ranks report chunk completion. The returned `Vec` reflects only
    /// the FIRST chunk per rank in progressive mode (callers should
    /// consume it as such).
    pub fn dispatch_epoch(
        &mut self,
        epoch: usize,
    ) -> Result<Vec<crate::distributed::wire::EpochPlanWire>> {
        if self.total_samples == 0 {
            return Err(TensorError::new(
                "cluster_coordinator: dispatch_epoch requires total_samples > 0; \
                 set ClusterCoordinatorConfig::total_samples before constructing.",
            ));
        }
        // Push the current epoch-callback role to workers BEFORE
        // StartEpoch arrives, so the worker's autonomous epoch_fn
        // fire-check sees a definite role on the first transition.
        // No-op on subsequent calls until `epoch_role_dirty` flips
        // (Fastest re-resolve on rank death).
        self.broadcast_epoch_callback_role_if_dirty()?;
        // User checkpoint cadence: when entering epoch N (N > 0) and
        // `N % checkpoint_every == 0`, broadcast a `Checkpoint(N)` frame
        // before `StartEpoch`. Workers fire `checkpoint_fn(N, &model)`
        // on the rank selected by [`EpochCallbackPolicy`]; others have
        // `checkpoint_fn = None` and treat the frame as a no-op. The
        // version reflects "model state at the end of epoch N-1", which
        // matches the threaded-path semantic `(epoch + 1) % every == 0`
        // (where the `+1` is the same off-by-one as treating epoch as
        // a 0-indexed counter).
        if epoch > 0 {
            if let Some(every) = self.checkpoint_every {
                if every > 0 && epoch % every == 0 {
                    // Targeted dispatch (S4 in #29): the coord's
                    // `checkpoint_role` is the sticky assignee; the
                    // worker no-ops unless `target_rank == self.rank`.
                    // Stays addressed to the SAME live rank across
                    // checkpoints until that rank fails or dies, at
                    // which point the controller fails over.
                    let target = self.checkpoint_role;
                    let msg = ControlMsgWire::Checkpoint {
                        version: epoch as u64,
                        target_rank: target as u64,
                    };
                    self.send_control(target, &msg)?;
                }
            }
            // Eval cadence: dispatch `ExecuteEvalCallback` to the
            // current `eval_role` when the boundary aligns with
            // `eval_every_epochs`. Targeted (parallels #29's
            // `Checkpoint` dispatch): the role is sticky across
            // cadences, re-resolved only on rank death when policy
            // is `Fastest`.
            if let Some(every) = self.eval_every_epochs {
                if every > 0 && epoch % every == 0 {
                    // schedule_id derived from epoch for now (one eval
                    // per cadence); a richer scheduler would mint a
                    // monotonic counter to disambiguate concurrent
                    // dispatches.
                    let target = self.eval_role;
                    let msg = ControlMsgWire::ExecuteEvalCallback {
                        schedule_id: epoch as u64,
                        epoch: epoch as u64,
                        target_rank: target as u64,
                    };
                    self.send_control(target, &msg)?;
                }
            }
        }
        if self.progressive {
            return self.start_epoch_progressive(epoch);
        }
        let plans = self.plans_for_epoch(epoch);
        for (rank, plan) in plans.iter().enumerate() {
            let msg = ControlMsgWire::StartEpoch(plan.clone());
            self.send_control(rank, &msg)?;
            self.rank_epoch[rank] = epoch;
            // Snapshot per-rank monotonic batch counter so a future
            // dead-rank declaration can compute how many of this
            // epoch's samples rank `r` had already processed (and
            // therefore how many remain to redistribute via
            // `ExtendPartition`).
            self.last_step_count_at_epoch_start[rank] = self.last_step_count[rank];
        }
        Ok(plans)
    }

    /// Start a new epoch in progressive mode: create a
    /// [`crate::distributed::chunk_pool::ChunkPool`] and dispatch the
    /// first chunk to every rank. Returns the per-rank
    /// [`crate::distributed::wire::EpochPlanWire`] of those first chunks
    /// so callers can pair the call with rank-side acknowledgments in
    /// tests. Subsequent chunks are dispatched from
    /// `drain_metrics_and_aggregate` on receipt of each rank's per-chunk
    /// MetricsMsg.
    ///
    /// Aligns the pool total to a batch boundary. Sub-batch remainders
    /// can't form a full batch and are dropped (standard DataLoader
    /// behaviour) — without this `is_epoch_done` never fires when
    /// `total_samples % batch_size != 0`.
    fn start_epoch_progressive(
        &mut self,
        epoch: usize,
    ) -> Result<Vec<crate::distributed::wire::EpochPlanWire>> {
        let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
        self.chunk_pools.insert(
            epoch,
            crate::distributed::chunk_pool::ChunkPool::new(
                epoch,
                batch_total,
                self.world_size,
            ),
        );
        let sizes: Vec<usize> = (0..self.world_size)
            .map(|r| self.compute_chunk_batches(r, epoch))
            .collect();
        crate::verbose!(
            "  ddp: epoch {epoch} progressive | initial chunks (batches) {sizes:?}"
        );
        let mut plans: Vec<crate::distributed::wire::EpochPlanWire> =
            Vec::with_capacity(self.world_size);
        for (rank, &batch_count) in sizes.iter().enumerate() {
            if let Some(plan) = self.dispatch_next_chunk_with_batches(
                rank, epoch, batch_count,
            )? {
                plans.push(plan);
            } else {
                // Rank received no work (e.g. world_size > batch_total).
                // Push an empty plan so the returned Vec has world_size
                // entries (callers / tests expect that shape).
                plans.push(crate::distributed::wire::EpochPlanWire {
                    epoch: epoch as u64,
                    partition_offset: 0,
                    partition_size: 0,
                });
            }
            self.last_step_count_at_epoch_start[rank] = self.last_step_count[rank];
        }
        Ok(plans)
    }

    /// Dispatch the next chunk to `rank` from the active pool. Called
    /// after a rank reports a chunk-complete MetricsMsg in progressive
    /// mode.
    ///
    /// Tries the rank's current epoch's pool first. If exhausted,
    /// streams ahead into the next epoch's pool (subject to the
    /// overshoot gate on CPU backends — NCCL backends skip the gate
    /// because overshoot is an async/CPU concept; NCCL coordinates via
    /// the AllReduce barrier itself).
    fn dispatch_next_chunk(&mut self, rank: usize) {
        let epoch = self.rank_epoch[rank];
        if self.chunk_pools.get(&epoch).is_some_and(|p| p.remaining() > 0) {
            let batches = self.compute_chunk_batches(rank, epoch);
            if let Err(e) = self.dispatch_next_chunk_with_batches(
                rank, epoch, batches,
            ) {
                crate::verbose!(
                    "  ddp: dispatch_next_chunk(rank={rank}, epoch={epoch}) error: {e}"
                );
            }
            return;
        }

        // Current pool exhausted for this rank: stream ahead. Skip
        // past already-aggregated epochs (their pools were removed by
        // `drain_metrics_and_aggregate`); re-creating them here would
        // produce an orphan pool that blocks all future aggregation
        // (BTreeMap walk stops at the first incomplete pool).
        let first_live = self.last_aggregated_epoch.map_or(0, |agg| agg + 1);
        let next_epoch = (epoch + 1).max(first_live);
        if next_epoch >= self.num_epochs {
            return;
        }

        // Overshoot gate (CPU backend only): don't dispatch into a
        // future epoch when the rank has streamed too far past its
        // planned batch count since the last averaging. NCCL backend
        // skips: blocking the fast GPU here would force it into
        // `wait_for_epoch_plan` where it can't send timing messages,
        // leaving nccl_ack permanently false and deadlocking
        // `should_average` + `check_throttle`.
        if !matches!(self.backend, AverageBackend::Nccl) {
            let current_aggregated = self.last_aggregated_epoch
                .is_some_and(|agg| epoch <= agg);
            if !current_aggregated {
                let planned = self.el_che.batch_counts().get(rank).copied().unwrap_or(0);
                if planned > 0
                    && self.steps_since_avg[rank] >= planned + self.max_overshoot
                {
                    crate::debug!(
                        "  ddp: overshoot gate BLOCKED rank {rank} | steps={} planned={} overshoot={}",
                        self.steps_since_avg[rank], planned, self.max_overshoot,
                    );
                    return;
                }
            }
        }

        if !self.chunk_pools.contains_key(&next_epoch) {
            let batch_total = (self.total_samples / self.batch_size) * self.batch_size;
            self.chunk_pools.insert(
                next_epoch,
                crate::distributed::chunk_pool::ChunkPool::new(
                    next_epoch,
                    batch_total,
                    self.world_size,
                ),
            );
            crate::verbose!("  ddp: streaming -> epoch {next_epoch} pool created");
        }
        let batches = self.compute_chunk_batches(rank, next_epoch);
        if let Err(e) = self.dispatch_next_chunk_with_batches(
            rank, next_epoch, batches,
        ) {
            crate::verbose!(
                "  ddp: dispatch_next_chunk(rank={rank}, next_epoch={next_epoch}) error: {e}"
            );
        }
    }

    /// Take `batches * batch_size` samples from the epoch's pool and
    /// dispatch a `StartEpoch` carrying the chunk slice. Returns the
    /// dispatched plan wire (for test callers) or `None` if the pool
    /// is exhausted / batches == 0.
    fn dispatch_next_chunk_with_batches(
        &mut self,
        rank: usize,
        epoch: usize,
        batches: usize,
    ) -> Result<Option<crate::distributed::wire::EpochPlanWire>> {
        let samples = batches * self.batch_size;
        if samples == 0 {
            return Ok(None);
        }
        let (offset, actual_size) = match self.chunk_pools.get_mut(&epoch) {
            Some(pool) => match pool.take_chunk(samples, rank) {
                Some(v) => v,
                None => return Ok(None),
            },
            None => return Ok(None),
        };
        self.rank_epoch[rank] = epoch;
        let plan = crate::distributed::wire::EpochPlanWire {
            epoch: epoch as u64,
            partition_offset: offset as u64,
            partition_size: actual_size as u64,
        };
        let msg = ControlMsgWire::StartEpoch(plan.clone());
        self.send_control(rank, &msg)?;
        Ok(Some(plan))
    }

    /// Compute how many batches the next chunk for `rank` in `epoch`
    /// should contain. Cold-start (pre-calibration) uses a small probe
    /// chunk (~10% of dataset per rank, floored at 4 batches) so
    /// ElChe gets enough averaging events to stabilise quickly.
    /// Post-calibration uses throughput-proportional sizing with a
    /// `min_chunk_batches` floor.
    fn compute_chunk_batches(&self, rank: usize, epoch: usize) -> usize {
        let pool = match self.chunk_pools.get(&epoch) {
            Some(p) => p,
            None => return 0,
        };
        let remaining_batches = pool.remaining() / self.batch_size;
        if remaining_batches == 0 {
            return 0;
        }
        if !self.el_che.is_calibrated() && !self.el_che.has_speed_hint() {
            // Probe: small equal chunks for fast calibration (~10%
            // per rank, min 4 batches).
            let probe = (self.total_samples
                / (self.world_size * 10 * self.batch_size))
                .max(4);
            return probe.min(remaining_batches);
        }
        // Calibrated: proportional to ElChe's throughput-derived batch counts.
        let counts = self.el_che.batch_counts();
        let total_counts: usize = counts.iter().sum();
        if total_counts == 0 {
            return remaining_batches.min(self.min_chunk_batches);
        }
        let ratio = counts.get(rank).copied().unwrap_or(0) as f64
            / total_counts as f64;
        let target = (remaining_batches as f64 * ratio).ceil() as usize;
        target.max(self.min_chunk_batches).min(remaining_batches)
    }

    /// Borrow per-rank current-epoch state for diagnostics / tests.
    pub fn rank_epoch(&self) -> &[usize] {
        &self.rank_epoch
    }

    /// Last globally-aggregated epoch (all ranks reported). `None`
    /// until the coord's metrics aggregator fires the first time.
    /// Current sticky `checkpoint_role` (rank ID). Test/diagnostic
    /// accessor; the role updates on success (stays put) or failure
    /// (failover to next live untried rank) or rank death.
    pub fn checkpoint_role(&self) -> usize {
        self.checkpoint_role
    }

    /// EWMA of recent successful checkpoint wall-times (ms). `None`
    /// until the first success lands. Reserved for v2 rendezvous-
    /// aware scheduling; test/diagnostic accessor for v1.
    pub fn last_checkpoint_elapsed_ms_ewma(&self) -> Option<f64> {
        self.last_checkpoint_elapsed_ms_ewma
    }

    /// EWMA of recent `eval_fn` wall-times (ms). `None` until the first
    /// eval report lands. Consumed by ElChe's last-batch slack
    /// reservation: the firing rank's trailing-batch share is reduced
    /// so the eval pass absorbs into idle slack.
    pub fn last_eval_elapsed_ms_ewma(&self) -> Option<f64> {
        self.last_eval_elapsed_ms_ewma
    }

    /// EWMA of recent `epoch_fn` wall-times (ms) on the role rank.
    /// `None` until the first `epoch_fn` post-fire report lands.
    /// Mirrors [`Self::last_eval_elapsed_ms_ewma`] for the autonomous
    /// `epoch_fn` path.
    pub fn last_epoch_fn_elapsed_ms_ewma(&self) -> Option<f64> {
        self.last_epoch_fn_elapsed_ms_ewma
    }

    /// Number of ranks recorded as having tried + failed for a given
    /// checkpoint `version`. Empty/zero on a clean run; populated by
    /// `handle_checkpoint_result` when an error arrives. Test
    /// accessor used to verify retry / exhaustion behavior.
    pub fn checkpoint_tried_count(&self, version: u64) -> usize {
        self.checkpoint_tried_ranks
            .get(&version)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    pub fn last_aggregated_epoch(&self) -> Option<usize> {
        self.last_aggregated_epoch
    }

    /// Batch size carried from config; used by progressive chunk
    /// dispatch to size per-chunk batches.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Number of epochs the trainer asked for; informs
    /// `dispatch_epoch` bounds and metrics aggregation.
    pub fn num_epochs(&self) -> usize {
        self.num_epochs
    }

    /// Total samples across the dataset; basis for partition sizing.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Build a headless ClusterCoordinator for unit-testing internal
    /// state-machine logic without spinning up TCP listeners or
    /// reader threads. `control_streams` and `reader_handles` are
    /// empty — calls into [`Self::send_control`] return a benign
    /// `TensorError` instead of panicking, so the
    /// retry-redispatch path in
    /// [`Self::handle_checkpoint_result`] surfaces as a log line
    /// rather than crashing the test. Test fixtures that need to
    /// drive the full wire path should use `spawn_coord` /
    /// `start_from_listener` instead.
    #[cfg(test)]
    pub(crate) fn for_test(mut config: ClusterCoordinatorConfig) -> Self {
        let world_size = config.world_size;
        let salt: SessionSalt =
            [0u8; crate::distributed::wire::SESSION_SALT_BYTES];
        let (_timing_tx, timing_rx) = mpsc::channel::<TimingMsgWire>();
        let (_metrics_tx, metrics_rx) =
            mpsc::channel::<crate::distributed::wire::MetricsMsgWire>();
        let el_che = std::mem::replace(
            &mut config.el_che,
            crate::distributed::ddp::ElChe::new(world_size.max(1), 1),
        );
        let calibrated = config.start_elche_state.is_some()
            && el_che.is_calibrated();
        ClusterCoordinator {
            policy: config.policy,
            backend: config.backend,
            world_size,
            overshoot_initial: config.overshoot_initial,
            overshoot_ceiling: config.overshoot_ceiling,
            overshoot_auto: config.overshoot_auto,
            elche_relax_up: config.elche_relax_up,
            el_che,
            convergence_guard: config.convergence_guard,
            version: 0,
            avg_count: config.start_avg_count,
            global_step: config.start_global_step,
            calibrated,
            active_count: world_size,
            max_overshoot: config.overshoot_initial,
            steps_since_avg: vec![0; world_size],
            wall_ms_accum: vec![0.0; world_size],
            last_batch_ms: vec![0.0; world_size],
            last_step_count: vec![0; world_size],
            nccl_sync_step: vec![0; world_size],
            nccl_ack: vec![true; world_size],
            nccl_sync_divergence: vec![None; world_size],
            nccl_sync_pre_norm: vec![None; world_size],
            nccl_sync_post_norm: None,
            throttled: vec![false; world_size],
            last_nccl_sync_ms: 0.0,
            nccl_sync_start: None,
            lr_event_meta: if config.meta_controller {
                Some(crate::distributed::lr_event_meta::LrEventMeta::with_default_config())
            } else {
                None
            },
            last_lr_per_rank: vec![None; world_size],
            cpu_avg_state: CpuAvgState::Idle,
            dead_ranks: config.dead_ranks,
            heartbeat_timeout_secs: config.heartbeat_timeout_secs,
            rendezvous_timeout_secs: config.rendezvous_timeout_secs,
            last_heartbeat: vec![Instant::now(); world_size],
            last_step_count_at_epoch_start: vec![0; world_size],
            nccl_rendezvous_pending: None,
            local_ranks: config.local_ranks.clone(),
            max_failure: config.max_failure,
            epoch_callback_policy: config.epoch_callback_policy,
            checkpoint_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            eval_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_callback_role: initial_callback_role(
                config.epoch_callback_policy,
                world_size,
            ),
            epoch_role_dirty: true,
            checkpoint_tried_ranks: std::collections::HashMap::new(),
            last_checkpoint_elapsed_ms_ewma: None,
            last_eval_elapsed_ms_ewma: None,
            last_epoch_fn_elapsed_ms_ewma: None,
            save_path: config.save_path.clone(),
            checkpoint_every: config.checkpoint_every,
            shutdown_with_save_dispatched: false,
            last_observed_sync_lag_ms: vec![None; world_size],
            last_observed_upload_ms: vec![None; world_size],
            rank_epoch: vec![0; world_size],
            last_aggregated_epoch: None,
            last_dispatched_epoch: None,
            shutdown_initiated: false,
            epoch_plan_cache: std::collections::HashMap::new(),
            total_samples: config.total_samples,
            batch_size: config.batch_size.max(1),
            num_epochs: config.num_epochs,
            partition_ratios: config.partition_ratios,
            timing_rx,
            metrics_rx,
            metrics_buffer: std::collections::BTreeMap::new(),
            chunk_pools: std::collections::BTreeMap::new(),
            progressive: config.progressive.unwrap_or(
                !matches!(config.policy, ApplyPolicy::Sync),
            ),
            min_chunk_batches: 4,
            metrics_fn: config.metrics_fn.clone(),
            metrics_sink_tx: config.metrics_sink_tx.clone(),
            eval_result_fn: config.eval_result_fn.clone(),
            eval_every_epochs: config.eval_every_epochs,
            metrics_device_indices: (0..world_size as u8).collect(),
            control_streams: Vec::new(),
            reader_handles: Vec::new(),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            bound_port: 0,
            salt,
        }
    }

    /// Test-only mutator for `wall_ms_accum[rank]`. Used by
    /// `checkpoint_time_excluded_from_wall_ms_accum` to set a known
    /// starting state before invoking `handle_checkpoint_result`.
    #[cfg(test)]
    pub(crate) fn set_wall_ms_accum_for_test(&mut self, rank: usize, ms: f64) {
        self.wall_ms_accum[rank] = ms;
    }

    /// Test-only accessor for `wall_ms_accum[rank]`.
    #[cfg(test)]
    pub(crate) fn wall_ms_accum_for_test(&self, rank: usize) -> f64 {
        self.wall_ms_accum[rank]
    }

    /// Test-only accessor for `heartbeat_timeout_secs`.
    #[cfg(test)]
    pub(crate) fn heartbeat_timeout_secs(&self) -> u64 {
        self.heartbeat_timeout_secs
    }

    /// Test-only mutator: force a rank's `last_heartbeat` to an
    /// arbitrary `Instant`. Used by
    /// `checkpoint_role_failover_on_rank_death` to age out a rank.
    #[cfg(test)]
    pub(crate) fn set_last_heartbeat_for_test(
        &mut self,
        rank: usize,
        when: Instant,
    ) {
        self.last_heartbeat[rank] = when;
    }

    /// Test-only wrapper around the private `check_dead_ranks` so
    /// tests can drive the heartbeat-stale path directly without
    /// spinning up the tick loop.
    #[cfg(test)]
    pub(crate) fn check_dead_ranks_for_test(&mut self) {
        self.check_dead_ranks();
    }

    /// Test-only wrappers exposing #28b's private Fastest-resolution
    /// helpers to the test module. The role accessors mirror the
    /// `checkpoint_role` public accessor for the other two roles
    /// (intentionally test-only since the public API surfaces are
    /// covered by the role-bearing wire messages workers see).
    #[cfg(test)]
    pub(crate) fn resolve_fastest_role_for_test(&self) -> usize {
        self.resolve_fastest_role()
    }
    #[cfg(test)]
    pub(crate) fn re_resolve_callback_roles_on_death_for_test(
        &mut self,
        dead_rank: usize,
    ) {
        self.re_resolve_callback_roles_on_death(dead_rank);
    }
    #[cfg(test)]
    pub(crate) fn set_callback_roles_for_test(
        &mut self,
        checkpoint: usize,
        eval: usize,
        epoch_cb: usize,
    ) {
        self.checkpoint_role = checkpoint;
        self.eval_role = eval;
        self.epoch_callback_role = epoch_cb;
    }
    #[cfg(test)]
    pub(crate) fn eval_role_for_test(&self) -> usize {
        self.eval_role
    }
    #[cfg(test)]
    pub(crate) fn epoch_callback_role_for_test(&self) -> usize {
        self.epoch_callback_role
    }
    #[cfg(test)]
    pub(crate) fn epoch_role_dirty_for_test(&self) -> bool {
        self.epoch_role_dirty
    }

    /// Test-only: install a `ChunkPool` for the given epoch so the
    /// `maybe_apply_callback_slack_for_next_cycle` path can read a
    /// known `remaining()`. Production code creates pools in
    /// `dispatch_epoch`; tests skip that scaffold.
    #[cfg(test)]
    pub(crate) fn install_chunk_pool_for_test(
        &mut self,
        epoch: usize,
        total_samples: usize,
    ) {
        self.chunk_pools.insert(
            epoch,
            crate::distributed::chunk_pool::ChunkPool::new(
                epoch,
                total_samples,
                self.world_size,
            ),
        );
    }

    /// Test-only: drive `rank_epoch[rank]` directly. Production sets
    /// it inside `dispatch_next_chunk_with_batches`.
    #[cfg(test)]
    pub(crate) fn set_rank_epoch_for_test(&mut self, rank: usize, epoch: usize) {
        self.rank_epoch[rank] = epoch;
    }

    /// Test-only wrapper around the private producer-side slack
    /// staging so tests can verify its effect on
    /// `el_che.pending_callback_slack_ms()` without driving an entire
    /// `finish_averaging_*` cycle.
    #[cfg(test)]
    pub(crate) fn maybe_apply_callback_slack_for_test(&mut self) {
        self.maybe_apply_callback_slack_for_next_cycle();
    }

    /// Test-only mutable accessor for the embedded `ElChe`. Used by
    /// slack-producer tests to drive `report_timing` calibrate the
    /// el_che before invoking the producer.
    #[cfg(test)]
    pub(crate) fn el_che_mut_for_test(
        &mut self,
    ) -> &mut crate::distributed::ddp::ElChe {
        &mut self.el_che
    }

    /// Test-only accessor for the embedded `ElChe`. Used by
    /// slack-producer tests to verify the pending slack vector was set.
    #[cfg(test)]
    pub(crate) fn el_che_for_test(&self) -> &crate::distributed::ddp::ElChe {
        &self.el_che
    }

    // -----------------------------------------------------------------
    // Outbound control frame I/O
    // -----------------------------------------------------------------

    fn send_control(&mut self, rank: usize, msg: &ControlMsgWire) -> Result<()> {
        if rank >= self.world_size {
            return Err(TensorError::new(&format!(
                "cluster_coordinator: send_control rank {rank} >= world_size {}",
                self.world_size
            )));
        }
        if rank >= self.control_streams.len() {
            // Headless coord (test fixtures via `for_test`) has no
            // streams populated. Return Err so callers that
            // tolerate transient send failures (e.g.
            // `handle_checkpoint_result`'s retry-dispatch path) can
            // log + continue rather than panic the test process.
            return Err(TensorError::new(&format!(
                "cluster_coordinator: send_control(rank={rank}): no stream \
                 (headless coord; index {rank} out of streams len {})",
                self.control_streams.len()
            )));
        }
        let frame = ControlFrame::encode(&self.salt, MsgKind::Control, msg)?;
        frame.write_to(&mut self.control_streams[rank]).map_err(|e| {
            TensorError::new(&format!(
                "cluster_coordinator: send_control(rank={rank}): {e}"
            ))
        })?;
        Ok(())
    }

    fn broadcast_control(&mut self, msg: &ControlMsgWire) -> Result<()> {
        for rank in 0..self.world_size {
            self.send_control(rank, msg)?;
        }
        Ok(())
    }

    /// Send Shutdown to every rank. Called from [`Self::shutdown`];
    /// kept public so callers running the coordinator inline can drop
    /// it from a different point in their loop if needed.
    pub fn shutdown_workers(&mut self) -> Result<()> {
        self.broadcast_control(&ControlMsgWire::Shutdown)
    }

    /// Stop reader threads, send Shutdown to every connected rank,
    /// join the threads, drop streams. Idempotent on the shutdown flag.
    pub fn shutdown(mut self) -> Result<()> {
        // Best-effort send Shutdown before tearing readers down. Ignore
        // write errors here: a rank may already have exited.
        let _ = self.shutdown_workers();
        self.shutdown_flag.store(true, Ordering::SeqCst);
        for handle_opt in self.reader_handles.iter_mut() {
            if let Some(handle) = handle_opt.take() {
                let _ = handle.join();
            }
        }
        Ok(())
    }
}

impl Drop for ClusterCoordinator {
    fn drop(&mut self) {
        // Best-effort shutdown if the caller forgot to call shutdown().
        self.shutdown_flag.store(true, Ordering::SeqCst);
        for handle_opt in self.reader_handles.iter_mut() {
            if let Some(handle) = handle_opt.take() {
                let _ = handle.join();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-rank reader thread
// ---------------------------------------------------------------------------

/// Read [`ControlFrame`]s from one rank's control stream, decode the
/// payload according to [`MsgKind`], and forward to the coordinator
/// via `tx`. Exits when:
///
/// - `shutdown` flips to true (set by [`ClusterCoordinator::shutdown`]),
/// - the stream EOFs cleanly (rank closed),
/// - or any wire-level error surfaces (HMAC mismatch, bincode decode,
///   bad msg_kind).
fn reader_loop(
    rank: usize,
    stream: &mut TcpStream,
    salt: &SessionSalt,
    shutdown: &Arc<AtomicBool>,
    tx: &mpsc::Sender<TimingMsgWire>,
    metrics_tx: &mpsc::Sender<crate::distributed::wire::MetricsMsgWire>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }
        match ControlFrame::try_read_from(stream, salt) {
            Ok(FrameRead::Frame(frame)) => match frame.kind {
                MsgKind::Timing => match frame.decode::<TimingMsgWire>() {
                    Ok(msg) => {
                        if tx.send(msg).is_err() {
                            // Coordinator dropped its receiver.
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "cluster_coordinator: reader r{rank} decode TimingMsg: {e}"
                        );
                        return;
                    }
                },
                MsgKind::Metrics => {
                    match frame.decode::<crate::distributed::wire::MetricsMsgWire>() {
                        Ok(msg) => {
                            if metrics_tx.send(msg).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "cluster_coordinator: reader r{rank} decode MetricsMsg: {e}"
                            );
                            return;
                        }
                    }
                }
                MsgKind::Heartbeat => {
                    // Orphan scaffolding from an earlier protocol draft.
                    // Heartbeats now flow through `TimingMsgWire::Heartbeat`
                    // over `MsgKind::Timing`, so this arm is intentionally
                    // unreached in current builds. Kept for wire-format
                    // stability (the enum value is part of protocol
                    // version 2's surface).
                }
                MsgKind::Control | MsgKind::ParamSnapshotMeta => {
                    eprintln!(
                        "cluster_coordinator: reader r{rank} got unexpected \
                         MsgKind {:?} on rank→coord path; dropping",
                        frame.kind
                    );
                }
            },
            Ok(FrameRead::WouldBlock) => {
                // Idle tick: re-check shutdown and keep reading.
                continue;
            }
            Ok(FrameRead::Eof) => {
                // Peer closed cleanly.
                return;
            }
            Err(e) => {
                // Wire errors at exit (rank closed connection mid-frame,
                // BrokenPipe on read) are the common case. Real
                // protocol violations are rare and would also show up
                // as decode errors above, which stay loud.
                crate::verbose!(
                    "cluster_coordinator: reader r{rank} wire error: {e}"
                );
                return;
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Tests live in a sibling file to keep this module navigable; the test
// body uses `use super::*` to access private items defined here.
#[cfg(test)]
#[path = "cluster_coordinator_tests.rs"]
mod tests;
