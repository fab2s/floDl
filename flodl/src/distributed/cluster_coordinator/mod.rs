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
//!
//! # File layout (internal)
//!
//! The implementation is split across sibling files for navigability;
//! all submodules add methods to the same `impl ClusterCoordinator`
//! type via Rust's split-impl support:
//!
//! - `config.rs` — builder-style [`ClusterCoordinatorConfig`].
//! - `lifecycle.rs` — `start` / `bind` / `start_from_listener` /
//!   `shutdown` / outbound control-frame I/O.
//! - `event_loop.rs` — `tick`, `process_timing_msg`, drain / throttle,
//!   per-cycle metrics aggregator.
//! - `averaging.rs` — `trigger_averaging`, `finish_averaging_{cpu,nccl}`,
//!   NCCL re-rendezvous retry.
//! - `callback_roles.rs` — sticky "fastest rank" election,
//!   `epoch_fn` / `eval_fn` / `checkpoint_fn` failover,
//!   `dispatch_shutdown_with_save`.
//! - `dead_ranks.rs` — heartbeat detection, partition redistribution,
//!   liveness queries.
//! - `epoch_dispatch.rs` — partition sizing, plan emission, progressive
//!   chunk-pool scheduling.
//! - `test_helpers.rs` — `#[cfg(test)]` constructors and accessors.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::Instant;

use hmac_sha256::HMAC;

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::convergence::ConvergenceGuard;
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::{
    ControlFrame, FrameRead, MsgKind, SessionSalt, TimingMsgWire,
};
use crate::tensor::{Result, TensorError};

pub mod config;
mod averaging;
mod callback_roles;
mod dead_ranks;
mod epoch_dispatch;
mod event_loop;
mod lifecycle;
#[cfg(test)]
mod test_helpers;

pub use config::ClusterCoordinatorConfig;

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
///
/// [`ControlMsgWire::ExtendPartition`]: crate::distributed::wire::ControlMsgWire::ExtendPartition
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


    pub fn max_overshoot(&self) -> usize {
        self.max_overshoot
    }

    pub fn el_che(&self) -> &ElChe {
        &self.el_che
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
                MsgKind::Control | MsgKind::ParamSnapshotMeta | MsgKind::Rendezvous => {
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

// Tests live in a sibling file (`tests.rs`) to keep this module
// navigable; the test body uses `use super::*` to access private items
// defined here.
#[cfg(test)]
mod tests;
