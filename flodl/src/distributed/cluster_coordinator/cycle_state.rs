//! Averaging-cycle state for [`super::ClusterCoordinator`]: the
//! per-cycle evidence slots (trigger instant, step snapshot, acks,
//! divergence/norm triple, measured latencies) plus the per-backend
//! machine ([`CycleMachine`]) that owns what only one backend has —
//! the CPU Pending window and its throttle ledger.
//!
//! # Why one struct with a backend sub-machine (audit I1)
//!
//! The AveragingCycle extraction moved the cycle *mechanics* into
//! `cycle_nccl.rs` / `cycle_cpu.rs` but left the cycle *state* flat in
//! the coordinator under NCCL-named fields the CPU cycle also ran on
//! (`nccl_sync_start` armed by `arm_cpu_cycle`, `nccl_sync_divergence`
//! gating the CPU finalize). Most slots are genuinely shared — both
//! backends snapshot steps at trigger, collect acks, and feed the same
//! divergence evidence to `finish_averaging_head` — so the partition
//! is: shared evidence here, backend-specific state in the machine.
//!
//! What differs per backend is *semantics*, not slots: a CPU bridge
//! `SyncAck` carries no meaningful `step_count` (the inner worker does
//! not bump `local_step` on `RequestParams`), so folding it into the
//! coordinator's `last_step_count` would poison the next cycle's step
//! snapshot and permanently wedge the NCCL re-arm gate — a bug that
//! happened once as an unguarded shared handler. That knowledge now
//! lives here ([`AvgCycleState::sync_ack_step_meaningful`]) instead of
//! as a backend `if` inside `event_loop.rs`.
//!
//! This struct is the state half of the eventual external-averager
//! decoupling (decomposition move 5): the cycle's state is one field
//! (`ClusterCoordinator::cycle`) extractable together with the
//! `cycle_*` mechanics.

use std::time::Instant;

use crate::distributed::ddp_run::AverageBackend;

/// CPU-backend averaging phase. The CPU reduce is asynchronous — the
/// worker bridges compute the AllReduce + weight-space divergence and
/// report back via `SyncAck` — so the cycle parks in [`Self::Pending`]
/// between `RequestParams` and the last alive rank's divergence
/// landing. The NCCL backend has no equivalent phase: its finish runs
/// inline at trigger (the collective itself is the rendezvous).
///
/// **Wait policy:** the coordinator waits **indefinitely** for every
/// rank's SyncAck. Dropping a CPU averaging cycle is a correctness
/// violation for Local SGD (per-rank drift accumulates super-linearly
/// across missed rendezvous points), so the only safe response to a
/// stalled rank is to keep waiting. **Liveness detection lives outside
/// the averaging path**: heartbeats feed the coordinator independently
/// and surface dead ranks; `poll_cpu_averaging`'s generous hard ceiling
/// catches the alive-but-wedged cohort. Slow (but live) ranks are
/// absorbed by ElChe on the next cycle.
///
/// The finalize gate is on `divergence[r].is_some()`, **not** on
/// `acked` — see [`super::ClusterCoordinator::poll_cpu_averaging`] for
/// why `acked` can flip early from an in-flight `Batch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CpuAvgPhase {
    /// No averaging cycle in flight.
    Idle,
    /// `RequestParams` broadcast; waiting for every alive rank's bridge
    /// SyncAck to populate the divergence slots.
    Pending,
}

/// The backend-specific half of the cycle state: what only one backend
/// has. Mirrors [`AverageBackend`] 1:1 at construction.
#[derive(Debug)]
pub(super) enum CycleMachine {
    /// NCCL: `finish_averaging_nccl` runs inline at trigger; no
    /// interior phase, no throttle frames (the collective paces).
    NcclInline,
    /// CPU: asynchronous finalize via the Pending window, plus the
    /// explicit throttle ledger (both the steps-diff throttle from
    /// `check_throttle` and the non-progressive reduce barrier).
    Cpu {
        phase: CpuAvgPhase,
        /// Wall-clock start of the current Pending window. Set when
        /// the phase flips to `Pending`; taken at finalize for the
        /// `CpuAvgEnd { duration_ms }` payload; read by the poll
        /// ceiling. Distinct from [`AvgCycleState::started_at`]
        /// (same trigger instant, different take-lifetime: the
        /// elapsed capture takes `started_at` on all-acked, which on
        /// the CPU path happens on post-update `Batch` evidence, not
        /// at finalize).
        pending_since: Option<Instant>,
        /// Per-rank: true while a `Throttle` frame is outstanding
        /// (set by `check_throttle`'s steps-diff pacing or the
        /// non-progressive reduce barrier in `arm_cpu_cycle`;
        /// cleared at `finish_averaging_tail`).
        throttled: Vec<bool>,
    },
}

/// One averaging cycle's state: shared evidence slots + the
/// per-backend [`CycleMachine`]. Owned by the coordinator as its
/// single `cycle` field; armed at `trigger_averaging`, consumed by the
/// `finish_averaging_*` pair and the event-loop frame handlers.
#[derive(Debug)]
pub(super) struct AvgCycleState {
    /// Instant the cycle's trigger broadcast (`SyncNow` /
    /// `RequestParams`) went out. Anchor for the lag / upload
    /// measurements; taken by [`Self::capture_sync_elapsed_if_complete`]
    /// once every alive rank has acked.
    pub(super) started_at: Option<Instant>,
    /// Per-rank `last_step_count` snapshot at trigger. The ack gate:
    /// a post-trigger frame whose `step_count` exceeds this proves the
    /// rank moved past the sync point.
    pub(super) step_snapshot: Vec<usize>,
    /// Per-rank: true once post-sync evidence has arrived (a `Batch`
    /// or `SyncAck` with `step_count` past the snapshot).
    pub(super) acked: Vec<bool>,
    /// Per-rank weight-space divergence from the last SyncAck. Doubles
    /// as the CPU finalize gate (`is_some` for every alive rank).
    pub(super) divergence: Vec<Option<f64>>,
    /// Per-rank pre-AllReduce L2 norm from the last SyncAck.
    pub(super) pre_norm: Vec<Option<f64>>,
    /// Post-AllReduce L2 norm (identical across ranks; populated by
    /// the first rank's SyncAck, cross-checked against the rest).
    pub(super) post_norm: Option<f64>,
    /// Wall-time (ms) of the last completed sync (trigger broadcast →
    /// all alive acks); fed to ElChe as `sync_ms` on the next cycle's
    /// window report.
    pub(super) last_sync_ms: f64,
    /// Per-rank wall-time (ms) from trigger broadcast to that rank's
    /// SyncAck arrival.
    ///
    /// **Caveat: barrier-correlated, NOT per-rank capacity.** A rank's
    /// bridge blocks inside the AllReduce barrier until every rank
    /// arrives, so individual lag values converge toward the slowest
    /// rank's contribution. Use as a *cycle latency* indicator only —
    /// honest per-rank capacity comes from the window ledger's
    /// per-batch wall and [`Self::upload_ms`].
    pub(super) sync_lag_ms: Vec<Option<f64>>,
    /// Per-rank wall-time (ms) from trigger broadcast to that rank's
    /// `SnapshotReady` arrival — emitted AFTER snapshot+upload but
    /// BEFORE the AllReduce barrier (see the param bridge in
    /// `cluster_worker.rs`), so it is clean of slowest-rank barrier
    /// contamination. Honest per-rank upload-capacity signal.
    ///
    /// Populated only by the CPU param bridge (the NCCL collective has
    /// no snapshot+upload step), but kept in the shared evidence so
    /// the public accessor returns a stable per-rank slice of `None`s
    /// on NCCL. Reset per-cycle by `arm_cpu_cycle`.
    pub(super) upload_ms: Vec<Option<f64>>,
    /// The backend-specific half. Mirrors the coordinator's backend.
    pub(super) machine: CycleMachine,
}

impl AvgCycleState {
    pub(super) fn new(backend: AverageBackend, world_size: usize) -> Self {
        AvgCycleState {
            started_at: None,
            step_snapshot: vec![0; world_size],
            // All-acked at rest: no sync is in flight, so settle gates
            // (`nccl_sync_settled`) and elapsed captures see a settled
            // cycle until the first arm.
            acked: vec![true; world_size],
            divergence: vec![None; world_size],
            pre_norm: vec![None; world_size],
            post_norm: None,
            last_sync_ms: 0.0,
            sync_lag_ms: vec![None; world_size],
            upload_ms: vec![None; world_size],
            machine: match backend {
                AverageBackend::Nccl => CycleMachine::NcclInline,
                AverageBackend::Cpu => CycleMachine::Cpu {
                    phase: CpuAvgPhase::Idle,
                    pending_since: None,
                    throttled: vec![false; world_size],
                },
            },
        }
    }

    // -----------------------------------------------------------------
    // Arming (shared)
    // -----------------------------------------------------------------

    /// Arm a new cycle: record the trigger instant and snapshot the
    /// per-rank step counters as the ack gate. Called by both
    /// `arm_nccl_cycle` and `arm_cpu_cycle` right before the trigger
    /// broadcast (the coordinator is single-threaded, so no frame can
    /// land between the arm and the broadcast).
    pub(super) fn arm(&mut self, now: Instant, last_step_count: &[usize]) {
        self.started_at = Some(now);
        self.step_snapshot.copy_from_slice(last_step_count);
        self.acked.fill(false);
    }

    // -----------------------------------------------------------------
    // Frame evidence (shared slots, backend-aware semantics)
    // -----------------------------------------------------------------

    /// Whether a `SyncAck`'s `step_count` is meaningful for the
    /// coordinator's `last_step_count` fold. Only the NCCL path uses
    /// it (for re-arm and global-step tracking); the CPU bridge's
    /// `SyncAck` has no meaningful step_count — the inner worker
    /// doesn't bump `local_step` on `RequestParams` — so folding it
    /// would poison `last_step_count` and, through the next cycle's
    /// step snapshot, permanently wedge the NCCL re-arm gate. CPU
    /// re-arm runs off the [`CpuAvgPhase`] instead.
    pub(super) fn sync_ack_step_meaningful(&self) -> bool {
        matches!(self.machine, CycleMachine::NcclInline)
    }

    /// Record a post-trigger `Batch` frame's ack evidence: a
    /// `step_count` past the trigger snapshot proves the rank moved
    /// past the sync point (on NCCL there is no separate bridge — the
    /// post-AllReduce Batch IS the sync evidence; on CPU it feeds the
    /// elapsed capture, while the finalize gate stays on divergence).
    /// Batch frames may piggy-back a divergence sample; it lands in
    /// the same slot a SyncAck would fill.
    pub(super) fn note_batch_ack(
        &mut self,
        rank: usize,
        step_count: usize,
        sync_divergence: Option<f64>,
    ) {
        if let Some(div) = sync_divergence {
            self.divergence[rank] = Some(div);
        }
        if rank < self.acked.len()
            && !self.acked[rank]
            && step_count > self.step_snapshot[rank]
        {
            self.acked[rank] = true;
            self.capture_sync_elapsed_if_complete();
        }
    }

    /// Record a `SyncAck` frame's evidence: divergence / norm slots,
    /// the ack gate, and the per-rank sync-lag capture. The caller
    /// handles `last_step_count` separately, gated on
    /// [`Self::sync_ack_step_meaningful`] (that counter is scheduling
    /// state the coordinator owns, not cycle state).
    pub(super) fn note_sync_ack(
        &mut self,
        rank: usize,
        step_count: usize,
        divergence: Option<f64>,
        pre_norm: Option<f64>,
        post_norm: Option<f64>,
    ) {
        if let Some(div) = divergence {
            self.divergence[rank] = Some(div);
        }
        if let Some(p) = pre_norm {
            self.pre_norm[rank] = Some(p);
        }
        if let Some(p) = post_norm {
            match self.post_norm {
                None => self.post_norm = Some(p),
                Some(prev) => debug_assert!(
                    (prev - p).abs() <= 1e-6 * prev.abs().max(1.0),
                    "post_norm rank-disagreement: prev={prev} new={p} (rank {rank})"
                ),
            }
        }
        if rank < self.acked.len()
            && !self.acked[rank]
            && step_count > self.step_snapshot[rank]
        {
            self.acked[rank] = true;
            // Per-rank sync lag (trigger broadcast → this rank's
            // SyncAck). Captured BEFORE the all-acked elapsed capture
            // takes `started_at`.
            if let Some(start) = self.started_at {
                if rank < self.sync_lag_ms.len() {
                    self.sync_lag_ms[rank] =
                        Some(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
            self.capture_sync_elapsed_if_complete();
        }
    }

    /// Record a `SnapshotReady` frame: honest per-rank upload latency
    /// from the trigger broadcast. If the cycle has already finalized
    /// (all acks in, the elapsed capture took `started_at`), the frame
    /// is a late-arriving straggler and is dropped — the slot keeps
    /// its prior value or `None`.
    pub(super) fn note_snapshot_ready(&mut self, rank: usize) {
        if rank < self.upload_ms.len() {
            if let Some(start) = self.started_at {
                self.upload_ms[rank] =
                    Some(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }

    /// Take `started_at` into `last_sync_ms` once every rank has
    /// acked. (Raw all-acked, not alive-acked: a dead rank never acks,
    /// deliberately leaving the cycle armed for the settle gates —
    /// see `nccl_sync_settled`.)
    pub(super) fn capture_sync_elapsed_if_complete(&mut self) {
        if self.acked.iter().all(|&a| a) {
            if let Some(start) = self.started_at.take() {
                self.last_sync_ms = start.elapsed().as_secs_f64() * 1000.0;
            }
        }
    }

    /// Every alive rank has acked (dead ranks count as acked — their
    /// evidence will never arrive and the surviving cohort has fully
    /// settled without them).
    pub(super) fn all_alive_acked(
        &self,
        mut is_dead: impl FnMut(usize) -> bool,
    ) -> bool {
        (0..self.acked.len()).all(|r| is_dead(r) || self.acked[r])
    }

    /// Every alive rank's bridge SyncAck has populated its divergence
    /// slot — the CPU finalize gate (see [`CpuAvgPhase`] for why the
    /// gate is divergence, not `acked`).
    pub(super) fn all_alive_diverged(
        &self,
        mut is_dead: impl FnMut(usize) -> bool,
    ) -> bool {
        (0..self.divergence.len())
            .all(|r| is_dead(r) || self.divergence[r].is_some())
    }

    /// Take `last_sync_ms` (returning it, resetting to 0.0) — consumed
    /// once per cycle by `finish_averaging_head` as the window
    /// report's `sync_ms`.
    pub(super) fn take_last_sync_ms(&mut self) -> f64 {
        std::mem::take(&mut self.last_sync_ms)
    }

    /// Build the cycle's [`convergence::DivergenceReport`] from the
    /// evidence slots. Pre-norms are all-or-none: a partial set means
    /// at least one rank's SyncAck predates the pre-norm field, so the
    /// report omits them entirely.
    ///
    /// [`convergence::DivergenceReport`]:
    ///     crate::distributed::ddp_run::convergence::DivergenceReport
    pub(super) fn divergence_report(
        &self,
    ) -> crate::distributed::ddp_run::convergence::DivergenceReport {
        let pre_norms: Option<Vec<f64>> =
            if self.pre_norm.iter().all(|p| p.is_some()) {
                Some(self.pre_norm.iter().map(|p| p.unwrap()).collect())
            } else {
                None
            };
        crate::distributed::ddp_run::convergence::DivergenceReport {
            deltas: self
                .divergence
                .iter()
                .map(|d| d.unwrap_or(0.0))
                .collect(),
            pre_norms,
            post_norm: self.post_norm,
        }
    }

    /// Reset the divergence / norm slots for the next cycle (from
    /// `finish_averaging_tail`).
    pub(super) fn reset_divergence_signals(&mut self) {
        crate::distributed::ddp_run::convergence::reset_divergence_signals(
            &mut self.divergence,
            &mut self.pre_norm,
            &mut self.post_norm,
        );
    }

    // -----------------------------------------------------------------
    // CPU machine
    // -----------------------------------------------------------------

    /// Reset the per-rank upload markers so a new cycle's measurements
    /// aren't read against a stale prior cycle. CPU arm only — on NCCL
    /// there's no SnapshotReady, so the slots stay `None` throughout.
    pub(super) fn reset_upload_markers(&mut self) {
        for slot in &mut self.upload_ms {
            *slot = None;
        }
    }

    /// Open the CPU Pending window (phase + its wall-clock). No-op on
    /// NCCL (the inline finish has no pending phase).
    pub(super) fn begin_cpu_pending(&mut self, now: Instant) {
        if let CycleMachine::Cpu { phase, pending_since, .. } = &mut self.machine {
            *phase = CpuAvgPhase::Pending;
            *pending_since = Some(now);
        }
    }

    /// Whether the CPU machine is parked in `Pending`. Always false on
    /// NCCL.
    pub(super) fn cpu_pending(&self) -> bool {
        matches!(
            self.machine,
            CycleMachine::Cpu { phase: CpuAvgPhase::Pending, .. }
        )
    }

    /// Flip the CPU phase back to `Idle` WITHOUT taking
    /// `pending_since` (the ceiling-escalation path: the window is
    /// abandoned, not finalized). No-op on NCCL.
    pub(super) fn abort_cpu_pending(&mut self) {
        if let CycleMachine::Cpu { phase, .. } = &mut self.machine {
            *phase = CpuAvgPhase::Idle;
        }
    }

    /// Close the CPU Pending window: flip to `Idle` and take
    /// `pending_since` (for the `CpuAvgEnd { duration_ms }` payload).
    /// Returns `None` on NCCL or when no window was open.
    pub(super) fn finish_cpu_pending(&mut self) -> Option<Instant> {
        if let CycleMachine::Cpu { phase, pending_since, .. } = &mut self.machine {
            *phase = CpuAvgPhase::Idle;
            pending_since.take()
        } else {
            None
        }
    }

    /// Read the Pending window's start without taking it (the poll
    /// ceiling). `None` on NCCL or outside a window.
    pub(super) fn cpu_pending_since(&self) -> Option<Instant> {
        match &self.machine {
            CycleMachine::Cpu { pending_since, .. } => *pending_since,
            CycleMachine::NcclInline => None,
        }
    }

    /// Whether a `Throttle` frame is outstanding for `rank`. Always
    /// false on NCCL (the collective paces; no throttle frames).
    pub(super) fn is_throttled(&self, rank: usize) -> bool {
        match &self.machine {
            CycleMachine::Cpu { throttled, .. } => throttled[rank],
            CycleMachine::NcclInline => false,
        }
    }

    /// Mark a `Throttle` as outstanding for `rank`. No-op on NCCL.
    pub(super) fn set_throttled(&mut self, rank: usize) {
        if let CycleMachine::Cpu { throttled, .. } = &mut self.machine {
            throttled[rank] = true;
        }
    }

    /// Mark every rank throttled (the non-progressive reduce barrier
    /// in `arm_cpu_cycle`). No-op on NCCL.
    pub(super) fn throttle_all(&mut self) {
        if let CycleMachine::Cpu { throttled, .. } = &mut self.machine {
            for t in throttled.iter_mut() {
                *t = true;
            }
        }
    }

    /// Clear every outstanding throttle (from `finish_averaging_tail`;
    /// the cycle's `Update` releases the workers). No-op on NCCL.
    pub(super) fn clear_throttled(&mut self) {
        if let CycleMachine::Cpu { throttled, .. } = &mut self.machine {
            for t in throttled.iter_mut() {
                *t = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn cpu_state(n: usize) -> AvgCycleState {
        AvgCycleState::new(AverageBackend::Cpu, n)
    }
    fn nccl_state(n: usize) -> AvgCycleState {
        AvgCycleState::new(AverageBackend::Nccl, n)
    }

    #[test]
    fn arm_snapshots_steps_and_clears_acks() {
        let mut c = nccl_state(2);
        assert!(c.acked.iter().all(|&a| a), "at rest = settled");
        c.arm(Instant::now(), &[5, 9]);
        assert_eq!(c.step_snapshot, vec![5, 9]);
        assert!(c.acked.iter().all(|&a| !a));
        assert!(c.started_at.is_some());
    }

    #[test]
    fn batch_ack_requires_step_past_snapshot() {
        let mut c = nccl_state(2);
        c.arm(Instant::now(), &[5, 9]);
        c.note_batch_ack(0, 5, None); // not past the snapshot
        assert!(!c.acked[0]);
        c.note_batch_ack(0, 6, None);
        assert!(c.acked[0]);
        // Elapsed capture only on ALL acked.
        assert!(c.started_at.is_some());
        c.note_batch_ack(1, 10, None);
        assert!(c.started_at.is_none(), "all-acked takes started_at");
        assert!(c.last_sync_ms >= 0.0);
    }

    #[test]
    fn sync_ack_step_meaningful_only_on_nccl() {
        assert!(nccl_state(2).sync_ack_step_meaningful());
        assert!(!cpu_state(2).sync_ack_step_meaningful());
    }

    #[test]
    fn sync_ack_populates_evidence_and_lag() {
        let mut c = cpu_state(2);
        c.arm(
            Instant::now()
                .checked_sub(Duration::from_millis(10))
                .unwrap(),
            &[3, 3],
        );
        c.note_sync_ack(0, 4, Some(0.5), Some(1.0), Some(2.0));
        assert_eq!(c.divergence[0], Some(0.5));
        assert_eq!(c.pre_norm[0], Some(1.0));
        assert_eq!(c.post_norm, Some(2.0));
        assert!(c.acked[0]);
        assert!(c.sync_lag_ms[0].is_some_and(|ms| ms > 0.0));
        assert!(!c.all_alive_diverged(|_| false));
        c.note_sync_ack(1, 4, Some(0.7), Some(1.0), None);
        assert!(c.all_alive_diverged(|_| false));
        assert!(c.all_alive_acked(|_| false));
    }

    #[test]
    fn dead_ranks_count_as_acked_and_diverged() {
        let mut c = cpu_state(2);
        c.arm(Instant::now(), &[0, 0]);
        c.note_sync_ack(0, 1, Some(0.1), None, None);
        assert!(c.all_alive_acked(|r| r == 1));
        assert!(c.all_alive_diverged(|r| r == 1));
    }

    #[test]
    fn snapshot_ready_dropped_after_finalize() {
        let mut c = cpu_state(1);
        c.arm(Instant::now(), &[0]);
        c.note_snapshot_ready(0);
        assert!(c.upload_ms[0].is_some());
        c.reset_upload_markers();
        assert!(c.upload_ms[0].is_none());
        // All-acked takes started_at; a straggler frame is dropped.
        c.note_batch_ack(0, 1, None);
        assert!(c.started_at.is_none());
        c.note_snapshot_ready(0);
        assert!(c.upload_ms[0].is_none(), "late straggler dropped");
    }

    #[test]
    fn cpu_pending_window_lifecycle() {
        let mut c = cpu_state(2);
        assert!(!c.cpu_pending());
        c.begin_cpu_pending(Instant::now());
        assert!(c.cpu_pending());
        assert!(c.cpu_pending_since().is_some());
        let start = c.finish_cpu_pending();
        assert!(start.is_some());
        assert!(!c.cpu_pending());
        assert!(c.cpu_pending_since().is_none());
        // Abort path leaves pending_since untaken but flips the phase.
        c.begin_cpu_pending(Instant::now());
        c.abort_cpu_pending();
        assert!(!c.cpu_pending());
        assert!(c.cpu_pending_since().is_some());
    }

    #[test]
    fn cpu_machine_hooks_are_noops_on_nccl() {
        let mut c = nccl_state(2);
        c.begin_cpu_pending(Instant::now());
        assert!(!c.cpu_pending());
        assert!(c.cpu_pending_since().is_none());
        assert!(c.finish_cpu_pending().is_none());
        c.set_throttled(0);
        c.throttle_all();
        assert!(!c.is_throttled(0));
    }

    #[test]
    fn throttle_ledger_on_cpu() {
        let mut c = cpu_state(3);
        c.set_throttled(1);
        assert!(c.is_throttled(1) && !c.is_throttled(0));
        c.throttle_all();
        assert!((0..3).all(|r| c.is_throttled(r)));
        c.clear_throttled();
        assert!((0..3).all(|r| !c.is_throttled(r)));
    }

    #[test]
    fn divergence_report_all_or_none_pre_norms() {
        let mut c = cpu_state(2);
        c.arm(Instant::now(), &[0, 0]);
        c.note_sync_ack(0, 1, Some(0.5), Some(1.0), Some(2.0));
        c.note_sync_ack(1, 1, Some(0.3), None, None);
        let report = c.divergence_report();
        assert_eq!(report.deltas, vec![0.5, 0.3]);
        assert!(report.pre_norms.is_none(), "partial pre-norms omitted");
        assert_eq!(report.post_norm, Some(2.0));
        c.reset_divergence_signals();
        assert!(c.divergence.iter().all(|d| d.is_none()));
        assert!(c.post_norm.is_none());
    }
}
