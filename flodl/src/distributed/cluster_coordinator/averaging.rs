//! Averaging-cycle methods for [`super::ClusterCoordinator`]: the
//! NCCL + CPU `trigger_averaging` / `finish_averaging_*` paths, the
//! 3-phase CPU averaging state machine, and the NCCL re-rendezvous
//! retry path.

use std::time::{Duration, Instant};

use crate::distributed::ddp_run::convergence::{self, ConvergenceAction};
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::{Result, TensorError};

use super::{ClusterCoordinator, CpuAvgState, EpochDSummary, NcclRendezvousPending};

impl ClusterCoordinator {
    /// Per-AllReduce d-aggregator update. Called once per
    /// `finish_averaging_{nccl,cpu}` after the convergence guard's
    /// `d_raw` + `k_max` are known. Mirrors threaded
    /// `coordinator/cpu_avg.rs::update_epoch_d_aggregator`.
    pub(super) fn update_epoch_d_aggregator(&mut self, d_raw: f64, k_max: usize) {
        self.epoch_d_count += 1;
        self.epoch_d_sum += d_raw;
        if d_raw < self.epoch_d_min {
            self.epoch_d_min = d_raw;
        }
        if d_raw > self.epoch_d_max {
            self.epoch_d_max = d_raw;
        }
        self.epoch_last_d = d_raw;
        self.epoch_last_k_max = k_max;
    }

    /// Drain the epoch d-aggregator + reset to identity. Called from
    /// the post-aggregate hook to build the `DivergenceEpoch` event
    /// payload. Mirrors threaded
    /// `coordinator/cpu_avg.rs::take_epoch_d_summary`.
    pub(super) fn take_epoch_d_summary(&mut self) -> EpochDSummary {
        let snap = EpochDSummary {
            count: self.epoch_d_count,
            d_min: self.epoch_d_min,
            d_max: self.epoch_d_max,
            d_sum: self.epoch_d_sum,
            d_at_epoch_end: self.epoch_last_d,
            k_at_epoch_end: self.epoch_last_k_max,
        };
        self.epoch_d_min = f64::INFINITY;
        self.epoch_d_max = f64::NEG_INFINITY;
        self.epoch_d_sum = 0.0;
        self.epoch_d_count = 0;
        snap
    }

    pub(super) fn capture_nccl_sync_elapsed_if_complete(&mut self) {
        if self.nccl_ack.iter().all(|&a| a) {
            if let Some(start) = self.nccl_sync_start.take() {
                self.last_nccl_sync_ms =
                    start.elapsed().as_secs_f64() * 1000.0;
            }
        }
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
        // Open a SyncStart window on the shared timeline so the user-
        // side `summary.sync_count` reflects this averaging cycle.
        // `sync_start` records wall-clock for the matching SyncEnd's
        // `duration_ms` in `finish_averaging_*`. Mirrors the threaded
        // coord (ddp_run/coordinator/cpu_avg.rs:124).
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::SyncStart);
        }
        self.sync_start = Some(Instant::now());
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
                // CpuAvgStart: open the Pending window. Mirrors threaded
                // `coordinator/cpu_avg.rs:634`. Closed in
                // `finish_averaging_cpu` via the `cpu_avg_start` field
                // so MSF / dashboard see the same `CpuAvgEnd
                // { duration_ms }` payload shape on cluster runs.
                self.cpu_avg_start = Some(Instant::now());
                if let Some(ref tl) = self.timeline {
                    tl.event(crate::monitor::EventKind::CpuAvgStart);
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
    pub(super) fn poll_cpu_averaging(&mut self) -> Result<()> {
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

    /// NCCL-backend re-rendezvous initiation. Called from
    /// [`Self::check_dead_ranks`] once a rank has been declared dead
    /// on the NCCL path. No-op on CPU backend (the controller-side
    /// release handles CPU AllReduces). No-op when a rendezvous is
    /// already pending — additional deaths during the wait will be
    /// rolled into the same rendezvous when it completes
    /// (`broadcast_new_nccl_session` reads the *current* alive set,
    /// not the snapshot from rendezvous-initiation time).
    pub(super) fn initiate_nccl_rendezvous_if_needed(&mut self) -> Result<()> {
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
    /// [`super::NCCL_RENDEZVOUS_TIMEOUT_SECS`]. Picks the next candidate from
    /// the rendezvous's `survivors_ordered` after filtering out dead
    /// ranks and already-tried generators. When the candidate pool is
    /// exhausted, falls back to
    /// [`Self::dispatch_shutdown_with_save`] so the cluster doesn't
    /// hang forever on a dead-on-arrival generator chain.
    ///
    /// No-op when no rendezvous is pending, when the backend isn't
    /// NCCL, or when the generator is still alive and inside its
    /// timeout window.
    pub(super) fn check_rendezvous_timeout(&mut self) {
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
    pub(super) fn pick_uid_generator(&self, survivors_ordered: &[usize]) -> usize {
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
    pub(super) fn per_rank_ms_per_batch(&self, r: usize) -> f64 {
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
    pub(super) fn broadcast_new_nccl_session(&mut self, uid_bytes: Vec<u8>) -> Result<()> {
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

    pub(super) fn finish_averaging_nccl(&mut self) -> Result<()> {
        let prev_sync_ms = self.last_nccl_sync_ms;
        self.last_nccl_sync_ms = 0.0;
        // Snapshot anchor BEFORE the guard verdict + meta-nudge so the
        // post-cycle `AnchorChanged` event captures the cycle's net
        // change. Mirrors threaded `coordinator/cpu_avg.rs:230`.
        let old_anchor = self.el_che.anchor();
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

        // Per-AllReduce divergence event + epoch aggregator update.
        // `d_raw` is the max relative delta across ranks for this
        // cycle; the epoch-level aggregator drains in
        // `try_advance_or_shutdown_after_aggregate`. Lambda fields are
        // intentionally None — analyze.rs recomputes guard-specific
        // λ̂ from observables now that the guard pipeline is plural.
        // Mirrors threaded `coordinator/cpu_avg.rs:299-334`.
        let d_raw = report.max_relative_delta();
        self.update_epoch_d_aggregator(d_raw, k_max);
        let in_flight_epoch = self.last_aggregated_epoch.map(|e| e + 1).unwrap_or(0);
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::Divergence {
                d_raw,
                lambda_raw: None,
                lambda_ema: None,
                k_used: cycle_batches,
                k_max,
                step: self.global_step,
                deltas: report.deltas.clone(),
                post_norm: report.post_norm,
                pre_norms: report.pre_norms.clone(),
                epoch: Some(in_flight_epoch),
            });
            let telemetry = self.convergence_guard.telemetry();
            if !telemetry.is_empty() {
                tl.event(crate::monitor::EventKind::GuardTelemetry {
                    epoch: in_flight_epoch,
                    step: self.global_step,
                    values: telemetry
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v))
                        .collect(),
                });
            }
            let new_anchor = self.el_che.anchor();
            if new_anchor != old_anchor {
                tl.event(crate::monitor::EventKind::AnchorChanged {
                    from: old_anchor,
                    to: new_anchor,
                });
            }
        }

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
        self.emit_sync_end();
        Ok(())
    }

    /// Close the SyncStart window opened in `trigger_averaging`. Emits
    /// `SyncEnd { duration_ms }` on the shared timeline if one is
    /// attached and a `sync_start` was recorded. No-op otherwise.
    /// Called from the end of both `finish_averaging_nccl` and
    /// `finish_averaging_cpu`.
    fn emit_sync_end(&mut self) {
        if let Some(start) = self.sync_start.take() {
            if let Some(ref tl) = self.timeline {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                tl.event(crate::monitor::EventKind::SyncEnd { duration_ms });
            }
        }
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
    pub(super) fn finish_averaging_cpu(&mut self) -> Result<()> {
        let prev_sync_ms = self.last_nccl_sync_ms;
        self.last_nccl_sync_ms = 0.0;
        let old_anchor = self.el_che.anchor();
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

        // Per-AllReduce divergence event + per-epoch aggregator update;
        // see `finish_averaging_nccl` for the rationale.
        let d_raw = report.max_relative_delta();
        self.update_epoch_d_aggregator(d_raw, k_max);
        let in_flight_epoch = self.last_aggregated_epoch.map(|e| e + 1).unwrap_or(0);
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::Divergence {
                d_raw,
                lambda_raw: None,
                lambda_ema: None,
                k_used: cycle_batches,
                k_max,
                step: self.global_step,
                deltas: report.deltas.clone(),
                post_norm: report.post_norm,
                pre_norms: report.pre_norms.clone(),
                epoch: Some(in_flight_epoch),
            });
            let telemetry = self.convergence_guard.telemetry();
            if !telemetry.is_empty() {
                tl.event(crate::monitor::EventKind::GuardTelemetry {
                    epoch: in_flight_epoch,
                    step: self.global_step,
                    values: telemetry
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v))
                        .collect(),
                });
            }
            let new_anchor = self.el_che.anchor();
            if new_anchor != old_anchor {
                tl.event(crate::monitor::EventKind::AnchorChanged {
                    from: old_anchor,
                    to: new_anchor,
                });
            }
        }

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
        // CpuAvgEnd: close the Pending window opened in trigger_averaging.
        // duration_ms is the bridge round-trip time (RequestParams
        // broadcast → all alive SyncAcks landed). Distinct from the
        // outer SyncEnd which also covers the post-finalize work.
        if let Some(start) = self.cpu_avg_start.take() {
            if let Some(ref tl) = self.timeline {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                tl.event(crate::monitor::EventKind::CpuAvgEnd { duration_ms });
            }
        }
        self.emit_sync_end();
        Ok(())
    }
}
