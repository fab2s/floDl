//! Averaging-cycle policy hooks for [`super::ClusterCoordinator`]:
//! the `trigger_averaging` dispatcher (shared preamble, then the
//! per-backend arm in `cycle_nccl.rs` / `cycle_cpu.rs`), the window
//! feed (`build_window_report` and its coherence attestation), and
//! the `finish_averaging_head` feedback half (ElChe verdict + guard +
//! meta-controller + telemetry). Transport mechanics live in the
//! `cycle_*` siblings; this file decides and retunes, it does not
//! move bytes.

use std::time::Instant;

use crate::distributed::ddp_run::convergence::ConvergenceAction;
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::distributed::el_che::{AnchorVerdict, WindowReport};
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::Result;

use super::{ClusterCoordinator, EpochDSummary};

/// The epoch a reduce cycle's work belongs to.
///
/// `last_aggregated + 1` is the epoch in flight — until the final epoch has
/// been aggregated, when nothing is in flight and a reduce still settling
/// carries the last epoch's residual work. Unclamped, that trailing window
/// report (and any alert riding the same value) is filed under an epoch the
/// run never runs: a 40-epoch run ended with a window row labelled epoch 41.
/// The work in that row is real; the epoch it was filed under was not.
fn in_flight_epoch(last_aggregated: Option<usize>, num_epochs: usize) -> usize {
    match last_aggregated {
        Some(e) => (e + 1).min(num_epochs.saturating_sub(1)),
        None => 0,
    }
}

impl ClusterCoordinator {
    /// Per-AllReduce d-aggregator update. Called once per
    /// `finish_averaging_{nccl,cpu}` after the convergence guard's
    /// `d_raw` + `k_max` are known.
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
    /// payload.
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

    /// The per-rank `(ms, batches)` pair fed to
    /// [`crate::distributed::ElChe::report_timing`] at each averaging cycle.
    /// ElChe derives `ms_per_batch[r] = ms[r] / batches[r]`.
    ///
    /// **Cadence and Async** (both progressive) feed the rank-reported
    /// DELIVERED cost — the window ledger's marginal delivered ms (Σ
    /// per-batch `batch_ms + data_ms` = compute + data) over its MATCHED
    /// batch count. Accumulated CONTINUOUSLY from each `Batch`
    /// report (see `event_loop`), so it is present at the reduce by
    /// construction — no completion-frame race. ElChe then schedules
    /// per-rank windows on realized wall instead of the compute-only
    /// wall (Σ per-batch `train_step` ms). This closes the
    /// cpu-cadence idle (a data-starved rank's delivered cost rises, so the
    /// balancer stops over-allocating the fast rank) AND makes the nccl
    /// path data-/transport-aware — required when identical GPUs sit at
    /// different network distances or behind asymmetric storage.
    ///
    /// The matched divisor is what makes this safe on BOTH backends. ms and
    /// batch count accumulate TOGETHER per `Batch`, so even when NCCL's
    /// `finish_averaging_nccl` runs INLINE in `trigger_averaging` (before
    /// the window's last completion drains), dividing the delivered sum by
    /// ITS OWN batch count yields a correct per-batch estimate — and a late
    /// batch leaking into the next window is benign (ms and count leak
    /// together). The accumulator is MARGINAL: the window's FIRST batch is
    /// routed to the fill slot by `WindowLedger::record_batch` so the
    /// per-chunk fixed fill cost never enters the quoted per-batch rate.
    ///
    /// Per-rank fallback to the compute-only `(wall, steps)` pair when a
    /// rank has no delivered sample this window (cold-start, or a
    /// single-batch window whose only batch the marginal skip dropped) so
    /// no spurious zero / zero-ms report poisons ElChe's trust window.
    ///
    /// **Sync** (non-progressive) keeps the compute-only `(wall, steps)`
    /// feed unchanged. Every alive mover (steps > 0) has a delivered
    /// sample this window (nonzero ms AND batches). This is both the
    /// all-or-none coherence predicate for the delivered feed in
    /// [`Self::build_window_report`] and the settle condition for
    /// `trigger_averaging`'s pre-finish drain.
    pub(super) fn movers_delivered_complete(&self) -> bool {
        (0..self.world_size)
            .filter(|&r| !self.is_dead(r) && self.window.steps(r) > 0)
            // REPORT-AT-SYNC: the delivered sample is present at the reduce
            // by construction from the continuous `Batch` reports — true for
            // every stepping rank with >= 2 batches this window. A
            // single-batch window (marginal skipped its only batch) has no
            // sample -> coherent compute-scale fallback for that (rare)
            // window.
            .all(|r| self.window.has_delivered_sample(r))
    }

    /// Whether this run's mode may ride the delivered timing scale at
    /// all: CPU Cadence/Async + NCCL Cadence. NCCL Cadence is
    /// transport-aware because the ledger's delivered pair accumulates
    /// continuously per `Batch`, so it is present at the inline finish
    /// by construction — the completion-frame race that originally
    /// forced NCCL onto the compute-only feed is gone. Without the
    /// delivered feed, NCCL allocation is blind to data + transport
    /// (x1-link rig: shares [0.53, 0.235, 0.235] vs the true ~4.9×
    /// delivered ratio → fast rank ~45% idle at every barrier). NCCL
    /// Async stays excluded: overshoot streaming under the inline
    /// finish is unvalidated there. Sync (non-progressive) always
    /// feeds the compute-only scale.
    fn delivered_capable(&self) -> bool {
        match self.backend {
            AverageBackend::Cpu => {
                matches!(self.policy, ApplyPolicy::Cadence | ApplyPolicy::Async)
            }
            AverageBackend::Nccl => matches!(self.policy, ApplyPolicy::Cadence),
        }
    }

    /// Assemble this window's [`WindowReport`] from the ledger — the
    /// event the coordinator feeds `ElChe::report_window` once per
    /// averaging cycle.
    ///
    /// The `delivered_coherent` attestation is the coordinator's half
    /// of the mixed-scale inversion guard (the scale-SELECTION half
    /// lives in `WindowReport::select_feed`, next to ElChe's relative
    /// allocation model): the delivered scale is only offered when the
    /// mode supports it AND every alive mover has a delivered sample
    /// this window ([`Self::movers_delivered_complete`] — the
    /// coordinator owns that predicate because it owns membership and
    /// the ledger). A single mover on the compute scale against peers
    /// on delivered would invert the allocation (rig: equal-speed
    /// Pascals drifting to 0.33 vs 0.10 shares on cpu-async; the
    /// x1-link Pascal drawing ~73% of all steps and diverging to NaN).
    pub(super) fn build_window_report(&self, sync_ms: f64) -> WindowReport {
        WindowReport {
            wall_ms: self.window.wall_ms_all().to_vec(),
            steps: self.window.steps_all().to_vec(),
            delivered_ms: self.window.delivered_ms_all().to_vec(),
            delivered_batches: self.window.delivered_batches_all().to_vec(),
            fill_ms: (0..self.world_size)
                .map(|r| self.window.fill_excess_ms(r))
                .collect(),
            delivered_coherent: self.delivered_capable() && self.movers_delivered_complete(),
            sync_ms,
        }
    }

    /// `-vvv` delivered-vs-compute per-cycle dump (Cadence + Async — the
    /// progressive policies that ride the delivered feed). Surfaces the gap
    /// the fix closes: `pb_delivered_ms/batch` (what ElChe schedules on,
    /// over the matched divisor) vs `compute_ms/batch` (what it used to,
    /// over `steps_since_avg`), per rank, against the resulting
    /// `batch_counts`. Call BEFORE the per-cycle counter resets. No-op
    /// unless `-vvv`.
    fn dump_delivered_timing(&self, reduce_ms: f64) {
        if !self.prof_enabled || !matches!(self.policy, ApplyPolicy::Cadence | ApplyPolicy::Async) {
            return;
        }
        let r1 = |v: &[f64]| -> Vec<f64> { v.iter().map(|m| (m * 10.0).round() / 10.0).collect() };
        let compute_per_batch: Vec<f64> = (0..self.world_size)
            .map(|r| {
                let n = self.window.steps(r).max(1);
                self.window.wall_ms(r) / n as f64
            })
            .collect();
        // The feed: rank-reported DELIVERED (compute+data), accumulated
        // continuously per `Batch` (marginal), present at sync by
        // construction.
        let pb_delivered_per_batch: Vec<f64> = (0..self.world_size)
            .map(|r| {
                let n = self.window.delivered_batches(r).max(1);
                self.window.delivered_ms(r) / n as f64
            })
            .collect();
        // Which feed did ElChe actually schedule on this cycle? `delivered`
        // means every stepping rank had a delivered sample;
        // `COMPUTE-FALLBACK` means the all-or-none coherence gate
        // ([`Self::movers_delivered_complete`]) dropped the WHOLE cohort to
        // compute-only because at least one mover lacked one. `missing`
        // names those movers — the culprits that trip the fallback. A run
        // that alternates delivered / COMPUTE-FALLBACK is mixing scales
        // across windows.
        let feed = if self.movers_delivered_complete() {
            "delivered"
        } else {
            "COMPUTE-FALLBACK"
        };
        let missing: Vec<usize> = (0..self.world_size)
            .filter(|&r| {
                !self.is_dead(r) && self.window.steps(r) > 0 && !self.window.has_delivered_sample(r)
            })
            .collect();
        eprintln!(
            "[coord-prof] {:?} {:?} | feed={feed} missing={missing:?} \
             pb_delivered_ms/batch={:?} compute_ms/batch={:?} steps={:?} \
             pb_batches={:?} batch_counts={:?} reduce_ms={:.1}",
            self.backend,
            self.policy,
            r1(&pb_delivered_per_batch),
            r1(&compute_per_batch),
            self.window.steps_all(),
            self.window.delivered_batches_all(),
            self.el_che.batch_counts(),
            reduce_ms,
        );
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
    ///   channel (`CpuReduceClient`)
    ///   between RequestParams and the next round.
    pub fn trigger_averaging(&mut self) -> Result<()> {
        // Open a SyncStart window on the shared timeline so the user-
        // side `summary.sync_count` reflects this averaging cycle.
        // `sync_start` records wall-clock for the matching SyncEnd's
        // `duration_ms` in `finish_averaging_*`.
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::SyncStart);
        }
        self.sync_start = Some(Instant::now());
        // CHECKPOINT ARM (before any RequestParams/SyncNow broadcast = before the
        // param freeze): if a checkpoint is due this reduce, capture coverage now
        // (so `covered ⊆ consensus`; a post-reduce capture would over-count under
        // async overshoot → lost data) and arm the consensus model write. The
        // `.meta.json` is written at the matching `finish_averaging_*` from the
        // stashed coverage + final counters.
        self.maybe_arm_checkpoint();
        // EVAL ARM (CPU path, same placement rationale): the arm frame must
        // precede this round's RequestParams on the control channel (FIFO),
        // so the elected rank is armed before it snapshots and the round's
        // realized Update fires the consensus eval.
        self.maybe_arm_eval();
        match self.backend {
            AverageBackend::Nccl => self.arm_nccl_cycle(),
            AverageBackend::Cpu => self.arm_cpu_cycle(),
        }
    }

    /// Shared first half of `finish_averaging_nccl` / `finish_averaging_cpu`:
    /// feed ElChe the window timing, run the convergence-guard verdict +
    /// LR-aware meta-controller, apply the anchor action, bump
    /// `version` / `avg_count` / `global_step`, and emit the per-cycle
    /// telemetry (Divergence / GuardTelemetry / AnchorChanged). Everything
    /// here is backend-independent; the per-backend middle (NCCL
    /// `SetGlobalStep` vs CPU fold + `Update` fan-out) follows it.
    pub(super) fn finish_averaging_head(&mut self) {
        let prev_sync_ms = self.cycle.take_last_sync_ms();
        // Snapshot anchor BEFORE the guard verdict + meta-nudge so the
        // post-cycle `AnchorChanged` event captures the cycle's net
        // change.
        let old_anchor = self.el_che.anchor();
        // Stage per-rank callback slack BEFORE report_timing so the
        // recompute inside ElChe applies it to the next cycle's
        // batch_counts (when the next cycle is the LAST cycle of the
        // current epoch).
        self.maybe_apply_callback_slack_for_next_cycle();
        // ONE timing event per cycle: the window's observations (both
        // scales + fill + the delivered-coherence attestation) go to
        // ElChe as a WindowReport; scale selection and the fill staging
        // happen inside `report_window` (a fully-idle window reports
        // nothing, so no zero-ms sample poisons the trust windows).
        let report_window = self.build_window_report(prev_sync_ms);
        self.el_che.report_window(&report_window);
        if !self.calibrated && self.el_che.is_calibrated() {
            self.calibrated = true;
        }
        self.dump_delivered_timing(prev_sync_ms);

        let report = self.cycle.divergence_report();
        let cycle_batches: usize = self.window.total_steps();
        let k_max = self.window.max_steps();
        let action = self.convergence_guard.report(&report, cycle_batches, k_max);

        // LR-aware meta-controller (OLD `observe_meta` parity): consult
        // the meta after the guard verdict; a `NudgeDown` MetaAction
        // dispatches to `el_che.nudge_anchor_down` and composes
        // multiplicatively with the guard's own anchor adjustment
        // below.
        self.observe_meta(action);

        self.version += 1;
        self.avg_count += 1;

        // Map the guard's action onto the source-agnostic verdict seam
        // (`ElChe::apply_verdict`); the coordinator's own overshoot knob
        // is mutated alongside — it is scheduling state ElChe does not
        // own. On Stable, ElChe may grow the window to amortize sync
        // cost; convergence is maintained separately by SuppressGrowth /
        // NudgeDown pulling the anchor back when weight-space divergence
        // rises — growth and convergence balance rather than being
        // hard-disabled.
        // The guard CAPS the overshoot budget; it no longer grows it. The
        // old `+1 per Stable` was a hill-climb toward a quantity that is
        // directly computable from the allocation and the measured reduce
        // (see `recompute_overshoot_budget`), and it could not arrive: at
        // +1 per cycle from an initial of 3, a run with 24 reduces tops out
        // at the ceiling of 15 while the rig's own measurements put full
        // cover near 110 for the fast rank. Derivation replaces the search;
        // the guard keeps its veto.
        match action {
            ConvergenceAction::Stable => {
                self.el_che.apply_verdict(AnchorVerdict::Stable {
                    relax_up: self.policy == ApplyPolicy::Async && self.elche_relax_up,
                });
                self.overshoot_suppressed = false;
            }
            ConvergenceAction::SuppressGrowth => {
                self.el_che.apply_verdict(AnchorVerdict::SuppressGrowth);
                self.overshoot_suppressed = true;
            }
            ConvergenceAction::NudgeDown { factor } => {
                self.el_che
                    .apply_verdict(AnchorVerdict::NudgeDown { factor });
                self.overshoot_suppressed = true;
            }
        }
        // Re-derive AFTER `report_window` (so `batch_counts` is this
        // window's) and after the verdict (so a suppressing verdict holds
        // the budget where it is). `prev_sync_ms` is the same measurement
        // ElChe just scheduled on, so the two cannot disagree about what
        // the reduce cost.
        self.recompute_overshoot_budget(prev_sync_ms);

        self.global_step += cycle_batches;

        // Per-AllReduce divergence event + epoch aggregator update.
        // `d_raw` is the max relative delta across ranks for this
        // cycle; the epoch-level aggregator drains in
        // `try_advance_or_shutdown_after_aggregate`. Lambda fields are
        // intentionally None — analyze.rs recomputes guard-specific
        // λ̂ from observables now that the guard pipeline is plural.
        let d_raw = report.max_relative_delta();
        self.update_epoch_d_aggregator(d_raw, k_max);
        let in_flight_epoch = in_flight_epoch(self.last_aggregated_epoch, self.num_epochs);

        // `drift` alert. The trigger is the guard's own `NudgeDown` verdict,
        // not a raw `d_raw` threshold invented here: the configured guard IS
        // the divergence threshold (LevelGuard's `divergence_threshold`, GrowthGuard's
        // λ̂, NoGuard's silence), and a second coordinator-side threshold would
        // read a different scale and disagree with it. `SuppressGrowth` is a
        // hold, not a correction, so it stays informational. A sustained bad
        // regime nudges every cycle — the lane's collapse window is what keeps
        // that to one alert per window.
        if let ConvergenceAction::NudgeDown { factor } = action {
            self.emit_alert(
                crate::monitor::event_lane::EventClass::Drift,
                "root".to_string(),
                format!(
                    "divergence {d_raw:.3e} — convergence guard nudged the \
                     anchor down (x{factor})",
                ),
            );
        }
        // Stash for the window report's curated `d_raw` / `lambda_ema`
        // (independent of the timeline sinks below). The lambda proxy comes
        // from the active guard's telemetry, so a guard that computes none
        // (NoGuard, LevelGuard) simply never sets it.
        self.last_divergence_d = Some(d_raw);
        if let Some((_, v)) = self
            .convergence_guard
            .telemetry()
            .iter()
            .find(|(k, _)| *k == "lambda_ema")
        {
            self.last_lambda_ema = Some(*v);
        }
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

        // Sub-epoch monitor report, gated by the `reports_per_epoch`
        // cadence. Last in the head so it observes the window's final
        // state (post `global_step` / `avg_count` bump) and can never
        // perturb the feedback path above it. Reads the ledger before
        // `finish_averaging_tail` resets it.
        self.maybe_emit_window_report(cycle_batches, in_flight_epoch);
    }

    /// Emit one per-window monitor record tree when the
    /// `reports_per_epoch` cadence fires at this reduce boundary.
    ///
    /// Rides the existing step clock as a read-only observer: it consumes
    /// the same `cycle_batches` the window just realized and never gates,
    /// delays, or reschedules anything. Disabled (`reports_per_epoch`
    /// unset) it costs one `Option` check per cycle.
    ///
    /// The `epoch` field is a label; a marker for "this tick closed the
    /// epoch" belongs to the slice that folds the per-epoch feed into the
    /// same record stream (the per-epoch `push_epoch_metrics` channel is
    /// untouched here).
    fn maybe_emit_window_report(&mut self, cycle_batches: usize, in_flight_epoch: usize) {
        if self.report_scheduler.is_none() || self.dashboard_sink.is_none() {
            return;
        }
        // Epoch rollover: restart the cadence so every epoch gets its own
        // `reports_per_epoch` budget (a single-epoch run simply never rolls).
        if self.report_epoch_seen != in_flight_epoch {
            self.report_epoch_seen = in_flight_epoch;
            self.report_in_epoch_steps = 0.0;
            if let Some(s) = self.report_scheduler.as_mut() {
                s.reset_epoch();
            }
        }
        self.report_in_epoch_steps += cycle_batches as f64;
        let in_epoch_work = self.report_in_epoch_steps;
        let fires = self
            .report_scheduler
            .as_mut()
            .is_some_and(|s| s.on_sync(in_epoch_work));
        if !fires {
            return;
        }

        // Drain each rank's accumulated interval. Draining resets, so the next
        // window summarises only its own samples and a window that saw none
        // leaves `res` absent rather than repeating a stale reading.
        let res_per_rank: Vec<crate::monitor::record::Res> = (0..self.world_size)
            .map(|r| match self.latest_res.get_mut(r) {
                Some(acc) => acc.take(),
                None => crate::monitor::record::Res::default(),
            })
            .collect();

        let stats: Vec<super::window_records::WindowRankStat> = (0..self.world_size)
            .map(|r| {
                // Marginal delivered rate = the honest per-rank capacity
                // signal (same feed ElChe schedules on), in samples/ms.
                let throughput = if self.window.has_delivered_sample(r) {
                    let ms = self.window.delivered_ms(r);
                    let batches = self.window.delivered_batches(r);
                    Some((batches * self.batch_size) as f64 / ms)
                } else {
                    None
                };
                super::window_records::WindowRankStat {
                    rank: r,
                    host: self.rank_hosts.get(r).cloned().unwrap_or_default(),
                    device: self.metrics_device_indices.get(r).copied(),
                    alive: !self.is_dead(r),
                    steps: self.window.steps(r),
                    mean_loss: self.window.mean_loss(r),
                    throughput,
                    compute_only_ms: self.window.wall_ms(r),
                    res: res_per_rank[r],
                }
            })
            .collect();

        let ts = super::alerts::now_ms();
        let mut tree = super::window_records::build_window_tree(&stats);
        super::window_records::insert_engine_metrics(
            &mut tree,
            &super::window_records::EngineWindow {
                anchor: self.el_che.anchor(),
                sync_ms: self.last_sync_ms,
                cpu_avg_ms: self.last_cpu_avg_ms,
                d_raw: self.last_divergence_d,
                lambda_ema: self.last_lambda_ema,
            },
        );
        let records = tree.flat_records(ts, Some(self.avg_count), Some(in_flight_epoch));
        if let Some(sink) = self.dashboard_sink.as_ref() {
            sink.push_window_records(records);
        }
    }

    /// Shared second half of `finish_averaging_nccl` / `finish_averaging_cpu`:
    /// reset the window accumulators, clear the throttle / HOLD / divergence
    /// slots, and kick idle progressive ranks back into motion. Callers
    /// finish with their own end-of-cycle events (`CpuAvgEnd`,
    /// `emit_sync_end`). `steps_since_avg` is NOT reset here — its
    /// placement is backend-specific (the CPU path must reset BEFORE the
    /// atomic-dispatch fold so `cap_to_reduce_budget` sees the fresh
    /// window).
    pub(super) fn finish_averaging_tail(&mut self) {
        // Window timing (compute wall + delivered + fill) resets with the
        // window; step counts reset backend-specifically (see callers).
        self.window.reset_timing();
        self.cycle.clear_throttled();
        self.dispatch_hold_logged.fill(false);
        self.cycle.reset_divergence_signals();
        // Overshoot gate is open again — kick any rank still sitting in
        // `wait_for_epoch_plan` (gated, or just finished its last chunk
        // before the cycle) so progressive dispatch doesn't stall until
        // the next epoch-aggregate hook.
        self.wake_idle_ranks_in_progressive();
    }

    /// Close the SyncStart window opened in `trigger_averaging`. Emits
    /// `SyncEnd { duration_ms }` on the shared timeline if one is
    /// attached and a `sync_start` was recorded. No-op otherwise.
    /// Called from the end of both `finish_averaging_nccl` and
    /// `finish_averaging_cpu`.
    pub(super) fn emit_sync_end(&mut self) {
        if let Some(start) = self.sync_start.take() {
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            // Stash for the window report's curated `sync_ms` regardless of
            // whether a timeline is attached — the record stream and the
            // timeline are independent sinks.
            self.last_sync_ms = Some(duration_ms);
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::SyncEnd { duration_ms });
            }
        }
    }

    /// ARM a coverage-granular consensus checkpoint at the START of a reduce
    /// cycle (called from `trigger_averaging`, before any
    /// `RequestParams`/`SyncNow` broadcast — i.e. before the workers freeze
    /// their params for this reduce).
    ///
    /// Four independent triggers share the one capture+arm tail:
    /// - **One-shot** (`checkpoint_at_epoch`): the first reduce where any
    ///   live rank has reached the target epoch; disarmed on fire.
    /// - **Recurring cadence** (`checkpoint_every`): the first reduce after
    ///   the cohort crosses a new multiple of `checkpoint_every`, once per
    ///   multiple (`last_cadence_arm_epoch`). Two independent consumers: the
    ///   consensus bundle (`save_path` — zero extra communication, the frame
    ///   is already in stream; each write atomically supersedes the previous
    ///   `<stem>.fdl`, so the bundle is always the latest resume point) and,
    ///   on the CPU path, the controller-side `checkpoint_fn` fire (the
    ///   launch-wrapped callback on the forge). NCCL's `checkpoint_fn` keeps
    ///   its elected-rank wire dispatch in `dispatch_epoch`.
    /// - **Cooperative intent** (`request_checkpoint`, CPU path): served at
    ///   this reduce; see the inline comment.
    /// - **Final consensus** (`save_path`): every reduce of the run's tail —
    ///   the final epoch with less than one window of work remaining
    ///   ([`Self::run_tail_within_one_window`]) — so a natural clean end
    ///   always persists the final consensus bundle, single-epoch runs
    ///   included. The last realized reduce (the forced end-of-run reduce
    ///   when trailing steps exist, the boundary-landing window otherwise)
    ///   wins by construction; a zero-mass tail round is skipped by the
    ///   forge's realized-work guard and leaves the previous write standing
    ///   (zero mass ⇔ zero new steps ⇔ coverage unchanged, so bundle and
    ///   meta stay paired).
    ///
    /// The shared tail:
    /// 1. captures coverage NOW (`snapshot_coverage`) and stashes it in
    ///    `pending_checkpoint_coverage` for the matching `finish_*` to write —
    ///    capturing before the param freeze guarantees `covered ⊆ consensus`
    ///    (a chunk completed-and-drained before the freeze is provably in each
    ///    rank's frozen params; anything later is recorded uncovered → bounded
    ///    redo on resume, never lost data);
    /// 2. arms the consensus MODEL write — CPU: the controller reduce thread
    ///    taps this round's averaged frame
    ///    ([`crate::distributed::CheckpointForge::arm`]); NCCL: elected-rank
    ///    post-collective write, dispatched from
    ///    `finish_pending_checkpoint_meta`.
    pub(super) fn maybe_arm_checkpoint(&mut self) {
        let cluster_epoch = (0..self.world_size)
            .filter(|&r| !self.is_dead(r))
            .map(|r| self.rank_epoch[r])
            .max()
            .unwrap_or(0);
        // One-shot: any live rank reached the target epoch (typically
        // mid-epoch, so the coverage block is non-trivial).
        let one_shot_due = self
            .checkpoint_at_epoch
            .is_some_and(|target| cluster_epoch >= target);
        // Recurring cadence: a new multiple of `checkpoint_every` crossed.
        // The crossing has two independent consumers — the bundle write
        // (needs `save_path`) and, on the CPU path, the controller-side
        // `checkpoint_fn` fire (needs the launch-wrapped callback on the
        // forge) — and advances once when either consumes it.
        let cadence_target = match self.checkpoint_every {
            Some(every) if every > 0 && cluster_epoch >= every => (cluster_epoch / every) * every,
            _ => 0,
        };
        let cadence_crossed = cadence_target > self.last_cadence_arm_epoch;
        let cpu = matches!(self.backend, AverageBackend::Cpu);
        let user_fn_ready = cpu
            && self
                .checkpoint_forge
                .as_ref()
                .is_some_and(|f| f.has_consensus_fn());
        let bundle_cadence_due = cadence_crossed && self.save_path.is_some();
        let user_cadence_due = cadence_crossed && user_fn_ready;
        // Cooperative `request_checkpoint` intent: on the CPU path it is
        // served HERE, at the next reduce (a coherent boundary that arrives
        // sooner than the next epoch); NCCL keeps the epoch-boundary wire
        // fold in `dispatch_epoch`. With NEITHER consumer configured the
        // request has nothing to do — drop it loudly instead of leaving it
        // pending forever (parity with the old elected-rank "checkpoint_fn
        // is None" error).
        let intent_due =
            cpu && self.pending_checkpoint_intent && (user_fn_ready || self.save_path.is_some());
        if cpu && self.pending_checkpoint_intent && !intent_due {
            self.pending_checkpoint_intent = false;
            eprintln!(
                "flodl ddp: request_checkpoint with no checkpoint_fn and no \
                 save_path configured; nothing to checkpoint"
            );
        }
        // Final consensus: the run's tail is in flight.
        let final_due = self.save_path.is_some()
            && self.num_epochs > 0
            && cluster_epoch + 1 >= self.num_epochs
            && self.run_tail_within_one_window(cluster_epoch);
        if !(one_shot_due || bundle_cadence_due || user_cadence_due || intent_due || final_due) {
            return;
        }
        if one_shot_due {
            self.checkpoint_at_epoch = None; // exactly once
        }
        if bundle_cadence_due || user_cadence_due {
            self.last_cadence_arm_epoch = cadence_target;
        }
        if intent_due {
            self.pending_checkpoint_intent = false;
        }
        // User-fire version: the crossed multiple for a cadence fire (the
        // same "entering epoch N" label the wire dispatch used), the
        // in-flight epoch for an on-request fire.
        let user_version = if user_cadence_due {
            Some(cadence_target as u64)
        } else if intent_due && user_fn_ready {
            Some(cluster_epoch as u64)
        } else {
            None
        };
        // Coverage + meta ride the bundle. The one-shot keeps its historical
        // shape: coverage is captured even without a save_path, and the
        // finish reports the missing destination loudly. A user-fire-only
        // arm writes no meta, so it captures nothing.
        let wants_bundle = one_shot_due
            || bundle_cadence_due
            || final_due
            || (intent_due && self.save_path.is_some());
        if wants_bundle {
            // Capture coverage at the freeze boundary; consumed at finish.
            self.pending_checkpoint_coverage = Some(self.snapshot_coverage());
        }
        match self.backend {
            AverageBackend::Cpu => {
                if let Some(forge) = self.checkpoint_forge.as_ref() {
                    let model_path = if wants_bundle {
                        match self.save_path.as_ref() {
                            Some(stem) if forge.can_write_model() => {
                                Some(crate::distributed::CheckpointBundle::model_path(stem))
                            }
                            Some(_) => {
                                crate::verbose!(
                                    "  ddp: checkpoint armed but no model schema captured; \
                                     writing meta-only (epoch {cluster_epoch})"
                                );
                                None
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                    forge.arm(model_path, user_version);
                }
            }
            AverageBackend::Nccl => {
                // NCCL consensus is on-device across ranks (nothing to arm
                // controller-side). The elected-rank model write is dispatched
                // from `finish_pending_checkpoint_meta` at the tail of
                // `finish_averaging_nccl`, AFTER the collective, so the rank
                // holds the post-collective consensus; the user callback keeps
                // its elected-rank wire dispatch in `dispatch_epoch`. Nothing
                // to do here beyond the coverage capture already done above.
            }
        }
    }

    /// Arm a consensus eval on the elected rank for the reduce round being
    /// triggered (CPU backend only — NCCL's eval stays boundary-dispatched
    /// in `dispatch_epoch`, where the post-collective model is the
    /// consensus). Two triggers, mirroring the checkpoint cadence shape:
    ///
    /// - **Recurring cadence** (`eval_every_epochs`): the first reduce after
    ///   the cohort crosses a new multiple, once per multiple
    ///   (`last_eval_arm_epoch`). Epochs are `epoch_splits` slices, so
    ///   single-pass runs get interior evals with no extra mechanism.
    /// - **Cooperative intent** (`request_eval`): served at this reduce —
    ///   sooner than the old epoch-boundary fold.
    ///
    /// The frame precedes the round's `RequestParams` broadcast (see the
    /// call site in `trigger_averaging`), so control-channel FIFO
    /// guarantees the rank is armed before it snapshots — the directive can
    /// never race its own round. Best-effort like the checkpoint dispatch:
    /// a missed arm is a missed cadence, and role failover re-targets on
    /// rank death. The worker scores the round's consensus and restores its
    /// state verbatim, so arming perturbs nothing.
    pub(super) fn maybe_arm_eval(&mut self) {
        if !matches!(self.backend, AverageBackend::Cpu) {
            return;
        }
        let cluster_epoch = (0..self.world_size)
            .filter(|&r| !self.is_dead(r))
            .map(|r| self.rank_epoch[r])
            .max()
            .unwrap_or(0);
        let cadence_target = match self.eval_every_epochs {
            Some(every) if every > 0 && cluster_epoch >= every => (cluster_epoch / every) * every,
            _ => 0,
        };
        let cadence_crossed = cadence_target > self.last_eval_arm_epoch;
        let intent_due = self.pending_eval_intent;
        if !(cadence_crossed || intent_due) {
            return;
        }
        if cadence_crossed {
            self.last_eval_arm_epoch = cadence_target;
        }
        self.pending_eval_intent = false;
        // Version label mirrors the checkpoint arm: the crossed multiple
        // for a cadence fire, the in-flight epoch for an on-request fire.
        let label = if cadence_crossed {
            cadence_target as u64
        } else {
            cluster_epoch as u64
        };
        let target = self.eval_role;
        if target >= self.world_size || self.is_dead(target) {
            return;
        }
        let msg = ControlMsgWire::ArmConsensusEval {
            schedule_id: label,
            epoch: label,
            target_rank: target as u64,
        };
        if let Err(e) = self.send_control(target, &msg) {
            eprintln!(
                "flodl ddp: consensus-eval arm to rank {target} failed \
                 (epoch {cluster_epoch}): {e}"
            );
        }
    }

    /// Whether the run's remaining work fits within one reduce window: the
    /// tail criterion the final-consensus arm in
    /// [`Self::maybe_arm_checkpoint`] checks for `epoch` (the in-flight FINAL
    /// epoch). An already-aggregated `epoch` is the tail unconditionally —
    /// see the first check below, which also covers its removed pool.
    /// Otherwise, same near-empty shape as `refresh_final_window_plan`
    /// (`remaining < Σcounts + world_size`), but policy-independent —
    /// progressive dispatch reads the epoch pool's undispatched remainder,
    /// non-progressive derives the remainder from the cached epoch plans
    /// minus each live rank's steps since epoch start. Both remainders can
    /// lag reality by up to a heartbeat (undispatched work excludes in-flight
    /// chunks; step counters ride timing reports), which only ever fires the
    /// arm a window early — an extra superseded bundle write, never a miss.
    pub(super) fn run_tail_within_one_window(&self, epoch: usize) -> bool {
        // An epoch that has already aggregated has zero work remaining by
        // definition — the deepest possible tail. This must be answered
        // BEFORE the pool lookup below: `aggregate_ready_epochs` REMOVES
        // an aggregated epoch's chunk pool, so a bare lookup misreads the
        // absence as "not dispatched yet" and the forced post-aggregate
        // end-of-run reduce becomes unarmable — the natural-end bundle
        // then silently never writes (the CI-only flake's second layer,
        // caught by the forge forensic counters: arms == 0 on a run whose
        // every step had provably been reduced).
        if self.last_aggregated_epoch.is_some_and(|agg| agg >= epoch) {
            return true;
        }
        let counts = self.el_che.batch_counts();
        let alive = |r: &usize| !self.is_dead(*r);
        let total_counts: usize = (0..self.world_size)
            .filter(alive)
            .map(|r| counts.get(r).copied().unwrap_or(0))
            .sum();
        let remaining = if self.progressive {
            let Some(pool) = self.chunk_pools.get(&epoch) else {
                return false; // final epoch not dispatched yet
            };
            pool.remaining() / self.batch_size
        } else {
            let Some(plans) = self.epoch_plan_cache.get(&epoch) else {
                return false; // final epoch not dispatched yet
            };
            (0..self.world_size)
                .filter(alive)
                .map(|r| {
                    let total_r = plans
                        .get(r)
                        .map(|p| p.partition_size as usize / self.batch_size)
                        .unwrap_or(0);
                    let done_r = self.last_step_count[r]
                        .saturating_sub(self.last_step_count_at_epoch_start[r]);
                    total_r.saturating_sub(done_r)
                })
                .sum()
        };
        if total_counts == 0 {
            // No schedule to size a window from (pre-calibration edge):
            // treat only a fully-drained tail as final.
            return remaining == 0;
        }
        remaining < total_counts + self.world_size
    }

    /// Write the stashed checkpoint `.meta.json` at the end of a reduce cycle
    /// (called from both `finish_averaging_*`). No-op unless
    /// `maybe_arm_checkpoint` captured coverage for this cycle. The meta pairs
    /// the trigger-time coverage with the now-final post-round counters
    /// (epoch / global_step / sync_round) + ElChe/guard state, stamped
    /// [`crate::distributed::SaveReason::Checkpoint`]. Mirrors the
    /// controller-side meta write in `dispatch_shutdown_with_save`.
    pub(super) fn finish_pending_checkpoint_meta(&mut self) {
        let Some(coverage) = self.pending_checkpoint_coverage.take() else {
            return;
        };
        let Some(stem) = self.save_path.as_ref() else {
            eprintln!(
                "flodl ddp: checkpoint coverage captured but save_path is unset; \
                 meta not written"
            );
            return;
        };
        let meta_path = crate::distributed::CheckpointBundle::meta_path(stem);
        // Cluster-wide epoch = max across live ranks (the highest any reached).
        let epoch = self.rank_epoch.iter().copied().max().unwrap_or(0);
        let mut elche_state = self.el_che.to_state();
        elche_state.trend_history = self.convergence_guard.trend_history();
        let meta = crate::distributed::CheckpointMeta::new(
            epoch,
            self.global_step,
            self.avg_count,
            self.world_size,
            crate::distributed::SaveReason::Checkpoint,
        )
        .with_elche_state(elche_state)
        .with_coverage(coverage);
        // Detach the meta file write so the checkpoint never touches the
        // training clock (matches the forge's detached `.fdl` write). The
        // meta is built synchronously from coordinator state above (cheap);
        // only the serialize + atomic disk write runs off-thread. Ownership
        // of `meta` + `meta_path` moves into the writer.
        let version = self.version;
        let spawn = std::thread::Builder::new()
            .name("flodl-ckpt-meta".to_string())
            .spawn(move || match meta.write_to_file(&meta_path) {
                Ok(()) => crate::verbose!(
                    "  ddp: checkpoint meta written {} (epoch {epoch}, version {version})",
                    meta_path.display(),
                ),
                Err(e) => eprintln!(
                    "flodl ddp: checkpoint meta write to {} failed: {e}",
                    meta_path.display(),
                ),
            });
        if let Err(e) = spawn {
            eprintln!("flodl ddp: failed to spawn checkpoint meta writer thread: {e}");
        }
        // NCCL consensus MODEL write: the consensus is on-device across ranks
        // (no controller-side frame to tap), so dispatch the elected rank to
        // write its post-collective `self.model`. We are at the tail of
        // `finish_averaging_nccl`, AFTER the in-place weighted AllReduce, so
        // the rank holds the pure consensus — params work-weighted, f32
        // buffers mover-averaged (matching the CPU forge's frame semantics);
        // mpsc/wire FIFO orders this frame after
        // the `SyncNow` it already processed. CPU is a no-op here — its model
        // was already written by the controller forge tap.
        if matches!(self.backend, AverageBackend::Nccl) {
            let target = self.checkpoint_role;
            if target < self.world_size && !self.is_dead(target) {
                let msg = ControlMsgWire::SaveConsensusModel {
                    target_rank: target as u64,
                };
                if let Err(e) = self.send_control(target, &msg) {
                    eprintln!(
                        "flodl ddp: SaveConsensusModel dispatch to rank {target} \
                         failed: {e}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::in_flight_epoch;

    /// Rig runs reproduce the trailing window only by timing accident (three
    /// verification runs across two models never produced one), so the clamp
    /// is pinned on the arithmetic instead of on a race.
    #[test]
    fn in_flight_epoch_never_names_an_epoch_the_run_does_not_run() {
        // Nothing aggregated yet: epoch 0 is in flight.
        assert_eq!(in_flight_epoch(None, 40), 0);
        // Mid-run: the next epoch is genuinely in flight.
        assert_eq!(in_flight_epoch(Some(0), 40), 1);
        assert_eq!(in_flight_epoch(Some(38), 40), 39);
        // The regression: with the final epoch aggregated nothing is in
        // flight, and a reduce still settling belongs to the last real epoch.
        // Unclamped this returned 40 on a 40-epoch run (0-based 0..=39),
        // which the dashboard rendered as "epoch 41".
        assert_eq!(in_flight_epoch(Some(39), 40), 39);
        // Single-epoch run: the only valid epoch is 0.
        assert_eq!(in_flight_epoch(Some(0), 1), 0);
        // Degenerate epoch count must not underflow.
        assert_eq!(in_flight_epoch(Some(0), 0), 0);
        assert_eq!(in_flight_epoch(None, 0), 0);
    }
}
