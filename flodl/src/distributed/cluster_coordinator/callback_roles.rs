//! Callback-role election and handling for
//! [`super::ClusterCoordinator`].
//!
//! Owns the sticky "fastest rank" selection that drives `epoch_fn` /
//! `eval_fn` / `checkpoint_fn` callbacks, the per-callback failover
//! logic, the per-cycle observation feed to the LR-aware
//! meta-controller, and the `ShutdownWithSave` broadcast that fires
//! on unrecoverable cluster failures.

use crate::distributed::wire::ControlMsgWire;
use crate::tensor::Result;

use super::ClusterCoordinator;

impl ClusterCoordinator {
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
    /// the cycle's NET anchor change is captured by the post-cycle
    /// `AnchorChanged` event emitted from `finish_averaging_*` (covers
    /// meta nudge composed with any guard-driven adjustment).
    /// `MetaNudge` here isolates the meta's contribution with the raw
    /// `factor` so MSF / dashboard tooling can attribute the cycle's
    /// anchor delta between the two sources.
    pub(super) fn observe_meta(
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
            if let Some(ref tl) = self.timeline {
                tl.event(crate::monitor::EventKind::MetaNudge {
                    factor,
                    from: old,
                    to: new,
                });
            }
            crate::verbose!(
                "  ddp: meta-controller nudge factor={:.3} anchor {} -> {}",
                factor, old, new,
            );
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
    pub(super) fn resolve_fastest_role(&self) -> usize {
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
    pub(super) fn re_resolve_callback_roles_on_death(&mut self, dead_rank: usize) {
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
    pub(super) fn broadcast_epoch_callback_role_if_dirty(&mut self) -> Result<()> {
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

    pub(super) fn handle_checkpoint_result(
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
    pub(super) fn handle_eval_result(
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
    pub(super) fn handle_epoch_fn_elapsed(&mut self, rank: usize, elapsed_ms: f64) {
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
    /// [`crate::distributed::ddp::ElChe::apply_callback_slack`]).
    pub(super) fn maybe_apply_callback_slack_for_next_cycle(&mut self) {
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
    pub(super) fn dispatch_shutdown_with_save(
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
}
