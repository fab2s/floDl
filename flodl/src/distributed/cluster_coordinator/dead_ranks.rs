//! Dead-rank detection, partition redistribution, and liveness queries
//! for [`super::ClusterCoordinator`].

use std::time::{Duration, Instant};

use crate::distributed::ddp_run::AverageBackend;
use crate::distributed::wire::ControlMsgWire;
use crate::tensor::{Result, TensorError};

use super::ClusterCoordinator;

impl ClusterCoordinator {
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
    pub(super) fn check_dead_ranks(&mut self) {
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
                        // Checkpoint-role failover: if Rank(n) policy
                        // and the dead rank happens to be the
                        // checkpoint role, fall over to lowest live as
                        // a best-effort. Eval/epoch roles stay pinned.
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
    pub(super) fn compute_dead_rank_remainder(&self, r: usize) -> Option<(u64, u64)> {
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
    pub(super) fn redistribute_dead_rank_partition(
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
    pub(super) fn is_dead(&self, rank: usize) -> bool {
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
    pub(super) fn unrecoverable_reason(&self) -> Option<crate::distributed::SaveReason> {
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

    /// True when `rank` is in the shared dead-rank ledger. `false` when
    /// elastic membership isn't configured (`dead_ranks` is `None`).
    pub(super) fn is_rank_dead(&self, rank: usize) -> bool {
        self.dead_ranks
            .as_ref()
            .map(|d| d.is_dead(rank))
            .unwrap_or(false)
    }
}
