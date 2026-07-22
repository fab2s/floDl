//! Unit tests for the coordinator-confirmed-exit gate in
//! `try_advance_or_shutdown_after_aggregate`: `Shutdown` must never be
//! broadcast while an NCCL sync is still in flight (some alive rank's
//! post-collective SyncAck missing). A rank that takes `Shutdown` early
//! exits its process and destroys its NCCL comm while peers' kernels of
//! the same collective are still running — stranding them in
//! `synchronize()` forever (the end-of-run cadence wedge). The gate
//! holds the broadcast until every alive rank has acked, and releases
//! on the first settled tick.
//!
//! The observable is the `run_phase` latch: `ShutdownInitiated` is set
//! immediately before the `Shutdown` broadcast and nowhere else on this
//! path (failed Shutdown sends are deliberately exempt from the
//! `LostBroadcast` trace, so the timeline carries no signal here).

use std::time::Instant;

use super::super::{ClusterCoordinator, RunPhase};
use super::cfg_sync_nccl;

#[test]
fn shutdown_waits_for_inflight_sync_to_settle() {
    let mut coord = ClusterCoordinator::for_test(cfg_sync_nccl(2).num_epochs(1));
    // Final epoch aggregated; a sync is in flight with rank 1's ack
    // still missing (its collective kernel has not retired yet).
    coord.last_aggregated_epoch = Some(0);
    coord.cycle.started_at = Some(Instant::now());
    coord.cycle.acked = vec![true, false];

    coord.try_advance_or_shutdown_after_aggregate();

    assert_eq!(
        coord.run_phase,
        RunPhase::Training,
        "an unsettled sync must hold the shutdown transition",
    );

    // Rank 1's ack lands (collective fully retired everywhere): the
    // next tick releases the broadcast.
    coord.cycle.acked = vec![true, true];
    coord.try_advance_or_shutdown_after_aggregate();

    assert_eq!(
        coord.run_phase,
        RunPhase::ShutdownInitiated,
        "the first settled tick must initiate shutdown",
    );
}

#[test]
fn shutdown_immediate_when_no_sync_in_flight() {
    // Cold acks (all-false, never synced) must NOT block shutdown
    // when no sync is in flight — the gate keys on the cycle's `started_at`,
    // not the raw ack array.
    let mut coord = ClusterCoordinator::for_test(cfg_sync_nccl(2).num_epochs(1));
    coord.last_aggregated_epoch = Some(0);
    coord.cycle.started_at = None;
    coord.cycle.acked = vec![false, false];

    coord.try_advance_or_shutdown_after_aggregate();

    assert_eq!(
        coord.run_phase,
        RunPhase::ShutdownInitiated,
        "no in-flight sync: shutdown must not be delayed",
    );
}

#[test]
fn shutdown_gate_dead_rank_counts_as_settled() {
    // A dead rank's ack never arrives; the alive cohort's collective was
    // already released by the abort/rebuild path, so waiting for the
    // dead rank would hang the shutdown forever.
    let world_size = 2;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    dead_ranks.declare_dead(1);
    let mut coord = ClusterCoordinator::for_test(
        cfg_sync_nccl(world_size)
            .num_epochs(1)
            .dead_ranks(dead_ranks),
    );
    coord.last_aggregated_epoch = Some(0);
    coord.cycle.started_at = Some(Instant::now());
    coord.cycle.acked = vec![true, false];

    coord.try_advance_or_shutdown_after_aggregate();

    assert_eq!(
        coord.run_phase,
        RunPhase::ShutdownInitiated,
        "alive cohort fully acked: the dead rank must not hold shutdown",
    );
}
