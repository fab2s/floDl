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
use super::{cfg_sync_cpu, cfg_sync_nccl};

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
fn final_reduce_decision_drains_pending_timing_first() {
    // The trailing-step view feeding `needs_final_consensus_reduce` is
    // built by `drain_timing`, which runs at the TOP of `tick()`; the
    // epoch aggregate that reaches this decision drains at the BOTTOM.
    // A rank's last step report and its epoch end arriving between the
    // two (inside the tick body, on a loaded box) would leave the
    // report enqueued but undrained at the decision: steps read 0, the
    // forced end-of-run consensus reduce is skipped silently, and the
    // run shuts down un-reduced with no final bundle (the CI-only
    // natural-end flake). The pre-decision drain closes that window;
    // this test enqueues WITHOUT draining and asserts the decision
    // still sees the trailing step.
    let (mut coord, timing_tx) =
        ClusterCoordinator::for_test_with_timing_tx(cfg_sync_cpu(2).num_epochs(1));
    coord.last_aggregated_epoch = Some(0);
    timing_tx
        .send(crate::distributed::wire::TimingMsgWire::Batch {
            rank: 0,
            batch_ms: 1.0,
            data_ms: 0.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })
        .expect("timing enqueue");

    coord.try_advance_or_shutdown_after_aggregate();

    assert!(
        coord.window.steps(0) > 0,
        "the decision must drain pending timing before reading trailing steps",
    );
    assert_eq!(
        coord.run_phase,
        RunPhase::Training,
        "an undrained trailing step must trigger the forced final reduce, \
         not the shutdown broadcast",
    );
}

#[test]
fn post_aggregate_forced_reduce_arms_the_final_bundle() {
    // The full CI-flake shape, both layers composed: the trailing-step
    // report and the epoch end land inside one tick body, so the last
    // reduce is the POST-aggregate forced one — by which point
    // `aggregate_ready_epochs` has removed the epoch's pool. The
    // pre-decision drain must surface the trailing step (layer one) and
    // the tail check must treat the aggregated epoch as the tail despite
    // its missing pool (layer two), so the forced reduce ARMS the forge —
    // arms == 0 here was exactly the CI forensics signature.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
    use crate::distributed::{CheckpointForge, ModelSchema};

    let world_size = 2;
    let model = crate::nn::Linear::on_device(4, 2, crate::tensor::Device::CPU).unwrap();
    let forge = CheckpointForge::new(Some(ModelSchema::from_module(&model)), None);
    let mut cfg = super::super::ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 1),
    )
    .no_divergence_guard()
    .total_samples(32)
    .batch_size(4)
    .num_epochs(1)
    .save_path("/nonexistent/never-written");
    cfg.checkpoint_forge = Some(std::sync::Arc::clone(&forge));
    let (mut coord, timing_tx) = ClusterCoordinator::for_test_with_timing_tx(cfg);

    // Post-aggregate, pool removed, trailing-step report still undrained.
    coord.last_aggregated_epoch = Some(0);
    assert!(coord.chunk_pools.is_empty());
    timing_tx
        .send(crate::distributed::wire::TimingMsgWire::Batch {
            rank: 0,
            batch_ms: 1.0,
            data_ms: 0.0,
            step_count: 8,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })
        .expect("timing enqueue");

    coord.try_advance_or_shutdown_after_aggregate();

    assert_eq!(
        coord.run_phase,
        RunPhase::Training,
        "the forced final reduce must fire, not the shutdown broadcast",
    );
    let (arms, _) = forge.forensics();
    assert_eq!(
        arms, 1,
        "the forced post-aggregate reduce must arm the final consensus bundle",
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
