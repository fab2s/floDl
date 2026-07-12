//! Unit tests for the delivered-cost timing feed: the all-or-none
//! coherence predicate and the feed-scale selection. These guard the
//! failure modes observed on the rig (mixed compute/delivered scales
//! inverting the allocation; partial delivered sets poisoning ElChe)
//! without needing a live cluster.

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};

use super::super::{ClusterCoordinator, ClusterCoordinatorConfig};

fn cadence_cpu_coord(world_size: usize) -> ClusterCoordinator {
    ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 4),
        )
        .no_divergence_guard()
        .progressive(true),
    )
}

/// Seed the per-rank delivered (ms, batches) credit pair the feed reads.
/// The feed + all-or-none predicate consume the ledger's marginal
/// delivered accumulator, so seed that.
fn set_delivered(coord: &mut ClusterCoordinator, ms: &[f64], batches: &[usize]) {
    for (i, (&m, &b)) in ms.iter().zip(batches).enumerate() {
        coord.window.set_delivered_for_test(i, m, b);
    }
}

/// Seed per-rank step counts on the window ledger.
fn set_steps(coord: &mut ClusterCoordinator, steps: &[usize]) {
    for (i, &n) in steps.iter().enumerate() {
        coord.window.set_steps_for_test(i, n);
    }
}

/// Seed per-rank compute wall on the window ledger.
fn set_wall(coord: &mut ClusterCoordinator, ms: &[f64]) {
    for (i, &m) in ms.iter().enumerate() {
        coord.window.set_wall_ms_for_test(i, m);
    }
}

#[test]
fn movers_delivered_complete_requires_every_mover() {
    let mut coord = cadence_cpu_coord(2);
    // Both ranks moved this window.
    set_steps(&mut coord, &[4, 4]);
    // Only rank 0 has a delivered sample.
    set_delivered(&mut coord, &[80.0, 0.0], &[4, 0]);
    assert!(
        !coord.movers_delivered_complete(),
        "a mover without a delivered sample must block the predicate",
    );
    // Rank 1's delivered sample lands: predicate completes.
    coord.window.set_delivered_for_test(1, 200.0, 4);
    assert!(coord.movers_delivered_complete());
}

#[test]
fn movers_delivered_complete_ignores_idle_and_dead_ranks() {
    let mut coord = cadence_cpu_coord(2);
    // Rank 1 did nothing this window (quiesced tail): not a mover, must
    // not block the predicate.
    set_steps(&mut coord, &[4, 0]);
    set_delivered(&mut coord, &[80.0, 0.0], &[4, 0]);
    assert!(
        coord.movers_delivered_complete(),
        "an idle rank must not block the predicate",
    );
}

#[test]
fn timing_feed_all_or_none_falls_back_to_compute_scale() {
    // THE MIXED-SCALE INVERSION GUARD. Delivered ms (compute + data +
    // transport) and compute-only wall ms are not comparable; feeding a
    // mix makes a starved rank look fast and inverts the allocation
    // (rig: the x1-link Pascal drew ~73% of all steps and diverged).
    // When ANY mover lacks a delivered sample, EVERY rank must fall
    // back to the compute-scale feed for this window.
    let mut coord = cadence_cpu_coord(2);
    set_steps(&mut coord, &[4, 4]);
    set_wall(&mut coord, &[40.0, 100.0]);
    // Rank 0 has a delivered sample (its delivered cost is 2x its
    // compute); rank 1 has none.
    set_delivered(&mut coord, &[80.0, 0.0], &[4, 0]);

    let (ms, batches) = coord.timing_feed();
    assert_eq!(
        ms,
        vec![40.0, 100.0],
        "incomplete window must feed the coherent compute scale for ALL ranks",
    );
    assert_eq!(batches, vec![4, 4]);
}

#[test]
fn timing_feed_uses_delivered_when_window_complete() {
    let mut coord = cadence_cpu_coord(2);
    set_steps(&mut coord, &[4, 4]);
    set_wall(&mut coord, &[40.0, 100.0]);
    // Both movers have delivered samples: the delivered scale is coherent
    // and must win (it carries the data/transport cost the balancer needs).
    set_delivered(&mut coord, &[80.0, 220.0], &[4, 4]);

    let (ms, batches) = coord.timing_feed();
    assert_eq!(ms, vec![80.0, 220.0], "complete window feeds delivered cost");
    assert_eq!(batches, vec![4, 4]);
}

#[test]
fn timing_feed_sync_keeps_compute_scale() {
    // Sync is non-progressive: no delivered samples exist; the compute
    // feed is the contract even when stray delivered values are present.
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Cpu,
            2,
            ElChe::new(2, 1),
        )
        .no_divergence_guard(),
    );
    set_steps(&mut coord, &[1, 1]);
    set_wall(&mut coord, &[10.0, 20.0]);
    set_delivered(&mut coord, &[99.0, 99.0], &[1, 1]);

    let (ms, _) = coord.timing_feed();
    assert_eq!(ms, vec![10.0, 20.0], "Sync stays on the compute feed");
}
