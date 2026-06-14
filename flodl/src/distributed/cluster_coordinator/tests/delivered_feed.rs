//! Unit tests for the delivered-cost timing feed: the all-or-none
//! coherence predicate and the feed-scale selection. These guard the
//! failure modes observed on the rig (mixed compute/delivered scales
//! inverting the allocation; partial spans poisoning ElChe) without
//! needing a live cluster.

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

/// Seed the per-rank delivered (ms, batches) credit pair from parallel
/// lists — the readable shape these tests were written in before the five
/// parallel `Vec`s collapsed into `Vec<DeliveredSpan>`.
fn set_delivered(coord: &mut ClusterCoordinator, ms: &[f64], batches: &[usize]) {
    for (i, (&m, &b)) in ms.iter().zip(batches).enumerate() {
        coord.delivered[i].ms_accum = m;
        coord.delivered[i].batches_accum = b;
    }
}

#[test]
fn movers_delivered_complete_requires_every_mover() {
    let mut coord = cadence_cpu_coord(2);
    // Both ranks moved this window.
    coord.steps_since_avg = vec![4, 4];
    // Only rank 0 has a closed span.
    set_delivered(&mut coord, &[80.0, 0.0], &[4, 0]);
    assert!(
        !coord.movers_delivered_complete(),
        "a mover without a delivered sample must block the predicate",
    );
    // Rank 1's span closes: predicate completes.
    coord.delivered[1].ms_accum = 200.0;
    coord.delivered[1].batches_accum = 4;
    assert!(coord.movers_delivered_complete());
}

#[test]
fn movers_delivered_complete_ignores_idle_and_dead_ranks() {
    let mut coord = cadence_cpu_coord(2);
    // Rank 1 did nothing this window (quiesced tail): not a mover, must
    // not block the predicate.
    coord.steps_since_avg = vec![4, 0];
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
    coord.steps_since_avg = vec![4, 4];
    coord.wall_ms_accum = vec![40.0, 100.0];
    // Rank 0 closed a delivered span (its delivered cost is 2x its
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
    coord.steps_since_avg = vec![4, 4];
    coord.wall_ms_accum = vec![40.0, 100.0];
    // Both movers closed spans: the delivered scale is coherent and
    // must win (it carries the data/transport cost the balancer needs).
    set_delivered(&mut coord, &[80.0, 220.0], &[4, 4]);

    let (ms, batches) = coord.timing_feed();
    assert_eq!(ms, vec![80.0, 220.0], "complete window feeds delivered cost");
    assert_eq!(batches, vec![4, 4]);
}

#[test]
fn timing_feed_sync_keeps_compute_scale() {
    // Sync is non-progressive: no spans exist; the compute feed is the
    // contract even when stray delivered values are present.
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Cpu,
            2,
            ElChe::new(2, 1),
        )
        .no_divergence_guard(),
    );
    coord.steps_since_avg = vec![1, 1];
    coord.wall_ms_accum = vec![10.0, 20.0];
    set_delivered(&mut coord, &[99.0, 99.0], &[1, 1]);

    let (ms, _) = coord.timing_feed();
    assert_eq!(ms, vec![10.0, 20.0], "Sync stays on the compute feed");
}
