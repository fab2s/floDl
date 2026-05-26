//! `should_average` firing-gate tests for the cluster coordinator.
//!
//! Headless tests using [`ClusterCoordinator::for_test`] — no TCP, no
//! reader threads, no rank fixtures. Drive the gate directly by
//! mutating `wall_ms_accum` / `steps_since_avg` / the embedded `ElChe`
//! to assert the structural invariant: count-based gating fires on
//! completed `batch_counts` regardless of trust-window contents.

use super::*;

#[test]
fn cadence_gate_fires_on_batch_counts_even_with_pathological_slow_ms() {
    // Structural invariant: even when `smoothed_slow_ms` in the trust
    // window is pathologically inflated (simulating cold-start
    // warmup, thermal throttle, or any other measurement spike), the
    // gate must still fire once each rank has completed its
    // `batch_counts[r]`. The wall-time gate this replaces had a
    // self-reinforcing deadlock loop: the target derived from the
    // same samples that only land when the gate fires, so any upward
    // spike could lock the gate above achievable wall time. The
    // count-based gate breaks the loop entirely.
    let world_size = 2;
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 2),
        )
        .no_divergence_guard(),
    );

    // Seed the ElChe trust window with a 10_000 ms/batch sample —
    // anchor_wall_ms() would compute target = 20_000 ms, which
    // `wall_ms_accum` could not realistically reach in any test
    // window. If wall-gating were still in effect, the gate would
    // never fire.
    coord
        .el_che_mut_for_test()
        .report_timing(&[10_000.0, 10_000.0], &[1, 1], 0.0);
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    assert!(
        counts.iter().all(|&c| c > 0),
        "batch_counts populated after first calibration",
    );

    // Set per-rank state so the gate's preamble passes
    // (`nccl_ack` true is the default in `for_test`; `is_dead` false
    // for every rank in a fresh ledger; `steps_since_avg[r] > 0` for
    // each rank means the early-return guard clears).
    for (r, &c) in counts.iter().enumerate() {
        coord.set_steps_since_avg_for_test(r, c);
        // wall_ms_accum stays trivially small. With the OLD wall-time
        // gate this is what locked the deadlock — target ~20_000 ms
        // > 1 ms accumulated.
        coord.set_wall_ms_accum_for_test(r, 1.0);
    }

    assert!(
        coord.should_average(),
        "count-based gate must fire on completed batch_counts \
         regardless of `smoothed_slow_ms` in the trust window — \
         this is the structural invariant that prevents wall-gate \
         deadlock",
    );
}

#[test]
fn cadence_gate_does_not_fire_below_batch_counts() {
    // Symmetric assertion: with the trust window calibrated to
    // realistic ms but per-rank steps below `batch_counts`, the
    // gate does NOT fire. Pins the "scheduled steps are the
    // phenomenological invariant" framing — timing alone cannot
    // make the gate fire if a rank hasn't done its share.
    let world_size = 2;
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 4),
        )
        .no_divergence_guard(),
    );

    coord
        .el_che_mut_for_test()
        .report_timing(&[10.0, 20.0], &[2, 2], 0.0);
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    assert!(counts[0] >= 2 && counts[1] >= 2);

    // Each rank one step short of its count.
    for (r, &c) in counts.iter().enumerate() {
        coord.set_steps_since_avg_for_test(r, c.saturating_sub(1));
        // Pile up wall_ms_accum — would have force-fired the OLD
        // wall-time gate (target = anchor * smoothed_slow_ms is
        // bounded; pumping wall to a huge value would always satisfy
        // min_wall >= target). Count-based gate ignores this.
        coord.set_wall_ms_accum_for_test(r, 1e9);
    }

    assert!(
        !coord.should_average(),
        "gate must NOT fire when ranks have not completed batch_counts, \
         regardless of accumulated wall time",
    );
}
