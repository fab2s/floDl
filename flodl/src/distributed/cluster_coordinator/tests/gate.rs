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
    // (the cycle's acks rest true in a fresh coord; `is_dead` false
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

#[test]
fn cpu_rearm_is_independent_of_nccl_ack() {
    // Regression guard for the CPU averaging stall: the CPU backend
    // re-arms via its Idle phase, NOT the ack slots (then named
    // `nccl_ack`). The cluster rewrite had forced CPU re-arm onto the
    // acks and the bridge faked a `usize::MAX / 2` step_count to
    // satisfy them, which poisoned `last_step_count` and wedged the
    // gate after a few cycles. Here we pin the acks all-false (the
    // wedged state) and assert the CPU gate STILL fires once each rank
    // completed its `batch_counts`.
    let world_size = 2;
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 2),
        )
        .no_divergence_guard(),
    );

    coord
        .el_che_mut_for_test()
        .report_timing(&[10.0, 20.0], &[2, 2], 0.0);
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    for (r, &c) in counts.iter().enumerate() {
        coord.set_steps_since_avg_for_test(r, c);
    }
    // The poisoned NCCL re-arm state: no rank's ack is set.
    coord.set_all_nccl_ack_for_test(false);

    assert!(
        coord.should_average(),
        "CPU re-arm must NOT depend on the ack slots — the cycle is Idle \
         and every rank completed its batch_counts, so the gate must \
         fire even with the acks all-false",
    );
}

#[test]
fn cpu_gate_blocks_while_cycle_in_flight() {
    // The CPU-phase re-arm gate: while a CPU averaging cycle is
    // Pending (snapshots / TCP all-reduce in flight), `should_average`
    // must not re-trigger, even though every rank has met its quota and
    // the acks are all-true. `poll_cpu_averaging` returns the phase to
    // `Idle` on finalize, which is what re-opens the gate.
    let world_size = 2;
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 2),
        )
        .no_divergence_guard(),
    );

    coord
        .el_che_mut_for_test()
        .report_timing(&[10.0, 20.0], &[2, 2], 0.0);
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    for (r, &c) in counts.iter().enumerate() {
        coord.set_steps_since_avg_for_test(r, c);
    }
    coord.set_cpu_avg_pending_for_test();

    assert!(
        !coord.should_average(),
        "CPU gate must not re-fire while a cycle is Pending in flight",
    );
}

#[test]
fn final_consensus_reduce_needed_only_with_trailing_steps() {
    // End-of-training coherence decision: a final reduce is forced before
    // shutdown iff some alive rank carries un-reduced trailing steps from
    // the edge schedule (and >= 2 ranks are alive). When every rank is at
    // 0 since the last reduce, the cohort is already coherent -> no reduce.
    let world_size = 2;
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 2),
        )
        .no_divergence_guard(),
    );

    // All ranks reduced clean (no trailing) -> already coherent.
    coord.set_steps_since_avg_for_test(0, 0);
    coord.set_steps_since_avg_for_test(1, 0);
    assert!(
        !coord.needs_final_consensus_reduce_for_test(),
        "no trailing steps -> no final reduce",
    );

    // One rank carries a trailing tail chunk that never filled a window.
    coord.set_steps_since_avg_for_test(1, 3);
    assert!(
        coord.needs_final_consensus_reduce_for_test(),
        "a rank with trailing steps -> final reduce before shutdown",
    );
}

#[test]
fn quiesced_zero_step_tail_rank_does_not_block_reduce() {
    // Step 3's edge schedule can hand a rank 0 steps in an epoch's final
    // window. Once that rank is quiesced (no in-flight chunk + its epoch
    // pool drained), it must NOT block the reduce gate -- otherwise the
    // movers held at the reduce barrier (steps_since_avg never reset) wedge.
    let world_size = 3;
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 4),
        )
        .no_divergence_guard(),
    );
    coord
        .el_che_mut_for_test()
        .report_timing(&[200.0, 1000.0, 1000.0], &[4, 4, 4], 10.0);
    let counts = coord.el_che_for_test().batch_counts().to_vec();

    // Tail state: epoch-0 pool drained (remaining 0), every rank at epoch 0.
    // Movers hit their counts; rank 2 got 0 (edge schedule).
    coord.install_chunk_pool_for_test(0, 0);
    for r in 0..world_size {
        coord.set_rank_epoch_for_test(r, 0);
    }
    coord.set_steps_since_avg_for_test(0, counts[0]);
    coord.set_steps_since_avg_for_test(1, counts[1]);
    coord.set_steps_since_avg_for_test(2, 0);
    assert!(
        coord.should_average(),
        "quiesced 0-step tail rank must not block the reduce (counts={counts:?})",
    );

    // Contrast: pool NOT drained -> rank 2's 0 means "still to be
    // dispatched", not quiesced -> the gate must hold.
    coord.install_chunk_pool_for_test(0, 1000);
    assert!(
        !coord.should_average(),
        "a 0-step rank that can still get work must block the gate",
    );
}
