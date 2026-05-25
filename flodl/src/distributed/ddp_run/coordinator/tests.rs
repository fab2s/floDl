    use super::*;
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // Helper: create a Coordinator without CUDA or real GPU threads.
    // -----------------------------------------------------------------------

    fn make_test_coordinator(
        world_size: usize,
        policy: ApplyPolicy,
        total_samples: usize,
    ) -> Coordinator {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        for _ in 0..world_size {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, _crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(world_size, 10);
        Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            policy, AverageBackend::Nccl, world_size, total_samples, el_che,
        )
        .num_epochs(10)
        .build()
    }

    /// Variant that returns the timing sender so tests can inject TimingMsg.
    fn make_test_coordinator_with_channels(
        world_size: usize,
        policy: ApplyPolicy,
        total_samples: usize,
    ) -> (
        Coordinator,
        mpsc::Sender<TimingMsg>,
        Vec<mpsc::Receiver<ControlMsg>>,
    ) {
        let (timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        let mut control_rxs = Vec::new();
        for _ in 0..world_size {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
            control_rxs.push(crx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(world_size, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            policy, AverageBackend::Nccl, world_size, total_samples, el_che,
        )
        .num_epochs(10)
        .build();
        (coord, timing_tx, control_rxs)
    }

    // -----------------------------------------------------------------------
    // compute_partition_sizes
    // -----------------------------------------------------------------------

    #[test]
    fn partition_sync_equal_sizes() {
        let coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        let sizes = coord.compute_partition_sizes();
        assert_eq!(sizes, vec![500, 500]);
    }

    #[test]
    fn partition_sync_uneven_remainder() {
        let coord = make_test_coordinator(3, ApplyPolicy::Sync, 100);
        let sizes = coord.compute_partition_sizes();
        // 100 / 3 = 33 remainder 1. First rank gets the extra.
        assert_eq!(sizes, vec![34, 33, 33]);
        assert_eq!(sizes.iter().sum::<usize>(), 100);
    }

    #[test]
    fn partition_cadence_uncalibrated_falls_back_to_equal() {
        // Before calibration, Cadence mode uses equal sizes.
        let coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        assert!(!coord.el_che.is_calibrated());
        assert!(!coord.el_che.has_speed_hint());
        let sizes = coord.compute_partition_sizes();
        assert_eq!(sizes, vec![500, 500]);
    }

    #[test]
    fn partition_cadence_calibrated_proportional() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Simulate calibration: rank 0 is 2x faster (5ms/batch vs 10ms/batch).
        // report_timing needs wall_ms and actual_batches.
        coord.el_che.report_timing(&[50.0, 100.0], &[10, 10], 5.0);
        assert!(coord.el_che.is_calibrated());
        let sizes = coord.compute_partition_sizes();
        // Rank 0 has throughput 1/5=0.2, rank 1 has throughput 1/10=0.1.
        // Proportions: 0.2/0.3 = 0.667, 0.1/0.3 = 0.333.
        // ~667 and ~333.
        assert!(sizes[0] > sizes[1], "fast rank should get more: {:?}", sizes);
        assert_eq!(sizes.iter().sum::<usize>(), 1000);
    }

    #[test]
    fn partition_with_user_ratios() {
        let (_timing_tx, timing_rx) = mpsc::channel();
        let (_metrics_tx, metrics_rx) = mpsc::channel();
        let (_param_tx, param_rx) = mpsc::channel();
        let mut final_rxs = Vec::new();
        let mut control_txs = Vec::new();
        for _ in 0..3 {
            let (_ftx, frx) = mpsc::channel::<ParamSnapshot>();
            final_rxs.push(frx);
            let (ctx, _crx) = mpsc::channel::<ControlMsg>();
            control_txs.push(ctx);
        }
        let el_che = crate::distributed::ddp::ElChe::new(3, 10);
        let coord = Coordinator::builder(
            timing_rx, metrics_rx, param_rx, final_rxs, control_txs,
            ApplyPolicy::Cadence, AverageBackend::Nccl, 3, 900, el_che,
        )
        .partition_ratios(Some(vec![3.0, 2.0, 1.0]))
        .num_epochs(5)
        .build();

        let sizes = coord.compute_partition_sizes();
        // Ratios 3:2:1 normalized -> 0.5, 0.333, 0.167. On 900 samples:
        // floor: 450, 300, 150 = 900. No remainder.
        assert_eq!(sizes, vec![450, 300, 150]);
        assert_eq!(sizes.iter().sum::<usize>(), 900);
    }

    // -----------------------------------------------------------------------
    // Free function edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn equal_sizes_single_worker() {
        let sizes = equal_sizes(1, 100);
        assert_eq!(sizes, vec![100]);
    }

    #[test]
    fn ratio_to_sizes_zero_sum() {
        // All zero ratios: falls back to equal.
        let sizes = ratio_to_sizes(&[0.0, 0.0], 100);
        assert_eq!(sizes, vec![50, 50]);
    }

    // -----------------------------------------------------------------------
    // should_average
    // -----------------------------------------------------------------------

    #[test]
    fn should_average_sync_triggers_when_all_have_one_step() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        // Initially no steps: should not average.
        assert!(!coord.should_average());

        // Only rank 0 has a step.
        coord.steps_since_avg[0] = 1;
        assert!(!coord.should_average());

        // Both ranks have a step: should trigger.
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_cadence_wall_time_trigger() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Calibrate ElChe so anchor_wall_ms() returns > 0.
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        assert!(coord.el_che.is_calibrated());
        let target = coord.el_che.anchor_wall_ms();
        assert!(target > 0.0, "anchor_wall_ms should be positive after calibration");

        // Satisfy preconditions: steps > 0 and nccl_ack = true.
        coord.steps_since_avg = vec![1, 1];
        coord.nccl_ack = vec![true, true];

        // Not enough wall time accumulated.
        coord.wall_ms_accum[0] = target * 0.5;
        coord.wall_ms_accum[1] = target * 0.5;
        assert!(!coord.should_average());

        // Both ranks reach the target.
        coord.wall_ms_accum[0] = target;
        coord.wall_ms_accum[1] = target;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_cadence_min_wall_governs() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        let target = coord.el_che.anchor_wall_ms();

        // Rank 0 has enough, rank 1 does not: min < target.
        coord.wall_ms_accum[0] = target * 2.0;
        coord.wall_ms_accum[1] = target * 0.3;
        assert!(!coord.should_average());
    }

    #[test]
    fn should_average_async_batch_count_trigger() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        // Uncalibrated ElChe: batch_counts are [10, 10] (equal, anchor=10).
        let counts = coord.el_che.batch_counts().to_vec();
        assert_eq!(counts, vec![10, 10]);

        // Rank 0 at 10, rank 1 at 9: not enough.
        coord.steps_since_avg[0] = 10;
        coord.steps_since_avg[1] = 9;
        assert!(!coord.should_average());

        // Both at target.
        coord.steps_since_avg[1] = 10;
        assert!(coord.should_average());
    }

    #[test]
    fn should_average_blocked_when_worker_exited() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // One worker exits.
        coord.active_count = 1;
        assert!(!coord.should_average(), "must not average with fewer active workers");
    }

    #[test]
    fn should_average_blocked_when_all_epochs_done() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // Mark all epochs aggregated (num_epochs=10, so epoch 9 is the last).
        coord.last_aggregated_epoch = Some(9);
        assert!(coord.all_epochs_done());
        assert!(!coord.should_average());
    }

    #[test]
    fn should_average_blocked_during_cpu_averaging() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 1;
        coord.steps_since_avg[1] = 1;
        assert!(coord.should_average());

        // Simulate CPU averaging in progress.
        coord.avg_state = CpuAvgState::Collecting {
            snapshots: Vec::new(),
            received: vec![false; 2],
            deadline: Instant::now() + std::time::Duration::from_secs(5),
            start: Instant::now(),
            steps_snapshot: vec![0; 2],
            wall_ms_snapshot: vec![0.0; 2],
        };
        assert!(!coord.should_average());
    }

    // -----------------------------------------------------------------------
    // process_timing_msg
    // -----------------------------------------------------------------------

    #[test]
    fn process_timing_msg_batch_accumulates() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        assert_eq!(coord.steps_since_avg[0], 0);
        assert_eq!(coord.wall_ms_accum[0], 0.0);

        coord.process_timing_msg(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None });
        assert_eq!(coord.steps_since_avg[0], 1);
        assert!((coord.wall_ms_accum[0] - 10.0).abs() < 1e-9);
        assert!((coord.last_batch_ms[0] - 10.0).abs() < 1e-9);

        // Second message accumulates.
        coord.process_timing_msg(TimingMsg::Batch { rank: 0, batch_ms: 15.0, step_count: 2, param_norm: None, batch_loss: 0.1, sync_divergence: None });
        assert_eq!(coord.steps_since_avg[0], 2);
        assert!((coord.wall_ms_accum[0] - 25.0).abs() < 1e-9);
        assert!((coord.last_batch_ms[0] - 15.0).abs() < 1e-9);

        // Rank 1 is independent.
        assert_eq!(coord.steps_since_avg[1], 0);
        assert_eq!(coord.wall_ms_accum[1], 0.0);
    }

    #[test]
    fn process_timing_msg_exiting_decrements_active() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        assert_eq!(coord.active_count, 2);

        coord.process_timing_msg(TimingMsg::Exiting { rank: 0 });
        assert_eq!(coord.active_count, 1);

        // Saturating: double exit won't underflow.
        coord.process_timing_msg(TimingMsg::Exiting { rank: 0 });
        assert_eq!(coord.active_count, 0);
    }

    // -----------------------------------------------------------------------
    // on_rank_done / on_epoch_aggregated
    // -----------------------------------------------------------------------

    #[test]
    fn on_rank_done_async_dispatches_within_lookahead() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        // Disable progressive so on_rank_done does the per-rank dispatch.
        coord.progressive = false;

        // Rank 0 finishes epoch 0. Next is epoch 1. No aggregation yet,
        // so lookahead allows epoch 0 and 1 (next <= 1).
        coord.on_rank_done(0, 0);
        // Should have sent StartEpoch(1) to rank 0.
        let msg = control_rxs[0].try_recv();
        assert!(
            matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 1),
            "expected StartEpoch(1)"
        );
    }

    #[test]
    fn on_rank_done_async_blocks_beyond_lookahead() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;

        // Rank 0 finishes epoch 1 but no aggregation yet.
        // next = 2, within_lookahead = (2 <= 1) = false. Should NOT dispatch.
        coord.on_rank_done(0, 1);
        assert!(
            control_rxs[0].try_recv().is_err(),
            "should NOT dispatch epoch 2 beyond lookahead"
        );
        assert!(coord.rank_waiting[0], "rank should be marked waiting");
    }

    #[test]
    fn on_rank_done_sync_is_noop() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 1000);
        // Sync policy: on_rank_done does nothing (early return for non-Async).
        coord.on_rank_done(0, 0);
        assert!(control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn on_rank_done_last_epoch_is_noop() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;
        // num_epochs = 10, so finishing epoch 9 means next = 10 >= 10.
        coord.on_rank_done(0, 9);
        assert!(control_rxs[0].try_recv().is_err());
    }

    #[test]
    fn on_epoch_aggregated_sends_shutdown_after_last_epoch() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 1000);
        // Aggregate the last epoch (num_epochs=10, so epoch 9).
        coord.on_epoch_aggregated(9);
        assert!(coord.all_epochs_done());
        // Both ranks should receive Shutdown.
        for rx in &control_rxs {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::Shutdown)),
                "expected Shutdown"
            );
        }
    }

    #[test]
    fn on_epoch_aggregated_dispatches_next_sync() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Sync, 100);
        // Disable progressive for Sync.
        coord.progressive = false;

        // Aggregate epoch 0: should dispatch epoch 1 to both ranks.
        coord.on_epoch_aggregated(0);
        assert_eq!(coord.last_aggregated_epoch, Some(0));
        for (rank, rx) in control_rxs.iter().enumerate() {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 1),
                "rank {rank}: expected StartEpoch(1)"
            );
        }
    }

    #[test]
    fn on_epoch_aggregated_unblocks_waiting_ranks_async() {
        let (mut coord, _timing_tx, control_rxs) =
            make_test_coordinator_with_channels(2, ApplyPolicy::Async, 1000);
        coord.progressive = false;

        // Simulate: rank 0 finished epoch 1 and is waiting (beyond lookahead).
        coord.rank_epoch[0] = 1;
        coord.rank_waiting[0] = true;
        // Rank 1 is also stuck.
        coord.rank_epoch[1] = 1;
        coord.rank_waiting[1] = true;

        // Aggregate epoch 0: should unblock waiting ranks, dispatching epoch 2.
        coord.on_epoch_aggregated(0);
        for (rank, rx) in control_rxs.iter().enumerate() {
            let msg = rx.try_recv();
            assert!(
                matches!(msg, Ok(ControlMsg::StartEpoch(plan)) if plan.epoch == 2),
                "rank {rank}: expected StartEpoch(2)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Coordinator state accessors
    // -----------------------------------------------------------------------

    #[test]
    fn all_epochs_done_boundary() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 100);
        // num_epochs = 10. Not done until epoch 9 is aggregated.
        assert!(!coord.all_epochs_done());
        coord.last_aggregated_epoch = Some(8);
        assert!(!coord.all_epochs_done());
        coord.last_aggregated_epoch = Some(9);
        assert!(coord.all_epochs_done());
    }

    #[test]
    fn finish_averaging_nccl_resets_counters() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        coord.steps_since_avg[0] = 5;
        coord.steps_since_avg[1] = 3;
        coord.wall_ms_accum[0] = 50.0;
        coord.wall_ms_accum[1] = 30.0;
        coord.throttled[0] = true;

        coord.finish_averaging_nccl();

        assert_eq!(coord.steps_since_avg, vec![0, 0]);
        assert_eq!(coord.wall_ms_accum, vec![0.0, 0.0]);
        assert!(!coord.throttled[0]);
        assert_eq!(coord.version, 1);
        assert_eq!(coord.avg_count, 1);
    }

    #[test]
    fn finish_averaging_cpu_subtracts_snapshots() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Sync, 1000);
        // Simulate: at trigger time, steps were [5, 3], wall was [50, 30].
        // During averaging, rank 0 did 2 more batches (20ms), rank 1 did 1 (10ms).
        coord.steps_since_avg = vec![7, 4];
        coord.wall_ms_accum = vec![70.0, 40.0];

        let steps_snap = vec![5, 3];
        let wall_snap = vec![50.0, 30.0];
        coord.finish_averaging_cpu(12.5, &steps_snap, &wall_snap, None);

        // Residual = current - snapshot.
        assert_eq!(coord.steps_since_avg, vec![2, 1]);
        assert!((coord.wall_ms_accum[0] - 20.0).abs() < 1e-9);
        assert!((coord.wall_ms_accum[1] - 10.0).abs() < 1e-9);
        assert_eq!(coord.version, 1);
        assert_eq!(coord.avg_count, 1);
    }

    // -----------------------------------------------------------------------
    // NCCL divergence detection (trend-based, gentle guardrail)
    // -----------------------------------------------------------------------

    /// Helper: simulate one averaging interval with given weight-space divergence.
    fn run_interval_with_divergence(coord: &mut Coordinator, div0: f64, div1: f64) {
        coord.nccl_sync_divergence = vec![Some(div0), Some(div1)];
        coord.wall_ms_accum = vec![100.0, 200.0];
        coord.steps_since_avg = vec![10, 10];
        coord.finish_averaging_nccl();
    }

    #[test]
    fn divergence_trend_suppresses_overshoot_growth() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        let overshoot_before = coord.max_overshoot;

        // 3 intervals with rising weight-space divergence.
        // Interval 1: max_relative_delta = 0.05
        run_interval_with_divergence(&mut coord, 0.03, 0.05);
        // Interval 2: max_relative_delta = 0.13
        run_interval_with_divergence(&mut coord, 0.08, 0.13);
        // Interval 3: max_relative_delta = 0.23
        run_interval_with_divergence(&mut coord, 0.15, 0.23);

        // Rising trend detected -> 3rd interval suppresses growth.
        // Overshoot should have grown for intervals 1-2 (no trend yet)
        // but NOT for interval 3.
        let cap = (coord.total_samples / coord.batch_size.max(1) / 50).clamp(3, 10);
        let expected = (overshoot_before + 2).min(cap).min(coord.overshoot_ceiling);
        assert_eq!(coord.max_overshoot, expected,
            "3rd interval with rising trend should suppress growth");
    }

    #[test]
    fn divergence_no_trend_allows_overshoot_growth() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        let overshoot_before = coord.max_overshoot;

        // 3 intervals with flat low divergence (no trend).
        for _ in 0..3 {
            run_interval_with_divergence(&mut coord, 0.005, 0.005);
        }

        let cap = (coord.total_samples / coord.batch_size.max(1) / 50).clamp(3, 10);
        let expected = (overshoot_before + 3).min(cap).min(coord.overshoot_ceiling);
        assert_eq!(coord.max_overshoot, expected,
            "flat low divergence should allow normal overshoot growth");
    }

    #[test]
    fn divergence_single_spike_does_not_suppress() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        let overshoot_before = coord.max_overshoot;

        // Single high-divergence interval (no trend, need 3 for rising).
        run_interval_with_divergence(&mut coord, 0.5, 0.8);

        assert!(coord.max_overshoot > overshoot_before,
            "single measurement should not suppress growth");
    }

    #[test]
    fn divergence_never_shrinks_anchor() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.el_che.report_timing(&[100.0, 200.0], &[10, 10], 5.0);
        let anchor_before = coord.el_che.anchor();

        // 5 intervals with rising weight-space divergence.
        for i in 0..5 {
            let div = 0.05 + i as f64 * 0.03; // rising
            run_interval_with_divergence(&mut coord, div * 0.8, div);
        }

        // Anchor must NEVER decrease from divergence detection.
        // ElChe's overhead auto-tune may change it, but nudge_anchor_down
        // is not called from the divergence path.
        assert!(coord.el_che.anchor() >= anchor_before,
            "divergence must never shrink anchor: was {anchor_before}, now {}",
            coord.el_che.anchor());
    }

    #[test]
    fn divergence_accumulators_reset_each_interval() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        coord.loss_accum = vec![5.0, 3.0];
        coord.loss_count = vec![100, 50];
        coord.nccl_sync_divergence = vec![Some(0.05), Some(0.03)];
        coord.wall_ms_accum = vec![100.0, 200.0];
        coord.steps_since_avg = vec![100, 50];

        coord.finish_averaging_nccl();

        assert_eq!(coord.loss_accum, vec![0.0, 0.0]);
        assert_eq!(coord.loss_count, vec![0, 0]);
        // nccl_sync_divergence is reset after averaging.
        assert_eq!(coord.nccl_sync_divergence, vec![None, None]);
    }

    // `divergence_history_capped_at_5` removed: the ring-buffer cap is
    // now an implementation detail of the concrete guard. Coverage moved
    // to `convergence::tests::trend_history_capped_at_5`.

    #[test]
    fn divergence_overshoot_ceiling_caps() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Async, 1000);
        coord.overshoot_ceiling = 3;
        coord.max_overshoot = 5;

        coord.wall_ms_accum = vec![100.0, 200.0];
        coord.steps_since_avg = vec![10, 10];
        coord.finish_averaging_nccl();

        assert!(coord.max_overshoot <= 3,
            "overshoot {} should be capped at ceiling 3", coord.max_overshoot);
    }

    #[test]
    fn process_timing_msg_accumulates_loss_and_norm() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        assert_eq!(coord.loss_accum[0], 0.0);
        assert_eq!(coord.loss_count[0], 0);

        coord.process_timing_msg(TimingMsg::Batch {
            rank: 0, batch_ms: 10.0, step_count: 1, param_norm: Some(5.5), batch_loss: 0.3, sync_divergence: None,
        });
        assert!((coord.loss_accum[0] - 0.3).abs() < 1e-9);
        assert_eq!(coord.loss_count[0], 1);

        coord.process_timing_msg(TimingMsg::Batch {
            rank: 0, batch_ms: 10.0, step_count: 2, param_norm: None, batch_loss: 0.2, sync_divergence: None,
        });
        assert!((coord.loss_accum[0] - 0.5).abs() < 1e-9);
        assert_eq!(coord.loss_count[0], 2);

        // SyncNow ack (batch_loss=0.0) is skipped.
        coord.process_timing_msg(TimingMsg::Batch {
            rank: 0, batch_ms: 0.0, step_count: 3, param_norm: None, batch_loss: 0.0, sync_divergence: None,
        });
        assert_eq!(coord.loss_count[0], 2); // unchanged

        assert_eq!(coord.loss_count[1], 0); // rank 1 independent
    }

    // -----------------------------------------------------------------------
    // SyncAck handling (regression guard: do not inflate steps_since_avg)
    // -----------------------------------------------------------------------

    /// SyncAck must NOT increment steps_since_avg, otherwise every NCCL sync
    /// inflates global_step by one batch per rank and the LR scheduler fires
    /// early. Real bug found 2026-04-13: with ~27 syncs/epoch * 2 ranks the
    /// inflation was 6.9% and a MultiStepLR milestone at total/2 fired at
    /// epoch 94 instead of 100.
    #[test]
    fn sync_ack_does_not_inflate_steps_since_avg() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);

        // Simulate 10 real batches on each rank.
        for step in 1..=10 {
            for rank in 0..2 {
                coord.process_timing_msg(TimingMsg::Batch {
                    rank, batch_ms: 5.0, step_count: step,
                    param_norm: None, batch_loss: 0.1, sync_divergence: None,
                });
            }
        }
        assert_eq!(coord.steps_since_avg, vec![10, 10]);
        let wall_before: Vec<f64> = coord.wall_ms_accum.clone();

        // Inject a SyncAck for each rank (post-NCCL ack).
        for rank in 0..2 {
            coord.process_timing_msg(TimingMsg::SyncAck {
                rank, step_count: 11, divergence: Some(0.01),
                post_norm: Some(1.0), pre_norm: Some(1.01),
            });
        }

        // The killer assertion: SyncAck must leave steps_since_avg untouched.
        assert_eq!(coord.steps_since_avg, vec![10, 10],
            "SyncAck must not be counted as a real batch");
        // Wall-time accumulator also untouched (it's per-batch wall time).
        assert_eq!(coord.wall_ms_accum, wall_before);
        // But last_step_count must reflect the post-sync step.
        assert_eq!(coord.last_step_count, vec![11, 11]);
        // Divergence captured.
        assert_eq!(coord.nccl_sync_divergence[0], Some(0.01));
        assert_eq!(coord.nccl_sync_divergence[1], Some(0.01));
    }

    /// SyncAck must satisfy nccl_ack so the next averaging cycle isn't blocked.
    #[test]
    fn sync_ack_satisfies_nccl_ack() {
        let mut coord = make_test_coordinator(2, ApplyPolicy::Cadence, 1000);
        // Snapshot the step counters at sync trigger time (rank0=5, rank1=5).
        coord.nccl_sync_step = vec![5, 5];
        coord.nccl_ack = vec![false, false];

        // Worker reports SyncAck with step_count=6 (one past snapshot).
        for rank in 0..2 {
            coord.process_timing_msg(TimingMsg::SyncAck {
                rank, step_count: 6, divergence: None,
                post_norm: None, pre_norm: None,
            });
        }
        assert_eq!(coord.nccl_ack, vec![true, true],
            "SyncAck with step_count > snapshot must flip nccl_ack");
    }
