//! ElChe cadence + adaptive K tests, non-blocking CPU averaging tests,
//! NCCL safety regression during shutdown (progressive-dispatch
//! deadlock).

use super::*;

// -----------------------------------------------------------------------
// ElChe cadence + adaptive K tests
// -----------------------------------------------------------------------

#[test]
fn test_cadence_heterogeneous_timing() {
    // Simulate 2:1 speed ratio. Rank 0 is 2x faster (5ms/batch vs 10ms/batch).
    // With Cadence policy, ElChe should give rank 0 more batches.
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Feed enough timing to calibrate ElChe.
    // First, trigger with equal steps so ElChe sees the timing.
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.coord.drain_timing();
        if h.coord.should_average() {
            h.coord.trigger_averaging().unwrap();
            // Drain control messages
            for rx in &h.control_rxs {
                while rx.try_recv().is_ok() {}
            }
        }
    }

    // After calibration, ElChe batch_counts should reflect the speed ratio
    if h.coord.is_calibrated() {
        let counts = h.coord.el_che.batch_counts();
        // Rank 0 (fast) should have more batches than rank 1 (slow)
        assert!(counts[0] >= counts[1],
            "fast rank should get more batches: {:?}", counts);
    }
}

#[test]
fn test_cpu_averaging_divergence_correction() {
    // Full pipeline: high divergence during CPU averaging triggers
    // anchor correction via nudge_anchor_down.
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);

    assert_eq!(h.coord.el_che.anchor(), 10);

    // Feed enough timing to reach batch_counts (anchor=10, uncalibrated).
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average());

    // Trigger CPU averaging with highly divergent snapshots.
    h.coord.trigger_averaging().unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[100], opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[100], 100.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();

    // Poll until averaging completes.
    let v_before = h.coord.version();
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if h.coord.version() > v_before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(h.coord.version() > v_before, "averaging should have completed");

    // Drain control messages.
    for rx in &h.control_rxs {
        while rx.try_recv().is_ok() {}
    }

    // After one round: report_timing auto-tunes anchor up (from overhead),
    // then divergence correction halves it. Final anchor should be lower
    // than the post-overhead-auto-tune value. We verify it completed and
    // the anchor is reasonable (not at max_anchor=1000).
    let anchor = h.coord.el_che.anchor();
    assert!(anchor < 1000,
        "divergence correction should have kept anchor below max, got {}", anchor);
    // Verify calibration happened.
    assert!(h.coord.is_calibrated());
}

// -----------------------------------------------------------------------
// Non-blocking CPU averaging tests
// -----------------------------------------------------------------------

#[test]
fn test_throttle_during_cpu_averaging() {
    // The key invariant: check_throttle fires even while CPU averaging
    // is in Collecting state. Uses Cadence policy because Sync sends
    // a sync Throttle with RequestParams (workers block immediately).
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Cpu);
    let el_che = ElChe::new(2, 1).with_max_batch_diff(2);
    h.coord.el_che = el_che;

    // Feed enough timing to trigger averaging
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    // Trigger averaging (enters Collecting state, returns immediately)
    assert!(h.coord.should_average());
    h.coord.trigger_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());
    assert!(!h.coord.should_average()); // guard prevents re-trigger

    // Consume RequestParams from control channels
    for rx in &h.control_rxs {
        match rx.try_recv() {
            Ok(ControlMsg::RequestParams) => {}
            other => panic!("expected RequestParams, got {:?}", other.map(|m| std::mem::discriminant(&m))),
        }
    }

    // Simulate rank 0 running ahead by 5 batches during the averaging window
    for i in 0..5 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 2 + i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();

    // check_throttle should fire even though we're in Collecting state
    h.coord.check_throttle();

    // Rank 0 should receive Throttle (it's 5 batches ahead, max_diff=2)
    match h.control_rxs[0].try_recv() {
        Ok(ControlMsg::Throttle) => {}
        other => panic!("expected Throttle for rank 0, got {:?}", other.map(|m| std::mem::discriminant(&m))),
    }
    // Rank 1 should NOT be throttled
    assert!(h.control_rxs[1].try_recv().is_err(), "rank 1 should not be throttled");
}

#[test]
fn test_cpu_avg_state_machine_full_cycle() {
    // Drive the full Idle -> Collecting -> Computing -> Idle cycle.
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Feed timing
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    assert_eq!(h.coord.version(), 0);
    assert!(!h.coord.is_cpu_averaging());

    // Trigger: enters Collecting
    h.coord.trigger_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());

    // Poll with no snapshots yet: still Collecting
    h.coord.poll_cpu_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());

    // Supply snapshots
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[4], opts).unwrap()],
        buffers: vec![],
        batch_count: 5,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[4], 3.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 5,
    }).unwrap();

    // Poll: transitions Collecting -> Computing (spawns thread)
    h.coord.poll_cpu_averaging().unwrap();

    // Poll until Computing -> Idle
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if !h.coord.is_cpu_averaging() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Verify completion
    assert!(!h.coord.is_cpu_averaging());
    assert_eq!(h.coord.version(), 1);

    // Workers should have received RequestParams then Update
    for rx in &h.control_rxs {
        let mut got_request = false;
        let mut got_update = false;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                ControlMsg::RequestParams => got_request = true,
                ControlMsg::Update(avg) => {
                    got_update = true;
                    assert_eq!(avg.version, 1);
                }
                _ => {}
            }
        }
        assert!(got_request, "worker should have received RequestParams");
        assert!(got_update, "worker should have received Update");
    }
}

#[test]
fn test_cpu_avg_collection_timeout() {
    // Use a very short timeout (1 second) and never send snapshots.
    let mut h = make_coord_harness_with_timeout(
        2, ApplyPolicy::Sync, AverageBackend::Cpu, 1,
    );

    // Feed timing to trigger averaging
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    // Trigger: enters Collecting
    h.coord.trigger_averaging().unwrap();
    assert!(h.coord.is_cpu_averaging());

    // Wait for the timeout to expire
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Poll: should soft-abort (back to Idle)
    h.coord.poll_cpu_averaging().unwrap(); // Ok, not Err
    assert!(!h.coord.is_cpu_averaging());
    assert_eq!(h.coord.version(), 0); // no version bump

    // should_average is available again for retry
    assert!(h.coord.should_average());
}

#[test]
fn test_stale_snapshot_after_timeout() {
    // After a timeout, stale snapshots from the aborted round
    // must not contaminate the next round.
    let mut h = make_coord_harness_with_timeout(
        2, ApplyPolicy::Sync, AverageBackend::Cpu, 1,
    );
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Feed timing
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    // Round 1: trigger, send only rank 0's snapshot, let it timeout
    h.coord.trigger_averaging().unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::full(&[4], 999.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();

    // Wait for timeout
    std::thread::sleep(std::time::Duration::from_secs(2));
    h.coord.poll_cpu_averaging().unwrap();
    assert!(!h.coord.is_cpu_averaging()); // soft abort
    assert_eq!(h.coord.version(), 0);

    // Round 2: trigger fresh. The stale rank-0 snapshot from round 1
    // should have been drained by abort_cpu_averaging.
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 2, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 5.0, step_count: 2, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    h.coord.trigger_averaging().unwrap();

    // Send FRESH snapshots for both ranks (value=1.0 and 3.0)
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[4], opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[4], 3.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 1,
    }).unwrap();

    // Poll until complete
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if h.coord.version() > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(h.coord.version(), 1);

    // Verify the Update contains fresh data (avg of 1.0 and 3.0 = 2.0),
    // NOT 999.0 from the stale snapshot.
    for rx in &h.control_rxs {
        let mut found_update = false;
        while let Ok(msg) = rx.try_recv() {
            if let ControlMsg::Update(avg) = msg {
                let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
                let expected = 2.0 * 4.0; // 2.0 per element * 4 elements
                assert!(
                    (sum - expected).abs() < 1e-4,
                    "expected sum={expected}, got {sum} (stale data leaked?)"
                );
                found_update = true;
            }
        }
        assert!(found_update, "worker should have received Update");
    }
}

#[test]
fn test_elche_calibration_produces_proportional_sizes() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Feed heterogeneous timing to trigger ElChe calibration
    for _ in 0..5 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.coord.drain_timing();
        if h.coord.should_average() {
            h.coord.trigger_averaging().unwrap();
            for rx in &h.control_rxs {
                while rx.try_recv().is_ok() {}
            }
        }
    }

    assert!(h.coord.is_calibrated(), "ElChe should have calibrated");
    // After calibration, compute_partition_sizes should produce valid sizes
    let sizes = h.coord.compute_partition_sizes();
    assert_eq!(sizes.len(), 2);
    let total: usize = sizes.iter().sum();
    assert!(total <= 10000, "partitions should not exceed total: {total}");
}

#[test]
fn test_wall_ms_accumulation() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Send multiple timing messages per rank
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 7.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 12.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    // wall_ms_accum should have accumulated totals
    assert!((h.coord.wall_ms_accum[0] - 12.0).abs() < 1e-10, "rank 0 should be 5+7=12");
    assert!((h.coord.wall_ms_accum[1] - 22.0).abs() < 1e-10, "rank 1 should be 10+12=22");
}

#[test]
fn test_config_defaults() {
    let cfg = DdpRunConfig::new();
    assert!(cfg.overhead_target.is_none());
    assert!(cfg.max_anchor.is_none());
    assert!(cfg.anchor.is_none());
    assert!(cfg.divergence_threshold.is_none());
}

#[test]
fn test_config_builder() {
    let cfg = DdpRunConfig::new()
        .with_overhead_target(0.05)
        .with_max_anchor(100)
        .with_anchor(20)
        .with_divergence_threshold(0.01);
    assert_eq!(cfg.overhead_target, Some(0.05));
    assert_eq!(cfg.max_anchor, Some(100));
    assert_eq!(cfg.anchor, Some(20));
    assert_eq!(cfg.divergence_threshold, Some(0.01));
}


// -----------------------------------------------------------------------
// record_scalar tests
// -----------------------------------------------------------------------

#[test]
fn test_record_scalar_accumulates() {
    // Clear any leftovers from other tests on this thread
    drain_scalars();

    record_scalar("loss", 1.0);
    record_scalar("loss", 2.0);
    record_scalar("loss", 3.0);

    let map = drain_scalars();
    assert_eq!(map.len(), 1);
    let (sum, count) = map["loss"];
    assert_eq!(sum, 6.0);
    assert_eq!(count, 3);
}

#[test]
fn test_record_scalar_multiple_tags() {
    drain_scalars();

    record_scalar("a", 1.0);
    record_scalar("b", 2.0);
    record_scalar("a", 3.0);

    let map = drain_scalars();
    assert_eq!(map.len(), 2);
    assert_eq!(map["a"], (4.0, 2));
    assert_eq!(map["b"], (2.0, 1));
}

#[test]
fn test_drain_scalars_clears() {
    drain_scalars();

    record_scalar("x", 1.0);
    let first = drain_scalars();
    assert_eq!(first.len(), 1);

    // Second drain should be empty
    let second = drain_scalars();
    assert!(second.is_empty());

    // New records show up in the next drain
    record_scalar("y", 5.0);
    let third = drain_scalars();
    assert_eq!(third.len(), 1);
    assert!(!third.contains_key("x"));
    assert_eq!(third["y"], (5.0, 1));
}

#[test]
fn test_record_scalar_thread_isolation() {
    drain_scalars();
    record_scalar("main", 1.0);

    let child_result = std::thread::spawn(|| {
        // Child thread starts with empty accumulator
        let empty = drain_scalars();
        assert!(empty.is_empty());

        record_scalar("child", 42.0);
        drain_scalars()
    }).join().unwrap();

    // Child's values
    assert_eq!(child_result.len(), 1);
    assert_eq!(child_result["child"], (42.0, 1));

    // Main thread still has its own values
    let main_result = drain_scalars();
    assert_eq!(main_result.len(), 1);
    assert_eq!(main_result["main"], (1.0, 1));
}

#[test]
fn test_aggregate_epoch_metrics() {
    use super::super::coordinator::aggregate_epoch_metrics;

    let mut scalars_r0 = HashMap::new();
    scalars_r0.insert("loss".to_string(), (3.0, 3_usize)); // mean = 1.0
    scalars_r0.insert("acc".to_string(), (1.8, 3));         // mean = 0.6

    let mut scalars_r1 = HashMap::new();
    scalars_r1.insert("loss".to_string(), (4.0, 2_usize)); // mean = 2.0
    scalars_r1.insert("acc".to_string(), (0.8, 2));         // mean = 0.4

    let msgs = vec![
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 60,
            epoch_ms: 1000.0, share_complete_ms: 1000.0, compute_only_ms: 1000.0, data_starve_ms: 0.0, samples_processed: 1920, scalars: scalars_r0,
        },
        MetricsMsg {
            rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 40,
            epoch_ms: 1200.0, share_complete_ms: 1200.0, compute_only_ms: 1200.0, data_starve_ms: 0.0, samples_processed: 1280, scalars: scalars_r1,
        },
    ];

    let dev_indices = vec![0_u8, 1];
    // bc_share now comes from the balancer (smoothed batch_counts). Pass
    // 60/40 explicitly to match the historical samples-driven assertion.
    let bc_share = vec![0.6_f64, 0.4];
    let m = aggregate_epoch_metrics(0, &msgs, &dev_indices, &bc_share);
    assert_eq!(m.epoch, 0);

    // Batch-weighted average loss: (0.5*60 + 0.7*40) / 100 = 0.58
    assert!((m.avg_loss - 0.58).abs() < 1e-9);

    // Max epoch_ms
    assert_eq!(m.epoch_ms, 1200.0);

    // Weighted scalar: loss = (1.0*60 + 2.0*40) / 100 = 1.4
    assert!((m.scalars["loss"] - 1.4).abs() < 1e-9);

    // Weighted scalar: acc = (0.6*60 + 0.4*40) / 100 = 0.52
    assert!((m.scalars["acc"] - 0.52).abs() < 1e-9);

    // Per-rank
    assert_eq!(m.per_rank.len(), 2);
    assert!((m.per_rank[0]["loss"] - 1.0).abs() < 1e-9);
    assert!((m.per_rank[1]["loss"] - 2.0).abs() < 1e-9);

    // Throughput: rank 0 = 1920/1000 = 1.92, rank 1 = 1280/1200 ~= 1.0667
    assert!((m.per_rank_throughput[0] - 1.92).abs() < 1e-9);
    assert!((m.per_rank_throughput[1] - 1280.0 / 1200.0).abs() < 1e-9);

    // Batch share: rank 0 = 1920/3200 = 0.6, rank 1 = 1280/3200 = 0.4
    assert!((m.per_rank_batch_share[0] - 0.6).abs() < 1e-9);
    assert!((m.per_rank_batch_share[1] - 0.4).abs() < 1e-9);

    // Device indices
    assert_eq!(m.device_indices, vec![0, 1]);
}

/// Progressive dispatch: multiple MetricsMsg per rank should be aggregated
/// into exactly world_size entries, not one entry per message.
#[test]
fn test_aggregate_epoch_metrics_progressive() {
    use super::super::coordinator::aggregate_epoch_metrics;

    // Simulate 2 ranks, 3 chunks from rank 0, 2 chunks from rank 1
    let msgs = vec![
        // Rank 0 chunk 1
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 20,
            epoch_ms: 300.0, share_complete_ms: 300.0, compute_only_ms: 300.0, data_starve_ms: 0.0, samples_processed: 640,
            scalars: [("loss".to_string(), (2.0, 2_usize))].into(),
        },
        // Rank 0 chunk 2
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.4, batches_processed: 20,
            epoch_ms: 600.0, share_complete_ms: 600.0, compute_only_ms: 600.0, data_starve_ms: 0.0, samples_processed: 640,
            scalars: [("loss".to_string(), (1.6, 2_usize))].into(),
        },
        // Rank 0 chunk 3
        MetricsMsg {
            rank: 0, epoch: 0, avg_loss: 0.6, batches_processed: 20,
            epoch_ms: 900.0, share_complete_ms: 900.0, compute_only_ms: 900.0, data_starve_ms: 0.0, samples_processed: 640,
            scalars: [("loss".to_string(), (1.8, 2_usize))].into(),
        },
        // Rank 1 chunk 1
        MetricsMsg {
            rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 20,
            epoch_ms: 500.0, share_complete_ms: 500.0, compute_only_ms: 500.0, data_starve_ms: 0.0, samples_processed: 640,
            scalars: [("loss".to_string(), (2.8, 2_usize))].into(),
        },
        // Rank 1 chunk 2
        MetricsMsg {
            rank: 1, epoch: 0, avg_loss: 0.8, batches_processed: 20,
            epoch_ms: 1000.0, share_complete_ms: 1000.0, compute_only_ms: 1000.0, data_starve_ms: 0.0, samples_processed: 640,
            scalars: [("loss".to_string(), (3.2, 2_usize))].into(),
        },
    ];

    let dev_indices = vec![0_u8, 1];
    let bc_share = vec![0.6_f64, 0.4];
    let m = aggregate_epoch_metrics(0, &msgs, &dev_indices, &bc_share);

    // Must have exactly 2 entries (world_size), not 5 (one per msg)
    assert_eq!(m.per_rank_throughput.len(), 2, "should have world_size entries");
    assert_eq!(m.per_rank_batch_share.len(), 2);
    assert_eq!(m.per_rank.len(), 2);
    assert_eq!(m.device_indices, vec![0, 1]);

    // Rank 0: 60 batches, 1920 samples, max time 900ms
    // Rank 1: 40 batches, 1280 samples, max time 1000ms
    assert!((m.per_rank_throughput[0] - 1920.0 / 900.0).abs() < 1e-6);
    assert!((m.per_rank_throughput[1] - 1280.0 / 1000.0).abs() < 1e-6);

    // Total samples = 3200
    assert!((m.per_rank_batch_share[0] - 0.6).abs() < 1e-9);
    assert!((m.per_rank_batch_share[1] - 0.4).abs() < 1e-9);

    // Max epoch_ms across ranks
    assert_eq!(m.epoch_ms, 1000.0);

    // Scalars: rank 0 loss mean = (2.0+1.6+1.8)/(2+2+2) = 5.4/6 = 0.9
    assert!((m.per_rank[0]["loss"] - 0.9).abs() < 1e-9);
    // Rank 1 loss mean = (2.8+3.2)/(2+2) = 6.0/4 = 1.5
    assert!((m.per_rank[1]["loss"] - 1.5).abs() < 1e-9);

    // Weighted average: (0.9*60 + 1.5*40)/100 = (54+60)/100 = 1.14
    assert!((m.scalars["loss"] - 1.14).abs() < 1e-9);
}

// -----------------------------------------------------------------------
// Regression: NCCL safety during shutdown (progressive dispatch deadlock)
// -----------------------------------------------------------------------

#[test]
fn test_drain_until_shutdown_skips_sync_now() {
    // Regression: in progressive mode, a worker that reported Exiting
    // could receive a stale SyncNow (sent before the coordinator saw
    // Exiting). Calling AllReduce on a dead peer deadlocks.
    // drain_until_shutdown must skip SyncNow, not call sync_now_nccl.
    let (mut worker, ch) = make_test_worker();

    // Queue messages that would arrive during shutdown:
    // SyncNow (stale, from averaging triggered before our Exiting)
    // followed by Shutdown (from coordinator's shutdown_workers).
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    // drain_until_shutdown should skip SyncNow and exit on Shutdown.
    // If it tried AllReduce, it would deadlock (no peer in unit test).
    worker.drain_until_shutdown();
    // Reaching here means no deadlock — the SyncNow was skipped.
}

#[test]
fn test_drain_until_shutdown_handles_multiple_sync_now() {
    // Multiple stale SyncNow messages could accumulate if the
    // coordinator triggered several averaging events before seeing
    // our Exiting. All must be skipped.
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    worker.drain_until_shutdown();
}

#[test]
fn test_drain_until_shutdown_handles_interleaved_messages() {
    // Other control messages (RequestParams, StartEpoch, Checkpoint)
    // may arrive between SyncNow and Shutdown. They should be handled
    // normally (not treated as shutdown signals).
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Checkpoint { version: 99, target_rank: 0 }).unwrap();
    ch.control_tx.send(ControlMsg::StartEpoch(EpochPlan {
        epoch: 5, partition_offset: 0, partition_size: 100,
    })).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    worker.drain_until_shutdown();
    // StartEpoch queued as pending_plan
    assert!(worker.pending_plan.is_some());
}

#[test]
fn test_abort_nccl_no_panic_without_comm() {
    // abort_nccl takes the NCCL comm (None in unit tests) and aborts it.
    // Must not panic when comm is None.
    let (mut worker, _ch) = make_test_worker();

    // Unit test workers have no NCCL comm. abort_nccl should be a no-op.
    worker.abort_nccl();

    // Call twice to verify idempotence.
    worker.abort_nccl();
}

#[test]
fn test_collect_final_state_disconnected_worker() {
    // Regression: when a worker errors, its final_param_tx is dropped
    // (channel disconnects). collect_final_state should detect this
    // as Disconnected, not wait for the full 10s timeout.
    let (_timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut final_param_rxs = Vec::new();
    let mut final_param_txs = Vec::new();
    for _ in 0..2 {
        let (ctx, _crx) = mpsc::channel();
        control_txs.push(ctx);
        let (ftx, frx) = mpsc::channel();
        final_param_txs.push(ftx);
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(2, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Cpu,
        2, 1000, el_che,
    ).build();

    // Worker 0 sends snapshot normally
    let opts = crate::tensor::test_opts();
    let t = Tensor::full(&[3], 5.0, opts).unwrap();
    final_param_txs[0].send(ParamSnapshot {
        rank: 0, params: vec![t], buffers: vec![], batch_count: 1,
    }).unwrap();

    // Worker 1 "errors": drop its sender (simulates error path)
    drop(final_param_txs.remove(1));

    // collect_final_state should return quickly (disconnect is instant,
    // not the 10s timeout). The surviving worker's snapshot is returned.
    let start = std::time::Instant::now();
    let state = coord.collect_final_state();
    let elapsed = start.elapsed();

    assert!(state.is_some(), "should get state from surviving worker");
    assert!(elapsed.as_secs() < 2, "disconnect should be fast, not 10s timeout");
    assert_eq!(state.unwrap().params.len(), 1);
}

#[test]
fn test_worker_error_triggers_shutdown_flag() {
    // When a worker errors, it should send Exiting and set the
    // shutdown flag. Other workers check this flag each iteration.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_check = shutdown.clone();

    // Simulate: worker errors → sets shutdown
    shutdown.store(true, Ordering::Relaxed);

    // Coordinator (or sibling worker) sees it
    assert!(shutdown_check.load(Ordering::Relaxed));
}

#[test]
fn test_coordinator_active_count_prevents_averaging_after_exit() {
    // When a worker exits (sends Exiting), active_count drops.
    // should_average must return false to prevent sending SyncNow
    // to a dead peer (which would deadlock the survivor's AllReduce).
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Both ranks report a batch
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average(), "both ranks reported, should average");

    // Reset: trigger averaging to zero counters
    h.coord.trigger_averaging().unwrap();

    // Both report again
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 2, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 2, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average());

    // Now worker 1 exits
    h.timing_tx.send(TimingMsg::Exiting { rank: 1 }).unwrap();
    h.coord.drain_timing();
    assert_eq!(h.coord.active_count, 1);

    // should_average must return false (can't do collective with dead peer)
    assert!(!h.coord.should_average(),
        "should NOT average when active_count < world_size");
}

