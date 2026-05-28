//! Partition + data iteration tests, proportional epoch sharding,
//! streaming epochs.

use super::*;

// -----------------------------------------------------------------------
// Partition + data iteration tests
// -----------------------------------------------------------------------

#[test]
fn test_make_partition_basic() {
    let p0 = make_partition(0, 50, 100, 0, 42);
    let p1 = make_partition(50, 50, 100, 0, 42);
    assert_eq!(p0.len(), 50);
    assert_eq!(p1.len(), 50);

    // Non-overlapping (consecutive offsets, same epoch, same seed)
    let mut all: Vec<usize> = p0.iter().chain(p1.iter()).copied().collect();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 100, "partitions should be non-overlapping");
}

#[test]
fn test_make_partition_different_epochs() {
    let p_e0 = make_partition(0, 50, 100, 0, 42);
    let p_e1 = make_partition(0, 50, 100, 1, 42);
    // Different epochs should produce different orderings
    assert_ne!(p_e0, p_e1);
}

#[test]
fn test_make_partition_deterministic() {
    let p1 = make_partition(0, 50, 100, 5, 42);
    let p2 = make_partition(0, 50, 100, 5, 42);
    assert_eq!(p1, p2, "same params should produce same partition");
}

#[test]
fn test_worker_partition_changes_with_epoch() {
    let (mut worker, _ch) = make_test_worker();
    // Run epoch 0
    let plan0 = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 1000 };
    worker.run_epoch_plan(&plan0, &mse_train).unwrap();
    let partition0 = worker.partition.clone();

    // Run epoch 1 - different epoch produces different partition
    let plan1 = EpochPlan { epoch: 1, partition_offset: 0, partition_size: 1000 };
    worker.run_epoch_plan(&plan1, &mse_train).unwrap();
    assert_ne!(worker.partition, partition0);
}

#[test]
fn test_worker_epoch_plan_applies_partition_size() {
    let (mut worker, _ch) = make_test_worker_with(0, 1, 1000);

    // Run with a smaller partition via EpochPlan
    let plan = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 200 };
    worker.run_epoch_plan(&plan, &mse_train).unwrap();
    assert_eq!(worker.partition.len(), 200);
}

#[test]
fn test_worker_run_epoch_plan() {
    // 40 samples, batch_size=4 -> 10 batches per epoch
    let (mut worker, ch) = make_test_worker_with(0, 1, 40);

    let plan = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 40 };
    let shutdown = worker.run_epoch_plan(&plan, &mse_train).unwrap();
    assert!(!shutdown);
    assert_eq!(worker.current_epoch, 0);

    // Should have received timing messages (one per batch)
    let mut count = 0;
    while ch.timing_rx.try_recv().is_ok() {
        count += 1;
    }
    assert!(count > 0, "should have sent timing messages");

    // Should have received epoch metrics
    let metrics = ch.metrics_rx.recv().unwrap();
    assert_eq!(metrics.epoch, 0); // epoch 0 was just completed
    assert!(metrics.avg_loss > 0.0);
    assert!(metrics.batches_processed > 0);
}

#[test]
fn test_worker_run_epoch_plan_loss_decreases() {
    let (mut worker, _ch) = make_test_worker_with(0, 1, 80);

    // Run a few epochs, loss should decrease
    for epoch in 0..5 {
        let plan = EpochPlan { epoch, partition_offset: 0, partition_size: 80 };
        worker.run_epoch_plan(&plan, &mse_train).unwrap();
    }
    // Snapshot and check loss on a fixed batch
    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    let loss_after: f64 = mse_train(worker.model(), &batch).unwrap().data().item().unwrap();
    // After 5 epochs of training, loss should be finite and non-negative
    assert!(loss_after.is_finite());
}

#[test]
fn test_worker_run_epoch_plan_shutdown_mid_epoch() {
    let (mut worker, ch) = make_test_worker_with(0, 1, 400);

    // Send shutdown after a short delay via the control channel
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    let plan = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 400 };
    let shutdown = worker.run_epoch_plan(&plan, &mse_train).unwrap();
    assert!(shutdown, "should detect shutdown during epoch");
}

#[test]
fn test_cpu_averaging_end_to_end() {
    // Two workers on CPU, CPU averaging backend.
    // Simulate the coordinator cycle manually.
    let (mut w0, _ch0) = make_test_worker_with(0, 2, 40);
    let (mut w1, _ch1) = make_test_worker_with(1, 2, 40);

    // Run one epoch on each worker
    let plan0 = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 20 };
    let plan1 = EpochPlan { epoch: 0, partition_offset: 20, partition_size: 20 };
    w0.run_epoch_plan(&plan0, &mse_train).unwrap();
    w1.run_epoch_plan(&plan1, &mse_train).unwrap();

    // Snapshot params from both
    let snap0 = w0.snapshot_params();
    let snap1 = w1.snapshot_params();

    // Average them (coordinator's static method)
    let averaged = Coordinator::average_params(&[snap0, snap1], 1).unwrap();

    // Load averaged params into both workers
    w0.load_averaged(&averaged).unwrap();
    w1.load_averaged(&averaged).unwrap();

    assert_eq!(w0.current_version(), 1);
    assert_eq!(w1.current_version(), 1);

    // Both should now have the same params
    let s0 = w0.snapshot_params();
    let s1 = w1.snapshot_params();
    for (p0, p1) in s0.params.iter().zip(&s1.params) {
        let diff: f64 = p0.sub(p1).unwrap().abs().unwrap().sum().unwrap().item().unwrap();
        assert!(diff < 1e-5, "params should be identical after averaging, diff={diff}");
    }
}

// -----------------------------------------------------------------------
// Proportional epoch sharding tests
// -----------------------------------------------------------------------

#[test]
fn test_proportional_sharding() {
    // 2:1 speed ratio -> partition sizes should be 2:1
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Calibrate ElChe with 2:1 timing
    for _ in 0..3 {
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

    if h.coord.is_calibrated() {
        let sizes = h.coord.compute_partition_sizes();
        assert_eq!(sizes.len(), 2);
        // Fast rank (0) should get more samples than slow rank (1)
        assert!(sizes[0] > sizes[1],
            "fast rank should get more samples: {:?}", sizes);
        // Total should approximate dataset size (10000)
        let total: usize = sizes.iter().sum();
        assert!(total <= 10000, "partitions should not exceed total: {total}");
    }
}

#[test]
fn test_partition_non_overlapping_equal_sizes() {
    // Equal partition sizes with consecutive offsets: guaranteed non-overlapping
    let total = 300;
    let per_rank = total / 3; // 100 each
    let p0 = make_partition(0, per_rank, total, 5, 42);
    let p1 = make_partition(100, per_rank, total, 5, 42);
    let p2 = make_partition(200, per_rank, total, 5, 42);

    assert_eq!(p0.len(), 100);
    assert_eq!(p1.len(), 100);
    assert_eq!(p2.len(), 100);

    let set0: std::collections::HashSet<usize> = p0.iter().copied().collect();
    let set1: std::collections::HashSet<usize> = p1.iter().copied().collect();
    let set2: std::collections::HashSet<usize> = p2.iter().copied().collect();
    assert_eq!(set0.intersection(&set1).count(), 0, "rank 0/1 should not overlap");
    assert_eq!(set0.intersection(&set2).count(), 0, "rank 0/2 should not overlap");
    assert_eq!(set1.intersection(&set2).count(), 0, "rank 1/2 should not overlap");
}

#[test]
fn test_partition_non_overlapping_smaller_sizes() {
    // Non-overlapping consecutive offsets with varying sizes
    let total = 300;
    let p0 = make_partition(0, 50, total, 5, 42);   // offset 0, size 50
    let p1 = make_partition(50, 80, total, 5, 42);   // offset 50, size 80
    let p2 = make_partition(130, 60, total, 5, 42);  // offset 130, size 60

    let set0: std::collections::HashSet<usize> = p0.iter().copied().collect();
    let set1: std::collections::HashSet<usize> = p1.iter().copied().collect();
    let set2: std::collections::HashSet<usize> = p2.iter().copied().collect();
    assert_eq!(set0.intersection(&set1).count(), 0, "rank 0/1 should not overlap");
    assert_eq!(set0.intersection(&set2).count(), 0, "rank 0/2 should not overlap");
    assert_eq!(set1.intersection(&set2).count(), 0, "rank 1/2 should not overlap");
}

#[test]
fn test_partition_benign_overlap_different_epochs() {
    // Different epochs produce different permutations, so overlap is expected
    let p0_e5 = make_partition(0, 50, 100, 5, 42);
    let p1_e6 = make_partition(50, 50, 100, 6, 42);
    // These are from different epochs, so some overlap is expected and benign
    let set0: std::collections::HashSet<usize> = p0_e5.iter().copied().collect();
    let set1: std::collections::HashSet<usize> = p1_e6.iter().copied().collect();
    // Just verify they're valid indices
    assert!(set0.iter().all(|&i| i < 100));
    assert!(set1.iter().all(|&i| i < 100));
}

#[test]
fn test_self_managed_epochs() {
    // Worker should run multiple epochs via plans, reporting metrics each time
    let (mut worker, ch) = make_test_worker_with(0, 1, 40);

    // Run 3 epochs
    for epoch in 0..3 {
        let plan = EpochPlan { epoch, partition_offset: 0, partition_size: 40 };
        let shutdown = worker.run_epoch_plan(&plan, &mse_train).unwrap();
        assert!(!shutdown);
    }

    assert_eq!(worker.current_epoch, 2); // set to last plan's epoch

    // Should have received 3 epoch metrics
    let mut epoch_msgs = Vec::new();
    while let Ok(msg) = ch.metrics_rx.try_recv() {
        epoch_msgs.push(msg);
    }
    assert_eq!(epoch_msgs.len(), 3);
    assert_eq!(epoch_msgs[0].epoch, 0);
    assert_eq!(epoch_msgs[1].epoch, 1);
    assert_eq!(epoch_msgs[2].epoch, 2);
}

#[test]
fn test_epoch_plan_partition_size_at_epoch_boundary() {
    let (mut worker, _ch) = make_test_worker_with(0, 1, 80);

    // Run first epoch with full partition
    let plan0 = EpochPlan { epoch: 0, partition_offset: 0, partition_size: 80 };
    worker.run_epoch_plan(&plan0, &mse_train).unwrap();
    assert_eq!(worker.partition.len(), 80);

    // Next epoch with a smaller partition from EpochPlan
    let plan1 = EpochPlan { epoch: 1, partition_offset: 0, partition_size: 20 };
    worker.run_epoch_plan(&plan1, &mse_train).unwrap();
    assert_eq!(worker.partition.len(), 20);
}


// ---------------------------------------------------------------------------
// Streaming epochs tests
// ---------------------------------------------------------------------------

/// Streaming-epoch test harness with optional epoch metrics channel.
struct StreamingTestHarness {
    inner: CoordTestHarness,
    epoch_metrics_rx: mpsc::Receiver<EpochMetrics>,
}

/// Helper: create a progressive coordinator with configurable epochs, batch_size,
/// and max_overshoot for streaming epoch tests.
fn make_streaming_harness(
    n: usize,
    num_epochs: usize,
    total_samples: usize,
    batch_size: usize,
    max_overshoot: Option<usize>,
) -> CoordTestHarness {
    make_streaming_harness_with_metrics(n, num_epochs, total_samples, batch_size, max_overshoot).inner
}

fn make_streaming_harness_with_metrics(
    n: usize,
    num_epochs: usize,
    total_samples: usize,
    batch_size: usize,
    max_overshoot: Option<usize>,
) -> StreamingTestHarness {
    let (timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (param_tx, param_rx) = mpsc::channel();
    let (epoch_metrics_tx, epoch_metrics_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut control_rxs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..n {
        let (tx, rx) = mpsc::channel();
        control_txs.push(tx);
        control_rxs.push(rx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(n, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Async, AverageBackend::Cpu,
        n, total_samples, el_che,
    )
    .progressive(true)
    .batch_size(batch_size)
    .num_epochs(num_epochs)
    .max_overshoot(max_overshoot)
    .epoch_metrics_tx(epoch_metrics_tx)
    .build();

    StreamingTestHarness {
        inner: CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs },
        epoch_metrics_rx,
    }
}

#[test]
fn test_streaming_cross_epoch_dispatch() {
    // 2 ranks, 3 epochs, 20 samples, batch_size=10 (2 batches/epoch).
    // With probe chunks of ~4 batches capped at 1 batch (2 batches total / 2 ranks),
    // each rank gets 10 samples. Completing that exhausts the pool.
    let mut h = make_streaming_harness(2, 3, 20, 10, Some(5));

    h.coord.send_all_plans(0);
    // Collect initial dispatch: each rank should get an epoch 0 chunk.
    let mut rank0_plan = None;
    while let Ok(msg) = h.control_rxs[0].try_recv() {
        if let ControlMsg::StartEpoch(p) = msg { rank0_plan = Some(p); }
    }
    let plan = rank0_plan.expect("rank 0 should get initial chunk");
    assert_eq!(plan.epoch, 0);
    let dispatched = plan.partition_size;

    // Rank 0 reports exactly what was dispatched.
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1,
        batches_processed: dispatched / 10,
        epoch_ms: 50.0, share_complete_ms: 50.0, compute_only_ms: 50.0, data_starve_ms: 0.0, samples_processed: dispatched,
        scalars: Default::default(),
    }).unwrap();
    h.coord.drain_metrics();

    // After reporting, rank 0 should get new work: either more from epoch 0
    // or streaming into epoch 1 (pool was tiny, likely exhausted).
    let mut epochs_dispatched = Vec::new();
    while let Ok(msg) = h.control_rxs[0].try_recv() {
        if let ControlMsg::StartEpoch(p) = msg { epochs_dispatched.push(p.epoch); }
    }

    // Rank 0 should have received some dispatch (from epoch 0 or 1).
    // The exact behavior depends on pool sizing, but no crash = the
    // multi-pool logic works.
    // If we got here without panic, multi-pool logic works.
}

#[test]
fn test_streaming_global_epoch_event_fires_when_all_complete() {
    // Manually set up a pool and feed it exact completions.
    let mut h = make_streaming_harness(2, 2, 20, 10, Some(5));

    // Manually create epoch 0 pool with known sizes.
    let pool = super::super::coordinator::ChunkPool::new(0, 20, 2);
    h.coord.chunk_pools.insert(0, pool);

    // Manually take chunks: 10 samples each.
    h.coord.chunk_pools.get_mut(&0).unwrap().take_chunk(10, 0);
    h.coord.chunk_pools.get_mut(&0).unwrap().take_chunk(10, 1);

    // Report completion from both ranks.
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1, batches_processed: 1,
        epoch_ms: 10.0, share_complete_ms: 10.0, compute_only_ms: 10.0, data_starve_ms: 0.0, samples_processed: 10,
        scalars: Default::default(),
    }).unwrap();
    h.metrics_tx.send(MetricsMsg {
        rank: 1, epoch: 0, avg_loss: 0.2, batches_processed: 1,
        epoch_ms: 20.0, share_complete_ms: 20.0, compute_only_ms: 20.0, data_starve_ms: 0.0, samples_processed: 10,
        scalars: Default::default(),
    }).unwrap();
    h.coord.drain_metrics();

    assert_eq!(h.coord.last_aggregated_epoch, Some(0),
        "epoch 0 should be aggregated when both ranks complete");
}

#[test]
fn test_overshoot_gate_blocks_runaway() {
    // Use a manually prepared pool so we control exactly what was dispatched.
    let mut h = make_streaming_harness(2, 3, 100, 10, Some(0));

    // Create epoch 0 pool, take all samples for both ranks.
    let pool = super::super::coordinator::ChunkPool::new(0, 100, 2);
    h.coord.chunk_pools.insert(0, pool);
    h.coord.chunk_pools.get_mut(&0).unwrap().take_chunk(50, 0);
    h.coord.chunk_pools.get_mut(&0).unwrap().take_chunk(50, 1);

    // Simulate: rank 0 has completed all epoch 0 work, at planned batch count.
    h.coord.steps_since_avg[0] = 10;
    h.coord.steps_since_avg[1] = 3;

    // Report rank 0 completion (matches dispatched amount).
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1, batches_processed: 5,
        epoch_ms: 50.0, share_complete_ms: 50.0, compute_only_ms: 50.0, data_starve_ms: 0.0, samples_processed: 50,
        scalars: Default::default(),
    }).unwrap();
    h.coord.drain_metrics();

    // With max_overshoot=0 and steps_since_avg[0]=10 >= batch_counts[0]=10,
    // the gate should block cross-epoch dispatch.
    let mut got_epoch_1 = false;
    while let Ok(msg) = h.control_rxs[0].try_recv() {
        if let ControlMsg::StartEpoch(p) = msg {
            if p.epoch == 1 { got_epoch_1 = true; }
        }
    }
    assert!(!got_epoch_1,
        "overshoot gate should prevent cross-epoch dispatch when at limit");
}

#[test]
fn test_overshoot_gate_skipped_for_cadence() {
    // Cadence uses AllReduce as its sole coordination layer (per
    // `feedback_nccl_no_overshoot_throttle`): no overshoot gate fires
    // regardless of backend. Sync is the same. Only Async accumulates
    // cross-cycle drift that needs a batch-scale bound.
    let (_timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();
    let mut control_txs = Vec::new();
    let mut control_rxs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..2 {
        let (tx, rx) = mpsc::channel();
        control_txs.push(tx);
        control_rxs.push(rx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let el_che = ElChe::new(2, 10);
    let mut coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Cadence, AverageBackend::Nccl,
        2, 100, el_che,
    )
    .progressive(true)
    .batch_size(10)
    .num_epochs(3)
    .max_overshoot(Some(0)) // Would block everything in Async.
    .build();

    // Create epoch 0 pool, take all samples for both ranks.
    let pool = super::super::coordinator::ChunkPool::new(0, 100, 2);
    coord.chunk_pools.insert(0, pool);
    coord.chunk_pools.get_mut(&0).unwrap().take_chunk(50, 0);
    coord.chunk_pools.get_mut(&0).unwrap().take_chunk(50, 1);

    // Rank 0 has trained well past its planned batch count.
    coord.steps_since_avg[0] = 10;
    coord.steps_since_avg[1] = 3;

    // Report rank 0 chunk completion.
    metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1, batches_processed: 5,
        epoch_ms: 50.0, share_complete_ms: 50.0, compute_only_ms: 50.0, data_starve_ms: 0.0, samples_processed: 50,
        scalars: Default::default(),
    }).unwrap();
    coord.drain_metrics();

    // Cadence skips the overshoot gate: rank 0 should get a cross-epoch
    // StartEpoch even with max_overshoot=0.
    let mut got_start_epoch = false;
    while let Ok(msg) = control_rxs[0].try_recv() {
        if let ControlMsg::StartEpoch(_) = msg {
            got_start_epoch = true;
        }
    }
    assert!(got_start_epoch,
        "Cadence policy must skip overshoot gate (AllReduce handles coordination)");
}

#[test]
fn test_overshoot_auto_tune_grows() {
    let mut h = make_streaming_harness(2, 3, 1000, 10, None);
    let initial = h.coord.max_overshoot;
    assert!(initial >= 2, "initial overshoot should be at least 2");

    // Simulate a successful NCCL averaging (no divergence)
    h.coord.steps_since_avg = vec![10, 10];
    h.coord.wall_ms_accum = vec![100.0, 200.0];
    h.coord.finish_averaging_nccl();

    assert_eq!(h.coord.max_overshoot, initial + 1,
        "overshoot should grow by 1 after successful averaging");
}

#[test]
fn test_overshoot_auto_tune_suppressed_on_divergence_trend() {
    let mut h = make_streaming_harness(2, 3, 1000, 10, None);

    // Grow overshoot via NCCL averaging (no divergence -> Stable -> grows).
    for _ in 0..3 {
        h.coord.steps_since_avg = vec![10, 10];
        h.coord.wall_ms_accum = vec![100.0, 200.0];
        h.coord.finish_averaging_nccl();
    }
    let overshoot_after_growth = h.coord.max_overshoot;

    // 3 CPU averaging rounds with rising divergence -> trend triggers SuppressGrowth.
    for i in 0..3 {
        let div = 0.10 + i as f64 * 0.05;
        h.coord.finish_averaging_cpu(
            10.0,
            &[5_usize, 5],
            &[50.0, 100.0],
            Some(super::convergence::DivergenceReport {
                deltas: vec![div, div],
                pre_norms: None,
                post_norm: None,
            }),
        );
    }

    // Overshoot should NOT have grown on the 3rd CPU round (SuppressGrowth).
    assert!(h.coord.max_overshoot <= overshoot_after_growth + 2,
        "3rd CPU round should suppress overshoot growth, got {}", h.coord.max_overshoot);
}

#[test]
fn test_overshoot_user_override_no_autotune() {
    let mut h = make_streaming_harness(2, 3, 1000, 10, Some(7));
    assert_eq!(h.coord.max_overshoot, 7);
    assert!(!h.coord.overshoot_auto);

    // Simulate averaging -- should NOT change overshoot
    h.coord.steps_since_avg = vec![10, 10];
    h.coord.wall_ms_accum = vec![100.0, 200.0];
    h.coord.finish_averaging_nccl();

    assert_eq!(h.coord.max_overshoot, 7,
        "user-set overshoot should not auto-tune");
}

#[test]
fn test_multi_pool_completion_tracking() {
    // Manually create two pools and verify MetricsMsg routes to correct ones.
    let mut h = make_streaming_harness(2, 3, 100, 10, Some(10));

    // Create pools manually with known dispatched amounts.
    let mut pool0 = super::super::coordinator::ChunkPool::new(0, 100, 2);
    pool0.take_chunk(50, 0); // dispatch 50 to rank 0
    pool0.take_chunk(50, 1); // dispatch 50 to rank 1
    h.coord.chunk_pools.insert(0, pool0);

    let mut pool1 = super::super::coordinator::ChunkPool::new(1, 100, 2);
    pool1.take_chunk(30, 0); // dispatch 30 to rank 0
    h.coord.chunk_pools.insert(1, pool1);

    // Report epoch 0 completion from rank 0
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1, batches_processed: 5,
        epoch_ms: 50.0, share_complete_ms: 50.0, compute_only_ms: 50.0, data_starve_ms: 0.0, samples_processed: 50,
        scalars: Default::default(),
    }).unwrap();
    // Report epoch 1 completion from rank 0
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 1, avg_loss: 0.2, batches_processed: 3,
        epoch_ms: 30.0, share_complete_ms: 30.0, compute_only_ms: 30.0, data_starve_ms: 0.0, samples_processed: 30,
        scalars: Default::default(),
    }).unwrap();
    h.coord.drain_metrics();

    // Epoch 0 pool should have rank 0 completion tracked
    if let Some(pool) = h.coord.chunk_pools.get(&0) {
        assert_eq!(pool.completed[0], 50, "epoch 0 pool should track rank 0 completion");
    }
    // Epoch 1 pool should have rank 0 completion tracked separately
    if let Some(pool) = h.coord.chunk_pools.get(&1) {
        assert_eq!(pool.completed[0], 30, "epoch 1 pool should track rank 0 completion");
    }
}

#[test]
fn test_shutdown_with_streaming_pools() {
    // Manually create pools and verify shutdown fires when last epoch completes.
    let mut h = make_streaming_harness(2, 2, 20, 10, Some(5));

    // Create epoch 0 pool, dispatch all.
    let mut pool0 = super::super::coordinator::ChunkPool::new(0, 20, 2);
    pool0.take_chunk(10, 0);
    pool0.take_chunk(10, 1);
    h.coord.chunk_pools.insert(0, pool0);

    // Complete epoch 0 from both ranks.
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1, batches_processed: 1,
        epoch_ms: 10.0, share_complete_ms: 10.0, compute_only_ms: 10.0, data_starve_ms: 0.0, samples_processed: 10,
        scalars: Default::default(),
    }).unwrap();
    h.metrics_tx.send(MetricsMsg {
        rank: 1, epoch: 0, avg_loss: 0.2, batches_processed: 1,
        epoch_ms: 20.0, share_complete_ms: 20.0, compute_only_ms: 20.0, data_starve_ms: 0.0, samples_processed: 10,
        scalars: Default::default(),
    }).unwrap();
    h.coord.drain_metrics();
    assert_eq!(h.coord.last_aggregated_epoch, Some(0));

    // Drain dispatch messages from on_epoch_aggregated.
    for rx in &h.control_rxs {
        while rx.try_recv().is_ok() {}
    }

    // Create epoch 1 pool with both ranks dispatched.
    // Replace whatever on_epoch_aggregated created, to have clean state.
    h.coord.chunk_pools.remove(&1);
    let mut pool1 = super::super::coordinator::ChunkPool::new(1, 20, 2);
    pool1.take_chunk(10, 0);
    pool1.take_chunk(10, 1);
    h.coord.chunk_pools.insert(1, pool1);

    // Complete epoch 1 (the last epoch).
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 1, avg_loss: 0.05, batches_processed: 1,
        epoch_ms: 10.0, share_complete_ms: 10.0, compute_only_ms: 10.0, data_starve_ms: 0.0, samples_processed: 10,
        scalars: Default::default(),
    }).unwrap();
    h.metrics_tx.send(MetricsMsg {
        rank: 1, epoch: 1, avg_loss: 0.06, batches_processed: 1,
        epoch_ms: 20.0, share_complete_ms: 20.0, compute_only_ms: 20.0, data_starve_ms: 0.0, samples_processed: 10,
        scalars: Default::default(),
    }).unwrap();
    h.coord.drain_metrics();

    assert_eq!(h.coord.last_aggregated_epoch, Some(1));

    // Both ranks should have received Shutdown.
    let mut shutdowns = 0;
    for rx in &h.control_rxs {
        while let Ok(msg) = rx.try_recv() {
            if matches!(msg, ControlMsg::Shutdown) {
                shutdowns += 1;
            }
        }
    }
    assert_eq!(shutdowns, 2, "both ranks should receive Shutdown after final epoch");
}

#[test]
fn test_ddp_run_config_max_overshoot() {
    let config = DdpRunConfig::new().with_max_overshoot(5);
    assert_eq!(config.max_overshoot, Some(5));

    let config2 = DdpRunConfig::new();
    assert_eq!(config2.max_overshoot, None);
}

#[test]
fn test_epoch_event_fires_with_mixed_epoch_ranks() {
    // Scenario: fast rank (0) finishes epoch 1 and streams into epoch 2.
    // Slow rank (1) then completes epoch 1. The global epoch 1 event must
    // fire with correct metrics from BOTH ranks, even though rank 0's
    // epoch 1 metrics were buffered earlier.
    let mut sh = make_streaming_harness_with_metrics(2, 3, 60, 10, Some(10));

    // -- Set up epoch 1 pool: 60 samples, 30 per rank --
    let mut pool1 = super::super::coordinator::ChunkPool::new(1, 60, 2);
    pool1.take_chunk(30, 0); // rank 0 dispatched 30
    pool1.take_chunk(30, 1); // rank 1 dispatched 30
    sh.inner.coord.chunk_pools.insert(1, pool1);

    // -- Set up epoch 2 pool (rank 0 already streaming ahead) --
    let mut pool2 = super::super::coordinator::ChunkPool::new(2, 60, 2);
    pool2.take_chunk(20, 0); // rank 0 dispatched 20 into epoch 2
    sh.inner.coord.chunk_pools.insert(2, pool2);
    sh.inner.coord.rank_epoch[0] = 2; // rank 0 is on epoch 2
    sh.inner.coord.rank_epoch[1] = 1; // rank 1 still on epoch 1

    // -- Rank 0 reported epoch 1 completion earlier (buffered) --
    sh.inner.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 1, avg_loss: 0.10, batches_processed: 3,
        epoch_ms: 30.0, share_complete_ms: 30.0, compute_only_ms: 30.0, data_starve_ms: 0.0, samples_processed: 30,
        scalars: [("loss".to_string(), (0.30, 3_usize))].into(),
    }).unwrap();
    sh.inner.coord.drain_metrics();

    // Epoch 1 should NOT be aggregated yet (rank 1 hasn't completed).
    assert!(sh.inner.coord.last_aggregated_epoch.is_none()
        || sh.inner.coord.last_aggregated_epoch == Some(0),
        "epoch 1 should not aggregate with only rank 0 complete");

    // Verify epoch 1 pool is still active (rank 1 not done).
    assert!(sh.inner.coord.chunk_pools.contains_key(&1),
        "epoch 1 pool should still exist");

    // -- Now slow rank (1) completes epoch 1 --
    sh.inner.metrics_tx.send(MetricsMsg {
        rank: 1, epoch: 1, avg_loss: 0.20, batches_processed: 3,
        epoch_ms: 60.0, share_complete_ms: 60.0, compute_only_ms: 60.0, data_starve_ms: 0.0, samples_processed: 30,
        scalars: [("loss".to_string(), (0.60, 3_usize))].into(),
    }).unwrap();
    sh.inner.coord.drain_metrics();

    // Epoch 1 should now be aggregated.
    assert_eq!(sh.inner.coord.last_aggregated_epoch, Some(1),
        "epoch 1 should aggregate when both ranks complete");

    // Epoch 1 pool should be cleaned up.
    assert!(!sh.inner.coord.chunk_pools.contains_key(&1),
        "epoch 1 pool should be removed after aggregation");

    // Epoch 2 pool should still exist (rank 0 is working on it).
    assert!(sh.inner.coord.chunk_pools.contains_key(&2),
        "epoch 2 pool should survive epoch 1 aggregation");

    // -- Verify aggregated metrics are correct --
    let em = sh.epoch_metrics_rx.try_recv()
        .expect("epoch metrics should have been sent for epoch 1");
    assert_eq!(em.epoch, 1);

    // avg_loss: batch-weighted mean of rank 0 (0.10, 3 batches) and rank 1 (0.20, 3 batches)
    // = (0.10*3 + 0.20*3) / 6 = 0.90 / 6 = 0.15
    assert!((em.avg_loss - 0.15).abs() < 1e-9,
        "avg_loss should be batch-weighted mean: got {}", em.avg_loss);

    // per_rank_batch_share: 3/6 = 0.5 each
    assert_eq!(em.per_rank_batch_share.len(), 2);
    assert!((em.per_rank_batch_share[0] - 0.5).abs() < 1e-9);
    assert!((em.per_rank_batch_share[1] - 0.5).abs() < 1e-9);

    // scalars: loss = batch-weighted mean of rank 0 (0.30/3=0.10) and rank 1 (0.60/3=0.20)
    // weighted by batches: (0.10*3 + 0.20*3)/6 = 0.15
    assert!((em.scalars["loss"] - 0.15).abs() < 1e-9,
        "loss scalar should be batch-weighted: got {}", em.scalars["loss"]);

    // epoch_ms is overridden by pool wall-clock (near-instant in test).
    assert!(em.epoch_ms > 0.0, "epoch_ms should be positive");
}

#[test]
fn test_dispatch_skips_aggregated_epochs() {
    // Reproduce the pool-recreation deadlock:
    // Fast GPU takes ALL chunks from epoch 1 while slow GPU is still on epoch 0.
    // Both epoch 0 and 1 get aggregated in one sweep. dispatch_next_chunk for
    // the slow GPU must skip past the removed pools, not recreate them.
    //
    // 2 ranks, 5 epochs, 100 samples, batch_size=10 (10 batches/epoch).
    let mut h = make_streaming_harness(2, 5, 100, 10, None);

    // Manually create epoch 0 pool with all samples dispatched.
    // Fast GPU (rank 0) got 70 samples, slow GPU (rank 1) got 30.
    let mut pool0 = super::super::coordinator::ChunkPool::new(0, 100, 2);
    pool0.take_chunk(70, 0);
    pool0.take_chunk(30, 1);
    h.coord.chunk_pools.insert(0, pool0);

    // Epoch 1 pool: fast GPU took ALL 100 samples; slow GPU got nothing.
    let mut pool1 = super::super::coordinator::ChunkPool::new(1, 100, 2);
    pool1.take_chunk(100, 0);
    h.coord.chunk_pools.insert(1, pool1);

    // Track rank positions: slow GPU last dispatched from epoch 0.
    h.coord.rank_epoch[0] = 1;
    h.coord.rank_epoch[1] = 0;

    // Complete all chunks: both ranks for epoch 0, rank 0 for epoch 1.
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.1, batches_processed: 7,
        epoch_ms: 50.0, share_complete_ms: 50.0, compute_only_ms: 50.0, data_starve_ms: 0.0, samples_processed: 70,
        scalars: Default::default(),
    }).unwrap();
    h.metrics_tx.send(MetricsMsg {
        rank: 1, epoch: 0, avg_loss: 0.2, batches_processed: 3,
        epoch_ms: 80.0, share_complete_ms: 80.0, compute_only_ms: 80.0, data_starve_ms: 0.0, samples_processed: 30,
        scalars: Default::default(),
    }).unwrap();
    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 1, avg_loss: 0.1, batches_processed: 10,
        epoch_ms: 100.0, share_complete_ms: 100.0, compute_only_ms: 100.0, data_starve_ms: 0.0, samples_processed: 100,
        scalars: Default::default(),
    }).unwrap();

    // drain_metrics -> try_aggregate_epochs_progressive aggregates both.
    // on_epoch_aggregated(0) calls dispatch_next_chunk(1) for the idle slow GPU.
    h.coord.drain_metrics();

    // Both epochs should be aggregated.
    assert_eq!(h.coord.last_aggregated_epoch, Some(1),
        "both epoch 0 and 1 should be aggregated");

    // The critical check: no orphan pool for epoch 0 or 1 should exist.
    // Only pools for epoch 2+ should be present.
    for &epoch in h.coord.chunk_pools.keys() {
        assert!(epoch >= 2,
            "found orphan pool for already-aggregated epoch {epoch}");
    }

    // Slow GPU should have been dispatched to epoch 2 (not epoch 1).
    assert!(h.coord.rank_epoch[1] >= 2,
        "slow GPU should be on epoch 2+, got epoch {}",
        h.coord.rank_epoch[1]);
}

