//! Coordinator + Throttle (max_batch_diff) unit tests.

use super::*;


#[test]
fn test_coordinator_initial_state() {
    let h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);
    assert_eq!(h.coord.version(), 0);
    assert!(!h.coord.is_calibrated());
    assert_eq!(h.coord.steps_since_avg(), &[0, 0]);
}

#[test]
fn test_coordinator_drain_timing() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();

    h.coord.drain_timing();

    assert_eq!(h.coord.steps_since_avg(), &[1, 1]);
}

#[test]
fn test_coordinator_should_average_sync() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Not ready yet (no steps)
    assert!(!h.coord.should_average());

    // One rank reports
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    assert!(!h.coord.should_average()); // rank 1 still at 0

    // Both ranks report
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average());
}

#[test]
fn test_coordinator_should_average_async() {
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

    // Async now uses batch_counts() same as Cadence (anchor=10 from harness).
    // Feed 9 steps per rank: not enough yet.
    for _ in 0..9 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(!h.coord.should_average());

    // 10th step: both ranks reach batch_counts (anchor=10, uncalibrated so equal).
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    assert!(h.coord.should_average());
}

#[test]
fn test_coordinator_should_average_wall_time() {
    // After calibration, Cadence uses wall-time trigger (not batch counts).
    // Async keeps batch-count trigger (overshooting is the feature).
    // Setup: 2 ranks, anchor=10, rank 0 = 5ms/batch (fast), rank 1 = 10ms/batch (slow).
    // anchor_wall_ms = 10 * 10 = 100ms.
    let mut h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Phase 1: calibrate ElChe (uncalibrated uses batch-count fallback).
    // Send 10 batches per rank to trigger initial averaging.
    // step_count must increment to satisfy NCCL ack tracking.
    for i in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i + 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: i + 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average()); // batch-count fallback: 10 >= 10
    h.coord.trigger_averaging().unwrap();
    for rx in &h.control_rxs { while rx.try_recv().is_ok() {} }

    assert!(h.coord.is_calibrated());
    let target = h.coord.el_che.anchor_wall_ms();
    assert!(target > 0.0, "anchor_wall_ms should be positive after calibration");

    // Phase 2: wall-time trigger. The slow rank needs target ms of compute.
    // Feed batches until slow rank reaches target, but NOT until batch_counts
    // are met. This proves wall time triggers, not batch counts.
    //
    // After calibration with 2:1 ratio, batch_counts ≈ [20, 10].
    // If we feed 10 batches to each: wall_ms_accum = [50, 100].
    // min(50, 100) = 50 < 100 → no trigger (fast rank hasn't accumulated enough).
    for i in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 11 + i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: 11 + i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(!h.coord.should_average(), "fast rank wall time < target");

    // Feed 10 more to rank 0 only (simulating fast GPU running ahead).
    // wall_ms_accum = [100, 100]. min = 100 >= target → trigger!
    for i in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 21 + i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average(), "both ranks at target wall time");
}

#[test]
fn test_async_uses_batch_count_not_wall_time() {
    // Async keeps batch-count trigger even after calibration.
    // The divergence between replicas IS the feature (exploration diversity).
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

    // Calibrate: 10 batches each at 2:1 speed ratio.
    // step_count must increment to satisfy NCCL ack tracking.
    for i in 0..10 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i + 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: i + 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average());
    h.coord.trigger_averaging().unwrap();
    for rx in &h.control_rxs { while rx.try_recv().is_ok() {} }
    assert!(h.coord.is_calibrated());

    // After calibration, batch_counts ~ [20, 10].
    // Feed exactly those counts. With wall-time trigger this would NOT
    // fire (fast rank wall = 100ms, slow = 100ms, but batch counts would
    // differ). With batch-count trigger it fires immediately.
    let counts = h.coord.el_che.batch_counts();
    for step0 in 11..(11 + counts[0]) {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: step0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    for step1 in 11..(11 + counts[1]) {
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: step1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    assert!(h.coord.should_average(), "async triggers on batch counts, not wall time");
}

#[test]
fn test_coordinator_trigger_nccl() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // Feed timing and trigger
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    h.coord.trigger_averaging().unwrap();

    // Workers should receive SyncNow
    for rx in &h.control_rxs {
        match rx.recv().unwrap() {
            ControlMsg::SyncNow => {}
            other => panic!("expected SyncNow, got {:?}", std::mem::discriminant(&other)),
        }
    }

    // Version bumped, steps reset
    assert_eq!(h.coord.version(), 1);
    assert_eq!(h.coord.steps_since_avg(), &[0, 0]);
}

#[test]
fn test_coordinator_trigger_cpu_averaging() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Cpu);
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Feed timing
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();

    // trigger_averaging now returns immediately (enters Collecting state)
    h.coord.trigger_averaging().unwrap();

    // Workers should receive RequestParams + Throttle (Sync policy blocks
    // workers during CPU averaging to prevent training with stale params).
    for rx in &h.control_rxs {
        match rx.recv().unwrap() {
            ControlMsg::RequestParams => {}
            other => panic!("expected RequestParams, got {:?}", std::mem::discriminant(&other)),
        }
        match rx.recv().unwrap() {
            ControlMsg::Throttle => {}
            other => panic!("expected Throttle, got {:?}", std::mem::discriminant(&other)),
        }
    }

    // Send snapshots (simulating workers responding)
    h.param_tx.send(ParamSnapshot {
        rank: 0,
        params: vec![Tensor::ones(&[2, 3], opts).unwrap()],
        buffers: vec![],
        batch_count: 10,
    }).unwrap();
    h.param_tx.send(ParamSnapshot {
        rank: 1,
        params: vec![Tensor::full(&[2, 3], 3.0, opts).unwrap()],
        buffers: vec![],
        batch_count: 10,
    }).unwrap();

    // Poll until the state machine completes (Collecting -> Computing -> Idle)
    for _ in 0..100 {
        h.coord.poll_cpu_averaging().unwrap();
        if h.coord.version() > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    assert_eq!(h.coord.version(), 1);

    // Workers should receive Update (Throttle handler dispatches it)
    for rx in &h.control_rxs {
        match rx.recv().unwrap() {
            ControlMsg::Update(avg) => {
                // Weighted average of 1.0 and 3.0 with equal batch counts = 2.0
                let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
                let expected = 2.0 * 6.0; // 2.0 * (2*3 elements)
                assert!((sum - expected).abs() < 1e-4,
                    "expected sum={expected}, got {sum}");
                assert_eq!(avg.version, 1);
            }
            other => panic!("expected Update, got {:?}", std::mem::discriminant(&other)),
        }
    }
}

#[test]
fn test_coordinator_average_params_weighted() {
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    // Rank 0: all 1.0, did 1 batch
    // Rank 1: all 5.0, did 3 batches
    // Weighted avg: (1*1.0 + 3*5.0) / (1+3) = 16/4 = 4.0
    let snapshots = vec![
        ParamSnapshot {
            rank: 0,
            params: vec![Tensor::ones(&[4], opts).unwrap()],
            buffers: vec![],
            batch_count: 1,
        },
        ParamSnapshot {
            rank: 1,
            params: vec![Tensor::full(&[4], 5.0, opts).unwrap()],
            buffers: vec![],
            batch_count: 3,
        },
    ];

    let avg = Coordinator::average_params(&snapshots, 42).unwrap();
    assert_eq!(avg.version, 42);
    assert_eq!(avg.params.len(), 1);

    // Each element should be (1*1.0 + 3*5.0) / (1+3) = 4.0
    let sum: f64 = avg.params[0].sum().unwrap().item().unwrap();
    let expected = 4.0 * 4.0; // 4.0 per element * 4 elements
    assert!((sum - expected).abs() < 1e-4, "expected sum={expected}, got {sum}");
}

#[test]
fn test_coordinator_tick_sync_flow() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    // No steps yet: tick should not trigger
    let metrics = h.coord.tick().unwrap();
    assert!(metrics.is_empty());
    assert_eq!(h.coord.version(), 0);

    // Feed steps from both ranks
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 10.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 20.0, step_count: 1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();

    // Tick: should trigger averaging
    let metrics = h.coord.tick().unwrap();
    assert!(metrics.is_empty());
    assert_eq!(h.coord.version(), 1);

    // Workers got SyncNow
    for rx in &h.control_rxs {
        assert!(matches!(rx.recv().unwrap(), ControlMsg::SyncNow));
    }
}

#[test]
fn test_coordinator_drain_metrics() {
    let mut h = make_coord_harness(2, ApplyPolicy::Sync, AverageBackend::Nccl);

    h.metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 1, avg_loss: 0.3, batches_processed: 50, epoch_ms: 2000.0, share_complete_ms: 2000.0, compute_only_ms: 2000.0, data_starve_ms: 0.0,
        samples_processed: 1600, scalars: HashMap::new(),
    }).unwrap();

    let metrics = h.coord.drain_metrics();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].rank, 0);
    assert_eq!(metrics[0].epoch, 1);
}

/// metrics_fn fires once per epoch, after all ranks report.
/// Both metrics_fn and the next_metrics queue receive the same EpochMetrics.
#[test]
fn test_coordinator_metrics_fn_fires_per_epoch() {
    use std::sync::Arc;
    use std::sync::Mutex;

    let (timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (param_tx, param_rx) = mpsc::channel();
    let _ = (timing_tx, param_tx);

    let mut control_txs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..2 {
        let (tx, _rx) = mpsc::channel();
        control_txs.push(tx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let captured: Arc<Mutex<Vec<super::EpochMetrics>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_cb = Arc::clone(&captured);
    let metrics_fn: super::MetricsFn = Arc::new(move |m: &super::EpochMetrics| {
        captured_cb.lock().unwrap().push(m.clone());
        Ok(())
    });

    let (queue_tx, queue_rx) = mpsc::channel();

    let el_che = ElChe::new(2, 10);
    let mut coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Nccl,
        2, 10000, el_che,
    )
    .epoch_metrics_tx(queue_tx)
    .metrics_fn(metrics_fn)
    .num_epochs(2)
    .build();

    // Both ranks report epoch 0 -> aggregator fires.
    metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 60, epoch_ms: 1000.0, share_complete_ms: 1000.0, compute_only_ms: 1000.0, data_starve_ms: 0.0,
        samples_processed: 1920,
        scalars: [("loss".to_string(), (3.0, 3_usize))].into(),
    }).unwrap();
    metrics_tx.send(MetricsMsg {
        rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 40, epoch_ms: 1200.0, share_complete_ms: 1200.0, compute_only_ms: 1200.0, data_starve_ms: 0.0,
        samples_processed: 1280,
        scalars: [("loss".to_string(), (4.0, 2_usize))].into(),
    }).unwrap();

    coord.drain_metrics();

    let cap = captured.lock().unwrap();
    assert_eq!(cap.len(), 1, "metrics_fn should fire exactly once for epoch 0");
    assert_eq!(cap[0].epoch, 0);
    // Batch-weighted: (0.5*60 + 0.7*40) / 100 = 0.58
    assert!((cap[0].avg_loss - 0.58).abs() < 1e-9);
    assert_eq!(cap[0].per_rank.len(), 2);

    // Same metric also reached the next_metrics queue.
    let queued = queue_rx.try_recv().expect("queue should have received the metric");
    assert_eq!(queued.epoch, 0);
    assert!((queued.avg_loss - 0.58).abs() < 1e-9);
}

/// Single-GPU fallback (run_single via Trainer::builder when no CUDA is
/// available, or when only one device is present): metrics_fn fires
/// per-epoch and next_metrics() drains the queued metrics afterwards.
/// This is the contract test for transparent 1-or-N GPU observability.
#[test]
fn test_run_single_metrics_fn_and_next_metrics() {
    use std::sync::Arc;
    use std::sync::Mutex;
    use crate::distributed::Trainer;

    // Skip if CUDA is available with >= 2 devices: this test targets the
    // single-GPU fallback code path. Single CUDA still hits run_single, so
    // either pure-CPU or single-CUDA environments exercise the same path.
    if crate::tensor::usable_cuda_devices().len() >= 2 {
        return;
    }

    let dataset: Arc<dyn crate::data::BatchDataSet> = Arc::new(TestDataset { n: 16 });

    let captured: Arc<Mutex<Vec<EpochMetrics>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_cb = Arc::clone(&captured);

    let handle = Trainer::builder(
        |d| Linear::on_device(4, 2, d),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(dataset)
    .batch_size(4)
    .num_epochs(3)
    .metrics_fn(move |m| {
        captured_cb.lock().unwrap().push(m.clone());
        Ok(())
    })
    .run()
    .unwrap();

    // metrics_fn fired per epoch as run_single progressed.
    let cap = captured.lock().unwrap();
    assert_eq!(cap.len(), 3, "metrics_fn should fire 3 times for 3 epochs");
    for (i, m) in cap.iter().enumerate() {
        assert_eq!(m.epoch, i);
        assert_eq!(m.per_rank.len(), 1, "single-GPU = single rank");
        assert!((m.per_rank_batch_share[0] - 1.0).abs() < 1e-9,
            "single rank gets 100% of batches");
    }
    drop(cap);

    // next_metrics() drains the queued metrics back-to-back, then None.
    let mut polled = Vec::new();
    while let Some(m) = handle.next_metrics() {
        polled.push(m);
    }
    assert_eq!(polled.len(), 3, "all 3 epochs should be queued");
    for (i, m) in polled.iter().enumerate() {
        assert_eq!(m.epoch, i);
    }

    let _ = handle.join().unwrap();
}

/// metrics_fn errors are logged but do not stop training: subsequent epochs
/// still aggregate and the callback fires again.
#[test]
fn test_coordinator_metrics_fn_error_continues_training() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (_timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..2 {
        let (tx, _rx) = mpsc::channel();
        control_txs.push(tx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_cb = Arc::clone(&calls);
    let metrics_fn: super::MetricsFn = Arc::new(move |_m: &super::EpochMetrics| {
        calls_cb.fetch_add(1, Ordering::Relaxed);
        Err(crate::tensor::TensorError::new("simulated callback failure"))
    });

    let el_che = ElChe::new(2, 10);
    let mut coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Nccl,
        2, 10000, el_che,
    )
    .metrics_fn(metrics_fn)
    .num_epochs(3)
    .build();

    // Fire two epochs; both should invoke the callback despite the error.
    for epoch in [0_usize, 1_usize] {
        metrics_tx.send(MetricsMsg {
            rank: 0, epoch, avg_loss: 0.5, batches_processed: 50, epoch_ms: 1000.0, share_complete_ms: 1000.0, compute_only_ms: 1000.0, data_starve_ms: 0.0,
            samples_processed: 1600, scalars: HashMap::new(),
        }).unwrap();
        metrics_tx.send(MetricsMsg {
            rank: 1, epoch, avg_loss: 0.5, batches_processed: 50, epoch_ms: 1000.0, share_complete_ms: 1000.0, compute_only_ms: 1000.0, data_starve_ms: 0.0,
            samples_processed: 1600, scalars: HashMap::new(),
        }).unwrap();
        coord.drain_metrics();
    }

    assert_eq!(calls.load(Ordering::Relaxed), 2,
        "metrics_fn should fire on every epoch even when it returns Err");
}

#[test]
fn test_coordinator_compute_partition_sizes() {
    let h = make_coord_harness(2, ApplyPolicy::Cadence, AverageBackend::Nccl);

    // Before calibration, partition sizes should be equal
    let sizes = h.coord.compute_partition_sizes();
    assert_eq!(sizes.len(), 2);
    assert_eq!(sizes[0], 5000); // 10000 / 2
    assert_eq!(sizes[1], 5000);
}

#[test]
fn test_divergence_correction_nudges_anchor_down() {
    // Rising divergence trend should suppress overshoot growth (unified guard).
    // The old single-shot NudgeDown behavior is replaced by trend detection.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);

    // Calibrate first so we have a stable anchor baseline.
    let steps = vec![10; 2];
    let wall_ms = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps, &wall_ms, None);
    let overshoot_before = h.coord.max_overshoot;

    // 3 intervals with rising divergence -> trigger SuppressGrowth.
    for i in 0..3 {
        let div = 0.10 + i as f64 * 0.05; // 0.10, 0.15, 0.20
        h.coord.finish_averaging_cpu(0.0, &[10, 10], &[100.0, 100.0],
            Some(super::convergence::DivergenceReport {
                deltas: vec![div, div],
                pre_norms: None,
                post_norm: None,
            }));
    }

    // Overshoot should NOT have grown on the 3rd interval (SuppressGrowth).
    // First 2 intervals grew normally, 3rd was suppressed.
    assert!(h.coord.max_overshoot <= overshoot_before + 2,
        "3rd interval should suppress overshoot growth, got {}", h.coord.max_overshoot);
}

#[test]
fn test_divergence_below_threshold_relaxes_anchor() {
    // Low divergence in async mode should relax the anchor upward by 1
    // (symmetric upward path to NudgeDown on stable convergence). The
    // convergence guard returns Stable, finish_averaging_cpu calls
    // ElChe::relax_anchor_up which grows anchor toward max_anchor.
    //
    // Relax-up is opt-in (default off) since 2026-05-01. This test
    // explicitly opts in to exercise the relax-up code path.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);
    h.coord.elche_relax_up = true;

    // Calibrate with zero sync_ms.
    let steps = vec![10; 2];
    let wall_ms = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps, &wall_ms, None);
    let anchor_after_calibration = h.coord.el_che.anchor();

    // Apply with low divergence.
    let steps2 = vec![10; 2];
    let wall_ms2 = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps2, &wall_ms2, Some(super::convergence::DivergenceReport {
        deltas: vec![0.01, 0.01],
        pre_norms: None,
        post_norm: None,
    }));

    // Divergence 0.01 < threshold: Stable. relax_anchor_up grows anchor by 1.
    assert_eq!(h.coord.el_che.anchor(), anchor_after_calibration + 1);
}

#[test]
fn test_relax_up_default_off_holds_anchor_on_stable() {
    // Default: relax_up disabled. On Stable convergence verdict the anchor
    // must NOT grow via relax_anchor_up; it stays where the overhead-based
    // auto-tune in el_che.report_timing left it.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);
    assert!(!h.coord.elche_relax_up, "default should be opt-out (off)");

    let steps = vec![10; 2];
    let wall_ms = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps, &wall_ms, None);
    let anchor_after_calibration = h.coord.el_che.anchor();

    let steps2 = vec![10; 2];
    let wall_ms2 = vec![100.0; 2];
    h.coord.finish_averaging_cpu(0.0, &steps2, &wall_ms2, Some(super::convergence::DivergenceReport {
        deltas: vec![0.01, 0.01],
        pre_norms: None,
        post_norm: None,
    }));

    assert_eq!(
        h.coord.el_che.anchor(), anchor_after_calibration,
        "relax_up default-off must hold anchor on Stable verdict"
    );
}

// -----------------------------------------------------------------------
// Throttle (max_batch_diff) tests
// -----------------------------------------------------------------------

fn make_throttle_harness(
    n: usize,
    max_batch_diff: usize,
) -> CoordTestHarness {
    let (timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (param_tx, param_rx) = mpsc::channel();

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

    let el_che = ElChe::new(n, 10).with_max_batch_diff(max_batch_diff);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Async, AverageBackend::Cpu,
        n, 10000, el_che,
    ).build();

    CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
}

#[test]
fn test_throttle_sends_when_diff_exceeded() {
    let mut h = make_throttle_harness(2, 3);

    // Rank 0 is 5 steps ahead, rank 1 at 0 -> diff = 5 > 3
    for i in 0..5 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    h.coord.check_throttle();

    // Rank 0 should receive Throttle
    match h.control_rxs[0].try_recv() {
        Ok(ControlMsg::Throttle) => {}
        _ => panic!("expected Throttle for rank 0"),
    }

    // Rank 1 should NOT receive Throttle
    assert!(h.control_rxs[1].try_recv().is_err(), "rank 1 should not be throttled");
}

#[test]
fn test_throttle_no_send_within_limit() {
    let mut h = make_throttle_harness(2, 5);

    // Rank 0 is 3 steps ahead, rank 1 at 0 -> diff = 3 <= 5
    for i in 0..3 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    h.coord.check_throttle();

    // No throttle for either rank
    assert!(h.control_rxs[0].try_recv().is_err());
    assert!(h.control_rxs[1].try_recv().is_err());
}

#[test]
fn test_throttle_zero_is_strict_lockstep() {
    let mut h = make_throttle_harness(2, 0);

    // Rank 0 does 1 batch, rank 1 does 0 -> diff = 1 > 0
    h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    h.coord.drain_timing();
    h.coord.check_throttle();

    // Rank 0 throttled immediately
    match h.control_rxs[0].try_recv() {
        Ok(ControlMsg::Throttle) => {}
        _ => panic!("expected Throttle for rank 0"),
    }
}

#[test]
fn test_throttle_skipped_for_nccl() {
    // NCCL cadence uses AllReduce as its coordination mechanism.
    // Throttle must be skipped to prevent deadlock when one rank is
    // idle (between epochs) and the other gets throttled waiting for
    // a SyncNow that can never fire.
    let (timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
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

    // NCCL backend with max_batch_diff = 3.
    let el_che = ElChe::new(2, 10).with_max_batch_diff(3);
    let mut coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Cadence, AverageBackend::Nccl,
        2, 10000, el_che,
    ).build();

    // Rank 0 is 10 steps ahead (would trigger throttle with CPU backend).
    for i in 0..10 {
        timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    coord.drain_timing();
    coord.check_throttle();

    // No throttle for NCCL -- cadence AllReduce handles coordination.
    assert!(control_rxs[0].try_recv().is_err(),
        "NCCL backend must not throttle (AllReduce is the coordination mechanism)");
}

#[test]
fn test_throttle_disabled_when_none() {
    // Default harness has no max_batch_diff
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Nccl);

    // Rank 0 far ahead
    for i in 0..50 {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: i, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    }
    h.coord.drain_timing();
    h.coord.check_throttle();

    // No throttle (feature disabled)
    assert!(h.control_rxs[0].try_recv().is_err());
}

#[test]
fn test_throttle_worker_unblocks_on_sync_now() {
    // Simulate: worker receives Throttle, then SyncNow unblocks it.
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::Throttle).unwrap();
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();

    // handle_control processes Throttle (blocks on recv), then
    // SyncNow arrives and unblocks it.
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown, "should not shutdown");
}

#[test]
fn test_throttle_worker_unblocks_on_shutdown() {
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::Throttle).unwrap();
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();

    let shutdown = worker.handle_control().unwrap();
    assert!(shutdown, "should signal shutdown");
}

#[test]
fn test_async_ddp_config_max_batch_diff() {
    let config = DdpRunConfig::new().with_max_batch_diff(5);
    assert_eq!(config.max_batch_diff, Some(5));

    let config2 = DdpRunConfig::new();
    assert_eq!(config2.max_batch_diff, None);
}

// -----------------------------------------------------------------------
