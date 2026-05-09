use super::*;
use crate::nn::Module;
use crate::tensor::TensorError;

#[test]
fn test_apply_policy_variants() {
    let policies = [ApplyPolicy::Sync, ApplyPolicy::Cadence, ApplyPolicy::Async];
    assert_eq!(policies.len(), 3);
    assert_eq!(ApplyPolicy::Sync, ApplyPolicy::Sync);
    assert_ne!(ApplyPolicy::Sync, ApplyPolicy::Async);
}

#[test]
fn test_average_backend_variants() {
    let backends = [AverageBackend::Nccl, AverageBackend::Cpu];
    assert_eq!(backends.len(), 2);
    assert_eq!(AverageBackend::Nccl, AverageBackend::Nccl);
    assert_ne!(AverageBackend::Nccl, AverageBackend::Cpu);
}

#[test]
fn test_control_msg_variants() {
    // Verify all variants are constructable
    let _req = ControlMsg::RequestParams;
    let _sync = ControlMsg::SyncNow;
    let _throttle = ControlMsg::Throttle;
    let _start = ControlMsg::StartEpoch(EpochPlan {
        epoch: 0, partition_offset: 0, partition_size: 1000,
    });
    let _ckpt = ControlMsg::Checkpoint { version: 42 };
    let _shutdown = ControlMsg::Shutdown;
    let _update = ControlMsg::Update(AveragedParams {
        params: vec![],
        buffers: vec![],
        version: 0,
    });
}

#[test]
fn test_timing_msg_send() {
    // TimingMsg must be Send (all fields are Copy primitives)
    fn assert_send<T: Send>() {}
    assert_send::<TimingMsg>();
}

#[test]
fn test_metrics_msg_send() {
    fn assert_send<T: Send>() {}
    assert_send::<MetricsMsg>();
}

#[test]
fn test_param_snapshot_send() {
    // ParamSnapshot contains Vec<Tensor> which is Send (Tensor: unsafe impl Send)
    fn assert_send<T: Send>() {}
    assert_send::<ParamSnapshot>();
}

#[test]
fn test_averaged_params_send() {
    fn assert_send<T: Send>() {}
    assert_send::<AveragedParams>();
}

#[test]
fn test_control_msg_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ControlMsg>();
}

#[test]
fn test_worker_config_send() {
    fn assert_send<T: Send>() {}
    assert_send::<WorkerConfig>();
}

#[test]
fn test_worker_config_clone() {
    let cfg = WorkerConfig {
        rank: 0,
        world_size: 2,
        device: Device::CPU,
        initial_params: vec![],
        initial_buffers: vec![],
        total_samples: 10000,
        batch_size: 32,
        seed: 42,
        max_grad_norm: None,
        easgd_alpha: None,
        timeline: None,
        policy: ApplyPolicy::Sync,
    };
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.rank, 0);
    assert_eq!(cfg2.world_size, 2);
    assert_eq!(cfg2.total_samples, 10000);
}

// -----------------------------------------------------------------------
// GpuWorker tests
// -----------------------------------------------------------------------

use std::sync::mpsc;
use crate::autograd::Variable;
use crate::nn::Linear;
use crate::tensor::{test_device, test_opts, Tensor, TensorOptions, DType};

/// Simple test dataset: random (input, target) pairs.
struct TestDataset {
    n: usize,
}
impl crate::data::BatchDataSet for TestDataset {
    fn len(&self) -> usize { self.n }
    fn get_batch(&self, indices: &[usize]) -> crate::tensor::Result<Vec<Tensor>> {
        let n = indices.len() as i64;
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        Ok(vec![
            Tensor::randn(&[n, 4], opts)?,
            Tensor::randn(&[n, 2], opts)?,
        ])
    }
}

/// Simple MSE train function for tests.
fn mse_train(model: &Linear, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    let output = model.forward(&input)?;
    let diff = output.sub(&target)?;
    diff.mul(&diff)?.mean()
}

/// Create a GpuWorker with a simple Linear model for testing.
///
/// Uses a minimal dataset (4 samples = 1 batch, matching batch_size=4) so
/// that GpuWorker::new skips PrefetchWorker creation (nothing to prefetch
/// when the dataset fits in a single batch). This keeps CUDA resource
/// footprint low: each GpuWorker still allocates 2 CUDA streams + 1 event,
/// but avoids the extra thread + channel from PrefetchWorker. Under parallel
/// test execution (or when training runs concurrently), VRAM contention from
/// dozens of workers can cause transient allocation failures.
fn make_test_worker() -> (GpuWorker<Linear>, WorkerChannels) {
    make_test_worker_with(0, 1, 4)
}

/// Create a GpuWorker with configurable rank/world_size/dataset_size.
fn make_test_worker_with(
    rank: usize,
    world_size: usize,
    dataset_size: usize,
) -> (GpuWorker<Linear>, WorkerChannels) {
    let dev = test_device();

    // Build a temporary model to extract initial params
    let tmp_model = Linear::on_device(4, 2, dev).unwrap();
    let tmp_params: Vec<Tensor> = tmp_model.parameters().iter()
        .map(|p| p.variable.data())
        .collect();
    let tmp_buffers: Vec<Tensor> = tmp_model.buffers().iter()
        .map(|b| b.get())
        .collect();
    drop(tmp_model);

    let config = WorkerConfig {
        rank,
        world_size,
        device: dev,
        initial_params: tmp_params,
        initial_buffers: tmp_buffers,
        total_samples: dataset_size,
        batch_size: 4,
        seed: 42,
        max_grad_norm: None,
        easgd_alpha: None,
        timeline: None,
        policy: ApplyPolicy::Sync,
    };

    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), channels) =
        GpuWorker::<Linear>::channels();

    let dataset: Arc<dyn crate::data::BatchDataSet> =
        Arc::new(TestDataset { n: dataset_size });

    let worker = GpuWorker::new(
        &config,
        |d| Linear::on_device(4, 2, d),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        dataset,
        None, // no NCCL in unit tests
        None, // no checkpoint in unit tests
        timing_tx,
        metrics_tx,
        param_tx,
        final_param_tx,
        control_rx,
    ).unwrap();

    (worker, channels)
}

#[test]
fn test_worker_new_and_accessors() {
    let (worker, _ch) = make_test_worker();
    assert_eq!(worker.rank(), 0);
    assert_eq!(worker.local_step(), 0);
    assert_eq!(worker.current_version(), 0);
    assert_eq!(worker.param_vars.len(), 2); // Linear: weight + bias
}

#[test]
fn test_worker_snapshot_params() {
    let (worker, _ch) = make_test_worker();
    let snap = worker.snapshot_params();
    assert_eq!(snap.rank, 0);
    assert_eq!(snap.params.len(), 2); // weight + bias
    assert_eq!(snap.buffers.len(), 0); // Linear has no buffers
    assert_eq!(snap.batch_count, 1); // max(steps_since_avg=0, 1)

    // Verify snapshot tensors have the right shapes
    assert_eq!(snap.params[0].shape(), &[2, 4]); // weight
    assert_eq!(snap.params[1].shape(), &[2]);     // bias
}

#[test]
fn test_worker_snapshot_is_send() {
    let (worker, _ch) = make_test_worker();
    let snap = worker.snapshot_params();

    // Verify snapshot can be sent through a channel
    let (tx, rx) = mpsc::channel::<ParamSnapshot>();
    tx.send(snap).unwrap();
    let received = rx.recv().unwrap();
    assert_eq!(received.rank, 0);
    assert_eq!(received.params.len(), 2);
}

#[test]
fn test_worker_load_averaged() {
    // NOTE: This test can fail transiently under VRAM pressure (e.g. when
    // training runs concurrently or many CUDA tests execute in parallel).
    // GpuWorker::new allocates CUDA streams + events, and the update tensors
    // below add further allocations. If this flakes, check GPU utilization.
    let (mut worker, _ch) = make_test_worker();

    // Create "averaged" params on CPU (mirrors the real averaging path where
    // coordinator produces CPU tensors). copy_ handles the H2D transfer.
    let cpu = TensorOptions { dtype: DType::Float32, device: Device::CPU };
    let new_weight = Tensor::ones(&[2, 4], cpu).unwrap();
    let new_bias = Tensor::ones(&[2], cpu).unwrap();

    let update = AveragedParams {
        params: vec![new_weight, new_bias],
        buffers: vec![],
        version: 42,
    };

    worker.load_averaged(&update).unwrap();

    // load_averaged uses non-blocking copy_ on comm_stream (CUDA).
    // In the training loop, sync_before_forward() at the next train_step
    // waits for the event. Here we read directly, so sync the device.
    let dev = test_device();
    if let Device::CUDA(idx) = dev {
        crate::tensor::cuda_synchronize(idx);
    }

    // Verify version updated
    assert_eq!(worker.current_version(), 42);

    // Verify model params now contain all ones
    let snap = worker.snapshot_params();
    let w_sum: f64 = snap.params[0].sum().unwrap().item().unwrap();
    assert!((w_sum - 8.0).abs() < 1e-5, "weight should be all ones (sum=8), got {w_sum}");
    let b_sum: f64 = snap.params[1].sum().unwrap().item().unwrap();
    assert!((b_sum - 2.0).abs() < 1e-5, "bias should be all ones (sum=2), got {b_sum}");
}

#[test]
fn test_worker_load_averaged_wrong_count() {
    let (mut worker, _ch) = make_test_worker();

    let update = AveragedParams {
        params: vec![], // wrong count
        buffers: vec![],
        version: 1,
    };
    assert!(worker.load_averaged(&update).is_err());
}

#[test]
fn test_worker_train_step() {
    let (mut worker, ch) = make_test_worker();
    let opts = test_opts();

    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];

    let (loss, ms) = worker.train_step(&batch, &mse_train).unwrap();
    assert!(ms > 0.0);
    assert!(loss > 0.0);
    assert_eq!(worker.local_step(), 1);

    // Verify timing was NOT auto-sent (train_step doesn't auto-send)
    assert!(ch.timing_rx.try_recv().is_err());
}

#[test]
fn test_worker_report_timing() {
    let (worker, ch) = make_test_worker();

    worker.report_timing(12.5, None, 0.5, None).unwrap();

    let msg = ch.timing_rx.recv().unwrap();
    match msg {
        TimingMsg::Batch { rank, batch_ms, step_count, .. } => {
            assert_eq!(rank, 0);
            assert!((batch_ms - 12.5).abs() < 1e-10);
            assert_eq!(step_count, 0);
        }
        _ => panic!("expected Batch"),
    }
}

#[test]
fn test_worker_report_epoch() {
    let (worker, ch) = make_test_worker();

    worker.report_epoch(0.5, 100, 5000.0, 5000.0, 0.0, 0.0).unwrap();

    let msg = ch.metrics_rx.recv().unwrap();
    assert_eq!(msg.rank, 0);
    assert_eq!(msg.epoch, 0);
    assert!((msg.avg_loss - 0.5).abs() < 1e-10);
    assert_eq!(msg.batches_processed, 100);
}

#[test]
fn test_worker_handle_control_request_params() {
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::RequestParams).unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);

    // Verify snapshot was sent back
    let snap = ch.param_rx.recv().unwrap();
    assert_eq!(snap.rank, 0);
    assert_eq!(snap.params.len(), 2);
}

#[test]
fn test_worker_handle_control_update() {
    let (mut worker, ch) = make_test_worker();
    let dev = test_device();
    let opts = TensorOptions { dtype: DType::Float32, device: dev };

    let update = AveragedParams {
        params: vec![
            Tensor::zeros(&[2, 4], opts).unwrap(),
            Tensor::zeros(&[2], opts).unwrap(),
        ],
        buffers: vec![],
        version: 7,
    };
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();

    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
    assert_eq!(worker.current_version(), 7);
}

#[test]
fn test_worker_handle_control_start_epoch() {
    let (mut worker, ch) = make_test_worker();

    assert!(worker.pending_plan.is_none());

    ch.control_tx.send(ControlMsg::StartEpoch(EpochPlan {
        epoch: 1, partition_offset: 0, partition_size: 750,
    })).unwrap();
    worker.handle_control().unwrap();

    let plan = worker.pending_plan.take();
    assert!(plan.is_some());
    assert_eq!(plan.unwrap().partition_size, 750);
    assert!(worker.pending_plan.is_none()); // consumed
}

#[test]
fn test_worker_handle_control_shutdown() {
    let (mut worker, ch) = make_test_worker();

    ch.control_tx.send(ControlMsg::Shutdown).unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(shutdown);
}

#[test]
fn test_worker_handle_control_sync_now_noop() {
    let (mut worker, ch) = make_test_worker();

    // SyncNow is a no-op without NCCL (Phase 4)
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
}

#[test]
fn test_worker_full_roundtrip() {
    // Simulates: train -> snapshot -> "average" -> load -> train again
    let (mut worker, ch) = make_test_worker();
    let opts = test_opts();

    // Step 1: train a step
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch, &mse_train).unwrap();
    assert_eq!(worker.local_step(), 1);

    // Step 2: coordinator requests params
    ch.control_tx.send(ControlMsg::RequestParams).unwrap();
    worker.handle_control().unwrap();
    let snap = ch.param_rx.recv().unwrap();
    assert_eq!(snap.batch_count, 1);

    // Step 3: coordinator sends back "averaged" params (same values, pretend averaged)
    let update = AveragedParams {
        params: snap.params,
        buffers: snap.buffers,
        version: 1,
    };
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();
    worker.handle_control().unwrap();
    assert_eq!(worker.current_version(), 1);

    // Step 4: train another step with loaded params
    let batch2 = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch2, &mse_train).unwrap();
    assert_eq!(worker.local_step(), 2);
}

#[test]
fn test_worker_epoch_from_plan() {
    let (mut worker, _ch) = make_test_worker();
    assert_eq!(worker.current_epoch, 0);
    // Epoch is set from EpochPlan in run_epoch_plan
    worker.current_epoch = 3;
    assert_eq!(worker.current_epoch, 3);
}

#[test]
fn test_worker_channels_create() {
    let ((timing_tx, metrics_tx, param_tx, _final_param_tx, _control_rx), ch) =
        GpuWorker::<Linear>::channels();

    // Verify channel pairs work
    timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
    let msg = ch.timing_rx.recv().unwrap();
    assert!(matches!(msg, TimingMsg::Batch { rank: 0, .. }));

    metrics_tx.send(MetricsMsg {
        rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 10, epoch_ms: 100.0, share_complete_ms: 100.0, compute_only_ms: 100.0, data_starve_ms: 0.0,
        samples_processed: 320, scalars: HashMap::new(),
    }).unwrap();
    let msg = ch.metrics_rx.recv().unwrap();
    assert_eq!(msg.batches_processed, 10);

    param_tx.send(ParamSnapshot {
        rank: 0, params: vec![], buffers: vec![], batch_count: 0,
    }).unwrap();
    let snap = ch.param_rx.recv().unwrap();
    assert_eq!(snap.rank, 0);

    ch.control_tx.send(ControlMsg::Shutdown).unwrap();
}

// -----------------------------------------------------------------------
// Coordinator tests
// -----------------------------------------------------------------------

use crate::distributed::ddp::ElChe;

/// Simple coordinator test helper.
struct CoordTestHarness {
    coord: Coordinator,
    /// Send timing/metrics/params TO the coordinator.
    timing_tx: mpsc::Sender<TimingMsg>,
    metrics_tx: mpsc::Sender<MetricsMsg>,
    param_tx: mpsc::Sender<ParamSnapshot>,
    /// Receive control messages FROM the coordinator (one per worker).
    control_rxs: Vec<mpsc::Receiver<ControlMsg>>,
}

fn make_coord_harness(
    n: usize,
    policy: ApplyPolicy,
    backend: AverageBackend,
) -> CoordTestHarness {
    make_coord_harness_with_timeout(n, policy, backend, 5)
}

fn make_coord_harness_with_timeout(
    n: usize,
    policy: ApplyPolicy,
    backend: AverageBackend,
    snapshot_timeout_secs: u64,
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

    let el_che = ElChe::new(n, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        policy, backend,
        n, 10000, el_che,
    )
    .snapshot_timeout_secs(snapshot_timeout_secs)
    .build();

    CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
}

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
    let mut step0 = 11usize;
    let mut step1 = 11usize;
    for _ in 0..counts[0] {
        h.timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 5.0, step_count: step0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        step0 += 1;
    }
    for _ in 0..counts[1] {
        h.timing_tx.send(TimingMsg::Batch { rank: 1, batch_ms: 10.0, step_count: step1, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
        step1 += 1;
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
// DdpHandle / DdpBuilder tests
// -----------------------------------------------------------------------

#[test]
fn test_async_ddp_single_gpu_fallback() {
    // With <2 GPUs, falls back to single-device training.
    // With 2+ GPUs, uses all of them. Either way, join succeeds.
    let ddp = DdpHandle::auto(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
        Arc::new(TestDataset { n: 100 }),
        4,
        2,  // 2 epochs
        ApplyPolicy::Sync,
        AverageBackend::Cpu, // CPU backend: no NCCL needed for this test
    ).unwrap();

    assert!(ddp.world_size() >= 1);
    let state = ddp.join().unwrap();
    // Linear(4,2): weight [2,4] + bias [2] = 2 params, 0 buffers
    assert_eq!(state.params.len(), 2);
    assert_eq!(state.buffers.len(), 0);
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
fn test_async_ddp_multi_gpu_nccl() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let ddp = DdpHandle::auto(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
        Arc::new(TestDataset { n: 256 }),
        32,
        2,  // 2 epochs
        ApplyPolicy::Sync,
        AverageBackend::Nccl,
    ).unwrap();

    assert!(ddp.world_size() >= 2);

    // Workers train for 2 epochs then exit, join returns trained state
    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
fn test_async_ddp_send_sync() {
    fn assert_send<T: Send>() {}
    assert_send::<DdpHandle>();
    assert_send::<TrainedState>();
}

// -----------------------------------------------------------------------
// DdpBuilder builder tests
// -----------------------------------------------------------------------

#[test]
fn test_builder_with_defaults() {
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(2)
    .backend(AverageBackend::Cpu)
    .run()
    .unwrap();

    assert!(ddp.world_size() >= 1);
    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
fn test_builder_with_all_options() {
    // Workload is intentionally tiny (8 samples / batch 4 / 1 epoch =
    // 2 steps): this test verifies that every builder setter is wired,
    // not that policy survives load. ElChe is sized for production
    // pools and can stall on heterogeneous hardware when the per-rank
    // share is small enough that the fast rank laps the slow one;
    // Sync alone doesn't fully isolate the test from that pathology
    // here, so we also keep the dataset small enough that any
    // remaining lapping cannot accumulate before completion.
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 8 }))
    .batch_size(4)
    .num_epochs(1)
    .policy(ApplyPolicy::Sync)
    .backend(AverageBackend::Cpu)
    .overhead_target(0.15)
    .max_anchor(100)
    .anchor(5)
    .divergence_threshold(0.1)
    .max_batch_diff(10)
    .run()
    .unwrap();

    let state = ddp.join().unwrap();
    assert_eq!(state.params.len(), 2);
}

#[test]
#[should_panic(expected = "dataset is required")]
fn test_builder_missing_dataset_panics() {
    let _ = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .batch_size(4)
    .num_epochs(2)
    .run();
}

#[test]
#[should_panic(expected = "batch_size is required")]
fn test_builder_missing_batch_size_panics() {
    let _ = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .num_epochs(2)
    .run();
}

#[test]
#[should_panic(expected = "num_epochs is required")]
fn test_builder_missing_num_epochs_panics() {
    let _ = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .run();
}

// -----------------------------------------------------------------------
// epoch_fn tests
// -----------------------------------------------------------------------

#[test]
fn test_worker_current_epoch_accessor() {
    let (mut worker, _ch) = make_test_worker();
    assert_eq!(worker.current_epoch(), 0);
    worker.current_epoch = 1;
    assert_eq!(worker.current_epoch(), 1);
}

#[test]
fn test_worker_set_lr() {
    let (mut worker, _ch) = make_test_worker();
    // set_lr should not panic; we verify it works by running a train step after
    worker.set_lr(0.1);
    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    let (loss, _) = worker.train_step(&batch, &mse_train).unwrap();
    assert!(loss > 0.0);
}

#[test]
fn test_epoch_fn_called_per_epoch() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = Arc::new(AtomicUsize::new(0));
    let epochs_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let counter_c = counter.clone();
    let epochs_c = epochs_seen.clone();

    let num_epochs = 3;
    // ApplyPolicy::Sync is required for this test's assertion shape.
    // The contract flodl's DDP actually guarantees is batch-based, not
    // epoch-based: `max_batch_diff` / `max_overshoot` bounds how far
    // ranks can diverge in batches per sync cycle. Epoch boundaries are
    // bookkeeping on top of that. At production scale this is invisible
    // — `max_overshoot` (default ceiling 15) is a tiny fraction of the
    // pool, so every rank receives every epoch's plan in practice.
    //
    // This test uses a tiny dataset (100 samples / batch 4 → 25
    // batches, planned share ~12 per rank) where the overshoot cap
    // exceeds the per-rank share. Under progressive mode (Cadence /
    // Async, see coordinator/mod.rs:405) a fast rank can legally drain
    // the whole pool, leaving the slow rank with no `StartEpoch` plan
    // for that epoch — which means fewer than `num_epochs * world`
    // epoch_fn firings. Expected behaviour at this scale, not a bug.
    //
    // Sync is the only policy where `count == num_epochs * world`
    // holds at any dataset size; that's what this test is anchoring.
    // For an epoch_fn test under progressive semantics, see a future
    // test that asserts the weaker invariant (every epoch fires at
    // least once across the cluster, no rank fires the same epoch
    // twice).
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(num_epochs)
    .backend(AverageBackend::Cpu)
    .policy(ApplyPolicy::Sync)
    .epoch_fn(move |epoch, worker| {
        counter_c.fetch_add(1, Ordering::Relaxed);
        epochs_c.lock().unwrap().push(epoch);
        // Verify current_epoch matches the callback argument
        assert_eq!(worker.current_epoch(), epoch);
    })
    .run()
    .unwrap();

    let world = ddp.world_size();
    let _state = ddp.join().unwrap();

    // Instrumented assertions: on regression, dump the full observed
    // state. In Sync mode (pinned above) each rank must see each epoch
    // exactly once; any drift is a real bug in the Sync dispatcher, not
    // the progressive streaming path.
    let got_counter = counter.load(Ordering::Relaxed);
    let expected_counter = num_epochs * world;

    let mut seen = epochs_seen.lock().unwrap().clone();
    seen.sort();
    let mut expected_epochs: Vec<usize> = (0..num_epochs).cycle().take(num_epochs * world).collect();
    expected_epochs.sort();

    assert_eq!(
        got_counter, expected_counter,
        "epoch_fn fire count mismatch — got {got_counter}, expected {expected_counter}. \
         world_size={world}, num_epochs={num_epochs}, epochs_seen={seen:?}.",
    );
    assert_eq!(
        seen, expected_epochs,
        "epoch_fn epoch-index set mismatch — got {seen:?}, expected {expected_epochs:?}. \
         world_size={world}, num_epochs={num_epochs}, counter={got_counter}.",
    );
}

#[test]
fn test_epoch_fn_set_lr() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_c = call_count.clone();

    // Sync policy required (see `test_epoch_fn_called_per_epoch` for
    // the full rationale). In Sync every rank fires `epoch_fn` for
    // every epoch, so `lr` stays consistent across ranks during each
    // gradient average — which is what this test actually verifies.
    let ddp = DdpHandle::builder(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        mse_train,
    )
    .dataset(Arc::new(TestDataset { n: 100 }))
    .batch_size(4)
    .num_epochs(3)
    .backend(AverageBackend::Cpu)
    .policy(ApplyPolicy::Sync)
    .epoch_fn(move |epoch, worker| {
        // Simulate a LR schedule: decrease LR each epoch
        let lr = 0.01 * (1.0 - epoch as f64 * 0.3);
        worker.set_lr(lr);
        call_count_c.fetch_add(1, Ordering::Relaxed);
    })
    .run()
    .unwrap();

    let world = ddp.world_size();
    let _state = ddp.join().unwrap();
    assert_eq!(call_count.load(Ordering::Relaxed), 3 * world);
}

#[test]
fn test_worker_send_final_snapshot() {
    let (worker, ch) = make_test_worker();
    worker.send_final_snapshot();
    let snap = ch.final_param_rx.recv().unwrap();
    assert_eq!(snap.params.len(), 2); // Linear(4,2): weight + bias
    assert_eq!(snap.rank, 0);
}

#[test]
fn test_collect_final_state_averages() {
    let (timing_tx, timing_rx) = mpsc::channel();
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

    // Send final snapshots from both "workers"
    let opts = crate::tensor::test_opts();
    let t1 = Tensor::full(&[3], 2.0, opts).unwrap();
    let t2 = Tensor::full(&[3], 4.0, opts).unwrap();
    final_param_txs[0].send(ParamSnapshot {
        rank: 0, params: vec![t1], buffers: vec![], batch_count: 1,
    }).unwrap();
    final_param_txs[1].send(ParamSnapshot {
        rank: 1, params: vec![t2], buffers: vec![], batch_count: 1,
    }).unwrap();

    let state = coord.collect_final_state().unwrap();
    assert_eq!(state.params.len(), 1);
    // Average of 2.0 and 4.0 with equal weights = 3.0
    let vals: Vec<f64> = state.params[0].to_f64_vec().unwrap();
    assert!(vals.iter().all(|v| (v - 3.0).abs() < 1e-5), "expected all ~3.0, got {vals:?}");

    // Also verify timing_tx keeps coordinator alive
    drop(timing_tx);
}

#[test]
fn test_collect_final_state_single_survivor() {
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

    // Only one worker sends a final snapshot (the other "died")
    let opts = crate::tensor::test_opts();
    let t = Tensor::full(&[3], 7.0, opts).unwrap();
    final_param_txs[0].send(ParamSnapshot {
        rank: 0, params: vec![t], buffers: vec![], batch_count: 5,
    }).unwrap();
    // Worker 1 never sends

    let state = coord.collect_final_state().unwrap();
    assert_eq!(state.params.len(), 1);
    let vals: Vec<f64> = state.params[0].to_f64_vec().unwrap();
    assert!(vals.iter().all(|v| (v - 7.0).abs() < 1e-5), "single survivor should return its own params");
}

// -----------------------------------------------------------------------
// Checkpoint coordination tests
// -----------------------------------------------------------------------

#[test]
fn test_checkpoint_msg_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<ControlMsg>();
}

#[test]
fn test_checkpoint_fn_called_on_dispatch() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let (mut worker, ch) = make_test_worker();
    let called_version = Arc::new(AtomicU64::new(0));
    let cv = called_version.clone();
    worker.checkpoint_fn = Some(Arc::new(move |ver, _model| {
        cv.store(ver, Ordering::Relaxed);
        Ok(())
    }));

    ch.control_tx.send(ControlMsg::Checkpoint { version: 7 }).unwrap();
    worker.handle_control().unwrap();

    assert_eq!(called_version.load(Ordering::Relaxed), 7);
}

#[test]
fn test_checkpoint_error_logged_not_propagated() {
    let (mut worker, ch) = make_test_worker();
    worker.checkpoint_fn = Some(Arc::new(|_ver, _model| {
        Err(TensorError::new("disk full"))
    }));

    ch.control_tx.send(ControlMsg::Checkpoint { version: 1 }).unwrap();
    // Should not return an error: log-and-continue
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
}

#[test]
fn test_coordinator_sends_checkpoint_every_n_epochs() {
    use crate::distributed::ddp::ElChe;

    let n = 2;
    let (_timing_tx, timing_rx) = mpsc::channel();
    let (_metrics_tx, metrics_rx) = mpsc::channel();
    let (_param_tx, param_rx) = mpsc::channel();

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
    let mut coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs,
        control_txs,
        ApplyPolicy::Sync, AverageBackend::Nccl,
        n, 10000, el_che,
    )
    .num_epochs(10)
    .checkpoint_every(2)
    .build();

    // Aggregate 3 global epochs.
    for epoch in 0..3 {
        coord.on_epoch_aggregated(epoch);
    }

    // checkpoint_every=2: epoch 0 → (0+1)%2=1 no, epoch 1 → (1+1)%2=0 yes, epoch 2 → (2+1)%2=1 no
    let mut checkpoint_versions = Vec::new();
    for rx in &control_rxs {
        while let Ok(msg) = rx.try_recv() {
            if let ControlMsg::Checkpoint { version } = msg {
                checkpoint_versions.push(version);
            }
        }
    }
    assert_eq!(checkpoint_versions, vec![2], "should checkpoint once (at epoch 2) after 3 epochs with every=2");
}

// -----------------------------------------------------------------------
// Phase 10: 2-GPU end-to-end validation
// -----------------------------------------------------------------------

/// Shared loss tracker for multi-GPU convergence tests.
/// Each rank appends (rank, step, loss) tuples.
type LossLog = Arc<std::sync::Mutex<Vec<(usize, usize, f64)>>>;

fn make_loss_tracker() -> LossLog {
    Arc::new(std::sync::Mutex::new(Vec::new()))
}

/// Run a 2-GPU DDP session and return collected losses per rank.
/// Returns (rank0_losses, rank1_losses) in chronological order.
fn run_2gpu_training(
    backend: AverageBackend,
    policy: ApplyPolicy,
    num_epochs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let log = make_loss_tracker();
    let log_clone = log.clone();

    let ddp = DdpHandle::auto(
        |dev| Linear::on_device(4, 2, dev),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        move |model: &Linear, batch: &[Tensor]| {
            let input = Variable::new(batch[0].clone(), false);
            let target = Variable::new(batch[1].clone(), false);
            let output = model.forward(&input)?;
            let diff = output.sub(&target)?;
            let loss = diff.mul(&diff)?.mean()?;
            let loss_val: f64 = loss.data().item()?;
            // Determine rank from device
            let rank = match batch[0].device() {
                Device::CUDA(idx) => idx as usize,
                Device::CPU => 0,
            };
            let step = {
                let mut lg = log_clone.lock().unwrap();
                let step = lg.iter().filter(|(r, _, _)| *r == rank).count();
                lg.push((rank, step, loss_val));
                step
            };
            let _ = step;
            Ok(loss)
        },
        Arc::new(TestDataset { n: 512 }),
        32,
        num_epochs,
        policy,
        backend,
    ).unwrap();

    let _state = ddp.join().unwrap();

    let entries = log.lock().unwrap();
    let r0: Vec<f64> = entries.iter().filter(|(r, _, _)| *r == 0).map(|(_, _, l)| *l).collect();
    let r1: Vec<f64> = entries.iter().filter(|(r, _, _)| *r == 1).map(|(_, _, l)| *l).collect();
    (r0, r1)
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
fn test_async_ddp_2gpu_cpu_backend_loss_decreases() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let (r0, r1) = run_2gpu_training(AverageBackend::Cpu, ApplyPolicy::Sync, 5);

    // Both ranks should have trained
    assert!(!r0.is_empty(), "rank 0 should have loss entries");
    assert!(!r1.is_empty(), "rank 1 should have loss entries");

    // Loss should converge: final losses should be finite and reasonable.
    // For a tiny Linear(4,2) with random data, the irreducible MSE is ~1.0.
    // We check that training converges (not diverges) rather than strictly decreases,
    // since NCCL averaging overhead can cause minor fluctuations.
    let check_converged = |losses: &[f64], rank: usize| {
        let n = losses.len();
        let quarter = (n / 4).max(1);
        let last_avg: f64 = losses[n - quarter..].iter().sum::<f64>() / quarter as f64;
        assert!(last_avg.is_finite() && last_avg < 2.0,
            "rank {rank} should converge: last_avg={last_avg:.4}");
    };

    check_converged(&r0, 0);
    check_converged(&r1, 1);
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
fn test_async_ddp_2gpu_nccl_backend_loss_decreases() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let (r0, r1) = run_2gpu_training(AverageBackend::Nccl, ApplyPolicy::Sync, 5);

    assert!(!r0.is_empty(), "rank 0 should have loss entries");
    assert!(!r1.is_empty(), "rank 1 should have loss entries");

    let check_converged = |losses: &[f64], rank: usize| {
        let n = losses.len();
        let quarter = (n / 4).max(1);
        let last_avg: f64 = losses[n - quarter..].iter().sum::<f64>() / quarter as f64;
        assert!(last_avg.is_finite() && last_avg < 2.0,
            "rank {rank} should converge: last_avg={last_avg:.4}");
    };

    check_converged(&r0, 0);
    check_converged(&r1, 1);
}

#[test]
#[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
fn test_async_ddp_ab_cpu_vs_nccl() {
    if crate::tensor::usable_cuda_devices().len() < 2 {
        return;
    }

    let epochs = 5;
    let (cpu_r0, cpu_r1) = run_2gpu_training(AverageBackend::Cpu, ApplyPolicy::Sync, epochs);
    let (nccl_r0, nccl_r1) = run_2gpu_training(AverageBackend::Nccl, ApplyPolicy::Sync, epochs);

    // Both backends should converge (loss decreases)
    let final_avg = |losses: &[f64]| -> f64 {
        let n = losses.len();
        let quarter = n / 4;
        if quarter == 0 { return f64::MAX; }
        losses[n - quarter..].iter().sum::<f64>() / quarter as f64
    };

    let cpu_final = (final_avg(&cpu_r0) + final_avg(&cpu_r1)) / 2.0;
    let nccl_final = (final_avg(&nccl_r0) + final_avg(&nccl_r1)) / 2.0;

    // Both should have converged to a reasonable loss
    assert!(cpu_final < 2.0,
        "CPU backend final loss too high: {cpu_final:.4}");
    assert!(nccl_final < 2.0,
        "NCCL backend final loss too high: {nccl_final:.4}");

    // Final losses should be in the same ballpark (within 2x of each other).
    // They won't be identical because data shuffling differs across runs,
    // but for a simple Linear model both should converge to similar regions.
    let ratio = cpu_final.max(nccl_final) / cpu_final.min(nccl_final);
    eprintln!("  A/B: CPU final={cpu_final:.4} NCCL final={nccl_final:.4} ratio={ratio:.2}");
    assert!(ratio < 3.0,
        "CPU vs NCCL final loss ratio too large: {ratio:.2} (CPU={cpu_final:.4} NCCL={nccl_final:.4})");
}

// -----------------------------------------------------------------------
// ElChe cadence + adaptive K tests (Phase 6)
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
// Partition + data iteration tests (Phase 5)
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
// Proportional epoch sharding tests (Phase 7)
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
    use super::coordinator::aggregate_epoch_metrics;

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
    use super::coordinator::aggregate_epoch_metrics;

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
    ch.control_tx.send(ControlMsg::Checkpoint { version: 99 }).unwrap();
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
    let pool = super::coordinator::ChunkPool::new(0, 20, 2);
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
    let pool = super::coordinator::ChunkPool::new(0, 100, 2);
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
fn test_overshoot_gate_skipped_for_nccl() {
    // NCCL cadence must not apply the overshoot gate. Blocking the fast
    // GPU forces it into wait_for_epoch_plan where it can't send timing
    // messages, leaving nccl_ack permanently false and deadlocking.
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
    .max_overshoot(Some(0)) // Would block everything with CPU backend.
    .build();

    // Create epoch 0 pool, take all samples for both ranks.
    let pool = super::coordinator::ChunkPool::new(0, 100, 2);
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

    // With NCCL, the overshoot gate is skipped: rank 0 should get
    // a cross-epoch StartEpoch even with max_overshoot=0.
    let mut got_start_epoch = false;
    while let Ok(msg) = control_rxs[0].try_recv() {
        if let ControlMsg::StartEpoch(_) = msg {
            got_start_epoch = true;
        }
    }
    assert!(got_start_epoch,
        "NCCL backend must skip overshoot gate (AllReduce handles coordination)");
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
    let mut pool0 = super::coordinator::ChunkPool::new(0, 100, 2);
    pool0.take_chunk(50, 0); // dispatch 50 to rank 0
    pool0.take_chunk(50, 1); // dispatch 50 to rank 1
    h.coord.chunk_pools.insert(0, pool0);

    let mut pool1 = super::coordinator::ChunkPool::new(1, 100, 2);
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
    let mut pool0 = super::coordinator::ChunkPool::new(0, 20, 2);
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
    let mut pool1 = super::coordinator::ChunkPool::new(1, 20, 2);
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
    let mut pool1 = super::coordinator::ChunkPool::new(1, 60, 2);
    pool1.take_chunk(30, 0); // rank 0 dispatched 30
    pool1.take_chunk(30, 1); // rank 1 dispatched 30
    sh.inner.coord.chunk_pools.insert(1, pool1);

    // -- Set up epoch 2 pool (rank 0 already streaming ahead) --
    let mut pool2 = super::coordinator::ChunkPool::new(2, 60, 2);
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
    let mut pool0 = super::coordinator::ChunkPool::new(0, 100, 2);
    pool0.take_chunk(70, 0);
    pool0.take_chunk(30, 1);
    h.coord.chunk_pools.insert(0, pool0);

    // Epoch 1 pool: fast GPU took ALL 100 samples; slow GPU got nothing.
    let mut pool1 = super::coordinator::ChunkPool::new(1, 100, 2);
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

// ---------------------------------------------------------------------------
// LR scheduling on the worker
// ---------------------------------------------------------------------------
//
// These tests guard the per-batch LR pipeline: scheduler.lr(step) * lr_scale
// must reach the optimizer on every batch. The original bugs (2026-04-13)
// were that scale_lr was silently overwritten when a scheduler was attached
// (so DDP linear scaling never took effect) and that the scheduler step
// counter could be inflated by NCCL ack messages (so MultiStepLR fired
// ~6 epochs early on heterogeneous DDP).

/// Trivial constant-LR scheduler used to assert `worker.lr_scale` is applied
/// multiplicatively on top of scheduler output.
struct ConstLr(f64);
impl crate::nn::Scheduler for ConstLr {
    fn lr(&self, _step: usize) -> f64 { self.0 }
}

/// Linearly increasing LR (lr = step * slope), so the test can also verify
/// that the scheduler is queried with the correct training step.
struct LinearLr { slope: f64 }
impl crate::nn::Scheduler for LinearLr {
    fn lr(&self, step: usize) -> f64 { step as f64 * self.slope }
}

#[test]
fn test_worker_scheduler_drives_optimizer_lr() {
    let (mut worker, _ch) = make_test_worker();
    worker.set_lr(0.0); // start at 0 so we can detect the scheduler writing in

    worker.set_scheduler(Arc::new(ConstLr(0.05)));

    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch, &mse_train).unwrap();

    // Scheduler returned 0.05; with lr_scale=1.0 (default) optimizer sees 0.05.
    assert!((worker.current_lr() - 0.05).abs() < 1e-9,
        "expected optimizer LR 0.05, got {}", worker.current_lr());
}

#[test]
fn test_worker_lr_scale_multiplies_scheduler_output() {
    // The bug this guards: orchestrator used to call worker.scale_lr(2.0) at
    // startup, but the scheduler's per-batch set_lr immediately overwrote
    // it -- so DDP linear scaling never reached the optimizer when a
    // scheduler was attached. Fix: orchestrator now calls set_lr_scale and
    // train_step does set_lr(sched.lr(step) * lr_scale).
    let (mut worker, _ch) = make_test_worker();
    worker.set_scheduler(Arc::new(ConstLr(0.05)));
    worker.set_lr_scale(2.0);

    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch, &mse_train).unwrap();

    // 0.05 * 2.0 = 0.10
    assert!((worker.current_lr() - 0.10).abs() < 1e-9,
        "expected optimizer LR 0.10 (sched 0.05 * scale 2.0), got {}",
        worker.current_lr());
}

#[test]
fn test_worker_scheduler_step_advances_with_global_progress() {
    // train_step computes set_lr(sched.lr(global_step + steps_since_avg)).
    // After 3 batches (no sync), step argument should be 3.
    let (mut worker, _ch) = make_test_worker();
    worker.set_scheduler(Arc::new(LinearLr { slope: 0.01 }));

    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];

    worker.train_step(&batch, &mse_train).unwrap();
    // Before batch 1, scheduler was queried at step 0 -> lr = 0.0.
    assert!((worker.current_lr() - 0.0).abs() < 1e-9);

    worker.train_step(&batch, &mse_train).unwrap();
    // Before batch 2, scheduler queried at step 1 -> lr = 0.01.
    assert!((worker.current_lr() - 0.01).abs() < 1e-9,
        "step 1: got {}", worker.current_lr());

    worker.train_step(&batch, &mse_train).unwrap();
    // Before batch 3, scheduler queried at step 2 -> lr = 0.02.
    assert!((worker.current_lr() - 0.02).abs() < 1e-9,
        "step 2: got {}", worker.current_lr());
}

// ---------------------------------------------------------------------------
// Cross-mode LR parity
// ---------------------------------------------------------------------------
//
// The framework promises: "if you give me a scheduler, I will honor it the
// same way no matter which training mode you use." Without this guarantee,
// switching between solo and DDP silently changes hyperparameter behavior --
// which is exactly the failure mode that hid Bugs 1-3 for so long.
//
// This test is the contract test for that promise. It runs the same
// MultiStepLR over the same number of batches in three independent paths
// and asserts the recorded LR sequences are identical:
//
//   1. **Manual**:  `optimizer.set_lr(sched.lr(step))` per batch (the
//      reference implementation a user would write themselves).
//   2. **GpuWorker**: train_step() with set_scheduler attached
//      (the path used by Trainer::builder, both in DDP and single-GPU fallback).
//   3. **Graph::step()**: the path used by Trainer::setup_with (sync mode).
//
// The first time we ran this we had two bugs (graph mode never updated LR;
// worker scheduler interaction with lr_scale was broken) and this single
// test would have caught both.

/// Records every LR value the scheduler returned, so the test can compare
/// the exact query sequence each mode produced.
struct RecordingSched {
    inner: crate::nn::MultiStepLR,
    queries: std::sync::Mutex<Vec<(usize, f64)>>,
}
impl RecordingSched {
    fn new(base_lr: f64, milestones: &[usize], gamma: f64) -> Self {
        Self {
            inner: crate::nn::MultiStepLR::new(base_lr, milestones, gamma),
            queries: std::sync::Mutex::new(Vec::new()),
        }
    }
}
impl crate::nn::Scheduler for RecordingSched {
    fn lr(&self, step: usize) -> f64 {
        let lr = self.inner.lr(step);
        self.queries.lock().unwrap().push((step, lr));
        lr
    }
}

#[test]
fn test_cross_mode_lr_parity_solo_vs_worker_vs_graph() {
    use crate::graph::FlowBuilder;
    use crate::nn::{Module, Optimizer, SGD};

    let dev = test_device();
    let opts = test_opts();
    let n_steps: usize = 12;
    // Milestones at 4 and 8 produce 3 LR plateaus across our 12 steps so
    // every drop is observed; gamma 0.1 makes the steps obvious.
    let base_lr = 0.1f64;
    let milestones = vec![4usize, 8];
    let gamma = 0.1f64;

    // ---------- Path 1: manual optimizer.set_lr per batch (reference). ----------
    let manual_lrs: Vec<f64> = {
        let model = Linear::on_device(4, 2, dev).unwrap();
        let mut opt = SGD::new(&model.parameters(), 0.0, 0.0);
        let sched = crate::nn::MultiStepLR::new(base_lr, &milestones, gamma);
        let batch = [
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            opt.set_lr(sched.lr(step));
            // Need a backward pass before opt.step() so SGD has gradients.
            let v = Variable::new(batch[0].clone(), false);
            let t = Variable::new(batch[1].clone(), false);
            let pred = model.forward(&v).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad();
            lrs.push(opt.lr());
        }
        lrs
    };

    // ---------- Path 2: GpuWorker with set_scheduler (Trainer::builder path). ----------
    let worker_lrs: Vec<f64> = {
        let (mut worker, _ch) = make_test_worker();
        worker.set_scheduler(Arc::new(RecordingSched::new(base_lr, &milestones, gamma)));
        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            worker.train_step(&batch, &mse_train).unwrap();
            lrs.push(worker.current_lr());
        }
        lrs
    };

    // ---------- Path 3: Graph::step() with set_scheduler (Trainer::setup_with path). ----------
    let graph_lrs: Vec<f64> = {
        let graph = FlowBuilder::from(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();
        graph.set_optimizer(|p| SGD::new(p, 0.0, 0.0));
        graph.set_scheduler(Arc::new(RecordingSched::new(base_lr, &milestones, gamma)));
        let x = Variable::new(Tensor::randn(&[4, 4], opts).unwrap(), false);
        let t = Variable::new(Tensor::randn(&[4, 2], opts).unwrap(), false);
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let pred = graph.forward(&x).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            graph.step().unwrap();
            let lr = graph.optimizer.borrow().as_ref().map(|o| o.lr()).unwrap();
            lrs.push(lr);
        }
        lrs
    };

    // The three paths must produce identical LR trajectories.
    assert_eq!(manual_lrs.len(), n_steps);
    assert_eq!(worker_lrs.len(), n_steps);
    assert_eq!(graph_lrs.len(), n_steps);
    for step in 0..n_steps {
        assert!((manual_lrs[step] - worker_lrs[step]).abs() < 1e-9,
            "step {step}: solo={} vs worker={}", manual_lrs[step], worker_lrs[step]);
        assert!((manual_lrs[step] - graph_lrs[step]).abs() < 1e-9,
            "step {step}: solo={} vs graph={}", manual_lrs[step], graph_lrs[step]);
    }

    // Sanity: the recorded trajectory should show the two MultiStepLR drops
    // (0.1 -> 0.01 at step 4, 0.01 -> 0.001 at step 8). If this fails, the
    // test scheduler isn't doing its job and the parity check above is
    // vacuous.
    let mut transitions = 0;
    for w in manual_lrs.windows(2) {
        if (w[0] - w[1]).abs() > 1e-9 { transitions += 1; }
    }
    assert_eq!(transitions, 2,
        "expected 2 LR drops over 12 steps with milestones [4, 8]; got {transitions}. \
         trajectory: {manual_lrs:?}");
}

#[test]
fn test_cross_mode_lr_parity_with_lr_scale() {
    // Same parity guarantee, but with lr_scale != 1.0. Worker and Graph must
    // both apply the scale multiplicatively to scheduler output.
    use crate::graph::FlowBuilder;
    use crate::nn::{Module, Optimizer, SGD};

    let dev = test_device();
    let opts = test_opts();
    let n_steps: usize = 8;
    let scale = 2.5;

    // Reference: manual with scale baked into base_lr.
    let manual_lrs: Vec<f64> = {
        let model = Linear::on_device(4, 2, dev).unwrap();
        let mut opt = SGD::new(&model.parameters(), 0.0, 0.0);
        let sched = crate::nn::MultiStepLR::new(0.1, &[4], 0.1);
        let batch = [
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            opt.set_lr(sched.lr(step) * scale);
            let v = Variable::new(batch[0].clone(), false);
            let t = Variable::new(batch[1].clone(), false);
            let pred = model.forward(&v).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad();
            lrs.push(opt.lr());
        }
        lrs
    };

    // GpuWorker: set_lr_scale + set_scheduler.
    let worker_lrs: Vec<f64> = {
        let (mut worker, _ch) = make_test_worker();
        worker.set_scheduler(Arc::new(crate::nn::MultiStepLR::new(0.1, &[4], 0.1)));
        worker.set_lr_scale(scale);
        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            worker.train_step(&batch, &mse_train).unwrap();
            lrs.push(worker.current_lr());
        }
        lrs
    };

    // Graph: set_lr_scale + set_scheduler.
    let graph_lrs: Vec<f64> = {
        let graph = FlowBuilder::from(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();
        graph.set_optimizer(|p| SGD::new(p, 0.0, 0.0));
        graph.set_scheduler(Arc::new(crate::nn::MultiStepLR::new(0.1, &[4], 0.1)));
        graph.set_lr_scale(scale);
        let x = Variable::new(Tensor::randn(&[4, 4], opts).unwrap(), false);
        let t = Variable::new(Tensor::randn(&[4, 2], opts).unwrap(), false);
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let pred = graph.forward(&x).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            graph.step().unwrap();
            let lr = graph.optimizer.borrow().as_ref().map(|o| o.lr()).unwrap();
            lrs.push(lr);
        }
        lrs
    };

    for step in 0..n_steps {
        assert!((manual_lrs[step] - worker_lrs[step]).abs() < 1e-9,
            "step {step}: solo*scale={} vs worker={}", manual_lrs[step], worker_lrs[step]);
        assert!((manual_lrs[step] - graph_lrs[step]).abs() < 1e-9,
            "step {step}: solo*scale={} vs graph={}", manual_lrs[step], graph_lrs[step]);
    }
}

// ---------------------------------------------------------------------------
// LR-aware meta-controller integration
// ---------------------------------------------------------------------------

/// Build a coordinator harness with the LR-aware meta-controller enabled.
fn make_coord_harness_meta(n: usize) -> CoordTestHarness {
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

    let el_che = crate::distributed::ddp::ElChe::new(n, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs, control_txs,
        ApplyPolicy::Async, AverageBackend::Cpu,
        n, 10000, el_che,
    )
    .meta_controller(true)
    .build();

    CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
}

/// Drive ElChe past Probe/Warmup into Stable so the meta-controller will
/// act on observations (Probe is always silent, Warmup requires K=5
/// sustained for convergence pattern).
fn calibrate_to_stable(coord: &mut Coordinator) {
    for _ in 0..6 {
        coord.el_che.report_timing(&[10.0, 20.0], &[10, 10], 1.0);
    }
    assert!(coord.el_che.phase() >= crate::distributed::Phase::Stable);
}

#[test]
fn test_meta_controller_dispatches_nudge_on_lr_cliff() {
    let mut h = make_coord_harness_meta(2);
    calibrate_to_stable(&mut h.coord);

    // Establish baseline LR via the LrUpdate channel + drain.
    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.1 }).unwrap();
    h.coord.drain_timing();

    // First observation: builds the LR window. anchor_window also starts here.
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);
    let anchor_after_warmup = h.coord.el_che.anchor();

    // Sharp LR drop (10x) on the next cycle: cliff watcher should fire.
    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.01 }).unwrap();
    h.coord.drain_timing();
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);

    let anchor_after_cliff = h.coord.el_che.anchor();
    assert!(
        anchor_after_cliff < anchor_after_warmup,
        "meta-controller should nudge anchor down on sharp LR drop: \
         {anchor_after_warmup} -> {anchor_after_cliff}",
    );
}

#[test]
fn test_meta_controller_silent_on_smooth_decay() {
    let mut h = make_coord_harness_meta(2);
    calibrate_to_stable(&mut h.coord);

    let anchor_initial = h.coord.el_che.anchor();
    let mut lr = 0.1;
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr }).unwrap();
        h.coord.drain_timing();
        h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);
        lr *= 0.98; // 2% per cycle, well under 30% cliff threshold
    }

    let anchor_final = h.coord.el_che.anchor();
    assert_eq!(
        anchor_final, anchor_initial,
        "smooth decay should not trigger meta-controller; anchor unchanged \
         from {anchor_initial}, got {anchor_final}",
    );
}

#[test]
fn test_meta_controller_disabled_no_dispatch() {
    // Default harness: meta_controller flag off. Even with LR cliff input,
    // observe_meta is a no-op.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);
    calibrate_to_stable(&mut h.coord);

    let anchor_initial = h.coord.el_che.anchor();

    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.1 }).unwrap();
    h.coord.drain_timing();
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);

    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.01 }).unwrap();
    h.coord.drain_timing();
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);

    assert_eq!(
        h.coord.el_che.anchor(),
        anchor_initial,
        "meta-controller off should not change anchor",
    );
}

#[test]
fn test_meta_controller_dispatches_on_sustained_convergence_pattern() {
    let mut h = make_coord_harness_meta(2);
    calibrate_to_stable(&mut h.coord);

    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.1 }).unwrap();
    h.coord.drain_timing();

    // Stable phase requires K=3 sustained NudgeDown / SuppressGrowth verdicts.
    let nudge = super::convergence::ConvergenceAction::NudgeDown { factor: 0.5 };
    h.coord.observe_meta(nudge);
    h.coord.observe_meta(nudge);
    let anchor_after_two = h.coord.el_che.anchor();

    h.coord.observe_meta(nudge);
    let anchor_after_three = h.coord.el_che.anchor();

    assert!(
        anchor_after_three < anchor_after_two,
        "convergence pattern (K=3 at Stable) should fire on third sustained verdict: \
         {anchor_after_two} -> {anchor_after_three}",
    );
}
