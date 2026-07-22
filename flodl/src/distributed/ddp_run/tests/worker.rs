//! GpuWorker unit tests.

use super::*;

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
    let (mut worker, _ch) = make_test_worker();
    let snap = worker.snapshot_params();
    assert_eq!(snap.rank, 0);
    assert_eq!(snap.params.len(), 2); // weight + bias
    assert_eq!(snap.buffers.len(), 0); // Linear has no buffers
    assert_eq!(snap.batch_count, 0); // true steps_since_avg (no .max(1) floor)

    // Verify snapshot tensors have the right shapes
    assert_eq!(snap.params[0].shape(), &[2, 4]); // weight
    assert_eq!(snap.params[1].shape(), &[2]);     // bias
}

#[test]
fn test_worker_snapshot_is_send() {
    let (mut worker, _ch) = make_test_worker();
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
fn test_update_subtracts_snapshot_steps_not_zeroes() {
    // cpu-async overshoot accounting: steps taken AFTER the snapshot but
    // BEFORE the Update survive the EASGD blend, so their mass must stay in
    // the counter for the next frame. Zeroing (the old behavior) dropped it.
    let (mut worker, _ch) = make_test_worker();

    // 5 steps trained, then the averaging trigger snapshots the params.
    worker.set_steps_since_avg(5);
    worker.dispatch_control(ControlMsg::RequestParams).unwrap();
    // Overshoot: 3 more steps land while the averaging round-trips.
    worker.set_steps_since_avg(worker.steps_since_avg() + 3);

    let cpu = TensorOptions { dtype: DType::Float32, device: Device::CPU };
    let update = AveragedParams {
        params: vec![
            Tensor::ones(&[2, 4], cpu).unwrap(),
            Tensor::ones(&[2], cpu).unwrap(),
        ],
        buffers: vec![],
        version: 1,
    };
    worker.dispatch_control(ControlMsg::Update(update)).unwrap();

    // The 5 shipped steps are consumed; the 3 overshoot steps remain.
    assert_eq!(worker.steps_since_avg(), 3, "overshoot steps must keep their mass credit");

    // A spurious second Update must subtract 0 (marker was reset), not
    // re-subtract the stale snapshot count.
    let update2 = AveragedParams {
        params: vec![
            Tensor::ones(&[2, 4], cpu).unwrap(),
            Tensor::ones(&[2], cpu).unwrap(),
        ],
        buffers: vec![],
        version: 2,
    };
    worker.dispatch_control(ControlMsg::Update(update2)).unwrap();
    assert_eq!(worker.steps_since_avg(), 3, "second Update without a snapshot subtracts nothing");
}

#[test]
fn test_snapshot_never_aliases_live_params() {
    // The snapshot must be a stable copy: mutating the live params after
    // snapshot_params returns must not change the snapshot's contents. On
    // CUDA the pinned-staging path always copied; on a CPU device the old
    // passthrough returned storage-sharing clones — under Async the bridge
    // thread serialized them while the worker kept training (torn floats).
    let (mut worker, _ch) = make_test_worker();
    let snap = worker.snapshot_params();
    let before: f64 = snap.params[0].sum().unwrap().item().unwrap();

    // Mutate the live weight in place (what an optimizer step does).
    let cpu = TensorOptions { dtype: DType::Float32, device: Device::CPU };
    let ones = Tensor::ones(&[2, 4], cpu).unwrap();
    crate::autograd::no_grad(|| -> crate::tensor::Result<()> {
        let live = worker.param_vars[0].data();
        let src = if live.device() == Device::CPU {
            ones
        } else {
            ones.to_device(live.device())?
        };
        live.copy_(&src, false)?;
        Ok(())
    })
    .unwrap();
    if let Device::CUDA(idx) = test_device() {
        crate::tensor::cuda_synchronize(idx);
    }

    let after: f64 = snap.params[0].sum().unwrap().item().unwrap();
    assert!(
        (after - before).abs() < 1e-6,
        "snapshot changed after live-param mutation (aliased storage): before={before} after={after}"
    );
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

    worker.report_timing(12.5, 2.0, None, 0.5, None).unwrap();

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

    // SyncNow is a no-op without NCCL.
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
    timing_tx.send(TimingMsg::Batch { rank: 0, batch_ms: 1.0, data_ms: 0.0, step_count: 0, param_norm: None, batch_loss: 0.1, sync_divergence: None }).unwrap();
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

/// Minimal `EpochMetrics` fixture stamped with an epoch + loss so the
/// stream-order assertion can tell frames apart.
fn epoch_metrics_fixture(epoch: usize, avg_loss: f64) -> EpochMetrics {
    EpochMetrics {
        epoch,
        scalars: HashMap::new(),
        per_rank: vec![],
        avg_loss,
        epoch_ms: 0.0,
        per_rank_throughput: vec![],
        per_rank_batch_share: vec![],
        per_rank_share_complete_ms: vec![],
        per_rank_compute_only_ms: vec![],
        per_rank_data_starve_ms: vec![],
        device_indices: vec![],
    }
}

/// The cooperative-tier metrics stream: once armed, every coord-broadcast
/// `EpochAggregated` is forwarded to the drain receiver in order (the full
/// per-epoch series), *and* the latest still lands in the `aggregated_metrics`
/// slot (which backs `Worker::epoch_metrics`). This is the receiving half of
/// the closure exercised end-to-end (with a live coordinator) by the NCCL
/// suite; here it is pinned on CPU without NCCL.
#[test]
fn test_worker_metrics_stream_forwards_every_epoch() {
    let (mut worker, ch) = make_test_worker();
    let rx = worker.enable_metrics_stream();

    // Two aggregated epochs arrive on the control channel (as the coordinator
    // broadcasts them); a single handle_control drain processes both.
    ch.control_tx
        .send(ControlMsg::EpochAggregated(epoch_metrics_fixture(0, 0.9)))
        .unwrap();
    ch.control_tx
        .send(ControlMsg::EpochAggregated(epoch_metrics_fixture(1, 0.5)))
        .unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);

    // Stream carries BOTH epochs, oldest first (no epoch is dropped).
    let drained: Vec<EpochMetrics> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
    assert_eq!(drained.len(), 2, "both aggregated epochs must be streamed");
    assert_eq!(drained[0].epoch, 0);
    assert_eq!(drained[1].epoch, 1);
    assert!((drained[1].avg_loss - 0.5).abs() < 1e-9);

    // Latest-only slot still tracks the most recent epoch (backs epoch_metrics).
    let latest = worker.aggregated_metrics().lock().unwrap().clone();
    assert_eq!(latest.unwrap().epoch, 1);
}

/// Without arming the stream (the managed / setup default), `EpochAggregated`
/// still updates the latest-only slot but nothing accumulates — no unbounded
/// queue in the tiers that never drain it.
#[test]
fn test_worker_metrics_slot_without_stream() {
    let (mut worker, ch) = make_test_worker();
    // No enable_metrics_stream(): metrics_stream_tx stays None.
    ch.control_tx
        .send(ControlMsg::EpochAggregated(epoch_metrics_fixture(3, 0.1)))
        .unwrap();
    worker.handle_control().unwrap();
    let latest = worker.aggregated_metrics().lock().unwrap().clone();
    assert_eq!(latest.unwrap().epoch, 3);
}

/// The cooperative-tier eval stream: once armed, every `EvalBroadcast` frame
/// is forwarded to the drain receiver in order (the receiving half of the
/// controller-elected eval surfaced by `Worker::poll_eval`, pinned on CPU
/// without NCCL). Unarmed, nothing accumulates.
#[test]
fn test_worker_eval_stream_forwards_each_broadcast() {
    let (mut worker, ch) = make_test_worker();
    let rx = worker.enable_eval_stream();

    ch.control_tx
        .send(ControlMsg::EvalBroadcast { epoch: 1, metric: 0.80 })
        .unwrap();
    ch.control_tx
        .send(ControlMsg::EvalBroadcast { epoch: 3, metric: 0.91 })
        .unwrap();
    assert!(!worker.handle_control().unwrap());

    let drained: Vec<(usize, f64)> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
    assert_eq!(drained.len(), 2, "both eval broadcasts must be streamed");
    assert_eq!(drained[0].0, 1);
    assert_eq!(drained[1].0, 3);
    assert!((drained[1].1 - 0.91).abs() < 1e-9);
}

/// Without arming the eval stream, `EvalBroadcast` is a no-op drain (no
/// accumulation in managed / setup mode).
#[test]
fn test_worker_eval_broadcast_without_stream_is_noop() {
    let (mut worker, ch) = make_test_worker();
    ch.control_tx
        .send(ControlMsg::EvalBroadcast { epoch: 2, metric: 0.5 })
        .unwrap();
    // Just must not panic / must drain cleanly (eval_stream_tx is None).
    assert!(!worker.handle_control().unwrap());
}

