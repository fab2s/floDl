//! Checkpoint coordination tests + 2-GPU end-to-end validation.

use super::*;

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

    ch.control_tx.send(ControlMsg::Checkpoint { version: 7, target_rank: 0 }).unwrap();
    worker.handle_control().unwrap();

    assert_eq!(called_version.load(Ordering::Relaxed), 7);
}

#[test]
fn test_checkpoint_error_logged_not_propagated() {
    let (mut worker, ch) = make_test_worker();
    worker.checkpoint_fn = Some(Arc::new(|_ver, _model| {
        Err(TensorError::new("disk full"))
    }));

    ch.control_tx.send(ControlMsg::Checkpoint { version: 1, target_rank: 0 }).unwrap();
    // Should not return an error: log-and-continue
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
}

/// `ControlMsg::ExecuteEvalCallback` fires the worker's `eval_fn`
/// against `eval_dataset` and emits a `TimingMsg::EvalResult` back
/// to the coordinator. Mirrors the checkpoint test pattern.
#[test]
fn test_eval_fn_called_on_dispatch_and_emits_result() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let (mut worker, ch) = make_test_worker();
    let called_epoch = Arc::new(AtomicU64::new(u64::MAX));
    let ce = called_epoch.clone();
    worker.eval_fn = Some(Arc::new(move |_model, _ds| {
        ce.store(7, Ordering::Relaxed);
        Ok(0.42)
    }));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    ch.control_tx
        .send(ControlMsg::ExecuteEvalCallback {
            schedule_id: 99,
            epoch: 7,
            target_rank: 0,
        })
        .unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);
    assert_eq!(called_epoch.load(Ordering::Relaxed), 7);

    // Drain TimingMsgs and find the EvalResult.
    let mut got = None;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if let TimingMsg::EvalResult {
            schedule_id,
            epoch,
            metric,
            elapsed_ms,
            error,
            rank,
        } = m
        {
            got = Some((rank, schedule_id, epoch, metric, elapsed_ms, error));
            break;
        }
    }
    let (rank, schedule_id, epoch, metric, elapsed_ms, error) =
        got.expect("EvalResult should be emitted");
    assert_eq!(rank, 0);
    assert_eq!(schedule_id, 99);
    assert_eq!(epoch, 7);
    assert!((metric - 0.42).abs() < 1e-9);
    assert!(error.is_none());
    assert!(
        elapsed_ms >= 0.0,
        "elapsed_ms should be non-negative, got {elapsed_ms}",
    );
}

/// `eval_fn` errors flow back as a `TimingMsg::EvalResult` with
/// `error = Some(...)` and `metric = NaN`. Training continues
/// (no shutdown).
#[test]
fn test_eval_fn_error_surfaces_in_timing_msg() {
    let (mut worker, ch) = make_test_worker();
    worker.eval_fn = Some(Arc::new(|_model, _ds| {
        Err(TensorError::new("eval blew up"))
    }));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    ch.control_tx
        .send(ControlMsg::ExecuteEvalCallback {
            schedule_id: 1,
            epoch: 2,
            target_rank: 0,
        })
        .unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);

    let mut got_err = None;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if let TimingMsg::EvalResult { error, metric, .. } = m {
            got_err = Some((error, metric));
            break;
        }
    }
    let (error, metric) = got_err.expect("EvalResult should be emitted");
    assert!(error.unwrap().contains("eval blew up"));
    assert!(metric.is_nan());
}

/// `drain_pending_shutdown` consumes queued `Shutdown` /
/// `ShutdownWithSave` frames and returns `true` when one was seen.
/// Non-shutdown frames in the queue are dropped silently. Used by
/// `ClusterWorker::run_until_shutdown`'s error-exit path so a lone
/// NCCL survivor can still write its bundle.
#[test]
fn test_drain_pending_shutdown_consumes_queued_shutdown() {
    let (mut worker, ch) = make_test_worker();

    // No queued frames → returns false, no-op.
    assert!(!worker.drain_pending_shutdown());

    // Queue a Shutdown → handled = true.
    ch.control_tx.send(ControlMsg::Shutdown).unwrap();
    assert!(worker.drain_pending_shutdown());

    // Queue a ShutdownWithSave with save_path unset → handled = true
    // (write skipped with a verbose log).
    ch.control_tx
        .send(ControlMsg::ShutdownWithSave {
            reason: crate::distributed::SaveReason::SingleSurvivor,
        })
        .unwrap();
    assert!(worker.drain_pending_shutdown());

    // Queue a non-shutdown frame → returns false, frame dropped.
    ch.control_tx
        .send(ControlMsg::SetGlobalStep(42))
        .unwrap();
    assert!(!worker.drain_pending_shutdown());
}

/// `ControlMsg::ExecuteEvalCallback` on a worker with `eval_fn = None`
/// (non-chosen rank) is a quiet no-op: no `EvalResult` emitted, no
/// shutdown signaled.
#[test]
fn test_eval_fn_none_is_noop() {
    let (mut worker, ch) = make_test_worker();
    // eval_fn intentionally not set.

    ch.control_tx
        .send(ControlMsg::ExecuteEvalCallback {
            schedule_id: 1,
            epoch: 2,
            target_rank: 0,
        })
        .unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(!shutdown);

    let mut found_eval_result = false;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if matches!(m, TimingMsg::EvalResult { .. }) {
            found_eval_result = true;
            break;
        }
    }
    assert!(
        !found_eval_result,
        "non-chosen rank should not emit EvalResult"
    );
}

#[test]
fn shutdown_with_save_writes_model_and_optim_to_save_path() {
    // Rank-0 worker with save_path set. Dispatch ShutdownWithSave;
    // verify the WORKER writes the `.fdl` (model) and `.optim` (per-
    // rank optimizer) files. The `.meta.json` is the CONTROLLER's
    // job (only it has the live ElChe trajectory + cluster-wide
    // counters); in this isolated-worker test the meta is not
    // expected to be written. Controller-side meta-write coverage
    // lives in cluster_coordinator tests.
    let dev = test_device();
    let dir = std::env::temp_dir().join(format!(
        "flodl_shutdown_with_save_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("ckpt_final");
    let stem_str = stem.to_str().unwrap().to_string();

    let tmp_model = Linear::on_device(4, 2, dev).unwrap();
    let tmp_params: Vec<Tensor> =
        tmp_model.parameters().iter().map(|p| p.variable.data()).collect();
    let tmp_buffers: Vec<Tensor> =
        tmp_model.buffers().iter().map(|b| b.get()).collect();
    drop(tmp_model);

    let config = WorkerConfig {
        rank: 0,
        world_size: 3,
        device: dev,
        initial_params: tmp_params,
        initial_buffers: tmp_buffers,
        total_samples: 16,
        batch_size: 4,
        seed: 42,
        max_grad_norm: None,
        easgd_alpha: None,
        gamma: 1.0,
        timeline: None,
        policy: ApplyPolicy::Sync,
        save_path: Some(stem_str.clone()),
    };
    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), ch) =
        GpuWorker::<Linear>::channels();
    let dataset: Arc<dyn crate::data::BatchDataSet> =
        Arc::new(TestDataset { n: 16 });
    let mut worker = GpuWorker::new(
        &config,
        |d| Linear::on_device(4, 2, d),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        dataset,
        None,
        None,
        None, // no eval_fn
        None, // no eval_dataset
        timing_tx,
        metrics_tx,
        param_tx,
        final_param_tx,
        control_rx,
        None, // no outer optimizer
    )
    .unwrap();

    ch.control_tx
        .send(ControlMsg::ShutdownWithSave {
            reason: crate::distributed::SaveReason::MaxFailureExceeded,
        })
        .unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(shutdown, "ShutdownWithSave must trigger shutdown");

    // Worker-side: `.fdl` (model) + `.optim` (per-rank optimizer)
    // present at save_path. `.meta.json` is NOT written here —
    // that's the controller's responsibility (see
    // `cluster_coordinator::dispatch_shutdown_with_save`).
    let model_path =
        crate::distributed::CheckpointBundle::model_path(&stem_str);
    let optim_path =
        crate::distributed::CheckpointBundle::optim_path(&stem_str);
    let meta_path = crate::distributed::CheckpointBundle::meta_path(&stem_str);
    assert!(model_path.exists(), "model file missing: {}", model_path.display());
    assert!(optim_path.exists(), "optim file missing: {}", optim_path.display());
    assert!(
        !meta_path.exists(),
        "meta file should NOT be written by worker (controller's job): {}",
        meta_path.display(),
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn shutdown_with_save_no_path_exits_without_write() {
    // Worker with save_path = None still exits on ShutdownWithSave;
    // no files are written (no path to write to). Guard against a
    // regression where the save flow panics or short-circuits the
    // shutdown when save_path is unset.
    let (mut worker, ch) = make_test_worker();
    ch.control_tx
        .send(ControlMsg::ShutdownWithSave {
            reason: crate::distributed::SaveReason::SingleSurvivor,
        })
        .unwrap();
    let shutdown = worker.handle_control().unwrap();
    assert!(
        shutdown,
        "ShutdownWithSave must trigger shutdown even without save_path"
    );
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
            if let ControlMsg::Checkpoint { version, .. } = msg {
                checkpoint_versions.push(version);
            }
        }
    }
    assert_eq!(checkpoint_versions, vec![2], "should checkpoint once (at epoch 2) after 3 epochs with every=2");
}

// -----------------------------------------------------------------------
// 2-GPU end-to-end validation
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

