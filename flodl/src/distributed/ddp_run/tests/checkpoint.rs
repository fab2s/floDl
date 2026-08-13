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

    ch.control_tx
        .send(ControlMsg::Checkpoint {
            version: 7,
            target_rank: 0,
        })
        .unwrap();
    worker.handle_control().unwrap();

    assert_eq!(called_version.load(Ordering::Relaxed), 7);
}

#[test]
fn test_checkpoint_error_logged_not_propagated() {
    let (mut worker, ch) = make_test_worker();
    worker.checkpoint_fn = Some(Arc::new(|_ver, _model| Err(TensorError::new("disk full"))));

    ch.control_tx
        .send(ControlMsg::Checkpoint {
            version: 1,
            target_rank: 0,
        })
        .unwrap();
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
            adopt_consensus: false,
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
            adopt_consensus: false,
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
    ch.control_tx.send(ControlMsg::SetGlobalStep(42)).unwrap();
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
            adopt_consensus: false,
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
    let dir = std::env::temp_dir().join(format!("flodl_shutdown_with_save_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("ckpt_final");
    let stem_str = stem.to_str().unwrap().to_string();

    let tmp_model = Linear::on_device(4, 2, dev).unwrap();
    let tmp_params: Vec<Tensor> = tmp_model
        .parameters()
        .iter()
        .map(|p| p.variable.data())
        .collect();
    let tmp_buffers: Vec<Tensor> = tmp_model.buffers().iter().map(|b| b.get()).collect();
    drop(tmp_model);

    let config = WorkerConfig {
        rank: 0,
        world_size: 3,
        device: dev,
        initial_params: tmp_params,
        initial_buffers: tmp_buffers,
        total_samples: 16,
        augment: 1,
        transform: None,
        vram_max_usage: 0.90,
        ram_max_usage: 0.50,
        gpu_ram_share: None,
        sample_cache: true,
        disk_stage_gb: 0,
        disk_stage_dir: None,
        batch_size: 4,
        seed: 42,
        epoch_splits: 1,
        max_grad_norm: None,
        vram_pool: false,
        easgd_alpha: None,
        gamma: 1.0,
        bf16_wire: false,
        timeline: None,
        policy: ApplyPolicy::Sync,
        save_path: Some(stem_str.clone()),
        model_sig: [0u8; 32],
        profile_graph: false,
        coord_liveness_timeout_secs:
            crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
    };
    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), ch) =
        GpuWorker::<Linear>::channels();
    let dataset: Arc<dyn crate::data::BatchDataSet> = Arc::new(TestDataset { n: 16 });
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
    let model_path = crate::distributed::CheckpointBundle::model_path(&stem_str);
    let optim_path = crate::distributed::CheckpointBundle::optim_path(&stem_str);
    let meta_path = crate::distributed::CheckpointBundle::meta_path(&stem_str);
    assert!(
        model_path.exists(),
        "model file missing: {}",
        model_path.display()
    );
    assert!(
        optim_path.exists(),
        "optim file missing: {}",
        optim_path.display()
    );
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

// `test_coordinator_sends_checkpoint_every_n_epochs` exercised the in-process
// Coordinator's epoch-cadence checkpoint trigger; it was removed with the
// engine. The process path's checkpoint cadence is covered under
// `cluster_coordinator/tests/`.

// 2-GPU end-to-end DDP validation lived here via the in-process engine
// (`DdpHandle::auto` + thread-per-GPU). That engine was removed; multi-GPU
// end-to-end training is now validated by the `ddp-bench` binary under
// `fdl gpu-test-nccl`, and the process-path logic by `cluster_worker_tests`.

/// An armed consensus eval on an EASGD rank scores THIS round's
/// consensus (not the blend) and restores the post-reduce blend
/// verbatim afterward — a round with an eval must leave the model
/// exactly where a round without one would.
#[test]
fn armed_consensus_eval_scores_consensus_and_restores_blend() {
    let alpha = 0.25;
    let (mut worker, ch) = make_test_worker_customized(0, 1, 4, |c| {
        c.policy = ApplyPolicy::Async;
        c.easgd_alpha = Some(alpha);
    });
    // The metric IS the model's first weight value at eval time: it
    // pins WHICH params the eval saw.
    worker.eval_fn = Some(Arc::new(|model, _ds| {
        let w = model.parameters()[0].variable.data().to_f32_vec()?;
        Ok(w[0] as f64)
    }));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    let pre_w = worker.param_vars[0].data().to_f32_vec().unwrap();

    let cpu = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let avg_val = 3.0;
    let update = AveragedParams {
        params: vec![
            Tensor::full(&[2, 4], avg_val, cpu).unwrap(),
            Tensor::full(&[2], -1.0, cpu).unwrap(),
        ],
        buffers: vec![],
        version: 7,
        realized: true,
    };
    ch.control_tx
        .send(ControlMsg::ArmConsensusEval {
            schedule_id: 21,
            epoch: 3,
            target_rank: 0,
        })
        .unwrap();
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();
    worker.handle_control().unwrap();

    if let Device::CUDA(idx) = test_device() {
        crate::tensor::gpu_synchronize(idx);
    }

    // The eval saw the consensus, not the blend.
    let mut got = None;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if let TimingMsg::EvalResult {
            schedule_id,
            epoch,
            metric,
            error,
            ..
        } = m
        {
            got = Some((schedule_id, epoch, metric, error));
            break;
        }
    }
    let (schedule_id, epoch, metric, error) = got.expect("EvalResult should be emitted");
    assert_eq!(schedule_id, 21, "EvalResult carries the armed schedule_id");
    assert_eq!(epoch, 3, "EvalResult carries the armed epoch");
    assert!(error.is_none(), "consensus eval must succeed: {error:?}");
    assert!(
        (metric - avg_val).abs() < 1e-5,
        "eval must score the CONSENSUS ({avg_val}), got {metric}"
    );

    // The model ends at the post-reduce blend, exactly as a round with
    // no eval would leave it.
    let post_w = worker.param_vars[0].data().to_f32_vec().unwrap();
    for (i, (pre, post)) in pre_w.iter().zip(&post_w).enumerate() {
        let want = (1.0 - alpha) as f32 * pre + alpha as f32 * avg_val as f32;
        assert!(
            (post - want).abs() < 1e-5,
            "weight[{i}] must be restored to the blend: want {want}, got {post}"
        );
    }
    // The armed directive is consumed.
    assert!(worker.pending_consensus_eval.is_none());
}

/// Without EASGD the writeback already overwrites the model with the
/// consensus — an armed eval scores in place, no swap.
#[test]
fn armed_consensus_eval_without_easgd_scores_in_place() {
    let (mut worker, ch) = make_test_worker();
    worker.eval_fn = Some(Arc::new(|model, _ds| {
        let w = model.parameters()[0].variable.data().to_f32_vec()?;
        Ok(w[0] as f64)
    }));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    let cpu = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let avg_val = 5.0;
    let update = AveragedParams {
        params: vec![
            Tensor::full(&[2, 4], avg_val, cpu).unwrap(),
            Tensor::full(&[2], -1.0, cpu).unwrap(),
        ],
        buffers: vec![],
        version: 1,
        realized: true,
    };
    ch.control_tx
        .send(ControlMsg::ArmConsensusEval {
            schedule_id: 1,
            epoch: 0,
            target_rank: 0,
        })
        .unwrap();
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();
    worker.handle_control().unwrap();

    let mut metric = None;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if let TimingMsg::EvalResult {
            metric: m2, error, ..
        } = m
        {
            assert!(error.is_none(), "in-place consensus eval failed: {error:?}");
            metric = Some(m2);
            break;
        }
    }
    let metric = metric.expect("EvalResult should be emitted");
    assert!(
        (metric - avg_val).abs() < 1e-5,
        "eval must score the consensus ({avg_val}), got {metric}"
    );
}

/// An unrealized (all-idle) round hands back the rank's own state, not
/// a consensus — the armed eval must survive it and fire at the next
/// realized round.
#[test]
fn armed_consensus_eval_survives_unrealized_round() {
    let (mut worker, ch) = make_test_worker();
    worker.eval_fn = Some(Arc::new(|_model, _ds| Ok(0.5)));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    let cpu = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let mk_update = |version, realized| AveragedParams {
        params: vec![
            Tensor::full(&[2, 4], 1.0, cpu).unwrap(),
            Tensor::full(&[2], 1.0, cpu).unwrap(),
        ],
        buffers: vec![],
        version,
        realized,
    };
    ch.control_tx
        .send(ControlMsg::ArmConsensusEval {
            schedule_id: 9,
            epoch: 1,
            target_rank: 0,
        })
        .unwrap();
    ch.control_tx
        .send(ControlMsg::Update(mk_update(1, false)))
        .unwrap();
    worker.handle_control().unwrap();
    assert!(
        worker.pending_consensus_eval.is_some(),
        "an unrealized round must not consume the armed eval"
    );
    assert!(
        !ch.timing_rx
            .try_iter()
            .any(|m| matches!(m, TimingMsg::EvalResult { .. })),
        "no EvalResult on an unrealized round"
    );

    ch.control_tx
        .send(ControlMsg::Update(mk_update(2, true)))
        .unwrap();
    worker.handle_control().unwrap();
    assert!(worker.pending_consensus_eval.is_none());
    assert!(
        ch.timing_rx
            .try_iter()
            .any(|m| matches!(m, TimingMsg::EvalResult { .. })),
        "the next realized round fires the armed eval"
    );
}

/// The final canonical eval (`adopt_consensus`) on an EASGD rank
/// overwrites the live blend with the retained last realized consensus
/// before scoring — and deliberately leaves the model there (training
/// is over; the elected rank ends holding the consensus).
#[test]
fn adopt_consensus_eval_scores_retained_consensus() {
    let alpha = 0.5;
    let (mut worker, ch) = make_test_worker_customized(0, 1, 4, |c| {
        c.policy = ApplyPolicy::Async;
        c.easgd_alpha = Some(alpha);
    });
    worker.eval_fn = Some(Arc::new(|model, _ds| {
        let w = model.parameters()[0].variable.data().to_f32_vec()?;
        Ok(w[0] as f64)
    }));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    let cpu = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let avg_val = 2.0;
    let update = AveragedParams {
        params: vec![
            Tensor::full(&[2, 4], avg_val, cpu).unwrap(),
            Tensor::full(&[2], 0.0, cpu).unwrap(),
        ],
        buffers: vec![],
        version: 4,
        realized: true,
    };
    // The realized round retains the consensus; the model itself holds
    // a BLEND afterwards (alpha 0.5 → halfway, never equal to avg).
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();
    ch.control_tx
        .send(ControlMsg::ExecuteEvalCallback {
            schedule_id: u64::MAX,
            epoch: 1,
            target_rank: 0,
            adopt_consensus: true,
        })
        .unwrap();
    worker.handle_control().unwrap();

    if let Device::CUDA(idx) = test_device() {
        crate::tensor::gpu_synchronize(idx);
    }

    let mut got = None;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if let TimingMsg::EvalResult { metric, error, .. } = m {
            got = Some((metric, error));
            break;
        }
    }
    let (metric, error) = got.expect("EvalResult should be emitted");
    assert!(error.is_none(), "final consensus eval failed: {error:?}");
    assert!(
        (metric - avg_val).abs() < 1e-5,
        "final eval must score the retained CONSENSUS ({avg_val}), got {metric}"
    );
    // Documented side effect: the elected rank ends holding the consensus.
    let post_w = worker.param_vars[0].data().to_f32_vec().unwrap();
    assert!(
        post_w.iter().all(|v| (v - avg_val as f32).abs() < 1e-5),
        "the elected rank ends at the consensus after the final adopt"
    );
}

/// `adopt_consensus` with nothing retained reports the error instead of
/// silently scoring the blend.
#[test]
fn adopt_consensus_eval_without_retained_reports_error() {
    let (mut worker, ch) = make_test_worker_customized(0, 1, 4, |c| {
        c.policy = ApplyPolicy::Async;
        c.easgd_alpha = Some(0.5);
    });
    worker.eval_fn = Some(Arc::new(|_model, _ds| Ok(0.1)));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    ch.control_tx
        .send(ControlMsg::ExecuteEvalCallback {
            schedule_id: u64::MAX,
            epoch: 1,
            target_rank: 0,
            adopt_consensus: true,
        })
        .unwrap();
    worker.handle_control().unwrap();

    let mut got = None;
    while let Ok(m) = ch.timing_rx.try_recv() {
        if let TimingMsg::EvalResult { metric, error, .. } = m {
            got = Some((metric, error));
            break;
        }
    }
    let (metric, error) = got.expect("EvalResult should be emitted");
    assert!(metric.is_nan(), "no consensus to score → NaN metric");
    assert!(
        error.is_some_and(|e| e.contains("no realized consensus")),
        "the missing retained consensus must be named in the error"
    );
}

/// `RequestParams` invalidates the retained consensus BEFORE the D2H
/// snapshot can clobber the staging it may alias — an adopt after a
/// snapshot (with no completing round) must refuse, not score torn
/// bytes.
#[test]
fn request_params_invalidates_retained_consensus() {
    let (mut worker, ch) = make_test_worker_customized(0, 1, 4, |c| {
        c.policy = ApplyPolicy::Async;
        c.easgd_alpha = Some(0.5);
    });
    worker.eval_fn = Some(Arc::new(|_model, _ds| Ok(0.1)));
    worker.eval_dataset = Some(Arc::new(TestDataset { n: 4 }));

    let cpu = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let update = AveragedParams {
        params: vec![
            Tensor::full(&[2, 4], 1.0, cpu).unwrap(),
            Tensor::full(&[2], 1.0, cpu).unwrap(),
        ],
        buffers: vec![],
        version: 1,
        realized: true,
    };
    ch.control_tx.send(ControlMsg::Update(update)).unwrap();
    worker.handle_control().unwrap();
    assert!(worker.last_consensus.is_some());

    ch.control_tx.send(ControlMsg::RequestParams).unwrap();
    worker.handle_control().unwrap();
    // Drain the snapshot the RequestParams produced.
    let _ = ch.param_rx.try_recv();
    assert!(
        worker.last_consensus.is_none(),
        "RequestParams must invalidate the retained consensus"
    );
}
