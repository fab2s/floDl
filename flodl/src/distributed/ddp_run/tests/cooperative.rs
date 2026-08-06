//! Cooperative-tier [`Worker`] unit tests (single-device / CPU).
//!
//! The cluster variant needs the live rig (bridges + coordinator over TCP) and
//! is validated by the 2-GPU equivalence test; here we lock the single-device
//! orchestration and — most importantly — that driving the cooperative loop
//! produces the same trained params as the managed `run_epoch_plan` (the "one
//! code path" claim, at the `Worker` level).

use super::*;

use std::sync::Arc;

use crate::data::BatchDataSet;

/// Deterministic dataset: `get_batch` is a pure function of the indices, so two
/// workers fed the same (seeded) partition see byte-identical data. Enables the
/// managed-vs-cooperative bit-identity comparison.
struct DeterministicDataset {
    n: usize,
}
impl BatchDataSet for DeterministicDataset {
    fn len(&self) -> usize {
        self.n
    }
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let n = indices.len() as i64;
        let opts = TensorOptions { dtype: DType::Float32, device: Device::CPU };
        // Scalar derived deterministically from the indices; uniform rows are
        // fine — only determinism across the two workers matters.
        let s: usize = indices.iter().copied().sum();
        let v = (s % 17) as f32 * 0.05 + 0.1;
        Ok(vec![
            Tensor::full(&[n, 4], v as f64, opts)?,
            Tensor::full(&[n, 2], (v * 0.5 + 0.2) as f64, opts)?,
        ])
    }
}

/// Build a single-device `GpuWorker<Linear>` over the deterministic dataset.
/// Mirrors `make_test_worker_with` but with reproducible data + a caller-chosen
/// dataset size (`total` must be a multiple of the batch size, 4).
fn make_det_worker(total: usize) -> (GpuWorker<Linear>, WorkerChannels) {
    let dev = test_device();

    let tmp = Linear::on_device(4, 2, dev).unwrap();
    let initial_params: Vec<Tensor> =
        tmp.parameters().iter().map(|p| p.variable.data()).collect();
    let initial_buffers: Vec<Tensor> = tmp.buffers().iter().map(|b| b.get()).collect();
    drop(tmp);

    let config = WorkerConfig {
        rank: 0,
        world_size: 1,
        device: dev,
        initial_params,
        initial_buffers,
        total_samples: total,
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
        save_path: None,
        model_sig: [0u8; 32],
        coord_liveness_timeout_secs:
            crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
    };

    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), channels) =
        GpuWorker::<Linear>::channels();
    let dataset: Arc<dyn BatchDataSet> = Arc::new(DeterministicDataset { n: total });

    let worker = GpuWorker::new(
        &config,
        |d| Linear::on_device(4, 2, d),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        dataset,
        None, // no NCCL
        None, // no checkpoint
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

    (worker, channels)
}

/// Copy `src`'s params + buffers into `dst` so the two workers start identical.
fn sync_params(dst: &mut GpuWorker<Linear>, src: &mut GpuWorker<Linear>) {
    let snap = src.snapshot_params();
    dst.load_averaged(&AveragedParams {
        params: snap.params,
        buffers: snap.buffers,
        version: 0,
    })
    .unwrap();
    if let Device::CUDA(idx) = test_device() {
        crate::tensor::gpu_synchronize(idx);
    }
}

#[test]
fn single_host_worker_trains_and_finishes() {
    let total = 16; // 4 batches/epoch at batch_size 4
    let epochs = 2;
    let (w, _ch) = make_det_worker(total);
    let mut coop = Worker::single(w, epochs, total);

    let mut steps = 0usize;
    while let Some(_plan) = coop.next_plan().unwrap() {
        while let Some(batch) = coop.next_batch().unwrap() {
            let loss = mse_train(coop.model(), &batch).unwrap();
            loss.backward().unwrap();
            let outcome = coop.step(&loss).unwrap();
            assert!(!outcome.shutdown, "no shutdown expected on single device");
            steps += 1;
        }
    }
    assert_eq!(steps, epochs * (total / 4), "one step per batch per epoch");

    let state = coop.finish().unwrap();
    assert_eq!(state.params.len(), 2, "Linear: weight + bias");
    assert_eq!(state.buffers.len(), 0, "Linear has no buffers");
}

#[test]
fn worker_next_plan_stops_after_num_epochs() {
    let (w, _ch) = make_det_worker(16);
    let mut coop = Worker::single(w, 3, 16);
    assert!(coop.next_plan().unwrap().is_some());
    assert!(coop.next_plan().unwrap().is_some());
    assert!(coop.next_plan().unwrap().is_some());
    assert!(coop.next_plan().unwrap().is_none(), "None after num_epochs");
    assert!(coop.next_plan().unwrap().is_none(), "stays None");
}

#[test]
fn worker_step_without_batch_errs() {
    let (w, _ch) = make_det_worker(16);
    let mut coop = Worker::single(w, 1, 16);
    // A dummy loss Variable; step should reject before touching it because no
    // batch is in flight.
    let opts = test_opts();
    let loss = Variable::new(Tensor::full(&[1], 1.0, opts).unwrap(), true);
    assert!(
        coop.step(&loss).is_err(),
        "step with no batch in flight must error"
    );
}

#[test]
fn worker_sync_now_drains_reduce_in_step() {
    // A coordinator SyncNow queued on the control channel must be drained by
    // step() (via after_step -> handle_control), driving the sync path. With no
    // NCCL comm the collective is a no-op, but the SyncNow arm still emits a
    // SyncAck on the timing channel — the observable proof the reduce path ran
    // inside step(), not decided by the Worker.
    let total = 16;
    let (w, ch) = make_det_worker(total);
    let mut coop = Worker::single(w, 1, total);

    // Advance into the epoch and run one batch's forward/backward.
    assert!(coop.next_plan().unwrap().is_some());
    let batch = coop.next_batch().unwrap().expect("first batch");
    let loss = mse_train(coop.model(), &batch).unwrap();
    loss.backward().unwrap();

    // Queue the coordinator's SyncNow, then step.
    ch.control_tx.send(ControlMsg::SyncNow).unwrap();
    coop.step(&loss).unwrap();

    // Drain timing messages; a SyncAck must be present.
    let mut saw_sync_ack = false;
    while let Ok(msg) = ch.timing_rx.try_recv() {
        if matches!(msg, TimingMsg::SyncAck { .. }) {
            saw_sync_ack = true;
        }
    }
    assert!(
        saw_sync_ack,
        "step() must drain the SyncNow and emit a SyncAck (reduce path ran)"
    );
}

#[test]
fn builder_into_worker_single_device_trains() {
    // End-to-end through the public entry: Trainer::builder(...).into_worker()
    // → DdpHandle::into_worker → run_single_worker → cooperative loop. Exercises
    // finalize_and_extract + the role dispatch's single-device fallback.
    //
    // Under cfg(test) auto-promote is gated off, so on a 2+-GPU rig
    // into_worker would hit the "in-process multi-GPU removed" error (the
    // cluster path needs process-per-rank, not unit-testable). Skip there; the
    // single-device path is device-agnostic and covered on CPU / single GPU.
    if crate::tensor::usable_gpu_devices().len() >= 2 {
        return;
    }
    use crate::distributed::Trainer;
    let total = 16;
    let dataset: Arc<dyn BatchDataSet> = Arc::new(DeterministicDataset { n: total });
    let mut w = Trainer::builder(
        |d| Linear::on_device(4, 2, d),
        |p| crate::nn::SGD::new(p, 0.01, 0.0),
        mse_train,
    )
    .dataset(dataset)
    .batch_size(4)
    .num_epochs(2)
    .into_worker()
    .unwrap();

    let mut steps = 0usize;
    while let Some(_plan) = w.next_plan().unwrap() {
        while let Some(batch) = w.next_batch().unwrap() {
            let loss = mse_train(w.model(), &batch).unwrap();
            loss.backward().unwrap();
            w.step(&loss).unwrap();
            steps += 1;
        }
    }
    assert_eq!(steps, 2 * (total / 4));
    let state = w.finish().unwrap();
    assert_eq!(state.params.len(), 2, "Linear: weight + bias");
}

#[test]
fn cooperative_worker_matches_managed_run_epoch_plan() {
    // The load-bearing test: driving the cooperative loop (next_plan /
    // next_batch / user fwd+bwd / step) produces the same trained params as the
    // managed run_epoch_plan, given identical init + identical (deterministic)
    // data. Locks the "one code path" claim at the Worker level.
    let total = 16;
    let epochs = 3;

    let (mut managed, _cm) = make_det_worker(total);
    let (mut coop_w, _cc) = make_det_worker(total);
    // Identical starting params.
    sync_params(&mut coop_w, &mut managed);

    // Managed: drive run_epoch_plan directly over synthesized whole-dataset
    // plans (the single-device managed shape).
    for epoch in 0..epochs {
        let plan = EpochPlan {
            epoch,
            partition_offset: 0,
            partition_size: total,
        };
        managed.run_epoch_plan(&plan, &mse_train).unwrap();
    }
    let managed_snap = managed.snapshot_params();

    // Cooperative: user owns the loop.
    let mut coop = Worker::single(coop_w, epochs, total);
    while let Some(_plan) = coop.next_plan().unwrap() {
        while let Some(batch) = coop.next_batch().unwrap() {
            let loss = mse_train(coop.model(), &batch).unwrap();
            loss.backward().unwrap();
            coop.step(&loss).unwrap();
        }
    }
    let coop_state = coop.finish().unwrap();

    assert_eq!(managed_snap.params.len(), coop_state.params.len());
    for (i, (m, c)) in managed_snap
        .params
        .iter()
        .zip(&coop_state.params)
        .enumerate()
    {
        // Both on CPU (snapshot + finish land on CPU). Same init + same data +
        // same ops => bit-identical up to a tiny tolerance.
        let diff: f64 = m.sub(c).unwrap().abs().unwrap().sum().unwrap().item().unwrap();
        assert!(
            diff < 1e-6,
            "param {i} diverged between managed and cooperative: L1 diff = {diff}"
        );
    }
}
