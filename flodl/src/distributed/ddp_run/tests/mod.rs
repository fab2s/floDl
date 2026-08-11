//! Top-level tests for [`super`] (the `ddp_run` module).
//!
//! Hosts the file-root basics tests (variant enumeration, `Send`/`Clone`
//! contracts) plus the shared test fixtures consumed by sibling files.
//! Each topic area lives in its own sibling under `tests/`; the fixtures
//! here are `pub(super)` so child modules can pull them with
//! `use super::*;`. Common imports (`ApplyPolicy`, `AverageBackend`,
//! `Tensor`, `Linear`, etc.) are re-exported via `pub use` so child
//! files don't have to repeat the boilerplate.

pub(crate) use super::*;
pub(crate) use crate::autograd::Variable;
pub(crate) use crate::nn::{Linear, Module};
pub(crate) use crate::tensor::{DType, Tensor, TensorError, TensorOptions, test_device, test_opts};
pub(crate) use std::sync::mpsc;

mod builder_and_resume;
mod checkpoint;
mod cooperative;
mod worker;

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
        epoch: 0,
        partition_offset: 0,
        partition_size: 1000,
    });
    let _ckpt = ControlMsg::Checkpoint {
        version: 42,
        target_rank: 0,
    };
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
        augment: 1,
        transform: None,
        vram_max_usage: 0.90,
        ram_max_usage: 0.50,
        gpu_ram_share: None,
        sample_cache: true,
        disk_stage_gb: 0,
        disk_stage_dir: None,
        batch_size: 32,
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
        profile_graph: false,
        coord_liveness_timeout_secs:
            crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
    };
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.rank, 0);
    assert_eq!(cfg2.world_size, 2);
    assert_eq!(cfg2.total_samples, 10000);
}

/// Simple test dataset: random (input, target) pairs.
pub(super) struct TestDataset {
    n: usize,
}
impl crate::data::BatchDataSet for TestDataset {
    fn len(&self) -> usize {
        self.n
    }
    fn get_batch(&self, indices: &[usize]) -> crate::tensor::Result<Vec<Tensor>> {
        let n = indices.len() as i64;
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: Device::CPU,
        };
        Ok(vec![
            Tensor::randn(&[n, 4], opts)?,
            Tensor::randn(&[n, 2], opts)?,
        ])
    }
}

/// Simple MSE train function for tests.
pub(super) fn mse_train(model: &Linear, batch: &[Tensor]) -> Result<Variable> {
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
pub(super) fn make_test_worker() -> (GpuWorker<Linear>, WorkerChannels) {
    make_test_worker_with(0, 1, 4)
}

/// Create a GpuWorker with configurable rank/world_size/dataset_size.
pub(super) fn make_test_worker_with(
    rank: usize,
    world_size: usize,
    dataset_size: usize,
) -> (GpuWorker<Linear>, WorkerChannels) {
    make_test_worker_customized(rank, world_size, dataset_size, |_| {})
}

/// [`make_test_worker_with`] with a config tweak applied before
/// construction — for tests exercising config-gated behavior the fixed
/// defaults can't reach (e.g. the EASGD blend needs `policy: Async` +
/// `easgd_alpha` to pass the constructor's single authoritative gate).
pub(super) fn make_test_worker_customized(
    rank: usize,
    world_size: usize,
    dataset_size: usize,
    tweak: impl FnOnce(&mut WorkerConfig),
) -> (GpuWorker<Linear>, WorkerChannels) {
    let dev = test_device();

    // Build a temporary model to extract initial params
    let tmp_model = Linear::on_device(4, 2, dev).unwrap();
    let tmp_params: Vec<Tensor> = tmp_model
        .parameters()
        .iter()
        .map(|p| p.variable.data())
        .collect();
    let tmp_buffers: Vec<Tensor> = tmp_model.buffers().iter().map(|b| b.get()).collect();
    drop(tmp_model);

    let mut config = WorkerConfig {
        rank,
        world_size,
        device: dev,
        initial_params: tmp_params,
        initial_buffers: tmp_buffers,
        total_samples: dataset_size,
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
        profile_graph: false,
        coord_liveness_timeout_secs:
            crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
    };
    tweak(&mut config);

    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), channels) =
        GpuWorker::<Linear>::channels();

    let dataset: Arc<dyn crate::data::BatchDataSet> = Arc::new(TestDataset { n: dataset_size });

    let worker = GpuWorker::new(
        &config,
        |d| Linear::on_device(4, 2, d),
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        dataset,
        None, // no NCCL in unit tests
        None, // no checkpoint in unit tests
        None, // no eval_fn
        None, // no eval_dataset
        timing_tx,
        metrics_tx,
        param_tx,
        final_param_tx,
        control_rx,
        None, // no outer optimizer in unit tests
    )
    .unwrap();

    (worker, channels)
}

// In-process coordinator test fixtures lived here. They were removed with
// the in-process orchestration engine; the process-per-rank path is covered
// by `cluster_coordinator/tests/` and `cluster_worker_tests.rs`.

#[test]
fn test_zero_param_model_is_rejected_loudly() {
    // A custom leaf module that forgets to override Module::parameters()
    // reaches the trainer with an empty parameter list; every training
    // entry rejects that instead of silently training nothing.
    let err = super::ensure_trainable_params(0, "ddp: single device")
        .expect_err("zero-parameter model must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("zero trainable parameters"),
        "unexpected: {msg}"
    );
    assert!(msg.contains("Module::parameters()"), "unexpected: {msg}");
    super::ensure_trainable_params(1, "ddp: single device").unwrap();
}

// ---------------------------------------------------------------------------
// Partition generation under epoch splits
// ---------------------------------------------------------------------------

#[test]
fn one_split_partitions_the_whole_pass() {
    // The default must resolve exactly the pass permutation, so runs
    // predating the splits knob reproduce bit for bit.
    assert_eq!(
        super::make_partition(0, 100, 100, 3, 1, 42),
        crate::rng::epoch_permutation(42, 3, 100),
    );
}

#[test]
fn partitions_tile_each_split_and_the_pass() {
    // Two ranks over four events of one pass. Coordinator-style
    // consecutive offsets, but relative to the EVENT's slice rather
    // than the pass — the property that lets ChunkPool keep working in
    // [0, epoch_len) while the pass mapping happens in make_partition.
    let (total, splits, seed) = (100usize, 4usize, 42u64);
    let mut pass: Vec<usize> = Vec::new();
    for event in 0..splits {
        let (_, len) = crate::rng::epoch_split_span(event, splits, total);
        let first = len / 2;
        let mut ev = super::make_partition(0, first, total, event, splits, seed);
        ev.extend(super::make_partition(
            first,
            len - first,
            total,
            event,
            splits,
            seed,
        ));
        assert_eq!(
            ev.len(),
            len,
            "event {event} partitions must tile its slice"
        );
        pass.extend(ev);
    }
    // Disjointness, coverage and ordering, through the partition layer.
    assert_eq!(pass, crate::rng::epoch_permutation(seed, 0, total));
}

#[test]
fn an_oversized_request_clamps_to_the_epoch_not_the_pass() {
    // Four splits of 100 picks: event 0 holds 25. A request for the
    // whole pass must stop at the event boundary rather than spill into
    // the next split's picks, which another event will train on.
    let got = super::make_partition(0, 100, 100, 0, 4, 42);
    assert_eq!(got.len(), 25);
    assert_eq!(got, crate::rng::epoch_split_permutation(42, 0, 4, 100));
}

#[test]
fn later_events_of_a_pass_reuse_nothing() {
    let (total, splits, seed) = (60usize, 3usize, 7u64);
    let mut seen = std::collections::HashSet::new();
    for event in 0..splits {
        let (_, len) = crate::rng::epoch_split_span(event, splits, total);
        for pick in super::make_partition(0, len, total, event, splits, seed) {
            assert!(
                seen.insert(pick),
                "pick {pick} served twice within one pass"
            );
        }
    }
    assert_eq!(seen.len(), total, "one pass must still cover the corpus");
}

#[test]
fn profile_graph_worker_accumulates_and_emits_at_teardown() {
    // The rank-side heat-map chain, no cluster required: profile_graph
    // in the WorkerConfig enables profiling on the factory-built Graph,
    // training-shaped forwards feed the accumulator, and
    // emit_graph_profile ships one GraphProfile frame on the timing
    // channel with the graph's identity attached.
    use crate::graph::{FlowBuilder, Graph};

    let dev = test_device();
    let build = |d: Device| -> crate::tensor::Result<Graph> {
        FlowBuilder::from(Linear::on_device(4, 2, d)?).build()
    };
    let tmp_model = build(dev).unwrap();
    let tmp_params: Vec<Tensor> = tmp_model
        .parameters()
        .iter()
        .map(|p| p.variable.data())
        .collect();
    let tmp_buffers: Vec<Tensor> = tmp_model.buffers().iter().map(|b| b.get()).collect();
    let expected_hash = tmp_model.structural_hash().to_string();
    drop(tmp_model);

    let config = WorkerConfig {
        rank: 0,
        world_size: 1,
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
        save_path: None,
        model_sig: [0u8; 32],
        profile_graph: true,
        coord_liveness_timeout_secs:
            crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
    };
    let ((timing_tx, metrics_tx, param_tx, final_param_tx, control_rx), ch) =
        GpuWorker::<Graph>::channels();
    let dataset: Arc<dyn crate::data::BatchDataSet> = Arc::new(TestDataset { n: 16 });
    let worker = GpuWorker::new(
        &config,
        build,
        |params| crate::nn::SGD::new(params, 0.01, 0.0),
        dataset,
        None,
        None,
        None,
        None,
        timing_tx,
        metrics_tx,
        param_tx,
        final_param_tx,
        control_rx,
        None,
    )
    .unwrap();

    // The constructor honoured the knob.
    assert!(worker.model().profiling());

    // Enough forwards to clear the 3-pass warmup (plus the one-pass lag
    // on a CUDA test device).
    let x = Variable::new(Tensor::from_f32(&[0.5; 4], &[1, 4], dev).unwrap(), false);
    for _ in 0..6 {
        worker.model().forward(&x).unwrap();
    }

    worker.emit_graph_profile();
    let msg = ch
        .timing_rx
        .try_recv()
        .expect("emit_graph_profile should queue one frame");
    match msg {
        TimingMsg::GraphProfile { rank, profile } => {
            assert_eq!(rank, 0);
            assert_eq!(profile.hash, expected_hash);
            assert_eq!(profile.nodes.len(), 1);
            assert!(profile.samples >= 1);
            assert!(profile.total_mean_ms >= 0.0);
            let expected_source = if dev.is_cuda() {
                "gpu events"
            } else {
                "host wall clock"
            };
            assert_eq!(profile.source, expected_source);
            assert!(!profile.gpu_model.is_empty());
        }
        other => panic!("expected GraphProfile, got {other:?}"),
    }
}
