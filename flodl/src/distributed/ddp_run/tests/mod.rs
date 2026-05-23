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
pub(crate) use crate::distributed::ddp::ElChe;
pub(crate) use crate::nn::{Linear, Module};
pub(crate) use crate::tensor::{DType, Tensor, TensorError, TensorOptions, test_device, test_opts};
pub(crate) use std::sync::mpsc;

mod averaging;
mod builder_and_resume;
mod checkpoint;
mod coordinator;
mod dispatch;
mod lr;
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
        epoch: 0, partition_offset: 0, partition_size: 1000,
    });
    let _ckpt = ControlMsg::Checkpoint { version: 42, target_rank: 0 };
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
        save_path: None,
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
        save_path: None,
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
        None, // no eval_fn
        None, // no eval_dataset
        timing_tx,
        metrics_tx,
        param_tx,
        final_param_tx,
        control_rx,
    ).unwrap();

    (worker, channels)
}

// ---------------------------------------------------------------------------
// Shared coordinator-test fixtures (used by coordinator.rs, averaging.rs,
// dispatch.rs, lr.rs, builder_and_resume.rs)
// ---------------------------------------------------------------------------

/// Simple coordinator test helper.
pub(super) struct CoordTestHarness {
    coord: Coordinator,
    /// Send timing/metrics/params TO the coordinator.
    timing_tx: mpsc::Sender<TimingMsg>,
    metrics_tx: mpsc::Sender<MetricsMsg>,
    param_tx: mpsc::Sender<ParamSnapshot>,
    /// Receive control messages FROM the coordinator (one per worker).
    control_rxs: Vec<mpsc::Receiver<ControlMsg>>,
}

pub(super) fn make_coord_harness(
    n: usize,
    policy: ApplyPolicy,
    backend: AverageBackend,
) -> CoordTestHarness {
    make_coord_harness_with_timeout(n, policy, backend, 5)
}

pub(super) fn make_coord_harness_with_timeout(
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
