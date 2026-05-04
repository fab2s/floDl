//! DDP run mode: thread-per-GPU training with Local SGD and adaptive parameter averaging.
//!
//! Each GPU runs its own optimizer independently (zero wait). A lightweight coordinator
//! triggers periodic parameter averaging at ElChe-determined intervals. Two orthogonal
//! knobs control the behavior: [`ApplyPolicy`] (when to average) and [`AverageBackend`]
//! (how to average).
//!
//! # Quick start
//!
//! ```ignore
//! use flodl::*;
//!
//! let handle = Trainer::builder(model_factory, optim_factory, train_fn)
//!     .dataset(dataset)
//!     .batch_size(32)
//!     .num_epochs(10)
//!     .policy(ApplyPolicy::Cadence)
//!     .backend(AverageBackend::Nccl)
//!     .checkpoint_every(5)
//!     .checkpoint_fn(|ver, g| g.save_checkpoint(&format!("ckpt_v{ver}.fdl")))
//!     .run()?;
//!
//! let state = handle.join()?;
//! // state.params[i] corresponds to model.parameters()[i]
//! // state.buffers[i] corresponds to model.buffers()[i]
//! ```
//!
//! # Architecture
//!
//! ```text
//! GPU Thread 0:  create model+Adam+dataset -> [fwd -> bwd -> adam step -> repeat]
//! GPU Thread 1:  create model+Adam+dataset -> [fwd -> bwd -> adam step -> repeat]
//! Coordinator:   collect timing/metrics -> trigger param averaging -> monitor divergence
//! ```
//!
//! # Choosing a policy
//!
//! | Policy | When to use | Tradeoff |
//! |--------|-------------|----------|
//! | [`ApplyPolicy::Sync`] | Correctness-first, small models, homogeneous GPUs | Identical to standard DDP. Fast GPU waits at every batch. |
//! | [`ApplyPolicy::Cadence`] | Heterogeneous GPUs (e.g. Pascal + Blackwell) | Fast GPU runs ahead by ElChe-determined batches. Good throughput/convergence balance. |
//! | [`ApplyPolicy::Async`] | Maximum throughput, large models, fault tolerance | Averaging interval auto-tunes from divergence monitoring. Best for experienced users. |
//!
//! # Choosing a backend
//!
//! | Backend | When to use | Tradeoff |
//! |---------|-------------|----------|
//! | [`AverageBackend::Nccl`] | Default choice. NVLink/PCIe peer-to-peer. | In-place AllReduce, zero extra memory, hard sync at averaging point. |
//! | [`AverageBackend::Cpu`] | No NVLink, A/B testing, debugging, CPU-only setups | Params copied to CPU for averaging. No GPU blocks, but uses O(world_size * model_size) CPU RAM and adds latency from GPU-CPU-GPU round-trip. |
//!
//! Start with `Cadence` + `Nccl` for heterogeneous setups, `Sync` + `Nccl` for
//! homogeneous. Use `Cpu` backend when debugging or when NCCL is unavailable.
//!
//! # Safety guards
//!
//! - [`with_max_batch_diff`](DdpRunConfig::with_max_batch_diff): hard limit on how far any GPU can
//!   run ahead. Set to `0` for strict lockstep. Prevents catastrophic divergence
//!   with large batches or extreme speed ratios.
//! - [`ElChe`](super::ddp::ElChe) adaptive speed tracking with dead-zone hysteresis:
//!   tolerates thermal jitter while adapting quickly to sustained speed changes.
//! - NCCL abort handles: if a worker dies mid-collective, surviving workers are
//!   unblocked via `ncclCommAbort` instead of hanging forever.

mod worker;
mod coordinator;
mod orchestrator;
pub mod convergence;

pub use worker::*;
pub use coordinator::*;
pub use orchestrator::*;
pub use convergence::{
    ConvergenceAction, ConvergenceGuard, DivergenceReport, LambdaEstimator, LambdaSample,
    MsfGuard, NoGuard, TrendGuard,
};

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::rng::Rng;
use crate::tensor::{Device, Result, Tensor};

// ---------------------------------------------------------------------------
// Thread-local scalar accumulator for DDP train_fn
// ---------------------------------------------------------------------------

thread_local! {
    static SCALAR_ACCUM: RefCell<HashMap<String, (f64, usize)>> = RefCell::new(HashMap::new());
}

/// Record a named scalar value from inside a DDP worker's `train_fn`.
///
/// Values are accumulated per-epoch and reported at epoch boundaries.
/// The epoch-level value for each tag is the mean over all recorded values.
///
/// If called outside a DDP training context (e.g. on the main thread),
/// the values accumulate in the thread-local but are never drained.
///
/// ```ignore
/// // Inside train_fn:
/// flodl::record_scalar("ce_loss", ce.item()?);
/// flodl::record_scalar("kl_loss", kl.item()?);
/// flodl::record_scalar("accuracy", acc);
/// ```
pub fn record_scalar(name: &str, value: f64) {
    SCALAR_ACCUM.with(|acc| {
        let mut map = acc.borrow_mut();
        let entry = map.entry(name.to_string()).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    });
}

/// Drain the thread-local scalar accumulator, returning `(sum, count)` per tag.
///
/// Called by [`GpuWorker`] at epoch boundaries to package accumulated scalars
/// into the [`MetricsMsg`].
pub fn drain_scalars() -> HashMap<String, (f64, usize)> {
    SCALAR_ACCUM.with(|acc| std::mem::take(&mut *acc.borrow_mut()))
}


/// Checkpoint callback type: `(version, &model) -> Result<()>`.
///
/// Called on rank 0 after averaging events (multi-GPU) or at epoch boundaries
/// (single-GPU). Errors are logged but do not stop training.
pub type CheckpointFn<M> = Arc<dyn Fn(u64, &M) -> Result<()> + Send + Sync>;

/// Epoch callback type: `(epoch, &mut worker)`.
///
/// Called at the start of each epoch inside each worker thread, before
/// [`run_epoch_plan`](GpuWorker::run_epoch_plan). Use this for epoch-level
/// scheduling such as learning rate schedules, noise curricula, or dynamic
/// loss weights.
///
/// The closure itself must be `Send + Sync` (its captures cross thread boundaries),
/// but the `&mut GpuWorker<M>` reference stays thread-local.
///
/// **Note (Auto mode):** In [`ApplyPolicy::Async`] with heterogeneous GPUs, fast
/// ranks may be up to 1 epoch ahead of slow ranks. If `epoch_fn` mutates shared
/// state (e.g. noise schedule via atomics), the fast rank's write is visible to
/// the slow rank before it reaches that epoch. The delta between adjacent epochs
/// is typically negligible.
pub type EpochFn<M> = Arc<dyn Fn(usize, &mut GpuWorker<M>) + Send + Sync>;

/// Host-side per-epoch metrics callback type: `(metrics) -> Result<()>`.
///
/// Called once per epoch with the aggregated [`EpochMetrics`]: on the
/// coordinator thread for multi-GPU, on the main thread for the single-GPU
/// fallback. Use this to log progress, update a monitor dashboard, or
/// capture per-rank fields without dropping out of the chained
/// `Trainer::builder(...).run()?.join()?` shape into an explicit polling loop.
///
/// The metric is also pushed to the [`DdpHandle::next_metrics`] queue, so
/// callers can register `metrics_fn` *and* keep polling — both surfaces
/// receive the same `EpochMetrics`.
///
/// Errors returned from the callback are logged to stderr; training continues.
/// Surfacing the error to [`DdpHandle::join`] (early-stop semantics) is a
/// future enhancement.
///
/// **Single-GPU semantic:** `run_single` is synchronous — it runs all epochs
/// to completion before returning the [`DdpHandle`]. The callback fires
/// per-epoch as training progresses, identically to multi-GPU; explicit
/// pollers via [`DdpHandle::next_metrics`] receive all queued metrics
/// back-to-back after `run()` returns. A single GPU has nothing to be
/// async with, so this is the natural shape, not a limitation.
pub type MetricsFn = Arc<dyn Fn(&EpochMetrics) -> Result<()> + Send + Sync>;

// ---------------------------------------------------------------------------
// Deprecated aliases (backward compatibility)
// ---------------------------------------------------------------------------

/// Deprecated: renamed to [`DdpHandle`].
#[deprecated(since = "0.3.0", note = "Renamed to DdpHandle. Use Trainer::builder() to create.")]
pub type AsyncDdp = DdpHandle;

/// Deprecated: renamed to [`DdpBuilder`].
#[deprecated(since = "0.3.0", note = "Renamed to DdpBuilder. Use Trainer::builder() to create.")]
pub type AsyncDdpBuilder<F, M, G, O, T> = DdpBuilder<F, M, G, O, T>;

/// Deprecated: renamed to [`DdpRunConfig`].
#[deprecated(since = "0.3.0", note = "Renamed to DdpRunConfig")]
pub type AsyncDdpConfig = DdpRunConfig;

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

/// Trained parameters and buffers returned by [`DdpHandle::join()`].
///
/// Contains the averaged final state from all workers. Parameters are on CPU.
/// Buffers include running statistics (e.g. BatchNorm mean/var) needed for inference.
///
/// # Example
///
/// ```ignore
/// let state = ddp.join()?;
/// // state.params[i] corresponds to model.parameters()[i]
/// // state.buffers[i] corresponds to model.buffers()[i]
/// ```
#[derive(Clone, Debug)]
pub struct TrainedState {
    /// Averaged parameter tensors (CPU). Same order as `Module::parameters()`.
    pub params: Vec<Tensor>,
    /// Averaged buffer tensors (CPU). Same order as `Module::buffers()`.
    pub buffers: Vec<Tensor>,
}

/// Aggregated epoch metrics from all DDP workers.
///
/// Available via [`DdpHandle::poll_metrics()`], [`DdpHandle::next_metrics()`],
/// and the host-side [`crate::distributed::DdpBuilder::metrics_fn`] callback.
/// The coordinator aggregates per-rank [`MetricsMsg`] into this structure once
/// all ranks have reported for the same epoch; the same `EpochMetrics` reaches
/// the callback (if registered) and the polling queue, so both surfaces compose.
///
/// # Example: explicit polling
///
/// ```ignore
/// let handle = Trainer::builder(...).run()?;
/// while let Some(m) = handle.next_metrics() {
///     for (name, value) in &m.scalars {
///         monitor.record_scalar(name, *value);
///     }
/// }
/// let state = handle.join()?;
/// ```
///
/// # Example: chained `.run()?.join()?` with `metrics_fn`
///
/// ```ignore
/// Trainer::builder(model_factory, optim_factory, train_step)
///     .dataset(dataset).batch_size(32).num_epochs(N)
///     .metrics_fn(move |m| {
///         println!("epoch {}: loss={:.4}", m.epoch, m.avg_loss);
///         Ok(())
///     })
///     .run()?
///     .join()?;
/// ```
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    /// Epoch number (0-based).
    pub epoch: usize,
    /// Weighted-average scalar metrics across all ranks.
    /// Each value is the batch-weighted mean.
    pub scalars: HashMap<String, f64>,
    /// Per-rank scalar metrics (index = rank).
    pub per_rank: Vec<HashMap<String, f64>>,
    /// Average loss across all ranks (batch-weighted).
    pub avg_loss: f64,
    /// Wall-clock epoch time (ms), max across ranks.
    pub epoch_ms: f64,
    /// Per-rank throughput in samples/ms (index = rank). Computed from
    /// `share_complete_ms` (epoch start to end of rank's last batch),
    /// not from `epoch_ms`, to exclude post-completion sync-barrier idle.
    /// This is the honest capacity signal that the balancer should consume.
    pub per_rank_throughput: Vec<f64>,
    /// Per-rank batch share as fraction 0.0..1.0 (index = rank).
    pub per_rank_batch_share: Vec<f64>,
    /// Per-rank time on assigned work (ms), from epoch start to last batch
    /// finishing. Excludes post-completion sync wait. Source of `per_rank_throughput`.
    pub per_rank_share_complete_ms: Vec<f64>,
    /// Per-rank pure compute time (ms): sum of train_step durations.
    /// Diagnostic only; not used by the balancer.
    pub per_rank_compute_only_ms: Vec<f64>,
    /// Per-rank cumulative data-wait time (ms): time blocked waiting for
    /// the next batch. Diagnostic for prefetch tuning; not a balancer input.
    pub per_rank_data_starve_ms: Vec<f64>,
    /// CUDA device index per rank (for dashboard GPU tabs).
    pub device_indices: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Configuration enums
// ---------------------------------------------------------------------------

/// Controls WHEN parameter averaging occurs (the interval K).
///
/// All three modes run the same architecture; only the averaging trigger differs.
/// The interval K determines how many batches each GPU processes with its own
/// local optimizer before parameters are synchronized across replicas.
///
/// - `Sync`: K=1 (every batch). Equivalent to standard DDP. Best convergence
///   guarantees, but fast GPUs idle waiting for slow ones.
/// - `Cadence`: K=N (ElChe anchor count). The slow GPU anchors the cadence,
///   fast GPUs fill the wall time with extra batches. Recommended for
///   heterogeneous hardware (e.g. mixing GPU generations).
/// - `Async`: same proportional scheduling as Cadence (ElChe batch counts),
///   but with divergence correction: if replicas drift apart, the anchor
///   is nudged down (tighter sync). Differs from Cadence only in epoch
///   dispatch (per-rank vs broadcast) in non-progressive mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ApplyPolicy {
    /// Average after every batch (K=1). Equivalent to standard synchronous DDP.
    /// Lowest risk of model divergence. Fast GPUs wait at the collective barrier.
    Sync,
    /// Average every N batches where N is determined by ElChe's cadence strategy.
    /// The slow device sets the pace; fast devices process proportionally more
    /// batches per averaging window. Good default for mixed GPU setups.
    Cadence,
    /// Same proportional scheduling as Cadence, plus divergence correction:
    /// if parameter norms drift apart, ElChe's anchor is nudged down
    /// (tighter sync). Differs from Cadence only in epoch dispatch
    /// (per-rank in non-progressive, identical in progressive mode).
    Async,
}

/// Controls HOW parameter averaging is performed.
///
/// Orthogonal to [`ApplyPolicy`]. All combinations are valid, enabling A/B testing:
/// same model, same K, NCCL vs CPU. If loss curves match, the cheaper backend is
/// validated for your workload.
///
/// # NCCL vs CPU tradeoffs
///
/// | | NCCL | CPU |
/// |---|---|---|
/// | **Memory** | Zero extra (in-place) | O(world_size * model_size) CPU RAM |
/// | **Latency** | GPU-to-GPU DMA (NVLink or PCIe) | GPU->CPU->average->CPU->GPU round-trip |
/// | **Blocking** | All GPUs sync at collective barrier | No GPU ever blocks |
/// | **Fault tolerance** | Abort handles unblock stuck collectives | Coordinator timeout (5s) detects dead workers |
/// | **Buffer averaging** | Natural (AllReduce averages everything) | Explicit (buffers averaged with equal weight) |
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AverageBackend {
    /// NCCL AllReduce in-place on GPU params. Default and recommended.
    ///
    /// GPU-to-GPU DMA via NVLink or PCIe peer-to-peer. Zero extra memory.
    /// At the averaging point, all GPUs participate in a collective barrier:
    /// the fast GPU waits for the slow GPU to arrive. If a worker dies
    /// mid-collective, abort handles unblock the survivors.
    Nccl,
    /// CPU-mediated parameter averaging through the coordinator.
    ///
    /// Workers send parameter snapshots to the coordinator, which computes
    /// a weighted average on CPU, then distributes the result back. No GPU
    /// ever blocks on another GPU. Useful when NVLink/PCIe peer access is
    /// unavailable, for debugging, or for A/B comparison with the NCCL path.
    ///
    /// Uses O(world_size * model_size) CPU RAM for snapshot collection.
    /// Averaging time is measured and fed to ElChe so the overhead auto-tune
    /// accounts for the CPU round-trip cost.
    Cpu,
}

/// Configuration for framework-managed DDP training.
///
/// All fields have sensible defaults. Use the builder methods to customize.
#[derive(Clone, Debug)]
pub struct DdpRunConfig {
    /// ElChe overhead target (fraction of compute time). Default: 0.10.
    pub overhead_target: Option<f64>,
    /// Maximum anchor count (gradient staleness limit). Default: 200.
    pub max_anchor: Option<usize>,
    /// Initial ElChe anchor (batches before first sync). Default: 10.
    pub anchor: Option<usize>,
    /// Divergence threshold for the trend guardrail. Default: 0.05.
    pub divergence_threshold: Option<f64>,
    /// Disable the divergence guardrail entirely. Default: false (enabled).
    /// When true, ElChe's overhead auto-tune handles cadence alone.
    pub no_divergence_guard: bool,
    /// Maximum batch lead of fastest over slowest worker.
    /// `Some(0)` = strict lockstep. `None` = unlimited. Default: `None`.
    pub max_batch_diff: Option<usize>,
    /// Save a checkpoint every N global epochs.
    /// `None` = no checkpointing. Default: `None`.
    pub checkpoint_every: Option<usize>,
    /// Timeout for CPU averaging snapshot collection (seconds). Default: 5.
    /// Only applies to [`AverageBackend::Cpu`].
    pub snapshot_timeout_secs: u64,
    /// Explicit per-rank partition ratios (e.g. `[0.7, 0.3]`).
    ///
    /// When set, disables automatic throughput-based rebalancing.
    /// Ratios must sum to approximately 1.0. Length must match `world_size`.
    /// Use this when you know your hardware and want fixed data splits.
    pub partition_ratios: Option<Vec<f64>>,
    /// Enable progressive chunk dispatch for cold-start calibration.
    ///
    /// Instead of sending the full epoch partition upfront, the coordinator
    /// streams work in small chunks, adapting sizes to measured throughput.
    /// This eliminates the idle time on fast GPUs during epoch 0.
    ///
    /// Default: `None` (auto: true for Cadence/Async, false for Sync).
    pub progressive_dispatch: Option<bool>,
    /// Maximum gradient norm for per-worker clipping.
    ///
    /// When set, each worker clips its accumulated gradients (L2 norm)
    /// after backward and before the optimizer step. Ensures gradient
    /// spikes on any GPU are bounded before they propagate through
    /// AllReduce averaging.
    pub max_grad_norm: Option<f64>,
    /// Optional high-frequency system timeline for profiling DDP behavior.
    ///
    /// When set, the coordinator and workers inject training events (sync,
    /// epoch boundaries, anchor changes, throttle) into the timeline.
    pub timeline: Option<Arc<crate::monitor::Timeline>>,
    /// Maximum batches past the planned sync point any GPU may execute.
    ///
    /// Controls how aggressively fast GPUs stream into the next epoch's
    /// data when the current epoch's pool is exhausted. This is NOT the
    /// same as `max_batch_diff` (which limits divergence between GPUs).
    ///
    /// `None` = auto-tuned from convergence feedback: starts conservative
    /// (`max(2, total_batches / 100)` capped at 5), grows by +1 after
    /// each successful sync with good convergence, resets on divergence.
    ///
    /// Default: `None` (auto).
    pub max_overshoot: Option<usize>,
    /// LR scaling ratio for multi-GPU training. Default: `1.0`.
    ///
    /// Controls how much the learning rate is scaled with `world_size`.
    /// Formula: `lr_factor = 1.0 + (world_size - 1) * ratio`.
    ///
    /// - `1.0` (default): full linear scaling (Goyal et al., 2017).
    ///   With 2 GPUs, LR is doubled. Compensates for the LR schedule
    ///   advancing faster when global_step counts all GPUs' batches.
    /// - `0.0`: no scaling. Each GPU uses the base LR as-is.
    /// - `0.5`: half linear scaling. With 2 GPUs, LR *= 1.5.
    ///
    /// Tune this if convergence degrades at higher GPU counts.
    pub lr_scale_ratio: f64,
    /// Allow ElChe to relax the anchor upward on stable convergence. Default: `false`.
    ///
    /// When `false` (default), the anchor only changes via the overhead-based
    /// auto-tune in `el_che.report_timing`. This matches pre-relax-up
    /// behavior and is the reproducibility-friendly default.
    ///
    /// When `true`, each `Stable` convergence-guard verdict triggers
    /// `el_che.relax_anchor_up()`, growing the anchor toward `max_anchor` to
    /// reduce sync frequency on workloads where convergence is healthy.
    /// Opt in when measuring the relax-up policy specifically; aware that
    /// the periodic loosen → drift → tighten → recover cycle adds an
    /// implicit regularizer to the optimizer's gradient-averaging schedule.
    pub elche_relax_up: bool,
    /// EASGD elastic averaging weight (0.0 < α ≤ 1.0). Default: `None`
    /// (current behavior: full overwrite of local params with averaged
    /// consensus, equivalent to α=1.0).
    ///
    /// When set to a value in `(0, 1)`, the cpu-async load_averaged path
    /// applies an elastic blend instead of full overwrite:
    ///
    /// ```text
    /// W_local := (1 − α) · W_local + α · W_avg
    /// ```
    ///
    /// Preserves the local progress made between snapshot-time and
    /// averaging-completion (the "ahead-of-sync drift" that current
    /// cpu-async discards). Reference: Zhang, Choromanska, LeCun 2015,
    /// "Deep learning with Elastic Averaging SGD," NeurIPS 2015
    /// (<https://arxiv.org/abs/1412.6651>).
    ///
    /// Honored on the cpu-async path only (`AverageBackend::Cpu` +
    /// `ApplyPolicy::Async`). NCCL paths use in-place AllReduce(Avg) and
    /// have no equivalent overwrite step. Values outside `(0, 1]` are
    /// rejected at config time.
    pub easgd_alpha: Option<f64>,
}

impl Default for DdpRunConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl DdpRunConfig {
    /// Create a default config (all defaults).
    pub fn new() -> Self {
        DdpRunConfig {
            overhead_target: None,
            max_anchor: None,
            anchor: None,
            divergence_threshold: None,
            no_divergence_guard: false,
            max_batch_diff: None,
            checkpoint_every: None,
            snapshot_timeout_secs: 5,
            partition_ratios: None,
            progressive_dispatch: None,
            max_grad_norm: None,
            timeline: None,
            max_overshoot: None,
            lr_scale_ratio: 1.0,
            elche_relax_up: false,
            easgd_alpha: None,
        }
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    pub fn with_overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = Some(target);
        self
    }

    /// Set the maximum anchor count.
    pub fn with_max_anchor(mut self, max: usize) -> Self {
        self.max_anchor = Some(max);
        self
    }

    /// Set the initial anchor count.
    pub fn with_anchor(mut self, anchor: usize) -> Self {
        self.anchor = Some(anchor);
        self
    }

    /// Set the divergence threshold for the trend guardrail.
    pub fn with_divergence_threshold(mut self, threshold: f64) -> Self {
        self.divergence_threshold = Some(threshold);
        self
    }

    /// Disable the divergence guardrail. ElChe's overhead auto-tune
    /// handles cadence alone. Use when you know your workload is stable.
    pub fn with_no_divergence_guard(mut self) -> Self {
        self.no_divergence_guard = true;
        self
    }

    /// Set the maximum batch lead of fastest over slowest worker.
    ///
    /// `0` = strict lockstep (sync DDP behavior). Workers that exceed
    /// this lead are paused until the slowest catches up.
    pub fn with_max_batch_diff(mut self, max: usize) -> Self {
        self.max_batch_diff = Some(max);
        self
    }

    /// Set the maximum overshoot past the planned sync point.
    ///
    /// Controls cross-epoch streaming aggressiveness. When a GPU finishes
    /// its epoch partition, it may stream into the next epoch's data up to
    /// this many batches past ElChe's planned sync count.
    ///
    /// `0` disables cross-epoch streaming. Default: auto-tuned.
    pub fn with_max_overshoot(mut self, max: usize) -> Self {
        self.max_overshoot = Some(max);
        self
    }

    /// Save a checkpoint every N global epochs.
    ///
    /// Requires a `checkpoint_fn` to be set on the builder.
    /// Errors from the checkpoint function are logged but do not stop training.
    pub fn with_checkpoint_every(mut self, n: usize) -> Self {
        self.checkpoint_every = Some(n);
        self
    }

    /// Set the timeout for CPU averaging snapshot collection (seconds).
    ///
    /// Default: 5. Only applies to [`AverageBackend::Cpu`]. If not all worker
    /// snapshots arrive within this timeout, the averaging attempt is aborted
    /// and retried on the next cycle.
    pub fn with_snapshot_timeout(mut self, secs: u64) -> Self {
        self.snapshot_timeout_secs = secs;
        self
    }

    /// Set explicit per-rank partition ratios (e.g. `&[0.7, 0.3]`).
    ///
    /// Disables automatic throughput-based rebalancing. Ratios are normalized
    /// so they sum to 1.0. Length must match `world_size` at launch time.
    pub fn with_partition_ratios(mut self, ratios: &[f64]) -> Self {
        self.partition_ratios = Some(ratios.to_vec());
        self
    }

    /// Enable or disable progressive chunk dispatch.
    ///
    /// When enabled, the coordinator streams work in small chunks instead of
    /// sending full epoch partitions. This allows continuous throughput
    /// adaptation and eliminates cold-start idle time.
    ///
    /// Default: auto (true for Cadence/Async, false for Sync).
    pub fn with_progressive_dispatch(mut self, enabled: bool) -> Self {
        self.progressive_dispatch = Some(enabled);
        self
    }

    /// Set maximum gradient norm for per-worker clipping.
    ///
    /// Each worker clips accumulated gradients to this L2 norm after backward
    /// and before the optimizer step. Prevents gradient spikes on any GPU from
    /// propagating through AllReduce.
    pub fn with_max_grad_norm(mut self, max_norm: f64) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Attach a high-frequency system timeline for profiling DDP behavior.
    ///
    /// When set, the coordinator and workers inject training events
    /// (sync, epoch, anchor changes, throttle) into the timeline.
    pub fn with_timeline(mut self, tl: Arc<crate::monitor::Timeline>) -> Self {
        self.timeline = Some(tl);
        self
    }

    /// Set the LR scaling ratio for multi-GPU training.
    ///
    /// Formula: `lr_factor = 1.0 + (world_size - 1) * ratio`.
    /// Default: 1.0 (full linear scaling). Set to 0.0 to disable.
    pub fn with_lr_scale_ratio(mut self, ratio: f64) -> Self {
        self.lr_scale_ratio = ratio;
        self
    }

    /// Allow or suppress ElChe's anchor relax-up on stable convergence.
    ///
    /// Default: `false` (off). Set to `true` to enable: each `Stable`
    /// convergence-guard verdict will grow the anchor via
    /// `el_che.relax_anchor_up()`. Opt in when measuring the relax-up
    /// regime; the default keeps the anchor under overhead-based control
    /// alone, matching pre-relax-up behavior.
    pub fn with_elche_relax_up(mut self, enabled: bool) -> Self {
        self.elche_relax_up = enabled;
        self
    }

    /// Set the EASGD elastic averaging weight α. Must be in `(0, 1]`.
    /// `None` (default) is full overwrite (equivalent to α=1.0 with the
    /// fast copy_ path). Values in `(0, 1)` enable elastic blending on
    /// the cpu-async path; α=1.0 also enables blending but is functionally
    /// identical to the overwrite default. See `easgd_alpha` field docs
    /// for the formula and reference.
    pub fn with_easgd_alpha(mut self, alpha: f64) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "easgd_alpha must be in (0, 1], got {alpha}"
        );
        self.easgd_alpha = Some(alpha);
        self
    }
}

// ---------------------------------------------------------------------------
// Worker -> Coordinator messages
// ---------------------------------------------------------------------------

/// Message from a GPU worker to the coordinator on the timing channel.
///
/// Batch reports are lightweight (sent every batch for ElChe throughput tracking).
/// Exiting is sent exactly once, before the worker thread terminates, so the
/// coordinator never sends NCCL collectives to a dead worker.
#[derive(Clone, Debug)]
pub enum TimingMsg {
    /// Per-batch timing report.
    Batch {
        /// Which GPU sent this.
        rank: usize,
        /// Wall-clock time for this batch (ms).
        batch_ms: f64,
        /// Worker's local step counter (monotonically increasing).
        step_count: usize,
        /// L2 norm of all parameters (computed periodically, not every batch).
        param_norm: Option<f64>,
        /// Training loss for this batch (accumulated for monitoring).
        batch_loss: f64,
        /// Weight-space divergence from the most recent AllReduce:
        /// `||params_before - params_after|| / ||params_after||`.
        /// Only set in the post-sync ack message; `None` for regular batches.
        sync_divergence: Option<f64>,
    },
    /// Post-SyncNow acknowledgment: proves the worker completed the NCCL
    /// AllReduce, without being counted as a real training batch.
    ///
    /// Satisfies the coordinator's `nccl_ack` check (`step_count >
    /// nccl_sync_step`) without inflating `steps_since_avg`. Using
    /// [`TimingMsg::Batch`] here would add a phantom batch per sync per
    /// rank, inflating `global_step` and firing the LR scheduler early.
    SyncAck {
        /// Which GPU sent this.
        rank: usize,
        /// Worker's local step counter after the sync.
        step_count: usize,
        /// Weight-space divergence from the AllReduce:
        /// `||params_before - params_after|| / ||params_after||`.
        divergence: Option<f64>,
        /// Post-AllReduce consensus L2 norm `||params_after||`. Identical
        /// across ranks (all params identical post-AllReduce); the coordinator
        /// can take any rank's value. Used for longitudinal meta-velocity
        /// tracking. `None` when divergence is also `None`.
        post_norm: Option<f64>,
        /// Pre-AllReduce per-rank L2 norm `||params_before||_i`. With
        /// `divergence` and `post_norm` this gives the cosine-similarity /
        /// magnitude-shift decomposition (MSF/SWA directional vs magnitude
        /// split). `None` when divergence is also `None`.
        pre_norm: Option<f64>,
    },
    /// Worker is about to exit. Coordinator must stop including this rank
    /// in collectives before processing any further messages.
    Exiting {
        /// Which GPU is exiting.
        rank: usize,
    },
}

/// Epoch-end metrics sent from a GPU worker to the coordinator.
///
/// Fire-and-forget: worker sends this and immediately starts the next epoch.
#[derive(Clone, Debug, Default)]
pub struct MetricsMsg {
    /// Which GPU sent this.
    pub rank: usize,
    /// Epoch number (local to this worker).
    pub epoch: usize,
    /// Average loss over this epoch.
    pub avg_loss: f64,
    /// Number of batches processed in this epoch.
    pub batches_processed: usize,
    /// Wall-clock time for this epoch (ms). Includes any post-completion
    /// idle time waiting for collective sync. Kept for backwards-compatibility
    /// and as a coarse outer-bound timing; do not use as a balancer denominator
    /// for heterogeneous DDP — see `share_complete_ms`.
    pub epoch_ms: f64,
    /// Total samples processed this epoch (batches * batch_size).
    pub samples_processed: usize,
    /// Time the rank spent on its assigned work, from epoch start to its last
    /// batch finishing (ms). Includes data-pipeline waits (the rank's own
    /// pipeline limitation), excludes post-completion sync-barrier idle.
    /// This is the honest balancer denominator: tput = samples_processed
    /// / share_complete_ms reflects the rank's actual capacity.
    pub share_complete_ms: f64,
    /// Pure compute time within the epoch (sum of forward+backward+optimizer
    /// step durations, ms). Diagnostic only — not used by the balancer.
    /// Useful for capacity / saturation analysis.
    pub compute_only_ms: f64,
    /// Cumulative time the rank was blocked waiting for data (ms).
    /// On the prefetch path this is time spent in `recv()` on the prefetch
    /// channel; on the sync path this is time spent in `dataset.get_batch()`
    /// plus host-to-device transfer. Diagnostic only: surfacing this drives
    /// prefetch-tuning decisions, not balancer share allocation. Feeding
    /// it back to the balancer would create a contaminated control loop.
    pub data_starve_ms: f64,
    /// Named scalar metrics recorded via [`record_scalar()`] during this epoch.
    /// Each value is `(sum, count)` for computing the mean.
    pub scalars: HashMap<String, (f64, usize)>,
}

/// Parameter snapshot sent from a GPU worker to the coordinator (CPU averaging path only).
///
/// Contains cloned Tensor handles (Send+Sync via libtorch refcount).
#[derive(Clone)]
pub struct ParamSnapshot {
    /// Which GPU sent this.
    pub rank: usize,
    /// Current parameter tensors (on this worker's GPU device).
    pub params: Vec<Tensor>,
    /// Current buffer tensors (BatchNorm running stats, etc.).
    pub buffers: Vec<Tensor>,
    /// Number of batches processed since last averaging (for weighting).
    pub batch_count: usize,
}

// ---------------------------------------------------------------------------
// Coordinator -> Worker messages
// ---------------------------------------------------------------------------

/// Coordinator-computed epoch assignment for a single worker.
///
/// Contains the partition offset and size so the worker can deterministically
/// reconstruct its sample indices from the global permutation. The coordinator
/// computes consecutive offsets for all ranks, guaranteeing no gaps or overlaps.
#[derive(Clone, Debug)]
pub struct EpochPlan {
    /// Global epoch number (0-based).
    pub epoch: usize,
    /// Start offset into the global permutation for this rank.
    pub partition_offset: usize,
    /// Number of samples assigned to this rank for this epoch.
    pub partition_size: usize,
}

/// Averaged parameters sent from the coordinator to a GPU worker (CPU averaging path only).
///
/// Contains pinned CPU tensors. Worker copies them into its Variables via `copy_(non_blocking=true)`.
#[derive(Clone, Debug)]
pub struct AveragedParams {
    /// Averaged parameter tensors (pinned CPU memory).
    pub params: Vec<Tensor>,
    /// Averaged buffer tensors.
    pub buffers: Vec<Tensor>,
    /// Monotonically increasing version number.
    pub version: u64,
}

/// Control signals from the coordinator to a GPU worker.
#[derive(Debug)]
pub enum ControlMsg {
    /// \[CPU path\] Request parameter snapshot for averaging.
    RequestParams,
    /// \[CPU path\] Deliver averaged parameters.
    Update(AveragedParams),
    /// \[NCCL path\] Trigger in-place AllReduce on this worker's own params.
    /// Worker runs AllReduce on comm_stream and records CudaEvent.
    SyncNow,
    /// Begin processing a new epoch with the given partition assignment.
    ///
    /// The coordinator computes partition sizes based on throughput ratios and
    /// sends consecutive, non-overlapping assignments to each worker. Workers
    /// reconstruct their sample indices from the global permutation using the
    /// plan's offset and size.
    StartEpoch(EpochPlan),
    /// Worker is too far ahead: block until the next real command arrives.
    /// Sent when the worker's batch lead exceeds `ElChe::max_batch_diff`.
    Throttle,
    /// Update the worker's global step count after averaging.
    ///
    /// `global_step` = cumulative total batches across all GPUs up to this
    /// sync point. Workers use this to compute per-batch LR:
    /// `scheduler.lr(global_step + steps_since_avg)`.
    SetGlobalStep(usize),
    /// Save a checkpoint from rank 0 after averaging.
    Checkpoint {
        /// Version number (averaging event count in multi-GPU, epoch in single-GPU).
        version: u64,
    },
    /// Shut down this worker.
    Shutdown,
}

// ---------------------------------------------------------------------------
// Initial setup
// ---------------------------------------------------------------------------

/// Configuration passed to a GPU worker at spawn time.
///
/// All fields are Send. The worker uses these to construct its thread-local
/// Graph, Optimizer, DataLoader, and streams inside the spawned thread.
#[derive(Clone)]
pub struct WorkerConfig {
    /// This worker's rank (0..world_size).
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    /// The CUDA device this worker operates on.
    pub device: Device,
    /// Initial parameter tensors in pinned CPU memory (from rank 0 snapshot).
    /// Worker copies these into its Variables at startup.
    pub initial_params: Vec<Tensor>,
    /// Initial buffer tensors in pinned CPU memory.
    pub initial_buffers: Vec<Tensor>,
    /// Total number of samples in the dataset.
    pub total_samples: usize,
    /// Batch size.
    pub batch_size: usize,
    /// RNG seed for deterministic shuffling.
    pub seed: u64,
    /// Maximum gradient norm for clipping (None = no clipping).
    pub max_grad_norm: Option<f64>,
    /// EASGD elastic averaging weight (0, 1]. `None` = full overwrite of
    /// local params with averaged consensus on the cpu-async path (current
    /// behavior). When set, [`super::worker::GpuWorker::load_averaged`]
    /// blends `W_local := (1-α)·W_local + α·W_avg` instead. See
    /// [`DdpRunConfig::easgd_alpha`] for details.
    pub easgd_alpha: Option<f64>,
    /// Optional system timeline for high-frequency profiling.
    pub timeline: Option<Arc<crate::monitor::Timeline>>,
    /// Training policy (Sync/Cadence/Async). Used to gate divergence measurement:
    /// Sync mode skips weight-space divergence (near-zero by construction).
    pub policy: ApplyPolicy,
}

// ---------------------------------------------------------------------------
// Partition generation
// ---------------------------------------------------------------------------

/// Generate a deterministic partition of sample indices from a global permutation.
///
/// All ranks sharing the same `(epoch, seed)` produce the same global permutation.
/// The coordinator computes consecutive `(offset, size)` pairs for each rank so
/// that slices are non-overlapping and cover the full dataset.
///
/// **Non-overlapping guarantee:** the coordinator assigns consecutive offsets
/// that sum to `total`, so all slices are disjoint by construction.
fn make_partition(
    offset: usize,
    size: usize,
    total: usize,
    epoch: usize,
    seed: u64,
) -> Vec<usize> {
    // Deterministic global shuffle (same seed = same permutation for all ranks)
    let mut rng = Rng::seed(seed.wrapping_add(epoch as u64));
    let mut all: Vec<usize> = (0..total).collect();
    rng.shuffle(&mut all);

    // This rank's consecutive slice
    let end = (offset + size).min(total);
    all[offset..end].to_vec()
}

#[cfg(test)]
#[allow(deprecated)]
#[path = "tests.rs"]
mod tests;
