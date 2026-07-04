//! DDP run mode: the `Trainer::builder()` / [`DdpBuilder`] entry, the
//! [`DdpHandle`] it returns, the per-rank [`GpuWorker`], and the shared
//! cadence config ([`ApplyPolicy`] / [`AverageBackend`] / `DdpRunConfig`).
//!
//! `DdpHandle::launch` dispatches by topology: a single visible device runs
//! the inline single-host fallback; 2+ devices or an active cluster overlay
//! auto-promote to the process-per-rank cluster path (launcher / controller /
//! `cluster_coordinator` / `cluster_worker`), where each rank process drives a
//! [`GpuWorker`] over the wire. The in-process thread-per-GPU engine that once
//! lived here was removed; thread-based multi-GPU is available only as the
//! lower-level `Ddp::wrap` primitive.
//!
//! Two orthogonal knobs control averaging cadence: [`ApplyPolicy`] (when to
//! average) and [`AverageBackend`] (how to average).
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
//!     .elche(ElCheConfig::nccl_cadence())
//!     .checkpoint_every(5)
//!     .checkpoint_fn(|ver, g| g.save_checkpoint(&format!("ckpt_v{ver}.fdl")))
//!     .run()?;
//!
//! let state = handle.join()?;
//! // state.params[i] corresponds to model.parameters()[i]
//! // state.buffers[i] corresponds to model.buffers()[i]
//! ```
//!
//! # Architecture (process-per-rank)
//!
//! ```text
//! Rank process 0:  create model+optim+dataset -> [fwd -> bwd -> step -> repeat]
//! Rank process 1:  create model+optim+dataset -> [fwd -> bwd -> step -> repeat]
//! Controller:      collect timing/metrics -> trigger param averaging -> monitor divergence
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
pub(crate) use worker::NcclAbortSlot;
mod orchestrator;
mod shared;
pub mod convergence;

pub use worker::*;
pub(crate) use shared::{
    aggregate_epoch_metrics, equal_sizes, ratio_to_sizes, throughput_sizes,
};
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
/// Called on the rank selected by [`EpochCallbackPolicy`] (default
/// `Fastest`) at checkpoint events (multi-GPU) or at epoch boundaries
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

/// Cadence for invoking the user-supplied [`EvalFn`].
///
/// Controls how often the framework dispatches an eval pass to the
/// rank chosen by [`EpochCallbackPolicy`]. Triggered from the
/// controller's `dispatch_epoch` on the configured epoch boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvalCadence {
    /// Fire eval every `n` epochs. `n == 0` is treated as "never".
    Epochs(usize),
}

/// Eval callback: receives `&model` and the held-out `&dyn BatchDataSet`,
/// returns the aggregated scalar metric for the eval pass. The user
/// implements the batch iteration loop — framework just hands over
/// the model + dataset and consumes the result.
///
/// Framework guarantees:
/// - Fires on the rank selected by [`EpochCallbackPolicy`].
/// - Sync-aligned: the chosen rank's model is at its post-AllReduce /
///   post-EASGD-blend state when invoked.
/// - `model.eval()` is called before and `model.train()` after the
///   user's closure; no explicit mode flip needed inside.
pub type EvalFn<M> = std::sync::Arc<
    dyn Fn(&M, &dyn crate::data::BatchDataSet) -> Result<f64> + Send + Sync,
>;

/// Receiver for the [`EvalFn`] scalar result on the controller side.
/// Mirrors [`MetricsFn`]'s shape — fires after the chosen rank's eval
/// metric flows back over `EvalResult`.
pub type EvalResultFn = std::sync::Arc<
    dyn Fn(usize, f64) -> Result<()> + Send + Sync,
>;

/// Scheduler factory type: `(world_size) -> Arc<dyn Scheduler>`.
///
/// Called once per rank-process (or once per worker thread in the threaded
/// path) to construct the per-batch LR scheduler. The `world_size` argument
/// is provided so user-supplied factories can scale base LR by replica
/// count (Goyal et al. linear-scaling) without re-implementing the math.
///
/// In cluster mode, every rank builds an identical scheduler from this
/// factory; the controller drives synchronization by broadcasting
/// `SetGlobalStep` after each averaging cycle. Schedulers are pure
/// functions of step (`fn lr(&self, step: usize) -> f64`), so every rank
/// computes the same LR for the same input — equivalent to broadcasting
/// LR directly but with one `u64` per averaging cycle instead of one
/// `Vec<f64>` per param group.
pub type SchedulerFn = Box<dyn Fn(usize) -> Arc<dyn crate::nn::Scheduler> + Send + Sync>;

/// Which rank fires user-supplied per-epoch callbacks (`epoch_fn`,
/// `checkpoint_fn`, and future `eval_fn`).
///
/// One logical epoch transition produces one callback invocation, on
/// the rank selected by this policy. The cluster looks like a single
/// meta-GPU from the user's perspective; firing the same callback on
/// every rank would multiply side effects (N file writes, N eval
/// passes, etc.) for no benefit.
///
/// Default is [`Self::Fastest`] — on heterogeneous rigs the fastest
/// rank has the most idle time at sync barriers, so eval / save / log
/// runs as free compute. On a single-GPU run the only rank is
/// trivially the fastest, so `Fastest` collapses to running on that
/// rank — no special-case needed. Pin to a specific rank with
/// [`Self::Rank`] when the research convention demands it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EpochCallbackPolicy {
    /// Fire on the explicitly-named **global rank** — the
    /// cluster-wide rank index in `[0, world_size)`, where ranks are
    /// assigned sequentially by worker order in the cluster topology
    /// (worker 0 owns ranks `[0..N0)`, worker 1 owns `[N0..N0+N1)`,
    /// etc.). On a 4-rank cluster across two 2-GPU hosts, `Rank(0)`
    /// fires on the first rank of the first worker host, `Rank(3)`
    /// fires on the last rank of the last host. Loud-errors at
    /// builder validation if `n >= world_size`.
    Rank(usize),
    /// Fire on the rank with the lowest `smoothed_ms_per_batch`
    /// (controller-resolved via ElChe). Sticky within a run: the
    /// chosen rank is re-selected only on rank death. Honors
    /// heterogeneous-DDP intuition — fastest rank has the most idle
    /// time at sync barriers, so callbacks are "free compute". On
    /// single-GPU runs the only rank trivially satisfies "fastest".
    #[default]
    Fastest,
}

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

impl From<EpochMetrics> for crate::distributed::wire::EpochMetricsWire {
    fn from(m: EpochMetrics) -> Self {
        crate::distributed::wire::EpochMetricsWire {
            epoch: m.epoch as u64,
            scalars: m.scalars,
            per_rank: m.per_rank,
            avg_loss: m.avg_loss,
            epoch_ms: m.epoch_ms,
            per_rank_throughput: m.per_rank_throughput,
            per_rank_batch_share: m.per_rank_batch_share,
            per_rank_share_complete_ms: m.per_rank_share_complete_ms,
            per_rank_compute_only_ms: m.per_rank_compute_only_ms,
            per_rank_data_starve_ms: m.per_rank_data_starve_ms,
            device_indices: m.device_indices,
        }
    }
}

impl From<crate::distributed::wire::EpochMetricsWire> for EpochMetrics {
    fn from(w: crate::distributed::wire::EpochMetricsWire) -> Self {
        EpochMetrics {
            epoch: w.epoch as usize,
            scalars: w.scalars,
            per_rank: w.per_rank,
            avg_loss: w.avg_loss,
            epoch_ms: w.epoch_ms,
            per_rank_throughput: w.per_rank_throughput,
            per_rank_batch_share: w.per_rank_batch_share,
            per_rank_share_complete_ms: w.per_rank_share_complete_ms,
            per_rank_compute_only_ms: w.per_rank_compute_only_ms,
            per_rank_data_starve_ms: w.per_rank_data_starve_ms,
            device_indices: w.device_indices,
        }
    }
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
#[non_exhaustive]
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

impl ApplyPolicy {
    /// Whether this policy runs on a single step-clock with a HARD reduce /
    /// epoch barrier: the coordinator never hands a rank a step that crosses
    /// a barrier, so the fast rank is HELD at its window until the reduce
    /// resets `steps_since_avg`, and no rank crosses an epoch boundary ahead
    /// of the cohort. True for `Sync` and `Cadence`.
    ///
    /// `Async` is the one regime allowed bounded lookahead (overrunning its
    /// window by `max_overshoot`), so it opts out.
    ///
    /// This is a property of the PACING policy alone — it is independent of
    /// [`AverageBackend`] (whether the reduce moves over NCCL or CPU sockets
    /// is transport, orthogonal to pacing). Gating these barriers on the
    /// backend instead conflates the two axes: it silently means "NCCL => no
    /// pacing", which is correct only because `Async` happens to be CPU-only
    /// and is flatly wrong for NCCL `Cadence` (the fast rank then streams
    /// across every epoch and the cohort wedges).
    pub fn is_barrier_paced(&self) -> bool {
        matches!(self, ApplyPolicy::Sync | ApplyPolicy::Cadence)
    }
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
#[non_exhaustive]
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
    /// The DDP coordination/convergence STRATEGY — the single source of
    /// truth for mode (canonical), cadence tuning (anchor / max_anchor /
    /// min_anchor / overhead_target / max_batch_diff), the convergence
    /// guard (`convergence_guard` override + `divergence_threshold` /
    /// `no_divergence_guard` primitives), `partition_ratios`,
    /// `easgd_alpha`, `meta_controller`, and `max_overshoot`. The
    /// builder's `policy`/`backend` are reconciled into `elche.mode` at
    /// build via `ElCheMode::from_parts` (the inverse of `mode.split()`).
    /// Everything else on `DdpRunConfig` is run-scope / topology.
    pub elche: crate::distributed::ElCheConfig,
    /// Save a checkpoint every N global epochs.
    /// `None` = no checkpointing. Default: `None`.
    pub checkpoint_every: Option<usize>,
    /// Timeout for CPU averaging snapshot collection (seconds). Default: 5.
    /// Only applies to [`AverageBackend::Cpu`].
    pub snapshot_timeout_secs: u64,
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
    /// Checkpoint bundle stem for the cluster-mode
    /// save-on-unrecoverable-failure path. When set on a cluster
    /// run, workers persist a `<save_path>.fdl` / `.optim` /
    /// `.meta.json` bundle on
    /// `ShutdownWithSave`
    /// receipt; see [`crate::distributed::CheckpointBundle`].
    /// Required for the via-coord cluster orchestrator entry;
    /// optional for non-cluster builds.
    pub save_path: Option<String>,

    /// Threshold for declaring a cluster run unrecoverable. When the
    /// dead-rank count reaches this limit, the coord broadcasts
    /// `ShutdownWithSave` to all survivors. `None` = no user-configured
    /// threshold; backend hard limits still apply (NCCL needs 2+
    /// survivors; CPU needs at least 1). Only honored on cluster-mode
    /// runs; non-cluster builds ignore this field.
    pub max_failure: Option<crate::distributed::max_failure::MaxFailureThreshold>,

    /// Cluster-mode heartbeat staleness threshold (seconds). If a
    /// rank's last `TimingMsg` frame is older than this, the
    /// controller declares the rank dead and triggers the
    /// elastic-membership / max_failure flow. `None` = use the
    /// controller's built-in default (currently 30s). Only honored on
    /// cluster-mode via_coord runs.
    pub heartbeat_timeout_secs: Option<u64>,

    /// Which rank fires user-supplied per-epoch callbacks (`epoch_fn`,
    /// `checkpoint_fn`, `eval_fn`). See [`EpochCallbackPolicy`] for the
    /// variants. Default [`EpochCallbackPolicy::Fastest`].
    pub epoch_callback_policy: EpochCallbackPolicy,

    /// Cadence (in epochs) for the user-supplied `eval_fn`. `Some(n)`
    /// triggers an eval dispatch every `n` epochs from the controller's
    /// `dispatch_epoch`. `None` or `0` disables. Builder sugar:
    /// [`DdpBuilder::eval_every`] (accepts [`EvalCadence`]).
    pub eval_every_epochs: Option<usize>,

    /// Checkpoint bundle stem for resume. When set, the cluster
    /// orchestrator reads `<stem>.meta.json` at `.run()` time, seeds
    /// the controller with the saved trajectory state (epoch,
    /// global_step, sync_round, ElChe state including TrendGuard
    /// history), and kicks the launcher off at `meta.epoch` instead of
    /// `0`.
    ///
    /// Model parameters and optimizer state are NOT auto-loaded from
    /// the bundle by this field — the user's `model_factory` /
    /// `optim_factory` closures are the right place for that (call
    /// [`crate::nn::load_checkpoint_file`] /
    /// [`crate::nn::optim::Stateful::load_state_file`] inside them).
    /// This field carries the controller-side trajectory only.
    ///
    /// `None` = fresh run. Builder sugar: [`DdpBuilder::resume_from`].
    pub resume_from: Option<String>,

    /// Arm a one-shot coverage-granular checkpoint at the first reduce
    /// where the cohort reaches this epoch. Progressive modes only
    /// (Cadence / Async — a Sync run has no chunk pools to snapshot).
    /// Pairs with [`Self::save_path`] for the bundle stem: the forged
    /// consensus model lands in `<stem>.fdl` and the trajectory +
    /// data-coverage in `<stem>.meta.json`. `None` = no mid-run
    /// checkpoint. Builder sugar: [`DdpBuilder::checkpoint_at_epoch`].
    pub checkpoint_at_epoch: Option<usize>,
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
            elche: crate::distributed::ElCheConfig::default(),
            checkpoint_every: None,
            snapshot_timeout_secs: 5,
            progressive_dispatch: None,
            max_grad_norm: None,
            timeline: None,
            lr_scale_ratio: 1.0,
            save_path: None,
            max_failure: None,
            heartbeat_timeout_secs: None,
            epoch_callback_policy: EpochCallbackPolicy::default(),
            eval_every_epochs: None,
            resume_from: None,
            checkpoint_at_epoch: None,
        }
    }

    /// Resume a cluster run from a previously-saved checkpoint bundle.
    ///
    /// `stem` is the path stem used at save time (the value passed to
    /// [`Self::with_save_path`] / [`DdpBuilder::save_path`]). The
    /// orchestrator reads `<stem>.meta.json` at `.run()` time and seeds
    /// the controller with the saved trajectory state. See
    /// [`Self::resume_from`] for details on what is and isn't restored.
    pub fn with_resume_from(mut self, stem: impl Into<String>) -> Self {
        self.resume_from = Some(stem.into());
        self
    }

    /// Arm a one-shot coverage-granular checkpoint at the given epoch.
    /// Pairs with [`Self::with_save_path`]. See [`Self::checkpoint_at_epoch`].
    pub fn with_checkpoint_at_epoch(mut self, epoch: usize) -> Self {
        self.checkpoint_at_epoch = Some(epoch);
        self
    }

    /// Override which rank fires user-supplied per-epoch callbacks.
    /// See [`EpochCallbackPolicy`]. Default is `Fastest`.
    pub fn with_epoch_callback_policy(mut self, policy: EpochCallbackPolicy) -> Self {
        self.epoch_callback_policy = policy;
        self
    }

    /// Set the cluster-mode heartbeat staleness threshold (seconds).
    /// See [`Self::heartbeat_timeout_secs`].
    pub fn with_heartbeat_timeout_secs(mut self, secs: u64) -> Self {
        self.heartbeat_timeout_secs = Some(secs);
        self
    }

    /// Set the checkpoint bundle stem for cluster-mode unrecoverable-
    /// failure persistence. See
    /// [`crate::distributed::CheckpointBundle`] for the layout.
    pub fn with_save_path(mut self, path: impl Into<String>) -> Self {
        self.save_path = Some(path.into());
        self
    }

    /// Set the unrecoverable-failure threshold for cluster mode.
    pub fn with_max_failure(
        mut self,
        threshold: crate::distributed::max_failure::MaxFailureThreshold,
    ) -> Self {
        self.max_failure = Some(threshold);
        self
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    pub fn with_overhead_target(mut self, target: f64) -> Self {
        self.elche.overhead_target = Some(target);
        self
    }

    /// Set the maximum anchor count.
    pub fn with_max_anchor(mut self, max: usize) -> Self {
        self.elche.max_anchor = Some(max);
        self
    }

    /// Set the minimum anchor count (auto-tune floor).
    ///
    /// Forces the overhead auto-tune above its natural equilibrium. Combined
    /// with `with_max_anchor(min)` (same value), pins the anchor at a fixed
    /// cadence — useful for fixed-k experiments. The convergence guard and
    /// divergence nudge-down paths BYPASS this floor; pair with
    /// `with_convergence_guard(NoGuard)` + `with_no_divergence_guard()` for
    /// truly hard pinning.
    pub fn with_min_anchor(mut self, min: usize) -> Self {
        self.elche.min_anchor = Some(min);
        self
    }

    /// Set the initial anchor count.
    pub fn with_anchor(mut self, anchor: usize) -> Self {
        self.elche.anchor = anchor;
        self
    }

    /// Set the divergence threshold for the trend guardrail.
    pub fn with_divergence_threshold(mut self, threshold: f64) -> Self {
        self.elche.divergence_threshold = Some(threshold);
        self
    }

    /// Disable the divergence guardrail. ElChe's overhead auto-tune
    /// handles cadence alone. Use when you know your workload is stable.
    pub fn with_no_divergence_guard(mut self) -> Self {
        self.elche.no_divergence_guard = true;
        self
    }

    /// Set the maximum batch lead of fastest over slowest worker.
    ///
    /// `0` = strict lockstep (sync DDP behavior). Workers that exceed
    /// this lead are paused until the slowest catches up.
    pub fn with_max_batch_diff(mut self, max: usize) -> Self {
        self.elche.max_batch_diff = Some(max);
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
        self.elche.max_overshoot = Some(max);
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

    /// Fire the user-supplied `eval_fn` every `n` epochs from the
    /// controller's `dispatch_epoch`. `n == 0` disables. Builder
    /// sugar [`DdpBuilder::eval_every`] takes the [`EvalCadence`]
    /// enum and forwards the integer here.
    pub fn with_eval_every_epochs(mut self, n: usize) -> Self {
        self.eval_every_epochs = if n == 0 { None } else { Some(n) };
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
        self.elche.partition_ratios = Some(ratios.to_vec());
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
        self.elche.relax_up = enabled;
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
        self.elche.easgd_alpha = Some(alpha);
        self
    }

    /// Enable the LR-aware meta-controller above ElChe.
    ///
    /// On by default. See the `meta_controller` field for behavior
    /// and `crate::distributed::lr_event_meta` for the design.
    pub fn with_meta_controller(mut self, enabled: bool) -> Self {
        self.elche.meta_controller = enabled;
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
pub(crate) enum TimingMsg {
    /// Per-batch timing report.
    Batch {
        /// Which GPU sent this.
        rank: usize,
        /// Compute-only wall-clock time for this batch (ms): the `train_step`.
        batch_ms: f64,
        /// Per-batch DATA wall (ms): prefetch/H2D stall (prefetch path) or
        /// dataset fetch+to-device (sync path). `batch_ms + data_ms` is the
        /// rank's realized DELIVERED per-batch wall; the coordinator
        /// accumulates it continuously (race-free, like `batch_ms`).
        data_ms: f64,
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
    /// `TimingMsg::Batch` here would add a phantom batch per sync per
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
    /// Per-batch learning rate snapshot from a worker, used by the LR-aware
    /// meta-controller to detect sharp drops between averaging cycles.
    ///
    /// Cheap fire-and-forget message: just a `(rank, lr)` pair. The
    /// coordinator caches the most recent value per rank and feeds it into
    /// [`crate::distributed::lr_event_meta::LrEventMeta::observe`] each
    /// averaging cycle. Workers can choose to emit only on LR change or on
    /// every batch — receiver is idempotent.
    LrUpdate {
        /// Which GPU sent this.
        rank: usize,
        /// Current optimizer learning rate.
        lr: f64,
    },
    /// Periodic liveness signal from a cluster worker's heartbeat
    /// thread. Cluster-only; OLD threaded path never emits these.
    /// See `Heartbeat` for
    /// the failure-detection rationale.
    Heartbeat {
        /// Which rank sent this.
        rank: usize,
        /// Worker's local step counter at emission time. Diagnostic.
        step_count: usize,
    },
    /// Cluster-mode "snapshot ready, entering AllReduce barrier"
    /// marker emitted by the worker's CPU-averaging param bridge.
    /// See `SnapshotReady`.
    SnapshotReady {
        /// Which rank sent this.
        rank: usize,
    },
    /// Response from the chosen surviving rank to coord's
    /// `ControlMsg::RequestNewNcclId`: 128 raw bytes of a freshly
    /// generated `NcclUniqueId`.
    NewNcclIdGenerated {
        /// Sender rank (so coord validates against its request).
        rank: usize,
        /// 128 bytes of NCCL unique-id.
        uid_bytes: Vec<u8>,
    },
    /// Eval result from the chosen rank back to the coord. See
    /// `EvalResult`.
    EvalResult {
        rank: usize,
        schedule_id: u64,
        epoch: u64,
        metric: f64,
        elapsed_ms: f64,
        error: Option<String>,
    },
    /// Checkpoint result from the role rank back to the coord. See
    /// `CheckpointResult`.
    /// Reported on success and failure; the coord decides the next
    /// action (retry on different live rank, give up + exhaust, time
    /// exclusion from `wall_ms_accum`).
    CheckpointResult {
        rank: usize,
        version: u64,
        elapsed_ms: f64,
        error: Option<String>,
    },
    /// Post-fire notice from the rank that ran `epoch_fn`. See
    /// `EpochFnElapsed`.
    /// Reported once per `epoch_fn` invocation; the coord time-excludes
    /// it from `wall_ms_accum[rank]` and updates
    /// `last_epoch_fn_elapsed_ms_ewma` for callback-aware scheduling.
    EpochFnElapsed {
        rank: usize,
        epoch: usize,
        elapsed_ms: f64,
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
pub(crate) enum ControlMsg {
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
    /// Mid-epoch partition extension. Coord-emitted when redistributing
    /// a freshly-dead rank's un-processed samples onto survivors so the
    /// epoch still processes its intended sample count. Worker appends
    /// the resolved indices to its in-flight `partition` Vec; the
    /// epoch loop re-checks `partition.len()` each iteration so the
    /// new batches are processed before declaring the epoch complete.
    ExtendPartition {
        /// Offset into the global epoch permutation where the new
        /// slice starts (resolved via the same `make_partition` call
        /// the worker would use for `StartEpoch`).
        partition_offset: usize,
        /// Number of additional sample indices to append.
        partition_size: usize,
    },
    /// Coord-emitted notification that a peer rank has been declared
    /// dead. The cluster-worker's inbound bridge converts this into
    /// a local-ledger update so the NCCL watchdog thread can react;
    /// it is NOT dispatched to the inner GpuWorker because the inner
    /// is typically blocked in an in-flight NCCL collective and
    /// cannot service control messages until the watchdog aborts the
    /// comm. See `DeclareDead`. Fieldless: the inbound bridge reads the
    /// `ControlMsgWire::DeclareDead { rank }` payload directly into the
    /// local ledger; this worker-facing token carries no payload because
    /// the inner GpuWorker never consumes it.
    DeclareDead,
    /// Coord-emitted directive to rebuild the local NCCL comm with
    /// the shrunken cohort. Sent after one or more `DeclareDead`s.
    /// See `NewNcclSession`. Fieldless for the same reason as
    /// [`DeclareDead`](Self::DeclareDead): the bridge stages the
    /// `ControlMsgWire::NewNcclSession` payload into the session mailbox;
    /// the inner GpuWorker never reads it off this token.
    NewNcclSession,
    /// Coord-emitted request to generate a fresh NCCL unique-id and
    /// ship its bytes back via the timing channel
    /// (`TimingMsg::NewNcclIdGenerated`). Only one rank receives
    /// this per re-rendezvous cycle (the lowest-numbered survivor).
    /// See `RequestNewNcclId`.
    RequestNewNcclId,
    /// Worker is too far ahead: block until the next real command arrives.
    /// Sent when the worker's batch lead exceeds `ElChe::max_batch_diff`.
    Throttle,
    /// Update the worker's global step count after averaging.
    ///
    /// `global_step` = cumulative total batches across all GPUs up to this
    /// sync point. Workers use this to compute per-batch LR:
    /// `scheduler.lr(global_step + steps_since_avg)`.
    SetGlobalStep(usize),
    /// Coord-emitted directive to persist a checkpoint bundle for the
    /// given `version`. Targeted: only the rank whose `rank ==
    /// target_rank` runs its `checkpoint_fn`; other ranks receiving
    /// this frame silently no-op. The coord owns role assignment
    /// (sticky `checkpoint_role` with failover on rank death or
    /// `CheckpointResult.error`); the worker never decides whether
    /// it is the checkpointer.
    ///
    /// Mirrors `Checkpoint`.
    /// In threaded DDP, the coord still broadcasts to every worker's
    /// mpsc channel (same as before) — the `target_rank` field tells
    /// each worker to no-op unless it matches its own rank, preserving
    /// the single-checkpointer semantic without per-worker dispatch.
    Checkpoint {
        /// Version (averaging event count in multi-GPU, epoch in single-GPU).
        version: u64,
        /// Rank that should execute `checkpoint_fn`. `usize::MAX` is
        /// reserved for v2 controller-as-checkpointer (CPU-async
        /// mode); the worker treats it as "not me" and no-ops.
        target_rank: usize,
    },
    /// Run the user's [`EvalFn`] on the rank's current model + eval
    /// dataset. Targeted: only the rank whose `rank == target_rank`
    /// runs; others silently no-op. Mirrors
    /// `ExecuteEvalCallback`.
    ExecuteEvalCallback {
        schedule_id: u64,
        epoch: u64,
        target_rank: usize,
    },
    /// Coord-pushed notification that the rank designated to fire the
    /// user-supplied `epoch_fn` has been resolved. Worker stores this
    /// in its local `epoch_callback_role` and consults it at every
    /// epoch transition. Mirrors
    /// `SetEpochCallbackRole`.
    SetEpochCallbackRole {
        rank: usize,
    },
    /// Shut down this worker.
    Shutdown,
    /// Persist a checkpoint bundle to the configured `save_path` then
    /// shut down. Emitted by the cluster coord when the run is
    /// unrecoverable (max_failure threshold breached, or NCCL cohort
    /// below 2 ranks). Workers write
    /// [`crate::distributed::CheckpointBundle`] members; rank 0 is the
    /// canonical writer for the model + meta files. See
    /// `ShutdownWithSave`.
    ShutdownWithSave {
        /// Why the cluster is saving + exiting; recorded in the
        /// `.meta.json` for post-mortem inspection.
        reason: crate::distributed::checkpoint_meta::SaveReason,
    },
    /// Coord-broadcast aggregated [`EpochMetrics`] for the just-completed
    /// epoch. Each rank's worker absorbs this into its local `Graph` so
    /// `latest_metrics()` / `graph_gpu_metrics()` surface the global
    /// cross-rank view (user-facing UX parity: `monitor.log(&model)`
    /// shows the same aggregated view regardless of single-GPU /
    /// local-multi-GPU / cluster). See
    /// `EpochAggregated`.
    EpochAggregated(EpochMetrics),
    /// NCCL consensus checkpoint: write the elected rank's CURRENT model
    /// (post-collective consensus) to `<save_path>.fdl`. See
    /// [`crate::distributed::wire::ControlMsgWire::SaveConsensusModel`]. No
    /// `.optim`, no shutdown; the worker no-ops unless `target_rank` matches.
    SaveConsensusModel {
        /// Elected rank that should write the consensus model.
        target_rank: usize,
    },
}

// ---------------------------------------------------------------------------
// Initial setup
// ---------------------------------------------------------------------------

/// Default base seed for deterministic per-epoch shuffling across the DDP
/// paths (single-host fallback, cluster rank entry, threaded coordinator).
///
/// The epoch `e` permutation is `Rng::seed(SHUFFLE_BASE_SEED + e)` (see
/// `make_partition` and [`crate::data::RandomSampler`]). Coverage-granular
/// resume records this value in
/// [`crate::distributed::CoverageBlock::seed`] so a resumed run can verify it
/// re-shuffles over the SAME index space; changing it between save and resume
/// invalidates recorded coverage. Single source of truth — every
/// `WorkerConfig.seed` default and the coverage snapshot read it from here.
pub const SHUFFLE_BASE_SEED: u64 = 42;

/// Resolve the data-shuffle base seed for a run.
///
/// On **resume** the seed is read from the checkpoint meta's
/// [`CoverageBlock::seed`](crate::distributed::CoverageBlock) so the resumed
/// permutation reproduces the saved one by *reading* the recorded value, not
/// by assuming the build's [`SHUFFLE_BASE_SEED`] still matches it. Every role
/// (coordinator + each rank) resolves it from the same meta file, so the value
/// is consistent across the cohort without a broadcast. A fresh run, or a meta
/// with no coverage block (e.g. a clean-boundary save), falls back to
/// [`SHUFFLE_BASE_SEED`]. Errors loudly if `resume_from` is set but the meta
/// can't be read — a silent fallback could desync a worker's permutation from
/// the recorded coverage.
pub(crate) fn resolve_shuffle_seed(resume_from: Option<&str>) -> crate::tensor::Result<u64> {
    let Some(stem) = resume_from else {
        return Ok(SHUFFLE_BASE_SEED);
    };
    let meta = crate::distributed::CheckpointMeta::read_from_file(
        &crate::distributed::CheckpointBundle::meta_path(stem),
    )?;
    Ok(meta
        .coverage
        .as_ref()
        .map(|c| c.seed)
        .unwrap_or(SHUFFLE_BASE_SEED))
}

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
    /// RNG base seed for deterministic shuffling; the epoch `e` permutation is
    /// `Rng::seed(seed + e)`. Defaults to [`SHUFFLE_BASE_SEED`] at every
    /// construction site.
    pub seed: u64,
    /// Maximum gradient norm for clipping (None = no clipping).
    pub max_grad_norm: Option<f64>,
    /// EASGD elastic averaging weight (0, 1]. `None` = full overwrite of
    /// local params with averaged consensus on the cpu-async path (current
    /// behavior). When set, [`crate::distributed::GpuWorker::load_averaged`]
    /// blends `W_local := (1-α)·W_local + α·W_avg` instead. See
    /// [`crate::distributed::ElCheConfig::easgd_alpha`] for details.
    pub easgd_alpha: Option<f64>,
    /// Consensus allocation-weighting exponent `γ`: rank weighted `nₖ^γ` in
    /// the CPU work-weighted average. `1.0` (default) = plain work-weighting,
    /// byte-identical to pre-gamma behavior. See
    /// [`crate::distributed::ElCheConfig::gamma`].
    pub gamma: f64,
    /// Optional system timeline for high-frequency profiling.
    pub timeline: Option<Arc<crate::monitor::Timeline>>,
    /// Training policy (Sync/Cadence/Async). Used to gate divergence measurement:
    /// Sync mode skips weight-space divergence (near-zero by construction).
    pub policy: ApplyPolicy,
    /// Checkpoint bundle stem for unrecoverable-failure persistence.
    ///
    /// When set, workers write a bundle (`.fdl` model, `.optim` optimizer
    /// state, `.meta.json` trajectory) on receipt of
    /// `ShutdownWithSave`; see
    /// [`crate::distributed::CheckpointBundle`] for path derivation.
    /// `None` in standalone single-GPU runs and CPU-only tests that
    /// don't exercise the save path.
    pub save_path: Option<String>,
    /// Coordinator-liveness deadline (seconds) for the cluster worker's
    /// inbound bridge: if no frame (coord heartbeat or real traffic) arrives
    /// within this window, the coord is presumed wedged-open and the rank
    /// bails. Mirrors the coordinator's own `heartbeat_timeout_secs` so both
    /// liveness directions share one timescale. Inert on the thread-based
    /// [`crate::distributed::GpuWorker`] path (single-host / tests) — only
    /// [`crate::distributed::cluster_worker`]'s TCP inbound loop reads it.
    /// Defaults to [`DEFAULT_COORD_LIVENESS_TIMEOUT_SECS`].
    pub coord_liveness_timeout_secs: u64,
}

/// Default coordinator-liveness deadline (seconds), matching the
/// coordinator's default `heartbeat_timeout_secs`. Used when a run does not
/// set `heartbeat_timeout_secs` explicitly.
pub const DEFAULT_COORD_LIVENESS_TIMEOUT_SECS: u64 = 30;

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
mod tests;
