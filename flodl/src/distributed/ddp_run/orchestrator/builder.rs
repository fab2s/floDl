//! `DdpBuilder`: fluent builder returned by `Trainer::builder()`.
//!
//! Configures dataset, batch size, epochs, policy/backend, callbacks, eval
//! pairing, and convergence guard, then dispatches into `DdpHandle::launch`
//! on `.run()`.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::autograd::Variable;
use crate::data::BatchDataSet;
use crate::nn::{Module, Optimizer, Parameter};
use crate::tensor::{Device, Result, Tensor};

use crate::distributed::ddp_run::{
    ApplyPolicy, AverageBackend, CheckpointFn, ConvergenceGuard, DdpRunConfig,
    EpochCallbackPolicy, EpochFn, EpochMetrics, EvalCadence, EvalFn, EvalResultFn,
    MetricsFn, SchedulerFn,
};
use crate::distributed::ddp_run::worker::GpuWorker;
use super::DdpHandle;


// ---------------------------------------------------------------------------
// DdpBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring and launching framework-managed DDP training.
///
/// Created via [`Trainer::builder()`](crate::distributed::Trainer::builder). Required fields must be set before
/// calling [`run`](Self::run); missing fields produce a clear panic message.
///
/// # Example
///
/// ```ignore
/// use flodl::*;
///
/// let handle = Trainer::builder(
///     |dev| model_factory(dev),
///     |params| Adam::new(params, 0.001),
///     |model, batch| { /* return loss Variable */ },
/// )
/// .dataset(dataset)
/// .batch_size(32)
/// .num_epochs(10)
/// .policy(ApplyPolicy::Cadence)
/// .backend(AverageBackend::Nccl)
/// .run()?;
///
/// let state = handle.join()?; // blocks until training completes
/// ```
#[allow(clippy::type_complexity)]
pub struct DdpBuilder<F, M, G, O, T>
where
    F: Fn(Device) -> Result<M> + Send + Sync + 'static,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
    O: Optimizer + 'static,
    T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
{
    model_factory: F,
    optim_factory: G,
    train_fn: T,
    dataset: Option<Arc<dyn BatchDataSet>>,
    batch_size: Option<usize>,
    num_epochs: Option<usize>,
    policy: ApplyPolicy,
    backend: AverageBackend,
    config: DdpRunConfig,
    checkpoint_fn: Option<CheckpointFn<M>>,
    epoch_fn: Option<EpochFn<M>>,
    metrics_fn: Option<MetricsFn>,
    /// Factory receives `world_size`, returns the scheduler.
    scheduler_fn: Option<SchedulerFn>,
    /// Pluggable convergence guard. When set, takes precedence over the
    /// legacy `divergence_threshold` / `no_divergence_guard` fields on
    /// [`DdpRunConfig`]. Boxed because trait-object guards aren't `Clone`.
    convergence_guard: Option<Box<dyn ConvergenceGuard>>,
    /// Rank-side eval callback. Fires on the rank chosen by
    /// `EpochCallbackPolicy` against `eval_dataset`; the framework flips
    /// `Module::eval`/`train` around the closure and ships the scalar
    /// result back to the controller.
    eval_fn: Option<EvalFn<M>>,
    /// Held-out dataset paired with `eval_fn`.
    eval_dataset: Option<Arc<dyn BatchDataSet>>,
    /// Controller-side callback receiving `(epoch, metric)` once the
    /// chosen rank's `eval_fn` result arrives over the wire.
    eval_result_fn: Option<EvalResultFn>,
    _phantom: PhantomData<(M, O)>,
}

impl<F, M, G, O, T> DdpBuilder<F, M, G, O, T>
where
    F: Fn(Device) -> Result<M> + Send + Sync + 'static,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
    O: Optimizer + 'static,
    T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
{
    /// Set the training dataset (required).
    pub fn dataset(mut self, dataset: Arc<dyn BatchDataSet>) -> Self {
        self.dataset = Some(dataset);
        self
    }

    /// Set the batch size (required).
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Set the number of epochs (required).
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.num_epochs = Some(n);
        self
    }

    /// Set the averaging policy. Default: [`ApplyPolicy::Cadence`].
    pub fn policy(mut self, policy: ApplyPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set the averaging backend. Default: [`AverageBackend::Nccl`].
    pub fn backend(mut self, backend: AverageBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the AllReduce overhead target (fraction of compute time).
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.config = self.config.with_overhead_target(target);
        self
    }

    /// Set the maximum anchor count.
    pub fn max_anchor(mut self, max: usize) -> Self {
        self.config = self.config.with_max_anchor(max);
        self
    }

    /// Set the minimum anchor count (auto-tune floor).
    ///
    /// Combined with `max_anchor(min)` (same value) plus
    /// [`Self::convergence_guard`] = `NoGuard` and
    /// [`Self::no_divergence_guard`], pins the anchor at a fixed cadence.
    pub fn min_anchor(mut self, min: usize) -> Self {
        self.config = self.config.with_min_anchor(min);
        self
    }

    /// Set the initial anchor count.
    pub fn anchor(mut self, anchor: usize) -> Self {
        self.config = self.config.with_anchor(anchor);
        self
    }

    /// Set the divergence threshold for the trend guardrail.
    pub fn divergence_threshold(mut self, threshold: f64) -> Self {
        self.config = self.config.with_divergence_threshold(threshold);
        self
    }

    /// Disable the divergence guardrail. ElChe's overhead auto-tune
    /// handles cadence alone. Use when you know your workload is stable.
    ///
    /// Equivalent to `.convergence_guard(NoGuard)` but kept for backward
    /// compatibility with the older boolean-flag API.
    pub fn no_divergence_guard(mut self) -> Self {
        self.config = self.config.with_no_divergence_guard();
        self
    }

    /// Install a custom convergence guard.
    ///
    /// When set, takes precedence over the legacy `divergence_threshold` and
    /// `no_divergence_guard` settings. Three concrete impls ship in flodl:
    /// [`crate::distributed::ddp_run::NoGuard`],
    /// [`crate::distributed::ddp_run::TrendGuard`] (production default), and
    /// [`crate::distributed::ddp_run::MsfGuard`] (rate-based detector with
    /// soft+hard thresholds).
    ///
    /// ```text
    /// .convergence_guard(
    ///     MsfGuard::default()
    ///         .with_suppress(1e-3, 3)
    ///         .with_nudge(1e-2, 3, 0.5),
    /// )
    /// ```
    pub fn convergence_guard<C>(mut self, guard: C) -> Self
    where
        C: ConvergenceGuard + 'static,
    {
        self.convergence_guard = Some(Box::new(guard));
        self
    }

    /// Set the maximum batch lead of fastest over slowest worker.
    /// `0` = strict lockstep.
    pub fn max_batch_diff(mut self, max: usize) -> Self {
        self.config = self.config.with_max_batch_diff(max);
        self
    }

    /// Set explicit per-rank partition ratios, e.g. `[0.55, 0.225, 0.225]`
    /// for a fast rank plus two slower ranks.
    ///
    /// **Static fixed splits — does not auto-rebalance.** When set, the
    /// coordinator dispatches each epoch's batches in proportion to these
    /// ratios and skips ElChe's throughput-based rebalancer. Length must
    /// match the auto-detected `world_size` and values must sum to ~1.0.
    ///
    /// **Currently honored in `Sync` policy only.** The `Cadence` and
    /// `Async` policies use progressive dispatch driven by ElChe; they
    /// do not consult `partition_ratios`. For dynamic heterogeneous
    /// scheduling under those policies, ElChe's auto-calibration is
    /// the intended path (see `speed_hint` for an initial seed).
    pub fn partition_ratios(mut self, ratios: &[f64]) -> Self {
        self.config = self.config.with_partition_ratios(ratios);
        self
    }

    /// Set the maximum overshoot past the planned sync point.
    ///
    /// Controls how far a fast GPU can stream past its planned batch count
    /// into the next epoch's data. Default: auto-tuned from convergence.
    pub fn max_overshoot(mut self, max: usize) -> Self {
        self.config = self.config.with_max_overshoot(max);
        self
    }

    /// Allow or suppress ElChe's anchor relax-up on stable convergence.
    ///
    /// Default: `false` (opt-in). When `true`, each `Stable` convergence
    /// verdict triggers `el_che.relax_anchor_up()` to grow the anchor toward
    /// `max_anchor`. Opt in when measuring the relax-up regime; the default
    /// keeps the anchor under overhead-based auto-tune alone.
    pub fn elche_relax_up(mut self, enabled: bool) -> Self {
        self.config = self.config.with_elche_relax_up(enabled);
        self
    }

    /// Enable EASGD elastic averaging on the cpu-async path with weight α.
    /// `α` must be in `(0, 1]`. See [`DdpRunConfig::with_easgd_alpha`].
    pub fn easgd_alpha(mut self, alpha: f64) -> Self {
        self.config = self.config.with_easgd_alpha(alpha);
        self
    }

    /// Enable the LR-aware meta-controller above ElChe. Default: `false`.
    ///
    /// When enabled, the coordinator constructs a
    /// [`crate::distributed::lr_event_meta::LrEventMeta`] that observes the
    /// LR trajectory, anchor trend, and convergence guard verdicts each
    /// averaging cycle. Sharp LR drops or sustained divergence patterns
    /// trigger reactive `nudge_anchor_down` calls; ElChe's overhead
    /// auto-tune handles recovery. Off by default until validation sweep.
    pub fn meta_controller(mut self, enabled: bool) -> Self {
        self.config = self.config.with_meta_controller(enabled);
        self
    }

    /// Save a checkpoint every N global epochs.
    pub fn checkpoint_every(mut self, n: usize) -> Self {
        self.config = self.config.with_checkpoint_every(n);
        self
    }

    /// Set the cluster-mode checkpoint bundle stem. Setting this also
    /// flips NCCL routing to the controller-driven via_coord path
    /// (elastic membership + persistence on unrecoverable failure).
    ///
    /// On unrecoverable failure, the controller writes
    /// `<stem>.meta.json` (trajectory + ElChe state) and each rank
    /// writes `<stem>.fdl` (model, rank 0) + `<stem>.optim`
    /// (per-rank optimizer state) before exiting. See
    /// [`crate::distributed::CheckpointBundle`].
    pub fn save_path(mut self, path: impl Into<String>) -> Self {
        self.config = self.config.with_save_path(path);
        self
    }

    /// Resume a prior cluster run from a checkpoint bundle.
    ///
    /// `stem` is the bundle stem used at the original save (the value
    /// passed to [`Self::save_path`] then). At `.run()`, the orchestrator
    /// reads `<stem>.meta.json` and seeds the controller with the saved
    /// trajectory:
    /// - starting epoch (the launcher kicks off `dispatch_epoch(meta.epoch)`
    ///   instead of `0`),
    /// - `global_step` and `sync_round` so the LR scheduler picks up where
    ///   it left off,
    /// - the [`crate::distributed::ElCheState`] snapshot including
    ///   [`crate::distributed::ddp_run::convergence::TrendGuard`] history,
    ///   so cadence + divergence trajectories don't re-warm from scratch.
    ///
    /// Model parameters and optimizer state are NOT auto-loaded here —
    /// load them inside `model_factory` / `optim_factory` so each rank's
    /// freshly-built model/optimizer reflects the saved weights:
    ///
    /// ```ignore
    /// Trainer::builder(
    ///     |dev| {
    ///         let model = build_model(dev)?;
    ///         flodl::nn::load_checkpoint_file("ckpt.fdl", &model, dev)?;
    ///         Ok(model)
    ///     },
    ///     |params| {
    ///         let mut opt = Adam::new(params, lr);
    ///         opt.load_state_file("ckpt.optim").ok();
    ///         opt
    ///     },
    ///     train_fn,
    /// )
    ///     .save_path("ckpt")
    ///     .resume_from("ckpt")
    ///     .dataset(dataset).batch_size(32).num_epochs(N)
    ///     .run()?
    ///     .join()?;
    /// ```
    pub fn resume_from(mut self, stem: impl Into<String>) -> Self {
        self.config = self.config.with_resume_from(stem);
        self
    }

    /// Cluster-mode threshold for declaring a run unrecoverable. When
    /// the dead-rank count reaches this limit, the controller
    /// broadcasts `ShutdownWithSave` to survivors and writes the
    /// `.meta.json` sidecar. Backend hard limits still apply (NCCL
    /// needs 2+ survivors; CPU needs at least 1).
    pub fn max_failure(
        mut self,
        threshold: crate::distributed::max_failure::MaxFailureThreshold,
    ) -> Self {
        self.config = self.config.with_max_failure(threshold);
        self
    }

    /// Cluster-mode heartbeat staleness threshold (seconds). If a
    /// rank's last `TimingMsg` frame is older than this, the
    /// controller declares the rank dead. Default: controller's
    /// built-in (currently 30s).
    pub fn heartbeat_timeout_secs(mut self, secs: u64) -> Self {
        self.config = self.config.with_heartbeat_timeout_secs(secs);
        self
    }

    /// Enable or disable progressive chunk dispatch.
    ///
    /// When enabled, the coordinator streams work in small chunks instead of
    /// sending full epoch partitions, adapting to throughput continuously.
    /// Default: auto (true for Cadence/Async, false for Sync).
    pub fn progressive_dispatch(mut self, enabled: bool) -> Self {
        self.config = self.config.with_progressive_dispatch(enabled);
        self
    }

    /// Set maximum gradient norm for per-worker clipping.
    ///
    /// Each worker clips accumulated gradients (L2 norm) after backward
    /// and before the optimizer step. Same knob as `DdpConfig::max_grad_norm`
    /// for the setup/El Che path.
    pub fn max_grad_norm(mut self, max_norm: f64) -> Self {
        self.config = self.config.with_max_grad_norm(max_norm);
        self
    }

    /// Attach a high-frequency system timeline for profiling DDP behavior.
    ///
    /// The coordinator and workers inject training events (sync, epoch,
    /// anchor changes, throttle) into the timeline for post-run analysis.
    pub fn timeline(mut self, tl: std::sync::Arc<crate::monitor::Timeline>) -> Self {
        self.config = self.config.with_timeline(tl);
        self
    }

    /// Set the LR scaling ratio for multi-GPU training.
    ///
    /// Formula: `lr_factor = 1.0 + (world_size - 1) * ratio`.
    ///
    /// - `1.0` (default): full linear scaling (Goyal et al., 2017).
    /// - `0.0`: no scaling.
    /// - `0.5`: half linear scaling.
    pub fn lr_scale_ratio(mut self, ratio: f64) -> Self {
        self.config = self.config.with_lr_scale_ratio(ratio);
        self
    }

    /// Set the checkpoint function called on rank 0 after averaging.
    ///
    /// Receives `(version, &model)`. Errors are logged but do not stop training.
    ///
    /// For Graph models: `.checkpoint_fn(|ver, g| g.save_checkpoint(&format!("ckpt_v{ver}.fdl")))`
    pub fn checkpoint_fn<C>(mut self, f: C) -> Self
    where
        C: Fn(u64, &M) -> Result<()> + Send + Sync + 'static,
    {
        self.checkpoint_fn = Some(Arc::new(f));
        self
    }

    /// Attach a per-batch LR scheduler factory.
    ///
    /// The factory receives `world_size` so user-defined schedulers can
    /// account for multi-GPU training (e.g. scale warmup duration).
    ///
    /// Each worker adjusts its optimizer's LR before every `optimizer.step()`:
    ///
    /// ```text
    /// lr = scheduler.lr(global_step + steps_since_last_sync)
    /// ```
    ///
    /// At each sync point, the coordinator broadcasts the updated global step
    /// so all workers track the same schedule.
    pub fn scheduler<S>(mut self, factory: S) -> Self
    where
        S: Fn(usize) -> Arc<dyn crate::nn::Scheduler> + Send + Sync + 'static,
    {
        self.scheduler_fn = Some(Box::new(factory));
        self
    }

    /// Set an epoch callback called at the start of each epoch inside each worker thread.
    ///
    /// Receives `(epoch, &mut GpuWorker<M>)`. Runs before [`run_epoch_plan`](GpuWorker::run_epoch_plan),
    /// so [`current_epoch()`](GpuWorker::current_epoch) is already correct.
    ///
    /// Typical uses: noise curricula, dynamic loss weights.
    /// For LR scheduling, prefer [`.scheduler()`](Self::scheduler) which
    /// provides per-batch granularity with global step tracking.
    ///
    /// ```text
    /// .epoch_fn(move |epoch, worker| {
    ///     // custom per-epoch logic
    /// })
    /// ```
    pub fn epoch_fn<E>(mut self, f: E) -> Self
    where
        E: Fn(usize, &mut GpuWorker<M>) + Send + Sync + 'static,
    {
        self.epoch_fn = Some(Arc::new(f));
        self
    }

    /// Set a host-side per-epoch metrics callback.
    ///
    /// Called once per epoch with the aggregated [`EpochMetrics`]:
    /// on the coordinator thread for multi-GPU, on the main thread for the
    /// single-GPU fallback. Errors are logged to stderr; training continues.
    /// The same metric is also pushed to the [`DdpHandle::next_metrics`]
    /// queue, so this composes with explicit polling rather than replacing it.
    ///
    /// Use this to keep the chained `Trainer::builder(...).run()?.join()?`
    /// shape observable without a manual polling loop:
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
    ///
    /// Works identically on 1-or-N GPUs: the single-GPU fallback path is
    /// synchronous, so `next_metrics()` returns all queued metrics
    /// back-to-back after `run()` returns; the callback itself fires
    /// per-epoch as training progresses, the same as multi-GPU.
    pub fn metrics_fn<E>(mut self, f: E) -> Self
    where
        E: Fn(&EpochMetrics) -> Result<()> + Send + Sync + 'static,
    {
        self.metrics_fn = Some(Arc::new(f));
        self
    }

    /// Attach a held-out eval dataset paired with [`Self::eval_fn`].
    ///
    /// The framework hands `&dyn BatchDataSet` to the eval closure each
    /// time it fires; the user iterates batches inside. Shared-storage
    /// baseline (NAS/SMB/NFS) means the same Arc can be cloned across
    /// ranks and still resolve to the same on-disk data.
    pub fn eval_dataset(mut self, dataset: Arc<dyn BatchDataSet>) -> Self {
        self.eval_dataset = Some(dataset);
        self
    }

    /// Eval cadence. Currently only [`EvalCadence::Epochs`] is
    /// honored; per-batch cadences may land in a follow-up.
    pub fn eval_every(mut self, cadence: EvalCadence) -> Self {
        let EvalCadence::Epochs(n) = cadence;
        self.config = self.config.with_eval_every_epochs(n);
        self
    }

    /// Rank-side eval callback. Fires on the rank chosen by
    /// [`EpochCallbackPolicy`] at the cadence set by
    /// [`Self::eval_every`].
    ///
    /// Receives `(&model, &dyn BatchDataSet)` and returns
    /// `Result<f64>` — the aggregated scalar metric. The framework
    /// flips `Module::eval` / `Module::train` around the closure and
    /// ships the result back to the controller, where
    /// [`Self::eval_result_fn`] fires.
    ///
    /// Errors propagate to the controller as a string and surface in
    /// the per-epoch log (no early stop today; configurable failure
    /// policy lands in a follow-up).
    pub fn eval_fn<E>(mut self, f: E) -> Self
    where
        E: Fn(&M, &dyn BatchDataSet) -> Result<f64> + Send + Sync + 'static,
    {
        self.eval_fn = Some(Arc::new(f));
        self
    }

    /// Controller-side callback receiving `(epoch, metric)` once the
    /// chosen rank's eval result arrives. Mirrors [`Self::metrics_fn`]
    /// in placement — controller-side, post-aggregation. Use it to
    /// record eval curves, gate early stopping, etc.
    pub fn eval_result_fn<E>(mut self, f: E) -> Self
    where
        E: Fn(usize, f64) -> Result<()> + Send + Sync + 'static,
    {
        self.eval_result_fn = Some(Arc::new(f));
        self
    }

    /// Override the epoch-callback policy (which rank fires user
    /// callbacks). Default: `Rank(0)`. Mirrors
    /// [`DdpRunConfig::with_epoch_callback_policy`].
    pub fn epoch_callback_policy(
        mut self,
        policy: EpochCallbackPolicy,
    ) -> Self {
        self.config = self.config.with_epoch_callback_policy(policy);
        self
    }

    // -- Internal pre-wrapped setters used by `Trainer::run` --------------
    //
    // The public closure-form setters wrap callbacks in `Arc`/`Box` for
    // storage. When the source is `TrainerConfig` (which already stores
    // them in their final wrapper), these pass them through unchanged
    // so we don't have to round-trip through a fresh closure.

    /// Pass-through for a pre-wrapped checkpoint callback. Internal —
    /// public users call [`Self::checkpoint_fn`].
    pub(crate) fn checkpoint_fn_arc(mut self, f: CheckpointFn<M>) -> Self {
        self.checkpoint_fn = Some(f);
        self
    }
    /// Pass-through for a pre-wrapped epoch callback.
    pub(crate) fn epoch_fn_arc(mut self, f: EpochFn<M>) -> Self {
        self.epoch_fn = Some(f);
        self
    }
    /// Pass-through for a pre-wrapped metrics callback.
    pub(crate) fn metrics_fn_arc(mut self, f: MetricsFn) -> Self {
        self.metrics_fn = Some(f);
        self
    }
    /// Pass-through for a pre-wrapped scheduler factory.
    pub(crate) fn scheduler_fn_boxed(mut self, f: SchedulerFn) -> Self {
        self.scheduler_fn = Some(f);
        self
    }
    /// Pass-through for a pre-wrapped eval callback.
    pub(crate) fn eval_fn_arc(mut self, f: EvalFn<M>) -> Self {
        self.eval_fn = Some(f);
        self
    }
    /// Pass-through for a pre-wrapped eval-result callback.
    pub(crate) fn eval_result_fn_arc(mut self, f: EvalResultFn) -> Self {
        self.eval_result_fn = Some(f);
        self
    }
    /// Pass-through for a pre-boxed convergence guard.
    pub(crate) fn convergence_guard_boxed(
        mut self,
        guard: Box<dyn ConvergenceGuard>,
    ) -> Self {
        self.convergence_guard = Some(guard);
        self
    }

    /// Launch training. Non-blocking: spawns threads and returns immediately.
    ///
    /// Call [`DdpHandle::join`] to block until training completes and retrieve
    /// the trained parameters and buffers.
    ///
    /// # Panics
    ///
    /// Panics if `dataset`, `batch_size`, or `num_epochs` were not set.
    pub fn run(self) -> Result<DdpHandle> {
        let dataset = self.dataset.expect("DdpBuilder: dataset is required");
        let batch_size = self.batch_size.expect("DdpBuilder: batch_size is required");
        let num_epochs = self.num_epochs.expect("DdpBuilder: num_epochs is required");

        DdpHandle::launch(
            self.model_factory,
            self.optim_factory,
            self.train_fn,
            dataset,
            batch_size,
            num_epochs,
            self.policy,
            self.backend,
            self.config,
            self.checkpoint_fn,
            self.epoch_fn,
            self.metrics_fn,
            self.scheduler_fn,
            self.convergence_guard,
            self.eval_fn,
            self.eval_dataset,
            self.eval_result_fn,
        )
    }
}

impl DdpHandle {
    /// Create a builder for configuring framework-managed DDP training.
    ///
    /// Prefer [`Trainer::builder()`](crate::distributed::Trainer::builder) as the primary entry point.
    /// This method exists for backward compatibility.
    #[deprecated(since = "0.3.0", note = "Use Trainer::builder() instead")]
    pub fn builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        Self::new_builder(model_factory, optim_factory, train_fn)
    }

    /// Internal builder constructor, called by [`Trainer::builder()`](crate::distributed::Trainer::builder).
    pub(crate) fn new_builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        DdpBuilder {
            model_factory,
            optim_factory,
            train_fn,
            dataset: None,
            batch_size: None,
            num_epochs: None,
            policy: ApplyPolicy::Cadence,
            backend: AverageBackend::Nccl,
            config: DdpRunConfig::new(),
            checkpoint_fn: None,
            epoch_fn: None,
            metrics_fn: None,
            scheduler_fn: None,
            convergence_guard: None,
            eval_fn: None,
            eval_dataset: None,
            eval_result_fn: None,
            _phantom: PhantomData,
        }
    }
}
