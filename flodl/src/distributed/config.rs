//! User-facing config types for [`Trainer::run`](crate::distributed::Trainer::run).
//!
//! Two structs:
//!
//! - [`ElCheConfig`] — controller-scope knobs: the [`ElCheMode`] (which
//!   collapses the old policy × backend pair into a single mode name),
//!   anchor tuning, heterogeneity (`partition_ratios`), LR-aware
//!   `meta_controller`, and the [`ConvergenceGuard`]. Everything here
//!   reaches the cluster controller; nothing here lives on a single
//!   rank.
//! - [`TrainerConfig`] — the umbrella: dataset / epochs / batch_size,
//!   the nested `elche` config, plus rank-scope knobs (gradient
//!   clipping, checkpointing, callbacks, eval). One object per
//!   training run.
//!
//! Both implement `Default` and chained setters; user code can
//! either populate via setters or via a struct literal with
//! `..Default::default()`.
//!
//! See [`Trainer::run`](crate::distributed::Trainer::run) for the
//! entry-point invariants (chiefly: no CUDA tensor construction before
//! `.run()`; use [`crate::sys::detect_gpus`] for pre-run GPU queries).
//!
//! [`ConvergenceGuard`]: super::ddp_run::ConvergenceGuard

use std::sync::Arc;

use crate::data::BatchDataSet;
use crate::nn::Module;

use super::ddp_run::{
    ApplyPolicy, AverageBackend, CheckpointFn, ConvergenceGuard,
    EpochCallbackPolicy, EpochFn, EvalFn, EvalResultFn,
    MetricsFn, SchedulerFn,
};
use super::launcher::FullCluster;

// ---------------------------------------------------------------------------
// ElCheMode
// ---------------------------------------------------------------------------

/// DDP mode — the (averaging-policy × averaging-backend) pair as a
/// single name. Matches the user-facing naming used in `ddp-bench`,
/// commits, and the design docs (`nccl-sync`, `cpu-async`, etc.).
///
/// All flodl DDP runs go through the ElChe machinery; `Sync` is the
/// degenerate-but-valid case where anchor=1 makes ElChe behave like
/// vanilla synchronous DDP. The other modes engage ElChe's anchor
/// auto-tuning and (in `Async`) overshoot scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElCheMode {
    /// NCCL averaging, all-reduce every batch. Vanilla synchronous DDP.
    /// Fast GPUs wait at each collective barrier — best when GPUs are
    /// homogeneous and inter-rank links are fast.
    NcclSync,
    /// NCCL averaging, anchor-based. ElChe tunes the anchor so the
    /// slow device sets the pace; fast devices process proportionally
    /// more batches per averaging window. Recommended for mixed GPU
    /// setups.
    NcclCadence,
    /// NCCL averaging, anchor + overshoot. Same proportional scheduling
    /// as Cadence plus divergence-driven anchor correction; fast ranks
    /// may stream into the next epoch's data.
    NcclAsync,
    /// CPU-mediated averaging via the coordinator, all-reduce every
    /// batch. Useful when NVLink / PCIe peer access is unavailable,
    /// for heterogeneous-mounted rigs, or for A/B against the NCCL
    /// path.
    CpuSync,
    /// CPU-mediated averaging, anchor-based. CPU's natural decoupling
    /// from the GPU's collective barrier makes this the most
    /// fault-tolerant cadence mode.
    CpuCadence,
    /// CPU-mediated averaging, anchor + overshoot. The CPU averaging
    /// path receives the elastic-blend (EASGD-style) update if
    /// [`ElCheConfig::easgd_alpha`] is set.
    CpuAsync,
}

impl ElCheMode {
    /// Decompose into the legacy `(ApplyPolicy, AverageBackend)` pair
    /// that the orchestrator's internals still speak. Internal — the
    /// public `Trainer::run` path uses [`ElCheMode`] directly.
    pub(crate) fn split(self) -> (ApplyPolicy, AverageBackend) {
        match self {
            Self::NcclSync => (ApplyPolicy::Sync, AverageBackend::Nccl),
            Self::NcclCadence => (ApplyPolicy::Cadence, AverageBackend::Nccl),
            Self::NcclAsync => (ApplyPolicy::Async, AverageBackend::Nccl),
            Self::CpuSync => (ApplyPolicy::Sync, AverageBackend::Cpu),
            Self::CpuCadence => (ApplyPolicy::Cadence, AverageBackend::Cpu),
            Self::CpuAsync => (ApplyPolicy::Async, AverageBackend::Cpu),
        }
    }
}

// ---------------------------------------------------------------------------
// ElCheConfig
// ---------------------------------------------------------------------------

/// Controller-scope DDP / heterogeneity configuration. All knobs here
/// reach the cluster controller's `ClusterCoordinator`; none of them
/// live on a single rank.
///
/// Use a preset constructor to start, then override the knobs you
/// care about. Sensible defaults are filled by `Default`:
///
/// ```ignore
/// // Preset + targeted overrides.
/// let elche = ElCheConfig::nccl_cadence()
///     .max_anchor(20)
///     .overhead_target(0.05)
///     .meta_controller(true);
///
/// // Struct literal + Default.
/// let elche = ElCheConfig {
///     mode: ElCheMode::NcclCadence,
///     max_anchor: Some(20),
///     ..Default::default()
/// };
/// ```
pub struct ElCheConfig {
    /// DDP mode (NcclSync, NcclCadence, ...).
    pub mode: ElCheMode,
    /// Initial anchor (batches before first averaging). Default: 10
    /// for cadence/async modes; presets force 1 for sync modes.
    pub anchor: usize,
    /// Maximum anchor count. `None` = framework default (1000).
    pub max_anchor: Option<usize>,
    /// Minimum anchor count. `None` = equals the initial anchor.
    pub min_anchor: Option<usize>,
    /// ElChe overhead target (fraction of compute time). `None` =
    /// framework default (0.10).
    pub overhead_target: Option<f64>,
    /// Maximum batch lead of fastest over slowest worker.
    /// `Some(0)` = strict lockstep. `None` = unlimited.
    pub max_batch_diff: Option<usize>,
    /// Allow ElChe to relax the anchor upward on stable convergence.
    /// Default: `false`.
    pub relax_up: bool,
    /// Explicit per-rank partition ratios (e.g. `[0.7, 0.3]`). When
    /// set, disables automatic throughput-based rebalancing. Sum must
    /// be ≈ 1.0; length must match world_size.
    pub partition_ratios: Option<Vec<f64>>,
    /// Enable the LR-aware meta-controller above ElChe. When `true`,
    /// the meta layer observes LR trajectory + anchor trend +
    /// convergence-guard verdicts and reactively nudges the anchor
    /// down on sharp LR drops or sustained divergence. Default
    /// `true` — LR drops are always worth catching; opt out with
    /// `.meta_controller(false)` when collecting an unconditioned
    /// trajectory.
    pub meta_controller: bool,
    /// Divergence guardrail. `None` = `TrendGuard::new(0.05)` default;
    /// set to a custom guard to override threshold or replace
    /// behavior. See [`crate::distributed::ddp_run::NoGuard`] /
    /// [`crate::distributed::ddp_run::TrendGuard`].
    pub convergence_guard: Option<Box<dyn ConvergenceGuard>>,
    /// EASGD elastic-averaging weight (0.0 < α ≤ 1.0). Honored on the
    /// CpuAsync path only; ignored elsewhere.
    pub easgd_alpha: Option<f64>,
}

impl ElCheConfig {
    // -- Presets ----------------------------------------------------------

    /// NCCL averaging, all-reduce every batch. anchor=1, no overshoot.
    pub fn nccl_sync() -> Self {
        Self {
            mode: ElCheMode::NcclSync,
            anchor: 1,
            ..Self::default_for(ElCheMode::NcclSync)
        }
    }

    /// NCCL averaging, anchor-based. Default anchor 10.
    pub fn nccl_cadence() -> Self {
        Self::default_for(ElCheMode::NcclCadence)
    }

    /// NCCL averaging, anchor + overshoot. Default anchor 10,
    /// overhead_target 0.10.
    pub fn nccl_async() -> Self {
        Self {
            overhead_target: Some(0.10),
            ..Self::default_for(ElCheMode::NcclAsync)
        }
    }

    /// CPU-mediated averaging, all-reduce every batch.
    pub fn cpu_sync() -> Self {
        Self {
            mode: ElCheMode::CpuSync,
            anchor: 1,
            ..Self::default_for(ElCheMode::CpuSync)
        }
    }

    /// CPU-mediated averaging, anchor-based.
    pub fn cpu_cadence() -> Self {
        Self::default_for(ElCheMode::CpuCadence)
    }

    /// CPU-mediated averaging, anchor + overshoot.
    pub fn cpu_async() -> Self {
        Self {
            overhead_target: Some(0.10),
            ..Self::default_for(ElCheMode::CpuAsync)
        }
    }

    fn default_for(mode: ElCheMode) -> Self {
        Self {
            mode,
            anchor: 10,
            max_anchor: None,
            min_anchor: None,
            overhead_target: None,
            max_batch_diff: None,
            relax_up: false,
            partition_ratios: None,
            meta_controller: true,
            convergence_guard: None,
            easgd_alpha: None,
        }
    }

    // -- Chained setters --------------------------------------------------

    /// Override the mode.
    pub fn mode(mut self, m: ElCheMode) -> Self { self.mode = m; self }
    /// Set the initial anchor.
    pub fn anchor(mut self, n: usize) -> Self { self.anchor = n; self }
    /// Set the maximum anchor (auto-tune ceiling).
    pub fn max_anchor(mut self, n: usize) -> Self { self.max_anchor = Some(n); self }
    /// Set the minimum anchor (auto-tune floor).
    pub fn min_anchor(mut self, n: usize) -> Self { self.min_anchor = Some(n); self }
    /// Set the ElChe overhead target (fraction of compute time).
    pub fn overhead_target(mut self, f: f64) -> Self { self.overhead_target = Some(f); self }
    /// Set the maximum batch lead between fastest and slowest worker.
    pub fn max_batch_diff(mut self, n: usize) -> Self { self.max_batch_diff = Some(n); self }
    /// Allow ElChe to relax the anchor upward on stable convergence.
    pub fn relax_up(mut self, on: bool) -> Self { self.relax_up = on; self }
    /// Set explicit per-rank partition ratios.
    pub fn partition_ratios(mut self, r: Vec<f64>) -> Self { self.partition_ratios = Some(r); self }
    /// Enable the LR-aware meta-controller.
    pub fn meta_controller(mut self, on: bool) -> Self { self.meta_controller = on; self }
    /// Override the convergence guard. Pass a [`Box<dyn ConvergenceGuard>`].
    pub fn convergence_guard<G>(mut self, g: G) -> Self
    where
        G: ConvergenceGuard + 'static,
    {
        self.convergence_guard = Some(Box::new(g));
        self
    }
    /// Set the EASGD elastic-averaging weight (CpuAsync only).
    pub fn easgd_alpha(mut self, a: f64) -> Self { self.easgd_alpha = Some(a); self }
}

impl Default for ElCheConfig {
    /// Default = [`Self::nccl_async`].
    ///
    /// On NCCL, `Async` and `Cadence` share the same in-epoch loop —
    /// the difference is cross-epoch lookahead: `Async` dispatches the
    /// next epoch's plan to a rank as soon as that rank finishes the
    /// current one (per-rank, up to a 1-epoch lookahead bound),
    /// without waiting for full-cluster aggregation. On heterogeneous
    /// rigs that fills the wall-time gap between fast-rank epoch
    /// completion and slow-rank epoch completion. Same numerics, same
    /// rendezvous-at-every-barrier guarantees, strictly better
    /// utilization. `Cadence` remains the right pick when you want
    /// every rank to start each epoch in lockstep.
    fn default() -> Self {
        Self::nccl_async()
    }
}

impl std::fmt::Debug for ElCheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElCheConfig")
            .field("mode", &self.mode)
            .field("anchor", &self.anchor)
            .field("max_anchor", &self.max_anchor)
            .field("min_anchor", &self.min_anchor)
            .field("overhead_target", &self.overhead_target)
            .field("max_batch_diff", &self.max_batch_diff)
            .field("relax_up", &self.relax_up)
            .field("partition_ratios", &self.partition_ratios)
            .field("meta_controller", &self.meta_controller)
            .field("convergence_guard", &self.convergence_guard.as_ref().map(|_| "<dyn ConvergenceGuard>"))
            .field("easgd_alpha", &self.easgd_alpha)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// TrainerConfig
// ---------------------------------------------------------------------------

/// Top-level training configuration for [`Trainer::run`](crate::distributed::Trainer::run).
///
/// Carries:
/// - Universal training-task knobs: `dataset`, `batch_size`, `num_epochs`.
/// - Nested ElChe / DDP config: `elche` (see [`ElCheConfig`]).
/// - Rank-scope knobs: `max_grad_norm`, checkpointing,
///   user callbacks, eval setup.
///
/// Use [`TrainerConfig::new`] to start with sensible defaults, or
/// build via struct literal with `..Default::default()`.
///
/// Generic over the model type `M` to keep user-supplied callbacks
/// (`checkpoint_fn`, `epoch_fn`, `eval_fn`) statically typed.
pub struct TrainerConfig<M: Module> {
    /// Training dataset.
    pub dataset: Arc<dyn BatchDataSet>,
    /// Batch size (per replica).
    pub batch_size: usize,
    /// Number of epochs to train.
    pub num_epochs: usize,

    /// ElChe / DDP / heterogeneity sub-config. Defaults to
    /// `ElCheConfig::nccl_cadence()` — recommended for mixed-GPU rigs;
    /// override with a preset (e.g. `ElCheConfig::nccl_sync()` for
    /// vanilla DDP) when appropriate.
    pub elche: ElCheConfig,

    /// Maximum gradient norm for per-worker clipping. `None` = no clipping.
    pub max_grad_norm: Option<f64>,
    /// Save a checkpoint every N global epochs. `None` = no periodic save.
    pub checkpoint_every: Option<usize>,
    /// Checkpoint bundle stem for cluster-mode unrecoverable-failure
    /// persistence. See [`crate::distributed::CheckpointBundle`].
    pub save_path: Option<String>,
    /// Resume from a previously-saved checkpoint bundle stem.
    pub resume_from: Option<String>,

    /// Per-checkpoint callback (`version, &model`). Fires on the rank
    /// chosen by [`crate::distributed::ddp_run::EpochCallbackPolicy`].
    pub checkpoint_fn: Option<CheckpointFn<M>>,
    /// Per-epoch callback (`epoch, &mut worker`). Fires inside each
    /// worker thread before the epoch plan runs.
    pub epoch_fn: Option<EpochFn<M>>,
    /// Per-epoch host-side metrics callback.
    pub metrics_fn: Option<MetricsFn>,
    /// LR-scheduler factory.
    pub scheduler_fn: Option<SchedulerFn>,
    /// Periodic eval callback.
    pub eval_fn: Option<EvalFn<M>>,
    /// Eval dataset (held-out).
    pub eval_dataset: Option<Arc<dyn BatchDataSet>>,
    /// Eval result handler (controller-side).
    pub eval_result_fn: Option<EvalResultFn>,

    /// Which rank fires user-supplied per-epoch callbacks. Default
    /// `Rank(0)`.
    pub epoch_callback_policy: EpochCallbackPolicy,

    /// Optional high-frequency system timeline for profiling.
    pub timeline: Option<Arc<crate::monitor::Timeline>>,

    /// Programmatic cluster topology. When `Some`, [`Trainer::run`]
    /// promotes this process to launcher role with the given cluster
    /// (same effect as setting `FLODL_FULL_CLUSTER_JSON` via fdl-cli's
    /// overlay parsing — the launcher contract is single-shape).
    ///
    /// Three precedence cases at `Trainer::run` time:
    /// - `FLODL_FULL_CLUSTER_JSON` already set in env (fdl-cli set
    ///   it before invoking the user binary) → that wins; this field
    ///   is ignored.
    /// - `cfg.cluster = Some(...)` and env unset → field wins; serializes
    ///   into `FLODL_FULL_CLUSTER_JSON` and dispatch fires Launcher.
    /// - Neither set + 2+ visible GPUs → auto-promote synthesizes a
    ///   localhost cluster via [`super::ClusterBuilder::all_local_gpus`].
    /// - Neither set + ≤1 GPU → single-device path.
    ///
    /// Construct via [`super::ClusterBuilder`].
    ///
    /// [`Trainer::run`]: crate::distributed::Trainer::run
    pub cluster: Option<FullCluster>,
}

impl<M: Module> TrainerConfig<M> {
    /// New config with sensible defaults: `batch_size=32`, `num_epochs=1`,
    /// `elche = ElCheConfig::default()` (NcclCadence), no callbacks.
    pub fn new(dataset: Arc<dyn BatchDataSet>) -> Self {
        Self {
            dataset,
            batch_size: 32,
            num_epochs: 1,
            elche: ElCheConfig::default(),
            max_grad_norm: None,
            checkpoint_every: None,
            save_path: None,
            resume_from: None,
            checkpoint_fn: None,
            epoch_fn: None,
            metrics_fn: None,
            scheduler_fn: None,
            eval_fn: None,
            eval_dataset: None,
            eval_result_fn: None,
            epoch_callback_policy: EpochCallbackPolicy::default(),
            timeline: None,
            cluster: None,
        }
    }

    // -- Chained setters --------------------------------------------------

    /// Set the batch size.
    pub fn batch_size(mut self, n: usize) -> Self { self.batch_size = n; self }
    /// Set the number of epochs.
    pub fn num_epochs(mut self, n: usize) -> Self { self.num_epochs = n; self }
    /// Replace the nested ElChe config.
    pub fn elche(mut self, cfg: ElCheConfig) -> Self { self.elche = cfg; self }
    /// Set the gradient-clipping max norm.
    pub fn max_grad_norm(mut self, n: f64) -> Self { self.max_grad_norm = Some(n); self }
    /// Set the periodic checkpoint cadence (in epochs).
    pub fn checkpoint_every(mut self, n: usize) -> Self { self.checkpoint_every = Some(n); self }
    /// Set the cluster-mode checkpoint bundle stem (save side).
    pub fn save_path(mut self, p: impl Into<String>) -> Self { self.save_path = Some(p.into()); self }
    /// Resume training from a previously-saved bundle stem.
    pub fn resume_from(mut self, p: impl Into<String>) -> Self { self.resume_from = Some(p.into()); self }

    /// Register a checkpoint callback. Fires on the selected rank.
    pub fn checkpoint_fn(mut self, f: CheckpointFn<M>) -> Self { self.checkpoint_fn = Some(f); self }
    /// Register a per-epoch callback (inside each worker thread).
    pub fn epoch_fn(mut self, f: EpochFn<M>) -> Self { self.epoch_fn = Some(f); self }
    /// Register a host-side metrics callback.
    pub fn metrics_fn(mut self, f: MetricsFn) -> Self { self.metrics_fn = Some(f); self }
    /// Register the LR-scheduler factory.
    pub fn scheduler_fn(mut self, f: SchedulerFn) -> Self { self.scheduler_fn = Some(f); self }
    /// Register the eval callback (paired with `eval_dataset`).
    pub fn eval_fn(mut self, f: EvalFn<M>) -> Self { self.eval_fn = Some(f); self }
    /// Set the held-out eval dataset.
    pub fn eval_dataset(mut self, ds: Arc<dyn BatchDataSet>) -> Self { self.eval_dataset = Some(ds); self }
    /// Register the controller-side eval-result handler.
    pub fn eval_result_fn(mut self, f: EvalResultFn) -> Self { self.eval_result_fn = Some(f); self }
    /// Override which rank fires per-epoch callbacks.
    pub fn epoch_callback_policy(mut self, p: EpochCallbackPolicy) -> Self {
        self.epoch_callback_policy = p;
        self
    }
    /// Attach a profiling timeline.
    pub fn timeline(mut self, t: Arc<crate::monitor::Timeline>) -> Self {
        self.timeline = Some(t);
        self
    }

    /// Attach a programmatic cluster topology. See the field docs for
    /// precedence with `FLODL_FULL_CLUSTER_JSON` env-var and the
    /// auto-promote path.
    pub fn cluster(mut self, c: FullCluster) -> Self {
        self.cluster = Some(c);
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elche_mode_split_round_trip() {
        for m in [
            ElCheMode::NcclSync, ElCheMode::NcclCadence, ElCheMode::NcclAsync,
            ElCheMode::CpuSync, ElCheMode::CpuCadence, ElCheMode::CpuAsync,
        ] {
            let (p, b) = m.split();
            // Sync → Sync, Cadence → Cadence, Async → Async + correct backend.
            let expected_backend = match m {
                ElCheMode::NcclSync | ElCheMode::NcclCadence | ElCheMode::NcclAsync => AverageBackend::Nccl,
                ElCheMode::CpuSync | ElCheMode::CpuCadence | ElCheMode::CpuAsync => AverageBackend::Cpu,
            };
            assert_eq!(b, expected_backend, "{:?} backend split", m);
            let expected_policy = match m {
                ElCheMode::NcclSync | ElCheMode::CpuSync => ApplyPolicy::Sync,
                ElCheMode::NcclCadence | ElCheMode::CpuCadence => ApplyPolicy::Cadence,
                ElCheMode::NcclAsync | ElCheMode::CpuAsync => ApplyPolicy::Async,
            };
            assert_eq!(p, expected_policy, "{:?} policy split", m);
        }
    }

    #[test]
    fn elche_presets_carry_sane_defaults() {
        assert_eq!(ElCheConfig::nccl_sync().anchor, 1);
        assert_eq!(ElCheConfig::nccl_sync().mode, ElCheMode::NcclSync);
        assert_eq!(ElCheConfig::cpu_sync().anchor, 1);
        assert_eq!(ElCheConfig::nccl_cadence().anchor, 10);
        assert_eq!(ElCheConfig::nccl_cadence().mode, ElCheMode::NcclCadence);
        assert_eq!(ElCheConfig::nccl_async().overhead_target, Some(0.10));
        assert_eq!(ElCheConfig::cpu_async().mode, ElCheMode::CpuAsync);
    }

    #[test]
    fn elche_chained_setters() {
        let cfg = ElCheConfig::nccl_cadence()
            .max_anchor(20)
            .overhead_target(0.05)
            .meta_controller(true)
            .partition_ratios(vec![0.7, 0.3])
            .relax_up(true);
        assert_eq!(cfg.max_anchor, Some(20));
        assert_eq!(cfg.overhead_target, Some(0.05));
        assert!(cfg.meta_controller);
        assert_eq!(cfg.partition_ratios.as_deref(), Some(&[0.7, 0.3][..]));
        assert!(cfg.relax_up);
    }

    #[test]
    fn elche_default_is_nccl_async() {
        let cfg = ElCheConfig::default();
        assert_eq!(cfg.mode, ElCheMode::NcclAsync);
        assert_eq!(cfg.anchor, 10);
    }

    #[test]
    fn meta_controller_default_is_on() {
        let cfg = ElCheConfig::default();
        assert!(cfg.meta_controller, "meta_controller defaults to true");
        // Spot-check all six presets agree on the default.
        for preset in [
            ElCheConfig::nccl_sync(),
            ElCheConfig::nccl_cadence(),
            ElCheConfig::nccl_async(),
            ElCheConfig::cpu_sync(),
            ElCheConfig::cpu_cadence(),
            ElCheConfig::cpu_async(),
        ] {
            assert!(preset.meta_controller, "preset for {:?}", preset.mode);
        }
    }
}
