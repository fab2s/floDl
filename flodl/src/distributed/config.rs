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
/// All flodl DDP runs go through the ElChe machinery; the `*Sync`
/// modes are its tightest cadence, NOT per-batch lockstep DDP: data
/// is dispatched as an equal split, and the reduce fires as soon as
/// every alive rank has made at least one step since the last reduce,
/// with contributions work-weighted (sum-and-count). On a homogeneous
/// rig that degenerates to vanilla synchronous DDP (one step per rank
/// per reduce); on a heterogeneous rig the fast GPU runs several
/// steps per reduce within its equal share, then idles once it is
/// exhausted. The other modes engage ElChe's anchor auto-tuning —
/// proportional dispatch, reduce once every rank completes its
/// planned window — and (in `Async`) overshoot scheduling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ElCheMode {
    /// NCCL averaging at the tightest cadence (reduce per slow-rank
    /// step over an equal data split — vanilla DDP only on homogeneous
    /// rigs; see the type-level note). Best when GPUs are homogeneous
    /// and inter-rank links are fast.
    NcclSync,
    /// NCCL averaging, anchor-based. ElChe tunes the anchor so the
    /// slow device sets the pace; fast devices process proportionally
    /// more batches per averaging window. Recommended NCCL default for
    /// mixed GPU setups.
    NcclCadence,
    /// CPU-mediated averaging via the coordinator at the tightest
    /// cadence (same gate as [`Self::NcclSync`]). Useful when NVLink /
    /// PCIe peer access is unavailable, for heterogeneous-mounted
    /// rigs, or for A/B against the NCCL path.
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
            Self::CpuSync => (ApplyPolicy::Sync, AverageBackend::Cpu),
            Self::CpuCadence => (ApplyPolicy::Cadence, AverageBackend::Cpu),
            Self::CpuAsync => (ApplyPolicy::Async, AverageBackend::Cpu),
        }
    }

    /// Recompose from the `(ApplyPolicy, AverageBackend)` pair — the
    /// inverse of [`Self::split`]. Used to reconcile the builder's
    /// transient `policy`/`backend` into the canonical `ElCheConfig::mode`
    /// at build time. `(Async, Nccl)` has no mode (NcclAsync was dropped),
    /// so it falls back to `NcclCadence` (debug-asserts in dev) — the
    /// orchestrator never produces that pair.
    pub(crate) fn from_parts(policy: ApplyPolicy, backend: AverageBackend) -> Self {
        match (policy, backend) {
            (ApplyPolicy::Sync, AverageBackend::Nccl) => Self::NcclSync,
            (ApplyPolicy::Cadence, AverageBackend::Nccl) => Self::NcclCadence,
            (ApplyPolicy::Sync, AverageBackend::Cpu) => Self::CpuSync,
            (ApplyPolicy::Cadence, AverageBackend::Cpu) => Self::CpuCadence,
            (ApplyPolicy::Async, AverageBackend::Cpu) => Self::CpuAsync,
            (ApplyPolicy::Async, AverageBackend::Nccl) => {
                debug_assert!(
                    false,
                    "NcclAsync was dropped; (Async, Nccl) has no ElCheMode"
                );
                Self::NcclCadence
            }
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
#[derive(Clone)]
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
    /// ElChe per-window fixed-overhead target (reduce + fill, as a fraction
    /// of the bottleneck rank's window wall). `None` = framework default
    /// (0.05).
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
    /// Divergence guardrail. `None` = `TrendGuard` at the EASGD-aware
    /// default threshold (0.05 for overwrite modes; 0.3 when
    /// `easgd_alpha` is set, whose elastic standing spread would keep a
    /// lower floor permanently armed); set to a custom guard to
    /// override threshold or replace behavior. See
    /// [`crate::distributed::ddp_run::NoGuard`] /
    /// [`crate::distributed::ddp_run::TrendGuard`].
    pub convergence_guard: Option<Box<dyn ConvergenceGuard>>,
    /// EASGD elastic-averaging weight (0.0 < α ≤ 1.0). Honored on the
    /// CpuAsync path only; ignored elsewhere.
    pub easgd_alpha: Option<f64>,
    /// Divergence threshold for the default convergence guard
    /// ([`crate::distributed::ddp_run::TrendGuard`]). `None` = framework
    /// default, keyed on param-adoption semantics: 0.05 for overwrite
    /// modes, 0.3 when [`Self::easgd_alpha`] is set (elastic blending
    /// keeps a deliberate standing spread ~0.1 that a lower floor would
    /// read as permanent divergence). Ignored when
    /// [`Self::convergence_guard`] supplies a custom guard (the override
    /// takes precedence) or when [`Self::no_divergence_guard`] is set.
    pub divergence_threshold: Option<f64>,
    /// Disable the convergence guard entirely
    /// ([`crate::distributed::ddp_run::NoGuard`]). Default `false`. When
    /// `true`, ElChe's overhead auto-tune drives cadence alone. Ignored
    /// when [`Self::convergence_guard`] supplies a custom guard.
    pub no_divergence_guard: bool,
    /// Max batches a rank may run past the planned sync point (CpuAsync
    /// streaming bound). `None` = auto-tuned from convergence feedback.
    /// The async-strategy lookahead knob; ignored outside CpuAsync.
    pub max_overshoot: Option<usize>,
    /// Consensus allocation-weighting exponent: rank `k` is weighted
    /// `nₖ^γ` in the work-weighted average (`nₖ` = batches processed this
    /// window). `γ = 1.0` (default) is plain work-weighting (more data =
    /// proportionally more weight); `γ = 0.0` is an unweighted average
    /// (each rank equal regardless of steps); `γ = −1.0` equalizes
    /// per-step trust. A single knob sweeping data-volume ↔ per-step
    /// fairness, primarily a diagnostic for the source of the
    /// heterogeneous-cadence regularization effect. Honored on BOTH
    /// backends: the CPU path applies it in the frame weighting, the
    /// NCCL path folds it into the fused PreMulSum factor. Idle ranks
    /// (`nₖ = 0`) contribute zero mass for any `γ` (the idle guard in
    /// the shared `realized_work` vocabulary).
    pub gamma: f64,
    /// Ship the CPU-averaging plane's model traffic (params / buffers)
    /// as bfloat16 instead of f32, halving the per-sync payload at
    /// every hop: the rank's pinned parameter snapshot, the rank→relay
    /// frames, the per-host fold shipped upstream, and the scattered
    /// consensus. Default `false` (f32, byte-identical to prior
    /// behavior).
    ///
    /// The averaging math itself stays in f32 — every accumulator (the
    /// relay fold, the controller's sum, the outer optimizer state)
    /// decodes bf16 at the boundary and re-encodes after; bf16 exists
    /// only on the wire and in the staging buffers. Control traffic
    /// (count gathers, the formation broadcast) always rides f32, so
    /// bookkeeping and initial state stay byte-exact. Consensus
    /// checkpoints are written f32 regardless.
    ///
    /// The cost is a bf16 quantization of the consensus each sync
    /// (~3 significant decimal digits, the same wire precision the
    /// bf16-compression hooks in other DDP stacks accept). Honored on
    /// the CPU backends in cluster (process-per-rank) runs; ignored by
    /// the NCCL modes (their reduce never leaves the GPUs) and by the
    /// single-process thread path. Must be set identically on every
    /// rank — a mixed cohort fails loudly at the first fold with a
    /// frame-schema dtype mismatch.
    pub bf16_wire: bool,
}

impl ElCheConfig {
    // -- Presets ----------------------------------------------------------

    /// NCCL averaging at the tightest cadence (reduce per slow-rank
    /// step; see [`ElCheMode::NcclSync`]). anchor=1, no overshoot.
    pub fn nccl_sync() -> Self {
        Self {
            mode: ElCheMode::NcclSync,
            anchor: 1,
            ..Self::default_for(ElCheMode::NcclSync)
        }
    }

    /// NCCL averaging, anchor-based. Default anchor 10. Recommended
    /// NCCL default for mixed-GPU rigs.
    pub fn nccl_cadence() -> Self {
        Self::default_for(ElCheMode::NcclCadence)
    }

    /// CPU-mediated averaging at the tightest cadence (reduce per
    /// slow-rank step; see [`ElCheMode::CpuSync`]).
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
            overhead_target: Some(0.05),
            ..Self::default_for(ElCheMode::CpuAsync)
        }
    }

    pub(crate) fn default_for(mode: ElCheMode) -> Self {
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
            // cpu-async defaults to the EASGD elastic blend (α=0.5, the
            // value the MSF study used). The `None`/full-overwrite path
            // is α=1.0, which discards the ahead-of-sync local progress
            // cpu-async accumulates between reduces — the degenerate
            // mode. Mode-gated: `easgd_alpha` drives `load_averaged`'s
            // blend regardless of policy, and Sync/Cadence MUST
            // full-overwrite to the consensus each window. Override via
            // [`Self::easgd_alpha`].
            easgd_alpha: match mode {
                ElCheMode::CpuAsync => Some(0.5),
                _ => None,
            },
            divergence_threshold: None,
            no_divergence_guard: false,
            max_overshoot: None,
            // Plain work-weighting (nₖ¹): the production default, byte-identical
            // to pre-gamma behavior.
            gamma: 1.0,
            // f32 wire: byte-identical to prior behavior; bf16 is opt-in
            // until the convergence A/B says otherwise.
            bf16_wire: false,
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
    /// Set the default convergence-guard divergence threshold.
    pub fn divergence_threshold(mut self, t: f64) -> Self { self.divergence_threshold = Some(t); self }
    /// Disable the convergence guard (overhead auto-tune drives cadence alone).
    pub fn no_divergence_guard(mut self) -> Self { self.no_divergence_guard = true; self }
    /// Set the CpuAsync streaming lookahead bound (max batches past the planned sync).
    pub fn max_overshoot(mut self, n: usize) -> Self { self.max_overshoot = Some(n); self }
    /// Set the consensus allocation-weighting exponent `γ` (see [`Self::gamma`]).
    /// `1.0` = plain work-weighting (default), `0.0` = unweighted average,
    /// `−1.0` = per-step-equal. Honored on both backends.
    pub fn gamma(mut self, g: f64) -> Self { self.gamma = g; self }
    /// Ship the CPU-averaging plane's model traffic as bfloat16 (see
    /// [`Self::bf16_wire`]). Halves snapshots, fold traffic, and wire
    /// payloads; averaging still accumulates in f32.
    pub fn bf16_wire(mut self, on: bool) -> Self { self.bf16_wire = on; self }
}

impl Default for ElCheConfig {
    /// Default = [`Self::nccl_cadence`].
    ///
    /// Recommended NCCL mode for heterogeneous rigs: anchor-based
    /// cadence with ElChe tuning the slow-device-anchored pace. Fast
    /// GPUs process proportionally more batches per averaging window;
    /// AllReduce coordinates at every cadence boundary. For decoupled
    /// asynchronous averaging on heterogeneous setups, see
    /// [`Self::cpu_async`].
    fn default() -> Self {
        Self::nccl_cadence()
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
            .field("divergence_threshold", &self.divergence_threshold)
            .field("no_divergence_guard", &self.no_divergence_guard)
            .field("max_overshoot", &self.max_overshoot)
            .field("gamma", &self.gamma)
            .field("bf16_wire", &self.bf16_wire)
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
    /// override with a preset (e.g. `ElCheConfig::nccl_sync()` for the
    /// tightest cadence — vanilla DDP on homogeneous rigs) when
    /// appropriate.
    pub elche: ElCheConfig,

    /// Maximum gradient norm for per-worker clipping. `None` = no clipping.
    pub max_grad_norm: Option<f64>,
    /// Device-resident sample pool on rank workers (default `true`):
    /// leftover VRAM retains samples after the first training step so
    /// later epochs gather them on device instead of re-uploading them.
    /// Sizing is automatic; `FLODL_VRAM_POOL=off` in a worker's `env:`
    /// block is the runtime kill-switch.
    pub vram_pool: bool,
    /// Augmentation multiplicity: each sample appears `k` times per
    /// epoch in the shared shuffle (pick space `dataset.len() * k`).
    /// Pure scheduling — data variation comes exclusively from
    /// [`Self::transform`], keyed per pick. Default: `1`.
    pub augment: usize,
    /// Deterministic delivery transform (the augmentation seam), keyed
    /// by [`crate::data::PickKey`] and applied on each rank at its
    /// delivery point, on freshly assembled raw rows. Default: `None`.
    pub transform: Option<crate::data::TransformFn>,
    /// Fraction of **total** VRAM each rank worker may use for its
    /// data plane (prefetch channel + device sample pool), clamped to
    /// `[0.50, 0.99]`. Default `0.90` — same knob and default as
    /// `DataLoaderBuilder::vram_max_usage` on the solo path.
    pub vram_max_usage: f64,
    /// Fraction of currently **available** host RAM (`MemAvailable`)
    /// each rank's staging tiers may retain, clamped to `[0.0, 0.90]`;
    /// co-hosted ranks split the share consumption-proportionally.
    /// `0.0` disables staging. Default `0.50` — same knob and default
    /// as `DataLoaderBuilder::ram_max_usage` on the solo path.
    pub ram_max_usage: f64,
    /// Fraction of **physical** host RAM (`MemTotal`) to hand the GPU on
    /// an **integrated (APU) target**, where device memory is carved out
    /// of system RAM rather than being a pool of its own — so the host
    /// staging tiers and the VRAM pool otherwise price the same DRAM
    /// twice and over-commit it.
    ///
    /// `None` (default) reserves whatever aperture the device reports.
    /// Ignored on discrete GPUs, where the two pools are genuinely
    /// separate. Values above `1.0` are allowed and meaningful: if a
    /// platform under-reports `MemTotal` relative to what the APU can
    /// address, a share above 1.0 is how you still express the true
    /// reservation. Same knob as `DataLoaderBuilder::gpu_ram_share` on
    /// the solo path.
    pub gpu_ram_share: Option<f64>,
    /// Pinned RAM sample retention on rank workers (default `true`):
    /// the staging tier's read-through cache keeps fetched samples for
    /// later epochs within the [`Self::ram_max_usage`] budget. `false`
    /// pins that cache's budget to zero — the flow window (in-order
    /// stream staging) keeps the whole staging share and nothing is
    /// retained across epochs. Same knob as
    /// `DataLoaderBuilder::sample_cache` on the solo path.
    pub sample_cache: bool,
    /// Local-disk overflow tier under each rank's sample cache, in GB
    /// (default `0` = off): samples evicted from (or never admitted to)
    /// the RAM budget spill to an ephemeral per-rank pack file and read
    /// back from local disk instead of the (possibly remote) source.
    /// Same knob as `DataLoaderBuilder::disk_stage` on the solo path.
    pub disk_stage_gb: u64,
    /// Directory for the disk-stage pack file (default: the system
    /// temp dir). Point it at a fast local drive when `/tmp` is small
    /// or RAM-backed. Same knob as `DataLoaderBuilder::disk_stage_dir`.
    pub disk_stage_dir: Option<std::path::PathBuf>,
    /// Cluster-mode stop threshold: how many ranks may be lost (spot
    /// reclaims, hardware, network) before the run is declared
    /// unrecoverable and survivors save-and-shutdown. `None` (default)
    /// tolerates any partial loss — a single rank vanishing from a
    /// large collective never kills the training; only losing every
    /// rank (or the backend hard floor: NCCL needs 2 survivors) stops
    /// it. Deaths and redistribution are surfaced in logs and on the
    /// live dashboard either way.
    pub max_failure: Option<crate::distributed::max_failure::MaxFailureThreshold>,
    /// Save a checkpoint every N global epochs. `None` = no periodic save.
    pub checkpoint_every: Option<usize>,
    /// Checkpoint bundle stem for cluster-mode unrecoverable-failure
    /// persistence. See [`crate::distributed::CheckpointBundle`].
    pub save_path: Option<String>,
    /// Resume from a previously-saved checkpoint bundle stem.
    pub resume_from: Option<String>,
    /// Arm a one-shot coverage-granular checkpoint at the first reduce
    /// where the cohort reaches this epoch (cluster progressive modes:
    /// Cadence / Async). The forged consensus model is written to
    /// `<save_path>.fdl` and the trajectory + data-coverage to
    /// `<save_path>.meta.json`; pair with [`Self::save_path`]. `None` =
    /// no mid-run checkpoint. See [`crate::distributed::CheckpointBundle`].
    pub checkpoint_at_epoch: Option<usize>,

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
    /// Eval cadence in epochs. `None` + an `eval_fn` registered =
    /// every epoch (a wired eval pipeline that silently never fires is
    /// a bug, not a default); `None` without an `eval_fn` = no evals.
    /// Set explicitly to space evals out.
    pub eval_every: Option<usize>,

    /// Sub-epoch monitor reports per epoch: up to `n` per-window metric
    /// reports at reduce boundaries, filling the curve *between* epoch
    /// points. `None` or `0` = off (per-epoch reporting only). Decisive
    /// for single-epoch (one-pass LLM) runs, where the per-epoch feed is
    /// a single point.
    pub reports_per_epoch: Option<usize>,

    /// Directory for the persisted monitor record stream (append-only
    /// JSONL, one bounded file per node mirroring its `path`). `None` =
    /// live-only, no persistence.
    pub record_log_dir: Option<String>,

    /// Per-node byte cap for the persisted record stream (drop-oldest
    /// ring), so a long run cannot fill the disk. `None` =
    /// [`DEFAULT_MAX_LOG_BYTES`](crate::monitor::record_log::DEFAULT_MAX_LOG_BYTES).
    pub max_log_size: Option<u64>,

    /// Where to write the self-contained dashboard archive at teardown, or
    /// `None` for live-only. One HTML file carrying the epoch feed, the record
    /// plane and the graph SVG, openable with no server.
    pub dashboard_html: Option<String>,

    /// Theme the saved dashboard opens with: `None` (default) follows the
    /// reader's `prefers-color-scheme`; `"light"` pins it for publication.
    pub dashboard_theme: Option<String>,

    /// How each **user scalar** rolls up across ranks in the record tree.
    ///
    /// Non-core keys default to
    /// [`Reduction::Mean`](crate::monitor::record::Reduction::Mean), which is
    /// wrong for a count (`tokens_seen`) or an extremum (`peak_mem_gb`) — and
    /// the portal *states* the reduction in its legend, so an undeclared count
    /// asserts something false rather than merely getting it wrong quietly.
    /// Declarations reach every consumer via the record stream's `meta` record.
    /// Empty by default; core keys ignore any entry here.
    pub scalar_reductions: crate::monitor::record::Reductions,

    /// Which rank fires user-supplied per-epoch callbacks. Default
    /// [`EpochCallbackPolicy::Fastest`].
    pub epoch_callback_policy: EpochCallbackPolicy,

    /// Optional high-frequency system timeline for profiling.
    pub timeline: Option<Arc<crate::monitor::Timeline>>,

    /// Programmatic cluster topology. When `Some`, [`Trainer::run`]
    /// promotes this process to launcher role with the given cluster
    /// (same effect as setting `FLODL_INTERNAL_FULL_CLUSTER_JSON` via fdl-cli's
    /// overlay parsing — the launcher contract is single-shape).
    ///
    /// Three precedence cases at `Trainer::run` time:
    /// - `FLODL_INTERNAL_FULL_CLUSTER_JSON` already set in env (fdl-cli set
    ///   it before invoking the user binary) → that wins; this field
    ///   is ignored.
    /// - `cfg.cluster = Some(...)` and env unset → field wins; serializes
    ///   into `FLODL_INTERNAL_FULL_CLUSTER_JSON` and dispatch fires Launcher.
    /// - Neither set + 2+ visible GPUs → auto-promote synthesizes a
    ///   localhost cluster via [`super::ClusterBuilder::all_local_gpus`].
    /// - Neither set + ≤1 GPU → single-device path.
    ///
    /// Construct via [`super::ClusterBuilder`].
    ///
    /// [`Trainer::run`]: crate::distributed::Trainer::run
    pub cluster: Option<FullCluster>,

    /// Outer optimizer applied to the consensus between reduce and broadcast
    /// (SlowMo / DiLoCo). `None` = today's plain averaging
    /// ([`crate::distributed::OuterAvg`]). Built once per site (controller on
    /// CPU, per rank on NCCL) from this factory. Set via
    /// [`Self::outer_optimizer`].
    pub outer_optimizer: Option<crate::distributed::outer_optimizer::OuterOptimizerFactory>,
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
            vram_pool: crate::data::vram_pool::VRAM_POOL_DEFAULT,
            augment: 1,
            transform: None,
            vram_max_usage: 0.90,
            ram_max_usage: 0.50,
            gpu_ram_share: None,
            sample_cache: true,
            disk_stage_gb: 0,
            disk_stage_dir: None,
            max_failure: None,
            checkpoint_every: None,
            save_path: None,
            resume_from: None,
            checkpoint_at_epoch: None,
            checkpoint_fn: None,
            epoch_fn: None,
            metrics_fn: None,
            scheduler_fn: None,
            eval_fn: None,
            eval_dataset: None,
            eval_result_fn: None,
            eval_every: None,
            reports_per_epoch: None,
            record_log_dir: None,
            dashboard_html: None,
            dashboard_theme: None,
            scalar_reductions: crate::monitor::record::Reductions::new(),
            max_log_size: None,
            epoch_callback_policy: EpochCallbackPolicy::default(),
            timeline: None,
            cluster: None,
            outer_optimizer: None,
        }
    }

    /// Like [`new`](Self::new), from a per-sample [`crate::data::DataSet`]
    /// (the one-method `get(index)` contract).
    ///
    /// Batching, caching, and staging are the framework's job: samples
    /// flow through the same read-through tiers the `DataLoader` uses,
    /// and DDP rank workers stage them ahead of the training frontier
    /// per their reservations. Implement `get()` (or use a shipped
    /// reader like [`crate::data::datasets::Cifar10Disk`]) and hand it
    /// here — no custom loader needed, at any dataset size.
    pub fn from_dataset(dataset: impl crate::data::DataSet + 'static) -> Self {
        Self::new(crate::data::batch_dataset_from(dataset))
    }

    // -- Chained setters --------------------------------------------------

    /// Set the batch size.
    pub fn batch_size(mut self, n: usize) -> Self { self.batch_size = n; self }

    /// Enable / disable the rank workers' device-resident sample pool
    /// (see [`Self::vram_pool`]).
    pub fn with_vram_pool(mut self, enabled: bool) -> Self { self.vram_pool = enabled; self }

    /// Augmentation multiplicity (see [`Self::augment`]).
    pub fn with_augment(mut self, k: usize) -> Self { self.augment = k.max(1); self }

    /// VRAM share for each rank's data plane (see [`Self::vram_max_usage`]).
    pub fn with_vram_max_usage(mut self, max_usage: f64) -> Self {
        self.vram_max_usage = max_usage.clamp(0.50, 0.99);
        self
    }

    /// Host-RAM share for each rank's staging tiers (see
    /// [`Self::ram_max_usage`]).
    pub fn with_ram_max_usage(mut self, max_usage: f64) -> Self {
        self.ram_max_usage = max_usage.clamp(0.0, 0.90);
        self
    }
    /// Fraction of physical host RAM (`MemTotal`) reserved for the GPU on
    /// an integrated (APU) target (see [`Self::gpu_ram_share`]). Ignored
    /// on discrete GPUs. Same knob as
    /// `DataLoaderBuilder::gpu_ram_share` on the solo path.
    pub fn with_gpu_ram_share(mut self, share: f64) -> Self {
        self.gpu_ram_share = Some(share.max(0.0));
        self
    }


    /// Pinned RAM sample retention on rank workers (see
    /// [`Self::sample_cache`]).
    pub fn with_sample_cache(mut self, enabled: bool) -> Self {
        self.sample_cache = enabled;
        self
    }

    /// Local-disk overflow tier in GB (see [`Self::disk_stage_gb`]).
    pub fn with_disk_stage(mut self, gb: u64) -> Self {
        self.disk_stage_gb = gb;
        self
    }

    /// Disk-stage directory (see [`Self::disk_stage_dir`]).
    pub fn with_disk_stage_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.disk_stage_dir = Some(dir.into());
        self
    }

    /// Delivery transform (see [`Self::transform`]).
    pub fn with_transform(
        mut self,
        f: impl Fn(Vec<crate::tensor::Tensor>, &[crate::data::PickKey]) -> crate::tensor::Result<Vec<crate::tensor::Tensor>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.transform = Some(crate::data::TransformFn::new(f));
        self
    }

    /// Set the outer optimizer (SlowMo / DiLoCo) applied to the consensus.
    /// The factory is invoked once per site (controller on CPU, per rank on
    /// NCCL). Absent = today's plain averaging. See
    /// [`crate::distributed::OuterOptimizer`].
    pub fn outer_optimizer<P>(mut self, factory: P) -> Self
    where
        P: Fn() -> Box<dyn crate::distributed::OuterOptimizer> + Send + Sync + 'static,
    {
        self.outer_optimizer = Some(Arc::new(factory));
        self
    }
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
    /// Arm a one-shot mid-run checkpoint at the given epoch. Pairs with
    /// [`Self::save_path`]. See [`Self::checkpoint_at_epoch`].
    pub fn checkpoint_at_epoch(mut self, n: usize) -> Self { self.checkpoint_at_epoch = Some(n); self }

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
    /// Set the eval cadence in epochs (see the `eval_every` field for
    /// the default when an `eval_fn` is registered without a cadence).
    pub fn eval_every(mut self, n: usize) -> Self { self.eval_every = Some(n); self }

    /// Emit up to `n` sub-epoch monitor reports per epoch, at reduce
    /// boundaries (see the `reports_per_epoch` field). `0` disables.
    pub fn reports_per_epoch(mut self, n: usize) -> Self {
        self.reports_per_epoch = if n == 0 { None } else { Some(n) };
        self
    }

    /// Persist the monitor record stream as JSONL under `dir`, each node
    /// capped at `max_bytes` (drop-oldest; `0` = the library default).
    pub fn record_log(mut self, dir: impl Into<String>, max_bytes: u64) -> Self {
        self.record_log_dir = Some(dir.into());
        self.max_log_size = if max_bytes == 0 { None } else { Some(max_bytes) };
        self
    }
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
    /// precedence with `FLODL_INTERNAL_FULL_CLUSTER_JSON` env-var and the
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
            ElCheMode::NcclSync, ElCheMode::NcclCadence,
            ElCheMode::CpuSync, ElCheMode::CpuCadence, ElCheMode::CpuAsync,
        ] {
            let (p, b) = m.split();
            let expected_backend = match m {
                ElCheMode::NcclSync | ElCheMode::NcclCadence => AverageBackend::Nccl,
                ElCheMode::CpuSync | ElCheMode::CpuCadence | ElCheMode::CpuAsync => AverageBackend::Cpu,
            };
            assert_eq!(b, expected_backend, "{:?} backend split", m);
            let expected_policy = match m {
                ElCheMode::NcclSync | ElCheMode::CpuSync => ApplyPolicy::Sync,
                ElCheMode::NcclCadence | ElCheMode::CpuCadence => ApplyPolicy::Cadence,
                ElCheMode::CpuAsync => ApplyPolicy::Async,
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
    fn elche_default_is_nccl_cadence() {
        let cfg = ElCheConfig::default();
        assert_eq!(cfg.mode, ElCheMode::NcclCadence);
        assert_eq!(cfg.anchor, 10);
    }

    #[test]
    fn meta_controller_default_is_on() {
        let cfg = ElCheConfig::default();
        assert!(cfg.meta_controller, "meta_controller defaults to true");
        // Spot-check all presets agree on the default.
        for preset in [
            ElCheConfig::nccl_sync(),
            ElCheConfig::nccl_cadence(),
            ElCheConfig::cpu_sync(),
            ElCheConfig::cpu_cadence(),
            ElCheConfig::cpu_async(),
        ] {
            assert!(preset.meta_controller, "preset for {:?}", preset.mode);
        }
    }
}
