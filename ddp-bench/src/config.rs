//! DDP mode and run configuration.

use std::fmt;

use flodl::distributed::{ApplyPolicy, AverageBackend};

/// A DDP training mode to benchmark.
#[derive(Debug, Clone)]
pub enum DdpMode {
    /// Single GPU, no DDP.
    Solo(usize),
    /// Framework-managed DDP via `Trainer::builder()` (process-per-rank
    /// on multi-GPU rigs).
    Builder {
        policy: ApplyPolicy,
        backend: AverageBackend,
    },
}

impl DdpMode {
    /// Parse a mode string like "solo-0", "nccl-sync", "nccl-cadence", "cpu-async".
    /// `nccl-async` was dropped (cross-epoch lookahead on NCCL gave near-zero
    /// real-world speedup vs `nccl-cadence` while complicating the dispatch
    /// path; CPU Async is the genuine async mode).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "nccl-sync" => Some(DdpMode::Builder {
                policy: ApplyPolicy::Sync,
                backend: AverageBackend::Nccl,
            }),
            "nccl-cadence" => Some(DdpMode::Builder {
                policy: ApplyPolicy::Cadence,
                backend: AverageBackend::Nccl,
            }),
            "cpu-sync" => Some(DdpMode::Builder {
                policy: ApplyPolicy::Sync,
                backend: AverageBackend::Cpu,
            }),
            "cpu-cadence" => Some(DdpMode::Builder {
                policy: ApplyPolicy::Cadence,
                backend: AverageBackend::Cpu,
            }),
            "cpu-async" => Some(DdpMode::Builder {
                policy: ApplyPolicy::Async,
                backend: AverageBackend::Cpu,
            }),
            _ if s.starts_with("solo-") => s[5..].parse::<usize>().ok().map(DdpMode::Solo),
            _ => None,
        }
    }

    /// All known mode names.
    pub fn all_names() -> &'static [&'static str] {
        &[
            "solo-0",
            "solo-1",
            "solo-2",
            "nccl-sync",
            "nccl-cadence",
            "cpu-sync",
            "cpu-cadence",
            "cpu-async",
        ]
    }

    /// Whether this mode requires multiple GPUs.
    pub fn requires_multi_gpu(&self) -> bool {
        !matches!(self, DdpMode::Solo(_))
    }
}

impl fmt::Display for DdpMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DdpMode::Solo(idx) => write!(f, "solo-{idx}"),
            DdpMode::Builder { policy, backend } => {
                let b = match backend {
                    AverageBackend::Nccl => "nccl",
                    AverageBackend::Cpu => "cpu",
                    _ => "unknown-backend",
                };
                let p = match policy {
                    ApplyPolicy::Sync => "sync",
                    ApplyPolicy::Cadence => "cadence",
                    ApplyPolicy::Async => "async",
                    _ => "unknown-policy",
                };
                write!(f, "{b}-{p}")
            }
        }
    }
}

/// Default training parameters for a model.
#[derive(Debug, Clone)]
pub struct ModelDefaults {
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub lr: f64,
}

/// Convergence guard selection. Materialised by the harness into a concrete
/// `flodl::distributed::ddp_run::ConvergenceGuard` and passed through
/// `DdpBuilder::convergence_guard`. Default is `Trend` at the production
/// threshold.
#[derive(Debug, Clone)]
pub enum GuardChoice {
    /// Pass-through: no convergence-driven anchor adjustments. ElChe's
    /// overhead auto-tune drives cadence alone.
    None,
    /// 3-rises-above-threshold rule on `||pre - post|| / ||post||`.
    Trend { threshold: f64 },
    /// Rate-based detector with soft (`SuppressGrowth`) + hard (`NudgeDown`)
    /// thresholds on the bias-corrected `λ_ema`.
    Msf {
        suppress_threshold: f64,
        suppress_sustain: usize,
        nudge_threshold: f64,
        nudge_sustain: usize,
        nudge_factor: f64,
        alpha: f64,
    },
}

impl Default for GuardChoice {
    fn default() -> Self {
        GuardChoice::Trend { threshold: 0.01 }
    }
}

/// Outer-optimizer selection. Materialised by the harness into a
/// `flodl::distributed::OuterOptimizer` factory passed to
/// `DdpBuilder::outer_optimizer`. Default `None` reproduces today's plain
/// weighted averaging (`OuterAvg`). Honored on the CPU backend (the
/// consensus is forged controller-side there); the NCCL per-rank site is a
/// follow-on.
#[derive(Debug, Clone, Default)]
pub enum OuterOptChoice {
    /// No outer optimizer: plain weighted averaging (`OuterAvg`).
    #[default]
    None,
    /// SlowMo heavy-ball slow momentum on the pseudo-gradient
    /// `g = prev_global - consensus`.
    SlowMomentum { lr: f64, mu: f64 },
    /// DiLoCo Nesterov momentum on the pseudo-gradient + disposable inner
    /// optimizer (worker resets its inner optimizer each outer round). Param
    /// adoption follows the mode: full-overwrite on cadence, EASGD-blended on
    /// cpu-async (the inner reset is orthogonal to param adoption).
    Nesterov { lr: f64, mu: f64 },
}

/// Runtime configuration for a single benchmark run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub seed: u64,
    pub output_dir: String,
    pub data_dir: std::path::PathBuf,
    pub monitor_port: Option<u16>,
    /// Explicit per-rank partition ratios for heterogeneous DDP. When set,
    /// passed to `DdpBuilder::partition_ratios` to disable the uniform
    /// default and dispatch batches in proportion. Length must match the
    /// visible GPU count and values must sum to ~1.0.
    pub partition_ratios: Option<Vec<f64>>,
    /// Enable ElChe's anchor relax-up on stable convergence verdicts.
    /// When true, passed as `DdpBuilder::elche_relax_up(true)`. Default
    /// false (relax-up disabled, anchor under overhead-based auto-tune only).
    pub elche_relax_up: bool,
    /// Enable the LR-aware meta-controller above ElChe. When true, passed
    /// as `DdpBuilder::meta_controller(true)`. Default false (opt-in until
    /// validation sweep).
    pub meta_controller: bool,
    /// Override ElChe's `max_anchor` (anchor upper bound, default 1000).
    /// Used by Sweep C to bracket the Pecora-Carroll synchronization
    /// threshold by walking k_max across multiples of the default. `None`
    /// preserves library default.
    pub max_anchor: Option<usize>,
    /// Override ElChe's `min_anchor` (anchor lower bound, defaults to the
    /// initial anchor). Forces the overhead auto-tune above its natural
    /// equilibrium. Pair with `max_anchor` set to the same value plus
    /// `guard=NoGuard` to pin the cadence at exactly N batches per cycle
    /// (Sweep B fixed-k probe). `None` preserves library default.
    pub min_anchor: Option<usize>,
    /// EASGD elastic averaging weight α. When `Some`, the cpu-async
    /// `load_averaged` path blends `W_local := (1-α)·W_local + α·W_avg`
    /// instead of full overwrite. `None` preserves current behavior.
    /// Honored on cpu-async only. Reference: Zhang, Choromanska, LeCun
    /// NeurIPS 2015.
    pub easgd_alpha: Option<f64>,
    /// Max batches a rank may run past its planned sync point (CpuAsync
    /// lookahead bound). `None` lets the framework auto-tune; `Some(n)`
    /// pins the ceiling (high `n` => convergence-guard-governed overshoot
    /// instead of a hard cap). Honored on cpu-async only.
    pub max_overshoot: Option<usize>,
    /// Run `eval_fn` at the end of every epoch (rank 0 only) and emit
    /// `epoch N: ... eval=X.XXXX` into `training.log` so the analysis
    /// pipeline can correlate λ̂ aggregates with held-out metric per epoch.
    /// Default false (only the post-training `final eval=...` line is
    /// emitted, matching the historical bench behavior).
    ///
    /// In Sync mode the eval is on consensus params (all ranks identical
    /// post-AllReduce). In Cadence/Async modes the eval is on rank-0's
    /// state at the start of the next epoch — near-consensus but not
    /// exact, since the coordinator doesn't force an AllReduce at the
    /// epoch boundary. Trend-correlation analyses are robust to that
    /// noise.
    pub per_epoch_eval: bool,
    /// Convergence-guard configuration. Default = `GuardChoice::Trend`
    /// with production threshold 0.01.
    pub guard: GuardChoice,
    /// Which rank fires user epoch callbacks (`epoch_fn`,
    /// `checkpoint_fn`, `eval_fn`). When `None`, the framework default
    /// (`Rank(0)`) is used. Set to `Fastest` to let ElChe pick the
    /// rank with the lowest `smoothed_ms_per_batch` at run start
    /// (re-resolved on rank death).
    ///
    /// Only meaningful for multi-rank cluster paths (the framework
    /// loud-errors if `Fastest` is configured on a non-via_coord run).
    /// Solo modes ignore this field.
    pub epoch_callback_policy: Option<flodl::distributed::ddp_run::EpochCallbackPolicy>,
    /// Cluster-mode checkpoint bundle stem (save side). When set, passed to
    /// `DdpBuilder::save_path` so the consensus forge writes `<stem>.fdl` +
    /// `<stem>.meta.json` on a mid-run checkpoint or unrecoverable failure.
    pub save_path: Option<String>,
    /// Cluster stop threshold (absolute rank-loss count) forwarded to
    /// `DdpBuilder::max_failure`. `None` tolerates any partial loss.
    pub max_failure: Option<usize>,
    /// Resume a cluster run from a previously-saved bundle stem. The model
    /// factory loads `<stem>.fdl` consensus weights into each freshly-built
    /// replica, and the coordinator reconstructs the saved data-coverage so
    /// only the uncovered remainder is dispatched (no data repeated).
    /// Progressive (cadence/async) cluster modes only.
    pub resume_from: Option<String>,
    /// Arm a one-shot coverage-granular checkpoint at the first reduce where
    /// the cohort reaches this epoch. Pairs with `save_path`. Progressive
    /// (cadence/async) cluster modes only.
    pub checkpoint_at_epoch: Option<usize>,
    /// Outer optimizer applied to the consensus between reduce and broadcast
    /// (SlowMo / DiLoCo A/B arm). Default `None` = plain weighted averaging.
    pub outer_optimizer: OuterOptChoice,
    /// Consensus allocation-weighting exponent `γ` (rank weighted `nₖ^γ`).
    /// `1.0` = plain work-weighting (default). CPU backend only.
    pub gamma: f64,
}
