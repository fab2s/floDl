//! DDP mode and run configuration.

use std::fmt;

use flodl::distributed::{ApplyPolicy, AverageBackend};

/// A DDP training mode to benchmark.
#[derive(Debug, Clone)]
pub enum DdpMode {
    /// Single GPU, no DDP.
    Solo(usize),
    /// Thread-per-GPU via `Trainer::builder()`.
    Builder {
        policy: ApplyPolicy,
        backend: AverageBackend,
    },
}

impl DdpMode {
    /// Parse a mode string like "solo-0", "nccl-sync", "nccl-cadence", "cpu-async".
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
            "nccl-async" => Some(DdpMode::Builder {
                policy: ApplyPolicy::Async,
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
            "nccl-async",
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
                };
                let p = match policy {
                    ApplyPolicy::Sync => "sync",
                    ApplyPolicy::Cadence => "cadence",
                    ApplyPolicy::Async => "async",
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
/// `DdpBuilder::convergence_guard`. Default is `Trend` with the production
/// threshold (matches pre-pluggable behavior).
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
}
