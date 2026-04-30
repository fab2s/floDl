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
}
