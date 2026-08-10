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

/// Execution tier: who owns the training loop.
///
/// `Managed` is the framework-driven path (`Trainer::builder().run()`): the
/// launcher narrates, the ranks self-drive. `Cooperative` is the decomposed
/// path (`.into_worker()`): the loop is hand-written in the bench and scales
/// unchanged from one device to N to a cluster, while the controller still
/// owns cadence / partition / averaging / eval-rank election. Same builder
/// config feeds both — only the terminal differs — so a cooperative run is
/// the managed run's parity twin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tier {
    /// Framework owns the loop (default).
    #[default]
    Managed,
    /// User (this harness) owns the loop via a `Worker`.
    Cooperative,
}

impl Tier {
    /// Parse `--tier managed|cooperative`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "managed" => Some(Tier::Managed),
            "cooperative" => Some(Tier::Cooperative),
            _ => None,
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Managed => write!(f, "managed"),
            Tier::Cooperative => write!(f, "cooperative"),
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
/// `DdpBuilder::convergence_guard`. Default is `Level` at the production
/// threshold.
#[derive(Debug, Clone)]
pub enum GuardChoice {
    /// Pass-through: no convergence-driven anchor adjustments. ElChe's
    /// overhead auto-tune drives cadence alone.
    None,
    /// Divergence-LEVEL detector: 3-rises-above-threshold rule on
    /// `||pre - post|| / ||post||`. `None` defers to the library default,
    /// which is EASGD-aware (0.05 overwrite modes / 0.3 when α-blending keeps
    /// a standing spread) and absorbs the saved history on resume.
    Level { threshold: Option<f64> },
    /// Divergence-growth-rate detector with soft (`SuppressGrowth`) + hard
    /// (`NudgeDown`) thresholds on the bias-corrected `λ_ema`.
    Growth {
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
        GuardChoice::Level { threshold: None }
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
    /// Where training data lives during the run (RAM tensors vs
    /// per-sample reads from the raw files). See
    /// [`crate::models::DataSource`].
    pub data_source: crate::models::DataSource,
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
    /// Convergence-guard configuration. Default = `GuardChoice::Level`
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
    /// Ship the CPU-averaging plane's model traffic as bfloat16
    /// (`--bf16-wire`): halves pinned snapshots, relay fold traffic, and
    /// wire payloads; averaging still accumulates in f32. CPU averaging
    /// modes only — the builder errors loudly on nccl-* modes.
    pub bf16_wire: bool,
    /// Augmentation multiplicity (`--augment`): each sample appears k
    /// times per epoch as distinct schedule picks. 1 = off.
    pub augment: usize,
    /// Per-view additive input-noise amplitude (`--augment-noise`),
    /// installed as a PickKey-keyed delivery transform. 0.0 = off.
    pub augment_noise: f64,
    /// Slices per data pass (`--epoch-splits`). `1` = an epoch is a full
    /// pass, the historical meaning. Above 1, `epochs * epoch_splits`
    /// events run and each sample is still seen exactly `epochs` times.
    pub epoch_splits: usize,
    /// Tokens of the training shard to stage (`--train-tokens`), token
    /// models only. `None` = the model's default corpus. The staged size
    /// is snapped so the pass divides into whole batched events; see
    /// `models::olmo::resolve_train_corpus`.
    pub train_tokens: Option<u64>,
    /// VRAM share for each rank's data plane (`--vram-max-usage`).
    /// `None` preserves the library default (0.90).
    pub vram_max_usage: Option<f64>,
    /// Host-RAM share for each rank's staging tiers
    /// (`--ram-max-usage`). `None` preserves the library default
    /// (0.50).
    pub ram_max_usage: Option<f64>,
    /// Pinned RAM sample retention in each rank's staging tier
    /// (`--sample-cache`): `false` pins the retained cache at zero and
    /// hands the whole staging share to the flow window. `None` =
    /// library default (enabled).
    pub sample_cache: Option<bool>,
    /// Local-disk overflow tier under each rank's sample cache in GB
    /// (`--disk-stage`). `None` = library default (off).
    pub disk_stage_gb: Option<u64>,
    /// Sub-epoch monitor reports per epoch (`--reports-per-epoch`).
    /// `None` = library default (off, per-epoch reporting only).
    pub reports_per_epoch: Option<usize>,
    /// Directory for the persisted monitor record stream (`--record-log`).
    /// `None` = library default (live-only, nothing written).
    pub record_log_dir: Option<String>,
    /// Save the self-contained dashboard archive to `<run_dir>/dashboard.html`
    /// (`--save-dashboard`). The location is fixed beside the run's other
    /// artifacts rather than configurable, so it cannot inherit the
    /// `--record-log` path ambiguity across launcher / container / worker.
    pub save_dashboard: bool,
    /// Theme pinned into the saved dashboard (`--dashboard-theme`). `None`
    /// leaves it to the reader's `prefers-color-scheme`.
    pub dashboard_theme: Option<String>,
    /// Execution tier (`--tier managed|cooperative`). `Managed` (default)
    /// runs `builder.run()`; `Cooperative` runs `builder.into_worker()` and
    /// hand-drives the loop. Both share the identical builder config, so a
    /// cooperative run is the managed run's parity twin. `DdpMode::Solo`
    /// ignores this (there is no builder path).
    pub tier: Tier,
}

// ---------------------------------------------------------------------------
// Token-count parsing
// ---------------------------------------------------------------------------

/// Parse a token count with an optional `K`/`M`/`G` suffix (`20M`).
///
/// Decimal multipliers, not binary: a token budget is a quantity of
/// training signal, and the field quotes those in powers of ten ("a 300B
/// token run"). Bytes are the thing that wants KiB; tokens are not.
pub fn parse_token_count(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if t.is_empty() {
        return Err("empty token count".to_string());
    }
    let (digits, mult) = match t.chars().last().unwrap().to_ascii_uppercase() {
        'K' => (&t[..t.len() - 1], 1_000u64),
        'M' => (&t[..t.len() - 1], 1_000_000),
        'G' => (&t[..t.len() - 1], 1_000_000_000),
        _ => (t, 1),
    };
    let n: u64 = digits
        .trim()
        .parse()
        .map_err(|_| format!("invalid token count '{s}' (expected e.g. 2097152, 20M, 1G)"))?;
    n.checked_mul(mult)
        .filter(|&v| v > 0)
        .ok_or_else(|| format!("token count '{s}' is zero or overflows"))
}

#[cfg(test)]
mod token_count_tests {
    use super::parse_token_count;

    #[test]
    fn plain_and_suffixed_counts() {
        assert_eq!(parse_token_count("2097152"), Ok(2_097_152));
        assert_eq!(parse_token_count("20M"), Ok(20_000_000));
        assert_eq!(parse_token_count("500K"), Ok(500_000));
        assert_eq!(parse_token_count("1G"), Ok(1_000_000_000));
    }

    #[test]
    fn suffix_is_case_insensitive_and_space_tolerant() {
        assert_eq!(parse_token_count("20m"), Ok(20_000_000));
        assert_eq!(parse_token_count("  20M "), Ok(20_000_000));
    }

    #[test]
    fn junk_is_refused_rather_than_coerced() {
        // A silently-coerced corpus size would change the run's whole
        // data pass without saying so.
        assert!(parse_token_count("").is_err());
        assert!(parse_token_count("20MB").is_err());
        assert!(parse_token_count("-5").is_err());
        assert!(parse_token_count("0").is_err());
        assert!(parse_token_count("1.5M").is_err());
    }
}
