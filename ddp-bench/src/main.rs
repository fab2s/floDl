//! ddp-bench: DDP validation and benchmark suite for flodl.
//!
//! Tests every common training pattern across all ElChe modes to validate
//! correctness and measure convergence, throughput, and GPU utilization.

mod analyze;
mod config;
mod data;
mod download;
mod harness;
mod models;
mod report;

use flodl_cli::{parse_or_schema, FdlArgs};

use config::{DdpMode, RunConfig};

/// DDP validation and benchmark suite for flodl.
#[derive(FdlArgs, Debug)]
struct Cli {
    /// Run specific model (name or "all").
    #[option]
    model: Option<String>,

    /// Run specific DDP mode (name or "all").
    #[option]
    mode: Option<String>,

    /// Override epoch count.
    #[option]
    epochs: Option<usize>,

    /// Override batches per epoch.
    #[option]
    batches: Option<usize>,

    /// Override batch size.
    #[option]
    batch_size: Option<usize>,

    /// Multiply default LR (auto-scales when epochs > default).
    #[option]
    lr_scale: Option<f64>,

    /// Output directory.
    #[option(default = "runs")]
    output: String,

    /// Dataset cache directory.
    #[option(default = "data")]
    data_dir: std::path::PathBuf,

    /// Training data source: "ram" parses the dataset into memory up
    /// front; "disk" reads per sample from the raw files through
    /// flodl's DataSet layer (CIFAR-10 models: resnet, resnet-graph).
    #[option(default = "ram")]
    data_source: String,

    /// Live dashboard port.
    #[option]
    monitor: Option<u16>,

    /// Check results against baselines.
    #[option]
    validate: bool,

    /// Save results as baseline.
    #[option]
    save_baseline: bool,

    /// Baseline file.
    #[option(default = "baselines/baseline.json")]
    baseline: String,

    /// Validation tolerance, 0.0-1.0.
    #[option(default = "0.15")]
    tolerance: f64,

    /// RNG seed.
    #[option(default = "42")]
    seed: u64,

    /// Analyze saved runs and write a markdown report (skips training).
    /// Bare `--report` uses the default path; pass a path to override.
    /// Use `-` for stdout.
    #[option(default = "runs/report.md")]
    report: Option<String>,

    /// With `--report`: generate SVG charts for the given model into
    /// `<output>/charts/` and embed them in the report (eval trajectory,
    /// fast-rank share, cumulative reduces, wall time, controller-GPU
    /// idle). One focus model keeps the report readable; tables stay the
    /// numeric record for everything else.
    #[option]
    charts: Option<String>,

    /// GPU selection: comma-separated physical indices ("0,1", "1,2") or "all".
    /// Sets CUDA_VISIBLE_DEVICES before libtorch init; selected GPUs are
    /// renumbered 0..N for the rest of the run (so `solo-N` picks among the
    /// survivors, not the original physical indices).
    #[option(default = "all")]
    gpus: Option<String>,

    /// Static per-rank partition ratios, e.g. "0.55,0.225,0.225".
    /// Values must sum to ~1.0 and the count must match the visible GPU count.
    ///
    /// Currently honored in nccl-sync and cpu-sync modes only (the framework's
    /// progressive dispatch path used by Cadence/Async modes does not consult
    /// these). Solo modes ignore this flag.
    #[option]
    partition_ratios: Option<String>,

    /// Enable ElChe anchor relax-up on stable convergence (default: disabled).
    ///
    /// When set, each `Stable` convergence verdict grows the ElChe anchor
    /// toward `max_anchor` to reduce sync frequency. Opt in to measure the
    /// relax-up regime explicitly. Default keeps the anchor under
    /// overhead-based auto-tune alone, matching pre-relax-up behavior.
    ///
    /// Honored in Cadence/Async modes only; ignored by Sync and solo modes.
    #[option]
    elche_relax_up: bool,

    /// Enable the LR-aware meta-controller above ElChe. Watches the LR
    /// trajectory + anchor trend + convergence-guard verdicts each
    /// averaging cycle and dispatches reactive `nudge_anchor_down` calls on
    /// sharp LR drops or sustained divergence patterns. Off by default
    /// until validation sweep.
    ///
    /// Honored in Cadence/Async modes only; ignored by Sync and solo modes.
    #[option]
    meta_controller: bool,

    /// Override ElChe's anchor upper bound (`max_anchor`, library default
    /// 200). When set, passed to `DdpBuilder::max_anchor(N)`. Used by
    /// Sweep C of the MSF cadence-control program to bracket the
    /// Pecora-Carroll synchronization threshold by walking the cap across
    /// multiples of the default (e.g. 200, 300, 400, 800).
    ///
    /// Honored in Cadence/Async modes only; ignored by Sync and solo modes.
    #[option]
    max_anchor: Option<usize>,

    /// Override ElChe's anchor lower bound (`min_anchor`, defaults to the
    /// initial anchor). Forces the overhead auto-tune above its natural
    /// equilibrium. Combined with `--max-anchor N` (same value) plus
    /// `--guard none`, pins the cadence at exactly N batches per cycle —
    /// the fixed-k probe used by Sweep B of the MSF cadence-control
    /// program to walk past the auto-tune's preferred operating point and
    /// locate the synchronization threshold k*(LR). The convergence
    /// guard's `NudgeDown` is the only path that bypasses `min_anchor`;
    /// disabling it via `--guard none` is sufficient for hard pinning.
    ///
    /// Honored in Cadence/Async modes only; ignored by Sync and solo modes.
    #[option]
    min_anchor: Option<usize>,

    /// EASGD elastic averaging weight α (must be in `(0, 1]` when set).
    ///
    /// When set, the cpu-async `load_averaged` path blends
    /// `W_local := (1-α)·W_local + α·W_avg` instead of full overwrite.
    /// Preserves the local progress made during the averaging window
    /// ("ahead-of-sync drift") that current cpu-async discards. Reference:
    /// Zhang, Choromanska, LeCun 2015, "Deep learning with Elastic
    /// Averaging SGD," NeurIPS 2015 (<https://arxiv.org/abs/1412.6651>).
    ///
    /// Honored on cpu-async only; ignored on NCCL paths (which use
    /// in-place AllReduce(Avg) and have no equivalent overwrite step).
    /// `None` (default) preserves current behavior (full overwrite, fast
    /// non-blocking copy_ path).
    #[option]
    easgd_alpha: Option<f64>,

    /// Max batches a rank may run past its planned sync point before being
    /// held (the CpuAsync lookahead bound). `None` (default) lets the
    /// framework auto-tune from a small initial up to its ceiling. Set high
    /// (e.g. `200`) to let the convergence guard, rather than a hard
    /// ceiling, govern how far the fast rank ranges ahead during averaging.
    ///
    /// Honored on cpu-async only; ignored by Sync/Cadence and solo modes.
    #[option]
    max_overshoot: Option<usize>,

    /// Outer optimizer applied to the consensus between reduce and broadcast:
    /// `none` (default, plain weighted averaging), `slowmo` (heavy-ball slow
    /// momentum on the pseudo-gradient), or `diloco` (Nesterov momentum +
    /// disposable inner optimizer — worker resets its inner optimizer each
    /// round; param adoption follows the mode, EASGD-blended on cpu-async).
    /// Honored on both CPU (controller-forged consensus) and NCCL (per-rank
    /// replicated step); pair with `--outer-lr` / `--outer-mu`.
    #[option]
    outer_optimizer: Option<String>,

    /// Outer learning rate for slowmo/diloco. Default 1.0 (slowmo) / 0.7 (diloco).
    #[option]
    outer_lr: Option<f64>,

    /// Outer momentum for slowmo/diloco. Default 0.9.
    #[option]
    outer_mu: Option<f64>,

    /// Consensus allocation-weighting exponent `γ`: rank weighted `nₖ^γ` in
    /// the work-weighted average. `1.0` (default) = plain work-weighting,
    /// `0.0` = unweighted average, `−1.0` = per-step-equal. The diagnostic
    /// for the source of the heterogeneous-cadence regularization effect.
    /// CPU averaging backend only (errors on nccl-* modes when ≠ 1.0).
    #[option]
    gamma: Option<f64>,

    /// Ship the CPU-averaging plane's model traffic as bfloat16: halves
    /// pinned snapshots, relay fold traffic, and wire payloads (averaging
    /// still accumulates in f32; checkpoints and the final weights stay
    /// exact f32). CPU averaging modes only — errors loudly on nccl-*.
    #[option]
    bf16_wire: bool,

    /// Augmentation multiplicity: each sample appears k times per epoch
    /// as distinct schedule picks (flodl `.augment(k)`) — the epoch's
    /// work scales by k and the picks shard/balance exactly like
    /// samples. 1 = off.
    #[option]
    augment: Option<usize>,

    /// With `--augment`: per-view additive input-noise amplitude,
    /// installed as a PickKey-keyed delivery transform (flodl
    /// `.transform`) so the k views carry distinct bytes. 0.0 (default)
    /// = no transform: `--augment` alone measures pure schedule
    /// multiplicity (oversampling).
    #[option]
    augment_noise: Option<f64>,

    /// Fraction of total VRAM each rank's data plane may use (flodl
    /// `.vram_max_usage`, clamped to [0.50, 0.99]). Default 0.90.
    /// A/B lever for the unified budget policy.
    #[option]
    vram_max_usage: Option<f64>,

    /// Fraction of available host RAM each rank's staging tiers may
    /// retain (flodl `.ram_max_usage`, clamped to [0.0, 0.90]);
    /// co-hosted ranks split it by schedule share. Default 0.50; 0.0
    /// disables staging retention. A/B lever for the unified budget
    /// policy.
    #[option]
    ram_max_usage: Option<f64>,

    /// Pinned RAM sample retention in each rank's staging tier
    /// (flodl `.sample_cache`): "false"/"off" pins the retained cache
    /// at zero — the flow window keeps the whole staging share.
    /// Default: library default (enabled). A/B lever for retention
    /// benefit.
    #[option]
    sample_cache: Option<bool>,

    /// Local-disk overflow tier under each rank's sample cache, in GB
    /// (flodl `.disk_stage`): samples the RAM budget declines spill to
    /// an ephemeral per-rank pack file. Default: library default (off).
    #[option]
    disk_stage: Option<u64>,

    /// Sub-epoch monitor reports per epoch (flodl
    /// `.reports_per_epoch`): the controller emits N per-window metric
    /// reports across each epoch, at reduce boundaries — the loss curve
    /// *between* epoch points. Default: library default (off).
    #[option]
    reports_per_epoch: Option<usize>,

    /// Run `eval_fn` at the end of every epoch and emit per-epoch
    /// `eval=X.XXXX` into `training.log`. Required for the MSF
    /// kill-criterion correlation `λ̂ → held-out accuracy`. Default off.
    ///
    /// Adds an eval pass per epoch on rank 0 (Sync: consensus params;
    /// Cadence/Async: rank-local at start of next epoch — near-consensus,
    /// trend-preserving for correlation analyses).
    #[option]
    per_epoch_eval: bool,

    /// Execution tier: "managed" (default) runs the framework-driven
    /// `Trainer::builder().run()`; "cooperative" runs `.into_worker()` and
    /// hand-drives the loop in this harness — the *decomposed* form, written
    /// once and scaling unchanged from one device to N to a cluster while the
    /// controller keeps owning cadence / partition / averaging / eval-rank
    /// election. Same builder config feeds both, so a cooperative run is the
    /// managed run's parity twin (that is what the toggle is for).
    ///
    /// Honored on the DDP `Builder` modes (nccl-*/cpu-*); solo modes ignore
    /// it. `--per-epoch-eval` is managed-only for now.
    #[option(default = "managed")]
    tier: String,

    /// Convergence guard selector. Default: `trend` (production behavior,
    /// 3-rises-above-threshold rule).
    ///
    /// - `none`: passive baseline; ElChe overhead-tune drives cadence.
    /// - `trend`: production guard (TrendGuard).
    /// - `msf`: MSF rate-based guard with soft (suppress) + hard (nudge)
    ///   thresholds on the bias-corrected `λ_ema`.
    #[option]
    guard: Option<String>,

    /// Primary divergence threshold. Trend: 3-rises-above-threshold cut-off
    /// (default: library default — 0.05, raised to 0.3 when EASGD blending
    /// is active, i.e. cpu-async, whose elastic standing spread would
    /// otherwise keep the guard permanently armed). MSF: soft
    /// (`SuppressGrowth`) threshold on `λ_ema` (default 1e-3).
    #[option]
    guard_threshold: Option<f64>,

    /// MSF only: number of consecutive events `λ_ema` must remain above
    /// `--guard-threshold` before `SuppressGrowth` fires. Default 3.
    #[option]
    guard_sustain: Option<usize>,

    /// MSF only: hard (`NudgeDown`) threshold on `λ_ema`. Default 1e-2.
    /// Set to a very large value (or use `--guard-no-nudge`) to disable.
    #[option]
    guard_nudge_threshold: Option<f64>,

    /// MSF only: consecutive events `λ_ema` must remain above
    /// `--guard-nudge-threshold` before `NudgeDown` fires. Default 3.
    #[option]
    guard_nudge_sustain: Option<usize>,

    /// MSF only: anchor reduction factor on `NudgeDown` (0.0-1.0).
    /// Default 0.5 (halve the anchor).
    #[option]
    guard_nudge_factor: Option<f64>,

    /// MSF only: disable the hard (`NudgeDown`) trigger entirely. Soft
    /// (`SuppressGrowth`) trigger remains active.
    #[option]
    guard_no_nudge: bool,

    /// MSF only: EMA smoothing coefficient (0.0-1.0). Default 0.9.
    #[option]
    guard_alpha: Option<f64>,

    /// Which rank fires per-epoch user callbacks (`epoch_fn`,
    /// `checkpoint_fn`, `eval_fn`). Accepts:
    ///
    /// - `rank-N`: explicit rank index (e.g. `rank-0`, `rank-1`).
    ///   Loud-errors at framework validation if `N >= world_size`.
    /// - `fastest`: ElChe picks the rank with the lowest
    ///   smoothed-ms-per-batch at run start, sticky thereafter
    ///   (re-resolved only on rank death). Only supported on the
    ///   process-per-rank cluster path (auto-promote or cluster
    ///   fan-out).
    ///
    /// Default: framework default (`Rank(0)`). Solo modes ignore this.
    #[option]
    epoch_callback_policy: Option<String>,

    /// Per-stage block count for `resnet-graph` (He et al. 2015 CIFAR family,
    /// total depth = 6n+2). Default 3 = ResNet-20.
    ///
    /// Recognized depths with published Table 6 evals:
    ///   n=3 → ResNet-20 (91.25%), n=5 → ResNet-32 (92.49%),
    ///   n=7 → ResNet-44 (92.83%), n=9 → ResNet-56 (93.03%),
    ///   n=18 → ResNet-110 (93.39%).
    /// Other values build a depth-n variant with no published_eval (delta
    /// reporting falls back to absolute eval only).
    ///
    /// Honored only when `--model resnet-graph` (or "all"); ignored by
    /// other models.
    #[option]
    depth_n: Option<usize>,

    /// Cluster checkpoint bundle stem (save side). The consensus forge
    /// writes `<stem>.fdl` (averaged weights) + `<stem>.meta.json`
    /// (trajectory + data-coverage). Pair with `--checkpoint-at-epoch` for
    /// a mid-run snapshot. Progressive (cadence/async) cluster modes only.
    #[option]
    save_path: Option<String>,

    /// Resume a cluster run from a bundle stem. Loads `<stem>.fdl` consensus
    /// weights into each replica AND reconstructs the saved data-coverage so
    /// the coordinator dispatches only the uncovered remainder — no data is
    /// repeated. Progressive (cadence/async) cluster modes only.
    #[option]
    resume_from: Option<String>,

    /// Arm a one-shot mid-run checkpoint at this epoch (the first reduce
    /// where any rank reaches it). Pair with `--save-path`. Progressive
    /// (cadence/async) cluster modes only.
    #[option]
    checkpoint_at_epoch: Option<usize>,

    /// Cluster stop threshold: number of rank losses before the run is
    /// declared unrecoverable and survivors save-and-shutdown. Omit to
    /// tolerate any partial loss (only losing every rank stops the run).
    #[option]
    max_failure: Option<usize>,

    /// Show available models and modes, then exit.
    #[option]
    list: bool,
}

fn main() {
    // Worker-role short-circuit (relay / dial-in agent) BEFORE any GPU
    // enumeration / dataset parsing / the 2-GPU auto-promote path.
    // Binaries that go straight to `Trainer::run` don't need this — the
    // dispatch inside `run()` catches every role. ddp-bench needs it
    // because its harness does GPU/mode/dataset work first, and a
    // worker-role process falling into that gating sees ONE host of a
    // multi-host world and exits without ever joining.
    flodl::distributed::launcher::exit_if_worker_role();
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Parse `--epoch-callback-policy` value into an `EpochCallbackPolicy`.
/// Accepts case-insensitive `rank-N` (any non-negative integer) or
/// `fastest`. Loud-errors with a hint on every other input.
fn parse_epoch_callback_policy(
    spec: &str,
) -> flodl::tensor::Result<flodl::distributed::ddp_run::EpochCallbackPolicy> {
    use flodl::distributed::ddp_run::EpochCallbackPolicy;
    let lower = spec.trim().to_ascii_lowercase();
    if lower == "fastest" {
        return Ok(EpochCallbackPolicy::Fastest);
    }
    if let Some(rest) = lower.strip_prefix("rank-") {
        let n: usize = rest.parse().map_err(|_| {
            flodl::tensor::TensorError::new(&format!(
                "invalid --epoch-callback-policy '{spec}': \
                 expected 'rank-N' (N >= 0) or 'fastest'"
            ))
        })?;
        return Ok(EpochCallbackPolicy::Rank(n));
    }
    Err(flodl::tensor::TensorError::new(&format!(
        "invalid --epoch-callback-policy '{spec}': \
         expected 'rank-N' (e.g. rank-0) or 'fastest'"
    )))
}

/// Parse `--partition-ratios "0.55,0.225,0.225"` into a `Vec<f64>`.
///
/// Sum-to-1 and length-vs-world-size validation lives in the framework
/// (the orchestrator checks before dispatch); we only enforce that the
/// string is non-empty, comma-separated, and parses as floats.
fn parse_partition_ratios(spec: &str) -> flodl::tensor::Result<Vec<f64>> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(flodl::tensor::TensorError::new(
            "invalid --partition-ratios: empty value",
        ));
    }
    let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
    let mut out = Vec::with_capacity(parts.len());
    for s in &parts {
        if s.is_empty() {
            return Err(flodl::tensor::TensorError::new(&format!(
                "invalid --partition-ratios value '{spec}': empty entry in list",
            )));
        }
        let v: f64 = s.parse().map_err(|_| flodl::tensor::TensorError::new(&format!(
            "invalid --partition-ratios entry '{s}' (expected float)",
        )))?;
        if !v.is_finite() || v < 0.0 {
            return Err(flodl::tensor::TensorError::new(&format!(
                "invalid --partition-ratios entry '{s}' (must be non-negative finite)",
            )));
        }
        out.push(v);
    }
    Ok(out)
}

/// Validate the `--guard*` flag bundle and return a
/// [`GuardChoice`](crate::config::GuardChoice) that the harness can
/// materialize into a concrete `ConvergenceGuard`.
///
/// Loud-error policy: every guard-specific flag that doesn't apply to the
/// selected `--guard` exits with a clear message rather than being silently
/// ignored. Default guard is `trend` (production behavior).
fn validate_guard_selection(cli: &Cli) -> flodl::tensor::Result<crate::config::GuardChoice> {
    use crate::config::GuardChoice;
    let kind = cli.guard.as_deref().unwrap_or("trend").trim().to_lowercase();
    let only_msf = |name: &str, present: bool| -> flodl::tensor::Result<()> {
        if present && kind != "msf" {
            return Err(flodl::tensor::TensorError::new(&format!(
                "--{name} is only valid with --guard msf (current: --guard {kind})",
            )));
        }
        Ok(())
    };
    only_msf("guard-sustain", cli.guard_sustain.is_some())?;
    only_msf("guard-nudge-threshold", cli.guard_nudge_threshold.is_some())?;
    only_msf("guard-nudge-sustain", cli.guard_nudge_sustain.is_some())?;
    only_msf("guard-nudge-factor", cli.guard_nudge_factor.is_some())?;
    only_msf("guard-no-nudge", cli.guard_no_nudge)?;
    only_msf("guard-alpha", cli.guard_alpha.is_some())?;
    if kind == "none" && cli.guard_threshold.is_some() {
        return Err(flodl::tensor::TensorError::new(
            "--guard-threshold is not used by --guard none",
        ));
    }
    match kind.as_str() {
        "none" => Ok(GuardChoice::None),
        "trend" => Ok(GuardChoice::Trend {
            threshold: cli.guard_threshold,
        }),
        "msf" => Ok(GuardChoice::Msf {
            suppress_threshold: cli.guard_threshold.unwrap_or(1.0e-3),
            suppress_sustain: cli.guard_sustain.unwrap_or(3),
            nudge_threshold: if cli.guard_no_nudge {
                f64::INFINITY
            } else {
                cli.guard_nudge_threshold.unwrap_or(1.0e-2)
            },
            nudge_sustain: cli.guard_nudge_sustain.unwrap_or(3),
            nudge_factor: cli.guard_nudge_factor.unwrap_or(0.5),
            alpha: cli.guard_alpha.unwrap_or(0.9),
        }),
        other => Err(flodl::tensor::TensorError::new(&format!(
            "unknown --guard '{other}' (expected: none, trend, msf)",
        ))),
    }
}

/// Validate the `--outer-optimizer*` flag bundle and return an
/// [`OuterOptChoice`](crate::config::OuterOptChoice) the harness materializes
/// into a `flodl` `OuterOptimizer` factory.
///
/// Loud-error policy (matching `--guard`): outer-optimizer-specific flags
/// that don't apply to the selected variant exit with a clear message.
/// Default is `none` (plain weighted averaging).
fn validate_outer_optimizer_selection(
    cli: &Cli,
) -> flodl::tensor::Result<crate::config::OuterOptChoice> {
    use crate::config::OuterOptChoice;
    let kind = cli
        .outer_optimizer
        .as_deref()
        .unwrap_or("none")
        .trim()
        .to_lowercase();
    // --outer-lr / --outer-mu apply to any momentum-bearing variant
    // (slowmo + diloco), not none.
    let only_momentum = |name: &str, present: bool| -> flodl::tensor::Result<()> {
        if present && kind == "none" {
            return Err(flodl::tensor::TensorError::new(&format!(
                "--{name} requires --outer-optimizer slowmo|diloco \
                 (current: --outer-optimizer {kind})",
            )));
        }
        Ok(())
    };
    only_momentum("outer-lr", cli.outer_lr.is_some())?;
    only_momentum("outer-mu", cli.outer_mu.is_some())?;
    match kind.as_str() {
        "none" => Ok(OuterOptChoice::None),
        "slowmo" => Ok(OuterOptChoice::SlowMomentum {
            lr: cli.outer_lr.unwrap_or(1.0),
            mu: cli.outer_mu.unwrap_or(0.9),
        }),
        // DiLoCo reference defaults: smaller outer lr (≈0.7), mu ≈ 0.9.
        "diloco" => Ok(OuterOptChoice::Nesterov {
            lr: cli.outer_lr.unwrap_or(0.7),
            mu: cli.outer_mu.unwrap_or(0.9),
        }),
        other => Err(flodl::tensor::TensorError::new(&format!(
            "unknown --outer-optimizer '{other}' (expected: none, slowmo, diloco)",
        ))),
    }
}

/// Resolve `--gpus` to a `CUDA_VISIBLE_DEVICES` value and set it before
/// libtorch sees any device. `"all"` is a no-op (lets the host env or
/// physical hardware decide).
fn apply_gpu_selection(spec: &str) -> flodl::tensor::Result<()> {
    let spec = spec.trim();
    if spec.is_empty() || spec == "all" {
        return Ok(());
    }
    let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
    if parts.iter().any(|s| s.is_empty()) {
        return Err(flodl::tensor::TensorError::new(&format!(
            "invalid --gpus value '{spec}': empty index in list",
        )));
    }
    for s in &parts {
        s.parse::<u32>().map_err(|_| flodl::tensor::TensorError::new(&format!(
            "invalid --gpus index '{s}' (expected non-negative integer)",
        )))?;
    }
    let canonical = parts.join(",");
    eprintln!("ddp-bench: --gpus {spec} -> CUDA_VISIBLE_DEVICES={canonical}");
    // SAFETY: we are still in `main`, no threads spawned, no libtorch
    // touched yet. Setting CUDA_VISIBLE_DEVICES from a single-threaded
    // context before any FFI call into libtorch is safe.
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", &canonical); }
    Ok(())
}

fn run() -> flodl::tensor::Result<()> {
    let cli: Cli = parse_or_schema();

    // GPU selection MUST be applied before any libtorch / CUDA init
    // (cuda_device_count() at line ~260 is the first such call). Once
    // libtorch latches onto a device list, CUDA_VISIBLE_DEVICES is ignored.
    if let Some(spec) = cli.gpus.as_deref() {
        apply_gpu_selection(spec)?;
    }

    // ResNet depth-n MUST be set before model_defs are constructed below;
    // `def()` reads the static to populate description / published_eval /
    // reference for the selected variant.
    if let Some(n) = cli.depth_n {
        if n < 1 {
            return Err(flodl::tensor::TensorError::new(
                "--depth-n must be >= 1 (He et al. CIFAR family, depth = 6n+2)",
            ));
        }
        models::resnet_graph::set_depth_n(n);
    }

    let epoch_callback_policy = match cli.epoch_callback_policy.as_deref() {
        None => None,
        Some(spec) => Some(parse_epoch_callback_policy(spec)?),
    };

    let partition_ratios = match cli.partition_ratios.as_deref() {
        None => None,
        Some(spec) => Some(parse_partition_ratios(spec)?),
    };

    let data_source = match cli.data_source.as_str() {
        "ram" => models::DataSource::Ram,
        "disk" => models::DataSource::Disk,
        other => {
            return Err(flodl::tensor::TensorError::new(&format!(
                "--data-source must be \"ram\" or \"disk\", got \"{other}\""
            )))
        }
    };

    // Execution tier. Loud-reject the combos that don't fit the cooperative
    // path yet, rather than silently ignoring them (loud-errors-over-silent).
    let tier = crate::config::Tier::parse(&cli.tier).ok_or_else(|| {
        flodl::tensor::TensorError::new(&format!(
            "--tier must be \"managed\" or \"cooperative\", got \"{}\"",
            cli.tier
        ))
    })?;
    if tier == crate::config::Tier::Cooperative && cli.per_epoch_eval {
        return Err(flodl::tensor::TensorError::new(
            "--per-epoch-eval is managed-only for now; the cooperative tier \
             surfaces the controller-elected final eval via Worker::poll_eval. \
             Drop --per-epoch-eval or use --tier managed.",
        ));
    }

    // Convergence guard selection + flag-compatibility validation.
    // Loud errors when guard-specific flags don't match the chosen guard
    // (the `--guard <name>` selector is the source of truth).
    let guard_choice = validate_guard_selection(&cli)?;
    let outer_opt_choice = validate_outer_optimizer_selection(&cli)?;

    // Map parsed fields to the variable names the rest of this function
    // already uses. Thin bridge keeps the business logic bit-for-bit
    // identical to before the flodl-cli migration.
    let model_filter = cli.model.clone();
    let mode_filter = cli.mode.clone();
    let epochs = cli.epochs;
    let batches = cli.batches;
    let batch_size = cli.batch_size;
    let output = cli.output.clone();
    let data_dir = cli.data_dir.clone();
    let monitor_port = cli.monitor;
    let validate = cli.validate;
    let save_baseline = cli.save_baseline;
    let baseline_path = cli.baseline.clone();
    let tolerance = cli.tolerance;
    let seed = cli.seed;
    let lr_scale = cli.lr_scale;
    let list = cli.list;

    // --report: None=training mode (no report); Some("-")=report to stdout;
    // Some(path)=report to file. Bare `--report` yields Some("runs/report.md")
    // via the declared default.
    let (do_report, report_file) = match cli.report.as_deref() {
        None => (false, None),
        Some("-") => (true, None),
        Some(path) => (true, Some(path.to_string())),
    };

    if list {
        eprintln!("Models:");
        for name in models::model_names() {
            if let Some(m) = models::find_model(name) {
                eprintln!("  {:<16} {}", m.name, m.description);
            }
        }
        eprintln!("\nModes:");
        for name in DdpMode::all_names() {
            eprintln!("  {name}");
        }
        return Ok(());
    }

    // Report mode: analyze saved timeline data from runs/
    if do_report {
        let runs = analyze::discover_runs(&output);

        // Apply filters
        let filtered: Vec<(String, String)> = runs
            .into_iter()
            .filter(|(m, _)| model_filter.as_ref().is_none_or(|f| f == "all" || f == m))
            .filter(|(_, mode)| mode_filter.as_ref().is_none_or(|f| f == "all" || f == mode))
            .collect();

        if filtered.is_empty() {
            eprintln!("no runs found in {output}/");
            return Ok(());
        }

        eprintln!("analyzing {} runs from {output}/", filtered.len());

        // Load and analyze
        let mut analyses: Vec<analyze::RunAnalysis> = Vec::new();
        let mut gpu_info: Vec<String> = Vec::new();
        // Chart inputs retained for the focus model only (`--charts`):
        // the per-epoch log detail and raw timeline that RunAnalysis
        // deliberately reduces away.
        let charts_model = cli.charts.clone();
        let mut charts_data: Vec<report::charts::ChartRun> = Vec::new();
        for (model, mode) in &filtered {
            let run_dir = std::path::Path::new(&output).join(model).join(mode);
            let log_path = run_dir.join("training.log");
            let tl_path = run_dir.join("timeline.json");

            // Parse training log (required -- source of truth for loss/eval).
            let log = match analyze::parse_training_log(&log_path) {
                Ok(log) => log,
                Err(e) => {
                    eprintln!("  skip {model}/{mode}: {e}");
                    continue;
                }
            };

            // Hardware section = union across every run's header (dedup
            // exact lines). First-log-wins hid every GPU the first run's
            // host couldn't see (solo/cluster logs written on different
            // hosts describe different hardware).
            for g in &log.gpu_info {
                if !gpu_info.contains(g) {
                    gpu_info.push(g.clone());
                }
            }
            // Cohort-format lines (`gpu rN [host:cudaD]: ...`, from
            // `launcher::cohort_inventory`) describe the whole rig with
            // host context; legacy `gpuN:` header lines duplicate them
            // with less information (and per-host device numbering that
            // collides across hosts). Prefer the cohort set when present.
            if gpu_info.iter().any(|g| g.starts_with("gpu r")) {
                gpu_info.retain(|g| g.starts_with("gpu r"));
            }

            // Timeline is optional (provides GPU utilization, idle, sync data).
            let tl = analyze::load_timeline(&tl_path).ok();
            let mut a = if let Some(ref tl) = tl {
                analyze::analyze(model, mode, tl)
            } else {
                analyze::empty_analysis(model, mode)
            };

            // Apply training log data (overrides timeline-derived loss/epochs).
            analyze::apply_training_log(&mut a, &log);

            if charts_model.as_deref() == Some(model.as_str()) {
                charts_data.push(report::charts::ChartRun {
                    mode: mode.clone(),
                    log,
                    timeline: tl,
                });
            }
            analyses.push(a);
        }

        // Group by model
        let mut groups: Vec<(String, Vec<analyze::RunAnalysis>)> = Vec::new();
        for a in analyses {
            if let Some(g) = groups.iter_mut().find(|(m, _)| *m == a.model) {
                g.1.push(a);
            } else {
                let model = a.model.clone();
                groups.push((model, vec![a]));
            }
        }

        // Collect all known mode names for missing-run detection.
        let all_modes: Vec<String> = config::DdpMode::all_names()
            .iter().map(|s| s.to_string()).collect();

        let refs: std::collections::HashMap<String, report::ModelRef> =
            models::model_references().into_iter()
                .map(|(k, note, eval, hib)| (k.to_string(), report::ModelRef {
                    note: note.to_string(),
                    published_eval: eval,
                    higher_is_better: hib,
                }))
                .collect();
        // Charts for the focus model: SVGs land in `<output>/charts/`,
        // the report embeds them by relative path (so they resolve when
        // the report lives in the output dir — its normal home).
        let mut chart_links: Vec<(String, String)> = Vec::new();
        if let Some(ref cm) = charts_model {
            if charts_data.is_empty() {
                eprintln!("--charts {cm}: no runs found for that model; skipping charts");
            } else {
                let model_analyses: Vec<&analyze::RunAnalysis> = groups
                    .iter()
                    .find(|(m, _)| m == cm)
                    .map(|(_, runs)| runs.iter().collect())
                    .unwrap_or_default();
                chart_links = report::charts::write_charts(
                    std::path::Path::new(&output),
                    cm,
                    &charts_data,
                    &model_analyses,
                )
                .map_err(|e| {
                    flodl::tensor::TensorError::new(&format!("chart generation failed: {e}"))
                })?;
                eprintln!("charts: {} SVGs in {output}/charts/", chart_links.len());
            }
        }

        let charts_arg = charts_model
            .as_deref()
            .filter(|_| !chart_links.is_empty())
            .map(|m| (m, chart_links.as_slice()));
        let md = report::generate_report(&groups, &refs, &gpu_info, &all_modes, charts_arg);
        if let Some(ref path) = report_file {
            std::fs::write(path, &md)
                .map_err(|e| flodl::tensor::TensorError::new(&format!("cannot write {path}: {e}")))?;
            eprintln!("report saved to {path}");
        } else {
            print!("{md}");
        }
        return Ok(());
    }

    // Resolve models
    let model_defs: Vec<models::ModelDef> = if let Some(ref name) = model_filter {
        if name == "all" {
            models::all_models()
        } else {
            vec![models::find_model(name).ok_or_else(|| {
                flodl::tensor::TensorError::new(&format!("unknown model: {name}"))
            })?]
        }
    } else {
        models::all_models()
    };

    // Resolve modes
    let modes: Vec<DdpMode> = if let Some(ref name) = mode_filter {
        if name == "all" {
            DdpMode::all_names()
                .iter()
                .filter_map(|n| DdpMode::parse(n))
                .collect()
        } else {
            vec![DdpMode::parse(name)
                .ok_or_else(|| flodl::tensor::TensorError::new(&format!("unknown mode: {name}")))?]
        }
    } else {
        DdpMode::all_names()
            .iter()
            .filter_map(|n| DdpMode::parse(n))
            .collect()
    };

    // CUDA-FREE GPU detection (uses `nvidia-smi`, no libtorch init).
    // Critical for cluster mode: the launcher process must not touch
    // libtorch's CUDA context before the launcher trampoline fans out;
    // touching CUDA in the launcher corrupts the spawned children's
    // context on heterogeneous-GPU rigs. See `flodl::sys::detect_gpus`
    // docs + the "no CUDA before Trainer::run" invariant in
    // `flodl::Trainer::run`.
    let detected_gpus = flodl::sys::detect_gpus();
    let gpu_count = detected_gpus.len();

    eprintln!(
        "ddp-bench: {} models x {} modes, {} GPUs available",
        model_defs.len(),
        modes.len(),
        gpu_count
    );
    for g in &detected_gpus {
        eprintln!(
            "  gpu{}: {} ({}GB, {})",
            g.index,
            g.short_name(),
            g.vram_bytes() / (1024 * 1024 * 1024),
            g.sm_version(),
        );
    }

    let mut all_results: Vec<Vec<harness::RunResult>> = Vec::new();

    for model_def in &model_defs {
        let defaults = &model_def.defaults;
        let mut model_results = Vec::new();

        for mode in &modes {
            // Skip multi-GPU modes when only 1 GPU is visible AND
            // we're not in a cluster (either as a rank child or as the
            // launcher process about to fan out). In PPR mode each
            // rank child sees only one GPU via `CUDA_VISIBLE_DEVICES`
            // scoping; the cluster as a whole has the multi-GPU world.
            // - `FLODL_INTERNAL_CLUSTER_JSON` (slim) = rank child
            // - `FLODL_INTERNAL_FULL_CLUSTER_JSON` (full) = launcher process
            //   (will SSH/fork ranks before training begins)
            // Either one means "don't bail on the local GPU count".
            let in_cluster =
                std::env::var_os("FLODL_INTERNAL_CLUSTER_JSON").is_some()
                    || std::env::var_os("FLODL_INTERNAL_FULL_CLUSTER_JSON").is_some();
            if mode.requires_multi_gpu() && gpu_count < 2 && !in_cluster {
                eprintln!("  skipping {} (requires 2+ GPUs, have {})", mode, gpu_count);
                continue;
            }
            // Skip solo-N when GPU index N is out of range (e.g. solo-2
            // on a 2-GPU rig). Renumbering via --gpus also folds in here.
            if let DdpMode::Solo(idx) = mode
                && *idx >= gpu_count
            {
                eprintln!("  skipping {} (GPU index {} not available, have {})", mode, idx, gpu_count);
                continue;
            }

            let run_epochs = epochs.unwrap_or(defaults.epochs);

            // LR scaling: explicit --lr-scale takes priority.
            // When unset and epochs > default, auto-scale LR down so
            // convergence stretches across the extra epochs.
            // E.g. 10 epochs with default 5 -> lr_scale = 0.5
            let effective_lr = if let Some(scale) = lr_scale {
                defaults.lr * scale
            } else if run_epochs > defaults.epochs {
                defaults.lr * (defaults.epochs as f64 / run_epochs as f64)
            } else {
                defaults.lr
            };

            let run_config = RunConfig {
                epochs: run_epochs,
                batches_per_epoch: batches.unwrap_or(defaults.batches_per_epoch),
                batch_size: batch_size.unwrap_or(defaults.batch_size),
                lr: effective_lr,
                seed,
                output_dir: output.clone(),
                data_dir: data_dir.clone(),
                data_source,
                monitor_port,
                partition_ratios: partition_ratios.clone(),
                elche_relax_up: cli.elche_relax_up,
                meta_controller: cli.meta_controller,
                max_anchor: cli.max_anchor,
                min_anchor: cli.min_anchor,
                easgd_alpha: cli.easgd_alpha,
                max_overshoot: cli.max_overshoot,
                per_epoch_eval: cli.per_epoch_eval,
                guard: guard_choice.clone(),
                epoch_callback_policy,
                save_path: cli.save_path.clone(),
                resume_from: cli.resume_from.clone(),
                checkpoint_at_epoch: cli.checkpoint_at_epoch,
                max_failure: cli.max_failure,
                outer_optimizer: outer_opt_choice.clone(),
                gamma: cli.gamma.unwrap_or(1.0),
                bf16_wire: cli.bf16_wire,
                augment: cli.augment.unwrap_or(1).max(1),
                augment_noise: cli.augment_noise.unwrap_or(0.0),
                vram_max_usage: cli.vram_max_usage,
                ram_max_usage: cli.ram_max_usage,
                sample_cache: cli.sample_cache,
                disk_stage_gb: cli.disk_stage,
                reports_per_epoch: cli.reports_per_epoch,
                tier,
            };

            match harness::run_combo(model_def, mode, &run_config) {
                Ok(result) => model_results.push(result),
                Err(e) => {
                    eprintln!("  FAILED: {} / {}: {}", model_def.name, mode, e);
                }
            }
        }

        all_results.push(model_results);
    }

    // Flatten results for baseline operations
    let flat_results: Vec<&harness::RunResult> = all_results.iter().flat_map(|v| v.iter()).collect();

    // Save baseline
    if save_baseline {
        let baselines = report::results_to_baselines(
            &flat_results.iter().map(|r| (*r).clone()).collect::<Vec<_>>(),
        );
        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(&baseline_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match report::save_baselines(&baseline_path, &baselines) {
            Ok(()) => eprintln!("\nBaseline saved to {baseline_path} ({} entries)", baselines.len()),
            Err(e) => eprintln!("\nFailed to save baseline: {e}"),
        }
    }

    // Validate against baseline
    if validate {
        match report::load_baselines(&baseline_path) {
            Ok(baselines) => {
                let results_owned: Vec<harness::RunResult> =
                    flat_results.iter().map(|r| (*r).clone()).collect();
                let (pass, fail, msgs) = report::validate_results(&results_owned, &baselines, tolerance);
                eprintln!("\n=== Validation ({:.0}% tolerance) ===\n", tolerance * 100.0);
                for msg in &msgs {
                    eprintln!("{msg}");
                }
                eprintln!("\n{pass} passed, {fail} failed");
                if fail > 0 {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("\nCannot validate: {e}");
                eprintln!("Run with --save-baseline first to generate baselines.");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

