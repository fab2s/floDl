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

    /// GPU selection: comma-separated physical indices ("0,1", "1,2") or "all".
    /// Sets CUDA_VISIBLE_DEVICES before libtorch init; selected GPUs are
    /// renumbered 0..N for the rest of the run (so `solo-N` picks among the
    /// survivors, not the original physical indices).
    #[option(default = "all")]
    gpus: Option<String>,

    /// Show available models and modes, then exit.
    #[option]
    list: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
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

            // Capture GPU info from the first log that has it.
            if gpu_info.is_empty() && !log.gpu_info.is_empty() {
                gpu_info.clone_from(&log.gpu_info);
            }

            // Timeline is optional (provides GPU utilization, idle, sync data).
            let mut a = if let Ok(tl) = analyze::load_timeline(&tl_path) {
                analyze::analyze(model, mode, &tl)
            } else {
                analyze::empty_analysis(model, mode)
            };

            // Apply training log data (overrides timeline-derived loss/epochs).
            analyze::apply_training_log(&mut a, &log);

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
        let md = report::generate_report(&groups, &refs, &gpu_info, &all_modes);
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

    // Check GPU availability for multi-GPU modes
    #[cfg(feature = "cuda")]
    let gpu_count = flodl::tensor::cuda_device_count() as usize;
    #[cfg(not(feature = "cuda"))]
    let gpu_count = 0usize;

    eprintln!(
        "ddp-bench: {} models x {} modes, {} GPUs available",
        model_defs.len(),
        modes.len(),
        gpu_count
    );
    #[cfg(feature = "cuda")]
    for dev in flodl::tensor::cuda_devices() {
        eprintln!(
            "  gpu{}: {} ({}GB, sm_{}{})",
            dev.index, dev.name, dev.total_memory / (1024 * 1024 * 1024),
            dev.sm_major, dev.sm_minor,
        );
    }

    let mut all_results: Vec<Vec<harness::RunResult>> = Vec::new();

    for model_def in &model_defs {
        let defaults = &model_def.defaults;
        let mut model_results = Vec::new();

        for mode in &modes {
            // Skip multi-GPU modes if only 1 GPU
            if mode.requires_multi_gpu() && gpu_count < 2 {
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
                monitor_port,
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

