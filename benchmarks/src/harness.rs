//! Benchmark harness: timing, warmup, VRAM sampling, JSON output.

use std::time::Instant;

use serde::Serialize;

/// Result of a single benchmark run.
#[derive(Debug, Serialize)]
pub struct BenchResult {
    pub name: String,
    pub device: String,
    pub runs: usize,
    pub warmup_epochs: usize,
    pub measured_epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub param_count: usize,
    pub epoch_times_ms: Vec<f64>,
    pub median_epoch_ms: f64,
    pub mean_epoch_ms: f64,
    pub min_epoch_ms: f64,
    pub max_epoch_ms: f64,
    /// Per-run medians (for stddev reporting in compare.py).
    pub run_medians_ms: Vec<f64>,
    pub final_loss: f64,
    /// Peak active tensor bytes (matches torch.cuda.max_memory_allocated semantics).
    pub vram_mb: Option<f64>,
    /// Peak allocator reservation (matches torch.cuda.max_memory_reserved semantics).
    pub vram_reserved_mb: Option<f64>,
    pub rss_mb: f64,
}

/// Configuration for running a benchmark.
pub struct BenchConfig {
    pub name: String,
    /// Number of full runs (each with warmup + measured epochs).
    /// The first run is a warmup run (discarded). Default: 4 (1 warmup + 3 measured).
    pub runs: usize,
    pub warmup_epochs: usize,
    pub measured_epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            runs: 4,
            warmup_epochs: 3,
            measured_epochs: 20,
            batches_per_epoch: 100,
            batch_size: 128,
        }
    }
}

/// Run a single measurement pass (warmup + measured epochs).
/// Returns (epoch_times_ms, final_loss).
fn run_single_pass(
    config: &BenchConfig,
    run_epoch: &mut dyn FnMut(usize, bool) -> flodl::Result<f64>,
) -> flodl::Result<(Vec<f64>, f64)> {
    // Warmup
    for i in 0..config.warmup_epochs {
        run_epoch(i, true)?;
    }

    // Sync before measurement
    #[cfg(feature = "gpu")]
    flodl::gpu_synchronize(0);

    // Measured epochs
    let mut epoch_times = Vec::with_capacity(config.measured_epochs);
    let mut final_loss = 0.0;

    for i in 0..config.measured_epochs {
        let start = Instant::now();
        final_loss = run_epoch(i, false)?;

        #[cfg(feature = "gpu")]
        flodl::gpu_synchronize(0);

        epoch_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    Ok((epoch_times, final_loss))
}

/// True median: interpolates for even-length arrays (matches Python statistics.median).
fn median_of(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Run a benchmark with the given configuration.
///
/// Runs `config.runs` passes (first is warmup, rest are measured).
/// Reports the best (lowest median) run to filter process-level noise.
///
/// VRAM is measured using peak stats (reset before benchmark, read after),
/// matching `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()`.
///
/// `run_epoch` is called for each epoch and should return the average loss.
/// It receives `(epoch_index, is_warmup)`.
pub fn run_benchmark(
    config: &BenchConfig,
    param_count: usize,
    mut run_epoch: impl FnMut(usize, bool) -> flodl::Result<f64>,
) -> flodl::Result<BenchResult> {
    let device_name = if flodl::gpu_available() {
        format!("CUDA ({})", flodl::gpu_device_name().unwrap_or_default())
    } else {
        "CPU".to_string()
    };

    let measured_runs = config.runs.max(2) - 1; // first run is warmup

    // Run 0: warmup run (discarded)
    eprintln!("    run 0/{measured_runs} (warmup)");
    run_single_pass(config, &mut run_epoch)?;

    // Reset VRAM peak tracking AFTER warmup so steady-state is measured,
    // not cuDNN autotuning / JIT workspace spikes.
    // Matches Python's torch.cuda.empty_cache() + reset_peak_memory_stats().
    if flodl::gpu_available() {
        flodl::gpu_empty_cache();
        flodl::gpu_reset_peak_stats();
    }

    // Measured runs
    let mut best_times: Option<Vec<f64>> = None;
    let mut best_median = f64::MAX;
    let mut run_medians = Vec::with_capacity(measured_runs);
    let mut final_loss = 0.0;

    for r in 0..measured_runs {
        eprintln!("    run {}/{measured_runs}", r + 1);
        let (times, loss) = run_single_pass(config, &mut run_epoch)?;
        let med = median_of(&times);
        run_medians.push(med);
        final_loss = loss;

        if med < best_median {
            best_median = med;
            best_times = Some(times);
        }
    }

    let epoch_times = best_times.unwrap();

    // VRAM — peak active tensors (matches torch.cuda.max_memory_allocated)
    let vram_mb = flodl::gpu_peak_active_bytes()
        .ok()
        .map(|bytes| bytes as f64 / (1024.0 * 1024.0));

    // VRAM — peak allocator reservation (matches torch.cuda.max_memory_reserved)
    let vram_reserved_mb = flodl::gpu_peak_reserved_bytes()
        .ok()
        .map(|bytes| bytes as f64 / (1024.0 * 1024.0));

    // RSS
    let rss_mb = flodl::rss_kb() as f64 / 1024.0;

    // Stats from best run
    let mut sorted = epoch_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = median_of(&epoch_times);
    let mean = epoch_times.iter().sum::<f64>() / epoch_times.len() as f64;
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    Ok(BenchResult {
        name: config.name.clone(),
        device: device_name,
        runs: measured_runs,
        warmup_epochs: config.warmup_epochs,
        measured_epochs: config.measured_epochs,
        batches_per_epoch: config.batches_per_epoch,
        batch_size: config.batch_size,
        param_count,
        epoch_times_ms: epoch_times,
        median_epoch_ms: median,
        mean_epoch_ms: mean,
        min_epoch_ms: min,
        max_epoch_ms: max,
        run_medians_ms: run_medians,
        final_loss,
        vram_mb,
        vram_reserved_mb,
        rss_mb,
    })
}

/// Pretty-print a benchmark result to stderr.
pub fn print_result(r: &BenchResult) {
    eprintln!("  {}", r.name);
    eprintln!("    device:     {}", r.device);
    eprintln!("    params:     {}", format_count(r.param_count));
    eprintln!("    batches:    {} x {}", r.batches_per_epoch, r.batch_size);
    eprintln!("    runs:       1 warmup + {} measured", r.runs);
    eprintln!("    epochs:     {} warmup + {} measured (per run)", r.warmup_epochs, r.measured_epochs);
    eprintln!("    median:     {:.1} ms/epoch (best run)", r.median_epoch_ms);
    eprintln!("    mean:       {:.1} ms/epoch", r.mean_epoch_ms);
    eprintln!("    range:      {:.1} - {:.1} ms", r.min_epoch_ms, r.max_epoch_ms);
    eprintln!("    final loss: {:.6}", r.final_loss);
    if let Some(vram) = r.vram_mb {
        eprintln!("    VRAM alloc: {:.0} MB", vram);
    }
    if let Some(vram) = r.vram_reserved_mb {
        eprintln!("    VRAM rsrvd: {:.0} MB", vram);
    }
    eprintln!("    RSS:        {:.0} MB", r.rss_mb);
    eprintln!();
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}
