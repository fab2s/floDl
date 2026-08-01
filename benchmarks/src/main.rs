//! flodl benchmark suite.
//!
//! Tier 1 — Baseline patterns (MLP, ConvNet, GRU): measures raw module
//! and optimizer throughput against PyTorch.
//!
//! Tier 2 — Graph builder patterns (residual tower, gated routing,
//! iterative refinement, feedback loop): measures graph builder overhead
//! and showcases patterns that have no clean PyTorch equivalent.
//!
//! Usage:
//!   cargo run --release [--features cuda] [-- OPTIONS]
//!
//! Options:
//!   --tier1        Run tier 1 only
//!   --tier2        Run tier 2 only
//!   --bench NAME   Run a single benchmark by name
//!   --json         Output results as JSON (to stdout)
//!   --epochs N     Measured epochs per run (default: 20)
//!   --warmup N     Warmup epochs per run (default: 3)
//!   --runs N       Total runs per benchmark: 1 warmup + N-1 measured (default: 4)

mod harness;
mod tier1;
mod tier2;

use flodl::{gpu_available, set_cudnn_benchmark, Device};
use harness::{BenchResult, print_result};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let json = args.iter().any(|a| a == "--json");
    let tier1_only = args.iter().any(|a| a == "--tier1");
    let tier2_only = args.iter().any(|a| a == "--tier2");
    let single = args.iter().position(|a| a == "--bench")
        .and_then(|i| args.get(i + 1).cloned());

    // Device selection
    let device = if gpu_available() {
        set_cudnn_benchmark(true);
        eprintln!("device: CUDA ({})", flodl::gpu_device_name().unwrap_or_default());
        Device::CUDA(0)
    } else {
        eprintln!("device: CPU");
        Device::CPU
    };
    eprintln!();

    let run_all = !tier1_only && !tier2_only && single.is_none();

    type BenchFn = Box<dyn Fn(Device) -> flodl::Result<BenchResult>>;

    let benchmarks: Vec<(&str, BenchFn)> = vec![
        // Tier 1
        ("mlp",               Box::new(tier1::mlp::run)),
        ("convnet",           Box::new(tier1::convnet::run)),
        ("gru_seq",           Box::new(tier1::gru_seq::run)),
        ("transformer",       Box::new(tier1::transformer::run)),
        ("lstm_seq",          Box::new(tier1::lstm_seq::run)),
        ("conv_autoenc",      Box::new(tier1::conv_autoencoder::run)),
        // Tier 2
        ("residual_tower",    Box::new(tier2::residual_tower::run)),
        ("gated_routing",     Box::new(tier2::gated_routing::run)),
        ("iterative_refine",  Box::new(tier2::iterative_refine::run)),
        ("feedback_fixed",   Box::new(tier2::feedback_loop_fixed::run)),
    ];

    let tier1_names = ["mlp", "convnet", "gru_seq", "transformer", "lstm_seq", "conv_autoenc"];

    let mut results = Vec::new();

    for (name, run_fn) in &benchmarks {
        let should_run = match &single {
            Some(s) => s == name,
            None if run_all => true,
            None if tier1_only => tier1_names.contains(name),
            None if tier2_only => !tier1_names.contains(name),
            _ => false,
        };

        if !should_run {
            continue;
        }

        eprintln!("--- {} ---", name);
        match run_fn(device) {
            Ok(result) => {
                if !json {
                    print_result(&result);
                }
                results.push(result);
            }
            Err(e) => {
                eprintln!("  FAILED: {e}");
                eprintln!();
            }
        }
    }

    // Summary table
    if !json && results.len() > 1 {
        print_summary(&results);
    }

    // JSON output to stdout
    if json {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    }
}

fn print_summary(results: &[BenchResult]) {
    eprintln!("=== Summary ===");
    eprintln!();
    eprintln!(
        "  {:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "benchmark", "median", "mean", "params", "alloc", "reserved"
    );
    eprintln!("  {}", "-".repeat(74));
    for r in results {
        let alloc = r.vram_mb
            .map(|v| format!("{:.0} MB", v))
            .unwrap_or_else(|| "—".into());
        let rsrvd = r.vram_reserved_mb
            .map(|v| format!("{:.0} MB", v))
            .unwrap_or_else(|| "—".into());
        eprintln!(
            "  {:<20} {:>8.1}ms {:>8.1}ms {:>10} {:>10} {:>10}",
            r.name, r.median_epoch_ms, r.mean_epoch_ms,
            format_count(r.param_count), alloc, rsrvd,
        );
    }
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
