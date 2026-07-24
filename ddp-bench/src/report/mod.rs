//! Comparison tables, baseline validation, and post-hoc report generation.

use std::collections::HashMap;
use std::fmt::Write;

use crate::analyze::RunAnalysis;
use crate::harness::RunResult;

pub mod charts;
mod tables;
mod elche;

use tables::{
    write_speed_ratio, write_model_table, write_loss_ratio_table, write_missing_runs,
    write_best_mode, write_eval_ratio_table, write_epoch_trajectory, write_speedup_table,
    write_vram_table, write_idle_analysis, write_idle_breakdown, write_per_rank_table,
    write_epoch_overlap,
};
use elche::write_elche_details;

/// Published reference data for a model.
pub struct ModelRef {
    /// Human-readable note with links.
    pub note: String,
    /// Published eval target (e.g. 0.9125 for 91.25% accuracy).
    pub published_eval: Option<f64>,
    /// True if higher eval is better (accuracy). False for loss-like metrics.
    pub higher_is_better: bool,
}

// ---------------------------------------------------------------------------
// Baselines
// ---------------------------------------------------------------------------

/// A baseline entry: expected loss for a (model, mode) pair.
#[derive(Debug, Clone)]
pub struct Baseline {
    pub model: String,
    pub mode: String,
    pub loss: f64,
    pub epochs: usize,
    pub batches: usize,
    pub batch_size: usize,
}

/// Load baselines from a JSON file.
///
/// Format: `[{"model":"linear","mode":"solo-0","loss":1.23,"epochs":5,"batches":1000,"batch_size":64}, ...]`
pub fn load_baselines(path: &str) -> Result<Vec<Baseline>, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {path}: {e}"))?;
    parse_baselines(&data)
}

/// Save baselines to a JSON file.
pub fn save_baselines(path: &str, baselines: &[Baseline]) -> Result<(), String> {
    let mut out = String::from("[\n");
    for (i, b) in baselines.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "  {{\"model\":\"{}\",\"mode\":\"{}\",\"loss\":{:.6},\"epochs\":{},\"batches\":{},\"batch_size\":{}}}",
            b.model, b.mode, b.loss, b.epochs, b.batches, b.batch_size,
        ));
    }
    out.push_str("\n]\n");
    std::fs::write(path, &out)
        .map_err(|e| format!("cannot write {path}: {e}"))
}

/// Build baselines from run results. Uses per-result config.
pub fn results_to_baselines(results: &[RunResult]) -> Vec<Baseline> {
    results
        .iter()
        .map(|r| Baseline {
            model: r.model_name.clone(),
            mode: r.mode.clone(),
            loss: r.final_loss,
            epochs: r.epochs,
            batches: r.batches_per_epoch,
            batch_size: r.batch_size,
        })
        .collect()
}

/// Validate run results against baselines. Returns (pass_count, fail_count, messages).
///
/// Matches by model name only (ignoring mode) so any DDP mode can be validated
/// against the solo-0 reference. If multiple baselines exist for a model, uses
/// the first one found.
///
/// A result passes if its final loss is within `tolerance` (relative) of the baseline.
/// Missing baselines are reported but not counted as failures.
pub fn validate_results(
    results: &[RunResult],
    baselines: &[Baseline],
    tolerance: f64,
) -> (usize, usize, Vec<String>) {
    let lookup: HashMap<&str, &Baseline> = baselines
        .iter()
        .map(|b| (b.model.as_str(), b))
        .collect();

    let mut pass = 0;
    let mut fail = 0;
    let mut msgs = Vec::new();

    for r in results {
        if let Some(b) = lookup.get(r.model_name.as_str()) {
            let rel_diff = if b.loss.abs() > 1e-10 {
                (r.final_loss - b.loss).abs() / b.loss.abs()
            } else {
                (r.final_loss - b.loss).abs()
            };

            if rel_diff <= tolerance {
                pass += 1;
                msgs.push(format!(
                    "  PASS  {:<16} {:<20} loss={:.6} (baseline={:.6}, diff={:.1}%)",
                    r.model_name, r.mode, r.final_loss, b.loss, rel_diff * 100.0,
                ));
            } else {
                fail += 1;
                msgs.push(format!(
                    "  FAIL  {:<16} {:<20} loss={:.6} (baseline={:.6}, diff={:.1}%)",
                    r.model_name, r.mode, r.final_loss, b.loss, rel_diff * 100.0,
                ));
            }
        } else {
            msgs.push(format!(
                "  SKIP  {:<16} {:<20} loss={:.6} (no baseline)",
                r.model_name, r.mode, r.final_loss,
            ));
        }
    }

    (pass, fail, msgs)
}

fn parse_baselines(json: &str) -> Result<Vec<Baseline>, String> {
    let val: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| format!("invalid JSON: {e}"))?;
    let arr = val.as_array().ok_or("expected JSON array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let model = item["model"].as_str().ok_or("missing model")?.to_string();
        let mode = item["mode"].as_str().ok_or("missing mode")?.to_string();
        let loss = item["loss"].as_f64().ok_or("missing loss")?;
        let epochs = item["epochs"].as_u64().unwrap_or(0) as usize;
        let batches = item["batches"].as_u64().unwrap_or(0) as usize;
        let batch_size = item["batch_size"].as_u64().unwrap_or(0) as usize;
        out.push(Baseline { model, mode, loss, epochs, batches, batch_size });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Post-hoc report from timeline analysis
// ---------------------------------------------------------------------------

/// Generate a markdown report from analyzed runs.
/// `groups` is by model: `Vec<(model, Vec<RunAnalysis>)>`.
/// `references` maps model name to published reference data.
/// `gpu_info` is the hardware description from training logs.
/// `all_modes` is every known DDP mode name (for missing-run detection).
/// `charts` is the focus model + `(relative_path, caption)` pairs already
/// written by [`charts::write_charts`]; `None` = no chart section.
pub fn generate_report(
    groups: &[(String, Vec<RunAnalysis>)],
    references: &HashMap<String, ModelRef>,
    gpu_info: &[String],
    all_modes: &[String],
    charts: Option<(&str, &[(String, String)])>,
) -> String {
    let mut md = String::with_capacity(16_000);

    md.push_str("# DDP Benchmark Report\n\n");

    // Hardware
    if !gpu_info.is_empty() {
        md.push_str("## Hardware\n\n");
        for g in gpu_info {
            let _ = writeln!(md, "- {g}");
        }
        md.push('\n');
    }

    // Setup
    let n_models = groups.len();
    let n_modes: usize = groups.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
    let _ = writeln!(md, "- **Models**: {n_models}");
    let _ = writeln!(md, "- **Modes**: {n_modes}");

    // Speed ratio from solo runs
    write_speed_ratio(&mut md, groups);
    md.push('\n');

    // Methodology note
    md.push_str("## Notes\n\n");
    md.push_str("DDP modes are expected to show slightly lower eval than solo on small models with few epochs. \
Distributed training converges slower in early epochs due to gradient averaging across devices with \
different data views, and ElChe (cadence/async) modes need calibration time to find the optimal sync \
interval, which further penalizes short runs. On longer training (200 epochs), every DDP mode \
surpasses solo convergence while completing faster -- the whole point of multi-GPU training.\n\n\
**Mode semantics**: every mode runs the same ElChe-scheduled engine with work-weighted \
(sum-and-count) averaging; they differ in reduce cadence and dispatch. `*-sync` = equal data \
split, reduce fires as soon as every alive rank has made at least one step since the last \
reduce (NOT per-batch lockstep DDP -- on heterogeneous rigs the fast GPU runs several steps \
per reduce within its equal share, then idles once it is exhausted; degenerates to classic \
DDP on homogeneous rigs). `*-cadence` = throughput-proportional dispatch, reduce fires once \
every rank completes its planned ElChe window. `cpu-async` = cadence plus bounded overshoot \
and decoupled averaging. `cpu-async` is expected to trail `cpu-cadence` on fixed-epoch wall \
time, most visibly on short runs: async's weight-space divergence runs higher by nature \
(EASGD blending, decoupled application), so the convergence guard grows its reduce window \
more slowly -- more reduces early in the run -- and each reduce triggers mid-flight \
(in-flight drain plus snapshot transport against continued streaming). The surplus \
amortizes as the window reaches the epoch cap on longer runs. Async's edge is jitter \
tolerance when rank speeds fluctuate, not steady-state wall time.\n\n");

    // Charts (focus model) - trajectories the tables can't show without
    // exploding; each image links its SVG next to the report.
    if let Some((model, links)) = charts
        && !links.is_empty()
    {
        md.push_str(&format!("## Charts - {model}\n\n"));
        for (path, caption) in links {
            md.push_str(&format!("![{caption}]({path})\n\n*{caption}*\n\n"));
        }
    }

    // Missing runs
    write_missing_runs(&mut md, groups, all_modes);

    // Per-model comparison
    md.push_str("## Per-Model Results\n\n");
    md.push_str("GPU columns = compute utilization % (not load); `-` = device not \
sampled by that run. Bare `GPUd` columns are the controller host's devices, \
sampled densely by the local poller and labeled by host-physical device id. \
`host:GPUd` columns are remote-host devices: the mean compute utilization over \
rank-reported samples that ride the metrics wire at reduce-window cadence (~one \
sample per window) - the `~` prefix marks that sparseness: direction, not \
precision. Idle = total time with <5% utilization, controller-host devices only \
(the sparse remote cadence cannot support gap detection).\n\n");
    for (model, runs) in groups {
        write_model_table(&mut md, model, runs, references.get(model));
    }

    // Best mode per model
    md.push_str("## Best Mode per Model\n\n");
    write_best_mode(&mut md, groups, references);

    // Convergence quality using eval (vs solo-0)
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.final_eval.is_some())) {
        md.push_str("## Eval Quality (vs solo-0)\n\n");
        write_eval_ratio_table(&mut md, groups);
    }

    // Convergence quality matrix (loss ratio vs solo-0)
    md.push_str("## Convergence Quality (loss ratio vs solo-0)\n\n");
    write_loss_ratio_table(&mut md, groups);

    // Per-epoch loss trajectory
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.epoch_data.len() > 1)) {
        md.push_str("## Per-Epoch Loss Trajectory\n\n");
        write_epoch_trajectory(&mut md, groups);
    }

    // Speedup vs solo-0
    if groups.iter().any(|(_, runs)| runs.len() > 1) {
        md.push_str("## Speedup vs solo-0\n\n");
        write_speedup_table(&mut md, groups);
    }

    // Per-rank schedule (heterogeneous-DDP key insight: fast rank gets
    // proportionally more work via batch_share, throughput in samples/ms
    // shows the raw GPU speed gap that justifies the asymmetry).
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| !r.per_rank_avg.is_empty())) {
        md.push_str("## Per-Rank Schedule\n\n");
        md.push_str("`share` is the balancer's smoothed allocation view (ElChe `batch_counts`, \
sums to ~1); `tput` is samples/ms of the rank's own work (peer-wait excluded). Under \
cadence/async modes dispatch follows the balancer, so `share` IS the delivered work split - \
the fast GPU consumes a proportionally larger share to keep pace with the slow ones. Under \
`*-sync` modes dispatch is an EQUAL split regardless: there `share` shows what ElChe would \
allocate (a capacity shadow), not delivered work - the delivered imbalance surfaces as \
fast-GPU idle instead. `host`/`util`/`vram peak` come from the ranks' own resource \
samples (reduce-window cadence; `~` marks the sparse mean; `device` is the rank's \
runtime index, which CUDA_VISIBLE_DEVICES scoping remaps - it need not match the \
host-physical id in the GPU columns).\n\n");
        write_per_rank_table(&mut md, groups);
    }

    // VRAM overhead. Rank-reported allocator stats count as data: in
    // cluster runs the controller poller's own allocator is empty (the
    // ranks are child processes), so local peaks alone would skip the
    // section exactly when the rank-derived numbers matter most.
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| {
        r.vram_stats.iter().any(|v| v.peak_allocated > 0)
            || r.rank_res.iter().any(|s| s.peak_allocated.is_some())
    })) {
        md.push_str("## VRAM Usage\n\n");
        md.push_str("Allocator bytes are per-process: cluster rows come from the \
ranks' own samples (the controller poller cannot see child-process \
allocations); solo rows from the local poller.\n\n");
        write_vram_table(&mut md, groups);
    }

    // Idle analysis (the main event)
    md.push_str("## GPU Idle Analysis\n\n");
    md.push_str("Idle gaps >= 500ms, classified by nearest event.\n\n");
    for (model, runs) in groups {
        write_idle_analysis(&mut md, model, runs);
    }

    // Idle summary by cause
    md.push_str("## Idle Breakdown by Cause\n\n");
    write_idle_breakdown(&mut md, groups);

    // ElChe details (anchor + throttle + sync intervals)
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.anchor_changes > 0 || r.sync_count > 0)) {
        md.push_str("## ElChe Calibration\n\n");
        write_elche_details(&mut md, groups);
    }

    // Epoch overlap (streaming epochs indicator)
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.epoch_overlap_ms > 0.0)) {
        md.push_str("## Streaming Epoch Overlap\n\n");
        write_epoch_overlap(&mut md, groups);
    }

    md
}

