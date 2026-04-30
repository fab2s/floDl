//! Comparison tables, baseline validation, and post-hoc report generation.

use std::collections::HashMap;
use std::fmt::Write;

use crate::analyze::RunAnalysis;
use crate::harness::RunResult;

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
/// `groups` is by model: Vec<(model, Vec<RunAnalysis>)>.
/// `references` maps model name to published reference data.
/// `gpu_info` is the hardware description from training logs.
/// `all_modes` is every known DDP mode name (for missing-run detection).
pub fn generate_report(
    groups: &[(String, Vec<RunAnalysis>)],
    references: &HashMap<String, ModelRef>,
    gpu_info: &[String],
    all_modes: &[String],
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
surpasses solo convergence while completing faster -- the whole point of multi-GPU training.\n\n");

    // Missing runs
    write_missing_runs(&mut md, groups, all_modes);

    // Per-model comparison
    md.push_str("## Per-Model Results\n\n");
    md.push_str("GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.\n\n");
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
        md.push_str("`share` is fraction of batches consumed by each rank (sums to ~1). \
`tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the \
fast GPU consumes a proportionally larger share to keep pace with the slow ones.\n\n");
        write_per_rank_table(&mut md, groups);
    }

    // VRAM overhead
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| !r.vram_stats.is_empty() && r.vram_stats[0].peak_allocated > 0)) {
        md.push_str("## VRAM Usage\n\n");
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

fn write_speed_ratio(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    let mut entries = Vec::new();
    for (model, runs) in groups {
        let s0 = runs.iter().find(|r| r.mode == "solo-0");
        let s1 = runs.iter().find(|r| r.mode == "solo-1");
        if let (Some(a), Some(b)) = (s0, s1)
            && a.total_ms > 0
        {
            entries.push((model.as_str(), a.total_ms, b.total_ms));
        }
    }
    if entries.is_empty() {
        return;
    }
    let _ = writeln!(md, "- **GPU speed ratio** (solo-1 / solo-0 wall time):");
    for (model, s0, s1) in &entries {
        let _ = writeln!(md, "  - {model}: {:.2}x ({:.0}s vs {:.0}s)",
            *s1 as f64 / *s0 as f64, *s0 as f64 / 1000.0, *s1 as f64 / 1000.0);
    }
}

fn write_model_table(md: &mut String, model: &str, runs: &[RunAnalysis], mref: Option<&ModelRef>) {
    if runs.is_empty() {
        return;
    }
    let has_eval = runs.iter().any(|r| r.final_eval.is_some());
    let pub_eval = mref.and_then(|r| r.published_eval);
    let has_delta = has_eval && pub_eval.is_some();

    let _ = writeln!(md, "### {model}\n");
    if let Some(r) = mref {
        let _ = writeln!(md, "> Published: {}\n", r.note);
    }

    // Header
    let _ = write!(md, "| Mode | Loss |");
    if has_eval { md.push_str(" Eval |"); }
    if has_delta { md.push_str(" vs Ref |"); }
    md.push_str(" Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |\n");

    let _ = write!(md, "|------|------|");
    if has_eval { md.push_str("------|"); }
    if has_delta { md.push_str("--------|"); }
    md.push_str("-----------|-------|--------------|------|------|----------|\n");

    for r in runs {
        let a0 = r.gpu_active_pct.first().copied().unwrap_or(0.0);
        let a1 = r.gpu_active_pct.get(1).copied().unwrap_or(0.0);
        let total_idle_s: f64 = r.idle_by_cause.iter()
            .map(|c| c.total_ms)
            .sum::<f64>() / 1000.0;

        let _ = write!(md, "| {} | {:.6} |", r.mode, r.final_loss);

        if has_eval {
            match r.final_eval {
                Some(v) => { let _ = write!(md, " {:.4} |", v); }
                None => md.push_str(" - |"),
            }
        }

        if has_delta {
            match (r.final_eval, pub_eval) {
                (Some(actual), Some(target)) => {
                    let diff = actual - target;
                    if diff.abs() < 0.00005 {
                        md.push_str(" 0 |");
                    } else {
                        let _ = write!(md, " {:+.4} |", diff);
                    }
                }
                _ => md.push_str(" - |"),
            }
        }

        let _ = writeln!(
            md,
            " {:.1} | {} | {:.1} | {:.0}% | {:.0}% | {:.1} |",
            r.total_ms as f64 / 1000.0,
            r.sync_count,
            r.avg_sync_ms,
            a0,
            a1,
            total_idle_s,
        );
    }
    md.push('\n');
}

/// Loss ratio table: mode_loss / solo-0_loss per model.
fn write_loss_ratio_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // Collect all mode names across all groups
    let mut all_modes: Vec<String> = Vec::new();
    for (_, runs) in groups {
        for r in runs {
            if !all_modes.contains(&r.mode) {
                all_modes.push(r.mode.clone());
            }
        }
    }

    // Header
    md.push_str("| Model |");
    for m in &all_modes {
        if m == "solo-0" { continue; }
        let _ = write!(md, " {} |", m);
    }
    md.push('\n');

    md.push_str("|-------|");
    for m in &all_modes {
        if m == "solo-0" { continue; }
        let _ = write!(md, "{}|", "-".repeat(m.len() + 2));
    }
    md.push('\n');

    for (model, runs) in groups {
        let solo0 = runs.iter().find(|r| r.mode == "solo-0");
        let solo_loss = solo0.map(|r| r.final_loss).unwrap_or(0.0);
        let canon_epochs = solo0.map(|r| r.n_epochs).unwrap_or(0);

        let _ = write!(md, "| {model} |");
        for m in &all_modes {
            if m == "solo-0" { continue; }
            if let Some(r) = runs.iter().find(|r| r.mode == *m) {
                if r.n_epochs != canon_epochs {
                    md.push_str(" - |");
                } else if solo_loss.abs() > 1e-10 {
                    let ratio = r.final_loss / solo_loss;
                    let _ = write!(md, " {:.2}x |", ratio);
                } else if r.final_loss.abs() < 1e-10 {
                    md.push_str(" 1.00x |");
                } else {
                    md.push_str(" >100x |");
                }
            } else {
                md.push_str(" - |");
            }
        }
        md.push('\n');
    }
    md.push('\n');
}

/// Missing runs: model/mode combos not present in the data.
fn write_missing_runs(md: &mut String, groups: &[(String, Vec<RunAnalysis>)], all_modes: &[String]) {
    let mut missing: Vec<String> = Vec::new();
    for (model, runs) in groups {
        let canon_epochs = runs.iter()
            .find(|r| r.mode == "solo-0")
            .map(|r| r.n_epochs)
            .unwrap_or(0);

        for mode in all_modes {
            if let Some(r) = runs.iter().find(|r| r.mode == *mode) {
                if r.n_epochs != canon_epochs {
                    missing.push(format!(
                        "{model}/{mode} ({} epochs, expected {canon_epochs})", r.n_epochs,
                    ));
                }
            } else {
                missing.push(format!("{model}/{mode}"));
            }
        }
    }

    if !missing.is_empty() {
        md.push_str("## Incomplete Runs\n\n");
        for m in &missing {
            let _ = writeln!(md, "- {m}");
        }
        md.push('\n');
    }
}

/// Best mode per model: which mode achieves the best eval, and which is fastest
/// while staying within 2% of solo-0 eval.
fn write_best_mode(md: &mut String, groups: &[(String, Vec<RunAnalysis>)], references: &HashMap<String, ModelRef>) {
    let has_eval = groups.iter().any(|(_, runs)| runs.iter().any(|r| r.final_eval.is_some()));

    if has_eval {
        md.push_str("| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |\n");
        md.push_str("|-------|-----------|------|-------------------------------|------|\n");
    } else {
        md.push_str("| Model | Best Loss | Mode | Fastest | Mode |\n");
        md.push_str("|-------|-----------|------|---------|------|\n");
    }

    for (model, runs) in groups {
        let solo0 = runs.iter().find(|r| r.mode == "solo-0");
        let canon_epochs = solo0.map(|r| r.n_epochs).unwrap_or(0);

        // Filter to runs that completed the full epoch count.
        let full_runs: Vec<&RunAnalysis> = runs.iter()
            .filter(|r| r.n_epochs == canon_epochs)
            .collect();

        if full_runs.is_empty() {
            let _ = writeln!(md, "| {model} | - | - | - | - |");
            continue;
        }

        if has_eval {
            let higher_is_better = references.get(model)
                .map(|r| r.higher_is_better)
                .unwrap_or(true);

            let best = full_runs.iter()
                .filter(|r| r.final_eval.is_some())
                .max_by(|a, b| {
                    let va = a.final_eval.unwrap_or(0.0);
                    let vb = b.final_eval.unwrap_or(0.0);
                    if higher_is_better {
                        va.partial_cmp(&vb).unwrap()
                    } else {
                        vb.partial_cmp(&va).unwrap()
                    }
                });

            let (best_eval, best_mode) = match best {
                Some(r) => (format!("{:.4}", r.final_eval.unwrap_or(0.0)), r.mode.as_str()),
                None => ("-".to_string(), "-"),
            };

            // Fastest within 2% of solo-0 eval.
            let solo_eval = solo0.and_then(|r| r.final_eval);
            let fastest = solo_eval.and_then(|se| {
                let threshold = if higher_is_better { se * 0.98 } else { se * 1.02 };
                full_runs.iter()
                    .filter(|r| !r.mode.starts_with("solo"))
                    .filter(|r| {
                        r.final_eval.map(|v| {
                            if higher_is_better { v >= threshold } else { v <= threshold }
                        }).unwrap_or(false)
                    })
                    .min_by_key(|r| r.total_ms)
            });

            let (fast_time, fast_mode) = match fastest {
                Some(r) => (format!("{:.1}s", r.total_ms as f64 / 1000.0), r.mode.as_str()),
                None => ("-".to_string(), "-"),
            };

            let _ = writeln!(md, "| {model} | {best_eval} | {best_mode} | {fast_time} | {fast_mode} |");
        } else {
            // Best loss (lowest)
            let best = full_runs.iter()
                .min_by(|a, b| a.final_loss.partial_cmp(&b.final_loss).unwrap());
            let (best_loss, best_mode) = match best {
                Some(r) => (format!("{:.6}", r.final_loss), r.mode.as_str()),
                None => ("-".to_string(), "-"),
            };

            // Fastest DDP mode
            let fastest = full_runs.iter()
                .filter(|r| !r.mode.starts_with("solo"))
                .min_by_key(|r| r.total_ms);
            let (fast_time, fast_mode) = match fastest {
                Some(r) => (format!("{:.1}s", r.total_ms as f64 / 1000.0), r.mode.as_str()),
                None => ("-".to_string(), "-"),
            };

            let _ = writeln!(md, "| {model} | {best_loss} | {best_mode} | {fast_time} | {fast_mode} |");
        }
    }
    md.push('\n');
}

/// Eval quality table: eval difference vs solo-0 per model/mode.
fn write_eval_ratio_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // Collect all mode names across all groups
    let mut all_modes: Vec<String> = Vec::new();
    for (_, runs) in groups {
        for r in runs {
            if !all_modes.contains(&r.mode) {
                all_modes.push(r.mode.clone());
            }
        }
    }

    // Header
    md.push_str("| Model |");
    for m in &all_modes {
        if m == "solo-0" { continue; }
        let _ = write!(md, " {} |", m);
    }
    md.push('\n');

    md.push_str("|-------|");
    for m in &all_modes {
        if m == "solo-0" { continue; }
        let _ = write!(md, "{}|", "-".repeat(m.len() + 2));
    }
    md.push('\n');

    for (model, runs) in groups {
        let solo0 = runs.iter().find(|r| r.mode == "solo-0");
        let solo_eval = solo0.and_then(|r| r.final_eval);
        let canon_epochs = solo0.map(|r| r.n_epochs).unwrap_or(0);

        let _ = write!(md, "| {model} |");
        for m in &all_modes {
            if m == "solo-0" { continue; }
            if let Some(r) = runs.iter().find(|r| r.mode == *m) {
                if r.n_epochs != canon_epochs {
                    md.push_str(" - |");
                } else if let (Some(actual), Some(base)) = (r.final_eval, solo_eval) {
                    let diff = actual - base;
                    if diff.abs() < 0.00005 {
                        md.push_str(" 0 |");
                    } else {
                        let _ = write!(md, " {:+.4} |", diff);
                    }
                } else {
                    md.push_str(" - |");
                }
            } else {
                md.push_str(" - |");
            }
        }
        md.push('\n');
    }
    md.push('\n');
}

/// Maximum epoch columns before switching to sampled display.
const MAX_TRAJECTORY_COLS: usize = 20;

/// Per-epoch loss trajectory for each model/mode.
/// For models with many epochs, samples at regular intervals.
fn write_epoch_trajectory(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    for (model, runs) in groups {
        let n_epochs = runs.iter().map(|r| r.epoch_data.len()).max().unwrap_or(0);
        if n_epochs < 2 { continue; }

        // Pick which epoch indices to show.
        let indices: Vec<usize> = if n_epochs <= MAX_TRAJECTORY_COLS {
            (0..n_epochs).collect()
        } else {
            // Sample: always include first, last, and evenly spaced in between.
            let step = (n_epochs - 1) as f64 / (MAX_TRAJECTORY_COLS - 1) as f64;
            (0..MAX_TRAJECTORY_COLS)
                .map(|i| (i as f64 * step).round() as usize)
                .collect()
        };

        let sampled = indices.len() < n_epochs;
        if sampled {
            let _ = writeln!(md, "### {model} (sampled, {n_epochs} epochs)\n");
        } else {
            let _ = writeln!(md, "### {model}\n");
        }

        // Header
        let _ = write!(md, "| Mode |");
        for &ep in &indices {
            let _ = write!(md, " E{ep} |");
        }
        md.push('\n');

        let _ = write!(md, "|------|");
        for _ in &indices {
            md.push_str("------|");
        }
        md.push('\n');

        for r in runs {
            let _ = write!(md, "| {} |", r.mode);
            for &ep in &indices {
                if let Some(ed) = r.epoch_data.get(ep) {
                    let _ = write!(md, " {:.4} |", ed.loss);
                } else {
                    md.push_str(" - |");
                }
            }
            md.push('\n');
        }
        md.push('\n');
    }
}

fn write_speedup_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // Collect mode names from first group
    let modes: Vec<&str> = if let Some((_, runs)) = groups.first() {
        runs.iter().map(|r| r.mode.as_str()).collect()
    } else {
        return;
    };

    md.push_str("| Model |");
    for m in &modes {
        if *m == "solo-0" { continue; }
        let _ = write!(md, " {m} |");
    }
    md.push('\n');

    md.push_str("|-------|");
    for m in &modes {
        if *m == "solo-0" { continue; }
        let _ = write!(md, "{}|", "-".repeat(m.len() + 2));
    }
    md.push('\n');

    for (model, runs) in groups {
        let solo0 = runs.iter().find(|r| r.mode == "solo-0");
        // Solo's `# train_only:` summary excludes per-epoch eval cost that
        // DDP modes don't pay. Use it when present so the speedup ratio is
        // training-time vs training-time, not a mixed wall-time comparison.
        let solo0_ms = solo0
            .and_then(|r| r.train_only_ms.map(|v| v as f64))
            .or_else(|| solo0.map(|r| r.total_ms as f64))
            .unwrap_or(0.0);
        let canon_epochs = solo0.map(|r| r.n_epochs).unwrap_or(0);

        let _ = write!(md, "| {model} |");
        for m in &modes {
            if *m == "solo-0" { continue; }
            if let Some(r) = runs.iter().find(|r| r.mode == *m) {
                if solo0_ms > 0.0 && r.total_ms > 0 && r.n_epochs == canon_epochs {
                    let _ = write!(md, " {:.1}x |", solo0_ms / r.total_ms as f64);
                } else {
                    md.push_str(" - |");
                }
            } else {
                md.push_str(" - |");
            }
        }
        md.push('\n');
    }
    md.push('\n');

    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.mode == "solo-0" && r.train_only_ms.is_some())) {
        md.push_str("\nSpeedup denominator uses solo-0's `# train_only:` time when reported \
(paper-baseline models), so the ratio compares DDP wall time against solo's \
training-only wall time. Solo's per-epoch eval is excluded from this comparison \
because DDP runs only eval once at the end.\n\n");
    }
}

/// VRAM usage table per mode per GPU.
fn write_vram_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |\n");
    md.push_str("|-------|------|---------------|---------------|---------------|---------------|\n");

    for (model, runs) in groups {
        for r in runs {
            let g0 = r.vram_stats.first();
            let g1 = r.vram_stats.get(1);

            let g0_peak = g0.map(|v| v.peak_allocated / (1024 * 1024)).unwrap_or(0);
            let g0_mean = g0.map(|v| v.mean_allocated / (1024 * 1024)).unwrap_or(0);
            let g1_peak = g1.map(|v| v.peak_allocated / (1024 * 1024)).unwrap_or(0);
            let g1_mean = g1.map(|v| v.mean_allocated / (1024 * 1024)).unwrap_or(0);

            if g0_peak == 0 && g1_peak == 0 { continue; }

            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} |",
                model, r.mode, g0_peak, g0_mean, g1_peak, g1_mean,
            );
        }
    }
    md.push('\n');
}

fn write_idle_analysis(md: &mut String, model: &str, runs: &[RunAnalysis]) {
    // Only show runs with idle gaps
    let runs_with_gaps: Vec<&RunAnalysis> = runs.iter()
        .filter(|r| !r.idle_gaps.is_empty())
        .collect();

    if runs_with_gaps.is_empty() {
        return;
    }

    let _ = writeln!(md, "### {model}\n");
    md.push_str("| Mode | GPU | Start (s) | Duration (s) | Cause |\n");
    md.push_str("|------|-----|-----------|-------------|-------|\n");

    for r in &runs_with_gaps {
        // Skip startup gaps, sort by duration descending
        let mut gaps: Vec<&crate::analyze::IdleGap> = r.idle_gaps.iter()
            .filter(|g| !matches!(g.cause, crate::analyze::IdleCause::Startup))
            .collect();
        gaps.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms));

        // Show top 10 longest gaps per run
        for g in gaps.iter().take(10) {
            let _ = writeln!(
                md,
                "| {} | gpu{} | {:.1} | {:.1} | {} |",
                r.mode,
                g.device,
                g.start_ms as f64 / 1000.0,
                g.duration_ms as f64 / 1000.0,
                g.cause,
            );
        }
    }
    md.push('\n');
}

fn write_idle_breakdown(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |\n");
    md.push_str("|-------|------|-----|---------------|------|---------|-------------|------------|\n");

    for (model, runs) in groups {
        for r in runs {
            // Skip solo modes and runs with no idle
            if r.mode.starts_with("solo") {
                continue;
            }
            for c in &r.idle_by_cause {
                if c.total_ms < 500.0 {
                    continue; // skip negligible
                }
                let _ = writeln!(
                    md,
                    "| {} | {} | gpu{} | {:.1}s | {:.1}s | {:.1}s | {:.1}s | {:.1}s |",
                    model,
                    r.mode,
                    c.device,
                    c.epoch_boundary_ms / 1000.0,
                    c.sync_ms / 1000.0,
                    c.cpu_avg_ms / 1000.0,
                    c.unexplained_ms / 1000.0,
                    c.total_ms / 1000.0,
                );
            }
        }
    }
    md.push('\n');
}

fn write_elche_details(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |\n");
    md.push_str("|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|\n");

    for (model, runs) in groups {
        for r in runs {
            // Skip solo modes with no DDP activity
            if r.mode.starts_with("solo") || r.sync_count == 0 {
                continue;
            }

            // Sync interval percentiles
            let interval_str = if r.sync_intervals.len() >= 2 {
                let mut sorted = r.sync_intervals.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = sorted[sorted.len() / 2];
                let p95_idx = (sorted.len() as f64 * 0.95) as usize;
                let p95 = sorted[p95_idx.min(sorted.len() - 1)];
                format!("{:.0}/{:.0}", p50, p95)
            } else {
                "-".to_string()
            };

            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {:.1} | {} | {} | {:.1} |",
                model,
                r.mode,
                r.anchor_changes,
                r.throttle_count,
                r.sync_count,
                r.avg_sync_ms,
                interval_str,
                r.cpu_avg_count,
                r.avg_cpu_avg_ms,
            );
        }
    }
    md.push('\n');
}

/// Streaming epoch overlap table.
fn write_per_rank_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | Rank | Device | Share | Tput (samp/ms) |\n");
    md.push_str("|-------|------|------|--------|-------|----------------|\n");
    for (model, runs) in groups {
        for r in runs {
            if r.per_rank_avg.is_empty() { continue; }
            for snap in &r.per_rank_avg {
                let _ = writeln!(
                    md,
                    "| {} | {} | {} | cuda:{} | {:.4} | {:.1} |",
                    model, r.mode, snap.rank, snap.device, snap.batch_share, snap.throughput,
                );
            }
        }
    }
    md.push('\n');
}

fn write_epoch_overlap(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | Overlap (s) | % of Total |\n");
    md.push_str("|-------|------|------------|------------|\n");

    for (model, runs) in groups {
        for r in runs {
            if r.epoch_overlap_ms <= 0.0 { continue; }
            let pct = if r.total_ms > 0 {
                r.epoch_overlap_ms / r.total_ms as f64 * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                md,
                "| {} | {} | {:.1} | {:.1}% |",
                model,
                r.mode,
                r.epoch_overlap_ms / 1000.0,
                pct,
            );
        }
    }
    md.push('\n');
}
