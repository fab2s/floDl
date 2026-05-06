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

    // MSF passive observation (lambda_hat trajectory + phase candidates).
    // Only emit when at least one run has MSF data; otherwise the section
    // is just empty noise.
    if groups.iter().any(|(_, runs)| runs.iter().any(|r| r.msf.has_data())) {
        md.push_str("## MSF Passive Observation\n\n");
        md.push_str("Per the v2 framing (`docs/design/msf-cadence-control-v2.md`), \
            DDP is a synchronization-of-coupled-chaotic-oscillators problem at \
            **two scales** linked by AllReduce. Each subsection below is tagged \
            by the scale it operates at:\n\n\
            - **Top scale (meta-oscillator)**: the cross-rank-collapsed observable \
            `D_mean(t)`, the OU process the system spirals toward. The model we \
            ship is the centroid that sits on the synchronization manifold; \
            convergence is exclusively a top-scale phenomenon.\n\
            - **Bottom scale (per-GPU)**: per-rank `D_i(τ)` within a cycle, \
            chaotic by construction with positive within-cycle Lyapunov \
            `λ_T(LR)`. Per-replica trajectories don't converge — that's by \
            design.\n\
            - **Cross-scale consistency**: cross-rank Pearson `r` and per-rank \
            vs meta slope agreement. The gate that validates the meta-oscillator \
            framing — when `r < 0.95` for any rank pair, the framing has broken \
            and bottom-scale per-rank treatment is required (e.g. cpu-async \
            backend's pipelined averaging is a special case of this gate \
            firing for backend reasons).\n\n\
            Historical proxy `λ̂ = (1/k) * log(D_t / D_{t-1})` from v1 doc \
            survives only as a coarse phase indicator; the v2 estimators are \
            the by-k OLS slope (within-cycle Lyapunov, bottom-scale) and \
            CUSUM-on-OU-residual (regime detection, top-scale).\n\n\
            Phase candidates flag epochs where `λ_min < -1e-2` AND \
            `D_end / prev_D_end < 1/3` (collapse signature, e.g. LR drop \
            boundary).\n\n");
        write_msf_section(&mut md, groups);
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
            .unwrap_or_else(|| runs.iter().map(|r| r.n_epochs).max().unwrap_or(0));

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
        let canon_epochs = solo0
            .map(|r| r.n_epochs)
            .unwrap_or_else(|| runs.iter().map(|r| r.n_epochs).max().unwrap_or(0));

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

// ---------------------------------------------------------------------------
// MSF section (lambda_hat passive observation)
// ---------------------------------------------------------------------------

/// Render a sparkline using Unicode block characters.
///
/// Returns 8-level sparkline matching `xs.len()` characters. NaN/None positions
/// render as a space. Width-bounded: input is downsampled (mean-binned) when
/// longer than `max_chars` so output fits markdown columns.
fn sparkline(xs: &[Option<f64>], max_chars: usize) -> String {
    const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let n = xs.len();
    if n == 0 {
        return String::new();
    }
    // Downsample by mean-binning if too wide.
    let target = max_chars.min(n).max(1);
    let bin = n.div_ceil(target);
    let mut binned: Vec<Option<f64>> = Vec::with_capacity(target);
    let mut i = 0;
    while i < n {
        let end = (i + bin).min(n);
        let mut sum = 0.0;
        let mut count = 0usize;
        for x in xs.iter().take(end).skip(i) {
            if let Some(v) = x
                && v.is_finite()
            {
                sum += v;
                count += 1;
            }
        }
        binned.push(if count == 0 { None } else { Some(sum / count as f64) });
        i = end;
    }
    // Find min/max across populated bins.
    let finite: Vec<f64> = binned.iter().filter_map(|x| *x).collect();
    if finite.is_empty() {
        return " ".repeat(binned.len());
    }
    let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    binned
        .iter()
        .map(|v| match v {
            None => ' ',
            Some(x) => {
                if span <= 0.0 {
                    BARS[BARS.len() / 2]
                } else {
                    let f = ((x - lo) / span).clamp(0.0, 1.0);
                    let idx = ((f * (BARS.len() - 1) as f64).round() as usize).min(BARS.len() - 1);
                    BARS[idx]
                }
            }
        })
        .collect()
}

fn write_msf_section(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // Per-run summary table: count, sync count, final phase regime.
    md.push_str("### Summary (top scale)\n\n");
    md.push_str("| Model | Mode | Epochs | Div Events | Phase Candidates | Final D | Final λ_ema |\n");
    md.push_str("|-------|------|--------|-----------:|-----------------:|--------:|------------:|\n");
    for (model, runs) in groups {
        for r in runs {
            if !r.msf.has_data() {
                continue;
            }
            let last = r.msf.epochs.last();
            let final_d = last.map(|e| e.d_at_epoch_end).unwrap_or(0.0);
            let final_ema = last
                .and_then(|e| e.lambda_ema_at_epoch_end)
                .map(|v| format!("{v:+.2e}"))
                .unwrap_or_else(|| "—".to_string());
            let _ = writeln!(
                md,
                "| {} | {} | {} | {} | {} | {:.2e} | {} |",
                model,
                r.mode,
                r.msf.epochs.len(),
                r.msf.div_event_count,
                r.msf.phase_candidates.len(),
                final_d,
                final_ema,
            );
        }
    }
    md.push('\n');

    // Phase candidates table: only when there are any.
    let any_candidates = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| !r.msf.phase_candidates.is_empty()));
    if any_candidates {
        md.push_str("### Phase-Transition Candidates\n\n");
        md.push_str(
            "Heuristic: `λ_min < -1e-2` AND `D_end / prev_D_end < 1/3`. Strong negative \
             excursion + sharp collapse fingerprints LR drops or other regime shifts.\n\n",
        );
        md.push_str("| Model | Mode | Epoch | λ_min | D_end | D ratio (vs prev) |\n");
        md.push_str("|-------|------|------:|------:|------:|------------------:|\n");
        for (model, runs) in groups {
            for r in runs {
                for c in &r.msf.phase_candidates {
                    let _ = writeln!(
                        md,
                        "| {} | {} | {} | {:+.2e} | {:.2e} | {:.3} |",
                        model, r.mode, c.epoch, c.lambda_min, c.d_end, c.d_ratio,
                    );
                }
            }
        }
        md.push('\n');
    }

    // Per-rank breakdown: D distribution + win share + per-rank lambda.
    let any_per_rank = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| !r.msf.per_rank.is_empty()));
    if any_per_rank {
        md.push_str("### Per-Rank Breakdown (bottom scale + cross-scale consistency)\n\n");
        md.push_str(
            "Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). \
             `win%` is the fraction of AllReduce events where this rank had the highest D \
             across ranks (uniform = 100/world_size). NCCL backends typically expose \
             per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); \
             CPU averaging hides it (~33% per rank).\n\n",
        );
        md.push_str(
            "| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |\n",
        );
        md.push_str(
            "|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|\n",
        );
        for (model, runs) in groups {
            for r in runs {
                for pr in &r.msf.per_rank {
                    let _ = writeln!(
                        md,
                        "| {} | {} | {} | {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.1} | {:+.2e} | {:.2e} |",
                        model,
                        r.mode,
                        pr.rank,
                        pr.n,
                        pr.d_mean,
                        pr.d_sd,
                        pr.d_min,
                        pr.d_max,
                        pr.win_pct,
                        pr.lambda_mean,
                        pr.lambda_sd,
                    );
                }
            }
        }
        md.push('\n');

        // Cross-rank Pearson correlations of D trajectories.
        let any_corr = groups
            .iter()
            .any(|(_, runs)| runs.iter().any(|r| !r.msf.rank_correlations.is_empty()));
        if any_corr {
            md.push_str("Cross-rank Pearson correlation of D trajectories. Values consistently \
                near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are \
                coupled views of one process, not independent oscillators.\n\n");
            md.push_str("| Model | Mode | Pair | Pearson r |\n");
            md.push_str("|-------|------|------|----------:|\n");
            for (model, runs) in groups {
                for r in runs {
                    for ((i, j), corr) in &r.msf.rank_correlations {
                        let _ = writeln!(
                            md,
                            "| {} | {} | rank{} ↔ rank{} | {:+.4} |",
                            model, r.mode, i, j, corr
                        );
                    }
                }
            }
            md.push('\n');
        }
    }

    // Guard comparison (per-event simulators on recomputed λ̂).
    let any_guard = groups.iter().any(|(_, runs)| {
        runs.iter().any(|r| {
            !r.msf.guard_comparison.current_fires.is_empty()
                || !r.msf.guard_comparison.msf_fires.is_empty()
        })
    });
    if any_guard {
        md.push_str("### Convergence Guard Comparison (per-event simulators) (top scale)\n\n");
        md.push_str(
            "Both guards replayed against the per-AllReduce divergence \
             trajectory (matching the production guard's temporal grain). \
             **Current**: ConvergenceGuard::check_trend — fires when 3 \
             consecutive `d_raw` events rise AND the latest exceeds 0.01 \
             (production default). **MSF**: fires when the recomputed bias-\
             corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-\
             style innovation rule, threshold and sustain are the centre \
             point of the sweep below). Lambda is recomputed in analyze.rs \
             from `(d_raw, k_max)` so old timelines and new ones produce \
             the same numbers. Counts are distinct firing epochs.\n\n",
        );
        md.push_str(
            "| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |\n",
        );
        md.push_str(
            "|-------|------|--------------:|----------:|-----:|-------------:|---------:|\n",
        );
        let fmt_eps = |v: &Vec<usize>| {
            if v.is_empty() {
                "—".to_string()
            } else if v.len() <= 12 {
                v.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")
            } else {
                let head: Vec<String> =
                    v.iter().take(8).map(|e| e.to_string()).collect();
                let tail: Vec<String> =
                    v.iter().rev().take(2).map(|e| e.to_string()).collect();
                format!("{}…{}", head.join(","), tail.into_iter().rev().collect::<Vec<_>>().join(","))
            }
        };
        for (model, runs) in groups {
            for r in runs {
                let g = &r.msf.guard_comparison;
                if g.current_fires.is_empty() && g.msf_fires.is_empty() {
                    continue;
                }
                let _ = writeln!(
                    md,
                    "| {} | {} | {} ({}) | {} ({}) | {} | {} | {} |",
                    model,
                    r.mode,
                    g.current_fires.len(),
                    fmt_eps(&g.current_fires),
                    g.msf_fires.len(),
                    fmt_eps(&g.msf_fires),
                    fmt_eps(&g.both),
                    fmt_eps(&g.current_only),
                    fmt_eps(&g.msf_only),
                );
            }
        }
        md.push('\n');
    }

    // MSF threshold sweep — sensitivity grid.
    let any_sweep = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| !r.msf.msf_threshold_sweep.is_empty()));
    if any_sweep {
        md.push_str("### MSF Guard Threshold Sweep (top scale)\n\n");
        md.push_str(
            "Per-event MSF guard fires across (threshold × sustain) grid. \
             `fires` = total firing events, `epochs` = distinct epochs \
             touched by ≥1 fire. Reading: a useful detector should have \
             monotone fall-off as `threshold` rises (signal-driven, not \
             threshold-driven), and `epochs` should concentrate near the \
             phase boundaries the design doc predicts (LR drops, warmup \
             stabilization).\n\n",
        );
        md.push_str("| Model | Mode | threshold | sustain | fires | epochs |\n");
        md.push_str("|-------|------|----------:|--------:|------:|------:|\n");
        for (model, runs) in groups {
            for r in runs {
                for row in &r.msf.msf_threshold_sweep {
                    let _ = writeln!(
                        md,
                        "| {} | {} | {:.0e} | {} | {} | {} |",
                        model,
                        r.mode,
                        row.threshold,
                        row.sustain,
                        row.fires,
                        row.epochs_covered,
                    );
                }
            }
        }
        md.push('\n');
    }

    // Stratified predictive by LR window — surfaces signal that the
    // run-global correlation dilutes when steady-state events outnumber
    // transient ones.
    let any_strat = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| !r.msf.predictive_by_lr_window.is_empty()));
    if any_strat {
        md.push_str("### Predictive Value by LR Window (top scale)\n\n");
        md.push_str(
            "Per-LR-window Pearson `r(λ_raw_t, ln(D_{t+1}))`. Pairs that \
             straddle a window boundary are excluded so the LR-drop \
             collapse cannot leak in as artefactual signal. Reading: a \
             clean R1 (exponential growth at fixed LR) shows up as \
             *non-zero* `r` within each window, with sign and magnitude \
             that may differ between warmup, post-drop transient, and \
             late-training phases.\n\n",
        );
        md.push_str("| Model | Mode | LR | Epochs | n_pairs | r(λ → ln D_{t+1}) |\n");
        md.push_str("|-------|------|---:|--------|--------:|----------------:|\n");
        for (model, runs) in groups {
            for r in runs {
                for w in &r.msf.predictive_by_lr_window {
                    let r_str = w
                        .r
                        .map(|x| format!("{x:+.3}"))
                        .unwrap_or_else(|| "—".into());
                    let _ = writeln!(
                        md,
                        "| {} | {} | {:.2e} | {}–{} | {} | {} |",
                        model,
                        r.mode,
                        w.lr,
                        w.epoch_start,
                        w.epoch_end,
                        w.n_pairs,
                        r_str,
                    );
                }
            }
        }
        md.push('\n');
    }

    // Predictive value (Phase-1 kill criterion).
    let any_pred = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| r.msf.predictive.is_some()));
    if any_pred {
        md.push_str("### Predictive Value (Phase-1 kill criterion) (top scale)\n\n");
        md.push_str(
            "Pearson correlations testing whether λ̂ carries forward-looking \
             signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: \
             does the rate at event t predict the next D? **`λ_mean_per_ep \
             → eval`**: does the epoch's mean λ̂ correlate with held-out \
             accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at \
             epoch boundary correlate with eval? All Pearson, scale-\
             invariant under the `k_used` ↔ `k_max` rescale, so values are \
             robust to the pipeline correction.\n\n",
        );
        md.push_str("| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |\n");
        md.push_str("|-------|------|------:|---------------:|-----:|---------------:|--------------:|\n");
        for (model, runs) in groups {
            for r in runs {
                let Some(p) = &r.msf.predictive else { continue };
                let fmt = |v: Option<f64>| {
                    v.map(|x| format!("{x:+.3}")).unwrap_or_else(|| "—".into())
                };
                let _ = writeln!(
                    md,
                    "| {} | {} | {} | {} | {} | {} | {} |",
                    model,
                    r.mode,
                    p.n_lambda_to_next_logd,
                    fmt(p.lambda_to_next_logd_r),
                    p.n_lambda_to_eval,
                    fmt(p.lambda_mean_to_eval_r),
                    fmt(p.lambda_ema_to_eval_r),
                );
            }
        }
        md.push('\n');
    }

    // Longitudinal meta-velocity (consensus magnitude motion).
    let any_long = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| r.msf.longitudinal.is_some()));
    if any_long {
        md.push_str("### Longitudinal Meta-Velocity (top scale)\n\n");
        md.push_str(
            "Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. \
             Independent of `D_t` (transversal): tracks LR schedule + gradient \
             size, not inter-rank synchronization. Phase-transition signal \
             complementary to λ̂. Only available on backends that report \
             `post_norm` (CPU averaging always; NCCL after post_norm wiring).\n\n",
        );
        md.push_str("| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |\n");
        md.push_str("|-------|------|--:|-------------|----------:|------:|----:|------:|\n");
        for (model, runs) in groups {
            for r in runs {
                if let Some(lg) = &r.msf.longitudinal {
                    let _ = writeln!(
                        md,
                        "| {} | {} | {} | {:.2e}–{:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
                        model,
                        r.mode,
                        lg.n,
                        lg.post_norm_min,
                        lg.post_norm_max,
                        lg.post_norm_mean,
                        lg.velocity_mean,
                        lg.velocity_sd,
                        lg.velocity_max,
                    );
                }
            }
        }
        md.push('\n');
    }

    // R1 informal: log(D) vs step linear fit per auto-detected LR window.
    // Two bases: D_max (per-event max across ranks; legacy) and D_mean
    // (meta-oscillator amplitude — averages out per-rank step asymmetry).
    let any_fits = groups
        .iter()
        .any(|(_, runs)| runs.iter().any(|r| !r.msf.lr_window_fits.is_empty()));
    if any_fits {
        md.push_str("### R1 informal: log(D) vs step per LR window (top scale)\n\n");
        md.push_str(
            "LR windows auto-detected from `EpochEnd` LR transitions (>5% change \
             starts a new window). Within each window, OLS fit of `ln(D)` vs \
             cumulative step on two bases: `D_max` (per-event max across ranks; \
             legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event \
             mean across ranks; the meta-oscillator amplitude — averages out \
             per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two \
             bases trace one process up to scale, but `D_mean` is the cleaner \
             estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 \
             (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the \
             marginal-stability prediction in noise-dominated equilibria. Rows \
             tagged `(post-transient, skipped N)` are sub-window fits emitted \
             alongside the first LR window when an initialization transient is \
             detected — separating the warmup spike from the stable-LR steady \
             state. The third basis `epoch d_mean` aggregates per-event \
             `ln(D_mean)` to one log-mean per epoch and fits those aggregates \
             — denoises intra-epoch SGD variance, which is the dominant \
             remaining noise source after the cross-rank swap. The fourth \
             basis `by k_used` reframes the x-axis: instead of cumulative \
             step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps \
             since the last AllReduce). Sync is a reset — D ≈ 0 immediately \
             after AllReduce — so the natural drift clock restarts at every \
             coupling event. If pure exponential growth holds *within* a \
             cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope \
             is the within-cycle Lyapunov exponent. If the system is in the \
             OU / spiral-to-consensus regime, D_t saturates toward a \
             setpoint D*(LR) and the by-k slope flattens for large k_used \
             (R² collapses).\n\n",
        );
        md.push_str(
            "| Model | Mode | LR | Epochs | n_events | Step range | \
             Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | \
             n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | \
             k_used range | Slope/k (mean) | R² (k, mean) |\n",
        );
        md.push_str(
            "|-------|------|---:|--------|--:|-----------:|----------:|----:|\
             ----------:|----:|--:|----------:|----:|------:|----------:|----:|\n",
        );
        for (model, runs) in groups {
            for r in runs {
                for f in &r.msf.lr_window_fits {
                    let label = if f.transient_skipped == 0 {
                        format!("{}–{}", f.epoch_start, f.epoch_end)
                    } else {
                        format!(
                            "{}–{} (post-transient, skipped {})",
                            f.epoch_start, f.epoch_end, f.transient_skipped
                        )
                    };
                    let dmean_slope = match f.slope_per_step_dmean {
                        Some(v) => format!("{v:+.3e}"),
                        None => "—".to_string(),
                    };
                    let dmean_r2 = match f.r2_dmean {
                        Some(v) => format!("{v:.3}"),
                        None => "—".to_string(),
                    };
                    let n_ep = match f.n_epoch_points {
                        Some(n) => n.to_string(),
                        None => "—".to_string(),
                    };
                    let ep_dmean_slope = match f.slope_per_step_epoch_dmean {
                        Some(v) => format!("{v:+.3e}"),
                        None => "—".to_string(),
                    };
                    let ep_dmean_r2 = match f.r2_epoch_dmean {
                        Some(v) => format!("{v:.3}"),
                        None => "—".to_string(),
                    };
                    let k_range = match (f.k_used_min, f.k_used_max) {
                        (Some(lo), Some(hi)) => format!("{lo}–{hi}"),
                        _ => "—".to_string(),
                    };
                    let by_k_slope = match f.slope_by_k_used_dmean {
                        Some(v) => format!("{v:+.3e}"),
                        None => "—".to_string(),
                    };
                    let by_k_r2 = match f.r2_by_k_used_dmean {
                        Some(v) => format!("{v:.3}"),
                        None => "—".to_string(),
                    };
                    let _ = writeln!(
                        md,
                        "| {} | {} | {:.2e} | {} | {} | {}–{} | {:+.3e} | {:.3} \
                         | {} | {} | {} | {} | {} | {} | {} | {} |",
                        model,
                        r.mode,
                        f.lr,
                        label,
                        f.n_events,
                        f.step_min,
                        f.step_max,
                        f.slope_per_step,
                        f.r2,
                        dmean_slope,
                        dmean_r2,
                        n_ep,
                        ep_dmean_slope,
                        ep_dmean_r2,
                        k_range,
                        by_k_slope,
                        by_k_r2,
                    );
                }
            }
        }
        md.push('\n');
    }

    // R1' per-rank by-k slopes — bottom-scale consistency check on the
    // meta-oscillator framing. Under cross-rank Pearson r > 0.99 (the
    // empirical anchor), per-rank within-cycle Lyapunov estimates should
    // match the meta-D_mean slope. Per-rank divergence indicates the
    // framing is breaking; warn when meta vs per-rank slopes differ by
    // more than a factor of 2 on any LR window.
    let any_per_rank = groups.iter().any(|(_, runs)| {
        runs.iter().any(|r| {
            r.msf
                .lr_window_fits
                .iter()
                .any(|f| !f.slope_by_k_per_rank.is_empty())
        })
    });
    if any_per_rank {
        md.push_str(
            "### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)\n\n",
        );
        md.push_str(
            "Bottom-scale within-cycle Lyapunov estimate computed independently \
             from each rank's `D_i` trajectory (instead of the cross-rank-collapsed \
             `D_mean`). The meta-oscillator framing predicts these per-rank slopes \
             match the meta `D_mean` by-k slope (already in the R1 table above) within \
             seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are \
             colinear at the bottom scale up to a per-rank scaling factor. Used as \
             a falsifier: if a per-rank slope diverges from the meta slope by more \
             than 2× on any LR window, the meta-oscillator framing has broken for \
             this run and bottom-scale per-rank treatment is required (cpu-async \
             backend's pipelined averaging is a special case of this gate firing \
             for backend reasons rather than dynamics reasons).\n\n",
        );
        md.push_str(
            "| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |\n",
        );
        md.push_str(
            "|-------|------|---:|--------|----------:|---|---|---|---|\n",
        );
        for (model, runs) in groups {
            for r in runs {
                for f in &r.msf.lr_window_fits {
                    if f.slope_by_k_per_rank.is_empty() {
                        continue;
                    }
                    let label = if f.transient_skipped == 0 {
                        format!("{}–{}", f.epoch_start, f.epoch_end)
                    } else {
                        format!(
                            "{}–{} (post-transient, skipped {})",
                            f.epoch_start, f.epoch_end, f.transient_skipped
                        )
                    };
                    let meta_slope = match f.slope_by_k_used_dmean {
                        Some(v) => format!("{v:+.3e}"),
                        None => "—".to_string(),
                    };
                    let per_rank_str = f
                        .slope_by_k_per_rank
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            if s.is_finite() {
                                format!("r{}: {:+.3e}", i, s)
                            } else {
                                format!("r{}: —", i)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let per_rank_r2 = f
                        .r2_by_k_per_rank
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            if v.is_finite() {
                                format!("r{}: {:.3}", i, v)
                            } else {
                                format!("r{}: —", i)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let finite_slopes: Vec<f64> = f
                        .slope_by_k_per_rank
                        .iter()
                        .copied()
                        .filter(|s| s.is_finite() && s.abs() > 1e-12)
                        .collect();
                    let (ratio_str, ok_str) = if finite_slopes.len() >= 2 {
                        let max_abs = finite_slopes.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
                        let min_abs = finite_slopes
                            .iter()
                            .map(|s| s.abs())
                            .fold(f64::INFINITY, f64::min);
                        let ratio = max_abs / min_abs;
                        let ok = ratio <= 2.0;
                        (
                            format!("{:.2}×", ratio),
                            if ok { "✓" } else { "⚠ framing breaking" }.to_string(),
                        )
                    } else {
                        ("—".to_string(), "—".to_string())
                    };
                    let _ = writeln!(
                        md,
                        "| {} | {} | {:.2e} | {} | {} | {} | {} | {} | {} |",
                        model,
                        r.mode,
                        f.lr,
                        label,
                        meta_slope,
                        per_rank_str,
                        per_rank_r2,
                        ratio_str,
                        ok_str,
                    );
                }
            }
        }
        md.push('\n');
    }

    // Per-run sparklines: log10(D_max) and lambda_ema_end across epochs.
    md.push_str("### Trajectories\n\n");
    md.push_str(
        "Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay \
         structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy \
         (zero-crossings = phase transitions; sharp negatives = collapse).\n\n",
    );
    md.push_str("| Model | Mode | log10(D_max) | λ_ema |\n");
    md.push_str("|-------|------|--------------|-------|\n");
    for (model, runs) in groups {
        for r in runs {
            if !r.msf.has_data() {
                continue;
            }
            let log_d: Vec<Option<f64>> = r
                .msf
                .epochs
                .iter()
                .map(|e| {
                    if e.d_max > 0.0 {
                        Some(e.d_max.log10())
                    } else {
                        None
                    }
                })
                .collect();
            let ema: Vec<Option<f64>> = r
                .msf
                .epochs
                .iter()
                .map(|e| e.lambda_ema_at_epoch_end)
                .collect();
            let _ = writeln!(
                md,
                "| {} | {} | `{}` | `{}` |",
                model,
                r.mode,
                sparkline(&log_d, 60),
                sparkline(&ema, 60),
            );
        }
    }
    md.push('\n');

    // Per-epoch detail table when narrowly scoped (one or two runs).
    let total_runs: usize = groups.iter().map(|(_, runs)| runs.iter().filter(|r| r.msf.has_data()).count()).sum();
    if total_runs <= 2 {
        for (model, runs) in groups {
            for r in runs {
                if !r.msf.has_data() {
                    continue;
                }
                let _ = writeln!(md, "### Per-Epoch Detail: {} / {}\n", model, r.mode);
                md.push_str(
                    "| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |\n",
                );
                md.push_str(
                    "|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|\n",
                );
                for e in &r.msf.epochs {
                    let lr = e.lr.map(|v| format!("{v:.2e}")).unwrap_or_else(|| "—".into());
                    let lmin = e.lambda_min.map(|v| format!("{v:+.2e}")).unwrap_or_else(|| "—".into());
                    let lmax = e.lambda_max.map(|v| format!("{v:+.2e}")).unwrap_or_else(|| "—".into());
                    let lmean = e.lambda_mean.map(|v| format!("{v:+.2e}")).unwrap_or_else(|| "—".into());
                    let lema = e.lambda_ema_at_epoch_end.map(|v| format!("{v:+.2e}")).unwrap_or_else(|| "—".into());
                    let _ = writeln!(
                        md,
                        "| {} | {} | {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {} | {} | {} | {} | {} |",
                        e.epoch,
                        lr,
                        e.sync_count,
                        e.d_min,
                        e.d_max,
                        e.d_mean,
                        e.d_at_epoch_end,
                        e.k_at_epoch_end,
                        lmin,
                        lmax,
                        lmean,
                        lema,
                    );
                }
                md.push('\n');
            }
        }
    }
}
