//! Generic report-table writers (per-model, speed ratio, loss ratio, idle, eval, VRAM, etc.).

use std::collections::HashMap;

use std::fmt::Write;

use crate::analyze::{RankResStats, RunAnalysis};
use super::ModelRef;

/// A run's rank-derived stats for remote hosts only — hosts the
/// controller's dense local poller cannot see. Without a controller
/// stamp (timelines predating it) every host-qualified entry counts as
/// remote: truthful columns, possibly duplicating a local device's
/// coverage. Empty-host entries (depositor had no topology) are
/// skipped — they cannot be labeled.
fn remote_rank_stats(r: &RunAnalysis) -> impl Iterator<Item = &RankResStats> {
    r.rank_res.iter().filter(|s| {
        !s.host.is_empty() && Some(s.host.as_str()) != r.controller_host.as_deref()
    })
}

/// Rank stats on the controller's own host — the devices that fold into
/// the dense `GPUd` columns. Empty when there is no controller stamp
/// (then every host-qualified rank is remote instead).
fn local_rank_stats(r: &RunAnalysis) -> impl Iterator<Item = &RankResStats> {
    r.rank_res.iter().filter(move |s| {
        r.controller_host.as_deref().is_some_and(|c| s.host == c)
    })
}

/// The bare-`GPUd` device set for a run: the local poller's devices,
/// unioned with any controller-host rank devices that fold in. Both are
/// host-physical indices in the controller box's NVML domain (same
/// box), so they share a numbering — the union never conflates hosts.
fn local_device_union(r: &RunAnalysis) -> Vec<u8> {
    let mut devices: Vec<u8> = r.gpu_devices.clone();
    for s in local_rank_stats(r) {
        if !devices.contains(&s.device) {
            devices.push(s.device);
        }
    }
    devices.sort_unstable();
    devices
}

/// Merge rank-derived VRAM stats for one (host, device) into
/// `(peak, mean, total)` bytes. Multiple ranks sharing a device SUM
/// (allocator bytes are per-process and co-resident — the peak sum is
/// an upper bound since per-rank peaks need not align in time); one
/// rank per device in practice.
fn merged_rank_vram<'a>(
    stats: impl Iterator<Item = &'a RankResStats>,
) -> Option<(u64, u64, u64)> {
    let mut peak = 0u64;
    let mut mean = 0u64;
    let mut total = 0u64;
    let mut any = false;
    for s in stats {
        if let Some(p) = s.peak_allocated {
            peak += p;
            mean += s.mean_allocated.unwrap_or(0);
            total = total.max(s.vram_total.unwrap_or(0));
            any = true;
        }
    }
    any.then_some((peak, mean, total))
}

pub(super) fn write_speed_ratio(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // One ratio line per solo-N present anywhere (N >= 1), each vs solo-0.
    let mut solo_modes: Vec<String> = Vec::new();
    for (_, runs) in groups {
        for r in runs {
            if r.mode.starts_with("solo-") && r.mode != "solo-0"
                && !solo_modes.contains(&r.mode)
            {
                solo_modes.push(r.mode.clone());
            }
        }
    }
    solo_modes.sort();

    for solo in &solo_modes {
        let mut entries = Vec::new();
        for (model, runs) in groups {
            let s0 = runs.iter().find(|r| r.mode == "solo-0");
            let sn = runs.iter().find(|r| &r.mode == solo);
            if let (Some(a), Some(b)) = (s0, sn)
                && a.total_ms > 0
            {
                entries.push((model.as_str(), a.total_ms, b.total_ms));
            }
        }
        if entries.is_empty() {
            continue;
        }
        let _ = writeln!(md, "- **GPU speed ratio** ({solo} / solo-0 wall time):");
        for (model, s0, sn) in &entries {
            let _ = writeln!(md, "  - {model}: {:.2}x ({:.0}s vs {:.0}s)",
                *sn as f64 / *s0 as f64, *s0 as f64 / 1000.0, *sn as f64 / 1000.0);
        }
    }
}

pub(super) fn write_model_table(md: &mut String, model: &str, runs: &[RunAnalysis], mref: Option<&ModelRef>) {
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

    // Header — one column per sampled device id (union across the group's
    // runs), labeled by the host-physical id the timeline recorded. A run
    // that never sampled a device renders `-` (not sampled), never a fake 0%.
    // Includes controller-host rank devices that fold into these columns.
    let mut devices: Vec<u8> = Vec::new();
    for r in runs {
        for d in local_device_union(r) {
            if !devices.contains(&d) {
                devices.push(d);
            }
        }
    }
    devices.sort_unstable();

    // Remote-host columns from rank-reported samples, keyed by
    // (host, device) — device ids collide across hosts, so a bare
    // device id can never label a remote column.
    let mut remote: Vec<(String, u8)> = Vec::new();
    for r in runs {
        for s in remote_rank_stats(r) {
            let key = (s.host.clone(), s.device);
            if !remote.contains(&key) {
                remote.push(key);
            }
        }
    }
    remote.sort();

    let _ = write!(md, "| Mode | Loss |");
    if has_eval { md.push_str(" Eval |"); }
    if has_delta { md.push_str(" vs Ref |"); }
    md.push_str(" Total (s) | Syncs | Avg Sync (ms) |");
    for d in &devices { let _ = write!(md, " GPU{d} |"); }
    for (h, d) in &remote { let _ = write!(md, " {h}:GPU{d} |"); }
    md.push_str(" Idle (s) |\n");

    let _ = write!(md, "|------|------|");
    if has_eval { md.push_str("------|"); }
    if has_delta { md.push_str("--------|"); }
    md.push_str("-----------|-------|--------------|");
    for _ in &devices { md.push_str("------|"); }
    for _ in &remote { md.push_str("------|"); }
    md.push_str("----------|\n");

    for r in runs {
        // `+ 0.0` normalizes the negative zero that an empty-iterator
        // `sum::<f64>()` yields (common now: a dedicated controller with
        // GPU polling gated off has no idle_by_cause entries) so the
        // cell renders "0.0", not "-0.0".
        let total_idle_s: f64 = r.idle_by_cause.iter()
            .map(|c| c.total_ms)
            .sum::<f64>() / 1000.0 + 0.0;

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

        let _ = write!(
            md,
            " {:.1} | {} | {:.1} |",
            r.total_ms as f64 / 1000.0,
            r.sync_count,
            r.avg_sync_ms,
        );
        for d in &devices {
            match r.gpu_devices.iter().position(|x| x == d) {
                Some(i) => {
                    let pct = r.gpu_active_pct.get(i).copied().unwrap_or(0.0);
                    let _ = write!(md, " {pct:.0}% |");
                }
                // Not in the dense poller, but a controller-host rank
                // folds here: use its sparse mean util (`~` marker).
                None => match local_rank_stats(r)
                    .filter(|s| s.device == *d)
                    .find_map(|s| s.mean_util)
                {
                    Some(u) => { let _ = write!(md, " ~{u:.0}% |"); }
                    None => md.push_str(" - |"),
                },
            }
        }
        // Remote cells: sparse MEAN util (`~` marks reduce-window
        // cadence). NVML's util is already a ~1s rolling mean, so a
        // sparse sample above the 5% active threshold is near-certain
        // on any working GPU — the dense columns' active% indicator
        // degenerates to ~100% here, while the mean of an
        // already-time-averaged signal is the honest sparse duty-cycle
        // estimator. Ranks sharing a device merge sample-count-weighted
        // (1:1 in practice).
        for (h, d) in &remote {
            let mut w = 0usize;
            let mut sum = 0.0;
            for s in remote_rank_stats(r)
                .filter(|s| &s.host == h && s.device == *d)
            {
                if let Some(u) = s.mean_util {
                    sum += u * s.n_samples as f64;
                    w += s.n_samples;
                }
            }
            if w > 0 {
                let _ = write!(md, " ~{:.0}% |", sum / w as f64);
            } else {
                md.push_str(" - |");
            }
        }
        let _ = writeln!(md, " {total_idle_s:.1} |");
    }
    md.push('\n');
}

/// Loss ratio table: mode_loss / solo-0_loss per model.
pub(super) fn write_loss_ratio_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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
pub(super) fn write_missing_runs(md: &mut String, groups: &[(String, Vec<RunAnalysis>)], all_modes: &[String]) {
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
pub(super) fn write_best_mode(md: &mut String, groups: &[(String, Vec<RunAnalysis>)], references: &HashMap<String, ModelRef>) {
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
pub(super) fn write_eval_ratio_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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
pub(super) fn write_epoch_trajectory(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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

pub(super) fn write_speedup_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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
(baseline-eval models), so the ratio compares DDP wall time against solo's \
training-only wall time. Solo's per-epoch eval is excluded from this comparison \
because DDP runs only eval once at the end.\n\n");
    }
}

/// VRAM usage table per mode per GPU.
///
/// Local columns (bare `GPUd`) show the controller poller's allocator
/// reading, which is only meaningful in single-process runs — in
/// cluster / auto-promoted runs the ranks are child processes and the
/// controller's own allocator stays empty. Wherever rank-reported
/// samples exist for a device they win: they carry the rank process's
/// actual allocator bytes. Remote devices get their own
/// `host:GPUd`-keyed columns.
pub(super) fn write_vram_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // Columns keyed by sampled device id (union across every run), same
    // convention as the per-model table: `-` = device not sampled by that
    // run's timeline.
    let mut devices: Vec<u8> = Vec::new();
    let mut remote: Vec<(String, u8)> = Vec::new();
    for (_, runs) in groups {
        for r in runs {
            for d in local_device_union(r) {
                if !devices.contains(&d) {
                    devices.push(d);
                }
            }
            for s in remote_rank_stats(r) {
                let key = (s.host.clone(), s.device);
                if !remote.contains(&key) {
                    remote.push(key);
                }
            }
        }
    }
    devices.sort_unstable();
    remote.sort();

    md.push_str("| Model | Mode |");
    for d in &devices { let _ = write!(md, " GPU{d} Peak (MB) | GPU{d} Mean (MB) |"); }
    for (h, d) in &remote { let _ = write!(md, " {h}:GPU{d} Peak (MB) | {h}:GPU{d} Mean (MB) |"); }
    md.push('\n');
    md.push_str("|-------|------|");
    for _ in 0..(devices.len() + remote.len()) { md.push_str("---------------|---------------|"); }
    md.push('\n');

    for (model, runs) in groups {
        for r in runs {
            let any_nonzero = r.vram_stats.iter().any(|v| v.peak_allocated > 0)
                || r.rank_res.iter().any(|s| s.peak_allocated.is_some());
            if !any_nonzero { continue; }

            let _ = write!(md, "| {} | {} |", model, r.mode);
            for d in &devices {
                // Rank-reported allocator bytes win when present for a
                // controller-host device — the local poller's reading is
                // the controller process's own (empty) allocator in any
                // multi-process run.
                let rank_derived = r.controller_host.as_deref().and_then(|ctrl| {
                    merged_rank_vram(
                        r.rank_res
                            .iter()
                            .filter(|s| s.host == ctrl && s.device == *d),
                    )
                });
                if let Some((peak, mean, _)) = rank_derived {
                    let _ = write!(
                        md,
                        " {} | {} |",
                        peak / (1024 * 1024),
                        mean / (1024 * 1024),
                    );
                    continue;
                }
                match r.vram_stats.iter().find(|v| v.device == *d && v.peak_allocated > 0) {
                    Some(s) => {
                        let _ = write!(
                            md,
                            " {} | {} |",
                            s.peak_allocated / (1024 * 1024),
                            s.mean_allocated / (1024 * 1024),
                        );
                    }
                    None => md.push_str(" - | - |"),
                }
            }
            for (h, d) in &remote {
                let stats = merged_rank_vram(
                    remote_rank_stats(r).filter(|s| &s.host == h && s.device == *d),
                );
                match stats {
                    Some((peak, mean, _)) => {
                        let _ = write!(
                            md,
                            " {} | {} |",
                            peak / (1024 * 1024),
                            mean / (1024 * 1024),
                        );
                    }
                    None => md.push_str(" - | - |"),
                }
            }
            md.push('\n');
        }
    }
    md.push('\n');
}

pub(super) fn write_idle_analysis(md: &mut String, model: &str, runs: &[RunAnalysis]) {
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
        gaps.sort_by_key(|g| std::cmp::Reverse(g.duration_ms));

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

pub(super) fn write_idle_breakdown(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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

pub(super) fn write_per_rank_table(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    md.push_str("| Model | Mode | Rank | Device | Host | Share | Tput (samp/ms) | Util | VRAM Peak (MB) |\n");
    md.push_str("|-------|------|------|--------|------|-------|----------------|------|----------------|\n");
    for (model, runs) in groups {
        for r in runs {
            if r.per_rank_avg.is_empty() { continue; }
            for snap in &r.per_rank_avg {
                // Resource columns join by rank alone: `snap.device` is
                // the rank's RUNTIME device index (CUDA_VISIBLE_DEVICES-
                // remapped) while rank_res carries the host-physical
                // index — the two domains must never be compared.
                let res = r.rank_res.iter().find(|s| s.rank == snap.rank);
                let host = res.map(|s| s.host.as_str()).filter(|h| !h.is_empty());
                let _ = write!(
                    md,
                    "| {} | {} | {} | cuda:{} | {} | {:.4} | {:.1} |",
                    model, r.mode, snap.rank, snap.device,
                    host.unwrap_or("-"),
                    snap.batch_share, snap.throughput,
                );
                match res.and_then(|s| s.mean_util) {
                    Some(u) => { let _ = write!(md, " ~{u:.0}% |"); }
                    None => md.push_str(" - |"),
                }
                match res.and_then(|s| s.peak_allocated) {
                    Some(p) => { let _ = writeln!(md, " {} |", p / (1024 * 1024)); }
                    None => md.push_str(" - |\n"),
                }
            }
        }
    }
    md.push('\n');
}

pub(super) fn write_epoch_overlap(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{empty_analysis, RankResStats};

    fn stat(rank: usize, host: &str, device: u8, util: f64, peak: u64) -> RankResStats {
        RankResStats {
            rank,
            host: host.to_string(),
            device,
            mean_util: Some(util),
            peak_allocated: Some(peak),
            mean_allocated: Some(peak),
            vram_total: Some(16_000_000_000),
            n_samples: 5,
        }
    }

    fn cluster_run(controller: Option<&str>) -> RunAnalysis {
        let mut a = empty_analysis("olmo", "cpu-cadence");
        a.controller_host = controller.map(str::to_string);
        a.rank_res = vec![
            stat(0, "ctrl-host", 0, 70.0, 5_000_000_000),
            stat(1, "remote-host", 0, 80.0, 3_000_000_000),
        ];
        a
    }

    /// With a controller stamp, the controller's own rank folds into
    /// the dense local column (no `ctrl-host:GPU0` column); only the
    /// genuinely remote host gets a `host:GPUd` column.
    #[test]
    fn model_table_folds_controller_host() {
        let runs = vec![cluster_run(Some("ctrl-host"))];
        let mut md = String::new();
        write_model_table(&mut md, "olmo", &runs, None);
        assert!(md.contains("remote-host:GPU0"), "remote column missing:\n{md}");
        assert!(!md.contains("ctrl-host:GPU0"), "controller host must fold:\n{md}");
        // The remote util renders as a sparse mean with the ~ marker.
        assert!(md.contains("~80%"), "remote util missing:\n{md}");
    }

    /// Without a stamp (solo-named rig / pre-stamp file), every
    /// host-qualified rank is remote — truthful, possibly redundant
    /// with a dense column for the same physical GPU.
    #[test]
    fn model_table_no_stamp_all_remote() {
        let runs = vec![cluster_run(None)];
        let mut md = String::new();
        write_model_table(&mut md, "olmo", &runs, None);
        assert!(md.contains("ctrl-host:GPU0"), "unstamped controller stays remote:\n{md}");
        assert!(md.contains("remote-host:GPU0"), "remote column missing:\n{md}");
    }

    /// Empty-host rank entries (depositor had no topology) can't be
    /// labeled, so they never produce a remote column.
    #[test]
    fn model_table_skips_empty_host() {
        let mut a = empty_analysis("olmo", "cpu-cadence");
        a.controller_host = Some("ctrl-host".to_string());
        a.rank_res = vec![stat(1, "", 0, 80.0, 3_000_000_000)];
        let mut md = String::new();
        write_model_table(&mut md, "olmo", &[a], None);
        assert!(!md.contains(":GPU0 |"), "empty-host entry must not render a remote column:\n{md}");
    }

    /// VRAM table: rank-reported allocator bytes win for the controller
    /// device (the local poller sees the controller process's own empty
    /// allocator in a multi-process run), and remote hosts get their own
    /// columns.
    #[test]
    fn vram_table_prefers_rank_allocator() {
        let groups = vec![("olmo".to_string(), vec![cluster_run(Some("ctrl-host"))])];
        let mut md = String::new();
        write_vram_table(&mut md, &groups);
        // Controller device peak comes from the rank sample (5000 MB-ish),
        // not the empty local poller.
        assert!(md.contains(&format!("{}", 5_000_000_000u64 / (1024 * 1024))), "rank-derived local VRAM missing:\n{md}");
        assert!(md.contains("remote-host:GPU0 Peak"), "remote VRAM column missing:\n{md}");
    }
}
