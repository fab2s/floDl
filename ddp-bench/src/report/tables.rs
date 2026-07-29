//! Generic report-table writers (per-model, speed ratio, loss ratio, idle, eval, VRAM, etc.).

use std::collections::HashMap;

use std::fmt::Write;

use crate::analyze::{RankResStats, RunAnalysis};
use super::ModelRef;

/// A run's host-qualified rank stats. Empty-host entries (depositor
/// had no topology) are skipped — they cannot be labeled.
fn hosted_rank_stats(r: &RunAnalysis) -> impl Iterator<Item = &RankResStats> {
    r.rank_res.iter().filter(|s| !s.host.is_empty())
}

/// Physical-device fingerprints from every host-qualified rank sample
/// in `runs`: `(device id, VRAM total) → host`. Fingerprints that
/// collide across hosts (two boxes with the same device index AND the
/// same VRAM size) are removed — identity folding only ever acts on a
/// unique physical match, never a guess.
fn host_fingerprints(runs: &[RunAnalysis]) -> Vec<((u8, u64), String)> {
    let mut prints: Vec<((u8, u64), String)> = Vec::new();
    let mut dead: Vec<(u8, u64)> = Vec::new();
    for r in runs {
        for s in hosted_rank_stats(r) {
            let Some(vt) = s.vram_total else { continue };
            let key = (s.device, vt);
            if dead.contains(&key) {
                continue;
            }
            match prints.iter().position(|(k, _)| *k == key) {
                Some(i) if prints[i].1 != s.host => {
                    prints.remove(i);
                    dead.push(key);
                }
                Some(_) => {}
                None => prints.push((key, s.host.clone())),
            }
        }
    }
    prints
}

/// The host that owns a run's DENSE (local-poller) devices, so they can
/// render under host-qualified columns instead of ambiguous bare
/// `GPUd` labels (a bare column means different hardware depending on
/// which box a row's poller ran on — pascal solos vs the controller).
///
/// Resolution order: the controller stamp when present; otherwise the
/// physical fingerprint `(device id, VRAM total)` of every dense device
/// matched against the group's rank-reported samples — all dense
/// devices must resolve to the SAME single host (one poller, one box),
/// anything else returns `None` and the run keeps bare columns plus the
/// legend.
fn dense_host(r: &RunAnalysis, prints: &[((u8, u64), String)]) -> Option<String> {
    if let Some(h) = &r.controller_host {
        return Some(h.clone());
    }
    let mut host: Option<&str> = None;
    for v in &r.vram_stats {
        let key = (v.device, v.total);
        let h = prints.iter().find(|(k, _)| *k == key).map(|(_, h)| h.as_str())?;
        match host {
            None => host = Some(h),
            Some(prev) if prev != h => return None,
            Some(_) => {}
        }
    }
    // Every dense device carries a vram_stats entry (same sample
    // stream), so an empty vram_stats means no dense devices — nothing
    // to label either way.
    if r.gpu_devices.is_empty() {
        return None;
    }
    host.map(str::to_string)
}

/// `(hosted columns, bare fallback columns, per-run dense hosts)` —
/// the return shape of [`gpu_columns`].
type GpuColumns = (Vec<(String, u8)>, Vec<u8>, Vec<Option<String>>);

/// The unified GPU column set for a group of runs: host-qualified
/// `(host, device)` columns from rank samples AND from dense devices
/// whose host resolved (see [`dense_host`]), plus bare `device`
/// leftovers for dense devices that could not be attributed.
fn gpu_columns(
    runs: &[RunAnalysis],
    prints: &[((u8, u64), String)],
) -> GpuColumns {
    let dense_hosts: Vec<Option<String>> =
        runs.iter().map(|r| dense_host(r, prints)).collect();
    let mut hosted: Vec<(String, u8)> = Vec::new();
    let mut bare: Vec<u8> = Vec::new();
    for (r, dh) in runs.iter().zip(&dense_hosts) {
        for s in hosted_rank_stats(r) {
            let key = (s.host.clone(), s.device);
            if !hosted.contains(&key) {
                hosted.push(key);
            }
        }
        for d in &r.gpu_devices {
            match dh {
                Some(h) => {
                    let key = (h.clone(), *d);
                    if !hosted.contains(&key) {
                        hosted.push(key);
                    }
                }
                None => {
                    if !bare.contains(d) {
                        bare.push(*d);
                    }
                }
            }
        }
    }
    hosted.sort();
    bare.sort_unstable();
    (hosted, bare, dense_hosts)
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

    // Header — one column per PHYSICAL device: host-qualified wherever
    // the device's box is known (rank samples carry it; dense-poller
    // devices resolve via the controller stamp or the group's physical
    // fingerprints — see `dense_host`), bare `GPUd` only as the
    // fallback for dense devices no identity reached. A run that never
    // sampled a device renders `-` (not sampled), never a fake 0%.
    let prints = host_fingerprints(runs);
    let (hosted, bare, dense_hosts) = gpu_columns(runs, &prints);

    let _ = write!(md, "| Mode | Loss |");
    if has_eval { md.push_str(" Eval |"); }
    if has_delta { md.push_str(" vs Ref |"); }
    md.push_str(" Total (s) | Syncs | Avg Sync (ms) |");
    for (h, d) in &hosted { let _ = write!(md, " {h}:GPU{d} |"); }
    for d in &bare { let _ = write!(md, " GPU{d} |"); }
    md.push_str(" Idle (s) |\n");

    let _ = write!(md, "|------|------|");
    if has_eval { md.push_str("------|"); }
    if has_delta { md.push_str("--------|"); }
    md.push_str("-----------|-------|--------------|");
    for _ in &hosted { md.push_str("------|"); }
    for _ in &bare { md.push_str("------|"); }
    md.push_str("----------|\n");

    for (r, dh) in runs.iter().zip(&dense_hosts) {
        // Idle detection reads the DENSE timeline samples: `idle_by_cause`
        // carries one entry per device in `gpu_devices`, which is itself the
        // union of devices the local poller sampled. So an empty
        // `gpu_devices` means the gap detector had no input at all, which is
        // routine now that a dedicated controller gates its GPU poll off
        // (every GPU column on such a run is rank-reported and sparse).
        // Summing nothing yields 0.0, and printing that would state a
        // measured zero where nothing was measured — the same absent-is-not
        // -zero rule the GPU columns already follow with `-`.
        let total_idle_s: Option<f64> = (!r.gpu_devices.is_empty()).then(|| {
            r.idle_by_cause.iter().map(|c| c.total_ms).sum::<f64>() / 1000.0
        });

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
        // Hosted cells: the dense poller's active% wins when this run's
        // dense devices resolved to the column's host (dense sampling,
        // no `~`); otherwise the sparse rank-reported MEAN util (`~`
        // marks reduce-window cadence). NVML's util is already a ~1s
        // rolling mean, so a sparse sample above the 5% active
        // threshold is near-certain on any working GPU — the dense
        // active% indicator degenerates to ~100% at sparse cadence,
        // while the mean of an already-time-averaged signal is the
        // honest sparse duty-cycle estimator. Ranks sharing a device
        // merge sample-count-weighted (1:1 in practice).
        for (h, d) in &hosted {
            if dh.as_deref() == Some(h.as_str())
                && let Some(i) = r.gpu_devices.iter().position(|x| x == d)
            {
                let pct = r.gpu_active_pct.get(i).copied().unwrap_or(0.0);
                let _ = write!(md, " {pct:.0}% |");
                continue;
            }
            let mut w = 0usize;
            let mut sum = 0.0;
            for s in hosted_rank_stats(r)
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
        // Bare fallback columns: dense devices whose box stayed unknown.
        for d in &bare {
            match (dh.is_none(), r.gpu_devices.iter().position(|x| x == d)) {
                (true, Some(i)) => {
                    let pct = r.gpu_active_pct.get(i).copied().unwrap_or(0.0);
                    let _ = write!(md, " {pct:.0}% |");
                }
                _ => md.push_str(" - |"),
            }
        }
        match total_idle_s {
            Some(s) => { let _ = writeln!(md, " {s:.1} |"); }
            None => md.push_str(" - |\n"),
        }
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
                Some(r) => (
                    format!("{:.1}s", r.total_ms as f64 / 1000.0),
                    r.mode.clone(),
                ),
                // Nothing inside the band: name the closest mode and
                // its eval gap instead of a bare dash that reads as
                // missing data (a RELATIVE 2% band is microscopic on a
                // near-zero loss-like eval — conv-ae's reconstruction
                // MSE was the motivating case).
                None => {
                    let closest = solo_eval.and_then(|se| {
                        full_runs.iter()
                            .filter(|r| !r.mode.starts_with("solo"))
                            .filter_map(|r| r.final_eval.map(|v| (r, v)))
                            .min_by(|(_, a), (_, b)| {
                                let da = if higher_is_better { se - a } else { a - se };
                                let db = if higher_is_better { se - b } else { b - se };
                                da.partial_cmp(&db).unwrap()
                            })
                            .map(|(r, v)| {
                                let gap = if se.abs() > 1e-12 {
                                    format!("{:+.1}%", (v - se) / se * 100.0)
                                } else {
                                    format!("{v:+.4} abs")
                                };
                                format!("none ≤2% (closest: {} {gap} eval)", r.mode)
                            })
                    });
                    ("-".to_string(), closest.unwrap_or_else(|| "-".to_string()))
                }
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
    if has_eval {
        md.push_str(
            "\"Fastest (within 2% of solo-0)\" = the fastest NON-solo mode \
whose final eval gives up no more than 2% relative quality vs solo-0 \
(a better-than-solo eval always qualifies): the speed a DDP mode buys \
without paying for it in quality. The band is relative, so it is \
strict to the point of unreachable for near-zero loss-like evals \
(a reconstruction MSE of 0.0006 leaves a ±0.00001 band); when no mode \
qualifies, the closest one and its eval gap are named instead.\n\n",
        );
    }
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
    // Same unified column convention as the per-model tables:
    // host-qualified wherever the device's box is known, bare `GPUd`
    // only for unattributed dense devices; `-` = not sampled.
    let all_runs: Vec<RunAnalysis> = groups
        .iter()
        .flat_map(|(_, runs)| runs.iter().cloned())
        .collect();
    let prints = host_fingerprints(&all_runs);
    let (hosted, bare, dense_hosts) = gpu_columns(&all_runs, &prints);

    md.push_str("| Model | Mode |");
    for (h, d) in &hosted { let _ = write!(md, " {h}:GPU{d} Peak (MB) | {h}:GPU{d} Mean (MB) |"); }
    for d in &bare { let _ = write!(md, " GPU{d} Peak (MB) | GPU{d} Mean (MB) |"); }
    md.push('\n');
    md.push_str("|-------|------|");
    for _ in 0..(hosted.len() + bare.len()) { md.push_str("---------------|---------------|"); }
    md.push('\n');

    let mut run_i = 0usize;
    for (model, runs) in groups {
        for r in runs {
            let dh = &dense_hosts[run_i];
            run_i += 1;
            let any_nonzero = r.vram_stats.iter().any(|v| v.peak_allocated > 0)
                || r.rank_res.iter().any(|s| s.peak_allocated.is_some());
            if !any_nonzero { continue; }

            let _ = write!(md, "| {} | {} |", model, r.mode);
            for (h, d) in &hosted {
                // Rank-reported allocator bytes win whenever a rank on
                // this (host, device) sampled them — the dense poller's
                // reading is its own process's allocator, which is empty
                // in any multi-process run. Dense fills in solo runs
                // (single process: the poller's allocator IS the run's).
                let rank_derived = merged_rank_vram(
                    hosted_rank_stats(r)
                        .filter(|s| &s.host == h && s.device == *d),
                );
                if let Some((peak, mean, _)) = rank_derived {
                    let _ = write!(
                        md,
                        " {} | {} |",
                        peak / (1024 * 1024),
                        mean / (1024 * 1024),
                    );
                    continue;
                }
                let dense = (dh.as_deref() == Some(h.as_str()))
                    .then(|| {
                        r.vram_stats
                            .iter()
                            .find(|v| v.device == *d && v.peak_allocated > 0)
                    })
                    .flatten();
                match dense {
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
            for d in &bare {
                let dense = dh
                    .is_none()
                    .then(|| {
                        r.vram_stats
                            .iter()
                            .find(|v| v.device == *d && v.peak_allocated > 0)
                    })
                    .flatten();
                match dense {
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
            md.push('\n');
        }
    }
    md.push('\n');
}

pub(super) fn write_idle_analysis(md: &mut String, model: &str, runs: &[RunAnalysis]) {
    // Only show runs with REPORTABLE gaps: the row loop below drops
    // Startup-classified gaps, so gating on any-gap-at-all printed a
    // header over an empty table for every run whose only >=500ms gaps
    // were startup (the 2026-07-29 sweep report's empty idle section).
    let runs_with_gaps: Vec<&RunAnalysis> = runs.iter()
        .filter(|r| r.idle_gaps.iter()
            .any(|g| !matches!(g.cause, crate::analyze::IdleCause::Startup)))
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

/// `(runs carrying a dense-sampled GPU, total runs)`. `gpu_devices` is the
/// union of devices the local poller recorded and every idle figure derives
/// from it, so this is the idle sections' actual coverage. Reported rather
/// than reduced to a boolean because the interesting case is PARTIAL: a
/// cluster sweep whose controller gates its poll off has dense data on its
/// solo rows only, and "no gap was detected" then describes those rows while
/// reading as though it described the sweep.
pub(super) fn dense_coverage(groups: &[(String, Vec<RunAnalysis>)]) -> (usize, usize) {
    let mut dense = 0;
    let mut total = 0;
    for (_, runs) in groups {
        for r in runs {
            total += 1;
            if !r.gpu_devices.is_empty() {
                dense += 1;
            }
        }
    }
    (dense, total)
}

pub(super) fn write_idle_breakdown(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
    // No rows → no table: headers over nothing read as broken output,
    // not as "no idle worth reporting" (same class as the idle-analysis
    // header fix). Say the good news explicitly instead - but only when
    // there was something to find it in. This table covers non-solo runs, so
    // scope the dense check the same way: a cluster sweep whose controller
    // gated its poll off has dense data on the solo rows alone, and claiming
    // a clean result from those would be claiming it for runs nothing looked at.
    let any_rows = groups.iter().any(|(_, runs)| {
        runs.iter()
            .filter(|r| !r.mode.starts_with("solo"))
            .any(|r| r.idle_by_cause.iter().any(|c| c.total_ms >= 500.0))
    });
    if !any_rows {
        let any_dense_ddp = groups.iter().any(|(_, runs)| {
            runs.iter()
                .filter(|r| !r.mode.starts_with("solo"))
                .any(|r| !r.gpu_devices.is_empty())
        });
        if any_dense_ddp {
            md.push_str(
                "No non-solo run accumulated >=0.5s of classified idle on a \
dense-sampled device.\n\n",
            );
        } else {
            md.push_str(
                "**Not measured in this sweep.** No non-solo run carried a \
dense-sampled device, so there was nothing to classify - see the note under \
GPU Idle Analysis.\n\n",
            );
        }
        return;
    }
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

    /// With a controller stamp, every device renders HOST-QUALIFIED —
    /// the controller's rank included (`ctrl-host:GPU0`); no bare
    /// `GPUd` column exists. (The pre-2026-07-29 convention folded the
    /// controller's ranks into a bare dense column instead — a bare
    /// label that meant different hardware depending on which box a
    /// row's poller ran on, the ambiguity this scheme retired.)
    #[test]
    fn model_table_folds_controller_host() {
        let runs = vec![cluster_run(Some("ctrl-host"))];
        let mut md = String::new();
        write_model_table(&mut md, "olmo", &runs, None);
        assert!(md.contains("remote-host:GPU0"), "remote column missing:\n{md}");
        assert!(md.contains("ctrl-host:GPU0"), "controller column missing:\n{md}");
        assert!(!md.contains("| GPU0 |"), "no bare column may remain:\n{md}");
        // Both utils render as sparse means with the ~ marker (the
        // fixture has no dense samples).
        assert!(md.contains("~80%"), "remote util missing:\n{md}");
        assert!(md.contains("~70%"), "controller-host util missing:\n{md}");
    }

    /// A solo run's DENSE devices attribute to their host via the
    /// physical fingerprint (device id + VRAM total) matched against a
    /// sibling cluster run's rank samples — the pascal-solo case that
    /// motivated the scheme: without it, solo rows rendered their
    /// pascal-local device ids under the same bare columns as the
    /// controller box's devices.
    #[test]
    fn model_table_attributes_solo_dense_by_fingerprint() {
        // Cluster run: a rank on "pascal" device 0 with a 6GB card.
        let mut cluster = empty_analysis("olmo", "cpu-cadence");
        cluster.rank_res = vec![RankResStats {
            rank: 1,
            host: "pascal".to_string(),
            device: 0,
            mean_util: Some(50.0),
            peak_allocated: Some(3_000_000_000),
            mean_allocated: Some(3_000_000_000),
            vram_total: Some(6_000_000_000),
            n_samples: 5,
        }];
        // Solo run: dense poller sampled device 0 with the same 6GB
        // total — physically the same card.
        let mut solo = empty_analysis("olmo", "solo-1");
        solo.gpu_devices = vec![0];
        solo.gpu_active_pct = vec![97.0];
        solo.vram_stats = vec![crate::analyze::VramStats {
            device: 0,
            peak_allocated: 2_000_000_000,
            mean_allocated: 1_500_000_000,
            total: 6_000_000_000,
        }];
        let runs = vec![cluster, solo];
        let mut md = String::new();
        write_model_table(&mut md, "olmo", &runs, None);
        assert!(md.contains("pascal:GPU0"), "fingerprint column missing:\n{md}");
        assert!(!md.contains("| GPU0 |"), "solo dense must not stay bare:\n{md}");
        // The solo row's dense reading lands qualified and UNmarked
        // (dense sampling, not sparse).
        assert!(md.contains(" 97% |"), "solo dense util missing:\n{md}");
    }

    /// An ambiguous fingerprint (same device id + same VRAM total seen
    /// on two hosts) must NOT fold — the run keeps the bare fallback
    /// column rather than guessing a box.
    #[test]
    fn model_table_ambiguous_fingerprint_stays_bare() {
        let mk_rank = |host: &str| RankResStats {
            rank: 0,
            host: host.to_string(),
            device: 0,
            mean_util: Some(50.0),
            peak_allocated: None,
            mean_allocated: None,
            vram_total: Some(6_000_000_000),
            n_samples: 1,
        };
        let mut cluster = empty_analysis("olmo", "cpu-cadence");
        cluster.rank_res = vec![mk_rank("box-a"), mk_rank("box-b")];
        let mut solo = empty_analysis("olmo", "solo-1");
        solo.gpu_devices = vec![0];
        solo.gpu_active_pct = vec![97.0];
        solo.vram_stats = vec![crate::analyze::VramStats {
            device: 0,
            peak_allocated: 2_000_000_000,
            mean_allocated: 1_500_000_000,
            total: 6_000_000_000,
        }];
        let runs = vec![cluster, solo];
        let mut md = String::new();
        write_model_table(&mut md, "olmo", &runs, None);
        assert!(md.contains("| GPU0 |"), "ambiguous print must fall back to bare:\n{md}");
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
