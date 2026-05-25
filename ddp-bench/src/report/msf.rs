//! MSF section: divergence-rate diagnostics + sparkline trajectories.

use std::fmt::Write;

use crate::analyze::RunAnalysis;

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

pub(super) fn write_msf_section(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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
