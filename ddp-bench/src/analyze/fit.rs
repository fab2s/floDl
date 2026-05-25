//! OLS fitting and per-LR-window MSF λ̂ estimation.

use super::msf::MsfLrWindowFit;
use super::{Event, EventKind};

/// LR change above this fraction starts a new auto-detected window.
/// Step-decays jump 10x and trigger cleanly; cosine schedules accumulate
/// into ~5%-step buckets which is acceptable resolution for analysis.
pub(super) const LR_WINDOW_CHANGE_FRAC: f64 = 0.05;

/// Result of an OLS fit over the (xs, ys) collected for an LR window.
/// `r2 = 1.0` when the y-series has zero variance (degenerate but finite).
struct OlsFit {
    slope: f64,
    r2: f64,
}

/// OLS slope + R² for a (xs, ys) sample. Returns `None` if `sxx <= 0`
/// (degenerate x — should not happen for distinct steps).
fn ols(xs: &[f64], ys: &[f64]) -> Option<OlsFit> {
    let n = xs.len();
    if n != ys.len() || n < 5 {
        return None;
    }
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for k in 0..n {
        let dx = xs[k] - mx;
        let dy = ys[k] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx <= 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    let r2 = if syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        1.0
    };
    Some(OlsFit { slope, r2 })
}

/// Collected event data for an LR window: per-event step + ln(D_max), and
/// (when per-rank `deltas` are available) ln(D_mean). Used as the input to
/// both bases' OLS fits and to the transient-detection heuristic.
struct LrWindowSamples {
    /// Per-event (step, epoch, ln_d_max).
    pts_max: Vec<(usize, usize, f64)>,
    /// Per-event ln(D_mean) parallel to `pts_max`. `None` entry when that
    /// event had empty `deltas`. The fit drops `None` entries; if any are
    /// present the d_mean fit reports on the surviving subset only.
    pts_mean: Vec<Option<f64>>,
    /// Per-event ln(D_i) for each rank i, parallel to `pts_max`. Outer Vec
    /// indexed by rank, inner Vec parallel to `pts_max` with `None` for
    /// events that had empty `deltas` or a different rank count than the
    /// dominant one for this window. Empty (outer Vec length 0) when no
    /// event had non-empty `deltas` or rank count varied.
    pts_per_rank: Vec<Vec<Option<f64>>>,
    /// Per-event `k_used` (cycle length: actual training steps elapsed since
    /// the previous AllReduce). Parallel to `pts_max`. The "step since last
    /// sync" axis for the alternative R1 framing.
    pts_k_used: Vec<usize>,
    /// True when at least one event in the window carried non-empty `deltas`.
    has_deltas: bool,
}

/// Walk events for the (start_ep, end_ep) range and collect the per-event
/// samples needed for both the D_max and D_mean OLS fits.
fn collect_lr_window_samples(
    events: &[Event],
    epoch_step_ranges: &[(usize, usize, usize)],
    start_ep: usize,
    end_ep: usize,
) -> Option<LrWindowSamples> {
    let mut step_lo = usize::MAX;
    let mut step_hi = 0usize;
    for (ep, lo, hi) in epoch_step_ranges {
        if *ep >= start_ep && *ep <= end_ep {
            step_lo = step_lo.min(*lo);
            step_hi = step_hi.max(*hi);
        }
    }
    if step_lo > step_hi {
        return None;
    }
    // Step → epoch lookup for events whose `epoch` field is `None` (old
    // timelines predating per-event epoch tagging).
    let epoch_for_step = |s: usize| -> usize {
        epoch_step_ranges
            .iter()
            .find_map(|(ep, lo, hi)| {
                if *lo <= s && s <= *hi {
                    Some(*ep)
                } else {
                    None
                }
            })
            .unwrap_or(0)
    };
    let mut pts_max: Vec<(usize, usize, f64)> = Vec::new();
    let mut pts_mean: Vec<Option<f64>> = Vec::new();
    let mut pts_k_used: Vec<usize> = Vec::new();
    // Collect raw deltas first; we'll build the per-rank ln-vectors after we
    // know the dominant rank count for this window (most-common deltas.len()).
    let mut pts_deltas: Vec<Option<Vec<f64>>> = Vec::new();
    let mut has_deltas = false;
    for e in events {
        if let EventKind::Divergence {
            d_raw,
            step,
            epoch,
            deltas,
            k_used,
            ..
        } = &e.kind
            && *step >= step_lo
            && *step <= step_hi
            && *d_raw > 1e-12
        {
            let resolved_epoch = epoch.unwrap_or_else(|| epoch_for_step(*step));
            pts_max.push((*step, resolved_epoch, d_raw.ln()));
            pts_k_used.push(*k_used);
            if !deltas.is_empty() {
                has_deltas = true;
                let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
                if mean > 1e-12 {
                    pts_mean.push(Some(mean.ln()));
                } else {
                    pts_mean.push(None);
                }
                pts_deltas.push(Some(deltas.clone()));
            } else {
                pts_mean.push(None);
                pts_deltas.push(None);
            }
        }
    }
    if pts_max.is_empty() {
        return None;
    }

    // Determine dominant rank count (events whose rank count differs are
    // dropped from the per-rank fit). For homogeneous runs this is always
    // the same number; defensive against partial events.
    let dominant_n_ranks = {
        use std::collections::HashMap;
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for v in pts_deltas.iter().flatten() {
            *counts.entry(v.len()).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(n, _)| n).unwrap_or(0)
    };

    // Build per-rank ln-vectors. Outer index = rank, inner parallel to pts_max.
    let pts_per_rank: Vec<Vec<Option<f64>>> = if dominant_n_ranks > 0 {
        (0..dominant_n_ranks)
            .map(|rank| {
                pts_deltas
                    .iter()
                    .map(|d| {
                        d.as_ref()
                            .filter(|v| v.len() == dominant_n_ranks)
                            .and_then(|v| {
                                let x = v[rank];
                                if x > 1e-12 { Some(x.ln()) } else { None }
                            })
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };

    Some(LrWindowSamples {
        pts_max,
        pts_mean,
        pts_per_rank,
        pts_k_used,
        has_deltas,
    })
}

/// Run OLS for both bases on the collected samples, optionally trimming the
/// first `skip_first_n_epochs` epochs of the window. Returns `None` if the
/// D_max fit doesn't have enough points.
fn fit_samples(samples: &LrWindowSamples, skip_first_n_epochs: usize) -> Option<MsfLrWindowFit> {
    let cutoff_epoch: Option<usize> = if skip_first_n_epochs == 0 {
        None
    } else {
        let first_ep = samples.pts_max.first().map(|(_, ep, _)| *ep)?;
        Some(first_ep + skip_first_n_epochs)
    };
    let mut xs_max: Vec<f64> = Vec::new();
    let mut ys_max: Vec<f64> = Vec::new();
    let mut xs_mean: Vec<f64> = Vec::new();
    let mut ys_mean: Vec<f64> = Vec::new();
    let mut step_lo = usize::MAX;
    let mut step_hi = 0usize;
    let mut start_ep = usize::MAX;
    let mut end_ep = 0usize;
    for (i, (step, epoch, ln_max)) in samples.pts_max.iter().enumerate() {
        if let Some(ce) = cutoff_epoch
            && *epoch < ce
        {
            continue;
        }
        xs_max.push(*step as f64);
        ys_max.push(*ln_max);
        step_lo = step_lo.min(*step);
        step_hi = step_hi.max(*step);
        start_ep = start_ep.min(*epoch);
        end_ep = end_ep.max(*epoch);
        if let Some(ln_mean) = samples.pts_mean.get(i).and_then(|p| *p) {
            xs_mean.push(*step as f64);
            ys_mean.push(ln_mean);
        }
    }
    let n = xs_max.len();
    if n < 5 {
        return None;
    }
    let fit_max = ols(&xs_max, &ys_max)?;
    let (slope_dmean, r2_dmean) = if samples.has_deltas {
        match ols(&xs_mean, &ys_mean) {
            Some(f) => (Some(f.slope), Some(f.r2)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    // Per-epoch aggregation: collapse intra-epoch SGD variance by averaging
    // `ln(D_mean)` and `step` within each epoch, then fit those aggregates.
    // Uses log-mean (mean of ln(D)) rather than ln of arithmetic-mean(D) so
    // that R1 linearity is preserved under the aggregation.
    let (slope_epoch_dmean, r2_epoch_dmean, n_epoch_points) = if samples.has_deltas {
        use std::collections::BTreeMap;
        let mut by_epoch: BTreeMap<usize, (Vec<f64>, Vec<usize>)> = BTreeMap::new();
        for (i, (step, epoch, _)) in samples.pts_max.iter().enumerate() {
            if let Some(ce) = cutoff_epoch
                && *epoch < ce
            {
                continue;
            }
            if let Some(ln_mean) = samples.pts_mean.get(i).and_then(|p| *p) {
                let entry = by_epoch.entry(*epoch).or_default();
                entry.0.push(ln_mean);
                entry.1.push(*step);
            }
        }
        let mut xs_ep: Vec<f64> = Vec::new();
        let mut ys_ep: Vec<f64> = Vec::new();
        for (ln_means, steps) in by_epoch.values() {
            if ln_means.is_empty() {
                continue;
            }
            let y_mean = ln_means.iter().sum::<f64>() / ln_means.len() as f64;
            let x_mean =
                steps.iter().map(|s| *s as f64).sum::<f64>() / steps.len() as f64;
            xs_ep.push(x_mean);
            ys_ep.push(y_mean);
        }
        let n_ep = xs_ep.len();
        match ols(&xs_ep, &ys_ep) {
            Some(f) => (Some(f.slope), Some(f.r2), Some(n_ep)),
            None => (None, None, if n_ep > 0 { Some(n_ep) } else { None }),
        }
    } else {
        (None, None, None)
    };

    // Alternative R1 axis: ln(D_mean) vs k_used (steps since last sync).
    // Sync resets transversal deviation to ~0, so the natural drift clock
    // restarts at every AllReduce. If pure exponential growth holds within
    // a cycle, then D_t ≈ ε·exp(λ_T · k_used) and ln(D_t) is linear in
    // k_used at fixed LR. If the system is in the OU/spiral-to-consensus
    // regime, D_t saturates toward a setpoint D*(LR) and the slope flattens
    // for large k_used.
    let mut xs_k: Vec<f64> = Vec::new();
    let mut ys_k: Vec<f64> = Vec::new();
    let mut k_lo = usize::MAX;
    let mut k_hi = 0usize;
    for (i, (_step, epoch, _ln_max)) in samples.pts_max.iter().enumerate() {
        if let Some(ce) = cutoff_epoch
            && *epoch < ce
        {
            continue;
        }
        if let Some(ln_mean) = samples.pts_mean.get(i).and_then(|p| *p) {
            let k = samples.pts_k_used[i];
            xs_k.push(k as f64);
            ys_k.push(ln_mean);
            k_lo = k_lo.min(k);
            k_hi = k_hi.max(k);
        }
    }
    let (slope_by_k_used_dmean, r2_by_k_used_dmean, k_used_min, k_used_max) =
        if samples.has_deltas && xs_k.len() >= 5 {
            match ols(&xs_k, &ys_k) {
                Some(f) => (
                    Some(f.slope),
                    Some(f.r2),
                    Some(k_lo),
                    Some(k_hi),
                ),
                None => (None, None, Some(k_lo), Some(k_hi)),
            }
        } else {
            (None, None, None, None)
        };

    // Per-rank by-k OLS — bottom-scale Lyapunov estimate per rank. Used as
    // a consistency check on the meta-oscillator framing: under r > 0.99
    // cross-rank Pearson, per-rank slopes should match the meta-D_mean
    // slope. Per-rank divergence indicates the framing is breaking and
    // bottom-scale per-rank treatment is required (e.g. cpu-async backend).
    let (slope_by_k_per_rank, r2_by_k_per_rank): (Vec<f64>, Vec<f64>) =
        if !samples.pts_per_rank.is_empty() {
            let mut slopes = Vec::with_capacity(samples.pts_per_rank.len());
            let mut r2s = Vec::with_capacity(samples.pts_per_rank.len());
            for rank_pts in &samples.pts_per_rank {
                let mut xs_rk: Vec<f64> = Vec::new();
                let mut ys_rk: Vec<f64> = Vec::new();
                for (i, (_step, epoch, _ln_max)) in samples.pts_max.iter().enumerate() {
                    if let Some(ce) = cutoff_epoch
                        && *epoch < ce
                    {
                        continue;
                    }
                    if let Some(ln_d_i) = rank_pts.get(i).and_then(|v| *v) {
                        xs_rk.push(samples.pts_k_used[i] as f64);
                        ys_rk.push(ln_d_i);
                    }
                }
                if xs_rk.len() >= 5
                    && let Some(f) = ols(&xs_rk, &ys_rk)
                {
                    slopes.push(f.slope);
                    r2s.push(f.r2);
                    continue;
                }
                // Fill placeholder so per-rank vector length stays = n_ranks.
                slopes.push(f64::NAN);
                r2s.push(f64::NAN);
            }
            (slopes, r2s)
        } else {
            (Vec::new(), Vec::new())
        };

    Some(MsfLrWindowFit {
        lr: 0.0, // filled in by caller
        epoch_start: start_ep,
        epoch_end: end_ep,
        n_events: n,
        step_min: step_lo,
        step_max: step_hi,
        slope_per_step: fit_max.slope,
        r2: fit_max.r2,
        slope_per_step_dmean: slope_dmean,
        r2_dmean,
        slope_per_step_epoch_dmean: slope_epoch_dmean,
        r2_epoch_dmean,
        n_epoch_points,
        slope_by_k_used_dmean,
        r2_by_k_used_dmean,
        k_used_min,
        k_used_max,
        slope_by_k_per_rank,
        r2_by_k_per_rank,
        transient_skipped: skip_first_n_epochs,
    })
}

/// Heuristic: count leading epochs in the window where D_max is anomalously
/// high vs the rest (initialization transient). Returns 0 if no clear
/// transient is detected. Threshold: epoch is "transient" if its peak D_max
/// exceeds 1.5× the median of the remaining window's per-epoch peak D_max.
/// Capped at 20% of window length to avoid eating into stable-LR data.
fn detect_transient_epochs(samples: &LrWindowSamples) -> usize {
    if samples.pts_max.is_empty() {
        return 0;
    }
    use std::collections::BTreeMap;
    let mut peak_by_epoch: BTreeMap<usize, f64> = BTreeMap::new();
    for (_, epoch, ln_max) in &samples.pts_max {
        let d_max = ln_max.exp();
        let cur = peak_by_epoch.entry(*epoch).or_insert(0.0);
        if d_max > *cur {
            *cur = d_max;
        }
    }
    let epochs_sorted: Vec<(usize, f64)> = peak_by_epoch.into_iter().collect();
    let total = epochs_sorted.len();
    if total < 10 {
        return 0;
    }
    let cap = (total / 5).max(1);
    let mut skipped = 0usize;
    while skipped < cap {
        let remainder = &epochs_sorted[skipped..];
        if remainder.len() < 5 {
            break;
        }
        let mut peaks: Vec<f64> = remainder.iter().map(|(_, p)| *p).collect();
        peaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = peaks[peaks.len() / 2];
        let head_peak = epochs_sorted[skipped].1;
        if med > 0.0 && head_peak > 1.5 * med {
            skipped += 1;
        } else {
            break;
        }
    }
    skipped
}

/// Compute log(D) vs cumulative step OLS within a (start_epoch, end_epoch)
/// range, on both D_max and D_mean bases. Emits one `MsfLrWindowFit` for the
/// full window; for the first window of the run, also emits a post-transient
/// fit when the initialization-transient heuristic detects leading epochs to
/// trim.
pub(super) fn fit_lr_window(
    events: &[Event],
    epoch_step_ranges: &[(usize, usize, usize)],
    start_ep: usize,
    end_ep: usize,
    is_first_window: bool,
) -> Vec<MsfLrWindowFit> {
    let Some(samples) = collect_lr_window_samples(events, epoch_step_ranges, start_ep, end_ep)
    else {
        return Vec::new();
    };
    let mut out: Vec<MsfLrWindowFit> = Vec::new();
    if let Some(full) = fit_samples(&samples, 0) {
        out.push(full);
    }
    if is_first_window {
        let skip = detect_transient_epochs(&samples);
        if skip > 0
            && let Some(post) = fit_samples(&samples, skip)
        {
            out.push(post);
        }
    }
    out
}
