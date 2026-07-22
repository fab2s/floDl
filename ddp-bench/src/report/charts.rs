//! SVG chart generation for the report's focus model (`--charts <model>`).
//!
//! Hand-rolled SVG, zero dependencies - GitHub and flodl.dev render the
//! files natively from markdown image links. Charts synthesize what the
//! tables can't show without exploding: trajectories (eval, allocation,
//! sync cadence) and per-mode comparisons at a glance.
//!
//! Data honesty: idle/utilization figures come from the controller host's
//! sampler only (cluster cells can't see remote GPUs - see the timeline
//! resource-plumb item); trajectory charts (eval, share, syncs) are built
//! from rank-reported data and are complete across hosts.

use std::io::Write as _;
use std::path::Path;

use crate::analyze::{EventKind, RunAnalysis, Timeline};
use crate::analyze::log::TrainingLog;

/// One run's chart-relevant artifacts, retained by the report loop for
/// the focus model only.
pub struct ChartRun {
    pub mode: String,
    pub log: TrainingLog,
    pub timeline: Option<Timeline>,
}

/// Fixed palette + legend order (matches the report's mode vocabulary).
const MODE_STYLE: &[(&str, &str)] = &[
    ("solo-0", "#888888"),
    ("solo-1", "#bbbbbb"),
    ("solo-2", "#d9d9d9"),
    ("nccl-sync", "#d62728"),
    ("nccl-cadence", "#ff7f0e"),
    ("cpu-sync", "#9467bd"),
    ("cpu-cadence", "#1f77b4"),
    ("cpu-async", "#2ca02c"),
    ("cpu-async-diloco", "#17becf"),
];

fn mode_color(mode: &str) -> &'static str {
    MODE_STYLE
        .iter()
        .find(|(m, _)| *m == mode)
        .map(|(_, c)| *c)
        .unwrap_or("#333333")
}

fn mode_order(mode: &str) -> usize {
    MODE_STYLE
        .iter()
        .position(|(m, _)| *m == mode)
        .unwrap_or(MODE_STYLE.len())
}

/// Generate the chart set into `<dir>/charts/`. Returns
/// `(relative_path, caption)` pairs for the report to embed, in display
/// order. Runs lacking the data a chart needs are skipped silently - a
/// chart with fewer series beats a missing report.
pub fn write_charts(
    out_dir: &Path,
    model: &str,
    runs: &[ChartRun],
    analyses: &[&RunAnalysis],
) -> std::io::Result<Vec<(String, String)>> {
    let charts_dir = out_dir.join("charts");
    std::fs::create_dir_all(&charts_dir)?;

    let mut runs_sorted: Vec<&ChartRun> = runs.iter().collect();
    runs_sorted.sort_by_key(|r| mode_order(&r.mode));

    let mut links = Vec::new();
    let mut emit = |name: &str, svg: String, caption: &str| -> std::io::Result<()> {
        let mut f = std::fs::File::create(charts_dir.join(name))?;
        f.write_all(svg.as_bytes())?;
        links.push((format!("charts/{name}"), caption.to_string()));
        Ok(())
    };

    // --- 1a. Training loss over epochs (every mode logs it) ---
    let loss_series: Vec<Series> = runs_sorted
        .iter()
        .filter_map(|r| {
            let pts: Vec<(f64, f64)> = r
                .log
                .epochs
                .iter()
                .filter(|e| e.loss > 0.0)
                .map(|e| (e.epoch as f64, e.loss))
                .collect();
            (pts.len() >= 2).then(|| Series {
                name: r.mode.clone(),
                color: mode_color(&r.mode),
                points: pts,
            })
        })
        .collect();
    if !loss_series.is_empty() {
        emit(
            "loss-epochs.svg",
            line_chart_scaled(
                &format!("{model} - training loss over epochs (log scale)"),
                "epoch",
                "loss",
                &loss_series,
                true,
            ),
            "Training loss per epoch, all modes (log scale - the convergence \
             tail is where modes differ).",
        )?;
    }

    // --- 1b. Train accuracy trajectories, final eval in the legend.
    //          train_acc exists per-epoch on EVERY mode (DDP modes eval
    //          once after training by design, so a per-epoch eval chart
    //          would show solo alone); the final eval rides the legend
    //          label rather than a terminal marker - it is always below
    //          the train curve and a dangling point reads as an outlier.
    let acc_series: Vec<Series> = runs_sorted
        .iter()
        .filter_map(|r| {
            let pts: Vec<(f64, f64)> = r
                .log
                .epochs
                .iter()
                .filter_map(|e| e.train_acc.map(|v| (e.epoch as f64, v)))
                .collect();
            let eval = r.log.final_eval.or_else(|| {
                r.log.epochs.iter().rev().find_map(|e| e.eval)
            });
            let name = match eval {
                Some(v) => format!("{} - eval {v:.4}", r.mode),
                None => r.mode.clone(),
            };
            (pts.len() >= 2).then(|| Series {
                name,
                color: mode_color(&r.mode),
                points: pts,
            })
        })
        .collect();
    if acc_series.len() >= 2 {
        emit(
            "train-acc-epochs.svg",
            line_chart(
                &format!("{model} - train accuracy over epochs"),
                "epoch",
                "train accuracy",
                &acc_series,
            ),
            "Train accuracy per epoch, all modes (the LR-schedule steps are \
             the visible jumps); each mode's final eval - measured once \
             after training - is in the legend.",
        )?;
    }

    // --- 2. Fast-rank (rank 0) allocation share over epochs ---
    let share_series: Vec<Series> = runs_sorted
        .iter()
        .filter_map(|r| {
            let pts: Vec<(f64, f64)> = r
                .log
                .epochs
                .iter()
                .filter_map(|e| {
                    e.per_rank
                        .iter()
                        .find(|s| s.rank == 0)
                        .map(|s| (e.epoch as f64, s.batch_share))
                })
                .collect();
            (pts.len() >= 2).then(|| Series {
                name: r.mode.clone(),
                color: mode_color(&r.mode),
                points: pts,
            })
        })
        .collect();
    if !share_series.is_empty() {
        emit(
            "fast-rank-share.svg",
            line_chart(
                &format!("{model} - fast-rank (rank 0) allocation share"),
                "epoch",
                "share of dispatched batches",
                &share_series,
            ),
            "ElChe's smoothed allocation share for the fast rank, per epoch. \
             Under `*-sync` dispatch is an equal split and this is the \
             balancer's capacity shadow.",
        )?;
    }

    // --- 3. Cumulative reduces over wall time ---
    let sync_series: Vec<Series> = runs_sorted
        .iter()
        .filter_map(|r| {
            let tl = r.timeline.as_ref()?;
            let mut pts = vec![(0.0, 0.0)];
            let mut n = 0.0;
            for ev in &tl.events {
                if matches!(ev.kind, EventKind::SyncStart) {
                    n += 1.0;
                    pts.push((ev.t as f64 / 1000.0, n));
                }
            }
            (pts.len() >= 3).then(|| Series {
                name: r.mode.clone(),
                color: mode_color(&r.mode),
                points: pts,
            })
        })
        .collect();
    if !sync_series.is_empty() {
        emit(
            "syncs-cumulative.svg",
            line_chart(
                &format!("{model} - cumulative reduces over wall time"),
                "wall time (s)",
                "reduces",
                &sync_series,
            ),
            "Window growth made visible: a flattening curve = the anchor \
             amortizing sync cost; a steep straight line = per-step reduces.",
        )?;
    }

    // --- 4 + 5. Wall time and controller-GPU idle bars ---
    let mut by_mode: Vec<&&RunAnalysis> = analyses.iter().collect();
    by_mode.sort_by_key(|a| mode_order(&a.mode));
    let wall_bars: Vec<Bar> = by_mode
        .iter()
        .filter(|a| a.total_ms > 0)
        .map(|a| Bar {
            name: a.mode.clone(),
            color: mode_color(&a.mode),
            value: a.total_ms as f64 / 1000.0,
        })
        .collect();
    if wall_bars.len() >= 2 {
        emit(
            "wall-time.svg",
            bar_chart(&format!("{model} - wall time"), "seconds", &wall_bars),
            "Total wall time per mode (same epochs, same seed).",
        )?;
    }
    let idle_bars: Vec<Bar> = by_mode
        .iter()
        .filter_map(|a| {
            let idx = a.gpu_devices.iter().position(|&d| d == 0)?;
            let active = a.gpu_active_pct.get(idx)?;
            Some(Bar {
                name: a.mode.clone(),
                color: mode_color(&a.mode),
                value: (100.0 - active).clamp(0.0, 100.0),
            })
        })
        .collect();
    if idle_bars.len() >= 2 {
        emit(
            "idle-gpu0.svg",
            bar_chart(
                &format!("{model} - GPU0 idle (controller host)"),
                "% of run below 5% utilization",
                &idle_bars,
            ),
            "Idle share of the controller host's GPU only - cluster cells \
             cannot sample remote GPUs (see Notes).",
        )?;
    }

    Ok(links)
}

// ---------------------------------------------------------------------------
// SVG primitives
// ---------------------------------------------------------------------------

struct Series {
    name: String,
    color: &'static str,
    points: Vec<(f64, f64)>,
}

struct Bar {
    name: String,
    color: &'static str,
    value: f64,
}

const FONT: &str = "font-family=\"sans-serif\"";

/// Multi-series line chart with axes, gridlines and a right-hand legend.
fn line_chart(title: &str, x_label: &str, y_label: &str, series: &[Series]) -> String {
    line_chart_scaled(title, x_label, y_label, series, false)
}

/// [`line_chart`] with an optional log10 y-axis (positive values only -
/// callers filter). Tick labels always show real values.
fn line_chart_scaled(
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[Series],
    y_log: bool,
) -> String {
    let (w, h) = (960.0, 480.0);
    let (l, r, t, b) = (72.0, 240.0, 48.0, 56.0);
    let (pw, ph) = (w - l - r, h - t - b);

    let ty = |y: f64| if y_log { y.max(1e-12).log10() } else { y };
    let all = series.iter().flat_map(|s| s.points.iter());
    let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for &(x, y) in all {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(ty(y));
        y1 = y1.max(ty(y));
    }
    if x1 <= x0 {
        x1 = x0 + 1.0;
    }
    if y1 <= y0 {
        y1 = y0 + 1.0;
    }
    // Pad y so curves don't hug the frame; keep zero-based when close
    // (linear scale only - log has no zero).
    let ypad = ((y1 - y0) * 0.08).max(1e-9);
    y0 = if !y_log && y0 > 0.0 && y0 < ypad * 4.0 { 0.0 } else { y0 - ypad };
    y1 += ypad;

    let px = |x: f64| l + (x - x0) / (x1 - x0) * pw;
    let py = |y: f64| t + ph - (ty(y) - y0) / (y1 - y0) * ph;

    let mut s = svg_open(w, h, title);
    // Gridlines + ticks (5 divisions each axis).
    for i in 0..=5 {
        let fx = x0 + (x1 - x0) * i as f64 / 5.0;
        // fy lives in transformed space; labels always show real values.
        let fy_t = y0 + (y1 - y0) * i as f64 / 5.0;
        let fy = if y_log { 10f64.powf(fy_t) } else { fy_t };
        let gx = px(fx);
        let gy = t + ph - (fy_t - y0) / (y1 - y0) * ph;
        s.push_str(&format!(
            "<line x1='{gx:.1}' y1='{t}' x2='{gx:.1}' y2='{:.1}' stroke='#eee'/>\
             <line x1='{l}' y1='{gy:.1}' x2='{:.1}' y2='{gy:.1}' stroke='#eee'/>\
             <text x='{gx:.1}' y='{:.1}' text-anchor='middle' font-size='12' {FONT} fill='#555'>{}</text>\
             <text x='{:.1}' y='{:.1}' text-anchor='end' font-size='12' {FONT} fill='#555'>{}</text>",
            t + ph,
            l + pw,
            t + ph + 18.0,
            fmt_tick(fx, x1 - x0),
            l - 8.0,
            gy + 4.0,
            fmt_tick(
                fy,
                if y_log {
                    10f64.powf(y1) - 10f64.powf(y0)
                } else {
                    y1 - y0
                },
            ),
        ));
    }
    // Axes frame.
    s.push_str(&format!(
        "<rect x='{l}' y='{t}' width='{pw}' height='{ph}' fill='none' stroke='#999'/>"
    ));
    // Axis labels.
    s.push_str(&format!(
        "<text x='{:.1}' y='{:.1}' text-anchor='middle' font-size='13' {FONT} fill='#333'>{x_label}</text>\
         <text x='18' y='{:.1}' text-anchor='middle' font-size='13' {FONT} fill='#333' \
          transform='rotate(-90 18 {:.1})'>{y_label}</text>",
        l + pw / 2.0,
        h - 14.0,
        t + ph / 2.0,
        t + ph / 2.0,
    ));
    // Series polylines + legend. Dense series are decimated to ~400
    // points (keeping the last) - a reduce-per-step run contributes
    // 15k+ points that render identically at chart resolution but
    // multiply the file size ~100x.
    for (i, sr) in series.iter().enumerate() {
        let stride = (sr.points.len() / 400).max(1);
        let pts: String = sr
            .points
            .iter()
            .enumerate()
            .filter(|(j, _)| j % stride == 0 || *j == sr.points.len() - 1)
            .map(|(_, &(x, y))| format!("{:.1},{:.1} ", px(x), py(y)))
            .collect();
        s.push_str(&format!(
            "<polyline points='{}' fill='none' stroke='{}' stroke-width='1.8'/>",
            pts.trim_end(),
            sr.color
        ));
        let ly = t + 8.0 + i as f64 * 20.0;
        s.push_str(&format!(
            "<line x1='{:.1}' y1='{ly:.1}' x2='{:.1}' y2='{ly:.1}' stroke='{}' stroke-width='3'/>\
             <text x='{:.1}' y='{:.1}' font-size='12' {FONT} fill='#333'>{}</text>",
            l + pw + 16.0,
            l + pw + 44.0,
            sr.color,
            l + pw + 50.0,
            ly + 4.0,
            sr.name,
        ));
    }
    s.push_str("</svg>\n");
    s
}

/// Horizontal bar chart, one bar per mode, value printed at the bar end.
fn bar_chart(title: &str, x_label: &str, bars: &[Bar]) -> String {
    let row = 34.0;
    let (l, r, t, b) = (150.0, 90.0, 48.0, 40.0);
    let w = 960.0;
    let h = t + b + bars.len() as f64 * row;
    let pw = w - l - r;
    let vmax = bars.iter().map(|b| b.value).fold(f64::MIN, f64::max).max(1e-9);

    let mut s = svg_open(w, h, title);
    for (i, bar) in bars.iter().enumerate() {
        let y = t + i as f64 * row;
        let bw = bar.value / vmax * pw;
        s.push_str(&format!(
            "<rect x='{l}' y='{:.1}' width='{bw:.1}' height='{:.1}' fill='{}'/>\
             <text x='{:.1}' y='{:.1}' text-anchor='end' font-size='12' {FONT} fill='#333'>{}</text>\
             <text x='{:.1}' y='{:.1}' font-size='12' {FONT} fill='#333'>{}</text>",
            y + 5.0,
            row - 12.0,
            bar.color,
            l - 8.0,
            y + row / 2.0 + 3.0,
            bar.name,
            l + bw + 6.0,
            y + row / 2.0 + 3.0,
            fmt_tick(bar.value, vmax),
        ));
    }
    s.push_str(&format!(
        "<text x='{:.1}' y='{:.1}' text-anchor='middle' font-size='13' {FONT} fill='#333'>{x_label}</text>",
        l + pw / 2.0,
        h - 12.0,
    ));
    s.push_str("</svg>\n");
    s
}

fn svg_open(w: f64, h: f64, title: &str) -> String {
    format!(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}'>\
         <rect width='{w}' height='{h}' fill='white'/>\
         <text x='{:.1}' y='28' text-anchor='middle' font-size='16' {FONT} fill='#111'>{title}</text>",
        w / 2.0,
    )
}

/// Tick formatting: precision follows the value range so 0.9155 and 1650
/// both come out readable.
fn fmt_tick(v: f64, range: f64) -> String {
    if range >= 50.0 {
        format!("{v:.0}")
    } else if range >= 2.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}
