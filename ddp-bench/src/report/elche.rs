//! ElChe-mode detail table (anchor sequence + cadence breakdown).

use std::fmt::Write;

use crate::analyze::RunAnalysis;

pub(super) fn write_elche_details(md: &mut String, groups: &[(String, Vec<RunAnalysis>)]) {
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
