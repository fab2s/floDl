//! Training log parser (`training.log` → `TrainingLog` → applied onto `RunAnalysis`).

use std::path::Path;

use super::{EpochData, PerRankAvg, RunAnalysis};

// ---------------------------------------------------------------------------
// Training log parser
// ---------------------------------------------------------------------------

/// Parsed training log data.
pub struct TrainingLog {
    pub epochs: Vec<LogEpoch>,
    /// Standalone `final eval=X.XXXX` line (modes that eval once after training).
    pub final_eval: Option<f64>,
    /// Total wall time from `# total:` footer (ms).
    pub total_ms: Option<f64>,
    /// Training-only wall time from `# train_only:` summary (ms). Set by
    /// `run_baseline_solo` so the report can compare DDP wall time against
    /// solo's training-only time, excluding the per-epoch eval cost solo
    /// pays but DDP only pays once at the end.
    pub train_only_ms: Option<f64>,
    /// GPU header lines (e.g. `gpu0: NVIDIA GeForce RTX 5060 Ti (15GB, sm_120)`).
    pub gpu_info: Vec<String>,
}

/// One epoch line from the training log.
pub struct LogEpoch {
    pub epoch: usize,
    pub loss: f64,
    pub eval: Option<f64>,
    /// Training-set accuracy for the epoch (`train_acc=X.XXXX`), emitted
    /// by models with an accuracy metric. Unlike eval it exists per-epoch
    /// on every mode (DDP modes eval once at the end), so it carries the
    /// per-epoch convergence trajectory in the charts.
    pub train_acc: Option<f64>,
    /// Training-only wall time for the epoch (ms). Parsed from `train=Xs`
    /// (new format) or `time=Xs` (legacy, where the value was already
    /// training-only).
    pub time_ms: f64,
    /// Per-rank breakdown from the `per-rank:` line that follows multi-rank
    /// epoch lines. Empty for solo and single-rank runs.
    pub per_rank: Vec<RankSnapshot>,
}

/// Per-rank stats for one epoch.
#[derive(Debug, Clone)]
pub struct RankSnapshot {
    pub rank: usize,
    pub device: u8,
    /// The balancer's smoothed allocation share (ElChe `batch_counts`,
    /// 0..1, sums to ~1 across ranks). Equals the delivered work split
    /// under cadence/async (dispatch follows the balancer); under
    /// `*-sync` modes dispatch is an equal split and this is ElChe's
    /// capacity shadow, not delivered work.
    pub batch_share: f64,
    /// Throughput in samples/ms (the rank's own work; peer-wait excluded).
    pub throughput: f64,
}

/// Parse a `training.log` file.
///
/// Format:
/// ```text
/// # gpu0: ...
/// epoch 0: loss=0.311125, eval=0.9732, time=2.2s
/// epoch 1: loss=0.131376, time=2.3s
/// final eval=0.9732
/// # total: 12.7s (0m 13s)
/// ```
pub fn parse_training_log(path: &Path) -> Result<TrainingLog, String> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

    let mut epochs: Vec<LogEpoch> = Vec::new();
    let mut final_eval = None;
    let mut total_ms = None;
    let mut train_only_ms = None;
    let mut gpu_info = Vec::new();

    for line in data.lines() {
        let line = line.trim();

        // # gpu0: NVIDIA GeForce RTX 5060 Ti (15GB, sm_120)
        if let Some(rest) = line.strip_prefix("# gpu") {
            if rest.contains(':') {
                gpu_info.push(format!("gpu{rest}"));
            }
            continue;
        }

        // epoch N: loss=X.XXXXXX[, eval=X.XXXX][, train_acc=X.XXXX][, train=X.Xs|time=X.Xs][, eval_time=X.Xs]
        if let Some(rest) = line.strip_prefix("epoch ") {
            if let Some((epoch_str, kv_part)) = rest.split_once(": ") {
                let epoch: usize = epoch_str.parse().unwrap_or(0);
                let mut loss = 0.0;
                let mut eval = None;
                let mut train_acc = None;
                let mut time_ms = 0.0;

                for kv in kv_part.split(", ") {
                    if let Some(v) = kv.strip_prefix("loss=") {
                        loss = v.parse().unwrap_or(0.0);
                    } else if let Some(v) = kv.strip_prefix("eval=")
                        .or_else(|| kv.strip_prefix("metric="))
                    {
                        eval = Some(v.parse().unwrap_or(0.0));
                    } else if let Some(v) = kv.strip_prefix("train_acc=") {
                        train_acc = Some(v.parse().unwrap_or(0.0));
                    } else if let Some(v) = kv.strip_prefix("train=")
                        .or_else(|| kv.strip_prefix("time="))
                    {
                        // `train=` is the new explicit form (training-only);
                        // `time=` is the legacy column (also training-only,
                        // since epoch_ms was always measured before eval).
                        if let Some(ms) = v.strip_suffix("ms") {
                            time_ms = ms.parse::<f64>().unwrap_or(0.0);
                        } else if let Some(secs) = v.strip_suffix('s') {
                            time_ms = secs.parse::<f64>().unwrap_or(0.0) * 1000.0;
                        }
                    }
                    // eval_time is parsed but not stored on LogEpoch yet; the
                    // total is in the `# train_only: Xs (eval: Ys)` summary.
                }

                // Merge into an existing same-epoch entry if one already
                // exists (the harness emits a "loss-only" line during the
                // metrics tick and may emit a supplemental "eval-only"
                // line afterwards when the EpochFn finishes after the
                // metrics line was already printed). Last non-default
                // value wins per field.
                if let Some(prev) = epochs.iter_mut().find(|e| e.epoch == epoch) {
                    if loss != 0.0 {
                        prev.loss = loss;
                    }
                    if eval.is_some() {
                        prev.eval = eval;
                    }
                    if train_acc.is_some() {
                        prev.train_acc = train_acc;
                    }
                    if time_ms != 0.0 {
                        prev.time_ms = time_ms;
                    }
                } else {
                    epochs.push(LogEpoch {
                        epoch, loss, eval, train_acc, time_ms, per_rank: Vec::new(),
                    });
                }
            }
        }
        // per-rank: rank0[cuda0,share=0.3447,tput=56.88] rank1[cuda1,share=...]
        else if let Some(rest) = line.strip_prefix("per-rank:") {
            let snapshots = parse_per_rank_line(rest);
            if let Some(last) = epochs.last_mut() {
                last.per_rank = snapshots;
            }
        }
        // final eval=X.XXXX
        else if let Some(v) = line.strip_prefix("final eval=") {
            final_eval = Some(v.parse().unwrap_or(0.0));
        }
        // # train_only: 13.6s (eval: 0.6s)
        else if let Some(rest) = line.strip_prefix("# train_only: ")
            && let Some(secs_str) = rest.split('s').next()
        {
            train_only_ms = secs_str.trim().parse::<f64>().ok().map(|s| s * 1000.0);
        }
        // # total: 12.7s (0m 13s)
        else if let Some(rest) = line.strip_prefix("# total: ")
            && let Some(secs_str) = rest.split('s').next()
        {
            total_ms = secs_str.trim().parse::<f64>().ok().map(|s| s * 1000.0);
        }
    }

    Ok(TrainingLog { epochs, final_eval, total_ms, train_only_ms, gpu_info })
}

/// Parse the body of a `per-rank:` line into [`RankSnapshot`]s.
///
/// Format (from `harness::run_unified`):
/// ` rank0[cuda0,share=0.3447,tput=56.88] rank1[cuda1,share=0.3533,tput=57.35]`
fn parse_per_rank_line(rest: &str) -> Vec<RankSnapshot> {
    let mut out = Vec::new();
    for token in rest.split_whitespace() {
        // token: rankN[cudaD,share=X,tput=Y]
        let Some(open) = token.find('[') else { continue };
        let Some(close_idx) = token.find(']') else { continue };
        let rank_str = &token[..open];
        let body = &token[open + 1..close_idx];
        let Some(rank_num) = rank_str.strip_prefix("rank") else { continue };
        let Ok(rank) = rank_num.parse::<usize>() else { continue };

        let mut device: u8 = 0;
        let mut batch_share = 0.0;
        let mut throughput = 0.0;
        for kv in body.split(',') {
            if let Some(v) = kv.strip_prefix("cuda") {
                device = v.parse().unwrap_or(0);
            } else if let Some(v) = kv.strip_prefix("share=") {
                batch_share = v.parse().unwrap_or(0.0);
            } else if let Some(v) = kv.strip_prefix("tput=") {
                throughput = v.parse().unwrap_or(0.0);
            }
        }
        out.push(RankSnapshot { rank, device, batch_share, throughput });
    }
    out
}

/// Apply training log data to a RunAnalysis, overriding timeline-derived
/// loss/eval/epoch data with the authoritative log values.
pub fn apply_training_log(analysis: &mut RunAnalysis, log: &TrainingLog) {
    if log.epochs.is_empty() {
        return;
    }

    // Override epoch data
    analysis.epoch_data = log.epochs.iter().map(|e| EpochData {
        epoch: e.epoch,
        loss: e.loss,
        eval: e.eval,
        wall_ms: e.time_ms,
    }).collect();
    analysis.n_epochs = analysis.epoch_data.len();

    // Final loss from last epoch
    analysis.final_loss = log.epochs.last().map(|e| e.loss).unwrap_or(0.0);

    // Final eval: standalone line wins, otherwise last per-epoch eval
    analysis.final_eval = log.final_eval.or_else(|| {
        log.epochs.iter().rev().find_map(|e| e.eval)
    });

    // Total time from log footer (if timeline had no samples)
    if analysis.total_ms == 0
        && let Some(ms) = log.total_ms
    {
        analysis.total_ms = ms as u64;
    }

    // Training-only time (set by run_baseline_solo via `# train_only:` line).
    analysis.train_only_ms = log.train_only_ms.map(|ms| ms as u64);

    // Per-rank averages: collect snapshots across epochs, average per rank.
    if log.epochs.iter().any(|e| !e.per_rank.is_empty()) {
        use std::collections::BTreeMap;
        let mut acc: BTreeMap<usize, (u8, f64, f64, usize)> = BTreeMap::new();
        for ep in &log.epochs {
            for snap in &ep.per_rank {
                let entry = acc.entry(snap.rank).or_insert((snap.device, 0.0, 0.0, 0));
                entry.0 = snap.device;
                entry.1 += snap.batch_share;
                entry.2 += snap.throughput;
                entry.3 += 1;
            }
        }
        analysis.per_rank_avg = acc.into_iter().map(|(rank, (device, share_sum, tput_sum, n))| {
            let n_f = n as f64;
            PerRankAvg {
                rank,
                device,
                batch_share: if n > 0 { share_sum / n_f } else { 0.0 },
                throughput: if n > 0 { tput_sum / n_f } else { 0.0 },
            }
        }).collect();
    }

}
