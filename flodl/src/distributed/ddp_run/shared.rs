//! Partition math and epoch-metrics aggregation shared by the single-host
//! fallback (`orchestrator::single_host`) and the process-per-rank
//! cluster coordinator (`cluster_coordinator`).
//!
//! These are pure functions with no coordinator state. They previously lived
//! in the in-process `coordinator` module; they outlived it because both the
//! single-GPU path and the cluster path still need consistent partition sizing
//! and per-rank metric aggregation.

use super::{EpochMetrics, MetricsMsg};

/// The pick space: `dataset.len() * augment`, the schedule space every
/// partition offset and size lives in. A pick decodes as
/// `(pick / augment, pick % augment)` = (sample, repeat).
///
/// One derivation, because agreement here is a correctness property
/// rather than a convention. The coordinator's ledger, each rank's
/// epoch expansion, the resume coverage map and the stager's read-ahead
/// all index the same permutation of this range: a consumer that sizes
/// it differently silently trains on a different shuffle, with no error
/// anywhere to catch it.
pub(crate) fn pick_space(dataset_len: usize, augment: usize) -> usize {
    dataset_len * augment.max(1)
}

/// The `window <= epoch` ceiling, in batches, or `None` when an epoch is
/// too small to bound (the caller then leaves the window uncapped).
///
/// A reduce window must never span more than one epoch: left unbounded, an
/// ElChe schedule grows the window past the epoch and syncs collapse to
/// under one per pass, serializing the cohort. Under epoch splits the
/// ceiling follows the slice, and the floor keeps it valid for every event
/// including the longer ones carrying the balanced remainder.
pub(crate) fn window_cap_batches(
    total_samples: usize,
    epoch_splits: usize,
    batch_size: usize,
) -> Option<usize> {
    if batch_size == 0 {
        return None;
    }
    let epoch_samples = total_samples / epoch_splits.max(1);
    (epoch_samples >= batch_size).then(|| epoch_samples / batch_size)
}

/// Validate and describe the epoch geometry before a run starts.
///
/// One home for both, called from every tier's entry point, because both
/// answers depend on the same three numbers and neither is specific to a
/// dispatch path. Refusing here beats refusing at dispatch: the failure it
/// prevents is silent at runtime, not loud.
///
/// `total_samples` is the pick space, `epoch_splits` the slices per data
/// pass, `num_epochs` the number of passes.
pub(crate) fn check_epoch_geometry(
    total_samples: usize,
    batch_size: usize,
    epoch_splits: usize,
    num_epochs: usize,
) -> crate::tensor::Result<()> {
    let splits = epoch_splits.max(1);
    if splits == 1 || batch_size == 0 || total_samples == 0 {
        return Ok(());
    }
    let epoch_samples = total_samples / splits;
    if epoch_samples < batch_size {
        return Err(crate::tensor::TensorError::new(&format!(
            "epoch_splits {splits} slices {total_samples} picks into epochs of \
             {epoch_samples} picks, short of one {batch_size}-pick batch: those \
             epochs would train nothing, and the `window <= epoch` reduce cap has \
             no epoch large enough to bound the window. Use at most {} splits, or \
             a larger dataset.",
            (total_samples / batch_size).max(1),
        )));
    }

    let multiple = splits * batch_size;
    let divides = total_samples % multiple == 0;
    eprintln!(
        "  ddp: pass = {splits} epochs x {} batches x {batch_size} of {total_samples} picks ({})",
        epoch_samples / batch_size,
        if divides {
            "exact".to_string()
        } else {
            format!("up to {} picks/pass outside the last batch", splits * (batch_size - 1))
        },
    );
    // A tail is re-drawn every pass, so multi-pass runs average it away
    // (200 passes miss a given sample ~0.3 times). One pass averages with
    // nothing, so say so rather than let it pass as rounding.
    if !divides && num_epochs <= 1 {
        eprintln!(
            "  ddp: WARNING single-pass run, so that tail is never re-drawn. Size the \
             dataset to a multiple of {multiple} to use all of it."
        );
    }
    Ok(())
}

/// Render an epoch index for a human reader.
///
/// Bare when unsplit, so existing logs are untouched; `3 (pass 0, split
/// 4/20)` when an epoch is a slice, because the bare index alone stops
/// answering "how far through the data am I". Internals keep counting
/// epochs — only what a person reads gains the extra vocabulary.
pub(crate) fn epoch_label(epoch: usize, epoch_splits: usize) -> String {
    if epoch_splits <= 1 {
        return epoch.to_string();
    }
    format!(
        "{epoch} (pass {}, split {}/{epoch_splits})",
        epoch / epoch_splits,
        epoch % epoch_splits + 1,
    )
}

/// Equal partition sizes with remainder distributed to the first ranks.
pub(crate) fn equal_sizes(world_size: usize, total: usize) -> Vec<usize> {
    let base = total / world_size;
    let remainder = total % world_size;
    (0..world_size)
        .map(|r| base + if r < remainder { 1 } else { 0 })
        .collect()
}

/// Throughput-proportional partition sizes from ElChe ms_per_batch.
///
/// Faster ranks (lower ms/batch) get more samples. Remainder distributed
/// to the fastest ranks.
pub(crate) fn throughput_sizes(
    el_che: &crate::distributed::ddp::ElChe,
    total: usize,
) -> Vec<usize> {
    let ms = el_che.ms_per_batch();
    // Inverse of ms_per_batch = throughput (batches/ms). Guard against zero.
    let throughputs: Vec<f64> = ms.iter().map(|&m| 1.0 / m.max(0.001)).collect();
    let total_tp: f64 = throughputs.iter().sum();
    if total_tp <= 0.0 {
        return equal_sizes(ms.len(), total);
    }
    let mut sizes: Vec<usize> = throughputs.iter()
        .map(|t| ((t / total_tp) * total as f64).floor() as usize)
        .collect();
    // Distribute remainder to fastest ranks (highest throughput first).
    let assigned: usize = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);
    if remaining > 0 {
        // Sort rank indices by throughput descending.
        let mut rank_order: Vec<usize> = (0..ms.len()).collect();
        rank_order.sort_by(|&a, &b| throughputs[b].partial_cmp(&throughputs[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &rank in &rank_order {
            if remaining == 0 { break; }
            sizes[rank] += 1;
            remaining -= 1;
        }
    }
    sizes
}

/// Convert user-specified ratios to absolute partition sizes.
///
/// Ratios are normalized to sum to 1.0. Remainder distributed to the
/// ranks with the largest ratios.
pub(crate) fn ratio_to_sizes(ratios: &[f64], total: usize) -> Vec<usize> {
    let sum: f64 = ratios.iter().sum();
    let norm: Vec<f64> = if sum > 0.0 {
        ratios.iter().map(|r| r / sum).collect()
    } else {
        vec![1.0 / ratios.len() as f64; ratios.len()]
    };
    let mut sizes: Vec<usize> = norm.iter()
        .map(|r| (r * total as f64).floor() as usize)
        .collect();
    let assigned: usize = sizes.iter().sum();
    let mut remaining = total.saturating_sub(assigned);
    if remaining > 0 {
        let mut rank_order: Vec<usize> = (0..ratios.len()).collect();
        rank_order.sort_by(|&a, &b| norm[b].partial_cmp(&norm[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &rank in &rank_order {
            if remaining == 0 { break; }
            sizes[rank] += 1;
            remaining -= 1;
        }
    }
    sizes
}

/// Aggregate per-rank `MetricsMsg` into a single `EpochMetrics`.
///
/// Loss and scalars are averaged weighted by batch count (proportional
/// to each rank's contribution). Epoch time is the max across ranks.
///
/// In progressive dispatch mode, each rank sends one [`MetricsMsg`] per
/// chunk (not per epoch), so there may be many more messages than ranks.
/// This function aggregates by rank first so the output always has
/// exactly `world_size` entries per vector.
/// `bc_share` is the smoothed per-rank batch-share view from the balancer
/// (i.e. `el_che.recent_batch_share()` averaged over the last few sync
/// snapshots). It replaces the prior "samples-consumed / total-samples"
/// share, which conflated cadence allocation with progressive dispatch
/// tail-balance dynamics. Caller is expected to pass `world_size` entries
/// summing to ~1.0; degenerate input falls back to equal shares.
pub(crate) fn aggregate_epoch_metrics(
    epoch: usize,
    msgs: &[MetricsMsg],
    device_indices: &[u8],
    bc_share: &[f64],
) -> EpochMetrics {
    let world_size = device_indices.len();

    // --- Step 1: Aggregate per-chunk messages by rank ---
    let mut rank_batches: Vec<usize> = vec![0; world_size];
    let mut rank_samples: Vec<usize> = vec![0; world_size];
    let mut rank_loss_sum: Vec<f64> = vec![0.0; world_size];
    let mut rank_time_ms: Vec<f64> = vec![0.0; world_size];
    let mut rank_share_complete_ms: Vec<f64> = vec![0.0; world_size];
    let mut rank_compute_only_ms: Vec<f64> = vec![0.0; world_size];
    let mut rank_data_starve_ms: Vec<f64> = vec![0.0; world_size];
    // Per-rank scalar accumulators: (sum, count) per key
    let mut rank_scalars: Vec<std::collections::HashMap<String, (f64, usize)>> =
        (0..world_size).map(|_| std::collections::HashMap::new()).collect();

    for m in msgs {
        let r = m.rank.min(world_size - 1);
        rank_batches[r] += m.batches_processed;
        rank_samples[r] += m.samples_processed;
        rank_loss_sum[r] += m.avg_loss * m.batches_processed as f64;
        // Sum across chunks (sequential within a rank). Each message's
        // durations cover ONE chunk — the worker's `EpochState` is fresh
        // per `EpochPlan` — so the rank's epoch total is the sum. The old
        // `max()` fold assumed cumulative-from-epoch-start values and
        // reported only the largest chunk in progressive dispatch (and
        // divided epoch-summed samples by one chunk's time in the
        // throughput below). Between-chunk gaps are not covered by any
        // chunk's wall; epoch_ms is therefore a lower bound in progressive
        // mode, tight because dispatch re-plans back-to-back.
        rank_time_ms[r] += m.epoch_ms;
        rank_share_complete_ms[r] += m.share_complete_ms;
        rank_compute_only_ms[r] += m.compute_only_ms;
        rank_data_starve_ms[r] += m.data_starve_ms;
        for (k, (sum, count)) in &m.scalars {
            let entry = rank_scalars[r].entry(k.clone()).or_insert((0.0, 0));
            entry.0 += sum;
            entry.1 += count;
        }
    }

    // --- Step 2: Compute aggregated metrics ---
    let total_batches: usize = rank_batches.iter().sum();

    // Batch-weighted average loss
    let avg_loss = if total_batches > 0 {
        rank_loss_sum.iter().sum::<f64>() / total_batches as f64
    } else {
        0.0
    };

    // Per-rank batch-weighted mean loss. A rank with no batches this epoch
    // yields None (absent, not zero) so downstream means exclude it.
    let per_rank_loss: Vec<Option<f64>> = (0..world_size)
        .map(|r| {
            if rank_batches[r] > 0 {
                Some(rank_loss_sum[r] / rank_batches[r] as f64)
            } else {
                None
            }
        })
        .collect();

    // Max epoch_ms across ranks
    let epoch_ms = rank_time_ms.iter().copied().fold(0.0_f64, f64::max);

    // Per-rank scalar means (each rank's sum/count)
    let per_rank: Vec<std::collections::HashMap<String, f64>> = rank_scalars
        .iter()
        .map(|scalars| {
            scalars
                .iter()
                .map(|(k, (sum, count))| {
                    (k.clone(), if *count > 0 { sum / *count as f64 } else { 0.0 })
                })
                .collect()
        })
        .collect();

    // Weighted-average scalars across ranks
    let mut scalars: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut weights: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (r, rank_sc) in rank_scalars.iter().enumerate() {
        let w = rank_batches[r] as f64;
        for (k, (sum, count)) in rank_sc {
            if *count > 0 {
                let mean = sum / *count as f64;
                *scalars.entry(k.clone()).or_default() += mean * w;
                *weights.entry(k.clone()).or_default() += w;
            }
        }
    }
    for (k, v) in &mut scalars {
        if let Some(w) = weights.get(k) {
            if *w > 0.0 {
                *v /= *w;
            }
        }
    }

    // Per-rank throughput (samples/ms) and batch share.
    //
    // Throughput uses share_complete_ms as the denominator, NOT epoch_ms.
    // epoch_ms includes any post-completion idle the fast rank spends
    // waiting at the sync barrier for slower ranks; dividing by it produces
    // an inverted tput signal (the fast rank looks slow because it idles
    // more), which feeds the balancer a signal that says "give the fast
    // rank less work" — exactly backwards. share_complete_ms = compute +
    // data-pipeline wait, measured per rank, excludes peer-induced idle,
    // so the resulting tput tracks the rank's actual capacity.
    //
    // Falls back to epoch_ms when share_complete_ms wasn't populated (legacy
    // call sites or test fixtures using the old MetricsMsg shape).
    let per_rank_throughput: Vec<f64> = (0..world_size).map(|r| {
        let denom = if rank_share_complete_ms[r] > 0.0 {
            rank_share_complete_ms[r]
        } else {
            rank_time_ms[r]
        };
        if denom > 0.0 { rank_samples[r] as f64 / denom } else { 0.0 }
    }).collect();
    // Per-rank batch share comes from the balancer's smoothed view of its
    // own recent batch_counts allocation (`el_che.recent_batch_share()`),
    // not from samples consumed. Under progressive dispatch the latter is
    // equalized by tail-balance and obscures the cadence's actual ratios.
    // Degenerate input (wrong length, all zeros) falls back to equal shares.
    let per_rank_batch_share: Vec<f64> = if bc_share.len() == world_size {
        let sum: f64 = bc_share.iter().sum();
        if sum > 0.0 {
            bc_share.to_vec()
        } else {
            vec![1.0 / world_size as f64; world_size]
        }
    } else {
        vec![1.0 / world_size as f64; world_size]
    };

    EpochMetrics {
        epoch, scalars, per_rank, avg_loss, epoch_ms,
        per_rank_loss,
        per_rank_samples: rank_samples,
        per_rank_throughput, per_rank_batch_share,
        per_rank_share_complete_ms: rank_share_complete_ms,
        per_rank_compute_only_ms: rank_compute_only_ms,
        per_rank_data_starve_ms: rank_data_starve_ms,
        device_indices: device_indices.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::{aggregate_epoch_metrics, pick_space, window_cap_batches};
    use super::super::MetricsMsg;
    use std::collections::HashMap;

    #[test]
    fn test_aggregate_epoch_metrics() {
        let mut scalars_r0 = HashMap::new();
        scalars_r0.insert("loss".to_string(), (3.0, 3_usize)); // mean = 1.0
        scalars_r0.insert("acc".to_string(), (1.8, 3));         // mean = 0.6

        let mut scalars_r1 = HashMap::new();
        scalars_r1.insert("loss".to_string(), (4.0, 2_usize)); // mean = 2.0
        scalars_r1.insert("acc".to_string(), (0.8, 2));         // mean = 0.4

        let msgs = vec![
            MetricsMsg {
                rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 60,
                epoch_ms: 1000.0, share_complete_ms: 1000.0, compute_only_ms: 1000.0, data_starve_ms: 0.0, samples_processed: 1920, scalars: scalars_r0,
            },
            MetricsMsg {
                rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 40,
                epoch_ms: 1200.0, share_complete_ms: 1200.0, compute_only_ms: 1200.0, data_starve_ms: 0.0, samples_processed: 1280, scalars: scalars_r1,
            },
        ];

        let dev_indices = vec![0_u8, 1];
        // bc_share now comes from the balancer (smoothed batch_counts). Pass
        // 60/40 explicitly to match the historical samples-driven assertion.
        let bc_share = vec![0.6_f64, 0.4];
        let m = aggregate_epoch_metrics(0, &msgs, &dev_indices, &bc_share);
        assert_eq!(m.epoch, 0);

        // Batch-weighted average loss: (0.5*60 + 0.7*40) / 100 = 0.58
        assert!((m.avg_loss - 0.58).abs() < 1e-9);

        // Max epoch_ms
        assert_eq!(m.epoch_ms, 1200.0);

        // Weighted scalar: loss = (1.0*60 + 2.0*40) / 100 = 1.4
        assert!((m.scalars["loss"] - 1.4).abs() < 1e-9);

        // Weighted scalar: acc = (0.6*60 + 0.4*40) / 100 = 0.52
        assert!((m.scalars["acc"] - 0.52).abs() < 1e-9);

        // Per-rank
        assert_eq!(m.per_rank.len(), 2);
        assert!((m.per_rank[0]["loss"] - 1.0).abs() < 1e-9);
        assert!((m.per_rank[1]["loss"] - 2.0).abs() < 1e-9);

        // Per-rank loss (batch-weighted within the rank) and realized samples.
        assert!((m.per_rank_loss[0].unwrap() - 0.5).abs() < 1e-9);
        assert!((m.per_rank_loss[1].unwrap() - 0.7).abs() < 1e-9);
        assert_eq!(m.per_rank_samples, vec![1920, 1280]);

        // Throughput: rank 0 = 1920/1000 = 1.92, rank 1 = 1280/1200 ~= 1.0667
        assert!((m.per_rank_throughput[0] - 1.92).abs() < 1e-9);
        assert!((m.per_rank_throughput[1] - 1280.0 / 1200.0).abs() < 1e-9);

        // Batch share: rank 0 = 0.6, rank 1 = 0.4
        assert!((m.per_rank_batch_share[0] - 0.6).abs() < 1e-9);
        assert!((m.per_rank_batch_share[1] - 0.4).abs() < 1e-9);

        // Device indices
        assert_eq!(m.device_indices, vec![0, 1]);
    }

    /// Progressive dispatch: multiple MetricsMsg per rank should be aggregated
    /// into exactly world_size entries, not one entry per message.
    #[test]
    fn test_aggregate_epoch_metrics_progressive() {
        // Simulate 2 ranks, 3 chunks from rank 0, 2 chunks from rank 1
        let msgs = vec![
            // Rank 0 chunk 1
            MetricsMsg {
                rank: 0, epoch: 0, avg_loss: 0.5, batches_processed: 20,
                epoch_ms: 300.0, share_complete_ms: 300.0, compute_only_ms: 300.0, data_starve_ms: 0.0, samples_processed: 640,
                scalars: [("loss".to_string(), (2.0, 2_usize))].into(),
            },
            // Rank 0 chunk 2 (durations are per-chunk, NOT cumulative)
            MetricsMsg {
                rank: 0, epoch: 0, avg_loss: 0.4, batches_processed: 20,
                epoch_ms: 300.0, share_complete_ms: 300.0, compute_only_ms: 300.0, data_starve_ms: 0.0, samples_processed: 640,
                scalars: [("loss".to_string(), (1.6, 2_usize))].into(),
            },
            // Rank 0 chunk 3
            MetricsMsg {
                rank: 0, epoch: 0, avg_loss: 0.6, batches_processed: 20,
                epoch_ms: 300.0, share_complete_ms: 300.0, compute_only_ms: 300.0, data_starve_ms: 0.0, samples_processed: 640,
                scalars: [("loss".to_string(), (1.8, 2_usize))].into(),
            },
            // Rank 1 chunk 1
            MetricsMsg {
                rank: 1, epoch: 0, avg_loss: 0.7, batches_processed: 20,
                epoch_ms: 500.0, share_complete_ms: 500.0, compute_only_ms: 500.0, data_starve_ms: 0.0, samples_processed: 640,
                scalars: [("loss".to_string(), (2.8, 2_usize))].into(),
            },
            // Rank 1 chunk 2
            MetricsMsg {
                rank: 1, epoch: 0, avg_loss: 0.8, batches_processed: 20,
                epoch_ms: 500.0, share_complete_ms: 500.0, compute_only_ms: 500.0, data_starve_ms: 0.0, samples_processed: 640,
                scalars: [("loss".to_string(), (3.2, 2_usize))].into(),
            },
        ];

        let dev_indices = vec![0_u8, 1];
        let bc_share = vec![0.6_f64, 0.4];
        let m = aggregate_epoch_metrics(0, &msgs, &dev_indices, &bc_share);

        // Must have exactly 2 entries (world_size), not 5 (one per msg)
        assert_eq!(m.per_rank_throughput.len(), 2, "should have world_size entries");
        assert_eq!(m.per_rank_batch_share.len(), 2);
        assert_eq!(m.per_rank.len(), 2);
        assert_eq!(m.device_indices, vec![0, 1]);

        // Rank 0: 60 batches, 1920 samples, summed chunk time 3×300 = 900ms
        // Rank 1: 40 batches, 1280 samples, summed chunk time 2×500 = 1000ms
        // (the old max() fold would have divided epoch samples by ONE chunk)
        assert!((m.per_rank_throughput[0] - 1920.0 / 900.0).abs() < 1e-6);
        assert!((m.per_rank_throughput[1] - 1280.0 / 1000.0).abs() < 1e-6);

        // Total samples = 3200
        assert!((m.per_rank_batch_share[0] - 0.6).abs() < 1e-9);
        assert!((m.per_rank_batch_share[1] - 0.4).abs() < 1e-9);

        // epoch_ms: per-rank chunk sums, then max across ranks
        assert_eq!(m.epoch_ms, 1000.0);

        // Scalars: rank 0 loss mean = (2.0+1.6+1.8)/(2+2+2) = 5.4/6 = 0.9
        assert!((m.per_rank[0]["loss"] - 0.9).abs() < 1e-9);
        // Rank 1 loss mean = (2.8+3.2)/(2+2) = 6.0/4 = 1.5
        assert!((m.per_rank[1]["loss"] - 1.5).abs() < 1e-9);

        // Weighted average: (0.9*60 + 1.5*40)/100 = (54+60)/100 = 1.14
        assert!((m.scalars["loss"] - 1.14).abs() < 1e-9);
    }

    #[test]
    fn pick_space_treats_zero_augment_as_one() {
        assert_eq!(pick_space(50, 1), 50);
        assert_eq!(pick_space(50, 3), 150);
        assert_eq!(pick_space(50, 0), 50);
    }

    #[test]
    fn epoch_label_stays_bare_when_unsplit() {
        // Existing logs must not gain noise for runs that never split.
        assert_eq!(super::epoch_label(7, 1), "7");
        assert_eq!(super::epoch_label(7, 0), "7");
    }

    #[test]
    fn epoch_label_locates_a_split_epoch_in_its_pass() {
        // Epoch 0..3 are pass 0, epoch 4 starts pass 1. Splits read
        // 1-based because "split 1/4" is the first one to a human.
        assert_eq!(super::epoch_label(0, 4), "0 (pass 0, split 1/4)");
        assert_eq!(super::epoch_label(3, 4), "3 (pass 0, split 4/4)");
        assert_eq!(super::epoch_label(4, 4), "4 (pass 1, split 1/4)");
    }

    #[test]
    fn window_cap_follows_the_split() {
        // Unsplit: the bound is the whole pass, as it always was.
        assert_eq!(window_cap_batches(2048, 1, 4), Some(512));
        // Split four ways: a window cannot outgrow one event's slice.
        assert_eq!(window_cap_batches(2048, 4, 4), Some(128));
        // A 0 reads as unsplit rather than dividing by zero.
        assert_eq!(window_cap_batches(2048, 0, 4), Some(512));
    }

    #[test]
    fn window_cap_floors_so_it_holds_for_the_longer_events() {
        // 100 over 7 events: the balanced remainder makes two events 15
        // picks and five 14. The cap must be sized on 14, not 15.
        assert_eq!(window_cap_batches(100, 7, 2), Some(7));
    }

    #[test]
    fn a_starved_epoch_is_refused_before_the_run_starts() {
        // 100 picks, batch 4, 50 splits => 2 picks per epoch. Those epochs
        // train nothing, and window_cap_batches declines so the reduce
        // window grows unbounded. Both are silent at runtime.
        let err = super::check_epoch_geometry(100, 4, 50, 1)
            .expect_err("a starved epoch must be refused, not scheduled");
        let msg = err.to_string();
        assert!(msg.contains("epoch_splits 50"), "unexpected: {msg}");
        assert!(msg.contains("train nothing"), "unexpected: {msg}");
        assert!(msg.contains("at most 25"), "must name a usable bound: {msg}");
    }

    #[test]
    fn geometry_that_fits_is_accepted() {
        // Exactly one batch per epoch is the boundary case, and it fits.
        super::check_epoch_geometry(100, 4, 25, 1).expect("25 splits of 100 at batch 4 fits");
        super::check_epoch_geometry(78_160, 4, 20, 1).expect("the snapped olmo corpus fits");
    }

    #[test]
    fn unsplit_runs_are_never_refused_or_narrated() {
        // splits == 1 is the historical shape: no new failure mode, and
        // no new line in anyone's logs.
        super::check_epoch_geometry(3, 4, 1, 1).expect("unsplit tiny dataset stays accepted");
        super::check_epoch_geometry(0, 4, 1, 1).expect("unsized dataset stays accepted");
        super::check_epoch_geometry(100, 0, 8, 1).expect("unknown batch size cannot be judged");
    }

    #[test]
    fn window_cap_declines_when_an_epoch_cannot_fill_a_batch() {
        assert_eq!(window_cap_batches(3, 1, 4), None);
        // Splitting too finely starves an epoch below one batch even
        // though the pass is ample — the cap declines rather than
        // pinning the window to zero.
        assert_eq!(window_cap_batches(100, 50, 4), None);
        assert_eq!(window_cap_batches(100, 1, 0), None);
    }
}
