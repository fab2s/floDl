//! Unit tests for the sub-epoch window-report plumb: per-batch losses
//! accumulated into the window ledger must surface, at the reduce
//! boundary and only when the `reports_per_epoch` cadence fires, as a
//! path-keyed record tree pushed to the dashboard sink.
//!
//! The per-epoch feed is a separate channel and is not exercised here;
//! what matters is that the sub-epoch feed is silent by default (no
//! cadence configured = zero behavior change) and exact when enabled.

use std::sync::{Arc, Mutex};

use super::super::ClusterCoordinator;
use super::{TimingMsgWire, cfg_sync_cpu};
use crate::distributed::DashboardSink;
use crate::distributed::wire::ResourceSampleWire;

/// Capturing [`DashboardSink`] stub: records every window push.
#[derive(Default)]
struct StubSink {
    windows: Mutex<Vec<Vec<serde_json::Value>>>,
}

impl StubSink {
    /// Number of window reports pushed so far.
    fn count(&self) -> usize {
        self.windows.lock().unwrap().len()
    }
    /// Root record of the most recent window report.
    fn last_root(&self) -> serde_json::Value {
        self.windows.lock().unwrap().last().unwrap()[0].clone()
    }
    /// All records of the most recent window report.
    fn last(&self) -> Vec<serde_json::Value> {
        self.windows.lock().unwrap().last().unwrap().clone()
    }
}

impl DashboardSink for StubSink {
    fn register_port(&self, _rank: usize, _port: u16) {}
    fn set_svg(
        &self,
        _rank: usize,
        _svg: String,
        _label: Option<String>,
        _hash: Option<String>,
    ) {
    }
    fn set_metadata(&self, _rank: usize, _json: String) {}
    fn set_hardware(&self, _rank: usize, _summary: String) {}
    fn push_resource_sample(&self, _rank: usize, _sample: ResourceSampleWire) {}
    fn push_epoch_metrics(&self, _metrics: &crate::distributed::ddp_run::EpochMetrics) {}
    fn push_window_records(&self, records: Vec<serde_json::Value>) {
        self.windows.lock().unwrap().push(records);
    }
}

/// Feed one completed batch for `rank` carrying `loss`.
fn batch(coord: &mut ClusterCoordinator, rank: u64, loss: f64) {
    coord.process_timing_msg(TimingMsgWire::Batch {
        rank,
        batch_ms: 10.0,
        data_ms: 2.0,
        step_count: 0,
        param_norm: None,
        batch_loss: loss,
        sync_divergence: None,
    });
}

/// A 2-rank coord whose epoch is 10 steps (100 samples / batch 10), with
/// `reports_per_epoch` reports per epoch, wired to a capturing sink.
fn coord_with_cadence(reports_per_epoch: usize) -> (ClusterCoordinator, Arc<StubSink>) {
    let sink = Arc::new(StubSink::default());
    let coord = ClusterCoordinator::for_test(
        cfg_sync_cpu(2)
            .total_samples(100)
            .batch_size(10)
            .reports_per_epoch(reports_per_epoch)
            .dashboard_sink(sink.clone()),
    );
    (coord, sink)
}

#[test]
fn window_report_carries_work_weighted_loss() {
    // epoch_work = 10 steps, x = 2 => a report every 5 steps.
    let (mut coord, sink) = coord_with_cadence(2);
    // rank0: 4 batches @ 0.2 ; rank1: 1 batch @ 0.7  => 5 steps, hits the
    // first threshold exactly.
    for _ in 0..4 {
        batch(&mut coord, 0, 0.2);
    }
    batch(&mut coord, 1, 0.7);

    coord.finish_averaging_head();

    assert_eq!(sink.count(), 1, "cadence crossed => exactly one report");
    let root = sink.last_root();
    assert_eq!(root["kind"], "node");
    assert_eq!(root["path"], "root");
    assert_eq!(root["epoch"], 0);
    // Work-weighted: (0.2*4 + 0.7*1) / 5 = 0.3 — NOT the flat mean 0.45.
    let loss = root["metrics"]["loss"].as_f64().unwrap();
    assert!((loss - 0.3).abs() < 1e-9, "got {loss}");
    // Work = total steps this window.
    assert_eq!(root["work"].as_f64().unwrap(), 5.0);
    // batch_share sums back to ~1 across ranks.
    let share = root["metrics"]["batch_share"].as_f64().unwrap();
    assert!((share - 1.0).abs() < 1e-9, "got {share}");

    // One record per node: root + 2 ranks (single host => no host tier).
    let recs = sink.last();
    assert_eq!(recs.len(), 3);
    let paths: Vec<&str> = recs.iter().map(|r| r["path"].as_str().unwrap()).collect();
    assert!(paths.contains(&"root/rank0"));
    assert!(paths.contains(&"root/rank1"));
    // The leaf keeps its own raw loss, unweighted.
    let r1 = recs.iter().find(|r| r["path"] == "root/rank1").unwrap();
    assert!((r1["metrics"]["loss"].as_f64().unwrap() - 0.7).abs() < 1e-9);
}

#[test]
fn no_cadence_configured_emits_nothing() {
    // The default: sub-epoch reporting off => the hot path is untouched.
    let sink = Arc::new(StubSink::default());
    let mut coord = ClusterCoordinator::for_test(
        cfg_sync_cpu(2)
            .total_samples(100)
            .batch_size(10)
            .dashboard_sink(sink.clone()),
    );
    for _ in 0..8 {
        batch(&mut coord, 0, 0.5);
    }
    coord.finish_averaging_head();
    assert_eq!(sink.count(), 0, "no reports_per_epoch => no window feed");
}

#[test]
fn cadence_throttles_windows_below_the_threshold() {
    // epoch_work = 10, x = 2 => threshold every 5 steps. A 2-step window
    // must NOT report; the cadence is a work fraction, not per-reduce.
    let (mut coord, sink) = coord_with_cadence(2);
    batch(&mut coord, 0, 0.4);
    batch(&mut coord, 1, 0.4);
    coord.finish_averaging_head();
    assert_eq!(sink.count(), 0, "2 steps < 5-step threshold");

    // Next window carries 3 more steps => cumulative 5, crossing it.
    coord.reset_window_for_test();
    for _ in 0..3 {
        batch(&mut coord, 0, 0.4);
    }
    coord.finish_averaging_head();
    assert_eq!(sink.count(), 1, "cumulative work crossed the threshold");
}

#[test]
fn report_is_capped_per_epoch() {
    // x = 1 => exactly one report per epoch no matter how many reduces.
    let (mut coord, sink) = coord_with_cadence(1);
    for _ in 0..12 {
        batch(&mut coord, 0, 0.5);
    }
    coord.finish_averaging_head();
    assert_eq!(sink.count(), 1);
    // A second window in the same epoch stays silent (budget spent).
    coord.reset_window_for_test();
    for _ in 0..12 {
        batch(&mut coord, 0, 0.5);
    }
    coord.finish_averaging_head();
    assert_eq!(sink.count(), 1, "per-epoch budget is x, not x per reduce");
}

#[test]
fn silent_rank_is_absent_from_the_report_not_zero() {
    // Only rank0 trains this window; rank1 reported nothing. The cohort
    // loss must be rank0's, not halved by a phantom zero for rank1.
    let (mut coord, sink) = coord_with_cadence(2);
    for _ in 0..5 {
        batch(&mut coord, 0, 0.8);
    }
    coord.finish_averaging_head();

    let root = sink.last_root();
    assert!((root["metrics"]["loss"].as_f64().unwrap() - 0.8).abs() < 1e-9);
    let recs = sink.last();
    let r1 = recs.iter().find(|r| r["path"] == "root/rank1").unwrap();
    assert!(
        r1["metrics"].get("loss").is_none(),
        "unmeasured rank carries no loss key: {r1}",
    );
    assert_eq!(r1["work"].as_f64().unwrap(), 0.0);
}

#[test]
fn multi_host_report_gets_a_host_tier() {
    let sink = Arc::new(StubSink::default());
    let mut coord = ClusterCoordinator::for_test(
        cfg_sync_cpu(2)
            .total_samples(100)
            .batch_size(10)
            .reports_per_epoch(2)
            .rank_hosts(vec!["exa".to_string(), "flodl-pascal".to_string()])
            .dashboard_sink(sink.clone()),
    );
    for _ in 0..3 {
        batch(&mut coord, 0, 0.2);
    }
    for _ in 0..2 {
        batch(&mut coord, 1, 0.7);
    }
    coord.finish_averaging_head();

    let recs = sink.last();
    let paths: Vec<&str> = recs.iter().map(|r| r["path"].as_str().unwrap()).collect();
    assert!(paths.contains(&"root/exa"), "{paths:?}");
    assert!(paths.contains(&"root/exa/rank0"), "{paths:?}");
    assert!(paths.contains(&"root/flodl-pascal/rank1"), "{paths:?}");
    // Hierarchical == flat: (0.2*3 + 0.7*2)/5 = 0.4
    let root = sink.last_root();
    assert!((root["metrics"]["loss"].as_f64().unwrap() - 0.4).abs() < 1e-9);
}
