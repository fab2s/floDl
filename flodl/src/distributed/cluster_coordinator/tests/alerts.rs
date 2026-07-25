//! Unit tests for the alert lane's coordinator seam: the three
//! alerting-class conditions must surface as `kind: "event"` records on the
//! record stream, at the origin path the node tree uses, collapsed so a
//! sustained fault cannot flood, and with the tail flushed at teardown.
//!
//! The lane's own collapse / cap arithmetic is covered in
//! `monitor::event_lane`; what is proved here is the wiring — that each real
//! producer reaches it, and that the paths line up with the node tree.

use std::sync::Arc;

use crate::distributed::controller::DeadRanks;
use crate::distributed::ddp_run::convergence::{
    ConvergenceAction, ConvergenceGuard, DivergenceReport,
};

use super::super::{ClusterCoordinator, ClusterCoordinatorConfig};
use super::{ControlMsgWire, StubSink, cfg_sync_cpu};

/// Guard that always demands a correction — the `drift` trigger.
#[derive(Clone)]
struct AlwaysNudge;

impl ConvergenceGuard for AlwaysNudge {
    fn report(&mut self, _: &DivergenceReport, _: usize, _: usize) -> ConvergenceAction {
        ConvergenceAction::NudgeDown { factor: 0.5 }
    }
    fn clone_box(&self) -> Box<dyn ConvergenceGuard> {
        Box::new(self.clone())
    }
}

/// A 3-rank coord with elastic membership, a capturing sink, and an
/// already-expired heartbeat window so `check_dead_ranks` fires on demand.
fn coord_with_deaths(hosts: Option<Vec<String>>) -> (ClusterCoordinator, Arc<StubSink>) {
    let sink = Arc::new(StubSink::default());
    let mut cfg = cfg_sync_cpu(3)
        .dead_ranks(DeadRanks::new(3))
        .heartbeat_timeout_secs(0)
        .dashboard_sink(sink.clone());
    if let Some(h) = hosts {
        cfg = cfg.rank_hosts(h);
    }
    (ClusterCoordinator::for_test(cfg), sink)
}

#[test]
fn rank_death_raises_a_critical_rank_lost_alert() {
    let (mut coord, sink) = coord_with_deaths(None);
    // Every rank's heartbeat is stale under a 0s window, so one sweep
    // declares all three dead — three distinct paths, three alerts.
    coord.check_dead_ranks_for_test();

    let lost = sink.events_of("rank_lost");
    assert_eq!(lost.len(), 3, "{:?}", sink.events());
    for (i, e) in lost.iter().enumerate() {
        assert_eq!(e["kind"], "event");
        assert_eq!(e["sev"], "critical");
        assert_eq!(e["count"], 1);
        // Single-host (unknown host) cohort => the flat node path.
        assert_eq!(e["path"], format!("root/rank{i}"));
        assert!(
            e["detail"].as_str().unwrap().contains("heartbeat stale"),
            "detector reason rides the alert: {e}",
        );
    }
}

#[test]
fn rank_lost_path_carries_the_host_tier_on_a_multi_host_cohort() {
    let (mut coord, sink) = coord_with_deaths(Some(vec![
        "exa".to_string(),
        "flodl-pascal".to_string(),
        "flodl-pascal".to_string(),
    ]));
    coord.check_dead_ranks_for_test();

    let paths: Vec<String> = sink
        .events_of("rank_lost")
        .iter()
        .map(|e| e["path"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        paths,
        vec![
            "root/exa/rank0",
            "root/flodl-pascal/rank1",
            "root/flodl-pascal/rank2",
        ],
    );
}

#[test]
fn dropped_control_broadcast_raises_control_drop() {
    let sink = Arc::new(StubSink::default());
    let mut coord =
        ClusterCoordinator::for_test(cfg_sync_cpu(2).dashboard_sink(sink.clone()));

    // `for_test` leaves `control_streams` empty, so the broadcast misses
    // every live rank.
    let _ = coord.broadcast_control(&ControlMsgWire::SyncNow);

    let drops = sink.events_of("control_drop");
    assert_eq!(drops.len(), 1);
    assert_eq!(drops[0]["sev"], "critical");
    assert_eq!(drops[0]["path"], "root", "a lost broadcast is cohort-level");
    assert!(drops[0]["detail"].as_str().unwrap().contains("SyncNow"));
}

#[test]
fn repeated_control_drops_collapse_into_one_record() {
    let sink = Arc::new(StubSink::default());
    let mut coord =
        ClusterCoordinator::for_test(cfg_sync_cpu(2).dashboard_sink(sink.clone()));

    for _ in 0..25 {
        let _ = coord.broadcast_control(&ControlMsgWire::SyncNow);
    }
    // Every drop is counted...
    assert_eq!(coord.lost_broadcasts, 25);
    // ...but the alert lane emits once for the open collapse window.
    assert_eq!(sink.events_of("control_drop").len(), 1);

    // Teardown closes the window, so the 24 absorbed repeats reach the
    // persisted stream rather than dying with the live feed.
    coord.shutdown().expect("headless shutdown");
    let drops = sink.events_of("control_drop");
    assert_eq!(drops.len(), 2, "{drops:?}");
    assert_eq!(drops[1]["count"], 24);
}

#[test]
fn exempt_shutdown_broadcast_raises_no_alert() {
    let sink = Arc::new(StubSink::default());
    let mut coord =
        ClusterCoordinator::for_test(cfg_sync_cpu(2).dashboard_sink(sink.clone()));
    let _ = coord.broadcast_control(&ControlMsgWire::Shutdown);
    assert!(
        sink.events().is_empty(),
        "a failed Shutdown is a teardown race, not an alert: {:?}",
        sink.events(),
    );
}

#[test]
fn guard_correction_raises_a_drift_warning() {
    let sink = Arc::new(StubSink::default());
    let mut coord = ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            crate::distributed::ddp_run::ApplyPolicy::Sync,
            crate::distributed::ddp_run::AverageBackend::Cpu,
            2,
            crate::distributed::ddp::ElChe::new(2, 1),
        )
        .with_convergence_guard(Box::new(AlwaysNudge))
        .dashboard_sink(sink.clone()),
    );

    coord.finish_averaging_head();

    let drift = sink.events_of("drift");
    assert_eq!(drift.len(), 1);
    assert_eq!(drift[0]["sev"], "warn");
    assert_eq!(drift[0]["path"], "root");
    assert!(
        drift[0]["detail"].as_str().unwrap().contains("nudged the anchor down"),
        "{}",
        drift[0],
    );

    // A sustained bad regime nudges every cycle; the lane keeps that to one
    // record per collapse window.
    for _ in 0..5 {
        coord.finish_averaging_head();
    }
    assert_eq!(sink.events_of("drift").len(), 1);
}

#[test]
fn a_healthy_run_emits_no_alerts() {
    // The default guard path with no deaths and no failed broadcasts: the
    // lane must stay silent, so an alert in a log always means something.
    let sink = Arc::new(StubSink::default());
    let mut coord =
        ClusterCoordinator::for_test(cfg_sync_cpu(2).dashboard_sink(sink.clone()));
    coord.finish_averaging_head();
    coord.finish_averaging_head();
    assert!(sink.events().is_empty(), "{:?}", sink.events());
}
