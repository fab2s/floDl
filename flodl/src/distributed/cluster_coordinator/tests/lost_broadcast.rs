//! Unit tests for the dropped best-effort broadcast trace: a failed
//! control broadcast to a live rank bumps `lost_broadcasts` and emits a
//! `LostBroadcast` timeline event, while a failed `Shutdown` (an expected
//! teardown race, not lost live coordination) is exempt — as is anything
//! sent once shutdown has already been broadcast.

use crate::distributed::ddp::ElChe;
use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
use crate::monitor::{EventKind, Timeline};

use super::super::{ClusterCoordinator, ClusterCoordinatorConfig, RunPhase};
use super::ControlMsgWire;

fn coord_with_timeline(
    world_size: usize,
    tl: std::sync::Arc<Timeline>,
) -> ClusterCoordinator {
    ClusterCoordinator::for_test(
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Cadence,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 4),
        )
        .no_divergence_guard()
        .progressive(true)
        .timeline(tl),
    )
}

#[test]
fn failed_broadcast_records_counter_and_timeline_event() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());

    // `for_test` leaves `control_streams` empty, so every `send_control`
    // fails -> both live ranks miss the SyncNow broadcast.
    let err = coord
        .broadcast_control(&ControlMsgWire::SyncNow)
        .expect_err("send to a streamless coord must fail");
    assert!(err.to_string().contains("broadcast_control failed"));

    assert_eq!(coord.lost_broadcasts, 1, "one dropped broadcast counted");

    let (_samples, events) = tl.drain();
    let lost: Vec<(String, usize)> = events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::LostBroadcast { control, failures } => {
                Some((control.clone(), *failures))
            }
            _ => None,
        })
        .collect();
    assert_eq!(lost, vec![("SyncNow".to_string(), 2)]);
}

/// A heartbeat that misses ranks AFTER shutdown was broadcast is a teardown
/// race, not lost coordination — the ranks are supposed to be gone.
///
/// From the rig: a 3-rank run that completed perfectly raised `[critical]
/// control_drop root — CoordHeartbeat did not reach 2 live rank(s)` one log
/// line after both of that host's ranks exited cleanly. `is_dead` only knows
/// heartbeat staleness, so a cleanly finished rank still counts as live and
/// its closed socket read as a dropped signal. Wiring twin of the predicate
/// matrix in `lifecycle::tests` — this pins the call site, those pin the rule.
#[test]
fn broadcast_failure_after_shutdown_initiated_is_exempt() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());
    coord.run_phase = RunPhase::ShutdownInitiated;

    let _ = coord.broadcast_control(&ControlMsgWire::CoordHeartbeat);
    assert_eq!(
        coord.lost_broadcasts, 0,
        "a teardown-phase heartbeat miss must not be traced",
    );

    let (_samples, events) = tl.drain();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.kind, EventKind::LostBroadcast { .. })),
        "no LostBroadcast event once shutdown has been broadcast",
    );
}

/// The guard is phase-scoped, not message-scoped: the SAME heartbeat missing
/// mid-run is still a real dropped signal and must still be traced. Without
/// this, "fix the false positive" could silently become "mute the lane".
#[test]
fn the_same_heartbeat_miss_before_shutdown_is_traced() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());

    let _ = coord.broadcast_control(&ControlMsgWire::CoordHeartbeat);
    assert_eq!(coord.lost_broadcasts, 1, "mid-run heartbeat miss is traced");

    let (_samples, events) = tl.drain();
    assert!(
        events
            .iter()
            .any(|e| matches!(&e.kind, EventKind::LostBroadcast { control, .. }
                              if control == "CoordHeartbeat")),
        "mid-run heartbeat miss emits a LostBroadcast event",
    );
}

#[test]
fn failed_shutdown_broadcast_is_exempt() {
    let tl = Timeline::new(1000);
    let mut coord = coord_with_timeline(2, tl.clone());

    // The Shutdown send also fails (no streams) but must NOT be traced.
    let _ = coord.broadcast_control(&ControlMsgWire::Shutdown);
    assert_eq!(coord.lost_broadcasts, 0, "Shutdown is exempt from the trace");

    let (_samples, events) = tl.drain();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e.kind, EventKind::LostBroadcast { .. })),
        "no LostBroadcast event for a failed Shutdown",
    );
}
