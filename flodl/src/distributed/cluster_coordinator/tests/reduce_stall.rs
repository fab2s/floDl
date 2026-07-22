//! Unit tests for the NCCL-backend reduce-stall ceiling
//! (`poll_nccl_reduce_stall`): the NCCL twin of `poll_cpu_averaging`'s
//! hard ceiling. An in-flight collective that never re-acks past the
//! ceiling, with the cohort alive, must escalate to ShutdownWithSave
//! rather than hang silently — while a fresh sync, an all-acked sync, and
//! the CPU backend are all left alone.

use std::time::{Duration, Instant};

use crate::monitor::{EventKind, Timeline};

use super::super::ClusterCoordinator;
use super::{cfg_sync_cpu, cfg_sync_nccl};

#[test]
fn nccl_reduce_stall_past_ceiling_escalates() {
    let tl = Timeline::new(1000);
    let mut coord = ClusterCoordinator::for_test(cfg_sync_nccl(2).timeline(tl.clone()));

    // A sync broadcast 50ms ago that no rank has acked yet, cohort alive.
    coord.cycle.started_at = Some(Instant::now() - Duration::from_millis(50));
    coord.cycle.acked = vec![false, false];

    coord
        .poll_nccl_reduce_stall_with(Duration::from_millis(1))
        .expect("poll returns Ok even when the escalating broadcast fails");

    // Escalation entered: the in-flight marker is disarmed so it fires once.
    assert!(
        coord.cycle.started_at.is_none(),
        "stall escalation disarms the in-flight sync marker",
    );
    // And it actually reached dispatch_shutdown_with_save: that broadcasts
    // ShutdownWithSave, which fails on the streamless test coord and so
    // surfaces as a LostBroadcast trace naming the message.
    let (_samples, events) = tl.drain();
    assert!(
        events.iter().any(|e| matches!(
            &e.kind,
            EventKind::LostBroadcast { control, .. } if control == "ShutdownWithSave"
        )),
        "escalation attempts a ShutdownWithSave broadcast",
    );
}

#[test]
fn nccl_reduce_stall_fresh_sync_is_not_a_stall() {
    let mut coord = ClusterCoordinator::for_test(cfg_sync_nccl(2));
    coord.cycle.started_at = Some(Instant::now());
    coord.cycle.acked = vec![false, false];

    coord
        .poll_nccl_reduce_stall_with(Duration::from_secs(3600))
        .unwrap();

    assert!(
        coord.cycle.started_at.is_some(),
        "a sync well within the ceiling must not escalate",
    );
}

#[test]
fn nccl_reduce_stall_all_acked_is_capture_paths_job() {
    let mut coord = ClusterCoordinator::for_test(cfg_sync_nccl(2));
    // Old broadcast but every rank has acked: not a stall — the elapsed
    // capture path takes the cycle's `started_at`, not this escalation.
    coord.cycle.started_at = Some(Instant::now() - Duration::from_millis(50));
    coord.cycle.acked = vec![true, true];

    coord
        .poll_nccl_reduce_stall_with(Duration::from_millis(1))
        .unwrap();

    assert!(
        coord.cycle.started_at.is_some(),
        "an all-acked sync is not escalated here",
    );
}

#[test]
fn nccl_reduce_stall_noop_on_cpu_backend() {
    let mut coord = ClusterCoordinator::for_test(cfg_sync_cpu(2));
    coord.cycle.started_at = Some(Instant::now() - Duration::from_millis(50));
    coord.cycle.acked = vec![false, false];

    coord
        .poll_nccl_reduce_stall_with(Duration::from_millis(1))
        .unwrap();

    assert!(
        coord.cycle.started_at.is_some(),
        "the CPU backend's ceiling lives in poll_cpu_averaging, not here",
    );
}
