//! Unit tests for the rank-resource → timeline plumb in
//! `drain_metrics`: a `MetricsMsgWire` carrying a resource sample must
//! deposit a host-qualified `RankTimelineSample` into the coord's
//! timeline (the persistence feed for remote hosts' GPU/VRAM activity),
//! while resource-less frames deposit nothing.

use crate::distributed::wire::{GpuSnapshotWire, ResourceSampleWire};
use crate::monitor::Timeline;

use super::super::ClusterCoordinator;
use super::{MetricsMsgWire, cfg_sync_cpu};

#[test]
fn drain_metrics_deposits_host_qualified_rank_samples() {
    let tl = Timeline::new(1000);
    let mut coord = ClusterCoordinator::for_test(
        cfg_sync_cpu(2)
            .timeline(tl.clone())
            .rank_hosts(vec!["exa".to_string(), "flodl-pascal".to_string()]),
    );
    let tx = coord.test_metrics_sender();

    tx.send(MetricsMsgWire {
        rank: 1,
        epoch: 0,
        resources: Some(ResourceSampleWire {
            cpu_percent: Some(41.0),
            ram_used_bytes: Some(3_000_000_000),
            gpus: vec![GpuSnapshotWire {
                device_index: 0,
                name: "GP106".to_string(),
                util_percent: Some(88.0),
                vram_allocated_bytes: Some(1_500_000_000),
                vram_total_bytes: Some(6_000_000_000),
            }],
            ..Default::default()
        }),
        ..Default::default()
    })
    .unwrap();
    coord.drain_metrics();

    let deposited = tl.rank_samples();
    assert_eq!(
        deposited.len(),
        1,
        "one resource-bearing frame → one deposit"
    );
    let s = &deposited[0];
    assert_eq!(s.rank, 1);
    assert_eq!(
        s.host, "flodl-pascal",
        "host resolved from rank_hosts by rank"
    );
    assert_eq!(s.cpu_util, Some(41.0));
    assert_eq!(s.ram_used_bytes, Some(3_000_000_000));
    assert_eq!(s.ram_total_bytes, None, "unsampled fields stay None");
    assert_eq!(s.gpus.len(), 1);
    assert_eq!(s.gpus[0].device, 0);
    assert_eq!(s.gpus[0].compute_util, Some(88.0));
    assert_eq!(s.gpus[0].vram_allocated_bytes, Some(1_500_000_000));
    assert_eq!(s.gpus[0].vram_total_bytes, Some(6_000_000_000));
}

/// Resource-less metrics frames (headless ranks) deposit nothing, and
/// a coord with no `rank_hosts` map deposits with an empty host rather
/// than misattributing or dropping the sample.
#[test]
fn drain_metrics_resourceless_frames_and_missing_topology() {
    let tl = Timeline::new(1000);
    let mut coord = ClusterCoordinator::for_test(cfg_sync_cpu(2).timeline(tl.clone()));
    let tx = coord.test_metrics_sender();

    tx.send(MetricsMsgWire {
        rank: 0,
        epoch: 0,
        resources: None,
        ..Default::default()
    })
    .unwrap();
    tx.send(MetricsMsgWire {
        rank: 1,
        epoch: 0,
        resources: Some(ResourceSampleWire::default()),
        ..Default::default()
    })
    .unwrap();
    coord.drain_metrics();

    let deposited = tl.rank_samples();
    assert_eq!(
        deposited.len(),
        1,
        "only the resource-bearing frame deposits"
    );
    assert_eq!(deposited[0].rank, 1);
    assert_eq!(
        deposited[0].host, "",
        "no topology → empty host, not a panic"
    );
}
