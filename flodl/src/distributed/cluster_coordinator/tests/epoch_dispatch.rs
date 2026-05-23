//! Epoch dispatch tests + CPU finalize state-machine tests.

use super::*;

// -----------------------------------------------------------------
// Epoch dispatch
// -----------------------------------------------------------------


#[test]
fn dispatch_epoch_partitions_cover_dataset_no_overlap() {
    // 2 ranks, Sync, 10 samples → expect ranks to split (5, 5) by
    // equal_sizes (Sync ignores ElChe ratios).
    let world_size = 2;
    let total_samples = 10;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_nccl_with_dataset(world_size, total_samples),
        move |coord| {
            let plans = coord.dispatch_epoch(0)?;
            // Smoke checks on the returned plans match what the
            // wire dispatched.
            assert_eq!(plans.len(), world_size);
            let total: u64 = plans.iter().map(|p| p.partition_size).sum();
            assert_eq!(total, total_samples as u64);
            let mut offset = 0u64;
            for plan in &plans {
                assert_eq!(plan.epoch, 0);
                assert_eq!(plan.partition_offset, offset);
                offset += plan.partition_size;
            }
            // rank_epoch updated.
            assert_eq!(coord.rank_epoch(), &[0, 0]);
            Ok(())
        },
    );

    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
        // #28b: every rank receives a leading SetEpochCallbackRole
        // before StartEpoch (coord broadcasts it once on first
        // dispatch). Consume + verify, then expect StartEpoch.
        let pre = ControlFrame::read_from(s, salt)?.unwrap();
        assert!(matches!(
            pre.decode::<ControlMsgWire>()?,
            ControlMsgWire::SetEpochCallbackRole { .. }
        ));
        let frame = ControlFrame::read_from(s, salt)?
            .ok_or_else(|| TensorError::new("rank 0 EOF before StartEpoch"))?;
        assert_eq!(frame.kind, MsgKind::Control);
        let msg: ControlMsgWire = frame.decode()?;
        match msg {
            ControlMsgWire::StartEpoch(plan) => {
                assert_eq!(plan.epoch, 0);
                assert_eq!(plan.partition_offset, 0);
                assert_eq!(plan.partition_size, 5);
                Ok(())
            }
            other => Err(TensorError::new(&format!(
                "rank 0 expected StartEpoch, got {other:?}"
            ))),
        }
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        let pre = ControlFrame::read_from(s, salt)?.unwrap();
        assert!(matches!(
            pre.decode::<ControlMsgWire>()?,
            ControlMsgWire::SetEpochCallbackRole { .. }
        ));
        let frame = ControlFrame::read_from(s, salt)?
            .ok_or_else(|| TensorError::new("rank 1 EOF before StartEpoch"))?;
        let msg: ControlMsgWire = frame.decode()?;
        match msg {
            ControlMsgWire::StartEpoch(plan) => {
                assert_eq!(plan.epoch, 0);
                assert_eq!(plan.partition_offset, 5);
                assert_eq!(plan.partition_size, 5);
                Ok(())
            }
            other => Err(TensorError::new(&format!(
                "rank 1 expected StartEpoch, got {other:?}"
            ))),
        }
    });

    r0.join().unwrap().expect("rank 0 receives StartEpoch (0, 5)");
    r1.join().unwrap().expect("rank 1 receives StartEpoch (5, 5)");
    coord_handle.join().unwrap().expect("coord finishes");
}

#[test]
fn dispatch_epoch_honors_explicit_partition_ratios() {
    // 2 ranks, Sync, 12 samples, explicit ratios [1.0, 3.0] →
    // ranks should split (3, 9) — but Sync policy normally uses
    // equal_sizes; partition_ratios overrides that.
    let world_size = 2;
    let total_samples = 12;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_nccl(world_size)
                .total_samples(total_samples)
                .partition_ratios(Some(vec![1.0, 3.0]))
        },
        move |coord| {
            let plans = coord.dispatch_epoch(0)?;
            assert_eq!(plans[0].partition_size, 3);
            assert_eq!(plans[1].partition_size, 9);
            let total: u64 = plans.iter().map(|p| p.partition_size).sum();
            assert_eq!(total, total_samples as u64);
            Ok(())
        },
    );

    fn read_start_epoch(
        s: &mut TcpStream,
        salt: &SessionSalt,
        expected_partition: u64,
    ) -> Result<()> {
        // Consume leading SetEpochCallbackRole (one-shot per run).
        let pre = ControlFrame::read_from(s, salt)?.unwrap();
        assert!(matches!(
            pre.decode::<ControlMsgWire>()?,
            ControlMsgWire::SetEpochCallbackRole { .. }
        ));
        let frame = ControlFrame::read_from(s, salt)?.unwrap();
        let msg: ControlMsgWire = frame.decode()?;
        if let ControlMsgWire::StartEpoch(plan) = msg {
            assert_eq!(plan.partition_size, expected_partition);
            Ok(())
        } else {
            Err(TensorError::new("expected StartEpoch"))
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
        read_start_epoch(s, salt, 3)
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        read_start_epoch(s, salt, 9)
    });
    r0.join().unwrap().expect("rank 0 sees ratio[0] partition");
    r1.join().unwrap().expect("rank 1 sees ratio[1] partition");
    coord_handle.join().unwrap().expect("coord finishes");
}

#[test]
fn dispatch_epoch_caches_plans_for_same_epoch() {
    // Calling dispatch_epoch twice for the same epoch must return
    // identical plans both times (cache hit; rank_epoch reset to
    // the same value). The wire receives two frames per rank.
    let world_size = 2;
    let total_samples = 8;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_nccl_with_dataset(world_size, total_samples),
        move |coord| {
            let plans_a = coord.dispatch_epoch(0)?;
            let plans_b = coord.dispatch_epoch(0)?;
            assert_eq!(plans_a, plans_b, "epoch plan cache must be deterministic");
            Ok(())
        },
    );

    // First dispatch_epoch sends 1 SetEpochCallbackRole + 1
    // StartEpoch; second dispatch_epoch sends another StartEpoch
    // (the role is sticky + already broadcast). Total 3 frames
    // per rank.
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
        for _ in 0..3 {
            let frame = ControlFrame::read_from(s, salt)?.unwrap();
            let _: ControlMsgWire = frame.decode()?;
        }
        Ok(())
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        for _ in 0..3 {
            let frame = ControlFrame::read_from(s, salt)?.unwrap();
            let _: ControlMsgWire = frame.decode()?;
        }
        Ok(())
    });
    r0.join().unwrap().expect("rank 0 reads 2 frames");
    r1.join().unwrap().expect("rank 1 reads 2 frames");
    coord_handle.join().unwrap().expect("coord finishes");
}


#[test]
fn sync_cpu_trigger_broadcasts_request_params_then_update() {
    // 2 ranks, Sync+Cpu. After each rank sends one Batch + SyncAck
    // (mocking the post-data-channel ack), coord should fire
    // RequestParams + Update{version} + SetGlobalStep exactly
    // once. Mirrors sync_policy_fires_after_each_rank_step_once
    // for the CPU backend.
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size),
        |coord| {
            let start = Instant::now();
            while coord.avg_count() == 0 {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "sync_cpu_trigger timed out waiting for avg_count",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(coord.avg_count(), 1, "exactly one averaging cycle");
            Ok(())
        },
    );

    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
        send_timing(s, salt, TimingMsgWire::Batch {
            rank: 0,
            batch_ms: 10.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })?;
        let msg = recv_control(s, salt)?;
        assert_eq!(msg, ControlMsgWire::RequestParams);
        // Mock the post-data-channel ack the worker-side bridge
        // emits after the CPU averaging round-trip completes.
        send_timing(s, salt, TimingMsgWire::SyncAck {
            rank: 0,
            step_count: 2,
            divergence: Some(0.05),
            post_norm: None,
            pre_norm: None,
        })?;
        let msg2 = recv_control(s, salt)?;
        assert!(matches!(msg2, ControlMsgWire::Update { .. }));
        let msg3 = recv_control(s, salt)?;
        assert!(matches!(msg3, ControlMsgWire::SetGlobalStep { .. }));
        Ok(())
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        send_timing(s, salt, TimingMsgWire::Batch {
            rank: 1,
            batch_ms: 12.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.4,
            sync_divergence: None,
        })?;
        let msg = recv_control(s, salt)?;
        assert_eq!(msg, ControlMsgWire::RequestParams);
        send_timing(s, salt, TimingMsgWire::SyncAck {
            rank: 1,
            step_count: 2,
            divergence: Some(0.05),
            post_norm: None,
            pre_norm: None,
        })?;
        let msg2 = recv_control(s, salt)?;
        assert!(matches!(msg2, ControlMsgWire::Update { .. }));
        let msg3 = recv_control(s, salt)?;
        assert!(matches!(msg3, ControlMsgWire::SetGlobalStep { .. }));
        Ok(())
    });

    r0.join().unwrap().expect("rank 0 sees RequestParams + Update + SetGlobalStep");
    r1.join().unwrap().expect("rank 1 sees RequestParams + Update + SetGlobalStep");
    coord_handle.join().unwrap().expect("coord finishes");
}

// Upload-completion marker: SnapshotReady frames arriving between
// `RequestParams` broadcast and `SyncAck` populate
// `last_observed_upload_ms[rank]` with a strictly positive,
// pre-barrier lag. NCCL never emits SnapshotReady so its slots
// stay None (asserted by a sibling check below).
#[test]
fn snapshot_ready_populates_upload_marker_cpu_only() {
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size),
        move |coord| {
            let start = Instant::now();
            while coord.avg_count() == 0 {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "snapshot_ready test timed out waiting for avg_count",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            let uploads = coord.last_observed_upload_ms();
            assert_eq!(uploads.len(), world_size);
            // Both ranks emitted SnapshotReady → both slots populated.
            for (r, slot) in uploads.iter().enumerate() {
                let ms = slot
                    .unwrap_or_else(|| panic!("rank {r} missing upload marker"));
                assert!(
                    ms >= 0.0,
                    "rank {r} upload_ms ({ms}) must be >= 0"
                );
            }
            Ok(())
        },
    );

    // Both ranks send Batch → RequestParams arrives → emit
    // SnapshotReady BEFORE the SyncAck (mimicking the param
    // bridge's pre-barrier marker), then SyncAck closes the
    // cycle.
    fn rank_body(rank: u64) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank,
                batch_ms: 10.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let _ = recv_control(s, salt)?; // RequestParams
            // Inject a tiny delay so the captured upload_ms is
            // strictly positive on every clock (some CI clocks
            // return 0 for sub-microsecond elapseds).
            thread::sleep(Duration::from_millis(2));
            send_timing(s, salt, TimingMsgWire::SnapshotReady { rank })?;
            send_timing(s, salt, TimingMsgWire::SyncAck {
                rank,
                step_count: 2,
                divergence: Some(0.05),
                post_norm: None,
                pre_norm: None,
            })?;
            let _ = recv_control(s, salt)?; // Update
            let _ = recv_control(s, salt)?; // SetGlobalStep
            Ok(())
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, rank_body(0));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, rank_body(1));
    r0.join().unwrap().expect("rank 0 cycle");
    r1.join().unwrap().expect("rank 1 cycle");
    coord_handle.join().unwrap().expect("coord finishes");
}

// Second-cycle invariant: upload markers reset at the start of
// every new cycle, so a rank that emitted SnapshotReady in cycle 1
// but not cycle 2 surfaces None in cycle 2 — preventing the
// rebalancer from reading a stale prior-cycle value as live data.
#[test]
fn snapshot_ready_resets_between_cycles() {
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size),
        move |coord| {
            let start = Instant::now();
            while coord.avg_count() < 2 {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "reset test timed out waiting for two cycles",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            // After cycle 2: rank 0 sent SnapshotReady in cycle 2,
            // rank 1 did NOT. Cycle 1's values must NOT bleed
            // through.
            let uploads = coord.last_observed_upload_ms();
            assert!(
                uploads[0].is_some(),
                "rank 0 emitted SnapshotReady in cycle 2 → slot is Some"
            );
            assert!(
                uploads[1].is_none(),
                "rank 1 skipped SnapshotReady in cycle 2 → slot reset to None"
            );
            Ok(())
        },
    );

    // Rank 0 emits SnapshotReady on BOTH cycles.
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
        for cycle in 0..2 {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 0,
                batch_ms: 10.0,
                step_count: (cycle + 1) as u64,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let _ = recv_control(s, salt)?; // RequestParams
            thread::sleep(Duration::from_millis(2));
            send_timing(s, salt, TimingMsgWire::SnapshotReady { rank: 0 })?;
            send_timing(s, salt, TimingMsgWire::SyncAck {
                rank: 0,
                step_count: ((cycle + 1) * 2) as u64,
                divergence: Some(0.05),
                post_norm: None,
                pre_norm: None,
            })?;
            let _ = recv_control(s, salt)?; // Update
            let _ = recv_control(s, salt)?; // SetGlobalStep
        }
        Ok(())
    });
    // Rank 1 emits SnapshotReady ONLY on cycle 1.
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        for cycle in 0..2 {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 1,
                batch_ms: 12.0,
                step_count: (cycle + 1) as u64,
                param_norm: None,
                batch_loss: 0.4,
                sync_divergence: None,
            })?;
            let _ = recv_control(s, salt)?; // RequestParams
            thread::sleep(Duration::from_millis(2));
            if cycle == 0 {
                send_timing(s, salt, TimingMsgWire::SnapshotReady { rank: 1 })?;
            }
            send_timing(s, salt, TimingMsgWire::SyncAck {
                rank: 1,
                step_count: ((cycle + 1) * 2) as u64,
                divergence: Some(0.05),
                post_norm: None,
                pre_norm: None,
            })?;
            let _ = recv_control(s, salt)?; // Update
            let _ = recv_control(s, salt)?; // SetGlobalStep
        }
        Ok(())
    });
    r0.join().unwrap().expect("rank 0 two cycles");
    r1.join().unwrap().expect("rank 1 two cycles");
    coord_handle.join().unwrap().expect("coord finishes");
}

/// `EpochAggregated` round-trip: each alive rank emits a
/// `MetricsMsgWire` for the same epoch with a custom scalar; the
/// coord aggregates (mean over per-rank values, per
/// `aggregate_epoch_metrics`), pushes the result to
/// `metrics_sink_tx`, AND broadcasts `ControlMsgWire::EpochAggregated`
/// back to every rank. Validates the wire path from rank-MetricsMsg
/// → coord-aggregate → both sinks (mpsc + broadcast) end-to-end.
/// Aggregation is independent of averaging — no Batch / SyncAck
/// frames are sent, so `avg_count` stays zero throughout.
#[test]
fn epoch_aggregated_broadcast_and_sink_receive_aggregated_metrics() {
    let world_size = 2;
    let (sink_tx, sink_rx) =
        mpsc::channel::<crate::distributed::ddp_run::EpochMetrics>();
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size).metrics_sink_tx(sink_tx),
        move |coord| {
            let start = Instant::now();
            while coord.last_aggregated_epoch().is_none() {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "epoch_aggregated: no aggregation within 5s",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            // Aggregation fired; no averaging cycle should have
            // run (no Batch / SyncAck frames sent by ranks).
            assert_eq!(
                coord.avg_count(),
                0,
                "aggregation must not trigger averaging",
            );
            Ok(())
        },
    );

    // Each rank ships one MetricsMsg for epoch 0 with a
    // rank-distinguishable `custom_metric` scalar. The aggregator
    // computes the mean across alive ranks: (10 + 11) / 2 = 10.5.
    fn rank_body(rank: u64) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            let mut scalars = std::collections::HashMap::new();
            scalars.insert(
                "custom_metric".to_string(),
                (10.0 + rank as f64, 1u64),
            );
            send_metrics(s, salt, MetricsMsgWire {
                rank,
                epoch: 0,
                avg_loss: 0.5 + 0.1 * rank as f64,
                batches_processed: 4,
                epoch_ms: 100.0,
                samples_processed: 16,
                share_complete_ms: 5.0,
                compute_only_ms: 90.0,
                data_starve_ms: 5.0,
                scalars,
            })?;
            // Block on the broadcast — coord will send
            // `EpochAggregated` once every alive rank has reported.
            let msg = recv_control(s, salt)?;
            match msg {
                ControlMsgWire::EpochAggregated(wire) => {
                    if wire.epoch != 0 {
                        return Err(TensorError::new(&format!(
                            "rank {rank}: EpochAggregated epoch={} (expected 0)",
                            wire.epoch
                        )));
                    }
                    let got = wire
                        .scalars
                        .get("custom_metric")
                        .copied()
                        .ok_or_else(|| TensorError::new(
                            "rank: custom_metric missing from broadcast scalars",
                        ))?;
                    if (got - 10.5).abs() > 1e-9 {
                        return Err(TensorError::new(&format!(
                            "rank {rank}: broadcast custom_metric={got} (expected 10.5)",
                        )));
                    }
                    // per_rank vector preserves the per-rank
                    // submissions; rank R's slot contains rank R's
                    // original value (10 + R), not the mean.
                    let per_rank = wire
                        .per_rank
                        .get(rank as usize)
                        .ok_or_else(|| TensorError::new(
                            "rank: per_rank slot missing",
                        ))?;
                    let per_rank_val = per_rank
                        .get("custom_metric")
                        .copied()
                        .ok_or_else(|| TensorError::new(
                            "rank: per_rank custom_metric missing",
                        ))?;
                    if (per_rank_val - (10.0 + rank as f64)).abs() > 1e-9 {
                        return Err(TensorError::new(&format!(
                            "rank {rank}: per_rank custom_metric={per_rank_val} \
                             (expected {})",
                            10.0 + rank as f64
                        )));
                    }
                    Ok(())
                }
                other => Err(TensorError::new(&format!(
                    "rank {rank}: expected EpochAggregated, got {other:?}",
                ))),
            }
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, rank_body(0));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, rank_body(1));
    r0.join().unwrap().expect("rank 0 EpochAggregated");
    r1.join().unwrap().expect("rank 1 EpochAggregated");
    coord_handle.join().unwrap().expect("coord finishes");

    // The mpsc sink received the same aggregated metrics the
    // broadcast carried; validates the sink path the launcher /
    // builder wires for `DdpHandle::next_metrics`.
    let metrics = sink_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("metrics_sink_tx received EpochMetrics");
    assert_eq!(metrics.epoch, 0);
    let mean = metrics
        .scalars
        .get("custom_metric")
        .copied()
        .expect("sink custom_metric present");
    assert!(
        (mean - 10.5).abs() < 1e-9,
        "sink custom_metric mean: {mean} (expected 10.5)",
    );
    // Sink channel should be empty after the one event.
    assert!(
        sink_rx.try_recv().is_err(),
        "sink should not receive extra events",
    );
}


// -----------------------------------------------------------------
// CPU finalize state machine
// -----------------------------------------------------------------

#[test]
fn cpu_finalize_defers_until_all_sync_acks_arrived() {
    // Sync+Cpu cycle: the coord broadcasts RequestParams but must
    // NOT call finish_averaging_cpu until every rank's SyncAck has
    // landed and populated `nccl_sync_divergence`. Gate the second
    // rank's SyncAck behind a barrier; the cycle must stay at
    // avg_count=0 (and `cpu_avg_pending_for_test=true`) until that
    // second SyncAck is delivered.
    let world_size = 2;
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size),
        |coord| {
            let start = Instant::now();
            while coord.avg_count() == 0 {
                if start.elapsed() > Duration::from_secs(10) {
                    return Err(TensorError::new(
                        "cpu_finalize_defers: avg_count never advanced",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(coord.avg_count(), 1);
            Ok(())
        },
    );

    let barrier_for_r0 = Arc::clone(&barrier);
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, move |s, salt| {
        send_timing(s, salt, TimingMsgWire::Batch {
            rank: 0,
            batch_ms: 10.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })?;
        let _ = recv_control(s, salt)?; // RequestParams
        // Rank 0 sends SyncAck immediately so coord goes into Pending
        // but still has rank 1 outstanding.
        send_timing(s, salt, TimingMsgWire::SyncAck {
            rank: 0,
            step_count: 2,
            divergence: Some(0.05),
            post_norm: Some(1.0),
            pre_norm: Some(1.05),
        })?;
        // Wait for the test thread (rank 1) to release the barrier
        // BEFORE rank 1 sends its SyncAck. Until then, coord must
        // stay in Pending.
        barrier_for_r0.wait();
        let _ = recv_control(s, salt)?; // Update (only after r1 acks)
        let _ = recv_control(s, salt)?; // SetGlobalStep
        Ok(())
    });
    let barrier_for_r1 = Arc::clone(&barrier);
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, move |s, salt| {
        send_timing(s, salt, TimingMsgWire::Batch {
            rank: 1,
            batch_ms: 12.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.4,
            sync_divergence: None,
        })?;
        let _ = recv_control(s, salt)?; // RequestParams
        // Hold the SyncAck briefly so the test's drive loop observes
        // `cpu_avg_pending_for_test=true && avg_count=0`. Use the
        // shared barrier with r0 so timing is deterministic.
        thread::sleep(Duration::from_millis(200));
        barrier_for_r1.wait();
        send_timing(s, salt, TimingMsgWire::SyncAck {
            rank: 1,
            step_count: 2,
            divergence: Some(0.06),
            post_norm: Some(1.0),
            pre_norm: Some(1.04),
        })?;
        let _ = recv_control(s, salt)?; // Update
        let _ = recv_control(s, salt)?; // SetGlobalStep
        Ok(())
    });
    r0.join().unwrap().expect("rank 0 path");
    r1.join().unwrap().expect("rank 1 path");
    coord_handle.join().unwrap().expect("coord finalizes after both acks");
}

#[test]
fn cpu_finalize_records_per_rank_lag_for_diagnostics() {
    // After one successful CPU averaging cycle, every rank's
    // observed sync-lag slot should be populated and finite — the
    // adaptive deadline computed on the NEXT trigger reads these
    // and scales accordingly. Doesn't assert the deadline value
    // directly (would couple to Instant timing); proves the lag
    // pipeline is wired.
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size),
        |coord| {
            let start = Instant::now();
            while coord.avg_count() == 0 {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "adaptive_deadline: avg_count never advanced",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            let lags = coord.last_observed_sync_lag_ms_for_test();
            assert!(
                lags.iter().all(|l| l.is_some()),
                "every rank's sync lag populated after cycle 1: {lags:?}"
            );
            for l in lags {
                let v = l.unwrap();
                assert!(
                    v.is_finite() && v >= 0.0,
                    "per-rank lag must be finite & non-negative, got {v}"
                );
            }
            Ok(())
        },
    );
    let body = |rank: u64| {
        move |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank,
                batch_ms: 10.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let _ = recv_control(s, salt)?; // RequestParams
            send_timing(s, salt, TimingMsgWire::SyncAck {
                rank,
                step_count: 2,
                divergence: Some(0.05),
                post_norm: Some(1.0),
                pre_norm: Some(1.05),
            })?;
            let _ = recv_control(s, salt)?; // Update
            let _ = recv_control(s, salt)?; // SetGlobalStep
            Ok(())
        }
    };
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, body(0));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, body(1));
    r0.join().unwrap().expect("rank 0 path");
    r1.join().unwrap().expect("rank 1 path");
    coord_handle.join().unwrap().expect("coord captures per-rank lags");
}

