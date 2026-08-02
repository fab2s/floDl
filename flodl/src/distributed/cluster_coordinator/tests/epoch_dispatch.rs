//! Epoch dispatch tests + CPU finalize state-machine tests.

use super::*;

// -----------------------------------------------------------------
// Epoch dispatch
// -----------------------------------------------------------------


#[test]
fn one_shot_checkpoint_meta_round_trips_to_resume_coverage() {
    // End-to-end coverage half of the async resume contract (no network):
    // build partial coverage on an epoch pool, fire the one-shot checkpoint
    // (coord writes .meta.json + coverage), then resume a fresh coord from
    // that meta and confirm it reconstructs the pool to exactly the holes.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::{AverageBackend, SHUFFLE_BASE_SEED};
    use crate::distributed::{CheckpointBundle, CheckpointMeta, SaveReason};

    let world_size = 2;
    let total = 100usize;

    let dir = std::env::temp_dir()
        .join(format!("flodl_ckpt_resume_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("ckpt").to_string_lossy().into_owned();

    // Coord A: progressive cpu-cadence, save_path set, one-shot armed @ epoch 2.
    let cfg_a = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .no_divergence_guard()
    .total_samples(total)
    .batch_size(1)
    .num_epochs(5)
    .save_path(stem.clone())
    .checkpoint_at_epoch(2);
    let mut coord_a = ClusterCoordinator::for_test(cfg_a);

    // Partial coverage on epoch 2 (reservation spans: rank 0 owns [0,50),
    // rank 1 owns [50,100)): rank 0 completes (0,40); rank 1 has (50,30)
    // still in flight. Covered = [0,40); uncovered = rank 0's span residue
    // (40,10) + rank 1's in-flight (50,30) + rank 1's span residue (80,20)
    // → coalesced (40,60).
    coord_a.install_chunk_pool_for_test(2, total);
    {
        let pool = coord_a.chunk_pools.get_mut(&2).unwrap();
        assert_eq!(pool.take_chunk(40, 0).unwrap(), (0, 40));
        assert_eq!(pool.take_chunk(30, 1).unwrap(), (50, 30));
        pool.mark_completed(0, 40);
    }
    coord_a.rank_epoch[0] = 2;
    coord_a.rank_epoch[1] = 2;

    // Drive the split trigger directly (no real reduce): arm at cycle start
    // (captures coverage + disarms), then write the stashed meta at finish.
    coord_a.maybe_arm_checkpoint();
    assert!(
        coord_a.checkpoint_at_epoch.is_none(),
        "one-shot disarms after firing"
    );
    assert!(
        coord_a.pending_checkpoint_coverage.is_some(),
        "coverage captured at arm time"
    );
    coord_a.finish_pending_checkpoint_meta();
    assert!(
        coord_a.pending_checkpoint_coverage.is_none(),
        "pending coverage consumed at finish"
    );

    // Read the meta back. The meta write is detached off the coordinator
    // finish path (production keeps checkpointing off the training clock) and
    // committed via temp+atomic-rename, so the path appearing means it is
    // whole — poll briefly for it rather than racing the writer thread.
    let meta_path = CheckpointBundle::meta_path(&stem);
    let mut waited = 0;
    while !meta_path.exists() && waited < 400 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        waited += 1;
    }
    let meta = CheckpointMeta::read_from_file(&meta_path).unwrap();
    assert_eq!(meta.save_reason, SaveReason::Checkpoint);
    let cov = meta.coverage.clone().expect("coverage block recorded");
    assert_eq!(cov.seed, SHUFFLE_BASE_SEED);
    assert_eq!(cov.per_epoch.len(), 1);
    assert_eq!(cov.per_epoch[0].epoch, 2);
    assert_eq!(cov.per_epoch[0].total_samples, total);
    assert_eq!(cov.per_epoch[0].uncovered_ranges, vec![(40, 60)]);

    // Coord B: resume from the recorded coverage.
    let cfg_b = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .no_divergence_guard()
    .total_samples(total)
    .batch_size(1)
    .num_epochs(5)
    .resume_from_meta(&meta);
    let mut coord_b = ClusterCoordinator::for_test(cfg_b);

    let handled = coord_b.resume_progressive_from_coverage().unwrap();
    assert!(handled, "coverage present → resume handled the kickoff");
    // Reconstructed pool holds only the holes. (Headless send_control fails and
    // rolls back the dispatch take, so `remaining` reflects the staged holes.)
    let pool_b = coord_b.chunk_pools.get(&2).expect("epoch 2 pool reconstructed");
    assert_eq!(pool_b.remaining(), 60);
    assert!(!pool_b.is_epoch_done());
    // Epochs 0,1 anchored as already aggregated; cohort placed at epoch 2.
    assert_eq!(coord_b.last_aggregated_epoch, Some(1));
    assert_eq!(coord_b.rank_epoch, vec![2, 2]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn progressive_kickoff_is_best_effort_per_rank() {
    // Headless coordinator: every send_control fails. First-chunk
    // dispatch must not escalate a rank's send failure into an Err —
    // the launcher treats Err as fatal, the coordinator thread dies,
    // and every healthy rank self-destructs at the 30s watchdog. The
    // kickoff logs, rolls back each failed take, and still returns
    // world_size plans (empty for the failed ranks), exactly like the
    // non-progressive loop.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;

    let world_size = 2;
    let total = 100usize;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .no_divergence_guard()
    .total_samples(total)
    .batch_size(1)
    .num_epochs(1)
    .progressive(true);
    let mut coord = ClusterCoordinator::for_test(cfg);

    let plans = coord
        .dispatch_epoch(0)
        .expect("headless send failures must not abort the kickoff");
    assert_eq!(plans.len(), world_size, "one plan slot per rank");

    // Every failed dispatch rolled back its pool take: no ghost
    // in-flight chunk may survive a send that never reached the rank.
    let pool = coord.chunk_pools.get(&0).expect("epoch 0 pool created");
    assert_eq!(pool.remaining(), total);
    assert!(!pool.is_epoch_done());
}

#[test]
fn advisory_spans_own_first_margins_last() {
    // Reservation advisory geometry: world 2, total 100, batch 1 →
    // equal spans [0,50) / [50,100); uncalibrated ElChe anchor 4 →
    // window counts [4,4] → margins are 4-sample span tails.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;

    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        2,
        ElChe::new(2, 4),
    )
    .no_divergence_guard()
    .total_samples(100)
    .batch_size(1)
    .num_epochs(1);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.install_chunk_pool_for_test(0, 100);

    let spans0 = coord.advisory_spans_for_rank(0, 0);
    assert_eq!(spans0[0], (0, 50), "own span first");
    assert_eq!(spans0[1], (96, 4), "then the other span's window-sized tail");
    assert_eq!(spans0.len(), 2);

    let spans1 = coord.advisory_spans_for_rank(0, 1);
    assert_eq!(spans1[0], (50, 50));
    assert_eq!(spans1[1], (46, 4));

    // No pool for the epoch: no advisory.
    assert!(coord.advisory_spans_for_rank(7, 0).is_empty());
}

#[test]
fn predicted_epoch_spans_match_table_and_stop_at_run_end() {
    // The predicted next-epoch segment uses the same ratio table the
    // pool will be built from (no pool exists yet); past the run's
    // last epoch there is nothing to predict.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;

    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        2,
        ElChe::new(2, 4),
    )
    .no_divergence_guard()
    .total_samples(100)
    .batch_size(1)
    .num_epochs(2);
    let coord = ClusterCoordinator::for_test(cfg);

    // Same geometry as the live table: equal spans + window tails.
    assert_eq!(
        coord.predicted_epoch_spans(1, 0),
        vec![(0, 50), (96, 4)],
    );
    assert_eq!(
        coord.predicted_epoch_spans(1, 1),
        vec![(50, 50), (46, 4)],
    );
    // Epoch 2 does not exist in a 2-epoch run.
    assert!(coord.predicted_epoch_spans(2, 0).is_empty());
}

#[test]
fn resume_from_coverage_rejects_seed_mismatch() {
    // The contract rests on the same permutation: a seed change must error
    // loudly rather than silently re-train the wrong samples.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;
    use crate::distributed::{CoverageBlock, EpochCoverage};

    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .no_divergence_guard()
    .total_samples(100)
    .batch_size(1)
    .num_epochs(5)
    .seed(7); // run seed
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.start_coverage = Some(CoverageBlock {
        seed: 999, // recorded with a DIFFERENT seed
        epoch_splits: 1,
        batch_size: 1,
        per_epoch: vec![EpochCoverage {
            epoch: 0,
            total_samples: 100,
            uncovered_ranges: vec![(0, 100)],
        }],
    });
    let err = coord.resume_progressive_from_coverage().unwrap_err();
    assert!(err.to_string().contains("seed"), "got: {err}");
}

#[test]
fn resume_from_coverage_absent_falls_back() {
    // No coverage → Ok(false): caller dispatches a fresh epoch.
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;

    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .no_divergence_guard()
    .total_samples(100)
    .batch_size(1)
    .num_epochs(5);
    let mut coord = ClusterCoordinator::for_test(cfg);
    assert!(!coord.resume_progressive_from_coverage().unwrap());
}

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
        // Every rank receives a leading SetEpochCallbackRole before
        // StartEpoch (coord broadcasts it once on first dispatch).
        // Consume + verify, then expect StartEpoch.
        let pre = recv_frame(s, salt)?.unwrap();
        assert!(matches!(
            pre.decode::<ControlMsgWire>()?,
            ControlMsgWire::SetEpochCallbackRole { .. }
        ));
        let frame = recv_frame(s, salt)?
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
        let pre = recv_frame(s, salt)?.unwrap();
        assert!(matches!(
            pre.decode::<ControlMsgWire>()?,
            ControlMsgWire::SetEpochCallbackRole { .. }
        ));
        let frame = recv_frame(s, salt)?
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
        let pre = recv_frame(s, salt)?.unwrap();
        assert!(matches!(
            pre.decode::<ControlMsgWire>()?,
            ControlMsgWire::SetEpochCallbackRole { .. }
        ));
        let frame = recv_frame(s, salt)?.unwrap();
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
            let frame = recv_frame(s, salt)?.unwrap();
            let _: ControlMsgWire = frame.decode()?;
        }
        Ok(())
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        for _ in 0..3 {
            let frame = recv_frame(s, salt)?.unwrap();
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
    // RequestParams + Throttle (the CPU hard barrier) + Update{version}
    // + SetGlobalStep exactly once. Mirrors
    // sync_policy_fires_after_each_rank_step_once for the CPU backend.
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
            batch_ms: 10.0, data_ms: 0.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })?;
        let msg = recv_control(s, salt)?;
        assert_eq!(msg, ControlMsgWire::RequestParams);
        // Sync (non-progressive) CPU broadcasts a hard barrier: the
        // fast rank is Throttled after snapshotting, released by the
        // averaged Update. Mirrors NCCL's AllReduce-block. (Cadence is
        // progressive and self-barriers via dispatch starvation, so it
        // no longer gets a Throttle.)
        let throttle = recv_control(s, salt)?;
        assert_eq!(throttle, ControlMsgWire::Throttle);
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
            batch_ms: 12.0, data_ms: 0.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.4,
            sync_divergence: None,
        })?;
        let msg = recv_control(s, salt)?;
        assert_eq!(msg, ControlMsgWire::RequestParams);
        let throttle = recv_control(s, salt)?;
        assert_eq!(throttle, ControlMsgWire::Throttle);
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

// atomic-dispatch: on Cadence, `finish_averaging_cpu` folds each rank's
// next reduce-window chunk into its `Update` frame. These unit-test the
// fold helper directly (no network): mid-epoch it hands back the rank's
// scheduled window; at an epoch boundary (drained / missing pool) it
// returns `None` so the epoch-advance path takes over.
#[test]
fn fold_next_chunk_returns_schedule_window_mid_epoch() {
    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 10),
    )
    .no_divergence_guard();
    let mut coord = ClusterCoordinator::for_test(cfg);
    // Calibrate: rank 0 fast, rank 1 slow → asymmetric batch_counts.
    coord
        .el_che_mut_for_test()
        .report_timing(&[500.0, 1000.0], &[10, 10], 10.0);
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    assert!(counts[0] > 0 && counts[1] > 0, "calibrated counts: {counts:?}");

    // batch_size defaults to 1 → partition_size (samples) == batches.
    // Pool far larger than one window so the fold stays intra-epoch.
    coord.install_chunk_pool_for_test(0, 1000);
    coord.set_rank_epoch_for_test(0, 0);
    coord.set_rank_epoch_for_test(1, 0);

    let p0 = coord
        .fold_next_chunk_for_rank(0)
        .expect("rank 0 mid-epoch chunk");
    assert_eq!(p0.epoch, 0);
    assert_eq!(
        p0.partition_size, counts[0] as u64,
        "rank 0 folded window must equal its schedule count",
    );
    let p1 = coord
        .fold_next_chunk_for_rank(1)
        .expect("rank 1 mid-epoch chunk");
    assert_eq!(
        p1.partition_size, counts[1] as u64,
        "rank 1 folded window must equal its schedule count",
    );
    // Disjoint slices of the pool.
    assert_ne!(p0.partition_offset, p1.partition_offset);
}

// Edge schedule: at the tail (less than a full window of work left),
// even the Async proportional path caps each rank at its OWN share
// (batch_counts[rank].min(remaining)) -- NOT a proportional split of the
// remainder, NOT floored up to min_chunk. This is the share-cap that
// keeps any rank from over-driving before the final reduce; a rank that
// finds the drained pool gets 0 and is excluded by the weighted average.
#[test]
fn tail_dispatch_is_share_capped_not_proportional() {
    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Async,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 10),
    )
    .no_divergence_guard();
    let mut coord = ClusterCoordinator::for_test(cfg);
    // Calibrate strongly asymmetric (rank 0 fast, rank 1 slow) so a
    // proportional split would differ from the per-rank share cap.
    for _ in 0..5 {
        coord
            .el_che_mut_for_test()
            .report_timing(&[200.0, 1000.0], &[10, 10], 10.0);
    }
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    let total: usize = counts.iter().sum();
    assert!(counts[0] > 0 && counts[1] > 0 && total > 2, "counts: {counts:?}");

    // batch_size defaults to 1 (samples == batches). Tail: one batch short
    // of a full window.
    let remaining = total - 1;
    coord.install_chunk_pool_for_test(0, remaining);
    coord.set_rank_epoch_for_test(0, 0);
    coord.set_rank_epoch_for_test(1, 0);

    for rank in 0..world_size {
        let got = coord.compute_chunk_batches_for_test(rank, 0);
        let want = counts[rank].min(remaining);
        assert_eq!(
            got, want,
            "tail rank {rank}: expected share-cap {want}, got {got} \
             (counts={counts:?}, remaining={remaining})",
        );
    }
}

// Cadence tail: the final reduce window is planned as one coherent split of
// the whole remainder, sized so no rank ends at exactly 1 step (which would
// zero its marginal delivered sample and trip the all-or-none feed fallback).
// Coverage stays exact and the fast rank is never over-driven.
#[test]
fn cadence_tail_plans_a_no_lone_one_window() {
    let world_size = 3;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 10),
    )
    .no_divergence_guard();
    let mut coord = ClusterCoordinator::for_test(cfg);
    // Asymmetric calibration: rank 0 fast, ranks 1/2 slow.
    for _ in 0..5 {
        coord
            .el_che_mut_for_test()
            .report_timing(&[200.0, 900.0, 1000.0], &[10, 10, 10], 10.0);
    }
    let counts = coord.el_che_for_test().batch_counts().to_vec();
    let total: usize = counts.iter().sum();
    let fast = counts.iter().enumerate().max_by_key(|&(_, &c)| c).unwrap().0;
    assert!(total > world_size + 2, "counts: {counts:?}");

    // One batch short of a full window: the whole remainder is the final
    // window (proportional sub-window regime).
    let remaining = total - 1;
    coord.install_chunk_pool_for_test(0, remaining);
    for r in 0..world_size {
        coord.set_rank_epoch_for_test(r, 0);
    }
    coord.refresh_final_window_plan_for_test(0);

    let sizes: Vec<usize> = (0..world_size)
        .map(|r| coord.compute_chunk_batches_for_test(r, 0))
        .collect();
    assert_eq!(
        sizes.iter().sum::<usize>(),
        remaining,
        "coverage exact: {sizes:?} (counts={counts:?}, remaining={remaining})",
    );
    assert!(
        sizes.iter().all(|&n| n != 1),
        "no rank at exactly 1 step: {sizes:?}",
    );
    assert!(
        sizes[fast] >= sizes.iter().enumerate()
            .filter(|&(r, _)| r != fast)
            .map(|(_, &n)| n)
            .max()
            .unwrap_or(0),
        "fast rank stays dominant: {sizes:?}",
    );
}

#[test]
fn fold_next_chunk_none_at_epoch_boundary() {
    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 10),
    )
    .no_divergence_guard();
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord
        .el_che_mut_for_test()
        .report_timing(&[500.0, 1000.0], &[10, 10], 10.0);
    // Drained pool (remaining == 0) == epoch boundary.
    coord.install_chunk_pool_for_test(0, 0);
    coord.set_rank_epoch_for_test(0, 0);
    assert!(
        coord.fold_next_chunk_for_rank(0).is_none(),
        "epoch boundary (drained pool) must fold no chunk",
    );
    // No pool for the rank's epoch at all → also None.
    coord.set_rank_epoch_for_test(1, 5);
    assert!(
        coord.fold_next_chunk_for_rank(1).is_none(),
        "missing pool must fold no chunk",
    );
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
                batch_ms: 10.0, data_ms: 0.0,
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
                batch_ms: 10.0, data_ms: 0.0,
                step_count: (cycle + 1) as u64,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let _ = recv_control(s, salt)?; // RequestParams
            let _ = recv_control(s, salt)?; // Throttle (Sync non-progressive barrier)
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
                batch_ms: 12.0, data_ms: 0.0,
                step_count: (cycle + 1) as u64,
                param_norm: None,
                batch_loss: 0.4,
                sync_divergence: None,
            })?;
            let _ = recv_control(s, salt)?; // RequestParams
            let _ = recv_control(s, salt)?; // Throttle (Sync non-progressive barrier)
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
                if start.elapsed() > Duration::from_secs(30) {
                    return Err(TensorError::new(
                        "epoch_aggregated: no aggregation within 30s",
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
                resources: None,
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
            batch_ms: 10.0, data_ms: 0.0,
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
            batch_ms: 12.0, data_ms: 0.0,
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
                batch_ms: 10.0, data_ms: 0.0,
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


// -----------------------------------------------------------------
// Epoch splits: an epoch is a slice of a pass, not the whole pass
// -----------------------------------------------------------------

#[test]
fn one_split_leaves_the_epoch_covering_the_whole_pass() {
    // The default. Every plan set must still tile all 100 picks.
    let mut coord =
        ClusterCoordinator::for_test(cfg_sync_cpu(2).total_samples(100).batch_size(1));
    assert_eq!(coord.epoch_samples(0), 100);
    let covered: u64 = coord.plans_for_epoch(0).iter().map(|p| p.partition_size).sum();
    assert_eq!(covered, 100);
}

#[test]
fn split_epochs_size_plans_to_the_slice_and_tile_the_pass() {
    let mut coord = ClusterCoordinator::for_test(
        cfg_sync_cpu(2).total_samples(100).epoch_splits(4).batch_size(1),
    );
    let mut total = 0u64;
    for event in 0..4 {
        assert_eq!(coord.epoch_samples(event), 25, "event {event}");
        let plans = coord.plans_for_epoch(event);
        // Offsets are relative to the epoch's own slice, which is what
        // keeps ChunkPool partitioning [0, epoch_len) unchanged.
        let mut at = 0u64;
        for p in &plans {
            assert_eq!(p.partition_offset, at, "event {event} offsets stay consecutive");
            at += p.partition_size;
        }
        assert_eq!(at, 25, "event {event} plans tile its slice");
        total += at;
    }
    assert_eq!(total, 100, "the four events tile exactly one data pass");
}

#[test]
fn an_uneven_split_spreads_the_remainder_and_still_tiles() {
    // 100 picks over 7 events: 14 r 2, so the first two events carry one
    // extra pick each. No event may be starved and none may double up.
    let mut coord = ClusterCoordinator::for_test(
        cfg_sync_cpu(2).total_samples(100).epoch_splits(7).batch_size(1),
    );
    let sizes: Vec<usize> = (0..7).map(|e| coord.epoch_samples(e)).collect();
    assert_eq!(sizes, vec![15, 15, 14, 14, 14, 14, 14]);
    assert_eq!(sizes.iter().sum::<usize>(), 100);
    for (event, &want) in sizes.iter().enumerate() {
        let covered: u64 =
            coord.plans_for_epoch(event).iter().map(|p| p.partition_size).sum();
        assert_eq!(covered as usize, want, "event {event}");
    }
}

// A changed `epoch_splits` corrupts resume exactly as a changed seed does:
// the pool totals restore from the recorded EpochCoverage, but ranks map
// those offsets back to picks through the LIVE value, so the run would
// repeat covered data and skip uncovered data — silently. Refuse instead.
#[test]
fn resume_from_coverage_rejects_epoch_splits_mismatch() {
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;
    use crate::distributed::{CoverageBlock, EpochCoverage};

    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .progressive(true)
    .total_samples(100)
    .batch_size(1)
    .num_epochs(5)
    .epoch_splits(4)
    .seed(7);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.start_coverage = Some(CoverageBlock {
        seed: 7, // same seed, so only the splits differ
        epoch_splits: 2, // recorded under a DIFFERENT slicing
        batch_size: 1,
        per_epoch: vec![EpochCoverage {
            epoch: 0,
            total_samples: 50,
            uncovered_ranges: vec![(0, 50)],
        }],
    });
    let err = coord.resume_progressive_from_coverage().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("epoch_splits 2"), "must name both values: {msg}");
    assert!(msg.contains("epoch_splits 4"), "must name both values: {msg}");
}

// Matching splits resume normally — the guard must not fire on the
// everyday case, including the unsplit default.
#[test]
fn resume_from_coverage_accepts_matching_epoch_splits() {
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::AverageBackend;
    use crate::distributed::{CoverageBlock, EpochCoverage};

    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Cadence,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 4),
    )
    .progressive(true)
    .total_samples(100)
    .batch_size(1)
    .num_epochs(5)
    .epoch_splits(4)
    .seed(7);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.start_coverage = Some(CoverageBlock {
        seed: 7,
        epoch_splits: 4,
        batch_size: 1,
        per_epoch: vec![EpochCoverage {
            epoch: 0,
            total_samples: 25,
            uncovered_ranges: vec![(0, 25)],
        }],
    });
    assert!(
        coord.resume_progressive_from_coverage().expect("matching splits resume"),
        "a matching coverage block must reconstruct the pool",
    );
}
