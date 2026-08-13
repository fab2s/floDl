//! User-callback tests for the cluster coordinator: checkpoint dispatch
//! (targeted), eval callback bookkeeping, epoch_fn callback
//! bookkeeping, and the last-cycle slack producer
//! (`maybe_apply_callback_slack_for_next_cycle`).

use super::*;

// -----------------------------------------------------------------
// Checkpoint dispatch (targeted), result reporting, retry,
// time exclusion, and role-failover on rank death. The coord is
// the sole decider; the worker is a pure executor that reports
// back via `TimingMsgWire::CheckpointResult` and never decides
// policy locally.
// -----------------------------------------------------------------

/// Pure unit test: `handle_checkpoint_result` subtracts the
/// reported `elapsed_ms` from `wall_ms_accum[rank]` so ElChe's
/// rebalancer does not interpret checkpoint time as compute
/// slowness. Clamps at 0 to absorb fp drift.
#[test]
fn checkpoint_time_excluded_from_wall_ms_accum() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);
    // Pre-load wall time: rank 0 has 100 ms of training; rank 1
    // has 50 ms. After CheckpointResult(rank=0, elapsed_ms=30),
    // rank 0 should drop to 70 ms; rank 1 untouched.
    coord.set_wall_ms_accum_for_test(0, 100.0);
    coord.set_wall_ms_accum_for_test(1, 50.0);
    coord.handle_checkpoint_result(0, 7, 30.0, None);
    assert!(
        (coord.wall_ms_accum_for_test(0) - 70.0).abs() < 1e-9,
        "wall_ms_accum[0] = {} (expected 70.0)",
        coord.wall_ms_accum_for_test(0),
    );
    assert!(
        (coord.wall_ms_accum_for_test(1) - 50.0).abs() < 1e-9,
        "wall_ms_accum[1] = {} (expected 50.0 untouched)",
        coord.wall_ms_accum_for_test(1),
    );
    // EWMA seeded by first success.
    assert_eq!(coord.last_checkpoint_elapsed_ms_ewma(), Some(30.0));
    // Role stays put on success.
    assert_eq!(coord.checkpoint_role(), 0);
    // No tried entries on success.
    assert_eq!(coord.checkpoint_tried_count(7), 0);
}

/// Pure unit test: failure path adds rank to `tried_ranks[version]`,
/// picks next live untried rank, fails over `checkpoint_role`.
/// The actual send_control to the new role would fail under
/// for_test (no streams attached); we observe state, not network.
#[test]
fn checkpoint_failure_records_tried_and_failovers_role() {
    let world_size = 3usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);
    assert_eq!(coord.checkpoint_role(), 0);

    // Rank 0 reports failure: tried={0}, role moves to next live
    // (rank 1). send_control to rank 1 fails under for_test
    // (no streams) — that's fine; the test asserts state, not IO.
    coord.handle_checkpoint_result(0, 5, 12.0, Some("disk full".into()));
    assert_eq!(
        coord.checkpoint_role(),
        1,
        "role should fail over to rank 1"
    );
    assert_eq!(coord.checkpoint_tried_count(5), 1);

    // Rank 1 also fails: tried={0,1}, role moves to rank 2.
    coord.handle_checkpoint_result(1, 5, 8.0, Some("io error".into()));
    assert_eq!(coord.checkpoint_role(), 2);
    assert_eq!(coord.checkpoint_tried_count(5), 2);

    // Rank 2 also fails: no more live untried ranks → exhaust +
    // clear tried_ranks (the next cadence boundary starts fresh).
    coord.handle_checkpoint_result(2, 5, 5.0, Some("permission denied".into()));
    assert_eq!(
        coord.checkpoint_tried_count(5),
        0,
        "exhaustion should clear tried_ranks[version]"
    );
}

/// Pure unit test: success after a failure clears tried_ranks +
/// updates the sticky role to the successful rank.
#[test]
fn checkpoint_success_after_failure_clears_tried() {
    let world_size = 3usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);

    coord.handle_checkpoint_result(0, 4, 10.0, Some("oom".into()));
    assert_eq!(coord.checkpoint_tried_count(4), 1);
    assert_eq!(coord.checkpoint_role(), 1);

    // Rank 1 succeeds: tried[4] is cleared; role stays at 1.
    coord.handle_checkpoint_result(1, 4, 7.0, None);
    assert_eq!(coord.checkpoint_tried_count(4), 0);
    assert_eq!(coord.checkpoint_role(), 1);
    assert_eq!(coord.last_checkpoint_elapsed_ms_ewma(), Some(7.0));
}

/// Pure unit test: EWMA blends successive successes with alpha=0.3
/// (the framework's standard recent-value smoother).
#[test]
fn checkpoint_ewma_blends_successive_successes() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);

    coord.handle_checkpoint_result(0, 1, 100.0, None);
    assert_eq!(coord.last_checkpoint_elapsed_ms_ewma(), Some(100.0));

    // alpha=0.3: 0.3 * 50 + 0.7 * 100 = 15 + 70 = 85
    coord.handle_checkpoint_result(0, 2, 50.0, None);
    let ewma = coord.last_checkpoint_elapsed_ms_ewma().unwrap();
    assert!(
        (ewma - 85.0).abs() < 1e-9,
        "EWMA after 100 then 50 (alpha=0.3): got {ewma}, expected 85.0"
    );
}

// -----------------------------------------------------------------
// Eval callback — time exclusion + EWMA. Mirrors the checkpoint
// tests above. The user-facing `eval_result_fn` dispatch is covered
// by integration tests elsewhere; these tests pin the bookkeeping
// contract that ElChe's last-batch slack reservation will consume.
// -----------------------------------------------------------------

/// `handle_eval_result` subtracts the reported `elapsed_ms` from
/// `wall_ms_accum[rank]` so ElChe's rebalancer does not interpret
/// eval cost as compute slowness. Clamps at 0 to absorb fp drift.
#[test]
fn eval_time_excluded_from_wall_ms_accum() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.set_wall_ms_accum_for_test(0, 100.0);
    coord.set_wall_ms_accum_for_test(1, 50.0);
    coord.handle_eval_result(0, 3, 0.42, 30.0, None);
    assert!(
        (coord.wall_ms_accum_for_test(0) - 70.0).abs() < 1e-9,
        "wall_ms_accum[0] = {} (expected 70.0)",
        coord.wall_ms_accum_for_test(0),
    );
    assert!(
        (coord.wall_ms_accum_for_test(1) - 50.0).abs() < 1e-9,
        "wall_ms_accum[1] = {} (expected 50.0 untouched)",
        coord.wall_ms_accum_for_test(1),
    );
    assert_eq!(coord.last_eval_elapsed_ms_ewma(), Some(30.0));
}

/// Eval EWMA blends successive samples with alpha=0.3.
#[test]
fn eval_ewma_blends_successive_results() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);

    coord.handle_eval_result(0, 1, 0.1, 100.0, None);
    assert_eq!(coord.last_eval_elapsed_ms_ewma(), Some(100.0));

    // alpha=0.3: 0.3 * 50 + 0.7 * 100 = 85
    coord.handle_eval_result(0, 2, 0.2, 50.0, None);
    let ewma = coord.last_eval_elapsed_ms_ewma().unwrap();
    assert!(
        (ewma - 85.0).abs() < 1e-9,
        "EWMA after 100 then 50 (alpha=0.3): got {ewma}, expected 85.0"
    );
}

/// Eval errors still update the time-exclusion bookkeeping: the
/// closure ate wall time even when it returned an error. EWMA + the
/// `wall_ms_accum` subtract both fire; the user-facing
/// `eval_result_fn` is skipped (just logged).
#[test]
fn eval_error_still_excludes_time_and_updates_ewma() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.set_wall_ms_accum_for_test(0, 100.0);
    coord.handle_eval_result(0, 7, f64::NAN, 30.0, Some("boom".into()));
    assert!(
        (coord.wall_ms_accum_for_test(0) - 70.0).abs() < 1e-9,
        "wall_ms_accum[0] = {} (expected 70.0 even on error)",
        coord.wall_ms_accum_for_test(0),
    );
    assert_eq!(coord.last_eval_elapsed_ms_ewma(), Some(30.0));
}

// -----------------------------------------------------------------
// epoch_fn callback — time exclusion + EWMA. Same bookkeeping shape
// as eval / checkpoint; no user-facing dispatch (epoch_fn fires
// autonomously on the role rank, the coord only sees the post-fire
// wall-time report).
// -----------------------------------------------------------------

/// `handle_epoch_fn_elapsed` subtracts the reported `elapsed_ms`
/// from `wall_ms_accum[rank]` and updates
/// `last_epoch_fn_elapsed_ms_ewma`.
#[test]
fn epoch_fn_time_excluded_from_wall_ms_accum() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.set_wall_ms_accum_for_test(0, 100.0);
    coord.set_wall_ms_accum_for_test(1, 50.0);
    coord.handle_epoch_fn_elapsed(0, 20.0);
    assert!(
        (coord.wall_ms_accum_for_test(0) - 80.0).abs() < 1e-9,
        "wall_ms_accum[0] = {} (expected 80.0)",
        coord.wall_ms_accum_for_test(0),
    );
    assert!(
        (coord.wall_ms_accum_for_test(1) - 50.0).abs() < 1e-9,
        "wall_ms_accum[1] = {} (expected 50.0 untouched)",
        coord.wall_ms_accum_for_test(1),
    );
    assert_eq!(coord.last_epoch_fn_elapsed_ms_ewma(), Some(20.0));
}

/// epoch_fn EWMA blends successive samples with alpha=0.3.
#[test]
fn epoch_fn_ewma_blends_successive_reports() {
    let world_size = 2usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);

    coord.handle_epoch_fn_elapsed(0, 100.0);
    assert_eq!(coord.last_epoch_fn_elapsed_ms_ewma(), Some(100.0));

    // alpha=0.3: 0.3 * 50 + 0.7 * 100 = 85
    coord.handle_epoch_fn_elapsed(0, 50.0);
    let ewma = coord.last_epoch_fn_elapsed_ms_ewma().unwrap();
    assert!(
        (ewma - 85.0).abs() < 1e-9,
        "EWMA after 100 then 50 (alpha=0.3): got {ewma}, expected 85.0"
    );
}

// -----------------------------------------------------------------
// Last-cycle slack producer — `maybe_apply_callback_slack_for_next_cycle`
// tests. These pin the coord-side trigger that stages per-rank
// callback wall-time on ElChe just before the recompute that
// shapes the LAST cycle of an epoch.
// -----------------------------------------------------------------

/// Helper: build a 2-rank coord with anchor=10 ElChe, calibrate it
/// (rank 0 50ms/batch, rank 1 100ms/batch), set rank 0 as the
/// callback role for every kind, and install a chunk pool sized so
/// the next cycle exhausts the epoch.
fn build_coord_for_slack(remaining_batches: usize) -> ClusterCoordinator {
    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Sync,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 10),
    )
    .no_divergence_guard();
    let mut coord = ClusterCoordinator::for_test(cfg);
    // Calibrate ElChe: rank 0 fast, rank 1 slow → batch_counts [20, 10].
    coord
        .el_che_mut_for_test()
        .report_timing(&[500.0, 1000.0], &[10, 10], 10.0);
    // Rank 0 fires all callbacks.
    coord.set_callback_roles_for_test(0, 0, 0);
    // Install pool for the current epoch (epoch 0 by default).
    // batch_size defaults to 1, so total_samples == remaining batches.
    coord.install_chunk_pool_for_test(0, remaining_batches);
    coord.set_rank_epoch_for_test(0, 0);
    coord.set_rank_epoch_for_test(1, 0);
    coord
}

/// Happy path: epoch_fn EWMA known, next cycle is the last (pool
/// remaining ≤ sum batch_counts), guard passes. Slack lands on the
/// firing rank, no slack on the other.
#[test]
fn callback_slack_stages_on_firing_rank_for_last_cycle() {
    // sum(batch_counts) = 30 → remaining=25 is "next cycle is last".
    let mut coord = build_coord_for_slack(25);
    // Drive last_epoch_fn_elapsed_ms_ewma to 1000ms (well above
    // both the 100ms absolute floor and 5% * anchor_wall_ms = 50ms).
    coord.handle_epoch_fn_elapsed(0, 1000.0);
    coord.maybe_apply_callback_slack_for_test();
    let slack = coord.el_che_for_test().pending_callback_slack_ms();
    assert!(
        (slack[0] - 1000.0).abs() < 1e-9,
        "rank 0 should have staged epoch_fn slack of 1000ms; got {slack:?}",
    );
    assert_eq!(slack[1], 0.0, "non-firing rank slack must stay zero");
}

/// Pool still has plenty of batches → not the last cycle → no slack.
#[test]
fn callback_slack_skips_when_not_last_cycle() {
    // sum(batch_counts) = 30. remaining=100 means many cycles ahead.
    let mut coord = build_coord_for_slack(100);
    coord.handle_epoch_fn_elapsed(0, 1000.0);
    coord.maybe_apply_callback_slack_for_test();
    let slack = coord.el_che_for_test().pending_callback_slack_ms();
    assert_eq!(
        slack,
        &[0.0, 0.0],
        "slack must not stage when next cycle is not the last",
    );
}

/// Slack below the `max(0.05 * cycle_ms, 100ms)` floor must NOT be
/// staged — sub-threshold callbacks are noise relative to cycle
/// wall-time and shifting work for them adds churn without payoff.
#[test]
fn callback_slack_guard_filters_sub_threshold() {
    let mut coord = build_coord_for_slack(25);
    // 50ms < 100ms absolute floor (and < 50ms = 5% of anchor_wall_ms).
    coord.handle_epoch_fn_elapsed(0, 50.0);
    coord.maybe_apply_callback_slack_for_test();
    let slack = coord.el_che_for_test().pending_callback_slack_ms();
    assert_eq!(
        slack,
        &[0.0, 0.0],
        "sub-threshold slack must be filtered out (50ms < max(50, 100))",
    );
}

/// Empty pool → epoch already exhausted → no slack (the in-flight
/// cycle isn't "last", there's no next cycle in this epoch).
#[test]
fn callback_slack_skips_when_pool_empty() {
    let mut coord = build_coord_for_slack(0);
    coord.handle_epoch_fn_elapsed(0, 1000.0);
    coord.maybe_apply_callback_slack_for_test();
    let slack = coord.el_che_for_test().pending_callback_slack_ms();
    assert_eq!(slack, &[0.0, 0.0]);
}

/// Without a calibrated ElChe, the partition is uniform anyway and
/// the slack would have no meaningful reduction effect. Producer
/// short-circuits.
#[test]
fn callback_slack_skips_when_elche_uncalibrated() {
    // Build a coord without calibration scaffold.
    let world_size = 2;
    let cfg = ClusterCoordinatorConfig::new(
        ApplyPolicy::Sync,
        AverageBackend::Cpu,
        world_size,
        ElChe::new(world_size, 10),
    )
    .no_divergence_guard();
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.set_callback_roles_for_test(0, 0, 0);
    coord.install_chunk_pool_for_test(0, 5);
    coord.handle_epoch_fn_elapsed(0, 1000.0);
    coord.maybe_apply_callback_slack_for_test();
    let slack = coord.el_che_for_test().pending_callback_slack_ms();
    assert_eq!(
        slack,
        &[0.0, 0.0],
        "uncalibrated ElChe → no slack staging (partition is uniform anyway)",
    );
}

/// Per-rank isolation: `handle_epoch_fn_elapsed` on rank 1 should
/// not touch rank 0's `wall_ms_accum` even when the rank index is
/// the second slot. Defensive check against an off-by-one in the
/// rank → slot mapping.
#[test]
fn epoch_fn_per_rank_isolation() {
    let world_size = 3usize;
    let cfg = cfg_sync_cpu(world_size);
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.set_wall_ms_accum_for_test(0, 100.0);
    coord.set_wall_ms_accum_for_test(1, 200.0);
    coord.set_wall_ms_accum_for_test(2, 300.0);
    coord.handle_epoch_fn_elapsed(1, 25.0);
    assert!((coord.wall_ms_accum_for_test(0) - 100.0).abs() < 1e-9);
    assert!((coord.wall_ms_accum_for_test(1) - 175.0).abs() < 1e-9);
    assert!((coord.wall_ms_accum_for_test(2) - 300.0).abs() < 1e-9);
}

/// Integration test: dispatch is targeted — only the role rank
/// receives the `Checkpoint` frame; non-role ranks never see it.
/// NCCL config: the `Checkpoint` wire dispatch is NCCL-only (the CPU
/// path fires the callback controller-side at the reduce).
#[test]
fn checkpoint_dispatched_to_role_only() {
    let world_size = 2;
    let r0_got = Arc::new(AtomicBool::new(false));
    let r1_got = Arc::new(AtomicBool::new(false));
    let r0_flag = Arc::clone(&r0_got);
    let r1_flag = Arc::clone(&r1_got);

    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_nccl(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(2)
                .checkpoint_every(1)
        },
        move |coord| {
            coord.dispatch_epoch(0)?;
            coord.dispatch_epoch(1)?;
            // Pump ticks so frames flush.
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) {
                coord.tick()?;
                thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        },
    );

    fn drain_until_shutdown(
        saw_checkpoint: Arc<AtomicBool>,
        send_ack_for: u64,
    ) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            loop {
                let msg = recv_control(s, salt)?;
                match msg {
                    ControlMsgWire::Checkpoint {
                        version,
                        target_rank,
                    } => {
                        saw_checkpoint.store(true, Ordering::Relaxed);
                        if target_rank == send_ack_for {
                            send_metrics(s, salt, MetricsMsgWire::default()).ok();
                            send_timing(
                                s,
                                salt,
                                TimingMsgWire::CheckpointResult {
                                    rank: send_ack_for,
                                    version,
                                    elapsed_ms: 1.0,
                                    error: None,
                                },
                            )?;
                        }
                    }
                    ControlMsgWire::Shutdown | ControlMsgWire::ShutdownWithSave { .. } => {
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
    }
    let r0 = fake_rank(
        port,
        0,
        world_size as u32,
        TEST_SALT,
        drain_until_shutdown(r0_flag, 0),
    );
    let r1 = fake_rank(
        port,
        1,
        world_size as u32,
        TEST_SALT,
        drain_until_shutdown(r1_flag, u64::MAX /* never matches */),
    );

    r0.join().unwrap().expect("rank 0 drained cleanly");
    r1.join().unwrap().expect("rank 1 drained cleanly");
    coord_handle.join().unwrap().expect("coord finishes");

    assert!(
        r0_got.load(Ordering::Relaxed),
        "rank 0 (role) must receive Checkpoint frame"
    );
    assert!(
        !r1_got.load(Ordering::Relaxed),
        "rank 1 (non-role) must NOT receive Checkpoint frame"
    );
}

/// Role failover on rank death: when the heartbeat detector
/// declares the sticky `checkpoint_role` rank dead, the coord
/// picks the lowest live rank as the new role. The next cadence
/// boundary will dispatch there; no immediate redispatch fires
/// (the dead rank had no in-flight checkpoint to recover).
#[test]
fn checkpoint_role_failover_on_rank_death() {
    let world_size = 3usize;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    let cfg = cfg_sync_cpu(world_size).dead_ranks(Arc::clone(&dead_ranks));
    let mut coord = ClusterCoordinator::for_test(cfg);
    assert_eq!(coord.checkpoint_role(), 0);

    // Force rank 0's last_heartbeat to be older than the
    // heartbeat_timeout_secs threshold, then drive
    // check_dead_ranks. heartbeat_timeout_secs default ~ 30s;
    // setting last_heartbeat[0] to (now - 60s) is a safe margin.
    let stale = Instant::now() - Duration::from_secs(coord.heartbeat_timeout_secs() * 2 + 5);
    coord.set_last_heartbeat_for_test(0, stale);
    coord.check_dead_ranks_for_test();

    assert!(dead_ranks.is_dead(0), "rank 0 must be declared dead");
    assert_eq!(
        coord.checkpoint_role(),
        1,
        "checkpoint_role must fail over to next live rank (1)"
    );
}

// -----------------------------------------------------------------
// Cooperative-tier intent channel (Worker::request_eval /
// request_checkpoint): the request arrives as TimingMsgWire::Intent,
// sets a cohort-wide pending flag, and dispatch_epoch folds it into
// the role-elected dispatch at the next epoch boundary (then clears),
// independent of any configured cadence.
// -----------------------------------------------------------------

/// Pure unit test: an `Intent` frame sets the matching pending flag,
/// and the next `dispatch_epoch` (epoch > 0) folds + clears it — with
/// NO eval/checkpoint cadence configured, so the dispatch fires purely
/// from the folded intent.
#[test]
fn cooperative_intent_sets_and_folds() {
    use crate::distributed::wire::IntentKind;
    let world_size = 2;
    let cfg = cfg_sync_cpu(world_size)
        .total_samples(8)
        .batch_size(4)
        .num_epochs(3);
    let mut coord = ClusterCoordinator::for_test(cfg);

    assert!(!coord.pending_eval_intent_for_test());
    assert!(!coord.pending_checkpoint_intent_for_test());

    // Intents from any rank set the cohort-wide flags (the requesting
    // rank is irrelevant — the controller's policy elects where the
    // folded task runs).
    coord.process_timing_msg(TimingMsgWire::Intent {
        rank: 1,
        kind: IntentKind::EvalNow,
    });
    coord.process_timing_msg(TimingMsgWire::Intent {
        rank: 0,
        kind: IntentKind::CheckpointNow,
    });
    assert!(
        coord.pending_eval_intent_for_test(),
        "EvalNow intent must set the pending flag"
    );
    assert!(
        coord.pending_checkpoint_intent_for_test(),
        "CheckpointNow intent must set the pending flag"
    );

    // The next epoch boundary folds the EVAL intent (send_control is
    // best-effort without a live rank connection; the fold + clear runs
    // regardless). The CHECKPOINT intent is CPU-served at the next reduce
    // (`maybe_arm_checkpoint`), not at the boundary — the wire dispatch is
    // NCCL-only.
    let _ = coord.dispatch_epoch(1);
    assert!(
        !coord.pending_eval_intent_for_test(),
        "dispatch_epoch must fold + clear the eval intent"
    );
    assert!(
        coord.pending_checkpoint_intent_for_test(),
        "the CPU boundary leaves the checkpoint intent for the reduce"
    );
    // No checkpoint_fn and no save_path on this coord: the reduce-side
    // service point drops the request loudly rather than leaving it
    // pending forever.
    coord.rank_epoch = vec![1, 1];
    coord.maybe_arm_checkpoint();
    assert!(
        !coord.pending_checkpoint_intent_for_test(),
        "an unserviceable intent is dropped (loudly), not left pending"
    );
}
