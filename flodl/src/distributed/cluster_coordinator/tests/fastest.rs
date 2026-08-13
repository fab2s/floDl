//! EpochCallbackPolicy::Fastest dispatcher (coord-side runtime
//! resolution + sticky role retention + re-resolution on rank death +
//! targeted eval dispatch + epoch_fn role broadcast) plus LR-aware
//! meta-controller integration tests.

use super::*;

// -----------------------------------------------------------------
// EpochCallbackPolicy::Fastest dispatcher: coord-side
// runtime resolution + sticky role retention + re-resolution on
// rank death + targeted eval dispatch + epoch_fn role broadcast.
// The all-Some closure population (orchestrator-side fix) is
// exercised implicitly by the integration tests; the unit tests
// here drive the coord's state machine directly.
// -----------------------------------------------------------------

use crate::distributed::ddp_run::EpochCallbackPolicy;

fn cfg_sync_cpu_with_policy(
    world_size: usize,
    policy: EpochCallbackPolicy,
) -> ClusterCoordinatorConfig {
    cfg_sync_cpu(world_size).epoch_callback_policy(policy)
}

/// Pure unit test: `resolve_fastest_role` returns the live rank
/// with the lowest smoothed ms-per-batch. Seeded via ElChe's
/// `report_timing` — wall_ms / actual_batches → ms_per_batch
/// values land in the trust window.
#[test]
fn fastest_role_resolves_to_lowest_smoothed_ms() {
    let world_size = 3usize;
    let mut el_che = ElChe::new(world_size, 1);
    // wall=[100, 50, 200], batches=10 each → ms_per_batch =
    // [10, 5, 20] → rank 1 fastest.
    el_che.report_timing(&[100.0, 50.0, 200.0], &[10, 10, 10], 0.0);
    let cfg =
        ClusterCoordinatorConfig::new(ApplyPolicy::Sync, AverageBackend::Cpu, world_size, el_che)
            .no_divergence_guard()
            .epoch_callback_policy(EpochCallbackPolicy::Fastest);
    let coord = ClusterCoordinator::for_test(cfg);
    assert_eq!(
        coord.resolve_fastest_role_for_test(),
        1,
        "rank 1 has lowest ms_per_batch",
    );
}

/// Pure unit test: `resolve_fastest_role` falls back to the
/// lowest live rank when ElChe has no calibrated samples yet
/// (Probe phase — every rank's smoothed window is empty).
#[test]
fn fastest_role_fallback_to_lowest_live_when_uncalibrated() {
    let world_size = 3usize;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    // Declare rank 0 dead before resolution to verify fallback
    // picks the lowest LIVE rank, not literal rank 0.
    dead_ranks.declare_dead(0);
    let cfg = cfg_sync_cpu_with_policy(world_size, EpochCallbackPolicy::Fastest)
        .dead_ranks(Arc::clone(&dead_ranks));
    let coord = ClusterCoordinator::for_test(cfg);
    assert_eq!(
        coord.resolve_fastest_role_for_test(),
        1,
        "uncalibrated + rank 0 dead → lowest live rank (1)",
    );
}

/// Pure unit test: under Fastest policy, when the current role
/// rank dies, `re_resolve_callback_roles_on_death` updates all
/// three role fields against the new live set + flips
/// `epoch_role_dirty` so the next dispatch broadcasts.
#[test]
fn fastest_re_resolve_on_role_rank_death() {
    let world_size = 3usize;
    let mut el_che = ElChe::new(world_size, 1);
    el_che.report_timing(&[100.0, 50.0, 200.0], &[10, 10, 10], 0.0);
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    let cfg =
        ClusterCoordinatorConfig::new(ApplyPolicy::Sync, AverageBackend::Cpu, world_size, el_che)
            .no_divergence_guard()
            .epoch_callback_policy(EpochCallbackPolicy::Fastest)
            .dead_ranks(Arc::clone(&dead_ranks));
    let mut coord = ClusterCoordinator::for_test(cfg);
    // Force role resolution by manually setting roles to the
    // fastest rank (the for_test constructor seeds them to 0; in
    // production the first dispatch broadcasts). After that,
    // declare rank 1 dead + invoke re-resolve.
    coord.set_callback_roles_for_test(1, 1, 1);
    dead_ranks.declare_dead(1);
    coord.re_resolve_callback_roles_on_death_for_test(1);
    // With rank 1 dead, remaining smoothed values are rank 0
    // (10 ms) and rank 2 (20 ms). Rank 0 wins.
    assert_eq!(coord.checkpoint_role(), 0);
    assert_eq!(coord.eval_role_for_test(), 0);
    assert_eq!(coord.epoch_callback_role_for_test(), 0);
    assert!(
        coord.epoch_role_dirty_for_test(),
        "role change must flip the dirty flag",
    );
}

/// Pure unit test: under `Rank(n)` policy (the default), rank
/// death does NOT trigger a Fastest re-resolve; the eval and
/// epoch_callback roles stay pinned. (The checkpoint-role-only
/// failover still applies for Rank policy; covered by
/// `checkpoint_role_failover_on_rank_death`.)
#[test]
fn rank_policy_skips_fastest_re_resolve() {
    let world_size = 3usize;
    let cfg = cfg_sync_cpu_with_policy(world_size, EpochCallbackPolicy::Rank(2));
    let mut coord = ClusterCoordinator::for_test(cfg);
    coord.set_callback_roles_for_test(2, 2, 2);
    // Call re-resolve directly with rank 1 (not the role) dying;
    // since policy != Fastest, function must early-return without
    // touching role fields.
    coord.re_resolve_callback_roles_on_death_for_test(1);
    assert_eq!(coord.eval_role_for_test(), 2);
    assert_eq!(coord.epoch_callback_role_for_test(), 2);
}

/// Integration test: `ExecuteEvalCallback` is dispatched ONLY to
/// the rank named in `target_rank`. Non-role ranks never see the
/// frame on their stream.
#[test]
fn eval_dispatched_to_role_only() {
    let world_size = 2;
    let r0_got = Arc::new(AtomicBool::new(false));
    let r1_got = Arc::new(AtomicBool::new(false));
    let r0_flag = Arc::clone(&r0_got);
    let r1_flag = Arc::clone(&r1_got);

    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_cpu(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(2)
                .eval_every_epochs(1)
        },
        move |coord| {
            coord.dispatch_epoch(0)?;
            coord.dispatch_epoch(1)?;
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) {
                coord.tick()?;
                thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        },
    );

    fn drain_eval(
        saw_eval: Arc<AtomicBool>,
    ) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            loop {
                let msg = recv_control(s, salt)?;
                match msg {
                    ControlMsgWire::ExecuteEvalCallback { .. } => {
                        saw_eval.store(true, Ordering::Relaxed);
                    }
                    ControlMsgWire::Shutdown | ControlMsgWire::ShutdownWithSave { .. } => {
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, drain_eval(r0_flag));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, drain_eval(r1_flag));

    r0.join().unwrap().expect("rank 0 drained cleanly");
    r1.join().unwrap().expect("rank 1 drained cleanly");
    coord_handle.join().unwrap().expect("coord finishes");

    assert!(
        r0_got.load(Ordering::Relaxed),
        "rank 0 (eval_role default) must receive ExecuteEvalCallback"
    );
    assert!(
        !r1_got.load(Ordering::Relaxed),
        "rank 1 (non-role) must NOT receive ExecuteEvalCallback"
    );
}

/// Integration test: `SetEpochCallbackRole` is broadcast to every
/// rank BEFORE the first `StartEpoch`, so workers have a definite
/// role before they could fire `epoch_fn`. Subsequent dispatches
/// for the same role do NOT re-broadcast (sticky retention).
#[test]
fn epoch_callback_role_broadcast_at_first_dispatch() {
    let world_size = 2;
    let r0_role_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let r1_role_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let r0_role = Arc::clone(&r0_role_count);
    let r1_role = Arc::clone(&r1_role_count);

    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_cpu(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(3)
        },
        move |coord| {
            coord.dispatch_epoch(0)?;
            coord.dispatch_epoch(1)?;
            coord.dispatch_epoch(2)?;
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) {
                coord.tick()?;
                thread::sleep(Duration::from_millis(5));
            }
            Ok(())
        },
    );

    fn drain_role(
        role_count: Arc<std::sync::atomic::AtomicUsize>,
    ) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            loop {
                let msg = recv_control(s, salt)?;
                match msg {
                    ControlMsgWire::SetEpochCallbackRole { rank: _ } => {
                        role_count.fetch_add(1, Ordering::Relaxed);
                    }
                    ControlMsgWire::Shutdown | ControlMsgWire::ShutdownWithSave { .. } => {
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, drain_role(r0_role));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, drain_role(r1_role));

    r0.join().unwrap().expect("rank 0 drained cleanly");
    r1.join().unwrap().expect("rank 1 drained cleanly");
    coord_handle.join().unwrap().expect("coord finishes");

    // Each rank sees exactly ONE SetEpochCallbackRole across 3
    // dispatch_epoch calls — sticky retention means the dirty
    // flag stays cleared after the first broadcast.
    assert_eq!(
        r0_role_count.load(Ordering::Relaxed),
        1,
        "rank 0 must see SetEpochCallbackRole exactly once",
    );
    assert_eq!(
        r1_role_count.load(Ordering::Relaxed),
        1,
        "rank 1 must see SetEpochCallbackRole exactly once",
    );
}

/// Integration test: a full success round-trip — rank reports
/// CheckpointResult{ok}; coord updates EWMA + clears tried set.
/// NCCL config: the `Checkpoint` wire dispatch is NCCL-only (the CPU
/// path fires the callback controller-side at the reduce).
#[test]
fn checkpoint_success_round_trip_updates_ewma() {
    let world_size = 2;
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
            let start = Instant::now();
            while coord.last_checkpoint_elapsed_ms_ewma().is_none() {
                if start.elapsed() > Duration::from_secs(2) {
                    return Err(TensorError::new(
                        "checkpoint_success_round_trip: no EWMA seeded",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(5));
            }
            assert!(
                (coord.last_checkpoint_elapsed_ms_ewma().unwrap() - 12.5).abs() < 1e-9,
                "EWMA = {:?} (expected 12.5)",
                coord.last_checkpoint_elapsed_ms_ewma(),
            );
            assert_eq!(coord.checkpoint_tried_count(1), 0);
            assert_eq!(coord.checkpoint_role(), 0);
            Ok(())
        },
    );

    fn rank_body(rank: u64) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            loop {
                let msg = recv_control(s, salt)?;
                match msg {
                    ControlMsgWire::Checkpoint {
                        version,
                        target_rank,
                    } if target_rank == rank => {
                        send_timing(
                            s,
                            salt,
                            TimingMsgWire::CheckpointResult {
                                rank,
                                version,
                                elapsed_ms: 12.5,
                                error: None,
                            },
                        )?;
                    }
                    ControlMsgWire::Shutdown | ControlMsgWire::ShutdownWithSave { .. } => {
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
    }

    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, rank_body(0));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, rank_body(1));
    r0.join().unwrap().expect("rank 0 success round-trip");
    r1.join().unwrap().expect("rank 1 drains cleanly");
    coord_handle.join().unwrap().expect("coord finishes");
}

#[test]
fn dispatch_epoch_errors_when_total_samples_zero() {
    // Forgetting to set total_samples on the config must surface
    // as a loud error, not silently produce empty partitions.
    let world_size = 2;
    let (listener, _port) =
        ClusterCoordinator::bind(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)).unwrap();
    let coord_thread = thread::spawn(move || -> Result<()> {
        let mut coord = ClusterCoordinator::start_from_listener(
            listener,
            TEST_SALT,
            cfg_sync_nccl(world_size),
        )?;
        let err = coord.dispatch_epoch(0).unwrap_err();
        assert!(
            err.to_string().contains("total_samples > 0"),
            "expected total_samples guard, got: {err}"
        );
        let _ = coord.shutdown();
        Ok(())
    });
    // No real coord port handshake here — bind happened on the
    // listener that lives inside the coord thread. We need
    // 2 ranks to handshake before start_from_listener returns.
    let port = _port;
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_, _| Ok(()));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |_, _| Ok(()));
    let _ = r0.join();
    let _ = r1.join();
    coord_thread.join().unwrap().expect("coord error path");
}

// -----------------------------------------------------------------
// LR-aware meta-controller
// -----------------------------------------------------------------

#[test]
fn meta_controller_default_is_on() {
    // Opt-out semantics: a config built without an explicit
    // `.meta_controller(...)` call must produce a coordinator with the
    // meta ENABLED. LR drops are always worth catching; callers
    // collecting an unconditioned trajectory opt out via
    // `.meta_controller(false)`.
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_nccl(world_size),
        |coord| {
            assert!(
                coord.meta_controller_enabled_for_test(),
                "meta_controller must default to ON"
            );
            let lrs = coord.last_lr_per_rank_for_test();
            assert_eq!(lrs.len(), 2);
            assert!(lrs.iter().all(|x| x.is_none()), "cold-start LRs = None");
            Ok(())
        },
    );
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_, _| Ok(()));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |_, _| Ok(()));
    r0.join().unwrap().expect("rank 0 handshake");
    r1.join().unwrap().expect("rank 1 handshake");
    coord_handle.join().unwrap().expect("coord drives clean");
}

#[test]
fn meta_controller_builder_enables_meta() {
    // `.meta_controller(true)` on the config must produce a
    // coordinator with `lr_event_meta = Some(_)`.
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_nccl(world_size).meta_controller(true),
        |coord| {
            assert!(
                coord.meta_controller_enabled_for_test(),
                "meta_controller(true) must produce an active meta"
            );
            Ok(())
        },
    );
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_, _| Ok(()));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |_, _| Ok(()));
    r0.join().unwrap().expect("rank 0 handshake");
    r1.join().unwrap().expect("rank 1 handshake");
    coord_handle.join().unwrap().expect("coord drives clean");
}

#[test]
fn lr_update_frame_populates_last_lr_per_rank() {
    // Worker-side `TimingMsg::LrUpdate` is forwarded as
    // `TimingMsgWire::LrUpdate` over the control channel. The coord
    // captures the most-recent LR per rank in `last_lr_per_rank`,
    // which is what `observe_meta` reads each averaging cycle.
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_nccl(world_size).meta_controller(true),
        |coord| {
            // Drain reader threads until both LRs have arrived.
            let start = Instant::now();
            while coord
                .last_lr_per_rank_for_test()
                .iter()
                .any(|lr| lr.is_none())
            {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "lr_update_frame_populates_last_lr_per_rank: \
                         LRs never arrived from fake ranks within 5s",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            let lrs = coord.last_lr_per_rank_for_test();
            assert_eq!(lrs[0], Some(0.01));
            assert_eq!(lrs[1], Some(0.02));
            Ok(())
        },
    );
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
        send_timing(s, salt, TimingMsgWire::LrUpdate { rank: 0, lr: 0.01 })?;
        // Stay alive long enough for the coord to drain the frame.
        thread::sleep(Duration::from_millis(200));
        Ok(())
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        send_timing(s, salt, TimingMsgWire::LrUpdate { rank: 1, lr: 0.02 })?;
        thread::sleep(Duration::from_millis(200));
        Ok(())
    });
    r0.join().unwrap().expect("rank 0 sends LrUpdate");
    r1.join().unwrap().expect("rank 1 sends LrUpdate");
    coord_handle
        .join()
        .unwrap()
        .expect("coord captures both LRs");
}
