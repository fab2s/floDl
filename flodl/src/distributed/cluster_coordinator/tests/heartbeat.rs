//! Heartbeat + dead-rank detection tests.

use super::*;

// -----------------------------------------------------------------
// Heartbeat + dead-rank detection
// -----------------------------------------------------------------

#[test]
fn heartbeat_stale_declares_rank_dead_and_unblocks_should_average() {
    // 3 ranks share a DeadRanks ledger with the coord. Ranks 0 and
    // 1 each send a Batch + SyncAck (a complete averaging cycle).
    // Rank 2 handshakes the control channel but emits no frames
    // at all — its `last_heartbeat` slot never updates past the
    // initial `Instant::now()` from coord construction, so once
    // the configured `heartbeat_timeout_secs` elapses `check_dead_ranks`
    // marks rank 2 dead. `should_average`'s `.filter(!is_dead)`
    // then lets the cycle fire with just ranks 0 & 1, and
    // `poll_cpu_averaging`'s "all-alive-acked" gate finalizes.
    let world_size = 3;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    let dead_for_coord = Arc::clone(&dead_ranks);
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Cpu,
                world_size,
                ElChe::new(world_size, 1),
            )
            .no_divergence_guard()
            .dead_ranks(dead_for_coord)
            // 1-second window so the test runs in ~1.5s rather than
            // the 30s production default.
            .heartbeat_timeout_secs(1)
        },
        |coord| {
            let start = Instant::now();
            while coord.avg_count() == 0 {
                if start.elapsed() > Duration::from_secs(10) {
                    return Err(TensorError::new(
                        "heartbeat_stale: avg_count never advanced",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(20));
            }
            assert!(
                coord.avg_count() >= 1,
                "cycle finalized with surviving ranks"
            );
            Ok(())
        },
    );

    // Rank 2: handshake only, then sleep without sending any
    // frames. Coord's heartbeat-staleness check will trigger.
    // Keep alive long enough for the cycle to complete.
    let dead_for_assertion = Arc::clone(&dead_ranks);
    let r2 = fake_rank(port, 2, world_size as u32, TEST_SALT, move |_s, _salt| {
        thread::sleep(Duration::from_millis(3500));
        Ok(())
    });

    let body = |rank: u64| {
        move |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            send_timing(
                s,
                salt,
                TimingMsgWire::Batch {
                    rank,
                    batch_ms: 10.0,
                    step_count: 1,
                    param_norm: None,
                    batch_loss: 0.5,
                    sync_divergence: None,
                },
            )?;
            let _ = recv_control(s, salt)?; // RequestParams
            send_timing(
                s,
                salt,
                TimingMsgWire::SyncAck {
                    rank,
                    step_count: 2,
                    divergence: Some(0.05),
                    post_norm: Some(1.0),
                    pre_norm: Some(1.05),
                },
            )?;
            let _ = recv_control(s, salt)?; // Update
            let _ = recv_control(s, salt)?; // SetGlobalStep
            Ok(())
        }
    };
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, body(0));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, body(1));
    r0.join().unwrap().expect("rank 0 completes averaging");
    r1.join().unwrap().expect("rank 1 completes averaging");
    let _ = r2.join();
    coord_handle.join().unwrap().expect("coord drives clean");

    // Post-hoc invariant: the shared ledger registers rank 2 dead.
    // Don't assert about ranks 0/1: by the time the test's r2
    // sleep elapses (3.5s) they've been silent past the 1s
    // threshold too and may also have been declared dead. That's
    // not a regression — it's the heartbeat detector doing its
    // job on what looks like additional dead ranks once the run
    // is wrapping up. The load-bearing invariant is that rank 2
    // (the silent one DURING the cycle) was detected as dead.
    assert!(
        dead_for_assertion.is_dead(2),
        "rank 2 must be flagged dead in shared ledger"
    );
}

#[test]
fn dead_rank_remainder_redistributed_via_extend_partition() {
    // 3-rank Sync+Cpu setup with dispatched epoch + shared
    // DeadRanks ledger + 1s heartbeat timeout. Ranks 0 + 1 send
    // ONE Batch each (representing one batch of training before
    // the failure), then rank 2 goes silent. The coord's
    // heartbeat-stale check declares rank 2 dead; the
    // redistribution path computes rank 2's un-processed
    // remainder (its full partition: 0 batches processed) and
    // emits one ExtendPartition frame to each survivor. The
    // survivors receive the frames over the wire.
    //
    // Invariant: total samples reshard = rank 2's partition size.
    // Each survivor's received slice sums with the others to
    // exactly that count — no samples lost, no samples duplicated.
    let world_size = 3;
    let total_samples = 30;
    let batch_size = 1;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    let dead_for_coord = Arc::clone(&dead_ranks);
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Cpu,
                world_size,
                ElChe::new(world_size, 1),
            )
            .no_divergence_guard()
            .dead_ranks(dead_for_coord)
            .heartbeat_timeout_secs(1)
            .total_samples(total_samples)
            .batch_size(batch_size)
            .num_epochs(1)
        },
        |coord| {
            coord.dispatch_epoch(0)?;
            let start = Instant::now();
            while !coord.dead_ranks.as_ref().unwrap().is_dead(2) {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "dead_rank_redistribute: rank 2 never declared dead",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        },
    );

    // Ranks 0 + 1 collect StartEpoch + count any ExtendPartition
    // they receive (size). Returned through a shared atomic
    // because fake_rank's body returns Result<()>.
    use std::sync::atomic::AtomicU64;
    let r0_extension = Arc::new(AtomicU64::new(0));
    let r1_extension = Arc::new(AtomicU64::new(0));
    let r0_acc = Arc::clone(&r0_extension);
    let r1_acc = Arc::clone(&r1_extension);

    let make_alive = |rank: u64, acc: Arc<AtomicU64>| {
        move |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            let mut received_start_epoch = false;
            let read_deadline = Instant::now() + Duration::from_secs(4);
            while Instant::now() < read_deadline {
                s.set_read_timeout(Some(Duration::from_millis(200))).ok();
                match ControlFrame::read_from(s, salt) {
                    Ok(Some(frame)) => match frame.decode::<ControlMsgWire>() {
                        Ok(ControlMsgWire::StartEpoch(_)) => {
                            received_start_epoch = true;
                            send_timing(
                                s,
                                salt,
                                TimingMsgWire::Batch {
                                    rank,
                                    batch_ms: 5.0,
                                    step_count: 1,
                                    param_norm: None,
                                    batch_loss: 0.1,
                                    sync_divergence: None,
                                },
                            )?;
                        }
                        Ok(ControlMsgWire::ExtendPartition {
                            partition_size,
                            ..
                        }) => {
                            acc.fetch_add(partition_size, Ordering::SeqCst);
                        }
                        Ok(_other) => {
                            // RequestParams / SetGlobalStep / etc.
                        }
                        Err(_) => break,
                    },
                    Ok(None) => break,
                    Err(_) => continue,
                }
            }
            assert!(received_start_epoch, "rank {rank} got StartEpoch");
            Ok(())
        }
    };

    // Rank 2 handshakes then sleeps — coord will declare it dead.
    let r2 = fake_rank(port, 2, world_size as u32, TEST_SALT, |_s, _salt| {
        thread::sleep(Duration::from_millis(3500));
        Ok(())
    });
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, make_alive(0, r0_acc));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, make_alive(1, r1_acc));

    r0.join().unwrap().expect("rank 0 path");
    r1.join().unwrap().expect("rank 1 path");
    let _ = r2.join();
    coord_handle.join().unwrap().expect("coord drives clean");

    // Rank 2's partition for a 3-rank, 30-sample equal-split is
    // 10 samples. Since rank 2 sent no Batch (its
    // last_step_count stayed at the epoch-start snapshot),
    // processed_samples = 0, so the entire partition (10) is
    // redistributed across the 2 survivors. Sum across survivors
    // must equal 10 exactly.
    let r0_total = r0_extension.load(Ordering::SeqCst);
    let r1_total = r1_extension.load(Ordering::SeqCst);
    let total_redistributed = r0_total + r1_total;
    assert_eq!(
        total_redistributed, 10,
        "dead rank 2's un-processed remainder (10) must be reshared \
         across survivors; got r0={r0_total}, r1={r1_total}"
    );
}

#[test]
fn dead_ranks_optional_default_disables_elastic_membership() {
    // When `dead_ranks` is None in the config (default), the
    // heartbeat-stale check is a no-op and no rank can be
    // declared dead. Same standard cycle as the basic Sync+CPU
    // test, just verifying the opt-in semantics: forgetting to
    // wire the ledger doesn't accidentally declare ranks dead.
    let world_size = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_cpu(world_size).heartbeat_timeout_secs(0)
            // No `.dead_ranks(...)` — elastic membership disabled.
        },
        move |coord| {
            // Drive a cycle; sleep enough that timeout-0 WOULD
            // declare every rank dead if elastic membership were
            // active. Verify nothing fires.
            thread::sleep(Duration::from_millis(50));
            coord.tick()?;
            // Still active; no decrement.
            assert_eq!(
                coord.active_count(),
                world_size,
                "dead-rank detection must be off without ledger"
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
fn observe_meta_runs_during_averaging_cycle_no_anchor_change_in_probe() {
    // Drive one averaging cycle on Sync+NCCL with meta enabled.
    // Each rank reports its LR + one Batch; coord triggers cycle 1.
    // observe_meta runs inside finish_averaging_nccl but the meta
    // is in Probe phase on the first cycle (no calibration), so
    // it returns Noop and the ElChe anchor stays at 1.
    // This guards the wire integrity: enabling the meta must not
    // crash or spuriously nudge the anchor at startup.
    let world_size = 2;
    let initial_anchor = 1usize;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_nccl(world_size).meta_controller(true),
        move |coord| {
            let start = Instant::now();
            while coord.avg_count() == 0 {
                if start.elapsed() > Duration::from_secs(5) {
                    return Err(TensorError::new(
                        "observe_meta_runs: avg_count never advanced",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(10));
            }
            assert_eq!(
                coord.el_che().anchor(),
                initial_anchor,
                "Probe-phase meta must NOT nudge the anchor on first cycle"
            );
            // LR was observed even though no MetaAction fired.
            let lrs = coord.last_lr_per_rank_for_test();
            assert!(lrs.iter().all(|lr| lr.is_some()), "LRs captured");
            Ok(())
        },
    );
    let body = |rank: u32| {
        let rank = rank as u64;
        move |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            send_timing(
                s,
                salt,
                TimingMsgWire::LrUpdate { rank, lr: 0.01 },
            )?;
            send_timing(
                s,
                salt,
                TimingMsgWire::Batch {
                    rank,
                    batch_ms: 10.0,
                    step_count: 1,
                    param_norm: None,
                    batch_loss: 1.0,
                    sync_divergence: None,
                },
            )?;
            // Wait for SyncNow + SetGlobalStep then ack with SyncAck.
            let _ = recv_control(s, salt)?;
            send_timing(
                s,
                salt,
                TimingMsgWire::SyncAck {
                    rank,
                    step_count: 2,
                    divergence: Some(0.05),
                    post_norm: Some(1.0),
                    pre_norm: Some(1.05),
                },
            )?;
            let _ = recv_control(s, salt)?; // SetGlobalStep
            Ok(())
        }
    };
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, body(0));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, body(1));
    r0.join().unwrap().expect("rank 0 path");
    r1.join().unwrap().expect("rank 1 path");
    coord_handle.join().unwrap().expect("coord cycle 1 with meta on");
}

#[test]
fn max_failure_threshold_breach_dispatches_shutdown_with_save() {
    // 3-rank CPU cluster, max_failure=Absolute(1). All ranks
    // handshake then go silent — within heartbeat_timeout_secs the
    // first stale-heartbeat detection trips the threshold, the
    // coord broadcasts ShutdownWithSave to every rank, and the
    // dispatched flag flips.
    let world_size = 3;
    let dead_ranks =
        crate::distributed::controller::DeadRanks::new(world_size);
    let dead_for_coord = Arc::clone(&dead_ranks);
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Cpu,
                world_size,
                ElChe::new(world_size, 1),
            )
            .no_divergence_guard()
            .dead_ranks(dead_for_coord)
            .heartbeat_timeout_secs(1)
            .max_failure(
                crate::distributed::max_failure::MaxFailureThreshold::Absolute(1),
            )
        },
        |coord| {
            let start = Instant::now();
            while !coord.shutdown_with_save_dispatched() {
                if start.elapsed() > Duration::from_secs(10) {
                    return Err(TensorError::new(
                        "max_failure: ShutdownWithSave never dispatched",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        },
    );

    fn drain_shutdown_with_save(
        s: &mut TcpStream,
        salt: &SessionSalt,
    ) -> Result<()> {
        // Bound the wait — the coord's heartbeat_timeout=1s means
        // dispatch lands ~1.0-1.5s after handshake.
        s.set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| TensorError::new(&format!("timeout: {e}")))?;
        let msg = recv_control(s, salt)?;
        match msg {
            ControlMsgWire::ShutdownWithSave { reason } => {
                let r = crate::distributed::SaveReason::from_u8(reason)
                    .expect("known SaveReason variant");
                if r != crate::distributed::SaveReason::MaxFailureExceeded {
                    return Err(TensorError::new(&format!(
                        "expected MaxFailureExceeded, got {r:?}"
                    )));
                }
                Ok(())
            }
            other => Err(TensorError::new(&format!(
                "expected ShutdownWithSave, got {other:?}"
            ))),
        }
    }

    let r0 = fake_rank(
        port,
        0,
        world_size as u32,
        TEST_SALT,
        drain_shutdown_with_save,
    );
    let r1 = fake_rank(
        port,
        1,
        world_size as u32,
        TEST_SALT,
        drain_shutdown_with_save,
    );
    let r2 = fake_rank(
        port,
        2,
        world_size as u32,
        TEST_SALT,
        drain_shutdown_with_save,
    );
    r0.join().unwrap().expect("rank 0 receives ShutdownWithSave");
    r1.join().unwrap().expect("rank 1 receives ShutdownWithSave");
    r2.join().unwrap().expect("rank 2 receives ShutdownWithSave");
    coord_handle.join().unwrap().expect("coord dispatched broadcast");
}

#[test]
fn controller_writes_meta_json_on_shutdown_with_save() {
    // 3-rank cluster with `save_path` configured. Force a
    // max_failure breach so the coord calls
    // `dispatch_shutdown_with_save`. Assert the controller wrote
    // `<save_path>.meta.json` with the expected reason +
    // world_size + ElCheState present. The `.fdl` + `.optim`
    // bundle members are the workers' responsibility (covered by
    // the worker-side `shutdown_with_save_writes_model_and_optim_*`
    // test) — only `.meta.json` is asserted here.
    let world_size = 3;
    let dir = std::env::temp_dir().join(format!(
        "flodl_coord_meta_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("coord_ckpt");
    let stem_str = stem.to_str().unwrap().to_string();

    let dead_ranks =
        crate::distributed::controller::DeadRanks::new(world_size);
    let dead_for_coord = Arc::clone(&dead_ranks);
    let stem_for_coord = stem_str.clone();
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Cpu,
                world_size,
                ElChe::new(world_size, 3),
            )
            .no_divergence_guard()
            .dead_ranks(dead_for_coord)
            .heartbeat_timeout_secs(1)
            .max_failure(
                crate::distributed::max_failure::MaxFailureThreshold::Absolute(1),
            )
            .save_path(stem_for_coord.clone())
        },
        |coord| {
            let start = Instant::now();
            while !coord.shutdown_with_save_dispatched() {
                if start.elapsed() > Duration::from_secs(10) {
                    return Err(TensorError::new(
                        "coord meta: ShutdownWithSave never dispatched",
                    ));
                }
                coord.tick()?;
                thread::sleep(Duration::from_millis(20));
            }
            Ok(())
        },
    );

    // Fake ranks: handshake, then go silent. Coord's heartbeat
    // timeout (1s) trips max_failure (1) → dispatch_shutdown_with_save
    // fires → meta.json gets written.
    let r0 = fake_rank(
        port,
        0,
        world_size as u32,
        TEST_SALT,
        |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            s.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let _ = recv_control(s, salt)?; // ShutdownWithSave
            Ok(())
        },
    );
    let r1 = fake_rank(
        port,
        1,
        world_size as u32,
        TEST_SALT,
        |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            s.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let _ = recv_control(s, salt)?;
            Ok(())
        },
    );
    let r2 = fake_rank(
        port,
        2,
        world_size as u32,
        TEST_SALT,
        |s: &mut TcpStream, salt: &SessionSalt| -> Result<()> {
            s.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let _ = recv_control(s, salt)?;
            Ok(())
        },
    );
    r0.join().unwrap().expect("rank 0 path");
    r1.join().unwrap().expect("rank 1 path");
    r2.join().unwrap().expect("rank 2 path");
    coord_handle.join().unwrap().expect("coord dispatched");

    let meta_path =
        crate::distributed::CheckpointBundle::meta_path(&stem_str);
    assert!(
        meta_path.exists(),
        "controller meta.json missing at {}",
        meta_path.display(),
    );
    let meta =
        crate::distributed::CheckpointMeta::read_from_file(&meta_path)
            .expect("controller-written meta parses");
    assert_eq!(meta.world_size_at_save, world_size);
    assert_eq!(
        meta.save_reason,
        crate::distributed::SaveReason::MaxFailureExceeded,
    );
    // ElCheState present and reflects coord's ElChe trajectory.
    let state = meta
        .elche_state
        .expect("controller writes elche_state into meta");
    assert_eq!(state.anchor, 3);
    assert_eq!(state.smoothed_ms_per_batch.len(), world_size);

    std::fs::remove_dir_all(&dir).ok();
}

// Rendezvous retry: the seeded generator (rank 0) is marked dead
// via the shared DeadRanks ledger, so `check_rendezvous_timeout`
// fires on the next tick and the coord retries from rank 1. Both
// the wire frame (rank 1 receives RequestNewNcclId) and the
// internal state (pending.generator_rank == 1, tried_generators ==
// [0]) are exercised.
#[test]
fn rendezvous_retry_picks_next_survivor_on_generator_death() {
    let world_size = 3;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    let dead_for_coord = Arc::clone(&dead_ranks);
    let dead_for_test = Arc::clone(&dead_ranks);
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_nccl(world_size)
                .dead_ranks(dead_for_coord)
                .heartbeat_timeout_secs(60)
                .rendezvous_timeout_secs(60)
        },
        move |coord| {
            // Seed a pending rendezvous where rank 0 was the
            // initially-picked generator; the retry path must skip
            // it (dead) and reach rank 1 next.
            coord.test_seed_rendezvous_pending(0, vec![0, 1, 2], 0);
            dead_for_test.declare_dead(0);
            coord.tick()?; // fires check_rendezvous_timeout
            assert_eq!(
                coord.rendezvous_pending_generator(),
                Some(1),
                "retry must pick rank 1 (next ascending survivor)"
            );
            assert_eq!(
                coord.rendezvous_tried_generators(),
                vec![0],
                "rank 0 recorded as tried"
            );
            Ok(())
        },
    );

    // Rank 0 mimics having died — handshakes (so the coord's
    // accept loop unblocks) then exits. Coord's send to rank 0
    // happens at seed time, which is before this rank exits; the
    // RETRY send (to rank 1) is what we observe.
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_s, _salt| Ok(()));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, move |s, salt| {
        // Expect the retry's RequestNewNcclId from the coord.
        let msg = recv_control(s, salt)?;
        match msg {
            ControlMsgWire::RequestNewNcclId => Ok(()),
            other => Err(TensorError::new(&format!(
                "rank 1 expected RequestNewNcclId, got {other:?}"
            ))),
        }
    });
    let r2 = fake_rank(port, 2, world_size as u32, TEST_SALT, |_s, _salt| Ok(()));

    r0.join().unwrap().expect("rank 0 handshake");
    r1.join().unwrap().expect("rank 1 receives RequestNewNcclId");
    r2.join().unwrap().expect("rank 2 handshake");
    coord_handle.join().unwrap().expect("coord drives clean");
}

// Slow-generator case: the seeded rendezvous's `initiated_at` is
// shifted into the past (10s) beyond a 1s timeout, no rank is
// declared dead. The retry fires on timeout alone.
#[test]
fn rendezvous_retry_fires_on_timeout_without_death() {
    let world_size = 3;
    let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
    let dead_for_coord = Arc::clone(&dead_ranks);
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_nccl(world_size)
                .dead_ranks(dead_for_coord)
                .heartbeat_timeout_secs(60)
                .rendezvous_timeout_secs(1)
        },
        move |coord| {
            coord.test_seed_rendezvous_pending(0, vec![0, 1, 2], 10);
            coord.tick()?; // timeout > 1s elapsed → retry
            assert_eq!(
                coord.rendezvous_pending_generator(),
                Some(1),
                "timeout retry must pick the next ascending survivor (rank 0 timed out)"
            );
            assert_eq!(
                coord.rendezvous_tried_generators(),
                vec![0],
                "rank 0 recorded as tried on timeout"
            );
            Ok(())
        },
    );

    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_s, _salt| Ok(()));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, move |s, salt| {
        let msg = recv_control(s, salt)?;
        match msg {
            ControlMsgWire::RequestNewNcclId => Ok(()),
            other => Err(TensorError::new(&format!(
                "rank 1 expected RequestNewNcclId on timeout retry, got {other:?}"
            ))),
        }
    });
    let r2 = fake_rank(port, 2, world_size as u32, TEST_SALT, |_s, _salt| Ok(()));

    r0.join().unwrap().expect("rank 0 handshake");
    r1.join().unwrap().expect("rank 1 receives RequestNewNcclId on timeout");
    r2.join().unwrap().expect("rank 2 handshake");
    coord_handle.join().unwrap().expect("coord drives clean");
}

// Exhaustion: the rendezvous's survivor pool is empty (manufactured
// via the test seam to short-circuit the cohort-filter logic), so
// `check_rendezvous_timeout` falls into the no-candidates branch
// and dispatches `ShutdownWithSave` instead of hanging the cohort
// on a rendezvous that can never complete. All three ranks remain
// alive in TCP for the broadcast to land cleanly.
#[test]
fn rendezvous_exhaustion_dispatches_shutdown_with_save() {
    let world_size = 3;

    let dir = std::env::temp_dir().join(format!(
        "flodl_rdv_exhaust_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let stem = dir.join("ckpt").to_string_lossy().into_owned();

    let (port, coord_handle) = spawn_coord(
        world_size,
        move || {
            cfg_sync_nccl_with_dataset(world_size, 12)
                .heartbeat_timeout_secs(60)
                .rendezvous_timeout_secs(1)
                .save_path(stem.clone())
        },
        move |coord| {
            // initiated 10s ago → timed_out=true; empty survivor
            // pool → no next candidate → exhaustion branch fires.
            coord.test_seed_rendezvous_pending(
                0,
                Vec::new(),
                10,
            );
            coord.tick()?;
            assert!(
                coord.rendezvous_pending_generator().is_none(),
                "exhausted pool must clear pending"
            );
            assert!(
                coord.shutdown_with_save_dispatched(),
                "exhausted pool must dispatch ShutdownWithSave"
            );
            Ok(())
        },
    );

    // Fake ranks must drain the ShutdownWithSave broadcast — otherwise
    // the coord's send into a closed socket trips a broken-pipe error.
    fn drain_shutdown(s: &mut TcpStream, salt: &SessionSalt) -> Result<()> {
        s.set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| TensorError::new(&format!("timeout: {e}")))?;
        match recv_control(s, salt)? {
            ControlMsgWire::ShutdownWithSave { .. } => Ok(()),
            other => Err(TensorError::new(&format!(
                "expected ShutdownWithSave, got {other:?}"
            ))),
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, drain_shutdown);
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, drain_shutdown);
    let r2 = fake_rank(port, 2, world_size as u32, TEST_SALT, drain_shutdown);
    r0.join().unwrap().expect("rank 0 receives ShutdownWithSave");
    r1.join().unwrap().expect("rank 1 receives ShutdownWithSave");
    r2.join().unwrap().expect("rank 2 receives ShutdownWithSave");
    coord_handle.join().unwrap().expect("coord drives clean");

    std::fs::remove_dir_all(&dir).ok();
}

/// Regression for the epoch-transition stall: after aggregating
/// epoch N (non-progressive Sync), the coord must dispatch epoch
/// N+1; once N+1 == num_epochs it must broadcast `Shutdown`.
/// Without this, workers idle in `wait_for_epoch_plan` after the
/// final `EpochAggregated` and the launcher hangs.
#[test]
fn epoch_transition_dispatches_next_then_shutdowns_at_horizon() {
    let world_size = 2;
    let num_epochs = 2;
    let (port, coord_handle) = spawn_coord(
        world_size,
        move || cfg_sync_cpu(world_size)
            .total_samples(8)
            .batch_size(4)
            .num_epochs(num_epochs),
        move |coord| {
            coord.dispatch_epoch(0)?;
            let start = Instant::now();
            // Drive ticks until the coord observes both ranks have
            // closed (tick returns false). The fix in
            // `drain_metrics_and_aggregate` broadcasts `Shutdown`
            // after the final epoch aggregates; readers see EOF
            // when ranks exit, `is_finished()` flips, alive=false.
            loop {
                if start.elapsed() > Duration::from_secs(10) {
                    return Err(TensorError::new(
                        "coord did not drain within 10s",
                    ));
                }
                if !coord.tick()? {
                    break;
                }
                thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(
                coord.last_aggregated_epoch(),
                Some(num_epochs - 1),
                "both epochs must have aggregated",
            );
            Ok(())
        },
    );

    fn rank_body(
        rank: u64,
        num_epochs: usize,
    ) -> impl Fn(&mut TcpStream, &SessionSalt) -> Result<()> {
        move |s, salt| {
            let mut completed = 0usize;
            let mut saw_shutdown = false;
            while !saw_shutdown {
                let msg = recv_control(s, salt)?;
                match msg {
                    ControlMsgWire::StartEpoch(plan) => {
                        send_metrics(s, salt, MetricsMsgWire {
                            rank,
                            epoch: plan.epoch,
                            avg_loss: 0.5,
                            batches_processed: 2,
                            epoch_ms: 50.0,
                            samples_processed: 4,
                            share_complete_ms: 0.0,
                            compute_only_ms: 50.0,
                            data_starve_ms: 0.0,
                            scalars: std::collections::HashMap::new(),
                        })?;
                        completed += 1;
                    }
                    ControlMsgWire::Shutdown
                    | ControlMsgWire::ShutdownWithSave { .. } => {
                        saw_shutdown = true;
                    }
                    // SetEpochCallbackRole / EpochAggregated / any
                    // unrelated control frames are observed but
                    // don't drive state in this regression test.
                    _ => {}
                }
            }
            if completed != num_epochs {
                return Err(TensorError::new(&format!(
                    "rank {rank}: received Shutdown after {completed} epochs \
                     (expected {num_epochs})",
                )));
            }
            Ok(())
        }
    }
    let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT,
        rank_body(0, num_epochs));
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT,
        rank_body(1, num_epochs));
    r0.join().unwrap().expect("rank 0 completed all epochs");
    r1.join().unwrap().expect("rank 1 completed all epochs");
    coord_handle.join().unwrap().expect("coord finishes cleanly");
}
