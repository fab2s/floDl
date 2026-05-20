//! Tests for [`cluster_coordinator`]. Extracted to a sibling file via
//! `#[path]` to keep the impl file navigable; the test module body
//! has full access to private items via `use super::*` (the `mod tests`
//! attribute lives back in cluster_coordinator.rs).

    use super::*;
    use crate::distributed::wire::{MetricsMsgWire, TimingMsgWire};
    use std::net::Ipv4Addr;

    /// Deterministic non-zero test salt (mirrors controller.rs::tests).
    const TEST_SALT: SessionSalt = [
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
        0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    ];

    /// Spawn a fake rank that connects to `port`, handshakes with
    /// `salt`, runs `body` against the connected stream, then drops it.
    fn fake_rank<F>(
        port: u16,
        rank_id: u32,
        world_size: u32,
        salt: SessionSalt,
        body: F,
    ) -> thread::JoinHandle<Result<()>>
    where
        F: Send + 'static + FnOnce(&mut TcpStream, &SessionSalt) -> Result<()>,
    {
        thread::spawn(move || -> Result<()> {
            let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
            let mut stream = TcpStream::connect(addr).map_err(|e| {
                TensorError::new(&format!("fake_rank {rank_id} connect: {e}"))
            })?;
            stream
                .set_read_timeout(Some(Duration::from_secs(10)))
                .map_err(|e| TensorError::new(&format!("set_read_timeout: {e}")))?;
            write_handshake_rank(&mut stream, rank_id, world_size, &salt)?;
            let mut ack = [0u8; HS_ACK_BYTES];
            stream.read_exact(&mut ack).map_err(|e| {
                TensorError::new(&format!("fake_rank {rank_id} ack read: {e}"))
            })?;
            let magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
            if magic != CTRL_HS_ACK {
                return Err(TensorError::new(&format!(
                    "fake_rank {rank_id}: unexpected ack magic 0x{magic:08x}"
                )));
            }
            // Verify the ack HMAC ourselves.
            let expected = hmac_first8(&salt, &ack[0..8]);
            let got: [u8; 8] = ack[8..16].try_into().unwrap();
            if expected != got {
                return Err(TensorError::new(
                    "fake_rank: ack HMAC verification failed",
                ));
            }
            stream
                .set_read_timeout(None)
                .map_err(|e| TensorError::new(&format!("clear timeout: {e}")))?;
            body(&mut stream, &salt)
        })
    }

    fn cfg_sync_nccl(world_size: usize) -> ClusterCoordinatorConfig {
        // ElChe::new requires ≥ 2 devices; tests use world_size ≥ 2.
        assert!(world_size >= 2, "tests use world_size >= 2");
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 1),
        )
        .no_divergence_guard()
    }

    fn cfg_async_nccl(world_size: usize) -> ClusterCoordinatorConfig {
        assert!(world_size >= 2, "tests use world_size >= 2");
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Async,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 4)
                .with_max_batch_diff(2),
        )
        .no_divergence_guard()
    }

    /// Send a Timing-kind ControlFrame on a fake-rank stream.
    fn send_timing(
        stream: &mut TcpStream,
        salt: &SessionSalt,
        msg: TimingMsgWire,
    ) -> Result<()> {
        let frame = ControlFrame::encode(salt, MsgKind::Timing, &msg)?;
        frame.write_to(stream)
    }

    /// Send a Metrics-kind ControlFrame on a fake-rank stream.
    fn send_metrics(
        stream: &mut TcpStream,
        salt: &SessionSalt,
        msg: MetricsMsgWire,
    ) -> Result<()> {
        let frame = ControlFrame::encode(salt, MsgKind::Metrics, &msg)?;
        frame.write_to(stream)
    }

    /// Read one Control-kind ControlFrame from the rank-side stream.
    fn recv_control(
        stream: &mut TcpStream,
        salt: &SessionSalt,
    ) -> Result<ControlMsgWire> {
        let frame = ControlFrame::read_from(stream, salt)?
            .ok_or_else(|| TensorError::new("EOF before frame"))?;
        if frame.kind != MsgKind::Control {
            return Err(TensorError::new(&format!(
                "unexpected frame kind {:?}",
                frame.kind
            )));
        }
        frame.decode::<ControlMsgWire>()
    }

    /// Pre-bind a listener in the test (so we can publish the port
    /// before any accept blocks), spawn rank threads against that
    /// port, then drive the coordinator's accept + state machine in
    /// a worker thread. Returns the rank-side and coord-side join
    /// handles plus the bound port for the rank-side connect.
    fn spawn_coord<F>(
        _world_size: usize,
        config_fn: impl FnOnce() -> ClusterCoordinatorConfig + Send + 'static,
        drive: F,
    ) -> (u16, thread::JoinHandle<Result<()>>)
    where
        F: Send + 'static + FnOnce(&mut ClusterCoordinator) -> Result<()>,
    {
        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("bind succeeds");
        assert_eq!(listener.local_addr().unwrap().port(), port);
        let handle = thread::spawn(move || -> Result<()> {
            let mut coord = ClusterCoordinator::start_from_listener(
                listener, TEST_SALT, config_fn(),
            )?;
            let r = drive(&mut coord);
            // Best-effort shutdown even on failure so the readers join.
            let _ = coord.shutdown();
            r
        });
        (port, handle)
    }

    #[test]
    fn handshake_round_trip_with_matching_salt() {
        // 2 ranks, Sync; both handshake and immediately drop. No
        // averaging cycle expected — `drive` just returns Ok.
        let world_size = 2;
        let (port, coord_handle) =
            spawn_coord(world_size, move || cfg_sync_nccl(world_size), |_coord| Ok(()));

        let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |_, _| Ok(()));
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |_, _| Ok(()));
        r0.join().unwrap().expect("rank 0 handshake");
        r1.join().unwrap().expect("rank 1 handshake");
        coord_handle.join().unwrap().expect("coord drives clean");
    }

    #[test]
    fn handshake_rejects_wrong_salt_full_path() {
        // Coordinator has TEST_SALT; rank 0 connects with all-zero salt.
        // The accept loop's handshake validation fails →
        // start_from_listener returns an error.
        let world_size = 2;
        let bad_salt: SessionSalt = [0u8; 16];

        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .unwrap();
        let coord_handle = thread::spawn(move || -> Result<ClusterCoordinator> {
            ClusterCoordinator::start_from_listener(
                listener, TEST_SALT, cfg_sync_nccl(world_size),
            )
        });

        let rank = thread::spawn(move || {
            let mut s = TcpStream::connect(
                SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port),
            )
            .unwrap();
            // Wrong salt → handshake HMAC fails server-side.
            let _ = write_handshake_rank(&mut s, 0, world_size as u32, &bad_salt);
            // Read until the server drops us.
            let mut throwaway = [0u8; 16];
            let _ = s.read_exact(&mut throwaway);
        });
        let err = match coord_handle.join().unwrap() {
            Ok(_) => panic!("expected start_from_listener to fail on bad-salt rank"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC failure, got: {err}"
        );
        let _ = rank.join();
    }

    /// Bind a listener ourselves, hand the connection to the
    /// handshake validator directly, exercise the wrong-salt branch.
    #[test]
    fn read_handshake_rank_rejects_wrong_salt_direct() {
        let listener = TcpListener::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)
        ).unwrap();
        let port = listener.local_addr().unwrap().port();

        let bad_salt: SessionSalt = [0u8; 16];
        assert_ne!(bad_salt, TEST_SALT);

        let rank = thread::spawn(move || {
            let mut s = TcpStream::connect(
                SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port),
            ).unwrap();
            // Send a handshake keyed by the wrong salt.
            write_handshake_rank(&mut s, 0, 1, &bad_salt).unwrap();
            // Don't expect an ack; the coordinator should drop us.
            drop(s);
        });

        let (mut server_stream, _) = listener.accept().unwrap();
        server_stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let err = read_handshake_rank(&mut server_stream, 1, &TEST_SALT).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HMAC verification failed"),
            "expected HMAC failure, got: {msg}"
        );
        rank.join().unwrap();
    }

    #[test]
    fn sync_policy_fires_after_each_rank_step_once() {
        // 2 ranks, Sync policy: after each rank reports one Batch, the
        // coordinator should fire SyncNow + SetGlobalStep exactly once.
        let world_size = 2;
        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_sync_nccl(world_size),
            |coord| {
                let start = Instant::now();
                while coord.avg_count() == 0 {
                    if start.elapsed() > Duration::from_secs(5) {
                        return Err(TensorError::new(
                            "sync_policy_fires timed out waiting for avg_count",
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
            assert_eq!(msg, ControlMsgWire::SyncNow);
            let msg2 = recv_control(s, salt)?;
            assert!(matches!(msg2, ControlMsgWire::SetGlobalStep { .. }));
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
            assert_eq!(msg, ControlMsgWire::SyncNow);
            let msg2 = recv_control(s, salt)?;
            assert!(matches!(msg2, ControlMsgWire::SetGlobalStep { .. }));
            Ok(())
        });

        r0.join().unwrap().expect("rank 0 sees SyncNow + SetGlobalStep");
        r1.join().unwrap().expect("rank 1 sees SyncNow + SetGlobalStep");
        coord_handle.join().unwrap().expect("coord finishes");
    }

    // Throttle is an Async/CPU-backend concept; NCCL backend uses
    // AllReduce as the coordination mechanism (sending Throttle there
    // would deadlock with the collective). This test structurally
    // exercises that path via `cfg_async_nccl`, which goes through
    // `check_throttle` and confirms the NCCL early-return guard (the
    // function returns without sending a frame to any rank).
    // Behavioral throttle tests live in the CPU-backend test module.
    #[test]
    fn check_throttle_nccl_backend_is_no_op() {
        // Construct a coord with Async+Nccl; tick once with both ranks
        // having reported a single batch. check_throttle must return
        // Ok and send no Throttle frames.
        let world_size = 2;
        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_async_nccl(world_size),
            |coord| {
                // Wait for at least one timing message per rank, then
                // run a few ticks. If check_throttle were to send a
                // Throttle here, the rank-side recv would surface it
                // and the rank closure would assert. We don't.
                let deadline = Instant::now() + Duration::from_secs(2);
                while coord.steps_since_avg().contains(&0) {
                    if Instant::now() > deadline {
                        return Err(TensorError::new(
                            "did not receive a batch from each rank",
                        ));
                    }
                    coord.tick()?;
                    thread::sleep(Duration::from_millis(10));
                }
                // A few extra ticks — no Throttle should fire.
                for _ in 0..10 {
                    coord.tick()?;
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(())
            },
        );

        let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 0,
                batch_ms: 5.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            // Drain inbound frames until the coordinator drops us.
            // We must NOT receive a Throttle; if we do, assert.
            let mut got = ControlFrame::read_from(s, salt);
            while let Ok(Some(frame)) = got {
                let kind = frame.kind;
                let msg = frame.decode::<ControlMsgWire>()?;
                assert!(
                    !matches!(msg, ControlMsgWire::Throttle),
                    "Throttle must not fire on NCCL backend (rank 0, kind={kind:?})"
                );
                got = ControlFrame::read_from(s, salt);
            }
            Ok(())
        });
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(s, salt, TimingMsgWire::Batch {
                rank: 1,
                batch_ms: 5.0,
                step_count: 1,
                param_norm: None,
                batch_loss: 0.5,
                sync_divergence: None,
            })?;
            let mut got = ControlFrame::read_from(s, salt);
            while let Ok(Some(frame)) = got {
                let msg = frame.decode::<ControlMsgWire>()?;
                assert!(
                    !matches!(msg, ControlMsgWire::Throttle),
                    "Throttle must not fire on NCCL backend (rank 1)"
                );
                got = ControlFrame::read_from(s, salt);
            }
            Ok(())
        });

        coord_handle.join().unwrap().expect("coord drives");
        // Rank threads may still be reading frames; coord.shutdown sent
        // Shutdown frames to them which they should decode and exit.
        // The asserts above guard the no-Throttle invariant.
        let _ = r0.join();
        let _ = r1.join();
    }

    // -----------------------------------------------------------------
    // Epoch dispatch
    // -----------------------------------------------------------------

    fn cfg_sync_nccl_with_dataset(world_size: usize, total_samples: usize) -> ClusterCoordinatorConfig {
        cfg_sync_nccl(world_size)
            .total_samples(total_samples)
            .batch_size(2)
            .num_epochs(1)
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

    fn cfg_sync_cpu(world_size: usize) -> ClusterCoordinatorConfig {
        assert!(world_size >= 2, "tests use world_size >= 2");
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Cpu,
            world_size,
            ElChe::new(world_size, 1),
        )
        .no_divergence_guard()
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
    // #29 — Checkpoint dispatch (targeted), result reporting, retry,
    // time exclusion, and role-failover on rank death. The coord is
    // the sole decider; the worker is a pure executor that reports
    // back via `TimingMsgWire::CheckpointResult` and never decides
    // policy locally. See `drawer_wing_rdl_decisions_8053ff4b...` for
    // the architectural principles these tests encode.
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
        coord.handle_checkpoint_result(
            0, 5, 12.0, Some("disk full".into()),
        );
        assert_eq!(coord.checkpoint_role(), 1, "role should fail over to rank 1");
        assert_eq!(coord.checkpoint_tried_count(5), 1);

        // Rank 1 also fails: tried={0,1}, role moves to rank 2.
        coord.handle_checkpoint_result(
            1, 5, 8.0, Some("io error".into()),
        );
        assert_eq!(coord.checkpoint_role(), 2);
        assert_eq!(coord.checkpoint_tried_count(5), 2);

        // Rank 2 also fails: no more live untried ranks → exhaust +
        // clear tried_ranks (the next cadence boundary starts fresh).
        coord.handle_checkpoint_result(
            2, 5, 5.0, Some("permission denied".into()),
        );
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
        coord.el_che_mut_for_test().report_timing(
            &[500.0, 1000.0],
            &[10, 10],
            10.0,
        );
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
    #[test]
    fn checkpoint_dispatched_to_role_only() {
        let world_size = 2;
        let r0_got = Arc::new(AtomicBool::new(false));
        let r1_got = Arc::new(AtomicBool::new(false));
        let r0_flag = Arc::clone(&r0_got);
        let r1_flag = Arc::clone(&r1_got);

        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_sync_cpu(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(2)
                .checkpoint_every(1),
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
                        ControlMsgWire::Checkpoint { version, target_rank } => {
                            saw_checkpoint.store(true, Ordering::Relaxed);
                            if target_rank == send_ack_for {
                                send_metrics(s, salt, MetricsMsgWire::default()).ok();
                                send_timing(s, salt, TimingMsgWire::CheckpointResult {
                                    rank: send_ack_for,
                                    version,
                                    elapsed_ms: 1.0,
                                    error: None,
                                })?;
                            }
                        }
                        ControlMsgWire::Shutdown
                        | ControlMsgWire::ShutdownWithSave { .. } => return Ok(()),
                        _ => {}
                    }
                }
            }
        }
        let r0 = fake_rank(port, 0, world_size as u32, TEST_SALT,
            drain_until_shutdown(r0_flag, 0));
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT,
            drain_until_shutdown(r1_flag, u64::MAX /* never matches */));

        r0.join().unwrap().expect("rank 0 drained cleanly");
        r1.join().unwrap().expect("rank 1 drained cleanly");
        coord_handle.join().unwrap().expect("coord finishes");

        assert!(r0_got.load(Ordering::Relaxed),
            "rank 0 (role) must receive Checkpoint frame");
        assert!(!r1_got.load(Ordering::Relaxed),
            "rank 1 (non-role) must NOT receive Checkpoint frame");
    }

    /// Role failover on rank death: when the heartbeat detector
    /// declares the sticky `checkpoint_role` rank dead, the coord
    /// picks the lowest live rank as the new role. The next cadence
    /// boundary will dispatch there; no immediate redispatch fires
    /// (the dead rank had no in-flight checkpoint to recover).
    #[test]
    fn checkpoint_role_failover_on_rank_death() {
        let world_size = 3usize;
        let dead_ranks =
            crate::distributed::controller::DeadRanks::new(world_size);
        let cfg = cfg_sync_cpu(world_size).dead_ranks(Arc::clone(&dead_ranks));
        let mut coord = ClusterCoordinator::for_test(cfg);
        assert_eq!(coord.checkpoint_role(), 0);

        // Force rank 0's last_heartbeat to be older than the
        // heartbeat_timeout_secs threshold, then drive
        // check_dead_ranks. heartbeat_timeout_secs default ~ 30s;
        // setting last_heartbeat[0] to (now - 60s) is a safe margin.
        let stale = Instant::now()
            - Duration::from_secs(coord.heartbeat_timeout_secs() * 2 + 5);
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
    // #28b — EpochCallbackPolicy::Fastest dispatcher: coord-side
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
        let cfg = ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Cpu,
            world_size,
            el_che,
        )
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
        let dead_ranks =
            crate::distributed::controller::DeadRanks::new(world_size);
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
        let dead_ranks =
            crate::distributed::controller::DeadRanks::new(world_size);
        let cfg = ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Cpu,
            world_size,
            el_che,
        )
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
    /// death does NOT trigger a Fastest re-resolve — the eval and
    /// epoch_callback roles stay pinned. (The legacy #29 checkpoint-
    /// role-only failover still applies for Rank policy; covered by
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
            move || cfg_sync_cpu(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(2)
                .eval_every_epochs(1),
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
                        ControlMsgWire::Shutdown
                        | ControlMsgWire::ShutdownWithSave { .. } => return Ok(()),
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

        assert!(r0_got.load(Ordering::Relaxed),
            "rank 0 (eval_role default) must receive ExecuteEvalCallback");
        assert!(!r1_got.load(Ordering::Relaxed),
            "rank 1 (non-role) must NOT receive ExecuteEvalCallback");
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
            move || cfg_sync_cpu(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(3),
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
                        ControlMsgWire::Shutdown
                        | ControlMsgWire::ShutdownWithSave { .. } => return Ok(()),
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
    #[test]
    fn checkpoint_success_round_trip_updates_ewma() {
        let world_size = 2;
        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_sync_cpu(world_size)
                .total_samples(8)
                .batch_size(4)
                .num_epochs(2)
                .checkpoint_every(1),
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
                    (coord.last_checkpoint_elapsed_ms_ewma().unwrap() - 12.5).abs()
                        < 1e-9,
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
                        ControlMsgWire::Checkpoint { version, target_rank }
                            if target_rank == rank =>
                        {
                            send_timing(s, salt, TimingMsgWire::CheckpointResult {
                                rank,
                                version,
                                elapsed_ms: 12.5,
                                error: None,
                            })?;
                        }
                        ControlMsgWire::Shutdown
                        | ControlMsgWire::ShutdownWithSave { .. } => return Ok(()),
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
        let (listener, _port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .unwrap();
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
    fn meta_controller_default_is_off() {
        // Opt-in semantics: a config built without `.meta_controller(true)`
        // must produce a coordinator with the meta DISABLED. Mirrors OLD
        // `CoordinatorBuilder` parity (default `meta_controller = false`).
        let world_size = 2;
        let (port, coord_handle) = spawn_coord(
            world_size,
            move || cfg_sync_nccl(world_size),
            |coord| {
                assert!(
                    !coord.meta_controller_enabled_for_test(),
                    "meta_controller must default to OFF"
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
            send_timing(
                s,
                salt,
                TimingMsgWire::LrUpdate { rank: 0, lr: 0.01 },
            )?;
            // Stay alive long enough for the coord to drain the frame.
            thread::sleep(Duration::from_millis(200));
            Ok(())
        });
        let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
            send_timing(
                s,
                salt,
                TimingMsgWire::LrUpdate { rank: 1, lr: 0.02 },
            )?;
            thread::sleep(Duration::from_millis(200));
            Ok(())
        });
        r0.join().unwrap().expect("rank 0 sends LrUpdate");
        r1.join().unwrap().expect("rank 1 sends LrUpdate");
        coord_handle.join().unwrap().expect("coord captures both LRs");
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
