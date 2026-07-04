    use super::*;
    use crate::distributed::cluster_coordinator::{
        ClusterCoordinator, ClusterCoordinatorConfig,
    };
    use crate::distributed::ddp::ElChe;
    use crate::distributed::ddp_run::{ApplyPolicy, AverageBackend};
    use crate::distributed::wire::SESSION_SALT_BYTES;
    use std::net::Ipv4Addr;
    use std::time::Instant;

    /// Deterministic non-zero test salt — same value as the
    /// cluster_coordinator / controller test salts so cross-module
    /// integration tests can chain freely.
    const TEST_SALT: SessionSalt = [0x42u8; SESSION_SALT_BYTES];

    fn coord_config_sync_nccl(world_size: usize) -> ClusterCoordinatorConfig {
        ClusterCoordinatorConfig::new(
            ApplyPolicy::Sync,
            AverageBackend::Nccl,
            world_size,
            ElChe::new(world_size, 1),
        )
        .no_divergence_guard()
    }

    /// Spawn a ClusterCoordinator that drives `drive` to completion,
    /// then shuts down. Returns the bound port + join handle.
    fn spawn_coord<D>(
        world_size: usize,
        drive: D,
    ) -> (u16, thread::JoinHandle<Result<()>>)
    where
        D: Send + 'static + FnOnce(&mut ClusterCoordinator) -> Result<()>,
    {
        let (listener, port) = ClusterCoordinator::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("bind succeeds");
        let h = thread::spawn(move || -> Result<()> {
            let mut coord = ClusterCoordinator::start_from_listener(
                listener,
                TEST_SALT,
                coord_config_sync_nccl(world_size),
            )?;
            let r = drive(&mut coord);
            let _ = coord.shutdown();
            r
        });
        (port, h)
    }

    /// Smoke test: a ClusterWorker can hold a TcpStream open against a
    /// real ClusterCoordinator after handshake, even with no inner
    /// GpuWorker constructed yet. Exercises just the handshake
    /// bytes (matching salt path, ack HMAC verification).
    #[test]
    fn handshake_with_real_coordinator() {
        let world_size = 1;
        // ClusterCoordinator demands world_size >= 2 (ElChe), so we
        // use 2 here and have a dummy second rank just complete its
        // handshake then drop.
        let world_size = world_size.max(2);
        let (port, coord_handle) = spawn_coord(world_size, |coord| {
            // Drive one tick to confirm the coord registered both
            // ranks before they drop.
            // The accept loop in start_from_listener already validated
            // both handshakes; coord.tick() just returns Ok.
            let _ = coord.tick();
            Ok(())
        });
        // Workers handshake with their host control relay (which terminates
        // the handshake and forwards to the coord). `_crelay_rx` drops at
        // scope end, shutting the relay down.
        let coord_real_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let (addr, _crelay_rx) =
            spawn_relay(ChannelKind::Control, coord_real_addr, world_size, TEST_SALT);

        // Direct-handshake closures (no inner GpuWorker required —
        // we exercise only the handshake bytes + ack here).
        fn raw_rank_handshake(addr: SocketAddr, rank: u32, ws: u32) {
            let mut stream =
                TcpStream::connect_timeout(&addr, Duration::from_secs(5)).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            write_handshake_rank(&mut stream, rank, ws, &TEST_SALT).unwrap();
            read_handshake_ack(&mut stream, &TEST_SALT).unwrap();
            // Hold the stream open briefly so the coord can register
            // before we drop.
            thread::sleep(Duration::from_millis(50));
        }
        let r0 = thread::spawn(move || raw_rank_handshake(addr, 0, world_size as u32));
        let r1 = thread::spawn(move || raw_rank_handshake(addr, 1, world_size as u32));
        r0.join().unwrap();
        r1.join().unwrap();
        coord_handle.join().unwrap().expect("coord drives clean");
    }

    /// Salt mismatch on the worker side surfaces loudly at handshake.
    /// Under the relay transport the worker handshakes with its host
    /// control relay, so the relay (which terminates the handshake with
    /// the correct salt) rejects the bad-salt worker during its accept
    /// phase — before it ever connects upstream to the coord.
    #[test]
    fn handshake_rejects_wrong_salt_on_worker_side() {
        let world_size = 2;
        let bad_salt: SessionSalt = [0u8; SESSION_SALT_BYTES];

        // Control relay with the correct salt. The upstream coord address
        // is never reached: the relay errors in its accept phase on the
        // bad-salt handshake, before the upstream-connect step.
        let dummy_upstream = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 1);
        let (relay_addr, relay_rx) =
            spawn_relay(ChannelKind::Control, dummy_upstream, world_size, TEST_SALT);

        let rank = thread::spawn(move || {
            let mut s = TcpStream::connect_timeout(&relay_addr, Duration::from_secs(5)).unwrap();
            let _ = write_handshake_rank(&mut s, 0, world_size as u32, &bad_salt);
            let _ = read_handshake_ack(&mut s, &bad_salt);
        });
        let err = match relay_rx.recv().unwrap() {
            Ok(_) => panic!("expected relay to reject wrong-salt handshake"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("HMAC verification failed"),
            "expected HMAC failure, got: {err}"
        );
        let _ = rank.join();
    }

    /// End-to-end Sync+Nccl smoke test. Requires CUDA + NCCL; runs
    /// only under `fdl cuda-test-nccl`. Validates the full
    /// connect → handshake → wait_for_epoch_plan → train_step → SyncNow
    /// → SyncAck → Shutdown round-trip with two real ranks doing real
    /// NCCL AllReduce(Avg) on their parameters.
    ///
    /// Acceptance: after a few averaging cycles, both ranks' parameter
    /// tensors are bit-identical (NCCL AllReduce-Avg makes them so).
    ///
    /// Marked `#[ignore]` so the CPU test suite skips it; lift the
    /// `ignore` (or run via `fdl cuda-test-nccl`) on the Pascal rig.
    #[test]
    #[ignore = "requires CUDA + NCCL — run via fdl cuda-test-nccl"]
    fn end_to_end_sync_nccl_smoke() {
        // The full body is left for the next slice's bring-up on the
        // Pascal rig. Once the rig is online we'll:
        //  1. Build a 2-rank NCCL communicator via NcclComms + split().
        //  2. Spawn ClusterCoordinator on controller_port + 3 with
        //     coord_config_sync_nccl(2).
        //  3. For each rank: in a thread, construct a tiny model
        //     (Linear with a few params), a small in-memory dataset,
        //     SGD optimizer, ClusterWorker::connect_and_build, then
        //     ClusterWorker::run_until_shutdown(train_fn).
        //  4. After workers exit, assert the two ranks' final
        //     parameters are bit-identical (collected via the
        //     final_param channel — this test can attach a
        //     non-discard final bridge for validation, even though
        //     the production bridge currently discards).
        //
        // For now the test is structural — when CUDA tests run it
        // simply asserts that the module compiles and links
        // cleanly. Body lands in the Pascal-rig follow-up.
    }

    /// Pascal-rig end-to-end test for the elastic-membership-aware
    /// `run_cluster_rank_sync_nccl_via_coord` path: spawn 3 ranks on
    /// 3 GPUs, kill rank 2's heartbeat thread mid-training, verify
    /// that ranks 0 and 1:
    ///
    /// 1. See `DeclareDead { rank: 2 }` on their inbound bridge → the
    ///    NCCL watchdog aborts the in-flight collective.
    /// 2. Receive a fresh `NewNcclSession` from the coord (one of the
    ///    survivors generated the UID).
    /// 3. Rebuild their NCCL comm with `world_size = 2` and re-issue
    ///    the failed AllReduce on the survivor cohort — `sync_now_nccl`
    ///    returns success on the retry.
    /// 4. Absorb rank 2's un-processed partition via `ExtendPartition`
    ///    and complete the epoch's intended sample count.
    /// 5. Sync_round counter reaches the expected post-recovery value.
    ///
    /// And a separate path: configure `max_failure = Absolute(2)`,
    /// kill ranks 1 and 2, verify that rank 0 receives
    /// `ShutdownWithSave { reason: MaxFailureExceeded }`, writes a
    /// bundle (model + optimizer + meta) to the configured save_path,
    /// and exits cleanly.
    ///
    /// Marked `#[ignore]` — requires 2+ visible GPUs + libnccl. Run
    /// via `fdl @cluster-test cuda-test-nccl` (env overlay defines
    /// the cluster topology) or with N visible GPUs locally
    /// (autodetect).
    ///
    /// Smoke test: happy path only. Confirms the via-coord wiring runs
    /// without crashing on real NCCL. Rank-death / max_failure
    /// validation lands as separate `#[ignore]` tests once this
    /// happy-path baseline is green on the rig.
    #[test]
    #[ignore = "requires CUDA + NCCL + 2+ GPUs — run via fdl @cluster-test cuda-test-nccl"]
    fn end_to_end_sync_nccl_via_coord_smoke() {
        use crate::distributed::testing::discover_test_cluster;
        use crate::distributed::nccl::NcclComms;

        // 1. Discover cluster topology. fdl-cli injects the rig topology
        //    via FLODL_TESTING_CLUSTER_JSON when `fdl @cluster-test`
        //    activates the overlay; locally we fall back to autodetect.
        let cluster = match discover_test_cluster() {
            Some(c) => c,
            None => {
                eprintln!(
                    "end_to_end_sync_nccl_via_coord_smoke: no cluster topology \
                     available (set FLODL_TESTING_CLUSTER_JSON via \
                     `fdl @cluster-test` or run on a CUDA host)"
                );
                return;
            }
        };
        let total_ranks: usize = cluster.workers.iter().map(|h| h.ranks.len()).sum();
        if total_ranks < 2 {
            eprintln!(
                "end_to_end_sync_nccl_via_coord_smoke: NCCL needs 2+ ranks \
                 (have {total_ranks}); skipping"
            );
            return;
        }

        // 2. Build a shared DeadRanks ledger + spawn the coord listener
        //    on a kernel-assigned port (test convention: ignore the
        //    cluster's controller_port = 0 sentinel and bind fresh).
        let world_size = total_ranks;
        let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
        let (coord_listener, coord_port) = CCoord::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("coord bind succeeds");
        // Workers reach the coord through their host control relay. The
        // relay handle sits buffered in `_ctrl_relay_rx` and drops at end
        // of scope (shutting the relay down). The coord listens on
        // `coord_port`; the relay forwards to it.
        let coord_real_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), coord_port);
        let (coord_addr, _ctrl_relay_rx) =
            spawn_relay(ChannelKind::Control, coord_real_addr, world_size, TEST_SALT);
        let dead_for_coord = Arc::clone(&dead_ranks);
        let total_samples = 16usize;
        let batch_size = 4usize;
        let config_for_coord = move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Nccl,
                world_size,
                crate::distributed::ddp::ElChe::new(world_size, 1),
            )
            .no_divergence_guard()
            .dead_ranks(dead_for_coord)
            .total_samples(total_samples)
            .batch_size(batch_size)
            .num_epochs(1)
        };
        let coord_thread = thread::spawn(move || -> Result<CCoord> {
            CCoord::start_from_listener(
                coord_listener,
                [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
                config_for_coord(),
            )
        });

        // 3. Build NCCL comms via NcclComms::new + split() — single-
        //    process multi-thread pattern. Each thread will own one
        //    NcclRankComm and run as a rank.
        let devices: Vec<Device> = (0..world_size as u8)
            .map(Device::CUDA)
            .collect();
        let group = NcclComms::new(&devices).expect("NcclComms::new succeeds");
        let rank_comms = group.split().expect("split succeeds");

        // 4. Capture initial params on CPU once so all ranks start
        //    aligned. Each worker thread re-creates its model on its
        //    device + overrides with these initial values.
        let ref_model = Linear::on_device(4, 2, Device::CPU).unwrap();
        let initial_params: Vec<Tensor> = ref_model
            .parameters()
            .iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = ref_model
            .buffers()
            .iter()
            .map(|b| b.get())
            .collect();
        drop(ref_model);

        // 5. Spawn one worker thread per rank. Each owns its
        //    NcclRankComm + connects to the coord via ClusterWorker
        //    + runs to shutdown.
        let salt = [0u8; crate::distributed::wire::SESSION_SALT_BYTES];
        let mut worker_handles: Vec<thread::JoinHandle<Result<()>>> = Vec::new();
        for (rank_id, comm) in rank_comms.into_iter().enumerate() {
            let initial_params = initial_params.clone();
            let initial_buffers = initial_buffers.clone();
            let device = Device::CUDA(rank_id as u8);
            worker_handles.push(thread::spawn(move || -> Result<()> {
                let config = WorkerConfig {
                    rank: rank_id,
                    world_size,
                    device,
                    initial_params,
                    initial_buffers,
                    total_samples,
                    batch_size,
                    seed: 42,
                    max_grad_norm: None,
                    easgd_alpha: None,
                    gamma: 1.0,
                    timeline: None,
                    policy: ApplyPolicy::Sync,
                    save_path: None,
                    coord_liveness_timeout_secs:
                        crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
                };
                let dataset: Arc<dyn crate::data::BatchDataSet> =
                    Arc::new(TestDataset { n: total_samples });
                let worker = ClusterWorker::connect_and_build(
                    coord_addr,
                    None, // no CPU data channel; NCCL handles its own
                    rank_id as u32,
                    salt,
                    config,
                    move |d| Linear::on_device(4, 2, d),
                    |params| crate::nn::SGD::new(params, 0.01, 0.0),
                    dataset,
                    Some(comm),
                    None,
                    None,
                    None, // no eval_fn
                    None, // no eval_dataset
                    None, // no outer optimizer
                )?;
                worker.run_until_shutdown(mse_train).map(|_| ())
            }));
        }

        // 6. Coord thread unblocks after every worker handshakes.
        let mut coord = coord_thread
            .join()
            .expect("coord thread join")
            .expect("start_from_listener succeeds");

        coord.dispatch_epoch(0).expect("dispatch_epoch(0) succeeds");
        let start = Instant::now();
        while coord.avg_count() == 0 {
            if start.elapsed() > Duration::from_secs(30) {
                panic!(
                    "end_to_end_sync_nccl_via_coord_smoke: avg_count never \
                     advanced (no NCCL AllReduce observed within 30s)"
                );
            }
            coord.tick().expect("tick");
            thread::sleep(Duration::from_millis(20));
        }
        assert!(coord.avg_count() >= 1, "at least one NCCL averaging cycle");

        coord.shutdown_workers().expect("shutdown_workers");
        coord.shutdown().expect("coord shutdown");
        for h in worker_handles {
            h.join().expect("worker thread join").expect("worker exits clean");
        }
    }

    /// End-to-end Cadence+Nccl via_coord smoke test — heterogeneous
    /// Local-SGD with ElChe-driven cadence and a real
    /// [`ClusterCoordinator`] driving the guard pipeline.
    ///
    /// Cadence and Sync share the worker-side code path under
    /// via_coord: the coord owns ElChe + ConvergenceGuard and broadcasts
    /// `SyncNow` at K-batch boundaries (see
    /// [`ClusterCoordinator::should_average`] +
    /// [`ClusterCoordinator::trigger_averaging`]). This test confirms
    /// that the routing flip in
    /// [`DdpHandle::run_cluster_rank_cadence_nccl_via_coord`] connects
    /// up correctly end-to-end: coord with `ApplyPolicy::Cadence`,
    /// workers with `WorkerConfig.policy = Cadence`, multiple AllReduce
    /// cycles complete cleanly, all ranks exit on `Shutdown`.
    ///
    /// Acceptance: at least two `coord.avg_count()` cycles fire
    /// (multi-cycle Cadence proven). Final params converge to
    /// bit-identical across ranks via NCCL AllReduce-Avg invariant.
    ///
    /// Marked `#[ignore]` — requires CUDA + NCCL + 2+ GPUs. Run via
    /// `fdl @cluster-test cuda-test-nccl` on the Pascal rig.
    ///
    /// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterCoordinator::should_average`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator
    /// [`ClusterCoordinator::trigger_averaging`]:
    ///     crate::distributed::cluster_coordinator::ClusterCoordinator::trigger_averaging
    #[test]
    #[ignore = "requires CUDA + NCCL + 2+ GPUs — run via fdl @cluster-test cuda-test-nccl"]
    fn end_to_end_cadence_nccl_via_coord_smoke() {
        use crate::distributed::testing::discover_test_cluster;
        use crate::distributed::nccl::NcclComms;

        let cluster = match discover_test_cluster() {
            Some(c) => c,
            None => {
                eprintln!(
                    "end_to_end_cadence_nccl_via_coord_smoke: no cluster topology \
                     available (set FLODL_TESTING_CLUSTER_JSON via \
                     `fdl @cluster-test` or run on a CUDA host)"
                );
                return;
            }
        };
        let total_ranks: usize = cluster.workers.iter().map(|h| h.ranks.len()).sum();
        if total_ranks < 2 {
            eprintln!(
                "end_to_end_cadence_nccl_via_coord_smoke: NCCL needs 2+ ranks \
                 (have {total_ranks}); skipping"
            );
            return;
        }

        let world_size = total_ranks;
        let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
        let (coord_listener, coord_port) = CCoord::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("coord bind succeeds");
        // Workers reach the coord through their host control relay. The
        // relay handle sits buffered in `_ctrl_relay_rx` and drops at end
        // of scope (shutting the relay down). The coord listens on
        // `coord_port`; the relay forwards to it.
        let coord_real_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), coord_port);
        let (coord_addr, _ctrl_relay_rx) =
            spawn_relay(ChannelKind::Control, coord_real_addr, world_size, TEST_SALT);
        let dead_for_coord = Arc::clone(&dead_ranks);

        // Anchor=2 batches per rank between syncs (uncalibrated ElChe;
        // first cycle reports timing, subsequent cycles may rebalance).
        // 16 samples / batch=4 / partition split = 2 batches per rank for
        // ws=2 with equal partition → exactly one sync per epoch per
        // rank's K; running 4 epochs guarantees multiple cycles.
        let total_samples = 32usize;
        let batch_size = 4usize;
        let elche_anchor = 2usize;
        let num_epochs = 4usize;
        let config_for_coord = move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Cadence,
                AverageBackend::Nccl,
                world_size,
                crate::distributed::ddp::ElChe::new(world_size, elche_anchor),
            )
            .no_divergence_guard()
            .dead_ranks(dead_for_coord)
            .total_samples(total_samples)
            .batch_size(batch_size)
            .num_epochs(num_epochs)
        };
        let coord_thread = thread::spawn(move || -> Result<CCoord> {
            CCoord::start_from_listener(
                coord_listener,
                [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
                config_for_coord(),
            )
        });

        let devices: Vec<Device> = (0..world_size as u8)
            .map(Device::CUDA)
            .collect();
        let group = NcclComms::new(&devices).expect("NcclComms::new succeeds");
        let rank_comms = group.split().expect("split succeeds");

        let ref_model = Linear::on_device(4, 2, Device::CPU).unwrap();
        let initial_params: Vec<Tensor> = ref_model
            .parameters()
            .iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = ref_model
            .buffers()
            .iter()
            .map(|b| b.get())
            .collect();
        drop(ref_model);

        let salt = [0u8; crate::distributed::wire::SESSION_SALT_BYTES];
        let mut worker_handles: Vec<thread::JoinHandle<Result<()>>> = Vec::new();
        for (rank_id, comm) in rank_comms.into_iter().enumerate() {
            let initial_params = initial_params.clone();
            let initial_buffers = initial_buffers.clone();
            let device = Device::CUDA(rank_id as u8);
            worker_handles.push(thread::spawn(move || -> Result<()> {
                let config = WorkerConfig {
                    rank: rank_id,
                    world_size,
                    device,
                    initial_params,
                    initial_buffers,
                    total_samples,
                    batch_size,
                    seed: 42,
                    max_grad_norm: None,
                    easgd_alpha: None,
                    gamma: 1.0,
                    timeline: None,
                    policy: ApplyPolicy::Cadence,
                    // save_path is None in this smoke — we're testing
                    // the via_coord protocol path, not persistence.
                    // Production callers using auto_with auto-route here
                    // when save_path is set on DdpRunConfig.
                    save_path: None,
                    coord_liveness_timeout_secs:
                        crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
                };
                let dataset: Arc<dyn crate::data::BatchDataSet> =
                    Arc::new(TestDataset { n: total_samples });
                let worker = ClusterWorker::connect_and_build(
                    coord_addr,
                    None,
                    rank_id as u32,
                    salt,
                    config,
                    move |d| Linear::on_device(4, 2, d),
                    |params| crate::nn::SGD::new(params, 0.01, 0.0),
                    dataset,
                    Some(comm),
                    None,
                    None,
                    None, // no eval_fn
                    None, // no eval_dataset
                    None, // no outer optimizer
                )?;
                worker.run_until_shutdown(mse_train).map(|_| ())
            }));
        }

        let mut coord = coord_thread
            .join()
            .expect("coord thread join")
            .expect("start_from_listener succeeds");

        coord.dispatch_epoch(0).expect("dispatch_epoch(0) succeeds");

        // Drive ticks until >= 2 Cadence sync cycles fire, or timeout.
        // Cadence's `should_average` decides cycle boundaries (every
        // anchor batches per rank, uncalibrated mode). Two cycles
        // confirms the coord drives the cadence loop, not just a
        // single-shot SyncNow.
        let start = Instant::now();
        while coord.avg_count() < 2 {
            if start.elapsed() > Duration::from_secs(60) {
                panic!(
                    "end_to_end_cadence_nccl_via_coord_smoke: avg_count={} \
                     never reached 2 within 60s",
                    coord.avg_count(),
                );
            }
            coord.tick().expect("tick");
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            coord.avg_count() >= 2,
            "Cadence drives multiple AllReduce cycles: avg_count={}",
            coord.avg_count(),
        );

        // Sanity: ElChe anchor stayed in a valid range (NoGuard never
        // emits NudgeDown so anchor should equal the initial value).
        let coord_anchor = coord.el_che().anchor();
        assert_eq!(
            coord_anchor, elche_anchor,
            "NoGuard: anchor stable at initial value ({elche_anchor}), got {coord_anchor}",
        );

        coord.shutdown_workers().expect("shutdown_workers");
        coord.shutdown().expect("coord shutdown");
        for h in worker_handles {
            h.join().expect("worker thread join").expect("worker exits clean");
        }
    }

    /// End-to-end Cadence + CPU averaging smoke — the regression guard
    /// for the CPU averaging stall.
    ///
    /// CPU-device, in-process: a real [`ClusterController`] (the
    /// CpuReduce server) + a real CPU-backend [`ClusterCoordinator`] +
    /// two ranks whose param bridge all-reduces through the controller.
    /// No GPU/NCCL — the fixed machinery (CPU re-arm via `cpu_avg_state`,
    /// the Sync/Cadence `Throttle` hard barrier, the bridge `SyncAck`
    /// with no `usize::MAX / 2` sentinel) is device-independent
    /// coordinator/bridge logic, so this runs in the ordinary CPU suite.
    ///
    /// The bug it guards: the cluster CPU path re-armed `should_average`
    /// off `nccl_ack`, which the bridge satisfied with a synthetic
    /// `usize::MAX / 2` step_count. That poisoned `last_step_count`, and
    /// after 3-6 cycles the re-arm gate wedged permanently — averaging
    /// flatlined for the rest of the run (replicas trained as
    /// near-independent solos). Acceptance: `avg_count` climbs well past
    /// that old stall ceiling, proving the cycle re-arms indefinitely.
    ///
    /// `#[ignore]`: this is a heavy in-process integration smoke (a live
    /// controller + coordinator + two worker threads + TCP, on a 30s
    /// budget). It is deterministic in isolation but timing-flaky under
    /// the parallel test harness (thread/CPU contention slows the
    /// averaging round-trips). The re-arm invariant it checks is also
    /// covered deterministically by the `gate.rs` unit tests; this smoke
    /// is the explicit end-to-end repro. Run via
    /// `fdl test -- --ignored end_to_end_cadence_cpu_via_coord_smoke`
    /// (single-threaded is most reliable).
    #[test]
    #[ignore = "heavy in-process integration smoke; timing-flaky under parallel load — run explicitly"]
    fn end_to_end_cadence_cpu_via_coord_smoke() {
        let world_size = 2;
        let total_samples = 32usize;
        let batch_size = 4usize;
        let elche_anchor = 1usize;
        let num_epochs = 16usize;
        // Old poison wedged the gate at 3-6 cycles; require comfortably
        // more so a regressed re-arm path cannot pass by luck.
        let target_cycles = 12u64;

        // CpuReduce server (the piece the loopback cluster-test rig
        // lacks). Shares no dead-rank ledger — fixed membership.
        let controller = ClusterController::start(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            world_size,
            TEST_SALT,
        )
        .expect("controller starts");
        let controller_addr =
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), controller.port());
        // Workers reach the controller through their host relay. The
        // started relay handle sits buffered in `_relay_rx` until it drops
        // at end of scope, which shuts the relay down (after the controller
        // + workers are torn down).
        let (reduce_addr, _relay_rx) =
            spawn_relay(ChannelKind::Data, controller_addr, world_size, TEST_SALT);

        let (coord_listener, coord_port) = CCoord::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("coord bind succeeds");
        // Workers reach the coord through their host control relay. The
        // relay handle sits buffered in `_ctrl_relay_rx` and drops at end
        // of scope (shutting the relay down). The coord listens on
        // `coord_port`; the relay forwards to it.
        let coord_real_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), coord_port);
        let (coord_addr, _ctrl_relay_rx) =
            spawn_relay(ChannelKind::Control, coord_real_addr, world_size, TEST_SALT);

        let config_for_coord = move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Cadence,
                AverageBackend::Cpu,
                world_size,
                // Cap `max_anchor` so this (pathologically cheap-compute)
                // in-process harness can't exercise the overhead-driven
                // window GROWTH — that throughput/convergence balance is
                // ElChe's job and is validated on the real rig, not here.
                // With the window bounded, this test isolates the thing it
                // is meant to guard: that CPU averaging RE-ARMS every
                // window (the `nccl_ack`/`MAX/2` poison regression), so
                // `avg_count` keeps climbing instead of wedging at ~3-6.
                ElChe::new(world_size, elche_anchor).with_max_anchor(2),
            )
            .total_samples(total_samples)
            .batch_size(batch_size)
            .num_epochs(num_epochs)
        };
        let coord_thread = thread::spawn(move || -> Result<CCoord> {
            CCoord::start_from_listener(coord_listener, TEST_SALT, config_for_coord())
        });

        // Identical initial params from a shared ref model (no
        // broadcast_from_root needed — every rank starts equal).
        let ref_model = Linear::on_device(4, 2, Device::CPU).unwrap();
        let initial_params: Vec<Tensor> = ref_model
            .parameters()
            .iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = ref_model
            .buffers()
            .iter()
            .map(|b| b.get())
            .collect();
        drop(ref_model);

        let mut worker_handles: Vec<thread::JoinHandle<Result<()>>> = Vec::new();
        for rank_id in 0..world_size {
            let initial_params = initial_params.clone();
            let initial_buffers = initial_buffers.clone();
            worker_handles.push(thread::spawn(move || -> Result<()> {
                let cpu_client = crate::distributed::cpu_reduce::CpuReduceClient::connect(
                    reduce_addr,
                    rank_id as u32,
                    world_size as u32,
                    TEST_SALT,
                )?;
                let config = WorkerConfig {
                    rank: rank_id,
                    world_size,
                    device: Device::CPU,
                    initial_params,
                    initial_buffers,
                    total_samples,
                    batch_size,
                    seed: 42,
                    max_grad_norm: None,
                    easgd_alpha: None,
                    gamma: 1.0,
                    timeline: None,
                    policy: ApplyPolicy::Cadence,
                    save_path: None,
                    coord_liveness_timeout_secs:
                        crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
                };
                let dataset: Arc<dyn crate::data::BatchDataSet> =
                    Arc::new(TestDataset { n: total_samples });
                let worker = ClusterWorker::connect_and_build(
                    coord_addr,
                    Some(cpu_client),
                    rank_id as u32,
                    TEST_SALT,
                    config,
                    move |d| Linear::on_device(4, 2, d),
                    |params| crate::nn::SGD::new(params, 0.01, 0.0),
                    dataset,
                    None, // no NCCL comm — CPU averaging via the bridge
                    None,
                    None,
                    None, // no eval_fn
                    None, // no eval_dataset
                    None, // no outer optimizer
                )?;
                worker.run_until_shutdown(mse_train).map(|_| ())
            }));
        }

        let mut coord = coord_thread
            .join()
            .expect("coord thread join")
            .expect("start_from_listener succeeds");

        coord.dispatch_epoch(0).expect("dispatch_epoch(0) succeeds");

        // Drive ticks until the cycle count clears the old stall
        // ceiling. Two regressions would re-trip this: (1) the re-arm
        // poison (CPU re-arm forced onto `nccl_ack` + the `usize::MAX/2`
        // sentinel) wedged averaging at ~3-6 cycles; (2) the overhead
        // auto-tune ballooning the cadence window (anchor grown to
        // amortize the expensive CPU sync) pushed the reduce window past
        // the dataset, so averaging died after warmup. Either way the
        // count would stop climbing and this times out.
        let start = Instant::now();
        while coord.avg_count() < target_cycles {
            if start.elapsed() > Duration::from_secs(30) {
                panic!(
                    "end_to_end_cadence_cpu_via_coord_smoke: avg_count={} \
                     never reached {target_cycles} within 30s — CPU averaging \
                     stalled (re-arm wedge or window-blowup regression?)",
                    coord.avg_count(),
                );
            }
            coord.tick().expect("tick");
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            coord.avg_count() >= target_cycles,
            "CPU Cadence re-arms every window: avg_count={} (>= {target_cycles})",
            coord.avg_count(),
        );

        // Sanity: the window honored the `max_anchor(2)` cap we set, so
        // the steady stream of reduces above reflects re-arm working, not
        // an unbounded window masking a wedge. (Growth itself is allowed
        // and is ElChe's call; we cap it here only to keep this harness
        // focused on the re-arm regression.)
        let final_anchor = coord.el_che().anchor();
        assert!(
            final_anchor <= 2,
            "window honored max_anchor cap: anchor={final_anchor}",
        );

        coord.shutdown_workers().expect("shutdown_workers");
        coord.shutdown().expect("coord shutdown");
        controller.shutdown().expect("controller shutdown");
        for h in worker_handles {
            h.join().expect("worker thread join").expect("worker exits clean");
        }
    }

    // -----------------------------------------------------------------
    // End-to-end Sync+Cpu smoke test scaffolding
    // -----------------------------------------------------------------

    use crate::distributed::cluster_coordinator::ClusterCoordinator as CCoord;
    use crate::distributed::controller::ClusterController;
    use crate::distributed::relay::agent::{ChannelKind, RelayChannel};
    use crate::nn::Linear;

    /// Stand up a per-host [`RelayChannel`] of `kind` in front of
    /// `upstream_addr` and return the loopback address workers should dial
    /// — the relay forwards their frames to the real controller (Data) or
    /// coordinator (Control). Under the uniform-relay transport every rank
    /// reaches both through its host relay, so the in-process sims wire one
    /// in per channel (the production worker dial-redirect is launch
    /// wiring).
    ///
    /// `RelayChannel::start` blocks through the rank-handshake phase, so it
    /// runs on a background thread; the started handle arrives on the
    /// returned receiver once the workers have connected. Hold the handle
    /// alive until end of scope — its `Drop` shuts the relay down.
    fn spawn_relay(
        kind: ChannelKind,
        upstream_addr: SocketAddr,
        world_size: usize,
        salt: SessionSalt,
    ) -> (SocketAddr, std::sync::mpsc::Receiver<Result<RelayChannel>>) {
        let (listener, relay_port) =
            RelayChannel::bind(SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0)).unwrap();
        let loopback = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), relay_port);
        let ranks: Vec<u32> = (0..world_size as u32).collect();
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let started = RelayChannel::start(
                listener,
                kind,
                upstream_addr,
                "test-host".into(),
                ranks,
                world_size,
                salt,
            );
            let _ = tx.send(started);
        });
        (loopback, rx)
    }

    /// Index-deterministic dataset on CPU. Each sample's values are
    /// derived from its index, so two ranks reading disjoint partitions
    /// see DIFFERENT samples (and thus DIFFERENT gradients post-SGD).
    /// `Tensor::randn` would produce shared values across threads under
    /// libtorch's global RNG, collapsing per-rank divergence to zero
    /// and defeating the divergence-wire assertion below.
    struct TestDataset {
        n: usize,
    }
    impl crate::data::BatchDataSet for TestDataset {
        fn len(&self) -> usize {
            self.n
        }
        fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
            let n = indices.len() as i64;
            // Inputs: each sample is [idx, idx+1, idx+2, idx+3] / 10.
            // Targets: each sample is [idx * 0.1, idx * 0.2].
            // Both deterministic in `indices`, so disjoint partitions
            // → distinct gradients → non-zero post-AllReduce divergence.
            let mut x_vals: Vec<f32> = Vec::with_capacity(indices.len() * 4);
            let mut y_vals: Vec<f32> = Vec::with_capacity(indices.len() * 2);
            for &idx in indices {
                let f = idx as f32;
                x_vals.extend_from_slice(&[
                    f / 10.0,
                    (f + 1.0) / 10.0,
                    (f + 2.0) / 10.0,
                    (f + 3.0) / 10.0,
                ]);
                y_vals.extend_from_slice(&[f * 0.1, f * 0.2]);
            }
            Ok(vec![
                Tensor::from_f32(&x_vals, &[n, 4], Device::CPU)?,
                Tensor::from_f32(&y_vals, &[n, 2], Device::CPU)?,
            ])
        }
    }

    /// Mirror of `ddp_run::tests::mse_train`. MSE between Linear's
    /// output and the dataset's target tensor.
    fn mse_train(model: &Linear, batch: &[Tensor]) -> Result<Variable> {
        let input = Variable::new(batch[0].clone(), false);
        let target = Variable::new(batch[1].clone(), false);
        let output = model.forward(&input)?;
        let diff = output.sub(&target)?;
        diff.mul(&diff)?.mean()
    }

    /// Records each `report` call's `deltas` into a shared vector so a
    /// test can verify the param bridge populated the divergence triple
    /// in its `SyncAck` AND that the coord's CPU finalize state
    /// machine deferred `finish_averaging_cpu` until the SyncAcks
    /// landed. Returns `Stable` so the test's anchor stays stable.
    struct RecordingGuard {
        captured: Arc<std::sync::Mutex<Vec<Vec<f64>>>>,
    }

    impl crate::distributed::ddp_run::convergence::ConvergenceGuard for RecordingGuard {
        fn clone_box(
            &self,
        ) -> Box<dyn crate::distributed::ddp_run::convergence::ConvergenceGuard> {
            Box::new(RecordingGuard {
                captured: self.captured.clone(),
            })
        }

        fn report(
            &mut self,
            report: &crate::distributed::ddp_run::convergence::DivergenceReport,
            _k_used: usize,
            _k_max: usize,
        ) -> crate::distributed::ddp_run::convergence::ConvergenceAction {
            self.captured.lock().unwrap().push(report.deltas.clone());
            crate::distributed::ddp_run::convergence::ConvergenceAction::Stable
        }
    }

    /// End-to-end Sync+Cpu smoke test (CPU device, no NCCL): spawn
    /// `ClusterController` (data) and `ClusterCoordinator` (control)
    /// alongside 2 `ClusterWorker` threads with a trivial Linear model
    /// and `TestDataset`, run one averaging cycle via the param bridge,
    /// assert avg_count fires + workers exit cleanly AND the coord's
    /// convergence guard received strictly-positive per-rank divergence
    /// on cycle 1 (validates the bridge's
    /// [`compute_divergence`](super::compute_divergence) flowed
    /// end-to-end AND that the CPU finalize state machine deferred the
    /// guard verdict until the bridge SyncAcks populated the captures).
    #[test]
    fn end_to_end_sync_cpu_smoke() {
        let world_size = 2usize;
        let total_samples = 8usize;
        let batch_size = 4usize;

        // 1. Shared DeadRanks ledger + ClusterController on data port.
        //    No rank is dead in this smoke test; the ledger is wired
        //    for API completeness and to prove the dead-rank-aware
        //    controller path doesn't regress the happy case.
        let dead_ranks = crate::distributed::controller::DeadRanks::new(world_size);
        let controller = ClusterController::start_with_dead_ranks(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
            world_size,
            TEST_SALT,
            Arc::clone(&dead_ranks),
            None,
            None,
        )
        .expect("ClusterController::start_with_dead_ranks succeeds");
        let controller_addr =
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), controller.port());
        // Workers reach the controller through their host relay. The
        // started relay handle sits buffered in `_relay_rx` until it drops
        // at end of scope, which shuts the relay down (after the controller
        // + workers are torn down).
        let (data_addr, _relay_rx) =
            spawn_relay(ChannelKind::Data, controller_addr, world_size, TEST_SALT);

        // 2. ClusterCoordinator listener. bind() returns the port
        //    before any accept blocks; start_from_listener (which
        //    blocks) runs on a dedicated thread so workers can connect
        //    in parallel.
        let (coord_listener, coord_port) = CCoord::bind(
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 0),
        )
        .expect("coord bind succeeds");
        // Workers reach the coord through their host control relay. The
        // relay handle sits buffered in `_ctrl_relay_rx` and drops at end
        // of scope (shutting the relay down). The coord listens on
        // `coord_port`; the relay forwards to it.
        let coord_real_addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), coord_port);
        let (coord_addr, _ctrl_relay_rx) =
            spawn_relay(ChannelKind::Control, coord_real_addr, world_size, TEST_SALT);

        // RecordingGuard captures the deltas every `finish_averaging_*`
        // pass — proves both the bridge wire AND the deferred
        // finalize are correct end-to-end on cycle 1.
        let captured_deltas: Arc<std::sync::Mutex<Vec<Vec<f64>>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured_for_coord = Arc::clone(&captured_deltas);
        let dead_ranks_for_coord = Arc::clone(&dead_ranks);
        let config_for_coord = move || {
            ClusterCoordinatorConfig::new(
                ApplyPolicy::Sync,
                AverageBackend::Cpu,
                world_size,
                crate::distributed::ddp::ElChe::new(world_size, 1),
            )
            .with_convergence_guard(Box::new(RecordingGuard {
                captured: captured_for_coord,
            }))
            .dead_ranks(dead_ranks_for_coord)
            .total_samples(total_samples)
            .batch_size(batch_size)
            .num_epochs(1)
        };
        let coord_thread = thread::spawn(move || -> Result<CCoord> {
            CCoord::start_from_listener(coord_listener, TEST_SALT, config_for_coord())
        });

        // 3. Build a reference Linear model on CPU to capture initial
        //    params/buffers. Each worker thread's model_factory builds
        //    its own fresh Linear; WorkerConfig.initial_params overrides
        //    the random init so all ranks align at startup.
        let ref_model = Linear::on_device(4, 2, Device::CPU).unwrap();
        let initial_params: Vec<Tensor> = ref_model
            .parameters()
            .iter()
            .map(|p| p.variable.data())
            .collect();
        let initial_buffers: Vec<Tensor> = ref_model
            .buffers()
            .iter()
            .map(|b| b.get())
            .collect();
        drop(ref_model);

        // 4. Spawn worker threads. Each connects to the coord (control)
        //    + builds a CpuReduceClient (data). connect_and_build is
        //    blocking on both handshakes; the coord_thread above
        //    unblocks once both workers handshake.
        let salt = TEST_SALT;
        let mut worker_handles: Vec<thread::JoinHandle<Result<()>>> = Vec::new();
        for rank_id in 0..world_size {
            let initial_params = initial_params.clone();
            let initial_buffers = initial_buffers.clone();
            worker_handles.push(thread::spawn(move || -> Result<()> {
                let config = WorkerConfig {
                    rank: rank_id,
                    world_size,
                    device: Device::CPU,
                    initial_params,
                    initial_buffers,
                    total_samples,
                    batch_size,
                    seed: 42,
                    max_grad_norm: None,
                    easgd_alpha: None,
                    gamma: 1.0,
                    timeline: None,
                    policy: ApplyPolicy::Sync,
                    save_path: None,
                    coord_liveness_timeout_secs:
                        crate::distributed::ddp_run::DEFAULT_COORD_LIVENESS_TIMEOUT_SECS,
                };
                let dataset: Arc<dyn crate::data::BatchDataSet> =
                    Arc::new(TestDataset { n: total_samples });
                let cpu_client = crate::distributed::cpu_reduce::CpuReduceClient::connect(
                    data_addr,
                    rank_id as u32,
                    world_size as u32,
                    salt,
                )?;
                let worker = ClusterWorker::connect_and_build(
                    coord_addr,
                    Some(cpu_client),
                    rank_id as u32,
                    salt,
                    config,
                    |d| Linear::on_device(4, 2, d),
                    |params| crate::nn::SGD::new(params, 0.01, 0.0),
                    dataset,
                    None, // no NCCL
                    None, // no checkpoint
                    None, // no epoch_fn
                    None, // no eval_fn
                    None, // no eval_dataset
                    None, // no outer optimizer
                )?;
                worker.run_until_shutdown(mse_train).map(|_| ())
            }));
        }

        // 5. Coord thread unblocks after both worker handshakes; recover
        //    the configured coord.
        let mut coord = coord_thread
            .join()
            .expect("coord thread join")
            .expect("start_from_listener succeeds");

        // 6. Dispatch the only epoch + drive ticks until at least one
        //    averaging cycle fires. Bound the wall budget so a buggy
        //    coord doesn't hang the suite. The test runs in ~1s in
        //    isolation; under heavy parallel CPU contention (full
        //    `fdl test` suite, ~1400 tests in flight) the libtorch
        //    forward/backward + CpuReduceClient round-trip can stretch
        //    well past the NCCL smokes' 30s budget. Pick 60s with
        //    explicit cleanup below so a hit still terminates the test
        //    process promptly instead of leaving orphan worker
        //    threads.
        coord.dispatch_epoch(0).expect("dispatch_epoch(0) succeeds");
        let start = Instant::now();
        let timed_out = loop {
            if coord.avg_count() > 0 {
                break false;
            }
            if start.elapsed() > Duration::from_secs(60) {
                break true;
            }
            coord.tick().expect("tick");
            thread::sleep(Duration::from_millis(10));
        };
        if timed_out {
            // Workers are blocked inside `wait_for_epoch_plan` on the
            // inbound bridge's mpsc — without an explicit Shutdown
            // broadcast they idle forever, and the panic below would
            // leave orphan threads + bridge sockets around, hanging
            // the test harness for a further heartbeat-timeout window
            // (and racking up cargo's "test has been running for over
            // 60 seconds" warning unnecessarily). Wake them, join
            // them, then panic so the failure surfaces immediately.
            coord.shutdown_workers().ok();
            coord.shutdown().ok();
            for h in worker_handles {
                let _ = h.join();
            }
            controller.shutdown().ok();
            panic!(
                "end_to_end_sync_cpu_smoke: avg_count never advanced \
                 (no averaging cycle observed within 60s — likely \
                 parallel-load CPU starvation, see test comment)"
            );
        }
        assert!(coord.avg_count() >= 1, "at least one averaging cycle");

        // 6b. With the deferred finalize gated on `nccl_sync_divergence`
        //     (not `nccl_ack`), cycle 1's guard sees REAL divergence:
        //     the coord waits for every bridge SyncAck to populate the
        //     divergence slot before running `finish_averaging_cpu`.
        //     The test asserts at-least-one rank reported a strictly-
        //     positive delta — sufficient evidence the bridge wire
        //     propagated `compute_divergence` end-to-end. A single 0.0
        //     is permitted because `compute_divergence` legitimately
        //     returns 0.0 when `post_norm <= 1e-10` (degenerate avg).
        //     A regression to gating on `nccl_ack` would surface as
        //     cycle-1 deltas == [0.0, 0.0] (all-Nones sentinel
        //     `unwrap_or(0.0)`) — `any` still catches that.
        //
        //     Failure path: capture into booleans, drive the full
        //     teardown sequence (steps 7–9) unconditionally, then
        //     panic at the end. Bare `assert!` in this position would
        //     unwind with worker threads still parked inside
        //     `wait_for_epoch_plan`, leaving orphan mpsc receivers
        //     and dangling sockets in the cargo test harness.
        let cycles = captured_deltas.lock().unwrap().clone();
        let no_cycles = cycles.is_empty();
        let (first_len, has_positive, first_dump) = if no_cycles {
            (0, false, Vec::new())
        } else {
            let f = &cycles[0];
            (
                f.len(),
                f.iter().any(|d| d.is_finite() && *d > 0.0),
                f.clone(),
            )
        };
        let len_ok = first_len == world_size;
        let div_check_passed = !no_cycles && len_ok && has_positive;

        // 7. Send Shutdown to workers and tear down the coord (always).
        coord.shutdown_workers().ok();
        coord.shutdown().ok();

        // 8. Collect worker join results without panicking; we want
        //    every thread joined before either the divergence-check
        //    panic or the worker-failure panic fires.
        let worker_results: Vec<(usize, std::thread::Result<Result<()>>)> =
            worker_handles
                .into_iter()
                .enumerate()
                .map(|(rank_id, h)| (rank_id, h.join()))
                .collect();

        // 9. Shut the controller down.
        controller.shutdown().ok();

        // Now panic if the divergence check failed.
        assert!(
            div_check_passed,
            "smoke divergence check failed: cycles_seen={} first_len={} \
             (expected {}) any_positive={} first_deltas={:?}",
            cycles.len(),
            first_len,
            world_size,
            has_positive,
            first_dump,
        );

        // Surface worker errors AFTER divergence + teardown succeeded.
        for (rank_id, r) in worker_results {
            let r = r.expect("worker thread join");
            r.unwrap_or_else(|e| {
                panic!("worker rank {rank_id} run_until_shutdown: {e}");
            });
        }
    }

    // ---- inbound-bridge failure discipline --------------------------------
    //
    // Losing the coordinator link while the main thread is inside an NCCL
    // collective is unreachable by the injected ControlMsg::Shutdown (the
    // control channel is never read there). The escape hatch declares all
    // peers dead in the local ledger so the NCCL watchdog aborts the comm
    // and the rank exits instead of zombifying — but ONLY on an abnormal
    // link loss: a clean Shutdown frame followed by EOF is the normal
    // teardown sequence and must leave the ledger untouched (the main
    // thread may still be draining the final coherent reduce).

    fn inbound_test_rig(
        world_size: usize,
        coord_liveness_timeout_secs: u64,
    ) -> (
        std::net::TcpStream,                                   // coord side
        std::thread::JoinHandle<()>,                            // inbound
        std::sync::mpsc::Receiver<ControlMsg>,                  // control_rx
        Arc<crate::distributed::controller::DeadRanks>,         // ledger
    ) {
        use std::net::{TcpListener, TcpStream};
        let listener =
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind");
        let addr = listener.local_addr().expect("addr");
        let coord_side = TcpStream::connect(addr).expect("connect");
        let (mut worker_side, _) = listener.accept().expect("accept");
        // Mirror production (connect_and_build): a read timeout so a silent
        // link surfaces as periodic WouldBlock, letting the coord-liveness
        // deadline fire rather than blocking the read forever.
        worker_side
            .set_read_timeout(Some(std::time::Duration::from_millis(100)))
            .expect("set_read_timeout");

        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (control_tx, control_rx) = std::sync::mpsc::channel();
        let (timing_tx, _timing_rx) = std::sync::mpsc::channel();
        let dead = crate::distributed::controller::DeadRanks::new(world_size);
        let mailbox = Arc::new(std::sync::Mutex::new(None));
        let dead_for_loop = Arc::clone(&dead);
        let handle = std::thread::spawn(move || {
            inbound_loop(
                0,
                &mut worker_side,
                &TEST_SALT,
                &shutdown,
                &control_tx,
                &dead_for_loop,
                &mailbox,
                &timing_tx,
                coord_liveness_timeout_secs,
            );
        });
        (coord_side, handle, control_rx, dead)
    }

    #[test]
    fn inbound_eof_without_shutdown_poisons_peer_ledger() {
        let (coord_side, handle, control_rx, dead) = inbound_test_rig(3, 30);
        // Abnormal loss: the coordinator link drops with no Shutdown frame.
        drop(coord_side);
        handle.join().expect("inbound join");
        assert!(!dead.is_dead(0), "own rank must never be poisoned");
        assert!(dead.is_dead(1) && dead.is_dead(2),
            "all peers must be declared dead so the NCCL watchdog can \
             abort an in-flight collective");
        // The recv-parked escape path still fires too.
        assert!(matches!(control_rx.recv(), Ok(ControlMsg::Shutdown)));
    }

    #[test]
    fn inbound_eof_after_clean_shutdown_leaves_ledger_alone() {
        let (mut coord_side, handle, control_rx, dead) = inbound_test_rig(3, 30);
        // Clean teardown: Shutdown frame first, then the link drops.
        let frame = crate::distributed::wire::ControlFrame::encode(
            &TEST_SALT,
            crate::distributed::wire::MsgKind::Control,
            &crate::distributed::wire::ControlMsgWire::Shutdown,
        )
        .expect("encode shutdown");
        let mut bytes = Vec::new();
        frame.write_to(&mut bytes).expect("serialize frame");
        crate::distributed::relay::mux::write_len_framed(&mut coord_side, &bytes)
            .expect("write len-framed");
        drop(coord_side);
        handle.join().expect("inbound join");
        assert_eq!(
            dead.dead_count(),
            0,
            "clean Shutdown-then-EOF is the normal teardown — poisoning \
             here would abort a final coherent reduce mid-flight"
        );
        assert!(matches!(control_rx.recv(), Ok(ControlMsg::Shutdown)));
    }

    // Wedged-open coordinator: the link stays ALIVE (no EOF, no error) but the
    // coord sends nothing — the SIGSTOP / deadlock case. Neither the EOF nor
    // the parse-error escape hatch can fire, so without a liveness deadline the
    // loop would poll WouldBlock forever. After `coord_liveness_timeout_secs`
    // of silence it must bail exactly like a hard link drop: poison peers (so
    // the NCCL watchdog aborts the in-flight collective) and inject Shutdown.
    #[test]
    fn inbound_wedged_open_coord_trips_liveness_deadline() {
        // 1s deadline keeps the test quick; the 100ms rig read-timeout gives
        // ~10 WouldBlock polls before the deadline trips.
        let (coord_side, handle, control_rx, dead) = inbound_test_rig(3, 1);
        // Hold the connection OPEN and SILENT — the wedged-open condition.
        // Dropping coord_side here would fire EOF instead and test the wrong
        // path, so keep it bound until after the deadline has been detected.
        handle.join().expect("inbound join");
        assert!(!dead.is_dead(0), "own rank must never be poisoned");
        assert!(
            dead.is_dead(1) && dead.is_dead(2),
            "a coord silent past the liveness deadline must poison peers so \
             the NCCL watchdog can abort an in-flight collective"
        );
        assert!(
            matches!(control_rx.recv(), Ok(ControlMsg::Shutdown)),
            "the recv-parked inner must be woken with Shutdown"
        );
        drop(coord_side);
    }
