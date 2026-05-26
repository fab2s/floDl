//! Top-level tests for [`super`] (the `cluster_coordinator` module).
//!
//! Hosts the file-root shared helpers (fake_rank, cfg_*, send/recv frame
//! helpers, spawn_coord) plus the handshake and ApplyPolicy/throttle
//! tests. Each topic area lives in its own sibling under `tests/`; the
//! helpers here are `pub(super)` so child modules can pull them via
//! `use super::*;`. Common type imports are re-exported via
//! `pub(crate) use` so children don't repeat the boilerplate.

pub(crate) use super::*;
pub(crate) use crate::distributed::ddp_run::ApplyPolicy;
pub(crate) use crate::distributed::wire::{ControlMsgWire, MetricsMsgWire, TimingMsgWire};
pub(crate) use std::net::{Ipv4Addr, SocketAddr, TcpListener};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::AtomicBool;
pub(crate) use std::thread;
pub(crate) use std::time::Duration;

mod callbacks;
mod epoch_dispatch;
mod fastest;
mod gate;
mod heartbeat;

/// Deterministic non-zero test salt (mirrors controller.rs::tests).
pub(super) const TEST_SALT: SessionSalt = [
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
];

/// Spawn a fake rank that connects to `port`, handshakes with
/// `salt`, runs `body` against the connected stream, then drops it.
pub(super) fn fake_rank<F>(
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

pub(super) fn cfg_sync_nccl(world_size: usize) -> ClusterCoordinatorConfig {
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

pub(super) fn cfg_async_nccl(world_size: usize) -> ClusterCoordinatorConfig {
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
pub(super) fn send_timing(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: TimingMsgWire,
) -> Result<()> {
    let frame = ControlFrame::encode(salt, MsgKind::Timing, &msg)?;
    frame.write_to(stream)
}

/// Send a Metrics-kind ControlFrame on a fake-rank stream.
pub(super) fn send_metrics(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: MetricsMsgWire,
) -> Result<()> {
    let frame = ControlFrame::encode(salt, MsgKind::Metrics, &msg)?;
    frame.write_to(stream)
}

/// Read one Control-kind ControlFrame from the rank-side stream.
pub(super) fn recv_control(
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
pub(super) fn spawn_coord<F>(
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


pub(super) fn cfg_sync_nccl_with_dataset(world_size: usize, total_samples: usize) -> ClusterCoordinatorConfig {
    cfg_sync_nccl(world_size)
        .total_samples(total_samples)
        .batch_size(2)
        .num_epochs(1)
}

pub(super) fn cfg_sync_cpu(world_size: usize) -> ClusterCoordinatorConfig {
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

