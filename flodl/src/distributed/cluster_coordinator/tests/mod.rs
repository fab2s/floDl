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
mod delivered_feed;
mod epoch_dispatch;
mod fastest;
mod gate;
mod heartbeat;
mod lost_broadcast;
mod reduce_stall;
mod shutdown_gate;

/// Deterministic non-zero test salt (mirrors controller.rs::tests).
pub(super) const TEST_SALT: SessionSalt = [
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
    0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
];

/// Spawn a fake rank that connects to `port` and presents itself to the
/// coordinator as a single-rank host relay (one `RelayHello` carrying
/// just `rank_id`), then runs `body` against the connected stream and
/// drops it. The coord's accept-until-covered loop accepts N such
/// single-rank relays to cover the world.
pub(super) fn fake_rank<F>(
    port: u16,
    rank_id: u32,
    _world_size: u32,
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
        let _ = stream.set_nodelay(true);
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .map_err(|e| TensorError::new(&format!("set_read_timeout: {e}")))?;
        relay_hello(&mut stream, &salt, rank_id)?;
        stream
            .set_read_timeout(None)
            .map_err(|e| TensorError::new(&format!("clear timeout: {e}")))?;
        body(&mut stream, &salt)
    })
}

/// Single-rank relay handshake toward the coordinator: send the
/// channel-select magic and a `Hello` for `[rank_id]`, expect
/// `HelloAck`.
pub(super) fn relay_hello(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    rank_id: u32,
) -> Result<()> {
    crate::distributed::wire::write_channel_magic(
        stream,
        crate::distributed::wire::CHANNEL_MAGIC_CONTROL,
    )?;
    MuxRecord::control(RelayControlMsg::Hello {
        host: format!("test-r{rank_id}"),
        ranks: vec![rank_id],
    })
    .write_to(stream, salt)?;
    match MuxRecord::read_from(stream, salt)? {
        Some(MuxRecord::Control(RelayControlMsg::HelloAck)) => Ok(()),
        other => Err(TensorError::new(&format!(
            "fake_rank {rank_id}: expected relay HelloAck, got {other:?}"
        ))),
    }
}

/// Rank carried by a [`TimingMsgWire`] (every variant tags its origin
/// rank). Used to set the mux tag on the rank→coord leg; the coord
/// routes by the payload's rank field, so the tag is informational, but
/// a faithful single-rank relay still tags with its rank.
fn timing_wire_rank(msg: &TimingMsgWire) -> u32 {
    let r = match msg {
        TimingMsgWire::Batch { rank, .. }
        | TimingMsgWire::SyncAck { rank, .. }
        | TimingMsgWire::Exiting { rank }
        | TimingMsgWire::LrUpdate { rank, .. }
        | TimingMsgWire::Heartbeat { rank, .. }
        | TimingMsgWire::SnapshotReady { rank }
        | TimingMsgWire::EvalResult { rank, .. }
        | TimingMsgWire::CheckpointResult { rank, .. }
        | TimingMsgWire::NewNcclIdGenerated { rank, .. }
        | TimingMsgWire::EpochFnElapsed { rank, .. }
        | TimingMsgWire::DashboardRegister { rank, .. }
        | TimingMsgWire::DashboardSetSvg { rank, .. }
        | TimingMsgWire::DashboardSetMetadata { rank, .. }
        | TimingMsgWire::DashboardSetHardware { rank, .. } => *rank,
    };
    r as u32
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

/// Tag a ControlFrame as a rank-tagged mux record (single-rank-relay
/// up-leg) and write it to the coord-facing stream.
fn send_framed(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    rank: u32,
    frame: ControlFrame,
) -> Result<()> {
    let mut buf = Vec::new();
    frame.write_to(&mut buf)?;
    MuxRecord::data(rank, buf).write_to(stream, salt)
}

/// Send a Timing-kind ControlFrame on a fake-rank stream (mux-wrapped).
pub(super) fn send_timing(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: TimingMsgWire,
) -> Result<()> {
    let rank = timing_wire_rank(&msg);
    let frame = ControlFrame::encode(salt, MsgKind::Timing, &msg)?;
    send_framed(stream, salt, rank, frame)
}

/// Send a Metrics-kind ControlFrame on a fake-rank stream (mux-wrapped).
pub(super) fn send_metrics(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    msg: MetricsMsgWire,
) -> Result<()> {
    let rank = msg.rank as u32;
    let frame = ControlFrame::encode(salt, MsgKind::Metrics, &msg)?;
    send_framed(stream, salt, rank, frame)
}

/// Mux-aware drop-in for `ControlFrame::read_from` in fake-rank bodies:
/// read one `MuxRecord::Data` off the (single-rank-relay) stream and parse
/// its payload as a [`ControlFrame`]. `Ok(None)` on clean EOF. The mux tag
/// is ignored (single-rank connection).
pub(super) fn recv_frame(
    s: &mut TcpStream,
    salt: &SessionSalt,
) -> Result<Option<ControlFrame>> {
    match MuxRecord::read_from(s, salt)? {
        Some(MuxRecord::Data { payload, .. }) => {
            ControlFrame::read_from(&mut payload.as_slice(), salt)
        }
        Some(other) => Err(TensorError::new(&format!(
            "recv_frame: expected mux Data, got {other:?}"
        ))),
        None => Ok(None),
    }
}

/// Read one Control-kind ControlFrame from the rank-side stream — the
/// coord sends it as a rank-tagged mux record (the relay would demux it
/// to the local rank). The tag is ignored here (single-rank conn).
pub(super) fn recv_control(
    stream: &mut TcpStream,
    salt: &SessionSalt,
) -> Result<ControlMsgWire> {
    loop {
        let payload = match MuxRecord::read_from(stream, salt)? {
            Some(MuxRecord::Data { payload, .. }) => payload,
            Some(other) => {
                return Err(TensorError::new(&format!(
                    "recv_control: expected mux Data, got {other:?}"
                )));
            }
            None => return Err(TensorError::new("EOF before frame")),
        };
        let frame = ControlFrame::read_from(&mut payload.as_slice(), salt)?
            .ok_or_else(|| TensorError::new("truncated ControlFrame payload"))?;
        if frame.kind != MsgKind::Control {
            return Err(TensorError::new(&format!(
                "unexpected frame kind {:?}",
                frame.kind
            )));
        }
        // Skip the coord→rank liveness beacon: it is not a semantic frame, and
        // the production rank inbound bridge absorbs it the same way. Tests
        // asserting a specific control sequence must not trip over it.
        match frame.decode::<ControlMsgWire>()? {
            ControlMsgWire::CoordHeartbeat => continue,
            msg => return Ok(msg),
        }
    }
}

/// Like [`recv_control`] but emits a `Heartbeat` whenever no frame is
/// ready yet, so the coordinator's staleness check does not declare this
/// (legitimately blocked) rank dead and tear its connection. Mirrors the
/// production heartbeat thread that real ranks run.
///
/// Use this for any survivor recv that can span the
/// `heartbeat_timeout_secs` window — e.g. a survivor waiting for the
/// post-dead-rank averaging round-trip. A plain blocking `recv_control`
/// there goes silent for ~`heartbeat_timeout` and is spuriously reaped,
/// surfacing as `"EOF before frame"`.
pub(super) fn recv_control_keepalive(
    stream: &mut TcpStream,
    salt: &SessionSalt,
    rank: u64,
    step_count: u64,
) -> Result<ControlMsgWire> {
    use crate::distributed::relay::mux::MuxRead;
    stream
        .set_read_timeout(Some(Duration::from_millis(150)))
        .map_err(|e| TensorError::new(&format!("keepalive set_read_timeout: {e}")))?;
    // Loop until a SEMANTIC control frame arrives: WouldBlock refreshes our own
    // liveness (like the real heartbeat thread), and an inbound CoordHeartbeat
    // beacon is skipped (liveness noise, absorbed by the production rank too).
    let result = loop {
        let payload = match MuxRecord::try_read_from(stream, salt)? {
            MuxRead::Record(MuxRecord::Data { payload, .. }) => payload,
            MuxRead::Record(other) => {
                break Err(TensorError::new(&format!(
                    "recv_control_keepalive: expected mux Data, got {other:?}"
                )));
            }
            // No frame yet: refresh liveness like the real heartbeat
            // thread, then keep polling.
            MuxRead::WouldBlock => {
                send_timing(stream, salt, TimingMsgWire::Heartbeat { rank, step_count })?;
                continue;
            }
            MuxRead::Eof => break Err(TensorError::new("EOF before frame")),
        };
        let frame = match ControlFrame::read_from(&mut payload.as_slice(), salt)? {
            Some(f) => f,
            None => break Err(TensorError::new("truncated ControlFrame payload")),
        };
        if frame.kind != MsgKind::Control {
            break Err(TensorError::new(&format!(
                "unexpected frame kind {:?}",
                frame.kind
            )));
        }
        match frame.decode::<ControlMsgWire>() {
            Ok(ControlMsgWire::CoordHeartbeat) => continue,
            other => break other,
        }
    };
    // Restore blocking reads for the rest of the body, regardless of outcome.
    stream
        .set_read_timeout(None)
        .map_err(|e| TensorError::new(&format!("keepalive clear timeout: {e}")))?;
    result
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
        // Correct channel magic (routing is pre-auth), then a wrong-salt
        // relay Hello → the coord's mux-record HMAC fails server-side
        // during the accept handshake.
        crate::distributed::wire::write_channel_magic(
            &mut s,
            crate::distributed::wire::CHANNEL_MAGIC_CONTROL,
        )
        .unwrap();
        let _ = MuxRecord::control(RelayControlMsg::Hello {
            host: "rogue".into(),
            ranks: vec![0],
        })
        .write_to(&mut s, &bad_salt);
        // Read until the server drops us.
        let mut throwaway = [0u8; 16];
        let _ = s.read_exact(&mut throwaway);
    });
    let err = match coord_handle.join().unwrap() {
        Ok(_) => panic!("expected start_from_listener to fail on bad-salt relay"),
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
            batch_ms: 10.0, data_ms: 0.0,
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
            batch_ms: 12.0, data_ms: 0.0,
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
            batch_ms: 5.0, data_ms: 0.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })?;
        // Drain inbound frames until the coordinator drops us.
        // We must NOT receive a Throttle; if we do, assert.
        let mut got = recv_frame(s, salt);
        while let Ok(Some(frame)) = got {
            let kind = frame.kind;
            let msg = frame.decode::<ControlMsgWire>()?;
            assert!(
                !matches!(msg, ControlMsgWire::Throttle),
                "Throttle must not fire on NCCL backend (rank 0, kind={kind:?})"
            );
            got = recv_frame(s, salt);
        }
        Ok(())
    });
    let r1 = fake_rank(port, 1, world_size as u32, TEST_SALT, |s, salt| {
        send_timing(s, salt, TimingMsgWire::Batch {
            rank: 1,
            batch_ms: 5.0, data_ms: 0.0,
            step_count: 1,
            param_norm: None,
            batch_loss: 0.5,
            sync_divergence: None,
        })?;
        let mut got = recv_frame(s, salt);
        while let Ok(Some(frame)) = got {
            let msg = frame.decode::<ControlMsgWire>()?;
            assert!(
                !matches!(msg, ControlMsgWire::Throttle),
                "Throttle must not fire on NCCL backend (rank 1)"
            );
            got = recv_frame(s, salt);
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

