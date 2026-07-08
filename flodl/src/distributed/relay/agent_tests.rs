//! Relay-agent tests: the mux byte-router core (incl. fault paths — the
//! deadlock-prone heart), the data-channel fold station, and end-to-end
//! handshake termination on both channels. All over loopback TCP; no GPU.

use super::*;
use crate::distributed::controller::{
    read_round_frame, write_round_frame, RoundKind, TensorPayload, DTYPE_F32,
};
use crate::distributed::relay::mux::read_len_framed;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

const SALT: SessionSalt = [
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x01,
];

const READ_TIMEOUT: Duration = Duration::from_secs(5);
const JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// A connected loopback TCP pair: `(client, server)`. Both nodelay'd.
fn loopback_pair() -> (TcpStream, TcpStream) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (server, _) = l.accept().unwrap();
    client.set_nodelay(true).unwrap();
    server.set_nodelay(true).unwrap();
    (client, server)
}

/// Start the mux core over established streams. Returns the shutdown flag
/// and join handles.
fn start_mux(
    kind: ChannelKind,
    rank_streams: Vec<(u32, TcpStream)>,
    upstream: TcpStream,
) -> (Arc<AtomicBool>, Vec<JoinHandle<()>>) {
    let shutdown = Arc::new(AtomicBool::new(false));
    let threads =
        spawn_mux(kind, rank_streams, upstream, SALT, Arc::clone(&shutdown)).unwrap();
    (shutdown, threads)
}

/// One-tensor f32 `RoundFrame` with the given values and mass.
fn frame(vals: &[f32], weight: f64) -> RoundFrame {
    RoundFrame {
        tensors: vec![TensorPayload {
            dtype: DTYPE_F32,
            shape: vec![vals.len() as u32],
            bytes: vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }],
        kind: RoundKind::Model,
        weight,
    }
}

/// Serialize `frame` as the rank↔relay loopback blob (RoundFrame bytes,
/// salt-signed).
fn frame_blob(frame: &RoundFrame) -> Vec<u8> {
    let mut buf = Vec::new();
    write_round_frame(&mut buf, frame, &SALT).unwrap();
    buf
}

/// Parse a `HostFrame` payload back into a `RoundFrame`.
fn parse_frame(payload: &[u8]) -> RoundFrame {
    read_round_frame(&mut &payload[..], &SALT)
        .unwrap()
        .expect("complete RoundFrame payload")
}

/// f32 view of a frame's single tensor.
fn frame_vals(frame: &RoundFrame) -> Vec<f32> {
    frame.tensors[0]
        .bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Join `threads` within `JOIN_TIMEOUT`, panicking on a hang (so a
/// deadlock fails the test loudly instead of stalling the suite).
fn join_within(threads: Vec<JoinHandle<()>>) {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for t in threads {
            let _ = t.join();
        }
        let _ = tx.send(());
    });
    rx.recv_timeout(JOIN_TIMEOUT)
        .expect("relay mux threads did not shut down in time (deadlock?)");
}

// --- mux core ---

#[test]
fn mux_forwards_rank_blob_to_controller_tagged() {
    let (mut rank_client, rank_relay) = loopback_pair();
    let (relay_up, mut controller) = loopback_pair();
    controller.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (shutdown, threads) = start_mux(ChannelKind::Control, vec![(2, rank_relay)], relay_up);

    write_len_framed(&mut rank_client, &[1, 2, 3, 4]).unwrap();
    let rec = MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap();
    assert_eq!(rec, MuxRecord::data(2, vec![1, 2, 3, 4]));

    shutdown.store(true, Ordering::SeqCst);
    drop(rank_client);
    drop(controller);
    join_within(threads);
}

#[test]
fn mux_routes_reply_to_correct_rank() {
    let (mut rank_client, rank_relay) = loopback_pair();
    rank_client.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (relay_up, mut controller) = loopback_pair();
    let (shutdown, threads) = start_mux(ChannelKind::Control, vec![(3, rank_relay)], relay_up);

    // Controller sends an averaged-reply blob tagged for rank 3.
    MuxRecord::data(3, vec![9, 8, 7])
        .write_to(&mut controller, &SALT)
        .unwrap();
    let blob = read_len_framed(&mut rank_client).unwrap().unwrap();
    assert_eq!(blob, vec![9, 8, 7]);

    shutdown.store(true, Ordering::SeqCst);
    drop(rank_client);
    drop(controller);
    join_within(threads);
}

#[test]
fn mux_demuxes_two_ranks_both_directions() {
    let (mut c0, r0) = loopback_pair();
    let (mut c1, r1) = loopback_pair();
    c0.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    c1.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (relay_up, mut controller) = loopback_pair();
    controller.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (shutdown, threads) = start_mux(ChannelKind::Control, vec![(0, r0), (1, r1)], relay_up);

    // Both ranks send up; the relay tags each. Order across ranks is
    // nondeterministic (two reader threads, one queue), so collect by tag.
    write_len_framed(&mut c0, &[0xA0]).unwrap();
    write_len_framed(&mut c1, &[0xB1, 0xB2]).unwrap();
    let mut got: HashMap<u32, Vec<u8>> = HashMap::new();
    for _ in 0..2 {
        match MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap() {
            MuxRecord::Data { rank, payload } => {
                got.insert(rank, payload);
            }
            other => panic!("expected Data, got {other:?}"),
        }
    }
    assert_eq!(got.get(&0), Some(&vec![0xA0]));
    assert_eq!(got.get(&1), Some(&vec![0xB1, 0xB2]));

    // Replies route back to the addressed rank only.
    MuxRecord::data(1, vec![0xCC])
        .write_to(&mut controller, &SALT)
        .unwrap();
    MuxRecord::data(0, vec![0xDD, 0xEE])
        .write_to(&mut controller, &SALT)
        .unwrap();
    assert_eq!(read_len_framed(&mut c1).unwrap().unwrap(), vec![0xCC]);
    assert_eq!(read_len_framed(&mut c0).unwrap().unwrap(), vec![0xDD, 0xEE]);

    shutdown.store(true, Ordering::SeqCst);
    drop(c0);
    drop(c1);
    drop(controller);
    join_within(threads);
}

#[test]
fn mux_emits_rank_exit_on_disconnect() {
    let (rank_client, rank_relay) = loopback_pair();
    let (relay_up, mut controller) = loopback_pair();
    controller.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    // Single rank: when it exits, the relay should also self-shut-down
    // (no ranks left to serve), so we DON'T set the shutdown flag here —
    // join_within proves the auto-shutdown path converges.
    let (_shutdown, threads) = start_mux(ChannelKind::Control, vec![(5, rank_relay)], relay_up);

    drop(rank_client); // rank process gone
    let rec = MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap();
    assert_eq!(
        rec,
        MuxRecord::control(RelayControlMsg::RankExit { rank: 5 })
    );

    drop(controller);
    join_within(threads); // last rank gone -> shutdown set internally
}

#[test]
fn mux_rank_exit_then_survivors_keep_flowing() {
    let (c0, r0) = loopback_pair();
    let (mut c1, r1) = loopback_pair();
    let (relay_up, mut controller) = loopback_pair();
    controller.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (shutdown, threads) = start_mux(ChannelKind::Control, vec![(0, r0), (1, r1)], relay_up);

    // Rank 0 dies; rank 1 keeps sending. The relay must surface RankExit{0}
    // and still forward rank 1's frames (no barrier wedge).
    drop(c0);
    write_len_framed(&mut c1, &[0x42]).unwrap();

    let mut saw_exit = false;
    let mut saw_data1 = false;
    for _ in 0..2 {
        match MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap() {
            MuxRecord::Control(RelayControlMsg::RankExit { rank: 0 }) => saw_exit = true,
            MuxRecord::Data { rank: 1, payload } => {
                assert_eq!(payload, vec![0x42]);
                saw_data1 = true;
            }
            other => panic!("unexpected record {other:?}"),
        }
    }
    assert!(saw_exit, "missing RankExit for dead rank 0");
    assert!(saw_data1, "survivor rank 1's frame was not forwarded");

    shutdown.store(true, Ordering::SeqCst);
    drop(c1);
    drop(controller);
    join_within(threads);
}

#[test]
fn mux_shuts_down_on_upstream_eof() {
    let (rank_client, rank_relay) = loopback_pair();
    let (relay_up, controller) = loopback_pair();
    let (_shutdown, threads) = start_mux(ChannelKind::Control, vec![(0, rank_relay)], relay_up);

    // Controller process dies: upstream EOF must tear the host down even
    // though the local rank is still alive.
    drop(controller);
    join_within(threads);
    drop(rank_client);
}

// --- end-to-end handshake termination via RelayChannel::start ---

fn data_rank_handshake(stream: &mut TcpStream, rank: u32, world_size: u32) {
    use crate::distributed::controller::{
        HANDSHAKE_MAGIC_CONTROLLER_ACK, HANDSHAKE_MAGIC_RANK, PROTOCOL_VERSION,
    };
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&HANDSHAKE_MAGIC_RANK.to_le_bytes());
    buf[4..8].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    buf[8..12].copy_from_slice(&rank.to_le_bytes());
    buf[12..16].copy_from_slice(&world_size.to_le_bytes());
    stream.write_all(&buf).unwrap();
    let mut ack = [0u8; 8];
    stream.read_exact(&mut ack).unwrap();
    let magic = u32::from_le_bytes(ack[0..4].try_into().unwrap());
    assert_eq!(magic, HANDSHAKE_MAGIC_CONTROLLER_ACK);
}

/// Drive a fake controller: accept the relay's upstream connection, read
/// the `Hello`, ack it, echo one inbound `Data` back to its rank, then
/// drain until EOF. Reports the announced ranks + first payload over the
/// channels.
fn spawn_fake_controller(
    controller: TcpListener,
    hello_tx: mpsc::Sender<Vec<u32>>,
    data_tx: mpsc::Sender<(u32, Vec<u8>)>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let (mut up, _) = controller.accept().unwrap();
        up.set_nodelay(true).unwrap();
        // Channel magic, then Hello.
        crate::distributed::wire::expect_channel_magic(
            &mut up,
            crate::distributed::wire::CHANNEL_MAGIC_CONTROL,
            "fake controller",
        )
        .unwrap();
        match MuxRecord::read_from(&mut up, &SALT).unwrap().unwrap() {
            MuxRecord::Control(RelayControlMsg::Hello { ranks, .. }) => {
                hello_tx.send(ranks).unwrap();
            }
            other => panic!("expected Hello, got {other:?}"),
        }
        MuxRecord::control(RelayControlMsg::HelloAck)
            .write_to(&mut up, &SALT)
            .unwrap();
        // First data frame: echo it straight back to its rank.
        up.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
        if let MuxRecord::Data { rank, payload } =
            MuxRecord::read_from(&mut up, &SALT).unwrap().unwrap()
        {
            data_tx.send((rank, payload.clone())).unwrap();
            MuxRecord::data(rank, payload).write_to(&mut up, &SALT).unwrap();
        }
        // Drain until the relay tears down.
        while let Ok(Some(_)) = MuxRecord::read_from(&mut up, &SALT) {}
    })
}

#[test]
fn relay_channel_data_channel_end_to_end() {
    let (listener, relay_port) = RelayChannel::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let controller = TcpListener::bind("127.0.0.1:0").unwrap();
    let controller_addr: SocketAddr = controller.local_addr().unwrap();

    // Fold-aware fake controller: Hello/ack, read the relay's HostFrame,
    // report it, and answer with a Broadcast consensus.
    let (hello_tx, hello_rx) = mpsc::channel();
    let (host_tx, host_rx) = mpsc::channel::<RoundFrame>();
    let consensus = frame(&[7.0, 8.0], 1.0);
    let consensus_c = consensus.clone();
    let ctrl_thread = thread::spawn(move || {
        let (mut up, _) = controller.accept().unwrap();
        up.set_nodelay(true).unwrap();
        crate::distributed::wire::expect_channel_magic(
            &mut up,
            crate::distributed::wire::CHANNEL_MAGIC_DATA,
            "fake controller",
        )
        .unwrap();
        match MuxRecord::read_from(&mut up, &SALT).unwrap().unwrap() {
            MuxRecord::Control(RelayControlMsg::Hello { ranks, .. }) => {
                hello_tx.send(ranks).unwrap();
            }
            other => panic!("expected Hello, got {other:?}"),
        }
        MuxRecord::control(RelayControlMsg::HelloAck)
            .write_to(&mut up, &SALT)
            .unwrap();
        up.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
        match MuxRecord::read_from(&mut up, &SALT).unwrap().unwrap() {
            MuxRecord::HostFrame { payload } => {
                host_tx.send(parse_frame(&payload)).unwrap();
            }
            other => panic!("expected HostFrame, got {other:?}"),
        }
        MuxRecord::broadcast(frame_blob(&consensus_c))
            .write_to(&mut up, &SALT)
            .unwrap();
        // Drain until the relay tears down.
        while let Ok(Some(_)) = MuxRecord::read_from(&mut up, &SALT) {}
    });

    // One rank dials the relay, does the data handshake, sends its
    // contribution, and reads the consensus back.
    let (echo_tx, echo_rx) = mpsc::channel::<RoundFrame>();
    let contribution = frame(&[1.5, 2.5], 2.0);
    let contribution_c = contribution.clone();
    let rank_thread = thread::spawn(move || {
        let mut s = TcpStream::connect(("127.0.0.1", relay_port)).unwrap();
        s.set_nodelay(true).unwrap();
        data_rank_handshake(&mut s, 0, 1);
        s.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
        write_len_framed(&mut s, &frame_blob(&contribution_c)).unwrap();
        // Report the consensus to main, which asserts it before tearing
        // the relay down (so shutdown can't race ahead of delivery).
        let blob = read_len_framed(&mut s).unwrap().expect("consensus blob");
        echo_tx.send(parse_frame(&blob)).unwrap();
        // Hold the socket open until the relay tears it down.
        while let Ok(Some(_)) = read_len_framed(&mut s) {}
    });

    let channel = RelayChannel::start(
        listener,
        ChannelKind::Data,
        controller_addr,
        "test-host".into(),
        vec![0],
        1,
        SALT,
    )
    .unwrap();

    assert_eq!(hello_rx.recv_timeout(READ_TIMEOUT).unwrap(), vec![0]);
    // 1-rank host: the fold is the identity — values and mass unchanged.
    let host = host_rx.recv_timeout(READ_TIMEOUT).unwrap();
    assert_eq!(frame_vals(&host), frame_vals(&contribution));
    assert!((host.weight - contribution.weight).abs() < 1e-9);
    // The Broadcast consensus reached the rank verbatim.
    let echoed = echo_rx.recv_timeout(READ_TIMEOUT).unwrap();
    assert_eq!(frame_vals(&echoed), frame_vals(&consensus));

    channel.shutdown().unwrap();
    let _ = rank_thread.join();
    let _ = ctrl_thread.join();
}

// --- data-channel fold station ---

/// Establish a 2-rank data-channel mux and return the pieces the fold
/// tests drive: rank clients, controller end, shutdown flag, threads.
fn start_fold_pair() -> (TcpStream, TcpStream, TcpStream, Arc<AtomicBool>, Vec<JoinHandle<()>>) {
    let (c0, r0) = loopback_pair();
    let (c1, r1) = loopback_pair();
    c0.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    c1.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (relay_up, controller) = loopback_pair();
    controller.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
    let (shutdown, threads) = start_mux(ChannelKind::Data, vec![(0, r0), (1, r1)], relay_up);
    (c0, c1, controller, shutdown, threads)
}

#[test]
fn fold_sums_local_ranks_into_one_host_frame() {
    let (mut c0, mut c1, mut controller, shutdown, threads) = start_fold_pair();

    // Both ranks deposit; the relay ships exactly ONE HostFrame with the
    // element-wise sum and the summed mass — never a divide.
    write_len_framed(&mut c0, &frame_blob(&frame(&[1.0, 2.0], 3.0))).unwrap();
    write_len_framed(&mut c1, &frame_blob(&frame(&[10.0, 20.0], 1.0))).unwrap();
    match MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap() {
        MuxRecord::HostFrame { payload } => {
            let folded = parse_frame(&payload);
            assert_eq!(frame_vals(&folded), vec![11.0, 22.0]);
            assert!((folded.weight - 4.0).abs() < 1e-9, "masses sum");
        }
        other => panic!("expected HostFrame, got {other:?}"),
    }

    shutdown.store(true, Ordering::SeqCst);
    drop(c0);
    drop(c1);
    drop(controller);
    join_within(threads);
}

#[test]
fn fold_broadcast_fans_out_to_all_local_ranks() {
    let (mut c0, mut c1, mut controller, shutdown, threads) = start_fold_pair();

    let consensus = frame(&[5.0], 2.0);
    MuxRecord::broadcast(frame_blob(&consensus))
        .write_to(&mut controller, &SALT)
        .unwrap();
    for c in [&mut c0, &mut c1] {
        let blob = read_len_framed(c).unwrap().expect("consensus blob");
        assert_eq!(frame_vals(&parse_frame(&blob)), vec![5.0]);
    }

    shutdown.store(true, Ordering::SeqCst);
    drop(c0);
    drop(c1);
    drop(controller);
    join_within(threads);
}

#[test]
fn fold_rank_exit_folds_the_remainder() {
    let (c0, mut c1, mut controller, shutdown, threads) = start_fold_pair();

    // Rank 0 dies mid-round AFTER rank 1 deposited: the exit removes
    // rank 0 from the barrier and the fold ships with rank 1's
    // contribution only (plus the RankExit riding up).
    write_len_framed(&mut c1, &frame_blob(&frame(&[4.0], 1.5))).unwrap();
    drop(c0);

    let mut saw_exit = false;
    let mut folded: Option<RoundFrame> = None;
    for _ in 0..2 {
        match MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap() {
            MuxRecord::Control(RelayControlMsg::RankExit { rank: 0 }) => saw_exit = true,
            MuxRecord::HostFrame { payload } => folded = Some(parse_frame(&payload)),
            other => panic!("unexpected record {other:?}"),
        }
    }
    assert!(saw_exit, "missing RankExit for dead rank 0");
    let folded = folded.expect("fold must ship once rank 0 leaves the barrier");
    assert_eq!(frame_vals(&folded), vec![4.0]);
    assert!((folded.weight - 1.5).abs() < 1e-9);

    shutdown.store(true, Ordering::SeqCst);
    drop(c1);
    drop(controller);
    join_within(threads);
}

#[test]
fn fold_declare_dead_unblocks_the_barrier() {
    let (c0, mut c1, mut controller, shutdown, threads) = start_fold_pair();

    // Rank 1 deposits; rank 0 is wedged (connected, silent). The
    // controller-forwarded DeclareDead removes it from the barrier and
    // the fold ships without it — the c1-only fix for a death only the
    // controller can see.
    write_len_framed(&mut c1, &frame_blob(&frame(&[9.0], 2.0))).unwrap();
    MuxRecord::control(RelayControlMsg::DeclareDead { rank: 0 })
        .write_to(&mut controller, &SALT)
        .unwrap();

    match MuxRecord::read_from(&mut controller, &SALT).unwrap().unwrap() {
        MuxRecord::HostFrame { payload } => {
            let folded = parse_frame(&payload);
            assert_eq!(frame_vals(&folded), vec![9.0]);
            assert!((folded.weight - 2.0).abs() < 1e-9);
        }
        other => panic!("expected HostFrame, got {other:?}"),
    }

    shutdown.store(true, Ordering::SeqCst);
    drop(c0);
    drop(c1);
    drop(controller);
    join_within(threads);
}

#[test]
fn relay_channel_control_channel_end_to_end() {
    use crate::distributed::cluster_coordinator::write_handshake_rank;

    let (listener, relay_port) = RelayChannel::bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let controller = TcpListener::bind("127.0.0.1:0").unwrap();
    let controller_addr: SocketAddr = controller.local_addr().unwrap();

    let (hello_tx, hello_rx) = mpsc::channel();
    let (data_tx, data_rx) = mpsc::channel();
    let (echo_tx, echo_rx) = mpsc::channel::<Option<Vec<u8>>>();
    let ctrl_thread = spawn_fake_controller(controller, hello_tx, data_tx);

    let rank_thread = thread::spawn(move || {
        let mut s = TcpStream::connect(("127.0.0.1", relay_port)).unwrap();
        s.set_nodelay(true).unwrap();
        // Control-channel handshake is salt-authenticated.
        write_handshake_rank(&mut s, 1, 2, &SALT).unwrap();
        let mut ack = [0u8; 16];
        s.read_exact(&mut ack).unwrap();
        s.set_read_timeout(Some(READ_TIMEOUT)).unwrap();
        write_len_framed(&mut s, &[0x01, 0x02, 0x03]).unwrap();
        echo_tx.send(read_len_framed(&mut s).unwrap()).unwrap();
        while let Ok(Some(_)) = read_len_framed(&mut s) {}
    });

    let channel = RelayChannel::start(
        listener,
        ChannelKind::Control,
        controller_addr,
        "test-host".into(),
        vec![1],
        2,
        SALT,
    )
    .unwrap();

    assert_eq!(hello_rx.recv_timeout(READ_TIMEOUT).unwrap(), vec![1]);
    assert_eq!(
        data_rx.recv_timeout(READ_TIMEOUT).unwrap(),
        (1u32, vec![0x01, 0x02, 0x03])
    );
    assert_eq!(
        echo_rx.recv_timeout(READ_TIMEOUT).unwrap(),
        Some(vec![0x01, 0x02, 0x03])
    );

    channel.shutdown().unwrap();
    let _ = rank_thread.join();
    let _ = ctrl_thread.join();
}
