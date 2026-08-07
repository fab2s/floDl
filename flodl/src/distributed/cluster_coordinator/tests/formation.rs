//! Formation dial-in deadline tests: the accept loop must fail loudly
//! (never spin) when relays don't cover the world in time.

use super::*;

/// No relay ever dials: the accept loop must give up at the configured
/// formation deadline with an error naming the coverage reached — not
/// spin until some other part of the cohort dies.
#[test]
fn formation_deadline_errs_when_relays_never_dial() {
    let (listener, _port) = ClusterCoordinator::bind(SocketAddr::new(
        Ipv4Addr::LOCALHOST.into(),
        0,
    ))
    .expect("bind succeeds");
    let started = std::time::Instant::now();
    let Err(err) = ClusterCoordinator::start_from_listener(
        listener,
        TEST_SALT,
        cfg_sync_nccl(3).formation_timeout_secs(1),
    ) else {
        panic!("no relay dialed — formation must time out");
    };
    assert!(
        err.to_string().contains("0/3 ranks covered"),
        "error should name the coverage reached, got: {err}"
    );
    // Bounded promptly: the 1s deadline plus poll slack, nowhere near
    // the historic spin-forever behavior.
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "deadline should fire near 1s, took {:?}",
        started.elapsed()
    );
}

/// Two single-rank relays announce different model signatures: the
/// accept loop must refuse formation naming both (host, rank) pairs,
/// instead of acking a cohort whose first collective would hang on
/// mismatched shapes. Sequenced so rank 0 seeds the expectation before
/// rank 1 dials, keeping the error's seed/offender roles deterministic.
#[test]
fn formation_refuses_mixed_model_signatures_by_name() {
    let (listener, port) = ClusterCoordinator::bind(SocketAddr::new(
        Ipv4Addr::LOCALHOST.into(),
        0,
    ))
    .expect("bind succeeds");
    let (seeded_tx, seeded_rx) = std::sync::mpsc::channel::<()>();
    let seeder = std::thread::spawn(move || {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let mut stream = TcpStream::connect(addr).expect("rank 0 connect");
        relay_hello_sigs(&mut stream, &TEST_SALT, 0, vec![[0xAA; 32]])
            .expect("rank 0 hello is acked (it seeds the expectation)");
        seeded_tx.send(()).expect("signal seeded");
        // Hold the stream open past the coordinator's verdict.
        std::thread::sleep(Duration::from_secs(2));
    });
    let offender = std::thread::spawn(move || {
        seeded_rx.recv().expect("wait for the seed hello");
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
        let mut stream = TcpStream::connect(addr).expect("rank 1 connect");
        // The coordinator errors before acking, so this hello fails.
        let _ = relay_hello_sigs(&mut stream, &TEST_SALT, 1, vec![[0xBB; 32]]);
    });
    let Err(err) = ClusterCoordinator::start_from_listener(
        listener,
        TEST_SALT,
        cfg_sync_nccl(2).formation_timeout_secs(10),
    ) else {
        panic!("mixed model signatures must refuse formation");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("model mismatch at formation"),
        "error should say what broke, got: {msg}"
    );
    assert!(
        msg.contains("test-r0") && msg.contains("test-r1"),
        "error should name both hosts, got: {msg}"
    );
    let _ = seeder.join();
    let _ = offender.join();
}

/// Matching signatures across relays form normally: the check refuses
/// mixed cohorts, never uniform ones. (Empty signature lists gating
/// nothing is exercised by every other test in this module — the
/// helpers announce none.)
#[test]
fn formation_accepts_matching_model_signatures() {
    let (listener, port) = ClusterCoordinator::bind(SocketAddr::new(
        Ipv4Addr::LOCALHOST.into(),
        0,
    ))
    .expect("bind succeeds");
    let mut relays = Vec::new();
    for rank in 0..2u32 {
        relays.push(std::thread::spawn(move || {
            let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port);
            let mut stream = TcpStream::connect(addr).expect("connect");
            relay_hello_sigs(&mut stream, &TEST_SALT, rank, vec![[0xAA; 32]])
                .expect("matching sig is acked");
            std::thread::sleep(Duration::from_millis(500));
        }));
    }
    let coord = ClusterCoordinator::start_from_listener(
        listener,
        TEST_SALT,
        cfg_sync_nccl(2).formation_timeout_secs(10),
    )
    .expect("matching signatures must form");
    drop(coord);
    for r in relays {
        let _ = r.join();
    }
}

/// Partial coverage: one of two relays dials and stays connected, the
/// other never comes. The deadline must still fire, and the error must
/// name the partial coverage.
#[test]
fn formation_deadline_errs_on_partial_coverage() {
    let (listener, port) = ClusterCoordinator::bind(SocketAddr::new(
        Ipv4Addr::LOCALHOST.into(),
        0,
    ))
    .expect("bind succeeds");
    // A single-rank relay for rank 0 that dials, handshakes, then just
    // holds its stream open (parks well past the coord's verdict).
    let holder = fake_rank(port, 0, 3, TEST_SALT, |_stream, _salt| {
        std::thread::sleep(Duration::from_secs(4));
        Ok(())
    });
    let Err(err) = ClusterCoordinator::start_from_listener(
        listener,
        TEST_SALT,
        cfg_sync_nccl(3).formation_timeout_secs(2),
    ) else {
        panic!("second relay never dialed — formation must time out");
    };
    assert!(
        err.to_string().contains("1/3 ranks covered"),
        "error should name the partial coverage, got: {err}"
    );
    let _ = holder.join();
}
