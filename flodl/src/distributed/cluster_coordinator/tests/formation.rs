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
