//! Port-mux dispatcher tests: routing by channel magic, hostile-peer
//! isolation, and closed-channel fail-fast. All over loopback TCP.

use super::*;
use crate::distributed::wire::write_channel_magic;
use std::io::{Read, Write};
use std::net::TcpListener;

fn start_test_mux() -> (PortMux, MuxAccept, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let abort = Arc::new(AtomicBool::new(false));
    let (mux, accept) = PortMux::start(listener, abort).unwrap();
    let port = mux.port();
    (mux, accept, port)
}

fn dial(port: u16, magic: u32, payload: &[u8]) -> TcpStream {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_nodelay(true).unwrap();
    write_channel_magic(&mut s, magic).unwrap();
    s.write_all(payload).unwrap();
    s
}

/// Read one routed stream off `rx` and assert its unread bytes are the
/// full magic + payload (the dispatcher peeks, never consumes).
fn recv_and_check(rx: &Receiver<TcpStream>, magic: u32, payload: &[u8]) {
    let mut stream = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("stream routed to channel");
    let mut buf = vec![0u8; 4 + payload.len()];
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[0..4], magic.to_le_bytes());
    assert_eq!(&buf[4..], payload);
}

#[test]
fn routes_each_channel_by_magic_over_one_port() {
    let (_mux, accept, port) = start_test_mux();

    let s1 = dial(port, CHANNEL_MAGIC_RENDEZVOUS, b"rdv");
    let s2 = dial(port, CHANNEL_MAGIC_DATA, b"data");
    let s3 = dial(port, CHANNEL_MAGIC_CONTROL, b"ctrl");
    let s4 = dial(port, CHANNEL_MAGIC_JOIN, b"join");

    recv_and_check(&accept.rendezvous, CHANNEL_MAGIC_RENDEZVOUS, b"rdv");
    recv_and_check(&accept.data, CHANNEL_MAGIC_DATA, b"data");
    recv_and_check(&accept.control, CHANNEL_MAGIC_CONTROL, b"ctrl");
    recv_and_check(&accept.join, CHANNEL_MAGIC_JOIN, b"join");

    drop((s1, s2, s3, s4));
}

#[test]
fn http_get_routes_to_status_leg_with_request_intact() {
    let (_mux, accept, port) = start_test_mux();

    // A plain HTTP client writes no flodl magic — the request line's
    // leading "GET " IS the routing key, and unlike the flodl channels
    // it must reach the consumer unconsumed (it is part of the request).
    let request = b"GET /state.json HTTP/1.1\r\nHost: t\r\n\r\n";
    let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
    client.set_nodelay(true).unwrap();
    client.write_all(request).unwrap();

    let mut routed = accept
        .status
        .recv_timeout(Duration::from_secs(5))
        .expect("HTTP GET routed to status leg");
    routed
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut buf = vec![0u8; request.len()];
    routed.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, request);
    drop(client);
}

#[test]
fn unknown_magic_dropped_and_dispatcher_continues() {
    let (_mux, accept, port) = start_test_mux();

    // Hostile/garbage dial: unknown magic. The dispatcher must drop it
    // (we observe EOF/reset) and keep serving honest peers.
    let mut rogue = dial(port, 0xDEAD_BEEF, b"junk");
    rogue
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut buf = [0u8; 1];
    match rogue.read(&mut buf) {
        Ok(0) | Err(_) => {}
        Ok(n) => panic!("rogue connection should be dropped, read {n} bytes"),
    }

    let honest = dial(port, CHANNEL_MAGIC_DATA, b"ok");
    recv_and_check(&accept.data, CHANNEL_MAGIC_DATA, b"ok");
    drop(honest);
}

#[test]
fn eof_before_magic_dropped_and_dispatcher_continues() {
    let (_mux, accept, port) = start_test_mux();

    // Connect and hang up without writing anything.
    let early_eof = TcpStream::connect(("127.0.0.1", port)).unwrap();
    drop(early_eof);

    let honest = dial(port, CHANNEL_MAGIC_CONTROL, b"ok");
    recv_and_check(&accept.control, CHANNEL_MAGIC_CONTROL, b"ok");
    drop(honest);
}

#[test]
fn closed_channel_resets_dialer_and_dispatcher_continues() {
    let (_mux, accept, port) = start_test_mux();
    // This run has no rendezvous subsystem.
    drop(accept.rendezvous);

    let mut stray = dial(port, CHANNEL_MAGIC_RENDEZVOUS, b"?");
    stray
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut buf = [0u8; 1];
    match stray.read(&mut buf) {
        Ok(0) | Err(_) => {}
        Ok(n) => panic!("stray dialer should be reset, read {n} bytes"),
    }

    // Other channels keep working.
    let honest = dial(port, CHANNEL_MAGIC_DATA, b"ok");
    recv_and_check(&accept.data, CHANNEL_MAGIC_DATA, b"ok");
    drop(honest);
}

#[test]
fn abort_flag_stops_dispatcher() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let abort = Arc::new(AtomicBool::new(false));
    let (mux, accept) = PortMux::start(listener, Arc::clone(&abort)).unwrap();
    abort.store(true, Ordering::SeqCst);
    // The dispatcher exits within one accept poll; the receivers then
    // report disconnection instead of blocking forever.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match accept.data.recv_timeout(Duration::from_millis(50)) {
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                assert!(
                    Instant::now() < deadline,
                    "dispatcher did not exit on abort"
                );
            }
            Ok(_) => panic!("no connection was made"),
        }
    }
    drop(mux); // Drop joins the (already exited) dispatcher.
}

#[test]
fn stream_source_listener_polls_and_accepts() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let source = StreamSource::from_listener(listener, "test").unwrap();

    // Nothing pending yet.
    assert!(source.try_accept("test").unwrap().is_none());

    let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
    client.write_all(b"x").unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut accepted = loop {
        if let Some(s) = source.try_accept("test").unwrap() {
            break s;
        }
        assert!(Instant::now() < deadline, "accept never surfaced");
        thread::sleep(Duration::from_millis(10));
    };
    // Accepted stream is blocking-mode: a timed read works.
    accepted
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut buf = [0u8; 1];
    accepted.read_exact(&mut buf).unwrap();
    assert_eq!(&buf, b"x");
}
