//! Tests for the dial-in membership gate: the pure ledger (admission,
//! rank assignment, window verdicts) and the join-window I/O loop over
//! real loopback sockets.

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::distributed::port_mux::StreamSource;
use crate::distributed::wire::{
    CHANNEL_MAGIC_JOIN, ControlFrame, JoinMsgWire, MsgKind, SESSION_SALT_BYTES,
    SessionSalt, salt_to_hex, write_channel_magic,
};
use crate::tensor::Result;

fn test_config() -> JoinConfig {
    JoinConfig::default()
}

fn sig(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn admit_host(
    ledger: &mut MembershipLedger,
    host: &str,
    rank_count: usize,
) -> std::result::Result<Vec<usize>, String> {
    ledger.admit(
        host,
        (0..rank_count as u8).collect(),
        vec!["GPU".to_string(); rank_count],
        "builds/test".to_string(),
        sig(7),
        Duration::from_secs(1),
    )
}

// ---------------------------------------------------------------------------
// Ledger: admission + rank assignment
// ---------------------------------------------------------------------------

#[test]
fn admission_assigns_contiguous_ranks_in_order() {
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    assert_eq!(admit_host(&mut ledger, "a", 1).unwrap(), vec![0]);
    assert_eq!(admit_host(&mut ledger, "b", 2).unwrap(), vec![1, 2]);
    assert_eq!(admit_host(&mut ledger, "c", 3).unwrap(), vec![3, 4, 5]);
    assert_eq!(ledger.joined_ranks(), 6);
}

#[test]
fn duplicate_host_rejected() {
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    let why = admit_host(&mut ledger, "a", 1).unwrap_err();
    assert!(why.contains("already joined"), "got: {why}");
    // The failed attempt must not have burned rank ids.
    assert_eq!(admit_host(&mut ledger, "b", 1).unwrap(), vec![1]);
}

#[test]
fn dataset_sig_reference_and_mismatch() {
    // No expected sig: the first joiner sets the reference.
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    ledger
        .admit("a", vec![0], vec![], String::new(), sig(1), Duration::ZERO)
        .unwrap();
    let why = ledger
        .admit("b", vec![0], vec![], String::new(), sig(2), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("dataset signature mismatch"), "got: {why}");

    // Expected sig provided: the very first mismatch is rejected.
    let mut ledger = MembershipLedger::new(test_config(), Some(sig(9))).unwrap();
    let why = ledger
        .admit("a", vec![0], vec![], String::new(), sig(1), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("dataset signature mismatch"), "got: {why}");
    ledger
        .admit("a", vec![0], vec![], String::new(), sig(9), Duration::ZERO)
        .unwrap();
}

#[test]
fn hostile_device_lists_rejected() {
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    let why = admit_host(&mut ledger, "zero", 0).unwrap_err();
    assert!(why.contains("must be non-empty"), "got: {why}");
    let why = ledger
        .admit(
            "huge",
            vec![0u8; 100_000],
            vec![],
            String::new(),
            sig(7),
            Duration::ZERO,
        )
        .unwrap_err();
    assert!(why.contains("exceeds the per-worker cap"), "got: {why}");
    let why = ledger
        .admit("dup", vec![0, 0], vec![], String::new(), sig(7), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("duplicate local device"), "got: {why}");
    let why = admit_host(&mut ledger, "  ", 1).unwrap_err();
    assert!(why.contains("non-empty"), "got: {why}");
    assert_eq!(ledger.joined_ranks(), 0);
}

#[test]
fn retract_last_returns_rank_ids_to_the_pool() {
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    assert_eq!(admit_host(&mut ledger, "b", 2).unwrap(), vec![1, 2]);
    // Only the tail can be retracted.
    assert!(ledger.retract_last("a").is_err());
    ledger.retract_last("b").unwrap();
    assert_eq!(ledger.joined_ranks(), 1);
    // The freed ids are reassigned to the next joiner.
    assert_eq!(admit_host(&mut ledger, "c", 2).unwrap(), vec![1, 2]);
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn config_cross_field_validation() {
    assert!(test_config().validate().is_ok());

    let zero_quorum = JoinConfig { min_rank_start: 0, ..test_config() };
    let msg = MembershipLedger::new(zero_quorum, None).unwrap_err().to_string();
    assert!(msg.contains("min_rank_start"), "got: {msg}");

    let target_below_quorum = JoinConfig {
        min_rank_start: 4,
        target_ranks: Some(2),
        ..test_config()
    };
    let msg = target_below_quorum.validate().unwrap_err().to_string();
    assert!(msg.contains("target_ranks"), "got: {msg}");

    let cap_below_window = JoinConfig {
        join_timeout_secs: 300,
        max_join_timeout_secs: 100,
        ..test_config()
    };
    let msg = cap_below_window.validate().unwrap_err().to_string();
    assert!(msg.contains("max_join_timeout"), "got: {msg}");
}

// ---------------------------------------------------------------------------
// Window verdicts (pure — fabricated elapsed times, no sleeping)
// ---------------------------------------------------------------------------

const WINDOW: Duration = Duration::from_secs(300);
const CAP: Duration = Duration::from_secs(600);

#[test]
fn verdict_target_closes_early() {
    let config = JoinConfig { target_ranks: Some(2), ..test_config() };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    assert_eq!(ledger.verdict(Duration::ZERO, WINDOW, CAP, false), WindowVerdict::Open);
    admit_host(&mut ledger, "b", 1).unwrap();
    assert!(matches!(
        ledger.verdict(Duration::ZERO, WINDOW, CAP, false),
        WindowVerdict::Formed(_)
    ));
}

#[test]
fn verdict_quorum_early_does_not_close_the_window() {
    // Quorum of 1 met immediately, but no target: the window stays open
    // so late workers within it are still admitted.
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    assert_eq!(
        ledger.verdict(Duration::from_secs(10), WINDOW, CAP, false),
        WindowVerdict::Open
    );
    // The moment the window expires, quorum forms the world.
    assert!(matches!(
        ledger.verdict(WINDOW, WINDOW, CAP, false),
        WindowVerdict::Formed(_)
    ));
}

#[test]
fn verdict_grace_range_waits_for_quorum_then_forms() {
    let config = JoinConfig { min_rank_start: 2, ..test_config() };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    // Window expired below quorum, cap not yet: keep waiting.
    let in_grace = WINDOW + Duration::from_secs(30);
    assert_eq!(ledger.verdict(in_grace, WINDOW, CAP, false), WindowVerdict::Open);
    // Quorum arriving in the grace range forms immediately.
    admit_host(&mut ledger, "b", 1).unwrap();
    assert!(matches!(
        ledger.verdict(in_grace, WINDOW, CAP, false),
        WindowVerdict::Formed(_)
    ));
}

#[test]
fn verdict_cap_expiry_fails_loudly() {
    let config = JoinConfig { min_rank_start: 2, ..test_config() };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    match ledger.verdict(CAP, WINDOW, CAP, false) {
        WindowVerdict::Failed(why) => {
            assert!(why.contains("quorum not met"), "got: {why}");
            assert!(why.contains("1/2"), "got: {why}");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn verdict_manual_holds_until_operator_start() {
    use crate::distributed::membership::StartMode;
    let config = JoinConfig {
        min_rank_start: 1,
        start_mode: StartMode::Manual,
        ..test_config()
    };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    // Quorum met, window expired: a manual hold stays open where auto
    // would have formed.
    assert_eq!(ledger.verdict(WINDOW, WINDOW, CAP, false), WindowVerdict::Open);
    assert_eq!(
        ledger.verdict(WINDOW + Duration::from_secs(60), WINDOW, CAP, false),
        WindowVerdict::Open
    );
    // The operator fires: formed, at any point in the open range.
    assert!(matches!(
        ledger.verdict(Duration::from_secs(5), WINDOW, CAP, true),
        WindowVerdict::Formed("operator start")
    ));
    // Never fired: the cap bounds the exposure with a distinct reason.
    match ledger.verdict(CAP, WINDOW, CAP, false) {
        WindowVerdict::Failed(why) => {
            assert!(why.contains("operator start not received"), "got: {why}");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
    // The staging phase names the hold; quorum-met + manual = staging.
    assert_eq!(ledger.open_phase(), ClusterPhase::Staging);
}

#[test]
fn verdict_manual_start_is_quorum_gated() {
    use crate::distributed::membership::StartMode;
    let config = JoinConfig {
        min_rank_start: 2,
        start_mode: StartMode::Manual,
        ..test_config()
    };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    // Armed below quorum: inert (the HTTP layer refuses this too, but
    // the verdict is the authority).
    assert_eq!(ledger.verdict(Duration::ZERO, WINDOW, CAP, true), WindowVerdict::Open);
    assert_eq!(ledger.open_phase(), ClusterPhase::Waiting);
    admit_host(&mut ledger, "b", 1).unwrap();
    assert!(matches!(
        ledger.verdict(Duration::ZERO, WINDOW, CAP, true),
        WindowVerdict::Formed("operator start")
    ));
}

#[test]
fn verdict_hybrid_keeps_the_clock_and_adds_the_operator() {
    use crate::distributed::membership::StartMode;
    let config = JoinConfig {
        min_rank_start: 1,
        target_ranks: Some(3),
        start_mode: StartMode::Hybrid,
        ..test_config()
    };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    // Clock semantics intact: below target, inside the window → open.
    assert_eq!(ledger.verdict(Duration::ZERO, WINDOW, CAP, false), WindowVerdict::Open);
    // Startable while waiting = staging.
    assert_eq!(ledger.open_phase(), ClusterPhase::Staging);
    // Operator fires early with quorum.
    assert!(matches!(
        ledger.verdict(Duration::ZERO, WINDOW, CAP, true),
        WindowVerdict::Formed("operator start")
    ));
    // Or the clock still closes on its own (window expiry with quorum).
    assert!(matches!(
        ledger.verdict(WINDOW, WINDOW, CAP, false),
        WindowVerdict::Formed(_)
    ));
    // Target auto-close survives in hybrid.
    admit_host(&mut ledger, "b", 1).unwrap();
    admit_host(&mut ledger, "c", 1).unwrap();
    assert!(matches!(
        ledger.verdict(Duration::ZERO, WINDOW, CAP, false),
        WindowVerdict::Formed("target ranks reached")
    ));
}

#[test]
fn verdict_auto_ignores_the_start_switch() {
    // The HTTP layer refuses /start in auto mode; if a flag ever leaks
    // through anyway, the verdict must not honor it.
    let mut ledger = MembershipLedger::new(test_config(), None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    assert_eq!(
        ledger.verdict(Duration::from_secs(1), WINDOW, CAP, true),
        WindowVerdict::Open
    );
    assert_eq!(ledger.open_phase(), ClusterPhase::Waiting);
}

#[test]
fn manual_mode_refuses_target_ranks() {
    use crate::distributed::membership::StartMode;
    let config = JoinConfig {
        start_mode: StartMode::Manual,
        target_ranks: Some(2),
        min_rank_start: 1,
        ..test_config()
    };
    let msg = config.validate().unwrap_err().to_string();
    assert!(msg.contains("manual"), "got: {msg}");
    assert!(msg.contains("target_ranks"), "got: {msg}");
}

// ---------------------------------------------------------------------------
// Admission-mode resolution + snapshot
// ---------------------------------------------------------------------------

#[test]
fn open_admission_resolution_follows_bind_scope_then_knob() {
    let closed = test_config();
    let open = JoinConfig { open_admission: true, ..test_config() };
    // Loopback bind: open admission is sound regardless of the knob.
    assert!(resolve_open_admission(&closed, true));
    assert!(resolve_open_admission(&open, true));
    // Non-loopback: pre-shared salt unless explicitly opened.
    assert!(!resolve_open_admission(&closed, false));
    assert!(resolve_open_admission(&open, false));
}

#[test]
fn snapshot_serializes_phase_and_countdowns() {
    let config = JoinConfig { target_ranks: Some(3), ..test_config() };
    let mut ledger = MembershipLedger::new(config, None).unwrap();
    admit_host(&mut ledger, "a", 2).unwrap();

    let snap = ledger.snapshot(ClusterPhase::Waiting, Duration::from_secs(10), false);
    assert_eq!(snap.joined_ranks, 2);
    assert_eq!(snap.joined_hosts, 1);
    assert_eq!(snap.target_ranks, Some(3));
    assert!(snap.window_remaining_secs.is_some());
    assert!(snap.cap_remaining_secs.is_some());
    let js = serde_json::to_string(&snap).unwrap();
    assert!(js.contains("\"phase\":\"waiting\""), "got: {js}");
    assert!(js.contains("\"host\":\"a\""), "got: {js}");

    // Past both deadlines the countdowns render as exhausted, and every
    // lifecycle phase has a stable snake_case rendering.
    let late = ledger.snapshot(ClusterPhase::Failed, Duration::from_secs(100_000), false);
    assert_eq!(late.window_remaining_secs, None);
    assert_eq!(late.cap_remaining_secs, None);
    for (phase, expect) in [
        (ClusterPhase::Waiting, "waiting"),
        (ClusterPhase::Staging, "staging"),
        (ClusterPhase::Forming, "forming"),
        (ClusterPhase::Training, "training"),
        (ClusterPhase::Done, "done"),
        (ClusterPhase::Failed, "failed"),
    ] {
        assert_eq!(serde_json::to_string(&phase).unwrap(), format!("\"{expect}\""));
    }
}

// ---------------------------------------------------------------------------
// Join-message wire round-trips
// ---------------------------------------------------------------------------

#[test]
fn join_messages_round_trip_through_control_frames() {
    let salt: SessionSalt = [3u8; SESSION_SALT_BYTES];
    let msgs = [
        JoinMsgWire::Hello {
            host: "pascal".to_string(),
            local_devices: vec![0, 1],
            gpus: vec!["GP106".to_string(), "GP106".to_string()],
            libtorch: "builds/sm61-sm120".to_string(),
            dataset_sig: sig(5),
        },
        JoinMsgWire::Accept {
            ranks: vec![1, 2],
            salt_hex: Some(salt_to_hex(&salt)),
            formation_wait_secs: 480,
        },
        JoinMsgWire::Reject { reason: "duplicate".to_string() },
        JoinMsgWire::WorldFormed {
            envelope_hex: "abcd".to_string(),
            relay_spec_hex: Some("ef01".to_string()),
        },
        JoinMsgWire::RankExited { rank: 1, code: -9 },
        JoinMsgWire::Abort { reason: "quorum".to_string() },
    ];
    for msg in msgs {
        let mut buf = Vec::new();
        ControlFrame::encode(&salt, MsgKind::Join, &msg)
            .unwrap()
            .write_to(&mut buf)
            .unwrap();
        let frame = ControlFrame::read_from(&mut buf.as_slice(), &salt)
            .unwrap()
            .expect("frame present");
        assert_eq!(frame.kind, MsgKind::Join);
        let decoded: JoinMsgWire = frame.decode().unwrap();
        assert_eq!(decoded, msg);
    }
}

// ---------------------------------------------------------------------------
// Join window I/O over real loopback sockets
// ---------------------------------------------------------------------------

fn spawn_window(
    config: JoinConfig,
    salt: SessionSalt,
    pre_shared_salt: bool,
    abort: Arc<AtomicBool>,
) -> (u16, std::thread::JoinHandle<Result<FormedWorld>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = std::thread::spawn(move || {
        let source = StreamSource::from_listener(listener, "membership-test")?;
        let status = crate::distributed::status::StatusBoard::new();
        run_join_window(
            &source, &config, &salt, pre_shared_salt, None, &abort, &status,
        )
    });
    (port, handle)
}

/// Dial the window like an agent: channel magic, then a keyed hello.
/// Returns the connection and the reply message.
fn dial_and_join(
    port: u16,
    key: &SessionSalt,
    host: &str,
    rank_count: u32,
) -> (TcpStream, JoinMsgWire) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write_channel_magic(&mut stream, CHANNEL_MAGIC_JOIN).unwrap();
    let hello = JoinMsgWire::Hello {
        host: host.to_string(),
        local_devices: (0..rank_count as u8).collect(),
        gpus: vec!["TestGPU".to_string(); rank_count as usize],
        libtorch: "builds/test".to_string(),
        dataset_sig: sig(7),
    };
    ControlFrame::encode(key, MsgKind::Join, &hello)
        .unwrap()
        .write_to(&mut stream)
        .unwrap();
    let reply = ControlFrame::read_from(&mut stream, key)
        .unwrap()
        .expect("reply frame");
    let msg: JoinMsgWire = reply.decode().unwrap();
    (stream, msg)
}

#[test]
fn window_open_mode_admits_hands_out_salt_and_closes_on_target() {
    let salt: SessionSalt = [9u8; SESSION_SALT_BYTES];
    let zero_key: SessionSalt = [0u8; SESSION_SALT_BYTES];
    let config = JoinConfig { target_ranks: Some(3), ..test_config() };
    let abort = Arc::new(AtomicBool::new(false));
    let (port, handle) = spawn_window(config, salt, false, Arc::clone(&abort));

    // Sequential dials so admission order (and thus rank assignment) is
    // deterministic.
    let (_conn_a, reply_a) = dial_and_join(port, &zero_key, "host-a", 1);
    match reply_a {
        JoinMsgWire::Accept { ranks, salt_hex, formation_wait_secs } => {
            assert_eq!(ranks, vec![0]);
            // Open admission hands the session salt out in the reply.
            assert_eq!(salt_hex.as_deref(), Some(salt_to_hex(&salt).as_str()));
            // The remaining hard-cap budget rides along so the worker's
            // WorldFormed read deadline self-describes.
            assert!(formation_wait_secs > 0);
        }
        other => panic!("expected Accept, got {other:?}"),
    }
    let (_conn_b, reply_b) = dial_and_join(port, &zero_key, "host-b", 2);
    match reply_b {
        JoinMsgWire::Accept { ranks, .. } => assert_eq!(ranks, vec![1, 2]),
        other => panic!("expected Accept, got {other:?}"),
    }

    // Third rank hit the target: the window closes immediately.
    let world = handle.join().unwrap().unwrap();
    assert_eq!(world.world_size, 3);
    let hosts: Vec<&str> =
        world.workers.iter().map(|w| w.member.host.as_str()).collect();
    assert_eq!(hosts, vec!["host-a", "host-b"]);
    assert_eq!(world.workers[1].member.ranks, vec![1, 2]);
}

#[test]
fn window_pre_shared_mode_drops_wrong_key_and_omits_salt() {
    let salt: SessionSalt = [9u8; SESSION_SALT_BYTES];
    let zero_key: SessionSalt = [0u8; SESSION_SALT_BYTES];
    let config = JoinConfig { target_ranks: Some(1), ..test_config() };
    let abort = Arc::new(AtomicBool::new(false));
    let (port, handle) = spawn_window(config, salt, true, Arc::clone(&abort));

    // A zero-keyed hello fails frame authentication: the connection is
    // dropped without an accept.
    let mut intruder = TcpStream::connect(("127.0.0.1", port)).unwrap();
    intruder
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write_channel_magic(&mut intruder, CHANNEL_MAGIC_JOIN).unwrap();
    let hello = JoinMsgWire::Hello {
        host: "intruder".to_string(),
        local_devices: vec![0],
        gpus: vec![],
        libtorch: String::new(),
        dataset_sig: sig(7),
    };
    ControlFrame::encode(&zero_key, MsgKind::Join, &hello)
        .unwrap()
        .write_to(&mut intruder)
        .unwrap();
    use std::io::Read;
    let mut byte = [0u8; 1];
    assert_eq!(
        intruder.read(&mut byte).unwrap_or(0),
        0,
        "wrong-key hello must be dropped without a reply"
    );

    // The salt-keyed hello is admitted, and the reply does NOT re-send
    // the pre-shared secret.
    let (_conn, reply) = dial_and_join(port, &salt, "honest", 1);
    match reply {
        JoinMsgWire::Accept { ranks, salt_hex, .. } => {
            assert_eq!(ranks, vec![0]);
            assert_eq!(salt_hex, None);
        }
        other => panic!("expected Accept, got {other:?}"),
    }
    let world = handle.join().unwrap().unwrap();
    assert_eq!(world.world_size, 1);
    assert_eq!(world.workers[0].member.host, "honest");
}

#[test]
fn window_rejects_non_hello_then_still_forms() {
    let salt: SessionSalt = [9u8; SESSION_SALT_BYTES];
    let zero_key: SessionSalt = [0u8; SESSION_SALT_BYTES];
    let config = JoinConfig { target_ranks: Some(1), ..test_config() };
    let abort = Arc::new(AtomicBool::new(false));
    let (port, handle) = spawn_window(config, salt, false, Arc::clone(&abort));

    // Protocol-conformant but wrong first message: rejected with a
    // reason, and the window keeps accepting.
    let mut confused = TcpStream::connect(("127.0.0.1", port)).unwrap();
    confused
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write_channel_magic(&mut confused, CHANNEL_MAGIC_JOIN).unwrap();
    ControlFrame::encode(
        &zero_key,
        MsgKind::Join,
        &JoinMsgWire::RankExited { rank: 0, code: 1 },
    )
    .unwrap()
    .write_to(&mut confused)
    .unwrap();
    let reply = ControlFrame::read_from(&mut confused, &zero_key)
        .unwrap()
        .expect("reject frame");
    match reply.decode::<JoinMsgWire>().unwrap() {
        JoinMsgWire::Reject { reason } => {
            assert!(reason.contains("must be Hello"), "got: {reason}");
        }
        other => panic!("expected Reject, got {other:?}"),
    }

    let (_conn, reply) = dial_and_join(port, &zero_key, "good", 1);
    assert!(matches!(reply, JoinMsgWire::Accept { .. }));
    let world = handle.join().unwrap().unwrap();
    assert_eq!(world.world_size, 1);
}

#[test]
fn window_aborts_promptly_on_launcher_flag() {
    let salt: SessionSalt = [9u8; SESSION_SALT_BYTES];
    let abort = Arc::new(AtomicBool::new(true));
    let (_port, handle) = spawn_window(test_config(), salt, false, abort);
    let err = handle.join().unwrap().unwrap_err().to_string();
    assert!(err.contains("aborted"), "got: {err}");
}
