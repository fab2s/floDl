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

/// A bare offer: every coherence fact at its "gates nothing" value, so
/// tests exercising OTHER checks stay independent of the fact set.
fn offer(host: &str, devices: Vec<u8>, libtorch: &str, dsig: [u8; 32]) -> JoinOffer {
    JoinOffer {
        host: host.to_string(),
        local_devices: devices,
        gpus: vec![],
        libtorch: libtorch.to_string(),
        dataset_sig: dsig,
        run_id: None,
        nccl_version: None,
        model_sig: None,
    }
}

fn admit_host(
    ledger: &mut MembershipLedger,
    host: &str,
    rank_count: usize,
) -> std::result::Result<Vec<usize>, String> {
    let mut o = offer(host, (0..rank_count as u8).collect(), "builds/test", sig(7));
    o.gpus = vec!["GPU".to_string(); rank_count];
    ledger.admit(o, Duration::from_secs(1))
}

// ---------------------------------------------------------------------------
// Ledger: admission + rank assignment
// ---------------------------------------------------------------------------

#[test]
fn admission_assigns_contiguous_ranks_in_order() {
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    assert_eq!(admit_host(&mut ledger, "a", 1).unwrap(), vec![0]);
    assert_eq!(admit_host(&mut ledger, "b", 2).unwrap(), vec![1, 2]);
    assert_eq!(admit_host(&mut ledger, "c", 3).unwrap(), vec![3, 4, 5]);
    assert_eq!(ledger.joined_ranks(), 6);
}

#[test]
fn duplicate_host_rejected() {
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    admit_host(&mut ledger, "a", 1).unwrap();
    let why = admit_host(&mut ledger, "a", 1).unwrap_err();
    assert!(why.contains("already joined"), "got: {why}");
    // The failed attempt must not have burned rank ids.
    assert_eq!(admit_host(&mut ledger, "b", 1).unwrap(), vec![1]);
}

#[test]
fn dataset_sig_reference_and_mismatch() {
    // No expected sig: the first joiner sets the reference.
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    ledger
        .admit(offer("a", vec![0], "", sig(1)), Duration::ZERO)
        .unwrap();
    let why = ledger
        .admit(offer("b", vec![0], "", sig(2)), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("dataset signature mismatch"), "got: {why}");

    // Expected sig provided: the very first mismatch is rejected.
    let mut ledger = MembershipLedger::new(test_config(), Some(sig(9)), None).unwrap();
    let why = ledger
        .admit(offer("a", vec![0], "", sig(1)), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("dataset signature mismatch"), "got: {why}");
    ledger
        .admit(offer("a", vec![0], "", sig(9)), Duration::ZERO)
        .unwrap();
}

#[test]
fn a_vendor_mixed_cohort_is_refused_under_an_nccl_data_plane() {
    // NCCL and RCCL export the same symbols and unique-id format, so
    // nothing structural rejects a mixed cohort — it hangs at formation
    // AFTER the window deadline was spent. The window is where the one
    // piece of information needed to refuse it early already is.
    let admit = |ledger: &mut MembershipLedger, host: &str, label: &str| {
        ledger.admit(offer(host, vec![0], label, sig(7)), Duration::ZERO)
    };
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    admit(&mut ledger, "green", "precompiled/cu128").unwrap();
    let why = admit(&mut ledger, "red", "precompiled/rocm70").unwrap_err();
    assert!(why.contains("vendor mismatch"), "got: {why}");
    assert!(why.contains("cpu_sync"), "the fix must be named: {why}");
    // The refusal names the label that seeded the cohort's vendor.
    assert!(why.contains("cu128"), "got: {why}");
    // A refused joiner burns nothing: the same box re-dialing with the
    // right build (or the right fleet) is welcome.
    assert_eq!(ledger.joined_ranks(), 1);

    // Labels that classify to no vendor gate nothing, in either order:
    // fan-out agents may send an empty label, and an out-of-convention
    // name is not an admission crime.
    admit(&mut ledger, "bare", "").unwrap();
    admit(&mut ledger, "odd", "builds/mystery").unwrap();
    admit(&mut ledger, "green2", "builds/sm61-sm120").unwrap();

    // A CPU data plane genuinely works cross-vendor, so the gate is off.
    let cpu_plane = JoinConfig { nccl_backend: false, ..test_config() };
    let mut ledger = MembershipLedger::new(cpu_plane, None, None).unwrap();
    admit(&mut ledger, "green", "precompiled/cu128").unwrap();
    admit(&mut ledger, "red", "precompiled/rocm70").unwrap();
    assert_eq!(ledger.joined_ranks(), 2);
}

#[test]
fn a_cohort_straddling_a_publish_boundary_is_refused() {
    // Two boxes that fetched across a publish boundary hold two
    // different runs — different args at minimum, and rank children
    // re-enter the binary with them. Same first-member seeding as the
    // dataset signature; `--bin` boxes carry no id and gate nothing.
    let with_run = |host: &str, run: Option<&str>| {
        let mut o = offer(host, vec![0], "", sig(7));
        o.run_id = run.map(str::to_string);
        o
    };
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    ledger.admit(with_run("a", Some("run-aaaa1111")), Duration::ZERO).unwrap();
    ledger.admit(with_run("bare", None), Duration::ZERO).unwrap();
    let why = ledger
        .admit(with_run("b", Some("run-bbbb2222")), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("run identity mismatch"), "got: {why}");
    assert!(why.contains("run-aaaa"), "the seeded id must be named: {why}");
    assert!(why.contains("next dial"), "the fix must be named: {why}");
    assert_eq!(ledger.joined_ranks(), 2);
    // The matching id keeps joining — the refusal condemned one
    // attempt, not the window.
    ledger.admit(with_run("c", Some("run-aaaa1111")), Duration::ZERO).unwrap();
}

#[test]
fn a_box_building_a_different_model_is_refused_at_admission() {
    // Mismatched parameter manifests corrupt CPU averaging exactly as
    // they hang NCCL, so the check is not plane-gated. Refusing here
    // costs the box only its own dial; the formation-time handshake
    // check stays the backstop for boxes that carry no signature.
    let with_model = |host: &str, m: Option<[u8; 32]>| {
        let mut o = offer(host, vec![0], "", sig(7));
        o.model_sig = m;
        o
    };
    // Controller-seeded: the launcher's own CPU-built model is the
    // run's truth, so the very first mismatching walk-in is refused —
    // no first-member luck.
    let mut ledger =
        MembershipLedger::new(test_config(), None, Some(sig(0xAA))).unwrap();
    let why = ledger
        .admit(with_model("odd", Some(sig(0xBB))), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("model mismatch"), "got: {why}");
    assert!(why.contains("bin:"), "the usual cause must be named: {why}");
    assert_eq!(ledger.joined_ranks(), 0);
    ledger.admit(with_model("good", Some(sig(0xAA))), Duration::ZERO).unwrap();
    // No signature gates nothing (a probe-less box is not an admission
    // crime — formation still checks the model it actually builds).
    ledger.admit(with_model("bare", None), Duration::ZERO).unwrap();

    // First-member seeding when the controller passed no seed.
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    ledger.admit(with_model("a", Some(sig(1))), Duration::ZERO).unwrap();
    ledger.admit(with_model("bare", None), Duration::ZERO).unwrap();
    let why = ledger
        .admit(with_model("b", Some(sig(2))), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("model mismatch"), "got: {why}");
    assert_eq!(ledger.joined_ranks(), 2);
}

#[test]
fn nccl_version_skew_is_refused_where_that_plane_forms() {
    let with_nccl = |host: &str, v: Option<(u32, u32, u32)>| {
        let mut o = offer(host, vec![0], "", sig(7));
        o.nccl_version = v;
        o
    };
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    ledger.admit(with_nccl("a", Some((2, 27, 5))), Duration::ZERO).unwrap();
    // Patch skew is interoperable, and an unknown version (a CPU build,
    // a failed read) gates nothing.
    ledger.admit(with_nccl("b", Some((2, 27, 3))), Duration::ZERO).unwrap();
    ledger.admit(with_nccl("cpu-build", None), Duration::ZERO).unwrap();
    let why = ledger
        .admit(with_nccl("c", Some((2, 26, 2))), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("NCCL version skew"), "got: {why}");
    assert!(why.contains("2.27"), "got: {why}");
    assert!(why.contains("fdl nccl build"), "the bridge must be named: {why}");
    // A CPU data plane has no NCCL handshake to protect: gate off.
    let cpu = JoinConfig { nccl_backend: false, ..test_config() };
    let mut ledger = MembershipLedger::new(cpu, None, None).unwrap();
    ledger.admit(with_nccl("a", Some((2, 27, 5))), Duration::ZERO).unwrap();
    ledger.admit(with_nccl("b", Some((2, 26, 2))), Duration::ZERO).unwrap();
    assert_eq!(ledger.joined_ranks(), 2);
}

#[test]
fn hostile_device_lists_rejected() {
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
    let why = admit_host(&mut ledger, "zero", 0).unwrap_err();
    assert!(why.contains("must be non-empty"), "got: {why}");
    let why = ledger
        .admit(offer("huge", vec![0u8; 100_000], "", sig(7)), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("exceeds the per-worker cap"), "got: {why}");
    let why = ledger
        .admit(offer("dup", vec![0, 0], "", sig(7)), Duration::ZERO)
        .unwrap_err();
    assert!(why.contains("duplicate local device"), "got: {why}");
    let why = admit_host(&mut ledger, "  ", 1).unwrap_err();
    assert!(why.contains("non-empty"), "got: {why}");
    assert_eq!(ledger.joined_ranks(), 0);
}

#[test]
fn retract_last_returns_rank_ids_to_the_pool() {
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
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
    let msg = MembershipLedger::new(zero_quorum, None, None).unwrap_err().to_string();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(test_config(), None, None).unwrap();
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
    let mut ledger = MembershipLedger::new(config, None, None).unwrap();
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
            run_id: Some("a1b2c3d4e5f6".to_string()),
            nccl_version: Some((2, 27, 5)),
            model_sig: Some(sig(6)),
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

/// Rig repro (C2-PR4): the full staging flow over a REAL port mux —
/// join window + status responder on their mux legs, one walk-in,
/// `POST /start` — then prove the DISPATCHER is still alive. The
/// controller/coordinator legs only come up AFTER formation, so a
/// dispatcher that dies anywhere in the staging flow surfaces as
/// "port mux dispatcher exited" at coordinator start (observed live).
#[test]
fn operator_start_leaves_the_mux_dispatcher_alive() {
    use crate::distributed::port_mux::PortMux;
    use crate::distributed::wire::CHANNEL_MAGIC_CONTROL;
    use std::io::{Read, Write};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let abort = Arc::new(AtomicBool::new(false));
    let (mux, accept) = PortMux::start(listener, Arc::clone(&abort)).unwrap();
    let port = mux.port();
    let salt: SessionSalt = [9u8; SESSION_SALT_BYTES];

    let board = crate::distributed::status::StatusBoard::new();
    board.configure_start(StartMode::Manual, salt_to_hex(&salt));
    let status_source = StreamSource::Mux(accept.status);
    let board_srv = board.clone();
    let abort_srv = Arc::clone(&abort);
    let status_srv = std::thread::spawn(move || {
        crate::distributed::status::serve_status(status_source, board_srv, abort_srv);
    });

    let config = JoinConfig {
        min_rank_start: 1,
        start_mode: StartMode::Manual,
        join_timeout_secs: 30,
        max_join_timeout_secs: 60,
        ..test_config()
    };
    let join_source = StreamSource::Mux(accept.join);
    let gate_salt = salt;
    let gate_abort = Arc::clone(&abort);
    let gate_board = board.clone();
    let gate = std::thread::spawn(move || {
        run_join_window(
            &join_source, &config, &gate_salt, true, None, None, &gate_abort,
            &gate_board,
        )
    });

    // Walk in (quorum met → staging hold), then fire the start switch
    // from loopback, exactly as `fdl start` does on the controller box.
    let (_conn, reply) = dial_and_join(port, &salt, "host-a", 1);
    assert!(matches!(reply, JoinMsgWire::Accept { .. }));
    let mut post = TcpStream::connect(("127.0.0.1", port)).unwrap();
    post.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    post.write_all(
        b"POST /start HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\
          Content-Length: 0\r\n\r\n",
    )
    .unwrap();
    let mut response = String::new();
    post.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    let formed = gate.join().unwrap().expect("world forms on operator start");
    assert_eq!(formed.world_size, 1);

    // THE invariant: a fresh dial on another channel still routes —
    // the dispatcher survived the staging flow.
    let mut ctrl = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write_channel_magic(&mut ctrl, CHANNEL_MAGIC_CONTROL).unwrap();
    let routed = accept.control.recv_timeout(Duration::from_secs(5));
    assert!(
        routed.is_ok(),
        "mux dispatcher must survive operator start (got {routed:?})",
    );

    abort.store(true, std::sync::atomic::Ordering::SeqCst);
    let _ = status_srv.join();
    drop(mux);
}

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
            &source, &config, &salt, pre_shared_salt, None, None, &abort, &status,
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
        run_id: None,
        nccl_version: None,
        model_sig: None,
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
        run_id: None,
        nccl_version: None,
        model_sig: None,
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
