//! Run-status observability: one shared membership snapshot behind a
//! tiny HTTP responder on the controller's mux port.
//!
//! The controller publishes every membership/phase transition to a
//! [`StatusBoard`]; [`serve_status`] answers plain HTTP GETs with the
//! latest snapshot as `state.json`. The responder rides the single-port
//! mux — an HTTP request's first four bytes (`"GET "`) route like any
//! channel-select magic — so status shares the controller address every
//! other channel uses and is live for the WHOLE run lifecycle: the mux
//! binds before the join window opens, which is what makes the
//! `waiting`/`forming` phases observable at all. (The training
//! dashboard cannot host this: its HTTP server binds lazily on the
//! first rank-emitted register frame, on a port only rank user-code
//! knows — pre-formation there are no ranks.)
//!
//! Trust follows bind scope, exactly like join admission: on a loopback
//! bind (all workers tunneled) the endpoint is reachable only through
//! sshd; on a network bind, anyone who can reach the training port can
//! read run metadata (phase, host names, GPU inventory) — the same
//! class of information the training dashboard exposes, and strictly
//! less than what the cleartext training frames on that port already
//! carry.
//!
//! The read surface never mutates run state. The one mutation is the
//! **operator start switch**: `POST /start` (`fdl start`) arms the
//! staging hold's topology freeze under `start: manual | hybrid`.
//! Authentication mirrors the join trust model: a loopback peer is
//! trusted (the local operator, or someone already authenticated by
//! sshd), any other peer must present the session credential
//! (`?token=<salt-hex>` — the `join.token` on runs meant to be started
//! remotely). Arming is idempotent and quorum-gated at BOTH layers
//! (refused here for operator feedback, re-checked by the window's
//! verdict poll, which is the authority).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::membership::{ClusterPhase, MembershipSnapshot, StartMode};
use super::port_mux::StreamSource;
use super::wire::scaled_deadline_secs;

/// Poll cadence of the responder's non-blocking accept loop.
const ACCEPT_POLL: Duration = Duration::from_millis(20);

/// Budget for a client to deliver its request line. HTTP clients send
/// the whole request immediately after `connect`; only a wedged peer
/// runs this out.
const REQUEST_TIMEOUT_SECS: u64 = 5;

/// Upper bound on the request bytes we read before answering. The
/// request line is all we route on; anything longer is header noise.
const MAX_REQUEST_BYTES: usize = 2048;

/// Operator start-switch wiring, configured by the launcher before the
/// window opens. `mode` gates whether `POST /start` is meaningful at
/// all; `token_hex` is the non-loopback credential (the session salt).
struct StartSwitch {
    mode: StartMode,
    token_hex: String,
}

/// Everything the board publishes, behind one lock. The rendered JSON
/// and its typed source live together so they cannot disagree (the
/// `POST /start` handler needs the typed form for phase/quorum gating,
/// and parsing the JSON back would be silly), and so recording the
/// dashboard port can re-render atomically against the same snapshot.
#[derive(Default)]
struct Published {
    /// Pre-serialized document served by the HTTP responder.
    json: Option<String>,
    /// The membership snapshot `json` was rendered from.
    latest: Option<MembershipSnapshot>,
    /// Where this run's training dashboard is listening, once a rank has
    /// asked for one and the launcher's sink has bound it. Not part of
    /// [`MembershipSnapshot`] on purpose: the ledger answers who joined,
    /// and has no business knowing about dashboards. The status
    /// *document* is a superset of it.
    dashboard_port: Option<u16>,
}

/// Render the status document: the membership snapshot plus the fields
/// the board owns. `dashboard_port` is always present so a consumer can
/// tell "no dashboard" (null) from "old flodl" (absent).
fn render(snapshot: &MembershipSnapshot, dashboard_port: Option<u16>) -> Option<String> {
    let mut value = serde_json::to_value(snapshot).ok()?;
    let obj = value.as_object_mut()?;
    obj.insert(
        "dashboard_port".to_string(),
        match dashboard_port {
            Some(port) => serde_json::json!(port),
            None => serde_json::Value::Null,
        },
    );
    serde_json::to_string(&value).ok()
}

/// Shared, cloneable slot holding the latest membership state as
/// pre-serialized JSON. The controller publishes transitions; the HTTP
/// responder (and the debug log) serve the SAME string, so the log and
/// the HTTP surface can never disagree. Also carries the operator
/// start switch: the HTTP responder arms it, the join window polls it.
#[derive(Clone, Default)]
pub(crate) struct StatusBoard {
    published: Arc<Mutex<Published>>,
    start: Arc<Mutex<Option<StartSwitch>>>,
    start_requested: Arc<AtomicBool>,
}

impl StatusBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a snapshot: serialize once, store for the HTTP endpoint,
    /// and mirror to the debug stream.
    pub fn publish(&self, snapshot: &MembershipSnapshot) {
        let mut published = self.published.lock().unwrap();
        let Some(js) = render(snapshot, published.dashboard_port) else {
            // Serialize on plain data types cannot realistically fail;
            // if it somehow does, keep serving the previous state.
            return;
        };
        if crate::log::enabled(crate::log::Verbosity::Debug) {
            crate::debug!("cluster membership: state {js}");
        }
        published.json = Some(js);
        published.latest = Some(snapshot.clone());
    }

    /// Record where the training dashboard bound, and re-render the
    /// current document so the endpoint carries it immediately.
    ///
    /// The re-render is the point: the sink binds on the first rank's
    /// register frame, which happens AFTER the last membership
    /// transition, so waiting for the next `publish` would mean waiting
    /// for one that may never come. This is what lets anything holding
    /// the controller address (`fdl status`, `fdl ui`'s run tab) find
    /// the dashboard without being told the port.
    pub fn set_dashboard_port(&self, port: u16) {
        let mut published = self.published.lock().unwrap();
        published.dashboard_port = Some(port);
        if let Some(snapshot) = published.latest.take() {
            if let Some(js) = render(&snapshot, Some(port)) {
                published.json = Some(js);
            }
            published.latest = Some(snapshot);
        }
    }

    /// Latest published state, if any.
    pub fn state_json(&self) -> Option<String> {
        self.published.lock().unwrap().json.clone()
    }

    /// Wire the operator start switch (launcher, before the window
    /// opens). Without this call `POST /start` answers "no start switch
    /// this run" — e.g. paths that never run a join window.
    pub fn configure_start(&self, mode: StartMode, token_hex: String) {
        *self.start.lock().unwrap() = Some(StartSwitch { mode, token_hex });
    }

    /// Whether the operator has armed the start switch. Polled by the
    /// join window's verdict loop.
    pub fn start_requested(&self) -> bool {
        self.start_requested.load(Ordering::SeqCst)
    }
}

/// Serve `state.json` to HTTP clients from the mux's GET leg until the
/// mux shuts down (source disconnect) or `abort` is raised. One request
/// per connection (`Connection: close`) — status polls are sparse and
/// short-lived, keep-alive buys nothing.
pub(crate) fn serve_status(source: StreamSource, board: StatusBoard, abort: Arc<AtomicBool>) {
    loop {
        if abort.load(Ordering::SeqCst) {
            return;
        }
        match source.try_accept("cluster status") {
            Ok(Some(stream)) => answer_status_request(stream, &board),
            Ok(None) => std::thread::sleep(ACCEPT_POLL),
            // Mux dispatcher exited — the run is tearing down.
            Err(_) => return,
        }
    }
}

/// Read one HTTP request and answer it. All failures just drop the
/// connection — a status client can never harm the run.
fn answer_status_request(mut stream: TcpStream, board: &StatusBoard) {
    let deadline = Duration::from_secs(scaled_deadline_secs(REQUEST_TIMEOUT_SECS));
    if stream.set_read_timeout(Some(deadline)).is_err()
        || stream
            .set_write_timeout(Some(super::wire::write_stall_timeout()))
            .is_err()
    {
        return;
    }
    let Some((method, target)) = read_request_line(&mut stream) else {
        return;
    };
    // Split `/start?token=...` into path + query.
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (target.as_str(), None),
    };

    let (status_line, body) = match (method.as_str(), path) {
        // `/` answers too: someone pointing a browser at the training
        // port should find the state, not a hang.
        ("GET", "/state.json") | ("GET", "/") => match board.state_json() {
            Some(js) => ("200 OK", js),
            None => (
                "503 Service Unavailable",
                r#"{"error":"no membership state published yet"}"#.to_string(),
            ),
        },
        ("POST", "/start") => {
            // Trust mirrors join admission: a loopback peer got here
            // through sshd (or IS the controller host); anyone else
            // must present the session credential.
            let peer_is_loopback = stream
                .peer_addr()
                .map(|a| a.ip().is_loopback())
                .unwrap_or(false);
            handle_start(peer_is_loopback, query, board)
        }
        _ => (
            "404 Not Found",
            r#"{"error":"not found","endpoints":["GET /state.json","POST /start"]}"#.to_string(),
        ),
    };
    let response = format!(
        "HTTP/1.1 {status_line}\r\n\
         Content-Type: application/json\r\n\
         Cache-Control: no-cache\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len(),
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// The operator start switch: authenticate, gate on mode/phase/quorum,
/// arm. Every refusal names its reason — the operator is reading this
/// in a terminal through `fdl start`. `peer_is_loopback` is the
/// caller-resolved trust scope (see the dispatch site).
fn handle_start(
    peer_is_loopback: bool,
    query: Option<&str>,
    board: &StatusBoard,
) -> (&'static str, String) {
    let start = board.start.lock().unwrap();
    let Some(switch) = start.as_ref() else {
        return (
            "409 Conflict",
            r#"{"error":"no start switch this run (no join window with a start mode)"}"#
                .to_string(),
        );
    };
    if switch.mode == StartMode::Auto {
        return (
            "409 Conflict",
            r#"{"error":"start mode is auto — the window closes on target/expiry; set `controller.join.start: manual` (or hybrid) to hold for an operator start"}"#
                .to_string(),
        );
    }
    if !peer_is_loopback {
        let presented = query.and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("token=")));
        if presented != Some(switch.token_hex.as_str()) {
            return (
                "403 Forbidden",
                r#"{"error":"start refused: non-loopback peer without a valid token (pass --token, or fire from the controller host / through the sshd tunnel)"}"#
                    .to_string(),
            );
        }
    }
    let published = board.published.lock().unwrap();
    let Some(snap) = published.latest.as_ref() else {
        return (
            "503 Service Unavailable",
            r#"{"error":"no membership state published yet"}"#.to_string(),
        );
    };
    match snap.phase {
        ClusterPhase::Waiting | ClusterPhase::Staging => {}
        phase => {
            return (
                "409 Conflict",
                format!(
                    r#"{{"error":"window is not open (phase: {})"}}"#,
                    serde_json::to_string(&phase)
                        .unwrap_or_default()
                        .trim_matches('"'),
                ),
            );
        }
    }
    if snap.joined_ranks < snap.min_rank_start {
        return (
            "409 Conflict",
            format!(
                r#"{{"error":"quorum not met: {}/{} ranks joined — check fdl status and fire again"}}"#,
                snap.joined_ranks, snap.min_rank_start,
            ),
        );
    }
    board.start_requested.store(true, Ordering::SeqCst);
    (
        "200 OK",
        format!(
            r#"{{"armed":true,"joined_ranks":{},"min_rank_start":{}}}"#,
            snap.joined_ranks, snap.min_rank_start,
        ),
    )
}

/// Read up to the end of the request line and extract the method + path
/// tokens (`GET <path> HTTP/1.1`). `None` on a malformed or truncated
/// request.
fn read_request_line(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0u8; 256];
    while !buf.windows(2).any(|w| w == b"\r\n") && buf.len() < MAX_REQUEST_BYTES {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => return None,
        }
    }
    let text = String::from_utf8_lossy(&buf);
    let line = text.lines().next()?;
    let mut tokens = line.split_whitespace();
    let method = tokens.next()?.to_string();
    let path = tokens.next()?.to_string();
    Some((method, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::membership::{ClusterPhase, MemberSnapshot};
    use std::net::TcpListener;

    fn snapshot(phase: ClusterPhase) -> MembershipSnapshot {
        MembershipSnapshot {
            phase,
            joined_ranks: 3,
            joined_hosts: 2,
            min_rank_start: 3,
            target_ranks: Some(3),
            window_remaining_secs: Some(120),
            cap_remaining_secs: Some(400),
            start_mode: StartMode::Auto,
            start_armed: false,
            members: vec![MemberSnapshot {
                host: "node-a".to_string(),
                ranks: vec![0],
                local_devices: vec![0],
                gpus: vec!["Fake GPU".to_string()],
                libtorch: "precompiled/cu128".to_string(),
                joined_at_secs: 2,
            }],
        }
    }

    #[test]
    fn the_dashboard_port_reaches_the_document_after_the_last_publish() {
        let board = StatusBoard::new();
        board.publish(&snapshot(ClusterPhase::Training));
        // No dashboard yet: the key is present and null, so a consumer
        // can tell "this run serves none" from "this flodl is older".
        assert!(
            board
                .state_json()
                .unwrap()
                .contains(r#""dashboard_port":null"#),
            "{}",
            board.state_json().unwrap()
        );

        // The sink binds after formation, which is after the last
        // membership transition — so recording the port must re-render
        // the current document rather than wait for a publish that may
        // never come. This is the whole point of the field.
        board.set_dashboard_port(8099);
        let js = board.state_json().unwrap();
        assert!(js.contains(r#""dashboard_port":8099"#), "{js}");
        // Re-rendered against the same snapshot, nothing else lost.
        assert!(js.contains(r#""phase":"training""#), "{js}");
        assert!(js.contains(r#""host":"node-a""#), "{js}");

        // And it survives later membership transitions (elastic
        // scale-down keeps publishing while the dashboard stays put).
        board.publish(&snapshot(ClusterPhase::Waiting));
        assert!(
            board
                .state_json()
                .unwrap()
                .contains(r#""dashboard_port":8099"#),
            "{}",
            board.state_json().unwrap()
        );
    }

    #[test]
    fn board_publish_and_read_back() {
        let board = StatusBoard::new();
        assert!(board.state_json().is_none());
        board.publish(&snapshot(ClusterPhase::Waiting));
        let js = board.state_json().unwrap();
        assert!(js.contains(r#""phase":"waiting""#));
        assert!(js.contains(r#""host":"node-a""#));
        // Later publishes replace, not append.
        board.publish(&snapshot(ClusterPhase::Training));
        let js = board.state_json().unwrap();
        assert!(js.contains(r#""phase":"training""#));
        assert!(!js.contains("waiting"));
    }

    /// Spawn the responder on a direct listener (byte-identical streams
    /// to the muxed path) and GET `path` against it.
    fn http_get(board: StatusBoard, path: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let source = StreamSource::from_listener(listener, "test").unwrap();
        let abort = Arc::new(AtomicBool::new(false));
        let abort_c = Arc::clone(&abort);
        let server = std::thread::spawn(move || {
            serve_status(source, board, abort_c);
        });

        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        drop(stream);

        abort.store(true, Ordering::SeqCst);
        server.join().unwrap();
        response
    }

    #[test]
    fn serves_state_json_over_http() {
        let board = StatusBoard::new();
        board.publish(&snapshot(ClusterPhase::Forming));
        let response = http_get(board, "/state.json");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains(r#""phase":"forming""#), "{response}");
        assert!(response.contains("Content-Type: application/json"));
    }

    #[test]
    fn root_path_aliases_state_json() {
        let board = StatusBoard::new();
        board.publish(&snapshot(ClusterPhase::Training));
        let response = http_get(board, "/");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains(r#""phase":"training""#), "{response}");
    }

    #[test]
    fn unpublished_board_answers_503() {
        let response = http_get(StatusBoard::new(), "/state.json");
        assert!(
            response.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{response}",
        );
    }

    #[test]
    fn unknown_path_answers_404_with_endpoint_hint() {
        let board = StatusBoard::new();
        board.publish(&snapshot(ClusterPhase::Waiting));
        let response = http_get(board, "/metrics");
        assert!(response.starts_with("HTTP/1.1 404 Not Found"), "{response}");
        assert!(response.contains("/state.json"), "{response}");
    }

    /// POST against the live responder (loopback peer by construction).
    fn http_post(board: StatusBoard, path: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let source = StreamSource::from_listener(listener, "test").unwrap();
        let abort = Arc::new(AtomicBool::new(false));
        let abort_c = Arc::clone(&abort);
        let server = std::thread::spawn(move || {
            serve_status(source, board, abort_c);
        });

        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .write_all(
                format!(
                    "POST {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
                     Content-Length: 0\r\n\r\n"
                )
                .as_bytes(),
            )
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        drop(stream);

        abort.store(true, Ordering::SeqCst);
        server.join().unwrap();
        response
    }

    fn staged_snapshot(mode: StartMode, joined: usize) -> MembershipSnapshot {
        MembershipSnapshot {
            phase: ClusterPhase::Staging,
            joined_ranks: joined,
            start_mode: mode,
            ..snapshot(ClusterPhase::Staging)
        }
    }

    #[test]
    fn start_arms_a_staged_manual_window_from_loopback() {
        let board = StatusBoard::new();
        board.configure_start(StartMode::Manual, "aa".repeat(16));
        board.publish(&staged_snapshot(StartMode::Manual, 3));
        assert!(!board.start_requested());
        let response = http_post(board.clone(), "/start");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains(r#""armed":true"#), "{response}");
        assert!(board.start_requested(), "flag must be armed");
    }

    #[test]
    fn start_refusals_name_their_reason() {
        // Unconfigured switch (no join window ran).
        let board = StatusBoard::new();
        board.publish(&snapshot(ClusterPhase::Waiting));
        let response = http_post(board, "/start");
        assert!(response.starts_with("HTTP/1.1 409"), "{response}");
        assert!(response.contains("no start switch"), "{response}");

        // Auto mode: the clock owns the close.
        let board = StatusBoard::new();
        board.configure_start(StartMode::Auto, "aa".repeat(16));
        board.publish(&snapshot(ClusterPhase::Waiting));
        let response = http_post(board.clone(), "/start");
        assert!(response.starts_with("HTTP/1.1 409"), "{response}");
        assert!(response.contains("auto"), "{response}");
        assert!(!board.start_requested());

        // Quorum unmet: counts in the reply.
        let board = StatusBoard::new();
        board.configure_start(StartMode::Manual, "aa".repeat(16));
        board.publish(&MembershipSnapshot {
            joined_ranks: 1,
            ..staged_snapshot(StartMode::Manual, 1)
        });
        let response = http_post(board.clone(), "/start");
        assert!(response.starts_with("HTTP/1.1 409"), "{response}");
        assert!(response.contains("quorum not met: 1/3"), "{response}");
        assert!(!board.start_requested());

        // Window already closed.
        let board = StatusBoard::new();
        board.configure_start(StartMode::Manual, "aa".repeat(16));
        board.publish(&MembershipSnapshot {
            phase: ClusterPhase::Training,
            ..staged_snapshot(StartMode::Manual, 3)
        });
        let response = http_post(board.clone(), "/start");
        assert!(response.starts_with("HTTP/1.1 409"), "{response}");
        assert!(response.contains("not open"), "{response}");
        assert!(!board.start_requested());
    }

    #[test]
    fn start_auth_requires_the_token_off_loopback() {
        // Non-loopback peers are exercised directly on the handler (a
        // test TCP peer is always loopback).
        let board = StatusBoard::new();
        board.configure_start(StartMode::Manual, "deadbeef".repeat(4));
        board.publish(&staged_snapshot(StartMode::Manual, 3));

        // No token → refused.
        let (code, body) = handle_start(false, None, &board);
        assert!(code.starts_with("403"), "{code} {body}");
        assert!(!board.start_requested());
        // Wrong token → refused.
        let (code, _) = handle_start(false, Some("token=wrong"), &board);
        assert!(code.starts_with("403"), "{code}");
        assert!(!board.start_requested());
        // Right token → armed.
        let (code, body) = handle_start(
            false,
            Some(&format!("token={}", "deadbeef".repeat(4))),
            &board,
        );
        assert!(code.starts_with("200"), "{code} {body}");
        assert!(board.start_requested());
    }
}
