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
//! carry. The endpoint is read-only and never mutates run state.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::membership::MembershipSnapshot;
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

/// Shared, cloneable slot holding the latest membership state as
/// pre-serialized JSON. The controller publishes transitions; the HTTP
/// responder (and the debug log) serve the SAME string, so the log and
/// the HTTP surface can never disagree.
#[derive(Clone, Default)]
pub(crate) struct StatusBoard {
    state_json: Arc<Mutex<Option<String>>>,
}

impl StatusBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a snapshot: serialize once, store for the HTTP endpoint,
    /// and mirror to the debug stream.
    pub fn publish(&self, snapshot: &MembershipSnapshot) {
        let Ok(js) = serde_json::to_string(snapshot) else {
            // Serialize on plain data types cannot realistically fail;
            // if it somehow does, keep serving the previous state.
            return;
        };
        if crate::log::enabled(crate::log::Verbosity::Debug) {
            crate::debug!("cluster membership: state {js}");
        }
        *self.state_json.lock().unwrap() = Some(js);
    }

    /// Latest published state, if any.
    pub fn state_json(&self) -> Option<String> {
        self.state_json.lock().unwrap().clone()
    }
}

/// Serve `state.json` to HTTP clients from the mux's GET leg until the
/// mux shuts down (source disconnect) or `abort` is raised. One request
/// per connection (`Connection: close`) — status polls are sparse and
/// short-lived, keep-alive buys nothing.
pub(crate) fn serve_status(
    source: StreamSource,
    board: StatusBoard,
    abort: Arc<AtomicBool>,
) {
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
    let deadline =
        Duration::from_secs(scaled_deadline_secs(REQUEST_TIMEOUT_SECS));
    if stream.set_read_timeout(Some(deadline)).is_err()
        || stream
            .set_write_timeout(Some(super::wire::write_stall_timeout()))
            .is_err()
    {
        return;
    }
    let Some(path) = read_request_path(&mut stream) else {
        return;
    };

    let (status_line, body) = match path.as_str() {
        // `/` answers too: someone pointing a browser at the training
        // port should find the state, not a hang.
        "/state.json" | "/" => match board.state_json() {
            Some(js) => ("200 OK", js),
            None => (
                "503 Service Unavailable",
                r#"{"error":"no membership state published yet"}"#.to_string(),
            ),
        },
        _ => (
            "404 Not Found",
            r#"{"error":"not found","endpoints":["/state.json"]}"#.to_string(),
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

/// Read up to the end of the request line and extract the path token
/// (`GET <path> HTTP/1.1`). `None` on a malformed or truncated request.
fn read_request_path(stream: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0u8; 256];
    while !buf.windows(2).any(|w| w == b"\r\n")
        && buf.len() < MAX_REQUEST_BYTES
    {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => return None,
        }
    }
    let text = String::from_utf8_lossy(&buf);
    let line = text.lines().next()?;
    line.split_whitespace().nth(1).map(str::to_string)
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
}
