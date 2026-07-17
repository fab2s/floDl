//! Embedded HTTP server for the live training dashboard.
//!
//! Serves a self-contained HTML page at `/` and pushes epoch updates
//! via Server-Sent Events at `/events`. No external dependencies.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Per-SSE-client event queue depth. A client that stops reading (stalled
/// browser tab, dead NAT entry) fills its queue and is disconnected instead
/// of growing server memory without bound.
const SSE_CLIENT_QUEUE: usize = 64;

/// Dashboard HTML — embedded at compile time.
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

/// Messages from the Monitor to the server.
pub(crate) enum ServerMsg {
    /// New epoch data as JSON string.
    Epoch(String),
    /// Updated SVG graph.
    SetSvg(String),
    /// Graph label and structural hash for dashboard header.
    SetLabelHash(Option<String>, Option<String>),
    /// Hardware summary for dashboard header.
    SetHardware(String),
    /// JSON metadata (training config, parameters, etc.).
    SetMetadata(String),
    /// GPU init data (JSON array of {dev, name, vram_total}).
    SetGpuInit(String),
    /// Clean shutdown.
    Shutdown,
}

/// A background HTTP server for the live training dashboard.
pub(crate) struct DashboardServer {
    tx: Sender<ServerMsg>,
    /// Bound address — `shutdown` dials it to wake the blocking accept.
    addr: SocketAddr,
    state: Arc<SharedState>,
    accept_handle: Option<JoinHandle<()>>,
    msg_handle: Option<JoinHandle<()>>,
}

/// Shared state between handler threads.
struct SharedState {
    /// All epoch events seen so far (for catch-up on new SSE connections).
    epochs: Mutex<Vec<String>>,
    /// Current SVG graph.
    svg: Mutex<Option<String>>,
    /// SSE client senders — each connected SSE client has a bounded
    /// channel (see [`SSE_CLIENT_QUEUE`]).
    sse_senders: Mutex<Vec<SyncSender<String>>>,
    /// Graph label for dashboard header.
    label: Mutex<Option<String>>,
    /// Structural hash for dashboard header.
    hash: Mutex<Option<String>>,
    /// Hardware summary string.
    hardware: Mutex<Option<String>>,
    /// JSON metadata string.
    metadata: Mutex<Option<String>>,
    /// GPU init data for immediate tab creation.
    gpu_init: Mutex<Option<String>>,
    /// Set (before the SSE sender list is cleared) once shutdown begins,
    /// so a connection racing shutdown never registers into the cleared
    /// list and blocks forever. Checked under the `sse_senders` lock.
    shutting_down: AtomicBool,
}

/// Loopback host literals that need no exposure warning.
fn is_loopback_addr(a: &str) -> bool {
    matches!(a, "127.0.0.1" | "::1" | "localhost")
}

/// Resolve the dashboard's bind address.
///
/// Defaults to loopback (`127.0.0.1`) so this unauthenticated metrics server is
/// never exposed to the network unless the operator explicitly asks. Set
/// `FLODL_DASHBOARD_BIND=<addr>` to widen it (e.g. `0.0.0.0` for LAN / remote
/// access); a non-loopback value prints a loud no-auth warning, since the
/// dashboard has no authentication and an SSH tunnel is the safer way to view
/// it remotely.
pub(crate) fn resolve_dashboard_bind() -> String {
    match std::env::var("FLODL_DASHBOARD_BIND") {
        Ok(a) if !a.trim().is_empty() => {
            let addr = a.trim().to_string();
            if !is_loopback_addr(&addr) {
                eprintln!(
                    "flodl: dashboard binding to {addr} — it has NO authentication, \
                     so anyone who can reach that address can view training metrics. \
                     Prefer an SSH tunnel: `ssh -L <port>:localhost:<port> <host>`."
                );
            }
            addr
        }
        _ => "127.0.0.1".to_string(),
    }
}

/// Whether the dashboard will bind loopback (the default). Mirrors
/// [`resolve_dashboard_bind`]'s classification WITHOUT emitting the warning, so
/// callers can tailor the printed URL (loopback → advise an SSH tunnel) without
/// double-warning.
pub(crate) fn dashboard_bind_is_loopback() -> bool {
    match std::env::var("FLODL_DASHBOARD_BIND") {
        Ok(a) => a.trim().is_empty() || is_loopback_addr(a.trim()),
        Err(_) => true,
    }
}

impl DashboardServer {
    /// Start the dashboard server on the given port.
    ///
    /// Binds loopback by default; widen via `FLODL_DASHBOARD_BIND`
    /// (see [`resolve_dashboard_bind`]).
    pub fn start(port: u16) -> std::io::Result<Self> {
        let bind = resolve_dashboard_bind();
        let listener = TcpListener::bind((bind.as_str(), port))?;
        let addr = listener.local_addr()?;
        let (tx, rx) = mpsc::channel::<ServerMsg>();

        let state = Arc::new(SharedState {
            epochs: Mutex::new(Vec::new()),
            svg: Mutex::new(None),
            sse_senders: Mutex::new(Vec::new()),
            label: Mutex::new(None),
            hash: Mutex::new(None),
            hardware: Mutex::new(None),
            metadata: Mutex::new(None),
            gpu_init: Mutex::new(None),
            shutting_down: AtomicBool::new(false),
        });

        // Message handler thread: receives from Monitor, broadcasts to SSE clients
        let state2 = state.clone();
        let msg_handle = thread::spawn(move || {
            handle_messages(rx, state2);
        });

        // Acceptor thread: accepts TCP connections, spawns handler per
        // connection. Exits when `shutdown` sets the flag and dials the
        // bound address to wake the blocking accept — dropping the
        // listener here is what frees the port.
        let state3 = state.clone();
        let accept_handle = thread::spawn(move || {
            for stream in listener.incoming() {
                if state3.shutting_down.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let state = state3.clone();
                thread::spawn(move || {
                    handle_connection(stream, &state);
                });
            }
        });

        Ok(Self {
            tx,
            addr,
            state,
            accept_handle: Some(accept_handle),
            msg_handle: Some(msg_handle),
        })
    }

    /// Push an epoch update to all connected dashboard clients.
    pub fn push_epoch(&self, json: String) {
        let _ = self.tx.send(ServerMsg::Epoch(json));
    }

    /// Update the graph SVG.
    pub fn set_svg(&self, svg: String) {
        let _ = self.tx.send(ServerMsg::SetSvg(svg));
    }

    /// Set graph label and structural hash for the dashboard header.
    pub fn set_label_hash(&self, label: Option<String>, hash: Option<String>) {
        let _ = self.tx.send(ServerMsg::SetLabelHash(label, hash));
    }

    /// Set hardware summary for the dashboard header.
    pub fn set_hardware(&self, hw: String) {
        let _ = self.tx.send(ServerMsg::SetHardware(hw));
    }

    /// Set JSON metadata for the dashboard.
    pub fn set_metadata(&self, json: String) {
        let _ = self.tx.send(ServerMsg::SetMetadata(json));
    }

    /// Set GPU init data for immediate tab creation on dashboard load.
    pub fn set_gpu_init(&self, json: String) {
        let _ = self.tx.send(ServerMsg::SetGpuInit(json));
    }

    /// Signal shutdown, wait for the message handler, and free the port.
    ///
    /// The message handler's `Shutdown` arm clears `sse_senders`, which
    /// disconnects every SSE handler's receive loop so those threads exit
    /// and close their connections. The acceptor is woken by a dial to
    /// the bound address and exits on the flag, dropping the listener —
    /// without this the port stayed bound (and SSE threads stayed
    /// blocked) until process exit. Idempotent.
    pub fn shutdown(&mut self) {
        // Flag first: connections racing shutdown must not register into
        // the sender list after the message handler clears it.
        self.state.shutting_down.store(true, Ordering::SeqCst);
        let _ = self.tx.send(ServerMsg::Shutdown);
        if let Some(h) = self.msg_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.accept_handle.take() {
            let _ = TcpStream::connect(self.addr);
            let _ = h.join();
        }
    }
}

impl Drop for DashboardServer {
    fn drop(&mut self) {
        // Belt-and-suspenders for callers that never reach the explicit
        // shutdown (both joins are `take`n, so a prior shutdown makes
        // this a no-op).
        self.shutdown();
    }
}

/// Process incoming messages from the Monitor.
fn handle_messages(rx: Receiver<ServerMsg>, state: Arc<SharedState>) {
    for msg in rx {
        match msg {
            ServerMsg::Epoch(json) => {
                let event = format!("event: epoch\ndata: {}\n\n", json);
                state.epochs.lock().unwrap().push(json);
                let mut senders = state.sse_senders.lock().unwrap();
                // try_send: a full queue means the client stopped reading —
                // drop it (its handler thread then exits) instead of
                // queueing events for it without bound.
                senders.retain(|tx| tx.try_send(event.clone()).is_ok());
            }
            ServerMsg::SetSvg(svg) => {
                *state.svg.lock().unwrap() = Some(svg);
            }
            ServerMsg::SetLabelHash(label, hash) => {
                *state.label.lock().unwrap() = label;
                *state.hash.lock().unwrap() = hash;
            }
            ServerMsg::SetHardware(hw) => {
                *state.hardware.lock().unwrap() = Some(hw);
            }
            ServerMsg::SetMetadata(json) => {
                *state.metadata.lock().unwrap() = Some(json);
            }
            ServerMsg::SetGpuInit(json) => {
                *state.gpu_init.lock().unwrap() = Some(json);
            }
            ServerMsg::Shutdown => {
                let event = "event: complete\ndata: {}\n\n".to_string();
                let mut senders = state.sse_senders.lock().unwrap();
                for tx in senders.iter() {
                    let _ = tx.try_send(event.clone());
                }
                // Dropping every sender ends each SSE handler's receive
                // loop; the threads exit and close their connections.
                senders.clear();
                break;
            }
        }
    }
}

/// Handle a single HTTP connection.
fn handle_connection(mut stream: TcpStream, state: &SharedState) {
    let mut buf = [0u8; 2048];
    let n = stream.read(&mut buf).unwrap_or(0);
    if n == 0 {
        return;
    }

    let request = String::from_utf8_lossy(&buf[..n]);
    let path = parse_path(&request);

    match path {
        "/" => serve_html(&mut stream, state),
        "/events" => serve_sse(stream, state),
        "/graph.svg" => serve_svg(&mut stream, state),
        "/api/history" => serve_history(&mut stream, state),
        _ => {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        }
    }
}

/// Extract the request path from the first line.
fn parse_path(request: &str) -> &str {
    request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Serve the dashboard HTML, injecting label/hash/metadata constants if set.
fn serve_html(stream: &mut TcpStream, state: &SharedState) {
    let label = state.label.lock().unwrap();
    let hash = state.hash.lock().unwrap();
    let hardware = state.hardware.lock().unwrap();
    let metadata = state.metadata.lock().unwrap();
    let gpu_init = state.gpu_init.lock().unwrap();

    let has_inject = label.is_some() || hash.is_some() || hardware.is_some()
        || metadata.is_some() || gpu_init.is_some();
    let body = if has_inject {
        let label_js = match &*label {
            Some(l) => format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let hash_js = match &*hash {
            Some(h) => format!("\"{}\"", h),
            None => "null".to_string(),
        };
        let hw_js = match &*hardware {
            Some(h) => format!("\"{}\"", h.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let meta_js = match &*metadata {
            Some(m) => m.clone(),
            None => "null".to_string(),
        };
        let gpu_init_js = match &*gpu_init {
            Some(j) => j.clone(),
            None => "null".to_string(),
        };
        let inject = format!(
            "<script>const LIVE_LABEL={};const LIVE_HASH={};const LIVE_HARDWARE={};const LIVE_META={};const LIVE_GPU_INIT={};</script>\n",
            label_js, hash_js, hw_js, meta_js, gpu_init_js,
        );
        DASHBOARD_HTML.replace("<script>", &format!("{}<script>", inject))
    } else {
        DASHBOARD_HTML.to_string()
    };

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body,
    );
    let _ = stream.write_all(response.as_bytes());
}

/// Hold the connection open as an SSE stream.
fn serve_sse(mut stream: TcpStream, state: &SharedState) {
    let headers = "HTTP/1.1 200 OK\r\n\
                   Content-Type: text/event-stream\r\n\
                   Cache-Control: no-cache\r\n\
                   Connection: keep-alive\r\n\
                   Access-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(headers.as_bytes()).is_err() {
        return;
    }

    // Send existing epochs as catch-up
    {
        let epochs = state.epochs.lock().unwrap();
        for json in epochs.iter() {
            let event = format!("event: epoch\ndata: {}\n\n", json);
            if stream.write_all(event.as_bytes()).is_err() {
                return;
            }
        }
        let _ = stream.flush();
    }

    // Register for future events (bounded — see SSE_CLIENT_QUEUE).
    let (tx, rx) = mpsc::sync_channel::<String>(SSE_CLIENT_QUEUE);
    {
        let mut senders = state.sse_senders.lock().unwrap();
        if state.shutting_down.load(Ordering::SeqCst) {
            // The clear already happened (or is imminent under this
            // lock): registering now would block this thread forever on
            // a sender nobody drops.
            return;
        }
        senders.push(tx);
    }

    // Block on receiving events until the client disconnects
    for event in rx {
        if stream.write_all(event.as_bytes()).is_err() {
            break;
        }
        let _ = stream.flush();
    }
}

/// Serve the current SVG graph.
fn serve_svg(stream: &mut TcpStream, state: &SharedState) {
    let svg = state.svg.lock().unwrap();
    if let Some(ref s) = *svg {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\nContent-Length: {}\r\n\r\n{}",
            s.len(),
            s,
        );
        let _ = stream.write_all(response.as_bytes());
    } else {
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
    }
}

/// Serve all epoch history as JSON (for late-connecting dashboards).
fn serve_history(stream: &mut TcpStream, state: &SharedState) {
    let epochs = state.epochs.lock().unwrap();
    let body = format!("[{}]", epochs.join(","));
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body,
    );
    let _ = stream.write_all(response.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_loopback_addr_classifies_bind_targets() {
        // M18: loopback literals need no exposure warning; everything else
        // (all-interfaces, a concrete LAN IP) does and is not "loopback".
        assert!(is_loopback_addr("127.0.0.1"));
        assert!(is_loopback_addr("::1"));
        assert!(is_loopback_addr("localhost"));
        assert!(!is_loopback_addr("0.0.0.0"));
        assert!(!is_loopback_addr("192.168.1.5"));
        assert!(!is_loopback_addr("::"));
    }

    #[test]
    fn shutdown_closes_sse_and_frees_port() {
        // Previously shutdown() joined only the message thread: the
        // acceptor kept the port bound and SSE handler threads stayed
        // blocked on their channels until process exit.
        let mut srv = DashboardServer::start(0).expect("bind ephemeral port");
        let addr = srv.addr;

        // Connect an SSE client and wait for the headers.
        let mut sse = TcpStream::connect(addr).unwrap();
        sse.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        sse.write_all(b"GET /events HTTP/1.1\r\nHost: x\r\n\r\n")
            .unwrap();
        let mut buf = [0u8; 512];
        let n = sse.read(&mut buf).unwrap();
        assert!(n > 0, "SSE headers expected");
        // Give the handler a moment to register its sender, so the test
        // exercises the registered-client teardown path.
        std::thread::sleep(std::time::Duration::from_millis(100));

        srv.push_epoch("{\"epoch\":1}".to_string());
        srv.shutdown();

        // The SSE stream must reach EOF (handler exited, socket closed),
        // not hang — the read timeout turns a regression into an Err.
        loop {
            match sse.read(&mut buf) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(e) => panic!("SSE socket still open after shutdown: {e}"),
            }
        }

        // And the port is actually free again.
        std::net::TcpListener::bind(addr).expect("port must be free after shutdown");
    }
}
