//! Embedded HTTP server for the live training dashboard.
//!
//! Serves a self-contained HTML page at `/` and pushes epoch updates
//! via Server-Sent Events at `/events`. No external dependencies.
//!
//! ## Two feeds, side by side
//!
//! `/events` is the original whole-run epoch feed: every client gets every
//! epoch. It is unchanged and self-contained.
//!
//! On top of it sits the **path-scoped record plane** of the monitoring portal
//! (`.design/monitoring-portal-b3.md`) — `/node`, `/history`, `/stream` — where
//! a client names one `path` and receives only that level's neighbourhood.
//! Lean is a property of the *subscription*, not of the data the controller
//! holds: the controller keeps the whole tree
//! ([`crate::monitor::record_store::RecordStore`]) and every viewer draws one
//! level from it. That is what lets a single exposed root serve any depth,
//! which at real cluster scale is the only option — workers sit on a private
//! network and are not browser-reachable.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use serde_json::Value;

use super::record_store::{RecordStore, delivers};

/// Per-SSE-client event queue depth. A client that stops reading (stalled
/// browser tab, dead NAT entry) fills its queue and is disconnected instead
/// of growing server memory without bound.
const SSE_CLIENT_QUEUE: usize = 64;

/// Default `n` for `/history` when the query omits it.
const DEFAULT_HISTORY: usize = 512;

/// Upper bound on a `/history` request, so one client cannot ask the server to
/// serialize its whole ring repeatedly.
const MAX_HISTORY: usize = crate::monitor::record_store::MAX_RECORDS;

/// Dashboard HTML — embedded at compile time.
const DASHBOARD_HTML: &str = include_str!("dashboard.html");

/// Messages from the Monitor to the server.
pub(crate) enum ServerMsg {
    /// New epoch data as JSON string.
    Epoch(String),
    /// Updated SVG graph.
    SetSvg(String),
    SetHeatmaps(Vec<(String, String)>),
    /// Graph label and structural hash for dashboard header.
    SetLabelHash(Option<String>, Option<String>),
    /// Hardware summary for dashboard header.
    SetHardware(String),
    /// JSON metadata (training config, parameters, etc.).
    SetMetadata(String),
    /// GPU init data (JSON array of {dev, name, vram_total}).
    SetGpuInit(String),
    /// Path-keyed monitor records (`node` / `event` / `meta`) for the portal's
    /// record plane. Fanned out to `/stream` subscribers by path scope.
    Records(Vec<Value>),
    /// Clean shutdown.
    Shutdown,
}

/// One registered SSE client.
struct SseClient {
    /// `None` = the whole-run `/events` feed (epoch + complete).
    /// `Some(path)` = a `/stream?path=` subscriber, scoped to that level.
    scope: Option<String>,
    tx: SyncSender<String>,
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
    /// Baked per-GPU-model timing heat maps `(model label, finished SVG)`.
    heatmaps: Mutex<Vec<(String, String)>>,
    /// SSE clients — each has a bounded channel (see [`SSE_CLIENT_QUEUE`])
    /// and a path scope (`None` for the whole-run `/events` feed).
    sse_senders: Mutex<Vec<SseClient>>,
    /// Live path-addressable view of the record stream, serving `/node`,
    /// `/history`, and each `/stream` subscriber's catch-up preamble.
    records: Arc<Mutex<RecordStore>>,
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
    /// Start with a private record store. Test convenience only — production
    /// goes through [`Self::start_with_records`] so the store is owned by the
    /// [`Monitor`](crate::monitor::Monitor) and outlives the server.
    #[cfg(test)]
    pub fn start(port: u16) -> std::io::Result<Self> {
        Self::start_with_records(port, Arc::new(Mutex::new(RecordStore::new())))
    }

    /// Bind sharing an externally-owned record store.
    ///
    /// The store must outlive the server, because the record plane is a data
    /// structure and not an HTTP concern: a headless run keeps recording into it
    /// with no server bound, which is what lets the saved archive carry the
    /// levels without a live dashboard port.
    pub fn start_with_records(
        port: u16,
        records: Arc<Mutex<RecordStore>>,
    ) -> std::io::Result<Self> {
        let bind = resolve_dashboard_bind();
        let listener = TcpListener::bind((bind.as_str(), port))?;
        let addr = listener.local_addr()?;
        let (tx, rx) = mpsc::channel::<ServerMsg>();

        let state = Arc::new(SharedState {
            epochs: Mutex::new(Vec::new()),
            svg: Mutex::new(None),
            heatmaps: Mutex::new(Vec::new()),
            sse_senders: Mutex::new(Vec::new()),
            label: Mutex::new(None),
            hash: Mutex::new(None),
            hardware: Mutex::new(None),
            metadata: Mutex::new(None),
            gpu_init: Mutex::new(None),
            records,
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

    /// Install the baked timing heat maps (replaces the whole set).
    pub fn set_heatmaps(&self, maps: Vec<(String, String)>) {
        let _ = self.tx.send(ServerMsg::SetHeatmaps(maps));
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

    /// Push path-keyed monitor records into the portal's record plane.
    pub fn push_records(&self, records: Vec<Value>) {
        if records.is_empty() {
            return;
        }
        let _ = self.tx.send(ServerMsg::Records(records));
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
                // queueing events for it without bound. Path-scoped
                // subscribers are not epoch consumers, so they are left
                // alone rather than retained-with-success.
                senders.retain(|c| c.scope.is_some() || c.tx.try_send(event.clone()).is_ok());
            }
            ServerMsg::Records(records) => {
                {
                    let mut store = state.records.lock().unwrap();
                    store.insert_all(&records);
                    if store.take_path_cap_hit() {
                        crate::msg!(
                            "  warning: monitor record plane hit its {}-path cap; \
                             new paths are not indexed (a producer emitting \
                             unbounded paths?)",
                            crate::monitor::record_store::MAX_PATHS,
                        );
                    }
                }
                // Fan out per subscriber: each gets only the records its own
                // path scope covers, so a root viewer never receives a deep
                // rank's metrics (see `record_store::delivers`).
                let mut senders = state.sse_senders.lock().unwrap();
                senders.retain(|c| {
                    let Some(scope) = c.scope.as_deref() else {
                        return true; // legacy epoch client — not ours
                    };
                    for rec in &records {
                        if !delivers(scope, rec) {
                            continue;
                        }
                        let event = format!("event: record\ndata: {rec}\n\n");
                        if c.tx.try_send(event).is_err() {
                            return false;
                        }
                    }
                    true
                });
            }
            ServerMsg::SetHeatmaps(maps) => {
                *state.heatmaps.lock().unwrap() = maps;
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
                for c in senders.iter() {
                    let _ = c.tx.try_send(event.clone());
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
    let target = parse_path(&request);
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    match path {
        "/" => serve_html(&mut stream, state),
        "/events" => serve_sse(stream, state, None),
        "/graph.svg" => serve_svg(&mut stream, state),
        "/api/heatmaps" => serve_heatmap_index(&mut stream, state),
        "/heatmap.svg" => serve_heatmap(&mut stream, state, query),
        "/api/history" => serve_history(&mut stream, state),
        // Portal record plane — all three keyed by `?path=`.
        "/node" => serve_node(&mut stream, state, query),
        "/history" => serve_record_history(&mut stream, state, query),
        "/paths" => serve_paths(&mut stream, state),
        "/stream" => {
            let scope = query_param(query, "path").unwrap_or_else(|| "root".to_string());
            serve_sse(stream, state, Some(scope));
        }
        _ => {
            let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        }
    }
}

/// Extract the request target (path + query) from the first line.
fn parse_path(request: &str) -> &str {
    request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Value of `key` in a `a=1&b=2` query string, percent-decoded.
fn query_param(query: &str, key: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| percent_decode(v))
}

/// Minimal percent-decoding (plus `+` as space) for query values. Record
/// paths are `/`-separated, and a browser may send that either raw or as
/// `%2F`, so both must land on the same scope.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                    }
                    // Not a valid escape — keep the '%' verbatim rather than
                    // silently eating a character.
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `200 OK` with a JSON body.
fn write_json(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body,
    );
    let _ = stream.write_all(response.as_bytes());
}

/// `GET /node?path=<p>` — one level: node `p`'s current aggregate plus one
/// record per direct child. O(children), never O(cluster).
fn serve_node(stream: &mut TcpStream, state: &SharedState, query: &str) {
    let path = query_param(query, "path").unwrap_or_else(|| "root".to_string());
    let snap = state.records.lock().unwrap().snapshot(&path);
    write_json(stream, &snap.to_string());
}

/// `GET /history?path=<p>&n=<N>` — the last N records a `/stream` subscriber
/// at `p` would have received, so a viewer can read-then-subscribe without a
/// gap or a duplicate at the handover.
fn serve_record_history(stream: &mut TcpStream, state: &SharedState, query: &str) {
    let path = query_param(query, "path").unwrap_or_else(|| "root".to_string());
    let n = query_param(query, "n")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_HISTORY)
        .min(MAX_HISTORY);
    // Serialize under the lock, write outside it: a client that stops reading
    // must not be able to hold the record plane's lock through a multi-MB
    // `write_all` and stall every producer behind it.
    let body = {
        let store = state.records.lock().unwrap();
        let lines: Vec<String> = store
            .history(&path, n)
            .iter()
            .map(|r| r.to_string())
            .collect();
        format!("[{}]", lines.join(","))
    };
    write_json(stream, &body);
}

/// `GET /paths` — every path currently carrying a `node` record, sorted. The
/// portal's navigation index (what is there to drill into).
fn serve_paths(stream: &mut TcpStream, state: &SharedState) {
    let body = {
        let store = state.records.lock().unwrap();
        serde_json::to_string(&store.paths()).unwrap_or_else(|_| "[]".to_string())
    };
    write_json(stream, &body);
}

/// Serve the dashboard HTML, injecting label/hash/metadata constants if set.
fn serve_html(stream: &mut TcpStream, state: &SharedState) {
    // Cloned out from under the locks before any socket write — the message
    // thread sets every one of these under its own lock.
    let label = state.label.lock().unwrap().clone();
    let hash = state.hash.lock().unwrap().clone();
    let hardware = state.hardware.lock().unwrap().clone();
    let metadata = state.metadata.lock().unwrap().clone();
    let gpu_init = state.gpu_init.lock().unwrap().clone();

    let has_inject = label.is_some()
        || hash.is_some()
        || hardware.is_some()
        || metadata.is_some()
        || gpu_init.is_some();
    let body = if has_inject {
        let label_js = match &label {
            Some(l) => format!("\"{}\"", l.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let hash_js = match &hash {
            Some(h) => format!("\"{}\"", h),
            None => "null".to_string(),
        };
        let hw_js = match &hardware {
            Some(h) => format!("\"{}\"", h.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let meta_js = match &metadata {
            Some(m) => m.clone(),
            None => "null".to_string(),
        };
        let gpu_init_js = match &gpu_init {
            Some(j) => j.clone(),
            None => "null".to_string(),
        };
        // Neutralize any </script> in the injected DATA before wrapping it
        // in the tag — the HTML parser would otherwise close the block early
        // on a label/hash/hardware/metadata value containing </script>.
        let consts = super::neutralize_script_close(&format!(
            "const LIVE_LABEL={};const LIVE_HASH={};const LIVE_HARDWARE={};const LIVE_META={};const LIVE_GPU_INIT={};",
            label_js, hash_js, hw_js, meta_js, gpu_init_js,
        ));
        let inject = format!("<script>{}</script>\n", consts);
        // `replacen(.., 1)`: the constants must land ahead of the FIRST script
        // block only. A plain `replace` prepended them to every `<script>` in
        // the page, which was invisible only while the template happened to
        // have exactly one.
        DASHBOARD_HTML.replacen("<script>", &format!("{}<script>", inject), 1)
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
///
/// `scope` is `None` for the whole-run `/events` epoch feed, or `Some(path)`
/// for a `/stream?path=` portal subscriber. The preamble differs accordingly —
/// epochs for the former, the `meta` declarations plus this level's record
/// history for the latter — but registration and the bounded-queue teardown
/// are shared.
fn serve_sse(mut stream: TcpStream, state: &SharedState, scope: Option<String>) {
    let headers = "HTTP/1.1 200 OK\r\n\
                   Content-Type: text/event-stream\r\n\
                   Cache-Control: no-cache\r\n\
                   Connection: keep-alive\r\n\
                   Access-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(headers.as_bytes()).is_err() {
        return;
    }

    // Register THEN preamble, both feeds. An event arriving in the gap must
    // not be LOST (it will never be replayed to this client), and it cannot
    // be harmful if duplicated: every `node` record is an absolute snapshot,
    // and the epoch feed's consumers already key on the epoch number because
    // an `EventSource` reconnect replays the whole history. Loss is
    // unrecoverable, a duplicate is not. (The previous preamble-then-register
    // order on `/events` never actually prevented the duplicate/miss — the
    // message thread releases the epochs lock before broadcasting — it only
    // held the epochs lock across socket writes, letting one stalled client
    // freeze the message thread and, with it, the whole record plane.)
    let (tx, rx) = mpsc::sync_channel::<String>(SSE_CLIENT_QUEUE);
    {
        let mut senders = state.sse_senders.lock().unwrap();
        if state.shutting_down.load(Ordering::SeqCst) {
            // The clear already happened (or is imminent under this
            // lock): registering now would block this thread forever on
            // a sender nobody drops.
            return;
        }
        senders.push(SseClient {
            scope: scope.clone(),
            tx,
        });
    }

    // Preamble, copied out from under the lock before any socket write, so a
    // stalled client can never hold a producer-side lock.
    let preamble: Vec<String> = match scope.as_deref() {
        // `/events`: the epoch history.
        None => state
            .epochs
            .lock()
            .unwrap()
            .iter()
            .map(|json| format!("event: epoch\ndata: {json}\n\n"))
            .collect(),
        // `/stream`: `meta` first — it declares how to roll up, so a client
        // must have it before interpreting any record. Replayed per client
        // rather than assumed-received, since a viewer connects at any time.
        Some(path) => {
            let store = state.records.lock().unwrap();
            store
                .meta()
                .into_iter()
                .chain(store.history(path, DEFAULT_HISTORY))
                .map(|r| format!("event: record\ndata: {r}\n\n"))
                .collect()
        }
    };
    for event in preamble {
        if stream.write_all(event.as_bytes()).is_err() {
            return;
        }
    }
    let _ = stream.flush();

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
    // Cloned out from under the lock before the write — the message thread
    // sets the SVG under this lock.
    let svg = state.svg.lock().unwrap().clone();
    if let Some(s) = svg {
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

/// Serve the heat-map index as JSON: `[{"model":..,"url":..}]`. The
/// page shows the switcher only when this is non-empty (frames arrive
/// at teardown, so it usually fills right as the run completes).
fn serve_heatmap_index(stream: &mut TcpStream, state: &SharedState) {
    let body = {
        let maps = state.heatmaps.lock().unwrap();
        let mut b = String::from("[");
        for (i, (model, _)) in maps.iter().enumerate() {
            if i > 0 {
                b.push(',');
            }
            let _ = write!(
                b,
                "{{\"model\":\"{}\",\"url\":\"/heatmap.svg?i={}\"}}",
                model.replace('\\', "\\\\").replace('"', "\\\""),
                i
            );
        }
        b.push(']');
        b
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body,
    );
    let _ = stream.write_all(response.as_bytes());
}

/// Serve one baked heat-map SVG by index (`?i=N`). The
/// `Content-Disposition` filename makes a browser "save as" land on a
/// publication-ready name; the bytes are exactly what the page displays.
fn serve_heatmap(stream: &mut TcpStream, state: &SharedState, query: &str) {
    let entry = query_param(query, "i")
        .and_then(|i| i.parse::<usize>().ok())
        .and_then(|i| state.heatmaps.lock().unwrap().get(i).cloned());
    if let Some((model, svg)) = entry {
        let slug: String = model
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/svg+xml\r\nContent-Disposition: inline; filename=\"heatmap-{}.svg\"\r\nContent-Length: {}\r\n\r\n{}",
            slug.trim_matches('-'),
            svg.len(),
            svg,
        );
        let _ = stream.write_all(response.as_bytes());
    } else {
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
    }
}

/// Serve all epoch history as JSON (for late-connecting dashboards).
fn serve_history(stream: &mut TcpStream, state: &SharedState) {
    // Body built under the lock, written outside it (same rule as the
    // record-plane endpoints): the message thread needs this lock.
    let body = {
        let epochs = state.epochs.lock().unwrap();
        format!("[{}]", epochs.join(","))
    };
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

    /// The two heat-map routes, over a real socketpair: the index lists
    /// installed maps with stable urls, the svg route serves the exact
    /// bytes with a save-as filename, and a bad index is a plain 404.
    #[test]
    fn heatmap_routes_serve_index_and_artifact() {
        let state = Arc::new(SharedState {
            epochs: Mutex::new(Vec::new()),
            svg: Mutex::new(None),
            heatmaps: Mutex::new(vec![(
                "RTX 5060 Ti".to_string(),
                "<svg>baked</svg>".to_string(),
            )]),
            sse_senders: Mutex::new(Vec::new()),
            label: Mutex::new(None),
            hash: Mutex::new(None),
            hardware: Mutex::new(None),
            metadata: Mutex::new(None),
            gpu_init: Mutex::new(None),
            records: Arc::new(Mutex::new(RecordStore::new())),
            shutting_down: AtomicBool::new(false),
        });

        let respond = |path: &str, query: &str| -> String {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let client = TcpStream::connect(addr).unwrap();
            let (mut server_side, _) = listener.accept().unwrap();
            match path {
                "/api/heatmaps" => serve_heatmap_index(&mut server_side, &state),
                "/heatmap.svg" => serve_heatmap(&mut server_side, &state, query),
                _ => unreachable!(),
            }
            drop(server_side);
            let mut out = String::new();
            let mut c = client;
            let _ = c.read_to_string(&mut out);
            out
        };

        let index = respond("/api/heatmaps", "");
        assert!(index.contains("200 OK"));
        assert!(index.contains(r#"{"model":"RTX 5060 Ti","url":"/heatmap.svg?i=0"}"#));

        let svg = respond("/heatmap.svg", "i=0");
        assert!(svg.contains("200 OK"));
        assert!(svg.contains("<svg>baked</svg>"));
        assert!(
            svg.contains("filename=\"heatmap-rtx-5060-ti.svg\""),
            "save-as filename missing: {svg}",
        );

        assert!(respond("/heatmap.svg", "i=7").contains("404"));
        assert!(respond("/heatmap.svg", "").contains("404"));
    }

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
    fn query_param_decodes_and_misses_cleanly() {
        assert_eq!(query_param("path=root", "path").as_deref(), Some("root"));
        // A record path may arrive raw or percent-encoded; both must land on
        // the same scope, or a browser's encoding choice would silently
        // change which node it subscribes to.
        assert_eq!(
            query_param("path=root%2Fexa%2Frank0", "path").as_deref(),
            Some("root/exa/rank0"),
        );
        assert_eq!(
            query_param("path=root/exa", "path").as_deref(),
            Some("root/exa")
        );
        assert_eq!(
            query_param("n=10&path=root", "path").as_deref(),
            Some("root")
        );
        assert_eq!(query_param("n=10", "path"), None);
        assert_eq!(query_param("", "path"), None);
        // A malformed escape keeps the '%' rather than eating a character.
        assert_eq!(query_param("path=a%zz", "path").as_deref(), Some("a%zz"));
        assert_eq!(query_param("path=a%", "path").as_deref(), Some("a%"));
    }

    // --- the portal page's contract with its two injectors ---

    #[test]
    fn injected_constants_land_ahead_of_the_first_reader() {
        // Both injectors (`serve_html` here, `Monitor::build_archive`) prepend
        // their constants to the FIRST `<script>` via `replacen(.., 1)`, so that
        // block must be the earliest one that READS a constant — otherwise the
        // page runs before the values exist, which surfaces as a silent
        // "ARCHIVE_* is not defined" in a browser and nowhere else.
        //
        // The theme init is deliberately first: it reads `ARCHIVE_THEME` and has
        // to run before the body paints, or the page flashes the wrong theme.
        let first = DASHBOARD_HTML.find("<script>").expect("no script block");
        let first_body = &DASHBOARD_HTML[first..];
        let first_body = &first_body[..first_body.find("</script>").unwrap()];
        assert!(
            first_body.contains("ARCHIVE_THEME"),
            "the first script block must be the theme init (it reads ARCHIVE_THEME)",
        );
        assert!(
            first_body.contains("data-theme"),
            "the theme init must set data-theme before the body paints",
        );
        // Balanced tags, so the split in the sibling test cannot run off the end.
        assert_eq!(
            DASHBOARD_HTML.matches("<script>").count(),
            DASHBOARD_HTML.matches("</script>").count(),
        );
    }

    #[test]
    fn the_page_reads_the_record_plane_and_the_epoch_feed() {
        // The portal renders levels from the record plane and keeps the epoch
        // feed as the run clock / fallback level source. Losing either
        // reference silently reduces the page to half a dashboard, which no
        // Rust-side test would otherwise notice.
        for needle in [
            "'/stream?path='",
            "'/history?path='",
            "EventSource('/events')",
            "ARCHIVE_DATA",
            "LIVE_GPU_INIT",
            // The heat-map plane: the archive writes ARCHIVE_HEATMAPS and
            // the live page polls the index route; losing either name
            // silently reduces the graph card to the plain SVG.
            "ARCHIVE_HEATMAPS",
            "'/api/heatmaps'",
        ] {
            assert!(DASHBOARD_HTML.contains(needle), "page lost {needle}");
        }
    }

    /// The page is a single string constant with no build step, so a typo'd
    /// element id or a handler that lost its function fails **silently** in the
    /// browser — no Rust gate would see it. These two are string-checkable, so
    /// they are checked here rather than discovered on a rig run.
    #[test]
    fn the_page_only_reaches_for_elements_and_handlers_it_has() {
        // Every script block counts as "js" and everything else as markup: the
        // page has more than one block (theme init + the page proper), and a
        // handler defined in either satisfies an inline `onclick`.
        let (markup, js) = {
            let mut markup = String::new();
            let mut js = String::new();
            let mut rest = DASHBOARD_HTML;
            while let Some(open) = rest.find("<script>") {
                markup.push_str(&rest[..open]);
                let after = &rest[open + "<script>".len()..];
                let close = after.find("</script>").expect("unclosed <script>");
                js.push_str(&after[..close]);
                rest = &after[close + "</script>".len()..];
            }
            markup.push_str(rest);
            (markup, js)
        };
        let js = js.as_str();

        /// Every `<needle><id><term>` occurrence's id, e.g. `id="foo"`.
        fn ids(hay: &str, needle: &str, term: char) -> Vec<String> {
            hay.match_indices(needle)
                .filter_map(|(i, _)| {
                    let rest = &hay[i + needle.len()..];
                    rest.find(term).map(|e| rest[..e].to_string())
                })
                .collect()
        }

        let declared = ids(&markup, "id=\"", '"');
        for want in ids(js, "getElementById('", '\'') {
            assert!(
                declared.contains(&want),
                "the page calls getElementById('{want}') but declares no such id",
            );
        }
        for handler in
            ids(&markup, "onclick=\"", '(')
                .into_iter()
                .chain(ids(&markup, "onchange=\"", '('))
        {
            assert!(
                js.contains(&format!("function {handler}(")),
                "inline handler {handler}() has no function in the page script",
            );
        }
    }

    #[test]
    fn serve_html_injects_constants_once_ahead_of_the_page() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;
        srv.set_hardware("2x GPU test rig".to_string());

        // Wait for the INJECTED constant, not the bare name: dashboard.html
        // itself carries `typeof LIVE_HARDWARE!=='undefined'`, so the weaker
        // needle is satisfied by every response and the poll returns before
        // `set_hardware` has been picked up. Normally injection wins that
        // race anyway; under coverage instrumentation it does not, which is
        // how a latent flake became a reproducible failure.
        let body = get_until(addr, "/", "const LIVE_HARDWARE=");
        assert_eq!(body.matches("const LIVE_HARDWARE=").count(), 1);
        // Constants first, page code second: the page reads them at load.
        let consts = body.find("const LIVE_HARDWARE=").unwrap();
        let boot = body.find("floDl monitoring portal").unwrap();
        assert!(consts < boot, "constants injected after the page body");
        srv.shutdown();
    }

    // --- portal record plane, over real TCP ---

    /// One-shot GET, body only.
    fn get(addr: SocketAddr, target: &str) -> String {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        s.write_all(format!("GET {target} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
            .unwrap();
        let mut raw = String::new();
        let _ = s.read_to_string(&mut raw);
        raw.split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or(raw)
    }

    /// Poll a GET until its body contains `needle`, so tests never depend on
    /// the message thread's timing. The budget (500 × 20ms = 10s) is sized
    /// for a full parallel suite run, where the message thread can be
    /// starved well past the 2s that once flaked here — the assertion is
    /// deterministic once the needle appears, so a generous cap costs
    /// nothing on the happy path.
    /// CHOOSING `needle` IS THE WHOLE CORRECTNESS OF THIS HELPER. It must be
    /// absent from the endpoint's NOT-READY answer, not merely present in the
    /// ready one. Both flakes this file has had were the same mistake:
    /// `"LIVE_HARDWARE"` also appears in the page's own
    /// `typeof LIVE_HARDWARE!=='undefined'` guard, and `"root"` also appears
    /// in `snapshot()`'s `{"node":null,"path":"root"}` empty answer. In both
    /// cases the loop returned on the first poll and the assertion then failed
    /// on data that had never arrived — which reads like a logic bug, not a
    /// race. Pick something only the populated response can contain.
    fn get_until(addr: SocketAddr, target: &str, needle: &str) -> String {
        for _ in 0..500 {
            let body = get(addr, target);
            if body.contains(needle) {
                return body;
            }
            thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("{target} never contained {needle}");
    }

    /// Open an SSE subscription and return the connected stream.
    fn open_sse(addr: SocketAddr, target: &str) -> TcpStream {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(std::time::Duration::from_millis(600)))
            .unwrap();
        s.write_all(format!("GET {target} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
            .unwrap();
        s
    }

    /// Drain whatever an SSE stream has produced so far.
    fn drain(s: &mut TcpStream) -> String {
        let mut out = String::new();
        let mut buf = [0u8; 8192];
        loop {
            match s.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => out.push_str(&String::from_utf8_lossy(&buf[..n])),
                Err(_) => break, // read timeout = nothing more for now
            }
        }
        out
    }

    fn node_rec(path: &str, tick: u64) -> Value {
        serde_json::json!({ "v": 1, "kind": "node", "path": path, "tick": tick,
                            "metrics": { "loss": 0.5 }, "work": 10.0 })
    }

    fn tree(tick: u64) -> Vec<Value> {
        vec![
            node_rec("root", tick),
            node_rec("root/exa", tick),
            node_rec("root/exa/rank0", tick),
            node_rec("root/pascal", tick),
            node_rec("root/pascal/rank1", tick),
        ]
    }

    /// The invariant behind [`get_until`], made executable.
    ///
    /// Pushes NOTHING, so `/node` necessarily answers with the not-ready
    /// shape. Deterministic: there is no race to lose. It asserts the
    /// needle the wait uses is absent from that body, and — to keep the
    /// reason legible — that the OLD needle was present in it, which is
    /// precisely why the wait was a no-op and the assertion downstream
    /// saw `node: null`.
    #[test]
    fn not_ready_node_body_cannot_match_the_wait_needle() {
        let srv = DashboardServer::start(0).expect("bind");
        let body = get(srv.addr, "/node?path=root");

        assert!(
            !body.contains("\"node\":{"),
            "the wait needle must not match an un-ingested path: {body}"
        );
        assert!(
            body.contains("\"root\""),
            "the old needle DID match this body — that was the bug: {body}"
        );
    }

    #[test]
    fn node_endpoint_serves_one_level_not_the_cluster() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;
        srv.push_records(tree(1));

        // Needle is `"node":{`, NOT `"root"`. `snapshot()` answers a path it
        // has not ingested yet with `{"children":[],"node":null,"path":"root"}`
        // — which already contains `"root"`, so that needle matched the
        // not-ready body and the wait returned immediately, leaving
        // `v["node"]["path"]` as Null. `node` is an object only once the
        // record has landed.
        let body = get_until(addr, "/node?path=root", "\"node\":{");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["node"]["path"], "root");
        let kids: Vec<&str> = v["children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["path"].as_str().unwrap())
            .collect();
        // Direct children only — the ranks are one level deeper.
        assert_eq!(kids, vec!["root/exa", "root/pascal"]);

        // Drilling in re-scopes to that host's own children.
        let body = get(addr, "/node?path=root/exa");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["node"]["path"], "root/exa");
        assert_eq!(v["children"].as_array().unwrap().len(), 1);
        assert_eq!(v["children"][0]["path"], "root/exa/rank0");

        // A percent-encoded path is the same subscription.
        let enc = get(addr, "/node?path=root%2Fexa");
        assert_eq!(enc, body);

        srv.shutdown();
    }

    #[test]
    fn history_endpoint_honors_scope_and_n() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;
        for t in 1..=5 {
            srv.push_records(tree(t));
        }

        let body = get_until(addr, "/history?path=root&n=100", "\"tick\":5");
        let v: Vec<Value> = serde_json::from_str(&body).unwrap();
        // 3 records per tick reach a root viewer (root + 2 hosts), never the
        // ranks — same shape the live stream delivers.
        assert_eq!(v.len(), 15, "{body}");
        assert!(v.iter().all(|r| r["path"] != "root/exa/rank0"));
        // Newest last.
        assert_eq!(v.last().unwrap()["tick"], 5);

        // `n` caps from the newest end.
        let v: Vec<Value> = serde_json::from_str(&get(addr, "/history?path=root&n=2")).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v.last().unwrap()["tick"], 5);

        srv.shutdown();
    }

    #[test]
    fn paths_endpoint_is_the_navigation_index() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;
        srv.push_records(tree(1));
        let body = get_until(addr, "/paths", "root/pascal/rank1");
        let v: Vec<String> = serde_json::from_str(&body).unwrap();
        assert_eq!(
            v,
            vec![
                "root",
                "root/exa",
                "root/exa/rank0",
                "root/pascal",
                "root/pascal/rank1",
            ],
        );
        srv.shutdown();
    }

    #[test]
    fn stream_is_path_scoped_per_subscriber() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;

        let mut at_root = open_sse(addr, "/stream?path=root");
        let mut at_exa = open_sse(addr, "/stream?path=root%2Fexa");
        // Both preambles are empty (nothing pushed yet); drain the headers.
        let _ = drain(&mut at_root);
        let _ = drain(&mut at_exa);

        srv.push_records(tree(1));
        srv.push_records(vec![serde_json::json!({
            "v": 1, "kind": "event", "path": "root/pascal/rank1",
            "class": "rank_lost", "sev": "critical", "detail": "died", "count": 1,
        })]);

        let root_feed = drain(&mut at_root);
        let exa_feed = drain(&mut at_exa);

        // Root viewer: its own + direct children's metrics, NOT the ranks'.
        assert!(root_feed.contains("\"path\":\"root\""), "{root_feed}");
        assert!(root_feed.contains("\"path\":\"root/exa\""));
        assert!(!root_feed.contains("\"path\":\"root/exa/rank0\""));
        // ...but the deep alert reaches it anyway — the deliberate asymmetry.
        assert!(root_feed.contains("rank_lost"), "{root_feed}");

        // Host viewer: its own + its rank's metrics, and nothing from the
        // sibling host (not even that host's alert).
        assert!(
            exa_feed.contains("\"path\":\"root/exa/rank0\""),
            "{exa_feed}"
        );
        assert!(!exa_feed.contains("\"path\":\"root/pascal\""));
        assert!(!exa_feed.contains("rank_lost"));

        srv.shutdown();
    }

    #[test]
    fn stream_preamble_replays_meta_then_history() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;
        srv.push_records(vec![serde_json::json!({
            "v": 1, "kind": "meta", "reductions": { "accuracy": "mean" },
        })]);
        srv.push_records(tree(1));
        srv.push_records(tree(2));
        // Wait for the message thread to have ingested both ticks.
        get_until(addr, "/history?path=root&n=100", "\"tick\":2");

        // A viewer connecting late still gets the reduction declarations and
        // the level's history before any live record.
        let mut late = open_sse(addr, "/stream?path=root");
        let feed = drain(&mut late);
        let meta_at = feed.find("\"kind\":\"meta\"").expect("meta replayed");
        let first_node = feed.find("\"kind\":\"node\"").expect("history replayed");
        assert!(meta_at < first_node, "meta must precede any record");
        assert!(feed.contains("\"tick\":1") && feed.contains("\"tick\":2"));

        srv.shutdown();
    }

    #[test]
    fn the_epoch_feed_is_untouched_by_the_record_plane() {
        // Regression guard: the original whole-run feed must not see records,
        // and a record-plane subscriber must not see epochs.
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;

        let mut epochs = open_sse(addr, "/events");
        let mut records = open_sse(addr, "/stream?path=root");
        let _ = drain(&mut epochs);
        let _ = drain(&mut records);

        srv.push_epoch("{\"epoch\":1}".to_string());
        srv.push_records(tree(1));

        let epoch_feed = drain(&mut epochs);
        let record_feed = drain(&mut records);
        assert!(epoch_feed.contains("event: epoch"), "{epoch_feed}");
        assert!(!epoch_feed.contains("event: record"), "{epoch_feed}");
        assert!(record_feed.contains("event: record"), "{record_feed}");
        assert!(!record_feed.contains("event: epoch"), "{record_feed}");

        srv.shutdown();
    }

    #[test]
    fn shutdown_closes_a_scoped_subscriber_too() {
        let mut srv = DashboardServer::start(0).expect("bind");
        let addr = srv.addr;
        let mut sub = open_sse(addr, "/stream?path=root");
        let _ = drain(&mut sub);
        thread::sleep(std::time::Duration::from_millis(100));
        srv.push_records(tree(1));
        srv.shutdown();

        // The socket must reach EOF, not hang.
        sub.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        let mut buf = [0u8; 512];
        loop {
            match sub.read(&mut buf) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(e) => panic!("scoped SSE socket still open after shutdown: {e}"),
            }
        }
        std::net::TcpListener::bind(addr).expect("port free after shutdown");
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
