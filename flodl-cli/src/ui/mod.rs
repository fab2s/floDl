//! `fdl ui` — the local operations page.
//!
//! One loopback HTTP server, one embedded page, zero dependencies: the
//! browser counterpart of the walk-in CLI surface. Reads (farms,
//! hardware probe, cluster run status, resolved config) and the first
//! actions: the join-config wizard as a preview-then-apply form, and
//! publish with its gate build streamed live.
//!
//! ## The page drives the CLI, it never reimplements it
//!
//! Every panel that runs something spawns `fdl` itself (the running
//! binary, via `current_exe`) with `--json` and returns argv + exit +
//! output verbatim. The page displays the exact command line beside the
//! result, so anything done here is reproducible in a terminal, and the
//! page structurally cannot drift from the CLI. The one exception is
//! pure local reads (the farm list), which call the same function the
//! CLI subcommand calls.
//!
//! ## Security model
//!
//! Binds 127.0.0.1 only; reaching it from another box is an ssh forward,
//! the same trust story as the cluster. On top of the bind, two checks:
//!
//! - **Host header**: every request must carry the loopback host:port it
//!   was served on. A DNS-rebinding page (attacker domain resolving to
//!   127.0.0.1) reaches the socket but carries its own Host, and is
//!   refused before any route runs.
//! - **Session token**: minted from OS entropy at startup, injected into
//!   the served page, required (header `x-fdl-token`) on every `/api/*`
//!   route. A cross-site request can fire blind at loopback but cannot
//!   read the page, so it never holds the token.
//!
//! ## The run ledger
//!
//! Each completed launch appends one JSON line to `.fdl/ui/runs.jsonl`
//! (project-local, inside the self-ignored `.fdl/`; `is_farm_dir`
//! keeps `.fdl/ui/` out of the farm list): `{v, ts, dur_s, farm, argv,
//! exit, port}` — invocations that actually ran, the durable index the
//! history tab reads. A failed *spawn* is not recorded (nothing ran;
//! the stream reports it live), and ledger I/O failures warn without
//! breaking the stream. Artifact pointers (record-log dir, archive
//! path) ride the args themselves; the history tab's disk scan finds
//! the artifacts regardless.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::builtins::UiArgs;
use crate::join_config::{farms_json, fresh_token, validate_label};
use crate::style;

/// The operations page — embedded at compile time, token injected at
/// serve time.
const PAGE_HTML: &str = include_str!("page.html");

/// Request-line + headers cap. Nothing this server accepts needs more,
/// and a client streaming an endless header list is disconnected instead
/// of growing a buffer.
const MAX_REQUEST_BYTES: usize = 16 * 1024;

/// POST body cap — the wizard and publish forms are a handful of short
/// fields.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// A driven value's length cap: nothing a form legitimately sends is
/// longer, and a bound keeps a hostile body from becoming a huge argv.
const MAX_VALUE_LEN: usize = 512;

/// Job line-buffer cap — a drop-OLDEST ring: when a run out-talks it,
/// the head falls away and the tail survives, because the tail is what
/// diagnoses a run (the same call `record_log` makes). The exit event
/// is always the last line by construction.
const JOB_MAX_LINES: usize = 20_000;

/// How long a job stream may go byte-silent before a no-op line goes
/// out. A cold coverage build sits minutes between output lines, and
/// an idle stream is prey to every reaper between here and the
/// browser — the coordinator's own beacon lesson: never go
/// heartbeat-silent while alive.
const STREAM_HEARTBEAT: Duration = Duration::from_secs(15);

/// Per-socket read/write budget. The page's fetches are small; anything
/// slower is a wedged client, not a request. Driven subcommands are NOT
/// under this budget — they run to completion and carry their own
/// network timeouts (`fdl status` gives an endpoint 5s, probe bounds its
/// ssh legs), so a slow probe shows as a slow panel, never a cut socket.
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Run `fdl ui`: bind, print the address, serve until killed.
pub fn run(cli: &UiArgs) -> i32 {
    let cwd = match std::env::current_dir() {
        Ok(c) => c,
        Err(e) => {
            crate::cli_error!("cannot read cwd: {e}");
            return 1;
        }
    };
    let root = crate::config::find_project_config(&cwd)
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or(cwd);
    let fdl_bin = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            crate::cli_error!("cannot resolve this fdl binary: {e}");
            return 1;
        }
    };
    let token = match fresh_token() {
        Ok(t) => t,
        Err(e) => {
            crate::cli_error!("{e}");
            return 1;
        }
    };
    let listener = match TcpListener::bind(("127.0.0.1", cli.port)) {
        Ok(l) => l,
        Err(e) => {
            crate::cli_error!(
                "cannot bind 127.0.0.1:{}: {e} — another server? pick one with --port",
                cli.port,
            );
            return 1;
        }
    };
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(cli.port);
    let server = Arc::new(UiServer {
        root: root.clone(),
        token,
        fdl_bin,
        port,
        job: JobSlot::default(),
        run_target: Mutex::new(None),
    });

    eprintln!("fdl ui — serving {}", root.display());
    eprintln!("  → {}", style::bold(&format!("http://127.0.0.1:{port}/")));
    eprintln!(
        "{}",
        style::dim(&format!(
            "  loopback only; from another box: ssh -L {port}:127.0.0.1:{port} <this-box>. \
             Ctrl+C stops it.",
        )),
    );

    serve(listener, server);
    0
}

/// Accept loop: one short-lived thread per connection. Requests are
/// small and localhost-few; a driven subcommand pins its thread for as
/// long as it runs, which is exactly what the page expects to wait for.
fn serve(listener: TcpListener, server: Arc<UiServer>) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
            let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
            handle(stream, &server);
        });
    }
}

/// Everything one instance serves from.
struct UiServer {
    /// Project root: the directory whose farms, config and runs the
    /// page shows, resolved once at startup like every fdl command.
    root: PathBuf,
    /// The per-session credential injected into the page.
    token: String,
    /// The binary driven for subcommand routes — the running fdl
    /// itself outside tests.
    fdl_bin: PathBuf,
    /// Bound port, for the Host check.
    port: u16,
    /// The single long-running job (a publish gate build). One at a
    /// time on purpose: two concurrent publishes race the manifest
    /// commit point.
    job: JobSlot,
    /// The dashboard slot's proxy target: a loopback PORT, set from the
    /// run tab. Port-only by construction — the host is always
    /// 127.0.0.1, so the proxy cannot be aimed off-box.
    run_target: Mutex<Option<u16>>,
}

fn handle(mut stream: TcpStream, server: &UiServer) {
    let Some(req) = read_request(&mut stream) else {
        return;
    };
    match respond(&req, server) {
        Reply::Bytes(response) => {
            let _ = stream.write_all(&response);
        }
        Reply::StartJob(argv, ledger) => stream_job(stream, server, argv, ledger),
        Reply::FollowJob { from } => follow_job(stream, server, from),
        Reply::Proxy(target) => proxy_dashboard(stream, server, &target),
    }
}

/// What a route resolves to: a complete response, or a directive that
/// needs the socket itself (the streaming legs). Every check — Host,
/// token, method, validation — happens before a `Reply` is chosen, so
/// the streaming legs inherit the same gates as everything else.
enum Reply {
    Bytes(Vec<u8>),
    /// Spawn this argv and stream its output; a launch also carries
    /// the ledger context its completion appends.
    StartJob(Vec<String>, Option<LedgerCtx>),
    FollowJob {
        /// Absolute line index to resume from (`?from=`).
        from: usize,
    },
    /// Forward this request target to the dashboard slot's loopback
    /// port and stream the response back.
    Proxy(String),
}

/// What a launch knows at spawn time; its exit completes the ledger
/// record.
struct LedgerCtx {
    /// `<root>/.fdl/ui/runs.jsonl`.
    path: PathBuf,
    farm: Option<String>,
    /// The dashboard slot's target at launch, best-effort — the run
    /// tab's port if the operator set one.
    port: Option<u16>,
}

impl From<Vec<u8>> for Reply {
    fn from(bytes: Vec<u8>) -> Self {
        Reply::Bytes(bytes)
    }
}

/// One parsed request — only what routing needs.
struct Request {
    method: String,
    /// Path without the query string.
    path: String,
    query: HashMap<String, String>,
    host: Option<String>,
    token: Option<String>,
    /// POST body, empty on GET.
    body: Vec<u8>,
}

/// Read and parse the request line + headers (any body is ignored:
/// every route here is a GET). `None` on anything malformed — a
/// non-HTTP client gets a hangup, not a parse attempt.
fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        if buf.len() >= MAX_REQUEST_BYTES {
            return None;
        }
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let head = String::from_utf8_lossy(&buf);
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?;
    let (path, query_str) = match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    };
    let mut query = HashMap::new();
    for pair in query_str.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        query.insert(percent_decode(k), percent_decode(v));
    }
    let mut host = None;
    let mut token = None;
    let mut content_length = 0usize;
    for line in lines {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim();
            if name.eq_ignore_ascii_case("host") {
                host = Some(value.to_string());
            } else if name.eq_ignore_ascii_case("x-fdl-token") {
                token = Some(value.to_string());
            } else if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().ok()?;
            }
        }
    }
    if content_length > MAX_BODY_BYTES {
        return None;
    }
    // Whatever of the body already arrived sits after the header
    // terminator; read the rest.
    let header_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)?;
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Some(Request {
        method,
        path: path.to_string(),
        query,
        host,
        token,
        body,
    })
}

/// Route a request. Every gate — Host, token, method, validation —
/// runs here, so a streaming directive comes out pre-authorized.
fn respond(req: &Request, server: &UiServer) -> Reply {
    // The Host check guards every route: a DNS-rebound page reaches the
    // socket but cannot present the loopback host it was served on.
    if !host_is_local(req.host.as_deref(), server.port) {
        return error_json("403 Forbidden", "wrong or missing Host header").into();
    }
    match req.path.as_str() {
        "/" => {
            if req.method != "GET" {
                return error_json("405 Method Not Allowed", "GET only").into();
            }
            let page = PAGE_HTML
                .replace("__FDL_TOKEN__", &server.token)
                .replace("__FDL_ROOT__", &server.root.display().to_string());
            http("200 OK", "text/html; charset=utf-8", page.as_bytes()).into()
        }
        // The dashboard slot. `/run` is the page, the rest are the
        // dashboard's own root-relative routes, forwarded verbatim so
        // the embedded page's fetches work unrewritten. Host-gated but
        // token-free BY NECESSITY (an iframe cannot send headers) and
        // by proportion: this is exactly the content the dashboard
        // already serves with no auth at all on its own port. The
        // dashboard's legacy `/api/history` route is deliberately NOT
        // forwarded — dashboard.html never fetches it, and `/api/` is
        // this server's own namespace.
        "/run" | "/events" | "/graph.svg" | "/node" | "/history" | "/paths" | "/stream" => {
            if req.method != "GET" {
                return error_json("405 Method Not Allowed", "GET only").into();
            }
            let upstream_path = if req.path == "/run" { "/" } else { &req.path };
            let query = req
                .query
                .iter()
                .map(|(k, v)| format!("{k}={}", percent_encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            let target = if query.is_empty() {
                upstream_path.to_string()
            } else {
                format!("{upstream_path}?{query}")
            };
            Reply::Proxy(target)
        }
        // Archived dashboards from disk — an iframe src, so Host-gated
        // like the proxy. What it can serve is bounded twice: the path
        // must resolve inside the project root, and the file must look
        // like a run artifact (the same predicate the scan uses).
        "/archive" => {
            if req.method != "GET" {
                return error_json("405 Method Not Allowed", "GET only").into();
            }
            serve_archive(req, server).into()
        }
        path if path.starts_with("/api/") => {
            // Every API route needs the session token: the page holds
            // it, a cross-site request cannot.
            if req.token.as_deref() != Some(server.token.as_str()) {
                return error_json("403 Forbidden", "missing or wrong x-fdl-token header").into();
            }
            api(req, path, server)
        }
        _ => error_json("404 Not Found", "unknown route").into(),
    }
}

fn api(req: &Request, path: &str, server: &UiServer) -> Reply {
    // Method per route: reads are GET, mutations are POST. A mutation
    // arriving as GET is refused even though the token already proved
    // the caller — links and prefetchers must never mutate.
    // (`/api/run-target` legitimately speaks both: GET reads, POST sets.)
    let want_post = matches!(path, "/api/join-config" | "/api/publish" | "/api/launch")
        || (path == "/api/run-target" && req.method == "POST");
    let expected = if want_post { "POST" } else { "GET" };
    if req.method != expected {
        return error_json("405 Method Not Allowed", &format!("{expected} only")).into();
    }
    // An env name arrives from a query string; it must survive the same
    // charset gate as a farm label before it becomes an argument.
    let env = match req.query.get("env").map(String::as_str) {
        None | Some("") => None,
        Some(e) => match validate_label(e) {
            Ok(()) => Some(e),
            Err(why) => return error_json("400 Bad Request", &why).into(),
        },
    };
    match path {
        // Pure local read — the same function `--list --json` prints.
        "/api/farms" => json_ok(&farms_json(&server.root)).into(),
        "/api/farm" => farm_artifacts(req, server).into(),
        "/api/probe" | "/api/status" | "/api/config" => {
            let argv = route_argv(path, env);
            let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
            json_ok(&run_fdl(&server.fdl_bin, &server.root, &refs)).into()
        }
        // The wizard is fast (key mint + file writes), so both its
        // dry-run and its apply answer synchronously with the report.
        "/api/join-config" => match parse_body(&req.body).and_then(|b| join_config_argv(&b)) {
            Ok(argv) => {
                let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
                json_ok(&run_fdl(&server.fdl_bin, &server.root, &refs)).into()
            }
            Err(why) => error_json("400 Bad Request", &why).into(),
        },
        // The publish gate build runs for minutes: the response is a
        // live NDJSON stream of the child's output.
        "/api/publish" => match parse_body(&req.body).and_then(|b| publish_argv(&b)) {
            Ok(argv) => Reply::StartJob(argv, None),
            Err(why) => error_json("400 Bad Request", &why).into(),
        },
        // The configured commands, with their cached `--fdl-schema`
        // when one exists — the launch form's menu and its field source.
        "/api/commands" => commands_route(env, server).into(),
        // Launch a configured command; the stream is the run's output
        // and its completion appends the run ledger.
        "/api/launch" => match parse_body(&req.body).and_then(|b| launch_argv(&b, server)) {
            Ok((argv, farm)) => {
                // Asking a command what it takes is an inspection, not a
                // run: it must not land in the run ledger. Decided from
                // the argv here rather than trusted from the body, so a
                // real launch cannot opt itself out of history.
                let inspecting = argv.iter().any(|a| a == "--help" || a == "-h");
                let ledger = (!inspecting).then(|| LedgerCtx {
                    path: server.root.join(".fdl/ui/runs.jsonl"),
                    farm,
                    port: *server.run_target.lock().expect("run target lock"),
                });
                Reply::StartJob(argv, ledger)
            }
            Err(why) => error_json("400 Bad Request", &why).into(),
        },
        // Reconnect road: replay the current/last job and follow it
        // live — from its start, or from `?from=<index>` for a client
        // resuming a lost transport.
        "/api/jobs/last" => Reply::FollowJob {
            from: req
                .query
                .get("from")
                .and_then(|f| f.parse().ok())
                .unwrap_or(0),
        },
        // The dashboard slot's target: GET = current port + a
        // reachability probe, POST = set (or clear with null).
        "/api/run-target" => run_target_route(req, server).into(),
        // Archived dashboards discovered on disk, newest first.
        "/api/archives" => json_ok(&serde_json::json!({
            "archives": scan_archives(&server.root),
        }))
        .into(),
        // The run ledger (written by the launch slice; empty until it
        // exists is the honest answer, not an error).
        "/api/runs" => {
            json_ok(&serde_json::json!({ "runs": read_runs_ledger(&server.root) })).into()
        }
        _ => error_json("404 Not Found", "unknown api route").into(),
    }
}

fn run_target_route(req: &Request, server: &UiServer) -> Vec<u8> {
    if req.method == "POST" {
        let body = match parse_body(&req.body) {
            Ok(b) => b,
            Err(why) => return error_json("400 Bad Request", &why),
        };
        let port = match body.get("port") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => match v.as_u64().filter(|p| (1..=65535).contains(p)) {
                Some(p) => Some(p as u16),
                None => return error_json("400 Bad Request", "port: expected 1..=65535 or null"),
            },
        };
        *server.run_target.lock().expect("run target lock") = port;
    }
    let port = *server.run_target.lock().expect("run target lock");
    let reachable = port.is_some_and(|p| {
        TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 1], p)),
            Duration::from_millis(300),
        )
        .is_ok()
    });
    json_ok(&serde_json::json!({ "port": port, "reachable": reachable }))
}

/// A farm's shareable artifacts, for the page's copy buttons. The
/// worker yml carries the admission token — same exposure as the
/// terminal render, behind the same token+Host gates as every route.
/// cloud-init is deliberately NOT served: it embeds the private key,
/// and a secret artifact stays a file path everywhere.
fn farm_artifacts(req: &Request, server: &UiServer) -> Vec<u8> {
    let Some(label) = req.query.get("label") else {
        return error_json("400 Bad Request", "missing ?label=");
    };
    if let Err(why) = validate_label(label) {
        return error_json("400 Bad Request", &why);
    }
    let farm_dir = server.root.join(".fdl").join(label);
    if !farm_dir.is_dir() {
        return error_json("404 Not Found", "no such farm dir");
    }
    let read = |name: &str| std::fs::read_to_string(farm_dir.join(name)).ok();
    json_ok(&serde_json::json!({
        "label": label,
        "worker_yml": read("worker.yml"),
        "install_notes": read("install-notes.md"),
        "sshd_conf": read(&format!("sshd-{label}.conf")),
        "cloud_init_path": farm_dir
            .join("cloud-init.yml")
            .is_file()
            .then(|| farm_dir.join("cloud-init.yml").display().to_string()),
    }))
}

/// Parse a POST body as a JSON object (an empty body is an empty
/// object).
fn parse_body(body: &[u8]) -> Result<serde_json::Value, String> {
    if body.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_slice(body).map_err(|e| format!("body is not JSON: {e}"))
}

/// One driven value: bounded, printable, and — unless the caller says
/// otherwise — not flag-shaped. A value opening with `-` would be read
/// by the child's parser as an option, which turns a form field into a
/// flag-injection road; the one place dashes are legal is a publish
/// args tail, which rides behind a standalone `--`.
fn safe_value(name: &str, value: &str, allow_dash: bool) -> Result<(), String> {
    if value.len() > MAX_VALUE_LEN {
        return Err(format!("{name}: value too long"));
    }
    if value.chars().any(|c| c.is_control()) {
        return Err(format!("{name}: control characters are not a value"));
    }
    if !allow_dash && value.starts_with('-') {
        return Err(format!("{name}: a value cannot start with `-`"));
    }
    Ok(())
}

/// A body string field, validated. Absent and empty are both `None`.
fn body_str<'a>(
    body: &'a serde_json::Value,
    key: &str,
    allow_dash: bool,
) -> Result<Option<&'a str>, String> {
    match body.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) if s.is_empty() => Ok(None),
        Some(serde_json::Value::String(s)) => {
            safe_value(key, s, allow_dash)?;
            Ok(Some(s))
        }
        Some(_) => Err(format!("{key}: expected a string")),
    }
}

/// The wizard form's argv. Always `--json` (the page reads the report)
/// and `--yes` (a subprocess has no tty; `--yes` accepts ordinary
/// defaults and still never consents to the authorized_keys install —
/// that stays the explicit `install_key` field, exactly the CLI's own
/// consent rule).
fn join_config_argv(body: &serde_json::Value) -> Result<Vec<String>, String> {
    let Some(label) = body_str(body, "label", false)? else {
        return Err("label: required".to_string());
    };
    validate_label(label)?;
    let mut argv = vec![
        "join-config".to_string(),
        label.to_string(),
        "--json".to_string(),
        "--yes".to_string(),
    ];
    if let Some(door) = body_str(body, "door", false)? {
        if !matches!(door, "a" | "b" | "nologin") {
            return Err("door: one of `a`, `b`, `nologin`".to_string());
        }
        argv.extend(["--door".to_string(), door.to_string()]);
    }
    for (key, flag) in [
        ("controller", "--controller"),
        ("data_path", "--data-path"),
        ("cloud_init_user", "--cloud-init-user"),
    ] {
        if let Some(v) = body_str(body, key, false)? {
            argv.extend([flag.to_string(), v.to_string()]);
        }
    }
    if let Some(share) = body.get("gpu_ram_share").and_then(|v| v.as_f64()) {
        argv.extend(["--gpu-ram-share".to_string(), share.to_string()]);
    }
    for (key, flag) in [
        ("cloud_init", "--cloud-init"),
        ("regen", "--regen"),
        ("install_key", "--install-key"),
        ("dry_run", "--dry-run"),
    ] {
        if body.get(key).and_then(|v| v.as_bool()) == Some(true) {
            argv.push(flag.to_string());
        }
    }
    Ok(argv)
}

/// The publish form's argv. The args tail rides behind a standalone
/// `--` (the manifest's own args), which is also why it is the one
/// place flag-shaped values are legal.
fn publish_argv(body: &serde_json::Value) -> Result<Vec<String>, String> {
    let mut argv = vec!["publish".to_string(), "--json".to_string()];
    if let Some(source) = body_str(body, "source", false)? {
        argv.push(source.to_string());
    }
    for (key, flag) in [("bin", "--bin"), ("cwd", "--cwd"), ("build", "--build")] {
        if let Some(v) = body_str(body, key, false)? {
            argv.extend([flag.to_string(), v.to_string()]);
        }
    }
    if body.get("no_build").and_then(|v| v.as_bool()) == Some(true) {
        argv.push("--no-build".to_string());
    }
    if let Some(args) = body.get("args") {
        let Some(args) = args.as_array() else {
            return Err("args: expected an array of strings".to_string());
        };
        if !args.is_empty() {
            argv.push("--".to_string());
            for a in args {
                let Some(a) = a.as_str() else {
                    return Err("args: expected an array of strings".to_string());
                };
                safe_value("args", a, true)?;
                argv.push(a.to_string());
            }
        }
    }
    Ok(argv)
}

/// The exact argv a route drives — pure, so tests pin it without
/// spawning anything. `--env` (position-independent) rather than the
/// `@env` sigil: the sigil is a human spelling.
fn route_argv(path: &str, env: Option<&str>) -> Vec<String> {
    let mut argv: Vec<String> = Vec::new();
    if let Some(e) = env
        && path != "/api/config"
    {
        argv.extend(["--env".to_string(), e.to_string()]);
    }
    match path {
        "/api/probe" => argv.extend(["probe".to_string(), "--json".to_string()]),
        "/api/status" => argv.extend(["status".to_string(), "--json".to_string()]),
        "/api/config" => {
            // `config show` takes the env as a positional, not via
            // `--env`: it inspects a named overlay rather than running
            // under one.
            argv.extend(["config".to_string(), "show".to_string()]);
            if let Some(e) = env {
                argv.push(e.to_string());
            }
        }
        _ => {}
    }
    argv
}

/// Drive one `fdl` subcommand and report it verbatim: argv (the
/// reproducible command line the page displays), exit code, both
/// streams. Never an error shape — a failed command IS the result.
fn run_fdl(fdl_bin: &Path, root: &Path, args: &[&str]) -> serde_json::Value {
    let out = Command::new(fdl_bin).args(args).current_dir(root).output();
    let mut cmd = vec!["fdl".to_string()];
    cmd.extend(args.iter().map(|s| s.to_string()));
    match out {
        Ok(out) => serde_json::json!({
            "cmd": cmd,
            "exit": out.status.code(),
            "stdout": String::from_utf8_lossy(&out.stdout),
            "stderr": String::from_utf8_lossy(&out.stderr),
        }),
        Err(e) => serde_json::json!({
            "cmd": cmd,
            "exit": serde_json::Value::Null,
            "stdout": "",
            "stderr": format!("cannot spawn {}: {e}", fdl_bin.display()),
        }),
    }
}

// ── Launch: the configured commands, driven with their own schema ──────

/// The `commands:` tree of the merged config (base, or base + the
/// named overlay — a farm's run command usually lives in its overlay),
/// each carrying the schema that command actually resolves to.
///
/// **A schema belongs to a command's own directory, not to the root.**
/// A `path:` command (the convention `./<name>/fdl.yml`, which is how a
/// training vehicle is declared) resolves its surface through
/// `load_command_with_env` — cached `--fdl-schema` output when fresh,
/// the inline `schema:` block otherwise — and that cache lives under
/// `<name>/.fdl/schema-cache/`. Reading a root-level cache instead
/// found nothing for anything, so every command claimed "no schema"
/// while `ddp-bench/.fdl/schema-cache/ddp-bench.json` sat right there.
///
/// The load is deliberately the **cheap** one: a page load must never
/// trigger a cargo compile. A `compile: true` command whose cache is
/// stale therefore reports no schema rather than blocking the page —
/// `fdl <cmd> --refresh-schema` is the (named) way to refill it.
///
/// `kind` rides along so the page can be honest about *why* a form is
/// absent: a `run:` command is a shell line and never grows one.
fn commands_route(env: Option<&str>, server: &UiServer) -> Vec<u8> {
    let Some(base) = crate::config::find_project_config(&server.root) else {
        return json_ok(&serde_json::json!({ "commands": [] }));
    };
    let project = match crate::config::load_project_with_env(&base, env) {
        Ok(p) => p,
        Err(e) => return error_json("400 Bad Request", &format!("config: {e}")),
    };
    let commands: Vec<serde_json::Value> = project
        .commands
        .iter()
        .map(|(name, spec)| {
            let kind = match spec.kind() {
                Ok(crate::config::CommandKind::Run) => "run",
                Ok(crate::config::CommandKind::Path) => "path",
                Ok(crate::config::CommandKind::Preset) => "preset",
                Err(_) => "invalid",
            };
            // Only a path command owns a directory, and therefore a
            // schema. Its own `description:` is better than the parent's
            // stub, so prefer it when the child config carries one.
            let (schema, description) = if kind == "path" {
                let dir = spec.resolve_path(name, &server.root);
                match crate::config::load_command_with_env(&dir, env) {
                    Ok(child) => (
                        child.schema.and_then(|s| serde_json::to_value(s).ok()),
                        child.description.or_else(|| spec.description.clone()),
                    ),
                    Err(_) => (None, spec.description.clone()),
                }
            } else {
                (None, spec.description.clone())
            };
            serde_json::json!({
                "name": name,
                "description": description,
                "cluster": spec.cluster.unwrap_or(false),
                "kind": kind,
                "schema": schema,
            })
        })
        .collect();
    json_ok(&serde_json::json!({ "commands": commands }))
}

/// fdl's own options, which sit BEFORE the command and apply to every
/// one of them (`fdl [options] <command> [command-options]`). They are
/// not in any command's schema — they belong to fdl — so the form has
/// to carry them separately or they are simply unreachable from the
/// page.
///
/// Taken as STRUCTURED fields rather than raw flags, like every other
/// driven form here: the allowlist is then implicit and a body cannot
/// smuggle an arbitrary pre-command argument.
///
/// `--ansi` / `--no-ansi` are deliberately absent: colour already
/// auto-disables off a tty, so a driven command's output is plain and
/// the only thing `--ansi` could add is escape sequences in the log.
fn global_argv(body: &serde_json::Value) -> Result<Vec<String>, String> {
    let mut argv = Vec::new();
    if let Some(v) = body_str(body, "verbosity", true)? {
        if !matches!(v, "-q" | "-v" | "-vv" | "-vvv") {
            return Err("verbosity: one of `-q`, `-v`, `-vv`, `-vvv`".to_string());
        }
        argv.push(v.to_string());
    }
    if let Some(spec) = body_str(body, "gpus", false)? {
        // fdl's own parser is the authority on the spec; this only
        // keeps a value from posing as a flag.
        if spec != "all" && !spec.chars().all(|c| c.is_ascii_digit() || c == ',') {
            return Err("gpus: a device list like `0,1`, or `all`".to_string());
        }
        argv.extend(["--gpus".to_string(), spec.to_string()]);
    }
    for (key, flag) in [
        ("no_append", "--no-append"),
        ("no_prebuild", "--no-prebuild"),
    ] {
        if body.get(key).and_then(|v| v.as_bool()) == Some(true) {
            argv.push(flag.to_string());
        }
    }
    Ok(argv)
}

/// The launch argv: `[globals] [--env <farm>] <command> <args...>`. The
/// command must be declared in the merged config — the launch surface
/// drives the project's own commands, never arbitrary fdl subcommands
/// (those have their own routes and their own consent rules). Args are
/// the command's own flags, so dashes are the point; they are still
/// bounded and control-character-free.
fn launch_argv(
    body: &serde_json::Value,
    server: &UiServer,
) -> Result<(Vec<String>, Option<String>), String> {
    let Some(command) = body_str(body, "command", false)? else {
        return Err("command: required".to_string());
    };
    let env = body_str(body, "env", false)?;
    if let Some(e) = env {
        validate_label(e)?;
    }
    let base = crate::config::find_project_config(&server.root)
        .ok_or("no fdl.yml here — nothing to launch")?;
    let project =
        crate::config::load_project_with_env(&base, env).map_err(|e| format!("config: {e}"))?;
    if !project.commands.contains_key(command) {
        return Err(format!(
            "command `{command}` is not declared in this project's `commands:`{}",
            env.map(|e| format!(" (env `{e}`)")).unwrap_or_default(),
        ));
    }
    // Everything fdl-level comes first, in the order its own usage
    // line states: `fdl [options] <command> [command-options]`.
    let mut argv: Vec<String> = global_argv(body)?;
    if let Some(e) = env {
        argv.extend(["--env".to_string(), e.to_string()]);
    }
    argv.push(command.to_string());
    if let Some(args) = body.get("args") {
        let Some(args) = args.as_array() else {
            return Err("args: expected an array of strings".to_string());
        };
        for a in args {
            let Some(a) = a.as_str() else {
                return Err("args: expected an array of strings".to_string());
            };
            safe_value("args", a, true)?;
            argv.push(a.to_string());
        }
    }
    Ok((argv, env.map(str::to_string)))
}

/// Append one launch record to the run ledger. Failures warn and never
/// break anything (`record_log`'s philosophy: a full disk must not
/// kill a run — nor, here, the stream reporting on it).
fn append_ledger(path: &Path, record: &serde_json::Value) {
    let write = || -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f, "{record}")
    };
    if let Err(e) = write() {
        eprintln!(
            "fdl ui: cannot append the run ledger at {}: {e}",
            path.display(),
        );
    }
}

// ── The job slot: one long-running command, streamed and replayable ────

/// The buffer one job accumulates: NDJSON lines, pushed by the reader
/// threads, drained by however many sockets are following.
#[derive(Debug)]
struct JobState {
    buf: Mutex<JobBuf>,
    done: AtomicBool,
}

/// The drop-oldest ring plus how much of the stream's history it no
/// longer holds — followers use `base` to say what they missed instead
/// of silently starting late.
#[derive(Debug, Default)]
struct JobBuf {
    lines: std::collections::VecDeque<String>,
    /// Absolute stream index of `lines[0]` — the count dropped from
    /// the front so far.
    base: usize,
}

impl JobState {
    /// Buffer one event, stamped with its ABSOLUTE stream index `i` —
    /// what lets a client that lost its transport reconnect with
    /// `?from=` and resume exactly where it died, instead of replaying
    /// or gapping. Synthetic lines a follower writes (heartbeats, gap
    /// markers) are never buffered and carry no index.
    fn push(&self, mut line: serde_json::Value) {
        let mut buf = self.buf.lock().expect("job buffer lock");
        let idx = buf.base + buf.lines.len();
        if let Some(obj) = line.as_object_mut() {
            obj.insert("i".to_string(), idx.into());
        }
        buf.lines.push_back(line.to_string());
        while buf.lines.len() > JOB_MAX_LINES {
            buf.lines.pop_front();
            buf.base += 1;
        }
    }

    fn push_final(&self, line: serde_json::Value) {
        self.push(line);
        self.done.store(true, Ordering::Release);
    }
}

/// At most one job at a time. Two concurrent publishes would race the
/// manifest commit point, so the second caller is told to wait rather
/// than silently queued.
#[derive(Default)]
struct JobSlot {
    current: Mutex<Option<Arc<JobState>>>,
}

impl JobSlot {
    /// Claim the slot, or say what is still running.
    fn try_start(&self) -> Result<Arc<JobState>, String> {
        let mut current = self.current.lock().expect("job slot lock");
        if let Some(job) = current.as_ref()
            && !job.done.load(Ordering::Acquire)
        {
            return Err("a job is already running — follow it at /api/jobs/last".to_string());
        }
        let job = Arc::new(JobState {
            buf: Mutex::new(JobBuf::default()),
            done: AtomicBool::new(false),
        });
        *current = Some(Arc::clone(&job));
        Ok(job)
    }

    fn last(&self) -> Option<Arc<JobState>> {
        self.current.lock().expect("job slot lock").clone()
    }
}

/// Spawn the job's command and stream its buffer to this socket. The
/// child is never killed on client loss: a publish must reach (or
/// cleanly fail before) its manifest commit point regardless of a
/// closed tab, so the readers keep buffering and `/api/jobs/last`
/// replays what the tab missed.
fn stream_job(
    mut stream: TcpStream,
    server: &UiServer,
    argv: Vec<String>,
    ledger: Option<LedgerCtx>,
) {
    let job = match server.job.try_start() {
        Ok(j) => j,
        Err(why) => {
            let _ = stream.write_all(&error_json("409 Conflict", &why));
            return;
        }
    };
    let mut cmd_line = vec!["fdl".to_string()];
    cmd_line.extend(argv.iter().cloned());
    job.push(serde_json::json!({ "cmd": cmd_line }));

    let spawned = Command::new(&server.fdl_bin)
        .args(&argv)
        .current_dir(&server.root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    match spawned {
        Ok(mut child) => {
            let readers: Vec<_> = [
                child
                    .stdout
                    .take()
                    .map(|o| ("out", BufReader::new(Box::new(o) as Box<dyn Read + Send>))),
                child
                    .stderr
                    .take()
                    .map(|e| ("err", BufReader::new(Box::new(e) as Box<dyn Read + Send>))),
            ]
            .into_iter()
            .flatten()
            .map(|(tag, reader)| {
                let job = Arc::clone(&job);
                std::thread::spawn(move || {
                    for line in reader.lines() {
                        let Ok(line) = line else { break };
                        job.push(serde_json::json!({ "s": tag, "t": line }));
                    }
                })
            })
            .collect();
            let job_done = Arc::clone(&job);
            let waiter_cmd = cmd_line.clone();
            let started = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let clock = std::time::Instant::now();
            std::thread::spawn(move || {
                // Output first, exit last: join the readers before the
                // exit event so nothing lands after it.
                for r in readers {
                    let _ = r.join();
                }
                let exit = child.wait().ok().and_then(|s| s.code());
                // A launch's completion is what makes it history: the
                // ledger records invocations that actually ran, with
                // whatever artifact pointers are knowable here.
                if let Some(ctx) = ledger {
                    append_ledger(
                        &ctx.path,
                        &serde_json::json!({
                            "v": 1,
                            "ts": started,
                            "dur_s": clock.elapsed().as_secs(),
                            "farm": ctx.farm,
                            "argv": waiter_cmd,
                            "exit": exit,
                            "port": ctx.port,
                        }),
                    );
                }
                job_done.push_final(serde_json::json!({ "exit": exit }));
            });
        }
        Err(e) => {
            job.push(serde_json::json!({
                "s": "err",
                "t": format!("cannot spawn {}: {e}", server.fdl_bin.display()),
            }));
            job.push_final(serde_json::json!({ "exit": serde_json::Value::Null }));
        }
    }
    follow(&mut stream, &job, 0);
}

/// `/api/jobs/last`: replay the current or finished job from its first
/// line and follow while it runs.
fn follow_job(mut stream: TcpStream, server: &UiServer, from: usize) {
    match server.job.last() {
        Some(job) => follow(&mut stream, &job, from),
        None => {
            let _ = stream.write_all(&error_json("404 Not Found", "no job has run yet"));
        }
    }
}

/// Stream a job's buffer as NDJSON from the start, then poll for new
/// lines until the job is done and drained. A dead socket ends the
/// following, never the job.
fn follow(stream: &mut TcpStream, job: &JobState, from: usize) {
    // Streaming legs drop the write timeout: a reader throttled by its
    // own rendering cost (a coverage run floods tens of thousands of
    // lines — found live, tab in the foreground) or by a backgrounded
    // tab builds backpressure, and a 10s budget then cuts a perfectly
    // healthy stream mid-run. A slow consumer is not a dead one; a
    // dead one still errors the write when TCP gives up.
    let _ = stream.set_write_timeout(None);
    let header = "HTTP/1.1 200 OK\r\n\
         Content-Type: application/x-ndjson\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Connection: close\r\n\r\n";
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    // `sent` is an ABSOLUTE stream position; the ring's `base` says
    // how much history is gone, so a follower that fell behind (or a
    // replay of a run that out-talked the ring) reports the gap
    // instead of silently starting late.
    let mut sent = from;
    let mut last_write = std::time::Instant::now();
    loop {
        // Batch and done-ness read under one lock. The final line is
        // pushed before `done` is stored (release), so seeing `done`
        // here guarantees the exit event is in the batch just taken —
        // this iteration drains everything and can end the stream.
        let (batch, dropped, done) = {
            let buf = job.buf.lock().expect("job buffer lock");
            let dropped = buf.base.saturating_sub(sent);
            let from = sent.max(buf.base) - buf.base;
            let batch: Vec<String> = buf.lines.iter().skip(from).cloned().collect();
            sent = buf.base + buf.lines.len();
            (batch, dropped, job.done.load(Ordering::Acquire))
        };
        if dropped > 0 {
            let gap = serde_json::json!({
                "s": "err",
                "t": format!(
                    "({dropped} earlier lines fell out of the buffer — it keeps \
                     the most recent {JOB_MAX_LINES})",
                ),
            });
            if stream.write_all(format!("{gap}\n").as_bytes()).is_err() {
                return;
            }
        }
        if !batch.is_empty() {
            let mut chunk = batch.join("\n");
            chunk.push('\n');
            if stream.write_all(chunk.as_bytes()).is_err() {
                return;
            }
            last_write = std::time::Instant::now();
        } else if !done && last_write.elapsed() >= STREAM_HEARTBEAT {
            // A no-op object the page ignores: bytes on the wire are
            // what keep an idle stream alive through whatever sits
            // between here and the reader.
            if stream.write_all(b"{}\n").is_err() {
                return;
            }
            last_write = std::time::Instant::now();
        }
        if done {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

// ── The dashboard slot: loopback proxy + archives ───────────────────────

/// Forward one GET to the dashboard's loopback port and relay the raw
/// response until either side closes. No rewriting: the dashboard's
/// routes are forwarded under their own paths, so its root-relative
/// fetches resolve correctly from inside the iframe. The upstream read
/// gets NO timeout on purpose — `/events` and `/stream` are SSE legs
/// that legitimately sit idle between window ticks.
fn proxy_dashboard(mut stream: TcpStream, server: &UiServer, target: &str) {
    // Same rule as `follow`: an SSE leg to a backgrounded tab may
    // legitimately stall past any fixed budget.
    let _ = stream.set_write_timeout(None);
    let Some(port) = *server.run_target.lock().expect("run target lock") else {
        let _ = stream.write_all(&error_json(
            "502 Bad Gateway",
            "no dashboard target set — set the port on the run tab",
        ));
        return;
    };
    let upstream = TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(500),
    );
    let Ok(mut upstream) = upstream else {
        let _ = stream.write_all(&error_json(
            "502 Bad Gateway",
            &format!("nothing answering on 127.0.0.1:{port} — is the run up?"),
        ));
        return;
    };
    if upstream
        .write_all(
            format!("GET {target} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .is_err()
    {
        let _ = stream.write_all(&error_json("502 Bad Gateway", "dashboard hung up"));
        return;
    }
    let mut buf = [0u8; 8192];
    loop {
        match upstream.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                if stream.write_all(&buf[..n]).is_err() {
                    return;
                }
            }
        }
    }
}

/// What counts as a servable run artifact — the single predicate both
/// the scan and `/archive` apply, so the list can never offer a file
/// the endpoint then refuses (or vice versa). Exactly the names the
/// harness writes: `dashboard.html` / `timeline.html`, bare or with a
/// `_<timestamp>` suffix. A dotted stem is refused — rustdoc source
/// pages are named `timeline.rs.html` and must never list.
fn is_archive_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".html") else {
        return false;
    };
    if stem.contains('.') {
        return false;
    }
    ["dashboard", "timeline"].iter().any(|prefix| {
        stem == *prefix
            || stem
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('_'))
    })
}

/// Walk the project for persisted dashboards/timelines, newest first.
/// Bounded: heavy build/vendor trees are skipped, depth is capped, and
/// the result is truncated (newest wins) with the truncation visible in
/// the count.
fn scan_archives(root: &Path) -> Vec<serde_json::Value> {
    // Dot-dirs are skipped wholesale (.git, .fdl, .target-docsrs,
    // .cargo-cache*, ...): no run artifact ever lives in hidden state,
    // and rustdoc trees under them are exactly the false-positive farm.
    // `src` is skipped because a crate's sources are where the
    // dashboard/timeline TEMPLATES live (flodl/src/monitor/), and a
    // template is an empty page, not a run.
    const SKIP: &[&str] = &["target", "libtorch", "node_modules", "_site", "src"];
    const MAX_DEPTH: usize = 6;
    const MAX_RESULTS: usize = 200;
    let mut found: Vec<(std::time::SystemTime, PathBuf, u64)> = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if path.is_dir() {
                if depth < MAX_DEPTH && !name.starts_with('.') && !SKIP.contains(&name) {
                    stack.push((path, depth + 1));
                }
            } else if is_archive_name(name)
                && let Ok(meta) = entry.metadata()
            {
                found.push((
                    meta.modified().unwrap_or(std::time::UNIX_EPOCH),
                    path,
                    meta.len(),
                ));
            }
        }
    }
    found.sort_by_key(|(mtime, _, _)| std::cmp::Reverse(*mtime));
    found.truncate(MAX_RESULTS);
    found
        .into_iter()
        .map(|(mtime, path, size)| {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            serde_json::json!({
                "path": rel,
                "mtime": mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                "size": size,
            })
        })
        .collect()
}

/// Serve one archived dashboard file. The ?path= is project-relative;
/// absolute paths and any `..` component are refused before the
/// filesystem is touched, the resolved file must still live under the
/// project root, and it must pass the same name predicate the scan
/// applies.
fn serve_archive(req: &Request, server: &UiServer) -> Vec<u8> {
    let Some(rel) = req.query.get("path") else {
        return error_json("400 Bad Request", "missing ?path=");
    };
    let rel_path = Path::new(rel);
    if rel_path.is_absolute()
        || rel_path
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return error_json("400 Bad Request", "path: project-relative, no `..`");
    }
    let name = rel_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if !is_archive_name(name) {
        return error_json("400 Bad Request", "path: not a run artifact");
    }
    let Ok(full) = server.root.join(rel_path).canonicalize() else {
        return error_json("404 Not Found", "no such archive");
    };
    let Ok(root) = server.root.canonicalize() else {
        return error_json("404 Not Found", "project root vanished");
    };
    if !full.starts_with(&root) {
        return error_json("400 Bad Request", "path: escapes the project root");
    }
    match std::fs::read(&full) {
        Ok(bytes) => http("200 OK", "text/html; charset=utf-8", &bytes),
        Err(_) => error_json("404 Not Found", "no such archive"),
    }
}

/// The run ledger, if the launch slice has written one yet. Bad lines
/// are skipped, not fatal — an append-only file's tail can be mid-write.
fn read_runs_ledger(root: &Path) -> Vec<serde_json::Value> {
    let Ok(content) = std::fs::read_to_string(root.join(".fdl/ui/runs.jsonl")) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// Re-encode one query value for the upstream request line (the parse
/// decoded it). Conservative: everything but unreserved characters is
/// escaped.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Percent-decode one query component (browsers encode `/`, `@`, `:`
/// in values like archive paths). `+` stays literal — these are path
/// components, not form encoding. Malformed escapes pass through
/// verbatim and fail whatever validation comes next.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (
                bytes.get(i + 1).and_then(|b| (*b as char).to_digit(16)),
                bytes.get(i + 2).and_then(|b| (*b as char).to_digit(16)),
            )
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Both spellings a loopback browser sends, port included — we never
/// serve on 80, so a portless Host is nothing we produced.
fn host_is_local(host: Option<&str>, port: u16) -> bool {
    let Some(host) = host else { return false };
    host == format!("127.0.0.1:{port}") || host == format!("localhost:{port}")
}

fn http(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Connection: close\r\n\r\n",
        body.len(),
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

fn json_ok(value: &serde_json::Value) -> Vec<u8> {
    http(
        "200 OK",
        "application/json",
        serde_json::to_string(value)
            .expect("api payloads serialize")
            .as_bytes(),
    )
}

fn error_json(status: &str, message: &str) -> Vec<u8> {
    http(
        status,
        "application/json",
        serde_json::json!({ "error": message })
            .to_string()
            .as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "fdl-ui-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A live server on an ephemeral port. The accept thread is leaked
    /// on purpose: it parks on a loopback listener the test binary
    /// tears down at exit.
    fn spawn_server(root: &Path) -> (u16, String) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = Arc::new(UiServer {
            root: root.to_path_buf(),
            token: "tok-test".to_string(),
            // Subprocess routes are exercised by argv unit tests + the
            // live smoke, never by tests execing something (an exec'd
            // route in these tests would hit the nonexistent binary and
            // still answer with a spawn-failure result, which is itself
            // asserted below).
            fdl_bin: PathBuf::from("fdl-never-spawned"),
            port,
            job: JobSlot::default(),
            run_target: Mutex::new(None),
        });
        std::thread::spawn(move || serve(listener, server));
        (port, "tok-test".to_string())
    }

    fn get(port: u16, target: &str, host: Option<&str>, token: Option<&str>) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let host_line = host.map(|h| format!("Host: {h}\r\n")).unwrap_or_default();
        let token_line = token
            .map(|t| format!("x-fdl-token: {t}\r\n"))
            .unwrap_or_default();
        s.write_all(format!("GET {target} HTTP/1.1\r\n{host_line}{token_line}\r\n").as_bytes())
            .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn the_page_carries_the_token_and_only_behind_the_host_check() {
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");

        // The served page embeds the session token and the root.
        let page = get(port, "/", Some(&local), None);
        assert!(page.starts_with("HTTP/1.1 200"), "{page}");
        assert!(page.contains(&token));
        assert!(page.contains(tmp.display().to_string().as_str()));

        // A DNS-rebound page carries its own Host and never sees it.
        let rebound = get(port, "/", Some("evil.example:80"), None);
        assert!(rebound.starts_with("HTTP/1.1 403"), "{rebound}");
        assert!(!rebound.contains(&token));
        // No Host at all is the same refusal.
        assert!(get(port, "/", None, None).starts_with("HTTP/1.1 403"));
        // localhost spelling is a loopback browser too.
        let localhost = format!("localhost:{port}");
        assert!(get(port, "/", Some(&localhost), None).starts_with("HTTP/1.1 200"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn api_routes_need_the_token_and_serve_the_farm_list() {
        let tmp = tempdir();
        std::fs::write(tmp.join("fdl.yml"), "# base\n").unwrap();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");

        // No token → refused; the error names the header.
        let bare = get(port, "/api/farms", Some(&local), None);
        assert!(bare.starts_with("HTTP/1.1 403"), "{bare}");
        assert!(bare.contains("x-fdl-token"));

        // With it → the same payload `--list --json` prints.
        let farms = get(port, "/api/farms", Some(&local), Some(&token));
        assert!(farms.starts_with("HTTP/1.1 200"), "{farms}");
        assert!(farms.contains("\"farms\""), "{farms}");
        assert!(farms.contains("\"other_envs\""), "{farms}");

        // Unknown routes and bad env names are named refusals.
        assert!(get(port, "/api/nope", Some(&local), Some(&token)).starts_with("HTTP/1.1 404"),);
        let bad = get(port, "/api/status?env=no/slash", Some(&local), Some(&token));
        assert!(bad.starts_with("HTTP/1.1 400"), "{bad}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn route_argv_is_the_exact_command_line() {
        assert_eq!(route_argv("/api/probe", None), vec!["probe", "--json"]);
        assert_eq!(
            route_argv("/api/probe", Some("rig")),
            vec!["--env", "rig", "probe", "--json"],
        );
        assert_eq!(route_argv("/api/status", None), vec!["status", "--json"]);
        assert_eq!(
            route_argv("/api/status", Some("b300")),
            vec!["--env", "b300", "status", "--json"],
        );
        // config show takes the env as its positional, not via --env.
        assert_eq!(route_argv("/api/config", None), vec!["config", "show"]);
        assert_eq!(
            route_argv("/api/config", Some("b300")),
            vec!["config", "show", "b300"],
        );
    }

    #[test]
    fn a_failed_spawn_is_a_result_not_an_error_shape() {
        let tmp = tempdir();
        let v = run_fdl(Path::new("fdl-that-does-not-exist"), &tmp, &["probe"]);
        assert_eq!(v["cmd"], serde_json::json!(["fdl", "probe"]));
        assert!(v["exit"].is_null());
        assert!(
            v["stderr"].as_str().unwrap().contains("cannot spawn"),
            "{v}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn post(port: u16, target: &str, token: Option<&str>, body: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let token_line = token
            .map(|t| format!("x-fdl-token: {t}\r\n"))
            .unwrap_or_default();
        s.write_all(
            format!(
                "POST {target} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{token_line}\
                 Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len(),
            )
            .as_bytes(),
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    // ── PR3: argv builders are the security boundary ────────────────

    #[test]
    fn the_wizard_argv_is_exact_and_flag_injection_proof() {
        let body = serde_json::json!({
            "label": "b300",
            "door": "a",
            "controller": "op@ctrl.example:2222",
            "data_path": "/flodl/data",
            "gpu_ram_share": 0.5,
            "cloud_init": true,
            "cloud_init_user": "root",
            "regen": true,
            "install_key": true,
            "dry_run": true,
        });
        assert_eq!(
            join_config_argv(&body).unwrap(),
            vec![
                "join-config",
                "b300",
                "--json",
                "--yes",
                "--door",
                "a",
                "--controller",
                "op@ctrl.example:2222",
                "--data-path",
                "/flodl/data",
                "--cloud-init-user",
                "root",
                "--gpu-ram-share",
                "0.5",
                "--cloud-init",
                "--regen",
                "--install-key",
                "--dry-run",
            ],
        );
        // Minimal body: label only, consent flags absent.
        assert_eq!(
            join_config_argv(&serde_json::json!({"label": "x"})).unwrap(),
            vec!["join-config", "x", "--json", "--yes"],
        );
        // No label is a named refusal, not a guess.
        assert!(
            join_config_argv(&serde_json::json!({}))
                .unwrap_err()
                .contains("label")
        );
        // A flag-shaped label would be parsed as an option by the
        // child; refused before it becomes argv.
        let err = join_config_argv(&serde_json::json!({"label": "--regen"})).unwrap_err();
        assert!(err.contains('-'), "got: {err}");
        // Same for any value field.
        assert!(
            join_config_argv(&serde_json::json!({"label": "x", "controller": "--install-key"}))
                .is_err(),
        );
        // An unknown door is refused by name.
        assert!(
            join_config_argv(&serde_json::json!({"label": "x", "door": "c"}))
                .unwrap_err()
                .contains("door"),
        );
    }

    #[test]
    fn the_publish_argv_keeps_dashes_behind_the_separator() {
        let body = serde_json::json!({
            "source": "file:///abs/tree",
            "bin": "target/release/train",
            "args": ["--model", "lenet", "--epochs", "2"],
        });
        assert_eq!(
            publish_argv(&body).unwrap(),
            vec![
                "publish",
                "--json",
                "file:///abs/tree",
                "--bin",
                "target/release/train",
                "--",
                "--model",
                "lenet",
                "--epochs",
                "2",
            ],
        );
        // Empty body publishes the standing fdl.yml publish: block.
        assert_eq!(
            publish_argv(&serde_json::json!({})).unwrap(),
            vec!["publish", "--json"]
        );
        // No args → no separator.
        assert!(
            !publish_argv(&serde_json::json!({"args": []}))
                .unwrap()
                .contains(&"--".to_string())
        );
        // Flag-shaped SOURCE is an injection attempt, not a spec.
        assert!(publish_argv(&serde_json::json!({"source": "--no-build"})).is_err());
    }

    // ── PR3: routes over real TCP ───────────────────────────────────

    #[test]
    fn the_wizard_post_flows_through_to_the_runner() {
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        // Mutations refuse GET even with the token.
        let local = format!("127.0.0.1:{port}");
        let as_get = get(port, "/api/join-config", Some(&local), Some(&token));
        assert!(as_get.starts_with("HTTP/1.1 405"), "{as_get}");
        // No token → refused before the body is even considered.
        assert!(post(port, "/api/join-config", None, "{}").starts_with("HTTP/1.1 403"));
        // A bad body is a named 400.
        let bad = post(port, "/api/join-config", Some(&token), "not json");
        assert!(bad.starts_with("HTTP/1.1 400"), "{bad}");
        let no_label = post(port, "/api/join-config", Some(&token), "{}");
        assert!(no_label.starts_with("HTTP/1.1 400"), "{no_label}");
        // A good body reaches the runner: the test binary's fake fdl
        // cannot spawn, and that surfaces as a RESULT carrying the
        // exact argv the page would display.
        let ok = post(
            port,
            "/api/join-config",
            Some(&token),
            r#"{"label":"smoke","dry_run":true}"#,
        );
        assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
        assert!(ok.contains("cannot spawn"), "{ok}");
        assert!(ok.contains("--dry-run"), "{ok}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_publish_post_streams_ndjson_and_the_slot_replays_it() {
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        let streamed = post(port, "/api/publish", Some(&token), "{}");
        assert!(streamed.starts_with("HTTP/1.1 200"), "{streamed}");
        assert!(streamed.contains("application/x-ndjson"), "{streamed}");
        // First event: the command line. Then the spawn failure and the
        // exit event, in order.
        let body = streamed.split("\r\n\r\n").nth(1).unwrap();
        let events: Vec<serde_json::Value> = body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            events[0]["cmd"],
            serde_json::json!(["fdl", "publish", "--json"])
        );
        assert!(
            events[1]["t"].as_str().unwrap().contains("cannot spawn"),
            "{events:?}",
        );
        assert!(events.last().unwrap()["exit"].is_null(), "{events:?}");
        // The finished job replays identically from /api/jobs/last...
        let local = format!("127.0.0.1:{port}");
        let replay = get(port, "/api/jobs/last", Some(&local), Some(&token));
        assert!(replay.contains("cannot spawn"), "{replay}");
        assert!(replay.contains("\"cmd\""), "{replay}");
        // ...and `?from=` resumes mid-stream: a client that lost its
        // transport at line 1 gets everything from there on, and
        // nothing it already has.
        let resumed = get(port, "/api/jobs/last?from=2", Some(&local), Some(&token));
        assert!(!resumed.contains("\"cmd\""), "{resumed}");
        assert!(resumed.contains("\"exit\""), "{resumed}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_job_buffer_is_a_drop_oldest_ring_and_the_exit_survives() {
        let slot = JobSlot::default();
        let job = slot.try_start().unwrap();
        for i in 0..(JOB_MAX_LINES + 50) {
            job.push(serde_json::json!({ "n": i }));
        }
        job.push_final(serde_json::json!({ "exit": 0 }));
        let buf = job.buf.lock().unwrap();
        // The tail survives, the head fell away, and the record of how
        // much fell away is exact.
        assert_eq!(buf.lines.len(), JOB_MAX_LINES);
        assert_eq!(buf.base, 51);
        assert!(buf.lines.back().unwrap().contains("\"exit\":0"));
        // Every buffered line carries its absolute index — the resume
        // cursor a reconnecting client hands back as ?from=.
        let front: serde_json::Value = serde_json::from_str(buf.lines.front().unwrap()).unwrap();
        assert_eq!(front["n"], 51);
        assert_eq!(front["i"], 51);
        let back: serde_json::Value = serde_json::from_str(buf.lines.back().unwrap()).unwrap();
        assert_eq!(back["i"], JOB_MAX_LINES + 50);
    }

    #[test]
    fn the_job_slot_admits_one_running_job() {
        let slot = JobSlot::default();
        assert!(slot.last().is_none());
        let first = slot.try_start().unwrap();
        // While it runs, the slot is taken and says where to look.
        let refused = slot.try_start().unwrap_err();
        assert!(refused.contains("/api/jobs/last"), "{refused}");
        // Finished → the slot frees, and the old buffer stays readable.
        first.push_final(serde_json::json!({ "exit": 0 }));
        assert!(slot.try_start().is_ok());
        let _ = first;
    }

    #[test]
    fn farm_artifacts_serve_the_copyable_files_and_never_cloud_init() {
        let tmp = tempdir();
        let farm = tmp.join(".fdl").join("f1");
        std::fs::create_dir_all(&farm).unwrap();
        std::fs::write(farm.join("worker.yml"), "join:\n  token: sekrit\n").unwrap();
        std::fs::write(farm.join("cloud-init.yml"), "SECRET KEY MATERIAL").unwrap();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");

        let ok = get(port, "/api/farm?label=f1", Some(&local), Some(&token));
        assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
        assert!(ok.contains("token: sekrit"), "{ok}");
        // cloud-init is reported as a path, its content never served.
        assert!(ok.contains("cloud-init.yml"), "{ok}");
        assert!(!ok.contains("SECRET KEY MATERIAL"), "{ok}");

        assert!(
            get(port, "/api/farm?label=no/pe", Some(&local), Some(&token))
                .starts_with("HTTP/1.1 400"),
        );
        assert!(
            get(port, "/api/farm?label=absent", Some(&local), Some(&token))
                .starts_with("HTTP/1.1 404"),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── PR4: the dashboard slot ─────────────────────────────────────

    #[test]
    fn query_components_percent_decode_and_reencode() {
        assert_eq!(percent_decode("a%2Fb%40c"), "a/b@c");
        assert_eq!(percent_decode("plain"), "plain");
        // Malformed escapes pass through and fail later validation.
        assert_eq!(percent_decode("%zz"), "%zz");
        assert_eq!(percent_encode("root/exa cuda"), "root/exa%20cuda");
        // A decoded slash in an env name is still refused by the label
        // gate — decoding must not widen what validation sees.
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");
        let bad = get(
            port,
            "/api/status?env=no%2Fslash",
            Some(&local),
            Some(&token),
        );
        assert!(bad.starts_with("HTTP/1.1 400"), "{bad}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_run_target_is_a_loopback_port_or_nothing() {
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");
        // Unset by default, and honestly unreachable.
        let out = get(port, "/api/run-target", Some(&local), Some(&token));
        assert!(out.contains("\"port\":null"), "{out}");
        assert!(out.contains("\"reachable\":false"), "{out}");
        // A nonsense port is refused by name.
        let bad = post(port, "/api/run-target", Some(&token), r#"{"port":0}"#);
        assert!(bad.starts_with("HTTP/1.1 400"), "{bad}");
        // Setting reflects back; clearing with null unsets.
        let set = post(port, "/api/run-target", Some(&token), r#"{"port":8099}"#);
        assert!(set.contains("\"port\":8099"), "{set}");
        let cleared = post(port, "/api/run-target", Some(&token), r#"{"port":null}"#);
        assert!(cleared.contains("\"port\":null"), "{cleared}");
        // With no target, the slot answers 502, not a hang.
        let run = get(port, "/run", Some(&local), None);
        assert!(run.starts_with("HTTP/1.1 502"), "{run}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_proxy_forwards_dashboard_routes_verbatim_and_relays_bytes() {
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");

        // A fake dashboard: answers two connections, recording the
        // request lines it saw.
        let upstream = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let up_port = upstream.local_addr().unwrap().port();
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_up = Arc::clone(&seen);
        std::thread::spawn(move || {
            // Serves until the test binary exits; the run-target
            // reachability probe connects and immediately hangs up, so
            // an empty read is skipped, not recorded.
            for conn in upstream.incoming() {
                let Ok(mut s) = conn else { break };
                let mut buf = [0u8; 2048];
                let Ok(n) = s.read(&mut buf) else { continue };
                if n == 0 {
                    continue;
                }
                let head = String::from_utf8_lossy(&buf[..n]).to_string();
                seen_up
                    .lock()
                    .unwrap()
                    .push(head.lines().next().unwrap_or("").to_string());
                let _ = s.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\nDASH-BODY",
                );
            }
        });

        let set = post(
            port,
            "/api/run-target",
            Some(&token),
            &format!(r#"{{"port":{up_port}}}"#),
        );
        assert!(set.contains("\"reachable\":true"), "{set}");

        // `/run` lands on the dashboard's root, response relayed raw.
        let page = get(port, "/run", Some(&local), None);
        assert!(page.ends_with("DASH-BODY"), "{page}");
        // A scoped route keeps its path and its (re-encoded) query.
        let node = get(port, "/node?path=root%2Fexa", Some(&local), None);
        assert!(node.ends_with("DASH-BODY"), "{node}");
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0], "GET / HTTP/1.1");
        // `/` is legal in a query and the dashboard resolves encoded
        // and raw ?path= to the same scope by design, so the decoded
        // spelling forwards as-is.
        assert_eq!(seen[1], "GET /node?path=root/exa HTTP/1.1");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn archives_are_scanned_and_served_inside_the_root_only() {
        let tmp = tempdir();
        let run_dir = tmp.join("ddp-bench/runs/mlp/nccl-sync");
        std::fs::create_dir_all(&run_dir).unwrap();
        std::fs::write(run_dir.join("dashboard.html"), "<html>ARCHIVE</html>").unwrap();
        std::fs::write(run_dir.join("timeline_1.html"), "<html>TL</html>").unwrap();
        std::fs::write(run_dir.join("notes.html"), "not a run artifact").unwrap();
        // Heavy trees are never walked, dot-dirs wholesale — and a
        // rustdoc source page named `timeline.rs.html` is not a run
        // artifact (both found live on the real repo before the
        // predicate learned to refuse them).
        let hidden = tmp.join("target/debug");
        std::fs::create_dir_all(&hidden).unwrap();
        std::fs::write(hidden.join("dashboard.html"), "build junk").unwrap();
        let docsrs = tmp.join(".target-docsrs/doc");
        std::fs::create_dir_all(&docsrs).unwrap();
        std::fs::write(docsrs.join("dashboard.html"), "rustdoc junk").unwrap();
        std::fs::write(run_dir.join("timeline.rs.html"), "rustdoc source page").unwrap();
        // The templates themselves live under a crate's src/ — an
        // empty page is not a run (found live: flodl/src/monitor).
        let templates = tmp.join("flodl/src/monitor");
        std::fs::create_dir_all(&templates).unwrap();
        std::fs::write(templates.join("dashboard.html"), "the template").unwrap();

        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");
        let list = get(port, "/api/archives", Some(&local), Some(&token));
        assert!(list.contains("dashboard.html"), "{list}");
        assert!(list.contains("timeline_1.html"), "{list}");
        assert!(!list.contains("notes.html"), "{list}");
        assert!(!list.contains("target"), "{list}");
        assert!(!list.contains("timeline.rs.html"), "{list}");
        assert!(!list.contains("monitor"), "{list}");

        // Served by relative path (encoded slashes, like a browser).
        let path = "ddp-bench%2Fruns%2Fmlp%2Fnccl-sync%2Fdashboard.html";
        let served = get(port, &format!("/archive?path={path}"), Some(&local), None);
        assert!(served.ends_with("<html>ARCHIVE</html>"), "{served}");

        // The two refusal classes: not-an-artifact, and escape attempts.
        let bad_name = get(
            port,
            "/archive?path=ddp-bench%2Fruns%2Fmlp%2Fnccl-sync%2Fnotes.html",
            Some(&local),
            None,
        );
        assert!(bad_name.starts_with("HTTP/1.1 400"), "{bad_name}");
        for escape in [
            "..%2Fdashboard.html",
            "%2Fetc%2Fpasswd",
            "a%2F..%2F..%2Fdashboard.html",
        ] {
            let out = get(port, &format!("/archive?path={escape}"), Some(&local), None);
            assert!(out.starts_with("HTTP/1.1 400"), "{escape}: {out}");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_runs_ledger_reads_when_present_and_answers_empty_when_not() {
        let tmp = tempdir();
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");
        let empty = get(port, "/api/runs", Some(&local), Some(&token));
        assert!(empty.contains("\"runs\":[]"), "{empty}");

        std::fs::create_dir_all(tmp.join(".fdl/ui")).unwrap();
        std::fs::write(
            tmp.join(".fdl/ui/runs.jsonl"),
            "{\"v\":1,\"farm\":\"rig\",\"exit\":0}\nnot json — a torn tail line\n{\"v\":1,\"farm\":\"b300\"}\n",
        )
        .unwrap();
        let runs = get(port, "/api/runs", Some(&local), Some(&token));
        assert!(runs.contains("\"farm\":\"rig\""), "{runs}");
        assert!(runs.contains("\"farm\":\"b300\""), "{runs}");
        assert!(!runs.contains("torn tail"), "{runs}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── PR5: launch ─────────────────────────────────────────────────

    /// A minimal project with one declared command and a cached schema
    /// for it — the launch surface's whole world.
    fn stage_launch_project(tmp: &Path) {
        std::fs::write(
            tmp.join("fdl.yml"),
            "commands:\n  train:\n  inline:\n  bare:\n    run: echo bare\n",
        )
        .unwrap();
        // Cached `--fdl-schema` output, in the command's OWN dir — the
        // real layout. The first cut of this fixture put the cache at
        // the project root, mirroring the bug it was meant to catch, so
        // it passed while every real command reported "no schema".
        let train = tmp.join("train");
        std::fs::create_dir_all(train.join(".fdl/schema-cache")).unwrap();
        std::fs::write(
            train.join("fdl.yml"),
            "description: the training run\nentry: cargo run --release\n",
        )
        .unwrap();
        std::fs::write(
            train.join(".fdl/schema-cache/train.json"),
            r#"{"args":[],"options":{"epochs":{"type":"int","default":10,"description":"how long"},"model":{"type":"string","choices":["lenet","resnet"]}}}"#,
        )
        .unwrap();
        // The other road to a schema: an inline `schema:` block. No
        // `entry:`, so nothing is probed or spawned by the load.
        let inline = tmp.join("inline");
        std::fs::create_dir_all(&inline).unwrap();
        std::fs::write(
            inline.join("fdl.yml"),
            "schema:\n  options:\n    alpha:\n      type: float\n",
        )
        .unwrap();
    }

    #[test]
    fn the_commands_route_serves_the_menu_with_cached_schemas() {
        let tmp = tempdir();
        stage_launch_project(&tmp);
        let (port, token) = spawn_server(&tmp);
        let local = format!("127.0.0.1:{port}");
        let out = get(port, "/api/commands", Some(&local), Some(&token));
        assert!(out.starts_with("HTTP/1.1 200"), "{out}");
        let body: serde_json::Value =
            serde_json::from_str(out.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        let cmds = body["commands"].as_array().unwrap();
        assert_eq!(cmds.len(), 3);
        // A path command's schema comes from ITS OWN dir's cache, and
        // its own `description:` beats the parent's silence.
        let train = cmds.iter().find(|c| c["name"] == "train").unwrap();
        assert_eq!(train["kind"], "path");
        assert_eq!(train["description"], "the training run");
        assert_eq!(train["schema"]["options"]["epochs"]["type"], "int");
        assert_eq!(train["schema"]["options"]["model"]["choices"][0], "lenet");
        // The inline-`schema:` road resolves too — the old root-cache
        // lookup missed both.
        let inline = cmds.iter().find(|c| c["name"] == "inline").unwrap();
        assert_eq!(inline["schema"]["options"]["alpha"]["type"], "float");
        // A shell command has no schema by nature; the page says so in
        // its own words rather than pointing at --refresh-schema.
        let bare = cmds.iter().find(|c| c["name"] == "bare").unwrap();
        assert_eq!(bare["kind"], "run");
        assert!(bare["schema"].is_null());
        // An unknown env is a loud 400, not an empty menu.
        assert!(
            get(port, "/api/commands?env=ghost", Some(&local), Some(&token))
                .starts_with("HTTP/1.1 400"),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_launch_drives_only_declared_commands() {
        let tmp = tempdir();
        stage_launch_project(&tmp);
        let (port, token) = spawn_server(&tmp);

        // A declared command flows to the runner with its args; the
        // fake binary cannot spawn, and that is a streamed result
        // carrying the exact argv (and NO ledger line — nothing ran).
        let out = post(
            port,
            "/api/launch",
            Some(&token),
            r#"{"command":"train","args":["--epochs","2","--model","lenet"]}"#,
        );
        assert!(out.starts_with("HTTP/1.1 200"), "{out}");
        assert!(
            out.contains(r#"["fdl","train","--epochs","2","--model","lenet"]"#),
            "{out}",
        );
        assert!(out.contains("cannot spawn"), "{out}");
        assert!(
            !tmp.join(".fdl/ui/runs.jsonl").exists(),
            "a failed spawn must not enter the ledger",
        );

        // Undeclared commands are refused by name — the launch surface
        // drives the project's own commands, never arbitrary fdl
        // subcommands.
        let refused = post(
            port,
            "/api/launch",
            Some(&token),
            r#"{"command":"join-config"}"#,
        );
        assert!(refused.starts_with("HTTP/1.1 400"), "{refused}");
        assert!(refused.contains("not declared"), "{refused}");
        // Bad env label and missing command: named refusals.
        assert!(
            post(
                port,
                "/api/launch",
                Some(&token),
                r#"{"command":"train","env":"a/b"}"#
            )
            .starts_with("HTTP/1.1 400"),
        );
        assert!(post(port, "/api/launch", Some(&token), "{}").starts_with("HTTP/1.1 400"),);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// fdl's own options precede the command, come from structured
    /// fields (so no body can smuggle a pre-command argument), and are
    /// validated by name.
    #[test]
    fn global_options_lead_the_argv_and_are_allowlisted() {
        let tmp = tempdir();
        stage_launch_project(&tmp);
        let (port, token) = spawn_server(&tmp);

        let out = post(
            port,
            "/api/launch",
            Some(&token),
            r#"{"command":"train","verbosity":"-vv","gpus":"0,1","no_prebuild":true,
                "no_append":true,"args":["--epochs","2"]}"#,
        );
        assert!(
            out.contains(
                r#"["fdl","-vv","--gpus","0,1","--no-append","--no-prebuild","train","--epochs","2"]"#
            ),
            "{out}",
        );
        // Only the four real verbosity spellings, and a gpu spec that
        // cannot pose as a flag.
        for bad in [
            r#"{"command":"train","verbosity":"-vvvv"}"#,
            r#"{"command":"train","verbosity":"--wat"}"#,
            r#"{"command":"train","gpus":"--no-build"}"#,
            r#"{"command":"train","gpus":"0;rm"}"#,
        ] {
            let r = post(port, "/api/launch", Some(&token), bad);
            assert!(r.starts_with("HTTP/1.1 400"), "{bad} → {r}");
        }
        // `all` is a legal spec.
        assert!(
            post(
                port,
                "/api/launch",
                Some(&token),
                r#"{"command":"train","gpus":"all"}"#
            )
            .contains(r#""--gpus","all""#),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Asking what a command takes is an inspection: it streams like a
    /// launch but never enters the run ledger, and the rule is decided
    /// from the argv rather than trusted from the body.
    #[test]
    fn a_help_invocation_streams_but_is_not_history() {
        let tmp = tempdir();
        stage_launch_project(&tmp);
        let (port, token) = spawn_server(&tmp);
        let out = post(
            port,
            "/api/launch",
            Some(&token),
            r#"{"command":"train","args":["--help"]}"#,
        );
        assert!(out.contains(r#"["fdl","train","--help"]"#), "{out}");
        assert!(
            !tmp.join(".fdl/ui/runs.jsonl").exists(),
            "a --help inspection must not enter the run ledger",
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn the_ledger_appends_and_survives_its_own_dir_being_absent() {
        let tmp = tempdir();
        let path = tmp.join(".fdl/ui/runs.jsonl");
        append_ledger(
            &path,
            &serde_json::json!({"v":1,"ts":1,"farm":"rig","argv":["fdl","train"],"exit":0}),
        );
        append_ledger(
            &path,
            &serde_json::json!({"v":1,"ts":2,"farm":null,"exit":1}),
        );
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content.lines().count(), 2);
        // Exactly what /api/runs reads back.
        let runs = read_runs_ledger(&tmp);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0]["farm"], "rig");
        assert_eq!(runs[1]["exit"], 1);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn oversized_and_malformed_requests_hang_up() {
        let tmp = tempdir();
        let (port, _) = spawn_server(&tmp);
        // A header stream past the cap is dropped without a response.
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.write_all(b"GET / HTTP/1.1\r\n").unwrap();
        let filler = format!("X-Filler: {}\r\n", "a".repeat(1000));
        for _ in 0..20 {
            if s.write_all(filler.as_bytes()).is_err() {
                break; // server already hung up mid-stream — the point
            }
        }
        let mut out = String::new();
        let _ = s.read_to_string(&mut out);
        assert!(out.is_empty(), "expected a hangup, got: {out}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
