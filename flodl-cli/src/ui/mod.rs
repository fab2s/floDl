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
//! ## Reserved: the run ledger (launch slice)
//!
//! When the page learns to launch training, each launch appends one JSON
//! line to `.fdl/ui/runs.jsonl` (project-local, inside the self-ignored
//! `.fdl/`): `{v, ts, farm, argv, exit, dashboard_port, record_log_dir,
//! archive_path}` — invocations plus artifact pointers, the durable
//! index the run-history view reads. Nothing writes it in this slice;
//! the name is reserved here so no other `.fdl/` consumer claims it
//! (`is_farm_dir` already keeps `.fdl/ui/` out of the farm list).

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

/// Job line-buffer cap. A publish gate build is hundreds of lines;
/// anything past this keeps running but stops being recorded (the final
/// exit event always lands).
const JOB_MAX_LINES: usize = 20_000;

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
}

fn handle(mut stream: TcpStream, server: &UiServer) {
    let Some(req) = read_request(&mut stream) else {
        return;
    };
    match respond(&req, server) {
        Reply::Bytes(response) => {
            let _ = stream.write_all(&response);
        }
        Reply::StartJob(argv) => stream_job(stream, server, argv),
        Reply::FollowJob => follow_job(stream, server),
    }
}

/// What a route resolves to: a complete response, or a directive that
/// needs the socket itself (the streaming legs). Every check — Host,
/// token, method, validation — happens before a `Reply` is chosen, so
/// the streaming legs inherit the same gates as everything else.
enum Reply {
    Bytes(Vec<u8>),
    StartJob(Vec<String>),
    FollowJob,
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
        query.insert(k.to_string(), v.to_string());
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
    let want_post = matches!(path, "/api/join-config" | "/api/publish");
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
            Ok(argv) => Reply::StartJob(argv),
            Err(why) => error_json("400 Bad Request", &why).into(),
        },
        // Reconnect road: replay the current/last job from its start
        // and follow it live.
        "/api/jobs/last" => Reply::FollowJob,
        _ => error_json("404 Not Found", "unknown api route").into(),
    }
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

// ── The job slot: one long-running command, streamed and replayable ────

/// The buffer one job accumulates: NDJSON lines, pushed by the reader
/// threads, drained by however many sockets are following.
#[derive(Debug)]
struct JobState {
    lines: Mutex<Vec<String>>,
    done: AtomicBool,
}

impl JobState {
    fn push(&self, line: String) {
        let mut lines = self.lines.lock().expect("job buffer lock");
        // The cap bounds memory, never the run: past it the child keeps
        // going unrecorded and the exit event still lands.
        if lines.len() < JOB_MAX_LINES {
            lines.push(line);
        }
    }

    fn push_final(&self, line: String) {
        self.lines.lock().expect("job buffer lock").push(line);
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
            lines: Mutex::new(Vec::new()),
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
fn stream_job(mut stream: TcpStream, server: &UiServer, argv: Vec<String>) {
    let job = match server.job.try_start() {
        Ok(j) => j,
        Err(why) => {
            let _ = stream.write_all(&error_json("409 Conflict", &why));
            return;
        }
    };
    let mut cmd_line = vec!["fdl".to_string()];
    cmd_line.extend(argv.iter().cloned());
    job.push(serde_json::json!({ "cmd": cmd_line }).to_string());

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
                        job.push(serde_json::json!({ "s": tag, "t": line }).to_string());
                    }
                })
            })
            .collect();
            let job_done = Arc::clone(&job);
            std::thread::spawn(move || {
                // Output first, exit last: join the readers before the
                // exit event so nothing lands after it.
                for r in readers {
                    let _ = r.join();
                }
                let exit = child.wait().ok().and_then(|s| s.code());
                job_done.push_final(serde_json::json!({ "exit": exit }).to_string());
            });
        }
        Err(e) => {
            job.push(
                serde_json::json!({
                    "s": "err",
                    "t": format!("cannot spawn {}: {e}", server.fdl_bin.display()),
                })
                .to_string(),
            );
            job.push_final(serde_json::json!({ "exit": serde_json::Value::Null }).to_string());
        }
    }
    follow(&mut stream, &job);
}

/// `/api/jobs/last`: replay the current or finished job from its first
/// line and follow while it runs.
fn follow_job(mut stream: TcpStream, server: &UiServer) {
    match server.job.last() {
        Some(job) => follow(&mut stream, &job),
        None => {
            let _ = stream.write_all(&error_json("404 Not Found", "no job has run yet"));
        }
    }
}

/// Stream a job's buffer as NDJSON from the start, then poll for new
/// lines until the job is done and drained. A dead socket ends the
/// following, never the job.
fn follow(stream: &mut TcpStream, job: &JobState) {
    let header = "HTTP/1.1 200 OK\r\n\
         Content-Type: application/x-ndjson\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Connection: close\r\n\r\n";
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    let mut sent = 0usize;
    loop {
        // Batch and done-ness read under one lock. The final line is
        // pushed before `done` is stored (release), so seeing `done`
        // here guarantees the exit event is in the batch just taken —
        // this iteration drains everything and can end the stream.
        let (batch, done) = {
            let lines = job.lines.lock().expect("job buffer lock");
            (lines[sent..].to_vec(), job.done.load(Ordering::Acquire))
        };
        if !batch.is_empty() {
            sent += batch.len();
            let mut chunk = batch.join("\n");
            chunk.push('\n');
            if stream.write_all(chunk.as_bytes()).is_err() {
                return;
            }
        }
        if done {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
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
        // The finished job replays identically from /api/jobs/last.
        let local = format!("127.0.0.1:{port}");
        let replay = get(port, "/api/jobs/last", Some(&local), Some(&token));
        assert!(replay.contains("cannot spawn"), "{replay}");
        assert!(replay.contains("\"cmd\""), "{replay}");
        let _ = std::fs::remove_dir_all(&tmp);
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
        first.push_final("{\"exit\":0}".to_string());
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
