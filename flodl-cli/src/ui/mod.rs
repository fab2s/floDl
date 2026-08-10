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

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
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
    pub(super) root: PathBuf,
    /// The per-session credential injected into the page.
    pub(super) token: String,
    /// The binary driven for subcommand routes — the running fdl
    /// itself outside tests.
    pub(super) fdl_bin: PathBuf,
    /// Bound port, for the Host check.
    pub(super) port: u16,
    /// The single long-running job (a publish gate build). One at a
    /// time on purpose: two concurrent publishes race the manifest
    /// commit point.
    pub(super) job: JobSlot,
    /// The dashboard slot's proxy target: a loopback PORT, set from the
    /// run tab. Port-only by construction — the host is always
    /// 127.0.0.1, so the proxy cannot be aimed off-box.
    pub(super) run_target: Mutex<Option<u16>>,
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
    pub(super) path: PathBuf,
    pub(super) farm: Option<String>,
    /// The dashboard slot's target at launch, best-effort — the run
    /// tab's port if the operator set one.
    pub(super) port: Option<u16>,
}

impl From<Vec<u8>> for Reply {
    fn from(bytes: Vec<u8>) -> Self {
        Reply::Bytes(bytes)
    }
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

mod drive;
mod http;
mod job;
mod slot;

use drive::{
    commands_route, farm_artifacts, launch_argv, parse_body, publish_argv, route_argv, run_fdl,
};
use drive::{join_config_argv, run_target_route};
use http::{Request, error_json, host_is_local, http, json_ok, percent_encode, read_request};
use job::{JobSlot, follow_job, stream_job};
use slot::{proxy_dashboard, read_runs_ledger, scan_archives, serve_archive};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
