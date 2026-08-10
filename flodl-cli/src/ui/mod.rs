//! `fdl ui` — the local operations page.
//!
//! One loopback HTTP server, one embedded page, zero dependencies: the
//! browser counterpart of the walk-in CLI surface. Read-only in this
//! slice — farms, hardware probe, cluster run status, resolved config —
//! with actions (the wizard form, publish) arriving on the same server.
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
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
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
}

fn handle(mut stream: TcpStream, server: &UiServer) {
    let Some(req) = read_request(&mut stream) else {
        return;
    };
    let response = respond(&req, server);
    let _ = stream.write_all(&response);
}

/// One parsed request — only what routing needs.
struct Request {
    method: String,
    /// Path without the query string.
    path: String,
    query: HashMap<String, String>,
    host: Option<String>,
    token: Option<String>,
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
            }
        }
    }
    Some(Request {
        method,
        path: path.to_string(),
        query,
        host,
        token,
    })
}

/// Route a request to its response bytes.
fn respond(req: &Request, server: &UiServer) -> Vec<u8> {
    // The Host check guards every route: a DNS-rebound page reaches the
    // socket but cannot present the loopback host it was served on.
    if !host_is_local(req.host.as_deref(), server.port) {
        return error_json("403 Forbidden", "wrong or missing Host header");
    }
    if req.method != "GET" {
        return error_json("405 Method Not Allowed", "GET only (in this slice)");
    }
    match req.path.as_str() {
        "/" => {
            let page = PAGE_HTML
                .replace("__FDL_TOKEN__", &server.token)
                .replace("__FDL_ROOT__", &server.root.display().to_string());
            http("200 OK", "text/html; charset=utf-8", page.as_bytes())
        }
        path if path.starts_with("/api/") => {
            // Every API route needs the session token: the page holds
            // it, a cross-site request cannot.
            if req.token.as_deref() != Some(server.token.as_str()) {
                return error_json("403 Forbidden", "missing or wrong x-fdl-token header");
            }
            api(req, path, server)
        }
        _ => error_json("404 Not Found", "unknown route"),
    }
}

fn api(req: &Request, path: &str, server: &UiServer) -> Vec<u8> {
    // An env name arrives from a query string; it must survive the same
    // charset gate as a farm label before it becomes an argument.
    let env = match req.query.get("env").map(String::as_str) {
        None | Some("") => None,
        Some(e) => match validate_label(e) {
            Ok(()) => Some(e),
            Err(why) => return error_json("400 Bad Request", &why),
        },
    };
    match path {
        // Pure local read — the same function `--list --json` prints.
        "/api/farms" => json_ok(&farms_json(&server.root)),
        "/api/probe" | "/api/status" | "/api/config" => {
            let argv = route_argv(path, env);
            let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
            json_ok(&run_fdl(&server.fdl_bin, &server.root, &refs))
        }
        _ => error_json("404 Not Found", "unknown api route"),
    }
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
            // live smoke, never by tests execing something.
            fdl_bin: PathBuf::from("fdl-never-spawned"),
            port,
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
