//! `fdl status` — live cluster run status.
//!
//! Fetches the controller's `state.json` and pretty-prints it. The
//! endpoint rides the controller's single training port (flodl's
//! port mux routes plain HTTP GETs to a status responder), so the only
//! thing this command needs is the controller address — resolved from
//! the active env overlay's `cluster.controller`, or passed explicitly
//! with `--addr` (e.g. by a self-deployed worker's owner who has
//! nothing but the address).
//!
//! The endpoint lives exactly as long as the launcher process:
//! connection-refused is the honest "no run listening" signal, not an
//! error in this command's plumbing — it is still reported as a
//! failure exit so scripts can gate on it.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::config::{self, DEFAULT_CONTROLLER_PORT};
use crate::style;

/// Connect/read budget per attempt. Status answers are one small JSON
/// body; anything slower than this is a wedged endpoint, not a run.
const HTTP_TIMEOUT: Duration = Duration::from_secs(5);

/// Run `fdl status`.
///
/// Address resolution, in order:
/// 1. `--addr <host[:port]>` — used exactly as given.
/// 2. Active env overlay (`fdl @cluster status` / `FDL_ENV=cluster`):
///    `cluster.controller.host:port`, with a loopback retry on
///    connection-refused (an all-tunneled run binds loopback only, and
///    `fdl status` typically runs on the controller box).
/// 3. Convention default `127.0.0.1:1337` (single-host / auto-promoted
///    runs), noted on stderr.
///
/// Exit code: 0 when the state was fetched and printed; 1 when no
/// endpoint answered.
pub fn run(json: bool, addr_override: Option<&str>) -> i32 {
    let (candidates, origin) = resolve_candidates(addr_override);

    let mut last_err = String::new();
    for addr in &candidates {
        match fetch_state(addr) {
            Ok(body) => {
                if json {
                    println!("{body}");
                } else {
                    print_status(addr, &body);
                }
                return 0;
            }
            Err(e) => last_err = e,
        }
    }
    crate::cli_error!(
        "no cluster run listening at {} ({origin}): {last_err}\n\
         The status endpoint lives on the controller's training port for \
         exactly as long as the run does — a refused connection usually \
         just means no run is up.",
        candidates.join(" / "),
    );
    1
}

/// Run `fdl start`: fire the operator start switch of a staging run.
/// Same address resolution as `fdl status`; the refusal reasons the
/// controller sends back (auto mode, quorum not met, window closed,
/// bad token) ARE the UX, so they print verbatim.
pub fn run_start(addr_override: Option<&str>, token: Option<&str>) -> i32 {
    let (candidates, origin) = resolve_candidates(addr_override);

    let mut last_err = String::new();
    for addr in &candidates {
        match post_start(addr, token) {
            Ok(body) => {
                let joined = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|v| v["joined_ranks"].as_u64());
                match joined {
                    Some(n) => println!(
                        "start armed @ {addr} — the world forms with the \
                         {n} rank(s) staged (watch `fdl status`)",
                    ),
                    None => println!("start armed @ {addr} — {body}"),
                }
                return 0;
            }
            // A served refusal means the controller WAS found — its
            // reason (auto mode, quorum, bad token) is the answer, and
            // trying further addresses would only mask it behind a
            // connect error.
            Err(e) if e.starts_with("endpoint answered") => {
                crate::cli_error!("start refused @ {addr}: {e}");
                return 1;
            }
            Err(e) => last_err = e,
        }
    }
    crate::cli_error!(
        "start not armed at {} ({origin}): {last_err}",
        candidates.join(" / "),
    );
    1
}

/// Resolve the ordered list of addresses to try + a human tag saying
/// where they came from (for the failure message).
fn resolve_candidates(addr_override: Option<&str>) -> (Vec<String>, String) {
    if let Some(addr) = addr_override {
        let addr = if addr.contains(':') {
            addr.to_string()
        } else {
            format!("{addr}:{DEFAULT_CONTROLLER_PORT}")
        };
        return (vec![addr], "--addr".to_string());
    }
    if let Ok(env_name) = std::env::var("FDL_ENV")
        && let Some(cluster) = load_cluster_for_env(&env_name) {
            let host = cluster.controller.host.clone();
            let port = cluster.controller.port;
            let mut candidates = vec![format!("{host}:{port}")];
            // All-tunneled runs bind loopback only; when fdl runs on the
            // controller box (the common case) the loopback retry finds
            // them without reimplementing flodl's bind-scope rules.
            if host != "127.0.0.1" && host != "localhost" {
                candidates.push(format!("127.0.0.1:{port}"));
            }
            return (candidates, format!("fdl.{env_name}.yml controller"));
        }
    eprintln!(
        "{}",
        style::dim(&format!(
            "fdl status: no cluster env active; trying \
             127.0.0.1:{DEFAULT_CONTROLLER_PORT} (pass --addr or use \
             `fdl @<env> status` to target a specific controller)"
        )),
    );
    (
        vec![format!("127.0.0.1:{DEFAULT_CONTROLLER_PORT}")],
        "convention default".to_string(),
    )
}

fn load_cluster_for_env(env_name: &str) -> Option<config::ClusterConfig> {
    // Project-level walk (not the plain context): run from inside a
    // command dir (e.g. ddp-bench/), the nearest fdl.yml is a command
    // config with no `cluster:` — the block lives a level up.
    let cwd = std::env::current_dir().ok()?;
    let config_path = config::find_project_config(&cwd)?;
    let project = config::load_project_with_env(&config_path, Some(env_name)).ok()?;
    project.cluster
}

/// One HTTP GET of `/state.json`. Hand-rolled over TcpStream: the
/// endpoint is plain HTTP on a cleartext port, no TLS involved.
fn fetch_state(addr: &str) -> Result<String, String> {
    http_round_trip(
        addr,
        &format!(
            "GET /state.json HTTP/1.1\r\nHost: {addr}\r\n\
             Connection: close\r\n\r\n"
        ),
    )
}

/// One `POST /start` (the operator start switch), token as a query
/// param when given.
fn post_start(addr: &str, token: Option<&str>) -> Result<String, String> {
    let path = match token {
        Some(t) => format!("/start?token={t}"),
        None => "/start".to_string(),
    };
    http_round_trip(
        addr,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: {addr}\r\n\
             Connection: close\r\nContent-Length: 0\r\n\r\n"
        ),
    )
}

/// Send one request, read the whole response, return the body on 200
/// and `Err(status — body)` otherwise.
fn http_round_trip(addr: &str, request: &str) -> Result<String, String> {
    let sock_addr = addr
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve {addr}: {e}"))?
        .next()
        .ok_or_else(|| format!("cannot resolve {addr}"))?;
    let mut stream = TcpStream::connect_timeout(&sock_addr, HTTP_TIMEOUT)
        .map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(HTTP_TIMEOUT))
        .and_then(|()| stream.set_write_timeout(Some(HTTP_TIMEOUT)))
        .map_err(|e| format!("socket setup: {e}"))?;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("send request: {e}"))?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|e| format!("read response: {e}"))?;

    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line = head.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        return Err(format!(
            "endpoint answered {} — {}",
            status_line.trim_start_matches("HTTP/1.1 "),
            body.trim(),
        ));
    }
    Ok(body.trim().to_string())
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Pretty-print a `state.json` body. Parsed as a loose `Value` so an
/// fdl one version ahead of (or behind) the running flodl still renders
/// what it recognizes instead of failing on an exact-shape mismatch.
fn print_status(addr: &str, body: &str) {
    let state: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => {
            // Not JSON we understand — show it raw rather than nothing.
            println!("{body}");
            return;
        }
    };

    let phase = state["phase"].as_str().unwrap_or("unknown");
    let painted_phase = match phase {
        "training" | "done" => style::green(phase),
        "waiting" | "staging" | "forming" => style::yellow(phase),
        "failed" => style::red(phase),
        other => other.to_string(),
    };
    println!(
        "cluster run @ {addr} — {}",
        style::bold(&painted_phase),
    );

    let joined_ranks = state["joined_ranks"].as_u64().unwrap_or(0);
    let joined_hosts = state["joined_hosts"].as_u64().unwrap_or(0);
    let quorum = state["min_rank_start"].as_u64().unwrap_or(0);
    let target = state["target_ranks"]
        .as_u64()
        .map(|t| t.to_string())
        .unwrap_or_else(|| "none".to_string());
    println!(
        "  ranks: {joined_ranks} joined across {joined_hosts} host(s)   \
         (quorum {quorum}, target {target})",
    );
    // The countdown is only meaningful while the window is open; once
    // formed, the snapshot's remaining-times are frozen at formation.
    if phase == "waiting" || phase == "staging" {
        let fmt_remaining = |v: &serde_json::Value| match v.as_u64() {
            Some(s) => format!("{s}s left"),
            None => "expired".to_string(),
        };
        println!(
            "  window: {}   hard cap: {}",
            fmt_remaining(&state["window_remaining_secs"]),
            fmt_remaining(&state["cap_remaining_secs"]),
        );
    }
    // Operator start switch: only rendered when the run has one (older
    // flodl snapshots have no start_mode field — absent ≠ auto).
    if let Some(mode) = state["start_mode"].as_str() {
        // Only meaningful while the window still holds — once the world
        // forms, the switch is history.
        if mode != "auto" && matches!(phase, "waiting" | "staging") {
            let armed = state["start_armed"].as_bool().unwrap_or(false);
            if armed {
                println!("  start: {mode} — armed (forming at the next poll)");
            } else if phase == "staging" {
                println!(
                    "  start: {mode} — {}",
                    style::bold("roster startable, fire with `fdl start`"),
                );
            } else if phase == "waiting" {
                println!("  start: {mode} — waiting for quorum");
            }
        }
    }

    let Some(members) = state["members"].as_array() else {
        return;
    };
    if members.is_empty() {
        println!("  hosts: none joined yet");
        return;
    }
    println!("  hosts:");
    let host_width = members
        .iter()
        .filter_map(|m| m["host"].as_str())
        .map(str::len)
        .max()
        .unwrap_or(0);
    for m in members {
        let host = m["host"].as_str().unwrap_or("?");
        let ranks: Vec<String> = m["ranks"]
            .as_array()
            .map(|a| a.iter().filter_map(|r| r.as_u64()).map(|r| r.to_string()).collect())
            .unwrap_or_default();
        let joined_at = m["joined_at_secs"].as_u64().unwrap_or(0);
        let libtorch = m["libtorch"].as_str().unwrap_or("?");
        // Pad BEFORE painting: ANSI escapes would break {:width$}.
        let padded_host = format!("{host:<host_width$}");
        println!(
            "    {}  ranks [{}]  {}  libtorch {}  joined +{joined_at}s",
            style::bold(&padded_host),
            ranks.join(", "),
            summarize_gpus(&m["gpus"]),
            libtorch,
        );
    }
}

/// Collapse a GPU label list: identical names group as `2x <name>`,
/// mixed inventories list out.
fn summarize_gpus(gpus: &serde_json::Value) -> String {
    let names: Vec<&str> = gpus
        .as_array()
        .map(|a| a.iter().filter_map(|g| g.as_str()).collect())
        .unwrap_or_default();
    if names.is_empty() {
        return "no GPUs listed".to_string();
    }
    if names.iter().all(|n| *n == names[0]) {
        return format!("{}x {}", names.len(), names[0]);
    }
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addr_override_gets_default_port_when_bare() {
        let (candidates, origin) = resolve_candidates(Some("10.0.0.7"));
        assert_eq!(candidates, vec!["10.0.0.7:1337".to_string()]);
        assert_eq!(origin, "--addr");
        let (candidates, _) = resolve_candidates(Some("10.0.0.7:9000"));
        assert_eq!(candidates, vec!["10.0.0.7:9000".to_string()]);
    }

    #[test]
    fn gpu_summary_groups_identical_names() {
        let gpus = serde_json::json!(["GP106", "GP106"]);
        assert_eq!(summarize_gpus(&gpus), "2x GP106");
        let gpus = serde_json::json!(["GP106", "RTX 5060 Ti"]);
        assert_eq!(summarize_gpus(&gpus), "GP106, RTX 5060 Ti");
        assert_eq!(summarize_gpus(&serde_json::json!([])), "no GPUs listed");
    }

    #[test]
    fn fetch_state_reports_non_200_with_body() {
        // Minimal one-shot HTTP server answering 503.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = stream.read(&mut buf);
            let body = r#"{"error":"no membership state published yet"}"#;
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 503 Service Unavailable\r\n\
                     Content-Type: application/json\r\n\
                     Connection: close\r\n\
                     Content-Length: {}\r\n\r\n{body}",
                    body.len(),
                )
                .as_bytes(),
            );
        });
        let err = fetch_state(&addr.to_string()).unwrap_err();
        assert!(err.contains("503"), "{err}");
        assert!(err.contains("no membership state"), "{err}");
        server.join().unwrap();
    }

    #[test]
    fn fetch_state_round_trips_200_body() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = stream.read(&mut buf);
            let body = r#"{"phase":"training","joined_ranks":3}"#;
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/json\r\n\
                     Connection: close\r\n\
                     Content-Length: {}\r\n\r\n{body}",
                    body.len(),
                )
                .as_bytes(),
            );
        });
        let body = fetch_state(&addr.to_string()).unwrap();
        assert_eq!(body, r#"{"phase":"training","joined_ranks":3}"#);
        server.join().unwrap();
    }
}
