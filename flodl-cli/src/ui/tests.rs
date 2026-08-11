//! Tests for the `fdl ui` server.
//!
//! Lives beside the module rather than inside it, the same shape
//! `overlay_tests.rs` / `cli_tests.rs` and the `join_config` and `probe`
//! module directories use: the server's own files stay about serving.
//!
//! Two standing rules here, both learned the hard way elsewhere in this
//! crate. Nothing EXECS a file the test just wrote (a fork in the write
//! window inherits the fd and the exec dies ETXTBSY), so the driven
//! routes are covered by pure argv builders plus a live smoke instead.
//! And no test asserts a platform's own rendering of anything.

use std::io::{Read as _, Write as _};

use super::drive::{append_ledger, join_config_argv, publish_argv, route_argv, run_fdl};
use super::http::{percent_decode, percent_encode};
use super::job::JobSlot;
use super::slot::read_runs_ledger;
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
fn the_run_tab_carries_the_dashboard_discovery_wiring() {
    // The discovery itself is page JS, which no Rust test can execute, so
    // this guards the wiring that makes it possible: the port comes from
    // the run's own status document, and a discovered-but-unreachable
    // port must name the tunnel rather than leave the operator guessing.
    // A refactor that drops either fails here instead of on the rig.
    let tmp = tempdir();
    let (port, token) = spawn_server(&tmp);
    let local = format!("127.0.0.1:{port}");
    let page = get(port, "/", Some(&local), None);
    assert!(page.contains("dashboardPortFrom"), "discovery helper gone");
    assert!(page.contains("dashboard_port"), "status field not read");
    assert!(page.contains("ssh -L "), "tunnel hint gone");
    // Discovery only ever proposes; the reachability probe decides.
    assert!(page.contains("/api/run-target"), "probe route gone");
    let _ = token;
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
        get(port, "/api/farm?label=no/pe", Some(&local), Some(&token)).starts_with("HTTP/1.1 400"),
    );
    assert!(
        get(port, "/api/farm?label=absent", Some(&local), Some(&token)).starts_with("HTTP/1.1 404"),
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

    // The path-preserving form serves the same file, and — the reason it
    // exists — a RELATIVE link inside the served page resolves under the
    // same prefix to a sibling artifact (the telemetry index card's
    // per-host pages). A query-shaped URL gave the browser no base.
    let served = get(
        port,
        "/archive/ddp-bench/runs/mlp/nccl-sync/dashboard.html",
        Some(&local),
        None,
    );
    assert!(served.ends_with("<html>ARCHIVE</html>"), "{served}");
    let telem = run_dir.join("telemetry/exa/run-1");
    std::fs::create_dir_all(&telem).unwrap();
    std::fs::write(telem.join("timeline.html"), "<html>HOSTPAGE</html>").unwrap();
    let sibling = get(
        port,
        "/archive/ddp-bench/runs/mlp/nccl-sync/telemetry/exa/run-1/timeline.html",
        Some(&local),
        None,
    );
    assert!(sibling.ends_with("<html>HOSTPAGE</html>"), "{sibling}");
    for escape in [
        "/archive/../flodl/src/monitor/dashboard.html",
        "/archive/ddp-bench/runs/mlp/nccl-sync/notes.html",
    ] {
        let out = get(port, escape, Some(&local), None);
        assert!(out.starts_with("HTTP/1.1 400"), "{escape}: {out}");
    }

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
