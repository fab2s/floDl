//! Multi-host launcher: ssh fan-out for `cluster:`-marked commands.
//!
//! Entry point [`dispatch`] iterates `project.cluster.hosts`, builds a
//! per-host slim envelope (via
//! [`ClusterConfig::local_envelope_for`](crate::config::ClusterConfig::local_envelope_for)),
//! hex-encodes the JSON into `FLODL_CLUSTER_JSON`, and spawns one child per
//! host:
//!
//! ```text
//! # remote host (host.name != controller's hostname)
//! ssh -T <host> bash -lc 'cd <path> && \
//!     FLODL_CLUSTER_JSON=<hex> FLODL_HOST_NAME=<name> \
//!     exec fdl <cmd> <user_args>'
//!
//! # local fast path (host.name == controller's hostname)
//! bash -lc 'cd <path> && FLODL_CLUSTER_JSON=<hex> FLODL_HOST_NAME=<name> \
//!     exec fdl <cmd> <user_args>'
//! ```
//!
//! The local fast path drops the sshd-on-controller prereq -- a fresh
//! workstation can run `fdl @cluster <cmd>` without first configuring
//! ssh-to-self. Both paths produce identical child shapes (piped stdout /
//! stderr, same env propagation via the inline shell prefix), so the
//! mux / wait / exit-code logic downstream doesn't branch.
//!
//! Stdout/stderr from each ssh child are line-prefixed with `[host] ` at the
//! launcher; library-prefixed lines (`[host:dev:rN]`) appear after the
//! launcher prefix, making the layering visible during debug.
//!
//! Recursion guard: a remote `fdl` invocation sees `FLODL_CLUSTER_JSON` in
//! its env and [`should_dispatch`] returns false -- the remote runs the
//! command locally instead of dispatching again.
//!
//! Controller participation is implicit: if the controller's hostname
//! appears in `cluster.hosts`, it gets an ssh (loopback) and participates as
//! a rank. If absent, the launcher emits a warning and runs orchestrator-
//! only -- no NCCL involvement on the controller. The warning catches
//! typos in `host.name` that would silently demote the controller.
//!
//! Backend: ssh-star-fanout via std `Command` + per-host reader threads. No
//! external deps. Scales comfortably to ~tens of hosts. Tree-fanout or
//! daemon-based transports for ~hundreds belong behind a `LauncherBackend`
//! seam (not built yet -- single function to swap when scale forces it).

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitCode, Stdio};
use std::thread;

use crate::config::{self, ProjectConfig};

/// Environment variable carrying the hex-encoded slim envelope to remote
/// nodes. Mirrors `flodl::distributed::cluster::ENV_CLUSTER_JSON`.
pub const ENV_CLUSTER_JSON: &str = "FLODL_CLUSTER_JSON";

/// Environment variable telling the remote library which host it is.
/// Mirrors `flodl::distributed::cluster::ENV_HOST_OVERRIDE`.
pub const ENV_HOST_OVERRIDE: &str = "FLODL_HOST_NAME";

/// Environment variable carrying the fdl overlay name (e.g. `cluster`).
/// fdl-cli reads this at startup; setting it on the remote ssh invocation
/// makes the remote see the same `commands:` resolution as the controller.
pub const ENV_FDL_ENV: &str = "FDL_ENV";

/// SSH options shared by every host invocation.
///
/// - `-T`: disable pseudo-terminal allocation (keeps stdout/stderr clean).
/// - `ServerAliveInterval=10` + `ServerAliveCountMax=3`: client gives up
///   after ~30s of silence so a dead remote doesn't hang the controller.
/// - `BatchMode=yes`: fail fast on auth issues; no interactive prompts. If
///   keys aren't set up, the launcher errors instead of stalling at a
///   password prompt that the user might miss while watching logs.
const SSH_OPTS: &[&str] = &[
    "-T",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "BatchMode=yes",
];

/// Top-level cluster-dispatch decision.
///
/// Returns `false` when `FLODL_CLUSTER_JSON` is set in the process env --
/// that signals we're a remote node, not the controller. Otherwise delegates
/// to [`config::cluster_dispatch_enabled`].
pub fn should_dispatch(project: &ProjectConfig, chain: &[Option<bool>]) -> bool {
    if std::env::var_os(ENV_CLUSTER_JSON).is_some() {
        return false;
    }
    config::cluster_dispatch_enabled(project, chain)
}

/// Dispatch `<cmd> <user_args>` across every host in `project.cluster.hosts`.
///
/// `overlay_env` is forwarded via `FDL_ENV` so the remote sees the same
/// `commands:` resolution (overlay-merged `fdl.<env>.yml`).
///
/// Returns `ExitCode::FAILURE` if any host's ssh child fails or the cluster
/// config validation fails. Output muxing prefixes every remote line with
/// `[host] ` at the controller side.
pub fn dispatch(
    project: &ProjectConfig,
    overlay_env: Option<&str>,
    cmd: &str,
    user_args: &[String],
) -> ExitCode {
    let cluster = match project.cluster.as_ref() {
        Some(c) => c,
        None => {
            crate::cli_error!(
                "cluster dispatch requested but no `cluster:` block in fdl.yml"
            );
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = cluster.validate() {
        crate::cli_error!("cluster config invalid: {e}");
        return ExitCode::FAILURE;
    }

    // Controller participation: implicit-by-presence in cluster.hosts.
    let me = resolve_local_hostname();
    if !cluster.hosts.iter().any(|h| h.name == me) {
        eprintln!(
            "fdl: controller hostname {me:?} not in cluster.hosts; \
             running orchestrator-only (no NCCL participation). \
             If you meant to participate, fix `host.name` to match `hostname`."
        );
    }

    // Spawn one ssh process per host; pin reader threads to each.
    let mut children: Vec<HostChild> = Vec::with_capacity(cluster.hosts.len());
    for host in &cluster.hosts {
        let envelope = cluster.local_envelope_for(host);
        let envelope_bytes = match serde_json::to_vec(&envelope) {
            Ok(b) => b,
            Err(e) => {
                crate::cli_error!(
                    "cluster: failed to serialize envelope for {:?}: {e}",
                    host.name
                );
                kill_all(&mut children);
                return ExitCode::FAILURE;
            }
        };
        let envelope_hex = hex_encode(&envelope_bytes);

        let remote_cmd = build_remote_command(
            &host.path,
            &envelope_hex,
            &host.name,
            overlay_env,
            cmd,
            user_args,
        );

        let mut child = match build_spawn_command(
            &host.name,
            &me,
            cluster.ssh_target(host),
            &remote_cmd,
        )
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let kind = if host.name == me { "bash" } else { "ssh" };
                crate::cli_error!(
                    "cluster: failed to spawn {kind} for {:?}: {e}",
                    host.name
                );
                kill_all(&mut children);
                return ExitCode::FAILURE;
            }
        };

        let prefix = format!("[{}] ", host.name);
        let mut threads = Vec::with_capacity(2);
        if let Some(out) = child.stdout.take() {
            threads.push(relay_lines(out, prefix.clone(), Stream::Stdout));
        }
        if let Some(err) = child.stderr.take() {
            threads.push(relay_lines(err, prefix.clone(), Stream::Stderr));
        }
        children.push(HostChild {
            name: host.name.clone(),
            child,
            threads,
        });
    }

    // Wait for all children. A non-zero exit on any host fails the run.
    let mut overall_ok = true;
    for hc in children {
        let HostChild {
            name,
            mut child,
            threads,
        } = hc;
        match child.wait() {
            Ok(status) => {
                if !status.success() {
                    overall_ok = false;
                    let code = status
                        .code()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "signal".into());
                    eprintln!("[{name}] fdl: remote exit status {code}");
                }
            }
            Err(e) => {
                overall_ok = false;
                eprintln!("[{name}] fdl: wait failed: {e}");
            }
        }
        for t in threads {
            let _ = t.join();
        }
    }

    if overall_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

struct HostChild {
    name: String,
    child: std::process::Child,
    threads: Vec<thread::JoinHandle<()>>,
}

fn kill_all(children: &mut Vec<HostChild>) {
    for hc in children.iter_mut() {
        let _ = hc.child.kill();
    }
    for hc in children.drain(..) {
        let _ = hc.child.wait_with_output();
        for t in hc.threads {
            let _ = t.join();
        }
    }
}

#[derive(Clone, Copy)]
enum Stream {
    Stdout,
    Stderr,
}

fn relay_lines<R: std::io::Read + Send + 'static>(
    reader: R,
    prefix: String,
    stream: Stream,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut buf = BufReader::new(reader);
        let mut line = Vec::new();
        loop {
            line.clear();
            match buf.read_until(b'\n', &mut line) {
                Ok(0) => break,
                Ok(_) => write_line(stream, prefix.as_bytes(), &line),
                Err(_) => break,
            }
        }
    })
}

fn write_line(stream: Stream, prefix: &[u8], line: &[u8]) {
    let needs_newline = !line.ends_with(b"\n");
    match stream {
        Stream::Stdout => {
            let mut w = std::io::stdout().lock();
            let _ = w.write_all(prefix);
            let _ = w.write_all(line);
            if needs_newline {
                let _ = w.write_all(b"\n");
            }
            let _ = w.flush();
        }
        Stream::Stderr => {
            let mut w = std::io::stderr().lock();
            let _ = w.write_all(prefix);
            let _ = w.write_all(line);
            if needs_newline {
                let _ = w.write_all(b"\n");
            }
            let _ = w.flush();
        }
    }
}

/// Build the `Command` that spawns the per-host child.
///
/// When `host_name == me`, returns a `bash -lc '<remote_cmd>'` Command --
/// the controller doesn't need sshd-on-self for its own rank. Otherwise
/// returns `ssh <SSH_OPTS> <ssh_target> '<remote_cmd>'`. Both shapes
/// produce identical child semantics (piped streams, same env propagation
/// via the inline shell prefix in `remote_cmd`).
pub(crate) fn build_spawn_command(
    host_name: &str,
    me: &str,
    ssh_target: &str,
    remote_cmd: &str,
) -> Command {
    if host_name == me {
        let mut c = Command::new("bash");
        c.arg("-lc").arg(remote_cmd);
        c
    } else {
        let mut c = Command::new("ssh");
        c.args(SSH_OPTS).arg(ssh_target).arg(remote_cmd);
        c
    }
}

/// Build the bash command shipped via ssh to the remote.
///
/// Single level of shell quoting: ssh delivers the string verbatim to the
/// remote login shell, which parses it once. Every interpolated value is
/// single-quoted via [`shell_quote`] (`'\''`-escape idiom for internal
/// quotes). `exec` replaces the bash process so the remote returns fdl's
/// exit code directly.
pub(crate) fn build_remote_command(
    path: &str,
    cluster_json_hex: &str,
    host_name: &str,
    overlay_env: Option<&str>,
    cmd: &str,
    user_args: &[String],
) -> String {
    let mut s = String::with_capacity(
        256
            + cluster_json_hex.len()
            + user_args.iter().map(|a| a.len() + 4).sum::<usize>(),
    );
    s.push_str("cd ");
    s.push_str(&shell_quote(path));
    s.push_str(" && ");
    s.push_str(ENV_CLUSTER_JSON);
    s.push('=');
    s.push_str(&shell_quote(cluster_json_hex));
    s.push(' ');
    s.push_str(ENV_HOST_OVERRIDE);
    s.push('=');
    s.push_str(&shell_quote(host_name));
    if let Some(env) = overlay_env {
        s.push(' ');
        s.push_str(ENV_FDL_ENV);
        s.push('=');
        s.push_str(&shell_quote(env));
    }
    s.push_str(" exec fdl ");
    s.push_str(&shell_quote(cmd));
    for a in user_args {
        s.push(' ');
        s.push_str(&shell_quote(a));
    }
    s
}

/// Single-quote a string for shell consumption. Internal single quotes are
/// escaped via the `'\''` idiom (close, backslash-escape, reopen).
fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Hex-encode raw bytes (lowercase, no separators). Mirrors the library's
/// `hex_decode` in `flodl::distributed::cluster`.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(TABLE[(b >> 4) as usize] as char);
        s.push(TABLE[(b & 0x0F) as usize] as char);
    }
    s
}

fn resolve_local_hostname() -> String {
    if let Ok(s) = std::env::var(ENV_HOST_OVERRIDE) {
        let s = s.trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "<unknown>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env-mutating tests in this module serialize on this Mutex so the
    // process-wide `FLODL_CLUSTER_JSON` var isn't observed half-set by
    // a parallel test.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn should_dispatch_returns_false_when_cluster_json_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized via ENV_MUTEX above.
        unsafe {
            std::env::set_var(ENV_CLUSTER_JSON, "deadbeef");
        }
        // Even with a valid project + chain marking cluster: true, the
        // env-set var forces should_dispatch to bail (recursion guard).
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  x: { cluster: true, run: \"echo hi\" }
";
        let project: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(
            !should_dispatch(&project, &[Some(true)]),
            "recursion guard: must return false when FLODL_CLUSTER_JSON is set"
        );
        unsafe {
            std::env::remove_var(ENV_CLUSTER_JSON);
        }
    }

    #[test]
    fn should_dispatch_delegates_when_env_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::remove_var(ENV_CLUSTER_JSON);
        }
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  x: { run: \"echo hi\" }
";
        let project: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        // chain resolves to false (no cluster: true on the leaf or anywhere)
        assert!(!should_dispatch(&project, &[None]));
        // chain resolves to true: delegates to config::cluster_dispatch_enabled
        assert!(should_dispatch(&project, &[Some(true)]));
    }

    #[test]
    fn shell_quote_simple() {
        assert_eq!(shell_quote("hello"), "'hello'");
    }

    #[test]
    fn shell_quote_with_spaces() {
        assert_eq!(shell_quote("/opt/my dir"), "'/opt/my dir'");
    }

    #[test]
    fn shell_quote_escapes_internal_quotes() {
        // The 'close - backslash-quote - reopen' idiom.
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn shell_quote_handles_double_quotes_verbatim() {
        // Inside single quotes, double quotes have no special meaning.
        assert_eq!(shell_quote("say \"hi\""), "'say \"hi\"'");
    }

    #[test]
    fn shell_quote_handles_backslashes_verbatim() {
        assert_eq!(shell_quote("a\\b"), "'a\\b'");
    }

    #[test]
    fn hex_encode_matches_library() {
        // The library's hex_decode of our hex_encode must round-trip.
        // We can't depend on flodl here, so we just verify the well-known
        // mapping for a few bytes.
        assert_eq!(hex_encode(b""), "");
        assert_eq!(hex_encode(&[0x00]), "00");
        assert_eq!(hex_encode(&[0xff]), "ff");
        assert_eq!(hex_encode(&[0x0f, 0xa0]), "0fa0");
        assert_eq!(hex_encode(b"hi"), "6869");
    }

    #[test]
    fn build_remote_command_shape() {
        let s = build_remote_command(
            "/opt/flodl",
            "abcd1234",
            "worker-host",
            Some("cluster"),
            "train",
            &["--epochs".into(), "10".into()],
        );
        assert!(s.starts_with("cd '/opt/flodl' && "), "got: {s}");
        assert!(s.contains("FLODL_CLUSTER_JSON='abcd1234'"), "got: {s}");
        assert!(s.contains("FLODL_HOST_NAME='worker-host'"), "got: {s}");
        assert!(s.contains("FDL_ENV='cluster'"), "got: {s}");
        assert!(s.contains(" exec fdl 'train' "), "got: {s}");
        assert!(s.contains("'--epochs'"), "got: {s}");
        assert!(s.contains("'10'"), "got: {s}");
    }

    #[test]
    fn build_remote_command_omits_fdl_env_when_none() {
        let s = build_remote_command(
            "/opt/flodl",
            "abcd",
            "node-a",
            None,
            "test",
            &[],
        );
        assert!(!s.contains("FDL_ENV"), "got: {s}");
    }

    #[test]
    fn build_remote_command_handles_path_with_spaces() {
        let s = build_remote_command(
            "/opt/my flodl",
            "00",
            "node",
            None,
            "x",
            &[],
        );
        assert!(s.contains("cd '/opt/my flodl' &&"), "got: {s}");
    }

    #[test]
    fn build_remote_command_quotes_dangerous_args() {
        // A user arg containing a single quote must survive the trip
        // through shell parsing on the remote.
        let s = build_remote_command(
            "/opt/flodl",
            "00",
            "node",
            None,
            "cmd",
            &["it's fine".into()],
        );
        assert!(s.contains("'it'\\''s fine'"), "got: {s}");
    }

    #[test]
    fn build_spawn_command_local_uses_bash() {
        let cmd = build_spawn_command("master-host", "master-host", "irrelevant", "echo hi");
        assert_eq!(cmd.get_program(), "bash");
        let args: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "-lc");
        assert_eq!(args[1], "echo hi");
    }

    #[test]
    fn build_spawn_command_remote_uses_ssh() {
        let cmd = build_spawn_command("worker-host", "master-host", "worker.lan", "echo hi");
        assert_eq!(cmd.get_program(), "ssh");
        let args: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
        // SSH_OPTS (7 entries) + ssh_target + remote_cmd = 9 args.
        assert_eq!(args.len(), SSH_OPTS.len() + 2);
        // Last two are target then command.
        assert_eq!(args[args.len() - 2], "worker.lan");
        assert_eq!(args[args.len() - 1], "echo hi");
        // SSH_OPTS includes -T and BatchMode=yes among others.
        let joined: Vec<String> = args.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert!(joined.iter().any(|s| s == "-T"), "got: {joined:?}");
        assert!(
            joined.iter().any(|s| s == "BatchMode=yes"),
            "got: {joined:?}"
        );
    }

    #[test]
    fn build_remote_command_uses_exec() {
        // `exec` is load-bearing: it replaces the bash process so fdl's
        // exit code propagates back through ssh unchanged.
        let s = build_remote_command(
            "/opt/flodl",
            "00",
            "n",
            None,
            "x",
            &[],
        );
        assert!(s.contains(" exec fdl "), "missing `exec`: {s}");
    }
}
