//! Process/SSH spawning, IPC plumbing, and remote-bash command assembly.
//!
//! Run-launcher uses these helpers to fan out to per-host children
//! (local fork+exec on the controller host, SSH for remotes), wire
//! up stdin/stdout/stderr, propagate signals, and clean up on exit.

use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde::Deserialize;

use crate::tensor::{Result, TensorError};

use super::{ENV_FDL_ENV, ENV_FULL_CLUSTER_JSON, ENV_PREBUILD_PER_HOST};
use super::{FullCluster, FullWorker};

/// Per-host pre-flight build artifact, as published by fdl-cli's
/// prebuild phase in [`ENV_PREBUILD_PER_HOST`]. The launcher reads
/// the env var, parses it as `BTreeMap<String, PerHostPrebuild>`
/// keyed by host name, and substitutes the direct-binary form on the
/// remote dispatch path for any matching host. Mirrors
/// `flodl_cli::prebuild::PerHostEnvelope`.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct PerHostPrebuild {
    /// Path to the compiled binary, relative to `host.path`.
    pub(super) bin: String,
    /// Absolute `LD_LIBRARY_PATH` for the libtorch the binary was
    /// linked against. The launcher emits this verbatim before exec
    /// so the binary finds its shared libs at runtime.
    pub(super) ld_library_path: String,
    /// Subdirectory under `host.path` to `cd` into before exec — the
    /// controller's view of the command's filesystem cwd, relative to
    /// the project root. Mirrors the cwd the controller-side build /
    /// invocation used (`docker compose run … bash -c "cd /workspace/<sub> && …"`).
    /// Empty string means "stay at `host.path`".
    #[serde(default)]
    pub(super) cwd_subpath: String,
}

/// Load and parse [`ENV_PREBUILD_PER_HOST`] from the current process
/// env. Returns an empty map when the env var is absent (legacy
/// fan-out via `fdl <cmd>` re-entry), or a `TensorError` on JSON
/// parse failure (loud — bad JSON masks a real misconfiguration in
/// the controller's prebuild step).
pub(super) fn load_prebuild_envelope() -> Result<BTreeMap<String, PerHostPrebuild>> {
    let raw = match env::var(ENV_PREBUILD_PER_HOST) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return Ok(BTreeMap::new()),
    };
    serde_json::from_str(&raw).map_err(|e| {
        TensorError::new(&format!(
            "cluster launcher: parse {ENV_PREBUILD_PER_HOST} JSON: {e}",
        ))
    })
}

/// SSH options shared by every remote host invocation. Match fdl-cli's
/// existing flodl-cli/src/cluster.rs constants verbatim:
/// - `-T`: disable PTY (keeps stdout/stderr clean)
/// - `ServerAliveInterval=10` + `ServerAliveCountMax=3`: client gives up
///   after ~30s of silence so a dead remote doesn't hang the controller
/// - `BatchMode=yes`: fail fast on auth issues; no interactive prompts
/// - `StrictHostKeyChecking=accept-new`: trust-on-first-use for hosts
///   the user listed in `cluster.yml`. Cluster mode is batch and
///   `BatchMode=yes` blocks the interactive "yes/no" host-key prompt;
///   without accept-new, the first connection from a fresh container
///   (no `known_hosts`) hard-fails with "Host key verification
///   failed". `accept-new` writes the key to known_hosts on first
///   contact and still errors loudly on subsequent mismatches, so
///   MITM detection survives.
pub(super) const SSH_OPTS: &[&str] = &[
    "-T",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
];

/// Drain `children` concurrently and return the first failure (if any).
///
/// One watcher thread per child polls `try_wait()` and posts the exit
/// status on an mpsc channel. The main loop receives events in
/// completion order; on the first non-zero exit it raises a shared
/// kill-all flag and each still-running watcher SIGKILLs its own child so
/// NCCL-blocked ranks abort their retry loop and exit instead of hanging.
/// Forwarder threads (one for each child's stdout / stderr) are joined
/// after the corresponding child's pipes close. Returns `None` when every
/// child exits cleanly, `Some(err)` attributing to the first failure
/// otherwise.
///
/// Kills go through each watcher's owned `Child::kill()`, never a raw
/// snapshotted PID: the kernel pins the PID as a zombie until the owning
/// watcher reaps it, and `Child::kill()` refuses to signal an already-reaped
/// child, so a kill can never race PID recycling onto an unrelated process.
/// This is SIGKILL (std has no graceful terminate without `libc`, which
/// `flodl` avoids as a direct dep); kill-all is the fatal pre-formation /
/// no-coordinator teardown, and remote children still tear down gracefully
/// via their ssh HUP trap when their local ssh process is killed here.
///
/// # No parent-driven process-group sweep
///
/// This handles the peer-failure direction (a child dies → terminate the
/// rest). It deliberately does NOT put local children in their own process
/// group and sweep them on launcher exit. The launcher hosts the
/// coordinator, so launcher death IS coordinator death — and ranks already
/// self-terminate on that: each rank's inbound bridge sees the control
/// stream EOF and injects `ControlMsg::Shutdown`, and the reduce loop's
/// per-read deadline (`cpu_reduce::REDUCE_READ_DEADLINE_SECS`) bails a rank
/// blocked reading from the vanished coordinator. Comms-loss self-exit is
/// strictly more robust than a `Drop`-based sweep, which would not run at
/// all on a launcher hard-kill (`SIGKILL`) — exactly the case that strands
/// children. Full coverage of hard-kill would need `PR_SET_PDEATHSIG`
/// (libc/prctl), outside the no-libc constraint. The one irreducible
/// residual is a rank wedged inside a hung NCCL collective / CUDA kernel
/// the driver never returns from: it holds the GPU and no pure-Rust
/// mechanism reaps it, so it stays a manual `docker restart` / `ssh pkill`
/// (the rig-hygiene backstop).
/// One supervised child: host, local rank (or [`RELAY_RANK_SENTINEL`] for
/// the relay), the global ranks the child carries (reported dead on a
/// non-zero exit under elastic supervision), the process handle, and
/// its output-forwarder threads.
pub(super) type SupervisedChild = (
    String,
    usize,
    Vec<usize>,
    std::process::Child,
    Vec<thread::JoinHandle<()>>,
);

/// Watcher poll cadence for `try_wait` / kill-flag observation. Children run
/// for the whole training, so this only bounds exit-detection and kill-all
/// latency; 100ms is imperceptible against that and cheap for a handful of
/// watcher threads.
const WATCH_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Local-rank slot value marking a per-host relay child (which carries a whole
/// host's rank set, not a single rank). Formatted as `"relay"` in supervision
/// diagnostics via [`child_label`] rather than leaking the raw sentinel.
pub(super) const RELAY_RANK_SENTINEL: usize = usize::MAX;

/// Human label for a supervised child in diagnostics: `"relay of <host>"` for
/// the relay sentinel, `"rank <n> of <host>"` for a rank child.
fn child_label(lr: usize, host: &str) -> String {
    if lr == RELAY_RANK_SENTINEL {
        format!("relay of {host}")
    } else {
        format!("rank {lr} of {host}")
    }
}

/// Describe how a child ended. On Unix a signal death leaves `code()` == None;
/// report the signal number instead of a meaningless `-1` so a SIGKILL/SIGSEGV
/// is distinguishable from a normal non-zero exit in the logs.
fn exit_status_desc(st: &ExitStatus) -> String {
    match st.code() {
        Some(c) => format!("status {c}"),
        None => {
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                match st.signal() {
                    Some(sig) => format!("signal {sig}"),
                    None => "unknown status".to_string(),
                }
            }
            #[cfg(not(unix))]
            {
                "unknown status".to_string()
            }
        }
    }
}

pub(super) fn supervise_children(
    children: Vec<SupervisedChild>,
    elastic: Option<ElasticSupervision>,
) -> Option<TensorError> {
    if children.is_empty() {
        return None;
    }
    // Kill-all signal, reuse-safe by construction. Instead of main signalling
    // raw snapshotted PIDs (which race PID recycling — a child can exit, be
    // reaped, and have its PID handed to an unrelated process before the
    // signal lands), main sets this flag and each watcher SIGKILLs its OWN,
    // still-unreaped `Child`. The kernel pins that PID as a zombie until the
    // owning watcher reaps it, and `Child::kill()` refuses to signal an
    // already-reaped child — so a kill can never hit a recycled PID.
    let kill_all = Arc::new(AtomicBool::new(false));

    let (tx, rx) =
        mpsc::channel::<(String, usize, Vec<usize>, std::io::Result<ExitStatus>)>();
    let mut watchers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(children.len());
    let mut all_forwarders: Vec<thread::JoinHandle<()>> = Vec::new();
    for (host, lr, granks, mut child, fwd) in children {
        all_forwarders.extend(fwd);
        let txc = tx.clone();
        let kill_flag = Arc::clone(&kill_all);
        watchers.push(thread::spawn(move || {
            // Poll instead of a blocking `wait()` so the kill-all flag can be
            // observed while the child still runs. `kill()` goes through the
            // owned `Child` (never a raw PID), so it is immune to PID reuse and
            // is only ever issued before this watcher reaps the child.
            let mut killed = false;
            let st = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => {
                        if !killed && kill_flag.load(Ordering::SeqCst) {
                            // SIGKILL (std has no graceful terminate without
                            // libc). Kill-all is only the fatal pre-formation /
                            // no-coordinator teardown; remote children still
                            // tear down gracefully via their ssh HUP trap when
                            // their local ssh process dies here.
                            let _ = child.kill();
                            killed = true;
                        }
                        thread::sleep(WATCH_POLL_INTERVAL);
                    }
                    Err(e) => break Err(e),
                }
            };
            // Channel send only fails if the receiver was dropped, which
            // only happens after the main loop has finished collecting
            // every expected event. Treat as best-effort.
            let _ = txc.send((host, lr, granks, st));
        }));
    }
    // Drop the producer handle held by main so `rx.recv()` terminates
    // when every watcher has finished.
    drop(tx);

    let mut any_failure: Option<TensorError> = None;
    let mut finished: std::collections::HashSet<(String, usize)> =
        std::collections::HashSet::new();
    let mut terminated_peers = false;
    let mut tolerated_deaths: usize = 0;
    while let Ok((host, lr, granks, st)) = rx.recv() {
        finished.insert((host.clone(), lr));
        let failure_msg: Option<String> = match st {
            Ok(s) if s.success() => None,
            Ok(s) => Some(format!(
                "cluster launcher: {} exited with {}",
                child_label(lr, &host),
                exit_status_desc(&s),
            )),
            Err(e) => Some(format!(
                "cluster launcher: wait on {} failed: {e}",
                child_label(lr, &host),
            )),
        };
        if let Some(msg) = failure_msg {
            // ELASTIC MEMBERSHIP: once the cohort has formed, a child
            // exit is a membership event, not a run failure — report
            // the child's global ranks to the coordinator (which
            // redistributes work, fails over callback roles, and
            // decides recoverability against max_failure) and keep
            // supervising the survivors. A single rank vanishing from a
            // large collective must never kill the training; the
            // coordinator owns the stop decision. Pre-formation (or
            // with no coordinator) the legacy first-failure kill-all
            // stands: a half-formed NCCL cohort blocks in connect-retry
            // with no comm to abort and no rebuild machinery running.
            let elastic_active = elastic
                .as_ref()
                .map(|e| e.cohort_formed.load(std::sync::atomic::Ordering::SeqCst))
                .unwrap_or(false);
            if elastic_active {
                let e = elastic.as_ref().expect("elastic_active implies Some");
                eprintln!(
                    "{msg} — tolerating (elastic membership): reporting rank(s) \
                     {granks:?} dead; the coordinator redistributes their work"
                );
                tolerated_deaths += 1;
                if let Ok(mut q) = e.reported_deaths.lock() {
                    q.extend(granks.iter().copied());
                }
            } else if any_failure.is_none() {
                any_failure = Some(TensorError::new(&msg));
                if !terminated_peers {
                    terminated_peers = true;
                    // Signal every watcher to SIGKILL its own still-running
                    // child. Reuse-safe (see `kill_all`): no raw PID leaves
                    // this loop. Watchers already past their reap ignore it.
                    eprintln!(
                        "cluster launcher: peer failure — terminating all \
                         still-running ranks"
                    );
                    kill_all.store(true, Ordering::SeqCst);
                }
            } else {
                eprintln!("{msg}");
            }
        }
    }

    for w in watchers {
        let _ = w.join();
    }
    for f in all_forwarders {
        let _ = f.join();
    }

    // Elastic verdict: the launcher's exit status reflects the RUN, not
    // individual children. Deaths within tolerance on a run that drained
    // to completion are a degraded-but-valid result.
    if any_failure.is_none() {
        if let Some(e) = elastic.as_ref() {
            let dead = e.dead_ranks.dead_count();
            let limit = e.max_failure.map(|t| t.limit_for(e.world_size));
            if dead >= e.world_size {
                any_failure = Some(TensorError::new(
                    "cluster launcher: every rank was lost; consensus checkpoint \
                     saved if a save path was armed",
                ));
            } else if let Some(l) = limit {
                if dead >= l {
                    any_failure = Some(TensorError::new(&format!(
                        "cluster launcher: max_failure exceeded ({dead}/{} ranks \
                         dead, threshold {l}); coordinator dispatched \
                         save-and-shutdown — consensus checkpoint saved if a \
                         save path was armed",
                        e.world_size,
                    )));
                }
            }
            if any_failure.is_none() && (dead > 0 || tolerated_deaths > 0) {
                eprintln!(
                    "cluster launcher: run completed DEGRADED — {dead} of {} \
                     ranks lost along the way (tolerated by elastic \
                     membership); survivors carried the full workload",
                    e.world_size,
                );
            }
        }
    }
    any_failure
}

/// Context handed to [`supervise_children`] when a coordinator is
/// running: lets child supervision defer rank-death decisions to the
/// coordinator's elastic membership instead of the legacy
/// first-failure kill-all.
pub(super) struct ElasticSupervision {
    /// Fast death reports into the coordinator tick (same side-effect
    /// chain as heartbeat-staleness detection, minus the 30s wait).
    pub reported_deaths: crate::distributed::cluster_coordinator::ReportedDeaths,
    /// Shared ledger — read for the end-of-run verdict (includes
    /// staleness-declared deaths supervision never saw as child exits).
    pub dead_ranks: std::sync::Arc<crate::distributed::controller::DeadRanks>,
    /// User stop-threshold; `None` = tolerate any partial loss (only
    /// all-ranks-lost fails the run).
    pub max_failure: Option<crate::distributed::max_failure::MaxFailureThreshold>,
    pub world_size: usize,
    /// Cohort-formation gate: elastic behavior only after the cohort
    /// bootstrapped (NCCL rendezvous complete / CPU immediately).
    pub cohort_formed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}


/// Build the `Command` that fork+execs a local rank child. Sets all the
/// env vars the rank-side `LocalCluster::from_env` + `dispatch` expect,
/// and strips `FLODL_FULL_CLUSTER_JSON` so the child detects `Role::Rank`.
pub(super) fn build_local_spawn_command(
    exe: &std::path::Path,
    user_args: &[String],
    envelope_hex: &str,
    local_rank: usize,
    local_phys_device: Option<u8>,
) -> Command {
    let mut cmd = Command::new(exe);
    cmd.args(user_args)
        .env(
            crate::distributed::cluster::ENV_CLUSTER_JSON,
            envelope_hex,
        )
        .env(
            crate::distributed::cluster::ENV_LOCAL_RANK,
            local_rank.to_string(),
        )
        .env_remove(ENV_FULL_CLUSTER_JSON);
    if let Some(phys) = local_phys_device {
        cmd.env("CUDA_VISIBLE_DEVICES", phys.to_string());
    }
    cmd
}

/// Build the `Command` that fork+execs a local per-host relay child.
/// Sets `FLODL_RELAY_JSON` (so the child detects `Role::Relay`) and strips
/// the launcher/rank role env vars. No CUDA scoping — the relay touches no
/// GPU.
pub(super) fn build_local_relay_command(
    exe: &std::path::Path,
    user_args: &[String],
    relay_spec_hex: &str,
) -> Command {
    let mut cmd = Command::new(exe);
    cmd.args(user_args)
        .env(super::ENV_RELAY_JSON, relay_spec_hex)
        .env_remove(ENV_FULL_CLUSTER_JSON)
        .env_remove(crate::distributed::cluster::ENV_CLUSTER_JSON)
        .env_remove(crate::distributed::cluster::ENV_LOCAL_RANK);
    cmd
}

/// Build the bash command shipped via ssh to run a remote per-host relay
/// child. Mirrors [`build_remote_bash_command`] but exports
/// `FLODL_RELAY_JSON` instead of the rank envelope/slot and never scopes
/// CUDA.
pub(super) fn build_remote_relay_bash_command(
    path: &str,
    fdl_cmd: &str,
    user_args: &[String],
    cluster_env: &std::collections::BTreeMap<String, String>,
    host_env: &std::collections::BTreeMap<String, String>,
    prebuild: Option<&PerHostPrebuild>,
) -> String {
    let host_env_has_ld_path = host_env.contains_key("LD_LIBRARY_PATH");
    let mut s = String::with_capacity(256);
    // SALT HYGIENE: the relay spec carries the session salt too — pipe it
    // via stdin rather than the argv-visible command string (see
    // `build_remote_bash_command`).
    s.push_str("IFS= read -r __FLODL_ENVELOPE\n");
    s.push_str("cd ");
    let remote_cwd: String = match prebuild {
        Some(pb) if !pb.cwd_subpath.is_empty() => {
            format!("{}/{}", path.trim_end_matches('/'), pb.cwd_subpath)
        }
        _ => path.to_string(),
    };
    s.push_str(&shell_quote(&remote_cwd));
    s.push_str(" && ");
    s.push_str(super::ENV_RELAY_JSON);
    s.push_str("=\"$__FLODL_ENVELOPE\"");
    // Forward verbosity so the relay's `-vvv` prof lines reach the user.
    if let Ok(v) = std::env::var(crate::log::ENV_VAR) {
        s.push(' ');
        s.push_str(crate::log::ENV_VAR);
        s.push('=');
        s.push_str(&shell_quote(&v));
    }
    if let Some(pb) = prebuild {
        if !host_env_has_ld_path && !cluster_env.contains_key("LD_LIBRARY_PATH") {
            s.push(' ');
            s.push_str("LD_LIBRARY_PATH=");
            s.push_str(&shell_quote(&pb.ld_library_path));
        }
    }
    for (k, v) in cluster_env {
        s.push(' ');
        s.push_str(k);
        s.push('=');
        s.push_str(&shell_quote(v));
    }
    for (k, v) in host_env {
        s.push(' ');
        s.push_str(k);
        s.push('=');
        s.push_str(&shell_quote(v));
    }
    if let Some(pb) = prebuild {
        s.push(' ');
        let abs_bin = format!("{}/{}", path.trim_end_matches('/'), pb.bin);
        s.push_str(&shell_quote(&abs_bin));
    } else {
        s.push_str(" fdl ");
        s.push_str(&shell_quote(fdl_cmd));
    }
    for a in user_args {
        s.push(' ');
        s.push_str(&shell_quote(a));
    }
    // Same trap wrapper as ranks: forward signals to the backgrounded
    // relay so launcher death / SIGTERM reaches it cleanly.
    s.push_str(" &\n");
    s.push_str("__flodl_pid=$!\n");
    s.push_str("trap 'kill -TERM \"$__flodl_pid\" 2>/dev/null' HUP TERM INT\n");
    s.push_str("wait \"$__flodl_pid\"\n");
    s.push_str("exit $?\n");
    s
}

/// Build the `Command` that ssh's into a remote host and runs the given
/// bash command string.
///
/// Reads connection details from the host:
/// - `host.ssh_port` → `-p <port>` (default: system ssh's 22)
/// - `host.ssh_user` → `-l <user>` (default: current user)
/// - `host.ssh_identity_file` → `-i <path>` (default: `~/.ssh/config` rules)
/// - `host.ssh_options` → `-o Key=Value ...` (pass-through, in order)
/// - `host.ssh.as_deref().unwrap_or(&host.host)` → the connect target
///
/// All fields are optional; when absent, the corresponding flag is
/// omitted and system ssh's defaults / `~/.ssh/config` rules apply —
/// preserving backward compat for configs that pre-date the new
/// fields.
pub(super) fn build_ssh_spawn_command(host: &FullWorker, remote_cmd: &str) -> Command {
    let ssh_target = host.ssh_target();
    let mut c = Command::new("ssh");
    c.args(SSH_OPTS);
    if let Some(p) = host.ssh.as_ref().and_then(|s| s.port) {
        c.arg("-p").arg(p.to_string());
    }
    if let Some(u) = host.ssh.as_ref().and_then(|s| s.user.as_deref()) {
        c.arg("-l").arg(u);
    } else if let Ok(host_user) = std::env::var("FLODL_HOST_USER") {
        // When the per-host ssh.user is unset, fall back to the
        // controller's OS user (set by fdl-cli via FLODL_HOST_USER).
        // Bridges the docker container's stock `ubuntu` UID-1000 user
        // vs. the user's actual remote account on cluster hosts.
        let trimmed = host_user.trim();
        if !trimmed.is_empty() {
            c.arg("-l").arg(trimmed);
        }
    }
    if let Some(i) = host.ssh.as_ref().and_then(|s| s.identity_file.as_deref()) {
        c.arg("-i").arg(i);
    }
    if let Some(opts) = host.ssh.as_ref().map(|s| &s.options) {
        for opt in opts {
            c.arg("-o").arg(opt);
        }
    }
    c.arg(ssh_target).arg(remote_cmd);
    c
}

/// Pipe the hex-encoded cluster/relay envelope to a REMOTE child's stdin
/// and close it. The remote bash command reads one line
/// (`IFS= read -r __FLODL_ENVELOPE`) and expands it into the rank/relay
/// binary's environment — so the session salt inside the envelope never
/// appears in the remote shell's argv (`ps` / `/proc/<pid>/cmdline`). The
/// payload is small (well under the 64 KiB pipe buffer) and the remote
/// `read` runs first thing after ssh connects, so the single write never
/// blocks; a failed write is non-fatal (the child will fail its own
/// handshake loudly). Local children carry the envelope via `Command::env`
/// and never reach here.
pub(super) fn pipe_envelope_to_child(child: &mut std::process::Child, envelope_hex: &str) {
    use std::io::Write;
    if let Some(mut sin) = child.stdin.take() {
        let _ = writeln!(sin, "{envelope_hex}");
        // drop(sin) → EOF, so a remote `read` that somehow got two lines
        // still terminates.
    }
}

/// Best-effort SSH pkill of any leftover `abs_bin` process on `host`.
///
/// Fired twice per remote host:
/// - **Pre-spawn**: clears orphans from a previous botched session before
///   this run's ranks come up. Guarantees a fresh start.
/// - **Post-exit**: belt-and-braces cleanup after the launcher's main
///   supervise loop returns. Catches the case where the remote bash trap
///   wrapper didn't fire (binary SIGKILL'd by itself, etc.).
///
/// Silent on failure: pkill returns 1 when nothing matches (the no-orphan
/// case, expected on most calls), and we explicitly mask that with the
/// trailing `true`. The remote bash trap wrapper is the additional
/// backstop on connection-drop, so this helper failing is non-fatal.
///
/// Sends SIGTERM first, sleeps briefly to let cooperative shutdown run,
/// then SIGKILL for anything that ignored SIGTERM (wedged in a syscall,
/// etc.).
pub(super) fn cleanup_remote_host(host: &FullWorker, abs_bin: &str) {
    let q = shell_quote(abs_bin);
    let payload = format!(
        "pkill -TERM -f {q} >/dev/null 2>&1; sleep 1; \
         pkill -KILL -f {q} >/dev/null 2>&1; true",
    );
    let _ = build_ssh_spawn_command(host, &payload)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Fire [`cleanup_remote_host`] on every entry in parallel and join.
///
/// Sequential SSH would add ~1-2s of handshake per host. Parallel keeps
/// the pre-spawn / post-exit cleanup near-constant in host count.
pub(super) fn cleanup_remote_hosts_parallel(remotes: Vec<(FullWorker, String)>) {
    let handles: Vec<thread::JoinHandle<()>> = remotes
        .into_iter()
        .map(|(host, abs_bin)| {
            thread::spawn(move || {
                cleanup_remote_host(&host, &abs_bin);
            })
        })
        .collect();
    for h in handles {
        let _ = h.join();
    }
}

/// Build the bash command shipped via ssh to the remote.
///
/// Single level of shell quoting: ssh delivers the string verbatim to
/// the remote login shell, which parses it once. Every interpolated
/// value is single-quoted via [`shell_quote`]. `exec` replaces the
/// bash process so the remote returns fdl's exit code directly.
///
/// Mirrors fdl-cli's `build_remote_command` exactly (this is the move
/// of that logic into flodl proper, per the 4b boundary lift).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_remote_bash_command(
    path: &str,
    host_name: &str,
    local_rank: usize,
    overlay_env: Option<&str>,
    fdl_cmd: &str,
    user_args: &[String],
    cluster_env: &std::collections::BTreeMap<String, String>,
    host_env: &std::collections::BTreeMap<String, String>,
    local_phys_device: Option<u8>,
    prebuild: Option<&PerHostPrebuild>,
) -> String {
    use crate::distributed::cluster::{ENV_CLUSTER_JSON, ENV_HOST_OVERRIDE, ENV_LOCAL_RANK};

    // host_env's LD_LIBRARY_PATH (if user-set) wins over the prebuild
    // default; lets the user augment with bare-metal libnccl paths
    // (e.g. `/usr/local/lib`) without losing the auto-derived libtorch
    // entry. Detected up-front so the LD_LIBRARY_PATH= emission can
    // skip itself when host_env already provides one.
    let host_env_has_ld_path = host_env.contains_key("LD_LIBRARY_PATH");

    let mut s = String::with_capacity(
        256 + user_args.iter().map(|a| a.len() + 4).sum::<usize>(),
    );
    // SALT HYGIENE: the rank envelope carries the session salt (the HMAC
    // key). Splicing it into the command string would leave it in the
    // remote shell's argv — world-readable via `ps` / `/proc/<pid>/cmdline`
    // for the whole run. Instead the launcher pipes the hex envelope on
    // this ssh child's STDIN; we read it into a shell var here and expand
    // it into the child's ENVIRONMENT (env-assignment prefix), which is
    // only owner/root-readable via `/proc/<pid>/environ`. `-r` + empty IFS:
    // the hex is one whitespace-free line, read verbatim.
    s.push_str("IFS= read -r __FLODL_ENVELOPE\n");
    // Pick the remote cwd. With a prebuild envelope, the controller
    // built the command from `<project_root>/<cwd_subpath>` (e.g.
    // `ddp-bench/`) so the remote should execute from the same offset
    // — relative paths the binary expects (`data/cifar10/`, default
    // `--output runs/`) only resolve correctly under that subpath.
    // Without a prebuild envelope (legacy `fdl <cmd>` re-entry path)
    // we cd to `host.path` and let the remote `fdl` walk the same
    // overlay-resolved cmd cwd itself.
    s.push_str("cd ");
    let remote_cwd: String = match prebuild {
        Some(pb) if !pb.cwd_subpath.is_empty() => {
            format!("{}/{}", path.trim_end_matches('/'), pb.cwd_subpath)
        }
        _ => path.to_string(),
    };
    s.push_str(&shell_quote(&remote_cwd));
    s.push_str(" && ");
    s.push_str(ENV_CLUSTER_JSON);
    s.push_str("=\"$__FLODL_ENVELOPE\" ");
    s.push_str(ENV_HOST_OVERRIDE);
    s.push('=');
    s.push_str(&shell_quote(host_name));
    s.push(' ');
    s.push_str(ENV_LOCAL_RANK);
    s.push('=');
    s.push_str(&local_rank.to_string());
    s.push(' ');
    // Forward the launcher's verbosity so `-vvv` (FLODL_VERBOSITY)
    // reaches the remote worker/coordinator processes — the local
    // spawn path inherits it via the process env, but SSH starts a
    // fresh environment, so without this the prof instrumentation can
    // never be enabled on remote ranks. Emitted only when set, so
    // normal-verbosity runs leave the remote default untouched.
    if let Ok(v) = std::env::var(crate::log::ENV_VAR) {
        s.push(' ');
        s.push_str(crate::log::ENV_VAR);
        s.push('=');
        s.push_str(&shell_quote(&v));
    }
    if let Some(phys) = local_phys_device {
        s.push(' ');
        s.push_str("CUDA_VISIBLE_DEVICES=");
        s.push_str(&phys.to_string());
    }
    // Auto-prepend the prebuild's LD_LIBRARY_PATH (if any) BEFORE
    // host_env / cluster_env, so the user can override it via
    // host.env: { LD_LIBRARY_PATH: ... } when they need a custom
    // value (e.g. bare-metal libnccl at /usr/local/lib).
    if let Some(pb) = prebuild {
        if !host_env_has_ld_path && !cluster_env.contains_key("LD_LIBRARY_PATH") {
            s.push(' ');
            s.push_str("LD_LIBRARY_PATH=");
            s.push_str(&shell_quote(&pb.ld_library_path));
        }
    }
    // Apply user-declared env: cluster-scope first, host-scope second
    // (host overrides cluster for matching keys). In a shell assignment
    // prefix the LAST duplicate wins, so these could override the
    // built-ins above — safe ONLY because reserved keys (FLODL_*,
    // CUDA_VISIBLE_DEVICES) are rejected at config parse
    // (`parse_env_block`).
    for (k, v) in cluster_env {
        s.push(' ');
        s.push_str(k);
        s.push('=');
        s.push_str(&shell_quote(v));
    }
    for (k, v) in host_env {
        s.push(' ');
        s.push_str(k);
        s.push('=');
        s.push_str(&shell_quote(v));
    }
    if let Some(env) = overlay_env {
        s.push(' ');
        s.push_str(ENV_FDL_ENV);
        s.push('=');
        s.push_str(&shell_quote(env));
    }
    if let Some(pb) = prebuild {
        // Direct binary launch (no cargo, no rustc, no fdl re-entry on
        // the remote). `pb.bin` is relative to `host.path` (the project
        // root on the remote); we issued `cd <host.path>/<cwd_subpath>`
        // above, so use the absolute form to find the binary
        // independent of the current cwd offset.
        s.push(' ');
        let abs_bin = format!("{}/{}", path.trim_end_matches('/'), pb.bin);
        s.push_str(&shell_quote(&abs_bin));
    } else {
        s.push_str(" fdl ");
        s.push_str(&shell_quote(fdl_cmd));
    }
    for a in user_args {
        s.push(' ');
        s.push_str(&shell_quote(a));
    }
    // Trap wrapper: background the binary, set a signal trap that
    // forwards SIGHUP/SIGTERM/SIGINT to the child, wait for it, and
    // propagate the exit code. Replaces the previous bare `exec`,
    // which left no shell on the remote to react to a connection
    // drop, orphaning the binary on launcher death. With this
    // wrapper, sshd's SIGHUP-on-disconnect (delivered within
    // ServerAliveInterval * ServerAliveCountMax) reaches bash, which
    // then signals the binary cleanly.
    s.push_str(" &\n");
    s.push_str("__flodl_pid=$!\n");
    // TERM, then escalate to KILL after a grace period — INSIDE the
    // trap. A rank wedged in an uninterruptible CUDA ioctl (or any
    // stuck signal state) ignores TERM forever, and the only cleanup
    // pass that escalated to KILL ran from a LIVE launcher; on launcher
    // death nothing on the remote ever escalated, leaving the orphan
    // holding its ports and its GPU.
    s.push_str(
        "trap 'kill -TERM \"$__flodl_pid\" 2>/dev/null; \
( sleep 10; kill -KILL \"$__flodl_pid\" 2>/dev/null ) &' HUP TERM INT\n",
    );
    s.push_str("wait \"$__flodl_pid\"\n");
    s.push_str("exit $?\n");
    s
}

/// Single-quote a string for shell consumption. Internal single quotes
/// are escaped via the `'\''` idiom (close, backslash-escape, reopen).
/// Same implementation as fdl-cli's; kept as a private helper here
/// rather than introducing a shared utilities crate.
pub(super) fn shell_quote(s: &str) -> String {
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

/// Build the slim per-host envelope JSON the rank process consumes via
/// [`LocalCluster::from_env`].
///
/// [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env
pub(super) fn build_slim_envelope_for(full: &FullCluster, worker: &FullWorker) -> serde_json::Value {
    use serde_json::Value;
    let mut host_obj = serde_json::Map::new();
    host_obj.insert("host".into(), Value::String(worker.host.clone()));
    host_obj.insert(
        "ranks".into(),
        Value::Array(worker.ranks.iter().map(|r| Value::from(*r)).collect()),
    );
    host_obj.insert(
        "local_devices".into(),
        match &worker.local_devices {
            None => Value::String("all".into()),
            Some(v) => Value::Array(v.iter().map(|d| Value::from(*d)).collect()),
        },
    );
    host_obj.insert(
        "nccl_socket_ifname".into(),
        Value::String(worker.nccl_socket_ifname.clone()),
    );
    host_obj.insert("path".into(), Value::String(worker.path.clone()));
    if let Some(a) = &worker.arch {
        host_obj.insert("arch".into(), Value::String(a.clone()));
    }

    let mut controller_obj = serde_json::Map::new();
    controller_obj.insert("host".into(), Value::String(full.controller.host.clone()));
    controller_obj.insert("port".into(), Value::from(full.controller.port));

    let mut envelope = serde_json::Map::new();
    envelope.insert("controller".into(), Value::Object(controller_obj));
    envelope.insert("world_size".into(), Value::from(full.world_size()));
    envelope.insert("num_workers".into(), Value::from(full.workers.len()));
    envelope.insert("worker".into(), Value::Object(host_obj));
    envelope.insert(
        "salt".into(),
        Value::String(crate::distributed::wire::salt_to_hex(&full.salt)),
    );
    Value::Object(envelope)
}

/// Forward a child stream line-by-line with a prefix, mirroring
/// fdl-cli's launcher behavior. `to_stderr=true` routes to stderr (per
/// `feedback_docker_stdout_buffering` for debug-level output), else
/// stdout.
pub(super) fn forward_lines<R: std::io::Read>(stream: R, prefix: String, to_stderr: bool) {
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        match line {
            Ok(l) => {
                if to_stderr {
                    eprintln!("{prefix}{l}");
                } else {
                    println!("{prefix}{l}");
                }
            }
            // A non-UTF8 line is recoverable: skip it and keep forwarding the
            // rest of the child's output (a single stray byte must not blind us
            // to everything after it). A genuine IO error (pipe torn down) is
            // terminal — stop, or we'd spin re-hitting the same error.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => continue,
            Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // L7: the relay's sentinel local-rank must render as "relay of <host>",
    // not the raw usize::MAX integer, in supervision diagnostics.
    #[test]
    fn child_label_relay_sentinel_reads_as_relay() {
        assert_eq!(child_label(RELAY_RANK_SENTINEL, "hostA"), "relay of hostA");
        assert_eq!(child_label(3, "hostB"), "rank 3 of hostB");
    }

    // L6b: a signal death leaves ExitStatus::code() == None; report the signal
    // number instead of a meaningless -1.
    #[test]
    fn exit_status_desc_reports_code_and_signal() {
        let ok = Command::new("sh")
            .args(["-c", "exit 7"])
            .status()
            .expect("spawn sh");
        assert_eq!(exit_status_desc(&ok), "status 7");

        // SIGKILL the child, then describe: code() is None, signal() is 9.
        let mut child = Command::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep");
        child.kill().expect("kill");
        let st = child.wait().expect("wait");
        assert_eq!(exit_status_desc(&st), "signal 9");
    }
}
