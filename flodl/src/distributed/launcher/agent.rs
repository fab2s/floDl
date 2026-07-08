//! Per-host worker agent: dial in, join, then run the host.
//!
//! The agent is the worker-side half of dial-in membership. It is the
//! ONLY process fan-out (or a cloud startup script, or a human shell)
//! needs to start on a worker host: it dials the controller's mux port
//! on the join channel, sends a hello, and — once the world is formed —
//! receives the exact artifacts the launcher used to ship at spawn time
//! (the slim per-host envelope and the relay spec) and spawns its relay
//! and rank children locally with them. Everything downstream of world
//! formation is byte-identical to the direct fan-out era; ranks never
//! know the join protocol exists.
//!
//! After spawning, the agent stays up as the host supervisor: it
//! forwards its children's output (prefixed, exactly as the launcher's
//! per-rank forwarders did), reports every rank child's exit upstream
//! as [`JoinMsgWire::RankExited`] (per-rank granularity for the
//! controller's elastic membership — the join connection's EOF alone
//! would only signal whole-host death), and tears the host down on a
//! controller [`JoinMsgWire::Abort`].
//!
//! The agent touches no CUDA: GPU inventory comes from
//! [`crate::sys::detect_gpus`] (nvidia-smi), honoring the "no CUDA
//! before `Trainer::run`" invariant — the agent process never trains.

use std::net::TcpStream;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::distributed::wire::{
    CHANNEL_MAGIC_JOIN, ControlFrame, JoinMsgWire, MsgKind, SESSION_SALT_BYTES,
    SessionSalt, connect_with_retry, salt_from_hex, scaled_deadline_secs,
    write_channel_magic, write_stall_timeout,
};
use crate::tensor::{Result, TensorError};

use super::spawn::{build_local_relay_command, build_local_spawn_command, forward_lines};
use super::ENV_AGENT_JSON;

/// Budget for the controller's reply to a hello (scaled). The accept /
/// reject decision is immediate on the controller; only the network sits
/// in between.
const JOIN_REPLY_TIMEOUT_SECS: u64 = 30;

/// Margin added to the controller-announced formation wait before the
/// agent gives up on `WorldFormed` (absorbs clock skew between the two
/// deadline clocks).
const FORMATION_WAIT_MARGIN_SECS: u64 = 30;

/// Poll cadence of the agent's child-supervision loop.
const SUPERVISE_POLL: Duration = Duration::from_millis(50);

/// Bootstrap spec for a worker agent, hex-encoded JSON in
/// [`ENV_AGENT_JSON`]. This is the WHOLE deployment payload of a
/// dial-in worker: the controller address, an optional credential, and
/// host-local scoping. Built by fan-out for managed hosts; a
/// self-deployed worker carries the same shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Logical worker host name (member identity in the join window).
    pub host: String,
    /// Controller address as seen FROM this host (a tunneled host dials
    /// its loopback end of the SSH forward).
    pub controller_host: String,
    /// Controller mux port.
    pub controller_port: u16,
    /// Pre-shared session salt (hex) for rig-mode admission. `None`
    /// means the controller runs open admission and hands the salt out
    /// in the accept reply.
    #[serde(default)]
    pub salt_hex: Option<String>,
    /// Physical CUDA device ids to run, one rank each. `None` means all
    /// GPUs visible on this host (resolved at agent start via
    /// `nvidia-smi`, never libtorch).
    #[serde(default)]
    pub local_devices: Option<Vec<u8>>,
    /// libtorch variant label for the hello (informational).
    #[serde(default)]
    pub libtorch: String,
    /// Dataset signature (64-char hex) for the hello. `None` sends
    /// all-zeros — the "no signature configured" convention shared with
    /// the rendezvous path.
    #[serde(default)]
    pub dataset_sig_hex: Option<String>,
}

impl AgentSpec {
    /// Read + parse the spec from [`ENV_AGENT_JSON`]. Loud errors.
    pub fn from_env() -> Result<Self> {
        let raw = std::env::var(ENV_AGENT_JSON).map_err(|e| {
            TensorError::new(&format!("cluster agent: {ENV_AGENT_JSON} unreadable: {e}"))
        })?;
        let bytes = crate::distributed::cluster::hex_decode(raw.trim())
            .map_err(|e| TensorError::new(&format!("cluster agent: spec hex-decode: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| TensorError::new(&format!("cluster agent: spec JSON parse: {e}")))
    }

    /// Hex-encode for [`ENV_AGENT_JSON`] (the spawner side).
    pub fn to_env_hex(&self) -> Result<String> {
        let json = serde_json::to_string(self).map_err(|e| {
            TensorError::new(&format!("cluster agent: spec JSON encode: {e}"))
        })?;
        Ok(crate::distributed::cluster::hex_encode(json.as_bytes()))
    }
}

/// Everything [`join_world`] brings back: the admitted identity and the
/// spawn artifacts, plus the live join connection (the host control
/// link for the rest of the run).
#[derive(Debug)]
pub(crate) struct JoinOutcome {
    /// Session salt (pre-shared, or received in the accept reply).
    pub salt: SessionSalt,
    /// Assigned global rank ids, one per local device.
    pub ranks: Vec<u32>,
    /// Hex-encoded slim per-host envelope for rank children.
    pub envelope_hex: String,
    /// Hex-encoded `RelaySpec` for the relay child; `None` on
    /// no-coordinator runs (no relay is spawned).
    pub relay_spec_hex: Option<String>,
    /// The join connection, kept open as the host control link.
    pub stream: TcpStream,
}

/// Client half of the join protocol: dial the controller, send `hello`,
/// wait through admission and world formation. Blocks up to the
/// controller-announced formation budget.
pub(crate) fn join_world(
    controller_host: &str,
    controller_port: u16,
    pre_shared: Option<SessionSalt>,
    hello: JoinMsgWire,
) -> Result<JoinOutcome> {
    use std::net::ToSocketAddrs;
    let addr = (controller_host, controller_port)
        .to_socket_addrs()
        .map_err(|e| {
            TensorError::new(&format!(
                "cluster agent: resolve {controller_host}:{controller_port}: {e}"
            ))
        })?
        .next()
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster agent: no address for {controller_host}:{controller_port}"
            ))
        })?;
    let mut stream = connect_with_retry(addr, "cluster agent join")?;
    let _ = stream.set_nodelay(true);
    stream
        .set_write_timeout(Some(write_stall_timeout()))
        .map_err(|e| TensorError::new(&format!("cluster agent: set_write_timeout: {e}")))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(scaled_deadline_secs(
            JOIN_REPLY_TIMEOUT_SECS,
        ))))
        .map_err(|e| TensorError::new(&format!("cluster agent: set_read_timeout: {e}")))?;
    crate::distributed::wire::warn_cleartext_public_peer(
        "cluster agent join",
        addr,
    );
    write_channel_magic(&mut stream, CHANNEL_MAGIC_JOIN)?;

    // Pre-admission frames ride the join key: the pre-shared salt in
    // rig mode, the all-zeros key under open admission (see the
    // membership module docs).
    let join_key = pre_shared.unwrap_or([0u8; SESSION_SALT_BYTES]);
    ControlFrame::encode(&join_key, MsgKind::Join, &hello)?.write_to(&mut stream)?;

    let reply = ControlFrame::read_from(&mut stream, &join_key)?.ok_or_else(|| {
        TensorError::new(
            "cluster agent: controller closed the connection before replying — \
             under pre-shared admission this usually means a session-salt \
             mismatch (frame authentication failed on the controller); \
             otherwise the controller went away mid-join",
        )
    })?;
    let (ranks, salt, formation_wait_secs) = match reply.decode::<JoinMsgWire>()? {
        JoinMsgWire::Accept { ranks, salt_hex, formation_wait_secs } => {
            let salt = match (pre_shared, salt_hex) {
                (Some(s), _) => s,
                (None, Some(hex)) => salt_from_hex(&hex)?,
                (None, None) => {
                    return Err(TensorError::new(
                        "cluster agent: open-admission accept carried no session \
                         salt — controller and worker disagree on the trust mode",
                    ));
                }
            };
            (ranks, salt, formation_wait_secs)
        }
        JoinMsgWire::Reject { reason } => {
            return Err(TensorError::new(&format!(
                "cluster agent: join REJECTED by the controller: {reason}"
            )));
        }
        other => {
            return Err(TensorError::new(&format!(
                "cluster agent: expected Accept or Reject, got {other:?}"
            )));
        }
    };
    eprintln!(
        "cluster agent: joined as rank(s) {ranks:?}; waiting for world \
         formation (up to {formation_wait_secs}s)"
    );

    // World formation can legitimately take the rest of the join window
    // — the controller told us how much budget is left.
    stream
        .set_read_timeout(Some(Duration::from_secs(
            formation_wait_secs.saturating_add(FORMATION_WAIT_MARGIN_SECS),
        )))
        .map_err(|e| TensorError::new(&format!("cluster agent: set_read_timeout: {e}")))?;
    let frame = ControlFrame::read_from(&mut stream, &salt)?.ok_or_else(|| {
        TensorError::new(
            "cluster agent: controller closed the connection while waiting for \
             world formation",
        )
    })?;
    match frame.decode::<JoinMsgWire>()? {
        JoinMsgWire::WorldFormed { envelope_hex, relay_spec_hex } => Ok(JoinOutcome {
            salt,
            ranks,
            envelope_hex,
            relay_spec_hex,
            stream,
        }),
        JoinMsgWire::Abort { reason } => Err(TensorError::new(&format!(
            "cluster agent: run aborted before world formation: {reason}"
        ))),
        other => Err(TensorError::new(&format!(
            "cluster agent: expected WorldFormed or Abort, got {other:?}"
        ))),
    }
}

/// Resolve the physical devices this worker offers: an explicit list
/// from the spec/config, or every GPU nvidia-smi sees. Returns the
/// device ids plus one inventory label per device. Resolution happens
/// HERE (on the worker) so `"all"` shorthands never cross the wire.
pub(crate) fn resolve_devices(requested: Option<&Vec<u8>>) -> (Vec<u8>, Vec<String>) {
    let detected = crate::sys::detect_gpus();
    let devices: Vec<u8> = match requested {
        Some(d) => d.clone(),
        None => detected.iter().map(|g| g.index).collect(),
    };
    let labels = devices
        .iter()
        .map(|d| {
            detected
                .iter()
                .find(|g| g.index == *d)
                .map(|g| g.name.clone())
                .unwrap_or_else(|| format!("cuda:{d}"))
        })
        .collect();
    (devices, labels)
}

/// Build the join hello for a worker described by `spec`, with devices
/// already resolved.
fn build_hello(spec: &AgentSpec, devices: &[u8], gpus: Vec<String>) -> Result<JoinMsgWire> {
    let dataset_sig: [u8; 32] = match &spec.dataset_sig_hex {
        None => [0u8; 32],
        Some(hex) => {
            let bytes = crate::distributed::cluster::hex_decode(hex.trim()).map_err(|e| {
                TensorError::new(&format!("cluster agent: dataset_sig hex-decode: {e}"))
            })?;
            bytes.try_into().map_err(|_| {
                TensorError::new("cluster agent: dataset_sig must be 32 bytes (64 hex chars)")
            })?
        }
    };
    Ok(JoinMsgWire::Hello {
        host: spec.host.clone(),
        local_devices: devices.to_vec(),
        gpus,
        libtorch: spec.libtorch.clone(),
        dataset_sig,
    })
}

/// Run this process as a worker agent: join the controller's window,
/// then spawn + supervise this host's relay and rank children until the
/// run ends. The caller (the dispatch site on `Role::Agent`) exits the
/// process when this returns.
pub fn run_agent() -> Result<()> {
    let spec = AgentSpec::from_env()?;
    let (devices, gpus) = resolve_devices(spec.local_devices.as_ref());
    if devices.is_empty() {
        return Err(TensorError::new(&format!(
            "cluster agent: host {:?} has no GPUs to offer (none detected and \
             no local_devices configured)",
            spec.host,
        )));
    }
    let pre_shared = spec
        .salt_hex
        .as_deref()
        .map(salt_from_hex)
        .transpose()?;

    eprintln!(
        "cluster agent: host {:?} dialing controller {}:{} with {} rank(s) \
         (devices {:?}, admission: {})",
        spec.host,
        spec.controller_host,
        spec.controller_port,
        devices.len(),
        devices,
        if pre_shared.is_some() { "pre-shared salt" } else { "open" },
    );

    let hello = build_hello(&spec, &devices, gpus)?;
    let outcome = join_world(
        &spec.controller_host,
        spec.controller_port,
        pre_shared,
        hello,
    )?;
    // Children inherit this process's environment (the fan-out ssh
    // command already applied cluster/host env blocks to it); no extra
    // per-child env is needed.
    let children = spawn_host_children(
        &spec.host,
        &devices,
        &outcome,
        &std::collections::BTreeMap::new(),
    )?;
    supervise(&spec.host, children, outcome.stream, &outcome.salt)
}

/// Join the window and spawn this host's children WITHOUT taking over
/// supervision — the launcher-local variant of the agent. The children
/// stay direct children of the calling (launcher) process, so its
/// existing supervision owns their lifecycle exactly as in the direct
/// fan-out era; the join connection is dropped once spawning is done
/// (the controller holds its own end, and local children need no
/// `RankExited` relay — the launcher watches them first-hand).
pub(crate) fn join_and_spawn_local(
    spec: AgentSpec,
    extra_env: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<HostChild>> {
    let (devices, gpus) = resolve_devices(spec.local_devices.as_ref());
    if devices.is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: local worker {:?} has no GPUs to offer (none \
             detected and no local_devices configured)",
            spec.host,
        )));
    }
    let pre_shared = spec
        .salt_hex
        .as_deref()
        .map(salt_from_hex)
        .transpose()?;
    let hello = build_hello(&spec, &devices, gpus)?;
    let outcome = join_world(
        &spec.controller_host,
        spec.controller_port,
        pre_shared,
        hello,
    )?;
    spawn_host_children(&spec.host, &devices, &outcome, extra_env)
}

/// One spawned host child, launcher-supervision-shaped.
pub(crate) struct HostChild {
    /// Display label for exit diagnostics.
    pub label: String,
    /// Local rank slot; [`super::spawn::RELAY_RANK_SENTINEL`] for the
    /// relay.
    pub slot: usize,
    /// Global rank this child carries; `None` for the relay.
    pub rank: Option<u32>,
    pub child: std::process::Child,
    pub forwarders: Vec<thread::JoinHandle<()>>,
}

/// Spawn the host's relay (when the world runs one) + rank children
/// from the formed-world artifacts. All-or-nothing: a mid-spawn failure
/// kills the already-spawned half before returning the error.
pub(crate) fn spawn_host_children(
    host: &str,
    devices: &[u8],
    outcome: &JoinOutcome,
    extra_env: &std::collections::BTreeMap<String, String>,
) -> Result<Vec<HostChild>> {
    if outcome.ranks.len() != devices.len() {
        return Err(TensorError::new(&format!(
            "cluster agent: controller assigned {} rank(s) for {} device(s) — \
             protocol violation",
            outcome.ranks.len(),
            devices.len(),
        )));
    }
    let exe = std::env::current_exe().map_err(|e| {
        TensorError::new(&format!("cluster agent: current_exe() failed: {e}"))
    })?;
    let user_args: Vec<String> = std::env::args().skip(1).collect();

    let mut children: Vec<HostChild> = Vec::with_capacity(devices.len() + 1);
    let spawn_result: Result<()> = (|| {
        // Relay first, mirroring the direct fan-out spawn order (ranks
        // dial its loopback channels). Absent on no-coordinator runs.
        if let Some(relay_spec_hex) = &outcome.relay_spec_hex {
            let mut cmd = build_local_relay_command(&exe, &user_args, relay_spec_hex);
            for (k, v) in extra_env {
                cmd.env(k, v);
            }
            cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
            let mut child = cmd.spawn().map_err(|e| {
                TensorError::new(&format!("cluster agent: spawn relay failed: {e}"))
            })?;
            let prefix = format!("[{host}:relay] ");
            let mut forwarders = Vec::with_capacity(2);
            if let Some(out) = child.stdout.take() {
                let p = prefix.clone();
                forwarders.push(thread::spawn(move || forward_lines(out, p, false)));
            }
            if let Some(err) = child.stderr.take() {
                let p = prefix;
                forwarders.push(thread::spawn(move || forward_lines(err, p, true)));
            }
            children.push(HostChild {
                label: format!("relay of {host}"),
                slot: super::spawn::RELAY_RANK_SENTINEL,
                rank: None,
                child,
                forwarders,
            });
        }

        for (local_rank, (&phys, &grank)) in
            devices.iter().zip(outcome.ranks.iter()).enumerate()
        {
            let mut cmd = build_local_spawn_command(
                &exe,
                &user_args,
                &outcome.envelope_hex,
                local_rank,
                Some(phys),
            );
            for (k, v) in extra_env {
                cmd.env(k, v);
            }
            cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
            let mut child = cmd.spawn().map_err(|e| {
                TensorError::new(&format!(
                    "cluster agent: spawn rank {grank} (device {phys}) failed: {e}"
                ))
            })?;
            let prefix = format!("[{host}:{phys}:r{grank}] ");
            let mut forwarders = Vec::with_capacity(2);
            if let Some(out) = child.stdout.take() {
                let p = prefix.clone();
                forwarders.push(thread::spawn(move || forward_lines(out, p, false)));
            }
            if let Some(err) = child.stderr.take() {
                let p = prefix;
                forwarders.push(thread::spawn(move || forward_lines(err, p, true)));
            }
            children.push(HostChild {
                label: format!("rank {grank} of {host}"),
                slot: local_rank,
                rank: Some(grank),
                child,
                forwarders,
            });
        }
        Ok(())
    })();
    if let Err(e) = spawn_result {
        eprintln!(
            "cluster agent: spawn failed; terminating {} already-spawned \
             child(ren): {e}",
            children.len(),
        );
        for mut c in children {
            let _ = c.child.kill();
            let _ = c.child.wait();
            for f in c.forwarders {
                let _ = f.join();
            }
        }
        return Err(e);
    }
    Ok(children)
}

/// Watch the children and the join connection until the host is done.
///
/// Exit-code contract: `Ok` when the AGENT did its job — children
/// spawned, every exit reported upstream, all children reaped — even if
/// some rank children failed (those are membership events the
/// controller already learned via `RankExited`; the launcher must not
/// read a whole-host death out of a single tolerated rank loss). `Err`
/// is reserved for host-level failures: relay death with ranks still
/// running, or a controller abort.
fn supervise(
    host: &str,
    mut children: Vec<HostChild>,
    stream: TcpStream,
    salt: &SessionSalt,
) -> Result<()> {
    // Reader thread on the join connection: a controller Abort (or the
    // connection dying — controller gone) flips the flag; the
    // supervision loop tears the host down. Blocking reads on a
    // dedicated thread avoid short-timeout frame reads entirely (a
    // partially read header would desync the channel).
    let abort = Arc::new(AtomicBool::new(false));
    let abort_reason = Arc::new(std::sync::Mutex::new(String::new()));
    let mut reader = stream.try_clone().map_err(|e| {
        TensorError::new(&format!("cluster agent: join stream try_clone: {e}"))
    })?;
    // The reply deadline from the join phase is still armed; supervision
    // reads must block indefinitely (training runs for hours).
    reader
        .set_read_timeout(None)
        .map_err(|e| TensorError::new(&format!("cluster agent: reader timeout reset: {e}")))?;
    let abort_r = Arc::clone(&abort);
    let reason_r = Arc::clone(&abort_reason);
    let salt_r = *salt;
    let reader_handle = thread::spawn(move || {
        loop {
            match ControlFrame::read_from(&mut reader, &salt_r) {
                Ok(Some(frame)) => match frame.decode::<JoinMsgWire>() {
                    Ok(JoinMsgWire::Abort { reason }) => {
                        if let Ok(mut r) = reason_r.lock() {
                            *r = reason;
                        }
                        abort_r.store(true, Ordering::SeqCst);
                        return;
                    }
                    Ok(other) => {
                        crate::verbose!(
                            "  cluster agent: unexpected control-link message \
                             {other:?}; ignoring"
                        );
                    }
                    Err(e) => {
                        eprintln!("cluster agent: control-link decode error: {e}");
                    }
                },
                // EOF: the run is over on the controller side. If our
                // children are still up, treat it as an abort (the
                // controller is the only thing they train against); if
                // they already exited, the loop below has already
                // drained and this flag is never read.
                Ok(None) => {
                    if let Ok(mut r) = reason_r.lock() {
                        *r = "controller closed the join connection".to_string();
                    }
                    abort_r.store(true, Ordering::SeqCst);
                    return;
                }
                Err(e) => {
                    if let Ok(mut r) = reason_r.lock() {
                        *r = format!("join connection error: {e}");
                    }
                    abort_r.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
    });

    let mut writer = stream;
    let mut rank_failures: usize = 0;
    let mut aborted = false;
    let mut relay_died_early = false;
    // `None` = reaped. The relay is slot 0 by construction.
    let mut live = children.len();
    let mut reaped: Vec<bool> = vec![false; children.len()];
    while live > 0 {
        if !aborted && abort.load(Ordering::SeqCst) {
            aborted = true;
            let reason = abort_reason
                .lock()
                .map(|r| r.clone())
                .unwrap_or_default();
            eprintln!(
                "cluster agent: tearing down host {host:?} ({reason}); killing \
                 {live} child(ren)"
            );
            for (i, c) in children.iter_mut().enumerate() {
                if !reaped[i] {
                    let _ = c.child.kill();
                }
            }
        }
        let mut progressed = false;
        for i in 0..children.len() {
            if reaped[i] {
                continue;
            }
            match children[i].child.try_wait() {
                Ok(Some(status)) => {
                    reaped[i] = true;
                    live -= 1;
                    progressed = true;
                    let code = status.code().unwrap_or(-1);
                    let clean = status.success();
                    if !clean {
                        eprintln!(
                            "cluster agent: {} exited with {status}",
                            children[i].label,
                        );
                    }
                    match children[i].rank {
                        Some(rank) => {
                            if !clean {
                                rank_failures += 1;
                            }
                            // Per-rank exit report — the controller's
                            // fast path to elastic membership decisions.
                            // Best-effort: near shutdown the controller
                            // may already be gone.
                            let msg = JoinMsgWire::RankExited { rank, code };
                            let _ = ControlFrame::encode(salt, MsgKind::Join, &msg)
                                .and_then(|f| f.write_to(&mut writer));
                        }
                        None => {
                            // Relay death with ranks still running is
                            // host-fatal: they just lost their only path
                            // to the controller.
                            if !clean && live > 0 && !aborted {
                                relay_died_early = true;
                                eprintln!(
                                    "cluster agent: relay died with {live} rank \
                                     child(ren) still running — tearing down \
                                     host {host:?}"
                                );
                                for (j, c) in children.iter_mut().enumerate() {
                                    if !reaped[j] {
                                        let _ = c.child.kill();
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    reaped[i] = true;
                    live -= 1;
                    progressed = true;
                    eprintln!(
                        "cluster agent: wait on {} failed: {e}",
                        children[i].label,
                    );
                }
            }
        }
        if !progressed {
            thread::sleep(SUPERVISE_POLL);
        }
    }
    for c in children {
        for f in c.forwarders {
            let _ = f.join();
        }
    }
    // All children are gone: close our side of the control link so the
    // controller sees a clean host EOF, and unblock the reader thread.
    let _ = writer.shutdown(std::net::Shutdown::Both);
    let _ = reader_handle.join();

    if aborted {
        let reason = abort_reason.lock().map(|r| r.clone()).unwrap_or_default();
        return Err(TensorError::new(&format!(
            "cluster agent: host {host:?} torn down: {reason}"
        )));
    }
    if relay_died_early {
        return Err(TensorError::new(&format!(
            "cluster agent: host {host:?} torn down: relay died mid-run"
        )));
    }
    if rank_failures > 0 {
        // Reported upstream per rank; the run-level verdict is the
        // controller's (elastic membership). The agent did its job.
        eprintln!(
            "cluster agent: host {host:?} finished DEGRADED — {rank_failures} \
             rank child(ren) exited non-zero (reported to the controller)"
        );
    } else {
        eprintln!("cluster agent: host {host:?} finished cleanly");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::wire::{expect_channel_magic, salt_to_hex};
    use std::net::TcpListener;

    fn test_salt() -> SessionSalt {
        [7u8; SESSION_SALT_BYTES]
    }

    fn test_hello() -> JoinMsgWire {
        JoinMsgWire::Hello {
            host: "worker-x".to_string(),
            local_devices: vec![0, 1],
            gpus: vec!["T".to_string(), "T".to_string()],
            libtorch: "builds/test".to_string(),
            dataset_sig: [0u8; 32],
        }
    }

    /// A fake controller serving exactly one join dial: consume the
    /// channel magic, read the hello with `key`, then play back
    /// `replies` (the first keyed with `key`, the rest with `salt` —
    /// the post-admission keying the real controller uses).
    fn fake_controller(
        key: SessionSalt,
        salt: SessionSalt,
        replies: Vec<JoinMsgWire>,
    ) -> (u16, std::thread::JoinHandle<JoinMsgWire>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            expect_channel_magic(&mut stream, CHANNEL_MAGIC_JOIN, "fake controller")
                .unwrap();
            let frame = ControlFrame::read_from(&mut stream, &key)
                .unwrap()
                .expect("hello frame");
            let hello: JoinMsgWire = frame.decode().unwrap();
            for (i, msg) in replies.iter().enumerate() {
                let frame_key = if i == 0 { key } else { salt };
                ControlFrame::encode(&frame_key, MsgKind::Join, msg)
                    .unwrap()
                    .write_to(&mut stream)
                    .unwrap();
            }
            hello
        });
        (port, handle)
    }

    #[test]
    fn agent_spec_round_trips_through_hex() {
        let spec = AgentSpec {
            host: "pascal".to_string(),
            controller_host: "192.168.122.1".to_string(),
            controller_port: 1337,
            salt_hex: Some(salt_to_hex(&test_salt())),
            local_devices: Some(vec![0, 1]),
            libtorch: "builds/sm61-sm120".to_string(),
            dataset_sig_hex: None,
        };
        let hex = spec.to_env_hex().unwrap();
        let bytes = crate::distributed::cluster::hex_decode(&hex).unwrap();
        let parsed: AgentSpec = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed, spec);

        // Minimal self-deploy shape: address only, everything else
        // defaulted (the "one address and a credential" deployment).
        let minimal: AgentSpec = serde_json::from_str(
            r#"{"host":"cloud-1","controller_host":"10.0.0.1","controller_port":1337}"#,
        )
        .unwrap();
        assert_eq!(minimal.salt_hex, None);
        assert_eq!(minimal.local_devices, None);
        assert_eq!(minimal.dataset_sig_hex, None);
    }

    #[test]
    fn join_world_open_admission_receives_salt_and_artifacts() {
        let salt = test_salt();
        let zero: SessionSalt = [0u8; SESSION_SALT_BYTES];
        let (port, controller) = fake_controller(
            zero,
            salt,
            vec![
                JoinMsgWire::Accept {
                    ranks: vec![3, 4],
                    salt_hex: Some(salt_to_hex(&salt)),
                    formation_wait_secs: 60,
                },
                JoinMsgWire::WorldFormed {
                    envelope_hex: "aa11".to_string(),
                    relay_spec_hex: Some("bb22".to_string()),
                },
            ],
        );
        let outcome = join_world("127.0.0.1", port, None, test_hello()).unwrap();
        assert_eq!(outcome.salt, salt);
        assert_eq!(outcome.ranks, vec![3, 4]);
        assert_eq!(outcome.envelope_hex, "aa11");
        assert_eq!(outcome.relay_spec_hex.as_deref(), Some("bb22"));
        // The controller saw the hello it was sent.
        assert_eq!(controller.join().unwrap(), test_hello());
    }

    #[test]
    fn join_world_pre_shared_keys_hello_with_the_salt() {
        let salt = test_salt();
        let (port, controller) = fake_controller(
            salt,
            salt,
            vec![
                JoinMsgWire::Accept {
                    ranks: vec![0],
                    salt_hex: None,
                    formation_wait_secs: 60,
                },
                JoinMsgWire::WorldFormed {
                    envelope_hex: String::new(),
                    relay_spec_hex: None,
                },
            ],
        );
        let outcome = join_world("127.0.0.1", port, Some(salt), test_hello()).unwrap();
        assert_eq!(outcome.salt, salt);
        controller.join().unwrap();
    }

    #[test]
    fn join_world_reject_and_abort_are_loud() {
        let salt = test_salt();
        let zero: SessionSalt = [0u8; SESSION_SALT_BYTES];
        let (port, controller) = fake_controller(
            zero,
            salt,
            vec![JoinMsgWire::Reject { reason: "dataset signature mismatch".to_string() }],
        );
        let err = join_world("127.0.0.1", port, None, test_hello())
            .unwrap_err()
            .to_string();
        assert!(err.contains("REJECTED"), "got: {err}");
        assert!(err.contains("dataset signature mismatch"), "got: {err}");
        controller.join().unwrap();

        let (port, controller) = fake_controller(
            zero,
            salt,
            vec![
                JoinMsgWire::Accept {
                    ranks: vec![0],
                    salt_hex: Some(salt_to_hex(&salt)),
                    formation_wait_secs: 60,
                },
                JoinMsgWire::Abort { reason: "quorum not met".to_string() },
            ],
        );
        let err = join_world("127.0.0.1", port, None, test_hello())
            .unwrap_err()
            .to_string();
        assert!(err.contains("aborted"), "got: {err}");
        assert!(err.contains("quorum not met"), "got: {err}");
        controller.join().unwrap();
    }

    #[test]
    fn join_world_open_accept_without_salt_is_a_trust_mode_error() {
        let salt = test_salt();
        let zero: SessionSalt = [0u8; SESSION_SALT_BYTES];
        let (port, controller) = fake_controller(
            zero,
            salt,
            vec![JoinMsgWire::Accept {
                ranks: vec![0],
                salt_hex: None,
                formation_wait_secs: 60,
            }],
        );
        let err = join_world("127.0.0.1", port, None, test_hello())
            .unwrap_err()
            .to_string();
        assert!(err.contains("trust mode"), "got: {err}");
        controller.join().unwrap();
    }
}
