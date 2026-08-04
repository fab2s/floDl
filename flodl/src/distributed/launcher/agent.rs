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
    /// Dataset source root on THIS host, resolved on the box itself
    /// (`fdl join`'s prepare phase mounts it when needed, then puts the
    /// resulting path here). Written into the host block of the
    /// controller-authored envelope so this host's ranks read it through
    /// the same `LocalCluster::data_path()` a fan-out rank uses.
    ///
    /// Set only for a walk-in: the controller never configured that
    /// host, so its roster entry carries no source root and has nothing
    /// to be overridden. `None` on the fan-out path, where the roster IS
    /// the authority.
    #[serde(default)]
    pub data_path: Option<String>,
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
    data_path: Option<&str>,
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
        JoinMsgWire::WorldFormed { envelope_hex, relay_spec_hex } => {
            // The controller authors dial addresses from ITS view of
            // the topology, but only THIS agent knows the address that
            // provably reaches the controller from THIS host — the one
            // the join it just completed used. A walk-in's tunnel
            // (`fdl join --ssh` forwards on an ephemeral local port)
            // or a NAT'd controller make the authored address plain
            // wrong on this host, and the relay dying on it takes the
            // whole host down. Rewrite both artifacts with the
            // join-verified address; fan-out agents rewrite to the
            // values already in place (their tunnels bind the
            // controller port itself), so this is a no-op there.
            //
            // A walk-in's dataset source root is the same shape of fact
            // — host-local truth the controller never configured — so it
            // rides the same rewrite.
            let envelope_hex = localize_envelope(
                &envelope_hex,
                controller_host,
                controller_port,
                data_path,
            )?;
            let relay_spec_hex = relay_spec_hex
                .map(|hex| {
                    rewrite_relay_controller(&hex, controller_host, controller_port)
                })
                .transpose()?;
            Ok(JoinOutcome {
                salt,
                ranks,
                envelope_hex,
                relay_spec_hex,
                stream,
            })
        }
        JoinMsgWire::Abort { reason } => Err(TensorError::new(&format!(
            "cluster agent: run aborted before world formation: {reason}"
        ))),
        other => Err(TensorError::new(&format!(
            "cluster agent: expected WorldFormed or Abort, got {other:?}"
        ))),
    }
}

/// Replace the parts of the slim envelope only this box can author: the
/// `controller.host` / `controller.port` the join just proved reachable,
/// and — for a walk-in — the `worker.data_path` its prepare phase
/// resolved (see the call site in [`join_world`]).
///
/// `data_path` of `None` leaves the envelope's own value alone, which is
/// what the fan-out path needs: there the roster IS the authority.
///
/// Loud on malformed artifacts — a truncated envelope must fail here,
/// named, not in a rank child's parser.
fn localize_envelope(
    envelope_hex: &str,
    host: &str,
    port: u16,
    data_path: Option<&str>,
) -> Result<String> {
    let bytes = crate::distributed::cluster::hex_decode(envelope_hex)
        .map_err(|e| TensorError::new(&format!("cluster agent: envelope hex: {e}")))?;
    let mut envelope: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| TensorError::new(&format!("cluster agent: envelope JSON: {e}")))?;
    let Some(controller) = envelope
        .get_mut("controller")
        .and_then(|c| c.as_object_mut())
    else {
        return Err(TensorError::new(
            "cluster agent: envelope carries no controller object",
        ));
    };
    controller.insert("host".into(), serde_json::Value::String(host.to_string()));
    controller.insert("port".into(), serde_json::Value::from(port));
    if let Some(data_path) = data_path {
        let Some(worker) = envelope.get_mut("worker").and_then(|w| w.as_object_mut())
        else {
            return Err(TensorError::new(
                "cluster agent: envelope carries no worker object",
            ));
        };
        worker.insert(
            "data_path".into(),
            serde_json::Value::String(data_path.to_string()),
        );
    }
    let json = serde_json::to_string(&envelope).map_err(|e| {
        TensorError::new(&format!("cluster agent: envelope re-encode: {e}"))
    })?;
    Ok(crate::distributed::cluster::hex_encode(json.as_bytes()))
}

/// Same rewrite for the relay spec: the relay is this host's out-dialer,
/// so it must dial the road the join proved works.
fn rewrite_relay_controller(
    relay_spec_hex: &str,
    host: &str,
    port: u16,
) -> Result<String> {
    let bytes = crate::distributed::cluster::hex_decode(relay_spec_hex)
        .map_err(|e| TensorError::new(&format!("cluster agent: relay spec hex: {e}")))?;
    let mut spec: super::RelaySpec = serde_json::from_slice(&bytes)
        .map_err(|e| TensorError::new(&format!("cluster agent: relay spec JSON: {e}")))?;
    spec.controller_host = host.to_string();
    spec.controller_port = port;
    let json = serde_json::to_string(&spec).map_err(|e| {
        TensorError::new(&format!("cluster agent: relay spec re-encode: {e}"))
    })?;
    Ok(crate::distributed::cluster::hex_encode(json.as_bytes()))
}

/// Resolve the physical devices this worker offers: an explicit list
/// from the spec/config, or every GPU detection sees for this build's
/// vendor (masks applied). Returns the
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
                .map(|g| {
                    format!(
                        "{} ({}GB, {})",
                        g.name,
                        g.total_memory_mb / 1024,
                        g.arch_label(),
                    )
                })
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
        spec.data_path.as_deref(),
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
        spec.data_path.as_deref(),
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

    /// Minimal controller-authored envelope, hex-encoded — the
    /// controller's (possibly wrong-for-this-host) dial address baked
    /// in, as `build_slim_envelope_for` does. The `worker` object is
    /// always present there, so it is present here.
    fn test_envelope_hex(host: &str, port: u16) -> String {
        let envelope = serde_json::json!({
            "controller": { "host": host, "port": port },
            "world_size": 2,
            "worker": { "host": "worker-x", "ranks": [3, 4] },
        });
        crate::distributed::cluster::hex_encode(envelope.to_string().as_bytes())
    }

    /// Controller-authored relay spec, hex-encoded.
    fn test_relay_hex(host: &str, port: u16) -> String {
        let spec = super::super::RelaySpec {
            host: "worker-x".to_string(),
            controller_host: host.to_string(),
            controller_port: port,
            ranks: vec![3, 4],
            salt_hex: salt_to_hex(&test_salt()),
            world_size: 2,
            data_channel: true,
            frame_ceiling_bytes: 1024,
        };
        crate::distributed::cluster::hex_encode(
            serde_json::to_string(&spec).unwrap().as_bytes(),
        )
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
            data_path: Some("/flodl/data".to_string()),
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
        // No source root declared on that box: the envelope's own value
        // stands, so the training binary keeps its default.
        assert_eq!(minimal.data_path, None);
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
                    // Controller-authored addresses deliberately WRONG
                    // for this host (the tunnel/NAT case): the agent
                    // must rewrite both with the address the join
                    // actually used.
                    envelope_hex: test_envelope_hex("192.168.1.1", 1337),
                    relay_spec_hex: Some(test_relay_hex("192.168.1.1", 1337)),
                },
            ],
        );
        let outcome =
            join_world("127.0.0.1", port, None, test_hello(), Some("/srv/corpus"))
                .unwrap();
        assert_eq!(outcome.salt, salt);
        assert_eq!(outcome.ranks, vec![3, 4]);
        // Join-verified address rewrite: the artifacts now dial where
        // THIS join provably reached the controller.
        let envelope: serde_json::Value = serde_json::from_slice(
            &crate::distributed::cluster::hex_decode(&outcome.envelope_hex).unwrap(),
        )
        .unwrap();
        assert_eq!(envelope["controller"]["host"], "127.0.0.1");
        assert_eq!(envelope["controller"]["port"], port);
        assert_eq!(envelope["world_size"], 2, "other fields untouched");
        // A walk-in's source root: resolved on this box, so this box
        // writes it into the envelope its ranks read.
        assert_eq!(envelope["worker"]["data_path"], "/srv/corpus");
        assert_eq!(envelope["worker"]["host"], "worker-x", "other fields untouched");
        let relay: super::super::RelaySpec = serde_json::from_slice(
            &crate::distributed::cluster::hex_decode(
                outcome.relay_spec_hex.as_deref().unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(relay.controller_host, "127.0.0.1");
        assert_eq!(relay.controller_port, port);
        assert_eq!(relay.ranks, vec![3, 4], "other fields untouched");
        // The controller saw the hello it was sent.
        assert_eq!(controller.join().unwrap(), test_hello());
    }

    /// The fan-out half of the same rewrite: a managed host's roster
    /// entry IS the authority on its source root, so an agent with
    /// nothing of its own must leave the controller's value in place.
    /// Getting this backwards would silently repoint every fan-out rank.
    #[test]
    fn localize_leaves_the_rosters_data_path_alone_when_the_box_has_none() {
        let envelope = serde_json::json!({
            "controller": { "host": "10.0.0.1", "port": 1337 },
            "worker": { "host": "exa", "data_path": "/flodl/data" },
        });
        let hex = crate::distributed::cluster::hex_encode(
            envelope.to_string().as_bytes(),
        );
        let out = localize_envelope(&hex, "127.0.0.1", 40123, None).unwrap();
        let got: serde_json::Value = serde_json::from_slice(
            &crate::distributed::cluster::hex_decode(&out).unwrap(),
        )
        .unwrap();
        assert_eq!(got["worker"]["data_path"], "/flodl/data");
        assert_eq!(got["controller"]["host"], "127.0.0.1");

        // ... and a walk-in's value wins over whatever the envelope
        // carried, because the controller cannot have known better.
        let out = localize_envelope(&hex, "127.0.0.1", 40123, Some("/srv/c")).unwrap();
        let got: serde_json::Value = serde_json::from_slice(
            &crate::distributed::cluster::hex_decode(&out).unwrap(),
        )
        .unwrap();
        assert_eq!(got["worker"]["data_path"], "/srv/c");
    }

    #[test]
    fn localize_is_loud_on_a_malformed_envelope() {
        let hex = crate::distributed::cluster::hex_encode(
            serde_json::json!({ "controller": { "host": "h", "port": 1 } })
                .to_string()
                .as_bytes(),
        );
        // No worker object: only reachable on a truncated envelope, and
        // it must name itself rather than surface in a rank's parser.
        let err = localize_envelope(&hex, "h", 1, Some("/srv/c"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no worker object"), "got: {err}");
        // Same envelope with nothing to localize stays fine.
        assert!(localize_envelope(&hex, "h", 1, None).is_ok());
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
                    envelope_hex: test_envelope_hex("10.0.0.1", 9000),
                    relay_spec_hex: None,
                },
            ],
        );
        let outcome = join_world("127.0.0.1", port, Some(salt), test_hello(), None).unwrap();
        assert_eq!(outcome.salt, salt);
        assert!(outcome.relay_spec_hex.is_none());
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
        let err = join_world("127.0.0.1", port, None, test_hello(), None)
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
        let err = join_world("127.0.0.1", port, None, test_hello(), None)
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
        let err = join_world("127.0.0.1", port, None, test_hello(), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("trust mode"), "got: {err}");
        controller.join().unwrap();
    }
}
