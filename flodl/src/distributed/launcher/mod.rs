//! Cluster launcher: role detection, dial-in membership, fan-out sugar,
//! controller orchestration.
//!
//! Slots transparently into [`Trainer::run`] / [`Trainer::builder`] on
//! cluster-mode startup. Each user-binary invocation routes to one of five
//! roles:
//!
//! - **Launcher**: the parent process that fdl-cli execs after parsing
//!   `fdl.yml`. Opens the membership join window (see the crate-internal
//!   `distributed::membership` module), fans out ONE worker
//!   agent per remote host over ssh (push-as-sugar: those agents dial back
//!   in and join like any self-deployed worker would; local hosts run the
//!   join in-process and keep their children direct), forms the world when
//!   the window closes, starts the controller-side infrastructure sized to
//!   it, ships each admitted worker its spawn artifacts, and supervises
//!   until every child exits.
//!
//! - **Agent**: a per-host worker process (the one thing a deployment has
//!   to start). Dials the controller's join channel, receives its rank
//!   assignment + the spawn artifacts at world formation, then spawns and
//!   supervises the host's relay + rank children. See [`run_agent`].
//!
//! - **Relay**: a per-host transport byte-router (spawned by the agent, or
//!   directly by the launcher for its local host). See [`run_relay`].
//!
//! - **Rank**: a spawned child running the user's training code. Inherits
//!   the slim per-host envelope and the rank-slot env var; the
//!   [`Trainer::run`] cluster-path logic handles the rest (rendezvous,
//!   `Ddp::wrap`, training loop). Envelopes are byte-identical whether the
//!   host was fan-out-managed or self-deployed — ranks never know the join
//!   protocol exists.
//!
//! - **Single-device**: no cluster envelope in env. Caller continues with
//!   today's single-device path. Bit-identical to pre-cluster behavior.
//!
//! # Wire protocol (env vars)
//!
//! One namespaced env var per spawned role; `dispatch` loud-errors on any
//! combination:
//!
//! - [`ENV_FULL_CLUSTER_JSON`] (`FLODL_INTERNAL_FULL_CLUSTER_JSON`): hex-encoded
//!   JSON of the *full* cluster topology (all hosts + ranks + devices).
//!   Set by fdl-cli when invoking the user binary as the launcher. The
//!   launcher reads it once to drive the window + fan-out; never
//!   propagated to children.
//!
//! - [`ENV_AGENT_JSON`] (`FLODL_INTERNAL_AGENT_JSON`): hex-encoded
//!   [`AgentSpec`] — controller address, optional pre-shared salt, device
//!   scoping. The whole deployment payload of a dial-in worker.
//!
//! - [`ENV_RELAY_JSON`] (`FLODL_INTERNAL_RELAY_JSON`): hex-encoded
//!   [`RelaySpec`], set on the relay child.
//!
//! - [`crate::distributed::cluster::ENV_CLUSTER_JSON`]
//!   (`FLODL_INTERNAL_CLUSTER_JSON`): hex-encoded slim per-host envelope, set
//!   on each rank child. Read by [`LocalCluster::from_env`].
//!
//! - [`crate::distributed::cluster::ENV_LOCAL_RANK`] (`FLODL_INTERNAL_LOCAL_RANK`):
//!   integer index into the slim envelope's `host.ranks`, set on each rank
//!   child. Read by [`crate::distributed::cluster::LocalCluster::my_rank`].
//!
//! Role detection table (see [`dispatch`]): exactly one of
//! agent/relay/full set → that role; slim+slot → rank; all unset →
//! single-device; anything else → loud error.
//!
//! [`Trainer::run`]: crate::distributed::Trainer::run
//! [`Trainer::builder`]: crate::distributed::Trainer::builder
//! [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env

use std::env;
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::thread;

use serde::{Deserialize, Serialize};

pub use crate::distributed::membership::JoinedMember;
use crate::distributed::relay::agent::{ChannelKind, RelayChannel};
use crate::distributed::relay::{RELAY_CONTROL_LOOPBACK_OFFSET, RELAY_DATA_LOOPBACK_OFFSET};
use crate::tensor::{Result, TensorError};

mod agent;
mod spawn;
#[cfg(test)]
mod tests;
mod types;

pub use agent::{AgentSpec, run_agent};
pub use types::{FullCluster, FullController, FullWorker, JoinKnobs, SshConfig};

use spawn::{
    AGENT_RANK_SENTINEL, ElasticSupervision, build_remote_agent_bash_command,
    build_slim_envelope_for, build_ssh_spawn_command, cleanup_remote_hosts_parallel, forward_lines,
    load_prebuild_envelope, supervise_children,
};

/// Cohort inventory captured at world formation (admission order): one
/// [`JoinedMember`] per admitted worker, carrying host, ranks, physical
/// devices, and the GPU labels from its join hello. The launcher is
/// once-per-process, so a process-wide slot fits.
static COHORT_INVENTORY: OnceLock<Vec<JoinedMember>> = OnceLock::new();

/// GPU inventory of the formed cohort, for post-run consumers running in
/// the launcher process (a bench harness writing a hardware header, run
/// summaries). Covers every admitted host — including remote GPUs this
/// process can never probe locally. `None` unless this process ran the
/// launcher role and a world formed.
pub fn cohort_inventory() -> Option<&'static [JoinedMember]> {
    COHORT_INVENTORY.get().map(|v| v.as_slice())
}

/// Environment variable carrying the *full* cluster topology to the
/// launcher process. Set by fdl-cli; consumed only by [`dispatch`]. Not
/// propagated to rank children (each child gets a slim per-host envelope
/// instead via `FLODL_INTERNAL_CLUSTER_JSON`).
pub const ENV_FULL_CLUSTER_JSON: &str = "FLODL_INTERNAL_FULL_CLUSTER_JSON";

/// Environment variable carrying the per-host relay spec (hex-encoded
/// JSON [`RelaySpec`]). Set by the launcher on the relay child it spawns
/// per host; consumed only by [`dispatch`] (→ [`Role::Relay`]) and
/// [`run_relay`]. Mutually exclusive with the launcher/rank env vars.
pub const ENV_RELAY_JSON: &str = "FLODL_INTERNAL_RELAY_JSON";

/// Environment variable carrying the per-host worker-agent bootstrap
/// spec (hex-encoded JSON [`AgentSpec`]). Set by the spawner of a
/// dial-in worker — fan-out for managed hosts, a startup script or a
/// human shell for self-deployed ones. Consumed only by [`dispatch`]
/// (→ [`Role::Agent`]) and [`run_agent`]. Mutually exclusive with every
/// other role env var.
pub const ENV_AGENT_JSON: &str = "FLODL_INTERNAL_AGENT_JSON";

/// Model-signature probe marker (`fdl join` sets it on a short-lived
/// re-invocation of the training binary, before the dial): when
/// present, `Trainer::run` / `into_worker` build the model on CPU,
/// print `flodl-model-sig: <64 hex>` on stdout and exit — before
/// auto-promote, before any cluster role, touching no CUDA context.
pub const ENV_MODEL_SIG_PROBE: &str = "FLODL_INTERNAL_MODEL_SIG_PROBE";

/// True when this process is `fdl join`'s model-signature probe.
///
/// The probe asks what the model IS, not whether this box can train it,
/// and it runs OUTSIDE any cluster: a `main()` that gates on the local
/// GPU count ahead of [`Trainer::run`] ("cpu-async needs 2+ GPUs, have
/// 1") therefore exits before the trainer can answer, and the join
/// proceeds without a signature. Let such a gate step aside when this
/// returns true; the same gate keeps its meaning for a real run.
///
/// ```no_run
/// use flodl::distributed::launcher::model_sig_probe_requested;
/// let in_cluster = std::env::var_os("FLODL_INTERNAL_CLUSTER_JSON").is_some()
///     || model_sig_probe_requested();
/// ```
///
/// [`Trainer::run`]: crate::distributed::Trainer::run
pub fn model_sig_probe_requested() -> bool {
    std::env::var_os(ENV_MODEL_SIG_PROBE).is_some()
}

/// Environment variable carrying the fdl command name (e.g. `train`) the
/// launcher should invoke on remote hosts via `ssh ... fdl <cmd>`. Set by
/// fdl-cli when invoking the user binary as a launcher; required by the
/// ssh fan-out path. Local fork+exec doesn't consume this — the launcher
/// re-execs `current_exe()` directly with its own argv.
pub const ENV_FDL_CMD: &str = "FLODL_INTERNAL_FDL_CMD";

/// Environment variable carrying the overlay-env name (e.g. `cluster`) so
/// the remote `fdl <cmd>` invocation resolves the same overlay-merged
/// `fdl.<env>.yml` view the controller did. Optional; absent means no
/// overlay (base `fdl.yml` only).
pub const ENV_FDL_ENV: &str = "FDL_ENV";

/// Environment variable carrying the per-host pre-flight build
/// envelope (a JSON map; format mirrors `flodl_cli::prebuild::
/// ENV_PREBUILD_PER_HOST`). When set, the launcher's remote dispatch
/// substitutes the direct-binary form for any host with an entry —
/// `ssh <host> "cd <path> && LD_LIBRARY_PATH=… exec <bin> <args>"`.
/// Hosts absent from the map fall back to the legacy `fdl <cmd>`
/// re-entry (requires cargo on the remote).
///
/// JSON shape per host: `{ "bin": "<path-relative-to-host.path>",
/// "ld_library_path": "<absolute path>" }`.
pub const ENV_PREBUILD_PER_HOST: &str = "FLODL_INTERNAL_PREBUILD_PER_HOST";

/// Role this process plays in the cluster, decided by [`dispatch`].
///
/// `dispatch` is a pure role detector — it never runs the launcher or
/// the rank loop itself. The caller drives both:
///
/// - On [`Role::Launcher`], the caller assembles the controller-scope
///   config (typically from the user's `DdpRunConfig` via
///   `super::ddp_run::build_coord_config_from_builder`), then calls
///   [`run_launcher_with_config`] and `clean_process_exit(0)` when it
///   returns. This is the "launcher trampoline": the user's `main()`
///   ran up to the `Trainer::builder(...).run()` boundary, which gives
///   the dispatch site native access to `Box<dyn ConvergenceGuard>` and
///   ElChe knobs that can't cross process boundaries.
/// - On [`Role::Rank`] / [`Role::SingleDevice`], the caller proceeds
///   with the training body.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Role {
    /// No cluster envelope in env. Continue with today's single-device
    /// training path.
    SingleDevice,
    /// This process is a rank. Continue with cluster-mode training
    /// (`Trainer::run` will read the slim envelope and rendezvous).
    Rank,
    /// This process is the launcher. Caller must run the fan-out via
    /// [`run_launcher_with_config`] and exit the program when it returns.
    Launcher,
    /// This process is a per-host transport relay. Caller must run
    /// [`run_relay`] and exit the program when it returns. Touches no
    /// CUDA; multiplexes its local ranks' frames to the controller.
    Relay,
    /// This process is a per-host worker agent (dial-in membership).
    /// Caller must run [`run_agent`] and exit the program when it
    /// returns. Touches no CUDA; joins the controller's window, then
    /// spawns + supervises this host's relay and rank children.
    Agent,
}

/// Detect this process's role from env vars. Pure function — no I/O,
/// no thread spawns, no process forks. The caller drives whatever
/// action the role demands (see [`Role`]).
///
/// Loud error on inconsistent env (e.g. both full-cluster and rank-slot
/// vars set — silently winning one over the other costs hours of
/// debugging on a misconfigured rig).
///
pub fn dispatch() -> Result<Role> {
    let agent_set = env::var_os(ENV_AGENT_JSON).is_some();
    let relay_set = env::var_os(ENV_RELAY_JSON).is_some();
    let full_set = env::var_os(ENV_FULL_CLUSTER_JSON).is_some();
    let slim_set = env::var_os(crate::distributed::cluster::ENV_CLUSTER_JSON).is_some();
    let slot_set = env::var_os(crate::distributed::cluster::ENV_LOCAL_RANK).is_some();

    match (agent_set, relay_set, full_set, slim_set, slot_set) {
        (false, false, false, false, false) => Ok(Role::SingleDevice),
        (false, false, false, true, true) => Ok(Role::Rank),
        (false, false, true, false, false) => Ok(Role::Launcher),
        (false, true, false, false, false) => Ok(Role::Relay),
        (true, false, false, false, false) => Ok(Role::Agent),
        // Any other combination is a misconfiguration. Loud error with
        // every bit named so the operator can see what's off.
        _ => Err(TensorError::new(&format!(
            "cluster launcher: inconsistent env (FLODL_INTERNAL_AGENT_JSON={}, \
             FLODL_INTERNAL_RELAY_JSON={}, \
             FLODL_INTERNAL_FULL_CLUSTER_JSON={}, FLODL_INTERNAL_CLUSTER_JSON={}, FLODL_INTERNAL_LOCAL_RANK={}). \
             Expected: all-unset (single-device), slim+slot only (rank), \
             full only (launcher), relay only (relay), or agent only (agent).",
            on_off(agent_set),
            on_off(relay_set),
            on_off(full_set),
            on_off(slim_set),
            on_off(slot_set),
        ))),
    }
}

/// True only when this process carries no cluster-role env at all —
/// i.e. [`dispatch`] resolves to [`Role::SingleDevice`]. Every
/// promotion site (programmatic cluster config, multi-GPU
/// auto-promote) must pass this gate before synthesizing
/// [`ENV_FULL_CLUSTER_JSON`]: launcher, rank and relay children all
/// carry a role var and must never re-promote — a child re-entering
/// the user binary would otherwise poison its own role env and die at
/// [`dispatch`]. An inconsistent env also returns `false`: promotion
/// is skipped and the launch-path [`dispatch`] reports it loudly.
///
/// Derived from [`dispatch`]'s truth table rather than a second var
/// list, so a future role var extends the gate automatically.
pub(crate) fn role_env_pristine() -> bool {
    matches!(dispatch(), Ok(Role::SingleDevice))
}

/// One-call short-circuit for the internal per-host worker roles
/// (transport relay, dial-in agent). Call it at the very top of
/// `main()` in any binary that **gates before** [`Trainer::run`] —
/// checks GPU counts, parses modes, validates datasets, and possibly
/// exits. If this process was spawned as a relay or agent, the role
/// runs to completion here and the process EXITS; otherwise the call
/// returns immediately and your `main()` proceeds.
///
/// ```no_run
/// // First statement of main():
/// flodl::distributed::launcher::exit_if_worker_role();
/// // ... your pre-run gating, then Trainer::run(...)
/// ```
///
/// Why it matters: cluster fan-out re-enters the user binary on every
/// host for these roles. A binary that goes straight to
/// [`Trainer::run`] needs nothing — the dispatch inside `run()`
/// catches every role. But gating logic ahead of `run()` executes in
/// the worker-role processes too, sees ONE host of a multi-host world
/// ("cpu-sync requires 2+ GPUs, have 1"), and exits without ever
/// joining — leaving the controller's join window to idle out its
/// hard cap.
///
/// Only the relay and agent roles are handled here: launcher and rank
/// processes are MEANT to run your `main()` up to `Trainer::run`, so
/// they pass through untouched.
///
/// [`Trainer::run`]: crate::distributed::Trainer::run
pub fn exit_if_worker_role() {
    // clean_process_exit throughout: same force-exit policy as the trampoline
    // arms in `DdpHandle::launch` (see its doc for the libtorch
    // static-teardown GP-fault this avoids). Relay/agent touch no CUDA today,
    // but the uniform policy costs nothing and can't rot.
    use crate::distributed::ddp_run::clean_process_exit;
    if env::var_os(ENV_RELAY_JSON).is_some() {
        match run_relay() {
            Ok(()) => clean_process_exit(0),
            Err(e) => {
                eprintln!("flodl relay: {e}");
                clean_process_exit(1);
            }
        }
    }
    if env::var_os(ENV_AGENT_JSON).is_some() {
        match run_agent() {
            Ok(()) => clean_process_exit(0),
            Err(e) => {
                eprintln!("flodl agent: {e}");
                clean_process_exit(1);
            }
        }
    }
}

/// Promote a programmatic [`FullCluster`] to the
/// [`ENV_FULL_CLUSTER_JSON`] env contract iff this process holds no
/// cluster role yet (see [`role_env_pristine`]). Returns whether
/// promotion happened.
///
/// Precedence: an fdl-cli-set full envelope wins (never overwritten),
/// and rank / relay children re-entering the user binary keep their
/// spawned role.
///
/// Caller contract: main(), before any thread spawning — the same
/// `set_var` invariant as fdl-cli's `prepare_cluster_env`.
pub(crate) fn promote_programmatic_cluster(full: &FullCluster) -> bool {
    if !role_env_pristine() {
        crate::debug!("cluster: role env already set; skipping programmatic cluster promotion");
        return false;
    }
    let hex = crate::distributed::cluster::hex_encode(full.to_json().to_string().as_bytes());
    // SAFETY: caller contract above — main(), before any thread spawning.
    unsafe { std::env::set_var(ENV_FULL_CLUSTER_JSON, hex) };
    true
}

/// Per-host relay launch spec, hex-encoded JSON in [`ENV_RELAY_JSON`].
/// Built by the launcher per host (one relay child each); consumed by
/// [`run_relay`]. Carries only what the transport relay needs — no model,
/// no CUDA, no full topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelaySpec {
    /// Relay host name (diagnostic, used in the upstream `RelayHello`).
    pub host: String,
    /// Controller host the relay dials upstream.
    pub controller_host: String,
    /// Controller port: both upstream legs dial it directly (the
    /// channel-select magic routes them through the controller's
    /// single-port mux); the relay binds loopback `+4` (data) / `+5`
    /// (control) for its local ranks.
    pub controller_port: u16,
    /// Global ranks this host carries (the relay's local rank set).
    pub ranks: Vec<u32>,
    /// Session salt, hex-encoded (HMAC key for the mux + forwarded frames).
    pub salt_hex: String,
    /// Cluster-wide rank count (validated in each rank's handshake).
    pub world_size: usize,
    /// Start the CPU-averaging data relay. `false` for NCCL backends —
    /// ranks never dial the data channel there, so an always-on data
    /// relay would block forever in `accept`.
    pub data_channel: bool,
    /// Model-derived frame ceiling (bytes) for the relay's
    /// length-prefixed readers, computed by the launcher's CPU probe.
    /// `0` (or absent, via serde default) keeps the 1 GiB default —
    /// purely a local reject-threshold, so no agreement is required.
    #[serde(default)]
    pub frame_ceiling_bytes: usize,
}

/// Run this process as a per-host transport relay: bind the loopback data
/// (`+4`) / control (`+5`) channels its local ranks dial, forward upstream
/// to the controller's single mux port, and stay up until every local
/// rank disconnects (training finished). Touches no CUDA.
///
/// The caller (the dispatch site on [`Role::Relay`]) runs this and exits
/// the process when it returns.
pub fn run_relay() -> Result<()> {
    let raw = env::var(ENV_RELAY_JSON)
        .map_err(|e| TensorError::new(&format!("relay: {ENV_RELAY_JSON} unreadable: {e}")))?;
    let bytes = crate::distributed::cluster::hex_decode(&raw)
        .map_err(|e| TensorError::new(&format!("relay: spec hex-decode: {e}")))?;
    let spec: RelaySpec = serde_json::from_slice(&bytes)
        .map_err(|e| TensorError::new(&format!("relay: spec JSON parse: {e}")))?;
    // Install the launcher-derived frame ceiling before any channel
    // reads a frame (zero = unset → keep the default; set_frame_ceiling
    // ignores it).
    crate::distributed::wire::set_frame_ceiling(spec.frame_ceiling_bytes);
    let salt = crate::distributed::wire::salt_from_hex(&spec.salt_hex)?;
    let base = spec.controller_port;

    let loopback = |off: u16| -> Result<std::net::SocketAddr> {
        format!("127.0.0.1:{}", base.saturating_add(off))
            .parse()
            .map_err(|e| TensorError::new(&format!("relay: loopback addr: {e}")))
    };
    let resolve = |host: &str, port: u16| -> Result<std::net::SocketAddr> {
        use std::net::ToSocketAddrs;
        (host, port)
            .to_socket_addrs()
            .map_err(|e| TensorError::new(&format!("relay: resolve {host}:{port}: {e}")))?
            .next()
            .ok_or_else(|| TensorError::new(&format!("relay: no address for {host}:{port}")))
    };

    eprintln!(
        "cluster relay: host '{}' ranks {:?} -> controller {}:{} (data_channel={})",
        spec.host, spec.ranks, spec.controller_host, base, spec.data_channel,
    );

    // Control channel (every backend uses it). Bind before accepting.
    // Both upstream legs dial the controller's single mux port; the
    // channel-select magic (written by `RelayChannel::start`) routes
    // each connection to its subsystem.
    let (ctrl_listener, _) = RelayChannel::bind(loopback(RELAY_CONTROL_LOOPBACK_OFFSET)?)?;
    let ctrl_upstream = resolve(&spec.controller_host, base)?;

    // Data channel (CPU backends only). Ranks connect to BOTH channels
    // (data first, then control in the CPU path), so the two accept loops
    // must run concurrently — the data relay runs on its own thread.
    let data_handle = if spec.data_channel {
        let (data_listener, _) = RelayChannel::bind(loopback(RELAY_DATA_LOOPBACK_OFFSET)?)?;
        let data_upstream = resolve(&spec.controller_host, base)?;
        let host = spec.host.clone();
        let ranks = spec.ranks.clone();
        let ws = spec.world_size;
        Some(
            thread::Builder::new()
                .name("flodl-relay-data".into())
                .spawn(move || -> Result<()> {
                    RelayChannel::start(
                        data_listener,
                        ChannelKind::Data,
                        data_upstream,
                        host,
                        ranks,
                        ws,
                        salt,
                    )?
                    .join()
                })
                .map_err(|e| TensorError::new(&format!("relay: spawn data thread: {e}")))?,
        )
    } else {
        None
    };

    // Control relay on this thread: blocks until ranks connect, then runs
    // until they all disconnect (training finished).
    RelayChannel::start(
        ctrl_listener,
        ChannelKind::Control,
        ctrl_upstream,
        spec.host.clone(),
        spec.ranks.clone(),
        spec.world_size,
        salt,
    )?
    .join()?;

    if let Some(h) = data_handle {
        h.join()
            .map_err(|_| TensorError::new("relay: data thread panicked"))??;
    }
    eprintln!("cluster relay: host '{}' shut down cleanly", spec.host);
    Ok(())
}

fn on_off(b: bool) -> &'static str {
    if b { "set" } else { "unset" }
}

/// One-cluster-run-per-process latch. The launcher-side infrastructure
/// (rendezvous listener, relay processes, coordinator, controller) is
/// built for exactly ONE training session: the rendezvous closes after
/// the first bootstrap, relays self-shut when their ranks disconnect,
/// and the coordinator is constructed from the first run's config. A
/// second cluster `Trainer::run` in the same process (e.g. a bench
/// binary looping over models) would dial infrastructure that no longer
/// exists and hang or get connection-refused — fail it loudly instead,
/// with the supported pattern in the message.
static CLUSTER_ENTRY_CONSUMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Claim this process's single cluster entry. `Err` on the second call.
pub(crate) fn claim_cluster_entry(role: &str) -> Result<()> {
    if CLUSTER_ENTRY_CONSUMED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Err(TensorError::new(&format!(
            "cluster {role}: a cluster training session was already run in \
             this process — the launcher infrastructure (rendezvous, relays, \
             coordinator) is per-session and has shut down. Run one cluster \
             `Trainer::run` per process: loop at the process level (e.g. \
             invoke the binary once per model) instead of looping inside it.",
        )));
    }
    Ok(())
}

/// Validate `tunnel: true` topology and derive the controller bind
/// scope. Returns whether the mux should bind loopback-only (every
/// remote worker is tunneled, so training traffic can only arrive
/// through sshd).
///
/// Loud errors (explicit selectors error; conventions warn):
/// - tunnel on the launcher-local host — there is no SSH session to
///   carry a forward, and loopback already reaches the controller;
/// - tunnel under an NCCL backend — the NCCL data plane is
///   peer-to-peer and cannot ride a controller tunnel (this also
///   covers the no-coordinator legacy path, which has no relay
///   session to attach the forward to).
fn validate_tunnel_topology(
    full: &FullCluster,
    local_host_name: &str,
    backend_is_nccl: bool,
) -> Result<bool> {
    let tunneled: Vec<&FullWorker> = full.workers.iter().filter(|w| w.tunnel).collect();
    if tunneled.is_empty() {
        return Ok(false);
    }
    if let Some(local) = tunneled.iter().find(|w| w.host == local_host_name) {
        return Err(TensorError::new(&format!(
            "cluster launcher: worker {:?} sets `tunnel: true` but runs on \
             the launcher host — there is no SSH session to carry a \
             forward, and loopback already reaches the controller. Remove \
             the flag from this host.",
            local.host,
        )));
    }
    if backend_is_nccl {
        return Err(TensorError::new(&format!(
            "cluster launcher: worker(s) {:?} set `tunnel: true` but the \
             run uses an NCCL backend. NCCL's data plane is peer-to-peer \
             and cannot ride a controller tunnel; tunnel mode requires a \
             CPU ElChe mode (cpu_sync / cpu_cadence / cpu_async), whose \
             traffic all flows through the per-host relay's single \
             upstream connection.",
            tunneled.iter().map(|w| w.host.as_str()).collect::<Vec<_>>(),
        )));
    }
    Ok(full
        .workers
        .iter()
        .filter(|w| w.host != local_host_name)
        .all(|w| w.tunnel))
}

/// Validate the discovery-family knob combinations and resolve the
/// run's session salt: the configured `token` verbatim, or a fresh
/// 128-bit salt. Loud on contradictions — `tunnel_only` outside
/// discovery mode (an enumerated roster already infers the bind scope
/// from its `tunnel:` flags), a token alongside `open_admission: true`
/// (a configured credential means admission checks it), and a token
/// that is not exactly [`SESSION_SALT_BYTES`] of hex.
///
/// [`SESSION_SALT_BYTES`]: crate::distributed::wire::SESSION_SALT_BYTES
fn resolve_session_salt(knobs: &JoinKnobs) -> Result<crate::distributed::wire::SessionSalt> {
    let discovery = knobs.discovery.unwrap_or(false);
    if knobs.tunnel_only.unwrap_or(false) && !discovery {
        return Err(TensorError::new(
            "cluster launcher: `controller.join.tunnel_only` is a \
             discovery-mode knob — an enumerated roster already infers the \
             bind scope from its per-worker `tunnel:` flags",
        ));
    }
    if knobs.token.is_some() && knobs.open_admission == Some(true) {
        return Err(TensorError::new(
            "cluster launcher: `controller.join.token` and `open_admission: \
             true` contradict — a configured token means admission is \
             credential-authenticated; drop one of them",
        ));
    }
    match &knobs.token {
        Some(hex) => {
            let bytes = crate::distributed::cluster::hex_decode(hex.trim()).map_err(|e| {
                TensorError::new(&format!(
                    "cluster launcher: controller.join.token hex-decode \
                         failed: {e}"
                ))
            })?;
            let want = crate::distributed::wire::SESSION_SALT_BYTES;
            let got = bytes.len();
            <[u8; crate::distributed::wire::SESSION_SALT_BYTES]>::try_from(bytes).map_err(|_| {
                TensorError::new(&format!(
                    "cluster launcher: controller.join.token must be \
                         {want} bytes ({} hex chars), got {got} bytes",
                    want * 2,
                ))
            })
        }
        None => Ok(crate::distributed::wire::generate_session_salt()),
    }
}

/// Fan-out derivation of the join-window quorum knobs: the configured
/// topology IS the capacity fan-out just started, so by default the
/// window closes the instant all of it is in (zero added latency vs the
/// direct-spawn era) and the run cannot start below it (same
/// all-or-nothing semantics). Every `controller.join:` field the user
/// set overrides its derived default; the hard cap stretches to cover
/// an enlarged window rather than failing validation. Discovery mode
/// derives from the window instead of the roster: the quorum must be
/// explicit and the early-close target stays unset unless asked for.
fn derive_join_config(
    knobs: Option<&JoinKnobs>,
    capacity: usize,
    nccl_backend: bool,
) -> Result<crate::distributed::membership::JoinConfig> {
    let defaults = crate::distributed::membership::JoinConfig::default();
    let knobs = knobs.cloned().unwrap_or_default();
    let join_timeout_secs = knobs
        .join_timeout_secs
        .unwrap_or(defaults.join_timeout_secs);
    let discovery = knobs.discovery.unwrap_or(false);
    // Discovery has no roster capacity to derive the quorum from, and
    // silently defaulting it would let a single walk-in start a world
    // meant for eight — the operator states it. The early-close target
    // stays unset unless asked for: with an unknown fleet the full
    // window runs (target-count auto-close remains available when the
    // provisioned count IS known).
    let (min_rank_start, target_ranks) = if discovery {
        let quorum = knobs.min_rank_start.ok_or_else(|| {
            TensorError::new(
                "cluster launcher: `controller.join.discovery: true` requires \
                 an explicit `min_rank_start` — a roster-free window has no \
                 configured capacity to derive the quorum from",
            )
        })?;
        (quorum, knobs.target_ranks)
    } else if knobs.start == Some(crate::distributed::membership::StartMode::Manual) {
        // Manual mode has no clock-side auto-close: the capacity default
        // for `target_ranks` must not apply, or validate() would refuse
        // a combination the user never wrote. An EXPLICIT target still
        // reaches validate() and errors loudly there.
        (knobs.min_rank_start.unwrap_or(capacity), knobs.target_ranks)
    } else {
        (
            knobs.min_rank_start.unwrap_or(capacity),
            Some(knobs.target_ranks.unwrap_or(capacity)),
        )
    };
    Ok(crate::distributed::membership::JoinConfig {
        min_rank_start,
        join_timeout_secs,
        target_ranks,
        max_join_timeout_secs: knobs
            .max_join_timeout_secs
            .unwrap_or(defaults.max_join_timeout_secs.max(join_timeout_secs)),
        open_admission: knobs.open_admission.unwrap_or(false),
        start_mode: knobs.start.unwrap_or_default(),
        nccl_backend,
    })
}

/// Build the formed world's topology from the join-window membership.
///
/// Rank ids and device lists come from admission (they ARE the world);
/// ssh/env/path/tunnel metadata carries over from the configured entry
/// when the joiner matches one (fan-out hosts always do). A walk-in
/// worker gets a minimal entry — the launcher never dials it, so
/// transport fields stay empty.
fn synthesize_world<'a>(
    config: &FullCluster,
    members: impl Iterator<Item = &'a crate::distributed::membership::JoinedMember>,
    salt: crate::distributed::wire::SessionSalt,
    bind_loopback: bool,
) -> FullCluster {
    let workers: Vec<FullWorker> = members
        .map(|m| match config.workers.iter().find(|w| w.host == m.host) {
            Some(w) => FullWorker {
                ranks: m.ranks.clone(),
                local_devices: Some(m.local_devices.clone()),
                ..w.clone()
            },
            None => FullWorker {
                host: m.host.clone(),
                ranks: m.ranks.clone(),
                local_devices: Some(m.local_devices.clone()),
                nccl_socket_ifname: String::new(),
                path: String::new(),
                arch: None,
                // The controller has nothing to say about the source
                // root of a host it never configured. A walk-in learns
                // its own from the join config on its box, which is
                // where the path is actually resolvable, and its agent
                // writes it into this envelope on arrival
                // (`AgentSpec::data_path`).
                data_path: None,
                // Same class of fact, same road: None here lets the
                // cluster-scope default flow into the slim envelope,
                // and the box's own `join.gpu_ram_share:` overrides it
                // at localization.
                gpu_ram_share: None,
                ssh: None,
                // On a loopback-bound mux a walk-in can only have
                // arrived through an sshd forward, so its rank children
                // must dial their loopback end of that forward too — a
                // config-host address would point at a port that is
                // unreachable except through sshd.
                tunnel: bind_loopback,
                env: Default::default(),
            },
        })
        .collect();
    FullCluster {
        controller: config.controller.clone(),
        workers,
        salt,
        env: config.env.clone(),
        gpu_ram_share: config.gpu_ram_share,
    }
}

/// A remote host's agent child during the join window: host name, the
/// ssh process, and its output forwarders. Its global ranks are known
/// only after formation, when it becomes a supervision entry.
type RemoteAgentChild = (String, std::process::Child, Vec<thread::JoinHandle<()>>);

/// A launcher-local host's in-process join: host name plus the thread
/// that dials in and spawns the host's children (taken exactly once).
type LocalJoin = (
    String,
    Option<thread::JoinHandle<Result<Vec<agent::HostChild>>>>,
);

/// Controller-scope coordinator wiring, handed to
/// [`run_launcher_with_config`] by the launcher-trampoline caller.
///
/// `world_size` is known only when the join window closes, so the
/// coordinator config is built by a factory at that moment instead of
/// being passed pre-built — every per-rank structure (ElChe, heartbeat
/// ledgers, callback roles) then sizes to the world that actually
/// formed, not the world the config file promised. `backend` is
/// duplicated out of the config because the launcher needs it BEFORE
/// formation (tunnel validation, rendezvous gating, relay data-channel
/// selection).
pub struct CoordSpec {
    /// Averaging backend of the run (must match what the factory bakes
    /// into its config).
    pub backend: crate::distributed::ddp_run::AverageBackend,
    /// Builds the coordinator config for the formed world size.
    pub config_factory: Box<
        dyn FnOnce(
                usize,
            )
                -> Result<crate::distributed::cluster_coordinator::ClusterCoordinatorConfig>
            + Send,
    >,
}

/// Tell every still-connected worker host to tear down: an Abort frame
/// down the host control link (the agent's teardown trigger — it kills
/// its relay and rank children on receipt), then a socket shutdown,
/// which also unblocks the link's reader thread. Best-effort per link —
/// a host that already vanished just fails the write.
fn abort_worker_links(
    links: &mut [(String, std::net::TcpStream)],
    salt: &crate::distributed::wire::SessionSalt,
    reason: &str,
) {
    for (host, link) in links {
        let abort_msg = crate::distributed::wire::JoinMsgWire::Abort {
            reason: reason.to_string(),
        };
        let send = crate::distributed::wire::ControlFrame::encode(
            salt,
            crate::distributed::wire::MsgKind::Join,
            &abort_msg,
        )
        .and_then(|f| f.write_to(link));
        if send.is_err() {
            crate::verbose!(
                "  cluster launcher: abort to worker host {host:?} failed \
                 (already gone)"
            );
        }
        let _ = link.shutdown(std::net::Shutdown::Both);
    }
}

/// Run this process as the cluster launcher: open the join window, fan
/// out one worker agent per configured host (push-as-sugar — the agents
/// dial back in like any self-deployed worker would), form the world,
/// start the controller-side infrastructure sized to it, ship each
/// admitted worker its spawn artifacts, and supervise until the run
/// ends.
///
/// `coord` carries the controller-scope coordinator wiring (see
/// [`CoordSpec`]); `None` preserves the legacy no-coordinator NCCL
/// routing (no relays, ranks dial the controller directly).
///
/// `expected_model_sig` seeds the join window's model-signature check
/// (the launcher's own CPU-built model is the run's truth); `None`
/// falls back to first-member seeding among the walk-ins that carry
/// one.
pub fn run_launcher_with_config(
    full: FullCluster,
    coord: Option<CoordSpec>,
    outer_optimizer: Option<Box<dyn crate::distributed::OuterOptimizer>>,
    expected_model_sig: Option<[u8; 32]>,
    abort: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    use crate::distributed::membership;

    claim_cluster_entry("launcher")?;
    // Discovery-family knobs, validated before anything binds or spawns.
    let knobs = full.controller.join.clone().unwrap_or_default();
    let tunnel_only = knobs.tunnel_only.unwrap_or(false);
    // Session salt: the HMAC key for every cross-process control + data
    // frame, handed to workers at admission (pre-shared via the agent
    // spec in rig mode, in the accept reply under open admission). A
    // configured join token IS the salt — injected at fleet-create time
    // so walk-ins can present it — otherwise a fresh 128-bit salt is
    // generated per launcher invocation.
    let salt = resolve_session_salt(&knobs)?;
    let full = full.with_session_salt(salt);
    let me = crate::distributed::cluster::resolve_hostname()?;

    // Backend, resolved before any bind/spawn decision: tunnel
    // validation and the rendezvous/relay gating below all key off it.
    // Unknown backend (no `coord`) defaults to NCCL, preserving prior
    // behavior for non-coordinator paths.
    let backend_is_nccl = coord
        .as_ref()
        .map(|c| matches!(c.backend, crate::distributed::ddp_run::AverageBackend::Nccl))
        .unwrap_or(true);
    let has_coord = coord.is_some();
    let relay_data_channel = has_coord && !backend_is_nccl;

    // Tunnel topology validation (loud, before anything binds or
    // spawns) + the resulting controller bind scope. `tunnel_only`
    // forces the loopback bind with no roster to infer it from; it
    // rides the same CPU-mode constraint as per-worker tunnels (NCCL's
    // peer-to-peer data plane cannot ride a controller tunnel).
    if tunnel_only && backend_is_nccl {
        return Err(TensorError::new(
            "cluster launcher: `controller.join.tunnel_only` requires a CPU \
             ElChe mode (cpu_sync / cpu_cadence / cpu_async) — NCCL's data \
             plane is peer-to-peer and cannot ride a controller tunnel",
        ));
    }
    // A loopback-bound mux is unreachable except through sshd, so any
    // enumerated remote worker that is not itself tunneled could never
    // dial in — its fan-out agent would idle the window to the hard cap.
    if tunnel_only
        && let Some(w) = full
            .workers
            .iter()
            .find(|w| w.host != me && !w.tunnel && !w.ranks.is_empty())
    {
        return Err(TensorError::new(&format!(
            "cluster launcher: `controller.join.tunnel_only` binds the \
             controller loopback-only, but enumerated worker {:?} is not \
             `tunnel: true` and could never reach it",
            w.host,
        )));
    }
    let bind_loopback = validate_tunnel_topology(&full, &me, backend_is_nccl)? || tunnel_only;

    // Single-port mux: every controller-side channel (join, NCCL
    // rendezvous, CPU-reduce data, coordinator control) accepts on ONE
    // port — `controller.port` — and dialers route themselves with a
    // channel-select magic (see `port_mux`). Bound to 0.0.0.0 so remote
    // hosts reach it; local ranks use the same port via loopback.
    // EXCEPT when every remote worker rides an SSH tunnel: then the mux
    // binds loopback only, so the port is unreachable except through
    // sshd — the bind-scope side of the trust model.
    //
    // Back-to-back runs on the fixed port are safe as-is: Rust's
    // `TcpListener::bind` sets SO_REUSEADDR on Unix, so TIME_WAIT
    // remnants from a previous run's connections never block this bind
    // (probed: rebind succeeds with the port verifiably in TIME_WAIT). A
    // genuinely LIVE listener from a still-running launcher still fails
    // loudly here (that would need SO_REUSEPORT) — the desirable
    // double-run guard.
    let mux_port = full.controller.port;
    let mux_bind_ip = if bind_loopback {
        "127.0.0.1"
    } else {
        "0.0.0.0"
    };
    let mux_bind = format!("{mux_bind_ip}:{mux_port}");
    let mux_listener = std::net::TcpListener::bind(&mux_bind).map_err(|e| {
        TensorError::new(&format!(
            "cluster launcher: bind {mux_bind} failed: {e}{}",
            crate::distributed::bind_diag::hint_suffix(mux_port, e.kind())
        ))
    })?;
    let (port_mux, mux_accept) =
        crate::distributed::port_mux::PortMux::start(mux_listener, Arc::clone(&abort))?;
    let crate::distributed::port_mux::MuxAccept {
        rendezvous: mux_rendezvous,
        data: mux_data,
        control: mux_control,
        join: mux_join,
        status: mux_status,
    } = mux_accept;
    eprintln!(
        "cluster launcher: port mux bound on {mux_bind_ip}:{} \
         (join + rendezvous + data + control + status{})",
        port_mux.port(),
        if bind_loopback {
            "; loopback-only, all workers tunneled"
        } else {
            ""
        },
    );

    // Status endpoint: plain HTTP GETs on the mux port answer with the
    // run's membership state as `state.json` (`fdl status`, curl, a
    // browser). Live from here — BEFORE the join window opens — so the
    // whole lifecycle is observable, `waiting`/`forming` included.
    // Trust follows bind scope, exactly like join admission.
    let status_board = crate::distributed::status::StatusBoard::new();
    let mut status_server = {
        let board = status_board.clone();
        let source = crate::distributed::port_mux::StreamSource::Mux(mux_status);
        let abort_c = Arc::clone(&abort);
        Some(
            thread::Builder::new()
                .name("flodl-status-http".to_string())
                .spawn(move || {
                    crate::distributed::status::serve_status(source, board, abort_c);
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster launcher: spawn status responder failed: {e}"
                    ))
                })?,
        )
    };

    // Controller address AS SEEN FROM a given worker: tunneled workers
    // dial their loopback end of the SSH forward; when the mux binds
    // loopback-only, launcher-local workers must dial loopback too.
    let controller_host_cfg = full.controller.host.clone();
    let launcher_host = me.clone();
    let controller_dial_host = move |worker: &FullWorker| -> String {
        if worker.tunnel || (bind_loopback && worker.host == launcher_host) {
            "127.0.0.1".to_string()
        } else {
            controller_host_cfg.clone()
        }
    };

    // ------------------------------------------------------------------
    // Membership window
    // ------------------------------------------------------------------
    let capacity = full.world_size();
    // The backend rides into admission so the window can refuse a
    // vendor-mixed cohort exactly when the data plane cannot carry one.
    let join_config = derive_join_config(full.controller.join.as_ref(), capacity, backend_is_nccl)?;
    // A configured token forces credential-authenticated admission even
    // behind a loopback bind: the sshd guardrail and the token are
    // LAYERS (reachability + possession), not alternatives — an
    // operator who configured a credential gets a window that checks
    // it. Without a token, bind scope decides as usual.
    let open_admission = if knobs.token.is_some() {
        false
    } else {
        membership::resolve_open_admission(&join_config, bind_loopback)
    };
    // Wire the operator start switch: the status responder arms it
    // (authenticated POST /start), the join window polls it. The
    // credential is the session salt — under a configured `join.token`
    // that is exactly the token the operator holds.
    status_board.configure_start(
        join_config.start_mode,
        crate::distributed::wire::salt_to_hex(&salt),
    );
    let gate_config = join_config.clone();
    let gate_salt = salt;
    let gate_abort = Arc::clone(&abort);
    let gate_status = status_board.clone();
    let gate_source = crate::distributed::port_mux::StreamSource::Mux(mux_join);
    let gate = thread::Builder::new()
        .name("flodl-join-gate".to_string())
        .spawn(move || {
            membership::run_join_window(
                &gate_source,
                &gate_config,
                &gate_salt,
                !open_admission,
                None,
                expected_model_sig,
                &gate_abort,
                &gate_status,
            )
        })
        .map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: spawn join-gate thread failed: {e}"
            ))
        })?;

    // ------------------------------------------------------------------
    // Fan-out: one agent per remote host; local hosts join in-process
    // ------------------------------------------------------------------
    // For remote hosts, fdl-cli must have passed the original fdl command
    // name so we can invoke `fdl <cmd>` over ssh. Loud error if absent.
    let has_remote = full
        .workers
        .iter()
        .any(|h| h.host != me && !h.ranks.is_empty());
    let fdl_cmd = if has_remote {
        Some(env::var(ENV_FDL_CMD).map_err(|_| {
            TensorError::new(&format!(
                "cluster launcher: topology has remote hosts but {ENV_FDL_CMD} \
                 is not set in env. fdl-cli must export the fdl command name \
                 (e.g. {ENV_FDL_CMD}=train) when invoking the launcher."
            ))
        })?)
    } else {
        None
    };
    let overlay_env = env::var(ENV_FDL_ENV).ok().filter(|s| !s.trim().is_empty());
    // Per-host pre-flight build envelope from fdl-cli. When a remote
    // host has an entry, the remote dispatch substitutes the direct
    // binary exec (no cargo on remote). Missing entry ⇒ legacy
    // `fdl <cmd>` fallback (requires cargo on the remote).
    let prebuild_envelope = load_prebuild_envelope()?;

    // Collect (host, abs_bin) for every remote host that has a prebuild
    // envelope entry. Used for both pre-spawn cleanup (clear orphans
    // from a previous botched session before workers come up) and post-
    // exit cleanup (catch anything the remote-side trap wrapper didn't
    // reap — including the agent's own children, which run the same
    // binary and therefore match the same pkill pattern).
    let remote_cleanup_targets: Vec<(FullWorker, String)> = full
        .workers
        .iter()
        .filter(|h| h.host != me)
        .filter_map(|h| {
            prebuild_envelope.get(&h.host).map(|pb| {
                let abs_bin = format!("{}/{}", h.path.trim_end_matches('/'), pb.bin,);
                (h.clone(), abs_bin)
            })
        })
        .collect();

    // Pre-spawn cleanup: SIGTERM/SIGKILL any leftover instance of this
    // run's binary on each remote host. Self-heals across sessions:
    // a previous launcher that died hard (SIGKILL, OOM, kernel panic)
    // can leave orphans the trap wrapper couldn't reap. This pass
    // guarantees a fresh start regardless.
    cleanup_remote_hosts_parallel(remote_cleanup_targets.clone());

    // Remote agents spawned during the window; their global ranks are
    // known only after formation, so supervision entries are assembled
    // then. Local hosts run the same join protocol on a thread and hand
    // their children back — they stay DIRECT children of this process,
    // so launcher supervision owns them first-hand exactly as before.
    let mut remote_agents: Vec<RemoteAgentChild> = Vec::new();
    let mut local_joins: Vec<LocalJoin> = Vec::new();
    let salt_hex_for_agents =
        (!open_admission).then(|| crate::distributed::wire::salt_to_hex(&salt));
    let spawn_result: Result<()> = (|| {
        for host in &full.workers {
            // Orchestrator-only entry (empty `ranks`): declared in
            // cluster.yml solely so fdl-cli's pre-flight can read its
            // `docker:` / `arch:` — no worker runs there.
            if host.ranks.is_empty() {
                continue;
            }
            let spec = agent::AgentSpec {
                host: host.host.clone(),
                // Fan-out hosts run what the controller dispatched, so
                // there is no published-run identity to disagree about.
                run_id: None,
                controller_host: if host.host == me {
                    // Local workers always reach the mux via loopback.
                    "127.0.0.1".to_string()
                } else {
                    controller_dial_host(host)
                },
                controller_port: mux_port,
                salt_hex: salt_hex_for_agents.clone(),
                local_devices: host.local_devices.clone(),
                libtorch: host.arch.clone().unwrap_or_default(),
                dataset_sig_hex: None,
                // Fan-out: the roster already carries this host's source
                // root and resolved RAM share into its envelope, so the
                // agent has nothing to localize.
                data_path: None,
                gpu_ram_share: None,
                // Fan-out hosts run what the controller dispatched, so
                // there is no model identity to probe; the controller's
                // own seed covers the ledger.
                model_sig_hex: None,
            };
            if host.host == me {
                // Merged env for the local children (cluster-scope
                // first, host-scope override) — same application the
                // remote path gets via the agent command's bash prefix.
                let mut extra_env = full.env.clone();
                extra_env.extend(host.env.clone());
                let host_name = host.host.clone();
                let handle = thread::Builder::new()
                    .name(format!("flodl-local-join:{host_name}"))
                    .spawn(move || agent::join_and_spawn_local(spec, &extra_env))
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster launcher: spawn local join thread failed: {e}"
                        ))
                    })?;
                local_joins.push((host.host.clone(), Some(handle)));
            } else {
                let spec_hex = spec.to_env_hex()?;
                let remote_cmd = build_remote_agent_bash_command(
                    &host.path,
                    &host.host,
                    overlay_env.as_deref(),
                    fdl_cmd
                        .as_deref()
                        .expect("ENV_FDL_CMD presence enforced above when has_remote"),
                    &env::args().skip(1).collect::<Vec<String>>(),
                    &full.env,
                    &host.env,
                    prebuild_envelope.get(&host.host),
                );
                // The agent session carries the host's training tunnel
                // when `tunnel: true` — the ONE ssh session per host.
                let mut cmd =
                    build_ssh_spawn_command(host, &remote_cmd, host.tunnel.then_some(mux_port));
                // The spec (salt-bearing) rides stdin, never argv.
                cmd.stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());
                let mut child = cmd.spawn().map_err(|e| {
                    TensorError::new(&format!(
                        "cluster launcher: spawn ssh agent for {:?} failed: {e}",
                        host.host
                    ))
                })?;
                spawn::pipe_envelope_to_child(&mut child, &spec_hex);
                // Forward the agent's output RAW: the agent already
                // prefixes its children (`[host:dev:rN]`, `[host:relay]`)
                // and names the host in its own diagnostics — a launcher
                // prefix here would double up on every training line.
                let mut forwarders = Vec::with_capacity(2);
                if let Some(out) = child.stdout.take() {
                    forwarders.push(thread::spawn(move || {
                        forward_lines(out, String::new(), false);
                    }));
                }
                if let Some(err) = child.stderr.take() {
                    forwarders.push(thread::spawn(move || {
                        forward_lines(err, String::new(), true);
                    }));
                }
                remote_agents.push((host.host.clone(), child, forwarders));
            }
        }
        Ok(())
    })();

    // Teardown helper for every pre-supervision failure path: stop the
    // window, reap agents + local join threads, clean remote hosts.
    let teardown_early = |remote_agents: &mut Vec<RemoteAgentChild>,
                          local_joins: &mut Vec<LocalJoin>| {
        abort.store(true, std::sync::atomic::Ordering::SeqCst);
        for (_, child, forwarders) in remote_agents.drain(..) {
            let mut child = child;
            let _ = child.kill();
            let _ = child.wait();
            for f in forwarders {
                let _ = f.join();
            }
        }
        for (_, handle) in local_joins.iter_mut() {
            if let Some(h) = handle.take() {
                // The gate aborts (Abort frame or dropped join leg), so
                // the local join thread unblocks promptly; kill whatever
                // it managed to spawn.
                if let Ok(Ok(children)) = h.join() {
                    for mut c in children {
                        let _ = c.child.kill();
                        let _ = c.child.wait();
                        for f in c.forwarders {
                            let _ = f.join();
                        }
                    }
                }
            }
        }
        cleanup_remote_hosts_parallel(remote_cleanup_targets.clone());
    };

    if let Err(e) = spawn_result {
        eprintln!(
            "cluster launcher: fan-out failed; tearing down {} agent(s): {e}",
            remote_agents.len() + local_joins.len(),
        );
        teardown_early(&mut remote_agents, &mut local_joins);
        let _ = gate.join();
        if let Some(h) = status_server.take() {
            let _ = h.join();
        }
        return Err(e);
    }

    // Wait for the window while watching the agents: an agent that dies
    // BEFORE the world forms can never join, so with fan-out's
    // all-of-capacity target the window would otherwise idle to its hard
    // cap — fail fast instead, matching the direct-spawn era's
    // first-failure behavior.
    let mut dead_agent: Option<String> = None;
    let formed = loop {
        if gate.is_finished() {
            break gate
                .join()
                .map_err(|_| TensorError::new("cluster launcher: join-gate thread panicked"))?;
        }
        if dead_agent.is_none() {
            for (host, child, _) in remote_agents.iter_mut() {
                // ANY pre-formation agent exit is fatal — even a clean
                // one (its ranks can never join, so the window would
                // otherwise idle out its full hard cap; observed with an
                // agent whose binary exited 0 through a pre-run gate).
                if let Ok(Some(st)) = child.try_wait() {
                    eprintln!(
                        "cluster launcher: agent of {host:?} exited with \
                         {st} before the world formed; aborting the window"
                    );
                    dead_agent = Some(host.clone());
                    abort.store(true, std::sync::atomic::Ordering::SeqCst);
                    break;
                }
            }
        }
        thread::sleep(std::time::Duration::from_millis(20));
    };
    let formed = match formed {
        Ok(f) => f,
        Err(e) => {
            teardown_early(&mut remote_agents, &mut local_joins);
            if let Some(h) = status_server.take() {
                let _ = h.join();
            }
            return Err(match dead_agent {
                Some(host) => TensorError::new(&format!(
                    "cluster launcher: agent of {host:?} died before the world \
                     formed ({e})"
                )),
                None => e,
            });
        }
    };

    // World synthesis: the formed membership becomes the topology.
    let membership::FormedWorld {
        workers: formed_workers,
        world_size,
        snapshot: mut membership_state,
    } = formed;
    let world = synthesize_world(
        &full,
        formed_workers.iter().map(|aw| &aw.member),
        salt,
        bind_loopback,
    );
    let my_host_idx = world.workers.iter().position(|h| h.host == me);

    // ------------------------------------------------------------------
    // Controller-side infrastructure, sized to the formed world
    // ------------------------------------------------------------------
    let mut coord_config = match coord {
        Some(spec) => Some((spec.config_factory)(world_size)?),
        None => None,
    };
    let elastic_max_failure = coord_config.as_ref().and_then(|c| c.max_failure);

    // Shared dead-rank ledger between ClusterController (CPU averaging
    // releases on heartbeat-stale) and ClusterCoordinator (NCCL
    // elastic-membership rendezvous trigger). Both consumers see the
    // same source of truth. Always constructed even on legacy NCCL
    // runs — the cost is negligible (a Vec<AtomicBool>) and the wiring
    // keeps both backends pluggable.
    let dead_ranks_shared = crate::distributed::controller::DeadRanks::new(world_size);
    // Consensus-checkpoint forge: holds the launch-captured model schema so the
    // controller reduce thread can write a named `.fdl` from the averaged
    // (name-less) frame. Shared with the coordinator (which arms it before a
    // checkpoint reduce) — same Arc-sharing pattern as `dead_ranks`. Take the
    // schema out of the coord config (the coord never writes the model itself).
    let model_schema = coord_config.as_mut().and_then(|c| c.model_schema.take());
    // The launch-wrapped user checkpoint_fn travels the same conduit and is
    // installed on the forge, which fires it on its detached writer thread
    // when an armed cycle's consensus materializes (CPU backend only).
    let consensus_fn = coord_config
        .as_mut()
        .and_then(|c| c.consensus_checkpoint_fn.take());
    let checkpoint_forge = crate::distributed::CheckpointForge::new(model_schema, consensus_fn);
    if let Some(cfg) = coord_config.as_mut() {
        cfg.checkpoint_forge = Some(Arc::clone(&checkpoint_forge));
    }
    // ClusterController on the mux's data leg. Always started, even on
    // NCCL-only clusters: the accept loop polls a shutdown flag every
    // 20ms, so an unused ClusterController exits cleanly when the
    // launcher signals shutdown after children finish. Cost is one idle
    // thread.
    let cpu_averager = crate::distributed::controller::ClusterController::start_from_source(
        crate::distributed::port_mux::StreamSource::Mux(mux_data),
        mux_port,
        world_size,
        salt,
        Arc::clone(&dead_ranks_shared),
        Some(Arc::clone(&checkpoint_forge)),
        outer_optimizer,
    )?;
    eprintln!(
        "cluster launcher: ClusterController up on port {} (world_size={})",
        cpu_averager.port(),
        world_size,
    );

    // ClusterCoordinator spawn on the mux's control leg for the elastic-
    // membership-aware NCCL path. `coord_config = Some(...)` means the
    // caller (the trampoline at `DdpHandle::launch`) built the
    // controller-scope config from the user's `DdpRunConfig` and wants
    // a coord spawned. `None` skips the coord — legacy NCCL routing
    // (worker self-driven ElChe, no elastic membership) handles that
    // path entirely on the rank side.
    let mut dashboard_sink_outer: Option<Arc<dyn crate::distributed::DashboardSink>> = None;
    let reported_deaths: crate::distributed::cluster_coordinator::ReportedDeaths =
        Arc::new(std::sync::Mutex::new(Vec::new()));

    // Infrastructure-thread handles, kept so the exit paths can JOIN them
    // (with `abort` raised on failure) instead of leaving them detached.
    let mut coord_driver: Option<thread::JoinHandle<()>> = None;
    let mut rdv_driver: Option<thread::JoinHandle<()>> = None;
    // Root cause slot for a coordinator that failed to START (formation
    // deadline, bind/handshake error): the coord thread records it here and
    // raises `abort`; the supervision verdict below prefers it over the
    // child exit statuses (the children died BECAUSE the brain did).
    let coord_fatal: Arc<std::sync::Mutex<Option<String>>> = Arc::new(std::sync::Mutex::new(None));

    // Ask ranks for resource samples when the harness carries a
    // Timeline that will persist them (host-qualified `rank_samples` in
    // timeline.json). Captured here because `coord_config` moves into
    // the coordinator below while the envelope-build loop runs after.
    let rank_resources = coord_config.as_ref().is_some_and(|c| c.timeline.is_some());

    if let Some(mut config) = coord_config {
        use crate::distributed::cluster_coordinator::ClusterCoordinator;

        // Launcher-side fields layered on top of the factory's config:
        // `local_ranks` (host-dependent: which global ranks landed on
        // the launcher's host) and `dead_ranks` (shared ledger with the
        // ClusterController already started above).
        let local_ranks: Vec<usize> = my_host_idx
            .map(|i| world.workers[i].ranks.clone())
            .unwrap_or_default();
        let dead_ranks = Arc::clone(&dead_ranks_shared);
        // Global-rank → host map so the coord can host-qualify the
        // rank-reported resource samples it deposits into the timeline.
        let mut rank_hosts: Vec<String> = vec![String::new(); world_size];
        for w in &world.workers {
            for &r in &w.ranks {
                if let Some(slot) = rank_hosts.get_mut(r) {
                    *slot = w.host.clone();
                }
            }
        }
        config = config
            .local_ranks(local_ranks.clone())
            .rank_hosts(rank_hosts)
            .dead_ranks(dead_ranks)
            .reported_deaths(Arc::clone(&reported_deaths));
        // Stamp the controller's world-map host name on the timeline so
        // post-hoc consumers can tell which rank_samples entries
        // duplicate the local poller's (dense) coverage. Only when the
        // launcher identifies itself in the world map — on rigs where
        // the box's `hostname` matches no cluster.yml worker name (e.g.
        // a controller co-located with a worker under a different
        // name), the honest answer is "unknown": we leave the stamp
        // absent rather than write a name (`me`) that appears nowhere
        // in the world, which no rank_samples host could match anyway.
        // Absent → the report treats every rank host as remote (its
        // documented fallback), so the co-located device surfaces as
        // both a dense-poller column and a rank-reported column.
        if let (Some(tl), Some(idx)) = (&config.timeline, my_host_idx) {
            tl.set_host(&world.workers[idx].host);
        }
        // A dedicated controller owns no local ranks: it must not report
        // GPU columns for devices another node trains on. Poll CPU/RAM
        // only, so every GPU column in the persisted timeline comes from
        // a worker's own `rank_samples`. When the controller IS
        // co-located with ranks (single-box auto-promote), local_ranks
        // is non-empty and its dense poll stays authoritative for them.
        if local_ranks.is_empty()
            && let Some(tl) = &config.timeline
        {
            tl.set_gpu_poll(false);
        }

        // Controller-hosted live dashboard. The sink owns a Monitor
        // that binds the HTTP port lazily on the first rank-emitted
        // `DashboardRegister` frame; absent that the sink stays idle
        // and the dashboard is simply never served.
        // Persisted record stream: opt-in via `record_log_dir`. The sink
        // writes each emitted record to its node's bounded log, so the
        // live stream and the on-disk history share one producer.
        let mut record_shipper = None;
        let record_log = config.record_log_dir.as_ref().map(|dir| {
            // Echo where the stream actually lands. A relative `--record-log`
            // resolves against the LAUNCHER's cwd, not the directory the user
            // typed it from -- which is how `ddp-bench/runs/x` silently becomes
            // `ddp-bench/ddp-bench/runs/x`. The directory is created lazily on
            // the first record, so resolve by joining cwd rather than
            // canonicalizing: the path legitimately does not exist yet.
            let resolved = std::path::Path::new(dir);
            let resolved = if resolved.is_absolute() {
                resolved.to_path_buf()
            } else {
                std::env::current_dir()
                    .map(|cwd| cwd.join(resolved))
                    .unwrap_or_else(|_| resolved.to_path_buf())
            };
            eprintln!("cluster launcher: record log -> {}", resolved.display());
            // The writer targets node-local disk; the shipper carries the
            // records to the configured destination on its own clock. With
            // `record_log_dir` on a shared mount (NFS/sshfs) a direct
            // writer would put every record append on the training-adjacent
            // path, where a hung mount strands the sink — the destination
            // must only ever stall the sacrificial shipper thread.
            let staged = crate::monitor::telemetry_dir("records");
            crate::verbose!(
                "cluster launcher: record log staged at {}",
                staged.display(),
            );
            record_shipper = Some(crate::monitor::TelemetryShipper::start(
                &staged,
                &resolved,
                crate::monitor::DEFAULT_SHIP_INTERVAL_MS,
            ));
            Arc::new(crate::monitor::record_log::RecordLog::new(
                staged,
                config
                    .max_log_size
                    .unwrap_or(crate::monitor::record_log::DEFAULT_MAX_LOG_BYTES),
            ))
        });
        let dashboard_sink: Arc<dyn crate::distributed::DashboardSink> = Arc::new(
            crate::distributed::ClusterDashboardSink::new(
                Arc::new(world.clone()),
                me.clone(),
                config.num_epochs,
            )
            // The engine scalars ride window records only, so their meta
            // declaration is honest exactly when the tick cadence is armed
            // — otherwise the stream (and the saved archive baking it)
            // would declare series no record ever carries.
            .with_engine_scalar_declarations(config.reports_per_epoch.is_some())
            .with_record_log(record_log)
            .with_record_shipper(record_shipper)
            .with_status_board(status_board.clone())
            .with_scalar_reductions(config.scalar_reductions.clone())
            .with_dashboard_html(config.dashboard_html.clone())
            .with_dashboard_theme(config.dashboard_theme.clone()),
        );
        dashboard_sink_outer = Some(Arc::clone(&dashboard_sink));
        config = config.dashboard_sink(Arc::clone(&dashboard_sink));

        let coord_salt = salt;
        eprintln!(
            "cluster launcher: ClusterCoordinator spawning on port {} \
             (world_size={}, local_ranks={:?})",
            mux_port, world_size, local_ranks,
        );
        // Capture the resume kickoff epoch before moving `config` into
        // `start()`. `start_epoch == 0` for fresh runs; resume runs
        // populate it from `CheckpointMeta::epoch` via
        // `ClusterCoordinatorConfig::resume_from_meta`.
        let start_epoch = config.start_epoch;
        // Attach the launcher's abort flag: the coord's cohort-formation
        // accept loop polls it (a pre-rendezvous failure means relays
        // never dial in), and the tick loop below checks it per
        // iteration — together they make this thread joinable from the
        // failure path instead of forcing process::exit(1).
        config = config.abort_flag(Arc::clone(&abort));
        let coord_abort = Arc::clone(&abort);
        let coord_fatal_slot = Arc::clone(&coord_fatal);
        let coord_source = crate::distributed::port_mux::StreamSource::Mux(mux_control);
        coord_driver = Some(thread::Builder::new()
            .name("flodl-cluster-coord".to_string())
            .spawn(move || {
                match ClusterCoordinator::start_from_source(
                    coord_source, mux_port, coord_salt, config,
                ) {
                    Ok(mut coord) => {
                        // Kickoff the first epoch dispatch. Without this,
                        // `tick()` never broadcasts `StartEpoch` to any
                        // rank and workers idle indefinitely in
                        // `wait_for_epoch_plan`. Resume runs pass
                        // `start_epoch = meta.epoch` to continue from
                        // the saved trajectory point.
                        //
                        // Coverage-granular resume first: if the loaded meta
                        // carried a coverage block, reconstruct the in-progress
                        // pools and dispatch only the uncovered remainder. When
                        // it handles the kickoff, skip the fresh full-epoch
                        // dispatch; otherwise fall back to it.
                        let kicked = match coord.resume_progressive_from_coverage() {
                            Ok(handled) => handled,
                            Err(e) => {
                                eprintln!(
                                    "cluster launcher: resume_progressive_from_coverage failed: {e}"
                                );
                                return;
                            }
                        };
                        if !kicked
                            && let Err(e) = coord.dispatch_epoch(start_epoch) {
                                eprintln!(
                                    "cluster launcher: dispatch_epoch({start_epoch}) failed: {e}"
                                );
                                return;
                            }
                        // Drive ticks until shutdown_workers fires (all
                        // ranks exited) or the process is killed. Paced
                        // on a short blocking timing-drain instead of a
                        // 100% busy-spin: the spin pegged one controller
                        // core (competing with rank 0's data pipeline —
                        // the documented starve lever on slow-PCIe rigs)
                        // and amplified every per-tick allocation
                        // millions of times per second for zero work.
                        // 2ms keeps reduce latency negligible while the
                        // blocking recv yields the core between frames.
                        loop {
                            // Launcher abort (failure path): bounded exit
                            // within one 2ms drain, so the failure path can
                            // join this thread and return Err.
                            if coord_abort.load(std::sync::atomic::Ordering::SeqCst) {
                                break;
                            }
                            coord.drain_timing_blocking(
                                std::time::Duration::from_millis(2),
                            );
                            match coord.tick() {
                                Ok(true) => continue,
                                Ok(false) => break,
                                Err(e) => {
                                    eprintln!(
                                        "cluster launcher: coord tick error: {e}"
                                    );
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "cluster launcher: ClusterCoordinator start failed: {e}"
                        );
                        // The run just lost its brain before it formed: no
                        // coordinator means nothing will ever consume death
                        // reports or broadcast Shutdown to the relays, so
                        // supervision would wait on agents that wait on
                        // relays that wait on ranks — forever (the
                        // formation-window wedge, observed live). Record
                        // the root cause and raise the run-wide abort so
                        // the child watchers terminate the cohort and the
                        // launcher surfaces THIS error.
                        if let Ok(mut slot) = coord_fatal_slot.lock() {
                            *slot = Some(e.to_string());
                        }
                        coord_abort.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            })
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster launcher: spawn coord thread failed: {e}"
                ))
            })?);
    } else {
        // No coordinator this run (legacy NCCL routing). Dropping the
        // receiver makes the mux dispatcher reset any stray control
        // dialer immediately instead of queueing it forever.
        drop(mux_control);
    }

    // Bootstrap rendezvous server (on the mux port). Every rank
    // dials in, the controller designates one rank as NCCL-UID generator
    // (default: a local-host worker's first rank if any, else
    // `workers[0].ranks[0]`), then broadcasts the UID.
    //
    // Cohort-formation gate for elastic child supervision. Pre-formation
    // a rank death must kill-all (peers are blocked in NCCL's
    // connect-retry with no comm to abort and no rebuild machinery
    // running); post-formation the coordinator's elastic membership owns
    // the decision. NCCL: flipped by the rendezvous thread after a
    // successful bootstrap. CPU: true from the start — there is no
    // init-hang window (the controller's round-wait polls the shared
    // ledger, so a pre-round death cannot wedge survivors).
    let cohort_formed = Arc::new(std::sync::atomic::AtomicBool::new(!backend_is_nccl));
    if backend_is_nccl {
        let rdv_full = world.clone();
        let rdv_me = me.clone();
        let formed_for_rdv = Arc::clone(&cohort_formed);
        let rdv_abort = Arc::clone(&abort);
        let rdv_source = crate::distributed::port_mux::StreamSource::Mux(mux_rendezvous);
        rdv_driver = Some(
            thread::Builder::new()
                .name("flodl-cluster-rendezvous".to_string())
                .spawn(move || {
                    match crate::distributed::rendezvous::run_controller_rendezvous_aborting(
                        &rdv_full, &rdv_me, rdv_source, &rdv_abort,
                    ) {
                        Ok(()) => formed_for_rdv.store(true, std::sync::atomic::Ordering::SeqCst),
                        Err(e) => {
                            eprintln!("cluster launcher: rendezvous server error: {e}");
                        }
                    }
                })
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster launcher: spawn rendezvous thread failed: {e}"
                    ))
                })?,
        );
    } else {
        // No rendezvous subsystem this run (CPU backend). Dropping the
        // receiver makes the mux dispatcher reset any stray rendezvous
        // dialer immediately instead of queueing it forever.
        drop(mux_rendezvous);
    }

    // Cohort inventory snapshot for post-run consumers in this process
    // (e.g. a bench harness writing a hardware header covering remote
    // hosts it can never probe itself). Captured here because the
    // admitted-worker list is consumed just below.
    let _ = COHORT_INVENTORY.set(formed_workers.iter().map(|aw| aw.member.clone()).collect());

    // ------------------------------------------------------------------
    // Ship each admitted worker its spawn artifacts
    // ------------------------------------------------------------------
    // The envelope and relay spec are byte-identical to what the direct
    // fan-out used to inject at spawn time; only the delivery changed
    // (the join connection instead of env/stdin). The connection then
    // stays open as the host control link: `RankExited` reports flow up
    // it into the coordinator's fast death queue.
    let frame_ceiling = crate::distributed::wire::frame_ceiling();
    let mut rank_exit_readers: Vec<thread::JoinHandle<()>> = Vec::new();
    // Write halves of the host control links, kept OUTSIDE the reader
    // threads: the failure path below uses them to send each connected
    // host an Abort (its agent's teardown trigger) and shut the socket
    // down, which also unblocks that link's reader. Best-effort — a
    // link whose clone failed just closes later, at agent exit.
    let mut worker_links: Vec<(String, std::net::TcpStream)> = Vec::new();
    for (idx, aw) in formed_workers.into_iter().enumerate() {
        let worker = &world.workers[idx];
        let member = aw.member;
        let mut stream = aw.stream;
        if let Ok(link) = stream.try_clone() {
            worker_links.push((member.host.clone(), link));
        }
        let dial_host = controller_dial_host(worker);
        let envelope = build_slim_envelope_for(&world, worker, &dial_host, rank_resources);
        let envelope_hex = crate::distributed::cluster::hex_encode(
            serde_json::to_string(&envelope)
                .map_err(|e| {
                    TensorError::new(&format!(
                        "cluster launcher: serialize slim envelope failed: {e}"
                    ))
                })?
                .as_bytes(),
        );
        let relay_spec_hex = if has_coord {
            let spec = RelaySpec {
                host: member.host.clone(),
                controller_host: dial_host,
                controller_port: mux_port,
                ranks: member.ranks.iter().map(|r| *r as u32).collect(),
                salt_hex: crate::distributed::wire::salt_to_hex(&salt),
                world_size,
                data_channel: relay_data_channel,
                frame_ceiling_bytes: frame_ceiling,
            };
            Some(crate::distributed::cluster::hex_encode(
                serde_json::to_string(&spec)
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster launcher: serialize relay spec failed: {e}"
                        ))
                    })?
                    .as_bytes(),
            ))
        } else {
            None
        };
        let msg = crate::distributed::wire::JoinMsgWire::WorldFormed {
            envelope_hex,
            relay_spec_hex,
        };
        let send = crate::distributed::wire::ControlFrame::encode(
            &salt,
            crate::distributed::wire::MsgKind::Join,
            &msg,
        )
        .and_then(|f| f.write_to(&mut stream));
        if let Err(e) = send {
            // The worker died between admission and formation: a
            // post-formation membership event. Report its ranks dead and
            // let elastic membership (or the NCCL rendezvous idle
            // timeout) take it from here.
            eprintln!(
                "cluster launcher: WorldFormed to {:?} failed ({e}); \
                 reporting its rank(s) {:?} dead",
                member.host, member.ranks,
            );
            if let Ok(mut q) = reported_deaths.lock() {
                q.extend(member.ranks.iter().copied());
            }
            continue;
        }
        // Host control link reader: per-rank exit reports feed the
        // coordinator's fast death queue (non-zero exits only — clean
        // exits are handled by the normal Exiting control flow). EOF is
        // the host closing shop; benign here because host-death shows up
        // through agent exit (remote) or direct child exits (local).
        let reader_deaths = Arc::clone(&reported_deaths);
        let reader_salt = salt;
        let reader_host = member.host.clone();
        rank_exit_readers.push(thread::spawn(move || {
            let _ = stream.set_read_timeout(None);
            loop {
                match crate::distributed::wire::ControlFrame::read_from(&mut stream, &reader_salt) {
                    Ok(Some(frame)) => {
                        match frame.decode::<crate::distributed::wire::JoinMsgWire>() {
                            Ok(crate::distributed::wire::JoinMsgWire::RankExited {
                                rank,
                                code,
                            }) if code != 0 => {
                                eprintln!(
                                    "cluster launcher: host {reader_host:?} reports \
                                     rank {rank} exited with code {code}; feeding \
                                     elastic membership"
                                );
                                if let Ok(mut q) = reader_deaths.lock() {
                                    q.push(rank as usize);
                                }
                            }
                            Ok(_) => {}
                            Err(e) => {
                                crate::verbose!(
                                    "  cluster launcher: control-link decode from \
                                     {reader_host:?}: {e}"
                                );
                            }
                        }
                    }
                    Ok(None) | Err(_) => return,
                }
            }
        }));
    }

    // ------------------------------------------------------------------
    // Supervision
    // ------------------------------------------------------------------
    let ranks_by_host: std::collections::BTreeMap<String, Vec<usize>> = world
        .workers
        .iter()
        .map(|w| (w.host.clone(), w.ranks.clone()))
        .collect();
    let mut children: Vec<spawn::SupervisedChild> = Vec::new();
    let mut collect_err: Option<TensorError> = None;
    for (host, handle) in local_joins.iter_mut() {
        let joined = handle
            .take()
            .expect("local join handle consumed once")
            .join();
        match joined {
            Ok(Ok(host_children)) => {
                let host_ranks = ranks_by_host.get(host).cloned().unwrap_or_default();
                for hc in host_children {
                    let granks = match hc.rank {
                        Some(r) => vec![r as usize],
                        None => host_ranks.clone(),
                    };
                    children.push((host.clone(), hc.slot, granks, hc.child, hc.forwarders));
                }
            }
            Ok(Err(e)) => {
                collect_err.get_or_insert_with(|| {
                    TensorError::new(&format!(
                        "cluster launcher: local worker {host:?} failed to spawn \
                         its children: {e}"
                    ))
                });
            }
            Err(_) => {
                collect_err.get_or_insert_with(|| {
                    TensorError::new(&format!(
                        "cluster launcher: local join thread for {host:?} panicked"
                    ))
                });
            }
        }
    }
    for (host, child, forwarders) in remote_agents.drain(..) {
        let granks = ranks_by_host.get(&host).cloned().unwrap_or_default();
        children.push((host, AGENT_RANK_SENTINEL, granks, child, forwarders));
    }
    if let Some(e) = collect_err {
        // A local worker failed post-formation: the world can't start
        // coherently. Kill everything spawned and take the cooperative
        // teardown path below.
        eprintln!("{e}");
        for (_, _, _, mut child, forwarders) in children.drain(..) {
            let _ = child.kill();
            let _ = child.wait();
            for f in forwarders {
                let _ = f.join();
            }
        }
        cleanup_remote_hosts_parallel(remote_cleanup_targets.clone());
        abort.store(true, std::sync::atomic::Ordering::SeqCst);
        // Walk-in hosts are not covered by the process kills above —
        // tell them to tear down and unblock their link readers (same
        // mechanics as the main failure path below).
        abort_worker_links(&mut worker_links, &salt, &e.to_string());
        if let Some(h) = coord_driver.take() {
            let _ = h.join();
        }
        if let Some(h) = rdv_driver.take() {
            let _ = h.join();
        }
        for r in rank_exit_readers {
            let _ = r.join();
        }
        if let Some(h) = status_server.take() {
            let _ = h.join();
        }
        return Err(e);
    }

    // Concurrent supervision: watch every child on its own thread and
    // collect exit events on an mpsc channel. The first non-zero exit
    // pre-formation triggers SIGKILL on every other still-running child;
    // post-formation, elastic membership owns the decision.
    membership_state.phase = membership::ClusterPhase::Training;
    status_board.publish(&membership_state);
    let any_failure = supervise_children(
        children,
        has_coord.then(|| ElasticSupervision {
            reported_deaths: Arc::clone(&reported_deaths),
            dead_ranks: Arc::clone(&dead_ranks_shared),
            max_failure: elastic_max_failure,
            world_size,
            cohort_formed: Arc::clone(&cohort_formed),
        }),
        // Run-wide abort: raised by the coord thread on a start failure
        // (formation deadline) and by the local-spawn failure path — the
        // watchers observe it and terminate their children, so supervision
        // is released instead of waiting on a brainless cohort.
        Some(Arc::clone(&abort)),
    );
    // Walk-in workers (`fdl join`) are nobody's process children: their
    // only tie to this launcher is the host control link, so its EOF IS
    // their exit event. `supervise_children` returning does NOT mean
    // the world is done — a discovery cohort can have ZERO process
    // children, making supervision return the instant it starts; the
    // teardown tail below would then abort a world that just formed
    // (observed live: coordinator killed mid-start, ranks reset
    // mid-write). Wait for every host control link to close before
    // declaring the run over. On a failure verdict, don't wait — tell
    // every connected host to tear down (the Abort frame is the agent's
    // teardown trigger) and shut the sockets, which also unblocks the
    // readers' blocking reads (this was the failure path's wedge: the
    // final reader join used to block on agents nobody had told to
    // die). A host that vanished without closing its TCP link can stall
    // the success-path wait — the same exposure the coordinator's
    // heartbeat machinery owns for the training plane, accepted here.
    if let Some(err) = &any_failure {
        abort_worker_links(&mut worker_links, &salt, &err.to_string());
    }
    for r in rank_exit_readers {
        let _ = r.join();
    }
    drop(worker_links);

    // A coordinator start failure is the root cause of everything the
    // watchers just reaped — surface it as THE run error rather than the
    // downstream child kill statuses. Folded AFTER the link wait: a
    // coordinator that fails while walk-in hosts are still connected
    // (e.g. its start aborts and the resulting cascade kills their
    // ranks) records the root cause DURING the wait — folding earlier
    // read `None` and declared the run a success (observed live: a
    // relay-less formation reported `done` with zero syncs).
    let any_failure = match coord_fatal.lock().ok().and_then(|mut s| s.take()) {
        Some(root) => Some(TensorError::new(&format!(
            "cluster launcher: coordinator failed at formation — {root}"
        ))),
        None => any_failure,
    };

    // Terminal phase, published while the status endpoint is still up:
    // the cleanup passes below can take seconds, and a `fdl status`
    // poll landing in them should read `done`/`failed`, not a stale
    // `training`. Once this function returns, the endpoint is gone and
    // connection-refused becomes the (honest) "no run" signal.
    membership_state.phase = if any_failure.is_some() {
        membership::ClusterPhase::Failed
    } else {
        membership::ClusterPhase::Done
    };
    status_board.publish(&membership_state);

    // Post-exit cleanup: belt-and-braces ssh-pkill on every remote host.
    // The remote-side trap wrapper handles SIGHUP-on-disconnect, but
    // that path waits for sshd's keepalive timeout (~30s) and only
    // triggers if SIGHUP is actually delivered (varies by sshd config).
    // This explicit pass fires immediately, so the user sees no leftover
    // process on the remote when the launcher returns — including agent
    // children, which run the same binary and match the same pattern.
    cleanup_remote_hosts_parallel(remote_cleanup_targets);

    // All children exited; flush the dashboard's SSE `complete` event
    // before the launcher process tears down so connected browsers
    // stop the elapsed counter and switch to "done". Safe even when
    // the sink was never registered (server stays None ⇒ no-op).
    if let Some(ref sink) = dashboard_sink_outer {
        sink.shutdown();
    }

    // All children exited; signal ClusterController shutdown and join.
    if let Err(e) = cpu_averager.shutdown() {
        // Don't mask a child-failure error with a ClusterController shutdown
        // error; log + continue. The child failure is the load-bearing
        // diagnostic.
        eprintln!("cluster launcher: ClusterController shutdown failed: {e}");
    }

    // Cooperative infrastructure teardown (both paths). Raise the abort
    // flag, then JOIN the coordinator + rendezvous threads:
    // - Success: both have typically exited already (coord tick returns
    //   Ok(false) once every rank stream EOFs; the rendezvous thread is
    //   short-lived) — the join is a no-wait formality that guarantees no
    //   infrastructure thread outlives this call in a library embedder.
    // - Failure: the abort flag wakes the coord's cohort-formation accept
    //   poll, its tick loop, and the rendezvous accept poll within one
    //   poll interval, so we can join and RETURN the error — a library
    //   embedder catches it via DdpHandle::join, and CLI consumers exit
    //   non-zero through their own main.
    abort.store(true, std::sync::atomic::Ordering::SeqCst);
    if let Some(h) = coord_driver.take() {
        let _ = h.join();
    }
    if let Some(h) = rdv_driver.take() {
        let _ = h.join();
    }
    if let Some(h) = status_server.take() {
        let _ = h.join();
    }
    use std::io::Write as _;
    let _ = std::io::stderr().flush();
    let _ = std::io::stdout().flush();
    if let Some(err) = any_failure {
        eprintln!("cluster launcher: fatal failure: {err}");
        let _ = std::io::stderr().flush();
        return Err(err);
    }
    Ok(())
}
