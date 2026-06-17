//! Cluster launcher: role detection + fan-out + controller orchestration.
//!
//! Slots transparently into [`Trainer::setup`] and friends on cluster-mode
//! startup. Each user-binary invocation routes to one of three roles:
//!
//! - **Launcher**: the parent process that fdl-cli execs after parsing
//!   `fdl.yml`. Reads the full cluster topology, spawns one child per rank
//!   (fork/exec for local hosts, ssh for remote), starts the controller
//!   thread (TCP byte router for CPU averaging + log fan-in), waits for
//!   every child to exit, then exits itself.
//!
//! - **Rank**: a spawned child running the user's training code. Inherits
//!   the slim per-host envelope and the rank-slot env var injected by the
//!   launcher; existing [`Trainer::setup`] cluster-path logic handles the
//!   rest (rendezvous, `Ddp::wrap`, training loop).
//!
//! - **Single-device**: no cluster envelope in env. Caller continues with
//!   today's single-device path. Bit-identical to pre-cluster behavior.
//!
//! # Wire protocol (env vars)
//!
//! Two env vars distinguish the launcher and rank roles. The names are
//! deliberately namespaced so a fdl-cli invocation in a cluster context
//! never sets them both at the same time:
//!
//! - [`ENV_FULL_CLUSTER_JSON`] (`FLODL_FULL_CLUSTER_JSON`): hex-encoded
//!   JSON of the *full* cluster topology (all hosts + ranks + devices).
//!   Set by fdl-cli when invoking the user binary as the launcher. The
//!   launcher reads it once to drive fan-out; never propagated to rank
//!   children.
//!
//! - [`crate::distributed::cluster::ENV_CLUSTER_JSON`]
//!   (`FLODL_CLUSTER_JSON`): hex-encoded slim per-host envelope, mirroring
//!   the existing rank-side wire format. Set by the launcher (not fdl-cli)
//!   when spawning each rank child. Read by [`LocalCluster::from_env`].
//!
//! - [`crate::distributed::cluster::ENV_LOCAL_RANK`] (`FLODL_LOCAL_RANK`):
//!   integer index into the slim envelope's `host.ranks`. Set by the
//!   launcher when spawning each rank child. Read by
//!   [`crate::distributed::cluster::LocalCluster::my_rank`].
//!
//! Role detection table:
//!
//! | `FLODL_FULL_CLUSTER_JSON` | `FLODL_CLUSTER_JSON` | `FLODL_LOCAL_RANK` | Role |
//! |---|---|---|---|
//! | unset | unset | unset | [`Role::SingleDevice`] |
//! | unset | set | set | [`Role::Rank`] |
//! | set | unset | unset | [`Role::Launcher`] (caller drives the fan-out) |
//! | other combinations | | | loud error |
//!
//! # Design notes
//!
//! The "two-env-var" wire protocol is the smallest additive change to
//! today's setup. Slim per-rank envelopes stay the same shape on the
//! rank side, so [`LocalCluster::from_env`] needs no change. The new
//! launcher-side parser ([`FullCluster::from_env`]) consumes the full
//! topology in a separate path. A future cleanup could unify both
//! shapes; for 4b the additive form keeps the blast radius small.
//!
//! [`Trainer::setup`]: crate::distributed::Trainer::setup
//! [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env

use std::env;
use std::process::Stdio;
use std::sync::Arc;
use std::thread;

use serde::{Deserialize, Serialize};

use crate::distributed::relay::agent::{ChannelKind, RelayChannel};
use crate::distributed::relay::{RELAY_CONTROL_LOOPBACK_OFFSET, RELAY_DATA_LOOPBACK_OFFSET};
use crate::tensor::{Result, TensorError};

mod spawn;
mod types;
#[cfg(test)]
mod tests;

pub use types::{SshConfig, FullCluster, FullController, FullWorker};

use spawn::{
    load_prebuild_envelope, supervise_children, build_local_spawn_command,
    build_ssh_spawn_command, cleanup_remote_hosts_parallel, build_remote_bash_command,
    build_local_relay_command, build_remote_relay_bash_command,
    build_slim_envelope_for, forward_lines,
};


/// Environment variable carrying the *full* cluster topology to the
/// launcher process. Set by fdl-cli; consumed only by [`dispatch`]. Not
/// propagated to rank children (each child gets a slim per-host envelope
/// instead via `FLODL_CLUSTER_JSON`).
pub const ENV_FULL_CLUSTER_JSON: &str = "FLODL_FULL_CLUSTER_JSON";

/// Environment variable carrying the per-host relay spec (hex-encoded
/// JSON [`RelaySpec`]). Set by the launcher on the relay child it spawns
/// per host; consumed only by [`dispatch`] (→ [`Role::Relay`]) and
/// [`run_relay`]. Mutually exclusive with the launcher/rank env vars.
pub const ENV_RELAY_JSON: &str = "FLODL_RELAY_JSON";

/// Environment variable carrying the fdl command name (e.g. `train`) the
/// launcher should invoke on remote hosts via `ssh ... fdl <cmd>`. Set by
/// fdl-cli when invoking the user binary as a launcher; required by the
/// ssh fan-out path. Local fork+exec doesn't consume this — the launcher
/// re-execs `current_exe()` directly with its own argv.
pub const ENV_FDL_CMD: &str = "FLODL_FDL_CMD";

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
pub const ENV_PREBUILD_PER_HOST: &str = "FLODL_PREBUILD_PER_HOST";

/// Role this process plays in the cluster, decided by [`dispatch`].
///
/// `dispatch` is a pure role detector — it never runs the launcher or
/// the rank loop itself. The caller drives both:
///
/// - On [`Role::Launcher`], the caller assembles the controller-scope
///   config (typically from the user's `DdpRunConfig` via
///   `super::ddp_run::build_coord_config_from_builder`), then calls
///   [`run_launcher_with_config`] and `std::process::exit(0)` when it
///   returns. This is the "launcher trampoline": the user's `main()`
///   ran up to the `Trainer::builder(...).run()` boundary, which gives
///   the dispatch site native access to `Box<dyn ConvergenceGuard>` and
///   ElChe knobs that can't cross process boundaries.
/// - On [`Role::Rank`] / [`Role::SingleDevice`], the caller proceeds
///   with the training body.
#[derive(Debug, PartialEq, Eq)]
pub enum Role {
    /// No cluster envelope in env. Continue with today's single-device
    /// training path.
    SingleDevice,
    /// This process is a rank. Continue with cluster-mode training
    /// (`Trainer::setup` will read the slim envelope and rendezvous).
    Rank,
    /// This process is the launcher. Caller must run the fan-out via
    /// [`run_launcher_with_config`] and exit the program when it returns.
    Launcher,
    /// This process is a per-host transport relay. Caller must run
    /// [`run_relay`] and exit the program when it returns. Touches no
    /// CUDA; multiplexes its local ranks' frames to the controller.
    Relay,
}

/// Detect this process's role from env vars. Pure function — no I/O,
/// no thread spawns, no process forks. The caller drives whatever
/// action the role demands (see [`Role`]).
///
/// Loud error on inconsistent env (e.g. both full-cluster and rank-slot
/// vars set — silently winning one over the other costs hours of
/// debugging on a misconfigured rig).
///
/// [`Trainer::setup`]: crate::distributed::Trainer::setup
pub fn dispatch() -> Result<Role> {
    let relay_set = env::var_os(ENV_RELAY_JSON).is_some();
    let full_set = env::var_os(ENV_FULL_CLUSTER_JSON).is_some();
    let slim_set = env::var_os(crate::distributed::cluster::ENV_CLUSTER_JSON).is_some();
    let slot_set = env::var_os(crate::distributed::cluster::ENV_LOCAL_RANK).is_some();

    match (relay_set, full_set, slim_set, slot_set) {
        (false, false, false, false) => Ok(Role::SingleDevice),
        (false, false, true, true) => Ok(Role::Rank),
        (false, true, false, false) => Ok(Role::Launcher),
        (true, false, false, false) => Ok(Role::Relay),
        // Any other combination is a misconfiguration. Loud error with
        // every bit named so the operator can see what's off.
        _ => Err(TensorError::new(&format!(
            "cluster launcher: inconsistent env (FLODL_RELAY_JSON={}, \
             FLODL_FULL_CLUSTER_JSON={}, FLODL_CLUSTER_JSON={}, FLODL_LOCAL_RANK={}). \
             Expected: all-unset (single-device), slim+slot only (rank), \
             full only (launcher), or relay only (relay).",
            on_off(relay_set),
            on_off(full_set),
            on_off(slim_set),
            on_off(slot_set),
        ))),
    }
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
    /// Controller base port: relay dials `+2` (data) / `+3` (control) and
    /// binds loopback `+4` / `+5`.
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
}

/// Run this process as a per-host transport relay: bind the loopback data
/// (`+4`) / control (`+5`) channels its local ranks dial, forward upstream
/// to the controller's `+2` / `+3`, and stay up until every local rank
/// disconnects (training finished). Touches no CUDA.
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
    let (ctrl_listener, _) = RelayChannel::bind(loopback(RELAY_CONTROL_LOOPBACK_OFFSET)?)?;
    let ctrl_upstream = resolve(&spec.controller_host, base.saturating_add(3))?;

    // Data channel (CPU backends only). Ranks connect to BOTH channels
    // (data first, then control in the CPU path), so the two accept loops
    // must run concurrently — the data relay runs on its own thread.
    let data_handle = if spec.data_channel {
        let (data_listener, _) = RelayChannel::bind(loopback(RELAY_DATA_LOOPBACK_OFFSET)?)?;
        let data_upstream = resolve(&spec.controller_host, base.saturating_add(2))?;
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


/// spawn a [`ClusterCoordinator`], fork rank children, wait for them to exit.
///
/// `coord_config` carries the user's controller-scope configuration — the
/// guard, ElChe knobs, policy, partition ratios — assembled by the
/// launcher-trampoline caller from the user's `DdpRunConfig`. `None`
/// preserves the legacy "no coord spawn" path (rank-side via_coord
/// routing is governed by `save_path` on `DdpRunConfig` per the
/// `DdpHandle::launch` `auto_with` flip, not by an env var anymore).
///
/// Local hosts (`host.host == this_hostname`) get fork+exec of
/// `current_exe()` with env vars set directly. Remote hosts get
/// `ssh <target> bash -lc '<remote_cmd>'`, where `<remote_cmd>` exports
/// env vars and execs `fdl <cmd>` — same shape fdl-cli used to use
/// before this lift. Both produce identical child semantics (piped
/// streams, `[host:dev:rN]` line-prefix on stdout/stderr).
///
/// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
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

pub fn run_launcher_with_config(
    full: FullCluster,
    mut coord_config: Option<crate::distributed::cluster_coordinator::ClusterCoordinatorConfig>,
) -> Result<()> {
    claim_cluster_entry("launcher")?;
    // Fresh 128-bit session salt per launcher invocation. Becomes the
    // HMAC key for every cross-process control + data frame; shipped
    // to ranks via their slim envelope.
    let salt = crate::distributed::wire::generate_session_salt();
    let full = full.with_session_salt(salt);
    let me = crate::distributed::cluster::resolve_hostname()?;

    // Controller-not-in-workers is the canonical pattern post controller-active
    // refactor: the controller drives orchestration, workers carry ranks.
    // Co-locating the controller with a worker host is still supported (via
    // hostname match below), it is just no longer the expected default.
    let my_host_idx = full.workers.iter().position(|h| h.host == me);

    // Start the ClusterController TCP server on controller_port + 2 so any rank
    // using AverageBackend::Cpu can connect. Bound to 0.0.0.0 so remote
    // ranks reach it; local ranks use the same address via loopback.
    //
    // Always started, even on NCCL-only clusters: the accept loop polls
    // a shutdown flag every 20ms, so an unused ClusterController exits cleanly
    // when launcher signals shutdown after children finish. Cost is one
    // idle thread + one bound port.
    let cpu_avg_port = full.controller.port.saturating_add(2);
    let cpu_avg_addr: std::net::SocketAddr = format!("0.0.0.0:{cpu_avg_port}")
        .parse()
        .map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: invalid ClusterController bind addr 0.0.0.0:{cpu_avg_port}: {e}"
            ))
        })?;
    // Shared dead-rank ledger between ClusterController (CPU averaging
    // releases on heartbeat-stale) and ClusterCoordinator (NCCL
    // elastic-membership rendezvous trigger). Both consumers see the
    // same source of truth. Always constructed even on legacy NCCL
    // runs — the cost is negligible (a Vec<AtomicBool>) and the wiring
    // keeps both backends pluggable.
    let dead_ranks_shared =
        crate::distributed::controller::DeadRanks::new(full.world_size());
    // Consensus-checkpoint forge: holds the launch-captured model schema so the
    // controller reduce thread can write a named `.fdl` from the averaged
    // (name-less) frame. Shared with the coordinator (which arms it before a
    // checkpoint reduce) — same Arc-sharing pattern as `dead_ranks`. Take the
    // schema out of the coord config (the coord never writes the model itself).
    let model_schema = coord_config
        .as_mut()
        .and_then(|c| c.model_schema.take());
    let checkpoint_forge =
        crate::distributed::CheckpointForge::new(model_schema);
    if let Some(cfg) = coord_config.as_mut() {
        cfg.checkpoint_forge = Some(Arc::clone(&checkpoint_forge));
    }
    let cpu_averager =
        crate::distributed::controller::ClusterController::start_with_dead_ranks(
            cpu_avg_addr,
            full.world_size(),
            full.salt,
            Arc::clone(&dead_ranks_shared),
            Some(Arc::clone(&checkpoint_forge)),
        )?;
    // Bound port stays the configured value (no kernel auto-assign here);
    // log it once for diagnostics.
    eprintln!(
        "cluster launcher: ClusterController bound on {} (world_size={})",
        cpu_averager.port(),
        full.world_size()
    );

    // ClusterCoordinator spawn at controller_port + 3 for the elastic-
    // membership-aware NCCL path. `coord_config = Some(...)` means the
    // caller (the trampoline at `DdpHandle::launch`) built the
    // controller-scope config from the user's `DdpRunConfig` and wants
    // a coord spawned. `None` skips the coord — legacy NCCL routing
    // (worker self-driven ElChe, no elastic membership) handles that
    // path entirely on the rank side.
    //
    // The spawned thread blocks on `start_from_listener.accept()` until
    // `world_size` ranks connect; if the via-coord routing isn't
    // exercised the thread sits idle until the launcher process exits
    // (process-exit kills the thread; no graceful shutdown plumbed yet).
    // Hoisted so `cluster_dashboard_sink.shutdown()` can fire after
    // children exit (emits the SSE `complete` event so connected
    // browsers stop the elapsed counter). `None` when coord_config is
    // None — legacy NCCL routing path with no dashboard wiring.
    let mut dashboard_sink_outer:
        Option<Arc<dyn crate::distributed::DashboardSink>> = None;

    // Decide relay spawn BEFORE `coord_config` is consumed below. A
    // per-host relay is spawned whenever a coordinator is in play (the
    // via-coord routing that uses the controller/coord channels). The
    // data-channel relay is needed only for the CPU averaging backend
    // (NCCL ranks never dial the data channel).
    let spawn_relays = coord_config.is_some();
    let relay_data_channel = coord_config
        .as_ref()
        .map(|c| matches!(c.backend, crate::distributed::ddp_run::AverageBackend::Cpu))
        .unwrap_or(false);
    // The NCCL-UID bootstrap rendezvous (port +0) is dialed only by NCCL
    // ranks. CPU-averaging ranks never connect, so spawning it on a CPU
    // backend just idles the accept loop to its timeout and logs a
    // spurious "rendezvous: timed out ... (0/N ranks in)" error while
    // training proceeds fine over the control/data channels. Gate the
    // spawn on the backend. Unknown backend (no `coord_config`) defaults
    // to spawning, preserving prior behavior for non-coordinator paths.
    let backend_is_nccl = coord_config
        .as_ref()
        .map(|c| matches!(c.backend, crate::distributed::ddp_run::AverageBackend::Nccl))
        .unwrap_or(true);

    if let Some(mut config) = coord_config {
        use crate::distributed::cluster_coordinator::ClusterCoordinator;

        let coord_port = full.controller.port.saturating_add(3);
        let coord_bind_addr: std::net::SocketAddr = format!("0.0.0.0:{coord_port}")
            .parse()
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster launcher: invalid ClusterCoordinator bind addr \
                     0.0.0.0:{coord_port}: {e}"
                ))
            })?;
        // Launcher-side fields layered on top of the caller's config:
        // `local_ranks` (host-dependent: which global ranks are on the
        // launcher's host) and `dead_ranks` (shared ledger with the
        // ClusterController already started above). The caller built
        // the controller-scope fields (policy, ElChe, guard, etc.) but
        // can't know these two — only the launcher does.
        let local_ranks: Vec<usize> = my_host_idx
            .map(|i| full.workers[i].ranks.clone())
            .unwrap_or_default();
        let dead_ranks = Arc::clone(&dead_ranks_shared);
        config = config
            .local_ranks(local_ranks.clone())
            .dead_ranks(dead_ranks);

        // Controller-hosted live dashboard. The sink owns a Monitor
        // that binds the HTTP port lazily on the first rank-emitted
        // `DashboardRegister` frame (the rank's `monitor.serve(port)`
        // call from user code triggers that emit; absent that the sink
        // stays idle and the dashboard is simply never served). The
        // sink itself is cheap (an unbound Monitor + per-rank state
        // maps) so we construct unconditionally and let the wire path
        // decide whether to bind.
        let dashboard_sink: Arc<dyn crate::distributed::DashboardSink> =
            Arc::new(crate::distributed::ClusterDashboardSink::new(
                Arc::new(full.clone()),
                me.clone(),
                config.num_epochs,
            ));
        dashboard_sink_outer = Some(Arc::clone(&dashboard_sink));
        config = config.dashboard_sink(Arc::clone(&dashboard_sink));
        // Heartbeat timeout: now flows from `DdpRunConfig.heartbeat_timeout_secs`
        // through `build_coord_config_from_builder` — no env var override.

        let coord_salt = full.salt;
        let coord_world = full.world_size();
        eprintln!(
            "cluster launcher: ClusterCoordinator spawning on {} (world_size={}, \
             local_ranks={:?})",
            coord_bind_addr, coord_world, local_ranks,
        );
        // Capture the resume kickoff epoch before moving `config` into
        // `start()`. `start_epoch == 0` for fresh runs; resume runs
        // populate it from `CheckpointMeta::epoch` via
        // `ClusterCoordinatorConfig::resume_from_meta`.
        let start_epoch = config.start_epoch;
        let _ = thread::Builder::new()
            .name("flodl-cluster-coord".to_string())
            .spawn(move || {
                match ClusterCoordinator::start(coord_bind_addr, coord_salt, config) {
                    Ok(mut coord) => {
                        // Kickoff the first epoch dispatch. Without this,
                        // `tick()` never broadcasts `StartEpoch` to any
                        // rank and workers idle indefinitely in
                        // `wait_for_epoch_plan`. Mirrors the threaded
                        // coordinator's `coord.send_all_plans(0)` in
                        // `orchestrator.rs`; resume runs pass
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
                        if !kicked {
                            if let Err(e) = coord.dispatch_epoch(start_epoch) {
                                eprintln!(
                                    "cluster launcher: dispatch_epoch({start_epoch}) failed: {e}"
                                );
                                return;
                            }
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
                    }
                }
            })
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster launcher: spawn coord thread failed: {e}"
                ))
            })?;
    }

    // Bootstrap rendezvous server (port +0). Controller binds, every rank
    // dials in, the controller designates one rank as NCCL-UID generator
    // (default: a local-host worker's first rank if any, else
    // `workers[0].ranks[0]`), then broadcasts the UID. The controller
    // cannot call `ncclGetUniqueId` itself — its process is
    // orchestration-only — so it delegates to a rank, same pattern as
    // elastic resize's `RequestNewNcclId` path.
    //
    // Spawned as a short-lived thread that exits once every rank has
    // its UID. If it errors, ranks fail to rendezvous and surface their
    // own loud errors; we eprintln any failure here for diagnostics.
    // Skipped entirely on CPU backends, which never dial it in.
    if backend_is_nccl {
        let rdv_full = full.clone();
        let rdv_me = me.clone();
        let _ = thread::Builder::new()
            .name("flodl-cluster-rendezvous".to_string())
            .spawn(move || {
                if let Err(e) =
                    crate::distributed::rendezvous::run_controller_rendezvous(&rdv_full, &rdv_me)
                {
                    eprintln!("cluster launcher: rendezvous server error: {e}");
                }
            })
            .map_err(|e| {
                TensorError::new(&format!(
                    "cluster launcher: spawn rendezvous thread failed: {e}"
                ))
            })?;
    }

    // For remote hosts, fdl-cli must have passed the original fdl command
    // name so we can invoke `fdl <cmd>` over ssh. Loud error if absent.
    let has_remote = full.workers.iter().any(|h| h.host != me);
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
    let user_args: Vec<String> = env::args().skip(1).collect();
    let exe = env::current_exe().map_err(|e| {
        TensorError::new(&format!(
            "cluster launcher: current_exe() failed: {e}"
        ))
    })?;
    // Per-host pre-flight build envelope from fdl-cli. When a remote
    // host has an entry, the remote dispatch substitutes the direct
    // binary exec (no cargo on remote). Missing entry ⇒ legacy
    // `fdl <cmd>` fallback (requires cargo on the remote).
    let prebuild_envelope = load_prebuild_envelope()?;

    // Collect (host, abs_bin) for every remote host that has a prebuild
    // envelope entry. Used for both pre-spawn cleanup (clear orphans
    // from a previous botched session before ranks come up) and post-
    // exit cleanup (catch any rank whose remote-side trap wrapper
    // didn't fire). Legacy `fdl <cmd>` re-entry path has no
    // well-defined process signature to pkill on, so it's excluded.
    let remote_cleanup_targets: Vec<(FullWorker, String)> = full
        .workers
        .iter()
        .filter(|h| h.host != me)
        .filter_map(|h| {
            prebuild_envelope.get(&h.host).map(|pb| {
                let abs_bin = format!(
                    "{}/{}",
                    h.path.trim_end_matches('/'),
                    pb.bin,
                );
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

    // Spawn one child per rank across every host.
    let mut children: Vec<(String, usize, std::process::Child, Vec<thread::JoinHandle<()>>)> =
        Vec::with_capacity(full.world_size());
    for host in &full.workers {
        // Spawn one transport relay per host (before its ranks), when the
        // via-coord routing is active. Ranks dial the relay's loopback
        // (+4/+5) instead of the controller directly. The relay child is
        // supervised alongside the ranks — it exits cleanly once its local
        // ranks disconnect, and SIGTERM reaches it if a peer fails.
        if spawn_relays {
            let spec = RelaySpec {
                host: host.host.clone(),
                controller_host: full.controller.host.clone(),
                controller_port: full.controller.port,
                ranks: host.ranks.iter().map(|r| *r as u32).collect(),
                salt_hex: crate::distributed::wire::salt_to_hex(&full.salt),
                world_size: full.world_size(),
                data_channel: relay_data_channel,
            };
            let spec_hex = crate::distributed::cluster::hex_encode(
                serde_json::to_string(&spec)
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster launcher: serialize relay spec failed: {e}"
                        ))
                    })?
                    .as_bytes(),
            );
            let mut cmd = if host.host == me {
                build_local_relay_command(&exe, &user_args, &spec_hex)
            } else {
                let remote_cmd = build_remote_relay_bash_command(
                    &host.path,
                    &spec_hex,
                    fdl_cmd
                        .as_deref()
                        .expect("ENV_FDL_CMD presence enforced above when has_remote"),
                    &user_args,
                    &full.env,
                    &host.env,
                    prebuild_envelope.get(&host.host),
                );
                build_ssh_spawn_command(host, &remote_cmd)
            };
            cmd.stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if host.host == me {
                for (k, v) in &full.env {
                    cmd.env(k, v);
                }
                for (k, v) in &host.env {
                    cmd.env(k, v);
                }
            }
            let mut child = cmd.spawn().map_err(|e| {
                let kind = if host.host == me { "local bash/exec" } else { "ssh" };
                TensorError::new(&format!(
                    "cluster launcher: spawn {kind} relay for {:?} failed: {e}",
                    host.host
                ))
            })?;
            let prefix = format!("[{}:relay] ", host.host);
            let mut forwarders = Vec::with_capacity(2);
            if let Some(out) = child.stdout.take() {
                let p = prefix.clone();
                forwarders.push(thread::spawn(move || forward_lines(out, p, false)));
            }
            if let Some(err) = child.stderr.take() {
                let p = prefix.clone();
                forwarders.push(thread::spawn(move || forward_lines(err, p, true)));
            }
            // `usize::MAX` local-rank sentinel marks the relay child in
            // supervision diagnostics (it has no rank slot).
            children.push((host.host.clone(), usize::MAX, child, forwarders));
        }
        for local_rank in 0..host.ranks.len() {
            let envelope = build_slim_envelope_for(&full, host);
            let envelope_hex = crate::distributed::cluster::hex_encode(
                serde_json::to_string(&envelope)
                    .map_err(|e| {
                        TensorError::new(&format!(
                            "cluster launcher: serialize slim envelope failed: {e}"
                        ))
                    })?
                    .as_bytes(),
            );

            // Scope each rank's child to its assigned physical GPU
            // via `CUDA_VISIBLE_DEVICES=<phys>`. Standard torchrun-
            // style recipe: the child sees only one GPU, addresses it
            // as CUDA(0). Required for multi-process CUDA on older CCs
            // (Pascal/sm_61 surfaced this: dual-process where both
            // ranks see both GPUs hits `cudaErrorNoKernelImageForDevice`
            // sticky on the first allocation, even though kernels are
            // present for sm_61 — the lazy module load picks the wrong
            // context). `cluster::my_rank` honors the scoping by
            // returning CUDA(0) when CUDA_VISIBLE_DEVICES is single-
            // valued.
            let local_phys = host
                .local_devices
                .as_ref()
                .and_then(|d| d.get(local_rank).copied());
            let mut cmd = if host.host == me {
                build_local_spawn_command(
                    &exe,
                    &user_args,
                    &envelope_hex,
                    local_rank,
                    local_phys,
                )
            } else {
                let remote_cmd = build_remote_bash_command(
                    &host.path,
                    &envelope_hex,
                    &host.host,
                    local_rank,
                    overlay_env.as_deref(),
                    fdl_cmd
                        .as_deref()
                        .expect("ENV_FDL_CMD presence enforced above when has_remote"),
                    &user_args,
                    &full.env,
                    &host.env,
                    local_phys,
                    prebuild_envelope.get(&host.host),
                );
                build_ssh_spawn_command(host, &remote_cmd)
            };
            cmd.stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());

            // Apply user-declared env from `full.env` (cluster-scope)
            // first, then `host.env` (per-host override). Built-in env
            // vars set later by build_local_spawn_command (e.g.
            // FLODL_LOCAL_RANK, CUDA_VISIBLE_DEVICES, FLODL_CLUSTER_JSON)
            // are not overridable here — the launcher owns those. SSH
            // path: env propagation is bash-level inside
            // build_remote_bash_command and not affected here.
            if host.host == me {
                for (k, v) in &full.env {
                    cmd.env(k, v);
                }
                for (k, v) in &host.env {
                    cmd.env(k, v);
                }
            }

            let mut child = cmd.spawn().map_err(|e| {
                let kind = if host.host == me { "local bash/exec" } else { "ssh" };
                TensorError::new(&format!(
                    "cluster launcher: spawn {kind} for rank {local_rank} of {:?} failed: {e}",
                    host.host
                ))
            })?;

            let global_rank = host.ranks[local_rank];
            // Match the `[host:dev:rN]` identity the child would otherwise
            // emit in-process (now suppressed to avoid a double prefix; see
            // `GpuWorker::new`). `dev` = the physical CUDA device this rank is
            // pinned to, falling back to the local rank slot when
            // `local_devices` is unset.
            let dev = local_phys.map(usize::from).unwrap_or(local_rank);
            let prefix = format!("[{}:{dev}:r{global_rank}] ", host.host);
            let mut forwarders = Vec::with_capacity(2);
            if let Some(out) = child.stdout.take() {
                let prefix_clone = prefix.clone();
                forwarders.push(thread::spawn(move || {
                    forward_lines(out, prefix_clone, false);
                }));
            }
            if let Some(err) = child.stderr.take() {
                let prefix_clone = prefix.clone();
                forwarders.push(thread::spawn(move || {
                    forward_lines(err, prefix_clone, true);
                }));
            }
            children.push((host.host.clone(), local_rank, child, forwarders));
        }
    }
    let _ = my_host_idx; // currently unused but kept for parity with future logic

    // Concurrent supervision: watch every child on its own thread and
    // collect exit events on an mpsc channel. The first non-zero exit
    // triggers SIGTERM on every other still-running child. Without this,
    // a peer that dies pre-rendezvous (e.g. SSH-spawned rank fails
    // before NCCL init completes) leaves the surviving ranks blocked in
    // NCCL's connect-retry loop forever, and the launcher's old
    // sequential `wait()` never even reached the dead peer's status to
    // react.
    let any_failure = supervise_children(children);

    // Post-exit cleanup: belt-and-braces ssh-pkill on every remote host.
    // The remote-side trap wrapper handles SIGHUP-on-disconnect, but
    // that path waits for sshd's keepalive timeout (~30s) and only
    // triggers if SIGHUP is actually delivered (varies by sshd config).
    // This explicit pass fires immediately, so the user sees no leftover
    // process on the remote when the launcher returns.
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

    // Success path returns to the launcher_driver thread, which posts
    // Ok(()) through DdpHandle::join so the caller's main thread reaches
    // its end-of-run summary (e.g. ddp-bench's `done: loss=...` line)
    // before the process terminates. Post-rendezvous + post-training the
    // ClusterCoordinator shutdown closes its metrics-sink end, so
    // next_metrics() drains cleanly without a hard process::exit.
    //
    // Failure path keeps process::exit(1): pre-rendezvous failures leave
    // the coord ticking indefinitely (shutdown_workers never fires),
    // which would otherwise keep the launcher process alive after every
    // rank child has exited, and with it the docker container.
    use std::io::Write as _;
    let _ = std::io::stderr().flush();
    let _ = std::io::stdout().flush();
    if let Some(err) = any_failure {
        eprintln!("cluster launcher: {err}");
        let _ = std::io::stderr().flush();
        std::process::exit(1);
    }
    Ok(())
}

