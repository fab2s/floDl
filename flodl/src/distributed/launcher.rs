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

use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use serde::Deserialize;

use crate::tensor::{Result, TensorError};

/// Per-host pre-flight build artifact, as published by fdl-cli's
/// prebuild phase in [`ENV_PREBUILD_PER_HOST`]. The launcher reads
/// the env var, parses it as `BTreeMap<String, PerHostPrebuild>`
/// keyed by host name, and substitutes the direct-binary form on the
/// remote dispatch path for any matching host. Mirrors
/// `flodl_cli::prebuild::PerHostEnvelope`.
#[derive(Clone, Debug, Deserialize)]
struct PerHostPrebuild {
    /// Path to the compiled binary, relative to `host.path`.
    bin: String,
    /// Absolute `LD_LIBRARY_PATH` for the libtorch the binary was
    /// linked against. The launcher emits this verbatim before exec
    /// so the binary finds its shared libs at runtime.
    ld_library_path: String,
    /// Subdirectory under `host.path` to `cd` into before exec — the
    /// controller's view of the command's filesystem cwd, relative to
    /// the project root. Mirrors the cwd the controller-side build /
    /// invocation used (`docker compose run … bash -c "cd /workspace/<sub> && …"`).
    /// Empty string means "stay at `host.path`".
    #[serde(default)]
    cwd_subpath: String,
}

/// Load and parse [`ENV_PREBUILD_PER_HOST`] from the current process
/// env. Returns an empty map when the env var is absent (legacy
/// fan-out via `fdl <cmd>` re-entry), or a `TensorError` on JSON
/// parse failure (loud — bad JSON masks a real misconfiguration in
/// the controller's prebuild step).
fn load_prebuild_envelope() -> Result<BTreeMap<String, PerHostPrebuild>> {
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

/// Environment variable carrying the *full* cluster topology to the
/// launcher process. Set by fdl-cli; consumed only by [`dispatch`]. Not
/// propagated to rank children (each child gets a slim per-host envelope
/// instead via `FLODL_CLUSTER_JSON`).
pub const ENV_FULL_CLUSTER_JSON: &str = "FLODL_FULL_CLUSTER_JSON";

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
const SSH_OPTS: &[&str] = &[
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
    let full_set = env::var_os(ENV_FULL_CLUSTER_JSON).is_some();
    let slim_set = env::var_os(crate::distributed::cluster::ENV_CLUSTER_JSON).is_some();
    let slot_set = env::var_os(crate::distributed::cluster::ENV_LOCAL_RANK).is_some();

    match (full_set, slim_set, slot_set) {
        (false, false, false) => Ok(Role::SingleDevice),
        (false, true, true) => Ok(Role::Rank),
        (true, false, false) => Ok(Role::Launcher),
        // Any other combination is a misconfiguration. Loud error with
        // every bit named so the operator can see what's off.
        _ => Err(TensorError::new(&format!(
            "cluster launcher: inconsistent env (FLODL_FULL_CLUSTER_JSON={}, \
             FLODL_CLUSTER_JSON={}, FLODL_LOCAL_RANK={}). \
             Expected: all-unset (single-device), slim+slot only (rank), \
             or full only (launcher).",
            on_off(full_set),
            on_off(slim_set),
            on_off(slot_set),
        ))),
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "set" } else { "unset" }
}

/// Launcher-mode orchestration. Spawn the [`ClusterController`], optionally
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
/// streams, `[host:rN]` line-prefix on stdout/stderr).
///
/// [`ClusterController`]: crate::distributed::controller::ClusterController
/// [`ClusterCoordinator`]: crate::distributed::cluster_coordinator::ClusterCoordinator
pub fn run_launcher_with_config(
    full: FullCluster,
    coord_config: Option<crate::distributed::cluster_coordinator::ClusterCoordinatorConfig>,
) -> Result<()> {
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
    let cpu_averager =
        crate::distributed::controller::ClusterController::start_with_dead_ranks(
            cpu_avg_addr,
            full.world_size(),
            full.salt,
            Arc::clone(&dead_ranks_shared),
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
                        if let Err(e) = coord.dispatch_epoch(start_epoch) {
                            eprintln!(
                                "cluster launcher: dispatch_epoch({start_epoch}) failed: {e}"
                            );
                            return;
                        }
                        // Drive ticks until shutdown_workers fires (all
                        // ranks exited) or the process is killed.
                        loop {
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
            let prefix = format!("[{}:r{global_rank}] ", host.host);
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

/// Drain `children` concurrently and return the first failure (if any).
///
/// One watcher thread per child blocks on `wait()` and posts the exit
/// status on an mpsc channel. The main loop receives events in
/// completion order; on the first non-zero exit it sends SIGTERM to
/// every still-running peer so NCCL-blocked ranks abort their retry
/// loop and exit instead of hanging. Forwarder threads (one for each
/// child's stdout / stderr) are joined after the corresponding child's
/// pipes close. Returns `None` when every child exits cleanly,
/// `Some(err)` attributing to the first failure otherwise.
///
/// `terminate_pid` shells out to `kill -TERM <pid>` rather than pulling
/// `libc` in, which keeps `flodl`'s direct deps unchanged. The child
/// `Child` value is owned by its watcher thread for the duration of
/// `wait()`, so we capture the PID up front and signal by PID.
fn supervise_children(
    children: Vec<(
        String,
        usize,
        std::process::Child,
        Vec<thread::JoinHandle<()>>,
    )>,
) -> Option<TensorError> {
    if children.is_empty() {
        return None;
    }
    // Snapshot identity + PID for every child up front. Used both to
    // attribute incoming events and to signal still-running peers when
    // any one of them fails.
    let pids: Vec<(String, usize, u32)> = children
        .iter()
        .map(|(host, lr, c, _)| (host.clone(), *lr, c.id()))
        .collect();

    let (tx, rx) = mpsc::channel::<(String, usize, std::io::Result<ExitStatus>)>();
    let mut watchers: Vec<thread::JoinHandle<()>> = Vec::with_capacity(children.len());
    let mut all_forwarders: Vec<thread::JoinHandle<()>> = Vec::new();
    for (host, lr, mut child, fwd) in children {
        all_forwarders.extend(fwd);
        let txc = tx.clone();
        watchers.push(thread::spawn(move || {
            let st = child.wait();
            // Channel send only fails if the receiver was dropped, which
            // only happens after the main loop has finished collecting
            // every expected event. Treat as best-effort.
            let _ = txc.send((host, lr, st));
        }));
    }
    // Drop the producer handle held by main so `rx.recv()` terminates
    // when every watcher has finished.
    drop(tx);

    let mut any_failure: Option<TensorError> = None;
    let mut finished: std::collections::HashSet<(String, usize)> =
        std::collections::HashSet::new();
    let mut terminated_peers = false;
    while let Ok((host, lr, st)) = rx.recv() {
        finished.insert((host.clone(), lr));
        let failure_msg: Option<String> = match st {
            Ok(s) if s.success() => None,
            Ok(s) => Some(format!(
                "cluster launcher: rank {lr} of {host} exited with status {}",
                s.code().unwrap_or(-1)
            )),
            Err(e) => Some(format!(
                "cluster launcher: wait on rank {lr} of {host} failed: {e}"
            )),
        };
        if let Some(msg) = failure_msg {
            if any_failure.is_none() {
                any_failure = Some(TensorError::new(&msg));
                if !terminated_peers {
                    terminated_peers = true;
                    for (h, l, pid) in &pids {
                        if !finished.contains(&(h.clone(), *l)) {
                            eprintln!(
                                "cluster launcher: terminating rank {l} of {h:?} (pid {pid}) \
                                 after peer failure"
                            );
                            terminate_pid(*pid);
                        }
                    }
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
    any_failure
}

/// Send SIGTERM to `pid` via the `kill` binary (PATH-resolved). The
/// child `Child` value is owned by its watcher thread for the duration
/// of `wait()`, so calling `Child::kill()` is not available on main;
/// shelling out by PID is the simplest portable path and avoids
/// pulling `libc` into `flodl`'s direct deps. Best-effort: silently
/// continue if the kill itself fails (caller is mid-error already, the
/// peer might have just exited, etc.).
fn terminate_pid(pid: u32) {
    let _ = Command::new("kill")
        .arg("-TERM")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// Build the `Command` that fork+execs a local rank child. Sets all the
/// env vars the rank-side `LocalCluster::from_env` + `dispatch` expect,
/// and strips `FLODL_FULL_CLUSTER_JSON` so the child detects `Role::Rank`.
fn build_local_spawn_command(
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
        // Defense-in-depth: enable NCCL's own watchdog so stuck
        // collectives get aborted independently of flodl's
        // cluster-mode NCCL watchdog (the latter only fires on
        // peer-death events the coord broadcasts; this catches
        // wedge cases beyond that surface).
        .env("NCCL_ASYNC_ERROR_HANDLING", "1")
        .env_remove(ENV_FULL_CLUSTER_JSON);
    if let Some(phys) = local_phys_device {
        cmd.env("CUDA_VISIBLE_DEVICES", phys.to_string());
    }
    cmd
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
fn build_ssh_spawn_command(host: &FullWorker, remote_cmd: &str) -> Command {
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
fn cleanup_remote_host(host: &FullWorker, abs_bin: &str) {
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
fn cleanup_remote_hosts_parallel(remotes: Vec<(FullWorker, String)>) {
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
fn build_remote_bash_command(
    path: &str,
    cluster_json_hex: &str,
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
        256 + cluster_json_hex.len() + user_args.iter().map(|a| a.len() + 4).sum::<usize>(),
    );
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
    s.push('=');
    s.push_str(&shell_quote(cluster_json_hex));
    s.push(' ');
    s.push_str(ENV_HOST_OVERRIDE);
    s.push('=');
    s.push_str(&shell_quote(host_name));
    s.push(' ');
    s.push_str(ENV_LOCAL_RANK);
    s.push('=');
    s.push_str(&local_rank.to_string());
    s.push(' ');
    // Defense-in-depth for NCCL stuck-collective detection; see the
    // matching env on the local spawn path.
    s.push_str("NCCL_ASYNC_ERROR_HANDLING=1");
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
    // (host overrides cluster for matching keys). Built-in env vars
    // above are not overridable here — the launcher owns those.
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
    s.push_str(
        "trap 'kill -TERM \"$__flodl_pid\" 2>/dev/null' HUP TERM INT\n",
    );
    s.push_str("wait \"$__flodl_pid\"\n");
    s.push_str("exit $?\n");
    s
}

/// Single-quote a string for shell consumption. Internal single quotes
/// are escaped via the `'\''` idiom (close, backslash-escape, reopen).
/// Same implementation as fdl-cli's; kept as a private helper here
/// rather than introducing a shared utilities crate.
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

/// Build the slim per-host envelope JSON the rank process consumes via
/// [`LocalCluster::from_env`].
///
/// [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env
fn build_slim_envelope_for(full: &FullCluster, worker: &FullWorker) -> serde_json::Value {
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
fn forward_lines<R: std::io::Read>(stream: R, prefix: String, to_stderr: bool) {
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
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// SshConfig: per-worker SSH endpoint knobs.
// ---------------------------------------------------------------------------

/// SSH endpoint configuration for a remote worker host.
///
/// Carries the per-host SSH knobs used by the launcher when fanning
/// out to remote ranks. All fields are optional; when absent, the
/// corresponding flag is omitted from the spawned `ssh` command and
/// system ssh defaults (or `~/.ssh/config` rules) apply.
///
/// In YAML, this lives under each worker's `ssh:` sub-block, e.g.:
///
/// ```yaml
/// workers:
///   - host: flodl-pascal
///     ssh:
///       target: flodl-pascal.lan
///       port: 2222
///       user: fab2s
///       identity_file: ~/.ssh/id_ed25519
///       options:
///         - ProxyJump=bastion
/// ```
///
/// The `host:` (logical name) and `ssh.target:` (network endpoint)
/// split lets the worker's logical identity differ from its SSH
/// target. When `target` is unset, the launcher falls back to the
/// worker's `host` name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SshConfig {
    /// SSH target hostname / IP / alias. Defaults to the worker's
    /// `host` when `None`.
    pub target: Option<String>,
    /// SSH port. Maps to `ssh -p <port>`.
    pub port: Option<u16>,
    /// SSH login user. Maps to `ssh -l <user>`. Falls back to the
    /// current user (or `FLODL_HOST_USER` from env) when `None`.
    pub user: Option<String>,
    /// Identity file (private key) path. Maps to `ssh -i <path>`.
    pub identity_file: Option<String>,
    /// Pass-through `-o Key=Value` SSH options (e.g.
    /// `"ProxyJump=bastion"`, `"StrictHostKeyChecking=no"`). Each
    /// entry becomes one `-o ...` arg on the spawned `ssh` command,
    /// in the order declared.
    pub options: Vec<String>,
}

// ---------------------------------------------------------------------------
// FullCluster: launcher-side parser for the multi-host topology.
// ---------------------------------------------------------------------------

/// Full cluster topology as seen by the launcher process.
///
/// Mirrors flodl-cli's `ClusterConfig` shape; lives on the flodl side so
/// the framework owns cluster orchestration end-to-end. The slim
/// per-rank envelopes parsed by [`LocalCluster`] are derived from this
/// view at fan-out time.
///
/// Like [`LocalCluster`], the rank-side `local_devices: "all"` shorthand
/// is resolved at parse time when applicable (host-side resolution
/// happens later, after envelope ship — see [`crate::distributed::cluster`]
/// for the slim path).
///
/// [`LocalCluster`]: crate::distributed::cluster::LocalCluster
#[derive(Debug, Clone)]
pub struct FullCluster {
    /// Controller's rendezvous bind point + pre-flight build context.
    pub controller: FullController,
    /// All rank-carrying entries.
    pub workers: Vec<FullWorker>,
    /// 128-bit session salt the launcher generates fresh per training
    /// session and propagates to every rank's slim envelope. Used as the
    /// HMAC key for the cross-process control + data channels. All
    /// zeros until [`FullCluster::with_session_salt`] (or
    /// [`run_launcher_with_config`]) populates it.
    pub salt: crate::distributed::wire::SessionSalt,
    /// Cluster-scope env vars exported into every rank child's
    /// environment. Cluster-yml `env:` block (mapping `NAME: VALUE`).
    /// Used for cluster-specific tuning that the launcher itself
    /// shouldn't hardcode — e.g. setting `NCCL_P2P_DISABLE=1` +
    /// `NCCL_SHM_DISABLE=1` for the Pascal-under-VFIO rig where NCCL's
    /// direct-IPC transports fail but socket transport works.
    ///
    /// Empty by default. Per-worker envs (see [`FullWorker::env`])
    /// override per-cluster ones for the matching worker.
    pub env: std::collections::BTreeMap<String, String>,
}

/// Controller-side fields, launcher view.
#[derive(Debug, Clone)]
pub struct FullController {
    pub host: String,
    pub port: u16,
    pub path: String,
    pub docker: Option<String>,
    pub arch: Option<String>,
    pub data_path: Option<String>,
}

impl FullCluster {
    /// Replace the session salt and return self for chaining. Called
    /// by [`run_launcher_with_config`] once it has generated a fresh salt.
    pub fn with_session_salt(mut self, salt: crate::distributed::wire::SessionSalt) -> Self {
        self.salt = salt;
        self
    }
}

/// One worker's entry in the full topology, launcher-side.
///
/// Differs from [`WorkerBlock`] by carrying `ssh:` (launcher-only field
/// stripped from slim envelopes) and the unresolved `local_devices:
/// "all"` form (which is only resolved on the host that will use it).
///
/// [`WorkerBlock`]: crate::distributed::cluster::WorkerBlock
#[derive(Debug, Clone)]
pub struct FullWorker {
    pub host: String,
    pub ranks: Vec<usize>,
    /// Either an explicit list of CUDA indices or `None` for the `"all"`
    /// shorthand (resolved at startup on the host that owns this entry).
    pub local_devices: Option<Vec<u8>>,
    pub nccl_socket_ifname: String,
    pub path: String,
    /// libtorch variant subpath under `<path>/libtorch/`. The runtime
    /// libtorch lives at `<path>/libtorch/<arch>/` by convention; the
    /// launcher uses this to build the remote-side LD_LIBRARY_PATH
    /// when no pre-flight envelope overrides it.
    pub arch: Option<String>,
    /// SSH endpoint for remote dispatch. `None` means the host runs
    /// on the same machine as the launcher (fork/exec path, no ssh).
    /// When `Some`, all fields inside are optional and fall back to
    /// system ssh defaults (or `~/.ssh/config` rules) when unset.
    pub ssh: Option<SshConfig>,
    /// Per-host env vars exported into this host's rank children.
    /// Override the cluster-scope [`FullCluster::env`] for matching
    /// keys. Use for host-specific tuning (e.g. an interface override
    /// only one host needs).
    pub env: std::collections::BTreeMap<String, String>,
}

impl FullWorker {
    /// SSH target for this worker, defaulting to `host` when
    /// `ssh.target` is unset or `ssh` itself is `None`. Used by the
    /// launcher's `build_ssh_spawn_command` and by `fdl probe`.
    pub fn ssh_target(&self) -> &str {
        self.ssh
            .as_ref()
            .and_then(|s| s.target.as_deref())
            .unwrap_or(&self.host)
    }
}

impl FullCluster {
    /// Read + parse the full topology from [`ENV_FULL_CLUSTER_JSON`].
    ///
    /// Loud errors on missing var, hex/JSON decode failure, or schema
    /// violations. The launcher-only path; not relevant on rank children.
    pub fn from_env() -> Result<Self> {
        let raw = env::var(ENV_FULL_CLUSTER_JSON).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: reading {ENV_FULL_CLUSTER_JSON} failed: {e}"
            ))
        })?;
        let bytes = crate::distributed::cluster::hex_decode(raw.trim()).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: {ENV_FULL_CLUSTER_JSON} hex-decode failed: {e}"
            ))
        })?;
        let val: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: {ENV_FULL_CLUSTER_JSON} JSON parse failed: {e}"
            ))
        })?;
        Self::from_value(&val)
    }

    /// Parse from a pre-decoded JSON value. Test entry point + future
    /// programmatic callers.
    pub fn from_value(val: &serde_json::Value) -> Result<Self> {
        let obj = val.as_object().ok_or_else(|| {
            TensorError::new("cluster launcher: top-level JSON must be an object")
        })?;

        let controller_val = obj
            .get("controller")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                TensorError::new("cluster launcher: controller (object) required")
            })?;
        let controller_host = controller_val
            .get("host")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TensorError::new("cluster launcher: controller.host (string) required")
            })?
            .to_string();
        if controller_host.trim().is_empty() {
            return Err(TensorError::new(
                "cluster launcher: controller.host must be non-empty",
            ));
        }
        let controller_port_u64 = controller_val
            .get("port")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                TensorError::new("cluster launcher: controller.port (u16) required")
            })?;
        let controller_port = u16::try_from(controller_port_u64).map_err(|_| {
            TensorError::new(&format!(
                "cluster launcher: controller.port must fit in u16 (got {controller_port_u64})"
            ))
        })?;
        let controller_path = controller_val
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TensorError::new("cluster launcher: controller.path (string) required")
            })?
            .to_string();
        let controller_docker = controller_val
            .get("docker")
            .and_then(|v| v.as_str())
            .map(String::from);
        let controller_arch = controller_val
            .get("arch")
            .and_then(|v| v.as_str())
            .map(String::from);
        let controller_data_path = controller_val
            .get("data_path")
            .and_then(|v| v.as_str())
            .map(String::from);

        let workers_val = obj
            .get("workers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorError::new("cluster launcher: workers (array) required"))?;
        if workers_val.is_empty() {
            return Err(TensorError::new(
                "cluster launcher: workers must be non-empty",
            ));
        }

        let workers: Vec<FullWorker> = workers_val
            .iter()
            .enumerate()
            .map(|(i, w)| parse_full_worker(w, i))
            .collect::<Result<_>>()?;

        // Cross-worker rank check: union must be exactly 0..world_size.
        let mut all: Vec<usize> = workers.iter().flat_map(|w| w.ranks.iter().copied()).collect();
        let ws = all.len();
        all.sort_unstable();
        let expected: Vec<usize> = (0..ws).collect();
        if all != expected {
            return Err(TensorError::new(&format!(
                "cluster launcher: ranks across workers must be exactly 0..{ws} \
                 with no duplicates or gaps, got sorted-unique sequence {all:?}"
            )));
        }

        // Optional cluster-scope `env:` block: mapping of NAME → VALUE
        // exported into every rank child. Missing → empty map.
        let env = parse_env_block(obj.get("env"), "cluster.env")?;

        Ok(FullCluster {
            controller: FullController {
                host: controller_host,
                port: controller_port,
                path: controller_path,
                docker: controller_docker,
                arch: controller_arch,
                data_path: controller_data_path,
            },
            workers,
            // ENV_FULL_CLUSTER_JSON is the config snapshot fdl-cli ships;
            // the session salt is generated freshly by `run_launcher` per
            // training session (override via [`Self::with_session_salt`]).
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
            env,
        })
    }

    /// Total ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.workers.iter().map(|w| w.ranks.len()).sum()
    }

    /// Whether the cluster spans more than one physical worker.
    pub fn spans_multiple_workers(&self) -> bool {
        self.workers.len() > 1
    }

    /// Serialize to the JSON shape [`Self::from_value`] parses. Symmetric
    /// round-trip: `FullCluster::from_value(&cluster.to_json()) == cluster`.
    /// Used by [`crate::distributed::Trainer::run`] to convert a
    /// programmatic [`super::ClusterBuilder`] result into the
    /// `FLODL_FULL_CLUSTER_JSON` env-var contract the launcher path
    /// reads. `salt` is intentionally NOT serialized — the launcher
    /// generates a fresh session salt per run.
    pub fn to_json(&self) -> serde_json::Value {
        let workers: Vec<serde_json::Value> = self
            .workers
            .iter()
            .map(|h| {
                let mut o = serde_json::Map::new();
                o.insert("host".into(), serde_json::Value::String(h.host.clone()));
                o.insert(
                    "ranks".into(),
                    serde_json::Value::Array(
                        h.ranks.iter().map(|r| serde_json::Value::from(*r)).collect(),
                    ),
                );
                let ld = match &h.local_devices {
                    None => serde_json::Value::String("all".into()),
                    Some(v) => serde_json::Value::Array(
                        v.iter().map(|d| serde_json::Value::from(*d)).collect(),
                    ),
                };
                o.insert("local_devices".into(), ld);
                o.insert(
                    "nccl_socket_ifname".into(),
                    serde_json::Value::String(h.nccl_socket_ifname.clone()),
                );
                o.insert("path".into(), serde_json::Value::String(h.path.clone()));
                if let Some(a) = &h.arch {
                    o.insert("arch".into(), serde_json::Value::String(a.clone()));
                }
                if let Some(s) = &h.ssh {
                    let mut ssh_obj = serde_json::Map::new();
                    if let Some(t) = &s.target {
                        ssh_obj.insert("target".into(), serde_json::Value::String(t.clone()));
                    }
                    if let Some(p) = s.port {
                        ssh_obj.insert("port".into(), serde_json::Value::from(p));
                    }
                    if let Some(u) = &s.user {
                        ssh_obj.insert("user".into(), serde_json::Value::String(u.clone()));
                    }
                    if let Some(i) = &s.identity_file {
                        ssh_obj.insert(
                            "identity_file".into(),
                            serde_json::Value::String(i.clone()),
                        );
                    }
                    if !s.options.is_empty() {
                        ssh_obj.insert(
                            "options".into(),
                            serde_json::Value::Array(
                                s.options
                                    .iter()
                                    .map(|opt| serde_json::Value::String(opt.clone()))
                                    .collect(),
                            ),
                        );
                    }
                    o.insert("ssh".into(), serde_json::Value::Object(ssh_obj));
                }
                if !h.env.is_empty() {
                    let mut env_obj = serde_json::Map::new();
                    for (k, v) in &h.env {
                        env_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
                    }
                    o.insert("env".into(), serde_json::Value::Object(env_obj));
                }
                serde_json::Value::Object(o)
            })
            .collect();
        let mut top = serde_json::Map::new();
        let mut controller_obj = serde_json::Map::new();
        controller_obj.insert(
            "host".into(),
            serde_json::Value::String(self.controller.host.clone()),
        );
        controller_obj.insert(
            "port".into(),
            serde_json::Value::from(self.controller.port),
        );
        controller_obj.insert(
            "path".into(),
            serde_json::Value::String(self.controller.path.clone()),
        );
        if let Some(s) = &self.controller.docker {
            controller_obj.insert("docker".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(s) = &self.controller.arch {
            controller_obj.insert("arch".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(s) = &self.controller.data_path {
            controller_obj.insert("data_path".into(), serde_json::Value::String(s.clone()));
        }
        top.insert("controller".into(), serde_json::Value::Object(controller_obj));
        top.insert("workers".into(), serde_json::Value::Array(workers));
        if !self.env.is_empty() {
            let mut env_obj = serde_json::Map::new();
            for (k, v) in &self.env {
                env_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            top.insert("env".into(), serde_json::Value::Object(env_obj));
        }
        serde_json::Value::Object(top)
    }
}

fn parse_full_worker(v: &serde_json::Value, i: usize) -> Result<FullWorker> {
    let obj = v.as_object().ok_or_else(|| {
        TensorError::new(&format!("cluster launcher: workers[{i}] must be an object"))
    })?;

    let host = obj
        .get("host")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}].host (string) required"
            ))
        })?
        .to_string();
    if host.trim().is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: workers[{i}].host must be non-empty"
        )));
    }
    let name = host;

    let ranks_arr = obj
        .get("ranks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): ranks (array) required"
            ))
        })?;
    // Empty ranks: orchestrator-only host entry. Declared in cluster.yml
    // solely so fdl-cli's pre-flight build can read its `docker:` /
    // `arch:` for controller-side build context; the launcher itself
    // skips it (no rank spawn for this host). Distinct from "host
    // absent from cluster.workers" — both result in orchestrator-only
    // launcher behavior, but the explicit entry surfaces config to
    // fdl-cli.
    let ranks: Vec<usize> = ranks_arr
        .iter()
        .enumerate()
        .map(|(j, e)| {
            let n = e.as_u64().ok_or_else(|| {
                TensorError::new(&format!(
                    "cluster launcher: workers[{i}].ranks[{j}]: non-integer entry"
                ))
            })?;
            usize::try_from(n).map_err(|_| {
                TensorError::new(&format!(
                    "cluster launcher: workers[{i}].ranks[{j}]: value {n} out of range"
                ))
            })
        })
        .collect::<Result<_>>()?;

    let local_devices = match obj.get("local_devices") {
        None => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): local_devices required"
            )));
        }
        Some(serde_json::Value::String(s)) if s == "all" => None,
        Some(serde_json::Value::String(s)) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): local_devices: \
                 expected \"all\" or array, got string {s:?}"
            )));
        }
        Some(serde_json::Value::Array(arr)) => {
            let v: Vec<u8> = arr
                .iter()
                .enumerate()
                .map(|(j, e)| {
                    let n = e.as_u64().ok_or_else(|| {
                        TensorError::new(&format!(
                            "cluster launcher: workers[{i}].local_devices[{j}]: \
                             non-integer entry"
                        ))
                    })?;
                    u8::try_from(n).map_err(|_| {
                        TensorError::new(&format!(
                            "cluster launcher: workers[{i}].local_devices[{j}]: \
                             value {n} does not fit in u8"
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            if v.len() != ranks.len() {
                return Err(TensorError::new(&format!(
                    "cluster launcher: workers[{i}] ({name:?}): ranks ({}) and \
                     local_devices ({}) length mismatch",
                    ranks.len(),
                    v.len()
                )));
            }
            Some(v)
        }
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): local_devices: \
                 expected \"all\" or array, got {other}"
            )));
        }
    };

    let nccl_socket_ifname = obj
        .get("nccl_socket_ifname")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): nccl_socket_ifname (string) required"
            ))
        })?
        .to_string();

    let path = obj
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): path (string) required"
            ))
        })?
        .to_string();
    if path.trim().is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: workers[{i}] ({name:?}): path must be non-empty"
        )));
    }

    let arch = obj
        .get("arch")
        .and_then(|v| v.as_str())
        .map(String::from);

    let ssh = parse_ssh_block(
        obj.get("ssh"),
        &format!("workers[{i}] ({name:?})"),
    )?;

    let env = parse_env_block(
        obj.get("env"),
        &format!("workers[{i}] ({name:?}).env"),
    )?;

    Ok(FullWorker {
        host: name,
        ranks,
        local_devices,
        nccl_socket_ifname,
        path,
        arch,
        ssh,
        env,
    })
}

/// Parse an `ssh:` sub-block. Expects a JSON object with optional
/// `target`, `port`, `user`, `identity_file`, and `options` fields;
/// missing/null produces `None` (meaning "no SSH overrides, fall back
/// to host name + system ssh defaults"). Loud errors on type
/// mismatches per field so typos surface immediately rather than
/// silently dropping a value.
fn parse_ssh_block(
    v: Option<&serde_json::Value>,
    label: &str,
) -> Result<Option<SshConfig>> {
    let obj = match v {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::Object(m)) => m,
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: {label}.ssh must be a map (target, port, \
                 user, identity_file, options), got {other}"
            )));
        }
    };

    let target = obj
        .get("target")
        .and_then(|v| v.as_str())
        .map(String::from);

    let port = match obj.get("port") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| {
                TensorError::new(&format!(
                    "cluster launcher: {label}.ssh.port must be integer"
                ))
            })?;
            Some(u16::try_from(n).map_err(|_| {
                TensorError::new(&format!(
                    "cluster launcher: {label}.ssh.port {n} does not fit in u16"
                ))
            })?)
        }
    };

    let user = obj
        .get("user")
        .and_then(|v| v.as_str())
        .map(String::from);

    let identity_file = obj
        .get("identity_file")
        .and_then(|v| v.as_str())
        .map(String::from);

    let options: Vec<String> = match obj.get("options") {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .enumerate()
            .map(|(j, e)| {
                e.as_str().map(String::from).ok_or_else(|| {
                    TensorError::new(&format!(
                        "cluster launcher: {label}.ssh.options[{j}]: must be string"
                    ))
                })
            })
            .collect::<Result<_>>()?,
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: {label}.ssh.options must be array of strings, got {other}"
            )));
        }
    };

    Ok(Some(SshConfig { target, port, user, identity_file, options }))
}

/// Parse an `env:` block from either a launcher-level or host-level
/// position. Expects a JSON object whose values are all strings
/// (`{"NAME": "value", ...}`); missing/null produces an empty map.
/// Loud-errors on anything else so a typo can't silently produce an
/// empty env that hides a real config error.
fn parse_env_block(
    v: Option<&serde_json::Value>,
    label: &str,
) -> Result<std::collections::BTreeMap<String, String>> {
    use std::collections::BTreeMap;
    match v {
        None | Some(serde_json::Value::Null) => Ok(BTreeMap::new()),
        Some(serde_json::Value::Object(map)) => {
            let mut out = BTreeMap::new();
            for (k, val) in map {
                let s = val.as_str().ok_or_else(|| {
                    TensorError::new(&format!(
                        "cluster launcher: {label}[{k:?}] must be a string, got {val}"
                    ))
                })?;
                out.insert(k.clone(), s.to_string());
            }
            Ok(out)
        }
        Some(other) => Err(TensorError::new(&format!(
            "cluster launcher: {label} must be an object (NAME → string VALUE), \
             got {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn canonical_full_json() -> serde_json::Value {
        json!({
            "controller": {
                "host": "192.168.122.1",
                "port": 29500,
                "path": "/opt/flodl"
            },
            "workers": [
                {
                    "host": "host-a",
                    "ranks": [0],
                    "local_devices": [0],
                    "nccl_socket_ifname": "virbr0",
                    "path": "/opt/flodl",
                    "arch": "precompiled/cu128"
                },
                {
                    "host": "host-b",
                    "ssh": { "target": "host-b" },
                    "ranks": [1, 2],
                    "local_devices": "all",
                    "nccl_socket_ifname": "enp1s0",
                    "path": "/srv/flodl"
                }
            ]
        })
    }

    #[test]
    fn parses_full_topology() {
        let c = FullCluster::from_value(&canonical_full_json()).unwrap();
        assert_eq!(c.controller.host, "192.168.122.1");
        assert_eq!(c.controller.port, 29500);
        assert_eq!(c.world_size(), 3);
        assert!(c.spans_multiple_workers());

        assert_eq!(c.workers.len(), 2);
        assert_eq!(c.workers[0].host, "host-a");
        assert_eq!(c.workers[0].ranks, vec![0]);
        assert_eq!(c.workers[0].local_devices, Some(vec![0]));
        assert_eq!(c.workers[0].ssh, None);

        assert_eq!(c.workers[1].host, "host-b");
        assert_eq!(c.workers[1].ranks, vec![1, 2]);
        // "all" stays unresolved at launcher-parse time; each host resolves
        // its own at startup.
        assert_eq!(c.workers[1].local_devices, None);
        assert_eq!(c.workers[1].ssh_target(), "host-b");
    }

    #[test]
    fn rejects_empty_workers() {
        let mut v = canonical_full_json();
        v["workers"] = json!([]);
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("workers must be non-empty"), "got: {err}");
    }

    #[test]
    fn rejects_rank_gap_across_hosts() {
        let mut v = canonical_full_json();
        v["workers"][1]["ranks"] = json!([2, 3]); // gap: 0 + (2,3) misses rank 1
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(
            err.to_string().contains("duplicates or gaps"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_duplicate_ranks() {
        let mut v = canonical_full_json();
        v["workers"][1]["ranks"] = json!([0, 1]); // collides with host-a's [0]
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(
            err.to_string().contains("duplicates or gaps"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_local_devices_length_mismatch_for_explicit() {
        let mut v = canonical_full_json();
        v["workers"][1]["local_devices"] = json!([0]); // ranks: [1, 2] needs 2 devices
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn accepts_local_devices_all_at_launcher_parse_time() {
        // "all" stays symbolic; resolution is deferred to startup on the
        // host that ends up parsing the slim envelope.
        let mut v = canonical_full_json();
        v["workers"][0]["local_devices"] = json!("all");
        let c = FullCluster::from_value(&v).unwrap();
        assert_eq!(c.workers[0].local_devices, None);
    }

    #[test]
    fn rejects_unknown_local_devices_string() {
        let mut v = canonical_full_json();
        v["workers"][0]["local_devices"] = json!("every");
        let err = FullCluster::from_value(&v).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("local_devices") && msg.contains("every"),
            "got: {msg}"
        );
    }

    #[test]
    fn rejects_controller_port_overflow() {
        let mut v = canonical_full_json();
        v["controller"]["port"] = json!(100_000);
        let err = FullCluster::from_value(&v).unwrap_err();
        assert!(err.to_string().contains("u16"), "got: {err}");
    }

    #[test]
    fn slim_envelope_strips_ssh_carries_metadata() {
        // Direct test of the build_slim_envelope_for helper: the slim
        // shape must round-trip through LocalCluster::from_env on the
        // rank side, so it has to match that parser's expectations
        // (controller/world_size/num_workers/worker with no ssh field).
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let worker = full.workers.iter().find(|h| h.host == "host-b").unwrap();
        let env = build_slim_envelope_for(&full, worker);

        assert_eq!(env["controller"]["host"], "192.168.122.1");
        assert_eq!(env["controller"]["port"], 29500);
        assert_eq!(env["world_size"], 3);
        assert_eq!(env["num_workers"], 2);
        assert_eq!(env["worker"]["host"], "host-b");
        assert_eq!(env["worker"]["ranks"], serde_json::json!([1, 2]));
        assert_eq!(env["worker"]["local_devices"], serde_json::json!("all"));
        assert_eq!(env["worker"]["nccl_socket_ifname"], "enp1s0");
        // ssh: stripped (launcher-only field; slim envelope is rank-side).
        assert!(env["worker"].get("ssh").is_none(), "ssh must be stripped");
    }

    #[test]
    fn slim_envelope_emits_explicit_local_devices_when_present() {
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let host_a = full.workers.iter().find(|h| h.host == "host-a").unwrap();
        let env = build_slim_envelope_for(&full, host_a);
        assert_eq!(env["worker"]["local_devices"], serde_json::json!([0]));
    }

    #[test]
    fn shell_quote_simple() {
        assert_eq!(shell_quote("foo"), "'foo'");
    }

    #[test]
    fn shell_quote_with_spaces() {
        assert_eq!(shell_quote("foo bar"), "'foo bar'");
    }

    #[test]
    fn shell_quote_escapes_internal_quotes() {
        assert_eq!(shell_quote("don't"), "'don'\\''t'");
    }

    fn empty_env() -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }

    #[test]
    fn build_remote_bash_command_shape() {
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv/flodl",
            "abcd1234",
            "host-b",
            0,
            Some("cluster"),
            "train",
            &["--epochs".to_string(), "10".to_string()],
            &cluster_env,
            &host_env,
            None,
            None,
        );
        assert!(s.starts_with("cd '/srv/flodl' && "));
        assert!(s.contains("FLODL_CLUSTER_JSON='abcd1234'"));
        assert!(s.contains("FLODL_HOST_NAME='host-b'"));
        assert!(s.contains("FLODL_LOCAL_RANK=0"));
        assert!(s.contains("FDL_ENV='cluster'"));
        assert!(s.contains("fdl 'train' '--epochs' '10' &\n"));
        assert!(s.contains("trap 'kill -TERM \"$__flodl_pid\"' HUP TERM INT") ||
                s.contains("trap 'kill -TERM \"$__flodl_pid\" 2>/dev/null' HUP TERM INT"));
        assert!(s.contains("wait \"$__flodl_pid\""));
        assert!(s.ends_with("exit $?\n"));
    }

    #[test]
    fn build_remote_bash_command_omits_fdl_env_when_none() {
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv/flodl",
            "abcd",
            "worker",
            0,
            None,
            "train",
            &[],
            &cluster_env,
            &host_env,
            None,
            None,
        );
        assert!(
            !s.contains("FDL_ENV"),
            "FDL_ENV must be absent when overlay_env is None; got: {s}"
        );
    }

    #[test]
    fn build_remote_bash_command_uses_trap_wrapper() {
        // The trap wrapper is load-bearing: it keeps a bash process
        // alive on the remote after launch so that a connection-drop
        // SIGHUP from sshd reaches a shell that can signal the binary,
        // instead of being lost to a bare `exec`'d binary that ignores
        // SIGHUP. Without this, every cluster smoke leaves an orphan
        // ddp-bench on the remote until manual pkill.
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv", "ff", "w", 0, None, "train", &[],
            &cluster_env, &host_env, None, None,
        );
        assert!(s.contains(" fdl "), "missing `fdl` invocation: {s}");
        assert!(s.contains(" &\n"), "missing background `&`: {s}");
        assert!(
            s.contains("__flodl_pid=$!"),
            "missing `__flodl_pid=$!`: {s}"
        );
        assert!(
            s.contains("trap 'kill -TERM \"$__flodl_pid\""),
            "missing trap line: {s}"
        );
        assert!(s.contains("wait \"$__flodl_pid\""), "missing wait: {s}");
        assert!(s.ends_with("exit $?\n"), "missing exit prop: {s}");
    }

    #[test]
    fn build_remote_bash_command_quotes_dangerous_path() {
        // Single quotes in the path must round-trip through the
        // single-quote-escape idiom.
        let cluster_env = empty_env();
        let host_env = empty_env();
        let s = build_remote_bash_command(
            "/srv/it's", "ff", "w", 0, None, "train", &[],
            &cluster_env, &host_env, None, None,
        );
        assert!(
            s.contains("cd '/srv/it'\\''s'"),
            "path with single quote not properly escaped: {s}"
        );
    }

    #[test]
    fn build_remote_bash_command_uses_prebuild_binary_and_ld_path() {
        // When the prebuild envelope provides an entry for this host,
        // the remote dispatch must (a) emit LD_LIBRARY_PATH, (b) launch
        // the binary directly via `<bin>` (no `fdl` re-entry), and (c)
        // close with the trap wrapper so the binary can be cleaned up
        // via SIGHUP on connection drop.
        let cluster_env = empty_env();
        let host_env = empty_env();
        let pb = PerHostPrebuild {
            bin: "target/cluster/worker/release/ddp-bench".into(),
            ld_library_path: "/opt/libtorch/lib".into(),
            cwd_subpath: "ddp-bench".into(),
        };
        let s = build_remote_bash_command(
            "/srv/flodl",
            "abcd",
            "worker",
            0,
            None,
            "ddp-bench",
            &["--mode".into(), "nccl-sync".into()],
            &cluster_env,
            &host_env,
            None,
            Some(&pb),
        );
        assert!(
            s.contains("LD_LIBRARY_PATH='/opt/libtorch/lib'"),
            "missing prebuild LD_LIBRARY_PATH: {s}",
        );
        assert!(
            s.contains("cd '/srv/flodl/ddp-bench'"),
            "remote cwd must cd into <host.path>/<cwd_subpath>: {s}",
        );
        assert!(
            s.contains(" '/srv/flodl/target/cluster/worker/release/ddp-bench'"),
            "binary path must be absolute (independent of cwd offset): {s}",
        );
        assert!(
            !s.contains("fdl 'ddp-bench'"),
            "prebuild path must NOT re-enter fdl on remote: {s}",
        );
        assert!(
            s.contains("'--mode' 'nccl-sync' &\n"),
            "user args must be appended ahead of the trap wrapper: {s}",
        );
        assert!(s.ends_with("exit $?\n"), "trap wrapper must end the cmd: {s}");
    }

    #[test]
    fn build_remote_bash_command_prebuild_yields_to_host_env_ld_path() {
        // If the user sets LD_LIBRARY_PATH via host.env, the
        // auto-derived prebuild LD_LIBRARY_PATH must yield (the user's
        // value is the source of truth; e.g. they need extra paths
        // for bare-metal libnccl alongside libtorch).
        let cluster_env = empty_env();
        let mut host_env = empty_env();
        host_env.insert(
            "LD_LIBRARY_PATH".into(),
            "/opt/libtorch/lib:/usr/local/lib".into(),
        );
        let pb = PerHostPrebuild {
            bin: "target/cluster/worker/release/ddp-bench".into(),
            ld_library_path: "/opt/libtorch/lib".into(),
            cwd_subpath: String::new(),
        };
        let s = build_remote_bash_command(
            "/srv", "ff", "w", 0, None, "ddp-bench", &[],
            &cluster_env, &host_env, None, Some(&pb),
        );
        // Only the host_env value should be present; the auto-derived
        // prebuild-only LD_LIBRARY_PATH must be suppressed.
        let host_pos = s.find("LD_LIBRARY_PATH='/opt/libtorch/lib:/usr/local/lib'").unwrap();
        // The auto-derived entry would have emitted exactly this
        // substring; assert it's absent.
        assert!(
            !s.contains(" LD_LIBRARY_PATH='/opt/libtorch/lib' "),
            "auto-derived LD_LIBRARY_PATH should yield to host_env: {s}",
        );
        let _ = host_pos;
    }

    #[test]
    fn build_remote_bash_command_exports_cluster_and_host_env() {
        // Cluster-scope and host-scope env vars round-trip into the
        // exported shell command. Host overrides cluster on key
        // collisions.
        let mut cluster_env = empty_env();
        cluster_env.insert("NCCL_P2P_DISABLE".into(), "1".into());
        cluster_env.insert("SHARED_FLAG".into(), "cluster-wins".into());
        let mut host_env = empty_env();
        host_env.insert("HOST_FLAG".into(), "host-val".into());
        host_env.insert("SHARED_FLAG".into(), "host-wins".into());
        let s = build_remote_bash_command(
            "/srv", "ff", "w", 0, None, "train", &[],
            &cluster_env, &host_env, Some(1), None,
        );
        assert!(s.contains("NCCL_P2P_DISABLE='1'"));
        assert!(s.contains("HOST_FLAG='host-val'"));
        // Host SHARED_FLAG export comes after cluster's; the shell
        // takes the last value when env vars are assigned multiple
        // times in a `K=V K=V ...` prefix.
        let cluster_pos = s.find("SHARED_FLAG='cluster-wins'").unwrap();
        let host_pos = s.find("SHARED_FLAG='host-wins'").unwrap();
        assert!(cluster_pos < host_pos, "host env must export after cluster env");
        assert!(s.contains("CUDA_VISIBLE_DEVICES=1"));
    }

    #[test]
    fn slim_envelope_round_trips_through_local_cluster_parser() {
        // Smoke test: the slim envelope built by the launcher must parse
        // cleanly via the rank-side LocalCluster::from_value. Same wire
        // contract, validated end-to-end.
        let full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let host_a = full.workers.iter().find(|h| h.host == "host-a").unwrap();
        let env = build_slim_envelope_for(&full, host_a);
        let parsed = crate::distributed::cluster::LocalCluster::from_value(&env)
            .expect("slim envelope must parse via LocalCluster::from_value");
        assert_eq!(parsed.world_size(), 3);
        assert_eq!(parsed.controller.host, "192.168.122.1");
        assert_eq!(parsed.worker.host, "host-a");
        assert_eq!(parsed.worker.ranks, vec![0]);
        assert_eq!(parsed.worker.local_devices, vec![0]);
        // FullCluster::from_value defaults salt to zeros; the envelope
        // carries that, and the rank-side parser reads it back.
        assert_eq!(parsed.salt, [0u8; crate::distributed::wire::SESSION_SALT_BYTES]);
    }

    #[test]
    fn slim_envelope_propagates_session_salt() {
        // The launcher generates a fresh salt and stamps it onto
        // FullCluster; every slim envelope it builds must carry that
        // salt unchanged through to the rank-side LocalCluster.
        let mut full = FullCluster::from_value(&canonical_full_json()).unwrap();
        let salt: crate::distributed::wire::SessionSalt = [
            0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04,
            0xfe, 0xed, 0xfa, 0xce, 0x05, 0x06, 0x07, 0x08,
        ];
        full = full.with_session_salt(salt);
        let host_a = full.workers.iter().find(|h| h.host == "host-a").unwrap();
        let env = build_slim_envelope_for(&full, host_a);
        // Salt field must be present as a 32-char lowercase hex string.
        let hex = env
            .get("salt")
            .and_then(|v| v.as_str())
            .expect("envelope.salt is a string");
        assert_eq!(hex.len(), 32);
        let parsed = crate::distributed::cluster::LocalCluster::from_value(&env).unwrap();
        assert_eq!(parsed.salt, salt);
    }

    #[test]
    fn supervise_children_clean_exit_returns_none() {
        // Both children exit cleanly. supervise_children should return
        // None without sending any kill signals.
        let mut children: Vec<(
            String,
            usize,
            std::process::Child,
            Vec<std::thread::JoinHandle<()>>,
        )> = Vec::new();
        for lr in 0..2 {
            let child = Command::new("true")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn `true`");
            children.push(("host".to_string(), lr, child, Vec::new()));
        }
        assert!(supervise_children(children).is_none());
    }

    #[test]
    fn supervise_children_failure_terminates_peers() {
        // One child exits immediately with status 1; the other would
        // sleep for 60s. Concurrent supervision must detect the failure
        // and SIGTERM the sleeper so the call returns promptly. The
        // assertion is the wall-clock budget: significantly less than
        // the sleeper's 60s argument.
        let fail_child = Command::new("sh")
            .args(["-c", "exit 1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn `sh -c 'exit 1'`");
        let sleep_child = Command::new("sleep")
            .arg("60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn `sleep 60`");
        let children = vec![
            ("host-fail".to_string(), 0, fail_child, Vec::new()),
            ("host-sleep".to_string(), 1, sleep_child, Vec::new()),
        ];

        let start = std::time::Instant::now();
        let err = supervise_children(children).expect("expected failure attribution");
        let elapsed = start.elapsed();

        assert!(
            err.to_string().contains("host-fail"),
            "attribution should name the first failed rank: {err}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "SIGTERM-on-failure must reap the sleeper well before its 60s budget; took {elapsed:?}"
        );
    }
}
