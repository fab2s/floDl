//! Cluster-mode env preparation.
//!
//! Process-per-rank model: flodl owns fan-out and controller
//! orchestration (`flodl::distributed::launcher::run_launcher`). fdl-cli's
//! job here is purely to ship the parsed cluster topology to the launcher
//! via env vars, then let the normal `RunScript` / `ExecCommand` dispatch
//! invoke the user binary. The user binary's
//! `flodl::distributed::launcher::dispatch` reads the env, detects
//! launcher role, and fans out (ssh for remote hosts, fork+exec for local
//! hosts). All log fan-in + ClusterController + exit-code propagation happen on
//! the flodl side.
//!
//! ```text
//! fdl @cluster train
//!   ↓ fdl-cli parses fdl.yml + fdl.cluster.yml overlay
//!   ↓ fdl-cli calls prepare_cluster_env: sets FLODL_INTERNAL_FULL_CLUSTER_JSON,
//!     FLODL_INTERNAL_FDL_CMD, FDL_ENV on its own process env
//!   ↓ fdl-cli falls through to normal RunScript / ExecCommand path
//!   ↓ resolved command (e.g. `cargo run --release --bin my-trainer`) runs
//!   ↓ my-trainer inherits env, flodl::launcher::dispatch detects Launcher
//!   ↓ launcher fans out: ssh per remote host, fork+exec per local rank
//!   ↓ each rank child has FLODL_INTERNAL_CLUSTER_JSON + FLODL_INTERNAL_LOCAL_RANK set
//!   ↓ rank-side flodl::launcher::dispatch returns Role::Rank, training runs
//! ```
//!
//! Recursion guard: the launcher's ssh fan-out invokes `fdl <cmd>` on the
//! remote, which re-enters fdl-cli with `FLODL_INTERNAL_CLUSTER_JSON` set (not
//! `FLODL_INTERNAL_FULL_CLUSTER_JSON`). `should_dispatch`
//! returns `false` in that case so the remote fdl-cli skips cluster setup
//! and just runs the user binary normally — the user binary's launcher
//! dispatch then detects `Role::Rank` (because `FLODL_INTERNAL_LOCAL_RANK` is also
//! set).

use std::path::Path;
use std::process::Command;

use crate::config::{self, ClusterConfig, ProjectConfig};

/// Env var name carrying the *full* multi-host topology (hex-encoded
/// JSON of [`ClusterConfig`]). Set by fdl-cli on its own process env so
/// the spawned user binary inherits it and detects launcher role.
/// Mirrors `flodl::distributed::launcher::ENV_FULL_CLUSTER_JSON`.
pub const ENV_FULL_CLUSTER_JSON: &str = "FLODL_INTERNAL_FULL_CLUSTER_JSON";

/// Env var name carrying the original fdl command name (e.g. `train`).
/// Read by the launcher when it needs to invoke `fdl <cmd>` over ssh
/// on remote hosts. Mirrors `flodl::distributed::launcher::ENV_FDL_CMD`.
pub const ENV_FDL_CMD: &str = "FLODL_INTERNAL_FDL_CMD";

/// Env var name picking the overlay env name (e.g. `cluster`). Set by
/// fdl-cli at env-selector parsing time; propagated through to remote
/// hosts by the launcher so they see the same overlay-merged view.
pub const ENV_FDL_ENV: &str = "FDL_ENV";

/// Env var name carrying the slim per-rank envelope. Set by the
/// launcher (not fdl-cli) on each rank child. Kept here so the
/// recursion guard can reference it by name. Mirrors
/// `flodl::distributed::cluster::ENV_CLUSTER_JSON`.
pub const ENV_CLUSTER_JSON: &str = "FLODL_INTERNAL_CLUSTER_JSON";

/// Pre-resolved `name:ip` pairs (space-separated) for every cluster
/// host, written by [`prepare_cluster_env`] using the controller's NSS
/// resolution. Consumed by run/prebuild/schema-cache when they build
/// `docker compose run --rm` commands: each pair is injected as a
/// `--add-host name:ip` flag so the containerized launcher can SSH
/// into cluster hosts without depending on the container's own
/// resolver (which lacks `libnss-libvirt` etc.).
pub const ENV_CLUSTER_EXTRA_HOSTS: &str = "FLODL_INTERNAL_CLUSTER_EXTRA_HOSTS";

/// Controller's OS user name (resolved on the host by fdl-cli, before
/// any docker spawn). The launcher in the container reads it as the
/// default `ssh -l` target when the per-host `ssh.user:` is unset.
/// Bridges the container-vs-host user mismatch (containers ship a
/// stock `ubuntu` UID-1000 user, but `ubuntu@<remote>` is rarely the
/// account the user actually uses on cluster hosts).
pub const ENV_HOST_USER: &str = "FLODL_INTERNAL_HOST_USER";

/// Env var name overriding the OS hostname for cluster lookups.
/// Mirrors `flodl::distributed::cluster::ENV_HOST_OVERRIDE`.
pub const ENV_HOST_OVERRIDE: &str = "FLODL_HOST_NAME";

/// Env var name picking this rank's local-rank index within its host.
/// Set by the launcher on rank children. Mirrors
/// `flodl::distributed::cluster::ENV_LOCAL_RANK`.
pub const ENV_LOCAL_RANK: &str = "FLODL_INTERNAL_LOCAL_RANK";

/// True if `key` must not appear in a user-supplied cluster/host `env`
/// map. Mirrors `flodl::distributed::cluster::is_reserved_cluster_env_key`
/// (kept in lockstep — flodl-cli is decoupled from the library crate, so
/// the reserved rule is duplicated, not imported). The launcher applies
/// user env after its own built-ins (shell last-wins), so a launcher-
/// owned key set via env would silently override device mapping / rank
/// identity / the HMAC envelope. Reserved: the loud `FLODL_INTERNAL_`
/// prefix (all launcher-private vars, future-proof) plus three names that
/// are user-facing elsewhere but launcher-owned per-rank here
/// (`CUDA_VISIBLE_DEVICES` + `CUDA_DEVICE_ORDER`, [`ENV_HOST_OVERRIDE`],
/// [`ENV_FDL_ENV`]).
/// User-facing knobs (`FLODL_VERBOSITY`, `FLODL_DASHBOARD_BIND`, NCCL
/// tuning, `LD_LIBRARY_PATH`) are deliberately allowed.
pub fn is_reserved_cluster_env_key(key: &str) -> bool {
    key.starts_with("FLODL_INTERNAL_")
        || key == "CUDA_VISIBLE_DEVICES"
        || key == "CUDA_DEVICE_ORDER"
        // The AMD masks outrank CUDA_VISIBLE_DEVICES for HIP (first one
        // set wins), so an env-block value would silently defeat the
        // launcher's per-rank pin — the same reservation, other vendor.
        || key == "HIP_VISIBLE_DEVICES"
        || key == "ROCR_VISIBLE_DEVICES"
        || key == "GPU_DEVICE_ORDINAL"
        || key == ENV_HOST_OVERRIDE
        || key == ENV_FDL_ENV
}

/// Env var scaling every flodl cluster network deadline together
/// (connect budget, write-stall, heartbeat staleness, coord-liveness,
/// CPU reduce read). Mirrors `flodl::distributed::wire`'s constant +
/// validation rule (kept in lockstep — flodl-cli is decoupled from the
/// library crate). The library reader warns-and-defaults on a bad
/// value; the cluster fan-out path calls
/// [`validate_net_timeout_scale`] first so an explicit-but-invalid
/// value errors loudly BEFORE any host is touched.
pub const ENV_NET_TIMEOUT_SCALE: &str = "FLODL_NET_TIMEOUT_SCALE";

/// Validate `FLODL_NET_TIMEOUT_SCALE` from the process env: unset is
/// fine (scale 1.0); a set value must parse as a finite float ≥ 0.1.
/// Mirrors the library's parse rule.
pub fn validate_net_timeout_scale() -> Result<(), String> {
    validate_net_timeout_scale_value(std::env::var(ENV_NET_TIMEOUT_SCALE).ok().as_deref())
}

/// Pure core of [`validate_net_timeout_scale`] (unit-testable without
/// env mutation).
fn validate_net_timeout_scale_value(raw: Option<&str>) -> Result<(), String> {
    let Some(raw) = raw else { return Ok(()) };
    let trimmed = raw.trim();
    match trimmed.parse::<f64>() {
        Ok(v) if v.is_finite() && v >= 0.1 => Ok(()),
        Ok(_) => Err(format!(
            "{ENV_NET_TIMEOUT_SCALE}={trimmed} is out of range; expected a \
             finite scale factor >= 0.1 (0.1 keeps every deadline above the \
             1s heartbeat cadence)"
        )),
        Err(_) => Err(format!(
            "{ENV_NET_TIMEOUT_SCALE}={trimmed:?} is not a number; expected a \
             scale factor >= 0.1 (e.g. 3 for a slow WAN link, 0.5 for a \
             fast-failure test rig)"
        )),
    }
}

/// Top-level cluster-dispatch decision.
///
/// Returns `false` when `FLODL_INTERNAL_CLUSTER_JSON` is set — that signals we're
/// a recursive fdl invocation on a remote host that the launcher's ssh
/// fan-out reached, and we should fall through to normal dispatch.
/// Otherwise delegates to [`config::cluster_dispatch_enabled`].
pub fn should_dispatch(project: &ProjectConfig, chain: &[Option<bool>]) -> bool {
    if is_recursive_invocation() {
        return false;
    }
    config::cluster_dispatch_enabled(project, chain)
}

/// Whether this fdl invocation is itself a spawned child of a launcher's
/// ssh fan-out (`FLODL_INTERNAL_CLUSTER_JSON` already set in env). Used as the
/// recursion guard everywhere cluster dispatch is evaluated.
pub fn is_recursive_invocation() -> bool {
    std::env::var_os(ENV_CLUSTER_JSON).is_some()
}

/// A configured join window that this command will not open.
///
/// A farm overlay declares `controller.join.discovery`, but the window
/// only opens for a command running in launcher mode, which is what
/// `cluster: true` selects. Miss that and the command resolves against
/// the base config and trains locally — and nothing about the run says
/// so, because training on this box is a legitimate thing to do. Silence
/// here reads as "my GPUs were not detected" rather than "my farm never
/// engaged", so the mismatch is worth one line on stderr.
///
/// Pure: takes the merged config, returns the text. Returns `None` when
/// there is no discovery window to miss.
pub fn unused_join_window_hint(project: &ProjectConfig, command: &str) -> Option<String> {
    let join = project.cluster.as_ref()?.controller.join.as_ref()?;
    if join.discovery != Some(true) {
        return None;
    }
    Some(format!(
        "this env declares a join window (controller.join.discovery), but \
         `{command}` is not a cluster command, so it runs HERE and no \
         window opens. Add it to the env's `commands:` with `cluster: \
         true` to put it in launcher mode."
    ))
}

/// Prepare the env vars needed for the user binary's flodl launcher to
/// detect launcher role and fan out. Caller continues to normal
/// dispatch (`RunScript` / `ExecCommand`); the spawned subprocess
/// inherits these env vars and the launcher takes over.
///
/// `overlay_env` is the overlay name from `fdl @<env>` (e.g.
/// `Some("cluster")`); propagated to remote hosts via the launcher so
/// they see the same overlay-merged `commands:` resolution.
///
/// Returns `Err` if the cluster config is invalid or JSON serialization
/// fails — surfaces the error before the user binary even starts.
///
/// On success returns a `Vec<String>` of non-fatal resolution warnings
/// (one entry per host whose NSS lookup failed or yielded only loopback
/// addresses). The cluster-dispatch site in `main.rs` is the one that
/// chooses to print them. Tests that exercise this function for its
/// env-setting behavior simply ignore the returned Vec.
pub fn prepare_cluster_env(
    cluster: &ClusterConfig,
    overlay_env: Option<&str>,
    cmd: &str,
    launcher_root: &Path,
) -> Result<Vec<String>, String> {
    cluster.validate()?;
    let mut warnings: Vec<String> = Vec::new();
    // Pre-resolve `controller.host` on the controller (where NSS knows
    // names declared in `/etc/hosts`, `libnss-libvirt`, mDNS, etc.)
    // and ship the resolved IP in the envelope to remote ranks. Remote
    // VMs that don't share the controller's NSS view (a Pascal VM on
    // libvirt's virbr0 has no plugin to resolve "exa") then connect
    // by numeric IP without needing their own resolver to know cluster
    // hostnames. If resolution fails on the controller, ship the
    // original string and let the remote try its own NSS as a last
    // resort.
    let mut shippable = cluster.clone();
    let (controller_ip, controller_warning) = resolve_host_to_ip(&shippable.controller.host);
    if let Some(ip) = controller_ip {
        shippable.controller.host = ip;
    }
    if let Some(w) = controller_warning {
        warnings.push(w);
    }
    // Probe device counts per worker, then populate global ranks by
    // sequential assignment. `ranks` is not user-facing in the YAML
    // schema (see `ClusterWorker::ranks` — `skip_deserializing`); the
    // probe result is authoritative. Probing only SSHes for workers
    // that use `local_devices: all`; explicit lists carry their own
    // count. After populate_ranks the cluster shape on the wire is
    // identical to the legacy explicit form.
    let counts = probe_worker_device_counts(&shippable)?;
    shippable.populate_ranks(&counts)?;
    shippable.validate()?;
    // Ship a relative `ssh.identity_file` ABSOLUTE in the launcher's frame.
    // The envelope's ssh consumer is the training binary's launcher (agent
    // spawn, tunnel sessions), which hands the value to ssh verbatim from
    // whatever cwd the run configured — while fdl's own ssh (the probe
    // above, prebuild) resolves a relative path against its project root
    // (see `resolve_identity_path`). `launcher_root` is the project root as
    // the TRAINING COMMAND sees it: the compose mount for a `docker:`
    // command, the host checkout otherwise — so one relative overlay value
    // names the same key file in every frame that opens an ssh connection.
    // Deliberately AFTER the device probe (which sshes with the un-rewritten
    // relative value, resolved host-side) and before envelope emission.
    for w in &mut shippable.workers {
        if let Some(id) = w.ssh.as_mut().and_then(|s| s.identity_file.as_mut())
            && !identity_passes_through(id)
        {
            *id = join_identity_path(launcher_root, id);
        }
    }
    let json = shippable.canonical_json()?;
    let hex = hex_encode(json.as_bytes());
    let (extra_hosts, host_warnings) = resolve_cluster_extra_hosts(cluster);
    warnings.extend(host_warnings);

    // Controller user for the launcher's default `ssh -l`. On the rare
    // double-failure (no USER, no whoami) leave the env UNSET — the
    // launcher then omits `-l` and ssh applies its own defaults — and
    // warn, instead of shipping a fabricated username into ssh auth.
    let host_user = resolve_local_user();
    if host_user.is_none() {
        warnings.push(
            "could not determine the controller's user (USER unset, whoami \
             unavailable); ssh will use its own defaults — set `ssh.user:` \
             per host in fdl.cluster.yml if remote accounts differ"
                .to_string(),
        );
    }
    // SAFETY: main() has not spawned threads at this point in the
    // dispatch flow (mirrors gpus::apply_cuda_visible_devices's
    // invariant; documented in main.rs).
    unsafe {
        std::env::set_var(ENV_FULL_CLUSTER_JSON, &hex);
        std::env::set_var(ENV_FDL_CMD, cmd);
        if let Some(u) = &host_user {
            std::env::set_var(ENV_HOST_USER, u);
        }
        if !extra_hosts.is_empty() {
            std::env::set_var(ENV_CLUSTER_EXTRA_HOSTS, extra_hosts.join(" "));
        }
        if let Some(e) = overlay_env
            && !e.trim().is_empty()
        {
            std::env::set_var(ENV_FDL_ENV, e);
        }
    }
    Ok(warnings)
}

/// Local-only sibling of [`probe_worker_device_counts`], used by the
/// testing-envelope export path in [`prepare_test_cluster_env`].
///
/// Testing-mode cluster invocations (`fdl @cluster-test <cmd>`) run the
/// test binary in-process on one host; there's no SSH fan-out, so any
/// worker declaring `local_devices: all` is by definition referring to
/// the local machine's visible GPUs. Use `nvidia-smi -L` locally
/// instead of SSHing back to ourselves.
///
/// - `LocalDevices::Explicit(v)` → `v.len()`
/// - `LocalDevices::All` → [`crate::gpus::local_gpu_count`]
///   (result cached across workers since they all resolve to the same
///   local box in testing mode).
///
/// Errors loudly when no GPU is detected, quoting the reason (caller
/// treats 0 as misconfiguration — no GPUs visible to the test).
fn probe_local_device_counts(cluster: &ClusterConfig) -> Result<Vec<usize>, String> {
    let mut counts = Vec::with_capacity(cluster.workers.len());
    let mut cached_local: Option<usize> = None;
    for (i, w) in cluster.workers.iter().enumerate() {
        let count = match &w.local_devices {
            config::LocalDevices::Explicit(v) => v.len(),
            config::LocalDevices::All => {
                if cached_local.is_none() {
                    cached_local = Some(crate::gpus::local_gpu_count().map_err(|e| {
                        format!(
                            "cluster.workers[{i}] ({:?}): local GPU probe \
                                 failed: {e}",
                            w.host,
                        )
                    })?);
                }
                cached_local.unwrap()
            }
        };
        if count == 0 {
            return Err(format!(
                "cluster.workers[{i}] ({:?}): 0 CUDA devices visible \
                 (local_devices: all). Run `fdl @cluster-test <cmd>` on a \
                 host with visible GPUs, or use an explicit \
                 `local_devices: [...]` list.",
                w.host,
            ));
        }
        counts.push(count);
    }
    Ok(counts)
}

/// Prepare the testing-cluster envelope (`FLODL_TESTING_CLUSTER_JSON`).
///
/// Mirrors [`prepare_cluster_env`] for the testing path: clones the
/// cluster, probes device counts LOCALLY (no SSH — tests run
/// in-process on the local host), populates ranks by sequential
/// assignment, then serializes for shipping.
///
/// Returns the hex-encoded envelope ready to set in the env. Errors
/// surface from `cluster.validate()` (structural), the local probe,
/// or `populate_ranks` (count/worker mismatch).
pub fn prepare_test_cluster_env(cluster: &ClusterConfig) -> Result<String, String> {
    cluster.validate()?;
    let mut shippable = cluster.clone();
    let counts = probe_local_device_counts(&shippable)?;
    shippable.populate_ranks(&counts)?;
    shippable.validate()?;
    let json = shippable.canonical_json()?;
    Ok(hex_encode(json.as_bytes()))
}

/// Probe each worker's CUDA device count.
///
/// - `LocalDevices::Explicit(v)` → `v.len()` (no SSH, no remote call)
/// - `LocalDevices::All` → SSH to the worker and run
///   `nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l`,
///   parse as `usize`. Errors loudly on SSH failure, parse failure, or
///   a 0 count (caller treats 0 as misconfiguration).
///
/// Returns one count per worker, in worker-declaration order. Used by
/// [`prepare_cluster_env`] before envelope emission so global rank
/// assignment can be sequential without requiring users to enumerate
/// `ranks:` in YAML.
fn probe_worker_device_counts(cluster: &ClusterConfig) -> Result<Vec<usize>, String> {
    let mut counts = Vec::with_capacity(cluster.workers.len());
    for (i, w) in cluster.workers.iter().enumerate() {
        let count = match &w.local_devices {
            config::LocalDevices::Explicit(v) => v.len(),
            config::LocalDevices::All => ssh_query_gpu_count(w)
                .map_err(|e| format!("cluster.workers[{i}] ({:?}): probe failed: {e}", w.host,))?,
        };
        if count == 0 {
            return Err(format!(
                "cluster.workers[{i}] ({:?}): probed 0 GPUs \
                 (local_devices: all). Either the host has no GPUs visible \
                 (NVIDIA: `nvidia-smi` and the visibility masks; AMD: \
                 `/dev/kfd` and a loaded amdgpu driver) or it's a \
                 misconfiguration — provide an explicit `local_devices: [...]` \
                 list instead.",
                w.host,
            ));
        }
        counts.push(count);
    }
    Ok(counts)
}

/// SSH to a worker and count visible CUDA devices via `nvidia-smi`.
///
/// Honors the worker's `ssh:` sub-block (target / port / user /
/// identity_file / options). Falls back to `host` as the ssh target
/// when no `ssh:` block is provided; system ssh resolves the rest via
/// `~/.ssh/config`.
///
/// Times out after 5s on connect to keep cluster startup snappy when a
/// host is unreachable. Uses `BatchMode=yes` so a passphrase prompt
/// doesn't hang the dispatch.
/// Apply a worker's `ssh:` sub-block (port / user / identity_file /
/// options) as flags on an `ssh` `Command`.
///
/// Call this BEFORE pushing flodl's default connect-behavior flags (`-T`,
/// `BatchMode`, timeouts), then the target host + remote command. User
/// `ssh.options` are emitted here first so they take precedence: OpenSSH uses
/// the first value seen for each `-o`, so flodl's later defaults fill in only
/// the options the user didn't set (M17). The one option flodl truly needs —
/// `BatchMode=yes`, so its non-interactive ssh never hangs on a prompt — is
/// still overridable, but [`batchmode_override_warning`] flags it loudly.
///
/// Shared by cluster GPU-count dispatch ([`ssh_query_gpu_count`]) and
/// `fdl probe`'s remote fan-out (`probe::probe_remote_via_ssh`) so both
/// reach a host the same way — notably a Docker-container rank exposed
/// on `127.0.0.1:<port>` with an `identity_file` (without these flags
/// ssh defaults to port 22 / the login user and the connect is
/// refused). The `target` itself is set by the caller (it falls back to
/// `worker.host` when no `ssh.target` is declared).
pub(crate) fn apply_worker_ssh_opts(cmd: &mut Command, worker: &config::ClusterWorker) {
    if let Some(ssh) = worker.ssh.as_ref() {
        if let Some(port) = ssh.port {
            cmd.arg("-p").arg(port.to_string());
        }
        if let Some(user) = ssh.user.as_deref() {
            cmd.arg("-l").arg(user);
        }
        if let Some(id) = ssh.identity_file.as_deref() {
            cmd.arg("-i").arg(resolve_identity_path(id));
        }
        if let Some(warning) = batchmode_override_warning(&ssh.options, &worker.host) {
            eprintln!("{warning}");
        }
        for opt in &ssh.options {
            cmd.arg("-o").arg(opt);
        }
    }
}

/// True when an `ssh.identity_file` value passes through to ssh untouched:
/// an explicit absolute path in ANY frame's spelling, or a `~`-prefixed
/// home path (ssh expands the tilde itself). The value may name a file in
/// a DIFFERENT filesystem namespace than the process reading it (a
/// container mount, a remote box), so a POSIX-rooted path must count as
/// absolute even on Windows — `Path::is_absolute()` alone calls
/// `/workspace/key` relative there and a `join` grafts the drive on
/// (`D:/workspace/key`, caught by the Windows CI leg). The reverse
/// spelling is covered by `is_absolute()` (`C:\...`, UNC).
pub(crate) fn identity_passes_through(id: &str) -> bool {
    id.starts_with('/') || id.starts_with('~') || Path::new(id).is_absolute()
}

/// Join a relative `ssh.identity_file` onto a project root with a plain
/// `/`, never [`Path::join`]: the result feeds ssh command lines and the
/// cross-frame cluster envelope, where a forward slash is right on every
/// platform (Windows OpenSSH included) — `Path::join` would render `\` on
/// Windows and ship a platform-spelled path into frames that are not that
/// platform.
pub(crate) fn join_identity_path(root: &Path, id: &str) -> String {
    format!("{}/{}", root.display(), id)
}

/// Resolve a relative `ssh.identity_file` against the PROJECT ROOT, not
/// the process cwd. The same overlay value is consumed from two execution
/// contexts — host-side (`fdl probe`'s remote fan-out) and container-side
/// (cluster dispatch, where the project root is the workspace mount) — so
/// an absolute path can only ever be right for one of them: a
/// container-absolute key path broke the host-side probe of a
/// containerized rank with "Identity file not accessible" + publickey
/// denial while dispatch kept working. A project-relative path names the
/// same file in both contexts, each side resolving against its own view
/// of the checkout. Pass-through spellings: [`identity_passes_through`].
/// Outside a project the fallback root is `~/.flodl` (the standard
/// [`Context::resolve`] semantics); a relative key path only makes sense
/// inside a checkout.
///
/// [`Context::resolve`]: crate::context::Context::resolve
fn resolve_identity_path(id: &str) -> std::ffi::OsString {
    if identity_passes_through(id) {
        return id.into();
    }
    join_identity_path(&crate::context::Context::resolve().root, id).into()
}

/// The warning message when a worker's `ssh.options` set `BatchMode` to a
/// non-`yes` value, else `None`. flodl's remote ssh (dispatch + probes) is
/// non-interactive, so a prompt hangs it; `BatchMode=yes` is the one truly
/// required ssh option. Every other flodl default is freely overridable (M17).
pub(crate) fn batchmode_override_warning(opts: &[String], host: &str) -> Option<String> {
    opts.iter().find_map(|opt| {
        let (k, v) = opt.split_once('=')?;
        (k.trim().eq_ignore_ascii_case("BatchMode") && !v.trim().eq_ignore_ascii_case("yes")).then(
            || {
                format!(
                    "fdl: host {host:?} ssh.options set `{}` — flodl's ssh is \
                 non-interactive and will hang on any prompt (passphrase, \
                 host-key). Proceeding as requested.",
                    opt.trim()
                )
            },
        )
    })
}

fn ssh_query_gpu_count(worker: &config::ClusterWorker) -> Result<usize, String> {
    let target = worker
        .ssh
        .as_ref()
        .and_then(|s| s.target.as_deref())
        .unwrap_or(&worker.host);
    let mut cmd = Command::new("ssh");
    // User ssh.options first (they win), then flodl's defaults (M17).
    apply_worker_ssh_opts(&mut cmd, worker);
    cmd.args(["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]);
    cmd.arg(target);
    // Both vendors in one round trip, because "how many GPUs" is not an
    // NVIDIA question: counting only nvidia-smi made an AMD worker probe
    // 0 and abort the fan-out, with an error naming a tool that host
    // does not have. Each count is piped to `wc -l` for a numeric line
    // and each error stream is silenced, so an absent driver or an
    // unmatched glob produces "0" rather than parse noise. The AMD side
    // reads the KFD topology's `vendor_id 4098` (0x1002), which is
    // flodl-hw's primary AMD gate and mask-proof by construction.
    cmd.arg(
        "n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l); \
         a=$(grep -l '^vendor_id 4098$' /sys/class/kfd/kfd/topology/nodes/*/properties \
         2>/dev/null | wc -l); echo \"$n $a\"",
    );

    let output = cmd.output().map_err(|e| format!("ssh spawn failed: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "ssh to {target:?} exited {} (stderr: {stderr})",
            output.status,
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let counts = stdout.trim();
    let (nvidia, amd) = parse_gpu_counts(counts)
        .ok_or_else(|| format!("could not parse the remote GPU counts (got {counts:?})"))?;
    pick_worker_count(nvidia, amd, worker.arch.as_deref(), target)
}

/// Split the two-number reply of the remote count probe.
///
/// Reads the LAST non-empty line: the probe echoes one, but a remote
/// profile script that prints anything would otherwise shift the fields
/// and turn a working host into a parse error.
fn parse_gpu_counts(text: &str) -> Option<(usize, usize)> {
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    let mut it = line.split_whitespace();
    let nvidia = it.next()?.parse().ok()?;
    let amd = it.next()?.parse().ok()?;
    Some((nvidia, amd))
}

/// Reduce the per-vendor counts to the number of ranks this worker owns.
///
/// A libtorch build serves one vendor, so a host's rank count is the
/// count for the vendor it will actually run, which its declared
/// `arch:` names. Without that declaration a single-vendor host is still
/// unambiguous, and a host reporting both is genuinely undecidable here:
/// the controller cannot know which build that box will load, and
/// guessing assigns ranks to devices nobody will address.
fn pick_worker_count(
    nvidia: usize,
    amd: usize,
    arch: Option<&str>,
    target: &str,
) -> Result<usize, String> {
    match arch.and_then(crate::libtorch::detect::variant_vendor) {
        Some(crate::util::system::GpuVendor::Nvidia) => return Ok(nvidia),
        Some(crate::util::system::GpuVendor::Amd) => return Ok(amd),
        _ => {}
    }
    match (nvidia, amd) {
        (0, a) => Ok(a),
        (n, 0) => Ok(n),
        (n, a) => Err(format!(
            "{target:?} reports {n} NVIDIA and {a} AMD GPU(s), and one \
             libtorch build serves one vendor, so which of them this host \
             trains on cannot be inferred. Declare the host's `arch:` (the \
             libtorch variant it uses) or pin `local_devices: [...]`."
        )),
    }
}

/// Resolve each cluster worker's `host` to an IP via the controller's
/// NSS (which on Linux includes static `/etc/hosts`, `libnss-libvirt`,
/// `libnss-mdns`, and DNS — anything `getaddrinfo` knows about).
/// Returns `(Vec<"host:ip">, Vec<warning>)`: the first list is suitable
/// for `--add-host` injection into `docker compose run`; the second is
/// human-readable warnings the cluster-dispatch site can surface to the
/// user. Workers that fail to resolve are skipped from the `host:ip`
/// list (better-than-nothing semantics for the launcher inside the
/// container — the unresolved host will retry via its own NSS).
fn resolve_cluster_extra_hosts(cluster: &ClusterConfig) -> (Vec<String>, Vec<String>) {
    let mut hosts = Vec::new();
    let mut warnings = Vec::new();
    for w in &cluster.workers {
        let (ip, warning) = resolve_host_to_ip(&w.host);
        if let Some(ip) = ip {
            hosts.push(format!("{}:{ip}", w.host));
        }
        // Suppress the warning when the worker carries an explicit
        // `ssh.target`. The `host:` value is then just a label — the
        // actual connection uses ssh.target (or `host:` only as the
        // default ssh target, which is itself a system-ssh lookup, not
        // a process-controller-side NSS lookup). Workers reached via
        // ~/.ssh/config aliases (e.g. a `node-b` alias mapping to
        // 127.0.0.1:2222) routinely don't resolve via host NSS, and
        // surfacing the warning every run is noise.
        let has_explicit_ssh_target = w.ssh.as_ref().and_then(|s| s.target.as_deref()).is_some();
        if let Some(msg) = warning
            && !has_explicit_ssh_target
        {
            warnings.push(msg);
        }
    }
    (hosts, warnings)
}

/// Resolve a hostname to an IP string via `getaddrinfo`. Returns
/// `(Option<ip>, Option<warning>)` — both are independently optional so
/// the caller can ship the resolved IP AND surface the warning, or
/// either alone. The function itself is silent; warnings are returned
/// for the cluster-dispatch site to emit (or ignore in non-dispatch
/// contexts like unit tests that exercise the env-setting paths).
///
/// Prefers a non-loopback address when `getaddrinfo` returns several
/// candidates. Debian/Ubuntu install `/etc/hosts` with a
/// `127.0.1.1 <hostname>` line by default, which `getaddrinfo` returns
/// FIRST — that IP works for the local host but is unreachable from
/// any peer (a libvirt VM, another rig). Skipping loopback in the
/// iterator picks the bridge / LAN address remote ranks can actually
/// dial. If ONLY loopback resolves, return it WITH a warning string
/// (likely misconfig — better to surface than to silently ship an
/// unreachable IP).
fn resolve_host_to_ip(host: &str) -> (Option<String>, Option<String>) {
    use std::net::ToSocketAddrs;
    // Already a numeric address? Skip the lookup, return as-is.
    if host.parse::<std::net::IpAddr>().is_ok() {
        return (Some(host.to_string()), None);
    }
    match (host, 0u16).to_socket_addrs() {
        Ok(iter) => {
            let (ip, only_loopback) = select_preferred_ip(iter.map(|sa| sa.ip()));
            let warning = match (&ip, only_loopback) {
                (Some(ip), true) => Some(format!(
                    "host {host:?} only resolves to loopback {ip} on the controller \
                     — remote ranks will fail to connect. Set the controller's `host:` \
                     in fdl.cluster.yml to a non-loopback IP reachable from peer nodes \
                     (e.g. the libvirt bridge IP 192.168.122.1 for virbr0). An explicit \
                     IP is used as-is (it is the address ranks dial), bypassing this \
                     hostname resolution; on a multi-NIC controller this is also how you \
                     pin which address is advertised."
                )),
                _ => None,
            };
            (ip, warning)
        }
        Err(e) => (
            None,
            Some(format!(
                "host {host:?} did not resolve on controller: {e} (remote ranks \
                 will retry via their own NSS — fix host-side resolution if they \
                 also fail)"
            )),
        ),
    }
}

/// Pick the best IP from a `getaddrinfo` iterator: first non-loopback
/// wins; if every candidate is loopback, return the first loopback
/// with `only_loopback=true` so the caller can warn. Pure function,
/// no NSS dependency — exists so the selection rule is unit-testable.
fn select_preferred_ip<I: IntoIterator<Item = std::net::IpAddr>>(
    iter: I,
) -> (Option<String>, bool) {
    let mut loopback_fallback: Option<String> = None;
    for ip in iter {
        if !ip.is_loopback() {
            return (Some(ip.to_string()), false);
        }
        loopback_fallback.get_or_insert_with(|| ip.to_string());
    }
    let only_loopback = loopback_fallback.is_some();
    (loopback_fallback, only_loopback)
}

/// Write a temporary docker-compose overlay (under `project_root`)
/// that populates `extra_hosts:` for the cluster-capable services
/// (`cuda`, `dev`, `bench`) from the controller-resolved cluster
/// hosts in [`ENV_CLUSTER_EXTRA_HOSTS`], then return the `-f` flag
/// sequence to splice in front of `docker compose run`.
///
/// `docker compose run` itself does not accept `--add-host` (that's
/// `docker run` only), but compose merges multiple `-f` files, so the
/// overlay extends the base config without mutating it.
///
/// Returns the empty string (and writes nothing) when not in cluster
/// mode — non-cluster runs keep their existing `docker compose run`
/// invocation unchanged.
pub fn cluster_compose_overlay_arg(project_root: &Path) -> String {
    let raw = match std::env::var(ENV_CLUSTER_EXTRA_HOSTS) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let pairs: Vec<&str> = raw.split_whitespace().filter(|p| !p.is_empty()).collect();
    if pairs.is_empty() {
        return String::new();
    }

    let mut entries = String::new();
    for pair in &pairs {
        entries.push_str("      - \"");
        // Escape for a YAML double-quoted scalar so a `\` or `"` in the
        // value can't break out of the entry. Hostnames/IPs shouldn't
        // contain these, but the `host:` half comes from user-authored
        // cluster.yml, so don't trust it into a quoted string unescaped.
        for ch in pair.chars() {
            match ch {
                '\\' => entries.push_str("\\\\"),
                '"' => entries.push_str("\\\""),
                _ => entries.push(ch),
            }
        }
        entries.push_str("\"\n");
    }

    // extra_hosts is per-service in compose; apply to every
    // cluster-capable service so the same override file works
    // regardless of which one the dispatch lands on.
    let overlay = format!(
        "# Generated by fdl-cli (cluster mode) — DO NOT EDIT BY HAND.\n\
         # Regenerated on every `fdl @cluster ...` invocation.\n\
         services:\n\
         \x20\x20cuda:\n\
         \x20\x20\x20\x20extra_hosts:\n{entries}\
         \x20\x20dev:\n\
         \x20\x20\x20\x20extra_hosts:\n{entries}\
         \x20\x20bench:\n\
         \x20\x20\x20\x20extra_hosts:\n{entries}",
    );

    let overlay_path = project_root.join(".fdl-cluster-overlay.yml");
    if let Err(e) = std::fs::write(&overlay_path, overlay) {
        eprintln!(
            "fdl: warning: failed to write cluster compose overlay at {:?}: {e} \
             (continuing without --add-host injection — remote hostnames \
             may not resolve inside the container)",
            overlay_path
        );
        return String::new();
    }

    // base docker-compose.yml first, then our overlay second, so the
    // overlay's extra_hosts merges into the base service definitions.
    // Quoted: this string is spliced into an `sh -c` command line, so a
    // project path with a space would otherwise shatter the parse.
    format!(
        " -f docker-compose.yml -f {}",
        crate::util::shell::posix_quote(&overlay_path.display().to_string())
    )
}

/// Hex-encode raw bytes (lowercase, no separators). Companion to the
/// library's `hex_decode` in `flodl::distributed::cluster`. Kept here
/// so `prepare_cluster_env` doesn't pull in a flodl runtime dep.
pub fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(TABLE[(b >> 4) as usize] as char);
        s.push(TABLE[(b & 0x0F) as usize] as char);
    }
    s
}

/// Resolve the controller's OS user name. Used to pre-populate
/// [`ENV_HOST_USER`] before docker spawn, so the launcher inside the
/// container can default `ssh -l <user>` to the host's identity.
/// Falls through `USER` then `whoami`; `None` when both fail — the
/// caller then leaves [`ENV_HOST_USER`] unset and the launcher omits
/// `-l` entirely, so ssh applies its own defaults (`ssh_config User`
/// directives, the effective uid). Never fabricate a name: a made-up
/// `-l unknown-user` produced a bare "Permission denied (publickey)"
/// with no hint the username was invented.
pub fn resolve_local_user() -> Option<String> {
    resolve_user_from(
        std::env::var("USER").ok().as_deref(),
        Command::new("whoami").output().ok().and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok()
            } else {
                None
            }
        }),
    )
}

/// Pure core of [`resolve_local_user`] (unit-testable without env
/// mutation): first non-empty of the `USER` value and the `whoami`
/// output, both trimmed.
fn resolve_user_from(user_env: Option<&str>, whoami_out: Option<String>) -> Option<String> {
    if let Some(s) = user_env {
        let s = s.trim();
        if !s.is_empty() {
            return Some(s.to_string());
        }
    }
    whoami_out
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Resolve the local OS hostname. Used by `gpus::synthesize_local_cluster`
/// (the `--gpus` single-host shorthand) and by `prebuild` to skip the
/// controller from the remote-host fan-out. Test/override seam via
/// [`ENV_HOST_OVERRIDE`]; falls back to the `hostname(1)` command.
pub fn resolve_local_hostname() -> String {
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
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown-host".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test_env::env_lock;

    /// A relative `identity_file` resolves against the project root (the
    /// same overlay is read host-side and container-side, and only a
    /// project-relative path names the same key file in both contexts);
    /// absolute and `~`-prefixed paths pass through untouched.
    #[test]
    fn a_relative_identity_file_resolves_against_the_project_root() {
        use crate::config::{ClusterWorker, LocalDevices, SshConfig};
        let worker_with_id = |id: &str| ClusterWorker {
            host: "h".into(),
            ranks: vec![0],
            local_devices: LocalDevices::Explicit(vec![0]),
            nccl_socket_ifname: "lo".into(),
            path: "/tmp".into(),
            ssh: Some(SshConfig {
                identity_file: Some(id.into()),
                ..SshConfig::default()
            }),
            tunnel: false,
            arch: None,
            data_path: None,
            gpu_ram_share: None,
            docker: None,
            env: std::collections::BTreeMap::new(),
        };
        let identity_arg = |id: &str| -> std::ffi::OsString {
            let mut cmd = Command::new("ssh");
            apply_worker_ssh_opts(&mut cmd, &worker_with_id(id));
            let args: Vec<_> = cmd.get_args().map(|a| a.to_os_string()).collect();
            let i = args
                .iter()
                .position(|a| a == "-i")
                .expect("-i flag present");
            args[i + 1].clone()
        };

        // Relative: project-root-joined (asserted against the same
        // resolver + joiner the code uses, so the test is cwd- and
        // platform-independent — the join is a plain `/`, never the
        // platform separator, because the value feeds ssh and
        // cross-frame envelopes).
        let expect: std::ffi::OsString =
            join_identity_path(&crate::context::Context::resolve().root, ".fdl/farm/key").into();
        assert_eq!(identity_arg(".fdl/farm/key"), expect);

        // Pass-through spellings, verbatim on EVERY platform: a
        // POSIX-rooted path may name a file in another frame (a container
        // mount), so it must not be called relative on Windows — the
        // first Windows CI run grafted the drive on (`D:/etc/flodl/key`).
        assert_eq!(identity_arg("/etc/flodl/key"), "/etc/flodl/key");
        assert_eq!(identity_arg("~/.ssh/key"), "~/.ssh/key");
        #[cfg(windows)]
        assert_eq!(identity_arg("C:\\keys\\id"), "C:\\keys\\id");
    }

    #[test]
    fn remote_gpu_counts_parse_as_a_vendor_pair() {
        assert_eq!(parse_gpu_counts("2 0"), Some((2, 0)));
        assert_eq!(parse_gpu_counts(" 0   8 "), Some((0, 8)));
        // A host answering with anything else must be a loud parse
        // failure, never a silent zero that reads as "no GPUs".
        assert_eq!(parse_gpu_counts("2"), None);
        assert_eq!(parse_gpu_counts(""), None);
        assert_eq!(parse_gpu_counts("bash: nvidia-smi: not found"), None);
        // A remote profile script that prints must not shift the fields:
        // the counts are the last line, not the first two tokens.
        assert_eq!(parse_gpu_counts("Welcome to node 7\n0 8"), Some((0, 8)));
        assert_eq!(parse_gpu_counts("2 0\n"), Some((2, 0)));
    }

    #[test]
    fn a_workers_rank_count_follows_the_vendor_it_declares() {
        // The bug this guards: the probe counted nvidia-smi only, so an
        // AMD worker with `local_devices: all` probed 0 and aborted the
        // fan-out, quoting a tool that host does not have.
        assert_eq!(
            pick_worker_count(0, 8, Some("precompiled/rocm70"), "h"),
            Ok(8)
        );
        assert_eq!(
            pick_worker_count(2, 0, Some("precompiled/cu128"), "h"),
            Ok(2)
        );
        // A declared arch outranks the other vendor's cards being present.
        assert_eq!(
            pick_worker_count(2, 8, Some("precompiled/rocm71"), "h"),
            Ok(8)
        );
        assert_eq!(
            pick_worker_count(2, 8, Some("builds/sm61-sm120"), "h"),
            Ok(2)
        );
    }

    #[test]
    fn an_undeclared_single_vendor_host_still_resolves() {
        assert_eq!(pick_worker_count(4, 0, None, "h"), Ok(4));
        assert_eq!(pick_worker_count(0, 4, None, "h"), Ok(4));
        assert_eq!(pick_worker_count(0, 0, None, "h"), Ok(0));
        // Both vendors and nothing declared: undecidable HERE (the
        // controller cannot know which build that box loads), so it must
        // say so rather than assign ranks to devices nobody addresses.
        let err = pick_worker_count(2, 8, None, "mixed-host").unwrap_err();
        assert!(err.contains("2 NVIDIA and 8 AMD"), "got {err}");
        assert!(err.contains("arch:"), "got {err}");
    }

    #[test]
    fn net_timeout_scale_validation_mirrors_library_rule() {
        // Pure core — no env mutation needed.
        assert!(validate_net_timeout_scale_value(None).is_ok());
        assert!(validate_net_timeout_scale_value(Some("3")).is_ok());
        assert!(validate_net_timeout_scale_value(Some("0.1")).is_ok());
        assert!(validate_net_timeout_scale_value(Some(" 2.0 ")).is_ok());
        assert!(validate_net_timeout_scale_value(Some("0.05")).is_err());
        assert!(validate_net_timeout_scale_value(Some("-1")).is_err());
        assert!(validate_net_timeout_scale_value(Some("inf")).is_err());
        assert!(validate_net_timeout_scale_value(Some("abc")).is_err());
    }

    #[test]
    fn resolve_user_never_fabricates() {
        // Pure core — no env mutation needed.
        assert_eq!(resolve_user_from(Some("fab"), None).as_deref(), Some("fab"));
        assert_eq!(
            resolve_user_from(Some(" fab \n"), None).as_deref(),
            Some("fab")
        );
        // Empty USER falls through to whoami.
        assert_eq!(
            resolve_user_from(Some(""), Some("who\n".into())).as_deref(),
            Some("who"),
        );
        assert_eq!(
            resolve_user_from(None, Some("who".into())).as_deref(),
            Some("who")
        );
        // Double failure -> None (never a fabricated "unknown-user").
        assert_eq!(resolve_user_from(None, None), None);
        assert_eq!(resolve_user_from(Some("  "), Some("  ".into())), None);
    }

    #[test]
    fn should_dispatch_returns_false_when_cluster_json_set() {
        let _guard = env_lock();
        // SAFETY: serialized via env_lock() above.
        unsafe {
            std::env::set_var(ENV_CLUSTER_JSON, "deadbeef");
        }
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  x: { cluster: true, run: \"echo hi\" }
";
        let project: ProjectConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(
            !should_dispatch(&project, &[Some(true)]),
            "recursion guard: must return false when FLODL_INTERNAL_CLUSTER_JSON is set"
        );
        unsafe {
            std::env::remove_var(ENV_CLUSTER_JSON);
        }
    }

    #[test]
    fn should_dispatch_delegates_when_env_unset() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var(ENV_CLUSTER_JSON);
        }
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  x: { run: \"echo hi\" }
";
        let project: ProjectConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert!(!should_dispatch(&project, &[None]));
        assert!(should_dispatch(&project, &[Some(true)]));
    }

    #[test]
    fn hex_encode_matches_library() {
        // Well-known mappings; library's flodl::distributed::cluster::hex_decode
        // is the round-trip partner.
        assert_eq!(hex_encode(b""), "");
        assert_eq!(hex_encode(&[0x00]), "00");
        assert_eq!(hex_encode(&[0xff]), "ff");
        assert_eq!(hex_encode(&[0x0f, 0xa0]), "0fa0");
        assert_eq!(hex_encode(b"hi"), "6869");
    }

    /// The envelope ships a relative `ssh.identity_file` ABSOLUTE in the
    /// launcher's frame (the launcher hands it to ssh verbatim from
    /// whatever cwd the run configured); absolute and `~` values pass
    /// through untouched. Asserted by decoding the envelope fdl actually
    /// exports, not by inspecting an intermediate.
    #[test]
    fn prepare_cluster_env_absolutizes_relative_identity_into_the_envelope() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
        }
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: relative-key
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
      ssh:
        target: 127.0.0.1
        identity_file: .fdl/farm/key
    - host: tilde-key
      local_devices: [1]
      nccl_socket_ifname: lo
      path: /opt/flodl
      ssh:
        target: 127.0.0.1
        identity_file: ~/.ssh/key
commands:
  train: { cluster: true, run: \"true\" }
";
        let project: ProjectConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let cluster = project.cluster.as_ref().unwrap();
        prepare_cluster_env(cluster, None, "train", Path::new("/container/mount"))
            .expect("prepare OK");

        let hex = std::env::var(ENV_FULL_CLUSTER_JSON).unwrap();
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let shipped: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let id_of = |host: &str| -> String {
            shipped["workers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|w| w["host"] == host)
                .and_then(|w| w["ssh"]["identity_file"].as_str())
                .unwrap_or_default()
                .to_string()
        };
        assert_eq!(id_of("relative-key"), "/container/mount/.fdl/farm/key");
        assert_eq!(id_of("tilde-key"), "~/.ssh/key");
        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
        }
    }

    #[test]
    fn prepare_cluster_env_sets_required_vars() {
        let _guard = env_lock();
        // Clear env first so we observe what prepare_cluster_env sets.
        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_FDL_CMD);
            std::env::remove_var(ENV_FDL_ENV);
        }
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  train: { cluster: true, run: \"true\" }
";
        let project: ProjectConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let cluster = project.cluster.as_ref().unwrap();
        prepare_cluster_env(cluster, Some("cluster"), "train", Path::new("/tmp"))
            .expect("prepare OK");

        assert!(!std::env::var(ENV_FULL_CLUSTER_JSON).unwrap().is_empty());
        assert_eq!(std::env::var(ENV_FDL_CMD).unwrap(), "train");
        assert_eq!(std::env::var(ENV_FDL_ENV).unwrap(), "cluster");

        // Verify the full envelope round-trips back to the canonical JSON.
        let hex = std::env::var(ENV_FULL_CLUSTER_JSON).unwrap();
        // Decode and parse it as JSON.
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));

        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_FDL_CMD);
            std::env::remove_var(ENV_FDL_ENV);
        }
    }

    #[test]
    fn prepare_cluster_env_skips_fdl_env_when_blank() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var(ENV_FDL_ENV);
        }
        let yaml = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 29500
    path: /opt/flodl
  workers:
    - host: solo
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /opt/flodl
commands:
  train: { cluster: true, run: \"true\" }
";
        let project: ProjectConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let cluster = project.cluster.as_ref().unwrap();
        // None overlay → no FDL_ENV var set.
        prepare_cluster_env(cluster, None, "train", Path::new("/tmp")).unwrap();
        assert!(std::env::var_os(ENV_FDL_ENV).is_none());

        // Empty overlay → also no FDL_ENV var.
        prepare_cluster_env(cluster, Some("   "), "train", Path::new("/tmp")).unwrap();
        assert!(std::env::var_os(ENV_FDL_ENV).is_none());

        unsafe {
            std::env::remove_var(ENV_FULL_CLUSTER_JSON);
            std::env::remove_var(ENV_FDL_CMD);
        }
    }

    #[test]
    fn select_preferred_ip_prefers_non_loopback() {
        use std::net::IpAddr;
        // The Debian/Ubuntu /etc/hosts shape we have to handle:
        // 127.0.1.1 comes back FIRST, the routable LAN/bridge IP second.
        let ips: Vec<IpAddr> = vec![
            "127.0.1.1".parse().unwrap(),
            "192.168.122.1".parse().unwrap(),
        ];
        let (ip, only_loopback) = select_preferred_ip(ips);
        assert_eq!(ip.as_deref(), Some("192.168.122.1"));
        assert!(!only_loopback);
    }

    #[test]
    fn select_preferred_ip_falls_back_to_loopback_with_flag() {
        use std::net::IpAddr;
        // Misconfig case: only loopback resolves. Return it so the
        // caller still has SOMETHING, but flip the flag so the caller
        // warns.
        let ips: Vec<IpAddr> = vec!["127.0.1.1".parse().unwrap(), "::1".parse().unwrap()];
        let (ip, only_loopback) = select_preferred_ip(ips);
        assert_eq!(ip.as_deref(), Some("127.0.1.1"));
        assert!(only_loopback);
    }

    #[test]
    fn select_preferred_ip_empty_iterator() {
        let (ip, only_loopback) = select_preferred_ip(std::iter::empty());
        assert!(ip.is_none());
        assert!(!only_loopback);
    }

    #[test]
    fn select_preferred_ip_skips_ipv6_loopback() {
        use std::net::IpAddr;
        // IPv6 loopback (::1) must be skipped just like 127.x.
        let ips: Vec<IpAddr> = vec!["::1".parse().unwrap(), "10.0.0.5".parse().unwrap()];
        let (ip, only_loopback) = select_preferred_ip(ips);
        assert_eq!(ip.as_deref(), Some("10.0.0.5"));
        assert!(!only_loopback);
    }

    #[test]
    fn prepare_cluster_env_validates_cluster() {
        let _guard = env_lock();
        // Empty controller.host → validate() fails → prepare_cluster_env errors.
        let cluster = ClusterConfig {
            controller: crate::config::ClusterController {
                host: String::new(),
                port: 1337,
                path: String::new(),
                docker: None,
                arch: None,
                data_path: None,
                join: None,
            },
            workers: Vec::new(),
            env: std::collections::BTreeMap::new(),
            gpu_ram_share: None,
        };
        let err = prepare_cluster_env(&cluster, None, "train", Path::new("/tmp")).unwrap_err();
        assert!(err.contains("controller.host"), "got: {err}");
    }

    /// Workers reached via `~/.ssh/config` aliases (or any setup where
    /// the YAML's `host:` is just a label, not a DNS-resolvable name)
    /// carry an explicit `ssh.target` in the worker block. In that case
    /// the controller's NSS does not need to know the `host:` value —
    /// the connection uses `ssh.target` — so the "did not resolve"
    /// warning is noise and gets suppressed.
    ///
    /// Uses `nonexistent.invalid.` (RFC 2606 reserved) to guarantee
    /// `getaddrinfo` returns Err on any system regardless of local DNS
    /// or /etc/hosts content.
    #[test]
    fn resolve_cluster_extra_hosts_suppresses_warning_when_ssh_target_explicit() {
        use crate::config::{ClusterController, ClusterWorker, LocalDevices, SshConfig};
        let cluster = ClusterConfig {
            controller: ClusterController {
                host: "127.0.0.1".into(),
                port: 1337,
                path: "/tmp".into(),
                docker: None,
                arch: None,
                data_path: None,
                join: None,
            },
            workers: vec![ClusterWorker {
                host: "nonexistent.invalid.".into(),
                ranks: vec![0],
                local_devices: LocalDevices::Explicit(vec![0]),
                nccl_socket_ifname: "lo".into(),
                path: "/tmp".into(),
                ssh: Some(SshConfig {
                    target: Some("127.0.0.1".into()),
                    ..SshConfig::default()
                }),
                tunnel: false,
                arch: None,
                data_path: None,
                gpu_ram_share: None,
                docker: None,
                env: std::collections::BTreeMap::new(),
            }],
            env: std::collections::BTreeMap::new(),
            gpu_ram_share: None,
        };
        let (_hosts, warnings) = resolve_cluster_extra_hosts(&cluster);
        assert!(
            warnings.is_empty(),
            "explicit ssh.target should suppress the resolution warning, \
             got warnings: {warnings:?}"
        );
    }

    /// Mirror of the suppression test: without `ssh.target` the warning
    /// must still fire (so legitimate misconfigurations stay visible).
    #[test]
    fn resolve_cluster_extra_hosts_warns_when_ssh_target_absent() {
        use crate::config::{ClusterController, ClusterWorker, LocalDevices};
        let cluster = ClusterConfig {
            controller: ClusterController {
                host: "127.0.0.1".into(),
                port: 1337,
                path: "/tmp".into(),
                docker: None,
                arch: None,
                data_path: None,
                join: None,
            },
            workers: vec![ClusterWorker {
                host: "nonexistent.invalid.".into(),
                ranks: vec![0],
                local_devices: LocalDevices::Explicit(vec![0]),
                nccl_socket_ifname: "lo".into(),
                path: "/tmp".into(),
                ssh: None,
                tunnel: false,
                arch: None,
                data_path: None,
                gpu_ram_share: None,
                docker: None,
                env: std::collections::BTreeMap::new(),
            }],
            env: std::collections::BTreeMap::new(),
            gpu_ram_share: None,
        };
        let (_hosts, warnings) = resolve_cluster_extra_hosts(&cluster);
        assert!(
            warnings.iter().any(|w| w.contains("did not resolve")),
            "missing ssh.target should keep the warning, got warnings: {warnings:?}"
        );
    }

    /// A farm overlay declares a join window; a command that is not a
    /// cluster command will not open it and trains here instead. The run
    /// looks entirely normal, which is why this needs saying out loud.
    #[test]
    fn a_declared_join_window_that_no_command_opens_is_called_out() {
        let with_window = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 1337
    path: /opt/flodl
    join:
      discovery: true
      token: aaaabbbbccccddddaaaabbbbccccdddd
  workers: []
";
        let project: ProjectConfig = serde_yaml_ng::from_str(with_window).unwrap();
        let hint = unused_join_window_hint(&project, "train").expect("a window nobody opens");
        assert!(hint.contains("train"), "names the command: {hint}");
        assert!(hint.contains("cluster:"), "names the fix: {hint}");

        // A roster-style cluster block without a discovery window has
        // nothing to miss: fan-out is the command's own business.
        let no_window = "\
cluster:
  controller:
    host: 127.0.0.1
    port: 1337
    path: /opt/flodl
  workers: []
";
        let project: ProjectConfig = serde_yaml_ng::from_str(no_window).unwrap();
        assert!(unused_join_window_hint(&project, "train").is_none());
    }
}
