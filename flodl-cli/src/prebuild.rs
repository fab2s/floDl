//! Pre-flight build for cluster commands.
//!
//! Heterogeneous-rig pain: source lives on a shared mount (NFS /
//! virtiofs / S3-FUSE) so editing on the controller is visible to
//! remote hosts, but each host needs its own libtorch-linked binary
//! (cu128 for a Blackwell host, cu126-pt27 for a Pascal host). Without
//! pre-flight, the first `fdl @cluster <cmd>` after an edit hits stale
//! remote binaries — confusing runtime errors or worse, silent wrong
//! behaviour.
//!
//! This module runs `cargo build` LOCALLY on the controller, once per
//! remote host, into per-`(host, arch)` `target/cluster/<host>/<arch>/`
//! directories with the right libtorch bind-mounted. The shared mount
//! delivers the resulting binary to the remote, which execs it directly
//! (no cargo, no rust toolchain on remote). Per-`(host, arch)` target
//! dirs isolate cargo's fingerprint cache so a libtorch swap on one host
//! doesn't invalidate anyone else's incremental build -- and so changing
//! a host's own `arch:` builds fresh rather than reusing a binary linked
//! against the old libtorch (in docker mode the container's
//! `LIBTORCH_PATH` is a fixed mount point, invisible to cargo's cache).
//!
//! Convention: command name == binary name. `fdl @cluster ddp-bench`
//! builds `--bin ddp-bench`. Features derive from the host's libtorch
//! `.arch` (cuda=12.x → `--features cuda`, cuda=none → no features).
//!
//! Builds run in parallel across hosts (per-host target dirs ⇒ zero
//! contention). All builds run to completion — there is no early abort:
//! every host is joined and every failure is collected, then surfaced
//! together so the user sees every host's diagnostic in one place.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};

use crate::cluster::apply_worker_ssh_opts;
use crate::config::{ClusterConfig, ClusterWorker};

/// Env var carrying the per-host pre-flight build envelope (a JSON
/// map) from fdl-cli's prebuild phase to flodl's launcher. The
/// launcher reads it on the controller side just before fan-out and
/// substitutes the direct-binary form for each host whose entry is
/// present.
///
/// Map shape: `{ "<host-name>": { "bin": "<relative path under
/// worker.path>", "ld_library_path": "<absolute LD_LIBRARY_PATH>" }, ...
/// }`. Hosts absent from the map fall back to the launcher's existing
/// `fdl <cmd>` re-entry on the remote.
pub const ENV_PREBUILD_PER_HOST: &str = "FLODL_INTERNAL_PREBUILD_PER_HOST";

/// Pre-fan-out readiness probe for every host, run BEFORE any build and
/// in BOTH prebuild and `--no-prebuild` modes (mount-readiness is
/// orthogonal to binary freshness). One ssh per remote host: always
/// verifies the shared `data_path` is mounted + readable; when
/// `prebuilding`, also runs the controller-vs-remote ABI gate (arch /
/// libc / `pkill`). The controller is checked LOCALLY (no ssh — it is
/// the build/dispatch host, so ABI is trivially satisfied).
///
/// A definitive failure aborts before fan-out with a per-host message:
/// an ABI mismatch, or a missing EXPLICIT `data_path` (a declared shared
/// mount that isn't there is a launch-breaking misconfig, matching
/// `fdl probe`'s stance). Missing convention-default paths and
/// unreachable hosts only warn. Running before the (multi-minute) builds
/// means a bad host fails fast without wasting a build on a good one.
///
/// `controller_host` is the local hostname (its worker entry, if any, is
/// covered by the local check and skipped from the ssh sweep).
pub fn preflight_hosts(
    cluster: &ClusterConfig,
    controller_host: &str,
    prebuilding: bool,
) -> Result<(), String> {
    // Controller: local shared-mount check (no ssh). Uses the controller
    // block's `data_path`; a same-host worker entry is skipped below.
    {
        let dp = cluster.controller.effective_data_path().to_string();
        let explicit = cluster.controller.data_path.is_some();
        let dir_ok = Path::new(&dp).is_dir();
        let read_ok = dir_ok && std::fs::read_dir(&dp).is_ok();
        if let Some(w) =
            check_remote_data_path(controller_host, &dp, explicit, dir_ok, read_ok)?
        {
            eprintln!("fdl: {w}");
        }
    }

    // Remote workers: one ssh each, in parallel (probe latency stays flat
    // regardless of host count).
    let remotes: Vec<ClusterWorker> = cluster
        .workers
        .iter()
        .filter(|w| w.host != controller_host)
        .cloned()
        .collect();
    if remotes.is_empty() {
        return Ok(());
    }

    let warnings: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::with_capacity(remotes.len());
    for worker in remotes {
        let warnings = Arc::clone(&warnings);
        let errors = Arc::clone(&errors);
        handles.push(thread::spawn(move || {
            match preflight_one_host(&worker, prebuilding) {
                Ok(ws) => warnings.lock().unwrap().extend(ws),
                Err(e) => errors.lock().unwrap().push(e),
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    for w in warnings.lock().unwrap().iter() {
        eprintln!("fdl: {w}");
    }
    let errs = Arc::try_unwrap(errors)
        .map_err(|_| "internal: preflight error collector still referenced".to_string())?
        .into_inner()
        .map_err(|e| format!("internal: preflight errors mutex poisoned: {e}"))?;
    if !errs.is_empty() {
        return Err(format!(
            "pre-flight host check failed on {} host(s):\n  {}",
            errs.len(),
            errs.join("\n  "),
        ));
    }
    Ok(())
}

/// Run pre-flight builds for every remote host in `cluster`. The
/// controller itself is skipped — its build is handled by the normal
/// dispatch path (`cargo run` in Docker against the local `.active`).
///
/// `cmd_name` is both the fdl command and the cargo `--bin` target.
/// `controller_host` is the local hostname (skipped from the remotes).
///
/// Each per-host build runs in a Docker compose service (`cuda` when
/// the host's libtorch advertises a CUDA version in `.arch`, `dev`
/// otherwise). The build's env is overridden so:
///   - `LIBTORCH_HOST_PATH` points at the resolved host libtorch dir
///   - `CARGO_TARGET_DIR` points at `target/cluster/<host>/<arch>/`
///
/// Returns `Ok(())` on universal success. Returns `Err(combined_msg)`
/// listing every host that failed (with its stderr tail) on any
/// failure. Builds running when a failure surfaces complete to natural
/// stopping — cargo's per-crate granularity means cancelling mid-build
/// would leave the per-host target dir in a half-baked state.
pub fn prebuild_remotes(
    project_root: &Path,
    cmd_cwd: &Path,
    cluster: &ClusterConfig,
    cmd_name: &str,
    controller_host: &str,
) -> Result<(), String> {
    // Skip only the worker whose `host` LABEL equals the controller's local
    // hostname — that entry is the controller building locally and sharing its
    // cargo target dir, so it needs no remote step. This compares the logical
    // host label, NOT resolved IPs, deliberately: a container or VM on the same
    // physical machine (e.g. an `ssh.target: 127.0.0.1:2222` worker) is a
    // DISTINCT build target with its own arch/libtorch and must get its own
    // per-host build. Canonicalizing by IP would wrongly fold such a worker
    // into the controller and break heterogeneous same-box rigs.
    let remotes: Vec<&ClusterWorker> = cluster
        .workers
        .iter()
        .filter(|w| w.host != controller_host)
        .collect();
    if remotes.is_empty() {
        return Ok(());
    }

    // Whether the controller runs builds inside Docker. Sourced from
    // the `controller.docker:` field in cluster.yml; `None` when
    // absent (native-Rust controllers get the bare cargo invocation).
    let controller_docker_svc: Option<String> = cluster.controller.docker.clone();

    eprintln!(
        "fdl: pre-flight build for {} remote worker(s): {}",
        remotes.len(),
        remotes.iter().map(|w| w.host.as_str()).collect::<Vec<_>>().join(", "),
    );

    // Controller's view of the shared project root (required field
    // per validator).
    let controller_path: std::path::PathBuf =
        std::path::PathBuf::from(&cluster.controller.path);

    let project_root = Arc::new(project_root.to_path_buf());
    let cmd_cwd = Arc::new(cmd_cwd.to_path_buf());
    let cmd_name = Arc::new(cmd_name.to_string());
    let controller_path = Arc::new(controller_path);
    let controller_docker_svc = Arc::new(controller_docker_svc);
    let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let envelope: Arc<Mutex<BTreeMap<String, PerHostEnvelope>>> =
        Arc::new(Mutex::new(BTreeMap::new()));

    let mut handles = Vec::with_capacity(remotes.len());
    for worker in remotes {
        let worker = worker.clone();
        let project_root = Arc::clone(&project_root);
        let cmd_cwd = Arc::clone(&cmd_cwd);
        let cmd_name = Arc::clone(&cmd_name);
        let controller_path = Arc::clone(&controller_path);
        let controller_docker_svc = Arc::clone(&controller_docker_svc);
        let errors = Arc::clone(&errors);
        let envelope = Arc::clone(&envelope);
        handles.push(thread::spawn(move || {
            match prebuild_one_worker(
                &project_root, &cmd_cwd, &controller_path,
                &worker, &cmd_name,
                controller_docker_svc.as_deref(),
            ) {
                Ok(env_entry) => {
                    eprintln!("fdl: pre-flight OK ({})", worker.host);
                    envelope.lock().unwrap().insert(worker.host.clone(), env_entry);
                }
                Err(e) => {
                    eprintln!("fdl: pre-flight FAILED ({}): {}", worker.host, e);
                    errors.lock().unwrap().push(format!("{}: {}", worker.host, e));
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    let errs = Arc::try_unwrap(errors)
        .map_err(|_| "internal: error collector still has outstanding refs".to_string())?
        .into_inner()
        .map_err(|e| format!("internal: errors mutex poisoned: {e}"))?;
    if !errs.is_empty() {
        return Err(format!(
            "pre-flight build failed on {} host(s):\n  {}",
            errs.len(),
            errs.join("\n  "),
        ));
    }

    // Emit the per-host envelope so the flodl launcher's remote
    // dispatch can substitute the direct-binary form for each host
    // (skipping the `fdl <cmd>` re-entry — no cargo on remote).
    let env_map = Arc::try_unwrap(envelope)
        .map_err(|_| "internal: envelope still has outstanding refs".to_string())?
        .into_inner()
        .map_err(|e| format!("internal: envelope mutex poisoned: {e}"))?;
    let json = serde_json::to_string(&env_map)
        .map_err(|e| format!("internal: serialize prebuild envelope: {e}"))?;
    // SAFETY: main has not spawned threads at this point in dispatch.
    unsafe { std::env::set_var(ENV_PREBUILD_PER_HOST, json); }
    Ok(())
}

/// Per-host pre-flight build artifact descriptor — exactly what the
/// launcher needs to substitute the remote dispatch with a direct
/// binary exec. Mirrors `flodl::distributed::launcher::PerHostPrebuild`
/// on the consumer side; the two structs share an on-the-wire JSON
/// schema but are independent types because the crates can't share
/// declarations without a circular dep.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerHostEnvelope {
    /// Path to the compiled binary, relative to the host's project
    /// checkout (`worker.path`). e.g.
    /// `target/cluster/node-b/precompiled-cu128/release/train`.
    pub bin: String,
    /// Absolute path the launcher should set as `LD_LIBRARY_PATH` so
    /// the binary finds its libtorch at runtime. e.g.
    /// `/home/me/rdl/libtorch/builds/sm61-sm120/lib`. The launcher may
    /// append host-specific extras (e.g. `:/usr/local/lib` for bare-
    /// metal libnccl) via `worker.env: { LD_LIBRARY_PATH: ... }`.
    pub ld_library_path: String,
    /// Subdirectory under the host's project checkout to `cd` into
    /// before exec — the relative offset of the command's filesystem
    /// cwd from `project_root`. Mirrors the cwd the controller-side
    /// build used (e.g. `ddp-bench` for `fdl ddp-bench`). Empty string
    /// means execute from `worker.path` directly. Relative-path defaults
    /// the binary expects (e.g. `--data-dir data`, `--output runs/`)
    /// only resolve correctly when the remote cwd matches.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cwd_subpath: String,
}

/// Build `cmd_name` for one worker. Picks docker service + cargo
/// features from the host's libtorch metadata. Returns a
/// [`PerHostEnvelope`] describing where the resulting binary lives
/// (so the launcher can substitute it on the remote-dispatch path)
/// and what `LD_LIBRARY_PATH` the remote should set.
///
/// `controller_path` is the controller's view of the shared project
/// root. The libtorch convention says the variant lives at
/// `<controller_path>/libtorch/<worker.arch>` for the build (controller
/// view) and `<worker.path>/libtorch/<worker.arch>` for the runtime
/// (remote view). Both point at the same physical libtorch via the
/// shared mount; the two paths differ only when controller and remote
/// see the project at different filesystem locations.
/// Outcome of the ABI-compatibility check between the controller's
/// build environment and a remote host.
#[derive(Debug, PartialEq)]
enum AbiCheck {
    /// Compatible (arch matches, remote is glibc). `warning` is `Some`
    /// for a soft glibc-version skew note the caller should surface.
    Ok { warning: Option<String> },
    /// Definitively incompatible — the prebuilt binary cannot exec on
    /// the remote. Hard error before fan-out.
    Incompatible(String),
}

/// Compare the controller build environment against a remote host's
/// `uname -m` + `ldd --version` output. Pure — no I/O — so it is
/// directly unit-tested; [`preflight_one_host`] supplies the live inputs.
///
/// - **arch** must match exactly: an x86-64 binary is `Exec format
///   error` on aarch64. Hard error.
/// - **libc flavor**: the flodl build images are glibc-based, so a musl
///   (Alpine) remote cannot run the glibc-linked binary. Hard error.
/// - **glibc version**: a soft warning only. It often works, fails only
///   when the remote glibc is OLDER than the build env's (`GLIBC_2.XX
///   not found`), and that error at least names glibc — far less
///   cryptic than the two hard cases. We do not parse/order versions
///   here (that needs the build-container glibc too); we just flag when
///   the remote's reported glibc line is worth the operator's eye.
fn check_remote_abi(
    host: &str,
    controller_arch: &str,
    remote_uname_m: &str,
    remote_ldd: &str,
) -> AbiCheck {
    let remote_arch = remote_uname_m.trim();
    if remote_arch.is_empty() {
        // Couldn't read arch — treat as indeterminate, not a mismatch
        // (see preflight_one_host's unreachable-is-a-warning discipline).
        return AbiCheck::Ok {
            warning: Some(format!(
                "host {host:?}: could not read remote `uname -m`; skipping \
                 ABI pre-check (a real mismatch would surface at fan-out)"
            )),
        };
    }
    if remote_arch != controller_arch {
        return AbiCheck::Incompatible(format!(
            "host {host:?}: CPU arch mismatch — the pre-built binary is \
             {controller_arch} (controller build env) but the remote is \
             {remote_arch}. A cross-arch binary cannot exec (`Exec format \
             error`). Run same-arch hosts, or build per-arch."
        ));
    }
    let ldd_lc = remote_ldd.to_ascii_lowercase();
    if ldd_lc.contains("musl") {
        return AbiCheck::Incompatible(format!(
            "host {host:?}: remote uses musl libc, but the pre-built binary \
             is glibc-linked (flodl build images are glibc-based) and cannot \
             run there. Match the libc (glibc remote), or build on the remote."
        ));
    }
    // Soft glibc note: surface the remote's reported version line when we
    // could read one, so a later `GLIBC_… not found` is unsurprising. No
    // hard gate — build on the OLDEST-glibc host to be safe.
    let looks_glibc = ldd_lc.contains("glibc") || ldd_lc.contains("gnu libc");
    let warning = remote_ldd
        .lines()
        .next()
        .map(str::trim)
        .filter(|l| !l.is_empty() && looks_glibc)
        .map(|l| {
            format!(
                "host {host:?}: remote glibc reports `{l}`. If the run later \
                 fails with `GLIBC_… not found`, the remote glibc is older \
                 than the controller build env — build on the oldest-glibc \
                 host."
            )
        });
    AbiCheck::Ok { warning }
}

/// Evaluate a host's shared-data-path readiness from probe results.
/// `dir_ok` = the path exists and is a directory; `read_ok` = it is
/// readable/listable. `explicit` = the host (or cluster.yml) declared
/// `data_path:` rather than falling back to the convention default.
///
/// Pure — no I/O — so it is directly unit-tested; the local (controller)
/// and remote (ssh) paths in [`preflight_hosts`] / [`preflight_one_host`]
/// supply `dir_ok` / `read_ok`. Mirrors `probe::check_data_path`'s
/// policy: a missing EXPLICIT path is launch-breaking (`Err`); a missing
/// CONVENTION-default is a warning (users without shared storage are
/// fine); a present-but-unreadable path is always an error.
fn check_remote_data_path(
    host: &str,
    path: &str,
    explicit: bool,
    dir_ok: bool,
    read_ok: bool,
) -> Result<Option<String>, String> {
    if !dir_ok {
        if explicit {
            return Err(format!(
                "host {host:?}: shared data path `{path}` does not exist (or is \
                 not a directory). flodl assumes a shared filesystem (NAS / SMB \
                 / virtiofs / SSHFS) mounted at the same logical path on every \
                 node — training reads data and writes checkpoints there. Mount \
                 it, or correct `data_path:` in cluster.yml."
            ));
        }
        return Ok(Some(format!(
            "host {host:?}: convention shared-data path `{path}` not present (no \
             `data_path:` declared). Ignore if you don't use shared storage; \
             otherwise set `data_path:` per host or mount `{path}`."
        )));
    }
    if !read_ok {
        return Err(format!(
            "host {host:?}: shared data path `{path}` exists but is not readable \
             by the remote user. Check mount permissions / uid mapping."
        ));
    }
    Ok(None)
}

/// One remote host's pre-fan-out readiness probe over a single ssh.
///
/// ALWAYS checks the shared `data_path` is mounted + readable. When
/// `prebuilding`, ALSO gathers `uname -m` + `ldd --version` + `pkill`
/// availability and runs the [`check_remote_abi`] gate — the ABI check
/// only matters when a controller-built binary is shipped to the remote
/// (the prebuild path); under `--no-prebuild` the remote re-enters
/// `fdl <cmd>` and builds/runs natively, so arch/libc can't mismatch.
/// Returns collected warnings on success, `Err(msg)` on a definitive
/// failure (ABI mismatch, or a missing EXPLICIT data_path).
///
/// UNREACHABLE-IS-A-WARNING: if the probe ssh itself fails (blip, host
/// down), we return warnings and proceed — a transient probe failure
/// must never make things worse than the status quo; the real fan-out
/// ssh surfaces a genuine connectivity error anyway. The probe only ever
/// ADDS loud errors for definitive mismatches / missing declared mounts.
fn preflight_one_host(worker: &ClusterWorker, prebuilding: bool) -> Result<Vec<String>, String> {
    let target = worker
        .ssh
        .as_ref()
        .and_then(|s| s.target.as_deref())
        .unwrap_or(&worker.host);
    let dp = worker.effective_data_path().to_string();
    let dp_explicit = worker.data_path.is_some();

    // One round-trip. data_path tests always; the ABI block (uname / ldd
    // banner — its version line lands on stderr for glibc, stdout for
    // some, so capture both — plus a `pkill` availability probe) only
    // when prebuilding. All sentinel-delimited.
    let mut script = format!(
        "if [ -d {q} ]; then echo __FLODL_DP_DIR__=1; else echo __FLODL_DP_DIR__=0; fi; \
         if [ -r {q} ]; then echo __FLODL_DP_READ__=1; else echo __FLODL_DP_READ__=0; fi",
        q = posix_quote(&dp),
    );
    if prebuilding {
        script.push_str(
            "; echo __FLODL_ABI__; uname -m; echo __FLODL_LDD__; \
             ldd --version 2>&1 | head -1; echo __FLODL_PKILL__; \
             command -v pkill >/dev/null 2>&1 && echo present || echo absent",
        );
    }

    let mut cmd = Command::new("ssh");
    // User ssh.options first (they win), then flodl's defaults (M17).
    apply_worker_ssh_opts(&mut cmd, worker);
    cmd.args(["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]);
    cmd.arg(target);
    cmd.arg(&script);
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            return Ok(vec![format!(
                "host {:?}: remote pre-check ssh spawn failed ({e}); skipping \
                 (a real mismatch / missing mount would surface at fan-out)",
                worker.host,
            )]);
        }
    };
    if !output.status.success() {
        return Ok(vec![format!(
            "host {:?}: remote pre-check ssh exited {}; skipping (a real \
             mismatch / missing mount would surface at fan-out)",
            worker.host, output.status,
        )]);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    let mut warnings = Vec::new();
    // data_path verdict (always). Exact-sentinel match: `=0` never
    // matches `=1`, so a plain `contains` is unambiguous.
    let dir_ok = stdout.contains("__FLODL_DP_DIR__=1");
    let read_ok = stdout.contains("__FLODL_DP_READ__=1");
    if let Some(w) = check_remote_data_path(&worker.host, &dp, dp_explicit, dir_ok, read_ok)? {
        warnings.push(w);
    }

    // ABI verdict (prebuild path only).
    if prebuilding {
        let abi_part = stdout
            .split_once("__FLODL_ABI__")
            .map(|(_, r)| r)
            .unwrap_or("");
        let (uname_m, rest) = abi_part
            .split_once("__FLODL_LDD__")
            .unwrap_or((abi_part, ""));
        let (ldd, pkill_field) = rest
            .split_once("__FLODL_PKILL__")
            .unwrap_or((rest, ""));
        let controller_arch = std::env::consts::ARCH;
        match check_remote_abi(&worker.host, controller_arch, uname_m, ldd) {
            AbiCheck::Ok { warning } => warnings.extend(warning),
            AbiCheck::Incompatible(msg) => return Err(msg),
        }
        // Only warn on an EXPLICIT "absent" — an empty/garbled field
        // (older remote, parse hiccup) is indeterminate and must not
        // false-warn, mirroring the empty-arch discipline in
        // `check_remote_abi`.
        if pkill_field.trim() == "absent" {
            warnings.push(format!(
                "host {:?}: `pkill` not found on the remote — belt-and-braces \
                 orphan cleanup is disabled; teardown relies solely on the \
                 per-host trap wrapper (kills by known pid, no external tool). \
                 Install procps if you want pre-spawn stale-orphan clearing.",
                worker.host,
            ));
        }
    }
    Ok(warnings)
}

/// Collapse a libtorch `arch:` subpath (`precompiled/cu128`,
/// `builds/sm61-sm120`) into a single path segment for use in the
/// per-`(host, arch)` cargo target dir. Only the path separators need
/// folding (`/` and, defensively, `\`); everything else in a libtorch
/// variant name is already a safe path atom.
fn arch_slug(arch: &str) -> String {
    arch.chars()
        .map(|c| if c == '/' || c == '\\' { '-' } else { c })
        .collect()
}

fn prebuild_one_worker(
    project_root: &Path,
    cmd_cwd: &Path,
    controller_path: &Path,
    worker: &ClusterWorker,
    cmd_name: &str,
    controller_docker_svc: Option<&str>,
) -> Result<PerHostEnvelope, String> {
    let arch = worker.arch.as_ref().ok_or_else(|| {
        format!(
            "host {:?} has no `arch:` set in cluster.yml — \
             pre-flight build needs the libtorch variant subpath \
             (e.g. `arch: precompiled/cu128` or `arch: builds/sm61-sm120`)",
            worker.host,
        )
    })?;
    // Controller-side libtorch variant dir, resolved via convention.
    let controller_variant_dir = controller_path.join("libtorch").join(arch);
    if !controller_variant_dir.join("lib").is_dir() {
        return Err(format!(
            "host {:?}: controller-side libtorch at `{}` (resolved from \
             `<controller_path>/libtorch/<arch>`) does not look like a \
             valid libtorch install (missing `lib/`?)",
            worker.host,
            controller_variant_dir.display(),
        ));
    }
    let host_path = controller_variant_dir.display().to_string();
    // ABI + shared-mount pre-checks ran earlier in `preflight_hosts`
    // (one ssh per host, before any build), so by the time we reach the
    // multi-minute build the host is already known compatible + mounted.
    // Derive features + docker service from the YAML-declared `arch:`
    // basename — single source of truth, no `.arch` metadata file
    // required. `cpu` is the only non-CUDA variant by convention; every
    // other basename (`cuNN`, `sm<NN>-sm<NN>`, etc.) is a GPU build.
    // `arch:` basename → cargo --features (`cuda` for GPU variants, none
    // for `cpu`). The compose SERVICE is `controller.docker` (above),
    // not arch-derived — the controller's toolchain builds every host.
    let (features_arg, _) = features_and_service_from_arch(arch);
    let cuda_version_for_image = cuda_version_from_arch(arch);
    // Key the target dir by host AND arch. Host alone is not enough: the
    // `arch:` field selects the libtorch variant, but in docker mode that
    // variant is bind-mounted onto the FIXED container path
    // (`/usr/local/libtorch`, `Dockerfile` ENV LIBTORCH_PATH), so
    // `LIBTORCH_PATH` inside the container never changes when `arch:`
    // does. Change a host's `arch:` to a different torch/CUDA build in the
    // same feature category (e.g. `precompiled/cu118` -> `precompiled/cu128`,
    // both `--features cuda`) and nothing cargo fingerprints changes:
    // cargo would reuse the stale binary linked against the OLD libtorch,
    // which then execs at fan-out with the NEW `LD_LIBRARY_PATH` -> ABI
    // mismatch (`undefined symbol`) or silently-wrong kernels. An env-only
    // nudge can't fix it: the mount path is fixed, so build.rs re-emits the
    // same `-L` string and cargo never relinks. A distinct target dir per
    // (host, arch) is the only thing that forces the rebuild. `arch` is a
    // libtorch subpath (`precompiled/cu128`, `builds/sm61-sm120`); slugify
    // its `/` so it is a single path segment.
    let target_dir_relative =
        format!("target/cluster/{}/{}", worker.host, arch_slug(arch));

    // Two execution modes — docker-backed (controller has `docker:`
    // set in cluster.yml) or native cargo on the host filesystem.
    //
    // Docker mode: the project root mounts at `/workspace`; cwd +
    // CARGO_TARGET_DIR are in the `/workspace/...` namespace. Docker
    // mode is gated on `controller.docker` being set, and that value
    // IS the build service — the controller owns one build toolchain
    // and compiles every host's binary in it (per-host libtorch comes
    // from LIBTORCH_HOST_PATH, per-host features from `arch:`). The
    // service must be a superset toolchain (`cuda` builds both CUDA and
    // CPU binaries); a service that lacks the CUDA toolkit fails loudly
    // at cargo build for a CUDA-arch worker, naming the chosen service.
    //
    // Native mode: cwd is the cmd's filesystem cwd, CARGO_TARGET_DIR
    // is the same project-root-relative path on the host, and
    // LIBTORCH_PATH is set directly on the cargo process (no Docker
    // bind-mount indirection).
    let (sh_cmd, cwd_for_spawn, extra_envs): (String, &Path, Vec<(&str, String)>) =
        if let Some(docker_svc) = controller_docker_svc {
            // Docker-backed build.
            let target_dir_in_container = format!("/workspace/{target_dir_relative}");
            let sub_path = cmd_cwd
                .strip_prefix(project_root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let cwd_in_container = if sub_path.is_empty() {
                "/workspace".to_string()
            } else {
                format!("/workspace/{sub_path}")
            };
            let build_cmd = if features_arg.is_empty() {
                format!(
                    "cd {cwd} && CARGO_TARGET_DIR={tgt} cargo build --release --bin {bin}",
                    cwd = posix_quote(&cwd_in_container),
                    tgt = posix_quote(&target_dir_in_container),
                    bin = posix_quote(cmd_name),
                )
            } else {
                format!(
                    "cd {cwd} && CARGO_TARGET_DIR={tgt} cargo build --release --features {feat} --bin {bin}",
                    cwd = posix_quote(&cwd_in_container),
                    tgt = posix_quote(&target_dir_in_container),
                    feat = posix_quote(features_arg),
                    bin = posix_quote(cmd_name),
                )
            };
            let docker_cmd = format!(
                "docker compose run --rm {svc} bash -c {inner}",
                svc = docker_svc,
                inner = posix_quote(&build_cmd),
            );
            (docker_cmd, project_root, vec![
                ("LIBTORCH_HOST_PATH", host_path.clone()),
                ("LIBTORCH_CPU_PATH", "./libtorch/precompiled/cpu".into()),
            ])
        } else {
            // Native build (no docker on controller).
            let target_dir_abs = project_root.join(&target_dir_relative);
            let bash_cmd = if features_arg.is_empty() {
                format!(
                    "cargo build --release --bin {bin}",
                    bin = posix_quote(cmd_name),
                )
            } else {
                format!(
                    "cargo build --release --features {feat} --bin {bin}",
                    feat = posix_quote(features_arg),
                    bin = posix_quote(cmd_name),
                )
            };
            (bash_cmd, cmd_cwd, vec![
                ("LIBTORCH_PATH", host_path.clone()),
                (
                    "CARGO_TARGET_DIR",
                    target_dir_abs.to_string_lossy().into_owned(),
                ),
            ])
        };

    let mut cmd = Command::new("sh");
    cmd.args(["-c", &sh_cmd])
        .current_dir(cwd_for_spawn)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::null());
    for (k, v) in &extra_envs {
        cmd.env(k, v);
    }

    if let Some(cuda_version) = &cuda_version_for_image {
        let normalised = if cuda_version.matches('.').count() < 2 {
            format!("{cuda_version}.0")
        } else {
            cuda_version.clone()
        };
        let cuda_tag = normalised
            .splitn(3, '.')
            .take(2)
            .collect::<Vec<_>>()
            .join(".");
        cmd.env("CUDA_VERSION", &normalised);
        cmd.env("CUDA_TAG", &cuda_tag);
    }

    let status = cmd
        .status()
        .map_err(|e| format!("spawn `{sh_cmd}`: {e}"))?;
    if !status.success() {
        return Err(format!(
            "cargo build exited {} (libtorch={host_path}, target={target_dir_relative}, \
             features={feat})",
            status.code().unwrap_or(-1),
            feat = if features_arg.is_empty() { "(none)" } else { features_arg },
        ));
    }
    // Runtime LD_LIBRARY_PATH uses the REMOTE-side view: the rank
    // exec's the binary on the remote, where libtorch is at
    // `<worker.path>/libtorch/<arch>/lib` per the convention.
    let runtime_lib = format!(
        "{path}/libtorch/{arch}/lib",
        path = worker.path.trim_end_matches('/'),
    );
    let _ = host_path; // controller-side path used only for the build above
    // cwd_subpath: the cmd's filesystem cwd relative to project_root.
    // For `fdl ddp-bench` invoked from the repo, cmd_cwd is
    // `<repo>/ddp-bench`, so subpath is `ddp-bench`. The remote
    // launcher uses this to cd into the matching subdir before exec.
    let cwd_subpath = cmd_cwd
        .strip_prefix(project_root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(PerHostEnvelope {
        bin: format!("{target_dir_relative}/release/{cmd_name}"),
        ld_library_path: runtime_lib,
        cwd_subpath,
    })
}

/// Pick cargo features + docker compose service from the host's
/// libtorch `.arch` metadata. `cuda=12.x` → (`cuda`, `cuda`); anything
/// else → (`""`, `dev`).
/// Derive `(cargo --features arg, docker-compose service name)` from
/// the YAML `arch:` path basename. The yml `arch:` IS the single
/// source of truth (no `.arch` metadata file required) — `cpu` is the
/// only non-CUDA convention; everything else is a GPU variant.
fn features_and_service_from_arch(arch: &str) -> (&'static str, &'static str) {
    let basename = std::path::Path::new(arch)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if basename == "cpu" {
        ("", "dev")
    } else {
        ("cuda", "cuda")
    }
}

/// Extract a CUDA major.minor string from a `precompiled/cuNN` arch
/// path basename (e.g. `cu128` → `"12.8"`). Returns `None` for source
/// builds (`builds/sm…`) where the arch alone does not encode a CUDA
/// version — the caller falls back to the `CUDA_VERSION` env var (or
/// docker-compose's own default) for the toolkit image tag.
fn cuda_version_from_arch(arch: &str) -> Option<String> {
    let basename = std::path::Path::new(arch)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let rest = basename.strip_prefix("cu")?;
    if rest.len() < 2 || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let major = &rest[..rest.len() - 1];
    let minor = &rest[rest.len() - 1..];
    Some(format!("{major}.{minor}"))
}

/// Single-quote a string for `sh -c` so embedded spaces/quotes don't
/// break the outer shell parse. Local copy to avoid `pub(crate)`
/// promotion of `run::posix_quote`.
fn posix_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = s.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ',' | '=' | ':')
    });
    if safe {
        return s.to_string();
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_arch_mismatch_is_hard_incompatible() {
        let r = check_remote_abi("gv", "x86_64", "aarch64", "ldd (GNU libc) 2.31");
        assert!(matches!(r, AbiCheck::Incompatible(m) if m.contains("arch mismatch")));
    }

    #[test]
    fn data_path_present_and_readable_is_clean() {
        assert!(check_remote_data_path("h", "/flodl/data", true, true, true)
            .expect("ok")
            .is_none());
    }

    #[test]
    fn data_path_missing_explicit_is_hard_error() {
        let err = check_remote_data_path("h", "/mnt/nas", true, false, false)
            .expect_err("explicit missing must be a hard error");
        assert!(err.contains("does not exist"), "err: {err}");
        assert!(err.contains("/mnt/nas"), "err: {err}");
    }

    #[test]
    fn data_path_missing_convention_default_is_warning() {
        // explicit=false -> convention default /flodl/data not present ->
        // warning (Ok), not a launch-breaking error.
        let w = check_remote_data_path("h", "/flodl/data", false, false, false)
            .expect("convention-default missing must be Ok(warning)")
            .expect("should carry a warning");
        assert!(w.contains("convention"), "warn: {w}");
    }

    #[test]
    fn data_path_present_but_unreadable_is_hard_error() {
        let err = check_remote_data_path("h", "/flodl/data", false, true, false)
            .expect_err("present-but-unreadable must be a hard error regardless of explicit");
        assert!(err.contains("not readable"), "err: {err}");
    }

    #[test]
    fn abi_musl_is_hard_incompatible_on_matching_arch() {
        let r = check_remote_abi("alp", "x86_64", "x86_64", "musl libc (x86_64)\nVersion 1.2.4");
        assert!(matches!(r, AbiCheck::Incompatible(m) if m.contains("musl")));
    }

    #[test]
    fn abi_matching_arch_glibc_ok_with_version_note() {
        let r = check_remote_abi(
            "w", "x86_64", "x86_64",
            "ldd (Ubuntu GLIBC 2.35-0ubuntu3.4) 2.35",
        );
        match r {
            AbiCheck::Ok { warning: Some(w) } => {
                assert!(w.contains("2.35"), "warning should quote the reported line: {w}");
            }
            other => panic!("expected Ok+warning, got {other:?}"),
        }
    }

    #[test]
    fn abi_empty_uname_is_indeterminate_not_mismatch() {
        let r = check_remote_abi("w", "x86_64", "", "");
        assert!(matches!(r, AbiCheck::Ok { warning: Some(_) }));
    }

    #[test]
    fn abi_arch_checked_before_musl() {
        let r = check_remote_abi("x", "x86_64", "aarch64", "musl libc");
        assert!(matches!(r, AbiCheck::Incompatible(m) if m.contains("arch mismatch")));
    }

    #[test]
    fn abi_matching_arch_no_ldd_ok_no_warning() {
        let r = check_remote_abi("w", "x86_64", "x86_64", "");
        assert_eq!(r, AbiCheck::Ok { warning: None });
    }

    #[test]
    fn features_and_service_precompiled_cuda_picks_cuda() {
        assert_eq!(
            features_and_service_from_arch("precompiled/cu128"),
            ("cuda", "cuda")
        );
    }

    #[test]
    fn features_and_service_precompiled_cpu_picks_dev() {
        assert_eq!(
            features_and_service_from_arch("precompiled/cpu"),
            ("", "dev")
        );
    }

    #[test]
    fn features_and_service_source_build_picks_cuda() {
        // Source builds under `builds/<gpu-arch>` are CUDA by
        // convention; only `cpu` basename is non-CUDA.
        assert_eq!(
            features_and_service_from_arch("builds/sm61-sm120"),
            ("cuda", "cuda")
        );
        assert_eq!(
            features_and_service_from_arch("builds/sm80"),
            ("cuda", "cuda")
        );
    }

    #[test]
    fn cuda_version_from_arch_extracts_precompiled_version() {
        assert_eq!(cuda_version_from_arch("precompiled/cu128"), Some("12.8".into()));
        assert_eq!(cuda_version_from_arch("precompiled/cu126"), Some("12.6".into()));
        assert_eq!(cuda_version_from_arch("precompiled/cu118"), Some("11.8".into()));
    }

    #[test]
    fn cuda_version_from_arch_none_for_source_builds_and_cpu() {
        assert_eq!(cuda_version_from_arch("builds/sm61-sm120"), None);
        assert_eq!(cuda_version_from_arch("builds/sm80"), None);
        assert_eq!(cuda_version_from_arch("precompiled/cpu"), None);
    }

    #[test]
    fn arch_slug_folds_path_separators_only() {
        // A change of `arch:` must yield a DISTINCT slug so the per-host
        // target dir keys on it — otherwise a docker-mode arch swap reuses
        // a stale binary (M24). Different variants -> different slugs.
        assert_eq!(arch_slug("precompiled/cu128"), "precompiled-cu128");
        assert_eq!(arch_slug("precompiled/cu118"), "precompiled-cu118");
        assert_ne!(arch_slug("precompiled/cu128"), arch_slug("precompiled/cu118"));
        assert_eq!(arch_slug("builds/sm61-sm120"), "builds-sm61-sm120");
        // Single-segment archs pass through unchanged.
        assert_eq!(arch_slug("cpu"), "cpu");
    }

    #[test]
    fn posix_quote_round_trips_safe_strings() {
        assert_eq!(posix_quote("ddp-bench"), "ddp-bench");
        assert_eq!(posix_quote("target/cluster/exa"), "target/cluster/exa");
        assert_eq!(posix_quote(""), "''");
    }

    #[test]
    fn posix_quote_wraps_unsafe_strings() {
        assert_eq!(posix_quote("a b"), "'a b'");
        assert_eq!(posix_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn envelope_serializes_to_stable_json() {
        let mut env = BTreeMap::new();
        env.insert(
            "host-b".to_string(),
            PerHostEnvelope {
                bin: "target/cluster/host-b/release/bench".into(),
                ld_library_path: "/opt/lt-b/lib".into(),
                cwd_subpath: String::new(),
            },
        );
        env.insert(
            "host-a".to_string(),
            PerHostEnvelope {
                bin: "target/cluster/host-a/release/bench".into(),
                ld_library_path: "/opt/lt-a/lib".into(),
                cwd_subpath: String::new(),
            },
        );
        let json = serde_json::to_string(&env).unwrap();
        // BTreeMap iterates in sorted key order ⇒ stable JSON output
        // regardless of insertion order.
        assert_eq!(
            json,
            r#"{"host-a":{"bin":"target/cluster/host-a/release/bench","ld_library_path":"/opt/lt-a/lib"},"host-b":{"bin":"target/cluster/host-b/release/bench","ld_library_path":"/opt/lt-b/lib"}}"#,
        );
    }

    #[test]
    fn envelope_round_trips_through_serde() {
        let mut env = BTreeMap::new();
        env.insert(
            "h1".to_string(),
            PerHostEnvelope {
                bin: "t/c/h1/release/x".into(),
                ld_library_path: "/opt/lt/lib".into(),
                cwd_subpath: "ddp-bench".into(),
            },
        );
        let json = serde_json::to_string(&env).unwrap();
        let back: BTreeMap<String, PerHostEnvelope> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(back.len(), 1);
        let e = back.get("h1").unwrap();
        assert_eq!(e.bin, "t/c/h1/release/x");
        assert_eq!(e.ld_library_path, "/opt/lt/lib");
    }
}
