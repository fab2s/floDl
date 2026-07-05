//! `fdl probe` — check host readiness for training.
//!
//! Single-host (default): probes the local box for GPU + libtorch +
//! shared-data path + NCCL. Cluster context (env overlay): each
//! configured host is probed via SSH and the per-host status is
//! aggregated into one report. See [`run`].
//!
//! Design notes
//! ============
//! flodl assumes shared storage is available to every node at the
//! same logical path (NAS / SMB / virtiofs / S3-FUSE / SSHFS). The
//! probe is the gate that confirms each host can see it BEFORE
//! training fans out, instead of discovering it mid-AllReduce when a
//! checkpoint write hangs on a stale mount. The convention default
//! ([`crate::config::DEFAULT_DATA_PATH`]) applies when a host does
//! not declare `data_path:` in `fdl.cluster.yml`.
//!
//! Probe is intentionally thin — it reuses
//! [`crate::libtorch::detect`] and [`crate::util::system`] for the
//! existing detection logic and adds shared-mount + NCCL discovery
//! on top. The format is `diagnose`-style (text by default, `--json`
//! emits machine-readable output for `fdl deploy` / CI to consume).

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cluster::resolve_local_hostname;
use crate::config::{self, ClusterWorker, DEFAULT_DATA_PATH};
use crate::context::Context;
use crate::libtorch::detect::{self, LibtorchInfo};
use crate::util::system::{self, GpuInfo};

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

/// Run the probe.
///
/// **Single-host** (no active env overlay): probes the local box and
/// emits one report. `--data-path` overrides config; `--skip-mount`
/// short-circuits the shared-data check.
///
/// **Cluster** (`fdl @cluster probe` / `FDL_ENV=cluster`): loads
/// `fdl.<env>.yml`'s `cluster.workers:` list. For each host: if it's
/// the local host, probes in-process; otherwise SSHes to it and runs
/// `<worker.path>/target/release/fdl probe --json` remotely. Per-host
/// JSON is parsed back into [`ProbeReport`] and aggregated.
///
/// Exit code: `0` when every probed host is green; `1` when any
/// host raised issues.
pub fn run(
    json: bool,
    skip_mount: bool,
    data_path_override: Option<PathBuf>,
    libtorch_path_override: Option<PathBuf>,
    via_docker: Option<String>,
) -> i32 {
    let ctx = Context::resolve();
    // Cluster fan-out only applies when no explicit libtorch override
    // is passed — overrides are how the *remote* probe is invoked, so
    // we must NOT recurse back into cluster mode on the remote side.
    if libtorch_path_override.is_none() {
        if let Ok(env_name) = std::env::var("FDL_ENV") {
            if let Some(cluster) = load_cluster_for_env(&ctx, &env_name) {
                return run_cluster(&cluster, json, skip_mount);
            }
        }
    }
    // Single-host (local OR remote-being-probed). When `--data-path` is
    // passed explicitly, treat a missing path as an error; when absent
    // (falling back to DEFAULT_DATA_PATH), treat it as a warning.
    let data_path_explicit = data_path_override.is_some();
    let report = probe_local(
        &ctx,
        skip_mount,
        data_path_override,
        libtorch_path_override,
        via_docker,
        data_path_explicit,
    );
    if json {
        print_json(&report);
    } else {
        print_report(&report);
    }
    if report.green() { 0 } else { 1 }
}

fn load_cluster_for_env(ctx: &Context, env_name: &str) -> Option<config::ClusterConfig> {
    let config_path = config::find_config(&ctx.root)?;
    let project = config::load_project_with_env(&config_path, Some(env_name)).ok()?;
    project.cluster
}

// ---------------------------------------------------------------------------
// Cluster fan-out
// ---------------------------------------------------------------------------

fn run_cluster(cluster: &config::ClusterConfig, json: bool, skip_mount: bool) -> i32 {
    let local = resolve_local_hostname();
    let mut reports: Vec<ProbeReport> = Vec::with_capacity(cluster.workers.len());
    for worker in &cluster.workers {
        let r = if worker.host == local {
            // Local rank: probe in-process, honor the host's data_path,
            // libtorch_path, and docker service (if set in cluster.yml).
            // Matches the remote-probe path so the local rank's report
            // shape is identical to the SSH-probed remotes. Only pass an
            // explicit data_path_override when the host declared one;
            // omitting it preserves the "default = warning, not error"
            // semantics in [`check_data_path`].
            let ctx = Context::resolve();
            let data_path_explicit = worker.data_path.is_some();
            probe_local(
                &ctx,
                skip_mount,
                worker.data_path.as_ref().map(PathBuf::from),
                // Convention: libtorch lives at `<worker.path>/libtorch/<worker.arch>`
                // when the host declares an arch; else probe walks
                // `<worker.path>/libtorch/.active` (single-host default).
                worker.arch
                    .as_ref()
                    .map(|a| PathBuf::from(&worker.path).join("libtorch").join(a)),
                worker.docker.clone(),
                data_path_explicit,
            )
        } else {
            probe_remote_via_ssh(worker, skip_mount)
        };
        reports.push(r);
    }
    let any_red = reports.iter().any(|r| !r.green());
    if json {
        print_cluster_json(&reports);
    } else {
        print_cluster_report(&reports);
    }
    if any_red { 1 } else { 0 }
}

/// SSH to `host` and run `fdl probe --json` there. The remote `fdl`
/// is resolved as `{worker.path}/target/release/fdl` (release build on
/// shared storage). Returns a synthetic `ProbeReport` carrying any
/// SSH/parse failure in `issues` when the remote call fails — caller
/// treats those as red verdicts.
fn probe_remote_via_ssh(worker: &ClusterWorker, skip_mount: bool) -> ProbeReport {
    let ssh_target = worker
        .ssh
        .as_ref()
        .and_then(|s| s.target.as_deref())
        .unwrap_or(&worker.host)
        .to_string();
    // Invoke bare `fdl` and rely on the remote shell's PATH. Each
    // host owns its fdl install (typically `cargo install flodl-cli`
    // into ~/.cargo/bin or ~/.local/bin); the controller does not
    // reach into the remote's build tree. If a host lacks `fdl` on
    // PATH the SSH command returns "fdl: command not found" exit
    // 127, which the probe-result parser surfaces as an SSH error
    // for that host.
    let mut remote_args: Vec<String> = vec![
        "fdl".into(),
        "probe".into(),
        "--json".into(),
    ];
    // Only forward --data-path when the host declared one. Without it,
    // the remote falls back to DEFAULT_DATA_PATH and the probe treats a
    // missing path as a WARNING (convention default) rather than an
    // ERROR (explicit promise the user made in cluster.yml).
    if let Some(dp) = &worker.data_path {
        remote_args.push("--data-path".into());
        remote_args.push(dp.clone());
    }
    if skip_mount {
        remote_args.push("--skip-mount".into());
    }
    // Pass the host's libtorch path to the remote probe so the worker
    // doesn't have to discover libtorch from its filesystem. Derived
    // from the convention `<worker.path>/libtorch/<worker.arch>` when
    // arch is declared; otherwise omitted, and the remote probe walks
    // `<worker.path>/libtorch/.active` (single-host default).
    if let Some(arch) = &worker.arch {
        remote_args.push("--libtorch-path".into());
        remote_args.push(format!(
            "{path}/libtorch/{arch}",
            path = worker.path.trim_end_matches('/'),
        ));
    }
    // Pass the host's docker: compose service. Tells the remote probe
    // that NCCL ships inside the container image, so it should report
    // "via Docker image <svc>" instead of scanning host library paths.
    if let Some(svc) = &worker.docker {
        remote_args.push("--docker".into());
        remote_args.push(svc.clone());
    }
    // Quote each remote arg into a single shell-safe command string;
    // remote PATH may not have `fdl`, hence the absolute path.
    let quoted = remote_args
        .iter()
        .map(|a| {
            // Conservative single-quote escape — fine for absolute
            // paths and known-good flags. No backslash interpretation.
            format!("'{}'", a.replace('\'', "'\\''"))
        })
        .collect::<Vec<_>>()
        .join(" ");
    // cd into the remote host's project path BEFORE invoking fdl so
    // `Context::resolve()` walks up from there + finds the shared
    // libtorch/.active. Without the cd, fdl walks from the SSH login
    // dir (typically ~) and either misses the project root or
    // resolves a stale local fdl install.
    let remote_cmd = format!(
        "cd '{}' && {quoted}",
        worker.path.replace('\'', "'\\''"),
    );

    // Honor the worker's `ssh:` sub-block (port / user / identity_file /
    // options) just like the cluster dispatch path — otherwise a
    // Docker-container rank on `127.0.0.1:2222` with an identity_file is
    // dialed on the default port 22 and the connect is refused (the
    // probe then reports the host red even though dispatch works fine).
    let mut cmd = Command::new("ssh");
    // User ssh.options first (they win), then flodl's defaults (M17).
    crate::cluster::apply_worker_ssh_opts(&mut cmd, worker);
    cmd.args([
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
    ]);
    cmd.arg(&ssh_target).arg(&remote_cmd);
    let output = cmd.output();

    let mut report = ProbeReport {
        host: worker.host.clone(),
        gpus: Vec::new(),
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: Vec::new(),
        },
        data_path: DataPathStatus {
            path: PathBuf::from(worker.effective_data_path()),
            exists: false,
            readable: false,
            fs_type: None,
            skipped: skip_mount,
        },
        nccl: NcclStatus {
            library_path: None,
            all_found: Vec::new(),
            via_docker: worker.docker.clone(),
        },
        issues: Vec::new(),
        warnings: Vec::new(),
    };
    match output {
        Err(e) => {
            report.issues.push(format!(
                "ssh to `{ssh_target}` failed before probe ran: {e}"
            ));
        }
        Ok(out) => {
            // The remote probe returns exit 1 when it found issues —
            // that's the SAME signal the remote report carries via
            // its own `issues` field. Don't treat it as fatal here;
            // try to parse stdout regardless. Only fall back to a
            // synthetic SSH-error report when parse actually fails.
            let stdout = String::from_utf8_lossy(&out.stdout);
            match parse_remote_json(&stdout, worker) {
                Ok(r) => report = r,
                Err(parse_err) => {
                    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    report.issues.push(format!(
                        "remote probe on `{ssh_target}` exited {} — \
                         stdout did not parse as JSON ({parse_err}); \
                         stderr: {stderr}; first 200 chars of stdout: {:?}",
                        out.status,
                        stdout.chars().take(200).collect::<String>(),
                    ));
                }
            }
        }
    }
    report
}

/// Parse the remote `fdl probe --json` output back into a
/// [`ProbeReport`]. Minimal parser — pulls the fields the report
/// formatter needs and trusts the remote produced what it produces.
/// `host` is the cluster.yml entry; used to fill the `host` field of
/// the report so name matches the topology (the remote returns its
/// `hostname(1)`, which may differ from the cluster.yml name and is
/// the more common source of "probe says host X but cluster.yml says
/// host Y" diagnostics).
fn parse_remote_json(json: &str, worker: &ClusterWorker) -> Result<ProbeReport, String> {
    let v: serde_json::Value = serde_json::from_str(json.trim())
        .map_err(|e| format!("JSON parse: {e}"))?;

    let mut report = ProbeReport {
        host: worker.host.clone(),
        gpus: Vec::new(),
        libtorch: LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: Vec::new(),
        },
        data_path: DataPathStatus {
            path: PathBuf::from(worker.effective_data_path()),
            exists: false,
            readable: false,
            fs_type: None,
            skipped: false,
        },
        nccl: NcclStatus {
            library_path: None,
            all_found: Vec::new(),
            via_docker: worker.docker.clone(),
        },
        issues: Vec::new(),
        warnings: Vec::new(),
    };

    if let Some(gpus) = v.get("gpus").and_then(|g| g.as_array()) {
        for g in gpus {
            let index = g.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
            let name = g.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let sm = g.get("sm").and_then(|v| v.as_str()).unwrap_or("sm_0");
            let (sm_major, sm_minor) = parse_sm(sm);
            let total_memory_mb = g.get("vram_mb").and_then(|v| v.as_u64()).unwrap_or(0);
            report.gpus.push(GpuInfo {
                index,
                name,
                sm_major,
                sm_minor,
                total_memory_mb,
            });
        }
    }

    if let Some(lt) = v.get("libtorch") {
        if !lt.is_null() {
            let path = lt.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let valid_dir = lt.get("valid_dir").and_then(|v| v.as_bool()).unwrap_or(false);
            let info = LibtorchInfo {
                path,
                torch_version: lt.get("torch").and_then(|v| v.as_str()).map(String::from),
                cuda_version: lt.get("cuda").and_then(|v| v.as_str()).map(String::from),
                archs: lt.get("archs").and_then(|v| v.as_str()).map(String::from),
                source: None,
            };
            let mut archs_match = Vec::new();
            if let Some(am) = lt.get("archs_match").and_then(|v| v.as_array()) {
                for entry in am {
                    let gpu = entry.get("gpu").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
                    let covered = entry.get("covered").and_then(|v| v.as_bool()).unwrap_or(false);
                    archs_match.push((gpu, covered));
                }
            }
            report.libtorch = LibtorchStatus { info: Some(info), valid_dir, archs_match };
        }
    }

    if let Some(dp) = v.get("data_path") {
        if !dp.is_null() {
            let path = dp
                .get("path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(worker.effective_data_path()));
            let exists = dp.get("exists").and_then(|v| v.as_bool()).unwrap_or(false);
            let readable = dp.get("readable").and_then(|v| v.as_bool()).unwrap_or(false);
            let fs_type = dp.get("fs_type").and_then(|v| v.as_str()).map(String::from);
            report.data_path = DataPathStatus {
                path,
                exists,
                readable,
                fs_type,
                skipped: false,
            };
        } else {
            report.data_path.skipped = true;
        }
    }

    if let Some(nccl) = v.get("nccl") {
        if !nccl.is_null() {
            let p = nccl.get("library_path").and_then(|v| v.as_str()).map(PathBuf::from);
            report.nccl.library_path = p.clone();
            if let Some(p) = p {
                report.nccl.all_found.push(p);
            }
            // Prefer the remote's reported via_docker over the
            // cluster.yml field — the controller already passed it in
            // via --docker so the remote echo confirms what was used;
            // they should match, and using the remote's keeps the
            // round-trip a single source of truth.
            if let Some(svc) = nccl.get("via_docker").and_then(|v| v.as_str()) {
                report.nccl.via_docker = Some(svc.to_string());
            }
        }
    }

    if let Some(issues) = v.get("issues").and_then(|v| v.as_array()) {
        for i in issues {
            if let Some(s) = i.as_str() {
                report.issues.push(s.to_string());
            }
        }
    }
    if let Some(warnings) = v.get("warnings").and_then(|v| v.as_array()) {
        for w in warnings {
            if let Some(s) = w.as_str() {
                report.warnings.push(s.to_string());
            }
        }
    }

    Ok(report)
}

fn parse_sm(s: &str) -> (u32, u32) {
    // "sm_NNN" → (sm_major, sm_minor). sm_NN concatenates the two
    // digits; reverse by treating last digit as minor when total
    // string after "sm_" is 2 chars, else split major/minor on the
    // canonical 2-digit minor convention NVIDIA uses (sm_120 = 12.0).
    let n = s.trim_start_matches("sm_");
    if let Ok(v) = n.parse::<u32>() {
        // sm_120 -> 12.0; sm_61 -> 6.1; sm_90 -> 9.0; sm_86 -> 8.6.
        // NVIDIA convention: last digit is minor.
        let major = v / 10;
        let minor = v % 10;
        return (major, minor);
    }
    (0, 0)
}

// ---------------------------------------------------------------------------
// Cluster output
// ---------------------------------------------------------------------------

fn print_cluster_report(reports: &[ProbeReport]) {
    println!("floDl Cluster Probe — {} hosts", reports.len());
    println!("{}", "=".repeat(40));
    println!();
    for (i, r) in reports.iter().enumerate() {
        if i > 0 {
            println!();
            println!("{}", "-".repeat(40));
            println!();
        }
        print_report(r);
    }
    println!();
    let red = reports.iter().filter(|r| !r.green()).count();
    let yellow = reports
        .iter()
        .filter(|r| r.green() && !r.warnings.is_empty())
        .count();
    let total = reports.len();
    match (red, yellow) {
        (0, 0) => println!("CLUSTER VERDICT: READY (all {total} hosts green)"),
        (0, y) => println!("CLUSTER VERDICT: READY ({y}/{total} hosts have warnings)"),
        (r, 0) => println!("CLUSTER VERDICT: ISSUES ({r}/{total} hosts have errors)"),
        (r, y) => println!(
            "CLUSTER VERDICT: ISSUES ({r}/{total} hosts have errors, \
             {y} also have warnings)"
        ),
    }
}

fn print_cluster_json(reports: &[ProbeReport]) {
    let mut b = String::with_capacity(4096);
    b.push_str("{\"hosts\":[");
    for (i, r) in reports.iter().enumerate() {
        if i > 0 {
            b.push(',');
        }
        b.push_str(&report_to_json_object(r));
    }
    b.push(']');
    let red = reports.iter().filter(|r| !r.green()).count();
    let _ = write!(b, ",\"hosts_total\":{}", reports.len());
    let _ = write!(b, ",\"hosts_red\":{}", red);
    let _ = write!(b, ",\"ready\":{}", red == 0);
    b.push('}');
    println!("{}", b);
}

// ---------------------------------------------------------------------------
// Report structs
// ---------------------------------------------------------------------------

/// Top-level probe verdict for one host. `green()` is the aggregate
/// gate.
///
/// `issues` are blocking errors (exit non-zero); `warnings` are advisory
/// (exit zero, surfaced in the report). The split matters because
/// "/flodl/data missing" on a single-host rig that doesn't use shared
/// storage is informational, while a worker host that declared an
/// explicit `data_path:` in cluster.yml and can't see it is broken.
pub struct ProbeReport {
    pub host: String,
    pub gpus: Vec<GpuInfo>,
    pub libtorch: LibtorchStatus,
    pub data_path: DataPathStatus,
    pub nccl: NcclStatus,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

impl ProbeReport {
    /// `true` when no issues were collected — every checked component
    /// passed (warnings do NOT flip this). Callers may still want to
    /// inspect individual statuses for diagnostic detail; the exit
    /// code follows this flag.
    pub fn green(&self) -> bool {
        self.issues.is_empty()
    }
}

/// libtorch directory + arch metadata + per-GPU compatibility verdict.
pub struct LibtorchStatus {
    /// Parsed `.arch` metadata (if libtorch is present + readable).
    pub info: Option<LibtorchInfo>,
    /// `lib/` subdirectory present (cheap "is this a libtorch dir?"
    /// check that doesn't require parsing).
    pub valid_dir: bool,
    /// Per-GPU `(gpu_index, archs_cover_this_gpu)`. Empty when libtorch
    /// is missing.
    pub archs_match: Vec<(u8, bool)>,
}

/// Shared-data path visibility + filesystem-type detection.
pub struct DataPathStatus {
    pub path: PathBuf,
    pub exists: bool,
    pub readable: bool,
    /// Underlying filesystem type from `/proc/mounts` (e.g. `virtiofs`,
    /// `nfs4`, `cifs`, `fuse.sshfs`, `ext4`). `None` when the path is
    /// not mounted (falls inside the parent FS) or when /proc/mounts
    /// is unavailable.
    pub fs_type: Option<String>,
    /// `true` when the check was explicitly bypassed via
    /// `--skip-mount`; the path/exists fields are unset (`PathBuf::new`
    /// + false) in that case.
    pub skipped: bool,
}

/// NCCL discovery result. NCCL is loaded dynamically by libtorch, so
/// the probe just hunts for `libnccl.so*` on the usual library paths
/// — unless [`Self::via_docker`] is set, in which case NCCL ships
/// inside the container image and the host scan is skipped.
pub struct NcclStatus {
    /// First `libnccl.so*` found, if any. Used in the report to show
    /// the user which install will be picked up.
    pub library_path: Option<PathBuf>,
    /// All discovered `libnccl.so*` paths (informational; multiple
    /// versions in different prefixes is a misconfiguration source).
    pub all_found: Vec<PathBuf>,
    /// Docker compose service that owns NCCL on this host. When set,
    /// the probe records "via Docker image `<svc>`" instead of scanning
    /// the host filesystem. `None` means the host runs flodl natively
    /// and NCCL must live on it.
    pub via_docker: Option<String>,
}

// ---------------------------------------------------------------------------
// Single-host probe
// ---------------------------------------------------------------------------

/// Probe the local host. `data_path_override` (from `--data-path` CLI
/// flag) overrides config; `skip_mount` short-circuits the shared-data
/// check (useful for single-host setups without a shared FS
/// configured); `libtorch_path_override` (from `--libtorch-path`)
/// points at a libtorch install outside the project tree (used by
/// cluster-mode remote probes where libtorch lives on a dedicated
/// share like `/mnt/libtorch`). `via_docker` (from `--docker <svc>` or
/// the cluster.yml host's `docker:` field) tells the probe NCCL ships
/// inside a container image, so host-level NCCL scanning is replaced
/// by an informational "via Docker image `<svc>`" line.
///
/// `data_path_explicit`: when `true`, a missing shared-data path is an
/// ERROR (the user/cluster.yml promised it); when `false`, it's a
/// WARNING (the convention default was used). Internal flag — callers
/// must derive it from "did the caller pass an explicit data_path".
pub fn probe_local(
    ctx: &Context,
    skip_mount: bool,
    data_path_override: Option<PathBuf>,
    libtorch_path_override: Option<PathBuf>,
    via_docker: Option<String>,
    data_path_explicit: bool,
) -> ProbeReport {
    let host = resolve_local_hostname();
    let gpus = system::detect_gpus();
    let mut issues: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let libtorch = match libtorch_path_override {
        Some(p) => check_libtorch_at(&p, &gpus, &mut issues),
        None => check_libtorch(&ctx.root, &gpus, &mut issues),
    };
    let data_path = check_data_path(
        data_path_override.unwrap_or_else(|| PathBuf::from(DEFAULT_DATA_PATH)),
        skip_mount,
        data_path_explicit,
        &mut issues,
        &mut warnings,
    );
    let nccl = check_nccl(via_docker, &mut issues);

    if gpus.is_empty() {
        issues.push(
            "no CUDA GPUs detected — nvidia-smi missing or driver \
             unhealthy. Single-host CPU training will still work; \
             multi-rank NCCL requires a working GPU stack."
                .into(),
        );
    }

    ProbeReport {
        host,
        gpus,
        libtorch,
        data_path,
        nccl,
        issues,
        warnings,
    }
}

/// Build a [`LibtorchStatus`] from a resolved [`LibtorchInfo`] (or
/// `None` when the pointer could not be resolved). Used by the
/// pointer-file shape of [`check_libtorch_at`]; mirrors the
/// arch-check and valid-dir logic from [`check_libtorch`] without
/// duplicating its `.active` walk.
fn libtorch_status_from_info(
    info: Option<LibtorchInfo>,
    libtorch_root: &Path,
    gpus: &[GpuInfo],
    issues: &mut Vec<String>,
) -> LibtorchStatus {
    let valid_dir = match &info {
        Some(i) => libtorch_root.join(&i.path).join("lib").is_dir(),
        None => false,
    };
    let mut archs_match: Vec<(u8, bool)> = Vec::new();
    if let Some(i) = &info {
        if let Some(archs) = &i.archs {
            for g in gpus {
                archs_match.push((g.index, detect::arch_compatible(g, archs)));
            }
            for g in gpus {
                if !detect::arch_compatible(g, archs) {
                    issues.push(format!(
                        "GPU {} ({}, {}) not covered by libtorch archs `{}`.",
                        g.index,
                        g.short_name(),
                        g.sm_version(),
                        archs
                    ));
                }
            }
        } else {
            issues.push(
                "libtorch is present but `.arch` metadata is missing — \
                 cannot verify GPU compatibility."
                    .into(),
            );
        }
    } else {
        issues.push(
            "libtorch pointer file did not resolve to a configured \
             variant (file empty or missing). Check the `.active*` \
             content names a real subdir under `libtorch/`."
                .into(),
        );
    }
    LibtorchStatus { info, valid_dir, archs_match }
}

/// Variant that takes an explicit libtorch path instead of walking
/// from the project root. Accepts three shapes:
///
/// 1. **Libtorch ROOT** (dir containing `.active` + `builds/` /
///    `precompiled/`) — delegates to [`check_libtorch`] which walks
///    `.active`.
/// 2. **Pointer file** (file path ending in `.active*`, e.g.
///    `libtorch/.active.blackwell`) — reads the pointer and resolves
///    the variant relative to the file's parent directory. Used for
///    heterogeneous rigs where each host's `cluster.yml` entry sets
///    `libtorch_path:` to a different case file.
/// 3. **Direct variant dir** (has `lib/libtorch.so` + optional
///    `.arch`) — used as-is.
fn check_libtorch_at(
    path: &Path,
    gpus: &[GpuInfo],
    issues: &mut Vec<String>,
) -> LibtorchStatus {
    // Shape 2: a regular file whose name starts with `.active` is a
    // pointer to a variant subdir. Resolve relative to the file's
    // parent (the libtorch root). Note: `.active` itself is also a
    // file but Shape 1 catches it via dir-containing-.active above.
    if path.is_file()
        && path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(".active"))
    {
        let libtorch_root = path.parent().unwrap_or(path);
        let info = detect::read_active_from(path, libtorch_root);
        return libtorch_status_from_info(info, libtorch_root, gpus, issues);
    }
    if path.join(".active").exists() {
        return check_libtorch(path, gpus, issues);
    }
    let dir = path;
    let valid_dir = dir.join("lib").is_dir();
    if !valid_dir {
        issues.push(format!(
            "libtorch directory `{}` does not contain `lib/` — pass \
             `--libtorch-path` pointing at a real libtorch install \
             (the directory with `lib/libtorch.so`).",
            dir.display()
        ));
        return LibtorchStatus {
            info: None,
            valid_dir: false,
            archs_match: Vec::new(),
        };
    }
    let mut info = LibtorchInfo {
        path: dir.display().to_string(),
        torch_version: None,
        cuda_version: None,
        archs: None,
        source: None,
    };
    if let Ok(content) = std::fs::read_to_string(dir.join(".arch")) {
        for line in content.lines() {
            if let Some(v) = line.strip_prefix("torch=") {
                info.torch_version = Some(v.into());
            } else if let Some(v) = line.strip_prefix("cuda=") {
                info.cuda_version = Some(v.into());
            } else if let Some(v) = line.strip_prefix("archs=") {
                info.archs = Some(v.into());
            } else if let Some(v) = line.strip_prefix("source=") {
                info.source = Some(v.into());
            }
        }
    }
    let mut archs_match = Vec::new();
    if let Some(archs) = &info.archs {
        for g in gpus {
            archs_match.push((g.index, detect::arch_compatible(g, archs)));
        }
        for g in gpus {
            if !detect::arch_compatible(g, archs) {
                issues.push(format!(
                    "GPU {} ({}, {}) not covered by libtorch archs `{}`.",
                    g.index,
                    g.short_name(),
                    g.sm_version(),
                    archs
                ));
            }
        }
    } else {
        issues.push(format!(
            "libtorch at `{}` is present but `.arch` metadata is \
             missing — cannot verify GPU compatibility.",
            dir.display()
        ));
    }
    LibtorchStatus { info: Some(info), valid_dir: true, archs_match }
}

fn check_libtorch(
    root: &Path,
    gpus: &[GpuInfo],
    issues: &mut Vec<String>,
) -> LibtorchStatus {
    // `root` can be the project root OR the libtorch root (latter is
    // what `--libtorch-path /path/to/libtorch` resolves to when the
    // dir has `.active`). `read_active` expects the parent of
    // `libtorch/`; if `root` is itself a libtorch root (has `.active`
    // directly under it), reframe.
    let info = if root.join(".active").exists() {
        // Synthesize the parent + variant path, then call read_active
        // with a synthetic parent that exposes `libtorch/.active`.
        let active_text = std::fs::read_to_string(root.join(".active")).ok();
        match active_text {
            Some(t) => {
                let variant = t.trim().to_string();
                if variant.is_empty() {
                    None
                } else {
                    let mut info = LibtorchInfo {
                        path: variant.clone(),
                        torch_version: None,
                        cuda_version: None,
                        archs: None,
                        source: None,
                    };
                    let arch_file = root.join(&variant).join(".arch");
                    if let Ok(c) = std::fs::read_to_string(arch_file) {
                        for line in c.lines() {
                            if let Some(v) = line.strip_prefix("torch=") {
                                info.torch_version = Some(v.into());
                            } else if let Some(v) = line.strip_prefix("cuda=") {
                                info.cuda_version = Some(v.into());
                            } else if let Some(v) = line.strip_prefix("archs=") {
                                info.archs = Some(v.into());
                            } else if let Some(v) = line.strip_prefix("source=") {
                                info.source = Some(v.into());
                            }
                        }
                    }
                    Some(info)
                }
            }
            None => None,
        }
    } else {
        detect::read_active(root)
    };
    let valid_dir = match &info {
        Some(i) => {
            if root.join(".active").exists() {
                root.join(&i.path).join("lib").is_dir()
            } else {
                detect::is_valid_variant(root, &i.path)
            }
        }
        None => false,
    };

    let mut archs_match: Vec<(u8, bool)> = Vec::new();
    if let Some(i) = &info {
        if let Some(archs) = &i.archs {
            for g in gpus {
                archs_match.push((g.index, detect::arch_compatible(g, archs)));
            }
            // Report any GPU not covered as an issue.
            for g in gpus {
                if !detect::arch_compatible(g, archs) {
                    issues.push(format!(
                        "GPU {} ({}, {}) not covered by libtorch archs `{}`. \
                         Rebuild libtorch with this arch or activate a \
                         compatible variant.",
                        g.index,
                        g.short_name(),
                        g.sm_version(),
                        archs
                    ));
                }
            }
        } else {
            issues.push(
                "libtorch is present but `.arch` metadata is missing — \
                 cannot verify GPU compatibility. Place an `.arch` file \
                 in the variant directory (cuda=, torch=, archs=, source=)."
                    .into(),
            );
        }
    } else {
        issues.push(
            "libtorch not configured — `libtorch/.active` missing or \
             empty. Run `fdl libtorch download` or `fdl libtorch build` \
             to provision a variant."
                .into(),
        );
    }

    LibtorchStatus { info, valid_dir, archs_match }
}

fn check_data_path(
    path: PathBuf,
    skip_mount: bool,
    explicit: bool,
    issues: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> DataPathStatus {
    if skip_mount {
        return DataPathStatus {
            path: PathBuf::new(),
            exists: false,
            readable: false,
            fs_type: None,
            skipped: true,
        };
    }
    let exists = path.exists();
    let readable = exists && std::fs::read_dir(&path).is_ok();
    let fs_type = detect_fs_type(&path);

    if !exists {
        if explicit {
            // The user (or cluster.yml) promised this path. Missing it
            // is a launch-breaking error — training fan-out would
            // discover this mid-run when a checkpoint write hangs.
            issues.push(format!(
                "shared data path `{}` does not exist on this host. flodl \
                 assumes a shared filesystem (NAS / SMB / virtiofs / SSHFS) \
                 mounted at the same logical path on every node. Mount the \
                 shared storage or correct `data_path:` in cluster.yml.",
                path.display()
            ));
        } else {
            // No explicit path was declared — the convention default
            // `/flodl/data` was tried. Missing it is fine for users who
            // don't use shared storage; surface it as a warning so they
            // know the default isn't wired up.
            warnings.push(format!(
                "convention shared-data path `{}` not present on this host \
                 (no `data_path:` declared in cluster.yml). Ignore if you \
                 don't use shared storage; otherwise set `data_path:` per \
                 host or mount `{}`.",
                path.display(),
                path.display()
            ));
        }
    } else if !readable {
        issues.push(format!(
            "shared data path `{}` exists but is not readable by the \
             current user. Check mount permissions / uid mapping.",
            path.display()
        ));
    }

    DataPathStatus { path, exists, readable, fs_type, skipped: false }
}

fn check_nccl(via_docker: Option<String>, issues: &mut Vec<String>) -> NcclStatus {
    // Docker-served host: NCCL lives inside the container image, not
    // on the host. Skip the host scan entirely — scanning would
    // false-positive on the false-error path that motivated the docker
    // field (host shows "no libnccl.so" while training actually runs
    // fine inside the cuda/dev image). Report as informational.
    if via_docker.is_some() {
        return NcclStatus {
            library_path: None,
            all_found: Vec::new(),
            via_docker,
        };
    }

    let mut found: Vec<PathBuf> = Vec::new();
    // Common search locations. Order matters — first match wins for
    // the diagnostic `library_path` field.
    let candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/usr/local/cuda/lib64",
        "/opt/cuda/lib64",
    ];
    for dir in candidates {
        let d = Path::new(dir);
        if let Ok(entries) = std::fs::read_dir(d) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let s = name.to_string_lossy();
                if s.starts_with("libnccl.so") {
                    found.push(entry.path());
                }
            }
        }
    }
    // Honor LD_LIBRARY_PATH so user-shipped NCCL (the Pascal rig keeps
    // libnccl.so under ~/nccl/build/lib for the CUDA-13 source build)
    // is discovered.
    if let Ok(paths) = std::env::var("LD_LIBRARY_PATH") {
        for dir in paths.split(':').filter(|p| !p.is_empty()) {
            let d = Path::new(dir);
            if let Ok(entries) = std::fs::read_dir(d) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let s = name.to_string_lossy();
                    if s.starts_with("libnccl.so") {
                        let p = entry.path();
                        if !found.iter().any(|f| f == &p) {
                            found.push(p);
                        }
                    }
                }
            }
        }
    }

    if found.is_empty() {
        issues.push(
            "no `libnccl.so` found on standard library paths or \
             $LD_LIBRARY_PATH. Multi-rank NCCL training will fail at \
             collective init. Install libnccl matching your CUDA \
             version or set LD_LIBRARY_PATH to a custom build (or \
             declare `docker:` on this host in cluster.yml if NCCL \
             ships inside the container image)."
                .into(),
        );
    }

    NcclStatus { library_path: found.first().cloned(), all_found: found, via_docker: None }
}

/// Best-effort filesystem-type lookup via `/proc/mounts`. Walks toward
/// the root looking for the closest mount-point that contains `path`.
/// Returns `None` on non-Linux or when `/proc/mounts` is unavailable.
fn detect_fs_type(path: &Path) -> Option<String> {
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut best: Option<(usize, String)> = None;
    for line in mounts.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 3 {
            continue;
        }
        let mountpoint = Path::new(cols[1]);
        let fs_type = cols[2].to_string();
        if abs.starts_with(mountpoint) {
            let depth = mountpoint.components().count();
            match &best {
                Some((prev_depth, _)) if depth <= *prev_depth => {}
                _ => best = Some((depth, fs_type)),
            }
        }
    }
    best.map(|(_, t)| t)
}

// ---------------------------------------------------------------------------
// Text output
// ---------------------------------------------------------------------------

fn print_report(r: &ProbeReport) {
    println!("floDl Probe — {}", r.host);
    println!("{}", "=".repeat(40));
    println!();

    println!("GPUs ({}):", r.gpus.len());
    for g in &r.gpus {
        println!(
            "  [{}] {} — {}, {} MB",
            g.index,
            g.short_name(),
            g.sm_version(),
            g.total_memory_mb
        );
    }
    println!();

    println!("libtorch:");
    match &r.libtorch.info {
        Some(info) => {
            println!("  path  : {}", info.path);
            if let Some(t) = &info.torch_version {
                println!("  torch : {}", t);
            }
            if let Some(c) = &info.cuda_version {
                println!("  cuda  : {}", c);
            }
            if let Some(a) = &info.archs {
                println!("  archs : {}", a);
            }
            if !r.libtorch.archs_match.is_empty() {
                let ok = r.libtorch.archs_match.iter().filter(|(_, b)| *b).count();
                println!(
                    "  match : {}/{} GPUs covered",
                    ok,
                    r.libtorch.archs_match.len()
                );
            }
            println!(
                "  valid : {}",
                if r.libtorch.valid_dir { "yes" } else { "no" }
            );
        }
        None => println!("  (not configured)"),
    }
    println!();

    println!("Shared data path:");
    if r.data_path.skipped {
        println!("  (skipped via --skip-mount)");
    } else {
        println!("  path     : {}", r.data_path.path.display());
        println!("  exists   : {}", yn(r.data_path.exists));
        println!("  readable : {}", yn(r.data_path.readable));
        if let Some(t) = &r.data_path.fs_type {
            println!("  fs       : {}", t);
        }
    }
    println!();

    println!("NCCL:");
    if let Some(svc) = &r.nccl.via_docker {
        println!("  via Docker image `{}` (host check skipped)", svc);
    } else {
        match &r.nccl.library_path {
            Some(p) => {
                println!("  found    : {}", p.display());
                if r.nccl.all_found.len() > 1 {
                    println!("  others   : {} more (check for version skew)", r.nccl.all_found.len() - 1);
                }
            }
            None => println!("  (no libnccl.so* discovered)"),
        }
    }
    println!();

    print_verdict_lines(&r.issues, &r.warnings);
}

/// Render the three-tier verdict + numbered errors/warnings.
fn print_verdict_lines(issues: &[String], warnings: &[String]) {
    let n_err = issues.len();
    let n_warn = warnings.len();
    let line = match (n_err, n_warn) {
        (0, 0) => "verdict: READY".to_string(),
        (0, m) => format!("verdict: READY ({m} warning{})", plural(m)),
        (n, 0) => format!("verdict: ISSUES ({n} error{})", plural(n)),
        (n, m) => format!(
            "verdict: ISSUES ({n} error{}, {m} warning{})",
            plural(n),
            plural(m)
        ),
    };
    println!("{line}");
    if !issues.is_empty() {
        println!("errors:");
        for (i, msg) in issues.iter().enumerate() {
            println!("  {}. {}", i + 1, msg);
        }
    }
    if !warnings.is_empty() {
        println!("warnings:");
        for (i, msg) in warnings.iter().enumerate() {
            println!("  {}. {}", i + 1, msg);
        }
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn yn(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

// ---------------------------------------------------------------------------
// JSON output (`fdl deploy` + CI consume this shape)
// ---------------------------------------------------------------------------

fn print_json(r: &ProbeReport) {
    println!("{}", report_to_json_object(r));
}

fn report_to_json_object(r: &ProbeReport) -> String {
    let mut b = String::with_capacity(2048);
    b.push('{');
    let _ = write!(b, "\"host\":\"{}\"", system::escape_json(&r.host));

    // GPUs
    b.push_str(",\"gpus\":[");
    for (i, g) in r.gpus.iter().enumerate() {
        if i > 0 { b.push(','); }
        let _ = write!(
            b,
            "{{\"index\":{},\"name\":\"{}\",\"sm\":\"{}\",\"vram_mb\":{}}}",
            g.index,
            system::escape_json(&g.name),
            g.sm_version(),
            g.total_memory_mb
        );
    }
    b.push(']');

    // libtorch
    b.push_str(",\"libtorch\":");
    match &r.libtorch.info {
        Some(info) => {
            let _ = write!(
                b,
                "{{\"path\":\"{}\",\"valid_dir\":{}",
                system::escape_json(&info.path),
                r.libtorch.valid_dir
            );
            if let Some(v) = &info.torch_version {
                let _ = write!(b, ",\"torch\":\"{}\"", system::escape_json(v));
            }
            if let Some(c) = &info.cuda_version {
                let _ = write!(b, ",\"cuda\":\"{}\"", system::escape_json(c));
            }
            if let Some(a) = &info.archs {
                let _ = write!(b, ",\"archs\":\"{}\"", system::escape_json(a));
            }
            b.push_str(",\"archs_match\":[");
            for (i, (gpu, ok)) in r.libtorch.archs_match.iter().enumerate() {
                if i > 0 { b.push(','); }
                let _ = write!(b, "{{\"gpu\":{},\"covered\":{}}}", gpu, ok);
            }
            b.push(']');
            b.push('}');
        }
        None => b.push_str("null"),
    }

    // Shared data path
    b.push_str(",\"data_path\":");
    if r.data_path.skipped {
        b.push_str("null");
    } else {
        let _ = write!(
            b,
            "{{\"path\":\"{}\",\"exists\":{},\"readable\":{}",
            system::escape_json(&r.data_path.path.display().to_string()),
            r.data_path.exists,
            r.data_path.readable
        );
        if let Some(t) = &r.data_path.fs_type {
            let _ = write!(b, ",\"fs_type\":\"{}\"", system::escape_json(t));
        }
        b.push('}');
    }

    // NCCL — always emit an object now (even when host scan was
    // skipped via Docker), so consumers can read `via_docker` without
    // null-checking.
    b.push_str(",\"nccl\":");
    if r.nccl.library_path.is_none() && r.nccl.via_docker.is_none() {
        b.push_str("null");
    } else {
        b.push('{');
        let mut first = true;
        if let Some(p) = &r.nccl.library_path {
            let _ = write!(
                b,
                "\"library_path\":\"{}\",\"count\":{}",
                system::escape_json(&p.display().to_string()),
                r.nccl.all_found.len()
            );
            first = false;
        }
        if let Some(svc) = &r.nccl.via_docker {
            if !first {
                b.push(',');
            }
            let _ = write!(b, "\"via_docker\":\"{}\"", system::escape_json(svc));
        }
        b.push('}');
    }

    // Issues (errors) + warnings + verdict.
    b.push_str(",\"issues\":[");
    for (i, msg) in r.issues.iter().enumerate() {
        if i > 0 { b.push(','); }
        let _ = write!(b, "\"{}\"", system::escape_json(msg));
    }
    b.push(']');
    b.push_str(",\"warnings\":[");
    for (i, msg) in r.warnings.iter().enumerate() {
        if i > 0 { b.push(','); }
        let _ = write!(b, "\"{}\"", system::escape_json(msg));
    }
    b.push(']');
    let _ = write!(b, ",\"ready\":{}", r.green());
    b.push('}');
    b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_path_check_skipped_when_flag_set() {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let status = check_data_path(
            PathBuf::from("/nonexistent"),
            true,
            false,
            &mut issues,
            &mut warnings,
        );
        assert!(status.skipped);
        assert!(issues.is_empty(), "skip_mount must suppress missing-path issue");
        assert!(warnings.is_empty(), "skip_mount must suppress missing-path warning");
    }

    #[test]
    fn data_path_check_explicit_missing_is_error() {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let status = check_data_path(
            PathBuf::from("/this/should/never/exist/flodl-probe-test"),
            false,
            true, // explicit
            &mut issues,
            &mut warnings,
        );
        assert!(!status.exists);
        assert!(!status.readable);
        assert_eq!(issues.len(), 1, "explicit missing path → error");
        assert!(warnings.is_empty());
    }

    #[test]
    fn data_path_check_default_missing_is_warning() {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let status = check_data_path(
            PathBuf::from("/this/should/never/exist/flodl-probe-test"),
            false,
            false, // convention default — not explicit
            &mut issues,
            &mut warnings,
        );
        assert!(!status.exists);
        assert!(issues.is_empty(), "default missing path must NOT error");
        assert_eq!(warnings.len(), 1, "default missing path → warning");
    }

    #[test]
    fn data_path_check_reports_readable_tmp() {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let status = check_data_path(
            PathBuf::from("/tmp"),
            false,
            false,
            &mut issues,
            &mut warnings,
        );
        // /tmp is virtually always readable on Linux; if not we'd see
        // it in `issues` and the test would surface the surprise.
        assert!(status.exists);
        assert!(status.readable);
        assert!(issues.is_empty(), "issues = {:?}", issues);
        assert!(warnings.is_empty(), "warnings = {:?}", warnings);
    }

    #[test]
    fn nccl_via_docker_skips_host_scan() {
        let mut issues = Vec::new();
        let status = check_nccl(Some("cuda".into()), &mut issues);
        assert!(issues.is_empty(), "docker-served NCCL must not produce errors");
        assert!(status.library_path.is_none());
        assert!(status.all_found.is_empty());
        assert_eq!(status.via_docker.as_deref(), Some("cuda"));
    }

    #[test]
    fn verdict_format_three_tier() {
        // No errors, no warnings → READY.
        let r0 = ProbeReport {
            host: "h".into(),
            gpus: vec![],
            libtorch: LibtorchStatus { info: None, valid_dir: false, archs_match: vec![] },
            data_path: DataPathStatus {
                path: PathBuf::new(), exists: false, readable: false, fs_type: None, skipped: true,
            },
            nccl: NcclStatus { library_path: None, all_found: vec![], via_docker: None },
            issues: vec![],
            warnings: vec![],
        };
        assert!(r0.green());

        // Warning-only is still green (exit 0).
        let r1 = ProbeReport { warnings: vec!["w".into()], ..clone_report(&r0) };
        assert!(r1.green());

        // Error flips green to false.
        let r2 = ProbeReport { issues: vec!["e".into()], ..clone_report(&r0) };
        assert!(!r2.green());
    }

    // Local clone helper — ProbeReport intentionally not Clone (Vec<GpuInfo>
    // has its own ownership).
    fn clone_report(r: &ProbeReport) -> ProbeReport {
        ProbeReport {
            host: r.host.clone(),
            gpus: vec![],
            libtorch: LibtorchStatus {
                info: None,
                valid_dir: r.libtorch.valid_dir,
                archs_match: vec![],
            },
            data_path: DataPathStatus {
                path: r.data_path.path.clone(),
                exists: r.data_path.exists,
                readable: r.data_path.readable,
                fs_type: r.data_path.fs_type.clone(),
                skipped: r.data_path.skipped,
            },
            nccl: NcclStatus {
                library_path: r.nccl.library_path.clone(),
                all_found: r.nccl.all_found.clone(),
                via_docker: r.nccl.via_docker.clone(),
            },
            issues: r.issues.clone(),
            warnings: r.warnings.clone(),
        }
    }

    #[test]
    fn json_emits_warnings_array() {
        let r = ProbeReport {
            host: "h".into(),
            gpus: vec![],
            libtorch: LibtorchStatus { info: None, valid_dir: false, archs_match: vec![] },
            data_path: DataPathStatus {
                path: PathBuf::new(), exists: false, readable: false, fs_type: None, skipped: true,
            },
            nccl: NcclStatus { library_path: None, all_found: vec![], via_docker: Some("cuda".into()) },
            issues: vec![],
            warnings: vec!["data-path missing".into()],
        };
        let j = report_to_json_object(&r);
        let v: serde_json::Value = serde_json::from_str(&j).expect("emit valid JSON");
        assert!(v["ready"].as_bool().unwrap());
        let warns = v["warnings"].as_array().expect("warnings: []");
        assert_eq!(warns.len(), 1);
        assert_eq!(v["nccl"]["via_docker"].as_str(), Some("cuda"));
    }

    #[test]
    fn fs_type_detected_for_root() {
        let t = detect_fs_type(Path::new("/"));
        // / is mounted on every Linux box; detection should not fail.
        // Skip on non-Linux (CI matrix) — /proc/mounts unavailable.
        if std::path::Path::new("/proc/mounts").exists() {
            assert!(t.is_some(), "expected fs_type for /");
        }
    }
}
