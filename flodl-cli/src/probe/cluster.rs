//! Cluster probing: fan out to every declared host, aggregate the
//! verdicts.
//!
//! A remote host is probed by running `fdl probe --json` over SSH and
//! parsing its report back, so a worker needs no special agent — the
//! same binary answers for itself. The local host is probed in-process.

use std::fmt::Write;
use std::path::PathBuf;
use std::process::Command;

use crate::cluster::resolve_local_hostname;
use crate::config::{self, ClusterWorker};
use crate::context::Context;
use crate::libtorch::detect::LibtorchInfo;
use crate::util::system::GpuInfo;
use flodl_hw::{GpuArch, GpuVendor};

use super::checks::probe_local;
use super::report::{print_report, report_to_json_object};
use super::{DataPathStatus, LibtorchStatus, NcclStatus, ProbeReport};

pub(super) fn load_cluster_for_env(ctx: &Context, env_name: &str) -> Option<config::ClusterConfig> {
    let config_path = config::find_config(&ctx.root)?;
    let project = config::load_project_with_env(&config_path, Some(env_name)).ok()?;
    project.cluster
}

// ---------------------------------------------------------------------------
// Cluster fan-out
// ---------------------------------------------------------------------------

pub(super) fn run_cluster(cluster: &config::ClusterConfig, json: bool, skip_mount: bool) -> i32 {
    let local = resolve_local_hostname();
    let mut reports: Vec<ProbeReport> = Vec::with_capacity(cluster.workers.len());
    for worker in &cluster.workers {
        let r = if worker.host == local {
            // Local rank: probe in-process, honor the host's data_path,
            // arch (libtorch variant), and docker service (if set in cluster.yml).
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
                worker
                    .arch
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
/// is invoked bare and resolved by the remote shell's PATH (each host
/// owns its own `fdl` install; the controller does not reach into the
/// remote's build tree). Returns a synthetic `ProbeReport` carrying any
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
    let mut remote_args: Vec<String> = vec!["fdl".into(), "probe".into(), "--json".into()];
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
    // Quote each remote arg into a single shell-safe command string
    // (paths and options may contain spaces / metacharacters).
    let quoted = remote_args
        .iter()
        .map(|a| crate::util::shell::posix_quote(a))
        .collect::<Vec<_>>()
        .join(" ");
    // cd into the remote host's project path BEFORE invoking fdl so
    // `Context::resolve()` walks up from there + finds the shared
    // libtorch/.active. Without the cd, fdl walks from the SSH login
    // dir (typically ~) and either misses the project root or
    // resolves a stale local fdl install.
    let remote_cmd = format!(
        "cd {} && {quoted}",
        crate::util::shell::posix_quote(&worker.path),
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
pub(super) fn parse_remote_json(json: &str, worker: &ClusterWorker) -> Result<ProbeReport, String> {
    let v: serde_json::Value =
        serde_json::from_str(json.trim()).map_err(|e| format!("JSON parse: {e}"))?;

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
            let name = g
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let total_memory_mb = g.get("vram_mb").and_then(|v| v.as_u64()).unwrap_or(0);
            // `vendor` + `arch` are the vendor-plural pair. `sm` is the
            // legacy NVIDIA-only key, still read so a probe against an
            // older remote fdl keeps working.
            let vendor = g
                .get("vendor")
                .and_then(|v| v.as_str())
                .and_then(GpuVendor::parse)
                .unwrap_or(GpuVendor::Nvidia);
            let token = g
                .get("arch")
                .and_then(|v| v.as_str())
                .or_else(|| g.get("sm").and_then(|v| v.as_str()))
                .unwrap_or_default();
            let Some(arch) = GpuArch::parse(vendor, token) else {
                // A device we cannot place is worse than one we drop: an
                // unparsed arch would silently compare as incompatible
                // against every libtorch variant. Say so instead.
                report.warnings.push(format!(
                    "host {:?}: GPU {index} reports an unrecognized {vendor} arch \
                     {token:?}; skipping it in the report",
                    worker.host,
                ));
                continue;
            };
            report.gpus.push(GpuInfo {
                index,
                vendor,
                name,
                arch,
                total_memory_mb,
            });
        }
    }

    if let Some(lt) = v.get("libtorch")
        && !lt.is_null()
    {
        let path = lt
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let valid_dir = lt
            .get("valid_dir")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
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
                let covered = entry
                    .get("covered")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                archs_match.push((gpu, covered));
            }
        }
        report.libtorch = LibtorchStatus {
            info: Some(info),
            valid_dir,
            archs_match,
        };
    }

    if let Some(dp) = v.get("data_path") {
        if !dp.is_null() {
            let path = dp
                .get("path")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(worker.effective_data_path()));
            let exists = dp.get("exists").and_then(|v| v.as_bool()).unwrap_or(false);
            let readable = dp
                .get("readable")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
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

    if let Some(nccl) = v.get("nccl")
        && !nccl.is_null()
    {
        let p = nccl
            .get("library_path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from);
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

    // Shape guard: the current emitter always writes these keys. Their
    // complete absence means the remote fdl speaks a different probe
    // schema (version skew) — surface that instead of letting the lenient
    // per-field defaults masquerade as "no GPUs" / "not ready".
    for key in ["gpus", "ready"] {
        if v.get(key).is_none() {
            report.issues.push(format!(
                "remote probe JSON has no {key:?} field — the remote fdl \
                 likely speaks a different probe schema (version skew); \
                 update fdl on `{}`",
                worker.host
            ));
        }
    }

    Ok(report)
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
