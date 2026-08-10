//! The checks themselves: what this box has, and whether it adds up.
//!
//! Each `check_*` appends to `issues` (blocking) or `warnings`
//! (advisory) rather than returning early, because a probe's value is
//! the FULL picture of what is wrong — stopping at the first gap would
//! send an operator round the loop once per problem.

use std::path::{Path, PathBuf};

use crate::cluster::resolve_local_hostname;
use crate::context::Context;

use crate::config::DEFAULT_DATA_PATH;
use crate::libtorch::detect::{self, LibtorchInfo};
use crate::util::requirements;
use crate::util::system::GpuInfo;
use flodl_hw::GpuVendor;

use super::{DataPathStatus, LibtorchStatus, NcclStatus, ProbeReport};

pub fn probe_local(
    ctx: &Context,
    skip_mount: bool,
    data_path_override: Option<PathBuf>,
    libtorch_path_override: Option<PathBuf>,
    via_docker: Option<String>,
    data_path_explicit: bool,
) -> ProbeReport {
    let host = resolve_local_hostname();
    let mut issues: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // The full sweep, not just its device list. A survey's findings are
    // the part a device list cannot express, and the case that matters
    // most for a second vendor has NO device at all: a card physically
    // present whose stack is not installed. `probe` exists to tell an
    // operator why a host is not ready, so it is the one command that
    // must never drop them.
    let sweep = flodl_hw::survey();
    for note in &sweep.notes {
        if note.kind.explains_absence() {
            issues.push(note.to_string());
        } else {
            warnings.push(note.to_string());
        }
    }
    // Read the vendor facts before `devices` is moved out.
    //
    // The NCCL scan looks for `libnccl.so`, an NVIDIA artifact, so it is
    // only meaningful when this host actually has an NVIDIA GPU. On an
    // AMD host the collective library is RCCL, which ships INSIDE
    // libtorch-rocm's own `lib/`; on a GPU-less host nothing collective
    // can run at all, and the "no usable GPUs" issue below already says
    // so. Either way "Install libnccl matching your CUDA version" points
    // the operator at the wrong thing.
    //
    // Note this reads the PHYSICAL sweep, not the masked one, so a rig
    // whose GPUs are temporarily hidden by CUDA_VISIBLE_DEVICES still
    // gets its NCCL install checked.
    let has_nvidia = sweep.has_vendor(GpuVendor::Nvidia);
    let gpus = sweep.devices;

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
    // The NCCL scan looks for `libnccl.so`, which is an NVIDIA artifact.
    // AMD's collective library is RCCL, and it ships INSIDE
    // libtorch-rocm's own `lib/` -- so on an AMD-only host there is
    // nothing to discover and a "libnccl not found" issue would be pure
    // noise telling the operator to install the wrong thing.
    //
    // The asymmetry is the distributions', not ours, and it is measured:
    // the published 2.10.0+rocm7.0 archive carries `lib/librccl.so`
    // (~340 MB), while the CUDA archives bundle no libnccl at all, which
    // is exactly why that one is worth probing for and this one is not.
    let nccl = if !has_nvidia {
        NcclStatus {
            library_path: None,
            all_found: vec![],
            via_docker: None,
        }
    } else {
        check_nccl(via_docker, &mut issues)
    };

    if gpus.is_empty() {
        // Say what was actually looked for. The old text named
        // nvidia-smi unconditionally, which is simply false on a host
        // whose GPU is AMD -- and that host is exactly the one whose
        // operator most needs an accurate message. Any vendor-specific
        // reason already rode in as a survey note above.
        issues.push(
            "no usable GPUs detected. Single-host CPU training will still \
             work; multi-rank training requires a working GPU stack."
                .into(),
        );
    }

    check_gpu_toolkit(libtorch.info.as_ref(), &mut warnings);

    // Host tools are a hard issue: without them `fdl` cannot download or
    // unpack anything, whatever the build strategy.
    let tools = requirements::missing_host_tools();
    if !tools.is_empty() {
        issues.push(format!(
            "missing host tools `fdl` needs: {}. Install with `sudo apt install {}` \
             (or the equivalent for your distribution).",
            tools.join(", "),
            tools.join(" "),
        ));
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
/// Report a variant the dynamic linker cannot satisfy on this host.
///
/// A libtorch archive is built against some baseline C library and the
/// baseline differs per variant: measured on 2.10.0, cpu and cu128 want
/// `GLIBC_2.29` while rocm7.0 wants `GLIBC_2.35`. RHEL 9 ships 2.34 and
/// cannot go further, so that pair compiles, links, and then dies in the
/// loader quoting symbol versions. Naming it here costs one `ldd`.
///
/// Called from every arm that produces a [`LibtorchStatus`]: the first
/// version of this check lived in one of them, and the explicit
/// `--libtorch-path` arm builds its status inline, so a real RHEL box
/// reported nothing at all.
pub(super) fn push_loader_issue(variant_dir: &Path, label: &str, issues: &mut Vec<String>) {
    let unmet = detect::unmet_loader_requirements(variant_dir);
    if unmet.is_empty() {
        return;
    }
    issues.push(format!(
        "libtorch variant `{label}` cannot load on this host: the dynamic \
         linker is missing {}. The archive was built against a newer C \
         library than this distribution ships, so it compiles and links and \
         then fails to start. Use a variant with an older baseline (cpu and \
         cu128 need less than the rocm archives) or a newer distribution.",
        unmet.join(", "),
    ));
}

pub(super) fn libtorch_status_from_info(
    info: Option<LibtorchInfo>,
    libtorch_root: &Path,
    gpus: &[GpuInfo],
    issues: &mut Vec<String>,
) -> LibtorchStatus {
    let valid_dir = match &info {
        Some(i) => libtorch_root.join(&i.path).join("lib").is_dir(),
        None => false,
    };
    if let Some(i) = &info {
        push_loader_issue(&libtorch_root.join(&i.path), &i.path, issues);
    }
    let archs_match = match &info {
        Some(i) => detect::arch_coverage(i, gpus, issues),
        None => {
            issues.push(
                "libtorch pointer file did not resolve to a configured \
                 variant (file empty or missing). Check the `.active*` \
                 content names a real subdir under `libtorch/`."
                    .into(),
            );
            Vec::new()
        }
    };
    LibtorchStatus {
        info,
        valid_dir,
        archs_match,
    }
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
///    `arch:` to a different case-file subpath (e.g. `.active.blackwell`).
/// 3. **Direct variant dir** (has `lib/libtorch.so` + optional
///    `.arch`) — used as-is.
pub(super) fn check_libtorch_at(
    path: &Path,
    gpus: &[GpuInfo],
    issues: &mut Vec<String>,
) -> LibtorchStatus {
    // Shape 2: a regular file whose name starts with `.active` is a
    // pointer to a variant subdir. Resolve relative to the file's
    // parent (the libtorch root). Note: `.active` itself is also a
    // file but Shape 1 catches it via dir-containing-.active above.
    if path.is_file()
        && path
            .file_name()
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
    let info = detect::libtorch_info_from_dir(dir.display().to_string(), dir);
    let archs_match = detect::arch_coverage(&info, gpus, issues);
    push_loader_issue(dir, &info.path, issues);
    LibtorchStatus {
        info: Some(info),
        valid_dir: true,
        archs_match,
    }
}

pub(super) fn check_libtorch(
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
                    let arch_dir = root.join(&variant);
                    Some(detect::libtorch_info_from_dir(variant, &arch_dir))
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

    let archs_match = match &info {
        Some(i) => detect::arch_coverage(i, gpus, issues),
        None => {
            issues.push(
                "libtorch not configured — `libtorch/.active` missing or \
                 empty. Run `fdl libtorch download` or `fdl libtorch build` \
                 to provision a variant."
                    .into(),
            );
            Vec::new()
        }
    };

    LibtorchStatus {
        info,
        valid_dir,
        archs_match,
    }
}

pub(super) fn check_data_path(
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

    DataPathStatus {
        path,
        exists,
        readable,
        fs_type,
        skipped: false,
    }
}

/// Report a missing vendor toolkit for the ACTIVE libtorch variant.
///
/// The active variant is what declares intent: `precompiled/rocm70` says
/// this project builds ROCm, so it will need HIP headers. That is the
/// same signal `$FDL_GPU_FEATURE` is derived from, so the two cannot
/// disagree about which vendor is in play.
///
/// Only headers are checked: libtorch bundles every library the link
/// needs, so headers are the whole gap.
///
/// A warning rather than an issue: the default workflow builds in the
/// dev container, where host headers are irrelevant. It applies to
/// native builds, and the text says so.
///
/// `flodl-sys/build.rs` guards the same requirement at compile time;
/// this reports it before a build is attempted.
pub(super) fn check_gpu_toolkit(info: Option<&LibtorchInfo>, warnings: &mut Vec<String>) {
    let Some(info) = info else { return };
    let Some(vendor) = detect::variant_vendor(&info.path) else {
        return; // CPU variant: no toolkit to want.
    };

    // `GpuVendor` is #[non_exhaustive] on purpose -- Intel is the planned
    // third. A vendor with no entry here has no known toolkit layout, and
    // guessing one would produce a confidently wrong apt command. Say
    // nothing until someone adds real facts.
    //
    // The header tables are `util::requirements`'s — the SAME set
    // flodl-sys/build.rs demands, covering the whole include chain. A
    // shorter hand-picked list here is the trap this replaced: probe
    // reports clean, the operator proceeds, and the build fails on a
    // header the short list never looked for. ROCm has no metapackage,
    // so its install line must name every package; `cuda-toolkit` IS a
    // metapackage, so the NVIDIA line stays that plus libnccl-dev
    // rather than version-placeholder package names.
    let plan = match vendor {
        GpuVendor::Amd => Some((
            "ROCM_PATH",
            flodl_hw::rocm_runtime_root()
                .map(|p| p.display().to_string())
                .or_else(|| std::env::var("ROCM_PATH").ok())
                .unwrap_or_else(|| "/opt/rocm".to_string()),
            crate::util::requirements::ROCM_HEADERS,
            None,
            "rocm",
        )),
        GpuVendor::Nvidia => Some((
            "CUDA_HOME",
            std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string()),
            crate::util::requirements::CUDA_HEADERS,
            Some("cuda-toolkit libnccl-dev"),
            "cuda",
        )),
        _ => None,
    };
    let Some((root_env, root, headers, metapackages, feature)) = plan else {
        return;
    };

    if let Some(w) = gpu_toolkit_warning(
        &info.path,
        Path::new(&root),
        root_env,
        headers,
        metapackages,
        feature,
    ) {
        warnings.push(w);
    }
}

/// Pure core of [`check_gpu_toolkit`]: the toolkit root is a parameter,
/// not an env read, so every arm is testable without mutating
/// process-global state. That matters more than usual here -- this
/// crate's test binary runs in parallel, and an env-mutating test only
/// works if every reader takes the same lock, which they do not.
///
/// `metapackages` overrides the per-header package list in the install
/// line, for the vendor whose metapackage covers the set.
pub(super) fn gpu_toolkit_warning(
    variant: &str,
    root: &Path,
    root_env: &str,
    headers: &[(&str, &str)],
    metapackages: Option<&str>,
    feature: &str,
) -> Option<String> {
    let missing = crate::util::requirements::missing_headers(root, headers);
    if missing.is_empty() {
        return None;
    }
    let packages: Vec<String> = match metapackages {
        Some(m) => m.split_whitespace().map(str::to_string).collect(),
        None => crate::util::requirements::packages_for(&missing),
    };
    let list: Vec<&str> = missing.iter().map(|(h, _)| *h).collect();
    let root = root.display();
    // Through `install_hint`, so the command names this family's package
    // manager and its own spelling of the packages. Hardcoding apt here
    // told a RHEL box to run `sudo apt install hip-dev`, which is two
    // kinds of wrong at once.
    let install = crate::util::requirements::install_hint(&packages);
    // No backticks around the install line: it carries its own trailing
    // caveat ("or your distribution's equivalent"), and quoting the pair
    // as one span invites a copy-paste that dnf rejects on the paren.
    Some(format!(
        "active libtorch is `{}` but its toolkit headers are missing under \
         `{root}` ({}). Native builds with `--features {feature}` will fail; \
         building in the dev container is unaffected. Install them with: \
         {install}. Set {root_env} if your install is elsewhere.",
        variant,
        list.join(", "),
    ))
}

pub(super) fn check_nccl(via_docker: Option<String>, issues: &mut Vec<String>) -> NcclStatus {
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

    NcclStatus {
        library_path: found.first().cloned(),
        all_found: found,
        via_docker: None,
    }
}

/// What is mounted AT `path` exactly: `(source, fs_type)` from
/// `/proc/mounts`, e.g. `("flodl@exa:/flodl/data", "fuse.sshfs")`.
/// `None` when `path` is not itself a mount point — which is how
/// [`crate::prepare`] tells "already mounted, nothing to do" from "mount
/// it now" without shelling out to `mountpoint(1)`. Contrast
/// [`detect_fs_type`], which walks toward the root and therefore always
/// answers something.
pub(crate) fn mounted_at(path: &Path) -> Option<(String, String)> {
    let mounts = std::fs::read_to_string("/proc/mounts").ok()?;
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    // Last match wins: a mount point can be stacked, and the effective
    // filesystem is the one mounted most recently.
    let mut found = None;
    for line in mounts.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() >= 3 && Path::new(cols[1]) == abs {
            found = Some((unescape_mount(cols[0]), cols[2].to_string()));
        }
    }
    found
}

/// `/proc/mounts` octal-escapes space, tab, newline and backslash in
/// the source and mount-point columns. Only the source is user-facing
/// here (it goes into a mismatch warning), and a path with a space in it
/// would otherwise print as `exa:/flodl\040data`.
pub(super) fn unescape_mount(field: &str) -> String {
    let mut out = String::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let digits: String = chars.clone().take(3).collect();
        match u8::from_str_radix(&digits, 8) {
            Ok(byte) if digits.len() == 3 => {
                out.push(byte as char);
                for _ in 0..3 {
                    chars.next();
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Best-effort filesystem-type lookup via `/proc/mounts`. Walks toward
/// the root looking for the closest mount-point that contains `path`.
/// Returns `None` on non-Linux or when `/proc/mounts` is unavailable.
pub(crate) fn detect_fs_type(path: &Path) -> Option<String> {
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
