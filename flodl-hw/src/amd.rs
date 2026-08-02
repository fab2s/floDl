//! The AMD backend: the kernel's KFD topology.
//!
//! # Why sysfs and not `amd-smi`
//!
//! The ROCm bring-up spec puts `rocminfo` / `amd-smi --json` at the
//! centre of AMD detection, behind a `/dev/kfd` gate. This backend
//! inverts that: **the KFD topology is the whole source, and no
//! subprocess runs at all.**
//!
//! Two reasons, and the second is the load-bearing one.
//!
//! `/sys/class/kfd/kfd/topology/nodes/*/properties` is defined by the
//! `amdgpu` driver, so its shape can be read off a running kernel and
//! reasoned about. `amd-smi`'s JSON is *shape-unstable* across versions
//! (the spec lists six distinct ways it has spelled a memory figure),
//! and we have no AMD box under ROCm to capture real output from --
//! the fixtures would be written against believed shapes. A parser
//! validated only against its author's assumptions is not a foundation
//! for the primary detection path. It is deferred until a real capture
//! exists, at which point it is worth adding for one thing only: the
//! marketing name, which is a display string and cannot break routing.
//!
//! Sysfs is also **mask-proof** (the kernel does not honour
//! `HIP_VISIBLE_DEVICES`) and free: zero process spawns, where the
//! NVIDIA path costs one.
//!
//! # The usability gate
//!
//! A KFD node is **not** the same as a device flodl can train on, and
//! conflating them is dangerous rather than merely inaccurate: a Ryzen
//! desktop chip exposes its integrated Radeon through KFD on any Linux
//! box with `amdgpu` loaded, ROCm installed or not. Counting that as a
//! device makes `Trainer::run` auto-promote (`detect_gpus() >= 2`) and
//! `--gpus all` spawn a rank on hardware that cannot run the workload.
//!
//! So an AMD GPU is reported as a **device** only when a ROCm userspace
//! runtime is findable; otherwise it becomes a
//! [`NoteKind::HardwareUnusable`] finding naming the arch and what to
//! install. The asymmetry is deliberate: a false positive breaks a
//! working rig, while a false negative degrades to fewer devices and
//! says why, which the operator can act on.
//!
//! # Provenance
//!
//! Field names, the `gfx_target_version` encoding and the CPU node's
//! `vendor_id 0` were read off a live KFD topology (a Ryzen host whose
//! Raphael iGPU reports `gfx_target_version 100306` = `gfx1036`), not
//! inferred. One assumption that reading *killed*: a node's `name` file
//! does **not** hold the ASIC or gfx name. On that host it reads
//! `"ip discovery"`, so names are synthesized from the arch instead.

use std::path::{Path, PathBuf};

use crate::report::{GpuSurvey, NoteKind};
use crate::vendor::{GpuArch, GpuVendor};
use crate::GpuInfo;

/// `vendor_id` a KFD node reports for AMD silicon (0x1002).
///
/// Unambiguous as a GPU-presence test: the KFD **CPU** node reports 0
/// (verified on a live topology), and NVIDIA's open kernel module
/// registers KFD nodes as 4318 (0x10DE).
const AMD_VENDOR_ID: u64 = 4098;

/// PCI vendor id for AMD, as written in `/sys/bus/pci/devices/*/vendor`.
const AMD_PCI_VENDOR: &str = "0x1002";

/// Probe AMD devices, appending to `out`.
pub(crate) fn probe(out: &mut GpuSurvey) {
    probe_at(
        Path::new("/sys"),
        Path::new("/dev/kfd"),
        rocm_runtime_root().as_deref(),
        out,
    );
}

/// [`probe`] with its filesystem roots injected, so every branch is
/// testable against a synthetic sysfs tree.
fn probe_at(sys: &Path, kfd_dev: &Path, rocm: Option<&Path>, out: &mut GpuSurvey) {
    let nodes_dir = sys.join("class/kfd/kfd/topology/nodes");
    let gpus = read_topology(&nodes_dir);

    if gpus.is_empty() {
        // No KFD GPU node. Either there is no AMD GPU, or the driver
        // is not loaded / the device node was not passed into this
        // container. PCI can tell those apart, and the difference is
        // the whole message.
        let pci = amd_display_devices(&sys.join("bus/pci/devices"));
        if !pci.is_empty() {
            out.note(
                GpuVendor::Amd,
                NoteKind::HardwareUnusable,
                format!(
                    "{} AMD display device(s) on PCI ({}) but no usable KFD GPU node. \
                     Load the `amdgpu` kernel module, or (in a container) pass the \
                     devices through with `--device=/dev/kfd --device=/dev/dri`.",
                    pci.len(),
                    pci.join(", "),
                ),
            );
        }
        return;
    }

    // The driver enumerated GPUs but the character device is missing:
    // the classic container mistake, where /sys is bind-mounted from
    // the host but /dev/kfd was never passed in. Worth its own message
    // because the hardware is visibly *there*.
    if !kfd_dev.exists() {
        out.note(
            GpuVendor::Amd,
            NoteKind::HardwareUnusable,
            format!(
                "the kernel lists {} AMD GPU(s) but {} does not exist, so no process \
                 can open them. In a container, add `--device=/dev/kfd \
                 --device=/dev/dri`; otherwise check the `amdgpu` module and the \
                 `render`/`video` group membership.",
                gpus.len(),
                kfd_dev.display(),
            ),
        );
        return;
    }

    let Some(rocm) = rocm else {
        // Hardware present, userspace absent. NOT a device: see the
        // usability gate in the module docs.
        let archs: Vec<String> = gpus.iter().map(|g| g.arch.to_string()).collect();
        out.note(
            GpuVendor::Amd,
            NoteKind::HardwareUnusable,
            format!(
                "{} AMD GPU(s) present ({}) but no ROCm runtime was found \
                 (`libhsa-runtime64.so` under $ROCM_PATH, $HIP_PATH, $HSA_PATH or \
                 /opt/rocm). They are not counted as usable devices. Install ROCm \
                 to train on them.",
                gpus.len(),
                archs.join(", "),
            ),
        );
        return;
    };
    let _ = rocm; // presence is the signal; the path itself matters to P5's loader ordering

    // The device node exists but this process cannot open it. On a
    // cloud host with ROCm installed natively, this is THE common
    // stumble: `/dev/kfd` is `crw-rw---- root render`, the sysfs
    // topology is world-readable, and cloud images routinely leave the
    // default user outside `render`. Everything else then looks
    // healthy -- the kernel lists the GPU, the runtime is installed --
    // and libtorch dies at device init with a permission error that
    // never mentions groups.
    //
    // Docker hides this (our compose service sets `group_add: video,
    // render`), which is exactly why it only surfaces on the native
    // path.
    //
    // Checked AFTER the ROCm gate on purpose. A box with neither ROCm
    // nor group membership should be told to install ROCm first: that
    // is the larger prerequisite, and the ROCm installer commonly adds
    // the render group itself. Reporting groups first would send the
    // operator down the wrong path and cost an extra round trip.
    //
    // `/dev/kfd` stands in for a PAIR: ROCm also needs the DRM render
    // node (`/dev/dri/renderD*`), which is why the compose service maps
    // `/dev/dri` as well. Both are `root:render`, so one membership
    // decides both and checking either answers the PERMISSION question
    // for both.
    //
    // It does NOT answer the render node's PRESENCE question: a
    // container given `--device=/dev/kfd` but not `--device=/dev/dri`
    // passes this check and still cannot run. Deliberately not checked
    // yet -- writing it safely needs a real container-vs-host device
    // layout to test against, the same "no capture, no parser" rule
    // this module applies to `amd-smi`. It is on the rental-capture
    // list.
    if device_rw_access(kfd_dev) == Some(false) {
        let archs: Vec<String> = gpus.iter().map(|g| g.arch.to_string()).collect();
        out.note(
            GpuVendor::Amd,
            NoteKind::HardwareUnusable,
            format!(
                "{} AMD GPU(s) present ({}) and {} exists, but this process cannot \
                 open it -- almost always missing `render`/`video` group membership. \
                 Fix with `sudo usermod -aG render,video $USER` then log out and back \
                 in (group changes do not apply to existing sessions). In a container, \
                 add `--group-add video --group-add render`.",
                gpus.len(),
                archs.join(", "),
                kfd_dev.display(),
            ),
        );
        return;
    }

    out.devices.extend(gpus);
}

/// Read every AMD GPU node from a KFD topology directory, in node
/// order, assigning contiguous device indices.
///
/// Node numbering is not device numbering: node 0 is the CPU on every
/// system observed, and a node is a GPU only when it reports AMD's
/// vendor id **and** a non-zero `simd_count`. HIP indexes GPUs in node
/// order, so enumerating the filtered, sorted list reproduces it.
fn read_topology(nodes_dir: &Path) -> Vec<GpuInfo> {
    let Ok(entries) = std::fs::read_dir(nodes_dir) else {
        return Vec::new();
    };
    // Node directories are named by number; sort numerically, not
    // lexically, or node 10 would sort before node 2.
    let mut dirs: Vec<(u64, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_str()?.parse::<u64>().ok()?;
            Some((n, e.path()))
        })
        .collect();
    dirs.sort_by_key(|(n, _)| *n);

    let mut out = Vec::new();
    for (_, dir) in dirs {
        if let Some(gpu) = read_node(&dir, out.len()) {
            out.push(gpu);
        }
    }
    out
}

/// Parse one KFD node directory into a device, or `None` when it is not
/// an AMD GPU.
fn read_node(dir: &Path, index: usize) -> Option<GpuInfo> {
    let props = std::fs::read_to_string(dir.join("properties")).ok()?;

    if prop(&props, "vendor_id")? != AMD_VENDOR_ID {
        return None;
    }
    // The CPU node carries AMD's vendor id on an AMD host but has no
    // shader engines. `simd_count > 0` is what separates a GPU from it.
    if prop(&props, "simd_count").unwrap_or(0) == 0 {
        return None;
    }

    let arch = gfx_from_target_version(prop(&props, "gfx_target_version").unwrap_or(0))
        .and_then(|token| GpuArch::parse(GpuVendor::Amd, &token))?;

    // `name` is NOT the ASIC name (a live Raphael node reads
    // "ip discovery"), so synthesize from the arch. A marketing name
    // needs amd-smi, deferred until real output can be captured.
    let name = format!("AMD GPU {arch}");

    Some(GpuInfo {
        index: u8::try_from(index).ok()?,
        vendor: GpuVendor::Amd,
        name,
        arch,
        total_memory_mb: node_memory_mb(dir),
    })
}

/// Total memory visible to a node, in MiB, from its `mem_banks`.
///
/// Takes the **largest** bank rather than filtering on `heap_type`: the
/// heap-type enum is not something we can verify without more hardware,
/// and on every layout the framebuffer (or, on an APU, the carve-out
/// from system RAM) is the dominant bank. `0` when nothing parses,
/// which is honest -- every consumer of this figure is advisory
/// (banners, reports, an ElChe tiebreak), so a missing number is
/// strictly better than a guessed one.
fn node_memory_mb(dir: &Path) -> u64 {
    let Ok(banks) = std::fs::read_dir(dir.join("mem_banks")) else {
        return 0;
    };
    banks
        .flatten()
        .filter_map(|b| std::fs::read_to_string(b.path().join("properties")).ok())
        .filter_map(|text| prop(&text, "size_in_bytes"))
        .max()
        .unwrap_or(0)
        / (1024 * 1024)
}

/// Read one `key value` line out of a KFD `properties` file.
///
/// Matches on the whitespace-separated key, never a substring. The
/// distinction is load-bearing: a detector that looks for a `gpu_id`
/// *line* in this file never matches, because `gpu_id` is a **sibling
/// file** rather than a property. The failure is silent -- it reports
/// no GPU at all, on every host.
fn prop(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        (parts.next()? == key).then(|| parts.next()?.parse().ok())?
    })
}

/// Render `gfx_target_version` as its gfx token.
///
/// The encoding is `major * 10000 + minor * 100 + step`, with the step
/// in **hex** so `gfx90a` round-trips. Verified against a live node
/// reporting `100306` on a Raphael iGPU, which is `gfx1036`.
///
/// `0` means the kernel did not publish one (pre-5.16 kernels, and the
/// CPU node), which is `None` rather than a fabricated arch.
fn gfx_from_target_version(v: u64) -> Option<String> {
    if v == 0 {
        return None;
    }
    let (major, minor, step) = (v / 10000, (v / 100) % 100, v % 100);
    Some(format!("gfx{major}{minor}{step:x}"))
}

/// AMD display-class devices on the PCI bus, as `<slot> (<device id>)`.
///
/// The sharpening the spec calls for when ROCm cannot use the card at
/// all: `vendor == 0x1002` and a `class` of `0x03....`. Distinguishes
/// "no AMD GPU here" from "an AMD GPU whose driver never loaded".
fn amd_display_devices(pci_dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(pci_dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            let vendor = std::fs::read_to_string(p.join("vendor")).ok()?;
            if vendor.trim() != AMD_PCI_VENDOR {
                return None;
            }
            let class = std::fs::read_to_string(p.join("class")).ok()?;
            if !class.trim().starts_with("0x03") {
                return None;
            }
            let device = std::fs::read_to_string(p.join("device"))
                .map(|d| d.trim().to_string())
                .unwrap_or_default();
            Some(format!("{} {device}", e.file_name().to_string_lossy()))
        })
        .collect();
    out.sort();
    out
}

/// Locate a ROCm userspace installation, or `None`.
///
/// Resolution order is the spec's, and the ordering is the point:
/// `$ROCM_PATH` / `$HIP_PATH` / `$HSA_PATH` first, `/opt/rocm` **only**
/// as a fallback, so a stale `/opt/rocm` cannot shadow the install that
/// actually matches the driver.
///
/// A candidate counts only when it actually contains
/// `libhsa-runtime64.so{,.1}`, probing both `lib` and `lib64`. An empty
/// directory named by `$ROCM_PATH` is not an installation.
///
/// P5 (the `LD_LIBRARY_PATH` ordering fix) needs exactly these
/// directories, which is why the resolution lives here rather than in
/// the caller.
pub fn rocm_runtime_root() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = ["ROCM_PATH", "HIP_PATH", "HSA_PATH"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .collect();
    candidates.push(PathBuf::from("/opt/rocm"));
    rocm_runtime_root_from(&candidates)
}

/// [`rocm_runtime_root`] over an explicit candidate list, for tests.
fn rocm_runtime_root_from(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates
        .iter()
        .find(|root| {
            ["lib", "lib64"].iter().any(|libdir| {
                ["libhsa-runtime64.so", "libhsa-runtime64.so.1"]
                    .iter()
                    .any(|so| root.join(libdir).join(so).exists())
            })
        })
        .cloned()
}

/// Whether this process could open `dev` for read/write, decided from
/// the node's mode and our own credentials rather than by opening it.
///
/// Deliberately does NOT open the device: `/dev/kfd` is the handle the
/// ROCm runtime attaches through, and hardware *detection* should stay
/// side-effect-free. Credentials come from `/proc/self/status`, which
/// keeps this crate dependency-free (no libc for `geteuid`/`getgroups`).
///
/// `None` = cannot tell (no `/proc`, unreadable node, non-unix). Callers
/// must treat that as "no opinion" and stay quiet: a false "your groups
/// are wrong" on a working box is worse than silence.
#[cfg(unix)]
fn device_rw_access(dev: &Path) -> Option<bool> {
    use std::os::unix::fs::MetadataExt;
    let md = std::fs::metadata(dev).ok()?;
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let euid = status_id_field(&status, "Uid:")?;
    let egid = status_id_field(&status, "Gid:")?;
    let groups = status_groups(&status)?;
    Some(mode_grants_rw(
        md.mode(),
        md.uid(),
        md.gid(),
        euid,
        egid,
        &groups,
    ))
}

#[cfg(not(unix))]
fn device_rw_access(_dev: &Path) -> Option<bool> {
    None
}

/// Effective id from a `/proc/self/status` `Uid:`/`Gid:` line, whose
/// fields are `real effective saved filesystem` -- the EFFECTIVE one
/// (index 1) is what the kernel checks on open.
#[cfg(any(unix, test))]
fn status_id_field(status: &str, label: &str) -> Option<u32> {
    status
        .lines()
        .find_map(|l| l.strip_prefix(label))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Supplementary gids from the `Groups:` line (may legitimately be empty).
#[cfg(any(unix, test))]
fn status_groups(status: &str) -> Option<Vec<u32>> {
    Some(
        status
            .lines()
            .find_map(|l| l.strip_prefix("Groups:"))?
            .split_whitespace()
            .filter_map(|g| g.parse().ok())
            .collect(),
    )
}

/// The kernel's permission check for opening a node read/write, as a
/// pure function so every branch is testable without a device node.
///
/// Order matters and mirrors the kernel's: root bypasses; otherwise the
/// OWNER bits apply if we own it (even when they grant *less* than the
/// group bits), then group, then other.
#[cfg(any(unix, test))]
fn mode_grants_rw(
    mode: u32,
    owner_uid: u32,
    owner_gid: u32,
    euid: u32,
    egid: u32,
    groups: &[u32],
) -> bool {
    const RW: u32 = 0o6;
    if euid == 0 {
        return true; // CAP_DAC_OVERRIDE in practice
    }
    if euid == owner_uid {
        return (mode >> 6) & RW == RW;
    }
    if egid == owner_gid || groups.contains(&owner_gid) {
        return (mode >> 3) & RW == RW;
    }
    mode & RW == RW
}

#[cfg(test)]
mod tests {
    // --- /dev/kfd accessibility --------------------------------------

    // The real shape on a Linux host: crw-rw---- root:render.
    const KFD_MODE: u32 = 0o660;
    const ROOT: u32 = 0;
    const RENDER_GID: u32 = 104;

    #[test]
    fn kfd_is_open_to_a_member_of_the_render_group() {
        assert!(mode_grants_rw(KFD_MODE, ROOT, RENDER_GID, 1000, 1000, &[44, RENDER_GID]));
    }

    #[test]
    fn kfd_is_closed_to_a_user_outside_the_render_group() {
        // THE cloud-host case: everything else looks healthy, this is
        // the only thing wrong, and the resulting libtorch error never
        // mentions groups.
        assert!(!mode_grants_rw(KFD_MODE, ROOT, RENDER_GID, 1000, 1000, &[44, 100]));
    }

    #[test]
    fn root_opens_it_regardless_of_groups() {
        assert!(mode_grants_rw(KFD_MODE, ROOT, RENDER_GID, 0, 0, &[]));
    }

    #[test]
    fn the_effective_gid_counts_as_membership() {
        // Supplementary list empty, but our primary group IS render.
        assert!(mode_grants_rw(KFD_MODE, ROOT, RENDER_GID, 1000, RENDER_GID, &[]));
    }

    #[test]
    fn owner_bits_apply_to_the_owner_even_when_group_bits_grant_more() {
        // Kernel order, not "most permissive wins": we own it, so the
        // OWNER bits decide, and here they grant nothing.
        assert!(!mode_grants_rw(0o060, 1000, RENDER_GID, 1000, 1000, &[RENDER_GID]));
    }

    #[test]
    fn world_writable_node_is_open_to_anyone() {
        assert!(mode_grants_rw(0o666, ROOT, ROOT, 1000, 1000, &[]));
    }

    #[test]
    fn read_only_group_access_is_not_enough() {
        // KFD needs read/write; r-- would fail at open.
        assert!(!mode_grants_rw(0o440, ROOT, RENDER_GID, 1000, 1000, &[RENDER_GID]));
    }

    #[test]
    fn parses_effective_ids_and_groups_from_proc_status() {
        // Fields are `real effective saved filesystem`; the EFFECTIVE
        // one is what the kernel checks, so index 1 not 0.
        let status = "Name:\tx\nUid:\t1000\t1001\t1000\t1000\nGid:\t1000\t1002\t1000\t1000\nGroups:\t4 24 104 \n";
        assert_eq!(status_id_field(status, "Uid:"), Some(1001));
        assert_eq!(status_id_field(status, "Gid:"), Some(1002));
        assert_eq!(status_groups(status), Some(vec![4, 24, 104]));
    }

    #[test]
    fn an_empty_groups_line_is_empty_not_missing() {
        // A process with no supplementary groups is normal; it must not
        // read as "cannot tell" and silence the check.
        assert_eq!(status_groups("Groups:\t\n"), Some(vec![]));
        assert_eq!(status_groups("Name:\tx\n"), None);
    }

    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// Hand-rolled scratch dir + RAII cleanup. flodl-hw takes no
    /// dev-dependencies, so no `tempfile`.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("flodl-hw-amd-{nanos}-{seq}"));
            fs::create_dir_all(&dir).expect("scratch");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Write a KFD node. `props` is the verbatim `properties` body.
    fn node(sys: &Path, n: u64, props: &str, bank_bytes: Option<u64>) {
        let dir = sys.join("class/kfd/kfd/topology/nodes").join(n.to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("properties"), props).unwrap();
        // `gpu_id` is a SIBLING FILE, not a properties line. Present in
        // every fixture so a detector that confuses the two is caught.
        fs::write(dir.join("gpu_id"), "6720\n").unwrap();
        // Nor is `name` the ASIC name; a live Raphael node reads this.
        fs::write(dir.join("name"), "ip discovery\n").unwrap();
        if let Some(bytes) = bank_bytes {
            let bank = dir.join("mem_banks/0");
            fs::create_dir_all(&bank).unwrap();
            fs::write(
                bank.join("properties"),
                format!("heap_type 1\nsize_in_bytes {bytes}\nflags 0\nwidth 128\n"),
            )
            .unwrap();
        }
    }

    /// The CPU node, verbatim from a live 24-thread Ryzen host.
    const CPU_NODE: &str = "cpu_cores_count 24\nsimd_count 0\ngfx_target_version 0\n\
                            vendor_id 0\ndevice_id 0\nlocation_id 0\n";

    /// A discrete-GPU node in the shape a live Raphael node reports.
    fn gpu_node_props(gfx_target: u64, simd: u64) -> String {
        format!(
            "cpu_cores_count 0\nsimd_count {simd}\nmax_waves_per_simd 32\n\
             wave_front_size 32\ngfx_target_version {gfx_target}\n\
             vendor_id 4098\ndevice_id 5056\nlocation_id 3328\n"
        )
    }

    /// Build one `/sys/bus/pci/devices/<slot>/` entry.
    ///
    /// `#[cfg(unix)]`, along with its three callers, because of the
    /// fixture rather than the code under test: a PCI slot is spelled
    /// `0000:0d:00.0`, and a colon is not legal in a Windows filename,
    /// so `create_dir_all` fails with `ERROR_INVALID_NAME` before any
    /// assertion runs.
    ///
    /// A colon-free slot name would keep the tests running without
    /// testing anything: `amd_display_devices` puts the directory name
    /// into the user-facing note, and these tests assert on it.
    ///
    /// Skipping them off-unix loses no coverage. `probe` reads
    /// `Path::new("/sys")`, so the KFD and PCI paths are inert there.
    #[cfg(unix)]
    fn pci_device(sys: &Path, slot: &str, vendor: &str, class: &str, device: &str) {
        let d = sys.join("bus/pci/devices").join(slot);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("vendor"), format!("{vendor}\n")).unwrap();
        fs::write(d.join("class"), format!("{class}\n")).unwrap();
        fs::write(d.join("device"), format!("{device}\n")).unwrap();
    }

    /// A ROCm install that satisfies the runtime probe.
    fn rocm_install(root: &Path, libdir: &str, soname: &str) -> PathBuf {
        let lib = root.join("rocm").join(libdir);
        fs::create_dir_all(&lib).unwrap();
        fs::write(lib.join(soname), "").unwrap();
        root.join("rocm")
    }

    // --- gfx_target_version -------------------------------------------

    #[test]
    fn decodes_gfx_target_version() {
        // 100306 is the value a live Raphael iGPU reports; the rest are
        // the arch families the ROCm spec enumerates.
        for (v, want) in [
            (100306, "gfx1036"),
            (90000, "gfx900"),
            (90006, "gfx906"),
            (90008, "gfx908"),
            (90402, "gfx942"),
            (100300, "gfx1030"),
            (110000, "gfx1100"),
            (110003, "gfx1103"),
            (110500, "gfx1150"),
            (110501, "gfx1151"),
            (120001, "gfx1201"),
        ] {
            assert_eq!(gfx_from_target_version(v).as_deref(), Some(want), "{v}");
        }
    }

    #[test]
    fn step_renders_in_hex_so_gfx90a_round_trips() {
        // The one case that proves the step is hex, not decimal:
        // gfx90a is major 9, minor 0, step 10.
        assert_eq!(gfx_from_target_version(90010).as_deref(), Some("gfx90a"));
    }

    #[test]
    fn absent_target_version_is_none_not_a_fabricated_arch() {
        // Pre-5.16 kernels and the CPU node both report 0. Inventing an
        // arch here would make the device compare as incompatible with
        // every libtorch variant, reading as a hardware fault.
        assert_eq!(gfx_from_target_version(0), None);
    }

    // --- properties parsing --------------------------------------------

    #[test]
    fn prop_matches_whole_keys_only() {
        let text = "cpu_cores_count 24\nsimd_count 0\nvendor_id 4098\n";
        assert_eq!(prop(text, "vendor_id"), Some(4098));
        assert_eq!(prop(text, "simd_count"), Some(0));
        // Substring matches must not resolve: `count` and `id` are
        // suffixes of real keys here.
        assert_eq!(prop(text, "count"), None);
        assert_eq!(prop(text, "id"), None);
        assert_eq!(prop(text, "missing"), None);
    }

    #[test]
    fn prop_does_not_find_gpu_id_which_is_a_sibling_file() {
        // `gpu_id` is a file next to `properties`, never a line inside
        // it. A detector that required the line would silently report
        // "no GPU" on every host, so this guards the whole-key match.
        assert_eq!(prop(&gpu_node_props(100306, 4), "gpu_id"), None);
    }

    // --- topology reading ----------------------------------------------

    #[test]
    fn skips_the_cpu_node_and_indexes_gpus_in_node_order() {
        let s = Scratch::new();
        node(s.path(), 0, CPU_NODE, None);
        node(s.path(), 1, &gpu_node_props(100300, 4), Some(16106430464));
        node(s.path(), 2, &gpu_node_props(110000, 8), Some(25769803776));
        let gpus = read_topology(&s.path().join("class/kfd/kfd/topology/nodes"));
        assert_eq!(gpus.len(), 2, "CPU node is not a GPU");
        assert_eq!(gpus[0].index, 0);
        assert_eq!(gpus[0].arch, GpuArch::Gfx("gfx1030".into()));
        assert_eq!(gpus[0].total_memory_mb, 15360);
        assert_eq!(gpus[1].index, 1);
        assert_eq!(gpus[1].arch, GpuArch::Gfx("gfx1100".into()));
    }

    #[test]
    fn node_dirs_sort_numerically_not_lexically() {
        // Node 10 must come after node 2, or device indices scramble on
        // any box with more than nine nodes.
        let s = Scratch::new();
        node(s.path(), 0, CPU_NODE, None);
        node(s.path(), 2, &gpu_node_props(100300, 4), None);
        node(s.path(), 10, &gpu_node_props(110000, 4), None);
        let gpus = read_topology(&s.path().join("class/kfd/kfd/topology/nodes"));
        assert_eq!(
            gpus.iter().map(|g| g.arch.to_string()).collect::<Vec<_>>(),
            vec!["gfx1030", "gfx1100"],
        );
    }

    #[test]
    fn an_nvidia_kfd_node_is_not_an_amd_gpu() {
        // NVIDIA's open kernel module (driver 560+) registers KFD nodes
        // too, as vendor 4318 (0x10DE).
        let s = Scratch::new();
        let props = gpu_node_props(100300, 4).replace("vendor_id 4098", "vendor_id 4318");
        node(s.path(), 1, &props, None);
        assert!(read_topology(&s.path().join("class/kfd/kfd/topology/nodes")).is_empty());
    }

    #[test]
    fn a_node_with_no_target_version_is_skipped_not_guessed() {
        let s = Scratch::new();
        node(s.path(), 1, &gpu_node_props(0, 4), None);
        assert!(read_topology(&s.path().join("class/kfd/kfd/topology/nodes")).is_empty());
    }

    #[test]
    fn missing_mem_banks_reports_zero_rather_than_guessing() {
        let s = Scratch::new();
        node(s.path(), 1, &gpu_node_props(100300, 4), None);
        let gpus = read_topology(&s.path().join("class/kfd/kfd/topology/nodes"));
        assert_eq!(gpus[0].total_memory_mb, 0);
    }

    #[test]
    fn absent_topology_is_empty_not_an_error() {
        assert!(read_topology(Path::new("/nonexistent/kfd/nodes")).is_empty());
    }

    // --- the usability gate --------------------------------------------

    #[test]
    fn gpu_without_rocm_is_a_finding_not_a_device() {
        // The state this dev box is actually in: a Ryzen iGPU visible
        // through KFD with no ROCm installed. Counting it would make
        // auto-promote spawn a rank on hardware that cannot train.
        let s = Scratch::new();
        node(s.path(), 0, CPU_NODE, None);
        node(s.path(), 1, &gpu_node_props(100306, 4), Some(16106430464));
        fs::write(s.path().join("kfd-dev"), "").unwrap();

        let mut out = GpuSurvey::default();
        probe_at(s.path(), &s.path().join("kfd-dev"), None, &mut out);

        assert!(out.devices.is_empty(), "must not report an unusable device");
        assert_eq!(out.notes.len(), 1);
        assert_eq!(out.notes[0].kind, NoteKind::HardwareUnusable);
        assert!(out.notes[0].message.contains("gfx1036"), "names the arch");
        assert!(out.notes[0].message.contains("Install ROCm"), "says what to do");
    }

    #[test]
    fn gpu_with_rocm_is_a_device() {
        let s = Scratch::new();
        node(s.path(), 0, CPU_NODE, None);
        node(s.path(), 1, &gpu_node_props(100300, 4), Some(17179869184));
        fs::write(s.path().join("kfd-dev"), "").unwrap();
        let rocm = rocm_install(s.path(), "lib", "libhsa-runtime64.so.1");

        let mut out = GpuSurvey::default();
        probe_at(s.path(), &s.path().join("kfd-dev"), Some(&rocm), &mut out);

        assert_eq!(out.devices.len(), 1);
        assert_eq!(out.devices[0].vendor, GpuVendor::Amd);
        assert_eq!(out.devices[0].arch, GpuArch::Gfx("gfx1030".into()));
        assert_eq!(out.devices[0].total_memory_mb, 16384);
        assert!(out.notes.is_empty(), "a healthy box has nothing to report");
    }

    #[test]
    fn sysfs_without_the_device_node_names_the_container_mistake() {
        // /sys bind-mounted from the host but /dev/kfd never passed in.
        let s = Scratch::new();
        node(s.path(), 1, &gpu_node_props(100300, 4), None);
        let rocm = rocm_install(s.path(), "lib", "libhsa-runtime64.so");

        let mut out = GpuSurvey::default();
        probe_at(s.path(), &s.path().join("absent"), Some(&rocm), &mut out);

        assert!(out.devices.is_empty());
        assert!(out.notes[0].message.contains("--device=/dev/kfd"), "{:?}", out.notes);
    }

    // Unix-only: see `pci_device` -- a PCI slot name contains colons,
    // which Windows rejects as a filename.
    #[cfg(unix)]
    #[test]
    fn pci_sharpens_the_no_kfd_case() {
        // No KFD node at all, but an AMD display device on the bus:
        // the driver never loaded. Distinct from "no AMD GPU here",
        // which must stay silent.
        let s = Scratch::new();
        pci_device(s.path(), "0000:0d:00.0", "0x1002", "0x030000", "0x13c0");
        let mut out = GpuSurvey::default();
        probe_at(s.path(), Path::new("/nonexistent"), None, &mut out);
        assert!(out.devices.is_empty());
        assert_eq!(out.notes.len(), 1);
        assert!(out.notes[0].message.contains("amdgpu"), "{:?}", out.notes);
        assert!(out.notes[0].message.contains("0000:0d:00.0"), "{:?}", out.notes);
    }

    // Unix-only: see `pci_device` -- a PCI slot name contains colons,
    // which Windows rejects as a filename.
    #[cfg(unix)]
    #[test]
    fn a_pure_nvidia_box_says_nothing_about_amd() {
        // The dev rig, minus its iGPU: NVIDIA display devices only. An
        // AMD note here would be pure noise on every NVIDIA host.
        let s = Scratch::new();
        pci_device(s.path(), "0000:01:00.0", "0x10de", "0x030000", "0x2d04");
        pci_device(s.path(), "0000:05:00.0", "0x10de", "0x030000", "0x1c03");
        let mut out = GpuSurvey::default();
        probe_at(s.path(), Path::new("/nonexistent"), None, &mut out);
        assert!(out.devices.is_empty());
        assert!(out.notes.is_empty(), "silent on a box with no AMD hardware");
    }

    // Unix-only: see `pci_device` -- a PCI slot name contains colons,
    // which Windows rejects as a filename.
    #[cfg(unix)]
    #[test]
    fn a_non_display_amd_device_is_not_a_gpu() {
        // Every AMD host has AMD-vendor chipset functions on the bus.
        // Only class 0x03 (display) counts.
        let s = Scratch::new();
        pci_device(s.path(), "0000:00:00.0", "0x1002", "0x060000", "0x1480");
        let mut out = GpuSurvey::default();
        probe_at(s.path(), Path::new("/nonexistent"), None, &mut out);
        assert!(out.notes.is_empty());
    }

    // --- ROCm runtime resolution ---------------------------------------

    #[test]
    fn runtime_root_requires_the_runtime_library() {
        // An empty directory named by $ROCM_PATH is not an install.
        let s = Scratch::new();
        let bare = s.path().join("bare");
        fs::create_dir_all(bare.join("lib")).unwrap();
        assert_eq!(rocm_runtime_root_from(&[bare]), None);
    }

    #[test]
    fn runtime_root_probes_lib_and_lib64_and_both_sonames() {
        for (libdir, soname) in [
            ("lib", "libhsa-runtime64.so"),
            ("lib", "libhsa-runtime64.so.1"),
            ("lib64", "libhsa-runtime64.so"),
            ("lib64", "libhsa-runtime64.so.1"),
        ] {
            let s = Scratch::new();
            let root = rocm_install(s.path(), libdir, soname);
            assert_eq!(
                rocm_runtime_root_from(std::slice::from_ref(&root)),
                Some(root),
                "{libdir}/{soname}"
            );
        }
    }

    #[test]
    fn an_earlier_candidate_wins_over_opt_rocm() {
        // Ordering is the point: a stale /opt/rocm must not shadow the
        // install that actually matches the driver.
        let s = Scratch::new();
        let explicit = rocm_install(&s.path().join("explicit"), "lib", "libhsa-runtime64.so");
        let stale = rocm_install(&s.path().join("stale"), "lib", "libhsa-runtime64.so");
        assert_eq!(
            rocm_runtime_root_from(&[explicit.clone(), stale]),
            Some(explicit),
        );
    }

    #[test]
    fn no_candidates_is_none() {
        assert_eq!(rocm_runtime_root_from(&[]), None);
        assert_eq!(rocm_runtime_root_from(&[PathBuf::from("/nonexistent")]), None);
    }
}
