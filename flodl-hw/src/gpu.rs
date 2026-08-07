//! Device identity and the top-level sweep entry points.

use std::collections::HashSet;

use crate::report::{GpuSurvey, NoteKind, SurveyNote};
use crate::vendor::{GpuArch, GpuVendor};

/// One GPU's identity, capability and VRAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// The vendor tool's device index (0-based). Matches the index
    /// libtorch would assign with no visibility mask set. NVML and
    /// amd-smi queries must use this too: neither honors a mask.
    pub index: u8,
    /// Which stack this device belongs to. Separate from the device
    /// string libtorch is handed, which is `CUDA` for AMD as well as
    /// NVIDIA. See [`GpuVendor`].
    pub vendor: GpuVendor,
    /// Marketing name (e.g. `"NVIDIA GeForce RTX 5060 Ti"`).
    pub name: String,
    /// Architecture, in the vendor's own shape. See [`GpuArch`].
    pub arch: GpuArch,
    /// Total VRAM in MiB.
    ///
    /// On a unified-memory part (an APU whose GPU and system RAM are one
    /// pool) this is a slice of host RAM, not a separate budget. Nothing
    /// in flodl reads it that way yet, which is a known gap.
    pub total_memory_mb: u64,
}

impl GpuInfo {
    /// Vendor-neutral architecture label: `"sm_120"`, `"gfx1030"`.
    /// What display and report code wants.
    pub fn arch_label(&self) -> String {
        self.arch.to_string()
    }

    /// NVIDIA compute capability as `"sm_NN"`, or `None` on any other
    /// vendor. Use [`GpuInfo::arch_label`] for display; this is for the
    /// NVIDIA-specific consumers (nvcc gencode flags, CUDA variant
    /// naming) that genuinely cannot proceed without the numeric pair.
    pub fn sm_version(&self) -> Option<String> {
        matches!(self.arch, GpuArch::Sm { .. }).then(|| self.arch.to_string())
    }

    /// NVIDIA compute-capability major, `None` on any other vendor.
    pub fn sm_major(&self) -> Option<u32> {
        self.arch.sm_major()
    }

    /// NVIDIA compute-capability minor, `None` on any other vendor.
    pub fn sm_minor(&self) -> Option<u32> {
        self.arch.sm_minor()
    }

    /// Total VRAM in bytes.
    pub fn vram_bytes(&self) -> u64 {
        self.total_memory_mb * 1024 * 1024
    }

    /// `name` with the common vendor prefixes stripped, which is kinder
    /// on `eprintln!`-style banners.
    pub fn short_name(&self) -> String {
        self.name
            .replace("NVIDIA ", "")
            .replace("GeForce ", "")
            .replace("AMD ", "")
            .replace("Advanced Micro Devices, Inc. ", "")
    }

    /// Whether a libtorch variant compiled for `archs` covers this
    /// device. Delegates to [`GpuArch::covered_by`], whose matching
    /// rules are vendor-specific.
    pub fn covered_by(&self, archs: &str) -> bool {
        self.arch.covered_by(archs)
    }
}

/// Sweep the machine for GPUs, ignoring visibility masks.
///
/// The full report: every vendor probed, every finding recorded. This is
/// the **provisioning** view (which libtorch variant covers this box,
/// what to print in a hardware report), so a container mask must not
/// change its answer.
///
/// Probes are ordered cheap-first: each vendor is gated behind
/// subprocess-free filesystem checks, so a pure-NVIDIA box never spawns
/// an AMD tool and a CPU-only box spawns nothing at all.
///
/// [`crate::ENV_TESTING_GPU_JSON`] replaces the whole sweep when set,
/// which is how a second vendor's detection and routing get tested on a
/// machine that has none of that hardware.
pub fn survey() -> GpuSurvey {
    // Checked before any probe so a spoofed run never touches the real
    // hardware it is standing in for.
    if let Some(spoofed) = crate::testing::spoofed_survey() {
        return spoofed;
    }
    let mut out = GpuSurvey::default();
    crate::nvidia::probe(&mut out);
    crate::amd::probe(&mut out);
    out
}

/// Sweep the machine and apply the visibility masks, reporting what the
/// **runtime** will actually see.
///
/// Use for decisions that must agree with what libtorch would do: DDP
/// auto-promotion, mode filtering, log banners, CLI-flag validation.
///
/// # Masks
///
/// Detection is mask-proof by construction (vendor tools report every
/// physical GPU, and sysfs ignores masks entirely), so the mask is
/// applied here instead -- and it is **per vendor**, because the
/// vendors do not read the same variable.
///
/// | Vendor | Variable, in precedence order |
/// |---|---|
/// | NVIDIA | `CUDA_VISIBLE_DEVICES` |
/// | AMD | `HIP_VISIBLE_DEVICES`, then `ROCR_VISIBLE_DEVICES`, then `CUDA_VISIBLE_DEVICES` |
///
/// HIP honours all three and the first one *set* wins, even when it is
/// empty. Applying a single variable to every device would mis-count in
/// both directions on a mixed box: `HIP_VISIBLE_DEVICES=0` would hide
/// nothing, and `CUDA_VISIBLE_DEVICES=0` would wrongly override an AMD
/// mask that HIP itself would have preferred.
///
/// `0,2` keeps those indices. An empty value, or `-1`, returns nothing
/// (libtorch's "explicitly no devices", and HIP's convention for the
/// same). Unset keeps everything. This lets tests scope down via
/// `CUDA_VISIBLE_DEVICES=0 cargo test` and stops auto-promote
/// surprising the harness on a multi-GPU box.
///
/// A mask that removes devices leaves a [`NoteKind::MaskApplied`] note,
/// so a caller reporting "0 GPUs" can say whether that was the
/// operator's own doing.
pub fn survey_visible() -> GpuSurvey {
    let mut out = survey();
    apply_visibility_masks(&mut out);
    out
}

/// [`survey_visible`], narrowed to the one vendor a caller can actually
/// address.
///
/// The masks answer "what will the runtime see"; this answers the
/// separate question "which of those can *this build* talk to". libtorch
/// is built for exactly one GPU backend and both claim
/// `DeviceType::CUDA`, so on a mixed box the other vendor's devices are
/// present, healthy, and unusable. Counting them is not cosmetic: it
/// feeds the `>= 2` DDP auto-promote decision, which would then hand a
/// rank a device the build cannot talk to.
///
/// Filtering also disambiguates the index space. [`GpuInfo::index`] is
/// the *vendor tool's* ordinal, so an unfiltered survey of a box with
/// two NVIDIA cards and one AMD card carries indices `0, 1, 0` -- only
/// meaningful once split by vendor.
///
/// Dropped devices leave a [`NoteKind::VendorMismatch`] note, and it
/// counts as explaining an absence: a ROCm build on an NVIDIA-only box
/// reports zero devices, and the note is what turns that into "you built
/// for ROCm and this machine has NVIDIA hardware" instead of a bare
/// "no GPUs found".
pub fn survey_visible_for(vendor: GpuVendor) -> GpuSurvey {
    let mut out = survey_visible();
    retain_vendor(&mut out, vendor);
    out
}

/// Enumerate the visible devices this build can address. Shorthand for
/// [`survey_visible_for`] when the caller does not need the findings.
pub fn detect_gpus_for(vendor: GpuVendor) -> Vec<GpuInfo> {
    survey_visible_for(vendor).devices
}

/// Drop every device that is not `vendor`, in place. Split out so the
/// semantics are testable without hardware of either vendor.
fn retain_vendor(out: &mut GpuSurvey, vendor: GpuVendor) {
    let before = out.devices.len();
    out.devices.retain(|g| g.vendor == vendor);
    let dropped = before - out.devices.len();
    if dropped == 0 {
        return;
    }
    out.notes.push(SurveyNote {
        vendor,
        kind: NoteKind::VendorMismatch,
        message: format!(
            "{dropped} device(s) of another vendor are installed and ignored: \
             this build targets {vendor} and libtorch can only address one \
             GPU backend per process."
        ),
    });
}

/// The mask variable a vendor actually reads, and its value.
///
/// Returns the **first variable that is set**, not the first non-empty
/// one: an explicitly empty `HIP_VISIBLE_DEVICES` means "no devices"
/// and must not fall through to `CUDA_VISIBLE_DEVICES`.
fn mask_for(vendor: GpuVendor) -> Option<(&'static str, String)> {
    let order: &[&str] = match vendor {
        GpuVendor::Nvidia => &["CUDA_VISIBLE_DEVICES"],
        // HIP's documented precedence. `GPU_DEVICE_ORDINAL` also exists
        // and is not handled: it is an OpenCL-era selector that HIP
        // treats differently across versions, and guessing at it would
        // be worse than the loud under-count a caller can see.
        GpuVendor::Amd => &[
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ],
    };
    order
        .iter()
        .find_map(|k| std::env::var(k).ok().map(|v| (*k, v)))
}

/// Apply each vendor's own mask to its own devices.
fn apply_visibility_masks(out: &mut GpuSurvey) {
    for vendor in out.vendors() {
        if let Some((var, value)) = mask_for(vendor) {
            apply_visibility_mask(out, vendor, var, &value);
        }
    }
}

/// Enumerate the GPUs the runtime will see. Shorthand for
/// [`survey_visible`] when the caller does not need the findings.
pub fn detect_gpus() -> Vec<GpuInfo> {
    survey_visible().devices
}

/// Enumerate every installed GPU, ignoring visibility masks. Shorthand
/// for [`survey`] when the caller does not need the findings.
pub fn detect_gpus_physical() -> Vec<GpuInfo> {
    survey().devices
}

/// Filter one vendor's devices in place by a mask value. Split out so
/// mask semantics are testable without a GPU.
fn apply_visibility_mask(out: &mut GpuSurvey, vendor: GpuVendor, var: &str, mask: &str) {
    let trimmed = mask.trim();
    let before = out.devices.iter().filter(|g| g.vendor == vendor).count();
    if before == 0 {
        return;
    }

    // Empty and `-1` are the two spellings of "explicitly none": CUDA
    // treats an empty value as zero devices, HIP accepts `-1` for the
    // same, and both must beat "unset means everything".
    if trimmed.is_empty() || trimmed == "-1" {
        out.devices.retain(|g| g.vendor != vendor);
        out.note(
            vendor,
            NoteKind::MaskApplied,
            format!(
                "{var}={trimmed:?} hides all {before} {vendor} device(s). \
                 Unset it to use them."
            ),
        );
        return;
    }

    let mut allowed: HashSet<u8> = HashSet::new();
    for entry in trimmed.split(',') {
        let entry = entry.trim();
        match entry.parse::<u8>() {
            Ok(idx) => {
                allowed.insert(idx);
            }
            Err(_) => {
                // CUDA also accepts GPU-<uuid> / MIG-<...> forms this
                // index filter cannot resolve. Silently dropping them
                // reported "no GPUs" while libtorch would happily see
                // one, which is exactly the runtime divergence
                // detect_gpus exists to prevent.
                out.note(
                    vendor,
                    NoteKind::MaskApplied,
                    format!(
                        "{var} entry {entry:?} is not a numeric index \
                         (UUID / MIG forms are not resolved here); {vendor} device \
                         detection may under-count."
                    ),
                );
            }
        }
    }
    out.devices
        .retain(|g| g.vendor != vendor || allowed.contains(&g.index));
    let hidden = before - out.devices.iter().filter(|g| g.vendor == vendor).count();
    if hidden > 0 {
        out.note(
            vendor,
            NoteKind::MaskApplied,
            format!("{var}={trimmed:?} hides {hidden} of {before} {vendor} device(s)."),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env mutations must be serialized: cargo test runs in parallel and
    // these variables are process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Acquire the env lock, **recovering from poison**.
    ///
    /// A `#[should_panic]` test that holds this lock poisons it, and a
    /// plain `.unwrap()` then turns that one intentional panic into a
    /// `PoisonError` cascade across every sibling test -- which is
    /// exactly what happened here the moment
    /// `a_malformed_spoof_panics_rather_than_using_real_hardware`
    /// landed: nine failures, only one of them real. Each locker resets
    /// the variables it cares about via `EnvGuard`, and those guards
    /// still run their `Drop` during unwind, so recovering is safe.
    /// Same reasoning, and same fix, as `flodl_cli::util::test_env`.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// RAII helper that snapshots an env var on construction and
    /// restores it on drop. Pair with `ENV_LOCK`.
    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: `ENV_LOCK` serializes env mutations across tests
            // in this module, and nothing else in this crate reads
            // these vars concurrently.
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: as above.
            unsafe { std::env::remove_var(key) };
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: as above.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    const CVD: &str = "CUDA_VISIBLE_DEVICES";
    const SPOOF: &str = crate::ENV_TESTING_GPU_JSON;

    fn gpu(index: u8, major: u32, minor: u32) -> GpuInfo {
        GpuInfo {
            index,
            vendor: GpuVendor::Nvidia,
            name: format!("NVIDIA Test {index}"),
            arch: GpuArch::Sm { major, minor },
            total_memory_mb: 8192,
        }
    }

    // --- vendor filtering -------------------------------------------

    #[test]
    fn retain_vendor_keeps_only_that_vendors_devices() {
        let mut sur = GpuSurvey {
            devices: vec![gpu(0, 12, 0), amd(0, "gfx1036"), gpu(1, 12, 0)],
            ..Default::default()
        };
        retain_vendor(&mut sur, GpuVendor::Nvidia);
        assert_eq!(sur.devices.len(), 2);
        assert!(sur.devices.iter().all(|g| g.vendor == GpuVendor::Nvidia));
    }

    #[test]
    fn retain_vendor_notes_what_it_dropped_and_explains_absence() {
        // The case that matters: a ROCm build on an NVIDIA-only box.
        // Zero devices must not read as "no GPU installed".
        let mut sur = GpuSurvey {
            devices: vec![gpu(0, 12, 0)],
            ..Default::default()
        };
        retain_vendor(&mut sur, GpuVendor::Amd);
        assert!(sur.devices.is_empty());
        let note = sur
            .notes
            .iter()
            .find(|n| n.kind == NoteKind::VendorMismatch)
            .expect("a dropped device must leave a note");
        assert!(note.kind.explains_absence());
        assert!(note.message.contains('1'), "note should say how many");
    }

    #[test]
    fn retain_vendor_is_silent_when_nothing_is_dropped() {
        let mut sur = GpuSurvey {
            devices: vec![gpu(0, 12, 0)],
            ..Default::default()
        };
        retain_vendor(&mut sur, GpuVendor::Nvidia);
        assert_eq!(sur.devices.len(), 1);
        assert!(!sur.notes.iter().any(|n| n.kind == NoteKind::VendorMismatch));
    }

    #[test]
    fn vendor_filter_resolves_the_duplicate_index_space() {
        // Per-vendor ordinals mean an unfiltered mixed box carries
        // indices 0, 1, 0 -- ambiguous until split by vendor.
        let mut sur = GpuSurvey {
            devices: vec![gpu(0, 12, 0), gpu(1, 12, 0), amd(0, "gfx1036")],
            ..Default::default()
        };
        let all: Vec<u8> = sur.devices.iter().map(|g| g.index).collect();
        assert_eq!(all, vec![0, 1, 0], "precondition: indices collide");
        retain_vendor(&mut sur, GpuVendor::Amd);
        assert_eq!(
            sur.devices.iter().map(|g| g.index).collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn detect_gpus_for_filters_a_spoofed_mixed_host() {
        let _lock = env_lock();
        let _cvd = EnvGuard::unset(CVD);
        // A real shape: AMD APU alongside a discrete NVIDIA card.
        let _spoof = EnvGuard::set(
            SPOOF,
            r#"[{"vendor":"nvidia","arch":"sm_120","vram_mb":16384},
                {"vendor":"amd","arch":"gfx1036","vram_mb":512}]"#,
        );
        assert_eq!(detect_gpus().len(), 2, "unfiltered sees both vendors");
        assert_eq!(detect_gpus_for(GpuVendor::Nvidia).len(), 1);
        assert_eq!(detect_gpus_for(GpuVendor::Amd).len(), 1);
        // The bug this guards: a CUDA build must not count the APU
        // toward the >= 2 auto-promote threshold.
        assert!(
            detect_gpus_for(GpuVendor::Nvidia).len() < 2,
            "a CUDA build must not see 2 GPUs on this box"
        );
    }

    fn amd(index: u8, gfx: &str) -> GpuInfo {
        GpuInfo {
            index,
            vendor: GpuVendor::Amd,
            name: format!("AMD Test {index}"),
            arch: GpuArch::Gfx(gfx.into()),
            total_memory_mb: 16384,
        }
    }

    fn masked(devices: Vec<GpuInfo>, mask: &str) -> GpuSurvey {
        let mut s = GpuSurvey { devices, notes: vec![] };
        apply_visibility_mask(&mut s, GpuVendor::Nvidia, CVD, mask);
        s
    }

    #[test]
    fn survey_never_panics_and_agrees_with_itself() {
        let _lock = env_lock();
        let _g = EnvGuard::unset(CVD);
        let _s = EnvGuard::unset(SPOOF);
        // On CI without GPUs: empty. On a GPU box: parseable info.
        // Either is fine. Must NOT panic and must NOT touch libtorch.
        let s = survey();
        for g in &s.devices {
            assert!(!g.name.is_empty(), "name parsed");
            assert!(g.total_memory_mb > 0, "VRAM parsed");
            assert!(!g.arch_label().is_empty(), "arch rendered");
        }
        assert_eq!(s.devices.len(), detect_gpus_physical().len());
    }

    #[test]
    fn gpu_info_projects_identity_and_capacity() {
        let g = GpuInfo {
            index: 0,
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GeForce Test".into(),
            arch: GpuArch::Sm { major: 12, minor: 0 },
            total_memory_mb: 16000,
        };
        assert_eq!(g.arch_label(), "sm_120");
        assert_eq!(g.sm_version().as_deref(), Some("sm_120"));
        assert_eq!(g.sm_major(), Some(12));
        assert_eq!(g.short_name(), "Test");
        assert_eq!(g.vram_bytes(), 16000 * 1024 * 1024);
    }

    #[test]
    fn an_amd_device_has_no_sm_version() {
        // The whole point of Option here: a caller reaching for a
        // compute capability on a gfx part gets None, not a fabricated
        // pair that would silently produce a wrong gencode flag.
        let g = GpuInfo {
            index: 0,
            vendor: GpuVendor::Amd,
            name: "AMD Radeon RX 6800".into(),
            arch: GpuArch::Gfx("gfx1030".into()),
            total_memory_mb: 16384,
        };
        assert_eq!(g.arch_label(), "gfx1030");
        assert_eq!(g.sm_version(), None);
        assert_eq!(g.sm_major(), None);
        assert_eq!(g.short_name(), "Radeon RX 6800");
        assert!(g.covered_by("gfx1030;gfx1100"));
    }

    #[test]
    fn empty_mask_hides_everything_and_says_so() {
        let s = masked(vec![gpu(0, 8, 6), gpu(1, 8, 6)], "");
        assert!(s.devices.is_empty());
        assert_eq!(s.notes.len(), 1);
        assert_eq!(s.notes[0].kind, NoteKind::MaskApplied);
        // An operator-caused zero must not read as a hardware fault.
        assert!(s.require_devices().unwrap_err().contains("no GPUs detected"));
    }

    #[test]
    fn empty_mask_on_a_gpuless_box_is_not_worth_a_note() {
        let s = masked(vec![], "");
        assert!(s.notes.is_empty(), "nothing was hidden, so nothing to report");
    }

    #[test]
    fn mask_filters_by_index_and_reports_the_hidden_count() {
        let s = masked(vec![gpu(0, 8, 6), gpu(1, 6, 1), gpu(2, 8, 6)], "0,2");
        assert_eq!(s.devices.iter().map(|g| g.index).collect::<Vec<_>>(), vec![0, 2]);
        assert!(s.notes.iter().any(|n| n.message.contains("hides 1 of 3")));
    }

    #[test]
    fn a_full_mask_is_silent() {
        // Listing every device is not a hazard worth a note.
        let s = masked(vec![gpu(0, 8, 6), gpu(1, 8, 6)], "0,1");
        assert_eq!(s.devices.len(), 2);
        assert!(s.notes.is_empty());
    }

    #[test]
    fn mask_tolerates_whitespace_and_ignores_unknown_indices() {
        let s = masked(vec![gpu(0, 8, 6), gpu(1, 8, 6)], " 1 , 99 ");
        assert_eq!(s.devices.iter().map(|g| g.index).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn mask_drops_uuid_forms_rather_than_inventing_devices() {
        let s = masked(vec![gpu(0, 8, 6)], "GPU-deadbeef");
        assert!(s.devices.is_empty());
        assert!(s.notes.iter().any(|n| n.message.contains("not a numeric index")));
    }

    #[test]
    fn detect_gpus_honors_the_live_mask() {
        let _lock = env_lock();
        let _s = EnvGuard::unset(SPOOF);
        let _g_unset = EnvGuard::unset(CVD);
        let physical = detect_gpus();
        if physical.is_empty() {
            return; // No GPUs on this box: nothing to filter.
        }
        let pick = physical[0].index;
        drop(_g_unset);
        let _g_set = EnvGuard::set(CVD, &pick.to_string());
        let filtered = detect_gpus();
        assert_eq!(filtered.len(), 1, "single-index filter narrows to one");
        assert_eq!(filtered[0].index, pick);
    }

    #[test]
    fn each_vendor_is_filtered_by_its_own_mask() {
        // A single mask applied to every device mis-counts in BOTH
        // directions on a mixed box, which is why the filter is
        // per-vendor.
        let mut s = GpuSurvey {
            devices: vec![gpu(0, 8, 6), gpu(1, 8, 6), amd(0, "gfx1030"), amd(1, "gfx1100")],
            notes: vec![],
        };
        apply_visibility_mask(&mut s, GpuVendor::Nvidia, CVD, "1");
        apply_visibility_mask(&mut s, GpuVendor::Amd, "HIP_VISIBLE_DEVICES", "0");
        let kept: Vec<(GpuVendor, u8)> =
            s.devices.iter().map(|g| (g.vendor, g.index)).collect();
        assert_eq!(kept, vec![(GpuVendor::Nvidia, 1), (GpuVendor::Amd, 0)]);
    }

    #[test]
    fn a_mask_for_one_vendor_leaves_the_other_alone() {
        let mut s = GpuSurvey {
            devices: vec![gpu(0, 8, 6), amd(0, "gfx1030")],
            notes: vec![],
        };
        apply_visibility_mask(&mut s, GpuVendor::Amd, "HIP_VISIBLE_DEVICES", "");
        assert_eq!(s.devices.len(), 1, "the NVIDIA device survives an AMD mask");
        assert_eq!(s.devices[0].vendor, GpuVendor::Nvidia);
    }

    #[test]
    fn minus_one_means_none_for_hip() {
        let mut s = GpuSurvey { devices: vec![amd(0, "gfx1030")], notes: vec![] };
        apply_visibility_mask(&mut s, GpuVendor::Amd, "HIP_VISIBLE_DEVICES", "-1");
        assert!(s.devices.is_empty());
        assert!(s.notes[0].message.contains("hides all 1"), "{:?}", s.notes);
    }

    #[test]
    fn masking_a_vendor_with_no_devices_is_silent() {
        // An AMD mask exported on a pure-NVIDIA box must not produce a
        // note about zero AMD devices.
        let mut s = GpuSurvey { devices: vec![gpu(0, 8, 6)], notes: vec![] };
        apply_visibility_mask(&mut s, GpuVendor::Amd, "HIP_VISIBLE_DEVICES", "");
        assert_eq!(s.devices.len(), 1);
        assert!(s.notes.is_empty());
    }

    #[test]
    fn hip_mask_precedence_prefers_the_first_variable_that_is_set() {
        let _lock = env_lock();
        let _c = EnvGuard::set(CVD, "9");
        let _r = EnvGuard::set("ROCR_VISIBLE_DEVICES", "5");
        {
            let _h = EnvGuard::set("HIP_VISIBLE_DEVICES", "1");
            assert_eq!(mask_for(GpuVendor::Amd).unwrap().0, "HIP_VISIBLE_DEVICES");
            // NVIDIA never reads the HIP variables.
            assert_eq!(
                mask_for(GpuVendor::Nvidia).unwrap(),
                ("CUDA_VISIBLE_DEVICES", "9".to_string()),
            );
        }
        let _h = EnvGuard::unset("HIP_VISIBLE_DEVICES");
        assert_eq!(mask_for(GpuVendor::Amd).unwrap().0, "ROCR_VISIBLE_DEVICES");
        let _r2 = EnvGuard::unset("ROCR_VISIBLE_DEVICES");
        assert_eq!(mask_for(GpuVendor::Amd).unwrap().0, "CUDA_VISIBLE_DEVICES");
    }

    #[test]
    fn an_empty_hip_mask_does_not_fall_through_to_cuda() {
        // First variable SET wins, not first non-empty: an explicitly
        // empty HIP_VISIBLE_DEVICES means "no AMD devices", and falling
        // through to CUDA_VISIBLE_DEVICES would silently un-hide them.
        let _lock = env_lock();
        let _c = EnvGuard::set(CVD, "0");
        let _h = EnvGuard::set("HIP_VISIBLE_DEVICES", "");
        assert_eq!(
            mask_for(GpuVendor::Amd).unwrap(),
            ("HIP_VISIBLE_DEVICES", String::new()),
        );
    }

    // --- the FLODL_TESTING_GPU_JSON injection point -------------------
    //
    // These four were reported as landed in P1 and were not: the edit
    // anchored on a function that had already moved to nvidia.rs, so
    // the replace silently no-op'd and the test count still rose from
    // testing.rs. Asserting on every anchor is now the rule.

    #[test]
    fn spoof_replaces_the_whole_sweep() {
        let _lock = env_lock();
        let _cvd = EnvGuard::unset(CVD);
        let _s = EnvGuard::set(
            SPOOF,
            r#"[{"vendor":"amd","arch":"gfx1030","vram_mb":16384},
                {"vendor":"amd","arch":"gfx1100","vram_mb":24576}]"#,
        );
        let s = survey();
        assert_eq!(s.devices.len(), 2, "spoof stands in for real hardware");
        assert!(s.devices.iter().all(|g| g.vendor == GpuVendor::Amd));
        // True on the NVIDIA dev rig too: the spoof is checked before
        // any probe runs, so the real cards are never consulted.
        assert!(!s.has_vendor(GpuVendor::Nvidia));
        assert_eq!(detect_gpus_physical().len(), 2);
    }

    #[test]
    fn spoof_composes_with_the_visibility_mask() {
        // The spoof replaces the HARDWARE, not the mask policy, so the
        // two layer. The docs promise this.
        let _lock = env_lock();
        let _s = EnvGuard::set(
            SPOOF,
            r#"[{"arch":"sm_86"},{"arch":"sm_86"},{"arch":"sm_86"},{"arch":"sm_86"}]"#,
        );
        let _cvd = EnvGuard::set(CVD, "2");
        assert_eq!(detect_gpus_physical().len(), 4, "physical view ignores the mask");
        let visible = detect_gpus();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].index, 2);
    }

    #[test]
    fn a_spoofed_amd_device_obeys_the_hip_mask_not_the_cuda_one() {
        // Ties the two halves of P2 together: spoofed AMD hardware,
        // filtered by HIP's variable while CUDA_VISIBLE_DEVICES says
        // something else entirely.
        let _lock = env_lock();
        let _s = EnvGuard::set(
            SPOOF,
            r#"[{"vendor":"amd","arch":"gfx1030"},{"vendor":"amd","arch":"gfx1100"}]"#,
        );
        let _cvd = EnvGuard::set(CVD, "0,1");
        let _hip = EnvGuard::set("HIP_VISIBLE_DEVICES", "1");
        let visible = detect_gpus();
        assert_eq!(visible.len(), 1, "HIP_VISIBLE_DEVICES wins over CUDA_VISIBLE_DEVICES");
        assert_eq!(visible[0].arch, GpuArch::Gfx("gfx1100".into()));
    }

    #[test]
    fn an_empty_spoof_falls_through_to_real_detection() {
        // Exporting the var as "" is how a shell unsets-in-practice;
        // treating it as an empty device list would silently claim the
        // box has no GPUs.
        let _lock = env_lock();
        let _cvd = EnvGuard::unset(CVD);
        let real = {
            let _s = EnvGuard::unset(SPOOF);
            detect_gpus_physical().len()
        };
        let _s = EnvGuard::set(SPOOF, "   ");
        assert_eq!(detect_gpus_physical().len(), real);
    }

    #[test]
    #[should_panic(expected = "could not be parsed")]
    fn a_malformed_spoof_panics_rather_than_using_real_hardware() {
        let _lock = env_lock();
        let _s = EnvGuard::set(SPOOF, "{ not json");
        let _ = survey();
    }

    #[test]
    fn detect_gpus_physical_ignores_the_mask() {
        let _lock = env_lock();
        let _s = EnvGuard::unset(SPOOF);
        let _g_unset = EnvGuard::unset(CVD);
        let physical = detect_gpus_physical();
        drop(_g_unset);
        // An empty mask zeroes the runtime view but must not touch the
        // physical one: the whole reason the two are named apart.
        let _g_set = EnvGuard::set(CVD, "");
        assert!(detect_gpus().is_empty());
        assert_eq!(detect_gpus_physical().len(), physical.len());
    }
}
