//! Device identity and the top-level sweep entry points.

use std::collections::HashSet;

use crate::report::{GpuSurvey, NoteKind};
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
/// Vendor tools report every physical GPU regardless of the mask, so it
/// is applied here. `CUDA_VISIBLE_DEVICES=0,2` keeps indices 0 and 2; an
/// empty value returns nothing (libtorch's "explicitly no CUDA"); unset
/// keeps everything. This lets tests scope down via
/// `CUDA_VISIBLE_DEVICES=0 cargo test` and stops auto-promote
/// surprising the harness on a multi-GPU box.
///
/// A mask that removes devices leaves a [`NoteKind::MaskApplied`] note,
/// so a caller reporting "0 GPUs" can say whether that was the
/// operator's own doing.
pub fn survey_visible() -> GpuSurvey {
    let mut out = survey();
    let Ok(mask) = std::env::var("CUDA_VISIBLE_DEVICES") else {
        return out;
    };
    apply_visibility_mask(&mut out, &mask);
    out
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

/// Filter a survey in place by a `CUDA_VISIBLE_DEVICES` value. Split out
/// so mask semantics are testable without a GPU.
fn apply_visibility_mask(out: &mut GpuSurvey, mask: &str) {
    let trimmed = mask.trim();
    if trimmed.is_empty() {
        // Explicit "no CUDA": libtorch treats this as zero devices.
        if !out.devices.is_empty() {
            let n = out.devices.len();
            out.devices.clear();
            out.note(
                GpuVendor::Nvidia,
                NoteKind::MaskApplied,
                format!(
                    "CUDA_VISIBLE_DEVICES is set but empty, hiding all {n} device(s). \
                     Unset it to use them."
                ),
            );
        }
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
                    GpuVendor::Nvidia,
                    NoteKind::MaskApplied,
                    format!(
                        "CUDA_VISIBLE_DEVICES entry {entry:?} is not a numeric index \
                         (UUID / MIG forms are not resolved here); GPU detection may \
                         under-count."
                    ),
                );
            }
        }
    }
    let before = out.devices.len();
    out.devices.retain(|g| allowed.contains(&g.index));
    let hidden = before - out.devices.len();
    if hidden > 0 {
        out.note(
            GpuVendor::Nvidia,
            NoteKind::MaskApplied,
            format!("CUDA_VISIBLE_DEVICES={trimmed:?} hides {hidden} of {before} device(s)."),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env mutations must be serialized: cargo test runs in parallel and
    // `CUDA_VISIBLE_DEVICES` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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

    fn masked(devices: Vec<GpuInfo>, mask: &str) -> GpuSurvey {
        let mut s = GpuSurvey { devices, notes: vec![] };
        apply_visibility_mask(&mut s, mask);
        s
    }

    #[test]
    fn survey_never_panics_and_agrees_with_itself() {
        let _lock = ENV_LOCK.lock().unwrap();
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
        let _lock = ENV_LOCK.lock().unwrap();
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
    fn detect_gpus_physical_ignores_the_mask() {
        let _lock = ENV_LOCK.lock().unwrap();
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
