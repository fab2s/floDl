//! The whole-machine GPU sweep: what is here, and what is wrong with it.

use std::fmt;

use crate::vendor::GpuVendor;
use crate::GpuInfo;

/// Why a survey has something to say beyond its device list.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteKind {
    /// Hardware is physically present but the stack cannot use it. The
    /// actionable case: an AMD card with no ROCm installed, an NVIDIA
    /// driver with no `nvidia-smi`.
    HardwareUnusable,
    /// A vendor tool was found and run, and it failed.
    ToolFailed,
    /// A device was dropped because its enumeration row did not parse.
    Unparsable,
    /// A visibility mask changed the answer, or could not be resolved.
    MaskApplied,
    /// A device was dropped because this build cannot address its
    /// vendor. libtorch is built for exactly one GPU backend, so on a
    /// mixed box the devices of the *other* vendor are present, healthy,
    /// and unusable -- counting them would hand a rank a device the
    /// build cannot talk to.
    VendorMismatch,
}

impl NoteKind {
    /// Every name [`NoteKind::parse`] accepts, comma-separated. For
    /// error messages, so the list cannot drift from the parser.
    pub const ALL_NAMES: &'static str =
        "hardware_unusable, tool_failed, unparsable, mask_applied, vendor_mismatch";

    /// Stable snake_case token.
    pub fn as_str(self) -> &'static str {
        match self {
            NoteKind::HardwareUnusable => "hardware_unusable",
            NoteKind::ToolFailed => "tool_failed",
            NoteKind::Unparsable => "unparsable",
            NoteKind::MaskApplied => "mask_applied",
            NoteKind::VendorMismatch => "vendor_mismatch",
        }
    }

    /// Parse the token produced by [`NoteKind::as_str`].
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hardware_unusable" => Some(NoteKind::HardwareUnusable),
            "tool_failed" => Some(NoteKind::ToolFailed),
            "unparsable" => Some(NoteKind::Unparsable),
            "mask_applied" => Some(NoteKind::MaskApplied),
            "vendor_mismatch" => Some(NoteKind::VendorMismatch),
            _ => None,
        }
    }

    /// Whether this note explains an *absence* of usable devices, as
    /// opposed to merely annotating the ones found. Drives whether
    /// [`GpuSurvey::require_devices`] quotes it.
    pub fn explains_absence(self) -> bool {
        matches!(
            self,
            NoteKind::HardwareUnusable
                | NoteKind::ToolFailed
                | NoteKind::Unparsable
                | NoteKind::VendorMismatch
        )
    }
}

/// One finding from a sweep: a fact the device list cannot express.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurveyNote {
    pub vendor: GpuVendor,
    pub kind: NoteKind,
    /// Human-facing and actionable. Callers print this verbatim, so it
    /// says what to do, not just what happened.
    pub message: String,
}

impl fmt::Display for SurveyNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.vendor, self.message)
    }
}

/// The result of sweeping a machine for GPUs.
///
/// The reason this is not just a `Vec<GpuInfo>`: an empty list has at
/// least four distinct causes, and they need different responses.
///
/// | Situation | `devices` | `notes` |
/// |---|---|---|
/// | CPU-only box | empty | empty |
/// | driver present, tool broken | empty | [`NoteKind::ToolFailed`] |
/// | card present, stack not installed | empty | [`NoteKind::HardwareUnusable`] |
/// | masked away | empty | [`NoteKind::MaskApplied`] |
/// | working rig | populated | empty |
///
/// Detection therefore *returns* its findings instead of printing them,
/// and the caller decides what to surface. `fdl probe` prints every
/// note; a library auto-promote check reads `devices.len()` and ignores
/// them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuSurvey {
    /// Usable devices, in vendor-probe order.
    pub devices: Vec<GpuInfo>,
    /// Everything the sweep learned that `devices` cannot express.
    pub notes: Vec<SurveyNote>,
}

impl GpuSurvey {
    /// Record a finding.
    pub(crate) fn note(&mut self, vendor: GpuVendor, kind: NoteKind, message: String) {
        self.notes.push(SurveyNote { vendor, kind, message });
    }

    /// Devices, or a caller-facing explanation of why there are none.
    ///
    /// This is what a command with an explicit GPU request wants
    /// (`--gpus all`, a cluster host declaring `devices: all`): those
    /// must fail loudly rather than quietly resolve to zero. The error
    /// quotes whichever notes explain the absence, and falls back to a
    /// plain "none found" when the sweep genuinely saw nothing, which is
    /// the honest message for a CPU-only box.
    pub fn require_devices(&self) -> Result<&[GpuInfo], String> {
        if !self.devices.is_empty() {
            return Ok(&self.devices);
        }
        let reasons: Vec<&str> = self
            .notes
            .iter()
            .filter(|n| n.kind.explains_absence())
            .map(|n| n.message.as_str())
            .collect();
        if reasons.is_empty() {
            return Err(
                "no GPUs detected on this host (no NVIDIA or AMD device found). \
                 Install a GPU driver, or select devices explicitly."
                    .to_string(),
            );
        }
        Err(format!("no usable GPUs detected: {}", reasons.join("; ")))
    }

    /// Whether any device of `vendor` was found.
    pub fn has_vendor(&self, vendor: GpuVendor) -> bool {
        self.devices.iter().any(|g| g.vendor == vendor)
    }

    /// The distinct vendors present, in first-seen order. A mixed box
    /// (NVIDIA plus AMD in one chassis) returns both, which is a
    /// configuration flodl reports rather than silently picking from.
    pub fn vendors(&self) -> Vec<GpuVendor> {
        let mut out: Vec<GpuVendor> = Vec::new();
        for g in &self.devices {
            if !out.contains(&g.vendor) {
                out.push(g.vendor);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vendor::GpuArch;

    fn dev(index: u8, vendor: GpuVendor, arch: GpuArch) -> GpuInfo {
        GpuInfo {
            index,
            vendor,
            name: format!("Test {index}"),
            arch,
            total_memory_mb: 8192,
        }
    }

    fn sm(major: u32, minor: u32) -> GpuArch {
        GpuArch::Sm { major, minor }
    }

    #[test]
    fn require_devices_passes_through_a_populated_sweep() {
        let s = GpuSurvey {
            devices: vec![dev(0, GpuVendor::Nvidia, sm(8, 6))],
            notes: vec![],
        };
        assert_eq!(s.require_devices().unwrap().len(), 1);
    }

    #[test]
    fn empty_sweep_says_no_gpus_rather_than_inventing_a_cause() {
        // A CPU-only box is not an error condition with a hidden reason.
        let err = GpuSurvey::default().require_devices().unwrap_err();
        assert!(err.contains("no GPUs detected"), "got: {err}");
    }

    #[test]
    fn absence_quotes_the_note_that_explains_it() {
        let mut s = GpuSurvey::default();
        s.note(
            GpuVendor::Amd,
            NoteKind::HardwareUnusable,
            "an AMD GPU is present but ROCm is not installed".into(),
        );
        let err = s.require_devices().unwrap_err();
        assert!(err.contains("ROCm is not installed"), "got: {err}");
    }

    #[test]
    fn a_mask_note_alone_does_not_masquerade_as_a_hardware_fault() {
        // Masking to zero is the operator's own doing. Reporting it as a
        // missing driver would send them chasing the wrong thing.
        let mut s = GpuSurvey::default();
        s.note(GpuVendor::Nvidia, NoteKind::MaskApplied, "hidden by mask".into());
        let err = s.require_devices().unwrap_err();
        assert!(err.contains("no GPUs detected"), "got: {err}");
        assert!(!err.contains("hidden by mask"), "got: {err}");
    }

    #[test]
    fn reports_vendors_present_without_deduping_away_a_mixed_box() {
        let s = GpuSurvey {
            devices: vec![
                dev(0, GpuVendor::Nvidia, sm(12, 0)),
                dev(1, GpuVendor::Amd, GpuArch::Gfx("gfx1030".into())),
                dev(2, GpuVendor::Nvidia, sm(6, 1)),
            ],
            notes: vec![],
        };
        assert_eq!(s.vendors(), vec![GpuVendor::Nvidia, GpuVendor::Amd]);
        assert!(s.has_vendor(GpuVendor::Amd));
    }

    #[test]
    fn has_vendor_is_false_when_only_a_note_mentions_it() {
        // A note about unusable AMD hardware must not read as "we have
        // an AMD GPU": that is exactly the distinction the type exists
        // to keep.
        let mut s = GpuSurvey {
            devices: vec![dev(0, GpuVendor::Nvidia, sm(8, 6))],
            notes: vec![],
        };
        s.note(GpuVendor::Amd, NoteKind::HardwareUnusable, "no ROCm".into());
        assert!(!s.has_vendor(GpuVendor::Amd));
        assert_eq!(s.vendors(), vec![GpuVendor::Nvidia]);
    }
}
