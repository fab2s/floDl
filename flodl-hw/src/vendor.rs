//! GPU vendor identity and the vendor-shaped architecture token.

use std::fmt;

/// Which GPU stack a device belongs to.
///
/// This is an **identity**, deliberately separate from the device string
/// a tensor library is handed. ROCm libtorch keeps `kCUDA`, the
/// `c10::cuda` namespaces, and RCCL exports the NCCL symbol names, so an
/// AMD device is still addressed as CUDA at the API surface while being
/// `Amd` here. Vendor drives detection, diagnostics, packaging and
/// feature derivation; the API surface is a different axis.
///
/// `#[non_exhaustive]`: Intel is the next entry, and it is *not* a
/// free-rider on the CUDA API surface the way AMD is (libtorch has a
/// genuinely distinct `XPU` device type), so adding it will be a real
/// change at every match site rather than a table row. Matches inside
/// this crate stay exhaustive, which is the point: the compiler
/// enumerates the work.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Amd,
}

impl GpuVendor {
    /// Lowercase stable token, as written into `.arch` metadata and
    /// cluster-probe JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
        }
    }

    /// Parse the token produced by [`GpuVendor::as_str`]. Case- and
    /// whitespace-insensitive; also accepts the stack names users type
    /// (`cuda`, `rocm`, `hip`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nvidia" | "cuda" => Some(GpuVendor::Nvidia),
            "amd" | "rocm" | "hip" => Some(GpuVendor::Amd),
            _ => None,
        }
    }

    /// The cargo feature that selects this vendor's libtorch link set.
    pub fn cargo_feature(self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "cuda",
            GpuVendor::Amd => "rocm",
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GpuVendor::Nvidia => "NVIDIA",
            GpuVendor::Amd => "AMD",
        })
    }
}

/// What a libtorch variant label says about its backend.
///
/// [`Cpu`](VariantClass::Cpu) and [`Unknown`](VariantClass::Unknown) are
/// separate on purpose: a CPU variant is a positive statement (this
/// build has no GPU backend), an unrecognized name says nothing — and
/// the two deserve different policies at every consumer (fdl warns and
/// assumes on Unknown; admission gates on neither).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantClass {
    /// `cpu` / `cpu-*`: a build with no GPU backend.
    Cpu,
    /// A recognized vendor naming (`cu<N>` / `sm<N>` → NVIDIA,
    /// `rocm<N>` / `gfx<N>` → AMD).
    Vendor(GpuVendor),
    /// A name outside the convention (including an empty label).
    Unknown,
}

/// Classify a libtorch variant label (`precompiled/cu128`,
/// `builds/sm61-sm120`, `rocm70`, …) by its basename.
///
/// This is the variant NAMING CONVENTION's single home; policy stays
/// with the callers (fdl's variant router warns and assumes NVIDIA on
/// [`VariantClass::Unknown`], the join admission gate treats it as
/// unclassifiable and lets it pass). A vendor prefix counts only when a
/// digit follows, so `cpu` cannot be read as a `cu`-something and a
/// stray directory cannot masquerade.
pub fn classify_variant_label(label: &str) -> VariantClass {
    let basename = std::path::Path::new(label)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let tagged = |prefix: &str| {
        basename
            .strip_prefix(prefix)
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    };
    if basename == "cpu" || basename.starts_with("cpu-") {
        return VariantClass::Cpu;
    }
    if tagged("cu") || tagged("sm") {
        return VariantClass::Vendor(GpuVendor::Nvidia);
    }
    if tagged("rocm") || tagged("gfx") {
        return VariantClass::Vendor(GpuVendor::Amd);
    }
    VariantClass::Unknown
}

/// A device's architecture token, in whatever shape its vendor uses.
///
/// Not flattened to a string: NVIDIA's numeric pair is load-bearing
/// (nvcc gencode flags, the min/max span logic that picks a libtorch
/// variant, the major-only compatibility fallback), and stringifying it
/// would force a re-parse at each of those. Not flattened to a numeric
/// pair either: `gfx1030` is not one.
///
/// `#[non_exhaustive]` for the same reason as [`GpuVendor`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuArch {
    /// NVIDIA compute capability. Renders as `sm_120`.
    Sm { major: u32, minor: u32 },
    /// AMD LLVM target. Renders as `gfx1030`.
    ///
    /// Always stored bare: the `:sramecc±:xnack±` feature suffix that
    /// `rocminfo` and `gcnArchName` append is stripped at parse, because
    /// every downstream comparison wants the bare token.
    Gfx(String),
}

impl GpuArch {
    /// Parse a vendor-appropriate arch token.
    ///
    /// - `Nvidia`: `"12.0"` or `"sm_120"` / `"sm120"`.
    /// - `Amd`: `"gfx1030"`, or `"gfx906:sramecc-:xnack-"` (suffix dropped).
    ///
    /// `None` when the token does not fit the vendor's shape, which is a
    /// real condition worth surfacing rather than defaulting through.
    pub fn parse(vendor: GpuVendor, token: &str) -> Option<Self> {
        let t = token.trim();
        match vendor {
            GpuVendor::Nvidia => Self::parse_sm(t),
            GpuVendor::Amd => {
                // Feature suffixes are colon-separated and never part of
                // the identity. Lowercase: rocminfo has shipped both cases.
                let bare = t.split(':').next()?.trim().to_ascii_lowercase();
                if !bare.starts_with("gfx") || bare.len() <= 3 {
                    return None;
                }
                Some(GpuArch::Gfx(bare))
            }
        }
    }

    /// Parse an NVIDIA capability in any of the forms that appear across
    /// nvidia-smi output, `.arch` metadata and cluster-probe JSON.
    ///
    /// `"8.6"` is the canonical `major.minor`. `"sm_86"` / `"sm86"` are
    /// the concatenated forms: the LAST digit is the minor, because the
    /// major grew past one digit at Blackwell (`sm_120` is 12.0, not
    /// 1.20).
    ///
    /// A `+PTX` suffix is dropped: `TORCH_CUDA_ARCH_LIST` accepts
    /// `"8.6+PTX"`, `fdl libtorch build --archs` passes that list through
    /// verbatim, and the resulting `.arch` line has to keep matching.
    fn parse_sm(t: &str) -> Option<Self> {
        let t = t.trim().split('+').next()?.trim();
        if let Some((maj, min)) = t.split_once('.') {
            return Some(GpuArch::Sm {
                major: maj.trim().parse().ok()?,
                minor: min.trim().parse().ok()?,
            });
        }
        let digits = t
            .trim_start_matches("sm_")
            .trim_start_matches("sm")
            .trim();
        if digits.len() < 2 || !digits.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let (maj, min) = digits.split_at(digits.len() - 1);
        Some(GpuArch::Sm {
            major: maj.parse().ok()?,
            minor: min.parse().ok()?,
        })
    }

    /// The vendor this arch shape belongs to.
    pub fn vendor(&self) -> GpuVendor {
        match self {
            GpuArch::Sm { .. } => GpuVendor::Nvidia,
            GpuArch::Gfx(_) => GpuVendor::Amd,
        }
    }

    /// NVIDIA compute-capability major, or `None` on a non-NVIDIA arch.
    /// For nvcc gencode flags and the variant-span logic; display code
    /// wants [`GpuArch`]'s `Display` instead.
    pub fn sm_major(&self) -> Option<u32> {
        match self {
            GpuArch::Sm { major, .. } => Some(*major),
            _ => None,
        }
    }

    /// NVIDIA compute-capability minor, or `None` on a non-NVIDIA arch.
    pub fn sm_minor(&self) -> Option<u32> {
        match self {
            GpuArch::Sm { minor, .. } => Some(*minor),
            _ => None,
        }
    }

    /// A monotonically-increasing number ordering this arch against
    /// others **of the same vendor**: newer hardware scores higher.
    /// `sm_86` is 86, `sm_120` is 120, `gfx1030` is 1030.
    ///
    /// **Not comparable across vendors.** The scales are unrelated, and
    /// the fact that `sm_120` and `gfx1030` land in the same order of
    /// magnitude is a coincidence of AMD's numbering, not a shared
    /// axis. Callers ranking a mixed cohort must fall back to something
    /// that genuinely compares (VRAM, or measured throughput) rather
    /// than pretending these are one scale.
    pub fn generation(&self) -> u32 {
        match self {
            GpuArch::Sm { major, minor } => major * 10 + minor,
            // Everything after "gfx" is the numeric target id. A
            // trailing letter exists on some parts (gfx90a), so take the
            // leading digits and let the letter break nothing.
            GpuArch::Gfx(g) => g
                .trim_start_matches("gfx")
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0),
        }
    }

    /// How this arch is spelled *inside* a `.arch` `archs=` list, which
    /// is not how it displays: NVIDIA writes `12.0` there (the CMake
    /// `TORCH_CUDA_ARCH_LIST` form) but shows `sm_120` to humans. AMD
    /// spells it the same both ways.
    ///
    /// Kept apart from `Display` on purpose. The two were the same
    /// string for exactly as long as NVIDIA was the only vendor, and
    /// conflating them puts `sm_120` into a list that
    /// [`GpuArch::covered_by`] then fails to match.
    pub fn archs_token(&self) -> String {
        match self {
            GpuArch::Sm { major, minor } => format!("{major}.{minor}"),
            GpuArch::Gfx(g) => g.clone(),
        }
    }

    /// Whether a libtorch variant compiled for `archs` covers this
    /// device.
    ///
    /// `archs` is the `.arch` metadata's `archs=` field, whose spelling
    /// is vendor-specific: `"6.1;12.0"` for NVIDIA (the CMake
    /// `TORCH_CUDA_ARCH_LIST` form), `"gfx1030;gfx1100"` for AMD.
    ///
    /// NVIDIA matches any listed capability of the same major, because
    /// PTX from another minor of that major is forward-compatible. AMD
    /// requires an **exact** gfx match: there is no PTX-equivalent
    /// fallback, and a near-miss fails at the first BLAS call rather
    /// than running slowly.
    ///
    /// **Both arms tokenize first.** The NVIDIA arm used to substring-
    /// search the raw list, and a bare digit is a substring of half the
    /// entries in a real one: a Maxwell `sm_50` device tested its major
    /// `"5"` against cu128's `"7.0 7.5 8.0 8.6 8.9 9.0 12.0"`, matched
    /// the `5` inside `7.5`, and was reported covered by a build that
    /// ships no kernel for it. `fdl diagnose` said OK and the first
    /// kernel launch said `no kernel image is available for execution
    /// on the device`. An unparsable token contributes nothing rather
    /// than matching loosely, which is also how `cpu` and a mixed-vendor
    /// list fall out for free.
    pub fn covered_by(&self, archs: &str) -> bool {
        archs.split([';', ',', ' ']).any(|token| {
            match (self, GpuArch::parse(self.vendor(), token)) {
                // Same major: exact capability, or another minor of it.
                (GpuArch::Sm { major, .. }, Some(GpuArch::Sm { major: m, .. })) => m == *major,
                (GpuArch::Gfx(gfx), Some(GpuArch::Gfx(g))) => g == *gfx,
                _ => false,
            }
        })
    }
}

impl fmt::Display for GpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuArch::Sm { major, minor } => write!(f, "sm_{major}{minor}"),
            GpuArch::Gfx(g) => f.write_str(g),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_round_trips_and_accepts_stack_names() {
        assert_eq!(GpuVendor::parse("nvidia"), Some(GpuVendor::Nvidia));
        assert_eq!(GpuVendor::parse(" CUDA "), Some(GpuVendor::Nvidia));
        assert_eq!(GpuVendor::parse("AMD"), Some(GpuVendor::Amd));
        assert_eq!(GpuVendor::parse("rocm"), Some(GpuVendor::Amd));
        assert_eq!(GpuVendor::parse("hip"), Some(GpuVendor::Amd));
        assert_eq!(GpuVendor::parse("intel"), None);
        for v in [GpuVendor::Nvidia, GpuVendor::Amd] {
            assert_eq!(GpuVendor::parse(v.as_str()), Some(v));
        }
    }

    #[test]
    fn variant_labels_classify_three_ways() {
        // Cpu and Unknown are distinct on purpose: a CPU variant is a
        // positive statement, an unrecognized name says nothing, and
        // the consumers apply different policies to each.
        for (label, class) in [
            ("cpu", VariantClass::Cpu),
            ("precompiled/cpu", VariantClass::Cpu),
            ("cpu-static", VariantClass::Cpu),
            ("precompiled/cu128", VariantClass::Vendor(GpuVendor::Nvidia)),
            ("builds/sm61-sm120", VariantClass::Vendor(GpuVendor::Nvidia)),
            ("precompiled/rocm70", VariantClass::Vendor(GpuVendor::Amd)),
            ("builds/gfx1030", VariantClass::Vendor(GpuVendor::Amd)),
            // The digit guard: a prefix alone is not a vendor claim.
            ("builds/gfx", VariantClass::Unknown),
            ("builds/mybuild", VariantClass::Unknown),
            ("", VariantClass::Unknown),
        ] {
            assert_eq!(classify_variant_label(label), class, "{label:?}");
        }
    }

    #[test]
    fn vendor_picks_its_cargo_feature() {
        assert_eq!(GpuVendor::Nvidia.cargo_feature(), "cuda");
        assert_eq!(GpuVendor::Amd.cargo_feature(), "rocm");
    }

    #[test]
    fn parses_nvidia_capability_forms() {
        let expect = GpuArch::Sm { major: 8, minor: 6 };
        for form in ["8.6", "sm_86", "sm86", " 8.6 "] {
            assert_eq!(GpuArch::parse(GpuVendor::Nvidia, form).unwrap(), expect, "{form}");
        }
    }

    #[test]
    fn concatenated_form_takes_the_last_digit_as_minor() {
        // sm_120 is 12.0, NOT 1.20: the major grew past one digit at
        // Blackwell, so a "first digit is major" rule silently mislabels
        // every current card.
        assert_eq!(
            GpuArch::parse(GpuVendor::Nvidia, "sm_120").unwrap(),
            GpuArch::Sm { major: 12, minor: 0 }
        );
        assert_eq!(
            GpuArch::parse(GpuVendor::Nvidia, "sm_61").unwrap(),
            GpuArch::Sm { major: 6, minor: 1 }
        );
    }

    #[test]
    fn rejects_malformed_capability() {
        for bad in ["", "sm_", "x.y", "sm_1", "notacap", "8."] {
            assert!(GpuArch::parse(GpuVendor::Nvidia, bad).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn strips_the_gfx_feature_suffix() {
        // rocminfo / gcnArchName append :sramecc±:xnack±. Every
        // downstream comparison wants the bare token.
        assert_eq!(
            GpuArch::parse(GpuVendor::Amd, "gfx906:sramecc-:xnack-").unwrap(),
            GpuArch::Gfx("gfx906".into())
        );
        assert_eq!(
            GpuArch::parse(GpuVendor::Amd, "GFX1030").unwrap(),
            GpuArch::Gfx("gfx1030".into())
        );
    }

    #[test]
    fn rejects_malformed_gfx() {
        for bad in ["", "gfx", "1030", "radeon"] {
            assert!(GpuArch::parse(GpuVendor::Amd, bad).is_none(), "{bad:?}");
        }
    }

    #[test]
    fn displays_in_vendor_form() {
        assert_eq!(GpuArch::Sm { major: 12, minor: 0 }.to_string(), "sm_120");
        assert_eq!(GpuArch::Gfx("gfx1100".into()).to_string(), "gfx1100");
    }

    #[test]
    fn archs_token_differs_from_display_on_nvidia_only() {
        // `.arch` archs= carries the CMake TORCH_CUDA_ARCH_LIST form.
        let sm = GpuArch::Sm { major: 12, minor: 0 };
        assert_eq!(sm.archs_token(), "12.0");
        assert_ne!(sm.archs_token(), sm.to_string());
        let gfx = GpuArch::Gfx("gfx1100".into());
        assert_eq!(gfx.archs_token(), gfx.to_string());
    }

    #[test]
    fn an_arch_is_covered_by_a_list_of_its_own_tokens() {
        // The round trip that keeps archs_token and covered_by honest:
        // whatever we write into `archs=` must match back.
        for a in [
            GpuArch::Sm { major: 6, minor: 1 },
            GpuArch::Sm { major: 12, minor: 0 },
            GpuArch::Gfx("gfx1030".into()),
        ] {
            assert!(a.covered_by(&a.archs_token()), "{a}");
            assert!(a.covered_by(&format!("gfx900;{};8.9", a.archs_token())), "{a}");
        }
    }

    #[test]
    fn nvidia_coverage_falls_back_to_major() {
        let sm86 = GpuArch::Sm { major: 8, minor: 6 };
        assert!(sm86.covered_by("6.1;8.6"));
        assert!(sm86.covered_by("8.0"), "same major is forward-compatible via PTX");
        assert!(!sm86.covered_by("6.1;12.0"));
    }

    #[test]
    fn nvidia_coverage_does_not_match_a_digit_inside_another_token() {
        // The regression: a substring search let a major match the minor
        // of an unrelated entry, so `fdl diagnose` reported OK and the
        // first kernel launch failed with "no kernel image".
        let cu128 = "7.0 7.5 8.0 8.6 8.9 9.0 12.0";
        for (maj, min) in [(5u32, 0u32), (5, 2)] {
            let dev = GpuArch::Sm { major: maj, minor: min };
            assert!(
                !dev.covered_by(cu128),
                "sm_{maj}{min} matched the 5 inside 7.5",
            );
        }
        // The mirror: a major that is a substring of a two-digit major.
        assert!(!GpuArch::Sm { major: 2, minor: 0 }.covered_by("12.0"));
        // ...while the two-digit major itself still matches.
        assert!(GpuArch::Sm { major: 12, minor: 0 }.covered_by(cu128));
        // Real coverage of the rig's own cards is unchanged.
        assert!(GpuArch::Sm { major: 6, minor: 1 }.covered_by("5.0 5.2 6.0 6.1 7.0"));
        assert!(!GpuArch::Sm { major: 6, minor: 1 }.covered_by(cu128));
    }

    #[test]
    fn a_ptx_suffixed_arch_list_entry_still_matches() {
        // TORCH_CUDA_ARCH_LIST accepts "8.6+PTX" and `fdl libtorch build`
        // writes the list through verbatim.
        let sm86 = GpuArch::Sm { major: 8, minor: 6 };
        assert!(sm86.covered_by("6.1;8.6+PTX"));
        assert_eq!(
            GpuArch::parse(GpuVendor::Nvidia, "8.6+PTX").unwrap(),
            sm86,
        );
    }

    #[test]
    fn a_cpu_variant_covers_nothing() {
        // `archs=cpu` is what the CPU download writes.
        assert!(!GpuArch::Sm { major: 8, minor: 6 }.covered_by("cpu"));
        assert!(!GpuArch::Gfx("gfx1030".into()).covered_by("cpu"));
    }

    #[test]
    fn amd_coverage_is_exact_only() {
        let gfx1030 = GpuArch::Gfx("gfx1030".into());
        assert!(gfx1030.covered_by("gfx900;gfx1030;gfx1100"));
        assert!(gfx1030.covered_by("gfx1030"));
        // No PTX equivalent on ROCm: a near-miss fails at the first BLAS
        // call, so substring/prefix leniency would be a false green.
        assert!(!gfx1030.covered_by("gfx1031"));
        assert!(!gfx1030.covered_by("gfx10"));
        assert!(!gfx1030.covered_by("gfx1100"));
    }

    #[test]
    fn arch_knows_its_own_vendor() {
        assert_eq!(GpuArch::Sm { major: 8, minor: 6 }.vendor(), GpuVendor::Nvidia);
        assert_eq!(GpuArch::Gfx("gfx942".into()).vendor(), GpuVendor::Amd);
        assert_eq!(GpuArch::Gfx("gfx942".into()).sm_major(), None);
    }
}
