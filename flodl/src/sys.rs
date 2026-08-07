//! System-level detection that does NOT touch libtorch / CUDA.
//!
//! flodl's main CUDA APIs (e.g. [`crate::tensor::gpu_device_count`])
//! initialize libtorch on first call. Once libtorch latches onto a
//! device list, `CUDA_VISIBLE_DEVICES` is ignored and — critically for
//! cluster mode — the spawned children inherit a corrupted CUDA
//! context on heterogeneous-GPU rigs.
//!
//! [`detect_gpus`] avoids both problems by asking the vendor's own
//! stack (nvidia-smi for NVIDIA, the kernel's KFD topology for AMD)
//! with no GPU runtime initialized. Use this when you need GPU info for
//! pre-`Trainer::run` decisions (mode filtering, log banners,
//! CLI-flag validation) — see the "no CUDA before `Trainer::run`"
//! invariant in the [`crate::distributed::Trainer`] docs.
//!
//! For *runtime* GPU queries (after dispatch), the libtorch-backed
//! APIs in [`crate::tensor`] remain the right tool.
//!
//! Returns an empty `Vec` when the vendor tool is missing or fails —
//! callers can treat that as "no GPU visible" without a separate
//! "did we have a driver" branch.
//!
//! # Three questions, three entry points
//!
//! They are named apart on purpose; conflating them is a real bug in
//! every direction.
//!
//! | Question | Use |
//! |---|---|
//! | What is installed on this box? (provisioning: which libtorch to fetch) | [`survey`] / [`detect_gpus_physical`] |
//! | What will the runtime see? (masks applied, all vendors) | [`survey_visible`] |
//! | What can *this build* train on? (masks **and** vendor) | [`detect_gpus`] / [`survey_visible_for`] |
//!
//! The vendor axis exists because libtorch is built for exactly one GPU
//! backend, so on a mixed box the other vendor's cards are present,
//! healthy and unusable. See [`build_vendor`].
//!
//! # Where this lives
//!
//! The implementation is the dependency-free [`flodl_hw`] crate, which
//! `flodl-cli` also depends on. `fdl` needs the same answers *before
//! libtorch exists at all* (to pick which variant to install) and so
//! cannot depend on `flodl`; both used to carry a hand-synchronized copy
//! of this struct and this parser. This module is a re-export of the
//! single source, kept as `flodl::sys` because that is the published
//! path.
//!
//! # Spoofing hardware in tests
//!
//! [`ENV_TESTING_GPU_JSON`] replaces the whole sweep with a described
//! one, the sibling of `FLODL_TESTING_CLUSTER_JSON` for hardware rather
//! than topology. It is how a second GPU vendor's detection and routing
//! get tested on a machine that has none of that hardware:
//!
//! ```text
//! FLODL_TESTING_GPU_JSON='[{"vendor":"amd","arch":"gfx1030","vram_mb":16384}]' fdl test
//! ```
//!
//! `fdl` forwards it across the docker boundary. Visibility masks still
//! apply on top, and a malformed value panics rather than quietly
//! falling back to the real hardware. Full format in
//! [`flodl_hw::testing`].

pub use flodl_hw::{
    cpu_package_count, detect_gpus_for, detect_gpus_physical, mem_info, survey, survey_visible,
    survey_visible_for,
    GpuArch, GpuInfo, GpuSurvey, GpuVendor, MemInfo, NoteKind, SurveyNote, ENV_TESTING_GPU_JSON,
};

/// The GPU vendor this build of `flodl` can address, or `None` for a
/// CPU-only build.
///
/// libtorch is built for exactly one GPU backend and both claim
/// `DeviceType::CUDA`, so the vendor is a **compile-time** property:
/// `libtorch_cuda.so` and `libtorch_hip.so` cannot coexist in a process,
/// as both register kernels against the same dispatch key. That is why
/// this reads cargo features rather than probing anything.
pub fn build_vendor() -> Option<GpuVendor> {
    // Mutually exclusive by flodl-sys/build.rs, which hard-errors on
    // both -- so the order here can never silently pick a winner.
    if cfg!(feature = "cuda") {
        Some(GpuVendor::Nvidia)
    } else if cfg!(feature = "rocm") {
        Some(GpuVendor::Amd)
    } else {
        None
    }
}

/// Enumerate the GPUs this build can actually train on: visibility masks
/// applied, then narrowed to [`build_vendor`].
///
/// The vendor filter is not cosmetic. A mixed AMD + NVIDIA host is an
/// ordinary machine (any box with an AMD APU and a discrete NVIDIA card
/// is one), and this count feeds the `>= 2` DDP auto-promote decision --
/// without the filter, a CUDA build on such a box spawns a rank for a
/// device it cannot address, and the failure surfaces far from its
/// cause. It also disambiguates indices, which are per-vendor ordinals.
///
/// A CPU-only build makes no vendor claim, so it filters nothing and
/// keeps reporting whatever is installed; nothing in that build will try
/// to place a rank on it.
///
/// Use [`survey_visible_for`] instead when the *reason* devices went
/// missing matters -- it carries the [`NoteKind::VendorMismatch`] note
/// that explains a zero count on a box whose hardware is the other
/// vendor's.
pub fn detect_gpus() -> Vec<GpuInfo> {
    match build_vendor() {
        Some(vendor) => detect_gpus_for(vendor),
        None => flodl_hw::detect_gpus(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: deliberately NO `ENV_TESTING_GPU_JSON` spoofing here.
    //
    // That variable is process-global, and this crate's test binary runs
    // ~2000 tests in parallel, several of which read the GPU survey.
    // Serializing writers under a local mutex does NOT help, because the
    // readers are arbitrary other tests that never take it -- setting the
    // spoof here made an unrelated ddp_run worker test fail while passing
    // in isolation. The mutex convention only works when reader and
    // writer share the lock.
    //
    // The spoof-driven cases (mixed-host filtering, the dropped-device
    // note) live in `flodl-hw`, whose test binary is isolated and whose
    // tests all take its `ENV_LOCK`. What is left to check here is the
    // wiring, which needs no hardware and no env at all.

    #[test]
    fn build_vendor_matches_the_feature_this_was_compiled_with() {
        let v = build_vendor();
        if cfg!(feature = "cuda") {
            assert_eq!(v, Some(GpuVendor::Nvidia));
        } else if cfg!(feature = "rocm") {
            assert_eq!(v, Some(GpuVendor::Amd));
        } else {
            assert_eq!(v, None, "a CPU build claims no vendor");
        }
    }

    #[test]
    fn detect_gpus_never_returns_a_vendor_this_build_cannot_address() {
        // Property, not a fixture: true on any hardware including none,
        // and a wiring inversion (filtering to the wrong vendor, or not
        // filtering at all on a mixed box) breaks it.
        let Some(vendor) = build_vendor() else {
            return; // a CPU build makes no vendor claim
        };
        for g in detect_gpus() {
            assert_eq!(
                g.vendor, vendor,
                "detect_gpus returned a {} device on a {} build",
                g.vendor, vendor
            );
        }
    }

    #[test]
    fn detect_gpus_is_a_subset_of_the_unfiltered_visible_sweep() {
        // The filter may only ever REMOVE devices; it must not invent or
        // reorder them.
        let filtered = detect_gpus();
        let all = survey_visible().devices;
        assert!(filtered.len() <= all.len());
        for g in &filtered {
            assert!(all.contains(g), "filtered device not in the visible sweep");
        }
    }
}
