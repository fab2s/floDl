//! `fdl libtorch download` -- download pre-built libtorch.

use std::fs;
use std::path::{Path, PathBuf};

use super::detect;
use crate::context::Context;
use crate::util::archive;
use crate::util::http;
use crate::util::system;
use crate::util::system::GpuVendor;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LIBTORCH_VERSION: &str = "2.10.0";

/// Pre-built variant metadata.
struct VariantSpec {
    /// Label for display (e.g. "CUDA 12.8").
    label: &'static str,
    /// Directory name under precompiled/ (e.g. "cu128").
    dir_name: &'static str,
    /// Value for .arch `cuda=` field.
    arch_cuda: &'static str,
    /// Space-separated compute capabilities covered.
    arch_archs: &'static str,
    /// Value for .arch `variant=` field.
    arch_variant: &'static str,
}

const CPU_SPEC: VariantSpec = VariantSpec {
    label: "CPU",
    dir_name: "cpu",
    arch_cuda: "none",
    arch_archs: "cpu",
    arch_variant: "cpu",
};

const CU126_SPEC: VariantSpec = VariantSpec {
    label: "CUDA 12.6",
    dir_name: "cu126",
    arch_cuda: "12.6",
    arch_archs: "5.0 5.2 6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0",
    arch_variant: "cu126",
};

const CU128_SPEC: VariantSpec = VariantSpec {
    label: "CUDA 12.8",
    dir_name: "cu128",
    arch_cuda: "12.8",
    arch_archs: "7.0 7.5 8.0 8.6 8.9 9.0 12.0",
    arch_variant: "cu128",
};

/// gfx targets the ROCm archives ship rocBLAS Tensile kernels for.
///
/// Read out of the published archives rather than inferred: both ROCm
/// buckets of a given libtorch version carry the same set, and a target
/// is only listed when the archive holds `.hsaco` or `TensileLibrary*`
/// payload for it. Targets with nothing but MIOpen performance
/// databases (`gfx900`, `gfx906`) are deliberately absent -- rocBLAS has
/// no kernels to load for them, so listing one would let the
/// arch-coverage gate admit a box that dies at its first BLAS call,
/// which is the death that gate exists to move before the dial.
///
/// Verifiable without downloading the archive: its central directory is
/// reachable with HTTP range requests (a few MB against ~5 GB), and the
/// host answers 403 without a User-Agent.
const ROCM_ARCHS: &str = "gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 \
                          gfx1102 gfx1150 gfx1151 gfx1200 gfx1201";

const ROCM70_SPEC: VariantSpec = VariantSpec {
    label: "ROCm 7.0",
    // `rocm70` (no dot) matches the cu128 style and satisfies
    // `detect::variant_vendor`'s `rocm<digit>` rule.
    dir_name: "rocm70",
    // `.arch` cuda= is the CUDA toolkit version; an AMD build has none.
    // The vendor is carried by the variant path, which is what
    // `variant_vendor` and prebuild's feature derivation read.
    arch_cuda: "none",
    arch_archs: ROCM_ARCHS,
    // Doubles as the URL bucket AND the `+<variant>` filename suffix,
    // exactly like `cu128` -- PyTorch dropped the `cxx11-abi-` filename
    // prefix, so the ROCm archives follow the same pattern as CUDA's and
    // need no special-casing in the URL builder.
    arch_variant: "rocm7.0",
};

const ROCM71_SPEC: VariantSpec = VariantSpec {
    label: "ROCm 7.1",
    dir_name: "rocm71",
    arch_cuda: "none",
    // Identical hardware reach to 7.0: the two buckets ship the same
    // gfx targets, so this variant exists for runtime matching, not for
    // coverage.
    arch_archs: ROCM_ARCHS,
    arch_variant: "rocm7.1",
};

// ---------------------------------------------------------------------------
// Download options
// ---------------------------------------------------------------------------

pub enum Variant {
    Cpu,
    Cuda126,
    Cuda128,
    Rocm70,
    Rocm71,
    Auto,
}

pub struct DownloadOpts {
    pub variant: Variant,
    pub custom_path: Option<PathBuf>,
    pub activate: bool,
    pub dry_run: bool,
    /// Force the Linux x86_64 build regardless of host OS. Set when the
    /// libtorch will be consumed inside a Linux Docker container rather
    /// than linked against host cargo — without this, macOS hosts pick
    /// `libtorch-macos-arm64-*.zip` (Mach-O dylibs) which then fail to
    /// load inside the Linux container that bind-mounts the directory.
    pub force_linux: bool,
}

impl Default for DownloadOpts {
    fn default() -> Self {
        Self {
            variant: Variant::Auto,
            custom_path: None,
            activate: true,
            dry_run: false,
            force_linux: false,
        }
    }
}

// ---------------------------------------------------------------------------
// URL construction
// ---------------------------------------------------------------------------

/// Absolute `libomp` dependencies in one `otool -L` dump.
///
/// Pure, so the parse is testable on any host without a Mach-O to hand.
/// `otool -L` prints the file name on the first line and one indented
/// dependency per line after it, each followed by version parens; only
/// the leading path matters. `@rpath/...` and `@loader_path/...` are
/// already relative and left alone, as is anything that is not libomp:
/// this rewrites ONE known upstream defect, not every absolute path.
fn absolute_libomp_refs(otool_output: &str) -> Vec<String> {
    otool_output
        .lines()
        .skip(1)
        .filter_map(|l| l.split_whitespace().next())
        .filter(|p| p.starts_with('/') && p.ends_with("/libomp.dylib"))
        .map(str::to_string)
        .collect()
}

/// Point the macOS archive's own dylibs at the `libomp` it ships with.
///
/// Upstream's `libtorch-macos-arm64` BUNDLES `lib/libomp.dylib` and then
/// has `libtorch_cpu.dylib` depend on it by absolute Homebrew path
/// (`/opt/homebrew/opt/libomp/lib/libomp.dylib`). On a box without that
/// Homebrew formula the load fails while the library it wants sits in the
/// same directory as the dylib asking for it, so a scaffolded project
/// compiles and dies at launch. `brew install libomp` is the wrong answer:
/// it installs a second, possibly ABI-divergent copy of something already
/// present.
///
/// `@loader_path/libomp.dylib` rather than `@rpath/...` deliberately: the
/// bundled copy is a sibling of every dylib referencing it, so
/// `@loader_path` resolves with no dependence on the referrer carrying a
/// correct `LC_RPATH`.
///
/// Advisory, never fatal: libtorch IS installed at this point, and the
/// docker path does not care about any of this. But the tools are checked
/// BEFORE anything is modified, because `install_name_tool` invalidates a
/// Mach-O signature and arm64 refuses to load an unsigned one -- patching
/// without being able to re-sign would leave the install worse than it
/// was found.
fn relink_bundled_libomp(lib_dir: &Path) {
    // The gate is capability, not `cfg`: `fdl setup` on Apple Silicon
    // fetches the LINUX archive for a docker project, which has no
    // dylibs at all and must not be touched.
    if !lib_dir.join("libomp.dylib").exists() {
        return;
    }
    let dylibs: Vec<PathBuf> = match fs::read_dir(lib_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "dylib"))
            .collect(),
        Err(_) => return,
    };

    let missing: Vec<&str> = ["otool", "install_name_tool", "codesign"]
        .into_iter()
        .filter(|t| !crate::util::system::has_command(t))
        .collect();
    if !missing.is_empty() {
        println!(
            "  note: cannot relink the bundled libomp ({} not found).\n\
             \x20       Upstream's libtorch_cpu.dylib asks for libomp at an absolute\n\
             \x20       Homebrew path, so a NATIVE run may fail to start; the docker\n\
             \x20       path is unaffected. Install the command line tools with\n\
             \x20       `xcode-select --install` and re-run this download to fix it.",
            missing.join(", "),
        );
        return;
    }

    let mut patched = 0usize;
    for f in &dylibs {
        let out = match std::process::Command::new("otool")
            .arg("-L")
            .arg(f)
            .output()
        {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => continue,
        };
        let refs = absolute_libomp_refs(&out);
        if refs.is_empty() {
            continue;
        }
        let mut cmd = std::process::Command::new("install_name_tool");
        for r in &refs {
            cmd.arg("-change").arg(r).arg("@loader_path/libomp.dylib");
        }
        match cmd.arg(f).output() {
            Ok(o) if o.status.success() => {}
            other => {
                println!(
                    "  note: install_name_tool failed on {}: {}",
                    f.display(),
                    match other {
                        Ok(o) => String::from_utf8_lossy(&o.stderr).trim().to_string(),
                        Err(e) => e.to_string(),
                    },
                );
                continue;
            }
        }
        // Ad-hoc re-sign, mandatory on arm64: the edit above invalidated
        // whatever signature the file carried.
        match std::process::Command::new("codesign")
            .args(["-f", "-s", "-"])
            .arg(f)
            .output()
        {
            Ok(o) if o.status.success() => patched += 1,
            other => println!(
                "  warning: {} was relinked but could NOT be re-signed ({}); \
                 it may fail to load. Re-run this download after \
                 `xcode-select --install`.",
                f.display(),
                match other {
                    Ok(o) => String::from_utf8_lossy(&o.stderr).trim().to_string(),
                    Err(e) => e.to_string(),
                },
            ),
        }
    }
    if patched > 0 {
        println!("  relinked {patched} dylib(s) to the bundled libomp");
    }
}

fn download_url(spec: &VariantSpec, force_linux: bool) -> Result<String, String> {
    // `force_linux` short-circuits host detection: the binary is destined
    // for a Linux Docker container, so we always want the Linux x86_64
    // build regardless of what the host is.
    let (os, arch) = if force_linux {
        ("linux", "x86_64")
    } else {
        (std::env::consts::OS, std::env::consts::ARCH)
    };

    download_url_for(spec, os, arch)
}

/// Pure core of [`download_url`]: the host is a parameter rather than a
/// global read.
///
/// Every platform arm is reachable from any test runner as a result, which
/// is the point. The Windows filename pattern differs from Linux's, this
/// function claimed in a comment that it did not, and the resulting 404
/// shipped unnoticed because nothing had ever evaluated the Windows arm on
/// a Windows host. Host-as-parameter is what makes that testable without
/// one.
fn download_url_for(spec: &VariantSpec, os: &str, arch: &str) -> Result<String, String> {
    match (os, arch) {
        ("linux", "x86_64") => {}
        ("macos", "aarch64") => {
            // `cuda=none` stopped meaning "CPU build" when a second
            // vendor arrived: a ROCm spec carries it too, and without
            // the second clause it resolves to the macOS CPU archive
            // and installs it under a ROCm directory name.
            if spec.arch_cuda != "none" || spec.arch_variant.starts_with("rocm") {
                return Err("macOS only supports CPU libtorch".into());
            }
        }
        ("macos", _) => {
            return Err(format!(
                "macOS libtorch requires Apple Silicon (arm64), got {}.\n\
                 macOS x86_64 was dropped after PyTorch 2.2.",
                arch
            ));
        }
        ("windows", "x86_64") => {
            // PyTorch publishes no ROCm build for Windows: the `rocm7.0`
            // bucket carries Linux archives only.
            if spec.arch_variant.starts_with("rocm") {
                return Err(format!(
                    "{} libtorch is not available for Windows.\n\
                     PyTorch publishes ROCm builds for Linux only.",
                    spec.label
                ));
            }
        }
        _ => {
            return Err(format!(
                "Unsupported platform: {} {}.\n\
                 libtorch is available for Linux x86_64, macOS arm64, and Windows x86_64.",
                os, arch
            ));
        }
    }

    // macOS ARM has a different filename pattern
    if os == "macos" {
        return Ok(format!(
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-{}.zip",
            LIBTORCH_VERSION
        ));
    }

    // Linux and Windows share the bucket layout but NOT the filename:
    // Windows archives carry a `-win-` infix. PyTorch also publishes a
    // `-debug-` Windows variant (built against the debug CRT); we fetch the
    // release one, which is what a release-mode consumer must link against.
    let infix = if os == "windows" { "win-" } else { "" };
    let filename = format!(
        "libtorch-{}shared-with-deps-{}%2B{}.zip",
        infix, LIBTORCH_VERSION, spec.arch_variant
    );

    let bucket = spec.arch_variant; // "cpu", "cu126", "cu128", "rocm7.0"
    Ok(format!(
        "https://download.pytorch.org/libtorch/{}/{}",
        bucket, filename
    ))
}

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------

fn auto_detect_variant() -> &'static VariantSpec {
    let survey = flodl_hw::survey();
    if survey.devices.is_empty() {
        // Say WHY before routing to CPU: the sweep deliberately reports
        // no device for an AMD card with no ROCm runtime, and that
        // finding names the fix — discarding it turns a provisioning
        // step ("install ROCm, then re-run") into a silent wrong
        // variant.
        for note in survey.notes.iter().filter(|n| n.kind.explains_absence()) {
            println!("  {}", note.message);
        }
    }
    variant_for_gpus(&survey.devices)
}

/// Route a detected GPU set to a libtorch variant.
///
/// Pure: the device list is a parameter rather than a probe, so every
/// vendor and coverage arm is testable without hardware and without the
/// process-global detection spoof.
fn variant_for_gpus(gpus: &[system::GpuInfo]) -> &'static VariantSpec {
    if gpus.is_empty() {
        println!("  No GPU detected. Using CPU variant.");
        return &CPU_SPEC;
    }

    // A libtorch build serves exactly one vendor, so a mixed box has to
    // pick. NVIDIA wins: in a box holding both, the AMD part is usually
    // an APU's integrated GPU and the NVIDIA one the training card.
    let amd: Vec<_> = gpus.iter().filter(|g| g.vendor == GpuVendor::Amd).collect();
    let has_nvidia = gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia);
    if !amd.is_empty() {
        if has_nvidia {
            println!(
                "  Both NVIDIA and AMD GPUs detected. One libtorch build serves\n  \
                 one vendor, so the NVIDIA cards are used and the AMD ones stay\n  \
                 idle. For the AMD cards instead: fdl libtorch download --rocm 7.0",
            );
        } else {
            return rocm_variant_for(&amd);
        }
    }

    // The CUDA variants below are selected on compute capability, which
    // only NVIDIA devices carry.
    let majors: Vec<u32> = gpus.iter().filter_map(|g| g.sm_major()).collect();
    if majors.is_empty() {
        let other: Vec<String> = gpus
            .iter()
            .map(|g| format!("{} ({})", g.short_name(), g.arch_label()))
            .collect();
        println!(
            "  Detected a GPU with no known libtorch variant ({}).\n  \
             Using the CPU variant.",
            other.join(", "),
        );
        return &CPU_SPEC;
    }
    let lo_major = majors.iter().copied().min().unwrap_or(0);
    let hi_major = majors.iter().copied().max().unwrap_or(0);

    // cu128 requires Volta+ (sm_70+), cu126 supports down to sm_50
    if lo_major >= 7 {
        println!("  Detected Volta+ GPU(s). Using cu128.");
        &CU128_SPEC
    } else if hi_major >= 10 {
        // Mixed: old + new GPUs. cu126 covers the old ones, cu128 covers the new.
        // Default to cu126 which covers more architectures.
        println!(
            "  Mixed GPU architectures (sm_{}.x to sm_{}.x).",
            lo_major, hi_major
        );
        println!("  Using cu126 (broadest pre-Volta coverage).");
        println!("  For all GPUs, consider: fdl libtorch build");
        &CU126_SPEC
    } else {
        println!("  Detected pre-Volta GPU(s). Using cu126.");
        &CU126_SPEC
    }
}

/// AMD devices the ROCm variant ships kernels for.
///
/// Exposed so the setup wizard routes on the same coverage list this
/// module downloads against: two independently-maintained copies is how
/// the wizard came to skip AMD boxes in silence.
pub fn rocm_covered(gpus: &[system::GpuInfo]) -> Vec<&system::GpuInfo> {
    gpus.iter()
        .filter(|g| g.vendor == GpuVendor::Amd && g.covered_by(ROCM_ARCHS))
        .collect()
}

/// The gfx targets the ROCm variants cover, for diagnostics.
pub fn rocm_archs() -> &'static str {
    ROCM_ARCHS
}

/// Pick between the ROCm variant and CPU for a set of AMD devices.
///
/// The ROCm archive carries pre-built rocBLAS Tensile kernels for a
/// fixed gfx list; a target outside it has no kernels, so the variant is
/// only worth downloading when it covers at least one device present.
///
/// Which ROCm bucket is not a hardware question: they cover the same
/// targets, so this routes to the OLDEST offered one on purpose. The
/// HIP runtime ordering rule puts the host's own ROCm ahead of the
/// bundle, and within a major version that ABI grows, so a bundle older
/// than the host loads while a newer one can fail on a symbol the host
/// runtime does not have. 7.0 therefore serves every 7.x host, where
/// 7.1 would drop the 7.0 ones. Picking the newest bundle that the
/// detected host runtime can satisfy is the better rule and needs the
/// ROCm version resolver; until then, oldest-serves-most. A host that
/// wants the exact match asks for it: `fdl libtorch download --rocm 7.1`.
fn rocm_variant_for(amd: &[&system::GpuInfo]) -> &'static VariantSpec {
    let (covered, uncovered): (Vec<_>, Vec<_>) = amd.iter().partition(|g| g.covered_by(ROCM_ARCHS));

    let describe = |gs: &[&&system::GpuInfo]| {
        gs.iter()
            .map(|g| format!("{} ({})", g.short_name(), g.arch_label()))
            .collect::<Vec<_>>()
            .join(", ")
    };

    if covered.is_empty() {
        println!(
            "  Detected AMD GPU(s) ({}) outside the ROCm build's gfx\n  \
             targets, so the CPU variant is selected.\n  \
             Covered targets: {}.",
            describe(&uncovered),
            ROCM_ARCHS,
        );
        return &CPU_SPEC;
    }
    if !uncovered.is_empty() {
        println!(
            "  Note: {} is not covered by the ROCm build and will be\n  \
             unusable. Covered targets: {}.",
            describe(&uncovered),
            ROCM_ARCHS,
        );
    }
    println!(
        "  Detected AMD GPU(s) ({}). Using ROCm 7.0.",
        describe(&covered)
    );
    &ROCM70_SPEC
}

fn resolve_variant(variant: &Variant) -> &'static VariantSpec {
    match variant {
        Variant::Cpu => &CPU_SPEC,
        Variant::Cuda126 => &CU126_SPEC,
        Variant::Cuda128 => &CU128_SPEC,
        Variant::Rocm70 => &ROCM70_SPEC,
        Variant::Rocm71 => &ROCM71_SPEC,
        Variant::Auto => auto_detect_variant(),
    }
}

// ---------------------------------------------------------------------------
// Core download logic
// ---------------------------------------------------------------------------

pub fn run(opts: DownloadOpts) -> Result<String, String> {
    let ctx = Context::resolve();
    run_with_context(opts, &ctx)
}

/// Run with an explicit context (used by `setup` which has its own
/// context).
///
/// Returns the variant id it resolved to (`precompiled/<dir>`), because
/// `Variant::Auto` only decides here: a caller that needs the path or the
/// label afterwards would otherwise have to re-run the detection, which
/// re-prints its reasoning and can only agree by accident.
pub fn run_with_context(opts: DownloadOpts, ctx: &Context) -> Result<String, String> {
    let spec = resolve_variant(&opts.variant);
    let url = download_url(spec, opts.force_linux)?;

    // Determine install path
    let install_path = if let Some(ref p) = opts.custom_path {
        p.clone()
    } else {
        ctx.root
            .join(format!("libtorch/precompiled/{}", spec.dir_name))
    };

    let variant_id = format!("precompiled/{}", spec.dir_name);

    println!();
    println!("  libtorch {} ({})", LIBTORCH_VERSION, spec.label);
    println!("  URL:  {}", url);
    println!("  Path: {}", install_path.display());

    if opts.dry_run {
        println!();
        println!("  [dry-run] Would download and extract to above path.");
        return Ok(variant_id);
    }

    // Check existing installation
    if install_path.exists() {
        let build_ver_path = install_path.join("build-version");
        let existing_ver = fs::read_to_string(&build_ver_path)
            .ok()
            .map(|s| s.trim().to_string());

        // build-version may contain variant suffix (e.g. "2.10.0+cpu")
        let ver_matches = existing_ver.as_deref().is_some_and(|v| {
            v == LIBTORCH_VERSION || v.starts_with(&format!("{}+", LIBTORCH_VERSION))
        });

        if ver_matches {
            println!();
            println!("  Already installed (version {}).", LIBTORCH_VERSION);
            return Ok(variant_id);
        }

        println!();
        println!(
            "  Removing existing installation (version: {})...",
            existing_ver.as_deref().unwrap_or("unknown")
        );
        fs::remove_dir_all(&install_path)
            .map_err(|e| format!("cannot remove {}: {}", install_path.display(), e))?;
    }

    // Stage BESIDE the destination, not in the system temp dir.
    //
    // `std::env::temp_dir()` is `/tmp`, which on a great many Linux
    // setups is a small RAM-backed tmpfs -- 16 GiB on the rig this was
    // found on. Staging there needs the archive AND its expansion at
    // once: ~20 GiB for a ROCm build, ~7 GiB even for CUDA. Blowing it
    // does not merely fail the download, it fills a tmpfs that the rest
    // of the system (and every shell's temp files) depends on.
    //
    // The destination's own filesystem is the one the user actually
    // sized for libtorch, and staging there makes the final move a
    // same-filesystem rename rather than a cross-device copy.
    let stage_root = install_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&stage_root)
        .map_err(|e| format!("cannot create {}: {}", stage_root.display(), e))?;
    let stage = Staging::new(stage_root.join(format!(".fdl-staging-{}", std::process::id())))?;

    let tmp_zip = stage.path().join(format!("libtorch-{}.zip", spec.dir_name));

    println!();
    println!("  Downloading...");
    http::download_file(&url, &tmp_zip)?;

    // Extract (the zip carries a top-level "libtorch/" dir)
    let tmp_extract = stage.path().join("extract");
    println!("  Extracting...");
    archive::extract_zip(&tmp_zip, &tmp_extract)?;

    // Move extracted contents to target path
    let extracted_lt = tmp_extract.join("libtorch");
    let source = if extracted_lt.is_dir() {
        &extracted_lt
    } else {
        &tmp_extract
    };

    fs::create_dir_all(&install_path)
        .map_err(|e| format!("cannot create {}: {}", install_path.display(), e))?;

    // Move all files from extracted dir to install path. Same
    // filesystem now, so `move_contents`'s rename path is the one that
    // fires.
    move_contents(source, &install_path)?;

    // `stage` cleans itself up on drop, including on the error paths
    // above -- the predecessor leaked its temp zip and extract dir
    // whenever anything failed, which on a tmpfs meant a failed
    // download left gigabytes behind until reboot.
    drop(stage);

    // Verify
    let lib_dir = install_path.join("lib");
    let has_lib = lib_dir.join("libtorch.so").exists()
        || lib_dir.join("libtorch.dylib").exists()
        || lib_dir.join("torch.lib").exists();

    if !has_lib {
        return Err(format!(
            "libtorch library not found at {}.\n\
             The archive structure may have changed.\n\
             Check: ls {}",
            lib_dir.display(),
            lib_dir.display()
        ));
    }

    relink_bundled_libomp(&lib_dir);

    // Write .arch metadata (always, both project and global)
    let arch_content = format!(
        "cuda={}\ntorch={}\narchs={}\nsource=precompiled\nvariant={}\n",
        spec.arch_cuda, LIBTORCH_VERSION, spec.arch_archs, spec.arch_variant
    );
    fs::write(install_path.join(".arch"), arch_content)
        .map_err(|e| format!("cannot write .arch: {}", e))?;

    if opts.activate {
        detect::set_active(&ctx.root, &variant_id)?;
    }

    println!();
    println!("  ================================================");
    println!("  libtorch {} ({}) installed", LIBTORCH_VERSION, spec.label);
    println!("  {}", install_path.display());
    println!("  ================================================");

    if ctx.is_project {
        println!();
        println!("  .arch:   {}/.arch", install_path.display());
        if opts.activate {
            println!("  .active: libtorch/.active -> {}", variant_id);
        }
        println!();
        // From the variant PATH, not `.arch`'s `cuda=`: a ROCm build has
        // no CUDA toolkit version and writes `cuda=none` there exactly
        // like a CPU build, so reading that field told anyone who had
        // just installed ROCm libtorch to run the CPU test suite.
        if detect::variant_vendor(&variant_id).is_some() {
            println!("  Run 'fdl gpu-test' to verify.");
        } else {
            println!("  Run 'fdl test' to verify.");
        }
    } else {
        println!();
        println!("  Installed to: {}", install_path.display());
        println!();
        println!("  To use with tch-rs or flodl, add to your shell profile:");
        println!();
        println!("    export LIBTORCH=\"{}\"", install_path.display());
        // Shared recipe: on a ROCm variant the system runtime has to come
        // first, and a recipe the user pastes is exactly where getting
        // that backwards costs a segfault at the first GPU op.
        let lib = format!("{}/lib", install_path.display());
        for line in detect::ld_library_path_lines(detect::variant_vendor(&variant_id), &lib) {
            println!("    {line}");
        }
        println!();
        println!("  Or start a new floDl project:");
        println!("    fdl init my-project");
    }

    Ok(variant_id)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Move all files and directories from `src` into `dest`.
/// A staging directory that removes itself on drop, however we leave.
///
/// The point is the failure paths: a download or extract that errors
/// out used to leave its partial archive and expansion behind, which on
/// a tmpfs is space nothing reclaims until reboot.
struct Staging(PathBuf);

impl Staging {
    fn new(path: PathBuf) -> Result<Self, String> {
        // A leftover from a crashed run would otherwise merge into this
        // one; start clean.
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path)
            .map_err(|e| format!("cannot create staging dir {}: {}", path.display(), e))?;
        Ok(Self(path))
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn move_contents(src: &Path, dest: &Path) -> Result<(), String> {
    let entries = fs::read_dir(src).map_err(|e| format!("cannot read {}: {}", src.display(), e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir error: {}", e))?;
        let from = entry.path();
        let name = entry.file_name();
        let to = dest.join(&name);

        // Try rename first (fast, same filesystem). Fall back to copy.
        if fs::rename(&from, &to).is_err() {
            if from.is_dir() {
                copy_dir_recursive(&from, &to)?;
            } else {
                fs::copy(&from, &to)
                    .map_err(|e| format!("copy {} -> {}: {}", from.display(), to.display(), e))?;
            }
        }
    }
    Ok(())
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|e| format!("cannot create {}: {}", dest.display(), e))?;

    for entry in fs::read_dir(src).map_err(|e| format!("read {}: {}", src.display(), e))? {
        let entry = entry.map_err(|e| format!("read_dir error: {}", e))?;
        let from = entry.path();
        let to = dest.join(entry.file_name());

        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            fs::copy(&from, &to)
                .map_err(|e| format!("copy {} -> {}: {}", from.display(), to.display(), e))?;
        }
    }
    Ok(())
}

/// Get the current libtorch version constant (for display and checks).
#[allow(dead_code)]
pub fn libtorch_version() -> &'static str {
    LIBTORCH_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    // These assert the exact upstream filename grammar, which differs per
    // OS in ways that are invisible from a Linux dev box. Each expectation
    // below was confirmed against download.pytorch.org with a bogus-name
    // control request, not inferred from the neighbouring arms.

    #[test]
    fn linux_url_has_no_os_infix() {
        let url = download_url_for(&CU128_SPEC, "linux", "x86_64").unwrap();
        assert_eq!(
            url,
            format!(
                "https://download.pytorch.org/libtorch/cu128/\
                 libtorch-shared-with-deps-{LIBTORCH_VERSION}%2Bcu128.zip"
            )
        );
    }

    #[test]
    fn windows_url_carries_the_win_infix() {
        // Regression: this arm used to build the Linux filename and 404.
        let url = download_url_for(&CU128_SPEC, "windows", "x86_64").unwrap();
        assert!(
            url.contains("libtorch-win-shared-with-deps-"),
            "windows archives need the `-win-` infix, got {url}"
        );
        assert_eq!(
            url,
            format!(
                "https://download.pytorch.org/libtorch/cu128/\
                 libtorch-win-shared-with-deps-{LIBTORCH_VERSION}%2Bcu128.zip"
            )
        );
    }

    #[test]
    fn windows_cpu_url_carries_the_win_infix() {
        let url = download_url_for(&CPU_SPEC, "windows", "x86_64").unwrap();
        assert_eq!(
            url,
            format!(
                "https://download.pytorch.org/libtorch/cpu/\
                 libtorch-win-shared-with-deps-{LIBTORCH_VERSION}%2Bcpu.zip"
            )
        );
    }

    #[test]
    fn windows_rejects_rocm() {
        // The ROCm buckets are Linux-only upstream; a `-win-` URL there is
        // a 404, so refuse before downloading rather than after.
        for spec in [&ROCM70_SPEC, &ROCM71_SPEC] {
            let err = download_url_for(spec, "windows", "x86_64").unwrap_err();
            assert!(err.contains("not available for Windows"), "got {err}");
        }
    }

    #[test]
    fn linux_accepts_rocm() {
        for spec in [&ROCM70_SPEC, &ROCM71_SPEC] {
            let url = download_url_for(spec, "linux", "x86_64").unwrap();
            let bucket = spec.arch_variant;
            assert_eq!(
                url,
                format!(
                    "https://download.pytorch.org/libtorch/{bucket}/\
                     libtorch-shared-with-deps-{LIBTORCH_VERSION}%2B{bucket}.zip"
                )
            );
        }
    }

    #[test]
    fn the_rocm_variants_differ_only_in_runtime_version() {
        // Same hardware reach, different bundled HIP runtime: the second
        // variant exists so a host can match its own ROCm, not so it can
        // reach a card the other one cannot.
        assert_eq!(ROCM70_SPEC.arch_archs, ROCM71_SPEC.arch_archs);
        assert_ne!(ROCM70_SPEC.arch_variant, ROCM71_SPEC.arch_variant);
        assert_ne!(ROCM70_SPEC.dir_name, ROCM71_SPEC.dir_name);
        // `variant_vendor` reads the directory basename, so both must
        // still say AMD to the feature derivation.
        for spec in [&ROCM70_SPEC, &ROCM71_SPEC] {
            assert_eq!(
                detect::variant_vendor(&format!("precompiled/{}", spec.dir_name)),
                Some(GpuVendor::Amd),
                "{} must derive the AMD feature",
                spec.dir_name
            );
        }
    }

    #[test]
    fn macos_arm_uses_its_own_filename_and_is_cpu_only() {
        let url = download_url_for(&CPU_SPEC, "macos", "aarch64").unwrap();
        assert_eq!(
            url,
            format!(
                "https://download.pytorch.org/libtorch/cpu/\
                 libtorch-macos-arm64-{LIBTORCH_VERSION}.zip"
            )
        );

        let err = download_url_for(&CU128_SPEC, "macos", "aarch64").unwrap_err();
        assert!(err.contains("only supports CPU"), "got {err}");
    }

    #[test]
    fn macos_rejects_rocm_rather_than_serving_the_cpu_archive() {
        // A ROCm spec has no CUDA version either, so the CUDA-shaped
        // guard passed it through and the macOS filename branch handed
        // back the CPU archive: a CPU libtorch installed as `rocm70`,
        // with nothing anywhere saying so.
        for spec in [&ROCM70_SPEC, &ROCM71_SPEC] {
            let err = download_url_for(spec, "macos", "aarch64").unwrap_err();
            assert!(err.contains("only supports CPU"), "got {err}");
        }
    }

    #[test]
    fn macos_intel_is_rejected_with_a_reason() {
        let err = download_url_for(&CPU_SPEC, "macos", "x86_64").unwrap_err();
        assert!(err.contains("Apple Silicon"), "got {err}");
    }

    #[test]
    fn unsupported_platform_is_rejected() {
        // linux-aarch64 has no upstream libtorch archive; `fdl libtorch
        // build` from source is the path there.
        let err = download_url_for(&CPU_SPEC, "linux", "aarch64").unwrap_err();
        assert!(err.contains("Unsupported platform"), "got {err}");
    }

    /// `otool -L` shape as upstream's arm64 archive actually prints it:
    /// the file name first, then one indented dependency per line with
    /// trailing version parens.
    const OTOOL_LIBTORCH_CPU: &str = "\
libtorch/lib/libtorch_cpu.dylib:
\t@rpath/libtorch_cpu.dylib (compatibility version 0.0.0, current version 0.0.0)
\t/opt/homebrew/opt/libomp/lib/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
\t@rpath/libc10.dylib (compatibility version 0.0.0, current version 0.0.0)
\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1700.255.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)
";

    #[test]
    fn the_absolute_libomp_dependency_is_the_only_one_rewritten() {
        // Precisely one line qualifies. The self-reference on line 2 and
        // the two /usr/lib system libraries must NOT be touched: this
        // fixes one upstream defect, and widening it to "every absolute
        // path" would repoint libc++ at a sibling that does not exist.
        assert_eq!(
            absolute_libomp_refs(OTOOL_LIBTORCH_CPU),
            vec!["/opt/homebrew/opt/libomp/lib/libomp.dylib".to_string()],
        );
    }

    #[test]
    fn an_already_relative_libomp_is_left_alone() {
        // Idempotence: the second `fdl libtorch download` over the same
        // variant must find nothing to do, or it re-signs on every run.
        let patched = OTOOL_LIBTORCH_CPU.replace(
            "/opt/homebrew/opt/libomp/lib/libomp.dylib",
            "@loader_path/libomp.dylib",
        );
        assert!(absolute_libomp_refs(&patched).is_empty(), "{patched}");
        // `@rpath` spelling too, in case upstream fixes it their way.
        let upstream_fixed = OTOOL_LIBTORCH_CPU.replace("/opt/homebrew/opt/libomp/lib/", "@rpath/");
        assert!(absolute_libomp_refs(&upstream_fixed).is_empty());
    }

    #[test]
    fn a_libomp_at_another_absolute_prefix_still_qualifies() {
        // The Homebrew prefix is not universal (Intel Macs use
        // /usr/local, and a custom prefix is legal), so the match is on
        // the library, not on the directory upstream happened to use.
        let intel = OTOOL_LIBTORCH_CPU.replace("/opt/homebrew/opt", "/usr/local/opt");
        assert_eq!(
            absolute_libomp_refs(&intel),
            vec!["/usr/local/opt/libomp/lib/libomp.dylib".to_string()],
        );
    }

    #[test]
    fn a_dump_with_no_dependencies_yields_nothing() {
        assert!(absolute_libomp_refs("").is_empty());
        assert!(absolute_libomp_refs("libomp.dylib:\n").is_empty());
    }

    #[test]
    fn force_linux_ignores_the_host() {
        // The container is Linux whatever the host is, so the docker path
        // must never pick up a macOS or Windows filename.
        let url = download_url(&CU128_SPEC, true).unwrap();
        assert!(url.contains("libtorch-shared-with-deps-"), "got {url}");
        assert!(!url.contains("-win-"), "got {url}");
        assert!(!url.contains("macos"), "got {url}");
    }

    // Variant routing. Asserted through the pure `variant_for_gpus` so no
    // arm depends on the host's own hardware.

    fn gpu(vendor: GpuVendor, arch: &str) -> system::GpuInfo {
        system::GpuInfo {
            index: 0,
            vendor,
            name: format!("test {arch}"),
            arch: flodl_hw::GpuArch::parse(vendor, arch)
                .unwrap_or_else(|| panic!("unparsable arch {arch}")),
            total_memory_mb: 8192,
        }
    }

    #[test]
    fn no_gpu_routes_to_cpu() {
        assert_eq!(variant_for_gpus(&[]).arch_variant, "cpu");
    }

    #[test]
    fn a_covered_amd_gpu_routes_to_rocm() {
        // The bug this guards: a gfx target the ROCm archive ships kernels
        // for was routed to the CPU variant, so an AMD box trained on CPU.
        // gfx950 (MI350 class) and gfx1150/gfx1151 (Strix APUs) are in the
        // archive and were missing from the covered list.
        for arch in [
            "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1151", "gfx1201",
        ] {
            let v = variant_for_gpus(&[gpu(GpuVendor::Amd, arch)]);
            assert_eq!(v.arch_variant, "rocm7.0", "{arch} should route to ROCm");
        }
    }

    #[test]
    fn a_perf_db_only_target_is_not_covered() {
        // gfx900 and gfx906 appear in the archive with MIOpen performance
        // databases and no rocBLAS kernels at all. Calling that "covered"
        // admits a box that dies at its first BLAS call instead of being
        // told, here, that CPU is what this build can honestly offer.
        for arch in ["gfx900", "gfx906"] {
            let v = variant_for_gpus(&[gpu(GpuVendor::Amd, arch)]);
            assert_eq!(v.arch_variant, "cpu", "{arch} ships no kernels");
            assert!(rocm_covered(&[gpu(GpuVendor::Amd, arch)]).is_empty());
        }
    }

    #[test]
    fn auto_never_picks_the_newer_rocm_bundle() {
        // Deliberate: the host's ROCm loads ahead of the bundle, so the
        // oldest offered bundle is the one that serves every 7.x host.
        // Reaching 7.1 is an explicit request, not a detection outcome.
        for arch in ["gfx942", "gfx950", "gfx1151"] {
            assert_eq!(
                variant_for_gpus(&[gpu(GpuVendor::Amd, arch)]).arch_variant,
                "rocm7.0"
            );
        }
        assert_eq!(resolve_variant(&Variant::Rocm71).arch_variant, "rocm7.1");
    }

    #[test]
    fn an_uncovered_amd_gpu_routes_to_cpu() {
        // No bundled Tensile kernels for this target, so ROCm would build
        // but not run. Proves the previous test is not vacuously green.
        let v = variant_for_gpus(&[gpu(GpuVendor::Amd, "gfx803")]);
        assert_eq!(v.arch_variant, "cpu");
    }

    #[test]
    fn a_partly_covered_amd_set_still_routes_to_rocm() {
        let v = variant_for_gpus(&[gpu(GpuVendor::Amd, "gfx942"), gpu(GpuVendor::Amd, "gfx803")]);
        assert_eq!(v.arch_variant, "rocm7.0");
    }

    #[test]
    fn a_mixed_vendor_box_routes_to_cuda() {
        // One libtorch build serves one vendor; NVIDIA is the pick, and
        // the AMD device must not drag the result to ROCm or to CPU.
        let v = variant_for_gpus(&[
            gpu(GpuVendor::Nvidia, "sm_120"),
            gpu(GpuVendor::Amd, "gfx1100"),
        ]);
        assert_eq!(v.arch_variant, "cu128");
    }

    #[test]
    fn rocm_covered_selects_only_supported_amd_devices() {
        // The setup wizard routes on this, so it must not count an NVIDIA
        // card nor an AMD target the archive ships no kernels for.
        let gpus = vec![
            gpu(GpuVendor::Nvidia, "sm_120"),
            gpu(GpuVendor::Amd, "gfx942"),
            gpu(GpuVendor::Amd, "gfx803"),
        ];
        let covered = rocm_covered(&gpus);
        assert_eq!(covered.len(), 1);
        assert_eq!(covered[0].arch_label(), "gfx942");
    }

    #[test]
    fn nvidia_routing_is_unchanged() {
        assert_eq!(
            variant_for_gpus(&[gpu(GpuVendor::Nvidia, "sm_120")]).arch_variant,
            "cu128"
        );
        assert_eq!(
            variant_for_gpus(&[gpu(GpuVendor::Nvidia, "sm_61")]).arch_variant,
            "cu126"
        );
    }
}
