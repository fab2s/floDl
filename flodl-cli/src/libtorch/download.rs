//! `fdl libtorch download` -- download pre-built libtorch.

use std::fs;
use std::path::{Path, PathBuf};

use crate::context::Context;
use crate::util::http;
use crate::util::archive;
use crate::util::system;
use super::detect;

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

// ---------------------------------------------------------------------------
// Download options
// ---------------------------------------------------------------------------

pub enum Variant {
    Cpu,
    Cuda126,
    Cuda128,
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

fn download_url(spec: &VariantSpec, force_linux: bool) -> Result<String, String> {
    // `force_linux` short-circuits host detection: the binary is destined
    // for a Linux Docker container, so we always want the Linux x86_64
    // build regardless of what the host is.
    let (os, arch) = if force_linux {
        ("linux", "x86_64")
    } else {
        (std::env::consts::OS, std::env::consts::ARCH)
    };

    match (os, arch) {
        ("linux", "x86_64") => {}
        ("macos", "aarch64") => {
            if spec.arch_cuda != "none" {
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
        ("windows", "x86_64") => {}
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

    // Linux and Windows use the same URL pattern
    let filename = match spec.arch_variant {
        "cpu" => format!(
            "libtorch-shared-with-deps-{}%2Bcpu.zip",
            LIBTORCH_VERSION
        ),
        variant => format!(
            "libtorch-shared-with-deps-{}%2B{}.zip",
            LIBTORCH_VERSION, variant
        ),
    };

    let bucket = spec.arch_variant; // "cpu", "cu126", "cu128"
    Ok(format!(
        "https://download.pytorch.org/libtorch/{}/{}",
        bucket, filename
    ))
}

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------

fn auto_detect_variant() -> &'static VariantSpec {
    let gpus = system::detect_gpus();
    if gpus.is_empty() {
        println!("  No GPU detected. Using CPU variant.");
        return &CPU_SPEC;
    }

    // Every variant below is an NVIDIA CUDA build, so the capability
    // span is computed over NVIDIA devices only.
    let majors: Vec<u32> = gpus.iter().filter_map(|g| g.sm_major()).collect();
    if majors.is_empty() {
        // A GPU box with no NVIDIA card. Falling through to CPU is the
        // right variant, but say WHY, or this reads as "no GPU found"
        // on a machine the user can see has one.
        let other: Vec<String> = gpus
            .iter()
            .map(|g| format!("{} ({})", g.short_name(), g.arch_label()))
            .collect();
        println!(
            "  Detected a non-NVIDIA GPU ({}). `fdl libtorch download` only \n\
             publishes CUDA and CPU variants today, so the CPU variant is \n\
             selected. Use `--variant` to override.",
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

fn resolve_variant(variant: &Variant) -> &'static VariantSpec {
    match variant {
        Variant::Cpu => &CPU_SPEC,
        Variant::Cuda126 => &CU126_SPEC,
        Variant::Cuda128 => &CU128_SPEC,
        Variant::Auto => auto_detect_variant(),
    }
}

// ---------------------------------------------------------------------------
// Core download logic
// ---------------------------------------------------------------------------

pub fn run(opts: DownloadOpts) -> Result<(), String> {
    let ctx = Context::resolve();
    run_with_context(opts, &ctx)
}

/// Run with an explicit context (used by `setup` which has its own context).
pub fn run_with_context(opts: DownloadOpts, ctx: &Context) -> Result<(), String> {
    let spec = resolve_variant(&opts.variant);
    let url = download_url(spec, opts.force_linux)?;

    // Determine install path
    let install_path = if let Some(ref p) = opts.custom_path {
        p.clone()
    } else {
        ctx.root.join(format!("libtorch/precompiled/{}", spec.dir_name))
    };

    let variant_id = format!("precompiled/{}", spec.dir_name);

    println!();
    println!("  libtorch {} ({})", LIBTORCH_VERSION, spec.label);
    println!("  URL:  {}", url);
    println!("  Path: {}", install_path.display());

    if opts.dry_run {
        println!();
        println!("  [dry-run] Would download and extract to above path.");
        return Ok(());
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
            return Ok(());
        }

        println!();
        println!(
            "  Removing existing installation (version: {})...",
            existing_ver.as_deref().unwrap_or("unknown")
        );
        fs::remove_dir_all(&install_path)
            .map_err(|e| format!("cannot remove {}: {}", install_path.display(), e))?;
    }

    // Download to temp file
    let tmp_dir = std::env::temp_dir();
    let tmp_zip = tmp_dir.join(format!("libtorch-{}-{}.zip", spec.dir_name, LIBTORCH_VERSION));

    println!();
    println!("  Downloading...");
    http::download_file(&url, &tmp_zip)?;

    // Extract to temp directory (zip contains a top-level "libtorch/" dir)
    let tmp_extract = tmp_dir.join(format!("libtorch-extract-{}", std::process::id()));
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

    // Move all files from extracted dir to install path
    move_contents(source, &install_path)?;

    // Cleanup temp files
    let _ = fs::remove_file(&tmp_zip);
    let _ = fs::remove_dir_all(&tmp_extract);

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
        if spec.arch_cuda != "none" {
            println!("  Run 'fdl cuda-test' to verify.");
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
        println!(
            "    export LD_LIBRARY_PATH=\"{}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\"",
            install_path.display()
        );
        println!();
        println!("  Or start a new floDl project:");
        println!("    fdl init my-project");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Move all files and directories from `src` into `dest`.
fn move_contents(src: &Path, dest: &Path) -> Result<(), String> {
    let entries = fs::read_dir(src)
        .map_err(|e| format!("cannot read {}: {}", src.display(), e))?;

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
    fs::create_dir_all(dest)
        .map_err(|e| format!("cannot create {}: {}", dest.display(), e))?;

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
