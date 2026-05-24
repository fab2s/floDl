//! `fdl nccl build` -- compile libnccl from NVIDIA's NCCL source.
//!
//! Why this exists: libtorch wheels statically link NCCL into
//! `libtorch_cuda.so`. On heterogeneous-arch rigs (Pascal sm_61 + Blackwell
//! sm_120, etc.), that static NCCL can corrupt the shared CUmodule loader
//! state during cross-host bootstrap, which surfaces as
//! `cudaErrorNoKernelImageForDevice` on otherwise-working kernels. Building
//! libnccl as a standalone `.so` and `LD_PRELOAD`-ing it gives NCCL its own
//! CUmodule scope and sidesteps the issue.
//!
//! Output layout (mirrors `fdl libtorch build`):
//!
//! ```text
//! libtorch/nccl/builds/v<version>-<archs>/
//!     include/nccl.h
//!     lib/libnccl.so          -> libnccl.so.2
//!     lib/libnccl.so.2        -> libnccl.so.<X.Y.Z>
//!     lib/libnccl.so.<X.Y.Z>
//!     lib/libnccl_static.a
//!     lib/pkgconfig/
//! ```

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::context::Context;
use crate::libtorch::detect as libtorch_detect;
use crate::util::docker;
use crate::util::system;

const DOCKERFILE_CONTENT: &str = include_str!("../../assets/Dockerfile.nccl.source");
const IMAGE_PREFIX: &str = "flodl-nccl-build";

pub struct BuildOpts {
    /// NCCL git tag (e.g. "v2.27.5-1"). None = infer from the active
    /// libtorch's bundled NCCL version (the version cross-rank handshake
    /// requires us to match).
    pub tag: Option<String>,
    /// Override CUDA architectures (semicolon-separated, e.g. "6.1;12.0").
    /// None = auto-detect from local GPUs.
    pub archs: Option<String>,
    /// Override MAX_JOBS for compilation. Default: 6.
    pub max_jobs: usize,
    /// Print what would happen without building.
    pub dry_run: bool,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            tag: None,
            archs: None,
            max_jobs: 6,
            dry_run: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-detect NCCL version from active libtorch
// ---------------------------------------------------------------------------

/// Scan a libtorch_cuda.so file for NCCL's self-identification string and
/// return a build tag like `v2.27.5-1`. Looks for `NCCL version <X.Y.Z>`
/// immediately followed by `+cuda` (NCCL's runtime format is
/// `NCCL version 2.27.5+cuda12.8`). The `+cuda` suffix disambiguates from
/// PyTorch's error-message strings that mention NCCL versions as version
/// floors (`NCCL version 2.27.0 or later`). The `-1` patch suffix is
/// NCCL's default tag revision for a given X.Y.Z release; override with
/// `--tag` if you need a later revision (`-2`, `-3`, ...).
fn detect_nccl_tag_from_libtorch_cuda(path: &std::path::Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let needle = b"NCCL version ";
    let mut idx = 0;
    while idx + needle.len() < bytes.len() {
        if &bytes[idx..idx + needle.len()] != needle {
            idx += 1;
            continue;
        }
        let after = &bytes[idx + needle.len()..];
        let end = after
            .iter()
            .position(|&b| !(b.is_ascii_digit() || b == b'.'))
            .unwrap_or(after.len());
        if end > 0 && after.get(end) == Some(&b'+') {
            let version = std::str::from_utf8(&after[..end]).ok()?;
            if !version.is_empty() {
                return Some(format!("v{}-1", version));
            }
        }
        idx += needle.len();
    }
    None
}

fn resolve_tag(ctx: &Context, override_tag: Option<String>) -> Result<String, String> {
    if let Some(tag) = override_tag {
        if tag.trim().is_empty() {
            return Err(
                "--tag cannot be empty. Pass a valid NCCL git tag (e.g. v2.27.5-1) \
                 or omit --tag to infer from the active libtorch."
                    .into(),
            );
        }
        return Ok(tag);
    }

    let active = libtorch_detect::read_active(&ctx.root).ok_or_else(|| {
        "No active libtorch variant; cannot infer NCCL version.\n\
         Either activate one with `fdl libtorch activate <variant>` \
         or pass --tag explicitly (e.g. --tag v2.27.5-1)."
            .to_string()
    })?;

    let libtorch_cuda = ctx
        .root
        .join("libtorch")
        .join(&active.path)
        .join("lib")
        .join("libtorch_cuda.so");
    if !libtorch_cuda.exists() {
        return Err(format!(
            "Active libtorch variant {} has no libtorch_cuda.so at {}.\n\
             Pass --tag explicitly or fix the libtorch installation.",
            active.path,
            libtorch_cuda.display()
        ));
    }

    let tag = detect_nccl_tag_from_libtorch_cuda(&libtorch_cuda).ok_or_else(|| {
        format!(
            "Could not detect bundled NCCL version in {}.\n\
             Pass --tag explicitly (e.g. --tag v2.27.5-1).",
            libtorch_cuda.display()
        )
    })?;
    println!(
        "  Inferred NCCL tag from libtorch ({}): {}",
        active.path, tag
    );
    Ok(tag)
}

// ---------------------------------------------------------------------------
// Auto-detect GPU architectures
// ---------------------------------------------------------------------------

fn detect_arch_list() -> Result<String, String> {
    let gpus = system::detect_gpus();
    if gpus.is_empty() {
        return Err(
            "No NVIDIA GPUs detected.\n\
             NCCL builds need GPU arch info to set NVCC_GENCODE.\n\
             Use --archs to specify manually (e.g. --archs \"6.1;12.0\")."
                .into(),
        );
    }

    let mut caps: Vec<(u32, u32)> = gpus
        .iter()
        .map(|g| (g.sm_major, g.sm_minor))
        .collect();
    caps.sort();
    caps.dedup();
    let caps: Vec<String> = caps.iter().map(|(ma, mi)| format!("{}.{}", ma, mi)).collect();

    println!("  GPUs detected:");
    for g in &gpus {
        println!(
            "    [{}] {} (sm_{}{})",
            g.index, g.short_name(), g.sm_major, g.sm_minor
        );
    }

    Ok(caps.join(";"))
}

/// Strip the trailing `-N` patch suffix from an NCCL tag for directory
/// naming. `v2.27.5-1` -> `v2.27.5`. Leaves tags without a numeric patch
/// suffix untouched.
fn version_dir_part(tag: &str) -> String {
    if let Some((base, rest)) = tag.rsplit_once('-') {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return base.to_string();
        }
    }
    tag.to_string()
}

/// Convert "6.1;12.0" -> "sm61-sm120" for directory naming. Matches the
/// existing convention from `fdl libtorch build`.
fn arch_dir_name(archs: &str) -> String {
    archs
        .split(';')
        .map(|cap| {
            let clean = cap.replace('.', "");
            format!("sm{}", clean)
        })
        .collect::<Vec<_>>()
        .join("-")
}

/// Convert "6.1;12.0" -> "-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_120,code=sm_120"
/// for the NVCC_GENCODE build arg.
fn arch_gencode(archs: &str) -> String {
    archs
        .split(';')
        .map(|cap| {
            let clean = cap.replace('.', "");
            format!(
                "-gencode=arch=compute_{},code=sm_{}",
                clean, clean
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(opts: BuildOpts) -> Result<(), String> {
    let ctx = Context::resolve();

    if !docker::has_docker() {
        return Err(
            "Docker is required for `fdl nccl build`.\n\
             Install Docker: https://docs.docker.com/engine/install/"
                .into(),
        );
    }

    let tag = resolve_tag(&ctx, opts.tag)?;

    let archs = match &opts.archs {
        Some(a) => {
            println!("  Using specified architectures: {}", a);
            a.clone()
        }
        None => detect_arch_list()?,
    };

    let arch_dir = arch_dir_name(&archs);
    let gencode = arch_gencode(&archs);
    let version_short = version_dir_part(&tag);
    let install_path = ctx.root.join(format!(
        "libtorch/nccl/builds/{}-{}",
        version_short, arch_dir
    ));
    let image_tag = format!("{}:{}-{}", IMAGE_PREFIX, version_short, arch_dir);

    println!();
    println!("  NCCL source build");
    println!("  Tag:      {}", tag);
    println!("  Archs:    {}", archs);
    println!("  Gencode:  {}", gencode);
    println!("  Output:   {}", install_path.display());
    println!("  Jobs:     {}", opts.max_jobs);
    println!("  Image:    {}", image_tag);
    println!();

    if opts.dry_run {
        println!("  [dry-run] Would build NCCL {} for {} via Docker.", tag, archs);
        println!("  This typically takes 5-15 minutes.");
        return Ok(());
    }

    println!("  Building (5-15 min, mostly nvcc compile time)...");
    println!();

    build_docker(&tag, &gencode, &image_tag, opts.max_jobs)?;

    println!();
    println!("  Extracting build artifacts...");
    extract_artifacts(&image_tag, &install_path)?;

    println!();
    println!("  ================================================");
    println!("  NCCL {} (source build) complete!", tag);
    println!("  Archs:  {}", archs);
    println!("  Path:   {}", install_path.display());
    println!("  ================================================");
    println!();
    println!("  Wire into a cluster worker via:");
    println!("    worker.env:");
    println!(
        "      LD_PRELOAD: {}/lib/libnccl.so.2",
        install_path.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Docker build
// ---------------------------------------------------------------------------

fn build_docker(version: &str, gencode: &str, image_tag: &str, max_jobs: usize) -> Result<(), String> {
    // Write Dockerfile to temp location.
    let tmp_dir = std::env::temp_dir();
    let dockerfile_path = tmp_dir.join("flodl-nccl-builder.Dockerfile");
    {
        let mut f = fs::File::create(&dockerfile_path)
            .map_err(|e| format!("cannot write Dockerfile: {}", e))?;
        f.write_all(DOCKERFILE_CONTENT.as_bytes())
            .map_err(|e| format!("cannot write Dockerfile: {}", e))?;
    }

    let status = docker::docker_run(&[
        "build",
        "-f",
        dockerfile_path.to_str().ok_or("temp path not UTF-8")?,
        "--build-arg",
        &format!("NCCL_VERSION={}", version),
        "--build-arg",
        &format!("NVCC_GENCODE={}", gencode),
        "--build-arg",
        &format!("MAX_JOBS={}", max_jobs),
        "-t",
        image_tag,
        ".",
    ])?;

    let _ = fs::remove_file(&dockerfile_path);

    if !status.success() {
        return Err(format!(
            "Docker build failed (exit code {}).\n\
             Check the output above for errors.\n\
             You can re-run this command to resume (BuildKit caches NCCL checkout).",
            status.code().unwrap_or(-1)
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Extract from builder image
// ---------------------------------------------------------------------------

fn extract_artifacts(image_tag: &str, install_path: &PathBuf) -> Result<(), String> {
    let container_out = docker::docker_output(&["create", image_tag])?;
    if !container_out.status.success() {
        return Err("failed to create container from builder image".into());
    }
    let container_id = String::from_utf8_lossy(&container_out.stdout)
        .trim()
        .to_string();

    fs::create_dir_all(install_path)
        .map_err(|e| format!("cannot create {}: {}", install_path.display(), e))?;

    let mut last_err: Option<String> = None;
    for sub in ["lib", "include"] {
        let cp_status = docker::docker_run(&[
            "cp",
            &format!("{}:/usr/local/nccl/{}", container_id, sub),
            install_path
                .to_str()
                .ok_or("install path not UTF-8")?,
        ])?;
        if !cp_status.success() {
            last_err = Some(format!(
                "failed to extract {} (docker cp exit {})",
                sub,
                cp_status.code().unwrap_or(-1)
            ));
            break;
        }
    }

    let _ = docker::docker_output(&["rm", &container_id]);

    if let Some(e) = last_err {
        return Err(e);
    }

    // Sanity check: did the .so land?
    let lib_dir = install_path.join("lib");
    if !lib_dir.join("libnccl.so.2").exists() && !lib_dir.join("libnccl.so").exists() {
        return Err(format!(
            "libnccl not found under {}.\n\
             The build may have completed but produced no artifacts.",
            lib_dir.display()
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_dir_name_single() {
        assert_eq!(arch_dir_name("12.0"), "sm120");
    }

    #[test]
    fn arch_dir_name_multi() {
        assert_eq!(arch_dir_name("6.1;12.0"), "sm61-sm120");
    }

    #[test]
    fn arch_gencode_single() {
        assert_eq!(
            arch_gencode("12.0"),
            "-gencode=arch=compute_120,code=sm_120"
        );
    }

    #[test]
    fn arch_gencode_multi() {
        assert_eq!(
            arch_gencode("6.1;12.0"),
            "-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_120,code=sm_120"
        );
    }

    #[test]
    fn version_dir_strips_patch() {
        assert_eq!(version_dir_part("v2.27.5-1"), "v2.27.5");
        assert_eq!(version_dir_part("v2.27.5-12"), "v2.27.5");
    }

    #[test]
    fn version_dir_keeps_non_numeric() {
        assert_eq!(version_dir_part("v2.27.5"), "v2.27.5");
        assert_eq!(version_dir_part("master"), "master");
        assert_eq!(version_dir_part("v2.27.5-rc1"), "v2.27.5-rc1");
    }

    #[test]
    fn detect_picks_self_id_not_error_message() {
        // Synthetic blob with PT's error-message version BEFORE NCCL's
        // self-identification string. The detector must pick the latter
        // (the `+cuda` suffix is the disambiguator).
        let tmp = std::env::temp_dir().join("flodl-nccl-detect-test.bin");
        let blob = b"\
            ProcessGroupNCCL::shrink requires NCCL version 2.27.0 or later.\x00\
            padding padding padding\x00\
            NCCL version 2.27.5+cuda12.8\x00";
        std::fs::write(&tmp, blob).expect("write fixture");
        let tag = detect_nccl_tag_from_libtorch_cuda(&tmp);
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(tag, Some("v2.27.5-1".to_string()));
    }

    #[test]
    fn detect_returns_none_without_self_id() {
        // Only error-message NCCL strings, no `+cuda`-suffixed self-id.
        let tmp = std::env::temp_dir().join("flodl-nccl-detect-test-2.bin");
        let blob = b"Mismatched NCCL version detected\x00NCCL version 2.27.0 or later\x00";
        std::fs::write(&tmp, blob).expect("write fixture");
        let tag = detect_nccl_tag_from_libtorch_cuda(&tmp);
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(tag, None);
    }
}
