//! `fdl libtorch build` -- compile libtorch from PyTorch source.
//!
//! Two backends: Docker (isolated, reproducible) or native (faster, requires
//! CUDA toolkit + build tools on host). Auto-detects available backends and
//! asks the user when both are present.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::context::Context;
use crate::util::docker;
use crate::util::prompt;
use crate::util::system;
use super::detect;

const DOCKERFILE_CONTENT: &str = include_str!("../../assets/Dockerfile.cuda.source");
const IMAGE_NAME: &str = "flodl-libtorch-builder";
const LIBTORCH_VERSION: &str = "2.10.0";
const PYTORCH_VERSION: &str = "v2.10.0";

const PYTHON_DEPS: &[&str] = &[
    "typing_extensions", "pyyaml", "filelock",
    "jinja2", "networkx", "sympy", "packaging",
];

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

#[derive(Default)]
pub enum BuildBackend {
    /// Auto-detect: ask user if both available, otherwise use whatever works.
    #[default]
    Auto,
    /// Force Docker build.
    Docker,
    /// Force native build (no Docker).
    Native,
}

pub struct BuildOpts {
    /// Override CUDA architectures (semicolon-separated, e.g. "6.1;12.0").
    /// None = auto-detect from GPUs.
    pub archs: Option<String>,
    /// Override MAX_JOBS for compilation. Default: 6.
    pub max_jobs: usize,
    /// Print what would happen without building.
    pub dry_run: bool,
    /// Which backend to use.
    pub backend: BuildBackend,
}

impl Default for BuildOpts {
    fn default() -> Self {
        Self {
            archs: None,
            max_jobs: 6,
            dry_run: false,
            backend: BuildBackend::Auto,
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-detect GPU architectures
// ---------------------------------------------------------------------------

fn detect_arch_list() -> Result<String, String> {
    let gpus = system::detect_gpus();
    if gpus.is_empty() {
        return Err(
            "No NVIDIA GPUs detected.\n\
             Source builds require GPUs to auto-detect architectures.\n\
             Use --archs to specify manually (e.g. --archs \"8.6;12.0\")."
                .into(),
        );
    }

    // Collect unique compute capabilities, sorted numerically. This
    // list feeds nvcc's TORCH_CUDA_ARCH_LIST, so it is NVIDIA-only by
    // construction: a non-NVIDIA device contributes no capability and
    // is skipped rather than defaulted to one.
    let mut caps: Vec<(u32, u32)> = gpus
        .iter()
        .filter_map(|g| Some((g.sm_major()?, g.sm_minor()?)))
        .collect();
    caps.sort();
    caps.dedup();
    let caps: Vec<String> = caps.iter().map(|(ma, mi)| format!("{}.{}", ma, mi)).collect();

    println!("  GPUs detected:");
    for g in &gpus {
        println!("    [{}] {} ({})", g.index, g.short_name(), g.arch_label());
    }

    Ok(caps.join(";"))
}

// ---------------------------------------------------------------------------
// Native toolchain detection
// ---------------------------------------------------------------------------

struct NativeTools {
    nvcc: bool,
    cmake: bool,
    python3: bool,
    git: bool,
    gcc: bool,
}

impl NativeTools {
    fn detect() -> Self {
        Self {
            nvcc: has_tool("nvcc"),
            cmake: has_tool("cmake"),
            python3: has_tool("python3"),
            git: has_tool("git"),
            gcc: has_tool("gcc") || has_tool("cc"),
        }
    }

    fn ready(&self) -> bool {
        self.nvcc && self.cmake && self.python3 && self.git && self.gcc
    }

    fn missing(&self) -> Vec<&'static str> {
        let mut m = Vec::new();
        if !self.nvcc { m.push("nvcc (CUDA toolkit)"); }
        if !self.cmake { m.push("cmake"); }
        if !self.python3 { m.push("python3"); }
        if !self.git { m.push("git"); }
        if !self.gcc { m.push("gcc/cc (C++ compiler)"); }
        m
    }
}

fn has_tool(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

fn select_backend(backend: &BuildBackend) -> Result<&'static str, String> {
    let has_docker = docker::has_docker();
    let native = NativeTools::detect();

    match backend {
        BuildBackend::Docker => {
            if !has_docker {
                return Err(
                    "Docker was requested but is not available.\n\
                     Install Docker: https://docs.docker.com/engine/install/"
                        .into(),
                );
            }
            Ok("docker")
        }
        BuildBackend::Native => {
            if !native.ready() {
                let missing = native.missing();
                return Err(format!(
                    "Native build was requested but these tools are missing:\n  {}\n\n\
                     Install them or use --docker instead.",
                    missing.join("\n  ")
                ));
            }
            Ok("native")
        }
        BuildBackend::Auto => {
            if has_docker && native.ready() {
                // Both available, ask the user
                println!();
                println!("  Both Docker and native toolchains are available.");
                println!();
                let choice = prompt::ask_choice(
                    "  Build method",
                    &[
                        "Docker (isolated, reproducible, resumes via layer cache)",
                        "Native (faster, uses your host CUDA toolkit directly)",
                    ],
                    1,
                );
                Ok(if choice == 2 { "native" } else { "docker" })
            } else if has_docker {
                println!("  Using Docker (native toolchain not complete).");
                Ok("docker")
            } else if native.ready() {
                println!("  Using native build (Docker not available).");
                Ok("native")
            } else {
                let missing = native.missing();
                Err(format!(
                    "Cannot build libtorch. Need either:\n\n\
                     \x20 Docker: https://docs.docker.com/engine/install/\n\n\
                     Or native tools (missing: {})",
                    missing.join(", ")
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(opts: BuildOpts) -> Result<(), String> {
    let ctx = Context::resolve();

    // Determine architectures
    let archs = match &opts.archs {
        Some(a) => {
            println!("  Using specified architectures: {}", a);
            a.clone()
        }
        None => detect_arch_list()?,
    };

    let arch_dir = system::arch_dir_name(&archs);
    let install_path = ctx.root.join(format!("libtorch/builds/{}", arch_dir));
    let variant_id = format!("builds/{}", arch_dir);

    // Select backend
    let backend = select_backend(&opts.backend)?;

    println!();
    println!("  libtorch source build");
    println!("  Archs:   {}", archs);
    println!("  Output:  {}", install_path.display());
    println!("  Jobs:    {}", opts.max_jobs);
    println!("  Method:  {}", backend);
    println!();

    if opts.dry_run {
        println!("  [dry-run] Would build libtorch from source via {}.", backend);
        println!("  This typically takes 2-6 hours depending on CPU cores.");
        return Ok(());
    }

    println!("  This will take 2-6 hours. You can safely Ctrl-C and resume later.");
    println!();

    let install_str = install_path.to_str().unwrap_or("libtorch/builds");
    match backend {
        "docker" => build_docker(&archs, install_str, opts.max_jobs)?,
        "native" => build_native(&archs, install_str, &ctx, opts.max_jobs)?,
        _ => unreachable!(),
    }

    // Verify
    let lib_dir = install_path.join("lib");
    if !lib_dir.join("libtorch.so").exists() && !lib_dir.join("libtorch.dylib").exists() {
        return Err(format!(
            "libtorch library not found at {}.\n\
             The build may have failed silently.",
            lib_dir.display()
        ));
    }

    // Write .arch metadata
    let arch_spaces = archs.replace(';', " ");
    let arch_content = format!(
        "cuda=12.8\ntorch={}\narchs={}\nsource=compiled\n",
        LIBTORCH_VERSION, arch_spaces
    );
    fs::write(install_path.join(".arch"), arch_content)
        .map_err(|e| format!("cannot write .arch: {}", e))?;

    // Set as active
    detect::set_active(&ctx.root, &variant_id)?;

    println!();
    println!("  ================================================");
    println!("  libtorch {} (source build) complete!", LIBTORCH_VERSION);
    println!("  Archs:  {}", arch_spaces);
    println!("  Path:   {}", install_path.display());
    println!("  Active: {}", variant_id);
    println!("  ================================================");
    println!();
    if ctx.is_project {
        println!("  Run 'fdl gpu-test' to verify.");
    } else {
        println!("  To use, add to your shell profile:");
        println!("    export LIBTORCH=\"{}\"", install_path.display());
        // Shared recipe. A source build can be AMD too -- `arch_dir_name`
        // maps a `gfx…` arch list to a `gfx…` variant directory, which
        // `variant_vendor` then reads as AMD.
        let lib = format!("{}/lib", install_path.display());
        for line in detect::ld_library_path_lines(detect::variant_vendor(&variant_id), &lib) {
            println!("    {line}");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Docker backend
// ---------------------------------------------------------------------------

fn build_docker(archs: &str, install_path: &str, max_jobs: usize) -> Result<(), String> {
    println!("  Docker layer caching means restarting picks up where it left off.");
    println!();

    // Write Dockerfile to temp location
    let tmp_dir = std::env::temp_dir();
    let dockerfile_path = tmp_dir.join("flodl-libtorch-builder.Dockerfile");
    {
        let mut f = fs::File::create(&dockerfile_path)
            .map_err(|e| format!("cannot write Dockerfile: {}", e))?;
        f.write_all(DOCKERFILE_CONTENT.as_bytes())
            .map_err(|e| format!("cannot write Dockerfile: {}", e))?;
    }

    // Build the Docker image
    println!("  Building Docker image...");
    let status = docker::docker_run(&[
        "build",
        "-t",
        IMAGE_NAME,
        "--build-arg",
        &format!("TORCH_CUDA_ARCH_LIST={}", archs),
        "--build-arg",
        &format!("MAX_JOBS={}", max_jobs),
        "-f",
        dockerfile_path
            .to_str()
            .ok_or("temp path not UTF-8")?,
        ".",
    ])?;

    let _ = fs::remove_file(&dockerfile_path);

    if !status.success() {
        return Err(format!(
            "Docker build failed (exit code {}).\n\
             Check the output above for errors.\n\
             You can re-run this command to resume (Docker caches completed layers).",
            status.code().unwrap_or(-1)
        ));
    }

    // Extract libtorch from the builder image
    println!();
    println!("  Extracting libtorch from builder image...");

    let container_out = docker::docker_output(&["create", IMAGE_NAME])?;
    if !container_out.status.success() {
        return Err("failed to create container from builder image".into());
    }
    let container_id = String::from_utf8_lossy(&container_out.stdout)
        .trim()
        .to_string();

    fs::create_dir_all(install_path)
        .map_err(|e| format!("cannot create {}: {}", install_path, e))?;

    let cp_status = docker::docker_run(&[
        "cp",
        &format!("{}:/usr/local/libtorch/.", container_id),
        install_path,
    ])?;

    let _ = docker::docker_output(&["rm", &container_id]);

    if !cp_status.success() {
        return Err("failed to extract libtorch from builder container".into());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Native backend
// ---------------------------------------------------------------------------

fn build_native(archs: &str, install_path: &str, ctx: &Context, max_jobs: usize) -> Result<(), String> {
    let build_dir = ctx.root.join("libtorch/.build-cache/pytorch");

    // Clone PyTorch if not cached
    if !build_dir.join(".git").exists() {
        println!("  Cloning PyTorch {}...", PYTORCH_VERSION);
        fs::create_dir_all(ctx.root.join("libtorch/.build-cache"))
            .map_err(|e| format!("cannot create build cache: {}", e))?;

        let status = Command::new("git")
            .args([
                "clone", "--depth", "1",
                "--branch", PYTORCH_VERSION,
                "--recurse-submodules", "--shallow-submodules",
                "https://github.com/pytorch/pytorch.git",
                build_dir.to_str().ok_or("path not UTF-8")?,
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|e| format!("failed to run git: {}", e))?;

        if !status.success() {
            // Clean up failed clone
            let _ = fs::remove_dir_all(build_dir);
            return Err("git clone failed. Check your network connection.".into());
        }
    } else {
        println!("  Using cached PyTorch source at {}", build_dir.display());
    }

    // Install Python dependencies
    println!("  Checking Python dependencies...");
    let pip_status = Command::new("pip3")
        .args(["install", "--quiet"])
        .args(PYTHON_DEPS)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status();

    // Try --break-system-packages if the first attempt fails (Ubuntu 24.04+)
    if pip_status.is_err() || !pip_status.unwrap().success() {
        let _ = Command::new("pip3")
            .args(["install", "--quiet", "--break-system-packages"])
            .args(PYTHON_DEPS)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status();
    }

    // Build libtorch
    println!("  Building libtorch (TORCH_CUDA_ARCH_LIST=\"{}\", MAX_JOBS={})...", archs, max_jobs);
    println!();

    let status = Command::new("python3")
        .arg("tools/build_libtorch.py")
        .current_dir(&build_dir)
        .env("TORCH_CUDA_ARCH_LIST", archs)
        .env("USE_CUDA", "1")
        .env("USE_CUDNN", "1")
        .env("USE_NCCL", "1")
        .env("USE_DISTRIBUTED", "1")
        .env("BUILD_SHARED_LIBS", "ON")
        .env("CMAKE_BUILD_TYPE", "Release")
        .env("MAX_JOBS", max_jobs.to_string())
        .env("BUILD_PYTHON", "OFF")
        .env("BUILD_TEST", "OFF")
        .env("BUILD_CAFFE2", "OFF")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run build_libtorch.py: {}", e))?;

    if !status.success() {
        return Err(format!(
            "Native build failed (exit code {}).\n\
             Check the output above for errors.\n\
             The PyTorch source is cached at {} -- re-running will skip the clone.",
            status.code().unwrap_or(-1),
            build_dir.display()
        ));
    }

    // Copy output to install path
    println!();
    println!("  Packaging libtorch to {}...", install_path);

    let torch_dir = build_dir.join("torch");
    fs::create_dir_all(install_path)
        .map_err(|e| format!("cannot create {}: {}", install_path, e))?;

    for subdir in ["lib", "include", "share"] {
        let src = torch_dir.join(subdir);
        let dst = Path::new(install_path).join(subdir);
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        }
    }

    // Bundle cuDNN: copy system cuDNN libs into the install path so
    // deploys to hosts without a system cuDNN install (e.g. a bare
    // VM guest) can still find libcudnn_graph.so.9 and the
    // other sub-libs via libtorch's lib dir on LD_LIBRARY_PATH.
    // Best-effort on the native path: walks common cuDNN install
    // prefixes and copies the first match it finds. Silently skipped
    // when system cuDNN isn't present (Docker path bundles them via
    // Dockerfile.cuda.source).
    let dst_lib = Path::new(install_path).join("lib");
    bundle_system_cudnn(&dst_lib);

    Ok(())
}

/// Locate system cuDNN libs (libcudnn.so.9 + sub-libs) and copy them
/// into `dst_lib`. Walks common install prefixes; copies every file
/// matching `libcudnn*.so*` from the first prefix that contains any.
/// Returns silently (best-effort) — the Docker build bundles via
/// Dockerfile, so failure here only affects the rarer native path.
fn bundle_system_cudnn(dst_lib: &Path) {
    let candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
    ];
    for prefix in candidates {
        let dir = Path::new(prefix);
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut copied = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            // Match libcudnn.so* and libcudnn_*.so* but skip static
            // archives (.a) and headers.
            if !name.starts_with("libcudnn") {
                continue;
            }
            if !name.contains(".so") {
                continue;
            }
            let dst = dst_lib.join(name);
            // `fs::copy` resolves symlinks → loses the SONAME chain
            // PyTorch's loader walks. Re-create symlinks; fall back to
            // copy for real files.
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.file_type().is_symlink() {
                let _ = fs::remove_file(&dst);
                // Re-create the link on unix (see the SONAME note above).
                // Non-unix has no unprivileged symlink — and this scan is
                // Linux-shaped anyway — so degrade to a resolving copy;
                // the gate is about compiling on Windows, not behavior.
                #[cfg(unix)]
                if let Ok(target) = fs::read_link(&path)
                    && std::os::unix::fs::symlink(&target, &dst).is_ok() {
                        copied += 1;
                    }
                #[cfg(not(unix))]
                if fs::copy(&path, &dst).is_ok() {
                    copied += 1;
                }
            } else if fs::copy(&path, &dst).is_ok() {
                copied += 1;
            }
        }
        if copied > 0 {
            println!(
                "  Bundled {copied} cuDNN file(s) from {prefix} into the libtorch install",
            );
            return;
        }
    }
    // No cuDNN found on the build host. Not fatal — most users build
    // with cuDNN present, and the Docker path bundles regardless. Note
    // the gap so it's visible if a deploy later fails to find cuDNN.
    println!(
        "  Note: no system cuDNN detected in /usr/lib/x86_64-linux-gnu, \
         /usr/local/cuda/lib64, or /usr/local/lib; install path will \
         rely on the target host providing cuDNN at runtime.",
    );
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
