//! `fdl setup` -- interactive guided setup wizard.
//!
//! Detects hardware, downloads libtorch, optionally builds Docker images.

use crate::context::Context;
use crate::libtorch::{build, detect, download};
use crate::util::{docker, prompt, requirements, system};

#[derive(Default)]
pub struct SetupOpts {
    /// Skip all prompts, use auto-detected defaults.
    pub non_interactive: bool,
    /// Re-download/rebuild even if libtorch exists.
    pub force: bool,
}

pub fn run(opts: SetupOpts) -> Result<(), String> {
    println!();
    println!("  floDl Setup");
    println!("  ===========");
    println!();
    println!("  floDl is a Rust deep learning framework built on libtorch");
    println!("  (PyTorch's C++ backend). This wizard will help you set up");
    println!("  your development environment.");
    println!();

    // ---- Step 1: Detect system ----

    println!("  Step 1: Detecting your system");
    println!("  -----------------------------");
    println!();

    let cpu = system::cpu_model().unwrap_or_else(|| "Unknown".into());
    let threads = system::cpu_threads();
    let ram_gb = system::ram_total_gb();
    println!("  CPU:    {} ({} threads, {}GB RAM)", cpu, threads, ram_gb);

    let has_docker = docker::has_docker();
    let has_cargo = system::has_cargo();

    if has_docker {
        if let Some(v) = system::docker_version() {
            println!("  Docker: {}", v);
        } else {
            println!("  Docker: available");
        }
    } else {
        println!("  Docker: not found");
    }

    if has_cargo {
        println!("  Rust:   available");
    } else {
        println!("  Rust:   not found");
    }

    let gpus = system::detect_gpus();
    if !gpus.is_empty() {
        println!();
        println!("  GPUs:");
        for g in &gpus {
            println!(
                "    [{}] {} -- {}, {}GB VRAM",
                g.index,
                g.name,
                g.arch_label(),
                g.total_memory_mb / 1024
            );
        }
    } else {
        println!();
        println!("  GPU:    not detected (CPU-only mode)");
    }

    if !has_docker && !has_cargo {
        println!();
        println!("  You need at least one of these to continue:");
        println!();
        println!("    Rust:   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh");
        println!("    Docker: https://docs.docker.com/engine/install/");
        println!();
        println!("  Install one or both and run 'fdl setup' again.");
        return Err("no Rust or Docker found".into());
    }

    // Native prerequisites apply only to a native build: the Docker
    // path carries its own toolchain in the image. Cargo without Docker
    // is unambiguously native; with both available the path is not yet
    // chosen, so it is phrased as a note rather than a blocker.
    let tools = requirements::missing_host_tools();
    if !tools.is_empty() && has_cargo {
        let owned: Vec<String> = tools.iter().map(|t| (*t).to_string()).collect();
        println!();
        if has_docker {
            println!("  Note: building natively would also need: {}", tools.join(", "));
            println!("        {}", requirements::install_hint(&owned));
            println!("        (not needed if you build in the dev container)");
        } else {
            println!("  Native builds need these first: {}", tools.join(", "));
            println!("    {}", requirements::install_hint(&owned));
        }
    }

    // ---- Step 2: libtorch ----

    println!();
    println!("  Step 2: libtorch");
    println!("  ----------------");
    println!();
    println!("  floDl needs libtorch, PyTorch's C++ library.");
    println!("  This downloads pre-built binaries (~2GB for CUDA, ~200MB for CPU).");
    println!();

    let ctx = Context::resolve();
    let root = &ctx.root;

    if !ctx.is_project {
        println!("  Not inside a floDl project.");
        println!("  libtorch will be installed to: {}", ctx.libtorch_dir().display());
        println!();
    }

    let existing = detect::read_active(root);
    let mut skip_download = false;

    if !opts.force {
        if let Some(ref info) = existing {
            let is_cuda = info.cuda_version.as_deref() != Some("none");
            if is_cuda {
                println!("  Found existing CUDA libtorch: {}", info.path);
                if opts.non_interactive {
                    println!("  Keeping existing installation.");
                    skip_download = true;
                } else if !prompt::ask_yn("  Download fresh?", false) {
                    skip_download = true;
                }
                println!();
            } else {
                println!("  Found existing CPU libtorch.");
            }
        }
    }

    if !skip_download {
        // Always download CPU variant (useful as fallback).
        //
        // Special case: on macOS in a Docker-Mounted project the libtorch
        // ends up bind-mounted into a Linux container, so the host's
        // macOS arm64 (Mach-O) build is useless. Force Linux x86_64 in
        // that combination so the mount works as expected.
        let mounted_docker_project =
            ctx.is_project && ctx.root.join("Dockerfile").exists();
        let force_linux = cfg!(target_os = "macos") && mounted_docker_project;
        if force_linux {
            println!("  macOS + Docker-Mounted project: fetching Linux libtorch");
            println!("  for the container (host arch would not load inside Linux).");
        }
        println!("  Downloading CPU libtorch...");
        let cpu_opts = download::DownloadOpts {
            variant: download::Variant::Cpu,
            activate: false, // don't activate CPU if we'll also get CUDA
            force_linux,
            ..Default::default()
        };
        download::run_with_context(cpu_opts, &ctx)?;

        // The variant table below is CUDA-only, so the capability span
        // is taken over NVIDIA devices; a non-NVIDIA card contributes
        // none and leaves this branch inert rather than skewing it.
        let majors: Vec<u32> = gpus.iter().filter_map(|g| g.sm_major()).collect();

        // AMD libtorch. One build serves one vendor, so ROCm is chosen
        // only where there is no NVIDIA card to prefer; on a mixed box
        // the CUDA branch below runs instead.
        let amd: Vec<_> = gpus
            .iter()
            .filter(|g| g.vendor == system::GpuVendor::Amd)
            .collect();
        if !amd.is_empty() {
            let covered = download::rocm_covered(&gpus);
            if !majors.is_empty() {
                println!();
                println!("  AMD GPU(s) detected alongside NVIDIA. One libtorch build");
                println!("  serves one vendor, so the NVIDIA cards are set up here.");
                println!("  For the AMD cards: fdl libtorch download --rocm 7.0");
            } else if covered.is_empty() {
                let names: Vec<String> = amd
                    .iter()
                    .map(|g| format!("{} ({})", g.short_name(), g.arch_label()))
                    .collect();
                println!();
                println!("  AMD GPU(s) detected ({}) outside the ROCm 7.0", names.join(", "));
                println!("  build's targets, so only CPU libtorch is installed.");
                println!("  Covered targets: {}", download::rocm_archs());
            } else {
                println!();
                println!("  Downloading ROCm libtorch (rocm7.0 for your AMD GPU)...");
                let rocm_opts = download::DownloadOpts {
                    variant: download::Variant::Rocm70,
                    ..Default::default()
                };
                download::run_with_context(rocm_opts, &ctx)?;
            }
        }

        // CUDA libtorch
        if !majors.is_empty() {
            let lo_major = majors.iter().copied().min().unwrap_or(0);
            let hi_major = majors.iter().copied().max().unwrap_or(0);

            if lo_major < 7 && hi_major >= 10 {
                // Mixed architectures -- no single prebuilt covers both
                println!();
                println!("  Your GPUs span sm_{}.x to sm_{}.x.", lo_major, hi_major);
                println!("  No pre-built libtorch covers both architectures.");
                println!();

                // Check for existing source build
                let has_source_build = detect::list_variants(root)
                    .iter()
                    .any(|v| v.starts_with("builds/"));

                if has_source_build {
                    println!("  Found existing source build in libtorch/builds/.");
                } else if opts.non_interactive {
                    println!("  Downloading cu126 (broadest coverage).");
                    let cuda_opts = download::DownloadOpts {
                        variant: download::Variant::Cuda126,
                        ..Default::default()
                    };
                    download::run_with_context(cuda_opts, &ctx)?;
                } else {
                    let choice = prompt::ask_choice(
                        "  Choice",
                        &[
                            "Build libtorch from source (2-6 hours, covers all GPUs)",
                            "Download cu128 (Volta+ only, your older GPU won't work)",
                            "Download cu126 (pre-Volta only, your newer GPU won't work)",
                            "Skip for now",
                        ],
                        4,
                    );

                    match choice {
                        1 => {
                            println!();
                            println!("  Starting libtorch source build...");
                            println!("  This will take 2-6 hours. You can safely Ctrl-C and");
                            println!("  resume later with: fdl libtorch build");
                            println!();
                            build::run(build::BuildOpts::default())?;
                        }
                        2 => {
                            println!("  Downloading cu128...");
                            let cuda_opts = download::DownloadOpts {
                                variant: download::Variant::Cuda128,
                                ..Default::default()
                            };
                            download::run_with_context(cuda_opts, &ctx)?;
                        }
                        3 => {
                            println!("  Downloading cu126...");
                            let cuda_opts = download::DownloadOpts {
                                variant: download::Variant::Cuda126,
                                ..Default::default()
                            };
                            download::run_with_context(cuda_opts, &ctx)?;
                        }
                        _ => {
                            println!("  Skipping CUDA libtorch. You can download later with:");
                            println!("    fdl libtorch download --cuda 12.8");
                            println!("    # or build from source:");
                            println!("    fdl libtorch build");
                        }
                    }
                }
            } else if lo_major < 7 {
                println!();
                println!("  Downloading CUDA libtorch (cu126 for your pre-Volta GPU)...");
                let cuda_opts = download::DownloadOpts {
                    variant: download::Variant::Cuda126,
                    ..Default::default()
                };
                download::run_with_context(cuda_opts, &ctx)?;
            } else {
                println!();
                println!("  Downloading CUDA libtorch (cu128 for your Volta+ GPU)...");
                let cuda_opts = download::DownloadOpts {
                    variant: download::Variant::Cuda128,
                    ..Default::default()
                };
                download::run_with_context(cuda_opts, &ctx)?;
            }
        }
    }

    // ---- Step 3: Build environment (project-only) ----

    if !ctx.is_project {
        // Skip Docker image building when running standalone
        println!();
        println!("  Setup complete!");
        println!("  ===============");
        println!();
        if let Some(info) = detect::read_active(root) {
            let cuda_str = if info.cuda_version.as_deref() != Some("none") { "CUDA" } else { "CPU" };
            println!("  libtorch:  {} ({})", info.path, cuda_str);
            println!("  Location:  {}", ctx.libtorch_dir().display());
        }
        println!();
        println!("  Next steps:");
        println!("    fdl init my-project  # scaffold a new project");
        println!("    fdl diagnose         # verify GPU compatibility");
        println!();
        return Ok(());
    }

    println!();
    println!("  Step 3: Build environment");
    println!("  -------------------------");
    println!();
    println!("  floDl compiles Rust code that links against libtorch.");
    println!("  You can build with Docker (isolated, reproducible) or");
    println!("  natively (faster iteration, requires Rust + C++ toolchain).");
    println!();

    let build_mode = if has_docker && has_cargo {
        if opts.non_interactive {
            "docker"
        } else {
            let choice = prompt::ask_choice(
                "  Choice",
                &[
                    "Docker (recommended) -- isolated, reproducible builds",
                    "Native -- faster iteration, requires C++ compiler on host",
                    "Both -- set up Docker and show native instructions",
                ],
                1,
            );
            match choice {
                1 => "docker",
                2 => "native",
                3 => "both",
                _ => "docker",
            }
        }
    } else if has_docker {
        if opts.non_interactive {
            "docker"
        } else {
            println!("  Docker is available. Rust is not installed on this machine.");
            println!("  Docker is the easiest way to get started (no Rust install needed).");
            println!();
            if prompt::ask_yn("  Set up Docker build environment?", true) {
                "docker"
            } else {
                // User declined Docker but has no Rust either. Show the
                // Rust install pointers and offer one chance to flip back
                // to Docker before settling on a "none" build mode.
                println!();
                println!("  No worries. To build flodl natively you need Rust on the host:");
                println!();
                println!("    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh");
                println!();
                println!("  More: https://www.rust-lang.org/tools/install");
                println!("  Then re-run `fdl setup` and the native path will be picked up.");
                println!();
                if prompt::ask_yn("  Or use Docker after all?", false) {
                    "docker"
                } else {
                    "none"
                }
            }
        }
    } else {
        println!("  Rust is available. Docker is not installed.");
        println!("  You can build natively (requires C++ compiler on the host).");
        println!();
        "native"
    };

    // Build Docker images
    if build_mode == "docker" || build_mode == "both" {
        println!();
        println!("  Building Docker images...");

        // Create cargo cache dirs
        let _ = std::fs::create_dir_all(".cargo-cache");
        let _ = std::fs::create_dir_all(".cargo-git");

        let status = docker::compose_run(".", &["build", "dev"])?;
        if !status.success() {
            println!("  Warning: CPU Docker image build failed.");
        }

        // CUDA image if we have GPUs and CUDA libtorch
        let has_cuda_lt = detect::read_active(root)
            .is_some_and(|i| i.cuda_version.as_deref() != Some("none"));

        if !gpus.is_empty() && has_cuda_lt {
            let _ = std::fs::create_dir_all(".cargo-cache-cuda");
            let _ = std::fs::create_dir_all(".cargo-git-cuda");

            let status = docker::compose_run(".", &["build", "cuda"])?;
            if !status.success() {
                println!("  Warning: CUDA Docker image build failed.");
            }
        }

        println!("  Docker images ready.");
    }

    // ---- Summary ----

    println!();
    println!("  Setup complete!");
    println!("  ===============");
    println!();

    // Show active libtorch
    if let Some(info) = detect::read_active(root) {
        let cuda_str = if info.cuda_version.as_deref() != Some("none") {
            "CUDA"
        } else {
            "CPU"
        };
        println!("  libtorch:  {} ({})", info.path, cuda_str);
    }

    // Docker instructions
    if build_mode == "docker" || build_mode == "both" {
        println!();
        println!("  Build with Docker:");
        let has_cuda_lt = detect::read_active(root)
            .is_some_and(|i| i.cuda_version.as_deref() != Some("none"));
        if !gpus.is_empty() && has_cuda_lt {
            println!("    fdl gpu-test        # run GPU tests");
            println!("    fdl gpu-build       # compile with CUDA");
            println!("    fdl gpu-shell       # interactive shell");
        } else {
            println!("    fdl test             # run tests");
            println!("    fdl build            # compile");
            println!("    fdl shell            # interactive shell");
        }
    }

    // Native instructions
    if build_mode == "native" || build_mode == "both" {
        if let Some(info) = detect::read_active(root) {
            let lt_path = format!("libtorch/{}", info.path);
            println!();
            println!("  Build natively:");
            println!("    export LIBTORCH_PATH=\"{}\"", lt_path);
            println!(
                "    export LD_LIBRARY_PATH=\"$LIBTORCH_PATH/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\""
            );
            let has_cuda_lt = info.cuda_version.as_deref() != Some("none");
            if !gpus.is_empty() && has_cuda_lt {
                println!("    cargo test --features cuda");
            } else {
                println!("    cargo test");
            }
        }
    }

    // No-build-environment fallback: only reachable from the
    // docker-only-no-cargo branch where the user declined Docker
    // twice. The Rust install pointers were already printed during
    // Step 3; the summary just re-anchors the next move so the
    // user doesn't drop into the trailing "Other commands" block
    // without context.
    if build_mode == "none" {
        println!();
        println!("  No build environment configured.");
        println!("  Install Rust (link above) for native builds, or re-run `fdl setup`");
        println!("  and pick Docker. libtorch is already in place either way.");
    }

    println!();
    println!("  Other commands:");
    println!("    fdl diagnose         # verify GPU compatibility");
    println!("    fdl init my-project  # scaffold a new project");
    println!();

    if !opts.non_interactive {
        crate::util::install_prompt::offer_global_install();
    }

    Ok(())
}
