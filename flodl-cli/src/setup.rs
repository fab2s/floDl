//! `fdl setup` -- interactive guided setup wizard.
//!
//! Detects hardware, downloads libtorch, optionally builds Docker images.

use crate::context::Context;
use crate::libtorch::{build, detect, download};
use crate::util::{docker, prompt, requirements, system};

/// The CPU variant's pointer value, as `download` installs it.

#[derive(Default)]
pub struct SetupOpts {
    /// Skip all prompts, use auto-detected defaults.
    pub non_interactive: bool,
    /// Re-download/rebuild even if libtorch exists.
    pub force: bool,
}

/// Which libtorch a macOS host in a Docker-mounted project should get.
///
/// The libtorch is bind-mounted into a Linux container, so the host's
/// Mach-O build cannot load there. What to fetch instead depends on the
/// host arch, and only one of the two cases has an answer upstream.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum MacDockerPlan {
    /// Not macOS, or not a Docker-mounted project: fetch for the host.
    HostBuild,
    /// macOS with a Docker-mounted project: fetch the container's build,
    /// Linux at the HOST's architecture. linux/amd64 comes from the
    /// libtorch zip on an Intel Mac; linux/arm64 is repackaged from the
    /// torch wheel on Apple Silicon (the downloader owns both). The
    /// suffixed CPU dir names keep it beside a host build rather than
    /// over one.
    ForceLinux,
}

/// Pure so both macOS arms are checkable from any host: the branch is
/// unreachable on the machine most of this is developed on, and picking
/// the wrong one installs a libtorch that cannot load in the container.
fn macos_docker_plan(os: &str, _arch: &str, docker_project: bool) -> MacDockerPlan {
    if os != "macos" || !docker_project {
        return MacDockerPlan::HostBuild;
    }
    MacDockerPlan::ForceLinux
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

    let survey = flodl_hw::survey();
    let gpus = &survey.devices;
    if !gpus.is_empty() {
        println!();
        println!("  GPUs:");
        for g in gpus {
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
        // The sweep's findings, not just its device list: an AMD card
        // with no ROCm runtime is a common first-contact state, and
        // "CPU-only" with the explanation discarded sends the operator
        // away thinking the box has nothing — setup is the entry point,
        // so it says what probe would say.
        for note in survey.notes.iter().filter(|n| n.kind.explains_absence()) {
            println!("          {}", note.message);
        }
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
            println!(
                "  Note: building natively would also need: {}",
                tools.join(", ")
            );
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
        println!(
            "  libtorch will be installed to: {}",
            ctx.libtorch_dir().display()
        );
        println!();
    }

    let existing = detect::read_active(root);
    let mut skip_download = false;

    if !opts.force
        && let Some(ref info) = existing
    {
        // The variant PATH carries the vendor, not `.arch`'s `cuda=`
        // field: a ROCm build has no CUDA toolkit version and writes
        // `cuda=none` there, exactly like a CPU build. Reading that
        // field as "is this a GPU install" labelled every existing
        // ROCm install CPU-only and re-downloaded over it.
        match detect::variant_vendor(&info.path) {
            Some(vendor) => {
                println!("  Found existing {vendor} libtorch: {}", info.path);
                if opts.non_interactive {
                    println!("  Keeping existing installation.");
                    skip_download = true;
                } else if !prompt::ask_yn("  Download fresh?", false) {
                    skip_download = true;
                }
                println!();
            }
            None => println!("  Found existing CPU libtorch."),
        }
    }

    if !skip_download {
        // Always download CPU variant (useful as fallback).
        let mounted_docker_project = ctx.is_project && ctx.root.join("Dockerfile").exists();
        let plan = macos_docker_plan(
            std::env::consts::OS,
            std::env::consts::ARCH,
            mounted_docker_project,
        );
        let force_linux = plan == MacDockerPlan::ForceLinux;
        if force_linux {
            println!("  macOS + Docker-mounted project: fetching Linux libtorch");
            println!("  for the container (the host's Mach-O build cannot load there).");
        }
        println!("  Downloading CPU libtorch...");
        let cpu_opts = download::DownloadOpts {
            variant: download::Variant::Cpu,
            activate: false, // don't activate CPU if we'll also get CUDA
            force_linux,
            ..Default::default()
        };
        // The id the download resolved to: the CPU dir name is
        // per-platform (`cpu`, `cpu-aarch64`, `cpu-macos`), so the
        // fallback activation below must use what was actually installed
        // rather than re-deriving a name.
        let cpu_variant_id = download::run_with_context(cpu_opts, &ctx)?;

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
            let covered = download::rocm_covered(gpus);
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
                println!(
                    "  AMD GPU(s) detected ({}) outside the ROCm 7.0",
                    names.join(", ")
                );
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

        // The CPU download above deliberately does not activate, so a
        // GPU variant fetched after it wins the pointer. When no GPU
        // variant follows -- a CPU-only box, or an AMD card outside the
        // ROCm build's gfx list -- nothing ever writes `.active` and
        // setup finishes with libtorch on disk that `fdl diagnose` then
        // reports as "no active variant". Claim the pointer for CPU
        // only if it is still unclaimed, so this can never demote a GPU
        // variant.
        if detect::read_active(root).is_none() && detect::is_valid_variant(root, &cpu_variant_id) {
            detect::set_active(root, &cpu_variant_id)?;
        }
    }

    // The active variant, resolved ONCE for every consumer below. Both
    // the vendor and the warning `variant_vendor` emits on an
    // unrecognised basename belong to the variant, not to each question
    // asked about it -- re-deriving per call-site printed the warning
    // four times.
    let active = detect::read_active(root);
    let active_vendor = active
        .as_ref()
        .and_then(|info| detect::variant_vendor(&info.path));
    let active_label = |v: Option<system::GpuVendor>| match v {
        Some(vendor) => vendor.to_string(),
        None => "CPU".to_string(),
    };

    // ---- Step 3: Build environment (project-only) ----

    if !ctx.is_project {
        // Skip Docker image building when running standalone
        println!();
        println!("  Setup complete!");
        println!("  ===============");
        println!();
        if let Some(info) = &active {
            println!(
                "  libtorch:  {} ({})",
                info.path,
                active_label(active_vendor)
            );
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

        // GPU image, when there is hardware AND a GPU libtorch to link
        // against. The compose service is SELECTED from the variant's
        // vendor rather than hardcoded: a CUDA image and a ROCm image are
        // genuinely different artifacts (different base, different device
        // nodes), so building `cuda` on an AMD box builds the wrong one.
        if let Some(vendor) = active_vendor.filter(|_| !gpus.is_empty()) {
            let service = crate::run::resolve_docker_service(crate::run::LOGICAL_GPU_SERVICE, root);
            let _ = std::fs::create_dir_all(format!(".cargo-cache-{service}"));
            let _ = std::fs::create_dir_all(format!(".cargo-git-{service}"));

            let status = docker::compose_run(".", &["build", &service])?;
            if !status.success() {
                println!("  Warning: {vendor} Docker image build failed.");
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
    if let Some(info) = &active {
        println!(
            "  libtorch:  {} ({})",
            info.path,
            active_label(active_vendor)
        );
    }

    let gpu_ready = !gpus.is_empty() && active_vendor.is_some();

    // Docker instructions
    if build_mode == "docker" || build_mode == "both" {
        println!();
        println!("  Build with Docker:");
        if gpu_ready {
            println!("    fdl gpu-test        # run GPU tests");
            println!("    fdl gpu-build       # compile for the GPU");
            println!("    fdl gpu-shell       # interactive shell");
        } else {
            println!("    fdl test             # run tests");
            println!("    fdl build            # compile");
            println!("    fdl shell            # interactive shell");
        }
    }

    // Native instructions
    if (build_mode == "native" || build_mode == "both")
        && let Some(info) = &active
    {
        let lt_path = format!("libtorch/{}", info.path);
        println!();
        println!("  Build natively:");
        println!("    export LIBTORCH_PATH=\"{}\"", lt_path);
        for line in detect::ld_library_path_lines(active_vendor, "$LIBTORCH_PATH/lib") {
            println!("    {line}");
        }
        match active_vendor.filter(|_| gpu_ready) {
            Some(vendor) => println!("    cargo test --features {}", vendor.cargo_feature()),
            None => println!("    cargo test"),
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

#[cfg(test)]
mod tests {
    use super::*;

    // The macOS arms never execute on the Linux dev box or on the Linux
    // CI legs, and the Apple Silicon one is the case where a wrong answer
    // installs a libtorch that cannot load inside the container.

    #[test]
    fn a_mac_docker_project_forces_the_linux_build_on_both_arches() {
        // The download keeps the host ARCH under a forced Linux OS, so
        // Apple Silicon gets the linux/arm64 wheel repackage and an
        // Intel Mac the linux/amd64 zip -- the container's build either
        // way. The Apple Silicon arm used to fetch the HOST build and
        // point at the guide's manual wheel-extraction steps instead.
        for arch in ["aarch64", "x86_64"] {
            assert_eq!(
                macos_docker_plan("macos", arch, true),
                MacDockerPlan::ForceLinux,
                "{arch} docker"
            );
        }
    }

    #[test]
    fn a_mac_without_a_docker_project_builds_for_the_host() {
        for arch in ["aarch64", "x86_64"] {
            assert_eq!(
                macos_docker_plan("macos", arch, false),
                MacDockerPlan::HostBuild,
                "{arch} native"
            );
        }
    }

    #[test]
    fn non_macos_hosts_are_unaffected() {
        for (os, arch) in [
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("windows", "x86_64"),
        ] {
            assert_eq!(
                macos_docker_plan(os, arch, true),
                MacDockerPlan::HostBuild,
                "{os}/{arch}"
            );
        }
    }
}
