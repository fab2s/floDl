//! `fdl init <name>` -- scaffold a new floDl project.
//!
//! Three modes, selected by flag or interactive prompt:
//! - `Mounted` (default): Docker with libtorch host-mounted at runtime.
//! - `Docker` (`--docker`): Docker with libtorch baked into the image.
//! - `Native` (`--native`): no Docker; libtorch and cargo provided on the host.

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::util::prompt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Mounted,
    Docker,
    Native,
}

pub fn run(
    name: Option<&str>,
    docker: bool,
    native: bool,
    with_hf: bool,
) -> Result<(), String> {
    let name = name.ok_or("usage: fdl init <project-name>")?;
    validate_name(name)?;

    if Path::new(name).exists() {
        return Err(format!("'{}' already exists", name));
    }

    if docker && native {
        return Err("--docker and --native are mutually exclusive".into());
    }
    let flag_driven = docker || native || with_hf;
    let mode = if docker {
        Mode::Docker
    } else if native {
        Mode::Native
    } else {
        pick_mode_interactively()
    };
    // `--with-hf` bypasses the prompt entirely for scripted init.
    // Without any flag, ask after mode selection; with *any* flag set
    // the user signalled non-interactive intent, so respect `--with-hf`
    // verbatim and skip the prompt.
    let include_hf = if flag_driven {
        with_hf
    } else {
        prompt::ask_yn(
            "Include flodl-hf (HuggingFace: BERT/RoBERTa/DistilBERT, Hub loader, tokenizer)?",
            false,
        )
    };

    let crate_name = name.replace('-', "_");
    let flodl_dep = resolve_flodl_dep();

    fs::create_dir_all(format!("{}/src", name))
        .map_err(|e| format!("cannot create directory: {}", e))?;

    match mode {
        Mode::Mounted => scaffold_mounted(name, &crate_name, &flodl_dep)?,
        Mode::Docker => scaffold_docker(name, &crate_name, &flodl_dep)?,
        Mode::Native => scaffold_native(name, &crate_name, &flodl_dep)?,
    }

    // Shared across all modes.
    write_file(
        &format!("{}/src/main.rs", name),
        &main_rs_template(),
    )?;
    write_file(
        &format!("{}/.gitignore", name),
        &gitignore_template(mode),
    )?;
    write_file(
        &format!("{}/fdl.yml.example", name),
        &fdl_yml_example_template(name, mode),
    )?;
    // Native mode generates no docker-compose, so there is nothing to read a
    // `.env`. Docker modes get the template that documents the knobs their
    // compose actually substitutes.
    if mode != Mode::Native {
        write_file(
            &format!("{}/.env.example", name),
            &env_example_template(mode),
        )?;
    }
    write_fdl_bootstrap(name)?;

    if include_hf {
        let project_dir = Path::new(name);
        if let Err(e) = crate::add::add_flodl_hf_at(project_dir) {
            // Scaffolded project is still usable even if the HF sub-crate
            // failed; surface the error but don't roll back.
            eprintln!("warning: flodl-hf scaffold failed: {e}");
            eprintln!("You can retry after `cd {}` with `fdl add flodl-hf`.", name);
        }
    }

    print_next_steps(name, mode, include_hf);
    crate::util::install_prompt::offer_global_install();
    Ok(())
}

/// Ask the user interactively which mode to generate. Falls through to
/// `Mounted` when no TTY is attached (the same default as passing no flag
/// to `--non-interactive` tooling).
fn pick_mode_interactively() -> Mode {
    println!();
    if !prompt::ask_yn("Use Docker for builds?", true) {
        return Mode::Native;
    }

    // On macOS, Docker runs Linux containers under Rosetta/QEMU emulation.
    // Builds and training are substantially slower than native cargo on the
    // host. Warn once and offer a chance to drop to Native before the user
    // commits to a Docker scaffold.
    if cfg!(target_os = "macos") {
        println!();
        println!("  Heads up: on macOS, Docker runs Linux containers under emulation");
        println!("  (Rosetta / QEMU). Builds and training will be substantially slower");
        println!("  than running cargo natively on the host. Native mode keeps");
        println!("  everything on the Mac and uses macOS libtorch directly.");
        println!();
        if !prompt::ask_yn("Continue with Docker?", true) {
            return Mode::Native;
        }
    }

    // 1-based: 1 = mounted (default), 2 = baked-in.
    let choice = prompt::ask_choice(
        "How should libtorch be provided to the container?",
        &[
            "Mount it from the host (recommended: lighter image, swap CUDA variants)",
            "Bake it into the image at build time (zero host setup)",
        ],
        1,
    );
    match choice {
        2 => Mode::Docker,
        _ => Mode::Mounted,
    }
}

fn print_next_steps(name: &str, mode: Mode, include_hf: bool) {
    println!();
    println!("Project '{}' created. Next steps:", name);
    println!();
    println!("  cd {}", name);
    match mode {
        Mode::Mounted => {
            println!("  ./fdl setup   # detect hardware + download libtorch");
            println!("  ./fdl build   # build the project");
        }
        Mode::Docker => {
            println!("  ./fdl build   # first build (downloads libtorch, ~5 min)");
        }
        Mode::Native => {
            println!("  ./fdl libtorch download --cpu     # or --cuda 12.8");
            println!("  ./fdl build                       # cargo build on the host");
        }
    }
    println!("  ./fdl test    # run tests");
    println!("  ./fdl run     # train the model");
    if mode != Mode::Native {
        println!("  ./fdl shell   # interactive shell");
    }
    if include_hf {
        println!();
        println!("  cd flodl-hf && fdl classify   # try the HuggingFace playground");
    }
    println!();
    println!("`./fdl --help` lists every command defined in fdl.yml.");
    println!("Edit src/main.rs to build your model.");
    println!();
    println!("Guides:");
    println!("  Tutorials:         https://flodl.dev/guide/tensors");
    println!("  Graph Tree:        https://flodl.dev/guide/graph-tree");
    println!("  PyTorch migration: https://flodl.dev/guide/pytorch/migration");
    println!("  Troubleshooting:   https://flodl.dev/guide/troubleshooting");
}

fn write_fdl_bootstrap(name: &str) -> Result<(), String> {
    let fdl_script = include_str!("../assets/fdl");
    write_file(&format!("{}/fdl", name), fdl_script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(
            format!("{}/fdl", name),
            fs::Permissions::from_mode(0o755),
        );
    }
    Ok(())
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("project name cannot be empty".into());
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err("project name must contain only letters, digits, hyphens, underscores".into());
    }
    Ok(())
}

fn resolve_flodl_dep() -> String {
    // Try crates.io for the latest version
    if let Some(version) = crates_io_version() {
        format!("flodl = \"{}\"", version)
    } else {
        "flodl = { git = \"https://github.com/flodl-labs/flodl.git\" }".into()
    }
}

fn crates_io_version() -> Option<String> {
    let output = Command::new("curl")
        .args(["-sL", "https://crates.io/api/v1/crates/flodl"])
        .output()
        .ok()?;
    let body = String::from_utf8_lossy(&output.stdout);
    // Extract "max_stable_version":"X.Y.Z"
    let marker = "\"max_stable_version\":\"";
    let start = body.find(marker)? + marker.len();
    let end = start + body[start..].find('"')?;
    let version = &body[start..end];
    if version.is_empty() { None } else { Some(version.to_string()) }
}

// ---------------------------------------------------------------------------
// Docker scaffold (standalone, libtorch baked into images)
// ---------------------------------------------------------------------------

fn scaffold_docker(name: &str, crate_name: &str, flodl_dep: &str) -> Result<(), String> {
    write_file(
        &format!("{}/Cargo.toml", name),
        &cargo_toml_template(crate_name, flodl_dep),
    )?;
    write_file(
        &format!("{}/Dockerfile.cpu", name),
        DOCKERFILE_CPU,
    )?;
    write_file(
        &format!("{}/Dockerfile.cuda", name),
        DOCKERFILE_CUDA,
    )?;
    write_file(
        &format!("{}/docker-compose.yml", name),
        &docker_compose_template(crate_name, true),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Mounted scaffold (libtorch from host, like the main repo)
// ---------------------------------------------------------------------------

fn scaffold_mounted(name: &str, crate_name: &str, flodl_dep: &str) -> Result<(), String> {
    write_file(
        &format!("{}/Cargo.toml", name),
        &cargo_toml_template(crate_name, flodl_dep),
    )?;
    write_file(
        &format!("{}/Dockerfile", name),
        DOCKERFILE_MOUNTED,
    )?;
    write_file(
        &format!("{}/Dockerfile.cuda", name),
        DOCKERFILE_CUDA_MOUNTED,
    )?;
    write_file(
        &format!("{}/docker-compose.yml", name),
        &docker_compose_template(crate_name, false),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Native scaffold (no Docker; libtorch and cargo live on the host)
// ---------------------------------------------------------------------------

fn scaffold_native(name: &str, crate_name: &str, flodl_dep: &str) -> Result<(), String> {
    write_file(
        &format!("{}/Cargo.toml", name),
        &cargo_toml_template(crate_name, flodl_dep),
    )?;
    // Intentionally no Dockerfile*/docker-compose.yml -- the user opted out
    // of Docker. They can switch later by regenerating or adding their own.
    Ok(())
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

fn cargo_toml_template(crate_name: &str, flodl_dep: &str) -> String {
    format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2024"

[dependencies]
{flodl_dep}

# GPU support is opt-in. `fdl gpu-*` picks the right one for you through
# $FDL_GPU_FEATURE, derived from the libtorch variant you have active.
# (Without this section `cargo build --features cuda` fails outright with
# "does not contain this feature" -- cargo resolves it against THIS
# package, not the dependency.)
[features]
cuda = ["flodl/cuda"]

# Optimize floDl in dev builds -- your code stays fast to compile.
# After the first build, only your graph code recompiles (~2s).
[profile.dev.package.flodl]
opt-level = 3

[profile.dev.package.flodl-sys]
opt-level = 3

# Release: cross-crate optimization for maximum throughput.
[profile.release]
lto = "thin"
codegen-units = 1
"#
    )
}

fn main_rs_template() -> String {
    r#"//! floDl training template.
//!
//! This is a starting point for your model. Edit the architecture,
//! data loading, and training loop to fit your task.
//!
//! New to Rust? Read: https://flodl.dev/guide/pytorch/rust-primer
//! Stuck?       Read: https://flodl.dev/guide/troubleshooting

use flodl::*;
use flodl::monitor::Monitor;

fn main() -> Result<()> {
    // --- Model ---
    let model = FlowBuilder::from(Linear::new(4, 32)?)
        .through(GELU)
        .through(LayerNorm::new(32)?)
        .also(Linear::new(32, 32)?)       // residual connection
        .through(Linear::new(32, 1)?)
        .build()?;

    // --- Optimizer ---
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.001);
    let scheduler = CosineScheduler::new(0.001, 1e-6, 100);
    model.train();

    // --- Data ---
    // Replace this with your data loading.
    let opts = TensorOptions::default();
    let batches: Vec<(Tensor, Tensor)> = (0..32)
        .map(|_| {
            let x = Tensor::randn(&[16, 4], opts).unwrap();
            let y = Tensor::randn(&[16, 1], opts).unwrap();
            (x, y)
        })
        .collect();

    // --- Training loop ---
    let num_epochs = 100usize;
    let mut monitor = Monitor::new(num_epochs);
    // monitor.serve(3000)?;              // uncomment for live dashboard
    // monitor.watch(&model);             // uncomment to show graph SVG
    // monitor.save_html("report.html");  // uncomment to save HTML report

    for epoch in 0..num_epochs {
        let t = std::time::Instant::now();
        let mut epoch_loss = 0.0;

        for (input_t, target_t) in &batches {
            let input = Variable::new(input_t.clone(), true);
            let target = Variable::new(target_t.clone(), false);

            optimizer.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            loss.backward()?;
            clip_grad_norm(&params, 1.0)?;
            optimizer.step()?;

            epoch_loss += loss.item()?;
        }

        let avg_loss = epoch_loss / batches.len() as f64;
        let lr = scheduler.lr(epoch);
        optimizer.set_lr(lr);
        monitor.log(epoch, t.elapsed(), &[("loss", avg_loss), ("lr", lr)]);
    }

    monitor.finish();
    Ok(())
}
"#
    .into()
}

fn gitignore_template(mode: Mode) -> String {
    let mut s = String::from(
        "/target
*.fdl
*.log
*.csv
*.html

# Local fdl config (fdl.yml.example is committed; fdl copies it on first run)
fdl.yml
fdl.yaml

# Local docker-compose env (per-machine: UID/GID, libtorch variant override,
# cargo job throttle)
.env
",
    );
    match mode {
        Mode::Docker => {
            // libtorch is baked into the image, nothing on host to ignore.
            s.push_str(
                ".cargo-cache/
.cargo-git/
.cargo-cache-cuda/
.cargo-git-cuda/
",
            );
        }
        Mode::Mounted => {
            // Mounted libtorch + separate cargo caches per docker service.
            s.push_str(
                ".cargo-cache/
.cargo-git/
.cargo-cache-cuda/
.cargo-git-cuda/
libtorch/
",
            );
        }
        Mode::Native => {
            // No docker, no container caches. libtorch/ is still ignored
            // because `./fdl libtorch download` installs it locally.
            s.push_str("libtorch/\n");
        }
    }
    s
}

fn docker_compose_template(crate_name: &str, baked: bool) -> String {
    if baked {
        format!(
            r#"services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile.cpu
    image: {crate_name}-dev
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache:/usr/local/cargo/registry
      - ./.cargo-git:/usr/local/cargo/git
    working_dir: /workspace
    stdin_open: true
    tty: true
    environment:
      # Throttle cargo's link parallelism. Unset on Linux native (empty →
      # cargo's default); Mac hosts set it in `.env` to keep `ld` within what
      # a virtiofs-backed workspace can serve. See docs/mac-apple-silicon.md.
      - CARGO_BUILD_JOBS
      # flodl runtime knobs, forwarded from the host (or `.env`):
      # verbosity is what `fdl -v/-vv/...` sets per invocation, and the
      # timeout scale stretches distributed network deadlines on slow links.
      - FLODL_VERBOSITY
      - FLODL_NET_TIMEOUT_SCALE

  cuda:
    build:
      context: .
      dockerfile: Dockerfile.cuda
    image: {crate_name}-cuda
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache-cuda:/usr/local/cargo/registry
      - ./.cargo-git-cuda:/usr/local/cargo/git
    working_dir: /workspace
    stdin_open: true
    tty: true
    environment:
      # flodl runtime knobs, forwarded from the host (or `.env`):
      # verbosity is what `fdl -v/-vv/...` sets per invocation, and the
      # timeout scale stretches distributed network deadlines on slow links.
      - FLODL_VERBOSITY
      - FLODL_NET_TIMEOUT_SCALE
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"#
        )
    } else {
        format!(
            r#"services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    image: {crate_name}-dev
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache:/usr/local/cargo/registry
      - ./.cargo-git:/usr/local/cargo/git
      - ${{LIBTORCH_CPU_PATH:-./libtorch/precompiled/cpu}}:/usr/local/libtorch:ro
    working_dir: /workspace
    stdin_open: true
    tty: true
    environment:
      # Throttle cargo's link parallelism. Required on macOS Docker / OrbStack,
      # where the virtiofs-mounted libtorch directory cannot serve many
      # concurrent `ld` lookups — the linker reports `cannot find -ltorch`
      # spuriously. Unset on Linux native (empty → cargo's default).
      - CARGO_BUILD_JOBS
      # flodl runtime knobs, forwarded from the host (or `.env`):
      # verbosity is what `fdl -v/-vv/...` sets per invocation, and the
      # timeout scale stretches distributed network deadlines on slow links.
      - FLODL_VERBOSITY
      - FLODL_NET_TIMEOUT_SCALE

  cuda:
    build:
      context: .
      dockerfile: Dockerfile.cuda
      args:
        CUDA_VERSION: ${{CUDA_VERSION:-12.8.0}}
    image: {crate_name}-cuda:${{CUDA_TAG:-12.8}}
    user: "${{UID:-1000}}:${{GID:-1000}}"
    volumes:
      - .:/workspace
      - ./.cargo-cache-cuda:/usr/local/cargo/registry
      - ./.cargo-git-cuda:/usr/local/cargo/git
      - ${{LIBTORCH_HOST_PATH:-./libtorch/precompiled/cu128}}:/usr/local/libtorch:ro
    working_dir: /workspace
    stdin_open: true
    tty: true
    environment:
      # flodl runtime knobs, forwarded from the host (or `.env`):
      # verbosity is what `fdl -v/-vv/...` sets per invocation, and the
      # timeout scale stretches distributed network deadlines on slow links.
      - FLODL_VERBOSITY
      - FLODL_NET_TIMEOUT_SCALE
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"#
        )
    }
}

// ---------------------------------------------------------------------------
// Dockerfile templates
// ---------------------------------------------------------------------------

// Docker mode: libtorch baked into images
const DOCKERFILE_CPU: &str = r#"# CPU-only dev image for floDl projects.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

# libtorch (CPU-only, ~200MB)
ARG LIBTORCH_VERSION=2.10.0
RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip \
    && unzip -q libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip -d /usr/local \
    && rm libtorch-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib"

WORKDIR /workspace
"#;

const DOCKERFILE_CUDA: &str = r#"# CUDA dev image for floDl projects.
# Requires: docker run --gpus all ...
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

# libtorch (CUDA 12.8)
ARG LIBTORCH_VERSION=2.10.0
RUN wget -q "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip" \
    && unzip -q "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu128.zip" -d /usr/local \
    && rm "libtorch-shared-with-deps-${LIBTORCH_VERSION}+cu128.zip"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

WORKDIR /workspace
"#;

// Mounted mode: libtorch provided at runtime via volume mount
const DOCKERFILE_MOUNTED: &str = r#"# CPU dev image for floDl projects (libtorch mounted at runtime).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib"

WORKDIR /workspace
"#;

const DOCKERFILE_CUDA_MOUNTED: &str = r#"# CUDA dev image for floDl projects (libtorch mounted at runtime).
# Requires: docker run --gpus all ...
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl unzip ca-certificates git gcc g++ pkg-config graphviz \
    && rm -rf /var/lib/apt/lists/*

# Rust
ENV CARGO_HOME="/usr/local/cargo"
ENV RUSTUP_HOME="/usr/local/rustup"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && chmod -R a+rwx "$CARGO_HOME" "$RUSTUP_HOME"
ENV PATH="${CARGO_HOME}/bin:${PATH}"

ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

WORKDIR /workspace
"#;

// ---------------------------------------------------------------------------
// fdl.yml.example template
// ---------------------------------------------------------------------------

/// The scaffold ships `fdl.yml.example` (committed) and fdl auto-copies it to
/// the gitignored `fdl.yml` on first use. Docker modes attach `docker:` to
/// every command; native mode drops `docker:` so the commands run directly
/// on the host. Libtorch env vars (`LIBTORCH_HOST_PATH`, `CUDA_VERSION`,
/// `CUDA_TAG`, etc.) are derived from `libtorch/.active` by
/// `flodl-cli/src/run.rs::libtorch_env` before each `docker compose run`
/// (Docker modes) or exported into the child process (native mode).
/// `.env.example` for a scaffolded project: the compose knobs that this
/// mode's generated `docker-compose.yml` actually substitutes, and no others.
/// Committed template, gitignored working copy — the same convention as
/// `fdl.yml.example`.
fn env_example_template(mode: Mode) -> String {
    let mut s = String::from(
        "# Local Docker environment overrides for docker-compose.yml.
# Copy this to `.env` (gitignored) and uncomment what you need:
#   cp .env.example .env
# docker-compose auto-reads `.env` from this directory. This `.env.example`
# is only a template; compose never reads it directly.

# Host user/group mapping, so files created in the container are owned by you
# rather than root. Defaults to 1000:1000 when unset; macOS is usually 501:20
# (`id -u` / `id -g`).
#UID=1000
#GID=1000
",
    );
    if mode == Mode::Mounted {
        s.push_str(
            "
# libtorch mount points (host paths). Defaults live in docker-compose.yml.
# Override to point at a different variant, e.g. an extracted linux-aarch64
# build on Apple Silicon.
#LIBTORCH_CPU_PATH=./libtorch/precompiled/cpu
#LIBTORCH_HOST_PATH=./libtorch/precompiled/cu128

# CUDA base image version and image tag for the `cuda` service.
# Only affects direct `docker compose` calls: `fdl` derives both from the active
# libtorch variant's `.arch` metadata and overrides whatever is set here.
#CUDA_VERSION=12.8.0
#CUDA_TAG=12.8
",
        );
    }
    s.push_str(
        "
# Throttle cargo build/link parallelism. Leave unset on Linux (uses all cores).
# On Apple Silicon via Docker/OrbStack, set to 2 to avoid spurious
# \"cannot find -ltorch\" linker errors caused by virtiofs mount latency.
#CARGO_BUILD_JOBS=2

# flodl log verbosity. `fdl -v/-vv/...` sets this per invocation; setting it
# here makes a level stick without the flag.
#FLODL_VERBOSITY=1

# Scale every distributed network timeout (socket setup, coordinator deadlines).
# Raise it on slow or congested links where the 30s LAN defaults are too tight.
#FLODL_NET_TIMEOUT_SCALE=2
",
    );
    s
}

fn fdl_yml_example_template(project_name: &str, mode: Mode) -> String {
    let use_docker = matches!(mode, Mode::Mounted | Mode::Docker);
    // `gpu` is fdl's logical service: it resolves to the container
    // matching the active libtorch variant (`cuda` / `rocm`), so a
    // scaffolded project does not hardcode a vendor either.
    let (cpu_svc, cuda_svc) = if use_docker {
        ("\n    docker: dev", "\n    docker: gpu")
    } else {
        ("", "")
    };
    let cuda_note = if use_docker {
        "(requires NVIDIA Container Toolkit)"
    } else {
        "(requires a matching CUDA toolkit on the host)"
    };
    let preamble = if use_docker {
        "# Run any of these with `./fdl <cmd>` (or `fdl <cmd>` once installed\n\
         # globally via `./fdl install`). Libtorch env vars are derived from\n\
         # `libtorch/.active` automatically; missing libtorch surfaces as a\n\
         # clean linker error, with `./fdl setup` one call away."
    } else {
        "# Native mode: commands run on the host. Make sure libtorch is\n\
         # installed (`./fdl libtorch download --cpu` or `--cuda 12.8`)\n\
         # and that `$LIBTORCH` / `$LD_LIBRARY_PATH` are exported so\n\
         # cargo can link. `./fdl libtorch info` prints the commands you\n\
         # need after a download."
    };

    let shell_block = if use_docker {
        format!(
            r#"  shell:
    description: Interactive shell (CPU container)
    run: bash{cpu_svc}

"#
        )
    } else {
        // Native mode: no container to drop into; users open their own shell.
        String::new()
    };

    let cuda_shell_block = if use_docker {
        format!(
            r#"  cuda-shell:
    description: Interactive shell (CUDA container)
    run: bash{cuda_svc}
"#
        )
    } else {
        String::new()
    };

    format!(
        r#"description: {project_name}

{preamble}

commands:
  # --- CPU ---
  build:
    description: Build (debug)
    run: cargo build{cpu_svc}
  test:
    description: Run CPU tests
    run: cargo test -- --nocapture{cpu_svc}
  run:
    description: cargo run
    run: cargo run{cpu_svc}
  check:
    description: Type-check without building
    run: cargo check{cpu_svc}
  clippy:
    description: Lint
    run: cargo clippy -- -W clippy::all{cpu_svc}
{shell_block}  # --- GPU {cuda_note} ---
  # $FDL_GPU_FEATURE is exported by fdl from the active libtorch variant,
  # so these stay correct if you switch variants.
  gpu-build:
    description: Build with GPU support
    run: cargo build --features $FDL_GPU_FEATURE{cuda_svc}
  gpu-test:
    description: Run GPU tests
    run: cargo test --features $FDL_GPU_FEATURE -- --nocapture{cuda_svc}
  gpu-run:
    description: cargo run with GPU support
    run: cargo run --features $FDL_GPU_FEATURE{cuda_svc}
{cuda_shell_block}"#
    )
}

// ---------------------------------------------------------------------------
// File writing helper
// ---------------------------------------------------------------------------

fn write_file(path: &str, content: &str) -> Result<(), String> {
    fs::write(path, content).map_err(|e| format!("cannot write {}: {}", path, e))
}

#[cfg(test)]
mod tests {
    use super::validate_name;

    #[test]
    fn validate_name_accepts_alnum_hyphen_underscore() {
        for ok in ["my_project", "my-project", "Proj123", "a", "x_1-2"] {
            assert!(validate_name(ok).is_ok(), "{ok:?} should be valid");
        }
    }

    #[test]
    fn validate_name_rejects_empty() {
        let err = validate_name("").unwrap_err();
        assert!(err.contains("empty"), "unexpected: {err}");
    }

    #[test]
    fn validate_name_rejects_disallowed_chars() {
        // Spaces, dots, and path separators are the realistic footguns
        // (a project name becomes a directory + a crate name).
        for bad in ["my project", "my.project", "a/b", "../evil", "name!"] {
            let err = validate_name(bad).unwrap_err();
            assert!(err.contains("only letters"), "{bad:?} -> unexpected: {err}");
        }
    }
}
