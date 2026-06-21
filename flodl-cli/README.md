# flodl-cli

`fdl` is the command-line tool for the [floDl](https://flodl.dev) Rust
deep-learning framework. See [flodl.dev](https://flodl.dev) and the
[main README](https://github.com/flodl-labs/flodl#readme) for the full
framework.

`fdl` drives first-time setup, libtorch management, project scaffolding,
hardware diagnostics, shell completions, and the declarative project
manifest (`fdl.yml`) used to dispatch training jobs, DDP runs, and
tooling inside a flodl workspace.

It is a pure-Rust binary with **zero native dependencies**. No libtorch, no
Python, and no Rust toolchain required to install. The CLI stays useful
even before anything else is set up, which is the whole point.

```sh
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl install                  # copy to ~/.local/bin/fdl (or `cargo install flodl-cli`)
fdl libtorch download          # auto-detect GPU, install the right libtorch variant
fdl diagnose                   # hardware + compatibility report
```

That alone gives you a working libtorch installation usable from any
Rust (tch-rs), C++, or Python project. The flodl framework
([flodl.dev](https://flodl.dev)) builds on top with `fdl setup`,
`fdl init`, and the `fdl.yml` manifest format covered in the
[full CLI reference](https://github.com/flodl-labs/flodl/blob/main/docs/cli.md).

## Design principles

- **Pure Rust, zero external crate deps beyond serde.** ~750KB static
  binary. Compiles in under a second. No C++, no libtorch linking.
- **Cross-platform.** Linux x86_64/aarch64, macOS arm64, Windows x86_64.
- **GPU-aware by default.** Probes `nvidia-smi`, maps CUDA architectures to
  the right pre-built variant, warns when no single pre-built variant covers
  your hardware.
- **Multi-variant aware.** Install CPU, cu126, cu128, and custom source
  builds side-by-side. Switch the active variant in one command; everything
  reading `.active` follows automatically.
- **Graceful degradation.** Works fine without a GPU, without Docker, without
  an internet connection once variants are cached.
- **Project-aware but not project-locked.** Inside a flodl project, `fdl`
  finds the project root and uses `./libtorch/`. Standalone, it uses
  `~/.flodl/libtorch/` (override with `$FLODL_HOME`).

## What `fdl` provides

### Before you have a project

- **`fdl setup`** - guided wizard: detect hardware, pick the right libtorch
  variant, configure Docker or native builds, optionally build images.
- **`fdl init <name>`** - scaffold a new flodl project with an annotated
  training template, Dockerfile, `fdl.yml`, and a `.gitignore`. The
  generated project uses the mounted-libtorch pattern by default; add
  `--docker` to bake libtorch into the image instead.
- **`fdl libtorch …`** - install, list, switch, remove, and source-build
  libtorch variants (covered in detail below).
- **`fdl diagnose`** - hardware and compatibility report.

### In any project that has a `fdl.yml`

The `fdl.yml` manifest turns `fdl` into a project task runner. It applies
equally to the flodl source checkout and to anything you scaffolded with
`fdl init`, as long as a `fdl.yml` sits at the project root (or above).

- **Manifest-driven commands.** `fdl.yml` declares a `commands:` map; each
  entry is either a `run:` shell script (optionally wrapped in
  `docker compose run --rm <service>` via `docker:`), a `path:` pointer to
  a nested sub-project with its own `fdl.yml`, or a preset that merges
  structured config over an enclosing `entry:`. Replaces the old
  `make`/`docker compose` workflow.
- **Environment overlays.** `fdl @ci test` loads `fdl.ci.yml` on top
  of the base config; `fdl --env ci test` and `FDL_ENV=ci` are equivalent.
  `fdl config show [env]` prints the resolved merged config with per-layer
  annotations.
- **Shell completions.** `fdl completions bash|zsh|fish` emits a
  project-aware completion script covering every built-in plus every
  command declared in the current `fdl.yml`. Flag completion is
  value-aware (`--cuda <TAB>` offers `12.6 12.8` from the `choices:`
  declaration).
- **Schema introspection for your binaries.** Any binary built with
  `#[derive(flodl_cli::FdlArgs)]` responds to `--fdl-schema` with a JSON
  description of its flags; `fdl schema list/clear/refresh` manages the
  cache that powers project-aware help and completion. The derive lives
  in the [`flodl-cli-macros`](https://crates.io/crates/flodl-cli-macros)
  crate (re-exported here) -- see its
  [README](https://github.com/flodl-labs/flodl/blob/main/flodl-cli-macros/README.md)
  or the [CLI reference](https://github.com/flodl-labs/flodl/blob/main/docs/cli.md#declaring-flags-in-rust)
  for the full attribute surface and a worked example.

### For multi-GPU and cluster training

`fdl` carries first-class commands for multi-host clusters:

- **`fdl probe`** — readiness audit (single-host) / **`fdl cluster
  probe`** — SSH-aggregated cluster audit. Surfaces GPU/libtorch arch
  mismatches, NCCL version skew across hosts, missing
  `nccl_socket_ifname:`, legacy schema keys in `fdl.cluster.yml`,
  shared-data mount divergence, and dashboard port collisions. Returns
  non-zero on errors — use as a CI gate before launching multi-hour
  jobs.
- **`fdl nccl build`** — compile `libnccl` from source for the local
  GPU archs + auto-detected target NCCL version, for the LD_PRELOAD
  bridge pattern that resolves NCCL major.minor skew on heterogeneous
  rigs.
- **`fdl --gpus <spec>`** — global GPU scope override. On
  cluster-aware commands, synthesizes a single-host cluster envelope;
  on non-cluster commands, sets `CUDA_VISIBLE_DEVICES`.
- **`fdl @cluster <cmd>`** — `@` selector for the `cluster` env
  overlay. Auto-builds the target binary per-host (resolving per-host
  libtorch variants from `fdl.cluster.yml`'s `arch:` declarations),
  SSHes each worker, fans out, supervises rank children, tears them
  down on parent exit.

See the [DDP Reference](https://github.com/flodl-labs/flodl/blob/main/docs/ddp.md)
for the topology side and the
[CLI reference](https://github.com/flodl-labs/flodl/blob/main/docs/cli.md)
for the full command surface.

### In the flodl source checkout specifically

The flodl repo ships its own `fdl.yml` with commands like `fdl test`,
`fdl cuda-test`, `fdl clippy`, `fdl shell`, and the `fdl ddp-bench`
sub-project for DDP validation runs. These are concrete examples of the
manifest format, not built into the CLI. Your scaffolded project starts
with a minimal `fdl.yml` and grows its own command set as needed.

## Also useful standalone: libtorch for any PyTorch / tch-rs project

The libtorch-management and diagnostics commands are independent of flodl
and handle a gap PyTorch itself never filled: a proper installer. `fdl`
works as a drop-in libtorch manager for:

- **tch-rs projects** - download the right libtorch, point `LIBTORCH` at it,
  build. No more hand-fetching URLs from the PyTorch get-started page.
- **PyTorch C++ development** - juggle CPU, CUDA 12.6, CUDA 12.8, and
  source-built variants on the same host without symlink choreography.
- **Mixed-GPU systems** - when no single pre-built variant covers your
  architectures (e.g. GTX 1060 sm_61 + RTX 5060 Ti sm_120), `fdl libtorch
  build` compiles PyTorch from source with the exact archs you need.
  Docker-isolated by default, native toolchain supported.
- **CI pipelines** - `fdl diagnose --json` emits a machine-readable
  hardware and compatibility report to gate jobs on GPU presence or
  libtorch version.

### Standalone example with tch-rs

```sh
export FLODL_HOME=~/.libtorch-variants
fdl libtorch download --cuda 12.8
fdl libtorch list
#   precompiled/cu128 (active)

export LIBTORCH=$FLODL_HOME/libtorch/precompiled/cu128
cargo add tch
cargo build
```

### Standalone example with PyTorch C++ (libtorch)

```sh
export FLODL_HOME=~/.libtorch-variants
fdl libtorch download --cuda 12.8
fdl libtorch list
#   precompiled/cu128 (active)

# Point CMake at the active variant
cmake -B build -DCMAKE_PREFIX_PATH=$FLODL_HOME/libtorch/precompiled/cu128
cmake --build build
```

### Switching variants for different projects

```sh
fdl libtorch download --cpu
fdl libtorch download --cuda 12.6     # legacy CUDA projects
fdl libtorch download --cuda 12.8     # latest

fdl libtorch activate precompiled/cu126   # work on legacy code
fdl libtorch activate precompiled/cu128   # work on RTX 50-series code
fdl libtorch info                         # confirm what's active
```

## Install

**From crates.io** (requires Rust):

```sh
cargo install flodl-cli
```

**Pre-compiled binaries** (no Rust needed):

```sh
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl install                # copy to ~/.local/bin/fdl
./fdl install --dev          # symlink (tracks local builds for developers)
./fdl install --check        # compare installed vs latest GitHub release
```

Binaries are published for Linux x86_64/aarch64, macOS arm64, and Windows
x86_64 on every
[GitHub Release](https://github.com/flodl-labs/flodl/releases).

## Command tour

### libtorch management

```sh
fdl libtorch download            # auto-detect GPU, pick variant
fdl libtorch download --cpu      # force CPU (~200MB)
fdl libtorch download --cuda 12.6
fdl libtorch download --cuda 12.8
fdl libtorch list                # all installed variants, active marked
fdl libtorch activate <name>     # switch active variant
fdl libtorch info                # metadata for the active variant
fdl libtorch remove <name>       # delete a variant
```

### Source builds for custom architectures

```sh
fdl libtorch build                        # auto-detect archs and backend
fdl libtorch build --archs "6.1;12.0"     # explicit arch list
fdl libtorch build --docker               # isolated Docker build (default)
fdl libtorch build --native               # host toolchain (faster)
fdl libtorch build --jobs 8               # parallel compilation
fdl libtorch build --dry-run              # show plan, build nothing
```

Docker builds resume via layer cache if interrupted. Output lands in
`libtorch/builds/<arch-signature>/`.

### Diagnostics

```sh
fdl diagnose             # human-readable hardware + compatibility report
fdl diagnose --json      # machine-readable for CI and tooling
```

Reports CPU, OS (with WSL2 detection), Docker version, CUDA driver,
per-GPU architecture and VRAM, active libtorch metadata, and per-GPU
compatibility with the active variant.

## GPU auto-detection

`fdl libtorch download` reads `nvidia-smi` and maps compute capabilities to
the right pre-built variant:

| Your hardware                   | Variant chosen  | Covers          |
|---------------------------------|-----------------|-----------------|
| No GPU                          | CPU             | any host        |
| Pre-Volta only (sm_50 - sm_61)  | cu126           | sm_50 - sm_90   |
| Volta+ only (sm_70+)            | cu128           | sm_70 - sm_120  |
| Mixed archs                     | cu126 + hint    | run `build`     |

Override with `--cpu` or `--cuda <version>`.

## Links

- [Full CLI reference](https://github.com/flodl-labs/flodl/blob/main/docs/cli.md) - every command, every flag, with examples
- [floDl framework](https://flodl.dev) - the Rust deep learning framework `fdl` was built for
- [GitHub Releases](https://github.com/flodl-labs/flodl/releases) - pre-compiled binaries
- [GitHub repository](https://github.com/flodl-labs/flodl)

## License

floDl is open-sourced software licensed under the [MIT license](https://github.com/flodl-labs/flodl/blob/main/LICENSE).
