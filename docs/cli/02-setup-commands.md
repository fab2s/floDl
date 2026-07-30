# Project and libtorch Commands

Interactive wizard that walks you through everything:

1. **Detects your system** - CPU, RAM, Docker, Rust, GPUs.
2. **Downloads libtorch** - auto-picks the right variant for your GPU(s).
3. **Configures your build** - Docker or native, builds images if needed.

```bash
fdl setup                      # interactive (asks questions)
fdl setup --non-interactive    # auto-detect everything, no prompts
fdl setup -y                   # alias for --non-interactive
fdl setup --force              # re-download even if libtorch exists
```

The wizard handles tricky scenarios automatically:

- **No GPU?** Downloads CPU libtorch.
- **Volta+ GPUs (sm_70+)?** Downloads cu128.
- **Pre-Volta GPUs (sm_50-sm_61)?** Downloads cu126.
- **Mixed GPUs (old + new)?** Offers to build from source or pick the best
  pre-built variant.

## `fdl libtorch`

Manage libtorch installations. Variants live under `libtorch/` in your
project (or `$FLODL_HOME/libtorch/` when standalone), each with a
metadata `.arch` file. An `.active` pointer selects the current one.

#### `fdl libtorch download`

Download a pre-built libtorch from PyTorch's official mirrors.

```bash
fdl libtorch download              # auto-detect GPU, pick best variant
fdl libtorch download --cpu        # force CPU-only (~200MB)
fdl libtorch download --cuda 12.8  # CUDA 12.8 / cu128 (~2GB)
fdl libtorch download --cuda 12.6  # CUDA 12.6 / cu126 (~2GB)
fdl libtorch download --path ~/lib # install to a custom directory
fdl libtorch download --no-activate # install but do not switch `.active`
fdl libtorch download --dry-run    # show what would happen
```

`--cuda` only accepts `12.6` or `12.8` (the published pre-built
versions). Auto-completion offers both.

**Variant coverage:**

| Variant | Architectures  | GPUs                              |
|---------|----------------|-----------------------------------|
| CPU     | -              | Any (no GPU acceleration)          |
| cu126   | sm_50 to sm_90 | Maxwell through Ada Lovelace      |
| cu128   | sm_70 to sm_120 | Volta through Blackwell          |

If your GPUs span both ranges (e.g. GTX 1060 + RTX 5060 Ti), no single
pre-built variant covers both. Use `fdl libtorch build` instead.

#### `fdl libtorch build`

Compile libtorch from PyTorch source for your exact GPU combination.
Takes 2-6 hours depending on CPU cores. Two build methods are available:

- **Docker** (default when available) - isolated, reproducible, resumes
  via layer caching. Requires Docker.
- **Native** - faster, builds directly on your host. Requires CUDA
  toolkit (nvcc), cmake, python3, git, and gcc.

When both are available, the CLI asks which you prefer. Use `--docker`
or `--native` to skip the prompt.

```bash
fdl libtorch build                         # auto-detect GPUs and backend
fdl libtorch build --native                # force native build
fdl libtorch build --docker                # force Docker build
fdl libtorch build --archs "6.1;12.0"      # explicit architectures
fdl libtorch build --jobs 8                # parallel compilation jobs (default: 6)
fdl libtorch build --dry-run               # show plan without building
```

Output lands in `libtorch/builds/<arch-signature>/` (e.g.
`libtorch/builds/sm61-sm120/`).

**Native build requirements:**

| Tool     | Purpose               | Install                                              |
|----------|-----------------------|------------------------------------------------------|
| nvcc     | CUDA compiler         | [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) |
| cmake    | Build system          | `apt install cmake` / `brew install cmake`           |
| python3  | PyTorch build scripts | Usually pre-installed                                 |
| git      | Clone PyTorch source  | `apt install git`                                     |
| gcc/g++  | C++ compilation       | `apt install gcc g++`                                 |

Python packages (pyyaml, jinja2, etc.) install automatically via pip.
The PyTorch source is cached at `libtorch/.build-cache/pytorch/`, so
re-running after a failure skips the clone.

#### `fdl libtorch list / info / activate / remove`

```bash
fdl libtorch list            # human-readable
fdl libtorch list --json     # machine-readable
fdl libtorch info            # show active variant details
fdl libtorch activate <name> # switch the active variant
fdl libtorch remove <name>   # delete a variant (clears .active if it was active)
```

`activate` and `remove` take a variant name as shown by
`fdl libtorch list` (e.g. `precompiled/cu128`, `builds/sm61-sm120`).
Passing no name prints the list and exits.

Example `info` output:

```
Active:   builds/sm61-sm120
Version:  2.10.0
CUDA:     12.8
Archs:    6.1 12.0
Source:   compiled
```

#### Using `fdl` as a standalone libtorch manager (tch-rs / PyTorch C++)

The libtorch-management and diagnostics commands are independent of
flodl and fill a gap PyTorch itself never filled: a proper installer.
`fdl` works as a drop-in libtorch manager for:

- **tch-rs projects** - download the right libtorch, point `LIBTORCH`
  at it, build. No more hand-fetching URLs from the PyTorch
  get-started page.
- **PyTorch C++ development** - juggle CPU, CUDA 12.6, CUDA 12.8, and
  source-built variants on the same host without symlink choreography.
- **Mixed-GPU systems** - when no single pre-built variant covers
  your architectures (e.g. GTX 1060 sm_61 + RTX 5060 Ti sm_120),
  `fdl libtorch build` compiles PyTorch from source with the exact
  archs you need. Docker-isolated by default, native toolchain
  supported.
- **CI pipelines** - `fdl diagnose --json` emits a machine-readable
  hardware and compatibility report to gate jobs on GPU presence or
  libtorch version.

Standalone (no project directory), everything installs under
`$FLODL_HOME` (default `~/.flodl/`). Pick any location you prefer and
export it before the first command:

```bash
export FLODL_HOME=~/.libtorch-variants
```

**Example A: PyTorch C++ (LibTorch via CMake) on an RTX 50-series GPU.**

This is the canonical C++ API workflow from
[pytorch.org/cppdocs](https://pytorch.org/cppdocs/installing.html),
with `fdl` replacing the manual URL-and-unzip dance:

```bash
# 1. Inspect hardware and download the matching libtorch.
fdl diagnose                          # confirm GPU arch (sm_120 in this case)
fdl libtorch download --cuda 12.8     # ~2GB, unpacks to $FLODL_HOME/libtorch/precompiled/cu128

# 2. Point CMake at it.
export LIBTORCH=$FLODL_HOME/libtorch/precompiled/cu128
```

Minimal `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(my_model)

find_package(Torch REQUIRED)
add_executable(my_model main.cpp)
target_link_libraries(my_model "${TORCH_LIBRARIES}")
set_property(TARGET my_model PROPERTY CXX_STANDARD 17)
```

Build and run:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH ..
cmake --build . --parallel

# Runtime: expose libtorch's shared libs.
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
./my_model
```

To switch CUDA versions (e.g. back to 12.6 for legacy code), install
the other variant with `fdl libtorch download --cuda 12.6`, flip it
with `fdl libtorch activate precompiled/cu126`, re-export `LIBTORCH`,
and re-run CMake. No reinstall, no URL hunting.

**Example B: Rust via tch-rs on the same hardware.**

```bash
# Same download + LIBTORCH export as Example A.
export LIBTORCH=$FLODL_HOME/libtorch/precompiled/cu128
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo add tch
cargo build
```

**Juggling variants across projects.** Install as many as you need
side by side, then flip the active pointer; `LIBTORCH` follows
`.active` when you source it from the `fdl libtorch info` output:

```bash
fdl libtorch download --cpu           # ~200MB, for laptops / CI
fdl libtorch download --cuda 12.6     # legacy CUDA projects
fdl libtorch download --cuda 12.8     # latest

fdl libtorch activate precompiled/cu126   # work on legacy code
fdl libtorch activate precompiled/cu128   # work on RTX 50-series code
fdl libtorch info                         # confirm what's active
```

**Mixed GPUs (no pre-built variant covers you).** If `fdl diagnose`
reports architectures that span both pre-built ranges, build from
source and `fdl` will pick up the compiled variant automatically:

```bash
fdl libtorch build --archs "6.1;12.0"     # Pascal + Blackwell
fdl libtorch list
#   builds/sm61-sm120 (active)
#   precompiled/cu128
export LIBTORCH=$FLODL_HOME/libtorch/builds/sm61-sm120
```

**CI gating example.** Use `diagnose --json` to skip GPU jobs when no
compatible device is present:

```bash
if fdl diagnose --json | jq -e '.cuda.devices | length > 0' > /dev/null; then
    cargo test --features cuda
else
    echo "no GPU detected, skipping CUDA tests"
fi
```

None of the above touches flodl itself - `fdl` is just the libtorch
installer / activator / diagnostics tool in this mode.

## `fdl init`

Scaffold a new floDl project. Three modes, mutually exclusive - pick via
flag, or accept the interactive prompt when none is passed:

```bash
fdl init my-model            # default: Docker with host-mounted libtorch (prompts if interactive)
fdl init my-model --docker   # Docker with libtorch baked into the image
fdl init my-model --native   # no Docker; libtorch and cargo on the host
```

Add `--with-hf` to include the
[flodl-hf](../tutorials/14-flodl-hf.md) HuggingFace playground in the
generated project:

```bash
fdl init my-model --with-hf            # Docker + flodl-hf side crate
fdl init my-model --native --with-hf   # Native + flodl-hf side crate
```

`--with-hf` skips the interactive "Include flodl-hf?" prompt when mode
flags are present. In fully interactive mode (`fdl init my-model` with
no flag), a prompt offers the same choice after the Docker / native
selection. See `fdl add` below for adding flodl-hf to an existing
project later.

In all three modes the scaffold generates:

- `Cargo.toml` - flodl dependency and optimized profiles.
- `src/main.rs` - complete training template.
- `fdl.yml.example` - committed manifest; fdl copies it to a gitignored
  `fdl.yml` on first use. Declares `build` / `test` / `run` / `check` /
  `clippy` (and `shell` / `cuda-shell` in Docker modes) plus the `cuda-*`
  siblings.
- `./fdl` - self-contained bootstrap script (`./fdl install` promotes it
  to `~/.local/bin/fdl`).
- `.gitignore`.

Docker modes additionally generate:

- `Dockerfile` / `Dockerfile.cuda` (mounted variant) or
  `Dockerfile.cpu` / `Dockerfile.cuda` (baked variant).
- `docker-compose.yml`.

Native mode skips all the Docker files - commands run on the host. Point
`$LIBTORCH` / `$LD_LIBRARY_PATH` at a libtorch install (use
`./fdl libtorch download --cpu` or `--cuda 12.8`) and `./fdl build`
dispatches straight to `cargo build`.

> The scaffold is fdl-native: there is no Makefile. Every task lives in
> `fdl.yml` and runs via `./fdl <cmd>`. Libtorch environment variables
> (`LIBTORCH_HOST_PATH`, `CUDA_VERSION`, `CUDA_TAG`) are derived from
> `libtorch/.active` by flodl-cli before each dispatch - the logic that
> used to live in the scaffolded Makefile now lives in one place inside
> the binary.

## `fdl add`

Add an ecosystem crate as a side playground inside an initialised flodl
project. Today this means `flodl-hf` (alias `hf`); the command is
designed to grow as more sibling crates land.

```bash
fdl add flodl-hf             # scaffold ./flodl-hf/
fdl add hf                   # short alias, same effect
```

The scaffold drops a standalone cargo crate under `./flodl-hf/` with
its own `Cargo.toml`, a one-file `AutoModel` classifier
(`src/main.rs`), a nested `fdl.yml` with runnable commands (`classify`,
`bert`, `roberta-sentiment`, `distilbert-sentiment`, plus `build` /
`check` / `shell`), and a `README` covering the three feature flavors
(full / vision-only / offline) and the `.bin`-to-safetensors conversion
workflow.

Key properties:

- **Version lockstep**: the scaffold parses the host project's
  `flodl = "X.Y.Z"` dependency and pins `flodl-hf` to the matching
  `=X.Y.Z`. Git-only or path-only flodl deps error with actionable
  guidance.
- **Scope contract**: no mutation of the host project's root
  `Cargo.toml` or `fdl.yml`. The playground is a side crate for
  discovery; wiring flodl-hf into the main code stays the caller's
  decision.
- **Mode detection**: `fdl add flodl-hf` inspects the parent dir to
  pick Docker or native mode. `docker-compose.yml` present, the
  scaffolded `fdl.yml` keeps `docker: dev` on each cargo command so
  commands dispatch into the `dev` service. `docker-compose.yml`
  absent, the `docker:` lines are stripped.
- **Idempotent**: refuses to overwrite an existing `./flodl-hf/`
  directory. Delete explicitly to regenerate.
- **Requires a flodl project**: either `fdl.yml` or `fdl.yml.example`
  must be present in the parent. Missing manifest errors with
  "expects an initialised flodl project".

See the
[HuggingFace Integration tutorial](../tutorials/14-flodl-hf.md) for the
full usage walkthrough of what the scaffold enables.

## `fdl diagnose`

Hardware and compatibility report. Useful for debugging setup issues or
verifying your GPU + libtorch combination works.

```bash
fdl diagnose             # human-readable report
fdl diagnose --json      # machine-readable for CI and tooling
```

Example output:

```
floDl Diagnostics
=================

System
  CPU:         Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz (16 threads, 24GB RAM)
  OS:          Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
  Docker:      29.3.1

CUDA
  Driver:      576.88
  Devices:     2
  [0] NVIDIA GeForce RTX 5060 Ti -- sm_120, 15GB VRAM
  [1] NVIDIA GeForce GTX 1060 6GB -- sm_61, 6GB VRAM

libtorch
  Active:      builds/sm61-sm120
  Version:     2.10.0
  CUDA:        12.8
  Archs:       6.1 12.0
  Source:      compiled
  Variants:    builds/sm61-sm120, precompiled/cpu

Compatibility
  GPU 0 (RTX 5060 Ti, sm_120):  OK
  GPU 1 (GTX 1060 6GB, sm_61):  OK

  All GPUs compatible with active libtorch.
```

The JSON output is useful for CI pipelines and automated tooling:

```bash
fdl diagnose --json | jq '.cuda.devices[] | .sm'
```

## libtorch directory layout

The CLI manages libtorch installations under `libtorch/` in your project
root:

```
libtorch/
  .active                          # points to current variant (e.g. "builds/sm61-sm120")
  precompiled/
    cpu/                           # pre-built CPU variant
      lib/ include/ share/
      .arch                        # metadata: cuda=none, torch=2.10.0, ...
    cu126/                         # pre-built CUDA 12.6
      ...
    cu128/                         # pre-built CUDA 12.8
      ...
  builds/
    sm61-sm120/                    # source-built for specific GPUs
      lib/ include/ share/
      .arch                        # metadata: cuda=12.8, archs=6.1 12.0, source=compiled
```

The `.arch` file format:

```
cuda=12.8
torch=2.10.0
archs=6.1 12.0
source=compiled
```

Docker Compose and Make targets read `.active` to mount the right
libtorch variant automatically. You never need to set `LIBTORCH_PATH`
manually when using Docker.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Install and Global Flags](01-install.md) | Next: [Cluster and Hardware Commands](03-cluster-commands.md)
