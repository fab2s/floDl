# Apple Silicon (Mac M1 / M2 / M3 / M4 / M5)

flodl runs on Apple Silicon Macs through Docker (Linux arm64). CUDA is not
available on macOS — the supported path on these machines is **CPU via the
`dev` Docker service**.

The native macOS build (using the `.dylib` libtorch from PyTorch) is not
currently wired into the project's `fdl` CLI. The bind-mounted Docker setup is
how `fdl build`, `fdl test`, `fdl clippy`, etc. are expected to work on a Mac.

## Prerequisites

- macOS 12+ on Apple Silicon (any M-series chip).
- Docker Desktop or [OrbStack](https://orbstack.dev/) (recommended — lighter,
  faster file sharing). The instructions below assume the Docker CLI works
  (`docker info` succeeds).
- ~2 GB free disk for libtorch + dependency caches, plus the usual cargo
  target directory.

You do **not** need Rust installed on the host: the dev container ships its
own toolchain. A native `fdl` binary on the host is convenient for
`fdl -h` / `fdl api-ref`, and is auto-downloaded on first `./fdl` invocation.

## One-time setup

### 1. Fetch the libtorch Linux arm64 build

PyTorch does not publish a pre-built `libtorch-linux-aarch64.zip` on
`download.pytorch.org/libtorch/`. We extract it from the official PyPI wheel
instead. This snippet runs inside a throw-away `python:3.12-slim`
container so the resulting `.so` files are unambiguously Linux arm64:

```bash
mkdir -p libtorch/precompiled/cpu-linux-aarch64
docker run --rm --platform linux/arm64 \
  -v "$PWD/libtorch/precompiled/cpu-linux-aarch64:/out" \
  python:3.12-slim bash -lc '
set -e
mkdir -p /tmp/dl && cd /tmp/dl
pip download --no-deps --only-binary=:all: \
  --platform manylinux_2_28_aarch64 \
  --python-version 3.12 --implementation cp --abi cp312 \
  torch==2.10.0 -d /tmp/dl
WHL=$(ls /tmp/dl/torch-*.whl | head -1)
python -c "import zipfile; zipfile.ZipFile(\"$WHL\").extractall(\"/tmp/whl\")"
find /out -mindepth 1 -delete
cp -a /tmp/whl/torch/include /out/
cp -a /tmp/whl/torch/lib     /out/
cp -a /tmp/whl/torch/share   /out/
'
```

Sanity checks:

```bash
test -f libtorch/precompiled/cpu-linux-aarch64/include/torch/csrc/api/include/torch/torch.h
ls    libtorch/precompiled/cpu-linux-aarch64/lib/libtorch*.so
du -sh libtorch/precompiled/cpu-linux-aarch64
# expected: libtorch.so + libtorch_cpu.so + libc10.so present, total ~380 MB
```

### 2. Point the default mount path at the Linux build

`docker-compose.yml` bind-mounts `./libtorch/precompiled/cpu` into the
container. On a Mac that directory ships PyTorch's macOS `.dylib` files, which
the Linux container can't use. Make `cpu/` a symlink to the Linux arm64 build:

```bash
cd libtorch/precompiled
mv cpu cpu-macos-arm64               # preserve the macOS .dylib files
ln -s cpu-linux-aarch64 cpu          # cpu/ now points at the Linux .so files
cd ../..
```

Why a symlink and not the `LIBTORCH_CPU_PATH` env override? `docker compose run`
honours `.env` for variable substitution, but the `fdl` CLI wrapper bypasses
that path and always expands `${LIBTORCH_CPU_PATH:-./libtorch/precompiled/cpu}`
to its hard-coded default. Re-pointing the default via a symlink is the
non-destructive fix.

### 3. Create `.env` for the docker-compose runtime

```bash
cat > .env <<'EOF'
# Throttle cargo's link parallelism. Required on macOS Docker / OrbStack
# where the virtiofs-mounted libtorch directory cannot serve many concurrent
# `ld` lookups — the linker reports `cannot find -ltorch` spuriously.
CARGO_BUILD_JOBS=2

# Host UID/GID so files written into the bind-mount aren't owned by root.
UID=501
GID=20
EOF
```

Adjust `UID` / `GID` to your account (`id -u` / `id -g`). The `.env` file is
gitignored — it's host-specific.

Those three values are all a Mac needs. `.env.example` at the repo root is the
full reference if you want the other compose knobs (libtorch variant overrides,
CUDA image tag, verbosity).

### 4. Build the dev image

```bash
docker compose build dev
```

This pulls `ubuntu:24.04` (arm64), installs the Rust toolchain, and is
unrelated to libtorch (libtorch is bind-mounted at run time). The image is
small (~2 GB with toolchain) and caches well between rebuilds.

The `cuda` and `bench` services in `docker-compose.yml` require an NVIDIA
runtime and are unusable on a Mac — leave them alone. `docker compose build dev`
only builds the dev service; avoid `docker compose up` (which would try to
start all services).

## Daily use

```bash
./fdl -h           # CLI help (runs natively on macOS arm64)
./fdl api-ref      # framework API reference (no container needed)
./fdl build        # cargo build in the dev container
./fdl test         # cargo test in the dev container
./fdl clippy       # workspace lint
./fdl shell        # interactive bash in the dev container
```

Expected timing on an M-series chip (first run includes deps download and
the C++ shim compile, ~17 s for the libtorch headers alone):

| Command       | First run | Incremental |
| ------------- | --------: | ----------: |
| `fdl build`   |   ~5 min  |     ~2 s    |
| `fdl test`    |  ~10 min  |    ~30 s    |
| `fdl clippy`  |   ~3 min  |     ~5 s    |

## Troubleshooting

### `ld: cannot find -ltorch` during `fdl test`

`CARGO_BUILD_JOBS` is unset or too high. The virtiofs bind-mount can't serve
many parallel `ld` library lookups against libtorch's 244 MB `libtorch_cpu.so`.
Verify `CARGO_BUILD_JOBS=2` is in `.env`, then check it reaches the container:

```bash
docker compose run --rm dev env | grep CARGO_BUILD_JOBS
# CARGO_BUILD_JOBS=2
```

If empty, confirm that `docker-compose.yml` includes `- CARGO_BUILD_JOBS` in
the `dev` service's `environment:` list.

### `libtorch_cpu.so: cannot open shared object file`

The container is mounting the macOS `.dylib` directory instead of the Linux
`.so` directory. Inspect what's actually mounted:

```bash
docker compose run --rm dev ls /usr/local/libtorch/lib | head
# you want .so files (libtorch.so, libtorch_cpu.so, libc10.so), not .dylib
```

If you see `.dylib`, the symlink swap in step 2 was not applied. Redo it:

```bash
ls -la libtorch/precompiled/cpu                 # should be a symlink
# lrwxr-xr-x ... cpu -> cpu-linux-aarch64
```

### `error[E0308]: mismatched types ... expected *const u8, found *const i8`

A portability bug, fixed at the source since 0.6.0: `c_char` is `u8` on Linux
aarch64 and `i8` on Linux x86_64, and `flodl-sys` used to hardcode `*mut i8` in
every extern signature returning a C string. Those signatures now say
`*mut c_char`, so on aarch64 the crate simply compiles and no call site needs a
cast of its own. If you hit this error, your checkout predates 0.6.0 — update
rather than adding casts locally.

### `cargo` slow even after the first build

OrbStack's virtiofs caching is per-path. Touching `Cargo.toml` invalidates
the entire `target/` tree from the host's point of view, but the host's
incremental compilation cache (`.cargo-cache/` and `.cargo-git/`, both
bind-mounted) survives. If you're seeing repeated full rebuilds, check that
`.cargo-cache/` exists at the project root and isn't being wiped between runs.

### `docker compose up` errors with `driver: nvidia`

Don't run `docker compose up`. The `cuda` and `bench` services declare
`driver: nvidia` deploy requirements that fail on a Mac. Use
`docker compose run --rm dev …` (which `./fdl` does for you) or
`docker compose build dev` to target only the CPU service.

## Going back to native macOS later

The macOS arm64 libtorch (PyTorch's `.dylib` build) is preserved in
`libtorch/precompiled/cpu-macos-arm64/`. To switch back from Docker to a
native macOS build:

```bash
cd libtorch/precompiled
rm cpu                                # remove the symlink
mv cpu-macos-arm64 cpu                # restore the macOS .dylib files at the
                                      # default location used by fdl
cd ../..
# … then point LIBTORCH_PATH at /full/path/to/libtorch/precompiled/cpu in
# your shell, install cargo natively, and run `cargo build` outside Docker.
```

Native macOS builds aren't covered by the project's `fdl` flows today; you'd
be invoking `cargo` directly. This is fine for development, but `fdl build` /
`fdl test` continue to assume the Docker dev container.
