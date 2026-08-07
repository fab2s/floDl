# Windows (WSL2)

The supported path on Windows hardware is **WSL2**. A WSL2 distribution is
ordinary `x86_64-unknown-linux-gnu`, so it runs the Linux `fdl` binary, the
Linux libtorch, full CUDA, and full NCCL multi-GPU training. Nothing is
emulated and nothing is cut down.

This is not a fallback. floDl's published benchmarks were produced on this
exact configuration (see [Benchmarks](benchmark.md)):

| | |
| --- | --- |
| OS | Ubuntu 24.04 (Docker on WSL2) |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| CPU | AMD Ryzen 7 7800X3D |

## What runs where

| | WSL2 | Native Windows (`fdl.exe`) |
| --- | --- | --- |
| `fdl libtorch download` / `list` | yes | yes |
| `fdl init`, `fdl setup` | yes | no |
| Build, test, train | yes | no |
| CUDA, multi-GPU, NCCL / DDP | yes | no |
| Cluster ops (`join`, `publish`, `join-config`) | yes | no |

`fdl.exe` exists as a **libtorch fetcher** for people consuming libtorch
directly from Windows (C++ LibTorch, `tch-rs`, or a PyTorch install that
needs the matching C++ archive). It downloads and unpacks the correct
`libtorch-win-shared-with-deps-*` archive for a chosen CUDA version and
stops there.

It cannot build or train, and this is not a gap waiting to be filled:
`flodl-sys` has never been compiled against MSVC, and more decisively
**NCCL has no Windows port** (neither does RCCL), so a native Windows build
could never do multi-GPU DDP at all. WSL2 has both. Use WSL2.

## Prerequisites

- Windows 11, or Windows 10 21H2+ (GPU passthrough needs `/dev/dxg`).
- The GPU driver installed **on Windows, not inside the distribution**.
  This is the one setup mistake worth calling out: installing a Linux GPU
  driver inside WSL overwrites the passthrough stubs and breaks CUDA. The
  Windows driver supplies `nvidia-smi` inside the distro through
  `/usr/lib/wsl/lib/`.
- Docker Desktop with the WSL2 backend, or Docker installed inside the
  distribution. Either works.

Verify the GPU is visible from inside WSL before going further:

```bash
nvidia-smi          # should list your GPU(s)
ls /dev/dxg         # the passthrough device
```

If `nvidia-smi` is missing, the Windows-side driver is too old or absent.
Fix that before installing anything in the distro.

## Setup

Inside the WSL distribution, follow the ordinary Linux instructions:

```bash
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl install
fdl init my-project
```

Then confirm what floDl sees. `fdl probe` labels the kernel explicitly, so
a WSL2 host is unambiguous in bug reports:

```
OS:          Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
```

## Two settings that matter for floDl

**Keep the project on the WSL filesystem.** Work in `~/src/...` inside the
distribution, not `/mnt/c/...`. The Windows drive is reached over a
translation layer, and a cargo target directory or a dataset served across
it is dramatically slower than the native ext4 one. This dominates every
other performance knob on WSL.

**Give WSL enough RAM, and know what it reports.** WSL2 runs in a
lightweight VM whose memory defaults to a fraction of the host's and grows
on demand. floDl's data plane budgets against `MemAvailable`, so it sizes
its caches to the VM's view, not the machine's. If the loader seems to
under-cache on a large-RAM box, raise the VM's ceiling in
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=48GB
processors=16
```

Restart with `wsl --shutdown` from PowerShell for it to take effect.

## Multi-GPU

Multi-GPU DDP works under WSL2 with NCCL, which is why WSL is the supported
path rather than a compromise. The usual commands apply:

```bash
fdl gpu-test          # GPU test suite
fdl gpu-test-nccl     # NCCL / DDP tests, isolated processes
```

See [Multi-GPU Training](tutorials/11-multi-gpu.md) for the full picture.

## AMD GPUs

Use bare-metal Linux. AMD's ROCm has a WSL path, but floDl's AMD support
detects GPUs through the kernel's KFD topology (`/dev/kfd` plus
`/sys/class/kfd/`), which is the bare-metal interface. We have not
validated floDl on ROCm-under-WSL and make no claim about it.

## Troubleshooting

### `nvidia-smi` works on Windows but not inside WSL

The distribution predates GPU support or the driver is too old. Update the
Windows-side driver, then `wsl --shutdown` and reopen. Do not install a
Linux GPU driver inside the distribution to work around this: it replaces
the passthrough libraries in `/usr/lib/wsl/lib/` and makes the problem
permanent.

### Builds are far slower than on native Linux

The project is almost certainly under `/mnt/c/`. Move it into the
distribution's own filesystem (`~/`) and rebuild. `df -h .` from the
project root tells you which you are on: the WSL filesystem shows as
`/dev/sd*`, the Windows drive as `drvfs`.

### The data loader caches less than expected

The WSL VM's memory ceiling, not the host's RAM, is what `MemAvailable`
reports. Raise `memory=` in `.wslconfig` (above) and `wsl --shutdown`.

### `fdl.exe` says a variant is unavailable

ROCm libtorch is published for Linux only, so `fdl.exe` refuses it rather
than fetching a URL that does not exist. CPU and CUDA variants are
available for Windows.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Distributed Architecture](distributed/architecture.md) | Next: [Mac / Apple Silicon](mac-apple-silicon.md)
