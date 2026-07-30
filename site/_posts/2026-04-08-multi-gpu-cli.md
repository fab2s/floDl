---
title: "Two GPUs, one bug, and a CLI"
subtitle: "v0.3.0: heterogeneous multi-GPU that works, a CPU path that doesn't (yet), and a standalone tool for libtorch"
date: 2026-04-08
description: "flodl v0.3.0 ships El Che DDP where a 5060 Ti + 1060 trains faster than the 5060 Ti alone, a standalone CLI for libtorch management, and an honest account of the CPU averaging bug we're still chasing."
---

I have two GPUs: an RTX 5060 Ti (16GB, Blackwell) and a GTX 1060 (6GB,
Pascal). The 5060 Ti is 2.8x faster. Every DDP framework I've tried makes
the fast GPU wait for the slow one. You end up with 1060-speed training on
5060 Ti hardware.

flodl v0.3.0 fixes that. Mostly.

## The part that works

Five epochs, 20k samples, batch size 52, same model and data across all
runs. Two solo baselines, three DDP policies on NCCL.

| Mode | Epoch 5 accuracy | Avg epoch (s) | GPU util |
|---|---:|---:|---:|
| Solo 5060 Ti | 89.1% | 135.7 | - |
| Solo 1060 | 88.4% | 380.4 | - |
| DDP Sync (NCCL) | 94.8% | 225.4 | 64% |
| DDP Cadence (NCCL) | 79.1% | 140.9 | 90% |
| DDP Async (NCCL) | 87.7% | 118.6 | 87% |

The async policy (El Che) trains at 118.6 seconds per epoch. That's
faster than the 5060 Ti alone (135.7s). Both GPUs contributing real
throughput, the fast one not waiting for the slow one, convergence
tracking the solo baselines.

Sync is the worst choice for mixed hardware: 225s/epoch because the
5060 Ti idles during every AllReduce barrier. You get great convergence
(94.8%) at terrible throughput. Cadence is the fastest per-epoch (90%
GPU utilization) but needs more epochs to close the convergence gap.

Async hits the sweet spot. The fast GPU runs ahead between averaging
events, creating useful parameter diversity. Both GPUs contribute
proportionally to their speed.

## The part that doesn't

flodl's DDP has two backends: NCCL (GPU-to-GPU AllReduce) and CPU
(snapshot parameters to host, average there, copy back). The NCCL
numbers above are solid. The CPU backend is broken.

| Mode | Epoch 5 accuracy | Notes |
|---|---:|---|
| NCCL Async | 87.7% | Clean convergence |
| NCCL Sync | 94.8% | Clean convergence |
| CPU Async | 23.1% | Attention collapse |
| CPU Sync | ~8% | Barely above random |

CPU sync is worse than CPU async. That's the clue. More frequent
averaging produces more damage. Each round-trip through the CPU path
(snapshot to host, average, copy back) is destroying the parameters a
little. Sync does it every batch, so the model never recovers.

We found one bug already: `snapshot_params()` was reading GPU memory
while `load_averaged()` was still writing to it via a non-blocking
CUDA stream copy. A classic race condition. We fixed it
(`comm_stream.synchronize()` before the snapshot). The fix was real, but
it wasn't enough.

The deeper issue is somewhere in the copy/average/load cycle itself.
NCCL does in-place AllReduce directly on GPU tensors with hardware-level
synchronization. The CPU path does manual GPU-to-CPU copy, CPU-side
weighted averaging, and CPU-to-GPU distribution. Something in that chain
degrades the parameters on every cycle. We're instrumenting the pipeline
with parameter checksums at each stage to find exactly where.

This is v0.3.0 with the CPU backend marked experimental. It ships because
the NCCL path works and the honest label is better than a delayed release.
The CPU investigation continues.

## Why heterogeneous DDP matters

My setup is the extreme case: a 2.8x speed gap between GPUs. But most
small labs and independent researchers have some version of this problem.
A 3090 paired with a 3060. A 4080 next to a 3070. A workstation with
last year's card and this year's upgrade. Even a 1.5x speed difference
means sync DDP wastes 30% of the faster card on idle wait.

The standard answer is "buy matching GPUs" or "rent a homogeneous
cluster." That's not an answer for a grad student, a small team, or
anyone doing research on their own hardware. Traditional DDP forces
the fast card down to the slow card's pace. El Che lets each GPU
contribute what it can.

This isn't about competing with 8xH100 clusters. It's about making
the hardware you already have useful for real training, not just
inference. If you have two GPUs of any generation, they can train
together without the fast one waiting for the slow one.

## How it works

```rust
use flodl::distributed::*;

let ddp = Trainer::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(52)
    .num_epochs(50)
    .max_grad_norm(5.0)
    .policy(ApplyPolicy::Async)
    .backend(AverageBackend::Nccl)
    .run()?;

let trained = ddp.join()?;
```

Change `.policy(ApplyPolicy::Async)` to `.policy(ApplyPolicy::Sync)` or
`.policy(ApplyPolicy::Cadence)` and rerun. Same model, same data, one
line different. Run 3-5 epochs with each, compare loss curves, commit
to the winner. A/B testing DDP configs takes minutes, not hours.

Three policies, each with a clear profile:

- **Async** (El Che): fast GPU runs ahead, averaging happens
  asynchronously. Best throughput, good convergence.
- **Cadence**: averaging at adaptive intervals. High GPU utilization,
  convergence needs more epochs.
- **Sync**: AllReduce every batch. Best convergence, slowest on mixed
  hardware.

Per-worker gradient clipping (`max_grad_norm`) turned out to be essential
for DDP stability. Without it, unclipped gradients on any single worker
propagate through AllReduce and destabilize all replicas.

## The CLI: `fdl`

v0.3.0 also ships `fdl`, a standalone CLI for libtorch management.
It works inside flodl projects and standalone (installs to `~/.flodl/`).
No libtorch, no CUDA toolkit, no Python needed to run it.

```bash
# Install
cargo install flodl-cli    # installs the fdl binary

# Or download pre-compiled (~750KB, no Rust needed)
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
```

**`fdl setup`** detects your GPUs, downloads the right libtorch variant,
and configures your build. One command, no manual CUDA version matching.

```
$ fdl setup
Detecting hardware...
  CPU: i9-9900K (16 threads, 24GB RAM)
  GPU 0: RTX 5060 Ti (sm_120, 15GB)
  GPU 1: GTX 1060 6GB (sm_61, 6GB)
  Docker: 29.3.1

GPUs span sm_61-sm_120. No single pre-built libtorch covers both.
Building from source for architectures: 6.1, 12.0
```

**`fdl diagnose`** gives you the hardware report you wish every ML
framework had:

```
$ fdl diagnose
libtorch
  Active:   builds/sm61-sm120
  Version:  2.10.0
  CUDA:     12.8
  Archs:    6.1 12.0

Compatibility
  GPU 0 (RTX 5060 Ti, sm_120):  OK
  GPU 1 (GTX 1060 6GB, sm_61):  OK
```

**`fdl libtorch`** manages multiple variants side by side. Download
pre-built CPU/CUDA variants, build from source for exotic GPU combos,
switch between them with `fdl libtorch activate`.

**`fdl init my-project`** scaffolds a complete project: Cargo.toml,
Dockerfile, Makefile, training template. From zero to `make build` in
one command.

The CLI is a pure Rust binary with zero crate dependencies. ~750KB,
compiles in under a second, works on any machine. GPU features degrade
gracefully when nvidia-smi is absent.

## What's next

The NCCL DDP benchmarks cover 5 epochs. Longer convergence runs and a
comprehensive benchmark suite across all configurations are in progress.

The CPU averaging bug is the immediate priority. The investigation points
to something in the snapshot/average/load round-trip that degrades
parameters on every cycle. We'll find it, fix it, and publish the results.

A comprehensive benchmark suite covering all DDP modes across common
training patterns is being built. When it's done, there will be definitive
data on which configuration works best for which kind of model. No
opinions, just numbers.

We ship what works. We're honest about what doesn't. We keep going.

---

[Full DDP reference](/guide/ddp/reference) | [Tutorial: DDP Builder](/guide/async-ddp) |
[CLI docs](/guide/cli/install) | [GitHub](https://github.com/flodl-labs/flodl)
