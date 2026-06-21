---
title: "70 runs, 8 models, zero failures"
subtitle: "v0.4.0: CPU averaging works, every DDP mode beats solo, and we have the numbers to prove it"
date: 2026-04-14
description: "flodl v0.4.0 ships a comprehensive DDP benchmark suite: 8 models, 8 distributed modes, 70 training runs. Every DDP mode surpasses solo GPU convergence on ResNet-20 CIFAR-10 while training faster. The CPU averaging bug from v0.3.0 is fixed."
---

Six days ago, v0.3.0 shipped with a confession: the CPU averaging
backend was broken. Three policies, all failing to converge. We shipped
anyway because the NCCL path worked and an honest label beats a delayed
release.

That bug is fixed. And we can prove it.

## The fix

One line. `comm_stream.synchronize()` before reading GPU parameters for
CPU averaging snapshots. The `snapshot_params()` call was racing with
`load_averaged()` - a non-blocking CUDA stream copy was still writing
to GPU memory when the next snapshot tried to read it. Classic race
condition, subtle because it only corrupted parameters gradually over
hundreds of averaging rounds.

We suspected this in v0.3.0 but couldn't confirm it was the only
bug. Now we can: 70 training runs across 8 models and 8 DDP modes,
every single CPU averaging configuration converges correctly.

## The proof

We built `ddp-bench`, a dedicated benchmark suite that runs every model
against every DDP mode and validates convergence against published
results. Not toy models - ResNet-20 on CIFAR-10, GPT-nano on
Shakespeare, char-RNN, LeNet, convolutional autoencoders. Real
architectures with known convergence curves.

The hardware: an RTX 5060 Ti (sm_120, 15GB) and a GTX 1060 (sm_61, 6GB).
A 2.5x compute gap between GPUs. No pre-built libtorch covers both
architectures - we compile from source with `fdl libtorch build`.
This is the kind of setup traditional DDP frameworks choke on.

The modes:

- **solo-0 / solo-1**: single-GPU baselines (fast GPU / slow GPU)
- **sync**: Graph-based lock-step DDP (every batch synchronized)
- **nccl-sync / nccl-cadence / nccl-async**: NCCL AllReduce with three El Che policies
- **cpu-sync / cpu-cadence / cpu-async**: CPU averaging with the same three policies

That's 9 configurations per model, 200 epochs on ResNet-20 and
ResNet-Graph, 5 epochs on the smaller models.

## ResNet-20: every DDP mode beats solo

This is the headline result. On CIFAR-10 with 200 epochs, every DDP
mode exceeds the solo-0 baseline (91.62% accuracy on the RTX 5060 Ti):

| Mode | Eval | vs Published | Wall time | Speedup |
|------|-----:|-------------:|----------:|--------:|
| solo-0 (5060 Ti) | 91.62% | +0.37pp | 2994s | 1.0x |
| solo-1 (1060) | 91.87% | +0.62pp | 3680s | 0.8x |
| nccl-sync | 92.19% | +0.94pp | 2491s | 1.2x |
| nccl-cadence | 92.00% | +0.75pp | 2574s | 1.2x |
| nccl-async | 92.10% | +0.85pp | 2522s | 1.2x |
| cpu-sync | 92.44% | +1.19pp | 4356s | 0.7x |
| cpu-cadence | 91.75% | +0.50pp | 2467s | 1.2x |
| cpu-async | 91.92% | +0.67pp | 2534s | 1.2x |

Published reference: 91.25% ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6).

Every mode beats the paper. Every mode beats solo-0. The fastest DDP
modes (cpu-cadence, nccl-async) finish in ~42 minutes vs solo-0's ~50
minutes. Two mismatched GPUs, training faster than the fast GPU alone,
with better convergence.

cpu-sync is the outlier: best eval (92.44%) but slowest, because
every-batch CPU round-trips are expensive. The El Che policies
(cadence, async) let the fast GPU range ahead between sync points,
which is where the speedup comes from.

## CPU averaging: from broken to production

In v0.3.0, all three CPU averaging policies failed to converge.
Parameters degraded over hundreds of averaging rounds until the model
collapsed. We shipped it labeled as a known bug.

In v0.4.0, all three work:

| Mode | v0.3.0 | v0.4.0 |
|------|-------:|-------:|
| CPU Sync | broken | 92.44% |
| CPU Cadence | broken | 91.75% |
| CPU Async | broken | 91.92% |

The CPU path now matches NCCL convergence across all three policies
and all eight benchmark models. Both backends are production-ready.

## Why this was hard

flodl doesn't use PyTorch's `c10d` distributed backend. We call NCCL
directly through FFI - raw `ncclAllReduce`, `ncclCommInitRank`,
manual stream synchronization. This gives us control (El Che's
per-rank cadence wouldn't be possible through `c10d`'s collective
API), but it means we own every synchronization bug.

The NCCL init pattern alone took days to get right. `ncclCommInitRank`
called from worker threads corrupts the CUDA context on heterogeneous
GPUs. The fix: always init on the main thread, then `ncclCommSplit()`
to create per-rank communicators. Not documented anywhere. We found it
by reading NCCL source and correlating CUBLAS failures with init order.

The CPU averaging path is a three-phase state machine
(Idle/Collecting/Computing) running on a coordinator thread. GPU
workers send parameter snapshots via channels, the coordinator averages
on CPU, then distributes back. Non-blocking at every stage - the
coordinator keeps processing control messages while averaging runs.
Getting the stream synchronization right between snapshot reads, NCCL
transfers, and CPU copies required instrumenting every stage with
checksums until we found the exact copy that was racing.

Building libtorch from source for sm_61 + sm_120 (Pascal + Blackwell)
takes 4 hours. No pre-built binary covers both. `fdl libtorch build`
automates this, but when something goes wrong in DDP, you're debugging
through three layers: your Rust code, the C++ FFI shim, and the
libtorch internals. There's no Python stacktrace to fall back on.

## El Che: the mismatched GPU problem

Traditional sync DDP makes the fast GPU wait for the slow one. With a
2.5x speed gap, you train at 1060 speed on 5060 Ti hardware.

El Che (Elastic Checkpoint) solves this with three policies:

**Sync**: every-batch AllReduce. Maximum gradient freshness, but the
fast GPU idles during every barrier. Best convergence, worst throughput
on mixed hardware.

**Cadence**: the slow GPU anchors the sync interval. The fast GPU
processes more batches between syncs, contributing proportionally to
its speed. Auto-tunes the anchor to keep AllReduce overhead below 10%.

**Async**: workers run independently, averaging when ready. The fast
GPU ranges ahead, creating useful parameter diversity between averaging
events. Fastest wall time, slightly noisier convergence that smooths
out over epochs.

On ResNet-20, all three converge to within 0.5pp of each other. The
choice is a throughput/freshness tradeoff, not a convergence risk.

## The benchmark suite

`ddp-bench` is included in the repository. Run it yourself:

```bash
fdl ddp-bench full-sweep          # all models, all modes
fdl ddp-bench --model resnet --mode nccl-async
fdl ddp-bench report              # generate markdown report
```

Every run produces a timeline (JSON, CSV, HTML) with GPU utilization
traces, sync event markers, idle gap analysis, and per-epoch
convergence data. The [full report](/ddp-benchmark) has the complete
results for all 70 runs, and the
[detailed data](https://github.com/flodl-labs/flodl/blob/main/docs/ddp-benchmark.md)
is on GitHub with reproduction instructions.

Eight models, each chosen to stress a different aspect:

| Model | Tests | Published |
|-------|-------|-----------|
| Logistic regression | Linear baseline | MNIST ~92% |
| 2-layer MLP | Dispatch overhead | MNIST ~97% |
| LeNet-5 | Conv + pooling | MNIST ~99% |
| Conv autoencoder | Reconstruction (MSE) | MNIST |
| Char-RNN | RNN sequence modeling | Shakespeare, loss ~1.5 |
| GPT-nano | Transformer + attention | Shakespeare, loss ~1.5-2.0 |
| ResNet-20 | Deep residual, BN, augmentation | CIFAR-10 91.25% |
| ResNet-20 (Graph) | Same via FlowBuilder | CIFAR-10 91.25% |

## What's next

This release validates DDP on local hardware. The next step is
cloud: multi-node training across machines, where network latency
replaces PCIe as the bottleneck. El Che's cadence policies were
designed with this in mind - the same anchor/range-ahead principle
applies whether the slow device is a weaker GPU or a node behind a
network hop.

The [full DDP benchmark report](/ddp-benchmark) is available on the
website with per-model tables, convergence trajectories, GPU idle
analysis, and El Che calibration data. The
[raw report](https://github.com/flodl-labs/flodl/blob/main/docs/ddp-benchmark.md)
on GitHub includes reproduction instructions.

[GitHub](https://github.com/flodl-labs/flodl) &#124;
[Documentation](https://flodl.dev/guide/) &#124;
[DDP Benchmark](/ddp-benchmark) &#124;
[DDP Report (GitHub)](https://github.com/flodl-labs/flodl/blob/main/docs/ddp-benchmark.md) &#124;
[Changelog](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md)
