---
title: "The number that matters isn't speed"
subtitle: "Seven models, ten rounds, one finding: Rust doesn't just run faster, it runs the same way every time"
date: 2026-03-25
description: "flodl vs PyTorch on 7 models with publication-grade methodology. Up to 30% faster, but the real story is 3-20x tighter variance from Rust's ownership model."
---

*Update: these v0.1.3 results have been superseded by the
[v0.2.2 benchmarks](/blog/benchmark-update) - 10 models, fused RNN
kernels, and improved statistical methodology.*

When I published the [first benchmark](/benchmark) a week ago, the headline
was 19% faster than PyTorch. A real model, a real workload, same GPU. People
noticed. But one number bugged me: the epoch standard deviation. flodl was
0.10s. PyTorch was 0.85s. Same model. Same CUDA kernels.

That wasn't a measurement artifact. It was the actual finding.

## The suite

The first benchmark was a single model - a fair criticism is that any one
workload might favor one framework by accident. So I built a proper suite.
Seven models spanning the architectures that matter:

| Model | What it tests | Params |
|---|---|---:|
| mlp | Raw matmul throughput (5-layer, 1024→2048) | 16.8M |
| convnet | Conv2d + BatchNorm pipeline | 4.0M |
| gru_seq | 50-step unrolled GRU (sequential dispatch) | 1.3M |
| residual_tower | 12 residual blocks via skip connections | 14.7M |
| gated_routing | 8-expert MoE with learned soft routing | 2.6M |
| iterative_refine | 8-step refinement loop | 3.2M |
| feedback_fixed | 10-step feedback loop (smaller model) | 0.8M |

Tier 1 (mlp, convnet, gru_seq) uses standard nn modules - Linear, Conv2d,
GRUCell. Both sides write the same code. Tier 2 uses flodl's graph builder
on the Rust side and equivalent manual `forward()` on the Python side. Same
architecture, same parameters, same optimizer.

## Methodology

I'm claiming publication-grade numbers. That means publication-grade
methodology:

- **10 interleaved rounds.** Odd rounds run Rust first, even rounds run
  Python first. Thermal drift and background noise distribute equally.
- **GPU clocks at 3090 MHz.** The RTX 5060 Ti ran at its default
  applications clock. On non-WSL systems the suite auto-locks clocks;
  on WSL2, the tight σ across 10 rounds confirms stability without it.
- **Best-of-3 per round.** Each round runs 4 complete passes (first is
  warmup). The best of the 3 measured runs is reported, filtering process-level noise.
- **20 measured epochs** with 3 warmup epochs per pass. 50 batches per epoch.
  Enough data points that the median is meaningful.
- **15-second GPU warmup** burst before any measurement.
- **Clean system.** Rebooted, non-essential services stopped, display off.
- **Docker isolation.** Both frameworks in the same container with identical
  CUDA toolkit and system libraries.

The statistical model: for each model, report the median of 10 best-run
medians (one per round) and their standard deviation. Ten independent samples
is enough for the σ values to mean something.

## Results

| Model | PyTorch | flodl | Delta | Py σ | Rs σ |
|---|---:|---:|---:|---:|---:|
| mlp | 271.0 ms | 188.5 ms | **-30%** | ±10.1 | ±2.9 |
| convnet | 1189.4 ms | 1190.5 ms | +0% | ±2.7 | ±1.0 |
| gru_seq | 1015.3 ms | 949.7 ms | **-6%** | ±222.4 | ±10.8 |
| residual_tower | 371.3 ms | 278.6 ms | **-25%** | ±25.9 | ±3.6 |
| gated_routing | 222.6 ms | 196.9 ms | **-12%** | ±13.8 | ±2.6 |
| iterative_refine | 208.7 ms | 186.7 ms | **-11%** | ±27.2 | ±5.6 |
| feedback_fixed | 250.2 ms | 207.2 ms | **-17%** | ±27.3 | ±8.7 |

flodl wins 6 of 7. The speed numbers are good. But look at the σ column.

## The number that matters

Every single model, without exception, shows tighter variance on flodl.
Not by a little - by 3x to 20x:

- **gru_seq:** PyTorch ±222ms on a 1-second epoch. That's a 22% swing between
  runs. flodl: ±10.8ms - a 1% swing. Same model, same GPU, same CUDA kernels.

- **residual_tower:** PyTorch ±25.9ms. flodl ±3.6ms. Seven times more
  predictable.

- **convnet:** The one model where speed is identical (+0%). Even here,
  flodl's σ is 2.7x tighter. The *consistency* advantage persists even when
  the *speed* advantage doesn't.

This isn't a tuning win. You can't profile your way to lower variance in
PyTorch. The variance comes from:

1. **Garbage collector pauses.** Python's GC runs at unpredictable intervals.
   When it fires during a training epoch, that epoch is slower. The GC doesn't
   know or care that you're in a tight loop.

2. **Interpreter overhead jitter.** Python's bytecode interpreter has variable
   dispatch cost depending on instruction cache state, branch prediction
   history, and what the OS scheduler decided to do between ops.

3. **Deferred deallocation.** Tensor memory freed by Python's reference
   counting + cyclic GC arrives at the CUDA allocator in bursts, causing
   pressure spikes that fragment the allocation pool.

Rust has none of these. Tensor memory is freed by `Drop` at scope exit -
deterministically, on the same thread, at the exact point where the value is no
longer needed. There is no GC to pause. There is no interpreter between you and
the FFI call. The memory behavior is the same on every run because the
execution model is the same on every run.

## The convnet proof

The convnet result at +0% is not a disappointment. It's a proof.

Convolution is compute-bound. Both frameworks spend >99% of their time inside
cuDNN kernels. The framework overhead - dispatch, memory management, Python
interpretation - is invisible because the GPU is doing all the work.

This proves that flodl and PyTorch dispatch identical CUDA kernels. The speed
advantage on other models comes entirely from what happens *between* kernel
launches. When the gap between ops is large enough for framework overhead to
matter (MLP with many small matmuls, GRU with 50 sequential steps), Rust's
zero-overhead dispatch pays off. When the ops are large enough to hide the
overhead (convnet), both frameworks converge to the same number.

This is the cleanest possible evidence that the benchmark is measuring
framework overhead, not CUDA kernel differences.

## What changed since v0.1.1

The first benchmark (19% on FBRL letter model) ran on flodl v0.1.1. Since
then:

- **Fused Adam/AdamW** - single multi-tensor CUDA kernel instead of 4N
  per-parameter launches
- **Foreach operations** - 7 batched tensor ops (zero, norm, scale, lerp,
  sqrt, add, mul) that replace N individual kernel launches
- **Fused gradient clipping** - 2 kernels total instead of 2N
- **CUDA Graphs** - capture/replay kernel sequences for static-shape models
- **Automatic mixed precision** - autocast + GradScaler for fp16/bf16
- **Channels-last memory** - NHWC layout for Conv2d on Tensor Core GPUs
- **Pre-computed graph routing** - Vec-indexed dispatch with cached buffers

All automatic. Users get them without changing their training code.

The benchmark suite intentionally *doesn't* enable CUDA Graphs, mixed
precision, or channels-last - those would give flodl an unfair advantage
over standard PyTorch usage. The numbers above are apples-to-apples: same
dtype (fp32), same memory layout (contiguous), same training loop structure.

## Why this matters for distributed training

In single-GPU training, variance is an annoyance. In distributed training,
it's a scaling bottleneck.

Synchronous data-parallel training has an all-reduce barrier at every step.
Every worker computes gradients independently, then they synchronize. The
step takes as long as the slowest worker.

With PyTorch's ±25ms variance (residual_tower), your effective step time
includes the tail of the distribution, not the median. With 8 workers, the
probability of *at least one* worker hitting a slow epoch is high. With 64
workers, it's near-certain.

flodl's ±3.6ms on the same model means the synchronization barrier is tight.
Every worker finishes within a narrow window. Your effective throughput scales
closer to the theoretical linear scaling.

And the deployment story compounds this: a flodl training image doesn't need
Python, pip, or the PyTorch distribution. Less to pull, less to cache, faster
cold starts on spot instances.

## Reproduce it

The entire suite runs in Docker. No local Rust or Python needed:

> **Update (flodl 0.5.1):** `make bench*` became `fdl bench [<preset>]`.
> See [docs/benchmark.md](https://github.com/flodl-labs/flodl/blob/main/docs/benchmark.md#reproduce)
> for the current invocations.

```bash
git clone https://github.com/flodl-labs/flodl.git
cd flodl

# Quick single-round
make bench

# Publication run (10 rounds, locked clocks, 15s warmup)
make bench-publish
```

Per-round JSON with full per-epoch timings is saved in `benchmarks/rounds/`.
The merged report goes to `benchmarks/report.txt`. If you have a different
GPU, run it and see - the absolute numbers will differ but the relative
story should hold.

The [full methodology](https://github.com/flodl-labs/flodl/blob/main/docs/benchmark.md)
documents the protocol, environment, statistical model, and all the
optimizations in detail. Every number is reproducible. Every asymmetry is
accounted for.

## What's next

This benchmark measures standard fp32 training without CUDA Graphs or mixed
precision - the optimizations that would further widen the gap. Future
benchmark updates will add:

- Mixed precision (autocast + GradScaler) comparison
- CUDA Graphs on eligible models
- Channels-last Conv2d
- The FBRL letter model re-run on the RTX 5060 Ti with all optimizations

The suite is designed to grow. Each benchmark is a Rust module with a matching
Python implementation. Adding a new model takes an afternoon.

But the core finding won't change. The variance advantage is structural. It
comes from the language, not the optimizer. And that's the number that matters.

---

*flodl is open source:
[GitHub](https://github.com/flodl-labs/flodl) |
[crates.io](https://crates.io/crates/flodl) |
[docs](https://docs.rs/flodl) |
[benchmark](https://flodl.dev/benchmark)*
