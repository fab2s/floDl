---
title: "Ten models later, the answer hasn't changed"
subtitle: "v0.2.2 benchmarks: fused RNN, honest variance, and still zero regressions against PyTorch"
date: 2026-03-31
description: "flodl v0.2.2 vs PyTorch 2.10.0 on 10 models. Up to 31% faster, wins 8 of 10, zero regressions. New: fused RNN kernels, scaled MAD variance, and a 30% smaller Docker image."
---

The [first benchmark](/blog/benchmarks) measured seven models on
flodl v0.1.3. That was two weeks and a lot of optimization ago. This update
adds three models (transformer, lstm_seq, conv_autoenc), lands fused RNN
kernels with C++-side parameter caching, and switches the variance metric
to something more honest.

## Results (v0.2.2 vs PyTorch 2.10.0)

| Model | PyTorch | flodl | Delta | Py σ | Rs σ |
|---|---:|---:|---:|---:|---:|
| transformer | 3183.0 ms | 2199.8 ms | **-31%** | +-0.8 | +-1.0 |
| mlp | 291.1 ms | 207.0 ms | **-29%** | +-4.0 | +-1.3 |
| residual_tower | 406.9 ms | 309.7 ms | **-24%** | +-6.0 | +-3.3 |
| feedback_fixed | 275.3 ms | 231.3 ms | **-16%** | +-10.0 | +-6.0 |
| gated_routing | 248.0 ms | 217.3 ms | **-12%** | +-9.7 | +-2.8 |
| iterative_refine | 230.7 ms | 206.0 ms | **-11%** | +-2.2 | +-3.1 |
| gru_seq | 1105.1 ms | 1057.5 ms | **-4%** | +-16.7 | +-25.4 |
| conv_autoenc | 398.2 ms | 395.3 ms | -1% | +-1.1 | +-3.7 |
| lstm_seq | 692.3 ms | 692.3 ms | 0% | +-23.3 | +-15.2 |
| convnet | 1298.0 ms | 1298.2 ms | 0% | +-0.3 | +-0.1 |

Eight wins, two ties, zero regressions. Same story as v0.1.3, but now with
the transformer headline at -31% and three more architectures confirming
the pattern.

## What's new

### Transformer at -31%

The biggest model in the suite (~25M params, 4-layer encoder with
MultiheadAttention + FFN + LayerNorm + residual, cross-entropy loss) shows
the largest speed advantage. This is dispatch-bound: hundreds of small ops
per forward pass, each going through Python -> TorchScript -> C++ in
PyTorch, versus direct FFI in flodl. The per-op overhead compounds across
attention heads, feed-forward layers, and residual connections.

Both frameworks produce identical attention weights - same cuDNN kernels,
same numerics. The 31% gap is pure framework overhead.

### Fused RNN kernels

LSTM and GRU now call cuDNN's fused sequence kernels (`at::lstm()` /
`at::gru()`) - a single kernel for the entire sequence across all layers,
replacing per-timestep cell dispatch. On top of that, flodl caches the
packed parameter tensors on the C++ side behind an opaque handle
(`RnnParams`). After the first forward call, subsequent calls pass a
single pointer to the pre-built parameter vector, eliminating per-forward
parameter collection, FFI array marshalling, and `std::vector`
reconstruction.

The result: lstm_seq matches PyTorch exactly (692.3ms vs 692.3ms), and
gru_seq edges ahead by 4%. Both are compute-bound - cuDNN does the real
work - so the tie confirms parity in the underlying kernel dispatch.

### Two new ties prove the architecture

convnet at 0% was already a proof that both frameworks dispatch identical
CUDA kernels. lstm_seq at 0% extends that proof to fused RNN. When the
GPU dominates, flodl converges to the same number. The speed advantage
appears precisely where framework overhead has room to matter.

## Honest variance: why we switched to MAD

The [v0.1.3 benchmarks](/blog/benchmarks) reported σ as standard
deviation. That was correct but misleading. Here's what I found when I dug
into the raw per-round data.

### The outlier problem

Standard deviation treats every data point equally. In GPU benchmarking,
that's a problem. Python's garbage collector fires at unpredictable
intervals, creating occasional 50-170% timing spikes:

- **gated_routing** rounds 2 and 8: PyTorch spiked from ~248ms to ~400ms.
  Two GC pauses in 10 rounds. Standard deviation: +-64.8ms. But 8 of 10
  rounds were within +-7ms.
- **feedback_fixed** round 6: PyTorch spiked from ~275ms to 447ms. One GC
  pause. Standard deviation: +-54.9ms. The other 9 rounds: +-10ms.

Rust has the same problem, differently shaped: rarer but sometimes larger
spikes from CUDA scheduling stalls. gru_seq round 2 spiked from ~1050ms
to 1447ms (+390ms). Standard deviation: +-124.7ms. The other 9 rounds:
+-25ms.

### Scaled MAD

Scaled MAD (Median Absolute Deviation x 1.4826) is σ-equivalent for
normally distributed data but treats these spikes as what they are:
interference from outside the benchmark, not real framework variance.

| metric | Py total σ | Rs total σ | who looks better? |
|---|---|---|---|
| **stddev** | 186 | 210 | PyTorch |
| **MAD** | 74 | 62 | flodl |

With stddev, flodl looks noisier (because of that one gru_seq spike).
With MAD, flodl looks tighter (because its steady-state variance is
actually lower on most models). Neither framing is more "favorable" -
MAD is simply more accurate about what each framework does when the OS
isn't interfering.

The full per-round JSON data is in `benchmarks/rounds/`. Anyone can
compute both metrics and inspect the outliers directly.

### Where flodl is tighter, and where it isn't

With MAD, the variance picture is honest and model-dependent:

- **Dispatch-bound models**: flodl is 1.5-3.5x tighter (mlp, gated_routing,
  convnet, residual_tower, feedback_fixed, lstm_seq). Rust's deterministic
  `Drop` and zero-GC execution model eliminates host-side jitter.
- **Fused-kernel workloads**: both frameworks show similar variance (gru_seq,
  conv_autoenc, transformer, iterative_refine). cuDNN dominates execution
  time; the variance that remains is inherent to GPU scheduling.

This is more nuanced than the "3-20x tighter on every model" from v0.1.3.
It's also more true.

## The deployment angle

One number that doesn't appear in the timing table: Docker image size.

| Image | Size |
|---|---|
| PyTorch benchmark | 38.45 GB |
| flodl benchmark | 26.86 GB |

That's 30% smaller. No Python, no pip, no PyTorch distribution - just the
Rust binary and libtorch.

On spot instances with cold starts, image pull time is real wall-clock cost.
On clusters with shared storage, it's 12 GB less per node. For distributed
training where you're spinning up dozens of workers, the deployment story
compounds with the per-epoch speed advantage.

## What changed since v0.1.3

| Optimization | Impact |
|---|---|
| **Fused RNN** (`at::lstm()` / `at::gru()`) | Single cuDNN kernel for full sequence |
| **RNN param caching** (C++ `RnnParams` handle) | Zero per-forward FFI overhead |
| **flatten_parameters** | Eliminates cuDNN contiguous-weight warning |
| **PyTorch parity** (v0.2.2) | 30+ modules, 15 losses, 7 optimizers, 769 tests |
| **Scaled MAD variance** | Honest σ resistant to GC/scheduling outliers |
| **PyTorch 2.10.0+cu128** | Updated baseline (was 2.6.0+cu126) |

## Reproduce

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

Same Docker setup, same methodology. The
[full benchmark report](https://github.com/flodl-labs/flodl/blob/main/docs/benchmark.md)
documents the protocol, environment, and statistical model in detail.

## Where this goes

The benchmark suite doesn't use CUDA Graphs, mixed precision, or
channels-last - features that would further widen the gap. Those are fair
game for a future comparison.

But the core result is structural. Ten models, two framework versions, and
the answer hasn't changed: flodl matches or beats PyTorch on every
architecture. The speed advantage comes from what Rust eliminates between
GPU kernels. The ties come from workloads where there's nothing left to
eliminate.

Zero regressions. That's the number.

---

*flodl is open source:
[GitHub](https://github.com/flodl-labs/flodl) |
[crates.io](https://crates.io/crates/flodl) |
[docs](https://docs.rs/flodl) |
[benchmark](/benchmark)*
