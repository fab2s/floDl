# Benchmark: flodl vs PyTorch

Ten models, ten interleaved rounds, idle machine. Same architectures, same
hyperparameters, same CUDA kernels - the only variable is the framework
overhead.

> This is a living document. The benchmark suite ships with flodl and runs in
> Docker - anyone can reproduce these numbers on their own hardware.

## Results (v0.3.0 - April 2026)

| Model | PyTorch | flodl | Delta | Py σ | Rs σ | Py alloc | Rs alloc | Py rsrvd | Rs rsrvd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer | 3183.0 ms | 2199.8 ms | **-31%** | ±0.8 | ±1.0 | 1395 MB | 1358 MB | 1544 MB | 1534 MB |
| mlp | 291.1 ms | 207.0 ms | **-29%** | ±4.0 | ±1.3 | 440 MB | 401 MB | 466 MB | 418 MB |
| residual_tower | 406.9 ms | 309.7 ms | **-24%** | ±6.0 | ±3.3 | 400 MB | 382 MB | 466 MB | 392 MB |
| feedback_fixed | 275.3 ms | 231.3 ms | **-16%** | ±10.0 | ±6.0 | 64 MB | 63 MB | 90 MB | 88 MB |
| gated_routing | 248.0 ms | 217.3 ms | **-12%** | ±9.7 | ±2.8 | 126 MB | 129 MB | 148 MB | 146 MB |
| iterative_refine | 230.7 ms | 206.0 ms | **-11%** | ±2.2 | ±3.1 | 196 MB | 195 MB | 234 MB | 214 MB |
| gru_seq | 1105.1 ms | 1057.5 ms | **-4%** | ±16.7 | ±25.4 | 523 MB | 526 MB | 566 MB | 558 MB |
| conv_autoenc | 398.2 ms | 395.3 ms | -1% | ±1.1 | ±3.7 | 403 MB | 395 MB | 484 MB | 442 MB |
| lstm_seq | 692.3 ms | 692.3 ms | 0% | ±23.3 | ±15.2 | 677 MB | 677 MB | 846 MB | 856 MB |
| convnet | 1298.0 ms | 1298.2 ms | 0% | ±0.3 | ±0.1 | 1102 MB | 1103 MB | 1420 MB | 1484 MB |

**flodl wins 8 of 10 benchmarks, ties 2, and loses none. The two ties
(convnet, lstm_seq) are compute-bound - both frameworks saturate the GPU,
confirming identical CUDA kernel dispatch.**

### Reading the table

- **Median epoch time** - median of the best run's 20 measured epochs, taken
  across 10 rounds. Lower is faster.
- **σ** - scaled MAD (Median Absolute Deviation x 1.4826) across rounds,
  σ-equivalent for normal data but robust to OS/GC outliers. Lower means
  more predictable. See [Why MAD, not standard deviation?](#why-mad-not-standard-deviation) below.
- **alloc** - peak active tensor bytes (`torch.cuda.max_memory_allocated` /
  `gpu_peak_active_bytes`). This is the memory your tensors actually use.
- **rsrvd** - peak allocator reservation (`torch.cuda.max_memory_reserved` /
  `gpu_peak_reserved_bytes`). This is the memory the caching allocator holds.

### Where flodl wins and why

The pattern is consistent: the speed advantage tracks framework overhead, not
CUDA kernel speed.

**Dispatch-bound models** (transformer -31%, mlp -29%): These models make
many calls to relatively small GPU kernels. PyTorch's path through Python
dispatch, TorchScript, and into C++ adds 3-5us per op. flodl calls
libtorch directly via FFI. When a model chains hundreds of ops per epoch,
that overhead compounds.

**Graph-builder architectures** (residual_tower -24%, feedback_fixed -16%,
gated_routing -12%, iterative_refine -11%): flodl's `Graph::build()`
pre-computes a flat Vec-indexed routing table. Forward dispatch is array
indexing - no HashMap lookups, no dynamic allocation. Gate combination uses
vectorized stack+broadcast+sum (3 kernels regardless of expert count).

**Fused RNN** (gru_seq -4%): Both frameworks call the same cuDNN fused kernel
for GRU sequences. The small gap comes from flodl's cached parameter
tensors on the C++ side, eliminating per-forward parameter collection
and FFI marshalling.

**Compute-bound ties** (convnet 0%, lstm_seq 0%): These models spend
>99% of time inside cuDNN kernels. Framework overhead is invisible. The tie
proves both frameworks dispatch identical CUDA kernels - the speed advantage
appears precisely where framework overhead dominates.

### Variance: what the σ column tells you

Variance reports the run-to-run consistency of each framework. Lower σ means
more predictable epoch times.

On dispatch-bound models, flodl shows 1.5-3.5x tighter variance: gated_routing
(3.5x), mlp (3.1x), convnet (3.0x), residual_tower (1.8x). This comes from
Rust's deterministic execution model: no garbage collector pauses, no
interpreter overhead jitter, deterministic `Drop` order.

On fused-kernel workloads (GRU, LSTM, conv_autoenc), both frameworks show
similar variance because cuDNN dominates execution time. The variance
that remains is inherent to GPU scheduling, not framework overhead.

For distributed training, tight variance means synchronization barriers
spend less time waiting for the slowest worker. On dispatch-bound
architectures, every node finishing within ±3ms instead of ±10ms improves
effective throughput scaling.

The deployment story compounds this: the flodl benchmark Docker image is
26.86 GB vs PyTorch's 38.45 GB - 30% smaller. No Python, no pip, no PyTorch
distribution, just the Rust binary and libtorch. On clusters with cold starts
or shared storage, 12 GB less per node adds up.

### Why MAD, not standard deviation?

Standard deviation is sensitive to outliers. In GPU benchmarking on
shared-driver environments (WSL2, background OS scheduling), occasional
rounds spike by 50-170% due to GC pauses, thermal transients, or CUDA
scheduling stalls. These spikes inflate standard deviation far beyond the
true run-to-run variability.

Scaled MAD (Median Absolute Deviation x 1.4826) is σ-equivalent for
normally distributed data, but treats outlier rounds as what they are:
interference from outside the benchmark, not real framework variance.

Both frameworks benefit: Python's garbage collector fires at unpredictable
intervals, creating 2-3 spiked rounds per benchmark (50-170% above median).
Rust has rarer but occasionally larger spikes from CUDA scheduling stalls.
Standard deviation would exaggerate whichever spike pattern is largest;
MAD reports the steady-state behavior of both frameworks honestly.

To validate the choice: the full per-round JSON data is published in
`benchmarks/rounds/`. Anyone can compute both metrics and see the outliers
for themselves.

## What was measured

### Tier 1 - Standard architectures

These models use standard nn modules (Linear, Conv2d, LSTM, GRU, MultiheadAttention,
ConvTranspose2d, activations, normalization). Both sides use identical PyTorch-equivalent code.

| Model | Architecture | Params | Batch | What it tests |
|---|---|---:|---:|---|
| **mlp** | 5-layer Linear+GELU+LayerNorm (1024->2048->1024) | 16.8M | 256 | Raw matmul + activation throughput |
| **convnet** | 4-layer Conv2d+BN+MaxPool -> classifier head | 4.0M | 128 | Convolution pipeline, cuDNN dispatch |
| **gru_seq** | 50-step GRU -> linear projection | 1.3M | 128 | Fused RNN sequence throughput |
| **transformer** | 4-layer encoder (MHA+FFN+LayerNorm+residual) | ~25M | 32 | Attention throughput, cross-entropy loss |
| **lstm_seq** | 2-layer LSTM -> linear projection | ~2.5M | 128 | Fused LSTM throughput |
| **conv_autoenc** | Conv2d encoder + ConvTranspose2d decoder | ~3.5M | 64 | Transposed convolution, image reconstruction |

### Tier 2 - Graph builder architectures

These models use flodl's `FlowBuilder` (residual connections, gated routing,
loops). The PyTorch equivalents use manual `forward()` implementations with
the same architecture.

| Model | Architecture | Params | Batch | What it tests |
|---|---|---:|---:|---|
| **residual_tower** | 12 residual blocks via `.also()` | 14.7M | 256 | Skip-connection overhead, deep graphs |
| **gated_routing** | 8-expert MoE via `.gate(SoftmaxRouter)` | 2.6M | 256 | Soft routing, vectorized gate combination |
| **iterative_refine** | Encoder -> 8-step `.loop_body().for_n()` -> decoder | 3.2M | 256 | Fixed-iteration loop overhead |
| **feedback_fixed** | Encoder -> 10-step feedback loop -> decoder | 0.8M | 128 | Recurrent loop with smaller model |

All models use Adam optimizer with lr=1e-3, MSE loss (or cross-entropy for
transformer), and random synthetic data. Each epoch runs 50 batches.

## Methodology

### Publication-grade protocol

The benchmark suite is designed for reproducibility and statistical rigor:

1. **Interleaved rounds** - 10 rounds alternate Rust-first/Python-first to
   distribute thermal drift and background noise equally.

2. **GPU clocks** - locked at 3090 MHz (RTX 5060 Ti default applications
   clock). On non-WSL systems, the suite auto-detects and locks the GPU
   clock; on WSL2, clock locking requires the host-side PowerShell script.

3. **Multi-run best-of** - each round runs 4 complete passes (1 warmup + 3
   measured). The best run's median is reported, filtering process-level noise.

4. **Per-epoch measurement** - each measured run collects 20 epoch times with
   3 warmup epochs. The median is used (not mean) to resist outliers.

5. **GPU warmup** - 15 seconds of synthetic GPU load before any measurement
   to stabilize thermal state and clocks.

6. **Idle machine** - system rebooted, non-essential services stopped, display
   off. No user processes competing for resources.

7. **Docker isolation** - both frameworks run in the same container image with
   identical CUDA toolkit, libtorch, and system libraries.

8. **Peak VRAM tracking** - VRAM measured via `gpu_reset_peak_stats()` /
   `gpu_peak_active_bytes()`, matching `torch.cuda.max_memory_allocated()`
   semantics. Not a snapshot - the true high-water mark.

### Statistical model

For N rounds, each with best-of-K runs:

- **Reported median** = median of N best-run medians (one per round)
- **Reported σ** = scaled MAD of N best-run medians (MAD x 1.4826, σ-equivalent
  for normal distributions)

This gives N independent samples (one per round) for robust statistics. Scaled
MAD resists the outlier rounds that inflate stdev on shared-driver environments
(WSL2, background OS scheduling), while remaining directly comparable to
standard deviation for well-behaved data.

### Environment

| Component | Version |
|---|---|
| flodl | 0.3.0 |
| PyTorch | 2.10.0+cu128 |
| Rust | 1.94.0 |
| CUDA | 12.8 (runtime) |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Driver | 595.79 |
| OS | Ubuntu 24.04 (Docker on WSL2) |
| CPU | AMD Ryzen 7 7800X3D |

## Why flodl is faster

The speed advantage comes from eliminating framework overhead, not from
different CUDA kernels. Both frameworks call the same libtorch C++ backend.
The difference is what happens between kernel launches:

### Host-side dispatch elimination

PyTorch's forward pass goes through Python -> TorchScript dispatch -> C++.
flodl calls libtorch directly from Rust via FFI. For models with many small
operations (MLP, transformer, residual tower, routing), the per-op dispatch
overhead dominates - hence the 24-31% improvement.

For compute-bound models (convnet, lstm_seq), both frameworks spend >99% of
time in cuDNN kernels and the dispatch overhead is invisible - hence the
0% delta.

### Fused operations

| Operation | PyTorch | flodl |
|---|---|---|
| Adam step | N*4 kernel launches (per-parameter) | 1 fused multi-tensor kernel (`_fused_adamw_`) |
| Gradient clipping | 2N kernels (norm + scale per param) | 2 kernels total (`_foreach_norm` + `_foreach_mul_`) |
| zero_grad | N zeroing kernels | set_to_none (no kernel) |
| RNN params | Per-forward tensor collection | C++ cached handle (`RnnParams`) |

With N = hundreds of parameters, this adds up every batch.

### Fused RNN kernels

Both frameworks call cuDNN's fused `at::lstm()` / `at::gru()` - a single
kernel for the entire sequence across all layers. flodl additionally caches
the packed parameter tensors on the C++ side (`RnnParams` handle), eliminating
per-forward parameter collection, FFI array marshalling, and `std::vector`
reconstruction. This is the same strategy as PyTorch's
`flatten_parameters()` but without the pointer-validation overhead.

### Pre-computed graph routing

flodl's `Graph::build()` pre-computes a flat Vec-indexed routing table.
Forward pass dispatch is array indexing - no HashMap lookups, no dynamic
allocation. Gate combination uses vectorized stack+broadcast+sum (3 kernels
regardless of expert count).

### Memory predictability

Rust's ownership model means tensor memory is freed deterministically at scope
exit. No GC pauses, no deferred deallocation, no memory pressure spikes. This
contributes to the tighter variance on dispatch-bound architectures where
host-side memory management has time to matter between kernel launches.

## Optimizations since v0.1.1

The first published benchmark (v0.1.1, March 18, 2026) measured 19% faster
than PyTorch on a real FBRL training workload. Since then, significant
optimization work has been done:

| Optimization | What it does | Impact |
|---|---|---|
| **Fused Adam/AdamW** | Single multi-tensor CUDA kernel for the full optimizer step | Eliminates N*4 per-parameter kernel launches |
| **Foreach ops** | 7 batched tensor operations (`foreach_zero_`, `foreach_norm`, `foreach_mul_scalar_`, etc.) | Single kernel for N parameters |
| **Fused gradient clipping** | `clip_grad_norm` via `_foreach_norm` + `_foreach_mul_` | 2 kernels instead of 2N |
| **Fused RNN** | `at::lstm()` / `at::gru()` - single cuDNN kernel for full sequence | Eliminates N*L per-timestep cell dispatch |
| **RNN param caching** | C++ `RnnParams` handle caches packed parameter tensors | Zero per-forward FFI overhead for LSTM/GRU |
| **CUDA Graphs** | Capture/replay kernel sequences, eliminating CPU dispatch overhead | Available for static-shape workloads |
| **Automatic mixed precision** | `AutocastGuard` / `autocast()` + `GradScaler` for fp16/bf16 | Up to 3x on Tensor Core GPUs |
| **Channels-last memory** | NHWC layout for Conv2d on Tensor Core GPUs | 8-35% for convolution-heavy models |
| **Async device transfer** | `pin_memory()` + `copy_(non_blocking)` + `to_device_async()` | Overlaps data transfer with computation |
| **Pre-computed graph routing** | Vec-indexed routing, cached execution buffers | Zero allocation in forward pass |
| **Loop fast-path** | Direct `forward()` call when loop body has no refs | Eliminates HashMap overhead per iteration |
| **Vectorized gate combination** | Stack+broadcast+sum for MoE routing | 3 kernels regardless of expert count |

These optimizations are automatic - users get them without changing their
training code. The benchmark suite does not enable CUDA Graphs, mixed
precision, or channels-last format to ensure a fair apples-to-apples
comparison with standard PyTorch.

## Earlier benchmark: FBRL letter model (v0.1.1)

> **Note:** This benchmark was run with flodl v0.1.1 before the optimizations
> listed above. It measures a different thing - real end-to-end training of a
> non-trivial model, including data loading, checkpointing, and a live
> dashboard.

| Metric | PyTorch 2.5.1 | flodl 0.1.1 | Delta |
|--------|--------------|-------|-------|
| Avg epoch | 49.7s | 40.3s | **-19%** |
| Total | 82m 50s | 67m 10s | **-19%** |
| GPU utilization | ~80% (spiky) | 88-92% (flat) | more stable |
| VRAM (reserved) | 2,805 MB | 2,977 MB | +6%* |

\* VRAM overhead is a fixed cost from CUDA toolkit version difference (cu126
vs cu124) and libtorch's caching allocator. Does not grow with model size.

**Model:** FBRL letter recognition - recurrent attention with GRU controller,
CNN encoder, deconv decoder, 9-component loss stack. 11,440 grayscale 128x128
images, 100 epochs on a GTX 1060 6GB.

**Key finding:** flodl ran a live training dashboard and async gzip
checkpointing - more work per epoch than PyTorch - and was still 19% faster.
The epoch time standard deviation was 0.10s (flodl) vs 0.85s (PyTorch).

Full details, raw data, and reproduction steps at
[fbrl@5c58d71](https://github.com/fab2s/fbrl/tree/5c58d71).

## Reproduce

### Synthetic benchmark suite

The benchmark suite runs entirely in Docker. No local Rust or Python
installation required.

```bash
git clone https://github.com/flodl-labs/flodl.git
cd flodl

# Quick single-round benchmark
fdl bench

# Publication benchmark (10 interleaved rounds, 15s warmup)
fdl bench publish

# CPU-only
fdl bench cpu

# Custom configuration
fdl bench publish --rounds 20 --lock-clocks 2407 --output benchmarks/report.txt
```

`fdl bench --help` lists every available flag - the schema is published by
`benchmarks/run.sh --fdl-schema` and auto-cached on first use.

Each round runs the full Rust suite then the full Python suite (alternating
order). Results are merged across rounds with median-of-medians aggregation.
The final report includes environment metadata, per-model results with σ, and
VRAM tracking.

### What you need

- Docker with NVIDIA Container Toolkit (for GPU benchmarks)
- `make`
- An NVIDIA GPU (any generation - the suite adapts)

The Docker image includes Rust, Python, PyTorch, libtorch, and all
dependencies. First build takes ~15 minutes; subsequent runs reuse the cached
image.

### Raw data

Per-round JSON files are saved in `benchmarks/rounds/` with full per-epoch
timings, VRAM measurements, and model metadata. The merged results and final
report are written to `benchmarks/report.txt`.

### Adding benchmarks

The suite is extensible. Each benchmark is a Rust module in `benchmarks/src/`
with a matching Python implementation in `benchmarks/python/`. See existing
tier1/tier2 benchmarks for the pattern. Both sides use the same harness with
identical measurement methodology.
