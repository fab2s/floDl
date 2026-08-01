# flodl Benchmark Suite

Head-to-head comparison of flodl (Rust) and PyTorch (Python) on identical
model architectures. Both frameworks use the same libtorch C++ backend --
the benchmarks measure framework overhead, not kernel speed.

## Quick start

```bash
fdl bench                        # CUDA benchmarks (all tiers)
fdl bench cpu                    # CPU-only
fdl bench --tier1                # tier 1 only
fdl bench --bench mlp            # single benchmark
```

Everything runs in Docker. No local Rust or Python required. Full option
list: `fdl bench --help` (auto-populated from `run.sh --fdl-schema`).

## Publication mode

```bash
fdl bench publish --lock-clocks 2407
```

The `publish` preset sets `--rounds 10 --warmup-secs 15`; override any
flag on the command line (e.g. `--rounds 20 --output …`).

This enables the full methodology designed for reproducible, publishable
results:

- **Interleaved rounds** -- alternates which framework runs first each round
  (odd: Rust first, even: Python first) to distribute thermal drift and
  background noise equally
- **GPU clock locking** -- pins the GPU to its base clock, eliminating boost
  clock variance (on WSL2, use `bench-publish.ps1` from the Windows host since
  WSL cannot access the driver clock control plane)
- **Extended warmup** -- 15-second GPU matmul burst before any measurement to
  stabilize thermals
- **Multi-round statistics** -- each round's best-run median becomes one sample;
  the final reported median is the median of those samples, with standard
  deviation across rounds

### WSL2 clock locking

WSL2's `nvidia-smi` is a shim that cannot lock GPU clocks. The
`bench-publish.ps1` / `bench-publish.cmd` scripts solve this by locking clocks
from Windows before launching the benchmark inside WSL:

```
benchmarks\bench-publish.cmd                  # auto-detects base clock
benchmarks\bench-publish.cmd -Rounds 20       # more rounds
benchmarks\bench-publish.cmd -Clock 1800      # manual frequency
```

Requires an admin command prompt.

## Measurement methodology

### Per-benchmark structure

Each benchmark consists of multiple **runs**. The first run is a warmup
(discarded). The remaining runs are measured.

Each run:
1. **3 warmup epochs** -- populate caches, trigger cuDNN autotuning
2. **20 measured epochs** -- timed individually with `gpu_synchronize()`
   barriers

The **best run** (lowest median across its 20 epochs) is reported. This
filters process-level noise (GC pauses, scheduling jitter) while preserving
intra-run variance.

### Multi-round aggregation

With `--rounds N`, the entire Rust+Python suite runs N times in alternating
order. For each benchmark:

- Each round contributes one best-run median
- The final median is the **median of N best-run medians**
- Standard deviation is computed across those N samples

This gives N independent measurements that are robust to thermal transients
and framework warm-up effects.

### VRAM measurement

Peak VRAM is reset **after** the warmup run to exclude cuDNN autotuning
workspace and JIT compilation buffers. Two metrics are reported:

- **alloc** -- peak active tensor bytes (`torch.cuda.max_memory_allocated()` /
  `gpu_peak_active_bytes()`)
- **reserved** -- peak allocator pool reservation
  (`torch.cuda.max_memory_reserved()` / `gpu_peak_reserved_bytes()`)

Both frameworks use the same libtorch caching allocator, so the numbers are
directly comparable.

### Statistical reporting

The comparison table reports:

| Column | Meaning |
|--------|---------|
| PyTorch / flodl | Best-run median epoch time (ms) |
| delta | Percentage difference (negative = flodl faster) |
| alloc / reserved | Peak VRAM (MB) |
| sigma | Standard deviation across rounds (multi-round only) |

## Benchmark tiers

### Tier 1 -- Baseline patterns

Pure module + optimizer throughput. These are standard architectures that both
frameworks express naturally.

| Benchmark | Architecture | What it measures |
|-----------|-------------|------------------|
| `mlp` | 4-layer MLP (784 -> 256 -> 128 -> 64 -> 10) | Linear + ReLU + Adam throughput |
| `convnet` | 3x Conv2d + MaxPool + 2x Linear | Conv + pooling + mixed pipeline |
| `gru_seq` | GRUCell unrolled over 50 timesteps | Per-timestep RNN overhead |

### Tier 2 -- Graph builder patterns

These use flodl's graph builder constructs. The PyTorch equivalents are
hand-written `nn.Module` subclasses -- the benchmark measures how much overhead
(if any) the declarative graph engine adds compared to manual Python wiring.

| Benchmark | Architecture | What it measures |
|-----------|-------------|------------------|
| `residual_tower` | 8 residual blocks with LayerNorm | `also()` residual connections |
| `gated_routing` | 3-expert soft gate with SoftmaxRouter | `gate()` mixture-of-experts |
| `iterative_refine` | Fixed 5-iteration refinement loop | `loop_body().for_n()` |
| `feedback_fixed` | 10-iteration loop (no adaptive halt) | Fixed loop overhead isolation |

## Project structure

```
benchmarks/
  src/
    main.rs          -- CLI entry point, tier dispatch
    harness.rs       -- Timing, warmup, VRAM tracking, JSON output
    tier1/           -- Rust benchmarks (mlp, convnet, gru_seq)
    tier2/           -- Rust benchmarks (residual_tower, gated_routing, ...)
  python/
    run_all.py       -- CLI entry point (mirrors main.rs)
    harness.py       -- Python harness (mirrors harness.rs)
    compare.py       -- Side-by-side comparison table
    merge_rounds.py  -- Multi-round aggregation
    tier1/           -- Python benchmarks
    tier2/           -- Python benchmarks
  run.sh             -- Orchestrator: build, warmup, interleave, merge, compare
  bench-publish.ps1  -- Windows host clock locking + WSL2 dispatch
  bench-publish.cmd  -- CMD wrapper for bench-publish.ps1
  rounds/            -- Per-round JSON data (generated)
  report.txt         -- Final report with metadata header (generated)
  IMPROVEMENTS.md    -- Known issues and planned improvements
```

## Harness parity

The Rust and Python harnesses are structurally identical:

- Same run/epoch/warmup structure
- Same best-of-N selection logic
- Same VRAM reset timing (after warmup run)
- Same JSON output schema
- Same `gpu_synchronize()` barriers at measurement boundaries
- Same cuDNN benchmark mode (`torch.backends.cudnn.benchmark = True` /
  `set_cudnn_benchmark(true)`)

This ensures the comparison measures framework overhead, not harness
differences.

## Adding a benchmark

1. Create `src/tier{1,2}/my_bench.rs` -- implement a function
   `pub fn run(device: Device) -> flodl::Result<BenchResult>` that builds
   the model, generates data, and calls `harness::run_benchmark()`
2. Create `python/tier{1,2}/my_bench.py` -- mirror the architecture exactly
3. Register in `src/tier{1,2}/mod.rs` and `python/tier{1,2}/__init__.py`
4. Add to the dispatch table in `src/main.rs` and `python/run_all.py`

The key rule: **identical architectures**. Same layer sizes, same optimizer,
same learning rate, same batch count. The only variable is the framework.
