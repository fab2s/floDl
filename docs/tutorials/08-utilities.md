# Training Utilities

This tutorial covers the utilities that sit around the training loop:
gradient clipping, checkpointing, weight initialization, LR scheduling,
reproducibility, trend-based training control, verbosity-gated logging,
peak-VRAM profiling, and CUDA Graphs. [Training](04-training.md) covers the
loop itself and points here for each of these.

> **Prerequisites**: [Training](04-training.md) introduces the
> backward/step loop. [The Graph Builder](05-graph-builder.md)
> introduces tags.

## Gradient clipping

Deep models - especially those with loops or long chains - can suffer
from exploding gradients. floDl provides two clipping strategies. Both
are called between `backward()` and `optimizer.step()`.

### clip_grad_norm

Scales all parameter gradients so that the total L2 norm does not
exceed `max_norm`. Returns the original norm before clipping.

```rust
loss.backward()?;
let orig_norm = clip_grad_norm(&params, 1.0)?;
optimizer.step()?;
```

### clip_grad_value

Clamps each individual gradient element to `[-max_val, max_val]`.

```rust
loss.backward()?;
clip_grad_value(&params, 0.5)?;
optimizer.step()?;
```

Use `clip_grad_norm` as the default - it preserves gradient direction.

Both are batched under the hood: `clip_grad_norm` uses `_foreach_norm` +
`_foreach_mul_`, so it costs two kernels regardless of how many parameters
the model has, instead of the `2N` a naive per-parameter loop would launch.
That matters most on CUDA, where launch overhead dominates for many small
per-parameter operations.

## Checkpoints

Save and restore model parameters with a compact binary format.

### Saving and loading

```rust
// Save - parameters + buffers + structural hash, one call
model.save_checkpoint("/tmp/model.fdl")?;

// Load - validates architecture, returns LoadReport
let report = model.load_checkpoint("/tmp/model.fdl")?;
```

`load_checkpoint` validates names and shapes for both parameters and buffers.
The returned `LoadReport` tells you exactly which entries were loaded,
skipped, or missing. Append `.gz` for gzip compression.

For custom I/O targets (network, in-memory buffer), use the lower-level API:

```rust
use flodl::{save_checkpoint, load_checkpoint};

let named = model.named_parameters();
let buffers = model.named_buffers();
let hash = Some(model.structural_hash());
save_checkpoint(&mut writer, &named, &buffers, hash)?;
let report = load_checkpoint(&mut reader, &named, &buffers, hash)?;
```

### Details

- Parameters and buffers store their native dtype - float16 params stay f16 on disk.
- Named checkpoints match entries by qualified name and validate shapes.
- The `io::Write` / `io::Read` variants (`save_checkpoint`, `load_checkpoint`)
  work with any destination: files, buffers, network connections.

### Partial loading (transfer learning)

Named checkpoints match by qualified name, which allows loading a subset
of parameters from a different model:

```rust
use flodl::*;

// Save with qualified names from a Graph
model.save_checkpoint("/tmp/model.fdl")?;

// Load into a different model - only matching names transfer
// Use lower-level API with None hash since architectures differ
let new_named = new_model.named_parameters();
let new_buffers = new_model.named_buffers();
let report = load_checkpoint_file("/tmp/model.fdl", &new_named, &new_buffers, None)?;

println!("loaded:  {:?}", report.loaded);   // matched and loaded
println!("skipped: {:?}", report.skipped);  // in checkpoint, not in model
println!("missing: {:?}", report.missing);  // in model, not in checkpoint
```

Names are `"prefix/param_name"` where the prefix is the tag name if
tagged (`"encoder/weight"`) or the node ID if not (`"linear_1/weight"`).
Shape mismatches on a matched name produce an error (not a silent skip).

### Freezing transferred parameters

After partial loading, freeze the transferred params and train only the
new ones:

```rust
for (name, param) in &new_named {
    if report.loaded.contains(name) {
        param.freeze()?;
    }
}

// Only unfrozen params get gradients - optimizer skips the rest
let fresh: Vec<Parameter> = new_named.iter()
    .filter(|(_, p)| !p.is_frozen())
    .map(|(_, p)| p.clone())
    .collect();
let mut opt = Adam::new(&fresh, 1e-3);
```

### Periodic checkpoints during training

```rust
for epoch in 0..num_epochs {
    // ... training loop ...

    if (epoch + 1) % 10 == 0 {
        let path = format!("/tmp/checkpoint_epoch_{}.fdl", epoch + 1);
        model.save_checkpoint(&path)?;
    }
}
```

### Background checkpoints with CpuWorker

During GPU training, saving a checkpoint blocks the GPU. Use `snapshot_cpu()`
to copy model state to CPU, then save it on a background thread:

```rust
use flodl::{CpuWorker, ModelSnapshot};

let worker = CpuWorker::new();

for epoch in 0..num_epochs {
    // ... training loop ...

    if (epoch + 1) % 10 == 0 {
        let snap = model.snapshot_cpu()?;
        let path = format!("/tmp/checkpoint_epoch_{}.fdl.gz", epoch + 1);

        // Skip if previous save is still running
        if worker.is_idle() {
            worker.submit(move || {
                snap.save_file(&path).unwrap();
            });
        }
    }
}
// worker.finish() is called on Drop - waits for queued saves
```

`snapshot_cpu()` copies all parameters (detached from autograd) and buffers to
CPU. The resulting `ModelSnapshot` is `Send`, so it can safely cross thread
boundaries. The checkpoint format is the same `.fdl` format used by
`save_checkpoint` - files saved with `save_file` can be loaded by
`load_checkpoint_file`.

## Weight initialization

floDl modules use sensible defaults - `Linear` initializes weights with
Kaiming uniform (suitable for ReLU) and bias with uniform. But you can
override this when needed.

### Built-in initializers

All initializers are available at the crate root (`use flodl::*`).

| Function | Distribution | Best for |
|----------|-------------|----------|
| `kaiming_uniform(shape, fan_in, a, device)` | U(-bound, bound) | ReLU activations (default for Linear) |
| `kaiming_normal(shape, fan_in, a, device)` | N(0, std) | ReLU activations |
| `xavier_uniform(shape, fan_in, fan_out, device)` | U(-bound, bound) | Sigmoid / Tanh |
| `xavier_normal(shape, fan_in, fan_out, device)` | N(0, std) | Sigmoid / Tanh |
| `uniform(shape, low, high, device)` | U(low, high) | General purpose |
| `normal(shape, mean, std, device)` | N(mean, std) | General purpose |
| `orthogonal(shape, gain, device)` | Orthogonal (Gram-Schmidt) | RNNs, preserves gradient norms |
| `trunc_normal(shape, mean, std, a, b, device)` | Truncated normal | Vision Transformers (ViT) |
| `uniform_bias(fan_in, shape, device)` | U(-1/sqrt(fan_in), 1/sqrt(fan_in)) | Bias terms |

### Custom initialization

Replace parameter data after constructing the module:

```rust
let layer = Linear::new(128, 64)?;

// Re-initialize weight with Xavier normal.
let w = xavier_normal(&[64, 128], 128, 64, Device::CPU)?;
layer.parameters()[0].set_data(&w);
```

## Reproducibility

### Seeding libtorch

`manual_seed` sets the global seed for all libtorch random operations:

```rust
flodl::manual_seed(42);
```

This controls `Tensor::rand`, `Tensor::randn`, dropout masks, and weight
initialization (kaiming, xavier). Call it before model creation.

On CUDA builds, `manual_seed` seeds both CPU and GPU. To re-seed CUDA
independently:

```rust
flodl::cuda_manual_seed_all(42);
```

### CPU-side RNG

For data loading, shuffling, and augmentation, use `Rng` - a lightweight
wrapper around SmallRng (Xoshiro256++):

```rust
use flodl::Rng;

let mut rng = Rng::seed(42);       // deterministic from seed
let mut rng = Rng::from_entropy(); // system-seeded

rng.usize(100)          // uniform [0, 100)
rng.f32()               // uniform [0, 1)
rng.f64()               // uniform [0, 1)
rng.shuffle(&mut data)  // Fisher-Yates
rng.bernoulli(0.5)      // true with probability p
rng.range(-5, 5)        // integer [low, high)
rng.normal(0.0, 1.0)    // Gaussian sample
```

`Rng` is `Clone` - clone it to fork independent streams from the same state.

### Full reproducibility setup

Seed both sides, and build the model *after* seeding so weight
initialization is covered too:

```rust
fn main() -> Result<()> {
    flodl::manual_seed(42);        // libtorch: rand, randn, dropout, weight init
    let mut rng = Rng::seed(42);   // CPU side: shuffling, augmentation

    let model = build_model()?;    // built after seeding - weight init uses it

    // Shuffle training order deterministically
    let mut indices: Vec<usize> = (0..dataset_len).collect();
    rng.shuffle(&mut indices);
    // ...
    Ok(())
}
```

The split is the whole story: `manual_seed` covers every libtorch operation
(tensor creation, dropout masks, weight init), `Rng` covers everything else
(data order, augmentation parameters). Seed both and a run is reproducible.

## LR Scheduling

Schedulers are pure LR calculators, decoupled from the optimizer. You call
`.lr(step)` and set the optimizer's LR yourself - no hidden optimizer coupling.

```rust
use flodl::*;

// Step decay: multiply by gamma every step_size steps
let scheduler = StepDecay::new(0.01, 30, 0.1);  // base_lr, step_size, gamma

// Cosine annealing
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);  // base_lr, min_lr, total_steps

// Exponential decay: lr = base_lr * gamma^step
let scheduler = ExponentialLR::new(0.001, 0.95);  // base_lr, gamma

// Multi-step decay: drop lr at specific milestones
let scheduler = MultiStepLR::new(0.001, &[30, 60, 90], 0.1);  // base_lr, milestones, gamma

// One-cycle: warmup then cosine decay (super-convergence)
let scheduler = OneCycleLR::new(0.01, 1000);  // max_lr, total_steps (30% warmup)
let scheduler = OneCycleLR::with_warmup_frac(0.01, 1000, 0.2);  // custom warmup fraction

// Cyclic LR: triangle wave between base and max
let scheduler = CyclicLR::new(1e-4, 1e-2, 500);  // base_lr, max_lr, step_size (symmetric)
let scheduler = CyclicLR::asymmetric(1e-4, 1e-2, 400, 600);  // different up/down phases

// Warmup wrapper (composes with any scheduler)
let inner = CosineScheduler::new(0.001, 1e-6, 100);
let scheduler = WarmupScheduler::new(inner, 0.001, 10);  // inner, target_lr, warmup_steps

// Reduce on plateau (reactive, driven by metrics)
let mut scheduler = PlateauScheduler::new(0.001, 5, 0.1, 1e-6);  // base_lr, patience, factor, min_lr
```

Step-based schedulers implement the `Scheduler` trait:

```rust
pub trait Scheduler {
    fn lr(&self, step: usize) -> f64;
}
```

Because nothing is hidden, wiring one into a loop is explicit - ask for the
step's LR, hand it to the optimizer:

```rust
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);  // base_lr, min_lr, total_steps

for step in 0..100 {
    let lr = scheduler.lr(step);
    optimizer.set_lr(lr);
    // ... forward, backward, optimizer.step()
}
```

`PlateauScheduler` is reactive (driven by metrics, not step count) and uses
`observe(metric)` / `lr()` instead.

## Trend-based training control

After collecting metrics and flushing epoch summaries (see
[Training - Observing Training](04-training.md#observing-training)),
query trends to make training decisions.

### Trend queries

Each `flush` adds one data point (the epoch mean) to the tag's history.
`trend` returns a queryable view:

```rust
let trend = g.trend("loss");
trend.latest();              // most recent epoch value
trend.mean();                // mean across all flushed epochs
trend.slope(5);              // OLS slope over last 5 epochs
trend.improving(5);          // is slope negative? (loss decreasing)
trend.stalled(5, 1e-4);     // is |slope| below tolerance?
trend.converged(5, 1e-5);   // is variance below tolerance?
```

### LR decay on plateau

```rust
if g.trend("loss").stalled(10, 1e-4) {
    // reduce learning rate
}
```

### Early stopping

```rust
if g.trend("loss").converged(5, 1e-5) {
    break;
}
```

### Recording external metrics

Losses computed outside the graph can be injected with `record`:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = g.forward(&input)?;
        let loss = cross_entropy_loss(&pred, &target)?;

        g.collect(&["hidden"])?;              // from graph tag
        g.record_scalar("loss", loss.item()?);  // external scalar
    }
    g.flush(&["hidden", "loss"]);
}
```

### Trend groups

When using `tag_group` with `split`, `trends` expands the group for
aggregate queries:

```rust
let g = FlowBuilder::from(encoder)
    .split(modules![head_a, head_b, head_c]).tag_group("head")
    .merge(MergeOp::Mean)
    .build()?;

// After training with collect/flush...
let tg = g.trends(&["head"]);  // expands to head_0, head_1, head_2
if tg.all_improving(5) {
    println!("all heads improving");
}
println!("mean slope: {:.4}", tg.mean_slope(5));
```

### ETA and timing

```rust
for epoch in 0..total_epochs {
    // ... training ...
    g.flush(&["loss"]);

    println!(
        "epoch {}  loss={:.4}  ETA {}",
        epoch + 1,
        g.trend("loss").latest(),
        format_duration(g.eta(total_epochs)),
    );
}
```

### Training curves

Export metrics as self-contained HTML or CSV:

```rust
g.plot_html("training.html", &["loss", "head"])?;
g.export_trends("metrics.csv", &["loss"])?;
g.write_log("training.log", total_epochs, &["loss"])?;
```

## Verbosity-gated logging

floDl ships a tiny logging system in `flodl::log`. Five levels, four
macros, one global atomic - no log targets, no module filtering, no
external dependencies. Designed so library code can chatter freely at
higher verbosities without forcing every user to wire up a logger.

### Levels

| Level | When | Where it goes |
|------|------|----------------|
| `Quiet` (0) | `--quiet` - errors only via `eprintln!` | (everything else suppressed) |
| `Normal` (1) | Default - epoch summaries, progress, milestones | stdout |
| `Verbose` (2) | `-v` - DDP sync, cadence changes, data loading detail | stdout |
| `Debug` (3) | `-vv` - per-batch timing, internal loops | **stderr** (unbuffered) |
| `Trace` (4) | `-vvv` - extreme granularity | **stderr** (unbuffered) |

Debug and Trace use stderr deliberately so they stay visible in Docker
non-TTY environments where stdout is block-buffered.

### Macros

```rust
// Default level is Normal - suppressed by --quiet, shown otherwise
flodl::msg!("epoch {}: loss={:.4}", epoch, loss);

// Explicit level via @ prefix
flodl::msg!(@Verbose, "AllReduce: {:.1}ms", elapsed);
flodl::msg!(@Debug,   "per-batch: {}ms", batch_ms);
flodl::msg!(@Trace,   "{:?}", tensor);

// Or use the level-named shortcuts
flodl::verbose!("AllReduce: {:.1}ms", elapsed);
flodl::debug!  ("per-batch: {}ms", batch_ms);
flodl::trace!  ("{:?}", tensor);
```

All four macros are zero-cost when the level is not enabled - the format
arguments are not evaluated.

### Configuration

Three ways to set the level, in priority order:

```bash
# 1. CLI (sets FLODL_VERBOSITY for child processes too, including Docker)
fdl -v   ddp-bench quick      # Verbose
fdl -vv  cuda-test            # Debug
fdl -vvv shell                # Trace
fdl --quiet test              # Errors only
```

```bash
# 2. Environment variable (no code, no CLI)
FLODL_VERBOSITY=verbose cargo run
FLODL_VERBOSITY=2       cargo run    # same - accepts integers 0..4
```

```rust
// 3. Programmatic - overrides the env var
use flodl::{Verbosity, set_verbosity};
set_verbosity(Verbosity::Debug);
```

The level is read once on first access, then cached in a single global
atomic - checking enablement on a hot loop is one relaxed atomic load.

### Recipe: gating expensive diagnostics

```rust
use flodl::log;

if log::enabled(log::Verbosity::Debug) {
    let stats = collect_expensive_stats();   // skipped at Normal level
    flodl::debug!("stats: {:?}", stats);
}
```

### Recipe: per-rank tracing in DDP

```rust
let rank = world.rank();
flodl::msg!(@Verbose, "rank {} | epoch {} done in {:.0}ms", rank, epoch, ms);
flodl::trace!("rank {} | grad norms: {:?}", rank, grad_norms);
```

Set `FLODL_VERBOSITY=trace` once at launch and every rank starts emitting
trace output without code changes.

## Peak VRAM profiling

When optimizing GPU memory, use the CUDA memory stats API to measure
actual allocation peaks. Reset counters before the region of interest,
then read the high-water marks:

```rust
cuda_empty_cache();
cuda_reset_peak_stats();

// ... training step or inference ...

let peak_alloc = cuda_peak_active_bytes()?;  // bytes used by tensors
let peak_reserved = cuda_peak_reserved_bytes()?;  // bytes held by allocator
println!("Peak alloc: {:.0} MB", peak_alloc as f64 / 1048576.0);
println!("Peak reserved: {:.0} MB", peak_reserved as f64 / 1048576.0);
```

These match `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()` semantics. `peak_alloc` is the memory
actively backing tensors; `peak_reserved` includes free blocks held by
the caching allocator.

## CUDA Graphs

For models with fixed tensor shapes, CUDA graph capture replays an
entire forward/backward/step sequence as a single GPU operation,
eliminating per-kernel launch overhead.

```rust
use flodl::{CudaGraph, cuda_graph_capture, CaptureMode};

// Static tensors (reused across replays via copy_)
let static_input = Tensor::zeros(&[batch, dim], cuda_opts)?;
let static_target = Tensor::zeros(&[batch, dim], cuda_opts)?;

// Capture training step
let graph = cuda_graph_capture(3, None, || {
    let inp = Variable::new(static_input.clone(), false);
    let tgt = Variable::new(static_target.clone(), false);
    let pred = model.forward(&inp)?;
    let loss = mse_loss(&pred, &tgt)?;
    optimizer.zero_grad();
    loss.backward()?;
    optimizer.step()
})?;

// Training loop: copy data, replay captured graph
for (x, y) in &batches {
    static_input.copy_(x, true)?;   // non_blocking
    static_target.copy_(y, true)?;
    graph.replay()?;
}
```

Expect 2-5x speedup for models with many small kernels (RNNs, GRUs).
All tensors involved in the captured region must have fixed shapes -
dynamic shapes require a new capture. Tests that use CUDA graphs must
run single-threaded (`fdl cuda-test-serial`).

## Putting it together

A complete training script using clipping, checkpoints, trends, and
scheduling:

```rust
// Build model with tagged sections.
let g = FlowBuilder::from(Linear::new(4, 64)?).tag("encoder")
    .through(GELU)
    .through(Linear::new(64, 64)?).tag("body")
    .through(GELU)
    .through(Linear::new(64, 2)?).tag("head")
    .build()?;

// Set up training.
let params = g.parameters();
let optimizer = Adam::new(&params, 0.001);
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);
g.train();

for epoch in 0..100 {
    for (input, target) in &batches {
        let input = Variable::new(input.clone(), true);
        let target = Variable::new(target.clone(), false);

        optimizer.zero_grad();
        let output = g.forward(&input)?;
        let loss = mse_loss(&output, &target)?;
        loss.backward()?;

        clip_grad_norm(&params, 1.0)?;
        optimizer.step()?;

        g.collect(&["head"])?;
        g.record_scalar("loss", loss.item()?);
    }
    g.flush(&["head", "loss"]);
    g.end_epoch();
    optimizer.set_lr(scheduler.lr(epoch));

    // Early stop when converged.
    if g.trend("loss").converged(5, 1e-5) {
        break;
    }
}

// Save.
g.eval();
g.save_checkpoint("/tmp/model.fdl")?;
```

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Visualization](07-visualization.md) | Next: [Training Monitor](09-monitor.md)
