# Resident Training

A design for device-resident data loading that eliminates transfer overhead
by keeping the entire dataset on the compute device alongside the model.

Combined with CUDA Graphs, this enables **zero-dispatch training** - the CPU
does nothing during an epoch except launch pre-recorded GPU operations.

**Status:** Device-resident, streaming, and distributed `DataLoader` modes
ship today, including DDP-aware per-device backend selection. Note: the
`ResidentLoader<const N>` const-generic surface described below is the
*originating design*, not the shipped type — the live helper is an internal
`pub(crate) struct ResidentLoader` (non-generic) behind the `DataLoader`
modes, not a public API. What remains is the zero-dispatch layer: full
train-step CUDA Graph capture (forward + backward + optimizer as one replay)
and double-buffered static I/O tensors for the Graph path. See
[Tutorial 13: Data Loading](../tutorials/13-data-loading.md) for the current
data pipeline.

---

## Motivation

The standard deep learning training loop moves data from CPU to GPU every batch:

```
CPU memory → pin_memory → async DMA → GPU kernel → backward → optimizer
                ↑
        this is the bottleneck for small models
```

For large models (LLMs, diffusion), the GPU compute dominates and transfer
overhead is noise. But for **small, focused models** - exactly the kind FBRL
produces - the transfer overhead is a significant fraction of total batch time.

### Why this matters now

Three things converge:

1. **VRAM is growing fast.** 16 GB today, 24-48 GB mid-range next generation.
   A letter model with its training data fits trivially. Word-level sub-models
   are close.

2. **FBRL decomposes by design.** The progressive composition strategy produces
   small, independently trained models. Each piece has fewer parameters, less
   optimizer state, and works with a subset of the full dataset. The same
   decomposition that makes the problem learnable also makes it fit in VRAM.

3. **flodl already has the primitives.** `to_device()`, `narrow()`,
   `index_select()`, `copy_()`, CUDA Graphs, foreach ops - everything needed
   for GPU-resident iteration exists. What's missing is the thin orchestration
   layer that makes it a one-liner.

### The compound effect with CUDA Graphs

VRAM-resident data alone eliminates transfer overhead. But combined with CUDA
Graphs, the entire training step (forward + backward + optimizer) becomes a
single pre-recorded GPU operation:

```
Standard:     CPU dispatches ~100s of kernels per batch, waits for each
CUDA Graphs:  CPU replays 1 recorded graph per batch, but still transfers data
VRAM + Graphs: CPU does copy_(GPU→GPU) + replay() -- two calls, both instant
```

For models with many small kernels (typical in FBRL's graph-based models), this
is the difference between CPU-bound and GPU-bound training. The CPU becomes
irrelevant.

### Driving use case: FBRL progressive training

```
Step 1: Train letter model on isolated letter data
        Dataset: ~50K samples × 32×32 pixels = ~200 MB
        Model: ~90K params = ~350 KB
        Optimizer (Adam): ~700 KB
        Total: ~201 MB -- fits in 4 GB VRAM with room to spare

Step 2: Train subscan on letter-in-word data
        Dataset: ~20K samples × 64×64 crops = ~320 MB
        Model: ~200K params = ~800 KB
        Total: ~322 MB -- trivially fits

Step 3: Train word model (meta scan only, rest frozen)
        Dataset: ~10K samples × 128×64 words = ~320 MB
        Model: ~500K params = ~2 MB
        Total: ~324 MB -- fits easily
```

Each FBRL training step produces a small, focused dataset. Even with generous
margins for activation memory and CUDA context overhead (~400 MB), these fit
in 16 GB with room for a desktop environment and a browser.

As VRAM scales to 24-48 GB, larger datasets and deeper compositions qualify
automatically - no code changes needed.

---

## Design principles

### 1. Thin layer, not a framework

`ResidentLoader` is a helper, not a DataLoader abstraction. It holds tensors on a
device and yields batches. No preprocessing pipelines, no augmentation hooks,
no worker threads. Those belong in user code before data enters the loader.

The pattern should feel like a natural extension of what users already do with
`Vec<(Tensor, Tensor)>`, not a paradigm shift.

### 2. Zero-copy when possible

Unshuffled iteration uses `narrow()` - a view into the existing GPU tensor,
no allocation. Shuffled iteration uses per-batch `index_select()` - allocates
only batch-sized tensors, not full dataset copies.

### 3. CUDA Graph compatibility as a first-class concern

`drop_last` mode ensures all batches have identical shapes, which is required
for CUDA Graph replay. The API is designed so that the CUDA Graph capture
pattern works without special casing.

### 4. Fail early, not during training

VRAM budget estimation before loading catches out-of-memory before training
starts. Better to error with a clear "dataset needs 2.1 GB but only 1.8 GB
available" than to OOM mid-epoch.

### 5. Device-agnostic

`ResidentLoader` works on any device - CPU, CUDA, and future backends (ROCm,
MPS, XPU). On CPU it's a fast batch iterator with zero transfer overhead. On
GPU it eliminates host-device transfers entirely. The budget check reports
available memory for the target device; the iteration pattern is universal.

---

## Design

### 1. ResidentLoader

The core data structure. Holds N tensors on a device, iterates in batches.

```rust
pub struct ResidentLoader<const N: usize> {
    /// Data tensors, all on the same device, all with the same dim-0 size.
    /// N is the dataset arity (number of tensors per sample).
    /// Always known at compile time: 2 for (input, target), 3 for (image, label, mask), etc.
    tensors: [Tensor; N],

    /// Samples per batch.
    batch_size: usize,

    /// Total samples (dim 0 of each tensor).
    num_samples: usize,

    /// Drop the last batch if smaller than batch_size.
    /// Required for CUDA Graph replay (fixed shapes).
    drop_last: bool,

    /// Current iteration order. Sequential by default.
    /// After shuffle(): a permutation of 0..num_samples.
    order: Order,

    /// Device where data lives.
    device: Device,
}

enum Order {
    /// Sequential iteration. Batches use narrow() -- zero allocation.
    Sequential,
    /// Shuffled indices. Batches use index_select() -- batch-sized allocation.
    Shuffled(Vec<usize>),
}
```

**Why const generic?** Dataset arity is always a compile-time fact - you
never load an unknown number of tensors. `ResidentLoader<2>` for (input, target),
`ResidentLoader<3>` for (image, label, case). The type system catches arity
mismatches at build time, not at epoch 47. Batch containers are `[Tensor; N]`
on the stack - zero heap allocation for the common case of 2-4 tensors.

**Constructor variants:**

```rust
impl<const N: usize> ResidentLoader<N> {
    /// Create from tensors already on the target device.
    /// All tensors must have the same size along dim 0.
    /// Returns Err if shapes mismatch or devices differ.
    pub fn new(tensors: [Tensor; N], batch_size: usize) -> Result<Self>;

    /// Move CPU tensors to the specified device, then create loader.
    /// Convenience: calls to_device() on each tensor.
    pub fn load(
        tensors: [Tensor; N],
        batch_size: usize,
        device: Device,
    ) -> Result<Self>;
}
```

**Iteration:**

```rust
impl<const N: usize> ResidentLoader<N> {
    /// Shuffle iteration order for the next epoch.
    /// Does not move data -- only changes which indices each batch selects.
    pub fn shuffle(&mut self, rng: &mut Rng);

    /// Reset to sequential order.
    pub fn reset_order(&mut self);

    /// Number of batches per epoch.
    pub fn len(&self) -> usize;

    /// Number of batches with potential partial last batch.
    pub fn num_batches_total(&self) -> usize;

    /// Iterate over batches. Each item is [Tensor; N] -- compile-time arity,
    /// zero heap allocation, native pattern matching.
    pub fn iter(&self) -> ResidentBatchIter<'_, N>;
}

impl<'a, const N: usize> Iterator for ResidentBatchIter<'a, N> {
    type Item = [Tensor; N];
}
```

**Idiomatic usage with destructuring:**

```rust
// Letter model: 2 tensors
let loader = ResidentLoader::load([images, labels], 64, device)?;
for [img, lbl] in loader.iter() {
    let input = Variable::new(img, true);
    let target = Variable::new(lbl, false);
    // ...
}

// SubScan: 3 tensors
let loader = ResidentLoader::load([images, letters, cases], 64, device)?;
for [img, target, case] in loader.iter() {
    // compile-time: can't destructure as [img, target] -- arity mismatch
}
```

**Configuration:**

```rust
impl<const N: usize> ResidentLoader<N> {
    /// Drop incomplete last batch (builder pattern).
    pub fn drop_last(mut self, drop: bool) -> Self;

    /// Total samples.
    pub fn num_samples(&self) -> usize;

    /// Device where data lives.
    pub fn device(&self) -> Device;

    /// Total bytes used by loaded data tensors.
    pub fn data_bytes(&self) -> usize;
}
```

**Batch extraction (internal):**

```rust
impl<const N: usize> ResidentLoader<N> {
    /// Sequential batch: narrow() each tensor. Zero allocation.
    fn batch_sequential(&self, batch_idx: usize) -> [Tensor; N] {
        let start = batch_idx * self.batch_size;
        let len = (self.num_samples - start).min(self.batch_size);
        std::array::from_fn(|i| {
            self.tensors[i].narrow(0, start as i64, len as i64).unwrap()
        })
    }

    /// Shuffled batch: index_select() each tensor. Batch-sized allocation per tensor.
    fn batch_shuffled(&self, indices: &[usize], batch_idx: usize) -> [Tensor; N] {
        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(indices.len());
        let batch_indices: Vec<i64> = indices[start..end].iter()
            .map(|&i| i as i64)
            .collect();
        let idx_tensor = Tensor::from_i64(&batch_indices, self.device).unwrap();
        std::array::from_fn(|i| {
            self.tensors[i].index_select(0, &idx_tensor).unwrap()
        })
    }
}
```

`std::array::from_fn` constructs `[Tensor; N]` inline - no Vec, no heap.
The compiler knows N at monomorphization time.

### 2. VRAM budget estimation

A pre-flight check that estimates whether the training setup fits in VRAM.

```rust
pub struct ResidentBudget {
    /// Bytes needed for dataset tensors.
    pub data_bytes: usize,
    /// Bytes for model parameters.
    pub param_bytes: usize,
    /// Bytes for gradients (same as param_bytes for trainable params).
    pub grad_bytes: usize,
    /// Bytes for optimizer state (e.g., 2× param_bytes for Adam m + v).
    pub optimizer_bytes: usize,
    /// Estimated activation memory (model-dependent, conservative).
    pub activation_estimate: usize,
    /// CUDA context and allocator overhead.
    pub overhead_bytes: usize,
    /// Sum of all above.
    pub total_bytes: usize,
    /// Available VRAM on the target device.
    pub available_bytes: usize,
    /// Whether total_bytes <= available_bytes.
    pub fits: bool,
    /// Headroom: available - total (can be negative).
    pub headroom_bytes: i64,
}

impl ResidentBudget {
    /// Estimate VRAM budget for a training setup.
    ///
    /// `data` -- tensors that will be loaded into VRAM.
    /// `params` -- model parameters (from model.parameters()).
    /// `optimizer` -- optimizer kind, determines state multiplier.
    ///
    /// Activation memory is estimated conservatively as 2× the largest
    /// single batch (batch_size × largest intermediate). This is a rough
    /// estimate -- actual usage depends on model depth and width.
    pub fn estimate(
        data: &[&Tensor],
        params: &[Parameter],
        optimizer: OptimizerKind,
        batch_size: usize,
        device: Device,
    ) -> Result<Self>;
}

pub enum OptimizerKind {
    /// SGD: momentum buffer only (1× param bytes, 0× without momentum).
    SGD { momentum: bool },
    /// Adam/AdamW: m + v buffers (2× param bytes).
    Adam,
}

impl fmt::Display for ResidentBudget {
    /// Human-readable budget breakdown:
    ///
    /// VRAM Budget:
    ///   Data:        201.3 MB
    ///   Parameters:    0.4 MB
    ///   Gradients:     0.4 MB
    ///   Optimizer:     0.7 MB (Adam)
    ///   Activations:  ~12.0 MB (estimate)
    ///   Overhead:     400.0 MB (CUDA context)
    ///   ─────────────────────
    ///   Total:        614.8 MB
    ///   Available:  16384.0 MB
    ///   Headroom:  15769.2 MB  ✓
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { ... }
}
```

**Activation memory estimation** is the one genuinely hard part. The exact
activation footprint depends on model architecture, batch size, and which ops
retain intermediates for backward. A conservative estimate (2× largest layer
output × batch_size × model_depth) is better than nothing, but the budget
should clearly label it as an estimate and recommend a trial run.

For FBRL's small models, the activation memory is typically a fraction of the
data. The budget is dominated by dataset size.

### 3. CUDA Graph captured training pattern

Not a new API - this is a usage pattern combining existing `CudaGraph` with
`ResidentLoader`. Documented as the recommended high-performance path.

```rust
// --- Setup ---
let device = Device::CUDA(0);
let mut rng = Rng::seed(42);

// Load data into VRAM
let loader = ResidentLoader::load(
    vec![x_data, y_data],
    batch_size,
    device,
)?.drop_last(true);  // fixed batch shapes for CUDA Graph

// Build model (already on device)
let model = FlowBuilder::from(Linear::on_device(input_dim, hidden, device)?)
    .through(GELU)
    .through(Linear::on_device(hidden, output_dim, device)?)
    .build()?;
model.train();

let params = model.parameters();
let mut optimizer = Adam::new(&params, 0.001);

// --- Allocate input buffers for CUDA Graph ---
// These are the tensors the captured graph reads from.
// We copy_ batch data into them before each replay.
let x_buf = Tensor::empty(&[batch_size as i64, input_dim as i64], device.opts())?;
let y_buf = Tensor::empty(&[batch_size as i64, output_dim as i64], device.opts())?;

// --- Capture training step ---
let graph = cuda_graph_capture(3, None, || {
    let input = Variable::new(x_buf.shallow_clone(), true);
    let target = Variable::new(y_buf.shallow_clone(), false);
    optimizer.zero_grad();
    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;
    loss.backward()?;
    optimizer.step()?;
    Ok(())
})?;

// --- Training loop: zero CPU dispatch ---
for epoch in 0..num_epochs {
    loader.shuffle(&mut rng);

    for batch in loader.iter() {
        x_buf.copy_(&batch[0], false)?;  // GPU→GPU, fast
        y_buf.copy_(&batch[1], false)?;
        graph.replay()?;                  // single launch
    }

    // Metrics require leaving the graph -- read loss outside capture
    // (or capture a separate forward-only graph for evaluation)
}
```

**What the CPU does per batch:** two `copy_()` calls and one `replay()`.
All three are near-instant GPU commands. The GPU does all the work.

**Constraints:**
- All batch shapes must be identical (`drop_last(true)`).
- No `loss.item()` inside the captured graph (CPU-GPU sync).
- No conditional control flow (no early stopping checks per batch).
- Model architecture must be static (no dynamic routing).

**Metrics workaround:** Run a separate non-captured forward pass every N
batches or at epoch end to read loss values. The cost is one extra forward
pass, amortized over many batches.

```rust
// Every 100 batches or at epoch end:
cuda_synchronize(0);
let eval_loss = no_grad(|| {
    let input = Variable::new(x_buf.shallow_clone(), false);
    let target = Variable::new(y_buf.shallow_clone(), false);
    let pred = model.forward(&input)?;
    mse_loss(&pred, &target)?.item()
})?;
model.record_scalar("loss", eval_loss);
```

### 4. Phase-level capture for conditional training

Full-graph capture is incompatible with conditional logic (loops that terminate
on a condition, retry mechanisms, confidence thresholds). But FBRL's subscan
training has exactly this pattern: an inner loop where the subscan proposes a
position, the frozen letter model tries to read, and the loop repeats until the
letter model succeeds or a max attempt count is reached.

The solution: **capture individual phases as separate CUDA Graphs**, and let
the CPU orchestrate the conditional flow between them.

```
Full capture:   [subscan → letter → check → loop? → loss → backward → step]
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 ONE graph -- no conditionals allowed

Phase capture:  [subscan → letter] + [loss → backward → step]
                 ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
                 attempt_graph        update_graph
                 replayed N times     replayed once
                 CPU checks between
```

**Driving use case: subscan training loop**

The subscan learns to locate a letter position. Each training sample runs an
inner loop: subscan predicts a position, the frozen letter model tries to read
at that position, and the result is checked. The loop repeats until the letter
model can read (confidence above threshold) or max attempts is reached.

```rust
// --- Capture individual phases ---

// Phase 1: one "attempt" (subscan predicts, letter reads)
let attempt_graph = cuda_graph_capture(3, pool, || {
    // Subscan predicts letter position from the glimpse
    position_buf.copy_(&subscan.forward(&glimpse_var)?.data(), false)?;
    // Letter model reads at that position (frozen, no grad)
    let letter_out = no_grad(|| letter.forward(&position_var))?;
    confidence_buf.copy_(&letter_out.data(), false)?;
    Ok(())
})?;

// Phase 2: loss + backward + optimizer step (runs once per sample)
let update_graph = cuda_graph_capture(3, pool, || {
    optimizer.zero_grad();
    let loss = cross_entropy_loss(&letter_out_var, &target_var)?;
    loss.backward()?;
    optimizer.step()?;
    Ok(())
})?;

// --- Training loop ---
for batch in loader.iter() {
    glimpse_buf.copy_(&batch[0], false)?;
    target_buf.copy_(&batch[1], false)?;

    // Inner loop: subscan proposes, letter checks, repeat until success
    let mut attempts = 0;
    loop {
        attempt_graph.replay()?;
        attempts += 1;

        // ONE sync to read the confidence score -- lightweight
        cuda_synchronize(0);
        let confidence = confidence_buf.item::<f32>()?;

        if confidence > threshold || attempts >= max_attempts {
            break;
        }
        // Subscan will try again with updated position
    }

    // Update subscan weights based on final attempt
    update_graph.replay()?;

    // --- Monitor integration: metrics are free here ---
    // Confidence was already synced (needed for the conditional).
    // Attempt count is a CPU counter.
    // Loss needs one sync after update_graph, but once per sample.
    cuda_synchronize(0);
    let loss_val = loss_buf.item::<f32>()?;

    model.record_scalar("loss", loss_val as f64);
    model.record_scalar("attempts", attempts as f64);
    model.record_scalar("confidence", confidence as f64);
}

// End of epoch: flush + log as usual
model.flush(&[]);
monitor.log(epoch, t.elapsed(), &model);

```

**What the CPU does per attempt:** one `replay()` + one `sync` + one `item()`.
The sync is the cost - it forces the CPU to wait for the GPU. But compare:

| Pattern | GPU ops per attempt | CPU-GPU syncs |
|---------|-------------------|---------------|
| No CUDA Graphs | ~100+ kernel launches | 1 (for item) |
| Phase-level graphs | 1 replay | 1 (for item) |

The sync cost is the same either way (you need the confidence value to decide).
But phase-level capture eliminates ~100 kernel dispatches per attempt. The GPU
runs the entire subscan + letter forward as a single pre-recorded operation.

**As training progresses:** the average attempt count drops (subscan gets better
at locating). Early in training: maybe 5 attempts per sample. Late: 1-2. The
conditional overhead decreases naturally as the model improves, converging
toward the zero-dispatch performance of full capture.

**The spectrum:**

| Approach | CPU work per batch | Conditionals | Best for |
|----------|-------------------|--------------|----------|
| Full CUDA Graph | ~0 | None | Fixed forward pass (letter model) |
| Phase-level graphs | Light (sync between phases) | Between phases | Conditional loops (subscan) |
| No CUDA Graphs | Heavy (every kernel) | Full | Prototyping, debugging |

All three approaches benefit equally from VRAM-resident data - the transfer
elimination is independent of how the compute is dispatched. Phase-level
capture is the natural fit for FBRL's progressive training, where inner loops
and retry mechanisms are structural.

**Shared memory pools:** when using multiple captured graphs (attempt_graph +
update_graph), share a memory pool so they can reuse VRAM allocations:

```rust
let pool = cuda_graph_pool_handle()?;
let attempt_graph = cuda_graph_capture(3, Some(pool), || { ... })?;
let update_graph = cuda_graph_capture(3, Some(pool), || { ... })?;
```

### 5. Live dashboard compatibility

The monitor has three data channels. All three work with VRAM-resident
training, including captured CUDA Graphs:

**Resource metrics** (VRAM, CPU, GPU utilization): sampled by `monitor.log()`
via NVML system calls. Pure CPU-side. Works unchanged with any training
pattern - no GPU sync needed.

**Training metrics** (loss, attempts, lr, confidence): pushed via
`record_scalar()`, which is pure Rust - appends an f64 to a Vec in the
Graph's batch buffer. The only GPU involvement is obtaining the value itself
(e.g., `loss_buf.item()` requires a sync).

**Epoch aggregation**: `flush()` computes batch means, `monitor.log()` pushes
to the SSE dashboard. All CPU-side.

| Capture pattern | Metric collection cost | Dashboard impact |
|-----------------|----------------------|-----------------|
| No capture | 1 sync per `.item()` call (same as today) | None |
| Full capture + async metrics | 0 syncs (async copy, 1-batch delay) | None |
| Phase-level | ~0 extra (sync already needed for conditional) | None |

#### Async metric push: zero-sync dashboard

The key insight: **metrics are observability, not control flow.** The dashboard
doesn't need the loss value from the current batch right now - it needs it
eventually. A one-batch delay is completely invisible on a training curve.

By combining `pin_memory()` + `copy_(non_blocking=true)`, the GPU writes
metric values to pinned CPU memory without waiting, and the CPU reads the
**previous** batch's value:

```
Batch N:    GPU computes loss → copy_ to pinned CPU (async) → GPU moves on
Batch N+1:  CPU reads batch N's loss (already in pinned buffer)
            GPU computes loss → copy_ to pinned CPU (async) → GPU moves on
```

Neither side ever waits for the other.

```rust
// --- Setup: pre-allocate pinned CPU metric buffers ---
let loss_gpu = Tensor::empty(&[], device.opts())?;        // GPU-side buffer
let loss_cpu = Tensor::empty(&[], TensorOptions::default())?.pin_memory()?;  // pinned CPU

// --- Capture: graph writes loss to GPU buffer ---
let graph = cuda_graph_capture(3, pool, || {
    let input = Variable::new(x_buf.shallow_clone(), true);
    let target = Variable::new(y_buf.shallow_clone(), false);
    optimizer.zero_grad();
    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;
    loss_gpu.copy_(&loss.data(), false)?;  // GPU→GPU, captured in graph
    loss.backward()?;
    optimizer.step()?;
    Ok(())
})?;

// --- Training loop: fully async metrics ---
for (i, batch) in loader.iter().enumerate() {
    // Read PREVIOUS batch's loss -- the async copy has completed by now
    // (the GPU is always ahead of the CPU in the pipeline)
    if i > 0 {
        let val = loss_cpu.to_f32_vec()?[0];
        model.record_scalar("loss", val as f64);
    }

    // Dispatch this batch
    x_buf.copy_(&batch[0], false)?;
    y_buf.copy_(&batch[1], false)?;
    graph.replay()?;

    // Kick off async copy: GPU→pinned CPU, non-blocking
    // GPU doesn't wait, CPU doesn't wait -- the copy streams in the background
    loss_cpu.copy_(&loss_gpu, true)?;  // non_blocking = true
}

// Flush final batch's metric
let val = loss_cpu.to_f32_vec()?[0];
model.record_scalar("loss", val as f64);
model.flush(&[]);
monitor.log(epoch, t.elapsed(), &model);
```

**Why `to_f32_vec()` instead of `.item()`?** `.item()` may trigger a sync
internally. Reading from the pinned CPU tensor via `to_f32_vec()` is a pure
CPU memory read - the data is already there because the non-blocking copy
completed while the GPU was working on the next batch.

**The principle:** separate **control flow values** from **observability values**.

| Value type | Needs real-time? | Pattern |
|-----------|-----------------|---------|
| Conditional (confidence for retry loop) | Yes - CPU must decide now | `sync` + `.item()` |
| Observability (loss for dashboard) | No - one-batch delay is fine | `copy_(non_blocking)` to pinned CPU |

This means even the **phase-level pattern** can avoid the loss sync:

```rust
// Inner loop: sync needed for confidence (control flow)
loop {
    attempt_graph.replay()?;
    cuda_synchronize(0);
    let confidence = confidence_buf.item::<f32>()?;  // sync: unavoidable
    if confidence > threshold || attempts >= max_attempts { break; }
    attempts += 1;
}

// Outer: loss is just observability -- async push
update_graph.replay()?;
loss_cpu.copy_(&loss_gpu, true)?;  // no sync needed

// Read previous sample's loss (already in pinned buffer)
if sample_idx > 0 {
    model.record_scalar("loss", prev_loss_cpu.to_f32_vec()?[0] as f64);
}
```

**Phase-level capture is dashboard-friendly by design.** The confidence sync
that drives the conditional loop also provides the confidence metric for free.
The loss uses async push - zero additional sync.

### 6. Shuffle strategy

Two modes, chosen automatically based on `Order`:

| Mode | Extraction | Allocation | When |
|------|-----------|------------|------|
| Sequential | `narrow(0, start, len)` | Zero (view) | Default, no shuffle |
| Shuffled | `index_select(0, batch_indices)` | Batch-sized per call | After `shuffle()` |

**Why not shuffle the full dataset?**

`index_select(0, full_permutation)` would create a complete copy of the dataset
in VRAM - temporarily doubling memory usage. For a 2 GB dataset on a 4 GB
budget, that's fatal.

Per-batch `index_select()` allocates only `batch_size` rows at a time. The
temporary is tiny and immediately freed after the training step.

**Shuffle is CPU-side:**

The permutation is generated on CPU via `Rng::shuffle()` on a `Vec<usize>`.
This is O(N) where N is the sample count. For 100K samples, this takes
microseconds - negligible compared to one GPU batch.

The per-batch index tensor creation (`Tensor::from_i64`) is a small CPU→GPU
transfer (batch_size × 8 bytes). For batch_size=64, that's 512 bytes -
invisible.

### 7. Integration with training monitor

`ResidentLoader` reports its memory usage to the monitor:

```rust
impl ResidentLoader {
    /// Summary string for monitor/dashboard display.
    pub fn summary(&self) -> String {
        // "ResidentLoader<2>: 50000 samples, 64/batch, 781 batches, 201.3 MB on CUDA:0"
    }
}
```

The monitor can display VRAM breakdown (data vs model vs overhead) in the
dashboard. The budget struct provides all the numbers.

### 8. Double-buffered async loading

Double-buffering serves two purposes: handling datasets larger than VRAM,
and - more importantly - providing a **natural hook for progressive training
strategies** like curriculum learning.

When the full dataset exceeds available VRAM but a single epoch's data fits
(with room for a second buffer), double-buffering eliminates transfer overhead
from epoch 2 onward:

```
Epoch 1:  [train on buffer A] ──────────────────────>
          [async load buffer B] ─────>  (ready before epoch 1 ends)

Epoch 2:  [train on buffer B] ──────────────────────>
          [async load buffer A] ─────>

Epoch 3:  [train on buffer A] ...
```

Two VRAM buffers, each holding one epoch's data. While the GPU trains on
buffer A, the next epoch's data loads asynchronously into buffer B via
`to_device_async()`. At epoch boundary: swap which buffer is active. The
first epoch pays the transfer cost; every subsequent epoch is transfer-free.

```rust
impl ResidentLoader {
    /// Create a double-buffered loader for datasets larger than VRAM.
    ///
    /// `epoch_data_fn` returns the N tensors for a given epoch index.
    /// Two epochs' worth of data must fit in VRAM simultaneously.
    /// The first epoch loads synchronously; subsequent epochs prefetch
    /// asynchronously during training.
    pub fn double_buffered(
        epoch_data_fn: impl Fn(usize) -> Result<[Tensor; N]>,
        batch_size: usize,
        device: Device,
    ) -> Result<Self>;

    /// Signal that the current epoch is done, swap buffers.
    /// If the prefetch isn't complete yet, blocks until it is.
    /// Starts async prefetch for the next-next epoch into the freed buffer.
    pub fn advance_epoch(&mut self) -> Result<()>;
}
```

**VRAM cost:** 2x one epoch's data. For FBRL's small datasets, this is
typically well under 1 GB total - trivial on 16+ GB hardware.

**The `epoch_data_fn` callback is the key.** It doesn't just return "the next
chunk of a too-large dataset" - it returns **whatever data this epoch should
train on.** This makes it the natural integration point for progressive
training strategies:

```rust
// SubScan curriculum: unlock harder positions as training progresses
let loader = ResidentLoader::double_buffered(
    |epoch| {
        let difficulty = (epoch as f64 / 50.0).min(1.0);  // ramp over 50 epochs
        Ok(generate_subscan_data(difficulty))
        // epoch 0:  centered letters, tight positions
        // epoch 25: off-center, wider position variance
        // epoch 50: full difficulty, noisy positions
    },
    batch_size,
    device,
)?;
```

The callback produces different tensors each epoch, and the double-buffer
ensures the next epoch's data is already in VRAM when training needs it.
This is not a "data doesn't fit" escape hatch - it's a **training strategy
primitive**.

**When double-buffering is the right choice:**
- Progressive difficulty / curriculum learning (SubScan position unlocking)
- Hard example mining (resample based on previous epoch's errors)
- Data augmentation that produces different tensors each epoch
- Dataset is too large to fit entirely, but one epoch's worth fits

**When to skip it:**
- Full dataset fits in VRAM and doesn't change - use `ResidentLoader::load()`
- Even one epoch's data doesn't fit - fall back to per-batch CPU→GPU transfer
  using `to_device()` or `to_device_async()` with `pin_memory()`

This is not a partial-VRAM compromise - it's full VRAM-resident training with
a streaming data source. The GPU never waits for data after epoch 1.

---

## Performance analysis

### Transfer elimination

| Operation | Standard | VRAM-resident |
|-----------|----------|---------------|
| Per-batch CPU→GPU | 0.1-2 ms (depends on batch size) | 0 |
| Per-batch GPU→GPU copy_ | - | ~0.01 ms |
| Epochs × batches saved | 100 × 781 = 78,100 transfers | 0 transfers |

For the letter model (781 batches/epoch, 100 epochs): eliminating 78,100
CPU→GPU transfers at ~0.5 ms each saves ~39 seconds of pure transfer time.

### CUDA Graph dispatch reduction

Without CUDA Graphs, each batch dispatches ~100-300 individual CUDA kernels
from the CPU. With CUDA Graphs, this becomes 1 replay call.

| Pattern | CPU dispatch calls/batch | GPU idle between kernels |
|---------|------------------------|------------------------|
| Standard | ~200 | ~2-5 μs × 200 = 0.4-1 ms |
| CUDA Graph | 1 | ~0 (pre-scheduled) |
| VRAM + Graph | 3 (copy, copy, replay) | ~0 |

### Memory overhead

ResidentLoader itself is negligible:
- `Vec<Tensor>`: N pointers (N = number of input tensors, typically 2-4)
- `Vec<usize>` for shuffled indices: 8 bytes × num_samples
- For 100K samples: 800 KB for the index array

The dataset tensors are the real cost, and they're the same tensors that would
exist on CPU anyway - just on a different device.

### Shuffle overhead

Per-epoch shuffle cost:
- CPU: `Rng::shuffle()` on Vec<usize> - O(N), ~10 μs for 100K samples
- Per-batch: `Tensor::from_i64(batch_indices)` - 512 bytes for batch_size=64
- Per-batch: `index_select()` - one GPU kernel, ~0.01 ms

Total shuffle overhead per epoch: ~0.01 ms × num_batches ≈ 8 ms for 781 batches.
Compared to the ~39 seconds saved from eliminated transfers, this is noise.

---

## Implementation plan

### Phase A: ResidentLoader core

**Files:** `flodl/src/data/mod.rs` (new module), `flodl/src/lib.rs`

1. Create `data` module with `ResidentLoader` struct.
2. `new()` - validate tensors (same dim 0, same device).
3. `load()` - move to device, then `new()`.
4. `drop_last()` builder method.
5. Sequential iteration via `narrow()`.
6. `shuffle()` via `Rng`, shuffled iteration via `index_select()`.
7. `reset_order()` to return to sequential.
8. `iter()` returning `ResidentBatchIter`.
9. `data_bytes()`, `summary()`, `len()`, `num_samples()`, `device()`.

### Phase B: VRAM budget estimation

**Files:** `flodl/src/data/budget.rs`

1. `ResidentBudget` struct with all fields.
2. `estimate()` - compute bytes from tensor shapes and dtypes.
3. `OptimizerKind` enum for state multiplier.
4. `Display` impl for human-readable output.
5. Activation estimate heuristic (conservative, clearly labeled).

### Phase C: Double-buffered async loading

**Files:** `flodl/src/data/mod.rs`

1. `double_buffered()` constructor with epoch data callback.
2. Two internal ResidentLoader buffers (active + prefetch).
3. `advance_epoch()` - swap buffers, start async prefetch for next epoch.
4. Async loading via `to_device_async()` + `pin_memory()` on a background thread.
5. `prefetch_ready()` check for monitoring.

### Phase D: Documentation and examples

**Files:** `docs/tutorials/`, `examples/vram_training/`

1. Tutorial: "VRAM-Resident Training" - when to use, budget check, basic loop.
2. Tutorial: "Zero-Dispatch Training" - ResidentLoader + CUDA Graphs pattern.
3. Example: `vram_training/main.rs` - complete working example.
4. Update `docs/pytorch_migration.md` with comparison to PyTorch DataLoader.

### Phase E: Monitor integration

**Files:** `flodl/src/monitor/`

1. Dashboard shows VRAM breakdown (data / model / optimizer / free).
2. `ResidentLoader::summary()` displayed in training header.
3. Optional: budget check result in verbose build output.

---

## API summary

New module: `flodl::data`

```rust
// Data loading (const generic: N = number of tensors per sample)
pub struct ResidentLoader<const N: usize> { ... }
impl<const N: usize> ResidentLoader<N> {
    pub fn new(tensors: [Tensor; N], batch_size: usize) -> Result<Self>;
    pub fn load(tensors: [Tensor; N], batch_size: usize, device: Device) -> Result<Self>;
    pub fn drop_last(self, drop: bool) -> Self;
    pub fn shuffle(&mut self, rng: &mut Rng);
    pub fn reset_order(&mut self);
    pub fn iter(&self) -> ResidentBatchIter<'_, N>;  // yields [Tensor; N]
    pub fn len(&self) -> usize;
    pub fn num_samples(&self) -> usize;
    pub fn device(&self) -> Device;
    pub fn data_bytes(&self) -> usize;
    pub fn summary(&self) -> String;

    // Double-buffered async loading
    pub fn double_buffered(
        epoch_data_fn: impl Fn(usize) -> Result<[Tensor; N]>,
        batch_size: usize,
        device: Device,
    ) -> Result<Self>;
    pub fn advance_epoch(&mut self) -> Result<()>;
    pub fn prefetch_ready(&self) -> bool;
}

// Budget estimation
pub struct ResidentBudget { ... }
pub enum OptimizerKind { SGD { momentum: bool }, Adam }
impl ResidentBudget {
    pub fn estimate(
        data: &[&Tensor],
        params: &[Parameter],
        optimizer: OptimizerKind,
        batch_size: usize,
        device: Device,
    ) -> Result<Self>;
}
```

Re-exported from `flodl::prelude` or `flodl::data`.

---

## Connection to FBRL thesis

This design directly enables the "efficiency through decomposition" property
of the trajectory thesis:

> Each layer is a tested, standalone model before it becomes a component.

VRAM-resident training is a **consequence** of this principle, not a separate
optimization. When you decompose a problem into small, independently trainable
pieces:

1. Each piece has a small model → small parameter + optimizer footprint
2. Each piece trains on a focused dataset → small data footprint
3. Small model + small data = fits in VRAM
4. Fits in VRAM = eliminate transfers = faster training
5. Faster training = faster iteration = more experiments = better models

The compound effect: FBRL doesn't just make problems tractable, it makes them
**fast**. And this advantage scales with hardware - as VRAM grows, larger
compositions qualify for full-VRAM training without code changes.

**Progressive VRAM residency:**

```
Letter model:  trivially fits (< 500 MB)     -- today, any GPU
Subscan model: easily fits (< 1 GB)          -- today, 8+ GB GPU
Word model:    fits with margins (< 2 GB)    -- today, 16+ GB GPU
Sentence model: fits (< 8 GB)               -- near future, 24+ GB GPU
Paragraph:     fits (< 16 GB)               -- next gen, 48+ GB GPU
```

Each level of the FBRL hierarchy will naturally cross the VRAM-residency
threshold as hardware improves. No architectural changes needed - just more
VRAM.

---

## Open questions

1. **Activation memory estimation.** The budget's activation estimate is
   conservative but imprecise. A more accurate approach would be to run one
   forward pass and measure peak VRAM delta. Adding a `budget.measure()`
   method that does this is worth it - cheap and removes the main source of
   budget uncertainty. The conservative estimate is good for pre-flight, but
   measured is better for tight fits.

2. ~~**Multi-tensor batch return type.**~~ **Resolved:** const generic
   `ResidentLoader<const N: usize>` with `[Tensor; N]` batch type. Compile-time
   arity check, zero heap allocation, idiomatic destructuring.

3. **Epoch-level CUDA Graph.** Instead of capturing one batch step and replaying
   N times, could we capture an entire epoch (all batches)? This would eliminate
   even the per-batch `copy_()` calls. But it requires knowing the batch count
   at capture time and the graph size grows linearly with batches. Probably not
   worth it for v1.

4. **Augmentation on GPU.** Some data augmentation (random crops, flips, noise)
   could run on GPU, avoiding CPU→GPU round trips. This is orthogonal to
   ResidentLoader but composes well with it. Worth a separate design?

5. **Multi-GPU data splitting.** Multi-GPU training is now implemented via
   `Trainer::setup()` and `Trainer::builder()`. The `DataLoader` supports distributed
   mode with proportional epoch sharding across devices. See
   [DDP Reference](../ddp.md) for details.

6. **Double-buffer epoch boundary.** When `advance_epoch()` is called and
   the async prefetch hasn't finished, the CPU blocks. In practice the
   prefetch should complete well before the epoch ends (data transfer is
   faster than training). But if the model is trivially fast, the prefetch
   might not finish in time. Worth adding a `prefetch_ready()` check so
   users can monitor this?
