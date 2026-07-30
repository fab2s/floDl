# Training

This tutorial puts everything together: loss functions, optimizers, gradient
clipping, mixed precision training, and the training loop. It builds on
[Modules](03-modules.md).

> **Runnable examples**: [`sine_wave`](../../flodl/examples/sine_wave/) is the
> shortest end-to-end loop, [`quickstart`](../../flodl/examples/quickstart/) adds a
> monitor, and [`mixed_precision`](../../flodl/examples/mixed_precision/) shows
> `GradScaler` in place.

## Loss Functions

All loss functions are free functions returning a scalar `Variable` ready
for `backward()`.

### Regression Losses

```rust
use flodl::*;

let loss = mse_loss(&pred, &target)?;              // Mean Squared Error
let loss = l1_loss(&pred, &target)?;               // Mean Absolute Error
let loss = smooth_l1_loss(&pred, &target, 1.0)?;   // Huber loss (beta=1.0)
let loss = poisson_nll_loss(&pred, &target, true)?; // Poisson NLL (log_input=true)
```

### Classification Losses

```rust
// Cross-Entropy from raw logits.
// pred: [batch, classes] logits.
// target: [batch] class indices (Int64) or [batch, classes] one-hot/soft labels.
let loss = cross_entropy_loss(&logits, &target)?;

// Negative Log Likelihood - use after log_softmax
let loss = nll_loss(&log_probs, &target)?;

// Binary Cross-Entropy (from probabilities, after sigmoid)
let loss = bce_loss(&probs, &target)?;

// Binary Cross-Entropy with logits (numerically stable - preferred)
let loss = bce_with_logits_loss(&logits, &target)?;

// Focal Loss - down-weights easy examples for class imbalance
let loss = focal_loss(&logits, &target, 0.25, 2.0)?;  // alpha, gamma

// KL Divergence
let loss = kl_div_loss(&log_pred, &target)?;

// CTC Loss - for sequence-to-sequence without alignment (speech, OCR)
let loss = ctc_loss(&log_probs, &targets, &input_lengths, &target_lengths, 0)?;
```

### Metric Learning Losses

```rust
// Triplet margin loss - push negatives away from anchor-positive pairs
let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0)?;

// Cosine embedding loss - similar pairs close, dissimilar far
let loss = cosine_embedding_loss(&x1, &x2, &labels, 0.5)?;

// Hinge embedding loss - for binary tasks with {-1, +1} labels
let loss = hinge_embedding_loss(&input, &labels, 1.0)?;

// Margin ranking loss - x1 should be ranked higher than x2
let loss = margin_ranking_loss(&x1, &x2, &labels, 0.0)?;
```

## Optimizers

All optimizers implement the `Optimizer` trait:

```rust
pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&self);
    fn set_lr(&mut self, lr: f64);
    fn set_group_lr(&mut self, group: usize, lr: f64);  // per-group LR
}
```

### SGD

```rust
let optimizer = SGD::new(&params, 0.01, 0.9);  // lr, momentum (0.0 for vanilla SGD)
```

### Adam / AdamW

```rust
let optimizer = Adam::new(&params, 0.001);          // default betas (0.9, 0.999), eps=1e-8
let optimizer = AdamW::new(&params, 0.001, 0.01);   // decoupled weight decay
```

### RMSprop

Adaptive learning rate with exponential moving average of squared gradients:

```rust
let optimizer = RMSprop::new(&params, 0.01);  // default alpha=0.99, eps=1e-8
```

### Adagrad

Accumulates all past squared gradients - works well for sparse features:

```rust
let optimizer = Adagrad::new(&params, 0.01);
```

### RAdam / NAdam

Rectified Adam (variance-aware warmup) and Nesterov-accelerated Adam:

```rust
let optimizer = RAdam::new(&params, 0.001);  // auto-warmup via variance rectification
let optimizer = NAdam::new(&params, 0.001);  // Nesterov momentum with Adam
```

### Fused CUDA Optimizers

On CUDA, both `Adam` and `AdamW` automatically use `_fused_adamw_` - a
single multi-tensor kernel that updates all parameters, gradients, and
moment buffers in one launch. A naive implementation would require 4N
separate kernels (one each for momentum update, variance update, bias
correction, and parameter update, per parameter). The fused path reduces
this to a single kernel launch for all parameters in a group.

This is completely automatic. `Adam::new()` and `AdamW::new()` use the
fused path whenever parameters live on CUDA. No API changes are needed.

The fused kernel also exposes `grad_scale` and `found_inf` tensor
parameters internally, which `GradScaler` uses for integrated mixed
precision training (see [Mixed Precision Training](#mixed-precision-training)
below).

## Gradient Clipping

Prevent exploding gradients by clipping between `backward()` and
`optimizer.step()`:

```rust
clip_grad_norm(&params, 1.0)?;   // scale so total L2 norm <= max_norm
```

`clip_grad_value` clamps per element instead. See
[Utilities - Gradient clipping](08-utilities.md#gradient-clipping) for when
to pick which, and why both cost two kernels rather than `2N`.

## Device Placement

By default, all tensors and parameters live on CPU. To train on CUDA, use
`move_to_device` on the graph.

### Moving the model

```rust
let model = build_model()?;

if flodl::cuda_available() {
    model.move_to_device(Device::CUDA(0));
}

// Create optimizer AFTER move_to_device.
let params = model.parameters();
let optimizer = Adam::new(&params, 0.001);
```

## Trainer: write a step, get the loop

flodl's `Trainer` is the universal training entry, working on any Module
(Graph or otherwise), CPU, single GPU, or multi-GPU, all with the same
code. You describe **one training step** as a closure (forward + loss);
`Trainer::builder` owns the rest: the loop, the backward pass, the
optimizer step, the gradient sync, and the device replication.

```rust
// Step closure: takes the replica's model and one batch, returns the
// loss Variable. The framework calls backward + optimizer step + sync.
fn train_step(model: &impl Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    let pred = model.forward(&input)?;
    cross_entropy_loss(&pred, &target)
}

// Three closures: model factory (per-device build), optimizer factory,
// step. Then run().
let handle = Trainer::builder(
    |dev| build_model_on(dev),
    |params| Adam::new(params, 0.001),
    train_step,
)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(10)
    .run()?;

let state = handle.join()?;  // averaged params + buffers, ready for inference
```

This is the highest-level entry: framework owns the loop, the data
dispatch, the gradient sync (NCCL or CPU averaging), and the
optimizer. The
[ddp-bench](https://github.com/flodl-labs/flodl/tree/main/ddp-bench) suite
is the canonical reference for this pattern across MLP, LeNet, ResNet,
GPT-nano, char-RNN, and conv-AE models, each wired through the same
`train_step` closure.

The same call scales transparently from CPU → single GPU → multi-GPU
single-host → multi-host cluster. On a host with 2+ visible CUDA
devices it auto-promotes to process-per-rank. For mode selection
(`NcclCadence` (default), `CpuAsync`, etc.), heterogeneous-rig
cadence (ElChe), and cluster topology (`fdl.cluster.yml` /
`ClusterBuilder`), see [Multi-GPU Training](11-multi-gpu.md), the
[Heterogeneous & Multi-Host DDP tutorial](12-async-ddp.md), and the
[DDP Reference](../ddp.md).

### `TrainerConfig` - the config-bag form

When the call site wants every knob in one data struct (e.g.
config-driven launchers), `Trainer::run(model_fn, opt_fn, step_fn,
cfg)` takes a `TrainerConfig`:

```rust
let cfg = TrainerConfig::new(dataset)
    .batch_size(64)
    .num_epochs(50)
    .elche(ElCheConfig::nccl_cadence())         // default; just shown explicitly
    .max_grad_norm(5.0)
    .checkpoint_every(5)
    .save_path("ckpts/run43")
    .resume_from("ckpts/run42")
    .metrics_fn(Arc::new(|m| {
        eprintln!("epoch={} loss={:.4} {:.0}ms", m.epoch, m.avg_loss, m.epoch_ms);
        Ok(())
    }));

Trainer::run(model_factory, optim_factory, train_step, cfg)?.join()?;
```

Same launcher trampoline as `Trainer::builder(...).run()`. Pick
whichever shape matches your call site. Full setter surface in [DDP
Reference: `TrainerConfig<M>`](../ddp.md#trainerconfigm--the-umbrella).

## Keep your own loop

Want explicit control of the training loop (multi-stage losses, per-step
observation hooks, conditional backward, custom gradient-clipping
placement)? Pick one:

- **Framework owns the loop** (managed tier, recommended):
  `Trainer::builder(...).run()` above; the `train_step` closure is your
  forward + loss.
- **You own the loop body, controller owns scheduling** (cooperative
  tier): `Trainer::builder(...).into_worker()?` returns a `Worker` -
  `next_plan()` / `next_batch()` / `step()` / `finish()` - while the
  controller keeps cadence, partition, eval-election, and checkpointing
  (see [trainer-execution-tiers](../design/trainer-execution-tiers.md)).
- **Explicit per-rank control** (bypass tier, multi-GPU):
  `Ddp::wrap(&model, device, rank, &rendezvous)?`, calling
  `sync_params()` / `all_reduce_gradients()` yourself (see
  [Multi-GPU](11-multi-gpu.md)).
- **Single-device manual loop**: the pattern below.

`flodl-hf` task-head wrappers (e.g. `BertForSequenceClassification`)
`impl Module` directly, so they ride the same `Trainer::builder(...)` /
`Trainer::run(...)` entry - see
[HuggingFace Integration](14-flodl-hf.md) for a fine-tune walkthrough.

## The Training Loop

When you can't or don't want to use `Trainer` (non-Graph custom code,
single-device prototype, or you're learning the mechanics), the manual
pattern is: **forward -> loss -> zero_grad -> backward -> clip -> step**.
The framework runs these same six steps for you inside
`Trainer::builder(...).run()`; the `train_step` closure is the forward +
loss portion.

```rust
model.train();

for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    // 1. Forward
    let pred = model.forward(&input)?;

    // 2. Loss
    let loss = mse_loss(&pred, &target)?;

    // 3. Zero gradients
    optimizer.zero_grad();

    // 4. Backward
    loss.backward()?;

    // 5. Clip gradients
    clip_grad_norm(&params, 1.0)?;

    // 6. Update parameters
    optimizer.step()?;
}
```

## Observing Training

Tag the nodes you want to monitor when building the graph:

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(Linear::new(16, 2)?).tag("output")
    .build()?;
```

### Collect and Flush

For epoch-level metrics, collect scalar values during the batch loop and
flush at epoch boundaries:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = model.forward(&Variable::new(input.clone(), true))?;
        let loss = mse_loss(&pred, &Variable::new(target.clone(), false))?;

        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;

        model.collect(&["output"])?;            // from graph tag
        model.record_scalar("loss", loss.item()?);  // external metric
    }
    model.flush(&["output", "loss"]);           // batch mean -> epoch history
    model.end_epoch();
}
```

`collect` appends the scalar value of each tagged node to a batch buffer.
`record` pushes raw `f64` values into the same buffer. `flush` computes
the mean, stores it in epoch history, and clears the buffer.

## Stateful Graphs - end_step

Call `end_step()` after each training step. It severs autograd references
held by the graph and increments the step counter (used by schedulers and
observation). It detaches:

- **Forward-reference state buffers** (recurrent state carried between calls)
- **Tagged outputs** (Variables captured by `tag()` for observation)
- **Module internal state** (e.g., recurrent hidden state in custom modules)

> **Warning:** Forgetting `end_step()` causes linear memory growth - the
> autograd graph accumulates across batches without bound. If you see
> steadily rising RAM during training, a missing `end_step()` is the most
> likely cause.

```rust
model.train();

for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;

    optimizer.zero_grad();
    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;
    optimizer.step()?;

    model.end_step();  // break gradient chains + increment step counter
}
```

**When is it needed?** For any graph with forward references (`using("x")`
before `tag("x")`) it is mandatory. For graphs that use `tag()` for
observation, it prevents tagged output Variables from holding stale
autograd graph references between batches. Even for simple graphs, it is
good practice - it keeps the step counter accurate and costs nothing.

The lower-level `detach_state()` is available if you need to break gradient
chains without incrementing the step counter.

## Parameter Groups

All optimizers support per-group learning rates via a builder API:

```rust
let mut opt = Adam::with_groups()
    .group(&scan_params, 1e-3)     // group 0: high LR
    .group(&read_params, 1e-5)     // group 1: low LR
    .build();

// Adjust one group
opt.set_group_lr(1, 1e-4);

// Adjust all groups at once
opt.set_lr(1e-3);
```

`Adam::new(&params, lr)` still works for single-group usage. `SGD` and
`AdamW` have the same builder pattern (`SGD::with_groups(momentum)`,
`AdamW::with_groups(weight_decay)`).

## Parameter Freezing

Freeze parameters to disable gradient tracking - useful for transfer
learning:

```rust
for param in &encoder_params {
    param.freeze()?;   // no gradients will accumulate
}

// Later, unfreeze for fine-tuning:
for param in &encoder_params {
    param.unfreeze()?;
}

// Check status:
if param.is_frozen() { /* ... */ }
```

Frozen parameters are automatically skipped by optimizers (they produce
no gradient). Freezing works through `Rc<RefCell>` - a freeze is visible
everywhere the parameter is referenced.

## Checkpoints

Save and restore parameters, buffers, and a structural hash in one call:

```rust
model.save_checkpoint("/tmp/model.fdl")?;
let report = model.load_checkpoint("/tmp/model.fdl")?;  // validates, returns LoadReport
```

[Utilities - Checkpoints](08-utilities.md#checkpoints) covers the rest: the
lower-level `io::Write`/`io::Read` API for custom destinations, partial loading
for transfer learning, freezing what transferred, periodic saves, and
non-blocking background saves with `CpuWorker`.

## LR Scheduling

Schedulers are pure calculators - they never own the optimizer. You ask for
the step's LR and set it yourself:

```rust
let scheduler = CosineScheduler::new(0.001, 1e-6, 100);  // base_lr, min_lr, total_steps
optimizer.set_lr(scheduler.lr(step));
```

[Utilities - LR Scheduling](08-utilities.md#lr-scheduling) has the full
catalogue (step decay, cosine, exponential, multi-step, one-cycle, cyclic,
warmup composition, plateau) and the `Scheduler` trait.

## Mixed Precision Training

Mixed precision training runs eligible operations (matmul, convolutions,
linear layers) in a reduced-precision dtype (typically `Float16` or
`BFloat16`) while keeping numerically sensitive operations (losses, norms,
softmax) in full `Float32`. On GPUs with Tensor Cores (RTX 30xx, RTX 40xx,
RTX 50xx), this can deliver up to 3x speedup with minimal accuracy impact.

### Autocast

The `AutocastGuard` RAII guard enables automatic dtype dispatch for the
duration of its lifetime. The `autocast()` closure helper provides a
convenient scoped interface:

```rust
use flodl::*;

// RAII guard style
let _amp = AutocastGuard::new(DType::Float16);
let output = model.forward(&input)?;  // matmul dispatches to fp16
let loss = mse_loss(&output, &target)?;  // stays fp32
drop(_amp);

// Closure style (preferred)
let loss = autocast(DType::Float16, || {
    let output = model.forward(&input)?;
    mse_loss(&output, &target)
})?;

// Query whether autocast is active
if is_autocast_enabled() {
    // inside an autocast region
}
```

### GradScaler

Half-precision gradients can underflow to zero. `GradScaler` solves this
by scaling the loss before backward (inflating gradient magnitudes), then
unscaling gradients before the optimizer step. It dynamically adjusts the
scale factor - growing it when gradients stay finite, backing off when
inf/nan is detected.

```rust
let mut scaler = GradScaler::new();
// Initial scale: 65536, growth: 2x, backoff: 0.5x, interval: 2000 steps
```

The `step` method handles unscaling, inf/nan checking, and the optimizer
step in a single call. It returns `true` if the step was taken, or `false`
if it was skipped due to non-finite gradients:

```rust
let stepped = scaler.step(&params, &mut || optimizer.step())?;
scaler.update();  // adjust scale factor -- call after every step()
```

### Complete Mixed Precision Loop

```rust
let mut scaler = GradScaler::new();

model.train();
for (x, y) in &batches {
    let input = Variable::new(x.clone(), false);
    let target = Variable::new(y.clone(), false);

    // Forward under autocast -- eligible ops run in fp16
    let loss = autocast(DType::Float16, || {
        let pred = model.forward(&input)?;
        mse_loss(&pred, &target)
    })?;

    // Scale loss and backward
    let scaled = scaler.scale(&loss)?;
    optimizer.zero_grad();
    scaled.backward()?;

    // Unscale gradients, check for inf/nan, clip, and step
    clip_grad_norm(&params, 1.0)?;
    let stepped = scaler.step(&params, &mut || optimizer.step())?;
    scaler.update();
}
```

### Manual Dtype Conversion

For cases where you need explicit control over parameter dtypes rather
than relying on autocast:

```rust
// Cast all parameters to fp16
cast_parameters(&params, DType::Float16);

// Cast back to fp32
cast_parameters(&params, DType::Float32);
```

Parameters already at the target dtype are skipped (no-op).

## Eval Mode

Switch to eval mode for inference:

```rust
model.eval();
no_grad(|| {
    let output = model.forward(&input)?;
    // No graph built, no gradient tracking overhead.
    Ok(output)
})?;
```

## Training Housekeeping

The graph tracks step and epoch counts for schedulers and observation.
`end_step()` should be called after every training step (it detaches state
and increments the counter - see above). `end_epoch()` closes out the epoch:

```rust
model.end_step();   // detach state + increment step counter (call every batch)
model.end_epoch();  // increment epoch counter, reset step count
```

## Reproducibility

Seed libtorch and the CPU-side RNG before building the model - weight
initialization draws on the seed too:

```rust
manual_seed(42);                // libtorch: rand, randn, dropout, weight init
let mut rng = Rng::seed(42);    // CPU side: shuffling, augmentation
```

[Utilities - Reproducibility](08-utilities.md#reproducibility) has the full
recipe, `Rng`'s method surface, and CUDA re-seeding.

## Complete Example

```rust
use flodl::*;

fn main() -> Result<()> {
    manual_seed(42);

    // Build model.
    let model = FlowBuilder::from(Linear::new(2, 16)?)
        .through(GELU)
        .through(LayerNorm::new(16)?)
        .also(Linear::new(16, 16)?)
        .through(Linear::new(16, 2)?)
        .build()?;

    // Set up training.
    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.01);
    model.train();

    // Training loop (simplified - no data loader yet).
    let input_t = Tensor::randn(&[20, 2], TensorOptions::default())?;
    let target_t = Tensor::randn(&[20, 2], TensorOptions::default())?;

    for epoch in 0..50 {
        let input = Variable::new(input_t.clone(), true);
        let target = Variable::new(target_t.clone(), false);

        let pred = model.forward(&input)?;
        let loss = mse_loss(&pred, &target)?;

        optimizer.zero_grad();
        loss.backward()?;
        clip_grad_norm(&params, 1.0)?;
        optimizer.step()?;

        if epoch % 10 == 0 {
            println!("epoch {}  loss={:.6}", epoch, loss.item()?);
        }
    }

    // Eval.
    model.eval();
    let test_input = Tensor::from_f32(&[0.5, 0.3], &[1, 2], Device::CPU)?;
    let pred = no_grad(|| {
        model.forward(&Variable::new(test_input, false))
    })?;
    println!("pred: {:?}", pred.data().to_f32_vec()?);

    Ok(())
}
```

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Modules](03-modules.md) | Next: [The Graph Builder](05-graph-builder.md)
