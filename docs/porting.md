# Porting from PyTorch

flodl is designed for PyTorch users. Same module names, same semantics,
same training loop structure. Most translations are mechanical. This
guide covers manual porting and AI-assisted porting.

## The fast path: AI-assisted porting

flodl ships with a porting skill that works with AI coding assistants.
The skill reads your PyTorch script, classifies each block by intent,
maps it to flodl equivalents, generates a complete Rust project, and
validates with `cargo check`.

**With Claude Code:**

```
/port my_model.py
```

**With any AI tool:**

Point it at `ai/skills/port/guide.md` in the flodl repo (or any flodl
project scaffolded with `fdl init`). The guide contains the complete
mapping and the process to follow.

The AI uses `fdl api-ref` to get the current API surface, so it stays
up to date across flodl versions.

## Project setup

Before porting, you need a build environment. The `fdl` CLI handles this:

```bash
# Install fdl (one time)
cargo install flodl-cli      # from crates.io
# or: curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl

# Scaffold a project
fdl init my-model            # generates Cargo.toml, Dockerfile, Makefile, etc.
cd my-model

# Detect hardware and download libtorch
fdl setup
```

All builds run inside Docker. You don't need Rust on your host machine.
For standalone Docker mode (libtorch baked into the image):

```bash
fdl init my-model --docker
```

See the [CLI documentation](cli.md) for full details.

## Module mapping

flodl uses the same names as PyTorch. The main differences are Rust syntax
(constructors return `Result`, builder pattern for conv layers) and the
Graph builder for model composition.

### Layers

| PyTorch | flodl |
|---------|-------|
| `nn.Linear(in, out)` | `Linear::new(in, out)?` |
| `nn.Conv2d(in, out, k, padding=1)` | `Conv2d::configure(in, out, k).with_padding(1).done()?` |
| `nn.BatchNorm2d(n)` | `BatchNorm::new(n)?` |
| `nn.LayerNorm(n)` | `LayerNorm::new(n)?` |
| `nn.Dropout(p)` | `Dropout::new(p)` |
| `nn.ReLU()` | `ReLU::new()` |
| `nn.GELU()` | `GELU` (erf form, default) - use `GELU::tanh()` for the tanh approximation |
| `nn.Embedding(n, d)` | `Embedding::new(n, d)?` |
| `nn.LSTM(in, h, layers)` | `LSTM::new(in, h, layers)?` |
| `nn.GRU(in, h, layers)` | `GRU::new(in, h, layers)?` |
| `nn.MultiheadAttention(d, h)` | `MultiheadAttention::new(d, h)?` |

Every module has an `::on_device(... , device)` variant for explicit
device placement.

For the full mapping (30+ modules, losses, optimizers, schedulers), see
`ai/skills/port/guide.md`.

### Losses

flodl losses are functions, not structs:

```rust
// PyTorch: criterion = nn.MSELoss(); loss = criterion(pred, target)
// flodl:
let loss = mse_loss(&pred, &target)?;
let loss = cross_entropy_loss(&pred, &target)?;
let loss = focal_loss(&pred, &target, alpha, gamma)?;
```

### Optimizers

```rust
let optimizer = Adam::new(&model.parameters(), 1e-3);
let optimizer = AdamW::new(&model.parameters(), 1e-3, 0.01);
let optimizer = SGD::new(&model.parameters(), 0.01).momentum(0.9);
```

## Model architecture: FlowBuilder

This is where flodl diverges from PyTorch in a good way. Instead of
writing a `forward()` method with imperative control flow, you describe
data flow declaratively with `FlowBuilder`:

### Sequential

```python
# PyTorch
model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
```

```rust
// flodl
let model = FlowBuilder::from(Linear::new(784, 256)?)
    .through(ReLU::new())
    .through(Linear::new(256, 10)?)
    .build()?;
```

### Residual connections

```python
# PyTorch: return x + self.layers(x)
```

```rust
// flodl: .also() adds a residual branch
let block = FlowBuilder::from(Linear::new(d, d)?)
    .through(ReLU::new())
    .also(Linear::new(d, d)?)
    .build()?;
```

### Skip connections / cross-attention

```python
# PyTorch: h = encoder(x); y = decoder(x); return cross_attn(y, h)
```

```rust
// flodl: .tag() saves, .using() retrieves
let model = FlowBuilder::from(encoder)
    .tag("hidden")
    .through(decoder)
    .through(cross_attn).using(&["hidden"])
    .build()?;
```

### Parallel branches

```python
# PyTorch: return head_a(x) + head_b(x)
```

```rust
// flodl: .split() + .merge()
let model = FlowBuilder::from(encoder)
    .split(modules![head_a, head_b])
    .merge(MergeOp::Add)
    .build()?;
```

### Iterative refinement

```python
# PyTorch: for _ in range(3): x = refine(x)
```

```rust
// flodl: .loop_body().for_n()
let model = FlowBuilder::from(encoder)
    .loop_body(refine_block).for_n(3)
    .build()?;
```

### Tags for observation and checkpoints

Tags make intermediate outputs observable and enable selective
checkpointing:

```rust
let model = FlowBuilder::from(encoder)
    .tag("encoder_out")        // observable, checkpointable
    .through(decoder)
    .tag("decoder_out")
    .label("my_model")         // graph-level label
    .build()?;
```

## Training loop

```python
# PyTorch
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()
        optimizer.step()
```

flodl ports the manual loop almost line-for-line, and also offers a
universal `Trainer` that can own the loop for you. Three tiers, same
code on CPU / single GPU / multi-GPU.

### Trainer::builder: framework-managed (universal)

You provide a step closure (forward + loss); the framework runs the
loop, backward, optimizer step, and gradient sync.

```rust
// Step closure: forward + loss, returns the loss Variable.
fn train_step(model: &impl Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    cross_entropy_loss(&model.forward(&input)?, &target)
}

let handle = Trainer::builder(
    |dev| build_model_on(dev),
    |params| Adam::new(params, 0.001),
    train_step,
)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(10)
    .run()?;
let state = handle.join()?;  // averaged params + buffers
```

### Trainer::setup: setup only, your loop

`Trainer::setup` runs device replication + optimizer setup +
training-mode toggle in one call; the loop stays yours.

```rust
let model = build_model()?;
Trainer::setup(&model, |dev| build_model_on(dev), |p| Adam::new(p, 0.001))?;

for epoch in 0..num_epochs {
    for batch in loader.epoch(epoch) {
        let batch = batch?;
        let pred = model.forward(&batch[0].into())?;
        let loss = cross_entropy_loss(&pred, &batch["label"].into())?;
        loss.backward()?;
        model.step()?;  // AllReduce + buffer sync + optimizer + zero_grad
    }
}
```

### Fully manual: closest port from PyTorch

```rust
// flodl
model.train();
for epoch in 0..num_epochs {
    for batch in loader.epoch(epoch) {
        let batch = batch?;
        let pred = model.forward(&batch[0].into())?;
        let loss = mse_loss(&pred, &Variable::new(batch[1].clone(), false))?;
        loss.backward()?;
        optimizer.step()?;
        optimizer.zero_grad();
    }
}
```

## Multi-GPU (DDP)

The `Trainer` tiers in [Training loop](#training-loop) above already cover
the multi-GPU story: both `Trainer::builder` and `Trainer::setup`
auto-detect available CUDA devices and fall back to single-GPU/CPU when
fewer than 2 GPUs are present. The same code runs on CPU, single GPU,
and multi-GPU with no process-group setup, no `torchrun`, no `mp.spawn`,
and no `DistributedSampler`.

For DDP-specific knobs (cadence × backend, ElChe tuning), pass an
`ElCheConfig` to `.elche(...)`:

```rust
let ddp = Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(10)
    .elche(ElCheConfig::nccl_cadence())  // default; five modes total
    .run()?;

let state = ddp.join()?;                // averaged params + buffers on CPU
```

ElChe cadence auto-detects heterogeneous GPU speeds and lets the
faster card run ahead while the slow one anchors synchronization.
The default mode is `NcclCadence` - anchor-based scheduling with
NCCL AllReduce at every cadence boundary. Other modes: `NcclSync`,
`CpuSync`, `CpuCadence`, `CpuAsync` (best-in-class on the reference
rig - genuine async via decoupled CPU averaging). See the
[DDP Reference](ddp.md) for the full mode surface, convergence guard,
metrics, and live-monitor wiring, and [DDP Benchmark](ddp-benchmark.md)
for results on mixed consumer hardware.

## Key differences from PyTorch

| Concept | PyTorch | flodl |
|---------|---------|-------|
| Error handling | Exceptions | `Result<T>` with `?` operator |
| Memory | Garbage collected | Reference counted (cheap clone) |
| Tensor copy | `.clone()` is a deep copy | `.clone()` is *shallow* (shares storage); use `.copy()` for a deep, independent copy |
| Model composition | `nn.Sequential` / manual `forward()` | `FlowBuilder` (declarative data flow) |
| Training mode | `model.train()` | `model.train()` |
| Eval mode | `model.eval()` | `model.eval()` |
| No-grad | `with torch.no_grad():` | `no_grad(\|\| { ... })` or `NoGradGuard::new()` |
| Device | `.to(device)` / `.cuda()` | `::on_device(... , device)` constructors |
| Checkpoint format | `.pt` (pickle) | `.fdl` (binary, architecture-validated) |
| Losses | Struct instances | Free functions |
| Conv options | Constructor kwargs | Builder pattern (`.with_padding()`, `.done()`) |

## Further reading

- [Full porting guide](https://github.com/flodl-labs/flodl/blob/main/ai/skills/port/guide.md) (30+ modules, all patterns)
- [API reference](cli.md#fdl-api-ref) (via `fdl api-ref`)
- [Graph builder tutorial](tutorials/05-graph-builder.md)
- [Training tutorial](tutorials/04-training.md)
- [CLI documentation](cli.md) (project setup, libtorch management)
