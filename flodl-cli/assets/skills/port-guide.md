# PyTorch to flodl Porting Guide

Universal porting instructions for AI-assisted translation of PyTorch
scripts to flodl. Tool-agnostic: works with any AI coding assistant.

## Phase 0: Project Setup

Before porting code, make sure the user has a build environment. The `fdl`
CLI handles everything:

### Option A: Mounted libtorch (recommended)

```bash
fdl init my-model            # scaffold with mounted libtorch
cd my-model
./fdl setup                  # detect GPUs, download libtorch, build image
./fdl build                  # verify it compiles
```

This generates:
- `Cargo.toml` with flodl dependency and optimized profiles
- `src/main.rs` with a training template (replace with ported code)
- `fdl.yml.example` with build/test/run/shell/cuda-* commands (copied to gitignored `fdl.yml` on first use)
- `Dockerfile` and `docker-compose.yml` (builds run in Docker)
- `./fdl` bootstrap script (use it straight away, or `./fdl install` to get it on PATH)
- `.gitignore`

All builds run inside Docker, so the host only needs Docker.
No Rust installation required on the host.

### Option B: Standalone Docker (no libtorch on host)

```bash
fdl init my-model --docker   # libtorch baked into Docker image
cd my-model
./fdl build                  # everything happens in Docker
```

Same files as Option A, but the Dockerfile downloads libtorch during
image build. Heavier image but zero host dependencies beyond Docker.

### Option C: No fdl available

Install fdl first:

```bash
# From crates.io (requires Rust)
cargo install flodl-cli

# Or download pre-compiled binary (no Rust needed)
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl install                # make it global (~/.local/bin/fdl)
```

Then use Option A or B above.

### GPU setup

If the user has GPU(s), `fdl setup` auto-detects hardware and downloads
the right libtorch variant:
- Single GPU: downloads matching pre-built variant (cu126 or cu128)
- Mixed GPUs: builds libtorch from source for the exact architectures
- No GPU: downloads CPU variant

### Where the ported code goes

After scaffolding, replace `src/main.rs` with the ported flodl code.
The `fdl.yml` commands (`./fdl build`, `./fdl test`, `./fdl cuda-test`)
work immediately. For multi-file projects, add modules under `src/`
and update `Cargo.toml` if needed.

## Phase 1: Bootstrap API Knowledge

Before porting, get the current flodl API surface:

```bash
fdl api-ref --json > /tmp/flodl-api.json
```

If `fdl` is not available, explore the flodl source directly:
- `flodl/src/nn/` for modules, losses, optimizers, schedulers
- `flodl/src/tensor/` for tensor operations
- `flodl/src/autograd/` for Variable (differentiable wrapper)
- `flodl/src/graph/` for FlowBuilder (computation graphs)
- `flodl/src/data/` for DataLoader, DataSet traits
- `flodl/src/distributed/` for multi-GPU (Ddp)

Cache the reference by flodl version. Only refresh when the version changes.

## Phase 2: Classify the PyTorch Source

Read the entire PyTorch script and classify each block by intent:

| Intent | PyTorch pattern | Look for |
|--------|----------------|----------|
| Model definition | `class Model(nn.Module)` | `__init__`, `forward` |
| Loss function | `criterion = nn.MSELoss()` | loss instantiation |
| Optimizer | `optim.Adam(params, lr=...)` | optimizer instantiation |
| Scheduler | `StepLR(optimizer, ...)` | scheduler instantiation |
| Data loading | `DataLoader(dataset, ...)` | dataset, dataloader |
| Training loop | `for epoch in range(...)` | epoch loop, backward, step |
| Checkpointing | `torch.save(state_dict)` | save/load patterns |
| Inference | `model.eval(); with torch.no_grad()` | eval mode, no_grad |
| Device management | `.to(device)`, `.cuda()` | device transfers |
| Mixed precision | `torch.cuda.amp` | autocast, GradScaler |
| Distributed training | `DistributedDataParallel`, `DataParallel`, `torch.distributed`, `torchrun`, `mp.spawn` | `init_process_group`, `dist.barrier`, `dist.all_reduce`, DDP wrapping |

## Phase 3: Map by Intent

### Model Definition

**Simple sequential models:**

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

```rust
// flodl -- use FlowBuilder for graph-based composition
let model = FlowBuilder::from(Linear::new(784, 256)?)
    .through(ReLU::new())
    .through(Linear::new(256, 10)?)
    .build()?;
```

**Models with residual connections:**

```python
# PyTorch
class ResBlock(nn.Module):
    def forward(self, x):
        return x + self.layers(x)
```

```rust
// flodl -- .also() adds a residual branch
let block = FlowBuilder::from(Linear::new(d, d)?)
    .through(ReLU::new())
    .also(Linear::new(d, d)?)  // output = main + residual
    .build()?;
```

**Models with skip connections / cross-attention:**

```python
# PyTorch
class Model(nn.Module):
    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(x)
        return self.cross_attn(y, h)  # uses earlier hidden state
```

```rust
// flodl -- .tag() saves, .using() retrieves
let model = FlowBuilder::from(encoder)
    .tag("hidden")             // save encoder output
    .through(decoder)
    .through(cross_attn)
    .using(&["hidden"])        // cross_attn receives [decoder_out, hidden]
    .build()?;
```

**Multi-head / parallel branches:**

```python
# PyTorch
class MultiHead(nn.Module):
    def forward(self, x):
        a = self.head_a(x)
        b = self.head_b(x)
        return a + b
```

```rust
// flodl -- .split() + .merge()
let model = FlowBuilder::from(shared_encoder)
    .split(modules![head_a, head_b])
    .merge(MergeOp::Add)       // or Cat, Mul, Mean
    .build()?;
```

**Iterative refinement / recurrent blocks:**

```python
# PyTorch
for _ in range(3):
    x = self.refine(x)
```

```rust
// flodl -- .loop_body().for_n()
let model = FlowBuilder::from(encoder)
    .loop_body(refine_block).for_n(3)
    .build()?;
```

**Naming / tagging for observation and checkpoints:**

```rust
// Tags make outputs observable and checkpointable
let model = FlowBuilder::from(encoder)
    .tag("encoder_out")
    .through(decoder)
    .tag("decoder_out")
    .label("my_model")    // graph-level label for checkpoints
    .build()?;
```

### Layers / Modules

Direct mapping (same names, same semantics):

| PyTorch | flodl | Notes |
|---------|-------|-------|
| `nn.Linear(in, out)` | `Linear::new(in, out)?` | Returns Result |
| `nn.Conv2d(in, out, k)` | `Conv2d::configure(in, out, k).done()?` | Builder pattern |
| `nn.Conv2d(in, out, k, padding=1)` | `Conv2d::configure(in, out, k).with_padding(1).done()?` | |
| `nn.BatchNorm2d(n)` | `BatchNorm::new(n)?` | |
| `nn.LayerNorm(n)` | `LayerNorm::new(n)?` | |
| `nn.RMSNorm(n)` | `RMSNorm::new(n)?` | |
| `nn.GroupNorm(g, c)` | `GroupNorm::new(g, c)?` | |
| `nn.Dropout(p)` | `Dropout::new(p)` | |
| `nn.ReLU()` | `ReLU::new()` | |
| `nn.GELU()` | `GELU` | Unit struct, no parens |
| `nn.SiLU()` | `SiLU` | |
| `nn.Sigmoid()` | `Sigmoid` | |
| `nn.Tanh()` | `Tanh` | |
| `nn.Softmax(dim)` | `Softmax::new(dim)` | |
| `nn.Embedding(n, d)` | `Embedding::new(n, d)?` | |
| `nn.LSTM(in, h, layers)` | `LSTM::new(in, h, layers)?` | |
| `nn.GRU(in, h, layers)` | `GRU::new(in, h, layers)?` | |
| `nn.MultiheadAttention(d, h)` | `MultiheadAttention::new(d, h)?` | |
| `nn.MaxPool2d(k)` | `MaxPool2d::new(k)` | |
| `nn.AvgPool2d(k)` | `AvgPool2d::new(k)` | |
| `nn.Flatten()` | `Flatten::new()` | |
| `nn.PixelShuffle(s)` | `PixelShuffle::new(s)` | |
| `nn.Upsample(scale)` | `Upsample::new(scale)` | |

Device variants: every module has `::on_device(... , device)` constructors.

### Losses

flodl losses are functions, not structs:

| PyTorch | flodl |
|---------|-------|
| `nn.MSELoss()(pred, target)` | `mse_loss(&pred, &target)?` |
| `nn.CrossEntropyLoss()(pred, target)` | `cross_entropy_loss(&pred, &target)?` |
| `nn.BCELoss()(pred, target)` | `bce_loss(&pred, &target)?` |
| `nn.BCEWithLogitsLoss()(pred, target)` | `bce_with_logits_loss(&pred, &target)?` |
| `nn.L1Loss()(pred, target)` | `l1_loss(&pred, &target)?` |
| `nn.SmoothL1Loss(beta=1.0)(pred, target)` | `smooth_l1_loss(&pred, &target, 1.0)?` |
| `nn.KLDivLoss()(input, target)` | `kl_div_loss(&input, &target)?` |
| `nn.NLLLoss()(input, target)` | `nll_loss(&input, &target)?` |
| `FocalLoss(alpha, gamma)` | `focal_loss(&pred, &target, alpha, gamma)?` |

### Optimizers

| PyTorch | flodl |
|---------|-------|
| `optim.Adam(params, lr=1e-3)` | `Adam::new(&params, 1e-3)` |
| `optim.AdamW(params, lr=1e-3, weight_decay=0.01)` | `AdamW::new(&params, 1e-3, 0.01)` |
| `optim.SGD(params, lr=0.01, momentum=0.9)` | `SGD::new(&params, 0.01).momentum(0.9)` |
| `optim.RMSprop(params, lr=1e-3)` | `RMSprop::new(&params, 1e-3)` |
| `optim.RAdam(params, lr=1e-3)` | `RAdam::new(&params, 1e-3)` |
| `optim.NAdam(params, lr=1e-3)` | `NAdam::new(&params, 1e-3)` |

### Schedulers

| PyTorch | flodl |
|---------|-------|
| `StepLR(opt, step=30, gamma=0.1)` | `StepDecay::new(30, 0.1)` |
| `CosineAnnealingLR(opt, T_max)` | `Cosine::new(t_max)` |
| `OneCycleLR(opt, max_lr, epochs, steps)` | `OneCycleLR::new(max_lr, epochs, steps)` |
| `ReduceLROnPlateau(opt, patience=10)` | `Plateau::new(10)` |
| `ExponentialLR(opt, gamma=0.95)` | `ExponentialLR::new(0.95)` |

### Training Loop

```python
# PyTorch
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

```rust
// flodl -- with Graph
model.train();
for epoch in 0..num_epochs {
    for batch in loader.epoch(epoch) {
        let batch = batch?;
        let output = model.forward(&batch[0].into())?;
        let loss = mse_loss(&output, &Variable::new(batch[1].clone(), false))?;
        loss.backward()?;
        optimizer.step()?;
        optimizer.zero_grad();
    }
    scheduler.step(&mut optimizer);
}
```

```rust
// flodl -- with Graph + integrated data loader (simplest)
model.train();
for epoch in 0..num_epochs {
    for batch in model.epoch(epoch) {
        let batch = batch?;
        let pred = model.forward_batch(&batch)?;
        let loss = mse_loss(&pred, &batch["target"].into())?;
        model.step(&loss)?;  // backward + optimizer + zero_grad
    }
}
```

### Distributed Training (DDP)

flodl has ONE training entry that runs everywhere: `Trainer::builder(...).run()`
(chained form) or `Trainer::run(model_factory, optim_factory, train_fn, cfg)`
(config-bag form). It auto-detects visible CUDA devices: 0-1 GPUs train
inline on a single device; 2+ GPUs auto-promote to the process-per-rank DDP
path (the launcher re-execs your binary as per-rank children and drives the
cluster coordinator). The same code runs on CPU, one GPU, N GPUs on a host,
and multi-host clusters, with no `init_process_group`, no `torchrun`, no
`mp.spawn`, and no `DistributedSampler`.

```python
# PyTorch
dist.init_process_group("nccl", rank=rank, world_size=world_size)
model = model.to(rank)
model = DistributedDataParallel(model, device_ids=[rank])
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
```

```rust
// flodl -- the framework owns the loop (backward, optimizer, gradient sync).
// model_factory: one model per replica, built on the given device.
fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let model = FlowBuilder::from(/* ... */).build()?;
    Ok(Box::new(model))
}

// Canonical step closure: forward + loss, nothing else.
fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    cross_entropy_loss(&model.forward(&input)?, &target)
}

let handle = Trainer::builder(build_model, |p| Adam::new(p, 1e-3), train_step)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(num_epochs)
    .elche(ElCheConfig::nccl_cadence())   // DDP cadence x backend (the default)
    .run()?;                              // inline on 0-1 GPU, process-per-rank on 2+

let state = handle.join()?;               // trained params + buffers
```

**Key translations:**

| PyTorch | flodl |
|---------|-------|
| `dist.init_process_group(...)` | handled inside `Trainer::run` (auto-promote) |
| `DistributedDataParallel(model, device_ids=[rank])` | `Trainer::builder(build, opt, step).run()?` |
| `DistributedSampler(dataset)` | Built-in: the DataLoader is DDP-aware, partitions automatically |
| `sampler.set_epoch(epoch)` | Not needed (deterministic per-epoch partitioning) |
| `torchrun --nproc_per_node=N` / `mp.spawn` | Not needed (`Trainer::run` auto-promotes to process-per-rank) |
| `dist.all_reduce(tensor)` / `dist.barrier()` | handled inside the framework loop |

**Heterogeneous clusters:** flodl's ElChe cadence auto-detects per-GPU
speed and lets faster cards run ahead while the slow one anchors
synchronization. Tune it with `.elche(ElCheConfig::...)` (five modes:
`nccl_sync` / `nccl_cadence` / `cpu_sync` / `cpu_cadence` / `cpu_async`).

**Manual per-rank control (bypass tier):** for explicit gradient-sync /
parameter-broadcast control (GAN, RL, progressive), drop to
`Ddp::wrap(&model, device, rank, &rendezvous)?` and call `sync_params()`
/ `all_reduce_gradients()` yourself.

**Own the loop body (cooperative tier):** `Trainer::builder(...).into_worker()?`
returns a `Worker` (`next_plan()` / `next_batch()` / `step()` / `finish()`)
while the controller keeps cadence, partition, eval-election, and
checkpointing.

For the full DDP surface (policies, backends, convergence guard, metrics,
live monitor integration, troubleshooting), see `docs/ddp/01-reference.md`.

### Data Loading

```python
# PyTorch
dataset = MyDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

```rust
// flodl
let loader = DataLoader::from_dataset(MyDataset::new(data))
    .batch_size(32)
    .shuffle(true)
    .device(Device::cuda_if_available())
    .build()?;
```

The `DataSet` trait requires:
```rust
impl DataSet for MyDataset {
    fn len(&self) -> usize { self.data.len() }
    fn get(&self, index: usize) -> Vec<Tensor> {
        vec![self.data[index].clone(), self.labels[index].clone()]
    }
}
```

### Checkpointing

```python
# PyTorch
torch.save(model.state_dict(), "checkpoint.pt")
model.load_state_dict(torch.load("checkpoint.pt"))
```

```rust
// flodl (Graph-based)
model.save_checkpoint("checkpoint.fdl")?;
model.load_checkpoint("checkpoint.fdl")?;

// flodl (any Module)
save_checkpoint("checkpoint.fdl", &model)?;
load_checkpoint("checkpoint.fdl", &mut model)?;
```

### Device Management

```python
# PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tensor.to(device)
```

```rust
// flodl
let device = Device::cuda_if_available();
// Modules: use on_device() constructors
let layer = Linear::on_device(784, 256, device)?;
// Tensors: explicit transfer
let t = tensor.to_device(device)?;
```

### No-Grad Context

```python
# PyTorch
with torch.no_grad():
    output = model(input)
```

```rust
// flodl
let output = no_grad(|| model.forward(&input))?;
// or RAII guard
let _guard = NoGradGuard::new();
let output = model.forward(&input)?;
```

## Phase 4: Validate

After generating the port:

1. Run `cargo check` -- fix compilation errors
2. Common errors and fixes:
   - **Missing `?`**: flodl constructors return `Result`, add `?`
   - **Borrow issues**: use `&variable` not `variable` in loss functions
   - **Type mismatch**: `Variable::new(tensor, false)` to wrap a Tensor as non-differentiable
   - **Missing import**: add `use flodl::*;` at the top
3. Run `cargo test` if tests were ported
4. Run the training loop, verify loss decreases

## Rust-Specific Patterns

### Error Handling
- All constructors return `Result<T>`. Use `?` in functions that return `Result`.
- Main can return `Result<()>`: `fn main() -> flodl::tensor::Result<()>`

### Ownership
- Tensors use reference counting (cheap clone, shared storage)
- Variables wrap tensors with gradient tracking
- `&variable` for read access in loss functions
- `.clone()` = shallow copy (shares storage; UNLIKE PyTorch's deep `.clone()`)
- `.copy()` = deep copy (independent storage; the PyTorch `.clone()` equivalent). Use when you keep a value while the original may be mutated in place.
- `.detach()` to break gradient graph (also shares storage)

### Builder Pattern
Conv layers, data loaders, and some optimizers use builders:
```rust
let conv = Conv2d::configure(3, 64, 3)
    .with_padding(1)
    .with_stride(2)
    .on_device(device)
    .done()?;
```

### The `modules!` Macro
For `split()` which needs `Vec<Box<dyn Module>>`:
```rust
.split(modules![branch_a, branch_b, branch_c])
```

### Training Mode
```rust
model.train();   // train mode
model.eval();  // eval mode
```
