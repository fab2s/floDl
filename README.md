<p align="center">
  <img src="https://raw.githubusercontent.com/flodl-labs/flodl/main/docs/floDl.png" alt="floDl" width="640">
</p>

<h1 align="center">floDl</h1>

<p align="center">
A Rust-native deep learning framework built on libtorch.<br>
Same GPU kernels as PyTorch. No Python. No GIL. No GC. Just Rust.
</p>

<p align="center">
  <a href="https://flodl.dev"><img src="https://img.shields.io/badge/web-flodl.dev-6c8cff" alt="Website"></a>
  <a href="https://github.com/flodl-labs/flodl/actions"><img src="https://github.com/flodl-labs/flodl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/flodl"><img src="https://img.shields.io/crates/v/flodl.svg" alt="crates.io"></a>
  <a href="https://docs.rs/flodl"><img src="https://docs.rs/flodl/badge.svg" alt="docs.rs"></a>
  <a href="https://github.com/flodl-labs/flodl/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#if-you-know-pytorch-you-know-flodl">PyTorch Users</a> &bull;
  <a href="https://flodl.dev/thesis"><b>Thesis</b></a> &bull;
  <a href="#getting-started">Getting Started</a> &bull;
  <a href="#the-graph-builder">Graph Builder</a> &bull;
  <a href="#graph-tree-hierarchical-composition">Graph Tree</a> &bull;
  <a href="#the-training-experience">Training</a> &bull;
  <a href="#multi-gpu-training">Multi-GPU</a> &bull;
  <a href="#huggingface-integration">HuggingFace</a> &bull;
  <a href="#pytorch-parity">Parity</a> &bull;
  <a href="#performance">Benchmarks</a> &bull;
  <a href="https://github.com/flodl-labs/flodl/blob/main/ROADMAP.md">Roadmap</a> &bull;
  <a href="https://github.com/flodl-labs/flodl/blob/main/docs/pytorch_migration.md">Migration Guide</a> &bull;
  <a href="https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/13-data-loading.md">Data Loading</a>
</p>

---

> **What's new** - full re-architecture of the distributed layer to
> process-per-rank with elastic membership, plus the same single
> entry (`Trainer::builder(...).run()`) now scales from CPU to
> single-host multi-GPU (auto-promoted on 2+ visible GPUs) to
> multi-host clusters via `fdl.cluster.yml` or `ClusterBuilder`.
> ElChe collapsed to `ElCheMode` (five modes,
> default `NcclCadence`); convergence guard is now authoritative over
> `overhead_target`; per-epoch callbacks default to the fastest rank
> (free compute on heterogeneous rigs). New CLI: `fdl probe` (cluster
> readiness audit), `fdl nccl build` (libnccl source builder for
> heterogeneous-NCCL clusters), `fdl --gpus` (global GPU scope
> override), `fdl @cluster <cmd>` (multi-host fan-out). See the
> CHANGELOG and [DDP Reference](docs/ddp.md) for the full surface.

---

## If You Know PyTorch, You Know floDl

<table>
<tr><th>PyTorch</th><th>floDl</th></tr>
<tr><td>

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.GELU(),
    nn.LayerNorm(16),
    nn.Linear(16, 2),
)

pred = model(x)
loss = F.mse_loss(pred, target)
loss.backward()
optimizer.step()
```

</td><td>

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(LayerNorm::new(16)?)
    .through(Linear::new(16, 2)?)
    .build()?;

let pred = model.forward(&x)?;
let loss = mse_loss(&pred, &target)?;
loss.backward()?;
optimizer.step()?;
```

</td></tr>
</table>

Same concepts, same names, same GPU kernels underneath. The `?` operator
replaces silent failures with compile-time error handling. `Drop` replaces the
garbage collector. The [full migration guide](https://github.com/flodl-labs/flodl/blob/main/docs/pytorch_migration.md) covers
every op, module, and pattern.

> **New to Rust?** Read [Rust for PyTorch Users](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/00-rust-primer.md) - 10 patterns in 15 minutes.

## Getting Started

**With the CLI** (recommended, no Rust needed):

```bash
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl setup          # detect hardware, download libtorch, configure build environment
./fdl init my-proj   # scaffold a new project with training template
```

The `fdl` script auto-downloads a pre-compiled CLI binary (~750KB, pure Rust,
no libtorch dependency). It detects your GPUs, downloads the right libtorch
variant, and configures Docker or native builds. See the [full CLI
reference](docs/cli.md) for all commands.

**One-liner with Docker** (no Rust, no setup):

```bash
curl -sL https://flodl.dev/init.sh | sh -s my-project
cd my-project
./fdl build   # first build (~5 min, downloads libtorch)
./fdl run     # train the model
```

**Native** - [Rust](https://rustup.rs/) 1.85+ and libtorch:

```bash
./fdl libtorch download    # auto-detects CPU or CUDA
cargo add flodl && cargo build
```

For CUDA: `cargo add flodl --features cuda` + [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).

> **Using tch-rs or PyTorch C++?** `fdl` also works as a standalone
> libtorch manager outside of flodl: download any CPU/CUDA variant,
> switch between installs, compile from source for mixed GPU
> architectures (e.g. sm_61 + sm_120 in one build), and emit a
> machine-readable diagnostics report. No flodl buy-in required.
> See [docs/cli.md § Standalone](docs/cli.md#1-standalone-no-project-required)
> and the [`flodl-cli` crate](https://crates.io/crates/flodl-cli).

Both paths generate an annotated training template. Edit `src/main.rs` to
build your model:

```rust
use flodl::*;

let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(LayerNorm::new(16)?)
    .also(Linear::new(16, 16)?)     // residual connection
    .through(Linear::new(16, 2)?)
    .build()?;

let params = model.parameters();
let mut optimizer = Adam::new(&params, 0.01);
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
}
```

For framework-managed training (same code on CPU, single GPU, multi-GPU),
the universal `Trainer` takes a step closure and owns the loop:

```rust
// One step: forward + loss, returns the loss Variable.
fn train_step(model: &impl Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].clone(), false);
    mse_loss(&model.forward(&input)?, &target)
}

Trainer::builder(
    |dev| build_model_on(dev),
    |params| Adam::new(params, 0.01),
    train_step,
)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(num_epochs)
    .run()?
    .join()?;
```

For an explicit loop, `Ddp::wrap` gives per-rank gradient-sync control
and a fully-manual single-device loop is the lowest tier. See
[Tutorial 4: Training](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/04-training.md)
for all three tiers. (The self-driven `Trainer::setup` tier is deprecated
pending its cooperative-tier replacement.)

## The Graph Builder

floDl's fluent graph builder lets you describe complex architectures as
readable data flow - no boilerplate, no `nn.Module` subclassing.

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)                        // activation
    .through(LayerNorm::new(16)?)         // normalization
    .also(Linear::new(16, 16)?)           // residual connection
    .through(Linear::new(16, 2)?)         // output projection
    .build()?;
```

`build()` returns a `Graph` that implements `Module` - you can nest it
inside other graphs. Things get interesting when architectures get complex:

```rust
let g = FlowBuilder::from(encoder).tag("encoded")
    .split(modules![head_a, head_b, head_c]).merge(MergeOp::Mean)
    .loop_body(refinement_block).for_n(3).tag("refined")
    .gate(router, modules![expert_a, expert_b]).using(&["encoded"])
    .switch(selector, modules![light_path, heavy_path]).using(&["refined"])
    .through(StateAdd).using(&["memory"]).tag("memory")
    .loop_body(decoder).while_cond(halt_condition, 10)
    .through(output_head)
    .build()?;
```

Every construct - `split/merge`, `also`, `loop_body`, `gate`, `switch`, `map`,
`tag/using` - composes cleanly. Forward references (`using` before `tag`) carry
state across calls, enabling recurrent architectures without special-casing.

| Method | What it does |
|--------|-------------|
| `from(m).through(m)` | Linear chain |
| `also(m)` | Residual: `input + m(input)` |
| `fork(m)` | Side branch: capture output as tag, stream continues |
| `split(modules![...]).merge(op)` | Parallel branches, merged by `Add` or `Mean` |
| `tag(name)` / `using(refs)` | Named references - backward or forward (across calls) |
| `loop_body(body).for_n(n)` | Fixed iteration with BPTT |
| `loop_body(body).while_cond` / `until_cond` | Conditional loops |
| `gate(router, modules![...])` | Soft routing - weighted combination |
| `switch(selector, modules![...])` | Hard routing - only selected branch |
| `map(body).each()` / `.over(tag)` / `.slices(n)` | Element-wise, tagged, or sliced iteration |
| `input(names)` | Auxiliary graph inputs for multi-input architectures |

See the **[Graph Builder Tutorial](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/05-graph-builder.md)** and
the [full showcase](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/showcase/).

## Graph Tree: Hierarchical Composition

This is where floDl goes beyond PyTorch. Graphs nest inside graphs with
**label-path addressing** - dot-separated paths that let you reach into any
subgraph from the root. Train components independently, compose them into
larger architectures, and control training phases declaratively.

```rust
// Build components independently
let scan = FlowBuilder::from(scan_net).tag("hidden")
    .label("scan").build()?;

let read = FlowBuilder::from(read_net).tag("confidence")
    .label("read").build()?;

let encoder = FlowBuilder::from(scan)
    .through(read)
    .label("encoder").build()?;

// Compose into full model
let model = FlowBuilder::from(encoder)
    .through(classifier)
    .build()?;
```

### Dotted paths reach anywhere

Every tag and subgraph is addressable through dotted paths from the root:

```rust
model.validate_path("encoder")?;                 // -> Subgraph
model.validate_path("encoder.scan.hidden")?;      // -> Tag (three levels deep)
model.validate_path("encoder.read.confidence")?;  // -> Tag
```

### Declarative training phases

Freeze and thaw entire subtrees by path - no manual parameter iteration:

```rust
// Phase 1: train only the classifier, encoder is frozen
model.freeze("encoder")?;
let fresh_params = model.parameters();  // only unfrozen params
let mut opt = Adam::new(&fresh_params, 1e-3);
// ... train ...

// Phase 2: thaw scan, keep read frozen (it's proven)
model.thaw("encoder.scan")?;
let mut opt = Adam::with_groups()
    .group(&model.parameters_at("encoder.scan")?, 1e-4)  // low LR
    .group(&model.parameters_at("classifier")?, 1e-3)
    .build();
```

### Subgraph checkpoints

Train a component standalone, save it, load it into a larger model:

```rust
// Pre-trained encoder saved earlier
encoder.save_checkpoint("encoder_v1.fdl.gz")?;

// Load into the composed model - namespace + hash validated
model.load_subgraph_checkpoint("encoder", "encoder_v1.fdl.gz")?;
model.freeze("encoder.read")?;  // lock what's proven
```

### Cross-boundary observation

Metrics flow up through the tree automatically:

```rust
model.record_at("encoder.scan.loss", scan_loss)?;
model.record_at("encoder.read.accuracy", read_acc)?;
model.record_scalar("total_loss", total)?;

model.flush(&[]);  // single call flushes the entire tree

// Trends across boundaries - drive training decisions
if model.trend_at("encoder.scan.loss")?.stalled(10, 1e-4) {
    model.thaw("encoder.read")?;  // scan stalled, unfreeze read
}

// Monitor sees all metrics with dotted names automatically
monitor.log(epoch, elapsed, &model);
// -> total_loss, encoder.scan.loss, encoder.read.accuracy
```

This is progressive model composition: each component is trained and
validated independently before becoming a building block in a larger
architecture. Checkpoints, metrics, and training phases compose just like
the graphs themselves.

See the full **[Graph Tree Tutorial](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/10-graph-tree.md)**.

## The Training Experience

### Training Monitor

Drop-in monitor with adaptive ETA, resource tracking, and a live web
dashboard - no external dependencies, no separate process.

```rust
use flodl::monitor::Monitor;

let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;  // optional: live dashboard at http://localhost:3000

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();
    // ... training ...
    monitor.log(epoch, t.elapsed(), &model);  // sees entire graph tree
}
monitor.finish();
```

```
  epoch   1/100  loss=1.5264  [49ms  ETA 4.8s]
  epoch  10/100  loss=0.3817  [25ms  ETA 2.2s]  VRAM: 2.1/6.0 GB (82%)
  epoch  50/100  loss=0.0023  [24ms  ETA 1.2s]  VRAM: 2.1/6.0 GB (82%)
  epoch 100/100  loss=0.0012  [23ms]             VRAM: 2.1/6.0 GB (82%)
  training complete in 2.8s  | loss: 0.0012
```

<p align="center">
  <a href="https://flodl.dev/benchmark">
    <img src="https://raw.githubusercontent.com/flodl-labs/flodl/main/docs/dashboard.gif" alt="floDl live training dashboard - click for interactive version" width="800">
  </a>
</p>
<p align="center"><em><a href="https://flodl.dev/benchmark">Interactive benchmark dashboard</a> - real data from a 100-epoch training run</em></p>

The live dashboard updates via Server-Sent Events (no WebSocket, no npm),
tracks CPU/GPU/RAM/VRAM, and supports late join - open it mid-training and
all past epochs backfill instantly.

```rust
monitor.save_html("training_report.html");  // self-contained archive
monitor.export_csv("training.csv")?;         // for external analysis
```

### Observation and Trend Queries

Tags double as observation points. Collect metrics during training and use
trend queries to make programmatic training decisions:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = graph.forward(&input)?;
        graph.collect(&["hidden"])?;                 // from graph tag
        graph.record_scalar("loss", loss.item()?);   // external metric
    }
    graph.flush(&["hidden", "loss"]);

    // Programmatic training control
    if graph.trend("loss").stalled(5, 1e-4) {
        optimizer.set_lr(optimizer.lr() * 0.5);      // decay LR
    }
    if graph.trend("loss").converged(5, 1e-5) {
        break;                                        // early stopping
    }
}
```

| Method | What it does |
|--------|-------------|
| `g.collect(tags)` / `g.flush(tags)` | Batch -> epoch metric aggregation |
| `g.record_scalar(tag, value)` | Inject external metrics (loss, accuracy) |
| `g.trend(tag).slope(n)` | OLS slope over last n epochs |
| `g.trend(tag).stalled(n, tol)` | Is \|slope\| below tolerance? |
| `g.trend(tag).improving(n)` | Is loss decreasing? |
| `g.trend(tag).converged(n, tol)` | Is variance below tolerance? |
| `g.trends(tags).all_improving(n)` | Group queries across branches |

### Visualization

```rust
let svg = g.svg(Some("model.svg"))?;              // architecture diagram
g.svg_with_profile(Some("profile.svg"))?;          // timing heatmap
g.plot_html("training.html", &["loss", "head"])?;  // interactive curves
```

See the **[Training Monitor Tutorial](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/09-monitor.md)** and
the **[Observation example](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/observation/)**.

## Multi-GPU Training

The **same `Trainer::builder` call** scales from CPU to N GPUs on one
host to N GPUs across many hosts. On a host with 2+ visible CUDA
devices, floDl auto-promotes to process-per-rank fan-out
automatically - zero training-loop changes.

```rust
// Universal entry: works on CPU, 1 GPU, N GPUs single-host, N GPUs multi-host.
let handle = Trainer::builder(model_factory, optim_factory, train_step)
    .dataset(dataset)
    .batch_size(64)
    .num_epochs(50)
    .run()?;

let state: TrainedState = handle.join()?;
```

**ElChe - heterogeneous-rig cadence.** Mixed GPU generations? ElChe
auto-balances: the slowest GPU anchors the pace; faster ones process
proportionally more batches per averaging window, AllReduce overhead
stays bounded, the convergence guard vetoes anchor growth when
weight-space divergence rises. No configuration needed for the common
case.

**Five DDP modes, one line each.** `ElCheConfig::default()` is
`nccl_cadence()` (recommended NCCL default). Swap to any of the five
modes for A/B testing:

```rust
.elche(ElCheConfig::cpu_async().easgd_alpha(0.6))   // best-in-class on reference rig
.elche(ElCheConfig::nccl_cadence())                 // default; slow rank anchors, fast ranks fill the window
.elche(ElCheConfig::nccl_sync())                    // per-batch AllReduce baseline
.elche(ElCheConfig::cpu_cadence())                  // CPU-mediated cadence (no NVLink/P2P needed)
```

**Multi-host clusters.** Add an `fdl.cluster.yml` next to your
`fdl.yml`, or build the topology programmatically with
`ClusterBuilder`. Then:

```bash
fdl probe              # readiness gate: GPU + libtorch + NCCL + shared-data audit
fdl @cluster train     # SSHes each worker, pre-builds, fans out
```

Heterogeneous-rig support extends to per-host libtorch variants
(`precompiled/cu128` on one host, `builds/sm61-sm120` on another),
NCCL version-skew handling via `fdl nccl build` (builds a matching
libnccl for `LD_PRELOAD`), and elastic membership (ranks can die and
rejoin without aborting the run).

> **Invariant - no CUDA before `Trainer::run`.** User binaries must
> not touch libtorch's CUDA context in `main()` (no
> `cuda_device_count()`, no `Module::on_device(CUDA(_))`, no CUDA
> tensors). The launcher exits without training; touching CUDA there
> poisons spawned children's contexts on heterogeneous rigs. Use
> `flodl::sys::detect_gpus()` (CUDA-free) for pre-run GPU queries.

See the **[Multi-GPU Tutorial](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/11-multi-gpu.md)**,
**[Heterogeneous & Multi-Host DDP](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/12-async-ddp.md)**,
**[Data Loading Tutorial](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/13-data-loading.md)**, and
**[DDP Reference](https://github.com/flodl-labs/flodl/blob/main/docs/ddp.md)**.

### Validation suite - `ddp-bench`

The repo ships with [`ddp-bench/`](https://github.com/flodl-labs/flodl/tree/main/ddp-bench),
a workspace member that reproduces published training setups (Logistic /
MLP / LeNet-5 / ResNet-20 / Char-RNN / GPT-nano / Conv-AE on MNIST,
CIFAR-10, Shakespeare) to build scientifically valid solo baselines, then
measures DDP/ElChe convergence quality against them across all 8
backend × policy combinations:

```bash
fdl ddp-bench --list                       # list models and modes
fdl ddp-bench quick                        # 1-epoch smoke test
fdl ddp-bench validate                     # full sweep vs structured baselines
fdl ddp-bench --model gpt-nano --mode nccl-cadence --epochs 50 --lr-scale 2
fdl ddp-bench --report runs/report.md      # convergence report from saved runs
```

Every run produces a high-frequency `Timeline` (CPU/GPU utilization, sync
events, anchor changes, idle gaps) saved as JSON / CSV / interactive HTML
under `runs/<model>/<mode>/`.

### Built-in datasets

The framework ships ready-to-use parsers for common benchmarks (all
implement `BatchDataSet`, plug straight into `DataLoader::builder`):

```rust
use flodl::data::datasets::{Cifar10, Mnist, Shakespeare};

let mnist = Mnist::parse(&images_gz, &labels_gz)?;
let cifar = Cifar10::parse(&[&batch1, &batch2, /* ... */])?;
let text  = Shakespeare::parse(&corpus, /*seq_len=*/ 128)?;
```

`ddp-bench` downloads and caches the underlying files on first run.

## HuggingFace Integration

The [`flodl-hf`](https://crates.io/crates/flodl-hf) sibling crate loads
real HuggingFace checkpoints directly into floDl, with no Python, no
ONNX detour, and no conversion script. Six BERT-family architectures
(BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, DeBERTa-v2) are wired
end-to-end with PyTorch-verified numerical parity across four task
heads each.

```rust
use flodl_hf::models::auto::AutoModelForSequenceClassification;

let clf = AutoModelForSequenceClassification::from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)?;

let results = clf.predict(&["I love this framework"])?;
for (label, score) in &results[0] {
    println!("{} ({:.3})", label, score);
}
```

`AutoModel` reads `config.json`, dispatches to the right family, and
returns a typed handle. Swap the repo id for `bert-base-uncased`,
`albert-base-v2`, `xlm-roberta-base`, `microsoft/deberta-v3-base`, or
any fine-tune on top of those, and the caller stays identical.

| Entry point | Task | Output |
|---|---|---|
| `AutoModel` | Backbone (hidden states) | `[batch, seq_len, hidden]` |
| `AutoModelForSequenceClassification` | Whole-text labels | `Vec<Vec<(label, score)>>` |
| `AutoModelForTokenClassification` | Per-token labels (NER) | `Vec<Vec<TokenPrediction>>` |
| `AutoModelForQuestionAnswering` | Extractive answer span | `Answer { text, start, end, score }` |
| `AutoModelForMaskedLM` | Fill-mask candidates | `Vec<(token, prob)>` (top-k) |

Three feature profiles cover deployment shapes: full Hub + tokenizer
(default), vision-only (Hub without the regex/unicode surface), and
offline `safetensors`-only (no network, no async runtime, no TLS).
Inside an existing flodl project, `fdl add flodl-hf --playground`
scaffolds a side crate with a runnable `AutoModel` example so you can
verify a real checkpoint loads before wiring it into your main code;
`fdl add flodl-hf --install` appends `flodl-hf = "=0.5.3"` to your
root `Cargo.toml`.

```bash
fdl add flodl-hf --playground   # try it: ./flodl-hf/ sandbox crate
fdl add flodl-hf --install      # wire it: pin to your root Cargo.toml
fdl flodl-hf classify           # runs AutoModel on a real fine-tune
```

Fine-tuning uses the same loop code on CPU, single GPU, or N GPUs:
`Trainer::setup_head` distributes the head transparently when more
devices are available, and `compute_loss(enc, labels)` mirrors HF
Python's `model(..., labels=...).loss` one-call shape. Round-trip the
trained head back out with `fdl flodl-hf export` then verify it loads
into HF Python's `AutoModelFor*` with `fdl flodl-hf verify-export`.

See the **[HuggingFace Integration Tutorial](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/14-flodl-hf.md)**
for the full feature matrix, per-family entry points, tokenizer usage,
local-disk loading, the fine-tune walkthrough, the export round-trip
recipe, and the 30-cell parity matrix.

## PyTorch Parity

floDl covers the modules, losses, and optimizers you actually use:

| Category | Count | Highlights |
|----------|------:|-----------|
| **NN Modules** | 30+ | `Linear`, `Conv1d`/`2d`/`3d` + transpose, `GRU`/`LSTM`, `MultiheadAttention`, `Bilinear`, all norms (`Layer`/`RMS`/`Group`/`Batch`/`Instance`), all pooling, `Embedding`/`EmbeddingBag`, `PixelShuffle`, `Upsample`, `Unfold`/`Fold` |
| **Activations** | 17 | `ReLU`, `LeakyReLU`, `ELU`, `GELU`, `SiLU`, `Mish`, `SELU`, `Softplus`, `Hardswish`, `PReLU`, `Softmax`, ... |
| **Losses** | 15 | MSE, CrossEntropy, BCE, NLL, CTC, Focal, Triplet, KLDiv, SmoothL1, Cosine, Hinge, Margin, Poisson, ... |
| **Optimizers** | 7 | `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad`, `RAdam`, `NAdam` - all with parameter groups |
| **Schedulers** | 8 | Step, Cosine, Exponential, MultiStep, OneCycle, Cyclic, Warmup (composable), Plateau |
| **Init** | 9 | Xavier, Kaiming, orthogonal, truncated normal, uniform, normal |
| **Tensor Ops** | 100+ | Full arithmetic, trig, reductions, shape, indexing, comparisons, fused ops |
| **Autograd** | 90+ | Differentiable backward for every op above |

Fused Adam/AdamW on CUDA (single kernel for all parameters). Fused gradient
clipping via foreach ops. Mixed precision with `AutocastGuard` + `GradScaler`.
CUDA Graphs for replay-based training.

The [full migration guide](https://github.com/flodl-labs/flodl/blob/main/docs/pytorch_migration.md) has side-by-side
code for every op, module, and pattern.

## Performance

Same CUDA kernels as PyTorch - the difference comes from what happens
*between* kernel launches. Ten models, ten interleaved rounds, locked GPU
clocks (RTX 5060 Ti, v0.3.0 vs PyTorch 2.10.0):

| Model | PyTorch | flodl | Delta |
|---|---:|---:|---:|
| transformer | 3183.0 ms | 2199.8 ms | **-31%** |
| mlp | 291.1 ms | 207.0 ms | **-29%** |
| residual_tower | 406.9 ms | 309.7 ms | **-24%** |
| feedback_fixed | 275.3 ms | 231.3 ms | **-16%** |
| gated_routing | 248.0 ms | 217.3 ms | **-12%** |
| iterative_refine | 230.7 ms | 206.0 ms | **-11%** |
| gru_seq | 1105.1 ms | 1057.5 ms | **-4%** |
| conv_autoenc | 398.2 ms | 395.3 ms | -1% |
| lstm_seq | 692.3 ms | 692.3 ms | 0% |
| convnet | 1298.0 ms | 1298.2 ms | 0% |

Wins 8 of 10, ties 2, zero regressions. The ties (convnet, lstm_seq) are
compute-bound - both frameworks saturate the GPU, confirming identical
CUDA kernels. The gap appears where framework overhead matters:
dispatch-bound architectures (transformer -31%, mlp -29%), graph routing
(residual_tower -24%), and recurrent loops (feedback_fixed -16%).

**[Benchmark Report](https://github.com/flodl-labs/flodl/blob/main/docs/benchmark.md)** |
[Interactive dashboard](https://flodl.dev/benchmark)

### Multi-GPU (DDP)

ResNet-20 on CIFAR-10, 200 epochs - heterogeneous GPUs (RTX 5060 Ti +
GTX 1060, 2.5x speed ratio). Published reference: 91.25%
([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6):

| Mode | Eval | vs Published | Time | vs Solo-0 |
|---|---:|---:|---:|---:|
| solo-0 (fast GPU only) | 91.66% | +0.41% | 3127s | - |
| nccl-cadence | **92.42%** | **+1.17%** | 2650s | 1.2x |
| cpu-async | **92.43%** | **+1.18%** | 2614s | 1.2x |
| cpu-cadence | **92.04%** | **+0.79%** | 2670s | 1.2x |

Every ElChe mode surpasses published accuracy while finishing faster
than the fast GPU alone. 200 epochs is where ElChe's proportional
scheduling has room to calibrate and shine - shorter models (logistic
through gpt-nano) confirm DDP convergence across architectures.

**[DDP Benchmark Report](https://github.com/flodl-labs/flodl/blob/main/docs/ddp-benchmark.md)** -
full results for 8 models across 9 DDP modes

## Why Rust for Deep Learning?

**Deterministic memory.** Python adds ~3-5 us of framework overhead per GPU
op. Go's GC can't manage VRAM - an [earlier Go implementation](https://github.com/fab2s/goDl)
required 5 phases of lifecycle management (refcounting, GC callbacks, VRAM
budgets, pending-free queues). Rust replaces all of that with
`impl Drop for Tensor`. Memory is freed the instant a tensor leaves scope.

**Zero-cost safety.** Every op returns `Result<T>` - no silent failures.
Ownership ensures tensors are freed exactly once. The borrow checker
prevents data races at compile time.

**Same GPU kernels.** floDl binds libtorch - the C++ library under
PyTorch. CUDA, cuBLAS, cuDNN are identical. floDl replaces the dispatch
path, autograd tracking, and graph execution.

## Features Reference

<details>
<summary><strong>Training Tools</strong></summary>

| Tool | What it does |
|------|-------------|
| `clip_grad_norm` / `clip_grad_value` | Fused gradient clipping (2 kernels total via foreach ops) |
| `save_checkpoint` / `load_checkpoint` | Named `.fdl` checkpoints, structural hash, partial loading, `LoadReport` |
| `save_state_file` / `load_state_file` | Optimizer state save/load (`.optim`, self-identifying header, gzip-aware, atomic) |
| `migrate_checkpoint` | Remap parameter names across versions |
| `migrate_optim_state_file` | Convert an old optimizer `.optim` file to the current format |
| `Parameter::freeze` / `unfreeze` | Per-parameter gradient control |
| `GradScaler` | Dynamic loss scaling for fp16 training |
| `cast_parameters` | Cast model parameters to any dtype |
| `CpuWorker` / `ModelSnapshot` | Background checkpoint saving |
| `CudaGraph` | Capture/replay training steps for fixed-shape models |

</details>

<details>
<summary><strong>Module Traits</strong></summary>

Beyond `forward`/`parameters`, `Module` provides optional methods the graph
recognizes automatically:

| Method | What happens |
|--------|-------------|
| `as_named_input()` | `using()` refs arrive as a named map |
| `reset()` | Loops auto-call before iterating - clears per-forward state |
| `detach_state()` | Break gradient chains on retained state |
| `sub_modules()` | Recursive device placement, training mode, parameter collection |

</details>

<details>
<summary><strong>Build Profiles</strong></summary>

```toml
# Optimize floDl in dev builds - your code stays fast to compile.
[profile.dev.package.flodl]
opt-level = 3

[profile.dev.package.flodl-sys]
opt-level = 3

# Release: cross-crate optimization for maximum throughput.
[profile.release]
lto = "thin"
codegen-units = 1
```

| Profile | flodl | Your code | Typical rebuild |
|---------|-------|-----------|-----------------|
| `cargo build` | `-O3` (cached) | `-O0` (fast) | < 2s |
| `cargo build --release` | `-O3` + LTO | `-O3` + LTO | full link |

</details>

<details>
<summary><strong>Multi-GPU (DDP)</strong></summary>

| Component | What it does |
|-----------|-------------|
| `Trainer::builder(...).run()` | Universal entry. Same call scales from CPU to multi-host cluster. |
| `Trainer::run(..., TrainerConfig)` | Config-bag form - same launcher, data-driven setup. |
| `Ddp::wrap(&model, device, rank, &rdv)` | Low-level per-rank gradient-sync for manual control (GAN/RL). |
| `ElCheMode` | `NcclSync/Cadence`, `CpuSync/Cadence/Async`. Default `NcclCadence`. |
| `ElCheConfig` | Anchor tuning, partition ratios, convergence guard, EASGD, meta-controller. |
| `TrainerConfig` | Umbrella: dataset, callbacks, checkpointing, resume, cluster topology. |
| `ClusterBuilder` | Programmatic cluster construction (mirrors `fdl.cluster.yml`). |
| `flodl::sys::detect_gpus` | CUDA-free GPU detection; canonical pre-`Trainer::run` query. |
| `TrendGuard` / `MsfGuard` / `NoGuard` | Convergence guards - TrendGuard is default. Guard is authoritative over `overhead_target`. |
| `EpochCallbackPolicy` | `Rank(global)` or `Fastest` (default - cost-aware, free-compute on heterogeneous rigs). |
| `NcclComms` / `NcclRankComm` / `NcclAbortHandle` | Low-level NCCL when you need it. Init-on-main + `split()` everywhere. |
| `CudaEvent` / `CudaStream` / `StreamGuard` | Async GPU-CPU pipeline, timing. |
| `Ddp::wrap` | Manual thread-based DDP primitive (per-rank gradient sync) for GAN / RL / explicit replica control. Production multi-GPU auto-promotes to process-per-rank instead. |

</details>

### Numerical Verification

Every differentiable path is verified against finite-difference gradients:
- 117 autograd op-level checks (every op + compositions)
- Module-level checks (every NN module, input + parameter gradients)
- Exact optimizer step verifications (SGD, Adam, AdamW, RMSprop, Adagrad, RAdam, NAdam)
- 1027 library tests, zero clippy warnings - all tests run on both CPU and CUDA

### Hardware Compatibility

Developed and tested from NVIDIA Pascal (GTX 1060 6GB) to Blackwell
(RTX 5060 Ti 16GB). PyTorch dropped Pascal support after 2.5.1 - floDl
links libtorch's stable C API, which supports every architecture the driver
supports. If `nvidia-smi` works, floDl trains on it.

## Documentation

### Choose your path

| Background | Start here |
|-----------|-----------|
| **New to Rust** | [Rust for PyTorch Users](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/00-rust-primer.md) - 10 patterns in 15 minutes |
| **Know Rust, new to DL** | [Tensors](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/01-tensors.md) then [Training](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/04-training.md) |
| **Know PyTorch** | [Porting Guide](https://github.com/flodl-labs/flodl/blob/main/docs/porting.md) (or `/port` with AI) then [Graph Builder](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/05-graph-builder.md) |
| **Scaling to multi-GPU** | [Multi-GPU Training](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/11-multi-gpu.md) then [Heterogeneous & Multi-Host DDP](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/12-async-ddp.md) |
| **Bringing a HuggingFace model** | [HuggingFace Integration](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/14-flodl-hf.md): load BERT, RoBERTa, DistilBERT, ALBERT, XLM-R, or DeBERTa-v2; classify, NER, QA, or fill-mask; fine-tune and round-trip back to the HF ecosystem |
| **Just show me code** | [`quickstart`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/quickstart/) or [`showcase`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/showcase/) |

### Tutorials

0. **[Rust for PyTorch Users](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/00-rust-primer.md)** - 10 Rust patterns in 15 minutes
1. **[Tensors](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/01-tensors.md)** - creation, ops, memory, CUDA
2. **[Autograd](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/02-autograd.md)** - variables, gradients, backward
3. **[Modules](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/03-modules.md)** - all layers, convolutions, RNNs, attention, normalization
4. **[Training](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/04-training.md)** - losses, optimizers, mixed precision, full loop
5. **[Graph Builder](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/05-graph-builder.md)** - fluent API from simple to complex
6. **[Advanced Graphs](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/06-advanced-graphs.md)** - forward refs, loops, gates, switches
7. **[Visualization](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/07-visualization.md)** - DOT/SVG, profiling heatmaps
8. **[Utilities](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/08-utilities.md)** - checkpoints, clipping, freezing, initialization, scheduling, verbosity-gated logging
9. **[Training Monitor](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/09-monitor.md)** - ETA, resource tracking, live dashboard
10. **[Graph Tree](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/10-graph-tree.md)** - hierarchical composition, freeze/thaw, subgraph checkpoints
11. **[Multi-GPU Training](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/11-multi-gpu.md)** - Trainer::run / Trainer::builder, process-per-rank auto-promote, ElChe, DataLoader integration
12. **[Heterogeneous & Multi-Host DDP](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/12-async-ddp.md)** - ElChe cadence, process-per-rank cluster, A/B testable backends
13. **[Data Loading](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/13-data-loading.md)** - DataLoader, resident/streaming modes, VRAM-aware prefetch, DDP integration
14. **[HuggingFace Integration](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/14-flodl-hf.md)** - load BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, DeBERTa-v2 checkpoints, AutoModel dispatch across four task heads (seqcls, NER, QA, MLM), fine-tune with `Trainer::setup_head`, round-trip export to the HF ecosystem, PyTorch parity

### Examples

- [`quickstart`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/quickstart/): build, train, and monitor a model with residual connections
- [`sine_wave`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/sine_wave/): sine regression with monitor, checkpoint round-trip
- [`regression`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/regression/): linear and logistic regression as `Linear` plus the right loss
- [`mixed_precision`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/mixed_precision/): float16 training with `GradScaler`
- [`transfer_learning`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/transfer_learning/): checkpoint, partial load, freeze, fine-tune
- [`schedulers`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/schedulers/): warmup + cosine + plateau composition
- [`observation`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/observation/): collect, flush, trend queries, early stopping
- [`showcase`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/showcase/): every graph builder method in one graph
- [`flowbuilder_residual`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/flowbuilder_residual/): MLP classifier with a tagged residual block via FlowBuilder
- [`auto_promote`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/auto_promote/): the same plain training code auto-scaling CPU → 1 GPU → N-GPU DDP, zero distributed code
- [`bio/dnaseq_conv1d`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/bio/dnaseq_conv1d/): DNA Conv1D motif classifier (synthetic ChIP-seq peak annotation)
- [`bio/dna_autoencoder`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/bio/dna_autoencoder/): DNA convolutional autoencoder with checkpoint round-trip

### Porting from PyTorch

- **[Porting Guide](https://github.com/flodl-labs/flodl/blob/main/docs/porting.md)** - module mapping, FlowBuilder patterns, training loop translation
- **[AI-assisted porting](https://github.com/flodl-labs/flodl/tree/main/ai/skills/port/)** - point any AI coding assistant at the skill guide for automated translation. With Claude Code: `/port my_model.py`
- **`fdl api-ref`** - generate a structured API reference for your flodl version. Used by AI tools and useful on its own.

### Architecture

```
+-----------------------------------------------------------+
|  User Code / Model Definitions                            |
+-----------------------------------------------------------+
|  monitor/  ETA, resource tracking, live web dashboard     |
+-----------------------------------------------------------+
|  graph/    Fluent builder, graph tree, execution, DOT/SVG |
+-----------------------------------------------------------+
|  data/     DataLoader, resident/streaming, prefetch       |
+-----------------------------------------------------------+
|  nn/       Modules, losses, optimizers, DDP, NCCL         |
+-----------------------------------------------------------+
|  autograd/ Reverse-mode AD, gradient tracking             |
+-----------------------------------------------------------+
|  tensor/   Owned tensors with Drop, CPU + CUDA            |
+-----------------------------------------------------------+
|  flodl-sys   FFI bindings to libtorch C++ shim            |
+-----------------------------------------------------------+
|  libtorch / CUDA / NCCL                                   |
+-----------------------------------------------------------+
```

## Story

floDl started as a question: what would a deep learning framework look like
if you designed it around Rust's ownership model instead of fighting a garbage
collector?

An [earlier attempt in Go](https://github.com/fab2s/goDl) proved the
architecture - the graph builder, the module system, the observation engine -
but hit a wall: Go's GC cannot manage GPU memory deterministically. That
required building five layers of memory management infrastructure on top of
the language, not with it.

Rust solved this at the language level. `impl Drop for Tensor` replaced
hundreds of lines of lifecycle management. The graph builder, module
composition, and design philosophy carried forward; the memory fights didn't.

## License

floDl is open-sourced software licensed under the [MIT license](https://github.com/flodl-labs/flodl/blob/main/LICENSE).
