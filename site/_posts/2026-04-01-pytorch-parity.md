---
title: "Everything you use in PyTorch, now in Rust"
subtitle: "30+ modules, 15 losses, 7 optimizers, 100+ tensor ops, 769 tests - flodl reaches PyTorch parity"
date: 2026-04-01
description: "flodl's parity release covers the full PyTorch API surface that DL researchers actually use: every conv, every norm, every pooling, every loss, every optimizer. Same names, same semantics, up to 31% faster."
---

I've been tracking a list. Every time I hit a PyTorch op that didn't exist
in flodl, it went on the list. Every time someone asked "does it have X?",
it went on the list. The list got long.

This release empties the list.

## What was missing

flodl started with the modules I needed for my own research: `Linear`,
`Conv2d`, `GRUCell`, `LSTMCell`, `Adam`, `LayerNorm`, `BatchNorm`. Enough
to train serious models. But "enough for me" isn't "enough for you."

A researcher porting a transformer needs `MultiheadAttention`. Someone doing
audio needs `Conv1d`. Object detection needs `focal_loss`. Super-resolution
needs `PixelShuffle`. Self-normalizing networks need `SELU` + `AlphaDropout`.
Sequence models need full `GRU` and `LSTM`, not just the cell variants.

Each gap was small. Together, they were the reason someone would look at
flodl, nod, and go back to PyTorch.

## What's here now

### Modules: 30+ and counting

**Convolutions** - the full family:

```rust
let conv = Conv1d::configure(3, 16, 5).with_stride(2).with_padding(2).done()?;
let conv = Conv2d::configure(3, 64, 3).with_padding(1).done()?;
let conv = Conv3d::configure(1, 32, [3,3,3]).with_padding([1,1,1]).done()?;

// And the transpose variants for upsampling
let deconv = ConvTranspose1d::new(16, 3, 5)?;
let deconv = ConvTranspose2d::new(64, 3, 4)?;
let deconv = ConvTranspose3d::new(32, 1, [3,3,3])?;
```

All share the same fluent builder pattern: `.with_stride()`, `.with_padding()`,
`.with_dilation()`, `.with_groups()`, `.on_device()`. If you've used one,
you've used them all.

**Normalization** - every variant:

```rust
let ln = LayerNorm::new(512)?;           // transformers
let rn = RMSNorm::new(512)?;             // LLaMA, Gemma
let gn = GroupNorm::new(4, 16)?;         // small batches
let bn = BatchNorm2d::new(64)?;          // conv nets
let inn = InstanceNorm::new(64, true)?;   // style transfer
```

`RMSNorm` was the most requested. It's simpler than `LayerNorm` (no mean
subtraction) and faster, the go-to for modern LLM architectures.

**Recurrent** - full sequence modules, not just cells:

```rust
// Multi-layer GRU matching nn.GRU exactly
let gru = GRU::new(128, 256, 2)?;  // 2 layers
let (output, h_n) = gru.forward_seq(&x, None)?;

// Multi-layer LSTM matching nn.LSTM
let lstm = LSTM::new(128, 256, 2)?;
let (output, (h_n, c_n)) = lstm.forward_seq(&x, None)?;
```

`GRUCell` and `LSTMCell` were always there, now you don't have to write
the loop yourself.

**Attention:**

```rust
let mha = MultiheadAttention::new(512, 8)?;
let y = mha.forward(&x)?;                              // self-attention
let y = mha.forward_ext(&query, &key, &value, Some(&mask))?;  // cross-attention
```

**Pooling** - the complete set:

`MaxPool1d`, `MaxPool2d`, `AvgPool1d`, `AvgPool2d`, `AdaptiveMaxPool2d`,
`PixelShuffle`, `PixelUnshuffle`, `Upsample`, `Unfold`, `Fold`. Plus
`adaptive_avg_pool2d` as a free function (it was already there).

And: `Bilinear`, `EmbeddingBag`, `AlphaDropout`, `ZeroPad2d`,
`ReflectionPad2d`.

### Activations: 17

The originals (`ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`) are joined by:

```rust
LeakyReLU::new(0.01)    // negative slope
ELU::new(1.0)           // exponential linear
Softplus::new(1.0, 20.0) // smooth ReLU
Mish                     // x * tanh(softplus(x))
SELU                     // self-normalizing
Hardswish                // efficient mobile Swish
Hardsigmoid              // piecewise-linear sigmoid
PReLU::new(1, device)?   // learnable slope
Softmax::new(-1)
LogSoftmax::new(-1)
Flatten::new(1, -1)
```

All zero-sized types compile to direct tensor calls. No allocation, no
indirection, no overhead.

### Losses: 15

The original six grew to fifteen:

| Loss | Use case |
|------|----------|
| `mse_loss` | Regression |
| `l1_loss` | Robust regression |
| `smooth_l1_loss` | Huber (outlier-resistant) |
| `cross_entropy_loss` | Classification |
| `nll_loss` | After log_softmax |
| `bce_loss` | Binary (from probabilities) |
| `bce_with_logits_loss` | Binary (numerically stable) |
| `kl_div_loss` | Distribution matching |
| `focal_loss` | Class imbalance (object detection) |
| `ctc_loss` | Sequence alignment (speech, OCR) |
| `triplet_margin_loss` | Metric learning |
| `cosine_embedding_loss` | Similarity learning |
| `hinge_embedding_loss` | SVM-style binary |
| `margin_ranking_loss` | Pairwise ranking |
| `poisson_nll_loss` | Count data |

Each is a single function call returning a differentiable `Variable`.

### Optimizers: 7

```rust
let opt = SGD::new(&params, 0.01, 0.9);
let opt = Adam::new(&params, 0.001);
let opt = AdamW::new(&params, 0.001, 0.01);
let opt = RMSprop::new(&params, 0.01);
let opt = Adagrad::new(&params, 0.01);
let opt = RAdam::new(&params, 0.001);   // rectified Adam - no warmup needed
let opt = NAdam::new(&params, 0.001);   // Nesterov-accelerated Adam
```

All support parameter groups for per-group learning rates. Adam and AdamW
automatically use fused CUDA kernels when parameters live on GPU, single
kernel launch for all parameters.

### Schedulers: 8

```rust
let s = StepDecay::new(0.001, 30, 0.1);
let s = CosineScheduler::new(0.001, 1e-6, 100);
let s = ExponentialLR::new(0.001, 0.95);
let s = MultiStepLR::new(0.001, &[30, 60, 90], 0.1);
let s = OneCycleLR::new(0.01, 1000);     // super-convergence
let s = CyclicLR::new(1e-4, 1e-2, 500);  // triangular wave
let s = WarmupScheduler::new(inner, 0.001, 10);  // composable wrapper
let mut s = PlateauScheduler::new(0.001, 5, 0.1, 1e-6);  // reactive
```

Schedulers are pure functions, `s.lr(step)` returns the LR, you set it.
No hidden optimizer coupling. `WarmupScheduler` wraps any other scheduler.

### Initialization: 9

```rust
let w = kaiming_uniform(&[out, inp], inp, 0.0, device)?;
let w = kaiming_normal(&[out, inp], inp, 0.0, device)?;
let w = xavier_uniform(&[out, inp], inp, out, device)?;
let w = xavier_normal(&[out, inp], inp, out, device)?;
let w = orthogonal(&[out, inp], 1.0, device)?;
let w = trunc_normal(&[out, inp], 0.0, 0.02, -2.0, 2.0, device)?;
let w = uniform(&[out, inp], -0.1, 0.1, device)?;
let w = normal(&[out, inp], 0.0, 0.01, device)?;
let w = uniform_bias(inp, &[out], device)?;
```

All at the crate root, `use flodl::*` and go. `orthogonal` for RNNs,
`trunc_normal` for Vision Transformers.

### Tensor ops: 100+

The full list is in the [migration guide](https://github.com/flodl-labs/flodl/blob/main/docs/pytorch_migration.md), but highlights:

- **Trig**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- **Numerically stable**: `log1p`, `expm1`, `log2`, `log10`, `erf`, `erfc`
- **Fused**: `addmm`, `addcmul`, `addcdiv`, `lerp`
- **Shape**: `flip`, `roll`, `diagonal`, `movedim`, `tile`, `split`, `unbind`, `meshgrid`
- **Reductions**: `prod`, `cumsum`, `logsumexp`
- **Similarity**: `cosine_similarity`, `normalize`, `masked_fill`

Every tensor op has a differentiable autograd path (90+ backward implementations).

## The test wall

769 tests. Every test runs on both CPU and CUDA. Zero clippy warnings.

The parity push alone added 165 tests:

- **55 autograd gradient checks** - finite-difference verification for every
  new differentiable op
- **60+ module tests** - forward shape, backward gradient, builder options,
  edge cases (batch-size-one GroupNorm, BatchNorm rejecting eval without
  training, SmoothL1 rejecting negative beta)
- **20+ loss tests** - including "focal reduces to cross-entropy at gamma=0"
  and "triplet loss is zero when negative is far"
- **7 mixed precision tests** - GradScaler growth, backoff, inf detection
- **6 gradient clipping tests** - fused foreach path verification

The gradient checks matter most. Every differentiable path is verified against
finite-difference approximation. If the analytical gradient disagrees with
the numerical gradient, the test fails. No untested backward paths.

## What this means

If you've been waiting for "does it have X?", the answer is probably yes now.
The [migration guide](https://github.com/flodl-labs/flodl/blob/main/docs/pytorch_migration.md) has
side-by-side code for every op, every module, every pattern. Same names,
same semantics.

And the [benchmarks keep improving](/blog/benchmark-update): up to 31%
faster, winning 8 of 10 architectures with zero regressions. Same CUDA
kernels, less overhead between them.

```bash
# Try it
curl -sL https://flodl.dev/init.sh | sh -s my-project
cd my-project && make run
```

The full changelog is on [GitHub](https://github.com/flodl-labs/flodl/blob/main/CHANGELOG.md).

---

*flodl is open source (MIT). Star us on [GitHub](https://github.com/flodl-labs/flodl),
try the [tutorials](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/01-tensors.md),
or dive into the [migration guide](https://github.com/flodl-labs/flodl/blob/main/docs/pytorch_migration.md).*
