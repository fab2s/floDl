---
permalink: /guide/0.8.x/modules
channel: 0.8.x
sitemap: true
layout: guide
title: "Modules"
prev_url: /guide/0.8.x/autograd
prev_title: "Autograd"
next_url: /guide/0.8.x/training
next_title: "Training"
anchors:
  - id: "the-module-trait"
    title: "The Module Trait"
  - id: "linear"
    title: "Linear"
  - id: "convolutions"
    title: "Convolutions"
  - id: "pooling"
    title: "Pooling"
  - id: "normalization"
    title: "Normalization"
  - id: "dropout"
    title: "Dropout"
  - id: "padding"
    title: "Padding"
  - id: "embedding"
    title: "Embedding"
  - id: "recurrent-layers"
    title: "Recurrent Layers"
  - id: "attention"
    title: "Attention"
  - id: "bilinear"
    title: "Bilinear"
  - id: "activations"
    title: "Activations"
  - id: "traineval-mode"
    title: "Train/Eval Mode"
  - id: "optional-module-traits"
    title: "Optional Module Traits"
  - id: "putting-it-together"
    title: "Putting it together"
  - id: "composing-modules-manually"
    title: "Composing Modules Manually"
last_modified_at: 2026-07-30T23:28:37+02:00
---

# Modules

The `nn` module provides neural network layers, activations, and the `Module`
trait that unifies them all. Modules compose naturally - a model is a Module
that contains other Modules.

This tutorial builds on [Automatic Differentiation](/guide/0.8.x/autograd).

> **Runnable examples**: [`regression`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/regression) builds
> linear and logistic regression out of `Linear` plus the right loss, and
> [`bio/dnaseq_conv1d`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/bio/dnaseq_conv1d) is a `Conv1d` motif
> classifier.

## The Module Trait

Every layer in floDl implements this trait:

```rust
pub trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable>;

    fn parameters(&self) -> Vec<Parameter> { vec![] }
    fn name(&self) -> &str { "module" }
    fn sub_modules(&self) -> Vec<Rc<dyn Module>> { vec![] }
    fn move_to_device(&self, _device: Device) {}
    fn set_training(&self, _training: bool) {}
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { None }
}
```

`forward` takes an input variable and returns an output variable. `parameters`
returns all learnable weights. Modules with no learnable parameters (like
activations) return an empty vec.

## Linear

Fully connected layer: `y = x @ W^T + b`.

```rust
let linear = Linear::new(784, 128)?;
```

Weights are Kaiming-initialized (suitable for ReLU). Input shape:
`[batch, in_features]`. Output shape: `[batch, out_features]`.

```rust
let output = linear.forward(&input)?;  // [batch, 784] -> [batch, 128]
```

### Builder options

```rust
// Without bias
let linear = Linear::no_bias(784, 128)?;

// On a specific device
let linear = Linear::on_device(784, 128, Device::CUDA(0))?;
```

## Convolutions

### Conv1d

1D convolution over `[N, C, L]` inputs. Same builder pattern as Conv2d.

```rust
let conv = Conv1d::new(1, 16, 3)?;  // in=1, out=16, kernel=3

// Fluent builder for full control
let conv = Conv1d::configure(3, 16, 5)
    .with_stride(2)
    .with_padding(2)
    .on_device(Device::CUDA(0))
    .done()?;
```

### Conv2d

2D convolution over `[N, C, H, W]` inputs.

```rust
let conv = Conv2d::new(3, 64, 3)?;  // in=3, out=64, kernel=3 (stride=1, padding=0)

// Fluent builder
let conv = Conv2d::configure(3, 64, 3)
    .with_padding(1)
    .with_stride(2)
    .done()?;

// Full control
let conv = Conv2d::build(3, 64, 3, true, [1,1], [1,1], [1,1], 1, Device::CPU)?;
```

### Conv3d

3D convolution over `[N, C, D, H, W]` inputs. For volumetric data (video, 3D medical).

```rust
let conv = Conv3d::new(1, 32, [3, 3, 3])?;

let conv = Conv3d::configure(1, 32, [3, 3, 3])
    .with_padding([1, 1, 1])
    .done()?;
```

### Transpose Convolutions

Transpose (deconvolution) variants for upsampling:

```rust
let deconv1d = ConvTranspose1d::new(16, 1, 3)?;
let deconv2d = ConvTranspose2d::new(64, 3, 4)?;
let deconv3d = ConvTranspose3d::new(32, 1, [3, 3, 3])?;
```

## Pooling

### MaxPool2d / AvgPool2d

2D pooling over `[N, C, H, W]` inputs. Stride defaults to kernel size.

```rust
let pool = MaxPool2d::new(2);                            // kernel=2, stride=2
let pool = MaxPool2d::with_stride(3, 2).padding(1);     // kernel=3, stride=2, padding=1
let output = pool.forward(&input)?;                      // [B, C, H, W] -> [B, C, H/2, W/2]

let pool = AvgPool2d::new(2);                            // average pooling
let pool = AvgPool2d::with_stride(3, 2).padding(1).count_include_pad(false);
```

No learnable parameters. Commonly paired with Conv2d + BatchNorm2d:

```rust
let model = FlowBuilder::from(Conv2d::new(3, 64, 3)?)
    .through(BatchNorm2d::new(64)?)
    .through(ReLU)
    .through(MaxPool2d::new(2))
    .build()?;
```

### MaxPool1d / AvgPool1d

1D pooling over `[N, C, L]` inputs, for sequence and signal processing:

```rust
let pool = MaxPool1d::new(2);
let pool = AvgPool1d::with_stride(3, 2).padding(1);
```

### Adaptive Pooling

Output a fixed spatial size regardless of input dimensions:

```rust
// As a free function
let pooled = adaptive_avg_pool2d(&input, [1, 1])?;  // [B, C, H, W] -> [B, C, 1, 1]

// As a module
let pool = AdaptiveMaxPool2d::new(7, 7);             // fixed 7x7 output
let pool = AdaptiveAvgPool2d::new([1, 1]);           // global avg (ResNet head)
let output = pool.forward(&input)?;
```

### PixelShuffle / PixelUnshuffle

Rearrange elements for sub-pixel convolution (super-resolution):

```rust
let shuffle = PixelShuffle::new(2);    // [B, C*4, H, W] -> [B, C, H*2, W*2]
let unshuffle = PixelUnshuffle::new(2); // inverse

let model = FlowBuilder::from(Conv2d::new(3, 48, 3)?)  // 48 = 3 * 4 (upscale=2)
    .through(PixelShuffle::new(2))                       // -> [B, 3, H*2, W*2]
    .build()?;
```

### Upsample

Resize spatial dimensions via interpolation:

```rust
let up = Upsample::new(&[64, 64], 1);  // output_size, mode (0=nearest, 1=bilinear, 2=bicubic)
```

### Unfold / Fold

Extract and reconstruct sliding local blocks (im2col / col2im as modules):

```rust
let unfold = Unfold::new([3, 3], [1, 1], [0, 0], [1, 1]);  // kernel, dilation, padding, stride
let fold = Fold::new([28, 28], [3, 3], [1, 1], [0, 0], [1, 1]);  // output_size, kernel, ...
```

## Normalization

### LayerNorm

Normalizes the last dimension. Commonly used in transformers.

```rust
let ln = LayerNorm::new(512)?;
let output = ln.forward(&input)?;  // [batch, 512] -> [batch, 512]
```

### RMSNorm

Root Mean Square normalization. Simpler and faster than LayerNorm - no mean
subtraction, just RMS scaling. Used in LLaMA, Gemma, and other modern
architectures:

```rust
let rn = RMSNorm::new(512)?;
let rn = RMSNorm::new(512)?.eps(1e-6);  // custom epsilon
let output = rn.forward(&input)?;
```

### BatchNorm

Normalizes over the batch dimension. Uses running statistics at inference.

```rust
// For fully-connected layers: input [batch, features]
let bn = BatchNorm::new(128)?;
let output = bn.forward(&input)?;  // [batch, 128] -> [batch, 128]

// For conv layers: input [batch, channels, height, width]
let bn2d = BatchNorm2d::new(64)?;
let output = bn2d.forward(&input)?;  // [B, 64, H, W] -> [B, 64, H, W]
```

Use `BatchNorm` after Linear layers and `BatchNorm2d` after Conv2d layers.
Both behave differently during training (batch statistics) vs. inference
(running statistics). They track `num_batches_tracked` and will error in eval
mode if no training has occurred - this catches a common silent bug.

See [Train/Eval Mode](#traineval-mode) below.

### GroupNorm

Normalizes over groups of channels. Independent of batch size - works well
with small batches where BatchNorm struggles:

```rust
let gn = GroupNorm::new(4, 16)?;   // 4 groups, 16 channels
let output = gn.forward(&input)?;  // [B, 16, H, W] -> [B, 16, H, W]
```

### InstanceNorm

Normalizes each channel independently. Standard for style transfer:

```rust
let inn = InstanceNorm::new(64, true)?;   // 64 features, affine=true
let output = inn.forward(&input)?;
```

## Dropout

Randomly zeroes elements during training. Uses inverted dropout so no
scaling is needed at inference.

```rust
let drop = Dropout::new(0.1);    // 10% drop probability - zeroes individual elements
let drop2d = Dropout2d::new(0.1); // drops entire channels (for conv features)
let adrop = AlphaDropout::new(0.1); // maintains self-normalizing property (for SELU networks)
let output = drop.forward(&input)?;
```

During inference, all dropout variants become identity functions.

## Padding

Padding modules for use in graph builder pipelines:

```rust
let pad = ZeroPad2d::new(1);                           // 1 pixel on all sides
let pad = ZeroPad2d::asymmetric(1, 1, 2, 2);           // left, right, top, bottom
let pad = ReflectionPad2d::new(1);                      // reflect at boundaries
let pad = ReflectionPad2d::asymmetric(1, 1, 2, 2);
```

## Embedding

Lookup table mapping integer indices to dense vectors.

```rust
let emb = Embedding::new(10000, 64)?;  // vocab=10000, dim=64
```

Input is a Variable wrapping an Int64 tensor:

```rust
// [batch, seq_len] -> [batch, seq_len, 64]
let output = emb.forward(&indices)?;
```

### EmbeddingBag

Computes bag-of-embeddings (sum, mean, or max of groups of indices).
Useful when input sequences have variable length and you need a fixed-size
output per bag:

```rust
let bag = EmbeddingBag::new(10000, 64)?;  // vocab=10000, dim=64

// indices: [total_indices], offsets: [num_bags] (start positions)
// mode: 0=sum, 1=mean, 2=max
let output = bag.forward_bag(&indices, &offsets, 1)?;  // [num_bags, 64]
```

## Recurrent Layers

### GRUCell / LSTMCell

Single-timestep cells. Backed by fused ATen kernels (~2 GPU kernels
instead of ~25-40):

```rust
let gru = GRUCell::new(128, 256)?;
let h = gru.forward_step(&x, None)?;      // first step: h initialized to zeros
let h = gru.forward_step(&x2, Some(&h))?; // subsequent steps

let lstm = LSTMCell::new(128, 256)?;
let state = lstm.forward_step(&x, None)?;             // first step
let state = lstm.forward_step(&x2, Some(&state))?;    // subsequent steps
```

### GRU / LSTM

Multi-layer sequence modules matching PyTorch's `nn.GRU` / `nn.LSTM`.
Process entire sequences and stack multiple layers. `forward_seq` uses
fused `at::lstm` / `at::gru` kernels (cuDNN-accelerated on CUDA) -
the full sequence is processed in a single kernel call, no per-timestep
dispatch overhead:

```rust
let gru = GRU::new(128, 256, 2)?;  // input=128, hidden=256, 2 layers
// Input: [seq_len, batch, input_size] (default) or [batch, seq_len, input_size] (batch_first)
let (output, h_n) = gru.forward_seq(&x, None)?;
// output: [seq_len, batch, hidden_size], h_n: [num_layers, batch, hidden_size]

let lstm = LSTM::new(128, 256, 2)?;
let (output, (h_n, c_n)) = lstm.forward_seq(&x, None)?;
// output: [seq_len, batch, hidden_size]
// h_n, c_n: [num_layers, batch, hidden_size]

// Batch-first ordering:
let gru = GRU::on_device(128, 256, 2, true, Device::CUDA(0))?;
let lstm = LSTM::on_device(128, 256, 2, true, Device::CUDA(0))?;
```

## Attention

### MultiheadAttention

Standard multi-head attention matching PyTorch's `nn.MultiheadAttention`.
Supports self-attention and cross-attention with optional masking:

```rust
let mha = MultiheadAttention::new(512, 8)?;  // embed_dim=512, 8 heads

// Self-attention (query = key = value)
let y = mha.forward(&x)?;                           // [B, seq, 512] -> [B, seq, 512]

// Cross-attention or masked attention
let y = mha.forward_ext(&query, &key, &value, Some(&mask))?;
```

### RotaryEmbedding (RoPE)

Rotary position embeddings encode positions as rotations of query/key
pairs, so attention scores depend only on relative offsets (LLaMA,
OLMo, and most modern decoder architectures). The sin/cos tables are
precomputed once; attach one to `MultiheadAttention` and it rotates
query and key after the head split, replacing additive position
embeddings:

```rust
// head_dim = embed_dim / num_heads = 512 / 8 = 64
let rope = RotaryEmbedding::new(64, 2048)?;          // head_dim, max_seq_len
let rope = RotaryEmbedding::on_device_theta(64, 2048, 500_000.0, device)?; // custom rope_theta

let mha = MultiheadAttention::new(512, 8)?.rotary(rope)?;
let y = mha.forward_ext(&x, &x, &x, Some(&causal_mask))?;  // decoder block, no pos_emb needed
```

For hand-rolled attention, `apply` rotates `[batch, heads, seq,
head_dim]` tensors directly:

```rust
let (q, k) = rope.apply(&q, &k)?;   // query/key may differ in heads/seq (GQA, cross-attention)
```

Cloning a `RotaryEmbedding` shares the tables (no copy), so one
instance serves every layer of a deep model.

## Bilinear

Bilinear transformation: `y = x1^T A x2 + b`. Useful for modeling
interactions between two feature sets:

```rust
let bi = Bilinear::new(128, 64, 32, true)?;  // in1=128, in2=64, out=32, bias=true
let y = bi.forward_bilinear(&x1, &x2)?;      // [B, 128] x [B, 64] -> [B, 32]
```

## Activations

Activation functions are also modules, making them composable in the graph
builder:

```rust
// Zero-sized types - no parameters, no allocation
ReLU              // max(0, x)
Sigmoid           // 1 / (1 + exp(-x))
Tanh              // hyperbolic tangent
GELU              // Gaussian Error Linear Unit
SiLU              // x * sigmoid(x), also called Swish
SELU              // scaled ELU - self-normalizing (pair with AlphaDropout)
Mish              // x * tanh(softplus(x))
Hardswish         // efficient Swish approximation
Hardsigmoid       // piecewise-linear sigmoid approximation
Identity          // pass-through

// Parameterized - take a config value at construction
LeakyReLU::new(0.01)         // max(x, slope * x)
ELU::new(1.0)                // alpha * (exp(x) - 1) for x < 0
Softplus::new(1.0, 20.0)     // smooth approximation of ReLU (beta, threshold)
Softmax::new(-1)              // softmax along dim
LogSoftmax::new(-1)           // log(softmax(x)) - numerically stable
Flatten::new(1, -1)           // flatten spatial dims (start_dim, end_dim)

// Learnable
let prelu = PReLU::new(1, Device::CPU)?;  // parametric ReLU (num_parameters, device)
```

All zero-sized activations compile to direct tensor calls with no overhead.

## Train/Eval Mode

Some modules (Dropout, BatchNorm) behave differently during training vs.
inference. The `set_training` method on Module controls this, and
convenience aliases `train()` / `eval()` make it concise:

```rust
model.eval();    // eval mode  - same as set_training(false)
model.train();   // training mode - same as set_training(true)
```

`train()` and `eval()` are convenience methods for `set_training(true)`
and `set_training(false)`. When using the graph builder,
`Graph::set_training(bool)` (and its aliases) propagates to all nodes
recursively.

## Optional Module Traits

### NamedInputModule

For modules that receive `using` references as a named map instead of
positional arguments:

```rust
pub trait NamedInputModule: Module {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable>;
}
```

### Stateful Module Methods

For modules with per-forward mutable state (attention location, counter),
override `reset()` on Module. Loops auto-call it before iterating:

```rust
impl Module for AttentionStep {
    fn reset(&self) {
        self.location.set(None); // clear stale state
    }
    // ...
}
```

For modules holding Variables across forward calls (recurrent state),
override `detach_state()` on Module. `graph.detach_state()` propagates
to all modules:

```rust
impl Module for RecurrentModule {
    fn detach_state(&self) {
        // break gradient chain on retained hidden state
    }
    // ...
}
```

## Putting it together

The catalogue above is a parts list; this is a whole model built from it. A
1-D CNN classifier, with the tensor shape after each stage so the arithmetic is
checkable by eye:

```rust
use flodl::*;

// Input: [B, 4, 200] - one-hot DNA, 4 channels, 200 positions.
let model = FlowBuilder::from(Conv1d::new(4, 32, 5)?)          // -> [B, 32, 196]
    .through(ReLU)
    .through(Conv1d::configure(32, 64, 5).with_padding(2).done()?)  // -> [B, 64, 196]
    .through(ReLU)
    .through(MaxPool1d::new(2))                                // -> [B, 64, 98]
    .through(Conv1d::configure(64, 128, 3).with_padding(1).done()?) // -> [B, 128, 98]
    .through(ReLU)
    .through(MaxPool1d::new(2))                                // -> [B, 128, 49]
    .through(Flatten::new(1, -1))                              // -> [B, 128*49]
    .through(Linear::new(128 * 49, 64)?)
    .through(ReLU)
    .through(Dropout::new(0.3))
    .through(Linear::new(64, 1)?)
    .through(Sigmoid)
    .build()?;
```

Three things this shows that the per-layer sections cannot:

- **Plain constructors and builders mix freely.** `Conv1d::new(4, 32, 5)?` when the
  defaults suit, `Conv1d::configure(...).with_padding(2).done()?` when they do not.
- **Activations are modules too**, so `ReLU` and `Sigmoid` sit in the chain rather
  than being applied separately.
- **`Dropout` needs no special handling here** - `model.train()` / `model.eval()`
  switches it, along with every `BatchNorm` in the graph.

This is the model from
[`bio/dnaseq_conv1d`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/bio/dnaseq_conv1d), which trains it end
to end on synthetic motif data. [The Graph Builder](/guide/0.8.x/graph-builder) covers
`FlowBuilder` itself - residuals, branches, and everything beyond a straight chain.

## Composing Modules Manually

Without the graph builder, you compose modules in plain Rust. Implement
`sub_modules()` to declare children - the framework then handles device
placement, training mode, and parameter collection:

```rust
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new() -> Result<Self> {
        Ok(MLP {
            fc1: Linear::new(784, 128)?,
            fc2: Linear::new(128, 10)?,
        })
    }
}

impl Module for MLP {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.fc1.forward(input)?;
        let x = x.relu()?;
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![Rc::new(self.fc1.clone()), Rc::new(self.fc2.clone())]
    }
}
```

This is the same pattern as PyTorch's `nn.Module` - declare children, let
the framework walk the tree. For anything involving residual connections,
parallel branches, loops, or conditional execution, the graph builder API
is more expressive and handles the wiring automatically.

