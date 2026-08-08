# PyTorch → flodl Migration Guide

A side-by-side reference for PyTorch users learning flodl.

> **Want the fast path?** flodl ships with an AI porting skill that reads
> your PyTorch script and generates a complete flodl project. Run `/port
> my_model.py` in Claude Code, or see the [Porting Guide](porting.md).
> You can also run `fdl api-ref` to get the full, up-to-date API surface
> in your terminal.

## Imports

In PyTorch, `import torch` gives you almost everything. Rust uses explicit
imports - flodl re-exports its full API from the crate root for convenience:

```rust
use flodl::*;  // brings in Tensor, Variable, all nn modules, graph builder, etc.
```

**Or import selectively by module:**

```rust
use flodl::tensor::{Tensor, TensorOptions, DType, Device};
use flodl::autograd::{Variable, no_grad};
use flodl::nn::{Linear, Adam, mse_loss, Module, Parameter};
use flodl::graph::{FlowBuilder, Graph, MergeOp};
```

| If you're doing... | You need |
|---------------------|----------|
| Defining a model with the graph builder | `nn`, `graph` |
| Writing a training loop | `nn`, `autograd`, `tensor` |
| Creating raw tensors | `tensor` |
| Everything (main training script) | `use flodl::*` |

**How it maps to PyTorch:**

| PyTorch | flodl | What's in it |
|---------|-------|-------------|
| `torch.*` | `flodl::tensor` | Creation (`zeros`, `rand`, `arange`...), math ops, shape ops |
| `torch.autograd` | `flodl::autograd` | `Variable`, `no_grad`, gradient tracking |
| `torch.nn` | `flodl::nn` | Modules (`Linear`, `Conv2d`...), activations, losses |
| `torch.optim` | `flodl::nn` | Optimizers (`Adam`, `SGD`, `AdamW`), LR schedulers |
| *(no equivalent)* | `flodl::graph` | Fluent computation graph builder |

## Core Concepts

| PyTorch | flodl | Notes |
|---------|-------|-------|
| `torch.Tensor` | `Tensor` | Immutable, `Drop`-based VRAM cleanup, `Send`+`Sync` |
| `torch.autograd` | `Variable` | Wraps Tensor, tracks gradients via `Rc<RefCell>` |
| `torch.nn.Module` | `Module` trait | `forward(&self, &Variable) -> Result<Variable>` + `parameters()` |
| `model.train()` | `module.train()` | Called on individual modules |
| `model.eval()` | `module.eval()` | Disables dropout, freezes BatchNorm stats |
| `with torch.no_grad():` | `no_grad(\|\| { ... })` | RAII guard disables gradient tracking |

**Error handling:** flodl returns `Result<T>` instead of panicking. Use `?` to propagate errors:

```rust
let y = x.matmul(&w)?.add(&b)?.relu()?;
```

## Reproducibility

```python
# PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

```rust
// flodl
flodl::manual_seed(42);
flodl::gpu_manual_seed_all(42);  // no-op without CUDA feature
```

`manual_seed` controls all libtorch random operations: `Tensor::rand`, `Tensor::randn`, dropout masks, weight initialization. Call it before model creation for full reproducibility.

For CPU-side randomness (data shuffling, augmentation), use `Rng`:

```rust
use flodl::Rng;

let mut rng = Rng::seed(42);       // deterministic
rng.shuffle(&mut indices);          // Fisher-Yates shuffle
let val = rng.f32();               // uniform [0, 1)
let coin = rng.bernoulli(0.5);    // true ~50%
```

## Tensor Creation

```python
# PyTorch
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.rand(2, 3)
x = torch.randn(2, 3)
x = torch.full((2, 3), 7.0)
x = torch.eye(4)
x = torch.arange(0, 10, 2)
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([0, 1, 2], dtype=torch.int64)
x = torch.linspace(0, 1, 10)
```

```rust
// flodl
let opts = TensorOptions::default();  // Float32, CPU
let x = Tensor::zeros(&[2, 3], opts)?;
let x = Tensor::ones(&[2, 3], opts)?;
let x = Tensor::rand(&[2, 3], opts)?;
let x = Tensor::randn(&[2, 3], opts)?;
let x = Tensor::full(&[2, 3], 7.0, opts)?;
let x = Tensor::eye(4, opts)?;
let x = Tensor::arange(0.0, 10.0, 2.0, opts)?;
let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU)?;
let x = Tensor::from_i64(&[0, 1, 2], &[3], Device::CPU)?;
let x = Tensor::linspace(0.0, 1.0, 10, opts)?;
```

**Creation helpers:**

```rust
// Like-tensors (same dtype/device as source)
let y = Tensor::zeros_like(&x)?;
let y = Tensor::ones_like(&x)?;

// Stacking
let y = Tensor::stack(&[&a, &b, &c], 0)?;
```

## Tensor Operations

### Arithmetic

```python
# PyTorch
c = a + b           # element-wise add
c = a - b           # element-wise sub
c = a * b           # element-wise mul
c = a / b           # element-wise div
c = a @ b           # matrix multiply
c = x * 2.0         # scalar multiply
c = x + 1.0         # scalar add
c = x / 3.0         # scalar divide
c = -x              # negation
```

```rust
// flodl
let c = a.add(&b)?;
let c = a.sub(&b)?;
let c = a.mul(&b)?;
let c = a.div(&b)?;
let c = a.matmul(&b)?;
let c = x.mul_scalar(2.0)?;
let c = x.add_scalar(1.0)?;
let c = x.div_scalar(3.0)?;
let c = x.neg()?;
```

### Math Functions

```python
# PyTorch
y = torch.exp(x)
y = torch.log(x)
y = torch.sqrt(x)
y = torch.abs(x)
y = torch.pow(x, 2.0)
y = torch.clamp(x, -1.0, 1.0)
y = torch.clamp(x, min=0.0)
y = torch.clamp(x, max=1.0)
y = torch.sin(x)
y = torch.cos(x)
y = torch.tan(x)
y = torch.asin(x)
y = torch.acos(x)
y = torch.atan(x)
y = torch.atan2(y, x)
y = torch.sign(x)
y = torch.floor(x)
y = torch.ceil(x)
y = torch.round(x)
y = torch.trunc(x)
y = torch.frac(x)
y = torch.reciprocal(x)
y = torch.log1p(x)
y = torch.expm1(x)
y = torch.log2(x)
y = torch.log10(x)
y = torch.erf(x)
y = torch.erfc(x)
y = torch.fmod(x, 3.0)
y = torch.remainder(x, 3.0)
y = torch.lerp(start, end, weight)
y = torch.addmm(bias, mat1, mat2)
y = torch.addcmul(input, t1, t2, value=1.0)
y = torch.addcdiv(input, t1, t2, value=1.0)
y = torch.isclose(x, y, rtol=1e-5, atol=1e-8)
y = torch.maximum(a, b)
y = torch.minimum(a, b)
y = x.masked_fill(mask, 0.0)
y = F.normalize(x, p=2, dim=1)
y = F.cosine_similarity(a, b, dim=1)
```

```rust
// flodl
let y = x.exp()?;
let y = x.log()?;
let y = x.sqrt()?;
let y = x.abs()?;
let y = x.pow_scalar(2.0)?;
let y = x.clamp(-1.0, 1.0)?;
let y = x.clamp_min(0.0)?;
let y = x.clamp_max(1.0)?;
let y = x.sin()?;
let y = x.cos()?;
let y = x.tan()?;
let y = x.asin()?;
let y = x.acos()?;
let y = x.atan()?;
let y = y_var.atan2(&x_var)?;            // Variable method
let y = x.sign()?;
let y = x.floor()?;
let y = x.ceil()?;
let y = x.round()?;
let y = x.trunc()?;
let y = x.frac()?;
let y = x.reciprocal()?;
let y = x.log1p()?;                      // ln(1+x), stable for small x
let y = x.expm1()?;                      // exp(x)-1, stable for small x
let y = x.log2()?;
let y = x.log10()?;
let y = x.erf()?;
let y = x.erfc()?;
let y = x.fmod(3.0)?;                    // C-style remainder
let y = x.remainder(3.0)?;               // Python-style modulo
let y = start.lerp(&end, 0.5)?;          // linear interpolation
let y = bias.addmm(&mat1, &mat2, 1.0, 1.0)?;  // beta*self + alpha*(mat1 @ mat2)
let y = inp.addcmul(&t1, &t2, 1.0)?;    // self + value * t1 * t2
let y = inp.addcdiv(&t1, &t2, 1.0)?;    // self + value * t1 / t2
let y = x.isclose(&y, 1e-5, 1e-8)?;
let y = a.maximum(&b)?;
let y = a.minimum(&b)?;
let y = x.masked_fill(&mask, 0.0)?;      // Variable method
let y = x.normalize(2.0, 1)?;            // Lp-normalize along dim
let y = a.cosine_similarity(&b, 1, 1e-8)?;
```

### Activations and Element-wise Ops

```python
# PyTorch
y = torch.relu(x)
y = torch.sigmoid(x)
y = torch.tanh(x)
y = F.gelu(x)
y = F.silu(x)
y = F.leaky_relu(x, 0.01)
y = F.elu(x, alpha=1.0)
y = F.softplus(x, beta=1.0)
y = F.mish(x)
y = torch.selu(x)
y = F.hardswish(x)
y = F.hardsigmoid(x)
y = torch.softmax(x, dim=1)
y = F.log_softmax(x, dim=1)
```

```rust
// flodl
let y = x.relu()?;
let y = x.sigmoid()?;
let y = x.tanh()?;
let y = x.gelu()?;
let y = x.silu()?;
let y = x.leaky_relu(0.01)?;
let y = x.elu(1.0)?;
let y = x.softplus(1.0, 20.0)?;
let y = x.mish()?;
let y = x.selu()?;
let y = x.hardswish()?;
let y = x.hardsigmoid()?;
let y = x.softmax(1)?;
let y = x.log_softmax(1)?;
```

### Reductions

```python
# PyTorch
s = x.sum()
s = x.sum(dim=1, keepdim=True)
m = x.mean()
m = x.mean(dim=1, keepdim=True)
v = x.var()
v = x.std()
v = x.max(dim=1, keepdim=True).values
v = x.min(dim=1, keepdim=True).values
idx = x.argmax(dim=1)
n = x.norm()
p = x.prod()
p = x.prod(dim=1, keepdim=True)
c = x.cumsum(dim=0)
l = torch.logsumexp(x, dim=1, keepdim=True)
```

```rust
// flodl
let s = x.sum()?;
let s = x.sum_dim(1, true)?;
let m = x.mean()?;
let m = x.mean_dim(1, true)?;
let v = x.var()?;
let v = x.std()?;
let v = x.max_dim(1, true)?;
let v = x.min_dim(1, true)?;
let idx = x.argmax(1, false)?;
let n = x.norm()?;
let p = x.prod()?;
let p = x.prod_dim(1, true)?;
let c = x.cumsum(0)?;
let l = x.logsumexp(1, true)?;
```

### Shape Operations

```python
# PyTorch
y = x.reshape(2, 3)
y = x.view(2, 3)           # same as reshape
y = x.squeeze(0)
y = x.unsqueeze(0)
y = x.flatten(1)
y = x.permute(0, 2, 1)
y = x.transpose(0, 1)
y = x.expand(4, 3)
y = x.contiguous()
y = x.movedim(0, 2)
y = x.flip([0, 1])
y = x.roll(2, dims=0)
y = x.diagonal(0, 0, 1)
y = x.tile((2, 3))
y = x.triu(0)
y = x.tril(0)
y = x.split(2, dim=0)
y = x.unbind(dim=0)
grids = torch.meshgrid(x, y, indexing='ij')
```

```rust
// flodl
let y = x.reshape(&[2, 3])?;
// no separate view - reshape handles it
let y = x.squeeze(0)?;
let y = x.unsqueeze(0)?;
let y = x.flatten(1, -1)?;
let y = x.permute(&[0, 2, 1])?;
let y = x.transpose(0, 1)?;
let y = x.expand(&[4, 3])?;
let y = x.contiguous()?;
let y = x.movedim(0, 2)?;
let y = x.flip(&[0, 1])?;
let y = x.roll(2, 0)?;
let y = x.diagonal(0, 0, 1)?;
let y = x.tile(&[2, 3])?;
let y = x.triu(0)?;
let y = x.tril(0)?;
let parts = x.split(2, 0)?;
let slices = x.unbind(0)?;
let grids = Tensor::meshgrid(&[&x, &y])?;
```

### Indexing and Slicing

```python
# PyTorch
y = x[0]                       # select first along dim 0
y = x[:, 1:3]                  # narrow: dim=1, start=1, length=2
y = x.index_select(0, indices) # gather rows
y = torch.cat([a, b], dim=0)
y = torch.stack([a, b], dim=0)
y = x.chunk(3, dim=1)
y = x.repeat(2, 3)
```

```rust
// flodl
let y = x.select(0, 0)?;
let y = x.narrow(1, 1, 2)?;
let y = x.index_select(0, &indices)?;
let y = a.cat(&b, 0)?;
let y = Tensor::cat_many(&[&a, &b, &c], 0)?;  // concatenate many tensors
let y = Tensor::stack(&[&a, &b], 0)?;
let chunks = x.chunk(3, 1)?;
let y = x.repeat(&[2, 3])?;
let y = x.pad(&[1, 1], 0.0)?;                  // constant-value pad
let y = x.pad_mode(&[1, 1], 1, 0.0)?;          // 0=constant, 1=reflect, 2=replicate, 3=circular

// Batch iteration (split along dim 0)
for batch in data.batches(32)? {
    let x = Variable::new(batch, false);
    // ...
}
```

### Comparisons and Conditionals

```python
# PyTorch
mask = x > threshold
mask = x >= 0
y = torch.where(mask, a, b)
```

```rust
// flodl
let mask = x.gt(&threshold)?;
let mask = x.ge_scalar(0.0)?;
let y = Tensor::where_cond(&mask, &a, &b)?;

// Scalar comparisons
let mask = x.gt_scalar(0.0)?;
let mask = x.lt_scalar(1.0)?;
let mask = x.ge_scalar(0.0)?;
let mask = x.le_scalar(1.0)?;
```

### Dtype Casting

```python
# PyTorch
y = x.float()       # → float32
y = x.double()      # → float64
y = x.half()        # → float16
y = x.to(torch.bfloat16)
```

```rust
// flodl
let y = x.to_dtype(DType::Float32)?;
let y = x.to_dtype(DType::Float64)?;
let y = x.to_dtype(DType::Float16)?;
let y = x.to_dtype(DType::BFloat16)?;
```

### Data Access

```python
# PyTorch
val = loss.item()          # scalar → float
data = x.numpy()           # → numpy array
data = x.tolist()          # → Python list
```

```rust
// flodl
let val = loss.item()?;           // scalar → f64
let data = x.to_f32_vec()?;      // → Vec<f32>
let data = x.to_f64_vec()?;      // → Vec<f64>
let data = x.to_i64_vec()?;      // → Vec<i64>
```

### Tensor Metadata

```python
# PyTorch
x.shape          # torch.Size([2, 3])
x.ndim           # 2
x.numel()        # 6
x.dtype          # torch.float32
x.device         # device(type='cpu')
```

```rust
// flodl
x.shape()        // Vec<i64>: [2, 3]
x.ndim()         // usize: 2
x.numel()        // i64: 6
x.dtype()        // DType::Float32
x.device()       // Device::CPU
```

## Autograd

```python
# PyTorch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # tensor([2.0, 4.0])
```

```rust
// flodl
let xt = Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU)?;
let x = Variable::new(xt, true);  // requires_grad = true
let y = x.pow_scalar(2.0)?.sum()?;
y.backward()?;
println!("{:?}", x.grad());  // Some(tensor([2.0, 4.0]))
```

### Key Differences

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Gradient access | `x.grad` (attribute) | `x.grad()` returns `Option<Tensor>` |
| Clear gradients | `x.grad.zero_()` | `x.zero_grad()` |
| Detach | `x.detach()` | `x.detach()` - returns new leaf Variable |
| No-grad block | `with torch.no_grad():` | `no_grad(\|\| { ... })` or `let _g = NoGradGuard::new();` |
| Check grad enabled | `torch.is_grad_enabled()` | `is_grad_enabled()` |
| Leaf check | `x.is_leaf` | `x.is_leaf()` |

**Differentiable ops on Variable:**

*Arithmetic:* Add, Sub, Mul, Div, Matmul, MulScalar, AddScalar, DivScalar, Neg

*Activations:* ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, ELU, Softplus, Mish, SELU, Hardswish, Hardsigmoid, PReLU, Softmax, LogSoftmax

*Math:* Exp, Log, Sqrt, Abs, Pow, Sin, Cos, Sign, Floor, Ceil, Round, Reciprocal, Clamp, ClampMin, ClampMax, Log1p, Expm1, Log2, Log10, Atan2, Maximum, Minimum, MaskedFill, Normalize, CosineSimilarity, Triu, Tril

*Reductions:* Sum, SumDim, Mean, MeanDim, Var, Std, VarDim, StdDim, Min, Max, MinDim, MaxDim, Prod, ProdDim, Cumsum, Logsumexp

*Shape:* Reshape, Transpose, Permute, Squeeze, Unsqueeze, UnsqueezeMany, Flatten, Expand, Select, Narrow, Cat, CatMany, Stack, Chunk, Repeat, Pad, IndexSelect, Gather, TopK, Sort

*NN:* Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d, Conv3d, ConvTranspose3d, MaxPool2d, AvgPool2d, MaxPool1d, AvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, InstanceNorm, GroupNorm, LayerNorm, GridSample, PixelShuffle, PixelUnshuffle, Bilinear, EmbeddingBag, Im2col, Col2im

## Neural Network Layers

```python
# PyTorch
layer = nn.Linear(784, 128)
layer = nn.Linear(784, 128, bias=False)
layer = nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)
layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
layer = nn.Conv3d(1, 32, kernel_size=3, padding=1)
layer = nn.ConvTranspose1d(16, 3, kernel_size=5)
layer = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
layer = nn.ConvTranspose3d(32, 1, kernel_size=3)
layer = nn.MaxPool2d(kernel_size=2)
layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
layer = nn.AvgPool2d(kernel_size=2)
layer = nn.MaxPool1d(kernel_size=2)
layer = nn.AvgPool1d(kernel_size=2)
layer = nn.AdaptiveMaxPool2d((7, 7))
layer = nn.AdaptiveAvgPool2d((1, 1))
layer = nn.PixelShuffle(2)
layer = nn.PixelUnshuffle(2)
layer = nn.Upsample(size=(64, 64), mode='bilinear')
layer = nn.Unfold(kernel_size=3)
layer = nn.Fold(output_size=(28, 28), kernel_size=3)
layer = nn.LayerNorm(128)
layer = nn.RMSNorm(128)
layer = nn.GroupNorm(4, 16)
layer = nn.BatchNorm1d(128)
layer = nn.BatchNorm2d(64)
layer = nn.InstanceNorm2d(64, affine=True)
layer = nn.Dropout(p=0.5)
layer = nn.Dropout2d(p=0.1)
layer = nn.AlphaDropout(p=0.1)
layer = nn.ZeroPad2d(1)
layer = nn.ReflectionPad2d(1)
layer = nn.Embedding(1000, 128)
layer = nn.EmbeddingBag(1000, 128)
cell = nn.GRUCell(128, 256)
cell = nn.LSTMCell(128, 256)
layer = nn.GRU(128, 256, num_layers=2)
layer = nn.LSTM(128, 256, num_layers=2)
layer = nn.MultiheadAttention(512, 8)
layer = nn.Bilinear(128, 64, 32)
```

```rust
// flodl
let layer = Linear::new(784, 128)?;
let layer = Linear::no_bias(784, 128)?;
let layer = Conv1d::configure(3, 16, 5).with_stride(2).with_padding(2).done()?;
let layer = Conv2d::new(3, 64, 3)?;                                          // defaults: stride=1, padding=0
let layer = Conv2d::configure(3, 64, 3).with_padding(1).with_stride(2).done()?;    // fluent builder
let layer = Conv2d::build(3, 64, 3, true, [1,1], [1,1], [1,1], 1, Device::CPU)?;   // full control
let layer = Conv3d::configure(1, 32, [3,3,3]).with_padding([1,1,1]).done()?;
let layer = ConvTranspose1d::new(16, 3, 5)?;
let layer = ConvTranspose2d::new(64, 3, 4)?;
let layer = ConvTranspose3d::new(32, 1, [3,3,3])?;
let pool = MaxPool2d::new(2);                                                // kernel=2, stride=2 (defaults to kernel)
let pool = MaxPool2d::with_stride(3, 2).padding(1);                          // kernel=3, stride=2, padding=1
let pool = AvgPool2d::new(2);
let pool = MaxPool1d::new(2);
let pool = AvgPool1d::new(2);
let pool = AdaptiveMaxPool2d::new(7, 7);
let pool = AdaptiveAvgPool2d::new([1, 1]);                                   // global avg pool (ResNet head)
let layer = PixelShuffle::new(2);
let layer = PixelUnshuffle::new(2);
let layer = Upsample::new(&[64, 64], 1);                                    // mode: 0=nearest, 1=bilinear
let layer = Unfold::new([3,3], [1,1], [0,0], [1,1]);
let layer = Fold::new([28,28], [3,3], [1,1], [0,0], [1,1]);
let layer = LayerNorm::new(128)?;
let layer = RMSNorm::new(128)?;
let layer = GroupNorm::new(4, 16)?;
let layer = BatchNorm::new(128)?;                                            // for [B, features] after Linear
let layer = BatchNorm2d::new(64)?;                                           // for [B, C, H, W] after Conv2d
let layer = InstanceNorm::new(64, true)?;                                    // affine=true
let layer = Dropout::new(0.5);
let layer = Dropout2d::new(0.1);
let layer = AlphaDropout::new(0.1);                                          // for SELU networks
let layer = ZeroPad2d::new(1);
let layer = ReflectionPad2d::new(1);
let layer = Embedding::new(1000, 128)?;
let layer = EmbeddingBag::new(1000, 128)?;
let cell = GRUCell::new(128, 256)?;
let cell = LSTMCell::new(128, 256)?;
let layer = GRU::new(128, 256, 2)?;                                          // 2-layer GRU
let layer = LSTM::new(128, 256, 2)?;                                         // 2-layer LSTM
let layer = MultiheadAttention::new(512, 8)?;
let rope = RotaryEmbedding::new(64, 2048)?;                                  // RoPE tables (head_dim, max_seq); hand-rolled in PyTorch
let layer = MultiheadAttention::new(512, 8)?.rotary(rope)?;                  // rotates q/k after the head split
let layer = Bilinear::new(128, 64, 32, true)?;

// On a specific device (all modules have on_device() variants):
let layer = Linear::on_device(784, 128, Device::CUDA(0))?;
let layer = Conv2d::configure(3, 64, 3).with_padding(1).on_device(Device::CUDA(0)).done()?;
let layer = LayerNorm::on_device(128, Device::CUDA(0))?;
let layer = RMSNorm::on_device(128, Device::CUDA(0))?;
let layer = GroupNorm::on_device(4, 16, Device::CUDA(0))?;
let layer = BatchNorm::on_device(128, Device::CUDA(0))?;
let layer = BatchNorm2d::on_device(64, Device::CUDA(0))?;
let layer = InstanceNorm::on_device(64, true, Device::CUDA(0))?;
let layer = Embedding::on_device(1000, 128, Device::CUDA(0))?;
let cell = GRUCell::on_device(128, 256, Device::CUDA(0))?;
let cell = LSTMCell::on_device(128, 256, Device::CUDA(0))?;
let layer = GRU::on_device(128, 256, 2, false, Device::CUDA(0))?;
let layer = LSTM::on_device(128, 256, 2, false, Device::CUDA(0))?;
let layer = MultiheadAttention::on_device(512, 8, Device::CUDA(0))?;
```

## Functional Ops (torch.nn.functional)

The `F.*` entry points that are not simple tensor methods live in
`flodl::autograd` as free functions over `Variable` (all re-exported at
the crate root):

```python
# PyTorch
y = F.scaled_dot_product_attention(q, k, v, attn_mask=m, is_causal=True)
y = F.max_pool2d(x, 2, stride=2)
y = F.adaptive_avg_pool2d(x, (1, 1))
y = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
y = F.embedding(indices, weight)
y = F.embedding_bag(indices, weight, offsets, mode="mean")
```

```rust
// flodl
use flodl::{scaled_dot_product_attention, max_pool2d, adaptive_avg_pool2d,
            grid_sample, embedding, embedding_bag};

// Fused attention kernel (libtorch picks the backend); attn_mask and
// is_causal are exclusive (libtorch rejects both), scale None = 1/sqrt(E):
let y = scaled_dot_product_attention(&q, &k, &v, Some(&m), 0.0, false, None)?;
let y = scaled_dot_product_attention(&q, &k, &v, None, 0.0, true, None)?;   // causal

let y = max_pool2d(&x, [2, 2], [2, 2], [0, 0], [1, 1], false)?;  // kernel, stride, padding, dilation, ceil_mode
let y = adaptive_avg_pool2d(&x, [1, 1])?;
let y = grid_sample(&x, &grid, 0, 0, false)?;                    // mode: 0=bilinear 1=nearest; padding: 0=zeros 1=border 2=reflection
let y = embedding(&weight, &indices, -1)?;                       // padding_idx: -1 = none
let y = embedding_bag(&weight, &indices, &offsets, 1)?;          // mode: 0=sum 1=mean 2=max
```

## Activations (as Modules)

```python
# PyTorch
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.GELU()
nn.SiLU()
nn.LeakyReLU(0.01)
nn.ELU(alpha=1.0)
nn.Softplus(beta=1.0)
nn.Mish()
nn.SELU()
nn.Hardswish()
nn.Hardsigmoid()
nn.PReLU(num_parameters=1)
nn.Softmax(dim=-1)
nn.LogSoftmax(dim=-1)
nn.Flatten(start_dim=1)
nn.Identity()
```

```rust
// flodl - unit structs (no allocation, bare-name usage)
ReLU
Sigmoid
Tanh
GELU                       // erf form (default) - GELU::tanh() for the tanh approximation
SiLU
Mish
SELU
Hardswish
Hardsigmoid
Identity

// Parameterized at construction
LeakyReLU::new(0.01)
ELU::new(1.0)
Softplus::new(1.0, 20.0)         // beta, threshold
Softmax::new(-1)                  // dim
LogSoftmax::new(-1)
Flatten::new(1, -1)               // start_dim, end_dim

// Learnable parameters
let prelu = PReLU::new(1, Device::CPU)?;
```

### Preprocessing Modules

```python
# PyTorch
blur = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=1.5)
# or functional:
y = torchvision.transforms.functional.gaussian_blur(x, kernel_size=7, sigma=1.5)
```

```rust
// flodl - as a Module (for use in FlowBuilder graphs)
let blur = GaussianBlur::new(1.5);  // kernel size auto-computed from sigma
let y = blur.forward(&x)?;

// flodl - as a free function
let y = gaussian_blur_2d(&x, 1.5)?;  // input must be [B, C, H, W]
```

`GaussianBlur` is stateless (no parameters). Kernel size is `2 * ceil(3 * sigma) + 1`,
matching OpenCV's default. Runs under `NoGradGuard` - no autograd graph built.

## Composite Modules

In PyTorch, `nn.Module.__init__` auto-discovers child modules assigned to `self`.
In flodl, composite modules implement the `Module` trait and declare children
via `sub_modules()` - enabling recursive device placement, training mode
toggling, and parameter collection.

```python
# PyTorch - children auto-discovered
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model.to(device)           # walks children automatically
model.parameters()         # collects from all children
model.train()              # propagates to children
```

```rust
// flodl - declare children via parameters()
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl Module for MLP {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let h = self.fc1.forward(input)?.relu()?;
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        nn::collect_parameters(&[&self.fc1 as &dyn Module, &self.fc2])
    }

    fn name(&self) -> &str { "MLP" }
}
```

Or skip manual structs entirely - use the **graph builder** (see below).

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Child discovery | Implicit (`self.x = ...`) | Explicit (`sub_modules()`) |
| Parameter collection | Automatic | `collect_parameters()` walks tree |
| Device move | `model.to(device)` | `module.move_to_device(Device::CUDA(0))` |
| Training mode | `model.train()` / `model.eval()` | `module.train()` / `module.eval()` |

## Loss Functions

```python
# PyTorch
loss = F.mse_loss(pred, target)
loss = F.cross_entropy(logits, labels)
loss = F.nll_loss(log_probs, labels)
loss = F.binary_cross_entropy(probs, target)
loss = F.binary_cross_entropy_with_logits(pred, target)
loss = F.l1_loss(pred, target)
loss = F.smooth_l1_loss(pred, target, beta=1.0)
loss = F.kl_div(log_probs, targets, reduction='batchmean')
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss = F.poisson_nll_loss(pred, target, log_input=True)
loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0)
loss = F.cosine_embedding_loss(x1, x2, labels, margin=0.0)
loss = F.hinge_embedding_loss(input, labels, margin=1.0)
loss = F.margin_ranking_loss(x1, x2, labels, margin=0.0)
# focal_loss - not in PyTorch, popular in object detection
```

```rust
// flodl - free functions, return Variable (differentiable)
let loss = mse_loss(&pred, &target)?;
let loss = cross_entropy_loss(&logits, &labels)?;   // labels: [B] indices or [B,C] one-hot
let loss = nll_loss(&log_probs, &labels)?;           // after log_softmax
let loss = bce_loss(&probs, &target)?;               // from probabilities
let loss = bce_with_logits_loss(&logits, &target)?;  // numerically stable
let loss = l1_loss(&pred, &target)?;
let loss = smooth_l1_loss(&pred, &target, 1.0)?;
let loss = kl_div_loss(&log_probs, &targets)?;
let loss = ctc_loss(&log_probs, &targets, &input_lengths, &target_lengths, 0)?;
let loss = poisson_nll_loss(&pred, &target, true)?;
let loss = focal_loss(&logits, &target, 0.25, 2.0)?;  // alpha, gamma - class imbalance
let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0)?;
let loss = cosine_embedding_loss(&x1, &x2, &labels, 0.0)?;
let loss = hinge_embedding_loss(&input, &labels, 1.0)?;
let loss = margin_ranking_loss(&x1, &x2, &labels, 0.0)?;
```

## Optimizers

```python
# PyTorch
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
opt = torch.optim.RMSprop(model.parameters(), lr=0.01)
opt = torch.optim.Adagrad(model.parameters(), lr=0.01)
opt = torch.optim.RAdam(model.parameters(), lr=0.001)
opt = torch.optim.NAdam(model.parameters(), lr=0.001)

opt.zero_grad()
loss.backward()
opt.step()
```

```rust
// flodl - optimizers own a clone of the param list
let mut opt = SGD::new(&params, 0.01, 0.9);     // lr, momentum
let mut opt = Adam::new(&params, 0.001);         // lr
let mut opt = AdamW::new(&params, 0.001, 0.01);  // lr, weight_decay
let mut opt = RMSprop::new(&params, 0.01);       // lr (alpha=0.99, eps=1e-8)
let mut opt = Adagrad::new(&params, 0.01);       // lr
let mut opt = RAdam::new(&params, 0.001);        // rectified Adam - auto warmup
let mut opt = NAdam::new(&params, 0.001);        // Nesterov-accelerated Adam

opt.zero_grad();
loss.backward()?;
opt.step()?;  // returns Result<()>
```

### Parameter groups

```python
# PyTorch
opt = torch.optim.Adam([
    {"params": encoder.parameters(), "lr": 1e-5},
    {"params": decoder.parameters(), "lr": 1e-3},
])
```

```rust
// flodl - builder API
let mut opt = Adam::with_groups()
    .group(&encoder_params, 1e-5)
    .group(&decoder_params, 1e-3)
    .build();

opt.set_group_lr(0, 1e-6);  // adjust one group
opt.set_lr(1e-4);           // adjust all groups
```

### Freezing parameters

```python
# PyTorch
for param in model.encoder.parameters():
    param.requires_grad = False
```

```rust
// flodl
for param in &encoder_params {
    param.freeze()?;
}
// Later: param.unfreeze()?;
// Check: param.is_frozen()
```

## Learning Rate Scheduling

```python
# PyTorch
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 90], gamma=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=1000)
scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-4, max_lr=1e-2, step_size_up=500)
scheduler.step()
```

```rust
// flodl - schedulers produce an lr, you apply it
let sched = StepDecay::new(0.001, 30, 0.1);
let sched = CosineScheduler::new(0.001, 1e-6, 100);
let sched = ExponentialLR::new(0.001, 0.95);
let sched = MultiStepLR::new(0.001, &[30, 60, 90], 0.1);
let sched = OneCycleLR::new(0.01, 1000);                  // 30% warmup
let sched = CyclicLR::new(1e-4, 1e-2, 500);               // symmetric triangle
let lr = sched.lr(step);
opt.set_lr(lr);

// Composable warmup:
let sched = WarmupScheduler::new(CosineScheduler::new(0.001, 1e-6, 100), 0.001, 10);

// Plateau (reduce on metric stall):
let mut sched = PlateauScheduler::new(0.001, 10, 0.1, 1e-6);
let lr = sched.observe(val_loss);
opt.set_lr(lr);
```

**Key difference:** PyTorch schedulers wrap optimizers. flodl schedulers are pure
functions - you call `.lr(step)` or `.observe(metric)` and set the lr yourself.

## Gradient Clipping

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

```rust
// flodl
let total_norm = clip_grad_norm(&params, 1.0)?;
let max_val = clip_grad_value(&params, 0.5)?;
```

## Saving and Loading

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

```rust
// flodl - one-call checkpoint (saves params + buffers + structural hash)
model.save_checkpoint("model.fdl")?;
let report = model.load_checkpoint("model.fdl")?;
// report.loaded, report.skipped, report.missing

// Or with any io::Write / io::Read for custom I/O:
let named = model.named_parameters();
let buffers = model.named_buffers();
let hash = Some(model.structural_hash());
save_checkpoint(&mut writer, &named, &buffers, hash)?;
let report = load_checkpoint(&mut reader, &named, &buffers, hash)?;
```

### Full training resume (model + optimizer)

Optimizers implement the `Stateful` trait for save/load. Use the file
helpers — they write a self-identifying header (so a file can't be loaded
into the wrong optimizer), commit atomically, and gzip when the path ends
in `.gz`:

```rust
// Save
model.save_checkpoint("model.fdl")?;
optimizer.save_state_file("optimizer.optim")?;

// Load
let report = model.load_checkpoint("model.fdl")?;
optimizer.load_state_file("optimizer.optim")?;
```

`Optimizer::save_state_to(path)` is the object-safe save (callable through
`&dyn Optimizer`, e.g. the cluster save-on-failure sidecar); loading needs
the concrete type via `load_state_file`. `save_state`/`load_state` (raw
`&mut Write` / `&mut Read`) are available for custom I/O, but they write no
header — pair them only with each other, not with `load_state_file`.

### Migrating checkpoints across versions

When parameter naming changes between flodl versions (e.g., tag renames from
the graph tree release), use `migrate_checkpoint_file()` to remap an old
checkpoint to match your current model:

```rust
use flodl::nn::{checkpoint_version, migrate_checkpoint_file};

// Check if migration is needed
if checkpoint_version("model.fdl")? < 2 {
    let report = migrate_checkpoint_file(
        "model.fdl",          // old checkpoint (v1)
        "model_v2.fdl",       // migrated output (v2)
        &model.named_parameters(),
        &model.named_buffers(),
    )?;
    println!("{}", report);
    // unchanged (1):
    //   shared/weight
    // remapped (2):
    //   linear_0/weight -> encoder/weight
    //   linear_0/bias -> encoder/bias
}

// Load the migrated checkpoint normally
model.load_checkpoint("model_v2.fdl")?;
```

The migration matches entries by exact name first, then by shape+dtype in
positional order. `MigrateReport::is_complete()` returns `true` when nothing
was dropped or missing. Only works for the same model architecture - if you
changed the architecture, retrain.

### Migrating optimizer state across versions

The optimizer `.optim` format gained a self-identifying `FDLO` header and
per-parameter step counts. An optimizer state file written by an earlier
flodl (no header) is rejected on load with a message pointing here; convert
it once with `migrate_optim_state_file()`. Unlike checkpoint migration, this
needs no model — just the paths and the optimizer kind that wrote the file
(the old format carried no type tag, so you supply it):

```rust
use flodl::nn::{migrate_optim_state_file, StateKind};

// Convert an old optimizer state file to the current format.
// In-place (src == dst) is safe; gzip is picked from each path's extension.
migrate_optim_state_file(
    "optimizer.optim",       // old file (pre-header)
    "optimizer.optim",       // in-place, or a new path
    StateKind::Adam,         // the optimizer that wrote it
)?;

// Now it loads normally:
optimizer.load_state_file("optimizer.optim")?;
```

For Adam/AdamW the old single global step is expanded into per-parameter
steps (exact for parameters trained from step 0). `SGD`, `RMSprop`, and
`GradScaler` convert verbatim under the new header; `Adagrad`, `RAdam`, and
`NAdam` never had a pre-header format, so they need no migration.

## Device Placement

```python
# PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
x = x.to(device)
```

```rust
// flodl
let device = if gpu_available() { Device::CUDA(0) } else { Device::CPU };

// Move model parameters
module.move_to_device(device);

// Move tensors
let x = x.to_device(device)?;
let x = x.to_device_of(&weights)?;  // match another tensor's device

// Move variables
let x = x.to_device(device)?;

// Create directly on device
let opts = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
let x = Tensor::zeros(&[2, 3], opts)?;
```

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Device check | `torch.cuda.is_available()` | `gpu_available()` |
| Device count (pre-`Trainer::run`) | `torch.cuda.device_count()` | `flodl::sys::detect_gpus().len()` - CUDA-free, no libtorch init |
| Device count (after `Trainer::run`) | `torch.cuda.device_count()` | `gpu_device_count()` - safe inside training; touches libtorch |
| Model move | `model.to(device)` | `module.move_to_device(device)` |
| Tensor move | `x.to(device)` | `x.to_device(device)?` |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = True` | `set_cudnn_benchmark(true)` |

## Weight Initialization

```python
# PyTorch - in-place mutation
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
nn.init.kaiming_normal_(layer.weight)
nn.init.uniform_(layer.weight, -0.1, 0.1)
nn.init.normal_(layer.weight, 0.0, 0.01)
nn.init.orthogonal_(layer.weight, gain=1.0)
nn.init.trunc_normal_(layer.weight, std=0.02)
```

```rust
// flodl - returns a new Tensor, then set_data() to apply
let w = xavier_uniform(&[out, inp], inp, out, device)?;
let w = xavier_normal(&[out, inp], inp, out, device)?;
let w = kaiming_uniform(&[out, inp], inp, 0.0, device)?;   // a=0.0 for ReLU
let w = kaiming_normal(&[out, inp], inp, 0.0, device)?;
let w = uniform(&[out, inp], -0.1, 0.1, device)?;
let w = normal(&[out, inp], 0.0, 0.01, device)?;
let w = orthogonal(&[out, inp], 1.0, device)?;             // 2D only
let w = trunc_normal(&[out, inp], 0.0, 0.02, -2.0, 2.0, device)?;  // mean, std, a, b

layer.parameters()[0].set_data(&w);
```

## Mixed Precision Training

```python
# PyTorch
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

```rust
// flodl
let mut scaler = GradScaler::new();

// Cast model to float16
cast_parameters(&params, DType::Float16);

// Forward + backward with scaled loss
let output = model.forward(&input)?;
let loss = mse_loss(&output, &target)?;
let scaled = scaler.scale(&loss)?;
scaled.backward()?;

// Step with automatic unscaling + inf/nan checking
let stepped = scaler.step(&params, &mut || opt.step())?;
scaler.update();
```

## Training Loop Pattern

```python
# PyTorch
model.train()
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

```rust
// flodl
model.train();
for epoch in 0..num_epochs {
    // your data loading here
    opt.zero_grad();
    let output = model.forward(&input)?;
    let loss = cross_entropy_loss(&output, &target)?;
    loss.backward()?;
    clip_grad_norm(&params, 1.0)?;
    opt.step()?;

    opt.set_lr(sched.lr(epoch));
    println!("Epoch {}: loss={:.4}", epoch, loss.item()?);
}

// Inference
model.eval();
let pred = no_grad(|| model.forward(&test_input))?;
```

## Error Handling

PyTorch raises Python exceptions. flodl returns `Result<T, TensorError>`.
Both `Result` and `TensorError` are re-exported from `flodl::*`.
Use Rust's `?` operator for clean propagation:

```rust
fn train_step(model: &Graph, input: &Variable, target: &Variable,
              optimizer: &mut Adam) -> Result<f64> {
    optimizer.zero_grad();
    let output = model.forward(input)?;
    let loss = mse_loss(&output, target)?;
    loss.backward()?;
    optimizer.step()?;
    loss.item()
}
```

## Memory Management

| Aspect | PyTorch | flodl |
|--------|---------|-------|
| Model memory | Python GC + reference counting | Rust `Drop` trait - deterministic deallocation |
| GPU memory | GC-delayed; `torch.cuda.empty_cache()` | Freed immediately when last reference drops |
| Gradient graph | Freed after `.backward()` | `backward()` also calls `detach_()` - grad_fn chain freed synchronously |
| No-grad inference | `with torch.no_grad():` | `no_grad(\|\| { ... })` or `NoGradGuard::new()` |
| Handle diagnostics | N/A | `live_tensor_count()`, `rss_kb()` |

No manual memory management needed. Rust's ownership system handles it.

## Graph Builder (flodl-specific)

flodl's unique feature: a fluent API for building computation graphs declaratively.
No PyTorch equivalent - this replaces manual `nn.Module` subclassing.

### Sequential

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

```rust
// flodl
let model = FlowBuilder::from(Linear::new(784, 128)?)
    .through(ReLU)
    .through(Linear::new(128, 10)?)
    .build()?;
```

### Residual Connections

```python
# PyTorch - manual
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))
    def forward(self, x):
        return x + self.net(x)
```

```rust
// flodl - one line
let model = FlowBuilder::from(Linear::new(128, 128)?)
    .also(ReLU)    // skip connection: output = input + ReLU(input)
    .build()?;
```

### Residual with Projection Skip (ResNet downsample)

When the skip path needs its own transform (e.g. 1×1 conv + BN to match
channel/stride changes in ResNet's downsample blocks), use `also_with`.
It generalizes `also` with an explicit `skip` path alongside the `main`
path: `output = skip(x) + main(x)`.

```python
# PyTorch - ResNet BasicBlock with downsample
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
        )
        self.downsample = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
            if stride != 1 or c_in != c_out else nn.Identity()
        )
    def forward(self, x):
        return F.relu(self.downsample(x) + self.main(x))
```

```rust
// flodl - same block, one builder chain
FlowBuilder::from(prev)
    .also_with(
        downsample_1x1_bn,         // skip branch (Identity if no projection needed)
        conv_bn_relu_conv_bn,      // main branch
    )
    .through(ReLU)
    .build()?;
```

See `ddp-bench/src/models/resnet_graph.rs` for a full ResNet-20 built
with `also_with`.

### Parallel Branches

```python
# PyTorch - manual fork/merge
class ParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Linear(128, 64)
        self.branch2 = nn.Linear(128, 64)
    def forward(self, x):
        return self.branch1(x) + self.branch2(x)
```

```rust
// flodl
let model = FlowBuilder::from(Linear::new(128, 128)?)
    .split(modules![Linear::new(128, 64)?, Linear::new(128, 64)?])
    .merge(MergeOp::Add)
    .build()?;
```

### Loops (Recurrent)

```rust
let model = FlowBuilder::from(init_module)
    .loop_body(step_module)
    .for_n(10)              // fixed 10 iterations
    .through(head_module)
    .build()?;

// Or with learned halting:
    .loop_body(step_module)
    .until_cond(ThresholdHalt::new(0.95), 20)  // max 20 iters
```

### Routing (Mixture of Experts)

```rust
// Hard routing - one expert per input
let model = FlowBuilder::from(encoder)
    .switch(ArgmaxSelector::new(128, 3)?, modules![expert1, expert2, expert3])
    .build()?;

// Soft routing - weighted mixture
let model = FlowBuilder::from(encoder)
    .gate(SoftmaxRouter::new(128, 3)?, modules![expert1, expert2, expert3])
    .build()?;
```

### Tags and Cross-References

```rust
// Tag intermediate outputs for later use
let model = FlowBuilder::from(encoder)
    .tag("encoded")
    .through(decoder)
    .input(&["encoded"])       // declare named input
    .using(&["encoded"])       // wire tagged value as named input
    .build()?;
```

### Observation and Profiling

```rust
let mut model = FlowBuilder::from(Linear::new(4, 8)?)
    .through(ReLU)
    .build()?;

// After each batch
model.end_step();

// After each epoch
model.end_epoch();

// Query metrics
let trend = model.trend("loss");
println!("mean: {}", trend.mean());

// Profiling
model.enable_profiling();
// ... run forward passes ...
if let Some(profile) = model.profile() {
    for t in &profile.nodes {
        println!("{}: {}", t.id, format_duration(t.duration.as_secs_f64()));
    }
}

// Visualization
model.dot();                         // GraphViz DOT string
model.svg(Some("model.svg"))?;      // render to SVG
```

The graph implements `Module`, so it works with optimizers, checkpointing, and everything else.

## Graph Tree (Hierarchical Composition)

PyTorch uses `nn.Module` nesting and `named_modules()` for hierarchical access.
flodl's graph tree provides label-path addressing for the same patterns - freeze
by path, per-subgraph optimizer groups, subgraph checkpoint loading, and
cross-boundary observation.

### Labeling subgraphs

```python
# PyTorch - child modules are auto-discovered
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
```

```rust
// flodl - label graphs for tree features
let encoder = FlowBuilder::from(scan_module)
    .through(read_module)
    .label("encoder")
    .build()?;

let model = FlowBuilder::from(encoder)  // child "encoder" registered
    .through(decoder)
    .build()?;
```

### Selective freeze/thaw by path

```python
# PyTorch
for param in model.encoder.parameters():
    param.requires_grad = False
# Thaw a sub-part:
for param in model.encoder.scan.parameters():
    param.requires_grad = True
```

```rust
// flodl - declarative, by label path
model.freeze("encoder")?;
model.thaw("encoder.scan")?;
assert!(model.is_frozen("encoder.read")?);  // read stays frozen
```

### Per-subgraph optimizer groups

```python
# PyTorch
optimizer = torch.optim.Adam([
    {'params': model.encoder.scan.parameters(), 'lr': 1e-4},
    {'params': model.meta.parameters(), 'lr': 1e-3},
])
```

```rust
// flodl
let mut optimizer = Adam::with_groups()
    .group(&model.parameters_at("encoder.scan")?, 0.0001)
    .group(&model.parameters_at("meta")?, 0.001)
    .build();
```

### Subgraph checkpoint loading

```python
# PyTorch - load weights into a submodule
state = torch.load("encoder_v1.pt")
model.encoder.load_state_dict(state)
```

```rust
// flodl - loads using the child's own namespace and hash validation
let report = model.load_subgraph_checkpoint("encoder", "encoder_v1.fdl.gz")?;
```

### Cross-boundary observation

```python
# PyTorch - manual: register hooks or store intermediates in forward()
```

```rust
// flodl - read tags and metrics across graph boundaries
model.forward(&input)?;
let hidden = model.tagged_at("encoder.hidden")?;  // Option<Variable>

// Record and track metrics in children
model.record_at("encoder.loss", loss_value)?;
model.flush(&[]);  // flushes entire tree automatically

let trend = model.trend_at("encoder.loss")?;
```

### Training mode propagation

```python
# PyTorch
model.encoder.eval()  # BatchNorm uses running stats
```

```rust
// flodl - by label path
model.set_training_at("encoder", false)?;
```

See [Graph Tree tutorial](tutorials/10-graph-tree.md) for the full API reference.

## Training Monitor (replaces TensorBoard)

PyTorch researchers typically use TensorBoard, Weights & Biases, or MLflow for
training visibility. In floDl, the training monitor is built in - no external
process, no pip install, no separate UI.

```python
# PyTorch + TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/experiment_1")

for epoch in range(num_epochs):
    # ... training ...
    writer.add_scalar("loss", loss.item(), epoch)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

# Then: tensorboard --logdir runs/
```

```rust
// flodl - built-in monitor with live dashboard
use flodl::Monitor;

let mut monitor = Monitor::new(num_epochs);
monitor.serve(3000)?;                        // live dashboard at http://localhost:3000
monitor.watch(&model);                       // graph SVG in dashboard
monitor.save_html("training_report.html");   // archive at finish

for epoch in 0..num_epochs {
    let t = std::time::Instant::now();
    // ... training ...

    monitor.log(epoch, t.elapsed(), &[("loss", loss_val), ("lr", lr)]);
    // epoch  42/100  loss=0.0023  lr=0.001  [1.2s  ETA 1m 10s]  VRAM: 2.1/6.0 GB (82%)
}

monitor.finish_with(&model);  // profiled SVG + archive saved
```

| Feature | TensorBoard | flodl Monitor |
|---------|-------------|---------------|
| Setup | `pip install tensorboard` + `SummaryWriter` + `tensorboard --logdir` | `monitor.serve(3000)` |
| Terminal output | None (web only) | One-line per epoch with ETA |
| Resource tracking | Manual (no GPU metrics built in) | CPU/RAM/GPU/VRAM automatic |
| Live charts | Yes (web) | Yes (SSE, no polling) |
| Architecture viz | `add_graph()` (limited) | `monitor.watch(&model)` - full DOT/SVG with profiling heat map |
| Offline archive | Log files (need TensorBoard to view) | Self-contained HTML |
| Dependencies | protobuf, gRPC, webpack frontend | Zero - 16KB inline HTML/JS |

## GPU Memory Queries

```python
# PyTorch
torch.cuda.memory_allocated()     # bytes allocated by tensors
torch.cuda.memory_reserved()      # bytes reserved by caching allocator
torch.cuda.max_memory_allocated()  # peak allocated
```

```rust
// flodl - hardware-level via cudaMemGetInfo
let (used, total) = gpu_memory_info()?;   // (bytes_used, bytes_total)
let util = gpu_utilization();              // Option<u32> - GPU % via NVML

// Allocator-level queries
let active = gpu_active_bytes()?;             // bytes backing live tensors
let peak = gpu_peak_active_bytes()?;          // max since last reset
let peak_reserved = gpu_peak_reserved_bytes()?;  // max allocator reservation

// Reset peak tracking (e.g., between profiling phases)
gpu_empty_cache();
gpu_reset_peak_stats();
```

| PyTorch | flodl | What it reports |
|---------|-------|----------------|
| `torch.cuda.mem_get_info()` | `gpu_memory_info()?` | `(used, total)` bytes via `cudaMemGetInfo` |
| `torch.cuda.memory_allocated()` | `gpu_active_bytes()?` | Bytes currently backing live tensors |
| `torch.cuda.memory_reserved()` | `gpu_allocated_bytes()?` | Bytes reserved by caching allocator (includes spill) |
| `torch.cuda.max_memory_allocated()` | `gpu_peak_active_bytes()?` | Peak allocated since last reset |
| `torch.cuda.max_memory_reserved()` | `gpu_peak_reserved_bytes()?` | Peak reserved since last reset |
| `torch.cuda.reset_peak_memory_stats()` | `gpu_reset_peak_stats()` | Reset peak counters |
| `torch.cuda.empty_cache()` | `gpu_empty_cache()` | Release unused cached blocks |
| *(no built-in)* | `gpu_utilization()` | GPU compute % via NVML |

The monitor samples these automatically on every `log()` call - you don't need
to query them manually during training.

## Multi-GPU Training (DDP)

PyTorch's DDP requires multi-process coordination, environment
variables, and a launcher (`torchrun`). flodl auto-promotes to
process-per-rank fan-out automatically when 2+ GPUs are visible - the
**same `Trainer::builder` call** runs on CPU, single GPU, N GPUs
single-host, or N GPUs across many hosts.

### Setup comparison

```python
# PyTorch: requires torchrun, process groups, env vars
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")
rank = dist.get_rank()
model = DDP(MyModel().to(rank), device_ids=[rank])
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Launch: torchrun --nproc_per_node=2 train.py
```

```rust
// flodl: same call on CPU, 1 GPU, N GPUs, or multi-host cluster.
let state = Trainer::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)
    .batch_size(32)
    .num_epochs(10)
    .run()?
    .join()?;
```

On 2+ visible CUDA devices, flodl auto-promotes the calling binary
process to a launcher, fork-execs one child per rank, sets up NCCL
rendezvous, and supervises children. No torchrun, no env-var dance.

For multi-host, add an `fdl.cluster.yml` (or build the topology
programmatically with `ClusterBuilder`); `fdl @cluster train` SSHes
every worker, pre-builds, and fans out.

### Concept mapping

| PyTorch | flodl | Notes |
|---------|-------|-------|
| `dist.init_process_group("nccl")` | Automatic | Launcher handles rendezvous; NCCL init on main + `split()` everywhere |
| `DistributedDataParallel(model)` | `Trainer::builder()` / `Trainer::run()` | Auto-promote on 2+ visible GPUs |
| `DistributedSampler` | Automatic | Each rank instantiates its own `DataLoader`; coordinator pushes per-epoch shard plan |
| `torchrun --nproc_per_node=N` | Auto-promote (single-host) / `fdl @cluster <cmd>` (multi-host) | One binary; the launcher trampoline forks rank children |
| `MASTER_ADDR` / `MASTER_PORT` env | `fdl.cluster.yml controller.host:port` / `ClusterBuilder.controller(host).port(p)` | First-class topology |
| `model.to(rank)` | `model_factory(device)` | Per-device model in closure |
| Equal batch per GPU only | `ElChe` cadence | Heterogeneous GPU support; weighted gradient averaging |
| `NCCL` / `Gloo` backend choice | `ElCheMode::{NcclSync, NcclCadence, CpuSync, CpuCadence, CpuAsync}` | Five modes via one enum; default `NcclCadence` |
| No built-in A/B testing | Swap `.elche(ElCheConfig::*)` | One line per mode |
| No elastic membership | The run survives rank death (survivors absorb the work); `max_failure` + `ShutdownWithSave` | Controller-driven checkpoint retry / role failover |

### Key differences

- **Same code, every tier**: no conditional code for CPU / 1 GPU /
  N GPUs / multi-host. `Trainer::builder(...).run()` is the universal
  entry.
- **Process-per-rank** (matches PyTorch's model) with controller-bound
  rendezvous, auto-promoted on 2+ GPUs visible. The launcher
  trampoline owns fan-out, supervision, and cleanup.
- **Heterogeneous GPUs**: PyTorch DDP requires equal batch sizes
  across ranks. flodl's ElChe assigns proportional work based on
  measured throughput; the convergence guard vetoes anchor growth
  when weight-space divergence rises.
- **A/B testing**: swap `ElCheConfig::nccl_cadence()` for any of the
  five modes with one line. PyTorch has no equivalent mechanism.
- **No CUDA before `Trainer::run`** invariant: user binaries must not
  instantiate CUDA tensors in `main()`. Use
  `flodl::sys::detect_gpus()` (loads no GPU runtime; reads `nvidia-smi`
  on NVIDIA, the KFD topology on AMD) for any pre-run GPU query.

See the [DDP Reference](ddp/01-reference.md) for complete API documentation.

## Quick Reference Table

| PyTorch | flodl | Notes |
|---------|-------|-------|
| `torch.zeros(2,3)` | `Tensor::zeros(&[2,3], opts)?` | Shape as slice |
| `x + y` | `x.add(&y)?` | Returns `Result` |
| `x.requires_grad_(True)` | `Variable::new(x, true)` | Set at creation |
| `loss.backward()` | `loss.backward()?` | Same pattern |
| `optimizer.zero_grad()` | `opt.zero_grad()` | Same pattern |
| `optimizer.step()` | `opt.step()?` | Returns `Result` |
| `nn.Linear(in, out)` | `Linear::new(in, out)?` | Returns `Result` |
| `F.relu(x)` | `x.relu()?` | Method on Variable |
| `F.mse_loss(a, b)` | `mse_loss(&a, &b)?` | Free function |
| `model.to(device)` | `module.move_to_device(device)` | |
| `with torch.no_grad():` | `no_grad(\|\| { })` or `NoGradGuard::new()` | Closure or RAII guard |
| `nn.Sequential(...)` | `FlowBuilder::from(...).through(...).build()?` | Fluent builder |
| `model.train()` | `module.train()` | |
| `model.eval()` | `module.eval()` | |
| `torch.save(...)` / `torch.load(...)` | `model.save_checkpoint("m.fdl")?` / `model.load_checkpoint("m.fdl")?` | Named `.fdl` format with `LoadReport` + structural hash validation |
| *(no built-in)* | `migrate_checkpoint_file(src, dst, &params, &bufs)?` | Remap parameter names across versions by shape+dtype matching |
| *(no built-in)* | `checkpoint_version(path)?` | Peek at checkpoint version (1=0.1.x, 2=0.2.0+) |
| `param.requires_grad = False` | `param.freeze()?` | Also: `unfreeze()`, `is_frozen()` |
| `Adam([{"params":..., "lr":...}])` | `Adam::with_groups().group(&p, lr).build()` | Per-group LR |
| `torch.cuda.memory_reserved()` | `gpu_allocated_bytes()?` | Bytes reserved by caching allocator |
| `x.pin_memory()` | `x.pin_memory()?` | Page-locked CPU memory for async transfers |
| `x.is_pinned()` | `x.is_pinned()` | Check if tensor is in pinned memory |
| `x.to(device, non_blocking=True)` | `x.to_device_async(device)?` | Non-blocking transfer (pair with `pin_memory`) |
| `x.to(memory_format=channels_last)` | `x.to_channels_last()?` | NHWC layout for Conv2d (8-35% on Tensor Cores) |
| `x.is_contiguous(channels_last)` | `x.is_channels_last()` | Check memory format |
| `torch.cuda.amp.autocast()` | `autocast(DType::Float16, \|\| { })` | Automatic mixed precision dispatch |
| `torch.cuda.amp.GradScaler()` | `GradScaler::new()` | Dynamic loss scaling for AMP |
| `torch.cuda.CUDAGraph()` | `GpuGraph::new()?` | CUDA graph capture/replay |
| `torch.cuda.graph(g)` | `gpu_graph_capture(warmup, pool, \|\| { })` | Convenience capture helper |
| `SummaryWriter` + TensorBoard | `Monitor::new(n).serve(3000)?` | Built-in live dashboard |

## See also

- [Porting Guide](porting.md) - AI-assisted porting with `fdl` and the `/port` skill
- [CLI documentation](cli/01-install.md) - project scaffolding (`fdl init`), libtorch management, `fdl api-ref`
- [Graph builder tutorial](tutorials/05-graph-builder.md) - FlowBuilder patterns in depth
- [DDP Reference](ddp/01-reference.md) - multi-GPU training

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Rust for PyTorch Users](tutorials/00-rust-primer.md) | Next: [Tensors](tutorials/01-tensors.md)
