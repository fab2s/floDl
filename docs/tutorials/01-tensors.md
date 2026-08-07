# Tensors

Tensors are the fundamental data type in floDl - n-dimensional arrays of numbers backed by libtorch. This tutorial covers creation, operations, error handling, and memory management.

## Creating Tensors

All creation functions return `Result<Tensor>`.

```rust
use flodl::{Tensor, TensorOptions, Device, DType};

// From Rust data - data is copied into libtorch
let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU)?;

// Filled tensors
let opts = TensorOptions::default();  // Float32, CPU
let zeros = Tensor::zeros(&[3, 4], opts)?;
let ones = Tensor::ones(&[3, 4], opts)?;

// Random tensors (seed with manual_seed() for reproducibility)
let uniform = Tensor::rand(&[2, 3], opts)?;   // values in [0, 1)
let normal = Tensor::randn(&[2, 3], opts)?;   // standard normal

// Integer tensor (for indices, e.g. Embedding lookups)
let idx = Tensor::from_i64(&[0, 3, 7], &[3], Device::CPU)?;
```

### Options

`TensorOptions` is a plain struct with `dtype` and `device` fields:

```rust
let opts = TensorOptions { dtype: DType::Float64, ..Default::default() };
let t = Tensor::ones(&[4], opts)?;

let gpu_opts = TensorOptions { device: Device::CUDA(0), ..Default::default() };
let t = Tensor::zeros(&[3, 3], gpu_opts)?;
```

## Shape Inspection

```rust
let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Device::CPU)?;

t.shape();   // [2, 3]
t.ndim();    // 2
t.numel();   // 6
t.dtype();   // DType::Float32
t.device();  // Device::CPU
```

## Operations

Operations return new tensors - originals are never modified. Every operation
returns `Result<Tensor>`, and the `?` operator propagates errors:

```rust
let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU)?;
let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], Device::CPU)?;

let result = a.add(&b)?.matmul(&b)?.relu()?;
```

### Arithmetic

```rust
a.add(&b)?          // element-wise a + b
a.sub(&b)?          // element-wise a - b
a.mul(&b)?          // element-wise a * b (Hadamard product)
a.div(&b)?          // element-wise a / b
a.matmul(&b)?       // matrix multiplication

a.mul_scalar(0.5)?  // multiply every element by a scalar
a.add_scalar(1.0)?  // add a scalar to every element
```

### Activations

```rust
t.relu()?           // max(0, x)
t.sigmoid()?        // 1 / (1 + exp(-x))
t.tanh()?           // hyperbolic tangent
t.gelu()?           // Gaussian Error Linear Unit
t.silu()?           // x * sigmoid(x) (Swish)
t.selu()?           // scaled ELU (self-normalizing)
t.hardswish()?      // efficient Swish approximation
t.hardsigmoid()?    // piecewise-linear sigmoid
t.prelu(&weight)?   // parametric ReLU
t.softmax(-1)?      // softmax along dimension
t.log_softmax(-1)?  // log(softmax(x)) - numerically stable
```

### Math

```rust
t.exp()?            // e^x
t.log()?            // ln(x)
t.sqrt()?           // square root
t.neg()?            // negation
t.abs()?            // absolute value
t.pow_scalar(2.0)?  // element-wise power
t.clamp(-1.0, 1.0)? // clamp to range
t.clamp_min(0.0)?   // clamp from below
t.clamp_max(1.0)?   // clamp from above
t.reciprocal()?     // 1/x
t.sign()?           // sign (-1, 0, or 1)
t.floor()?          // round down
t.ceil()?           // round up
t.round()?          // round to nearest
t.trunc()?          // truncate toward zero
t.frac()?           // fractional part
```

#### Trigonometry

```rust
t.sin()?            // sine
t.cos()?            // cosine
t.tan()?            // tangent
t.asin()?           // arc sine
t.acos()?           // arc cosine
t.atan()?           // arc tangent
```

#### Numerically Stable

```rust
t.log1p()?          // ln(1 + x) - stable for small x
t.expm1()?          // exp(x) - 1 - stable for small x
t.log2()?           // log base 2
t.log10()?          // log base 10
t.erf()?            // Gauss error function
t.erfc()?           // complementary error function (1 - erf)
```

#### Modular Arithmetic

```rust
t.fmod(3.0)?                // C-style remainder (scalar)
t.fmod_tensor(&divisor)?    // C-style remainder (tensor)
t.remainder(3.0)?           // Python-style modulo (scalar)
t.remainder_tensor(&other)? // Python-style modulo (tensor)
```

#### Fused Operations

```rust
// self + mat1 @ mat2 * alpha + self * beta - single kernel
t.addmm(&mat1, &mat2, 1.0, 1.0)?;

// self + tensor1 * tensor2 * value - fused multiply-accumulate
t.addcmul(&t1, &t2, 1.0)?;

// self + tensor1 / tensor2 * value - fused divide-accumulate
t.addcdiv(&t1, &t2, 1.0)?;

// linear interpolation: self + (end - self) * weight
t.lerp(&end, 0.5)?;
t.lerp_tensor(&end, &weights)?;  // per-element weight

// element-wise closeness check
t.isclose(&other, 1e-5, 1e-8)?;
```

### Reductions

```rust
t.sum()?                      // reduce all elements to scalar
t.mean()?                     // mean of all elements
t.sum_dim(1, true)?           // reduce along dim, keep dimension
t.mean_dim(1, true)?          // mean along dim
t.max()?                      // scalar max
t.min()?                      // scalar min
t.max_dim(1, true)?           // max along dim, keep dimension
t.min_dim(1, true)?           // min along dim, keep dimension
t.argmax(-1)?                 // index of max along dim
t.var()?                      // variance of all elements
t.std()?                      // standard deviation
t.var_dim(1, true)?           // variance along dim
t.std_dim(1, true)?           // std along dim
t.prod()?                     // product of all elements
t.prod_dim(1, true)?          // product along dim
t.cumsum(0)?                  // cumulative sum along dim
t.logsumexp(1, true)?         // log(sum(exp(x))) - numerically stable
t.norm()?                     // L2 norm of all elements
```

### Shape Manipulation

```rust
t.reshape(&[6, 1])?          // new shape, same data
t.transpose(0, 1)?           // swap two dimensions
t.flatten(0, -1)?            // flatten all dims
t.squeeze(0)?                // remove dim of size 1
t.unsqueeze(0)?              // add dim of size 1
t.unsqueeze_many(&[0, 2])?   // add multiple dims at once
t.permute(&[1, 0])?          // arbitrary axis reorder
t.contiguous()?              // ensure contiguous memory layout
t.movedim(0, 2)?             // move dimension to new position
t.flip(&[0, 1])?            // reverse along dimensions
t.roll(2, 0)?               // circular shift along dim
t.diagonal(0, 0, 1)?        // extract diagonal
t.tile(&[2, 3])?            // repeat by tiling
t.triu(0)?                   // upper triangular
t.tril(0)?                   // lower triangular
```

### Slicing and Joining

```rust
t.narrow(0, 1, 2)?                  // extract a contiguous slice along dim
t.select(0, 1)?                     // pick one index along dim, removing that dim
a.cat(&b, 0)?                       // concatenate two tensors along dim
Tensor::cat_many(&[&a, &b, &c], 0)? // concatenate many tensors
Tensor::stack(&[&a, &b], 0)?        // stack along new dim
t.index_select(0, &indices)?        // gather slices at given indices
t.split(2, 0)?                      // split into chunks of size 2 along dim
t.chunk(3, 0)?                      // split into N equal chunks along dim
t.unbind(0)?                        // remove a dim, returning Vec<Tensor>
t.repeat(&[2, 3])?                  // repeat tensor along each dim
t.pad(&[1, 1], 0.0)?               // constant-value padding
t.pad_mode(&[1, 1], 1, 0.0)?       // mode: 0=constant, 1=reflect, 2=replicate, 3=circular
t.batches(32)?                       // split into mini-batches along dim 0
Tensor::meshgrid(&[&x, &y])?        // coordinate grids (ij indexing)
```

### Comparisons and Conditionals

```rust
let mask = x.gt(&threshold)?;        // element-wise >
let mask = x.gt_scalar(0.0)?;        // compare with scalar
let mask = x.lt_scalar(1.0)?;        // less than scalar
let mask = x.ge_scalar(0.0)?;        // greater or equal
let mask = x.le_scalar(1.0)?;        // less or equal
let y = Tensor::where_cond(&mask, &a, &b)?;  // conditional select

// Element-wise min/max of two tensors
let z = a.maximum(&b)?;
let z = a.minimum(&b)?;
```

### Similarity and Normalization

```rust
// Cosine similarity along a dimension
let sim = a.cosine_similarity(&b, 1, 1e-8)?;

// Lp normalization along a dimension
let normed = t.normalize(2.0, 1)?;   // L2-normalize along dim 1

// Masked fill - set elements where mask is true
let y = t.masked_fill(&mask, 0.0)?;
```

## Extracting Data

Copy tensor data back to Rust vectors:

```rust
let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU)?;

let data: Vec<f32> = t.to_f32_vec()?;   // [1.0, 2.0, 3.0]
let item: f64 = t.select(0, 0)?.item()?; // scalar value as f64
```

`to_f64_vec()` and `to_i64_vec()` are also available.

## Memory Management

Tensors are backed by C++ memory managed through libtorch. Rust's ownership
system handles cleanup automatically via `Drop` - you never need to free
tensors manually.

```rust
{
    let t = Tensor::zeros(&[1000, 1000], TensorOptions::default())?;
    // ... use t ...
} // t is dropped here - C++ memory freed immediately
```

This is a fundamental advantage over garbage-collected languages. In Go,
tensor memory can linger until the GC runs, requiring explicit `Release()`
calls and VRAM budget heuristics. In Rust, memory is freed deterministically
at the end of the owning scope.

`Clone` on a `Tensor` shares the underlying data (like PyTorch's shallow
copy). The C++ `TensorImpl` is reference-counted internally by libtorch.

### Diagnostics

Utilities for debugging memory and performance issues:

```rust
use flodl::{live_tensor_count, rss_kb};

// Number of live C++ Tensor handles (created but not yet dropped).
// If this grows over time, there is a tensor handle leak.
println!("live handles: {}", live_tensor_count());

// Current process RSS in kilobytes (Linux only).
println!("RSS: {}MB", rss_kb() / 1024);

// Count autograd nodes reachable from a tensor - measures graph
// complexity. Compare against Python to detect decomposed-op bloat.
let loss = mse_loss(&pred, &target)?;
println!("autograd nodes: {}", loss.data().autograd_node_count());
```

### Peak VRAM Tracking

On CUDA devices, you can track peak GPU memory usage during a training step or
any other section of code. These match the standard PyTorch memory diagnostics:

| flodl | PyTorch equivalent |
|---|---|
| `gpu_peak_active_bytes()` | `torch.cuda.max_memory_allocated()` |
| `gpu_peak_reserved_bytes()` | `torch.cuda.max_memory_reserved()` |
| `gpu_reset_peak_stats()` | `torch.cuda.reset_peak_memory_stats()` |

"Active" bytes are memory currently holding tensor data. "Reserved" bytes
include the CUDA caching allocator's free pool - memory that libtorch has
obtained from the driver but is not currently in use by any tensor. The gap
between the two tells you how much allocator headroom exists.

A typical pattern for profiling a training step:

```rust
use flodl::{gpu_reset_peak_stats, gpu_peak_active_bytes, gpu_peak_reserved_bytes,
            gpu_empty_cache};

// Flush the allocator cache so reserved starts from a clean baseline
gpu_empty_cache();
gpu_reset_peak_stats();

// --- run one training step ---
let output = model.forward(&batch)?;
let loss = mse_loss(&output, &targets)?;
loss.backward()?;
optimizer.step()?;
optimizer.zero_grad()?;

// Read peaks
let active_mb = gpu_peak_active_bytes()? as f64 / 1048576.0;
let reserved_mb = gpu_peak_reserved_bytes()? as f64 / 1048576.0;
println!("peak active: {active_mb:.1} MB, peak reserved: {reserved_mb:.1} MB");
```

The `_idx` variants (`gpu_peak_active_bytes_idx`, `gpu_peak_reserved_bytes_idx`,
`gpu_reset_peak_stats_idx`) accept an explicit device index for multi-GPU setups.

## Device Transfer

```rust
let gpu = t.to_device(Device::CUDA(0))?;   // move to GPU
let cpu = gpu.to_device(Device::CPU)?;  // move back to CPU

if flodl::gpu_available() {
    println!("CUDA devices: {}", flodl::gpu_device_count());
}
```

> **Note**: `gpu_device_count()` and `gpu_available()` initialize
> libtorch's CUDA context as a side effect. That's safe in
> single-process code or after `Trainer::run` has fanned out. If you
> need a GPU count *before* `Trainer::run` (e.g. to log the topology
> from `main()`), use `flodl::sys::detect_gpus()` instead - it shells
> out to `nvidia-smi` without touching libtorch, so it won't poison
> spawned children's CUDA contexts on multi-GPU rigs. See
> [Multi-GPU Training](11-multi-gpu.md) for the "no CUDA before
> `Trainer::run`" invariant.

### Non-blocking Device Transfer

The default `to_device()` blocks until the transfer completes. For CPU-to-GPU
transfers, you can overlap the copy with CPU work by using pinned memory and an
asynchronous transfer:

```rust
use flodl::{Tensor, Device, TensorOptions, gpu_synchronize};

let cpu_tensor = Tensor::randn(&[256, 512], TensorOptions::default())?;

// Pin the CPU tensor into page-locked memory (requires CUDA)
let pinned = cpu_tensor.pin_memory()?;

// Launch the transfer - returns immediately
let gpu = pinned.to_device_async(Device::CUDA(0))?;

// ... do CPU work while the DMA transfer runs ...

// Synchronize before using the GPU tensor
gpu_synchronize(0);
```

`pin_memory()` allocates the tensor in page-locked (pinned) host memory, which
the GPU can DMA directly without an intermediate staging copy. This matters most
when you are streaming large batches to the GPU and want to keep the CPU busy
with data preprocessing for the next batch.

You can check whether a tensor is already pinned with `t.is_pinned()`.

## Reproducibility

Seed libtorch's RNG before creating random tensors or models:

```rust
flodl::manual_seed(42);  // seeds CPU + CUDA RNGs
```

After seeding, `Tensor::rand`, `Tensor::randn`, dropout masks, and weight
initialization all produce deterministic results.

For CPU-side randomness (shuffling datasets, augmentation), use the `Rng`
struct:

```rust
use flodl::Rng;

let mut rng = Rng::seed(42);
rng.shuffle(&mut data);
```

See [Training](04-training.md) for full reproducibility setup.

## cuDNN Benchmark Mode

For fixed-size workloads (fixed batch size, fixed image dimensions), enabling cuDNN
benchmark mode lets cuDNN auto-tune convolution algorithms on the first call:

```rust
flodl::set_cudnn_benchmark(true);  // opt-in, 5-10% speedup for fixed shapes
```

Leave this off for dynamic-shape workloads (variable-length sequences,
multi-resolution images) - the warmup cost can hurt throughput.

## Memory Format (Channels Last)

By default, 4D tensors use the NCHW memory layout (batch, channels, height, width).
GPUs with Tensor Cores (NVIDIA Volta and newer) can run convolutions significantly
faster when tensors are stored in NHWC order, also called "channels last."

```rust
let images = Tensor::randn(&[8, 3, 224, 224], TensorOptions::default())?;

// Convert to channels-last layout
let images_cl = images.to_channels_last()?;

assert!(images_cl.is_channels_last());
assert_eq!(images_cl.shape(), &[8, 3, 224, 224]); // logical shape unchanged
```

The logical shape stays the same - only the physical memory stride order changes.
This avoids format-conversion overhead inside cuDNN convolution kernels. On Tensor
Core GPUs, expect an 8--35% speedup for Conv2d-heavy workloads, depending on layer
sizes and batch dimensions.

Convert your input tensors and model weights to channels-last format before
training. flodl's Conv2d will pick up the layout automatically.

## Foreach Operations

Foreach operations apply the same operation to a list of tensors in a single
fused CUDA kernel launch. When you have dozens or hundreds of parameter tensors
(typical in any real model), this eliminates per-tensor kernel launch overhead.

All foreach functions are associated functions on `Tensor`:

```rust
use flodl::Tensor;

let params: Vec<Tensor> = model.parameters();

// Zero all parameter gradients in one launch
Tensor::foreach_zero_(&params)?;

// Scale every tensor by a constant
Tensor::foreach_mul_scalar_(&params, 0.99)?;

// Add a constant to every tensor
Tensor::foreach_add_scalar_(&params, 1e-8)?;
```

There are also multi-list variants that operate on pairs of tensor lists
element-wise:

```rust
// params[i] += grads[i] * alpha, for all i
Tensor::foreach_add_list_(&params, &grads, -0.01)?;

// params[i] = lerp(params[i], targets[i], weight), for all i
Tensor::foreach_lerp_scalar_(&params, &targets, 0.1)?;

// In-place sqrt of every tensor
Tensor::foreach_sqrt_(&params)?;

// Compute the L2 norm of each tensor - returns a Vec<Tensor> of scalars
let norms: Vec<Tensor> = Tensor::foreach_norm(&params, 2.0)?;
```

You typically do not need to call these directly. flodl's fused optimizers
(Adam, AdamW) and gradient clipping routines use foreach operations internally
to minimize kernel launch overhead. They are exposed in the public API for
advanced use cases like custom optimizers or manual parameter surgery.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [PyTorch Migration Reference](../pytorch_migration.md) | Next: [Autograd](02-autograd.md)
