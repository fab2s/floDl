# Troubleshooting

Real error messages and how to fix them.

---

## Build Errors

### `error: could not find native static library 'torch_cpu'`

libtorch is not found. The build system needs `LIBTORCH_PATH` set to the
libtorch directory.

**Fix (Docker - recommended):** All builds should run in the Docker container.
```bash
fdl build        # CPU
fdl cuda-build   # CUDA
```

**Fix (host):** Use the download script to install libtorch and set up paths:
```bash
curl -sL https://raw.githubusercontent.com/flodl-labs/flodl/main/download-libtorch.sh | sh
```
This downloads libtorch to `~/.local/lib/libtorch` and prints the exports to
add to your shell profile. Or set the variables manually:
```bash
export LIBTORCH_PATH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib
export LIBRARY_PATH=$LIBTORCH_PATH/lib
```

### `error[E0433]: could not find 'flodl' in the list of imported crates`

Cargo can't resolve the flodl dependency.

**Fix:** Check your `Cargo.toml`. If using a git dependency:
```toml
[dependencies]
flodl = { git = "https://github.com/flodl-labs/flodl.git" }
```

Make sure you have network access inside the container. If behind a proxy,
configure git:
```bash
git config --global http.proxy http://proxy:port
```

### `undefined reference to 'cudaGetDeviceCount'`

CUDA libraries aren't linked. This happens on CUDA builds when the linker
drops CUDA libs with `--as-needed`.

**Fix:** Make sure your `main.rs` calls the CUDA link anchor:
```rust
fn main() -> flodl::Result<()> {
    flodl_sys::flodl_force_cuda_link();
    // ... rest of your code
}
```

And build with the `cuda` feature:
```bash
cargo build --features cuda
```

### `error: linker 'cc' not found`

Missing C/C++ toolchain. The Docker images include this, but if building on
host:

**Fix:**
```bash
# Ubuntu/Debian
sudo apt-get install gcc g++ pkg-config

# macOS
xcode-select --install
```

---

## Docker Errors

### `permission denied while trying to connect to the Docker daemon`

**Fix:** Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in, then retry
```

### `could not select device driver "nvidia"` / `nvidia-container-cli: initialization error`

NVIDIA Container Toolkit is not installed or the driver is missing.

**Fix:**
```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify:**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### `service "cuda" depends on ... which is not healthy`

GPU not available to Docker. For CPU-only development:

**Fix:** Use CPU targets instead:
```bash
fdl build    # not cuda-build
fdl test     # not cuda-test
```

---

## Runtime Errors

### `TensorError: shape mismatch in matmul: [2, 3] x [4, 5]`

Matrix dimensions don't align. Inner dimensions must match for matmul.

**Fix:** Check your layer dimensions. For `Linear::new(in, out)`, the input
tensor's last dimension must equal `in`:
```rust
// Input is [batch, 10] - so first linear must accept 10
let model = FlowBuilder::from(Linear::new(10, 32)?)  // [batch, 10] -> [batch, 32]
    .through(Linear::new(32, 1)?)                     // [batch, 32] -> [batch, 1]
    .build()?;
```

### `TensorError: shape mismatch: cannot add [4, 8] and [4, 16]`

Shapes don't broadcast. Common with `also()` (residual connections) - input
and output must have the same shape.

**Fix:** Make sure the residual branch preserves dimensions:
```rust
// Wrong: also() adds input (dim 8) to Linear output (dim 16)
FlowBuilder::from(Linear::new(4, 8)?)
    .also(Linear::new(8, 16)?)   // 8 != 16, shape mismatch

// Right: preserve dimension through the residual
FlowBuilder::from(Linear::new(4, 8)?)
    .also(Linear::new(8, 8)?)    // 8 == 8, ok
```

### `cannot borrow 'optimizer' as mutable, as it is not declared as mutable`

Rust borrow checker: you need `mut` for things that change state.

**Fix:**
```rust
let mut optimizer = Adam::new(&params, 0.001);  // add mut
```

### `thread 'main' panicked at 'called Result::unwrap() on an Err value'`

An operation failed and you used `.unwrap()` instead of `?`.

**Fix:** Use `?` to propagate errors, and make sure your function returns `Result<T>`:
```rust
fn main() -> flodl::Result<()> {
    let t = Tensor::zeros(&[2, 3], opts)?;  // ? not unwrap()
    Ok(())
}
```

---

## Checkpoint Errors

### `checkpoint error: parameter count mismatch (expected 6, got 4)`

The model architecture changed since the checkpoint was saved.

**Fix:** Ensure the model definition is identical when loading:
```rust
// Save and load must use the same architecture
let model = FlowBuilder::from(Linear::new(1, 32)?)
    .through(GELU)
    .through(Linear::new(32, 1)?)
    .build()?;

model.save_checkpoint("model.fdl")?;

// Later - rebuild the SAME architecture before loading
let model = FlowBuilder::from(Linear::new(1, 32)?)
    .through(GELU)
    .through(Linear::new(32, 1)?)
    .build()?;

let report = model.load_checkpoint("model.fdl")?;
```

### `checkpoint error: invalid magic bytes`

The file is not a valid `.fdl` checkpoint (wrong format, corrupted, or a
PyTorch `.pt` file).

**Fix:** floDl uses its own `.fdl` format. You cannot load PyTorch checkpoints
directly. Retrain or export from PyTorch as numpy and reconstruct tensors.

---

## Visualization

### `dot: command not found` / SVG rendering fails

Graphviz is not installed.

**Fix (Docker):** The floDl Docker images include graphviz. Use `fdl shell`.

**Fix (host):**
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz
```

### `model.svg()` returns empty or minimal output

The graph has no nodes. Make sure you've built the graph with `FlowBuilder`:

```rust
let model = FlowBuilder::from(Linear::new(4, 8)?)
    .through(GELU)
    .build()?;
// Now model.dot() and model.svg() produce output
```

---

## Training Issues

### Loss is NaN

Common causes:
1. **Learning rate too high** - gradients explode
2. **Log of zero or negative** - `log(0)` = -inf, poisons everything
3. **Division by zero** in normalization

**Fix:**
```rust
// Lower the learning rate
let mut optimizer = Adam::new(&params, 1e-4);  // try 10x smaller

// Add gradient clipping
clip_grad_norm(&params, 1.0)?;

// Check for NaN in loss
let loss_val = loss.item()?;
if loss_val.is_nan() {
    eprintln!("NaN detected at epoch {}", epoch);
    break;
}
```

### Loss not decreasing

1. **Learning rate too low** - try 10x higher
2. **Model too small** - add capacity
3. **Data issue** - verify inputs and targets are correct

**Debugging checklist:**
```rust
// Print loss every epoch
println!("epoch {}: loss={:.6}", epoch, loss.item()?);

// Verify gradients exist
for p in &params {
    if let Some(g) = p.var().grad() {
        let norm = g.norm()?.item()?;
        println!("grad norm: {:.6}", norm);
    }
}

// Try a known-working config first (sine wave example)
// cargo run --example sine_wave
```

### Loss oscillates wildly

**Fix:** Reduce learning rate and/or add gradient clipping:
```rust
let mut optimizer = Adam::new(&params, 1e-4);
// ...
clip_grad_norm(&params, 0.5)?;
```

### Out of VRAM

The CUDA allocator ran out of GPU memory. Common when batch sizes are too large
or inference runs without disabling gradients.

**Diagnose first** - check how much VRAM you actually have and how it's used:
```rust
if let Ok((used, total)) = cuda_memory_info() {
    let active = cuda_active_bytes().unwrap_or(0);
    let reserved = cuda_allocated_bytes().unwrap_or(0);
    println!("VRAM: {:.0}/{:.0} MB (active: {:.0} MB, reserved: {:.0} MB)",
        used as f64 / 1e6, total as f64 / 1e6,
        active as f64 / 1e6, reserved as f64 / 1e6);
}
```

If `reserved` is much larger than `active`, the allocator is holding freed
blocks. Call `cuda_empty_cache()` to release them before checking again.

**Fix:**
- **Reduce batch size** - the single biggest lever for VRAM usage
- **Use `no_grad` for inference** - backward graphs consume significant memory:
  ```rust
  let pred = no_grad(|| model.forward(&input))?;
  ```
- **Use mixed precision** - `Float16`/`BFloat16` halves parameter memory:
  ```rust
  cast_parameters(&params, DType::Float16);
  let scaler = GradScaler::new();
  ```
- **Use smaller model dimensions** - reduce hidden sizes, fewer layers
- **Detach state between steps** - for recurrent models, call `model.detach_state()`
  to break gradient chains across time steps
- **Check for tensor leaks** - if VRAM grows linearly across epochs, tensors
  are being retained. Use `live_tensor_count()` to track:
  ```rust
  println!("live tensors: {}", live_tensor_count());
  ```

---

## Common Rust Compiler Messages

### `value used here after move`

You used a variable after it was moved into something else.

**Fix:** Clone before the move, or restructure to use references:
```rust
let t = Tensor::zeros(&[2, 3], opts)?;
let input = Variable::new(t.clone(), true);  // clone t before moving into Variable
// t is still usable
```

### `expected &Tensor, found Tensor`

A function expects a reference but you passed an owned value.

**Fix:** Add `&`:
```rust
let result = a.add(&b)?;  // &b, not b
```

### `the trait 'Module' is not implemented for ...`

You're passing something that doesn't implement `Module` to a graph builder method.

**Fix:** Check the type. All built-in layers implement `Module`. Custom types
need `impl Module for YourType { ... }`.

---

## Multi-GPU / DDP

**Start with `fdl probe`** (single-host) or `fdl @cluster probe`
(multi-host). It surfaces GPU/libtorch arch mismatches, NCCL version
skew across hosts, missing `nccl_socket_ifname:`, legacy schema keys
in `fdl.cluster.yml`, shared-data mount divergence, and dashboard
port collisions - most "it should work" cases land in its output.

For deeper troubleshooting (NCCL init failure, parameter mismatch,
CUDA context corruption, OOM on smaller GPU, cluster progressive
hangs), see the [DDP Reference - Troubleshooting](ddp.md#troubleshooting)
section.

Common quick fixes:
- **NCCL init fails**: `fdl probe` first. Then check `nvidia-smi topo -m` for peer connectivity. Switch to a `Cpu*` ElCheMode (`ElCheConfig::cpu_async()`) to bypass NCCL entirely.
- **NCCL version skew across hosts**: one host's libtorch ships NCCL 2.27, another's ships 2.26. Build a matching libnccl with `fdl nccl build` and wire it via the worker's `env: LD_PRELOAD:` block in `fdl.cluster.yml`.
- **CUBLAS_STATUS_EXECUTION_FAILED after NCCL**: also covered by the "no CUDA before `Trainer::run`" invariant - don't instantiate CUDA tensors in `main()`. Use `flodl::sys::detect_gpus()` for pre-run GPU queries.
- **Training hangs (cluster)**: usually stale child processes from a previous aborted run holding rendezvous ports or GPU memory. `fdl @cluster <cmd>` cleans these up pre-spawn, but kill-9 on the launcher bypasses cleanup - clear stragglers with `pkill -f flodl-rank` on each worker.
- **OOM on smaller GPU**: any anchor-based mode (`NcclCadence`, `CpuAsync`, `CpuCadence`) routes through ElChe, which proportionally shrinks the smaller GPU's batch count. Also: per-rank DataLoader backend selection - the larger GPU can go resident while the smaller streams.

---

## Getting Help

- **Examples:** `cargo run --example quickstart` and `cargo run --example sine_wave`
- **Tutorials:** Start with [Rust for PyTorch Users](tutorials/00-rust-primer.md)
- **Multi-GPU:** [DDP Reference](ddp.md) and [Multi-GPU Tutorial](tutorials/11-multi-gpu.md)
- **Migration guide:** [PyTorch -> floDl](pytorch_migration.md)
- **Issues:** [github.com/flodl-labs/flodl/issues](https://github.com/flodl-labs/flodl/issues)

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Mac / Apple Silicon](mac-apple-silicon.md)
