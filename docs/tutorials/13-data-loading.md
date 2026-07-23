# Tutorial 13: Data Loading

Async data loading with automatic VRAM management. The `DataLoader`
handles batching, shuffling, device transfer, and prefetching so your
training loop stays clean.

> **Prerequisites**: [Tensors](01-tensors.md) and
> [Training](04-training.md). CUDA GPU recommended but not required.

> **Time**: ~15 minutes.

## Quick start

```rust
use flodl::*;
use std::sync::Arc;

// Implement the dataset trait
struct MyDataset { /* ... */ }

impl BatchDataSet for MyDataset {
    fn len(&self) -> usize { 60_000 }
    fn get_batch(&self, indices: &[usize]) -> Vec<Tensor> {
        let images = /* load images for indices */;
        let labels = /* load labels for indices */;
        vec![images, labels]
    }
}

let dataset = Arc::new(MyDataset::new());

let mut loader = DataLoader::from_batch_dataset(dataset)
    .batch_size(32)
    .device(Device::CUDA(0))
    .names(&["image", "label"])
    .build()?;

for epoch in 0..100 {
    for batch in loader.epoch(epoch) {
        let batch = batch?;
        let images = &batch["image"];   // already on GPU
        let labels = &batch["label"];
        // ... training ...
    }
}
```

## Dataset traits

floDl provides two dataset traits. Choose based on how your data is
stored:

### DataSet (per-item)

Returns one sample at a time. The loader stacks samples into batches
automatically.

```rust
impl DataSet for MnistDataset {
    fn len(&self) -> usize { self.images.len() }

    fn get(&self, index: usize) -> Vec<Tensor> {
        vec![
            self.images[index].clone(),
            self.labels[index].clone(),
        ]
    }
}
```

Best for: datasets where each sample is a separate file or DB row.

### BatchDataSet (per-batch)

Returns an entire batch at once. More efficient when the data source
supports bulk access (memory-mapped files, databases, pre-batched
tensors).

```rust
impl BatchDataSet for PreloadedDataset {
    fn len(&self) -> usize { self.num_samples }

    fn get_batch(&self, indices: &[usize]) -> Vec<Tensor> {
        let idx = Tensor::from_slice_i64(
            &indices.iter().map(|&i| i as i64).collect::<Vec<_>>()
        );
        vec![
            self.images.index_select(0, &idx),
            self.labels.index_select(0, &idx),
        ]
    }
}
```

Best for: pre-loaded datasets, memory-mapped files, GPU-resident data.

Both traits require `Send + Sync` so the prefetch worker can access
them from a background thread.

## Bundled datasets

`flodl::data::datasets` ships parsers for common benchmark data — pure
parsers with no download logic (fetch the files however you like):

- `Mnist` / `Cifar10` — classic vision sets, parsed into RAM.
- `Cifar10Disk` — same files read per sample from disk (the
  larger-than-RAM path; implements `DataSet`, so the staging tiers
  compose above it).
- `Shakespeare` — character-level LM sequences from raw text.
- `TokenShards` — pre-tokenized LM corpora in NumPy `.npy` shard files
  (OLMo / olmo-mix, nanoGPT-style dumps).

`TokenShards` is the path for real LM pretraining data: each shard is
a 1-D array of token ids (`|u1`, `<u2`, `<u4`, `<i4` or `<i8`), and
samples are non-overlapping `seq_len` windows with the target shifted
by one. Shards are read lazily with positioned reads — a multi-GB
corpus costs no RAM up front — and windows never cross shard
boundaries:

```rust
use flodl::data::datasets::TokenShards;

// Every .npy in the directory, sorted by name; or TokenShards::open(&paths, seq_len)
let data = TokenShards::open_dir("data/olmo-mix", 1024)?;
println!("{} windows over {} tokens", data.len(), data.total_tokens());

// Per-sample entry point -> staging cascade (RAM cache, disk stage, VRAM pool)
let mut loader = DataLoader::from_dataset(data)
    .batch_size(32)
    .streaming()
    .build()?;
// batch[0]: [32, 1024] Int64 inputs, batch[1]: targets shifted by one
```

## Named batch access

The `Batch` type supports both positional and named access:

```rust
let loader = DataLoader::from_batch_dataset(dataset)
    .batch_size(32)
    .names(&["image", "letter", "case", "origin"])
    .build()?;

for batch in loader.epoch(0) {
    let batch = batch?;

    // Positional access
    let image = &batch[0];

    // Named access
    let image = &batch["image"];
    let letter = &batch["letter"];

    // Introspection
    assert!(batch.has("origin"));
    assert_eq!(batch.names(), &["image", "letter", "case", "origin"]);
    assert_eq!(batch.len(), 4);
}
```

If `.names()` is not called, auto-generated positional names ("0", "1",
...) are used.

## Resident vs streaming mode

The loader automatically selects the best mode based on available VRAM:

### Resident mode

When the dataset fits in 75% of free VRAM, the entire dataset is loaded
onto the GPU once at `build()` time. Per-epoch reshuffling uses GPU-side
`index_select` with a shuffled permutation tensor. Zero CPU-GPU transfer
after the initial load.

```
Build:   pin_memory() -> to_device() (one-time transfer)
Epoch:   index_select(shuffled_permutation) (GPU-only)
```

### Streaming mode

When the dataset is too large, a persistent background worker thread
handles batching and transfer:

```
Worker thread:  get_batch(indices) -> pin_memory() -> StreamGuard + to_device_async()
                -> CudaEvent (signals readiness)
Main thread:    event.synchronize() (typically instant due to prefetch)
                -> use batch
```

The worker runs on a dedicated CUDA stream, overlapping data transfer
with training computation on the default stream.

### Forcing a mode

```rust
// Force streaming (useful for benchmarking or preserving VRAM headroom)
.streaming()

// Force a specific prefetch depth
.prefetch(8)
```

## VRAM-aware prefetch

In streaming mode, the prefetch depth adapts automatically to VRAM:

- **Bootstrap**: 4 batches at `build()` time (conservative, model not yet loaded)
- **epoch(0)**: re-probes free VRAM after model allocation, fills to cap
- **epoch(N)**: re-probes each epoch, adapts to fragmentation and
  activation memory changes
- **OOM fallback**: if resident mode fails with CUDA OOM, automatically
  retries with streaming

### Configuration

```rust
// Use up to 90% of total VRAM for data (default)
.vram_max_usage(0.90)

// Use up to 80% (more headroom for activations)
.vram_max_usage(0.80)

// Manual override (disables automatic adaptation)
.prefetch(16)

// Manual resize between epochs
loader.auto_resize();
```

The default cap of 90% leaves 10% headroom for activation memory,
gradients, and CUDA allocator overhead.

## Shuffling and sampling

By default, data is shuffled each epoch using a `RandomSampler` with
deterministic per-epoch permutations:

```rust
// epoch 0: seed=42+0 -> permutation A
// epoch 1: seed=42+1 -> permutation B
// epoch 0 again: same seed -> same permutation A (reproducible)
```

### Control shuffling

```rust
// Custom seed
.seed(12345)

// Disable shuffling (sequential order every epoch)
.shuffle(false)

// Custom sampler
.sampler(Box::new(MyCustomSampler::new()))
```

### Drop last batch

```rust
// Drop incomplete final batch (default: true)
.drop_last(true)
```

The default is `true` to avoid a BatchNorm footgun: a final batch of
size 1 produces NaN variance. Set to `false` for evaluation/inference
where every sample matters.

## Augmentation: repeated picks + a keyed transform

flodl treats augmentation as two orthogonal, deterministic pieces
instead of per-call randomness hidden in the dataset:

```rust
let loader = DataLoader::from_dataset(my_data)
    .batch_size(64)
    .augment(4)                       // each sample: 4 views per epoch
    .transform(|mut rows, keys| {     // derive each view from its key
        for (i, key) in keys.iter().enumerate() {
            let mut rng = key.rng();  // same key = same bytes, every run
            // e.g. flip row i when rng.bernoulli(0.5), crop offset from
            // rng.usize(pad), noise from rng.f32() ...
        }
        Ok(rows)
    })
    .build()?;
```

- **`.augment(k)`** is pure scheduling: the epoch becomes a shuffle of
  `len() * k` *picks*, so each sample appears `k` times, spread across
  the epoch, and batch counts scale by `k`. Every pick fetches the same
  raw bytes - staged once across the caching tiers - and counts as one
  unit of work for DDP scheduling. Without a transform, `k > 1` is
  plain oversampling.
- **`.transform(f)`** runs at delivery, after device transfer, on every
  batch. It receives the rows plus one `PickKey { sample, repeat,
  epoch, seed }` per row; `key.rng()` gives a stateless RNG unique to
  that view, so augmentation is exactly reproducible across runs, ranks,
  and checkpoint resumes - statistically equivalent to stochastic
  augmentation, strictly better for debugging. Both knobs exist on
  `TrainerConfig` / `Trainer::builder` for DDP with identical semantics.

### Why not augment inside `get()`?

The PyTorch `__getitem__` habit - random crop/flip inside the dataset -
breaks under flodl's staging tiers: the RAM sample cache, disk stage,
and VRAM pool (all on by default) retain samples **by index** and
re-serve those bytes on later epochs, so per-call randomness would be
silently frozen at its first realization. `DataSet::get()` /
`BatchDataSet::get_batch` therefore carry a purity contract: same
index, same bytes, every call. Debug builds probe it (one double-fetch
compare per run) and panic with an explanation if it is violated;
release builds skip the probe.

The payoff for keeping raw bytes in the tiers: a VRAM-pooled sample
uploads once and derives all `k` views on device, and the transform can
never corrupt the retained data - delivered batches are always freshly
assembled storage. Keep transforms as tensor ops (they run on the
target device); a transform that round-trips to host defeats residency.

## DDP integration

Pass the dataset directly to `Trainer::builder` or `TrainerConfig` -
the framework constructs a per-rank `DataLoader` against each rank's
dataset shard automatically:

```rust
let ddp = Trainer::builder(model_factory, optim_factory, train_fn)
    .dataset(dataset)    // Arc<dyn BatchDataSet>
    .batch_size(32)
    .num_epochs(10)
    .run()?;
```

Under DDP, each rank's loader operates independently:

- **Per-rank backend selection**: a 16 GB rank can go resident while a
  6 GB rank on the same training run streams. No
  lowest-common-denominator constraint.
- **Proportional sharding**: the coordinator computes shard sizes from
  `partition_ratios` (or auto-balances by throughput via ElChe) and
  pushes the epoch plan to each worker. Fast ranks get larger shards.
- **No cross-rank transfer in the data path**: each rank loads its own
  shard from its own `DataSet` impl. The DataLoader is otherwise
  unaware that a cluster exists.

### Streaming from external sources

`DataSet` / `BatchDataSet` are pull-based traits - the body of `get()`
/ `get_batch()` decides where the samples come from. The framework's
"resident" vs "streaming" modes are about CPU → VRAM transfer; the
underlying source can be RAM, mmap, disk, network, S3, a database, or
anything else accessible from Rust.

For source-streaming patterns:

```rust
struct S3Dataset { /* ... bucket handle, prefetch pool, etc. ... */ }

impl BatchDataSet for S3Dataset {
    fn len(&self) -> usize { /* total samples in dataset */ self.total }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        // Fetch on-demand. Cache locally as you go, parallelize
        // requests if useful - all up to your impl.
        let bytes = self.fetch_indices(indices)?;
        self.decode_to_tensors(&bytes, indices.len())
    }
}
```

The DataLoader's prefetch worker keeps `K` future batches in flight on
a dedicated CUDA stream, so the network round-trip for batch N+1 can
overlap with batch N's compute. No special hooks needed; just
implement the trait and pass it.

## Builder reference

| Method | Default | Description |
|--------|---------|-------------|
| `.batch_size(usize)` | Required | Batch size per rank |
| `.device(Device)` | CPU | Target device (the per-rank loader auto-targets its own CUDA device under DDP) |
| `.seed(u64)` | 42 | RNG seed for shuffling |
| `.shuffle(bool)` | true | Enable shuffling |
| `.sampler(Box<dyn Sampler>)` | - | Custom sampler (overrides shuffle) |
| `.prefetch(usize)` | Auto | Override auto-detected prefetch depth |
| `.vram_max_usage(f64)` | 0.90 | Max VRAM fraction for prefetch |
| `.ram_max_usage(f64)` | 0.50 | Available-RAM fraction for the reader ring + sample cache |
| `.sample_cache(bool)` | true | Read-through RAM sample cache (later epochs read from RAM) |
| `.disk_stage(gb)` | 0 = off | Local-disk overflow tier under the sample cache |
| `.disk_stage_dir(path)` | temp dir | Where the disk stage's pack file lives |
| `.vram_pool(bool)` | true | Device-resident sample pool in leftover VRAM |
| `.activation_reserve(bytes)` | Auto | Declared first-step VRAM reserve for prefetch sizing |
| `.augment(usize)` | 1 | Views per sample per epoch (pick-space schedule) |
| `.transform(fn)` | - | Deterministic delivery transform, keyed per `PickKey` |
| `.streaming()` | Auto | Force streaming mode |
| `.names(&[&str])` | Positional | Name batch tensor positions |
| `.drop_last(bool)` | true | Drop incomplete final batch |

Under DDP the same memory knobs exist on the trainer — `TrainerConfig::with_vram_max_usage` / `with_ram_max_usage` / `with_sample_cache` / `with_disk_stage` / `with_disk_stage_dir` (or the chained `DdpBuilder` twins) — and govern each rank's prefetch channel, device sample pool, and staging tiers with the same defaults and clamps. One sizing policy serves both paths; co-hosted ranks split the host-RAM share in proportion to their schedule, each rank's disk stage writes its own pid-unique pack file, and `FLODL_VRAM_POOL=off` now disables the device sample pool on the solo loader path too (previously DDP-only).

## DataLoader methods

| Method | Description |
|--------|-------------|
| `.epoch(n)` | Get epoch iterator (reshuffles, adapts prefetch) |
| `.len()` | Number of samples |
| `.num_batches()` | Number of batches per epoch |
| `.batch_size()` | Batch size |
| `.device()` | Target or gather device |
| `.is_resident()` | Whether in resident mode |
| `.prefetch_depth()` | Current prefetch depth |
| `.set_prefetch_depth(n)` | Override prefetch depth |
| `.auto_resize()` | Re-probe VRAM and adapt prefetch |
| `.names()` | Tensor names for each batch position |

---

Previous: [DDP Builder](12-async-ddp.md) |
Next: [Tutorial 14: HuggingFace Integration](14-flodl-hf.md)
