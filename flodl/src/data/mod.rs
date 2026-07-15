//! Async data loading pipeline.
//!
//! Prefetches batches in a background thread with pinned memory and async
//! CUDA transfers, keeping GPUs fed without stalling.
//!
//! Two dataset traits: [`DataSet`] for per-item access (simple) and
//! [`BatchDataSet`] for bulk loading (efficient). Both return `Vec<Tensor>`
//! to support arbitrary numbers of tensors per sample (input, target,
//! mask, weight, etc).
//!
//! Build the model first, then the loader, so VRAM probing reflects actual
//! free memory.
//!
//! # Quick start
//!
//! ```ignore
//! use flodl::data::*;
//!
//! struct MyData { x: Tensor, y: Tensor }
//! impl DataSet for MyData {
//!     fn len(&self) -> usize { self.x.shape()[0] as usize }
//!     fn get(&self, i: usize) -> Result<Vec<Tensor>> {
//!         Ok(vec![self.x.select(0, i as i64)?, self.y.select(0, i as i64)?])
//!     }
//! }
//!
//! let loader = DataLoader::from_dataset(MyData { x, y })
//!     .batch_size(64)
//!     .device(Device::CUDA(0))
//!     .build()?;
//!
//! for epoch in 0..100 {
//!     for batch in loader.epoch(epoch) {
//!         let b = batch?;
//!         let input = Variable::new(b[0].clone(), true);
//!         let target = Variable::new(b[1].clone(), false);
//!         // ...
//!     }
//! }
//! ```

pub mod sampler;
pub mod loader;
pub mod datasets;
pub mod records;
pub(crate) mod budget;
pub(crate) mod prefetch;
pub(crate) mod sample_cache;
pub(crate) mod vram_pool;

pub use sampler::{Sampler, RandomSampler, SequentialSampler};
pub use loader::{DataLoader, DataLoaderBuilder, EpochIterator};
pub use records::FixedStrideRecords;
pub(crate) use budget::prefetch_depth_from_vram;

use crate::tensor::{Result, Tensor};

// ---------------------------------------------------------------------------
// DataSet (per-item)
// ---------------------------------------------------------------------------

/// A dataset that provides individual samples.
///
/// Each call to [`get`](DataSet::get) returns one sample as a `Vec<Tensor>`.
/// Position 0 is typically the input, position 1 the target, and so on.
/// All tensors should be on CPU.
///
/// The loader handles batching (stacking), shuffling, device transfer,
/// and prefetching automatically.
///
/// # Purity: `get(index)` returns the raw sample, every time
///
/// `get` must be a **pure function of the index**: the same index always
/// yields the same bytes, with no per-call randomness. The data plane
/// retains samples **by index** across its staging tiers (RAM sample
/// cache, disk stage, VRAM sample pool — all on by default) and re-serves
/// the retained bytes on every later epoch, so a `get` that augments
/// per call (the PyTorch `__getitem__` convention) would have its first
/// realization silently frozen and served for the rest of the run.
///
/// Augmentation therefore does not belong in `get`. Apply it downstream
/// of the loader as a deterministic on-device transform — e.g. a graph
/// `.map` stage keyed by sample/step — so the raw sample stays resident
/// once and each use derives its variant on device.
///
/// Debug builds probe this contract: the first staged fetch of a run is
/// fetched twice and compared, and a divergence panics with this
/// explanation. Release builds skip the probe entirely.
///
/// ```should_panic
/// use flodl::data::{DataLoader, DataSet};
/// use flodl::tensor::{Device, Result, Tensor};
/// use std::sync::atomic::{AtomicU32, Ordering};
///
/// struct AugmentsInGet(AtomicU32);
///
/// impl DataSet for AugmentsInGet {
///     fn len(&self) -> usize {
///         4
///     }
///     fn get(&self, _index: usize) -> Result<Vec<Tensor>> {
///         // Per-call randomness — the contract violation: the staged
///         // copy would be frozen at whatever this returned first.
///         let noise = self.0.fetch_add(1, Ordering::Relaxed) as f32;
///         Ok(vec![Tensor::from_f32(&[noise], &[1], Device::CPU)?])
///     }
/// }
///
/// // Debug builds catch it at the first staged fetch.
/// let _ = DataLoader::from_dataset(AugmentsInGet(AtomicU32::new(0)))
///     .batch_size(2)
///     .build();
/// ```
///
/// # Thread safety
///
/// Requires `Send + Sync` because a background thread calls `get()` while
/// the GPU trains. In practice, datasets backed by `Vec`, `Tensor`, file
/// handles, or mmap'd buffers satisfy this automatically. If you have
/// `Rc`-based data, wrap it in `Arc` instead.
///
/// # Example
///
/// ```ignore
/// struct Mnist { images: Tensor, labels: Tensor }
///
/// impl DataSet for Mnist {
///     fn len(&self) -> usize { self.images.shape()[0] as usize }
///     fn get(&self, index: usize) -> Result<Vec<Tensor>> {
///         Ok(vec![
///             self.images.select(0, index as i64)?,
///             self.labels.select(0, index as i64)?,
///         ])
///     }
/// }
/// ```
pub trait DataSet: Send + Sync {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Fetch a single sample by index.
    ///
    /// Returns a `Vec<Tensor>` where each position has a consistent meaning
    /// across calls (e.g., position 0 = input, position 1 = target).
    /// Tensors should be on CPU.
    fn get(&self, index: usize) -> Result<Vec<Tensor>>;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DataSet for std::sync::Arc<dyn DataSet> {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn get(&self, index: usize) -> Result<Vec<Tensor>> {
        (**self).get(index)
    }

    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

// ---------------------------------------------------------------------------
// BatchDataSet (per-batch)
// ---------------------------------------------------------------------------

/// A dataset that provides entire batches at once.
///
/// Implement this when your storage can produce a batch more efficiently
/// than N individual gets (e.g., contiguous memory-mapped arrays, database
/// bulk reads, or pre-stacked tensors).
///
/// [`DataSet`] is automatically promoted to `BatchDataSet` via
/// `DataSetAdapter` (call `get()` N times and stack position-wise).
///
/// Requires `Send + Sync` for background prefetch (see [`DataSet`] docs).
///
/// # Contract
///
/// Each tensor in the returned `Vec` must have dimension 0 as the batch
/// dimension, with length equal to `indices.len()`. The number of tensors
/// and their shapes (beyond dim 0) must be consistent across calls.
///
/// Row content must be a **pure function of the row's index** — same
/// purity contract as [`DataSet::get`], for the same reason: the staging
/// tiers (notably the VRAM sample pool, on by default) retain rows by
/// index and re-serve them on later epochs, so per-call randomness in
/// `get_batch` is silently frozen at its first realization. Augmentation
/// belongs downstream, as a deterministic on-device transform. Debug
/// builds probe the contract once per prefetch worker.
pub trait BatchDataSet: Send + Sync {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Fetch a batch of samples by indices.
    ///
    /// Returns `Vec<Tensor>` where each tensor has `indices.len()` rows
    /// along dimension 0. Position i must have consistent shape[1..]
    /// across calls.
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>>;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Picks: augmentation as repeated indices + the keyed transform seam
// ---------------------------------------------------------------------------

/// The identity of one scheduled use of a sample — the key the
/// transform seam derives its per-view randomness from.
///
/// With `.augment(k)` every sample appears `k` times per epoch in the
/// shuffled schedule; each appearance is one *pick*, and `repeat` says
/// which of the `k` views this delivery is. The same `(sample, repeat,
/// epoch, seed)` always keys the same bytes — augmentation stays
/// deterministic and reproducible, statistically equivalent to
/// per-call randomness without ever violating the raw-sample purity
/// contract on [`DataSet::get`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PickKey {
    /// Sample (chunk) id — what the staging tiers key by.
    pub sample: usize,
    /// Which of the epoch's `k` views of this sample, in `0..k`.
    pub repeat: u32,
    /// Epoch number.
    pub epoch: u64,
    /// The run's shuffle seed.
    pub seed: u64,
}

impl PickKey {
    pub(crate) fn from_pick(pick: usize, augment: usize, epoch: u64, seed: u64) -> Self {
        let k = augment.max(1);
        PickKey {
            sample: pick / k,
            repeat: (pick % k) as u32,
            epoch,
            seed,
        }
    }

    /// A deterministic RNG unique to this pick: same key, same stream,
    /// every run. The mixing constants are frozen — checkpointed runs
    /// reproduce their augmentation across flodl versions.
    pub fn rng(&self) -> crate::rng::Rng {
        let mut h = self
            .seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.epoch.wrapping_mul(0xBF58_476D_1CE4_E5B9));
        h ^= (self.sample as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= u64::from(self.repeat).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^= h >> 33;
        crate::rng::Rng::seed(h)
    }
}

type TransformInner = dyn Fn(Vec<Tensor>, &[PickKey]) -> Result<Vec<Tensor>> + Send + Sync;

/// The deterministic per-batch transform applied at delivery — the
/// sanctioned home for augmentation. Receives the delivered rows (raw,
/// already on the target device, freshly assembled — never aliasing
/// the staging tiers) and one [`PickKey`] per row; must be a pure
/// function of `(rows, keys)` and preserve the row count. Runs live on
/// every delivery: retained tiers hold raw samples only, and each pick
/// re-derives its view. Construct via the `transform(..)` builder
/// setters (or [`TransformFn::new`]).
#[derive(Clone)]
pub struct TransformFn(std::sync::Arc<TransformInner>);

impl TransformFn {
    pub fn new(
        f: impl Fn(Vec<Tensor>, &[PickKey]) -> Result<Vec<Tensor>> + Send + Sync + 'static,
    ) -> Self {
        TransformFn(std::sync::Arc::new(f))
    }
}

impl std::fmt::Debug for TransformFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TransformFn(..)")
    }
}

/// Decode a pick stream into the sample (chunk) ids the data plane
/// fetches and stages by. At `augment = 1` this is the identity.
pub(crate) fn picks_to_samples(picks: &[usize], augment: usize) -> Vec<usize> {
    let k = augment.max(1);
    if k == 1 {
        return picks.to_vec();
    }
    picks.iter().map(|&p| p / k).collect()
}

/// Apply the delivery transform: build one key per pick and hand the
/// batch over. Callers pass the PICK stream (not decoded sample ids).
pub(crate) fn apply_transform(
    transform: &TransformFn,
    tensors: Vec<Tensor>,
    picks: &[usize],
    augment: usize,
    epoch: usize,
    seed: u64,
) -> Result<Vec<Tensor>> {
    let keys: Vec<PickKey> = picks
        .iter()
        .map(|&p| PickKey::from_pick(p, augment, epoch as u64, seed))
        .collect();
    // Determinism probe (debug builds, once per process): the same
    // keys must yield the same bytes — a transform drawing from global
    // RNG instead of `PickKey::rng` silently breaks reproducibility
    // and the seed-computable-ahead property the data plane relies on.
    // Inputs are deep-copied first (the transform may mutate in place).
    #[cfg(all(debug_assertions, not(test)))]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static TRANSFORM_PROBED: AtomicBool = AtomicBool::new(false);
        if !TRANSFORM_PROBED.swap(true, Ordering::Relaxed) {
            if let (Ok(c1), Ok(c2)) = (deep_copy_rows(&tensors), deep_copy_rows(&tensors)) {
                if let (Ok(a), Ok(b)) =
                    ((transform.0)(c1, &keys), (transform.0)(c2, &keys))
                {
                    let identical = a.len() == b.len()
                        && a.iter().zip(&b).all(|(x, y)| tensor_identical(x, y));
                    assert!(
                        identical,
                        "flodl data: the delivery transform returned different \
                         content for the same PickKeys. It must be a pure \
                         function of (rows, keys) — derive per-view randomness \
                         from PickKey::rng(), never from global RNG state, or \
                         augmentation stops being reproducible and the \
                         schedule stops being computable ahead. This probe \
                         runs in debug builds only."
                    );
                }
            }
        }
    }
    (transform.0)(tensors, &keys)
}

/// Owned deep copies for the determinism probe: the transform may
/// mutate its input in place, so each probe application needs its own
/// storage.
#[cfg(all(debug_assertions, not(test)))]
fn deep_copy_rows(rows: &[Tensor]) -> Result<Vec<Tensor>> {
    rows.iter()
        .map(|t| {
            let out = Tensor::empty(
                &t.shape(),
                crate::tensor::TensorOptions {
                    dtype: t.dtype(),
                    device: t.device(),
                },
            )?;
            out.copy_(t, false)?;
            Ok(out)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Purity probe (debug builds)
// ---------------------------------------------------------------------------

/// Debug-build purity probe: compare two fetches of the same index and
/// panic with the contract explanation if they diverge. The staging
/// tiers retain samples by index, so an impure fetch is silently frozen
/// at its first realization — a correctness bug worth a loud stop in
/// debug builds. One probe per run/worker; it is a spot check, not a
/// proof.
#[cfg(debug_assertions)]
pub(crate) fn assert_fetch_pure(what: &str, first: &[Tensor], second: &[Tensor]) {
    let identical = first.len() == second.len()
        && first.iter().zip(second).all(|(a, b)| tensor_identical(a, b));
    if identical {
        return;
    }
    panic!(
        "flodl data: {what} returned different content for the same index. \
         It must be a pure function of the index: the staging cascade (RAM \
         sample cache, disk stage, VRAM sample pool) retains samples by \
         index and re-serves them on later epochs, so per-call randomness \
         (e.g. augmentation inside the dataset, the PyTorch __getitem__ \
         convention) is silently frozen at its first realization. Move \
         augmentation out of the dataset and apply it downstream as a \
         deterministic on-device transform (e.g. a graph `.map` stage). \
         This probe runs in debug builds only."
    );
}

/// Exact elementwise equality with a NaN-tolerant confirmation pass:
/// `eq_tensor` is the fast device-side compare, but NaN != NaN would
/// accuse a legitimately pure dataset that contains NaNs, so a mismatch
/// is re-checked host-side treating NaN == NaN before declaring
/// impurity. Unreadable tensors never accuse.
#[cfg(debug_assertions)]
fn tensor_identical(a: &Tensor, b: &Tensor) -> bool {
    if a.shape() != b.shape() || a.dtype() != b.dtype() {
        return false;
    }
    let n = a.numel();
    if n == 0 {
        return true;
    }
    let eq_count = a
        .eq_tensor(b)
        .and_then(|e| e.sum())
        .and_then(|s| s.item());
    match eq_count {
        Ok(c) if c as i64 == n => true,
        _ => match (a.to_f64_vec(), b.to_f64_vec()) {
            (Ok(x), Ok(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .zip(&y)
                        .all(|(p, q)| p == q || (p.is_nan() && q.is_nan()))
            }
            _ => true,
        },
    }
}

// ---------------------------------------------------------------------------
// DataSetAdapter (bridges DataSet -> BatchDataSet)
// ---------------------------------------------------------------------------

/// Adapter that promotes a [`DataSet`] into a [`BatchDataSet`] by calling
/// [`get()`](DataSet::get) for each index and stacking position-wise.
///
/// Sample fetches go through a read-through
/// [`SampleCache`](sample_cache::SampleCache): dormant (budget 0, pure
/// pass-through) until the streaming loader installs a RAM budget at
/// `epoch()`.
/// Opaque [`BatchDataSet`] implementors bypass this adapter entirely
/// and stay uncached by design.
pub(crate) struct DataSetAdapter<D: DataSet> {
    pub(crate) inner: D,
    cache: std::sync::Arc<sample_cache::SampleCache>,
}

impl<D: DataSet> DataSetAdapter<D> {
    /// Adapter with a dormant, self-owned cache (pass-through until a
    /// loader installs a budget; nothing ever does for adapters built
    /// via [`batch_dataset_from`] — the caching tier there is the DDP
    /// worker's staging layer).
    pub(crate) fn new(inner: D) -> Self {
        let cache = std::sync::Arc::new(sample_cache::SampleCache::new(inner.len()));
        DataSetAdapter { inner, cache }
    }

    /// Adapter sharing `cache` with the loader that will budget it.
    pub(crate) fn with_cache(
        inner: D,
        cache: std::sync::Arc<sample_cache::SampleCache>,
    ) -> Self {
        DataSetAdapter { inner, cache }
    }

    fn fetch(&self, index: usize) -> Result<Vec<Tensor>> {
        self.cache.get_or_fetch(index, || self.inner.get(index))
    }
}

impl<D: DataSet> BatchDataSet for DataSetAdapter<D> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        if indices.is_empty() {
            return Ok(Vec::new());
        }

        let n = indices.len() as i64;

        // Fetch first sample to learn shapes and tensor count
        let first = self.fetch(indices[0])?;
        let n_tensors = first.len();

        // Pre-allocate output tensors with batch dim prepended: [n, ...sample_shape]
        let mut result: Vec<Tensor> = Vec::with_capacity(n_tensors);
        for t in &first {
            let sample_shape = t.shape();
            let mut batch_shape = Vec::with_capacity(1 + sample_shape.len());
            batch_shape.push(n);
            batch_shape.extend_from_slice(&sample_shape);
            result.push(Tensor::empty(
                &batch_shape,
                crate::tensor::TensorOptions {
                    dtype: t.dtype(),
                    device: t.device(),
                },
            )?);
        }

        // Copy first sample into row 0
        for (pos, t) in first.iter().enumerate() {
            result[pos].select(0, 0)?.copy_(t, false)?;
        }
        drop(first);

        // Fetch remaining samples one at a time: copy into pre-allocated output, then drop
        for (batch_idx, &idx) in indices.iter().enumerate().skip(1) {
            let sample = self.fetch(idx)?;
            if sample.len() != n_tensors {
                return Err(crate::tensor::TensorError::new(&format!(
                    "DataSetAdapter: sample {} has {} tensors, expected {} (same as sample 0)",
                    idx,
                    sample.len(),
                    n_tensors
                )));
            }
            for (pos, t) in sample.iter().enumerate() {
                result[pos].select(0, batch_idx as i64)?.copy_(t, false)?;
            }
        }

        Ok(result)
    }
}

/// Promote a per-sample [`DataSet`] into an opaque [`BatchDataSet`].
///
/// Batches are assembled by fetching each index and stacking
/// position-wise (`[n, ...sample_shape]`). Use this where an API takes
/// a `BatchDataSet` and you have per-sample data; APIs with a native
/// per-sample entry are better served directly ([`DataLoader::from_dataset`]
/// budgets the read-through sample cache, and the DDP trainer entries
/// stage per-sample through the reservation tier).
pub fn batch_dataset_from(dataset: impl DataSet + 'static) -> std::sync::Arc<dyn BatchDataSet> {
    std::sync::Arc::new(DataSetAdapter::new(dataset))
}

// ---------------------------------------------------------------------------
// Batch (named accessor wrapper)
// ---------------------------------------------------------------------------

/// A loaded batch of tensors with optional named access.
///
/// Supports both positional indexing (`batch[0]`) and named indexing
/// (`batch["image"]`). Names are set via [`DataLoaderBuilder::names`].
/// When names are not explicitly set, auto-generated positional names
/// ("0", "1", "2", ...) are used so both access patterns always work.
///
/// Batch owns its tensors. For resident mode, these are `index_select`
/// results (not views into the full dataset). For streaming mode, they
/// come from the prefetch channel. Ownership is consistent across all
/// paths.
///
/// # Example
///
/// ```ignore
/// let loader = DataLoader::from_dataset(data)
///     .batch_size(64)
///     .names(&["image", "letter", "case", "origin"])
///     .build()?;
///
/// for batch in loader.epoch(epoch) {
///     let b = batch?;
///     let images = &b["image"];
///     let letters = &b["letter"];
///     // positional still works:
///     let also_images = &b[0];
/// }
/// ```
pub struct Batch {
    names: Vec<String>,
    tensors: Vec<Tensor>,
}

impl Batch {
    /// Create a new batch from tensors with explicit names.
    pub(crate) fn new(tensors: Vec<Tensor>, names: Vec<String>) -> Self {
        debug_assert_eq!(
            names.len(),
            tensors.len(),
            "Batch: names count ({}) must match tensor count ({})",
            names.len(),
            tensors.len(),
        );
        Batch { names, tensors }
    }

    /// Create a new batch with auto-generated positional names ("0", "1", ...).
    #[allow(dead_code)]
    pub(crate) fn new_unnamed(tensors: Vec<Tensor>) -> Self {
        let names: Vec<String> = (0..tensors.len()).map(|i| i.to_string()).collect();
        Batch { names, tensors }
    }

    /// Get a tensor by position.
    pub fn get(&self, index: usize) -> &Tensor {
        &self.tensors[index]
    }

    /// Get a tensor by name. Returns `None` if the name is not found.
    pub fn get_named(&self, name: &str) -> Option<&Tensor> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.tensors[i])
    }

    /// Whether the batch contains a tensor with the given name.
    pub fn has(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    /// The names of the tensors in this batch.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Number of tensors in this batch.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the batch contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Consume the batch and return the underlying tensors.
    pub fn into_vec(self) -> Vec<Tensor> {
        self.tensors
    }

    /// Consume the batch and return names and tensors.
    pub fn into_parts(self) -> (Vec<String>, Vec<Tensor>) {
        (self.names, self.tensors)
    }
}

impl std::ops::Index<usize> for Batch {
    type Output = Tensor;
    fn index(&self, i: usize) -> &Tensor {
        &self.tensors[i]
    }
}

impl std::ops::Index<&str> for Batch {
    type Output = Tensor;
    fn index(&self, name: &str) -> &Tensor {
        let pos = self.names.iter().position(|n| n == name);
        match pos {
            Some(i) => &self.tensors[i],
            None => panic!(
                "Batch: unknown field '{}'. Available: [{}]",
                name,
                self.names.join(", ")
            ),
        }
    }
}

impl std::fmt::Debug for Batch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("names", &self.names)
            .field("len", &self.tensors.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::test_opts;

    #[test]
    fn pick_key_decode_and_rng_determinism() {
        // Intrinsic decode: pick / k = sample, pick % k = repeat.
        let k = PickKey::from_pick(7, 3, 5, 42);
        assert_eq!((k.sample, k.repeat), (2, 1));
        // k = 1: picks and samples coincide, repeat is always 0.
        let k1 = PickKey::from_pick(7, 1, 5, 42);
        assert_eq!((k1.sample, k1.repeat), (7, 0));

        // Same key = same stream, every run.
        let mut r1 = PickKey::from_pick(7, 3, 5, 42).rng();
        let mut r2 = PickKey::from_pick(7, 3, 5, 42).rng();
        let a: Vec<f64> = (0..8).map(|_| r1.f64()).collect();
        let b: Vec<f64> = (0..8).map(|_| r2.f64()).collect();
        assert_eq!(a, b);

        // Any component change = a different stream (repeat here: the
        // whole point of keyed views).
        let mut r3 = PickKey {
            repeat: 2,
            ..PickKey::from_pick(7, 3, 5, 42)
        }
        .rng();
        let c: Vec<f64> = (0..8).map(|_| r3.f64()).collect();
        assert_ne!(a, c);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "pure function of the index")]
    fn purity_probe_panics_on_divergent_fetches() {
        use crate::tensor::Device;
        let a = vec![Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap()];
        let b = vec![Tensor::from_f32(&[1.0, 3.0], &[2], Device::CPU).unwrap()];
        assert_fetch_pure("DataSet::get", &a, &b);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn purity_probe_accepts_identical_and_nan_content() {
        use crate::tensor::Device;
        // Identical values compare equal on the fast device path.
        let a = vec![Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap()];
        let b = vec![Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap()];
        assert_fetch_pure("DataSet::get", &a, &b);

        // NaN != NaN fails eq_tensor, but a pure dataset containing
        // NaNs must not be accused: the NaN-tolerant host pass accepts.
        let n1 =
            vec![Tensor::from_f32(&[f32::NAN, 1.0], &[2], Device::CPU).unwrap()];
        let n2 =
            vec![Tensor::from_f32(&[f32::NAN, 1.0], &[2], Device::CPU).unwrap()];
        assert_fetch_pure("DataSet::get", &n1, &n2);

        // Empty tensors are trivially identical.
        let e1 = vec![Tensor::from_f32(&[], &[0], Device::CPU).unwrap()];
        let e2 = vec![Tensor::from_f32(&[], &[0], Device::CPU).unwrap()];
        assert_fetch_pure("DataSet::get", &e1, &e2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "pure function of the index")]
    fn purity_probe_panics_on_shape_divergence() {
        use crate::tensor::Device;
        let a = vec![Tensor::from_f32(&[1.0, 2.0], &[2], Device::CPU).unwrap()];
        let b = vec![Tensor::from_f32(&[1.0], &[1], Device::CPU).unwrap()];
        assert_fetch_pure("DataSet::get", &a, &b);
    }

    struct SimplePairs {
        x: Tensor,
        y: Tensor,
    }

    impl DataSet for SimplePairs {
        fn len(&self) -> usize {
            self.x.shape()[0] as usize
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            Ok(vec![
                self.x.select(0, index as i64)?,
                self.y.select(0, index as i64)?,
            ])
        }
    }

    struct MultiTarget {
        images: Tensor,
        letters: Tensor,
        cases: Tensor,
    }

    impl DataSet for MultiTarget {
        fn len(&self) -> usize {
            self.images.shape()[0] as usize
        }
        fn get(&self, index: usize) -> Result<Vec<Tensor>> {
            Ok(vec![
                self.images.select(0, index as i64)?,
                self.letters.select(0, index as i64)?,
                self.cases.select(0, index as i64)?,
            ])
        }
    }

    fn make_simple_data(n: usize) -> SimplePairs {
        let opts = test_opts();
        SimplePairs {
            x: Tensor::randn(&[n as i64, 4], opts).unwrap(),
            y: Tensor::randn(&[n as i64, 2], opts).unwrap(),
        }
    }

    fn make_multi_target(n: usize) -> MultiTarget {
        let opts = test_opts();
        MultiTarget {
            images: Tensor::randn(&[n as i64, 3, 8, 8], opts).unwrap(),
            letters: Tensor::randn(&[n as i64, 26], opts).unwrap(),
            cases: Tensor::randn(&[n as i64, 2], opts).unwrap(),
        }
    }

    #[test]
    fn test_dataset_adapter_stacks_position_wise() {
        let data = make_simple_data(10);
        let adapter = DataSetAdapter::new(data);
        let batch = adapter.get_batch(&[0, 1, 2]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].shape(), &[3, 4]); // 3 samples, 4 features
        assert_eq!(batch[1].shape(), &[3, 2]); // 3 samples, 2 targets
    }

    #[test]
    fn test_batch_dataset_from_promotes_dataset() {
        // The public promotion path (Trainer entries delegate here),
        // including through an already-erased Arc<dyn DataSet>.
        let erased: std::sync::Arc<dyn DataSet> =
            std::sync::Arc::new(make_simple_data(10));
        let batched = batch_dataset_from(erased);
        assert_eq!(batched.len(), 10);
        let batch = batched.get_batch(&[7, 8]).unwrap();
        assert_eq!(batch[0].shape(), &[2, 4]);
        assert_eq!(batch[1].shape(), &[2, 2]);
    }

    #[test]
    fn test_dataset_adapter_multi_target() {
        let data = make_multi_target(20);
        let adapter = DataSetAdapter::new(data);
        let batch = adapter.get_batch(&[5, 10, 15, 19]).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].shape(), &[4, 3, 8, 8]); // images
        assert_eq!(batch[1].shape(), &[4, 26]);        // letters
        assert_eq!(batch[2].shape(), &[4, 2]);          // cases
    }

    #[test]
    fn test_dataset_adapter_single_item() {
        let data = make_simple_data(5);
        let adapter = DataSetAdapter::new(data);
        let batch = adapter.get_batch(&[3]).unwrap();
        assert_eq!(batch[0].shape(), &[1, 4]);
        assert_eq!(batch[1].shape(), &[1, 2]);
    }

    #[test]
    fn test_dataset_adapter_empty_indices() {
        let data = make_simple_data(5);
        let adapter = DataSetAdapter::new(data);
        let batch = adapter.get_batch(&[]).unwrap();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_positional_indexing() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let b = Batch::new_unnamed(vec![t0, t1]);
        assert_eq!(b.len(), 2);
        assert!(!b.is_empty());
        assert_eq!(b[0].shape(), &[2, 3]);
        assert_eq!(b[1].shape(), &[2, 5]);
        assert_eq!(b.get(0).shape(), &[2, 3]);
    }

    #[test]
    fn test_batch_named_indexing() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let names = vec!["image".to_string(), "label".to_string()];
        let b = Batch::new(vec![t0, t1], names);
        assert_eq!(b["image"].shape(), &[2, 3]);
        assert_eq!(b["label"].shape(), &[2, 5]);
        // Positional still works
        assert_eq!(b[0].shape(), &[2, 3]);
        assert_eq!(b[1].shape(), &[2, 5]);
    }

    #[test]
    fn test_batch_has_and_names() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let names = vec!["image".to_string(), "label".to_string()];
        let b = Batch::new(vec![t0, t1], names);
        assert!(b.has("image"));
        assert!(b.has("label"));
        assert!(!b.has("mask"));
        assert_eq!(b.names(), &["image", "label"]);
    }

    #[test]
    fn test_batch_get_named() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let names = vec!["x".to_string(), "y".to_string()];
        let b = Batch::new(vec![t0, t1], names);
        assert!(b.get_named("x").is_some());
        assert_eq!(b.get_named("x").unwrap().shape(), &[2, 3]);
        assert!(b.get_named("z").is_none());
    }

    #[test]
    fn test_batch_auto_names() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let b = Batch::new_unnamed(vec![t0, t1]);
        // Auto-generated names are positional strings
        assert_eq!(b.names(), &["0", "1"]);
        assert_eq!(b["0"].shape(), &[2, 3]);
        assert_eq!(b["1"].shape(), &[2, 5]);
    }

    #[test]
    #[should_panic(expected = "unknown field 'missing'")]
    fn test_batch_named_index_panics_on_missing() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let b = Batch::new(vec![t0], vec!["image".to_string()]);
        let _ = &b["missing"];
    }

    #[test]
    fn test_batch_into_vec() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let b = Batch::new_unnamed(vec![t0, t1]);
        let v = b.into_vec();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].shape(), &[2, 3]);
    }

    #[test]
    fn test_batch_into_parts() {
        let opts = test_opts();
        let t0 = Tensor::zeros(&[2, 3], opts).unwrap();
        let t1 = Tensor::ones(&[2, 5], opts).unwrap();
        let names = vec!["a".to_string(), "b".to_string()];
        let b = Batch::new(vec![t0, t1], names);
        let (n, v) = b.into_parts();
        assert_eq!(n, &["a", "b"]);
        assert_eq!(v.len(), 2);
    }
}
