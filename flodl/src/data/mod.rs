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
pub(crate) mod prefetch;

pub use sampler::{Sampler, RandomSampler, SequentialSampler};
pub use loader::{DataLoader, DataLoaderBuilder, EpochIterator};
pub(crate) use loader::prefetch_depth_from_vram;

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
// DataSetAdapter (bridges DataSet -> BatchDataSet)
// ---------------------------------------------------------------------------

/// Adapter that promotes a [`DataSet`] into a [`BatchDataSet`] by calling
/// [`get()`](DataSet::get) for each index and stacking position-wise.
pub(crate) struct DataSetAdapter<D: DataSet> {
    pub(crate) inner: D,
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
        let first = self.inner.get(indices[0])?;
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
            let sample = self.inner.get(idx)?;
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
        let adapter = DataSetAdapter { inner: data };
        let batch = adapter.get_batch(&[0, 1, 2]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].shape(), &[3, 4]); // 3 samples, 4 features
        assert_eq!(batch[1].shape(), &[3, 2]); // 3 samples, 2 targets
    }

    #[test]
    fn test_dataset_adapter_multi_target() {
        let data = make_multi_target(20);
        let adapter = DataSetAdapter { inner: data };
        let batch = adapter.get_batch(&[5, 10, 15, 19]).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].shape(), &[4, 3, 8, 8]); // images
        assert_eq!(batch[1].shape(), &[4, 26]);        // letters
        assert_eq!(batch[2].shape(), &[4, 2]);          // cases
    }

    #[test]
    fn test_dataset_adapter_single_item() {
        let data = make_simple_data(5);
        let adapter = DataSetAdapter { inner: data };
        let batch = adapter.get_batch(&[3]).unwrap();
        assert_eq!(batch[0].shape(), &[1, 4]);
        assert_eq!(batch[1].shape(), &[1, 2]);
    }

    #[test]
    fn test_dataset_adapter_empty_indices() {
        let data = make_simple_data(5);
        let adapter = DataSetAdapter { inner: data };
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
