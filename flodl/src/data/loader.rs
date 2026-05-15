//! DataLoader: async data pipeline with automatic prefetching.
//!
//! Manages a background pipeline that keeps GPU(s) fed with data.
//! Supports two modes:
//!
//! - **Resident**: entire dataset fits in VRAM. Loaded once, reshuffled per epoch.
//! - **Streaming**: prefetch ring buffer with async H2D transfers.
//!
//! Mode is auto-detected at build time based on available VRAM.
//!
//! Build the model first, then the loader, so VRAM probing reflects
//! actual free memory after model allocation.

use std::sync::Arc;

use super::prefetch::PrefetchWorker;
use super::sampler::{RandomSampler, Sampler, SequentialSampler};
use super::{Batch, BatchDataSet, DataSet, DataSetAdapter};
use crate::tensor::{Device, Result, Tensor, TensorError};

/// Default fraction of total VRAM to use. Reserves 10% for activations,
/// gradients, and CUDA allocator overhead.
const VRAM_MAX_USAGE: f64 = 0.90;

/// Check whether the full dataset fits in VRAM (or RAM for CPU).
///
/// For CPU targets, always returns true (RAM is plentiful).
/// For CUDA targets, probes free VRAM and checks if the dataset fits
/// within the headroom budget.
fn can_fit_resident(n: usize, per_sample_bytes: usize, device: Device) -> bool {
    if !device.is_cuda() {
        return true;
    }

    let total_bytes = per_sample_bytes as u64 * n as u64;
    let idx = device.index() as i32;

    match crate::tensor::cuda_memory_info_idx(idx) {
        Ok((free, total)) => {
            let used = total.saturating_sub(free);
            let cap = (total as f64 * VRAM_MAX_USAGE) as u64;
            let budget = cap.saturating_sub(used);
            total_bytes < budget
        }
        Err(_) => false, // can't probe -> assume won't fit
    }
}

/// Bootstrap prefetch depth: small buffer for the period between
/// `build()` and the first `epoch()` call. The real depth is computed
/// at `epoch()` time when free VRAM reflects actual model allocation.
const BOOTSTRAP_PREFETCH: usize = 4;

/// Compute prefetch depth from VRAM usage cap.
///
/// `max_usage` is the fraction of **total** VRAM to use (default 0.90).
/// The prefetch budget is the gap between current usage and the cap,
/// minus `activation_reserve` bytes reserved for forward/backward
/// activation memory and gradients.
///
/// Called at each `epoch()` boundary. By that point the model, optimizer,
/// and any other allocations are done, so current usage is the real baseline.
pub(crate) fn prefetch_depth_from_vram(
    per_sample_bytes: usize,
    batch_size: usize,
    device: Device,
    max_usage: f64,
    activation_reserve: usize,
) -> usize {
    if !device.is_cuda() {
        return 2; // CPU: just double-buffer
    }

    let batch_bytes = per_sample_bytes * batch_size;
    if batch_bytes == 0 {
        return 2;
    }

    let idx = device.index() as i32;
    let (free, total) = crate::tensor::cuda_memory_info_idx(idx)
        .unwrap_or((0, 0));

    let used = (total as usize).saturating_sub(free as usize);
    let cap = (total as f64 * max_usage.clamp(0.5, 0.99)) as usize;
    let budget = cap.saturating_sub(used + activation_reserve);

    budget / batch_bytes
}

// ---------------------------------------------------------------------------
// DataLoaderBuilder
// ---------------------------------------------------------------------------

/// Builder for [`DataLoader`]. Constructed via
/// [`DataLoader::from_dataset`] or [`DataLoader::from_batch_dataset`].
pub struct DataLoaderBuilder {
    dataset: Box<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Option<Box<dyn Sampler>>,
    prefetch_depth: Option<usize>,
    seed: u64,
    drop_last: bool,
    force_streaming: bool,
    names: Option<Vec<String>>,
    vram_max_usage: f64,
}

impl DataLoaderBuilder {
    fn new(dataset: Box<dyn BatchDataSet>) -> Self {
        DataLoaderBuilder {
            dataset,
            batch_size: 0,
            device: Device::CPU,
            sampler: None,
            prefetch_depth: None,
            seed: 42,
            drop_last: true,
            force_streaming: false,
            names: None,
            vram_max_usage: 0.90,
        }
    }

    /// Set the batch size. Required.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Target device for loaded batches. Default: `Device::CPU`.
    ///
    /// For single-GPU training, set to `Device::CUDA(0)`. Data arrives
    /// on the GPU ready for forward pass.
    ///
    /// For DDP training, leave as `Device::CPU` -- data arrives in pinned
    /// memory and `forward_distributed` scatters to devices efficiently.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set the RNG seed for shuffling. Default: 42.
    ///
    /// Each epoch derives its permutation from `seed + epoch`, so different
    /// epochs produce different orderings but the same seed is always
    /// reproducible.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable shuffling. Default: `true` (uses [`RandomSampler`]).
    ///
    /// When `false`, uses [`SequentialSampler`] (indices in order every epoch).
    /// This is overridden if [`sampler`](DataLoaderBuilder::sampler) is called.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        if !shuffle {
            let n = self.dataset.len();
            self.sampler = Some(Box::new(SequentialSampler::new(n)));
        }
        self
    }

    /// Custom sampler. Overrides the [`shuffle`](DataLoaderBuilder::shuffle) setting.
    pub fn sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Override auto-detected prefetch depth (streaming mode only).
    ///
    /// Auto-detection fills `(1 - margin)` of free VRAM at build time.
    /// Use this to set a specific depth instead. Disables automatic
    /// per-epoch adaptation.
    ///
    /// Set to 0 for synchronous loading (no background thread).
    pub fn prefetch(mut self, depth: usize) -> Self {
        self.prefetch_depth = Some(depth);
        self
    }

    /// Maximum fraction of total VRAM to use for prefetch (streaming mode).
    ///
    /// Default: 0.90 (use up to 90% of total VRAM). At each `epoch()` call,
    /// the loader probes current VRAM usage and fills the gap between that
    /// usage and the cap with prefetch batches. The remaining headroom covers
    /// activation memory, gradients, and CUDA allocator overhead.
    ///
    /// The budget is computed at `epoch()` time (not `build()`), so the model
    /// can be loaded in any order. Clamped to `[0.50, 0.99]`.
    pub fn vram_max_usage(mut self, max_usage: f64) -> Self {
        self.vram_max_usage = max_usage.clamp(0.50, 0.99);
        self
    }

    /// Force streaming mode even when the dataset fits in memory.
    ///
    /// Useful for preserving VRAM headroom, testing the prefetch pipeline,
    /// or benchmarking resident vs streaming performance.
    pub fn streaming(mut self) -> Self {
        self.force_streaming = true;
        self
    }

    /// Name the tensor positions in each batch.
    ///
    /// Names enable `batch["image"]` access alongside positional `batch[0]`.
    /// The number of names must match the number of tensors returned by the
    /// dataset's `get()` / `get_batch()`.
    ///
    /// If not called, auto-generated positional names ("0", "1", ...) are used.
    ///
    /// ```ignore
    /// let loader = DataLoader::from_dataset(data)
    ///     .batch_size(64)
    ///     .names(&["image", "letter", "case", "origin"])
    ///     .build()?;
    ///
    /// let b = loader.epoch(0).next().unwrap()?;
    /// let images = &b["image"];
    /// ```
    pub fn names(mut self, names: &[&str]) -> Self {
        self.names = Some(names.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Drop the last incomplete batch if dataset size is not divisible
    /// by batch_size. Default: `true`.
    ///
    /// Defaulting to `true` avoids the well-known BatchNorm footgun:
    /// a final batch of size 1 produces NaN variance. Set to `false`
    /// for evaluation or inference where every sample matters.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Build the DataLoader.
    ///
    /// Performs auto-detection of resident vs streaming mode based on
    /// available VRAM. For resident mode, loads the entire dataset into
    /// GPU memory at this point.
    ///
    /// Build the model first, then the loader, so VRAM probing reflects
    /// actual free memory after model allocation.
    pub fn build(self) -> Result<DataLoader> {
        if self.dataset.is_empty() {
            return Err(TensorError::new("DataLoader: empty dataset"));
        }
        if self.batch_size == 0 {
            return Err(TensorError::new("DataLoader: batch_size must be > 0"));
        }

        // Destructure to avoid partial-move issues
        let DataLoaderBuilder {
            dataset,
            batch_size,
            device,
            sampler,
            prefetch_depth,
            seed,
            drop_last,
            force_streaming,
            names,
            vram_max_usage,
        } = self;

        let n = dataset.len();

        // Probe dataset size for mode decision
        let sample = dataset.get_batch(&[0])?;
        if sample.is_empty() {
            return Err(TensorError::new(
                "DataLoader: dataset returned empty tensor list",
            ));
        }
        let num_tensors = sample.len();
        let per_sample_bytes: usize = sample.iter().map(|t| t.nbytes()).sum();
        drop(sample);

        // Resolve names: validate if provided, auto-generate if not
        let names = match names {
            Some(ref n) if n.len() != num_tensors => {
                return Err(TensorError::new(&format!(
                    "DataLoader: names count ({}) does not match dataset tensor count ({})",
                    n.len(),
                    num_tensors,
                )));
            }
            Some(n) => n,
            None => (0..num_tensors).map(|i| i.to_string()).collect(),
        };

        let use_resident = !force_streaming && can_fit_resident(n, per_sample_bytes, device);

        // Wrap in Arc early so both paths can share it, and OOM fallback
        // from resident to streaming keeps the dataset alive.
        let dataset: Arc<dyn BatchDataSet> = Arc::from(dataset);
        let shuffle = sampler.is_none();

        let sampler = sampler.unwrap_or_else(|| {
            Box::new(RandomSampler::new(n, seed))
        });

        let user_set_depth = prefetch_depth.is_some();
        // Bootstrap depth: small buffer to start. The real depth is
        // computed at epoch() time when free VRAM reflects the actual
        // model allocation. User override skips adaptive sizing.
        let streaming_depth = prefetch_depth.unwrap_or(BOOTSTRAP_PREFETCH);
        if use_resident {
            match build_resident(Arc::clone(&dataset), batch_size, device, sampler, drop_last, names.clone()) {
                Ok(loader) => Ok(loader),
                Err(e) if device.is_cuda() && e.is_cuda_oom() => {
                    // VRAM estimate was wrong, fall back to streaming.
                    // Recreate sampler since build_resident consumed it.
                    let sampler: Box<dyn Sampler> = if shuffle {
                        Box::new(RandomSampler::new(n, seed))
                    } else {
                        Box::new(SequentialSampler::new(n))
                    };
                    crate::tensor::cuda_empty_cache();
                    build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, per_sample_bytes, vram_max_usage, user_set_depth, names)
                }
                Err(e) => Err(e),
            }
        } else {
            build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, per_sample_bytes, vram_max_usage, user_set_depth, names)
        }
    }
}

fn build_resident(
    dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    names: Vec<String>,
) -> Result<DataLoader> {
    let n = dataset.len();
    let all_indices: Vec<usize> = (0..n).collect();
    let tensors = dataset.get_batch(&all_indices)?;

    if tensors.is_empty() {
        return Err(TensorError::new(
            "DataLoader: dataset returned empty tensor list",
        ));
    }

    let gpu_data = if device.is_cuda() {
        let mut on_device = Vec::with_capacity(tensors.len());
        for t in &tensors {
            let pinned = t.pin_memory()?;
            on_device.push(pinned.to_device(device)?);
        }
        on_device
    } else {
        tensors
    };

    Ok(DataLoader {
        inner: LoaderInner::Resident(ResidentLoader {
            gpu_data,
            _dataset: dataset,
            device,
            batch_size,
            sampler,
            drop_last,
            names,
        }),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_streaming(
    dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    prefetch_depth: usize,
    per_sample_bytes: usize,
    vram_max_usage: f64,
    user_set_depth: bool,
    names: Vec<String>,
) -> Result<DataLoader> {
    let worker = PrefetchWorker::new(Arc::clone(&dataset), device, prefetch_depth);

    Ok(DataLoader {
        inner: LoaderInner::Streaming(StreamingLoader {
            _dataset: dataset,
            batch_size,
            device,
            sampler,
            drop_last,
            worker,
            names,
            per_sample_bytes,
            vram_max_usage,
            user_set_depth,
        }),
    })
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

/// Async data loader with automatic prefetching and device transfer.
///
/// Manages a background pipeline that keeps GPU(s) fed with data.
///
/// # Construction
///
/// ```ignore
/// let loader = DataLoader::from_dataset(my_data)
///     .batch_size(64)
///     .device(Device::CUDA(0))
///     .build()?;
/// ```
///
/// # Training loop
///
/// ```ignore
/// for epoch in 0..100 {
///     for batch in loader.epoch(epoch) {
///         let b = batch?;
///         let input = Variable::new(b[0].clone(), true);
///         let target = Variable::new(b[1].clone(), false);
///         let pred = model.forward(&input)?;
///         let loss = mse_loss(&pred, &target)?;
///         loss.backward()?;
///         model.step()?;
///     }
/// }
/// ```
pub struct DataLoader {
    pub(crate) inner: LoaderInner,
}

pub(crate) enum LoaderInner {
    Resident(ResidentLoader),
    Streaming(StreamingLoader),
}

impl DataLoader {
    /// Access the internal loader variant (for Graph integration).
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> &LoaderInner {
        &self.inner
    }
}

impl DataLoader {
    /// Create a DataLoader from a per-item [`DataSet`].
    ///
    /// Items are automatically stacked into batches.
    pub fn from_dataset<D: DataSet + 'static>(dataset: D) -> DataLoaderBuilder {
        DataLoaderBuilder::new(Box::new(DataSetAdapter { inner: dataset }))
    }

    /// Create a DataLoader from a per-batch [`BatchDataSet`].
    ///
    /// The dataset is responsible for returning properly batched tensors.
    pub fn from_batch_dataset<D: BatchDataSet + 'static>(dataset: D) -> DataLoaderBuilder {
        DataLoaderBuilder::new(Box::new(dataset))
    }

    /// Get an epoch iterator.
    ///
    /// Each call reshuffles the data (if using a random sampler) and
    /// returns an iterator over batches. Each batch is a [`Batch`]
    /// containing tensors already on the target device.
    ///
    /// The epoch number is passed to the sampler for deterministic
    /// reproducibility.
    ///
    /// For distributed loaders, use `Graph::epoch()` instead (which
    /// provides chunk_ratios from the auto-balancer).
    pub fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        match &mut self.inner {
            LoaderInner::Resident(loader) => loader.epoch(epoch),
            LoaderInner::Streaming(loader) => loader.epoch(epoch),
        }
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(l) => l.sampler.len(),
            LoaderInner::Streaming(l) => l.sampler.len(),
        }
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        let (n, bs, dl) = match &self.inner {
            LoaderInner::Resident(l) => (l.sampler.len(), l.batch_size, l.drop_last),
            LoaderInner::Streaming(l) => (l.sampler.len(), l.batch_size, l.drop_last),
        };
        if dl { n / bs } else { n.div_ceil(bs) }
    }

    /// Batch size.
    pub fn batch_size(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(l) => l.batch_size,
            LoaderInner::Streaming(l) => l.batch_size,
        }
    }

    /// Target device for the loader.
    pub fn device(&self) -> Device {
        match &self.inner {
            LoaderInner::Resident(l) => l.device,
            LoaderInner::Streaming(l) => l.device,
        }
    }

    /// Whether the loader is in resident mode (full dataset in memory on one device).
    pub fn is_resident(&self) -> bool {
        matches!(&self.inner, LoaderInner::Resident(_))
    }

    /// Tensor names for each batch position.
    pub fn names(&self) -> &[String] {
        match &self.inner {
            LoaderInner::Resident(l) => &l.names,
            LoaderInner::Streaming(l) => &l.names,
        }
    }

    /// Current prefetch depth (streaming mode). Returns 0 for resident loaders.
    pub fn prefetch_depth(&self) -> usize {
        match &self.inner {
            LoaderInner::Resident(_) => 0,
            LoaderInner::Streaming(l) => l.worker.prefetch_depth(),
        }
    }

    /// Set prefetch depth for streaming backends. Takes effect on the next epoch.
    ///
    /// Disables automatic resize (the loader won't override your setting).
    /// No-op for resident loaders (the entire dataset is already in VRAM).
    pub fn set_prefetch_depth(&mut self, depth: usize) {
        match &mut self.inner {
            LoaderInner::Resident(_) => {}
            LoaderInner::Streaming(l) => {
                l.worker.set_prefetch_depth(depth);
                l.user_set_depth = true;
            }
        }
    }

    /// Measure free VRAM and resize prefetch buffers to fill available space.
    ///
    /// **This happens automatically** at every epoch boundary (epoch 1+).
    /// The loader re-probes free VRAM each epoch and fills 90% of it.
    /// You only need to call this manually if you want to resize at a
    /// different point (e.g., mid-epoch during an AllReduce window).
    ///
    /// Calling this (or [`set_prefetch_depth`](DataLoader::set_prefetch_depth))
    /// disables automatic adaptation -- the loader assumes you're managing
    /// depth yourself.
    ///
    /// The data is static across epochs, so a deeper buffer means more of
    /// the dataset stays in VRAM and fewer H2D transfers are needed. If the
    /// buffer covers the entire epoch, performance converges to resident mode.
    ///
    /// Returns the new prefetch depth (0 for resident loaders).
    pub fn auto_resize(&mut self) -> usize {
        match &mut self.inner {
            LoaderInner::Resident(_) => 0,
            LoaderInner::Streaming(l) => {
                let depth = prefetch_depth_from_vram(l.per_sample_bytes, l.batch_size, l.device, l.vram_max_usage, 0);
                l.worker.set_prefetch_depth(depth);
                l.user_set_depth = true;
                depth
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ResidentLoader
// ---------------------------------------------------------------------------

pub(crate) struct ResidentLoader {
    /// Full dataset tensors on target device, one per position.
    gpu_data: Vec<Tensor>,
    /// Original dataset (kept for upgrade_distributed).
    _dataset: Arc<dyn BatchDataSet>,
    device: Device,
    batch_size: usize,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    names: Vec<String>,
}

impl ResidentLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        let indices = self.sampler.indices(epoch);
        let n = indices.len();
        let bs = self.batch_size;

        // Compute batch boundaries
        let mut batch_ranges = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + bs).min(n);
            if self.drop_last && (end - start) < bs {
                break;
            }
            batch_ranges.push((start, end - start));
            start = end;
        }

        // Build index tensor on the target device (i64 for index_select)
        let i64_indices: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        let perm = Tensor::from_i64(
            &i64_indices,
            &[i64_indices.len() as i64],
            self.device,
        )
        .expect("failed to create permutation tensor");

        EpochIterator {
            inner: EpochIteratorInner::Resident(ResidentEpochIter {
                data: &self.gpu_data,
                perm,
                batch_ranges,
                pos: 0,
                names: &self.names,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingLoader
// ---------------------------------------------------------------------------

pub(crate) struct StreamingLoader {
    /// Dataset shared with the worker thread.
    _dataset: Arc<dyn BatchDataSet>,
    batch_size: usize,
    device: Device,
    sampler: Box<dyn Sampler>,
    drop_last: bool,
    worker: PrefetchWorker,
    names: Vec<String>,
    /// Per-sample bytes (for adaptive resize depth calculation).
    per_sample_bytes: usize,
    /// Maximum fraction of total VRAM to use for prefetch.
    vram_max_usage: f64,
    /// True when the user explicitly set depth (`.prefetch()` or `set_prefetch_depth()`).
    /// Skips automatic adaptation so we don't override the user's choice.
    user_set_depth: bool,
}

impl StreamingLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        // Probe VRAM usage and size the prefetch buffer to fill up to cap.
        // At epoch 0 this is the real signal: model is loaded, VRAM is known.
        // At epoch N>0: re-probe in case conditions changed.
        if !self.user_set_depth {
            let depth = prefetch_depth_from_vram(
                self.per_sample_bytes, self.batch_size, self.device, self.vram_max_usage, 0,
            );
            self.worker.set_prefetch_depth(depth);
        }

        let indices = self.sampler.indices(epoch);
        let n = indices.len();
        let bs = self.batch_size;

        // Count batches
        let num_batches = if self.drop_last {
            n / bs
        } else {
            n.div_ceil(bs)
        };

        // Start the epoch: gets a fresh per-epoch batch channel.
        // If the previous epoch was dropped mid-way, the old channel is already
        // closed (old batch_tx dropped by the worker when send fails or epoch ends).
        let batch_rx = self.worker.start_epoch(indices, bs, self.drop_last);

        EpochIterator {
            inner: EpochIteratorInner::Streaming(StreamingEpochIter {
                batch_rx,
                remaining: num_batches,
                names: &self.names,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// EpochIterator
// ---------------------------------------------------------------------------

/// Iterator over batches for one training epoch.
///
/// Created by [`DataLoader::epoch`]. Each element is a
/// `Result<`[`Batch`]`>` containing tensors already on the target device.
///
/// Dropping the iterator mid-epoch is safe and cancels any outstanding
/// prefetch work (in streaming mode).
pub struct EpochIterator<'a> {
    inner: EpochIteratorInner<'a>,
}

enum EpochIteratorInner<'a> {
    Resident(ResidentEpochIter<'a>),
    Streaming(StreamingEpochIter<'a>),
}

struct ResidentEpochIter<'a> {
    data: &'a [Tensor],
    perm: Tensor,
    /// (start_in_perm, batch_len)
    batch_ranges: Vec<(usize, usize)>,
    pos: usize,
    names: &'a [String],
}

struct StreamingEpochIter<'a> {
    batch_rx: std::sync::mpsc::Receiver<Result<super::prefetch::PrefetchedBatch>>,
    remaining: usize,
    names: &'a [String],
}

impl<'a> Iterator for EpochIterator<'a> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            EpochIteratorInner::Resident(iter) => iter.next(),
            EpochIteratorInner::Streaming(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.inner {
            EpochIteratorInner::Resident(iter) => {
                let remaining = iter.batch_ranges.len() - iter.pos;
                (remaining, Some(remaining))
            }
            EpochIteratorInner::Streaming(iter) => {
                (iter.remaining, Some(iter.remaining))
            }
        }
    }
}

impl ExactSizeIterator for EpochIterator<'_> {}

impl<'a> ResidentEpochIter<'a> {
    fn next(&mut self) -> Option<Result<Batch>> {
        if self.pos >= self.batch_ranges.len() {
            return None;
        }
        let (start, len) = self.batch_ranges[self.pos];
        self.pos += 1;

        // Slice the permutation tensor for this batch
        let batch_perm = match self.perm.narrow(0, start as i64, len as i64) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };

        // index_select each tensor position
        let mut tensors = Vec::with_capacity(self.data.len());
        for t in self.data {
            match t.index_select(0, &batch_perm) {
                Ok(selected) => tensors.push(selected),
                Err(e) => return Some(Err(e)),
            }
        }

        Some(Ok(Batch::new(tensors, self.names.to_vec())))
    }
}

impl StreamingEpochIter<'_> {
    fn next(&mut self) -> Option<Result<Batch>> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        // Receive the next ready batch from the worker
        match self.batch_rx.recv() {
            Ok(Ok(batch)) => {
                // Wait for async H2D copy to complete (typically instant
                // since the batch was submitted prefetch_depth steps ago)
                #[cfg(feature = "cuda")]
                if let Some(ref event) = batch.ready_event {
                    if let Err(e) = event.synchronize() {
                        return Some(Err(e));
                    }
                }
                Some(Ok(Batch::new(batch.tensors, self.names.to_vec())))
            }
            Ok(Err(e)) => Some(Err(e)),
            Err(_) => {
                // Channel closed (worker stopped or panicked)
                self.remaining = 0;
                Some(Err(TensorError::new(
                    "DataLoader: prefetch worker stopped unexpectedly",
                )))
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------


#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
