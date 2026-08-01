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
use super::sample_cache::SampleCache;
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
        // The probe returns (used, total) — used first, not free.
        Ok((used, total)) => {
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

/// Where the streaming loader's activation reserve came from. Decides
/// how much of the computed budget the FIRST fill may take: before the
/// first training step, the VRAM probe cannot see activations,
/// gradients, or lazily created optimizer state, so the fill is
/// discounted by how much of that gap the reserve already covers.
/// After the first step the probe is honest (the allocator retains
/// those blocks) and the full budget applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReserveSource {
    /// No information: reserve is 0, first fill takes 1/3 of budget.
    Bare,
    /// Framework-derived (3x parameter bytes via `Graph::set_data_loader`):
    /// gradients (~1x) and lazy optimizer state (~2x, Adam-family) are
    /// covered; activations are not. First fill takes 1/2.
    Auto,
    /// User-declared via `activation_reserve()`: full trust, no haircut.
    User,
}

/// First-fill target from the computed full-budget depth: the graduated
/// haircut for the sizing done before any training step has run.
pub(crate) fn initial_fill_target(full_depth: usize, source: ReserveSource) -> usize {
    let divisor = match source {
        ReserveSource::Bare => 3,
        ReserveSource::Auto => 2,
        ReserveSource::User => 1,
    };
    (full_depth / divisor).max(1)
}

// Budget policy (ring sizing, cache budget, prefetch depth) lives in
// `data::budget` — one law shared with the DDP worker/stager paths.
// Re-exported here for this module's call sites and its test file.
pub(crate) use super::budget::{
    prefetch_depth_from_vram, ring_slots_from_ram, sample_cache_budget, RING_SLOTS_WITH_CACHE,
};
#[cfg(test)]
pub(crate) use super::budget::RING_SLOTS_FALLBACK;

// ---------------------------------------------------------------------------
// DataLoaderBuilder
// ---------------------------------------------------------------------------

/// Pick-space context shared by both loader modes: augmentation
/// multiplicity, the shuffle seed (which keys the transform), and the
/// delivery transform itself. At `augment = 1` with no transform this
/// is inert — picks and sample ids coincide.
#[derive(Clone)]
pub(crate) struct PickCtx {
    pub(crate) augment: usize,
    pub(crate) seed: u64,
    pub(crate) transform: Option<crate::data::TransformFn>,
}

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
    ram_max_usage: f64,
    activation_reserve: Option<usize>,
    /// Read-through sample cache created by `from_dataset` (None for
    /// opaque `BatchDataSet` loaders). The adapter inside `dataset`
    /// holds the other Arc.
    pub(crate) sample_cache: Option<Arc<SampleCache>>,
    sample_cache_enabled: bool,
    disk_stage_bytes: u64,
    disk_stage_dir: Option<std::path::PathBuf>,
    vram_pool_enabled: bool,
    no_shuffle: bool,
    augment: usize,
    transform: Option<crate::data::TransformFn>,
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
            ram_max_usage: 0.50,
            activation_reserve: None,
            sample_cache: None,
            sample_cache_enabled: true,
            disk_stage_bytes: 0,
            disk_stage_dir: None,
            vram_pool_enabled: super::vram_pool::VRAM_POOL_DEFAULT,
            no_shuffle: false,
            augment: 1,
            transform: None,
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
        self.no_shuffle = !shuffle;
        self
    }

    /// Custom sampler. Overrides the [`shuffle`](DataLoaderBuilder::shuffle) setting.
    pub fn sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Augmentation multiplicity: each sample appears `k` times per
    /// epoch, spread by the shuffle. Default: 1.
    ///
    /// Pure scheduling — every one of the `k` picks fetches the same
    /// raw bytes (staged once across the tiers); data variation comes
    /// exclusively from the [`transform`](DataLoaderBuilder::transform),
    /// keyed per pick. Without a transform, `k > 1` is plain
    /// oversampling (`k` identical views per epoch). An epoch is one
    /// pass over the `len() * k` picks, so batch counts scale by `k`.
    ///
    /// Composes with the built-in samplers only; combining with
    /// [`sampler`](DataLoaderBuilder::sampler) is a build error.
    pub fn augment(mut self, k: usize) -> Self {
        self.augment = k.max(1);
        self
    }

    /// Deterministic per-batch transform applied at delivery — the
    /// sanctioned home for augmentation (see [`PickKey`]).
    ///
    /// Receives the delivered rows (raw bytes, already on the target
    /// device, freshly assembled — never aliasing the staging tiers)
    /// and one [`PickKey`] per row. Must be a pure function of
    /// `(rows, keys)` and preserve the row count; derive per-view
    /// randomness from [`PickKey::rng`]. Runs live on every delivery:
    /// the tiers retain raw samples only, and each pick re-derives its
    /// view — for a VRAM-pooled sample that is one upload and `k`
    /// on-device realizations.
    ///
    /// [`PickKey`]: crate::data::PickKey
    /// [`PickKey::rng`]: crate::data::PickKey::rng
    pub fn transform(
        mut self,
        f: impl Fn(Vec<Tensor>, &[crate::data::PickKey]) -> Result<Vec<Tensor>>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.transform = Some(crate::data::TransformFn::new(f));
        self
    }

    /// Override auto-detected prefetch depth (streaming mode only).
    ///
    /// Auto-detection fills `(1 - margin)` of free VRAM at build time.
    /// Use this to set a specific depth instead. Disables automatic
    /// per-epoch adaptation.
    ///
    /// The streaming loader always runs a background prefetch thread; a
    /// depth of `0` is clamped to `1`, not switched to synchronous loading.
    /// Synchronous, single-threaded loading is the *resident* path, chosen
    /// automatically when the dataset fits in VRAM (it is not selectable
    /// here).
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

    /// Maximum fraction of **available** host RAM the reader stage may
    /// claim while buffering batches ahead (streaming mode, CUDA
    /// targets).
    ///
    /// Default: 0.50. The streaming pipeline runs two stages on CUDA
    /// targets: a reader thread fetches batches from the dataset into
    /// a pageable-RAM ring while the transfer thread pins and copies
    /// to the device, so storage-read latency (network shares, slow
    /// disks) overlaps transfer work instead of adding to it. The ring
    /// is sized at each `epoch()` as this fraction of `MemAvailable`,
    /// which already excludes every other process on the box, permanent
    /// fixtures included (pinned VM memory, hugepages), so the budget
    /// tracks what is actually free and self-adjusts as the box fills
    /// or drains.
    ///
    /// The ceiling is **per loader**: each loader sizes its ring
    /// independently, so when several rank processes with CUDA-target
    /// loaders share one host, give each a divided fraction (e.g.
    /// `0.50 / local_ranks`) — they cannot see each other's rings at
    /// sizing time.
    ///
    /// `0.0` disables the reader stage (single-stage pipeline).
    /// Clamped to `[0.0, 0.90]`.
    pub fn ram_max_usage(mut self, max_usage: f64) -> Self {
        self.ram_max_usage = max_usage.clamp(0.0, 0.90);
        self
    }

    /// Enable / disable the read-through sample cache (streaming mode,
    /// [`DataSet`]-backed loaders). Default: enabled.
    ///
    /// The cache retains samples in RAM as epoch 1 reads them, so later
    /// epochs read at RAM speed instead of storage speed. It is keyed
    /// by sample identity, which makes staged content reshuffle-proof:
    /// a reshuffle changes only the order, never the content set. It
    /// shares the [`ram_max_usage`](Self::ram_max_usage) budget with
    /// the reader ring (the ring keeps a small flow-buffer slice, the
    /// cache gets the rest) and never evicts — with every epoch
    /// touching each sample exactly once in fresh random order, no
    /// eviction policy beats filling until the budget is reached and
    /// keeping what is there.
    ///
    /// Disable for single-pass training over a dataset far larger than
    /// RAM, where retained samples are never revisited and the whole
    /// budget is better spent on the reader ring. Opaque
    /// [`BatchDataSet`] loaders have no sample layer and are never
    /// cached; `ram_max_usage(0.0)` also stops all admissions.
    pub fn sample_cache(mut self, enabled: bool) -> Self {
        self.sample_cache_enabled = enabled;
        self
    }

    /// Enable / disable the device-resident sample pool (streaming
    /// mode, CUDA targets). Default: enabled.
    ///
    /// The pool retains as many samples as fit in leftover VRAM
    /// (measured after the first training step, under the same cap as
    /// prefetch) and assembles batches by gathering retained rows on
    /// device instead of uploading them — H2D traffic shrinks by the
    /// hit rate, the middle ground between resident and streaming
    /// modes. Samples enter by device-side capture from batches that
    /// were uploaded anyway, so filling costs no extra transfers.
    /// Sizing is automatic; there is no fraction knob.
    ///
    /// Disable for single-pass training (retained rows are never
    /// revisited) or to keep every spare byte of VRAM for something
    /// else.
    pub fn vram_pool(mut self, enabled: bool) -> Self {
        self.vram_pool_enabled = enabled;
        self
    }

    /// Local-disk overflow tier below the sample cache, sized in GB
    /// (`0` = off, the default).
    ///
    /// Samples the RAM cache declines are staged once in an
    /// append-only pack file on a local drive; later epochs read them
    /// at disk speed instead of source speed. Pays exactly when the
    /// source is slower than local disk (network mounts) and data is
    /// revisited — for a dataset already on local SSD it buys nothing,
    /// the source IS the disk. The pack file is ephemeral (removed
    /// when the loader drops) and lives in the system temp directory
    /// unless [`disk_stage_dir`](Self::disk_stage_dir) says otherwise;
    /// a RAM-backed temp dir (tmpfs) triggers a loud warning, since a
    /// stage that spends RAM defeats its purpose.
    ///
    /// Requires the sample layer: `build()` errors loudly on an opaque
    /// [`BatchDataSet`] loader or with `sample_cache(false)`. Ignored
    /// in resident mode (the dataset already lives on-device).
    pub fn disk_stage(mut self, gb: u64) -> Self {
        self.disk_stage_bytes = gb.saturating_mul(1 << 30);
        self
    }

    /// Directory for the disk-stage pack file (default: the system
    /// temp directory). Point this at a real local drive — not a
    /// network mount (that would re-buy the latency the stage exists
    /// to avoid) and not tmpfs (that spends RAM).
    pub fn disk_stage_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.disk_stage_dir = Some(dir.into());
        self
    }

    /// Bytes to reserve for forward/backward memory in streaming-mode
    /// VRAM sizing (activations, gradients, lazily created optimizer
    /// state).
    ///
    /// The adaptive prefetch sizes its buffer from a VRAM probe, and
    /// before the first training step that probe cannot see step
    /// memory: it does not exist yet. Setting an explicit reserve
    /// deducts it from the prefetch budget and disables the
    /// conservative first-fill discount entirely (full trust). When
    /// unset, the loader falls back to a graduated first fill: 1/2 of
    /// the budget when the framework derived a reserve from model size
    /// ([`crate::graph::Graph::set_data_loader`] wires 3x parameter
    /// bytes to cover gradients + optimizer state), 1/3 with no
    /// information at all. From the second consumed batch on, the
    /// probe is honest and the full budget applies either way.
    pub fn activation_reserve(mut self, bytes: usize) -> Self {
        self.activation_reserve = Some(bytes);
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
            ram_max_usage,
            activation_reserve,
            sample_cache,
            sample_cache_enabled,
            disk_stage_bytes,
            disk_stage_dir,
            vram_pool_enabled,
            no_shuffle,
            augment,
            transform,
        } = self;

        // `FLODL_VRAM_POOL=off` runtime kill-switch — same parse as
        // the DDP rank workers (audit D7: it used to be honored only
        // there, so a scripted A/B silently no-op'ed on the solo path).
        let vram_pool_enabled =
            vram_pool_enabled && !super::vram_pool::vram_pool_env_off();

        // Augmentation is pick-space scheduling over the built-in
        // samplers; a custom sampler owns its own index stream, so the
        // combination has no defined meaning — error loudly.
        if augment > 1 && sampler.is_some() {
            return Err(TensorError::new(
                "DataLoader: augment(k) composes with the built-in samplers only \
                 (the schedule becomes a shuffle of len()*k picks). A custom \
                 sampler owns its index stream; emit repeated indices from it \
                 directly if you need multiplicity, or drop the custom sampler.",
            ));
        }

        // The off switch drops the loader-side handle; the adapter's
        // clone stays dormant (budget 0 = pure pass-through).
        let sample_cache = if sample_cache_enabled {
            sample_cache
        } else {
            None
        };

        // disk_stage is an explicit knob: with no sample layer to
        // stage, error loudly instead of silently doing nothing.
        if disk_stage_bytes > 0 && sample_cache.is_none() {
            return Err(TensorError::new(
                "DataLoader: disk_stage requires the sample layer — a DataSet-backed \
                 loader with the sample cache enabled. Opaque BatchDataSet loaders \
                 have no per-sample access to stage; sample_cache(false) disables \
                 the tier the stage overflows from.",
            ));
        }

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
        let shuffle = sampler.is_none() && !no_shuffle;
        // The schedule runs over PICKS: n samples × augment views,
        // shuffled as one space so a sample's k views spread across
        // the epoch instead of clustering.
        let picks = n * augment;
        let pick_ctx = PickCtx {
            augment,
            seed,
            transform,
        };

        let sampler = sampler.unwrap_or_else(|| -> Box<dyn Sampler> {
            if no_shuffle {
                Box::new(SequentialSampler::new(picks))
            } else {
                Box::new(RandomSampler::new(picks, seed))
            }
        });

        let user_set_depth = prefetch_depth.is_some();
        // Bootstrap depth: small buffer to start. The real depth is
        // computed at epoch() time when free VRAM reflects the actual
        // model allocation. User override skips adaptive sizing.
        let streaming_depth = prefetch_depth.unwrap_or(BOOTSTRAP_PREFETCH);
        if use_resident {
            match build_resident(Arc::clone(&dataset), batch_size, device, sampler, drop_last, names.clone(), pick_ctx.clone()) {
                Ok(loader) => Ok(loader),
                Err(e) if device.is_cuda() && e.is_cuda_oom() => {
                    // VRAM estimate was wrong, fall back to streaming.
                    // Recreate sampler since build_resident consumed it.
                    let sampler: Box<dyn Sampler> = if shuffle {
                        Box::new(RandomSampler::new(picks, seed))
                    } else {
                        Box::new(SequentialSampler::new(picks))
                    };
                    crate::tensor::cuda_empty_cache();
                    build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, per_sample_bytes, vram_max_usage, ram_max_usage, user_set_depth, activation_reserve, sample_cache, disk_stage_bytes, &disk_stage_dir, vram_pool_enabled, names, pick_ctx)
                }
                Err(e) => Err(e),
            }
        } else {
            build_streaming(dataset, batch_size, device, sampler, drop_last, streaming_depth, per_sample_bytes, vram_max_usage, ram_max_usage, user_set_depth, activation_reserve, sample_cache, disk_stage_bytes, &disk_stage_dir, vram_pool_enabled, names, pick_ctx)
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
    pick_ctx: PickCtx,
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
            pick_ctx,
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
    ram_max_usage: f64,
    user_set_depth: bool,
    activation_reserve: Option<usize>,
    sample_cache: Option<Arc<SampleCache>>,
    disk_stage_bytes: u64,
    disk_stage_dir: &Option<std::path::PathBuf>,
    vram_pool_enabled: bool,
    names: Vec<String>,
    pick_ctx: PickCtx,
) -> Result<DataLoader> {
    // Local-disk overflow tier under the sample cache. Attached here,
    // not in build(): the resident path never reads through the cache,
    // so it must not create a pack file it would never use.
    if disk_stage_bytes > 0 {
        if let Some(cache) = &sample_cache {
            let dir = disk_stage_dir
                .clone()
                .unwrap_or_else(std::env::temp_dir);
            cache.attach_disk(super::sample_cache::DiskStage::create(
                &dir,
                disk_stage_bytes,
                dataset.len(),
            )?);
        }
    }

    let worker = PrefetchWorker::new(
        Arc::clone(&dataset),
        device,
        prefetch_depth,
        vram_pool_enabled,
        pick_ctx.augment,
    );
    let (reserve, reserve_source) = match activation_reserve {
        Some(bytes) => (bytes, ReserveSource::User),
        None => (0, ReserveSource::Bare),
    };

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
            ram_max_usage,
            sample_cache,
            user_set_depth,
            activation_reserve: reserve,
            reserve_source,
            governor: Arc::new(super::prefetch::GovernorCtl::new(prefetch_depth)),
            pick_ctx,
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
    /// Items are automatically stacked into batches. Sample fetches go
    /// through a read-through RAM cache in streaming mode (see
    /// [`DataLoaderBuilder::sample_cache`]).
    pub fn from_dataset<D: DataSet + 'static>(dataset: D) -> DataLoaderBuilder {
        let cache = Arc::new(SampleCache::new(dataset.len()));
        let mut builder = DataLoaderBuilder::new(Box::new(DataSetAdapter::with_cache(
            dataset,
            Arc::clone(&cache),
        )));
        builder.sample_cache = Some(cache);
        builder
    }

    /// Create a DataLoader from a per-batch [`BatchDataSet`].
    ///
    /// The dataset is responsible for returning properly batched
    /// tensors. Batches are opaque to the loader, so the sample cache
    /// does not apply (batching is the dataset's own affair).
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
                l.worker.set_prefetch_depth(depth.max(1));
                // Apply immediately: the governor target is the live
                // in-flight bound (the channel capacity, set above, only
                // takes effect at the next epoch and acts as a ceiling).
                l.governor
                    .target
                    .store(depth.max(1), std::sync::atomic::Ordering::Relaxed);
                l.user_set_depth = true;
            }
        }
    }

    /// Bytes reserved for forward/backward memory in streaming-mode
    /// VRAM sizing. Same contract as
    /// [`DataLoaderBuilder::activation_reserve`]: an explicit value is
    /// fully trusted (no first-fill discount). No-op for resident
    /// loaders.
    pub fn set_activation_reserve(&mut self, bytes: usize) {
        if let LoaderInner::Streaming(l) = &mut self.inner {
            l.activation_reserve = bytes;
            l.reserve_source = ReserveSource::User;
        }
    }

    /// Framework-derived reserve (`Graph::set_data_loader` wires 3x
    /// parameter bytes: gradients ~1x + lazy optimizer state ~2x).
    /// Never overrides a user-declared reserve; keeps the halved
    /// first-fill discount since activations remain unaccounted for.
    pub(crate) fn set_activation_reserve_auto(&mut self, bytes: usize) {
        if let LoaderInner::Streaming(l) = &mut self.inner {
            if l.reserve_source == ReserveSource::Bare {
                l.activation_reserve = bytes;
                l.reserve_source = ReserveSource::Auto;
            }
        }
    }

    /// Measure free VRAM and resize the prefetch in-flight target to
    /// fill available space.
    ///
    /// **This happens automatically**: at every `epoch()` call, and
    /// once mid-run after the first training step (when the probe
    /// first sees activations/gradients/optimizer state). You only
    /// need this manually to resize at a different point (e.g.,
    /// mid-epoch during an AllReduce window) -- the target applies
    /// immediately, capped for the current epoch by the channel
    /// capacity chosen at its start.
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
                use std::sync::atomic::Ordering;
                // Deduct the reserve only while the probe is still
                // blind to step memory; afterwards it would double-count.
                let reserve = if l.governor.honest_resize_done.load(Ordering::Relaxed) {
                    0
                } else {
                    l.activation_reserve
                };
                let depth = prefetch_depth_from_vram(
                    l.per_sample_bytes, l.batch_size, l.device, l.vram_max_usage, reserve,
                );
                let depth = depth.max(1);
                l.worker.set_prefetch_depth(depth);
                l.governor.target.store(depth, Ordering::Relaxed);
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
    pick_ctx: PickCtx,
}

impl ResidentLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        // The sampler yields PICKS; the resident data is stored by
        // sample id, so the gather runs on the decoded ids (the same
        // sample id may appear several times in one batch — that is
        // augmentation, and index_select handles repeats natively).
        let picks = self.sampler.indices(epoch);
        let n = picks.len();
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
        let k = self.pick_ctx.augment.max(1) as i64;
        let i64_indices: Vec<i64> = picks.iter().map(|&i| i as i64 / k).collect();
        let perm = match Tensor::from_i64(
            &i64_indices,
            &[i64_indices.len() as i64],
            self.device,
        ) {
            Ok(t) => t,
            Err(e) => {
                return EpochIterator {
                    inner: EpochIteratorInner::Failed(Some(TensorError::new(&format!(
                        "resident loader: failed to upload the epoch permutation: {e}"
                    )))),
                }
            }
        };

        EpochIterator {
            inner: EpochIteratorInner::Resident(ResidentEpochIter {
                data: &self.gpu_data,
                perm,
                batch_ranges,
                pos: 0,
                names: &self.names,
                picks,
                pick_ctx: &self.pick_ctx,
                epoch,
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
    /// Host RAM ceiling for the reader-stage ring (see
    /// [`DataLoaderBuilder::ram_max_usage`]). `0.0` = single-stage.
    ram_max_usage: f64,
    /// Read-through sample cache shared with the adapter inside the
    /// worker's dataset. `None` = opaque `BatchDataSet` loader or
    /// `.sample_cache(false)`. Budgeted at each `epoch()`.
    sample_cache: Option<Arc<SampleCache>>,
    /// True when the user explicitly set depth (`.prefetch()` or `set_prefetch_depth()`).
    /// Skips automatic adaptation so we don't override the user's choice.
    user_set_depth: bool,
    /// Bytes deducted from the prefetch budget before the first
    /// training step has run (see [`ReserveSource`]).
    activation_reserve: usize,
    /// Where the reserve came from; picks the first-fill discount.
    reserve_source: ReserveSource,
    /// Depth governor shared with the worker and the live epoch
    /// iterator. The worker keeps at most `target` batches in flight;
    /// the target is adjustable at any moment (epoch sizing, one-shot
    /// honest resize, `auto_resize`, worker OOM halving).
    governor: Arc<super::prefetch::GovernorCtl>,
    pick_ctx: PickCtx,
}

impl StreamingLoader {
    fn epoch(&mut self, epoch: usize) -> EpochIterator<'_> {
        use std::sync::atomic::Ordering;

        // Size the epoch. Two regimes, keyed to information rather than
        // epoch count:
        // - Before the first training step of the RUN, the VRAM probe
        //   cannot see activations/gradients/lazy optimizer state, so
        //   the initial target deducts the activation reserve and takes
        //   the graduated first-fill discount. Once the consumer drains
        //   its second batch (first step demonstrably done), the epoch
        //   iterator re-probes and raises the target to the honest
        //   budget mid-epoch.
        // - After that, probes are honest (the allocator retains step
        //   memory as "used"), so epoch starts take the full budget
        //   with no reserve.
        if !self.user_set_depth {
            // Reserve-free depth: the capacity ceiling for this epoch.
            // Any later raise (honest resize, auto_resize) stays <= it,
            // because "used" only grows once training runs.
            let full = prefetch_depth_from_vram(
                self.per_sample_bytes, self.batch_size, self.device, self.vram_max_usage, 0,
            );
            let target = if self.governor.honest_resize_done.load(Ordering::Relaxed) {
                full.max(1)
            } else {
                let reserved = prefetch_depth_from_vram(
                    self.per_sample_bytes,
                    self.batch_size,
                    self.device,
                    self.vram_max_usage,
                    self.activation_reserve,
                );
                initial_fill_target(reserved, self.reserve_source)
            };
            self.worker.set_prefetch_depth(full.max(2));
            self.governor.begin_epoch(target);
        } else {
            self.governor.begin_epoch(self.worker.prefetch_depth());
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

        // One RAM probe per epoch serves both RAM consumers below.
        let mem = crate::sys::mem_info().map(|m| m.available_bytes);

        // Reader-ring sizing: CUDA targets only. On CPU targets the
        // batch channel itself is the read-ahead buffer (no transfer
        // stage to overlap), so the pipeline stays single-stage. The
        // mechanism in the worker is device-agnostic; this is policy.
        // While the sample cache is active the ring is capped to a
        // flow-buffer depth: jitter absorption saturates fast, retained
        // samples pay again every later epoch.
        let ring_slots = if self.device.is_cuda() {
            let sized = ring_slots_from_ram(
                self.per_sample_bytes,
                bs,
                self.ram_max_usage,
                mem,
                num_batches,
            );
            if self.sample_cache.is_some() {
                sized.min(RING_SLOTS_WITH_CACHE)
            } else {
                sized
            }
        } else {
            0
        };

        // Sample-cache budget refresh: same available-RAM share as the
        // ring, recomputed from the live probe once per epoch (see
        // `sample_cache_budget` for why held bytes are added back to
        // the probe before taking the share). A shrinking budget stops
        // new admissions, never drops staged content. Without RAM
        // visibility the budget stays as it was (initially 0: no
        // admissions on hosts we cannot measure).
        if let Some(cache) = &self.sample_cache {
            if let Some(available) = mem {
                let ring_bytes = (ring_slots as u64)
                    .saturating_mul(self.per_sample_bytes.saturating_mul(bs) as u64);
                let budget = sample_cache_budget(
                    available,
                    cache.bytes() as u64,
                    ring_bytes,
                    self.ram_max_usage,
                );
                cache.set_budget(usize::try_from(budget).unwrap_or(usize::MAX));
            }
        }

        // Start the epoch: gets a fresh per-epoch batch channel.
        // If the previous epoch was dropped mid-way, the old channel is already
        // closed (old batch_tx dropped by the worker when send fails or epoch ends).
        let batch_rx = self.worker.start_epoch(
            indices,
            bs,
            self.drop_last,
            Arc::clone(&self.governor),
            ring_slots,
        );

        EpochIterator {
            inner: EpochIteratorInner::Streaming(StreamingEpochIter {
                batch_rx,
                remaining: num_batches,
                names: &self.names,
                governor: Arc::clone(&self.governor),
                adaptive: !self.user_set_depth,
                per_sample_bytes: self.per_sample_bytes,
                batch_size: self.batch_size,
                device: self.device,
                vram_max_usage: self.vram_max_usage,
                pick_ctx: &self.pick_ctx,
                epoch,
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
    /// Epoch setup failed before the first batch (e.g. the resident path's
    /// permutation-tensor upload). `epoch()` cannot return `Result` without
    /// breaking every training loop, but `Item = Result<Batch>` already is
    /// the error channel — the failure is delivered as the first item.
    Failed(Option<TensorError>),
}

struct ResidentEpochIter<'a> {
    data: &'a [Tensor],
    perm: Tensor,
    /// (start_in_perm, batch_len)
    batch_ranges: Vec<(usize, usize)>,
    pos: usize,
    names: &'a [String],
    /// The epoch's pick stream (perm holds the decoded sample ids;
    /// picks key the transform).
    picks: Vec<usize>,
    pick_ctx: &'a PickCtx,
    epoch: usize,
}

struct StreamingEpochIter<'a> {
    batch_rx: std::sync::mpsc::Receiver<Result<super::prefetch::PrefetchedBatch>>,
    remaining: usize,
    names: &'a [String],
    /// Shared with the loader and worker: this side bumps `consumed`
    /// per drained batch and performs the one-shot honest resize.
    governor: Arc<super::prefetch::GovernorCtl>,
    /// False when the user pinned the depth (no honest resize).
    adaptive: bool,
    // Sizing snapshot for the honest resize probe.
    per_sample_bytes: usize,
    batch_size: usize,
    device: Device,
    vram_max_usage: f64,
    pick_ctx: &'a PickCtx,
    epoch: usize,
}

impl Drop for StreamingEpochIter<'_> {
    fn drop(&mut self) {
        // Unblock the worker's governor gate: `consumed` stops
        // advancing once this iterator is gone, so an abandoned
        // mid-epoch iterator would otherwise leave the worker waiting
        // at the gate forever instead of reaching the failed send that
        // ends the epoch.
        self.governor
            .abandoned
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl<'a> Iterator for EpochIterator<'a> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            EpochIteratorInner::Resident(iter) => iter.next(),
            EpochIteratorInner::Streaming(iter) => iter.next(),
            EpochIteratorInner::Failed(err) => err.take().map(Err),
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
            EpochIteratorInner::Failed(err) => {
                let n = usize::from(err.is_some());
                (n, Some(n))
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

        // Delivery transform: keyed per pick, on the freshly gathered
        // rows (index_select allocates — resident data is never
        // aliased into the batch).
        if let Some(ref f) = self.pick_ctx.transform {
            let batch_picks = &self.picks[start..start + len];
            tensors = match crate::data::apply_transform(
                f,
                tensors,
                batch_picks,
                self.pick_ctx.augment,
                self.epoch,
                self.pick_ctx.seed,
            ) {
                Ok(t) => t,
                Err(e) => return Some(Err(e)),
            };
        }

        Some(Ok(Batch::new(tensors, self.names.to_vec())))
    }
}

impl StreamingEpochIter<'_> {
    fn next(&mut self) -> Option<Result<Batch>> {
        use std::sync::atomic::Ordering;

        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        // Receive the next ready batch from the worker
        match self.batch_rx.recv() {
            Ok(Ok(batch)) => {
                // Wait for async H2D copy to complete (typically instant
                // since the batch was submitted prefetch_depth steps ago)
                #[cfg(feature = "gpu")]
                if let Some(ref event) = batch.ready_event {
                    if let Err(e) = event.synchronize() {
                        return Some(Err(e));
                    }
                    // Cross-stream lifetime pin (same hazard as the DDP
                    // worker's delivery, see epoch_plan): the blocks were
                    // allocated on the prefetch copy stream, and the
                    // consumer drops the batch while its own stream's
                    // kernels (backward reads the labels) may still be in
                    // flight — freed, the blocks guard only against the
                    // copy stream and the next upload can overwrite them
                    // mid-read.
                    match crate::tensor::cuda_stream::CudaStream::current(self.device) {
                        Ok(cur) => {
                            for t in &batch.tensors {
                                if let Err(e) = t.record_stream(&cur) {
                                    return Some(Err(e));
                                }
                            }
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                self.governor.consumed.fetch_add(1, Ordering::Relaxed);
                let run_consumed =
                    self.governor.run_consumed.fetch_add(1, Ordering::Relaxed) + 1;
                // Honest-probe latch, once per RUN: draining the second
                // batch means the first batch's forward/backward/step
                // have executed, so a probe now sees activations,
                // gradients, and lazily created optimizer state as
                // "used". Keyed to consumption, not epoch boundaries, so
                // single-pass training benefits too. The latch must be
                // set regardless of who owns the depth — it marks probe
                // honesty, and the VRAM sample pool's one-shot budget
                // decision (`maybe_install`) gates on it; an explicit
                // `.prefetch(N)` must pin the in-flight depth, not
                // silently disable the pool tier.
                if run_consumed >= 2
                    && !self.governor.honest_resize_done.load(Ordering::Relaxed)
                {
                    self.governor.honest_resize_done.store(true, Ordering::Relaxed);
                    // Honest resize of the in-flight target: adaptive
                    // mode only — a user-set depth stays exactly where
                    // the user put it. Full budget, no reserve (the
                    // probe accounts for step memory itself).
                    if self.adaptive {
                        let depth = prefetch_depth_from_vram(
                            self.per_sample_bytes,
                            self.batch_size,
                            self.device,
                            self.vram_max_usage,
                            0,
                        );
                        self.governor.target.store(depth.max(1), Ordering::Relaxed);
                    }
                }
                // Delivery transform: after the copy event, so the ops
                // are ordered against the async H2D; keyed by the
                // batch's picks, on freshly assembled rows.
                let tensors = if let Some(ref f) = self.pick_ctx.transform {
                    match crate::data::apply_transform(
                        f,
                        batch.tensors,
                        &batch.picks,
                        self.pick_ctx.augment,
                        self.epoch,
                        self.pick_ctx.seed,
                    ) {
                        Ok(t) => t,
                        Err(e) => return Some(Err(e)),
                    }
                } else {
                    batch.tensors
                };
                Some(Ok(Batch::new(tensors, self.names.to_vec())))
            }
            Ok(Err(e)) => Some(Err(e)),
            Err(_) => {
                // Channel closed. Dataset errors AND dataset panics are
                // reported per-batch (see `guarded_get_batch`), so the
                // worker dying is flodl's own fault — say so.
                self.remaining = 0;
                Some(Err(TensorError::new(
                    "DataLoader: prefetch worker stopped unexpectedly \
                     (dataset errors are reported per-batch, so this is \
                     likely a flodl bug — please report it)",
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
