//! Sampling strategies for dataset index ordering.
//!
//! The [`Sampler`] trait controls how dataset indices are visited each epoch.
//! Built-in implementations cover the common cases:
//!
//! - [`RandomSampler`] -- deterministic shuffle per epoch (default)
//! - [`SequentialSampler`] -- in-order, same every epoch (for eval/inference)
//!
//! Custom samplers (weighted, stratified, curriculum learning) implement
//! the [`Sampler`] trait directly.

/// Controls the order in which dataset indices are visited each epoch.
///
/// # Implementing a custom sampler
///
/// ```ignore
/// struct CurriculumSampler {
///     n: usize,
///     difficulty: Vec<f64>,
/// }
///
/// impl Sampler for CurriculumSampler {
///     fn len(&self) -> usize { self.n }
///     fn indices(&mut self, epoch: usize) -> Vec<usize> {
///         // Early epochs: easy samples first
///         // Later epochs: full shuffle
///         let mut idx: Vec<usize> = (0..self.n).collect();
///         if epoch < 10 {
///             idx.sort_by(|a, b| self.difficulty[*a].partial_cmp(&self.difficulty[*b]).unwrap());
///         } else {
///             let mut rng = Rng::seed(42 + epoch as u64);
///             rng.shuffle(&mut idx);
///         }
///         idx
///     }
/// }
/// ```
pub trait Sampler: Send {
    /// Total number of samples. Must match the dataset length.
    fn len(&self) -> usize;

    /// Whether the sampler is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate the index ordering for a given epoch.
    ///
    /// Must return indices in `[0, len())`, as many as
    /// [`epoch_len()`](Sampler::epoch_len) reports. Called once per epoch.
    fn indices(&mut self, epoch: usize) -> Vec<usize>;

    /// How many indices one epoch visits.
    ///
    /// Defaults to [`len()`](Sampler::len) — an epoch is a full pass over
    /// the data, which is what every sampler did before epoch splitting
    /// existed. [`SplitSampler`] overrides it, since there an epoch is a
    /// slice of a pass; the count is nominal in that case, as balanced
    /// slicing gives some epochs one extra index.
    ///
    /// Distinct from `len()` on purpose: `len()` describes the *dataset*
    /// (what [`DataLoader::len`](super::DataLoader::len) reports) while
    /// this describes an *epoch* (what
    /// [`DataLoader::num_batches`](super::DataLoader::num_batches)
    /// counts). They coincide unless the sampler splits.
    fn epoch_len(&self) -> usize {
        self.len()
    }
}

/// Deterministic random sampler. Default for [`DataLoader`](super::DataLoader).
///
/// Uses a per-epoch seed derived from `base_seed + epoch` to produce a
/// fresh permutation each epoch while remaining reproducible across runs.
pub struct RandomSampler {
    n: usize,
    seed: u64,
}

impl RandomSampler {
    /// Create a random sampler for `n` samples with the given base seed.
    pub fn new(n: usize, seed: u64) -> Self {
        RandomSampler { n, seed }
    }
}

impl Sampler for RandomSampler {
    fn len(&self) -> usize {
        self.n
    }

    fn indices(&mut self, epoch: usize) -> Vec<usize> {
        crate::rng::epoch_permutation(self.seed, epoch, self.n)
    }
}

/// Like [`RandomSampler`], but an epoch is a *slice* of a data pass.
///
/// `splits` says how finely to cut one pass. The pass permutation is
/// unchanged and still covers every sample exactly once; splitting only
/// decides how much of it one epoch consumes, so `splits` epochs make one
/// pass and no sample is seen twice along the way.
///
/// This is what makes single-pass training (the normal regime for LLM
/// pretraining) workable: everything that keys off the epoch boundary —
/// eval, checkpointing, reporting — gets a boundary to key off, where a
/// naive one-epoch run has none until teardown.
///
/// ```ignore
/// use flodl::SplitSampler;
///
/// // One pass over 10k samples, delivered as 20 epochs of 500.
/// let sampler = SplitSampler::new(10_000, 42, 20);
/// ```
///
/// At `splits = 1` it behaves exactly like [`RandomSampler`].
pub struct SplitSampler {
    n: usize,
    seed: u64,
    splits: usize,
}

impl SplitSampler {
    /// Create a split sampler for `n` samples with the given base seed.
    ///
    /// `splits` is clamped to at least 1.
    pub fn new(n: usize, seed: u64, splits: usize) -> Self {
        SplitSampler {
            n,
            seed,
            splits: splits.max(1),
        }
    }

    /// Slices per data pass.
    pub fn splits(&self) -> usize {
        self.splits
    }
}

impl Sampler for SplitSampler {
    fn len(&self) -> usize {
        self.n
    }

    fn epoch_len(&self) -> usize {
        // The base slice. Balanced splitting hands the first `n % splits`
        // epochs one extra index, so this is the nominal size — callers
        // that need the exact count read `indices(epoch).len()`.
        self.n / self.splits
    }

    fn indices(&mut self, epoch: usize) -> Vec<usize> {
        crate::rng::epoch_split_permutation(self.seed, epoch, self.splits, self.n)
    }
}

/// Sequential sampler: indices in order, same every epoch.
///
/// Use for evaluation or inference where order matters or
/// shuffling is undesirable.
pub struct SequentialSampler {
    n: usize,
}

impl SequentialSampler {
    /// Create a sequential sampler for `n` samples.
    pub fn new(n: usize) -> Self {
        SequentialSampler { n }
    }
}

impl Sampler for SequentialSampler {
    fn len(&self) -> usize {
        self.n
    }

    fn indices(&mut self, _epoch: usize) -> Vec<usize> {
        (0..self.n).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_sampler_permutation() {
        let mut sampler = RandomSampler::new(10, 42);
        let idx = sampler.indices(0);
        assert_eq!(idx.len(), 10);
        // Must contain all indices exactly once
        let mut sorted = idx.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_random_sampler_different_epochs() {
        let mut sampler = RandomSampler::new(100, 42);
        let epoch0 = sampler.indices(0);
        let epoch1 = sampler.indices(1);
        // Different epochs should produce different orderings
        assert_ne!(epoch0, epoch1);
    }

    #[test]
    fn test_random_sampler_reproducible() {
        let mut s1 = RandomSampler::new(100, 42);
        let mut s2 = RandomSampler::new(100, 42);
        // Same seed + same epoch = same permutation
        assert_eq!(s1.indices(5), s2.indices(5));
    }

    #[test]
    fn test_random_sampler_different_seeds() {
        let mut s1 = RandomSampler::new(100, 42);
        let mut s2 = RandomSampler::new(100, 99);
        // Different seeds = different permutation
        assert_ne!(s1.indices(0), s2.indices(0));
    }

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler::new(5);
        assert_eq!(sampler.indices(0), vec![0, 1, 2, 3, 4]);
        assert_eq!(sampler.indices(10), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sequential_sampler_stable() {
        let mut sampler = SequentialSampler::new(20);
        let a = sampler.indices(0);
        let b = sampler.indices(1);
        assert_eq!(a, b);
    }

    #[test]
    fn split_sampler_epochs_tile_one_pass() {
        // Four epochs consume one pass between them, in pass order and
        // with no sample served twice.
        let mut split = SplitSampler::new(100, 42, 4);
        let mut seen = Vec::new();
        for epoch in 0..4 {
            seen.extend(split.indices(epoch));
        }
        assert_eq!(seen, RandomSampler::new(100, 42).indices(0));
    }

    #[test]
    fn split_sampler_at_one_split_matches_random_sampler() {
        let mut split = SplitSampler::new(100, 42, 1);
        let mut random = RandomSampler::new(100, 42);
        for epoch in 0..3 {
            assert_eq!(split.indices(epoch), random.indices(epoch), "epoch {epoch}");
        }
    }

    #[test]
    fn split_sampler_reports_dataset_len_and_epoch_len_apart() {
        let split = SplitSampler::new(100, 42, 4);
        // The dataset is still 100 samples; an epoch is 25 of them.
        assert_eq!(split.len(), 100);
        assert_eq!(split.epoch_len(), 25);
        assert_eq!(split.splits(), 4);
    }

    #[test]
    fn unsplit_samplers_report_one_epoch_per_pass() {
        // The defaulted trait method: sampler types that predate
        // splitting must keep reporting the whole pass.
        assert_eq!(RandomSampler::new(50, 0).epoch_len(), 50);
        assert_eq!(SequentialSampler::new(30).epoch_len(), 30);
    }

    #[test]
    fn split_sampler_clamps_zero_splits() {
        let mut sampler = SplitSampler::new(10, 1, 0);
        assert_eq!(sampler.splits(), 1);
        assert_eq!(sampler.indices(0).len(), 10);
    }

    #[test]
    fn test_sampler_len() {
        let s1 = RandomSampler::new(50, 0);
        assert_eq!(s1.len(), 50);
        let s2 = SequentialSampler::new(30);
        assert_eq!(s2.len(), 30);
    }
}
