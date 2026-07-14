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
    /// Must return exactly [`len()`](Sampler::len) indices, each in `[0, len())`.
    /// Called once per epoch.
    fn indices(&mut self, epoch: usize) -> Vec<usize>;
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
    fn test_sampler_len() {
        let s1 = RandomSampler::new(50, 0);
        assert_eq!(s1.len(), 50);
        let s2 = SequentialSampler::new(30);
        assert_eq!(s2.len(), 30);
    }
}
