//! CPU-side random number generator for data loading, shuffling, and augmentation.
//!
//! Wraps `SmallRng` (Xoshiro256++) — fast, correct, and audited. Not
//! cryptographic, but the right tier for ML workloads.
//!
//! For seeding libtorch tensor operations (dropout, randn, etc.), use
//! [`manual_seed`](crate::manual_seed) instead.

use rand::rngs::SmallRng;
use rand::distr::{Distribution, Uniform};
use rand::{RngExt, SeedableRng};
use rand::seq::SliceRandom;

/// A lightweight, deterministic random number generator.
///
/// ```ignore
/// use flodl::Rng;
///
/// let mut rng = Rng::seed(42);
/// let idx = rng.usize(100);        // uniform [0, 100)
/// let val = rng.f32();             // uniform [0, 1)
/// let coin = rng.bernoulli(0.5);   // true ~50% of the time
///
/// let mut data = vec![1, 2, 3, 4, 5];
/// rng.shuffle(&mut data);
/// ```
#[derive(Clone)]
pub struct Rng {
    inner: SmallRng,
}

impl Rng {
    /// Create a deterministic RNG from a fixed seed.
    pub fn seed(seed: u64) -> Self {
        Self { inner: SmallRng::seed_from_u64(seed) }
    }

    /// Create an RNG seeded from the operating system.
    pub fn from_entropy() -> Self {
        Self { inner: rand::make_rng() }
    }

    /// Uniform random `usize` in `[0, n)`.
    ///
    /// # Panics
    /// Panics if `n == 0`.
    pub fn usize(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::usize(0) is undefined");
        Uniform::new(0, n).unwrap().sample(&mut self.inner)
    }

    /// Uniform random `f32` in `[0, 1)`.
    pub fn f32(&mut self) -> f32 {
        self.inner.random()
    }

    /// Uniform random `f64` in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        self.inner.random()
    }

    /// Fisher-Yates shuffle of a mutable slice.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        slice.shuffle(&mut self.inner);
    }

    /// Returns `true` with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Uniform random `i64` in `[low, high)`.
    ///
    /// # Panics
    /// Panics if `low >= high`.
    pub fn range(&mut self, low: i64, high: i64) -> i64 {
        assert!(low < high, "Rng::range requires low < high, got {low} >= {high}");
        Uniform::new(low, high).unwrap().sample(&mut self.inner)
    }

    /// Sample from a normal distribution with given `mean` and `std`.
    ///
    /// Uses the Box-Muller transform to avoid pulling in `rand_distr`.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        // Box-Muller: two uniforms in (0,1) → one standard normal
        let u1: f64 = 1.0 - self.inner.random::<f64>(); // (0, 1] to avoid ln(0)
        let u2: f64 = self.inner.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
}

/// The epoch permutation every scheduling consumer shares: a
/// deterministic shuffle of `0..picks` seeded by `seed + epoch`.
/// `RandomSampler` (solo) and the coordinator's partition expansion
/// (DDP) both call this — one scheme, one home — so the stager's warm
/// hits and coverage-granular resume stay aligned with the training
/// order by construction. The output is a stream of PICKS: with
/// augmentation, `picks = samples * k` and a pick decodes as
/// `(pick / k, pick % k)` = (sample, repeat); at `k = 1` picks and
/// samples coincide and the scheme is byte-identical to what shipped.
pub(crate) fn epoch_permutation(seed: u64, epoch: usize, picks: usize) -> Vec<usize> {
    let mut rng = Rng::seed(seed.wrapping_add(epoch as u64));
    let mut all: Vec<usize> = (0..picks).collect();
    rng.shuffle(&mut all);
    all
}

/// Where one split lives inside its data pass: `(start, len)` into the
/// pass permutation.
///
/// `event` counts training events over the whole run, so `event /
/// splits` names the data pass and `event % splits` the slice within
/// it. Only the slice matters here — the pass index is what
/// [`epoch_permutation`] consumes.
///
/// Sizes are balanced: the first `picks % splits` splits carry one
/// extra pick. The alternative (last split absorbs the whole
/// remainder, as `ChunkPool` does for per-rank spans) is equivalent
/// when splits are few, but a split length IS the reduce-window bound
/// and the eval interval, and at fine splitting last-absorbs makes the
/// final one several times every other — `picks = 2047, splits = 100`
/// gives a 67-pick tail against 20-pick siblings. Balanced keeps every
/// event the same size to within one pick.
///
/// # Panics
/// Panics if `splits == 0`.
pub(crate) fn epoch_split_span(event: usize, splits: usize, picks: usize) -> (usize, usize) {
    assert!(splits > 0, "epoch splits must be >= 1, got 0");
    let split = event % splits;
    let base = picks / splits;
    let extra = picks % splits;
    // Splits before this one each took `base`, and the first `extra` of
    // them took one more.
    let start = split * base + split.min(extra);
    (start, base + usize::from(split < extra))
}

/// One split of the epoch permutation: the `event`-th slice of a
/// contiguous partition of [`epoch_permutation`].
///
/// `splits` separates the two meanings "epoch" used to fuse — a full
/// pass over the data, and a periodic event during training. The pass
/// permutation is unchanged and still covers every pick exactly once;
/// splitting only decides how much of it one event consumes.
/// Concatenating events `p * splits .. (p + 1) * splits` reproduces
/// pass `p` exactly, so splits are disjoint and jointly complete by
/// construction — a run gains interior boundaries without seeing any
/// sample twice.
///
/// Every consumer that keys off the epoch boundary (the `window <=
/// epoch` reduce cap, eval cadence, checkpointing, coverage resume)
/// inherits the finer boundary with no further plumbing, which is what
/// makes a single-pass run checkpointable and evaluable at all.
///
/// At `splits = 1` this is `epoch_permutation(seed, event, picks)` byte
/// for byte, so defaults reproduce shipped runs exactly.
///
/// Cost is one full shuffle per event rather than per pass. Callers
/// stepping many splits over a large corpus can hold the pass
/// permutation across events of the same pass and slice it with
/// [`epoch_split_span`] instead.
///
/// # Panics
/// Panics if `splits == 0`.
pub(crate) fn epoch_split_permutation(
    seed: u64,
    event: usize,
    splits: usize,
    picks: usize,
) -> Vec<usize> {
    let (start, len) = epoch_split_span(event, splits, picks);
    let all = epoch_permutation(seed, event / splits, picks);
    all[start..start + len].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut a = Rng::seed(42);
        let mut b = Rng::seed(42);
        let va: Vec<f64> = (0..100).map(|_| a.f64()).collect();
        let vb: Vec<f64> = (0..100).map(|_| b.f64()).collect();
        assert_eq!(va, vb);
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Rng::seed(1);
        let mut b = Rng::seed(2);
        let va: Vec<f64> = (0..20).map(|_| a.f64()).collect();
        let vb: Vec<f64> = (0..20).map(|_| b.f64()).collect();
        assert_ne!(va, vb);
    }

    #[test]
    fn usize_in_range() {
        let mut rng = Rng::seed(0);
        for _ in 0..1000 {
            let v = rng.usize(10);
            assert!(v < 10);
        }
    }

    #[test]
    #[should_panic(expected = "usize(0) is undefined")]
    fn usize_zero_panics() {
        Rng::seed(0).usize(0);
    }

    #[test]
    fn f32_in_unit_interval() {
        let mut rng = Rng::seed(0);
        for _ in 0..1000 {
            let v = rng.f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut rng = Rng::seed(0);
        for _ in 0..1000 {
            let v = rng.f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = Rng::seed(42);
        let mut data = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut data);
        data.sort();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn shuffle_deterministic() {
        let mut a = Rng::seed(42);
        let mut b = Rng::seed(42);
        let mut da = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut db = da.clone();
        a.shuffle(&mut da);
        b.shuffle(&mut db);
        assert_eq!(da, db);
    }

    #[test]
    fn bernoulli_extremes() {
        let mut rng = Rng::seed(0);
        // p=0 always false
        for _ in 0..100 {
            assert!(!rng.bernoulli(0.0));
        }
        // p=1 always true
        for _ in 0..100 {
            assert!(rng.bernoulli(1.0));
        }
    }

    #[test]
    fn bernoulli_roughly_half() {
        let mut rng = Rng::seed(42);
        let n = 10_000;
        let hits = (0..n).filter(|_| rng.bernoulli(0.5)).count();
        let ratio = hits as f64 / n as f64;
        assert!((0.45..0.55).contains(&ratio), "bernoulli(0.5) ratio = {ratio}");
    }

    #[test]
    fn range_bounds() {
        let mut rng = Rng::seed(0);
        for _ in 0..1000 {
            let v = rng.range(-5, 5);
            assert!((-5..5).contains(&v));
        }
    }

    #[test]
    #[should_panic(expected = "low < high")]
    fn range_empty_panics() {
        Rng::seed(0).range(5, 5);
    }

    #[test]
    fn normal_statistical() {
        let mut rng = Rng::seed(42);
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal(3.0, 0.5)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        assert!((2.95..3.05).contains(&mean), "normal mean = {mean}");
        assert!((0.47..0.53).contains(&std), "normal std = {std}");
    }

    #[test]
    fn epoch_permutation_matches_the_released_scheme() {
        // The shared scheme must stay byte-identical to what shipped —
        // coverage-granular resume and the stager's warm hits both
        // depend on `Rng::seed(seed + epoch)` over `0..picks` exactly.
        let (seed, epoch, n) = (42u64, 3usize, 100usize);
        let mut rng = Rng::seed(seed.wrapping_add(epoch as u64));
        let mut expected: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut expected);
        assert_eq!(epoch_permutation(seed, epoch, n), expected);
    }

    #[test]
    fn one_split_is_the_unsplit_scheme() {
        // The default must reproduce shipped runs, event for event.
        for event in 0..6 {
            assert_eq!(
                epoch_split_permutation(42, event, 1, 100),
                epoch_permutation(42, event, 100),
                "event {event}"
            );
        }
        assert_eq!(epoch_split_span(3, 1, 100), (0, 100));
    }

    #[test]
    fn splits_of_a_pass_concatenate_to_the_pass() {
        // Disjointness, coverage AND ordering in one assert: the splits
        // of pass `p` are that pass's permutation, cut in place.
        let (seed, picks, splits, pass) = (42u64, 100usize, 7usize, 2usize);
        let mut cat = Vec::new();
        for split in 0..splits {
            cat.extend(epoch_split_permutation(seed, pass * splits + split, splits, picks));
        }
        assert_eq!(cat, epoch_permutation(seed, pass, picks));
    }

    #[test]
    fn split_sizes_are_balanced_and_contiguous() {
        let (picks, splits) = (100usize, 7usize);
        // 100 / 7 = 14 r 2 → the first two splits carry the remainder.
        let sizes: Vec<usize> =
            (0..splits).map(|s| epoch_split_span(s, splits, picks).1).collect();
        assert_eq!(sizes, vec![15, 15, 14, 14, 14, 14, 14]);

        let mut at = 0;
        for split in 0..splits {
            let (start, len) = epoch_split_span(split, splits, picks);
            assert_eq!(start, at, "split {split} must start where the previous ended");
            at += len;
        }
        assert_eq!(at, picks, "splits must cover the pass exactly");
    }

    #[test]
    fn split_sizes_stay_balanced_when_the_pass_divides_evenly() {
        let sizes: Vec<usize> = (0..4).map(|s| epoch_split_span(s, 4, 100).1).collect();
        assert_eq!(sizes, vec![25, 25, 25, 25]);
    }

    #[test]
    fn splits_more_numerous_than_picks_degrade_gracefully() {
        // Degenerate but not wrong: three picks, five events. The
        // trailing events are empty rather than the last one hoarding.
        let sizes: Vec<usize> = (0..5).map(|s| epoch_split_span(s, 5, 3).1).collect();
        assert_eq!(sizes, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn crossing_splits_starts_a_new_pass() {
        let (seed, picks, splits) = (42u64, 100usize, 4usize);
        // event 0 and event `splits` are both split 0, of pass 0 and 1.
        assert_eq!(epoch_split_span(0, splits, picks), epoch_split_span(splits, splits, picks));
        assert_ne!(
            epoch_split_permutation(seed, 0, splits, picks),
            epoch_split_permutation(seed, splits, splits, picks),
        );
    }

    #[test]
    fn split_permutation_is_deterministic() {
        let a = epoch_split_permutation(7, 11, 5, 250);
        let b = epoch_split_permutation(7, 11, 5, 250);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "epoch splits must be >= 1")]
    fn zero_splits_panics() {
        epoch_split_span(0, 0, 100);
    }

    #[test]
    #[should_panic(expected = "epoch splits must be >= 1")]
    fn zero_splits_panics_before_dividing_by_it() {
        // The span check must run ahead of `event / splits`, or the
        // caller gets a bare divide-by-zero instead of the reason.
        epoch_split_permutation(42, 3, 0, 100);
    }

    #[test]
    fn clone_preserves_state() {
        let mut a = Rng::seed(42);
        // advance a few steps
        for _ in 0..10 { a.f64(); }
        let mut b = a.clone();
        let va: Vec<f64> = (0..50).map(|_| a.f64()).collect();
        let vb: Vec<f64> = (0..50).map(|_| b.f64()).collect();
        assert_eq!(va, vb);
    }

    #[test]
    fn from_entropy_works() {
        let mut rng = Rng::from_entropy();
        // just verify it doesn't panic and produces values
        let v = rng.f64();
        assert!((0.0..1.0).contains(&v));
    }
}
