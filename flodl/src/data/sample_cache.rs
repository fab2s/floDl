//! Read-through, sample-keyed RAM cache for streaming datasets.
//!
//! Not part of the public API. Sits behind [`DataSet::get`](super::DataSet::get)
//! (inside `DataSetAdapter`), below batching: batches are not reusable
//! across epochs (a reshuffle changes their composition), samples are.
//! Epoch 1 populates while training; later epochs read at RAM speed
//! instead of storage speed. Staged content is reshuffle-invariant by
//! construction: the cache is keyed by sample identity, and a reshuffle
//! changes only the order function, never the content set.
//!
//! # Admission policy: fill until full, evict nothing
//!
//! Each epoch touches every sample exactly once in a fresh random
//! order, so for a cache holding K of N samples the expected hit rate
//! is K/N for ANY eviction policy — no choice of which K to keep beats
//! any other against a uniformly reshuffled scan. Admit-until-full
//! delivers that same K/N with zero churn (no eviction traffic, no
//! write contention after warm-up). Smarter eviction only becomes
//! meaningful when a disk tier exists below (spill-vs-drop rather than
//! keep-vs-drop).
//!
//! # Concurrency
//!
//! Lock-free by construction: reads are one atomic load per lookup
//! (`OnceLock::get`), writes are per-slot one-time. No global lock
//! anywhere near the data path. The byte counter is advisory under
//! concurrent writers (two racing inserts can overshoot the budget by
//! less than one sample each); in practice a single reader thread
//! populates it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::tensor::{Result, Tensor};

/// Sample-keyed read-through cache. One instance per `from_dataset`
/// loader, shared between the adapter (inside the `dyn BatchDataSet`
/// box) and the loader (which refreshes the budget at each `epoch()`).
///
/// Dormant until a budget is installed: with `budget == 0` a miss is a
/// pure pass-through (two atomic loads of overhead) and nothing is
/// retained.
pub(crate) struct SampleCache {
    /// One slot per sample index, set at most once.
    slots: Vec<OnceLock<Vec<Tensor>>>,
    /// Bytes currently retained (sum of cached samples' nbytes).
    bytes: AtomicUsize,
    /// Admission ceiling in bytes. Includes already-retained bytes:
    /// the per-epoch refresh computes `bytes() + headroom`, so a
    /// shrinking headroom stops NEW admissions but never drops staged
    /// content ("keep the subset already there").
    budget: AtomicUsize,
}

impl SampleCache {
    pub(crate) fn new(n: usize) -> Self {
        let mut slots = Vec::with_capacity(n);
        slots.resize_with(n, OnceLock::new);
        SampleCache {
            slots,
            bytes: AtomicUsize::new(0),
            budget: AtomicUsize::new(0),
        }
    }

    /// Bytes currently retained.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Number of samples currently cached (test/diagnostic).
    #[cfg(test)]
    pub(crate) fn cached_count(&self) -> usize {
        self.slots.iter().filter(|s| s.get().is_some()).count()
    }

    /// Install the admission ceiling. Called once per `epoch()` by the
    /// streaming loader; `0` means no new admissions (retained content
    /// stays served).
    pub(crate) fn set_budget(&self, bytes: usize) {
        self.budget.store(bytes, Ordering::Relaxed);
    }

    /// Read-through lookup: return the cached sample (shallow tensor
    /// clones — refcount bumps, no data copy) or run `fetch`, admitting
    /// the result while the budget allows.
    ///
    /// Cached tensors share storage with every clone handed out; the
    /// adapter's stacking path only reads them (`copy_` out of the
    /// sample into the batch row), never mutates. Datasets returning
    /// views into their own buffers cache the view (no byte
    /// duplication; `nbytes` accounting is conservative).
    pub(crate) fn get_or_fetch(
        &self,
        index: usize,
        fetch: impl FnOnce() -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        if let Some(hit) = self.slots.get(index).and_then(|s| s.get()) {
            return Ok(hit.clone());
        }

        let sample = fetch()?;

        if let Some(slot) = self.slots.get(index) {
            let sample_bytes: usize = sample.iter().map(|t| t.nbytes()).sum();
            if self.bytes.load(Ordering::Relaxed) + sample_bytes
                <= self.budget.load(Ordering::Relaxed)
                && slot.set(sample.clone()).is_ok()
            {
                self.bytes.fetch_add(sample_bytes, Ordering::Relaxed);
            }
        }

        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    fn sample(value: f32) -> Vec<Tensor> {
        // 4 bytes per sample (one f32).
        vec![Tensor::from_f32(&[value], &[1], Device::CPU).unwrap()]
    }

    #[test]
    fn dormant_cache_is_pure_pass_through() {
        let cache = SampleCache::new(4);
        let mut fetches = 0;
        for _ in 0..3 {
            let s = cache
                .get_or_fetch(1, || {
                    fetches += 1;
                    Ok(sample(1.0))
                })
                .unwrap();
            assert_eq!(s[0].to_f64_vec().unwrap(), vec![1.0]);
        }
        assert_eq!(fetches, 3, "budget 0: nothing retained, every call fetches");
        assert_eq!(cache.bytes(), 0);
        assert_eq!(cache.cached_count(), 0);
    }

    #[test]
    fn admits_until_budget_then_serves_hits() {
        let cache = SampleCache::new(4);
        cache.set_budget(8); // room for exactly two 4-byte samples

        let mut fetches = 0;
        for idx in 0..4 {
            let _ = cache
                .get_or_fetch(idx, || {
                    fetches += 1;
                    Ok(sample(idx as f32))
                })
                .unwrap();
        }
        assert_eq!(fetches, 4, "first pass: all misses");
        assert_eq!(cache.cached_count(), 2, "admission stopped at the budget");
        assert_eq!(cache.bytes(), 8);

        // Second pass: cached indices hit, the rest re-fetch.
        for idx in 0..4 {
            let s = cache
                .get_or_fetch(idx, || {
                    fetches += 1;
                    Ok(sample(idx as f32))
                })
                .unwrap();
            assert_eq!(s[0].to_f64_vec().unwrap(), vec![idx as f64]);
        }
        assert_eq!(fetches, 6, "two hits, two re-fetches");
    }

    #[test]
    fn budget_shrink_keeps_retained_content() {
        let cache = SampleCache::new(2);
        cache.set_budget(64);
        let _ = cache.get_or_fetch(0, || Ok(sample(7.0))).unwrap();
        assert_eq!(cache.cached_count(), 1);

        // Shrinking the budget below current bytes stops NEW
        // admissions but keeps serving what is already staged.
        cache.set_budget(0);
        let mut fetched = false;
        let s = cache
            .get_or_fetch(0, || {
                fetched = true;
                Ok(sample(0.0))
            })
            .unwrap();
        assert!(!fetched, "still a hit after budget shrink");
        assert_eq!(s[0].to_f64_vec().unwrap(), vec![7.0]);

        let _ = cache.get_or_fetch(1, || Ok(sample(1.0))).unwrap();
        assert_eq!(cache.cached_count(), 1, "no new admission at budget 0");
    }

    #[test]
    fn fetch_errors_pass_through_and_cache_nothing() {
        let cache = SampleCache::new(2);
        cache.set_budget(64);
        let out = cache.get_or_fetch(0, || {
            Err(crate::tensor::TensorError::new("io failure"))
        });
        assert!(out.is_err());
        assert_eq!(cache.cached_count(), 0);

        // A later successful fetch still admits.
        let _ = cache.get_or_fetch(0, || Ok(sample(1.0))).unwrap();
        assert_eq!(cache.cached_count(), 1);
    }

    #[test]
    fn out_of_range_index_is_fetch_only() {
        let cache = SampleCache::new(1);
        cache.set_budget(64);
        let s = cache.get_or_fetch(9, || Ok(sample(3.0))).unwrap();
        assert_eq!(s[0].to_f64_vec().unwrap(), vec![3.0]);
        assert_eq!(cache.bytes(), 0);
    }
}
