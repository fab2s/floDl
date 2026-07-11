//! Background data stager for cluster workers.
//!
//! Walks the coordinator's reservation advisories
//! ([`ControlMsg::StageAdvisory`](super::super::ControlMsg)) and reads
//! upcoming samples ahead of the training frontier, warming a
//! sample-keyed staging tier that the live prefetch path shares. The
//! coordinator's chunk allocation stays the only execution authority:
//! staging may overlap across ranks near reservation boundaries, but
//! only allocated work executes, so staged-and-allocated-elsewhere
//! data needs no invalidation — it just ages out.
//!
//! The tier is [`SampleCache`] from the data module, reused whole:
//! sample-keyed means staged content is reshuffle-invariant, and the
//! read-through wrapper below makes the training path populate and
//! consume the same cache the stager warms. Dormant (budget 0, pure
//! pass-through) until the first advisory arrives — non-progressive
//! runs, tests, and thread-based DDP never pay for it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::data::sample_cache::SampleCache;
use crate::data::BatchDataSet;
use crate::tensor::{Result, Tensor, TensorOptions};

use super::super::make_partition;

/// One reservation advisory: certainty-ordered `(offset, size)` spans
/// into `epoch`'s global permutation (own span first, truing margins
/// last). Latest advisory wins.
pub(crate) struct StageAdvisory {
    pub epoch: usize,
    pub spans: Vec<(usize, usize)>,
}

/// `BatchDataSet` wrapper making `get_batch` read-through against the
/// staging tier: cached rows are served from RAM/disk, the misses go to
/// the inner dataset in ONE bulk call, and fetched rows are admitted on
/// the way out. Row order is preserved exactly. With the cache dormant
/// the overhead is one atomic load per row.
pub(crate) struct StagedBatchDataSet {
    inner: Arc<dyn BatchDataSet>,
    cache: Arc<SampleCache>,
}

impl StagedBatchDataSet {
    pub(crate) fn new(inner: Arc<dyn BatchDataSet>, cache: Arc<SampleCache>) -> Self {
        StagedBatchDataSet { inner, cache }
    }

    /// Owned single-row copy of row `j` (batch dim kept at 1). A plain
    /// `narrow` view would keep the WHOLE source batch alive from the
    /// cache; the explicit copy guarantees per-row storage.
    fn owned_row(batch: &[Tensor], j: i64) -> Result<Vec<Tensor>> {
        batch
            .iter()
            .map(|t| {
                let view = t.narrow(0, j, 1)?;
                let out = Tensor::empty(
                    &view.shape(),
                    TensorOptions {
                        dtype: t.dtype(),
                        device: t.device(),
                    },
                )?;
                out.copy_(&view, false)?;
                Ok(out)
            })
            .collect()
    }

    fn admit_rows(&self, indices: &[usize], batch: &[Tensor]) -> Result<()> {
        for (j, &idx) in indices.iter().enumerate() {
            let row = Self::owned_row(batch, j as i64)?;
            self.cache.admit(idx, &row);
        }
        Ok(())
    }
}

impl BatchDataSet for StagedBatchDataSet {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        if indices.is_empty() {
            return self.inner.get_batch(indices);
        }

        let mut rows: Vec<Option<Vec<Tensor>>> = Vec::with_capacity(indices.len());
        let mut missing: Vec<usize> = Vec::new();
        let mut missing_pos: Vec<usize> = Vec::new();
        for (pos, &idx) in indices.iter().enumerate() {
            match self.cache.lookup(idx) {
                Some(hit) => rows.push(Some(hit?)),
                None => {
                    rows.push(None);
                    missing.push(idx);
                    missing_pos.push(pos);
                }
            }
        }

        // Fast path (and the dormant-cache path): nothing staged — one
        // bulk fetch, admitted read-through on the way out.
        if missing.len() == indices.len() {
            let batch = self.inner.get_batch(indices)?;
            self.admit_rows(indices, &batch)?;
            return Ok(batch);
        }

        // Mixed: one bulk fetch for the misses, then stitch rows back
        // in the caller's order.
        if !missing.is_empty() {
            let fetched = self.inner.get_batch(&missing)?;
            self.admit_rows(&missing, &fetched)?;
            for (j, &pos) in missing_pos.iter().enumerate() {
                let row: Vec<Tensor> = fetched
                    .iter()
                    .map(|t| t.narrow(0, j as i64, 1))
                    .collect::<Result<_>>()?;
                rows[pos] = Some(row);
            }
        }

        let first = rows[0].as_ref().expect("all rows resolved");
        let n_tensors = first.len();
        let mut out = Vec::with_capacity(n_tensors);
        for p in 0..n_tensors {
            let parts: Vec<&Tensor> = rows
                .iter()
                .map(|r| &r.as_ref().expect("all rows resolved")[p])
                .collect();
            out.push(Tensor::cat_many(&parts, 0)?);
        }
        Ok(out)
    }
}

/// Handle to the background stager: advisory inbox + join-on-drop.
pub(crate) struct StagerHandle {
    tx: Option<mpsc::Sender<StageAdvisory>>,
    join: Option<JoinHandle<()>>,
    /// Samples staged so far (read by tests; written by the thread).
    #[allow(dead_code)]
    staged: Arc<AtomicUsize>,
}

impl StagerHandle {
    /// Forward an advisory to the stager (never blocks; best-effort).
    pub(crate) fn advise(&self, advisory: StageAdvisory) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(advisory);
        }
    }

    #[cfg(test)]
    pub(crate) fn staged_count(&self) -> usize {
        self.staged.load(Ordering::Relaxed)
    }
}

impl Drop for StagerHandle {
    fn drop(&mut self) {
        // Disconnect first so the thread's recv unblocks, then join.
        self.tx.take();
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// The stager's RAM budget: half the host's current headroom under a
/// 50% total-usage cap, split by world size — the conservative reading
/// of "several ranks may share this host" until the controller sends
/// consumption-proportional shares. `0` = do not stage.
fn stager_ram_budget(world_size: usize) -> usize {
    let Some(m) = crate::sys::mem_info() else {
        return 0;
    };
    let cap = m.total_bytes / 2;
    let used = m.total_bytes.saturating_sub(m.available_bytes);
    let headroom = cap.saturating_sub(used);
    usize::try_from(headroom / world_size.max(1) as u64).unwrap_or(usize::MAX)
}

/// Spawn the background stager thread.
///
/// It waits for advisories, expands their spans into concrete sample
/// indices via the shared permutation (`make_partition`, same seed the
/// training path uses), and reads them through `dataset` (the
/// [`StagedBatchDataSet`] wrapper) in stream order — which admits them
/// into the shared tier. A newer advisory replaces the current walk
/// (latest wins, checked between samples). The tier budget installs on
/// the first advisory; if the host has no headroom the thread exits
/// and staging stays off.
pub(crate) fn spawn_stager(
    dataset: Arc<dyn BatchDataSet>,
    cache: Arc<SampleCache>,
    base_seed: u64,
    world_size: usize,
) -> StagerHandle {
    let (tx, rx) = mpsc::channel::<StageAdvisory>();
    let staged = Arc::new(AtomicUsize::new(0));
    let staged_in_thread = Arc::clone(&staged);

    let join = std::thread::spawn(move || {
        stager_loop(dataset, cache, rx, base_seed, world_size, &staged_in_thread);
    });

    StagerHandle {
        tx: Some(tx),
        join: Some(join),
        staged,
    }
}

fn stager_loop(
    dataset: Arc<dyn BatchDataSet>,
    cache: Arc<SampleCache>,
    rx: mpsc::Receiver<StageAdvisory>,
    base_seed: u64,
    world_size: usize,
    staged: &AtomicUsize,
) {
    let dataset_len = dataset.len();
    let mut budget_installed = false;
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut pending: Option<StageAdvisory> = None;

    loop {
        // Latest advisory wins: drain the inbox.
        loop {
            match rx.try_recv() {
                Ok(a) => pending = Some(a),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        }

        if let Some(a) = pending.take() {
            if !budget_installed {
                let budget = stager_ram_budget(world_size);
                if budget == 0 {
                    // No headroom: reading ahead with nothing retained
                    // would spend source bandwidth for nothing.
                    return;
                }
                cache.set_budget(budget);
                budget_installed = true;
            }
            queue.clear();
            for &(offset, size) in &a.spans {
                queue.extend(make_partition(
                    offset,
                    size,
                    dataset_len,
                    a.epoch,
                    base_seed,
                ));
            }
        }

        match queue.pop_front() {
            Some(idx) => {
                // Read-through: the wrapper admits the row; the result
                // itself is discarded. Errors are the training path's
                // to surface (it reads the same source) — the stager
                // just moves on.
                if dataset.get_batch(&[idx]).is_ok() {
                    staged.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => {
                // Nothing to stage: block until the next advisory.
                match rx.recv() {
                    Ok(a) => pending = Some(a),
                    Err(_) => return,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Device;

    /// Counts bulk fetches and records which indices were requested.
    struct Probe {
        n: usize,
        calls: Arc<AtomicUsize>,
    }

    impl BatchDataSet for Probe {
        fn len(&self) -> usize {
            self.n
        }
        fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let v: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
            Ok(vec![Tensor::from_f32(
                &v,
                &[v.len() as i64, 1],
                Device::CPU,
            )?])
        }
    }

    fn staged_setup(n: usize) -> (Arc<StagedBatchDataSet>, Arc<SampleCache>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn BatchDataSet> = Arc::new(Probe {
            n,
            calls: Arc::clone(&calls),
        });
        let cache = Arc::new(SampleCache::new(n));
        let staged = Arc::new(StagedBatchDataSet::new(inner, Arc::clone(&cache)));
        (staged, cache, calls)
    }

    #[test]
    fn dormant_wrapper_is_pass_through() {
        let (staged, cache, calls) = staged_setup(8);
        // Budget 0: every batch is one bulk inner call, nothing retained.
        let b = staged.get_batch(&[1, 3, 5]).unwrap();
        assert_eq!(b[0].to_f64_vec().unwrap(), vec![1.0, 3.0, 5.0]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn staged_rows_stitch_with_misses_in_order() {
        let (staged, cache, calls) = staged_setup(8);
        cache.set_budget(1 << 20);

        // Stage rows 2 and 5 the way the stager does.
        let _ = staged.get_batch(&[2]).unwrap();
        let _ = staged.get_batch(&[5]).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // Mixed batch: only the misses hit the inner dataset (one bulk
        // call), and row order is the caller's.
        let b = staged.get_batch(&[5, 0, 2, 7]).unwrap();
        assert_eq!(b[0].to_f64_vec().unwrap(), vec![5.0, 0.0, 2.0, 7.0]);
        assert_eq!(b[0].shape(), &[4, 1]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);

        // Fully staged batch: zero inner calls.
        let b = staged.get_batch(&[7, 2]).unwrap();
        assert_eq!(b[0].to_f64_vec().unwrap(), vec![7.0, 2.0]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn stager_walks_advisory_and_warms_shared_tier() {
        let (staged, cache, calls) = staged_setup(12);
        let dataset: Arc<dyn BatchDataSet> = Arc::clone(&staged) as Arc<dyn BatchDataSet>;

        let handle = spawn_stager(dataset, Arc::clone(&cache), 42, 1);
        // Advisory: own span (0,4) of epoch 0 + a margin span (8,2).
        handle.advise(StageAdvisory {
            epoch: 0,
            spans: vec![(0, 4), (8, 2)],
        });

        // Wait for the stager to drain the advisory.
        let mut waited = 0;
        while handle.staged_count() < 6 && waited < 400 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            waited += 1;
        }
        assert_eq!(handle.staged_count(), 6, "all advised samples staged");
        assert!(cache.bytes() > 0, "tier warmed");

        // The training path now hits the warm tier: a batch drawn from
        // the advised region makes no inner call.
        let before = calls.load(Ordering::Relaxed);
        let plan = make_partition(0, 4, 12, 0, 42);
        let _ = staged.get_batch(&plan).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), before, "served from the tier");

        drop(handle); // disconnect + join
    }
}
