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

/// One reservation advisory: the rank's upcoming run-stream as
/// `(epoch, spans)` segments in walk order — each segment's spans in
/// certainty order (own span first, truing margins last), cross-epoch
/// segments walking into the next epoch's permutation. `counts` is the
/// current reduce-window schedule, used to split the host RAM budget
/// consumption-proportionally among co-hosted ranks. Latest advisory
/// wins.
pub(crate) struct StageAdvisory {
    pub counts: Vec<usize>,
    pub segments: Vec<(usize, Vec<(usize, usize)>)>,
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

/// The stager's RAM budget: the host's current headroom under a 50%
/// total-usage cap, split consumption-proportionally among the ranks
/// sharing this host — `budget_i ∝ rate_i` gives every rank the same
/// seconds of lookahead (equal time, not equal bytes). Co-hosted ranks
/// come from the cluster envelope when present; without one (thread
/// DDP, single host without an envelope) every rank is assumed
/// co-hosted — the conservative reading. `0` = do not stage.
fn stager_ram_budget(rank: usize, world_size: usize, counts: &[usize]) -> usize {
    let Some(m) = crate::sys::mem_info() else {
        return 0;
    };
    let cap = m.total_bytes / 2;
    let used = m.total_bytes.saturating_sub(m.available_bytes);
    let headroom = cap.saturating_sub(used);

    let local_ranks: Vec<usize> = crate::distributed::cluster::LocalCluster::from_env()
        .ok()
        .flatten()
        .map(|c| c.worker.ranks.clone())
        .unwrap_or_else(|| (0..world_size).collect());
    let share = host_share(rank, &local_ranks, counts);
    usize::try_from((headroom as f64 * share) as u64).unwrap_or(usize::MAX)
}

/// This rank's fraction of its host's staging budget: its schedule
/// count over the co-hosted ranks' total. Equal split when the
/// schedule is empty/zero (pre-calibration) or the rank is not in the
/// local list (defensive).
fn host_share(rank: usize, local_ranks: &[usize], counts: &[usize]) -> f64 {
    let n = local_ranks.len().max(1) as f64;
    if !local_ranks.contains(&rank) {
        return 1.0 / n;
    }
    let mine = counts.get(rank).copied().unwrap_or(0);
    let total: usize = local_ranks
        .iter()
        .map(|&r| counts.get(r).copied().unwrap_or(0))
        .sum();
    if mine == 0 || total == 0 {
        return 1.0 / n;
    }
    mine as f64 / total as f64
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
    rank: usize,
    world_size: usize,
) -> StagerHandle {
    let (tx, rx) = mpsc::channel::<StageAdvisory>();
    let staged = Arc::new(AtomicUsize::new(0));
    let staged_in_thread = Arc::clone(&staged);

    let join = std::thread::spawn(move || {
        stager_loop(dataset, cache, rx, base_seed, rank, world_size, &staged_in_thread);
    });

    StagerHandle {
        tx: Some(tx),
        join: Some(join),
        staged,
    }
}

#[allow(clippy::too_many_arguments)]
fn stager_loop(
    dataset: Arc<dyn BatchDataSet>,
    cache: Arc<SampleCache>,
    rx: mpsc::Receiver<StageAdvisory>,
    base_seed: u64,
    rank: usize,
    world_size: usize,
    staged: &AtomicUsize,
) {
    let dataset_len = dataset.len();
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
            // Budget refresh rides the advisory (which rides the reduce
            // clock): live host headroom × this rank's consumption
            // share among co-hosted ranks. A shrink stops new
            // admissions, never drops staged content.
            let budget = stager_ram_budget(rank, world_size, &a.counts);
            cache.set_budget(budget);
            queue.clear();
            if budget == 0 {
                // No headroom right now: reading ahead with nothing
                // retained would spend source bandwidth for nothing.
                // Stay alive — a later advisory may find room.
                continue;
            }
            for &(epoch, ref spans) in &a.segments {
                for &(offset, size) in spans {
                    queue.extend(make_partition(
                        offset,
                        size,
                        dataset_len,
                        epoch,
                        base_seed,
                    ));
                }
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

        let handle = spawn_stager(dataset, Arc::clone(&cache), 42, 0, 1);
        // Advisory: own span (0,4) + a margin span (8,2) of epoch 0,
        // plus a cross-epoch segment into epoch 1 — the stager walks
        // across the boundary without ceremony.
        handle.advise(StageAdvisory {
            counts: vec![4],
            segments: vec![(0, vec![(0, 4), (8, 2)]), (1, vec![(0, 2)])],
        });

        // Wait for the stager to drain the advisory.
        let mut waited = 0;
        while handle.staged_count() < 8 && waited < 400 {
            std::thread::sleep(std::time::Duration::from_millis(5));
            waited += 1;
        }
        assert_eq!(handle.staged_count(), 8, "all advised samples staged");
        assert!(cache.bytes() > 0, "tier warmed");

        // The training path now hits the warm tier: batches drawn from
        // the advised regions of BOTH epochs make no inner call.
        let before = calls.load(Ordering::Relaxed);
        let plan_e0 = make_partition(0, 4, 12, 0, 42);
        let _ = staged.get_batch(&plan_e0).unwrap();
        let plan_e1 = make_partition(0, 2, 12, 1, 42);
        let _ = staged.get_batch(&plan_e1).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), before, "served from the tier");

        drop(handle); // disconnect + join
    }

    #[test]
    fn host_share_is_consumption_proportional() {
        // Rank 1 does half the host's work → half the host's budget:
        // equal lookahead time, not equal bytes.
        let local = vec![0, 1, 2];
        let counts = vec![10, 20, 10, 40]; // rank 3 is on another host
        assert_eq!(host_share(1, &local, &counts), 0.5);
        assert_eq!(host_share(0, &local, &counts), 0.25);

        // Pre-calibration (zero counts) and foreign-rank fall back to
        // an equal split.
        assert_eq!(host_share(0, &local, &[0, 0, 0, 0]), 1.0 / 3.0);
        assert_eq!(host_share(3, &local, &counts), 1.0 / 3.0);

        // Lone rank on the host owns the whole budget regardless of
        // the global schedule.
        assert_eq!(host_share(3, &[3], &counts), 1.0);
    }
}
