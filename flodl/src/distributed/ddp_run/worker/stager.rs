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

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::data::sample_cache::SampleCache;
use crate::data::BatchDataSet;
use crate::tensor::{Result, Tensor, TensorOptions};

use super::super::make_partition;

// ---------------------------------------------------------------------------
// StreamPool (the flow window beyond the pinned tier)
// ---------------------------------------------------------------------------

/// Bounded pool for the beyond-pinned-budget portion of the advised
/// stream. Where the pinned tier keeps a static set (optimal for what
/// it holds: any K-set ties under a reshuffled scan), this pool is a
/// sliding window over the known future: the stager fills it ahead of
/// the training frontier, training consumption pops entries as the
/// frontier passes (drop-behind), and admission under pressure evicts
/// the entry whose next use in the advised stream is FARTHEST — keep
/// what recurs soonest, throw away last what is needed first. Next-use
/// positions are recomputed from each advisory (the window clock), so
/// the priority is always against the current stream.
pub(crate) struct StreamPool {
    entries: HashMap<usize, StreamEntry>,
    bytes: usize,
    budget: usize,
}

struct StreamEntry {
    rows: Vec<Tensor>,
    bytes: usize,
    /// Position of this sample's next use in the advised stream —
    /// smaller = needed sooner. Refreshed on each advisory; entries
    /// absent from the new stream get `usize::MAX` (evicted first).
    next_use: usize,
}

impl StreamPool {
    pub(crate) fn new() -> Self {
        StreamPool {
            entries: HashMap::new(),
            bytes: 0,
            budget: 0,
        }
    }

    pub(crate) fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
    }

    pub(crate) fn contains(&self, index: usize) -> bool {
        self.entries.contains_key(&index)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Pop-on-hit: consumption IS the drop-behind. The frontier passed
    /// this sample; its slot goes to the lookahead.
    pub(crate) fn take(&mut self, index: usize) -> Option<Vec<Tensor>> {
        let entry = self.entries.remove(&index)?;
        self.bytes -= entry.bytes;
        Some(entry.rows)
    }

    /// Admit with next-use-priority eviction: make room by evicting
    /// strictly-farther entries; decline when the pool is full of
    /// sooner-needed samples (the caller pauses rather than fetching
    /// past a full window). `false` = declined.
    pub(crate) fn offer(&mut self, index: usize, rows: Vec<Tensor>, next_use: usize) -> bool {
        let bytes: usize = rows.iter().map(|t| t.nbytes()).sum();
        if bytes > self.budget {
            return false;
        }
        while self.bytes + bytes > self.budget {
            let farthest = match self
                .entries
                .iter()
                .max_by_key(|(_, e)| e.next_use)
                .map(|(&i, e)| (i, e.next_use))
            {
                Some(f) => f,
                None => return false, // nothing held, still no room
            };
            if farthest.1 <= next_use {
                return false; // everything held is needed sooner
            }
            let evicted = self.entries.remove(&farthest.0).expect("just found");
            self.bytes -= evicted.bytes;
        }
        self.bytes += bytes;
        self.entries.insert(
            index,
            StreamEntry {
                rows,
                bytes,
                next_use,
            },
        );
        true
    }

    /// Re-key every held entry's next-use position against a fresh
    /// advised stream (called per advisory). Absent entries get
    /// `usize::MAX`: not in the visible future, first to go.
    pub(crate) fn refresh_positions(&mut self, positions: &HashMap<usize, usize>) {
        for (idx, entry) in self.entries.iter_mut() {
            entry.next_use = positions.get(idx).copied().unwrap_or(usize::MAX);
        }
    }

    /// Whether a sample at stream position `next_use` (of estimated
    /// size `bytes`) could currently be admitted.
    fn has_room_for(&self, bytes: usize, next_use: usize) -> bool {
        if bytes > self.budget {
            return false;
        }
        if self.bytes + bytes <= self.budget {
            return true;
        }
        self.entries.values().any(|e| e.next_use > next_use)
    }
}

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
    /// Flow window beyond the pinned tier. Consumption pops entries
    /// (drop-behind).
    stream: Arc<Mutex<StreamPool>>,
}

impl StagedBatchDataSet {
    pub(crate) fn new(
        inner: Arc<dyn BatchDataSet>,
        cache: Arc<SampleCache>,
        stream: Arc<Mutex<StreamPool>>,
    ) -> Self {
        StagedBatchDataSet {
            inner,
            cache,
            stream,
        }
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
                    // Flow window next: a hit is popped — consumption
                    // is the drop-behind.
                    if let Some(row) = self.stream.lock().ok().and_then(|mut p| p.take(idx)) {
                        rows.push(Some(row));
                    } else {
                        rows.push(None);
                        missing.push(idx);
                        missing_pos.push(pos);
                    }
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
    stream: Arc<Mutex<StreamPool>>,
    base_seed: u64,
    rank: usize,
    world_size: usize,
) -> StagerHandle {
    let (tx, rx) = mpsc::channel::<StageAdvisory>();
    let staged = Arc::new(AtomicUsize::new(0));
    let staged_in_thread = Arc::clone(&staged);

    let join = std::thread::spawn(move || {
        stager_loop(
            dataset,
            cache,
            stream,
            rx,
            base_seed,
            rank,
            world_size,
            &staged_in_thread,
        );
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
    stream: Arc<Mutex<StreamPool>>,
    rx: mpsc::Receiver<StageAdvisory>,
    base_seed: u64,
    rank: usize,
    world_size: usize,
    staged: &AtomicUsize,
) {
    let dataset_len = dataset.len();
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    // Stream position of the queue front (the next-use priority key).
    let mut pos: usize = 0;
    let mut pinned_budget: usize = 0;
    // Learned from the first staged sample; prices the room checks.
    let mut sample_bytes: usize = 0;
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
            // share among co-hosted ranks, split between the pinned
            // tier and the flow window. A shrink stops new admissions,
            // never drops staged content.
            let share = stager_ram_budget(rank, world_size, &a.counts);
            let stream_budget = share / 4;
            pinned_budget = share - stream_budget;
            cache.set_budget(pinned_budget);
            queue.clear();
            pos = 0;
            if share == 0 {
                // No headroom right now: reading ahead with nothing
                // retained would spend source bandwidth for nothing.
                // Stay alive — a later advisory may find room.
                if let Ok(mut p) = stream.lock() {
                    p.set_budget(0);
                }
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
            // Re-key the flow window's next-use priorities against the
            // fresh stream (first occurrence wins).
            let mut positions: HashMap<usize, usize> = HashMap::new();
            for (i, &idx) in queue.iter().enumerate() {
                positions.entry(idx).or_insert(i);
            }
            if let Ok(mut p) = stream.lock() {
                p.set_budget(stream_budget);
                p.refresh_positions(&positions);
            }
        }

        match queue.front().copied() {
            Some(idx) => {
                // Already staged in either tier: cheap skip, which is
                // what makes the per-window stream re-walk affordable.
                let in_stream = stream.lock().map(|p| p.contains(idx)).unwrap_or(false);
                if cache.contains_ram(idx) || in_stream {
                    queue.pop_front();
                    pos += 1;
                    staged.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Room check BEFORE fetching — never spend a source
                // read on a sample nothing can retain. The flow window
                // frees room as training consumes (pop-on-hit), so
                // full-of-sooner-data means: wait for the frontier or
                // the next advisory.
                let pinned_room = cache.bytes() + sample_bytes <= pinned_budget;
                let stream_room = stream
                    .lock()
                    .map(|p| p.has_room_for(sample_bytes.max(1), pos))
                    .unwrap_or(false);
                if !pinned_room && !stream_room {
                    match rx.recv_timeout(std::time::Duration::from_millis(20)) {
                        Ok(a) => pending = Some(a),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                    continue;
                }

                queue.pop_front();
                // Read-through: the wrapper admits to the pinned tier
                // while it has room; what pinned declined goes to the
                // flow window with its next-use position. Errors are
                // the training path's to surface (same source) — the
                // stager moves on.
                if let Ok(batch) = dataset.get_batch(&[idx]) {
                    if sample_bytes == 0 {
                        sample_bytes = batch.iter().map(|t| t.nbytes()).sum();
                    }
                    if !cache.contains_ram(idx) {
                        if let Ok(mut p) = stream.lock() {
                            let _ = p.offer(idx, batch, pos);
                        }
                    }
                    staged.fetch_add(1, Ordering::Relaxed);
                }
                pos += 1;
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

    fn staged_setup(
        n: usize,
    ) -> (
        Arc<StagedBatchDataSet>,
        Arc<SampleCache>,
        Arc<Mutex<StreamPool>>,
        Arc<AtomicUsize>,
    ) {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn BatchDataSet> = Arc::new(Probe {
            n,
            calls: Arc::clone(&calls),
        });
        let cache = Arc::new(SampleCache::new(n));
        let stream = Arc::new(Mutex::new(StreamPool::new()));
        let staged = Arc::new(StagedBatchDataSet::new(
            inner,
            Arc::clone(&cache),
            Arc::clone(&stream),
        ));
        (staged, cache, stream, calls)
    }

    /// One-row sample as the pool stores it ([1, 1] f32).
    fn row(v: f32) -> Vec<Tensor> {
        vec![Tensor::from_f32(&[v], &[1, 1], Device::CPU).unwrap()]
    }

    #[test]
    fn stream_pool_evicts_farthest_next_use() {
        let mut pool = StreamPool::new();
        pool.set_budget(8); // two 4-byte rows

        assert!(pool.offer(10, row(10.0), 5));
        assert!(pool.offer(11, row(11.0), 9));
        assert_eq!(pool.len(), 2);

        // Nearer sample evicts the farthest-needed entry (11 @ 9).
        assert!(pool.offer(12, row(12.0), 2));
        assert_eq!(pool.len(), 2);
        assert!(pool.contains(10) && pool.contains(12));

        // Farther than everything held: declined.
        assert!(!pool.offer(13, row(13.0), 20));
        assert!(!pool.has_room_for(4, 20));
        assert!(pool.has_room_for(4, 1), "room by evicting a farther entry");

        // Consumption pops (drop-behind) and frees room.
        let r = pool.take(12).unwrap();
        assert_eq!(r[0].to_f64_vec().unwrap(), vec![12.0]);
        assert!(!pool.contains(12));
        assert!(pool.offer(13, row(13.0), 20), "room after the frontier passed");
    }

    #[test]
    fn stream_pool_refresh_rekeys_next_use() {
        let mut pool = StreamPool::new();
        pool.set_budget(8);
        assert!(pool.offer(1, row(1.0), 3));
        assert!(pool.offer(2, row(2.0), 4));

        // New advised stream: sample 2 recurs at position 0, sample 1
        // vanished (consumed, not visible ahead) → MAX, evicted first.
        let positions: HashMap<usize, usize> = [(2usize, 0usize)].into_iter().collect();
        pool.refresh_positions(&positions);
        assert!(pool.offer(3, row(3.0), 7));
        assert!(!pool.contains(1), "vanished-from-stream entry evicted first");
        assert!(pool.contains(2), "soonest-recurring entry kept");
    }

    #[test]
    fn dormant_wrapper_is_pass_through() {
        let (staged, cache, _stream, calls) = staged_setup(8);
        // Budget 0: every batch is one bulk inner call, nothing retained.
        let b = staged.get_batch(&[1, 3, 5]).unwrap();
        assert_eq!(b[0].to_f64_vec().unwrap(), vec![1.0, 3.0, 5.0]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn staged_rows_stitch_with_misses_in_order() {
        let (staged, cache, _stream, calls) = staged_setup(8);
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
        let (staged, cache, stream, calls) = staged_setup(12);
        let dataset: Arc<dyn BatchDataSet> = Arc::clone(&staged) as Arc<dyn BatchDataSet>;

        let handle = spawn_stager(dataset, Arc::clone(&cache), stream, 42, 0, 1);
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
