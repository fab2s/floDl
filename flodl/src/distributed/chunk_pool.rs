//! Chunk pool for progressive dispatch, with per-rank reservations.
//!
//! Tracks remaining unassigned samples for one epoch during progressive
//! dispatch. Instead of sending the full partition at epoch start, the
//! coordinator hands out small chunks from this pool.
//!
//! # Reservations
//!
//! The epoch's permutation is partitioned into contiguous per-rank
//! **spans** (sized by ElChe throughput ratios at pool creation; equal
//! when uncalibrated). A rank's chunks come from the front of its own
//! span via a per-rank cursor, so each rank's upcoming data is
//! deterministic for the whole epoch — the basis for staging it ahead.
//! When a rank exhausts its span while others lag (throughput drift
//! beyond the reservation ratios), it **steals from the tail of the
//! largest-residue span** (reservation truing: the boundary moves, the
//! books stay exact). Tails are therefore the only region whose owner
//! is uncertain — which is why the staging layer prefetches everyone's
//! tails last, as margin.
//!
//! Non-overlap invariant: a span's owner consumes it front-to-back, a
//! thief peels its tail back-to-front; they can meet but never cross,
//! and spans are disjoint by construction.

use std::collections::VecDeque;
use std::time::Instant;

pub struct ChunkPool {
    /// Epoch this pool belongs to. Stored for diagnostics and test access;
    /// the canonical key is the BTreeMap entry in `Coordinator::chunk_pools`.
    #[allow(dead_code)]
    pub epoch: usize,
    pub total_samples: usize,
    /// Per-rank reserved `(start, end)` spans — a partition of
    /// `[0, total_samples)`. `end` moves down when a faster rank steals
    /// the tail (truing).
    spans: Vec<(usize, usize)>,
    /// Per-rank: next unassigned offset within the rank's own span.
    rank_cursor: Vec<usize>,
    /// Per-rank: samples dispatched (sum of all chunk sizes sent).
    pub dispatched: Vec<usize>,
    /// Per-rank: samples completed (from MetricsMsg.samples_processed).
    pub completed: Vec<usize>,
    /// Per-rank: number of chunks sent.
    pub chunks_sent: Vec<usize>,
    /// Per-rank FIFO of dispatched-but-not-completed `(offset, size)`
    /// chunks. Completions consume from the front (workers process
    /// chunks in dispatch order); [`Self::forfeit`] drains the whole
    /// queue back into `reclaimed` when the rank dies.
    outstanding: Vec<VecDeque<(usize, usize)>>,
    /// `(offset, size)` ranges returned to the pool by [`Self::forfeit`]
    /// (a dead rank's never-completed chunks). Served by `take_chunk`
    /// BEFORE the cursor advances, so a dead rank's samples are
    /// re-dispatched to survivors instead of silently dropped.
    reclaimed: VecDeque<(usize, usize)>,
    /// Wall-clock start of this epoch (for EpochMetrics).
    pub epoch_start: Instant,
}

impl ChunkPool {
    /// Pool with equal per-rank spans (uncalibrated / test default).
    pub fn new(epoch: usize, total_samples: usize, world_size: usize) -> Self {
        let base = total_samples / world_size.max(1);
        let mut sizes = vec![base; world_size];
        if let Some(last) = sizes.last_mut() {
            *last += total_samples - base * world_size;
        }
        Self::new_with_spans(epoch, total_samples, &sizes)
    }

    /// Pool with per-rank reservation spans of the given sizes (must sum
    /// to `total_samples`; the caller batch-aligns them). Spans are laid
    /// out contiguously in rank order.
    pub fn new_with_spans(epoch: usize, total_samples: usize, span_sizes: &[usize]) -> Self {
        debug_assert_eq!(span_sizes.iter().sum::<usize>(), total_samples);
        let world_size = span_sizes.len();
        let mut spans = Vec::with_capacity(world_size);
        let mut rank_cursor = Vec::with_capacity(world_size);
        let mut at = 0usize;
        for &s in span_sizes {
            spans.push((at, at + s));
            rank_cursor.push(at);
            at += s;
        }
        ChunkPool {
            epoch,
            total_samples,
            spans,
            rank_cursor,
            dispatched: vec![0; world_size],
            completed: vec![0; world_size],
            chunks_sent: vec![0; world_size],
            outstanding: vec![VecDeque::new(); world_size],
            reclaimed: VecDeque::new(),
            epoch_start: Instant::now(),
        }
    }

    /// Unassigned samples left in a rank's own span.
    pub fn residue(&self, rank: usize) -> usize {
        self.spans[rank].1.saturating_sub(self.rank_cursor[rank])
    }

    /// The rank's remaining reserved stream: `(next_offset, len)` of the
    /// unassigned front of its own span. The deterministic "will train
    /// next" view the staging layer prefetches from.
    #[allow(dead_code)]
    pub fn reservation(&self, rank: usize) -> (usize, usize) {
        (self.rank_cursor[rank], self.residue(rank))
    }

    /// Take the next chunk of `size` samples for `rank`.
    ///
    /// Returns `(offset, actual_size)` or `None` if the pool is exhausted.
    /// Actual size may be smaller than requested (own-span residue, a
    /// reclaimed range, or a donor tail smaller than the ask); a chunk is
    /// always one contiguous slice. Source order:
    /// 1. reclaimed ranges (a dead rank's forfeited chunks) — coverage
    ///    holes must close first;
    /// 2. the front of the rank's own reserved span;
    /// 3. truing: the tail of the largest-residue span (the rank
    ///    out-ran its reservation; the boundary moves).
    pub fn take_chunk(&mut self, size: usize, rank: usize) -> Option<(usize, usize)> {
        if size > 0
            && let Some((off, range_size)) = self.reclaimed.pop_front()
        {
            let actual = size.min(range_size);
            if actual < range_size {
                // Partial take: return the tail of the range for the
                // next caller.
                self.reclaimed
                    .push_front((off + actual, range_size - actual));
            }
            self.dispatched[rank] += actual;
            self.chunks_sent[rank] += 1;
            self.outstanding[rank].push_back((off, actual));
            return Some((off, actual));
        }

        // Own span front.
        let residue = self.residue(rank);
        if residue > 0 {
            let actual = size.min(residue);
            let offset = self.rank_cursor[rank];
            self.rank_cursor[rank] += actual;
            self.dispatched[rank] += actual;
            self.chunks_sent[rank] += 1;
            if actual > 0 {
                self.outstanding[rank].push_back((offset, actual));
            }
            return Some((offset, actual));
        }

        // Truing steal: peel the tail of the largest-residue span. The
        // donor consumes front-to-back, the thief takes back-to-front —
        // they can meet but never cross.
        let donor = (0..self.spans.len())
            .filter(|&r| r != rank)
            .max_by_key(|&r| self.residue(r))
            .filter(|&r| self.residue(r) > 0)?;
        let actual = size.min(self.residue(donor));
        self.spans[donor].1 -= actual;
        let offset = self.spans[donor].1;
        self.dispatched[rank] += actual;
        self.chunks_sent[rank] += 1;
        if actual > 0 {
            self.outstanding[rank].push_back((offset, actual));
        }
        Some((offset, actual))
    }

    /// Roll back the most recent [`Self::take_chunk`] for `rank` — the
    /// transactional escape for a dispatch whose send failed AFTER the
    /// pool was mutated. Without it the taken samples stay
    /// dispatched-but-never-completed: `in_flight` sticks,
    /// `is_epoch_done` never fires, and the epoch wedges permanently on
    /// a single transient write error.
    ///
    /// Only valid for the LAST take by this rank (`(offset, size)` must
    /// match its newest outstanding chunk). When the take came off the
    /// cursor and nothing was taken since, the cursor rewinds; otherwise
    /// the range goes to the front of `reclaimed` for re-dispatch.
    pub fn rollback_take(&mut self, rank: usize, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        // Fast path: the newest outstanding chunk — the only case the
        // serial coordinator produces (a take immediately followed by its
        // failed send's rollback).
        let matched = match self.outstanding[rank].back() {
            Some(&(off, sz)) if off == offset && sz == size => {
                self.outstanding[rank].pop_back();
                true
            }
            _ => {
                // Defensive: a rollback that isn't the newest take should
                // not reach here under the current serial coordinator. The
                // old code asserted-then-silently-returned, which in
                // release LEAKS the taken samples (`in_flight` sticks,
                // `is_epoch_done` never fires, the epoch wedges). Instead
                // find the exact chunk wherever it sits and reclaim THAT;
                // if it isn't outstanding at all there is genuinely nothing
                // to roll back (reclaiming an unknown range would risk
                // double-serving completed samples). Loud in every build.
                match self.outstanding[rank]
                    .iter()
                    .rposition(|&(off, sz)| off == offset && sz == size)
                {
                    Some(pos) => {
                        eprintln!(
                            "flodl ddp: rollback_take(rank {rank}, {offset}, \
                             {size}) was not the newest outstanding chunk — \
                             reclaiming the matched entry (dispatch-ordering \
                             anomaly)"
                        );
                        self.outstanding[rank].remove(pos);
                        true
                    }
                    None => {
                        eprintln!(
                            "flodl ddp: rollback_take(rank {rank}, {offset}, \
                             {size}) matched no outstanding chunk — ignored \
                             (nothing to roll back)"
                        );
                        false
                    }
                }
            }
        };
        if !matched {
            return;
        }
        self.dispatched[rank] = self.dispatched[rank].saturating_sub(size);
        self.chunks_sent[rank] = self.chunks_sent[rank].saturating_sub(1);
        if self.rank_cursor[rank] == offset + size {
            // The take came off the rank's own span front and nothing was
            // taken from it since: rewind the cursor.
            self.rank_cursor[rank] -= size;
        } else {
            // A steal (or an own-span take followed by another): the range
            // goes to reclaimed for re-dispatch — un-stealing a donor tail
            // is not worth the bookkeeping for this rare transactional path.
            self.reclaimed.push_front((offset, size));
        }
    }

    /// Return a dead rank's dispatched-but-never-completed chunks to the
    /// pool so survivors can re-dispatch them, and zero its in-flight
    /// accounting so `is_epoch_done` / the reduce gate stop waiting on a
    /// rank that will never report. Returns the number of reclaimed
    /// samples. Idempotent (a second call finds nothing outstanding).
    pub fn forfeit(&mut self, rank: usize) -> usize {
        let mut reclaimed_total = 0;
        while let Some(range) = self.outstanding[rank].pop_front() {
            reclaimed_total += range.1;
            self.reclaimed.push_back(range);
        }
        self.dispatched[rank] = self.dispatched[rank].saturating_sub(reclaimed_total);
        reclaimed_total
    }

    /// Samples not yet assigned to any rank (every span's residue plus
    /// any forfeited ranges awaiting re-dispatch).
    pub fn remaining(&self) -> usize {
        (0..self.spans.len())
            .map(|r| self.residue(r))
            .sum::<usize>()
            + self.reclaimed.iter().map(|&(_, s)| s).sum::<usize>()
    }

    /// Record that a rank completed processing some samples.
    ///
    /// A completion exceeding the rank's dispatched count is clamped (with
    /// a log): it means the rank was falsely declared dead — its chunks
    /// were forfeited and re-dispatched — and then it reported anyway.
    /// The samples get double-trained (harmless); the books must not
    /// underflow.
    pub fn mark_completed(&mut self, rank: usize, samples: usize) {
        self.completed[rank] += samples;
        if self.completed[rank] > self.dispatched[rank] {
            crate::verbose!(
                "  ddp: rank {rank} completion after forfeit (completed {} > \
                 dispatched {}), clamping",
                self.completed[rank],
                self.dispatched[rank],
            );
            self.completed[rank] = self.dispatched[rank];
        }
        // Consume the rank's outstanding FIFO front-first (chunks complete
        // in dispatch order; a partial credit shrinks the front range).
        let mut left = samples;
        while left > 0 {
            match self.outstanding[rank].front_mut() {
                Some(front) if front.1 <= left => {
                    left -= front.1;
                    self.outstanding[rank].pop_front();
                }
                Some(front) => {
                    front.0 += left;
                    front.1 -= left;
                    left = 0;
                }
                None => break,
            }
        }
    }

    /// Samples dispatched but not yet completed for a given rank.
    pub fn in_flight(&self, rank: usize) -> usize {
        self.dispatched[rank].saturating_sub(self.completed[rank])
    }

    /// The offset ranges NOT yet covered (dispatched-AND-completed by some
    /// rank) at this instant: the unassigned tail `[cursor, total)`, every
    /// in-flight `outstanding` chunk across all ranks, and every `reclaimed`
    /// range awaiting re-dispatch. Coalesced and sorted by offset; covered =
    /// everything else in `[0, total)`.
    ///
    /// In-flight chunks are reported UNCOVERED by design: a chunk dispatched
    /// but not yet completed has not had its gradient folded into the
    /// consensus, so resume must re-dispatch it as first-coverage (not a
    /// repeat). This is the snapshot half of the coverage-granular resume
    /// contract — see [`Self::from_coverage`] and
    /// `docs/design/epoch-tail-allocation.md` (## Async).
    pub fn uncovered_ranges(&self) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (r, &(_, end)) in self.spans.iter().enumerate() {
            if self.rank_cursor[r] < end {
                ranges.push((self.rank_cursor[r], end - self.rank_cursor[r]));
            }
        }
        for q in &self.outstanding {
            ranges.extend(q.iter().copied().filter(|&(_, sz)| sz > 0));
        }
        ranges.extend(self.reclaimed.iter().copied().filter(|&(_, sz)| sz > 0));
        ranges.sort_by_key(|&(off, _)| off);
        // Coalesce contiguous ranges (the tail abutting a reclaimed range, etc.)
        // so the recorded block is minimal; offsets are disjoint by the pool's
        // non-overlap invariant, so a simple adjacent-merge is exact.
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (off, sz) in ranges {
            match out.last_mut() {
                Some(last) if last.0 + last.1 == off => last.1 += sz,
                _ => out.push((off, sz)),
            }
        }
        out
    }

    /// Reconstruct a pool whose only remaining work is `uncovered` — the
    /// ranges recorded by [`Self::uncovered_ranges`] at a checkpoint reduce.
    /// Covered samples are treated as already done; the uncovered ranges are
    /// staged in the `reclaimed` queue, so [`Self::take_chunk`] hands out
    /// exactly the holes (in offset order, splitting as needed) — each once —
    /// and [`Self::is_epoch_done`] fires when they are all completed.
    ///
    /// The resume half of the coverage-granular contract. The cursor is parked
    /// at `total_samples` so no fresh samples are served; only the staged holes
    /// remain. Re-dispatching an in-flight-at-checkpoint range here is
    /// first-coverage, not a repeat (the snapshot recorded it uncovered).
    pub fn from_coverage(
        epoch: usize,
        total_samples: usize,
        world_size: usize,
        uncovered: &[(usize, usize)],
    ) -> Self {
        let mut pool = ChunkPool::new(epoch, total_samples, world_size);
        // The covered region is settled and gone; only the holes remain to
        // dispatch. Empty every span (no fresh samples) and stage the holes
        // for the reclaimed-first `take_chunk` path.
        for r in 0..pool.spans.len() {
            pool.rank_cursor[r] = pool.spans[r].1;
        }
        pool.reclaimed = uncovered
            .iter()
            .copied()
            .filter(|&(_, sz)| sz > 0)
            .collect();
        pool
    }

    /// True when all samples have been dispatched AND all ranks have
    /// reported completion for everything dispatched to them. Forfeited
    /// ranges awaiting re-dispatch count as un-dispatched work.
    pub fn is_epoch_done(&self) -> bool {
        (0..self.spans.len()).all(|r| self.residue(r) == 0)
            && self.reclaimed.is_empty()
            && self
                .dispatched
                .iter()
                .zip(&self.completed)
                .all(|(d, c)| c >= d)
    }

    /// Epoch wall-clock time in milliseconds.
    pub fn epoch_elapsed_ms(&self) -> f64 {
        self.epoch_start.elapsed().as_secs_f64() * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_pool_basic() {
        // Equal spans: rank 0 owns [0,500), rank 1 owns [500,1000).
        let mut pool = ChunkPool::new(0, 1000, 2);
        assert_eq!(pool.remaining(), 1000);
        assert!(!pool.is_epoch_done());

        // Each rank's chunks come from the front of its own span.
        let (off, size) = pool.take_chunk(300, 0).unwrap();
        assert_eq!(off, 0);
        assert_eq!(size, 300);
        assert_eq!(pool.remaining(), 700);

        let (off, size) = pool.take_chunk(200, 1).unwrap();
        assert_eq!(off, 500);
        assert_eq!(size, 200);
        assert_eq!(pool.remaining(), 500);

        // The reservation view: what each rank will train next.
        assert_eq!(pool.reservation(0), (300, 200));
        assert_eq!(pool.reservation(1), (700, 300));

        // Not done yet (nothing completed)
        assert!(!pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_exhaustion() {
        // Spans: [0,50) / [50,100).
        let mut pool = ChunkPool::new(0, 100, 2);

        // Take more than the own-span residue: clamped to the span.
        let (off, size) = pool.take_chunk(80, 0).unwrap();
        assert_eq!((off, size), (0, 50));

        let (off, size) = pool.take_chunk(50, 1).unwrap();
        assert_eq!((off, size), (50, 50));

        // Pool exhausted: nothing left to steal either.
        assert!(pool.take_chunk(10, 0).is_none());
        assert_eq!(pool.remaining(), 0);
    }

    #[test]
    fn chunk_pool_take_steals_largest_residue_tail_when_span_exhausted() {
        // Ratio spans: rank 0 owns [0,70), rank 1 owns [70,100).
        let mut pool = ChunkPool::new_with_spans(0, 100, &[70, 30]);
        assert_eq!(pool.reservation(0), (0, 70));
        assert_eq!(pool.reservation(1), (70, 30));

        // Rank 1 drains its own span, then out-runs it: the next chunk is
        // peeled from the tail of rank 0's span (reservation truing).
        assert_eq!(pool.take_chunk(30, 1).unwrap(), (70, 30));
        assert_eq!(pool.take_chunk(20, 1).unwrap(), (50, 20));
        assert_eq!(
            pool.reservation(0),
            (0, 50),
            "donor span shrank from the tail"
        );

        // Rank 0 still consumes its (reduced) span front-to-back.
        assert_eq!(pool.take_chunk(60, 0).unwrap(), (0, 50));
        // Everything dispatched exactly once.
        assert!(pool.take_chunk(10, 0).is_none());
        assert_eq!(pool.remaining(), 0);
        pool.mark_completed(0, 50);
        pool.mark_completed(1, 50);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_is_epoch_done() {
        let mut pool = ChunkPool::new(0, 100, 2);

        pool.take_chunk(50, 0).unwrap();
        pool.take_chunk(50, 1).unwrap();
        assert!(pool.take_chunk(1, 0).is_none()); // exhausted

        // All dispatched but nothing completed
        assert!(!pool.is_epoch_done());

        // Rank 0 completes
        pool.mark_completed(0, 50);
        assert!(!pool.is_epoch_done()); // rank 1 still pending

        // Rank 1 completes
        pool.mark_completed(1, 50);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_incremental_completion() {
        // Spans: [0,100) / [100,200).
        let mut pool = ChunkPool::new(0, 200, 2);

        // Two chunks per rank, each from its own span.
        assert_eq!(pool.take_chunk(50, 0).unwrap(), (0, 50));
        assert_eq!(pool.take_chunk(50, 1).unwrap(), (100, 50));
        assert_eq!(pool.take_chunk(50, 0).unwrap(), (50, 50));
        assert_eq!(pool.take_chunk(50, 1).unwrap(), (150, 50));
        assert_eq!(pool.remaining(), 0);

        // Complete in stages
        pool.mark_completed(0, 50); // first chunk
        pool.mark_completed(1, 50);
        assert!(!pool.is_epoch_done()); // rank 0 dispatched 100, only 50 done

        pool.mark_completed(0, 50); // second chunk
        pool.mark_completed(1, 50);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_no_overlap() {
        let mut pool = ChunkPool::new(0, 500, 3);
        let mut all_offsets = Vec::new();

        while pool.remaining() > 0 {
            for rank in 0..3 {
                if let Some((off, size)) = pool.take_chunk(60, rank) {
                    // Verify no overlap with previous chunks
                    for &(prev_off, prev_size) in &all_offsets {
                        let prev_end: usize = prev_off + prev_size;
                        let this_end = off + size;
                        assert!(
                            off >= prev_end || this_end <= prev_off,
                            "overlap: ({off}, {size}) vs ({prev_off}, {prev_size})"
                        );
                    }
                    all_offsets.push((off, size));
                }
            }
        }

        // Total coverage = total_samples
        let total: usize = all_offsets.iter().map(|(_, s)| s).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn chunk_pool_epoch_elapsed() {
        let pool = ChunkPool::new(0, 100, 2);
        // Just verify it returns something reasonable (not zero, not huge)
        std::thread::sleep(std::time::Duration::from_millis(5));
        let ms = pool.epoch_elapsed_ms();
        assert!((4.0..1000.0).contains(&ms), "elapsed {ms}ms");
    }

    // -----------------------------------------------------------------------
    // ChunkPool edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_pool_zero_total_samples() {
        let mut pool = ChunkPool::new(0, 0, 2);
        assert_eq!(pool.remaining(), 0);
        assert!(pool.take_chunk(10, 0).is_none());
        // All dispatched (0) == all completed (0), so epoch is trivially done.
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_single_rank() {
        let mut pool = ChunkPool::new(0, 50, 1);
        let (off, size) = pool.take_chunk(50, 0).unwrap();
        assert_eq!((off, size), (0, 50));
        assert_eq!(pool.remaining(), 0);
        assert!(!pool.is_epoch_done());
        pool.mark_completed(0, 50);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_take_chunk_size_zero() {
        let mut pool = ChunkPool::new(0, 100, 2);
        // take_chunk with size=0 should return (cursor, 0) since min(0, remaining)=0
        // Actually, 0.min(100) = 0, cursor doesn't move, dispatched stays 0.
        // But cursor == 0 < total_samples == 100, so it enters the body,
        // actual = 0.min(100-0) = 0. Returns Some((0, 0)).
        let result = pool.take_chunk(0, 0);
        assert_eq!(result, Some((0, 0)));
        // Cursor should not have advanced.
        assert_eq!(pool.remaining(), 100);
    }

    #[test]
    fn chunk_pool_forfeit_reclaims_in_flight() {
        // Spans: [0,50) / [50,100).
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(50, 0).unwrap();
        pool.take_chunk(50, 1).unwrap();
        assert_eq!(pool.remaining(), 0);

        // Rank 1 completes its first chunk... then dies with 0 in flight,
        // while rank 0 dies with its whole chunk in flight.
        pool.mark_completed(1, 50);
        assert_eq!(pool.forfeit(1), 0); // nothing outstanding
        assert_eq!(pool.forfeit(0), 50);

        // The forfeited 50 samples are back in the pool.
        assert_eq!(pool.remaining(), 50);
        assert_eq!(pool.in_flight(0), 0);
        assert!(!pool.is_epoch_done()); // reclaimed work pending

        // Survivor re-takes the exact forfeited range.
        let (off, size) = pool.take_chunk(50, 1).unwrap();
        assert_eq!((off, size), (0, 50));
        pool.mark_completed(1, 50);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_forfeit_split_reclaim() {
        // Spans: [0,50) / [50,100). Rank 0 dies with its whole span in flight.
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(50, 0).unwrap();
        assert_eq!(pool.forfeit(0), 50);

        // Smaller takes split the reclaimed range.
        assert_eq!(pool.take_chunk(25, 1), Some((0, 25)));
        assert_eq!(pool.take_chunk(25, 1), Some((25, 25)));
        // Then back to the survivor's own span for fresh samples.
        assert_eq!(pool.take_chunk(25, 1), Some((50, 25)));
        assert_eq!(pool.remaining(), 25);
    }

    #[test]
    fn chunk_pool_forfeit_multiple_outstanding() {
        // Async-style: two chunks in flight when the rank dies.
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(30, 0).unwrap();
        pool.take_chunk(20, 0).unwrap();
        assert_eq!(pool.in_flight(0), 50);
        assert_eq!(pool.forfeit(0), 50);
        assert_eq!(pool.in_flight(0), 0);
        assert_eq!(pool.remaining(), 100);
        // Ranges come back in dispatch order.
        assert_eq!(pool.take_chunk(30, 1), Some((0, 30)));
        assert_eq!(pool.take_chunk(20, 1), Some((30, 20)));
    }

    #[test]
    fn chunk_pool_rollback_take_rewinds_cursor() {
        let mut pool = ChunkPool::new(0, 100, 2);
        let (off, size) = pool.take_chunk(40, 0).unwrap();
        pool.rollback_take(0, off, size);
        assert_eq!(pool.remaining(), 100);
        assert_eq!(pool.in_flight(0), 0);
        assert_eq!(pool.chunks_sent[0], 0);
        // The rank's next take re-issues the same slice (cursor rewound).
        assert_eq!(pool.take_chunk(40, 0), Some((0, 40)));
    }

    #[test]
    fn chunk_pool_rollback_take_after_other_take_reclaims() {
        let mut pool = ChunkPool::new(0, 100, 2);
        let (off0, size0) = pool.take_chunk(20, 0).unwrap(); // (0,20)
        pool.take_chunk(20, 0).unwrap(); // (20,20) — rank 0's cursor moved on
        pool.rollback_take(0, off0, size0);
        // Can't rewind the rank cursor past a later take; range is
        // reclaimed for re-dispatch (to anyone).
        assert_eq!(pool.remaining(), 100 - 40 + 20);
        assert_eq!(pool.take_chunk(20, 1), Some((0, 20)));
    }

    #[test]
    fn chunk_pool_rollback_take_non_newest_reclaims_matched_entry() {
        // Defensive path: same rank holds two outstanding chunks and the
        // OLDER (non-back) one is rolled back. It must be found, removed,
        // and reclaimed — not silently dropped (which would leak its
        // samples) and not mis-accounted against the newest chunk.
        // Spans: [0,100) / [100,200).
        let mut pool = ChunkPool::new(0, 200, 2);
        let (off0, size0) = pool.take_chunk(40, 0).unwrap(); // rank 0: (0,40)
        pool.take_chunk(30, 0).unwrap(); // rank 0: (40,30) — now the back
        pool.rollback_take(0, off0, size0);
        // Newest chunk still in flight; only the rolled-back one is gone.
        assert_eq!(pool.in_flight(0), 30);
        assert_eq!(pool.chunks_sent[0], 1);
        // The reclaimed range is served before anyone's span.
        assert_eq!(pool.take_chunk(40, 1), Some((0, 40)));
    }

    #[test]
    fn chunk_pool_rollback_take_unknown_chunk_is_noop() {
        // Rolling back a range that was never taken must not underflow the
        // books nor reclaim a phantom range (which would double-serve).
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap(); // (0,40)
        pool.rollback_take(0, 50, 10); // never outstanding
        assert_eq!(pool.in_flight(0), 40);
        assert_eq!(pool.chunks_sent[0], 1);
        assert_eq!(pool.remaining(), 60); // 100 - 40 dispatched, nothing reclaimed
    }

    #[test]
    fn chunk_pool_death_after_take_reclaims_and_completes_exactly_once() {
        // Death-after-take-before-send: a rank takes a chunk, dies before
        // it completes (the send/compute never happened), the range is
        // reclaimed and re-dispatched to a survivor. Every sample must be
        // trained EXACTLY once and the epoch must eventually complete.
        let total = 100;
        let mut pool = ChunkPool::new(0, total, 2);

        let (off, size) = pool.take_chunk(40, 0).unwrap();
        assert_eq!((off, size), (0, 40));
        assert_eq!(pool.forfeit(0), 40); // rank 0 dies, range reclaimed

        // Survivor drains the whole pool (reclaimed range first, then the
        // cursor residue), completing each chunk.
        let mut covered = vec![0u32; total];
        loop {
            match pool.take_chunk(40, 1) {
                Some((o, s)) if s > 0 => {
                    for slot in covered.iter_mut().skip(o).take(s) {
                        *slot += 1;
                    }
                    pool.mark_completed(1, s);
                }
                _ => break,
            }
        }

        assert!(
            covered.iter().all(|&c| c == 1),
            "every sample trained exactly once: {covered:?}",
        );
        assert_eq!(pool.in_flight(0), 0, "dead rank trained nothing");
        assert_eq!(pool.in_flight(1), 0, "survivor's chunks all completed");
        assert!(
            pool.is_epoch_done(),
            "epoch completes after reclaim + drain"
        );
    }

    #[test]
    fn chunk_pool_completion_after_forfeit_clamps() {
        // Spans: [0,50) / [50,100).
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap();
        pool.forfeit(0);
        // Falsely-declared-dead rank reports anyway: books must not
        // underflow and the epoch must still complete.
        pool.mark_completed(0, 40);
        assert_eq!(pool.in_flight(0), 0);
        // Survivor drains: reclaimed range, own span, then the dead
        // rank's span residue via the tail-steal.
        let (off, size) = pool.take_chunk(40, 1).unwrap();
        assert_eq!((off, size), (0, 40));
        pool.mark_completed(1, 40);
        let (off, size) = pool.take_chunk(60, 1).unwrap();
        assert_eq!((off, size), (50, 50));
        pool.mark_completed(1, 50);
        let (off, size) = pool.take_chunk(60, 1).unwrap();
        assert_eq!((off, size), (40, 10));
        pool.mark_completed(1, 10);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn uncovered_ranges_tail_only_on_fresh_pool() {
        let pool = ChunkPool::new(0, 100, 2);
        // Nothing dispatched: the whole pool is the uncovered tail.
        assert_eq!(pool.uncovered_ranges(), vec![(0, 100)]);
    }

    #[test]
    fn uncovered_ranges_excludes_completed_includes_inflight_and_tail() {
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(30, 0).unwrap(); // (0,30) -> rank 0
        pool.take_chunk(20, 1).unwrap(); // (30,20) -> rank 1, in-flight
        pool.mark_completed(0, 30); // rank 0's chunk now COVERED
        // Covered: [0,30). Uncovered: rank 1's in-flight (30,20) + tail [50,100).
        // (30,20) abuts (50,50) -> coalesced to (30,70).
        assert_eq!(pool.uncovered_ranges(), vec![(30, 70)]);
    }

    #[test]
    fn uncovered_ranges_includes_reclaimed() {
        // Spans: [0,50) / [50,100).
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap(); // (0,40)
        pool.take_chunk(20, 1).unwrap(); // (50,20)
        pool.mark_completed(1, 20); // [50,70) covered
        pool.forfeit(0); // (0,40) back to reclaimed (uncovered)
        // Uncovered: reclaimed (0,40) + rank 0 span residue (40,10) —
        // coalesced (0,50) — and rank 1 span residue (70,30). The gap is
        // the covered [50,70).
        assert_eq!(pool.uncovered_ranges(), vec![(0, 50), (70, 30)]);
    }

    #[test]
    fn from_coverage_dispatches_only_holes_each_once() {
        // Snapshot a partially-covered pool, reconstruct, and verify the
        // reconstructed pool serves EXACTLY the uncovered ranges, once.
        // Spans: [0,33) / [33,66) / [66,100).
        let mut orig = ChunkPool::new(0, 100, 3);
        orig.take_chunk(30, 0).unwrap(); // (0,30)
        orig.take_chunk(20, 1).unwrap(); // (33,20)
        orig.take_chunk(10, 2).unwrap(); // (66,10)
        orig.mark_completed(0, 30); // (0,30) covered
        orig.mark_completed(2, 10); // (66,10) covered
        // Uncovered: rank 0 residue (30,3) + rank 1 in-flight (33,20) +
        // rank 1 residue (53,13) — coalesced (30,36) — and rank 2 residue
        // (76,24).
        let uncovered = orig.uncovered_ranges();
        assert_eq!(uncovered, vec![(30, 36), (76, 24)]);

        let mut resumed = ChunkPool::from_coverage(0, 100, 3, &uncovered);
        assert_eq!(resumed.remaining(), 60, "only the holes remain");
        assert!(!resumed.is_epoch_done());

        // Drain it and confirm coverage = exactly the uncovered set, once.
        let mut covered = vec![0u32; 100];
        while resumed.remaining() > 0 {
            for rank in 0..3 {
                if let Some((o, s)) = resumed.take_chunk(15, rank)
                    && s > 0
                {
                    for slot in covered.iter_mut().skip(o).take(s) {
                        *slot += 1;
                    }
                    resumed.mark_completed(rank, s);
                }
            }
        }
        assert!(resumed.is_epoch_done(), "resumed epoch completes");
        // Exactly the holes covered once; the covered-at-snapshot samples never
        // re-served.
        for (i, &c) in covered.iter().enumerate() {
            let in_hole = (30..66).contains(&i) || (76..100).contains(&i);
            assert_eq!(c, u32::from(in_hole), "sample {i}");
        }
    }

    #[test]
    fn from_coverage_empty_holes_is_done() {
        // A pool snapshotted fully-covered (no holes) reconstructs as already
        // done — resume dispatches nothing for that epoch.
        let resumed = ChunkPool::from_coverage(0, 100, 2, &[]);
        assert_eq!(resumed.remaining(), 0);
        assert!(resumed.is_epoch_done());
    }

    #[test]
    fn chunk_pool_in_flight_tracking() {
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap();
        pool.take_chunk(30, 1).unwrap();
        assert_eq!(pool.in_flight(0), 40);
        assert_eq!(pool.in_flight(1), 30);

        pool.mark_completed(0, 20);
        assert_eq!(pool.in_flight(0), 20);
        assert_eq!(pool.in_flight(1), 30);

        pool.mark_completed(0, 20);
        assert_eq!(pool.in_flight(0), 0);
    }
}
