//! Chunk pool for progressive dispatch.
//!
//! Tracks remaining unassigned samples for one epoch during progressive dispatch.
//! Instead of sending the full partition at epoch start, the coordinator hands
//! out small chunks from this pool. Each `take_chunk` advances a monotonic
//! cursor, guaranteeing non-overlapping slices into the global permutation.

use std::collections::VecDeque;
use std::time::Instant;

pub struct ChunkPool {
    /// Epoch this pool belongs to. Stored for diagnostics and test access;
    /// the canonical key is the BTreeMap entry in `Coordinator::chunk_pools`.
    #[allow(dead_code)]
    pub epoch: usize,
    pub total_samples: usize,
    /// Next unassigned offset into the global permutation.
    pub cursor: usize,
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
    pub fn new(epoch: usize, total_samples: usize, world_size: usize) -> Self {
        ChunkPool {
            epoch,
            total_samples,
            cursor: 0,
            dispatched: vec![0; world_size],
            completed: vec![0; world_size],
            chunks_sent: vec![0; world_size],
            outstanding: vec![VecDeque::new(); world_size],
            reclaimed: VecDeque::new(),
            epoch_start: Instant::now(),
        }
    }

    /// Take the next chunk of `size` samples from the pool.
    ///
    /// Returns `(offset, actual_size)` or `None` if the pool is exhausted.
    /// Actual size may be smaller than requested if near the end.
    /// Reclaimed ranges (a dead rank's forfeited chunks) are served first;
    /// a chunk is always one contiguous slice, so at most one reclaimed
    /// range is consumed per call (split if larger than `size`).
    pub fn take_chunk(&mut self, size: usize, rank: usize) -> Option<(usize, usize)> {
        if size > 0 {
            if let Some((off, range_size)) = self.reclaimed.pop_front() {
                let actual = size.min(range_size);
                if actual < range_size {
                    // Partial take: return the tail of the range for the
                    // next caller.
                    self.reclaimed.push_front((off + actual, range_size - actual));
                }
                self.dispatched[rank] += actual;
                self.chunks_sent[rank] += 1;
                self.outstanding[rank].push_back((off, actual));
                return Some((off, actual));
            }
        }
        if self.cursor >= self.total_samples {
            return None;
        }
        let actual = size.min(self.total_samples - self.cursor);
        let offset = self.cursor;
        self.cursor += actual;
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
        match self.outstanding[rank].back() {
            Some(&(off, sz)) if off == offset && sz == size => {
                self.outstanding[rank].pop_back();
            }
            other => {
                debug_assert!(
                    false,
                    "rollback_take(rank {rank}, {offset}, {size}) does not match \
                     newest outstanding chunk {other:?}"
                );
                return;
            }
        }
        self.dispatched[rank] = self.dispatched[rank].saturating_sub(size);
        self.chunks_sent[rank] = self.chunks_sent[rank].saturating_sub(1);
        if self.cursor == offset + size {
            self.cursor -= size;
        } else {
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

    /// Samples not yet assigned to any rank (cursor residue plus any
    /// forfeited ranges awaiting re-dispatch).
    pub fn remaining(&self) -> usize {
        self.total_samples.saturating_sub(self.cursor)
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

    /// True when all samples have been dispatched AND all ranks have
    /// reported completion for everything dispatched to them. Forfeited
    /// ranges awaiting re-dispatch count as un-dispatched work.
    pub fn is_epoch_done(&self) -> bool {
        self.cursor >= self.total_samples
            && self.reclaimed.is_empty()
            && self.dispatched.iter().zip(&self.completed).all(|(d, c)| c >= d)
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
        let mut pool = ChunkPool::new(0, 1000, 2);
        assert_eq!(pool.remaining(), 1000);
        assert!(!pool.is_epoch_done());

        // Take a chunk for rank 0
        let (off, size) = pool.take_chunk(300, 0).unwrap();
        assert_eq!(off, 0);
        assert_eq!(size, 300);
        assert_eq!(pool.remaining(), 700);

        // Take a chunk for rank 1
        let (off, size) = pool.take_chunk(200, 1).unwrap();
        assert_eq!(off, 300);
        assert_eq!(size, 200);
        assert_eq!(pool.remaining(), 500);

        // Not done yet (nothing completed)
        assert!(!pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_exhaustion() {
        let mut pool = ChunkPool::new(0, 100, 2);

        // Take more than available: clamped
        let (off, size) = pool.take_chunk(80, 0).unwrap();
        assert_eq!((off, size), (0, 80));

        let (off, size) = pool.take_chunk(50, 1).unwrap();
        assert_eq!((off, size), (80, 20)); // only 20 left

        // Pool exhausted
        assert!(pool.take_chunk(10, 0).is_none());
        assert_eq!(pool.remaining(), 0);
    }

    #[test]
    fn chunk_pool_is_epoch_done() {
        let mut pool = ChunkPool::new(0, 100, 2);

        pool.take_chunk(60, 0).unwrap();
        pool.take_chunk(40, 1).unwrap();
        assert!(pool.take_chunk(1, 0).is_none()); // exhausted

        // All dispatched but nothing completed
        assert!(!pool.is_epoch_done());

        // Rank 0 completes
        pool.mark_completed(0, 60);
        assert!(!pool.is_epoch_done()); // rank 1 still pending

        // Rank 1 completes
        pool.mark_completed(1, 40);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_incremental_completion() {
        let mut pool = ChunkPool::new(0, 200, 2);

        // Two chunks for rank 0
        pool.take_chunk(50, 0).unwrap();
        pool.take_chunk(50, 1).unwrap();
        pool.take_chunk(60, 0).unwrap();
        pool.take_chunk(40, 1).unwrap();
        assert_eq!(pool.remaining(), 0);

        // Complete in stages
        pool.mark_completed(0, 50); // first chunk
        pool.mark_completed(1, 50);
        assert!(!pool.is_epoch_done()); // rank 0 dispatched 110, only 50 done

        pool.mark_completed(0, 60); // second chunk
        pool.mark_completed(1, 40);
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
                        assert!(off >= prev_end || this_end <= prev_off,
                            "overlap: ({off}, {size}) vs ({prev_off}, {prev_size})");
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
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap();
        pool.take_chunk(60, 1).unwrap();
        assert_eq!(pool.remaining(), 0);

        // Rank 1 completes its first chunk... then dies with 0 in flight,
        // while rank 0 dies with its whole chunk in flight.
        pool.mark_completed(1, 60);
        assert_eq!(pool.forfeit(1), 0); // nothing outstanding
        assert_eq!(pool.forfeit(0), 40);

        // The forfeited 40 samples are back in the pool.
        assert_eq!(pool.remaining(), 40);
        assert_eq!(pool.in_flight(0), 0);
        assert!(!pool.is_epoch_done()); // reclaimed work pending

        // Survivor re-takes the exact forfeited range.
        let (off, size) = pool.take_chunk(40, 1).unwrap();
        assert_eq!((off, size), (0, 40));
        pool.mark_completed(1, 40);
        assert!(pool.is_epoch_done());
    }

    #[test]
    fn chunk_pool_forfeit_split_reclaim() {
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(60, 0).unwrap();
        assert_eq!(pool.forfeit(0), 60);

        // Smaller takes split the reclaimed range.
        assert_eq!(pool.take_chunk(25, 1), Some((0, 25)));
        assert_eq!(pool.take_chunk(25, 1), Some((25, 25)));
        assert_eq!(pool.take_chunk(25, 1), Some((50, 10))); // range tail
        // Then back to the cursor for fresh samples.
        assert_eq!(pool.take_chunk(25, 1), Some((60, 25)));
        assert_eq!(pool.remaining(), 15);
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
        // The next take re-issues the same slice.
        assert_eq!(pool.take_chunk(40, 1), Some((0, 40)));
    }

    #[test]
    fn chunk_pool_rollback_take_after_other_take_reclaims() {
        let mut pool = ChunkPool::new(0, 100, 2);
        let (off0, size0) = pool.take_chunk(40, 0).unwrap();
        pool.take_chunk(30, 1).unwrap(); // cursor moved past rank 0's chunk
        pool.rollback_take(0, off0, size0);
        // Can't rewind the cursor; range is reclaimed for re-dispatch.
        assert_eq!(pool.remaining(), 30 + 40);
        assert_eq!(pool.take_chunk(40, 1), Some((0, 40)));
    }

    #[test]
    fn chunk_pool_completion_after_forfeit_clamps() {
        let mut pool = ChunkPool::new(0, 100, 2);
        pool.take_chunk(40, 0).unwrap();
        pool.forfeit(0);
        // Falsely-declared-dead rank reports anyway: books must not
        // underflow and the epoch must still complete.
        pool.mark_completed(0, 40);
        assert_eq!(pool.in_flight(0), 0);
        let (off, size) = pool.take_chunk(40, 1).unwrap();
        assert_eq!((off, size), (0, 40));
        pool.mark_completed(1, 40);
        let (off, size) = pool.take_chunk(60, 1).unwrap();
        assert_eq!((off, size), (40, 60));
        pool.mark_completed(1, 60);
        assert!(pool.is_epoch_done());
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
