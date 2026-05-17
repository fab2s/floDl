//! Chunk pool for progressive dispatch.
//!
//! Tracks remaining unassigned samples for one epoch during progressive dispatch.
//! Instead of sending the full partition at epoch start, the coordinator hands
//! out small chunks from this pool. Each `take_chunk` advances a monotonic
//! cursor, guaranteeing non-overlapping slices into the global permutation.

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
            epoch_start: Instant::now(),
        }
    }

    /// Take the next chunk of `size` samples from the pool.
    ///
    /// Returns `(offset, actual_size)` or `None` if the pool is exhausted.
    /// Actual size may be smaller than requested if near the end.
    pub fn take_chunk(&mut self, size: usize, rank: usize) -> Option<(usize, usize)> {
        if self.cursor >= self.total_samples {
            return None;
        }
        let actual = size.min(self.total_samples - self.cursor);
        let offset = self.cursor;
        self.cursor += actual;
        self.dispatched[rank] += actual;
        self.chunks_sent[rank] += 1;
        Some((offset, actual))
    }

    /// Samples not yet assigned to any rank.
    pub fn remaining(&self) -> usize {
        self.total_samples.saturating_sub(self.cursor)
    }

    /// Record that a rank completed processing some samples.
    pub fn mark_completed(&mut self, rank: usize, samples: usize) {
        self.completed[rank] += samples;
        debug_assert!(
            self.completed[rank] <= self.dispatched[rank],
            "rank {} completed {} samples but only {} were dispatched",
            rank,
            self.completed[rank],
            self.dispatched[rank],
        );
    }

    /// Samples dispatched but not yet completed for a given rank.
    pub fn in_flight(&self, rank: usize) -> usize {
        self.dispatched[rank].saturating_sub(self.completed[rank])
    }

    /// True when all samples have been dispatched AND all ranks have
    /// reported completion for everything dispatched to them.
    pub fn is_epoch_done(&self) -> bool {
        self.cursor >= self.total_samples
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
