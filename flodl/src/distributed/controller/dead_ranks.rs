//! Shared dead-rank ledger for the cluster controller / coordinator.

use super::*;

/// Shared dead-rank ledger. Set by the cluster coordinator when it
/// declares a rank dead (stale heartbeat). Read by the controller's
/// reduce loop to skip the rank's contribution in the current and future
/// rounds, and by the coord-side `should_average` /
/// `poll_cpu_averaging` gates to exclude dead ranks from quorum
/// counting.
///
/// Under the per-host relay transport, the controller no longer owns a
/// stream per rank, so there is nothing to shut down to wake it. The
/// reduce loop polls this ledger (its round-wait uses a timeout), so a
/// coord-declared death is observed on the next poll tick; rank death
/// also arrives directly as a relay
/// [`crate::distributed::relay::mux::RelayControlMsg::RankExit`].
#[derive(Debug)]
pub struct DeadRanks {
    flags: Vec<AtomicBool>,
}

impl DeadRanks {
    /// Create a fresh dead-rank ledger sized for `world_size`. All
    /// ranks start alive.
    pub fn new(world_size: usize) -> Arc<Self> {
        Arc::new(Self {
            flags: (0..world_size).map(|_| AtomicBool::new(false)).collect(),
        })
    }

    /// Declare `rank` permanently dead for the rest of this run.
    /// Idempotent flag set. No-op if `rank >= world_size`. The
    /// controller's reduce loop picks this up on its next round-wait
    /// poll tick.
    pub fn declare_dead(&self, rank: usize) {
        if let Some(flag) = self.flags.get(rank) {
            flag.store(true, Ordering::SeqCst);
        }
    }

    /// Check if `rank` is dead.
    pub fn is_dead(&self, rank: usize) -> bool {
        self.flags
            .get(rank)
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// Count of dead ranks.
    pub fn dead_count(&self) -> usize {
        self.flags
            .iter()
            .filter(|f| f.load(Ordering::SeqCst))
            .count()
    }

    /// World size the ledger was sized for.
    pub fn world_size(&self) -> usize {
        self.flags.len()
    }
}
