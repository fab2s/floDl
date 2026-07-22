//! Unrecoverable-failure threshold for cluster training.
//!
//! When the count of dead ranks reaches the configured threshold, the
//! coordinator broadcasts a save-and-shutdown signal to surviving
//! workers so they persist model + optimizer + meta state to disk
//! before exiting (rather than hanging indefinitely waiting for a
//! re-rendezvous that cannot complete).
//!
//! Two threshold flavors:
//!
//! - [`MaxFailureThreshold::Absolute`] — fail after exactly N ranks die.
//!   Best for small clusters where a single rank is a meaningful
//!   fraction (e.g. 4-GPU box, `Absolute(2)` means "fail when half are
//!   gone").
//! - [`MaxFailureThreshold::Percent`] — fail after `fraction * world_size`
//!   ranks die. Best for larger clusters where you want the threshold to
//!   scale with size (e.g. 32-GPU cluster, `Percent(0.2)` triggers at 7
//!   dead).
//!
//! Configured via
//! [`ClusterCoordinatorConfig::max_failure`](crate::distributed::cluster_coordinator::ClusterCoordinatorConfig::max_failure)
//! and surfaced on the top-level cluster builder.

/// Threshold for declaring a cluster run unrecoverable.
///
/// Marked `#[non_exhaustive]` so adding new threshold modes (e.g. a
/// time-window variant) is a non-breaking change for downstream code
/// that pattern-matches.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MaxFailureThreshold {
    /// Fail after exactly N ranks die.
    ///
    /// `Absolute(3)` triggers when the 3rd rank is declared dead
    /// (dead_count >= 3).
    Absolute(usize),
    /// Fail after `fraction * world_size` ranks die (rounded up).
    ///
    /// `Percent(0.2)` on a 10-rank cohort triggers at the 2nd dead
    /// rank; on a 5-rank cohort it triggers at the 1st. Values outside
    /// `[0.0, 1.0]` are clamped at evaluation time.
    Percent(f64),
}

impl MaxFailureThreshold {
    /// Compute the dead-rank count at which this threshold is breached.
    ///
    /// Returns the minimum `dead_count` value for which the threshold
    /// is considered exceeded. Callers check `dead_count >= limit_for(world_size)`.
    ///
    /// `Absolute(n)` returns `n` directly. `Percent(p)` clamps `p` to
    /// `[0.0, 1.0]`, multiplies by `world_size`, and rounds up. A
    /// `Percent(0.0)` threshold yields 0 — any death triggers it; a
    /// `Percent(1.0)` yields `world_size` — only all-dead triggers it.
    pub fn limit_for(&self, world_size: usize) -> usize {
        match self {
            MaxFailureThreshold::Absolute(n) => *n,
            MaxFailureThreshold::Percent(p) => {
                let clamped = p.clamp(0.0, 1.0);
                (clamped * world_size as f64).ceil() as usize
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_limit_returns_n_directly() {
        let t = MaxFailureThreshold::Absolute(3);
        assert_eq!(t.limit_for(10), 3);
        assert_eq!(t.limit_for(4), 3);
        assert_eq!(t.limit_for(2), 3); // can exceed world_size; caller decides
    }

    #[test]
    fn percent_limit_rounds_up() {
        let t = MaxFailureThreshold::Percent(0.2);
        assert_eq!(t.limit_for(10), 2); // 2.0 -> 2
        assert_eq!(t.limit_for(11), 3); // 2.2 -> 3
        assert_eq!(t.limit_for(5), 1); // 1.0 -> 1
        assert_eq!(t.limit_for(3), 1); // 0.6 -> 1
    }

    #[test]
    fn percent_zero_is_any_death_triggers() {
        let t = MaxFailureThreshold::Percent(0.0);
        assert_eq!(t.limit_for(10), 0);
    }

    #[test]
    fn percent_one_is_all_dead_triggers() {
        let t = MaxFailureThreshold::Percent(1.0);
        assert_eq!(t.limit_for(10), 10);
        assert_eq!(t.limit_for(1), 1);
    }

    #[test]
    fn percent_out_of_range_is_clamped() {
        let t_high = MaxFailureThreshold::Percent(2.5);
        assert_eq!(t_high.limit_for(10), 10);
        let t_low = MaxFailureThreshold::Percent(-0.3);
        assert_eq!(t_low.limit_for(10), 0);
    }

    #[test]
    fn percent_handles_zero_world_size() {
        let t = MaxFailureThreshold::Percent(0.5);
        assert_eq!(t.limit_for(0), 0);
    }
}
