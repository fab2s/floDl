//! Min / mean / max of a quantity observed repeatedly over an interval.
//!
//! A gauge sampled far faster than it is published has to be summarised, and
//! the summary must not lie about what happened between publications. Keeping
//! the *latest* sample is the tempting default and it is the wrong one: it
//! reports whatever the value happened to be at publication time, so a GPU
//! pegged at 100% for a minute renders as a single spike (or nothing at all)
//! depending on where the tick landed.
//!
//! The honest summary of an interval is its envelope. `mean` says what the
//! interval was like on average, `max` says what it reached, and `min` says
//! whether it ever dropped. None of the three can be reconstructed from the
//! others, and none of them can be reconstructed from a latest-wins sample.
//!
//! This mirrors the decimation rule the record `/history` design already
//! settled on — *min/max envelope, not mean* — because a mean alone hides
//! exactly the excursions a reader is looking for.
//!
//! # Absent is not zero
//!
//! [`EnvelopeAcc::take`] returns `None` when nothing was observed. An interval
//! with no samples is *unmeasured*, not measured-as-zero, and must stay absent
//! all the way through the roll-up — averaging a fabricated 0 into a parent
//! silently drags every ancestor down.
//!
//! # Reuse
//!
//! Deliberately knows nothing about resources, records, or timing. Anything
//! sampled more often than it is reported can accumulate into one of these:
//! per-window resource gauges, ranged history decimation, and per-node timings
//! aggregated across repeated graph blocks are all the same shape.
//!
//! ```
//! use flodl::monitor::envelope::EnvelopeAcc;
//!
//! let mut acc = EnvelopeAcc::default();
//! assert!(acc.take().is_none()); // nothing observed yet -> absent, not 0
//!
//! for util in [10.0, 100.0, 100.0, 10.0] {
//!     acc.push(util);
//! }
//! let env = acc.take().expect("four samples observed");
//! assert_eq!(env.min, 10.0);
//! assert_eq!(env.max, 100.0);
//! assert_eq!(env.mean, 55.0);
//!
//! // `take` drains: the next interval starts empty.
//! assert!(acc.take().is_none());
//! ```

/// The min, mean and max a quantity took over one interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    /// Lowest value observed.
    pub min: f64,
    /// Arithmetic mean over the observations (each sample weighted equally —
    /// the sampler's cadence is what makes that a time-average).
    pub mean: f64,
    /// Highest value observed.
    pub max: f64,
}

/// Accumulates observations into an [`Envelope`], draining on [`Self::take`].
///
/// One accumulator serves one (quantity, consumer) pair. Two consumers reading
/// on different cadences need two accumulators — draining is destructive, so a
/// shared one would let whichever consumer publishes first blank the interval
/// for the other.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvelopeAcc {
    min: f64,
    max: f64,
    sum: f64,
    n: u64,
}

impl EnvelopeAcc {
    /// Record one observation.
    ///
    /// Non-finite values (NaN / ±inf) are ignored rather than poisoning the
    /// envelope: one bad reading from a probe would otherwise make `mean` NaN
    /// for the whole interval, and NaN propagates silently through the tree
    /// roll-up where a missing sample would have been visibly absent.
    pub fn push(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        if self.n == 0 {
            self.min = v;
            self.max = v;
        } else {
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
        }
        self.sum += v;
        self.n += 1;
    }

    /// Record an observation that may be absent; `None` is a no-op.
    ///
    /// Convenience for the common `Option<f64>` sample field, so callers do
    /// not each re-derive "absent means skip, not zero".
    pub fn push_opt(&mut self, v: Option<f64>) {
        if let Some(v) = v {
            self.push(v);
        }
    }

    /// Observations recorded since the last drain.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Take the interval's envelope and reset, or `None` if nothing was
    /// observed.
    pub fn take(&mut self) -> Option<Envelope> {
        if self.n == 0 {
            return None;
        }
        let env = Envelope {
            min: self.min,
            mean: self.sum / self.n as f64,
            max: self.max,
        };
        *self = Self::default();
        Some(env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unobserved_interval_is_absent_not_zero() {
        let mut acc = EnvelopeAcc::default();
        assert_eq!(acc.take(), None);
        assert_eq!(acc.count(), 0);
    }

    #[test]
    fn the_envelope_keeps_what_a_mean_alone_would_hide() {
        // The rig case: a GPU pegged for most of the interval, sampled a few
        // times while idle. A latest-wins sample reports whichever came last;
        // the mean alone hides that it ever reached 100.
        let mut acc = EnvelopeAcc::default();
        for v in [0.0, 100.0, 100.0, 100.0, 0.0] {
            acc.push(v);
        }
        let env = acc.take().unwrap();
        assert_eq!(env.max, 100.0, "the excursion survives");
        assert_eq!(env.min, 0.0);
        assert_eq!(env.mean, 60.0);
    }

    #[test]
    fn take_drains_so_intervals_do_not_bleed_into_each_other() {
        let mut acc = EnvelopeAcc::default();
        acc.push(50.0);
        assert_eq!(acc.take().unwrap().mean, 50.0);
        assert_eq!(acc.take(), None, "second interval starts empty");
        acc.push(10.0);
        let env = acc.take().unwrap();
        assert_eq!(
            (env.min, env.mean, env.max),
            (10.0, 10.0, 10.0),
            "no contribution from the drained interval"
        );
    }

    #[test]
    fn a_single_observation_is_a_degenerate_envelope() {
        let mut acc = EnvelopeAcc::default();
        acc.push(42.0);
        assert_eq!(
            acc.take(),
            Some(Envelope {
                min: 42.0,
                mean: 42.0,
                max: 42.0
            })
        );
    }

    #[test]
    fn absent_observations_are_skipped_not_counted() {
        let mut acc = EnvelopeAcc::default();
        acc.push_opt(None);
        acc.push_opt(Some(20.0));
        acc.push_opt(None);
        acc.push_opt(Some(40.0));
        let env = acc.take().unwrap();
        assert_eq!(env.mean, 30.0, "Nones must not count as zeros in the mean");
        assert_eq!(env.min, 20.0);
    }

    #[test]
    fn non_finite_readings_cannot_poison_the_interval() {
        // A NaN would otherwise make `mean` NaN and propagate silently through
        // the tree roll-up, where an absent sample would have been visible.
        let mut acc = EnvelopeAcc::default();
        acc.push(10.0);
        acc.push(f64::NAN);
        acc.push(f64::INFINITY);
        acc.push(30.0);
        let env = acc.take().unwrap();
        assert!(
            env.min.is_finite() && env.mean.is_finite() && env.max.is_finite(),
            "no field may inherit NaN/inf from a bad reading",
        );
        assert_eq!(env.mean, 20.0, "only the two finite readings count");
        assert_eq!(env.max, 30.0);
    }

    #[test]
    fn negative_values_are_ordered_correctly() {
        let mut acc = EnvelopeAcc::default();
        for v in [-5.0, -1.0, -10.0] {
            acc.push(v);
        }
        let env = acc.take().unwrap();
        assert_eq!(env.min, -10.0);
        assert_eq!(env.max, -1.0);
    }
}
