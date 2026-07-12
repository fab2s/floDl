//! Realized-work vocabulary: the mass semantics every plane of the
//! distributed reduce shares.
//!
//! A *mass* is the scalar realized-work weight riding each contribution
//! to a reduce round. The algebra has three moves, and only three:
//!
//! 1. **Pre-scale at the source.** The contributing unit scales its
//!    tensors by its mass and ships the mass atomically with them (the
//!    same frame on the CPU wire; the same fused collective on NCCL).
//! 2. **Sum through folds.** Scaled tensors and masses sum element-wise
//!    through any number of fold tiers (per-host relay today,
//!    intermediate aggregators tomorrow). Summation is associative, so
//!    fold depth never changes the result: no averaging-of-averages.
//! 3. **Divide once at the root.** The final reduce divides the summed
//!    tensors exactly once, by the summed mass of exactly the
//!    contributions it accepted into the round.
//!
//! The load-bearing law: **the sum and its divisor come from the same
//! accepted contributions.** Work that was never accepted (dead rank,
//! lost frame) enters neither, so they cannot disagree, whatever the
//! cohort did between rounds. A fold tier that divided would break the
//! law; only the root divides (see
//! [`crate::distributed::controller`]'s reduce).
//!
//! **Mass is policy-supplied, not definitionally `n^γ`.** What a unit's
//! mass *is* belongs to the caller's policy; this module provides the
//! current policies and the shared rules:
//!
//! - [`gamma_mass`] — params consensus: `n^γ` for `n` optimizer steps
//!   since the last sync, with the idle guard below. `γ = 1.0`
//!   (default) is plain work-weighting; `γ = 0.0` an unweighted
//!   average over movers; `γ = −1.0` per-step-equal.
//! - [`mover_mass`] — buffers consensus: a 0/1 mover indicator (moved
//!   ranks equal-weighted, idle excluded).
//! - [`is_realized`] — the zero-mass round rule: a round whose accepted
//!   mass is zero realized nothing; its summed tensors are meaningless
//!   zeros and receivers keep their local state.
//!
//! **The idle guard is part of the vocabulary, not an implementation
//! detail.** A unit that did no work has zero mass for *any* policy
//! parameter. Left to raw `powf`, IEEE gives `0^0 = 1` (an idle unit
//! voting with full weight at `γ = 0`) and `0^{γ<0} = ∞` (poisoning the
//! cohort mass into NaN). Both backends import [`gamma_mass`] so the
//! guard cannot drift between them.

/// Realized-work mass of one unit's params contribution under the gamma
/// allocation-weighting policy: `n^γ`, where `n` is the unit's optimizer
/// steps since the last sync.
///
/// Idle units (`n <= 0`) have zero mass for any `γ` — see the module
/// docs for why this is guarded here rather than left to `powf`.
/// `γ = 1.0` short-circuits to `n` (the default path stays exact and
/// cheap).
pub(crate) fn gamma_mass(n: f64, gamma: f64) -> f64 {
    if n <= 0.0 {
        0.0
    } else if gamma == 1.0 {
        n
    } else {
        n.powf(gamma)
    }
}

/// Realized-work mass of one unit's buffers contribution: the 0/1 mover
/// indicator. Buffers (BatchNorm running stats and the like) average
/// equal-weighted over the units that moved; idle units are excluded by
/// zero mass, not by leaving the collective.
pub(crate) fn mover_mass(n: f64) -> f64 {
    if n > 0.0 { 1.0 } else { 0.0 }
}

/// The zero-mass round rule: `true` when `mass` represents realized
/// work. A round whose accepted mass is not positive realized nothing —
/// the summed tensors are meaningless zeros and the receiver must keep
/// its local state (and an outer optimizer must not step on it).
pub(crate) fn is_realized(mass: f64) -> bool {
    mass > 0.0
}

#[cfg(test)]
mod tests {
    use super::{gamma_mass, is_realized, mover_mass};

    #[test]
    fn gamma_one_is_plain_work_weighting() {
        assert_eq!(gamma_mass(4.0, 1.0), 4.0);
        assert_eq!(gamma_mass(2.0, 1.0), 2.0);
    }

    #[test]
    fn gamma_zero_is_unweighted_over_movers() {
        // Every mover has mass 1; the cohort mass (summed wherever the
        // fold happens) is the mover count.
        assert_eq!(gamma_mass(4.0, 0.0), 1.0);
        assert_eq!(gamma_mass(2.0, 0.0), 1.0);
    }

    #[test]
    fn gamma_negative_one_is_per_step_equal() {
        assert!((gamma_mass(4.0, -1.0) - 0.25).abs() < 1e-12);
        assert!((gamma_mass(2.0, -1.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn idle_unit_has_zero_mass_for_any_gamma() {
        // The idle guard: raw powf would give 0^0 = 1 (idle unit voting
        // at γ=0) and 0^-1 = ∞ (NaN consensus); the vocabulary gives 0.
        for gamma in [1.0, 0.5, 0.0, -0.5, -1.0] {
            assert_eq!(gamma_mass(0.0, gamma), 0.0, "gamma = {gamma}");
        }
    }

    #[test]
    fn cohort_mass_is_the_sum_of_member_masses() {
        // One vocabulary, two transports: the CPU divisor (masses summed
        // through frames at the root) and the NCCL normalizer (collective
        // sum of per-rank `n^γ`) are the same expression. Idle members
        // contribute nothing to either.
        let counts = [2.0, 4.0, 0.0];
        let gamma = 0.5;
        let cohort: f64 = counts.iter().map(|&n| gamma_mass(n, gamma)).sum();
        assert!((cohort - (2.0f64.sqrt() + 2.0)).abs() < 1e-12);
    }

    #[test]
    fn mover_mass_is_zero_one() {
        assert_eq!(mover_mass(3.0), 1.0);
        assert_eq!(mover_mass(1.0), 1.0);
        assert_eq!(mover_mass(0.0), 0.0);
    }

    #[test]
    fn zero_mass_round_is_not_realized() {
        assert!(!is_realized(0.0));
        assert!(!is_realized(-1.0));
        assert!(is_realized(f64::MIN_POSITIVE));
        assert!(is_realized(6.0));
    }
}
