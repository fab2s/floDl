//! Memory-budget policy — the single home for how flodl prices and
//! bounds every retention/flow tier, host RAM and VRAM alike.
//!
//! Mechanisms live with their tiers (sample cache, reader ring, stager,
//! VRAM pool, prefetch channel); the *policy* — what fraction of which
//! probe, anchored how, priced by what — lives here, once. The solo
//! loader and the DDP worker/stager are both consumers, so their
//! budgets cannot drift apart: before this module the stager hardcoded
//! `available/2` while the loader ran the `ram_max_usage` knob, and the
//! DDP prefetch hardcoded `0.90` while the loader ran `vram_max_usage`
//! — two parallel calibration machines for one machine's memory.

use crate::tensor::{Device, Result, Tensor, TensorOptions};

/// Hard ceiling on any host-RAM share, whatever the knob says: the host
/// runs the OS, the source readers, and everything else too.
pub(crate) const RAM_SHARE_CEILING: f64 = 0.90;

/// The one host-RAM budget law: `r.min(0.90) × (available + held)`.
///
/// `available` is `MemAvailable` at probe time; `held` is what the
/// asking tier already retains. Held bytes are no longer "available"
/// to the probe, so they are added back before taking the share — the
/// cap stays anchored to the total the run started with (fixed point
/// `r·A0`). Both single-sided variants are bugs this definition exists
/// to prevent: skipping the add-back collapses the budget toward
/// `r·A0/(1+r)` as the tier fills (self-starvation), while adding the
/// FULL held bytes back *after* taking the share ratchets it toward
/// all of MemAvailable (the F1 OOM).
pub(crate) fn anchored_ram_budget(available: u64, held_bytes: u64, ram_max_usage: f64) -> u64 {
    let total = available.saturating_add(held_bytes);
    (total as f64 * ram_max_usage.min(RAM_SHARE_CEILING)) as u64
}

/// Sample-cache RAM budget: the anchored share with the reader ring's
/// slice off the top (both consumers draw on the same `ram_max_usage`
/// budget; the ring is a flow buffer and gets its fixed cut first).
pub(crate) fn sample_cache_budget(
    available: u64,
    held_bytes: u64,
    ring_bytes: u64,
    ram_max_usage: f64,
) -> u64 {
    anchored_ram_budget(available, held_bytes, ram_max_usage).saturating_sub(ring_bytes)
}

/// DDP stager RAM budget: this rank's consumption share of the host's
/// anchored budget — same law, same knob, same ceiling as the solo
/// loader. `host_share` is the rank's schedule count over its
/// co-hosted ranks' total (`budget_i ∝ rate_i`: every rank gets the
/// same seconds of lookahead, not the same bytes). `held_bytes` is
/// what this rank's tiers already retain, anchoring the share exactly
/// like the solo cache — without it the budget self-collapses as the
/// tiers fill (the probe sees its own admissions as lost headroom).
pub(crate) fn stager_ram_budget(
    available: u64,
    held_bytes: u64,
    ram_max_usage: f64,
    host_share: f64,
) -> u64 {
    (anchored_ram_budget(available, held_bytes, ram_max_usage) as f64 * host_share) as u64
}

/// Split a stager budget between the pinned tier and the flow window,
/// 3:1 — retention compounds across epochs while flow value saturates
/// after a handful of batches (same arbitration as the reader ring's
/// [`RING_SLOTS_WITH_CACHE`] cap one tier down).
pub(crate) fn split_stager_budget(total: usize) -> (usize, usize) {
    let stream = total / 4;
    (total - stream, stream)
}

/// Reader-ring size when host RAM cannot be measured (non-Linux) or
/// batches cannot be priced: small enough to be safe anywhere, still
/// enough to pipeline reads against transfers and absorb some jitter.
pub(crate) const RING_SLOTS_FALLBACK: usize = 4;

/// Reader-ring ceiling while the sample cache is active. The ring is a
/// flow buffer: its value is jitter absorption, which saturates after
/// a handful of batches, while every byte the retained tier (sample
/// cache) holds pays again on every later epoch. So when both compete
/// for the RAM budget, the ring is capped here and the cache gets the
/// rest.
pub(crate) const RING_SLOTS_WITH_CACHE: usize = 8;

/// Reader-ring capacity (in batches) for the two-stage prefetch
/// pipeline, from the host RAM budget.
///
/// `ram_max_usage` is the fraction of currently **available** host RAM
/// the reader may claim (default 0.50, contrast with VRAM's 0.90: the
/// host runs everything else too). `available` is `MemAvailable` from
/// [`crate::sys::mem_info`]: it already excludes every other process on
/// the box, including permanent fixtures (pinned VM memory, hugepages)
/// that a total-anchored cap would misread as transient pressure. The
/// budget is a slice of what is actually free, priced in batches, and
/// self-adjusts at each `epoch()` probe as the box fills or drains.
///
/// Returns `0` (single-stage pipeline) when the reader stage is
/// disabled (`ram_max_usage <= 0.0`) or the budget cannot fit even one
/// batch. Capped at the epoch's batch count: buffering past the epoch
/// buys nothing until cross-epoch prefetch lands.
pub(crate) fn ring_slots_from_ram(
    per_sample_bytes: usize,
    batch_size: usize,
    ram_max_usage: f64,
    available: Option<u64>,
    epoch_batches: usize,
) -> usize {
    if ram_max_usage <= 0.0 {
        return 0;
    }
    let Some(available) = available else {
        return RING_SLOTS_FALLBACK.min(epoch_batches);
    };
    let batch_bytes = per_sample_bytes.saturating_mul(batch_size) as u64;
    if batch_bytes == 0 {
        return RING_SLOTS_FALLBACK.min(epoch_batches);
    }
    let budget = anchored_ram_budget(available, 0, ram_max_usage);
    (budget / batch_bytes).min(epoch_batches as u64) as usize
}

/// Compute prefetch depth from VRAM usage cap.
///
/// `max_usage` is the fraction of **total** VRAM to use (default 0.90).
/// The prefetch budget is the gap between current usage and the cap,
/// minus `activation_reserve` bytes reserved for forward/backward
/// activation memory and gradients.
///
/// Called at each `epoch()` boundary. By that point the model, optimizer,
/// and any other allocations are done, so current usage is the real baseline.
pub(crate) fn prefetch_depth_from_vram(
    per_sample_bytes: usize,
    batch_size: usize,
    device: Device,
    max_usage: f64,
    activation_reserve: usize,
) -> usize {
    if !device.is_cuda() {
        return 2; // CPU: just double-buffer
    }

    let batch_bytes = per_sample_bytes * batch_size;
    if batch_bytes == 0 {
        return 2;
    }

    let idx = device.index() as i32;
    // The probe returns (used, total) — used first, not free.
    let (used, total) = crate::tensor::cuda_memory_info_idx(idx)
        .unwrap_or((u64::MAX, 0));

    let cap = (total as f64 * max_usage.clamp(0.5, 0.99)) as usize;
    let budget = cap.saturating_sub(used as usize + activation_reserve);

    budget / batch_bytes
}

// ---------------------------------------------------------------------------
// Retention pricing (what one held sample really costs)
// ---------------------------------------------------------------------------

/// Backing-storage slack above which a view is materialized instead of
/// retained: keeping a clone of a view pins the view's WHOLE storage
/// (a 4KB row `select`ed out of a 500MB row-group pins the 500MB), so
/// past this factor a deep copy into owned storage is cheaper than the
/// pinning. Within the factor the view is kept (no copy) and charged
/// its full storage bytes — never its logical size, which is the
/// under-count that lets admission run past the budget.
const VIEW_SLACK_FACTOR: usize = 2;

/// What retaining `rows` will actually charge, without copying
/// anything: the pricing half of [`retain_rows`], for room checks that
/// must run before the bytes exist (the stager's pause-before-fetch).
pub(crate) fn retained_cost_estimate(rows: &[Tensor]) -> usize {
    rows.iter()
        .map(|t| {
            let logical = t.nbytes();
            let storage = t.storage_nbytes();
            if storage > VIEW_SLACK_FACTOR.saturating_mul(logical) {
                logical
            } else {
                storage
            }
        })
        .sum()
}

/// Prepare `rows` for retention and price them honestly. Oversized
/// views (backing storage > [`VIEW_SLACK_FACTOR`] × logical size) are
/// materialized into their own storage and charged their logical
/// bytes; everything else is kept as-is (shallow clone) and charged
/// its full storage bytes. Errors only if a materializing copy fails —
/// callers should decline the admission rather than retain unpriced
/// bytes.
pub(crate) fn retain_rows(rows: &[Tensor]) -> Result<(Vec<Tensor>, usize)> {
    let mut out = Vec::with_capacity(rows.len());
    let mut cost = 0usize;
    for t in rows {
        let logical = t.nbytes();
        let storage = t.storage_nbytes();
        if storage > VIEW_SLACK_FACTOR.saturating_mul(logical) {
            let owned = Tensor::empty(
                &t.shape(),
                TensorOptions {
                    dtype: t.dtype(),
                    device: t.device(),
                },
            )?;
            owned.copy_(t, false)?;
            out.push(owned);
            cost += logical;
        } else {
            out.push(t.clone());
            cost += storage;
        }
    }
    Ok((out, cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchored_budget_is_a_fixed_point_both_ways() {
        // As the tier fills (held grows, available shrinks by the same
        // amount), the budget must not move: no ratchet toward 100%
        // (F1), no collapse toward r·A0/(1+r) (the stager's mirror).
        let a0: u64 = 1000;
        let r = 0.5;
        let b0 = anchored_ram_budget(a0, 0, r);
        for held in [100u64, 400, 500] {
            assert_eq!(anchored_ram_budget(a0 - held, held, r), b0);
        }
    }

    #[test]
    fn ram_share_is_ceilinged() {
        assert_eq!(anchored_ram_budget(1000, 0, 5.0), 900);
        assert_eq!(stager_ram_budget(1000, 0, 5.0, 1.0), 900);
    }

    #[test]
    fn stager_budget_shares_and_anchors() {
        // Half the host's schedule → half the anchored budget.
        assert_eq!(stager_ram_budget(1000, 0, 0.5, 0.5), 250);
        // The stager's own retained bytes anchor its share exactly
        // like the solo cache's.
        assert_eq!(stager_ram_budget(800, 200, 0.5, 0.5), 250);
    }

    #[test]
    fn stager_split_is_three_to_one() {
        assert_eq!(split_stager_budget(100), (75, 25));
        assert_eq!(split_stager_budget(0), (0, 0));
        // Remainder goes to the pinned tier, nothing is lost.
        let (p, s) = split_stager_budget(7);
        assert_eq!(p + s, 7);
    }

    #[test]
    fn retention_prices_views_by_storage() {
        use crate::tensor::Device;
        let base = Tensor::from_f32(&[0.0; 64], &[8, 8], Device::CPU).unwrap();

        // Owner: logical == storage, kept, charged its bytes.
        let (rows, cost) = retain_rows(std::slice::from_ref(&base)).unwrap();
        assert_eq!(cost, base.nbytes());
        assert_eq!(rows[0].storage_nbytes(), base.storage_nbytes());

        // A one-row view of the 8x8 buffer: 32 logical bytes pinning
        // 256 — past the slack factor, so it must be materialized and
        // charged its logical size only.
        let row = base.select(0, 0).unwrap();
        assert_eq!(row.nbytes(), 32);
        assert!(row.storage_nbytes() >= 256);
        assert_eq!(retained_cost_estimate(std::slice::from_ref(&row)), 32);
        let (rows, cost) = retain_rows(std::slice::from_ref(&row)).unwrap();
        assert_eq!(cost, 32);
        assert_eq!(rows[0].storage_nbytes(), 32);
        assert_eq!(rows[0].to_f32_vec().unwrap(), row.to_f32_vec().unwrap());

        // A view within the slack (half the buffer): kept as a view,
        // charged the FULL storage — the honest cost of the pin.
        let half = base.narrow(0, 0, 4).unwrap();
        assert_eq!(half.nbytes(), 128);
        let est = retained_cost_estimate(std::slice::from_ref(&half));
        assert_eq!(est, half.storage_nbytes());
        let (rows, cost) = retain_rows(std::slice::from_ref(&half)).unwrap();
        assert_eq!(cost, half.storage_nbytes());
        assert_eq!(rows[0].storage_nbytes(), base.storage_nbytes());
    }
}
