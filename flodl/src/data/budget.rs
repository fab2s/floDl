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

/// The smallest in-flight window that still overlaps data with compute: one
/// batch under the training step while the next is in transit. Below this the
/// feed is not a pipeline at all — the caller takes the synchronous path and
/// pays fetch + H2D inline on the training thread.
const DOUBLE_BUFFER: usize = 2;

/// Physical VRAM headroom the rate-matcher floor demands per byte it holds.
/// At 256x a 16KB batch pair asks for 8MB free — nothing next to a card that
/// can hold the model at all — while a model with hundred-MB batches never
/// clears the bar and keeps the synchronous fallback it needs.
const FLOOR_FREE_RATIO: usize = 256;

/// Compute prefetch depth from VRAM usage cap.
///
/// `max_usage` is the fraction of **total** VRAM to use (default 0.90).
/// The prefetch budget is the gap between current usage and the cap,
/// minus `activation_reserve` bytes reserved for forward/backward
/// activation memory and gradients.
///
/// Called at each `epoch()` boundary. By that point the model, optimizer,
/// and any other allocations are done, so current usage is the real baseline
/// — which is why callers past their first step pass `activation_reserve = 0`:
/// the caching allocator is already holding those blocks and `used` counts
/// them, so subtracting the peak again would charge it twice.
///
/// A budget too small for one batch does not disable the feed outright; see
/// [`DOUBLE_BUFFER`] for the rate-matcher floor that keeps a tight card
/// pipelined.
pub(crate) fn prefetch_depth_from_vram(
    per_sample_bytes: usize,
    batch_size: usize,
    device: Device,
    max_usage: f64,
    activation_reserve: usize,
) -> usize {
    if !device.is_cuda() {
        return DOUBLE_BUFFER; // CPU: no VRAM to budget, just overlap
    }

    let batch_bytes = per_sample_bytes * batch_size;
    if batch_bytes == 0 {
        return DOUBLE_BUFFER; // unpriceable batch: overlap, claim nothing
    }

    let idx = device.index() as i32;
    // The probe returns (used, total) — used first, not free.
    let (used, total) = crate::tensor::cuda_memory_info_idx(idx)
        .unwrap_or((u64::MAX, 0));

    depth_from_probe(used, total, max_usage, activation_reserve, batch_bytes)
}

/// The sizing policy, split from the probe so it is testable at the exact
/// numbers a rig produced. `used` / `total` are as [`crate::tensor::cuda_memory_info_idx`]
/// reports them (used first, driver-level, counting everything the caching
/// allocator has reserved).
fn depth_from_probe(
    used: u64,
    total: u64,
    max_usage: f64,
    activation_reserve: usize,
    batch_bytes: usize,
) -> usize {
    let cap = (total as f64 * max_usage.clamp(0.5, 0.99)) as usize;
    let budget = cap.saturating_sub(used as usize + activation_reserve);

    let depth = budget / batch_bytes;
    if depth > 0 {
        return depth;
    }

    // Rate-matcher floor. `budget` is a CAPACITY-claim ceiling — what the
    // sample pool and the resident tier may lay claim to. A double-buffered
    // flow window is not a capacity claim; it is the thing that lets data
    // overlap compute at all ("with a capacity tier active, prefetch depth is
    // a rate-matcher, not a capacity claim" — `vram_pool`). Once a model's own
    // footprint sits above the cap, `budget` saturates to zero and every batch
    // takes the synchronous fetch+H2D path — measured at 45% of delivered wall
    // on a GTX 1060 running OLMo-150M, given up to protect 32KB.
    //
    // So the floor is priced against PHYSICAL free rather than the cap, and
    // taken only while it is negligible there: it can never be the reason a
    // training step OOMs, and the governor's OOM halving stays the backstop.
    // A failed probe reports `used = u64::MAX` with `total = 0`, so `free`
    // saturates to 0 and the floor is declined — the safe direction.
    let free = total.saturating_sub(used) as usize;
    let floor_bytes = DOUBLE_BUFFER.saturating_mul(batch_bytes);
    if floor_bytes.saturating_mul(FLOOR_FREE_RATIO) <= free {
        DOUBLE_BUFFER
    } else {
        0
    }
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

    // The numbers a 3-rank olmo/cpu-cadence rig run reported at a plan
    // boundary (2026-07-28), so the regression is pinned to a real card and
    // not to a hypothetical one.
    const PASCAL_USED: u64 = 5592 << 20; // GTX 1060 6GB, OLMo-150M + AdamW
    const PASCAL_TOTAL: u64 = 6_360_465_408; // cudaMemGetInfo total, not nvidia-smi's
    const OLMO_ACTIVATION_PEAK: usize = 1682 << 20;
    const OLMO_BATCH_BYTES: usize = 16 << 10; // seq 256 x batch 4, 2 int64 tensors

    #[test]
    fn a_model_above_the_cap_still_gets_a_double_buffer() {
        // `used` exceeds 0.90 x total on its own here (5592MB vs 5459MB), so
        // the capacity budget is zero however the reserve is set. Refusing the
        // feed over that cost 45% of delivered wall on the rig, to protect
        // 32KB out of 473MB physically free.
        let cap = (PASCAL_TOTAL as f64 * 0.90) as usize;
        assert!(PASCAL_USED as usize > cap, "premise: the model alone is over the cap");

        let depth = depth_from_probe(PASCAL_USED, PASCAL_TOTAL, 0.90, 0, OLMO_BATCH_BYTES);
        assert_eq!(depth, DOUBLE_BUFFER);

        // And the floor holds even when a caller still passes the peak, so a
        // missed call site degrades to "pipelined" rather than "synchronous".
        let with_reserve = depth_from_probe(
            PASCAL_USED, PASCAL_TOTAL, 0.90, OLMO_ACTIVATION_PEAK, OLMO_BATCH_BYTES,
        );
        assert_eq!(with_reserve, DOUBLE_BUFFER);
    }

    #[test]
    fn the_activation_peak_is_what_drove_the_budget_to_zero() {
        // Same card with the model's steady state comfortably under the cap:
        // reserve 0 buys real depth, subtracting the peak a second time takes
        // it away. This is the double-count the DDP worker path was making.
        let used = 3 << 30; // 3GiB resident, ~2.3GiB under the cap
        let honest = depth_from_probe(used, PASCAL_TOTAL, 0.90, 0, OLMO_BATCH_BYTES);
        let double_counted = depth_from_probe(
            used, PASCAL_TOTAL, 0.90, OLMO_ACTIVATION_PEAK, OLMO_BATCH_BYTES,
        );
        assert!(honest > 100_000, "honest probe should afford the whole chunk: {honest}");
        assert!(
            double_counted < honest / 2,
            "charging the peak twice must visibly shrink the budget: {double_counted} vs {honest}",
        );
    }

    #[test]
    fn a_batch_too_large_for_the_headroom_keeps_the_sync_fallback() {
        // The floor is not a blanket "always prefetch": when one batch is a
        // real fraction of what is free, buffering two could be why a step
        // OOMs, and the synchronous path is the correct answer.
        let free = PASCAL_TOTAL - PASCAL_USED; // 473MB
        let fat_batch = (free / 4) as usize;
        let depth = depth_from_probe(PASCAL_USED, PASCAL_TOTAL, 0.90, 0, fat_batch);
        assert_eq!(depth, 0);
    }

    #[test]
    fn a_failed_probe_declines_the_floor() {
        // `cuda_memory_info_idx` failure surfaces as (u64::MAX, 0). Free
        // saturates to 0, so the floor must not be handed out on no data.
        let depth = depth_from_probe(u64::MAX, 0, 0.90, 0, OLMO_BATCH_BYTES);
        assert_eq!(depth, 0);
    }

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
