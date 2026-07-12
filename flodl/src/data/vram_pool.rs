//! Device-resident sample pool: the VRAM tier of the sample-keyed
//! staging cascade.
//!
//! Streaming mode uploads every batch to the device; resident mode
//! keeps the whole dataset there. This pool is the slope between those
//! two cliffs: it retains K of N samples on the device and lets batch
//! assembly gather retained rows in place of uploading them, so H2D
//! traffic shrinks by the hit rate. Under per-epoch reshuffle any
//! K-set of samples hits K/N of reads, so admission is fill-until-full
//! with no steady-state eviction — the same argument as the RAM sample
//! cache, one tier up. Sample-keyed, therefore reshuffle-invariant.
//!
//! # Admission: capture at delivery
//!
//! Every sample crosses PCIe at least once per epoch anyway. The pool
//! admits by copying rows device-to-device out of each just-uploaded
//! batch (on the transfer stream, raw pre-augmentation samples), so
//! filling costs zero extra H2D: the first epoch populates the pool as
//! a side effect, later epochs gather hits on device and upload only
//! misses.
//!
//! # Sizing: automatic, honest-probe-gated, conservative
//!
//! VRAM's other tenants are the model (params, grads, optimizer state,
//! activations — unknowable before the first training step) and the
//! prefetch in-flight window (governed, reactive). The pool therefore
//! stays dormant until the post-first-step honest probe (the
//! governor's latch, or the rank worker's explicit signal on
//! coordinator-paced paths), then takes one budget decision from
//! measured free VRAM minus a flow-buffer in-flight reserve
//! ([`FLOW_RESERVE_BATCHES`]) minus a safety margin — with a capacity
//! tier active, prefetch depth is a rate-matcher, not a capacity
//! claim, the same arbitration as the reader ring one tier down. The
//! pool must never be the reason a training step OOMs: on transient
//! data-plane OOM the governor's target halving runs first, slab
//! eviction is the last resort.
//!
//! # Storage: slabs
//!
//! Rows live in slab tensors of [`SLAB_BYTES`] each (a single bulk
//! tensor could not be partially freed; per-sample tensors fragment
//! the allocator). Slabs fill in consumption order and free LIFO under
//! last-resort eviction.

use std::collections::HashMap;

use crate::tensor::{Device, Result, Tensor, TensorOptions};
use super::prefetch::GovernorCtl;

/// Target bytes per slab (all data positions combined). Small enough
/// that last-resort eviction frees useful amounts without giving up
/// the whole tier, large enough that slab count stays a handful.
const SLAB_BYTES: usize = 64 << 20;

/// Safety margin left free on top of the governor's reserve when the
/// budget is decided: allocator variance, eval-time model copies, and
/// anything else the one probe cannot see.
const MARGIN_BYTES: u64 = 512 << 20;

/// In-flight batches reserved for the prefetch pipeline when the pool
/// sizes itself. With a capacity tier active, in-flight depth is a
/// rate-matching flow buffer, not a capacity claim — the same
/// arbitration as the reader ring's `RING_SLOTS_WITH_CACHE` cap one
/// tier down. A pre-pool depth target computed against the whole free
/// VRAM would otherwise starve the pool at its one-shot decision; if
/// the prefetch overfills during the install epoch, the existing OOM
/// backoff halves it and the next sizing probe (which sees pool bytes
/// as used) rights the target for good.
pub(crate) const FLOW_RESERVE_BATCHES: u64 = 16;

/// One allocation unit of the pool: per data position, a device tensor
/// of `[rows, ...sample_dims]`; `used` rows are filled so far.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Slab {
    tensors: Vec<Tensor>,
    used: usize,
}

/// Device-resident, sample-keyed pool. Owned by the prefetch worker
/// thread — single-threaded by construction, no locks anywhere.
pub(crate) struct VramSamplePool {
    /// `false` = permanent pass-through (CPU target, `vram_pool(false)`).
    enabled: bool,
    device: Device,
    /// One-shot budget-decision latch (mirrors the honest resize: one
    /// truthful probe, one decision).
    decided: bool,
    /// Admission ceiling in bytes. `0` = dormant.
    budget: usize,
    /// Bytes allocated in slabs so far.
    bytes: usize,
    /// Rows per slab, fixed at first admission from the sample shape.
    rows_per_slab: usize,
    /// Bytes per row across all data positions.
    row_bytes: usize,
    slabs: Vec<Slab>,
    /// Sample index -> (slab, row).
    slots: HashMap<usize, (u32, u32)>,
    full_logged: bool,
    // Per-epoch telemetry (reset by `epoch_report`).
    hit_rows: usize,
    miss_rows: usize,
    captured_rows: usize,
}

impl VramSamplePool {
    /// A pool that will decide its budget at the first honest probe.
    /// `enabled = false` (or a non-CUDA device) makes it a permanent
    /// pass-through.
    pub(crate) fn new(device: Device, enabled: bool) -> Self {
        VramSamplePool {
            enabled: enabled && device.is_cuda(),
            device,
            decided: false,
            budget: 0,
            bytes: 0,
            rows_per_slab: 0,
            row_bytes: 0,
            slabs: Vec::new(),
            slots: HashMap::new(),
            full_logged: false,
            hit_rows: 0,
            miss_rows: 0,
            captured_rows: 0,
        }
    }

    /// Whether the pool can currently hold or serve rows.
    pub(crate) fn active(&self) -> bool {
        self.budget > 0
    }

    /// One-shot budget decision, gated on the governor's honest
    /// (post-first-step) probe: free VRAM minus a flow-buffer in-flight
    /// reserve minus [`MARGIN_BYTES`]. Called cheaply per batch until
    /// it fires; a host that never trains (probe never fires) never
    /// installs a budget.
    pub(crate) fn maybe_install(&mut self, governor: &GovernorCtl, batch: &[Tensor]) {
        if !self.enabled || self.decided {
            return;
        }
        if !governor
            .honest_resize_done
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let batch_bytes: u64 = batch
            .iter()
            .map(|t| (t.numel() as u64) * t.dtype().element_size() as u64)
            .sum();
        let target = governor
            .target
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1) as u64;
        self.install_with_reserve(target.min(FLOW_RESERVE_BATCHES) * batch_bytes);
    }

    /// One-shot budget decision from an explicit in-flight reserve, for
    /// paths without a governor (coordinator-paced rank workers, where
    /// the caller signals the post-first-step moment itself and knows
    /// its channel depth). Idempotent after the first call.
    pub(crate) fn install_with_reserve(&mut self, reserve_bytes: u64) {
        if !self.enabled || self.decided {
            return;
        }
        self.decided = true;

        // The probe returns (used, total) — used first, not free.
        let Ok((used, total)) =
            crate::tensor::cuda_memory_info_idx(self.device.index() as i32)
        else {
            return; // no probe, no budget: stay dormant
        };
        let free = total.saturating_sub(used);
        let reserve = reserve_bytes + MARGIN_BYTES;
        let budget = free.saturating_sub(reserve);
        if (budget as usize) < SLAB_BYTES {
            crate::verbose!(
                "vram-pool: dormant on {:?} | free {}MB - reserve {}MB leaves no slab",
                self.device,
                free >> 20,
                reserve >> 20,
            );
            return;
        }
        self.budget = budget as usize;
        crate::verbose!(
            "vram-pool: {:?} budget {}MB (free {}MB - in-flight reserve {}MB)",
            self.device,
            budget >> 20,
            free >> 20,
            reserve >> 20,
        );
    }

    /// Split batch positions into pool hits and misses, in caller
    /// order. Positions index into `indices`.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn partition(&mut self, indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
        if !self.active() || self.slots.is_empty() {
            return (Vec::new(), (0..indices.len()).collect());
        }
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for (pos, idx) in indices.iter().enumerate() {
            if self.slots.contains_key(idx) {
                hits.push(pos);
            } else {
                misses.push(pos);
            }
        }
        self.hit_rows += hits.len();
        self.miss_rows += misses.len();
        (hits, misses)
    }

    /// Gather pooled rows for `indices[pos]` (every `pos` must be a
    /// hit), stacked in `positions` order. Device ops: the caller runs
    /// this on the transfer stream so batch delivery events cover it.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn gather(&self, indices: &[usize], positions: &[usize]) -> Result<Vec<Tensor>> {
        // Group requested rows by slab, remembering each row's rank in
        // the caller's order.
        let mut per_slab: HashMap<u32, (Vec<i64>, Vec<i64>)> = HashMap::new();
        for (rank, &pos) in positions.iter().enumerate() {
            let &(slab, row) = self
                .slots
                .get(&indices[pos])
                .expect("gather called on a non-hit index");
            let entry = per_slab.entry(slab).or_default();
            entry.0.push(row as i64);
            entry.1.push(rank as i64);
        }

        let n_positions = self.slabs[0].tensors.len();
        let mut out = Vec::with_capacity(n_positions);
        for p in 0..n_positions {
            // One index_select per touched slab, concatenated, then
            // permuted back to caller order.
            let mut pieces = Vec::with_capacity(per_slab.len());
            let mut ranks = Vec::with_capacity(positions.len());
            for (&slab, (rows, rs)) in &per_slab {
                let rows_t =
                    Tensor::from_i64(rows, &[rows.len() as i64], self.device)?;
                pieces.push(self.slabs[slab as usize].tensors[p].index_select(0, &rows_t)?);
                ranks.extend_from_slice(rs);
            }
            let cat = if pieces.len() == 1 {
                pieces.pop().expect("one piece")
            } else {
                Tensor::cat_many(&pieces.iter().collect::<Vec<_>>(), 0)?
            };
            // `cat` row j holds the caller's rank `ranks[j]`; invert.
            let mut inv = vec![0i64; ranks.len()];
            for (j, &r) in ranks.iter().enumerate() {
                inv[r as usize] = j as i64;
            }
            let inv_t = Tensor::from_i64(&inv, &[inv.len() as i64], self.device)?;
            out.push(cat.index_select(0, &inv_t)?);
        }
        Ok(out)
    }

    /// Admit samples out of a delivered device batch, while the budget
    /// allows: `tensors[p]` row `r` holds sample `sample_indices[r]`.
    /// Rows already pooled are skipped. Device-to-device on the
    /// caller's stream.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn capture(
        &mut self,
        sample_indices: &[usize],
        tensors: &[Tensor],
    ) -> Result<()> {
        if !self.active() {
            return Ok(());
        }
        // New, deduplicated rows only (row positions into `tensors`).
        let mut fresh: Vec<usize> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (row, &idx) in sample_indices.iter().enumerate() {
            if !self.slots.contains_key(&idx) && seen.insert(idx) {
                fresh.push(row);
            }
        }
        if fresh.is_empty() {
            return Ok(());
        }

        if self.rows_per_slab == 0 {
            // First admission: fix the slab geometry from the sample
            // shape (dims beyond the batch dimension).
            let row_bytes: usize = tensors
                .iter()
                .map(|t| {
                    let numel: i64 = t.shape().iter().skip(1).product::<i64>().max(1);
                    numel as usize * t.dtype().element_size()
                })
                .sum();
            self.row_bytes = row_bytes.max(1);
            self.rows_per_slab = (SLAB_BYTES / self.row_bytes).max(1);
        }

        let mut cursor = 0;
        while cursor < fresh.len() {
            // Room in the tail slab, or allocate a new one within budget.
            let space = self
                .slabs
                .last()
                .map(|s| self.rows_per_slab - s.used)
                .unwrap_or(0);
            if space == 0 {
                let slab_bytes = self.rows_per_slab * self.row_bytes;
                if self.bytes + slab_bytes > self.budget || self.slabs.len() >= u32::MAX as usize {
                    if !self.full_logged {
                        self.full_logged = true;
                        crate::verbose!(
                            "vram-pool: {:?} full | {} rows in {} slab(s), {}MB",
                            self.device,
                            self.slots.len(),
                            self.slabs.len(),
                            self.bytes >> 20,
                        );
                    }
                    return Ok(());
                }
                let mut slab_tensors = Vec::with_capacity(tensors.len());
                for t in tensors {
                    let mut shape: Vec<i64> = vec![self.rows_per_slab as i64];
                    shape.extend(t.shape().iter().skip(1));
                    slab_tensors.push(Tensor::empty(
                        &shape,
                        TensorOptions { dtype: t.dtype(), device: self.device },
                    )?);
                }
                self.slabs.push(Slab { tensors: slab_tensors, used: 0 });
                self.bytes += slab_bytes;
                continue;
            }

            let take = space.min(fresh.len() - cursor);
            let chunk = &fresh[cursor..cursor + take];
            let rows: Vec<i64> = chunk.iter().map(|&r| r as i64).collect();
            let rows_t = Tensor::from_i64(&rows, &[rows.len() as i64], self.device)?;
            let slab_id = (self.slabs.len() - 1) as u32;
            let slab = self.slabs.last_mut().expect("tail slab exists");
            for (p, t) in tensors.iter().enumerate() {
                let src = t.index_select(0, &rows_t)?;
                slab.tensors[p]
                    .narrow(0, slab.used as i64, take as i64)?
                    .copy_(&src, true)?;
            }
            for (off, &row) in chunk.iter().enumerate() {
                self.slots
                    .insert(sample_indices[row], (slab_id, (slab.used + off) as u32));
            }
            slab.used += take;
            self.captured_rows += take;
            cursor += take;
        }
        Ok(())
    }

    /// Last-resort relief valve: free the newest slab and forget its
    /// rows. Returns `false` when there is nothing left to give back.
    pub(crate) fn evict_one_slab(&mut self) -> bool {
        let Some(slab) = self.slabs.pop() else {
            return false;
        };
        let slab_id = self.slabs.len() as u32;
        self.slots.retain(|_, &mut (s, _)| s != slab_id);
        self.bytes -= self.rows_per_slab * self.row_bytes;
        // Stop admitting: pressure that reached eviction will not
        // clear by re-filling what was just freed.
        self.budget = self.bytes;
        self.full_logged = false;
        crate::verbose!(
            "vram-pool: {:?} evicted a slab under memory pressure | {} rows retained, budget now {}MB",
            self.device,
            self.slots.len(),
            self.budget >> 20,
        );
        drop(slab);
        crate::tensor::cuda_empty_cache();
        true
    }

    /// Per-epoch telemetry line (verbose); resets the counters.
    pub(crate) fn epoch_report(&mut self) {
        if !self.active() && self.hit_rows + self.miss_rows == 0 {
            return;
        }
        let seen = self.hit_rows + self.miss_rows;
        if seen == 0 {
            return;
        }
        crate::verbose!(
            "vram-pool: {:?} epoch | {}/{} rows served on-device ({}MB H2D saved), {} captured, {} pooled",
            self.device,
            self.hit_rows,
            seen,
            (self.hit_rows * self.row_bytes) >> 20,
            self.captured_rows,
            self.slots.len(),
        );
        self.hit_rows = 0;
        self.miss_rows = 0;
        self.captured_rows = 0;
    }

    #[cfg(test)]
    pub(crate) fn pooled_rows(&self) -> usize {
        self.slots.len()
    }

    #[cfg(test)]
    pub(crate) fn set_budget_for_test(&mut self, bytes: usize) {
        self.enabled = true;
        self.decided = true;
        self.budget = bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DType;

    /// A CPU-device pool with a hand-installed budget: all pool ops are
    /// plain tensor ops, so the logic tests run without CUDA.
    fn test_pool(budget: usize) -> VramSamplePool {
        let mut pool = VramSamplePool::new(Device::CPU, false);
        pool.set_budget_for_test(budget);
        pool
    }

    /// A device "delivered batch": one f32 data position [n, 4] with
    /// row r = [idx, idx, idx, idx], one i64 label position [n].
    fn make_batch(indices: &[usize]) -> Vec<Tensor> {
        let n = indices.len();
        let data: Vec<f32> = indices
            .iter()
            .flat_map(|&i| std::iter::repeat_n(i as f32, 4))
            .collect();
        let labels: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        vec![
            Tensor::from_f32(&data, &[n as i64, 4], Device::CPU).unwrap(),
            Tensor::from_i64(&labels, &[n as i64], Device::CPU).unwrap(),
        ]
    }

    #[test]
    fn capture_then_gather_roundtrip() {
        let mut pool = test_pool(1 << 30);
        let indices = [7usize, 3, 11, 5];
        let batch = make_batch(&indices);
        pool.capture(&indices, &batch).unwrap();
        assert_eq!(pool.pooled_rows(), 4);

        // Gather in a different caller order and check content.
        let want = [11usize, 7, 5];
        let (hits, misses) = pool.partition(&want);
        assert_eq!(hits, vec![0, 1, 2]);
        assert!(misses.is_empty());
        let out = pool.gather(&want, &hits).unwrap();
        assert_eq!(out[0].shape(), &[3, 4]);
        let data = out[0].to_f32_vec().unwrap();
        assert_eq!(&data[0..4], &[11.0; 4]);
        assert_eq!(&data[4..8], &[7.0; 4]);
        assert_eq!(&data[8..12], &[5.0; 4]);
        let labels = out[1].to_i64_vec().unwrap();
        assert_eq!(labels, vec![11, 7, 5]);
    }

    #[test]
    fn partition_splits_hits_and_misses_in_order() {
        let mut pool = test_pool(1 << 30);
        let batch = make_batch(&[1, 2]);
        pool.capture(&[1, 2], &batch).unwrap();

        let (hits, misses) = pool.partition(&[9, 1, 8, 2]);
        assert_eq!(hits, vec![1, 3]);
        assert_eq!(misses, vec![0, 2]);
    }

    #[test]
    fn budget_declines_admissions_and_dedups() {
        // Budget below one slab's bytes: nothing admits.
        let mut pool = test_pool(1);
        let batch = make_batch(&[1, 2]);
        pool.capture(&[1, 2], &batch).unwrap();
        assert_eq!(pool.pooled_rows(), 0);

        // Duplicate rows in one capture admit once.
        let mut pool = test_pool(1 << 30);
        let dup = [4usize, 4, 4];
        let batch = make_batch(&dup);
        pool.capture(&dup, &batch).unwrap();
        assert_eq!(pool.pooled_rows(), 1);
    }

    #[test]
    fn eviction_forgets_newest_slab_and_latches_budget() {
        let mut pool = test_pool(1 << 30);
        // Row bytes: 4*4 + 8 = 24; force tiny slabs via a tiny budget?
        // Slab geometry is SLAB_BYTES-derived, so spill across slabs is
        // impractical here; single-slab eviction must empty the pool.
        let indices: Vec<usize> = (0..10).collect();
        let batch = make_batch(&indices);
        pool.capture(&indices, &batch).unwrap();
        assert_eq!(pool.pooled_rows(), 10);

        assert!(pool.evict_one_slab());
        assert_eq!(pool.pooled_rows(), 0);
        // Budget latched down to retained bytes: re-admission declined.
        pool.capture(&indices, &batch).unwrap();
        assert_eq!(pool.pooled_rows(), 0);
        assert!(!pool.evict_one_slab());
    }

    #[test]
    fn dormant_pool_is_pass_through() {
        let mut pool = VramSamplePool::new(Device::CPU, false);
        let (hits, misses) = pool.partition(&[1, 2, 3]);
        assert!(hits.is_empty());
        assert_eq!(misses, vec![0, 1, 2]);
        let batch = make_batch(&[1, 2, 3]);
        pool.capture(&[1, 2, 3], &batch).unwrap();
        assert_eq!(pool.pooled_rows(), 0);
        assert!(!pool.active());
    }

    #[test]
    fn label_dtype_survives_the_pool() {
        let mut pool = test_pool(1 << 30);
        let batch = make_batch(&[42]);
        pool.capture(&[42], &batch).unwrap();
        let out = pool.gather(&[42], &[0]).unwrap();
        assert_eq!(out[1].dtype(), DType::Int64);
        assert_eq!(out[0].dtype(), DType::Float32);
    }
}
