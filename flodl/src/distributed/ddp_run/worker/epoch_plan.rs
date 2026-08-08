//! Epoch-plan lifecycle: `wait_for_epoch_plan`, `run_epoch_plan`, and `write_checkpoint_bundle`.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::cuda_stream::StreamGuard;
use crate::tensor::{Result, Tensor, TensorError};

use super::super::{ControlMsg, EpochPlan, make_partition, pick_space};
use super::GpuWorker;

/// Per-epoch training cursor, extracted from `run_epoch_plan`'s locals so the
/// batch loop can be driven one primitive at a time
/// (`begin_epoch` -> `next_batch_inner` -> `after_step` -> `end_epoch`).
///
/// The managed tier re-expresses `run_epoch_plan` on these primitives (one
/// code path); the cooperative `Worker` tier wedges the user's forward +
/// backward between `next_batch_inner` and `after_step`. This is a plain
/// value: it borrows nothing from the worker (the prefetch `Receiver` is
/// owned; the background prefetch thread stays owned by `GpuWorker`), so a
/// `next_batch_inner(&mut self, &mut EpochState)` call is borrow-clean and the
/// worker can carry it as `Option<EpochState>` in the cooperative tier.
pub(crate) struct EpochState {
    /// The epoch this chunk belongs to (from the coordinator's `EpochPlan`).
    plan_epoch: usize,
    /// Batches consumed so far this chunk. The loop bound is re-read off the
    /// live `partition.len()` each call, so a mid-epoch `ExtendPartition`
    /// reshard is picked up transparently.
    batch_done: usize,
    total_loss: f64,
    /// Sum of per-batch compute wall (the `train_step` window). Feeds
    /// `share_complete_ms` and the run-level prof split.
    compute_ms_total: f64,
    /// Sum of per-batch data wall (prefetch stall / synchronous fetch).
    data_starve_ms_total: f64,
    /// The most recent batch's data wall, stamped by `next_batch_inner` and
    /// read by `after_step` for the per-batch `report_timing`.
    last_data_ms: f64,
    epoch_start: Instant,
    /// Anchor for the verbose per-chunk timing-breakdown line (prefetch path).
    chunk_diag_start: Instant,
    /// CUDA async prefetch path (vs the synchronous fetch fallback).
    use_prefetch: bool,
    /// Sync path only: the activation peak is still uncalibrated, so measure
    /// it after this chunk's first `train_step`.
    measuring_peak: bool,
    /// Set when a control drain saw `Shutdown` — the caller must skip
    /// `end_epoch` (no coverage report on a shutdown mid-chunk), matching the
    /// historical `return Ok(true)`.
    shutdown: bool,
    /// Prefetch path: the bounded batch channel from `start_distributed_epoch`
    /// (the background prefetch thread fills it). `None` on the sync path.
    batch_rx: Option<mpsc::Receiver<Result<crate::data::prefetch::PrefetchedBatch>>>,
}

impl EpochState {
    /// A no-work chunk (partition shorter than one batch): `next_batch_inner`
    /// yields `None` immediately and `end_epoch` still emits the coordinator's
    /// "done" signal. Matches the historical `num_batches == 0` early return.
    fn empty(plan_epoch: usize) -> Self {
        EpochState {
            plan_epoch,
            batch_done: 0,
            total_loss: 0.0,
            compute_ms_total: 0.0,
            data_starve_ms_total: 0.0,
            last_data_ms: 0.0,
            epoch_start: Instant::now(),
            chunk_diag_start: Instant::now(),
            use_prefetch: false,
            measuring_peak: false,
            shutdown: false,
            batch_rx: None,
        }
    }

    /// Whether a control drain (in `next_batch_inner` while waiting, or in
    /// `after_step`) saw `Shutdown`. The cooperative `Worker` reads this to
    /// decide whether to skip `end_epoch` (no coverage report on a mid-chunk
    /// shutdown) and to surface the shutdown up its own loop.
    pub(crate) fn shutdown(&self) -> bool {
        self.shutdown
    }
}

impl<M: Module> GpuWorker<M> {
    /// Block until the coordinator sends a StartEpoch or Shutdown.
    ///
    /// Handles intermediate control messages (SyncNow, RequestParams, etc.)
    /// to prevent NCCL deadlock while waiting between epochs.
    /// Returns `Some(plan)` for the next epoch, or `None` on Shutdown/disconnect.
    pub fn wait_for_epoch_plan(&mut self) -> Result<Option<EpochPlan>> {
        crate::debug!(
            "  ddp-worker: rank {} waiting for plan (step={})",
            self.rank,
            self.local_step
        );
        let wait_start = Instant::now();
        loop {
            // Check if a plan was queued by dispatch_control (e.g. StartEpoch
            // arrived during Throttle handler). Must be checked each iteration,
            // not just at entry, because dispatch_control may set it mid-loop.
            if let Some(plan) = self.pending_plan.take() {
                let waited = wait_start.elapsed().as_secs_f64() * 1000.0;
                crate::verbose!(
                    "  ddp-dispatch-diag: rank {} waited {:.0}ms (pending plan)",
                    self.rank,
                    waited
                );
                crate::debug!(
                    "  ddp-worker: rank {} got plan (pending) epoch={}",
                    self.rank,
                    plan.epoch
                );
                return Ok(Some(plan));
            }
            match self.control_rx.recv() {
                Ok(ControlMsg::StartEpoch(plan)) => {
                    let waited = wait_start.elapsed().as_secs_f64() * 1000.0;
                    crate::verbose!(
                        "  ddp-dispatch-diag: rank {} waited {:.0}ms for StartEpoch",
                        self.rank,
                        waited
                    );
                    crate::debug!(
                        "  ddp-worker: rank {} got plan epoch={}",
                        self.rank,
                        plan.epoch
                    );
                    return Ok(Some(plan));
                }
                Ok(ControlMsg::Shutdown) => return Ok(None),
                Ok(msg) => {
                    crate::debug!(
                        "  ddp-worker: rank {} wait_for_plan got {:?}",
                        self.rank,
                        match &msg {
                            ControlMsg::SyncNow => "SyncNow",
                            ControlMsg::Throttle => "Throttle",
                            ControlMsg::RequestParams => "RequestParams",
                            ControlMsg::Update(_) => "Update",
                            ControlMsg::SetGlobalStep(_) => "SetGlobalStep",
                            ControlMsg::Checkpoint { .. } => "Checkpoint",
                            ControlMsg::Shutdown => "Shutdown",
                            ControlMsg::StartEpoch(_) => "StartEpoch",
                            ControlMsg::ExtendPartition { .. } => "ExtendPartition",
                            ControlMsg::DeclareDead => "DeclareDead",
                            ControlMsg::NewNcclSession => "NewNcclSession",
                            ControlMsg::RequestNewNcclId => "RequestNewNcclId",
                            ControlMsg::ShutdownWithSave { .. } => "ShutdownWithSave",
                            ControlMsg::ExecuteEvalCallback { .. } => "ExecuteEvalCallback",
                            ControlMsg::SetEpochCallbackRole { .. } => "SetEpochCallbackRole",
                            ControlMsg::EpochAggregated(_) => "EpochAggregated",
                            ControlMsg::EvalBroadcast { .. } => "EvalBroadcast",
                            ControlMsg::SaveConsensusModel { .. } => "SaveConsensusModel",
                            ControlMsg::StageAdvisory { .. } => "StageAdvisory",
                        }
                    );
                    if self.dispatch_control(msg)? {
                        return Ok(None); // Shutdown consumed by handler (e.g. Throttle)
                    }
                }
                Err(_) => return Ok(None), // disconnected
            }
        }
    }

    /// Process one partition (or chunk) from the coordinator's plan.
    ///
    /// Generates sample indices from the plan's offset and size using the
    /// same deterministic shuffle as all other ranks. Reports metrics at
    /// the end so the coordinator can track completion.
    ///
    /// On CUDA, batches are prefetched asynchronously via a background
    /// worker thread with a VRAM-sized buffer (gauge model). On CPU,
    /// batches are loaded synchronously.
    ///
    /// Returns `true` if a Shutdown was received mid-plan.
    ///
    /// Re-expressed on the extracted per-epoch primitives
    /// (`begin_epoch` -> `next_batch_inner` -> `train_step` -> `after_step` ->
    /// `end_epoch`) so the managed tier here and the cooperative `Worker` tier
    /// run one and the same code path — the only difference being who writes
    /// the loop and where the user's forward + backward sits.
    pub fn run_epoch_plan(
        &mut self,
        plan: &EpochPlan,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<bool> {
        let mut st = self.begin_epoch(plan)?;
        while let Some(batch) = self.next_batch_inner(&mut st)? {
            let (loss, ms) = self.train_step(&batch, train_fn)?;
            self.after_step(&mut st, loss, ms)?;
            if st.shutdown {
                return Ok(true); // Shutdown drained in after_step (bottom).
            }
        }
        if st.shutdown {
            return Ok(true); // Shutdown drained in next_batch_inner (top / wait).
        }
        self.end_epoch(&mut st)?;
        Ok(false)
    }

    /// Per-epoch setup: resolve this rank's partition, calibrate the
    /// activation-peak / prefetch-depth / VRAM-pool budget, fire the
    /// `EpochStart` timeline event, and (on the prefetch path) open the batch
    /// channel and submit every batch for async H2D. Returns the epoch cursor.
    ///
    /// A partition shorter than one batch produces an [`EpochState::empty`]:
    /// no setup, no timeline events, `next_batch_inner` yields nothing, and
    /// `end_epoch` still reports the coordinator's "done" signal (the
    /// historical `num_batches == 0` early return).
    pub(crate) fn begin_epoch(&mut self, plan: &EpochPlan) -> Result<EpochState> {
        self.current_epoch = plan.epoch;
        self.partition = make_partition(
            plan.partition_offset,
            plan.partition_size,
            pick_space(self.dataset.len(), self.augment),
            plan.epoch,
            self.epoch_splits,
            self.base_seed,
        );

        let num_batches = self.partition.len() / self.batch_size;
        if num_batches == 0 {
            return Ok(EpochState::empty(plan.epoch));
        }

        // ALL CUDA work must avoid the default stream and device-wide sync.
        // The CUDA default stream implicitly synchronizes with every other
        // stream, and gpu_synchronize waits for ALL streams on the device.
        // If a SyncNow triggered AllReduce on comm_stream (via the other rank)
        // while this rank touches the default stream or calls device sync,
        // it blocks waiting for comm_stream which waits for this rank -> deadlock.
        //
        // Solution: use compute_stream for all ops, sync compute_stream only.
        // The guard is owned (see `StreamGuard`), so it does not borrow `self`
        // and each primitive re-installs its own for exactly its CUDA work —
        // current-stream is compute_stream during every op, as before.
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        // NOTE: gpu_empty_cache() was here to defragment VRAM between chunks,
        // but it internally does a device-wide sync that deadlocks with pending
        // NCCL AllReduce on comm_stream. Removed: the caching allocator handles
        // fragmentation adequately without explicit cache flushes.

        // Update activation peak from the previous chunk's high-water mark.
        // Uses max() so the budget never grows beyond the worst observed peak.
        // Sync compute_stream only (NOT device-wide gpu_synchronize which
        // would block on comm_stream's pending AllReduce -> deadlock).
        if self.device.is_cuda() && self.activation_peak_bytes > 0 {
            let idx = self.device.index() as i32;
            if let Some(ref stream) = self.compute_stream {
                let _ = stream.synchronize();
            }
            if let Ok(peak) = crate::tensor::gpu_peak_active_bytes_idx(idx)
                && let Ok(baseline) = crate::tensor::gpu_active_bytes_idx(idx)
            {
                let overhead = (peak as usize).saturating_sub(baseline as usize);
                let batch_bytes = self.per_sample_bytes * self.batch_size;
                let activation = overhead.saturating_sub(batch_bytes);
                self.activation_peak_bytes = self.activation_peak_bytes.max(activation);
            }
            crate::tensor::gpu_reset_peak_stats_idx(idx);
        }

        // Sync-path activation-peak calibration marker: captured after the
        // recalc above (which is a no-op while the peak is still 0), matching
        // the historical placement at the top of the sync branch.
        let measuring_peak = self.activation_peak_bytes == 0 && self.device.is_cuda();

        // Reader ring: the same two-stage policy as the solo streaming
        // loader (this path only exists on CUDA targets — the constructor
        // gates `prefetch` on it), sized fresh per plan boundary from live
        // MemAvailable and capped at the flow-buffer depth. The cap is
        // unconditional here where the solo loader conditions it on its
        // sample cache: this path always has a retained tier beside the
        // ring (the stager's pinned cache + stream pool), so the ring is a
        // jitter absorber, never a capacity tier. Its (bounded) bytes are
        // seen by the stager's next advisory refresh, which re-probes
        // MemAvailable — the anchored law self-corrects rather than
        // double-claims. `--ram-max-usage 0` keeps the single-stage shape
        // (the sizing returns 0), same kill-switch semantics as solo.
        let ring_slots = if self.prefetch.is_some() {
            crate::data::budget::ring_slots_from_ram(
                self.per_sample_bytes,
                self.batch_size,
                self.ram_max_usage,
                // Corrected for a unified-memory (APU) target: the VRAM
                // pool is carved out of this same DRAM, so the raw probe
                // would let the ring claim memory the device already
                // owns. Same correction the stager applies to its own
                // share.
                crate::sys::mem_info().map(|m| {
                    crate::data::budget::unified_adjusted_available(
                        m.available_bytes,
                        self.device,
                        self.gpu_ram_share,
                    )
                }),
                num_batches,
            )
            .min(crate::data::budget::RING_SLOTS_WITH_CACHE)
        } else {
            0
        };

        // Recalculate prefetch depth at each plan boundary (VRAM may vary).
        // Cap at num_batches: no point buffering more than the chunk contains.
        // Depth 0 means VRAM is too tight for any prefetch buffer.
        //
        // If activation peak hasn't been measured yet, force depth=0 (sync
        // fallback) so the first chunk can calibrate safely.
        let install_pool_budget =
            !self.vram_pool_budget_sent && self.device.is_cuda() && self.activation_peak_bytes > 0;
        let use_prefetch = if let Some(ref mut pw) = self.prefetch {
            if self.activation_peak_bytes == 0 && self.device.is_cuda() {
                pw.set_prefetch_depth(0);
                false
            } else {
                // Reserve 0, NOT `activation_peak_bytes`: this branch only runs
                // once a step has executed (the `== 0` arm above owns the
                // uncalibrated case), so the caching allocator is already
                // holding the activation blocks and the probe's `used` counts
                // them. Passing the measured peak here charged it twice and
                // drove the budget to zero on any card whose model footprint
                // approaches the cap. Same policy as the solo loader's
                // post-honest-resize probe (`loader.rs`, "probe accounts for
                // step memory itself").
                let vram_depth = crate::data::prefetch_depth_from_vram(
                    self.per_sample_bytes,
                    self.batch_size,
                    self.device,
                    self.vram_max_usage,
                    0,
                );
                let mut depth = vram_depth.min(num_batches);
                // On the pool-install chunk the channel collapses to
                // the flow reserve: the pool is about to budget
                // `free − reserve`, and a channel sized against that
                // same free VRAM would claim the bytes twice (transient
                // install-chunk OOM). From the next plan boundary the
                // probe sees pool bytes as used and the depth re-sizes
                // honestly.
                if install_pool_budget {
                    depth = depth.min(crate::data::vram_pool::FLOW_RESERVE_BATCHES as usize);
                }
                pw.set_prefetch_depth(depth);
                // The governor's verdict is otherwise invisible: a depth of 0
                // silently downgrades the feed to the synchronous path, where
                // the whole data cost lands on the training thread instead of
                // overlapping. That reads as "this GPU is slow", not as "its
                // prefetch was switched off", so say which one it is.
                // Probe only when the line will actually be emitted: the
                // sizing call above already spent one `cudaMemGetInfo`, and a
                // diagnostic must not add a second to every chunk boundary.
                let (used, total) = if crate::log::enabled(crate::log::Verbosity::Debug) {
                    crate::tensor::gpu_memory_info_idx(self.device.index() as i32).unwrap_or((0, 0))
                } else {
                    (0, 0)
                };
                crate::debug!(
                    "  ddp-worker: rank {} prefetch depth={} ring={} (vram_depth={} chunk={} \
                     batch={}KB activation_peak={}MB max_usage={:.2} \
                     used={}MB cap={}MB free={}MB){}",
                    self.rank,
                    depth,
                    ring_slots,
                    vram_depth,
                    num_batches,
                    (self.per_sample_bytes * self.batch_size) >> 10,
                    self.activation_peak_bytes >> 20,
                    self.vram_max_usage,
                    used >> 20,
                    ((total as f64 * self.vram_max_usage) as u64) >> 20,
                    total.saturating_sub(used) >> 20,
                    if depth == 0 {
                        " -> SYNC FALLBACK (data cost unoverlapped)"
                    } else {
                        ""
                    },
                );
                depth > 0
            }
        } else {
            false
        };

        // First post-calibration plan boundary: the activation peak is
        // measured, so the VRAM probe is honest — let the device sample
        // pool take its one-shot budget decision, leaving a flow-buffer
        // reserve for the batch channel (in-flight depth is a
        // rate-matcher once a capacity tier is active). The channel
        // depth was collapsed to the reserve above, so both sides of
        // the install agree on who owns the free VRAM.
        if install_pool_budget && let Some(ref pw) = self.prefetch {
            let batch_bytes = (self.per_sample_bytes * self.batch_size) as u64;
            let reserve =
                crate::data::vram_pool::flow_reserve_bytes(pw.prefetch_depth() as u64, batch_bytes);
            crate::debug!(
                "  ddp-worker: rank {} vram pool budget signal (reserve {}MB)",
                self.rank,
                reserve >> 20
            );
            pw.install_vram_pool_budget(reserve);
            self.vram_pool_budget_sent = true;
        }

        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::EpochStart { epoch: plan.epoch });
        }
        let epoch_start = Instant::now();

        // Prefetch path: open the batch channel and submit all batch indices
        // for async H2D now (the background worker fills the channel; a
        // mid-epoch ExtendPartition submits its own extra batches).
        let batch_rx = if use_prefetch {
            let prefetch = self.prefetch.as_ref().unwrap();
            // start_distributed_epoch creates a fresh bounded channel whose
            // capacity equals the prefetch depth (VRAM budget). The prefetch
            // thread fills it; SyncSender blocks when VRAM is full.
            let rx = prefetch.start_distributed_epoch(ring_slots);
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                prefetch.load_batch(self.partition[start..end].to_vec());
            }
            Some(rx)
        } else {
            None
        };

        Ok(EpochState {
            plan_epoch: plan.epoch,
            batch_done: 0,
            total_loss: 0.0,
            compute_ms_total: 0.0,
            data_starve_ms_total: 0.0,
            last_data_ms: 0.0,
            epoch_start,
            chunk_diag_start: Instant::now(),
            use_prefetch,
            measuring_peak,
            shutdown: false,
            batch_rx,
        })
    }

    /// Yield the next device-ready, transformed batch for this chunk, or
    /// `None` at the shard end. Drains control while waiting on the prefetch
    /// path (a blocking `recv` would deadlock a peer's AllReduce), stamps the
    /// batch's data wall into the cursor for `after_step`'s `report_timing`,
    /// and advances the consumed count. On a `Shutdown` seen while waiting it
    /// sets `st.shutdown` and returns `None` (the caller skips `end_epoch`).
    ///
    /// The returned batch is **owned** and borrows nothing from the worker, so
    /// the cooperative tier can run the user's forward + backward against it
    /// while still calling `&mut self` worker methods.
    pub(crate) fn next_batch_inner(&mut self, st: &mut EpochState) -> Result<Option<Vec<Tensor>>> {
        // Loop bound re-read off the live partition each call, so a mid-epoch
        // `ExtendPartition` reshard is consumed before the shard completes.
        if st.batch_done >= self.partition.len() / self.batch_size {
            return Ok(None);
        }

        if st.use_prefetch {
            // Guard installed before the control drain so a SyncNow-driven
            // collective's divergence readout runs on compute_stream, not the
            // default stream (the deadlock hazard the epoch-scoped guard used
            // to cover). Owned: does not borrow self, so handle_control's
            // `&mut self` call below is fine.
            let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);
            // Interleave control message processing with prefetch waiting.
            // SyncNow can arrive at any time; if we block on batch_rx.recv()
            // the peer enters AllReduce waiting for us -> deadlock.
            if self.handle_control()? {
                st.shutdown = true;
                return Ok(None);
            }

            let wait_start = Instant::now();
            // Control work handled while waiting is NOT data starvation: a
            // window's `Update` (H2D writeback), `RequestParams` (full D2H
            // snapshot) or a SyncNow collective landing in this loop is
            // param-plane transport, and booking it as data wait made slow
            // ranks read as I/O-starved (the pascal "58s/epoch data wait").
            // It stays inside `last_data_ms` — the delivered-cost ledger
            // deliberately prices compute + data + transport — and is only
            // subtracted from the starve diagnostic below.
            let mut control_ms = 0.0_f64;
            // Stuck-detector (debug): if we spin here waiting for a
            // prefetched batch that never arrives, dump the worker's
            // state once so the tight-window fold freeze can be
            // pinned (is the worker starved mid-chunk after an Update/
            // StartEpoch landed?). ~3s of consecutive 10ms timeouts.
            let mut stuck_polls: u32 = 0;
            let mut stuck_dumped = false;
            let prefetched = loop {
                let rx = st.batch_rx.as_ref().expect("prefetch path has a batch_rx");
                match rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(batch) => {
                        break batch
                            .map_err(|e| TensorError::new(&format!("prefetch error: {e}")))?;
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let ctl_start = Instant::now();
                        if self.handle_control()? {
                            st.shutdown = true;
                            return Ok(None);
                        }
                        control_ms += ctl_start.elapsed().as_secs_f64() * 1000.0;
                        stuck_polls += 1;
                        if self.prof_enabled && stuck_polls >= 300 && !stuck_dumped {
                            stuck_dumped = true;
                            eprintln!(
                                "[worker-stuck] rank={} STUCK in prefetch recv >{:.0}s | \
                                 batch_done={} target={} epoch={} partition_len={} \
                                 steps_since_avg={} pending_plan={:?}",
                                self.rank,
                                wait_start.elapsed().as_secs_f64(),
                                st.batch_done,
                                self.partition.len() / self.batch_size,
                                st.plan_epoch,
                                self.partition.len(),
                                self.steps_since_avg,
                                self.pending_plan.as_ref().map(|p| (
                                    p.epoch,
                                    p.partition_offset,
                                    p.partition_size
                                )),
                            );
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(TensorError::new("prefetch channel closed"));
                    }
                }
            };
            // Per-batch DATA wall for the delivered feed (prefetch stall).
            // Full wall including control — ElChe schedules on delivered
            // cost (compute + data + transport), so the ledger keeps it.
            st.last_data_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
            // The starve diagnostic excludes it — that time was spent
            // working the param plane, not starving for data.
            st.data_starve_ms_total += (st.last_data_ms - control_ms).max(0.0);

            // Ensure compute stream waits for async H2D copy to finish
            #[cfg(feature = "gpu")]
            if let Some(ref event) = prefetched.ready_event {
                if let Some(ref stream) = self.compute_stream {
                    stream.wait_event(event)?;
                }
            }

            // Cross-stream lifetime pin. The batch's device blocks were
            // allocated on the prefetch worker's copy stream but are
            // consumed on the compute stream and dropped on this thread
            // while compute kernels (backward reads the labels) may
            // still be in flight. The wait_event above orders the work;
            // it does NOT extend the blocks' allocator lifetime — freed,
            // they guard only against the COPY stream and the next
            // upload can reuse and overwrite them mid-read (observed as
            // whole-slab garbage labels → device-side nll_loss assert
            // on fast small-model ranks in free-running CPU modes).
            #[cfg(feature = "gpu")]
            if let Some(ref stream) = self.compute_stream {
                for t in &prefetched.tensors {
                    t.record_stream(stream)?;
                }
            }

            // Delivery transform: after the copy dependency is
            // installed, keyed by the batch's picks.
            #[cfg(feature = "gpu")]
            let transformed = self.transform.is_some();
            let tensors = if let Some(ref f) = self.transform {
                crate::data::apply_transform(
                    f,
                    prefetched.tensors,
                    &prefetched.picks,
                    self.augment,
                    st.plan_epoch,
                    self.base_seed,
                )?
            } else {
                prefetched.tensors
            };
            // Transform outputs are fresh allocations on this thread's
            // current stream — pin them to the compute stream for the
            // same freed-block-reuse reason as the uploaded batch.
            #[cfg(feature = "gpu")]
            if transformed {
                if let Some(ref stream) = self.compute_stream {
                    for t in &tensors {
                        t.record_stream(stream)?;
                    }
                }
            }
            st.batch_done += 1;
            Ok(Some(tensors))
        } else {
            // Sync path: load one batch at a time, move to device if needed.
            // Used for CPU devices, or CUDA when VRAM is too tight for prefetch.
            let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

            let batch_idx = st.batch_done;
            let start = batch_idx * self.batch_size;
            let end = start + self.batch_size;
            // The partition is a PICK stream; fetch by the decoded
            // sample ids, key the transform by the picks. Own the picks so no
            // borrow of `self.partition` is held across the fetch.
            let picks: Vec<usize> = self.partition[start..end].to_vec();
            let samples = crate::data::picks_to_samples(&picks, self.augment);
            let data_start = Instant::now();
            let batch = self.dataset.get_batch(&samples)?;

            let batch: Vec<Tensor> = if self.device.is_cuda() {
                batch
                    .into_iter()
                    .map(|t| t.to_device(self.device))
                    .collect::<Result<Vec<_>>>()?
            } else {
                batch
            };
            let batch: Vec<Tensor> = if let Some(ref f) = self.transform {
                crate::data::apply_transform(
                    f,
                    batch,
                    &picks,
                    self.augment,
                    st.plan_epoch,
                    self.base_seed,
                )?
            } else {
                batch
            };
            // Per-batch DATA wall for the delivered feed (fetch+to-device).
            st.last_data_ms = data_start.elapsed().as_secs_f64() * 1000.0;
            st.data_starve_ms_total += st.last_data_ms;
            st.batch_done += 1;
            Ok(Some(batch))
        }
    }

    /// The framework-owned bookkeeping that follows a training step: sync-path
    /// activation-peak calibration (first batch), the periodic param-norm, the
    /// per-batch `report_timing`, and the bottom-of-loop control drain. Feeds
    /// the compute wall and loss into the cursor.
    ///
    /// `loss` / `ms` come from the step just executed (`train_step` in the
    /// managed tier, the user's forward + backward + `optimizer_step_and_bookkeep`
    /// in the cooperative tier). Runs under its own owned `compute_stream`
    /// guard so the param-norm and any control-driven collective see
    /// current-stream == compute_stream, exactly as the old epoch-scoped guard.
    pub(crate) fn after_step(&mut self, st: &mut EpochState, loss: f64, ms: f64) -> Result<()> {
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);
        st.compute_ms_total += ms;
        st.total_loss += loss;

        // After the first batch (sync path only): measure activation peak from
        // CUDA stats. `peak` and `current` are read at the same point — both
        // include model, optimizer state, and the still-resident batch — so
        // their difference is already the transient forward/backward/step
        // overhead (activations + gradients) net of the batch. This is the
        // reserve that prefetch_depth_from_vram must account for. The batch
        // must not be subtracted again: it cancels between the two terms, and
        // re-subtracting saturated the reserve to 0 whenever activations +
        // gradients fit inside one batch — and 0 is the not-yet-measured
        // sentinel, so the sync fallback and the VRAM-pool gate held for the
        // whole run. `st.batch_done == 1` == the first batch just processed.
        if st.measuring_peak && st.batch_done == 1 {
            if let Some(ref stream) = self.compute_stream {
                let _ = stream.synchronize();
            }
            let idx = self.device.index() as i32;
            if let Ok(peak) = crate::tensor::gpu_peak_active_bytes_idx(idx)
                && let Ok(current) = crate::tensor::gpu_active_bytes_idx(idx)
            {
                let overhead = (peak as usize).saturating_sub(current as usize);
                // Floor a completed measurement to 1 byte so a degenerate
                // reading cannot collide with the sentinel and re-arm
                // calibration every chunk.
                self.activation_peak_bytes = overhead.max(1);
            }
            // Reset for ongoing monitoring in subsequent chunks.
            crate::tensor::gpu_reset_peak_stats_idx(idx);
        }

        let norm = if self.steps_since_avg.is_multiple_of(10) {
            self.compute_param_norm().ok()
        } else {
            None
        };
        let _ = self.report_timing(ms, st.last_data_ms, norm, loss, None);
        if self.handle_control()? {
            st.shutdown = true; // Shutdown
        }
        Ok(())
    }

    /// End-of-chunk accounting: the verbose prefetch timing breakdown, the
    /// run-level prof sums, and `report_epoch` (coverage + `avg_loss`, with the
    /// batch count re-read off the live partition so an `ExtendPartition`
    /// reshard is reflected). An empty chunk still emits the "done" signal.
    pub(crate) fn end_epoch(&mut self, st: &mut EpochState) -> Result<()> {
        if st.use_prefetch {
            let chunk_total_ms = st.chunk_diag_start.elapsed().as_secs_f64() * 1000.0;
            let prefetch_ms = st.data_starve_ms_total;
            let other_ms = chunk_total_ms - prefetch_ms - st.compute_ms_total;
            crate::verbose!(
                "  ddp-worker-diag: rank {} chunk={} batches | total={:.0}ms compute={:.0}ms prefetch_wait={:.0}ms other(sync/ctrl)={:.0}ms",
                self.rank,
                st.batch_done,
                chunk_total_ms,
                st.compute_ms_total,
                prefetch_ms,
                other_ms,
            );
            crate::debug!(
                "  ddp-worker: rank {} epoch {} chunk done ({} batches)",
                self.rank,
                st.plan_epoch,
                st.batch_done
            );
        }

        // Recompute batch count from current partition length so an
        // `ExtendPartition`-driven reshard (cluster-mode dead-rank
        // recovery) is reflected in `avg_loss` and the report.
        let num_batches = self.partition.len() / self.batch_size;
        if num_batches == 0 {
            // Still report so coordinator gets the "done" signal.
            let _ = self.report_epoch(0.0, 0, 0.0, 0.0, 0.0, 0.0);
            return Ok(());
        }

        let epoch_ms = st.epoch_start.elapsed().as_secs_f64() * 1000.0;
        // Honest balancer denominator: time the rank spent on its assigned
        // work (compute + data wait), excluding any post-completion idle
        // waiting at a sync barrier. epoch_ms includes that idle on the
        // fast rank, which inverts the tput signal the balancer reads.
        // share_complete_ms is computed from the rank's own pipeline times
        // (compute_ms_total + data_starve_ms_total), so it tracks the
        // rank's actual capacity, not how long it idles for peers.
        let share_complete_ms = st.compute_ms_total + st.data_starve_ms_total;
        // Instrumentation (gated): accumulate run-level compute/data so
        // the teardown worker-prof can split run_epoch into compute /
        // data / other(ctrl/sync/transport) — the last being what
        // ElChe's share_complete_ms denominator omits.
        if self.prof_enabled {
            self.compute_ms_run_total += st.compute_ms_total;
            self.data_ms_run_total += st.data_starve_ms_total;
        }
        let avg_loss = st.total_loss / num_batches as f64;
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::EpochEnd {
                epoch: st.plan_epoch,
                loss: avg_loss,
                lr: self.optimizer.lr(),
            });
        }
        let _ = self.report_epoch(
            avg_loss,
            num_batches,
            epoch_ms,
            share_complete_ms,
            st.compute_ms_total,
            st.data_starve_ms_total,
        );

        Ok(())
    }

    /// Write the checkpoint bundle for an unrecoverable-failure save.
    ///
    /// Bundle members at `<stem>.{fdl,optim,meta.json}` per
    /// [`crate::distributed::CheckpointBundle`]. The controller-
    /// designated callback rank (carried by `epoch_callback_role`,
    /// set via `ControlMsg::SetEpochCallbackRole`) is the canonical
    /// writer for the model + meta files — all ranks see identical
    /// post-sync params so duplicating across ranks is wasted I/O.
    /// All ranks write their optimizer state; the callback rank uses
    /// the canonical `.optim` filename, others suffix `.r<N>` (per-
    /// rank momentum buffers differ and a future resume API may
    /// choose to average them).
    ///
    /// Falls back to rank 0 as primary if the controller has not yet
    /// pushed a callback role (cold-failure path before the first
    /// epoch transition).
    ///
    /// All save errors are logged + ignored — we'd rather surface a
    /// disk-full or permission error in the logs than deadlock the
    /// cluster on shutdown.
    /// Write this rank's CURRENT model (params + buffers) to `<stem>.fdl` via
    /// [`crate::nn::save_checkpoint_file`]. Shared by the failure-save bundle
    /// (primary rank) and the NCCL `SaveConsensusModel` consensus checkpoint
    /// (elected rank, post-collective). Errors are logged + ignored — a
    /// disk/permission failure should surface in logs, never deadlock the
    /// cluster.
    pub(super) fn write_model_to_fdl(&self, stem: &str) {
        use crate::distributed::CheckpointBundle;
        use crate::distributed::checkpoint_forge::{consensus_buffer_key, consensus_param_key};
        let model_path = CheckpointBundle::model_path(stem);
        // Positional keys (p{i}/b{j}) — NOT the model's own names, which repeat
        // across stacked layers and would collide in the on-disk map. Matches
        // the CPU forge + `load_consensus_checkpoint` convention so any
        // consensus / failure-save bundle reloads positionally.
        let params: Vec<(String, _)> = self
            .model
            .parameters()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (consensus_param_key(i), p))
            .collect();
        let buffers: Vec<(String, _)> = self
            .model
            .buffers()
            .into_iter()
            .enumerate()
            .map(|(j, b)| (consensus_buffer_key(j), b))
            .collect();
        match model_path.to_str() {
            Some(path_str) => {
                if let Err(e) = crate::nn::save_checkpoint_file(path_str, &params, &buffers, None) {
                    eprintln!(
                        "ddp-worker: rank {} model save to {path_str} failed: {e}",
                        self.rank,
                    );
                }
            }
            None => eprintln!(
                "ddp-worker: rank {} model path is not utf-8: {}",
                self.rank,
                model_path.display(),
            ),
        }
    }

    /// Write this rank's replicated outer-optimizer momentum to
    /// `<stem>.outer.fdl` (one tensor per parameter, positional `p{i}`), the
    /// NCCL elected-rank counterpart to the CPU forge's `.outer.fdl`. No-op
    /// when there is no outer optimizer or it is stateless
    /// ([`crate::distributed::OuterAvg`] returns `None`, so no artifact).
    /// Errors are logged + ignored (a disk
    /// failure must never deadlock the cohort), mirroring [`Self::write_model_to_fdl`].
    pub(super) fn write_outer_momentum_to_fdl(&self, stem: &str) {
        use crate::distributed::CheckpointBundle;
        use crate::distributed::checkpoint_forge::consensus_param_key;
        use crate::nn::Parameter;
        let Some(outer) = self.outer_optimizer.as_ref() else {
            return;
        };
        let Some(momentum) = outer.checkpoint_state() else {
            return; // stateless outer optimizer — no artifact
        };
        let outer_path = CheckpointBundle::model_path(stem).with_extension("outer.fdl");
        let params: Vec<(String, Parameter)> = momentum
            .into_iter()
            .enumerate()
            .map(|(i, t)| (consensus_param_key(i), Parameter::new(t, "outer_momentum")))
            .collect();
        match outer_path.to_str() {
            Some(path_str) => {
                if let Err(e) = crate::nn::save_checkpoint_file(path_str, &params, &[], None) {
                    eprintln!(
                        "ddp-worker: rank {} outer-momentum save to {path_str} failed: {e}",
                        self.rank,
                    );
                }
            }
            None => eprintln!(
                "ddp-worker: rank {} outer-momentum path is not utf-8: {}",
                self.rank,
                outer_path.display(),
            ),
        }
    }

    /// Resume this rank's replicated outer-optimizer momentum from
    /// `<stem>.outer.fdl` (positional `p{i}`, shaped by the model's
    /// parameters). No-op when there is no outer optimizer, the variant is
    /// stateless (load ignored), or the sidecar is absent (fresh / OuterAvg
    /// run). Called once per rank at setup so the NCCL outer step resumes from
    /// the saved momentum instead of re-seeding from zero. Replicates: every
    /// rank loads the same file, matching the model's replicated resume.
    pub(crate) fn resume_outer_momentum(&mut self, stem: &str) -> Result<()> {
        use crate::distributed::CheckpointBundle;
        let Some(outer) = self.outer_optimizer.as_mut() else {
            return Ok(());
        };
        let outer_path = CheckpointBundle::model_path(stem).with_extension("outer.fdl");
        if !outer_path.exists() {
            return Ok(()); // fresh run / OuterAvg — no sidecar to load
        }
        let path = outer_path.to_str().ok_or_else(|| {
            crate::tensor::TensorError::new("resume: non-utf8 outer-momentum path")
        })?;
        let momentum = crate::distributed::load_outer_momentum(&self.model, path)?;
        outer.load_checkpoint_state(momentum)?;
        eprintln!(
            "  resume: rank {} loaded outer-optimizer momentum from {path}",
            self.rank,
        );
        Ok(())
    }

    pub(super) fn write_checkpoint_bundle(
        &self,
        stem: &str,
        reason: crate::distributed::SaveReason,
    ) {
        use crate::distributed::CheckpointBundle;

        let primary_rank = self.epoch_callback_role.unwrap_or(0);

        // Primary rank: model file (params + buffers).
        if self.rank == primary_rank {
            self.write_model_to_fdl(stem);
        }

        // All ranks: optimizer state. Primary rank uses the canonical
        // `.optim`; others suffix `.r<N>`.
        let optim_path = CheckpointBundle::optim_path(stem);
        let rank_optim_path = if self.rank == primary_rank {
            optim_path
        } else {
            let mut p = optim_path;
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                let new_name = format!("{name}.r{}", self.rank);
                p.set_file_name(new_name);
            }
            p
        };
        match rank_optim_path.to_str() {
            Some(path_str) => {
                if let Err(e) = self.optimizer.save_state_to(path_str) {
                    eprintln!(
                        "ddp-worker: rank {} optimizer save to {} failed: {}",
                        self.rank, path_str, e,
                    );
                }
            }
            None => eprintln!(
                "ddp-worker: rank {} optim path is not utf-8: {}",
                self.rank,
                rank_optim_path.display(),
            ),
        }

        // Meta JSON is the controller's job (only it has the live
        // ElChe trajectory + cluster-wide epoch/step/sync-round
        // counters). Worker writes only model + per-rank optimizer.

        crate::verbose!(
            "  ddp-worker: rank {} wrote checkpoint bundle to stem {} \
             (reason {:?})",
            self.rank,
            stem,
            reason,
        );
    }
}
