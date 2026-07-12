//! Epoch-plan lifecycle: `wait_for_epoch_plan`, `run_epoch_plan`, and `write_checkpoint_bundle`.

use std::time::Instant;

use crate::autograd::Variable;
use crate::distributed::cuda_stream::StreamGuard;
use crate::nn::Module;
use crate::tensor::{Result, Tensor, TensorError};

use super::super::{
    ControlMsg, EpochPlan, make_partition,
};
use super::GpuWorker;

impl<M: Module> GpuWorker<M> {
    /// Block until the coordinator sends a StartEpoch or Shutdown.
    ///
    /// Handles intermediate control messages (SyncNow, RequestParams, etc.)
    /// to prevent NCCL deadlock while waiting between epochs.
    /// Returns `Some(plan)` for the next epoch, or `None` on Shutdown/disconnect.
    pub fn wait_for_epoch_plan(&mut self) -> Result<Option<EpochPlan>> {
        crate::debug!("  ddp-worker: rank {} waiting for plan (step={})", self.rank, self.local_step);
        let wait_start = Instant::now();
        loop {
            // Check if a plan was queued by dispatch_control (e.g. StartEpoch
            // arrived during Throttle handler). Must be checked each iteration,
            // not just at entry, because dispatch_control may set it mid-loop.
            if let Some(plan) = self.pending_plan.take() {
                let waited = wait_start.elapsed().as_secs_f64() * 1000.0;
                crate::verbose!("  ddp-dispatch-diag: rank {} waited {:.0}ms (pending plan)", self.rank, waited);
                crate::debug!("  ddp-worker: rank {} got plan (pending) epoch={}", self.rank, plan.epoch);
                return Ok(Some(plan));
            }
            match self.control_rx.recv() {
                Ok(ControlMsg::StartEpoch(plan)) => {
                    let waited = wait_start.elapsed().as_secs_f64() * 1000.0;
                    crate::verbose!("  ddp-dispatch-diag: rank {} waited {:.0}ms for StartEpoch", self.rank, waited);
                    crate::debug!("  ddp-worker: rank {} got plan epoch={}", self.rank, plan.epoch);
                    return Ok(Some(plan));
                }
                Ok(ControlMsg::Shutdown) => return Ok(None),
                Ok(msg) => {
                    crate::debug!("  ddp-worker: rank {} wait_for_plan got {:?}", self.rank,
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
    pub fn run_epoch_plan(
        &mut self,
        plan: &EpochPlan,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<bool> {
        self.current_epoch = plan.epoch;
        self.partition = make_partition(
            plan.partition_offset, plan.partition_size,
            self.dataset.len(), plan.epoch, self.base_seed,
        );

        let num_batches = self.partition.len() / self.batch_size;
        if num_batches == 0 {
            // Still report so coordinator gets the "done" signal.
            let _ = self.report_epoch(0.0, 0, 0.0, 0.0, 0.0, 0.0);
            return Ok(false);
        }

        // ALL CUDA work must avoid the default stream and device-wide sync.
        // The CUDA default stream implicitly synchronizes with every other
        // stream, and cuda_synchronize waits for ALL streams on the device.
        // If a SyncNow triggered AllReduce on comm_stream (via the other rank)
        // while this rank touches the default stream or calls device sync,
        // it blocks waiting for comm_stream which waits for this rank -> deadlock.
        //
        // Solution: use compute_stream for all ops, sync compute_stream only.
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        // NOTE: cuda_empty_cache() was here to defragment VRAM between chunks,
        // but it internally does a device-wide sync that deadlocks with pending
        // NCCL AllReduce on comm_stream. Removed: the caching allocator handles
        // fragmentation adequately without explicit cache flushes.

        // Update activation peak from the previous chunk's high-water mark.
        // Uses max() so the budget never grows beyond the worst observed peak.
        // Sync compute_stream only (NOT device-wide cuda_synchronize which
        // would block on comm_stream's pending AllReduce -> deadlock).
        if self.device.is_cuda() && self.activation_peak_bytes > 0 {
            let idx = self.device.index() as i32;
            if let Some(ref stream) = self.compute_stream {
                let _ = stream.synchronize();
            }
            if let Ok(peak) = crate::tensor::cuda_peak_active_bytes_idx(idx) {
                if let Ok(baseline) = crate::tensor::cuda_active_bytes_idx(idx) {
                    let overhead = (peak as usize).saturating_sub(baseline as usize);
                    let batch_bytes = self.per_sample_bytes * self.batch_size;
                    let activation = overhead.saturating_sub(batch_bytes);
                    self.activation_peak_bytes = self.activation_peak_bytes.max(activation);
                }
            }
            crate::tensor::cuda_reset_peak_stats_idx(idx);
        }

        // Recalculate prefetch depth at each plan boundary (VRAM may vary).
        // Cap at num_batches: no point buffering more than the chunk contains.
        // Depth 0 means VRAM is too tight for any prefetch buffer.
        //
        // If activation peak hasn't been measured yet, force depth=0 (sync
        // fallback) so the first chunk can calibrate safely.
        let use_prefetch = if let Some(ref mut pw) = self.prefetch {
            if self.activation_peak_bytes == 0 && self.device.is_cuda() {
                pw.set_prefetch_depth(0);
                false
            } else {
                let vram_depth = crate::data::prefetch_depth_from_vram(
                    self.per_sample_bytes, self.batch_size, self.device, 0.90,
                    self.activation_peak_bytes,
                );
                let depth = vram_depth.min(num_batches);
                pw.set_prefetch_depth(depth);
                depth > 0
            }
        } else {
            false
        };

        // First post-calibration plan boundary: the activation peak is
        // measured, so the VRAM probe is honest — let the device sample
        // pool take its one-shot budget decision, leaving a flow-buffer
        // reserve for the batch channel (in-flight depth is a
        // rate-matcher once a capacity tier is active).
        if !self.vram_pool_budget_sent
            && self.device.is_cuda()
            && self.activation_peak_bytes > 0
        {
            if let Some(ref pw) = self.prefetch {
                let batch_bytes = (self.per_sample_bytes * self.batch_size) as u64;
                let flow = (pw.prefetch_depth() as u64)
                    .min(crate::data::vram_pool::FLOW_RESERVE_BATCHES);
                crate::debug!(
                    "  ddp-worker: rank {} vram pool budget signal (reserve {}MB)",
                    self.rank, (flow * batch_bytes) >> 20
                );
                pw.install_vram_pool_budget(flow * batch_bytes);
                self.vram_pool_budget_sent = true;
            }
        }

        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::EpochStart { epoch: plan.epoch });
        }
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        // Per-chunk timing accumulators populated by both prefetch and sync
        // paths. Read at chunk end to populate MetricsMsg fields and feed
        // the balancer with an honest tput signal.
        let mut compute_ms_total = 0.0_f64;
        let mut data_starve_ms_total = 0.0_f64;

        if use_prefetch {
            // CUDA async path: prefetch with VRAM gauge.
            let prefetch = self.prefetch.as_ref().unwrap();
            // start_distributed_epoch creates a fresh bounded channel whose
            // capacity equals the prefetch depth (VRAM budget). The prefetch
            // thread fills it; SyncSender blocks when VRAM is full.
            let batch_rx = prefetch.start_distributed_epoch();

            // Submit all batch indices for async H2D transfer
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                prefetch.load_batch(self.partition[start..end].to_vec());
            }

            // Consume prefetched batches as they become ready. Loop
            // bound is re-evaluated each iteration off
            // `self.partition.len()` so a mid-epoch
            // `ControlMsg::ExtendPartition` (cluster-mode reshard
            // after a rank dies) injects extra batches and they get
            // processed before the epoch completes. The
            // `ExtendPartition` arm also submits the new batches to
            // the prefetch worker's load queue, so the background
            // worker has work to feed this consumer.
            let mut batch_done = 0usize;
            let chunk_diag_start = Instant::now();
            let mut prefetch_wait_diag = std::time::Duration::ZERO;
            let mut compute_ms_diag = 0.0_f64;
            while batch_done < self.partition.len() / self.batch_size {
                // Interleave control message processing with prefetch waiting.
                // SyncNow can arrive at any time; if we block on batch_rx.recv()
                // the peer enters AllReduce waiting for us -> deadlock.
                // Use recv_timeout to periodically check for control messages.
                if self.handle_control()? {
                    return Ok(true);
                }
                let wait_start = Instant::now();
                // Stuck-detector (debug): if we spin here waiting for a
                // prefetched batch that never arrives, dump the worker's
                // state once so the tight-window fold freeze can be
                // pinned (is the worker starved mid-chunk after an Update/
                // StartEpoch landed?). ~3s of consecutive 10ms timeouts.
                let mut stuck_polls: u32 = 0;
                let mut stuck_dumped = false;
                let prefetched = loop {
                    match batch_rx.recv_timeout(std::time::Duration::from_millis(10)) {
                        Ok(batch) => break batch
                            .map_err(|e| TensorError::new(&format!("prefetch error: {e}")))?,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if self.handle_control()? {
                                return Ok(true);
                            }
                            stuck_polls += 1;
                            if self.prof_enabled && stuck_polls >= 300 && !stuck_dumped {
                                stuck_dumped = true;
                                eprintln!(
                                    "[worker-stuck] rank={} STUCK in prefetch recv >{:.0}s | \
                                     batch_done={} target={} epoch={} partition_len={} \
                                     steps_since_avg={} pending_plan={:?}",
                                    self.rank,
                                    wait_start.elapsed().as_secs_f64(),
                                    batch_done,
                                    self.partition.len() / self.batch_size,
                                    plan.epoch,
                                    self.partition.len(),
                                    self.steps_since_avg,
                                    self.pending_plan.as_ref().map(|p| (p.epoch, p.partition_offset, p.partition_size)),
                                );
                            }
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return Err(TensorError::new("prefetch channel closed"));
                        }
                    }
                };
                let batch_data = wait_start.elapsed();
                prefetch_wait_diag += batch_data;
                // Per-batch DATA wall for the delivered feed (prefetch stall).
                let data_ms_batch = batch_data.as_secs_f64() * 1000.0;

                // Ensure compute stream waits for async H2D copy to finish
                #[cfg(feature = "cuda")]
                if let Some(ref event) = prefetched.ready_event {
                    if let Some(ref stream) = self.compute_stream {
                        stream.wait_event(event)?;
                    }
                }

                let (loss, ms) = self.train_step(&prefetched.tensors, train_fn)?;
                compute_ms_diag += ms;
                batch_done += 1;
                total_loss += loss;
                let norm = if self.steps_since_avg % 10 == 0 {
                    self.compute_param_norm().ok()
                } else {
                    None
                };
                let _ = self.report_timing(ms, data_ms_batch, norm, loss, None);
                if self.handle_control()? {
                    return Ok(true); // Shutdown
                }
            }
            let chunk_total_ms = chunk_diag_start.elapsed().as_secs_f64() * 1000.0;
            let prefetch_ms = prefetch_wait_diag.as_secs_f64() * 1000.0;
            let other_ms = chunk_total_ms - prefetch_ms - compute_ms_diag;
            crate::verbose!(
                "  ddp-worker-diag: rank {} chunk={} batches | total={:.0}ms compute={:.0}ms prefetch_wait={:.0}ms other(sync/ctrl)={:.0}ms",
                self.rank, batch_done, chunk_total_ms, compute_ms_diag, prefetch_ms, other_ms,
            );
            compute_ms_total = compute_ms_diag;
            data_starve_ms_total = prefetch_ms;
            crate::debug!("  ddp-worker: rank {} epoch {} chunk done ({} batches)", self.rank, plan.epoch, batch_done);
        } else {
            // Sync path: load one batch at a time, move to device if needed.
            // Used for CPU devices, or CUDA when VRAM is too tight for prefetch.
            //
            // The loop bound is re-evaluated each iteration off
            // `self.partition.len()` so a mid-epoch
            // `ControlMsg::ExtendPartition` (cluster-mode reshard
            // after a rank dies) injects extra batches and they get
            // processed before the epoch completes — same contract as
            // the prefetch branch above.
            let measuring_peak = self.activation_peak_bytes == 0 && self.device.is_cuda();

            let mut batch_idx: usize = 0;
            while batch_idx < self.partition.len() / self.batch_size {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let data_start = Instant::now();
                let batch = self.dataset.get_batch(indices)?;

                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch.into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };
                // Per-batch DATA wall for the delivered feed (fetch+to-device).
                let data_ms_batch = data_start.elapsed().as_secs_f64() * 1000.0;
                data_starve_ms_total += data_ms_batch;

                let (loss, ms) = self.train_step(&batch, train_fn)?;
                compute_ms_total += ms;
                total_loss += loss;

                // After first batch: measure activation peak from CUDA stats.
                // The peak includes model + batch + activations + gradients.
                // Subtract baseline (model/optimizer/NCCL) and one batch to
                // isolate the activation + gradient overhead. This is the
                // reserve that prefetch_depth_from_vram must account for.
                if measuring_peak && batch_idx == 0 {
                    if let Some(ref stream) = self.compute_stream {
                        let _ = stream.synchronize();
                    }
                    let idx = self.device.index() as i32;
                    if let Ok(peak) = crate::tensor::cuda_peak_active_bytes_idx(idx) {
                        if let Ok(current) = crate::tensor::cuda_active_bytes_idx(idx) {
                            let overhead = (peak as usize).saturating_sub(current as usize);
                            let batch_bytes = self.per_sample_bytes * self.batch_size;
                            self.activation_peak_bytes = overhead.saturating_sub(batch_bytes);
                        }
                    }
                    // Reset for ongoing monitoring in subsequent chunks.
                    crate::tensor::cuda_reset_peak_stats_idx(idx);
                }

                let norm = if self.steps_since_avg % 10 == 0 {
                    self.compute_param_norm().ok()
                } else {
                    None
                };
                let _ = self.report_timing(ms, data_ms_batch, norm, loss, None);
                if self.handle_control()? {
                    return Ok(true); // Shutdown
                }
                batch_idx += 1;
            }
        }

        let epoch_ms = epoch_start.elapsed().as_secs_f64() * 1000.0;
        // Honest balancer denominator: time the rank spent on its assigned
        // work (compute + data wait), excluding any post-completion idle
        // waiting at a sync barrier. epoch_ms includes that idle on the
        // fast rank, which inverts the tput signal the balancer reads.
        // share_complete_ms is computed from the rank's own pipeline times
        // (compute_ms_total + data_starve_ms_total), so it tracks the
        // rank's actual capacity, not how long it idles for peers.
        let share_complete_ms = compute_ms_total + data_starve_ms_total;
        // Instrumentation (gated): accumulate run-level compute/data so
        // the teardown worker-prof can split run_epoch into compute /
        // data / other(ctrl/sync/transport) — the last being what
        // ElChe's share_complete_ms denominator omits.
        if self.prof_enabled {
            self.compute_ms_run_total += compute_ms_total;
            self.data_ms_run_total += data_starve_ms_total;
        }
        // Recompute batch count from current partition length so an
        // `ExtendPartition`-driven reshard (cluster-mode dead-rank
        // recovery) is reflected in `avg_loss` and the report.
        let num_batches = self.partition.len() / self.batch_size;
        let avg_loss = total_loss / num_batches as f64;
        if let Some(ref tl) = self.timeline {
            tl.event(crate::monitor::EventKind::EpochEnd {
                epoch: plan.epoch,
                loss: avg_loss,
                lr: self.optimizer.lr(),
            });
        }
        let _ = self.report_epoch(
            avg_loss, num_batches, epoch_ms,
            share_complete_ms, compute_ms_total, data_starve_ms_total,
        );

        Ok(false)
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
                if let Err(e) =
                    crate::nn::save_checkpoint_file(path_str, &params, &buffers, None)
                {
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
