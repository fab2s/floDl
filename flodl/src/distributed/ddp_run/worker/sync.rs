//! Parameter snapshot / load and NCCL sync paths + per-step train_step.

use std::time::{Duration, Instant};

use crate::autograd::{NoGradGuard, Variable};
use crate::distributed::cuda_stream::StreamGuard;
use crate::distributed::nccl::ReduceOp;
use crate::nn::Module;
use crate::tensor::{Device, Result, Tensor, TensorError, TensorOptions};

use super::super::{
    AveragedParams, ParamSnapshot,
};
use super::GpuWorker;

/// Allocate a pinned (page-locked) CPU staging tensor matching `t`'s shape
/// and dtype. Pinned memory is required for true async D2H copies
/// (`cudaMemcpyAsync` from pageable host memory silently falls back to a
/// synchronous staged bounce copy). Allocated once per param/buffer and
/// reused every reduce window — see `GpuWorker::snapshot_pinned_params`.
fn pinned_like(t: &Tensor) -> Result<Tensor> {
    let opts = TensorOptions { dtype: t.dtype(), device: Device::CPU };
    Tensor::empty(&t.shape(), opts)?.pin_memory()
}

impl<M: Module> GpuWorker<M> {
    /// Extract current parameter values as a [`ParamSnapshot`].
    ///
    /// Tensors are copied to CPU so that the coordinator's compute thread
    /// never needs CUDA access (avoiding slow CUDA context init on the
    /// compute thread, which can deadlock with `drain_avg_state`).
    ///
    /// On the CUDA path the readout is a batched async D2H into REUSED
    /// pinned host buffers followed by a SINGLE `synchronize()` (see
    /// `read_params_pinned` and the `snapshot_pinned_params` field for the
    /// reuse / single-consumer invariant). The previous
    /// implementation issued one synchronous `to_device(CPU)` per param,
    /// i.e. N serialized device syncs per window — the cpu-cadence idle
    /// floor on slow-PCIe ranks. A failure (or the CPU device, which needs
    /// no transfer) falls back to the per-tensor passthrough.
    ///
    /// Synchronizes comm_stream before reading, so Update + RequestParams
    /// processed in the same `handle_control()` call cannot read mid-copy data.
    pub fn snapshot_params(&mut self) -> ParamSnapshot {
        // Wait for any pending load_averaged() non-blocking copy to finish,
        // so we read post-update weights, never mid-writeback bytes.
        if let Some(stream) = &self.comm_stream {
            let _ = stream.synchronize();
        }

        let (params, buffers) = if self.comm_stream.is_some() {
            // CUDA path: one synchronize for the whole readout.
            self.read_params_pinned()
                .unwrap_or_else(|_| self.read_params_passthrough())
        } else {
            // CPU device: params already live on host, no transfer needed.
            self.read_params_passthrough()
        };

        ParamSnapshot {
            rank: self.rank,
            params,
            buffers,
            // TRUE step count since the last sync — NOT floored to 1. A rank
            // that did 0 steps (a reduce landing with no leftover work to
            // dispatch, e.g. the edge schedule at an epoch/run tail) still
            // holds the previous consensus; averaging it back in with a
            // weight of 1 would skew the consensus toward stale weights.
            // The averaging weights by this count and excludes zero-step
            // ranks (sum-and-count: scale by `batch_count`, divide once).
            batch_count: self.steps_since_avg,
        }
    }

    /// Batched async GPU->CPU readout of params + buffers into the reused
    /// pinned staging buffers, collapsing the per-param synchronous D2H
    /// into a single `comm_stream.synchronize()`. Lazily allocates the
    /// pinned buffers on first call. Returns clones of the staging buffers
    /// (shared storage — caller must obey the single-consumer-per-window
    /// invariant documented on `snapshot_pinned_params`).
    fn read_params_pinned(&mut self) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        if self.snapshot_pinned_params.is_empty() && !self.param_vars.is_empty() {
            let mut bufs = Vec::with_capacity(self.param_vars.len());
            for v in &self.param_vars {
                bufs.push(pinned_like(&v.data())?);
            }
            self.snapshot_pinned_params = bufs;
        }
        if self.snapshot_pinned_buffers.is_empty() && !self.buffer_list.is_empty() {
            let mut bufs = Vec::with_capacity(self.buffer_list.len());
            for b in &self.buffer_list {
                bufs.push(pinned_like(&b.get())?);
            }
            self.snapshot_pinned_buffers = bufs;
        }

        let stream = self.comm_stream.as_ref().ok_or_else(|| {
            TensorError::new("read_params_pinned: comm_stream absent")
        })?;
        {
            // copy_ respects the current stream; pinned dst + non_blocking
            // makes each D2H a true async cudaMemcpyAsync on comm_stream.
            let _guard = StreamGuard::new(stream);
            for (dst, v) in self.snapshot_pinned_params.iter().zip(&self.param_vars) {
                dst.copy_(&v.data(), true)?;
            }
            for (dst, b) in self.snapshot_pinned_buffers.iter().zip(&self.buffer_list) {
                dst.copy_(&b.get(), true)?;
            }
        }
        // One host-sync for the entire window's readout (was N).
        stream.synchronize()?;

        Ok((
            self.snapshot_pinned_params.clone(),
            self.snapshot_pinned_buffers.clone(),
        ))
    }

    /// Per-tensor synchronous readout fallback: copy each param / buffer to
    /// CPU individually (no-op for tensors already on CPU). Used on the CPU
    /// device and as the safety net if the pinned async path errors.
    fn read_params_passthrough(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        let params = self.param_vars.iter()
            .map(|v| {
                let t = v.data();
                if t.device() == Device::CPU { t } else { t.to_device(Device::CPU).unwrap_or(t) }
            })
            .collect();
        let buffers = self.buffer_list.iter()
            .map(|b| {
                let t = b.get();
                if t.device() == Device::CPU { t } else { t.to_device(Device::CPU).unwrap_or(t) }
            })
            .collect();
        (params, buffers)
    }

    /// Load averaged parameters from the coordinator (CPU averaging path).
    ///
    /// Uses `copy_(non_blocking=true)` on the comm stream for GPU overlap.
    /// Records a `CudaEvent` so the compute stream waits before the next forward.
    pub fn load_averaged(&mut self, update: &AveragedParams) -> Result<()> {
        if update.params.len() != self.param_vars.len() {
            return Err(TensorError::new(&format!(
                "load_averaged: expected {} params, got {}",
                self.param_vars.len(), update.params.len()
            )));
        }

        let non_blocking = self.comm_stream.is_some();
        // Set comm stream as current if available (copy_ respects current stream)
        let _guard = self.comm_stream.as_ref().map(StreamGuard::new);

        // no_grad: parameters are leaf tensors with requires_grad=true
        //
        // Two paths: full overwrite (default, fast — single non-blocking
        // copy_ per param) and EASGD elastic blend (opt-in, when
        // `easgd_alpha` is set — stages averaged params on GPU, then
        // applies a batched in-place lerp `var := var + α(avg − var)`
        // = `(1−α)·var + α·avg` across all params via one CUDA kernel
        // launch). Buffers (BatchNorm running stats etc.) always overwrite
        // — blending them is undefined under the EASGD framework.
        {
            let _no_grad = NoGradGuard::new();
            match self.easgd_alpha {
                None => {
                    for (var, src) in self.param_vars.iter().zip(&update.params) {
                        var.data().copy_(src, non_blocking)?;
                    }
                }
                Some(alpha) => {
                    let mut avg_staged: Vec<crate::tensor::Tensor> =
                        Vec::with_capacity(update.params.len());
                    let mut dst_handles: Vec<crate::tensor::Tensor> =
                        Vec::with_capacity(update.params.len());
                    for (var, src) in self.param_vars.iter().zip(&update.params) {
                        let dst = var.data();
                        // zeros_like allocates a same-shape/dtype/device
                        // tensor; we immediately overwrite via copy_ so the
                        // initial zeroing is unused (no `empty_like` in flodl).
                        let avg_gpu = crate::tensor::Tensor::zeros_like(&dst)?;
                        avg_gpu.copy_(src, non_blocking)?;
                        avg_staged.push(avg_gpu);
                        dst_handles.push(dst);
                    }
                    crate::tensor::Tensor::foreach_lerp_scalar_(
                        &dst_handles, &avg_staged, alpha,
                    )?;
                }
            }
        }
        for (buf, src) in self.buffer_list.iter().zip(&update.buffers) {
            buf.get().copy_(src, non_blocking)?;
        }

        // Record event on comm_stream so compute_stream can wait
        if let (Some(ev), Some(stream)) = (&self.copy_done, &self.comm_stream) {
            ev.record_on(stream)?;
        }
        // Mark H2D pending so the next `sync_before_forward` host-syncs the
        // comm stream BEFORE `train_step`'s timing window opens. Without
        // this, the wait_event-based path leaks the H2D wait into
        // `batch_ms` via the implicit GPU sync at `loss.item()`.
        self.pending_param_h2d = true;

        self.current_version = update.version;
        // Instrumentation: mark when averaged params landed so the run
        // loop can split the between-chunk wait at this boundary.
        if self.prof_enabled {
            self.last_update_at = Some(std::time::Instant::now());
        }
        Ok(())
    }

    /// Perform in-place NCCL AllReduce(Avg) on this rank's parameters.
    ///
    /// All ranks must process SyncNow concurrently for the collective
    /// to complete. Runs on `comm_stream` and records `copy_done` so
    /// the compute stream waits before the next forward.
    ///
    /// Returns the divergence triple `(divergence, post_norm, pre_norm)`:
    /// - `divergence = ||pre - post|| / ||post||` (this rank's transversal
    ///   deviation from the post-AllReduce consensus),
    /// - `post_norm = ||post||` (the L2 norm of the consensus weights after
    ///   AllReduce; identical across ranks by construction),
    /// - `pre_norm = ||W_i||` (this rank's pre-AllReduce L2 norm; per-rank).
    ///
    /// All three are `None` together when scratch buffers are absent
    /// (Sync mode or no NCCL comm).
    ///
    /// **Cluster-mode abort recovery:** if the in-flight collective
    /// aborts (the cluster_worker watchdog called `abort()` on this
    /// comm because a peer died), this function restores params from
    /// the pre-sync scratch, waits for `NewNcclSession` bytes in the
    /// mailbox, rebuilds the comm via `replace_nccl_comm`, and retries
    /// the AllReduce on the survivor cohort. The averaging cycle MUST
    /// complete (with one fewer rank's contribution) — silently
    /// skipping it would leave survivors drifted from each other and
    /// violate the sync semantics. Bounded by `MAX_REBUILD_ATTEMPTS`
    /// as a safety against cascading-failure pathologies.
    pub(super) fn sync_now_nccl(&mut self) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
        const MAX_REBUILD_ATTEMPTS: usize = 32;

        let _diag_start = Instant::now();
        if self.nccl_comm.is_none() {
            return Ok((None, None, None));
        }

        let param_tensors: Vec<_> = self.param_vars.iter().map(|v| v.data()).collect();
        let mut nccl_ms_total = 0.0_f64;

        for attempt in 0..MAX_REBUILD_ATTEMPTS {
            // Snapshot or restore params <-> scratch.
            // First attempt: snapshot params -> scratch (pre-sync state).
            // Retries: restore params <- scratch because the failed
            // in-place AllReduce may have left params partially mutated.
            if let Some(ref scratch) = self.pre_sync_scratch {
                let _guard = self.comm_stream.as_ref().map(StreamGuard::new);
                if attempt == 0 {
                    for (dst, src) in scratch.iter().zip(&param_tensors) {
                        dst.copy_(src, true)?; // scratch <- params
                    }
                } else {
                    for (dst, src) in param_tensors.iter().zip(scratch.iter()) {
                        dst.copy_(src, true)?; // params <- scratch
                    }
                }
            } else if attempt > 0 {
                // No scratch and we need to retry — can't recover param
                // state cleanly. Cluster NCCL mode must allocate scratch
                // unconditionally (the orchestrator entry does this).
                return Err(TensorError::new(
                    "sync_now_nccl: NCCL aborted but pre_sync_scratch is None; \
                     cannot restore params for retry. Cluster NCCL mode must \
                     allocate scratch unconditionally.",
                ));
            }

            let param_refs: Vec<&Tensor> = param_tensors.iter().collect();
            let comm = self.nccl_comm.as_ref().expect("nccl_comm present");

            let nccl_start = Instant::now();
            let attempt_result: Result<()> = if let Some(stream) = &self.comm_stream {
                match comm.all_reduce_on_stream(&param_refs, ReduceOp::Avg, stream) {
                    Ok(()) => stream.synchronize(),
                    Err(e) => Err(e),
                }
            } else {
                comm.all_reduce(&param_refs, ReduceOp::Avg)
            };
            nccl_ms_total += nccl_start.elapsed().as_secs_f64() * 1000.0;

            match attempt_result {
                Ok(()) => {
                    // Successful AllReduce. Compute divergence (if scratch
                    // is present), record event, log, return.
                    let divg_start = Instant::now();
                    let divergence = if let Some(ref scratch) = self.pre_sync_scratch {
                        // scratch = pre. Compute pre-norm BEFORE mutating
                        // scratch (next foreach_add_list_ overwrites in
                        // place to scratch = pre - post).
                        let pre_norm_tensors = Tensor::foreach_norm(scratch, 2.0)?;
                        let mut pre_sq = 0.0f64;
                        for n in &pre_norm_tensors {
                            let v: f64 = n.item()?;
                            pre_sq += v * v;
                        }
                        let pre_norm = pre_sq.sqrt();

                        Tensor::foreach_add_list_(scratch, &param_tensors, -1.0)?;

                        let diff_norms = Tensor::foreach_norm(scratch, 2.0)?;
                        let post_norms = Tensor::foreach_norm(&param_tensors, 2.0)?;

                        let mut diff_sq = 0.0f64;
                        for n in &diff_norms {
                            let v: f64 = n.item()?;
                            diff_sq += v * v;
                        }
                        let mut post_sq = 0.0f64;
                        for n in &post_norms {
                            let v: f64 = n.item()?;
                            post_sq += v * v;
                        }

                        let post_norm = post_sq.sqrt();
                        let div = if post_norm > 1e-10 {
                            diff_sq.sqrt() / post_norm
                        } else {
                            0.0
                        };

                        crate::verbose!(
                            "  ddp-worker: rank {} sync divergence={:.6} \
                             (||delta||={:.4}, ||pre||={:.4}, ||post||={:.4})",
                            self.rank, div, diff_sq.sqrt(), pre_norm, post_norm,
                        );
                        (Some(div), Some(post_norm), Some(pre_norm))
                    } else {
                        (None, None, None)
                    };

                    if let (Some(ev), Some(stream)) =
                        (&self.copy_done, &self.comm_stream)
                    {
                        ev.record_on(stream)?;
                    }
                    let divg_ms = divg_start.elapsed().as_secs_f64() * 1000.0;
                    let total_ms = _diag_start.elapsed().as_secs_f64() * 1000.0;
                    crate::verbose!(
                        "  ddp-sync-diag: rank {} sync_total={:.1}ms (nccl={:.1}ms divg={:.1}ms attempts={})",
                        self.rank, total_ms, nccl_ms_total, divg_ms, attempt + 1,
                    );
                    return Ok(divergence);
                }
                Err(e) => {
                    let aborted = self
                        .nccl_abort_handle
                        .as_ref()
                        .is_some_and(|h| h.is_aborted());
                    if !aborted {
                        // Not our abort — propagate.
                        return Err(e);
                    }
                    if attempt + 1 == MAX_REBUILD_ATTEMPTS {
                        return Err(TensorError::new(&format!(
                            "sync_now_nccl: rank {} hit max NCCL rebuild \
                             attempts ({}) without successful AllReduce",
                            self.rank, MAX_REBUILD_ATTEMPTS,
                        )));
                    }
                    crate::verbose!(
                        "  ddp-worker: rank {} NCCL collective aborted on \
                         attempt {} (err: {}), waiting for new comm and \
                         retrying",
                        self.rank,
                        attempt + 1,
                        e,
                    );
                    // Wait for NewNcclSession bytes in the mailbox + rebuild.
                    let pending = self.wait_for_nccl_session()?;
                    let uid_bytes: [u8; crate::distributed::NCCL_UNIQUE_ID_BYTES] =
                        pending.uid_bytes.as_slice().try_into().map_err(|_| {
                            TensorError::new(
                                "sync_now_nccl: NewNcclSession uid_bytes \
                                 wrong length (expected NCCL_UNIQUE_ID_BYTES)",
                            )
                        })?;
                    let uid =
                        crate::distributed::nccl::NcclUniqueId::from_bytes(uid_bytes);
                    let new_comm = crate::distributed::nccl::NcclRankComm::init_rank(
                        pending.new_rank,
                        pending.new_world_size,
                        &uid,
                    )?;
                    self.replace_nccl_comm(new_comm);
                    // Loop continues: retry the AllReduce on the new comm.
                }
            }
        }
        // Unreachable: every loop iteration either returns or errors
        // out at the max-attempts edge.
        Err(TensorError::new(&format!(
            "sync_now_nccl: rank {} unexpected exit from retry loop",
            self.rank,
        )))
    }

    /// Block until the cluster-mode session mailbox is populated, then
    /// drain it. Called from [`Self::sync_now_nccl`] after an NCCL
    /// abort to wait for the coord's `NewNcclSession` broadcast.
    ///
    /// 60-second safety cap — production rendezvous typically completes
    /// in <100ms once the coord receives the generator survivor's UID,
    /// so a 60s wait surfaces a stuck or misconfigured rendezvous loudly
    /// rather than hanging indefinitely.
    pub(super) fn wait_for_nccl_session(
        &self,
    ) -> Result<crate::distributed::cluster_worker::PendingNcclSession> {
        let mailbox = self.nccl_session_mailbox.as_ref().ok_or_else(|| {
            TensorError::new(
                "sync_now_nccl: NCCL aborted but no session mailbox attached; \
                 cluster_worker must call attach_nccl_session_mailbox before \
                 run_until_shutdown.",
            )
        })?;
        let start = Instant::now();
        let max_wait = Duration::from_secs(60);
        loop {
            if let Ok(mut g) = mailbox.lock() {
                if let Some(p) = g.take() {
                    return Ok(p);
                }
            }
            // Lone-NCCL-survivor early exit: NCCL requires `world_size
            // >= 2`. When the local dead-rank ledger reports `dead_count
            // >= world_size - 1`, no surviving peer can rendezvous with
            // us — the coord will broadcast `ShutdownWithSave` (or
            // already has) instead of `NewNcclSession`. Bail fast so
            // the worker terminates within one poll tick rather than
            // sitting in the 60s timeout. The coord-side ShutdownWithSave
            // dispatch is still authoritative for `.meta.json`; the
            // rank-side bundle write fires from `handle_control` once
            // we exit this wait and the main loop drains the queued
            // `ShutdownWithSave` frame.
            if let Some(ref dead_ranks) = self.local_dead_ranks {
                let dead = dead_ranks.dead_count();
                if dead >= self.world_size.saturating_sub(1) {
                    return Err(TensorError::new(&format!(
                        "sync_now_nccl: rank {} is lone NCCL survivor \
                         ({} of {} ranks dead); no rendezvous possible",
                        self.rank, dead, self.world_size,
                    )));
                }
            }
            if start.elapsed() > max_wait {
                return Err(TensorError::new(&format!(
                    "sync_now_nccl: rank {} timed out waiting for new \
                     NCCL session after {:?}",
                    self.rank, max_wait,
                )));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Host-synchronize the comm stream so any pending cpu-avg H2D copy
    /// completes BEFORE `train_step`'s timing window opens.
    ///
    /// Must be called before each forward pass. The previous implementation
    /// queued a `compute_stream.wait_event(copy_done)` here — a stream-level
    /// dependency that returned to the host immediately. The catch: the
    /// implicit GPU sync at `loss.data().item()?` inside the timing window
    /// then propagated the H2D wait into the measured `batch_ms`, polluting
    /// ElChe's throughput signal asymmetrically (cpu-avg only — NCCL's
    /// `sync_now_nccl` host-synchronizes internally, so there's nothing to
    /// wait for here on that path).
    ///
    /// The host-sync moves that wait outside the timing window. Total wall
    /// time is identical (we pay the H2D wait either way); only the clock
    /// that measures it changes. The `pending_param_h2d` flag avoids
    /// per-batch synchronize overhead on batches that don't follow an Update.
    /// No-op on CPU (no streams).
    pub(super) fn sync_before_forward(&mut self) -> Result<()> {
        if self.pending_param_h2d
            && let Some(stream) = &self.comm_stream
        {
            let t = Instant::now();
            stream.synchronize()?;
            self.last_h2d_wait_ms = t.elapsed().as_secs_f64() * 1000.0;
            if self.prof_enabled {
                self.h2d_wait_ms_total += self.last_h2d_wait_ms;
            }
            self.pending_param_h2d = false;
        }
        Ok(())
    }

    /// Run one forward + backward + optimizer step.
    ///
    /// `train_fn` receives a reference to the concrete model `M` and the batch
    /// tensors, and must return the scalar loss [`Variable`]. The worker handles
    /// stream sync, backward, optimizer step, and zero_grad.
    ///
    /// Returns `(loss_value, wall_ms)`.
    pub fn train_step(
        &mut self,
        batch: &[Tensor],
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<(f64, f64)> {
        self.sync_before_forward()?;

        // Pin all CUDA work for this call to compute_stream. The
        // AccumulateGrad nodes for this worker's parameters are pinned to
        // compute_stream in GpuWorker::new (see `_grad_accumulators`). The
        // gradient-producing kernels invoked here must arrive on the same
        // stream or libtorch fires the input_buffer.cpp:240
        // "AccumulateGrad node's stream does not match" warning.
        // dispatch_next_chunk already wraps the chunk loop in StreamGuard,
        // so this nests harmlessly there. Direct callers (custom training
        // loops, tests) need it here for the same guarantee.
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        let start = Instant::now();

        // User-provided forward + loss computation
        let loss = train_fn(&self.model, batch)?;
        let loss_val: f64 = loss.data().item()?;

        // Backward
        loss.backward()?;

        // Per-worker gradient clipping (before optimizer step).
        if let Some(max_norm) = self.max_grad_norm {
            let params: Vec<Tensor> = self.model.parameters()
                .iter()
                .filter(|p| p.variable.grad().is_some())
                .map(|p| p.variable.data())
                .collect();
            if !params.is_empty() {
                Tensor::clip_grad_norm_fused(&params, max_norm)?;
            }
        }

        // Per-batch LR: scheduler tracks global progress.
        // global_step = total batches at last sync, steps_since_avg = local
        // batches since then. The LR reflects this worker's real position
        // in the global schedule, multiplied by the DDP linear-scaling factor
        // (1.0 when lr_scale_ratio == 0 or world_size == 1).
        if let Some(ref sched) = self.scheduler {
            let base = sched.lr(self.global_step + self.steps_since_avg);
            self.optimizer.set_lr(base * self.lr_scale);
        }

        // Optimizer step (GPU-local Adam, ~0.1ms fused kernel)
        self.optimizer.step()?;
        self.optimizer.zero_grad();

        self.local_step += 1;
        self.steps_since_avg += 1;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((loss_val, elapsed_ms))
    }
}
