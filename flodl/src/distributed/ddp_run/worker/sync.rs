//! Parameter snapshot / load and NCCL sync paths + per-step train_step.

use std::time::{Duration, Instant};

use crate::autograd::{NoGradGuard, Variable};
use crate::tensor::cuda_stream::StreamGuard;
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
///
/// `as_bf16` stages a Float32 tensor as BFloat16 instead (the `copy_`
/// into the buffer casts on the source device, so the D2H moves half
/// the bytes) — the [`WorkerConfig::bf16_wire`] param staging. Non-f32
/// tensors keep their dtype regardless (integer buffers must round-trip
/// exactly, and the reduce bridge selects f32 buffers by dtype).
///
/// [`WorkerConfig::bf16_wire`]: crate::distributed::ddp_run::WorkerConfig::bf16_wire
fn pinned_like(t: &Tensor, as_bf16: bool) -> Result<Tensor> {
    let dtype = if as_bf16 && t.dtype() == crate::tensor::DType::Float32 {
        crate::tensor::DType::BFloat16
    } else {
        t.dtype()
    };
    let opts = TensorOptions { dtype, device: Device::CPU };
    Tensor::empty(&t.shape(), opts)?.pin_memory()
}

/// Work-weighted in-place AllReduce of the params across the NCCL
/// cohort, as ONE fused collective: each rank's contribution is
/// premultiplied INSIDE the collective by its normalized work factor
/// `fᵢ = nᵢ^γ / Σn^γ` (`ncclRedOpCreatePreMulSum`), so the output is
/// directly `Σ fᵢ·Wᵢ` — the work-weighted consensus. No pre-scale
/// kernel, no post-divide kernel: the collective's own write is the
/// last write, so there is nothing for the caller's divergence readout,
/// outer-optimizer writeback, or next forward to race (the historical
/// divide-race class is gone STRUCTURALLY, not fenced).
///
/// `γ` is the consensus allocation-weighting exponent (1.0 = plain
/// work-weighting); it folds into the factor's numerator and the count
/// collective (which sums `nᵢ^γ`), giving the NCCL path the same
/// γ-aware weighting as the CPU backend.
///
/// `buffer_refs` (the model's f32 buffers — BatchNorm running stats and
/// the like) ride the same sync as a SECOND premul collective, weighted
/// by the 0/1 mover indicator (`moverᵢ / Σmover`) — equal weight among
/// the ranks that moved, idle ranks contributing nothing but adopting
/// the consensus in place. This mirrors the CPU backend's buffer
/// semantics exactly (see `param_bridge`): only the params consensus is
/// γ-weighted; running stats must not inherit a fast rank's dominance.
/// Non-f32 buffers never reach here (the caller filters; NCCL premul is
/// f32-only, and integer counters are deterministic values updated
/// identically on every rank — passing them through locally is correct,
/// not a dropped sync). An empty `buffer_refs` skips the collective on
/// every rank alike (model-structural, so collectively consistent).
///
/// A rank that did 0 steps since the last sync still holds the previous
/// consensus; its factor is 0 and it contributes nothing — but it STILL
/// joins both collectives, so the cohort never stalls. If the WHOLE
/// cohort is idle (`Σn == 0`), the params already equal the consensus
/// and the param reduce is skipped (every rank sees the same gathered
/// `Σn`, so the skip is collective-consistent).
///
/// A cheap `ReduceOp::Sum` of the per-rank `[nᵢ^γ, moverᵢ]` pair
/// supplies `Σn^γ` and `Σmover` (a few bytes vs the multi-MB params)
/// BEFORE the param collective — that is what lets each rank compute
/// both pre-normalized factors. Running it on the SAME comm as the
/// param reduce keeps the totals consistent with the live cohort across
/// an abort-driven rebuild (they reflect the survivors). On an abort
/// the caller restores params AND buffers from the pre-sync scratches
/// before retrying; the retry re-runs the collectives on the rebuilt
/// comm, deriving fresh factors from the survivor cohort (the premul op
/// is comm-bound and created per call).
///
/// Requires NCCL >= 2.11 (build + runtime; the shim errors loudly
/// naming the found version).
// 9 args: the `rank`/`seq` pair is collective-step diagnostic context
// (-vvv ENTER/EXIT logging); a struct wrapper would obscure the hot path for
// a private single-caller helper (pub(crate) only for the NCCL test suite).
#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_allreduce_nccl(
    comm: &crate::distributed::nccl::NcclRankComm,
    stream: Option<&crate::tensor::cuda_stream::CudaStream>,
    param_refs: &[&Tensor],
    buffer_refs: &[&Tensor],
    n_i: f64,
    gamma: f64,
    device: Device,
    rank: usize,
    seq: usize,
) -> Result<()> {
    // Collective-step instrumentation (-vvv): this fn issues up to THREE
    // collectives per sync (count-reduce, then a CONDITIONAL param-reduce,
    // then a CONDITIONAL buffer-reduce when the model has f32 buffers). A
    // cohort desync shows up as ranks issuing a DIFFERENT number of collectives for
    // the same `seq` — e.g. one rank takes the `total_n <= 0` skip and does 1
    // all_reduce while peers do 2, leaving them waiting on a phantom. Logging
    // ENTER/EXIT of each collective per rank pins which one a stuck rank
    // parks in (NCCL busy-waits at 100% CPU with no peers).
    crate::debug!(
        "  ddp-areduce: rank {rank} seq={seq} COUNT enter (n_i={n_i}, gamma={gamma})"
    );
    // γ-effective work: `gamma_mass` owns the idle guard (an idle rank
    // has zero mass for ANY γ — raw powf gives 0^0 = 1, a stale rank
    // voting at full weight, or 0^{γ<0} = ∞, a NaN consensus) and the
    // γ = 1.0 identity path. Shared with the CPU backend's frame
    // weighting so the two cannot drift.
    let n_eff = crate::distributed::realized_work::gamma_mass(n_i, gamma);
    // STREAM-ORDER THE BODY ON THE COMM STREAM. Tensor ops (the count
    // tensor below) enqueue on the thread's CURRENT stream while the
    // collectives are enqueued on `stream` — and pool streams are
    // non-blocking, so nothing implicitly orders them. Pinning the body
    // to the comm stream keeps the sequence totally ordered regardless
    // of where the sync interrupts training (mid-chunk vs between
    // chunks). The historical scale/divide bookend kernels this guard
    // used to protect are gone (premultiplied inside the collective);
    // the guard remains for the count tensor and the readout ordering.
    let _stream_guard = stream.map(StreamGuard::new);
    // 0/1 mover indicator: the buffer consensus is equal-weight among the
    // ranks that moved (never γ-weighted — see the doc above).
    let mover = crate::distributed::realized_work::mover_mass(n_i);
    // [Σn^γ, Σmover] over the live cohort in one small collective.
    let count = Tensor::from_f32(&[n_eff as f32, mover as f32], &[2], device)?;
    let totals = match stream {
        Some(s) => {
            comm.all_reduce_on_stream(&[&count], ReduceOp::Sum, s)?;
            s.synchronize()?;
            count.to_f64_vec()?
        }
        None => {
            comm.all_reduce(&[&count], ReduceOp::Sum)?;
            count.to_f64_vec()?
        }
    };
    let (total_n, total_movers) = (totals[0], totals[1]);
    crate::debug!(
        "  ddp-areduce: rank {rank} seq={seq} COUNT exit (total_n={total_n}, total_movers={total_movers})"
    );
    if !crate::distributed::realized_work::is_realized(total_n) {
        // Whole cohort idle since the last sync: consensus already holds
        // (params AND buffers — no step means no forward, so no running-
        // stat drift either; both reduces are skipped together).
        crate::debug!(
            "  ddp-areduce: rank {rank} seq={seq} SKIP param-reduce (total_n={total_n}) \
             -- 1 collective this seq (peers doing more would deadlock)"
        );
        return Ok(());
    }
    // Pre-normalized per-rank factor: the collective output needs no
    // post-divide. NCCL's premul scalar is f32 (matching the f32 params).
    let factor = (n_eff / total_n) as f32;
    crate::debug!(
        "  ddp-areduce: rank {rank} seq={seq} PARAM enter (nparams={}, factor={factor})",
        param_refs.len()
    );
    match stream {
        Some(s) => {
            comm.all_reduce_premul_sum(param_refs, factor, Some(s))?;
        }
        None => {
            // No dedicated comm stream: the collectives share the
            // caller's current stream, so plain enqueue order fences them
            // against every downstream consumer on that stream.
            comm.all_reduce_premul_sum(param_refs, factor, None)?;
        }
    }
    crate::debug!("  ddp-areduce: rank {rank} seq={seq} PARAM exit");
    // Buffer consensus (BatchNorm running stats etc.): equal weight among
    // movers. `total_movers >= 1` is implied by the realized `total_n`
    // above (γ-mass is nonzero only for n_i > 0, which makes mover 1),
    // and it is identical on every rank (it came out of the collective),
    // so the divide is safe and the branch collectively consistent.
    if !buffer_refs.is_empty() {
        let buf_factor = (mover / total_movers) as f32;
        crate::debug!(
            "  ddp-areduce: rank {rank} seq={seq} BUFFER enter (nbuffers={}, factor={buf_factor})",
            buffer_refs.len()
        );
        comm.all_reduce_premul_sum(buffer_refs, buf_factor, stream)?;
        crate::debug!("  ddp-areduce: rank {rank} seq={seq} BUFFER exit");
    }
    if let Some(s) = stream {
        // EXIT FENCE: retire the collectives before returning. Their own
        // writes are the LAST writes to params/buffers (no divide kernel
        // follows anymore), so this synchronize is purely the entry-fence
        // contract ("the weighted reduce retires everything before
        // returning") that lets the caller's divergence readout, the
        // outer-optimizer writeback, and the next forward run unfenced on
        // THEIR streams. A wedged collective still parks the host here,
        // where the NCCL watchdog's abort can free it.
        s.synchronize()?;
    }
    Ok(())
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
        // SNAPSHOT-ENTRY FENCE: order the comm-stream readout after ALL
        // pending compute-stream work. A `RequestParams` processed mid-chunk
        // arrives with the previous batch's backward + optimizer-step kernels
        // possibly still in flight on compute_stream (the per-batch loss
        // readout happens BEFORE backward, so it guarantees nothing about the
        // step); `read_params_pinned` enqueues its D2H copies on the COMM
        // stream, and without this fence they race the in-flight update —
        // a torn snapshot fed into the cluster average. Same fence class as
        // `sync_now_nccl`'s sync-entry fence (the old per-param synchronous
        // `to_device(CPU)` ran on the current stream and was accidentally
        // ordered; the pinned rewrite hops streams and needs it explicit).
        if let Some(stream) = &self.compute_stream {
            let _ = stream.synchronize();
        }
        // Wait for any pending load_averaged() non-blocking copy to finish,
        // so we read post-update weights, never mid-writeback bytes.
        if let Some(stream) = &self.comm_stream {
            let _ = stream.synchronize();
        }

        let (params, buffers) = if self.comm_stream.is_some() {
            // CUDA path: one synchronize for the whole readout.
            match self.read_params_pinned() {
                Ok(out) => out,
                Err(e) => {
                    // The passthrough fallback is correct but serializes one
                    // device sync per param — the slow-PCIe idle floor the
                    // pinned path exists to remove. Surface the regression
                    // once instead of silently eating it every window.
                    if !self.pinned_fallback_logged {
                        self.pinned_fallback_logged = true;
                        eprintln!(
                            "flodl ddp: rank {} pinned snapshot readout failed ({e}); \
                             falling back to per-param synchronous D2H (slower)",
                            self.rank
                        );
                    }
                    self.read_params_passthrough()
                }
            }
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
                bufs.push(pinned_like(&v.data(), self.bf16_wire)?);
            }
            self.snapshot_pinned_params = bufs;
        }
        if self.snapshot_pinned_buffers.is_empty() && !self.buffer_list.is_empty() {
            let mut bufs = Vec::with_capacity(self.buffer_list.len());
            for b in &self.buffer_list {
                // Never bf16 (see the field doc: the bridge's f32 filter
                // + exact integer counters).
                bufs.push(pinned_like(&b.get(), false)?);
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

    /// Exact (dtype-preserving) snapshot for end-of-training reporting:
    /// always the per-tensor passthrough, never the pinned staging. Under
    /// `bf16_wire` the staging quantizes params to bf16 — fine for the
    /// averaging plane it feeds, wrong for the trained weights the final
    /// snapshot becomes (`TrainedState` / checkpoints must not inherit
    /// wire quantization). One synchronous readout at the very end of
    /// training costs nothing on any clock that matters.
    pub fn snapshot_params_exact(&mut self) -> ParamSnapshot {
        // Same entry fences as `snapshot_params`: the readout must not
        // race in-flight optimizer kernels or a pending writeback.
        if let Some(stream) = &self.compute_stream {
            let _ = stream.synchronize();
        }
        if let Some(stream) = &self.comm_stream {
            let _ = stream.synchronize();
        }
        let (params, buffers) = self.read_params_passthrough();
        ParamSnapshot {
            rank: self.rank,
            params,
            buffers,
            batch_count: self.steps_since_avg,
        }
    }

    /// Per-tensor synchronous readout fallback: copy each param / buffer to
    /// CPU individually (no-op for tensors already on CPU). Used on the CPU
    /// device and as the safety net if the pinned async path errors.
    fn read_params_passthrough(&self) -> (Vec<Tensor>, Vec<Tensor>) {
        // SNAPSHOT NEVER ALIASES LIVE STORAGE. A CUDA-resident tensor gets a
        // real copy from `to_device(CPU)`, but a CPU-resident one must be
        // deep-copied explicitly: the shallow clone shares storage with the
        // live param, and under Async the worker resumes training while the
        // bridge thread is still serializing the snapshot — an aliased
        // buffer ships torn floats (an element mix of adjacent optimizer
        // steps that is a valid state of NO step). One host memcpy per
        // window, the same cost class the CUDA path pays for its D2H. On a
        // copy failure fall back to the alias — matches the `to_device`
        // fallback below; sync/cadence consume the snapshot before training
        // resumes, so the fallback is only torn where it was always torn.
        fn cpu_detached(t: Tensor) -> Tensor {
            if t.device() != Device::CPU {
                return t.to_device(Device::CPU).unwrap_or(t);
            }
            Tensor::zeros_like(&t)
                .and_then(|c| c.copy_(&t, false).map(|()| c))
                .unwrap_or(t)
        }
        let params = self.param_vars.iter()
            .map(|v| cpu_detached(v.data()))
            .collect();
        let buffers = self.buffer_list.iter()
            .map(|b| cpu_detached(b.get()))
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

        // WRITEBACK-ENTRY FENCE: order the comm-stream H2D writes after ALL
        // pending compute-stream work. Under `Async` the worker keeps
        // training while the averaging round-trips, so an `Update` can land
        // with the previous batch's optimizer step still in flight on
        // compute_stream writing the same params the copies below overwrite.
        // Mirror of the `snapshot_params` entry fence (read side).
        if let Some(stream) = &self.compute_stream {
            stream.synchronize()?;
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
        // DiLoCo signal (disposable inner state): query this worker's own
        // per-site outer-optimizer instance (built-but-unused for stepping on
        // the CPU path, but it carries the policy bit). `resets_inner` governs
        // ONLY the inner-optimizer reset below — it is ORTHOGONAL to param
        // adoption. Param adoption stays governed by `easgd_alpha`: on cadence
        // (α=None) the new global is full-overwritten = textbook DiLoCo; on
        // cpu-async (α set) the ahead-of-sync overshoot is EASGD-blended into
        // the outer-stepped global, preserving that local work. DiLoCo's
        // resume edge comes from the momentum reset (per-rank inner momentum
        // has no consensus to checkpoint), NOT from the param overwrite, so
        // blending the overshoot keeps that edge.
        let resets_inner = self
            .outer_optimizer
            .as_ref()
            .is_some_and(|o| o.resets_inner());
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

        // DiLoCo: reset the inner optimizer after adopting the new global, so
        // its momentum / step count restart each outer round (disposable
        // inner state — the resume-faithful axis). Orthogonal to how params
        // were adopted above. No-op for SlowMo / OuterAvg / no outer optimizer.
        if resets_inner {
            self.optimizer.reset_state();
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

    /// Perform the in-place NCCL weighted AllReduce on this rank's
    /// parameters (work-weighted) and f32 buffers (mover-averaged —
    /// BatchNorm running stats and the like; see
    /// [`weighted_allreduce_nccl`] for the weighting asymmetry). Non-f32
    /// buffers keep their local value — deterministic counters updated
    /// identically on every rank, mirroring the CPU bridge's filter.
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

        // SYNC-ENTRY FENCE: order the comm-stream sync work after ALL pending
        // compute-stream work. At a mid-chunk sync the previous batch's
        // optimizer step can still be in flight on compute_stream (the
        // per-batch loss readout does not guarantee the step kernels have
        // retired); the scratch copy and the n_i scaling below read + mutate
        // params on the COMM stream, and without this fence they race the
        // in-flight update. The corruption is per-sync-rare — invisible at
        // smoke scale, near-certain across the thousands of syncs of a long
        // run (rig, 200ep: discrete loss perturbation at epoch 120, NaN at
        // epoch 124). The exit side needs no fence HERE because the weighted
        // reduce provides its own: it retires the final divide on the comm
        // stream and host-synchronizes before returning (the EXIT FENCE in
        // `weighted_allreduce_nccl`), so training resumes only after the
        // consensus has fully landed. That exit fence is load-bearing for
        // this contract — the divergence readout and outer-optimizer
        // writeback below rely on it.
        if let Some(ref cs) = self.compute_stream {
            cs.synchronize()?;
        }

        let param_tensors: Vec<_> = self.param_vars.iter().map(|v| v.data()).collect();
        // f32 buffers ride the sync (mover-averaged in place through the
        // storage-sharing handles); the subset is model-structural, so it
        // is identical in count/order on every rank and the collective
        // stays balanced. Non-f32 buffers keep their local value.
        let buffer_tensors: Vec<Tensor> = self
            .buffer_list
            .iter()
            .map(|b| b.get())
            .filter(|t| t.dtype() == crate::tensor::DType::Float32)
            .collect();
        let mut nccl_ms_total = 0.0_f64;

        // Monotonic per-rank sync sequence for the collective-step diagnostic.
        // Bumped once per sync (NOT per retry attempt) so a `seq` value names
        // the same logical reduce on every rank.
        let seq = self.nccl_sync_seq;
        self.nccl_sync_seq += 1;
        let rank = self.rank;

        for attempt in 0..MAX_REBUILD_ATTEMPTS {
            // Snapshot or restore params <-> scratch.
            // First attempt: snapshot params -> scratch (pre-sync state).
            // Retries: restore params <- scratch because the failed
            // in-place AllReduce may have left params partially mutated.
            if let Some(ref scratch) = self.pre_sync_scratch {
                let _guard = self.comm_stream.as_ref().map(StreamGuard::new);
                // no_grad: the RETRY branch copies INTO `param_tensors`,
                // which are `.data()` of the leaf `requires_grad` params —
                // an in-place op on a grad-requiring leaf raises libtorch's
                // `check_inplace` c10::Error (crosses FFI as a hard abort).
                // Retries only happen after an NCCL abort, so this fired
                // exclusively on the rebuild path (attempt 0's dst is
                // scratch, which is fine ungated); guarding the whole
                // block is harmless and mirrors `load_averaged` /
                // `weighted_allreduce_nccl`.
                let _no_grad = NoGradGuard::new();
                if attempt == 0 {
                    for (dst, src) in scratch.iter().zip(&param_tensors) {
                        dst.copy_(src, true)?; // scratch <- params
                    }
                } else {
                    for (dst, src) in param_tensors.iter().zip(scratch.iter()) {
                        dst.copy_(src, true)?; // params <- scratch
                    }
                }
                // Buffers mirror the params' snapshot/restore: an aborted
                // buffer collective can leave them partially premultiplied
                // (a torn running-var is eval garbage until BN momentum
                // heals it), so the retry must restart from clean state.
                if let Some(ref buf_scratch) = self.pre_sync_buffer_scratch {
                    if attempt == 0 {
                        for (dst, src) in buf_scratch.iter().zip(&buffer_tensors) {
                            dst.copy_(src, true)?; // scratch <- buffers
                        }
                    } else {
                        for (dst, src) in buffer_tensors.iter().zip(buf_scratch.iter()) {
                            dst.copy_(src, true)?; // buffers <- scratch
                        }
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
            let buffer_refs: Vec<&Tensor> = buffer_tensors.iter().collect();
            let n_i = self.steps_since_avg as f64;
            let device = self.device;
            let comm = self.nccl_comm.as_ref().expect("nccl_comm present");

            let nccl_start = Instant::now();
            // Work-weighted reduce: each rank premultiplies by its
            // normalized work factor inside the collective (PreMulSum),
            // yielding the consensus directly. Was an unweighted
            // ReduceOp::Avg, which over-represented idle / under-worked
            // ranks under proportional sharding, then a sum-and-count
            // form with bookend scale/divide kernels.
            let attempt_result: Result<()> = weighted_allreduce_nccl(
                comm,
                self.comm_stream.as_ref(),
                &param_refs,
                &buffer_refs,
                n_i,
                self.gamma,
                device,
                rank,
                seq,
            );
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

                    // OUTER STEP (NCCL, replicated per rank): transform the
                    // work-weighted consensus now in `param_tensors` into the
                    // new global on this rank's GPU. Identical consensus +
                    // replicated prev_global / momentum + a deterministic op
                    // give every rank the same new global and momentum update,
                    // so the cohort stays in lock-step with NO extra collective
                    // (the outer optimizer is replicated state, like the model).
                    // Runs AFTER divergence (which host-syncs its reads via
                    // `.item()`, so overwriting params here cannot race those
                    // reads) and BEFORE the copy_done event record below (so
                    // compute_stream waits for the stepped params). The whole
                    // step runs on comm_stream so it is ordered after the
                    // AllReduce and before that event. `None` => plain
                    // averaging (today's behavior, no extra work). Taken out of
                    // `self` so the comm-stream guard can coexist with the
                    // prev_global field updates without overlapping borrows.
                    if let Some(mut outer) = self.outer_optimizer.take() {
                        let _guard = self.comm_stream.as_ref().map(StreamGuard::new);
                        // First window: no prior anchor — use the consensus, so
                        // the outer gradient is zero (a no-op for any
                        // well-behaved variant) and the momentum seeds at zero.
                        let new_global = {
                            let prev: &[Tensor] =
                                self.outer_prev_global.as_deref().unwrap_or(&param_tensors);
                            outer.outer_step(prev, &param_tensors)?
                        };
                        {
                            let _no_grad = NoGradGuard::new();
                            for (p, ng) in param_tensors.iter().zip(&new_global) {
                                p.copy_(ng, true)?;
                            }
                        }
                        self.outer_prev_global = Some(new_global);
                        // DiLoCo: disposable inner state. The new global is now
                        // in params (full overwrite — the NCCL path already
                        // overwrites via the in-place AllReduce, so nothing
                        // extra needed there); reset the inner optimizer so its
                        // momentum / step count restart from the new global.
                        // SlowMo / OuterAvg keep the inner loop continuous.
                        let resets_inner = outer.resets_inner();
                        self.outer_optimizer = Some(outer);
                        if resets_inner {
                            self.optimizer.reset_state();
                        }
                    }

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
    ) -> Result<crate::distributed::nccl_session::PendingNcclSession> {
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
    /// An owned `compute_stream` guard, or `None` on CPU. The cooperative
    /// [`Worker`](crate::distributed::ddp_run::Worker) holds this across the
    /// user's forward + backward so the gradient-producing kernels arrive on
    /// the same stream as the `AccumulateGrad` nodes (pinned to `compute_stream`
    /// in `GpuWorker::new`) — the guarantee `train_step`'s internal guard gives
    /// the managed tier. Without it libtorch warns about an AccumulateGrad
    /// stream mismatch (and CUDA-graph capture would break).
    pub(crate) fn compute_stream_guard(&self) -> Option<StreamGuard> {
        self.compute_stream.as_ref().map(StreamGuard::new)
    }

    pub(crate) fn sync_before_forward(&mut self) -> Result<()> {
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

        // Gradient clip + LR schedule + optimizer step + zero_grad + counters.
        self.optimizer_step_and_bookkeep()?;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((loss_val, elapsed_ms))
    }

    /// The post-backward tail of a training step: gradient clipping, per-batch
    /// LR schedule, the fused optimizer step, `zero_grad`, and the step
    /// counters. Assumes the gradients for this step are already populated
    /// (the caller ran forward + `backward`).
    ///
    /// Split out of [`Self::train_step`] so the cooperative execution tier can
    /// run the user's own forward + backward and then hand control back here
    /// for the framework-owned bookkeeping the coordinator's schedule depends
    /// on (`steps_since_avg`, `local_step`, the LR trajectory). `train_step`
    /// remains the single-call form (forward + backward + this tail).
    ///
    /// Installs its own `compute_stream` guard so the optimizer kernels arrive
    /// on the same stream as the `AccumulateGrad` nodes (see the guard note in
    /// `train_step`). It nests harmlessly inside `train_step`'s guard, and is
    /// self-sufficient when the cooperative `Worker::step` drives it directly.
    pub(crate) fn optimizer_step_and_bookkeep(&mut self) -> Result<()> {
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

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
        Ok(())
    }
}
