//! Self-driven training loops for the process-per-rank model: 5 entry points (`run_self_driven_{sync,cadence,async}_{nccl,cpu}`).

use std::time::Instant;

use crate::autograd::Variable;
use crate::distributed::cuda_stream::StreamGuard;
use crate::nn::Module;
use crate::tensor::{Device, Result, Tensor, TensorError};

use super::super::make_partition;
use super::GpuWorker;

impl<M: Module> GpuWorker<M> {
    /// Trigger an NCCL AllReduce across all ranks on this worker's
    /// parameters and reset the local steps-since-avg counter.
    ///
    /// Self-driven entry point for the cluster-rank inline loop.
    /// Mirrors the SyncNow handler in `dispatch_control` but skips
    /// the SyncAck — there's no coordinator listening in the
    /// process-per-rank model.
    ///
    /// All ranks must reach `sync_now` concurrently for the AllReduce
    /// collective to complete. Caller is responsible for cadence
    /// decisions (Sync: every batch; Cadence/Async: ElChe-determined K).
    pub fn sync_now(&mut self) -> Result<()> {
        let _ = self.sync_now_nccl()?;
        self.steps_since_avg = 0;
        self.local_step += 1;
        Ok(())
    }

    /// Self-driven inline training loop for `ApplyPolicy::Sync +
    /// AverageBackend::Nccl` (the simplest cluster-rank case).
    ///
    /// Runs for `num_epochs`. For each epoch:
    /// 1. Compute this rank's slice of the global permutation (via
    ///    the module-private `make_partition` — same deterministic
    ///    shuffle the coordinator used so the cross-rank
    ///    disjoint-coverage guarantee holds).
    /// 2. For each batch:
    ///    - Synchronous data load + H2D transfer
    ///    - `train_step` (forward + backward + optional grad clipping +
    ///      scheduler + optimizer step)
    ///    - `sync_now` (NCCL AllReduce on params)
    /// 3. Track `total_loss` across batches; caller can read it via
    ///    [`GpuWorker::current_epoch`] etc.
    ///
    /// Sync policy: this rank decides `K = 1` (AllReduce every batch)
    /// internally, with no coordinator driving SyncNow control messages.
    ///
    /// **Not supported by this self-driven entry:**
    /// - VRAM-aware prefetch (sync data loading only)
    /// - Per-batch metrics + per-epoch MetricsMsg (DdpHandle metrics
    ///   queue is empty on the self-driven path)
    /// - `Cadence` / `Async` policies (loud error in the cluster-rank
    ///   dispatch — those policies go through the controller-driven
    ///   `via_coord` paths instead)
    /// - `epoch_fn` callback (controller-driven path only)
    pub fn run_self_driven_sync_nccl(
        &mut self,
        num_epochs: usize,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<()> {
        if self.nccl_comm.is_none() {
            return Err(TensorError::new(
                "GpuWorker::run_self_driven_sync_nccl requires an NCCL comm; \
                 caller must build the worker with Some(comm).",
            ));
        }

        // Pin CUDA work to compute_stream for the full training session
        // (same invariant run_epoch_plan relies on; protects AccumulateGrad
        // node stream-pinning from interleaving with default-stream ops).
        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        for epoch in 0..num_epochs {
            self.current_epoch = epoch;
            let total_samples = self.dataset.len();
            // Use the same global-permutation slice the coordinator would
            // have assigned this rank. Equal share — no throughput-based
            // rebalancing in this slice (future work).
            let share = total_samples / self.world_size;
            let offset = self.rank * share;
            let size = if self.rank == self.world_size - 1 {
                total_samples - offset // last rank picks up remainder
            } else {
                share
            };
            self.partition = make_partition(
                offset, size, total_samples, epoch, self.base_seed,
            );

            let num_batches = self.partition.len() / self.batch_size;
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let batch = self.dataset.get_batch(indices)?;
                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch
                        .into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };

                let (_loss, _ms) = self.train_step(&batch, train_fn)?;
                // ApplyPolicy::Sync: AllReduce after every batch.
                self.sync_now()?;
            }
        }
        Ok(())
    }

    /// Self-driven inline training loop for `ApplyPolicy::Sync +
    /// AverageBackend::Cpu` — CPU-averaging counterpart of
    /// [`Self::run_self_driven_sync_nccl`].
    ///
    /// Same per-batch protocol (forward + backward + optimizer step) and
    /// same K=1 cadence (AllReduce-Avg after every batch). The reduce
    /// mechanism switches from NCCL collective to TCP round-trip via
    /// [`CpuReduceClient`]:
    ///
    /// - Each batch: `train_step` (Local SGD, identical to NCCL path)
    /// - Then: `cpu_client.all_reduce_tensors(&params_on_cpu)` → averaged
    ///   tensors on CPU
    /// - Load averaged tensors back into the live params via `copy_`
    ///   (handles the CPU-to-GPU move when params live on a CUDA device)
    ///
    /// Local-SGD semantics (`train_step` then AllReduce-Avg every batch)
    /// over the same `param_vars` set; the only switch from the NCCL
    /// variant is TCP rather than NCCL for the reduction.
    ///
    /// **Not supported by this self-driven entry:**
    /// VRAM-aware prefetch, per-epoch [`super::super::MetricsMsg`] aggregation,
    /// `epoch_fn` / `metrics_fn` / `scheduler_fn` / `checkpoint_every`
    /// callbacks, EASGD blending. Use the controller-driven via_coord
    /// paths for the full callback + persistence surface.
    ///
    /// [`CpuReduceClient`]: crate::distributed::CpuReduceClient
    pub fn run_self_driven_sync_cpu(
        &mut self,
        cpu_client: &mut crate::distributed::cpu_reduce::CpuReduceClient,
        num_epochs: usize,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<()> {
        if (cpu_client.world_size() as usize) != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_sync_cpu: cpu_client world_size ({}) \
                 must equal worker world_size ({})",
                cpu_client.world_size(), self.world_size,
            )));
        }

        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        for epoch in 0..num_epochs {
            self.current_epoch = epoch;
            let total_samples = self.dataset.len();
            let share = total_samples / self.world_size;
            let offset = self.rank * share;
            let size = if self.rank == self.world_size - 1 {
                total_samples - offset
            } else {
                share
            };
            self.partition = make_partition(
                offset, size, total_samples, epoch, self.base_seed,
            );

            let num_batches = self.partition.len() / self.batch_size;
            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let batch = self.dataset.get_batch(indices)?;
                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch
                        .into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };

                let (_loss, _ms) = self.train_step(&batch, train_fn)?;

                // CPU AllReduce-Avg: ship live params (move to CPU
                // happens inside tensors_to_round_frame), receive
                // averaged tensors, load back into live params.
                let param_tensors: Vec<Tensor> = self
                    .param_vars
                    .iter()
                    .map(|v| v.data())
                    .collect();
                let param_refs: Vec<&Tensor> = param_tensors.iter().collect();
                let averaged = cpu_client.all_reduce_tensors(&param_refs)?;
                for (dst, src) in param_tensors.iter().zip(&averaged) {
                    dst.copy_(src, false)?;
                }
                self.steps_since_avg = 0;
                self.local_step += 1;
            }
        }
        Ok(())
    }

    /// Self-driven inline training loop for `ApplyPolicy::Cadence` /
    /// `ApplyPolicy::Async` + `AverageBackend::Nccl` — heterogeneous
    /// Local-SGD with ElChe-driven K and the full convergence-guard
    /// pipeline.
    ///
    /// Under the NCCL backend, Cadence and Async share the same algorithm
    /// (overshoot machinery is an async/CPU concept and has no meaning on
    /// NCCL), so both policies route through this single loop.
    ///
    /// Per-cycle protocol (mirrors `Coordinator::finish_averaging_nccl`):
    ///
    /// 1. Run `train_step` (forward + backward + optimizer step) until the
    ///    rank's `local_batch_idx` reaches `el_che.batch_counts()[rank]`.
    /// 2. Measure cycle wall time and AllReduce the per-rank vector via
    ///    `ddp.all_reduce_per_rank_f64` — deterministic gather, no broadcast.
    /// 3. AllReduce-Avg parameters with weight-space divergence
    ///    measurement via `ddp.average_params_with_divergence(&scratch)`.
    /// 4. AllReduce per-rank `divergence` and per-rank `pre_norm` for the
    ///    [`DivergenceReport`]; `post_norm` is identical across ranks
    ///    post-AllReduce so it's used directly from this rank.
    /// 5. Feed timing into [`crate::distributed::ElChe::report_timing`].
    /// 6. Run the guard: `convergence_guard.report(&report, k_used, k_max)`
    ///    → [`ConvergenceAction`]:
    ///    - [`ConvergenceAction::NudgeDown`] `{ factor }` →
    ///      [`ElChe::nudge_anchor_down`] (applied for all non-Sync policies).
    ///    - [`ConvergenceAction::Stable`] + `elche_relax_up` →
    ///      [`ElChe::relax_anchor_up`], bounded by `max_anchor`. Honored
    ///      on Cadence as well as Async.
    ///    - [`ConvergenceAction::SuppressGrowth`] → no-op (hold cadence).
    ///
    /// **Not supported by this self-driven entry:** VRAM-aware
    /// prefetch, per-epoch [`super::super::MetricsMsg`] aggregation, `epoch_fn` /
    /// `metrics_fn` / `scheduler_fn` / `checkpoint_every` callbacks,
    /// LR-aware meta-controller (needs scheduler_fn flow), Timeline
    /// events for `Divergence` / `SyncEnd` / `AnchorChanged` /
    /// `GuardTelemetry`, per-epoch partition recompute from current
    /// ElChe ratios, Async-mode overshoot growth (irrelevant for NCCL —
    /// async/CPU concept).
    ///
    /// [`ConvergenceAction`]: super::super::convergence::ConvergenceAction
    /// [`ConvergenceAction::NudgeDown`]: super::super::convergence::ConvergenceAction::NudgeDown
    /// [`ConvergenceAction::Stable`]: super::super::convergence::ConvergenceAction::Stable
    /// [`ConvergenceAction::SuppressGrowth`]: super::super::convergence::ConvergenceAction::SuppressGrowth
    /// [`DivergenceReport`]: super::super::convergence::DivergenceReport
    /// [`ElChe::nudge_anchor_down`]: super::super::super::ddp::ElChe::nudge_anchor_down
    /// [`ElChe::relax_anchor_up`]: super::super::super::ddp::ElChe::relax_anchor_up
    #[allow(clippy::too_many_arguments)]
    pub fn run_self_driven_cadence_nccl(
        &mut self,
        ddp: &super::super::super::ddp::Ddp,
        el_che: &mut super::super::super::ddp::ElChe,
        convergence_guard: &mut dyn super::super::convergence::ConvergenceGuard,
        scratch: &[Tensor],
        partition_sizes: &[usize],
        elche_relax_up: bool,
        num_epochs: usize,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<()> {
        use super::super::convergence::{ConvergenceAction, DivergenceReport};

        if partition_sizes.len() != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_cadence_nccl: partition_sizes len ({}) \
                 must equal world_size ({})",
                partition_sizes.len(), self.world_size,
            )));
        }
        if ddp.world_size() != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_cadence_nccl: ddp world_size ({}) \
                 must equal worker world_size ({})",
                ddp.world_size(), self.world_size,
            )));
        }

        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        let total_samples = self.dataset.len();
        let my_offset: usize = partition_sizes.iter().take(self.rank).sum();
        let my_size = partition_sizes[self.rank];

        for epoch in 0..num_epochs {
            self.current_epoch = epoch;
            self.partition = make_partition(
                my_offset, my_size, total_samples, epoch, self.base_seed,
            );

            let num_batches = self.partition.len() / self.batch_size;
            let mut local_batch_idx: usize = 0;
            let mut cycle_start: Option<Instant> = None;

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let batch = self.dataset.get_batch(indices)?;
                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch
                        .into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };

                if cycle_start.is_none() {
                    cycle_start = Some(Instant::now());
                }
                let (_loss, _ms) = self.train_step(&batch, train_fn)?;
                local_batch_idx += 1;

                let target = el_che.batch_counts()[self.rank];
                if local_batch_idx >= target {
                    let cycle_wall_ms = cycle_start
                        .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);

                    // Cross-rank timing AllReduce: each rank writes its own
                    // slot, AllReduce-Sum yields the gathered vector on every
                    // rank — ElChe::report_timing's input.
                    let mut wall_ms_vec = vec![0.0f64; self.world_size];
                    wall_ms_vec[self.rank] = cycle_wall_ms;
                    ddp.all_reduce_per_rank_f64(&mut wall_ms_vec)?;

                    // Snapshot batch_counts BEFORE averaging — report_timing
                    // and nudge_anchor_down both mutate ElChe.
                    let counts: Vec<usize> = el_che.batch_counts().to_vec();

                    // Param AllReduce-Avg + weight-space divergence triple
                    // for this rank.
                    let sync_start = Instant::now();
                    let (local_div, local_post, local_pre) =
                        ddp.average_params_with_divergence(scratch)?;
                    let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;

                    // Gather divergence + pre_norm across ranks for the
                    // ConvergenceGuard's DivergenceReport. post_norm is
                    // identical on every rank post-AllReduce (modulo
                    // float-rounding); use this rank's directly.
                    let mut deltas = vec![0.0f64; self.world_size];
                    deltas[self.rank] = local_div;
                    ddp.all_reduce_per_rank_f64(&mut deltas)?;

                    let mut pre_norms_vec = vec![0.0f64; self.world_size];
                    pre_norms_vec[self.rank] = local_pre.unwrap_or(0.0);
                    ddp.all_reduce_per_rank_f64(&mut pre_norms_vec)?;
                    let pre_norms: Option<Vec<f64>> = if local_pre.is_some() {
                        Some(pre_norms_vec)
                    } else {
                        None
                    };

                    el_che.report_timing(&wall_ms_vec, &counts, sync_ms);

                    let report = DivergenceReport {
                        deltas,
                        pre_norms,
                        post_norm: local_post,
                    };
                    let cycle_batches: usize = counts.iter().sum();
                    let k_max = counts.iter().copied().max().unwrap_or(0);
                    let action =
                        convergence_guard.report(&report, cycle_batches, k_max);
                    match action {
                        ConvergenceAction::Stable => {
                            // Commit any overhead-tune proposal, then
                            // optionally relax_up on top.
                            el_che.commit_proposed_anchor();
                            if elche_relax_up {
                                el_che.relax_anchor_up();
                            }
                        }
                        ConvergenceAction::SuppressGrowth => {
                            // Veto growth; allow shrink (the safe
                            // direction when divergence is rising).
                            el_che.veto_proposed_growth();
                        }
                        ConvergenceAction::NudgeDown { factor } => {
                            // Drop the proposal; nudge supersedes.
                            el_che.discard_proposed_anchor();
                            el_che.nudge_anchor_down(factor);
                        }
                    }

                    self.steps_since_avg = 0;
                    local_batch_idx = 0;
                    cycle_start = None;
                }
            }
        }
        Ok(())
    }

    /// Self-driven inline training loop for `ApplyPolicy::Cadence +
    /// AverageBackend::Cpu` — CPU-averaging counterpart of
    /// [`Self::run_self_driven_cadence_nccl`].
    ///
    /// Same per-cycle protocol as the NCCL version (timing AllReduce →
    /// param AllReduce-Avg with weight-space divergence → cross-rank
    /// gather of `(divergence, pre_norm)` → `ElChe::report_timing` →
    /// `convergence_guard.report(...)` → [`ConvergenceAction`] applied
    /// to ElChe), but every collective routes through
    /// [`CpuReduceClient`] instead of NCCL:
    ///
    /// - `cpu_client.all_reduce_per_rank_f64(...)` — gather timing /
    ///   divergence / pre_norm vectors via the avg-trick
    /// - `cpu_client.average_params_with_divergence(&params, &scratch)`
    ///   — TCP round-trip averaging + scratch-based divergence triple
    ///
    /// `Async + Cpu` is **not** routed here: genuine async semantics
    /// require the 3-phase Idle/Collecting/Computing machine that
    /// lives on the controller-driven `via_coord` path. Cadence + Cpu
    /// is the blocking-reduce variant.
    ///
    /// [`CpuReduceClient`]: crate::distributed::CpuReduceClient
    /// [`ConvergenceAction`]: super::super::convergence::ConvergenceAction
    #[allow(clippy::too_many_arguments)]
    pub fn run_self_driven_cadence_cpu(
        &mut self,
        cpu_client: &mut crate::distributed::cpu_reduce::CpuReduceClient,
        el_che: &mut super::super::super::ddp::ElChe,
        convergence_guard: &mut dyn super::super::convergence::ConvergenceGuard,
        scratch: &[Tensor],
        partition_sizes: &[usize],
        elche_relax_up: bool,
        num_epochs: usize,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<()> {
        use super::super::convergence::{ConvergenceAction, DivergenceReport};

        if partition_sizes.len() != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_cadence_cpu: partition_sizes len ({}) \
                 must equal world_size ({})",
                partition_sizes.len(), self.world_size,
            )));
        }
        if (cpu_client.world_size() as usize) != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_cadence_cpu: cpu_client world_size \
                 ({}) must equal worker world_size ({})",
                cpu_client.world_size(), self.world_size,
            )));
        }

        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        let total_samples = self.dataset.len();
        let my_offset: usize = partition_sizes.iter().take(self.rank).sum();
        let my_size = partition_sizes[self.rank];

        for epoch in 0..num_epochs {
            self.current_epoch = epoch;
            self.partition = make_partition(
                my_offset, my_size, total_samples, epoch, self.base_seed,
            );

            let num_batches = self.partition.len() / self.batch_size;
            let mut local_batch_idx: usize = 0;
            let mut cycle_start: Option<Instant> = None;

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let batch = self.dataset.get_batch(indices)?;
                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch
                        .into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };

                if cycle_start.is_none() {
                    cycle_start = Some(Instant::now());
                }
                let (_loss, _ms) = self.train_step(&batch, train_fn)?;
                local_batch_idx += 1;

                let target = el_che.batch_counts()[self.rank];
                if local_batch_idx >= target {
                    let cycle_wall_ms = cycle_start
                        .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                        .unwrap_or(0.0);

                    // Cross-rank timing AllReduce via CPU avg-trick.
                    let mut wall_ms_vec = vec![0.0f64; self.world_size];
                    wall_ms_vec[self.rank] = cycle_wall_ms;
                    cpu_client.all_reduce_per_rank_f64(&mut wall_ms_vec)?;

                    let counts: Vec<usize> = el_che.batch_counts().to_vec();

                    // Param AllReduce-Avg + weight-space divergence triple
                    // via CPU client. Build the &[&Tensor] view over
                    // self.param_vars once for this cycle.
                    let param_tensors: Vec<Tensor> = self
                        .param_vars
                        .iter()
                        .map(|v| v.data())
                        .collect();
                    let param_refs: Vec<&Tensor> = param_tensors.iter().collect();
                    let sync_start = Instant::now();
                    let (local_div, local_post, local_pre) = cpu_client
                        .average_params_with_divergence(&param_refs, scratch)?;
                    let sync_ms = sync_start.elapsed().as_secs_f64() * 1000.0;

                    // Gather divergence + pre_norm across ranks.
                    let mut deltas = vec![0.0f64; self.world_size];
                    deltas[self.rank] = local_div;
                    cpu_client.all_reduce_per_rank_f64(&mut deltas)?;

                    let mut pre_norms_vec = vec![0.0f64; self.world_size];
                    pre_norms_vec[self.rank] = local_pre.unwrap_or(0.0);
                    cpu_client.all_reduce_per_rank_f64(&mut pre_norms_vec)?;
                    let pre_norms: Option<Vec<f64>> = if local_pre.is_some() {
                        Some(pre_norms_vec)
                    } else {
                        None
                    };

                    el_che.report_timing(&wall_ms_vec, &counts, sync_ms);

                    let report = DivergenceReport {
                        deltas,
                        pre_norms,
                        post_norm: local_post,
                    };
                    let cycle_batches: usize = counts.iter().sum();
                    let k_max = counts.iter().copied().max().unwrap_or(0);
                    let action =
                        convergence_guard.report(&report, cycle_batches, k_max);
                    match action {
                        ConvergenceAction::Stable => {
                            el_che.commit_proposed_anchor();
                            if elche_relax_up {
                                el_che.relax_anchor_up();
                            }
                        }
                        ConvergenceAction::SuppressGrowth => {
                            el_che.veto_proposed_growth();
                        }
                        ConvergenceAction::NudgeDown { factor } => {
                            el_che.discard_proposed_anchor();
                            el_che.nudge_anchor_down(factor);
                        }
                    }

                    self.steps_since_avg = 0;
                    local_batch_idx = 0;
                    cycle_start = None;
                }
            }
        }
        Ok(())
    }

    /// Self-driven inline training loop for `ApplyPolicy::Async +
    /// AverageBackend::Cpu` — the only **truly** asynchronous combo.
    ///
    /// Each rank submits a parameter-snapshot round to the controller
    /// (non-blocking), keeps training while the round is in flight,
    /// polls for completion every batch, and on receipt of the
    /// averaged response performs an EASGD elastic blend with the
    /// (now-drifted) live params. The convergence-guard pipeline runs
    /// against `(snapshot_at_submit, averaged_response)`: averaging is
    /// on the parameters that were submitted, not on the live drifted ones.
    ///
    /// **Overshoot bound:** `max_overshoot = 1`. If a round is still in
    /// flight when the K-boundary triggers the next, we
    /// [`crate::distributed::AsyncCpuReduceClient::block_poll`] until the
    /// previous round completes before submitting the new one. The
    /// previous round's EASGD-blend + guard verdict still applies
    /// before the new snapshot is taken.
    ///
    /// **EASGD blend formula:** `W := (1-α)·W_local + α·W_avg` when
    /// `easgd_alpha = Some(α)`; full overwrite (`copy_`) when `None`.
    ///
    /// **Local-only guard verdict:** the cross-rank divergence gather
    /// is not performed on this path; the guard runs with
    /// `deltas[rank] = local_div`, others zero. EASGD plus the
    /// convergence guard (LR-drop aware) absorb the per-rank
    /// drift as long as overshoot stays bounded.
    ///
    /// **No ElChe timing report:** without a cross-rank wall_ms gather
    /// ElChe's auto-tune would drift across ranks, so cadence is static
    /// (anchor stays at config init).
    ///
    /// **Epoch-event semantics:** LR scheduler updates fire per-rank
    /// locally on epoch crossing (the fast rank applies LR drop for
    /// epoch E+1 ahead of slow rank still finishing E). Per-rank
    /// epoch-completion aggregation for metrics reporting is not
    /// supported on this path.
    #[allow(clippy::too_many_arguments)]
    pub fn run_self_driven_async_cpu(
        &mut self,
        async_client: &mut crate::distributed::cpu_reduce::AsyncCpuReduceClient,
        el_che: &mut super::super::super::ddp::ElChe,
        convergence_guard: &mut dyn super::super::convergence::ConvergenceGuard,
        partition_sizes: &[usize],
        elche_relax_up: bool,
        easgd_alpha: Option<f64>,
        num_epochs: usize,
        train_fn: &impl Fn(&M, &[Tensor]) -> Result<Variable>,
    ) -> Result<()> {
        use super::super::convergence::{ConvergenceAction, DivergenceReport};
        use crate::distributed::cpu_reduce::round_frame_to_tensors;
        use crate::distributed::controller::RoundFrame;

        if partition_sizes.len() != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_async_cpu: partition_sizes len ({}) \
                 must equal world_size ({})",
                partition_sizes.len(), self.world_size,
            )));
        }
        if (async_client.world_size() as usize) != self.world_size {
            return Err(TensorError::new(&format!(
                "GpuWorker::run_self_driven_async_cpu: async_client world_size \
                 ({}) must equal worker world_size ({})",
                async_client.world_size(), self.world_size,
            )));
        }

        let _stream_guard = self.compute_stream.as_ref().map(StreamGuard::new);

        let total_samples = self.dataset.len();
        let my_offset: usize = partition_sizes.iter().take(self.rank).sum();
        let my_size = partition_sizes[self.rank];

        // In-flight snapshot: the deep-cloned params at submit time.
        // Kept alive across batches; consumed when the round completes
        // (used as the "pre" side of the divergence math).
        let mut in_flight_snapshot: Option<Vec<Tensor>> = None;

        // Closure-like helper inlined twice (after poll-completion and
        // after block-poll at max_overshoot=1). Encapsulates: divergence
        // math, EASGD blend / full-overwrite, local-only guard report,
        // action → ElChe.
        let apply_completed_round = |snapshot: Vec<Tensor>,
                                     avg_frame: RoundFrame,
                                     el_che: &mut super::super::super::ddp::ElChe,
                                     guard: &mut dyn super::super::convergence::ConvergenceGuard,
                                     param_vars: &[Variable],
                                     device: Device,
                                     rank: usize,
                                     world_size: usize,
                                     elche_relax_up: bool,
                                     easgd_alpha: Option<f64>|
         -> Result<()> {
            // Averaged frame comes back as CPU tensors. Move to the
            // worker's device so foreach math + blending stay on-device.
            let averaged_cpu = round_frame_to_tensors(&avg_frame)?;
            let averaged: Vec<Tensor> = averaged_cpu
                .iter()
                .map(|t| t.to_device(device))
                .collect::<Result<Vec<_>>>()?;

            let param_tensors: Vec<Tensor> =
                param_vars.iter().map(|v| v.data()).collect();

            // Divergence math: pre_norm BEFORE mutating snapshot.
            let pre_norm_tensors = Tensor::foreach_norm(&snapshot, 2.0)?;
            let mut pre_sq = 0.0f64;
            for n in &pre_norm_tensors {
                let v: f64 = n.item()?;
                pre_sq += v * v;
            }
            let pre_norm = pre_sq.sqrt();

            // snapshot[i] += -1 * averaged[i]  →  snapshot = pre - post.
            Tensor::foreach_add_list_(&snapshot, &averaged, -1.0)?;
            let diff_norms = Tensor::foreach_norm(&snapshot, 2.0)?;
            let post_norms = Tensor::foreach_norm(&averaged, 2.0)?;

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
            let divergence = if post_norm > 1e-10 {
                diff_sq.sqrt() / post_norm
            } else {
                0.0
            };

            // EASGD blend or full overwrite into live params.
            match easgd_alpha {
                Some(alpha) => {
                    let beta = 1.0 - alpha;
                    for live in &param_tensors {
                        live.mul_scalar_(beta)?;
                    }
                    Tensor::foreach_add_list_(&param_tensors, &averaged, alpha)?;
                }
                None => {
                    for (live, avg) in param_tensors.iter().zip(averaged.iter()) {
                        live.copy_(avg, false)?;
                    }
                }
            }

            // Local-only guard: deltas[rank] = local_div, others zero.
            // Cross-rank gather is deferred (see method doc).
            let mut deltas = vec![0.0f64; world_size];
            deltas[rank] = divergence;
            let mut pre_norms_vec = vec![0.0f64; world_size];
            pre_norms_vec[rank] = pre_norm;
            let report = DivergenceReport {
                deltas,
                pre_norms: Some(pre_norms_vec),
                post_norm: Some(post_norm),
            };
            // K-used / k-max approximations from this rank's slot only:
            // local cycle batches. Cross-rank gather deferred.
            let k_used = el_che.batch_counts()[rank];
            let k_max = k_used;
            let action = guard.report(&report, k_used, k_max);
            match action {
                ConvergenceAction::Stable => {
                    el_che.commit_proposed_anchor();
                    if elche_relax_up {
                        el_che.relax_anchor_up();
                    }
                }
                ConvergenceAction::SuppressGrowth => {
                    el_che.veto_proposed_growth();
                }
                ConvergenceAction::NudgeDown { factor } => {
                    el_che.discard_proposed_anchor();
                    el_che.nudge_anchor_down(factor);
                }
            }
            Ok(())
        };

        for epoch in 0..num_epochs {
            self.current_epoch = epoch;
            self.partition = make_partition(
                my_offset, my_size, total_samples, epoch, self.base_seed,
            );

            let num_batches = self.partition.len() / self.batch_size;
            let mut local_batch_idx: usize = 0;

            for batch_idx in 0..num_batches {
                let start = batch_idx * self.batch_size;
                let end = start + self.batch_size;
                let indices = &self.partition[start..end];
                let batch = self.dataset.get_batch(indices)?;
                let batch: Vec<Tensor> = if self.device.is_cuda() {
                    batch
                        .into_iter()
                        .map(|t| t.to_device(self.device))
                        .collect::<Result<Vec<_>>>()?
                } else {
                    batch
                };

                let (_loss, _ms) = self.train_step(&batch, train_fn)?;
                local_batch_idx += 1;

                // Drain any completed round non-blocking.
                if in_flight_snapshot.is_some() {
                    if let Some(avg_frame) = async_client.poll_round()? {
                        let snapshot = in_flight_snapshot.take().unwrap();
                        apply_completed_round(
                            snapshot, avg_frame,
                            el_che, convergence_guard,
                            &self.param_vars, self.device,
                            self.rank, self.world_size,
                            elche_relax_up, easgd_alpha,
                        )?;
                    }
                }

                // K-boundary: submit a new round.
                let target = el_che.batch_counts()[self.rank];
                if local_batch_idx >= target {
                    // max_overshoot = 1: if previous round still in
                    // flight, block until it completes before submitting.
                    if let Some(snapshot) = in_flight_snapshot.take() {
                        let avg_frame = async_client.block_poll()?;
                        apply_completed_round(
                            snapshot, avg_frame,
                            el_che, convergence_guard,
                            &self.param_vars, self.device,
                            self.rank, self.world_size,
                            elche_relax_up, easgd_alpha,
                        )?;
                    }

                    // Deep-clone the live params as the new snapshot
                    // (shallow clone shares storage and would drift
                    // when subsequent train_step mutates params).
                    let snapshot: Vec<Tensor> = self
                        .param_vars
                        .iter()
                        .map(|v| {
                            let data = v.data();
                            let copy = Tensor::zeros_like(&data)?;
                            copy.copy_(&data, false)?;
                            Ok(copy)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Build the wire frame and submit. The submit_round
                    // call writes synchronously but doesn't read the
                    // response — the reader thread does that later.
                    let snap_refs: Vec<&Tensor> = snapshot.iter().collect();
                    let frame = crate::distributed::cpu_reduce::tensors_to_round_frame(&snap_refs)?;
                    async_client.submit_round(&frame)?;
                    in_flight_snapshot = Some(snapshot);

                    self.steps_since_avg = 0;
                    local_batch_idx = 0;
                }
            }
        }

        // Drain any in-flight round at end of training so the last
        // average gets applied before snapshot_params() captures the
        // final state.
        if let Some(snapshot) = in_flight_snapshot.take() {
            let avg_frame = async_client.block_poll()?;
            apply_completed_round(
                snapshot, avg_frame,
                el_che, convergence_guard,
                &self.param_vars, self.device,
                self.rank, self.world_size,
                elche_relax_up, easgd_alpha,
            )?;
        }

        Ok(())
    }
}
