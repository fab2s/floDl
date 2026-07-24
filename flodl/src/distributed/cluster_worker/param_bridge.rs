//! `ClusterWorker` CPU-averaging param bridge: receives
//! `ParamSnapshot`s from the inner `GpuWorker`, all-reduces them via
//! `CpuReduceClient`, feeds the averaged tensors back as
//! `ControlMsg::Update`, and emits the `SyncAck` divergence triple.

use super::*;

/// CPU-averaging param bridge: receives
/// [`ParamSnapshot`](crate::distributed::ddp_run::ParamSnapshot)s from the
/// inner [`GpuWorker`] (triggered by `RequestParams`), runs an
/// all-reduce round-trip through the data channel via
/// [`crate::distributed::cpu_reduce::CpuReduceClient`], and feeds the
/// averaged tensors back to the inner as `ControlMsg::Update`. Also
/// emits `TimingMsg::SyncAck` on the timing channel so the
/// coordinator's `nccl_ack` gate releases. The SyncAck carries the
/// weight-space divergence triple (`||pre - post|| / ||post||`,
/// `pre_norm`, `post_norm`) so the coord's
/// [`ConvergenceGuard`](crate::distributed::ddp_run::convergence::ConvergenceGuard)
/// sees real signal on the CPU averaging path.
///
/// When `cpu_client` is `None`, the bridge degrades to a discard
/// drainer (NCCL-only worker layout — the inner never emits
/// ParamSnapshot in that mode either, so the channel idles).
pub(super) fn param_bridge_loop(
    rank: u64,
    param_rx: mpsc::Receiver<crate::distributed::ddp_run::ParamSnapshot>,
    cpu_client: Option<crate::distributed::cpu_reduce::CpuReduceClient>,
    control_tx: mpsc::Sender<ControlMsg>,
    timing_tx: mpsc::Sender<TimingMsg>,
    gamma: f64,
) {
    use crate::distributed::ddp_run::{AveragedParams, ParamSnapshot};
    let Some(mut client) = cpu_client else {
        // Discard mode (NCCL-only worker).
        while param_rx.recv().is_ok() {}
        return;
    };
    // ESCAPE HATCH (same contract as the inbound bridge): a mid-round error
    // here means this rank can no longer participate in CPU averaging — the
    // inner GpuWorker would otherwise wait at the reduce barrier for an
    // `Update` that will never come. Wake it with Shutdown so the rank exits
    // (checkpoint drain + final snapshot + Exiting) instead of parking
    // forever; the coordinator's heartbeat staleness then reaps it.
    let inject_shutdown = || {
        let _ = control_tx.send(ControlMsg::Shutdown);
    };
    // Monotonic local version counter; bumped per round so the
    // synthesized AveragedParams.version increases consistently.
    let mut version: u64 = 0;
    // Pre-sync scratch for weight-space divergence math. Allocated
    // lazily on the first ParamSnapshot (shapes match the inner
    // GpuWorker's param tensors; reused unchanged across rounds).
    let mut pre_scratch: Option<Vec<Tensor>> = None;

    while let Ok(snapshot) = param_rx.recv() {
        let ParamSnapshot {
            rank: snap_rank,
            params,
            buffers,
            batch_count: n_i,
        } = snapshot;
        debug_assert_eq!(
            snap_rank as u64, rank,
            "param bridge: snapshot.rank mismatch with bridge rank"
        );

        // One-time scratch allocation matched to the snapshot shapes.
        // Explicitly f32 (NOT zeros_like): under `bf16_wire` the snapshot
        // tensors are bf16, and the divergence math mutates the scratch
        // in place (`foreach_add_list_`), where a bf16 output cannot
        // absorb the f32 consensus. The `copy_` below upcasts bf16 →
        // f32 exactly, so the triple keeps full precision either way.
        if pre_scratch.is_none() {
            let allocated: Result<Vec<Tensor>> = params
                .iter()
                .map(|t| {
                    Tensor::zeros(
                        &t.shape(),
                        crate::tensor::TensorOptions {
                            dtype: crate::tensor::DType::Float32,
                            device: crate::tensor::Device::CPU,
                        },
                    )
                })
                .collect();
            match allocated {
                Ok(s) => pre_scratch = Some(s),
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} scratch alloc: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            }
        }
        let scratch = pre_scratch.as_ref().expect("scratch just allocated");

        // Capture pre-sync params into scratch (deep copy; scratch
        // never shares storage with snapshot.params, so the math
        // stays correct regardless of device or ApplyPolicy).
        let mut copy_failed = false;
        for (dst, src) in scratch.iter().zip(params.iter()) {
            if let Err(e) = dst.copy_(src, false) {
                eprintln!(
                    "cluster_worker: param bridge r{rank} pre_scratch copy_: {e}"
                );
                copy_failed = true;
                break;
            }
        }
        if copy_failed {
            inject_shutdown();
            return;
        }

        // Emit SnapshotReady BEFORE entering the AllReduce barrier so
        // the coord's per-rank capacity signal (T_ready - T_request)
        // measures snapshot + upload only, NOT polluted by slowest-
        // rank barrier wait. Failure to send is non-fatal — channel
        // closed means the coord-side bridge is gone, and the next
        // op will surface the real error.
        let _ = timing_tx.send(TimingMsg::SnapshotReady {
            rank: rank as usize,
        });

        // Realized-work weighted AllReduce. Each rank scales its
        // contribution by the work it did since the last sync (params:
        // batch_count `n_i` raised to gamma; buffers: a 0/1 mover
        // indicator) and ships the mass IN THE SAME FRAME — the
        // controller divides the summed contributions ONCE by the mass
        // of exactly the frames it accepted, so the divisor can never
        // disagree with the sum and the scattered frame IS the
        // consensus (also what the forge writes and the outer optimizer
        // steps). Sum stays associative, so it composes with a future
        // per-host partial sum (the relay sum-and-count) without
        // averaging-of-averages. A rank that did 0 steps still holds
        // the previous consensus, so it contributes zero mass — but it
        // STILL joins every collective, so no rank stalls. Model frames
        // ride the client's wire dtype (f32, or bf16 under
        // `ElCheConfig::bf16_wire` — the client casts whatever staging
        // dtype the snapshot arrived in, so the frame schema stays
        // uniform across the cohort).
        //
        // The count gather is the all-idle skip signal (and schedule
        // telemetry): every rank learns Σ n_i and makes the identical
        // enter-or-skip decision. The authoritative divisor is NOT
        // taken from here — it rides each reduce frame.
        let world = client.world_size() as usize;
        let mut counts = vec![0.0f64; world];
        if (rank as usize) < world {
            counts[rank as usize] = n_i as f64;
        }
        if let Err(e) = client.all_reduce_per_rank_f64(&mut counts) {
            eprintln!("cluster_worker: param bridge r{rank} count gather: {e}");
            inject_shutdown();
            return;
        }
        let total_n: f64 = counts.iter().sum();

        // All-idle (no rank moved since the last sync — every snapshot
        // already equals the consensus): skip the reduce, leave params /
        // buffers unchanged. Every rank sees the same gathered `total_n ==
        // 0`, so the skip is collective-consistent (no rank left waiting in
        // an all_reduce its peers did not call).
        // Gamma allocation-weighting: rank k is weighted nₖ^γ (γ=1.0 = plain
        // work-weighting, byte-identical to pre-gamma; γ<1 compresses the
        // fast/over-allocated rank's dominance, γ=0 = unweighted average,
        // γ<0 = per-step-equal). Idle ranks (nₖ=0) contribute zero mass for
        // any γ (the idle guard in `realized_work`). No local normalizer:
        // the divisor rides the frames — the controller divides once by
        // the mass it accepted. Only the params consensus is
        // gamma-weighted; buffers stay equal-weighted among movers.
        let my_w = crate::distributed::realized_work::gamma_mass(n_i as f64, gamma);
        // All-idle keep-local: the snapshot clones go back down as the
        // "consensus". Under bf16 staging that re-adopts bf16-quantized
        // params — idempotent after the first sync (adopted consensus
        // values are bf16-representable, so re-quantizing is a no-op);
        // only an EASGD-blended state could pick up one ~2⁻⁹-relative
        // rounding on an (already rare) all-idle round.
        let avg_params = if total_n == 0.0 {
            params.clone()
        } else {
            match sumcount_reduce(&mut client, &params, my_w) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} all_reduce params: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            }
        };

        // Weight-space divergence (||pre - post|| / ||post||, plus
        // pre_norm / post_norm) computed before the buffer reduce so
        // a later buffer error path can't mask the params triple.
        let (divergence, post_norm, pre_norm) =
            match crate::distributed::divergence::divergence_triple(scratch, &avg_params) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} divergence: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            };

        // Buffers (BatchNorm running stats etc.): equal weight among the
        // ranks that moved (idle excluded via the 0/1 indicator); the
        // controller divides by the accepted mover count.
        //
        // Only the f32 subset rides the reduce — the CPU transport is
        // f32-only, and non-f32 buffers (integer counters and the like)
        // are deterministic values updated identically on every rank,
        // so passing the local value through unchanged is correct, not
        // a dropped sync (mirrors the bootstrap broadcast's filter in
        // `rank_entry`). The merge is positional, so the consumer's
        // zip against the live buffer list stays aligned; every rank
        // builds the same model, so the subset matches in count/order
        // across ranks and the collective stays balanced.
        let f32_buffer_idx: Vec<usize> = buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.dtype() == crate::tensor::DType::Float32)
            .map(|(i, _)| i)
            .collect();
        let avg_buffers = if f32_buffer_idx.is_empty() || total_n == 0.0 {
            buffers.clone()
        } else {
            let subset: Vec<Tensor> =
                f32_buffer_idx.iter().map(|&i| buffers[i].clone()).collect();
            let my_indicator = crate::distributed::realized_work::mover_mass(n_i as f64);
            match sumcount_reduce(&mut client, &subset, my_indicator) {
                Ok(reduced) => {
                    let mut merged = buffers.clone();
                    for (k, &i) in f32_buffer_idx.iter().enumerate() {
                        merged[i] = reduced[k].clone();
                    }
                    merged
                }
                Err(e) => {
                    eprintln!(
                        "cluster_worker: param bridge r{rank} all_reduce buffers: {e}"
                    );
                    inject_shutdown();
                    return;
                }
            }
        };
        version += 1;
        let avg = AveragedParams {
            params: avg_params,
            buffers: avg_buffers,
            version,
        };
        if control_tx.send(ControlMsg::Update(avg)).is_err() {
            // Inner GpuWorker dropped its receiver; tear down.
            return;
        }
        // Ack the coordinator. CPU re-arm runs off the coord's
        // `cpu_avg_state` machine (finalized by `poll_cpu_averaging` once
        // every rank's divergence has landed), NOT off `step_count`, so
        // this ack carries no synthetic step. A real step_count isn't
        // available here anyway — the inner GpuWorker doesn't bump
        // `local_step` on `RequestParams`. Sending 0 keeps the coord's
        // `last_step_count` clean (it ignores CPU-path step_counts). The
        // previous `usize::MAX / 2` sentinel poisoned `last_step_count`,
        // wedging the NCCL-style re-arm gate after a few cycles.
        let _ = timing_tx.send(TimingMsg::SyncAck {
            rank: rank as usize,
            step_count: 0,
            divergence: Some(divergence),
            post_norm,
            pre_norm,
        });
    }
    // Channel closed (clean training end): emit the accumulated reduce
    // profile so the cpu-cadence reduce floor can be attributed to
    // serialize / wire / deserialize. One line per rank, on stderr.
    client.log_profile_summary();
}

/// Sum-and-count weighted AllReduce over the CPU data channel: returns
/// `Σ_r (w_r · T_r) / total_weight` per tensor, computed as a plain Sum
/// followed by a SINGLE divide.
///
/// `CpuReduceClient::all_reduce_tensors` implements the avg-trick (the
/// controller sums every rank's contribution then divides by
/// `world_size`), so this pre-multiplies each rank's scaled contribution
/// by `world_size` to recover a plain Sum, reduces, then divides once by
/// `total_weight`. Deferring the divide to a single final step (rather
/// than forming `w_r/total` before the reduce) keeps the operation
/// associative: a future per-host partial sum can fold local ranks into
/// one `(Σ w·T, Σ w)` pair and the root still divides exactly once — no
/// averaging-of-averages. A zero-weight rank contributes a zeroed tensor
/// but still joins the collective, so the cohort never stalls.
/// Realized-work weighted reduce: ship `my_weight`-scaled tensors with
/// the mass riding the SAME frame, so the controller's divisor is the
/// summed mass of exactly the contributions it accepted into the round
/// — the sum and its divisor can never disagree, whatever the cohort
/// did between rounds. The controller divides ONCE and scatters the
/// consensus, which is therefore also what the checkpoint forge writes
/// and what the outer optimizer steps.
///
/// Returns the adopted tensors: the scattered consensus, or shallow
/// clones of `tensors` when the round realized no work (returned mass
/// `0.0`, e.g. every accepted contributor was idle) — the caller keeps
/// its local state, mirroring the all-idle collective skip.
pub(crate) fn sumcount_reduce(
    client: &mut crate::distributed::cpu_reduce::CpuReduceClient,
    tensors: &[Tensor],
    my_weight: f64,
) -> Result<Vec<Tensor>> {
    let scaled: Vec<Tensor> = tensors
        .iter()
        .map(|t| t.mul_scalar(my_weight))
        .collect::<Result<_>>()?;
    // Ownership handoff: the scaled scratch (a whole model copy on the
    // params reduce) is freed the moment the wire frame is encoded,
    // instead of sitting live across the barrier on every rank at once.
    let (consensus, realized) = client.all_reduce_weighted_owned(
        scaled,
        crate::distributed::controller::RoundKind::Model,
        my_weight,
    )?;
    if !crate::distributed::realized_work::is_realized(realized) {
        crate::debug!(
            "cluster_worker: reduce round realized no work (mass 0); keeping local state"
        );
        return Ok(tensors.to_vec());
    }
    Ok(consensus)
}
