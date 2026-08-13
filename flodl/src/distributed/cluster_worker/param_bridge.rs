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
        //
        // Weight-space divergence (||pre - post|| / ||post||, plus
        // pre_norm / post_norm) streams THROUGH the reduce: the client
        // is armed with the pre-images (the snapshot staging) and folds
        // each (pre, consensus) pair into the accumulator at decode
        // time, holding one f32 tensor transient instead of the retired
        // model-sized `pre_scratch`. On the all-idle skip (no reduce at
        // all) the triple degenerates to (0, n, n) — pre == post — from
        // a norm-only pass. Computed before the buffer reduce so a
        // later buffer error path can't mask the params triple.
        let (avg_params, divergence, post_norm, pre_norm, realized) = if total_n == 0.0 {
            let n = match crate::distributed::divergence::exact_norm(&params) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("cluster_worker: param bridge r{rank} all-idle norm: {e}");
                    inject_shutdown();
                    return;
                }
            };
            (params.clone(), 0.0, n, n, false)
        } else {
            client.arm_divergence(&params);
            let (adopted, realized) = match sumcount_reduce(&mut client, &params, my_w) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("cluster_worker: param bridge r{rank} all_reduce params: {e}");
                    inject_shutdown();
                    return;
                }
            };
            let Some(accum) = client.take_divergence() else {
                eprintln!(
                    "cluster_worker: param bridge r{rank} divergence accumulator \
                     missing after armed reduce (protocol bug)"
                );
                inject_shutdown();
                return;
            };
            // Keep-local (zero realized mass): pre == post by
            // definition — the decoded payloads were meaningless, only
            // the pre sums are trusted.
            let (d, post_n, pre_n) = if realized {
                accum.finish()
            } else {
                accum.finish_keep_local()
            };
            (adopted, d, post_n, pre_n, realized)
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
            let subset: Vec<Tensor> = f32_buffer_idx.iter().map(|&i| buffers[i].clone()).collect();
            let my_indicator = crate::distributed::realized_work::mover_mass(n_i as f64);
            match sumcount_reduce(&mut client, &subset, my_indicator) {
                Ok((reduced, _realized)) => {
                    let mut merged = buffers.clone();
                    for (k, &i) in f32_buffer_idx.iter().enumerate() {
                        merged[i] = reduced[k].clone();
                    }
                    merged
                }
                Err(e) => {
                    eprintln!("cluster_worker: param bridge r{rank} all_reduce buffers: {e}");
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
            realized,
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
/// Returns `(adopted, realized)`: the scattered consensus and `true`,
/// or shallow clones of `tensors` and `false` when the round realized
/// no work (returned mass `0.0`, e.g. every accepted contributor was
/// idle) — the caller keeps its local state, mirroring the all-idle
/// collective skip, and `realized` tells a divergence-armed caller to
/// trust only the accumulator's pre sums (`finish_keep_local`).
///
/// When the client has decode-into-request enabled (barrier-paced CUDA
/// ranks — see [`CpuReduceClient::arm_decode_into`]), the reply decodes
/// into `tensors` THEMSELVES — the pinned snapshot staging, whose bytes
/// are dead once the streamed encode has read them — making the
/// writeback H2D truly async with zero marginal locked RAM; otherwise
/// into fresh allocs. On a zero-mass round the client leaves `tensors`
/// untouched (fresh decode), so the keep-local return below hands back
/// genuinely local state.
///
/// [`CpuReduceClient::arm_decode_into`]: crate::distributed::cpu_reduce::CpuReduceClient::arm_decode_into
pub(crate) fn sumcount_reduce(
    client: &mut crate::distributed::cpu_reduce::CpuReduceClient,
    tensors: &[Tensor],
    my_weight: f64,
) -> Result<(Vec<Tensor>, bool)> {
    // The `my_weight` pre-scale is FUSED into the streaming wire encode
    // (byte-level, per tensor) — no model-sized scaled scratch exists,
    // and reading `tensors` (the pinned snapshot staging) at stream time
    // is this window's single consumption of it.
    let refs: Vec<&Tensor> = tensors.iter().collect();
    client.arm_decode_into(tensors);
    let (consensus, realized) = client.all_reduce_scaled(
        &refs,
        my_weight,
        crate::distributed::controller::RoundKind::Model,
        my_weight,
    )?;
    if !crate::distributed::realized_work::is_realized(realized) {
        crate::debug!(
            "cluster_worker: reduce round realized no work (mass 0); keeping local state"
        );
        return Ok((tensors.to_vec(), false));
    }
    Ok((consensus, true))
}
