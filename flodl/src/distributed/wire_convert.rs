//! One home for every in-process message ↔ wire conversion.
//!
//! The domain messages (`TimingMsg`, `ControlMsg`, `MetricsMsg`,
//! `EpochMetrics`) travel the in-process mpsc channels; their `*Wire`
//! twins (in [`crate::distributed::wire`]) travel the TCP control/timing/
//! metrics frames. The mapping used to be smeared across `cluster_worker`,
//! `ddp_run`, and scattered `From` impls, so every new message variant
//! touched several files. Keeping all of it here means one place to read
//! the protocol and one place to edit when a message changes.

use crate::distributed::ddp_run::{ControlMsg, EpochMetrics, EpochPlan, MetricsMsg, TimingMsg};
use crate::distributed::wire::{ControlMsgWire, EpochMetricsWire, MetricsMsgWire, TimingMsgWire};
use crate::tensor::{Result, TensorError};

/// Convert an in-process `TimingMsg` into the bincode-serializable
/// [`TimingMsgWire`] for transit over the TCP timing channel.
pub(crate) fn timing_msg_to_wire(msg: TimingMsg) -> TimingMsgWire {
    match msg {
        TimingMsg::Batch {
            rank,
            batch_ms,
            data_ms,
            step_count,
            param_norm,
            batch_loss,
            sync_divergence,
        } => TimingMsgWire::Batch {
            rank: rank as u64,
            batch_ms,
            data_ms,
            step_count: step_count as u64,
            param_norm,
            batch_loss,
            sync_divergence,
        },
        TimingMsg::SyncAck {
            rank,
            step_count,
            divergence,
            post_norm,
            pre_norm,
        } => TimingMsgWire::SyncAck {
            rank: rank as u64,
            step_count: step_count as u64,
            divergence,
            post_norm,
            pre_norm,
        },
        TimingMsg::Exiting { rank } => TimingMsgWire::Exiting { rank: rank as u64 },
        TimingMsg::LrUpdate { rank, lr } => TimingMsgWire::LrUpdate {
            rank: rank as u64,
            lr,
        },
        TimingMsg::Heartbeat { rank, step_count } => TimingMsgWire::Heartbeat {
            rank: rank as u64,
            step_count: step_count as u64,
        },
        TimingMsg::Intent { rank, kind } => TimingMsgWire::Intent {
            rank: rank as u64,
            kind,
        },
        TimingMsg::SnapshotReady { rank } => TimingMsgWire::SnapshotReady { rank: rank as u64 },
        TimingMsg::NewNcclIdGenerated { rank, uid_bytes } => TimingMsgWire::NewNcclIdGenerated {
            rank: rank as u64,
            uid_bytes,
        },
        TimingMsg::EvalResult {
            rank,
            schedule_id,
            epoch,
            metric,
            elapsed_ms,
            error,
        } => TimingMsgWire::EvalResult {
            rank: rank as u64,
            schedule_id,
            epoch,
            metric,
            elapsed_ms,
            error,
        },
        TimingMsg::CheckpointResult {
            rank,
            version,
            elapsed_ms,
            error,
        } => TimingMsgWire::CheckpointResult {
            rank: rank as u64,
            version,
            elapsed_ms,
            error,
        },
        TimingMsg::EpochFnElapsed {
            rank,
            epoch,
            elapsed_ms,
        } => TimingMsgWire::EpochFnElapsed {
            rank: rank as u64,
            epoch: epoch as u64,
            elapsed_ms,
        },
        TimingMsg::GraphProfile { rank, profile } => TimingMsgWire::DashboardGraphTimings {
            rank: rank as u64,
            profile,
        },
    }
}

/// Convert an inbound [`ControlMsgWire`] from the coordinator into an
/// optional in-process `ControlMsg` for [`GpuWorker::dispatch_control`](
/// crate::distributed::ddp_run::GpuWorker).
///
/// Returns `Ok(None)` for wire variants that don't need in-process
/// dispatch:
///
/// - `ControlMsgWire::Update { version, next_plan }`: the wire-side
///   notification that the averaging cycle is complete. The real
///   in-process `ControlMsg::Update(AveragedParams)` flows through the
///   param bridge (where the param bridge synthesizes one with the
///   actual averaged tensors from the data channel), so the wire-Update
///   is informational here. Its atomic-dispatch `next_plan` (when
///   `Some`) is consumed at the inbound-bridge call site, which
///   synthesises a `StartEpoch` for the inner; it is not handled by
///   this function.
///
/// All other wire variants map 1:1.
pub(crate) fn control_wire_to_msg(wire: ControlMsgWire) -> Result<Option<ControlMsg>> {
    match wire {
        ControlMsgWire::RequestParams => Ok(Some(ControlMsg::RequestParams)),
        // Informational on its own; the param bridge drives the real
        // `ControlMsg::Update(AveragedParams)`. The atomic-dispatch
        // `next_plan` is handled at the inbound-bridge call site (it
        // synthesises a `StartEpoch` there), so it never reaches here in
        // production; ignored for the rare direct callers (tests).
        ControlMsgWire::Update { .. } => Ok(None),
        ControlMsgWire::SyncNow => Ok(Some(ControlMsg::SyncNow)),
        ControlMsgWire::StartEpoch(plan) => Ok(Some(ControlMsg::StartEpoch(EpochPlan {
            epoch: plan.epoch as usize,
            partition_offset: plan.partition_offset as usize,
            partition_size: plan.partition_size as usize,
        }))),
        ControlMsgWire::ExtendPartition {
            partition_offset,
            partition_size,
        } => Ok(Some(ControlMsg::ExtendPartition {
            partition_offset: partition_offset as usize,
            partition_size: partition_size as usize,
        })),
        ControlMsgWire::DeclareDead { .. } => Ok(Some(ControlMsg::DeclareDead)),
        ControlMsgWire::NewNcclSession { .. } => Ok(Some(ControlMsg::NewNcclSession)),
        ControlMsgWire::RequestNewNcclId => Ok(Some(ControlMsg::RequestNewNcclId)),
        ControlMsgWire::StageAdvisory { counts, segments } => Ok(Some(ControlMsg::StageAdvisory {
            counts: counts.into_iter().map(|c| c as usize).collect(),
            segments: segments
                .into_iter()
                .map(|(epoch, spans)| {
                    (
                        epoch as usize,
                        spans
                            .into_iter()
                            .map(|(o, s)| (o as usize, s as usize))
                            .collect(),
                    )
                })
                .collect(),
        })),
        ControlMsgWire::Throttle => Ok(Some(ControlMsg::Throttle)),
        ControlMsgWire::SetGlobalStep { global_step } => {
            Ok(Some(ControlMsg::SetGlobalStep(global_step as usize)))
        }
        ControlMsgWire::Checkpoint {
            version,
            target_rank,
        } => {
            // `u64::MAX` is reserved for v2 controller-as-checkpointer
            // (CPU-async mode where the coord holds the canonical
            // averaged tensors). In v1 the coord must never dispatch
            // it; if a buggy/future coord does, surface loudly so we
            // don't silently fall through to "no-op for every rank".
            if target_rank == u64::MAX {
                return Err(TensorError::new(
                    "cluster_worker: Checkpoint target_rank=u64::MAX is reserved \
                     for controller-as-checkpointer (v2); v1 must dispatch to a \
                     worker rank ID",
                ));
            }
            Ok(Some(ControlMsg::Checkpoint {
                version,
                target_rank: target_rank as usize,
            }))
        }
        ControlMsgWire::ExecuteEvalCallback {
            schedule_id,
            epoch,
            target_rank,
            adopt_consensus,
        } => {
            if target_rank == u64::MAX {
                return Err(TensorError::new(
                    "cluster_worker: ExecuteEvalCallback target_rank=u64::MAX \
                     is reserved (controller-as-evaluator, future); v1 must \
                     dispatch to a worker rank ID",
                ));
            }
            Ok(Some(ControlMsg::ExecuteEvalCallback {
                schedule_id,
                epoch,
                target_rank: target_rank as usize,
                adopt_consensus,
            }))
        }
        ControlMsgWire::ArmConsensusEval {
            schedule_id,
            epoch,
            target_rank,
        } => Ok(Some(ControlMsg::ArmConsensusEval {
            schedule_id,
            epoch,
            target_rank: target_rank as usize,
        })),
        ControlMsgWire::SetEpochCallbackRole { rank } => {
            Ok(Some(ControlMsg::SetEpochCallbackRole {
                rank: rank as usize,
            }))
        }
        ControlMsgWire::Shutdown => Ok(Some(ControlMsg::Shutdown)),
        ControlMsgWire::ShutdownWithSave { reason } => {
            // Forward-compat: unknown reason byte falls back to
            // GracefulShutdown so a newer coord doesn't crash older
            // workers. The save still happens; only the recorded
            // reason loses fidelity.
            let reason = crate::distributed::checkpoint_meta::SaveReason::from_u8(reason)
                .unwrap_or(crate::distributed::checkpoint_meta::SaveReason::GracefulShutdown);
            Ok(Some(ControlMsg::ShutdownWithSave { reason }))
        }
        ControlMsgWire::EpochAggregated(metrics_wire) => Ok(Some(ControlMsg::EpochAggregated(
            Box::new((*metrics_wire).into()),
        ))),
        ControlMsgWire::EvalBroadcast { epoch, metric } => Ok(Some(ControlMsg::EvalBroadcast {
            epoch: epoch as usize,
            metric,
        })),
        ControlMsgWire::SaveConsensusModel { target_rank } => {
            if target_rank == u64::MAX {
                return Err(TensorError::new(
                    "cluster_worker: SaveConsensusModel target_rank=u64::MAX is \
                     reserved; the coordinator must dispatch to a worker rank ID",
                ));
            }
            Ok(Some(ControlMsg::SaveConsensusModel {
                target_rank: target_rank as usize,
            }))
        }
        // Pure liveness beacon: the inbound bridge intercepts it before this
        // point (resetting its coord-liveness deadline) and never forwards it
        // to the inner worker. Reached only by direct callers (tests); no
        // inner dispatch, like `Update { .. }`.
        ControlMsgWire::CoordHeartbeat => Ok(None),
    }
}

/// Convert an in-process [`MetricsMsg`] into wire-compatible
/// [`MetricsMsgWire`] for transit over the metrics-channel TCP frame.
pub(crate) fn metrics_msg_to_wire(msg: MetricsMsg) -> MetricsMsgWire {
    MetricsMsgWire {
        rank: msg.rank as u64,
        epoch: msg.epoch as u64,
        avg_loss: msg.avg_loss,
        batches_processed: msg.batches_processed as u64,
        epoch_ms: msg.epoch_ms,
        samples_processed: msg.samples_processed as u64,
        share_complete_ms: msg.share_complete_ms,
        compute_only_ms: msg.compute_only_ms,
        data_starve_ms: msg.data_starve_ms,
        scalars: msg
            .scalars
            .into_iter()
            .map(|(k, (sum, count))| (k, (sum, count as u64)))
            .collect(),
        // Populated by the dashboard-aware emit path when the launcher
        // hosts a dashboard; the plain wire conversion leaves it None
        // and lets the worker layer (which holds the ResourceSampler)
        // attach a sample before writing.
        resources: None,
    }
}

impl From<EpochMetrics> for EpochMetricsWire {
    fn from(m: EpochMetrics) -> Self {
        EpochMetricsWire {
            epoch: m.epoch as u64,
            scalars: m.scalars,
            per_rank: m.per_rank,
            avg_loss: m.avg_loss,
            epoch_ms: m.epoch_ms,
            per_rank_throughput: m.per_rank_throughput,
            per_rank_batch_share: m.per_rank_batch_share,
            per_rank_share_complete_ms: m.per_rank_share_complete_ms,
            per_rank_compute_only_ms: m.per_rank_compute_only_ms,
            per_rank_data_starve_ms: m.per_rank_data_starve_ms,
            device_indices: m.device_indices,
            per_rank_loss: m.per_rank_loss,
            per_rank_samples: m.per_rank_samples.iter().map(|&s| s as u64).collect(),
        }
    }
}

impl From<EpochMetricsWire> for EpochMetrics {
    fn from(w: EpochMetricsWire) -> Self {
        EpochMetrics {
            epoch: w.epoch as usize,
            scalars: w.scalars,
            per_rank: w.per_rank,
            avg_loss: w.avg_loss,
            epoch_ms: w.epoch_ms,
            per_rank_loss: w.per_rank_loss,
            per_rank_samples: w.per_rank_samples.iter().map(|&s| s as usize).collect(),
            per_rank_throughput: w.per_rank_throughput,
            per_rank_batch_share: w.per_rank_batch_share,
            per_rank_share_complete_ms: w.per_rank_share_complete_ms,
            per_rank_compute_only_ms: w.per_rank_compute_only_ms,
            per_rank_data_starve_ms: w.per_rank_data_starve_ms,
            device_indices: w.device_indices,
        }
    }
}
