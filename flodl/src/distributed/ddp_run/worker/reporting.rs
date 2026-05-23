//! Reporting and observability hooks: aggregated_metrics, report_*, compute_param_norm, send_final_snapshot, abort_nccl.

use std::sync::{Arc, Mutex};

use crate::nn::Module;
use crate::tensor::{Result, Tensor, TensorError};

use super::super::{
    ControlMsg, EpochMetrics,
    MetricsMsg, TimingMsg,
};
use super::GpuWorker;

impl<M: Module> GpuWorker<M> {
    /// Shared handle to the most recent aggregated [`EpochMetrics`]
    /// broadcast from the coord. Clone this `Arc` so the user's
    /// `Graph` (or any other reader) sees updates as they arrive
    /// without coupling through the worker. Returns `None` inside the
    /// mutex until the coord has aggregated at least one epoch.
    pub fn aggregated_metrics(&self) -> Arc<Mutex<Option<EpochMetrics>>> {
        Arc::clone(&self.aggregated_metrics)
    }

    /// Drain any queued `Shutdown` / `ShutdownWithSave` messages from
    /// `control_rx` and process them. Called from `ClusterWorker`'s
    /// teardown path so that — when the worker exits the main loop
    /// with an error (e.g. lone NCCL survivor bailing out of
    /// `wait_for_nccl_session`) — any pending coord-sent
    /// `ShutdownWithSave` still gets handled and the rank-side
    /// checkpoint bundle gets written. Non-shutdown messages in the
    /// queue are dropped (the worker is on its way out).
    ///
    /// Returns `true` if a shutdown frame was processed.
    pub fn drain_pending_shutdown(&mut self) -> bool {
        let mut handled = false;
        while let Ok(msg) = self.control_rx.try_recv() {
            match msg {
                ControlMsg::ShutdownWithSave { reason } => {
                    if let Some(stem) = self.save_path.clone() {
                        self.write_checkpoint_bundle(&stem, reason);
                    } else {
                        crate::verbose!(
                            "  ddp-worker: rank {} drain_pending_shutdown saw \
                             ShutdownWithSave but save_path is unset; \
                             exiting without saving",
                            self.rank,
                        );
                    }
                    handled = true;
                }
                ControlMsg::Shutdown => {
                    handled = true;
                }
                _ => {
                    // Drop — worker is exiting, other messages are stale.
                }
            }
        }
        handled
    }

    /// Report the wall-time the rank just spent inside `epoch_fn`. Called
    /// by the cluster worker's main loop on the role rank after firing
    /// the user closure, so the coord can time-exclude callback cost from
    /// `wall_ms_accum[rank]` and update `last_epoch_fn_elapsed_ms_ewma`.
    /// Fire-and-forget: a disconnected timing channel is non-fatal here
    /// (the loop is exiting anyway).
    pub fn report_epoch_fn_elapsed(&self, epoch: usize, elapsed_ms: f64) {
        let _ = self.timing_tx.send(TimingMsg::EpochFnElapsed {
            rank: self.rank,
            epoch,
            elapsed_ms,
        });
    }

    /// Send a timing report to the coordinator.
    ///
    /// Also emits a [`TimingMsg::LrUpdate`] piggyback message so the
    /// coordinator's LR-aware meta-controller (when enabled) can track the
    /// LR trajectory between averaging cycles. Cheap fire-and-forget; the
    /// coordinator caches only the most recent value per rank.
    pub fn report_timing(
        &self,
        batch_ms: f64,
        param_norm: Option<f64>,
        batch_loss: f64,
        sync_divergence: Option<f64>,
    ) -> Result<()> {
        let res = self.timing_tx.send(TimingMsg::Batch {
            rank: self.rank,
            batch_ms,
            step_count: self.local_step,
            param_norm,
            batch_loss,
            sync_divergence,
        }).map_err(|_| TensorError::new("timing channel disconnected"));
        // Piggyback the current LR after the primary Batch. Failures are
        // tolerated — the meta layer simply observes a stale value next cycle.
        let _ = self.timing_tx.send(TimingMsg::LrUpdate {
            rank: self.rank,
            lr: self.optimizer.lr(),
        });
        res
    }

    /// Compute the L2 norm of all model parameters.
    ///
    /// Uses `Tensor::foreach_norm` for a single batched CUDA kernel instead
    /// of per-parameter norm calls. Returns the global L2 norm (sqrt of sum
    /// of squared per-tensor norms). Used for NCCL divergence detection.
    pub(super) fn compute_param_norm(&self) -> Result<f64> {
        let data: Vec<Tensor> = self.param_vars.iter().map(|v| v.data()).collect();
        if data.is_empty() {
            return Ok(0.0);
        }
        let norms = Tensor::foreach_norm(&data, 2.0)?;
        let mut total_sq = 0.0f64;
        for n in &norms {
            let val: f64 = n.item()?;
            total_sq += val * val;
        }
        Ok(total_sq.sqrt())
    }

    /// Send the final parameter snapshot on the dedicated channel before exiting.
    ///
    /// This uses `final_param_tx` (not `param_tx`) to avoid racing with
    /// CPU averaging snapshot collection on the same channel.
    pub fn send_final_snapshot(&self) {
        let _ = self.final_param_tx.send(self.snapshot_params());
    }

    /// Abort the NCCL communicator, unblocking any stuck collective.
    ///
    /// Must be called before [`Self::send_final_snapshot`] when the training loop
    /// exits due to shutdown. A pending AllReduce on `comm_stream` (from a
    /// SyncNow whose peer died) would block `to_device(CPU)` in snapshot_params
    /// because the CUDA default stream synchronizes with all other streams.
    pub fn abort_nccl(&mut self) {
        if let Some(comm) = self.nccl_comm.take() {
            let _ = comm.abort_handle().abort();
        }
    }

    /// Notify the coordinator that this worker is about to exit.
    ///
    /// Must be called before the thread terminates so the coordinator
    /// stops including this rank in NCCL collectives.
    pub fn report_exiting(&self) {
        let _ = self.timing_tx.send(TimingMsg::Exiting { rank: self.rank });
    }

    /// Send epoch-end metrics to the coordinator.
    ///
    /// Drains the thread-local scalar accumulator populated by
    /// [`record_scalar()`](super::super::record_scalar) calls during this epoch.
    pub fn report_epoch(
        &self,
        avg_loss: f64,
        batches: usize,
        epoch_ms: f64,
        share_complete_ms: f64,
        compute_only_ms: f64,
        data_starve_ms: f64,
    ) -> Result<()> {
        let scalars = super::super::drain_scalars();
        self.metrics_tx.send(MetricsMsg {
            rank: self.rank,
            epoch: self.current_epoch,
            avg_loss,
            batches_processed: batches,
            epoch_ms,
            samples_processed: batches * self.batch_size,
            share_complete_ms,
            compute_only_ms,
            data_starve_ms,
            scalars,
        }).map_err(|_| TensorError::new("metrics channel disconnected"))
    }
}
