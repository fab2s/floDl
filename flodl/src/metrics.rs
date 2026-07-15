//! Aggregated training-metrics vocabulary.
//!
//! A leaf module by design: [`EpochMetrics`] is plain data (no tensors,
//! no handles) consumed across the stack — the `nn` [`Module`] trait
//! exposes a shared slot for it, [`Graph`] owns such a slot for its
//! monitor integration, the distributed coordinator aggregates into it,
//! and [`Monitor`] renders it. Housing it here keeps those consumers
//! from depending on each other just to name the type. The historical
//! paths (`flodl::EpochMetrics`, `flodl::distributed::EpochMetrics`)
//! remain valid re-exports.
//!
//! [`Module`]: crate::nn::Module
//! [`Graph`]: crate::graph::Graph
//! [`Monitor`]: crate::monitor::Monitor

use std::collections::HashMap;

/// Aggregated epoch metrics from all DDP workers.
///
/// Available via `DdpHandle::poll_metrics()`, `DdpHandle::next_metrics()`,
/// and the host-side [`crate::distributed::DdpBuilder::metrics_fn`] callback.
/// The coordinator aggregates per-rank metrics into this structure once
/// all ranks have reported for the same epoch; the same `EpochMetrics` reaches
/// the callback (if registered) and the polling queue, so both surfaces compose.
///
/// # Example: explicit polling
///
/// ```ignore
/// let handle = Trainer::builder(...).run()?;
/// while let Some(m) = handle.next_metrics() {
///     for (name, value) in &m.scalars {
///         monitor.record_scalar(name, *value);
///     }
/// }
/// let state = handle.join()?;
/// ```
///
/// # Example: chained `.run()?.join()?` with `metrics_fn`
///
/// ```ignore
/// Trainer::builder(model_factory, optim_factory, train_step)
///     .dataset(dataset).batch_size(32).num_epochs(N)
///     .metrics_fn(move |m| {
///         println!("epoch {}: loss={:.4}", m.epoch, m.avg_loss);
///         Ok(())
///     })
///     .run()?
///     .join()?;
/// ```
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    /// Epoch number (0-based).
    pub epoch: usize,
    /// Weighted-average scalar metrics across all ranks.
    /// Each value is the batch-weighted mean.
    pub scalars: HashMap<String, f64>,
    /// Per-rank scalar metrics (index = rank).
    pub per_rank: Vec<HashMap<String, f64>>,
    /// Average loss across all ranks (batch-weighted).
    pub avg_loss: f64,
    /// Wall-clock epoch time (ms), max across ranks.
    pub epoch_ms: f64,
    /// Per-rank throughput in samples/ms (index = rank). Computed from
    /// `share_complete_ms` (epoch start to end of rank's last batch),
    /// not from `epoch_ms`, to exclude post-completion sync-barrier idle.
    /// This is the honest capacity signal that the balancer should consume.
    pub per_rank_throughput: Vec<f64>,
    /// Per-rank batch share as fraction 0.0..1.0 (index = rank).
    pub per_rank_batch_share: Vec<f64>,
    /// Per-rank time on assigned work (ms), from epoch start to last batch
    /// finishing. Excludes post-completion sync wait. Source of `per_rank_throughput`.
    pub per_rank_share_complete_ms: Vec<f64>,
    /// Per-rank pure compute time (ms): sum of train_step durations.
    /// Diagnostic only; not used by the balancer.
    pub per_rank_compute_only_ms: Vec<f64>,
    /// Per-rank cumulative data-wait time (ms): time blocked waiting for
    /// the next batch. Diagnostic for prefetch tuning; not a balancer input.
    pub per_rank_data_starve_ms: Vec<f64>,
    /// CUDA device index per rank (for dashboard GPU tabs).
    pub device_indices: Vec<u8>,
}
