//! Training entry points for flodl.
//!
//! The primary entry point is [`Trainer`]. It works transparently on 1 or
//! N GPUs - single-device training has zero DDP overhead. Reach for
//! [`Trainer`] by default; drop to [`Ddp`] only when you need explicit
//! multi-GPU control.
//!
//! **Default** ([`Trainer::setup()`], [`Trainer::builder()`]): user-owned or
//! framework-owned training loop, transparent single/multi-GPU. Same API in
//! both cases.
//!
//! **Explicit multi-GPU** ([`Ddp::wrap()`]): manual control over gradient
//! sync and parameter broadcast for advanced patterns (GAN, RL, progressive).
//!
//! # Setup mode (user owns the loop)
//!
//! ```ignore
//! Trainer::setup(&model, |dev| build_model(dev), |p| Adam::new(p, 0.001))?;
//!
//! // Training loop is identical for 1 or N GPUs:
//! for (x, y) in &train_loader {
//!     let out = model.forward(&x)?;
//!     let loss = cross_entropy_loss(&out, &y)?;
//!     loss.backward()?;
//!     model.step()?;
//! }
//! ```
//!
//! # Builder mode (framework owns the loop)
//!
//! ```ignore
//! let handle = Trainer::builder(model_factory, optim_factory, train_fn)
//!     .dataset(dataset)
//!     .batch_size(32)
//!     .num_epochs(10)
//!     .run()?;
//!
//! let state = handle.join()?;
//! ```
//!
//! # Manual DDP
//!
//! ```ignore
//! let ddp = Ddp::wrap(&[&model0, &model1], &devices)?;
//! ddp.sync_params()?;
//! // ... custom forward/backward ...
//! ddp.all_reduce_gradients()?;
//! ```

use crate::autograd::Variable;
use crate::graph::Graph;
use crate::nn::{Buffer, Module, Optimizer, Parameter};
use super::cuda_event::CudaEvent;
use super::nccl::{NcclComms, NcclRankComm, ReduceOp};
use super::rendezvous::TcpRendezvous;
use super::ddp_run::{DdpBuilder, DdpHandle};
pub use super::el_che::ElChe;
use crate::tensor::{Device, Result, Tensor, TensorError};


/// Shared lock for serializing NCCL communicator creation across test modules.
/// NCCL init is a collective operation that deadlocks if two tests try to
/// create communicators simultaneously.
#[cfg(test)]
pub(crate) static NCCL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Default number of steps before the first rebalance.
pub(crate) const DEFAULT_CALIBRATION_STEPS: usize = 10;

/// How often to re-evaluate chunk ratios after calibration.
pub(crate) const DEFAULT_REBALANCE_INTERVAL: usize = 50;

/// EMA smoothing factor for throughput tracking (higher = more reactive).
const EMA_ALPHA: f64 = 0.3;

/// Minimum ratio any device can receive (prevents starving a GPU entirely).
const MIN_CHUNK_RATIO: f64 = 0.05;

// ---------------------------------------------------------------------------
// Internal distributed state (held by Graph)
// ---------------------------------------------------------------------------

/// Internal distributed state held by Graph when `distribute()` is called.
pub(crate) struct DistributedState {
    /// Model replicas for ranks 1..N (rank 0 is the Graph itself).
    pub replicas: Vec<Box<dyn Module>>,
    /// NCCL communicators (one per device).
    pub comms: NcclComms,
    /// All devices including rank 0.
    pub devices: Vec<Device>,
    /// Per-replica optimizers indexed by rank (including rank 0).
    pub optimizers: Vec<Box<dyn Optimizer>>,
    /// Chunk ratios for auto-balancing (sum = 1.0). Default: equal.
    pub chunk_ratios: Vec<f64>,
    /// Parameters matched across replicas: param_groups\[param_idx\]\[rank\].
    pub param_groups: Vec<Vec<Variable>>,
    /// Buffers matched across replicas: buffer_groups\[buf_idx\]\[rank\].
    pub buffer_groups: Vec<Vec<Buffer>>,

    // -- Auto-balancer state --

    /// Per-rank forward timing events from last forward pass: (start, end).
    /// Set by forward_distributed(), read by step().
    pub last_timing: Option<Vec<(CudaEvent, CudaEvent)>>,
    /// Shard sizes from last forward pass (for throughput calculation).
    pub last_shard_sizes: Vec<i64>,
    /// EMA throughput per rank (samples/ms). Zero until first measurement.
    pub ema_throughput: Vec<f64>,
    /// Number of completed training steps.
    pub step_count: usize,
    /// Steps of equal-split calibration before first rebalance.
    pub calibration_steps: usize,
    /// Steps between ratio recalculations after calibration.
    pub rebalance_interval: usize,

    // -- El Che cadence (heterogeneous DDP) --

    /// El Che cadence strategy. When Some, Graph uses per-device multi-batch
    /// forward instead of per-batch scatter. When None, existing scatter path.
    pub el_che: Option<ElChe>,
    /// Per-rank batch counts from the last El Che forward pass.
    /// Set by forward_distributed_el_che(), read by step().
    pub last_el_che_counts: Vec<usize>,
    /// Wall-clock time at end of last El Che AllReduce.
    pub last_el_che_sync: Option<std::time::Instant>,
    /// Maximum gradient norm for per-rank clipping in El Che mode.
    pub max_grad_norm: Option<f64>,
    /// Optional system timeline for high-frequency profiling.
    pub timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,
}

impl DistributedState {
    /// AllReduce-average gradients across all replicas.
    pub fn all_reduce_gradients(&self) -> Result<()> {
        for group in &self.param_groups {
            // Skip frozen parameters (no gradient on rank 0)
            if group[0].grad().is_none() {
                continue;
            }
            let grads: Vec<Tensor> = group
                .iter()
                .map(|v| v.grad().expect("gradient missing on replica"))
                .collect();
            let refs: Vec<&Tensor> = grads.iter().collect();
            self.comms.all_reduce(&refs, ReduceOp::Avg)?;
        }
        Ok(())
    }

    /// Broadcast buffers from rank 0 to all replicas (BatchNorm stats etc).
    pub fn sync_buffers(&self) -> Result<()> {
        for group in &self.buffer_groups {
            let tensors: Vec<Tensor> = group.iter().map(|b| b.get()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        Ok(())
    }

    /// Broadcast parameters and buffers from rank 0 to all replicas.
    pub fn sync_params(&self) -> Result<()> {
        for group in &self.param_groups {
            let tensors: Vec<Tensor> = group.iter().map(|v| v.data()).collect();
            let refs: Vec<&Tensor> = tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        self.sync_buffers()
    }

    /// Compute shard sizes from chunk ratios, guaranteeing they sum to batch_size.
    pub fn compute_shard_sizes(&self, batch_size: i64) -> Vec<i64> {
        let n = self.devices.len();
        let mut sizes = Vec::with_capacity(n);
        let mut remaining = batch_size;

        for i in 0..n {
            if i == n - 1 {
                // Last device gets whatever is left
                sizes.push(remaining);
            } else {
                let s = (batch_size as f64 * self.chunk_ratios[i]).round() as i64;
                let s = s.max(1).min(remaining - (n - i - 1) as i64); // leave at least 1 per remaining device
                sizes.push(s);
                remaining -= s;
            }
        }

        sizes
    }

    /// Number of devices.
    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Whether chunk ratios are meaningfully unequal (need weighted gradients).
    pub fn is_balanced(&self) -> bool {
        let first = self.chunk_ratios[0];
        self.chunk_ratios.iter().all(|r| (r - first).abs() < 1e-6)
    }

    /// AllReduce gradients with weighted averaging for unequal shard sizes.
    ///
    /// Each replica's gradient is scaled by `(shard_size / batch_size)` before
    /// AllReduce Sum, which produces the correct mean gradient regardless of
    /// how the batch was split.
    pub fn weighted_all_reduce_gradients(&self, batch_size: i64) -> Result<()> {
        for group in &self.param_groups {
            if group[0].grad().is_none() {
                continue;
            }
            let grads: Vec<Tensor> = group
                .iter()
                .enumerate()
                .map(|(rank, v)| {
                    let g = v.grad().expect("gradient missing on replica");
                    let weight = self.last_shard_sizes[rank] as f64 / batch_size as f64;
                    g.mul_scalar_(weight).ok();
                    g
                })
                .collect();
            let refs: Vec<&Tensor> = grads.iter().collect();
            self.comms.all_reduce(&refs, ReduceOp::Sum)?;
        }
        Ok(())
    }

    /// Read timing from last forward pass, update EMA throughput, and
    /// rebalance chunk ratios if it's time.
    ///
    /// Called from Graph::step() after gradient sync. Returns true if
    /// chunk ratios were updated this step.
    pub fn update_balance(&mut self) -> Result<bool> {
        self.step_count += 1;

        // Read timing events (set by forward_distributed)
        if let Some(timing) = self.last_timing.take() {
            for (rank, (start, end)) in timing.iter().enumerate() {
                let ms = CudaEvent::elapsed_time(start, end)?;
                if ms > 0.0 && self.last_shard_sizes[rank] > 0 {
                    let throughput = self.last_shard_sizes[rank] as f64 / ms as f64;
                    if self.ema_throughput[rank] == 0.0 {
                        // First measurement: initialize directly
                        self.ema_throughput[rank] = throughput;
                    } else {
                        self.ema_throughput[rank] =
                            EMA_ALPHA * throughput + (1.0 - EMA_ALPHA) * self.ema_throughput[rank];
                    }
                }
            }
        }

        // Check if it's time to rebalance
        let should_rebalance = if self.step_count == self.calibration_steps {
            true
        } else if self.step_count > self.calibration_steps {
            (self.step_count - self.calibration_steps) % self.rebalance_interval == 0
        } else {
            false
        };

        if should_rebalance {
            self.rebalance();
            return Ok(true);
        }

        Ok(false)
    }

    /// Recompute chunk_ratios proportional to EMA throughput.
    fn rebalance(&mut self) {
        let total: f64 = self.ema_throughput.iter().sum();
        if total <= 0.0 {
            return; // no data yet
        }

        let n = self.devices.len();
        let min_total = MIN_CHUNK_RATIO * n as f64;

        // Compute raw proportional ratios
        let mut ratios: Vec<f64> = self.ema_throughput.iter().map(|t| t / total).collect();

        // Clamp: no device below MIN_CHUNK_RATIO
        let mut deficit = 0.0;
        let mut unclamped = 0;
        for r in &mut ratios {
            if *r < MIN_CHUNK_RATIO {
                deficit += MIN_CHUNK_RATIO - *r;
                *r = MIN_CHUNK_RATIO;
            } else {
                unclamped += 1;
            }
        }

        // Redistribute deficit from unclamped devices proportionally
        if deficit > 0.0 && unclamped > 0 {
            let unclamped_total: f64 = ratios
                .iter()
                .filter(|&&r| r > MIN_CHUNK_RATIO + 1e-9)
                .sum();
            if unclamped_total > min_total {
                for r in &mut ratios {
                    if *r > MIN_CHUNK_RATIO + 1e-9 {
                        *r -= deficit * (*r / unclamped_total);
                        *r = r.max(MIN_CHUNK_RATIO);
                    }
                }
            }
        }

        // Normalize to sum exactly to 1.0
        let sum: f64 = ratios.iter().sum();
        if sum > 0.0 {
            for r in &mut ratios {
                *r /= sum;
            }
        }

        self.chunk_ratios = ratios;
    }

    /// Configure El Che cadence from a [`DdpConfig`].
    ///
    /// Creates an internal ElChe when enabled (max_anchor != Some(0)),
    /// seeds chunk_ratios from speed_hint if provided.
    pub(crate) fn configure_el_che(&mut self, config: &DdpConfig) {
        let n = self.devices.len();
        if n < 2 {
            return;
        }

        // max_anchor = Some(0) → disabled (traditional DDP)
        if config.max_anchor == Some(0) {
            self.el_che = None;
            return;
        }

        // Build ElChe with sensible defaults
        let anchor = 10; // initial anchor, auto-tunes from timing
        let mut el_che = ElChe::new(n, anchor);

        if let Some(target) = config.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }
        if let Some((slow_rank, ratio)) = config.speed_hint {
            el_che = el_che.with_speed_ratio(slow_rank, ratio);
            // Also seed chunk_ratios for the existing auto-balancer
            self.apply_speed_hint(slow_rank, ratio);
        }

        self.el_che = Some(el_che);
        self.max_grad_norm = config.max_grad_norm;
    }

    /// Seed chunk_ratios from a speed hint.
    fn apply_speed_hint(&mut self, slow_rank: usize, ratio: f64) {
        let n = self.devices.len();
        if slow_rank >= n {
            return;
        }
        let ratio = ratio.max(1.0);
        let mut weights = vec![ratio; n];
        weights[slow_rank] = 1.0;
        let total: f64 = weights.iter().sum();
        self.chunk_ratios = weights.iter().map(|w| w / total).collect();
    }
}

// ---------------------------------------------------------------------------
// Manual DDP coordinator
// ---------------------------------------------------------------------------

/// Manual DDP coordinator for multi-GPU gradient sync.
///
/// For complex training patterns (GAN, RL, progressive) where transparent
/// Graph-level DDP doesn't fit. Provides explicit control over parameter
/// broadcast and gradient averaging.
///
/// For standard training, use [`crate::graph::Graph::distribute`] instead.
pub struct Ddp {
    inner: DdpInner,
}

/// Variant-specific state for [`Ddp`].
///
/// Two variants during migration to the C-model (process-per-rank):
/// - `Group`: legacy single-process multi-rank (created by [`Ddp::wrap`]).
///   Holds an in-process [`NcclComms`] and N replicas worth of parameters /
///   buffers, where N = world_size on this single process.
/// - `Rank`: process-per-rank (created by [`Ddp::wrap_single`]). Holds a
///   single [`NcclRankComm`] joining a cross-process / cross-host group, plus
///   this process's slice of parameters / buffers.
///
/// Stage 3 of the multi-host DDP arc deletes the `Group` variant; the enum
/// collapses to one variant at that point.
enum DdpInner {
    Group {
        comms: NcclComms,
        devices: Vec<Device>,
        param_groups: Vec<Vec<Variable>>,
        buffer_groups: Vec<Vec<Buffer>>,
    },
    Rank {
        comms: NcclRankComm,
        device: Device,
        params: Vec<Variable>,
        buffers: Vec<Buffer>,
    },
}

impl Ddp {
    /// Wrap pre-created model replicas for manual DDP control (single-process,
    /// multiple ranks). Legacy path; use [`Ddp::wrap_single`] for the
    /// process-per-rank cluster mode.
    ///
    /// Models must have identical architecture (same parameter count/shapes).
    /// Each model should already reside on its target device.
    pub fn wrap(models: &[&dyn Module], devices: &[Device]) -> Result<Self> {
        if models.len() < 2 {
            return Err(TensorError::new("Ddp::wrap requires at least 2 models"));
        }
        if models.len() != devices.len() {
            return Err(TensorError::new(
                "Ddp::wrap: model count must match device count",
            ));
        }

        let comms = NcclComms::new(devices)?;

        // Match parameters across models
        let all_params: Vec<Vec<Parameter>> =
            models.iter().map(|m| m.parameters()).collect();
        let n_params = all_params[0].len();
        for (rank, params) in all_params.iter().enumerate().skip(1) {
            if params.len() != n_params {
                return Err(TensorError::new(&format!(
                    "Ddp: replica {} has {} parameters, expected {}",
                    rank,
                    params.len(),
                    n_params
                )));
            }
        }

        let mut param_groups = Vec::with_capacity(n_params);
        for pi in 0..n_params {
            let group: Vec<Variable> =
                all_params.iter().map(|p| p[pi].variable.clone()).collect();
            param_groups.push(group);
        }

        // Match buffers
        let all_buffers: Vec<Vec<Buffer>> =
            models.iter().map(|m| m.buffers()).collect();
        let n_buffers = all_buffers[0].len();
        let mut buffer_groups = Vec::with_capacity(n_buffers);
        for bi in 0..n_buffers {
            let group: Vec<Buffer> =
                all_buffers.iter().map(|b| b[bi].clone()).collect();
            buffer_groups.push(group);
        }

        Ok(Ddp {
            inner: DdpInner::Group {
                comms,
                devices: devices.to_vec(),
                param_groups,
                buffer_groups,
            },
        })
    }

    /// Wrap a single model replica joined to a cross-process NCCL group.
    ///
    /// Each process in the cluster calls this with its own model, its own
    /// CUDA device, its own global rank (typically from
    /// [`super::LocalCluster::my_rank`]), and the rendezvous's shared
    /// [`NcclUniqueId`](super::NcclUniqueId) (from
    /// [`super::LocalCluster::rendezvous`]). NCCL synchronizes the group
    /// internally via the UID handshake.
    ///
    /// In contrast to [`Ddp::wrap`], this owns exactly one replica per
    /// process — the runtime model used by `torchrun` and friends.
    ///
    /// Loud errors: `global_rank >= rdv.world_size()`. NCCL init failures
    /// propagate from [`NcclRankComm::init_rank`].
    pub fn wrap_single(
        model: &dyn Module,
        device: Device,
        global_rank: usize,
        rdv: &TcpRendezvous,
    ) -> Result<Self> {
        let world_size = rdv.world_size();
        if global_rank >= world_size {
            return Err(TensorError::new(&format!(
                "Ddp::wrap_single: global_rank {global_rank} >= world_size {world_size}"
            )));
        }
        if let Device::CUDA(idx) = device {
            crate::tensor::set_current_cuda_device(idx);
        }
        let comms = NcclRankComm::init_rank(global_rank, world_size, rdv.unique_id())?;

        let params: Vec<Variable> = model
            .parameters()
            .into_iter()
            .map(|p| p.variable)
            .collect();
        let buffers: Vec<Buffer> = model.buffers();

        Ok(Ddp {
            inner: DdpInner::Rank {
                comms,
                device,
                params,
                buffers,
            },
        })
    }

    /// Broadcast all parameters and buffers from rank 0 to all replicas.
    pub fn sync_params(&self) -> Result<()> {
        match &self.inner {
            DdpInner::Group {
                comms,
                param_groups,
                buffer_groups,
                ..
            } => {
                for group in param_groups {
                    let tensors: Vec<Tensor> = group.iter().map(|v| v.data()).collect();
                    let refs: Vec<&Tensor> = tensors.iter().collect();
                    comms.broadcast(&refs, 0)?;
                }
                for group in buffer_groups {
                    let tensors: Vec<Tensor> = group.iter().map(|b| b.get()).collect();
                    let refs: Vec<&Tensor> = tensors.iter().collect();
                    comms.broadcast(&refs, 0)?;
                }
                Ok(())
            }
            DdpInner::Rank {
                comms,
                params,
                buffers,
                ..
            } => {
                let p_tensors: Vec<Tensor> = params.iter().map(|v| v.data()).collect();
                if !p_tensors.is_empty() {
                    let refs: Vec<&Tensor> = p_tensors.iter().collect();
                    comms.broadcast(&refs, 0)?;
                }
                let b_tensors: Vec<Tensor> = buffers.iter().map(|b| b.get()).collect();
                if !b_tensors.is_empty() {
                    let refs: Vec<&Tensor> = b_tensors.iter().collect();
                    comms.broadcast(&refs, 0)?;
                }
                Ok(())
            }
        }
    }

    /// AllReduce-average gradients across all replicas.
    /// Call after backward(), before optimizer.step().
    pub fn all_reduce_gradients(&self) -> Result<()> {
        match &self.inner {
            DdpInner::Group {
                comms,
                param_groups,
                ..
            } => {
                for group in param_groups {
                    if group[0].grad().is_none() {
                        continue;
                    }
                    let grads: Vec<Tensor> = group
                        .iter()
                        .map(|v| v.grad().expect("gradient missing on replica"))
                        .collect();
                    let refs: Vec<&Tensor> = grads.iter().collect();
                    comms.all_reduce(&refs, ReduceOp::Avg)?;
                }
                Ok(())
            }
            DdpInner::Rank { comms, params, .. } => {
                // Batch every grad on this rank into a single NCCL group call.
                // Frozen params (no grad) are skipped; collective ranks must
                // call all_reduce with the same tensor count, so the user
                // contract is "freeze the same params on every rank".
                let grads: Vec<Tensor> = params.iter().filter_map(|v| v.grad()).collect();
                if grads.is_empty() {
                    return Ok(());
                }
                let refs: Vec<&Tensor> = grads.iter().collect();
                comms.all_reduce(&refs, ReduceOp::Avg)?;
                Ok(())
            }
        }
    }

    /// Broadcast buffers from rank 0 (BatchNorm running stats etc).
    pub fn sync_buffers(&self) -> Result<()> {
        match &self.inner {
            DdpInner::Group {
                comms,
                buffer_groups,
                ..
            } => {
                for group in buffer_groups {
                    let tensors: Vec<Tensor> = group.iter().map(|b| b.get()).collect();
                    let refs: Vec<&Tensor> = tensors.iter().collect();
                    comms.broadcast(&refs, 0)?;
                }
                Ok(())
            }
            DdpInner::Rank { comms, buffers, .. } => {
                let tensors: Vec<Tensor> = buffers.iter().map(|b| b.get()).collect();
                if tensors.is_empty() {
                    return Ok(());
                }
                let refs: Vec<&Tensor> = tensors.iter().collect();
                comms.broadcast(&refs, 0)?;
                Ok(())
            }
        }
    }

    /// AllReduce gradients weighted by per-device batch contribution.
    ///
    /// For heterogeneous DDP where devices process different numbers of
    /// batches per sync step. Each replica's gradient is scaled by
    /// `(batch_counts[rank] / total)` before AllReduce Sum, producing
    /// the correct mean gradient.
    ///
    /// Use with [`ElChe::batch_counts`] for automatic weighting
    /// (see [`ElChe`] for the full heterogeneous DDP strategy):
    ///
    /// ```ignore
    /// ddp.weighted_all_reduce_gradients(cadence.batch_counts())?;
    /// ```
    pub fn weighted_all_reduce_gradients(&self, batch_counts: &[usize]) -> Result<()> {
        match &self.inner {
            DdpInner::Group {
                comms,
                devices,
                param_groups,
                ..
            } => {
                if batch_counts.len() != devices.len() {
                    return Err(TensorError::new(&format!(
                        "weighted_all_reduce: batch_counts len ({}) != device count ({})",
                        batch_counts.len(),
                        devices.len(),
                    )));
                }
                let total: usize = batch_counts.iter().sum();
                if total == 0 {
                    return Err(TensorError::new(
                        "weighted_all_reduce: total batch count is 0",
                    ));
                }
                for group in param_groups {
                    if group[0].grad().is_none() {
                        continue;
                    }
                    let grads: Vec<Tensor> = group
                        .iter()
                        .enumerate()
                        .map(|(rank, v)| {
                            let g = v.grad().expect("gradient missing on replica");
                            let weight = batch_counts[rank] as f64 / total as f64;
                            g.mul_scalar_(weight).ok();
                            g
                        })
                        .collect();
                    let refs: Vec<&Tensor> = grads.iter().collect();
                    comms.all_reduce(&refs, ReduceOp::Sum)?;
                }
                Ok(())
            }
            DdpInner::Rank { comms, params, .. } => {
                if batch_counts.len() != comms.world_size() {
                    return Err(TensorError::new(&format!(
                        "weighted_all_reduce: batch_counts len ({}) != world_size ({})",
                        batch_counts.len(),
                        comms.world_size(),
                    )));
                }
                let total: usize = batch_counts.iter().sum();
                if total == 0 {
                    return Err(TensorError::new(
                        "weighted_all_reduce: total batch count is 0",
                    ));
                }
                let my_rank = comms.rank();
                let weight = batch_counts[my_rank] as f64 / total as f64;
                let grads: Vec<Tensor> = params
                    .iter()
                    .filter_map(|v| {
                        v.grad().inspect(|g| {
                            g.mul_scalar_(weight).ok();
                        })
                    })
                    .collect();
                if grads.is_empty() {
                    return Ok(());
                }
                let refs: Vec<&Tensor> = grads.iter().collect();
                comms.all_reduce(&refs, ReduceOp::Sum)?;
                Ok(())
            }
        }
    }

    /// World size: total ranks participating in the collective.
    ///
    /// For `Ddp::wrap` (legacy), this is the in-process device count.
    /// For `Ddp::wrap_single` (cluster mode), this is the cross-process /
    /// cross-host total from the rendezvous.
    pub fn world_size(&self) -> usize {
        match &self.inner {
            DdpInner::Group { devices, .. } => devices.len(),
            DdpInner::Rank { comms, .. } => comms.world_size(),
        }
    }

    /// Devices owned by this `Ddp` instance.
    ///
    /// `Ddp::wrap` returns all devices in this process. `Ddp::wrap_single`
    /// returns a one-element slice — this process owns exactly one device.
    pub fn devices(&self) -> &[Device] {
        match &self.inner {
            DdpInner::Group { devices, .. } => devices,
            DdpInner::Rank { device, .. } => std::slice::from_ref(device),
        }
    }

    // --- Deprecated aliases: use Trainer:: as the primary entry point ---

    /// Deprecated: use [`Trainer::setup()`] instead.
    ///
    /// [`Trainer`] is now the primary training entry point and carries the
    /// same behavior for 1 or N GPUs. [`Ddp`] remains for explicit
    /// multi-GPU control via [`Ddp::wrap`].
    #[deprecated(note = "use Trainer::setup() - same behavior. Ddp::setup will be removed in a future release.")]
    pub fn setup<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Trainer::setup(model, builder, optimizer)
    }

    /// Deprecated: use [`Trainer::setup_with()`] instead.
    #[deprecated(note = "use Trainer::setup_with() - same behavior. Ddp::setup_with will be removed in a future release.")]
    pub fn setup_with<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
        config: DdpConfig,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Trainer::setup_with(model, builder, optimizer, config)
    }

    /// Deprecated: renamed to [`Trainer::setup()`].
    #[deprecated(since = "0.3.0", note = "Renamed to Trainer::setup()")]
    pub fn auto<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Trainer::setup(model, builder, optimizer)
    }

    /// Deprecated: renamed to [`Trainer::setup_with()`].
    #[deprecated(since = "0.3.0", note = "Renamed to Trainer::setup_with()")]
    pub fn auto_with<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
        config: DdpConfig,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Trainer::setup_with(model, builder, optimizer, config)
    }

    // -------------------------------------------------------------------
    // Deprecated builder entry: use Trainer::builder instead
    // -------------------------------------------------------------------

    /// Deprecated: use [`Trainer::builder()`] instead.
    ///
    /// [`Trainer`] is the primary training entry point and works
    /// transparently for single-GPU and multi-GPU. This alias is retained
    /// for backwards compatibility and will be removed in a future release.
    #[deprecated(note = "use Trainer::builder() - same behavior. Ddp::builder will be removed in a future release.")]
    pub fn builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        Trainer::builder(model_factory, optim_factory, train_fn)
    }

    /// Detect whether the current CUDA setup has different GPU models.
    fn is_heterogeneous() -> bool {
        use crate::tensor::{cuda_available, cuda_device_count, cuda_device_name_idx};
        if !cuda_available() || cuda_device_count() < 2 {
            return false;
        }
        let n = cuda_device_count();
        let names: Vec<Option<String>> = (0..n)
            .map(cuda_device_name_idx)
            .collect();
        names.windows(2).any(|w| w[0] != w[1])
    }

    /// Print a diagnostic summary of detected CUDA devices to stderr.
    fn print_device_summary() {
        use crate::tensor::{
            cuda_available, cuda_device_count,
            cuda_device_name_idx, cuda_memory_info_idx,
        };
        use crate::monitor::format_bytes;

        if !cuda_available() || cuda_device_count() == 0 {
            crate::verbose!("  ddp: no CUDA available | CPU mode");
            return;
        }

        let n = cuda_device_count();
        let mut names = Vec::with_capacity(n as usize);
        let mut parts = Vec::with_capacity(n as usize);

        for i in 0..n {
            let raw_name = cuda_device_name_idx(i)
                .unwrap_or_else(|| format!("CUDA({})", i));
            let short = raw_name
                .strip_prefix("NVIDIA ")
                .unwrap_or(&raw_name)
                .to_string();
            let vram = cuda_memory_info_idx(i)
                .map(|(_, total)| format!(" ({})", format_bytes(total)))
                .unwrap_or_default();
            parts.push(format!("{}{}", short, vram));
            names.push(raw_name);
        }

        let heterogeneous = names.windows(2).any(|w| w[0] != w[1]);

        if n == 1 {
            crate::verbose!("  ddp: 1 GPU | {} | single-device mode", parts[0]);
        } else if heterogeneous {
            crate::verbose!(
                "  ddp: {} GPUs (heterogeneous) | {}",
                n,
                parts.join(" | "),
            );
        } else {
            crate::verbose!("  ddp: {} GPUs | {}", n, parts.join(" | "));
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer: primary training entry point
// ---------------------------------------------------------------------------

/// Primary entry point for training in flodl.
///
/// `Trainer` is the default API for training a model, whether you have one
/// GPU, many GPUs, or no GPU at all. The training loop is identical in all
/// cases: [`Trainer::setup`] (or [`Trainer::builder`]) configures the model,
/// detects the hardware, and enables distributed training automatically when
/// multiple CUDA devices are available. On a single GPU or CPU it's a no-op
/// wrapper with zero DDP overhead.
///
/// For explicit multi-GPU control (manual gradient sync, custom replica
/// wrapping) use [`Ddp`] directly. [`Ddp::wrap`] remains the entry point for
/// advanced patterns (GAN, RL, progressive).
///
/// # Setup mode (user owns the loop)
///
/// ```ignore
/// Trainer::setup(&model, |dev| build_model(dev), |p| Adam::new(p, 0.001))?;
///
/// for (x, y) in &train_loader {
///     let out = model.forward(&x)?;
///     let loss = cross_entropy_loss(&out, &y)?;
///     loss.backward()?;
///     model.step()?;
/// }
/// ```
///
/// # Builder mode (framework owns the loop)
///
/// ```ignore
/// let handle = Trainer::builder(model_factory, optim_factory, train_fn)
///     .dataset(dataset)
///     .batch_size(32)
///     .num_epochs(10)
///     .run()?;
///
/// let state = handle.join()?;
/// ```
pub struct Trainer;

impl Trainer {
    /// One-call setup: auto-detect GPUs, distribute the model, set the
    /// optimizer, and enable training mode.
    ///
    /// - **Multi-GPU** (2+ usable CUDA devices): replicates via
    ///   [`Graph::distribute`], creates per-replica optimizers, enables training.
    /// - **Single-GPU / CPU**: sets optimizer and training mode only (no DDP
    ///   overhead).
    ///
    /// Always prints a diagnostic summary to stderr showing detected hardware.
    ///
    /// ```ignore
    /// Trainer::setup(&model, |dev| build_model(dev), |p| Adam::new(p, 0.001))?;
    ///
    /// for batch in model.epoch(epoch).activate() {
    ///     let out = model.forward_batch(&batch?)?;
    ///     loss.backward()?;
    ///     model.step()?;
    /// }
    /// ```
    pub fn setup<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Ddp::print_device_summary();
        model.distribute(builder)?;
        model.set_optimizer(optimizer);
        model.set_training(true);

        // Auto-enable El Che for heterogeneous GPU setups
        if Ddp::is_heterogeneous() {
            model.configure_el_che(&DdpConfig::new());
        }

        Ok(())
    }

    /// One-call setup with explicit configuration.
    ///
    /// Like [`setup()`](Self::setup) but accepts a [`DdpConfig`] for
    /// controlling El Che cadence, speed hints, and overhead targets.
    ///
    /// ```ignore
    /// Trainer::setup_with(&model, builder, optimizer,
    ///     DdpConfig::new().speed_hint(1, 2.3))?;
    /// ```
    pub fn setup_with<F, M, G, O>(
        model: &Graph,
        builder: F,
        optimizer: G,
        config: DdpConfig,
    ) -> Result<()>
    where
        F: Fn(Device) -> Result<M>,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Ddp::print_device_summary();
        model.distribute(builder)?;
        model.set_optimizer(optimizer);
        model.set_training(true);
        model.configure_el_che(&config);
        // Pass timeline to distributed state for event injection in step().
        if let Some(tl) = config.timeline {
            if let Some(ref mut state) = *model.distributed.borrow_mut() {
                state.timeline = Some(tl);
            }
        }
        Ok(())
    }

    /// Create a builder for framework-managed training.
    ///
    /// The framework owns the training loop, data pipeline, and epoch
    /// management. On multi-GPU hardware, each device gets its own model
    /// replica and optimizer, and a coordinator triggers periodic
    /// parameter averaging based on the configured [`ApplyPolicy`] and
    /// [`AverageBackend`]. On a single GPU, training runs on the main
    /// thread with no coordination - the API is identical in both cases.
    ///
    /// Returns a [`DdpBuilder`] for fluent configuration. Call `.run()` to
    /// spawn training, then `.join()` on the returned [`DdpHandle`] to
    /// block until completion.
    ///
    /// [`ApplyPolicy`]: crate::distributed::ApplyPolicy
    /// [`AverageBackend`]: crate::distributed::AverageBackend
    ///
    /// # Example
    ///
    /// ```ignore
    /// use flodl::*;
    ///
    /// let handle = Trainer::builder(
    ///     |dev| model_factory(dev),
    ///     |params| Adam::new(params, 0.001),
    ///     |model, batch| { /* forward + loss */ },
    /// )
    /// .dataset(dataset)
    /// .batch_size(32)
    /// .num_epochs(10)
    /// .policy(ApplyPolicy::Cadence)
    /// .backend(AverageBackend::Nccl)
    /// .run()?;
    ///
    /// let state = handle.join()?;
    /// ```
    pub fn builder<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
    ) -> DdpBuilder<F, M, G, O, T>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        DdpHandle::new_builder(model_factory, optim_factory, train_fn)
    }

    /// One-call setup for a task-head wrapper (e.g. `flodl-hf`'s
    /// `BertForSequenceClassification`). The wrapper must implement
    /// [`HasGraph`] so `Trainer` can reach the underlying [`Graph`].
    ///
    /// Semantics match [`Trainer::setup`] exactly; the only difference is
    /// that `head_factory` builds a fresh wrapper (not a bare `Graph`) on
    /// each replica device. Useful when the training-loop code holds onto
    /// the wrapper's richer surface (`compute_loss`, `predict`, attached
    /// tokenizer) but still wants transparent 1-or-N-GPU DDP.
    ///
    /// ```ignore
    /// let head = DistilBertForSequenceClassification::from_pretrained(repo)?;
    /// let config = head.config().clone();
    /// let num_labels = head.labels().len() as i64;
    ///
    /// Trainer::setup_head(
    ///     &head,
    ///     move |dev| DistilBertForSequenceClassification::on_device(&config, num_labels, dev),
    ///     |p| Adam::new(p, 5e-5),
    /// )?;
    ///
    /// for (enc, labels) in &batches {
    ///     let loss = head.compute_loss(&enc, &labels)?;
    ///     loss.backward()?;
    ///     head.graph().step()?;
    /// }
    /// ```
    pub fn setup_head<H, F, G, O>(
        head: &H,
        head_factory: F,
        optimizer: G,
    ) -> Result<()>
    where
        H: HasGraph,
        F: Fn(Device) -> Result<H> + 'static,
        H: 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Ddp::print_device_summary();
        let graph = head.graph();
        graph.distribute(move |dev| head_factory(dev).map(|h| HeadReplica { head: h }))?;
        graph.set_optimizer(optimizer);
        graph.set_training(true);

        if Ddp::is_heterogeneous() {
            graph.configure_el_che(&DdpConfig::new());
        }

        Ok(())
    }

    /// Task-head variant of [`Trainer::setup_with`]. Same behaviour as
    /// [`Trainer::setup_head`] but takes an explicit [`DdpConfig`].
    pub fn setup_head_with<H, F, G, O>(
        head: &H,
        head_factory: F,
        optimizer: G,
        config: DdpConfig,
    ) -> Result<()>
    where
        H: HasGraph,
        F: Fn(Device) -> Result<H> + 'static,
        H: 'static,
        G: Fn(&[Parameter]) -> O,
        O: Optimizer + 'static,
    {
        Ddp::print_device_summary();
        let graph = head.graph();
        graph.distribute(move |dev| head_factory(dev).map(|h| HeadReplica { head: h }))?;
        graph.set_optimizer(optimizer);
        graph.set_training(true);
        graph.configure_el_che(&config);
        if let Some(tl) = config.timeline {
            if let Some(ref mut state) = *graph.distributed.borrow_mut() {
                state.timeline = Some(tl);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HasGraph trait: lets wrapper types plug into Trainer::setup_head
// ---------------------------------------------------------------------------

/// A wrapper type that exposes an inner [`Graph`].
///
/// Implement on any wrapper around a `Graph` that should participate in
/// [`Trainer::setup_head`] or other graph-aware DDP machinery. The
/// reference returned must outlive `&self` and point at the same graph
/// used for the wrapper's forward / loss calls.
///
/// [`Graph`] implements this trivially (returns `self`) so bare-graph
/// callers can pass a `&Graph` wherever `&impl HasGraph` is accepted.
///
/// ```ignore
/// impl HasGraph for BertForSequenceClassification {
///     fn graph(&self) -> &Graph { &self.graph }
/// }
/// ```
pub trait HasGraph {
    /// Borrow the inner training graph.
    fn graph(&self) -> &Graph;
}

impl HasGraph for Graph {
    fn graph(&self) -> &Graph { self }
}

/// Internal Module adapter used by [`Trainer::setup_head`] to feed a
/// `HasGraph` replica through [`Graph::distribute`].
///
/// `distribute` boxes each replica as `Box<dyn Module>`. Task-head
/// wrappers don't implement `Module` directly (their true forward is
/// multi-input via [`Graph::forward_multi`], which doesn't fit the
/// single-Variable `Module::forward` signature). `HeadReplica` delegates
/// every Module method through to the inner graph and overrides
/// [`Module::as_graph`] so DDP's multi-input replica paths downcast
/// cleanly rather than hitting the single-input fallback.
struct HeadReplica<H: HasGraph + 'static> {
    head: H,
}

impl<H: HasGraph + 'static> Module for HeadReplica<H> {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Single-input fallback. Task-head DDP paths reach
        // forward_multi via `as_graph()` below, so this is only
        // exercised on single-input replica paths (e.g. the scatter
        // forward in `forward_distributed_scatter`). For multi-input
        // heads that path is never triggered because the user calls
        // the head's own `compute_loss` / `forward_encoded`, which
        // route through `Graph::forward_multi` directly.
        self.head.graph().forward(input)
    }
    fn parameters(&self) -> Vec<Parameter> { self.head.graph().parameters() }
    fn buffers(&self) -> Vec<Buffer> { self.head.graph().buffers() }
    fn name(&self) -> &str { "head_replica" }
    fn set_training(&self, training: bool) { self.head.graph().set_training(training); }
    fn as_graph(&self) -> Option<&Graph> { Some(self.head.graph()) }
}

// ---------------------------------------------------------------------------
// DDP configuration
// ---------------------------------------------------------------------------

/// Configuration for [`Trainer::setup_with()`].
///
/// Controls El Che cadence behavior for heterogeneous multi-GPU training.
/// Use [`DdpConfig::new()`] for defaults or build with method chaining.
///
/// ```ignore
/// Trainer::setup_with(&model, builder, optimizer,
///     DdpConfig::new()
///         .speed_hint(1, 2.3)     // rank 1 is slow, 2.3x ratio
///         .overhead_target(0.08)  // tune to 8% overhead
/// )?;
/// ```
#[derive(Debug, Clone)]
pub struct DdpConfig {
    /// Initial speed ratio hint: (slow_rank, fast_to_slow_ratio).
    /// Applied before the first timing measurement.
    pub speed_hint: Option<(usize, f64)>,
    /// AllReduce overhead target for anchor auto-tune (default: 0.10).
    pub overhead_target: Option<f64>,
    /// Max batches on slow device before AllReduce.
    /// - `None` = auto (El Che decides, default).
    /// - `Some(0)` = disabled (traditional per-batch DDP, no El Che).
    /// - `Some(n)` = fixed anchor at n.
    pub max_anchor: Option<usize>,
    /// Maximum gradient norm for per-rank clipping in El Che mode.
    ///
    /// When set, each rank's accumulated gradients are clipped (L2 norm)
    /// before the normalize-by-count and weighted AllReduce steps. This
    /// ensures replica gradients (which the caller cannot reach) are bounded
    /// identically to rank 0.
    ///
    /// Standard DDP does not need this because the caller clips rank 0's
    /// gradients and AllReduce averages them.
    pub max_grad_norm: Option<f64>,
    /// Optional system timeline for high-frequency profiling.
    pub timeline: Option<std::sync::Arc<crate::monitor::Timeline>>,
}

impl DdpConfig {
    /// Default configuration: El Che auto-enabled for heterogeneous GPUs.
    pub fn new() -> Self {
        DdpConfig {
            speed_hint: None,
            overhead_target: None,
            max_anchor: None,
            max_grad_norm: None,
            timeline: None,
        }
    }

    /// Set initial speed ratio hint.
    ///
    /// `slow_rank`: which device is slowest.
    /// `ratio`: how many times faster the fastest device is (e.g., 2.3).
    ///
    /// After the first AllReduce, El Che discovers actual speeds and
    /// self-corrects even a wrong guess.
    pub fn speed_hint(mut self, slow_rank: usize, ratio: f64) -> Self {
        self.speed_hint = Some((slow_rank, ratio));
        self
    }

    /// Set AllReduce overhead target (fraction of compute time).
    ///
    /// Default: 0.10 (10%). Lower values = fewer AllReduces = more
    /// gradient accumulation. El Che auto-tunes the anchor to stay
    /// below this target.
    pub fn overhead_target(mut self, target: f64) -> Self {
        self.overhead_target = Some(target.clamp(0.01, 0.50));
        self
    }

    /// Set max batches on slow device before AllReduce.
    ///
    /// - `None` (default): El Che auto-tunes from overhead measurement.
    /// - `Some(0)`: disable El Che entirely (traditional per-batch sync).
    /// - `Some(n)`: fixed anchor at n (fast device gets proportionally more).
    pub fn max_anchor(mut self, max: Option<usize>) -> Self {
        self.max_anchor = max;
        self
    }

    /// Set maximum gradient norm for per-rank clipping in El Che mode.
    ///
    /// When set, each rank's accumulated gradients are clipped to this L2
    /// norm before normalize-by-count and AllReduce. Essential for
    /// heterogeneous DDP where replica gradients are otherwise unreachable
    /// by the caller.
    pub fn max_grad_norm(mut self, max_norm: f64) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Attach a system timeline for high-frequency profiling.
    pub fn timeline(mut self, tl: std::sync::Arc<crate::monitor::Timeline>) -> Self {
        self.timeline = Some(tl);
        self
    }
}

impl Default for DdpConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "ddp_tests.rs"]
mod tests;
