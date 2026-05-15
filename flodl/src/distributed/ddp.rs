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
use super::cluster::LocalCluster;
use super::nccl::{NcclRankComm, ReduceOp};
use super::rendezvous::TcpRendezvous;
use super::ddp_run::{DdpBuilder, DdpHandle};
pub use super::el_che::ElChe;
use crate::tensor::{Device, Result, Tensor, TensorError};


/// Shared lock for serializing NCCL communicator creation across test modules.
/// NCCL init is a collective operation that deadlocks if two tests try to
/// create communicators simultaneously.
#[cfg(test)]
pub(crate) static NCCL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// Cluster-mode ElChe state (process-per-rank heterogeneous DDP)
// ---------------------------------------------------------------------------

/// Per-process state for cluster-mode El Che heterogeneous DDP.
///
/// Each rank holds its own copy. On every cadence boundary all ranks run
/// a cross-process timing AllReduce ([`Ddp::all_reduce_per_rank_f64`]) and
/// then call [`ElChe::report_timing`] with identical inputs — so the
/// anchor and per-rank batch counts stay coherent across the cluster
/// without a separate broadcast step.
///
/// Lives behind `Graph.cluster_el_che`; [`Graph::step`] reads the local
/// cadence target (`el_che.batch_counts()[my_rank]`) and only fires the
/// actual sync + optimizer step when `local_batch_idx` reaches it. Between
/// cadences, `step()` increments the counter and returns (gradients
/// accumulate on the replica's parameters via subsequent `backward()`
/// calls).
pub(crate) struct ClusterElCheState {
    /// El Che cadence strategy (anchor auto-tune lives here).
    pub el_che: ElChe,
    /// Batches processed locally since last sync. Reaches the local target
    /// (`el_che.batch_counts()[my_rank]`) at the cadence boundary.
    pub local_batch_idx: usize,
    /// Wall-clock at start of current cycle. Set on the first `step()` of
    /// the cycle, cleared after the cadence-boundary sync. Used to feed
    /// `ElChe::report_timing` per-rank compute wall times.
    pub cycle_start: Option<std::time::Instant>,
    /// Optional per-rank gradient clipping norm. Applied to the accumulated
    /// gradients before normalize-by-count and the weighted AllReduce.
    pub max_grad_norm: Option<f64>,
}

impl ClusterElCheState {
    /// Default initial anchor — matches single-process El Che semantics.
    /// Auto-tunes from observed timing after the warmup window.
    pub(crate) const DEFAULT_INITIAL_ANCHOR: usize = 10;

    pub(crate) fn from_config(world_size: usize, config: &DdpConfig) -> Self {
        let anchor = ClusterElCheState::DEFAULT_INITIAL_ANCHOR;
        let mut el_che = ElChe::new(world_size, anchor);
        if let Some(target) = config.overhead_target {
            el_che = el_che.with_overhead_target(target);
        }
        if let Some(max) = config.max_anchor {
            el_che = el_che.with_max_anchor(max);
        }
        if let Some((slow_rank, ratio)) = config.speed_hint {
            el_che = el_che.with_speed_ratio(slow_rank, ratio);
        }
        Self {
            el_che,
            local_batch_idx: 0,
            cycle_start: None,
            max_grad_norm: config.max_grad_norm,
        }
    }
}

// ---------------------------------------------------------------------------
// Manual DDP coordinator
// ---------------------------------------------------------------------------

/// Manual DDP coordinator for cluster-mode (process-per-rank) gradient sync.
///
/// Each process in the cluster holds one `Ddp` joining a cross-process NCCL
/// group. For standard training, use [`Trainer::setup`] / [`Trainer::setup_with`].
pub struct Ddp {
    comms: NcclRankComm,
    device: Device,
    params: Vec<Variable>,
    buffers: Vec<Buffer>,
}

impl Ddp {
    /// Wrap a single model replica joined to a cross-process NCCL group.
    ///
    /// Each process in the cluster calls this with its own model, its own
    /// CUDA device, its own global rank (typically from
    /// [`super::LocalCluster::my_rank`]), and the rendezvous's shared
    /// [`NcclUniqueId`](super::NcclUniqueId) (from
    /// [`super::LocalCluster::rendezvous`]). NCCL synchronizes the group
    /// internally via the UID handshake.
    ///
    /// Loud errors: `global_rank >= rdv.world_size()`. NCCL init failures
    /// propagate from [`NcclRankComm::init_rank`].
    pub fn wrap(
        model: &dyn Module,
        device: Device,
        global_rank: usize,
        rdv: &TcpRendezvous,
    ) -> Result<Self> {
        let world_size = rdv.world_size();
        if global_rank >= world_size {
            return Err(TensorError::new(&format!(
                "Ddp::wrap: global_rank {global_rank} >= world_size {world_size}"
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

        Ok(Ddp { comms, device, params, buffers })
    }

    /// Broadcast parameters and buffers from rank 0 to all ranks.
    pub fn sync_params(&self) -> Result<()> {
        let p_tensors: Vec<Tensor> = self.params.iter().map(|v| v.data()).collect();
        if !p_tensors.is_empty() {
            let refs: Vec<&Tensor> = p_tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        let b_tensors: Vec<Tensor> = self.buffers.iter().map(|b| b.get()).collect();
        if !b_tensors.is_empty() {
            let refs: Vec<&Tensor> = b_tensors.iter().collect();
            self.comms.broadcast(&refs, 0)?;
        }
        Ok(())
    }

    /// AllReduce-average gradients across all ranks.
    /// Call after backward(), before optimizer.step().
    pub fn all_reduce_gradients(&self) -> Result<()> {
        // Batch every grad on this rank into a single NCCL group call.
        // Frozen params (no grad) are skipped; collective ranks must call
        // all_reduce with the same tensor count, so the user contract is
        // "freeze the same params on every rank".
        let grads: Vec<Tensor> = self.params.iter().filter_map(|v| v.grad()).collect();
        if grads.is_empty() {
            return Ok(());
        }
        let refs: Vec<&Tensor> = grads.iter().collect();
        self.comms.all_reduce(&refs, ReduceOp::Avg)?;
        Ok(())
    }

    /// Broadcast buffers from rank 0 (BatchNorm running stats etc).
    pub fn sync_buffers(&self) -> Result<()> {
        let tensors: Vec<Tensor> = self.buffers.iter().map(|b| b.get()).collect();
        if tensors.is_empty() {
            return Ok(());
        }
        let refs: Vec<&Tensor> = tensors.iter().collect();
        self.comms.broadcast(&refs, 0)?;
        Ok(())
    }

    /// AllReduce gradients weighted by per-rank batch contribution.
    ///
    /// For heterogeneous DDP where ranks process different numbers of batches
    /// per sync step. This rank's gradient is scaled by
    /// `(batch_counts[my_rank] / total)` before AllReduce Sum, producing the
    /// correct mean gradient.
    ///
    /// Use with [`ElChe::batch_counts`] for automatic weighting
    /// (see [`ElChe`] for the full heterogeneous DDP strategy):
    ///
    /// ```ignore
    /// ddp.weighted_all_reduce_gradients(cadence.batch_counts())?;
    /// ```
    pub fn weighted_all_reduce_gradients(&self, batch_counts: &[usize]) -> Result<()> {
        if batch_counts.len() != self.comms.world_size() {
            return Err(TensorError::new(&format!(
                "weighted_all_reduce: batch_counts len ({}) != world_size ({})",
                batch_counts.len(),
                self.comms.world_size(),
            )));
        }
        let total: usize = batch_counts.iter().sum();
        if total == 0 {
            return Err(TensorError::new(
                "weighted_all_reduce: total batch count is 0",
            ));
        }
        let my_rank = self.comms.rank();
        let weight = batch_counts[my_rank] as f64 / total as f64;
        let grads: Vec<Tensor> = self.params
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
        self.comms.all_reduce(&refs, ReduceOp::Sum)?;
        Ok(())
    }

    /// World size: total ranks in the cross-process group.
    pub fn world_size(&self) -> usize {
        self.comms.world_size()
    }

    /// This process's global rank in the cluster.
    pub fn rank(&self) -> usize {
        self.comms.rank()
    }

    /// AllReduce a per-rank `f64` measurement vector across the cluster.
    ///
    /// `local` must be length `world_size`. Caller writes its measurement
    /// into its own slot (other slots zero); on return every rank sees the
    /// sum vector. With each rank contributing only its slot, the sum is
    /// the gathered vector — which lets every rank run identical bookkeeping
    /// downstream (e.g. `ElChe::report_timing`) without a separate broadcast.
    ///
    /// Internally allocates a small CUDA tensor on this rank's device,
    /// NCCL AllReduce Sum, copies back.
    pub fn all_reduce_per_rank_f64(&self, local: &mut [f64]) -> Result<()> {
        let world_size = self.comms.world_size();
        if local.len() != world_size {
            return Err(TensorError::new(&format!(
                "all_reduce_per_rank_f64: vector len ({}) must equal world_size ({})",
                local.len(),
                world_size,
            )));
        }
        let t = Tensor::from_f64(local, &[world_size as i64], self.device)?;
        self.comms.all_reduce(&[&t], ReduceOp::Sum)?;
        let out = t.to_f64_vec()?;
        local.copy_from_slice(&out);
        Ok(())
    }

    /// Device owned by this `Ddp` instance (this process owns exactly one).
    pub fn device(&self) -> Device {
        self.device
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
        dispatch_launcher_or_continue()?;
        Ddp::print_device_summary();

        if let Some(cluster) = LocalCluster::from_env()? {
            return setup_cluster(model, &cluster, builder, optimizer, None);
        }

        // No cluster envelope: single-device mode. The `builder` is only
        // used to construct per-rank replicas; in single-device mode the
        // user already passed a fully-built `model` on the target device.
        let _ = builder;
        model.set_optimizer(optimizer);
        model.set_training(true);
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
        dispatch_launcher_or_continue()?;
        Ddp::print_device_summary();

        if let Some(cluster) = LocalCluster::from_env()? {
            // Cluster mode: honor DdpConfig (max_anchor, speed_hint,
            // overhead_target, max_grad_norm). Cross-process timeline
            // event protocol lands in a follow-up.
            return setup_cluster(model, &cluster, builder, optimizer, Some(&config));
        }

        // No cluster envelope: single-device mode. El Che cadence and
        // cross-rank timeline are no-ops with one rank; the config knobs
        // would have nothing to act on. The builder is only used for
        // replication.
        let _ = builder;
        let _ = config;
        model.set_optimizer(optimizer);
        model.set_training(true);
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
        dispatch_launcher_or_continue()?;
        Ddp::print_device_summary();
        let graph = head.graph();

        if let Some(cluster) = LocalCluster::from_env()? {
            return setup_head_cluster(graph, &cluster, head_factory, optimizer, None);
        }

        // No cluster envelope: single-device mode.
        let _ = head_factory;
        graph.set_optimizer(optimizer);
        graph.set_training(true);
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
        dispatch_launcher_or_continue()?;
        Ddp::print_device_summary();
        let graph = head.graph();

        if let Some(cluster) = LocalCluster::from_env()? {
            return setup_head_cluster(graph, &cluster, head_factory, optimizer, Some(&config));
        }

        // No cluster envelope: single-device mode.
        let _ = head_factory;
        let _ = config;
        graph.set_optimizer(optimizer);
        graph.set_training(true);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cluster-mode setup helper (process-per-rank)
// ---------------------------------------------------------------------------

/// Wire `model` into the cluster's NCCL group for this process's rank.
///
/// Each process in the cluster runs this once on startup:
/// 1. Read this process's slot (`(global_rank, device)`) from the envelope.
/// 2. Rendezvous to obtain the shared `NcclUniqueId`.
/// 3. Build one local replica on this rank's device via `builder`.
/// 4. Wrap it with [`Ddp::wrap`] (one process = one rank, joined to
///    the cross-process group).
/// 5. Broadcast rank-0 parameters/buffers to every rank for an identical
///    starting point.
/// 6. Hand `(ddp, replica)` off to the graph; from that point on, the
///    [`Module`] surface on `model` routes to the replica.
///
/// Loud errors at every step — silent fallthrough on a misconfigured
/// cluster is the worst class of bug (data shard divergence, hangs,
/// silently-wrong gradients).
/// Cluster-role detection slot-in called from every `Trainer::setup*`.
///
/// Three outcomes:
/// - [`Role::SingleDevice`]: no cluster env → return Ok so caller proceeds.
/// - [`Role::Rank`]: rank slot env → return Ok so caller's cluster-path
///   logic ([`LocalCluster::from_env`] + rendezvous + `Ddp::wrap`) runs.
/// - [`Role::LauncherDone`]: this process was the launcher; fan-out
///   completed, ranks all exited cleanly. The user's training-loop code
///   has nothing to do here — exit the program with status 0.
///
/// Wrapping the role check in `Trainer::setup*` keeps the user code
/// transparent: same `Trainer::setup(&model, factory, optim)?` line on
/// single-host, multi-process single-host, and multi-host invocations.
/// The launcher branch never returns to user code; everything below the
/// `setup` call runs only on the rank-side.
///
/// [`Role::SingleDevice`]: crate::distributed::launcher::Role::SingleDevice
/// [`Role::Rank`]: crate::distributed::launcher::Role::Rank
/// [`Role::LauncherDone`]: crate::distributed::launcher::Role::LauncherDone
/// [`LocalCluster::from_env`]: crate::distributed::cluster::LocalCluster::from_env
fn dispatch_launcher_or_continue() -> Result<()> {
    match crate::distributed::launcher::dispatch()? {
        crate::distributed::launcher::Role::LauncherDone => std::process::exit(0),
        crate::distributed::launcher::Role::Rank
        | crate::distributed::launcher::Role::SingleDevice => Ok(()),
    }
}

fn setup_cluster<F, M, G, O>(
    model: &Graph,
    cluster: &LocalCluster,
    builder: F,
    optimizer: G,
    config: Option<&DdpConfig>,
) -> Result<()>
where
    F: Fn(Device) -> Result<M>,
    M: Module + 'static,
    G: Fn(&[Parameter]) -> O,
    O: Optimizer + 'static,
{
    let (global_rank, device) = cluster.my_rank()?;
    // Caller-declared dataset fingerprint. Default `[0u8; 32]` (every rank
    // trivially agrees); when set, mismatching ranks fail loudly at the
    // rendezvous instead of silently training on divergent shards.
    let dataset_sig = config.map_or([0u8; 32], |c| c.dataset_signature);
    let rdv = cluster.rendezvous(dataset_sig)?;
    let world_size = rdv.world_size();
    let replica = builder(device)?;
    let ddp = Ddp::wrap(&replica, device, global_rank, &rdv)?;
    ddp.sync_params()?;
    model.set_cluster_ddp(ddp, Box::new(replica));
    model.set_optimizer(optimizer);
    model.set_training(true);
    enable_cluster_el_che(model, world_size, cluster, config);
    Ok(())
}

/// Decide whether to enable cluster-mode El Che for this rank.
///
/// Mirrors the single-process auto-enable rule but reads the cluster
/// envelope: a cluster that spans multiple hosts is, by construction,
/// the heterogeneous case El Che is designed for. An explicit
/// [`DdpConfig::max_anchor`] of `Some(0)` always opts out.
fn enable_cluster_el_che(
    model: &Graph,
    world_size: usize,
    cluster: &LocalCluster,
    config: Option<&DdpConfig>,
) {
    if world_size < 2 {
        return;
    }
    if let Some(cfg) = config {
        if cfg.max_anchor == Some(0) {
            return;
        }
        model.set_cluster_el_che(ClusterElCheState::from_config(world_size, cfg));
        return;
    }
    // No explicit config: auto-enable only when the cluster spans hosts.
    // Single-host multi-process clusters skip El Che by default; users can
    // opt in via `Trainer::setup_with`.
    if cluster.spans_multiple_hosts() {
        model.set_cluster_el_che(ClusterElCheState::from_config(
            world_size,
            &DdpConfig::new(),
        ));
    }
}

/// Cluster-mode setup for a [`HasGraph`] task-head wrapper.
///
/// Same shape as [`setup_cluster`] but the local replica is the
/// task-head wrapper (`H`) adapted to [`Module`] via [`HeadReplica`].
/// User-side training code keeps working through the original `head`
/// reference because `head.compute_loss` / `head.graph().forward_multi`
/// route through the graph's cluster-aware short-circuits to the local
/// replica's graph (which [`HeadReplica::as_graph`] exposes).
fn setup_head_cluster<H, F, G, O>(
    graph: &Graph,
    cluster: &LocalCluster,
    head_factory: F,
    optimizer: G,
    config: Option<&DdpConfig>,
) -> Result<()>
where
    H: HasGraph + 'static,
    F: Fn(Device) -> Result<H>,
    G: Fn(&[Parameter]) -> O,
    O: Optimizer + 'static,
{
    let (global_rank, device) = cluster.my_rank()?;
    let dataset_sig = config.map_or([0u8; 32], |c| c.dataset_signature);
    let rdv = cluster.rendezvous(dataset_sig)?;
    let world_size = rdv.world_size();
    let head_local = head_factory(device)?;
    let replica = HeadReplica { head: head_local };
    let ddp = Ddp::wrap(&replica, device, global_rank, &rdv)?;
    ddp.sync_params()?;
    graph.set_cluster_ddp(ddp, Box::new(replica));
    graph.set_optimizer(optimizer);
    graph.set_training(true);
    enable_cluster_el_che(graph, world_size, cluster, config);
    Ok(())
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
    /// Cluster-mode dataset fingerprint exchanged at rendezvous.
    ///
    /// Every rank must present the same 32-byte signature; a mismatch is a
    /// loud error at startup rather than silent data divergence during
    /// training. Default all-zeros means "no fingerprint declared" — every
    /// rank trivially agrees by construction.
    ///
    /// Single-host (non-cluster) DDP ignores this field.
    pub dataset_signature: [u8; 32],
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
            dataset_signature: [0u8; 32],
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

    /// Declare a 32-byte dataset fingerprint exchanged at cluster rendezvous.
    ///
    /// Every rank must present the same signature; mismatches fail loudly
    /// at startup instead of silently training on divergent shards. Typical
    /// derivation: a sha-256 of the dataset's manifest, split config, and
    /// version string.
    pub fn dataset_signature(mut self, sig: [u8; 32]) -> Self {
        self.dataset_signature = sig;
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
