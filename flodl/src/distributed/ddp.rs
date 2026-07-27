//! Training entry points for flodl.
//!
//! The primary entry point is [`Trainer`]. It works transparently on 1 or
//! N GPUs - single-device training has zero DDP overhead. Reach for
//! [`Trainer`] by default; drop to [`Ddp`] only when you need explicit
//! multi-GPU control.
//!
//! **Default** ([`Trainer::builder()`], [`Trainer::run()`]): framework-owned
//! training loop driven by the authoritative controller, transparent
//! single/multi-GPU/cluster from one code path.
//!
//! **Explicit multi-GPU** ([`Ddp::wrap()`]): manual per-rank control over
//! gradient sync and parameter broadcast for advanced patterns (GAN, RL,
//! progressive).
//!
//! **User-owned loop, controller-authoritative**
//! ([`Trainer::builder()`]`.into_worker()`): the cooperative tier. You own
//! the loop body while the controller stays authoritative over cadence,
//! partition, eval election, and checkpointing (see
//! `docs/design/trainer-execution-tiers.md`).
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
//! # Manual DDP (one process per rank)
//!
//! ```ignore
//! let ddp = Ddp::wrap(&model, device, global_rank, &rendezvous)?;
//! ddp.sync_params()?;
//! // ... custom forward/backward ...
//! ddp.all_reduce_gradients()?;
//! ```

use crate::autograd::Variable;
use crate::graph::Graph;
use crate::nn::{Buffer, Module, Optimizer, Parameter};
use super::nccl::{NcclRankComm, ReduceOp};
use super::rendezvous::TcpRendezvous;
use super::config::TrainerConfig;
use super::ddp_run::{DdpBuilder, DdpHandle};
pub use super::el_che::ElChe;
use crate::tensor::{Device, Result, Tensor, TensorError};


/// Shared lock for serializing NCCL communicator creation across test modules.
/// NCCL init is a collective operation that deadlocks if two tests try to
/// create communicators simultaneously.
#[cfg(test)]
pub(crate) static NCCL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// Manual DDP coordinator
// ---------------------------------------------------------------------------

/// Manual DDP coordinator for cluster-mode (process-per-rank) gradient sync.
///
/// Each process in the cluster holds one `Ddp` joining a cross-process NCCL
/// group. For standard training, use [`Trainer::builder`] / [`Trainer::run`].
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
        crate::distributed::ddp_run::ensure_trainable_params(params.len(), "Ddp::wrap")?;
        let buffers: Vec<Buffer> = model.buffers();

        Ok(Ddp { comms, device, params, buffers })
    }

    /// Build a `Ddp` from an existing per-rank NCCL communicator.
    ///
    /// Unlike [`Ddp::wrap`], which initializes a fresh
    /// [`NcclRankComm`] via `init_rank`, this constructor
    /// takes ownership of one that's already joined to the cluster group.
    /// Use when the rendezvous + `init_rank` are driven externally (e.g. the
    /// cluster-rank inline loops in
    /// [`crate::distributed::ddp_run::DdpBuilder`], which need
    /// access to the raw comm for broadcasting initial state before wrapping).
    ///
    /// Loud errors: `device` mismatch with the rank's bound CUDA device is
    /// the caller's responsibility — no runtime check (FFI-level guarantees
    /// already enforce same-device tensors per AllReduce).
    pub fn from_comm(
        comms: NcclRankComm,
        model: &dyn Module,
        device: Device,
    ) -> Result<Self> {
        let params: Vec<Variable> = model
            .parameters()
            .into_iter()
            .map(|p| p.variable)
            .collect();
        crate::distributed::ddp_run::ensure_trainable_params(params.len(), "Ddp::from_comm")?;
        let buffers: Vec<Buffer> = model.buffers();
        Ok(Ddp { comms, device, params, buffers })
    }

    /// In-place AllReduce-average of parameters across all ranks (Local SGD).
    ///
    /// Use this at the cadence boundary of a Local-SGD loop: each rank does
    /// `forward → backward → optimizer.step()` every batch independently,
    /// then every K batches the param vectors are averaged across ranks via
    /// NCCL AllReduce-Avg. Convergence properties match PyTorch's
    /// `PostLocalSGDOptimizer` family.
    ///
    /// Distinct from [`all_reduce_gradients`](Self::all_reduce_gradients),
    /// which averages **gradients** before the optimizer step (synchronous
    /// minibatch SGD). The two are different algorithms — choose the one
    /// matching the cadence policy.
    ///
    /// All ranks must call concurrently.
    pub fn average_params(&self) -> Result<()> {
        let tensors: Vec<Tensor> = self.params.iter().map(|v| v.data()).collect();
        if tensors.is_empty() {
            return Ok(());
        }
        let refs: Vec<&Tensor> = tensors.iter().collect();
        self.comms.all_reduce(&refs, ReduceOp::Avg)?;
        Ok(())
    }

    /// Allocate the pre-sync scratch buffer used by
    /// [`average_params_with_divergence`](Self::average_params_with_divergence).
    ///
    /// Returns one zero-initialized tensor per parameter, matching shape /
    /// dtype / device. Caller pins this for the lifetime of the cadence loop
    /// so divergence measurement avoids per-cycle allocations.
    pub fn make_divergence_scratch(&self) -> Result<Vec<Tensor>> {
        self.params
            .iter()
            .map(|v| Tensor::zeros_like(&v.data()))
            .collect()
    }

    /// AllReduce-average parameters and return this rank's weight-space
    /// divergence triple `(divergence, post_norm, pre_norm)`.
    ///
    /// The same param-Avg primitive as [`average_params`](Self::average_params),
    /// plus the per-cycle telemetry the convergence-guard pipeline needs:
    ///
    /// - `divergence = ||W_pre − W_post|| / ||W_post||` — this rank's
    ///   transversal weight-space drift across the AllReduce. Fed (after
    ///   cross-rank AllReduce-gather) into
    ///   [`ConvergenceGuard::report`](crate::distributed::ddp_run::ConvergenceGuard::report)
    ///   to drive [`ElChe::nudge_anchor_down`](super::ddp::ElChe::nudge_anchor_down)
    ///   on rising drift.
    /// - `post_norm = ||W_post||` — global L2 norm of averaged params.
    ///   Identical across ranks post-AllReduce (modulo float-rounding
    ///   noise), so a single scalar from any rank suffices.
    /// - `pre_norm = ||W_pre||` — global L2 norm of this rank's pre-sync
    ///   params. Diverges across ranks pre-sync; gather like `divergence`.
    ///
    /// `scratch` is the per-param scratch buffer from
    /// [`make_divergence_scratch`](Self::make_divergence_scratch), reused
    /// across cycles. `scratch.len()` must equal the number of parameters.
    ///
    /// All ranks must call concurrently.
    pub fn average_params_with_divergence(
        &self,
        scratch: &[Tensor],
    ) -> Result<(f64, Option<f64>, Option<f64>)> {
        let param_tensors: Vec<Tensor> = self.params.iter().map(|v| v.data()).collect();
        if param_tensors.is_empty() {
            return Ok((0.0, None, None));
        }
        if scratch.len() != param_tensors.len() {
            return Err(TensorError::new(&format!(
                "average_params_with_divergence: scratch.len() ({}) must equal \
                 number of parameters ({})",
                scratch.len(),
                param_tensors.len(),
            )));
        }

        // Snapshot pre-sync params into scratch.
        for (dst, src) in scratch.iter().zip(&param_tensors) {
            dst.copy_(src, false)?;
        }

        // In-place AllReduce-Avg on params.
        let refs: Vec<&Tensor> = param_tensors.iter().collect();
        self.comms.all_reduce(&refs, ReduceOp::Avg)?;

        // Divergence triple from the shared math (scratch = pre snapshot,
        // param_tensors = post-AllReduce). One definition across backends
        // so the convergence guard's cross-backend comparison stays honest.
        crate::distributed::divergence::divergence_triple(scratch, &param_tensors)
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
}

// ---------------------------------------------------------------------------
// Trainer: primary training entry point
// ---------------------------------------------------------------------------

/// Primary entry point for training in flodl.
///
/// `Trainer` is the default API for training a model, whether you have one
/// GPU, many GPUs, or no GPU at all. The training loop is identical in all
/// cases: [`Trainer::builder`] / [`Trainer::run`] configure the model,
/// detect the hardware, and enable distributed training automatically when
/// multiple CUDA devices are available. On a single GPU or CPU it's a no-op
/// wrapper with zero DDP overhead.
///
/// For explicit multi-GPU control (manual gradient sync, custom replica
/// wrapping) use [`Ddp`] directly. [`Ddp::wrap`] remains the entry point for
/// advanced patterns (GAN, RL, progressive).
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
///
/// # Cooperative mode (user owns the loop, controller-authoritative)
///
/// ```ignore
/// let mut worker = Trainer::builder(model_factory, optim_factory, train_fn)
///     .dataset(dataset)
///     .batch_size(32)
///     .num_epochs(10)
///     .into_worker()?;
///
/// while let Some(plan) = worker.next_plan()? {
///     while let Some(batch) = worker.next_batch()? {
///         let loss = train_step(worker.model(), &batch)?;
///         worker.step(&loss)?;
///     }
/// }
/// let state = worker.finish()?;
/// ```
pub struct Trainer;

impl Trainer {
    /// Create a builder for framework-managed training.
    ///
    /// The framework owns the training loop, data pipeline, and epoch
    /// management. On multi-GPU hardware, each device gets its own model
    /// replica and optimizer, and a coordinator triggers periodic
    /// parameter averaging per the configured [`ElCheMode`] (set through
    /// [`ElCheConfig`], e.g. `.elche(ElCheConfig::nccl_cadence())`). On a
    /// single GPU, training runs on the main thread with no coordination
    /// - the API is identical in both cases.
    ///
    /// Returns a [`DdpBuilder`] for fluent configuration. Call `.run()` to
    /// spawn training, then `.join()` on the returned [`DdpHandle`] to
    /// block until completion.
    ///
    /// [`ElCheMode`]: crate::distributed::ElCheMode
    /// [`ElCheConfig`]: crate::distributed::ElCheConfig
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
    /// .elche(ElCheConfig::nccl_cadence())
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

    /// Run training from a single [`TrainerConfig`].
    ///
    /// The canonical entry for framework-managed training. Takes the
    /// three factory closures (`model_factory`, `optim_factory`,
    /// `train_fn`) and one config bag — no chained setters, no
    /// top-of-main bootstrap, no separate launcher entry. Cluster
    /// dispatch happens INSIDE this call (via the existing launcher
    /// trampoline), so the launcher process never executes user code
    /// past `Trainer::run`.
    ///
    /// # Invariant — no CUDA before `Trainer::run`
    ///
    /// User code MUST NOT touch libtorch's CUDA context before this
    /// call. That means: no [`crate::tensor::cuda_device_count`] /
    /// [`crate::tensor::cuda_devices`] / `Tensor` construction on a
    /// CUDA device / `Module::on_device(Device::CUDA(_))`. Pre-run GPU
    /// queries must go through [`crate::sys::detect_gpus`] (which uses
    /// `nvidia-smi` and does NOT init libtorch).
    ///
    /// Why: on cluster fan-out the parent (launcher) process exits
    /// without running training, and on heterogeneous GPUs touching
    /// CUDA in the launcher corrupts the spawned children's CUDA
    /// context (see `feedback_nccl_exclusive_gpu`).
    ///
    /// # Composition with the builder
    ///
    /// `Trainer::builder(...).chain().run()` continues to work
    /// unchanged. Internally both surfaces drive the same launch
    /// path; pick whichever style fits the call-site better.
    pub fn run<F, M, G, O, T>(
        model_factory: F,
        optim_factory: G,
        train_fn: T,
        cfg: TrainerConfig<M>,
    ) -> Result<DdpHandle>
    where
        F: Fn(Device) -> Result<M> + Send + Sync + 'static,
        M: Module + 'static,
        G: Fn(&[Parameter]) -> O + Send + Sync + 'static,
        O: Optimizer + 'static,
        T: Fn(&M, &[Tensor]) -> Result<Variable> + Send + Sync + 'static,
    {
        // Single bridge: the whole ElChe strategy (mode + cadence tuning +
        // guard + partition_ratios + easgd_alpha + max_overshoot) lands via
        // one call. `.elche()` derives policy/backend from `mode` and moves
        // the guard override onto the builder.
        let mut b = DdpHandle::new_builder(model_factory, optim_factory, train_fn)
            .dataset(cfg.dataset)
            .batch_size(cfg.batch_size)
            .num_epochs(cfg.num_epochs)
            .elche(cfg.elche)
            .vram_pool(cfg.vram_pool)
            .vram_max_usage(cfg.vram_max_usage)
            .ram_max_usage(cfg.ram_max_usage)
            .sample_cache(cfg.sample_cache)
            .disk_stage(cfg.disk_stage_gb)
            .augment(cfg.augment);
        if let Some(dir) = cfg.disk_stage_dir {
            b = b.disk_stage_dir(dir);
        }
        if let Some(f) = cfg.transform {
            b = b.transform_fn(f);
        }

        if let Some(n) = cfg.max_grad_norm {
            b = b.max_grad_norm(n);
        }
        if let Some(t) = cfg.max_failure {
            b = b.max_failure(t);
        }
        if let Some(n) = cfg.checkpoint_every {
            b = b.checkpoint_every(n);
        }
        if let Some(p) = cfg.save_path {
            b = b.save_path(p);
        }
        if let Some(p) = cfg.resume_from {
            b = b.resume_from(p);
        }
        if let Some(e) = cfg.checkpoint_at_epoch {
            b = b.checkpoint_at_epoch(e);
        }
        if let Some(f) = cfg.outer_optimizer {
            b = b.outer_optimizer_arc(f);
        }
        if let Some(f) = cfg.checkpoint_fn {
            b = b.checkpoint_fn_arc(f);
        }
        if let Some(f) = cfg.epoch_fn {
            b = b.epoch_fn_arc(f);
        }
        if let Some(f) = cfg.metrics_fn {
            b = b.metrics_fn_arc(f);
        }
        if let Some(f) = cfg.scheduler_fn {
            b = b.scheduler_fn_boxed(f);
        }
        // Eval cadence: an eval_fn registered without an explicit
        // cadence runs EVERY epoch — `eval_every_epochs` defaults to
        // `None` (= disabled) downstream, which silently turned a fully
        // wired eval pipeline into dead code on this entry point.
        let has_eval_fn = cfg.eval_fn.is_some();
        if let Some(f) = cfg.eval_fn {
            b = b.eval_fn_arc(f);
        }
        match (cfg.eval_every, has_eval_fn) {
            (Some(n), _) => b = b.eval_every(crate::distributed::ddp_run::EvalCadence::Epochs(n)),
            (None, true) => b = b.eval_every(crate::distributed::ddp_run::EvalCadence::Epochs(1)),
            (None, false) => {}
        }
        if let Some(n) = cfg.reports_per_epoch {
            b = b.reports_per_epoch(n);
        }
        if let Some(dir) = cfg.record_log_dir {
            b = b.record_log(dir, cfg.max_log_size.unwrap_or(0));
        }
        if let Some(path) = cfg.dashboard_html {
            b = b.save_dashboard(path);
        }
        if let Some(theme) = cfg.dashboard_theme {
            b = b.dashboard_theme(theme);
        }
        for (key, reduction) in cfg.scalar_reductions {
            b = b.scalar_reduction(key, reduction);
        }
        if let Some(ds) = cfg.eval_dataset {
            b = b.eval_dataset(ds);
        }
        if let Some(f) = cfg.eval_result_fn {
            b = b.eval_result_fn_arc(f);
        }
        if let Some(t) = cfg.timeline {
            b = b.timeline(t);
        }
        b = b.epoch_callback_policy(cfg.epoch_callback_policy);
        // Programmatic cluster rides the builder; the env promotion
        // (single site, role-gated) happens in `DdpBuilder::run`.
        if let Some(c) = cfg.cluster {
            b = b.cluster(c);
        }

        b.run()
    }
}

// ---------------------------------------------------------------------------
// HasGraph trait: lets wrapper types expose their inner Graph
// ---------------------------------------------------------------------------

/// A wrapper type that exposes an inner [`Graph`].
///
/// Implement on any wrapper around a `Graph` that should present the
/// underlying graph to graph-aware framework paths (e.g. `flodl-hf`'s
/// task heads). The reference returned must outlive `&self` and point at
/// the same graph used for the wrapper's forward / loss calls.
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

#[cfg(test)]
#[path = "ddp_tests.rs"]
mod tests;
