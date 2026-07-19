//! Neural network modules, losses, optimizers, and training utilities.
//!
//! All layers implement the [`Module`] trait (forward + parameters).
//! Optimizers implement [`Optimizer`] (step + zero_grad).
//! Both compose naturally with the graph builder in [`crate::graph`].
//!
//! # Features
//!
//! - **Layers**: Linear, Conv1d/Conv2d/Conv3d, ConvTranspose1d/2d/3d, MaxPool1d/2d, AvgPool1d/2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d, PixelShuffle/Unshuffle, Upsample, Unfold/Fold, LayerNorm, RMSNorm, GroupNorm, BatchNorm/BatchNorm2d, InstanceNorm, Dropout/Dropout2d/AlphaDropout, Embedding/EmbeddingBag, GRUCell, GRU, LSTMCell, LSTM, MultiheadAttention, Bilinear, ZeroPad2d, ReflectionPad2d
//! - **Activations**: Identity, ReLU, LeakyReLU, ELU, Sigmoid, Tanh, GELU, SiLU, Softplus, Mish, SELU, Hardswish, Hardsigmoid, PReLU, Softmax, LogSoftmax, Flatten, GaussianBlur
//! - **Losses**: MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1, KLDiv, NLL, CTC, Focal, TripletMargin, CosineEmbedding, HingeEmbedding, MarginRanking, PoissonNLL
//! - **Optimizers**: SGD (momentum), Adam, AdamW, RMSprop, Adagrad, RAdam, NAdam -- fused Adam/AdamW uses `_fused_adamw_` on CUDA for single-kernel multi-tensor updates
//! - **Schedulers**: StepDecay, Cosine, Warmup, Plateau, ExponentialLR, MultiStepLR, OneCycleLR, CyclicLR
//! - **Gradient clipping**: `clip_grad_norm` / `clip_grad_value` -- fused clipping via foreach ops (2 kernels instead of 2N)
//! - **Mixed precision**: [`AutocastGuard`] / [`autocast`] for automatic dtype casting, [`GradScaler`] for loss scaling, [`cast_parameters`] for dtype conversion
//! - **CUDA Graphs**: [`CudaGraph`] capture/replay/reset via [`cuda_graph_capture`], memory pool handles, configurable capture modes
//! - **Foreach operations**: 7 multi-tensor ops (`foreach_zero_`, `foreach_add_scalar_`, `foreach_mul_scalar_`, etc.) used internally by optimizers and gradient clipping
//! - **Checkpointing**: save/load with named parameters, dtype-aware, partial loading
//! - **Initialization**: Xavier uniform/normal, Kaiming uniform/normal, uniform, normal, orthogonal, truncated normal

pub mod parameter;
pub mod buffer;
pub mod init;
pub mod linear;
pub mod activation;
pub mod loss;
pub mod optim;
pub mod clip;
pub mod scheduler;
pub mod dropout;
pub mod padding;
pub mod layernorm;
pub mod rmsnorm;
pub mod embedding;
pub mod grucell;
pub mod gru;
pub mod lstmcell;
pub mod lstm;
pub mod conv1d;
pub mod conv2d;
pub mod conv_transpose1d;
pub mod conv_transpose2d;
pub mod conv3d;
pub mod conv_transpose3d;
pub mod groupnorm;
pub mod batchnorm;
pub mod instancenorm;
pub mod pooling;
pub mod bilinear;
pub mod attention;
pub mod checkpoint;
pub mod amp;
pub mod cuda_graph;
pub mod functional;

pub use parameter::Parameter;
pub use buffer::Buffer;
pub use linear::Linear;
pub use activation::{
    Identity, ReLU, Sigmoid, Tanh, GELU, GeluApprox, SiLU,
    LeakyReLU, ELU, Softplus, Mish,
    SELU, Hardswish, Hardsigmoid, PReLU,
    Softmax, LogSoftmax, Flatten,
};
pub use loss::{
    mse_loss, cross_entropy_loss, bce_loss, bce_with_logits_loss,
    l1_loss, smooth_l1_loss, kl_div_loss,
    nll_loss, ctc_loss, focal_loss,
    triplet_margin_loss, cosine_embedding_loss,
    hinge_embedding_loss, margin_ranking_loss, poisson_nll_loss,
};
pub use optim::{Optimizer, Stateful, StateKind, migrate_optim_state_file, SGD, SGDBuilder, Adam, AdamBuilder, AdamW, AdamWBuilder, RMSprop, RMSpropBuilder, Adagrad, AdagradBuilder, RAdam, NAdam};
pub use checkpoint::{
    save_checkpoint, load_checkpoint, save_checkpoint_file, load_checkpoint_file,
    migrate_checkpoint, migrate_checkpoint_file, checkpoint_version, checkpoint_keys,
    LoadReport, MigrateReport,
};
pub use amp::{GradScaler, cast_parameters, AutocastGuard, autocast, is_autocast_enabled, is_autocast_enabled_for};
pub use clip::{clip_grad_norm, clip_grad_value};
pub use scheduler::{Scheduler, StepDecay, CosineScheduler, WarmupScheduler, PlateauScheduler, ExponentialLR, MultiStepLR, OneCycleLR, CyclicLR};
pub use dropout::{Dropout, Dropout2d, AlphaDropout};
pub use padding::{ZeroPad2d, ReflectionPad2d};
pub use layernorm::LayerNorm;
pub use rmsnorm::RMSNorm;
pub use embedding::{Embedding, EmbeddingBag};
pub use grucell::GRUCell;
pub use gru::GRU;
pub use lstmcell::LSTMCell;
pub use lstm::LSTM;
pub use conv1d::{Conv1d, Conv1dBuilder};
pub use conv2d::{Conv2d, Conv2dBuilder};
pub use conv_transpose1d::{ConvTranspose1d, ConvTranspose1dBuilder};
pub use conv_transpose2d::{ConvTranspose2d, ConvTranspose2dBuilder};
pub use conv3d::{Conv3d, Conv3dBuilder};
pub use conv_transpose3d::{ConvTranspose3d, ConvTranspose3dBuilder};
pub use groupnorm::GroupNorm;
pub use batchnorm::{BatchNorm, BatchNorm2d};
pub use instancenorm::InstanceNorm;
pub use pooling::{MaxPool2d, AvgPool2d, MaxPool1d, AvgPool1d, AdaptiveMaxPool2d, AdaptiveAvgPool2d, PixelShuffle, PixelUnshuffle, Upsample, Unfold, Fold};
pub use bilinear::Bilinear;
pub use attention::MultiheadAttention;
pub use init::{xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform_bias, uniform, normal, orthogonal, trunc_normal};
pub use functional::{gaussian_blur_2d, GaussianBlur};
pub use cuda_graph::{CudaGraph, MemPoolId, CaptureMode, cuda_graph_capture, cuda_graph_pool_handle};

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::autograd::Variable;
use crate::tensor::Result;

/// The core module trait: forward pass + parameter access.
///
/// All neural network layers implement Module. Composite modules (Graph, loops,
/// gates) implement Module too, so they compose like any other layer.
///
/// ```ignore
/// let model = Linear::new(4, 2)?;
/// let x = Variable::new(Tensor::randn(&[1, 4], opts)?, false);
/// let y = model.forward(&x)?; // [1, 4] → [1, 2]
/// ```
pub trait Module {
    /// Run the forward pass on `input` and return the result.
    fn forward(&self, input: &Variable) -> Result<Variable>;
    /// Return this module's learnable parameters.
    /// Default: recursively collects from `sub_modules()` with pointer dedup.
    ///
    /// Leaf modules holding parameters MUST override this — for a module
    /// with no sub-modules the default returns an empty list, so a
    /// forgotten override reaches training as a model with nothing to
    /// train. Trainer entries reject zero-parameter models loudly for
    /// exactly this reason.
    fn parameters(&self) -> Vec<Parameter> {
        let subs = self.sub_modules();
        if subs.is_empty() {
            return vec![];
        }
        let mut params = Vec::new();
        let mut seen = HashSet::new();
        let mut visited = HashSet::new();
        for child in &subs {
            walk_modules_visited(child.as_ref(), &mut visited, &mut |m| {
                for p in m.parameters() {
                    let ptr = p.variable.id();
                    if seen.insert(ptr) {
                        params.push(p);
                    }
                }
            });
        }
        params
    }

    /// Return this module's non-learnable persistent buffers (e.g., running stats).
    /// Default: recursively collects from `sub_modules()` with pointer dedup.
    /// Leaf modules should override to return their own buffers.
    fn buffers(&self) -> Vec<Buffer> {
        let subs = self.sub_modules();
        if subs.is_empty() {
            return vec![];
        }
        let mut bufs = Vec::new();
        let mut seen = HashSet::new();
        let mut visited = HashSet::new();
        for child in &subs {
            walk_modules_visited(child.as_ref(), &mut visited, &mut |m| {
                for b in m.buffers() {
                    let ptr = b.id();
                    if seen.insert(ptr) {
                        bufs.push(b);
                    }
                }
            });
        }
        bufs
    }

    /// Human-readable type name used as node ID prefix in graph visualization.
    /// Override to return a lowercase identifier (e.g., "linear", "gelu").
    fn name(&self) -> &str { "module" }

    /// Return direct child modules for recursive tree walks.
    /// Override in composite modules (loops, switches, gates).
    fn sub_modules(&self) -> Vec<Rc<dyn Module>> { vec![] }

    /// Move all parameters and buffers to the given device.
    ///
    /// The default moves everything reachable through [`Module::parameters`]
    /// and [`Module::buffers`], so it covers leaf and composite modules
    /// alike. Parameters are re-leafed on the way (detach → move →
    /// [`Variable::set_data`](crate::autograd::Variable::set_data), the same
    /// recipe as `Graph::set_device`), which also bumps their data
    /// generation so parameter-derived caches (cuDNN flattened RNN weights)
    /// rebuild. Already-on-device tensors are skipped, so repeated or
    /// overlapping moves are cheap no-ops.
    ///
    /// Panics if a move fails: a half-moved model would otherwise only
    /// surface later as a confusing cross-device op error far from the
    /// cause. Override in modules holding device-resident state reachable
    /// through neither accessor.
    fn move_to_device(&self, device: crate::tensor::Device) {
        for p in self.parameters() {
            let data = p.variable.data();
            if data.device() != device {
                let moved = data
                    .detach()
                    .and_then(|d| d.to_device(device))
                    .unwrap_or_else(|e| {
                        panic!(
                            "Module::move_to_device: failed to move parameter '{}' to {device:?}: {e}",
                            p.name
                        )
                    });
                p.variable.set_data(moved);
            }
        }
        for b in self.buffers() {
            if b.get().device() != device {
                b.to_device(device).unwrap_or_else(|e| {
                    panic!("Module::move_to_device: failed to move buffer to {device:?}: {e}")
                });
            }
        }
    }

    /// Set training/eval mode. Affects Dropout, BatchNorm, etc.
    /// Override in modules with mode-dependent behavior.
    fn set_training(&self, _training: bool) {}

    /// Set training mode. Shorthand for `set_training(true)`.
    fn train(&self) { self.set_training(true); }

    /// Set eval mode. Shorthand for `set_training(false)`.
    fn eval(&self) { self.set_training(false); }

    /// Return per-iteration side output for loop tracing.
    /// Override in loop body modules that capture trajectory data
    /// (e.g., attention fixation points). Returns `None` by default.
    /// When `Some`, the loop executor collects traces accessible via
    /// `Graph::traces()`.
    fn trace(&self) -> Option<Variable> { None }

    /// Upcast to [`NamedInputModule`] for multi-input graphs.
    /// Override in types that implement `NamedInputModule` to enable
    /// receiving additional named inputs via graph `using()`.
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { None }

    /// Upcast to [`LoopBody`] for loop bodies that publish named per-iteration traces.
    /// Override in types that implement `LoopBody` to enable multi-output trace
    /// publishing via [`TraceEmit::publish`]. Default returns `None`, in which
    /// case the loop runner falls back to the legacy [`Module::trace`] path.
    fn as_loop_body(&self) -> Option<&dyn LoopBody> { None }

    /// Opt-in identity hook for framework downcasts.
    ///
    /// Override to return `Some(self)` in composite types the framework
    /// needs to recognize behind `dyn Module` (e.g. `Graph` for
    /// hierarchical tree composition — see `flodl::graph::GraphExt`,
    /// whose `.as_graph()` sugar downcasts through this hook).
    /// Transparent wrappers may return their *inner* composite instead
    /// of `self` — the contract is "the object this module presents to
    /// the framework", not strict identity. Default: `None` (plain
    /// leaf module, nothing to present).
    fn as_any(&self) -> Option<&dyn std::any::Any> { None }

    /// SHA-256 hex hash of module architecture for checkpoint validation.
    /// Override in composite modules (Graph) that compute a deterministic
    /// hash from their topology and parameter shapes.
    fn structural_hash(&self) -> Option<String> { None }

    /// Reset internal state (e.g. recurrent hidden state) between sequences.
    /// Called by loops before iterating to clear stale tensors whose
    /// grad_fns may reference freed saved tensors.
    /// Override in stateful modules.
    fn reset(&self) {}

    /// Detach internal state from the computation graph (for truncated BPTT).
    /// Called between training steps to break gradient chains on state
    /// carried across forward passes (e.g., recurrent hidden state).
    /// Override in stateful modules.
    fn detach_state(&self) {}

    /// Hand the framework the model's shared slot for coord-broadcast
    /// aggregated [`crate::metrics::EpochMetrics`]. The cluster-
    /// rank worker setup calls this at construction and stores the
    /// returned `Arc` clone alongside its own — both ends then point
    /// at the same `Mutex`, so the bridge thread's writes are visible
    /// to the user's main-thread reads (`Graph::latest_metrics`,
    /// `Graph::aggregated_gpu_tabs` — with `Graph` = `flodl::graph::Graph`).
    ///
    /// Default: `None` (model doesn't expose a slot; the worker
    /// creates a private one). `Graph` overrides to return its
    /// owned slot so user code sees the aggregated view through the
    /// same `Graph` reference it already passes to
    /// [`crate::monitor::Monitor::log`].
    fn aggregated_metrics_slot(
        &self,
    ) -> Option<std::sync::Arc<
        std::sync::Mutex<Option<crate::metrics::EpochMetrics>>,
    >> {
        None
    }
}

impl Module for Box<dyn Module> {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        (**self).forward(input)
    }
    fn parameters(&self) -> Vec<Parameter> {
        (**self).parameters()
    }
    fn buffers(&self) -> Vec<Buffer> {
        (**self).buffers()
    }
    fn name(&self) -> &str {
        (**self).name()
    }
    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        (**self).sub_modules()
    }
    fn move_to_device(&self, device: crate::tensor::Device) {
        (**self).move_to_device(device);
    }
    fn set_training(&self, training: bool) {
        (**self).set_training(training);
    }
    fn trace(&self) -> Option<Variable> {
        (**self).trace()
    }
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        (**self).as_named_input()
    }
    fn as_loop_body(&self) -> Option<&dyn LoopBody> {
        (**self).as_loop_body()
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        (**self).as_any()
    }
    fn structural_hash(&self) -> Option<String> {
        (**self).structural_hash()
    }
    fn reset(&self) {
        (**self).reset();
    }
    fn detach_state(&self) {
        (**self).detach_state();
    }
    fn aggregated_metrics_slot(
        &self,
    ) -> Option<std::sync::Arc<
        std::sync::Mutex<Option<crate::metrics::EpochMetrics>>,
    >> {
        (**self).aggregated_metrics_slot()
    }
}

/// Module that can receive additional named inputs via graph `using()`.
pub trait NamedInputModule: Module {
    /// Forward pass with additional named inputs from tagged graph nodes.
    /// `refs` maps tag names to their current values, as wired by `FlowBuilder::using()`.
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable>;
}

/// Per-iteration emit channel handed to [`LoopBody::step`] by the loop runner.
///
/// Body code calls [`TraceEmit::publish`] to publish named auxiliary outputs
/// (one per published name per iteration). The runner harvests the map after
/// each step and appends each entry into the loop's per-name vector, surfaced
/// downstream via [`crate::graph::Graph::traces`] and `LossContext::traces`.
///
/// Use [`TraceEmit::discard`] when calling `step` outside a loop runner
/// (typically from a `Module::forward` shim that delegates to `step` for
/// bodies that don't have a separate non-loop forward path).
pub struct TraceEmit<'a> {
    named: Option<&'a mut HashMap<String, Variable>>,
}

impl<'a> TraceEmit<'a> {
    /// Construct an emitter that drops every publish call. Useful when calling
    /// [`LoopBody::step`] from a non-loop context where traces are irrelevant.
    /// Lifetime is generic so the result coerces into any caller's expected
    /// `&mut TraceEmit<'_>` slot (mutable references are invariant in their
    /// lifetime parameter, so a fixed `'static` would not coerce).
    pub fn discard() -> TraceEmit<'a> {
        TraceEmit { named: None }
    }

    pub(crate) fn new(named: &'a mut HashMap<String, Variable>) -> Self {
        TraceEmit { named: Some(named) }
    }

    /// Publish a named per-iteration value. Panics if `name` was already
    /// published in the same step (last-write-wins is not the contract;
    /// duplicate publishes within one step are a body-author bug).
    pub fn publish(&mut self, name: &str, v: Variable) {
        if let Some(named) = self.named.as_deref_mut() {
            if named.contains_key(name) {
                panic!(
                    "TraceEmit::publish: name {:?} already published this step",
                    name
                );
            }
            named.insert(name.to_string(), v);
        }
    }
}

/// Loop body trait for modules that publish multiple named per-iteration traces.
///
/// Implementing `LoopBody` is opt-in. Bodies that don't implement it stay on
/// the legacy single-stream [`Module::trace`] path. Bodies that do implement
/// it can publish any number of named values per iteration via the [`TraceEmit`]
/// passed into [`step`](LoopBody::step), with no body-side `RefCell` state.
///
/// Refs are always passed (possibly empty), folding what would otherwise be a
/// separate "with refs" trait variant. This mirrors how
/// [`crate::graph::loop_node`] already handles ref-bearing bodies.
///
/// Bodies that have no meaningful standalone forward path can implement
/// `Module::forward` as a one-line shim using [`forward_via_step`]:
///
/// ```ignore
/// impl Module for ScanStep {
///     fn forward(&self, x: &Variable) -> Result<Variable> {
///         forward_via_step(self, x)
///     }
///     fn sub_modules(&self) -> Vec<Rc<dyn Module>> { /* ... */ vec![] }
/// }
/// impl LoopBody for ScanStep {
///     fn step(
///         &self,
///         x: &Variable,
///         _refs: &HashMap<String, Variable>,
///         emit: &mut TraceEmit<'_>,
///     ) -> Result<Variable> {
///         let h = self.h_proj.forward(x)?;
///         emit.publish("location", self.location_proj.forward(&h)?);
///         emit.publish("content_logit", self.out_proj.forward(&h)?);
///         Ok(h)
///     }
/// }
/// impl Module for ScanStep {
///     fn as_loop_body(&self) -> Option<&dyn LoopBody> { Some(self) }
/// }
/// ```
pub trait LoopBody: Module {
    /// Per-iteration step. Same contract as [`Module::forward`] but with
    /// auxiliary refs (always present, possibly empty) and an emitter for
    /// per-iteration named traces.
    fn step(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
        emit: &mut TraceEmit<'_>,
    ) -> Result<Variable>;
}

/// Convenience helper for implementing `Module::forward` on a `LoopBody`
/// that has no separate non-loop forward path. Allocates an empty refs map,
/// calls `step` with a discarding emitter, and returns the result.
pub fn forward_via_step<B: LoopBody + ?Sized>(
    body: &B,
    input: &Variable,
) -> Result<Variable> {
    let refs: HashMap<String, Variable> = HashMap::new();
    let mut emit = TraceEmit::discard();
    body.step(input, &refs, &mut emit)
}

/// Recursively walk a module tree, calling f on each module exactly once.
pub fn walk_modules(module: &dyn Module, f: &mut dyn FnMut(&dyn Module)) {
    let mut visited = HashSet::new();
    walk_modules_visited(module, &mut visited, f);
}

/// Walk a module tree with an externally-managed visited set.
/// Use instead of [`walk_modules`] when walking multiple root modules
/// (e.g., all graph nodes) while sharing dedup state to avoid visiting
/// shared sub-modules more than once.
pub fn walk_modules_visited(
    module: &dyn Module,
    visited: &mut HashSet<usize>,
    f: &mut dyn FnMut(&dyn Module),
) {
    let ptr = module as *const dyn Module as *const () as usize;
    if !visited.insert(ptr) {
        return;
    }
    f(module);
    for child in module.sub_modules() {
        walk_modules_visited(child.as_ref(), visited, f);
    }
}

/// Collect parameters from multiple modules (convenience function).
/// Does not deduplicate across modules -- use `parameters()` on a single
/// composite module (e.g., Graph) for pointer-based dedup.
///
/// ```ignore
/// let l1 = Linear::new(3, 4)?;
/// let l2 = Linear::new(4, 2)?;
/// let params = collect_parameters(&[&l1, &l2]); // 4 params (2 per layer)
/// ```
pub fn collect_parameters(modules: &[&dyn Module]) -> Vec<Parameter> {
    let mut params = Vec::new();
    for m in modules {
        params.extend(m.parameters());
    }
    params
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
