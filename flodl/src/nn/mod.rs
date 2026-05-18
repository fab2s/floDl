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
pub use optim::{Optimizer, Stateful, SGD, SGDBuilder, Adam, AdamBuilder, AdamW, AdamWBuilder, RMSprop, RMSpropBuilder, Adagrad, AdagradBuilder, RAdam, NAdam};
pub use checkpoint::{
    save_checkpoint, load_checkpoint, save_checkpoint_file, load_checkpoint_file,
    migrate_checkpoint, migrate_checkpoint_file, checkpoint_version, checkpoint_keys,
    LoadReport, MigrateReport,
};
pub use amp::{GradScaler, cast_parameters, AutocastGuard, autocast, is_autocast_enabled};
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
use crate::graph::Graph;
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
    /// Leaf modules should override to return their own parameters.
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
                    let ptr = Rc::as_ptr(&p.variable.inner) as usize;
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
                    let ptr = Rc::as_ptr(&b.inner) as usize;
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
    /// Override in modules like BatchNorm that hold non-parameter state.
    fn move_to_device(&self, _device: crate::tensor::Device) {}

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

    /// Upcast to [`Graph`] for hierarchical tree composition.
    /// Override in Graph to enable subgraph nesting with label-path addressing.
    fn as_graph(&self) -> Option<&Graph> { None }

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
    /// aggregated [`crate::distributed::EpochMetrics`]. The cluster-
    /// rank worker setup calls this at construction and stores the
    /// returned `Arc` clone alongside its own — both ends then point
    /// at the same `Mutex`, so the bridge thread's writes are visible
    /// to the user's main-thread reads (`Graph::latest_metrics`,
    /// `Graph::aggregated_gpu_tabs`).
    ///
    /// Default: `None` (model doesn't expose a slot; the worker
    /// creates a private one). [`Graph`] overrides to return its
    /// owned slot so user code sees the aggregated view through the
    /// same `Graph` reference it already passes to
    /// [`crate::monitor::Monitor::log`].
    fn aggregated_metrics_slot(
        &self,
    ) -> Option<std::sync::Arc<
        std::sync::Mutex<Option<crate::distributed::EpochMetrics>>,
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
    fn as_graph(&self) -> Option<&Graph> {
        (**self).as_graph()
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
        std::sync::Mutex<Option<crate::distributed::EpochMetrics>>,
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
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::tensor::Tensor;

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
    }

    #[test]
    fn test_linear_forward() {
        let model = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();

        // Set known weights for deterministic test
        // W = [[1,2,3],[4,5,6]] shape [2,3]
        model.weight.variable.set_data(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        );
        model.bias.as_ref().unwrap().variable.set_data(
            from_f32(&[0.1, 0.2], &[2]),
        );

        // x = [[1, 1, 1]] shape [1, 3]
        let x = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[1, 3]), false);
        let y = model.forward(&x).unwrap();

        // y = x @ W^T + b = [[1,1,1]] @ [[1,4],[2,5],[3,6]] + [0.1,0.2]
        // = [[6, 15]] + [[0.1, 0.2]] = [[6.1, 15.2]]
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 6.1).abs() < 1e-5);
        assert!((data[1] - 15.2).abs() < 1e-5);
    }

    #[test]
    fn test_linear_backward() {
        let model = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        model.weight.variable.set_data(
            from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        );
        model.bias.as_ref().unwrap().variable.set_data(
            from_f32(&[0.0, 0.0], &[2]),
        );

        let x = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[1, 3]), true);
        let y = model.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // dL/dW = grad_out^T @ x
        // grad_out = [[1,1]], x = [[1,1,1]]
        // dL/dW = [[1],[1]] @ [[1,1,1]] = [[1,1,1],[1,1,1]]
        let gw = model.weight.variable.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gw, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // dL/db = sum of grad_out along batch = [1, 1]
        let gb = model.bias.as_ref().unwrap().variable.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gb, vec![1.0, 1.0]);

        // dL/dx = grad_out @ W = [[1,1]] @ [[1,2,3],[4,5,6]] = [[5,7,9]]
        let gx = x.grad().unwrap().to_f32_vec().unwrap();
        assert_eq!(gx, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let target = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let loss = mse_loss(&pred, &target).unwrap();
        assert!((loss.item().unwrap()).abs() < 1e-7);

        let pred2 = Variable::new(from_f32(&[2.0, 3.0, 4.0], &[3]), false);
        let loss2 = mse_loss(&pred2, &target).unwrap();
        // (1² + 1² + 1²) / 3 = 1.0
        assert!((loss2.item().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_step() {
        let model = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();
        model.weight.variable.set_data(from_f32(&[1.0, 1.0], &[1, 2]));
        model.bias.as_ref().unwrap().variable.set_data(from_f32(&[0.0], &[1]));

        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[5.0], &[1, 1]), false);

        // Forward + backward
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        let loss_before = loss.item().unwrap();
        loss.backward().unwrap();
        optim.step().unwrap();

        // Loss should decrease after one step
        optim.zero_grad();
        let pred2 = model.forward(&x).unwrap();
        let loss2 = mse_loss(&pred2, &target).unwrap();
        assert!(loss2.item().unwrap() < loss_before, "loss should decrease");
    }

    #[test]
    fn test_linear_regression_sgd() {
        // y = 2*x + 1
        let model = Linear::on_device(1, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.0);

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]),
            false,
        );
        let target = Variable::new(
            from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]),
            false,
        );

        let mut last_loss = f64::MAX;
        for _ in 0..800 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(
            last_loss < 0.01,
            "SGD should converge on linear regression, got loss={}",
            last_loss
        );
    }

    #[test]
    fn test_linear_regression_adam() {
        // y = 2*x + 1
        let model = Linear::on_device(1, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = Adam::new(&params, 0.1);

        let x = Variable::new(
            from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]),
            false,
        );
        let target = Variable::new(
            from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]),
            false,
        );

        let mut last_loss = f64::MAX;
        for _ in 0..500 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(
            last_loss < 0.02,
            "Adam should converge on linear regression, got loss={}",
            last_loss
        );
    }

    #[test]
    fn test_relu_module() {
        let relu = ReLU::new();
        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0, -2.0], &[4]), false);
        let y = relu.forward(&x).unwrap();
        assert_eq!(y.data().to_f32_vec().unwrap(), vec![1.0, 0.0, 2.0, 0.0]);
        assert!(relu.parameters().is_empty());
    }

    #[test]
    fn test_collect_parameters() {
        let l1 = Linear::on_device(3, 4, crate::tensor::test_device()).unwrap();
        let l2 = Linear::on_device(4, 2, crate::tensor::test_device()).unwrap();
        let params = collect_parameters(&[&l1, &l2]);
        // l1: weight + bias = 2, l2: weight + bias = 2
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_sgd_momentum() {
        let model = Linear::on_device(1, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.9);

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]), false);
        let target = Variable::new(from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..200 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(
            last_loss < 0.01,
            "SGD with momentum should converge, got loss={}",
            last_loss
        );
    }

    // --- Loss function tests ---

    #[test]
    fn test_cross_entropy_loss() {
        // 2 classes, batch of 2
        // Logits: [[2.0, 1.0], [1.0, 3.0]]
        // One-hot targets: [[1, 0], [0, 1]]
        let pred = Variable::new(
            from_f32(&[2.0, 1.0, 1.0, 3.0], &[2, 2]),
            true,
        );
        let target = Variable::new(
            from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            false,
        );
        let loss = cross_entropy_loss(&pred, &target).unwrap();
        let val = loss.item().unwrap();

        // Manual: log_softmax([2,1]) = [2-log(e²+e), 1-log(e²+e)]
        // e² + e ≈ 10.107
        // log_softmax([2,1]) ≈ [2-2.313, 1-2.313] = [-0.313, -1.313]
        // For sample 0: target=[1,0] → -(-0.313) = 0.313
        // log_softmax([1,3]) = [1-log(e+e³), 3-log(e+e³)]
        // e + e³ ≈ 22.804
        // log_softmax([1,3]) ≈ [1-3.127, 3-3.127] = [-2.127, -0.127]
        // For sample 1: target=[0,1] → -(-0.127) = 0.127
        // mean = (0.313 + 0.127) / 2 = 0.220
        assert!(val > 0.0, "cross entropy should be positive");
        assert!((val - 0.22).abs() < 0.02, "expected ~0.22, got {}", val);

        // Test backward
        loss.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_cross_entropy_converges() {
        // Train a Linear to classify 2 classes
        let model = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        // Data: class 0 = [1, 0], class 1 = [0, 1]
        let x = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), false);
        let target = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..200 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = cross_entropy_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(last_loss < 0.1, "cross entropy should converge, got {}", last_loss);
    }

    #[test]
    fn test_bce_with_logits_loss() {
        // Logits = 0 → sigmoid(0) = 0.5, target = 1
        // BCE = -log(0.5) = 0.693
        let pred = Variable::new(from_f32(&[0.0], &[1]), true);
        let target = Variable::new(from_f32(&[1.0], &[1]), false);
        let loss = bce_with_logits_loss(&pred, &target).unwrap();
        let val = loss.item().unwrap();
        assert!(
            (val - 0.693).abs() < 0.01,
            "expected ~0.693, got {}",
            val
        );

        // Large positive logit, target=1 → loss ≈ 0
        let pred2 = Variable::new(from_f32(&[10.0], &[1]), false);
        let loss2 = bce_with_logits_loss(&pred2, &target).unwrap();
        assert!(loss2.item().unwrap() < 0.001);

        // Backward
        loss.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_l1_loss() {
        let pred = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let target = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
        let loss = l1_loss(&pred, &target).unwrap();
        assert!((loss.item().unwrap()).abs() < 1e-6);

        let pred2 = Variable::new(from_f32(&[2.0, 4.0, 6.0], &[3]), true);
        let loss2 = l1_loss(&pred2, &target).unwrap();
        // |2-1| + |4-2| + |6-3| = 1+2+3 = 6, mean = 2.0
        assert!((loss2.item().unwrap() - 2.0).abs() < 1e-5);

        loss2.backward().unwrap();
        assert!(pred2.grad().is_some());
    }

    #[test]
    fn test_smooth_l1_loss() {
        // Small diff (within beta): 0.5 * x² / beta
        let pred = Variable::new(from_f32(&[1.5], &[1]), true);
        let target = Variable::new(from_f32(&[1.0], &[1]), false);
        let loss = smooth_l1_loss(&pred, &target, 1.0).unwrap();
        // |0.5| < 1.0 → 0.5 * 0.25 / 1.0 = 0.125
        assert!((loss.item().unwrap() - 0.125).abs() < 1e-5, "got {}", loss.item().unwrap());

        // Large diff (outside beta): |x| - 0.5*beta
        let pred2 = Variable::new(from_f32(&[3.0], &[1]), true);
        let loss2 = smooth_l1_loss(&pred2, &target, 1.0).unwrap();
        // |2.0| >= 1.0 → 2.0 - 0.5 = 1.5
        assert!((loss2.item().unwrap() - 1.5).abs() < 1e-5);

        loss2.backward().unwrap();
        assert!(pred2.grad().is_some());
    }

    #[test]
    fn test_kl_div_loss() {
        // KL(P || Q) where Q = log_probs, P = probs
        // Uniform distribution: KL = 0
        let log_probs = Variable::new(
            from_f32(&[-0.693, -0.693, -0.693, -0.693], &[2, 2]),
            true,
        );
        let probs = Variable::new(
            from_f32(&[0.5, 0.5, 0.5, 0.5], &[2, 2]),
            false,
        );
        let loss = kl_div_loss(&log_probs, &probs).unwrap();
        // When Q ≈ P, KL ≈ 0
        assert!(loss.item().unwrap().abs() < 0.01, "KL should be ~0, got {}", loss.item().unwrap());

        loss.backward().unwrap();
        assert!(log_probs.grad().is_some());
    }

    // --- Gradient clipping tests ---

    #[test]
    fn test_clip_grad_norm() {
        crate::manual_seed(42);
        let model = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();

        let x = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[0.0], &[1, 1]), false);
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        loss.backward().unwrap();

        let norm_before = clip_grad_norm(&params, 1.0).unwrap();
        assert!(norm_before > 1.0, "large input should produce large gradients");

        // After clipping, total norm should be <= max_norm
        let mut total_sq = 0.0f64;
        for p in &params {
            if let Some(g) = p.variable.grad() {
                for &v in &g.to_f32_vec().unwrap() {
                    total_sq += (v as f64) * (v as f64);
                }
            }
        }
        let clipped_norm = total_sq.sqrt();
        assert!(
            clipped_norm <= 1.0 + 1e-5,
            "clipped norm should be <= 1.0, got {}",
            clipped_norm
        );
    }

    #[test]
    fn test_clip_grad_value() {
        let model = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();

        let x = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[0.0], &[1, 1]), false);
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        loss.backward().unwrap();

        clip_grad_value(&params, 0.5).unwrap();

        for p in &params {
            if let Some(g) = p.variable.grad() {
                for &v in &g.to_f32_vec().unwrap() {
                    assert!(
                        v.abs() <= 0.5 + 1e-6,
                        "all grads should be clamped to [-0.5, 0.5], got {}",
                        v
                    );
                }
            }
        }
    }

    // --- Scheduler tests ---

    #[test]
    fn test_step_decay_scheduler() {
        let sched = StepDecay::new(0.1, 3, 0.5);

        assert!((sched.lr(0) - 0.1).abs() < 1e-10);   // step 0: no decay
        assert!((sched.lr(1) - 0.1).abs() < 1e-10);   // step 1: no decay
        assert!((sched.lr(2) - 0.1).abs() < 1e-10);   // step 2: no decay
        assert!((sched.lr(3) - 0.05).abs() < 1e-10);  // step 3: first decay
        assert!((sched.lr(4) - 0.05).abs() < 1e-10);  // step 4
        assert!((sched.lr(5) - 0.05).abs() < 1e-10);  // step 5
        assert!((sched.lr(6) - 0.025).abs() < 1e-10); // step 6: second decay
    }

    #[test]
    fn test_cosine_scheduler() {
        let sched = CosineScheduler::new(0.1, 0.001, 100);

        // At step 0, lr = base_lr
        assert!((sched.lr(0) - 0.1).abs() < 1e-10);

        // At step 50, lr should be roughly midpoint
        let mid_lr = sched.lr(50);
        assert!(mid_lr > 0.001 && mid_lr < 0.1, "mid lr={}", mid_lr);

        // At step 100, lr should be min_lr
        assert!((sched.lr(100) - 0.001).abs() < 1e-5, "end lr={}", sched.lr(100));
    }

    #[test]
    fn test_plateau_scheduler() {
        let mut sched = PlateauScheduler::new(0.1, 3, 0.5, 0.001);

        // Improving metrics: no decay
        sched.observe(1.0);
        sched.observe(0.9);
        sched.observe(0.8);
        assert!((sched.lr() - 0.1).abs() < 1e-10);

        // Stagnating: patience=3
        sched.observe(0.81); // wait=1
        sched.observe(0.82); // wait=2
        sched.observe(0.83); // wait=3 → decay
        assert!((sched.lr() - 0.05).abs() < 1e-10);
    }

    // --- GELU / SiLU tests ---

    #[test]
    fn test_gelu() {
        let gelu = GELU;
        let x = Variable::new(from_f32(&[0.0, 1.0, -1.0], &[3]), true);
        let y = gelu.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
        assert!(data[0].abs() < 0.01, "GELU(0)={}", data[0]);
        assert!((data[1] - 0.841).abs() < 0.01, "GELU(1)={}", data[1]);
        assert!((data[2] - (-0.159)).abs() < 0.01, "GELU(-1)={}", data[2]);

        // Backward
        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
    }

    /// Both [`GeluApprox`] variants dispatch to libtorch and produce
    /// mathematically close output (within 1e-2 of each other) but
    /// must be distinct — bitwise equality at every non-zero input
    /// would mean the dispatch silently picked the same path.
    #[test]
    fn test_gelu_variants_distinct() {
        // Avoid 0 — both forms agree exactly there.
        let xs = [0.5_f32, 1.0, -1.0, 2.0];
        let x = Variable::new(from_f32(&xs, &[xs.len() as i64]), false);

        let erf  = GELU::exact();
        let tanh = GELU::tanh();

        let v_erf  = erf.forward(&x).unwrap().data().to_f32_vec().unwrap();
        let v_tanh = tanh.forward(&x).unwrap().data().to_f32_vec().unwrap();

        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
        };

        let d_erf_tanh = max_diff(&v_erf, &v_tanh);
        assert!(d_erf_tanh < 1e-2, "erf/tanh too far: {d_erf_tanh}");
        assert!(d_erf_tanh > 1e-5, "erf/tanh suspiciously close — same path?");
    }

    #[test]
    fn test_silu() {
        let silu = SiLU::new();
        let x = Variable::new(from_f32(&[0.0, 2.0, -2.0], &[3]), true);
        let y = silu.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // SiLU(0) = 0, SiLU(2) = 2*sigmoid(2) ≈ 1.762, SiLU(-2) = -2*sigmoid(-2) ≈ -0.238
        assert!(data[0].abs() < 0.01);
        assert!((data[1] - 1.762).abs() < 0.02, "SiLU(2)={}", data[1]);

        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
    }

    // --- Dropout test ---

    #[test]
    fn test_dropout() {
        let drop = Dropout::new(0.5);

        // Training mode: some values should be zeroed
        let x = Variable::new(from_f32(&[1.0; 100], &[10, 10]), false);
        let y = drop.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        let zeros = data.iter().filter(|&&v| v == 0.0).count();
        // With p=0.5, roughly half should be zero (very approximate)
        assert!(zeros > 10 && zeros < 90, "zeros={} of 100", zeros);
        // Non-zero values should be scaled by 1/(1-0.5) = 2
        for &v in &data {
            if v != 0.0 {
                assert!((v - 2.0).abs() < 1e-5, "scaled value should be 2.0, got {}", v);
            }
        }

        // Eval mode: identity
        drop.set_training(false);
        let y_eval = drop.forward(&x).unwrap();
        let eval_data = y_eval.data().to_f32_vec().unwrap();
        assert!(eval_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    // --- LayerNorm test ---

    #[test]
    fn test_layernorm() {
        let ln = LayerNorm::on_device(4, crate::tensor::test_device()).unwrap();
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let y = ln.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        assert_eq!(y.shape(), vec![1, 4]);

        // After normalization, mean ≈ 0, std ≈ 1 (before gamma/beta)
        // With gamma=1, beta=0 (defaults), output should be normalized
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 0.1, "mean should be ~0, got {}", mean);

        let loss = y.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(ln.parameters().len(), 2); // weight + bias
    }

    // --- Embedding test ---

    #[test]
    fn test_embedding() {
        let dev = crate::tensor::test_device();
        let emb = Embedding::on_device(5, 3, dev).unwrap();

        // Look up indices [0, 2, 4]
        let indices = Variable::new(
            Tensor::from_i64(&[0, 2, 4], &[3], dev).unwrap(), false,
        );
        let y = emb.forward(&indices).unwrap();
        assert_eq!(y.shape(), vec![3, 3]); // 3 tokens × 3 dims

        // Same index should give same embedding
        let idx_same = Variable::new(
            Tensor::from_i64(&[1, 1], &[2], dev).unwrap(), false,
        );
        let y2 = emb.forward(&idx_same).unwrap();
        let data = y2.data().to_f32_vec().unwrap();
        assert_eq!(&data[0..3], &data[3..6]);

        assert_eq!(emb.parameters().len(), 1); // weight only
    }

    #[test]
    fn test_embedding_backward() {
        let dev = crate::tensor::test_device();
        let emb = Embedding::on_device(5, 3, dev).unwrap();
        let indices = Variable::new(
            Tensor::from_i64(&[0, 2], &[2], dev).unwrap(), false,
        );
        let y = emb.forward(&indices).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = emb.weight.variable.grad().unwrap();
        let grad_data = grad.to_f32_vec().unwrap();
        // Rows 0 and 2 should have gradient, others zero
        assert!(grad_data[0..3].iter().all(|&v| (v - 1.0).abs() < 1e-5)); // row 0
        assert!(grad_data[3..6].iter().all(|&v| v.abs() < 1e-5)); // row 1 (unused)
        assert!(grad_data[6..9].iter().all(|&v| (v - 1.0).abs() < 1e-5)); // row 2
    }

    // --- GRUCell test ---

    #[test]
    fn test_grucell() {
        let gru = GRUCell::on_device(4, 3, crate::tensor::test_device()).unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let h1 = gru.forward_step(&x, None).unwrap();
        assert_eq!(h1.shape(), vec![1, 3]);

        // Second step with previous hidden state
        let h2 = gru.forward_step(&x, Some(&h1)).unwrap();
        assert_eq!(h2.shape(), vec![1, 3]);

        // Backward
        let loss = h2.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // 4 packed params: w_ih, w_hh, b_ih, b_hh
        assert_eq!(gru.parameters().len(), 4);
    }

    // --- LSTMCell test ---

    #[test]
    fn test_lstmcell() {
        let lstm = LSTMCell::on_device(4, 3, crate::tensor::test_device()).unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), true);
        let state1 = lstm.forward_step(&x, None).unwrap();
        assert_eq!(state1.shape(), vec![1, 6]); // packed: [batch, 2*hidden]

        // Unpack h
        let h1 = state1.narrow(1, 0, 3).unwrap();
        assert_eq!(h1.shape(), vec![1, 3]);

        // Second step
        let state2 = lstm.forward_step(&x, Some(&state1)).unwrap();
        assert_eq!(state2.shape(), vec![1, 6]);

        // Backward
        let loss = state2.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // 4 packed params: w_ih, w_hh, b_ih, b_hh
        assert_eq!(lstm.parameters().len(), 4);
    }

    #[test]
    fn test_conv2d() {
        let conv = Conv2d::build(1, 2, 3, true, [1, 1], [0, 0], [1, 1], 1, crate::tensor::test_device()).unwrap();
        // Input: [batch=1, channels=1, h=5, w=5]
        let x = Variable::new(
            Tensor::randn(&[1, 1, 5, 5], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        // With kernel=3, no padding: output = 5-3+1 = 3
        assert_eq!(out.shape(), vec![1, 2, 3, 3]);

        // Backward
        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // weight + bias = 2 params
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_conv2d_no_bias() {
        let conv = Conv2d::build(3, 8, 3, false, [1, 1], [0, 0], [1, 1], 1, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 3, 8, 8], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![2, 8, 6, 6]);
        assert_eq!(conv.parameters().len(), 1); // weight only
    }

    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2d::build(1, 1, 3, true, [1, 1], [1, 1], [1, 1], 1, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 5, 5], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        // Same padding: output = input size
        assert_eq!(out.shape(), vec![1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d() {
        let conv = ConvTranspose2d::build(2, 1, 3, true, [1, 1], [0, 0], [0, 0], [1, 1], 1, crate::tensor::test_device()).unwrap();
        // Input: [batch=1, channels=2, h=3, w=3]
        let x = Variable::new(
            Tensor::randn(&[1, 2, 3, 3], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        // With kernel=3, no padding: output = 3+3-1 = 5
        assert_eq!(out.shape(), vec![1, 1, 5, 5]);

        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_training() {
        let bn = BatchNorm::on_device(4, crate::tensor::test_device()).unwrap();
        // Input: [batch=8, features=4]
        let x = Variable::new(
            Tensor::randn(&[8, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![8, 4]);

        // Output should be roughly normalized: mean ≈ 0 (with gamma=1, beta=0)
        let out_data = out.data().to_f32_vec().unwrap();
        let mean: f32 = out_data.iter().sum::<f32>() / out_data.len() as f32;
        assert!(mean.abs() < 0.5, "mean should be close to 0, got {}", mean);

        // Backward
        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_eval() {
        let bn = BatchNorm::on_device(3, crate::tensor::test_device()).unwrap();

        // Run a few training steps to populate running stats
        for _ in 0..5 {
            let x = Variable::new(
                Tensor::randn(&[4, 3], crate::tensor::test_opts()).unwrap(),
                false,
            );
            bn.forward(&x).unwrap();
        }

        // Switch to eval mode
        bn.set_training(false);
        let x = Variable::new(
            Tensor::randn(&[2, 3], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![2, 3]);
    }

    #[test]
    fn test_adamw() {
        let w = Parameter {
            variable: Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true),
            name: "w".into(),
        };
        let params = vec![w.clone()];
        let mut opt = AdamW::new(&params, 0.01, 0.01);

        // Simulate a gradient
        let loss = w.variable.mul_scalar(2.0).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let before = w.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let after = w.variable.data().to_f32_vec().unwrap();

        // Parameters should have changed
        assert!(before != after, "AdamW should update parameters");
        opt.zero_grad();
    }

    #[test]
    fn test_xavier_init() {
        let t = init::xavier_uniform(&[10, 20], 10, 20, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![10, 20]);
        let data = t.to_f32_vec().unwrap();
        let bound = (6.0 / 30.0_f64).sqrt() as f32;
        for &v in &data {
            assert!(v >= -bound - 0.01 && v <= bound + 0.01,
                "xavier_uniform value {} out of bounds [{}, {}]", v, -bound, bound);
        }

        let t = init::xavier_normal(&[10, 20], 10, 20, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![10, 20]);
    }

    // --- New tensor ops tests ---

    #[test]
    fn test_linspace_arange() {
        let t = Tensor::linspace(0.0, 1.0, 5, crate::tensor::test_opts()).unwrap();
        assert_eq!(t.shape(), vec![5]);
        let data = t.to_f32_vec().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[4] - 1.0).abs() < 1e-5);

        let t = Tensor::arange(0.0, 5.0, 1.0, crate::tensor::test_opts()).unwrap();
        assert_eq!(t.shape(), vec![5]);
        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_min_max_argmax() {
        let t = from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
        assert!((t.min().unwrap().item().unwrap() - 1.0).abs() < 1e-5);

        let min_d1 = t.min_dim(1, false).unwrap();
        assert_eq!(min_d1.shape(), vec![2]);
        let data = min_d1.to_f32_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);

        let max_d1 = t.max_dim(1, false).unwrap();
        let data = max_d1.to_f32_vec().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 9.0).abs() < 1e-5);

        let am = t.argmax(1, false).unwrap();
        assert_eq!(am.shape(), vec![2]);
    }

    #[test]
    fn test_comparisons() {
        let t = from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let ge = t.ge_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(ge, vec![0.0, 1.0, 1.0, 1.0]);
        let le = t.le_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(le, vec![1.0, 1.0, 0.0, 0.0]);
        let lt = t.lt_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(lt, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let squeezed = x.squeeze(0).unwrap();
        assert_eq!(squeezed.shape(), vec![3]);

        let unsqueezed = squeezed.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), vec![3, 1]);

        // Backward
        let loss = unsqueezed.sum().unwrap();
        loss.backward().unwrap();
        assert_eq!(x.grad().unwrap().shape(), vec![1, 3]);
    }

    #[test]
    fn test_where_cond() {
        let cond = from_f32(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let x = from_f32(&[10.0, 20.0, 30.0, 40.0], &[4]);
        let y = from_f32(&[-1.0, -2.0, -3.0, -4.0], &[4]);
        let result = Tensor::where_cond(&cond, &x, &y).unwrap().to_f32_vec().unwrap();
        assert_eq!(result, vec![10.0, -2.0, 30.0, -4.0]);
    }

    #[test]
    fn test_to_dtype() {
        use crate::tensor::DType;
        let t = from_f32(&[1.5, 2.7], &[2]);
        let t64 = t.to_dtype(DType::Float64).unwrap();
        assert_eq!(t64.dtype(), DType::Float64);
    }

    #[test]
    fn test_all_finite() {
        let t = from_f32(&[1.0, 2.0, 3.0], &[3]);
        assert!(t.all_finite().unwrap());
    }

    // --- Checkpoint tests ---

    #[test]
    fn test_save_load_checkpoint() {
        let model = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let params = model.parameters();

        // Set known weights
        params[0].variable.set_data(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        params[1].variable.set_data(from_f32(&[0.1, 0.2], &[2]));

        // Save as named params
        let named: Vec<(String, Parameter)> = params.into_iter()
            .map(|p| (format!("linear/{}", p.name), p))
            .collect();
        let mut buf = Vec::new();
        checkpoint::save_checkpoint(&mut buf, &named, &[], None).unwrap();

        // Create new model, load by name
        let model2 = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let named2: Vec<(String, Parameter)> = model2.parameters().into_iter()
            .map(|p| (format!("linear/{}", p.name), p))
            .collect();
        let mut cursor = std::io::Cursor::new(&buf);
        let report = checkpoint::load_checkpoint(&mut cursor, &named2, &[], None).unwrap();

        assert_eq!(report.loaded.len(), 2);
        assert!(report.missing.is_empty());

        // Verify loaded weights match
        let w = named2[0].1.variable.data().to_f32_vec().unwrap();
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = named2[1].1.variable.data().to_f32_vec().unwrap();
        assert!((b[0] - 0.1).abs() < 1e-5 && (b[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_save_load_sgd_state() {
        use optim::Stateful;
        let model = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.9);

        // Do a step to populate velocity
        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[5.0], &[1, 1]), false);
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        loss.backward().unwrap();
        optim.step().unwrap();

        // Save state
        let mut buf = Vec::new();
        optim.save_state(&mut buf).unwrap();

        // Create new optimizer, load state
        let mut optim2 = SGD::new(&params, 0.5, 0.9); // different lr
        let mut cursor = std::io::Cursor::new(&buf);
        optim2.load_state(&mut cursor).unwrap();

        assert!((optim2.lr() - 0.1).abs() < 1e-10, "lr should be restored");
    }

    #[test]
    fn test_save_load_adam_state() {
        use optim::Stateful;
        let model = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = Adam::new(&params, 0.01);

        // Do two steps
        for _ in 0..2 {
            optim.zero_grad();
            let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
            let target = Variable::new(from_f32(&[5.0], &[1, 1]), false);
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        // Save
        let mut buf = Vec::new();
        optim.save_state(&mut buf).unwrap();

        // Load into new optimizer
        let mut optim2 = Adam::new(&params, 0.5);
        let mut cursor = std::io::Cursor::new(&buf);
        optim2.load_state(&mut cursor).unwrap();

        assert!((optim2.lr() - 0.01).abs() < 1e-10, "lr should be restored");
    }

    // --- Mixed precision tests ---

    #[test]
    fn test_cast_parameters() {
        use crate::tensor::DType;
        let model = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        assert_eq!(params[0].variable.data().dtype(), DType::Float32);

        amp::cast_parameters(&params, DType::Float64);
        assert_eq!(params[0].variable.data().dtype(), DType::Float64);
        assert_eq!(params[1].variable.data().dtype(), DType::Float64);

        // Round-trip back
        amp::cast_parameters(&params, DType::Float32);
        assert_eq!(params[0].variable.data().dtype(), DType::Float32);
    }

    #[test]
    fn test_grad_scaler_finite() {
        use optim::Stateful;
        let model = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = SGD::new(&params, 0.1, 0.0);

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let target = Variable::new(from_f32(&[5.0], &[1, 1]), false);

        let mut scaler = amp::GradScaler::new();
        let pred = model.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        let scaled_loss = scaler.scale(&loss).unwrap();
        scaled_loss.backward().unwrap();

        let params_for_step = model.parameters();
        let success = scaler.step(&params_for_step, &mut || optim.step()).unwrap();
        assert!(success, "step should succeed with finite gradients");
        scaler.update();

        // Save/load state
        let mut buf = Vec::new();
        scaler.save_state(&mut buf).unwrap();
        let mut scaler2 = amp::GradScaler::new();
        // scaler2 has default scale (65536), will be overwritten by load
        let mut cursor = std::io::Cursor::new(&buf);
        scaler2.load_state(&mut cursor).unwrap();
        assert!((scaler2.scale_factor() - scaler.scale_factor()).abs() < 1e-10);
    }

    #[test]
    fn test_autocast_guard_toggle() {
        assert!(!amp::is_autocast_enabled());
        {
            let _guard = amp::AutocastGuard::new(crate::tensor::DType::Float16);
            assert!(amp::is_autocast_enabled());
        }
        assert!(!amp::is_autocast_enabled());
    }

    #[test]
    fn test_autocast_nesting() {
        assert!(!amp::is_autocast_enabled());
        let outer = amp::AutocastGuard::new(crate::tensor::DType::Float16);
        assert!(amp::is_autocast_enabled());
        {
            let _inner = amp::AutocastGuard::new(crate::tensor::DType::Float16);
            assert!(amp::is_autocast_enabled());
        }
        // Outer still active
        assert!(amp::is_autocast_enabled());
        drop(outer);
        assert!(!amp::is_autocast_enabled());
    }

    #[test]
    fn test_autocast_closure() {
        assert!(!amp::is_autocast_enabled());
        amp::autocast(crate::tensor::DType::Float16, || {
            assert!(amp::is_autocast_enabled());
        });
        assert!(!amp::is_autocast_enabled());
    }

    #[test]
    fn test_autocast_matmul_output_dtype() {
        // Under autocast, matmul of fp32 tensors should produce lower-precision output
        let opts = crate::tensor::test_opts();
        let dev = crate::tensor::test_device();
        let (device_type, _) = dev.to_ffi();
        let a = Tensor::randn(&[4, 4], opts).unwrap();
        let b = Tensor::randn(&[4, 4], opts).unwrap();

        // Without autocast: fp32 * fp32 = fp32
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dtype(), crate::tensor::DType::Float32);

        // CUDA autocast uses Float16, CPU autocast uses BFloat16
        let cast_dtype = if dev.is_cuda() {
            crate::tensor::DType::Float16
        } else {
            crate::tensor::DType::BFloat16
        };
        let _guard = amp::AutocastGuard::for_device(device_type, cast_dtype);
        let c_amp = a.matmul(&b).unwrap();
        assert_eq!(c_amp.dtype(), cast_dtype,
            "matmul under autocast should produce {:?}, got {:?}", cast_dtype, c_amp.dtype());
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        use crate::autograd::adaptive_avg_pool2d;
        // Input: [batch=1, channels=1, h=4, w=4]
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = adaptive_avg_pool2d(&x, [2, 2]).unwrap();
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);

        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(x.grad().unwrap().shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_grid_sample() {
        use crate::autograd::grid_sample;
        // Input: [batch=1, channels=1, h=4, w=4]
        let input = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        // Grid: [batch=1, out_h=2, out_w=2, 2]
        let grid = Variable::new(
            Tensor::rand(&[1, 2, 2, 2], crate::tensor::test_opts()).unwrap()
                .mul_scalar(2.0).unwrap()
                .add_scalar(-1.0).unwrap(), // map to [-1, 1]
            true,
        );
        // mode=0 (bilinear), padding_mode=0 (zeros), align_corners=true
        let out = grid_sample(&input, &grid, 0, 0, true).unwrap();
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);

        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(input.grad().is_some());
        assert!(grid.grad().is_some());
    }

    // --- Identity module ---

    #[test]
    fn test_identity() {
        let id = activation::Identity;
        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
        let y = id.forward(&x).unwrap();
        assert_eq!(y.data().to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
        assert!(id.parameters().is_empty());
    }

    // --- Cross-entropy with class indices ---

    #[test]
    fn test_cross_entropy_indices() {
        // Same test as test_cross_entropy_loss but with integer indices
        // Logits: [[2.0, 1.0], [1.0, 3.0]]
        // Targets: class 0, class 1 (instead of one-hot)
        let pred = Variable::new(
            from_f32(&[2.0, 1.0, 1.0, 3.0], &[2, 2]),
            true,
        );
        let target_idx = Variable::new(
            Tensor::from_i64(&[0, 1], &[2], crate::tensor::test_device()).unwrap(),
            false,
        );
        let loss_idx = cross_entropy_loss(&pred, &target_idx).unwrap();

        // Compare with one-hot version
        let pred2 = Variable::new(
            from_f32(&[2.0, 1.0, 1.0, 3.0], &[2, 2]),
            true,
        );
        let target_oh = Variable::new(
            from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            false,
        );
        let loss_oh = cross_entropy_loss(&pred2, &target_oh).unwrap();

        let v1 = loss_idx.item().unwrap();
        let v2 = loss_oh.item().unwrap();
        assert!(
            (v1 - v2).abs() < 1e-5,
            "index loss ({}) should match one-hot loss ({})", v1, v2
        );

        // Backward works
        loss_idx.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    #[test]
    fn test_cross_entropy_indices_converges() {
        let model = Linear::on_device(2, 3, crate::tensor::test_device()).unwrap();
        let params = model.parameters();
        let mut optim = Adam::new(&params, 0.05);

        // 3 samples, 3 classes: class 0=[1,0], class 1=[0,1], class 2=[0.5,0.5]
        let x = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[3, 2]), false);
        let target = Variable::new(
            Tensor::from_i64(&[0, 1, 2], &[3], crate::tensor::test_device()).unwrap(),
            false,
        );

        let mut last_loss = f64::MAX;
        for _ in 0..200 {
            optim.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = cross_entropy_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }
        assert!(last_loss < 0.5, "cross entropy with indices should converge, got {}", last_loss);
    }

    // --- BatchNorm2d ---

    #[test]
    fn test_batchnorm2d() {
        let bn = BatchNorm2d::on_device(3, crate::tensor::test_device()).unwrap();
        // [batch=4, channels=3, height=8, width=8]
        let x = Variable::new(
            Tensor::randn(&[4, 3, 8, 8], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![4, 3, 8, 8]);

        // Backward
        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(bn.parameters().len(), 2); // gamma, beta
    }

    #[test]
    fn test_batchnorm2d_eval() {
        let bn = BatchNorm2d::on_device(4, crate::tensor::test_device()).unwrap();

        // Run training to populate running stats
        for _ in 0..3 {
            let x = Variable::new(
                Tensor::randn(&[4, 4, 6, 6], crate::tensor::test_opts()).unwrap(),
                false,
            );
            bn.forward(&x).unwrap();
        }

        bn.set_training(false);
        let x = Variable::new(
            Tensor::randn(&[2, 4, 6, 6], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let out = bn.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![2, 4, 6, 6]);
    }

    // --- Conv1d tests ---

    #[test]
    fn test_conv1d() {
        let conv = Conv1d::build(1, 2, 3, true, 1, 0, 1, 1, crate::tensor::test_device()).unwrap();
        // Input: [batch=1, channels=1, length=10]
        let x = Variable::new(
            Tensor::randn(&[1, 1, 10], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        // With kernel=3, no padding: output = 10-3+1 = 8
        assert_eq!(out.shape(), vec![1, 2, 8]);

        // Backward
        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // weight + bias = 2 params
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_conv1d_no_bias() {
        let conv = Conv1d::build(3, 8, 3, false, 1, 0, 1, 1, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 3, 20], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![2, 8, 18]);
        assert_eq!(conv.parameters().len(), 1); // weight only
    }

    #[test]
    fn test_conv1d_with_padding() {
        let conv = Conv1d::build(1, 1, 3, true, 1, 1, 1, 1, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 10], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        // Same padding: output = input size
        assert_eq!(out.shape(), vec![1, 1, 10]);
    }

    #[test]
    fn test_conv1d_builder() {
        let conv = Conv1d::configure(3, 16, 5)
            .with_stride(2)
            .with_padding(2)
            .on_device(crate::tensor::test_device())
            .done()
            .unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 3, 100], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let out = conv.forward(&x).unwrap();
        // L_out = (100 + 2*2 - 1*(5-1) - 1) / 2 + 1 = (100+4-4-1)/2+1 = 99/2+1 = 50
        assert_eq!(out.shape(), vec![1, 16, 50]);
    }

    // --- ConvTranspose1d tests ---

    #[test]
    fn test_conv_transpose1d() {
        let conv = ConvTranspose1d::build(2, 1, 3, true, 1, 0, 0, 1, 1, crate::tensor::test_device()).unwrap();
        // Input: [batch=1, channels=2, length=5]
        let x = Variable::new(
            Tensor::randn(&[1, 2, 5], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = conv.forward(&x).unwrap();
        // With kernel=3, no padding: output = 5+3-1 = 7
        assert_eq!(out.shape(), vec![1, 1, 7]);

        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());
        assert_eq!(conv.parameters().len(), 2);
    }

    // --- GroupNorm tests ---

    #[test]
    fn test_groupnorm() {
        let gn = GroupNorm::on_device(4, 8, crate::tensor::test_device()).unwrap();
        // Input: [batch=2, channels=8, height=4, width=4]
        let x = Variable::new(
            Tensor::randn(&[2, 8, 4, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let out = gn.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![2, 8, 4, 4]);

        // Backward
        let loss = out.sum().unwrap();
        loss.backward().unwrap();
        assert!(x.grad().is_some());

        // weight + bias = 2 params
        assert_eq!(gn.parameters().len(), 2);
    }

    #[test]
    fn test_groupnorm_1d() {
        let gn = GroupNorm::on_device(2, 4, crate::tensor::test_device()).unwrap();
        // GroupNorm also works on [B, C, L] input
        let x = Variable::new(
            Tensor::randn(&[3, 4, 10], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let out = gn.forward(&x).unwrap();
        assert_eq!(out.shape(), vec![3, 4, 10]);
    }

    // --- Cosine similarity tests ---

    #[test]
    fn test_cosine_similarity() {
        let a = Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], crate::tensor::test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3], crate::tensor::test_device()).unwrap();
        let sim = a.cosine_similarity(&b, 1, 1e-8).unwrap();
        let data = sim.to_f32_vec().unwrap();
        // First pair: identical -> cos_sim = 1.0
        assert!((data[0] - 1.0).abs() < 1e-5);
        // Second pair: orthogonal -> cos_sim = 0.0
        assert!(data[1].abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_autograd() {
        let a = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            true,
        );
        let b = Variable::new(
            Tensor::from_f32(&[4.0, 5.0, 6.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let sim = a.cosine_similarity(&b, 1, 1e-8).unwrap();
        let loss = sim.sum().unwrap();
        loss.backward().unwrap();
        assert!(a.grad().is_some());
    }

    // --- BCE loss tests ---

    #[test]
    fn test_bce_loss() {
        // Probabilities (after sigmoid)
        let pred = Variable::new(
            Tensor::from_f32(&[0.8, 0.2, 0.9, 0.1], &[4], crate::tensor::test_device()).unwrap(),
            true,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 0.0, 1.0, 0.0], &[4], crate::tensor::test_device()).unwrap(),
            false,
        );
        let loss = bce_loss(&pred, &target).unwrap();
        let val = loss.data().item().unwrap();
        // BCE should be small for these good predictions
        assert!(val > 0.0 && val < 1.0, "bce_loss = {}", val);

        // Backward
        loss.backward().unwrap();
        assert!(pred.grad().is_some());
    }

    // --- Pad mode tests ---

    #[test]
    fn test_pad_mode_constant() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6], crate::tensor::test_device()).unwrap();
        // Constant pad with value 0 (same as regular pad)
        let padded = t.pad_mode(&[1, 1], 0, 0.0).unwrap();
        assert_eq!(padded.shape(), vec![1, 1, 8]);
    }

    #[test]
    fn test_pad_mode_reflect() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6], crate::tensor::test_device()).unwrap();
        let padded = t.pad_mode(&[2, 2], 1, 0.0).unwrap();
        assert_eq!(padded.shape(), vec![1, 1, 10]);
        let data = padded.to_f32_vec().unwrap();
        // Reflect padding: [3, 2, 1, 2, 3, 4, 5, 6, 5, 4]
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_pad_mode_replicate() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], crate::tensor::test_device()).unwrap();
        let padded = t.pad_mode(&[1, 1], 2, 0.0).unwrap();
        assert_eq!(padded.shape(), vec![1, 1, 6]);
        let data = padded.to_f32_vec().unwrap();
        // Replicate: first value replicated left, last value replicated right
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[5] - 4.0).abs() < 1e-5);
    }

    // --- Interpolate tests ---

    #[test]
    fn test_interpolate_nearest() {
        let t = Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap();
        let up = t.interpolate(&[8, 8], 0, false).unwrap();
        assert_eq!(up.shape(), vec![1, 1, 8, 8]);
    }

    #[test]
    fn test_interpolate_bilinear() {
        let t = Tensor::randn(&[1, 3, 4, 4], crate::tensor::test_opts()).unwrap();
        let up = t.interpolate(&[8, 8], 1, false).unwrap();
        assert_eq!(up.shape(), vec![1, 3, 8, 8]);
    }

    #[test]
    fn test_interpolate_bicubic() {
        let t = Tensor::randn(&[1, 3, 8, 8], crate::tensor::test_opts()).unwrap();
        let down = t.interpolate(&[4, 4], 2, false).unwrap();
        assert_eq!(down.shape(), vec![1, 3, 4, 4]);
    }

    // --- Comparison ops with integer input ---

    #[test]
    fn test_eq_tensor_int64() {
        let a = Tensor::from_i64(&[1, 2, 3], &[3], crate::tensor::test_device()).unwrap();
        let b = Tensor::from_i64(&[1, 5, 3], &[3], crate::tensor::test_device()).unwrap();
        let eq = a.eq_tensor(&b).unwrap();
        // Should be Float32 (not Int64) so mean() works
        assert_eq!(eq.dtype(), crate::tensor::DType::Float32);
        let data = eq.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 1.0]);
        // mean should work without to_dtype
        let m = eq.mean().unwrap().item().unwrap();
        assert!((m - 2.0 / 3.0).abs() < 1e-5);
    }
}
