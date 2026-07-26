use std::collections::HashMap;
use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Linear, Module, NamedInputModule};
use crate::tensor::{Device, Result, Tensor};

/// Softmax-normalized gate router for Mixture-of-Experts.
///
/// Projects input to `num_experts` logits, applies softmax so weights
/// sum to 1. When receiving multiple inputs (via Gate.Using), they are
/// summed element-wise before projection.
///
/// ```ignore
/// FlowBuilder::from(embed)
///     .tag("ctx")
///     .through(layer)
///     .gate(SoftmaxRouter::new(hidden, 3)?, modules![a, b, c])
///     .build()
/// ```
pub struct SoftmaxRouter {
    proj: Rc<Linear>,
}

impl SoftmaxRouter {
    /// Create a softmax router projecting `input_dim` to `num_experts` on CPU.
    pub fn new(input_dim: i64, num_experts: i64) -> Result<Self> {
        Self::on_device(input_dim, num_experts, Device::CPU)
    }

    /// Create a softmax router on the specified device.
    pub fn on_device(input_dim: i64, num_experts: i64, device: Device) -> Result<Self> {
        Ok(SoftmaxRouter {
            proj: Rc::new(Linear::on_device(input_dim, num_experts, device)?),
        })
    }
}

impl Module for SoftmaxRouter {
    fn name(&self) -> &str { "softmax_router" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let out = self.proj.forward(input)?;
        let dim = out.data().ndim() as i32 - 1;
        out.softmax(dim)
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for SoftmaxRouter {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        self.forward(&super::node::sum_named_refs(input, refs)?)
    }
}

/// Independent sigmoid gate router.
///
/// Projects input to `num_experts` logits, applies sigmoid independently.
/// Weights do NOT sum to 1 — each expert is gated between 0 and 1.
pub struct SigmoidRouter {
    proj: Rc<Linear>,
}

impl SigmoidRouter {
    /// Create a sigmoid router projecting `input_dim` to `num_experts` on CPU.
    pub fn new(input_dim: i64, num_experts: i64) -> Result<Self> {
        Self::on_device(input_dim, num_experts, Device::CPU)
    }

    /// Create a sigmoid router on the specified device.
    pub fn on_device(input_dim: i64, num_experts: i64, device: Device) -> Result<Self> {
        Ok(SigmoidRouter {
            proj: Rc::new(Linear::on_device(input_dim, num_experts, device)?),
        })
    }
}

impl Module for SigmoidRouter {
    fn name(&self) -> &str { "sigmoid_router" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.proj.forward(input)?.sigmoid()
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for SigmoidRouter {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        self.forward(&super::node::sum_named_refs(input, refs)?)
    }
}

/// Fixed branch selector for Switch — always picks the same branch.
///
/// Useful for testing, ablation studies, or static configurations.
pub struct FixedSelector {
    index: f32,
}

impl FixedSelector {
    /// Create a selector that always routes to `index` (0-based branch).
    pub fn new(index: usize) -> Self {
        FixedSelector { index: index as f32 }
    }
}

impl Module for FixedSelector {
    fn name(&self) -> &str { "fixed_selector" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(Variable::new(
            Tensor::from_f32(&[self.index], &[1], input.device())?,
            false,
        ))
    }
}

/// Learnable branch selector for Switch via argmax.
///
/// Projects input to `num_branches` logits and selects the highest one
/// **per sample**: a `[Batch, features]` stream yields one branch index per
/// row, so each sample is routed to its own branch and only the branches that
/// received rows run. An unbatched `[features]` stream yields a single index.
/// Selection is non-differentiable — gradients flow through whichever branch
/// each sample was routed to. The projection parameters are included in
/// Parameters() for policy-gradient training.
///
/// Routing is per row of dim 0. A stream with more than one leading dim
/// (e.g. `[Batch, Time, features]`) yields one index per (row, step), which
/// Switch rejects with a loud error rather than misroute — flatten the
/// leading dims into rows first if per-step routing is what you want.
pub struct ArgmaxSelector {
    proj: Rc<Linear>,
}

impl ArgmaxSelector {
    /// Create an argmax selector projecting `input_dim` to `num_branches` on CPU.
    pub fn new(input_dim: i64, num_branches: i64) -> Result<Self> {
        Self::on_device(input_dim, num_branches, Device::CPU)
    }

    /// Create an argmax selector on the specified device.
    pub fn on_device(input_dim: i64, num_branches: i64, device: Device) -> Result<Self> {
        Ok(ArgmaxSelector {
            proj: Rc::new(Linear::on_device(input_dim, num_branches, device)?),
        })
    }
}

impl Module for ArgmaxSelector {
    fn name(&self) -> &str { "argmax_selector" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let logits = self.proj.forward(input)?;
        let data = logits.data();
        // Reduce over the branch-logit dim only: [B, n] -> [B] (one index per
        // sample), [n] -> scalar. Flattening first would make the index depend
        // on batch size instead of naming a branch.
        let dim = data.ndim() as i32 - 1;
        Ok(Variable::new(data.argmax(dim, false)?, false))
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}

impl NamedInputModule for ArgmaxSelector {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        self.forward(&super::node::sum_named_refs(input, refs)?)
    }
}
