use std::rc::Rc;

use crate::autograd::Variable;
use crate::nn::{Linear, Module};
use crate::tensor::{Device, Result, Tensor};

/// Threshold-based halt condition for Loop.While / Loop.Until.
///
/// Signals halt (positive output) when the maximum element of the state
/// exceeds the threshold.
///
/// ```ignore
/// FlowBuilder::from(body)
///     .loop_body(body).until_cond(ThresholdHalt::new(50.0), 20)
/// ```
pub struct ThresholdHalt {
    threshold: f32,
}

impl ThresholdHalt {
    /// Create a halt condition that triggers when max(state) > `threshold`.
    pub fn new(threshold: f32) -> Self {
        ThresholdHalt { threshold }
    }
}

impl Module for ThresholdHalt {
    fn name(&self) -> &str { "threshold_halt" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let data = input.data().to_f32_vec()?;
        let max_val = data
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let val = max_val - self.threshold; // positive when exceeded → halt
        Ok(Variable::new(
            Tensor::from_f32(&[val], &[1], input.device())?,
            false,
        ))
    }
}

/// Learnable halt condition (Adaptive Computation Time / ACT pattern).
///
/// A linear probe projects the state to a scalar — iteration stops when
/// the output is positive. Fully differentiable.
///
/// The decision is **batch-level**: a loop advances one state tensor for every
/// sample together, so a batched state's per-sample probe outputs are averaged
/// into the single scalar the loop needs. Per-sample halting (true ACT, where
/// each sample stops on its own iteration) would need masked state updates and
/// is not what this construct does.
///
/// ```ignore
/// FlowBuilder::from(body)
///     .loop_body(body).until_cond(LearnedHalt::new(hidden_dim)?, 20)
/// ```
pub struct LearnedHalt {
    proj: Rc<Linear>,
}

impl LearnedHalt {
    /// Create a learned halt probe projecting `input_dim` to a scalar on CPU.
    pub fn new(input_dim: i64) -> Result<Self> {
        Self::on_device(input_dim, Device::CPU)
    }

    /// Create a learned halt probe on the specified device.
    pub fn on_device(input_dim: i64, device: Device) -> Result<Self> {
        Ok(LearnedHalt {
            proj: Rc::new(Linear::on_device(input_dim, 1, device)?),
        })
    }
}

impl Module for LearnedHalt {
    fn name(&self) -> &str { "learned_halt" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let probe = self.proj.forward(input)?;
        if probe.data().numel() == 1 {
            return Ok(probe);
        }
        // Batched state: pool the per-sample probes rather than let row 0
        // decide the iteration count for the whole batch. mean() keeps every
        // sample in the gradient path.
        probe.mean()
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.proj.clone()]
    }
}
