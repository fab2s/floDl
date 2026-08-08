//! Top-level tests for [`super`] (the `graph` module).
//!
//! Hosts the test helpers (Doubler, BiasStep, etc.) plus basic graph
//! mechanics tests (single module, chains, also/fork/split-merge,
//! parameters, training_loop, build errors, using-by-tag). Each topic
//! area lives in its own sibling under `tests/`; the helpers here are
//! `pub(super)` so child modules can pull them via `use super::*;`.

pub(crate) use super::*;
pub(crate) use crate::autograd::Variable;
pub(crate) use crate::graph::{
    ArgmaxSelector, FixedSelector, FlowBuilder, LearnedHalt, MergeOp, Reduce, SigmoidRouter,
    SoftmaxRouter, ThresholdHalt,
};
pub(crate) use crate::nn::{
    Identity, Linear, LoopBody, NamedInputModule, Optimizer, ReLU, SGD, Sigmoid, TraceEmit,
    forward_via_step, mse_loss,
};
pub(crate) use crate::tensor::Tensor;
pub(crate) use std::collections::HashMap;

mod flow_and_routing;
mod loops;
mod map_and_inputs;
mod misc;
mod observation;

pub(super) fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
    Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
}

// --- Helper modules for testing ---

/// Doubles the input: forward(x) = 2*x
pub(super) struct Doubler;
impl Module for Doubler {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.add(input)
    }
}

/// Adds a learnable bias at each step (for gradient accumulation testing).
pub(super) struct BiasStep {
    bias: Parameter,
}
impl BiasStep {
    fn new(size: i64) -> Result<Self> {
        let data = Tensor::zeros(&[size], crate::tensor::test_opts())?;
        let var = Variable::new(data, true);
        Ok(BiasStep {
            bias: Parameter {
                variable: var,
                name: "loop_bias".to_string(),
            },
        })
    }
}
impl Module for BiasStep {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.add(&self.bias.variable)
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.bias.clone()]
    }
}

/// Module that adds a tagged ref to the stream (for Using tests).
pub(super) struct AddRefModule;
impl Module for AddRefModule {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}
impl NamedInputModule for AddRefModule {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        if let Some(ctx) = refs.get("ctx") {
            input.add(ctx)
        } else {
            Ok(input.clone())
        }
    }
}

// --- Core graph tests (from before) ---

#[test]
fn test_single_module() {
    let l = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
    let graph = FlowBuilder::from(l).build().unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
}

#[test]
fn test_linear_chain() {
    let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
}

#[test]
fn test_also_residual() {
    let l1 = Linear::on_device(3, 3, crate::tensor::test_device()).unwrap();
    l1.weight.variable.set_data(from_f32(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
    ));
    l1.bias
        .as_ref()
        .unwrap()
        .variable
        .set_data(from_f32(&[0.0, 0.0, 0.0], &[3]));

    let l2 = Linear::on_device(3, 3, crate::tensor::test_device()).unwrap();
    l2.weight.variable.set_data(from_f32(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
    ));
    l2.bias
        .as_ref()
        .unwrap()
        .variable
        .set_data(from_f32(&[1.0, 1.0, 1.0], &[3]));

    // l1(x) + l2(l1(x)) = x + (x + 1) = 2x + 1
    let graph = FlowBuilder::from(l1).also(l2).build().unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert!((data[0] - 3.0).abs() < 1e-5);
    assert!((data[1] - 5.0).abs() < 1e-5);
    assert!((data[2] - 7.0).abs() < 1e-5);
}

// --- Fork tests ---

#[test]
fn test_fork_basic() {
    // Fork runs a side module but main stream continues unchanged.
    // identity(x) → fork(linear) tagged "side" → through(ReLU)
    // Main stream: ReLU(identity(x)) = ReLU(x)
    // Side output: linear(x) accessible via tagged("side")
    let l = Linear::on_device(2, 3, crate::tensor::test_device()).unwrap();

    let graph = FlowBuilder::from(Identity)
        .fork(l)
        .tag("side")
        .through(ReLU::new())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, -2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();

    // Main stream went through ReLU(identity(x)) → shape [1, 2]
    assert_eq!(y.shape(), vec![1, 2]);
    let data = y.data().to_f32_vec().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 0.0).abs() < 1e-5); // ReLU(-2) = 0

    // Side output is linear(x) → shape [1, 3]
    let side = graph.tagged("side").unwrap();
    assert_eq!(side.shape(), vec![1, 3]);
}

#[test]
fn test_fork_multiple() {
    // Two forks from the same stream: letter_head and case_head pattern
    let head_a = Linear::on_device(4, 3, crate::tensor::test_device()).unwrap();
    let head_b = Linear::on_device(4, 2, crate::tensor::test_device()).unwrap();

    let graph = FlowBuilder::from(Linear::on_device(2, 4, crate::tensor::test_device()).unwrap())
        .tag("latent")
        .fork(head_a)
        .tag("head_a")
        .fork(head_b)
        .tag("head_b")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();

    // Main stream is still the linear(2→4) output
    assert_eq!(y.shape(), vec![1, 4]);

    // Both forks produced their outputs
    let a = graph.tagged("head_a").unwrap();
    assert_eq!(a.shape(), vec![1, 3]);
    let b = graph.tagged("head_b").unwrap();
    assert_eq!(b.shape(), vec![1, 2]);
}

#[test]
fn test_fork_backward() {
    // Gradients flow through both forks and the main stream
    let graph = FlowBuilder::from(Linear::on_device(2, 4, crate::tensor::test_device()).unwrap())
        .fork(Linear::on_device(4, 3, crate::tensor::test_device()).unwrap())
        .tag("side")
        .through(Linear::on_device(4, 1, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();

    // Loss from main stream + side output
    let side = graph.tagged("side").unwrap();
    let loss = y.sum().unwrap().add(&side.sum().unwrap()).unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some(), "input should have gradient");
    for p in graph.parameters() {
        assert!(
            p.variable.grad().is_some(),
            "{} should have gradient",
            p.name
        );
    }
}

// --- Split/Merge tests ---

#[test]
fn test_split_merge_add() {
    let graph = FlowBuilder::from(Linear::on_device(3, 3, crate::tensor::test_device()).unwrap())
        .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
        .merge(MergeOp::Add)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, -1.0, 2.0], &[1, 3]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 3]);
}

#[test]
fn test_split_merge_mean() {
    let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
    l.weight
        .variable
        .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    l.bias
        .as_ref()
        .unwrap()
        .variable
        .set_data(from_f32(&[0.0, 0.0], &[2]));

    let b1 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
    b1.weight
        .variable
        .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    b1.bias
        .as_ref()
        .unwrap()
        .variable
        .set_data(from_f32(&[0.0, 0.0], &[2]));
    let b2 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
    b2.weight
        .variable
        .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    b2.bias
        .as_ref()
        .unwrap()
        .variable
        .set_data(from_f32(&[0.0, 0.0], &[2]));

    let graph = FlowBuilder::from(l)
        .split(vec![Box::new(b1), Box::new(b2)])
        .merge(MergeOp::Mean)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert!((data[0] - 3.0).abs() < 1e-5);
    assert!((data[1] - 7.0).abs() < 1e-5);
}

#[test]
fn test_parameters() {
    let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let params = graph.parameters();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_graph_backward() {
    let l1 = Linear::on_device(3, 2, crate::tensor::test_device()).unwrap();
    let l2 = Linear::on_device(2, 1, crate::tensor::test_device()).unwrap();

    let graph = FlowBuilder::from(l1)
        .through(ReLU::new())
        .through(l2)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    for p in graph.parameters() {
        assert!(
            p.variable.grad().is_some(),
            "{} should have gradient",
            p.name
        );
    }
    assert!(x.grad().is_some());
}

#[test]
fn test_graph_as_module() {
    let inner = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .through(ReLU::new())
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let y = outer.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
    assert_eq!(outer.parameters().len(), 4);
}

#[test]
fn test_training_loop() {
    let graph = FlowBuilder::from(Linear::on_device(1, 1, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let params = graph.parameters();
    let mut optim = SGD::new(&params, 0.01, 0.0);

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]), false);
    let target = Variable::new(from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]), false);

    let mut last_loss = f64::MAX;
    for _ in 0..800 {
        optim.zero_grad();
        let pred = graph.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        last_loss = loss.item().unwrap();
        loss.backward().unwrap();
        optim.step().unwrap();
    }

    assert!(last_loss < 0.01, "got loss={}", last_loss);
}

#[test]
fn test_also_backward() {
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .also(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(
            p.variable.grad().is_some(),
            "{} should have gradient",
            p.name
        );
    }
}

#[test]
fn test_split_merge_backward() {
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .split(vec![
            Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
        ])
        .merge(MergeOp::Add)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(
            p.variable.grad().is_some(),
            "{} should have gradient",
            p.name
        );
    }
}

#[test]
fn test_build_error_open_streams() {
    let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
        .build();
    assert!(result.is_err());
}

#[test]
fn test_build_error_duplicate_tag() {
    let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .tag("features")
        .through(ReLU::new())
        .tag("features")
        .build();
    assert!(result.is_err());
}

// --- Using tests ---

#[test]
fn test_using_backward_ref() {
    // Tag a point, then use it downstream
    // Graph: linear(x) → tag("ctx") → through(AddRef).using("ctx")
    // AddRef adds ctx to stream: stream + ctx = 2 * linear(x)
    let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
    l.weight
        .variable
        .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    l.bias
        .as_ref()
        .unwrap()
        .variable
        .set_data(from_f32(&[0.0, 0.0], &[2]));

    let graph = FlowBuilder::from(l)
        .tag("ctx")
        .through(AddRefModule)
        .using(&["ctx"])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[3.0, 5.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    // identity(x) = [3, 5], then AddRef adds ctx ([3, 5]) = [6, 10]
    assert!((data[0] - 6.0).abs() < 1e-5);
    assert!((data[1] - 10.0).abs() < 1e-5);
}

#[test]
fn test_using_backward_gradients() {
    let l = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
    let graph = FlowBuilder::from(l)
        .tag("ctx")
        .through(AddRefModule)
        .using(&["ctx"])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(
            p.variable.grad().is_some(),
            "{} should have gradient",
            p.name
        );
    }
}

#[test]
fn test_using_error_plain_module() {
    // Using on a plain module (not NamedInputModule) should error
    let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .tag("ctx")
        .through(ReLU::new())
        .using(&["ctx"])
        .build();
    assert!(result.is_err());
}

#[test]
fn test_using_error_unknown_tag() {
    let result = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .through(AddRefModule)
        .using(&["nonexistent"])
        .build();
    assert!(result.is_err());
}

// --- Loop tests ---

// ---------------------------------------------------------------------------
// Shared helpers (used by sibling test files)
// ---------------------------------------------------------------------------

// --- from observation.rs ---
pub(super) struct SumRefs;
impl Module for SumRefs {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}
impl NamedInputModule for SumRefs {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let mut result = input.clone();
        for v in refs.values() {
            result = result.add(v)?;
        }
        Ok(result)
    }
}

// --- from loops.rs ---
pub(super) struct NilSafeAdd;
impl Module for NilSafeAdd {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }
}
impl NamedInputModule for NilSafeAdd {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        if let Some(memory) = refs.get("memory") {
            input.add(memory)
        } else {
            Ok(input.clone())
        }
    }
}

// --- from map_and_inputs.rs ---
pub(super) struct ScalarSum;
impl Module for ScalarSum {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sum()
    }
}

// --- from map_and_inputs.rs ---
pub(super) struct LinearSched(f64);
impl crate::nn::Scheduler for LinearSched {
    fn lr(&self, step: usize) -> f64 {
        step as f64 * self.0
    }
}

/// Build a tiny Graph + optimizer + a fake gradient so step() can run end
/// to end on CPU. Keeps the test cheap (no CUDA needed).
pub(super) fn graph_with_optim(initial_lr: f64) -> (crate::graph::Graph, Variable) {
    use crate::nn::SGD;
    let dev = crate::tensor::test_device();
    let graph = FlowBuilder::from(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();
    graph.set_optimizer(|p| SGD::new(p, initial_lr, 0.0));
    // Run one forward+backward so .grad() is populated and step() can do work.
    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    (graph, x)
}

pub(super) fn current_optim_lr(graph: &crate::graph::Graph) -> f64 {
    graph.optimizer.borrow().as_ref().map(|o| o.lr()).unwrap()
}

// --- from flow_and_routing.rs ---
pub(super) struct Tripler;
impl Module for Tripler {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.add(&input.add(input)?)
    }
    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

// --- from misc.rs ---
pub(super) struct TracingDoubler {
    last_output: RefCell<Option<Variable>>,
}
impl TracingDoubler {
    fn new() -> Self {
        TracingDoubler {
            last_output: RefCell::new(None),
        }
    }
}
impl Module for TracingDoubler {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let out = input.add(input)?;
        *self.last_output.borrow_mut() = Some(out.clone());
        Ok(out)
    }
    fn trace(&self) -> Option<Variable> {
        self.last_output.borrow().clone()
    }
}

// --- Batched merge and ref wiring ---
//
// Both combine tensors, and at batch size 1 a row-wise combination and a
// broadcast of row 0 agree. Row-distinct inputs are what separate them.

#[test]
fn test_split_merge_mean_batched() {
    let graph = FlowBuilder::from(Identity)
        .split(vec![Box::new(Doubler), Box::new(Tripler)])
        .merge(MergeOp::Mean)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 10.0, 20.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 2]);

    let d = y.data().to_f32_vec().unwrap();
    for (i, base) in [1.0f32, 2.0, 10.0, 20.0].iter().enumerate() {
        let want = base * 2.5; // (2x + 3x) / 2, row by row
        assert!(
            (d[i] - want).abs() < 1e-4,
            "elem {i}: want {want}, got {}",
            d[i]
        );
    }
}

#[test]
fn test_using_ref_is_row_wise_batched() {
    let graph = FlowBuilder::from(Identity)
        .tag("ctx")
        .through(AddRefModule)
        .using(&["ctx"])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 10.0, 20.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();

    let d = y.data().to_f32_vec().unwrap();
    for (i, base) in [1.0f32, 2.0, 10.0, 20.0].iter().enumerate() {
        let want = base * 2.0; // stream + ctx, and ctx is the stream here
        assert!(
            (d[i] - want).abs() < 1e-5,
            "elem {i}: want {want}, got {}",
            d[i]
        );
    }
}

#[test]
fn test_split_merge_add_batched() {
    // MergeOp::Add is element-wise today, so this cannot fail against the
    // current implementation. It is here as a regression guard: `gate` in this
    // same module already vectorizes a combine via stack + broadcast + sum_dim,
    // and if the merge machinery ever gets that treatment, per-row correctness
    // stops being free. Mean was covered; Add is the other half of the path.
    let graph = FlowBuilder::from(Identity)
        .split(vec![Box::new(Doubler), Box::new(Tripler)])
        .merge(MergeOp::Add)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 10.0, 20.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 2]);

    let d = y.data().to_f32_vec().unwrap();
    for (i, base) in [1.0f32, 2.0, 10.0, 20.0].iter().enumerate() {
        let want = base * 5.0; // 2x + 3x, row by row
        assert!(
            (d[i] - want).abs() < 1e-4,
            "elem {i}: want {want}, got {}",
            d[i]
        );
    }
}

#[test]
fn test_fork_batched_main_and_side_are_row_wise() {
    // Fork hands the same batch to a side module while the main stream carries
    // on. Row-distinct values pin that neither path mixes rows, and that
    // advancing the main stream does not disturb the side output — `Tensor`'s
    // Clone is shallow, so aliasing is precisely the hazard a future in-place
    // optimization on either path would introduce.
    let graph = FlowBuilder::from(Identity)
        .fork(Tripler)
        .tag("side")
        .through(Doubler)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, -2.0, 10.0, -20.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 2]);

    let main = y.data().to_f32_vec().unwrap();
    for (i, base) in [1.0f32, -2.0, 10.0, -20.0].iter().enumerate() {
        let want = base * 2.0;
        assert!(
            (main[i] - want).abs() < 1e-5,
            "main elem {i}: want {want}, got {}",
            main[i]
        );
    }

    let side = graph.tagged("side").unwrap().data().to_f32_vec().unwrap();
    for (i, base) in [1.0f32, -2.0, 10.0, -20.0].iter().enumerate() {
        let want = base * 3.0;
        assert!(
            (side[i] - want).abs() < 1e-5,
            "side elem {i}: want {want}, got {}",
            side[i]
        );
    }
}
