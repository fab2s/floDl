//! Loop tests: `loop_for` / `loop_while` / `loop_until` / `loop_in_chain` /
//! `loop_using_backward`, plus loop trace + loop-body emit tests.

use super::*;

#[test]
fn test_loop_for() {
    // Doubler × 3 iterations: [1, 2] → [8, 16]
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Doubler)
        .for_n(3)
        .build()
        .unwrap();

    // Set linear to identity
    let params = graph.parameters();
    params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert!((data[0] - 8.0).abs() < 1e-5, "1*2^3=8, got {}", data[0]);
    assert!((data[1] - 16.0).abs() < 1e-5, "2*2^3=16, got {}", data[1]);
}

#[test]
fn test_loop_for_backward() {
    // Loop with a learnable bias — gradient should accumulate across iterations
    let bias_step = BiasStep::new(2).unwrap();
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(bias_step)
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    // All parameters should have gradients
    for p in graph.parameters() {
        assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
    }

    // The bias gradient should be 3 (accumulated from 3 iterations)
    // dL/db = 1 per iteration, 3 iterations → grad = [3, 3]
    // (because sum reduces to scalar, dL/d_each_element = 1, and bias contributes at each step)
    let all_params = graph.parameters();
    // Find the loop_bias parameter (from BiasStep, not Linear's "bias")
    let bias_param = all_params.iter().find(|p| p.name == "loop_bias").unwrap();
    let grad = bias_param.variable.grad().unwrap().to_f32_vec().unwrap();
    assert!(
        (grad[0] - 3.0).abs() < 1e-5,
        "bias grad should be 3, got {}",
        grad[0]
    );
}

#[test]
fn test_loop_while() {
    // While max < 10: double. Input [1, 2] → double until max >= 10
    // Iter 0: check [1,2] max=2 < 10 → double → [2,4]
    // Iter 1: check [2,4] max=4 < 10 → double → [4,8]
    // Iter 2: check [4,8] max=8 < 10 → double → [8,16]
    // Iter 3: check [8,16] max=16 >= 10 → halt
    // Result: [8, 16]
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Doubler)
        .while_cond(ThresholdHalt::new(10.0), 20)
        .build()
        .unwrap();

    let params = graph.parameters();
    params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
    assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
}

#[test]
fn test_loop_while_immediate_halt() {
    // Threshold 0.5 — input [1, 2] max=2 > 0.5, halt immediately
    // While checks before body, so body never runs
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Doubler)
        .while_cond(ThresholdHalt::new(0.5), 20)
        .build()
        .unwrap();

    let params = graph.parameters();
    params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    // Body never ran — output = input
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_loop_until() {
    // Until max > 10: double. Body runs at least once.
    // Input [1, 2]
    // Iter 0: double → [2, 4], check max=4 <= 10 → continue
    // Iter 1: double → [4, 8], check max=8 <= 10 → continue
    // Iter 2: double → [8, 16], check max=16 > 10 → halt
    // Result: [8, 16]
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Doubler)
        .until_cond(ThresholdHalt::new(10.0), 20)
        .build()
        .unwrap();

    let params = graph.parameters();
    params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
    assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
}

#[test]
fn test_loop_until_at_least_once() {
    // Until with threshold 0.5 — input [1, 2] would halt immediately in While,
    // but Until always runs body at least once
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Doubler)
        .until_cond(ThresholdHalt::new(0.5), 20)
        .build()
        .unwrap();

    let params = graph.parameters();
    params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
    params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    // Body ran once: [2, 4]
    assert!((data[0] - 2.0).abs() < 1e-5, "got {}", data[0]);
    assert!((data[1] - 4.0).abs() < 1e-5, "got {}", data[1]);
}

#[test]
fn test_loop_parameters() {
    // Loop with learnable body — parameters should include body params
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .for_n(3)
        .build()
        .unwrap();

    let params = graph.parameters();
    // From module: weight + bias = 2, loop body Linear: weight + bias = 2
    assert_eq!(params.len(), 4);
}

#[test]
fn test_loop_while_parameters() {
    // While loop with body + condition — both contribute parameters
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .while_cond(Linear::on_device(2, 1, crate::tensor::test_device()).unwrap(), 10)
        .build()
        .unwrap();

    let params = graph.parameters();
    // From module: 2, loop body: 2, condition: 2 = 6
    assert_eq!(params.len(), 6);
}

#[test]
fn test_loop_in_chain() {
    // Linear → Loop(ReLU) × 3 → Linear
    let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .loop_body(ReLU::new())
        .for_n(3)
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
}

#[test]
fn test_loop_using_backward_ref() {
    // Tag a tensor, then use it inside a loop body via .using()
    // Graph: identity → tag("ctx") → loop_body(AddRefModule).for_n(3).using("ctx")
    // Each iteration: state = state + ctx
    // So after 3 iterations: state = x + 3*x = 4*x
    let graph = FlowBuilder::from(Identity)
        .tag("ctx")
        .loop_body(AddRefModule)
        .for_n(3)
        .using(&["ctx"])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[2.0, 3.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    // x = [2, 3], after 3 iterations of (state + ctx): [8, 12]
    assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
    assert!((data[1] - 12.0).abs() < 1e-5, "got {}", data[1]);
}

#[test]
fn test_loop_using_backward_gradients() {
    // Ensure gradients flow through loop+using
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .tag("ctx")
        .loop_body(AddRefModule)
        .for_n(2)
        .using(&["ctx"])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some(), "input should have gradient");
    for p in graph.parameters() {
        assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
    }
}

// --- Forward reference tests ---

/// Nil-safe add: skips nil inputs, adds rest. For forward ref state accumulation.

#[test]
fn test_loop_traces() {
    // Loop(TracingDoubler) × 3: [1,2] → [2,4] → [4,8] → [8,16]
    // traces should capture [2,4], [4,8], [8,16]
    let graph = FlowBuilder::from(Identity)
        .loop_body(TracingDoubler::new())
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    assert!((data[0] - 8.0).abs() < 1e-5);

    // Get traces — should find them on the loop node
    let traces = graph.traces("any").unwrap();
    assert_eq!(traces.len(), 3, "3 iterations = 3 traces");

    let t0 = traces[0].data().to_f32_vec().unwrap();
    assert!((t0[0] - 2.0).abs() < 1e-5, "iter0: [2,4], got {}", t0[0]);

    let t1 = traces[1].data().to_f32_vec().unwrap();
    assert!((t1[0] - 4.0).abs() < 1e-5, "iter1: [4,8], got {}", t1[0]);

    let t2 = traces[2].data().to_f32_vec().unwrap();
    assert!((t2[0] - 8.0).abs() < 1e-5, "iter2: [8,16], got {}", t2[0]);
}

#[test]
fn test_loop_traces_cleared_each_forward() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(TracingDoubler::new())
        .for_n(2)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    graph.forward(&x).unwrap();
    let traces1 = graph.traces("any").unwrap();
    assert_eq!(traces1.len(), 2);

    // Second forward should clear and re-populate
    graph.forward(&x).unwrap();
    let traces2 = graph.traces("any").unwrap();
    assert_eq!(traces2.len(), 2);
}

#[test]
fn test_loop_no_traces_without_trace_impl() {
    // Doubler doesn't implement trace() (returns None by default)
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    graph.forward(&x).unwrap();

    // No traces since Doubler's trace() returns None
    assert!(graph.traces("any").is_none());
}

// --- LoopBody / TraceEmit tests ---

/// LoopBody that publishes two named per-iteration traces.
/// Returns 2*x as the next state, emits "double" = 2*x and "quad" = 4*x.
struct EmittingDoubler;
impl Module for EmittingDoubler {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        forward_via_step(self, input)
    }
    fn as_loop_body(&self) -> Option<&dyn LoopBody> { Some(self) }
}
impl LoopBody for EmittingDoubler {
    fn step(
        &self,
        input: &Variable,
        _refs: &HashMap<String, Variable>,
        emit: &mut TraceEmit<'_>,
    ) -> Result<Variable> {
        let two_x = input.add(input)?;
        let four_x = two_x.add(&two_x)?;
        emit.publish("double", two_x.clone());
        emit.publish("quad", four_x);
        Ok(two_x)
    }
}

/// LoopBody that emits "always" each iter and "odd_only" on iters 0, 2.
/// Used to verify sparse emits — vec length matches publish count, not n_iter.
struct SparseEmitter {
    step_count: RefCell<usize>,
}
impl SparseEmitter {
    fn new() -> Self { SparseEmitter { step_count: RefCell::new(0) } }
}
impl Module for SparseEmitter {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        forward_via_step(self, input)
    }
    fn as_loop_body(&self) -> Option<&dyn LoopBody> { Some(self) }
    fn reset(&self) { *self.step_count.borrow_mut() = 0; }
}
impl LoopBody for SparseEmitter {
    fn step(
        &self,
        input: &Variable,
        _refs: &HashMap<String, Variable>,
        emit: &mut TraceEmit<'_>,
    ) -> Result<Variable> {
        let i = *self.step_count.borrow();
        *self.step_count.borrow_mut() += 1;
        let out = input.add(input)?;
        emit.publish("always", out.clone());
        if i % 2 == 0 {
            emit.publish("even_only", out.clone());
        }
        Ok(out)
    }
}

/// LoopBody that publishes the same name twice per step — must panic.
struct DupEmitter;
impl Module for DupEmitter {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        forward_via_step(self, input)
    }
    fn as_loop_body(&self) -> Option<&dyn LoopBody> { Some(self) }
}
impl LoopBody for DupEmitter {
    fn step(
        &self,
        input: &Variable,
        _refs: &HashMap<String, Variable>,
        emit: &mut TraceEmit<'_>,
    ) -> Result<Variable> {
        let two_x = input.add(input)?;
        emit.publish("dup", two_x.clone());
        emit.publish("dup", two_x.clone());
        Ok(two_x)
    }
}

#[test]
fn test_loop_body_emits_two_named_traces() {
    // Loop(EmittingDoubler) × 3: x=[1,2] → 2x=[2,4] → 4x=[4,8] → 8x=[8,16]
    // Each iter emits "double" = 2*current and "quad" = 4*current.
    let graph = FlowBuilder::from(Identity)
        .loop_body(EmittingDoubler)
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    assert!((data[0] - 8.0).abs() < 1e-5, "final 8x = [8,16], got {}", data[0]);

    let doubles = graph.traces("double").expect("double stream");
    assert_eq!(doubles.len(), 3, "3 iterations = 3 emits of 'double'");

    let quads = graph.traces("quad").expect("quad stream");
    assert_eq!(quads.len(), 3, "3 iterations = 3 emits of 'quad'");

    // double[i] = 2 * input_i, where input_0 = [1,2], input_1 = [2,4], input_2 = [4,8]
    let d0 = doubles[0].data().to_f32_vec().unwrap();
    assert!((d0[0] - 2.0).abs() < 1e-5);
    let q0 = quads[0].data().to_f32_vec().unwrap();
    assert!((q0[0] - 4.0).abs() < 1e-5);

    let d2 = doubles[2].data().to_f32_vec().unwrap();
    assert!((d2[0] - 8.0).abs() < 1e-5);
    let q2 = quads[2].data().to_f32_vec().unwrap();
    assert!((q2[0] - 16.0).abs() < 1e-5);

    // traces_named (named-only lookup) returns the same data
    assert_eq!(graph.traces_named("double").unwrap().len(), 3);
    assert_eq!(graph.traces_named("quad").unwrap().len(), 3);
    assert!(graph.traces_named("nonexistent").is_none());
}

#[test]
fn test_loop_body_emit_cleared_each_forward() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(EmittingDoubler)
        .for_n(2)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    graph.forward(&x).unwrap();
    assert_eq!(graph.traces("double").unwrap().len(), 2);

    // Second forward should clear and re-populate — not append
    graph.forward(&x).unwrap();
    assert_eq!(graph.traces("double").unwrap().len(), 2);
    assert_eq!(graph.traces("quad").unwrap().len(), 2);
}

#[test]
fn test_loop_body_emit_sparse() {
    // 4 iters: "always" emits all 4 times, "even_only" emits on i=0,2 (2 times)
    let graph = FlowBuilder::from(Identity)
        .loop_body(SparseEmitter::new())
        .for_n(4)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    graph.forward(&x).unwrap();

    assert_eq!(graph.traces("always").unwrap().len(), 4);
    assert_eq!(graph.traces("even_only").unwrap().len(), 2);
}

#[should_panic(expected = "already published this step")]
#[test]
fn test_loop_body_emit_dup_panics() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(DupEmitter)
        .for_n(1)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    // Must panic on the second emit.publish("dup", ...) within the same step.
    let _ = graph.forward(&x);
}

// --- Router tests ---


// --- Batched loop control ---
//
// Loop tests ran almost entirely at batch size 1, the same blind spot that hid
// issue #32. A loop advances one state tensor for every row together, so what
// these pin is that rows stay independent through the iterations and that the
// gradient reaches all of them.

#[test]
fn test_loop_for_batched() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 2]);

    let d = y.data().to_f32_vec().unwrap();
    for (i, base) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
        let want = base * 8.0; // doubled three times
        assert!((d[i] - want).abs() < 1e-5, "elem {i}: want {want}, got {}", d[i]);
    }
}

#[test]
fn test_loop_for_backward_batched() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    graph.forward(&x).unwrap().sum().unwrap().backward().unwrap();

    let g = x.grad().expect("input must receive gradient").to_f32_vec().unwrap();
    for (i, v) in g.iter().enumerate() {
        assert!((v - 8.0).abs() < 1e-5, "elem {i} grad: want 8, got {v}");
    }
}

#[test]
fn test_loop_traces_batched_keep_full_batch() {
    // Regression guard, not a bug hunt: traces hold one Variable per iteration,
    // and a future memory optimization that narrowed or reduced them (keeping
    // one row, or a summary) would still satisfy every batch-1 trace test. The
    // shape assertion is what pins the whole batch; the row-distinct values pin
    // that iterations do not mix rows.
    let graph = FlowBuilder::from(Identity)
        .loop_body(TracingDoubler::new())
        .for_n(3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 10.0, 20.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 2]);

    let traces = graph.traces("any").unwrap();
    assert_eq!(traces.len(), 3, "3 iterations = 3 traces");

    // Doubling row-wise each iteration: 2x, 4x, 8x of the original rows.
    for (iter, trace) in traces.iter().enumerate() {
        assert_eq!(
            trace.shape(),
            vec![2, 2],
            "trace {iter} must keep the whole batch"
        );
        let factor = 2.0f32.powi(iter as i32 + 1);
        let d = trace.data().to_f32_vec().unwrap();
        for (i, base) in [1.0f32, 2.0, 10.0, 20.0].iter().enumerate() {
            let want = base * factor;
            assert!(
                (d[i] - want).abs() < 1e-4,
                "trace {iter} elem {i}: want {want}, got {}",
                d[i]
            );
        }
    }
}
