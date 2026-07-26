//! FlowBuilder + forward_ref tests, switch/gate branching, routers
//! (Softmax/Sigmoid/Fixed/Argmax selectors), and halt conditions
//! (Threshold + Learned).

use super::*;

#[test]
fn test_flowbuilder_new() {
    // FlowBuilder::new() starts with implicit Identity
    let graph = FlowBuilder::new()
        .tag("input")
        .through(Linear::on_device(3, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
}

#[test]
fn test_deferred_error_carries_chain_position() {
    // A deferred builder error (here: `merge` on a single, unsplit stream)
    // must name where in the chain it happened, not just the bare guard
    // message. Without the position stamp, a failure among many chained
    // calls is unlocatable (GD15).
    let dev = crate::tensor::test_device();
    let result = FlowBuilder::from(Linear::on_device(4, 8, dev).unwrap())
        .through(Linear::on_device(8, 8, dev).unwrap())
        .merge(MergeOp::Add) // illegal: only one open stream
        .build();

    let msg = result.err().expect("expected build to fail").to_string();
    assert!(
        msg.contains("merge requires multiple streams"),
        "expected the guard message; got: {msg}"
    );
    // The stamp names the most recently built node, whose id encodes the
    // module type + its global sequence number (e.g. `Linear_2`).
    assert!(
        msg.contains("after node '"),
        "GD15: deferred error must carry chain position; got: {msg}"
    );
}

#[test]
fn test_unknown_port_name_errors_at_build() {
    // A port name that isn't among the node's declared ports must fail the
    // build loudly instead of silently routing to port 0 (which would train
    // on wrong data). FlowBuilder can't produce such an edge today, so this
    // drives Graph::build directly.
    use crate::graph::node::{DEFAULT_INPUT, DEFAULT_OUTPUT, Edge, ExposedPort, Node};
    use indexmap::IndexMap;
    use std::collections::HashSet;

    let mk_node = |id: &str| Node {
        id: id.to_string(),
        input_ports: vec![DEFAULT_INPUT.to_string()],
        output_ports: vec![DEFAULT_OUTPUT.to_string()],
        run: Box::new(|inputs| Ok(inputs.to_vec())),
        module: None,
        ref_forward: None,
        trace_buf: None,
        named_trace_buf: None,
        loop_ports: None,
    };
    let mut nodes = IndexMap::new();
    nodes.insert("a".to_string(), mk_node("a"));
    nodes.insert("b".to_string(), mk_node("b"));

    let result = Graph::build(
        nodes,
        vec![Edge {
            from_node: "a".into(),
            from_port: DEFAULT_OUTPUT.into(),
            to_node: "b".into(),
            to_port: "bogus".into(),
        }],
        vec![ExposedPort {
            name: "input".into(),
            node_id: "a".into(),
            port: DEFAULT_INPUT.into(),
        }],
        vec![ExposedPort {
            name: "output".into(),
            node_id: "b".into(),
            port: DEFAULT_OUTPUT.into(),
        }],
        HashMap::new(),
        Vec::new(),
        HashMap::new(),
        None,
        HashSet::new(),
        false,
    );
    let msg = match result {
        Ok(_) => panic!("build must reject the unknown port"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("bogus") && msg.contains("edge target") && msg.contains("\"b\""),
        "error must name the port, the resolution kind, and the node: {msg}"
    );
}

#[test]
fn test_forward_ref() {
    // Forward reference: using() before tag(). State carries between forward() calls.
    // Graph: entry → NilSafeAdd.Using("memory") → Identity.Tag("memory")
    // Pass 1: add gets [stream, zeros] (memory is nil/zeroed) → Identity → state captured
    // Pass 2: add gets [stream, prev_output] → sum → Identity → state captured
    let graph = FlowBuilder::from(Identity)
        .through(NilSafeAdd)
        .using(&["memory"])
        .through(Identity)
        .tag("memory")
        .build()
        .unwrap();

    assert!(graph.has_state());

    // Pass 1: [1,2] + zeros → [1,2]
    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y1 = graph.forward(&x).unwrap();
    let d1 = y1.data().to_f32_vec().unwrap();
    assert!((d1[0] - 1.0).abs() < 1e-5, "pass1[0]: got {}", d1[0]);
    assert!((d1[1] - 2.0).abs() < 1e-5, "pass1[1]: got {}", d1[1]);

    // Pass 2: [1,2] + [1,2] → [2,4]
    let y2 = graph.forward(&x).unwrap();
    let d2 = y2.data().to_f32_vec().unwrap();
    assert!((d2[0] - 2.0).abs() < 1e-5, "pass2[0]: got {}", d2[0]);
    assert!((d2[1] - 4.0).abs() < 1e-5, "pass2[1]: got {}", d2[1]);

    // Pass 3: [1,2] + [2,4] → [3,6]
    let y3 = graph.forward(&x).unwrap();
    let d3 = y3.data().to_f32_vec().unwrap();
    assert!((d3[0] - 3.0).abs() < 1e-5, "pass3[0]: got {}", d3[0]);
    assert!((d3[1] - 6.0).abs() < 1e-5, "pass3[1]: got {}", d3[1]);
}

#[test]
fn test_forward_ref_reset_state() {
    let graph = FlowBuilder::from(Identity)
        .through(NilSafeAdd)
        .using(&["memory"])
        .through(Identity)
        .tag("memory")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);

    // Build up state
    graph.forward(&x).unwrap();
    graph.forward(&x).unwrap();
    let y_before = graph.forward(&x).unwrap();
    let d_before = y_before.data().to_f32_vec().unwrap();
    assert!((d_before[0] - 3.0).abs() < 1e-5);

    // Reset and verify state is cleared
    graph.reset_state();
    let y_after = graph.forward(&x).unwrap();
    let d_after = y_after.data().to_f32_vec().unwrap();
    assert!((d_after[0] - 1.0).abs() < 1e-5, "after reset: got {}", d_after[0]);
}

#[test]
fn test_forward_ref_detach_state() {
    let graph = FlowBuilder::from(Identity)
        .through(NilSafeAdd)
        .using(&["memory"])
        .through(Identity)
        .tag("memory")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);

    // Run forward, accumulate state
    let y1 = graph.forward(&x).unwrap();
    let _ = y1.sum().unwrap();

    // Detach state — values preserved but gradient chain broken
    graph.detach_state();

    // State should still have values (not reset)
    let y2 = graph.forward(&x).unwrap();
    let d2 = y2.data().to_f32_vec().unwrap();
    assert!((d2[0] - 2.0).abs() < 1e-5, "detach preserves values: got {}", d2[0]);
}

#[test]
fn test_forward_ref_backward() {
    // Gradients should flow through forward-ref connections
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .through(NilSafeAdd)
        .using(&["memory"])
        .through(Identity)
        .tag("memory")
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

#[test]
fn test_forward_ref_unresolved_error() {
    // Using a tag that is never defined should error at build
    let result = FlowBuilder::from(Identity)
        .through(NilSafeAdd)
        .using(&["nonexistent"])
        .build();
    assert!(result.is_err());
}

#[test]
fn test_forward_ref_mixed_refs() {
    // Mix backward ref (tag before using) and forward ref (using before tag)
    // "ctx" is backward (AddRefModule expects "ctx"), "memory" is forward (NilSafeAdd expects "memory")
    let graph = FlowBuilder::from(Identity)
        .tag("ctx")
        .through(AddRefModule)
        .using(&["ctx"])
        .through(NilSafeAdd)
        .using(&["memory"])
        .through(Identity)
        .tag("memory")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);

    // Pass 1: entry=[1,2], AddRef adds ctx=[1,2] → [2,4], NilSafeAdd +zeros → [2,4]
    let y1 = graph.forward(&x).unwrap();
    let d1 = y1.data().to_f32_vec().unwrap();
    assert!((d1[0] - 2.0).abs() < 1e-5, "mixed pass1[0]: got {}", d1[0]);

    // Pass 2: entry=[1,2], AddRef adds ctx=[1,2] → [2,4], NilSafeAdd +[2,4] → [4,8]
    let y2 = graph.forward(&x).unwrap();
    let d2 = y2.data().to_f32_vec().unwrap();
    assert!((d2[0] - 4.0).abs() < 1e-5, "mixed pass2[0]: got {}", d2[0]);
}

// --- Switch tests ---

/// Triples input.

#[test]
fn test_switch_selects_branch() {
    // Branch 0: double, Branch 1: triple. Router selects branch 1.
    let graph = FlowBuilder::from(Identity)
        .switch(FixedSelector::new(1), vec![Box::new(Doubler), Box::new(Tripler)])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    assert!((data[0] - 3.0).abs() < 1e-5, "triple [1]=3, got {}", data[0]);
    assert!((data[1] - 6.0).abs() < 1e-5, "triple [2]=6, got {}", data[1]);
}

#[test]
fn test_switch_branch0() {
    let graph = FlowBuilder::from(Identity)
        .switch(FixedSelector::new(0), vec![Box::new(Doubler), Box::new(Tripler)])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    assert!((data[0] - 2.0).abs() < 1e-5, "double [1]=2, got {}", data[0]);
    assert!((data[1] - 4.0).abs() < 1e-5, "double [2]=4, got {}", data[1]);
}

#[test]
fn test_switch_backward() {
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .switch(FixedSelector::new(0), vec![
            Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
        ])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    // Only entry + selected branch params should have gradients
    // (router has no params, unselected branch wasn't executed)
}

#[test]
fn test_switch_parameters() {
    let graph = FlowBuilder::from(Identity)
        .switch(
            Linear::on_device(2, 1, crate::tensor::test_device()).unwrap(),
            vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ],
        )
        .build()
        .unwrap();

    let params = graph.parameters();
    // Router: 2, Branch0: 2, Branch1: 2 = 6
    assert_eq!(params.len(), 6);
}

// --- Gate tests ---

/// Router that outputs equal weights for all experts.
struct EqualRouter(usize);
impl Module for EqualRouter {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let batch = input.shape()[0];
        let w = 1.0 / self.0 as f32;
        let data = vec![w; batch as usize * self.0];
        Ok(Variable::new(
            Tensor::from_f32(&data, &[batch, self.0 as i64], crate::tensor::test_device())?,
            false,
        ))
    }
    fn parameters(&self) -> Vec<Parameter> { vec![] }
}

#[test]
fn test_gate_equal_weights() {
    // Equal weights: output = mean of expert outputs
    let graph = FlowBuilder::from(Identity)
        .gate(EqualRouter(2), vec![Box::new(Doubler), Box::new(Tripler)])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[2.0, 4.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // double=[4,8], triple=[6,12], mean = [5, 10]
    assert!((data[0] - 5.0).abs() < 1e-5, "gate[0]=5, got {}", data[0]);
    assert!((data[1] - 10.0).abs() < 1e-5, "gate[1]=10, got {}", data[1]);
}

#[test]
fn test_gate_backward() {
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .gate(
            Linear::on_device(2, 2, crate::tensor::test_device()).unwrap(),
            vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
    }
}

#[test]
fn test_gate_parameters() {
    let graph = FlowBuilder::from(Identity)
        .gate(
            Linear::on_device(2, 2, crate::tensor::test_device()).unwrap(),
            vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ],
        )
        .build()
        .unwrap();

    let params = graph.parameters();
    // Router: 2, Expert0: 2, Expert1: 2 = 6
    assert_eq!(params.len(), 6);
}

// --- Map tests ---


#[test]
fn test_softmax_router_gate() {
    // SoftmaxRouter with 2 experts: double + triple, weights from learned router
    let graph = FlowBuilder::from(Identity)
        .gate(
            SoftmaxRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
            vec![Box::new(Doubler), Box::new(Tripler)],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    // Output should be a weighted combination — just verify it runs and has correct shape
    assert_eq!(y.shape(), vec![1, 2]);
    // Router has 2 params (weight + bias), experts have 0
    let params = graph.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_softmax_router_backward() {
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .gate(
            SoftmaxRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
            vec![
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap()),
            ],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(p.variable.grad().is_some(), "{} missing gradient", p.name);
    }
}

#[test]
fn test_sigmoid_router_gate() {
    let graph = FlowBuilder::from(Identity)
        .gate(
            SigmoidRouter::on_device(2, 2, crate::tensor::test_device()).unwrap(),
            vec![Box::new(Doubler), Box::new(Tripler)],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
}

#[test]
fn test_fixed_selector_switch() {
    // FixedSelector(1) always picks branch 1 (Tripler)
    let graph = FlowBuilder::from(Identity)
        .switch(FixedSelector::new(1), vec![Box::new(Doubler), Box::new(Tripler)])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[2.0, 3.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    assert!((data[0] - 6.0).abs() < 1e-5, "triple 2=6, got {}", data[0]);
    assert!((data[1] - 9.0).abs() < 1e-5, "triple 3=9, got {}", data[1]);
}

#[test]
fn test_argmax_selector_switch() {
    let graph = FlowBuilder::from(Identity)
        .switch(
            ArgmaxSelector::on_device(2, 2, crate::tensor::test_device()).unwrap(),
            vec![Box::new(Doubler), Box::new(Tripler)],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    // Should select one branch — just verify it runs and has correct shape
    assert_eq!(y.shape(), vec![1, 2]);
    // ArgmaxSelector has params from its Linear projection
    assert_eq!(graph.parameters().len(), 2);
}

// --- Per-sample switch routing (issue #32) ---

/// Routes each row by the sign of its first feature: negative → branch 0,
/// non-negative → branch 1. Deterministic, so dispatch is assertable.
pub(super) struct SignSelector;
impl Module for SignSelector {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let rows = input.shape()[0];
        let data = input.data().to_f32_vec()?;
        let cols = data.len() / rows as usize;
        let idx: Vec<f32> = (0..rows as usize)
            .map(|r| if data[r * cols] < 0.0 { 0.0 } else { 1.0 })
            .collect();
        Ok(Variable::new(from_f32(&idx, &[rows]), false))
    }
    fn parameters(&self) -> Vec<Parameter> { vec![] }
}

/// Emits `count` indices regardless of the stream's row count.
struct BadCountSelector { count: i64 }
impl Module for BadCountSelector {
    fn forward(&self, _input: &Variable) -> Result<Variable> {
        let idx = vec![0.0f32; self.count as usize];
        Ok(Variable::new(from_f32(&idx, &[self.count]), false))
    }
    fn parameters(&self) -> Vec<Parameter> { vec![] }
}

/// Emits a per-row index that names a branch that does not exist.
struct OutOfRangeSelector;
impl Module for OutOfRangeSelector {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let rows = input.shape()[0];
        let mut idx = vec![0.0f32; rows as usize];
        idx[1] = 7.0; // row 1 asks for branch 7
        Ok(Variable::new(from_f32(&idx, &[rows]), false))
    }
    fn parameters(&self) -> Vec<Parameter> { vec![] }
}

#[test]
fn test_switch_per_sample_dispatch() {
    // Rows 0 and 2 are negative → Doubler; rows 1 and 3 → Tripler.
    let graph = FlowBuilder::from(Identity)
        .switch(SignSelector, vec![Box::new(Doubler), Box::new(Tripler)])
        .build()
        .unwrap();

    let x = Variable::new(
        from_f32(&[-1.0, -2.0, 1.0, 2.0, -3.0, -4.0, 3.0, 4.0], &[4, 2]),
        false,
    );
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![4, 2]);

    // Each row must carry its own branch's factor, in the original row order.
    let d = y.data().to_f32_vec().unwrap();
    let expect = [
        -2.0, -4.0, // row 0: doubled
        3.0, 6.0,   // row 1: tripled
        -6.0, -8.0, // row 2: doubled
        9.0, 12.0,  // row 3: tripled
    ];
    for (i, want) in expect.iter().enumerate() {
        assert!(
            (d[i] - want).abs() < 1e-5,
            "elem {i}: want {want}, got {}",
            d[i]
        );
    }
}

#[test]
fn test_switch_per_sample_gradient() {
    // Gradients must reach every row through whichever branch ran it.
    let graph = FlowBuilder::from(Identity)
        .switch(SignSelector, vec![Box::new(Doubler), Box::new(Tripler)])
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[-1.0, -2.0, 1.0, 2.0], &[2, 2]), true);
    let y = graph.forward(&x).unwrap();
    y.sum().unwrap().backward().unwrap();

    let g = x.grad().expect("input must receive gradient").to_f32_vec().unwrap();
    // Row 0 went through Doubler (d/dx = 2), row 1 through Tripler (d/dx = 3).
    assert!((g[0] - 2.0).abs() < 1e-5, "row 0 grad: got {}", g[0]);
    assert!((g[1] - 2.0).abs() < 1e-5, "row 0 grad: got {}", g[1]);
    assert!((g[2] - 3.0).abs() < 1e-5, "row 1 grad: got {}", g[2]);
    assert!((g[3] - 3.0).abs() < 1e-5, "row 1 grad: got {}", g[3]);
}

#[test]
fn test_argmax_selector_emits_one_index_per_sample() {
    // Regression for issue #32: the selector used to flatten [B, n] logits and
    // return a single flat argmax, so the "branch index" scaled with batch size.
    let sel = ArgmaxSelector::on_device(3, 2, crate::tensor::test_device()).unwrap();
    let x = Variable::new(
        from_f32(
            &[
                1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.5, 0.0, -0.5, 4.0, -4.0, 1.0, 2.0,
                2.0, 2.0, -1.0, 3.0, 0.0,
            ],
            &[6, 3],
        ),
        false,
    );

    let out = sel.forward(&x).unwrap();
    assert_eq!(out.data().numel(), 6, "one branch index per row, not one flat index");
    for v in out.data().to_f64_vec().unwrap() {
        assert!((0.0..2.0).contains(&v), "branch index {v} outside [0, 2)");
    }
}

#[test]
fn test_argmax_selector_switch_batched() {
    // The issue #32 reproduction, scaled down: batch >> branch count used to
    // yield an out-of-bounds branch index.
    let graph = FlowBuilder::from(Linear::on_device(16, 8, crate::tensor::test_device()).unwrap())
        .switch(
            ArgmaxSelector::on_device(8, 2, crate::tensor::test_device()).unwrap(),
            vec![
                Box::new(Linear::on_device(8, 8, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(8, 8, crate::tensor::test_device()).unwrap()),
            ],
        )
        .build()
        .unwrap();

    let x = Variable::new(
        Tensor::randn(&[32, 16], crate::tensor::test_opts()).unwrap(),
        false,
    );
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![32, 8], "every input row must yield one output row");
}

#[test]
fn test_switch_index_count_mismatch_errors() {
    let graph = FlowBuilder::from(Identity)
        .switch(
            BadCountSelector { count: 3 },
            vec![Box::new(Doubler), Box::new(Tripler)],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
    let err = graph.forward(&x).unwrap_err().to_string();
    assert!(
        err.contains("3 branch indices") && err.contains("2 rows"),
        "error must name the mismatch, got: {err}"
    );
}

#[test]
fn test_switch_per_sample_out_of_range_errors() {
    let graph = FlowBuilder::from(Identity)
        .switch(
            OutOfRangeSelector,
            vec![Box::new(Doubler), Box::new(Tripler)],
        )
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
    let err = graph.forward(&x).unwrap_err().to_string();
    assert!(
        err.contains("branch 7") && err.contains("row 1"),
        "error must name the branch and the offending row, got: {err}"
    );
}

#[test]
fn test_argmax_selector_accepts_using_refs() {
    // ArgmaxSelector implements NamedInputModule, so .using() on a switch it
    // routes must build — the tutorial documents exactly this shape.
    let graph = FlowBuilder::from(Linear::on_device(4, 6, crate::tensor::test_device()).unwrap())
        .tag("features")
        .switch(
            ArgmaxSelector::on_device(6, 2, crate::tensor::test_device()).unwrap(),
            vec![
                Box::new(Linear::on_device(6, 6, crate::tensor::test_device()).unwrap()),
                Box::new(Linear::on_device(6, 6, crate::tensor::test_device()).unwrap()),
            ],
        )
        .using(&["features"])
        .build()
        .expect("switch with ArgmaxSelector must accept .using() refs");

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 6]);
}

// --- Halt tests ---


#[test]
fn test_threshold_halt_while() {
    // body = Doubler, halt when max > 10
    // input [1,2] → iter1 [2,4] → iter2 [4,8] → iter3 [8,16] halt (16 > 10)
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .while_cond(ThresholdHalt::new(10.0), 20)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // Should stop at [8, 16] (max=16 > 10)
    assert!((data[0] - 8.0).abs() < 1e-5, "expected 8, got {}", data[0]);
    assert!((data[1] - 16.0).abs() < 1e-5, "expected 16, got {}", data[1]);
}

#[test]
fn test_threshold_halt_until() {
    // Until: body runs first, then check
    // input [1,2] → iter1 body [2,4] check (max=4 < 10 continue)
    //             → iter2 body [4,8] check (max=8 < 10 continue)
    //             → iter3 body [8,16] check (max=16 > 10 halt)
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .until_cond(ThresholdHalt::new(10.0), 20)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // Should stop at [8, 16] (max=16 > 10)
    assert!((data[0] - 8.0).abs() < 1e-5, "expected 8, got {}", data[0]);
    assert!((data[1] - 16.0).abs() < 1e-5, "expected 16, got {}", data[1]);
}

#[test]
fn test_threshold_halt_immediate() {
    // Threshold already exceeded: while should not iterate
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .while_cond(ThresholdHalt::new(0.5), 20)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // max=2.0 > 0.5 → halt immediately, input passes through
    assert!((data[0] - 1.0).abs() < 1e-5, "expected 1, got {}", data[0]);
    assert!((data[1] - 2.0).abs() < 1e-5, "expected 2, got {}", data[1]);
}

#[test]
fn test_learned_halt_parameters() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .until_cond(LearnedHalt::on_device(2, crate::tensor::test_device()).unwrap(), 5)
        .build()
        .unwrap();

    // Body Linear: 2 params, LearnedHalt Linear(2→1): 2 params = 4
    let params = graph.parameters();
    assert_eq!(params.len(), 4);
}

/// Halt condition that returns one value per row instead of a scalar.
struct PerRowHalt;
impl Module for PerRowHalt {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let rows = input.shape()[0];
        Ok(Variable::new(from_f32(&vec![-1.0; rows as usize], &[rows]), false))
    }
    fn parameters(&self) -> Vec<Parameter> { vec![] }
}

#[test]
fn test_learned_halt_pools_batched_state() {
    // A batched state gives LearnedHalt one probe per row; it must pool them
    // into the single scalar the loop needs instead of letting row 0 decide.
    let halt = LearnedHalt::on_device(2, crate::tensor::test_device()).unwrap();
    let batched = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);

    let decision = halt.forward(&batched).unwrap();
    assert_eq!(
        decision.data().numel(),
        1,
        "halt decision must be one scalar for the whole batch"
    );
}

#[test]
fn test_learned_halt_loop_runs_batched() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .while_cond(LearnedHalt::on_device(2, crate::tensor::test_device()).unwrap(), 3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![3, 2]);
}

#[test]
fn test_loop_rejects_non_scalar_condition() {
    let graph = FlowBuilder::from(Identity)
        .loop_body(Doubler)
        .while_cond(PerRowHalt, 3)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
    let err = graph.forward(&x).unwrap_err().to_string();
    assert!(
        err.contains("2 values") && err.contains("whole batch"),
        "error must explain the scalar contract, got: {err}"
    );
}

