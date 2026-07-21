//! Map tests: `map_each` / `map_batched` / `map_backward`, plus
//! advanced map (`map_over_tag`, `map_slices` variants) and input
//! port tests.

use super::*;

#[test]
fn test_map_each() {
    // Map doubler over 3 elements along dim 0
    let graph = FlowBuilder::from(Identity)
        .map(Doubler)
        .each()
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert_eq!(y.shape(), vec![3, 2]);
    assert!((data[0] - 2.0).abs() < 1e-5);
    assert!((data[5] - 12.0).abs() < 1e-5);
}

#[test]
fn test_map_batched() {
    // Batched: pass full tensor, skip element-wise
    let graph = FlowBuilder::from(Identity)
        .map(Doubler)
        .batched()
        .each()
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_map_backward() {
    let graph = FlowBuilder::from(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .map(Linear::on_device(2, 2, crate::tensor::test_device()).unwrap())
        .each()
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let y = graph.forward(&x).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
    }
}

// --- Observation tests ---

/// Scalar output module: sum all elements to a single value.
#[test]
fn test_input_auxiliary() {
    // Graph with auxiliary inputs: From(identity) + Input("ctx")
    // Downstream: through(SumRefs).using("ctx")
    let graph = FlowBuilder::from(Identity)
        .input(&["ctx"])
        .through(SumRefs)
        .using(&["ctx"])
        .build()
        .unwrap();

    let main = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let ctx = Variable::new(from_f32(&[10.0, 20.0], &[1, 2]), false);

    let y = graph.forward_multi(&[main, ctx]).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // SumRefs adds ctx to main: [1+10, 2+20] = [11, 22]
    assert!((data[0] - 11.0).abs() < 1e-5, "got {}", data[0]);
    assert!((data[1] - 22.0).abs() < 1e-5, "got {}", data[1]);
}

#[test]
fn test_input_multiple() {
    // Graph with two auxiliary inputs
    let graph = FlowBuilder::from(Identity)
        .input(&["a", "b"])
        .through(SumRefs)
        .using(&["a", "b"])
        .build()
        .unwrap();

    let main = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    let a = Variable::new(from_f32(&[10.0], &[1, 1]), false);
    let b = Variable::new(from_f32(&[100.0], &[1, 1]), false);

    let y = graph.forward_multi(&[main, a, b]).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // 1 + 10 + 100 = 111
    assert!((data[0] - 111.0).abs() < 1e-5, "got {}", data[0]);
}

#[test]
fn test_input_error_count_mismatch() {
    let graph = FlowBuilder::from(Identity)
        .input(&["ctx"])
        .build()
        .unwrap();

    // forward() with single input should fail (expects 2: main + ctx)
    let x = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    assert!(graph.forward(&x).is_err());
}

// --- Graph set_training test ---


#[test]
fn test_map_over_tag() {
    // Tag a tensor, then map over it from a different stream position
    let graph = FlowBuilder::from(Identity)
        .tag("features")
        .through(Doubler)        // stream is now 2x
        .map(Doubler)
        .over("features")        // map over original (1x), not current stream (2x)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false);
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();
    // .over("features") maps Doubler over the tagged value (original x)
    // Doubler: x + x = 2x, applied element-wise along dim 0
    assert_eq!(y.shape(), vec![2, 2]);
    assert!((data[0] - 2.0).abs() < 1e-5);  // 1.0 * 2
    assert!((data[1] - 4.0).abs() < 1e-5);  // 2.0 * 2
    assert!((data[2] - 6.0).abs() < 1e-5);  // 3.0 * 2
    assert!((data[3] - 8.0).abs() < 1e-5);  // 4.0 * 2
}

#[test]
fn test_map_over_unknown_tag_error() {
    let result = FlowBuilder::from(Identity)
        .map(Doubler)
        .over("nonexistent")
        .build();
    assert!(result.is_err());
}

#[test]
fn test_map_slices() {
    // Input [2, 4], slices(2): decompose → [4, 2], map Doubler, recompose → [2, 4]
    let graph = FlowBuilder::from(Identity)
        .map(Doubler)
        .slices(2)
        .build()
        .unwrap();

    let x = Variable::new(
        from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
        false,
    );
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    // Each element doubled
    assert_eq!(y.shape(), vec![2, 4]);
    assert!((data[0] - 2.0).abs() < 1e-5);
    assert!((data[7] - 16.0).abs() < 1e-5);
}

#[test]
fn test_map_slices_batched() {
    // Same as above but with batched fast path
    let graph = FlowBuilder::from(Identity)
        .map(Doubler)
        .batched()
        .slices(2)
        .build()
        .unwrap();

    let x = Variable::new(
        from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]),
        false,
    );
    let y = graph.forward(&x).unwrap();
    let data = y.data().to_f32_vec().unwrap();

    assert_eq!(y.shape(), vec![2, 4]);
    assert!((data[0] - 2.0).abs() < 1e-5);
    assert!((data[7] - 16.0).abs() < 1e-5);
}

#[test]
fn test_map_slices_gradient() {
    // Input [2, 4] → slices(2) decomposes to [4, 2] → Linear(2, 3) → [4, 3] → recompose [2, 6]
    let graph = FlowBuilder::from(Identity)
        .map(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
        .slices(2)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]), true);
    let y = graph.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![2, 6]); // 3 * 2 slices = 6
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    assert!(x.grad().is_some());
    for p in graph.parameters() {
        assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
    }
}

#[test]
fn test_map_slices_not_divisible_error() {
    let graph = FlowBuilder::from(Identity)
        .map(Doubler)
        .slices(3)
        .build()
        .unwrap();

    // [2, 4] with slices(3) — 4 not divisible by 3
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]), false);
    assert!(graph.forward(&x).is_err());
}

// -----------------------------------------------------------------------
// Graph::set_scheduler -- regression guard
// -----------------------------------------------------------------------
//
// Original bug (2026-04-13): sync mode (graph.set_optimizer + graph.step())
// had no scheduler plumbing, so the optimizer LR stayed constant for the
// entire run regardless of what scheduler the user attached. These tests
// assert that set_scheduler drives the optimizer LR through step(), that
// training_step advances once per step(), and that lr_scale is applied
// multiplicatively.

