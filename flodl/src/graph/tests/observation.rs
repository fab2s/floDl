//! Observation tests: `tagged` capture, `collect` / `flush_trend` /
//! `reset_trend`, tag-groups, and `collect` with reduction
//! (sum/mean/max/min/norm/scalar).

use super::*;

#[test]
fn test_tagged_capture() {
    // Tag intermediate output and retrieve it after forward
    let graph = FlowBuilder::from(Identity)
        .tag("features")
        .through(Doubler)
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let _ = graph.forward(&x).unwrap();

    // Tagged value should be the identity output (before doubling)
    let features = graph.tagged("features").unwrap();
    let data = features.data().to_f32_vec().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-5);
    assert!((data[1] - 2.0).abs() < 1e-5);

    assert!(graph.tagged("nonexistent").is_none());
}

#[test]
fn test_tagged_updates_each_forward() {
    let graph = FlowBuilder::from(Doubler)
        .tag("doubled")
        .build()
        .unwrap();

    let x1 = Variable::new(from_f32(&[1.0], &[1, 1]), false);
    let _ = graph.forward(&x1).unwrap();
    let v1 = graph.tagged("doubled").unwrap().item().unwrap();
    assert!((v1 - 2.0).abs() < 1e-5);

    let x2 = Variable::new(from_f32(&[5.0], &[1, 1]), false);
    let _ = graph.forward(&x2).unwrap();
    let v2 = graph.tagged("doubled").unwrap().item().unwrap();
    assert!((v2 - 10.0).abs() < 1e-5);
}

#[test]
fn test_tag_names() {
    let graph = FlowBuilder::from(Identity)
        .tag("a")
        .through(Identity)
        .tag("b")
        .build()
        .unwrap();

    let mut names = graph.tag_names();
    names.sort();
    assert_eq!(names, vec!["a", "b"]);
}


#[test]
fn test_collect_flush_trend() {
    // Simulate a training loop with collect → flush → trend
    let graph = FlowBuilder::from(ScalarSum)
        .tag("loss")
        .build()
        .unwrap();

    // Epoch 1: 3 batches with different inputs
    for val in &[1.0f32, 2.0, 3.0] {
        let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect(&["loss"]).unwrap();
    }
    // batch buffer should have [1, 2, 3]
    let collected = graph.collected("loss");
    assert_eq!(collected.len(), 3);

    graph.flush(&["loss"]);
    assert_eq!(graph.flush_count(), 1);

    // Epoch 2: 3 batches
    for val in &[0.5f32, 0.3, 0.2] {
        let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect(&["loss"]).unwrap();
    }
    graph.flush(&["loss"]);
    assert_eq!(graph.flush_count(), 2);

    // Trend should show decrease: epoch1 mean=2.0, epoch2 mean≈0.333
    let trend = graph.trend("loss");
    assert_eq!(trend.len(), 2);
    assert!((trend.values()[0] - 2.0).abs() < 1e-5);
    assert!((trend.values()[1] - (1.0 / 3.0)).abs() < 1e-5);
    assert!(trend.improving(0));
}

#[test]
fn test_record_external_values() {
    let graph = FlowBuilder::from(Identity).build().unwrap();

    graph.record("external_loss", &[0.5, 0.4, 0.3]);
    graph.flush(&["external_loss"]);

    graph.record("external_loss", &[0.1, 0.05]);
    graph.flush(&["external_loss"]);

    let trend = graph.trend("external_loss");
    assert_eq!(trend.len(), 2);
    assert!((trend.values()[0] - 0.4).abs() < 1e-5); // mean(0.5, 0.4, 0.3)
    assert!((trend.values()[1] - 0.075).abs() < 1e-5); // mean(0.1, 0.05)
    assert!(trend.improving(0));
}

#[test]
fn test_flush_all() {
    let graph = FlowBuilder::from(Identity).build().unwrap();

    graph.record("a", &[1.0, 2.0]);
    graph.record("b", &[3.0, 4.0]);
    graph.flush(&[]); // flush all

    assert_eq!(graph.trend("a").len(), 1);
    assert_eq!(graph.trend("b").len(), 1);
}

#[test]
fn test_reset_trend() {
    let graph = FlowBuilder::from(Identity).build().unwrap();

    graph.record("loss", &[1.0]);
    graph.flush(&[]);
    assert_eq!(graph.trend("loss").len(), 1);

    graph.reset_trend(&["loss"]);
    assert_eq!(graph.trend("loss").len(), 0);
}

#[test]
fn test_trends_group() {
    let graph = FlowBuilder::from(Identity).build().unwrap();

    // Two decreasing series
    for epoch in &[10.0, 8.0, 6.0, 4.0] {
        graph.record("a", &[*epoch]);
        graph.record("b", &[*epoch * 0.5]);
        graph.flush(&[]);
    }

    let tg = graph.trends(&["a", "b"]);
    assert_eq!(tg.len(), 2);
    assert!(tg.all_improving(0));
}

// --- TagGroup tests ---


#[test]
fn test_tag_group() {
    // Split into 3 branches with tag_group, then merge
    let graph = FlowBuilder::from(Identity)
        .split(vec![
            Box::new(Doubler),
            Box::new(Tripler),
            Box::new(Identity),
        ])
        .tag_group("branch")
        .merge(MergeOp::Add)
        .build()
        .unwrap();

    // Check group registration
    let members = graph.tag_group("branch").unwrap();
    assert_eq!(members, &["branch_0", "branch_1", "branch_2"]);

    // Non-existent group returns None
    assert!(graph.tag_group("nonexistent").is_none());

    // Tags work for observation
    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let _ = graph.forward(&x).unwrap();

    let b0 = graph.tagged("branch_0").unwrap();
    let b0_data = b0.data().to_f32_vec().unwrap();
    assert!((b0_data[0] - 2.0).abs() < 1e-5, "doubler: got {}", b0_data[0]);

    let b1 = graph.tagged("branch_1").unwrap();
    let b1_data = b1.data().to_f32_vec().unwrap();
    assert!((b1_data[0] - 3.0).abs() < 1e-5, "tripler: got {}", b1_data[0]);
}

#[test]
fn test_tag_group_observation() {
    // Tag group with collect/flush and trends expansion
    let graph = FlowBuilder::from(Identity)
        .split(vec![Box::new(ScalarSum), Box::new(ScalarSum)])
        .tag_group("head")
        .merge(MergeOp::Add)
        .build()
        .unwrap();

    // Run a few epochs
    for epoch in &[1.0f32, 2.0, 3.0] {
        let x = Variable::new(from_f32(&[*epoch], &[1, 1]), false);
        let _ = graph.forward(&x).unwrap();
        graph.collect(&["head_0", "head_1"]).unwrap();
        graph.flush(&["head_0", "head_1"]);
    }

    // Trends with group expansion
    let tg = graph.trends(&["head"]);
    assert_eq!(tg.len(), 2); // head_0 and head_1
}

#[test]
fn test_tag_group_errors() {
    // tag_group on single stream should error
    let result = FlowBuilder::from(Identity)
        .tag_group("bad")
        .build();
    assert!(result.is_err());

    // Duplicate group name
    let result = FlowBuilder::from(Identity)
        .split(vec![Box::new(Doubler), Box::new(Tripler)])
        .tag_group("x")
        .merge(MergeOp::Add)
        .split(vec![Box::new(Doubler), Box::new(Tripler)])
        .tag_group("x")
        .merge(MergeOp::Add)
        .build();
    assert!(result.is_err());
}

// --- Input tests ---

/// Module that adds all refs to input (for multi-input testing).
#[test]
fn test_collect_with_sum_reduction() {
    // Non-scalar tagged output reduced via Sum
    let graph = FlowBuilder::from(Identity)
        .tag("features")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    let _ = graph.forward(&x).unwrap();
    graph.collect_with(&["features"], Reduce::Sum).unwrap();

    let collected = graph.collected("features");
    assert_eq!(collected.len(), 1);
    assert!((collected[0] - 6.0).abs() < 1e-5, "sum([1,2,3]) = 6, got {}", collected[0]);
}

#[test]
fn test_collect_with_mean_reduction() {
    let graph = FlowBuilder::from(Identity)
        .tag("out")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[2.0, 4.0, 6.0], &[1, 3]), false);
    let _ = graph.forward(&x).unwrap();
    graph.collect_with(&["out"], Reduce::Mean).unwrap();

    let collected = graph.collected("out");
    assert!((collected[0] - 4.0).abs() < 1e-5, "mean([2,4,6]) = 4, got {}", collected[0]);
}

#[test]
fn test_collect_with_max_reduction() {
    let graph = FlowBuilder::from(Identity)
        .tag("out")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 5.0, 3.0], &[1, 3]), false);
    let _ = graph.forward(&x).unwrap();
    graph.collect_with(&["out"], Reduce::Max).unwrap();

    let collected = graph.collected("out");
    assert!((collected[0] - 5.0).abs() < 1e-5, "max([1,5,3]) = 5, got {}", collected[0]);
}

#[test]
fn test_collect_with_min_reduction() {
    let graph = FlowBuilder::from(Identity)
        .tag("out")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[-2.0, 0.0, 3.0], &[1, 3]), false);
    let _ = graph.forward(&x).unwrap();
    graph.collect_with(&["out"], Reduce::Min).unwrap();

    let collected = graph.collected("out");
    assert!((collected[0] - (-2.0)).abs() < 1e-5, "min([-2,0,3]) = -2, got {}", collected[0]);
}

#[test]
fn test_collect_with_norm_reduction() {
    let graph = FlowBuilder::from(Identity)
        .tag("out")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[3.0, 4.0], &[1, 2]), false);
    let _ = graph.forward(&x).unwrap();
    graph.collect_with(&["out"], Reduce::Norm).unwrap();

    let collected = graph.collected("out");
    // L2 norm of [3, 4] = 5
    assert!((collected[0] - 5.0).abs() < 1e-4, "norm([3,4]) = 5, got {}", collected[0]);
}

#[test]
fn test_collect_rejects_non_scalar() {
    // Plain collect() should reject non-scalar outputs
    let graph = FlowBuilder::from(Identity)
        .tag("out")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let _ = graph.forward(&x).unwrap();
    assert!(graph.collect(&["out"]).is_err());
}

#[test]
fn test_collect_with_scalar_passthrough() {
    // collect_with on already-scalar output should work without reduction
    let graph = FlowBuilder::from(ScalarSum)
        .tag("loss")
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
    let _ = graph.forward(&x).unwrap();
    graph.collect_with(&["loss"], Reduce::Max).unwrap();

    let collected = graph.collected("loss");
    // ScalarSum yields 10.0 (scalar), so it should pass through directly
    assert!((collected[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_collect_with_flush_trend_pipeline() {
    // Full pipeline: non-scalar → reduce → flush → trend
    let graph = FlowBuilder::from(Identity)
        .tag("h")
        .build()
        .unwrap();

    // Epoch 1: two batches with decreasing norms
    let x1 = Variable::new(from_f32(&[3.0, 4.0], &[1, 2]), false);
    let _ = graph.forward(&x1).unwrap();
    graph.collect_with(&["h"], Reduce::Norm).unwrap();

    let x2 = Variable::new(from_f32(&[1.0, 0.0], &[1, 2]), false);
    let _ = graph.forward(&x2).unwrap();
    graph.collect_with(&["h"], Reduce::Norm).unwrap();

    graph.flush(&["h"]);

    // Epoch 2
    let x3 = Variable::new(from_f32(&[0.5, 0.5], &[1, 2]), false);
    let _ = graph.forward(&x3).unwrap();
    graph.collect_with(&["h"], Reduce::Norm).unwrap();
    graph.flush(&["h"]);

    let trend = graph.trend("h");
    assert_eq!(trend.len(), 2);
    // Epoch 1 mean: (5.0 + 1.0) / 2 = 3.0
    assert!((trend.values()[0] - 3.0).abs() < 1e-4);
    assert!(trend.improving(0)); // norms should be decreasing
}

// --- Map.over and Map.slices tests ---
