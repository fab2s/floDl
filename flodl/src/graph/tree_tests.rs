use super::PathKind;
use crate::autograd::Variable;
use crate::graph::FlowBuilder;
use crate::nn::ReLU;
use crate::nn::{Linear, Module};
use crate::tensor::{Tensor, test_device, test_opts};

#[test]
fn test_unlabeled_graph_no_children() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(ReLU::new())
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Unlabeled child is NOT registered
    assert!(outer.tree_children().is_empty());
    // But parameters are still collected (backward compat)
    assert_eq!(outer.parameters().len(), 4); // 2 from inner Linear + 2 from outer Linear
}

#[test]
fn test_labeled_child_registered() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(ReLU::new())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    assert_eq!(outer.tree_children().len(), 1);
    assert!(outer.tree_children().contains_key("encoder"));
    assert!(outer.child_graph("encoder").is_some());
    assert_eq!(
        outer.child_graph("encoder").unwrap().label(),
        Some("encoder")
    );
}

#[test]
fn test_composed_flag() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("child")
        .build()
        .unwrap();

    // Standalone: not composed
    assert!(!inner.is_composed());

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // After composition: child is composed
    let child = outer.child_graph("child").unwrap();
    assert!(child.is_composed());
    // Parent is not composed
    assert!(!outer.is_composed());
}

#[test]
fn test_label_collision_error() {
    let dev = test_device();

    let a = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("dupe")
        .build()
        .unwrap();
    let b = FlowBuilder::from(Linear::on_device(4, 2, dev).unwrap())
        .label("dupe")
        .build()
        .unwrap();

    let result = FlowBuilder::from(a).through(b).build();

    let msg = result.err().expect("should be Err").to_string();
    assert!(msg.contains("duplicate child graph label"), "got: {}", msg);
}

#[test]
fn test_dot_in_label_error() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("a.b")
        .build()
        .unwrap();

    let result = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build();

    let msg = result.err().expect("should be Err").to_string();
    assert!(msg.contains("contains a dot"), "got: {}", msg);
}

#[test]
fn test_label_tag_same_node_ok() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    // Tag the same node as the child graph label
    let outer = FlowBuilder::from(inner)
        .tag("encoder")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build();

    assert!(outer.is_ok());
}

#[test]
fn test_resolve_single_segment_child() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    assert_eq!(outer.validate_path("encoder").unwrap(), PathKind::Subgraph);
}

#[test]
fn test_resolve_single_segment_tag() {
    let dev = test_device();

    let outer = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("hidden")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    assert_eq!(outer.validate_path("hidden").unwrap(), PathKind::Tag);
}

#[test]
fn test_resolve_multi_segment() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("hidden")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    assert_eq!(
        outer.validate_path("encoder.hidden").unwrap(),
        PathKind::Tag
    );
}

#[test]
fn test_resolve_multi_level() {
    let dev = test_device();

    let innermost = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("read")
        .build()
        .unwrap();
    let middle = FlowBuilder::from(innermost)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("letter")
        .build()
        .unwrap();
    let outer = FlowBuilder::from(middle)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    assert_eq!(outer.validate_path("letter").unwrap(), PathKind::Subgraph);
    assert_eq!(
        outer.validate_path("letter.read").unwrap(),
        PathKind::Subgraph
    );
}

#[test]
fn test_resolve_invalid_path_error() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Non-existent single segment
    assert!(outer.validate_path("nonexistent").is_err());
    // Non-existent dotted path
    assert!(outer.validate_path("encoder.nonexistent").is_err());
    // Dotting into non-child first segment
    assert!(outer.validate_path("nonexistent.foo").is_err());
}

#[test]
fn test_subgraph_returns_graph() {
    let dev = test_device();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    let sub = outer.subgraph("encoder").unwrap();
    assert_eq!(sub.label(), Some("encoder"));
    assert_eq!(sub.parameters().len(), 2); // 1 Linear: weight + bias
}

#[test]
fn test_forward_still_works_with_tree() {
    let dev = test_device();
    let opts = test_opts();

    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(ReLU::new())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(Tensor::randn(&[1, 3], opts).unwrap(), false);
    let y = outer.forward(&x).unwrap();
    assert_eq!(y.shape(), vec![1, 2]);
}

// ── Phase B: Training control ────────────────────────────────────

#[test]
fn test_parameters_at_subgraph() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Child has 2 Linear layers = 4 params (2 weight + 2 bias)
    let params = outer.parameters_at("encoder").unwrap();
    assert_eq!(params.len(), 4);
    // Outer total = 4 (child) + 2 (outer Linear) = 6
    assert_eq!(outer.parameters().len(), 6);
}

#[test]
fn test_parameters_at_tag() {
    let dev = test_device();
    let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("first")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    let params = g.parameters_at("first").unwrap();
    assert_eq!(params.len(), 2); // 1 Linear: weight + bias
}

#[test]
fn test_freeze_thaw_roundtrip() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Initially not frozen
    assert!(!outer.is_frozen("encoder").unwrap());

    // Freeze child
    outer.freeze("encoder").unwrap();
    assert!(outer.is_frozen("encoder").unwrap());
    // All child params should have requires_grad = false
    for p in outer.parameters_at("encoder").unwrap() {
        assert!(p.is_frozen());
    }
    // Outer params still trainable
    let outer_params = outer.parameters();
    let outer_only: Vec<_> = outer_params.iter().filter(|p| !p.is_frozen()).collect();
    assert_eq!(outer_only.len(), 2); // outer Linear: weight + bias

    // Thaw child
    outer.thaw("encoder").unwrap();
    assert!(!outer.is_frozen("encoder").unwrap());
    for p in outer.parameters_at("encoder").unwrap() {
        assert!(!p.is_frozen());
    }
}

#[test]
fn test_freeze_deep_path() {
    let dev = test_device();
    let innermost = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("read")
        .build()
        .unwrap();
    let middle = FlowBuilder::from(innermost)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("letter")
        .build()
        .unwrap();
    let outer = FlowBuilder::from(middle)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Freeze only the innermost
    outer.freeze("letter.read").unwrap();
    assert!(outer.is_frozen("letter.read").unwrap());
    // "letter" overall is NOT fully frozen (it has its own Linear too)
    assert!(!outer.is_frozen("letter").unwrap());
}

#[test]
fn test_named_parameters_at_uses_target_namespace() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("hidden")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Subgraph: uses child's own namespace
    let named = outer.named_parameters_at("encoder").unwrap();
    assert_eq!(named.len(), 4);
    // Names should use child-local prefixes (tag "hidden" and node id)
    assert!(named.iter().any(|(n, _)| n.starts_with("hidden/")));
}

#[test]
fn test_freeze_invalid_path_error() {
    let dev = test_device();
    let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .build()
        .unwrap();

    assert!(g.freeze("nonexistent").is_err());
    assert!(g.thaw("nonexistent").is_err());
    assert!(g.is_frozen("nonexistent").is_err());
    assert!(g.parameters_at("nonexistent").is_err());
}

#[test]
fn test_set_training_at() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(crate::nn::Dropout::new(0.5))
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Set child to eval mode
    outer.set_training_at("encoder", false).unwrap();
    // Set child back to training mode
    outer.set_training_at("encoder", true).unwrap();
    // Invalid path errors
    assert!(outer.set_training_at("nonexistent", false).is_err());
}

// ── Phase C: Checkpoint composition ──────────────────────────────

#[test]
fn test_subgraph_checkpoint_roundtrip() {
    let dev = test_device();
    // Build and "train" a child graph standalone
    let child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    // Save child checkpoint
    let dir = std::env::temp_dir().join("flodl_test_subgraph_ckpt");
    std::fs::create_dir_all(&dir).unwrap();
    let ckpt_path = dir.join("encoder.fdl");
    child.save_checkpoint(ckpt_path.to_str().unwrap()).unwrap();

    // Build parent with a fresh (randomly initialized) child of same architecture
    let fresh_child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let parent = FlowBuilder::from(fresh_child)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Load child checkpoint into parent's subgraph
    let report = parent
        .load_subgraph_checkpoint("encoder", ckpt_path.to_str().unwrap())
        .unwrap();
    assert!(report.loaded.len() >= 4); // At least weight+bias from 2 Linears
    assert!(report.missing.is_empty());

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_subgraph_checkpoint_preserves_parent_params() {
    let dev = test_device();
    let child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let dir = std::env::temp_dir().join("flodl_test_preserve_parent");
    std::fs::create_dir_all(&dir).unwrap();
    let ckpt_path = dir.join("encoder.fdl");
    child.save_checkpoint(ckpt_path.to_str().unwrap()).unwrap();

    // Build parent with fresh child
    let fresh_child = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();
    let parent = FlowBuilder::from(fresh_child)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Snapshot parent-level param data
    let parent_w = parent.parameters().last().unwrap().variable.data().clone();

    // Load child checkpoint
    parent
        .load_subgraph_checkpoint("encoder", ckpt_path.to_str().unwrap())
        .unwrap();

    // Parent param unchanged
    let parent_w_after = parent.parameters().last().unwrap().variable.data().clone();
    let diff = parent_w
        .sub(&parent_w_after)
        .unwrap()
        .abs()
        .unwrap()
        .sum()
        .unwrap()
        .item()
        .unwrap();
    assert!(
        diff < 1e-10,
        "parent params should be unchanged, diff={}",
        diff
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Phase D: Cross-boundary observation ──────────────────────────

#[test]
fn test_tagged_at_returns_value_after_forward() {
    let dev = test_device();
    let opts = test_opts();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("hidden")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(Tensor::randn(&[1, 3], opts).unwrap(), false);
    outer.forward(&x).unwrap();

    let val = outer.tagged_at("encoder.hidden").unwrap();
    assert!(val.is_some());
    assert_eq!(val.unwrap().shape(), vec![1, 4]);
}

#[test]
fn test_tagged_at_before_forward_returns_none() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("hidden")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Before forward: path exists but no value computed
    let val = outer.tagged_at("encoder.hidden").unwrap();
    assert!(val.is_none());
}

#[test]
fn test_tagged_at_invalid_path_returns_err() {
    let dev = test_device();
    let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .build()
        .unwrap();

    assert!(g.tagged_at("nonexistent.tag").is_err());
}

#[test]
fn test_record_at_and_trend_at() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Record into child's buffer
    outer.record_at("encoder.loss", 0.5).unwrap();
    outer.record_at("encoder.loss", 0.3).unwrap();

    // Flush child's buffers to see the trend
    let child = outer.child_graph("encoder").unwrap();
    child.flush(&[]);

    let trend = outer.trend_at("encoder.loss").unwrap();
    assert_eq!(trend.len(), 1); // one epoch flushed
}

// ── Phase E: Developer experience ────────────────────────────────

#[test]
fn test_internal_tag_hidden_from_parent() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("_plumbing")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .tag("output")
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Auto-internal: _plumbing starts with underscore
    assert!(
        outer
            .child_graph("encoder")
            .unwrap()
            .internal_tags()
            .contains("_plumbing")
    );
    // Internal tag blocked from parent
    assert!(outer.tagged_at("encoder._plumbing").is_err());
    // Non-internal tag accessible
    assert_eq!(
        outer.validate_path("encoder.output").unwrap(),
        PathKind::Tag
    );
}

#[test]
fn test_explicit_internal_tag() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("intermediate")
        .internal("intermediate")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Explicitly internal: blocked from parent
    assert!(outer.tagged_at("encoder.intermediate").is_err());
}

#[test]
fn test_tree_summary_output() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .tag("hidden")
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    let summary = outer.tree_summary();
    assert!(
        summary.contains("Graph Tree"),
        "missing header:\n{}",
        summary
    );
    assert!(
        summary.contains("encoder"),
        "missing child label:\n{}",
        summary
    );
    assert!(
        summary.contains("Parameter Summary"),
        "missing param summary:\n{}",
        summary
    );
}

#[test]
fn test_param_summary_output() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    let summary = outer.param_summary();
    assert!(summary.contains("encoder"), "missing child:\n{}", summary);
    assert!(
        summary.contains("(own)"),
        "missing own params:\n{}",
        summary
    );
    assert!(
        summary.contains("trainable"),
        "missing trainable:\n{}",
        summary
    );
}

// ── Phase F: Tree-aware observation ──────────────────────────────

#[test]
fn test_flush_recurses_into_children() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Record into child via tree path
    outer.record_at("encoder.loss", 0.5).unwrap();
    outer.record_at("encoder.loss", 0.3).unwrap();
    // Record into parent
    outer.record_scalar("parent_loss", 1.0);

    // Single flush on parent should flush both
    outer.flush(&[]);

    // Parent flushed
    assert_eq!(outer.flush_count(), 1);
    assert_eq!(outer.trend("parent_loss").len(), 1);

    // Child also flushed
    let child = outer.child_graph("encoder").unwrap();
    assert_eq!(child.flush_count(), 1);
    assert_eq!(child.trend("loss").len(), 1);
}

#[test]
fn test_latest_metrics_includes_children() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    // Record and flush
    outer.record_at("encoder.ce", 0.5).unwrap();
    outer.record_scalar("total_loss", 1.0);
    outer.flush(&[]);

    let metrics = outer.latest_metrics();
    let names: Vec<&str> = metrics.iter().map(|(n, _)| n.as_str()).collect();

    // Parent metric present
    assert!(
        names.contains(&"total_loss"),
        "missing parent metric: {:?}",
        names
    );
    // Child metric present with dotted prefix
    assert!(
        names.contains(&"encoder.ce"),
        "missing child metric: {:?}",
        names
    );
}

#[test]
fn test_latest_metrics_local_excludes_children() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    outer.record_at("encoder.ce", 0.5).unwrap();
    outer.record_scalar("total_loss", 1.0);
    outer.flush(&[]);

    let local = outer.latest_metrics_local();
    let names: Vec<&str> = local.iter().map(|(n, _)| n.as_str()).collect();

    assert!(names.contains(&"total_loss"));
    assert!(
        !names.contains(&"encoder.ce"),
        "local should not include children: {:?}",
        names
    );
}

#[test]
fn test_double_flush_is_safe() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    outer.record_at("encoder.loss", 0.5).unwrap();

    // Flush child explicitly first
    let child = outer.child_graph("encoder").unwrap();
    child.flush(&[]);
    assert_eq!(child.flush_count(), 1);

    // Parent flush recurses — child buffer already empty, no double epoch
    outer.flush(&[]);
    assert_eq!(child.flush_count(), 1); // still 1, not 2
    assert_eq!(child.trend("loss").len(), 1); // one epoch, not two
}

#[test]
fn test_flush_local_skips_children() {
    let dev = test_device();
    let inner = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("encoder")
        .build()
        .unwrap();

    let outer = FlowBuilder::from(inner)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .build()
        .unwrap();

    outer.record_at("encoder.loss", 0.5).unwrap();
    outer.record_scalar("parent_loss", 1.0);

    // flush_local: only parent
    outer.flush_local(&[]);

    assert_eq!(outer.flush_count(), 1);
    assert_eq!(outer.trend("parent_loss").len(), 1);

    // Child NOT flushed — data still in batch buffer
    let child = outer.child_graph("encoder").unwrap();
    assert_eq!(child.flush_count(), 0);
    assert_eq!(child.collected("loss").len(), 1); // still in batch buffer
}

#[test]
fn test_flush_recurses_multi_level() {
    let dev = test_device();
    let innermost = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .label("read")
        .build()
        .unwrap();
    let middle = FlowBuilder::from(innermost)
        .through(Linear::on_device(4, 2, dev).unwrap())
        .label("letter")
        .build()
        .unwrap();
    let outer = FlowBuilder::from(middle)
        .through(Linear::on_device(2, 1, dev).unwrap())
        .build()
        .unwrap();

    // Record into deepest child
    outer.record_at("letter.read.hidden_loss", 0.7).unwrap();
    // Record into middle child
    outer.record_at("letter.mid_loss", 0.4).unwrap();

    outer.flush(&[]);

    // All levels flushed
    let metrics = outer.latest_metrics();
    let names: Vec<&str> = metrics.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        names.contains(&"letter.mid_loss"),
        "missing middle: {:?}",
        names
    );
    assert!(
        names.contains(&"letter.read.hidden_loss"),
        "missing deep: {:?}",
        names
    );
}

#[test]
fn test_metrics_no_children_unchanged() {
    // Verify single-graph behavior is identical (no regression)
    let dev = test_device();
    let g = FlowBuilder::from(Linear::on_device(3, 4, dev).unwrap())
        .build()
        .unwrap();

    g.record_scalar("loss", 0.5);
    g.record_scalar("loss", 0.3);
    g.flush(&[]);

    let metrics = g.latest_metrics();
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].0, "loss");
    assert!((metrics[0].1 - 0.4).abs() < 1e-10); // mean of 0.5 and 0.3
}
