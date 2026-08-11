//! Miscellaneous graph tests: training-mode + module-walker,
//! profiling + DOT, save/load checkpoint + sidecars,
//! named-parameter scoping, structural-hash + labels, and
//! LR-scheduler integration.

use super::*;

#[test]
fn test_graph_set_training() {
    use crate::nn::Dropout;

    let graph = FlowBuilder::from(Linear::on_device(3, 3, crate::tensor::test_device()).unwrap())
        .through(Dropout::new(0.5))
        .build()
        .unwrap();

    // Training mode: dropout is active
    let x = Variable::new(from_f32(&[1.0; 12], &[4, 3]), false);
    let y1 = graph.forward(&x).unwrap();
    assert_eq!(y1.shape(), vec![4, 3]);

    // Set eval via graph
    graph.set_training(false);
    let y2 = graph.forward(&x).unwrap();
    let y3 = graph.forward(&x).unwrap();
    assert_eq!(y2.shape(), vec![4, 3]);

    // In eval: dropout is identity, so repeated forward gives same output
    let d2 = y2.data().to_f32_vec().unwrap();
    let d3 = y3.data().to_f32_vec().unwrap();
    let same = d2.iter().zip(d3.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(same, "eval mode should be deterministic (no dropout)");
}

// --- walk_modules test ---

#[test]
fn test_walk_modules() {
    use crate::nn::walk_modules;

    let l1 = Linear::on_device(2, 2, crate::tensor::test_device()).unwrap();
    let mut count = 0;
    walk_modules(&l1, &mut |_| count += 1);
    assert_eq!(count, 1); // leaf module, no children
}

// --- Profiling tests ---

#[test]
fn test_profiling_basic() {
    let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .tag("encoder")
        .through(ReLU::new())
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .tag("decoder")
        .build()
        .unwrap();

    // No profiling by default
    assert!(!graph.profiling());
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
    graph.forward(&x).unwrap();
    assert!(graph.profile().is_none());

    // Enable profiling
    graph.enable_profiling();
    assert!(graph.profiling());
    graph.forward(&x).unwrap();

    let p = graph.profile().unwrap();
    assert!(p.total.as_nanos() > 0, "total should be nonzero");
    assert!(!p.nodes.is_empty(), "should have node timings");
    assert!(!p.levels.is_empty(), "should have level timings");
    let expected_source = if crate::tensor::test_device().is_cuda() {
        ProfileSource::GpuEvents
    } else {
        ProfileSource::HostWallClock
    };
    assert_eq!(p.source, expected_source);

    // Tagged node timing
    let enc_dur = p.timing("encoder");
    assert!(enc_dur.as_nanos() > 0, "encoder timing should be nonzero");
    let dec_dur = p.timing("decoder");
    assert!(dec_dur.as_nanos() > 0, "decoder timing should be nonzero");
    assert!(p.timing("nonexistent").is_zero());

    // Graph-level timing shortcut
    assert!(graph.timing("encoder").as_nanos() > 0);

    // Display
    let s = p.to_string();
    assert!(s.contains("Forward:"));
    assert!(s.contains("Level"));

    // Disable
    graph.disable_profiling();
    assert!(!graph.profiling());
    graph.forward(&x).unwrap();
    assert!(graph.profile().is_none());
}

#[test]
fn test_profiling_gpu_event_telescoping() {
    if !crate::tensor::test_device().is_cuda() {
        return;
    }
    let dev = crate::tensor::test_device();
    let graph = FlowBuilder::from(Linear::on_device(64, 64, dev).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(64, 64, dev).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(64, 8, dev).unwrap())
        .build()
        .unwrap();
    graph.enable_profiling();

    let x = Variable::new(from_f32(&[0.5; 64], &[1, 64]), false);
    // Several passes: pass N resolves at the start of pass N+1, so the
    // event pool is exercised through re-records, not just once.
    for _ in 0..3 {
        graph.forward(&x).unwrap();
    }

    let p = graph.profile().unwrap();
    assert_eq!(p.source, ProfileSource::GpuEvents);
    assert_eq!(p.nodes.len(), 5);
    assert_eq!(
        p.levels.iter().map(|l| l.num_nodes).sum::<usize>(),
        p.nodes.len()
    );

    // Boundary events telescope: the node sum IS the pass total, up to
    // the f32-millisecond rounding of each delta.
    let sum: f64 = p.nodes.iter().map(|n| n.duration.as_secs_f64()).sum();
    let total = p.total.as_secs_f64();
    assert!(total > 0.0, "event-timed total should be nonzero");
    assert!(
        (sum - total).abs() <= total * 0.05 + 5e-6,
        "node sum {sum}s should telescope to total {total}s"
    );

    // A read right after a forward (pass still pending) serves the
    // freshest drained pass without erroring.
    graph.forward(&x).unwrap();
    assert!(graph.profile().is_some());
}

#[test]
fn test_profile_stats_min_mean_warmup() {
    let graph = FlowBuilder::from(Linear::on_device(8, 8, crate::tensor::test_device()).unwrap())
        .tag("enc")
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();
    let x = Variable::new(from_f32(&[0.5; 8], &[1, 8]), false);

    // No profiling: no stats.
    graph.forward(&x).unwrap();
    assert!(graph.profile_stats().is_none());

    graph.enable_profiling();

    // The first 3 profiled passes are warmup: skipped, no stats yet.
    // (On CUDA, profiles lag one pass, so run one extra and read
    // through the pull path, which folds a drained pending pass in.)
    for _ in 0..3 {
        graph.forward(&x).unwrap();
    }
    let after_warmup = graph.profile_stats();
    if crate::tensor::test_device().is_cuda() {
        // At most the warmup has resolved; nothing accumulated.
        assert!(after_warmup.is_none());
    } else {
        assert!(after_warmup.is_none(), "3 passes are all warmup");
    }

    for _ in 0..5 {
        graph.forward(&x).unwrap();
    }
    let stats = graph.profile_stats().unwrap();
    assert!(stats.samples >= 4, "8 passes minus 3 warmup minus lag");
    assert_eq!(stats.nodes.len(), 3);
    assert_eq!(stats.structural_hash, graph.structural_hash());
    assert_eq!(stats.nodes[0].tag, "enc");
    for n in &stats.nodes {
        assert!(
            n.min <= n.mean,
            "{}: min {:?} > mean {:?}",
            n.id,
            n.min,
            n.mean
        );
    }
    assert!(stats.total_min <= stats.total_mean);
    // The node means telescope into the total mean on the event path
    // and nearly so on the host path (per-node Instant reads add up).
    let node_mean_sum: f64 = stats.nodes.iter().map(|n| n.mean.as_secs_f64()).sum();
    assert!(node_mean_sum <= stats.total_mean.as_secs_f64() * 1.5 + 1e-4);

    // Disable wipes the accumulator with the rest of profiling state.
    graph.disable_profiling();
    assert!(graph.profile_stats().is_none());
}

#[test]
fn test_profiling_timing_trend() {
    let graph = FlowBuilder::from(ScalarSum).tag("loss").build().unwrap();

    graph.enable_profiling();

    // Simulate 2 epochs, 3 batches each
    for _ in 0..2 {
        for val in &[1.0f32, 2.0, 3.0] {
            let x = Variable::new(from_f32(&[*val], &[1, 1]), false);
            graph.forward(&x).unwrap();
            graph.collect_timings(&["loss"]);
        }
        graph.flush_timings(&[]);
    }

    let trend = graph.timing_trend("loss");
    assert_eq!(trend.len(), 2, "2 epochs flushed");
    assert!(trend.values()[0] > 0.0, "timing values should be positive");

    // Reset
    graph.reset_timing_trend(&["loss"]);
    assert_eq!(graph.timing_trend("loss").len(), 0);
}

// --- DOT tests ---

#[test]
fn test_dot_basic() {
    let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .tag("enc")
        .through(ReLU::new())
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let dot = graph.dot();
    assert!(dot.contains("digraph G"));
    assert!(dot.contains("level 0"));
    assert!(dot.contains("#enc"));
    assert!(dot.contains("->"));
}

#[test]
fn test_dot_with_profile() {
    let graph = FlowBuilder::from(Linear::on_device(3, 4, crate::tensor::test_device()).unwrap())
        .tag("enc")
        .through(Linear::on_device(4, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);

    // Without profiling: dot_with_profile falls back to structural
    let dot1 = graph.dot_with_profile();
    assert!(dot1.contains("digraph G"));

    // With profiling: includes timing annotations
    graph.enable_profiling();
    graph.forward(&x).unwrap();
    let dot2 = graph.dot_with_profile();
    assert!(dot2.contains("digraph G"));
    assert!(dot2.contains("Forward:"));
}

// --- Traced tests ---

/// A loop body that implements trace() — captures per-iteration side data.
#[test]
fn test_named_parameters_unique() {
    let graph = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let named = graph.named_parameters();
    // Two Linear layers: 2 params each (weight + bias) = 4
    assert_eq!(named.len(), 4);

    // All names should be unique
    let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    assert_eq!(names.len(), unique.len(), "duplicate names: {:?}", names);
}

#[test]
fn test_named_parameters_tagged_prefix() {
    let graph = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .tag("encoder")
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let named = graph.named_parameters();
    // First Linear is tagged "encoder", second is untagged
    let encoder_params: Vec<&str> = named
        .iter()
        .filter(|(n, _)| n.starts_with("encoder/"))
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(
        encoder_params.len(),
        2,
        "tagged node should have 2 params with 'encoder/' prefix"
    );

    // Untagged node uses its node_id (like "linear_2")
    let untagged: Vec<&str> = named
        .iter()
        .filter(|(n, _)| !n.starts_with("encoder/"))
        .map(|(n, _)| n.as_str())
        .collect();
    assert_eq!(untagged.len(), 2, "untagged node should have 2 params");
    assert!(
        untagged[0].contains('/'),
        "should have prefix/name format: {}",
        untagged[0]
    );
}

// --- Structural hash tests ---

#[test]
fn test_structural_hash_deterministic() {
    let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    assert_eq!(g1.structural_hash(), g2.structural_hash());
}

#[test]
fn test_structural_hash_differs() {
    let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    // Different architecture: different hidden size
    let g2 = FlowBuilder::from(Linear::on_device(4, 16, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(16, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    assert_ne!(g1.structural_hash(), g2.structural_hash());
}

#[test]
fn test_short_hash_length() {
    let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    assert_eq!(g.structural_hash().len(), 64);
    assert_eq!(g.short_hash().len(), 8);
    assert!(g.structural_hash().starts_with(g.short_hash()));
}

#[test]
fn test_label_default_none() {
    let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();
    assert!(g.label().is_none());
}

#[test]
fn test_label_set() {
    let g = FlowBuilder::from(Linear::on_device(2, 3, crate::tensor::test_device()).unwrap())
        .label("my-model")
        .build()
        .unwrap();
    assert_eq!(g.label(), Some("my-model"));
}

#[test]
fn test_label_does_not_affect_hash() {
    let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .label("different-label")
        .build()
        .unwrap();

    assert_eq!(g1.structural_hash(), g2.structural_hash());
}

#[test]
fn test_graph_save_load_checkpoint() {
    let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .tag("enc")
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .tag("dec")
        .build()
        .unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("test_graph_ckpt.fdl");
    let path_str = path.to_str().unwrap();

    // Save
    g.save_checkpoint(path_str).unwrap();

    // Build identical architecture, load into it
    let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .tag("enc")
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .tag("dec")
        .build()
        .unwrap();

    let report = g2.load_checkpoint(path_str).unwrap();
    assert_eq!(report.loaded.len(), 4); // 2 Linear × (weight + bias)
    assert!(report.skipped.is_empty());
    assert!(report.missing.is_empty());

    // Verify weights match
    for ((n1, p1), (n2, p2)) in g
        .named_parameters()
        .iter()
        .zip(g2.named_parameters().iter())
    {
        assert_eq!(n1, n2);
        assert_eq!(
            p1.variable.data().to_f32_vec().unwrap(),
            p2.variable.data().to_f32_vec().unwrap()
        );
    }

    std::fs::remove_file(path_str).ok();
}

#[test]
fn test_graph_checkpoint_hash_mismatch() {
    let g1 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("test_graph_ckpt_mismatch.fdl");
    let path_str = path.to_str().unwrap();

    g1.save_checkpoint(path_str).unwrap();

    // Different architecture
    let g2 = FlowBuilder::from(Linear::on_device(4, 16, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(16, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let result = g2.load_checkpoint(path_str);
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("architecture mismatch"));

    std::fs::remove_file(path_str).ok();
}

#[test]
fn test_save_checkpoint_emits_sidecar_when_source_config_set() {
    use crate::graph::checkpoint::sidecar_config_path;

    let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let payload = r#"{"model_type":"bert","hidden_size":768}"#;
    g.set_source_config(payload.to_string());

    let dir = std::env::temp_dir();
    let path = dir.join("test_sidecar_emit.fdl");
    let path_str = path.to_str().unwrap();

    g.save_checkpoint(path_str).unwrap();

    let sidecar = sidecar_config_path(path_str);
    let written = std::fs::read_to_string(&sidecar).unwrap();
    assert_eq!(written, payload);

    std::fs::remove_file(path_str).ok();
    std::fs::remove_file(sidecar).ok();
}

#[test]
fn test_save_checkpoint_no_sidecar_when_source_config_unset() {
    use crate::graph::checkpoint::sidecar_config_path;

    let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("test_sidecar_none.fdl");
    let path_str = path.to_str().unwrap();
    let sidecar = sidecar_config_path(path_str);

    // Pre-clean any leftover sidecar from a previous test run so the
    // assertion "no sidecar written" is meaningful.
    std::fs::remove_file(&sidecar).ok();

    g.save_checkpoint(path_str).unwrap();
    assert!(
        !sidecar.exists(),
        "sidecar must not be written when source_config is unset"
    );

    std::fs::remove_file(path_str).ok();
}

#[test]
fn test_sidecar_path_strips_fdl_and_gz() {
    use crate::graph::checkpoint::sidecar_config_path;

    assert_eq!(
        sidecar_config_path("/tmp/model.fdl"),
        std::path::PathBuf::from("/tmp/model.config.json"),
    );
    assert_eq!(
        sidecar_config_path("/tmp/model.fdl.gz"),
        std::path::PathBuf::from("/tmp/model.config.json"),
    );
    assert_eq!(
        sidecar_config_path("relative/v3.fdl"),
        std::path::PathBuf::from("relative/v3.config.json"),
    );
}

#[test]
fn test_clear_source_config_disables_sidecar() {
    use crate::graph::checkpoint::sidecar_config_path;

    let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    g.set_source_config("payload".to_string());
    assert_eq!(g.source_config().as_deref(), Some("payload"));
    g.clear_source_config();
    assert_eq!(g.source_config(), None);

    let dir = std::env::temp_dir();
    let path = dir.join("test_sidecar_cleared.fdl");
    let path_str = path.to_str().unwrap();
    let sidecar = sidecar_config_path(path_str);
    std::fs::remove_file(&sidecar).ok();

    g.save_checkpoint(path_str).unwrap();
    assert!(
        !sidecar.exists(),
        "cleared source_config must not emit sidecar"
    );

    std::fs::remove_file(path_str).ok();
}

#[test]
fn test_graph_checkpoint_gz() {
    let g = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let dir = std::env::temp_dir();
    let path = dir.join("test_graph_ckpt.fdl.gz");
    let path_str = path.to_str().unwrap();

    g.save_checkpoint(path_str).unwrap();

    let g2 = FlowBuilder::from(Linear::on_device(4, 8, crate::tensor::test_device()).unwrap())
        .through(Linear::on_device(8, 2, crate::tensor::test_device()).unwrap())
        .build()
        .unwrap();

    let report = g2.load_checkpoint(path_str).unwrap();
    assert_eq!(report.loaded.len(), 4);

    std::fs::remove_file(path_str).ok();
}

// --- collect_with reduction tests ---

#[test]
fn test_graph_set_scheduler_drives_optimizer_lr() {
    let (graph, x) = graph_with_optim(0.0); // start at 0 so we detect writes
    graph.set_scheduler(std::sync::Arc::new(LinearSched(0.1)));
    assert_eq!(graph.training_step(), 0);

    // Three step()s: scheduler queried at training_step before increment.
    for expected_step in 0..3 {
        // Forward + backward to populate gradients (step() needs them).
        let y = graph.forward(&x).unwrap();
        y.sum().unwrap().backward().unwrap();
        graph.step().unwrap();
        // After step(), training_step has advanced and the LR set BEFORE
        // optimizer.step() reflects the *previous* training_step value.
        let expected_lr = expected_step as f64 * 0.1;
        assert!(
            (current_optim_lr(&graph) - expected_lr).abs() < 1e-9,
            "after step {}: expected LR {expected_lr}, got {}",
            expected_step + 1,
            current_optim_lr(&graph)
        );
        assert_eq!(graph.training_step(), expected_step + 1);
    }
}

#[test]
fn test_graph_lr_scale_multiplies_scheduler_output() {
    let (graph, x) = graph_with_optim(0.0);
    graph.set_scheduler(std::sync::Arc::new(LinearSched(0.1)));
    graph.set_lr_scale(2.5);

    let y = graph.forward(&x).unwrap();
    y.sum().unwrap().backward().unwrap();
    graph.step().unwrap();
    // Step 0: scheduler returns 0.0 -> 0.0 * 2.5 = 0.0 (boring)
    assert!(current_optim_lr(&graph).abs() < 1e-9);

    let y = graph.forward(&x).unwrap();
    y.sum().unwrap().backward().unwrap();
    graph.step().unwrap();
    // Step 1: scheduler returns 0.1 -> 0.1 * 2.5 = 0.25
    assert!(
        (current_optim_lr(&graph) - 0.25).abs() < 1e-9,
        "expected LR 0.25 (sched 0.1 * scale 2.5), got {}",
        current_optim_lr(&graph)
    );
}

#[test]
fn test_graph_no_scheduler_leaves_lr_alone() {
    // Without a scheduler, step() must NOT touch the optimizer's LR.
    let (graph, x) = graph_with_optim(0.123);
    // Don't attach any scheduler.
    let y = graph.forward(&x).unwrap();
    y.sum().unwrap().backward().unwrap();
    graph.step().unwrap();
    assert!(
        (current_optim_lr(&graph) - 0.123).abs() < 1e-9,
        "no scheduler attached: LR must be untouched, got {}",
        current_optim_lr(&graph)
    );
    // training_step still increments (it's a per-step counter, scheduler-independent).
    assert_eq!(graph.training_step(), 1);
}

/// The Module↔Graph downcast seam: `GraphExt::as_graph` must recover
/// the graph behind `dyn Module` (identity — pointer-equal), forward
/// through `Box<dyn Module>`, and reject plain leaf modules. This is
/// the contract that replaced the trait-level `Module::as_graph` hook
/// when `nn` was de-cycled from `graph`.
#[test]
fn graph_ext_downcasts_graph_and_rejects_leaves() {
    use crate::graph::GraphExt;

    let graph = FlowBuilder::from(Doubler).build().unwrap();
    let as_dyn: &dyn Module = &graph;
    let recovered = as_dyn.as_graph().expect("Graph must downcast to itself");
    assert!(std::ptr::eq(recovered, &graph), "identity, not a copy");

    let boxed: Box<dyn Module> = Box::new(FlowBuilder::from(Doubler).build().unwrap());
    assert!(
        boxed.as_graph().is_some(),
        "Box<dyn Module> forwards as_any"
    );

    let leaf: &dyn Module = &Doubler;
    assert!(leaf.as_graph().is_none(), "leaf modules present nothing");
}
