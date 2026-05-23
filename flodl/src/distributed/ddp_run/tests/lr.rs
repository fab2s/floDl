//! LR scheduling on the worker, cross-mode LR parity tests, and the
//! LR-aware meta-controller integration tests.

use super::*;

// ---------------------------------------------------------------------------
// LR scheduling on the worker
// ---------------------------------------------------------------------------
//
// These tests guard the per-batch LR pipeline: scheduler.lr(step) * lr_scale
// must reach the optimizer on every batch. The original bugs (2026-04-13)
// were that scale_lr was silently overwritten when a scheduler was attached
// (so DDP linear scaling never took effect) and that the scheduler step
// counter could be inflated by NCCL ack messages (so MultiStepLR fired
// ~6 epochs early on heterogeneous DDP).

/// Trivial constant-LR scheduler used to assert `worker.lr_scale` is applied
/// multiplicatively on top of scheduler output.
struct ConstLr(f64);
impl crate::nn::Scheduler for ConstLr {
    fn lr(&self, _step: usize) -> f64 { self.0 }
}

/// Linearly increasing LR (lr = step * slope), so the test can also verify
/// that the scheduler is queried with the correct training step.
struct LinearLr { slope: f64 }
impl crate::nn::Scheduler for LinearLr {
    fn lr(&self, step: usize) -> f64 { step as f64 * self.slope }
}

#[test]
fn test_worker_scheduler_drives_optimizer_lr() {
    let (mut worker, _ch) = make_test_worker();
    worker.set_lr(0.0); // start at 0 so we can detect the scheduler writing in

    worker.set_scheduler(Arc::new(ConstLr(0.05)));

    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch, &mse_train).unwrap();

    // Scheduler returned 0.05; with lr_scale=1.0 (default) optimizer sees 0.05.
    assert!((worker.current_lr() - 0.05).abs() < 1e-9,
        "expected optimizer LR 0.05, got {}", worker.current_lr());
}

#[test]
fn test_worker_lr_scale_multiplies_scheduler_output() {
    // The bug this guards: orchestrator used to call worker.scale_lr(2.0) at
    // startup, but the scheduler's per-batch set_lr immediately overwrote
    // it -- so DDP linear scaling never reached the optimizer when a
    // scheduler was attached. Fix: orchestrator now calls set_lr_scale and
    // train_step does set_lr(sched.lr(step) * lr_scale).
    let (mut worker, _ch) = make_test_worker();
    worker.set_scheduler(Arc::new(ConstLr(0.05)));
    worker.set_lr_scale(2.0);

    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];
    worker.train_step(&batch, &mse_train).unwrap();

    // 0.05 * 2.0 = 0.10
    assert!((worker.current_lr() - 0.10).abs() < 1e-9,
        "expected optimizer LR 0.10 (sched 0.05 * scale 2.0), got {}",
        worker.current_lr());
}

#[test]
fn test_worker_scheduler_step_advances_with_global_progress() {
    // train_step computes set_lr(sched.lr(global_step + steps_since_avg)).
    // After 3 batches (no sync), step argument should be 3.
    let (mut worker, _ch) = make_test_worker();
    worker.set_scheduler(Arc::new(LinearLr { slope: 0.01 }));

    let opts = test_opts();
    let batch = vec![
        Tensor::randn(&[4, 4], opts).unwrap(),
        Tensor::randn(&[4, 2], opts).unwrap(),
    ];

    worker.train_step(&batch, &mse_train).unwrap();
    // Before batch 1, scheduler was queried at step 0 -> lr = 0.0.
    assert!((worker.current_lr() - 0.0).abs() < 1e-9);

    worker.train_step(&batch, &mse_train).unwrap();
    // Before batch 2, scheduler queried at step 1 -> lr = 0.01.
    assert!((worker.current_lr() - 0.01).abs() < 1e-9,
        "step 1: got {}", worker.current_lr());

    worker.train_step(&batch, &mse_train).unwrap();
    // Before batch 3, scheduler queried at step 2 -> lr = 0.02.
    assert!((worker.current_lr() - 0.02).abs() < 1e-9,
        "step 2: got {}", worker.current_lr());
}

// ---------------------------------------------------------------------------
// Cross-mode LR parity
// ---------------------------------------------------------------------------
//
// The framework promises: "if you give me a scheduler, I will honor it the
// same way no matter which training mode you use." Without this guarantee,
// switching between solo and DDP silently changes hyperparameter behavior --
// which is exactly the failure mode that hid Bugs 1-3 for so long.
//
// This test is the contract test for that promise. It runs the same
// MultiStepLR over the same number of batches in three independent paths
// and asserts the recorded LR sequences are identical:
//
//   1. **Manual**:  `optimizer.set_lr(sched.lr(step))` per batch (the
//      reference implementation a user would write themselves).
//   2. **GpuWorker**: train_step() with set_scheduler attached
//      (the path used by Trainer::builder, both in DDP and single-GPU fallback).
//   3. **Graph::step()**: the path used by Trainer::setup_with (sync mode).
//
// The first time we ran this we had two bugs (graph mode never updated LR;
// worker scheduler interaction with lr_scale was broken) and this single
// test would have caught both.

/// Records every LR value the scheduler returned, so the test can compare
/// the exact query sequence each mode produced.
struct RecordingSched {
    inner: crate::nn::MultiStepLR,
    queries: std::sync::Mutex<Vec<(usize, f64)>>,
}
impl RecordingSched {
    fn new(base_lr: f64, milestones: &[usize], gamma: f64) -> Self {
        Self {
            inner: crate::nn::MultiStepLR::new(base_lr, milestones, gamma),
            queries: std::sync::Mutex::new(Vec::new()),
        }
    }
}
impl crate::nn::Scheduler for RecordingSched {
    fn lr(&self, step: usize) -> f64 {
        let lr = self.inner.lr(step);
        self.queries.lock().unwrap().push((step, lr));
        lr
    }
}

#[test]
fn test_cross_mode_lr_parity_solo_vs_worker_vs_graph() {
    use crate::graph::FlowBuilder;
    use crate::nn::{Module, Optimizer, SGD};

    let dev = test_device();
    let opts = test_opts();
    let n_steps: usize = 12;
    // Milestones at 4 and 8 produce 3 LR plateaus across our 12 steps so
    // every drop is observed; gamma 0.1 makes the steps obvious.
    let base_lr = 0.1f64;
    let milestones = vec![4usize, 8];
    let gamma = 0.1f64;

    // ---------- Path 1: manual optimizer.set_lr per batch (reference). ----------
    let manual_lrs: Vec<f64> = {
        let model = Linear::on_device(4, 2, dev).unwrap();
        let mut opt = SGD::new(&model.parameters(), 0.0, 0.0);
        let sched = crate::nn::MultiStepLR::new(base_lr, &milestones, gamma);
        let batch = [
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            opt.set_lr(sched.lr(step));
            // Need a backward pass before opt.step() so SGD has gradients.
            let v = Variable::new(batch[0].clone(), false);
            let t = Variable::new(batch[1].clone(), false);
            let pred = model.forward(&v).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad();
            lrs.push(opt.lr());
        }
        lrs
    };

    // ---------- Path 2: GpuWorker with set_scheduler (Trainer::builder path). ----------
    let worker_lrs: Vec<f64> = {
        let (mut worker, _ch) = make_test_worker();
        worker.set_scheduler(Arc::new(RecordingSched::new(base_lr, &milestones, gamma)));
        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            worker.train_step(&batch, &mse_train).unwrap();
            lrs.push(worker.current_lr());
        }
        lrs
    };

    // ---------- Path 3: Graph::step() with set_scheduler (Trainer::setup_with path). ----------
    let graph_lrs: Vec<f64> = {
        let graph = FlowBuilder::from(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();
        graph.set_optimizer(|p| SGD::new(p, 0.0, 0.0));
        graph.set_scheduler(Arc::new(RecordingSched::new(base_lr, &milestones, gamma)));
        let x = Variable::new(Tensor::randn(&[4, 4], opts).unwrap(), false);
        let t = Variable::new(Tensor::randn(&[4, 2], opts).unwrap(), false);
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let pred = graph.forward(&x).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            graph.step().unwrap();
            let lr = graph.optimizer.borrow().as_ref().map(|o| o.lr()).unwrap();
            lrs.push(lr);
        }
        lrs
    };

    // The three paths must produce identical LR trajectories.
    assert_eq!(manual_lrs.len(), n_steps);
    assert_eq!(worker_lrs.len(), n_steps);
    assert_eq!(graph_lrs.len(), n_steps);
    for step in 0..n_steps {
        assert!((manual_lrs[step] - worker_lrs[step]).abs() < 1e-9,
            "step {step}: solo={} vs worker={}", manual_lrs[step], worker_lrs[step]);
        assert!((manual_lrs[step] - graph_lrs[step]).abs() < 1e-9,
            "step {step}: solo={} vs graph={}", manual_lrs[step], graph_lrs[step]);
    }

    // Sanity: the recorded trajectory should show the two MultiStepLR drops
    // (0.1 -> 0.01 at step 4, 0.01 -> 0.001 at step 8). If this fails, the
    // test scheduler isn't doing its job and the parity check above is
    // vacuous.
    let mut transitions = 0;
    for w in manual_lrs.windows(2) {
        if (w[0] - w[1]).abs() > 1e-9 { transitions += 1; }
    }
    assert_eq!(transitions, 2,
        "expected 2 LR drops over 12 steps with milestones [4, 8]; got {transitions}. \
         trajectory: {manual_lrs:?}");
}

#[test]
fn test_cross_mode_lr_parity_with_lr_scale() {
    // Same parity guarantee, but with lr_scale != 1.0. Worker and Graph must
    // both apply the scale multiplicatively to scheduler output.
    use crate::graph::FlowBuilder;
    use crate::nn::{Module, Optimizer, SGD};

    let dev = test_device();
    let opts = test_opts();
    let n_steps: usize = 8;
    let scale = 2.5;

    // Reference: manual with scale baked into base_lr.
    let manual_lrs: Vec<f64> = {
        let model = Linear::on_device(4, 2, dev).unwrap();
        let mut opt = SGD::new(&model.parameters(), 0.0, 0.0);
        let sched = crate::nn::MultiStepLR::new(0.1, &[4], 0.1);
        let batch = [
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            opt.set_lr(sched.lr(step) * scale);
            let v = Variable::new(batch[0].clone(), false);
            let t = Variable::new(batch[1].clone(), false);
            let pred = model.forward(&v).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad();
            lrs.push(opt.lr());
        }
        lrs
    };

    // GpuWorker: set_lr_scale + set_scheduler.
    let worker_lrs: Vec<f64> = {
        let (mut worker, _ch) = make_test_worker();
        worker.set_scheduler(Arc::new(crate::nn::MultiStepLR::new(0.1, &[4], 0.1)));
        worker.set_lr_scale(scale);
        let batch = vec![
            Tensor::randn(&[4, 4], opts).unwrap(),
            Tensor::randn(&[4, 2], opts).unwrap(),
        ];
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            worker.train_step(&batch, &mse_train).unwrap();
            lrs.push(worker.current_lr());
        }
        lrs
    };

    // Graph: set_lr_scale + set_scheduler.
    let graph_lrs: Vec<f64> = {
        let graph = FlowBuilder::from(Linear::on_device(4, 2, dev).unwrap())
            .build()
            .unwrap();
        graph.set_optimizer(|p| SGD::new(p, 0.0, 0.0));
        graph.set_scheduler(Arc::new(crate::nn::MultiStepLR::new(0.1, &[4], 0.1)));
        graph.set_lr_scale(scale);
        let x = Variable::new(Tensor::randn(&[4, 4], opts).unwrap(), false);
        let t = Variable::new(Tensor::randn(&[4, 2], opts).unwrap(), false);
        let mut lrs = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let pred = graph.forward(&x).unwrap();
            let loss = pred.sub(&t).unwrap();
            loss.mul(&loss).unwrap().mean().unwrap().backward().unwrap();
            graph.step().unwrap();
            let lr = graph.optimizer.borrow().as_ref().map(|o| o.lr()).unwrap();
            lrs.push(lr);
        }
        lrs
    };

    for step in 0..n_steps {
        assert!((manual_lrs[step] - worker_lrs[step]).abs() < 1e-9,
            "step {step}: solo*scale={} vs worker={}", manual_lrs[step], worker_lrs[step]);
        assert!((manual_lrs[step] - graph_lrs[step]).abs() < 1e-9,
            "step {step}: solo*scale={} vs graph={}", manual_lrs[step], graph_lrs[step]);
    }
}

// ---------------------------------------------------------------------------
// LR-aware meta-controller integration
// ---------------------------------------------------------------------------

/// Build a coordinator harness with the LR-aware meta-controller enabled.
fn make_coord_harness_meta(n: usize) -> CoordTestHarness {
    let (timing_tx, timing_rx) = mpsc::channel();
    let (metrics_tx, metrics_rx) = mpsc::channel();
    let (param_tx, param_rx) = mpsc::channel();

    let mut control_txs = Vec::new();
    let mut control_rxs = Vec::new();
    let mut final_param_rxs = Vec::new();
    for _ in 0..n {
        let (tx, rx) = mpsc::channel();
        control_txs.push(tx);
        control_rxs.push(rx);
        let (_ftx, frx) = mpsc::channel();
        final_param_rxs.push(frx);
    }

    let el_che = crate::distributed::ddp::ElChe::new(n, 10);
    let coord = Coordinator::builder(
        timing_rx, metrics_rx, param_rx,
        final_param_rxs, control_txs,
        ApplyPolicy::Async, AverageBackend::Cpu,
        n, 10000, el_che,
    )
    .meta_controller(true)
    .build();

    CoordTestHarness { coord, timing_tx, metrics_tx, param_tx, control_rxs }
}

/// Drive ElChe past Probe/Warmup into Stable so the meta-controller will
/// act on observations (Probe is always silent, Warmup requires K=5
/// sustained for convergence pattern).
fn calibrate_to_stable(coord: &mut Coordinator) {
    for _ in 0..6 {
        coord.el_che.report_timing(&[10.0, 20.0], &[10, 10], 1.0);
    }
    assert!(coord.el_che.phase() >= crate::distributed::Phase::Stable);
}

#[test]
fn test_meta_controller_dispatches_nudge_on_lr_cliff() {
    let mut h = make_coord_harness_meta(2);
    calibrate_to_stable(&mut h.coord);

    // Establish baseline LR via the LrUpdate channel + drain.
    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.1 }).unwrap();
    h.coord.drain_timing();

    // First observation: builds the LR window. anchor_window also starts here.
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);
    let anchor_after_warmup = h.coord.el_che.anchor();

    // Sharp LR drop (10x) on the next cycle: cliff watcher should fire.
    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.01 }).unwrap();
    h.coord.drain_timing();
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);

    let anchor_after_cliff = h.coord.el_che.anchor();
    assert!(
        anchor_after_cliff < anchor_after_warmup,
        "meta-controller should nudge anchor down on sharp LR drop: \
         {anchor_after_warmup} -> {anchor_after_cliff}",
    );
}

#[test]
fn test_meta_controller_silent_on_smooth_decay() {
    let mut h = make_coord_harness_meta(2);
    calibrate_to_stable(&mut h.coord);

    let anchor_initial = h.coord.el_che.anchor();
    let mut lr = 0.1;
    for _ in 0..10 {
        h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr }).unwrap();
        h.coord.drain_timing();
        h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);
        lr *= 0.98; // 2% per cycle, well under 30% cliff threshold
    }

    let anchor_final = h.coord.el_che.anchor();
    assert_eq!(
        anchor_final, anchor_initial,
        "smooth decay should not trigger meta-controller; anchor unchanged \
         from {anchor_initial}, got {anchor_final}",
    );
}

#[test]
fn test_meta_controller_disabled_no_dispatch() {
    // Default harness: meta_controller flag off. Even with LR cliff input,
    // observe_meta is a no-op.
    let mut h = make_coord_harness(2, ApplyPolicy::Async, AverageBackend::Cpu);
    calibrate_to_stable(&mut h.coord);

    let anchor_initial = h.coord.el_che.anchor();

    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.1 }).unwrap();
    h.coord.drain_timing();
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);

    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.01 }).unwrap();
    h.coord.drain_timing();
    h.coord.observe_meta(super::convergence::ConvergenceAction::Stable);

    assert_eq!(
        h.coord.el_che.anchor(),
        anchor_initial,
        "meta-controller off should not change anchor",
    );
}

#[test]
fn test_meta_controller_dispatches_on_sustained_convergence_pattern() {
    let mut h = make_coord_harness_meta(2);
    calibrate_to_stable(&mut h.coord);

    h.timing_tx.send(TimingMsg::LrUpdate { rank: 0, lr: 0.1 }).unwrap();
    h.coord.drain_timing();

    // Stable phase requires K=3 sustained NudgeDown / SuppressGrowth verdicts.
    let nudge = super::convergence::ConvergenceAction::NudgeDown { factor: 0.5 };
    h.coord.observe_meta(nudge);
    h.coord.observe_meta(nudge);
    let anchor_after_two = h.coord.el_che.anchor();

    h.coord.observe_meta(nudge);
    let anchor_after_three = h.coord.el_che.anchor();

    assert!(
        anchor_after_three < anchor_after_two,
        "convergence pattern (K=3 at Stable) should fire on third sustained verdict: \
         {anchor_after_two} -> {anchor_after_three}",
    );
}
