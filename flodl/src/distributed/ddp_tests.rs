    use super::*;
    use crate::tensor::{
        cuda_device_count, cuda_synchronize, test_device, DType, TensorOptions,
    };

    fn require_multi_gpu() -> bool {
        if !test_device().is_cuda() || cuda_device_count() < 2 {
            return false;
        }
        for i in 0..2 {
            let opts = TensorOptions {
                dtype: DType::Float32,
                device: Device::CUDA(i),
            };
            if Tensor::zeros(&[1], opts).is_err() {
                eprintln!(
                    "Device CUDA({i}) cannot run compute kernels, skipping multi-GPU test"
                );
                return false;
            }
        }
        true
    }

    // -- Cross-device autograd verification ---------------------------------

    #[test]
    fn test_cross_device_autograd_gradient_flow() {
        if !require_multi_gpu() {
            return;
        }

        let opts0 = TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(0),
        };
        let opts1 = TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(1),
        };

        // Parameters on two different devices
        let w0 = Variable::new(Tensor::ones(&[4, 3], opts0).unwrap(), true);
        let w1 = Variable::new(Tensor::ones(&[4, 3], opts1).unwrap(), true);

        // Input on device 0 (no requires_grad, like training data)
        let input = Variable::new(
            Tensor::ones(&[4, 4], opts0).unwrap(),
            false,
        );

        // Chunk along batch dim: 2 shards of size 2
        let chunks = input.chunk(2, 0).unwrap();
        assert_eq!(chunks.len(), 2);

        // Shard 0: forward on device 0
        let out0 = chunks[0].matmul(&w0).unwrap(); // [2, 3] on dev0

        // Shard 1: move to device 1, forward there, move output back to device 0
        let shard1_dev1 = chunks[1].to_device(Device::CUDA(1)).unwrap();
        let out1_dev1 = shard1_dev1.matmul(&w1).unwrap(); // [2, 3] on dev1
        let out1_dev0 = out1_dev1.to_device(Device::CUDA(0)).unwrap(); // [2, 3] on dev0

        // Gather: cat outputs on device 0
        let gathered = Variable::cat_many(&[&out0, &out1_dev0], 0).unwrap(); // [4, 3]

        // Compute scalar loss
        let loss = gathered.sum().unwrap();

        // Backward
        loss.backward().unwrap();

        // Verify: both parameters received gradients on their own devices
        let grad0 = w0.grad();
        let grad1 = w1.grad();
        assert!(
            grad0.is_some(),
            "w0 on device 0 should have gradient after backward"
        );
        assert!(
            grad1.is_some(),
            "w1 on device 1 should have gradient after backward"
        );

        // Verify gradients are on the correct devices
        let g0 = grad0.unwrap();
        let g1 = grad1.unwrap();
        assert_eq!(g0.device(), Device::CUDA(0), "w0 gradient should be on device 0");
        assert_eq!(g1.device(), Device::CUDA(1), "w1 gradient should be on device 1");

        // Verify gradient values are non-zero
        let g0_sum = g0.sum().unwrap().item().unwrap();
        let g1_sum = g1.sum().unwrap().item().unwrap();
        assert!(
            g0_sum.abs() > 1e-6,
            "w0 gradient should be non-zero, got {g0_sum}"
        );
        assert!(
            g1_sum.abs() > 1e-6,
            "w1 gradient should be non-zero, got {g1_sum}"
        );

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    #[test]
    fn test_cross_device_autograd_values() {
        // Verify that cross-device backward produces the SAME gradients
        // as single-device backward (correctness check).
        if !require_multi_gpu() {
            return;
        }

        // Use deterministic values
        let w_data = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            Device::CUDA(0),
        )
        .unwrap();

        // Single-device reference: forward all on device 0
        let w_ref = Variable::new(w_data.clone(), true);
        let x = Tensor::from_f32(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            &[4, 4],
            Device::CUDA(0),
        )
        .unwrap();
        let x_var = Variable::new(x.clone(), false);
        let out_ref = x_var.matmul(&w_ref).unwrap();
        let loss_ref = out_ref.sum().unwrap();
        loss_ref.backward().unwrap();
        let grad_ref = w_ref.grad().unwrap();
        let grad_ref_vals = grad_ref.to_f32_vec().unwrap();

        // Cross-device: split batch across 2 devices.
        // Create w0 and w1 from fresh tensors (not clones of w_data,
        // which was tainted by set_requires_grad through w_ref's shallow clone).
        let w0 = Variable::new(
            Tensor::from_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[4, 2],
                Device::CUDA(0),
            )
            .unwrap(),
            true,
        );
        let w1 = Variable::new(
            Tensor::from_f32(
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                &[4, 2],
                Device::CUDA(1),
            )
            .unwrap(),
            true,
        );
        let x_var2 = Variable::new(x, false);

        let chunks = x_var2.chunk(2, 0).unwrap();

        let out0 = chunks[0].matmul(&w0).unwrap();
        let shard1 = chunks[1].to_device(Device::CUDA(1)).unwrap();
        let out1_dev1 = shard1.matmul(&w1).unwrap();
        let out1_dev0 = out1_dev1.to_device(Device::CUDA(0)).unwrap();
        let gathered = Variable::cat_many(&[&out0, &out1_dev0], 0).unwrap();
        let loss = gathered.sum().unwrap();
        loss.backward().unwrap();

        // Sum of cross-device gradients should equal single-device gradient
        let g0 = w0.grad().unwrap().to_f32_vec().unwrap();
        let g1 = w1.grad().unwrap().to_f32_vec().unwrap();

        for i in 0..g0.len() {
            let cross_sum = g0[i] + g1[i];
            let diff = (cross_sum - grad_ref_vals[i]).abs();
            assert!(
                diff < 1e-5,
                "gradient mismatch at index {i}: cross-device sum {cross_sum} vs reference {}",
                grad_ref_vals[i]
            );
        }

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    // -- Graph integration tests (CPU, single-GPU fallback) -----------------

    #[test]
    fn test_graph_set_optimizer_and_step() {
        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};

        let model = FlowBuilder::from(Linear::new(4, 8).unwrap())
            .through(ReLU::new())
            .through(Linear::new(8, 2).unwrap())
            .build()
            .unwrap();

        model.set_optimizer(|p| Adam::new(p, 0.01));
        model.set_training(true);

        // Snapshot initial params
        let params_before: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        // One training step
        let x = Variable::new(
            Tensor::randn(&[4, 4], Default::default()).unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::randn(&[4, 2], Default::default()).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        let loss = mse_loss(&out, &target).unwrap();
        loss.backward().unwrap();
        model.step().unwrap();

        // Params should have changed
        let params_after: Vec<f32> = model
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
            .collect();

        let changed = params_before
            .iter()
            .zip(&params_after)
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "parameters should change after step()");
    }

    #[test]
    fn test_graph_step_without_optimizer() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        // step() without set_optimizer() should be a no-op, not a crash
        let result = model.step();
        assert!(result.is_ok());
    }

    #[test]
    fn test_graph_set_lr() {
        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear};

        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        model.set_optimizer(|p| Adam::new(p, 0.01));
        // Should not panic
        model.set_lr(0.001);
    }

    // -- El Che unit tests (CPU, no NCCL needed) ----------------------------

    #[test]
    fn test_cadence_initial_equal() {
        let c = ElChe::new(2, 10);
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 10);
        assert_eq!(c.total_batches(), 20);
        assert_eq!(c.anchor(), 10);
        assert!(!c.is_calibrated());
    }

    #[test]
    fn test_cadence_initial_three_devices() {
        let c = ElChe::new(3, 15);
        assert_eq!(c.batches(0), 15);
        assert_eq!(c.batches(1), 15);
        assert_eq!(c.batches(2), 15);
        assert_eq!(c.total_batches(), 45);
    }

    #[test]
    fn test_cadence_ratio_discovery_2x() {
        // Device 0 is 2x faster than device 1.
        // Equal counts (10:10), device 0 finishes in 500ms, device 1 in 1000ms.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50); // high target to avoid anchor auto-tune
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 10.0);

        assert!(c.is_calibrated());
        // Slow device (rank 1) keeps anchor=10, fast device (rank 0) gets ~20.
        assert_eq!(c.batches(1), 10);
        assert_eq!(c.batches(0), 20);
    }

    #[test]
    fn test_cadence_ratio_discovery_fbrl_like() {
        // Simulates RTX 5060 Ti vs GTX 1060 (~2.3:1 speed ratio).
        // Anchor=10 on slow device, equal initial counts.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50); // no auto-tune

        // Both ran 10 batches; fast took 730ms (73ms/batch), slow took 1640ms (164ms/batch).
        let bc = c.batch_counts().to_vec(); c.report_timing(&[730.0, 1640.0], &bc, 50.0);

        assert!(c.is_calibrated());
        assert_eq!(c.batches(1), 10); // slow device: anchor
        // Fast device: 164/73 * 10 ≈ 22.5, rounds to 22 or 23
        let fast = c.batches(0);
        assert!(
            (22..=23).contains(&fast),
            "expected ~22-23, got {fast}"
        );
    }

    #[test]
    fn test_cadence_anchor_auto_tune() {
        // High per-window fixed overhead should trigger anchor growth.
        // No fill is staged (no coordinator), so the signal is reduce-only:
        // overhead = sync / (anchor·marginal + sync). With marginal 100ms,
        // anchor 10 => window_compute 1000, sync 500 => 500/1500 = 0.33 > 0.10.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.10);

        // Auto-tune is gated to Phase::Stable+ to prevent warmup over-reaction.
        // Prime with five low-overhead reports of equal-speed timings to reach
        // Stable, then issue the high-overhead trigger.
        for _ in 0..5 {
            let bc = c.batch_counts().to_vec();
            c.report_timing(&[1000.0, 1000.0], &bc, 5.0);
        }
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[1000.0, 1000.0], &bc, 500.0);

        // window-pressure proposes; commit to apply (Stable verdict).
        c.commit_proposed_anchor();

        // scale = min(0.33/0.10, GROWTH_STEP_CAP=2.0) = 2.0
        // new anchor = ceil(10 * 2) = 20 (capped per cycle; multi-cycle climb)
        assert_eq!(c.anchor(), 20);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 20);
    }

    #[test]
    fn test_cadence_anchor_auto_tune_with_speed_ratio() {
        // Heterogeneous: fast device 2x, high sync overhead.
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.10);

        // Prime to Stable phase. Pass fixed bc=[10,10] each call so the
        // synthetic wall_ms keeps a stable per-batch ratio across reports
        // (in production wall_ms would scale with n; in the test it does
        // not, so we keep n fixed instead).
        for _ in 0..5 {
            c.report_timing(&[500.0, 1000.0], &[10, 10], 5.0);
        }
        c.report_timing(&[500.0, 1000.0], &[10, 10], 400.0);
        c.commit_proposed_anchor();

        // anchor rank = slow rank 1 (100ms/batch). window_compute = 10·100 = 1000,
        // sync 400 => overhead 400/1400 = 0.286. scale = min(2.86, 2.0) = 2.0.
        // new anchor = ceil(10 * 2) = 20.
        assert_eq!(c.anchor(), 20);
        assert_eq!(c.batches(1), 20); // slow device
        // fast device: 100ms/batch vs 50ms/batch => 2x ratio => 40
        assert_eq!(c.batches(0), 40);
    }

    #[test]
    fn test_cap_binding_suppresses_anchor_growth() {
        // Anchor wind-up guard: when the window cap is binding (counts
        // scaled down to fit the epoch), measured overhead stays above
        // target forever — growth proposals must be suppressed, or the
        // anchor ratchets toward max_anchor while delivered counts stay
        // pinned and every anchor-derived quantity lies.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.10);
        c.set_max_total_batches(30); // binding almost immediately

        for _ in 0..5 {
            c.report_timing(&[500.0, 1000.0], &[10, 10], 5.0);
        }
        // Force the cap to bind once.
        c.report_timing(&[500.0, 1000.0], &[10, 10], 400.0);
        c.commit_proposed_anchor();
        let anchor_after_first = c.anchor();

        // Keep reporting pathological overhead: with the cap binding, no
        // further growth proposals may land.
        for _ in 0..10 {
            c.report_timing(&[500.0, 1000.0], &[10, 10], 400.0);
            c.commit_proposed_anchor();
        }
        assert!(
            c.anchor() <= anchor_after_first,
            "anchor ratcheted under a binding window cap: {} -> {}",
            anchor_after_first,
            c.anchor(),
        );
        let total = c.batches(0) + c.batches(1);
        assert!(total <= 30, "cap still enforced: total={total}");
    }

    #[test]
    fn test_speed_ratio_clamped_against_degenerate_sample() {
        // One legitimate-but-tiny reading (sub-ms wall over many batches)
        // must not blow the schedule up by a 1e4x ratio on paths without
        // max_batch_diff / max_total_batches.
        let mut c = ElChe::new(2, 10);
        for _ in 0..6 {
            // rank 0: absurdly fast reading; rank 1: 100 ms/batch.
            c.report_timing(&[0.1, 1000.0], &[10, 10], 1.0);
        }
        // ratio would be 10_000x unclamped; with the 64x clamp the fast
        // rank gets at most anchor * 64 batches.
        assert!(
            c.batches(0) <= 10 * 64,
            "ratio clamp failed: fast rank got {} batches",
            c.batches(0),
        );
    }

    #[test]
    fn test_warmup_unsticks_when_pinned_anchor_never_reports() {
        // A pinned anchor rank that never produces a valid reading must
        // not freeze the controller in Warmup forever: after a full trust
        // window of misses, election falls back to ranks with data.
        let mut c = ElChe::new(2, 10).with_initial_anchor(1);
        for _ in 0..10 {
            // rank 1 (the pinned anchor) reports nothing valid; rank 0
            // reports steadily.
            c.report_timing(&[100.0, 0.0], &[10, 0], 1.0);
        }
        assert!(
            c.is_calibrated(),
            "controller stayed un-calibrated: pinned dead anchor froze Warmup",
        );
    }

    #[test]
    fn test_nudge_anchor_down_ignores_nan_factor() {
        let mut c = ElChe::new(2, 10);
        for _ in 0..6 {
            c.report_timing(&[100.0, 100.0], &[10, 10], 1.0);
        }
        let before = c.anchor();
        c.nudge_anchor_down(f64::NAN);
        assert_eq!(c.anchor(), before, "NaN factor must be a no-op");
        c.nudge_anchor_down(0.5);
        assert!(c.anchor() < before, "finite factor still nudges");
    }

    #[test]
    fn test_cadence_window_capped_to_max_total() {
        // Window cap (set by the cluster coordinator to the epoch's batch
        // count): the overhead auto-tune may grow the schedule to amortize
        // an expensive sync, but `recompute_batch_counts` must scale the
        // per-rank counts down proportionally so their sum never exceeds
        // the cap — a reduce window must fit within one epoch. Mirrors
        // `test_cadence_anchor_auto_tune_with_speed_ratio` (which grows to
        // [80, 40], sum 120) but with the total capped at 60.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.10);
        c.set_max_total_batches(40);

        for _ in 0..5 {
            c.report_timing(&[500.0, 1000.0], &[10, 10], 5.0);
        }
        c.report_timing(&[500.0, 1000.0], &[10, 10], 400.0);
        c.commit_proposed_anchor();

        // Uncapped (×2 grow) this is [40, 20] (sum 60). Capped to 40 and
        // scaled proportionally: ~[26, 13] (sum <= 40, ~2x ratio preserved).
        let total = c.batches(0) + c.batches(1);
        assert!(total <= 40, "window capped to max_total: total={total} (<= 40)");
        assert!(
            c.batches(0) > c.batches(1),
            "speed ratio preserved after cap: fast={} slow={}",
            c.batches(0),
            c.batches(1),
        );
    }

    #[test]
    fn test_cadence_anchor_capped_at_max() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.01)
            .with_max_anchor(15);

        // Prime to Stable phase before triggering auto-tune.
        for _ in 0..5 {
            let bc = c.batch_counts().to_vec();
            c.report_timing(&[100.0, 100.0], &bc, 0.5);
        }
        // Extreme overhead: sync dominates.
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[100.0, 100.0], &bc, 500.0);
        c.commit_proposed_anchor();

        // ×2 grow wants ceil(10*2)=20, but max_anchor clamps it to 15.
        assert_eq!(c.anchor(), 15);
        assert_eq!(c.batches(0), 15);
    }

    #[test]
    fn test_cadence_stable_when_overhead_low() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.10);

        // sync=5ms on 1000ms compute => 0.5% overhead, well below 10%.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[1000.0, 1000.0], &bc, 5.0);

        assert_eq!(c.anchor(), 10); // no change
    }

    #[test]
    fn test_overhead_proposal_committed_on_stable_verdict() {
        // High-overhead trigger should propose a grow; commit applies it.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.10);
        for _ in 0..5 {
            c.report_timing(&[1000.0, 1000.0], &[10, 10], 5.0);
        }
        // overhead = 500/1500 = 0.33 > 0.10; scale = min(3.3, 2.0) = 2.0
        c.report_timing(&[1000.0, 1000.0], &[10, 10], 500.0);
        // Before commit, anchor unchanged.
        assert_eq!(c.anchor(), 10, "report_timing must not mutate anchor");
        c.commit_proposed_anchor();
        assert_eq!(c.anchor(), 20, "commit applies the ×2-capped grow");
    }

    #[test]
    fn test_overhead_grow_vetoed_on_suppress_growth() {
        // Grow proposal + SuppressGrowth verdict → anchor stays put.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.10);
        for _ in 0..5 {
            c.report_timing(&[1000.0, 1000.0], &[10, 10], 5.0);
        }
        c.report_timing(&[1000.0, 1000.0], &[10, 10], 500.0);
        c.veto_proposed_growth();
        assert_eq!(c.anchor(), 10, "SuppressGrowth vetoes the grow proposal");
    }

    #[test]
    fn test_growth_latched_off_after_suppress_growth() {
        // SuppressGrowth latches growth OFF; it re-arms only after
        // GROWTH_REARM_STABLE (5) consecutive Stable verdicts — the margin
        // to the convergence cliff. The controller must not re-attempt
        // growth every cycle while latched.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.10);
        for _ in 0..5 {
            c.report_timing(&[1000.0, 1000.0], &[10, 10], 5.0);
        }
        // Grow proposed, then SuppressGrowth: drop it AND latch growth off.
        c.report_timing(&[1000.0, 1000.0], &[10, 10], 500.0);
        c.veto_proposed_growth();
        assert_eq!(c.anchor(), 10, "SuppressGrowth vetoes the grow");
        assert!(!c.growth_enabled(), "growth latched off");

        // While latched, a high-overhead report proposes nothing → no grow
        // even on a Stable commit. Five Stable commits re-arm the latch.
        for _ in 0..5 {
            c.report_timing(&[1000.0, 1000.0], &[10, 10], 500.0);
            c.commit_proposed_anchor();
            assert_eq!(c.anchor(), 10, "no growth while latched off / re-arming");
        }
        assert!(c.growth_enabled(), "5 consecutive Stable verdicts re-arm growth");

        // A fresh high-overhead report now grows again.
        c.report_timing(&[1000.0, 1000.0], &[10, 10], 500.0);
        c.commit_proposed_anchor();
        assert!(c.anchor() > 10, "growth resumes after re-arm");
    }

    #[test]
    fn test_overhead_proposal_discarded_on_nudge_down() {
        // Grow proposal + NudgeDown verdict → proposal dropped, nudge
        // operates on the current (pre-proposal) anchor.
        let mut c = ElChe::new(2, 20).with_overhead_target(0.10);
        for _ in 0..5 {
            c.report_timing(&[1000.0, 1000.0], &[20, 20], 5.0);
        }
        c.report_timing(&[1000.0, 1000.0], &[20, 20], 500.0);
        // Proposal: ×2-capped grow to ceil(20 * 2.0) = 40. NudgeDown discards
        // that and applies factor 0.5 to the current anchor (20).
        c.discard_proposed_anchor();
        c.nudge_anchor_down(0.5);
        assert_eq!(c.anchor(), 10, "nudge halves the pre-proposal anchor");
    }

    #[test]
    fn test_cadence_three_devices_mixed_speed() {
        let mut c = ElChe::new(3, 10)
            .with_overhead_target(0.50); // no auto-tune

        // Device 0: 3x fast (333ms), device 1: 2x fast (500ms), device 2: slow (1000ms).
        let bc = c.batch_counts().to_vec(); c.report_timing(&[333.0, 500.0, 1000.0], &bc, 10.0);

        assert_eq!(c.batches(2), 10); // slow: anchor
        // Device 1: 100ms/batch vs 33.3ms/batch for device 0
        // Device 0: ratio 100/33.3 = 3.0 => 30
        // Device 1: ratio 100/50 = 2.0 => 20
        assert_eq!(c.batches(0), 30);
        assert_eq!(c.batches(1), 20);
    }

    #[test]
    fn test_cadence_successive_reports_refine() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50);

        // First report: 2x speed ratio.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 10.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);

        // Second report: new counts, faster device did 20 in 1000ms (50ms/batch),
        // slow did 10 in 1000ms (100ms/batch). Ratio stays 2:1.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[1000.0, 1000.0], &bc, 10.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_callback_slack_reduces_firing_rank_count() {
        // Calibrate: rank 0 fast (50 ms/batch), rank 1 slow (100 ms/batch).
        // Ratio 2:1 → rank 0 gets 20 batches, rank 1 gets 10 (anchor).
        // wall_ms must scale with `bc` to keep ms-per-batch stable in
        // the trust window (50 vs 25 with different bc → different
        // smoothed ratio, which is fine in production but obscures
        // slack arithmetic in a unit test).
        let mut c = ElChe::new(2, 10).with_overhead_target(0.50);
        c.report_timing(&[500.0, 1000.0], &[10, 10], 10.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);

        // Stage 200 ms of callback slack on rank 0. At 50 ms/batch
        // that is 4 batches' worth.
        c.apply_callback_slack(&[200.0, 0.0]);

        // Recompute with timings that keep ms/batch at 50/100:
        // rank 0 doing 20 batches in 1000ms; rank 1 doing 10 in 1000ms.
        c.report_timing(&[1000.0, 1000.0], &[20, 10], 10.0);
        assert_eq!(c.batches(0), 16);
        assert_eq!(c.batches(1), 10);

        // Slack auto-clears: next recompute returns rank 0 to 20 (the
        // un-slacked target). Dead-zone hysteresis allows the jump from
        // 16 to 20 since the delta is > 5% of 16.
        c.report_timing(&[800.0, 1000.0], &[16, 10], 10.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_callback_slack_clamps_at_one() {
        // Pathologically large slack must not starve the rank entirely.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.50);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[500.0, 1000.0], &bc, 10.0);
        assert_eq!(c.batches(0), 20);

        // 10s of slack on a 50ms/batch rank = 200 batches, far above
        // the rank's 20-batch quota. Target should clamp at 1.
        c.apply_callback_slack(&[10_000.0, 0.0]);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[500.0, 1000.0], &bc, 10.0);
        assert_eq!(c.batches(0), 1, "slack must clamp at 1, not starve to 0");
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_callback_slack_size_mismatch_is_noop() {
        // Wrong-length slack vectors are silently ignored — a misconfig
        // should never crash a running cluster, just leave behavior at
        // the unslacked baseline.
        let mut c = ElChe::new(2, 10).with_overhead_target(0.50);
        c.report_timing(&[500.0, 1000.0], &[10, 10], 10.0);
        assert_eq!(c.batches(0), 20);

        // Length 3 on a 2-rank cluster: rejected.
        c.apply_callback_slack(&[200.0, 0.0, 0.0]);
        assert_eq!(c.pending_callback_slack_ms(), &[0.0, 0.0]);

        // Length 0: also rejected.
        c.apply_callback_slack(&[]);
        assert_eq!(c.pending_callback_slack_ms(), &[0.0, 0.0]);

        // Behavior unchanged after the no-op set. Use stable timings.
        c.report_timing(&[1000.0, 1000.0], &[20, 10], 10.0);
        assert_eq!(c.batches(0), 20);
    }

    #[test]
    fn test_callback_slack_multi_rank() {
        // Slack on multiple ranks simultaneously: each rank's reduction
        // is independent. ms/batch: rank 0 = 33.3ms, rank 1 = 50ms,
        // rank 2 = 100ms (anchor). Targets: rank 0 = 10 * (100/33.3) ≈ 30,
        // rank 1 = 10 * (100/50) = 20, rank 2 = 10.
        let mut c = ElChe::new(3, 10).with_overhead_target(0.50);
        c.report_timing(&[333.0, 500.0, 1000.0], &[10, 10, 10], 10.0);
        let baseline_0 = c.batches(0);
        let baseline_1 = c.batches(1);
        let baseline_2 = c.batches(2);

        // 100ms slack on rank 0 (= 4 batches @ ceil(100/33) = 4)
        // 100ms slack on rank 1 (= 2 batches @ 50ms)
        c.apply_callback_slack(&[100.0, 100.0, 0.0]);
        // Keep ms/batch stable: wall_ms = bc * ms_per_batch.
        // rank 0: 30 * 33.3 ≈ 999; rank 1: 20 * 50 = 1000; rank 2: 10 * 100 = 1000.
        c.report_timing(
            &[baseline_0 as f64 * 33.3, baseline_1 as f64 * 50.0, 1000.0],
            &[baseline_0, baseline_1, baseline_2],
            10.0,
        );

        // Rank 0 drops by 3-4 (33ms/batch → ceil(100/33) = 4).
        assert!(
            c.batches(0) == baseline_0 - 4 || c.batches(0) == baseline_0 - 3,
            "rank 0 expected baseline-3 or baseline-4, got {} (baseline {baseline_0})",
            c.batches(0),
        );
        // Rank 1 drops by 2.
        assert_eq!(c.batches(1), baseline_1 - 2);
        // Rank 2 (no slack) unchanged.
        assert_eq!(c.batches(2), baseline_2);
    }

    #[test]
    fn test_cadence_clamp_total() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50);

        // Fast device gets 20, slow gets 10. Total = 30.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[500.0, 1000.0], &bc, 10.0);

        // Only 15 batches remain in the epoch.
        let clamped = c.clamp_total(15);
        assert_eq!(clamped.iter().sum::<usize>(), 15);
        // Proportions roughly preserved (2:1).
        assert!(clamped[0] >= clamped[1], "fast device should still get more");
    }

    #[test]
    fn test_cadence_clamp_total_no_op_when_within() {
        let c = ElChe::new(2, 10);
        // Total is 20, max is 30 => no clamping needed.
        let clamped = c.clamp_total(30);
        assert_eq!(clamped, vec![10, 10]);
    }

    #[test]
    fn test_cadence_builders() {
        let c = ElChe::new(2, 10)
            .with_overhead_target(0.20)
            .with_max_anchor(100);
        assert_eq!(c.anchor(), 10);
        assert!(!c.is_calibrated());

        // Overhead target clamped to valid range
        let c2 = ElChe::new(2, 5)
            .with_overhead_target(0.001); // below min 0.01
        // Would be clamped to 0.01 internally
        let _ = c2;
    }

    #[test]
    fn test_cadence_max_batch_diff() {
        let c = ElChe::new(2, 10).with_max_batch_diff(5);
        assert_eq!(c.max_batch_diff(), Some(5));

        let c2 = ElChe::new(2, 10);
        assert_eq!(c2.max_batch_diff(), None);
    }

    #[test]
    fn test_batch_count_clamped_to_max_diff() {
        // Setup: 2 GPUs, anchor=10, max_batch_diff=3.
        let mut c = ElChe::new(2, 10).with_max_batch_diff(3);

        // First report (calibration): GPU 0 slow (10ms/batch), GPU 1 fast (2ms/batch).
        // batch_counts are [10, 10] initially, so wall = ms_per_batch * count.
        // GPU 0: 10 batches * 10ms = 100ms. GPU 1: 10 batches * 2ms = 20ms.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 20.0], &bc, 0.0);
        assert!(c.is_calibrated());
        // Calibration pass: no clamping. GPU 1 gets 50 batches (ratio 10/2 * 10).
        let counts_after_cal = c.batch_counts().to_vec();
        assert_eq!(counts_after_cal[0], 10);
        assert_eq!(counts_after_cal[1], 50);

        // Second report: GPU 1 suddenly slows to near GPU 0 speed.
        // batch_counts now [10, 50]. GPU 0: 10*10ms=100ms. GPU 1: 50*9ms=450ms.
        // ms_per_batch[1] EMA: alpha=clamp(|9-2|/2, 0.1, 0.8)=0.8, new=0.8*9+0.2*2=7.6
        // slow_ms = max(10, 7.6) = 10. target[1] = 10*(10/7.6)=13.
        // Without clamping: 50 -> 13 (drop of 37). With max_batch_diff=3: 50 -> 47.
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 450.0], &bc, 0.0);
        let counts = c.batch_counts();
        assert!(counts[1] >= counts_after_cal[1] - 3,
            "batch count drop should be clamped to 3, was {} now {}",
            counts_after_cal[1], counts[1]);
    }

    #[test]
    fn test_cadence_weighted_allreduce_validation() {
        // Validates that Ddp::weighted_all_reduce_gradients rejects
        // mismatched batch_counts length (tested indirectly via the
        // assertion in ElChe that world_size >= 2).
        let c = ElChe::new(2, 10);
        assert_eq!(c.batch_counts().len(), 2);
    }

    #[test]
    #[should_panic(expected = "El Che requires at least 2 devices")]
    fn test_cadence_requires_two_devices() {
        ElChe::new(1, 10);
    }

    #[test]
    #[should_panic(expected = "anchor must be >= 1")]
    fn test_cadence_requires_positive_anchor() {
        ElChe::new(2, 0);
    }

    #[test]
    fn test_cadence_speed_ratio_2x() {
        // Rank 1 is slow, rank 0 is 2x faster
        let c = ElChe::new(2, 10).with_speed_ratio(1, 2.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_fbrl() {
        // RTX 5060 Ti (rank 0) ~2.3x faster than GTX 1060 (rank 1)
        let c = ElChe::new(2, 10).with_speed_ratio(1, 2.3);
        assert_eq!(c.batches(0), 23);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_slow_rank_0() {
        // Rank 0 is the slow one (unusual but valid)
        let c = ElChe::new(2, 10).with_speed_ratio(0, 3.0);
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 30);
    }

    #[test]
    fn test_cadence_speed_ratio_equal() {
        let c = ElChe::new(2, 10).with_speed_ratio(1, 1.0);
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_three_devices() {
        // Rank 2 is slow, others are 3x faster
        let c = ElChe::new(3, 10).with_speed_ratio(2, 3.0);
        assert_eq!(c.batches(0), 30);
        assert_eq!(c.batches(1), 30);
        assert_eq!(c.batches(2), 10);
    }

    #[test]
    fn test_cadence_speed_ratio_three_devices_mid_slow() {
        // Rank 1 is slow, 0 and 2 are fast
        let c = ElChe::new(3, 10).with_speed_ratio(1, 2.0);
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
        assert_eq!(c.batches(2), 20);
    }

    #[test]
    fn test_cadence_max_anchor_one() {
        // max_anchor=1: minimal cadence, sync after every slow-device batch
        let mut c = ElChe::new(2, 1)
            .with_max_anchor(1)
            .with_speed_ratio(1, 2.0);

        assert_eq!(c.batches(0), 2);
        assert_eq!(c.batches(1), 1);

        // High overhead won't increase anchor past 1
        let bc = c.batch_counts().to_vec(); c.report_timing(&[100.0, 200.0], &bc, 500.0);
        assert_eq!(c.anchor(), 1);
    }

    #[test]
    fn test_nudge_anchor_down() {
        // Need calibrated ElChe so recompute_batch_counts works.
        let mut c = ElChe::new(2, 20)
            .with_overhead_target(0.50); // high target to avoid auto-tune interference
        // Calibrate with 2:1 speed ratio (rank 1 slow).
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[50.0, 100.0], &bc, 0.0);
        assert!(c.is_calibrated());
        assert_eq!(c.anchor(), 20);
        assert_eq!(c.batches(0), 40); // fast rank
        assert_eq!(c.batches(1), 20); // slow rank (anchor)

        // Halve the anchor
        c.nudge_anchor_down(0.5);
        assert_eq!(c.anchor(), 10);
        // Batch counts recomputed proportionally
        assert_eq!(c.batches(0), 20);
        assert_eq!(c.batches(1), 10);
    }

    #[test]
    fn test_nudge_anchor_down_clamped_to_one() {
        // Nudging can go below min_anchor but never below 1.
        let mut c = ElChe::new(2, 5);
        assert_eq!(c.anchor(), 5);

        // factor=0.1 -> ceil(5 * 0.1) = 1
        c.nudge_anchor_down(0.1);
        assert_eq!(c.anchor(), 1, "should clamp to 1");
    }

    #[test]
    fn test_nudge_anchor_down_never_increases() {
        let mut c = ElChe::new(2, 10);
        // factor > 1.0 is clamped to 1.0
        c.nudge_anchor_down(2.0);
        assert_eq!(c.anchor(), 10, "should never increase");
    }

    #[test]
    fn test_cadence_speed_ratio_self_corrects() {
        // Start with wrong guess: say rank 0 is slow, but it's actually fast
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50)
            .with_speed_ratio(0, 2.0);

        // Wrong: rank 0 gets 10, rank 1 gets 20
        assert_eq!(c.batches(0), 10);
        assert_eq!(c.batches(1), 20);

        // Election can change the anchor only once the balancer enters
        // `Phase::Stable` (≥5 calibrations) — by design, no single noisy
        // reading can flip the initial pick. Feed corrective timings (rank
        // 0 is actually 2x faster) for six reports so the 6th sees Stable
        // on entry and re-elects on the trust window. bc passed verbatim
        // each call so the synthetic per-batch arithmetic stays stable.
        for _ in 0..6 {
            c.report_timing(&[500.0, 2000.0], &[10, 20], 10.0);
        }

        // Self-corrected: rank 1 is slow (anchor), rank 0 gets more
        assert_eq!(c.batches(1), c.anchor());
        assert!(c.batches(0) > c.batches(1), "fast device should get more batches");
    }

    // -- PR 1: Phase machine + tie-band anchor election -----------------------

    use crate::distributed::Phase;

    #[test]
    fn test_phase_starts_at_probe() {
        let c = ElChe::new(3, 10);
        assert_eq!(c.phase(), Phase::Probe);
        assert_eq!(c.anchor_rank(), None);
    }

    #[test]
    fn test_phase_advances_on_first_calibration() {
        let mut c = ElChe::new(3, 10).with_overhead_target(0.50);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[100.0, 380.0, 395.0], &bc, 10.0);
        assert_eq!(c.phase(), Phase::Warmup);
        assert!(c.anchor_rank().is_some());
    }

    #[test]
    fn test_phase_warmup_to_stable_at_5() {
        let mut c = ElChe::new(2, 10).with_overhead_target(0.50);
        for _ in 0..5 {
            let bc = c.batch_counts().to_vec();
            c.report_timing(&[500.0, 1000.0], &bc, 10.0);
        }
        assert_eq!(c.phase(), Phase::Stable);
    }

    #[test]
    fn test_phase_stable_to_mature_at_20() {
        let mut c = ElChe::new(2, 10).with_overhead_target(0.50);
        for _ in 0..20 {
            let bc = c.batch_counts().to_vec();
            c.report_timing(&[500.0, 1000.0], &bc, 10.0);
        }
        assert_eq!(c.phase(), Phase::Mature);
    }

    #[test]
    fn test_anchor_stable_under_tied_slow_ranks() {
        // The 3-GPU bug case: rank 0 fast (100ms), ranks 1 and 2 within 5%
        // of each other (380 vs 395). Old argmax flapped between 1 and 2 each
        // cycle; tie-band + sticky should pin one and keep it.
        let mut c = ElChe::new(3, 10).with_overhead_target(0.50);

        let bc = c.batch_counts().to_vec();
        c.report_timing(&[100.0, 380.0, 395.0], &bc, 10.0);
        let first = c.anchor_rank().expect("anchor elected");

        // Subsequent cycles with the slowest swapping inside the tie band.
        for (a, b) in &[(390.0, 380.0), (385.0, 388.0), (392.0, 386.0)] {
            let bc = c.batch_counts().to_vec();
            c.report_timing(&[100.0, *a, *b], &bc, 10.0);
            assert_eq!(
                c.anchor_rank(), Some(first),
                "anchor must stay sticky across tied slow-rank fluctuations",
            );
        }
    }

    #[test]
    fn test_anchor_switches_when_clear_winner_emerges() {
        // Outside the cohort band (>15% margin), anchor must follow the real slow.
        let mut c = ElChe::new(3, 10).with_overhead_target(0.50);

        c.report_timing(&[100.0, 400.0, 200.0], &[10, 10, 10], 10.0);
        assert_eq!(c.anchor_rank(), Some(1));

        // Rank 2 becomes clearly slower. Anchor swaps are gated to Stable
        // (≥5 calibrations) so only the 6th call onward sees the new
        // election. Push five corrective reports for the trust window to
        // dominate, then assert. bc fixed to keep ms_per_batch stable.
        for _ in 0..5 {
            c.report_timing(&[100.0, 200.0, 600.0], &[10, 10, 10], 10.0);
        }
        assert_eq!(c.anchor_rank(), Some(2), "real slowdown must be tracked");
    }

    #[test]
    fn test_relax_anchor_up_grows_anchor() {
        let mut c = ElChe::new(2, 10).with_overhead_target(0.50);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[500.0, 1000.0], &bc, 5.0);
        let before = c.anchor();
        c.relax_anchor_up();
        assert_eq!(c.anchor(), before + 1, "anchor should grow by 1 on relax");
    }

    #[test]
    fn test_relax_anchor_up_capped_by_max_batch_diff() {
        // Ratio 1:3 means at anchor=N, batch_counts=[N, 3N], diff=2N.
        // With max_batch_diff=20, anchor caps at 10 (yielding [10, 30]).
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50)
            .with_max_batch_diff(20);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[300.0, 900.0], &bc, 5.0); // 1:3 speed ratio
        // Already at anchor=10, [10,30], diff=20. Next relax would project
        // anchor=11 → [11, 33], diff=22 > 20 → refuse.
        let before = c.anchor();
        c.relax_anchor_up();
        assert_eq!(c.anchor(), before, "relax must refuse when projected diff exceeds cap");
    }

    #[test]
    fn test_relax_anchor_up_capped_by_max_anchor() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.50)
            .with_max_anchor(11);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[500.0, 1000.0], &bc, 5.0);
        c.relax_anchor_up();
        assert_eq!(c.anchor(), 11);
        c.relax_anchor_up();
        assert_eq!(c.anchor(), 11, "relax must respect max_anchor");
    }

    #[test]
    fn test_anchor_election_lowest_rank_tiebreak() {
        // No prior anchor (Probe phase first call): with all ranks tied, the
        // deterministic tiebreak picks the lowest-indexed candidate.
        let mut c = ElChe::new(3, 10).with_overhead_target(0.50);
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[100.0, 100.0, 100.0], &bc, 10.0);
        assert_eq!(c.anchor_rank(), Some(0));
    }
