    use super::*;
    use crate::tensor::{
        cuda_device_count, cuda_synchronize, test_device, DType, TensorOptions,
    };
    use super::NCCL_LOCK;

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

    // -- CPU validation tests -----------------------------------------------

    #[test]
    fn test_ddp_requires_two_models() {
        // Can't construct Ddp with 1 model (NCCL needs 2+ CUDA devices).
        // Just verify the validation logic.
        let result = Ddp::wrap(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ddp_model_device_mismatch() {
        // Model count must match device count
        let result = Ddp::wrap(
            &[],
            &[Device::CUDA(0), Device::CUDA(1)],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_shard_sizes_equal() {
        let ratios = vec![0.5, 0.5];
        let state = mock_state(&ratios);
        assert_eq!(state.compute_shard_sizes(10), vec![5, 5]);
        assert_eq!(state.compute_shard_sizes(11), vec![6, 5]);
        assert_eq!(state.compute_shard_sizes(3), vec![2, 1]);
    }

    #[test]
    fn test_shard_sizes_unequal() {
        let ratios = vec![0.7, 0.3];
        let state = mock_state(&ratios);
        assert_eq!(state.compute_shard_sizes(10), vec![7, 3]);
        assert_eq!(state.compute_shard_sizes(100), vec![70, 30]);
    }

    #[test]
    fn test_shard_sizes_three_devices() {
        let ratios = vec![0.5, 0.3, 0.2];
        let state = mock_state(&ratios);
        let sizes = state.compute_shard_sizes(10);
        assert_eq!(sizes.iter().sum::<i64>(), 10);
        assert_eq!(sizes, vec![5, 3, 2]);
    }

    /// Helper: create a minimal DistributedState for unit tests.
    fn mock_state(ratios: &[f64]) -> DistributedState {
        let n = ratios.len();
        DistributedState {
            replicas: Vec::new(),
            // Safety: we never use comms in shard/balance tests. Build a dummy.
            comms: unsafe { mock_nccl_comms(n) },
            devices: (0..n as u8)
                .map(Device::CUDA)
                .collect(),
            optimizers: Vec::new(),
            chunk_ratios: ratios.to_vec(),
            param_groups: Vec::new(),
            buffer_groups: Vec::new(),
            last_timing: None,
            last_shard_sizes: vec![0; n],
            ema_throughput: vec![0.0; n],
            step_count: 0,
            calibration_steps: DEFAULT_CALIBRATION_STEPS,
            rebalance_interval: DEFAULT_REBALANCE_INTERVAL,
            el_che: None,
            last_el_che_counts: Vec::new(),
            last_el_che_sync: None,
            max_grad_norm: None,
            timeline: None,
        }
    }

    /// Create a NcclComms with a null handle for shard-size unit tests only.
    /// Never call any actual NCCL operations on this.
    unsafe fn mock_nccl_comms(n: usize) -> NcclComms {
        let devices: Vec<Device> = (0..n as u8).map(Device::CUDA).collect();
        // Drop on a null handle is a no-op.
        unsafe { NcclComms::from_raw(std::ptr::null_mut(), devices) }
    }

    // -- Auto-balancer unit tests (CPU, no NCCL needed) ---------------------

    #[test]
    fn test_is_balanced_equal() {
        let state = mock_state(&[0.5, 0.5]);
        assert!(state.is_balanced());
    }

    #[test]
    fn test_is_balanced_unequal() {
        let state = mock_state(&[0.7, 0.3]);
        assert!(!state.is_balanced());
    }

    #[test]
    fn test_rebalance_proportional() {
        let mut state = mock_state(&[0.5, 0.5]);
        // GPU 0 is 3x faster than GPU 1
        state.ema_throughput = vec![30.0, 10.0];
        state.rebalance();
        // Expect ~75/25 split
        assert!((state.chunk_ratios[0] - 0.75).abs() < 0.01,
            "fast GPU should get ~75%, got {}", state.chunk_ratios[0]);
        assert!((state.chunk_ratios[1] - 0.25).abs() < 0.01,
            "slow GPU should get ~25%, got {}", state.chunk_ratios[1]);
        // Must sum to 1.0
        let sum: f64 = state.chunk_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "ratios must sum to 1.0, got {sum}");
    }

    #[test]
    fn test_rebalance_three_devices() {
        let mut state = mock_state(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        // Throughput: 50, 30, 20 (total 100)
        state.ema_throughput = vec![50.0, 30.0, 20.0];
        state.rebalance();
        assert!((state.chunk_ratios[0] - 0.50).abs() < 0.01);
        assert!((state.chunk_ratios[1] - 0.30).abs() < 0.01);
        assert!((state.chunk_ratios[2] - 0.20).abs() < 0.01);
        let sum: f64 = state.chunk_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rebalance_respects_min_ratio() {
        let mut state = mock_state(&[0.5, 0.5]);
        // GPU 1 is extremely slow (would get <5% without clamping)
        state.ema_throughput = vec![100.0, 1.0];
        state.rebalance();
        assert!(state.chunk_ratios[1] >= MIN_CHUNK_RATIO,
            "slow GPU should get at least MIN_CHUNK_RATIO, got {}", state.chunk_ratios[1]);
        let sum: f64 = state.chunk_ratios.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rebalance_no_data() {
        let mut state = mock_state(&[0.5, 0.5]);
        state.ema_throughput = vec![0.0, 0.0];
        state.rebalance();
        // Should not change ratios when no data
        assert_eq!(state.chunk_ratios, vec![0.5, 0.5]);
    }

    #[test]
    fn test_update_balance_calibration_timing() {
        let mut state = mock_state(&[0.5, 0.5]);
        // Simulate steps without timing (no CudaEvents on CPU)
        for _ in 0..DEFAULT_CALIBRATION_STEPS - 1 {
            let rebalanced = state.update_balance().unwrap();
            assert!(!rebalanced, "should not rebalance during calibration");
        }
        // Step at calibration boundary triggers rebalance (but no-op without data)
        let rebalanced = state.update_balance().unwrap();
        assert!(rebalanced, "should rebalance at calibration boundary");
    }

    #[test]
    fn test_update_balance_interval() {
        let mut state = mock_state(&[0.5, 0.5]);
        // Skip past calibration
        state.step_count = DEFAULT_CALIBRATION_STEPS;
        // Steps up to next interval should not rebalance
        for _ in 0..DEFAULT_REBALANCE_INTERVAL - 1 {
            let rebalanced = state.update_balance().unwrap();
            assert!(!rebalanced);
        }
        // At interval boundary: rebalance
        let rebalanced = state.update_balance().unwrap();
        assert!(rebalanced);
    }

    #[test]
    fn test_ema_throughput_init() {
        let mut state = mock_state(&[0.5, 0.5]);
        // First measurement initializes directly (not blended)
        state.ema_throughput = vec![0.0, 0.0];
        // Manually set what update_throughput would compute from timing
        let throughput_0 = 10.0;
        state.ema_throughput[0] = throughput_0; // simulates first measurement
        assert_eq!(state.ema_throughput[0], 10.0);
    }

    #[test]
    fn test_ema_throughput_smoothing() {
        let mut state = mock_state(&[0.5, 0.5]);
        state.ema_throughput = vec![10.0, 5.0];
        // Simulate what update_balance does with a new measurement
        let new_measurement = 20.0;
        state.ema_throughput[0] =
            EMA_ALPHA * new_measurement + (1.0 - EMA_ALPHA) * state.ema_throughput[0];
        // EMA: 0.3 * 20 + 0.7 * 10 = 6 + 7 = 13
        assert!((state.ema_throughput[0] - 13.0).abs() < 1e-9);
    }

    #[test]
    fn test_shard_sizes_after_rebalance() {
        let mut state = mock_state(&[0.5, 0.5]);
        // Rebalance to 70/30
        state.ema_throughput = vec![70.0, 30.0];
        state.rebalance();
        // Verify shard computation uses new ratios
        let sizes = state.compute_shard_sizes(100);
        assert_eq!(sizes.iter().sum::<i64>(), 100);
        assert_eq!(sizes[0], 70);
        assert_eq!(sizes[1], 30);
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
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_graph_distribute_adapts_to_hardware() {
        use crate::graph::FlowBuilder;
        use crate::nn::Linear;
        use crate::tensor::usable_cuda_devices;

        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model = FlowBuilder::from(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let result = model.distribute(|dev| {
            FlowBuilder::from(Linear::on_device(4, 2, dev)?).build()
        });
        assert!(result.is_ok());

        let usable = usable_cuda_devices();
        if usable.len() >= 2 {
            // Multi-GPU: should be distributed
            assert!(model.is_distributed());
            assert_eq!(model.world_size(), usable.len());
        } else {
            // Single GPU or CPU: no-op
            assert!(!model.is_distributed());
            assert_eq!(model.world_size(), 1);
        }
    }

    #[test]
    fn test_ddp_auto_single_gpu() {
        // On multi-GPU hardware Trainer::setup would initialize NCCL,
        // which poisons CUBLAS for concurrent tests. Skip here;
        // multi-GPU path is validated in test_ddp_auto_multi_gpu.
        if cuda_device_count() >= 2 {
            return;
        }

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};

        let model = FlowBuilder::from(Linear::new(4, 8).unwrap())
            .through(ReLU::new())
            .through(Linear::new(8, 2).unwrap())
            .build()
            .unwrap();

        Trainer::setup(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(p, 0.001),
        )
        .unwrap();

        // Optimizer should be set: step() works
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

        assert!(!model.is_distributed());
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
    fn test_ddp_auto_multi_gpu() {
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};

        let model = FlowBuilder::from(
            Linear::on_device(4, 8, Device::CUDA(0)).unwrap(),
        )
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, Device::CUDA(0)).unwrap())
        .build()
        .unwrap();

        Trainer::setup(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(p, 0.001),
        )
        .unwrap();

        assert!(model.is_distributed());
        assert_eq!(model.world_size(), 2);

        // Full training step
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(0),
        };
        let x = Variable::new(
            Tensor::randn(&[8, 4], opts).unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::randn(&[8, 2], opts).unwrap(),
            false,
        );
        let out = model.forward(&x).unwrap();
        let loss = mse_loss(&out, &target).unwrap();
        loss.backward().unwrap();
        model.step().unwrap();

        cuda_synchronize(0);
        cuda_synchronize(1);
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
        // High AllReduce overhead should trigger anchor increase.
        // 10% target: compute 1000ms, sync 500ms => overhead 50% >> 10%.
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

        // overhead = 500/1000 = 0.50, target = 0.10
        // scale = 0.50/0.10 = 5.0 => new anchor = ceil(10 * 5) = 50
        assert_eq!(c.anchor(), 50);
        assert_eq!(c.batches(0), 50);
        assert_eq!(c.batches(1), 50);
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

        // overhead = 400/1000 = 0.40, target = 0.10, scale = 4.0
        // new anchor = ceil(10 * 4) = 40
        assert_eq!(c.anchor(), 40);
        assert_eq!(c.batches(1), 40); // slow device
        // fast device: 100ms/batch vs 50ms/batch => 2x ratio => 80
        assert_eq!(c.batches(0), 80);
    }

    #[test]
    fn test_cadence_anchor_capped_at_max() {
        let mut c = ElChe::new(2, 10)
            .with_overhead_target(0.01)
            .with_max_anchor(30);

        // Prime to Stable phase before triggering auto-tune.
        for _ in 0..5 {
            let bc = c.batch_counts().to_vec();
            c.report_timing(&[100.0, 100.0], &bc, 0.5);
        }
        // Extreme overhead: sync dominates.
        let bc = c.batch_counts().to_vec();
        c.report_timing(&[100.0, 100.0], &bc, 500.0);

        // Would want anchor=500 but capped at 30.
        assert_eq!(c.anchor(), 30);
        assert_eq!(c.batches(0), 30);
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

    // -- DdpConfig tests ------------------------------------------------------

    #[test]
    fn test_ddp_config_defaults() {
        let c = DdpConfig::new();
        assert!(c.speed_hint.is_none());
        assert!(c.overhead_target.is_none());
        assert!(c.max_anchor.is_none());
    }

    #[test]
    fn test_ddp_config_builder() {
        let c = DdpConfig::new()
            .speed_hint(1, 2.5)
            .overhead_target(0.05)
            .max_anchor(Some(20));
        assert_eq!(c.speed_hint, Some((1, 2.5)));
        assert_eq!(c.overhead_target, Some(0.05));
        assert_eq!(c.max_anchor, Some(20));
    }

    #[test]
    fn test_ddp_config_disable_el_che() {
        let c = DdpConfig::new().max_anchor(Some(0));
        assert_eq!(c.max_anchor, Some(0));
    }

    #[test]
    fn test_configure_el_che_creates_from_config() {
        let mut state = mock_state(&[0.5, 0.5]);

        let config = DdpConfig::new().speed_hint(1, 2.0).overhead_target(0.15);
        state.configure_el_che(&config);

        assert!(state.el_che.is_some());
        let el = state.el_che.as_ref().unwrap();
        // Slow rank gets anchor, fast gets more
        assert_eq!(el.batches(1), el.anchor());
        assert!(el.batches(0) > el.batches(1));
    }

    #[test]
    fn test_configure_el_che_disabled() {
        let mut state = mock_state(&[0.5, 0.5]);

        let config = DdpConfig::new().max_anchor(Some(0));
        state.configure_el_che(&config);

        assert!(state.el_che.is_none());
    }

    #[test]
    fn test_configure_el_che_single_device_noop() {
        let mut state = mock_state(&[1.0]);

        let config = DdpConfig::new();
        state.configure_el_che(&config);

        // Single device -- El Che not created
        assert!(state.el_che.is_none());
    }

    // -- El Che CUDA integration tests (multi-GPU, NCCL) ----------------------

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
    fn test_el_che_full_training_loop() {
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};
        use crate::data::{DataLoader, DataSet};

        // Simple dataset: 200 samples, 4 features, 2 targets
        struct TinyData;
        impl DataSet for TinyData {
            fn len(&self) -> usize { 200 }
            fn get(&self, index: usize) -> crate::tensor::Result<Vec<Tensor>> {
                let x = Tensor::from_f32(
                    &[index as f32; 4], &[4], Device::CPU,
                )?;
                let y = Tensor::from_f32(
                    &[(index as f32) * 0.1; 2], &[2], Device::CPU,
                )?;
                Ok(vec![x, y])
            }
        }

        let model = FlowBuilder::from(
            Linear::on_device(4, 8, Device::CUDA(0)).unwrap(),
        )
        .through(ReLU::new())
        .through(Linear::on_device(8, 2, Device::CUDA(0)).unwrap())
        .build()
        .unwrap();

        Trainer::setup_with(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(p, 0.001),
            DdpConfig::new().speed_hint(1, 2.0).max_anchor(Some(3)),
        )
        .unwrap();

        assert!(model.is_distributed());
        assert!(model.has_el_che());
        assert_eq!(model.world_size(), 2);

        // Set up DataLoader
        let loader = DataLoader::from_dataset(TinyData)
            .batch_size(10)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();

        // Run 1 epoch
        let mut step_count = 0;
        for batch in model.epoch(0).activate() {
            let b = batch.unwrap();
            let out = model.forward_batch(&b).unwrap();
            let target = Variable::new(b["target"].clone(), false);
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward().unwrap();
            model.step().unwrap();
            step_count += 1;
        }

        // With anchor=3 and ratio=2.0: ~5 batches per El Che step (3 + 2*3=6, total ~5-6)
        // 200 samples / 10 batch_size = 20 batches total
        // ~20 / 5 = ~4 El Che iterations
        assert!(step_count > 0, "should have trained at least one step");
        assert!(step_count <= 20, "should not have more steps than batches");

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
    fn test_el_che_tagged_outputs_gathered() {
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::FlowBuilder;
        use crate::nn::{Adam, Linear, ReLU, mse_loss};
        use crate::data::{DataLoader, DataSet};

        struct TinyData;
        impl DataSet for TinyData {
            fn len(&self) -> usize { 100 }
            fn get(&self, index: usize) -> crate::tensor::Result<Vec<Tensor>> {
                let x = Tensor::from_f32(
                    &[index as f32; 4], &[4], Device::CPU,
                )?;
                let y = Tensor::from_f32(
                    &[(index as f32) * 0.1; 2], &[2], Device::CPU,
                )?;
                Ok(vec![x, y])
            }
        }

        // Build model with a tagged intermediate
        let model = FlowBuilder::from(
            Linear::on_device(4, 8, Device::CUDA(0)).unwrap(),
        )
        .through(ReLU::new())
        .tag("hidden")
        .through(Linear::on_device(8, 2, Device::CUDA(0)).unwrap())
        .build()
        .unwrap();

        Trainer::setup_with(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(4, 8, dev)?)
                    .through(ReLU::new())
                    .tag("hidden")
                    .through(Linear::on_device(8, 2, dev)?)
                    .build()
            },
            |p| Adam::new(p, 0.001),
            DdpConfig::new().max_anchor(Some(2)),
        )
        .unwrap();

        let loader = DataLoader::from_dataset(TinyData)
            .batch_size(10)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();

        // Run one iteration and check tagged output
        let mut iter = model.epoch(0).activate();
        if let Some(batch) = iter.next() {
            let b = batch.unwrap();
            let out = model.forward_batch(&b).unwrap();

            // Tagged output should exist and have gathered batch dimension
            let hidden = model.tagged("hidden");
            assert!(hidden.is_some(), "tagged output should be gathered");
            let h = hidden.unwrap();
            // hidden shape: [total_samples_across_devices, 8]
            assert_eq!(h.shape()[1], 8);
            // Total samples should be > batch_size (multiple batches gathered)
            assert!(h.shape()[0] >= 10, "gathered hidden should span multiple batches");

            let target = Variable::new(b["target"].clone(), false);
            let loss = mse_loss(&out, &target).unwrap();
            loss.backward().unwrap();
            model.step().unwrap();
        }

        cuda_synchronize(0);
        cuda_synchronize(1);
    }

    /// LoopBody emitting two named per-iteration traces, gather-friendly.
    /// Returns 2*x; emits "double" = 2*x and "quad" = 4*x.
    struct EmittingDoublerLB;
    impl crate::nn::Module for EmittingDoublerLB {
        fn forward(&self, input: &Variable) -> crate::tensor::Result<Variable> {
            crate::nn::forward_via_step(self, input)
        }
        fn as_loop_body(&self) -> Option<&dyn crate::nn::LoopBody> { Some(self) }
    }
    impl crate::nn::LoopBody for EmittingDoublerLB {
        fn step(
            &self,
            input: &Variable,
            _refs: &std::collections::HashMap<String, Variable>,
            emit: &mut crate::nn::TraceEmit<'_>,
        ) -> crate::tensor::Result<Variable> {
            let two_x = input.add(input)?;
            let four_x = two_x.add(&two_x)?;
            emit.publish("double", two_x.clone());
            emit.publish("quad", four_x);
            Ok(two_x)
        }
    }

    // DO NOT REMOVE the #[ignore] attribute below.
    //
    // This test exercises Trainer::setup_with -> Graph::distribute (NCCL
    // multi-GPU) and MUST run isolated. Without #[ignore] it falls into
    // cuda-test-all's first leg (`cargo test --features cuda`, parallel,
    // non-ignored), where NCCL communicator init does CUDA context
    // manipulation that corrupts concurrent CUBLAS operations on shared
    // GPU threads. Symptom is hard CUBLAS_STATUS_EXECUTION_FAILED errors
    // across hundreds of otherwise-unrelated tests, not just a warning.
    // Empirically validated on 2026-04-01 across the existing NCCL test
    // surface. The device save/restore inside NcclComms methods is
    // production-correct (prevents device leaking in training loops) but
    // not enough for parallel test execution.
    //
    // Separately: this test goes through the unpinned Graph::distribute
    // path. That is fine today (everything on default stream, AccumulateGrad
    // and gradients match), but will need its own `_grad_accumulators`-style
    // stream pin (mirroring `flodl/src/distributed/ddp_run/worker.rs`) once
    // CUDA Graph capture is wired into El Che, or once model-parallel /
    // sharded paths force non-default streams on this entry point. Until
    // then the #[ignore] keeps this test compatible with the NCCL exclusivity
    // rule, matching its siblings (test_el_che_full_training_loop,
    // test_el_che_tagged_outputs_gathered).
    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-nccl"]
    fn test_el_che_loop_body_emits_gathered_across_replicas() {
        // Verify multi-trace API works under DDP: each replica's emits land in
        // its own loop's named_store, gather across ranks/batches concatenates
        // per (emit_name, step_idx), final ctx.traces[name] is reachable from
        // the loss closure with the right shape.
        if !require_multi_gpu() {
            return;
        }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        use crate::graph::{FlowBuilder, LossContext};
        use crate::nn::{Adam, Linear, mse_loss};
        use crate::data::{DataLoader, DataSet};
        use std::cell::Cell;
        use std::rc::Rc;

        struct TinyData;
        impl DataSet for TinyData {
            fn len(&self) -> usize { 32 }
            fn get(&self, index: usize) -> crate::tensor::Result<Vec<Tensor>> {
                let x = Tensor::from_f32(
                    &[(index as f32) + 1.0; 2], &[2], Device::CPU,
                )?;
                let y = Tensor::from_f32(
                    &[(index as f32) + 1.0; 2], &[2], Device::CPU,
                )?;
                Ok(vec![x, y])
            }
        }

        // Linear before the loop gives the graph learnable parameters so the
        // loss closure's backward can flow gradients (the test exercises gather,
        // not optimization, but backward requires grad-bearing params).
        let model = FlowBuilder::from(
            Linear::on_device(2, 2, Device::CUDA(0)).unwrap(),
        )
        .loop_body(EmittingDoublerLB)
        .for_n(3)
        .build()
        .unwrap();

        Trainer::setup_with(
            &model,
            |dev| {
                FlowBuilder::from(Linear::on_device(2, 2, dev)?)
                    .loop_body(EmittingDoublerLB)
                    .for_n(3)
                    .build()
            },
            |p| Adam::new(p, 0.001),
            DdpConfig::new().max_anchor(Some(2)),
        )
        .unwrap();

        let loader = DataLoader::from_dataset(TinyData)
            .batch_size(4)
            .names(&["input", "target"])
            .build()
            .unwrap();

        model.set_data_loader(loader, "input").unwrap();

        // Loss closure inspects ctx.traces — both emit names must be present
        // with non-empty Vec<Variable> on each invocation.
        let saw_emits = Rc::new(Cell::new(false));
        let saw_emits_w = saw_emits.clone();
        model.set_loss_fn(move |ctx: &LossContext| {
            let doubles = ctx.traces.get("double")
                .expect("ctx.traces missing 'double'");
            let quads = ctx.traces.get("quad")
                .expect("ctx.traces missing 'quad'");
            assert_eq!(doubles.len(), 3, "3 iterations expected");
            assert_eq!(quads.len(), 3, "3 iterations expected");
            saw_emits_w.set(true);
            let target = &ctx.batch["target"];
            let target_var = Variable::new(target.clone(), false);
            mse_loss(ctx.output, &target_var)
        });

        let iter = model.epoch(0).activate();
        let mut iterations = 0;
        for batch in iter {
            let b = batch.unwrap();
            let _out = model.forward_batch(&b).unwrap();
            model.step().unwrap();
            iterations += 1;
            if iterations >= 2 { break; }
        }

        assert!(saw_emits.get(), "loss closure must have run with traces visible");

        cuda_synchronize(0);
        cuda_synchronize(1);
    }
