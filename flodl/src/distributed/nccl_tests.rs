    use super::*;
    use crate::tensor::{test_device, cuda_device_count, cuda_synchronize, TensorOptions, DType};
    use crate::distributed::ddp::NCCL_LOCK;

    fn require_multi_gpu() -> bool {
        require_n_gpu(2)
    }

    fn require_n_gpu(n: i32) -> bool {
        if !test_device().is_cuda() || cuda_device_count() < n {
            return false;
        }
        // Verify all required devices can run compute kernels (e.g., GTX 1060
        // sm_61 is unsupported by libtorch cu128 builds).
        for i in 0..n {
            let dev = Device::CUDA(i as u8);
            let opts = TensorOptions { dtype: DType::Float32, device: dev };
            if Tensor::zeros(&[1], opts).is_err() {
                eprintln!("Device CUDA({i}) cannot run compute kernels, skipping {n}-GPU test");
                return false;
            }
        }
        true
    }

    #[test]
    fn test_nccl_requires_two_devices() {
        let result = NcclComms::new(&[Device::CUDA(0)]);
        assert!(result.is_err(), "NcclComms should require 2+ devices");
    }

    #[test]
    fn test_nccl_rejects_cpu() {
        let result = NcclComms::new(&[Device::CPU, Device::CPU]);
        assert!(result.is_err(), "NcclComms should reject CPU devices");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_init_destroy() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();
        assert_eq!(comms.size(), 2);
        assert_eq!(comms.devices(), &[Device::CUDA(0), Device::CUDA(1)]);
        // Drop cleans up
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_broadcast() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // Set values on device 0, zeros on device 1
        let t0 = Tensor::full(&[64], 42.0, opts0).unwrap();
        let t1 = Tensor::zeros(&[64], opts1).unwrap();

        // Broadcast from device 0
        comms.broadcast(&[&t0, &t1], 0).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "device 0 should still have 42.0");
        assert!(vals1.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "device 1 should have 42.0 after broadcast");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_all_reduce_sum() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // 1.0 on device 0, 2.0 on device 1
        let t0 = Tensor::full(&[128], 1.0, opts0).unwrap();
        let t1 = Tensor::full(&[128], 2.0, opts1).unwrap();

        comms.all_reduce(&[&t0, &t1], ReduceOp::Sum).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        // Sum: 1.0 + 2.0 = 3.0 on both devices
        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "device 0 should have 3.0 after AllReduce Sum");
        assert!(vals1.iter().all(|&v| (v - 3.0).abs() < 1e-5),
            "device 1 should have 3.0 after AllReduce Sum");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_all_reduce_avg() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // 10.0 on device 0, 20.0 on device 1
        let t0 = Tensor::full(&[64], 10.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 20.0, opts1).unwrap();

        comms.all_reduce(&[&t0, &t1], ReduceOp::Avg).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);

        // Avg: (10.0 + 20.0) / 2 = 15.0
        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "device 0 should have 15.0 after AllReduce Avg");
        assert!(vals1.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "device 1 should have 15.0 after AllReduce Avg");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_all_reduce_on_streams() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[Device::CUDA(0), Device::CUDA(1)]).unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        let stream0 = CudaStream::new(Device::CUDA(0), false).unwrap();
        let stream1 = CudaStream::new(Device::CUDA(1), false).unwrap();

        let t0 = Tensor::full(&[32], 5.0, opts0).unwrap();
        let t1 = Tensor::full(&[32], 7.0, opts1).unwrap();

        comms.all_reduce_on_streams(
            &[&t0, &t1], ReduceOp::Sum, &[&stream0, &stream1],
        ).unwrap();

        stream0.synchronize().unwrap();
        stream1.synchronize().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 12.0).abs() < 1e-5),
            "device 0 should have 12.0 after AllReduce Sum on streams");
        assert!(vals1.iter().all(|&v| (v - 12.0).abs() < 1e-5),
            "device 1 should have 12.0 after AllReduce Sum on streams");
    }

    // --- NcclRankComm tests ---

    #[test]
    fn test_nccl_rank_comm_rejects_invalid_rank() {
        let result = NcclRankComm::init_rank(2, 2, &NcclUniqueId { bytes: [0; NCCL_UNIQUE_ID_BYTES] });
        assert!(result.is_err(), "rank >= world_size should fail");
    }

    #[test]
    fn test_nccl_rank_comm_rejects_world_size_one() {
        let result = NcclRankComm::init_rank(0, 1, &NcclUniqueId { bytes: [0; NCCL_UNIQUE_ID_BYTES] });
        assert!(result.is_err(), "world_size < 2 should fail");
    }

    #[test]
    fn test_nccl_unique_id_clone() {
        // NcclUniqueId must be cloneable for distribution to worker threads
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<NcclUniqueId>();
    }

    #[test]
    fn test_nccl_rank_comm_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NcclRankComm>();
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_premul_sum_weighted_consensus() {
        // PreMulSum: each rank premultiplies by ITS OWN factor inside the
        // collective. With factors 0.75 / 0.25 over values 10 / 20 the
        // output must be 0.75·10 + 0.25·20 = 12.5 on BOTH ranks — the
        // work-weighted consensus with zero bookend kernels. Also
        // exercises the full dynamic-op lifecycle (create → collective →
        // destroy) and the f32 dtype guard. Skips on <2 GPUs (mirrors
        // its sibling rank-comm tests); the cluster rig smokes exercise
        // the same path at world 3.
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;
        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let t0 = Tensor::full(&[64], 10.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 20.0, opts1).unwrap();

        // Non-f32 rejection is a pure pre-collective guard — safe to
        // check on one rank without a matching collective on the peer.
        let opts_i = TensorOptions { dtype: DType::Int64, device: Device::CUDA(0) };
        let ti = Tensor::full(&[4], 1.0, opts_i).unwrap();
        let err = comm0.all_reduce_premul_sum(&[&ti], 0.5, None).unwrap_err();
        assert!(err.to_string().contains("f32"), "got: {err}");

        let t0c = t0.clone();
        let t1c = t1.clone();
        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce_premul_sum(&[&t0c], 0.75, None).unwrap();
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce_premul_sum(&[&t1c], 0.25, None).unwrap();
        });
        h0.join().unwrap();
        h1.join().unwrap();
        crate::tensor::cuda_synchronize(0);
        crate::tensor::cuda_synchronize(1);

        let v0: f64 = t0.mean().unwrap().item().unwrap();
        let v1: f64 = t1.mean().unwrap().item().unwrap();
        assert!((v0 - 12.5).abs() < 1e-5, "rank0 consensus should be 12.5, got {v0}");
        assert!((v1 - 12.5).abs() < 1e-5, "rank1 consensus should be 12.5, got {v1}");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_init_and_reduce() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        // Each rank must call init_rank concurrently. Use two threads.
        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        assert_eq!(comm0.rank(), 0);
        assert_eq!(comm0.world_size(), 2);
        assert_eq!(comm1.rank(), 1);

        // AllReduce Avg: 10.0 on dev0, 20.0 on dev1 -> 15.0 on both
        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let t0 = Tensor::full(&[64], 10.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 20.0, opts1).unwrap();

        // AllReduce must be called concurrently from different threads
        let t0_clone = t0.clone();
        let t1_clone = t1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce(&[&t0_clone], ReduceOp::Avg).unwrap();
            cuda_synchronize(0);
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce(&[&t1_clone], ReduceOp::Avg).unwrap();
            cuda_synchronize(1);
        });
        h0.join().unwrap();
        h1.join().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "rank 0 should have 15.0 after AllReduce Avg, got {}", vals0[0]);
        assert!(vals1.iter().all(|&v| (v - 15.0).abs() < 1e-5),
            "rank 1 should have 15.0 after AllReduce Avg, got {}", vals1[0]);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_on_stream() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let stream0 = CudaStream::new(Device::CUDA(0), false).unwrap();
        let stream1 = CudaStream::new(Device::CUDA(1), false).unwrap();

        let t0 = Tensor::full(&[32], 3.0, opts0).unwrap();
        let t1 = Tensor::full(&[32], 7.0, opts1).unwrap();
        let t0c = t0.clone();
        let t1c = t1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce_on_stream(&[&t0c], ReduceOp::Sum, &stream0).unwrap();
            stream0.synchronize().unwrap();
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce_on_stream(&[&t1c], ReduceOp::Sum, &stream1).unwrap();
            stream1.synchronize().unwrap();
        });
        h0.join().unwrap();
        h1.join().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 10.0).abs() < 1e-5),
            "rank 0 should have 10.0 after Sum, got {}", vals0[0]);
        assert!(vals1.iter().all(|&v| (v - 10.0).abs() < 1e-5),
            "rank 1 should have 10.0 after Sum, got {}", vals1[0]);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_multi_tensor_batch() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };

        // Two tensors per rank (simulates multiple params)
        let a0 = Tensor::full(&[16], 1.0, opts0).unwrap();
        let b0 = Tensor::full(&[8], 100.0, opts0).unwrap();
        let a1 = Tensor::full(&[16], 3.0, opts1).unwrap();
        let b1 = Tensor::full(&[8], 200.0, opts1).unwrap();

        let a0c = a0.clone();
        let b0c = b0.clone();
        let a1c = a1.clone();
        let b1c = b1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.all_reduce(&[&a0c, &b0c], ReduceOp::Avg).unwrap();
            cuda_synchronize(0);
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.all_reduce(&[&a1c, &b1c], ReduceOp::Avg).unwrap();
            cuda_synchronize(1);
        });
        h0.join().unwrap();
        h1.join().unwrap();

        // a: avg(1.0, 3.0) = 2.0, b: avg(100.0, 200.0) = 150.0
        let va0 = a0.to_f32_vec().unwrap();
        let vb0 = b0.to_f32_vec().unwrap();
        assert!(va0.iter().all(|&v| (v - 2.0).abs() < 1e-5), "a0 should be 2.0");
        assert!(vb0.iter().all(|&v| (v - 150.0).abs() < 1e-5), "b0 should be 150.0");

        let va1 = a1.to_f32_vec().unwrap();
        let vb1 = b1.to_f32_vec().unwrap();
        assert!(va1.iter().all(|&v| (v - 2.0).abs() < 1e-5), "a1 should be 2.0");
        assert!(vb1.iter().all(|&v| (v - 150.0).abs() < 1e-5), "b1 should be 150.0");
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_broadcast() {
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let comm1 = h1.join().unwrap();

        // Rank 0 (root) holds 42.0; rank 1 holds 0.0. After broadcast(root=0)
        // both ranks must hold 42.0.
        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let t0 = Tensor::full(&[64], 42.0, opts0).unwrap();
        let t1 = Tensor::zeros(&[64], opts1).unwrap();
        let t0c = t0.clone();
        let t1c = t1.clone();

        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            comm0.broadcast(&[&t0c], 0).unwrap();
            cuda_synchronize(0);
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            comm1.broadcast(&[&t1c], 0).unwrap();
            cuda_synchronize(1);
        });
        h0.join().unwrap();
        h1.join().unwrap();

        let vals0 = t0.to_f32_vec().unwrap();
        let vals1 = t1.to_f32_vec().unwrap();
        assert!(vals0.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "rank 0 (root) should retain 42.0, got {}", vals0[0]);
        assert!(vals1.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "rank 1 should receive 42.0 from root, got {}", vals1[0]);
    }

    #[test]
    #[ignore = "NCCL init needs exclusive GPU; run with: fdl cuda-test-all"]
    fn test_nccl_rank_comm_broadcast_rejects_oob_root() {
        // The root-vs-world_size check is in Rust (pre-FFI), but constructing
        // a real NcclRankComm needs live NCCL init -- there's no public
        // builder for a stub instance. So this test runs only on CUDA builds.
        if !require_multi_gpu() { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let uid = NcclUniqueId::new().unwrap();
        let uid0 = uid.clone();
        let uid1 = uid;
        let h0 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(0);
            NcclRankComm::init_rank(0, 2, &uid0).unwrap()
        });
        let h1 = std::thread::spawn(move || {
            crate::tensor::set_current_cuda_device(1);
            NcclRankComm::init_rank(1, 2, &uid1).unwrap()
        });
        let comm0 = h0.join().unwrap();
        let _comm1 = h1.join().unwrap();

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let t = Tensor::zeros(&[4], opts0).unwrap();
        // world_size = 2; root=2 must error before invoking FFI.
        let err = comm0.broadcast(&[&t], 2).unwrap_err();
        assert!(err.to_string().contains("out of range"), "got: {err}");
    }

    // 3-GPU smoke: heterogeneous topology (e.g. RTX 5060 Ti sm_120 +
    // 2x GTX 1060 sm_61). Uses NcclComms main-thread init since
    // ncclCommInitRank from worker threads corrupts CUDA context across
    // heterogeneous architectures.
    #[test]
    #[ignore = "NCCL init needs exclusive GPU; needs 3 GPUs; run with: fdl cuda-test-all"]
    fn test_nccl_three_gpu_all_reduce_sum() {
        if !require_n_gpu(3) { return; }
        let _lock = NCCL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let comms = NcclComms::new(&[
            Device::CUDA(0), Device::CUDA(1), Device::CUDA(2),
        ]).unwrap();
        assert_eq!(comms.size(), 3);

        let opts0 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(0) };
        let opts1 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(1) };
        let opts2 = TensorOptions { dtype: DType::Float32, device: Device::CUDA(2) };

        let t0 = Tensor::full(&[64], 1.0, opts0).unwrap();
        let t1 = Tensor::full(&[64], 2.0, opts1).unwrap();
        let t2 = Tensor::full(&[64], 3.0, opts2).unwrap();

        comms.all_reduce(&[&t0, &t1, &t2], ReduceOp::Sum).unwrap();
        cuda_synchronize(0);
        cuda_synchronize(1);
        cuda_synchronize(2);

        // Sum: 1 + 2 + 3 = 6 on every device.
        for (i, t) in [&t0, &t1, &t2].iter().enumerate() {
            let vals = t.to_f32_vec().unwrap();
            assert!(vals.iter().all(|&v| (v - 6.0).abs() < 1e-5),
                "rank {i} should have 6.0 after AllReduce Sum, got {}", vals[0]);
        }
    }
