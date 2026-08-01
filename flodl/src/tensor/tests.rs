    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.dtype(), DType::Float32);
        assert_eq!(t.device(), test_device());
        assert_eq!(t.numel(), 6);

        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![0.0; 6]);
    }

    #[test]
    fn test_copy_is_deep_clone_is_shallow() {
        // The TF9 invariant, pinned: `clone()` aliases storage, `copy()`
        // is an independent deep copy. An in-place mutation of the source
        // is visible through a shallow clone but never through a copy.
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let shallow = a.clone();
        let deep = a.copy().unwrap();

        let ones = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], test_device()).unwrap();
        a.add_(&ones).unwrap();

        // Shallow clone shares storage -> sees the mutation.
        assert_eq!(shallow.to_f32_vec().unwrap(), vec![2.0, 3.0, 4.0]);
        // Deep copy is independent -> unchanged.
        assert_eq!(deep.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_nbytes() {
        let f32_t = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        assert_eq!(f32_t.nbytes(), 6 * 4); // 6 elements * 4 bytes

        let f64_t = Tensor::zeros(&[2, 3], TensorOptions { dtype: DType::Float64, device: test_device() }).unwrap();
        assert_eq!(f64_t.nbytes(), 6 * 8); // 6 elements * 8 bytes

        let i64_t = Tensor::from_i64(&[1, 2, 3], &[3], test_device()).unwrap();
        assert_eq!(i64_t.nbytes(), 3 * 8); // 3 elements * 8 bytes
    }

    #[test]
    fn test_from_f32() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        assert_eq!(t.shape(), vec![3]);
        let data = t.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_blob_f16_roundtrip() {
        // IEEE 754 binary16 bit patterns: 1.0, -1.0, 0.5, 0.0
        let bits: [u16; 4] = [0x3C00, 0xBC00, 0x3800, 0x0000];
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();

        let t = Tensor::from_blob(&bytes, &[4], DType::Float16, test_device()).unwrap();
        assert_eq!(t.dtype(), DType::Float16);
        assert_eq!(t.shape(), vec![4]);

        // to_blob returns the same f16 bytes back.
        assert_eq!(t.to_blob().unwrap(), bytes);

        // to_f32_vec routes through to_dtype(F32) and produces the
        // mathematical values, not raw bit reinterpretation.
        assert_eq!(t.to_f32_vec().unwrap(), vec![1.0, -1.0, 0.5, 0.0]);
    }

    #[test]
    fn test_from_blob_size_mismatch() {
        let bytes = vec![0u8; 6]; // 3 f16 values = 6 bytes, claim shape needing 4.
        let err = Tensor::from_blob(&bytes, &[4], DType::Float16, test_device());
        assert!(err.is_err(), "expected size mismatch error");
    }

    #[test]
    fn test_typed_constructors_length_mismatch_errors() {
        // Regression: these used to hand the slice pointer to the shim
        // unchecked, so a short slice was an out-of-bounds read.
        let err = Tensor::from_f32(&[1.0], &[1000, 1000], test_device());
        assert!(err.is_err(), "expected length mismatch error");
        let msg = err.unwrap_err().to_string();
        assert!(msg.contains("from_f32"), "error should name the constructor: {msg}");

        assert!(Tensor::from_f64(&[1.0, 2.0, 3.0], &[2, 2], test_device()).is_err());
        assert!(Tensor::from_i64(&[1, 2, 3, 4, 5], &[2, 2], test_device()).is_err());
    }

    #[test]
    fn test_item_all_dtypes() {
        // Regression: the non-f64 branch used to memcpy NATIVE bytes into
        // an f32 buffer — garbage-as-Ok for f16/bf16/i32, Err for i64.
        let f16_one =
            Tensor::from_blob(&0x3C00u16.to_le_bytes(), &[1], DType::Float16, test_device())
                .unwrap();
        assert_eq!(f16_one.item().unwrap(), 1.0);

        let bf16_one =
            Tensor::from_blob(&0x3F80u16.to_le_bytes(), &[1], DType::BFloat16, test_device())
                .unwrap();
        assert_eq!(bf16_one.item().unwrap(), 1.0);

        let i64_t = Tensor::from_i64(&[42], &[1], test_device()).unwrap();
        assert_eq!(i64_t.item().unwrap(), 42.0);

        let i32_t = Tensor::from_i64(&[7], &[1], test_device())
            .unwrap()
            .to_dtype(DType::Int32)
            .unwrap();
        assert_eq!(i32_t.item().unwrap(), 7.0);

        let f64_t = Tensor::from_f64(&[std::f64::consts::PI], &[1], test_device()).unwrap();
        assert_eq!(f64_t.item().unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn test_to_i64_vec_casts_non_int64() {
        // Regression: a Float32 input used to memcpy float bit patterns
        // into the front of the i64 buffer and return Ok.
        let f = Tensor::from_f32(&[1.9, -2.7, 3.0], &[3], test_device()).unwrap();
        // Truncation toward zero, like PyTorch's .long().
        assert_eq!(f.to_i64_vec().unwrap(), vec![1, -2, 3]);

        let i = Tensor::from_i64(&[5, 6], &[2], test_device()).unwrap();
        assert_eq!(i.to_i64_vec().unwrap(), vec![5, 6]);
    }

    #[test]
    fn test_to_f64_vec_int64_full_precision() {
        // Regression: the old f32 waypoint truncated integers above 2^24.
        let big = (1i64 << 40) + 1;
        let t = Tensor::from_i64(&[big], &[1], test_device()).unwrap();
        assert_eq!(t.to_f64_vec().unwrap()[0] as i64, big);
    }

    #[test]
    fn test_from_blob_rejects_negative_and_overflowing_shape() {
        let bytes = [0u8; 16];
        assert!(
            Tensor::from_blob(&bytes, &[-4], DType::Float32, test_device()).is_err(),
            "negative dimension must error, not wrap"
        );
        assert!(
            Tensor::from_blob(&bytes, &[i64::MAX, 8], DType::Float32, test_device()).is_err(),
            "overflowing shape product must error, not wrap"
        );
    }

    #[test]
    fn test_drop_frees_memory() {
        // Create and immediately drop -- verifies Drop doesn't crash.
        let _ = Tensor::zeros(&[1000, 1000], test_opts()).unwrap();
        // If Drop is broken, this would leak or crash.
    }

    #[test]
    fn test_debug_format() {
        let t = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("[2, 3]"));
        assert!(s.contains("Float32"));
    }

    #[test]
    fn test_ones_from_f64_from_i64() {
        let o = Tensor::ones(&[2, 3], test_opts()).unwrap();
        assert_eq!(o.to_f32_vec().unwrap(), vec![1.0; 6]);

        let f = Tensor::from_f64(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        assert_eq!(f.dtype(), DType::Float64);
        assert_eq!(f.to_f64_vec().unwrap(), vec![1.0, 2.0, 3.0]);

        let i = Tensor::from_i64(&[10, 20, 30], &[3], test_device()).unwrap();
        assert_eq!(i.dtype(), DType::Int64);
        assert_eq!(i.to_i64_vec().unwrap(), vec![10, 20, 30]);
    }

    #[test]
    fn test_eye_full() {
        let eye = Tensor::eye(3, test_opts()).unwrap();
        assert_eq!(eye.shape(), vec![3, 3]);
        let data = eye.to_f32_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let f = Tensor::full(&[2, 3], 7.0, test_opts()).unwrap();
        assert_eq!(f.shape(), vec![2, 3]);
        assert_eq!(f.to_f32_vec().unwrap(), vec![7.0; 6]);
    }

    #[test]
    fn test_zeros_like_ones_like() {
        let t = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let zl = Tensor::zeros_like(&t).unwrap();
        assert_eq!(zl.to_f32_vec().unwrap(), vec![0.0, 0.0]);
        assert_eq!(zl.dtype(), DType::Float32);

        let ol = Tensor::ones_like(&t).unwrap();
        assert_eq!(ol.to_f32_vec().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_from_i64_device() {
        let t = Tensor::from_i64(&[1, 2, 3], &[3], test_device()).unwrap();
        assert_eq!(t.device(), test_device());
        assert_eq!(t.dtype(), DType::Int64);
        assert_eq!(t.to_i64_vec().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_pin_memory() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], Device::CPU).unwrap();
        assert!(!t.is_pinned(), "regular CPU tensor should not be pinned");

        if gpu_available() {
            let pinned = t.pin_memory().unwrap();
            assert!(pinned.is_pinned(), "pin_memory() result should be pinned");
            assert_eq!(pinned.device(), Device::CPU, "pinned tensor should stay on CPU");
            assert_eq!(pinned.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0],
                "data should be preserved after pinning");
        } else {
            // pin_memory requires CUDA -- verify it returns an error on CPU-only
            assert!(t.pin_memory().is_err(),
                "pin_memory should fail without CUDA");
        }
    }

    #[test]
    fn test_channels_last() {
        let t = Tensor::randn(&[1, 3, 4, 4], test_opts()).unwrap();
        assert!(!t.is_channels_last());
        let cl = t.to_channels_last().unwrap();
        assert!(cl.is_channels_last());
        assert_eq!(cl.shape(), vec![1, 3, 4, 4]); // shape unchanged
    }

    #[test]
    fn test_adam_step_basic() {
        // Basic smoke test for the fused adam_step at tensor level
        let param = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let grad = Tensor::from_f32(&[0.5, 0.5], &[2], test_device()).unwrap();
        let m = Tensor::zeros(&[2], test_opts()).unwrap();
        let v = Tensor::zeros(&[2], test_opts()).unwrap();

        param.adam_step(&grad, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1).unwrap();

        let p = param.to_f32_vec().unwrap();
        assert!(p[0] < 1.0, "param[0] should decrease");
        assert!(p[1] < 2.0, "param[1] should decrease");
        // m and v should be non-zero after the step
        let m_data = m.to_f32_vec().unwrap();
        let v_data = v.to_f32_vec().unwrap();
        assert!(m_data[0] > 0.0, "m should be updated");
        assert!(v_data[0] > 0.0, "v should be updated");
    }

    // --- Device model tests ---

    #[test]
    fn test_device_enum_basics() {
        assert_eq!(Device::CPU, Device::CPU);
        assert_eq!(Device::CUDA(0), Device::CUDA(0));
        assert_ne!(Device::CUDA(0), Device::CUDA(1));
        assert_ne!(Device::CPU, Device::CUDA(0));

        assert!(!Device::CPU.is_cuda());
        assert!(Device::CUDA(0).is_cuda());
        assert!(Device::CUDA(1).is_cuda());

        assert_eq!(Device::CPU.index(), 0);
        assert_eq!(Device::CUDA(0).index(), 0);
        assert_eq!(Device::CUDA(1).index(), 1);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(format!("{}", Device::CPU), "cpu");
        assert_eq!(format!("{}", Device::CUDA(0)), "cuda");
        assert_eq!(format!("{}", Device::CUDA(1)), "cuda:1");
    }

    #[test]
    fn test_device_ffi_roundtrip() {
        let devices = [Device::CPU, Device::CUDA(0), Device::CUDA(1), Device::CUDA(7)];
        for dev in &devices {
            let (dt, di) = dev.to_ffi();
            let back = Device::from_ffi(dt, di);
            assert_eq!(*dev, back, "FFI roundtrip failed for {:?}", dev);
        }
    }

    #[test]
    fn test_device_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Device::CPU);
        set.insert(Device::CUDA(0));
        set.insert(Device::CUDA(1));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Device::CPU));
        assert!(set.contains(&Device::CUDA(0)));
        assert!(set.contains(&Device::CUDA(1)));
    }

    // --- Send + Sync compile-time checks ---

    #[test]
    fn test_tensor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor>();
    }

    /// Run with `cargo test manual_seed -- --test-threads=1 --ignored`
    /// (global RNG is shared across threads -- parallel tests consume state).
    #[test]
    #[ignore]
    fn test_manual_seed_reproducible() {
        let opts = test_opts();
        manual_seed(123);
        let a = Tensor::randn(&[4, 4], opts).unwrap().to_f32_vec().unwrap();
        manual_seed(123);
        let b = Tensor::randn(&[4, 4], opts).unwrap().to_f32_vec().unwrap();
        assert_eq!(a, b);
    }

    // --- fused adam tests ---

    #[test]
    fn test_fused_adamw_matches_batched() {
        // Run the same update with both implementations, verify results match
        let dev = test_device();
        let opts = test_opts();

        // Create two identical copies of params/moments
        manual_seed(42);
        let p1 = Tensor::randn(&[4, 3], opts).unwrap();
        let p2 = Tensor::from_f32(&p1.to_f32_vec().unwrap(), &[4, 3], dev).unwrap();
        let g = Tensor::randn(&[4, 3], opts).unwrap();
        let m1 = Tensor::zeros(&[4, 3], opts).unwrap();
        let m2 = Tensor::zeros(&[4, 3], opts).unwrap();
        let v1 = Tensor::zeros(&[4, 3], opts).unwrap();
        let v2 = Tensor::zeros(&[4, 3], opts).unwrap();

        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;

        // Batched (old path)
        p1.adam_step(&g, &m1, &v1, lr, beta1, beta2, eps, wd, 1).unwrap();

        // Fused (new path)
        Tensor::fused_adamw_(
            std::slice::from_ref(&p2), std::slice::from_ref(&g),
            std::slice::from_ref(&m2), std::slice::from_ref(&v2),
            lr, beta1, beta2, eps, wd, &[1], None, None,
        ).unwrap();

        let p1_data = p1.to_f32_vec().unwrap();
        let p2_data = p2.to_f32_vec().unwrap();
        for (i, (a, b)) in p1_data.iter().zip(&p2_data).enumerate() {
            assert!((a - b).abs() < 1e-5,
                "param mismatch at {}: batched={}, fused={}", i, a, b);
        }

        let m1_data = m1.to_f32_vec().unwrap();
        let m2_data = m2.to_f32_vec().unwrap();
        for (i, (a, b)) in m1_data.iter().zip(&m2_data).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "m mismatch at {}: batched={}, fused={}", i, a, b);
        }
    }

    #[test]
    fn test_fused_adam_no_weight_decay() {
        let opts = test_opts();
        let p = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let g = Tensor::from_f32(&[0.1, 0.2, 0.3, 0.4], &[4], test_device()).unwrap();
        let m = Tensor::zeros(&[4], opts).unwrap();
        let v = Tensor::zeros(&[4], opts).unwrap();

        Tensor::fused_adamw_(
            std::slice::from_ref(&p), std::slice::from_ref(&g),
            std::slice::from_ref(&m), std::slice::from_ref(&v),
            0.001, 0.9, 0.999, 1e-8, 0.0, &[1], None, None,
        ).unwrap();

        let p_data = p.to_f32_vec().unwrap();
        // Each param should decrease by ~lr
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        for (i, &o) in orig.iter().enumerate() {
            assert!((p_data[i] - (o - 0.001)).abs() < 1e-4,
                "p[{}]: got {}, expected ~{}", i, p_data[i], o - 0.001);
        }
    }

    #[test]
    fn test_fused_adam_multi_step() {
        let opts = test_opts();
        let p = Tensor::from_f32(&[5.0], &[1], test_device()).unwrap();
        let g = Tensor::from_f32(&[1.0], &[1], test_device()).unwrap();
        let m = Tensor::zeros(&[1], opts).unwrap();
        let v = Tensor::zeros(&[1], opts).unwrap();

        for step in 1..=10 {
            Tensor::fused_adamw_(
                std::slice::from_ref(&p), std::slice::from_ref(&g),
                std::slice::from_ref(&m), std::slice::from_ref(&v),
                0.01, 0.9, 0.999, 1e-8, 0.0, &[step], None, None,
            ).unwrap();
        }

        let p_data = p.to_f32_vec().unwrap();
        assert!(p_data[0] < 5.0, "param should decrease: got {}", p_data[0]);
        let m_data = m.to_f32_vec().unwrap();
        assert!((m_data[0] - 0.6513).abs() < 0.01,
            "m after 10 steps: got {}", m_data[0]);
    }

    #[test]
    fn test_fused_adam_empty_is_noop() {
        Tensor::fused_adamw_(&[], &[], &[], &[], 0.001, 0.9, 0.999, 1e-8, 0.0, &[], None, None).unwrap();
        Tensor::fused_adam_(&[], &[], &[], &[], 0.001, 0.9, 0.999, 1e-8, 0.0, &[], None, None).unwrap();
    }

    // --- foreach ops tests ---

    #[test]
    fn test_foreach_add_scalar() {
        let dev = test_device();
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0, 5.0], &[3], dev).unwrap();
        Tensor::foreach_add_scalar_(&[a.clone(), b.clone()], 10.0).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![11.0, 12.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_foreach_mul_scalar() {
        let dev = test_device();
        let a = Tensor::from_f32(&[2.0, 3.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &[2], dev).unwrap();
        Tensor::foreach_mul_scalar_(&[a.clone(), b.clone()], 0.5).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![1.0, 1.5]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![2.0, 2.5]);
    }

    #[test]
    fn test_foreach_zero() {
        let dev = test_device();
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[3.0, 4.0], &[2], dev).unwrap();
        Tensor::foreach_zero_(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![0.0, 0.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_foreach_add_list() {
        let dev = test_device();
        let a = Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[10.0, 20.0], &[2], dev).unwrap();
        let x = Tensor::from_f32(&[0.5, 0.5], &[2], dev).unwrap();
        let y = Tensor::from_f32(&[1.0, 1.0], &[2], dev).unwrap();
        // a += 2.0 * x, b += 2.0 * y
        Tensor::foreach_add_list_(
            &[a.clone(), b.clone()],
            &[x, y],
            2.0,
        ).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![2.0, 3.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![12.0, 22.0]);
    }

    #[test]
    fn test_foreach_norm() {
        let dev = test_device();
        let a = Tensor::from_f32(&[3.0, 4.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[1.0, 0.0], &[1, 2], dev).unwrap();
        let norms = Tensor::foreach_norm(&[a, b], 2.0).unwrap();
        assert_eq!(norms.len(), 2);
        let n0: f64 = norms[0].item().unwrap();
        let n1: f64 = norms[1].item().unwrap();
        assert!((n0 - 5.0).abs() < 1e-5, "norm of [3,4] should be 5, got {}", n0);
        assert!((n1 - 1.0).abs() < 1e-5, "norm of [1,0] should be 1, got {}", n1);
    }

    #[test]
    fn test_foreach_lerp_scalar() {
        let dev = test_device();
        let a = Tensor::from_f32(&[0.0, 10.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[10.0, 0.0], &[2], dev).unwrap();
        // a = a + 0.5 * (b_target - a), where b_target is the second list
        let a_target = Tensor::from_f32(&[10.0, 10.0], &[2], dev).unwrap();
        let b_target = Tensor::from_f32(&[10.0, 10.0], &[2], dev).unwrap();
        Tensor::foreach_lerp_scalar_(
            &[a.clone(), b.clone()],
            &[a_target, b_target],
            0.5,
        ).unwrap();
        // a = 0 + 0.5*(10-0) = 5, 10 + 0.5*(10-10) = 10
        assert_eq!(a.to_f32_vec().unwrap(), vec![5.0, 10.0]);
        // b = 10 + 0.5*(10-10) = 10, 0 + 0.5*(10-0) = 5
        assert_eq!(b.to_f32_vec().unwrap(), vec![10.0, 5.0]);
    }

    #[test]
    fn test_foreach_sqrt() {
        let dev = test_device();
        let a = Tensor::from_f32(&[4.0, 9.0], &[2], dev).unwrap();
        let b = Tensor::from_f32(&[16.0, 25.0], &[2], dev).unwrap();
        Tensor::foreach_sqrt_(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![2.0, 3.0]);
        assert_eq!(b.to_f32_vec().unwrap(), vec![4.0, 5.0]);
    }

    #[test]
    fn test_foreach_empty_list_is_noop() {
        // All foreach ops should handle empty lists gracefully
        Tensor::foreach_add_scalar_(&[], 1.0).unwrap();
        Tensor::foreach_mul_scalar_(&[], 1.0).unwrap();
        Tensor::foreach_zero_(&[]).unwrap();
        Tensor::foreach_add_list_(&[], &[], 1.0).unwrap();
        assert!(Tensor::foreach_norm(&[], 2.0).unwrap().is_empty());
        Tensor::foreach_lerp_scalar_(&[], &[], 0.5).unwrap();
        Tensor::foreach_sqrt_(&[]).unwrap();
    }

    #[test]
    fn foreach_list_length_mismatch_is_err_not_panic() {
        // Mismatched list lengths surface as Err (was an assert_eq! panic),
        // matching cat_many/stack — consistent fallible-op policy.
        let dev = test_device();
        let a = Tensor::ones(&[2], TensorOptions { device: dev, ..Default::default() }).unwrap();
        let b = Tensor::ones(&[2], TensorOptions { device: dev, ..Default::default() }).unwrap();
        assert!(Tensor::foreach_add_list_(&[a.clone(), b.clone()], std::slice::from_ref(&a), 1.0).is_err());
        assert!(Tensor::foreach_lerp_scalar_(std::slice::from_ref(&a), &[a.clone(), b.clone()], 0.5).is_err());
    }

    // --- Tier 2 creation ops ---

    #[test]
    fn test_full_like() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let fl = Tensor::full_like(&t, 7.0).unwrap();
        assert_eq!(fl.to_f32_vec().unwrap(), vec![7.0, 7.0, 7.0]);
        assert_eq!(fl.dtype(), DType::Float32);
    }

    #[test]
    fn test_rand_like_randn_like() {
        let t = Tensor::ones(&[3, 4], test_opts()).unwrap();
        let rl = Tensor::rand_like(&t).unwrap();
        assert_eq!(rl.shape(), vec![3, 4]);
        let data = rl.to_f32_vec().unwrap();
        // All values should be in [0, 1)
        assert!(data.iter().all(|&v| (0.0..1.0).contains(&v)));

        let nl = Tensor::randn_like(&t).unwrap();
        assert_eq!(nl.shape(), vec![3, 4]);
    }

    #[test]
    fn test_randint() {
        let mut opts = test_opts();
        opts.dtype = DType::Int64;
        let t = Tensor::randint(0, 10, &[100], opts).unwrap();
        assert_eq!(t.shape(), vec![100]);
        let data = t.to_i64_vec().unwrap();
        assert!(data.iter().all(|&v| (0..10).contains(&v)));
    }

    #[test]
    fn test_empty() {
        let t = Tensor::empty(&[2, 3], test_opts()).unwrap();
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.dtype(), DType::Float32);
    }

    #[test]
    fn test_one_hot() {
        let t = Tensor::from_i64(&[0, 1, 2], &[3], test_device()).unwrap();
        let oh = t.one_hot(4).unwrap();
        assert_eq!(oh.shape(), vec![3, 4]);
        let data = oh.to_f32_vec().unwrap();
        // class 0: [1, 0, 0, 0]
        assert_eq!(&data[0..4], &[1.0, 0.0, 0.0, 0.0]);
        // class 1: [0, 1, 0, 0]
        assert_eq!(&data[4..8], &[0.0, 1.0, 0.0, 0.0]);
        // class 2: [0, 0, 1, 0]
        assert_eq!(&data[8..12], &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_bernoulli() {
        let probs = Tensor::from_f32(&[0.0, 1.0, 0.0, 1.0], &[4], test_device()).unwrap();
        let samples = probs.bernoulli().unwrap();
        assert_eq!(samples.shape(), vec![4]);
        let data = samples.to_f32_vec().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
        assert!((data[2] - 0.0).abs() < 1e-5);
        assert!((data[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_is_contiguous() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        assert!(t.is_contiguous());
    }

    // --- Tier 2 in-place ops ---

    #[test]
    fn test_mul_inplace() {
        let a = Tensor::from_f32(&[2.0, 3.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &[2], test_device()).unwrap();
        a.mul_(&b).unwrap();
        assert_eq!(a.to_f32_vec().unwrap(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_div_scalar_inplace() {
        let t = Tensor::from_f32(&[6.0, 9.0], &[2], test_device()).unwrap();
        t.div_scalar_(3.0).unwrap();
        let data = t.to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_div_inplace() {
        let a = Tensor::from_f32(&[8.0, 15.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0], &[2], test_device()).unwrap();
        a.div_(&b).unwrap();
        let data = a.to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_fill_inplace() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        t.fill_(42.0).unwrap();
        assert_eq!(t.to_f32_vec().unwrap(), vec![42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_probe_device_cpu() {
        // CPU probe should always succeed
        assert!(probe_device(Device::CPU).is_ok());
    }

    #[test]
    #[ignore = "GPU probe needs CUDA; run with: fdl gpu-test-all"]
    fn test_probe_device_cuda() {
        if !test_device().is_cuda() { return; }
        // Device 0 should always work in a CUDA build
        assert!(probe_device(Device::CUDA(0)).is_ok());
    }

    #[test]
    #[ignore = "GPU diagnostics need CUDA; run with: fdl gpu-test-all"]
    fn test_cuda_devices_has_compute_capability() {
        if !test_device().is_cuda() { return; }
        let devices = gpu_devices();
        assert!(!devices.is_empty());
        for info in &devices {
            assert!(info.sm_major > 0, "compute capability should be detected");
            eprintln!("  CUDA({}) {} {} {:.1}GB",
                info.index, info.name, info.sm_version(),
                info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
        }
    }

    #[test]
    #[ignore = "GPU diagnostics need CUDA; run with: fdl gpu-test-all"]
    fn test_gpu_arch_name_is_vendor_shaped() {
        if !test_device().is_cuda() { return; }
        for i in 0..gpu_device_count() {
            let arch = gpu_arch_name(i).expect("arch name should resolve on a live device");
            eprintln!("  device {i}: {arch}");
            // Shape, not a hardcoded value: NVIDIA reports sm_<major><minor>,
            // AMD reports a gfx target (optionally with `:feature` suffixes).
            // Anything else means the FFI picked the wrong branch.
            assert!(
                arch.starts_with("sm_") || arch.starts_with("gfx"),
                "unexpected arch shape {arch:?}"
            );
            // The step digit is why this exists rather than a numeric pair,
            // so a bare family prefix is not enough.
            assert!(
                arch.len() > 3,
                "arch {arch:?} carries no architecture digits"
            );
        }
    }

    #[test]
    #[ignore = "GPU diagnostics need CUDA; run with: fdl gpu-test-all"]
    fn test_usable_cuda_devices() {
        if !test_device().is_cuda() { return; }
        let usable = usable_gpu_devices();
        assert!(!usable.is_empty(), "at least one device should be usable");
        // Device 0 should always be usable in a CUDA build
        assert!(usable.contains(&Device::CUDA(0)));
    }

    #[test]
    #[ignore = "GPU diagnostics need CUDA; run with: fdl gpu-test-all"]
    fn test_cuda_primary_context_and_nvml_probes() {
        if !test_device().is_cuda() { return; }
        // Force a real CUDA touch on runtime device 0, then the
        // context query must report it.
        let _t = Tensor::zeros(&[4], TensorOptions {
            dtype: DType::Float32,
            device: Device::CUDA(0),
        }).unwrap();
        assert!(gpu_has_primary_context(0),
            "primary context must exist after tensor work on the device");

        // NVML memory info takes PHYSICAL indices; resolve them via
        // sys::detect_gpus rather than assuming runtime == physical.
        let gpus = crate::sys::detect_gpus();
        assert!(!gpus.is_empty(), "detect_gpus should see the test GPU");
        for g in &gpus {
            let (used, total) = gpu_smi_memory_info_idx(g.index as i32)
                .expect("NVML memory info should be available on a CUDA rig");
            assert!(total > 0, "device-wide VRAM total should be non-zero");
            assert!(used <= total, "used VRAM cannot exceed total");
        }
    }

    #[test]
    fn test_cuda_has_primary_context_is_false_without_cuda_use() {
        // In a process that has done no CUDA work the query must
        // report false without erroring or initializing anything.
        // Only assertable where no other test can have touched CUDA
        // (CPU builds / CPU-only rigs); on CUDA rigs the parallel
        // harness makes context presence nondeterministic.
        if test_device().is_cuda() { return; }
        assert!(!gpu_has_primary_context(0));
    }
