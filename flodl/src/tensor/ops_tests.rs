    use super::super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 5.0, 6.0], &[3], test_device()).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], test_device()).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.to_f32_vec().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_chaining() {
        let a = Tensor::from_f32(&[1.0, -2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 1.0, 1.0], &[3], test_device()).unwrap();
        let result = a.add(&b).unwrap().relu().unwrap().sum().unwrap();
        // [1+1, -2+1, 3+1] = [2, -1, 4] -> relu -> [2, 0, 4] -> sum -> 6
        let val = result.item().unwrap();
        assert!((val - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_div_scalar() {
        let t = Tensor::from_f32(&[6.0, 9.0], &[2], test_device()).unwrap();
        let r = t.div_scalar(3.0).unwrap();
        let data = r.to_f32_vec().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], test_device()).unwrap();
        let m = t.mean().unwrap();
        assert!((m.item().unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_sub_mul_div() {
        let a = Tensor::from_f32(&[6.0, 8.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[2.0, 3.0], &[2], test_device()).unwrap();
        assert_eq!(a.sub(&b).unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
        assert_eq!(a.mul(&b).unwrap().to_f32_vec().unwrap(), vec![12.0, 24.0]);
        let d = a.div(&b).unwrap().to_f32_vec().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-5);
        assert!((d[1] - 8.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_ops() {
        let t = Tensor::from_f32(&[2.0, 4.0], &[2], test_device()).unwrap();
        assert_eq!(t.add_scalar(1.0).unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);
        assert_eq!(t.mul_scalar(3.0).unwrap().to_f32_vec().unwrap(), vec![6.0, 12.0]);
        assert_eq!(t.neg().unwrap().to_f32_vec().unwrap(), vec![-2.0, -4.0]);
    }

    #[test]
    fn test_exp_log_sqrt_abs_pow() {
        let t = Tensor::from_f32(&[1.0, 4.0], &[2], test_device()).unwrap();
        let e = t.exp().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 1.0_f32.exp()).abs() < 1e-5);

        let l = t.log().unwrap().to_f32_vec().unwrap();
        assert!((l[1] - 4.0_f32.ln()).abs() < 1e-5);

        let s = t.sqrt().unwrap().to_f32_vec().unwrap();
        assert!((s[1] - 2.0).abs() < 1e-5);

        let a = Tensor::from_f32(&[-3.0, 5.0], &[2], test_device()).unwrap();
        assert_eq!(a.abs().unwrap().to_f32_vec().unwrap(), vec![3.0, 5.0]);

        let p = t.pow_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert!((p[0] - 1.0).abs() < 1e-5);
        assert!((p[1] - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_clamp() {
        let t = Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], test_device()).unwrap();
        let c = t.clamp(0.0, 1.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(c, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_sum_dim_mean_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let s = t.sum_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert_eq!(s, vec![3.0, 7.0]);

        let m = t.mean_dim(0, false).unwrap().to_f32_vec().unwrap();
        assert!((m[0] - 2.0).abs() < 1e-5);
        assert!((m[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm() {
        let t = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let n = t.norm().unwrap().item().unwrap();
        assert!((n - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        assert_eq!(t.relu().unwrap().to_f32_vec().unwrap(), vec![0.0, 0.0, 1.0]);

        let sig = t.sigmoid().unwrap().to_f32_vec().unwrap();
        assert!((sig[2] - 0.7310586).abs() < 1e-5);

        let th = t.tanh().unwrap().to_f32_vec().unwrap();
        assert!((th[2] - 1.0_f32.tanh()).abs() < 1e-5);

        // gelu/silu just check they don't crash and return right shape
        assert_eq!(t.gelu().unwrap().shape(), vec![3]);
        assert_eq!(t.silu().unwrap().shape(), vec![3]);

        // gelu_tanh: same shape, and the two forms should produce close
        // but not identical values — bitwise equality would mean the FFI
        // dispatched to the wrong libtorch entry point.
        let g_erf  = t.gelu().unwrap().to_f32_vec().unwrap();
        let g_tanh = t.gelu_tanh().unwrap().to_f32_vec().unwrap();
        assert_eq!(g_tanh.len(), 3);
        let close: bool = g_erf.iter().zip(&g_tanh).all(|(a, b)| (a - b).abs() < 1e-2);
        let exact: bool = g_erf.iter().zip(&g_tanh).all(|(a, b)| (a - b).abs() < 1e-7);
        assert!(close && !exact, "erf vs tanh GELU: erf={g_erf:?}, tanh={g_tanh:?}");
    }

    #[test]
    fn test_softmax_log_softmax() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let sm = t.softmax(0).unwrap().to_f32_vec().unwrap();
        let total: f32 = sm.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);

        let lsm = t.log_softmax(0).unwrap().to_f32_vec().unwrap();
        assert!(lsm[0] < 0.0 && lsm[1] < 0.0 && lsm[2] < 0.0);
    }

    #[test]
    fn test_eq_ne_tensor() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 5.0, 3.0], &[3], test_device()).unwrap();

        let eq = a.eq_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(eq, vec![1.0, 0.0, 1.0]);

        let ne = a.ne_tensor(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(ne, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt_lt_ge_le_tensor() {
        let a = Tensor::from_f32(&[1.0, 3.0, 2.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[2.0, 2.0, 2.0], &[3], test_device()).unwrap();

        assert_eq!(a.gt(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(a.lt(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 0.0]);
        assert_eq!(a.ge(&b).unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0, 1.0]);
        assert_eq!(a.le(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sign_floor_ceil_round() {
        let t = Tensor::from_f32(&[-2.7, 0.0, 1.3], &[3], test_device()).unwrap();
        assert_eq!(t.sign().unwrap().to_f32_vec().unwrap(), vec![-1.0, 0.0, 1.0]);
        assert_eq!(t.floor().unwrap().to_f32_vec().unwrap(), vec![-3.0, 0.0, 1.0]);
        assert_eq!(t.ceil().unwrap().to_f32_vec().unwrap(), vec![-2.0, 0.0, 2.0]);

        let r = Tensor::from_f32(&[-0.6, 0.4, 1.5], &[3], test_device()).unwrap();
        let rv = r.round().unwrap().to_f32_vec().unwrap();
        assert!((rv[0] - (-1.0)).abs() < 1e-5);
        assert!((rv[1] - 0.0).abs() < 1e-5);
        assert!((rv[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmin() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let idx = t.argmin(0, false).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn test_var_std() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        // Bessel: var = ((1-2)^2+(2-2)^2+(3-2)^2)/2 = 1.0
        assert!((t.var().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
        assert!((t.std().unwrap().item().unwrap() - 1.0).abs() < 1e-5);

        // dim variant: [[1,2],[3,4]] var along dim=1 = [0.5, 0.5]
        let t2 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let vd = t2.var_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert!((vd[0] - 0.5).abs() < 1e-5);
        assert!((vd[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sin_cos_reciprocal() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let s = t.sin().unwrap().to_f32_vec().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-5);
        assert!((s[1] - 1.0_f32.sin()).abs() < 1e-5);

        let c = t.cos().unwrap().to_f32_vec().unwrap();
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 1.0_f32.cos()).abs() < 1e-5);

        let r = Tensor::from_f32(&[2.0, 5.0], &[2], test_device()).unwrap();
        let rec = r.reciprocal().unwrap().to_f32_vec().unwrap();
        assert!((rec[0] - 0.5).abs() < 1e-5);
        assert!((rec[1] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_gather_scatter_add() {
        // gather: pick elements by index
        let t = Tensor::from_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2], test_device()).unwrap();
        let idx = Tensor::from_i64(&[1, 0, 0, 1], &[2, 2], test_device()).unwrap();
        let g = t.gather(1, &idx).unwrap().to_f32_vec().unwrap();
        assert_eq!(g, vec![20.0, 10.0, 30.0, 40.0]);

        // scatter_add: accumulate into base at positions
        let base = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let src = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let idx2 = Tensor::from_i64(&[0, 2, 1, 0], &[2, 2], test_device()).unwrap();
        let sa = base.scatter_add(1, &idx2, &src).unwrap();
        let data = sa.to_f32_vec().unwrap();
        // Row 0: pos 0 += 1.0, pos 2 += 2.0 -> [1, 0, 2]
        // Row 1: pos 1 += 3.0, pos 0 += 4.0 -> [4, 3, 0]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
        assert!((data[4] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_topk_sort() {
        let t = Tensor::from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], test_device()).unwrap();
        let (vals, idxs) = t.topk(3, 0, true, true).unwrap();
        assert_eq!(vals.to_f32_vec().unwrap(), vec![5.0, 4.0, 3.0]);
        let idx_data = idxs.to_i64_vec().unwrap();
        assert_eq!(idx_data, vec![4, 2, 0]);

        let (svals, sidxs) = t.sort(0, false).unwrap();
        assert_eq!(svals.to_f32_vec().unwrap(), vec![1.0, 1.0, 3.0, 4.0, 5.0]);
        let si = sidxs.to_i64_vec().unwrap();
        assert_eq!(si[4], 4); // 5.0 was at index 4
    }

    #[test]
    fn test_masked_fill() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let mask = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], test_device()).unwrap();
        let filled = t.masked_fill(&mask, -1e9).unwrap().to_f32_vec().unwrap();
        assert!(filled[0] < -1e8); // masked
        assert!((filled[1] - 2.0).abs() < 1e-5); // kept
        assert!((filled[2] - 3.0).abs() < 1e-5); // kept
        assert!(filled[3] < -1e8); // masked
    }

    #[test]
    fn test_tril() {
        let t = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3], test_device(),
        ).unwrap();
        let lo = t.tril(0).unwrap().to_f32_vec().unwrap();
        assert_eq!(lo, vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_prod() {
        let t = Tensor::from_f32(&[2.0, 3.0, 4.0], &[3], test_device()).unwrap();
        let p = t.prod().unwrap().item().unwrap();
        assert!((p - 24.0).abs() < 1e-4);
    }

    #[test]
    fn test_prod_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let p = t.prod_dim(1, false).unwrap().to_f32_vec().unwrap();
        assert!((p[0] - 2.0).abs() < 1e-4);
        assert!((p[1] - 12.0).abs() < 1e-4);
    }

    #[test]
    fn test_cumsum() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let c = t.cumsum(1).unwrap().to_f32_vec().unwrap();
        assert_eq!(c, vec![1.0, 3.0, 3.0, 7.0]);
    }

    #[test]
    fn test_logsumexp() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let lse = t.logsumexp(0, false).unwrap().item().unwrap();
        // log(e^1 + e^2 + e^3) ~ 3.4076
        assert!((lse - 3.4076).abs() < 1e-3);
    }

    #[test]
    fn test_multinomial() {
        let probs = Tensor::from_f32(&[0.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let samples = probs.multinomial(2, true).unwrap();
        // All probability mass on index 2 -- both samples must be 2.
        let vals = samples.to_i64_vec().unwrap();
        assert_eq!(vals, vec![2, 2]);
    }

    #[test]
    fn test_normalize() {
        let t = Tensor::from_f32(&[3.0, 4.0], &[2], test_device()).unwrap();
        let n = t.normalize(2.0, 0).unwrap().to_f32_vec().unwrap();
        // L2 norm is 5, so [0.6, 0.8]
        assert!((n[0] - 0.6).abs() < 1e-5);
        assert!((n[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu() {
        let t = Tensor::from_f32(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], test_device()).unwrap();
        let r = t.leaky_relu(0.1).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - (-0.2)).abs() < 1e-5);
        assert!((r[1] - (-0.1)).abs() < 1e-5);
        assert!((r[2] - 0.0).abs() < 1e-5);
        assert!((r[3] - 1.0).abs() < 1e-5);
        assert!((r[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_elu() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let r = t.elu(1.0).unwrap().to_f32_vec().unwrap();
        // ELU(-1) = 1*(exp(-1)-1) ~ -0.6321
        assert!((r[0] - (-0.6321)).abs() < 1e-3);
        assert!((r[1] - 0.0).abs() < 1e-5);
        assert!((r[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softplus() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let r = t.softplus(1.0, 20.0).unwrap().to_f32_vec().unwrap();
        // softplus(0) = ln(2)
        assert!((r[1] - std::f32::consts::LN_2).abs() < 1e-3);
        // softplus(x) > 0 for all x
        assert!(r[0] > 0.0);
    }

    #[test]
    fn test_mish() {
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let r = t.mish().unwrap().to_f32_vec().unwrap();
        // mish(0) = 0 * tanh(softplus(0)) = 0
        assert!((r[1] - 0.0).abs() < 1e-5);
        // mish(1) ~ 0.8651
        assert!((r[2] - 0.8651).abs() < 1e-3);
    }

    #[test]
    fn test_cdist() {
        // Two 2D points: [0,0] and [3,4] -> distance = 5
        let x = Tensor::from_f32(&[0.0, 0.0], &[1, 1, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[3.0, 4.0], &[1, 1, 2], test_device()).unwrap();
        let d = x.cdist(&y).unwrap();
        assert_eq!(d.shape(), vec![1, 1, 1]);
        assert!((d.item().unwrap() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_cdist_p1() {
        // L1: |3| + |4| = 7
        let x = Tensor::from_f32(&[0.0, 0.0], &[1, 1, 2], test_device()).unwrap();
        let y = Tensor::from_f32(&[3.0, 4.0], &[1, 1, 2], test_device()).unwrap();
        let d = x.cdist_p(&y, 1.0).unwrap();
        assert!((d.item().unwrap() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_clamp_min_max() {
        let t = Tensor::from_f32(&[-2.0, 0.5, 3.0], &[3], test_device()).unwrap();
        let cmin = t.clamp_min(0.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(cmin, vec![0.0, 0.5, 3.0]);
        let cmax = t.clamp_max(1.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(cmax, vec![-2.0, 0.5, 1.0]);
    }

    #[test]
    fn test_log1p_expm1() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let l = t.log1p().unwrap().to_f32_vec().unwrap();
        assert!((l[0] - 0.0).abs() < 1e-5); // log(1+0) = 0
        assert!((l[1] - 2.0_f32.ln()).abs() < 1e-5); // log(1+1) = ln(2)

        let e = t.expm1().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 0.0).abs() < 1e-5); // exp(0)-1 = 0
        assert!((e[1] - (1.0_f32.exp() - 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_log2_log10() {
        let t = Tensor::from_f32(&[1.0, 8.0, 100.0], &[3], test_device()).unwrap();
        let l2 = t.log2().unwrap().to_f32_vec().unwrap();
        assert!((l2[0] - 0.0).abs() < 1e-5);
        assert!((l2[1] - 3.0).abs() < 1e-4);

        let l10 = t.log10().unwrap().to_f32_vec().unwrap();
        assert!((l10[0] - 0.0).abs() < 1e-5);
        assert!((l10[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_eq_ne_scalar() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let eq = t.eq_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(eq, vec![0.0, 1.0, 0.0]);
        let ne = t.ne_scalar(2.0).unwrap().to_f32_vec().unwrap();
        assert_eq!(ne, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isnan_isinf() {
        let t = Tensor::from_f32(&[1.0, f32::NAN, f32::INFINITY], &[3], test_device()).unwrap();
        let nan = t.isnan().unwrap().to_f32_vec().unwrap();
        assert_eq!(nan, vec![0.0, 1.0, 0.0]);
        let inf = t.isinf().unwrap().to_f32_vec().unwrap();
        assert_eq!(inf, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_logical_ops() {
        let a = Tensor::from_f32(&[1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[0.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let and = a.logical_and(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(and, vec![0.0, 0.0, 1.0]);
        let or = a.logical_or(&b).unwrap().to_f32_vec().unwrap();
        assert_eq!(or, vec![1.0, 0.0, 1.0]);
        let not = a.logical_not().unwrap().to_f32_vec().unwrap();
        assert_eq!(not, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_any_all() {
        let t = Tensor::from_f32(&[0.0, 0.0, 1.0], &[3], test_device()).unwrap();
        assert!((t.any().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
        assert!((t.all().unwrap().item().unwrap() - 0.0).abs() < 1e-5);

        let all_true = Tensor::from_f32(&[1.0, 1.0], &[2], test_device()).unwrap();
        assert!((all_true.all().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_atan2() {
        let y = Tensor::from_f32(&[1.0, 0.0], &[2], test_device()).unwrap();
        let x = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let result = y.atan2(&x).unwrap().to_f32_vec().unwrap();
        // atan2(1, 0) = pi/2
        assert!((result[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        // atan2(0, 1) = 0
        assert!((result[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_maximum_minimum() {
        let a = Tensor::from_f32(&[1.0, 5.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[4.0, 2.0, 3.0], &[3], test_device()).unwrap();
        assert_eq!(a.maximum(&b).unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0, 3.0]);
        assert_eq!(a.minimum(&b).unwrap().to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_argsort() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let idx = t.argsort(0, false).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1, 2, 0]); // ascending: 1.0(1), 2.0(2), 3.0(0)
    }

    #[test]
    fn test_scatter() {
        let base = Tensor::zeros(&[2, 3], test_opts()).unwrap();
        let idx = Tensor::from_i64(&[0, 2, 1, 0], &[2, 2], test_device()).unwrap();
        let src = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let result = base.scatter(1, &idx, &src).unwrap().to_f32_vec().unwrap();
        // Row 0: pos 0 = 1.0, pos 2 = 2.0 -> [1, 0, 2]
        // Row 1: pos 1 = 3.0, pos 0 = 4.0 -> [4, 3, 0]
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[2] - 2.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
        assert!((result[4] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_randperm() {
        let mut opts = test_opts();
        opts.dtype = DType::Int64;
        let p = Tensor::randperm(5, opts).unwrap();
        assert_eq!(p.shape(), vec![5]);
        // All values 0..5 must be present (it's a permutation).
        let mut vals = p.to_i64_vec().unwrap();
        vals.sort();
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_tan() {
        let t = Tensor::from_f32(&[0.0, std::f32::consts::FRAC_PI_4], &[2], test_device()).unwrap();
        let r = t.tan().unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 0.0).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_asin_acos_atan() {
        let t = Tensor::from_f32(&[0.0, 0.5, 1.0], &[3], test_device()).unwrap();
        let as_ = t.asin().unwrap().to_f32_vec().unwrap();
        assert!((as_[0] - 0.0).abs() < 1e-5);
        assert!((as_[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-3);

        let ac = t.acos().unwrap().to_f32_vec().unwrap();
        assert!((ac[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert!((ac[2] - 0.0).abs() < 1e-5);

        let at = t.atan().unwrap().to_f32_vec().unwrap();
        assert!((at[0] - 0.0).abs() < 1e-5);
        assert!((at[2] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    }

    #[test]
    fn test_erf_erfc() {
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let e = t.erf().unwrap().to_f32_vec().unwrap();
        assert!((e[0] - 0.0).abs() < 1e-5);
        assert!((e[1] - 0.8427).abs() < 1e-3);

        let ec = t.erfc().unwrap().to_f32_vec().unwrap();
        assert!((ec[0] - 1.0).abs() < 1e-5);
        assert!((ec[1] - 0.1573).abs() < 1e-3);
    }

    #[test]
    fn test_trunc_frac() {
        let t = Tensor::from_f32(&[2.7, -1.3], &[2], test_device()).unwrap();
        let tr = t.trunc().unwrap().to_f32_vec().unwrap();
        assert!((tr[0] - 2.0).abs() < 1e-5);
        assert!((tr[1] - (-1.0)).abs() < 1e-5);

        let fr = t.frac().unwrap().to_f32_vec().unwrap();
        assert!((fr[0] - 0.7).abs() < 1e-5);
        assert!((fr[1] - (-0.3)).abs() < 1e-5);
    }

    #[test]
    fn test_fmod() {
        let t = Tensor::from_f32(&[5.0, -5.0, 7.5], &[3], test_device()).unwrap();
        let r = t.fmod(3.0).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - (-2.0)).abs() < 1e-5);
        assert!((r[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_remainder() {
        let t = Tensor::from_f32(&[5.0, -5.0], &[2], test_device()).unwrap();
        let r = t.remainder(3.0).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-5); // Python semantics: sign matches divisor
    }

    #[test]
    fn test_lerp() {
        let a = Tensor::from_f32(&[0.0, 10.0], &[2], test_device()).unwrap();
        let b = Tensor::from_f32(&[10.0, 20.0], &[2], test_device()).unwrap();
        let r = a.lerp(&b, 0.3).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 3.0).abs() < 1e-5);
        assert!((r[1] - 13.0).abs() < 1e-5);
    }

    #[test]
    fn test_isclose() {
        let a = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], test_device()).unwrap();
        let b = Tensor::from_f32(&[1.0, 2.001, 5.0], &[3], test_device()).unwrap();
        let r = a.isclose(&b, 1e-5, 1e-2).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5); // exact match
        assert!((r[1] - 1.0).abs() < 1e-5); // within atol=0.01
        assert!((r[2] - 0.0).abs() < 1e-5); // not close
    }

    #[test]
    fn test_addmm() {
        let bias = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let m1 = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let m2 = Tensor::from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2], test_device()).unwrap();
        // 1.0 * bias + 1.0 * (m1 @ m2) = bias + m1 (identity)
        let r = bias.addmm(&m1, &m2, 1.0, 1.0).unwrap().to_f32_vec().unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5); // 1 + 1
        assert!((r[1] - 4.0).abs() < 1e-5); // 2 + 2
        assert!((r[2] - 4.0).abs() < 1e-5); // 1 + 3
        assert!((r[3] - 6.0).abs() < 1e-5); // 2 + 4
    }

    #[test]
    fn test_addcmul_addcdiv() {
        let s = Tensor::from_f32(&[1.0, 1.0], &[2], test_device()).unwrap();
        let t1 = Tensor::from_f32(&[2.0, 3.0], &[2], test_device()).unwrap();
        let t2 = Tensor::from_f32(&[4.0, 5.0], &[2], test_device()).unwrap();

        // addcmul: 1 + 0.5 * (2*4) = 5, 1 + 0.5 * (3*5) = 8.5
        let cm = s.addcmul(&t1, &t2, 0.5).unwrap().to_f32_vec().unwrap();
        assert!((cm[0] - 5.0).abs() < 1e-5);
        assert!((cm[1] - 8.5).abs() < 1e-5);

        // addcdiv: 1 + 0.5 * (2/4) = 1.25, 1 + 0.5 * (3/5) = 1.3
        let cd = s.addcdiv(&t1, &t2, 0.5).unwrap().to_f32_vec().unwrap();
        assert!((cd[0] - 1.25).abs() < 1e-5);
        assert!((cd[1] - 1.3).abs() < 1e-5);
    }

    #[test]
    fn test_cumprod() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], test_device()).unwrap();
        let r = t.cumprod(0).unwrap().to_f32_vec().unwrap();
        assert_eq!(r, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_norm_p_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        // L1 norm along dim 1: [3.0, 7.0]
        let l1 = t.norm_p(1.0, 1, false).unwrap().to_f32_vec().unwrap();
        assert!((l1[0] - 3.0).abs() < 1e-5);
        assert!((l1[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_dims() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], test_device()).unwrap();
        // Sum over both dims should give scalar 21
        let s = t.sum_dims(&[0, 1], false).unwrap();
        assert!((s.item().unwrap() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn test_median() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0], &[3], test_device()).unwrap();
        let m = t.median().unwrap().item().unwrap();
        assert!((m - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_median_dim() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], test_device()).unwrap();
        let (vals, idxs) = t.median_dim(1, false).unwrap();
        let v = vals.to_f32_vec().unwrap();
        let i = idxs.to_i64_vec().unwrap();
        assert!((v[0] - 2.0).abs() < 1e-5);
        assert!((v[1] - 5.0).abs() < 1e-5);
        assert_eq!(i[0], 2);
        assert_eq!(i[1], 2);
    }

    #[test]
    fn test_count_nonzero() {
        let t = Tensor::from_f32(&[0.0, 1.0, 0.0, 2.0, 3.0], &[5], test_device()).unwrap();
        let c = t.count_nonzero().unwrap().to_i64_vec().unwrap();
        assert_eq!(c[0], 3);
    }

    #[test]
    fn test_nonzero() {
        let t = Tensor::from_f32(&[0.0, 1.0, 0.0, 2.0], &[4], test_device()).unwrap();
        let nz = t.nonzero().unwrap();
        assert_eq!(nz.shape(), vec![2, 1]); // 2 non-zero entries, 1D
        let vals = nz.to_i64_vec().unwrap();
        assert_eq!(vals, vec![1, 3]);
    }

    #[test]
    fn test_unique() {
        let t = Tensor::from_f32(&[3.0, 1.0, 2.0, 1.0, 3.0], &[5], test_device()).unwrap();
        let (u, inv) = t.unique(true, true).unwrap();
        let uv = u.to_f32_vec().unwrap();
        assert_eq!(uv, vec![1.0, 2.0, 3.0]);
        let iv = inv.to_i64_vec().unwrap();
        // 3->2, 1->0, 2->1, 1->0, 3->2
        assert_eq!(iv, vec![2, 0, 1, 0, 2]);
    }

    #[test]
    fn test_searchsorted() {
        let sorted = Tensor::from_f32(&[1.0, 3.0, 5.0, 7.0], &[4], test_device()).unwrap();
        let vals = Tensor::from_f32(&[2.0, 4.0, 6.0], &[3], test_device()).unwrap();
        let idx = sorted.searchsorted(&vals).unwrap().to_i64_vec().unwrap();
        assert_eq!(idx, vec![1, 2, 3]);
    }
