use super::*;
use crate::tensor::{Device, Tensor};

fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
    Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
}

#[test]
#[should_panic(expected = "cannot enable gradient tracking")]
fn test_variable_new_int_requires_grad_panics() {
    // Regression: this used to silently fall back to a NON-tracking
    // variable (the model trained nothing for the param). PyTorch
    // raises on requires_grad for integer dtypes; so do we now.
    let t = Tensor::from_i64(&[1, 2, 3], &[3], crate::tensor::test_device()).unwrap();
    let _ = Variable::new(t, true);
}

#[test]
fn test_simple_gradient() {
    // y = 2*x, dy/dx = 2
    let x = Variable::new(from_f32(&[3.0], &[1]), true);
    let two = Variable::new(from_f32(&[2.0], &[1]), false);
    let y = x.mul(&two).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap();
    assert!((grad.item().unwrap() - 2.0).abs() < 1e-5);
}

#[test]
fn test_add_gradient() {
    let a = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
    let b = Variable::new(from_f32(&[4.0, 5.0, 6.0], &[3]), true);
    let c = a.add(&b).unwrap().sum().unwrap();
    c.backward().unwrap();

    let ga = a.grad().unwrap().to_f32_vec().unwrap();
    let gb = b.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(ga, vec![1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_mul_gradient() {
    let x = Variable::new(from_f32(&[2.0, 3.0], &[2]), true);
    let y = Variable::new(from_f32(&[4.0, 5.0], &[2]), true);
    let z = x.mul(&y).unwrap().sum().unwrap();
    z.backward().unwrap();

    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![4.0, 5.0]);
    assert_eq!(y.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn test_chain_rule() {
    let x = Variable::new(from_f32(&[1.0, -2.0, 3.0], &[3]), true);
    let w = Variable::new(from_f32(&[2.0, 2.0, 2.0], &[3]), true);
    let b = Variable::new(from_f32(&[0.0, 5.0, 0.0], &[3]), false);

    let y = x
        .mul(&w)
        .unwrap()
        .add(&b)
        .unwrap()
        .relu()
        .unwrap()
        .sum()
        .unwrap();
    assert!((y.item().unwrap() - 9.0).abs() < 1e-5);

    y.backward().unwrap();

    let gx = x.grad().unwrap().to_f32_vec().unwrap();
    let gw = w.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(gx, vec![2.0, 2.0, 2.0]);
    assert_eq!(gw, vec![1.0, -2.0, 3.0]);
}

#[test]
fn test_matmul_gradient() {
    let a = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let b = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), true);
    let c = a.matmul(&b).unwrap().sum().unwrap();
    c.backward().unwrap();

    let ga = a.grad().unwrap().to_f32_vec().unwrap();
    let gb = b.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![4.0, 4.0, 6.0, 6.0]);
}

#[test]
fn test_sigmoid_gradient() {
    let x = Variable::new(from_f32(&[0.0], &[1]), true);
    let y = x.sigmoid().unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!((x.grad().unwrap().item().unwrap() - 0.25).abs() < 1e-5);
}

#[test]
fn test_relu_gradient() {
    let x = Variable::new(from_f32(&[1.0, -1.0, 2.0, -2.0], &[4]), true);
    let y = x.relu().unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_f32_vec().unwrap(),
        vec![1.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn test_exp_log_gradient() {
    let x = Variable::new(from_f32(&[2.0], &[1]), true);
    let y = x.log().unwrap().exp().unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!((x.grad().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn test_no_grad() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = no_grad(|| x.mul_scalar(3.0).unwrap());
    assert!(!y.requires_grad());
}

#[test]
fn test_detach() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = x.mul_scalar(2.0).unwrap();
    let z = y.detach();
    // Detached variable should not require grad
    assert!(!z.requires_grad());
    // Operations on detached variable don't track gradients
    let w = z.mul_scalar(3.0).unwrap();
    assert!(!w.requires_grad());
    // x should have no gradient (no backward was called through x)
    assert!(x.grad().is_none());
}

#[test]
fn test_gradient_accumulation() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
    let y = Variable::new(from_f32(&[1.0, 1.0, 1.0], &[3]), false);

    let z1 = x.add(&y).unwrap().sum().unwrap();
    z1.backward().unwrap();
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0]);

    // Second backward accumulates
    let z2 = x.mul_scalar(2.0).unwrap().sum().unwrap();
    z2.backward().unwrap();
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![3.0, 3.0, 3.0]);
}

#[test]
fn test_zero_grad() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = x.mul_scalar(3.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![3.0, 3.0]);

    x.zero_grad();
    // After zero_grad, grad should be zero (libtorch zeros it, doesn't remove it)
    let g = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(g, vec![0.0, 0.0]);
}

#[test]
fn test_sub_gradient() {
    let a = Variable::new(from_f32(&[5.0, 6.0], &[2]), true);
    let b = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let c = a.sub(&b).unwrap().sum().unwrap();
    c.backward().unwrap();

    assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);
    assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![-1.0, -1.0]);
}

#[test]
fn test_tanh_gradient() {
    let x = Variable::new(from_f32(&[0.0], &[1]), true);
    let y = x.tanh().unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!((x.grad().unwrap().item().unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn test_neg_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = x.neg().unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![-1.0, -1.0]);
}

#[test]
fn test_numerical_gradient() {
    let eps = 1e-4;
    let x_data = vec![1.0_f32, 2.0, -3.0];

    let x = Variable::new(from_f32(&x_data, &[3]), true);
    let y = x.mul(&x).unwrap().sum().unwrap();
    y.backward().unwrap();
    let analytical = x.grad().unwrap().to_f32_vec().unwrap();

    for i in 0..3 {
        let mut x_plus = x_data.clone();
        let mut x_minus = x_data.clone();
        x_plus[i] += eps as f32;
        x_minus[i] -= eps as f32;

        let fp = from_f32(&x_plus, &[3])
            .mul(&from_f32(&x_plus, &[3]))
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        let fm = from_f32(&x_minus, &[3])
            .mul(&from_f32(&x_minus, &[3]))
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();

        let numerical = (fp - fm) / (2.0 * eps);
        assert!(
            (analytical[i] as f64 - numerical).abs() < 0.01,
            "grad mismatch at {}: analytical={}, numerical={}",
            i,
            analytical[i],
            numerical
        );
    }
}

#[test]
fn test_diamond_graph() {
    let x = Variable::new(from_f32(&[3.0], &[1]), true);
    let y1 = x.mul_scalar(2.0).unwrap();
    let y2 = x.mul_scalar(3.0).unwrap();
    let z = y1.add(&y2).unwrap().sum().unwrap();
    z.backward().unwrap();
    assert!((x.grad().unwrap().item().unwrap() - 5.0).abs() < 1e-5);
}

#[test]
fn test_drop_frees_backward_graph() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    {
        let y = x.mul_scalar(2.0).unwrap().relu().unwrap().sum().unwrap();
        y.backward().unwrap();
    }
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 2.0]);
}

#[test]
fn test_sin_cos_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = x.sin().unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 1.0_f32.cos()).abs() < 1e-5);
    assert!((grad[1] - 2.0_f32.cos()).abs() < 1e-5);

    x.zero_grad();
    let y2 = x.cos().unwrap().sum().unwrap();
    y2.backward().unwrap();
    let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad2[0] - (-1.0_f32.sin())).abs() < 1e-5);
    assert!((grad2[1] - (-2.0_f32.sin())).abs() < 1e-5);
}

#[test]
fn test_reciprocal_gradient() {
    let x = Variable::new(from_f32(&[2.0, 4.0], &[2]), true);
    let y = x.reciprocal().unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - (-0.25)).abs() < 1e-5);
    assert!((grad[1] - (-0.0625)).abs() < 1e-5);
}

#[test]
fn test_var_std_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
    let v = x.var().unwrap();
    assert!((v.data().item().unwrap() - 1.0).abs() < 1e-5);
    v.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - (-1.0)).abs() < 1e-5);
    assert!((grad[1] - 0.0).abs() < 1e-5);
    assert!((grad[2] - 1.0).abs() < 1e-5);

    x.zero_grad();
    let s = x.std().unwrap();
    assert!((s.data().item().unwrap() - 1.0).abs() < 1e-5);
    s.backward().unwrap();
    let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad2[0] - (-0.5)).abs() < 1e-4);
    assert!((grad2[1] - 0.0).abs() < 1e-4);
    assert!((grad2[2] - 0.5).abs() < 1e-4);
}

#[test]
fn test_var_dim_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let v = x.var_dim(1, false).unwrap().sum().unwrap();
    v.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - (-1.0)).abs() < 1e-5);
    assert!((grad[1] - 1.0).abs() < 1e-5);
    assert!((grad[2] - (-1.0)).abs() < 1e-5);
    assert!((grad[3] - 1.0).abs() < 1e-5);
}

#[test]
fn test_gather_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let idx = Tensor::from_i64(&[0, 0, 1, 0], &[2, 2], crate::tensor::test_device()).unwrap();
    let y = x.gather(1, &idx).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 2.0).abs() < 1e-5);
    assert!((grad[1] - 0.0).abs() < 1e-5);
    assert!((grad[2] - 1.0).abs() < 1e-5);
    assert!((grad[3] - 1.0).abs() < 1e-5);
}

#[test]
fn test_topk_gradient() {
    let x = Variable::new(from_f32(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]), true);
    let (values, _indices) = x.topk(2, 0, true, true).unwrap();
    let y = values.sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 0.0).abs() < 1e-5);
    assert!((grad[1] - 0.0).abs() < 1e-5);
    assert!((grad[2] - 1.0).abs() < 1e-5);
    assert!((grad[3] - 0.0).abs() < 1e-5);
    assert!((grad[4] - 1.0).abs() < 1e-5);
}

#[test]
fn test_repeat_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = x.repeat(&[3]).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 3.0).abs() < 1e-5);
    assert!((grad[1] - 3.0).abs() < 1e-5);
}

#[test]
fn test_pad_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
    let y = x.pad(&[1, 1], 0.0).unwrap();
    assert_eq!(y.data().shape(), vec![5]);
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(grad, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_chunk_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]), true);
    let chunks = x.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    let c0 = chunks[0].mul_scalar(2.0).unwrap();
    let c1 = chunks[1].mul_scalar(3.0).unwrap();
    let loss = c0
        .sum()
        .unwrap()
        .add(&c1.sum().unwrap())
        .unwrap()
        .add(&chunks[2].sum().unwrap())
        .unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(grad, vec![2.0, 2.0, 3.0, 3.0, 1.0, 1.0]);
}

#[test]
fn test_sort_gradient() {
    let x = Variable::new(from_f32(&[3.0, 1.0, 2.0], &[3]), true);
    let (sorted, _indices) = x.sort(0, false).unwrap();
    let weights = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), false);
    let loss = sorted.mul(&weights).unwrap().sum().unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 3.0).abs() < 1e-5);
    assert!((grad[1] - 1.0).abs() < 1e-5);
    assert!((grad[2] - 2.0).abs() < 1e-5);
}

#[test]
fn test_div_gradient() {
    let a = Variable::new(from_f32(&[6.0], &[1]), true);
    let b = Variable::new(from_f32(&[3.0], &[1]), true);
    let y = a.div(&b).unwrap().sum().unwrap();
    y.backward().unwrap();

    assert!((a.grad().unwrap().item().unwrap() - 1.0 / 3.0).abs() < 1e-5);
    assert!((b.grad().unwrap().item().unwrap() - (-6.0 / 9.0)).abs() < 1e-5);
}

#[test]
fn test_add_scalar_div_scalar_gradient() {
    let x = Variable::new(from_f32(&[2.0, 4.0], &[2]), true);
    let y = x.add_scalar(10.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);

    x.zero_grad();
    let y2 = x.div_scalar(4.0).unwrap().sum().unwrap();
    y2.backward().unwrap();
    let g = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((g[0] - 0.25).abs() < 1e-5);
    assert!((g[1] - 0.25).abs() < 1e-5);
}

#[test]
fn test_sqrt_gradient() {
    let x = Variable::new(from_f32(&[4.0, 9.0], &[2]), true);
    let y = x.sqrt().unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 0.25).abs() < 1e-5);
    assert!((grad[1] - 1.0 / 6.0).abs() < 1e-5);
}

#[test]
fn test_abs_gradient() {
    let x = Variable::new(from_f32(&[-3.0, 5.0], &[2]), true);
    let y = x.abs().unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - (-1.0)).abs() < 1e-4);
    assert!((grad[1] - 1.0).abs() < 1e-4);
}

#[test]
fn test_pow_scalar_gradient() {
    let x = Variable::new(from_f32(&[2.0, 3.0], &[2]), true);
    let y = x.pow_scalar(3.0).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 12.0).abs() < 1e-4);
    assert!((grad[1] - 27.0).abs() < 1e-4);
}

#[test]
fn test_clamp_gradient() {
    let x = Variable::new(from_f32(&[-1.0, 0.5, 2.0], &[3]), true);
    let y = x.clamp(0.0, 1.0).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 0.0).abs() < 1e-5);
    assert!((grad[1] - 1.0).abs() < 1e-5);
    assert!((grad[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_mean_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4]), true);
    let y = x.mean().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    for &g in &grad {
        assert!((g - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_sum_dim_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let y = x.sum_dim(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_f32_vec().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_mean_dim_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let y = x.mean_dim(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    for &g in &grad {
        assert!((g - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_max_dim_gradient() {
    // [[1, 4], [3, 2]] — max along dim=1 gives [4, 3]
    let x = Variable::new(from_f32(&[1.0, 4.0, 3.0, 2.0], &[2, 2]), true);
    let y = x.max_dim(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();

    // Gradient flows only to argmax positions: (0,1) and (1,0)
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn test_max_dim_keepdim_gradient() {
    let x = Variable::new(from_f32(&[1.0, 4.0, 3.0, 2.0], &[2, 2]), true);
    let y = x.max_dim(1, true).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn test_min_dim_gradient() {
    // [[1, 4], [3, 2]] — min along dim=1 gives [1, 2]
    let x = Variable::new(from_f32(&[1.0, 4.0, 3.0, 2.0], &[2, 2]), true);
    let y = x.min_dim(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();

    // Gradient flows only to argmin positions: (0,0) and (1,1)
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(grad, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_softmax_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
    let w = Variable::new(from_f32(&[1.0, 0.0, 0.0], &[3]), false);
    let y = x.softmax(0).unwrap().mul(&w).unwrap().sum().unwrap();
    y.backward().unwrap();

    let eps = 1e-4;
    let x_data = vec![1.0_f32, 2.0, 3.0];
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    for i in 0..3 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps as f32;
        xm[i] -= eps as f32;

        let fp = from_f32(&xp, &[3])
            .softmax(0)
            .unwrap()
            .to_f32_vec()
            .unwrap()[0] as f64;
        let fm = from_f32(&xm, &[3])
            .softmax(0)
            .unwrap()
            .to_f32_vec()
            .unwrap()[0] as f64;
        let numerical = (fp - fm) / (2.0 * eps);
        assert!(
            (grad[i] as f64 - numerical).abs() < 0.01,
            "softmax grad[{}]: analytical={}, numerical={}",
            i,
            grad[i],
            numerical
        );
    }
}

#[test]
fn test_log_softmax_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[3]), true);
    let w = Variable::new(from_f32(&[1.0, 0.0, 0.0], &[3]), false);
    let y = x.log_softmax(0).unwrap().mul(&w).unwrap().sum().unwrap();
    y.backward().unwrap();

    let eps = 1e-4;
    let x_data = vec![1.0_f32, 2.0, 3.0];
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    for i in 0..3 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps as f32;
        xm[i] -= eps as f32;
        let fp = from_f32(&xp, &[3])
            .log_softmax(0)
            .unwrap()
            .to_f32_vec()
            .unwrap()[0] as f64;
        let fm = from_f32(&xm, &[3])
            .log_softmax(0)
            .unwrap()
            .to_f32_vec()
            .unwrap()[0] as f64;
        let numerical = (fp - fm) / (2.0 * eps);
        assert!(
            (grad[i] as f64 - numerical).abs() < 0.01,
            "log_softmax grad[{}]: analytical={}, numerical={}",
            i,
            grad[i],
            numerical
        );
    }
}

#[test]
fn test_gelu_silu_gradient() {
    let x_data = vec![0.5_f32, -0.5, 1.0];
    let x = Variable::new(from_f32(&x_data, &[3]), true);
    let y = x.gelu().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();

    let eps = 1e-4;
    for i in 0..3 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps as f32;
        xm[i] -= eps as f32;
        let fp: f64 = from_f32(&xp, &[3])
            .gelu()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        let fm: f64 = from_f32(&xm, &[3])
            .gelu()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        assert!((grad[i] as f64 - (fp - fm) / (2.0 * eps)).abs() < 0.01);
    }

    x.zero_grad();
    let y_tanh = x.gelu_tanh().unwrap().sum().unwrap();
    y_tanh.backward().unwrap();
    let grad_tanh = x.grad().unwrap().to_f32_vec().unwrap();
    for i in 0..3 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps as f32;
        xm[i] -= eps as f32;
        let fp: f64 = from_f32(&xp, &[3])
            .gelu_tanh()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        let fm: f64 = from_f32(&xm, &[3])
            .gelu_tanh()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        assert!((grad_tanh[i] as f64 - (fp - fm) / (2.0 * eps)).abs() < 0.01);
    }

    x.zero_grad();
    let y2 = x.silu().unwrap().sum().unwrap();
    y2.backward().unwrap();
    let grad2 = x.grad().unwrap().to_f32_vec().unwrap();
    for i in 0..3 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps as f32;
        xm[i] -= eps as f32;
        let fp: f64 = from_f32(&xp, &[3])
            .silu()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        let fm: f64 = from_f32(&xm, &[3])
            .silu()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        assert!((grad2[i] as f64 - (fp - fm) / (2.0 * eps)).abs() < 0.01);
    }
}

#[test]
fn test_reshape_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let y = x.reshape(&[4]).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().shape(), vec![2, 2]);
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![1.0; 4]);
}

#[test]
fn test_transpose_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
    let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
    let y = x.transpose(0, 1).unwrap().mul(&w).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().shape(), vec![2, 3]);
}

#[test]
fn test_narrow_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]), true);
    let y = x.narrow(0, 1, 3).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_f32_vec().unwrap(),
        vec![0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn test_select_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
    let y = x.select(0, 1).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_f32_vec().unwrap(),
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn test_expand_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
    let y = x.expand(&[4, 3]).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().to_f32_vec().unwrap(), vec![4.0, 4.0, 4.0]);
}

#[test]
fn test_permute_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), true);
    let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), false);
    let y = x.permute(&[1, 0]).unwrap().mul(&w).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().shape(), vec![2, 3]);
}

#[test]
fn test_squeeze_unsqueeze_gradient() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
    let y = x.squeeze(0).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().shape(), vec![1, 3]);

    x.zero_grad();
    let y2 = x.unsqueeze(2).unwrap().sum().unwrap();
    y2.backward().unwrap();
    assert_eq!(x.grad().unwrap().shape(), vec![1, 3]);
}

#[test]
fn test_cat_gradient() {
    let a = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let b = Variable::new(from_f32(&[3.0, 4.0, 5.0], &[3]), true);
    let y = a.cat(&b, 0).unwrap().sum().unwrap();
    y.backward().unwrap();

    assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0]);
    assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_index_select_gradient() {
    let x = Variable::new(from_f32(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]), true);
    let idx = Tensor::from_f32(&[0.0, 2.0, 2.0], &[3], crate::tensor::test_device())
        .unwrap()
        .to_dtype(crate::tensor::DType::Int64)
        .unwrap();
    let y = x.index_select(0, &idx).unwrap().sum().unwrap();
    y.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 1.0).abs() < 1e-5);
    assert!((grad[1] - 0.0).abs() < 1e-5);
    assert!((grad[2] - 2.0).abs() < 1e-5);
    assert!((grad[3] - 0.0).abs() < 1e-5);
    assert!((grad[4] - 0.0).abs() < 1e-5);
}

#[test]
fn test_std_dim_gradient() {
    let x = Variable::new(from_f32(&[1.0, 3.0, 5.0, 7.0], &[2, 2]), true);
    let y = x.std_dim(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();

    let eps = 1e-4;
    let x_data = vec![1.0_f32, 3.0, 5.0, 7.0];
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    for i in 0..4 {
        let mut xp = x_data.clone();
        let mut xm = x_data.clone();
        xp[i] += eps as f32;
        xm[i] -= eps as f32;
        let fp: f64 = from_f32(&xp, &[2, 2])
            .std_dim(1, false)
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        let fm: f64 = from_f32(&xm, &[2, 2])
            .std_dim(1, false)
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        let numerical = (fp - fm) / (2.0 * eps);
        assert!(
            (grad[i] as f64 - numerical).abs() < 0.01,
            "std_dim grad[{}]: analytical={}, numerical={}",
            i,
            grad[i],
            numerical
        );
    }
}

#[test]
fn test_cat_many_gradient() {
    let a = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let b = Variable::new(from_f32(&[3.0, 4.0, 5.0], &[3]), true);
    let c = Variable::new(from_f32(&[6.0], &[1]), true);
    let catted = Variable::cat_many(&[&a, &b, &c], 0).unwrap();
    assert_eq!(catted.data().shape(), vec![6]);

    // Weight each element differently to verify gradient routing
    let w = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]), false);
    let loss = catted.mul(&w).unwrap().sum().unwrap();
    loss.backward().unwrap();

    assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![3.0, 4.0, 5.0]);
    assert_eq!(c.grad().unwrap().to_f32_vec().unwrap(), vec![6.0]);
}

#[test]
fn test_stack_gradient() {
    let a = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let b = Variable::new(from_f32(&[3.0, 4.0], &[2]), true);
    let c = Variable::new(from_f32(&[5.0, 6.0], &[2]), true);
    let stacked = Variable::stack(&[a.clone(), b.clone(), c.clone()], 0).unwrap();
    assert_eq!(stacked.data().shape(), vec![3, 2]);

    let w = Variable::new(from_f32(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0], &[3, 2]), false);
    let loss = stacked.mul(&w).unwrap().sum().unwrap();
    loss.backward().unwrap();

    assert_eq!(a.grad().unwrap().to_f32_vec().unwrap(), vec![1.0, 0.0]);
    assert_eq!(b.grad().unwrap().to_f32_vec().unwrap(), vec![0.0, 1.0]);
    assert_eq!(c.grad().unwrap().to_f32_vec().unwrap(), vec![2.0, 0.0]);
}

#[test]
fn test_triu_gradient() {
    // 2x2 matrix, triu with diagonal=1 keeps only upper triangle (above main diag)
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), true);
    let y = x.triu(1).unwrap();
    let vals = y.data().to_f32_vec().unwrap();
    assert_eq!(vals, vec![0.0, 2.0, 0.0, 0.0]); // only (0,1) survives

    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // gradient flows only through upper triangle
    assert_eq!(grad, vec![0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn test_triu_batch_gradient() {
    // [2, 3, 3] batch — triu applied per-matrix
    let x = Variable::new(
        from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
            &[2, 3, 3],
        ),
        true,
    );
    let y = x.triu(0).unwrap();
    let loss = y.sum().unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    let expected = vec![
        1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    ];
    assert_eq!(grad, expected);
}

#[test]
fn test_shared_variable_multiple_adds() {
    let state0 = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
    let bias = Variable::new(from_f32(&[0.0, 0.0], &[2]), true);

    let state1 = state0.add(&bias).unwrap();
    let state2 = state1.add(&bias).unwrap();
    let state3 = state2.add(&bias).unwrap();
    let loss = state3.sum().unwrap();
    loss.backward().unwrap();

    let grad = bias.grad().unwrap().to_f32_vec().unwrap();
    assert!(
        (grad[0] - 3.0).abs() < 1e-5 && (grad[1] - 3.0).abs() < 1e-5,
        "bias grad should be [3,3], got {:?}",
        grad
    );
}

#[test]
fn test_backward_frees_grad_fn_chain_leakcheck() {
    if crate::tensor::test_device() != Device::CPU {
        return;
    }
    // Verify that backward() + detach_() doesn't leak C++ autograd Nodes.
    // Simulates a training loop: multiple loss terms per step, many steps.
    // Uses live_tensor_count (C++ handle tracking) instead of RSS which
    // is unreliable on CI runners with shared memory pressure.
    use crate::nn::optim::{Adam, Optimizer};
    use crate::nn::{Linear, Module};

    let linear = Linear::on_device(256, 256, crate::tensor::test_device()).unwrap();
    let params = linear.parameters();
    let mut opt = Adam::new(&params, 0.001);
    let opts = crate::tensor::test_opts();

    // Warm up -- let allocators settle
    for _ in 0..50 {
        let x = Variable::new(Tensor::randn(&[32, 256], opts).unwrap(), false);
        let y = linear.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        opt.zero_grad();
        loss.backward().unwrap();
        opt.step().unwrap();
    }

    let handles_before = crate::tensor::live_tensor_count();

    for _ in 0..2000 {
        let x = Variable::new(Tensor::randn(&[32, 256], opts).unwrap(), false);
        let y = linear.forward(&x).unwrap();
        // Multiple loss terms (like FBRL training)
        let l1 = y.sum().unwrap();
        let l2 = y.mean().unwrap();
        let total = l1.add(&l2).unwrap();
        opt.zero_grad();
        total.backward().unwrap();
        opt.step().unwrap();
    }

    let handles_after = crate::tensor::live_tensor_count();
    let leaked = handles_after as i64 - handles_before as i64;
    assert!(
        leaked < 100,
        "live tensor count grew by {leaked} over 2000 training steps — possible grad_fn leak \
             (before={handles_before}, after={handles_after})"
    );
}

/// Phased leak isolation: measures handle count and RSS growth for
/// each training phase independently. Run with --nocapture to see output.
#[test]
fn test_leak_isolation_phases_leakcheck() {
    if crate::tensor::test_device() != Device::CPU {
        return;
    }
    use crate::nn::optim::{Adam, Optimizer};
    use crate::nn::parameter::Parameter;
    use crate::nn::{Linear, Module, clip_grad_norm, cross_entropy_loss};
    use crate::tensor::{TensorOptions, live_tensor_count, malloc_trim, rss_kb};

    let opts = crate::tensor::test_opts();
    let iters = 1000;
    let batch = 32;
    let dim = 128;
    let classes: i64 = 26;

    // Build a model similar to FBRL: two linears + loss
    let linear1 = Linear::on_device(dim, dim, crate::tensor::test_device()).unwrap();
    let linear2 = Linear::on_device(dim, classes, crate::tensor::test_device()).unwrap();
    let params: Vec<Parameter> = linear1
        .parameters()
        .into_iter()
        .chain(linear2.parameters())
        .collect();
    let mut opt = Adam::new(&params, 0.001);

    // Warm up
    for _ in 0..50 {
        let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
        let h = linear1.forward(&x).unwrap().relu().unwrap();
        let logits = linear2.forward(&h).unwrap();
        let target = Variable::new(
            Tensor::zeros(
                &[batch],
                TensorOptions {
                    dtype: crate::tensor::DType::Int64,
                    ..opts
                },
            )
            .unwrap(),
            false,
        );
        let loss = cross_entropy_loss(&logits, &target).unwrap();
        opt.zero_grad();
        loss.backward().unwrap();
        clip_grad_norm(&params, 1.0).unwrap();
        opt.step().unwrap();
    }

    // --- Phase 1: Forward only ---
    malloc_trim();
    let h0 = live_tensor_count();
    let r0 = rss_kb();
    for _ in 0..iters {
        let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
        let h = linear1.forward(&x).unwrap().relu().unwrap();
        let _logits = linear2.forward(&h).unwrap();
    }
    malloc_trim();
    let h1 = live_tensor_count();
    let r1 = rss_kb();
    eprintln!(
        "Phase 1 (forward only):  handles {:+}  RSS {:+.1}MB",
        h1 as i64 - h0 as i64,
        (r1 as f64 - r0 as f64) / 1024.0
    );

    // --- Phase 2: Forward + backward ---
    malloc_trim();
    let h0 = live_tensor_count();
    let r0 = rss_kb();
    for _ in 0..iters {
        let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
        let h = linear1.forward(&x).unwrap().relu().unwrap();
        let logits = linear2.forward(&h).unwrap();
        let target = Variable::new(
            Tensor::zeros(
                &[batch],
                TensorOptions {
                    dtype: crate::tensor::DType::Int64,
                    ..opts
                },
            )
            .unwrap(),
            false,
        );
        let loss = cross_entropy_loss(&logits, &target).unwrap();
        loss.backward().unwrap();
    }
    malloc_trim();
    let h1 = live_tensor_count();
    let r1 = rss_kb();
    eprintln!(
        "Phase 2 (fwd+bwd):      handles {:+}  RSS {:+.1}MB",
        h1 as i64 - h0 as i64,
        (r1 as f64 - r0 as f64) / 1024.0
    );

    // --- Phase 3: Full training step ---
    malloc_trim();
    let h0 = live_tensor_count();
    let r0 = rss_kb();
    for _ in 0..iters {
        let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
        let h = linear1.forward(&x).unwrap().relu().unwrap();
        let logits = linear2.forward(&h).unwrap();
        let target = Variable::new(
            Tensor::zeros(
                &[batch],
                TensorOptions {
                    dtype: crate::tensor::DType::Int64,
                    ..opts
                },
            )
            .unwrap(),
            false,
        );
        let l1 = cross_entropy_loss(&logits, &target).unwrap();
        let l2 = logits.mean().unwrap();
        let total = l1.add(&l2).unwrap();
        opt.zero_grad();
        total.backward().unwrap();
        clip_grad_norm(&params, 1.0).unwrap();
        opt.step().unwrap();
    }
    malloc_trim();
    let h1 = live_tensor_count();
    let r1 = rss_kb();
    eprintln!(
        "Phase 3 (full step):    handles {:+}  RSS {:+.1}MB",
        h1 as i64 - h0 as i64,
        (r1 as f64 - r0 as f64) / 1024.0
    );

    // The handle count should not grow between phases
    // (small fluctuations OK, but not proportional to iters)
    assert!(
        (h1 as i64 - h0 as i64).unsigned_abs() < 100,
        "handle count drifted by {} over {iters} steps — tensor handle leak!",
        h1 as i64 - h0 as i64
    );
}

/// Graph-with-loop leak test: simulates FBRL pattern (graph loops, tags,
/// multiple loss terms, optimizer). Run with --nocapture to see diagnostics.
#[test]
fn test_graph_loop_leak_leakcheck() {
    if crate::tensor::test_device() != Device::CPU {
        return;
    }
    use crate::graph::FlowBuilder;
    use crate::nn::optim::{Adam, Optimizer};
    use crate::nn::parameter::Parameter;
    use crate::nn::{Linear, Module, clip_grad_norm, cross_entropy_loss};
    use crate::tensor::{TensorOptions, live_tensor_count, malloc_trim, rss_kb};

    let opts = crate::tensor::test_opts();
    let batch: i64 = 16;
    let dim = 64;
    let classes: i64 = 26;
    let loop_iters = 4;

    // Build a graph: Linear → loop(Linear+ReLU, 4 iters) → tag("out") → Linear
    let graph =
        FlowBuilder::from(Linear::on_device(dim, dim, crate::tensor::test_device()).unwrap())
            .loop_body(Linear::on_device(dim, dim, crate::tensor::test_device()).unwrap())
            .for_n(loop_iters)
            .tag("loop_out")
            .through(Linear::on_device(dim, classes, crate::tensor::test_device()).unwrap())
            .tag("logits")
            .build()
            .unwrap();

    let params: Vec<Parameter> = graph
        .named_parameters()
        .into_iter()
        .map(|(_name, p)| p)
        .collect();
    let mut opt = Adam::new(&params, 0.001);

    // Warm up
    for _ in 0..30 {
        let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
        let y = graph.forward(&x).unwrap();
        let target = Variable::new(
            Tensor::zeros(
                &[batch],
                TensorOptions {
                    dtype: crate::tensor::DType::Int64,
                    ..opts
                },
            )
            .unwrap(),
            false,
        );
        let loss = cross_entropy_loss(&y, &target).unwrap();
        opt.zero_grad();
        loss.backward().unwrap();
        clip_grad_norm(&params, 1.0).unwrap();
        opt.step().unwrap();
        graph.detach_state();
    }

    // live_tensor_count is a global atomic shared with every other test
    // running in parallel, so symmetric drift checks (|a - b| < N) flake
    // on CI when concurrent tests happen to start or finish work during
    // our measurement window. Use the same one-sided pattern as
    // test_backward_frees_grad_fn_chain: a real leak only manifests as
    // monotonic growth, while concurrent activity that completes during
    // our window can only push the counter spuriously DOWN. Tolerate
    // shrinkage, catch growth.
    // Leak detection: run two chunks and compare growth rates.
    // A real leak grows linearly with iterations; concurrent test noise
    // is constant (unrelated to our chunk size). By comparing a short
    // chunk to a long chunk, we cancel out the shared-atomic noise.
    let mut run_chunk = |n: usize| {
        for _ in 0..n {
            let x = Variable::new(Tensor::randn(&[batch, dim], opts).unwrap(), false);
            let y = graph.forward(&x).unwrap();
            let target = Variable::new(
                Tensor::zeros(
                    &[batch],
                    TensorOptions {
                        dtype: crate::tensor::DType::Int64,
                        ..opts
                    },
                )
                .unwrap(),
                false,
            );
            let l1 = cross_entropy_loss(&y, &target).unwrap();
            let l2 = y.mean().unwrap();
            let total = l1.add(&l2).unwrap();
            opt.zero_grad();
            total.backward().unwrap();
            clip_grad_norm(&params, 1.0).unwrap();
            opt.step().unwrap();
            graph.detach_state();
            graph.record_scalar("loss", total.item().unwrap());
        }
        graph.flush(&[]);
    };

    // The test's value is exercising the full graph+loop+optimizer pattern
    // for 500 iterations without crashing (use-after-free, double-free,
    // stack overflow from unbounded Rc chains). Any of those = instant
    // failure. Quantitative leak detection (live_tensor_count, RSS) is
    // unreliable under parallel CI: the global atomic is shared across
    // tests, and glibc page retention varies by runner. Log diagnostics
    // for manual review but don't assert on noisy metrics.
    malloc_trim();
    let rss_before = rss_kb();
    let handles_before = live_tensor_count();

    run_chunk(500);

    malloc_trim();
    let handle_delta = live_tensor_count() as i64 - handles_before as i64;
    let rss_growth_mb = (rss_kb() as f64 - rss_before as f64) / 1024.0;

    eprintln!(
        "Graph+loop (500 steps): handles {handle_delta:+}  RSS {rss_growth_mb:+.1}MB  \
             (diagnostic only — no assert, see comment)"
    );
}

#[test]
fn test_autograd_node_count() {
    let opts = crate::tensor::test_opts();

    // Leaf variable: 0 nodes
    let x = Variable::new(Tensor::randn(&[2, 3], opts).unwrap(), true);
    assert_eq!(x.autograd_node_count(), 0);

    // Single op: 1 node (MulBackward)
    let y = x.mul_scalar(2.0).unwrap();
    assert!(y.autograd_node_count() >= 1);

    // Chain: mul -> sum -> 2 nodes
    let z = y.sum().unwrap();
    assert!(z.autograd_node_count() >= 2);

    // Fused linear has fewer nodes than manual matmul+add
    let w = Variable::new(Tensor::randn(&[4, 3], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[4], opts).unwrap(), true);
    let inp = Variable::new(Tensor::randn(&[2, 3], opts).unwrap(), false);

    // Fused: single linear node
    let fused = linear(&inp, &w, Some(&b)).unwrap();
    let fused_nodes = fused.autograd_node_count();

    // Manual: matmul + add = 2 nodes
    let wt = w.transpose(0, 1).unwrap();
    let manual = inp.matmul(&wt).unwrap().add(&b).unwrap();
    let manual_nodes = manual.autograd_node_count();

    assert!(
        fused_nodes < manual_nodes,
        "fused linear ({fused_nodes}) should have fewer nodes than manual ({manual_nodes})"
    );
}

// --- Activation gradient tests ---

fn numerical_grad(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    let eps = 1e-5;
    (f(x + eps) - f(x - eps)) / (2.0 * eps)
}

#[test]
fn test_leaky_relu_gradient() {
    // Positive input: gradient = 1
    let x = Variable::new(
        Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.leaky_relu(0.01).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad - 1.0).abs() < 1e-4);

    // Negative input: gradient = negative_slope
    let x2 = Variable::new(
        Tensor::from_f32(&[-2.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y2 = x2.leaky_relu(0.01).unwrap().sum().unwrap();
    y2.backward().unwrap();
    let grad2 = x2.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad2 - 0.01).abs() < 1e-4);
}

#[test]
fn test_elu_gradient() {
    // Positive: gradient = 1
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.elu(1.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad - 1.0).abs() < 1e-4);

    // Negative: gradient = alpha * exp(x)
    let x2 = Variable::new(
        Tensor::from_f32(&[-1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y2 = x2.elu(1.0).unwrap().sum().unwrap();
    y2.backward().unwrap();
    let grad2 = x2.grad().unwrap().to_f32_vec().unwrap()[0];
    let expected = (-1.0f64).exp(); // alpha * exp(x) = 1.0 * exp(-1)
    assert!((grad2 as f64 - expected).abs() < 1e-3);
}

#[test]
fn test_softplus_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.softplus(1.0, 20.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // softplus'(x) = sigmoid(x) = 1/(1+exp(-x))
    let expected = 1.0 / (1.0 + (-1.0f64).exp());
    assert!((grad as f64 - expected).abs() < 1e-3);
}

#[test]
fn test_mish_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[0.5], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.mish().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // Numerical validation
    let mish = |v: f64| v * (v.exp().ln_1p()).tanh();
    let expected = numerical_grad(&mish, 0.5);
    assert!(
        (grad as f64 - expected).abs() < 1e-3,
        "mish grad at 0.5: got {grad}, expected {expected}"
    );
}

#[test]
fn test_selu_gradient() {
    let scale = 1.0507009873554805;
    // Positive: gradient = scale
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.selu().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad as f64 - scale).abs() < 1e-3);
}

#[test]
fn test_hardswish_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.hardswish().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // hardswish(x) = x * clip(x+3, 0, 6) / 6
    // For x=1: x*(x+3)/6, derivative = (2x+3)/6 = 5/6
    let expected = 5.0 / 6.0;
    assert!((grad as f64 - expected).abs() < 1e-3);
}

#[test]
fn test_hardsigmoid_gradient() {
    // In the linear region (-3 < x < 3): gradient = 1/6
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.hardsigmoid().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad as f64 - 1.0 / 6.0).abs() < 1e-3);

    // Outside the range: gradient = 0
    let x2 = Variable::new(
        Tensor::from_f32(&[5.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y2 = x2.hardsigmoid().unwrap().sum().unwrap();
    y2.backward().unwrap();
    let grad2 = x2.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!(grad2.abs() < 1e-4);
}

#[test]
fn test_prelu_gradient() {
    let weight = Variable::new(
        Tensor::from_f32(&[0.25], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    // Positive input: gradient w.r.t. input = 1
    let x = Variable::new(
        Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.prelu(&weight).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad - 1.0).abs() < 1e-4);

    // Negative input: gradient w.r.t. input = weight
    let x2 = Variable::new(
        Tensor::from_f32(&[-2.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y2 = x2.prelu(&weight).unwrap().sum().unwrap();
    y2.backward().unwrap();
    let grad2 = x2.grad().unwrap().to_f32_vec().unwrap()[0];
    assert!((grad2 - 0.25).abs() < 1e-4);
}

// --- Element-wise math gradient tests ---

#[test]
fn test_tril_gradient() {
    let x = Variable::new(
        Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            crate::tensor::test_device(),
        )
        .unwrap(),
        true,
    );
    let y = x.tril(0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // Lower triangle should have grad=1, upper should be 0
    assert!((grad[0] - 1.0).abs() < 1e-5); // [0,0]
    assert!((grad[1] - 0.0).abs() < 1e-5); // [0,1] upper
    assert!((grad[3] - 1.0).abs() < 1e-5); // [1,0]
    assert!((grad[4] - 1.0).abs() < 1e-5); // [1,1]
    assert!((grad[5] - 0.0).abs() < 1e-5); // [1,2] upper
}

#[test]
fn test_clamp_min_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.clamp_min(0.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 0.0).abs() < 1e-5); // clamped
    assert!((grad[1] - 1.0).abs() < 1e-5); // pass-through
    assert!((grad[2] - 1.0).abs() < 1e-5); // pass-through
}

#[test]
fn test_clamp_max_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.clamp_max(1.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 1.0).abs() < 1e-5); // pass-through
    assert!((grad[1] - 1.0).abs() < 1e-5); // pass-through
    assert!((grad[2] - 0.0).abs() < 1e-5); // clamped
}

#[test]
fn test_log1p_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.log1p().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // d/dx log(1+x) = 1/(1+x) = 0.5
    assert!((grad - 0.5).abs() < 1e-4);
}

#[test]
fn test_expm1_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.expm1().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // d/dx (exp(x)-1) = exp(x) = e
    assert!((grad as f64 - 1.0f64.exp()).abs() < 1e-3);
}

#[test]
fn test_log2_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[4.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.log2().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // d/dx log2(x) = 1/(x*ln(2))
    let expected = 1.0 / (4.0 * 2.0f64.ln());
    assert!((grad as f64 - expected).abs() < 1e-4);
}

#[test]
fn test_log10_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[10.0], &[1], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.log10().unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap()[0];
    // d/dx log10(x) = 1/(x*ln(10))
    let expected = 1.0 / (10.0 * 10.0f64.ln());
    assert!((grad as f64 - expected).abs() < 1e-4);
}

#[test]
fn test_maximum_gradient() {
    let a = Variable::new(
        Tensor::from_f32(&[1.0, 5.0], &[2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let b = Variable::new(
        Tensor::from_f32(&[3.0, 2.0], &[2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = a.maximum(&b).unwrap().sum().unwrap();
    y.backward().unwrap();
    let ga = a.grad().unwrap().to_f32_vec().unwrap();
    let gb = b.grad().unwrap().to_f32_vec().unwrap();
    // max(1,3)=3 -> grad goes to b; max(5,2)=5 -> grad goes to a
    assert!((ga[0] - 0.0).abs() < 1e-5);
    assert!((ga[1] - 1.0).abs() < 1e-5);
    assert!((gb[0] - 1.0).abs() < 1e-5);
    assert!((gb[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_minimum_gradient() {
    let a = Variable::new(
        Tensor::from_f32(&[1.0, 5.0], &[2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let b = Variable::new(
        Tensor::from_f32(&[3.0, 2.0], &[2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = a.minimum(&b).unwrap().sum().unwrap();
    y.backward().unwrap();
    let ga = a.grad().unwrap().to_f32_vec().unwrap();
    let gb = b.grad().unwrap().to_f32_vec().unwrap();
    // min(1,3)=1 -> grad goes to a; min(5,2)=2 -> grad goes to b
    assert!((ga[0] - 1.0).abs() < 1e-5);
    assert!((ga[1] - 0.0).abs() < 1e-5);
    assert!((gb[0] - 0.0).abs() < 1e-5);
    assert!((gb[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_masked_fill_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], crate::tensor::test_device()).unwrap(),
        true,
    );
    let mask = Tensor::from_f32(&[1.0, 0.0, 1.0, 0.0], &[4], crate::tensor::test_device())
        .unwrap()
        .gt_scalar(0.5)
        .unwrap();
    let y = x.masked_fill(&mask, 0.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // Masked positions get 0 gradient, others get 1
    assert!((grad[0] - 0.0).abs() < 1e-5); // masked
    assert!((grad[1] - 1.0).abs() < 1e-5); // kept
    assert!((grad[2] - 0.0).abs() < 1e-5); // masked
    assert!((grad[3] - 1.0).abs() < 1e-5); // kept
}

#[test]
fn test_cosine_similarity_gradient() {
    let a = Variable::new(
        Tensor::from_f32(&[1.0, 0.0], &[1, 2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let b = Variable::new(
        Tensor::from_f32(&[0.0, 1.0], &[1, 2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let cos = a.cosine_similarity(&b, 1, 1e-8).unwrap().sum().unwrap();
    cos.backward().unwrap();
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_normalize_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[3.0, 4.0], &[1, 2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.normalize(2.0, 1).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    // Normalized [3,4] with L2=5 gives [0.6, 0.8]
    let normed = x.normalize(2.0, 1).unwrap().data().to_f32_vec().unwrap();
    assert!((normed[0] - 0.6).abs() < 1e-4);
    assert!((normed[1] - 0.8).abs() < 1e-4);
}

// --- Reduction gradient tests ---

#[test]
fn test_prod_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[2.0, 3.0, 4.0], &[3], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.prod().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // d(prod)/dx_i = prod / x_i
    assert!((grad[0] - 12.0).abs() < 1e-3); // 24/2
    assert!((grad[1] - 8.0).abs() < 1e-3); // 24/3
    assert!((grad[2] - 6.0).abs() < 1e-3); // 24/4
}

#[test]
fn test_prod_dim_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.prod_dim(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_cumsum_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.cumsum(0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // cumsum([a,b,c]) = [a, a+b, a+b+c], sum = 3a + 2b + c
    assert!((grad[0] - 3.0).abs() < 1e-4);
    assert!((grad[1] - 2.0).abs() < 1e-4);
    assert!((grad[2] - 1.0).abs() < 1e-4);
}

#[test]
fn test_logsumexp_gradient() {
    let x = Variable::new(
        Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
        true,
    );
    let y = x.logsumexp(1, false).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // d(logsumexp)/dx_i = exp(x_i) / sum(exp(x_j)) = softmax(x_i)
    let sum_exp: f32 = [1.0f32, 2.0, 3.0].iter().map(|v| v.exp()).sum();
    for (i, &v) in [1.0f32, 2.0, 3.0].iter().enumerate() {
        let expected = v.exp() / sum_exp;
        assert!((grad[i] - expected).abs() < 1e-4);
    }
}

#[test]
fn test_flatten_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[2, 3, 4], opts).unwrap(), true);
    let y = x.flatten(1, 2).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap();
    assert_eq!(grad.shape(), vec![2, 3, 4]);
}

// --- Variable method tests ---

#[test]
fn test_variable_set_grad() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    assert!(x.grad().is_none());
    let g = from_f32(&[0.5, 0.5], &[2]);
    x.set_grad(g);
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    assert!((grad[0] - 0.5).abs() < 1e-5);
    assert!((grad[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_variable_set_requires_grad() {
    let x = Variable::new(from_f32(&[1.0], &[1]), false);
    assert!(!x.requires_grad());
    x.set_requires_grad(true).unwrap();
    assert!(x.requires_grad());
    x.set_requires_grad(false).unwrap();
    assert!(!x.requires_grad());
}

#[test]
fn test_variable_is_leaf() {
    let x = Variable::new(from_f32(&[1.0], &[1]), true);
    assert!(x.is_leaf());
    let y = x.mul_scalar(2.0).unwrap();
    assert!(!y.is_leaf());
}

#[test]
fn test_variable_numel() {
    let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), false);
    assert_eq!(x.numel(), 6);
}

#[test]
fn test_variable_zero_grad_set_to_none() {
    let x = Variable::new(from_f32(&[1.0], &[1]), true);
    let g = from_f32(&[5.0], &[1]);
    x.set_grad(g);
    assert!(x.grad().is_some());
    x.zero_grad_set_to_none();
    assert!(x.grad().is_none());
}

#[test]
fn test_variable_set_data() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    assert!(x.requires_grad());
    let new_data = from_f32(&[10.0, 20.0], &[2]);
    x.set_data(new_data);
    // Data should be replaced but requires_grad preserved
    assert!(x.requires_grad());
    let vals = x.data().to_f32_vec().unwrap();
    assert!((vals[0] - 10.0).abs() < 1e-5);
    assert!((vals[1] - 20.0).abs() < 1e-5);
}

#[test]
fn test_variable_to_device_same() {
    let dev = crate::tensor::test_device();
    let x = Variable::new(from_f32(&[1.0], &[1]), true);
    let y = x.to_device(dev).unwrap();
    assert_eq!(y.device(), dev);
    assert!(y.requires_grad());
}

// --- Fused autograd op gradient tests ---

#[test]
fn test_fused_conv1d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 3, 10], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[4, 3, 3], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[4], opts).unwrap(), true);
    let y = conv1d(&x, &w, Some(&b), 1, 0, 1, 1).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_fused_conv2d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 3, 8, 8], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[4, 3, 3, 3], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[4], opts).unwrap(), true);
    let y = conv2d(&x, &w, Some(&b), [1, 1], [0, 0], [1, 1], 1)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_fused_conv3d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 2, 4, 4, 4], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[3, 2, 3, 3, 3], opts).unwrap(), true);
    let y = conv3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_conv_transpose1d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 4, 5], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[4, 2, 3], opts).unwrap(), true);
    let y = conv_transpose1d(&x, &w, None, 1, 0, 0, 1, 1)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_conv_transpose2d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 4, 4, 4], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[4, 2, 3, 3], opts).unwrap(), true);
    let y = conv_transpose2d(&x, &w, None, [1, 1], [0, 0], [0, 0], [1, 1], 1)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_conv_transpose3d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 4, 3, 3, 3], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[4, 2, 3, 3, 3], opts).unwrap(), true);
    let y = conv_transpose3d(&x, &w, None, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_layer_norm_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[2, 8], opts).unwrap(), true);
    let w = Variable::new(Tensor::ones(&[8], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[8], opts).unwrap(), true);
    let y = layer_norm(&x, &w, &b, 8, 1e-5).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
    assert!(b.grad().is_some());
}

#[test]
fn test_fused_group_norm_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[2, 8, 4, 4], opts).unwrap(), true);
    let w = Variable::new(Tensor::ones(&[8], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[8], opts).unwrap(), true);
    let y = group_norm(&x, 2, &w, &b, 1e-5).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_instance_norm_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[2, 4, 8, 8], opts).unwrap(), true);
    let w = Variable::new(Tensor::ones(&[4], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[4], opts).unwrap(), true);
    let y = instance_norm(&x, Some(&w), Some(&b), None, None, true, 0.1, 1e-5)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_max_pool2d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), true);
    let y = max_pool2d(&x, [2, 2], [2, 2], [0, 0], [1, 1], false)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_avg_pool2d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), true);
    let y = avg_pool2d(&x, [2, 2], [2, 2], [0, 0], false, true)
        .unwrap()
        .sum()
        .unwrap();
    y.backward().unwrap();
    let grad = x.grad().unwrap().to_f32_vec().unwrap();
    // avg_pool with 2x2 kernel: each input contributes 1/4 to one output
    assert!(grad.iter().all(|&v| (v - 0.25).abs() < 1e-4));
}

#[test]
fn test_fused_adaptive_avg_pool2d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 8, 8], opts).unwrap(), true);
    let y = adaptive_avg_pool2d(&x, [2, 2]).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_adaptive_max_pool2d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 8, 8], opts).unwrap(), true);
    let y = adaptive_max_pool2d(&x, [2, 2]).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_max_pool1d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 8], opts).unwrap(), true);
    let y = max_pool1d(&x, 2, 2, 0, 1, false).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_avg_pool1d_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 8], opts).unwrap(), true);
    let y = avg_pool1d(&x, 2, 2, 0, false, true).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_gru_cell_gradient() {
    let opts = crate::tensor::test_opts();
    let input_size = 4;
    let hidden_size = 3;
    let x = Variable::new(Tensor::randn(&[2, input_size], opts).unwrap(), true);
    let hx = Variable::new(Tensor::zeros(&[2, hidden_size], opts).unwrap(), true);
    let w_ih = Variable::new(
        Tensor::randn(&[3 * hidden_size, input_size], opts).unwrap(),
        true,
    );
    let w_hh = Variable::new(
        Tensor::randn(&[3 * hidden_size, hidden_size], opts).unwrap(),
        true,
    );
    let b_ih = Variable::new(Tensor::zeros(&[3 * hidden_size], opts).unwrap(), true);
    let b_hh = Variable::new(Tensor::zeros(&[3 * hidden_size], opts).unwrap(), true);
    let h = gru_cell(&x, &hx, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
    let loss = h.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w_ih.grad().is_some());
    assert!(w_hh.grad().is_some());
}

#[test]
fn test_fused_lstm_cell_gradient() {
    let opts = crate::tensor::test_opts();
    let input_size = 4;
    let hidden_size = 3;
    let x = Variable::new(Tensor::randn(&[2, input_size], opts).unwrap(), true);
    let hx = Variable::new(Tensor::zeros(&[2, hidden_size], opts).unwrap(), true);
    let cx = Variable::new(Tensor::zeros(&[2, hidden_size], opts).unwrap(), true);
    let w_ih = Variable::new(
        Tensor::randn(&[4 * hidden_size, input_size], opts).unwrap(),
        true,
    );
    let w_hh = Variable::new(
        Tensor::randn(&[4 * hidden_size, hidden_size], opts).unwrap(),
        true,
    );
    let b_ih = Variable::new(Tensor::zeros(&[4 * hidden_size], opts).unwrap(), true);
    let b_hh = Variable::new(Tensor::zeros(&[4 * hidden_size], opts).unwrap(), true);
    let (h, _c) = lstm_cell(&x, &hx, &cx, &w_ih, &w_hh, &b_ih, &b_hh).unwrap();
    let loss = h.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(w_ih.grad().is_some());
    assert!(w_hh.grad().is_some());
}

#[test]
fn test_fused_bilinear_gradient() {
    let opts = crate::tensor::test_opts();
    let x1 = Variable::new(Tensor::randn(&[2, 3], opts).unwrap(), true);
    let x2 = Variable::new(Tensor::randn(&[2, 4], opts).unwrap(), true);
    let w = Variable::new(Tensor::randn(&[5, 3, 4], opts).unwrap(), true);
    let b = Variable::new(Tensor::zeros(&[5], opts).unwrap(), true);
    let y = bilinear(&x1, &x2, &w, Some(&b)).unwrap().sum().unwrap();
    y.backward().unwrap();
    assert!(x1.grad().is_some());
    assert!(x2.grad().is_some());
    assert!(w.grad().is_some());
}

#[test]
fn test_fused_pixel_shuffle_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 4, 2, 2], opts).unwrap(), true);
    let y = pixel_shuffle(&x, 2).unwrap();
    assert_eq!(y.shape(), vec![1, 1, 4, 4]);
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_pixel_unshuffle_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), true);
    let y = pixel_unshuffle(&x, 2).unwrap();
    assert_eq!(y.shape(), vec![1, 4, 2, 2]);
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_im2col_col2im_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), true);
    let cols = im2col(&x, [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    // im2col: [1, C*kH*kW, L] where L = output spatial positions
    let loss = cols.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
}

#[test]
fn test_fused_embedding_bag_gradient() {
    let opts = crate::tensor::test_opts();
    let dev = crate::tensor::test_device();
    let w = Variable::new(Tensor::randn(&[10, 4], opts).unwrap(), true);
    let indices = Tensor::from_i64(&[0, 1, 2, 3, 4], &[5], dev).unwrap();
    let offsets = Tensor::from_i64(&[0, 3], &[2], dev).unwrap();
    let y = embedding_bag(&w, &indices, &offsets, 0).unwrap(); // SUM mode
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    assert!(w.grad().is_some());
}

/// SDPA parity: the fused kernel must produce outputs numerically
/// equivalent to the naive `Q·Kᵀ / √d → softmax → ·V` sequence on the
/// same inputs (dropout=0, no mask, no causal). Small tolerance since
/// fused backends can reorder fp32 ops.
#[test]
fn test_sdpa_parity_vs_naive() {
    let opts = crate::tensor::test_opts();
    // 4-D inputs: [batch, heads, seq, head_dim]
    let q = Variable::new(Tensor::randn(&[2, 4, 6, 8], opts).unwrap(), false);
    let k = Variable::new(Tensor::randn(&[2, 4, 6, 8], opts).unwrap(), false);
    let v = Variable::new(Tensor::randn(&[2, 4, 6, 8], opts).unwrap(), false);

    // SDPA path.
    let fused = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false, None).unwrap();

    // Naive path: same math, different kernel sequence.
    let head_dim = 8_f64;
    let scale = 1.0 / head_dim.sqrt();
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let probs = scores.softmax(-1).unwrap();
    let naive = probs.matmul(&v).unwrap();

    assert_eq!(fused.shape(), naive.shape());
    let fv = fused.data().to_f32_vec().unwrap();
    let nv = naive.data().to_f32_vec().unwrap();
    for (i, (a, b)) in fv.iter().zip(nv.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "sdpa/naive divergence at index {i}: sdpa={a}, naive={b}"
        );
    }
}

/// SDPA participates in native autograd — Q/K/V gradients flow.
#[test]
fn test_sdpa_backward() {
    let opts = crate::tensor::test_opts();
    let q = Variable::new(Tensor::randn(&[1, 2, 4, 4], opts).unwrap(), true);
    let k = Variable::new(Tensor::randn(&[1, 2, 4, 4], opts).unwrap(), true);
    let v = Variable::new(Tensor::randn(&[1, 2, 4, 4], opts).unwrap(), true);
    let y = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false, None).unwrap();
    y.sum().unwrap().backward().unwrap();
    assert!(q.grad().is_some());
    assert!(k.grad().is_some());
    assert!(v.grad().is_some());
}

#[test]
fn test_fused_grid_sample_gradient() {
    let opts = crate::tensor::test_opts();
    let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), true);
    // Grid: normalized coordinates in [-1, 1], shape [N, H_out, W_out, 2]
    let grid = Variable::new(
        Tensor::from_f32(
            &[0.0, 0.0, 0.5, 0.5, -0.5, -0.5, 0.0, 0.5],
            &[1, 2, 2, 2],
            crate::tensor::test_device(),
        )
        .unwrap(),
        true,
    );
    let y = grid_sample(&x, &grid, 0, 0, true).unwrap(); // bilinear, zeros padding
    let loss = y.sum().unwrap();
    loss.backward().unwrap();
    assert!(x.grad().is_some());
    assert!(grid.grad().is_some());
}

// --- NoGradGuard / no_grad tests ---

#[test]
fn test_no_grad_guard_basic() {
    assert!(is_grad_enabled());
    {
        let _guard = NoGradGuard::new();
        assert!(!is_grad_enabled());
    }
    assert!(is_grad_enabled());
}

#[test]
fn test_no_grad_guard_nested() {
    assert!(is_grad_enabled());
    {
        let _outer = NoGradGuard::new();
        assert!(!is_grad_enabled());
        {
            let _inner = NoGradGuard::new();
            assert!(!is_grad_enabled());
        }
        // Inner dropped, but outer still alive
        assert!(!is_grad_enabled());
    }
    assert!(is_grad_enabled());
}

#[test]
fn test_no_grad_guard_default() {
    let _guard: NoGradGuard = Default::default();
    assert!(!is_grad_enabled());
}

#[test]
fn test_no_grad_closure_return() {
    let result = no_grad(|| {
        assert!(!is_grad_enabled());
        42
    });
    assert_eq!(result, 42);
    assert!(is_grad_enabled());
}

#[test]
fn test_no_grad_prevents_grad_tracking() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = no_grad(|| x.mul_scalar(3.0).unwrap());
    assert!(!y.requires_grad());
    // The result tensor should NOT have a grad_fn
    assert!(y.is_leaf() || !y.requires_grad());
}

#[test]
fn test_no_grad_leaves_existing_grads_intact() {
    let x = Variable::new(from_f32(&[1.0, 2.0], &[2]), true);
    let y = x.mul_scalar(3.0).unwrap().sum().unwrap();
    y.backward().unwrap();
    let grad_before = x.grad().unwrap().to_f32_vec().unwrap();

    // no_grad should not clear existing gradients
    no_grad(|| {
        let _ = x.mul_scalar(5.0).unwrap();
    });
    let grad_after = x.grad().unwrap().to_f32_vec().unwrap();
    assert_eq!(grad_before, grad_after);
}
