use crate::tensor::{Result, Tensor};

use super::Parameter;

/// Clip gradients by total L2 norm. Scales all gradients so that the
/// total norm does not exceed `max_norm`. Returns the original total norm.
///
/// Uses a single fused C++ call — computes global norm across all params
/// and scales in-place. No per-param FFI roundtrips.
pub fn clip_grad_norm(params: &[Parameter], max_norm: f64) -> Result<f64> {
    let handles: Vec<_> = params.iter().map(|p| p.variable.data()).collect();
    Tensor::clip_grad_norm_fused(&handles, max_norm)
}

/// Clip gradients element-wise to `[-max_val, max_val]`.
/// Returns the maximum absolute gradient value before clipping.
///
/// Clamping stays on-device. Only one scalar per parameter is read
/// to track the global maximum.
pub fn clip_grad_value(params: &[Parameter], max_val: f64) -> Result<f64> {
    let mut global_max = 0.0f64;

    for p in params {
        if let Some(grad) = p.variable.grad() {
            // Read one scalar per param to track max (tiny transfer)
            let local_max = grad.abs()?.max()?.item()?;
            if local_max > global_max {
                global_max = local_max;
            }
            // Clamp on-device — no data copy
            p.variable.set_grad(grad.clamp(-max_val, max_val)?);
        }
    }

    Ok(global_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::tensor::{Tensor, test_device};

    fn make_param_with_grad(values: &[f32], grad_values: &[f32]) -> Parameter {
        let t = Tensor::from_f32(values, &[values.len() as i64], test_device()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        let g = Tensor::from_f32(grad_values, &[grad_values.len() as i64], test_device()).unwrap();
        p.variable.set_grad(g);
        p
    }

    #[test]
    fn test_clip_grad_norm_scales_down() {
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, 4.0]);
        // L2 norm of [3,4] = 5.0, max_norm = 1.0 -> scale by 1/5
        let original_norm = clip_grad_norm(std::slice::from_ref(&p), 1.0).unwrap();
        assert!((original_norm - 5.0).abs() < 1e-3);
        let g = p.variable.grad().unwrap().to_f32_vec().unwrap();
        let clipped_norm: f64 = g.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_clip_grad_norm_no_op_when_small() {
        let p = make_param_with_grad(&[1.0], &[0.5]);
        // L2 norm = 0.5, max_norm = 10.0 -> no clipping
        let norm = clip_grad_norm(std::slice::from_ref(&p), 10.0).unwrap();
        assert!((norm - 0.5).abs() < 1e-3);
        let g = p.variable.grad().unwrap().to_f32_vec().unwrap();
        assert!((g[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_multiple_params() {
        let p1 = make_param_with_grad(&[1.0], &[3.0]);
        let p2 = make_param_with_grad(&[1.0], &[4.0]);
        // Global L2 norm = sqrt(9+16) = 5.0
        let norm = clip_grad_norm(&[p1, p2], 1.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_clip_grad_value_clamps() {
        let p = make_param_with_grad(&[1.0, 2.0], &[10.0, -5.0]);
        let max_before = clip_grad_value(std::slice::from_ref(&p), 2.0).unwrap();
        assert!((max_before - 10.0).abs() < 1e-3);
        let g = p.variable.grad().unwrap().to_f32_vec().unwrap();
        assert!((g[0] - 2.0).abs() < 1e-4); // clamped from 10
        assert!((g[1] - (-2.0)).abs() < 1e-4); // clamped from -5
    }

    #[test]
    fn test_clip_grad_value_no_op_when_small() {
        let p = make_param_with_grad(&[1.0], &[0.3]);
        let max = clip_grad_value(std::slice::from_ref(&p), 1.0).unwrap();
        assert!((max - 0.3).abs() < 1e-3);
        let g = p.variable.grad().unwrap().to_f32_vec().unwrap();
        assert!((g[0] - 0.3).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_value_no_grad() {
        // Param with no gradient should be skipped
        let t = Tensor::from_f32(&[1.0], &[1], test_device()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        let max = clip_grad_value(&[p], 1.0).unwrap();
        assert_eq!(max, 0.0);
    }
}
