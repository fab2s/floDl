use crate::tensor::{Device, Result, Tensor, TensorError, TensorOptions};

/// Validate grouped-convolution channel arithmetic (shared by all six
/// conv / conv-transpose constructors). PyTorch parity: `groups` must be
/// positive and divide both channel counts — `groups = 0` would be an i64
/// division by zero, and a non-dividing value would silently truncate the
/// weight shape (integer division) into a wrong-shaped kernel.
pub(crate) fn validate_conv_groups(
    what: &str,
    in_channels: i64,
    out_channels: i64,
    groups: i64,
) -> Result<()> {
    if groups < 1 {
        return Err(TensorError::new(&format!(
            "{what}: groups must be a positive integer, got {groups}"
        )));
    }
    if in_channels % groups != 0 {
        return Err(TensorError::new(&format!(
            "{what}: in_channels ({in_channels}) must be divisible by groups ({groups})"
        )));
    }
    if out_channels % groups != 0 {
        return Err(TensorError::new(&format!(
            "{what}: out_channels ({out_channels}) must be divisible by groups ({groups})"
        )));
    }
    Ok(())
}

/// Kaiming uniform initialization.
///
/// `a` is the negative slope of the nonlinearity (0 for ReLU, sqrt(5) for PyTorch Linear default).
/// - `a = 0.0` (ReLU):     bound = sqrt(6 / fan_in)       — standard He init
/// - `a = sqrt(5)` (Linear): bound = 1 / sqrt(fan_in)       — matches PyTorch nn.Linear
pub fn kaiming_uniform(shape: &[i64], fan_in: i64, a: f64, device: Device) -> Result<Tensor> {
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (fan_in as f64).sqrt();
    let bound = 3.0_f64.sqrt() * std;
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Kaiming normal initialization.
///
/// `a` is the negative slope of the nonlinearity (0 for ReLU, sqrt(5) for PyTorch Linear default).
/// - `a = 0.0` (ReLU):     std = sqrt(2 / fan_in)           — standard He init
/// - `a = sqrt(5)` (Linear): std = 1 / sqrt(3 * fan_in)       — matches PyTorch nn.Linear
pub fn kaiming_normal(shape: &[i64], fan_in: i64, a: f64, device: Device) -> Result<Tensor> {
    let gain = (2.0 / (1.0 + a * a)).sqrt();
    let std = gain / (fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)
}

/// Xavier (Glorot) uniform initialization (for sigmoid/tanh networks).
/// bound = sqrt(6 / (fan_in + fan_out)), uniform(-bound, bound)
pub fn xavier_uniform(shape: &[i64], fan_in: i64, fan_out: i64, device: Device) -> Result<Tensor> {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Xavier (Glorot) normal initialization (for sigmoid/tanh networks).
/// std = sqrt(2 / (fan_in + fan_out)), normal(0, std)
pub fn xavier_normal(shape: &[i64], fan_in: i64, fan_out: i64, device: Device) -> Result<Tensor> {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)
}

/// Uniform bias initialization.
/// bound = 1 / sqrt(fan_in), uniform(-bound, bound)
pub fn uniform_bias(fan_in: i64, shape: &[i64], device: Device) -> Result<Tensor> {
    let bound = 1.0 / (fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Uniform initialization: values drawn from U[low, high).
///
/// ```ignore
/// let t = uniform(&[3, 4], -1.0, 1.0, Device::CPU)?;
/// ```
pub fn uniform(shape: &[i64], low: f64, high: f64, device: Device) -> Result<Tensor> {
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(high - low)?.add_scalar(low)
}

/// Normal initialization: values drawn from N(mean, std).
///
/// ```ignore
/// let t = normal(&[3, 4], 0.0, 0.02, Device::CPU)?;
/// ```
pub fn normal(shape: &[i64], mean: f64, std: f64, device: Device) -> Result<Tensor> {
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)?.add_scalar(mean)
}

/// Orthogonal initialization via Gram-Schmidt orthogonalization.
///
/// Produces a matrix with orthonormal rows (if rows <= cols) or columns
/// (if rows > cols), scaled by `gain`. This is the standard initialization
/// for RNNs and other architectures that benefit from preserving gradient norms.
///
/// Requires a 2D shape. For higher-dimensional tensors, reshape first.
///
/// ```ignore
/// let t = orthogonal(&[4, 4], 1.0, Device::CPU)?;
/// // t @ t^T is approximately identity
/// ```
pub fn orthogonal(shape: &[i64], gain: f64, device: Device) -> Result<Tensor> {
    assert!(shape.len() == 2, "orthogonal init requires a 2D shape");
    let rows = shape[0];
    let cols = shape[1];
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };

    // Generate a square random matrix of size max(rows, cols) and orthogonalize
    // via Gram-Schmidt, then slice to the requested shape.
    // This matches PyTorch's approach (QR of random matrix, then slice).
    let n = rows.max(cols) as usize;

    let a = Tensor::randn(&[n as i64, n as i64], opts)?;
    let mut data = a.to_f32_vec()?;

    // Gram-Schmidt on rows: for each row i, subtract projections onto
    // all previous orthonormalized rows, then normalize.
    for i in 0..n {
        let row_start = i * n;
        for j in 0..i {
            let prev_start = j * n;
            let mut dot = 0.0f64;
            for k in 0..n {
                dot += data[row_start + k] as f64 * data[prev_start + k] as f64;
            }
            for k in 0..n {
                data[row_start + k] -= (dot * data[prev_start + k] as f64) as f32;
            }
        }
        let mut norm = 0.0f64;
        for k in 0..n {
            norm += data[row_start + k] as f64 * data[row_start + k] as f64;
        }
        let norm = norm.sqrt().max(1e-10);
        for k in 0..n {
            data[row_start + k] = (data[row_start + k] as f64 / norm) as f32;
        }
    }

    // Reconstruct full orthogonal matrix [n, n]
    let q = Tensor::from_f32(&data, &[n as i64, n as i64], device)?;

    // Slice to [rows, cols]
    let result = q.narrow(0, 0, rows)?.narrow(1, 0, cols)?.contiguous()?;

    if (gain - 1.0).abs() > 1e-10 {
        result.mul_scalar(gain)
    } else {
        Ok(result)
    }
}

/// Truncated normal initialization: values drawn from N(mean, std), clamped to [a, b].
///
/// Values outside `[mean + a * std, mean + b * std]` are clamped. This matches the
/// commonly used approach (e.g., in Vision Transformers) where `a=-2, b=2` clips
/// values beyond 2 standard deviations.
///
/// Note: clamping slightly distorts the distribution near the bounds compared to
/// true rejection sampling, but this is the standard practical implementation
/// used by most frameworks.
///
/// ```ignore
/// let t = trunc_normal(&[768, 768], 0.0, 0.02, -2.0, 2.0, Device::CPU)?;
/// ```
pub fn trunc_normal(
    shape: &[i64],
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
    device: Device,
) -> Result<Tensor> {
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    let low = mean + a * std;
    let high = mean + b * std;
    Tensor::randn(shape, opts)?.mul_scalar(std)?.add_scalar(mean)?.clamp(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_range() {
        let t = uniform(&[1000], -2.0, 3.0, crate::tensor::test_device()).unwrap();
        let data = t.to_f32_vec().unwrap();
        for &v in &data {
            assert!((-2.0..=3.0).contains(&v), "value {} out of range [-2, 3]", v);
        }
        // Mean should be approximately 0.5
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 0.5).abs() < 0.2, "mean {} too far from 0.5", mean);
    }

    #[test]
    fn test_normal_stats() {
        let t = normal(&[10000], 5.0, 0.1, crate::tensor::test_device()).unwrap();
        let data = t.to_f32_vec().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 5.0).abs() < 0.05, "mean {} too far from 5.0", mean);

        let var: f32 = data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / data.len() as f32;
        let std_dev = var.sqrt();
        assert!((std_dev - 0.1).abs() < 0.02, "std {} too far from 0.1", std_dev);
    }

    #[test]
    fn test_orthogonal_square() {
        let t = orthogonal(&[4, 4], 1.0, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![4, 4]);

        // Q @ Q^T should be approximately identity
        let qt = t.transpose(0, 1).unwrap();
        let qqt = t.matmul(&qt).unwrap();
        let data = qqt.to_f32_vec().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (data[i * 4 + j] - expected).abs() < 0.01,
                    "Q@Q^T[{},{}] = {}, expected {}",
                    i, j, data[i * 4 + j], expected
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_tall() {
        // More rows than columns: columns should be orthonormal
        let t = orthogonal(&[6, 4], 1.0, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![6, 4]);

        let qt = t.transpose(0, 1).unwrap();
        let qtq = qt.matmul(&t).unwrap();
        let data = qtq.to_f32_vec().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (data[i * 4 + j] - expected).abs() < 0.01,
                    "Q^T@Q[{},{}] = {}, expected {}",
                    i, j, data[i * 4 + j], expected
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_wide() {
        // More columns than rows: rows should be orthonormal
        let t = orthogonal(&[4, 6], 1.0, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![4, 6]);

        let qt = t.transpose(0, 1).unwrap();
        let qqt = t.matmul(&qt).unwrap();
        let data = qqt.to_f32_vec().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (data[i * 4 + j] - expected).abs() < 0.01,
                    "Q@Q^T[{},{}] = {}, expected {}",
                    i, j, data[i * 4 + j], expected
                );
            }
        }
    }

    #[test]
    fn test_orthogonal_gain() {
        let t = orthogonal(&[4, 4], 2.0, crate::tensor::test_device()).unwrap();
        // Row norms should be approximately gain=2.0
        let data = t.to_f32_vec().unwrap();
        for i in 0..4 {
            let norm: f32 = (0..4).map(|j| data[i * 4 + j] * data[i * 4 + j]).sum::<f32>().sqrt();
            assert!(
                (norm - 2.0).abs() < 0.1,
                "row {} norm = {}, expected ~2.0", i, norm
            );
        }
    }

    #[test]
    fn test_trunc_normal_bounds() {
        let t = trunc_normal(&[10000], 0.0, 1.0, -2.0, 2.0, crate::tensor::test_device()).unwrap();
        let data = t.to_f32_vec().unwrap();
        for &v in &data {
            assert!((-2.0..=2.0).contains(&v), "value {} out of [-2, 2]", v);
        }
    }

    #[test]
    fn test_trunc_normal_centered() {
        let t = trunc_normal(&[10000], 3.0, 0.5, -2.0, 2.0, crate::tensor::test_device()).unwrap();
        let data = t.to_f32_vec().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        // Mean should be near 3.0
        assert!((mean - 3.0).abs() < 0.1, "mean {} too far from 3.0", mean);
        // All values in [3 + (-2)*0.5, 3 + 2*0.5] = [2.0, 4.0]
        for &v in &data {
            assert!((2.0..=4.0).contains(&v), "value {} out of [2.0, 4.0]", v);
        }
    }

    #[test]
    fn test_kaiming_uniform_shape() {
        let t = kaiming_uniform(&[3, 4], 4, 0.0, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![3, 4]);
    }

    #[test]
    fn test_xavier_uniform_shape() {
        let t = xavier_uniform(&[3, 4], 3, 4, crate::tensor::test_device()).unwrap();
        assert_eq!(t.shape(), vec![3, 4]);
    }
}
