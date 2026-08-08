use crate::autograd::Variable;
use crate::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::Module;
use super::parameter::Parameter;

/// Root Mean Square Layer Normalization.
///
/// Normalizes the last dimension without centering (no bias subtraction),
/// making it cheaper than LayerNorm. Used in LLaMA, Gemma, and other
/// modern architectures.
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * weight`
pub struct RMSNorm {
    pub weight: Parameter,
    eps: f64,
}

impl RMSNorm {
    /// Create an RMSNorm normalizing over the last `size` elements on CPU.
    pub fn new(size: i64) -> Result<Self> {
        Self::on_device(size, Device::CPU)
    }

    /// Create an RMSNorm on a specific device.
    pub fn on_device(size: i64, device: Device) -> Result<Self> {
        let opts = TensorOptions {
            dtype: DType::Float32,
            device,
        };
        let weight = Variable::new(Tensor::ones(&[size], opts)?, true);

        Ok(RMSNorm {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            eps: 1e-5,
        })
    }

    /// Set epsilon for numerical stability (default: 1e-5).
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Module for RMSNorm {
    fn name(&self) -> &str {
        "rmsnorm"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // RMS = sqrt(mean(x^2) + eps)
        let x2 = input.pow_scalar(2.0)?;
        let mean_x2 = x2.mean_dim(-1, true)?;
        let rms = mean_x2.add_scalar(self.eps)?.sqrt()?;
        let normalized = input.div(&rms)?;
        normalized.mul(&self.weight.variable)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::test_device;

    #[test]
    fn test_rmsnorm_shape() {
        let device = test_device();
        let norm = RMSNorm::on_device(4, device).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 3, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 4]);
    }

    #[test]
    fn test_rmsnorm_unit_rms() {
        let device = test_device();
        let norm = RMSNorm::on_device(4, device).unwrap();
        // Input with known RMS: [1, 1, 1, 1] → RMS=1 → output ≈ [1, 1, 1, 1]
        let x = Variable::new(
            Tensor::ones(&[1, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let y = norm.forward(&x).unwrap().data().to_f32_vec().unwrap();
        for v in &y {
            assert!((v - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_rmsnorm_gradient() {
        let device = test_device();
        let norm = RMSNorm::on_device(4, device).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let y = norm.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![2, 4]);
        // Weight should also have gradients
        let w_grad = norm.weight.variable.grad().unwrap();
        assert_eq!(w_grad.shape(), vec![4]);
    }

    #[test]
    fn test_rmsnorm_parameters() {
        let norm = RMSNorm::new(8).unwrap();
        let params = norm.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, "weight");
    }
}
