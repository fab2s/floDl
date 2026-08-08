use crate::autograd::{Variable, instance_norm};
use crate::tensor::{Device, Result, Tensor};

use super::Module;
use super::parameter::Parameter;

/// Instance normalization layer.
///
/// Normalizes each instance (per-channel, per-sample) independently.
/// Equivalent to PyTorch's `nn.InstanceNorm1d` / `nn.InstanceNorm2d` / `nn.InstanceNorm3d`.
///
/// Input: `[N, C, ...]` (any number of spatial dims).
/// Output: same shape, normalized per (N, C) slice.
pub struct InstanceNorm {
    weight: Option<Parameter>,
    bias: Option<Parameter>,
    /// Number of channels (C) expected in the input.
    pub num_features: i64,
    eps: f64,
    momentum: f64,
    /// Whether this layer has learnable affine parameters.
    pub affine: bool,
}

impl InstanceNorm {
    /// Create an InstanceNorm layer. If `affine`, learns weight and bias.
    pub fn new(num_features: i64, affine: bool) -> Result<Self> {
        Self::build(num_features, affine, 1e-5, 0.1, Device::CPU)
    }

    /// Create on a specific device.
    pub fn on_device(num_features: i64, affine: bool, device: Device) -> Result<Self> {
        Self::build(num_features, affine, 1e-5, 0.1, device)
    }

    fn build(
        num_features: i64,
        affine: bool,
        eps: f64,
        momentum: f64,
        device: Device,
    ) -> Result<Self> {
        let (weight, bias) = if affine {
            let w = Tensor::ones(
                &[num_features],
                crate::tensor::TensorOptions {
                    dtype: crate::tensor::DType::Float32,
                    device,
                },
            )?;
            let b = Tensor::zeros(
                &[num_features],
                crate::tensor::TensorOptions {
                    dtype: crate::tensor::DType::Float32,
                    device,
                },
            )?;
            (
                Some(Parameter::new(w, "weight")),
                Some(Parameter::new(b, "bias")),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            weight,
            bias,
            num_features,
            eps,
            momentum,
            affine,
        })
    }
}

impl Module for InstanceNorm {
    fn name(&self) -> &str {
        "instance_norm"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        instance_norm(
            input,
            self.weight.as_ref().map(|p| &p.variable),
            self.bias.as_ref().map(|p| &p.variable),
            None,
            None,
            true,
            self.momentum,
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(w.clone());
        }
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_instance_norm_forward() {
        let norm = InstanceNorm::on_device(3, true, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 3, 4, 4], test_opts()).unwrap(), false);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 4, 4]);
    }

    #[test]
    fn test_instance_norm_no_affine() {
        let norm = InstanceNorm::on_device(3, false, test_device()).unwrap();
        assert_eq!(norm.parameters().len(), 0);
        let x = Variable::new(Tensor::randn(&[1, 3, 8, 8], test_opts()).unwrap(), false);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 3, 8, 8]);
    }

    #[test]
    fn test_instance_norm_gradient() {
        let norm = InstanceNorm::on_device(3, true, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 3, 4, 4], test_opts()).unwrap(), true);
        let y = norm.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_instance_norm_3d_input() {
        let norm = InstanceNorm::on_device(4, true, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 4, 16], test_opts()).unwrap(), false);
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 4, 16]);
    }

    #[test]
    fn test_instance_norm_affine_parameters() {
        let norm = InstanceNorm::on_device(8, true, test_device()).unwrap();
        assert_eq!(norm.parameters().len(), 2);
    }
}
