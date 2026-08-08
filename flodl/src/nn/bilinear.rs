use crate::autograd::{Variable, bilinear};
use crate::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::Module;
use super::parameter::Parameter;

/// Bilinear transformation: `y = x1^T A x2 + b`.
///
/// Weight shape: `[out_features, in1_features, in2_features]`.
/// Bias shape: `[out_features]` (optional).
///
/// Equivalent to PyTorch's `nn.Bilinear`.
pub struct Bilinear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
}

impl Bilinear {
    /// Create a Bilinear layer.
    pub fn new(
        in1_features: i64,
        in2_features: i64,
        out_features: i64,
        with_bias: bool,
    ) -> Result<Self> {
        Self::build(
            in1_features,
            in2_features,
            out_features,
            with_bias,
            Device::CPU,
        )
    }

    /// Create on a specific device.
    pub fn on_device(
        in1_features: i64,
        in2_features: i64,
        out_features: i64,
        device: Device,
    ) -> Result<Self> {
        Self::build(in1_features, in2_features, out_features, true, device)
    }

    fn build(
        in1_features: i64,
        in2_features: i64,
        out_features: i64,
        with_bias: bool,
        device: Device,
    ) -> Result<Self> {
        let opts = TensorOptions {
            dtype: DType::Float32,
            device,
        };
        let bound = 1.0 / (in1_features as f64).sqrt();
        let w = Tensor::rand(&[out_features, in1_features, in2_features], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let weight = Parameter::new(w, "weight");

        let bias = if with_bias {
            let b = Tensor::rand(&[out_features], opts)?
                .mul_scalar(2.0 * bound)?
                .add_scalar(-bound)?;
            Some(Parameter::new(b, "bias"))
        } else {
            None
        };

        Ok(Self { weight, bias })
    }
}

impl Module for Bilinear {
    fn name(&self) -> &str {
        "bilinear"
    }

    fn forward(&self, _input: &Variable) -> Result<Variable> {
        Err(crate::tensor::TensorError::new(
            "Bilinear requires two inputs; use forward_bilinear(x1, x2) instead",
        ))
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

impl Bilinear {
    /// Forward pass with two inputs.
    pub fn forward_bilinear(&self, input1: &Variable, input2: &Variable) -> Result<Variable> {
        bilinear(
            input1,
            input2,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_bilinear_forward() {
        let layer = Bilinear::on_device(3, 4, 5, test_device()).unwrap();
        let x1 = Variable::new(Tensor::randn(&[2, 3], test_opts()).unwrap(), false);
        let x2 = Variable::new(Tensor::randn(&[2, 4], test_opts()).unwrap(), false);
        let y = layer.forward_bilinear(&x1, &x2).unwrap();
        assert_eq!(y.shape(), vec![2, 5]);
    }

    #[test]
    fn test_bilinear_params() {
        let layer = Bilinear::on_device(3, 4, 5, test_device()).unwrap();
        assert_eq!(layer.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_bilinear_gradient() {
        let layer = Bilinear::on_device(3, 4, 5, test_device()).unwrap();
        let x1 = Variable::new(Tensor::randn(&[2, 3], test_opts()).unwrap(), true);
        let x2 = Variable::new(Tensor::randn(&[2, 4], test_opts()).unwrap(), true);
        let y = layer.forward_bilinear(&x1, &x2).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x1.grad().is_some());
        assert!(x2.grad().is_some());
        assert!(layer.weight.variable.grad().is_some());
    }

    #[test]
    fn test_bilinear_no_bias() {
        let layer = Bilinear::build(3, 4, 5, false, test_device()).unwrap();
        assert_eq!(layer.parameters().len(), 1); // weight only
        assert!(layer.bias.is_none());
        let x1 = Variable::new(Tensor::randn(&[2, 3], test_opts()).unwrap(), false);
        let x2 = Variable::new(Tensor::randn(&[2, 4], test_opts()).unwrap(), false);
        let y = layer.forward_bilinear(&x1, &x2).unwrap();
        assert_eq!(y.shape(), vec![2, 5]);
    }

    #[test]
    fn test_bilinear_forward_rejects_single_input() {
        let layer = Bilinear::on_device(3, 4, 5, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 3], test_opts()).unwrap(), false);
        assert!(layer.forward(&x).is_err());
    }
}
