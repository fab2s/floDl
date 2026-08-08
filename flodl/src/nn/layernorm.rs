use crate::autograd::{Variable, layer_norm};
use crate::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::Module;
use super::parameter::Parameter;

/// Layer normalization over the last dimension.
///
/// Uses native libtorch `layer_norm` for PyTorch numerical parity.
pub struct LayerNorm {
    pub weight: Parameter, // gamma
    pub bias: Parameter,   // beta
    size: i64,
    eps: f64,
}

impl LayerNorm {
    /// Default epsilon, matching PyTorch `nn.LayerNorm`.
    pub const DEFAULT_EPS: f64 = 1e-5;

    /// Create a LayerNorm normalizing over the last `size` elements on CPU.
    ///
    /// Uses the PyTorch default `eps = 1e-5`. For HuggingFace BERT-style
    /// checkpoints (which use `eps = 1e-12`), construct with
    /// [`LayerNorm::with_eps`] instead.
    pub fn new(size: i64) -> Result<Self> {
        Self::on_device_with_eps(size, Self::DEFAULT_EPS, Device::CPU)
    }

    /// Create a LayerNorm on CPU with a custom epsilon.
    ///
    /// Required for parity with HuggingFace BERT (`eps = 1e-12`) and any
    /// architecture that deviates from the PyTorch default.
    pub fn with_eps(size: i64, eps: f64) -> Result<Self> {
        Self::on_device_with_eps(size, eps, Device::CPU)
    }

    /// Create a LayerNorm on a specific device with the default epsilon.
    pub fn on_device(size: i64, device: Device) -> Result<Self> {
        Self::on_device_with_eps(size, Self::DEFAULT_EPS, device)
    }

    /// Create a LayerNorm on a specific device with a custom epsilon.
    pub fn on_device_with_eps(size: i64, eps: f64, device: Device) -> Result<Self> {
        let opts = TensorOptions {
            dtype: DType::Float32,
            device,
        };
        let weight = Variable::new(Tensor::ones(&[size], opts)?, true);
        let bias = Variable::new(Tensor::zeros(&[size], opts)?, true);

        Ok(LayerNorm {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            bias: Parameter {
                variable: bias,
                name: "bias".into(),
            },
            size,
            eps,
        })
    }
}

impl Module for LayerNorm {
    fn name(&self) -> &str {
        "layernorm"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        layer_norm(
            input,
            &self.weight.variable,
            &self.bias.variable,
            self.size,
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_layernorm_forward_shape() {
        let ln = LayerNorm::on_device(8, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 8], test_opts()).unwrap(), false);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 8]);
    }

    #[test]
    fn test_layernorm_normalizes() {
        let ln = LayerNorm::on_device(4, test_device()).unwrap();
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], test_device()).unwrap(),
            false,
        );
        let y = ln.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Output should be approximately normalized (mean ~0, std ~1)
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
    }

    #[test]
    fn test_layernorm_3d_input() {
        let ln = LayerNorm::on_device(16, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 5, 16], test_opts()).unwrap(), false);
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 5, 16]);
    }

    #[test]
    fn test_layernorm_gradient() {
        let ln = LayerNorm::on_device(8, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[4, 8], test_opts()).unwrap(), true);
        let y = ln.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
        // Weight and bias should also have gradients
        assert!(ln.weight.variable.grad().is_some());
        assert!(ln.bias.variable.grad().is_some());
    }

    #[test]
    fn test_layernorm_custom_eps_reaches_forward() {
        // With a very large eps, normalization is heavily smoothed: the
        // denominator sqrt(var + eps) is dominated by eps, so outputs stay
        // much closer to the centered-but-not-scaled input.
        let ln_huge = LayerNorm::on_device_with_eps(4, 1e6, test_device()).unwrap();
        let ln_def = LayerNorm::on_device(4, test_device()).unwrap();
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], test_device()).unwrap(),
            false,
        );
        let y_huge = ln_huge.forward(&x).unwrap().data().to_f32_vec().unwrap();
        let y_def = ln_def.forward(&x).unwrap().data().to_f32_vec().unwrap();
        // Default eps gives a properly normalized output (std ~= 1).
        let std_def = (y_def.iter().map(|v| v * v).sum::<f32>() / y_def.len() as f32).sqrt();
        assert!(
            (std_def - 1.0).abs() < 0.2,
            "default-eps std ~1, got {std_def}"
        );
        // Huge eps collapses the scaling: output magnitude << default.
        let max_huge = y_huge.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        assert!(
            max_huge < 0.01,
            "huge-eps output should be tiny, got max {max_huge}"
        );
    }

    #[test]
    fn test_layernorm_bert_eps_constructor() {
        let ln = LayerNorm::on_device_with_eps(8, 1e-12, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 8], test_opts()).unwrap(), false);
        // Smoke: constructor wires through and forward runs without panic.
        let y = ln.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 8]);
    }

    /// Hand-computed golden values: for input [1, 2, 3, 4] with weight=1,
    /// bias=0, LayerNorm computes centered/sqrt(variance + eps).
    /// mean=2.5, variance=1.25, centered=[-1.5, -0.5, 0.5, 1.5].
    /// At eps=1e-12, std ≈ sqrt(1.25) ≈ 1.1180339887, giving the values below.
    /// This anchors BOTH the implementation correctness AND that the eps
    /// parameter actually flows through to the libtorch kernel.
    #[test]
    fn test_layernorm_bert_eps_golden_values() {
        let ln = LayerNorm::on_device_with_eps(4, 1e-12, test_device()).unwrap();
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], test_device()).unwrap(),
            false,
        );
        let y = ln.forward(&x).unwrap().data().to_f32_vec().unwrap();
        let expected = [
            -1.341_640_8_f32,
            -0.447_213_6_f32,
            0.447_213_6_f32,
            1.341_640_8_f32,
        ];
        for (i, (got, exp)) in y.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "bert-eps layernorm dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_layernorm_parameters() {
        let ln = LayerNorm::on_device(16, test_device()).unwrap();
        let params = ln.parameters();
        assert_eq!(params.len(), 2);
        // Weight should be ones, bias should be zeros
        let w = params[0].variable.data().to_f32_vec().unwrap();
        let b = params[1].variable.data().to_f32_vec().unwrap();
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(b.iter().all(|&v| v.abs() < 1e-6));
    }
}
