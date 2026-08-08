use std::cell::Cell;

use crate::autograd::Variable;
use crate::tensor::Result;

use super::Module;

/// Inverted dropout module.
///
/// Uses a single fused `torch::dropout` kernel (1 autograd node).
/// During training: randomly zeros elements with probability `p`,
/// scales remaining by `1/(1-p)`.
/// During eval (`model.eval()`): identity function.
pub struct Dropout {
    p: f64,
    training: Cell<bool>,
}

impl Dropout {
    /// Create a dropout module with drop probability `p` (0.0 to 1.0).
    /// Use `set_training(false)` to disable during inference.
    pub fn new(p: f64) -> Self {
        Dropout {
            p,
            training: Cell::new(true),
        }
    }
}

impl Module for Dropout {
    fn name(&self) -> &str {
        "dropout"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input.clone());
        }
        let result = input.data().dropout(self.p, true)?;
        Ok(Variable::wrap(result))
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

/// 2D channel dropout — drops entire channels (feature maps) at once.
///
/// Uses a single fused `torch::feature_dropout` kernel (1 autograd node).
/// During training: randomly zeros entire channels with probability `p`,
/// scales remaining by `1/(1-p)`. Mask shape is `[B, C, 1, 1]`.
/// During eval: identity function.
///
/// Expects 4-D input `[B, C, H, W]`.
pub struct Dropout2d {
    p: f64,
    training: Cell<bool>,
}

impl Dropout2d {
    /// Create a 2D dropout module with channel drop probability `p`.
    pub fn new(p: f64) -> Self {
        Dropout2d {
            p,
            training: Cell::new(true),
        }
    }
}

impl Module for Dropout2d {
    fn name(&self) -> &str {
        "dropout2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input.clone());
        }
        let result = input.data().feature_dropout(self.p, true)?;
        Ok(Variable::wrap(result))
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

/// Alpha dropout for self-normalizing networks (SELU).
///
/// Unlike standard dropout, AlphaDropout maintains the self-normalizing
/// property by saturating dropped elements to a specific negative value
/// rather than zero, then rescaling to preserve mean and variance.
///
/// Use with SELU activation for self-normalizing networks.
pub struct AlphaDropout {
    p: f64,
    training: Cell<bool>,
}

impl AlphaDropout {
    /// Create an alpha dropout module with drop probability `p`.
    pub fn new(p: f64) -> Self {
        AlphaDropout {
            p,
            training: Cell::new(true),
        }
    }
}

impl Module for AlphaDropout {
    fn name(&self) -> &str {
        "alpha_dropout"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        if !self.training.get() || self.p == 0.0 {
            return Ok(input.clone());
        }
        // SELU constants
        let alpha = 1.6732632423543772;
        let scale = 1.0507009873554805;
        let sat = -alpha * scale; // saturation value for dropped elements

        // Generate keep mask: rand >= p means keep
        let shape = input.shape();
        let opts = crate::tensor::TensorOptions {
            dtype: crate::tensor::DType::Float32,
            device: input.data().device(),
        };
        let rand = crate::tensor::Tensor::rand(&shape, opts)?;
        let mask = rand.ge_scalar(self.p)?; // 1.0 where kept, 0.0 where dropped

        // Apply: keep * mask + sat * (1 - mask)
        let kept = input.data().mul(&mask)?;
        let inv_mask = mask.neg()?.add_scalar(1.0)?;
        let dropped = inv_mask.mul_scalar(sat)?;
        let out = kept.add(&dropped)?;

        // Affine correction to preserve mean/variance
        let a = (1.0 - self.p) * (1.0 + self.p * sat * sat);
        let a = 1.0 / a.sqrt();
        let b = -a * self.p * sat;
        let result = out.mul_scalar(a)?.add_scalar(b)?;
        Ok(Variable::wrap(result))
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Tensor, TensorOptions};

    #[test]
    fn test_dropout2d_whole_channels_zeroed() {
        let d = Dropout2d::new(0.5);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[2, 8, 4, 4], opts).unwrap(), false);

        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();

        // Each channel should be either all-zero or all-scaled
        let h = 4_usize;
        let w = 4_usize;
        let scale = 1.0 / 0.5;
        for b in 0..2_usize {
            for c in 0..8_usize {
                let start = b * 8 * h * w + c * h * w;
                let channel: Vec<f32> = data[start..start + h * w].to_vec();
                let first = channel[0];
                // All elements in channel should be equal (either 0 or scale)
                for &v in &channel {
                    assert!(
                        (v - first).abs() < 1e-5,
                        "channel [{},{}] not uniform: {} vs {}",
                        b,
                        c,
                        v,
                        first
                    );
                }
                assert!(
                    first.abs() < 1e-5 || (first - scale as f32).abs() < 1e-5,
                    "channel value should be 0 or {}: got {}",
                    scale,
                    first
                );
            }
        }
    }

    #[test]
    fn test_dropout2d_eval_identity() {
        let d = Dropout2d::new(0.5);
        d.set_training(false);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[1, 3, 4, 4], opts).unwrap(), false);

        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_alpha_dropout_eval_identity() {
        let d = AlphaDropout::new(0.5);
        d.set_training(false);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[2, 10], opts).unwrap(), false);
        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_alpha_dropout_training() {
        let d = AlphaDropout::new(0.5);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[2, 100], opts).unwrap(), false);
        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        // Some values should differ from 1.0 (dropped elements get saturated)
        let changed = data.iter().filter(|&&v| (v - 1.0).abs() > 0.1).count();
        assert!(
            changed > 0,
            "alpha dropout should modify some elements during training"
        );
    }

    #[test]
    fn test_dropout_training() {
        let d = Dropout::new(0.5);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[2, 100], opts).unwrap(), false);
        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        let zeros = data.iter().filter(|&&v| v.abs() < 1e-5).count();
        let nonzeros = data.iter().filter(|&&v| v.abs() > 1e-5).count();
        // With p=0.5, roughly half should be zeroed
        assert!(zeros > 30, "expected ~100 zeros, got {zeros}");
        assert!(nonzeros > 30, "expected ~100 nonzeros, got {nonzeros}");
        // Nonzero values should be scaled by 1/(1-p) = 2
        for &v in &data {
            if v.abs() > 1e-5 {
                assert!(
                    (v - 2.0).abs() < 1e-5,
                    "nonzero values should be 2.0 (scaled), got {v}"
                );
            }
        }
    }

    #[test]
    fn test_dropout_eval_identity() {
        let d = Dropout::new(0.5);
        d.set_training(false);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[2, 10], opts).unwrap(), false);
        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn test_dropout_p_zero_is_identity() {
        let d = Dropout::new(0.0);
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[2, 10], opts).unwrap(), false);
        let output = d.forward(&input).unwrap();
        let data = output.data().to_f32_vec().unwrap();
        assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }
}
