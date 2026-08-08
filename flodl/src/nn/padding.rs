//! Padding layers.

use crate::autograd::Variable;
use crate::tensor::Result;

use super::Module;
use super::parameter::Parameter;

/// Zero-padding module for 2D inputs.
///
/// Pads `[B, C, H, W]` inputs with zeros on each side.
/// Equivalent to PyTorch's `nn.ZeroPad2d`.
pub struct ZeroPad2d {
    padding: [i64; 4], // left, right, top, bottom
}

impl ZeroPad2d {
    /// Pad all four sides with the same amount.
    pub fn new(padding: i64) -> Self {
        Self {
            padding: [padding, padding, padding, padding],
        }
    }

    /// Pad with different amounts: `(left, right, top, bottom)`.
    pub fn asymmetric(left: i64, right: i64, top: i64, bottom: i64) -> Self {
        Self {
            padding: [left, right, top, bottom],
        }
    }
}

impl Module for ZeroPad2d {
    fn name(&self) -> &str {
        "zero_pad2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let result = input.data().pad(&self.padding, 0.0)?;
        Ok(Variable::wrap(result))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Reflection-padding module for 2D inputs.
///
/// Pads `[B, C, H, W]` inputs using reflection of the input boundary.
/// Equivalent to PyTorch's `nn.ReflectionPad2d`.
pub struct ReflectionPad2d {
    padding: [i64; 4], // left, right, top, bottom
}

impl ReflectionPad2d {
    /// Pad all four sides with the same amount.
    pub fn new(padding: i64) -> Self {
        Self {
            padding: [padding, padding, padding, padding],
        }
    }

    /// Pad with different amounts: `(left, right, top, bottom)`.
    pub fn asymmetric(left: i64, right: i64, top: i64, bottom: i64) -> Self {
        Self {
            padding: [left, right, top, bottom],
        }
    }
}

impl Module for ReflectionPad2d {
    fn name(&self) -> &str {
        "reflection_pad2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let result = input.data().pad_mode(&self.padding, 1, 0.0)?; // 1 = reflect
        Ok(Variable::wrap(result))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, Tensor, TensorOptions};

    #[test]
    fn test_zero_pad2d() {
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[1, 1, 2, 2], opts).unwrap(), false);
        let pad = ZeroPad2d::new(1);
        let y = pad.forward(&input).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);
        let data = y.data().to_f32_vec().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-5); // top-left corner
    }

    #[test]
    fn test_reflection_pad2d() {
        let device = crate::tensor::test_device();
        let input = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], device).unwrap(),
            false,
        );
        let pad = ReflectionPad2d::new(1);
        let y = pad.forward(&input).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_zero_pad2d_asymmetric() {
        let opts = TensorOptions {
            dtype: DType::Float32,
            device: crate::tensor::test_device(),
        };
        let input = Variable::new(Tensor::ones(&[1, 1, 3, 3], opts).unwrap(), false);
        let pad = ZeroPad2d::asymmetric(1, 2, 0, 3);
        let y = pad.forward(&input).unwrap();
        // Width: 1 + 3 + 2 = 6, Height: 0 + 3 + 3 = 6
        assert_eq!(y.shape(), vec![1, 1, 6, 6]);
    }

    #[test]
    fn test_reflection_pad2d_asymmetric() {
        let device = crate::tensor::test_device();
        let input = Variable::new(
            Tensor::randn(
                &[1, 1, 4, 4],
                TensorOptions {
                    dtype: DType::Float32,
                    device,
                },
            )
            .unwrap(),
            false,
        );
        let pad = ReflectionPad2d::asymmetric(1, 2, 1, 2);
        let y = pad.forward(&input).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 7, 7]);
    }

    #[test]
    fn test_zero_pad2d_no_parameters() {
        let pad = ZeroPad2d::new(2);
        assert_eq!(pad.parameters().len(), 0);
    }

    #[test]
    fn test_reflection_pad2d_values() {
        let device = crate::tensor::test_device();
        // [[1, 2], [3, 4]] with padding 1 should reflect boundary values
        let input = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], device).unwrap(),
            false,
        );
        let pad = ReflectionPad2d::new(1);
        let y = pad.forward(&input).unwrap();
        let data = y.data().to_f32_vec().unwrap();
        // Center should contain original values
        // Row 1, Col 1 = [1], Row 1, Col 2 = [2]
        assert!((data[5] - 1.0).abs() < 1e-5); // [1,1] in 4x4
        assert!((data[6] - 2.0).abs() < 1e-5); // [1,2] in 4x4
    }
}
