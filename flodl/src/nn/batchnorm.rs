use std::cell::Cell;

use crate::autograd::Variable;
use crate::tensor::{Result, Tensor, TensorError, TensorOptions};

use super::Module;
use super::buffer::Buffer;
use super::parameter::Parameter;

/// Batch normalization over the first (batch) dimension.
///
/// Uses a single fused `torch::batch_norm` kernel (1 autograd node).
/// Training mode: uses batch statistics, updates running stats in-place.
/// Eval mode: uses running statistics (persisted via `buffers()`).
pub struct BatchNorm {
    pub weight: Parameter, // gamma
    pub bias: Parameter,   // beta
    running_mean: Buffer,
    running_var: Buffer,
    #[allow(dead_code)]
    num_features: i64,
    eps: f64,
    momentum: f64,
    training: Cell<bool>,
}

impl BatchNorm {
    /// Create a BatchNorm layer for `num_features` channels/features on CPU.
    pub fn new(num_features: i64) -> Result<Self> {
        Self::on_device(num_features, crate::tensor::Device::CPU)
    }

    /// Create a BatchNorm layer on a specific device.
    pub fn on_device(num_features: i64, device: crate::tensor::Device) -> Result<Self> {
        let opts = TensorOptions {
            dtype: crate::tensor::DType::Float32,
            device,
        };
        let weight = Variable::new(Tensor::ones(&[num_features], opts)?, true);
        let bias = Variable::new(Tensor::zeros(&[num_features], opts)?, true);

        Ok(BatchNorm {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            bias: Parameter {
                variable: bias,
                name: "bias".into(),
            },
            running_mean: Buffer::new(Tensor::zeros(&[num_features], opts)?, "running_mean"),
            running_var: Buffer::new(Tensor::ones(&[num_features], opts)?, "running_var"),
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            training: Cell::new(true),
        })
    }
}

/// Batch normalization for 4D `[B, C, H, W]` inputs (conv layers).
///
/// Uses fused `torch::batch_norm` which handles 4D input natively —
/// normalizes over `(B, H, W)` dimensions per channel, matching
/// PyTorch's `nn.BatchNorm2d`.
pub struct BatchNorm2d {
    inner: BatchNorm,
}

impl BatchNorm2d {
    /// Create a BatchNorm2d layer for `num_channels` channels on CPU.
    pub fn new(num_channels: i64) -> Result<Self> {
        Ok(Self {
            inner: BatchNorm::new(num_channels)?,
        })
    }

    /// Create a BatchNorm2d layer on a specific device.
    pub fn on_device(num_channels: i64, device: crate::tensor::Device) -> Result<Self> {
        Ok(Self {
            inner: BatchNorm::on_device(num_channels, device)?,
        })
    }
}

impl Module for BatchNorm2d {
    fn name(&self) -> &str {
        "batchnorm2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(TensorError::new(
                "BatchNorm2d: input must be 4D [B, C, H, W]",
            ));
        }
        // Fused batch_norm handles 4D input natively
        self.inner.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.inner.parameters()
    }

    fn buffers(&self) -> Vec<Buffer> {
        self.inner.buffers()
    }

    fn set_training(&self, training: bool) {
        self.inner.set_training(training);
    }
}

impl Module for BatchNorm {
    fn name(&self) -> &str {
        "batchnorm"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let training = self.training.get();
        if training {
            let batch_size = input.shape()[0];
            if batch_size < 2 {
                return Err(TensorError::new(
                    "BatchNorm requires batch_size >= 2 in training mode \
                     (Bessel's correction divides by batch_size-1)",
                ));
            }
        }

        // Single fused call: computes norm, applies affine, updates running stats.
        // Running stats are updated in-place via shared storage when training=true.
        let rm = self.running_mean.get();
        let rv = self.running_var.get();
        let result = input.data().batch_norm(
            Some(&self.weight.variable.data()),
            Some(&self.bias.variable.data()),
            Some(&rm),
            Some(&rv),
            training,
            self.momentum,
            self.eps,
        )?;
        Ok(Variable::wrap(result))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn buffers(&self) -> Vec<Buffer> {
        vec![self.running_mean.clone(), self.running_var.clone()]
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_batchnorm_forward_training() {
        let bn = BatchNorm::on_device(4, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[8, 4], test_opts()).unwrap(), false);
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![8, 4]);
    }

    #[test]
    fn test_batchnorm_eval_mode() {
        let bn = BatchNorm::on_device(4, test_device()).unwrap();
        // First run in training to populate running stats
        let x = Variable::new(Tensor::randn(&[8, 4], test_opts()).unwrap(), false);
        let _ = bn.forward(&x).unwrap();
        // Switch to eval
        bn.set_training(false);
        // Eval should work with batch_size=1
        let x_single = Variable::new(Tensor::randn(&[1, 4], test_opts()).unwrap(), false);
        let y = bn.forward(&x_single).unwrap();
        assert_eq!(y.shape(), vec![1, 4]);
    }

    #[test]
    fn test_batchnorm_training_requires_batch_ge_2() {
        let bn = BatchNorm::on_device(4, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[1, 4], test_opts()).unwrap(), false);
        assert!(bn.forward(&x).is_err());
    }

    #[test]
    fn test_batchnorm_running_stats_update() {
        let bn = BatchNorm::on_device(2, test_device()).unwrap();
        // Initial running_mean should be zero
        let rm_before = bn.buffers()[0].get().to_f32_vec().unwrap();
        assert!(rm_before.iter().all(|&v| v.abs() < 1e-6));

        // Forward with non-zero-mean data should update running_mean
        let x = Variable::new(
            Tensor::from_f32(
                &[10.0, 20.0, 12.0, 22.0, 11.0, 21.0, 9.0, 19.0],
                &[4, 2],
                test_device(),
            )
            .unwrap(),
            false,
        );
        let _ = bn.forward(&x).unwrap();
        let rm_after = bn.buffers()[0].get().to_f32_vec().unwrap();
        // Running mean should have moved toward batch mean (~10.5, ~20.5)
        assert!(
            rm_after[0].abs() > 0.5,
            "running_mean should have updated: {}",
            rm_after[0]
        );
    }

    #[test]
    fn test_batchnorm_parameters() {
        let bn = BatchNorm::on_device(8, test_device()).unwrap();
        assert_eq!(bn.parameters().len(), 2); // weight + bias
        assert_eq!(bn.buffers().len(), 2); // running_mean + running_var
    }

    #[test]
    fn test_batchnorm_gradient() {
        let bn = BatchNorm::on_device(3, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[4, 3], test_opts()).unwrap(), true);
        let y = bn.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_batchnorm2d_forward() {
        let bn = BatchNorm2d::on_device(3, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 3, 4, 4], test_opts()).unwrap(), false);
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 4, 4]);
    }

    #[test]
    fn test_batchnorm2d_rejects_non_4d() {
        let bn = BatchNorm2d::on_device(3, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 3], test_opts()).unwrap(), false);
        assert!(bn.forward(&x).is_err());
    }

    #[test]
    fn test_batchnorm2d_training_eval_differ() {
        let bn = BatchNorm2d::on_device(2, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[4, 2, 3, 3], test_opts()).unwrap(), false);
        let train_out = bn.forward(&x).unwrap().data().to_f32_vec().unwrap();

        // Populate running stats, then switch to eval
        bn.set_training(false);
        let eval_out = bn.forward(&x).unwrap().data().to_f32_vec().unwrap();

        // Outputs should differ (training uses batch stats, eval uses running stats)
        let diff: f32 = train_out
            .iter()
            .zip(&eval_out)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "train and eval outputs should differ");
    }
}
