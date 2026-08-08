use crate::autograd::{Variable, group_norm};
use crate::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::Module;
use super::parameter::Parameter;

/// Group normalization over channel groups.
///
/// Divides channels into `num_groups` groups and normalizes within each group.
/// Uses native libtorch `group_norm` for PyTorch numerical parity.
///
/// Input: `[N, C, *]` where C must be divisible by `num_groups`.
/// Output: same shape as input.
///
/// ```ignore
/// let gn = GroupNorm::new(4, 16)?; // 4 groups, 16 channels
/// let x = Variable::new(Tensor::randn(&[2, 16, 8, 8], opts)?, false);
/// let y = gn.forward(&x)?; // [2, 16, 8, 8]
/// ```
pub struct GroupNorm {
    pub weight: Parameter, // gamma, shape [num_channels]
    pub bias: Parameter,   // beta, shape [num_channels]
    num_groups: i64,
    eps: f64,
}

impl GroupNorm {
    /// Create a GroupNorm normalizing `num_channels` across `num_groups` groups on CPU.
    pub fn new(num_groups: i64, num_channels: i64) -> Result<Self> {
        Self::on_device(num_groups, num_channels, Device::CPU)
    }

    /// Create a GroupNorm on a specific device.
    pub fn on_device(num_groups: i64, num_channels: i64, device: Device) -> Result<Self> {
        let opts = TensorOptions {
            dtype: DType::Float32,
            device,
        };
        let weight = Variable::new(Tensor::ones(&[num_channels], opts)?, true);
        let bias = Variable::new(Tensor::zeros(&[num_channels], opts)?, true);

        Ok(GroupNorm {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            bias: Parameter {
                variable: bias,
                name: "bias".into(),
            },
            num_groups,
            eps: 1e-5,
        })
    }
}

impl Module for GroupNorm {
    fn name(&self) -> &str {
        "groupnorm"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        group_norm(
            input,
            self.num_groups,
            &self.weight.variable,
            &self.bias.variable,
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
    fn test_groupnorm_forward() {
        let gn = GroupNorm::on_device(4, 16, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 16, 4, 4], test_opts()).unwrap(), false);
        let y = gn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 16, 4, 4]);
    }

    #[test]
    fn test_groupnorm_single_group() {
        // groups=1 is equivalent to LayerNorm over channels
        let gn = GroupNorm::on_device(1, 8, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 8, 4, 4], test_opts()).unwrap(), false);
        let y = gn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 8, 4, 4]);
    }

    #[test]
    fn test_groupnorm_groups_equal_channels() {
        // groups=channels is equivalent to InstanceNorm
        let gn = GroupNorm::on_device(4, 4, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[2, 4, 8, 8], test_opts()).unwrap(), false);
        let y = gn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 4, 8, 8]);
    }

    #[test]
    fn test_groupnorm_gradient() {
        let gn = GroupNorm::on_device(2, 8, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[4, 8, 3, 3], test_opts()).unwrap(), true);
        let y = gn.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
        assert!(gn.weight.variable.grad().is_some());
    }

    #[test]
    fn test_groupnorm_parameters() {
        let gn = GroupNorm::on_device(4, 16, test_device()).unwrap();
        assert_eq!(gn.parameters().len(), 2);
    }

    #[test]
    fn test_groupnorm_batch_size_one() {
        // GroupNorm should work with batch=1 (unlike BatchNorm)
        let gn = GroupNorm::on_device(2, 4, test_device()).unwrap();
        let x = Variable::new(Tensor::randn(&[1, 4, 4, 4], test_opts()).unwrap(), false);
        let y = gn.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 4, 4]);
    }
}
