use crate::autograd::{Variable, conv3d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// 3D convolution layer.
///
/// Weight shape: `[out_channels, in_channels / groups, kD, kH, kW]`.
/// Bias shape: `[out_channels]` (optional).
///
/// Input: `[batch, in_channels, D, H, W]`.
/// Output: `[batch, out_channels, D_out, H_out, W_out]`.
pub struct Conv3d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: [i64; 3],
    pub padding: [i64; 3],
    pub dilation: [i64; 3],
    pub groups: i64,
}

/// Builder for configuring Conv3d layers with a fluent API.
pub struct Conv3dBuilder {
    in_channels: i64,
    out_channels: i64,
    kernel_size: [i64; 3],
    with_bias: bool,
    stride: [i64; 3],
    padding: [i64; 3],
    dilation: [i64; 3],
    groups: i64,
    device: Device,
}

impl Conv3dBuilder {
    /// Set the convolution stride (default: [1, 1, 1]). Controls output spatial size per dimension.
    pub fn with_stride(mut self, stride: [i64; 3]) -> Self { self.stride = stride; self }

    /// Set zero-padding added to input (default: [0, 0, 0]). One value per spatial dimension.
    pub fn with_padding(mut self, padding: [i64; 3]) -> Self { self.padding = padding; self }

    /// Set kernel dilation (default: [1, 1, 1]). Increases receptive field without adding parameters.
    pub fn with_dilation(mut self, dilation: [i64; 3]) -> Self { self.dilation = dilation; self }

    /// Set grouped convolution (default: 1). Groups=in_channels gives depthwise convolution.
    pub fn with_groups(mut self, groups: i64) -> Self { self.groups = groups; self }

    /// Disable the bias term.
    pub fn without_bias(mut self) -> Self { self.with_bias = false; self }

    /// Set the target device (default: CPU).
    pub fn on_device(mut self, device: Device) -> Self { self.device = device; self }

    /// Build the convolution layer with the configured parameters.
    pub fn done(self) -> Result<Conv3d> {
        Conv3d::build(
            self.in_channels, self.out_channels, self.kernel_size,
            self.with_bias, self.stride, self.padding, self.dilation,
            self.groups, self.device,
        )
    }
}

impl Conv3d {
    /// Create a Conv3d layer with default stride=1, padding=0, dilation=1, groups=1, with bias.
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: [i64; 3],
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true,
                    [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, Device::CPU)
    }

    /// Create a Conv3d layer on a specific device.
    pub fn on_device(
        in_channels: i64, out_channels: i64, kernel_size: [i64; 3], device: Device,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true,
                    [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, device)
    }

    /// Start a fluent builder for full configuration.
    ///
    /// ```ignore
    /// let conv = Conv3d::configure(1, 4, [3, 3, 3])
    ///     .with_padding([1, 1, 1])
    ///     .done()?;
    /// ```
    pub fn configure(in_channels: i64, out_channels: i64, kernel_size: [i64; 3]) -> Conv3dBuilder {
        Conv3dBuilder {
            in_channels, out_channels, kernel_size,
            with_bias: true,
            stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1],
            groups: 1, device: Device::CPU,
        }
    }

    /// Fully configurable Conv3d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: [i64; 3],
        with_bias: bool,
        stride: [i64; 3], padding: [i64; 3], dilation: [i64; 3],
        groups: i64, device: Device,
    ) -> Result<Self> {
        super::init::validate_conv_groups("Conv3d", in_channels, out_channels, groups)?;
        let shape = [out_channels, in_channels / groups,
                     kernel_size[0], kernel_size[1], kernel_size[2]];
        let fan_in = (in_channels / groups) * kernel_size[0] * kernel_size[1] * kernel_size[2];

        let weight_data = kaiming_uniform(&shape, fan_in, 5.0_f64.sqrt(), device)?;

        let bias = if with_bias {
            let bias_data = uniform_bias(fan_in, &[out_channels], device)?;
            Some(Parameter::new(bias_data, "bias"))
        } else {
            None
        };

        Ok(Conv3d {
            weight: Parameter::new(weight_data, "weight"),
            bias, stride, padding, dilation, groups,
        })
    }
}

impl Module for Conv3d {
    fn name(&self) -> &str { "conv3d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv3d(
            input, &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
            self.stride, self.padding, self.dilation, self.groups,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
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
    fn test_conv3d_forward() {
        let conv = Conv3d::on_device(1, 4, [3, 3, 3], test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8, 8, 8], test_opts()).unwrap(),
            false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 6, 6, 6]);
    }

    #[test]
    fn test_conv3d_gradient() {
        let conv = Conv3d::on_device(1, 2, [3, 3, 3], test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 6, 6, 6], test_opts()).unwrap(),
            true,
        );
        let y = conv.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_conv3d_no_bias() {
        let conv = Conv3d::configure(1, 4, [3, 3, 3])
            .without_bias()
            .on_device(test_device())
            .done().unwrap();
        assert_eq!(conv.parameters().len(), 1);
        assert!(conv.bias.is_none());
        let x = Variable::new(
            Tensor::randn(&[1, 1, 6, 6, 6], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 4, 4, 4]);
    }

    #[test]
    fn test_conv3d_with_padding() {
        let conv = Conv3d::configure(1, 2, [3, 3, 3])
            .with_padding([1, 1, 1])
            .on_device(test_device())
            .done().unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4, 4], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        // Same padding: output = input size
        assert_eq!(y.shape(), vec![1, 2, 4, 4, 4]);
    }
}
