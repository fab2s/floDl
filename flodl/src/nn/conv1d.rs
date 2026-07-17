use crate::autograd::{Variable, conv1d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// 1D convolution layer.
///
/// Weight shape: `[out_channels, in_channels / groups, kernel_size]`.
/// Bias shape: `[out_channels]` (optional).
///
/// Input: `[batch, in_channels, length]`.
/// Output: `[batch, out_channels, L_out]` where
/// `L_out = (L + 2*padding - dilation*(kernel-1) - 1) / stride + 1`.
///
/// ```ignore
/// let conv = Conv1d::new(3, 16, 3)?; // 3->16 channels, kernel=3
/// let x = Variable::new(Tensor::randn(&[1, 3, 100], opts)?, false);
/// let y = conv.forward(&x)?; // [1, 16, 98]
/// ```
pub struct Conv1d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
}

/// Builder for configuring Conv1d layers with a fluent API.
///
/// ```ignore
/// let conv = Conv1d::configure(3, 16, 3)
///     .with_stride(2)
///     .with_padding(1)
///     .on_device(Device::CUDA(0))
///     .done()?;
/// ```
pub struct Conv1dBuilder {
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    with_bias: bool,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
    device: Device,
}

impl Conv1dBuilder {
    /// Set the convolution stride (default: 1). Controls output spatial size.
    pub fn with_stride(mut self, stride: i64) -> Self {
        self.stride = stride;
        self
    }

    /// Set zero-padding added to input (default: 0).
    pub fn with_padding(mut self, padding: i64) -> Self {
        self.padding = padding;
        self
    }

    /// Set kernel dilation (default: 1). Increases receptive field without adding parameters.
    pub fn with_dilation(mut self, dilation: i64) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set grouped convolution (default: 1). Groups=in_channels gives depthwise convolution.
    pub fn with_groups(mut self, groups: i64) -> Self {
        self.groups = groups;
        self
    }

    /// Disable the bias term.
    pub fn without_bias(mut self) -> Self {
        self.with_bias = false;
        self
    }

    /// Set the target device (default: CPU).
    pub fn on_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Build the convolution layer with the configured parameters.
    pub fn done(self) -> Result<Conv1d> {
        Conv1d::build(
            self.in_channels, self.out_channels, self.kernel_size,
            self.with_bias, self.stride, self.padding, self.dilation,
            self.groups, self.device,
        )
    }
}

impl Conv1d {
    /// Create a Conv1d layer with default stride=1, padding=0, dilation=1, groups=1, with bias.
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, 1, 0, 1, 1, Device::CPU)
    }

    /// Create a Conv1d layer without bias.
    pub fn no_bias(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, false, 1, 0, 1, 1, Device::CPU)
    }

    /// Create a Conv1d layer on a specific device.
    pub fn on_device(
        in_channels: i64, out_channels: i64, kernel_size: i64, device: Device,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, 1, 0, 1, 1, device)
    }

    /// Start a fluent builder for full configuration.
    ///
    /// ```ignore
    /// let conv = Conv1d::configure(3, 16, 5)
    ///     .with_stride(2)
    ///     .with_padding(2)
    ///     .done()?;
    /// ```
    pub fn configure(in_channels: i64, out_channels: i64, kernel_size: i64) -> Conv1dBuilder {
        Conv1dBuilder {
            in_channels,
            out_channels,
            kernel_size,
            with_bias: true,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            device: Device::CPU,
        }
    }

    /// Fully configurable Conv1d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: i64,
        with_bias: bool,
        stride: i64, padding: i64, dilation: i64,
        groups: i64, device: Device,
    ) -> Result<Self> {
        super::init::validate_conv_groups("Conv1d", in_channels, out_channels, groups)?;
        let shape = [out_channels, in_channels / groups, kernel_size];
        let fan_in = (in_channels / groups) * kernel_size;

        let weight_data = kaiming_uniform(&shape, fan_in, 5.0_f64.sqrt(), device)?;
        let weight = Variable::new(weight_data, true);

        let bias = if with_bias {
            let bias_data = uniform_bias(fan_in, &[out_channels], device)?;
            Some(Parameter {
                variable: Variable::new(bias_data, true),
                name: "bias".into(),
            })
        } else {
            None
        };

        Ok(Conv1d {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }
}

impl Module for Conv1d {
    fn name(&self) -> &str { "conv1d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv1d(
            input,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
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
    fn test_conv1d_forward() {
        let conv = Conv1d::on_device(3, 16, 3, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 3, 100], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 16, 98]); // (100 - 3 + 1) = 98
    }

    #[test]
    fn test_conv1d_no_bias() {
        let conv = Conv1d::configure(3, 8, 5)
            .without_bias()
            .on_device(test_device())
            .done().unwrap();
        assert_eq!(conv.parameters().len(), 1); // weight only
        let x = Variable::new(
            Tensor::randn(&[2, 3, 50], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 8, 46]);
    }

    #[test]
    fn test_conv1d_with_stride_padding() {
        let conv = Conv1d::configure(1, 4, 3)
            .with_stride(2)
            .with_padding(1)
            .on_device(test_device())
            .done().unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 1, 20], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        // (20 + 2*1 - 3) / 2 + 1 = 10
        assert_eq!(y.shape(), vec![1, 4, 10]);
    }

    #[test]
    fn test_conv1d_gradient() {
        let conv = Conv1d::on_device(2, 4, 3, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 2, 10], test_opts()).unwrap(), true,
        );
        let y = conv.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
        assert!(conv.weight.variable.grad().is_some());
    }

    #[test]
    fn test_conv1d_builder_groups() {
        let conv = Conv1d::configure(4, 8, 3)
            .with_groups(2)
            .on_device(test_device())
            .done().unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 4, 20], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 8, 18]);
        // Weight shape: [8, 4/2, 3] = [8, 2, 3]
        assert_eq!(conv.weight.variable.shape(), vec![8, 2, 3]);
    }
}
