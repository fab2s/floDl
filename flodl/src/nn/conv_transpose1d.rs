use crate::autograd::{Variable, conv_transpose1d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// Transposed 1D convolution (deconvolution) layer.
///
/// Weight shape: `[in_channels, out_channels / groups, kernel_size]`.
/// Bias shape: `[out_channels]` (optional).
pub struct ConvTranspose1d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: i64,
    pub padding: i64,
    pub output_padding: i64,
    pub dilation: i64,
    pub groups: i64,
}

/// Builder for configuring ConvTranspose1d layers with a fluent API.
///
/// ```ignore
/// let conv = ConvTranspose1d::configure(4, 2, 3)
///     .with_stride(2)
///     .with_padding(1)
///     .on_device(Device::CUDA(0))
///     .done()?;
/// ```
pub struct ConvTranspose1dBuilder {
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    with_bias: bool,
    stride: i64,
    padding: i64,
    output_padding: i64,
    dilation: i64,
    groups: i64,
    device: Device,
}

impl ConvTranspose1dBuilder {
    /// Set the convolution stride (default: 1). Controls output spatial size.
    pub fn with_stride(mut self, stride: i64) -> Self {
        self.stride = stride;
        self
    }

    /// Set zero-padding applied to input (default: 0).
    pub fn with_padding(mut self, padding: i64) -> Self {
        self.padding = padding;
        self
    }

    /// Set output padding to resolve ambiguous output sizes (default: 0).
    pub fn with_output_padding(mut self, output_padding: i64) -> Self {
        self.output_padding = output_padding;
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

    /// Build the transposed convolution layer with the configured parameters.
    pub fn done(self) -> Result<ConvTranspose1d> {
        ConvTranspose1d::build(
            self.in_channels, self.out_channels, self.kernel_size,
            self.with_bias, self.stride, self.padding, self.output_padding,
            self.dilation, self.groups, self.device,
        )
    }
}

impl ConvTranspose1d {
    /// Create a ConvTranspose1d layer with default settings and bias.
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, 1, 0, 0, 1, 1, Device::CPU)
    }

    /// Create a ConvTranspose1d layer on a specific device.
    pub fn on_device(
        in_channels: i64, out_channels: i64, kernel_size: i64, device: Device,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, 1, 0, 0, 1, 1, device)
    }

    /// Start a fluent builder for full configuration.
    ///
    /// ```ignore
    /// let conv = ConvTranspose1d::configure(4, 2, 3)
    ///     .with_stride(2)
    ///     .with_output_padding(1)
    ///     .done()?;
    /// ```
    pub fn configure(in_channels: i64, out_channels: i64, kernel_size: i64) -> ConvTranspose1dBuilder {
        ConvTranspose1dBuilder {
            in_channels,
            out_channels,
            kernel_size,
            with_bias: true,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
            device: Device::CPU,
        }
    }

    /// Fully configurable ConvTranspose1d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: i64,
        with_bias: bool,
        stride: i64, padding: i64, output_padding: i64,
        dilation: i64, groups: i64, device: Device,
    ) -> Result<Self> {
        super::init::validate_conv_groups("ConvTranspose1d", in_channels, out_channels, groups)?;
        // Note: weight shape is [in_channels, out_channels/groups, K]
        let shape = [in_channels, out_channels / groups, kernel_size];
        let fan_in = in_channels * kernel_size;

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

        Ok(ConvTranspose1d {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        })
    }
}

impl Module for ConvTranspose1d {
    fn name(&self) -> &str { "conv_t1d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv_transpose1d(
            input,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
            self.stride,
            self.padding,
            self.output_padding,
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
    fn test_conv_transpose1d_forward() {
        let conv = ConvTranspose1d::on_device(4, 2, 3, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 4, 10], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        // output_size = (10-1)*1 - 2*0 + 3 = 12
        assert_eq!(y.shape(), vec![1, 2, 12]);
    }

    #[test]
    fn test_conv_transpose1d_gradient() {
        let conv = ConvTranspose1d::on_device(4, 2, 3, test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 4, 8], test_opts()).unwrap(), true,
        );
        let y = conv.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
        assert!(conv.weight.variable.grad().is_some());
    }

    #[test]
    fn test_conv_transpose1d_parameters() {
        let conv = ConvTranspose1d::new(4, 2, 3).unwrap();
        assert_eq!(conv.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_conv_transpose1d_with_stride() {
        let conv = ConvTranspose1d::build(
            4, 2, 3, true, 2, 0, 0, 1, 1, test_device(),
        ).unwrap();
        let x = Variable::new(
            Tensor::randn(&[1, 4, 5], test_opts()).unwrap(), false,
        );
        let y = conv.forward(&x).unwrap();
        // output_size = (5-1)*2 - 2*0 + 3 = 11
        assert_eq!(y.shape(), vec![1, 2, 11]);
    }
}
