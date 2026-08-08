use crate::autograd::{Variable, conv_transpose2d};
use crate::tensor::{Device, Result};

use super::Module;
use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;

/// Transposed 2D convolution (deconvolution) layer.
///
/// Weight shape: `[in_channels, out_channels / groups, kernel_h, kernel_w]`.
/// Bias shape: `[out_channels]` (optional).
pub struct ConvTranspose2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
    pub output_padding: [i64; 2],
    pub dilation: [i64; 2],
    pub groups: i64,
}

/// Builder for configuring ConvTranspose2d layers with a fluent API.
///
/// ```ignore
/// let conv = ConvTranspose2d::configure(8, 3, 3)
///     .with_stride(2)
///     .with_padding(1)
///     .on_device(Device::CUDA(0))
///     .done()?;
/// ```
pub struct ConvTranspose2dBuilder {
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    with_bias: bool,
    stride: [i64; 2],
    padding: [i64; 2],
    output_padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
    device: Device,
}

impl ConvTranspose2dBuilder {
    /// Set the convolution stride (default: 1). Applied to both H and W dimensions.
    pub fn with_stride(mut self, stride: i64) -> Self {
        self.stride = [stride, stride];
        self
    }

    /// Set zero-padding applied to input (default: 0). Applied to both H and W dimensions.
    pub fn with_padding(mut self, padding: i64) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Set output padding to resolve ambiguous output sizes (default: 0). Applied to both dimensions.
    pub fn with_output_padding(mut self, output_padding: i64) -> Self {
        self.output_padding = [output_padding, output_padding];
        self
    }

    /// Set kernel dilation (default: 1). Increases receptive field without adding parameters.
    pub fn with_dilation(mut self, dilation: i64) -> Self {
        self.dilation = [dilation, dilation];
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
    pub fn done(self) -> Result<ConvTranspose2d> {
        ConvTranspose2d::build(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.with_bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
            self.device,
        )
    }
}

impl ConvTranspose2d {
    /// Create a ConvTranspose2d layer with default settings and bias.
    pub fn new(in_channels: i64, out_channels: i64, kernel_size: i64) -> Result<Self> {
        Self::build(
            in_channels,
            out_channels,
            kernel_size,
            true,
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            1,
            Device::CPU,
        )
    }

    /// Create a ConvTranspose2d layer on a specific device.
    pub fn on_device(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        device: Device,
    ) -> Result<Self> {
        Self::build(
            in_channels,
            out_channels,
            kernel_size,
            true,
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            1,
            device,
        )
    }

    /// Start a fluent builder for full configuration.
    ///
    /// ```ignore
    /// let conv = ConvTranspose2d::configure(8, 3, 3)
    ///     .with_stride(2)
    ///     .with_padding(1)
    ///     .done()?;
    /// ```
    pub fn configure(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
    ) -> ConvTranspose2dBuilder {
        ConvTranspose2dBuilder {
            in_channels,
            out_channels,
            kernel_size,
            with_bias: true,
            stride: [1, 1],
            padding: [0, 0],
            output_padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            device: Device::CPU,
        }
    }

    /// Fully configurable ConvTranspose2d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        with_bias: bool,
        stride: [i64; 2],
        padding: [i64; 2],
        output_padding: [i64; 2],
        dilation: [i64; 2],
        groups: i64,
        device: Device,
    ) -> Result<Self> {
        super::init::validate_conv_groups("ConvTranspose2d", in_channels, out_channels, groups)?;
        // Note: weight shape is [in_channels, out_channels/groups, kH, kW]
        let shape = [in_channels, out_channels / groups, kernel_size, kernel_size];
        let fan_in = in_channels * kernel_size * kernel_size;

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

        Ok(ConvTranspose2d {
            weight: Parameter {
                variable: weight,
                name: "weight".into(),
            },
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        })
    }
}

impl Module for ConvTranspose2d {
    fn name(&self) -> &str {
        "conv_t2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv_transpose2d(
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
    fn test_conv_transpose2d_forward() {
        let conv = ConvTranspose2d::build(
            8,
            3,
            3,
            true,
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            1,
            test_device(),
        )
        .unwrap();
        let x = Variable::new(Tensor::randn(&[1, 8, 4, 4], test_opts()).unwrap(), false);
        let y = conv.forward(&x).unwrap();
        // (4-1)*1 - 2*0 + 3 = 6
        assert_eq!(y.shape(), vec![1, 3, 6, 6]);
    }

    #[test]
    fn test_conv_transpose2d_gradient() {
        let conv = ConvTranspose2d::build(
            4,
            2,
            3,
            true,
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            1,
            test_device(),
        )
        .unwrap();
        let x = Variable::new(Tensor::randn(&[1, 4, 4, 4], test_opts()).unwrap(), true);
        let y = conv.forward(&x).unwrap().sum().unwrap();
        y.backward().unwrap();
        assert!(x.grad().is_some());
        assert!(conv.weight.variable.grad().is_some());
    }

    #[test]
    fn test_conv_transpose2d_with_stride() {
        let conv = ConvTranspose2d::build(
            4,
            2,
            3,
            true,
            [2, 2],
            [1, 1],
            [0, 0],
            [1, 1],
            1,
            test_device(),
        )
        .unwrap();
        let x = Variable::new(Tensor::randn(&[1, 4, 4, 4], test_opts()).unwrap(), false);
        let y = conv.forward(&x).unwrap();
        // (4-1)*2 - 2*1 + 3 = 7
        assert_eq!(y.shape(), vec![1, 2, 7, 7]);
    }

    #[test]
    fn test_conv_transpose2d_no_bias() {
        let conv = ConvTranspose2d::build(
            4,
            2,
            3,
            false,
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            1,
            test_device(),
        )
        .unwrap();
        assert_eq!(conv.parameters().len(), 1);
        assert!(conv.bias.is_none());
    }
}
