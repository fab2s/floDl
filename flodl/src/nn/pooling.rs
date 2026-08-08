//! Pooling layers.

use crate::autograd::{self, Variable};
use crate::tensor::Result;

use super::Module;
use super::parameter::Parameter;

/// 2D max pooling layer.
///
/// Applies max pooling over a 4D input tensor, selecting the maximum value
/// in each pooling window. Equivalent to PyTorch's `nn.MaxPool2d`.
///
/// - **Input shape:** `[N, C, H_in, W_in]`
/// - **Output shape:** `[N, C, H_out, W_out]`
///
/// Output size formula:
///
/// ```text
/// H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
/// ```
///
/// # Example
///
/// ```ignore
/// let pool = MaxPool2d::new(2);              // kernel_size=2, stride=2
/// let pool = MaxPool2d::with_stride(3, 2)    // kernel_size=3, stride=2
///     .padding(1)
///     .dilation(1);
/// let y = pool.forward(&x)?;
/// ```
pub struct MaxPool2d {
    kernel_size: [i64; 2],
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    ceil_mode: bool,
}

impl MaxPool2d {
    /// Create a MaxPool2d with the given square kernel size.
    ///
    /// Stride defaults to `kernel_size` (non-overlapping windows).
    /// Padding defaults to 0, dilation to 1.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [kernel_size, kernel_size],
            padding: [0, 0],
            dilation: [1, 1],
            ceil_mode: false,
        }
    }

    /// Create with explicit kernel size and stride.
    ///
    /// Use this when the pooling stride differs from the kernel size
    /// (e.g., overlapping windows with `stride < kernel_size`).
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [stride, stride],
            padding: [0, 0],
            dilation: [1, 1],
            ceil_mode: false,
        }
    }

    /// Set symmetric padding added to both sides of the input.
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Set dilation (spacing between kernel elements). Default is 1.
    pub fn dilation(mut self, dilation: i64) -> Self {
        self.dilation = [dilation, dilation];
        self
    }

    /// Use ceiling (instead of floor) when computing the output size.
    ///
    /// When true, the output may include a partial pooling window at the edge.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.ceil_mode = ceil_mode;
        self
    }
}

impl Module for MaxPool2d {
    fn name(&self) -> &str {
        "maxpool2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![] // pooling has no learnable parameters
    }
}

/// 2D average pooling layer.
///
/// Computes the mean of each pooling window over a 4D input tensor.
/// Equivalent to PyTorch's `nn.AvgPool2d`.
///
/// - **Input shape:** `[N, C, H_in, W_in]`
/// - **Output shape:** `[N, C, H_out, W_out]`
///
/// Output size formula (same as [`MaxPool2d`] with dilation=1):
///
/// ```text
/// H_out = floor((H_in + 2*padding - kernel_size) / stride + 1)
/// ```
///
/// # Example
///
/// ```ignore
/// let pool = AvgPool2d::new(2);              // kernel_size=2, stride=2
/// let pool = AvgPool2d::with_stride(3, 2)    // kernel_size=3, stride=2
///     .padding(1)
///     .count_include_pad(false);
/// let y = pool.forward(&x)?;
/// ```
pub struct AvgPool2d {
    kernel_size: [i64; 2],
    stride: [i64; 2],
    padding: [i64; 2],
    ceil_mode: bool,
    count_include_pad: bool,
}

impl AvgPool2d {
    /// Create an AvgPool2d with the given square kernel size.
    ///
    /// Stride defaults to `kernel_size` (non-overlapping windows).
    /// Padding defaults to 0, `count_include_pad` defaults to true.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [kernel_size, kernel_size],
            padding: [0, 0],
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Create with explicit kernel size and stride.
    ///
    /// Use this when the pooling stride differs from the kernel size.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            stride: [stride, stride],
            padding: [0, 0],
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Set symmetric padding added to both sides of the input.
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Use ceiling (instead of floor) when computing the output size.
    pub fn ceil_mode(mut self, ceil_mode: bool) -> Self {
        self.ceil_mode = ceil_mode;
        self
    }

    /// Whether to include zero-padding positions in the average calculation.
    ///
    /// When false, the average is computed only over non-padded elements,
    /// which can produce more accurate values at the borders. Default is true.
    pub fn count_include_pad(mut self, count_include_pad: bool) -> Self {
        self.count_include_pad = count_include_pad;
        self
    }
}

impl Module for AvgPool2d {
    fn name(&self) -> &str {
        "avgpool2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// 1D max pooling layer.
///
/// Applies max pooling over a 3D input tensor, selecting the maximum value
/// in each sliding window along the temporal/sequence dimension.
/// Equivalent to PyTorch's `nn.MaxPool1d`.
///
/// - **Input shape:** `[N, C, L_in]`
/// - **Output shape:** `[N, C, L_out]`
///
/// Output size formula:
///
/// ```text
/// L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
/// ```
///
/// # Example
///
/// ```ignore
/// let pool = MaxPool1d::new(2);            // kernel_size=2, stride=2
/// let pool = MaxPool1d::with_stride(3, 1)  // kernel_size=3, stride=1
///     .padding(1);
/// let y = pool.forward(&x)?;
/// ```
pub struct MaxPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
    ceil_mode: bool,
}

impl MaxPool1d {
    /// Create a MaxPool1d with the given kernel size.
    ///
    /// Stride defaults to `kernel_size` (non-overlapping windows).
    /// Padding defaults to 0, dilation to 1.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        }
    }

    /// Create with explicit kernel size and stride.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            dilation: 1,
            ceil_mode: false,
        }
    }

    /// Set padding added to both sides of the input sequence.
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for MaxPool1d {
    fn name(&self) -> &str {
        "maxpool1d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// 1D average pooling layer.
///
/// Computes the mean of each sliding window along the temporal/sequence
/// dimension. Equivalent to PyTorch's `nn.AvgPool1d`.
///
/// - **Input shape:** `[N, C, L_in]`
/// - **Output shape:** `[N, C, L_out]`
///
/// Output size formula:
///
/// ```text
/// L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
/// ```
///
/// # Example
///
/// ```ignore
/// let pool = AvgPool1d::new(3);            // kernel_size=3, stride=3
/// let pool = AvgPool1d::with_stride(3, 1)  // kernel_size=3, stride=1
///     .padding(1)
///     .count_include_pad(false);
/// let y = pool.forward(&x)?;
/// ```
pub struct AvgPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool,
}

impl AvgPool1d {
    /// Create an AvgPool1d with the given kernel size.
    ///
    /// Stride defaults to `kernel_size` (non-overlapping windows).
    /// Padding defaults to 0, `count_include_pad` defaults to true.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Create with explicit kernel size and stride.
    pub fn with_stride(kernel_size: i64, stride: i64) -> Self {
        Self {
            kernel_size,
            stride,
            padding: 0,
            ceil_mode: false,
            count_include_pad: true,
        }
    }

    /// Set padding added to both sides of the input sequence.
    pub fn padding(mut self, padding: i64) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for AvgPool1d {
    fn name(&self) -> &str {
        "avgpool1d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// 2D adaptive max pooling layer.
///
/// Automatically selects kernel size and stride to produce a fixed output
/// spatial size regardless of input dimensions. Equivalent to PyTorch's
/// `nn.AdaptiveMaxPool2d`.
///
/// This is commonly used before fully-connected layers to accept
/// variable-size inputs.
///
/// - **Input shape:** `[N, C, H_in, W_in]` (any spatial size)
/// - **Output shape:** `[N, C, output_h, output_w]` (fixed)
///
/// # Example
///
/// ```ignore
/// let pool = AdaptiveMaxPool2d::new(1, 1);  // global max pooling
/// let pool = AdaptiveMaxPool2d::new(7, 7);  // common before FC layers
/// let y = pool.forward(&x)?;
/// ```
pub struct AdaptiveMaxPool2d {
    output_size: [i64; 2],
}

impl AdaptiveMaxPool2d {
    /// Create with the desired output spatial dimensions `(output_h, output_w)`.
    ///
    /// The layer will internally compute kernel size, stride, and padding
    /// to produce exactly `[N, C, output_h, output_w]` for any input size.
    pub fn new(output_h: i64, output_w: i64) -> Self {
        Self {
            output_size: [output_h, output_w],
        }
    }
}

impl Module for AdaptiveMaxPool2d {
    fn name(&self) -> &str {
        "adaptive_maxpool2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::adaptive_max_pool2d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Adaptive average pooling that outputs a fixed spatial size regardless of input.
///
/// Wraps [`crate::autograd::adaptive_avg_pool2d`].
/// Produces exactly `[N, C, output_h, output_w]` from any spatial input, enabling
/// variable-size inputs.
///
/// - **Input shape:** `[N, C, H_in, W_in]` (any spatial size)
/// - **Output shape:** `[N, C, output_h, output_w]` (fixed)
///
/// # Example
///
/// ```ignore
/// let pool = AdaptiveAvgPool2d::new([1, 1]);  // global average pooling
/// let pool = AdaptiveAvgPool2d::new([7, 7]);  // common before FC layers
/// let y = pool.forward(&x)?;
/// ```
pub struct AdaptiveAvgPool2d {
    output_size: [i64; 2],
}

impl AdaptiveAvgPool2d {
    /// Create with the desired output spatial dimensions `[output_h, output_w]`.
    pub fn new(output_size: [i64; 2]) -> Self {
        Self { output_size }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn name(&self) -> &str {
        "adaptive_avgpool2d"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::adaptive_avg_pool2d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Sub-pixel convolution upsampling (pixel shuffle).
///
/// Rearranges elements from the channel dimension into spatial dimensions,
/// increasing resolution by `upscale_factor` in both height and width.
/// Equivalent to PyTorch's `nn.PixelShuffle`. See also [`PixelUnshuffle`]
/// for the inverse operation.
///
/// The input must have at least `C * r^2` channels, where `r` is the
/// upscale factor.
///
/// - **Input shape:** `[N, C * r^2, H, W]`
/// - **Output shape:** `[N, C, H * r, W * r]`
///
/// # Example
///
/// ```ignore
/// // Upscale 2x: 4 channels -> 1 channel, spatial dims doubled
/// let ps = PixelShuffle::new(2);
/// let x = Variable::new(Tensor::randn(&[1, 4, 8, 8], opts)?, false);
/// let y = ps.forward(&x)?;  // [1, 1, 16, 16]
/// ```
pub struct PixelShuffle {
    upscale_factor: i64,
}

impl PixelShuffle {
    /// Create a PixelShuffle with the given upscale factor `r`.
    ///
    /// The input channel count must be divisible by `r^2`.
    pub fn new(upscale_factor: i64) -> Self {
        Self { upscale_factor }
    }
}

impl Module for PixelShuffle {
    fn name(&self) -> &str {
        "pixel_shuffle"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::pixel_shuffle(input, self.upscale_factor)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Inverse of [`PixelShuffle`]: packs spatial elements back into channels.
///
/// Rearranges elements from spatial dimensions into the channel dimension,
/// reducing resolution by `downscale_factor` in both height and width.
/// Equivalent to PyTorch's `nn.PixelUnshuffle`.
///
/// The input spatial dimensions must both be divisible by `r` (the
/// downscale factor).
///
/// - **Input shape:** `[N, C, H * r, W * r]`
/// - **Output shape:** `[N, C * r^2, H, W]`
///
/// # Example
///
/// ```ignore
/// // Downscale 2x: 1 channel -> 4 channels, spatial dims halved
/// let pu = PixelUnshuffle::new(2);
/// let x = Variable::new(Tensor::randn(&[1, 1, 16, 16], opts)?, false);
/// let y = pu.forward(&x)?;  // [1, 4, 8, 8]
/// ```
pub struct PixelUnshuffle {
    downscale_factor: i64,
}

impl PixelUnshuffle {
    /// Create a PixelUnshuffle with the given downscale factor `r`.
    ///
    /// Input height and width must each be divisible by `r`.
    pub fn new(downscale_factor: i64) -> Self {
        Self { downscale_factor }
    }
}

impl Module for PixelUnshuffle {
    fn name(&self) -> &str {
        "pixel_unshuffle"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::pixel_unshuffle(input, self.downscale_factor)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Spatial upsampling via interpolation.
///
/// Wraps `F.interpolate` as an `nn.Module`, resizing the spatial dimensions
/// of the input to a fixed target size. Equivalent to PyTorch's `nn.Upsample`.
///
/// Supported interpolation modes:
/// - `0` -- nearest neighbor
/// - `1` -- bilinear (4D input)
/// - `2` -- bicubic (4D input)
/// - `3` -- trilinear (5D input)
///
/// - **Input shape:** `[N, C, ...]` (3D, 4D, or 5D depending on mode)
/// - **Output shape:** `[N, C, *output_size]`
///
/// # Example
///
/// ```ignore
/// let up = Upsample::new(&[64, 64], 1);  // bilinear to 64x64
/// let up = Upsample::new(&[128, 128], 0).align_corners(false);  // nearest
/// let y = up.forward(&x)?;
/// ```
pub struct Upsample {
    output_size: Vec<i64>,
    mode: i32,
    align_corners: bool,
}

impl Upsample {
    /// Create an Upsample module targeting the given spatial output size.
    ///
    /// - `output_size` -- desired spatial dimensions (e.g., `&[H, W]` for 4D).
    /// - `mode` -- interpolation mode: 0=nearest, 1=bilinear, 2=bicubic, 3=trilinear.
    ///
    /// `align_corners` defaults to false.
    pub fn new(output_size: &[i64], mode: i32) -> Self {
        Self {
            output_size: output_size.to_vec(),
            mode,
            align_corners: false,
        }
    }

    /// Set the `align_corners` flag.
    ///
    /// When true, the corner pixels of input and output tensors are aligned,
    /// preserving the values at those pixels. Only meaningful for bilinear,
    /// bicubic, and trilinear modes (ignored for nearest).
    pub fn align_corners(mut self, align: bool) -> Self {
        self.align_corners = align;
        self
    }
}

impl Module for Upsample {
    fn name(&self) -> &str {
        "upsample"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let result = input
            .data()
            .interpolate(&self.output_size, self.mode, self.align_corners)?;
        Ok(Variable::wrap(result))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Extracts sliding local blocks from a batched 2D input (im2col).
///
/// Unfolds the input into a column matrix where each column contains the
/// flattened elements of one sliding window. Equivalent to PyTorch's
/// `nn.Unfold`. See also [`Fold`] for the inverse operation (col2im).
///
/// The number of output blocks `L` is:
///
/// ```text
/// L = product over d in {H, W} of:
///     floor((input_d + 2*padding[d] - dilation[d]*(kernel_size[d]-1) - 1) / stride[d] + 1)
/// ```
///
/// - **Input shape:** `[N, C, H, W]`
/// - **Output shape:** `[N, C * kernel_h * kernel_w, L]`
///
/// # Parameters
///
/// - `kernel_size` -- size of the sliding window
/// - `stride` -- step between consecutive windows (default 1)
/// - `padding` -- zero-padding on both sides (default 0)
/// - `dilation` -- spacing between kernel elements (default 1)
///
/// # Example
///
/// ```ignore
/// let unfold = Unfold::new(3).stride([2, 2]).padding([1, 1]);
/// let cols = unfold.forward(&x)?;  // [N, C*9, L]
/// ```
pub struct Unfold {
    kernel_size: [i64; 2],
    dilation: [i64; 2],
    padding: [i64; 2],
    stride: [i64; 2],
}

impl Unfold {
    /// Create with a square kernel of size `kernel_size x kernel_size`.
    ///
    /// Stride defaults to 1, padding to 0, dilation to 1.
    pub fn new(kernel_size: i64) -> Self {
        Self {
            kernel_size: [kernel_size, kernel_size],
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Create with a rectangular kernel `[kernel_h, kernel_w]`.
    ///
    /// Stride defaults to 1, padding to 0, dilation to 1.
    pub fn with_kernel(kernel_size: [i64; 2]) -> Self {
        Self {
            kernel_size,
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Set dilation (spacing between kernel elements) as `[dH, dW]`.
    pub fn dilation(mut self, dilation: [i64; 2]) -> Self {
        self.dilation = dilation;
        self
    }
    /// Set zero-padding added to both sides as `[padH, padW]`.
    pub fn padding(mut self, padding: [i64; 2]) -> Self {
        self.padding = padding;
        self
    }
    /// Set stride of the sliding blocks as `[strideH, strideW]`.
    pub fn stride(mut self, stride: [i64; 2]) -> Self {
        self.stride = stride;
        self
    }
}

impl Module for Unfold {
    fn name(&self) -> &str {
        "unfold"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::im2col(
            input,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Reassembles columns into a batched 2D tensor (col2im).
///
/// Inverse of [`Unfold`]. Combines the sliding local blocks back into a
/// full spatial tensor. Equivalent to PyTorch's `nn.Fold`.
///
/// Where blocks overlap (stride < kernel_size), values are summed. For a
/// perfect roundtrip with non-overlapping blocks, use `stride == kernel_size`.
///
/// - **Input shape:** `[N, C * kernel_h * kernel_w, L]`
/// - **Output shape:** `[N, C, output_h, output_w]`
///
/// # Parameters
///
/// - `output_size` -- target spatial size `[output_h, output_w]`
/// - `kernel_size` -- size of the sliding window (must match the Unfold)
/// - `stride` -- step between consecutive windows (default 1)
/// - `padding` -- zero-padding that was applied during Unfold (default 0)
/// - `dilation` -- spacing between kernel elements (default 1)
///
/// # Example
///
/// ```ignore
/// let fold = Fold::new([32, 32], 3).stride([2, 2]).padding([1, 1]);
/// let img = fold.forward(&cols)?;  // [N, C, 32, 32]
/// ```
pub struct Fold {
    output_size: [i64; 2],
    kernel_size: [i64; 2],
    dilation: [i64; 2],
    padding: [i64; 2],
    stride: [i64; 2],
}

impl Fold {
    /// Create with target output size `[output_h, output_w]` and a square kernel.
    ///
    /// Stride defaults to 1, padding to 0, dilation to 1.
    pub fn new(output_size: [i64; 2], kernel_size: i64) -> Self {
        Self {
            output_size,
            kernel_size: [kernel_size, kernel_size],
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Create with target output size and a rectangular kernel `[kernel_h, kernel_w]`.
    ///
    /// Stride defaults to 1, padding to 0, dilation to 1.
    pub fn with_kernel(output_size: [i64; 2], kernel_size: [i64; 2]) -> Self {
        Self {
            output_size,
            kernel_size,
            dilation: [1, 1],
            padding: [0, 0],
            stride: [1, 1],
        }
    }

    /// Set dilation (spacing between kernel elements) as `[dH, dW]`.
    pub fn dilation(mut self, dilation: [i64; 2]) -> Self {
        self.dilation = dilation;
        self
    }
    /// Set zero-padding as `[padH, padW]` (must match the Unfold that produced the input).
    pub fn padding(mut self, padding: [i64; 2]) -> Self {
        self.padding = padding;
        self
    }
    /// Set stride of the sliding blocks as `[strideH, strideW]`.
    pub fn stride(mut self, stride: [i64; 2]) -> Self {
        self.stride = stride;
        self
    }
}

impl Module for Fold {
    fn name(&self) -> &str {
        "fold"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        autograd::col2im(
            input,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_max_pool2d_basic() {
        let opts = crate::tensor::test_opts();
        // [1, 1, 4, 4] → kernel=2, stride=2 → [1, 1, 2, 2]
        let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), false);
        let pool = MaxPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        let opts = crate::tensor::test_opts();
        // [2, 3, 8, 8] → kernel=3, stride=2, padding=1 → [2, 3, 4, 4]
        let x = Variable::new(Tensor::randn(&[2, 3, 8, 8], opts).unwrap(), false);
        let pool = MaxPool2d::with_stride(3, 2).padding(1);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3, 4, 4]);
    }

    #[test]
    fn test_max_pool2d_gradient() {
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[2, 1, 4, 4], opts).unwrap(), true);
        let pool = MaxPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![2, 1, 4, 4]);
        // Gradient should be sparse: only max elements get gradient
    }

    #[test]
    fn test_max_pool2d_values() {
        let device = crate::tensor::test_device();
        // Manual check: 2x2 max pool on a known 4x4 input
        let data = vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let x = Variable::new(
            Tensor::from_f32(&data, &[1, 1, 4, 4], device).unwrap(),
            false,
        );
        let pool = MaxPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let y_data = y.data().to_f32_vec().unwrap();
        // Each 2x2 block → max: [6, 8, 14, 16]
        assert_eq!(y_data, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool2d_basic() {
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[1, 1, 4, 4], opts).unwrap(), false);
        let pool = AvgPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 2, 2]);
    }

    #[test]
    fn test_avg_pool2d_values() {
        let device = crate::tensor::test_device();
        let data = vec![
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let x = Variable::new(
            Tensor::from_f32(&data, &[1, 1, 4, 4], device).unwrap(),
            false,
        );
        let pool = AvgPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let y_data = y.data().to_f32_vec().unwrap();
        // Each 2x2 block → mean: [3.5, 5.5, 11.5, 13.5]
        assert_eq!(y_data, vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_avg_pool2d_gradient() {
        let opts = crate::tensor::test_opts();
        let x = Variable::new(Tensor::randn(&[2, 1, 4, 4], opts).unwrap(), true);
        let pool = AvgPool2d::new(2);
        let y = pool.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap();
        assert_eq!(grad.shape(), vec![2, 1, 4, 4]);
    }

    #[test]
    fn test_max_pool1d() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pool = MaxPool1d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4]);
    }

    #[test]
    fn test_avg_pool1d() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pool = AvgPool1d::new(2);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4]);
    }

    #[test]
    fn test_adaptive_max_pool2d() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 8, 8], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pool = AdaptiveMaxPool2d::new(3, 3);
        let y = pool.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_pixel_shuffle() {
        let x = Variable::new(
            Tensor::randn(&[1, 4, 2, 2], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let ps = PixelShuffle::new(2);
        let y = ps.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_pixel_unshuffle() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let pu = PixelUnshuffle::new(2);
        let y = pu.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 2, 2]);
    }

    #[test]
    fn test_upsample() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 2, 2], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let up = Upsample::new(&[4, 4], 0); // nearest
        let y = up.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_unfold() {
        // [1, 1, 4, 4] with kernel=2, stride=1 -> L = 3*3 = 9 blocks
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let unfold = Unfold::new(2);
        let y = unfold.forward(&x).unwrap();
        // Output: [1, 1*2*2, 3*3] = [1, 4, 9]
        assert_eq!(y.shape(), vec![1, 4, 9]);
    }

    #[test]
    fn test_unfold_with_stride() {
        // [1, 1, 4, 4] with kernel=2, stride=2 -> L = 2*2 = 4 blocks
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let unfold = Unfold::new(2).stride([2, 2]);
        let y = unfold.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 4, 4]);
    }

    #[test]
    fn test_fold_unfold_roundtrip() {
        let x = Variable::new(
            Tensor::randn(&[1, 1, 4, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        // Unfold with non-overlapping blocks (stride=kernel)
        let unfold = Unfold::new(2).stride([2, 2]);
        let cols = unfold.forward(&x).unwrap();
        assert_eq!(cols.shape(), vec![1, 4, 4]);

        // Fold back
        let fold = Fold::new([4, 4], 2).stride([2, 2]);
        let y = fold.forward(&cols).unwrap();
        assert_eq!(y.shape(), vec![1, 1, 4, 4]);

        // Non-overlapping roundtrip should be exact
        let xv = x.data().to_f32_vec().unwrap();
        let yv = y.data().to_f32_vec().unwrap();
        for (a, b) in xv.iter().zip(yv.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip mismatch: {} vs {}", a, b);
        }
    }
}
