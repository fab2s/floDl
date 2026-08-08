//! Variable operations backed by libtorch native autograd.
//!
//! Every op calls the Tensor FFI method and wraps the result.
//! libtorch's C++ autograd engine tracks the computation graph
//! and computes gradients natively — no Rust-side backward closures.

use crate::tensor::{Result, Tensor};

use super::variable::Variable;

impl Variable {
    // --- Arithmetic ---

    /// Element-wise addition.
    pub fn add(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().add(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().sub(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().mul(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise division.
    pub fn div(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().div(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Matrix multiplication (`torch.matmul`). Supports batched matmul.
    pub fn matmul(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().matmul(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Multiply all elements by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.data().mul_scalar(scalar)?;
        Ok(Variable::wrap(result))
    }

    /// Divide all elements by a scalar.
    pub fn div_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.data().div_scalar(scalar)?;
        Ok(Variable::wrap(result))
    }

    /// Add a scalar to all elements.
    pub fn add_scalar(&self, scalar: f64) -> Result<Variable> {
        let result = self.data().add_scalar(scalar)?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Result<Variable> {
        let result = self.data().neg()?;
        Ok(Variable::wrap(result))
    }

    // --- Activations ---

    /// ReLU activation: `max(0, x)`.
    pub fn relu(&self) -> Result<Variable> {
        let result = self.data().relu()?;
        Ok(Variable::wrap(result))
    }

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    pub fn sigmoid(&self) -> Result<Variable> {
        let result = self.data().sigmoid()?;
        Ok(Variable::wrap(result))
    }

    /// Tanh activation: element-wise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Variable> {
        let result = self.data().tanh()?;
        Ok(Variable::wrap(result))
    }

    /// GELU activation (Gaussian Error Linear Unit, erf form):
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`.
    pub fn gelu(&self) -> Result<Variable> {
        let result = self.data().gelu()?;
        Ok(Variable::wrap(result))
    }

    /// Tanh-approximation GELU. Matches HuggingFace `hidden_act="gelu_new"`
    /// and PyTorch `F.gelu(x, approximate="tanh")`. Used by ALBERT, GPT-2,
    /// and any HF checkpoint that ships `hidden_act` in {`gelu_new`,
    /// `gelu_pytorch_tanh`}.
    pub fn gelu_tanh(&self) -> Result<Variable> {
        let result = self.data().gelu_tanh()?;
        Ok(Variable::wrap(result))
    }

    /// SiLU activation (Sigmoid Linear Unit, aka Swish): `x * sigmoid(x)`.
    pub fn silu(&self) -> Result<Variable> {
        let result = self.data().silu()?;
        Ok(Variable::wrap(result))
    }

    /// Leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Variable> {
        let result = self.data().leaky_relu(negative_slope)?;
        Ok(Variable::wrap(result))
    }

    /// ELU: `max(0, x) + min(0, alpha * (exp(x) - 1))`.
    pub fn elu(&self, alpha: f64) -> Result<Variable> {
        let result = self.data().elu(alpha)?;
        Ok(Variable::wrap(result))
    }

    /// Softplus: `(1/beta) * log(1 + exp(beta * x))`.
    pub fn softplus(&self, beta: f64, threshold: f64) -> Result<Variable> {
        let result = self.data().softplus(beta, threshold)?;
        Ok(Variable::wrap(result))
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(&self) -> Result<Variable> {
        let result = self.data().mish()?;
        Ok(Variable::wrap(result))
    }

    /// SELU: self-normalizing ELU with fixed alpha/lambda.
    pub fn selu(&self) -> Result<Variable> {
        let result = self.data().selu()?;
        Ok(Variable::wrap(result))
    }

    /// Hardswish: `x * clamp(x + 3, 0, 6) / 6`.
    pub fn hardswish(&self) -> Result<Variable> {
        let result = self.data().hardswish()?;
        Ok(Variable::wrap(result))
    }

    /// Hardsigmoid: `clamp(x + 3, 0, 6) / 6`.
    pub fn hardsigmoid(&self) -> Result<Variable> {
        let result = self.data().hardsigmoid()?;
        Ok(Variable::wrap(result))
    }

    /// PReLU: `max(0, x) + weight * min(0, x)`.
    pub fn prelu(&self, weight: &Variable) -> Result<Variable> {
        let result = self.data().prelu(&weight.data())?;
        Ok(Variable::wrap(result))
    }

    // --- Reductions ---

    /// Sum of all elements, returning a scalar.
    pub fn sum(&self) -> Result<Variable> {
        let result = self.data().sum()?;
        Ok(Variable::wrap(result))
    }

    /// Mean of all elements, returning a scalar.
    pub fn mean(&self) -> Result<Variable> {
        let result = self.data().mean()?;
        Ok(Variable::wrap(result))
    }

    /// Sum along a dimension. If `keepdim`, the reduced dimension is retained with size 1.
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().sum_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Mean along a dimension. If `keepdim`, the reduced dimension is retained with size 1.
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().mean_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Product of all elements, returning a scalar.
    pub fn prod(&self) -> Result<Variable> {
        let result = self.data().prod()?;
        Ok(Variable::wrap(result))
    }

    /// Product along a dimension.
    pub fn prod_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().prod_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Cumulative sum along a dimension.
    pub fn cumsum(&self, dim: i32) -> Result<Variable> {
        let result = self.data().cumsum(dim)?;
        Ok(Variable::wrap(result))
    }

    /// Log of summed exponentials along a dimension (numerically stable).
    pub fn logsumexp(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().logsumexp(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Minimum of all elements, returning a scalar.
    pub fn min(&self) -> Result<Variable> {
        let result = self.data().min()?;
        Ok(Variable::wrap(result))
    }

    /// Maximum of all elements, returning a scalar.
    pub fn max(&self) -> Result<Variable> {
        let result = self.data().max()?;
        Ok(Variable::wrap(result))
    }

    /// Minimum values along a dimension.
    pub fn min_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().min_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Maximum values along a dimension.
    pub fn max_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().max_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Variance of all elements (Bessel-corrected).
    pub fn var(&self) -> Result<Variable> {
        let result = self.data().var()?;
        Ok(Variable::wrap(result))
    }

    /// Standard deviation of all elements (Bessel-corrected).
    pub fn std(&self) -> Result<Variable> {
        let result = self.data().std()?;
        Ok(Variable::wrap(result))
    }

    /// Variance along a dimension (Bessel-corrected).
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().var_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    /// Standard deviation along a dimension (Bessel-corrected).
    pub fn std_dim(&self, dim: i32, keepdim: bool) -> Result<Variable> {
        let result = self.data().std_dim(dim, keepdim)?;
        Ok(Variable::wrap(result))
    }

    // --- Softmax ---

    /// Softmax along `dim` (output sums to 1).
    pub fn softmax(&self, dim: i32) -> Result<Variable> {
        let result = self.data().softmax(dim)?;
        Ok(Variable::wrap(result))
    }

    /// Log-softmax along `dim` (numerically stable `log(softmax(x))`).
    pub fn log_softmax(&self, dim: i32) -> Result<Variable> {
        let result = self.data().log_softmax(dim)?;
        Ok(Variable::wrap(result))
    }

    // --- Element-wise math ---

    /// Element-wise exponential.
    pub fn exp(&self) -> Result<Variable> {
        let result = self.data().exp()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Variable> {
        let result = self.data().log()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Variable> {
        let result = self.data().sqrt()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Variable> {
        let result = self.data().abs()?;
        Ok(Variable::wrap(result))
    }

    /// Raise each element to a scalar power.
    pub fn pow_scalar(&self, exponent: f64) -> Result<Variable> {
        let result = self.data().pow_scalar(exponent)?;
        Ok(Variable::wrap(result))
    }

    /// Upper triangular part. `diagonal=0` keeps the main diagonal; positive shifts up.
    pub fn triu(&self, diagonal: i64) -> Result<Variable> {
        let result = self.data().triu(diagonal)?;
        Ok(Variable::wrap(result))
    }

    /// Lower triangular part. `diagonal=0` keeps the main diagonal; negative shifts down.
    pub fn tril(&self, diagonal: i64) -> Result<Variable> {
        let result = self.data().tril(diagonal)?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Variable> {
        let result = self.data().sin()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Variable> {
        let result = self.data().cos()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise sign (-1, 0, or 1).
    pub fn sign(&self) -> Result<Variable> {
        let result = self.data().sign()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise floor (round toward negative infinity).
    pub fn floor(&self) -> Result<Variable> {
        let result = self.data().floor()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise ceil (round toward positive infinity).
    pub fn ceil(&self) -> Result<Variable> {
        let result = self.data().ceil()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise round to nearest integer.
    pub fn round(&self) -> Result<Variable> {
        let result = self.data().round()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise reciprocal (`1/x`).
    pub fn reciprocal(&self) -> Result<Variable> {
        let result = self.data().reciprocal()?;
        Ok(Variable::wrap(result))
    }

    /// Clamp all elements to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Result<Variable> {
        let result = self.data().clamp(min, max)?;
        Ok(Variable::wrap(result))
    }

    /// Clamp all elements to be at least `min`.
    pub fn clamp_min(&self, min: f64) -> Result<Variable> {
        let result = self.data().clamp_min(min)?;
        Ok(Variable::wrap(result))
    }

    /// Clamp all elements to be at most `max`.
    pub fn clamp_max(&self, max: f64) -> Result<Variable> {
        let result = self.data().clamp_max(max)?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise `log(1 + x)`, numerically stable for small x.
    pub fn log1p(&self) -> Result<Variable> {
        let result = self.data().log1p()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise `exp(x) - 1`, numerically stable for small x.
    pub fn expm1(&self) -> Result<Variable> {
        let result = self.data().expm1()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> Result<Variable> {
        let result = self.data().log2()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise base-10 logarithm.
    pub fn log10(&self) -> Result<Variable> {
        let result = self.data().log10()?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise atan2 (arc tangent of y/x).
    pub fn atan2(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().atan2(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise maximum of two variables.
    pub fn maximum(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().maximum(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Element-wise minimum of two variables.
    pub fn minimum(&self, other: &Variable) -> Result<Variable> {
        let result = self.data().minimum(&other.data())?;
        Ok(Variable::wrap(result))
    }

    /// Fill elements where `mask` is true (non-zero) with `value`.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Result<Variable> {
        let result = self.data().masked_fill(mask, value)?;
        Ok(Variable::wrap(result))
    }

    /// L_p normalize along a dimension.
    pub fn normalize(&self, p: f64, dim: i32) -> Result<Variable> {
        let result = self.data().normalize(p, dim)?;
        Ok(Variable::wrap(result))
    }

    /// Cosine similarity between two variables along a dimension.
    pub fn cosine_similarity(&self, other: &Variable, dim: i64, eps: f64) -> Result<Variable> {
        let result = self.data().cosine_similarity(&other.data(), dim, eps)?;
        Ok(Variable::wrap(result))
    }

    // --- Shape operations ---

    /// Return a view with a different shape. Use `-1` for one inferred dimension.
    pub fn reshape(&self, shape: &[i64]) -> Result<Variable> {
        let result = self.data().reshape(shape)?;
        Ok(Variable::wrap(result))
    }

    /// Swap two dimensions.
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Variable> {
        let result = self.data().transpose(dim0, dim1)?;
        Ok(Variable::wrap(result))
    }

    /// Reorder dimensions according to the given permutation.
    pub fn permute(&self, dims: &[i64]) -> Result<Variable> {
        let result = self.data().permute(dims)?;
        Ok(Variable::wrap(result))
    }

    /// Remove a dimension of size 1.
    pub fn squeeze(&self, dim: i32) -> Result<Variable> {
        let result = self.data().squeeze(dim)?;
        Ok(Variable::wrap(result))
    }

    /// Insert a dimension of size 1 at `dim`.
    pub fn unsqueeze(&self, dim: i32) -> Result<Variable> {
        let result = self.data().unsqueeze(dim)?;
        Ok(Variable::wrap(result))
    }

    /// Insert size-1 dimensions at multiple positions.
    pub fn unsqueeze_many(&self, dims: &[i32]) -> Result<Variable> {
        let result = self.data().unsqueeze_many(dims)?;
        Ok(Variable::wrap(result))
    }

    /// Flatten dimensions from `start_dim` to `end_dim` (inclusive) into one.
    pub fn flatten(&self, start_dim: i32, end_dim: i32) -> Result<Variable> {
        let result = self.data().flatten(start_dim, end_dim)?;
        Ok(Variable::wrap(result))
    }

    /// Broadcast to `shape` without copying data. Use `-1` to keep a dimension.
    pub fn expand(&self, shape: &[i64]) -> Result<Variable> {
        let result = self.data().expand(shape)?;
        Ok(Variable::wrap(result))
    }

    // --- Indexing & slicing ---

    /// Slice along `dim` from `start` for `length` elements.
    pub fn narrow(&self, dim: i32, start: i64, length: i64) -> Result<Variable> {
        let result = self.data().narrow(dim, start, length)?;
        Ok(Variable::wrap(result))
    }

    /// Select a single index along `dim`, removing that dimension.
    pub fn select(&self, dim: i32, index: i64) -> Result<Variable> {
        let result = self.data().select(dim, index)?;
        Ok(Variable::wrap(result))
    }

    /// Select multiple indices along `dim` using a 1-D index tensor.
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Variable> {
        let result = self.data().index_select(dim, index)?;
        Ok(Variable::wrap(result))
    }

    /// Gather values along `dim` using an index tensor of the same rank.
    pub fn gather(&self, dim: i32, index: &Tensor) -> Result<Variable> {
        let result = self.data().gather(dim, index)?;
        Ok(Variable::wrap(result))
    }

    /// Concatenate two variables along `dim`.
    pub fn cat(&self, other: &Variable, dim: i32) -> Result<Variable> {
        let result = self.data().cat(&other.data(), dim)?;
        Ok(Variable::wrap(result))
    }

    /// Concatenate multiple variables along `dim` (`torch.cat`).
    pub fn cat_many(vars: &[&Variable], dim: i32) -> Result<Variable> {
        let tensors: Vec<Tensor> = vars.iter().map(|v| v.data()).collect();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let result = Tensor::cat_many(&refs, dim)?;
        Ok(Variable::wrap(result))
    }

    /// Stack variables along a new dimension (`torch.stack`).
    pub fn stack(vars: &[Variable], dim: i32) -> Result<Variable> {
        let tensors: Vec<Tensor> = vars.iter().map(|v| v.data()).collect();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let result = Tensor::stack(&refs, dim)?;
        Ok(Variable::wrap(result))
    }

    /// Split into `chunks` pieces along `dim` (`torch.chunk`).
    pub fn chunk(&self, chunks: i32, dim: i32) -> Result<Vec<Variable>> {
        let tensors = self.data().chunk(chunks, dim)?;
        Ok(tensors.into_iter().map(Variable::wrap).collect())
    }

    /// Repeat the tensor along each dimension by the given counts.
    pub fn repeat(&self, repeats: &[i64]) -> Result<Variable> {
        let result = self.data().repeat(repeats)?;
        Ok(Variable::wrap(result))
    }

    /// Constant-value padding. `padding` is `[left, right, ...]` from the last dim inward.
    pub fn pad(&self, padding: &[i64], value: f64) -> Result<Variable> {
        let result = self.data().pad(padding, value)?;
        Ok(Variable::wrap(result))
    }

    // --- Sorting ---

    /// Return the `k` largest (or smallest) values and their indices along `dim`.
    pub fn topk(
        &self,
        k: i64,
        dim: i32,
        largest: bool,
        sorted: bool,
    ) -> Result<(Variable, Tensor)> {
        let (values, indices) = self.data().topk(k, dim, largest, sorted)?;
        Ok((Variable::wrap(values), indices))
    }

    /// Sort along `dim`. Returns `(sorted_values, indices)`.
    pub fn sort(&self, dim: i32, descending: bool) -> Result<(Variable, Tensor)> {
        let (values, indices) = self.data().sort(dim, descending)?;
        Ok((Variable::wrap(values), indices))
    }
}

// --- Standalone differentiable ops ---

/// Fused linear: `y = input @ weight^T + bias` with autograd support.
/// Uses `torch::linear()` (single BLAS kernel) instead of separate
/// transpose + matmul + add.
pub fn linear(input: &Variable, weight: &Variable, bias: Option<&Variable>) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().linear(&weight.data(), bias_tensor.as_ref())?;
    Ok(Variable::wrap(result))
}

/// Fused GRU cell with autograd support.
/// Uses `torch::gru_cell()` (~2 kernels) instead of 6 separate linear ops.
#[allow(clippy::too_many_arguments)]
pub fn gru_cell(
    input: &Variable,
    hx: &Variable,
    w_ih: &Variable,
    w_hh: &Variable,
    b_ih: &Variable,
    b_hh: &Variable,
) -> Result<Variable> {
    let result = input.data().gru_cell(
        &hx.data(),
        &w_ih.data(),
        &w_hh.data(),
        &b_ih.data(),
        &b_hh.data(),
    )?;
    Ok(Variable::wrap(result))
}

/// Fused LSTM cell with autograd support.
/// Uses `torch::lstm_cell()` (~2 kernels) instead of 8 separate linear ops.
/// Returns `(h', c')`.
#[allow(clippy::too_many_arguments)]
pub fn lstm_cell(
    input: &Variable,
    hx: &Variable,
    cx: &Variable,
    w_ih: &Variable,
    w_hh: &Variable,
    b_ih: &Variable,
    b_hh: &Variable,
) -> Result<(Variable, Variable)> {
    let (h, c) = input.data().lstm_cell(
        &hx.data(),
        &cx.data(),
        &w_ih.data(),
        &w_hh.data(),
        &b_ih.data(),
        &b_hh.data(),
    )?;
    Ok((Variable::wrap(h), Variable::wrap(c)))
}

/// Layer normalization with autograd support.
/// Normalizes over the last `normalized_size` elements using `weight` (gamma) and `bias` (beta).
pub fn layer_norm(
    input: &Variable,
    weight: &Variable,
    bias: &Variable,
    normalized_size: i64,
    eps: f64,
) -> Result<Variable> {
    let (output, _mean, _rstd) =
        input
            .data()
            .native_layer_norm(&weight.data(), &bias.data(), normalized_size, eps)?;
    Ok(Variable::wrap(output))
}

/// 2D convolution with autograd support (`F.conv2d`).
/// `input` is `[N, C_in, H, W]`, `weight` is `[C_out, C_in/groups, kH, kW]`.
pub fn conv2d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv2d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride,
        padding,
        dilation,
        groups,
    )?;
    Ok(Variable::wrap(result))
}

/// Transposed 2D convolution with autograd support (`F.conv_transpose2d`).
/// `output_padding` resolves the ambiguity in output shape for fractionally-strided convolutions.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: [i64; 2],
    padding: [i64; 2],
    output_padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv_transpose2d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    Ok(Variable::wrap(result))
}

/// 1D convolution with autograd support (`F.conv1d`).
/// `input` is `[N, C_in, L]`, `weight` is `[C_out, C_in/groups, K]`.
pub fn conv1d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: i64,
    padding: i64,
    dilation: i64,
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv1d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride,
        padding,
        dilation,
        groups,
    )?;
    Ok(Variable::wrap(result))
}

/// Transposed 1D convolution with autograd support (`F.conv_transpose1d`).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: i64,
    padding: i64,
    output_padding: i64,
    dilation: i64,
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv_transpose1d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    Ok(Variable::wrap(result))
}

/// Group normalization with autograd support.
/// `weight` (gamma) and `bias` (beta) are shape `[num_channels]`.
pub fn group_norm(
    input: &Variable,
    num_groups: i64,
    weight: &Variable,
    bias: &Variable,
    eps: f64,
) -> Result<Variable> {
    let result =
        input
            .data()
            .group_norm(num_groups, Some(&weight.data()), Some(&bias.data()), eps)?;
    Ok(Variable::wrap(result))
}

/// Max pooling over a 2D input with autograd support (`F.max_pool2d`).
pub fn max_pool2d(
    input: &Variable,
    kernel_size: [i64; 2],
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    ceil_mode: bool,
) -> Result<Variable> {
    let result = input
        .data()
        .max_pool2d(kernel_size, stride, padding, dilation, ceil_mode)?;
    Ok(Variable::wrap(result))
}

/// Average pooling over spatial dimensions.
pub fn avg_pool2d(
    input: &Variable,
    kernel_size: [i64; 2],
    stride: [i64; 2],
    padding: [i64; 2],
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Variable> {
    let result =
        input
            .data()
            .avg_pool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)?;
    Ok(Variable::wrap(result))
}

/// Adaptive average pooling that outputs a fixed `[H, W]` regardless of input size.
pub fn adaptive_avg_pool2d(input: &Variable, output_size: [i64; 2]) -> Result<Variable> {
    let result = input.data().adaptive_avg_pool2d(output_size)?;
    Ok(Variable::wrap(result))
}

/// Unfold (im2col) with autograd support.
pub fn im2col(
    input: &Variable,
    kernel_size: [i64; 2],
    dilation: [i64; 2],
    padding: [i64; 2],
    stride: [i64; 2],
) -> Result<Variable> {
    let result = input
        .data()
        .im2col(kernel_size, dilation, padding, stride)?;
    Ok(Variable::wrap(result))
}

/// Fold (col2im) with autograd support.
pub fn col2im(
    input: &Variable,
    output_size: [i64; 2],
    kernel_size: [i64; 2],
    dilation: [i64; 2],
    padding: [i64; 2],
    stride: [i64; 2],
) -> Result<Variable> {
    let result = input
        .data()
        .col2im(output_size, kernel_size, dilation, padding, stride)?;
    Ok(Variable::wrap(result))
}

/// 3D convolution with autograd support (`F.conv3d`).
/// `input` is `[N, C_in, D, H, W]`, `weight` is `[C_out, C_in/groups, kD, kH, kW]`.
#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: [i64; 3],
    padding: [i64; 3],
    dilation: [i64; 3],
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv3d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride,
        padding,
        dilation,
        groups,
    )?;
    Ok(Variable::wrap(result))
}

/// Transposed 3D convolution with autograd support.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d(
    input: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
    stride: [i64; 3],
    padding: [i64; 3],
    output_padding: [i64; 3],
    dilation: [i64; 3],
    groups: i64,
) -> Result<Variable> {
    let bias_tensor = bias.map(|b| b.data());
    let result = input.data().conv_transpose3d(
        &weight.data(),
        bias_tensor.as_ref(),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;
    Ok(Variable::wrap(result))
}

/// 1D max pooling with autograd support.
pub fn max_pool1d(
    input: &Variable,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
    ceil_mode: bool,
) -> Result<Variable> {
    let result = input
        .data()
        .max_pool1d(kernel_size, stride, padding, dilation, ceil_mode)?;
    Ok(Variable::wrap(result))
}

/// 1D average pooling with autograd support.
pub fn avg_pool1d(
    input: &Variable,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool,
) -> Result<Variable> {
    let result =
        input
            .data()
            .avg_pool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)?;
    Ok(Variable::wrap(result))
}

/// Adaptive max pooling 2D with autograd support.
pub fn adaptive_max_pool2d(input: &Variable, output_size: [i64; 2]) -> Result<Variable> {
    let result = input.data().adaptive_max_pool2d(output_size)?;
    Ok(Variable::wrap(result))
}

/// Instance normalization with autograd support.
#[allow(clippy::too_many_arguments)]
pub fn instance_norm(
    input: &Variable,
    weight: Option<&Variable>,
    bias: Option<&Variable>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    use_input_stats: bool,
    momentum: f64,
    eps: f64,
) -> Result<Variable> {
    let w = weight.map(|v| v.data());
    let b = bias.map(|v| v.data());
    let result = input.data().instance_norm(
        w.as_ref(),
        b.as_ref(),
        running_mean,
        running_var,
        use_input_stats,
        momentum,
        eps,
    )?;
    Ok(Variable::wrap(result))
}

/// Pixel shuffle with autograd support.
pub fn pixel_shuffle(input: &Variable, upscale_factor: i64) -> Result<Variable> {
    let result = input.data().pixel_shuffle(upscale_factor)?;
    Ok(Variable::wrap(result))
}

/// Pixel unshuffle with autograd support.
pub fn pixel_unshuffle(input: &Variable, downscale_factor: i64) -> Result<Variable> {
    let result = input.data().pixel_unshuffle(downscale_factor)?;
    Ok(Variable::wrap(result))
}

/// Bilinear transformation with autograd support.
pub fn bilinear(
    input1: &Variable,
    input2: &Variable,
    weight: &Variable,
    bias: Option<&Variable>,
) -> Result<Variable> {
    let b = bias.map(|v| v.data());
    let result = Tensor::bilinear(&input1.data(), &input2.data(), &weight.data(), b.as_ref())?;
    Ok(Variable::wrap(result))
}

/// Grid sampling with autograd support (`F.grid_sample`).
/// `mode`: 0=bilinear, 1=nearest, 2=bicubic. `padding_mode`: 0=zeros, 1=border, 2=reflection.
pub fn grid_sample(
    input: &Variable,
    grid: &Variable,
    mode: i32,
    padding_mode: i32,
    align_corners: bool,
) -> Result<Variable> {
    let result = input
        .data()
        .grid_sample(&grid.data(), mode, padding_mode, align_corners)?;
    Ok(Variable::wrap(result))
}

/// Fused scaled dot-product attention with autograd support.
///
/// Thin autograd wrapper around libtorch's `at::scaled_dot_product_attention`.
/// Backward is handled by libtorch's native autograd (same pattern as
/// [`embedding`]); Q/K/V weights receive gradients automatically when they
/// are `Variable`s produced by `Linear::forward` etc.
///
/// Shapes and semantics match [`Tensor::scaled_dot_product_attention`].
pub fn scaled_dot_product_attention(
    query: &Variable,
    key: &Variable,
    value: &Variable,
    attn_mask: Option<&Tensor>,
    dropout_p: f64,
    is_causal: bool,
    scale: Option<f64>,
) -> Result<Variable> {
    let result = Tensor::scaled_dot_product_attention(
        &query.data(),
        &key.data(),
        &value.data(),
        attn_mask,
        dropout_p,
        is_causal,
        scale,
    )?;
    Ok(Variable::wrap(result))
}

/// Plain embedding lookup with autograd support.
///
/// `weight`: learnable embedding table (Variable).
/// `indices`: i64 index tensor of any shape.
/// `padding_idx`: row masked to zero gradient during backward, or `-1` to
/// disable padding entirely.
///
/// Output shape: `[*indices.shape, embedding_dim]`.
pub fn embedding(weight: &Variable, indices: &Tensor, padding_idx: i64) -> Result<Variable> {
    let result = Tensor::embedding(&weight.data(), indices, padding_idx)?;
    Ok(Variable::wrap(result))
}

/// Fused embedding lookup + reduction with autograd support.
///
/// `weight`: learnable embedding table (Variable).
/// `indices`: 1-D i64 index tensor.
/// `offsets`: 1-D i64 tensor marking the start of each bag.
/// `mode`: 0 = sum, 1 = mean, 2 = max.
pub fn embedding_bag(
    weight: &Variable,
    indices: &Tensor,
    offsets: &Tensor,
    mode: i64,
) -> Result<Variable> {
    let result = Tensor::embedding_bag(&weight.data(), indices, offsets, mode)?;
    Ok(Variable::wrap(result))
}
