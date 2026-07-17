//! Pure tensor operations: arithmetic, element-wise math, activations,
//! reductions, comparisons, masking, sorting, and advanced indexing.

use std::ptr;
use flodl_sys::{self as ffi, FlodlTensor};
use super::{Tensor, check_err, Result};

impl Tensor {
    // --- Arithmetic (chainable) ---

    /// Element-wise addition. Shapes must be broadcastable.
    ///
    /// ```ignore
    /// let c = a.add(&b)?; // [2, 3] + [2, 3] → [2, 3]
    /// ```
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_add(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise subtraction. Shapes must be broadcastable.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sub(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise (Hadamard) multiplication. Shapes must be broadcastable.
    /// For matrix multiplication, use [`matmul`](Self::matmul).
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Matrix multiplication.
    ///
    /// ```ignore
    /// // [batch, M, K] @ [batch, K, N] → [batch, M, N]
    /// let c = a.matmul(&b)?;
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_matmul(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Multiply every element by a scalar. Like `tensor * 0.5` in PyTorch.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mul_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_div(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Negate every element.
    pub fn neg(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_neg(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Add a scalar to every element.
    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_add_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Divide every element by a scalar.
    pub fn div_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_div_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Element-wise math ---

    /// Element-wise exponential.
    pub fn exp(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_exp(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sqrt(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_abs(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Upper triangle of a matrix (or batch of matrices).
    /// Elements below the `diagonal`-th diagonal are zeroed.
    /// `diagonal=0` keeps the main diagonal; `diagonal=1` excludes it.
    pub fn triu(&self, diagonal: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_triu(self.handle, diagonal, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Lower triangle of a matrix (or batch of matrices).
    /// Elements above the `diagonal`-th diagonal are zeroed.
    /// `diagonal=0` keeps the main diagonal; `diagonal=-1` excludes it.
    pub fn tril(&self, diagonal: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tril(self.handle, diagonal, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Raise every element to a scalar exponent.
    pub fn pow_scalar(&self, exponent: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_pow_scalar(self.handle, exponent, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp(self.handle, min, max, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to be at least `min`.
    pub fn clamp_min(&self, min: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp_min(self.handle, min, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Clamp all elements to be at most `max`.
    pub fn clamp_max(&self, max: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_clamp_max(self.handle, max, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise `log(1 + x)`, numerically stable for small x.
    pub fn log1p(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log1p(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise `exp(x) - 1`, numerically stable for small x.
    pub fn expm1(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_expm1(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log2(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise base-10 logarithm.
    pub fn log10(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log10(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sin(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cos(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise tangent.
    pub fn tan(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tan(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise arcsine (inverse sine).
    pub fn asin(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_asin(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise arccosine (inverse cosine).
    pub fn acos(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_acos(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise arctangent (inverse tangent).
    pub fn atan(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_atan(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise sign (-1, 0, or +1).
    pub fn sign(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sign(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floor.
    pub fn floor(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_floor(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise ceiling.
    pub fn ceil(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ceil(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise rounding to nearest integer.
    pub fn round(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_round(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise reciprocal (1/x).
    pub fn reciprocal(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_reciprocal(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise Gauss error function.
    pub fn erf(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_erf(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise complementary error function (1 - erf(x)).
    pub fn erfc(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_erfc(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise truncation (round towards zero).
    pub fn trunc(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_trunc(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise fractional part (x - trunc(x)).
    pub fn frac(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_frac(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floating-point remainder (C fmod semantics).
    pub fn fmod(&self, divisor: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_fmod_scalar(self.handle, divisor, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise floating-point remainder with tensor divisor.
    pub fn fmod_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_fmod_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise remainder (Python modulo semantics).
    pub fn remainder(&self, divisor: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_remainder_scalar(self.handle, divisor, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise remainder with tensor divisor (Python modulo semantics).
    pub fn remainder_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_remainder_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Linear interpolation: self + weight * (end - self).
    pub fn lerp(&self, end: &Tensor, weight: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lerp(self.handle, end.handle, weight, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Linear interpolation with per-element weight tensor.
    pub fn lerp_tensor(&self, end: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lerp_tensor(self.handle, end.handle, weight.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise closeness check: |self - other| <= atol + rtol * |other|.
    pub fn isclose(&self, other: &Tensor, rtol: f64, atol: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_isclose(self.handle, other.handle, rtol, atol, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused: beta * self + alpha * (mat1 @ mat2).
    pub fn addmm(&self, mat1: &Tensor, mat2: &Tensor, beta: f64, alpha: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_addmm(self.handle, mat1.handle, mat2.handle, beta, alpha, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused: self + value * (tensor1 * tensor2).
    pub fn addcmul(&self, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_addcmul(self.handle, tensor1.handle, tensor2.handle, value, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Fused: self + value * (tensor1 / tensor2).
    pub fn addcdiv(&self, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_addcdiv(self.handle, tensor1.handle, tensor2.handle, value, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Activations ---

    /// SELU: `lambda * (max(0, x) + min(0, alpha * (exp(x) - 1)))`.
    /// Self-normalizing activation with fixed alpha and lambda.
    pub fn selu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_selu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Hardswish: `x * clamp(x + 3, 0, 6) / 6`.
    pub fn hardswish(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_hardswish(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Hardsigmoid: `clamp(x + 3, 0, 6) / 6`.
    pub fn hardsigmoid(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_hardsigmoid(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// PReLU: `max(0, x) + weight * min(0, x)` (learnable weight).
    pub fn prelu(&self, weight: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_prelu(self.handle, weight.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_relu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sigmoid activation: 1 / (1 + exp(-x)).
    pub fn sigmoid(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sigmoid(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Tanh activation: element-wise hyperbolic tangent.
    pub fn tanh(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_tanh_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Softmax along a dimension.
    pub fn softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Log-softmax along a dimension (numerically stable).
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_log_softmax(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// GELU activation (native libtorch, erf form):
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`.
    pub fn gelu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gelu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Tanh-approximation GELU (native libtorch). Matches HuggingFace
    /// `hidden_act="gelu_new"` and PyTorch `F.gelu(x, approximate="tanh")`:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    pub fn gelu_tanh(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gelu_tanh(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// SiLU activation (native libtorch).
    pub fn silu(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_silu(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_leaky_relu(self.handle, negative_slope, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// ELU: `max(0, x) + min(0, alpha * (exp(x) - 1))`.
    pub fn elu(&self, alpha: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_elu(self.handle, alpha, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Softplus: `(1/beta) * log(1 + exp(beta * x))`.
    /// Reverts to linear when `beta * x > threshold`.
    pub fn softplus(&self, beta: f64, threshold: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_softplus(self.handle, beta, threshold, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mish: `x * tanh(softplus(x))`.
    pub fn mish(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mish(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Reductions ---

    /// Sum of all elements (scalar result).
    pub fn sum(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_sum(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean of all elements (scalar result).
    pub fn mean(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_mean(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sum along a dimension.
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sum_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Mean along a dimension.
    pub fn mean_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_mean_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Product of all elements (scalar result).
    pub fn prod(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_prod(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Product along a dimension.
    pub fn prod_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_prod_dim(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cumulative sum along a dimension.
    pub fn cumsum(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cumsum(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Log of summed exponentials along a dimension (numerically stable).
    pub fn logsumexp(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_logsumexp(self.handle, dim, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scalar minimum.
    pub fn min(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_min(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scalar maximum.
    pub fn max(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_max(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// L2 (Frobenius) norm of all elements.
    pub fn norm(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_norm(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// p-norm along a dimension.
    pub fn norm_p(&self, p: f64, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_norm_p_dim(self.handle, p, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Sum over multiple dimensions at once.
    pub fn sum_dims(&self, dims: &[i32], keepdim: bool) -> Result<Tensor> {
        let mut dims64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sum_dims(self.handle, dims64.as_mut_ptr(), dims.len() as i32, keepdim as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cumulative product along a dimension.
    pub fn cumprod(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cumprod(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Median of all elements (scalar).
    pub fn median(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_median(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Median along a dimension, returns (values, indices).
    pub fn median_dim(&self, dim: i32, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let mut vals: FlodlTensor = ptr::null_mut();
        let mut idxs: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_median_dim(self.handle, dim, keepdim as i32, &mut vals, &mut idxs) };
        check_err(err)?;
        Ok((Tensor::from_raw(vals), Tensor::from_raw(idxs)))
    }

    /// Count of non-zero elements (scalar).
    pub fn count_nonzero(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_count_nonzero(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Count of non-zero elements along a dimension.
    pub fn count_nonzero_dim(&self, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_count_nonzero_dim(self.handle, dim, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Indices of non-zero elements (N x ndim tensor).
    pub fn nonzero(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_nonzero(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Unique elements (global dedup). Returns (output, inverse_indices).
    /// If `return_inverse` is false, inverse_indices is empty.
    ///
    /// `sorted` controls output ordering only; with `sorted = false` the
    /// order is kernel-defined (PyTorch `torch.unique` semantics). For
    /// adjacent-run dedup, use [`Tensor::unique_consecutive`].
    pub fn unique(&self, sorted: bool, return_inverse: bool) -> Result<(Tensor, Tensor)> {
        let mut output: FlodlTensor = ptr::null_mut();
        let mut inverse: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_unique(self.handle, sorted as i32, return_inverse as i32, &mut output, &mut inverse)
        };
        check_err(err)?;
        let inv = if inverse.is_null() {
            Tensor::from_i64(&[], &[0], super::Device::CPU)?
        } else {
            Tensor::from_raw(inverse)
        };
        Ok((Tensor::from_raw(output), inv))
    }

    /// Eliminate consecutive duplicates only (PyTorch `torch.unique_consecutive`):
    /// `[1, 1, 2, 2, 1]` -> `[1, 2, 1]`. Returns (output, inverse_indices).
    /// If `return_inverse` is false, inverse_indices is empty.
    pub fn unique_consecutive(&self, return_inverse: bool) -> Result<(Tensor, Tensor)> {
        let mut output: FlodlTensor = ptr::null_mut();
        let mut inverse: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_unique_consecutive(self.handle, return_inverse as i32, &mut output, &mut inverse)
        };
        check_err(err)?;
        let inv = if inverse.is_null() {
            Tensor::from_i64(&[], &[0], super::Device::CPU)?
        } else {
            Tensor::from_raw(inverse)
        };
        Ok((Tensor::from_raw(output), inv))
    }

    /// Binary search: find insertion indices in a sorted sequence.
    pub fn searchsorted(&self, values: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_searchsorted(self.handle, values.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Minimum along a dimension (values only).
    pub fn min_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_min_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Maximum along a dimension (values only).
    pub fn max_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_max_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Argmax along a dimension.
    pub fn argmax(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_argmax(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Argmin along a dimension.
    pub fn argmin(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_argmin(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Variance of all elements (Bessel-corrected).
    pub fn var(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_var(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Standard deviation of all elements (Bessel-corrected).
    #[allow(clippy::should_implement_trait)]
    pub fn std(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_std_op(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Variance along a dimension (Bessel-corrected).
    pub fn var_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_var_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Standard deviation along a dimension (Bessel-corrected).
    pub fn std_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_std_dim(self.handle, dim, keepdim as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Comparisons ---

    /// Element-wise greater-than comparison against a scalar.
    pub fn gt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than-or-equal comparison against a scalar.
    pub fn ge_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ge_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than-or-equal comparison against a scalar.
    pub fn le_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_le_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than comparison against a scalar.
    pub fn lt_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lt_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise equality comparison against a scalar (returns float mask: 0.0 or 1.0).
    pub fn eq_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_eq_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise not-equal comparison against a scalar (returns float mask: 0.0 or 1.0).
    pub fn ne_scalar(&self, scalar: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ne_scalar(self.handle, scalar, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise NaN detection (returns float mask: 0.0 or 1.0).
    pub fn isnan(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_isnan(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise infinity detection (returns float mask: 0.0 or 1.0).
    pub fn isinf(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_isinf(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise logical AND of two tensors (returns float mask).
    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_logical_and(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise logical OR of two tensors (returns float mask).
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_logical_or(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise logical NOT (returns float mask).
    pub fn logical_not(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_logical_not(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Returns a scalar float tensor: 1.0 if any element is non-zero, 0.0 otherwise.
    pub fn any(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_any(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Returns a scalar float tensor: 1.0 if all elements are non-zero, 0.0 otherwise.
    pub fn all(&self) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_all(self.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise atan2 (arc tangent of y/x).
    pub fn atan2(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_atan2(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise maximum of two tensors.
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_maximum(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise minimum of two tensors.
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_minimum(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than (returns float mask: 0.0 or 1.0).
    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_gt_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than (returns float mask: 0.0 or 1.0).
    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_lt_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise greater-than-or-equal (returns float mask: 0.0 or 1.0).
    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ge_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise less-than-or-equal (returns float mask: 0.0 or 1.0).
    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_le_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise equality. Returns a mask (0.0 or 1.0) in the input's
    /// dtype for float inputs, or Float32 for integer/bool inputs.
    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_eq_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Element-wise not-equal. Returns a mask (0.0 or 1.0) in the input's
    /// dtype for float inputs, or Float32 for integer/bool inputs.
    pub fn ne_tensor(&self, other: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_ne_tensor(self.handle, other.handle, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Masking/conditional ---

    /// Fill elements where `mask` is true (non-zero) with `value`.
    /// The mask is broadcast to match the tensor shape.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_masked_fill(self.handle, mask.handle, value, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Conditional select: where(condition, self, other).
    pub fn where_cond(condition: &Tensor, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_where(condition.handle, x.handle, y.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Sorting ---

    /// Top-k values and indices along a dimension. Returns (values, indices).
    pub fn topk(&self, k: i64, dim: i32, largest: bool, sorted: bool) -> Result<(Tensor, Tensor)> {
        let mut values: FlodlTensor = ptr::null_mut();
        let mut indices: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_topk(
                self.handle, k, dim, largest as i32, sorted as i32,
                &mut values, &mut indices,
            )
        };
        check_err(err)?;
        Ok((Tensor::from_raw(values), Tensor::from_raw(indices)))
    }

    /// Sort along a dimension. Returns (sorted_values, indices).
    pub fn sort(&self, dim: i32, descending: bool) -> Result<(Tensor, Tensor)> {
        let mut values: FlodlTensor = ptr::null_mut();
        let mut indices: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_sort(self.handle, dim, descending as i32, &mut values, &mut indices)
        };
        check_err(err)?;
        Ok((Tensor::from_raw(values), Tensor::from_raw(indices)))
    }

    /// Return indices that would sort the tensor along a dimension.
    pub fn argsort(&self, dim: i32, descending: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_argsort(self.handle, dim, descending as i32, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Advanced indexing ---

    /// Gather values along a dimension using an index tensor.
    pub fn gather(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_gather(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add: accumulate src into self at index positions along dim.
    pub fn scatter_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_scatter_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter: write src values into self at index positions along dim (replaces, not adds).
    pub fn scatter(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_scatter(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Select rows/elements along a dimension using an index tensor.
    pub fn index_select(&self, dim: i32, index: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_index_select(self.handle, dim, index.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter-add src into self along dim at positions given by index.
    pub fn index_add(&self, dim: i32, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_index_add(self.handle, dim, index.handle, src.handle, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Scatter a selected index back into a tensor.
    pub fn select_scatter(&self, src: &Tensor, dim: i32, index: i64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_select_scatter(self.handle, src.handle, dim, index, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    // --- Other ---

    /// L_p normalize along a dimension (default: L2, dim=-1).
    pub fn normalize(&self, p: f64, dim: i32) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_normalize(self.handle, p, dim, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Draw samples from a multinomial distribution.
    /// `self` contains unnormalized probabilities (one row per distribution).
    pub fn multinomial(&self, num_samples: i64, replacement: bool) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_multinomial(
                self.handle, num_samples, replacement as i32, &mut handle,
            )
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Pairwise L2 distance between rows of two batched matrices.
    /// Input shapes: `[B, P, D]` and `[B, R, D]` -> output `[B, P, R]`.
    pub fn cdist(&self, other: &Tensor) -> Result<Tensor> {
        self.cdist_p(other, 2.0)
    }

    /// Pairwise distance with custom p-norm.
    pub fn cdist_p(&self, other: &Tensor, p: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_cdist(self.handle, other.handle, p, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cosine similarity between two tensors along a dimension.
    /// Default dim=1, eps=1e-8 (matches PyTorch).
    pub fn cosine_similarity(&self, other: &Tensor, dim: i64, eps: f64) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe {
            ffi::flodl_cosine_similarity(self.handle, other.handle, dim, eps, &mut handle)
        };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Cast to a different dtype.
    pub fn to_dtype(&self, dtype: super::DType) -> Result<Tensor> {
        let mut handle: FlodlTensor = ptr::null_mut();
        let err = unsafe { ffi::flodl_to_dtype(self.handle, dtype as i32, &mut handle) };
        check_err(err)?;
        Ok(Tensor::from_raw(handle))
    }

    /// Check if all elements are finite (no inf/nan).
    pub fn all_finite(&self) -> Result<bool> {
        let mut result: i32 = 0;
        let err = unsafe { ffi::flodl_all_finite(self.handle, &mut result) };
        check_err(err)?;
        Ok(result != 0)
    }
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
