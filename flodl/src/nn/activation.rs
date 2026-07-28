use crate::autograd::Variable;
use crate::tensor::Result;

use super::{Module, Parameter};

/// Identity pass-through module. Returns its input unchanged.
///
/// Useful as a tagging entry point in graphs:
/// ```ignore
/// FlowBuilder::from(Identity).tag("image")
/// ```
pub struct Identity;

impl Default for Identity {
    fn default() -> Self {
        Identity
    }
}

impl Identity {
    /// Create an Identity module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    fn name(&self) -> &str { "identity" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
}

/// ReLU activation: `max(0, x)`. Zeroes negative values.
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        ReLU
    }
}

impl ReLU {
    /// Create a ReLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn name(&self) -> &str { "relu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.relu()
    }
}

/// Sigmoid activation: `1 / (1 + exp(-x))`. Maps to (0, 1).
pub struct Sigmoid;

impl Default for Sigmoid {
    fn default() -> Self {
        Sigmoid
    }
}

impl Sigmoid {
    /// Create a Sigmoid activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn name(&self) -> &str { "sigmoid" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sigmoid()
    }
}

/// Tanh activation: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Maps to (-1, 1).
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Tanh
    }
}

impl Tanh {
    /// Create a Tanh activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn name(&self) -> &str { "tanh" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.tanh()
    }
}

/// GELU approximation form.
///
/// Default ([`GeluApprox::Exact`]) is the erf-form GELU. The tanh
/// approximation ([`GeluApprox::Tanh`]) is what HuggingFace ships under
/// `hidden_act="gelu_new"` (and `"gelu_pytorch_tanh"`), used by ALBERT
/// (v1+v2), GPT-2, and derivative checkpoints. Picking the wrong form
/// silently produces a numerically small per-token diff that compounds
/// across encoder layers — typical end-to-end max-abs-diff is ~1e-2
/// vs ~1e-5 with the matching form, large enough to fail any
/// meaningful parity test.
///
/// Adding a new variant later (e.g. a polynomial fit) is the reason
/// this is an enum rather than a `bool`: each downstream `match` site
/// fails to compile until the new path is handled, which is exactly
/// what we want for a numerically-distinct activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeluApprox {
    /// Exact erf form: `0.5 * x * (1 + erf(x / sqrt(2)))`. Matches
    /// PyTorch `F.gelu(x)` and HuggingFace `hidden_act="gelu"`.
    #[default]
    Exact,
    /// Tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    /// Matches PyTorch `F.gelu(x, approximate="tanh")` and HuggingFace
    /// `hidden_act` in {`"gelu_new"`, `"gelu_pytorch_tanh"`}.
    Tanh,
}

/// GELU activation (Gaussian Error Linear Unit).
///
/// The bare name [`GELU`](struct@GELU) (used in value position, e.g.
/// `.through(GELU)`) resolves to the erf-form default and works
/// identically to the other unit-struct activations (`ReLU`, `SiLU`, …).
/// For the tanh approximation — HuggingFace `hidden_act="gelu_new"`,
/// ALBERT, GPT-2 — write [`GELU::tanh()`]. The explicit erf form is
/// [`GELU::exact()`] for symmetry, and [`GELU::with_approximate`]
/// accepts a runtime [`GeluApprox`] (used by `flodl-hf` config loaders).
///
/// ```ignore
/// use flodl::nn::{GELU, GeluApprox};
///
/// let g0 = GELU;                              // erf form (default)
/// let g1 = GELU::exact();                     // same, explicit
/// let g2 = GELU::tanh();                      // tanh approximation
/// let g3 = GELU::with_approximate(GeluApprox::Tanh); // runtime-chosen form
/// ```
///
/// This is the canonical pattern in flodl for parametrising a module
/// while keeping the bare-name ergonomics of a unit struct: a `const`
/// of the type re-exports the default-constructed value under the bare
/// type name, so `.through(GELU)` resolves to the default variant while
/// opt-in constructors (`::tanh()`, …) cover the others.
pub struct GELU {
    approximate: GeluApprox,
}

impl Default for GELU {
    fn default() -> Self {
        Self::exact()
    }
}

impl GELU {
    /// Erf-form GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`. The default,
    /// matching PyTorch `nn.GELU()` and HuggingFace `hidden_act="gelu"`.
    pub const fn exact() -> Self {
        Self { approximate: GeluApprox::Exact }
    }

    /// Tanh-approximation GELU:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    /// Matches PyTorch `nn.GELU(approximate='tanh')` and HuggingFace
    /// `hidden_act` in {`"gelu_new"`, `"gelu_pytorch_tanh"`} — required
    /// for ALBERT, GPT-2, and derivative checkpoints.
    pub const fn tanh() -> Self {
        Self { approximate: GeluApprox::Tanh }
    }

    /// Build a GELU module with a runtime-chosen [`GeluApprox`].
    /// Mirrors PyTorch `nn.GELU(approximate='none' | 'tanh')`. Used by
    /// `flodl-hf` config loaders that map `hidden_act` strings to a
    /// [`GeluApprox`] value at load time.
    pub const fn with_approximate(approximate: GeluApprox) -> Self {
        Self { approximate }
    }

    /// Which approximation form this module dispatches.
    pub const fn approximate(&self) -> GeluApprox {
        self.approximate
    }
}

/// Default-constructed [`GELU`](struct@GELU) (erf form), exposed so the
/// bare type name works in value position: `.through(GELU)`.
///
/// See [`struct@GELU`] for the full pattern rationale.
#[allow(non_upper_case_globals)]
pub const GELU: GELU = GELU::exact();

impl Module for GELU {
    fn name(&self) -> &str { "gelu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        match self.approximate {
            GeluApprox::Exact => input.gelu(),
            GeluApprox::Tanh => input.gelu_tanh(),
        }
    }
}

/// Sigmoid Linear Unit (Swish): `x * sigmoid(x)`.
/// Self-gated activation with smooth gradient flow.
pub struct SiLU;

impl Default for SiLU {
    fn default() -> Self {
        SiLU
    }
}

impl SiLU {
    /// Create a SiLU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for SiLU {
    fn name(&self) -> &str { "silu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.silu()
    }
}

/// SwiGLU gate: split the last dimension in half and return
/// `value * silu(gate)`. Halves the width, so `[.., 2n] -> [.., n]`.
///
/// The gating half of a SwiGLU feed-forward block, not the whole block — it
/// carries no parameters, so it composes between the two projections a
/// transformer FFN already has:
///
/// ```text
/// Linear(d -> 2h)  ->  SwiGLU  ->  Linear(h -> d)
/// ```
///
/// Keeping the projections outside is what lets the same module serve both a
/// hand-written `forward` and a [`FlowBuilder`](crate::FlowBuilder) graph,
/// where it appears as one node rather than an exploded split-and-multiply
/// subgraph.
///
/// # Which half is the gate
///
/// **First half is the value, second half is the gate** — `chunk(2, -1)` then
/// `x[0] * silu(x[1])`. This follows OLMo and the fused-projection convention
/// generally: a single `ff_proj` emits `[value | gate]` in that order.
///
/// It matters. LLaMA-style implementations with *separate* `gate_proj` and
/// `up_proj` compute `silu(gate_proj(x)) * up_proj(x)`, which is the same
/// formula but reaches it from two tensors, so there is no ordering to get
/// wrong. Fused ones have to pick, and picking the other way silently trains
/// a different (still-plausible) network — the loss curve looks fine, the
/// weights are not interchangeable with the reference. If you are porting a
/// checkpoint, check the source's chunk order.
///
/// # Panics / errors
///
/// Errors when the last dimension is odd, since there is no way to split it
/// into a value and a gate.
///
/// ```
/// use flodl::nn::{Module, SwiGLU};
/// use flodl::autograd::Variable;
/// use flodl::tensor::{Device, Tensor};
///
/// // [1, 4] -> [1, 2]: value = [1, 2], gate = [3, 4].
/// let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], Device::CPU)?;
/// let y = SwiGLU.forward(&Variable::new(t, false))?;
/// assert_eq!(y.shape(), vec![1, 2]);
/// # Ok::<(), flodl::tensor::TensorError>(())
/// ```
pub struct SwiGLU;

impl Default for SwiGLU {
    fn default() -> Self {
        SwiGLU
    }
}

impl SwiGLU {
    /// Create a SwiGLU gating module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for SwiGLU {
    fn name(&self) -> &str { "swiglu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let shape = input.shape();
        let last = *shape.last().ok_or_else(|| {
            crate::tensor::TensorError::new("SwiGLU expects a tensor with at least one dimension")
        })?;
        if last % 2 != 0 {
            return Err(crate::tensor::TensorError::new(&format!(
                "SwiGLU needs an even last dimension to split into value and gate, got {last} \
                 (shape {shape:?}) — the projection feeding it should emit 2x the block width"
            )));
        }
        let halves = input.chunk(2, -1)?;
        halves[1].silu()?.mul(&halves[0])
    }
}

/// Leaky ReLU: `max(0, x) + negative_slope * min(0, x)`.
pub struct LeakyReLU {
    negative_slope: f64,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        LeakyReLU { negative_slope: 0.01 }
    }
}

impl LeakyReLU {
    /// Create a LeakyReLU with the given negative slope (default: 0.01).
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Module for LeakyReLU {
    fn name(&self) -> &str { "leaky_relu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.leaky_relu(self.negative_slope)
    }
}

/// ELU: `max(0, x) + min(0, alpha * (exp(x) - 1))`.
pub struct ELU {
    alpha: f64,
}

impl Default for ELU {
    fn default() -> Self {
        ELU { alpha: 1.0 }
    }
}

impl ELU {
    /// Create an ELU with the given alpha (default: 1.0).
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Module for ELU {
    fn name(&self) -> &str { "elu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.elu(self.alpha)
    }
}

/// Softplus: smooth approximation to ReLU.
/// `(1/beta) * log(1 + exp(beta * x))`, reverts to linear above `threshold`.
pub struct Softplus {
    beta: f64,
    threshold: f64,
}

impl Default for Softplus {
    fn default() -> Self {
        Softplus { beta: 1.0, threshold: 20.0 }
    }
}

impl Softplus {
    /// Create a Softplus with given beta and threshold (defaults: 1.0, 20.0).
    pub fn new(beta: f64, threshold: f64) -> Self {
        Self { beta, threshold }
    }
}

impl Module for Softplus {
    fn name(&self) -> &str { "softplus" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.softplus(self.beta, self.threshold)
    }
}

/// Mish: `x * tanh(softplus(x))`. Self-regularizing activation.
pub struct Mish;

impl Default for Mish {
    fn default() -> Self {
        Mish
    }
}

impl Mish {
    /// Create a Mish activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Mish {
    fn name(&self) -> &str { "mish" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.mish()
    }
}

/// Softmax along a dimension. Output sums to 1.
pub struct Softmax {
    dim: i32,
}

impl Softmax {
    /// Create a Softmax module along the given dimension.
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Module for Softmax {
    fn name(&self) -> &str { "softmax" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.softmax(self.dim)
    }
}

/// Log-softmax along a dimension. Numerically stable `log(softmax(x))`.
pub struct LogSoftmax {
    dim: i32,
}

impl LogSoftmax {
    /// Create a LogSoftmax module along the given dimension.
    pub fn new(dim: i32) -> Self {
        Self { dim }
    }
}

impl Module for LogSoftmax {
    fn name(&self) -> &str { "log_softmax" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.log_softmax(self.dim)
    }
}

/// Flatten dimensions into a single dimension.
/// Default: flattens all dims except batch (start=1, end=-1).
pub struct Flatten {
    start_dim: i32,
    end_dim: i32,
}

impl Default for Flatten {
    fn default() -> Self {
        Flatten { start_dim: 1, end_dim: -1 }
    }
}

impl Flatten {
    /// Create a Flatten module with custom start and end dimensions.
    pub fn new(start_dim: i32, end_dim: i32) -> Self {
        Self { start_dim, end_dim }
    }
}

impl Module for Flatten {
    fn name(&self) -> &str { "flatten" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.flatten(self.start_dim, self.end_dim)
    }
}

/// SELU: Self-Normalizing ELU.
/// `lambda * (max(0, x) + min(0, alpha * (exp(x) - 1)))` with fixed constants.
/// Designed for self-normalizing networks with `AlphaDropout`.
pub struct SELU;

impl Default for SELU {
    fn default() -> Self {
        SELU
    }
}

impl SELU {
    /// Create a SELU activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for SELU {
    fn name(&self) -> &str { "selu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.selu()
    }
}

/// Hardswish: `x * clamp(x + 3, 0, 6) / 6`.
/// Efficient approximation of Swish for mobile architectures (MobileNetV3).
pub struct Hardswish;

impl Default for Hardswish {
    fn default() -> Self {
        Hardswish
    }
}

impl Hardswish {
    /// Create a Hardswish activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Hardswish {
    fn name(&self) -> &str { "hardswish" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.hardswish()
    }
}

/// Hardsigmoid: `clamp(x + 3, 0, 6) / 6`.
/// Efficient piecewise-linear approximation of sigmoid.
pub struct Hardsigmoid;

impl Default for Hardsigmoid {
    fn default() -> Self {
        Hardsigmoid
    }
}

impl Hardsigmoid {
    /// Create a Hardsigmoid activation module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for Hardsigmoid {
    fn name(&self) -> &str { "hardsigmoid" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.hardsigmoid()
    }
}

/// PReLU: Parametric ReLU with learnable weight.
/// `max(0, x) + weight * min(0, x)` where weight is learned per-channel or shared.
pub struct PReLU {
    weight: Parameter,
}

impl PReLU {
    /// Create a PReLU with `num_parameters` learnable weights (1 for shared, C for per-channel).
    pub fn new(num_parameters: i64, device: crate::tensor::Device) -> Result<Self> {
        let init = crate::tensor::Tensor::full(&[num_parameters], 0.25, crate::tensor::TensorOptions {
            dtype: crate::tensor::DType::Float32,
            device,
        })?;
        Ok(Self {
            weight: Parameter::new(init, "weight"),
        })
    }

    /// Create a PReLU on the given device.
    pub fn on_device(num_parameters: i64, device: crate::tensor::Device) -> Result<Self> {
        Self::new(num_parameters, device)
    }
}

impl Module for PReLU {
    fn name(&self) -> &str { "prelu" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.prelu(&self.weight.variable)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device};

    /// Pins the ORDER, which is the part that fails silently. Value is the
    /// first half, gate the second: `x[0] * silu(x[1])`. Swapping them still
    /// trains and still converges — it is just a different network, and the
    /// weights stop being interchangeable with the reference implementation.
    #[test]
    fn test_swiglu_gates_with_the_second_half() {
        // value = [1, 2], gate = [3, 4]  ->  [1*silu(3), 2*silu(4)]
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], test_device()).unwrap();
        let y = SwiGLU.forward(&Variable::new(t, false)).unwrap();
        assert_eq!(y.shape(), vec![1, 2], "the last dim halves");
        let got = y.data().to_f32_vec().unwrap();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        assert!((got[0] - 1.0 * silu(3.0)).abs() < 1e-5, "got {got:?}");
        assert!((got[1] - 2.0 * silu(4.0)).abs() < 1e-5, "got {got:?}");
        // The swapped reading would be [3*silu(1), 4*silu(2)] — check we are
        // not accidentally that, since both are plausible-looking outputs.
        assert!((got[0] - 3.0 * silu(1.0)).abs() > 1e-3, "value/gate are swapped");
    }

    #[test]
    fn test_swiglu_rejects_an_odd_last_dim() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let err = SwiGLU
            .forward(&Variable::new(t, false))
            .expect_err("an odd width cannot split into value and gate");
        assert!(err.to_string().contains("even last dimension"), "{err}");
    }

    /// Gradients must reach BOTH halves — the value linearly, the gate through
    /// silu'. A split that detached either side would still forward correctly
    /// and quietly train half the projection.
    #[test]
    fn test_swiglu_backward_reaches_both_halves() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], test_device()).unwrap();
        let x = Variable::new(t, true);
        SwiGLU.forward(&x).unwrap().sum().unwrap().backward().unwrap();
        let g = x.grad().expect("input must receive a gradient").to_f32_vec().unwrap();
        assert_eq!(g.len(), 4);
        assert!(g[0].abs() > 1e-6 && g[1].abs() > 1e-6, "value half unreached: {g:?}");
        assert!(g[2].abs() > 1e-6 && g[3].abs() > 1e-6, "gate half unreached: {g:?}");
    }

    #[test]
    fn test_leaky_relu_module() {
        let m = LeakyReLU::new(0.2);
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - (-0.2)).abs() < 1e-5);
        assert!((y[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu_default() {
        let m = LeakyReLU::default();
        let t = Tensor::from_f32(&[-1.0], &[1], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - (-0.01)).abs() < 1e-5);
    }

    #[test]
    fn test_elu_module() {
        let m = ELU::default();
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!(y[0] < 0.0); // negative for negative input
        assert!((y[1] - 0.0).abs() < 1e-5);
        assert!((y[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softplus_module() {
        let m = Softplus::default();
        let t = Tensor::from_f32(&[0.0], &[1], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - std::f32::consts::LN_2).abs() < 1e-3);
    }

    #[test]
    fn test_mish_module() {
        let m = Mish::new();
        let t = Tensor::from_f32(&[0.0, 1.0], &[2], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - 0.0).abs() < 1e-5);
        assert!((y[1] - 0.8651).abs() < 1e-3);
    }

    #[test]
    fn test_softmax_module() {
        let m = Softmax::new(-1);
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_module() {
        let m = LogSoftmax::new(-1);
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        // log_softmax values should all be negative
        assert!(y.iter().all(|&v| v < 0.0));
    }

    #[test]
    fn test_flatten_module() {
        let m = Flatten::default();
        let t = Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 2], test_device(),
        ).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap();
        assert_eq!(y.data().shape(), vec![2, 4]); // batch dim preserved
    }

    #[test]
    fn test_flatten_all() {
        let m = Flatten::new(0, -1);
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap();
        assert_eq!(y.data().shape(), vec![4]);
    }

    #[test]
    fn test_selu_module() {
        let m = SELU::new();
        let t = Tensor::from_f32(&[-1.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        // SELU(0) = 0
        assert!((y[1] - 0.0).abs() < 1e-5);
        // SELU(1) = lambda * 1 ~ 1.0507
        assert!((y[2] - 1.0507).abs() < 1e-3);
        // SELU(-1) < 0
        assert!(y[0] < 0.0);
    }

    #[test]
    fn test_hardswish_module() {
        let m = Hardswish::new();
        let t = Tensor::from_f32(&[-4.0, 0.0, 4.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - 0.0).abs() < 1e-5); // x * 0 / 6 = 0 for x < -3
        assert!((y[1] - 0.0).abs() < 1e-5); // 0 * 3/6 = 0
        assert!((y[2] - 4.0).abs() < 1e-5); // x for x > 3
    }

    #[test]
    fn test_hardsigmoid_module() {
        let m = Hardsigmoid::new();
        let t = Tensor::from_f32(&[-4.0, 0.0, 4.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        assert!((y[0] - 0.0).abs() < 1e-5); // clamp to 0
        assert!((y[1] - 0.5).abs() < 1e-5); // (0+3)/6 = 0.5
        assert!((y[2] - 1.0).abs() < 1e-5); // clamp to 1
    }

    #[test]
    fn test_prelu_module() {
        let m = PReLU::new(1, test_device()).unwrap();
        let t = Tensor::from_f32(&[-2.0, 0.0, 1.0], &[3], test_device()).unwrap();
        let x = Variable::new(t, false);
        let y = m.forward(&x).unwrap().data().to_f32_vec().unwrap();
        // PReLU(-2) = 0.25 * -2 = -0.5 (default init is 0.25)
        assert!((y[0] - (-0.5)).abs() < 1e-5);
        assert!((y[1] - 0.0).abs() < 1e-5);
        assert!((y[2] - 1.0).abs() < 1e-5);
        // Has learnable parameters
        assert_eq!(m.parameters().len(), 1);
    }
}
