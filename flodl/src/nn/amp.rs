use std::ffi::c_void;
use std::io::{Read, Write};
use std::ptr;

use flodl_sys as ffi;

use crate::autograd::Variable;
use crate::tensor::{DType, Result, Tensor};

use super::checkpoint::{write_f64_le, read_f64_le, write_i64_le, read_i64_le};
use super::optim::Stateful;
use super::parameter::Parameter;

/// RAII guard that enables automatic mixed precision.
///
/// While active, eligible operations (matmul, conv, linear) dispatch to the
/// given reduced-precision dtype. Numerically sensitive operations (losses,
/// norms, softmax) stay in fp32 automatically.
///
/// Requires Tensor Core hardware (Volta/Turing/Ampere/Ada/Blackwell) for
/// actual speedup. On older GPUs or CPU, ops still run but without perf gain.
///
/// ```ignore
/// let _amp = AutocastGuard::new(DType::Float16);
/// let output = model.forward(&input)?;  // matmul runs in fp16
/// let loss = mse_loss(&output, &target)?;  // stays fp32
/// drop(_amp);
/// ```
pub struct AutocastGuard {
    guard: *mut c_void,
}

impl AutocastGuard {
    /// Enable CUDA autocast with the given dtype (typically `Float16` or `BFloat16`).
    pub fn new(dtype: DType) -> Self {
        let guard = unsafe {
            ffi::flodl_autocast_guard_new(ffi::FLODL_CUDA, dtype as i32)
        };
        AutocastGuard { guard }
    }

    /// Enable autocast for a specific device type with the given dtype.
    pub fn for_device(device_type: i32, dtype: DType) -> Self {
        let guard = unsafe {
            ffi::flodl_autocast_guard_new(device_type, dtype as i32)
        };
        AutocastGuard { guard }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        if !self.guard.is_null() {
            unsafe { ffi::flodl_autocast_guard_delete(self.guard) };
            self.guard = ptr::null_mut();
        }
    }
}

/// Run a closure with CUDA autocast enabled.
///
/// Equivalent to creating an [`AutocastGuard`] for the duration of `f`.
///
/// ```ignore
/// let loss = autocast(DType::Float16, || {
///     let output = model.forward(&input)?;
///     mse_loss(&output, &target)
/// })?;
/// ```
pub fn autocast<F, R>(dtype: DType, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = AutocastGuard::new(dtype);
    f()
}

/// Returns true if CUDA autocast is currently enabled.
pub fn is_autocast_enabled() -> bool {
    unsafe { ffi::flodl_is_autocast_enabled(ffi::FLODL_CUDA) != 0 }
}

/// Cast all parameters to a different dtype.
///
/// No-op for parameters already at the target dtype.
pub fn cast_parameters(params: &[Parameter], dtype: DType) {
    for p in params {
        if p.variable.data().dtype() != dtype
            && let Ok(t) = p.variable.data().to_dtype(dtype)
        {
            p.variable.set_data(t);
        }
    }
}

/// GradScaler for mixed precision training.
///
/// Scales loss before backward to prevent gradient underflow in float16,
/// then unscales gradients before optimizer step. Dynamically adjusts
/// scale factor based on whether inf/nan gradients are detected.
///
/// ```ignore
/// let mut scaler = GradScaler::new();
/// let scaled_loss = scaler.scale(&loss)?;
/// scaled_loss.backward()?;
/// let stepped = scaler.step(&params, &mut || optim.step())?;
/// scaler.update();
/// ```
pub struct GradScaler {
    scale: f64,
    growth: f64,
    backoff: f64,
    interval: i64,
    steps_since_growth: i64,
    found_inf: bool,
}

impl Default for GradScaler {
    fn default() -> Self {
        GradScaler {
            scale: 65536.0,
            growth: 2.0,
            backoff: 0.5,
            interval: 2000,
            steps_since_growth: 0,
            found_inf: false,
        }
    }
}

impl GradScaler {
    /// Create a new GradScaler with default settings.
    ///
    /// Initial scale: 2^16 = 65536, growth: 2.0, backoff: 0.5, interval: 2000.
    pub fn new() -> Self {
        Self::default()
    }

    /// Scale the loss before backward. Returns loss * scale.
    pub fn scale(&self, loss: &Variable) -> Result<Variable> {
        loss.mul_scalar(self.scale)
    }

    /// Current scale factor.
    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    /// Unscale gradients, check for inf/nan, and step the optimizer.
    ///
    /// Returns true if the step was taken (all gradients finite).
    /// Returns false if inf/nan detected (optimizer step skipped).
    pub fn step(&mut self, params: &[Parameter], step_fn: &mut dyn FnMut() -> Result<()>) -> Result<bool> {
        let inv_scale = 1.0 / self.scale;

        // Unscale and check all gradients
        let mut unscaled_grads: Vec<Option<Tensor>> = Vec::with_capacity(params.len());
        for p in params {
            if let Some(grad) = p.variable.grad() {
                let unscaled = grad.mul_scalar(inv_scale)?;
                if !unscaled.all_finite()? {
                    self.found_inf = true;
                    return Ok(false);
                }
                unscaled_grads.push(Some(unscaled));
            } else {
                unscaled_grads.push(None);
            }
        }

        // Replace gradients with unscaled versions
        for (p, ug) in params.iter().zip(unscaled_grads) {
            if let Some(g) = ug {
                p.variable.set_grad(g);
            }
        }

        // Step the optimizer
        step_fn()?;
        Ok(true)
    }

    /// Update the scale factor after each step.
    ///
    /// Call this after every `step()` call, regardless of whether it succeeded.
    pub fn update(&mut self) {
        if self.found_inf {
            self.scale *= self.backoff;
            self.steps_since_growth = 0;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.interval {
                self.scale *= self.growth;
                self.steps_since_growth = 0;
            }
        }
        self.found_inf = false;
    }
}

impl Stateful for GradScaler {
    fn state_kind(&self) -> crate::nn::StateKind { crate::nn::StateKind::GradScaler }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_f64_le(w, self.scale)?;
        write_i64_le(w, self.steps_since_growth)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.scale = read_f64_le(r)?;
        self.steps_since_growth = read_i64_le(r)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, test_device, test_opts};

    #[test]
    fn test_autocast_guard_lifecycle() {
        assert!(!is_autocast_enabled());
        {
            let _guard = AutocastGuard::new(DType::Float16);
            assert!(is_autocast_enabled());
        }
        // Guard dropped, autocast should be disabled
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_autocast_closure() {
        assert!(!is_autocast_enabled());
        let was_enabled = autocast(DType::Float16, || {
            is_autocast_enabled()
        });
        assert!(was_enabled);
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_cast_parameters() {
        let t = Tensor::ones(&[4], test_opts()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        assert_eq!(p.variable.data().dtype(), DType::Float32);
        cast_parameters(std::slice::from_ref(&p), DType::Float64);
        assert_eq!(p.variable.data().dtype(), DType::Float64);
    }

    #[test]
    fn test_cast_parameters_noop_same_dtype() {
        let t = Tensor::ones(&[4], test_opts()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        cast_parameters(std::slice::from_ref(&p), DType::Float32);
        assert_eq!(p.variable.data().dtype(), DType::Float32);
    }

    #[test]
    fn test_grad_scaler_defaults() {
        let scaler = GradScaler::new();
        assert_eq!(scaler.scale_factor(), 65536.0);
    }

    #[test]
    fn test_grad_scaler_scale() {
        let scaler = GradScaler::new();
        let loss = Variable::new(
            Tensor::from_f32(&[1.0], &[1], test_device()).unwrap(), true,
        );
        let scaled = scaler.scale(&loss).unwrap();
        assert!((scaled.item().unwrap() - 65536.0).abs() < 1.0);
    }

    #[test]
    fn test_grad_scaler_step_finite() {
        let mut scaler = GradScaler::new();
        let t = Tensor::from_f32(&[1.0, 2.0], &[2], test_device()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        // Set a finite gradient
        let grad = Tensor::from_f32(&[0.1, 0.2], &[2], test_device()).unwrap();
        p.variable.set_grad(grad);

        let mut stepped = false;
        let ok = scaler.step(std::slice::from_ref(&p), &mut || { stepped = true; Ok(()) }).unwrap();
        assert!(ok);
        assert!(stepped);
    }

    #[test]
    fn test_grad_scaler_step_inf() {
        let mut scaler = GradScaler::new();
        let t = Tensor::from_f32(&[1.0], &[1], test_device()).unwrap();
        let p = Parameter {
            variable: Variable::new(t, true),
            name: "w".into(),
        };
        // Set an infinite gradient
        let grad = Tensor::from_f32(&[f32::INFINITY], &[1], test_device()).unwrap();
        p.variable.set_grad(grad);

        let mut stepped = false;
        let ok = scaler.step(&[p], &mut || { stepped = true; Ok(()) }).unwrap();
        assert!(!ok, "step should be skipped on inf");
        assert!(!stepped, "optimizer should not have stepped");
    }

    #[test]
    fn test_grad_scaler_update_growth() {
        let mut scaler = GradScaler {
            scale: 100.0,
            growth: 2.0,
            backoff: 0.5,
            interval: 3,
            steps_since_growth: 2,
            found_inf: false,
        };
        scaler.update(); // steps_since_growth becomes 3 >= interval
        assert_eq!(scaler.scale_factor(), 200.0);
    }

    #[test]
    fn test_grad_scaler_update_backoff() {
        let mut scaler = GradScaler::new();
        let initial = scaler.scale_factor();
        scaler.found_inf = true;
        scaler.update();
        assert_eq!(scaler.scale_factor(), initial * 0.5);
    }

    #[test]
    fn test_grad_scaler_state_roundtrip() {
        let mut scaler = GradScaler::new();
        // Modify state
        scaler.found_inf = true;
        scaler.update(); // backoff
        scaler.update(); // one good step

        let mut buf = Vec::new();
        scaler.save_state(&mut buf).unwrap();

        let mut loaded = GradScaler::new();
        loaded.load_state(&mut &buf[..]).unwrap();
        assert_eq!(loaded.scale_factor(), scaler.scale_factor());
    }
}
