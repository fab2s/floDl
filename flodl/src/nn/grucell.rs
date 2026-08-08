use crate::autograd::{self, Variable};
use crate::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::Module;
use super::parameter::Parameter;

/// Single-step GRU cell backed by fused ATen `gru_cell` kernel.
///
/// Uses 4 packed parameters (`w_ih`, `w_hh`, `b_ih`, `b_hh`) instead
/// of 6 separate Linear modules, reducing ~25 kernel launches to ~2.
///
/// Takes `(x, h)` where h is optional (None -> zeros).
/// For use with graph forward refs: input = x, ref "hidden" = h.
///
/// Standalone usage: call `forward_step(x, h)` directly.
///
/// ```ignore
/// let gru = GRUCell::new(4, 3)?; // input_size=4, hidden_size=3
/// let x = Variable::new(Tensor::randn(&[1, 4], opts)?, false);
/// let h1 = gru.forward_step(&x, None)?;     // [1, 3]
/// let h2 = gru.forward_step(&x, Some(&h1))?; // [1, 3]
/// ```
pub struct GRUCell {
    w_ih: Parameter,
    w_hh: Parameter,
    b_ih: Parameter,
    b_hh: Parameter,
    hidden_size: i64,
}

impl GRUCell {
    /// Create a GRU cell with given input and hidden dimensions.
    /// All parameters initialized with PyTorch-style uniform(-1/sqrt(hs), 1/sqrt(hs)).
    pub fn new(input_size: i64, hidden_size: i64) -> Result<Self> {
        Self::on_device(input_size, hidden_size, Device::CPU)
    }

    /// Create a GRU cell on a specific device.
    pub fn on_device(input_size: i64, hidden_size: i64, device: Device) -> Result<Self> {
        let bound = 1.0 / (hidden_size as f64).sqrt();
        let opts = TensorOptions {
            dtype: DType::Float32,
            device,
        };

        let w_ih = Tensor::rand(&[3 * hidden_size, input_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let w_hh = Tensor::rand(&[3 * hidden_size, hidden_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let b_ih = Tensor::rand(&[3 * hidden_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let b_hh = Tensor::rand(&[3 * hidden_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;

        Ok(GRUCell {
            w_ih: Parameter::new(w_ih, "w_ih"),
            w_hh: Parameter::new(w_hh, "w_hh"),
            b_ih: Parameter::new(b_ih, "b_ih"),
            b_hh: Parameter::new(b_hh, "b_hh"),
            hidden_size,
        })
    }

    /// Single GRU step: returns new hidden state `[batch, hidden_size]`.
    /// Pass `None` for `h` to zero-initialize the hidden state.
    pub fn forward_step(&self, x: &Variable, h: Option<&Variable>) -> Result<Variable> {
        let batch = x.shape()[0];

        let h = match h {
            Some(h) => h.clone(),
            None => {
                let opts = TensorOptions {
                    dtype: DType::Float32,
                    device: self.w_ih.variable.device(),
                };
                Variable::new(Tensor::zeros(&[batch, self.hidden_size], opts)?, false)
            }
        };

        autograd::gru_cell(
            x,
            &h,
            &self.w_ih.variable,
            &self.w_hh.variable,
            &self.b_ih.variable,
            &self.b_hh.variable,
        )
    }
}

impl Module for GRUCell {
    fn name(&self) -> &str {
        "grucell"
    }

    /// Module trait forward: runs a single step with zero-initialized hidden state.
    /// Does not accept or return hidden state -- use [`forward_step`](GRUCell::forward_step)
    /// for explicit hidden state control in recurrent loops.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_step(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            self.w_ih.clone(),
            self.w_hh.clone(),
            self.b_ih.clone(),
            self.b_hh.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, crate::tensor::test_device()).unwrap()
    }

    #[test]
    fn test_grucell_forward() {
        let gru = GRUCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let h = gru.forward_step(&x, None).unwrap();
        assert_eq!(h.shape(), vec![2, 3]);

        // Second step with hidden state
        let h2 = gru.forward_step(&x, Some(&h)).unwrap();
        assert_eq!(h2.shape(), vec![2, 3]);
    }

    #[test]
    fn test_grucell_gradient() {
        let gru = GRUCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let h = gru.forward_step(&x, None).unwrap();
        let loss = h.sum().unwrap();
        loss.backward().unwrap();

        // All parameters should have gradients
        for p in gru.parameters() {
            assert!(p.variable.grad().is_some(), "missing grad for {}", p.name);
        }
        // Input should have gradient
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_grucell_multi_step() {
        let gru = GRUCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );

        // Run 5 steps, each feeding previous hidden state
        let mut h: Option<Variable> = None;
        for _ in 0..5 {
            let h_new = gru.forward_step(&x, h.as_ref()).unwrap();
            h = Some(h_new);
        }
        assert_eq!(h.unwrap().shape(), vec![2, 3]);
    }

    #[test]
    fn test_grucell_finite_difference() {
        let eps = 1e-3;
        let gru = GRUCell::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let x_data = vec![0.5_f32, -0.3, 0.1, 0.8];
        let x = Variable::new(from_f32(&x_data, &[2, 2]), true);

        let h = gru.forward_step(&x, None).unwrap();
        let loss = h.sum().unwrap();
        loss.backward().unwrap();
        let analytical = x.grad().unwrap().to_f32_vec().unwrap();

        for i in 0..4 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;

            let xp_var = Variable::new(from_f32(&xp, &[2, 2]), false);
            let xm_var = Variable::new(from_f32(&xm, &[2, 2]), false);
            let fp: f64 = gru
                .forward_step(&xp_var, None)
                .unwrap()
                .sum()
                .unwrap()
                .item()
                .unwrap();
            let fm: f64 = gru
                .forward_step(&xm_var, None)
                .unwrap()
                .sum()
                .unwrap()
                .item()
                .unwrap();
            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (analytical[i] as f64 - numerical).abs() < 0.05,
                "grad mismatch at {}: analytical={}, numerical={}",
                i,
                analytical[i],
                numerical
            );
        }
    }

    #[test]
    fn test_grucell_module_forward() {
        use crate::nn::Module;
        let gru = GRUCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let y = gru.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3]);
    }
}
