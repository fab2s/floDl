use crate::autograd::{self, Variable};
use crate::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::Module;
use super::parameter::Parameter;

/// Single-step LSTM cell backed by fused ATen `lstm_cell` kernel.
///
/// Uses 4 packed parameters (`w_ih`, `w_hh`, `b_ih`, `b_hh`) instead
/// of 8 separate Linear modules, reducing ~40 kernel launches to ~2.
///
/// State is packed as `cat(h, c, dim=1)` with shape `[batch, 2*hidden_size]`.
/// This allows single-Variable passthrough with graph forward refs.
///
/// Standalone usage: call `forward_step(x, state)` directly.
///
/// ```ignore
/// let lstm = LSTMCell::new(4, 3)?; // input_size=4, hidden_size=3
/// let x = Variable::new(Tensor::randn(&[1, 4], opts)?, false);
/// let state = lstm.forward_step(&x, None)?;  // [1, 6] (h and c packed)
/// let h = state.narrow(1, 0, 3)?;            // [1, 3] just h
/// ```
pub struct LSTMCell {
    w_ih: Parameter,
    w_hh: Parameter,
    b_ih: Parameter,
    b_hh: Parameter,
    hidden_size: i64,
}

impl LSTMCell {
    /// Create an LSTM cell with given input and hidden dimensions.
    /// Parameters initialized with PyTorch-style uniform(-1/sqrt(hs), 1/sqrt(hs)).
    pub fn new(input_size: i64, hidden_size: i64) -> Result<Self> {
        Self::on_device(input_size, hidden_size, Device::CPU)
    }

    /// Create an LSTM cell on a specific device.
    pub fn on_device(input_size: i64, hidden_size: i64, device: Device) -> Result<Self> {
        let bound = 1.0 / (hidden_size as f64).sqrt();
        let opts = TensorOptions {
            dtype: DType::Float32,
            device,
        };

        let w_ih = Tensor::rand(&[4 * hidden_size, input_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let w_hh = Tensor::rand(&[4 * hidden_size, hidden_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let b_ih = Tensor::rand(&[4 * hidden_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;
        let b_hh = Tensor::rand(&[4 * hidden_size], opts)?
            .mul_scalar(2.0 * bound)?
            .add_scalar(-bound)?;

        Ok(LSTMCell {
            w_ih: Parameter::new(w_ih, "w_ih"),
            w_hh: Parameter::new(w_hh, "w_hh"),
            b_ih: Parameter::new(b_ih, "b_ih"),
            b_hh: Parameter::new(b_hh, "b_hh"),
            hidden_size,
        })
    }

    /// Single LSTM step. `state` is packed `cat(h, c, dim=1)` or `None` for zeros.
    /// Returns packed `cat(h', c', dim=1)` with shape `[batch, 2*hidden_size]`.
    pub fn forward_step(&self, x: &Variable, state: Option<&Variable>) -> Result<Variable> {
        let batch = x.shape()[0];
        let hs = self.hidden_size;

        let (h, c) = match state {
            Some(s) => {
                let h = s.narrow(1, 0, hs)?;
                let c = s.narrow(1, hs, hs)?;
                (h, c)
            }
            None => {
                let opts = TensorOptions {
                    dtype: DType::Float32,
                    device: self.w_ih.variable.device(),
                };
                let h = Variable::new(Tensor::zeros(&[batch, hs], opts)?, false);
                let c = Variable::new(Tensor::zeros(&[batch, hs], opts)?, false);
                (h, c)
            }
        };

        let (h_new, c_new) = autograd::lstm_cell(
            x,
            &h,
            &c,
            &self.w_ih.variable,
            &self.w_hh.variable,
            &self.b_ih.variable,
            &self.b_hh.variable,
        )?;

        // Pack state: cat(h', c', dim=1)
        h_new.cat(&c_new, 1)
    }
}

impl Module for LSTMCell {
    fn name(&self) -> &str {
        "lstmcell"
    }

    /// Module trait forward: returns only h `[batch, hidden_size]` (no cell state).
    /// Use [`forward_step`](LSTMCell::forward_step) for full packed state access.
    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Extract just h from the packed state output
        let state = self.forward_step(input, None)?;
        state.narrow(1, 0, self.hidden_size)
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
    fn test_lstmcell_forward() {
        let lstm = LSTMCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let state = lstm.forward_step(&x, None).unwrap();
        assert_eq!(state.shape(), vec![2, 6]); // h + c packed

        // Extract h and c
        let h = state.narrow(1, 0, 3).unwrap();
        let c = state.narrow(1, 3, 3).unwrap();
        assert_eq!(h.shape(), vec![2, 3]);
        assert_eq!(c.shape(), vec![2, 3]);

        // Second step with state
        let state2 = lstm.forward_step(&x, Some(&state)).unwrap();
        assert_eq!(state2.shape(), vec![2, 6]);
    }

    #[test]
    fn test_lstmcell_gradient() {
        let lstm = LSTMCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            true,
        );
        let state = lstm.forward_step(&x, None).unwrap();
        let loss = state.sum().unwrap();
        loss.backward().unwrap();

        // All parameters should have gradients
        for p in lstm.parameters() {
            assert!(p.variable.grad().is_some(), "missing grad for {}", p.name);
        }
        // Input should have gradient
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_lstmcell_multi_step() {
        let lstm = LSTMCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );

        // Run 5 steps, each feeding previous state
        let mut state: Option<Variable> = None;
        for _ in 0..5 {
            let s = lstm.forward_step(&x, state.as_ref()).unwrap();
            state = Some(s);
        }
        let final_state = state.unwrap();
        assert_eq!(final_state.shape(), vec![2, 6]);
    }

    #[test]
    fn test_lstmcell_finite_difference() {
        let eps = 1e-3;
        let lstm = LSTMCell::on_device(2, 2, crate::tensor::test_device()).unwrap();
        let x_data = vec![0.5_f32, -0.3, 0.1, 0.8];
        let x = Variable::new(from_f32(&x_data, &[2, 2]), true);

        let state = lstm.forward_step(&x, None).unwrap();
        let loss = state.sum().unwrap();
        loss.backward().unwrap();
        let analytical = x.grad().unwrap().to_f32_vec().unwrap();

        for i in 0..4 {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps as f32;
            xm[i] -= eps as f32;

            let xp_var = Variable::new(from_f32(&xp, &[2, 2]), false);
            let xm_var = Variable::new(from_f32(&xm, &[2, 2]), false);
            let fp: f64 = lstm
                .forward_step(&xp_var, None)
                .unwrap()
                .sum()
                .unwrap()
                .item()
                .unwrap();
            let fm: f64 = lstm
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
    fn test_lstmcell_module_forward() {
        use crate::nn::Module;
        let lstm = LSTMCell::on_device(4, 3, crate::tensor::test_device()).unwrap();
        let x = Variable::new(
            Tensor::randn(&[2, 4], crate::tensor::test_opts()).unwrap(),
            false,
        );
        let y = lstm.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![2, 3]); // Only h returned from Module::forward
    }
}
