//! NAdam (Nesterov-accelerated Adam) optimizer.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    write_tensor_state, read_tensor_state, write_f64_le, read_f64_le,
    write_u32_le, read_u32_le, write_i64_le, read_i64_le,
};
use crate::nn::parameter::Parameter;

use super::{Optimizer, Stateful};

/// NAdam optimizer (Dozat, 2016).
///
/// Incorporates Nesterov momentum into Adam. Equivalent to Adam with
/// a look-ahead gradient, providing faster convergence on some tasks.
///
/// Update rule:
///   m = beta1 * m + (1 - beta1) * grad
///   v = beta2 * v + (1 - beta2) * grad^2
///   m_hat = beta1 * m / (1 - beta1^(t+1)) + (1 - beta1) * grad / (1 - beta1^t)
///   param -= lr * m_hat / (sqrt(v / (1 - beta2^t)) + eps)
pub struct NAdam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    /// Per-param step counts, incremented only when the param has a grad
    /// (the Nesterov bias-correction schedule is per-param).
    steps: Vec<i64>,
}

impl NAdam {
    /// Create a new NAdam optimizer with default betas (0.9, 0.999), eps (1e-8),
    /// and no weight decay.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        NAdam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0,
            m: vec![None; n], v: vec![None; n], steps: vec![0; n],
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 { self.lr }
}

impl Optimizer for NAdam {
    fn lr(&self) -> f64 { self.lr }
    fn step(&mut self) -> Result<()> {
        let b1 = self.beta1;
        let b2 = self.beta2;

        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    // Per-param step: the Nesterov bias-correction schedule
                    // restarts for a late-unfrozen param.
                    self.steps[i] += 1;
                    let t = self.steps[i] as f64;
                    let b1t = b1.powf(t);
                    let b2t = b2.powf(t);
                    let b1t1 = b1.powf(t + 1.0);
                    let data = param.data().detach()?;
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }

                    let m_new = match self.m[i].take() {
                        Some(m) => m.mul_scalar(b1)?.add(&grad.mul_scalar(1.0 - b1)?)?,
                        None => grad.mul_scalar(1.0 - b1)?,
                    };
                    let grad2 = grad.mul(&grad)?;
                    let v_new = match self.v[i].take() {
                        Some(v) => v.mul_scalar(b2)?.add(&grad2.mul_scalar(1.0 - b2)?)?,
                        None => grad2.mul_scalar(1.0 - b2)?,
                    };

                    // Nesterov-corrected first moment
                    let m_hat = m_new.mul_scalar(b1 / (1.0 - b1t1))?
                        .add(&grad.mul_scalar((1.0 - b1) / (1.0 - b1t))?)?;
                    let v_hat = v_new.mul_scalar(1.0 / (1.0 - b2t))?;

                    let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(self.lr)?;
                    data.sub_(&update)?;

                    self.m[i] = Some(m_new);
                    self.v[i] = Some(v_new);
                }
            }
            Ok(())
        })
    }

    fn reset_state(&mut self) {
        // Moment estimates back to fresh, step counts to 0 (Nesterov
        // momentum schedule restarts). Lengths preserved for per-param indexing.
        for slot in &mut self.m {
            *slot = None;
        }
        for slot in &mut self.v {
            *slot = None;
        }
        for s in &mut self.steps {
            *s = 0;
        }
    }

    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad_set_to_none(); }
    }

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }

    fn save_state_to(&self, path: &str) -> Result<()> {
        <Self as Stateful>::save_state_file(self, path)
    }
}

impl Stateful for NAdam {
    fn state_kind(&self) -> super::StateKind { super::StateKind::NAdam }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_f64_le(w, self.beta1)?;
        write_f64_le(w, self.beta2)?;
        write_f64_le(w, self.eps)?;
        write_f64_le(w, self.weight_decay)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.m[i].as_ref())?;
            write_tensor_state(w, self.v[i].as_ref())?;
            write_i64_le(w, self.steps[i])?;
        }
        // Empty group table: NAdam has no group support yet, but the
        // slot keeps the payload shape uniform with grouped optimizers.
        super::write_groups(w, &[])?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "NAdam: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.beta1 = read_f64_le(r)?;
        self.beta2 = read_f64_le(r)?;
        self.eps = read_f64_le(r)?;
        self.weight_decay = read_f64_le(r)?;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.m[i] = read_tensor_state(r, dev)?;
            self.v[i] = read_tensor_state(r, dev)?;
            self.steps[i] = read_i64_le(r)?;
        }
        let groups = super::read_groups(r, self.params.len(), "NAdam")?;
        if !groups.is_empty() {
            return Err(crate::tensor::TensorError::new(
                "NAdam: state file carries a group table, but this flodl's \
                 NAdam has no parameter-group support",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_helpers::make_param;
    use crate::tensor::Tensor;

    #[test]
    fn test_nadam_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = NAdam::new(std::slice::from_ref(&p), 0.01);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(), false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        assert!((after - before).abs() > 1e-6, "NAdam step should change parameter");
    }

    #[test]
    fn test_nadam_convergence_100_steps() {
        use crate::nn::{Linear, Module, loss::mse_loss};

        let dev = crate::tensor::test_device();
        let model = Linear::on_device(4, 1, dev).unwrap();
        let mut opt = NAdam::new(&model.parameters(), 0.01);

        let x = Variable::new(
            Tensor::from_f32(
                &[1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0],
                &[4, 4], dev,
            ).unwrap(),
            false,
        );
        let target = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1], dev).unwrap(),
            false,
        );

        let first_loss;
        {
            let pred = model.forward(&x).unwrap();
            first_loss = mse_loss(&pred, &target).unwrap().item().unwrap();
        }

        for _ in 0..100 {
            opt.zero_grad();
            let pred = model.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        let pred = model.forward(&x).unwrap();
        let final_loss = mse_loss(&pred, &target).unwrap().item().unwrap();
        assert!(final_loss < first_loss * 0.5,
            "NAdam should converge: first={}, final={}", first_loss, final_loss);
    }
}
