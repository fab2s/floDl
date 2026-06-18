//! NAdam (Nesterov-accelerated Adam) optimizer.

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::parameter::Parameter;

use super::Optimizer;

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
    step_count: u64,
}

impl NAdam {
    /// Create a new NAdam optimizer with default betas (0.9, 0.999), eps (1e-8),
    /// and no weight decay.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        NAdam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0,
            m: vec![None; n], v: vec![None; n], step_count: 0,
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 { self.lr }
}

impl Optimizer for NAdam {
    fn lr(&self) -> f64 { self.lr }
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let b1t = b1.powf(t);
        let b2t = b2.powf(t);
        let b1t1 = b1.powf(t + 1.0);

        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
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
        // Moment estimates back to fresh, step counter to 0 (Nesterov
        // momentum schedule restarts). Lengths preserved for per-param indexing.
        for slot in &mut self.m {
            *slot = None;
        }
        for slot in &mut self.v {
            *slot = None;
        }
        self.step_count = 0;
    }

    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad_set_to_none(); }
    }

    fn set_lr(&mut self, lr: f64) { self.lr = lr; }
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
