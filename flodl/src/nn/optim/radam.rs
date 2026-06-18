//! RAdam (Rectified Adam) optimizer.

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::parameter::Parameter;

use super::Optimizer;

/// RAdam optimizer (Liu et al., 2020).
///
/// Rectified Adam: uses a variance-rectification term to automatically
/// switch between SGD-like updates (early training) and Adam updates.
/// No warmup scheduler needed.
pub struct RAdam {
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

impl RAdam {
    /// Create a new RAdam optimizer with default betas (0.9, 0.999), eps (1e-8),
    /// and no weight decay.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        RAdam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0,
            m: vec![None; n], v: vec![None; n], step_count: 0,
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 { self.lr }
}

impl Optimizer for RAdam {
    fn lr(&self) -> f64 { self.lr }
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let b1t = b1.powf(t);
        let b2t = b2.powf(t);
        // Maximum length of approximated SMA
        let rho_inf = 2.0 / (1.0 - b2) - 1.0;
        let rho_t = rho_inf - 2.0 * t * b2t / (1.0 - b2t);

        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    let data = param.data().detach()?;
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }

                    // Update biased first moment
                    let m_new = match self.m[i].take() {
                        Some(m) => m.mul_scalar(b1)?.add(&grad.mul_scalar(1.0 - b1)?)?,
                        None => grad.mul_scalar(1.0 - b1)?,
                    };
                    // Update biased second moment
                    let grad2 = grad.mul(&grad)?;
                    let v_new = match self.v[i].take() {
                        Some(v) => v.mul_scalar(b2)?.add(&grad2.mul_scalar(1.0 - b2)?)?,
                        None => grad2.mul_scalar(1.0 - b2)?,
                    };

                    let m_hat = m_new.mul_scalar(1.0 / (1.0 - b1t))?;

                    if rho_t > 5.0 {
                        // Variance is tractable: use Adam-like update
                        let v_hat = v_new.mul_scalar(1.0 / (1.0 - b2t))?;
                        let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                                    ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                        let update = m_hat.div(&v_hat.sqrt()?.add_scalar(self.eps)?)?.mul_scalar(self.lr * rect)?;
                        data.sub_(&update)?;
                    } else {
                        // Variance is intractable: use SGD-like update
                        let update = m_hat.mul_scalar(self.lr)?;
                        data.sub_(&update)?;
                    }
                    self.m[i] = Some(m_new);
                    self.v[i] = Some(v_new);
                }
            }
            Ok(())
        })
    }

    fn reset_state(&mut self) {
        // Moment estimates back to fresh, step counter to 0 (rectification
        // schedule restarts). Lengths preserved for per-param indexing.
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
    fn test_radam_steps() {
        let p = make_param("w", &[1]);
        let before = p.variable.data().item().unwrap();
        let mut opt = RAdam::new(std::slice::from_ref(&p), 0.01);
        let x = Variable::new(
            Tensor::from_f32(&[2.0], &[1], crate::tensor::test_device()).unwrap(), false,
        );
        let loss = x.mul(&p.variable).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().item().unwrap();
        assert!((after - before).abs() > 1e-6, "RAdam step should change parameter");
    }

    #[test]
    fn test_radam_convergence_100_steps() {
        use crate::nn::{Linear, Module, loss::mse_loss};

        let dev = crate::tensor::test_device();
        let model = Linear::on_device(4, 1, dev).unwrap();
        // RAdam uses SGD-like updates for early steps (rho_t <= 5), so needs
        // more iterations and a slightly higher LR than vanilla Adam.
        let mut opt = RAdam::new(&model.parameters(), 0.05);

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
            "RAdam should converge: first={}, final={}", first_loss, final_loss);
    }
}
