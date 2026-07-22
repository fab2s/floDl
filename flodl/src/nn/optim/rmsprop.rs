//! RMSprop optimizer.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    write_tensor_state, read_tensor_state, write_f64_le, read_f64_le,
    write_u32_le, read_u32_le,
};
use crate::nn::parameter::Parameter;

use super::{GroupMeta, Optimizer, Stateful};

/// RMSprop optimizer (Hinton, 2012).
///
/// Maintains a running average of squared gradients to normalize the update.
/// Optionally supports momentum and weight decay.
///
/// Update rule (without momentum):
///   v = alpha * v + (1 - alpha) * grad^2
///   param -= lr * grad / (sqrt(v) + eps)
///
/// With momentum:
///   v = alpha * v + (1 - alpha) * grad^2
///   buf = momentum * buf + grad / (sqrt(v) + eps)
///   param -= lr * buf
///
/// ```ignore
/// let mut optim = RMSprop::new(&model.parameters(), 0.01);
/// // Or with options:
/// let mut optim = RMSprop::builder(&model.parameters(), 0.01)
///     .alpha(0.99)
///     .momentum(0.9)
///     .weight_decay(1e-4)
///     .build();
/// ```
pub struct RMSprop {
    params: Vec<Variable>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
    /// Running average of squared gradients
    v: Vec<Option<crate::tensor::Tensor>>,
    /// Momentum buffer (only used when momentum > 0)
    buf: Vec<Option<crate::tensor::Tensor>>,
    groups: Vec<GroupMeta>,
}

impl RMSprop {
    /// Create a new RMSprop optimizer with default parameters:
    /// alpha=0.99, eps=1e-8, weight_decay=0, momentum=0.
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        RMSprop {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            v: vec![None; n],
            buf: vec![None; n],
            groups: vec![],
        }
    }

    /// Create a builder for RMSprop with customizable options.
    pub fn builder(params: &[Parameter], lr: f64) -> RMSpropBuilder {
        RMSpropBuilder {
            params: params.to_vec(),
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
        }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    fn lr_for_param(&self, i: usize) -> f64 {
        for g in &self.groups {
            if g.range.contains(&i) {
                return g.lr;
            }
        }
        self.lr
    }
}

/// Builder for RMSprop with customizable options.
pub struct RMSpropBuilder {
    params: Vec<Parameter>,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
    momentum: f64,
}

impl RMSpropBuilder {
    /// Smoothing constant (default 0.99).
    pub fn alpha(mut self, alpha: f64) -> Self { self.alpha = alpha; self }
    /// Term added for numerical stability (default 1e-8).
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }
    /// Weight decay (L2 penalty, default 0).
    pub fn weight_decay(mut self, wd: f64) -> Self { self.weight_decay = wd; self }
    /// Momentum factor (default 0).
    pub fn momentum(mut self, momentum: f64) -> Self { self.momentum = momentum; self }

    /// Build the RMSprop optimizer.
    pub fn build(self) -> RMSprop {
        let n = self.params.len();
        RMSprop {
            params: self.params.iter().map(|p| p.variable.clone()).collect(),
            lr: self.lr,
            alpha: self.alpha,
            eps: self.eps,
            weight_decay: self.weight_decay,
            momentum: self.momentum,
            v: vec![None; n],
            buf: vec![None; n],
            groups: vec![],
        }
    }
}

impl Optimizer for RMSprop {
    fn lr(&self) -> f64 { self.lr }
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (i, param) in self.params.iter().enumerate() {
                if let Some(mut grad) = param.grad() {
                    let lr = self.lr_for_param(i);
                    let data = param.data().detach()?;

                    // Weight decay: add wd * param to gradient
                    if self.weight_decay > 0.0 {
                        grad = grad.add(&data.mul_scalar(self.weight_decay)?)?;
                    }

                    // Update running average of squared gradients
                    // v = alpha * v + (1 - alpha) * grad^2
                    let grad_sq = grad.mul(&grad)?;
                    let v = match self.v[i].take() {
                        Some(v) => {
                            v.mul_scalar_(self.alpha)?;
                            let scaled = grad_sq.mul_scalar(1.0 - self.alpha)?;
                            v.add_(&scaled)?;
                            v
                        }
                        None => grad_sq.mul_scalar(1.0 - self.alpha)?,
                    };

                    // Compute update: grad / (sqrt(v) + eps)
                    let denom = v.sqrt()?.add_scalar(self.eps)?;
                    let update = grad.div(&denom)?;

                    if self.momentum > 0.0 {
                        // buf = momentum * buf + update
                        let b = match self.buf[i].take() {
                            Some(b) => {
                                b.mul_scalar_(self.momentum)?;
                                b.add_(&update)?;
                                b
                            }
                            None => update.copy()?,
                        };
                        let scaled = b.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                        self.buf[i] = Some(b);
                    } else {
                        let scaled = update.mul_scalar(lr)?;
                        data.sub_(&scaled)?;
                    }

                    self.v[i] = Some(v);
                }
            }
            Ok(())
        })
    }

    fn reset_state(&mut self) {
        // Squared-grad running average + momentum buffer back to fresh.
        // Lengths preserved for per-param indexing.
        for slot in &mut self.v {
            *slot = None;
        }
        for slot in &mut self.buf {
            *slot = None;
        }
    }

    fn zero_grad(&self) {
        for param in &self.params {
            param.zero_grad_set_to_none();
        }
    }

    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        for g in &mut self.groups {
            g.lr = lr;
        }
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        if let Some(g) = self.groups.get_mut(group) {
            g.lr = lr;
        }
    }

    fn save_state_to(&self, path: &str) -> Result<()> {
        <Self as Stateful>::save_state_file(self, path)
    }
}

impl Stateful for RMSprop {
    fn state_kind(&self) -> super::StateKind { super::StateKind::RMSprop }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_f64_le(w, self.alpha)?;
        write_f64_le(w, self.eps)?;
        write_f64_le(w, self.weight_decay)?;
        write_f64_le(w, self.momentum)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.v[i].as_ref())?;
            write_tensor_state(w, self.buf[i].as_ref())?;
        }
        // Groups
        super::write_groups(w, &self.groups)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "RMSprop: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.alpha = read_f64_le(r)?;
        self.eps = read_f64_le(r)?;
        self.weight_decay = read_f64_le(r)?;
        self.momentum = read_f64_le(r)?;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.v[i] = read_tensor_state(r, dev)?;
            self.buf[i] = read_tensor_state(r, dev)?;
        }
        // Groups
        self.groups = super::read_groups(r, self.params.len(), "RMSprop")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_helpers::make_param;
    use crate::tensor::Tensor;

    #[test]
    fn test_rmsprop_basic() {
        let p = make_param("w", &[3, 2]);
        let mut opt = RMSprop::new(std::slice::from_ref(&p), 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        let before = p.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().to_f32_vec().unwrap();
        assert_ne!(before, after, "params should change after step");
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let p = make_param("w", &[3, 2]);
        let mut opt = RMSprop::builder(std::slice::from_ref(&p), 0.01)
            .momentum(0.9)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );

        // Two steps to exercise momentum buffer
        for _ in 0..2 {
            opt.zero_grad();
            let y = x.matmul(&p.variable).unwrap();
            let loss = y.sum().unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        // Should not crash and param should have changed
        let data = p.variable.data().to_f32_vec().unwrap();
        assert!(data.iter().any(|&v| v.abs() > 0.0), "params should be non-zero");
    }

    #[test]
    fn test_rmsprop_with_weight_decay() {
        // Create two params with identical initial values via from_f32
        let init = [0.5f32, -0.3, 0.1, 0.8, -0.2, 0.4];
        let dev = crate::tensor::test_device();

        let p1 = Parameter::new(
            Tensor::from_f32(&init, &[3, 2], dev).unwrap(), "w1");
        let p2 = Parameter::new(
            Tensor::from_f32(&init, &[3, 2], dev).unwrap(), "w2");

        let mut opt_wd = RMSprop::builder(std::slice::from_ref(&p1), 0.01)
            .weight_decay(0.1)
            .build();
        let mut opt_plain = RMSprop::new(std::slice::from_ref(&p2), 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], dev).unwrap(),
            false,
        );

        // Run multiple steps so accumulated v differs enough to produce
        // visibly different updates (single step RMSprop is dominated by sign)
        for _ in 0..10 {
            opt_wd.zero_grad();
            let y1 = x.matmul(&p1.variable).unwrap();
            y1.sum().unwrap().backward().unwrap();
            opt_wd.step().unwrap();

            opt_plain.zero_grad();
            let y2 = x.matmul(&p2.variable).unwrap();
            y2.sum().unwrap().backward().unwrap();
            opt_plain.step().unwrap();
        }

        let d1 = p1.variable.data().to_f32_vec().unwrap();
        let d2 = p2.variable.data().to_f32_vec().unwrap();
        // Weight decay should cause different parameter trajectories
        assert_ne!(d1, d2, "weight decay should produce different results after 10 steps");
    }

    #[test]
    fn test_rmsprop_convergence() {
        // Verify step works and changes param
        let p = Parameter::new(
            Tensor::from_f32(&[5.0], &[1], crate::tensor::test_device()).unwrap(),
            "x",
        );
        let mut opt = RMSprop::new(std::slice::from_ref(&p), 0.1);

        let x = Variable::new(
            Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.mul(&p.variable).unwrap();
        let loss = y.mul(&y).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();

        let val = p.variable.data().to_f32_vec().unwrap()[0];
        assert!(val < 5.0, "param should decrease from 5.0, got {}", val);
    }

    #[test]
    fn test_rmsprop_save_load() {
        let p = make_param("w", &[3, 2]);
        let mut opt = RMSprop::builder(std::slice::from_ref(&p), 0.01)
            .momentum(0.9)
            .alpha(0.95)
            .weight_decay(0.01)
            .build();

        // Do a step to populate buffers
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        y.sum().unwrap().backward().unwrap();
        opt.step().unwrap();

        // Save
        let mut buf = Vec::new();
        opt.save_state(&mut buf).unwrap();

        // Load into fresh optimizer
        let mut opt2 = RMSprop::builder(std::slice::from_ref(&p), 0.99)
            .build();
        let mut cursor = std::io::Cursor::new(&buf);
        opt2.load_state(&mut cursor).unwrap();

        assert!((opt2.lr - 0.01).abs() < 1e-12);
        assert!((opt2.alpha - 0.95).abs() < 1e-12);
        assert!((opt2.momentum - 0.9).abs() < 1e-12);
        assert!((opt2.weight_decay - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_rmsprop_builder_defaults() {
        let p = make_param("w", &[2]);
        let opt = RMSprop::new(std::slice::from_ref(&p), 0.01);
        assert!((opt.alpha - 0.99).abs() < 1e-12);
        assert!((opt.eps - 1e-8).abs() < 1e-15);
        assert!((opt.weight_decay).abs() < 1e-12);
        assert!((opt.momentum).abs() < 1e-12);
    }

    #[test]
    fn test_rmsprop_frozen_params() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);
        p1.freeze().unwrap();

        let mut opt = RMSprop::new(&[p1, p2.clone()], 0.01);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p2.variable).unwrap();
        y.sum().unwrap().backward().unwrap();

        // Should not crash with frozen param
        opt.step().unwrap();
        opt.zero_grad();
    }
}
