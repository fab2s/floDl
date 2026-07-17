//! Adam and AdamW optimizers.
//!
//! AdamW is colocated with Adam because it wraps `Adam` and calls its private
//! `adam_update` helper directly; keeping them in one file avoids exposing
//! cross-module internals.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    write_tensor_state, read_tensor_state, write_f64_le, read_f64_le,
    write_u32_le, read_u32_le, write_i64_le, read_i64_le,
};
use crate::nn::parameter::Parameter;

use super::{GroupMeta, Optimizer, Stateful};

/// Adam optimizer with bias correction (Kingma & Ba, 2014).
///
/// Maintains per-parameter first and second moment estimates with
/// bias correction. Default betas: (0.9, 0.999), eps: 1e-8.
///
/// ```ignore
/// let mut optim = Adam::new(&model.parameters(), 0.001);
/// ```
pub struct Adam {
    params: Vec<Variable>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    m: Vec<Option<crate::tensor::Tensor>>,
    v: Vec<Option<crate::tensor::Tensor>>,
    t: usize,
    groups: Vec<GroupMeta>,
}

impl Adam {
    /// Create a new Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
    pub fn new(params: &[Parameter], lr: f64) -> Self {
        let n = params.len();
        Adam {
            params: params.iter().map(|p| p.variable.clone()).collect(),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
            groups: vec![],
        }
    }

    /// Create a builder for Adam with per-group learning rates.
    pub fn with_groups() -> AdamBuilder {
        AdamBuilder { beta1: 0.9, beta2: 0.999, eps: 1e-8, groups: vec![] }
    }

    /// Current learning rate (base LR, or first group's LR).
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

/// Builder for Adam with per-group learning rates and customizable hyperparameters.
pub struct AdamBuilder {
    beta1: f64,
    beta2: f64,
    eps: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl AdamBuilder {
    /// Set exponential decay rates for moment estimates (default: (0.9, 0.999)).
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8).
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }

    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the Adam optimizer.
    pub fn build(self) -> Adam {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(1e-3);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let n = all_params.len();
        Adam {
            params: all_params,
            lr: base_lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
            groups,
        }
    }
}

impl Optimizer for Adam {
    fn lr(&self) -> f64 { self.lr }
    fn step(&mut self) -> Result<()> {
        self.adam_update(0.0)
    }

    fn reset_state(&mut self) {
        // First + second moment estimates back to fresh, step counter to 0
        // (bias correction restarts). Lengths preserved for per-param indexing.
        for slot in &mut self.m {
            *slot = None;
        }
        for slot in &mut self.v {
            *slot = None;
        }
        self.t = 0;
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

impl Adam {
    fn adam_update(&mut self, weight_decay: f64) -> Result<()> {
        self.t += 1;

        no_grad(|| {
            // Determine effective groups (single group if none configured)
            let effective_groups: Vec<(f64, std::ops::Range<usize>)> = if self.groups.is_empty() {
                vec![(self.lr, 0..self.params.len())]
            } else {
                self.groups.iter().map(|g| (g.lr, g.range.clone())).collect()
            };

            for (lr, range) in &effective_groups {
                let mut p_tensors = Vec::new();
                let mut g_tensors = Vec::new();
                let mut m_tensors = Vec::new();
                let mut v_tensors = Vec::new();

                for i in range.clone() {
                    if let Some(grad) = self.params[i].grad() {
                        // Lazy-init moment buffers as zeros on first step
                        if self.m[i].is_none() {
                            self.m[i] = Some(crate::tensor::Tensor::zeros_like(&grad)?);
                        }
                        if self.v[i].is_none() {
                            self.v[i] = Some(crate::tensor::Tensor::zeros_like(&grad)?);
                        }

                        p_tensors.push(self.params[i].data());
                        g_tensors.push(grad);
                        m_tensors.push(self.m[i].as_ref().unwrap().clone());
                        v_tensors.push(self.v[i].as_ref().unwrap().clone());
                    }
                }

                if !p_tensors.is_empty() {
                    // Single fused kernel for all params in this group
                    crate::tensor::Tensor::fused_adamw_(
                        &p_tensors, &g_tensors, &m_tensors, &v_tensors,
                        *lr, self.beta1, self.beta2, self.eps,
                        weight_decay, self.t as i64, None, None,
                    )?;
                }
            }
            Ok(())
        })
    }
}

impl Stateful for Adam {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        write_i64_le(w, self.t as i64)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.m[i].as_ref())?;
            write_tensor_state(w, self.v[i].as_ref())?;
        }
        // Groups
        super::write_groups(w, &self.groups)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "Adam: param count mismatch: checkpoint={} optimizer={}", count, self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        self.t = read_i64_le(r)? as usize;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.m[i] = read_tensor_state(r, dev)?;
            self.v[i] = read_tensor_state(r, dev)?;
        }
        // Groups
        self.groups = super::read_groups(r, self.params.len(), "Adam")?;
        Ok(())
    }
}

/// AdamW optimizer — Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
///
/// Unlike L2 regularization, weight decay is applied directly to parameters,
/// not to gradients. This distinction matters for adaptive optimizers and
/// generally improves generalization.
///
/// ```ignore
/// let mut optim = AdamW::new(&model.parameters(), 0.001, 0.01);
/// ```
pub struct AdamW {
    adam: Adam,
    weight_decay: f64,
}

impl AdamW {
    /// Create a new AdamW optimizer. `weight_decay` is applied directly to
    /// parameters (decoupled), not to gradients. Typical values: 0.01--0.1.
    pub fn new(params: &[Parameter], lr: f64, weight_decay: f64) -> Self {
        AdamW {
            adam: Adam::new(params, lr),
            weight_decay,
        }
    }

    /// Create a builder for AdamW with per-group learning rates.
    pub fn with_groups(weight_decay: f64) -> AdamWBuilder {
        AdamWBuilder { beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay, groups: vec![] }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.adam.lr
    }
}

/// Builder for AdamW with per-group learning rates and customizable hyperparameters.
pub struct AdamWBuilder {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    groups: Vec<(Vec<Variable>, f64)>,
}

impl AdamWBuilder {
    /// Set exponential decay rates for moment estimates (default: (0.9, 0.999)).
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8).
    pub fn eps(mut self, eps: f64) -> Self { self.eps = eps; self }

    /// Add a parameter group with its own learning rate.
    pub fn group(mut self, params: &[Parameter], lr: f64) -> Self {
        let vars: Vec<Variable> = params.iter().map(|p| p.variable.clone()).collect();
        self.groups.push((vars, lr));
        self
    }

    /// Build the AdamW optimizer.
    pub fn build(self) -> AdamW {
        let mut all_params = Vec::new();
        let mut groups = Vec::new();
        let base_lr = self.groups.first().map(|(_, lr)| *lr).unwrap_or(1e-3);

        for (vars, lr) in self.groups {
            let start = all_params.len();
            all_params.extend(vars);
            let end = all_params.len();
            groups.push(GroupMeta { lr, range: start..end });
        }

        let n = all_params.len();
        AdamW {
            adam: Adam {
                params: all_params,
                lr: base_lr,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.eps,
                m: vec![None; n],
                v: vec![None; n],
                t: 0,
                groups,
            },
            weight_decay: self.weight_decay,
        }
    }
}

impl Optimizer for AdamW {
    fn lr(&self) -> f64 { self.adam.lr }
    fn step(&mut self) -> Result<()> {
        self.adam.adam_update(self.weight_decay)
    }

    fn reset_state(&mut self) {
        self.adam.reset_state()
    }

    fn zero_grad(&self) {
        self.adam.zero_grad()
    }

    fn set_lr(&mut self, lr: f64) {
        self.adam.set_lr(lr);
    }

    fn set_group_lr(&mut self, group: usize, lr: f64) {
        self.adam.set_group_lr(group, lr);
    }

    fn save_state_to(&self, path: &str) -> Result<()> {
        <Self as Stateful>::save_state_file(self, path)
    }
}

impl Stateful for AdamW {
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_f64_le(w, self.weight_decay)?;
        self.adam.save_state(w)
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.weight_decay = read_f64_le(r)?;
        self.adam.load_state(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_helpers::make_param;
    use crate::tensor::Tensor;

    #[test]
    fn test_adam_backward_compat() {
        // Adam::new still works with a single LR
        let p = make_param("w", &[3, 2]);
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.01);

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
    fn test_reset_state_matches_fresh_optimizer() {
        // After warming an optimizer (advancing `t`, filling `m`/`v`),
        // `reset_state` must make its next step identical to a freshly
        // constructed optimizer over the same parameter values — i.e. the
        // moment estimates and step counter are genuinely wiped (the DiLoCo
        // disposable-inner property).
        let dev = crate::tensor::test_device();
        let init = [0.5f32, -0.3, 0.8, 0.2, -0.1, 0.4];
        let shape = [3i64, 2];
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], dev).unwrap();
        // Grad of sum(x · p) wrt p is independent of p's values, so two params
        // at the same values get identical grads from this expression.
        let loss_of = |p: &crate::nn::Parameter| {
            Variable::new(x.clone(), false)
                .matmul(&p.variable)
                .unwrap()
                .sum()
                .unwrap()
        };

        let p_warm = crate::nn::Parameter::new(
            Tensor::from_f32(&init, &shape, dev).unwrap(),
            "w",
        );
        let mut opt_warm = Adam::new(std::slice::from_ref(&p_warm), 0.01);
        for _ in 0..5 {
            loss_of(&p_warm).backward().unwrap();
            opt_warm.step().unwrap();
            opt_warm.zero_grad();
        }

        // A truly fresh optimizer over a param at the SAME (warmed) values.
        let warmed = p_warm.variable.data().to_f32_vec().unwrap();
        let p_fresh = crate::nn::Parameter::new(
            Tensor::from_f32(&warmed, &shape, dev).unwrap(),
            "w",
        );
        let mut opt_fresh = Adam::new(std::slice::from_ref(&p_fresh), 0.01);

        opt_warm.reset_state();

        loss_of(&p_warm).backward().unwrap();
        opt_warm.step().unwrap();
        loss_of(&p_fresh).backward().unwrap();
        opt_fresh.step().unwrap();

        let after_reset = p_warm.variable.data().to_f32_vec().unwrap();
        let after_fresh = p_fresh.variable.data().to_f32_vec().unwrap();
        for (i, (a, b)) in after_reset.iter().zip(&after_fresh).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "param[{i}]: reset-then-step {a} != fresh {b} \
                 (reset_state must wipe m/v and the step counter)"
            );
        }
    }

    #[test]
    fn test_adam_two_groups_different_lr() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        // Group 0: high LR, Group 1: very low LR
        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.1)
            .group(std::slice::from_ref(&p2), 1e-10)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        // Both params participate
        let y1 = x.matmul(&p1.variable).unwrap();
        let y2 = x.matmul(&p2.variable).unwrap();
        let loss = y1.add(&y2).unwrap().sum().unwrap();
        loss.backward().unwrap();

        let p1_before = p1.variable.data().to_f32_vec().unwrap();
        let p2_before = p2.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let p1_after = p1.variable.data().to_f32_vec().unwrap();
        let p2_after = p2.variable.data().to_f32_vec().unwrap();

        // p1 should change substantially (high LR), p2 barely moves (tiny LR)
        let p1_delta: f64 = p1_before.iter().zip(&p1_after)
            .map(|(a, b)| (a - b).abs() as f64).sum();
        let p2_delta: f64 = p2_before.iter().zip(&p2_after)
            .map(|(a, b)| (a - b).abs() as f64).sum();

        assert!(p1_delta > p2_delta * 1e6,
            "high-LR group should move much more: p1_delta={}, p2_delta={}", p1_delta, p2_delta);
    }

    #[test]
    fn test_set_group_lr_changes_one_group() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.01)
            .build();

        opt.set_group_lr(1, 0.99);
        // Group 0 unchanged, group 1 updated
        assert!((opt.groups[0].lr - 0.01).abs() < 1e-12);
        assert!((opt.groups[1].lr - 0.99).abs() < 1e-12);
    }

    #[test]
    fn test_set_lr_changes_all_groups() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.05)
            .build();

        opt.set_lr(0.42);
        assert!((opt.lr - 0.42).abs() < 1e-12);
        assert!((opt.groups[0].lr - 0.42).abs() < 1e-12);
        assert!((opt.groups[1].lr - 0.42).abs() < 1e-12);
    }

    #[test]
    fn test_frozen_params_in_group_no_crash() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);
        p1.freeze().unwrap();

        let mut opt = Adam::with_groups()
            .group(&[p1, p2.clone()], 0.01)
            .build();

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p2.variable).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // Should not crash even though p1 is frozen (no grad)
        opt.step().unwrap();
        opt.zero_grad();
    }

    #[test]
    fn test_adam_save_load_with_groups() {
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);

        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.05)
            .build();

        // Do a step to populate moment buffers
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y1 = x.matmul(&p1.variable).unwrap();
        let y2 = x.matmul(&p2.variable).unwrap();
        let loss = y1.add(&y2).unwrap().sum().unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();

        // Save
        let mut buf = Vec::new();
        opt.save_state(&mut buf).unwrap();

        // Load into fresh optimizer with same structure
        let mut opt2 = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.99)
            .group(std::slice::from_ref(&p2), 0.99)
            .build();

        let mut cursor = std::io::Cursor::new(&buf);
        opt2.load_state(&mut cursor).unwrap();

        assert_eq!(opt2.t, opt.t);
        assert!((opt2.groups[0].lr - 0.01).abs() < 1e-12);
        assert!((opt2.groups[1].lr - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_load_state_rejects_corrupt_group_ranges() {
        // A corrupt group table must error at load, not restore ranges that
        // index out of bounds at the next step() (or silently skip params).
        let p1 = make_param("w1", &[3, 2]);
        let p2 = make_param("w2", &[3, 2]);
        let mut opt = Adam::with_groups()
            .group(std::slice::from_ref(&p1), 0.01)
            .group(std::slice::from_ref(&p2), 0.05)
            .build();

        let mut buf = Vec::new();
        opt.save_state(&mut buf).unwrap();

        // The last 8 bytes are the final group's `end` (i64 LE) — inflate it
        // past the param count.
        let n = buf.len();
        buf[n - 8..].copy_from_slice(&999i64.to_le_bytes());

        let err = opt
            .load_state(&mut std::io::Cursor::new(&buf))
            .expect_err("inflated group range must be rejected");
        assert!(
            err.to_string().contains("corrupt optimizer state"),
            "unexpected error: {err}"
        );

        // Non-contiguous coverage (a gap would silently skip params): patch
        // the SECOND group's start (bytes n-16..n-8) to overlap group 0.
        let mut buf2 = Vec::new();
        opt.save_state(&mut buf2).unwrap();
        let n2 = buf2.len();
        buf2[n2 - 16..n2 - 8].copy_from_slice(&0i64.to_le_bytes());
        buf2[n2 - 8..].copy_from_slice(&2i64.to_le_bytes());
        let err2 = opt
            .load_state(&mut std::io::Cursor::new(&buf2))
            .expect_err("non-contiguous group table must be rejected");
        assert!(err2.to_string().contains("corrupt optimizer state"));
    }

    #[test]
    fn test_fused_adam_numerical_correctness() {
        // Known param/grad/m/v, verify against hand-computed expected values
        let param = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[0.1, 0.2, 0.3, 0.4], &[4], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap();

        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let step: i64 = 1;

        param.adam_step(&grad, &m, &v, lr, beta1, beta2, eps, 0.0, step).unwrap();

        // After step 1 with zero initial moments:
        // m = 0.1 * grad, v = 0.001 * grad^2
        // bc1 = 0.1, bc2 = 0.001
        // step_size = lr / bc1 = 0.01
        // denom = sqrt(v / bc2) + eps = |grad| + eps
        // update = step_size * m / denom ≈ step_size * 0.1*grad / |grad| ≈ 0.001 * sign(grad)
        // With positive grad: param -= 0.001

        let p_data = param.to_f32_vec().unwrap();
        let m_data = m.to_f32_vec().unwrap();
        let v_data = v.to_f32_vec().unwrap();

        // m = (1-beta1)*grad = 0.1 * [0.1, 0.2, 0.3, 0.4]
        for (i, &g) in [0.1f32, 0.2, 0.3, 0.4].iter().enumerate() {
            assert!((m_data[i] - 0.1 * g).abs() < 1e-6,
                "m[{}]: got {}, expected {}", i, m_data[i], 0.1 * g);
        }

        // v = (1-beta2)*grad^2 = 0.001 * [0.01, 0.04, 0.09, 0.16]
        for (i, &g) in [0.1f32, 0.2, 0.3, 0.4].iter().enumerate() {
            assert!((v_data[i] - 0.001 * g * g).abs() < 1e-9,
                "v[{}]: got {}, expected {}", i, v_data[i], 0.001 * g * g);
        }

        // Each param element should have decreased by approximately lr
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        for (i, &o) in orig.iter().enumerate() {
            assert!((p_data[i] - (o - lr as f32)).abs() < 1e-5,
                "p[{}]: got {}, expected ~{}", i, p_data[i], o - lr as f32);
        }
    }

    #[test]
    fn test_fused_adamw_weight_decay() {
        let param = Tensor::from_f32(&[1.0, 2.0], &[2], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[0.1, 0.1], &[2], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[2], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[2], crate::tensor::test_opts()).unwrap();

        let lr = 0.001;
        let wd = 0.01;

        param.adam_step(&grad, &m, &v, lr, 0.9, 0.999, 1e-8, wd, 1).unwrap();

        let p_data = param.to_f32_vec().unwrap();
        // Weight decay: p *= (1 - lr * wd) = (1 - 0.00001)
        // Then Adam update subtracts ~lr from each element
        // param[0] should be slightly less than 1.0 - 0.001
        // param[1] should be slightly less than 2.0 - 0.001, but also
        // decayed more because 2.0 * lr * wd > 1.0 * lr * wd
        assert!(p_data[0] < 1.0, "p[0] should decrease: got {}", p_data[0]);
        assert!(p_data[1] < 2.0, "p[1] should decrease: got {}", p_data[1]);
        // Weight decay asymmetry: param[1] decays more (larger value)
        let decay_0 = 1.0 - p_data[0] as f64;
        let decay_1 = 2.0 - p_data[1] as f64;
        assert!(decay_1 > decay_0, "larger param should decay more: d0={}, d1={}", decay_0, decay_1);
    }

    #[test]
    fn test_fused_adam_multi_step_convergence() {
        // Run multiple steps, verify m/v accumulate correctly
        let param = Tensor::from_f32(&[5.0], &[1], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[1], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[1], crate::tensor::test_opts()).unwrap();

        for step in 1..=10 {
            param.adam_step(&grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, step).unwrap();
        }

        // After 10 steps with constant gradient=1:
        // m should converge toward 1.0, v should converge toward 1.0
        let m_data = m.to_f32_vec().unwrap();
        let p_data = param.to_f32_vec().unwrap();

        // m = 1 - 0.9^10 ≈ 0.6513
        assert!((m_data[0] - 0.6513).abs() < 0.01,
            "m after 10 steps: got {}", m_data[0]);
        // v should be non-zero (accumulating)
        assert!(v.to_f32_vec().unwrap()[0] > 0.0, "v should accumulate");
        // param should have decreased
        assert!(p_data[0] < 5.0, "param should decrease: got {}", p_data[0]);
    }

    #[test]
    fn test_adam_zero_lr_no_param_change() {
        let p = make_param("w", &[3, 2]);
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.0);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let before = p.variable.data().to_f32_vec().unwrap();
        let y = x.matmul(&p.variable).unwrap();
        y.sum().unwrap().backward().unwrap();
        opt.step().unwrap();
        let after = p.variable.data().to_f32_vec().unwrap();
        assert_eq!(before, after, "lr=0 should leave parameters unchanged");
    }

    #[test]
    fn test_adam_very_small_lr_no_nan() {
        let p = make_param("w", &[4, 3]);
        let mut opt = Adam::new(std::slice::from_ref(&p), 1e-30);

        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        y.sum().unwrap().backward().unwrap();
        opt.step().unwrap();

        let vals = p.variable.data().to_f32_vec().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "param[{}] is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_double_step_without_backward_is_noop() {
        let p = make_param("w", &[3, 2]);
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.01);

        // Do one forward+backward+step
        let x = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0], &[1, 3], crate::tensor::test_device()).unwrap(),
            false,
        );
        let y = x.matmul(&p.variable).unwrap();
        y.sum().unwrap().backward().unwrap();
        opt.step().unwrap();
        opt.zero_grad();

        // Now step again without backward: no gradients, should be a no-op
        let after_first = p.variable.data().to_f32_vec().unwrap();
        opt.step().unwrap();
        let after_second = p.variable.data().to_f32_vec().unwrap();

        assert_eq!(after_first, after_second,
            "second step without backward should not change params");
    }
}
