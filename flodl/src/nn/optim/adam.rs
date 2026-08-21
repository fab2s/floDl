//! Adam and AdamW optimizers.
//!
//! AdamW is colocated with Adam because it wraps `Adam` and calls its private
//! `adam_update` helper directly; keeping them in one file avoids exposing
//! cross-module internals.

use std::io::{Read, Write};

use crate::autograd::{Variable, no_grad};
use crate::tensor::Result;

use crate::nn::checkpoint::{
    read_f64_le, read_i64_le, read_tensor_state, read_u32_le, write_f64_le, write_i64_le,
    write_tensor_state, write_u32_le,
};
use crate::nn::parameter::Parameter;

use super::{GroupMeta, Optimizer, Stateful};

/// Adam optimizer with bias correction (Kingma & Ba, 2014).
///
/// Maintains per-parameter first and second moment estimates with
/// per-parameter bias correction (PyTorch-parity `state_steps`): a
/// parameter that only starts receiving gradients at step N — e.g.
/// unfrozen mid-run for fine-tuning — bias-corrects from ITS first
/// step, not the optimizer's global one.
/// Default betas: (0.9, 0.999), eps: 1e-8.
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
    /// Per-param step counts, incremented only when the param has a grad.
    steps: Vec<i64>,
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
            steps: vec![0; n],
            groups: vec![],
        }
    }

    /// Create a builder for Adam with per-group learning rates.
    pub fn with_groups() -> AdamBuilder {
        AdamBuilder {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            groups: vec![],
        }
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
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

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
            groups.push(GroupMeta {
                lr,
                range: start..end,
            });
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
            steps: vec![0; n],
            groups,
        }
    }
}

impl Optimizer for Adam {
    fn lr(&self) -> f64 {
        self.lr
    }
    fn step(&mut self) -> Result<()> {
        self.adam_update(0.0)
    }

    fn reset_state(&mut self) {
        // First + second moment estimates back to fresh, step counts to 0
        // (bias correction restarts). Lengths preserved for per-param indexing.
        for slot in &mut self.m {
            *slot = None;
        }
        for slot in &mut self.v {
            *slot = None;
        }
        self.steps.fill(0);
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
        no_grad(|| {
            // Determine effective groups (single group if none configured)
            let effective_groups: Vec<(f64, std::ops::Range<usize>)> = if self.groups.is_empty() {
                vec![(self.lr, 0..self.params.len())]
            } else {
                self.groups
                    .iter()
                    .map(|g| (g.lr, g.range.clone()))
                    .collect()
            };

            for (lr, range) in &effective_groups {
                let mut p_tensors = Vec::new();
                let mut g_tensors = Vec::new();
                let mut m_tensors = Vec::new();
                let mut v_tensors = Vec::new();
                let mut step_vals = Vec::new();

                for i in range.clone() {
                    if let Some(grad) = self.params[i].grad() {
                        // Lazy-init moment buffers as zeros on first step
                        if self.m[i].is_none() {
                            self.m[i] = Some(crate::tensor::Tensor::zeros_like(&grad)?);
                        }
                        if self.v[i].is_none() {
                            self.v[i] = Some(crate::tensor::Tensor::zeros_like(&grad)?);
                        }
                        // Per-param step: a param unfrozen at global step N
                        // bias-corrects from its own first step.
                        self.steps[i] += 1;

                        p_tensors.push(self.params[i].data());
                        g_tensors.push(grad);
                        m_tensors.push(self.m[i].as_ref().unwrap().clone());
                        v_tensors.push(self.v[i].as_ref().unwrap().clone());
                        step_vals.push(self.steps[i]);
                    }
                }

                if !p_tensors.is_empty() {
                    // Single fused kernel for all params in this group
                    crate::tensor::Tensor::fused_adamw_(
                        &p_tensors,
                        &g_tensors,
                        &m_tensors,
                        &v_tensors,
                        *lr,
                        self.beta1,
                        self.beta2,
                        self.eps,
                        weight_decay,
                        &step_vals,
                        None,
                        None,
                    )?;
                }
            }
            Ok(())
        })
    }
}

impl Stateful for Adam {
    fn state_kind(&self) -> super::StateKind {
        super::StateKind::Adam
    }

    fn save_state<W: Write>(&self, w: &mut W) -> Result<()> {
        write_u32_le(w, self.params.len() as u32)?;
        write_f64_le(w, self.lr)?;
        for i in 0..self.params.len() {
            write_tensor_state(w, self.m[i].as_ref())?;
            write_tensor_state(w, self.v[i].as_ref())?;
            write_i64_le(w, self.steps[i])?;
        }
        // Groups
        super::write_groups(w, &self.groups)?;
        Ok(())
    }

    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let count = read_u32_le(r)? as usize;
        if count != self.params.len() {
            return Err(crate::tensor::TensorError::new(&format!(
                "Adam: param count mismatch: checkpoint={} optimizer={}",
                count,
                self.params.len()
            )));
        }
        self.lr = read_f64_le(r)?;
        for i in 0..self.params.len() {
            let dev = self.params[i].data().device();
            self.m[i] = read_tensor_state(r, dev)?;
            self.v[i] = read_tensor_state(r, dev)?;
            self.steps[i] = read_i64_le(r)?;
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
        AdamWBuilder {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            groups: vec![],
        }
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
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

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
            groups.push(GroupMeta {
                lr,
                range: start..end,
            });
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
                steps: vec![0; n],
                groups,
            },
            weight_decay: self.weight_decay,
        }
    }
}

impl Optimizer for AdamW {
    fn lr(&self) -> f64 {
        self.adam.lr
    }
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
    fn state_kind(&self) -> super::StateKind {
        super::StateKind::AdamW
    }

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
    use super::super::test_helpers::{make_param, state_tmp};
    use super::*;
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

        let p_warm = crate::nn::Parameter::new(Tensor::from_f32(&init, &shape, dev).unwrap(), "w");
        let mut opt_warm = Adam::new(std::slice::from_ref(&p_warm), 0.01);
        for _ in 0..5 {
            loss_of(&p_warm).backward().unwrap();
            opt_warm.step().unwrap();
            opt_warm.zero_grad();
        }

        // A truly fresh optimizer over a param at the SAME (warmed) values.
        let warmed = p_warm.variable.data().to_f32_vec().unwrap();
        let p_fresh =
            crate::nn::Parameter::new(Tensor::from_f32(&warmed, &shape, dev).unwrap(), "w");
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
        let p1_delta: f64 = p1_before
            .iter()
            .zip(&p1_after)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();
        let p2_delta: f64 = p2_before
            .iter()
            .zip(&p2_after)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();

        assert!(
            p1_delta > p2_delta * 1e6,
            "high-LR group should move much more: p1_delta={}, p2_delta={}",
            p1_delta,
            p2_delta
        );
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

        let mut opt = Adam::with_groups().group(&[p1, p2.clone()], 0.01).build();

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

        assert_eq!(opt2.steps, opt.steps);
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
        let param =
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], crate::tensor::test_device()).unwrap();
        let grad =
            Tensor::from_f32(&[0.1, 0.2, 0.3, 0.4], &[4], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap();

        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let step: i64 = 1;

        param
            .adam_step(&grad, &m, &v, lr, beta1, beta2, eps, 0.0, step)
            .unwrap();

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
            assert!(
                (m_data[i] - 0.1 * g).abs() < 1e-6,
                "m[{}]: got {}, expected {}",
                i,
                m_data[i],
                0.1 * g
            );
        }

        // v = (1-beta2)*grad^2 = 0.001 * [0.01, 0.04, 0.09, 0.16]
        for (i, &g) in [0.1f32, 0.2, 0.3, 0.4].iter().enumerate() {
            assert!(
                (v_data[i] - 0.001 * g * g).abs() < 1e-9,
                "v[{}]: got {}, expected {}",
                i,
                v_data[i],
                0.001 * g * g
            );
        }

        // Each param element should have decreased by approximately lr
        let orig = [1.0f32, 2.0, 3.0, 4.0];
        for (i, &o) in orig.iter().enumerate() {
            assert!(
                (p_data[i] - (o - lr as f32)).abs() < 1e-5,
                "p[{}]: got {}, expected ~{}",
                i,
                p_data[i],
                o - lr as f32
            );
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

        param
            .adam_step(&grad, &m, &v, lr, 0.9, 0.999, 1e-8, wd, 1)
            .unwrap();

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
        assert!(
            decay_1 > decay_0,
            "larger param should decay more: d0={}, d1={}",
            decay_0,
            decay_1
        );
    }

    #[test]
    fn test_fused_adam_multi_step_convergence() {
        // Run multiple steps, verify m/v accumulate correctly
        let param = Tensor::from_f32(&[5.0], &[1], crate::tensor::test_device()).unwrap();
        let grad = Tensor::from_f32(&[1.0], &[1], crate::tensor::test_device()).unwrap();
        let m = Tensor::zeros(&[1], crate::tensor::test_opts()).unwrap();
        let v = Tensor::zeros(&[1], crate::tensor::test_opts()).unwrap();

        for step in 1..=10 {
            param
                .adam_step(&grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, step)
                .unwrap();
        }

        // After 10 steps with constant gradient=1:
        // m should converge toward 1.0, v should converge toward 1.0
        let m_data = m.to_f32_vec().unwrap();
        let p_data = param.to_f32_vec().unwrap();

        // m = 1 - 0.9^10 ≈ 0.6513
        assert!(
            (m_data[0] - 0.6513).abs() < 0.01,
            "m after 10 steps: got {}",
            m_data[0]
        );
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

        assert_eq!(
            after_first, after_second,
            "second step without backward should not change params"
        );
    }

    #[test]
    fn test_late_unfrozen_param_bias_corrects_from_its_first_step() {
        // A param that receives its first gradient at global step N must
        // bias-correct from ITS step 1, not the optimizer's global count —
        // with a shared global t its first update lands ~3x too large
        // (m-hat under-boosted 10x, v-hat denominator under-boosted ~31x).
        let dev = crate::tensor::test_device();
        let a = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "a");
        let b = Parameter::new(Tensor::from_f32(&[3.0, 4.0], &[2], dev).unwrap(), "b");
        let mut opt = Adam::new(&[a.clone(), b.clone()], 0.01);

        let ga = Tensor::from_f32(&[0.5, -0.5], &[2], dev).unwrap();
        for _ in 0..5 {
            a.variable.set_grad(ga.clone());
            opt.step().unwrap();
            opt.zero_grad();
        }

        // b's first gradient arrives at global step 6.
        let gb = Tensor::from_f32(&[0.3, -0.2], &[2], dev).unwrap();
        b.variable.set_grad(gb.clone());
        opt.step().unwrap();
        let b_after = b.variable.data().to_f32_vec().unwrap();

        // Reference: a fresh Adam taking its first step on an identical param.
        let b_ref = Parameter::new(Tensor::from_f32(&[3.0, 4.0], &[2], dev).unwrap(), "b_ref");
        let mut opt_ref = Adam::new(std::slice::from_ref(&b_ref), 0.01);
        b_ref.variable.set_grad(gb.clone());
        opt_ref.step().unwrap();
        let ref_after = b_ref.variable.data().to_f32_vec().unwrap();

        for i in 0..2 {
            assert!(
                (b_after[i] - ref_after[i]).abs() < 1e-6,
                "late-unfrozen update must match a fresh first step: \
                 got {}, expected {}",
                b_after[i],
                ref_after[i]
            );
        }
        assert_eq!(opt.steps, vec![5, 1], "per-param step counts");
    }

    #[test]
    fn test_state_file_roundtrip_with_header_and_steps() {
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "w");
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.02);
        p.variable
            .set_grad(Tensor::from_f32(&[0.1, 0.2], &[2], dev).unwrap());
        opt.step().unwrap();

        let path = state_tmp("adam_roundtrip.optim");
        opt.save_state_to(&path).unwrap();

        let mut opt2 = Adam::new(std::slice::from_ref(&p), 0.5);
        opt2.load_state_file(&path).unwrap();
        assert_eq!(opt2.steps, opt.steps);
        assert!((opt2.lr - 0.02).abs() < 1e-12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_state_file_kind_mismatch_is_rejected() {
        use super::super::SGD;
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0], &[1], dev).unwrap(), "w");
        let sgd = SGD::new(std::slice::from_ref(&p), 0.01, 0.9);
        let path = state_tmp("sgd_into_adam.optim");
        sgd.save_state_to(&path).unwrap();

        let mut adam = Adam::new(std::slice::from_ref(&p), 0.01);
        let err = adam
            .load_state_file(&path)
            .expect_err("SGD state must not load into Adam");
        let msg = err.to_string();
        assert!(msg.contains("written by SGD"), "unexpected: {msg}");
        assert!(msg.contains("Adam"), "unexpected: {msg}");
        let _ = std::fs::remove_file(&path);
    }

    /// Hand-built pre-header Adam stream: `count | lr | t | (m,v)* | ng=0`.
    fn old_format_adam_bytes(dev: crate::tensor::Device) -> Vec<u8> {
        let mut old = Vec::new();
        write_u32_le(&mut old, 1).unwrap();
        write_f64_le(&mut old, 0.02).unwrap();
        write_i64_le(&mut old, 7).unwrap();
        let m = Tensor::from_f32(&[0.1, 0.2], &[2], dev).unwrap();
        let v = Tensor::from_f32(&[0.3, 0.4], &[2], dev).unwrap();
        write_tensor_state(&mut old, Some(&m)).unwrap();
        write_tensor_state(&mut old, Some(&v)).unwrap();
        write_u32_le(&mut old, 0).unwrap();
        old
    }

    #[test]
    fn test_pre_header_state_file_is_rejected_with_converter_pointer() {
        let dev = crate::tensor::test_device();
        let path = state_tmp("adam_old_format.optim");
        std::fs::write(&path, old_format_adam_bytes(dev)).unwrap();

        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "w");
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.01);
        let err = opt
            .load_state_file(&path)
            .expect_err("pre-header file must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("migrate_optim_state_file"),
            "unexpected: {msg}"
        );
        assert!(msg.contains("Adam"), "unexpected: {msg}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_migrate_optim_state_file_expands_global_t_to_per_param_steps() {
        use super::super::{StateKind, migrate_optim_state_file};
        let dev = crate::tensor::test_device();
        let src = state_tmp("adam_migrate_src.optim");
        let dst = state_tmp("adam_migrate_dst.optim");
        std::fs::write(&src, old_format_adam_bytes(dev)).unwrap();

        migrate_optim_state_file(&src, &dst, StateKind::Adam).unwrap();

        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "w");
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.5);
        opt.load_state_file(&dst).unwrap();
        assert_eq!(opt.steps, vec![7], "old global t expands to every param");
        assert!((opt.lr - 0.02).abs() < 1e-12);
        let m = opt.m[0].as_ref().unwrap().to_f32_vec().unwrap();
        let v = opt.v[0].as_ref().unwrap().to_f32_vec().unwrap();
        assert_eq!(m, vec![0.1, 0.2]);
        assert_eq!(v, vec![0.3, 0.4]);

        // A second migrate on the converted file must refuse (already headed).
        let err = migrate_optim_state_file(&dst, &dst, StateKind::Adam)
            .expect_err("already-converted file must be refused");
        assert!(err.to_string().contains("already"), "unexpected: {err}");

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn test_migrate_optim_state_file_adamw_weight_decay_prefix() {
        use super::super::{StateKind, migrate_optim_state_file};
        let dev = crate::tensor::test_device();
        // Old AdamW stream = weight_decay(f64) | old Adam stream.
        let mut old = Vec::new();
        write_f64_le(&mut old, 0.04).unwrap();
        old.extend_from_slice(&old_format_adam_bytes(dev));
        let src = state_tmp("adamw_migrate_src.optim");
        let dst = state_tmp("adamw_migrate_dst.optim");
        std::fs::write(&src, &old).unwrap();

        migrate_optim_state_file(&src, &dst, StateKind::AdamW).unwrap();

        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "w");
        let mut opt = AdamW::new(std::slice::from_ref(&p), 0.5, 0.9);
        opt.load_state_file(&dst).unwrap();
        assert!((opt.weight_decay - 0.04).abs() < 1e-12);
        assert_eq!(opt.adam.steps, vec![7]);

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn test_migrate_verbatim_sgd_roundtrips() {
        // SGD's payload did not change this cycle, so it takes the migrate
        // verbatim branch: the migrated file is exactly a 12-byte FDLO header
        // followed by the old payload byte-for-byte. Raw `save_state` writes
        // that pre-header payload, so it faithfully stands in for an old file.
        use super::super::{SGD, StateKind, migrate_optim_state_file};
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], dev).unwrap(), "w");
        let mut sgd = SGD::new(std::slice::from_ref(&p), 0.05, 0.9);
        p.variable
            .set_grad(Tensor::from_f32(&[0.1, 0.2, 0.3], &[3], dev).unwrap());
        sgd.step().unwrap(); // populate the velocity buffer

        let mut old = Vec::new();
        sgd.save_state(&mut old).unwrap();
        let src = state_tmp("sgd_verbatim_src.optim");
        let dst = state_tmp("sgd_verbatim_dst.optim");
        std::fs::write(&src, &old).unwrap();

        migrate_optim_state_file(&src, &dst, StateKind::Sgd).unwrap();

        let migrated = std::fs::read(&dst).unwrap();
        assert_eq!(
            &migrated[12..],
            &old[..],
            "SGD payload must migrate verbatim under the 12-byte FDLO header"
        );

        // And the migrated file loads.
        let mut sgd2 = SGD::new(std::slice::from_ref(&p), 0.99, 0.0);
        sgd2.load_state_file(&dst).unwrap();
        assert!((sgd2.lr() - 0.05).abs() < 1e-12);

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn test_migrate_verbatim_rmsprop_roundtrips() {
        // Same verbatim branch, a different payload shape (alpha/eps/momentum
        // + v/buf tensors) to confirm the copy is shape-agnostic.
        use super::super::{RMSprop, StateKind, migrate_optim_state_file};
        let dev = crate::tensor::test_device();
        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "w");
        let mut opt = RMSprop::new(std::slice::from_ref(&p), 0.01);
        p.variable
            .set_grad(Tensor::from_f32(&[0.3, -0.1], &[2], dev).unwrap());
        opt.step().unwrap();

        let mut old = Vec::new();
        opt.save_state(&mut old).unwrap();
        let src = state_tmp("rmsprop_verbatim_src.optim");
        let dst = state_tmp("rmsprop_verbatim_dst.optim");
        std::fs::write(&src, &old).unwrap();

        migrate_optim_state_file(&src, &dst, StateKind::RMSprop).unwrap();

        let migrated = std::fs::read(&dst).unwrap();
        assert_eq!(
            &migrated[12..],
            &old[..],
            "RMSprop payload must migrate verbatim"
        );

        let mut opt2 = RMSprop::new(std::slice::from_ref(&p), 0.99);
        opt2.load_state_file(&dst).unwrap();
        assert!((opt2.lr() - 0.01).abs() < 1e-12);

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn test_migrate_in_place_old_adam_file() {
        // Genuine in-place (src == dst) conversion of an OLD file: the source
        // must be fully read before the atomic rename replaces it. (The
        // existing double-migrate test only covers the already-headed refusal.)
        use super::super::{StateKind, migrate_optim_state_file};
        let dev = crate::tensor::test_device();
        let path = state_tmp("adam_in_place.optim");
        std::fs::write(&path, old_format_adam_bytes(dev)).unwrap();

        migrate_optim_state_file(&path, &path, StateKind::Adam).unwrap();

        let p = Parameter::new(Tensor::from_f32(&[1.0, 2.0], &[2], dev).unwrap(), "w");
        let mut opt = Adam::new(std::slice::from_ref(&p), 0.5);
        opt.load_state_file(&path).unwrap();
        assert_eq!(
            opt.steps,
            vec![7],
            "in-place migrate must preserve the payload"
        );
        assert!((opt.lr - 0.02).abs() < 1e-12);
        // No stray .tmp left behind.
        assert!(!std::path::Path::new(&format!("{path}.tmp")).exists());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_migrate_rejects_kinds_that_never_had_a_format() {
        // Adagrad/RAdam/NAdam gained Stateful only under the FDLO header, so
        // there is no pre-header file to convert — asking is a loud error, not
        // a silent no-op that could emit a bogus "converted" file.
        use super::super::{StateKind, migrate_optim_state_file};
        let src = state_tmp("never_had_format_src.optim");
        let dst = state_tmp("never_had_format_dst.optim");
        std::fs::write(&src, [0u8; 16]).unwrap();
        for kind in [StateKind::Adagrad, StateKind::RAdam, StateKind::NAdam] {
            let err = migrate_optim_state_file(&src, &dst, kind)
                .expect_err("kinds with no pre-header format must be rejected");
            assert!(
                err.to_string().contains("nothing to migrate"),
                "unexpected error for {kind:?}: {err}"
            );
        }
        assert!(
            !std::path::Path::new(&dst).exists(),
            "no output file on rejection"
        );
        let _ = std::fs::remove_file(&src);
    }
}
