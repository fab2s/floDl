/// Learning rate scheduler interface.
///
/// Schedulers are pure strategy objects — they compute the LR for a given step
/// without holding a reference to the optimizer. The user bridges them:
///
/// ```ignore
/// optimizer.set_lr(sched.lr(step));
/// ```
pub trait Scheduler: Send + Sync {
    /// Learning rate at the given step (0-based).
    fn lr(&self, step: usize) -> f64;
}

/// Staircase decay: multiply lr by gamma every `step_size` steps.
pub struct StepDecay {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
}

impl StepDecay {
    /// Create a step decay scheduler: lr *= gamma every `step_size` steps.
    ///
    /// # Panics
    /// If `step_size == 0` (would divide by zero in [`lr`](Self::lr)).
    /// Panics at construction — a fail-fast config check before the run,
    /// not a cryptic mid-training divide-by-zero.
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        assert!(step_size >= 1, "StepDecay: step_size must be >= 1, got {step_size}");
        StepDecay {
            base_lr,
            step_size,
            gamma,
        }
    }

    /// Learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        let decays = step / self.step_size;
        self.base_lr * self.gamma.powi(decays as i32)
    }
}

impl Scheduler for StepDecay {
    fn lr(&self, step: usize) -> f64 {
        StepDecay::lr(self, step)
    }
}

/// Cosine annealing from base_lr to min_lr over total_steps.
pub struct CosineScheduler {
    base_lr: f64,
    min_lr: f64,
    total_steps: usize,
}

impl CosineScheduler {
    /// Create a cosine annealing scheduler from `base_lr` to `min_lr` over `total_steps`.
    ///
    /// # Panics
    /// If `total_steps == 0` (would yield a NaN LR via 0/0 in [`lr`](Self::lr)).
    /// Fail-fast at construction rather than a NaN partway through training.
    pub fn new(base_lr: f64, min_lr: f64, total_steps: usize) -> Self {
        assert!(total_steps >= 1, "CosineScheduler: total_steps must be >= 1, got {total_steps}");
        CosineScheduler {
            base_lr,
            min_lr,
            total_steps,
        }
    }

    /// Learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        let t = (step.min(self.total_steps) as f64) / (self.total_steps as f64);
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (t * std::f64::consts::PI).cos())
    }
}

impl Scheduler for CosineScheduler {
    fn lr(&self, step: usize) -> f64 {
        CosineScheduler::lr(self, step)
    }
}

/// Linear warmup followed by another scheduler.
/// Ramps lr from 0 to target_lr over warmup_steps, then delegates.
pub struct WarmupScheduler<S: Scheduler> {
    inner: S,
    target_lr: f64,
    warmup_steps: usize,
}

impl<S: Scheduler> WarmupScheduler<S> {
    /// Create a warmup scheduler that linearly ramps from 0 to `target_lr` over
    /// `warmup_steps`, then delegates to `inner` (whose step counter starts at 0
    /// after warmup completes).
    pub fn new(inner: S, target_lr: f64, warmup_steps: usize) -> Self {
        WarmupScheduler {
            inner,
            target_lr,
            warmup_steps,
        }
    }

    /// Learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            self.target_lr * (step as f64 + 1.0) / (self.warmup_steps as f64)
        } else {
            self.inner.lr(step - self.warmup_steps)
        }
    }
}

impl<S: Scheduler> Scheduler for WarmupScheduler<S> {
    fn lr(&self, step: usize) -> f64 {
        WarmupScheduler::lr(self, step)
    }
}

/// Reduce learning rate when a metric plateaus.
/// Call `observe(metric)` each epoch (lower is better).
///
/// Unlike step-based schedulers (`StepDecay`, `CosineScheduler`, etc.),
/// PlateauScheduler is reactive — it does not implement the [`Scheduler`]
/// trait because its LR depends on observed metrics, not step count.
/// This matches PyTorch's `ReduceLROnPlateau` which also has a different
/// interface from other schedulers.
///
/// ```ignore
/// let mut sched = PlateauScheduler::new(0.01, 5, 0.5, 1e-6);
/// for epoch in 0..100 {
///     let loss = train_epoch(&model, &data)?;
///     let lr = sched.observe(loss);
///     optimizer.set_lr(lr);
/// }
/// ```
pub struct PlateauScheduler {
    patience: usize,
    factor: f64,
    min_lr: f64,
    current_lr: f64,
    best: f64,
    wait: usize,
}

impl PlateauScheduler {
    /// Create a plateau scheduler that reduces lr by `factor` after `patience` epochs
    /// without improvement, down to `min_lr`.
    pub fn new(
        base_lr: f64,
        patience: usize,
        factor: f64,
        min_lr: f64,
    ) -> Self {
        PlateauScheduler {
            patience,
            factor,
            min_lr,
            current_lr: base_lr,
            best: f64::INFINITY,
            wait: 0,
        }
    }

    /// Feed an observed metric (lower is better). Reduces lr after
    /// `patience` epochs without improvement. Returns the current lr.
    pub fn observe(&mut self, metric: f64) -> f64 {
        if metric < self.best {
            self.best = metric;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.wait = 0;
            }
        }
        self.current_lr
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        self.current_lr
    }
}

/// Exponential decay: multiply lr by gamma each epoch.
///
/// `lr = base_lr * gamma^step`
///
/// ```ignore
/// let sched = ExponentialLR::new(0.1, 0.95);
/// assert!((sched.lr(0) - 0.1).abs() < 1e-10);
/// assert!((sched.lr(10) - 0.1 * 0.95f64.powi(10)).abs() < 1e-10);
/// ```
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    /// Create an exponential decay scheduler: lr = base_lr * gamma^step.
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        ExponentialLR { base_lr, gamma }
    }

    /// Learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        self.base_lr * self.gamma.powi(step as i32)
    }
}

impl Scheduler for ExponentialLR {
    fn lr(&self, step: usize) -> f64 {
        ExponentialLR::lr(self, step)
    }
}

/// Multi-step decay: multiply lr by gamma at each milestone.
///
/// `lr = base_lr * gamma^(number_of_milestones_passed)`
///
/// ```ignore
/// let sched = MultiStepLR::new(0.1, &[30, 60, 90], 0.1);
/// assert!((sched.lr(0) - 0.1).abs() < 1e-10);
/// assert!((sched.lr(30) - 0.01).abs() < 1e-10);
/// assert!((sched.lr(60) - 0.001).abs() < 1e-10);
/// ```
pub struct MultiStepLR {
    base_lr: f64,
    milestones: Vec<usize>,
    gamma: f64,
}

impl MultiStepLR {
    /// Create a multi-step scheduler that decays lr by `gamma` at each milestone epoch.
    /// Milestones should be provided in ascending order.
    pub fn new(base_lr: f64, milestones: &[usize], gamma: f64) -> Self {
        let mut ms = milestones.to_vec();
        ms.sort();
        MultiStepLR {
            base_lr,
            milestones: ms,
            gamma,
        }
    }

    /// Learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        let passed = self.milestones.iter().filter(|&&m| step >= m).count();
        self.base_lr * self.gamma.powi(passed as i32)
    }
}

impl Scheduler for MultiStepLR {
    fn lr(&self, step: usize) -> f64 {
        MultiStepLR::lr(self, step)
    }
}

/// One-cycle learning rate policy (Smith, 2018).
///
/// Linearly warms up to `max_lr` over the first 30% of `total_steps`,
/// then cosine-anneals down to `max_lr / 25` over the remaining 70%.
///
/// ```ignore
/// let sched = OneCycleLR::new(0.01, 100);
/// // Peak at step 30, then decay
/// ```
pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    warmup_steps: usize,
}

impl OneCycleLR {
    /// Create a one-cycle scheduler with given max learning rate and total steps.
    /// Warmup occupies the first 30% of steps.
    pub fn new(max_lr: f64, total_steps: usize) -> Self {
        let warmup_steps = (total_steps as f64 * 0.3).round() as usize;
        OneCycleLR {
            max_lr,
            total_steps,
            warmup_steps,
        }
    }

    /// Create with explicit warmup fraction (0.0 to 1.0).
    pub fn with_warmup_frac(max_lr: f64, total_steps: usize, warmup_frac: f64) -> Self {
        let warmup_steps = (total_steps as f64 * warmup_frac.clamp(0.0, 1.0)).round() as usize;
        OneCycleLR {
            max_lr,
            total_steps,
            warmup_steps,
        }
    }

    /// Learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps);
        let min_lr = self.max_lr / 25.0;

        if step < self.warmup_steps {
            // Linear warmup from min_lr to max_lr
            let frac = step as f64 / self.warmup_steps.max(1) as f64;
            min_lr + frac * (self.max_lr - min_lr)
        } else {
            // Cosine anneal from max_lr to min_lr
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
            let t = (step - self.warmup_steps) as f64 / decay_steps as f64;
            min_lr + 0.5 * (self.max_lr - min_lr) * (1.0 + (t * std::f64::consts::PI).cos())
        }
    }
}

impl Scheduler for OneCycleLR {
    fn lr(&self, step: usize) -> f64 {
        OneCycleLR::lr(self, step)
    }
}

/// Cyclic learning rate scheduler (Smith, 2017).
///
/// Cycles the learning rate between `base_lr` and `max_lr` with a triangular
/// policy. Each cycle has `step_size_up` steps going up and `step_size_down`
/// steps going down (default: same as up).
///
/// ```ignore
/// let sched = CyclicLR::new(0.001, 0.01, 2000); // half-cycle = 2000 steps
/// optimizer.set_lr(sched.lr(step));
/// ```
pub struct CyclicLR {
    base_lr: f64,
    max_lr: f64,
    step_size_up: usize,
    step_size_down: usize,
}

impl CyclicLR {
    /// Create a CyclicLR with symmetric up/down phases.
    /// `step_size`: number of steps in each half-cycle.
    ///
    /// # Panics
    /// If `step_size == 0` (each phase is a divisor, and the cycle length
    /// `up + down` is a modulus, in [`lr`](Self::lr)).
    pub fn new(base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        assert!(step_size >= 1, "CyclicLR: step_size must be >= 1, got {step_size}");
        CyclicLR { base_lr, max_lr, step_size_up: step_size, step_size_down: step_size }
    }

    /// Create with asymmetric up/down phase lengths.
    ///
    /// # Panics
    /// If either `step_size_up` or `step_size_down` is 0 (each is a divisor,
    /// and their sum is the cycle-length modulus, in [`lr`](Self::lr)).
    pub fn asymmetric(base_lr: f64, max_lr: f64, step_size_up: usize, step_size_down: usize) -> Self {
        assert!(
            step_size_up >= 1 && step_size_down >= 1,
            "CyclicLR: step_size_up and step_size_down must be >= 1, got up={step_size_up} down={step_size_down}"
        );
        CyclicLR { base_lr, max_lr, step_size_up, step_size_down }
    }

    /// Compute learning rate at the given step.
    pub fn lr(&self, step: usize) -> f64 {
        let cycle_len = self.step_size_up + self.step_size_down;
        let pos = step % cycle_len;
        let scale = if pos <= self.step_size_up {
            pos as f64 / self.step_size_up as f64
        } else {
            1.0 - (pos - self.step_size_up) as f64 / self.step_size_down as f64
        };
        self.base_lr + (self.max_lr - self.base_lr) * scale
    }
}

impl Scheduler for CyclicLR {
    fn lr(&self, step: usize) -> f64 {
        CyclicLR::lr(self, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "step_size must be >= 1")]
    fn test_step_decay_zero_step_size_panics_at_construction() {
        // Fail-fast: catches the config error at new(), not as a runtime
        // divide-by-zero deep in training.
        let _ = StepDecay::new(0.1, 0, 0.5);
    }

    #[test]
    #[should_panic(expected = "total_steps must be >= 1")]
    fn test_cosine_zero_total_steps_panics_at_construction() {
        let _ = CosineScheduler::new(0.1, 0.0, 0);
    }

    #[test]
    #[should_panic(expected = "step_size must be >= 1")]
    fn test_cyclic_zero_step_size_panics_at_construction() {
        let _ = CyclicLR::new(0.001, 0.01, 0);
    }

    #[test]
    #[should_panic(expected = "must be >= 1")]
    fn test_cyclic_asymmetric_zero_phase_panics_at_construction() {
        let _ = CyclicLR::asymmetric(0.001, 0.01, 10, 0);
    }

    #[test]
    fn test_step_decay() {
        let sched = StepDecay::new(0.1, 10, 0.5);
        assert!((sched.lr(0) - 0.1).abs() < 1e-10);
        assert!((sched.lr(9) - 0.1).abs() < 1e-10);
        assert!((sched.lr(10) - 0.05).abs() < 1e-10);
        assert!((sched.lr(20) - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_scheduler() {
        let sched = CosineScheduler::new(0.1, 0.001, 100);
        assert!((sched.lr(0) - 0.1).abs() < 1e-5);
        assert!((sched.lr(100) - 0.001).abs() < 1e-5);
        // Mid-point should be near average
        let mid = sched.lr(50);
        assert!(mid > 0.04 && mid < 0.06, "mid={}", mid);
    }

    #[test]
    fn test_exponential_lr() {
        let sched = ExponentialLR::new(0.1, 0.9);
        assert!((sched.lr(0) - 0.1).abs() < 1e-10);
        assert!((sched.lr(1) - 0.09).abs() < 1e-10);
        assert!((sched.lr(10) - 0.1 * 0.9f64.powi(10)).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_lr_scheduler_trait() {
        let sched = ExponentialLR::new(0.1, 0.95);
        let s: &dyn Scheduler = &sched;
        assert!((s.lr(0) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_multi_step_lr() {
        let sched = MultiStepLR::new(0.1, &[30, 60, 90], 0.1);
        assert!((sched.lr(0) - 0.1).abs() < 1e-10);
        assert!((sched.lr(29) - 0.1).abs() < 1e-10);
        assert!((sched.lr(30) - 0.01).abs() < 1e-10);
        assert!((sched.lr(59) - 0.01).abs() < 1e-10);
        assert!((sched.lr(60) - 0.001).abs() < 1e-10);
        assert!((sched.lr(89) - 0.001).abs() < 1e-10);
        assert!((sched.lr(90) - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_multi_step_lr_unsorted_milestones() {
        let sched = MultiStepLR::new(0.1, &[60, 30, 90], 0.5);
        // Should sort internally
        assert!((sched.lr(29) - 0.1).abs() < 1e-10);
        assert!((sched.lr(30) - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_multi_step_lr_scheduler_trait() {
        let sched = MultiStepLR::new(0.1, &[10], 0.5);
        let s: &dyn Scheduler = &sched;
        assert!((s.lr(10) - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_one_cycle_lr_shape() {
        let sched = OneCycleLR::new(0.01, 100);
        let min_lr = 0.01 / 25.0;

        // Start near min
        assert!((sched.lr(0) - min_lr).abs() < 1e-8, "start={}", sched.lr(0));

        // Peak at warmup end (step 30)
        let peak = sched.lr(30);
        assert!((peak - 0.01).abs() < 1e-6, "peak={}", peak);

        // End near min
        let end = sched.lr(100);
        assert!((end - min_lr).abs() < 1e-6, "end={}", end);
    }

    #[test]
    fn test_one_cycle_lr_monotonic_warmup() {
        let sched = OneCycleLR::new(0.01, 100);
        let mut prev = 0.0;
        for step in 0..=30 {
            let lr = sched.lr(step);
            assert!(lr >= prev, "LR should increase during warmup: step={}, lr={}, prev={}", step, lr, prev);
            prev = lr;
        }
    }

    #[test]
    fn test_one_cycle_lr_monotonic_decay() {
        let sched = OneCycleLR::new(0.01, 100);
        let mut prev = f64::MAX;
        for step in 30..=100 {
            let lr = sched.lr(step);
            assert!(lr <= prev + 1e-10, "LR should decrease during decay: step={}, lr={}, prev={}", step, lr, prev);
            prev = lr;
        }
    }

    #[test]
    fn test_one_cycle_lr_custom_warmup() {
        let sched = OneCycleLR::with_warmup_frac(0.01, 100, 0.1);
        // Peak at step 10
        let peak = sched.lr(10);
        assert!((peak - 0.01).abs() < 1e-6, "peak={}", peak);
    }

    #[test]
    fn test_one_cycle_lr_scheduler_trait() {
        let sched = OneCycleLR::new(0.01, 100);
        let s: &dyn Scheduler = &sched;
        assert!(s.lr(30) > s.lr(0));
    }

    #[test]
    fn test_plateau_scheduler() {
        let mut sched = PlateauScheduler::new(0.1, 3, 0.5, 1e-6);
        assert!((sched.observe(1.0) - 0.1).abs() < 1e-10);
        assert!((sched.observe(1.1) - 0.1).abs() < 1e-10);
        assert!((sched.observe(1.2) - 0.1).abs() < 1e-10);
        // 3 epochs without improvement -> decay
        assert!((sched.observe(1.3) - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_warmup_cosine() {
        let inner = CosineScheduler::new(0.1, 0.001, 90);
        let sched = WarmupScheduler::new(inner, 0.1, 10);
        // During warmup: linear ramp
        assert!(sched.lr(0) < 0.02);
        assert!((sched.lr(9) - 0.1).abs() < 1e-5);
        // After warmup: delegates to cosine (step 0 of inner)
        assert!((sched.lr(10) - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_cyclic_lr() {
        let sched = CyclicLR::new(0.001, 0.01, 10);
        // At step 0: base_lr
        assert!((sched.lr(0) - 0.001).abs() < 1e-6);
        // At step 5 (middle of up phase): midpoint
        assert!((sched.lr(5) - 0.0055).abs() < 1e-4);
        // At step 10: max_lr
        assert!((sched.lr(10) - 0.01).abs() < 1e-6);
        // At step 15 (middle of down phase)
        assert!((sched.lr(15) - 0.0055).abs() < 1e-4);
        // At step 20: back to base (full cycle)
        assert!((sched.lr(20) - 0.001).abs() < 1e-6);
    }
}
