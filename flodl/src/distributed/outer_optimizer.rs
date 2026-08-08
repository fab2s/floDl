//! Outer optimizer: a pluggable step applied to the **consensus** between
//! reduce and broadcast, at the same tier as the convergence guard.
//!
//! The cluster path param-averages (Local-SGD / EASGD): each window the
//! cohort produces a single work-weighted consensus model. An *outer*
//! optimizer transforms that consensus into the new global before it is
//! scattered back to the ranks, giving the cohort a **canonical global**
//! optimizer state (e.g. SlowMo's slow momentum, DiLoCo's outer Nesterov).
//! That state is the same on every rank by construction, so unlike the
//! per-rank inner momentum it is checkpointable and survives a resume
//! faithfully.
//!
//! # Averaged-consensus form (not per-worker deltas)
//!
//! [`OuterOptimizer::outer_step`] consumes the consensus the reduce
//! *already* produces — the sum-and-count work-weighted average — plus the
//! previous global. The outer gradient is `g = prev_global - consensus`,
//! which equals `mean_k(prev_global - theta_k)` under that weighting, so no
//! per-rank state is needed and the step composes with the relay's per-host
//! fold. (An earlier sketch took per-worker deltas; that does not compose
//! with the sum-and-count relay, so the consensus form is the one wired.)
//!
//! # Two backends, one trait
//!
//! The trait is backend-agnostic; only *where* it runs and *where* its
//! momentum lives differ:
//!
//! - **CPU**: the consensus is forged host-side in the controller reduce
//!   thread, so the step runs once at the controller on the averaged frame
//!   before scatter; the momentum is a single host buffer.
//! - **NCCL**: the in-place AllReduce leaves the consensus on every rank's
//!   GPU, so the step runs replicated, once per rank, on identical inputs
//!   (deterministic op → identical result → cohort stays in lock-step with
//!   no extra collective); the momentum is a replicated GPU buffer per rank.
//!
//! The selector is therefore instantiated **per-site**: once at the
//! controller for CPU, once per rank for NCCL.
//!
//! # Momentum-bearing variants step parameters only
//!
//! The outer momentum is one buffer per *parameter*. Buffers (BatchNorm
//! running stats etc.) are not optimized and must pass through unchanged —
//! a momentum-bearing variant must only step the parameter set. [`OuterAvg`]
//! is element-wise identity, so the distinction is moot for it; it is the
//! default and reproduces today's behavior byte-for-byte.

use std::sync::Arc;

use crate::distributed::controller::{RoundFrame, RoundKind};
use crate::distributed::cpu_reduce::{round_frame_to_tensors, tensors_to_round_frame};
use crate::tensor::{Result, Tensor};

/// Per-site factory for an [`OuterOptimizer`]. The selector ships as a
/// factory (not a built instance) because the step runs once per site —
/// once at the controller on the CPU backend, once per rank on NCCL — and
/// each site owns its own instance (replicated momentum on NCCL). A boxed
/// trait object keeps the factory off the [`crate::DdpBuilder`] type
/// parameters (it is not generic in the user's model/optimizer).
pub type OuterOptimizerFactory = Arc<dyn Fn() -> Box<dyn OuterOptimizer> + Send + Sync>;

/// A step applied to the work-weighted **consensus** between reduce and
/// broadcast, transforming it into the new global parameters.
///
/// Implementations hold any momentum internally. The default [`OuterAvg`]
/// is a stateless identity passthrough (`new_global = consensus`), exactly
/// today's averaging behavior.
///
/// See the [module docs](self) for the averaged-consensus form, the two
/// backends, and the parameters-only rule for momentum-bearing variants.
pub trait OuterOptimizer: Send {
    /// Transform `consensus` (this window's work-weighted average) into the
    /// new global parameters, given `prev_global` (the global adopted at the
    /// end of the previous window). Momentum is updated internally.
    ///
    /// `prev_global` and `consensus` are parallel slices: same length, same
    /// per-tensor shape, one entry per model **parameter** (buffers are not
    /// passed through this trait). The returned vector has the same shape.
    ///
    /// On the very first window there is no prior anchor; the caller passes
    /// `consensus` as `prev_global`, so a well-behaved variant returns
    /// `consensus` unchanged (zero outer gradient) and seeds its momentum at
    /// zero.
    fn outer_step(&mut self, prev_global: &[Tensor], consensus: &[Tensor]) -> Result<Vec<Tensor>>;

    /// Snapshot the checkpointable outer state (the slow momentum), one
    /// tensor per model parameter in `Module::parameters()` order, for
    /// `<stem>.outer.fdl`. `None` for stateless variants ([`OuterAvg`]),
    /// which write no artifact. Returns clones (shallow); the caller
    /// serializes them synchronously, so the live state may be overwritten by
    /// the next window without racing the detached write.
    fn checkpoint_state(&self) -> Option<Vec<Tensor>> {
        None
    }

    /// Restore outer state from `<stem>.outer.fdl` on resume. `state` is
    /// positional, one tensor per parameter (`Module::parameters()` order).
    /// Stateless variants ignore it.
    fn load_checkpoint_state(&mut self, _state: Vec<Tensor>) -> Result<()> {
        Ok(())
    }

    /// Whether this variant requires **disposable inner optimizer state**:
    /// the worker, each sync, fully overwrites its params with the new global
    /// and **resets its inner optimizer** (clears momentum, step count). This
    /// is DiLoCo's contract — the inner optimizer is restarted every outer
    /// round, which is what makes the *outer* momentum the canonical,
    /// resume-faithful state. `false` (default) keeps the inner loop
    /// continuous ([`OuterAvg`], [`SlowMomentum`]).
    ///
    /// The worker queries its own (per-site) instance for this, so the signal
    /// rides the same factory that selects the variant — no extra config.
    fn resets_inner(&self) -> bool {
        false
    }
}

/// Identity outer optimizer: `new_global = consensus`. Stateless, no
/// momentum, no checkpoint artifact. This is the default and reproduces the
/// pre-outer-optimizer averaging behavior exactly.
#[derive(Debug, Default, Clone, Copy)]
pub struct OuterAvg;

impl OuterOptimizer for OuterAvg {
    fn outer_step(&mut self, _prev_global: &[Tensor], consensus: &[Tensor]) -> Result<Vec<Tensor>> {
        // Element-wise identity: return the consensus unchanged. The clones
        // are shallow (shared storage); the caller serializes them read-only
        // and never mutates, so sharing storage is safe here.
        Ok(consensus.to_vec())
    }
}

/// SlowMo outer optimizer: heavy-ball slow momentum on the pseudo-gradient.
///
/// Each window the outer (pseudo) gradient is `g = prev_global - consensus`
/// (the drift the inner steps produced, in the averaged-consensus form);
/// the slow momentum accumulates it and the global takes a momentum-SGD
/// step:
///
/// ```text
/// g = prev_global - consensus
/// v = mu * v + g            (slow momentum, persisted across windows)
/// new_global = prev_global - lr * v
/// ```
///
/// The inner optimizer runs **continuously** (no reset between windows), so
/// SlowMo needs no worker-side change — it is purely a transform on the
/// consensus at the outer tier. (DiLoCo's Nesterov variant instead applies
/// `new = prev - lr * (mu*v + g)` and resets the inner optimizer each round;
/// see [`OuterOptimizer`].)
///
/// On the first window the driver passes `prev_global == consensus`, so
/// `g = 0`, the momentum seeds at zero, and the step is a no-op — matching
/// [`OuterAvg`] for that window.
pub struct SlowMomentum {
    /// Outer (slow) learning rate.
    lr: f64,
    /// Outer (slow) momentum coefficient.
    mu: f64,
    /// Per-parameter slow-momentum buffer, persisted across windows. Empty
    /// until the first step (then sized to the parameter count).
    velocity: Vec<Tensor>,
}

impl SlowMomentum {
    /// New SlowMo outer optimizer with slow learning rate `lr` and slow
    /// momentum `mu`. Typical SlowMo settings are `lr ≈ 1.0`, `mu ≈ 0.9`.
    pub fn new(lr: f64, mu: f64) -> Self {
        SlowMomentum {
            lr,
            mu,
            velocity: Vec::new(),
        }
    }
}

impl OuterOptimizer for SlowMomentum {
    fn outer_step(&mut self, prev_global: &[Tensor], consensus: &[Tensor]) -> Result<Vec<Tensor>> {
        let n = consensus.len();
        // A parameter-count change (only on a fresh / mismatched buffer)
        // restarts the momentum from zero. `mu * 0 + g == g`, so the first
        // step just uses `g` directly.
        let fresh = self.velocity.len() != n;
        let mut new_global = Vec::with_capacity(n);
        let mut new_velocity = Vec::with_capacity(n);
        for i in 0..n {
            let g = prev_global[i].sub(&consensus[i])?;
            let v = if fresh {
                g
            } else {
                self.velocity[i].mul_scalar(self.mu)?.add(&g)?
            };
            let step = v.mul_scalar(self.lr)?;
            new_global.push(prev_global[i].sub(&step)?);
            new_velocity.push(v);
        }
        self.velocity = new_velocity;
        Ok(new_global)
    }

    fn checkpoint_state(&self) -> Option<Vec<Tensor>> {
        // No step taken yet (velocity unseeded) => nothing to checkpoint.
        if self.velocity.is_empty() {
            None
        } else {
            Some(self.velocity.clone())
        }
    }

    fn load_checkpoint_state(&mut self, state: Vec<Tensor>) -> Result<()> {
        // Restore the slow-momentum buffer; the next outer_step resumes
        // accumulation from it (a faithful resume, vs re-seeding from zero).
        self.velocity = state;
        Ok(())
    }
}

/// DiLoCo outer optimizer: Nesterov momentum on the pseudo-gradient, paired
/// with **disposable inner optimizer state** (the worker resets its inner
/// optimizer each outer round; see [`OuterOptimizer::resets_inner`]).
///
/// ```text
/// g = prev_global - consensus
/// v = mu * v + g                       (momentum, persisted across windows)
/// new_global = prev_global - lr * (mu * v + g)   (Nesterov look-ahead)
/// ```
///
/// The look-ahead term `mu * v` distinguishes it from [`SlowMomentum`]'s
/// heavy-ball `new = prev - lr * v`. DiLoCo's reference settings are a
/// smaller outer lr (≈0.7) with `mu ≈ 0.9`, run over many inner steps `H`.
/// Because the inner optimizer is reset every round, the *outer* momentum is
/// the canonical optimizer state — checkpointed to `<stem>.outer.fdl` and
/// restored faithfully on resume (unlike a continuous inner optimizer, whose
/// per-rank state has no consensus to save).
///
/// First window: the driver passes `prev_global == consensus`, so `g = 0`,
/// momentum seeds at zero, and the step is a no-op.
pub struct NesterovMomentum {
    lr: f64,
    mu: f64,
    velocity: Vec<Tensor>,
}

impl NesterovMomentum {
    /// New DiLoCo outer optimizer with outer learning rate `lr` and outer
    /// momentum `mu`. Reference DiLoCo settings: `lr ≈ 0.7`, `mu ≈ 0.9`.
    pub fn new(lr: f64, mu: f64) -> Self {
        NesterovMomentum {
            lr,
            mu,
            velocity: Vec::new(),
        }
    }
}

impl OuterOptimizer for NesterovMomentum {
    fn outer_step(&mut self, prev_global: &[Tensor], consensus: &[Tensor]) -> Result<Vec<Tensor>> {
        let n = consensus.len();
        let fresh = self.velocity.len() != n;
        let mut new_global = Vec::with_capacity(n);
        let mut new_velocity = Vec::with_capacity(n);
        for i in 0..n {
            let g = prev_global[i].sub(&consensus[i])?;
            // v = mu * v_prev + g  (mu*0 + g = g on a fresh buffer). On the
            // fresh branch `copy()` makes an independent deep copy of g
            // (NOT a shared-storage shallow clone) so the buffer and g
            // stay distinct.
            let v = if fresh {
                g.copy()?
            } else {
                self.velocity[i].mul_scalar(self.mu)?.add(&g)?
            };
            // Nesterov look-ahead: step by (mu * v + g), not v.
            let look_ahead = v.mul_scalar(self.mu)?.add(&g)?;
            let step = look_ahead.mul_scalar(self.lr)?;
            new_global.push(prev_global[i].sub(&step)?);
            new_velocity.push(v);
        }
        self.velocity = new_velocity;
        Ok(new_global)
    }

    fn checkpoint_state(&self) -> Option<Vec<Tensor>> {
        if self.velocity.is_empty() {
            None
        } else {
            Some(self.velocity.clone())
        }
    }

    fn load_checkpoint_state(&mut self, state: Vec<Tensor>) -> Result<()> {
        self.velocity = state;
        Ok(())
    }

    fn resets_inner(&self) -> bool {
        true
    }
}

/// Controller-side driver that applies an [`OuterOptimizer`] to the CPU
/// reduce stream, keeping `controller.rs` a model-agnostic byte reducer
/// (the same separation the consensus forge keeps).
///
/// One CPU sync window issues, in order, a `Control` count-gather, then a
/// `Model` parameters reduce (skipped when the whole cohort was idle), then
/// a `Model` buffers reduce (skipped when there are no buffers / the cohort
/// was idle). The outer step applies to **parameters only**: it transforms
/// the parameters frame and passes the `Control` and buffers frames through
/// untouched.
///
/// The parameters frame is identified as the **first `Model` frame of each
/// window**; the window is opened by the `Control` count-gather, which
/// resets the per-window flag. This driver holds `prev_global` (the
/// parameters global the cohort adopted at the end of the previous window)
/// across windows so a momentum-bearing variant has its anchor; on the
/// first window there is no anchor, so the consensus is used as
/// `prev_global` (zero outer gradient).
pub struct OuterStepper {
    opt: Box<dyn OuterOptimizer>,
    /// Parameters global adopted at the end of the previous window. `None`
    /// until the first parameters frame is processed.
    prev_global: Option<Vec<Tensor>>,
    /// Whether the parameters frame of the current window has been seen.
    /// Reset by each `Control` frame.
    seen_params_this_window: bool,
}

impl OuterStepper {
    /// Wrap an outer optimizer instance for the CPU reduce stream.
    pub fn new(opt: Box<dyn OuterOptimizer>) -> Self {
        OuterStepper {
            opt,
            prev_global: None,
            seen_params_this_window: false,
        }
    }

    /// Process one averaged reduce frame, returning the frame to scatter.
    ///
    /// `Control` frames reset the per-window flag and pass through. The
    /// first `Model` frame of a window is the parameters frame: it is
    /// materialized to CPU tensors, stepped, and re-serialized. Subsequent
    /// `Model` frames (buffers) pass through. With [`OuterAvg`] the step is
    /// element-wise identity, so the materialize → step → serialize
    /// round-trip reproduces the input frame byte-for-byte.
    pub fn process_frame(&mut self, frame: RoundFrame) -> Result<RoundFrame> {
        match frame.kind {
            RoundKind::Control => {
                self.seen_params_this_window = false;
                Ok(frame)
            }
            RoundKind::Model if !self.seen_params_this_window => {
                // Zero accepted mass: a degenerate all-idle round whose
                // tensors are a meaningless zero sum. Pass it through so
                // ranks see `weight == 0` and keep local state; do NOT
                // step the outer optimizer on it.
                if !crate::distributed::realized_work::is_realized(frame.weight) {
                    return Ok(frame);
                }
                self.seen_params_this_window = true;
                let weight = frame.weight;
                // Scatter in the dtype the round arrived in (the wire
                // dtype is a rank-side choice; the stepper must not
                // change the frame schema mid-stream).
                let wire_dtype = frame
                    .tensors
                    .first()
                    .map_or(crate::distributed::controller::DTYPE_F32, |t| t.dtype);
                let consensus = round_frame_to_tensors(&frame)?;
                // First window: no prior anchor — use the consensus, so the
                // outer gradient is zero and the step is a no-op for any
                // well-behaved variant.
                let prev = self.prev_global.take().unwrap_or_else(|| consensus.clone());
                let new_global = self.opt.outer_step(&prev, &consensus)?;
                self.prev_global = Some(new_global.clone());
                let refs: Vec<&Tensor> = new_global.iter().collect();
                // Preserve the realized-work mass on the rebuilt frame:
                // ranks treat `weight == 0` as "keep local state", so
                // dropping it would turn every outer step into a
                // cohort-wide no-op adopt.
                let mut stepped = tensors_to_round_frame(&refs, wire_dtype)?;
                stepped.weight = weight;
                Ok(stepped)
            }
            RoundKind::Model => Ok(frame),
        }
    }

    /// Snapshot the outer optimizer's checkpointable state (delegates to the
    /// variant). `None` for stateless variants ([`OuterAvg`]), so no
    /// `<stem>.outer.fdl` is written.
    pub fn checkpoint_state(&self) -> Option<Vec<Tensor>> {
        self.opt.checkpoint_state()
    }

    /// Restore the outer optimizer's state on resume (delegates to the
    /// variant). `prev_global` stays `None` — the first post-resume window
    /// re-anchors on that window's consensus (zero outer gradient), while
    /// the restored momentum carries the accumulated drift.
    pub fn load_checkpoint_state(&mut self, state: Vec<Tensor>) -> Result<()> {
        self.opt.load_checkpoint_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::controller::{DTYPE_BF16, DTYPE_F32};
    use crate::tensor::{Device, test_device};

    fn t(vals: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(vals, shape, test_device()).unwrap()
    }

    #[test]
    fn outer_avg_is_identity() {
        let mut opt = OuterAvg;
        let prev = vec![t(&[1.0, 2.0, 3.0], &[3]), t(&[4.0, 5.0], &[2])];
        let consensus = vec![t(&[10.0, 20.0, 30.0], &[3]), t(&[40.0, 50.0], &[2])];

        let out = opt.outer_step(&prev, &consensus).unwrap();

        assert_eq!(out.len(), consensus.len());
        assert_eq!(out[0].to_f32_vec().unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(out[1].to_f32_vec().unwrap(), vec![40.0, 50.0]);
    }

    #[test]
    fn outer_avg_ignores_prev_global() {
        // First-window convention: prev_global == consensus. Identity must
        // return consensus regardless of what prev_global holds.
        let mut opt = OuterAvg;
        let consensus = vec![t(&[7.0, 8.0], &[2])];
        let bogus_prev = vec![t(&[0.0, 0.0], &[2])];

        let out = opt.outer_step(&bogus_prev, &consensus).unwrap();
        assert_eq!(out[0].to_f32_vec().unwrap(), vec![7.0, 8.0]);

        let out2 = opt.outer_step(&consensus, &consensus).unwrap();
        assert_eq!(out2[0].to_f32_vec().unwrap(), vec![7.0, 8.0]);
    }

    #[test]
    fn stepper_with_outer_avg_is_byte_identical() {
        // The materialize -> step -> serialize round-trip with OuterAvg must
        // reproduce the parameters frame byte-for-byte, and pass Control /
        // buffers frames through unchanged.
        let p0 = t(&[1.25, -2.5, 3.75], &[3]);
        let p1 = t(&[4.0, 5.0], &[2]);
        let buf = t(&[9.0, 8.0], &[2]);

        let mut params_frame = tensors_to_round_frame(&[&p0, &p1], DTYPE_F32).unwrap();
        params_frame.weight = 1.0; // realized mass: the stepper skips zero-mass frames
        let buffers_frame = tensors_to_round_frame(&[&buf], DTYPE_F32).unwrap();
        let mut control_frame = tensors_to_round_frame(&[&p0], DTYPE_F32).unwrap();
        control_frame.kind = RoundKind::Control;

        let mut stepper = OuterStepper::new(Box::new(OuterAvg));

        // Window: Control -> params -> buffers.
        let c_out = stepper.process_frame(control_frame.clone()).unwrap();
        assert_eq!(c_out, control_frame, "Control frame must pass through");

        let p_out = stepper.process_frame(params_frame.clone()).unwrap();
        assert_eq!(
            p_out, params_frame,
            "params frame must be byte-identical under OuterAvg"
        );

        let b_out = stepper.process_frame(buffers_frame.clone()).unwrap();
        assert_eq!(b_out, buffers_frame, "buffers frame must pass through");

        // Second window confirms prev_global tracking does not corrupt
        // identity output.
        stepper.process_frame(control_frame).unwrap();
        let p_out2 = stepper.process_frame(params_frame.clone()).unwrap();
        assert_eq!(
            p_out2, params_frame,
            "params still byte-identical second window"
        );
    }

    /// A bf16 params frame steps in f32 and re-serializes as bf16 — the
    /// stepper never changes the frame schema mid-stream. With OuterAvg
    /// the identity round trip is byte-exact (bf16 → f32 → bf16 RNE is
    /// the identity on bf16-valued inputs).
    #[test]
    fn stepper_preserves_bf16_frame_dtype() {
        let p = t(&[1.25, -2.5, 3.75], &[3]);
        let mut params_frame = tensors_to_round_frame(&[&p], DTYPE_BF16).unwrap();
        params_frame.weight = 1.0;
        assert_eq!(params_frame.tensors[0].dtype, DTYPE_BF16);

        let mut stepper = OuterStepper::new(Box::new(OuterAvg));
        let out = stepper.process_frame(params_frame.clone()).unwrap();
        assert_eq!(
            out, params_frame,
            "OuterAvg identity is byte-exact in bf16 too"
        );
    }

    #[test]
    fn slow_momentum_heavy_ball_math() {
        // lr=0.5, mu=0.9. Drive prev/consensus by hand, mimicking the driver
        // (prev = last new_global).
        let mut opt = SlowMomentum::new(0.5, 0.9);

        // First window: prev == consensus -> g=0 -> no-op, momentum seeded 0.
        let w1 = opt
            .outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[1.0, 2.0], &[2])])
            .unwrap();
        assert_eq!(w1[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);

        // Window 2: g=[0.5,1.0], v=[0.5,1.0], step=lr*v=[0.25,0.5],
        // new = prev - step = [0.75, 1.5].
        let w2 = opt
            .outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[0.5, 1.0], &[2])])
            .unwrap();
        let w2v = w2[0].to_f32_vec().unwrap();
        assert!(
            (w2v[0] - 0.75).abs() < 1e-6 && (w2v[1] - 1.5).abs() < 1e-6,
            "got {w2v:?}"
        );

        // Window 3: g=[0.05,0.1], v=0.9*[0.5,1.0]+g=[0.5,1.0],
        // step=[0.25,0.5], new=[0.75,1.5]-step=[0.5,1.0]. Confirms momentum
        // carried across windows (the 0.9*v term).
        let w3 = opt
            .outer_step(&[t(&[0.75, 1.5], &[2])], &[t(&[0.7, 1.4], &[2])])
            .unwrap();
        let w3v = w3[0].to_f32_vec().unwrap();
        assert!(
            (w3v[0] - 0.5).abs() < 1e-6 && (w3v[1] - 1.0).abs() < 1e-6,
            "got {w3v:?}"
        );
    }

    #[test]
    fn stepper_slow_momentum_steps_params_only() {
        // Through the driver: parameters are stepped, buffers pass through.
        let mut control = tensors_to_round_frame(&[&t(&[0.0], &[1])], DTYPE_F32).unwrap();
        control.kind = RoundKind::Control;
        let buffers = tensors_to_round_frame(&[&t(&[9.0, 8.0], &[2])], DTYPE_F32).unwrap();

        let mut stepper = OuterStepper::new(Box::new(SlowMomentum::new(0.5, 0.9)));

        // Window 1: params1=[2,4] (prev==consensus first window -> unchanged),
        // buffers untouched.
        stepper.process_frame(control.clone()).unwrap();
        let mut p1 = tensors_to_round_frame(&[&t(&[2.0, 4.0], &[2])], DTYPE_F32).unwrap();
        p1.weight = 1.0; // realized mass: the stepper skips zero-mass frames
        let p1_out = stepper.process_frame(p1.clone()).unwrap();
        assert_eq!(p1_out, p1, "first-window params unchanged (g=0)");
        let b1_out = stepper.process_frame(buffers.clone()).unwrap();
        assert_eq!(b1_out, buffers, "buffers pass through");

        // Window 2: consensus=[1,2], prev=[2,4]. g=[1,2], v=[1,2],
        // step=0.5*[1,2]=[0.5,1], new=[1.5,3]. Buffers still pass through.
        stepper.process_frame(control).unwrap();
        let mut p2 = tensors_to_round_frame(&[&t(&[1.0, 2.0], &[2])], DTYPE_F32).unwrap();
        p2.weight = 1.0;
        let p2_out = stepper.process_frame(p2).unwrap();
        let stepped = round_frame_to_tensors(&p2_out).unwrap()[0]
            .to_f32_vec()
            .unwrap();
        assert!(
            (stepped[0] - 1.5).abs() < 1e-6 && (stepped[1] - 3.0).abs() < 1e-6,
            "second-window params stepped: got {stepped:?}"
        );
        let b2_out = stepper.process_frame(buffers.clone()).unwrap();
        assert_eq!(
            b2_out, buffers,
            "buffers still pass through after a real step"
        );
    }

    #[test]
    fn outer_avg_checkpoint_state_is_none() {
        // Stateless: no momentum, so no <stem>.outer.fdl artifact.
        let opt = OuterAvg;
        assert!(opt.checkpoint_state().is_none());
    }

    #[test]
    fn slow_momentum_checkpoint_round_trip_is_faithful() {
        // Warm a SlowMomentum, snapshot its momentum, restore into a fresh
        // instance, and confirm the next outer_step matches the warmed one's
        // (the resume is faithful, vs re-seeding velocity from zero).
        let mut warm = SlowMomentum::new(0.5, 0.9);
        // Two windows to build non-trivial momentum.
        warm.outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[1.0, 2.0], &[2])])
            .unwrap();
        warm.outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[0.5, 1.0], &[2])])
            .unwrap();

        let saved = warm
            .checkpoint_state()
            .expect("has momentum after stepping");
        let mut resumed = SlowMomentum::new(0.5, 0.9);
        resumed.load_checkpoint_state(saved).unwrap();

        // Identical next step from identical inputs => momentum was carried.
        let prev = [t(&[0.75, 1.5], &[2])];
        let cons = [t(&[0.7, 1.4], &[2])];
        let a = warm.outer_step(&prev, &cons).unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let b = resumed.outer_step(&prev, &cons).unwrap()[0]
            .to_f32_vec()
            .unwrap();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "param[{i}]: resumed {y} != warmed {x}"
            );
        }

        // Sanity: a from-zero instance would differ on this step (proves the
        // round-trip carried real state, not nothing).
        let mut fresh = SlowMomentum::new(0.5, 0.9);
        let c = fresh.outer_step(&prev, &cons).unwrap()[0]
            .to_f32_vec()
            .unwrap();
        assert!(
            (a[0] - c[0]).abs() > 1e-6 || (a[1] - c[1]).abs() > 1e-6,
            "warmed and from-zero should differ, else the test is vacuous"
        );
    }

    #[test]
    fn nesterov_resets_inner_others_dont() {
        assert!(
            NesterovMomentum::new(0.7, 0.9).resets_inner(),
            "DiLoCo resets inner"
        );
        assert!(
            !SlowMomentum::new(0.5, 0.7).resets_inner(),
            "SlowMo keeps inner continuous"
        );
        assert!(!OuterAvg.resets_inner(), "OuterAvg keeps inner continuous");
    }

    #[test]
    fn nesterov_look_ahead_math() {
        // lr=0.5, mu=0.9.
        let mut opt = NesterovMomentum::new(0.5, 0.9);

        // First window: prev==consensus -> g=0 -> no-op.
        let w1 = opt
            .outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[1.0, 2.0], &[2])])
            .unwrap();
        assert_eq!(w1[0].to_f32_vec().unwrap(), vec![1.0, 2.0]);

        // Window 2: prev=[1,2], consensus=[0.5,1.0]. g=[0.5,1.0].
        // fresh velocity -> v=g=[0.5,1.0]. look_ahead = mu*v+g =
        // 0.9*[0.5,1.0]+[0.5,1.0] = [0.95,1.9]. step = lr*look_ahead =
        // [0.475,0.95]. new = prev-step = [0.525,1.05].
        let w2 = opt
            .outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[0.5, 1.0], &[2])])
            .unwrap();
        let w2v = w2[0].to_f32_vec().unwrap();
        assert!(
            (w2v[0] - 0.525).abs() < 1e-6 && (w2v[1] - 1.05).abs() < 1e-6,
            "got {w2v:?}"
        );

        // Window 3: prev=[0.525,1.05], consensus=[0.5,1.0]. g=[0.025,0.05].
        // v = 0.9*[0.5,1.0]+[0.025,0.05] = [0.475,0.95]. look_ahead =
        // 0.9*[0.475,0.95]+[0.025,0.05] = [0.4525,0.905]. step =
        // [0.22625,0.4525]. new = [0.525,1.05]-step = [0.29875,0.5975].
        // (confirms momentum carried across windows via the 0.9*v term).
        let w3 = opt
            .outer_step(&[t(&[0.525, 1.05], &[2])], &[t(&[0.5, 1.0], &[2])])
            .unwrap();
        let w3v = w3[0].to_f32_vec().unwrap();
        assert!(
            (w3v[0] - 0.29875).abs() < 1e-5 && (w3v[1] - 0.5975).abs() < 1e-5,
            "got {w3v:?}"
        );
    }

    #[test]
    fn nesterov_checkpoint_round_trip() {
        let mut warm = NesterovMomentum::new(0.5, 0.9);
        warm.outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[1.0, 2.0], &[2])])
            .unwrap();
        warm.outer_step(&[t(&[1.0, 2.0], &[2])], &[t(&[0.5, 1.0], &[2])])
            .unwrap();
        let saved = warm.checkpoint_state().expect("has momentum");
        let mut resumed = NesterovMomentum::new(0.5, 0.9);
        resumed.load_checkpoint_state(saved).unwrap();
        let prev = [t(&[0.525, 1.05], &[2])];
        let cons = [t(&[0.5, 1.0], &[2])];
        let a = warm.outer_step(&prev, &cons).unwrap()[0]
            .to_f32_vec()
            .unwrap();
        let b = resumed.outer_step(&prev, &cons).unwrap()[0]
            .to_f32_vec()
            .unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "resumed {y} != warmed {x}");
        }
    }

    #[test]
    fn outer_avg_as_trait_object() {
        // The controller holds a `Box<dyn OuterOptimizer>`; confirm dyn
        // dispatch works and OuterAvg is object-safe.
        let mut opt: Box<dyn OuterOptimizer> = Box::new(OuterAvg);
        let c = vec![t(&[1.5], &[1])];
        let out = opt.outer_step(&c, &c).unwrap();
        assert_eq!(out[0].to_f32_vec().unwrap(), vec![1.5]);
        // Keep Device imported (CPU fallback parity with other dist tests).
        let _ = Device::CPU;
    }
}
