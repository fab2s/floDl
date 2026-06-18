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
pub type OuterOptimizerFactory =
    Arc<dyn Fn() -> Box<dyn OuterOptimizer> + Send + Sync>;

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
    fn outer_step(
        &mut self,
        prev_global: &[Tensor],
        consensus: &[Tensor],
    ) -> Result<Vec<Tensor>>;
}

/// Identity outer optimizer: `new_global = consensus`. Stateless, no
/// momentum, no checkpoint artifact. This is the default and reproduces the
/// pre-outer-optimizer averaging behavior exactly.
#[derive(Debug, Default, Clone, Copy)]
pub struct OuterAvg;

impl OuterOptimizer for OuterAvg {
    fn outer_step(
        &mut self,
        _prev_global: &[Tensor],
        consensus: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Element-wise identity: return the consensus unchanged. The clones
        // are shallow (shared storage); the caller serializes them read-only
        // and never mutates, so sharing storage is safe here.
        Ok(consensus.to_vec())
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
                self.seen_params_this_window = true;
                let consensus = round_frame_to_tensors(&frame)?;
                // First window: no prior anchor — use the consensus, so the
                // outer gradient is zero and the step is a no-op for any
                // well-behaved variant.
                let prev = self.prev_global.take().unwrap_or_else(|| consensus.clone());
                let new_global = self.opt.outer_step(&prev, &consensus)?;
                self.prev_global = Some(new_global.clone());
                let refs: Vec<&Tensor> = new_global.iter().collect();
                tensors_to_round_frame(&refs)
            }
            RoundKind::Model => Ok(frame),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let params_frame = tensors_to_round_frame(&[&p0, &p1]).unwrap();
        let buffers_frame = tensors_to_round_frame(&[&buf]).unwrap();
        let mut control_frame = tensors_to_round_frame(&[&p0]).unwrap();
        control_frame.kind = RoundKind::Control;

        let mut stepper = OuterStepper::new(Box::new(OuterAvg));

        // Window: Control -> params -> buffers.
        let c_out = stepper.process_frame(control_frame.clone()).unwrap();
        assert_eq!(c_out, control_frame, "Control frame must pass through");

        let p_out = stepper.process_frame(params_frame.clone()).unwrap();
        assert_eq!(p_out, params_frame, "params frame must be byte-identical under OuterAvg");

        let b_out = stepper.process_frame(buffers_frame.clone()).unwrap();
        assert_eq!(b_out, buffers_frame, "buffers frame must pass through");

        // Second window confirms prev_global tracking does not corrupt
        // identity output.
        stepper.process_frame(control_frame).unwrap();
        let p_out2 = stepper.process_frame(params_frame.clone()).unwrap();
        assert_eq!(p_out2, params_frame, "params still byte-identical second window");
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
