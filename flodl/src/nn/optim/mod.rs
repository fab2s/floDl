//! Optimizers and optimizer state serialization.
//!
//! Each optimizer lives in its own private submodule and is re-exported here:
//! [`SGD`], [`Adam`] (also hosts [`AdamW`]), [`RMSprop`], [`Adagrad`],
//! [`RAdam`], [`NAdam`]. The shared [`Optimizer`] and [`Stateful`] traits plus
//! per-group LR metadata live here.

use std::io::{Read, Write};

use crate::tensor::Result;

mod sgd;
mod adam;
mod rmsprop;
mod adagrad;
mod radam;
mod nadam;

pub use sgd::{SGD, SGDBuilder};
pub use adam::{Adam, AdamBuilder, AdamW, AdamWBuilder};
pub use rmsprop::{RMSprop, RMSpropBuilder};
pub use adagrad::{Adagrad, AdagradBuilder};
pub use radam::RAdam;
pub use nadam::NAdam;

/// Optimizer trait: step, zero gradients, and adjust learning rate.
pub trait Optimizer {
    /// Perform a single optimization step using accumulated gradients.
    fn step(&mut self) -> Result<()>;
    /// Reset all parameter gradients to zero.
    fn zero_grad(&self);
    /// Current learning rate (group 0 for grouped optimizers).
    fn lr(&self) -> f64;
    /// Update the learning rate (all groups if grouped).
    fn set_lr(&mut self, lr: f64);
    /// Set learning rate for a specific parameter group (0-indexed).
    /// Falls back to `set_lr` for single-group optimizers.
    fn set_group_lr(&mut self, _group: usize, lr: f64) {
        self.set_lr(lr);
    }
    /// Multiply the learning rate by a factor (all groups).
    fn scale_lr(&mut self, factor: f64) {
        self.set_lr(self.lr() * factor);
    }
    /// Reset all internal optimizer state — momentum / velocity buffers and
    /// step counters — to the fresh, pre-training values, as if the
    /// optimizer were just constructed over the same parameters. Learning
    /// rate and hyperparameters (betas, weight decay, etc.) are unchanged,
    /// and the parameter set is untouched.
    ///
    /// Used by the DiLoCo outer-optimizer regime: each round the worker
    /// adopts the new global by full overwrite and discards its inner
    /// optimizer state (the inner optimizer is disposable, so resume is
    /// faithful from the canonical *outer* momentum). The default is a
    /// no-op — stateless optimizers need no override; every stateful
    /// flodl optimizer overrides it to clear its buffers and counters.
    fn reset_state(&mut self) {}

    /// Persist optimizer state (LR + momentum buffers + step counters)
    /// to `path`. Object-safe wrapper around [`Stateful::save_state_file`]
    /// — necessary because `Stateful::save_state` is generic in `W: Write`
    /// and so not dyn-callable. Optimizers that implement [`Stateful`]
    /// override this method with a one-line delegate; those that do not
    /// inherit the default impl which returns an explicit "unsupported"
    /// error so the cluster save flow can log it and move on rather than
    /// silently producing an empty `.optim` file.
    ///
    /// Used by the cluster save-on-unrecoverable-failure flow to write
    /// `<save_path>.optim` alongside the model + meta sidecars; see
    /// [`crate::distributed::CheckpointBundle`].
    fn save_state_to(&self, _path: &str) -> Result<()> {
        Err(crate::tensor::TensorError::new(
            "Optimizer::save_state_to: this optimizer does not yet \
             implement Stateful; optimizer state cannot be persisted. \
             Open a follow-up to add Stateful for this optimizer.",
        ))
    }
}

/// Per-group learning rate metadata. Private to `optim`; submodules inherit access.
struct GroupMeta {
    lr: f64,
    range: std::ops::Range<usize>,
}

/// Save/load training state (learning rates, momentum buffers, step counters).
/// Implement for optimizers and other stateful training components.
pub trait Stateful {
    /// Serialize optimizer state (lr, momentum buffers, etc.) to a writer.
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()>;
    /// Restore optimizer state from a reader.
    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()>;

    /// Save state to a file. Uses gzip compression if path ends with `.gz`.
    fn save_state_file(&self, path: &str) -> Result<()> {
        let f = std::fs::File::create(path).map_err(|e| {
            crate::tensor::TensorError::new(&format!("io: {}", e))
        })?;
        if path.ends_with(".gz") {
            let mut w = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            self.save_state(&mut w)?;
            w.finish().map_err(|e| {
                crate::tensor::TensorError::new(&format!("io: {}", e))
            })?;
            Ok(())
        } else {
            let mut w = std::io::BufWriter::new(f);
            self.save_state(&mut w)
        }
    }

    /// Load state from a file. Detects gzip from `.gz` extension.
    fn load_state_file(&mut self, path: &str) -> Result<()> {
        let f = std::fs::File::open(path).map_err(|e| {
            crate::tensor::TensorError::new(&format!("io: {}", e))
        })?;
        if path.ends_with(".gz") {
            let mut r = flate2::read::GzDecoder::new(f);
            self.load_state(&mut r)
        } else {
            let mut r = std::io::BufReader::new(f);
            self.load_state(&mut r)
        }
    }
}

#[cfg(test)]
mod test_helpers {
    use crate::nn::parameter::Parameter;
    use crate::tensor::{Tensor, TensorOptions};

    pub(super) fn make_param(name: &str, shape: &[i64]) -> Parameter {
        let t = Tensor::randn(shape, TensorOptions {
            dtype: crate::tensor::DType::Float32,
            device: crate::tensor::test_device(),
        }).unwrap();
        Parameter::new(t, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_helpers::make_param;
    use crate::nn::parameter::Parameter;

    #[test]
    fn test_empty_params_optimizers_no_panic() {
        let empty: &[Parameter] = &[];

        let mut adam = Adam::new(empty, 0.001);
        adam.step().unwrap();
        adam.zero_grad();

        let mut sgd = SGD::new(empty, 0.01, 0.9);
        sgd.step().unwrap();
        sgd.zero_grad();

        let mut adamw = AdamW::new(empty, 0.001, 0.01);
        adamw.step().unwrap();
        adamw.zero_grad();

        let mut rmsprop = RMSprop::new(empty, 0.01);
        rmsprop.step().unwrap();
        rmsprop.zero_grad();

        let mut adagrad = Adagrad::new(empty, 0.01);
        adagrad.step().unwrap();
        adagrad.zero_grad();

        let mut radam = RAdam::new(empty, 0.01);
        radam.step().unwrap();
        radam.zero_grad();

        let mut nadam = NAdam::new(empty, 0.01);
        nadam.step().unwrap();
        nadam.zero_grad();
    }

    #[test]
    fn test_step_after_zero_grad_on_fresh_optimizer() {
        let p = make_param("w", &[3, 2]);
        let mut adam = Adam::new(std::slice::from_ref(&p), 0.001);
        let mut sgd = SGD::new(std::slice::from_ref(&p), 0.01, 0.9);

        // zero_grad then step on a fresh optimizer (no backward ever called)
        adam.zero_grad();
        adam.step().unwrap();
        sgd.zero_grad();
        sgd.step().unwrap();

        let vals = p.variable.data().to_f32_vec().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "param[{}] should be finite after step-without-backward: {}", i, v);
        }
    }

    #[test]
    fn test_set_lr_all_optimizers() {
        let p = make_param("w", &[2]);

        let mut adam = Adam::new(std::slice::from_ref(&p), 0.001);
        adam.set_lr(0.42);
        assert!((adam.lr() - 0.42).abs() < 1e-12, "Adam set_lr failed");

        let mut sgd = SGD::new(std::slice::from_ref(&p), 0.01, 0.0);
        sgd.set_lr(0.42);
        assert!((sgd.lr() - 0.42).abs() < 1e-12, "SGD set_lr failed");

        let mut adamw = AdamW::new(std::slice::from_ref(&p), 0.001, 0.01);
        adamw.set_lr(0.42);
        assert!((adamw.lr() - 0.42).abs() < 1e-12, "AdamW set_lr failed");

        let mut rmsprop = RMSprop::new(std::slice::from_ref(&p), 0.01);
        rmsprop.set_lr(0.42);
        assert!((rmsprop.lr() - 0.42).abs() < 1e-12, "RMSprop set_lr failed");

        let mut nadam = NAdam::new(std::slice::from_ref(&p), 0.01);
        nadam.set_lr(0.42);
        assert!((nadam.lr() - 0.42).abs() < 1e-12, "NAdam set_lr failed");

        let mut radam = RAdam::new(std::slice::from_ref(&p), 0.01);
        radam.set_lr(0.42);
        assert!((radam.lr() - 0.42).abs() < 1e-12, "RAdam set_lr failed");

        let mut adagrad = Adagrad::new(std::slice::from_ref(&p), 0.01);
        adagrad.set_lr(0.42);
        assert!((adagrad.lr() - 0.42).abs() < 1e-12, "Adagrad set_lr failed");
    }
}
