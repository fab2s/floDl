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

/// Serialize the group table: `count(u32) | (lr(f64), start(i64), end(i64))*`.
/// One codec for every grouped optimizer's `save_state`.
fn write_groups<W: Write>(w: &mut W, groups: &[GroupMeta]) -> Result<()> {
    use crate::nn::checkpoint::{write_f64_le, write_i64_le, write_u32_le};
    write_u32_le(w, groups.len() as u32)?;
    for g in groups {
        write_f64_le(w, g.lr)?;
        write_i64_le(w, g.range.start as i64)?;
        write_i64_le(w, g.range.end as i64)?;
    }
    Ok(())
}

/// Read and validate the group table against the optimizer's param count.
///
/// Every builder's `build()` produces a contiguous ascending partition of
/// `0..n_params`, and `step()` updates ONLY group-covered params — so a table
/// violating that invariant would index out of bounds in a fused kernel at
/// the next `step()`, or worse, silently skip parameters. A corrupt `.optim`
/// file errors here at load instead.
fn read_groups<R: Read>(r: &mut R, n_params: usize, what: &str) -> Result<Vec<GroupMeta>> {
    use crate::nn::checkpoint::{read_f64_le, read_i64_le, read_u32_le};
    let ng = read_u32_le(r)? as usize;
    let mut groups = Vec::with_capacity(ng.min(1024));
    let mut expected_start = 0i64;
    for i in 0..ng {
        let lr = read_f64_le(r)?;
        let start = read_i64_le(r)?;
        let end = read_i64_le(r)?;
        if start != expected_start || end < start || end > n_params as i64 {
            return Err(crate::tensor::TensorError::new(&format!(
                "{what}: corrupt optimizer state: group {i} range {start}..{end} \
                 (expected a contiguous partition of 0..{n_params})"
            )));
        }
        expected_start = end;
        groups.push(GroupMeta {
            lr,
            range: start as usize..end as usize,
        });
    }
    if ng > 0 && expected_start != n_params as i64 {
        return Err(crate::tensor::TensorError::new(&format!(
            "{what}: corrupt optimizer state: groups cover 0..{expected_start} \
             of {n_params} params"
        )));
    }
    Ok(groups)
}

/// Identifies which component wrote a serialized state stream.
///
/// Stored in the state-file header so a file can never be positionally
/// misparsed by the wrong optimizer, and passed to
/// [`migrate_optim_state_file`] to identify pre-header files (the old
/// format carries no self-identification).
///
/// On-disk tags are stable — never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKind {
    /// [`SGD`]
    Sgd,
    /// [`Adam`]
    Adam,
    /// [`AdamW`]
    AdamW,
    /// [`RMSprop`]
    RMSprop,
    /// [`Adagrad`]
    Adagrad,
    /// [`RAdam`]
    RAdam,
    /// [`NAdam`]
    NAdam,
    /// [`crate::nn::GradScaler`]
    GradScaler,
}

impl StateKind {
    fn tag(self) -> u32 {
        match self {
            StateKind::Sgd => 1,
            StateKind::Adam => 2,
            StateKind::AdamW => 3,
            StateKind::RMSprop => 4,
            StateKind::Adagrad => 5,
            StateKind::RAdam => 6,
            StateKind::NAdam => 7,
            StateKind::GradScaler => 8,
        }
    }

    fn from_tag(tag: u32) -> Option<StateKind> {
        Some(match tag {
            1 => StateKind::Sgd,
            2 => StateKind::Adam,
            3 => StateKind::AdamW,
            4 => StateKind::RMSprop,
            5 => StateKind::Adagrad,
            6 => StateKind::RAdam,
            7 => StateKind::NAdam,
            8 => StateKind::GradScaler,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            StateKind::Sgd => "SGD",
            StateKind::Adam => "Adam",
            StateKind::AdamW => "AdamW",
            StateKind::RMSprop => "RMSprop",
            StateKind::Adagrad => "Adagrad",
            StateKind::RAdam => "RAdam",
            StateKind::NAdam => "NAdam",
            StateKind::GradScaler => "GradScaler",
        }
    }
}

/// State-file header magic: `FDLO(4) | version(u32) | kind tag(u32)`.
pub(crate) const STATE_MAGIC: [u8; 4] = *b"FDLO";
/// Current state-file format version.
pub(crate) const STATE_VERSION: u32 = 1;

/// Write the state-file header for `kind`.
fn write_state_header<W: Write>(w: &mut W, kind: StateKind) -> Result<()> {
    use crate::nn::checkpoint::write_u32_le;
    w.write_all(&STATE_MAGIC).map_err(|e| {
        crate::tensor::TensorError::new(&format!("io: {}", e))
    })?;
    write_u32_le(w, STATE_VERSION)?;
    write_u32_le(w, kind.tag())?;
    Ok(())
}

/// Read and validate the state-file header against the loading component.
fn read_state_header<R: Read>(r: &mut R, expected: StateKind, path: &str) -> Result<()> {
    use crate::nn::checkpoint::read_u32_le;
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| {
        crate::tensor::TensorError::new(&format!("{path}: io: {}", e))
    })?;
    if magic != STATE_MAGIC {
        return Err(crate::tensor::TensorError::new(&format!(
            "{path}: not a current flodl optimizer state file (missing FDLO \
             header). If this file was written by an earlier flodl, convert \
             it once with flodl::nn::migrate_optim_state_file(src, dst, \
             StateKind::{:?}) — the old format carries no type tag, so the \
             kind must be supplied.",
            expected
        )));
    }
    let version = read_u32_le(r)?;
    if version > STATE_VERSION {
        return Err(crate::tensor::TensorError::new(&format!(
            "{path}: state file version {version} is newer than this flodl \
             supports (max {STATE_VERSION}) — upgrade flodl to load it"
        )));
    }
    let tag = read_u32_le(r)?;
    let found = StateKind::from_tag(tag).ok_or_else(|| {
        crate::tensor::TensorError::new(&format!(
            "{path}: unknown state kind tag {tag} (corrupt file, or written \
             by a newer flodl)"
        ))
    })?;
    if found != expected {
        return Err(crate::tensor::TensorError::new(&format!(
            "{path}: state file was written by {} but is being loaded into {}",
            found.name(), expected.name()
        )));
    }
    Ok(())
}

/// Save/load training state (learning rates, momentum buffers, step counters).
/// Implement for optimizers and other stateful training components.
pub trait Stateful {
    /// Which component this state stream belongs to — written into the
    /// file header by [`save_state_file`](Stateful::save_state_file) and
    /// validated by [`load_state_file`](Stateful::load_state_file) so a
    /// file can never be positionally misparsed by the wrong optimizer.
    fn state_kind(&self) -> StateKind;

    /// Serialize optimizer state (lr, momentum buffers, etc.) to a writer.
    ///
    /// Raw payload only — the file header is written by
    /// [`save_state_file`](Stateful::save_state_file), so wrapper
    /// optimizers (AdamW) can delegate to their inner payload without
    /// double headers.
    fn save_state<W: Write>(&self, w: &mut W) -> Result<()>;
    /// Restore optimizer state from a raw payload stream (no header).
    fn load_state<R: Read>(&mut self, r: &mut R) -> Result<()>;

    /// Save state to a file. Uses gzip compression if path ends with `.gz`.
    ///
    /// Writes the `FDLO | version | kind` header, then the
    /// [`save_state`](Stateful::save_state) payload.
    ///
    /// Atomic: streams into `<path>.tmp` then renames over the final path, so
    /// a crash mid-write never leaves a torn `<stem>.optim` that resume could
    /// mistake for valid — it leaves a stale `.tmp` instead. This matches the
    /// crash-safety of [`crate::nn::save_checkpoint_file`] (the `.fdl` writer)
    /// so every artifact in an NCCL consensus checkpoint commits atomically.
    /// gzip is chosen from the FINAL extension, not the tmp name.
    fn save_state_file(&self, path: &str) -> Result<()> {
        let kind = self.state_kind();
        crate::nn::checkpoint::write_file_atomic(path, |mut w| {
            write_state_header(&mut w, kind)?;
            self.save_state(&mut w)
        })
    }

    /// Load state from a file. Detects gzip from `.gz` extension.
    ///
    /// Validates the `FDLO` header first: files from before the header
    /// existed are rejected with a pointer to
    /// [`migrate_optim_state_file`], and files written by a different
    /// optimizer are rejected by kind.
    fn load_state_file(&mut self, path: &str) -> Result<()> {
        let f = std::fs::File::open(path).map_err(|e| {
            crate::tensor::TensorError::new(&format!("io: {}", e))
        })?;
        let expected = self.state_kind();
        if path.ends_with(".gz") {
            let mut r = flate2::read::GzDecoder::new(f);
            read_state_header(&mut r, expected, path)?;
            self.load_state(&mut r)
        } else {
            let mut r = std::io::BufReader::new(f);
            read_state_header(&mut r, expected, path)?;
            self.load_state(&mut r)
        }
    }
}

/// Convert a pre-header optimizer state file (flodl ≤ 0.5.x) to the
/// current `FDLO`-headed format.
///
/// The old format carries no self-identification, so the `kind` of the
/// optimizer that wrote it must be supplied — the loader's error message
/// names the right one. In-place conversion (`src == dst`) is safe: the
/// destination is streamed into a temporary file and renamed over `dst`
/// only at the end (the same atomic recipe every state writer uses).
/// gzip is detected from each path's `.gz` extension independently.
///
/// For Adam/AdamW files the old single global step counter is expanded
/// into the current per-parameter step counts (every param gets the
/// global value — exact for params that trained from step 0, and the old
/// behavior's best available truth for the rest). Other kinds convert
/// verbatim under the new header. [`StateKind::Adagrad`], `RAdam` and
/// `NAdam` never had a pre-header format — passing them errors.
pub fn migrate_optim_state_file(src: &str, dst: &str, kind: StateKind) -> Result<()> {
    use crate::nn::checkpoint::{read_u32_le, write_f64_le};

    if matches!(kind, StateKind::Adagrad | StateKind::RAdam | StateKind::NAdam) {
        return Err(crate::tensor::TensorError::new(&format!(
            "migrate_optim_state_file: {} had no serialized state format \
             before the FDLO header — nothing to migrate",
            kind.name()
        )));
    }

    let f = std::fs::File::open(src).map_err(|e| {
        crate::tensor::TensorError::new(&format!("{src}: io: {}", e))
    })?;
    let mut r: Box<dyn Read> = if src.ends_with(".gz") {
        Box::new(flate2::read::GzDecoder::new(f))
    } else {
        Box::new(std::io::BufReader::new(f))
    };

    // Old-format check: a headed file starts with the magic; the old
    // payloads start with a param count (SGD/Adam/RMSprop), the low half
    // of a weight-decay f64 (AdamW) or of a power-of-two scale (GradScaler)
    // — none of which collide with `FDLO`.
    let mut first = [0u8; 4];
    r.read_exact(&mut first).map_err(|e| {
        crate::tensor::TensorError::new(&format!("{src}: io: {}", e))
    })?;
    if first == STATE_MAGIC {
        return Err(crate::tensor::TensorError::new(&format!(
            "migrate_optim_state_file: {src} already has the current FDLO \
             header — nothing to migrate"
        )));
    }

    let io_err = |e: std::io::Error| {
        crate::tensor::TensorError::new(&format!("io: {}", e))
    };

    // Transform the old Adam payload (count | lr | t | (m,v)* | groups)
    // into the current one (count | lr | (m,v,step)* | groups), expanding
    // the global t into per-param steps. `count` was already consumed by
    // the magic sniff.
    fn migrate_adam_payload<R: Read, W: Write>(
        r: &mut R, w: &mut W, count: u32,
    ) -> Result<()> {
        use crate::nn::checkpoint::{
            read_f64_le, read_i64_le, read_tensor_state,
            write_f64_le, write_i64_le, write_u32_le, write_tensor_state,
        };
        write_u32_le(w, count)?;
        let lr = read_f64_le(r)?;
        write_f64_le(w, lr)?;
        let t = read_i64_le(r)?;
        for _ in 0..count {
            let m = read_tensor_state(r, crate::tensor::Device::CPU)?;
            let v = read_tensor_state(r, crate::tensor::Device::CPU)?;
            write_tensor_state(w, m.as_ref())?;
            write_tensor_state(w, v.as_ref())?;
            write_i64_le(w, t)?;
        }
        std::io::copy(r, w).map_err(|e| {
            crate::tensor::TensorError::new(&format!("io: {}", e))
        })?;
        Ok(())
    }

    crate::nn::checkpoint::write_file_atomic(dst, |mut w| {
        write_state_header(&mut w, kind)?;
        match kind {
            StateKind::Adam => {
                migrate_adam_payload(&mut r, &mut w, u32::from_le_bytes(first))?;
            }
            StateKind::AdamW => {
                // Old AdamW payload = weight_decay(f64) | Adam payload;
                // `first` holds the f64's low half.
                let mut rest = [0u8; 4];
                r.read_exact(&mut rest).map_err(io_err)?;
                let mut wd = [0u8; 8];
                wd[..4].copy_from_slice(&first);
                wd[4..].copy_from_slice(&rest);
                write_f64_le(&mut w, f64::from_le_bytes(wd))?;
                let count = read_u32_le(&mut r)?;
                migrate_adam_payload(&mut r, &mut w, count)?;
            }
            _ => {
                // SGD / RMSprop / GradScaler payloads are unchanged —
                // re-emit verbatim under the new header.
                w.write_all(&first).map_err(io_err)?;
                std::io::copy(&mut r, &mut w).map_err(io_err)?;
            }
        }
        Ok(())
    })
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
