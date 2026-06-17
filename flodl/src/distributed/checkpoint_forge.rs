//! Controller-side consensus-checkpoint writer + the coordinator→reduce-thread
//! signal that arms it.
//!
//! In CPU averaging the consensus is forged in the controller's reduce thread
//! ([`crate::distributed::controller`]'s `run_reduce_thread` →
//! `average_and_scatter`) as a name-less, ordered
//! [`RoundFrame`](crate::distributed::controller::RoundFrame). [`CheckpointForge`]
//! lets the coordinator **arm** a model save (before the reduce it wants
//! captured); the reduce thread then hands the freshly-averaged frame to the
//! forge, which — holding the static [`ModelSchema`] captured at launch —
//! pairs names to the frame's tensors and writes a loadable `.fdl` on a
//! **detached** thread so the reduce loop never blocks.
//!
//! This keeps `controller.rs` a model-agnostic byte reducer: all nn/model
//! knowledge (names, `save_checkpoint_file`) lives here. The moving weights
//! already flow as averaging traffic; only the static schema is captured once,
//! so the forge writes a named checkpoint without routing the model through a
//! training rank.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::distributed::ModelSchema;
use crate::distributed::controller::RoundFrame;
use crate::distributed::cpu_reduce::round_frame_to_tensors;
use crate::nn::{Buffer, Parameter};
use crate::tensor::{Result, TensorError};

/// Coordinator→reduce-thread checkpoint signal, holding the static model
/// schema needed to name the averaged frame.
///
/// Shared as an `Arc` between the [`crate::distributed::cluster_coordinator::ClusterCoordinator`]
/// (which **arms** it before a checkpoint reduce) and the controller reduce
/// thread (which **consumes** the arm on the next forged frame) — the same
/// sharing pattern as the [`crate::distributed::controller::DeadRanks`] ledger.
pub struct CheckpointForge {
    /// Static param/buffer names captured at launch. `None` when no schema was
    /// captured (factory failure) → model writes are skipped (meta-only).
    schema: Option<ModelSchema>,
    /// Armed model path for the NEXT forged consensus frame. The coordinator
    /// sets it before the reduce it wants captured; the reduce thread takes it.
    pending: Mutex<Option<PathBuf>>,
}

impl CheckpointForge {
    /// Build a forge holding the launch-captured schema (or `None`).
    pub fn new(schema: Option<ModelSchema>) -> Arc<Self> {
        Arc::new(CheckpointForge {
            schema,
            pending: Mutex::new(None),
        })
    }

    /// Whether a model write is possible at all (a schema was captured). Lets
    /// the coordinator decide whether to arm the CPU forge or fall back to a
    /// meta-only checkpoint.
    pub fn can_write_model(&self) -> bool {
        self.schema.is_some()
    }

    /// Coordinator: arm a consensus model save to `model_path` (`<stem>.fdl`)
    /// for the NEXT forged frame. The latest arm wins (a prior un-consumed arm
    /// is overwritten — a new checkpoint supersedes a missed one).
    pub fn arm(&self, model_path: PathBuf) {
        *self.pending.lock().expect("checkpoint forge mutex poisoned") =
            Some(model_path);
    }

    /// Reduce thread: if armed, take the path and spawn a **detached** writer
    /// for `averaged` (one host-tensor-bytes clone), returning immediately so
    /// the reduce loop keeps scattering. No-op when not armed.
    pub fn maybe_write(&self, averaged: &RoundFrame) {
        let path = match self
            .pending
            .lock()
            .expect("checkpoint forge mutex poisoned")
            .take()
        {
            Some(p) => p,
            None => return,
        };
        let Some(schema) = self.schema.clone() else {
            eprintln!(
                "flodl ddp: consensus checkpoint armed but no model schema was \
                 captured at launch; .fdl skipped (meta-only) for {}",
                path.display(),
            );
            return;
        };
        // One clone of the host-resident averaged frame; the detached writer
        // owns it so the reduce loop is free to continue immediately.
        let frame = averaged.clone();
        let spawn = std::thread::Builder::new()
            .name("flodl-ckpt-writer".to_string())
            .spawn(move || {
                if let Err(e) = write_consensus_fdl(&schema, &frame, &path) {
                    eprintln!(
                        "flodl ddp: consensus checkpoint write to {} failed: {e}",
                        path.display(),
                    );
                }
            });
        if let Err(e) = spawn {
            eprintln!("flodl ddp: failed to spawn checkpoint writer thread: {e}");
        }
    }
}

/// Build a named, loadable `.fdl` from the ordered averaged frame + the static
/// schema. Positional pairing: the first `param_names.len()` tensors are
/// params, the rest buffers — the `GpuWorker::snapshot_params` layout. Loud on
/// a tensor-count mismatch (schema and frame disagree → a wiring bug; never
/// silently write a corrupt checkpoint).
fn write_consensus_fdl(
    schema: &ModelSchema,
    frame: &RoundFrame,
    path: &Path,
) -> Result<()> {
    let tensors = round_frame_to_tensors(frame)?;
    if tensors.len() != schema.tensor_count() {
        return Err(TensorError::new(&format!(
            "checkpoint_forge: averaged frame has {} tensors but schema expects \
             {} ({} params + {} buffers) — schema/frame mismatch",
            tensors.len(),
            schema.tensor_count(),
            schema.param_names.len(),
            schema.buffer_names.len(),
        )));
    }
    let mut iter = tensors.into_iter();
    let params: Vec<(String, Parameter)> = schema
        .param_names
        .iter()
        .map(|name| {
            let t = iter.next().expect("tensor count checked above");
            (name.clone(), Parameter::new(t, name))
        })
        .collect();
    let buffers: Vec<(String, Buffer)> = schema
        .buffer_names
        .iter()
        .map(|name| {
            let t = iter.next().expect("tensor count checked above");
            (name.clone(), Buffer::new(t, name))
        })
        .collect();
    let path_str = path.to_str().ok_or_else(|| {
        TensorError::new(&format!(
            "checkpoint_forge: non-utf8 checkpoint path {}",
            path.display(),
        ))
    })?;
    crate::nn::save_checkpoint_file(path_str, &params, &buffers, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::cpu_reduce::tensors_to_round_frame;
    use crate::tensor::{Device, Tensor};

    fn cpu_tensor(vals: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(vals, shape, Device::CPU).unwrap()
    }

    #[test]
    fn write_consensus_fdl_round_trips_through_load() {
        // Schema: 2 params (w, b) + 1 buffer (running_mean), in snapshot order.
        let schema = ModelSchema {
            param_names: vec!["w".to_string(), "b".to_string()],
            buffer_names: vec!["running_mean".to_string()],
        };
        let w = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = cpu_tensor(&[5.0, 6.0], &[2]);
        let rm = cpu_tensor(&[7.0, 8.0], &[2]);
        let frame = tensors_to_round_frame(&[&w, &b, &rm]).unwrap();

        let dir = std::env::temp_dir()
            .join(format!("flodl_forge_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");
        write_consensus_fdl(&schema, &frame, &path).unwrap();

        // Load back into zero-initialised targets keyed by the same names.
        let tw = Parameter::new(cpu_tensor(&[0.0; 4], &[2, 2]), "w");
        let tb = Parameter::new(cpu_tensor(&[0.0; 2], &[2]), "b");
        let trm = Buffer::new(cpu_tensor(&[0.0; 2], &[2]), "running_mean");
        crate::nn::load_checkpoint_file(
            path.to_str().unwrap(),
            &[("w".to_string(), tw.clone()), ("b".to_string(), tb.clone())],
            &[("running_mean".to_string(), trm.clone())],
            None,
        )
        .unwrap();

        assert_eq!(tw.variable.data().to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tb.variable.data().to_f32_vec().unwrap(), vec![5.0, 6.0]);
        assert_eq!(trm.get().to_f32_vec().unwrap(), vec![7.0, 8.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_consensus_fdl_rejects_tensor_count_mismatch() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
        };
        // Frame carries 2 tensors but schema expects 1.
        let a = cpu_tensor(&[1.0], &[1]);
        let b = cpu_tensor(&[2.0], &[1]);
        let frame = tensors_to_round_frame(&[&a, &b]).unwrap();
        let path = std::env::temp_dir().join("flodl_forge_mismatch.fdl");
        let err = write_consensus_fdl(&schema, &frame, &path).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "got: {err}");
    }

    #[test]
    fn arm_then_maybe_write_consumes_once() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
        };
        let forge = CheckpointForge::new(Some(schema));
        assert!(forge.can_write_model());
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        let frame = tensors_to_round_frame(&[&w]).unwrap();

        let dir = std::env::temp_dir()
            .join(format!("flodl_forge_arm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.fdl");

        forge.arm(path.clone());
        forge.maybe_write(&frame); // spawns detached writer
        // A second call with no re-arm is a no-op (arm consumed).
        forge.maybe_write(&frame);

        // Join is implicit (detached); poll briefly for the file.
        let mut found = false;
        for _ in 0..200 {
            if path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "armed write produced the .fdl");
        std::fs::remove_dir_all(&dir).ok();
    }
}
