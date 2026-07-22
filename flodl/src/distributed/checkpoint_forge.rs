//! Controller-side consensus-checkpoint writer + the coordinator→reduce-thread
//! signal that arms it.
//!
//! In CPU averaging the consensus is forged in the controller's reduce thread
//! ([`crate::distributed::controller`]'s `run_reduce_thread` →
//! `average_and_scatter`) as name-less, ordered
//! `RoundFrame`s. One sync cycle
//! issues several reduces over the same channel — a `Control` per-rank
//! count-gather, then one or two `Model` reduces (params, then buffers) — so no
//! single frame ever carries the whole model. [`CheckpointForge`] lets the
//! coordinator **arm** a save (before the cycle it wants captured); the reduce
//! thread then hands each `Model` frame to the forge, which **accumulates**
//! them (params' tensors then buffers') until it holds the full model, pairs
//! the held static [`ModelSchema`] names to the accumulated tensors, and writes
//! a loadable `.fdl` on a **detached** thread so the reduce loop never blocks.
//!
//! Two properties matter:
//! - **Async**: frames are scattered to ranks *before* the forge tap, and the
//!   `.fdl` is serialized off the reduce thread — training never waits on the
//!   checkpoint.
//! - **Zero extra copy**: the averaged frame is *moved* into the accumulator
//!   (not cloned), and the writer emits the frame's raw native bytes straight
//!   to disk via [`save_checkpoint_from_raw_file`] — no bytes→`Tensor`→bytes
//!   round-trip, no duplicate model in RAM.
//!
//! This keeps `controller.rs` a model-agnostic byte reducer: all nn/model
//! knowledge (names, the `.fdl` format) lives here.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::distributed::ModelSchema;
use crate::distributed::controller::{DTYPE_F32, RoundFrame, TensorPayload};
use crate::nn::checkpoint::{LoadReport, RawCheckpointEntry, dtype_tag, save_checkpoint_from_raw_file};
use crate::nn::{Buffer, Module, Parameter};
use crate::tensor::{DType, Result, TensorError};

/// Positional `.fdl` key for the `i`-th model **parameter** in a cluster
/// consensus checkpoint (`Module::parameters()` order).
///
/// Consensus bundles are positional by nature — the forge pairs the reduce's
/// averaged tensors to the model by index, never by the model's own parameter
/// names, which routinely repeat across a layer stack (bare `"weight"` /
/// `"bias"` per `Linear`/`Conv`). Keying by those names would collide in the
/// on-disk map and load the wrong tensor. Synthetic `p{i}` / `b{j}` keys are
/// unique by construction. Writers and [`load_consensus_checkpoint`] share
/// these helpers so the convention has a single definition.
pub(crate) fn consensus_param_key(i: usize) -> String {
    format!("p{i}")
}

/// Positional `.fdl` key for the `j`-th model **buffer** (`Module::buffers()`
/// order). See [`consensus_param_key`].
pub(crate) fn consensus_buffer_key(j: usize) -> String {
    format!("b{j}")
}

/// Load a cluster consensus checkpoint (written by `CheckpointForge` or the
/// elected-rank / failure-save model writer) into `model`, matching tensors
/// **positionally** via `consensus_param_key` / `consensus_buffer_key`.
///
/// Use this on resume instead of keying by the model's own parameter names:
/// stacked layers reuse bare names (`"weight"`, `"bias"`), which a name-keyed
/// load would collapse and mismatch. Positional keys load each tensor into the
/// same slot it was saved from, given the same `model_factory` (deterministic
/// `parameters()` / `buffers()` order).
pub fn load_consensus_checkpoint<M: Module + ?Sized>(model: &M, path: &str) -> Result<LoadReport> {
    let params: Vec<(String, Parameter)> = model
        .parameters()
        .into_iter()
        .enumerate()
        .map(|(i, p)| (consensus_param_key(i), p))
        .collect();
    let buffers: Vec<(String, Buffer)> = model
        .buffers()
        .into_iter()
        .enumerate()
        .map(|(j, b)| (consensus_buffer_key(j), b))
        .collect();
    crate::nn::load_checkpoint_file(path, &params, &buffers, None)
}

/// Coordinator→reduce-thread checkpoint signal, holding the static model
/// schema needed to name the accumulated frames.
///
/// Shared as an `Arc` between the [`crate::distributed::cluster_coordinator::ClusterCoordinator`]
/// (which **arms** it before a checkpoint cycle) and the controller reduce
/// thread (which **accumulates** the cycle's `Model` frames) — the same sharing
/// pattern as the [`crate::distributed::controller::DeadRanks`] ledger.
pub struct CheckpointForge {
    /// Static param/buffer names captured at launch. `None` when no schema was
    /// captured (factory failure) → model writes are skipped (meta-only).
    schema: Option<ModelSchema>,
    /// Mutable accumulation state (armed path + tensors gathered so far).
    inner: Mutex<ForgeState>,
}

/// What the forge has gathered toward the next consensus checkpoint.
#[derive(Default)]
struct ForgeState {
    /// Armed model path for the current checkpoint cycle (`<stem>.fdl`). Set by
    /// the coordinator before the cycle it wants captured; taken when the write
    /// is spawned. `None` = not armed (frames are ignored).
    pending_path: Option<PathBuf>,
    /// `Model`-reduce tensors gathered this cycle, in arrival order (params'
    /// frame, then buffers' frame). Moved in, never copied. Drained on write.
    accumulated: Vec<TensorPayload>,
    /// Outer-optimizer momentum payloads for this cycle (one tensor per model
    /// parameter), stashed by the controller when an outer optimizer carries
    /// state. `None` for stateless outer optimizers (OuterAvg) — no
    /// `<stem>.outer.fdl` is written. Taken alongside the model on write.
    pending_outer: Option<Vec<TensorPayload>>,
}

impl CheckpointForge {
    /// Build a forge holding the launch-captured schema (or `None`).
    pub fn new(schema: Option<ModelSchema>) -> Arc<Self> {
        Arc::new(CheckpointForge {
            schema,
            inner: Mutex::new(ForgeState::default()),
        })
    }

    /// Whether a model write is possible at all (a schema was captured). Lets
    /// the coordinator decide whether to arm the CPU forge or fall back to a
    /// meta-only checkpoint.
    pub fn can_write_model(&self) -> bool {
        self.schema.is_some()
    }

    /// Coordinator: arm a consensus model save to `model_path` (`<stem>.fdl`)
    /// for the NEXT checkpoint cycle. Clears any partial accumulation (a new
    /// checkpoint supersedes a missed one), so the forge starts the cycle
    /// fresh.
    pub fn arm(&self, model_path: PathBuf) {
        let mut st = self.inner.lock().expect("checkpoint forge mutex poisoned");
        st.pending_path = Some(model_path);
        st.accumulated.clear();
        st.pending_outer = None;
    }

    /// Whether a checkpoint is currently armed (a `<stem>.fdl` path is
    /// pending). The controller checks this to decide whether to snapshot the
    /// outer-optimizer momentum this window — a rare event, so the (cheap)
    /// momentum serialize only happens when a checkpoint is actually pending.
    pub fn is_armed(&self) -> bool {
        self.inner
            .lock()
            .expect("checkpoint forge mutex poisoned")
            .pending_path
            .is_some()
    }

    /// Stash this cycle's outer-optimizer momentum payloads (one tensor per
    /// model parameter). Written to `<stem>.outer.fdl` alongside the consensus
    /// model when this cycle's accumulation completes. No-op contract: pass
    /// only when the outer optimizer carries state (stateless OuterAvg never
    /// stashes, so no artifact is written).
    pub fn stash_outer_momentum(&self, payloads: Vec<TensorPayload>) {
        let mut st = self.inner.lock().expect("checkpoint forge mutex poisoned");
        st.pending_outer = Some(payloads);
    }

    /// Reduce thread: fold one `Model` reduce's averaged frame into the
    /// accumulation. The frame is **moved** in (no clone). When the gathered
    /// tensor count reaches the schema's total (params + buffers), take the
    /// armed path + accumulated tensors and spawn a **detached** writer,
    /// returning immediately so the reduce loop keeps scattering. No-op when
    /// not armed or no schema was captured.
    ///
    /// Caller must only pass [`crate::distributed::controller::RoundKind::Model`]
    /// frames; the per-rank count-gather (`Control`) is never the model.
    pub fn accumulate(&self, frame: RoundFrame) {
        let Some(schema) = self.schema.as_ref() else {
            return; // no schema captured; .fdl skipped (meta-only)
        };
        let want = schema.tensor_count();
        let (path, payloads, outer) = {
            let mut st = self.inner.lock().expect("checkpoint forge mutex poisoned");
            if st.pending_path.is_none() {
                return; // not armed — drop this cycle's frame
            }
            // MOVE the frame's tensors into the accumulator (no copy).
            st.accumulated.extend(frame.tensors);
            if st.accumulated.len() < want {
                return; // more model reduces to come this cycle (buffers)
            }
            let path = st.pending_path.take().expect("armed checked above");
            let payloads = std::mem::take(&mut st.accumulated);
            // Outer-optimizer momentum for this cycle, if any (None for
            // stateless OuterAvg => no `.outer.fdl`).
            let outer = st.pending_outer.take();
            (path, payloads, outer)
        };
        if payloads.len() != want {
            // Overshoot: more tensors arrived than the schema expects — a wiring
            // bug. Skip rather than write a misaligned checkpoint.
            eprintln!(
                "flodl ddp: consensus checkpoint accumulated {} tensors but schema \
                 expects {} ({} params + {} buffers); .fdl skipped for {}",
                payloads.len(),
                want,
                schema.param_names.len(),
                schema.buffer_names.len(),
                path.display(),
            );
            return;
        }
        // Validate the outer momentum count here (on the reduce thread, where
        // the error is actionable) before handing it to the detached writer:
        // it is model-sized (one tensor per parameter), so it must match the
        // schema's param count. A mismatch drops just the `.outer.fdl` (the
        // consensus model still writes); a stateless outer optimizer passes
        // `None` and writes no artifact.
        let outer = outer.filter(|o| {
            let ok = o.len() == schema.param_names.len();
            if !ok {
                eprintln!(
                    "flodl ddp: outer-momentum has {} tensors but schema expects \
                     {} params; .outer.fdl skipped for {}",
                    o.len(),
                    schema.param_names.len(),
                    path.display(),
                );
            }
            ok
        });
        let schema = schema.clone();
        let spawn = std::thread::Builder::new()
            .name("flodl-ckpt-writer".to_string())
            .spawn(move || {
                if let Err(e) = write_consensus_fdl(&schema, &payloads, &path) {
                    eprintln!(
                        "flodl ddp: consensus checkpoint write to {} failed: {e}",
                        path.display(),
                    );
                }
                // Outer-optimizer momentum rides the same writer: `<stem>.fdl`
                // -> `<stem>.outer.fdl`, same atomic-rename commit. Written
                // after the model so a present `.outer.fdl` implies a present
                // `.fdl`.
                if let Some(outer_payloads) = outer {
                    let outer_path = path.with_extension("outer.fdl");
                    if let Err(e) = write_outer_momentum_fdl(&outer_payloads, &outer_path) {
                        eprintln!(
                            "flodl ddp: outer-momentum checkpoint write to {} failed: {e}",
                            outer_path.display(),
                        );
                    }
                }
            });
        if let Err(e) = spawn {
            eprintln!("flodl ddp: failed to spawn checkpoint writer thread: {e}");
        }
    }
}

/// Write a named, loadable `.fdl` straight from the accumulated averaged
/// payloads + the static schema. Positional pairing: the first
/// `param_names.len()` tensors are params, the rest buffers — the
/// `GpuWorker::snapshot_params` / reduce order. Emits the payloads' raw native
/// bytes directly (no `Tensor` reconstruction) and commits with a temp-file +
/// atomic rename, so a crash mid-write never leaves a torn `.fdl` that resume
/// could mistake for valid.
fn write_consensus_fdl(schema: &ModelSchema, payloads: &[TensorPayload], path: &Path) -> Result<()> {
    if payloads.len() != schema.tensor_count() {
        return Err(TensorError::new(&format!(
            "checkpoint_forge: accumulated {} tensors but schema expects \
             {} ({} params + {} buffers) — schema/accumulation mismatch",
            payloads.len(),
            schema.tensor_count(),
            schema.param_names.len(),
            schema.buffer_names.len(),
        )));
    }
    // Pre-convert u32 wire shapes to the i64 dims the .fdl format stores;
    // entries borrow these.
    let shapes: Vec<Vec<i64>> = payloads
        .iter()
        .map(|p| p.shape.iter().map(|&d| d as i64).collect())
        .collect();
    // Positional keys: the first `param_names.len()` payloads are params
    // (`p{i}`), the rest buffers (`b{j}`). Synthetic + unique — never the
    // model's own (possibly repeated) names. Matches `load_consensus_checkpoint`.
    let param_count = schema.param_names.len();
    let keys: Vec<String> = (0..payloads.len())
        .map(|i| {
            if i < param_count {
                consensus_param_key(i)
            } else {
                consensus_buffer_key(i - param_count)
            }
        })
        .collect();
    let mut entries = Vec::with_capacity(payloads.len());
    for (i, p) in payloads.iter().enumerate() {
        if p.dtype != DTYPE_F32 {
            return Err(TensorError::new(&format!(
                "checkpoint_forge: payload[{i}] dtype {} not supported (v1 f32 only)",
                p.dtype,
            )));
        }
        entries.push(RawCheckpointEntry {
            name: keys[i].as_str(),
            shape: &shapes[i],
            dtype_tag: dtype_tag(DType::Float32),
            raw: &p.bytes,
        });
    }
    let path_str = path.to_str().ok_or_else(|| {
        TensorError::new(&format!(
            "checkpoint_forge: non-utf8 checkpoint path {}",
            path.display(),
        ))
    })?;
    let tmp = format!("{path_str}.tmp");
    save_checkpoint_from_raw_file(&tmp, &entries, None)?;
    std::fs::rename(&tmp, path_str).map_err(|e| {
        TensorError::new(&format!(
            "checkpoint_forge: atomic rename {tmp} -> {path_str} failed: {e}"
        ))
    })?;
    Ok(())
}

/// Write the outer-optimizer momentum to `<stem>.outer.fdl`: one tensor per
/// model **parameter**, keyed positionally (`p{i}`) exactly like the
/// consensus params (no buffers), committed with a temp-file + atomic rename.
/// Loaded back on resume by [`load_outer_momentum`].
fn write_outer_momentum_fdl(payloads: &[TensorPayload], path: &Path) -> Result<()> {
    let shapes: Vec<Vec<i64>> = payloads
        .iter()
        .map(|p| p.shape.iter().map(|&d| d as i64).collect())
        .collect();
    let keys: Vec<String> = (0..payloads.len()).map(consensus_param_key).collect();
    let mut entries = Vec::with_capacity(payloads.len());
    for (i, p) in payloads.iter().enumerate() {
        if p.dtype != DTYPE_F32 {
            return Err(TensorError::new(&format!(
                "checkpoint_forge: outer payload[{i}] dtype {} not supported (v1 f32 only)",
                p.dtype,
            )));
        }
        entries.push(RawCheckpointEntry {
            name: keys[i].as_str(),
            shape: &shapes[i],
            dtype_tag: dtype_tag(DType::Float32),
            raw: &p.bytes,
        });
    }
    let path_str = path.to_str().ok_or_else(|| {
        TensorError::new(&format!(
            "checkpoint_forge: non-utf8 outer-momentum path {}",
            path.display(),
        ))
    })?;
    let tmp = format!("{path_str}.tmp");
    save_checkpoint_from_raw_file(&tmp, &entries, None)?;
    std::fs::rename(&tmp, path_str).map_err(|e| {
        TensorError::new(&format!(
            "checkpoint_forge: atomic rename {tmp} -> {path_str} failed: {e}"
        ))
    })?;
    Ok(())
}

/// Load `<stem>.outer.fdl` (outer-optimizer momentum) into a fresh tensor per
/// model parameter, matched positionally (`p{i}`) like
/// [`load_consensus_checkpoint`]. Returns the momentum tensors in
/// `Module::parameters()` order, ready for
/// [`crate::distributed::OuterOptimizer::load_checkpoint_state`].
///
/// Used by the launcher on resume: the outer momentum lives controller-side
/// (CPU backend), so the launcher reconstructs it here from a throwaway probe
/// model's parameter shapes before handing the seeded optimizer to the
/// controller.
pub fn load_outer_momentum<M: Module + ?Sized>(
    model: &M,
    path: &str,
) -> Result<Vec<crate::tensor::Tensor>> {
    let params = model.parameters();
    // Fresh zero targets shaped like each parameter; the load overwrites them.
    let mut targets: Vec<(String, Parameter)> = Vec::with_capacity(params.len());
    for (i, p) in params.iter().enumerate() {
        let zeros = crate::tensor::Tensor::zeros_like(&p.variable.data())?;
        targets.push((consensus_param_key(i), Parameter::new(zeros, "outer_momentum")));
    }
    crate::nn::load_checkpoint_file(path, &targets, &[], None)?;
    Ok(targets.iter().map(|(_, p)| p.variable.data()).collect())
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
    fn accumulate_params_then_buffers_round_trips_through_load() {
        // Schema: 2 params (w, b) + 1 buffer (running_mean), in snapshot order.
        let schema = ModelSchema {
            param_names: vec!["w".to_string(), "b".to_string()],
            buffer_names: vec!["running_mean".to_string()],
        };
        let forge = CheckpointForge::new(Some(schema));
        assert!(forge.can_write_model());

        let w = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = cpu_tensor(&[5.0, 6.0], &[2]);
        let rm = cpu_tensor(&[7.0, 8.0], &[2]);
        // Two model reduces: params frame (w, b) then buffers frame (rm).
        let params_frame = tensors_to_round_frame(&[&w, &b]).unwrap();
        let buffers_frame = tensors_to_round_frame(&[&rm]).unwrap();

        let dir = std::env::temp_dir().join(format!("flodl_forge_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");

        forge.arm(path.clone());
        forge.accumulate(params_frame); // partial: 2 of 3, no write yet
        assert!(!path.exists(), "no write before all model frames arrive");
        forge.accumulate(buffers_frame); // completes 3/3 → detached write

        // Join is implicit (detached); poll briefly for the file.
        let mut found = false;
        for _ in 0..200 {
            if path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "completed accumulation produced the .fdl");

        // Load back into zero-initialised targets keyed by the same names.
        use crate::nn::{Buffer, Parameter};
        let tw = Parameter::new(cpu_tensor(&[0.0; 4], &[2, 2]), "w");
        let tb = Parameter::new(cpu_tensor(&[0.0; 2], &[2]), "b");
        let trm = Buffer::new(cpu_tensor(&[0.0; 2], &[2]), "running_mean");
        // Load by the positional keys the writer emits (p0, p1, b0) — not the
        // model's own names (which can repeat across layers).
        crate::nn::load_checkpoint_file(
            path.to_str().unwrap(),
            &[("p0".to_string(), tw.clone()), ("p1".to_string(), tb.clone())],
            &[("b0".to_string(), trm.clone())],
            None,
        )
        .unwrap();

        assert_eq!(tw.variable.data().to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tb.variable.data().to_f32_vec().unwrap(), vec![5.0, 6.0]);
        assert_eq!(trm.get().to_f32_vec().unwrap(), vec![7.0, 8.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn outer_momentum_writes_sidecar_and_round_trips() {
        // When outer momentum is stashed, the completing cycle writes both
        // <stem>.fdl and <stem>.outer.fdl, and the sidecar loads back.
        let schema = ModelSchema {
            param_names: vec!["w".to_string(), "b".to_string()],
            buffer_names: vec!["running_mean".to_string()],
        };
        let forge = CheckpointForge::new(Some(schema));

        let w = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = cpu_tensor(&[5.0, 6.0], &[2]);
        let rm = cpu_tensor(&[7.0, 8.0], &[2]);
        // Momentum is one tensor per PARAM (not buffers): 2 here.
        let mw = cpu_tensor(&[0.1, 0.2, 0.3, 0.4], &[2, 2]);
        let mb = cpu_tensor(&[0.5, 0.6], &[2]);

        let dir = std::env::temp_dir().join(format!("flodl_forge_outer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");

        forge.arm(path.clone());
        assert!(forge.is_armed());
        // Stash momentum, then complete the model accumulation (params, buffers).
        forge.stash_outer_momentum(tensors_to_round_frame(&[&mw, &mb]).unwrap().tensors);
        forge.accumulate(tensors_to_round_frame(&[&w, &b]).unwrap());
        forge.accumulate(tensors_to_round_frame(&[&rm]).unwrap()); // completes

        let outer_path = path.with_extension("outer.fdl");
        let mut found = false;
        for _ in 0..200 {
            if path.exists() && outer_path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "both .fdl and .outer.fdl written");

        // Load the momentum sidecar by positional keys (p0, p1).
        use crate::nn::Parameter;
        let t0 = Parameter::new(cpu_tensor(&[0.0; 4], &[2, 2]), "m0");
        let t1 = Parameter::new(cpu_tensor(&[0.0; 2], &[2]), "m1");
        crate::nn::load_checkpoint_file(
            outer_path.to_str().unwrap(),
            &[("p0".to_string(), t0.clone()), ("p1".to_string(), t1.clone())],
            &[],
            None,
        )
        .unwrap();
        assert_eq!(t0.variable.data().to_f32_vec().unwrap(), vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(t1.variable.data().to_f32_vec().unwrap(), vec![0.5, 0.6]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_outer_momentum_writes_no_sidecar() {
        // Stateless outer optimizer stashes nothing => no .outer.fdl.
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
        };
        let forge = CheckpointForge::new(Some(schema));
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        let dir = std::env::temp_dir().join(format!("flodl_forge_noouter_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.fdl");

        forge.arm(path.clone());
        forge.accumulate(tensors_to_round_frame(&[&w]).unwrap()); // completes, no stash

        let outer_path = path.with_extension("outer.fdl");
        let mut model_found = false;
        for _ in 0..200 {
            if path.exists() {
                model_found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(model_found, ".fdl written");
        // Give any (erroneous) sidecar writer a chance, then assert absence.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!outer_path.exists(), "no .outer.fdl when no momentum stashed");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unarmed_accumulate_is_noop() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
        };
        let forge = CheckpointForge::new(Some(schema));
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        // No arm() — accumulate must not crash or write.
        forge.accumulate(tensors_to_round_frame(&[&w]).unwrap());
    }

    #[test]
    fn arm_clears_partial_accumulation() {
        // A new arm supersedes a missed one: a stale partial frame must not
        // bleed into the next cycle's model.
        let schema = ModelSchema {
            param_names: vec!["w".to_string(), "b".to_string()],
            buffer_names: vec![],
        };
        let forge = CheckpointForge::new(Some(schema));
        let stale = cpu_tensor(&[9.0], &[1]);
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        let b = cpu_tensor(&[3.0], &[1]);

        let dir = std::env::temp_dir().join(format!("flodl_forge_arm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.fdl");

        forge.arm(path.clone());
        forge.accumulate(tensors_to_round_frame(&[&stale]).unwrap()); // partial 1/2
        // Re-arm: the stale partial is dropped, cycle restarts.
        forge.arm(path.clone());
        forge.accumulate(tensors_to_round_frame(&[&w]).unwrap()); // 1/2
        forge.accumulate(tensors_to_round_frame(&[&b]).unwrap()); // 2/2 → write

        let mut found = false;
        for _ in 0..200 {
            if path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "re-armed accumulation produced the .fdl");

        use crate::nn::Parameter;
        let tw = Parameter::new(cpu_tensor(&[0.0; 2], &[2]), "w");
        let tb = Parameter::new(cpu_tensor(&[0.0; 1], &[1]), "b");
        crate::nn::load_checkpoint_file(
            path.to_str().unwrap(),
            &[("p0".to_string(), tw.clone()), ("p1".to_string(), tb.clone())],
            &[],
            None,
        )
        .unwrap();
        assert_eq!(tw.variable.data().to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(tb.variable.data().to_f32_vec().unwrap(), vec![3.0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn write_consensus_fdl_rejects_tensor_count_mismatch() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
        };
        // Two payloads but schema expects 1.
        let a = cpu_tensor(&[1.0], &[1]);
        let b = cpu_tensor(&[2.0], &[1]);
        let frame = tensors_to_round_frame(&[&a, &b]).unwrap();
        let path = std::env::temp_dir().join("flodl_forge_mismatch.fdl");
        let err = write_consensus_fdl(&schema, &frame.tensors, &path).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "got: {err}");
    }
}
