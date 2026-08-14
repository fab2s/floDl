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
use crate::nn::checkpoint::{
    LoadReport, RawCheckpointEntry, dtype_tag, save_checkpoint_from_raw_file,
};
use crate::nn::{Buffer, Module, Parameter};
use crate::tensor::{DType, Result, TensorError};

/// Type-erased consensus-model callback: `(version, schema, payloads)`.
///
/// Built at launch by [`consensus_checkpoint_fn`] around the user's
/// `checkpoint_fn` (CPU backend only — NCCL fires the user callback on the
/// elected rank, whose post-collective model IS the consensus) and installed
/// on the forge, which fires it on its **detached** writer thread when an
/// armed cycle's accumulation completes — the reduce loop never blocks on it
/// (the coordinator must never go heartbeat-silent). Type-erased so the
/// controller plumbing stays model-agnostic: the closure owns the only
/// model-typed state.
pub type ConsensusModelFn =
    Arc<dyn Fn(u64, &ModelSchema, &[TensorPayload]) -> Result<()> + Send + Sync>;

/// Wrap a user `checkpoint_fn` into a [`ConsensusModelFn`]: each fire builds
/// a CPU probe model from `model_factory`, loads the consensus payloads into
/// it positionally, calls `f(version, &model)`, and drops the probe. The
/// residency is transient per fire, never steady — models hold `Rc` internals
/// (not `Send`), so a cached probe could not live inside this `Send + Sync`
/// closure anyway, and a fresh build per (rare) checkpoint fire is cheaper
/// than a model-sized standing allocation. Panics in the user closure are
/// caught and surfaced as errors so a misbehaving callback cannot kill the
/// forge's writer thread.
///
/// Mental-model outcome, both backends: **`checkpoint_fn` always receives the
/// consensus model** — here by construction (the frame is the consensus), on
/// NCCL by post-collective timing. The model it receives lives on the CPU.
pub(crate) fn consensus_checkpoint_fn<M, F>(
    model_factory: F,
    f: crate::distributed::ddp_run::CheckpointFn<M>,
) -> ConsensusModelFn
where
    M: Module + 'static,
    F: Fn(crate::tensor::Device) -> Result<M> + Send + Sync + 'static,
{
    Arc::new(move |version, schema, payloads| {
        let model = model_factory(crate::tensor::Device::CPU)?;
        load_payloads_into_model(&model, schema, payloads)?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(version, &model))).map_err(
            |p| {
                let what = p
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                TensorError::new(&format!("checkpoint_fn panicked: {what}"))
            },
        )?
    })
}

/// Load a consensus cycle's accumulated payloads into `model` positionally:
/// the first `param_names.len()` payloads are params, the rest the f32
/// buffer subset, each mapped to its full-list buffer index via
/// [`ModelSchema::f32_buffer_idx`]. Non-f32 buffers keep the model's own
/// values (the reduce's passthrough semantics). Counts are validated so a
/// factory/schema drift errors loudly instead of loading tensors into the
/// wrong slots.
fn load_payloads_into_model<M: Module + ?Sized>(
    model: &M,
    schema: &ModelSchema,
    payloads: &[TensorPayload],
) -> Result<()> {
    let params = model.parameters();
    let buffers = model.buffers();
    if params.len() != schema.param_names.len()
        || buffers.len() != schema.buffer_names.len()
        || payloads.len() != schema.tensor_count()
    {
        return Err(TensorError::new(&format!(
            "consensus checkpoint_fn: model has {} params + {} buffers, schema \
             expects {} + {} ({} f32), frame carries {} tensors — factory/schema drift",
            params.len(),
            buffers.len(),
            schema.param_names.len(),
            schema.buffer_names.len(),
            schema.f32_buffer_idx.len(),
            payloads.len(),
        )));
    }
    let _no_grad = crate::autograd::NoGradGuard::new();
    let load_one = |dst: &crate::tensor::Tensor, p: &TensorPayload| -> Result<()> {
        let vals = crate::distributed::controller::payload_to_f32(p)?;
        let shape: Vec<i64> = p.shape.iter().map(|&d| d as i64).collect();
        let src = crate::tensor::Tensor::from_f32(&vals, &shape, crate::tensor::Device::CPU)?;
        dst.copy_(&src, false)
    };
    for (i, param) in params.iter().enumerate() {
        load_one(&param.variable.data(), &payloads[i])?;
    }
    for (k, &bi) in schema.f32_buffer_idx.iter().enumerate() {
        load_one(&buffers[bi].get(), &payloads[schema.param_names.len() + k])?;
    }
    Ok(())
}

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
    /// Consensus-model callback (the launch-wrapped user `checkpoint_fn`),
    /// fired on the detached writer thread when an armed cycle with a
    /// `user_version` completes. `None` when no callback is configured or on
    /// the NCCL path (elected-rank fire).
    consensus_fn: Option<ConsensusModelFn>,
    /// Mutable accumulation state (armed checkpoint + tensors gathered so far).
    inner: Mutex<ForgeState>,
    /// Lifetime count of arms taken (relaxed; forensics only). With
    /// `writers_spawned` this splits a missing artifact three ways: never
    /// armed, armed but the cycle's frame never completed the accumulation,
    /// or handed to a writer that has not finished yet.
    arms_taken: std::sync::atomic::AtomicUsize,
    /// Lifetime count of detached writer threads successfully spawned.
    writers_spawned: std::sync::atomic::AtomicUsize,
}

/// One armed consensus capture: what to do with the cycle's materialized
/// frame. Bundle write and user-callback fire are independent — a cadence
/// with `save_path` does both, a cadence with only a `checkpoint_fn` fires
/// without writing, the final-consensus arm writes without firing.
struct ArmedCheckpoint {
    /// `<stem>.fdl` destination; `None` = no bundle this cycle.
    path: Option<PathBuf>,
    /// Fire the consensus callback with this version after materialization;
    /// `None` = no user fire this cycle.
    user_version: Option<u64>,
}

/// What the forge has gathered toward the next consensus checkpoint.
#[derive(Default)]
struct ForgeState {
    /// Armed capture for the current checkpoint cycle. Set by the coordinator
    /// before the cycle it wants captured; taken when the writer is spawned.
    /// `None` = not armed (frames are ignored).
    pending: Option<ArmedCheckpoint>,
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
    /// Build a forge holding the launch-captured schema (or `None`) and the
    /// launch-wrapped consensus callback (or `None`).
    pub fn new(schema: Option<ModelSchema>, consensus_fn: Option<ConsensusModelFn>) -> Arc<Self> {
        Arc::new(CheckpointForge {
            schema,
            consensus_fn,
            inner: Mutex::new(ForgeState::default()),
            arms_taken: std::sync::atomic::AtomicUsize::new(0),
            writers_spawned: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Forensic counters: `(arms taken, writers spawned)`. Meant for test
    /// failure messages and post-mortems, not control flow.
    pub fn forensics(&self) -> (usize, usize) {
        (
            self.arms_taken.load(std::sync::atomic::Ordering::Relaxed),
            self.writers_spawned
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Whether a consensus callback is installed (the coordinator's gate for
    /// user-fire arms on the CPU path).
    pub fn has_consensus_fn(&self) -> bool {
        self.consensus_fn.is_some()
    }

    /// Whether a model write is possible at all (a schema was captured). Lets
    /// the coordinator decide whether to arm the CPU forge or fall back to a
    /// meta-only checkpoint.
    pub fn can_write_model(&self) -> bool {
        self.schema.is_some()
    }

    /// Coordinator: arm a consensus capture for the NEXT checkpoint cycle —
    /// a model save to `model_path` (`<stem>.fdl`), a consensus-callback fire
    /// at `user_version`, or both. A both-`None` arm is a no-op. Clears any
    /// partial accumulation (a new checkpoint supersedes a missed one), so
    /// the forge starts the cycle fresh.
    pub fn arm(&self, model_path: Option<PathBuf>, user_version: Option<u64>) {
        if model_path.is_none() && user_version.is_none() {
            return;
        }
        let mut st = self.inner.lock().expect("checkpoint forge mutex poisoned");
        st.pending = Some(ArmedCheckpoint {
            path: model_path,
            user_version,
        });
        st.accumulated.clear();
        st.pending_outer = None;
        self.arms_taken
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a bundle write is currently armed (a `<stem>.fdl` path is
    /// pending). The controller checks this to decide whether to snapshot the
    /// outer-optimizer momentum this window — a rare event, so the (cheap)
    /// momentum serialize only happens when a checkpoint is actually pending.
    /// A user-fire-only arm does not count: the `.outer.fdl` sidecar rides
    /// the bundle, and there is none.
    pub fn is_armed(&self) -> bool {
        self.inner
            .lock()
            .expect("checkpoint forge mutex poisoned")
            .pending
            .as_ref()
            .is_some_and(|a| a.path.is_some())
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
    /// tensor count reaches the schema's total (params + the f32 buffer
    /// subset), take the armed capture + accumulated tensors and spawn a
    /// **detached** writer — bundle write, consensus-callback fire, or both —
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
        let (armed, payloads, outer) = {
            let mut st = self.inner.lock().expect("checkpoint forge mutex poisoned");
            if st.pending.is_none() {
                return; // not armed — drop this cycle's frame
            }
            // MOVE the frame's tensors into the accumulator (no copy).
            st.accumulated.extend(frame.tensors);
            if st.accumulated.len() < want {
                return; // more model reduces to come this cycle (buffers)
            }
            let armed = st.pending.take().expect("armed checked above");
            let payloads = std::mem::take(&mut st.accumulated);
            // Outer-optimizer momentum for this cycle, if any (None for
            // stateless OuterAvg => no `.outer.fdl`).
            let outer = st.pending_outer.take();
            (armed, payloads, outer)
        };
        if payloads.len() != want {
            // Overshoot: more tensors arrived than the schema expects — a wiring
            // bug. Skip rather than write a misaligned checkpoint.
            eprintln!(
                "flodl ddp: consensus checkpoint accumulated {} tensors but schema \
                 expects {} ({} params + {} f32 buffers); checkpoint skipped",
                payloads.len(),
                want,
                schema.param_names.len(),
                schema.f32_buffer_idx.len(),
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
                     {} params; .outer.fdl skipped",
                    o.len(),
                    schema.param_names.len(),
                );
            }
            ok
        });
        let schema = schema.clone();
        let consensus_fn = armed
            .user_version
            .is_some()
            .then(|| self.consensus_fn.clone())
            .flatten();
        let spawn = std::thread::Builder::new()
            .name("flodl-ckpt-writer".to_string())
            .spawn(move || {
                if let Some(path) = armed.path.as_ref() {
                    if let Err(e) = write_consensus_fdl(&schema, &payloads, path) {
                        eprintln!(
                            "flodl ddp: consensus checkpoint write to {} failed: {e}",
                            path.display(),
                        );
                    }
                    // Outer-optimizer momentum rides the same writer:
                    // `<stem>.fdl` -> `<stem>.outer.fdl`, same atomic-rename
                    // commit. Written after the model so a present
                    // `.outer.fdl` implies a present `.fdl`.
                    if let Some(outer_payloads) = outer {
                        let outer_path = path.with_extension("outer.fdl");
                        if let Err(e) = write_outer_momentum_fdl(&outer_payloads, &outer_path) {
                            eprintln!(
                                "flodl ddp: outer-momentum checkpoint write to {} failed: {e}",
                                outer_path.display(),
                            );
                        }
                    }
                }
                // Consensus-callback fire, AFTER the bundle write (a callback
                // that reads the bundle back sees this cycle's write). Errors
                // (including caught user panics) report loudly and never kill
                // the run — parity with the elected-rank CheckpointResult
                // error path on NCCL.
                if let (Some(version), Some(f)) = (armed.user_version, consensus_fn)
                    && let Err(e) = f(version, &schema, &payloads)
                {
                    eprintln!("flodl ddp: checkpoint_fn (v{version}, consensus) failed: {e}");
                }
            });
        match spawn {
            Ok(_) => {
                self.writers_spawned
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => {
                eprintln!("flodl ddp: failed to spawn checkpoint writer thread: {e}");
            }
        }
    }
}

/// Write a named, loadable `.fdl` straight from the accumulated averaged
/// payloads + the static schema. Positional pairing: the first
/// `param_names.len()` tensors are params, the rest the f32 buffer subset —
/// the `GpuWorker::snapshot_params` / reduce order. Emits the payloads' raw native
/// bytes directly (no `Tensor` reconstruction) and commits with a temp-file +
/// atomic rename, so a crash mid-write never leaves a torn `.fdl` that resume
/// could mistake for valid.
fn write_consensus_fdl(
    schema: &ModelSchema,
    payloads: &[TensorPayload],
    path: &Path,
) -> Result<()> {
    if payloads.len() != schema.tensor_count() {
        return Err(TensorError::new(&format!(
            "checkpoint_forge: accumulated {} tensors but schema expects \
             {} ({} params + {} f32 buffers) — schema/accumulation mismatch",
            payloads.len(),
            schema.tensor_count(),
            schema.param_names.len(),
            schema.f32_buffer_idx.len(),
        )));
    }
    // Pre-convert u32 wire shapes to the i64 dims the .fdl format stores;
    // entries borrow these.
    let shapes: Vec<Vec<i64>> = payloads
        .iter()
        .map(|p| p.shape.iter().map(|&d| d as i64).collect())
        .collect();
    // Positional keys: the first `param_names.len()` payloads are params
    // (`p{i}`), the rest the f32 buffer subset the frame carries — each keyed
    // by its FULL buffer-list index (`b{f32_buffer_idx[k]}`), so the bundle
    // stays load-compatible with rank-written bundles that carry every
    // buffer. Non-f32 buffers never reach the frame; on load they surface in
    // `LoadReport::missing` and the model keeps its constructed values (the
    // same passthrough semantics the reduce gives them). Synthetic + unique
    // keys — never the model's own (possibly repeated) names. Matches
    // `load_consensus_checkpoint`.
    let param_count = schema.param_names.len();
    let keys: Vec<String> = (0..payloads.len())
        .map(|i| {
            if i < param_count {
                consensus_param_key(i)
            } else {
                consensus_buffer_key(schema.f32_buffer_idx[i - param_count])
            }
        })
        .collect();
    // Consensus checkpoints are ALWAYS f32 on disk: a bf16-wire payload
    // (see `ElCheConfig::bf16_wire`) is upcast here — exact, since bf16
    // is a truncated f32 — so resume never depends on the wire dtype
    // the run happened to use. f32 payloads keep the zero-copy borrow.
    let upcast: Vec<Option<Vec<u8>>> = payloads
        .iter()
        .map(|p| {
            if p.dtype == DTYPE_F32 {
                if p.is_elided() && p.numel() > 0 {
                    // Wire zero-elision: no bytes to borrow — materialize
                    // the zeros for disk. Unreachable from the tap (a
                    // realized round's consensus is never elided), kept
                    // so the writer is total over every valid payload.
                    Ok(Some(vec![0u8; p.numel() * 4]))
                } else {
                    Ok(None)
                }
            } else {
                let vals = crate::distributed::controller::payload_to_f32(p)?;
                Ok(Some(
                    crate::distributed::controller::f32_slice_to_payload_bytes(&vals, DTYPE_F32)?,
                ))
            }
        })
        .collect::<Result<_>>()?;
    let mut entries = Vec::with_capacity(payloads.len());
    for (i, p) in payloads.iter().enumerate() {
        entries.push(RawCheckpointEntry {
            name: keys[i].as_str(),
            shape: &shapes[i],
            dtype_tag: dtype_tag(DType::Float32),
            raw: upcast[i].as_deref().unwrap_or(&p.bytes),
        });
    }
    let path_str = path.to_str().ok_or_else(|| {
        TensorError::new(&format!(
            "checkpoint_forge: non-utf8 checkpoint path {}",
            path.display(),
        ))
    })?;
    crate::distributed::checkpoint_meta::ensure_parent_dir(path);
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
    crate::distributed::checkpoint_meta::ensure_parent_dir(path);
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
        targets.push((
            consensus_param_key(i),
            Parameter::new(zeros, "outer_momentum"),
        ));
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
            f32_buffer_idx: vec![0],
        };
        let forge = CheckpointForge::new(Some(schema), None);
        assert!(forge.can_write_model());

        let w = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = cpu_tensor(&[5.0, 6.0], &[2]);
        let rm = cpu_tensor(&[7.0, 8.0], &[2]);
        // Two model reduces: params frame (w, b) then buffers frame (rm).
        let params_frame = tensors_to_round_frame(&[&w, &b], DTYPE_F32).unwrap();
        let buffers_frame = tensors_to_round_frame(&[&rm], DTYPE_F32).unwrap();

        let dir = std::env::temp_dir().join(format!("flodl_forge_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");

        forge.arm(Some(path.clone()), None);
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
            &[
                ("p0".to_string(), tw.clone()),
                ("p1".to_string(), tb.clone()),
            ],
            &[("b0".to_string(), trm.clone())],
            None,
        )
        .unwrap();

        assert_eq!(
            tw.variable.data().to_f32_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(tb.variable.data().to_f32_vec().unwrap(), vec![5.0, 6.0]);
        assert_eq!(trm.get().to_f32_vec().unwrap(), vec![7.0, 8.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A model with a non-f32 buffer: the frame carries the f32 subset only,
    /// so the forge completes at params + f32 buffers and keys each written
    /// buffer by its FULL-list index. The non-f32 buffer's key is absent from
    /// the bundle — a positional load reports it `missing` and the model
    /// keeps its constructed value (the reduce's own passthrough semantics).
    #[test]
    fn non_f32_buffer_excluded_but_keys_stay_full_index() {
        // Buffers: [step (Int64, idx 0), running (f32, idx 1)] — only
        // `running` rides the frame, keyed b1 (never b0).
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec!["step".to_string(), "running".to_string()],
            f32_buffer_idx: vec![1],
        };
        let forge = CheckpointForge::new(Some(schema), None);

        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        let running = cpu_tensor(&[0.5], &[1]);
        let params_frame = tensors_to_round_frame(&[&w], DTYPE_F32).unwrap();
        let buffers_frame = tensors_to_round_frame(&[&running], DTYPE_F32).unwrap();

        let dir = std::env::temp_dir().join(format!("flodl_forge_nonf32_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");

        forge.arm(Some(path.clone()), None);
        forge.accumulate(params_frame); // 1 of 2
        assert!(!path.exists(), "no write before the f32 subset completes");
        forge.accumulate(buffers_frame); // completes 2/2 → detached write

        let mut found = false;
        for _ in 0..200 {
            if path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "f32-subset accumulation produced the .fdl");

        // Positional load over the FULL buffer list: b0 (int) is missing from
        // the bundle and keeps its constructed value; b1 loads the consensus.
        use crate::nn::{Buffer, Parameter};
        let tw = Parameter::new(cpu_tensor(&[0.0; 2], &[2]), "w");
        let tstep = Buffer::new(Tensor::from_i64(&[7], &[1], Device::CPU).unwrap(), "step");
        let trun = Buffer::new(cpu_tensor(&[0.0], &[1]), "running");
        let report = crate::nn::load_checkpoint_file(
            path.to_str().unwrap(),
            &[("p0".to_string(), tw.clone())],
            &[
                ("b0".to_string(), tstep.clone()),
                ("b1".to_string(), trun.clone()),
            ],
            None,
        )
        .unwrap();
        assert_eq!(tw.variable.data().to_f32_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(trun.get().to_f32_vec().unwrap(), vec![0.5]);
        assert_eq!(
            tstep.get().to_i64_vec().unwrap(),
            vec![7],
            "non-f32 buffer keeps its constructed value"
        );
        assert_eq!(report.missing, vec!["b0".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A bf16-wire run's consensus frames write an EXACT F32 checkpoint:
    /// the forge upcasts bf16 payloads (lossless) so resume never
    /// depends on the wire dtype. Values chosen bf16-representable.
    #[test]
    fn bf16_frames_write_f32_checkpoint() {
        use crate::distributed::controller::DTYPE_BF16;
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let forge = CheckpointForge::new(Some(schema), None);
        let w = cpu_tensor(&[1.5, -2.0, 0.25, 42.0], &[4]);
        let frame = tensors_to_round_frame(&[&w], DTYPE_BF16).unwrap();
        assert_eq!(frame.tensors[0].dtype, DTYPE_BF16);

        let dir = std::env::temp_dir().join(format!("flodl_forge_bf16_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");
        forge.arm(Some(path.clone()), None);
        forge.accumulate(frame); // completes 1/1 → detached write

        let mut found = false;
        for _ in 0..200 {
            if path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "bf16 accumulation produced the .fdl");

        use crate::nn::Parameter;
        let tw = Parameter::new(cpu_tensor(&[0.0; 4], &[4]), "w");
        crate::nn::load_checkpoint_file(
            path.to_str().unwrap(),
            &[("p0".to_string(), tw.clone())],
            &[],
            None,
        )
        .unwrap();
        let loaded = tw.variable.data();
        assert_eq!(loaded.dtype(), crate::tensor::DType::Float32);
        assert_eq!(loaded.to_f32_vec().unwrap(), vec![1.5, -2.0, 0.25, 42.0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn outer_momentum_writes_sidecar_and_round_trips() {
        // When outer momentum is stashed, the completing cycle writes both
        // <stem>.fdl and <stem>.outer.fdl, and the sidecar loads back.
        let schema = ModelSchema {
            param_names: vec!["w".to_string(), "b".to_string()],
            buffer_names: vec!["running_mean".to_string()],
            f32_buffer_idx: vec![0],
        };
        let forge = CheckpointForge::new(Some(schema), None);

        let w = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = cpu_tensor(&[5.0, 6.0], &[2]);
        let rm = cpu_tensor(&[7.0, 8.0], &[2]);
        // Momentum is one tensor per PARAM (not buffers): 2 here.
        let mw = cpu_tensor(&[0.1, 0.2, 0.3, 0.4], &[2, 2]);
        let mb = cpu_tensor(&[0.5, 0.6], &[2]);

        let dir = std::env::temp_dir().join(format!("flodl_forge_outer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("consensus.fdl");

        forge.arm(Some(path.clone()), None);
        assert!(forge.is_armed());
        // Stash momentum, then complete the model accumulation (params, buffers).
        forge.stash_outer_momentum(
            tensors_to_round_frame(&[&mw, &mb], DTYPE_F32)
                .unwrap()
                .tensors,
        );
        forge.accumulate(tensors_to_round_frame(&[&w, &b], DTYPE_F32).unwrap());
        forge.accumulate(tensors_to_round_frame(&[&rm], DTYPE_F32).unwrap()); // completes

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
            &[
                ("p0".to_string(), t0.clone()),
                ("p1".to_string(), t1.clone()),
            ],
            &[],
            None,
        )
        .unwrap();
        assert_eq!(
            t0.variable.data().to_f32_vec().unwrap(),
            vec![0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(t1.variable.data().to_f32_vec().unwrap(), vec![0.5, 0.6]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_outer_momentum_writes_no_sidecar() {
        // Stateless outer optimizer stashes nothing => no .outer.fdl.
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let forge = CheckpointForge::new(Some(schema), None);
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        let dir = std::env::temp_dir().join(format!("flodl_forge_noouter_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.fdl");

        forge.arm(Some(path.clone()), None);
        forge.accumulate(tensors_to_round_frame(&[&w], DTYPE_F32).unwrap()); // completes, no stash

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
        assert!(
            !outer_path.exists(),
            "no .outer.fdl when no momentum stashed"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A user-fire-only arm (no `save_path`): the consensus callback fires
    /// with the armed version and this cycle's payloads, no `.fdl` is
    /// written, and `is_armed` (the outer-momentum gate, bundle-only) stays
    /// false throughout.
    #[test]
    fn user_fire_without_bundle_fires_consensus_fn_only() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let f: ConsensusModelFn = Arc::new(move |version, schema, payloads| {
            let vals = crate::distributed::controller::payload_to_f32(&payloads[0])?;
            tx.send((version, schema.param_names.len(), vals)).ok();
            Ok(())
        });
        let forge = CheckpointForge::new(Some(schema), Some(f));
        assert!(forge.has_consensus_fn());

        forge.arm(None, Some(3));
        assert!(!forge.is_armed(), "a user-fire arm is not a bundle arm");
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        forge.accumulate(tensors_to_round_frame(&[&w], DTYPE_F32).unwrap());
        let (version, nparams, vals) = rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("consensus fn fired");
        assert_eq!(version, 3);
        assert_eq!(nparams, 1);
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    /// A cadence arm with BOTH destinations: the bundle writes and the
    /// callback fires, callback strictly after the write (it may read the
    /// bundle back).
    #[test]
    fn bundle_and_user_fire_arm_does_both() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let f: ConsensusModelFn = Arc::new(move |version, _schema, _payloads| {
            tx.send(version).ok();
            Ok(())
        });
        let forge = CheckpointForge::new(Some(schema), Some(f));
        let dir = std::env::temp_dir().join(format!("flodl_forge_both_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.fdl");

        forge.arm(Some(path.clone()), Some(4));
        assert!(forge.is_armed(), "bundle armed");
        let w = cpu_tensor(&[1.0], &[1]);
        forge.accumulate(tensors_to_round_frame(&[&w], DTYPE_F32).unwrap());
        let version = rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("consensus fn fired");
        assert_eq!(version, 4);
        assert!(
            path.exists(),
            "bundle written before the callback fired (same thread, in order)"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The launch wrap end-to-end: a fresh CPU probe is built per fire, the
    /// consensus payloads land in its params positionally, and the user's
    /// typed callback sees exactly the consensus values.
    #[test]
    fn consensus_checkpoint_fn_loads_consensus_into_probe_model() {
        use crate::nn::Linear;
        type Seen = Option<(u64, Vec<f32>, Vec<f32>)>;
        let factory = |dev| Linear::on_device(2, 1, dev);
        let got: Arc<Mutex<Seen>> = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&got);
        let user: crate::distributed::ddp_run::CheckpointFn<Linear> =
            Arc::new(move |version, model| {
                let ps = model.parameters();
                *sink.lock().unwrap() = Some((
                    version,
                    ps[0].variable.data().to_f32_vec()?,
                    ps[1].variable.data().to_f32_vec()?,
                ));
                Ok(())
            });
        let wrap = consensus_checkpoint_fn(factory, user);

        let probe = Linear::on_device(2, 1, Device::CPU).unwrap();
        let schema = ModelSchema::from_module(&probe);
        let w = cpu_tensor(&[0.5, -1.5], &[1, 2]);
        let b = cpu_tensor(&[0.25], &[1]);
        let frame = tensors_to_round_frame(&[&w, &b], DTYPE_F32).unwrap();
        wrap(7, &schema, &frame.tensors).unwrap();

        let (version, wv, bv) = got.lock().unwrap().take().expect("user fn fired");
        assert_eq!(version, 7);
        assert_eq!(wv, vec![0.5, -1.5]);
        assert_eq!(bv, vec![0.25]);
    }

    /// A panicking user callback surfaces as an error (the forge's writer
    /// thread reports it), never as a thread death.
    #[test]
    fn consensus_checkpoint_fn_catches_user_panic() {
        use crate::nn::Linear;
        let factory = |dev| Linear::on_device(2, 1, dev);
        let user: crate::distributed::ddp_run::CheckpointFn<Linear> =
            Arc::new(|_, _| panic!("boom"));
        let wrap = consensus_checkpoint_fn(factory, user);

        let probe = Linear::on_device(2, 1, Device::CPU).unwrap();
        let schema = ModelSchema::from_module(&probe);
        let w = cpu_tensor(&[0.5, -1.5], &[1, 2]);
        let b = cpu_tensor(&[0.25], &[1]);
        let frame = tensors_to_round_frame(&[&w, &b], DTYPE_F32).unwrap();
        let err = wrap(1, &schema, &frame.tensors).unwrap_err();
        assert!(err.to_string().contains("panicked"), "got: {err}");
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    /// A stem whose parent directory does not exist yet: the writer creates
    /// it (`mkdir -p` semantics) instead of failing the run's only persist
    /// with ENOENT — a fresh run layout is normal, not an error.
    #[test]
    fn writer_creates_missing_parent_dirs() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let forge = CheckpointForge::new(Some(schema), None);
        let dir = std::env::temp_dir().join(format!("flodl_forge_mkdirp_{}", std::process::id()));
        // Two levels of missing parents below an existing root.
        let path = dir.join("nested").join("deeper").join("c.fdl");

        forge.arm(Some(path.clone()), None);
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        forge.accumulate(tensors_to_round_frame(&[&w], DTYPE_F32).unwrap());

        let mut found = false;
        for _ in 0..200 {
            if path.exists() {
                found = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(found, "missing parents were created and the .fdl written");

        // The meta writer shares the semantics.
        let meta_path = dir.join("also").join("new").join("c.meta.json");
        crate::distributed::CheckpointMeta::new(
            0,
            0,
            0,
            2,
            crate::distributed::SaveReason::Checkpoint,
        )
        .write_to_file(&meta_path)
        .expect("meta write creates its parents");
        assert!(meta_path.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unarmed_accumulate_is_noop() {
        let schema = ModelSchema {
            param_names: vec!["w".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let forge = CheckpointForge::new(Some(schema), None);
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        // No arm() — accumulate must not crash or write.
        forge.accumulate(tensors_to_round_frame(&[&w], DTYPE_F32).unwrap());
    }

    #[test]
    fn arm_clears_partial_accumulation() {
        // A new arm supersedes a missed one: a stale partial frame must not
        // bleed into the next cycle's model.
        let schema = ModelSchema {
            param_names: vec!["w".to_string(), "b".to_string()],
            buffer_names: vec![],
            f32_buffer_idx: vec![],
        };
        let forge = CheckpointForge::new(Some(schema), None);
        let stale = cpu_tensor(&[9.0], &[1]);
        let w = cpu_tensor(&[1.0, 2.0], &[2]);
        let b = cpu_tensor(&[3.0], &[1]);

        let dir = std::env::temp_dir().join(format!("flodl_forge_arm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.fdl");

        forge.arm(Some(path.clone()), None);
        forge.accumulate(tensors_to_round_frame(&[&stale], DTYPE_F32).unwrap()); // partial 1/2
        // Re-arm: the stale partial is dropped, cycle restarts.
        forge.arm(Some(path.clone()), None);
        forge.accumulate(tensors_to_round_frame(&[&w], DTYPE_F32).unwrap()); // 1/2
        forge.accumulate(tensors_to_round_frame(&[&b], DTYPE_F32).unwrap()); // 2/2 → write

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
            &[
                ("p0".to_string(), tw.clone()),
                ("p1".to_string(), tb.clone()),
            ],
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
            f32_buffer_idx: vec![],
        };
        // Two payloads but schema expects 1.
        let a = cpu_tensor(&[1.0], &[1]);
        let b = cpu_tensor(&[2.0], &[1]);
        let frame = tensors_to_round_frame(&[&a, &b], DTYPE_F32).unwrap();
        let path = std::env::temp_dir().join("flodl_forge_mismatch.fdl");
        let err = write_consensus_fdl(&schema, &frame.tensors, &path).unwrap_err();
        assert!(err.to_string().contains("mismatch"), "got: {err}");
    }
}
