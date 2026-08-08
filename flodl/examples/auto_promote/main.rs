//! Auto-promote — the same plain training code on 1 CPU, 1 GPU, or N GPUs.
//!
//! This binary contains ZERO distributed code: no cluster config, no rank
//! handling, no launcher wiring. `Trainer::builder(...).run()` decides the
//! topology at runtime:
//!
//! - 0-1 visible GPUs → ordinary single-device training, inline.
//! - 2+ visible GPUs  → auto-promoted to the process-per-rank DDP path:
//!   `run()` synthesizes a localhost cluster, re-execs this same binary as
//!   per-rank (and per-host relay) children, and drives the cluster
//!   coordinator. The children's `run()` calls detect their role from the
//!   environment and become ranks — `main()` needs no awareness of any of
//!   it.
//!
//! Scope down with `CUDA_VISIBLE_DEVICES=0` to force the single-GPU path.
//!
//! The dataset here is generated ANALYTICALLY (no RNG): every spawned rank
//! process re-runs `main()` and must construct the identical dataset, and
//! seeding an RNG before `run()` is off-limits in auto-promoted binaries
//! (touching CUDA state before `run()` corrupts the spawned children's
//! context — the "no CUDA before `Trainer::run`" rule; see
//! `flodl::sys::detect_gpus` for the CUDA-free way to probe hardware).
//!
//! Run: `cargo run --release --example auto_promote --features cuda`
//! (CPU-only builds work too: `cargo run --example auto_promote`).

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{DType, Device, Result, Tensor};
use flodl::*;

/// Two deterministic, linearly-separable blobs in 8 dimensions.
///
/// Sample `i` belongs to class `i % 2`; its features are the class
/// centroid (±0.5 in every dimension) plus a small analytic perturbation
/// derived from `(i, j)` — fully reproducible across processes without
/// any RNG.
struct TwoBlobs {
    xs: Tensor,
    ys: Tensor,
    len: usize,
}

const DIM: usize = 8;

impl TwoBlobs {
    fn new(len: usize) -> Result<Self> {
        let mut xs = Vec::with_capacity(len * DIM);
        let mut ys = Vec::with_capacity(len);
        for i in 0..len {
            let class = (i % 2) as f32;
            let center = class - 0.5; // class 0 → -0.5, class 1 → +0.5
            for j in 0..DIM {
                // Deterministic jitter in (-0.25, 0.25).
                let jitter = ((((i * 31 + j * 17) % 97) as f32) / 97.0 - 0.5) * 0.5;
                xs.push(center + jitter);
            }
            ys.push(class);
        }
        Ok(Self {
            xs: Tensor::from_f32(&xs, &[len as i64, DIM as i64], Device::CPU)?,
            ys: Tensor::from_f32(&ys, &[len as i64], Device::CPU)?,
            len,
        })
    }
}

impl BatchDataSet for TwoBlobs {
    fn len(&self) -> usize {
        self.len
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let idx: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        let idx = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
        Ok(vec![
            self.xs.index_select(0, &idx)?,
            self.ys.index_select(0, &idx)?,
        ])
    }
}

fn build_model(device: Device) -> Result<impl Module + use<>> {
    FlowBuilder::from(Linear::on_device(DIM as i64, 16, device)?)
        .through(GELU)
        .through(Linear::on_device(16, 2, device)?)
        .build()
}

fn train_step(model: &impl Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].clone(), false);
    let target = Variable::new(batch[1].to_dtype(DType::Int64)?, false);
    cross_entropy_loss(&model.forward(&input)?, &target)
}

fn main() -> Result<()> {
    // CUDA-free hardware probe (allowed before `run()`).
    let gpus = flodl::sys::detect_gpus();
    println!(
        "auto_promote: {} visible GPU(s) — {}",
        gpus.len(),
        if gpus.len() >= 2 {
            "expecting auto-promotion to process-per-rank DDP"
        } else {
            "expecting single-device training"
        }
    );

    let dataset: Arc<dyn BatchDataSet> = Arc::new(TwoBlobs::new(512)?);

    let handle = Trainer::builder(build_model, |params| Adam::new(params, 1e-2), train_step)
        .dataset(dataset)
        .batch_size(32)
        .num_epochs(3)
        .run()?;

    let state = handle.join()?;
    // Launcher mode returns an empty TrainedState (ranks are separate
    // processes — see `DdpHandle::join`); single-device returns the
    // trained tensors.
    println!(
        "auto_promote: done (returned {} params, {} buffers)",
        state.params.len(),
        state.buffers.len()
    );
    Ok(())
}
