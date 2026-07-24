//! OLMo-150M on olmo-mix (books slice).
//!
//! Ref: allenai/OLMo `configs/tiny/OLMo-150M.yaml` — d_model=768,
//! 12 layers, 12 heads, RMSNorm (eps 1e-6), SwiGLU (mlp_ratio 8),
//! RoPE, no biases, no weight tying, vocab 50,280 padded to 50,304
//! embeddings. AdamW lr 6e-4 betas (0.9, 0.95) wd 0.1, cosine decay
//! to 0.1x with warmup.
//!
//! Bench deviations from the reference config, applied identically to
//! the PyTorch control arm (the comparison is self-controlled — AI2
//! publishes no curves for the tiny configs):
//! - seq 256 instead of 4096 (Pascal 6GB fit: attention scores are
//!   materialized, not flash — seq 512 x batch 8 measured 10.3 GB);
//! - `MultiheadAttention` carries projection biases (the reference has
//!   none; ~37k params on ~190M, negligible);
//! - train data = the leading slice of one olmo-mix books shard (see
//!   `download::OLMO_TRAIN_BYTES`), eval = OLMo's C4-en validation
//!   shard, so eval is a held-out-domain CE loss.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::datasets::{TokenDtype, TokenShards};
use flodl::data::BatchDataSet;
use flodl::nn::{
    AdamW, CosineScheduler, Embedding, Linear, Module, MultiheadAttention,
    Parameter, RMSNorm, RotaryEmbedding, WarmupScheduler,
};
use flodl::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;
use crate::download::{ensure_olmo_eval, ensure_olmo_train, OLMO_TRAIN_BYTES};

const D_MODEL: i64 = 768;
const N_HEADS: i64 = 12;
const HEAD_DIM: i64 = D_MODEL / N_HEADS; // 64
const N_LAYERS: usize = 12;
/// OLMo mlp_ratio 8: ff_proj widens to 8 * d_model, the SwiGLU gate
/// halves it to an effective intermediate of 4 * d_model.
const MLP_HIDDEN: i64 = 8 * D_MODEL; // 6144, gate-split to 3072
const EMBEDDING_SIZE: i64 = 50_304; // vocab 50,280 padded (OLMo convention)
const LN_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000.0;
const SEQ_LEN: usize = 256;

pub fn def() -> ModelDef {
    ModelDef {
        name: "olmo",
        description: "OLMo-150M (configs/tiny) LM pretraining on an olmo-mix books slice",
        build: build_model,
        dataset: make_dataset,
        // Windows over the staged slice: tokens = bytes / 2 (u16), a
        // window needs seq_len + 1 tokens. Pure arithmetic, so cluster
        // launchers size partitions without touching the data.
        dataset_size_hint: |_| Ok(((OLMO_TRAIN_BYTES / 2 - 1) / SEQ_LEN as u64) as usize),
        train_fn: train_step,
        eval_fn: Some(eval_loss),
        test_dataset: Some(make_eval_dataset),
        augment_fn: None,
        // OLMo-150M: AdamW lr 6e-4, betas (0.9, 0.95), wd 0.1, eps 1e-8
        optimizer: |p, lr| {
            Box::new(AdamW::with_groups(0.1).betas(0.9, 0.95).eps(1e-8).group(p, lr).build())
        },
        // Cosine to alpha_f = 0.1x base LR; the reference warms up 5k of
        // 407k steps (~1.2%) — at bench scale use 5% of total batches.
        scheduler: Some(|lr, total, _world_size| {
            Box::new(WarmupScheduler::new(
                CosineScheduler::new(lr, lr * 0.1, total),
                lr,
                (total / 20).max(1),
            ))
        }),
        reference: "OLMo-150M tiny config, self-controlled vs the PyTorch arm \
                    (no published tiny curves); eval = C4-en val CE loss \
                    ([OLMo](https://github.com/allenai/OLMo), configs/tiny)",
        eval_higher_is_better: false,
        published_eval: None,
        needs_baseline_eval: false,
        defaults: ModelDefaults {
            epochs: 3,
            batches_per_epoch: 0, // real data: epoch = the staged slice
            // Pascal 6GB envelope: measured 6.3 GB at seq 256 x batch 8,
            // ~4.7 GB at batch 4 (params + AdamW are 3 GB of it).
            batch_size: 4,
            lr: 6.0e-4,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    Ok(Box::new(Olmo::new(device)?))
}

fn make_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let shard = ensure_olmo_train(&cfg.data_dir)?;
    Ok(Arc::new(TokenShards::open_raw(&[shard], TokenDtype::U16, SEQ_LEN)?))
}

fn make_eval_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let shard = ensure_olmo_eval(&cfg.data_dir)?;
    Ok(Arc::new(TokenShards::open_raw(&[shard], TokenDtype::U16, SEQ_LEN)?))
}

fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].to_dtype(DType::Int64)?, false);
    let target = batch[1].to_dtype(DType::Int64)?;

    let pred = model.forward(&input)?;
    let shape = pred.shape();
    let flat_pred = pred.reshape(&[-1, shape[shape.len() - 1]])?;
    let flat_target = Variable::new(target.reshape(&[-1])?, false);
    flodl::cross_entropy_loss(&flat_pred, &flat_target)
}

/// Held-out C4-en CE loss (lower is better; exp(this) = perplexity).
fn eval_loss(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
    train_step(model, batch)?.item()
}

// ---------------------------------------------------------------------------
// OLMo block: pre-norm attention (RoPE) + pre-norm SwiGLU FFN
// ---------------------------------------------------------------------------

struct OlmoBlock {
    attn_norm: RMSNorm,
    attn: MultiheadAttention,
    ff_norm: RMSNorm,
    ff_proj: Linear,
    ff_out: Linear,
}

impl OlmoBlock {
    fn new(device: Device, rope: &RotaryEmbedding) -> Result<Self> {
        Ok(OlmoBlock {
            attn_norm: RMSNorm::on_device(D_MODEL, device)?.eps(LN_EPS),
            attn: MultiheadAttention::on_device(D_MODEL, N_HEADS, device)?
                .rotary(rope.clone())?,
            ff_norm: RMSNorm::on_device(D_MODEL, device)?.eps(LN_EPS),
            ff_proj: Linear::no_bias_on_device(D_MODEL, MLP_HIDDEN, device)?,
            ff_out: Linear::no_bias_on_device(MLP_HIDDEN / 2, D_MODEL, device)?,
        })
    }

    fn forward(&self, x: &Variable, causal_mask: &Tensor) -> Result<Variable> {
        // Pre-norm attention + residual.
        let normed = self.attn_norm.forward(x)?;
        let attn_out = self.attn.forward_ext(&normed, &normed, &normed, Some(causal_mask))?;
        let x = x.add(&attn_out)?;

        // Pre-norm SwiGLU FFN + residual. OLMo's SwiGLU:
        // h, gate = ff_proj(x).chunk(2); ff_out(silu(gate) * h).
        let normed = self.ff_norm.forward(&x)?;
        let proj = self.ff_proj.forward(&normed)?;
        let halves = proj.chunk(2, -1)?;
        let gated = halves[1].silu()?.mul(&halves[0])?;
        let ff = self.ff_out.forward(&gated)?;
        x.add(&ff)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.attn_norm.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ff_norm.parameters());
        params.extend(self.ff_proj.parameters());
        params.extend(self.ff_out.parameters());
        params
    }
}

// ---------------------------------------------------------------------------
// OLMo-150M
// ---------------------------------------------------------------------------

struct Olmo {
    emb: Embedding,
    blocks: Vec<OlmoBlock>,
    final_norm: RMSNorm,
    head: Linear,
    device: Device,
}

impl Olmo {
    fn new(device: Device) -> Result<Self> {
        let emb = Embedding::on_device(EMBEDDING_SIZE, D_MODEL, device)?;

        // One RoPE table set, shared by every block (Clone shares).
        let rope = RotaryEmbedding::on_device_theta(
            HEAD_DIM, SEQ_LEN as i64, ROPE_THETA, device,
        )?;
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            blocks.push(OlmoBlock::new(device, &rope)?);
        }

        Ok(Olmo {
            emb,
            blocks,
            final_norm: RMSNorm::on_device(D_MODEL, device)?.eps(LN_EPS),
            head: Linear::no_bias_on_device(D_MODEL, EMBEDDING_SIZE, device)?,
            device,
        })
    }
}

impl Module for Olmo {
    fn name(&self) -> &str { "olmo" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let seq_len = input.shape()[1];

        // Token embeddings only — positions come from RoPE inside attention.
        let mut x = self.emb.forward(input)?;

        // Causal mask: upper triangle = true (positions to mask).
        let opts = TensorOptions { dtype: DType::Float32, device: self.device };
        let mask = Tensor::ones(&[seq_len, seq_len], opts)?.triu(1)?;

        for block in &self.blocks {
            x = block.forward(&x, &mask)?;
        }

        let x = self.final_norm.forward(&x)?;
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.emb.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn set_training(&self, _mode: bool) {
        // All OLMo-150M dropout rates are 0.0; nothing is mode-dependent.
    }
}
