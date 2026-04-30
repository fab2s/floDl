//! GPT-nano on Shakespeare.
//!
//! Ref: nanoGPT (Karpathy), Vaswani et al. "Attention Is All You Need".
//! Architecture: 4 pre-norm transformer layers, 4 heads, d_model=128.
//! Expected: CE loss ~1.5-2.0 at convergence.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::{
    Adam, CosineScheduler, Dropout, Embedding, LayerNorm, Linear, Module,
    MultiheadAttention, Parameter, WarmupScheduler,
};
use flodl::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;

const VOCAB_SIZE: i64 = 65; // Shakespeare character set
const D_MODEL: i64 = 128;
const D_FF: i64 = 512;
const N_HEADS: i64 = 4;
const N_LAYERS: usize = 4;
const SEQ_LEN: usize = 128;

pub fn def() -> ModelDef {
    ModelDef {
        name: "gpt-nano",
        description: "GPT-nano on Shakespeare (nanoGPT, loss ~1.5-2.0)",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        eval_fn: Some(eval_loss),
        test_dataset: Some(make_test_dataset),
        augment_fn: None,
        // nanoGPT: Adam lr=3e-4, warmup 20% then cosine decay to 1e-5
        // (published: 10/50 epochs warmup; total is in batches)
        optimizer: |p, lr| Box::new(Adam::new(p, lr)),
        scheduler: Some(|lr, total, _world_size| {
            Box::new(WarmupScheduler::new(
                CosineScheduler::new(lr, 1e-5, total),
                lr,
                total / 5, // 20% warmup
            ))
        }),
        reference: "Shakespeare CE loss ~1.5-2.0, eval=val loss ([nanoGPT](https://github.com/karpathy/nanoGPT), [Vaswani 2017](https://arxiv.org/abs/1706.03762))",
        eval_higher_is_better: false,
        published_eval: None,
        needs_baseline_eval: false,
        defaults: ModelDefaults {
            epochs: 50,
            batches_per_epoch: 0, // full dataset
            batch_size: 64,
            lr: 0.0003,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    Ok(Box::new(GptNano::new(device)?))
}

fn make_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let shakespeare = crate::download::ensure_shakespeare_train(&cfg.data_dir, SEQ_LEN)?;
    Ok(Arc::new(shakespeare))
}

fn make_test_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let shakespeare = crate::download::ensure_shakespeare_test(&cfg.data_dir, SEQ_LEN)?;
    Ok(Arc::new(shakespeare))
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

/// Held-out CE loss (lower is better).
fn eval_loss(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
    train_step(model, batch)?.item()
}

// ---------------------------------------------------------------------------
// Transformer block (pre-norm)
// ---------------------------------------------------------------------------

struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiheadAttention,
    ln2: LayerNorm,
    ff_up: Linear,
    ff_down: Linear,
    dropout: Dropout,
}

impl TransformerBlock {
    fn new(device: Device) -> Result<Self> {
        Ok(TransformerBlock {
            ln1: LayerNorm::on_device(D_MODEL, device)?,
            attn: MultiheadAttention::on_device(D_MODEL, N_HEADS, device)?,
            ln2: LayerNorm::on_device(D_MODEL, device)?,
            ff_up: Linear::on_device(D_MODEL, D_FF, device)?,
            ff_down: Linear::on_device(D_FF, D_MODEL, device)?,
            dropout: Dropout::new(0.1),
        })
    }

    fn forward(&self, x: &Variable, causal_mask: &Tensor) -> Result<Variable> {
        // Pre-norm attention + residual
        let normed = self.ln1.forward(x)?;
        let attn_out = self.attn.forward_ext(&normed, &normed, &normed, Some(causal_mask))?;
        let attn_out = self.dropout.forward(&attn_out)?;
        let x = x.add(&attn_out)?;

        // Pre-norm FFN + residual
        let normed = self.ln2.forward(&x)?;
        let ff_out = self.ff_up.forward(&normed)?;
        let ff_out = ff_out.gelu()?;
        let ff_out = self.ff_down.forward(&ff_out)?;
        let ff_out = self.dropout.forward(&ff_out)?;
        x.add(&ff_out)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.ff_up.parameters());
        params.extend(self.ff_down.parameters());
        params
    }

    fn set_training(&self, mode: bool) {
        self.dropout.set_training(mode);
    }
}

// ---------------------------------------------------------------------------
// GPT-nano
// ---------------------------------------------------------------------------

struct GptNano {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    head: Linear,
    dropout: Dropout,
    device: Device,
}

impl GptNano {
    fn new(device: Device) -> Result<Self> {
        let tok_emb = Embedding::on_device(VOCAB_SIZE, D_MODEL, device)?;
        let pos_emb = Embedding::on_device(SEQ_LEN as i64, D_MODEL, device)?;

        let mut blocks = Vec::with_capacity(N_LAYERS);
        for _ in 0..N_LAYERS {
            blocks.push(TransformerBlock::new(device)?);
        }

        let ln_f = LayerNorm::on_device(D_MODEL, device)?;
        let head = Linear::on_device(D_MODEL, VOCAB_SIZE, device)?;
        let dropout = Dropout::new(0.1);

        Ok(GptNano { tok_emb, pos_emb, blocks, ln_f, head, dropout, device })
    }
}

impl Module for GptNano {
    fn name(&self) -> &str { "gpt_nano" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let seq_len = input.shape()[1];

        // Token + position embeddings
        let tok = self.tok_emb.forward(input)?;
        let pos_idx = Variable::new(
            Tensor::from_i64(
                &(0..seq_len).collect::<Vec<i64>>(),
                &[seq_len],
                self.device,
            )?,
            false,
        );
        let pos = self.pos_emb.forward(&pos_idx)?;
        let mut x = tok.add(&pos)?;
        x = self.dropout.forward(&x)?;

        // Causal mask: upper triangle = true (positions to mask)
        let opts = TensorOptions { dtype: DType::Float32, device: self.device };
        let mask = Tensor::ones(&[seq_len, seq_len], opts)?.triu(1)?;

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, &mask)?;
        }

        // Final layer norm + projection to vocab
        let x = self.ln_f.forward(&x)?;
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.tok_emb.parameters());
        params.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn set_training(&self, mode: bool) {
        self.dropout.set_training(mode);
        for block in &self.blocks {
            block.set_training(mode);
        }
    }
}
