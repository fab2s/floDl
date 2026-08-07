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
    Parameter, RMSNorm, RotaryEmbedding, SwiGLU, WarmupScheduler,
};
use flodl::tensor::{DType, Device, Result, Tensor, TensorOptions};

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;
use crate::download::{ensure_olmo_eval, ensure_olmo_train, OLMO_TRAIN_BYTES};

// `pub(super)` so `olmo_graph` builds from the SAME numbers rather than its
// own copy. A parity claim between the two arms is only worth anything if a
// changed hyperparameter cannot land in one and miss the other.
pub(super) const D_MODEL: i64 = 768;
pub(super) const N_HEADS: i64 = 12;
pub(super) const HEAD_DIM: i64 = D_MODEL / N_HEADS; // 64
pub(super) const N_LAYERS: usize = 12;
/// OLMo mlp_ratio 8: ff_proj widens to 8 * d_model, the SwiGLU gate
/// halves it to an effective intermediate of 4 * d_model.
pub(super) const MLP_HIDDEN: i64 = 8 * D_MODEL; // 6144, gate-split to 3072
pub(super) const EMBEDDING_SIZE: i64 = 50_304; // vocab 50,280 padded (OLMo convention)
pub(super) const LN_EPS: f64 = 1e-6;
pub(super) const ROPE_THETA: f64 = 10_000.0;
pub(super) const SEQ_LEN: usize = 256;

pub fn def() -> ModelDef {
    ModelDef {
        name: "olmo",
        description: "OLMo-150M (configs/tiny) LM pretraining on an olmo-mix books slice",
        build: build_model,
        dataset: make_dataset,
        // Pure arithmetic, so cluster launchers size partitions without
        // touching the data — and the same derivation the ranks use.
        dataset_size_hint: |cfg| Ok(resolve_train_corpus(cfg).windows),
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

/// The staged training corpus, resolved once and shared by the launcher's
/// `dataset_size_hint` and the ranks' `make_dataset`.
///
/// Both must agree exactly — the launcher sizes the coordinator's ledger
/// from the hint while the ranks index the real shard, and a disagreement
/// is a silent partition mismatch. One derivation, for the same reason
/// `pick_space` is one derivation in flodl.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TrainCorpus {
    /// Bytes to stage from the shard head.
    pub bytes: u64,
    /// Windows those bytes yield, i.e. `dataset.len()`.
    pub windows: usize,
    /// Tokens asked for, before snapping.
    pub requested_tokens: u64,
    /// Tokens actually staged.
    pub staged_tokens: u64,
}

impl TrainCorpus {
    pub fn was_snapped(&self) -> bool {
        self.requested_tokens != self.staged_tokens
    }

    /// Say what was staged, at most once per process.
    ///
    /// Both `dataset_size_hint` and `make_dataset` resolve the corpus, and
    /// on a cluster every rank resolves it too. Once per process means the
    /// launcher and each rank each state their geometry, which is also the
    /// cheapest way to see that they agree.
    fn announce_once(&self) {
        static SAID: std::sync::Once = std::sync::Once::new();
        if !self.was_snapped() {
            return;
        }
        SAID.call_once(|| {
            eprintln!(
                "  olmo corpus: {} tokens requested -> {} staged ({} windows at seq {}), \
                 so a pass divides into equal epochs of whole batches",
                self.requested_tokens, self.staged_tokens, self.windows, SEQ_LEN,
            );
        });
    }
}

/// Size the staged corpus so one data pass divides exactly into whole
/// batched events.
///
/// A window needs `SEQ_LEN + 1` tokens, so `windows = (tokens - 1) /
/// SEQ_LEN`. Snapping rounds that window count to the nearest multiple of
/// `epoch_splits * batch_size` and derives the byte count back from it, so
/// the pass splits into equal events of whole batches with no remainder
/// dropped anywhere.
///
/// Snapping to a multiple of `batch_size` does make the staged corpus
/// depend on the batch size, so two runs at different batch sizes stage
/// corpora differing by under one event. Nothing is excluded (the staged
/// corpus IS the dataset) but the shuffle differs, which is why the run
/// reports the geometry it settled on rather than assuming it.
pub(super) fn resolve_train_corpus(cfg: &DatasetConfig) -> TrainCorpus {
    let requested_tokens = cfg.train_tokens.unwrap_or(OLMO_TRAIN_BYTES / 2);
    let multiple = (cfg.epoch_splits.max(1) * cfg.batch_size.max(1)) as u64;
    let seq = SEQ_LEN as u64;

    let want = requested_tokens.saturating_sub(1) / seq;
    // Nearest multiple, but never zero: a corpus has to hold one event.
    let windows = (((want + multiple / 2) / multiple) * multiple).max(multiple);
    let staged_tokens = windows * seq + 1;

    let corpus = TrainCorpus {
        bytes: staged_tokens * 2,
        windows: windows as usize,
        requested_tokens,
        staged_tokens,
    };
    corpus.announce_once();
    corpus
}

pub(super) fn make_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let corpus = resolve_train_corpus(cfg);
    let shard = ensure_olmo_train(&cfg.data_dir, corpus.bytes)?;
    Ok(Arc::new(TokenShards::open_raw(&[shard], TokenDtype::U16, SEQ_LEN)?))
}

pub(super) fn make_eval_dataset(cfg: &DatasetConfig) -> Result<Arc<dyn BatchDataSet>> {
    let shard = ensure_olmo_eval(&cfg.data_dir)?;
    Ok(Arc::new(TokenShards::open_raw(&[shard], TokenDtype::U16, SEQ_LEN)?))
}

pub(super) fn train_step(model: &dyn Module, batch: &[Tensor]) -> Result<Variable> {
    let input = Variable::new(batch[0].to_dtype(DType::Int64)?, false);
    let target = batch[1].to_dtype(DType::Int64)?;

    let pred = model.forward(&input)?;
    let shape = pred.shape();
    let flat_pred = pred.reshape(&[-1, shape[shape.len() - 1]])?;
    let flat_target = Variable::new(target.reshape(&[-1])?, false);
    flodl::cross_entropy_loss(&flat_pred, &flat_target)
}

/// Held-out C4-en CE loss (lower is better; exp(this) = perplexity).
pub(super) fn eval_loss(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
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
        //
        // Shares `nn::SwiGLU` with the graph arm ON PURPOSE. The eager/graph
        // comparison is an experiment, and it should vary ONE thing — the
        // engine. Keeping a second, inline copy of the gate here would vary
        // two, so a divergence could not be attributed. The independent
        // implementation that keeps the gate honest is `scripts/olmo_control.py`,
        // which is a different language and framework entirely.
        let normed = self.ff_norm.forward(&x)?;
        let proj = self.ff_proj.forward(&normed)?;
        let gated = SwiGLU.forward(&proj)?;
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

#[cfg(test)]
mod corpus_tests {
    use super::*;
    use crate::models::DataSource;

    fn cfg(train_tokens: Option<u64>, epoch_splits: usize, batch_size: usize) -> DatasetConfig {
        DatasetConfig {
            seed: 42,
            data_dir: std::path::PathBuf::from("data"),
            virtual_len: 0,
            pool_size: 0,
            data_source: DataSource::Ram,
            train_tokens,
            epoch_splits,
            batch_size,
        }
    }

    /// The staged byte count must yield exactly the promised window count.
    /// The launcher sizes the coordinator's ledger from `windows` while the
    /// ranks open the real shard; a round-trip that loses a window is a
    /// silent partition mismatch.
    #[test]
    fn bytes_round_trip_to_the_promised_window_count() {
        for (tokens, splits, bs) in
            [(20_000_000u64, 20usize, 4usize), (2_097_152, 1, 4), (500_000, 7, 8), (1_000, 1, 1)]
        {
            let c = resolve_train_corpus(&cfg(Some(tokens), splits, bs));
            let from_bytes = ((c.bytes / 2 - 1) / SEQ_LEN as u64) as usize;
            assert_eq!(
                from_bytes, c.windows,
                "tokens={tokens} splits={splits} bs={bs}: bytes imply {from_bytes} windows, \
                 corpus promised {}",
                c.windows,
            );
        }
    }

    #[test]
    fn a_pass_divides_into_whole_batched_events() {
        for (tokens, splits, bs) in
            [(20_000_000u64, 20usize, 4usize), (2_097_152, 1, 4), (500_000, 7, 8), (77, 3, 5)]
        {
            let c = resolve_train_corpus(&cfg(Some(tokens), splits, bs));
            assert_eq!(
                c.windows % (splits * bs),
                0,
                "tokens={tokens} splits={splits} bs={bs} left {} windows over",
                c.windows % (splits * bs),
            );
        }
    }

    #[test]
    fn snapping_lands_on_the_nearest_corpus() {
        // 20M tokens at seq 256 wants 78,124 windows; the multiple is 80,
        // and 78,124 / 80 = 976.55, so the nearest is 977 * 80 = 78,160.
        let c = resolve_train_corpus(&cfg(Some(20_000_000), 20, 4));
        assert_eq!(c.windows, 78_160);
        assert_eq!(c.staged_tokens, 78_160 * 256 + 1);
        assert!(c.was_snapped());
    }

    #[test]
    fn a_corpus_that_already_divides_is_left_alone() {
        let exact = resolve_train_corpus(&cfg(Some(20_000_000), 20, 4));
        let again = resolve_train_corpus(&cfg(Some(exact.staged_tokens), 20, 4));
        assert_eq!(again.windows, exact.windows);
        assert!(!again.was_snapped(), "a snapped corpus must be a fixed point");
    }

    #[test]
    fn a_tiny_request_still_yields_one_whole_event() {
        // Never round down to an empty corpus.
        let c = resolve_train_corpus(&cfg(Some(1), 20, 4));
        assert_eq!(c.windows, 80);
        assert!(c.windows >= 20 * 4);
    }

    #[test]
    fn the_default_corpus_is_the_shipped_slice_snapped() {
        let c = resolve_train_corpus(&cfg(None, 1, 4));
        assert_eq!(c.requested_tokens, OLMO_TRAIN_BYTES / 2);
        assert_eq!(c.windows % 4, 0);
        // 4 MiB is 2,097,152 tokens = 8191 windows + 1 spare token; the
        // snap to a multiple of 4 lands on 8192.
        assert_eq!(c.windows, 8192);
    }
}
