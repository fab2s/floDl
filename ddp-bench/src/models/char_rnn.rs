//! Karpathy's Char-RNN on Shakespeare.
//!
//! Ref: Karpathy 2015 "The Unreasonable Effectiveness of Recurrent Neural Networks".
//! Expected: CE loss ~1.5 at convergence.

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, DType, Result, Tensor};
use flodl::*;

use super::{DatasetConfig, ModelDef};
use crate::config::ModelDefaults;
use flodl::nn::RMSprop;

const VOCAB_SIZE: i64 = 65; // Shakespeare character set
const EMBED_DIM: i64 = 128;
const HIDDEN_DIM: i64 = 256;
const NUM_LAYERS: usize = 2;
const SEQ_LEN: usize = 128;

pub fn def() -> ModelDef {
    ModelDef {
        name: "char-rnn",
        description: "Char-RNN on Shakespeare (Karpathy 2015, loss ~1.5)",
        build: build_model,
        dataset: make_dataset,
        train_fn: train_step,
        eval_fn: Some(eval_loss),
        test_dataset: Some(make_test_dataset),
        augment_fn: None,
        // Karpathy char-rnn: RMSprop lr=2e-3
        optimizer: |p, lr| Box::new(RMSprop::new(p, lr)),
        scheduler: None,
        reference: "Shakespeare CE loss ~1.5, eval=val loss ([Karpathy 2015](https://karpathy.github.io/2015/05/21/rnn-effectiveness/))",
        eval_higher_is_better: false,
        published_eval: Some(1.5),
        needs_baseline_eval: false,
        defaults: ModelDefaults {
            epochs: 50,
            batches_per_epoch: 0, // full dataset
            batch_size: 64,
            lr: 0.002,
        },
    }
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    let model = FlowBuilder::from(Embedding::on_device(VOCAB_SIZE, EMBED_DIM, device)?)
        .through(LSTM::on_device(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, true, device)?)
        .through(Dropout::new(0.2))
        .through(Linear::on_device(HIDDEN_DIM, VOCAB_SIZE, device)?)
        .build()?;
    Ok(Box::new(model))
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
    // batch[0] = input sequences [B, seq_len] Int64
    // batch[1] = target sequences [B, seq_len] Int64 (shifted by 1)
    let input = Variable::new(batch[0].to_dtype(DType::Int64)?, false);
    let target = batch[1].to_dtype(DType::Int64)?;

    let pred = model.forward(&input)?;
    // pred: [B, seq_len, vocab_size] -> flatten for CE
    let shape = pred.shape();
    let flat_pred = pred.reshape(&[-1, shape[shape.len() - 1]])?;
    let flat_target = Variable::new(target.reshape(&[-1])?, false);
    cross_entropy_loss(&flat_pred, &flat_target)
}

/// Held-out CE loss (lower is better).
fn eval_loss(model: &dyn Module, batch: &[Tensor]) -> Result<f64> {
    train_step(model, batch)?.item()
}
