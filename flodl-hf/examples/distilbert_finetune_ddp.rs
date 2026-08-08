//! Fine-tune a DistilBERT sentiment classifier through the **cooperative
//! training tier** — the user owns the loop, the controller stays authoritative
//! over cadence / data partition / gradient averaging / role election.
//!
//! This is the DDP-aware counterpart to `distilbert_finetune.rs` (which keeps a
//! gentle single-device loop driving `head.graph()` directly). The loop here is
//! written once and scales unchanged from one CPU to N GPUs to a multi-host
//! cluster: `Trainer::builder(...).into_worker()` resolves the topology
//! (single-device fallback, single-host multi-GPU auto-promote, or an
//! `fdl @<cluster-env>` fan-out) and the same `next_plan` / `next_batch` /
//! `step` loop runs on each rank.
//!
//! Two things differ from the single-device example, both intrinsic to going
//! through the unified `Trainer::builder` path:
//!
//! 1. **The model factory loads pretrained weights on the target device.** The
//!    builder constructs *every* rank's model from the factory (there is no
//!    "rank 0 uses this pre-built instance" injection), so `from_pretrained`
//!    must live inside the factory. On multi-GPU each rank loads independently
//!    (cheap: the hub download is cached) and the framework's initial broadcast
//!    keeps them identical.
//! 2. **Training feeds from a [`BatchDataSet`].** The inline pairs are
//!    tokenized once into `[input_ids, attention_mask, labels]` tensors;
//!    `get_batch` gathers rows by index. Swap `SentimentDataset` for a real
//!    `DataSet` and the loop is unchanged.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example distilbert-finetune-ddp
//! # or directly (single device):
//! cargo run --release --example distilbert_finetune_ddp
//! # multi-host: drive through a cluster env, e.g.
//! # fdl @cluster flodl-hf example distilbert-finetune-ddp
//! ```

use std::sync::Arc;

use flodl::data::BatchDataSet;
use flodl::{Adam, Device, Result, Tensor, Trainer, Variable};
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;
use flodl_hf::tokenizer::{EncodedBatch, HfTokenizer};

/// Inline sentiment dataset, tokenized once up front. Holds `input_ids`
/// `[N, L]`, `attention_mask` `[N, L]`, and `labels` `[N]` (all on CPU); the
/// data plane moves the per-batch rows to each rank's device. Padding is
/// batch-longest across the whole set, so every row shares one length `L` and
/// `get_batch` is a plain row gather.
struct SentimentDataset {
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    n: usize,
}

impl SentimentDataset {
    fn new(tok: &HfTokenizer, pairs: &[(&str, i64)]) -> Result<Self> {
        let texts: Vec<&str> = pairs.iter().map(|(t, _)| *t).collect();
        let enc = tok.encode(&texts)?;
        let label_ids: Vec<i64> = pairs.iter().map(|(_, l)| *l).collect();
        Ok(Self {
            input_ids: enc.input_ids.data(),
            attention_mask: enc.attention_mask.data(),
            labels: Tensor::from_i64(&label_ids, &[pairs.len() as i64], Device::CPU)?,
            n: pairs.len(),
        })
    }
}

impl BatchDataSet for SentimentDataset {
    fn len(&self) -> usize {
        self.n
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let idx: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        let idx = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;
        // Order fixed by contract with `train_step`: ids, mask, labels.
        Ok(vec![
            self.input_ids.index_select(0, &idx)?,
            self.attention_mask.index_select(0, &idx)?,
            self.labels.index_select(0, &idx)?,
        ])
    }
}

/// The user's forward + loss for the cooperative loop. Reconstructs the head's
/// [`EncodedBatch`] from the batch tensors and runs the head's own
/// `compute_loss` — which applies DistilBERT's extended-attention-mask step and
/// the sequence-classification cross-entropy, so this stays faithful to the
/// head's real forward rather than re-deriving it.
///
/// DistilBERT's `encoder_inputs` consumes only `input_ids` + `attention_mask`;
/// the other `EncodedBatch` fields are unused placeholders here. A
/// segment-sensitive backbone (e.g. BERT with `token_type_ids`) would fill them
/// from the batch instead.
fn train_step(model: &DistilBertForSequenceClassification, batch: &[Tensor]) -> Result<Variable> {
    let placeholder = Variable::new(Tensor::zeros_like(&batch[0])?, false);
    let enc = EncodedBatch {
        input_ids: Variable::new(batch[0].clone(), false),
        attention_mask: Variable::new(batch[1].clone(), false),
        token_type_ids: placeholder.clone(),
        position_ids: placeholder.clone(),
        sequence_ids: placeholder,
    };
    let labels = Variable::new(batch[2].clone(), false);
    model.compute_loss(&enc, &labels)
}

fn main() -> Result<()> {
    let model_repo = "distilbert/distilbert-base-uncased-finetuned-sst-2-english";
    // The SST-2 checkpoint ships only the legacy tokenizer triple; grab the
    // fast tokenizer from the base repo (identical vocabulary).
    let tok_repo = "distilbert/distilbert-base-uncased";
    let tok = HfTokenizer::from_pretrained(tok_repo)?;

    let train: &[(&str, i64)] = &[
        ("This framework is a real joy to work with", 1),
        ("I absolutely love the clean API surface", 1),
        ("Releases land on schedule and the diff is readable", 1),
        ("The documentation is thorough and honest", 1),
        ("Fine-tuning just worked on the first try", 1),
        ("The tokenizer is painfully slow", 0),
        ("I wasted an afternoon chasing a silent shape bug", 0),
        ("The error messages are useless", 0),
        ("I cannot figure out which feature flag I need", 0),
        ("Performance fell off a cliff after the update", 0),
    ];
    let n = train.len();
    let dataset: Arc<dyn BatchDataSet> = Arc::new(SentimentDataset::new(&tok, train)?);

    // Cooperative tier. `into_worker` resolves the topology and hands back a
    // `Worker` the loop below drives; the controller (on multi-GPU / cluster)
    // owns cadence, partition, and gradient averaging. `train_step` is passed
    // to the builder (API symmetry with the managed `.run()`) and reused as the
    // user's forward in the loop.
    let repo = model_repo.to_string();
    let mut w = Trainer::builder(
        move |dev| DistilBertForSequenceClassification::from_pretrained_on_device(&repo, dev),
        |p| Adam::new(p, 5e-5),
        train_step,
    )
    .dataset(dataset)
    .batch_size(5)
    .num_epochs(3)
    .into_worker()?;

    println!("fine-tuning {model_repo} on {n} examples via the cooperative tier");
    println!("{:>5} {:>8}", "batch", "loss");

    // `next_plan` yields the controller's next unit of work (a chunk under
    // progressive cadence, a whole epoch under Sync / single-device); the inner
    // loop drains its batches. The reduce, when it happens, rides `step`'s
    // control drain — the loop never names it.
    let mut batch_no = 0usize;
    while let Some(_plan) = w.next_plan()? {
        while let Some(batch) = w.next_batch()? {
            let loss = train_step(w.model(), &batch)?;
            let loss_val = loss.item()?;
            loss.backward()?;
            let outcome = w.step(&loss)?;
            batch_no += 1;
            println!("{batch_no:>5} {loss_val:>8.4}");
            if outcome.shutdown {
                break;
            }
        }
    }

    // Intent channel: ask the controller to eval / checkpoint at its next
    // coherent occasion. A request, not a command — on multi-GPU the controller
    // folds these into its role-elected dispatch on the rank it elects; on a
    // single device (no controller) they are no-ops, shown here for the call
    // site. Attach `.eval_dataset(...)` / `.checkpoint_fn(...)` on the builder
    // to give the controller something to run.
    w.request_eval();
    w.request_checkpoint();

    let state = w.finish()?;
    println!(
        "\ndone: cooperative fine-tune complete, {} trained parameter tensors (CPU)",
        state.params.len(),
    );
    Ok(())
}
