//! Cross-host / auto-promote validation vehicle for a flodl-hf head through
//! the DDP tiers. Mirrors `flodl-hf/examples/distilbert_finetune_ddp.rs`, but
//! as a `--bin` (so it can ride the `fdl @cluster` pre-flight, which builds
//! `--bin`, not `--example`) and with a `--tier managed|cooperative` toggle so
//! both tiers can be validated on the same vehicle.
//!
//! Dataset is the example's inline sentiment pairs repeated so a 3-rank
//! cross-host cohort has real per-rank work (the example's 10 samples are too
//! thin to shard across 3 ranks).
//!
//! ```text
//! # cross-host (3 ranks): built + fanned out by the pre-flight
//! fdl @cluster hf-ddp --tier cooperative --mode nccl-cadence
//! # single-host auto-promote (run the pre-built binary in a 2-GPU box, no cluster env)
//! LD_LIBRARY_PATH=<libtorch>/lib <binary> --tier cooperative --mode nccl-cadence
//! ```

use std::sync::Arc;

use flodl::data::BatchDataSet;
use flodl::distributed::{ApplyPolicy, AverageBackend};
use flodl::{Adam, Device, Result, Tensor, Trainer, Variable};
use flodl_cli::{parse_or_schema, FdlArgs};
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;
use flodl_hf::tokenizer::{EncodedBatch, HfTokenizer};

/// CLI surface (derived → `--fdl-schema` / `--help` come from the binary, so
/// `fdl hf-ddp -h` stays in sync with the code and the cluster help never
/// drifts — same pattern as ddp-bench).
#[derive(FdlArgs, Debug)]
struct Cli {
    /// Execution tier.
    #[option(default = "cooperative", choices = &["managed", "cooperative"])]
    tier: String,

    /// DDP averaging mode.
    #[option(
        default = "nccl-cadence",
        choices = &["nccl-sync", "nccl-cadence", "cpu-sync", "cpu-cadence", "cpu-async"]
    )]
    mode: String,
}

/// Inline sentiment dataset, tokenized once (identical to the example's).
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
        Ok(vec![
            self.input_ids.index_select(0, &idx)?,
            self.attention_mask.index_select(0, &idx)?,
            self.labels.index_select(0, &idx)?,
        ])
    }
}

/// Head forward + loss (identical to the example): reconstruct the
/// `EncodedBatch` and run the head's own `compute_loss`.
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

/// Parse `--mode` into (policy, backend). Defaults to nccl-cadence.
fn parse_mode(mode: &str) -> Option<(ApplyPolicy, AverageBackend)> {
    match mode {
        "nccl-sync" => Some((ApplyPolicy::Sync, AverageBackend::Nccl)),
        "nccl-cadence" => Some((ApplyPolicy::Cadence, AverageBackend::Nccl)),
        "cpu-sync" => Some((ApplyPolicy::Sync, AverageBackend::Cpu)),
        "cpu-cadence" => Some((ApplyPolicy::Cadence, AverageBackend::Cpu)),
        "cpu-async" => Some((ApplyPolicy::Async, AverageBackend::Cpu)),
        _ => None,
    }
}

fn main() -> Result<()> {
    // Derived CLI: intercepts --fdl-schema (emits the schema fdl caches for
    // help/completion) and --help, else parses argv. Because it answers
    // --fdl-schema, the fdl.yml can set `compile: true` — a probe emits the
    // schema and exits, it never launches training.
    let cli: Cli = parse_or_schema();
    let tier = cli.tier;
    let mode = cli.mode;
    let (policy, backend) = parse_mode(&mode)
        .ok_or_else(|| flodl::tensor::TensorError::new(&format!("unknown --mode {mode}")))?;

    let model_repo = "distilbert-base-uncased-finetuned-sst-2-english";
    let tok_repo = "distilbert-base-uncased";
    let tok = HfTokenizer::from_pretrained(tok_repo)?;

    let base: &[(&str, i64)] = &[
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
    // Repeat so a 3-rank cohort has real per-rank work.
    let train: Vec<(&str, i64)> = (0..6).flat_map(|_| base.iter().copied()).collect();
    let n = train.len();
    let dataset: Arc<dyn BatchDataSet> = Arc::new(SentimentDataset::new(&tok, &train)?);

    let repo = model_repo.to_string();
    let builder = Trainer::builder(
        move |dev| DistilBertForSequenceClassification::from_pretrained_on_device(&repo, dev),
        |p| Adam::new(p, 5e-5),
        train_step,
    )
    .dataset(dataset)
    .batch_size(5)
    .num_epochs(3)
    .policy(policy)
    .backend(backend);

    println!("hf-ddp: {model_repo}, {n} samples, tier={tier}, mode={mode}");

    match tier.as_str() {
        "managed" => {
            let handle = builder.run()?;
            while let Some(m) = handle.next_metrics() {
                println!("epoch {}: loss={:.6}", m.epoch, m.avg_loss);
            }
            let state = handle.join()?;
            println!(
                "done: managed HF DDP complete, {} trained parameter tensors",
                state.params.len(),
            );
        }
        "cooperative" => {
            let mut w = builder.into_worker()?;
            let mut batch_no = 0usize;
            while let Some(_plan) = w.next_plan()? {
                while let Some(batch) = w.next_batch()? {
                    let loss = train_step(w.model(), &batch)?;
                    let loss_val = loss.item()?;
                    loss.backward()?;
                    let outcome = w.step(&loss)?;
                    batch_no += 1;
                    println!("batch {batch_no}: loss={loss_val:.4}");
                    if outcome.shutdown {
                        break;
                    }
                }
                for m in w.poll_metrics() {
                    println!("epoch {}: loss={:.6}", m.epoch, m.avg_loss);
                }
            }
            for m in w.poll_metrics() {
                println!("epoch {}: loss={:.6}", m.epoch, m.avg_loss);
            }
            for (ep, metric) in w.poll_eval() {
                println!("eval (epoch {ep}): {metric:.4}");
            }
            let state = w.finish()?;
            println!(
                "done: cooperative HF DDP complete, {} trained parameter tensors",
                state.params.len(),
            );
        }
        other => {
            return Err(flodl::tensor::TensorError::new(&format!(
                "--tier must be managed|cooperative, got {other}"
            )));
        }
    }
    Ok(())
}
