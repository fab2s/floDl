//! Fine-tune a DistilBERT sentiment classifier on a tiny inline
//! dataset. Full fine-tuning (backbone + classifier), CPU-only,
//! ~30 seconds after the one-time weight download (~250 MB cached
//! via `hf-hub`).
//!
//! Starting point: `distilbert-base-uncased-finetuned-sst-2-english`,
//! a two-class positive / negative sentiment head. The example
//! continues training on a hand-crafted domain-specific dataset -
//! the canonical "calibrate a generic sentiment model to your own
//! corpus" pattern. Same training loop scales to a real dataset by
//! swapping the inline pairs for a proper [`DataSet`](flodl::DataSet)
//! implementation.
//!
//! After the fine-tune the trained head is saved as a flodl
//! `.fdl` checkpoint plus auto-emitted `<stem>.config.json` sidecar
//! and a `tokenizer.json` next to it (so the downstream export step
//! picks the tokenizer up via its auto-copy whitelist) under
//! `target/distilbert_finetune/`. The example then prints the two
//! host-side `fdl` commands to re-export it as a
//! HuggingFace-compatible directory and verify the export with HF
//! Python (`AutoModelFor*` loadability).
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example distilbert-finetune
//! # or directly:
//! cargo run --release --example distilbert_finetune
//! ```

use flodl::{clip_grad_norm, Adam, Device, Module, Result, Tensor, TensorError, Trainer, Variable};
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;
use flodl_hf::tokenizer::HfTokenizer;

fn main() -> Result<()> {
    let model_repo = "distilbert-base-uncased-finetuned-sst-2-english";
    // The SST-2 checkpoint only ships the legacy vocab.txt / tokenizer_config
    // triple, not a fast `tokenizer.json`. Grab the tokenizer from the base
    // repo - the vocabulary is identical since SST-2 fine-tuning does not
    // retrain it.
    let tok_repo = "distilbert-base-uncased";

    // Load pre-trained model. The head has 2 classes:
    // id 0 = NEGATIVE, id 1 = POSITIVE (matches HF convention).
    let head = DistilBertForSequenceClassification::from_pretrained(model_repo)?;
    let tok = HfTokenizer::from_pretrained(tok_repo)?;

    // Wire the optimizer and training mode through `Trainer::setup_head`.
    // On CPU or a single GPU this is a thin wrapper; on multi-GPU hosts
    // the same call auto-distributes, so the loop below is identical for
    // 1 or N devices. The factory closure is only invoked for additional
    // replica devices.
    //
    // NOTE: the setup tier is deprecated pending its cooperative-tier
    // replacement on the controller engine (the user-owned loop below is
    // exactly the shape that tier keeps); it still works until then. See
    // docs/design/trainer-execution-tiers.md.
    let replica_config = head.config().clone();
    let num_labels = head.labels().len() as i64;
    #[allow(deprecated)]
    Trainer::setup_head(
        &head,
        move |dev| DistilBertForSequenceClassification::on_device(
            &replica_config, num_labels, dev,
        ),
        |p| Adam::new(p, 5e-5),
    )?;

    // Tiny domain-specific dataset. Swap this for a real one by
    // feeding a `DataSet` through `DataLoader::new` - the training
    // body below is unchanged.
    let train: &[(&str, i64)] = &[
        ("This framework is a real joy to work with",          1),
        ("I absolutely love the clean API surface",            1),
        ("Releases land on schedule and the diff is readable", 1),
        ("The documentation is thorough and honest",           1),
        ("Fine-tuning just worked on the first try",           1),
        ("The tokenizer is painfully slow",                    0),
        ("I wasted an afternoon chasing a silent shape bug",   0),
        ("The error messages are useless",                     0),
        ("I cannot figure out which feature flag I need",      0),
        ("Performance fell off a cliff after the update",      0),
    ];

    let params = head.graph().parameters();
    println!("fine-tuning {model_repo} on {} examples", train.len());
    println!("{:>4} {:>8}", "step", "loss");

    for step in 0..5 {
        let texts: Vec<&str> = train.iter().map(|(t, _)| *t).collect();
        let enc = tok.encode(&texts)?;
        let label_ids: Vec<i64> = train.iter().map(|(_, l)| *l).collect();
        let labels = Variable::new(
            Tensor::from_i64(&label_ids, &[train.len() as i64], Device::CPU)?,
            false,
        );

        let loss = head.compute_loss(&enc, &labels)?;
        let loss_val = loss.item()?;
        println!("{:>4} {:>8.4}", step, loss_val);

        loss.backward()?;
        // Standard fine-tuning clip. Keeps early steps stable even
        // when a noisy mini-batch produces a large gradient.
        clip_grad_norm(&params, 1.0)?;
        head.graph().step()?;
    }

    // Close the loop: run one last eval forward (no backward) and
    // peek at the logits for the first example. POSITIVE logit
    // should sit comfortably above NEGATIVE after the fine-tune.
    head.graph().eval();
    let enc = tok.encode(&[train[0].0])?;
    let logits = head.forward_encoded(&enc)?;
    let flat = logits.data().to_f32_vec()?;
    println!(
        "\nafter fine-tuning, {:?} -> NEGATIVE={:.3}, POSITIVE={:.3}",
        train[0].0, flat[0], flat[1],
    );

    // Persist the fine-tuned head for the export → verify recipe
    // below. Path is project-relative under `target/` so it is
    // visible on the host through the dev-container bind mount.
    // `from_pretrained` already stamped the graph's source_config
    // with `architectures: ["DistilBertForSequenceClassification"]`,
    // so `save_checkpoint` automatically drops the matching
    // `<stem>.config.json` sidecar; no hand-rolled config write
    // needed for downstream `--checkpoint` re-export.
    let scratch_dir = "target/distilbert_finetune";
    std::fs::create_dir_all(scratch_dir).map_err(|e| {
        TensorError::new(&format!("create {scratch_dir}: {e}"))
    })?;
    let ckpt_path = format!("{scratch_dir}/sst2_finetuned.fdl");
    let tokenizer_path = format!("{scratch_dir}/tokenizer.json");
    let export_dir = format!("{scratch_dir}/sst2_export");
    head.graph().save_checkpoint(&ckpt_path)?;
    // Persist the tokenizer next to the checkpoint so
    // `fdl flodl-hf export --checkpoint` picks it up via the
    // auto-tokenizer-copy whitelist. Without this, the export step
    // would warn "no tokenizer files matched" and downstream
    // `verify-export` (with hub source) would fail forward parity
    // for lack of an `AutoTokenizer`.
    tok.save(&tokenizer_path)?;
    println!(
        "\ncheckpoint    -> {ckpt_path}\nsidecar config -> {scratch_dir}/sst2_finetuned.config.json\ntokenizer     -> {tokenizer_path}",
    );

    // Recipe printed instead of self-orchestrated: the example runs
    // inside the `dev` container while `verify-export` needs the
    // `hf-parity` container, so spawning them from in-process would
    // require docker-in-docker. Host-side `fdl` brings up each
    // container itself, same pattern as `fdl flodl-hf verify-matrix`.
    println!(
        "\nNext steps (run from host):\n  \
         fdl flodl-hf export --checkpoint {ckpt_path} --out {export_dir}\n  \
         fdl flodl-hf verify-export {export_dir} --no-hub-source",
    );

    Ok(())
}
