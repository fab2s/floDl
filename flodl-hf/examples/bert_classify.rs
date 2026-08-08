//! Sequence classification with `BertForSequenceClassification`.
//!
//! Closed-loop demo: download a fine-tuned sentiment / emotion
//! classifier from the HuggingFace Hub, run it on a few prompts, and
//! print per-label probabilities.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example bert-classify
//! # or, inside the dev container directly:
//! cargo run --release --example bert_classify
//! ```
//!
//! First run downloads ~440 MB of weights plus a ~460 KB tokenizer
//! into `hf_hub`'s cache (`~/.cache/huggingface/` by default).

use flodl_hf::models::bert::BertForSequenceClassification;

fn main() -> flodl::Result<()> {
    // Six emotion classes: sadness, joy, love, anger, fear, surprise.
    // The checkpoint ships with `id2label` so the returned strings are
    // human-readable without any extra wiring on our side.
    //
    // This repo ships only `pytorch_model.bin` — run
    // `fdl flodl-hf convert nateraw/bert-base-uncased-emotion`
    // once before the first call.
    let clf = BertForSequenceClassification::from_pretrained("nateraw/bert-base-uncased-emotion")?;

    let texts = &[
        "I love this framework so much",
        "I can't believe they shut down the old service",
        "I'm a little anxious about the release",
    ];
    let preds = clf.predict(texts)?;

    for (text, row) in texts.iter().zip(&preds) {
        let (top_label, top_score) = row.first().expect("predict returns at least one label");
        println!("{text:?}");
        println!("  top: {top_label} ({top_score:.3})");
        for (label, score) in row.iter().skip(1).take(2) {
            println!("  ..  {label} ({score:.3})");
        }
    }
    Ok(())
}
