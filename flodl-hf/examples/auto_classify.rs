//! Family-agnostic sequence classification via `AutoModelForSequenceClassification`.
//!
//! Closed-loop demo: pass a repo id on the command line, let
//! [`AutoConfig`](flodl_hf::models::auto::AutoConfig) figure out the
//! family (BERT / RoBERTa / DistilBERT) from `config.json`, then
//! classify a few prompts. Same code path for every supported
//! family — the dispatch lives inside
//! [`AutoModelForSequenceClassification::from_pretrained`].
//!
//! Run with:
//!
//! ```text
//! # BERT sentiment
//! cargo run --release --example auto_classify -- nlptown/bert-base-multilingual-uncased-sentiment
//!
//! # RoBERTa sentiment
//! cargo run --release --example auto_classify -- cardiffnlp/twitter-roberta-base-sentiment-latest
//!
//! # DistilBERT sentiment (multilingual)
//! cargo run --release --example auto_classify -- lxyuan/distilbert-base-multilingual-cased-sentiments-student
//! ```
//!
//! With no argument, defaults to `cardiffnlp/twitter-roberta-base-sentiment-latest`.

use flodl_hf::models::auto::AutoModelForSequenceClassification;

fn main() -> flodl::Result<()> {
    let repo_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cardiffnlp/twitter-roberta-base-sentiment-latest".to_string());

    eprintln!("loading {repo_id} via AutoModel...");
    let clf = AutoModelForSequenceClassification::from_pretrained(&repo_id)?;

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
