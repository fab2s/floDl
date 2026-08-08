//! Named-entity recognition with `DistilBertForTokenClassification`.
//!
//! Closed-loop demo: download `dslim/distilbert-NER` (a distilled
//! version of the widely-used `dslim/bert-base-NER` English NER
//! checkpoint), tag a sentence, and print each subword's predicted
//! entity label.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example distilbert-ner
//! # or directly:
//! cargo run --release --example distilbert_ner
//! ```

use flodl_hf::models::distilbert::DistilBertForTokenClassification;

fn main() -> flodl::Result<()> {
    let ner = DistilBertForTokenClassification::from_pretrained("dslim/distilbert-NER")?;

    let sentences = &[
        "fab2s writes Rust code in Latent",
        "Anthropic built Claude in San Francisco",
    ];
    let tagged = ner.predict(sentences)?;

    for (sentence, tokens) in sentences.iter().zip(&tagged) {
        println!("{sentence:?}");
        for t in tokens {
            if !t.attends {
                continue;
            }
            if t.label == "O" {
                continue;
            }
            println!("  {:<15} {:<8} ({:.3})", t.token, t.label, t.score);
        }
    }
    Ok(())
}
