//! Named-entity recognition with `RobertaForTokenClassification`.
//!
//! Closed-loop demo: download a RoBERTa-based English NER checkpoint,
//! tag a sentence, and print each subword's predicted entity label.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example roberta-ner
//! # or directly:
//! cargo run --release --example roberta_ner
//! ```
//!
//! Note: first run downloads ~1.4 GB of weights (`roberta-large-ner-english`
//! is the canonical English NER fine-tune in the RoBERTa family). The
//! BERT-based `dslim/bert-base-NER` is the smaller alternative if disk
//! budget matters more than head-family parity.

use flodl_hf::models::roberta::RobertaForTokenClassification;

fn main() -> flodl::Result<()> {
    let ner =
        RobertaForTokenClassification::from_pretrained("Jean-Baptiste/roberta-large-ner-english")?;

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
