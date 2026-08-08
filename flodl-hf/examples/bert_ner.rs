//! Named-entity recognition with `BertForTokenClassification`.
//!
//! Closed-loop demo: download a CoNLL-2003 NER checkpoint, tag a
//! sentence, and print each subword's predicted entity label.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example bert-ner
//! # or directly:
//! cargo run --release --example bert_ner
//! ```

use flodl_hf::models::bert::BertForTokenClassification;

fn main() -> flodl::Result<()> {
    let ner = BertForTokenClassification::from_pretrained("dslim/bert-base-NER")?;

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
            } // drop padding
            if t.label == "O" {
                continue;
            } // drop non-entity tokens for readability
            println!("  {:<15} {:<8} ({:.3})", t.token, t.label, t.score);
        }
    }
    Ok(())
}
