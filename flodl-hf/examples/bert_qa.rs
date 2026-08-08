//! Extractive question answering with `BertForQuestionAnswering`.
//!
//! Closed-loop demo: download a SQuAD-fine-tuned checkpoint, answer a
//! few questions against short contexts, and print the extracted spans.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example bert-qa
//! # or directly:
//! cargo run --release --example bert_qa
//! ```

use flodl_hf::models::bert::BertForQuestionAnswering;

fn main() -> flodl::Result<()> {
    let qa = BertForQuestionAnswering::from_pretrained("csarron/bert-base-uncased-squad-v1")?;

    let pairs = &[
        (
            "Where does fab2s live?",
            "fab2s lives in Latent and writes Rust deep learning code.",
        ),
        (
            "What does flodl wrap?",
            "flodl is a Rust deep learning framework built on top of libtorch via an FFI shim.",
        ),
    ];
    let answers = qa.answer_batch(pairs)?;

    for ((q, c), a) in pairs.iter().zip(&answers) {
        println!("Q: {q}");
        println!("C: {c}");
        println!(
            "A: {:?}  (tokens [{}..={}], score={:.3})",
            a.text, a.start, a.end, a.score
        );
        println!();
    }
    Ok(())
}
