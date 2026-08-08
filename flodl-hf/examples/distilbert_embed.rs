//! Minimal distilbert/distilbert-base-uncased embedding example.
//!
//! Closed-loop demo: download a tokenizer and model from the
//! HuggingFace Hub, encode a batch of sentences, and print the
//! first-token (`[CLS]`) hidden representation for each — DistilBERT
//! has no pooler, so the CLS row of `last_hidden_state` is the closest
//! analogue to BERT's pooled output.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example distilbert-embed
//! # or directly:
//! cargo run --release --example distilbert_embed
//! ```
//!
//! First run downloads ~270 MB of weights plus the tokenizer into
//! `hf_hub`'s cache (`~/.cache/huggingface/` by default).

use flodl::nn::Module;
use flodl::{DType, Variable};
use flodl_hf::models::bert::build_extended_attention_mask;
use flodl_hf::models::distilbert::DistilBertModel;
use flodl_hf::tokenizer::HfTokenizer;

fn main() -> flodl::Result<()> {
    let repo = "distilbert/distilbert-base-uncased";
    let tok = HfTokenizer::from_pretrained(repo)?;
    let graph = DistilBertModel::from_pretrained(repo)?;
    graph.eval();

    let texts = &["hello world", "flodl brings libtorch to Rust"];
    let enc = tok.encode(texts)?;

    let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
    let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);

    // DistilBERT graph takes 2 inputs: input_ids + attention_mask.
    // Position ids are sequential and computed internally.
    let hidden = graph.forward_multi(&[enc.input_ids, mask])?;

    // [B, S, dim] — take the [CLS] row (index 0 along seq axis).
    let shape = hidden.shape();
    let seq = shape[1] as usize;
    let dim = shape[2] as usize;
    let flat = hidden.data().to_f32_vec()?;

    for (i, text) in texts.iter().enumerate() {
        let base = i * seq * dim; // start of batch entry
        let cls = &flat[base..base + dim]; // seq index 0
        let l2 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "{text:?} | dim={dim} L2={l2:.3} head=[{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            cls[0], cls[1], cls[2], cls[3], cls[4],
        );
    }
    Ok(())
}
