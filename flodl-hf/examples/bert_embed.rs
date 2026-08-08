//! Minimal BERT-base-uncased embedding example.
//!
//! Closed-loop demo of the flodl-hf API: download a tokenizer and model
//! from the HuggingFace Hub, encode a batch of sentences, and print the
//! pooled sentence representation for each.
//!
//! Run with:
//!
//! ```text
//! fdl flodl-hf example bert-embed
//! # or, inside the dev container directly:
//! cargo run --release --example bert_embed
//! ```
//!
//! First run downloads ~440 MB of weights plus a ~460 KB `tokenizer.json`
//! into `hf_hub`'s cache (`~/.cache/huggingface/` by default).
//! Subsequent runs read from disk.

use flodl::nn::Module;
use flodl::{DType, Variable};
use flodl_hf::models::bert::{BertModel, build_extended_attention_mask};
use flodl_hf::tokenizer::HfTokenizer;

fn main() -> flodl::Result<()> {
    let tok = HfTokenizer::from_pretrained("bert-base-uncased")?;
    let graph = BertModel::from_pretrained("bert-base-uncased")?;
    graph.eval();

    let texts = &["hello world", "flodl brings libtorch to Rust"];
    let enc = tok.encode(texts)?;

    // Raw 0/1 mask -> additive f32 form BERT expects.
    let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
    let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);

    let pooled =
        graph.forward_multi(&[enc.input_ids, enc.position_ids, enc.token_type_ids, mask])?;

    let shape = pooled.shape();
    let hidden = shape[1] as usize;
    let flat = pooled.data().to_f32_vec()?;

    for (i, text) in texts.iter().enumerate() {
        let emb = &flat[i * hidden..(i + 1) * hidden];
        let l2 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "{text:?} | dim={hidden} L2={l2:.3} head=[{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            emb[0], emb[1], emb[2], emb[3], emb[4],
        );
    }
    Ok(())
}
