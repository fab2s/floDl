//! Roundtrip: HF → flodl load → flodl save → bytes-equality vs HF.
//!
//! Inverse of `xlm_roberta_parity.rs`. Same `_live` mechanics — pulls
//! real `FacebookAI/xlm-roberta-base` weights from the Hub (~1.1GB cache write on
//! first run).
//!
//! See `roundtrip_common::run_roundtrip` for the assertion shape.

mod roundtrip_common;

use flodl_hf::models::xlm_roberta::XlmRobertaModel;

#[test]
#[ignore = "live: roundtrip via safetensors save; requires network + hf-hub cache"]
fn xlm_roberta_roundtrip_vs_pytorch_live() {
    let graph = XlmRobertaModel::from_pretrained("FacebookAI/xlm-roberta-base").unwrap();
    roundtrip_common::run_roundtrip(&graph, "FacebookAI/xlm-roberta-base", "xlm-roberta");
}
