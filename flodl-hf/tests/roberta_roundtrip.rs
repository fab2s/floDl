//! Roundtrip: HF → flodl load → flodl save → bytes-equality vs HF.
//!
//! Inverse of `roberta_parity.rs`. Same `_live` mechanics — pulls real
//! `FacebookAI/roberta-base` weights from the Hub (cached after the first run).
//!
//! See `roundtrip_common::run_roundtrip` for the assertion shape.

mod roundtrip_common;

use flodl_hf::models::roberta::RobertaModel;

#[test]
#[ignore = "live: roundtrip via safetensors save; requires network + hf-hub cache"]
fn roberta_roundtrip_vs_pytorch_live() {
    let graph = RobertaModel::from_pretrained("FacebookAI/roberta-base").unwrap();
    roundtrip_common::run_roundtrip(&graph, "FacebookAI/roberta-base", "roberta");
}
