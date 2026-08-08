//! Roundtrip: HF → flodl load → flodl save → bytes-equality vs HF.
//!
//! Inverse of `bert_parity.rs`. Parity validates we read HF correctly;
//! roundtrip validates we write back HF-compatibly. Same `_live`
//! mechanics — pulls real `google-bert/bert-base-uncased` weights from the Hub
//! (cached after the first run via `hf-hub`).
//!
//! See `roundtrip_common::run_roundtrip` for the full assertion shape.

mod roundtrip_common;

use flodl_hf::models::bert::BertModel;

#[test]
#[ignore = "live: roundtrip via safetensors save; requires network + hf-hub cache"]
fn bert_roundtrip_vs_pytorch_live() {
    let graph = BertModel::from_pretrained("google-bert/bert-base-uncased").unwrap();
    roundtrip_common::run_roundtrip(&graph, "google-bert/bert-base-uncased", "bert");
}
