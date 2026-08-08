//! Roundtrip: HF → flodl load → flodl save → bytes-equality vs HF.
//!
//! Inverse of `albert_parity.rs`. Same `_live` mechanics — pulls real
//! `albert/albert-base-v2` weights from the Hub (cached after the first run).
//!
//! See `roundtrip_common::run_roundtrip` for the assertion shape.

mod roundtrip_common;

use flodl_hf::models::albert::AlbertModel;

#[test]
#[ignore = "live: roundtrip via safetensors save; requires network + hf-hub cache"]
fn albert_roundtrip_vs_pytorch_live() {
    let graph = AlbertModel::from_pretrained("albert/albert-base-v2").unwrap();
    roundtrip_common::run_roundtrip(&graph, "albert/albert-base-v2", "albert");
}
