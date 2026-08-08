//! Roundtrip: HF → flodl load → flodl save → bytes-equality vs HF.
//!
//! Inverse of `deberta_v2_parity.rs`. Same `_live` mechanics — pulls
//! real `microsoft/deberta-v3-base` weights from the Hub (the deberta-v2
//! family fixture pins the v3-base checkpoint; architecture is "v2").
//!
//! See `roundtrip_common::run_roundtrip` for the assertion shape.

mod roundtrip_common;

use flodl_hf::models::deberta_v2::DebertaV2Model;

#[test]
#[ignore = "live: roundtrip via safetensors save; requires network + hf-hub cache"]
fn deberta_v2_roundtrip_vs_pytorch_live() {
    let Some(graph) = roundtrip_common::or_skip(
        DebertaV2Model::from_pretrained("microsoft/deberta-v3-base"),
        "microsoft/deberta-v3-base",
    ) else {
        return;
    };
    roundtrip_common::run_roundtrip(&graph, "microsoft/deberta-v3-base", "deberta-v2");
}
