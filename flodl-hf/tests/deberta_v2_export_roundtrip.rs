//! Export roundtrip for DeBERTa-v3. See `bert_export_roundtrip.rs`
//! for shape + intent.
//!
//! Note: `microsoft/deberta-v3-base` ships as `.bin` — flodl-hf's
//! `from_pretrained` uses the `flodl-converted` cache path produced by
//! `fdl flodl-hf convert microsoft/deberta-v3-base`, which must be run
//! once in the shared hf-cache before this test can pass.

mod roundtrip_common;

use flodl_hf::models::deberta_v2::{DebertaV2Config, DebertaV2Model};

#[test]
#[ignore = "live: HF export roundtrip; requires network + hf-hub cache + flodl-converted"]
fn deberta_v2_export_roundtrip_live() {
    let repo = "microsoft/deberta-v3-base";
    let Some(graph) = roundtrip_common::or_skip(DebertaV2Model::from_pretrained(repo), repo) else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DebertaV2Config::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(&graph, &config.to_json_str(), repo, "deberta-v2");
}
