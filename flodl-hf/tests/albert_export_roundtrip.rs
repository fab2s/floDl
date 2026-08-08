//! Export roundtrip for ALBERT. See `bert_export_roundtrip.rs` for
//! shape + intent.

mod roundtrip_common;

use flodl_hf::models::albert::{AlbertConfig, AlbertModel};

#[test]
#[ignore = "live: HF export roundtrip; requires network + hf-hub cache"]
fn albert_export_roundtrip_live() {
    let repo = "albert/albert-base-v2";
    let graph = AlbertModel::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = AlbertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(&graph, &config.to_json_str(), repo, "albert");
}
