//! Export roundtrip for DistilBERT. See `bert_export_roundtrip.rs`
//! for shape + intent.

mod roundtrip_common;

use flodl_hf::models::distilbert::{DistilBertConfig, DistilBertModel};

#[test]
#[ignore = "live: HF export roundtrip; requires network + hf-hub cache"]
fn distilbert_export_roundtrip_live() {
    let repo = "distilbert/distilbert-base-uncased";
    let graph = DistilBertModel::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DistilBertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(&graph, &config.to_json_str(), repo, "distilbert");
}
