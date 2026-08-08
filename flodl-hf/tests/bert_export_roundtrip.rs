//! Export roundtrip: HF → flodl load → `export_hf_dir` → verify
//! `model.safetensors` bit-exact vs HF + `config.json` present and
//! emits a `model_type` dispatch key.
//!
//! Sibling of `bert_roundtrip.rs` (save API) and `bert_parity.rs`
//! (forward parity). Exercises the full user-facing export path end-to-end
//! against real HF weights, catching wiring regressions that the
//! in-process unit tests don't cover (fetch from Hub, legacy key
//! rename, dtype widening) while staying Rust-only — HF Python
//! `AutoModel` loadability is a separate manual gate.

mod roundtrip_common;

use flodl_hf::models::bert::{BertConfig, BertModel};

#[test]
#[ignore = "live: HF export roundtrip; requires network + hf-hub cache"]
fn bert_export_roundtrip_live() {
    let repo = "google-bert/bert-base-uncased";
    let graph = BertModel::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = BertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(&graph, &config.to_json_str(), repo, "bert");
}
