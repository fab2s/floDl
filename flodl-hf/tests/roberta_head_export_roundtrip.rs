//! Export roundtrips for the four RoBERTa task heads.
//!
//! Sibling of `roberta_export_roundtrip.rs`. See
//! `bert_head_export_roundtrip.rs` for the structural rationale.

mod roundtrip_common;

use flodl_hf::models::roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification,
};

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn roberta_seqcls_export_roundtrip_live() {
    let repo = "cardiffnlp/twitter-roberta-base-sentiment-latest";
    let head = RobertaForSequenceClassification::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = RobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "roberta_seqcls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn roberta_tokencls_export_roundtrip_live() {
    let repo = "Jean-Baptiste/roberta-large-ner-english";
    let head = RobertaForTokenClassification::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = RobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "roberta_tokencls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn roberta_qa_export_roundtrip_live() {
    let repo = "deepset/roberta-base-squad2";
    let head = RobertaForQuestionAnswering::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = RobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(head.graph(), &config.to_json_str(), repo, "roberta_qa");
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn roberta_mlm_export_roundtrip_live() {
    let repo = "roberta-base";
    let head = RobertaForMaskedLM::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = RobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "roberta_mlm",
    );
}
