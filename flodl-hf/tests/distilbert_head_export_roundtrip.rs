//! Export roundtrips for the four DistilBERT task heads.
//!
//! Sibling of `distilbert_roundtrip.rs`. See
//! `bert_head_export_roundtrip.rs` for the structural rationale.

mod roundtrip_common;

use flodl_hf::models::distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification,
};

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn distilbert_seqcls_export_roundtrip_live() {
    let repo = "lxyuan/distilbert-base-multilingual-cased-sentiments-student";
    let head = DistilBertForSequenceClassification::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DistilBertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "distilbert_seqcls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn distilbert_tokencls_export_roundtrip_live() {
    let repo = "dslim/distilbert-NER";
    let head = DistilBertForTokenClassification::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DistilBertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "distilbert_tokencls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn distilbert_qa_export_roundtrip_live() {
    let repo = "distilbert/distilbert-base-cased-distilled-squad";
    let head = DistilBertForQuestionAnswering::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DistilBertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "distilbert_qa",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn distilbert_mlm_export_roundtrip_live() {
    let repo = "distilbert/distilbert-base-uncased";
    let head = DistilBertForMaskedLM::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DistilBertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "distilbert_mlm",
    );
}
