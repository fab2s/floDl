//! Export roundtrips for the four ALBERT task heads.
//!
//! Sibling of `albert_export_roundtrip.rs`. See
//! `bert_head_export_roundtrip.rs` for the structural rationale.

mod roundtrip_common;

use flodl_hf::models::albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
    AlbertForTokenClassification,
};

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn albert_seqcls_export_roundtrip_live() {
    let repo = "bhadresh-savani/albert-base-v2-emotion";
    let Some(head) =
        roundtrip_common::or_skip(AlbertForSequenceClassification::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = AlbertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "albert_seqcls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn albert_tokencls_export_roundtrip_live() {
    let repo = "ArBert/albert-base-v2-finetuned-ner";
    let Some(head) =
        roundtrip_common::or_skip(AlbertForTokenClassification::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = AlbertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "albert_tokencls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn albert_qa_export_roundtrip_live() {
    let repo = "twmkn9/albert-base-v2-squad2";
    let Some(head) =
        roundtrip_common::or_skip(AlbertForQuestionAnswering::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = AlbertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(head.graph(), &config.to_json_str(), repo, "albert_qa");
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn albert_mlm_export_roundtrip_live() {
    let repo = "albert/albert-base-v2";
    let Some(head) = roundtrip_common::or_skip(AlbertForMaskedLM::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = AlbertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(head.graph(), &config.to_json_str(), repo, "albert_mlm");
}
