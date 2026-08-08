//! Export roundtrips for the four XLM-RoBERTa task heads.
//!
//! Sibling of `xlm_roberta_export_roundtrip.rs`. See
//! `bert_head_export_roundtrip.rs` for the structural rationale.

mod roundtrip_common;

use flodl_hf::models::xlm_roberta::{
    XlmRobertaConfig, XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering,
    XlmRobertaForSequenceClassification, XlmRobertaForTokenClassification,
};

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn xlm_roberta_seqcls_export_roundtrip_live() {
    let repo = "cardiffnlp/twitter-xlm-roberta-base-sentiment";
    let Some(head) = roundtrip_common::or_skip(
        XlmRobertaForSequenceClassification::from_pretrained(repo),
        repo,
    ) else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = XlmRobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "xlm_roberta_seqcls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn xlm_roberta_tokencls_export_roundtrip_live() {
    let repo = "Davlan/xlm-roberta-large-ner-hrl";
    let Some(head) = roundtrip_common::or_skip(
        XlmRobertaForTokenClassification::from_pretrained(repo),
        repo,
    ) else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = XlmRobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "xlm_roberta_tokencls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn xlm_roberta_qa_export_roundtrip_live() {
    let repo = "deepset/xlm-roberta-large-squad2";
    let Some(head) =
        roundtrip_common::or_skip(XlmRobertaForQuestionAnswering::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = XlmRobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "xlm_roberta_qa",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn xlm_roberta_mlm_export_roundtrip_live() {
    let repo = "FacebookAI/xlm-roberta-base";
    let Some(head) = roundtrip_common::or_skip(XlmRobertaForMaskedLM::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = XlmRobertaConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "xlm_roberta_mlm",
    );
}
