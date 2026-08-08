//! Export roundtrips for the four BERT task heads.
//!
//! Sibling of `bert_export_roundtrip.rs` (base backbone). Each test
//! pulls a fine-tuned head checkpoint from the Hub via
//! `BertFor*::from_pretrained`, exports via `export_hf_dir`, and
//! verifies bit-exact bytes against the HF reference using the same
//! comparator as the base test (`run_export_roundtrip`).
//!
//! Repo ids mirror the existing `bert_*_parity.rs` choices so the
//! same hf-hub cache entries are reused; `.bin`-only repos
//! (e.g. `nateraw/bert-base-uncased-emotion`) need
//! `fdl flodl-hf convert <repo>` first — `hf_safetensors_path`
//! preferentially reads the converted file as the reference.

mod roundtrip_common;

use flodl_hf::models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification,
};

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn bert_seqcls_export_roundtrip_live() {
    let repo = "nateraw/bert-base-uncased-emotion";
    let head = BertForSequenceClassification::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = BertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "bert_seqcls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn bert_tokencls_export_roundtrip_live() {
    let repo = "dslim/bert-base-NER";
    let head = BertForTokenClassification::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = BertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "bert_tokencls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn bert_qa_export_roundtrip_live() {
    let repo = "csarron/bert-base-uncased-squad-v1";
    let head = BertForQuestionAnswering::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = BertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(head.graph(), &config.to_json_str(), repo, "bert_qa");
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn bert_mlm_export_roundtrip_live() {
    // `bert-base-uncased` ships as `BertForPreTraining` (MLM + NSP);
    // BertForMaskedLM ignores the NSP keys and loads the rest.
    let repo = "bert-base-uncased";
    let head = BertForMaskedLM::from_pretrained(repo).unwrap();
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = BertConfig::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(head.graph(), &config.to_json_str(), repo, "bert_mlm");
}
