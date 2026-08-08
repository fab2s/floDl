//! Export roundtrips for the four DeBERTa-v2 task heads.
//!
//! Sibling of `deberta_v2_export_roundtrip.rs`. See
//! `bert_head_export_roundtrip.rs` for the structural rationale.
//!
//! All four picks are DeBERTa-v3 checkpoints (deberta-v3 keeps
//! `model_type: "deberta-v2"` in config.json — the v3 paper is a
//! pre-training-recipe change, not an architecture change).
//!
//! `microsoft/deberta-v3-base` ships `pytorch_model.bin` only; the
//! comparator's `hf_safetensors_path` will trigger
//! `fdl flodl-hf convert microsoft/deberta-v3-base` on first run.

mod roundtrip_common;

use flodl_hf::models::deberta_v2::{
    DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification,
};

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn deberta_v2_seqcls_export_roundtrip_live() {
    let repo = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli";
    let Some(head) = roundtrip_common::or_skip(
        DebertaV2ForSequenceClassification::from_pretrained(repo),
        repo,
    ) else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DebertaV2Config::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "deberta_v2_seqcls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn deberta_v2_tokencls_export_roundtrip_live() {
    let repo = "blaze999/Medical-NER";
    let Some(head) =
        roundtrip_common::or_skip(DebertaV2ForTokenClassification::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DebertaV2Config::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "deberta_v2_tokencls",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn deberta_v2_qa_export_roundtrip_live() {
    let repo = "deepset/deberta-v3-base-squad2";
    let Some(head) =
        roundtrip_common::or_skip(DebertaV2ForQuestionAnswering::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DebertaV2Config::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "deberta_v2_qa",
    );
}

#[test]
#[ignore = "live: HF head export roundtrip; requires network + hf-hub cache"]
fn deberta_v2_mlm_export_roundtrip_live() {
    // `microsoft/deberta-v3-base` was pre-trained with replacement-
    // token-detection (ELECTRA-style) but ships fill-mask-compatible
    // weights; `DebertaV2ForMaskedLM` ignores RTD-only keys and loads
    // the rest. The repo is `.bin`-only, so the comparator triggers
    // `fdl flodl-hf convert` on first run.
    let repo = "microsoft/deberta-v3-base";
    let Some(head) = roundtrip_common::or_skip(DebertaV2ForMaskedLM::from_pretrained(repo), repo)
    else {
        return;
    };
    let config_json = roundtrip_common::fetch_hf_config_json(repo);
    let config = DebertaV2Config::from_json_str(&config_json).unwrap();
    roundtrip_common::run_export_roundtrip(
        head.graph(),
        &config.to_json_str(),
        repo,
        "deberta_v2_mlm",
    );
}
