//! CUDA smoke tier for the XLM-RoBERTa family.
//!
//! Real [`xlm_roberta_base`](flodl_hf::models::xlm_roberta::XlmRobertaConfig::xlm_roberta_base)
//! preset, fresh random weights, one forward + loss + backward per
//! head. Graph layout delegates to RoBERTa (same 3 `forward_multi`
//! inputs), but the 250k vocab makes the MLM tied decoder
//! substantially larger — worth exercising on GPU.

#![cfg(feature = "gpu")]

mod common;

use flodl_hf::models::xlm_roberta::{
    XlmRobertaConfig, XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering,
    XlmRobertaForSequenceClassification, XlmRobertaForTokenClassification,
};
use flodl_hf::task_heads::{
    masked_lm_loss, question_answering_loss, sequence_classification_loss,
    token_classification_loss,
};

use common::{
    BATCH, CUDA, SEQ, assert_grads_flowed, extended_attention_mask, input_ids, mlm_labels,
    qa_positions, seqcls_labels, tokcls_labels, token_type_ids,
};

const NUM_LABELS: i64 = 3;

/// Graph input order (delegates to `roberta_backbone_flow`):
/// `[input_ids, token_type_ids, attention_mask_additive]`.
fn xlm_roberta_inputs(cfg: &XlmRobertaConfig) -> Vec<flodl::Variable> {
    vec![
        input_ids(BATCH, SEQ, cfg.vocab_size),
        token_type_ids(BATCH, SEQ),
        extended_attention_mask(BATCH, SEQ),
    ]
}

#[test]
fn xlm_roberta_seqcls_cuda_smoke() {
    let cfg = XlmRobertaConfig::xlm_roberta_base();
    let head = XlmRobertaForSequenceClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&xlm_roberta_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, NUM_LABELS]);

    let labels = seqcls_labels(BATCH, NUM_LABELS);
    let loss = sequence_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn xlm_roberta_tokcls_cuda_smoke() {
    let cfg = XlmRobertaConfig::xlm_roberta_base();
    let head = XlmRobertaForTokenClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&xlm_roberta_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, NUM_LABELS]);

    let labels = tokcls_labels(BATCH, SEQ, NUM_LABELS);
    let loss = token_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn xlm_roberta_qa_cuda_smoke() {
    let cfg = XlmRobertaConfig::xlm_roberta_base();
    let head = XlmRobertaForQuestionAnswering::on_device(&cfg, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&xlm_roberta_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, 2]);

    let (starts, ends) = qa_positions(BATCH, SEQ);
    let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn xlm_roberta_mlm_cuda_smoke() {
    let cfg = XlmRobertaConfig::xlm_roberta_base();
    let head = XlmRobertaForMaskedLM::on_device(&cfg, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&xlm_roberta_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, cfg.vocab_size]);

    let labels = mlm_labels(BATCH, SEQ, cfg.vocab_size);
    let loss = masked_lm_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}
