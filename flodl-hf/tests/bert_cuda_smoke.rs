//! CUDA smoke tier for the BERT family.
//!
//! Real [`bert_base_uncased`](flodl_hf::models::bert::BertConfig::bert_base_uncased)
//! preset, fresh random weights, one forward + loss + backward per head,
//! assert output shape and that gradients flowed. Catches CUDA-specific
//! regressions that CPU unit tests can miss.
//!
//! All tests are gated `#[cfg(feature = "gpu")]` — compiled-out under
//! `fdl test` (GitHub CI CPU leg), compiled and run under `fdl cuda-test`.

#![cfg(feature = "gpu")]

mod common;

use flodl_hf::models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification,
};
use flodl_hf::task_heads::{
    masked_lm_loss, question_answering_loss, sequence_classification_loss,
    token_classification_loss,
};

use common::{
    assert_grads_flowed, extended_attention_mask, input_ids, mlm_labels, position_ids,
    qa_positions, seqcls_labels, tokcls_labels, token_type_ids, BATCH, CUDA, SEQ,
};

const NUM_LABELS: i64 = 3;

/// Graph input order (per `bert_backbone_flow`):
/// `[input_ids, position_ids, token_type_ids, attention_mask_additive]`.
fn bert_inputs(cfg: &BertConfig) -> Vec<flodl::Variable> {
    vec![
        input_ids(BATCH, SEQ, cfg.vocab_size),
        position_ids(BATCH, SEQ),
        token_type_ids(BATCH, SEQ),
        extended_attention_mask(BATCH, SEQ),
    ]
}

#[test]
fn bert_seqcls_cuda_smoke() {
    let cfg = BertConfig::bert_base_uncased();
    let head = BertForSequenceClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head.graph().forward_multi(&bert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, NUM_LABELS]);

    let labels = seqcls_labels(BATCH, NUM_LABELS);
    let loss = sequence_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn bert_tokcls_cuda_smoke() {
    let cfg = BertConfig::bert_base_uncased();
    let head = BertForTokenClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head.graph().forward_multi(&bert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, NUM_LABELS]);

    let labels = tokcls_labels(BATCH, SEQ, NUM_LABELS);
    let loss = token_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn bert_qa_cuda_smoke() {
    let cfg = BertConfig::bert_base_uncased();
    let head = BertForQuestionAnswering::on_device(&cfg, CUDA).unwrap();

    let logits = head.graph().forward_multi(&bert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, 2]);

    let (starts, ends) = qa_positions(BATCH, SEQ);
    let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn bert_mlm_cuda_smoke() {
    let cfg = BertConfig::bert_base_uncased();
    let head = BertForMaskedLM::on_device(&cfg, CUDA).unwrap();

    let logits = head.graph().forward_multi(&bert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, cfg.vocab_size]);

    let labels = mlm_labels(BATCH, SEQ, cfg.vocab_size);
    let loss = masked_lm_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}
