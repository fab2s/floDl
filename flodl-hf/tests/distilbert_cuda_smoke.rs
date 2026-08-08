//! CUDA smoke tier for the DistilBERT family.
//!
//! Real [`distilbert_base_uncased`](flodl_hf::models::distilbert::DistilBertConfig::distilbert_base_uncased)
//! preset, fresh random weights, one forward + loss + backward per head.
//! DistilBERT graphs take only 2 `forward_multi` inputs — no
//! `position_ids` (baked into the embedding module) and no
//! `token_type_ids` (DistilBERT drops segment embeddings entirely).

#![cfg(feature = "gpu")]

mod common;

use flodl_hf::models::distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification,
};
use flodl_hf::task_heads::{
    masked_lm_loss, question_answering_loss, sequence_classification_loss,
    token_classification_loss,
};

use common::{
    BATCH, CUDA, SEQ, assert_grads_flowed, extended_attention_mask, input_ids, mlm_labels,
    qa_positions, seqcls_labels, tokcls_labels,
};

const NUM_LABELS: i64 = 3;

/// Graph input order (per `distilbert_backbone_flow`):
/// `[input_ids, attention_mask_additive]`.
fn distilbert_inputs(cfg: &DistilBertConfig) -> Vec<flodl::Variable> {
    vec![
        input_ids(BATCH, SEQ, cfg.vocab_size),
        extended_attention_mask(BATCH, SEQ),
    ]
}

#[test]
fn distilbert_seqcls_cuda_smoke() {
    let cfg = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForSequenceClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&distilbert_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, NUM_LABELS]);

    let labels = seqcls_labels(BATCH, NUM_LABELS);
    let loss = sequence_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn distilbert_tokcls_cuda_smoke() {
    let cfg = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForTokenClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&distilbert_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, NUM_LABELS]);

    let labels = tokcls_labels(BATCH, SEQ, NUM_LABELS);
    let loss = token_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn distilbert_qa_cuda_smoke() {
    let cfg = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForQuestionAnswering::on_device(&cfg, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&distilbert_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, 2]);

    let (starts, ends) = qa_positions(BATCH, SEQ);
    let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn distilbert_mlm_cuda_smoke() {
    let cfg = DistilBertConfig::distilbert_base_uncased();
    let head = DistilBertForMaskedLM::on_device(&cfg, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&distilbert_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, cfg.vocab_size]);

    let labels = mlm_labels(BATCH, SEQ, cfg.vocab_size);
    let loss = masked_lm_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}
