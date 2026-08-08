//! CUDA smoke tier for the DeBERTa-v2 / DeBERTa-v3 family.
//!
//! [`deberta_v3_base`](flodl_hf::models::deberta_v2::DebertaV2Config::deberta_v3_base)
//! preset, fresh random weights, one forward + loss + backward per
//! head. DeBERTa-v2 graphs take **2** `forward_multi` inputs —
//! `input_ids` and a flat `[B, S]` padding mask — unlike BERT's 4 or
//! RoBERTa's 3. The encoder expands the flat mask to the
//! `[B, 1, S, S]` disentangled-attention form internally.
//!
//! This tier catches CUDA-path regressions the structural CPU unit
//! tests can miss (disentangled-attention gather kernels, rel_embeddings
//! LayerNorm on GPU, log-bucket integer arithmetic on the int64 path).

#![cfg(feature = "gpu")]

mod common;

use flodl::{Tensor, Variable};
use flodl_hf::models::deberta_v2::{
    DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification,
};
use flodl_hf::task_heads::{
    masked_lm_loss, question_answering_loss, sequence_classification_loss,
    token_classification_loss,
};

use common::{
    BATCH, CUDA, SEQ, assert_grads_flowed, input_ids, mlm_labels, qa_positions, seqcls_labels,
    tokcls_labels,
};

const NUM_LABELS: i64 = 3;

/// Flat `[B, S]` int64 attention mask (all-ones, no padding). DeBERTa's
/// encoder expands this to `[B, 1, S, S]` internally; unlike BERT/RoBERTa
/// the graph does NOT want a pre-built extended additive mask.
fn flat_attention_mask(batch: i64, seq: i64) -> Variable {
    let data = vec![1i64; (batch * seq) as usize];
    Variable::new(
        Tensor::from_i64(&data, &[batch, seq], CUDA).expect("from_i64 flat_mask"),
        false,
    )
}

/// Graph input order: `[input_ids, flat_attention_mask]`.
fn deberta_v2_inputs(cfg: &DebertaV2Config) -> Vec<Variable> {
    vec![
        input_ids(BATCH, SEQ, cfg.vocab_size),
        flat_attention_mask(BATCH, SEQ),
    ]
}

#[test]
fn deberta_v2_seqcls_cuda_smoke() {
    let cfg = DebertaV2Config::deberta_v3_base();
    let head = DebertaV2ForSequenceClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&deberta_v2_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, NUM_LABELS]);

    let labels = seqcls_labels(BATCH, NUM_LABELS);
    let loss = sequence_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn deberta_v2_tokcls_cuda_smoke() {
    let cfg = DebertaV2Config::deberta_v3_base();
    let head = DebertaV2ForTokenClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&deberta_v2_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, NUM_LABELS]);

    let labels = tokcls_labels(BATCH, SEQ, NUM_LABELS);
    let loss = token_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn deberta_v2_qa_cuda_smoke() {
    let cfg = DebertaV2Config::deberta_v3_base();
    let head = DebertaV2ForQuestionAnswering::on_device(&cfg, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&deberta_v2_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, 2]);

    let (starts, ends) = qa_positions(BATCH, SEQ);
    let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn deberta_v2_mlm_cuda_smoke() {
    let cfg = DebertaV2Config::deberta_v3_base();
    let head = DebertaV2ForMaskedLM::on_device(&cfg, CUDA).unwrap();

    let logits = head
        .graph()
        .forward_multi(&deberta_v2_inputs(&cfg))
        .unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, cfg.vocab_size]);

    let labels = mlm_labels(BATCH, SEQ, cfg.vocab_size);
    let loss = masked_lm_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}
