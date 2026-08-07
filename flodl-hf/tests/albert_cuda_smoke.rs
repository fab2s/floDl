//! CUDA smoke tier for the ALBERT family.
//!
//! Real [`albert_base_v2`](flodl_hf::models::albert::AlbertConfig::albert_base_v2)
//! preset, fresh random weights, one forward + loss + backward per
//! head. ALBERT graphs take the same 4 `forward_multi` inputs as BERT.
//!
//! An extra `albert_cross_layer_sharing_cuda_smoke` test verifies that
//! gradients from all `num_hidden_layers` applications accumulate into
//! the single shared [`AlbertLayerStack`](flodl_hf::models::albert::AlbertLayerStack)
//! — the load-bearing structural invariant that distinguishes ALBERT
//! from every other family in this crate.

#![cfg(feature = "gpu")]

mod common;

use flodl_hf::models::albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering,
    AlbertForSequenceClassification, AlbertForTokenClassification, AlbertModel,
};
use flodl_hf::path::hf_key_from_flodl_key;
use flodl_hf::task_heads::{
    masked_lm_loss, question_answering_loss, sequence_classification_loss,
    token_classification_loss,
};

use common::{
    assert_grads_flowed, extended_attention_mask, input_ids, mlm_labels, position_ids,
    qa_positions, seqcls_labels, tokcls_labels, token_type_ids, BATCH, CUDA, SEQ,
};

const NUM_LABELS: i64 = 3;

/// Graph input order (per `albert_backbone_flow`):
/// `[input_ids, position_ids, token_type_ids, attention_mask_additive]`.
fn albert_inputs(cfg: &AlbertConfig) -> Vec<flodl::Variable> {
    vec![
        input_ids(BATCH, SEQ, cfg.vocab_size),
        position_ids(BATCH, SEQ),
        token_type_ids(BATCH, SEQ),
        extended_attention_mask(BATCH, SEQ),
    ]
}

#[test]
fn albert_seqcls_cuda_smoke() {
    let cfg = AlbertConfig::albert_base_v2();
    let head = AlbertForSequenceClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head.graph().forward_multi(&albert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, NUM_LABELS]);

    let labels = seqcls_labels(BATCH, NUM_LABELS);
    let loss = sequence_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn albert_tokcls_cuda_smoke() {
    let cfg = AlbertConfig::albert_base_v2();
    let head = AlbertForTokenClassification::on_device(&cfg, NUM_LABELS, CUDA).unwrap();

    let logits = head.graph().forward_multi(&albert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, NUM_LABELS]);

    let labels = tokcls_labels(BATCH, SEQ, NUM_LABELS);
    let loss = token_classification_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn albert_qa_cuda_smoke() {
    let cfg = AlbertConfig::albert_base_v2();
    let head = AlbertForQuestionAnswering::on_device(&cfg, CUDA).unwrap();

    let logits = head.graph().forward_multi(&albert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, 2]);

    let (starts, ends) = qa_positions(BATCH, SEQ);
    let loss = question_answering_loss(&logits, &starts, &ends).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

#[test]
fn albert_mlm_cuda_smoke() {
    let cfg = AlbertConfig::albert_base_v2();
    let head = AlbertForMaskedLM::on_device(&cfg, CUDA).unwrap();

    let logits = head.graph().forward_multi(&albert_inputs(&cfg)).unwrap();
    assert_eq!(logits.shape(), vec![BATCH, SEQ, cfg.vocab_size]);

    let labels = mlm_labels(BATCH, SEQ, cfg.vocab_size);
    let loss = masked_lm_loss(&logits, &labels).unwrap();
    assert!(loss.data().to_f32_vec().unwrap()[0].is_finite());

    loss.backward().unwrap();
    assert_grads_flowed(head.graph());
}

/// Cross-layer weight sharing is ALBERT's defining feature: the same
/// [`AlbertLayerStack`] parameters are reused `num_hidden_layers`
/// times. On backward, gradient contributions from all applications
/// must accumulate into those shared tensors — not overwrite each
/// other.
///
/// This test validates accumulation by comparing two graphs that
/// differ only in `num_hidden_layers`. The gradient sum on the shared
/// block's Q-projection weight should grow roughly with the layer
/// count. A regression where the Graph treats each application as
/// fresh (last-writer-wins) would leave both graphs with roughly the
/// same gradient magnitude, tripping the inequality below.
#[test]
fn albert_cross_layer_sharing_cuda_smoke() {
    fn grad_norm_on_shared_block(num_layers: i64) -> f32 {
        let mut cfg = AlbertConfig::albert_base_v2();
        cfg.num_hidden_layers = num_layers;
        let graph = AlbertModel::on_device(&cfg, CUDA).unwrap();
        let last_hidden = graph
            .forward_multi(&[
                input_ids(BATCH, SEQ, cfg.vocab_size),
                position_ids(BATCH, SEQ),
                token_type_ids(BATCH, SEQ),
                extended_attention_mask(BATCH, SEQ),
            ])
            .unwrap();
        // Sum of pooler output as a scalar "loss" — simple, non-zero
        // signal that flows through every layer application.
        let loss = last_hidden.sum().unwrap();
        loss.backward().unwrap();

        // Address the shared block's query weight by its canonical HF
        // key. `named_parameters()` emits `<tag>/<leaf>` internally;
        // [`hf_key_from_flodl_key`] swaps the final `/` for `.` so we
        // match against the same dotted form HF state_dicts ship.
        let target = "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight";
        let mut sum_abs = 0.0f32;
        let mut found = false;
        for (name, p) in graph.named_parameters() {
            if hf_key_from_flodl_key(&name) == target
                && let Some(g) = p.variable.grad()
            {
                sum_abs = g.abs().unwrap().sum().unwrap().to_f32_vec().unwrap()[0];
                found = true;
                break;
            }
        }
        assert!(found, "shared block query.weight not found under HF key {target}");
        sum_abs
    }

    let g3 = grad_norm_on_shared_block(3);
    let g1 = grad_norm_on_shared_block(1);

    // With healthy accumulation, g3 should be materially larger than g1
    // (3 applications vs 1). The 1.5x lower bound gives slack for the
    // fact that gradients through deeper layers are not exactly 3x larger
    // — they compose nonlinearly — while still catching a last-writer-wins
    // regression where g3 would hover around g1.
    eprintln!("albert sharing grad sum: layers=1 -> {g1:.4e}, layers=3 -> {g3:.4e}");
    assert!(
        g3 > g1 * 1.5,
        "ALBERT cross-layer sharing regressed: grad sum with 3 layers ({g3:.4e}) \
         should be > 1.5x grad sum with 1 layer ({g1:.4e}) — same shared block \
         accumulated across 3 applications. Close-to-equal values suggest the \
         Graph treated each application as a fresh module."
    );
}
