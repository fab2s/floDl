//! Shared helpers for the `*_cuda_smoke.rs` integration tests.
//!
//! The smoke tier runs real model presets (`bert_base_uncased`,
//! `roberta_base`, `distilbert_base_uncased`, `xlm_roberta_base`,
//! `albert_base_v2`) with fresh random weights on
//! [`Device::CUDA(0)`], feeds synthetic-but-shape-realistic input
//! tensors through each head, and asserts that backward flows a non-
//! zero gradient into the backbone. This catches CUDA-path regressions
//! that structural CPU unit tests can miss (kernel selection,
//! stream/event wiring, tied-weight grad accumulation, etc.).
//!
//! The whole module is gated behind `#[cfg(feature = "gpu")]` so
//! non-CUDA builds (including GitHub CI's CPU test leg) compile and
//! run unaffected.

#![cfg(feature = "gpu")]
#![allow(dead_code)] // not every smoke test file uses every helper

use flodl::{Device, Tensor, Variable};
use flodl_hf::models::bert::build_extended_attention_mask;

/// Single-GPU target for the smoke tier. Multi-GPU validation is the
/// job of `flodl`'s DDP tests, not of per-family head smoke.
pub const CUDA: Device = Device::CUDA(0);

/// Batch size used across the smoke tier. Chosen to exercise real
/// batched attention (`batch > 1`) without blowing VRAM on
/// 250k-vocab XLM-R MLM heads.
pub const BATCH: i64 = 4;

/// Sequence length used across the smoke tier. Long enough to split
/// work across CUDA blocks in the softmax / matmul kernels, short
/// enough to keep the full `base`-preset smoke suite in the low tens
/// of seconds on a single consumer GPU.
pub const SEQ: i64 = 32;

/// Deterministic `[batch, seq]` int64 token ids in `[0, vocab)`.
/// Using a ramp-mod-vocab keeps every seed reproducible without
/// pulling in an RNG dependency.
pub fn input_ids(batch: i64, seq: i64, vocab: i64) -> Variable {
    let data: Vec<i64> = (0..batch * seq).map(|i| i % vocab).collect();
    Variable::new(
        Tensor::from_i64(&data, &[batch, seq], CUDA).expect("from_i64 input_ids"),
        false,
    )
}

/// `[batch, seq]` int64 position ids `[0, seq)` repeated per batch row.
pub fn position_ids(batch: i64, seq: i64) -> Variable {
    let mut data = Vec::with_capacity((batch * seq) as usize);
    for _ in 0..batch {
        for p in 0..seq {
            data.push(p);
        }
    }
    Variable::new(
        Tensor::from_i64(&data, &[batch, seq], CUDA).expect("from_i64 position_ids"),
        false,
    )
}

/// `[batch, seq]` all-zero int64 token type ids. Single-sentence
/// input is enough to exercise the `token_type_embeddings` lookup;
/// sentence-pair coverage belongs in the live parity suite.
pub fn token_type_ids(batch: i64, seq: i64) -> Variable {
    let data = vec![0i64; (batch * seq) as usize];
    Variable::new(
        Tensor::from_i64(&data, &[batch, seq], CUDA).expect("from_i64 token_type_ids"),
        false,
    )
}

/// `[batch, 1, 1, seq]` additive attention mask built from an all-
/// ones raw mask. Every family's graph expects this extended form (see
/// [`build_extended_attention_mask`]).
pub fn extended_attention_mask(batch: i64, seq: i64) -> Variable {
    let data = vec![1.0f32; (batch * seq) as usize];
    let raw = Tensor::from_f32(&data, &[batch, seq], CUDA).expect("from_f32 attention_mask");
    Variable::new(
        build_extended_attention_mask(&raw).expect("build_extended_attention_mask"),
        false,
    )
}

/// `[batch]` int64 class labels cycling through `[0, num_labels)`.
pub fn seqcls_labels(batch: i64, num_labels: i64) -> Variable {
    let data: Vec<i64> = (0..batch).map(|i| i % num_labels).collect();
    Variable::new(
        Tensor::from_i64(&data, &[batch], CUDA).expect("from_i64 seqcls_labels"),
        false,
    )
}

/// `[batch, seq]` int64 per-token labels. Every fifth position is
/// `-100` so the loss fn's `ignore_index` branch gets exercised.
pub fn tokcls_labels(batch: i64, seq: i64, num_labels: i64) -> Variable {
    let mut data = Vec::with_capacity((batch * seq) as usize);
    for i in 0..batch * seq {
        data.push(if i % 5 == 0 { -100 } else { i % num_labels });
    }
    Variable::new(
        Tensor::from_i64(&data, &[batch, seq], CUDA).expect("from_i64 tokcls_labels"),
        false,
    )
}

/// `[batch, seq]` int64 MLM labels. Every seventh position carries
/// the original token id (≈ 15% masking); the rest are `-100`. Keeps
/// the cross-entropy reduction working on a non-empty subset while
/// exercising the ignore path on the majority.
pub fn mlm_labels(batch: i64, seq: i64, vocab: i64) -> Variable {
    let mut data = Vec::with_capacity((batch * seq) as usize);
    for i in 0..batch * seq {
        data.push(if i % 7 == 0 { i % vocab } else { -100 });
    }
    Variable::new(
        Tensor::from_i64(&data, &[batch, seq], CUDA).expect("from_i64 mlm_labels"),
        false,
    )
}

/// `(start_positions, end_positions)`: two `[batch]` int64 tensors in
/// `[0, seq)`, with `end > start` per batch row.
pub fn qa_positions(batch: i64, seq: i64) -> (Variable, Variable) {
    let starts: Vec<i64> = (0..batch).map(|i| i % seq).collect();
    let ends: Vec<i64> = (0..batch).map(|i| (i + 3) % seq).collect();
    (
        Variable::new(
            Tensor::from_i64(&starts, &[batch], CUDA).expect("from_i64 qa starts"),
            false,
        ),
        Variable::new(
            Tensor::from_i64(&ends, &[batch], CUDA).expect("from_i64 qa ends"),
            false,
        ),
    )
}

/// Assert that `backward()` actually populated gradients on at least
/// one parameter with non-zero magnitude. Catches two failure modes:
///
/// 1. Autograd graph broken upstream of the params (no param sees a
///    gradient at all).
/// 2. Every param sees a grad tensor but it's all zeros — typically
///    a detach that shouldn't have happened.
pub fn assert_grads_flowed(graph: &flodl::Graph) {
    use flodl::nn::Module;
    let mut any_grad = false;
    let mut any_nonzero = false;
    for p in graph.parameters() {
        if let Some(g) = p.variable.grad() {
            any_grad = true;
            let sum_abs = g
                .abs()
                .expect("abs on grad")
                .sum()
                .expect("sum on |grad|")
                .to_f32_vec()
                .expect("to_f32_vec on scalar")[0];
            if sum_abs > 0.0 {
                any_nonzero = true;
                break;
            }
        }
    }
    assert!(any_grad, "no parameter received a gradient — autograd graph broken upstream of params");
    assert!(any_nonzero, "all gradients were zero — backward did not flow real signal");
}
