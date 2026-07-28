//! OLMo-150M expressed as a FlowBuilder graph.
//!
//! Same architecture as `olmo.rs`, same constants (imported from it, not
//! copied), through the graph engine instead of a hand-written `forward`.
//! The pair is the transformer counterpart of `resnet` / `resnet-graph`:
//! eager and graph arms of one model, each validating the other.
//!
//! # Why a pre-norm transformer maps cleanly onto `also()`
//!
//! `also(sub_graph)` computes `x + branch(x)`, which IS a pre-norm block:
//!
//! ```text
//! x + attn(norm(x))          ->  .also(RMSNorm -> CausalSelfAttention)
//! x + ffn(norm(x))           ->  .also(RMSNorm -> Linear -> SwiGLU -> Linear)
//! ```
//!
//! Unlike ResNet there is no post-residual activation, so the blocks are
//! `.also()` calls with nothing between them — twelve blocks are twenty-four
//! chained calls. No new graph capability was needed to express any of it.
//!
//! # What the SVG buys
//!
//! The model graph renders as a diagram, which the eager arm cannot do — the
//! reason this exists beyond engine parity. `SwiGLU` being one module rather
//! than an inlined split-and-multiply keeps the FFN readable as three boxes.
//! Expect a TALL picture: twelve structurally identical stanzas is an honest
//! rendering of a twelve-block transformer.

use flodl::autograd::Variable;
use flodl::graph::Graph;
use flodl::nn::{
    AdamW, Buffer, CosineScheduler, Embedding, Linear, Module, MultiheadAttention,
    Parameter, RMSNorm, RotaryEmbedding, SwiGLU, WarmupScheduler,
};
use flodl::tensor::{DType, Device, Result, Tensor, TensorOptions};
use flodl::FlowBuilder;

use super::olmo::{
    D_MODEL, EMBEDDING_SIZE, HEAD_DIM, LN_EPS, MLP_HIDDEN, N_HEADS, N_LAYERS, ROPE_THETA,
    SEQ_LEN,
};
use super::ModelDef;
use crate::config::ModelDefaults;
use crate::download::OLMO_TRAIN_BYTES;

pub fn def() -> ModelDef {
    ModelDef {
        name: "olmo-graph",
        description: "OLMo-150M (configs/tiny) LM pretraining, Graph builder",
        build: build_model,
        // Everything below is shared with the eager arm by reference, so the
        // two cannot drift on data, loss, or optimisation — only on how the
        // forward pass is expressed, which is the thing under comparison.
        dataset: super::olmo::make_dataset,
        dataset_size_hint: |_| Ok(((OLMO_TRAIN_BYTES / 2 - 1) / SEQ_LEN as u64) as usize),
        train_fn: super::olmo::train_step,
        eval_fn: Some(super::olmo::eval_loss),
        test_dataset: Some(super::olmo::make_eval_dataset),
        augment_fn: None,
        optimizer: |p, lr| {
            Box::new(AdamW::with_groups(0.1).betas(0.9, 0.95).eps(1e-8).group(p, lr).build())
        },
        scheduler: Some(|lr, total, _world_size| {
            Box::new(WarmupScheduler::new(
                CosineScheduler::new(lr, lr * 0.1, total),
                lr,
                (total / 20).max(1),
            ))
        }),
        reference: "OLMo-150M tiny config, Graph builder; parity arm of `olmo` \
                    ([OLMo](https://github.com/allenai/OLMo), configs/tiny)",
        eval_higher_is_better: false,
        published_eval: None,
        needs_baseline_eval: false,
        defaults: ModelDefaults {
            epochs: 3,
            batches_per_epoch: 0,
            batch_size: 4,
            lr: 6.0e-4,
        },
    }
}

// ---------------------------------------------------------------------------
// Self-attention as a single-input node
// ---------------------------------------------------------------------------

/// Causal self-attention with a single input, so it can be a graph node.
///
/// A graph node is `forward(&Variable) -> Variable`, while
/// [`MultiheadAttention::forward_ext`] wants query, key, value and a mask.
/// Self-attention passes the same tensor three times, and the mask is derived
/// from the input's own sequence length — so the extra arguments are not
/// really inputs, they are a shape the wrapper can supply.
struct CausalSelfAttention {
    attn: MultiheadAttention,
    device: Device,
}

impl CausalSelfAttention {
    fn new(device: Device, rope: &RotaryEmbedding) -> Result<Self> {
        Ok(Self {
            attn: MultiheadAttention::on_device(D_MODEL, N_HEADS, device)?
                .rotary(rope.clone())?,
            device,
        })
    }
}

/// Upper triangle above the diagonal = positions a token may not attend to.
///
/// Built per call rather than cached in a field, deliberately. A stored mask
/// is a tensor the module owns but `Module::move_to_device` (which walks
/// parameters and buffers) would not follow, and registering it as a *buffer*
/// to fix that would enrol a 256x256 constant in DDP averaging every round —
/// pure wire traffic for a value every rank can compute. Two tiny kernels next
/// to twelve blocks of attention is not worth either hazard.
fn causal_mask(seq_len: i64, device: Device) -> Result<Tensor> {
    let opts = TensorOptions { dtype: DType::Float32, device };
    Tensor::ones(&[seq_len, seq_len], opts)?.triu(1)
}

impl Module for CausalSelfAttention {
    fn name(&self) -> &str { "causal_self_attn" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let mask = causal_mask(input.shape()[1], self.device)?;
        self.attn.forward_ext(input, input, input, Some(&mask))
    }

    // Delegated rather than left to the `sub_modules` default: the attention
    // is owned by value, not behind an `Rc<dyn Module>`. Forgetting these
    // would hand the trainer a model with nothing to train.
    fn parameters(&self) -> Vec<Parameter> { self.attn.parameters() }
    fn buffers(&self) -> Vec<Buffer> { self.attn.buffers() }
    fn set_training(&self, mode: bool) { self.attn.set_training(mode) }
}

// ---------------------------------------------------------------------------
// Blocks and model
// ---------------------------------------------------------------------------

/// Pre-norm attention branch: `norm -> causal self-attention`.
/// `also()` adds the residual, so it is absent here by design.
fn attn_branch(device: Device, rope: &RotaryEmbedding) -> Result<Graph> {
    FlowBuilder::from(RMSNorm::on_device(D_MODEL, device)?.eps(LN_EPS))
        .through(CausalSelfAttention::new(device, rope)?)
        .build()
}

/// Pre-norm SwiGLU FFN branch: `norm -> proj(d -> 8d) -> SwiGLU -> out(4d -> d)`.
///
/// `MLP_HIDDEN` is 8*d and the gate halves it, which is why the output
/// projection reads `MLP_HIDDEN / 2` — OLMo's mlp_ratio 8 yields an effective
/// intermediate of 4*d.
fn ffn_branch(device: Device) -> Result<Graph> {
    FlowBuilder::from(RMSNorm::on_device(D_MODEL, device)?.eps(LN_EPS))
        .through(Linear::no_bias_on_device(D_MODEL, MLP_HIDDEN, device)?)
        .through(SwiGLU)
        .through(Linear::no_bias_on_device(MLP_HIDDEN / 2, D_MODEL, device)?)
        .build()
}

fn build_model(device: Device) -> Result<Box<dyn Module>> {
    // One RoPE table set shared by every block, exactly as the eager arm does
    // (Clone shares the tables rather than copying them).
    let rope = RotaryEmbedding::on_device_theta(HEAD_DIM, SEQ_LEN as i64, ROPE_THETA, device)?;

    // Token embeddings only — positions enter through RoPE inside attention.
    let mut fb = FlowBuilder::from(Embedding::on_device(EMBEDDING_SIZE, D_MODEL, device)?);
    for _ in 0..N_LAYERS {
        fb = fb.also(attn_branch(device, &rope)?).also(ffn_branch(device)?);
    }
    let graph = fb
        .through(RMSNorm::on_device(D_MODEL, device)?.eps(LN_EPS))
        .through(Linear::no_bias_on_device(D_MODEL, EMBEDDING_SIZE, device)?)
        .build()?;
    Ok(Box::new(graph))
}
