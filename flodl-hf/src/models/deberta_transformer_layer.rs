//! DeBERTa-v2 / DeBERTa-v3 encoder layer with disentangled self-attention.
//!
//! This layer is NOT a drop-in for the cross-family
//! [`TransformerLayer`](crate::models::transformer_layer::TransformerLayer) —
//! disentangled attention changes the math, not just the weight-key naming.
//! Each attention step computes three additive score components
//! (content-to-content, content-to-position, position-to-content) instead of
//! one, scaled by `sqrt(head_dim * scale_factor)` where `scale_factor` counts
//! how many position components are active.
//!
//! # Supported configurations
//!
//! This port hard-requires the exact knobs `microsoft/deberta-v3-base`
//! ships; deviations surface as loud parse-time errors from
//! [`DebertaV2Config::from_json_str`](crate::models::deberta_v2::DebertaV2Config::from_json_str).
//! In particular:
//!
//! - `relative_attention: true`
//! - `share_att_key: true` — Q/K projections are reused to project
//!   `rel_embeddings`; no separate `pos_query_proj` / `pos_key_proj` paths.
//! - `pos_att_type` contains both `c2p` and `p2c` — `scale_factor = 3` is
//!   baked in.
//! - `position_biased_input: false` — consumed by embeddings, not this
//!   layer.
//! - `norm_rel_ebd: "layer_norm"` — the encoder LayerNorm-normalises
//!   `rel_embeddings.weight` once before threading it through every layer.
//! - `legacy: false` — MLM head is the non-legacy variant (hidden-size
//!   predictions; see [`crate::models::deberta_v2`]).
//!
//! v1 DeBERTa checkpoints, experimental `share_att_key=false` fine-tunes,
//! and any `conv_kernel_size > 0` variant are rejected at config time.
//!
//! # Weight-key layout
//!
//! Each [`DebertaV2TransformerLayer`] emits parameters under these suffixes,
//! relative to the layer tag `deberta.encoder.layer.{i}`:
//!
//! ```text
//! attention.self.query_proj.{weight,bias}
//! attention.self.key_proj.{weight,bias}
//! attention.self.value_proj.{weight,bias}
//! attention.output.dense.{weight,bias}
//! attention.output.LayerNorm.{weight,bias}
//! intermediate.dense.{weight,bias}
//! output.dense.{weight,bias}
//! output.LayerNorm.{weight,bias}
//! ```
//!
//! This is the state_dict layout HuggingFace `DebertaV2Model` serialises.
//!
//! # Attention-mask convention
//!
//! This layer's attention forward consumes an additive f32 mask of shape
//! `[B, 1, S, S]` (0.0 for attending positions, `f32::MIN` for masked
//! positions). Building that mask from a `[B, S]` flat `{0, 1}` mask is
//! the encoder's job — see
//! [`build_deberta_attention_mask`](crate::models::deberta_v2::build_deberta_attention_mask).
//! The mask masks both query and key dimensions of the attention score
//! grid (HF `get_attention_mask`), unlike BERT which masks only keys.

use std::cell::Cell;

use flodl::nn::{Dropout, GELU, GeluApprox, LayerNorm, Linear, Module, Parameter};
use flodl::{DType, Device, Result, Tensor, TensorOptions, Variable};

use crate::path::prefix_params;

// ─── Config ─────────────────────────────────────────────────────────────

/// Hyperparameters for one DeBERTa-v2 transformer layer.
///
/// `share_att_key` / `pos_att_type` / `relative_attention` are NOT
/// fields — this layer hard-requires the v3-base-equivalent values
/// (share_att_key=true, pos_att_type=c2p+p2c, relative_attention=true)
/// and the config parser rejects anything else. Keeping them out of
/// this struct means no dead branches in the hot path.
#[derive(Debug, Clone, Copy)]
pub struct DebertaV2LayerConfig {
    pub hidden_size: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub layer_norm_eps: f64,
    /// Number of log-buckets for relative positions (256 for v3-base).
    pub position_buckets: i64,
    /// Max relative position range after bucketing (512 for v3-base,
    /// taken from `max_position_embeddings` when config says -1).
    pub max_relative_positions: i64,
    /// FFN activation form (parsed from HF `hidden_act` upstream).
    pub hidden_act: GeluApprox,
}

// ─── Relative-position helpers ─────────────────────────────────────────

/// Map raw integer relative positions to log-bucketed positions.
///
/// Mirrors HF Python `make_log_bucket_position`:
/// - positions with `|rp| <= mid` pass through unchanged (`mid = bucket_size/2`)
/// - positions with `|rp| > mid` are logarithmically compressed into
///   `[mid, 2*mid - 1]` with the original sign preserved
///
/// Returns int64 bucket positions with the same shape as `rel_pos`. The
/// computation is done in f32 (the intermediate log/ceil steps need
/// float precision) and cast back to int64 at the end.
fn make_log_bucket_position(
    rel_pos: &Tensor,
    bucket_size: i64,
    max_position: i64,
) -> Result<Tensor> {
    let mid = bucket_size / 2;
    let device = rel_pos.device();
    let f32_opts = TensorOptions {
        dtype: DType::Float32,
        device,
    };

    let rp_f = rel_pos.to_dtype(DType::Float32)?;

    // sign(rp), used to re-sign the log-compressed magnitude.
    let sign = rp_f.sign()?;

    // abs_pos = |rp| outside the near range; mid-1 inside (the inside
    // branch is masked away by the outer where, but we still need a
    // positive value for log() not to NaN).
    let abs_rp = rp_f.abs()?;
    let near_mask = rp_f
        .lt_scalar(mid as f64)?
        .logical_and(&rp_f.gt_scalar(-(mid as f64))?)?;
    let mid_minus_one_tensor = Tensor::full(&rel_pos.shape(), (mid - 1) as f64, f32_opts)?;
    let abs_pos = Tensor::where_cond(&near_mask, &mid_minus_one_tensor, &abs_rp)?;

    // log_pos = ceil( log(abs_pos/mid) / log((max_position-1)/mid) * (mid-1) ) + mid
    //
    // Precision note: the denominator log is a constant. Computing it as
    // a host `f64.ln()` and then `.div_scalar(f64)` against a tensor
    // whose log was taken in f32 leaks precision — `log(3.5)` via the
    // two paths can disagree by ~1 ulp, pushing `ratio = 1.0 + ε` and
    // making `ceil(ratio * (mid-1))` overshoot by 1. Clamp the ratio
    // to `[0, 1]` before the ceil so far-range buckets land cleanly in
    // `[mid, 2*mid - 1]`. Matches the Python output bit-for-bit on
    // every case exercised by the parity fixture.
    let log_denom = ((max_position as f64 - 1.0) / mid as f64).ln();
    let log_pos = abs_pos
        .div_scalar(mid as f64)?
        .log()?
        .div_scalar(log_denom)?
        .clamp(0.0, 1.0)?
        .mul_scalar((mid - 1) as f64)?
        .ceil()?
        .add_scalar(mid as f64)?;

    // bucket_pos = where(|rp| <= mid, rp, log_pos * sign)
    let in_range = abs_rp.le_scalar(mid as f64)?;
    let log_signed = log_pos.mul(&sign)?;
    let bucket = Tensor::where_cond(&in_range, &rp_f, &log_signed)?;

    bucket.to_dtype(DType::Int64)
}

/// Build the relative-position grid for a self-attention call, shape
/// `[1, seq_len, seq_len]`, int64.
///
/// Mirrors HF Python `build_relative_position` restricted to the
/// `bucket_size > 0` path (the only path we support). Entry `[0, q, k]`
/// is the bucketed relative distance from query position `q` to key
/// position `k`.
pub fn build_relative_position(
    seq_len: i64,
    position_buckets: i64,
    max_relative_positions: i64,
    device: Device,
) -> Result<Tensor> {
    let i64_opts = TensorOptions {
        dtype: DType::Int64,
        device,
    };
    let ids = Tensor::arange(0.0, seq_len as f64, 1.0, i64_opts)?;
    // q_ids[:, None] - k_ids[None, :]
    let q = ids
        .unsqueeze(-1)?
        .expand(&[seq_len, seq_len])?
        .contiguous()?;
    let k = ids
        .unsqueeze(0)?
        .expand(&[seq_len, seq_len])?
        .contiguous()?;
    let rel = q.sub(&k)?;

    let bucketed = make_log_bucket_position(&rel, position_buckets, max_relative_positions)?;
    bucketed.unsqueeze(0)
}

// ─── DisentangledSelfAttention ──────────────────────────────────────────

/// DeBERTa-v2 disentangled self-attention.
///
/// Forward takes four inputs (hidden_states, attention_mask,
/// relative_pos, rel_embeddings) and returns the attention output
/// projected back to `[B, S, H]` — before the residual-plus-LayerNorm
/// step owned by [`DebertaV2SelfOutput`].
///
/// Always computes `content-to-content + content-to-position +
/// position-to-content` (scale_factor = 3). The `share_att_key=true`
/// path is the only path: the relative-position embedding is projected
/// through the same `query_proj` / `key_proj` linear layers that project
/// content, so no separate position-projection weights exist.
pub struct DisentangledSelfAttention {
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    num_heads: i64,
    head_dim: i64,
    attn_dropout: Dropout,
    pos_dropout: Dropout,
    position_buckets: i64,
    #[allow(dead_code)]
    max_relative_positions: i64,
    training: Cell<bool>,
}

impl DisentangledSelfAttention {
    pub fn on_device(config: &DebertaV2LayerConfig, device: Device) -> Result<Self> {
        assert!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            config.hidden_size,
            config.num_attention_heads,
        );
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(DisentangledSelfAttention {
            query_proj: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            key_proj: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            value_proj: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            num_heads: config.num_attention_heads,
            head_dim,
            attn_dropout: Dropout::new(config.attention_probs_dropout_prob),
            pos_dropout: Dropout::new(config.hidden_dropout_prob),
            position_buckets: config.position_buckets,
            max_relative_positions: config.max_relative_positions,
            training: Cell::new(true),
        })
    }

    /// Reshape `[B, S, H]` → `[B*Nh, S, D]` ready for 3-D batched matmul.
    /// Matches HF's `transpose_for_scores`.
    fn split_heads(&self, x: &Variable) -> Result<Variable> {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        x.reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .reshape(&[batch * self.num_heads, seq, self.head_dim])
    }

    /// Merge heads back: `[B*Nh, S, D]` → `[B, S, H]`.
    fn merge_heads(&self, x: &Variable, batch: i64) -> Result<Variable> {
        let shape = x.shape();
        let seq = shape[1];
        x.reshape(&[batch, self.num_heads, seq, self.head_dim])?
            .transpose(1, 2)?
            .reshape(&[batch, seq, self.num_heads * self.head_dim])
    }

    /// Compute the disentangled score bias: c2p + p2c components, both
    /// divided by `sqrt(head_dim * scale_factor)` where scale_factor = 3.
    ///
    /// Returns `[B*Nh, S, S]`.
    fn disentangled_bias(
        &self,
        query_layer: &Variable,    // [B*Nh, S, D]
        key_layer: &Variable,      // [B*Nh, S, D]
        relative_pos: &Tensor,     // [1, S, S] int64
        rel_embeddings: &Variable, // [2P, H] — already LayerNormed by encoder
        scale: f64,
    ) -> Result<Variable> {
        let att_span = self.position_buckets;
        let two_span = att_span * 2;

        // Take first 2*att_span rows, add batch dim → [1, 2P, H]
        let rel = rel_embeddings.narrow(0, 0, two_span)?.unsqueeze(0)?;
        let rel = self.pos_dropout.forward(&rel)?;

        // share_att_key=true: reuse query_proj / key_proj to project rel_embeddings
        let bh = query_layer.shape()[0]; // B*Nh
        let batch = bh / self.num_heads;

        let pos_key = self
            .split_heads(&self.key_proj.forward(&rel)?)? // [Nh, 2P, D]  (batch=1 so B*Nh = Nh)
            .repeat(&[batch, 1, 1])?; // [B*Nh, 2P, D]
        let pos_query = self
            .split_heads(&self.query_proj.forward(&rel)?)?
            .repeat(&[batch, 1, 1])?;

        // c2p: query @ pos_key.T, gather at (rel_pos + att_span)
        //
        // `add_scalar` / `clamp` with f64 scalars promote an int64 input
        // to f32 in libtorch, but `gather` requires int32/int64 indices.
        // Cast back to int64 after the clamp.
        let c2p_scores = query_layer.matmul(&pos_key.transpose(-1, -2)?)?; // [B*Nh, S, 2P]
        let c2p_pos = relative_pos
            .add_scalar(att_span as f64)?
            .clamp(0.0, (two_span - 1) as f64)?
            .to_dtype(DType::Int64)?;
        let s = c2p_scores.shape()[1]; // seq length (query size)
        let c2p_idx = c2p_pos.squeeze(0)?.expand(&[bh, s, s])?.contiguous()?;
        let c2p_att = c2p_scores.gather(-1, &c2p_idx)?; // [B*Nh, S, S]

        // p2c: key @ pos_query.T, gather at (-rel_pos + att_span), then transpose
        let p2c_scores = key_layer.matmul(&pos_query.transpose(-1, -2)?)?; // [B*Nh, S, 2P]
        let p2c_pos = relative_pos
            .mul_scalar(-1.0)?
            .add_scalar(att_span as f64)?
            .clamp(0.0, (two_span - 1) as f64)?
            .to_dtype(DType::Int64)?;
        let p2c_idx = p2c_pos.squeeze(0)?.expand(&[bh, s, s])?.contiguous()?;
        let p2c_att = p2c_scores.gather(-1, &p2c_idx)?.transpose(-1, -2)?;

        let scaled_c2p = c2p_att.div_scalar(scale)?;
        let scaled_p2c = p2c_att.div_scalar(scale)?;
        scaled_c2p.add(&scaled_p2c)
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: `[B, S, H]`
    /// - `attention_mask`: additive mask `[B, 1, S, S]`, f32 (0.0 attend,
    ///   `f32::MIN` blocked). The encoder builds this from the flat
    ///   `{0,1}` mask.
    /// - `relative_pos`: `[1, S, S]` int64 bucket indices.
    /// - `rel_embeddings`: `[2*position_buckets, H]` LayerNormed once by
    ///   the encoder, shared across all layers.
    pub fn forward(
        &self,
        hidden_states: &Variable,
        attention_mask: &Variable,
        relative_pos: &Tensor,
        rel_embeddings: &Variable,
    ) -> Result<Variable> {
        let batch = hidden_states.shape()[0];
        let seq = hidden_states.shape()[1];

        let q = self.split_heads(&self.query_proj.forward(hidden_states)?)?;
        let k = self.split_heads(&self.key_proj.forward(hidden_states)?)?;
        let v = self.split_heads(&self.value_proj.forward(hidden_states)?)?;

        // scale_factor = 1 + 1 (c2p) + 1 (p2c) = 3, baked in.
        let scale = ((self.head_dim as f64) * 3.0).sqrt();

        // Base content-to-content scores: Q @ K.T / scale  [B*Nh, S, S]
        let kt = k.transpose(-1, -2)?;
        let c2c = q.matmul(&kt)?.div_scalar(scale)?;

        // Disentangled additive bias (c2p + p2c)
        let bias = self.disentangled_bias(&q, &k, relative_pos, rel_embeddings, scale)?;
        let scores = c2c.add(&bias)?; // [B*Nh, S, S]

        // Reshape to [B, Nh, S, S] for broadcast with [B, 1, S, S] mask
        let scores = scores.reshape(&[batch, self.num_heads, seq, seq])?;
        let scores = scores.add(attention_mask)?;

        let probs = scores.softmax(-1)?;
        let probs = self.attn_dropout.forward(&probs)?;
        let probs = probs.reshape(&[batch * self.num_heads, seq, seq])?;

        let context = probs.matmul(&v)?; // [B*Nh, S, D]
        self.merge_heads(&context, batch)
    }

    /// Emit parameter keys for this attention sub-module. The caller
    /// prefixes these with `attention.self.` to match HF's state_dict.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("query_proj", self.query_proj.parameters()));
        out.extend(prefix_params("key_proj", self.key_proj.parameters()));
        out.extend(prefix_params("value_proj", self.value_proj.parameters()));
        out
    }

    pub fn set_training(&self, training: bool) {
        self.training.set(training);
        self.attn_dropout.set_training(training);
        self.pos_dropout.set_training(training);
    }
}

// ─── SelfOutput / Intermediate / Output ───────────────────────────────

/// Post-attention residual stage: `dense → dropout → LN(x + residual)`.
///
/// Matches HF's `DebertaV2SelfOutput` exactly.
pub struct DebertaV2SelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DebertaV2SelfOutput {
    pub fn on_device(config: &DebertaV2LayerConfig, device: Device) -> Result<Self> {
        Ok(DebertaV2SelfOutput {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, hidden: &Variable, residual: &Variable) -> Result<Variable> {
        let x = self.dense.forward(hidden)?;
        let x = self.dropout.forward(&x)?;
        self.layer_norm.forward(&x.add(residual)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("dense", self.dense.parameters());
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

/// FFN up-projection: `dense(H, 4H) → GELU`.
pub struct DebertaV2Intermediate {
    dense: Linear,
    activation: GELU,
}

impl DebertaV2Intermediate {
    pub fn on_device(config: &DebertaV2LayerConfig, device: Device) -> Result<Self> {
        Ok(DebertaV2Intermediate {
            dense: Linear::on_device(config.hidden_size, config.intermediate_size, device)?,
            activation: GELU::with_approximate(config.hidden_act),
        })
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.dense.forward(input)?;
        self.activation.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }
}

/// FFN down-projection + post-residual LN: `dense(4H, H) → dropout →
/// LN(x + residual)`. Matches HF's `DebertaV2Output`.
pub struct DebertaV2Output {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DebertaV2Output {
    pub fn on_device(config: &DebertaV2LayerConfig, device: Device) -> Result<Self> {
        Ok(DebertaV2Output {
            dense: Linear::on_device(config.intermediate_size, config.hidden_size, device)?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, input: &Variable, residual: &Variable) -> Result<Variable> {
        let x = self.dense.forward(input)?;
        let x = self.dropout.forward(&x)?;
        self.layer_norm.forward(&x.add(residual)?)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("dense", self.dense.parameters());
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ─── DebertaV2TransformerLayer ───────────────────────────────────────────

/// One DeBERTa-v2 transformer block: disentangled self-attention →
/// residual + LN → GELU FFN → residual + LN.
///
/// Unlike the cross-family
/// [`crate::models::transformer_layer::TransformerLayer`], this is
/// *not* a [`Module`] — its `forward` takes four inputs (hidden, mask,
/// relative position grid, rel_embeddings) which doesn't fit the
/// single-input trait signature. The encoder calls it directly as part
/// of its own [`flodl::nn::NamedInputModule::forward_named`]
/// implementation.
pub struct DebertaV2TransformerLayer {
    attention_self: DisentangledSelfAttention,
    attention_output: DebertaV2SelfOutput,
    intermediate: DebertaV2Intermediate,
    output: DebertaV2Output,
}

impl DebertaV2TransformerLayer {
    pub fn on_device(config: &DebertaV2LayerConfig, device: Device) -> Result<Self> {
        Ok(DebertaV2TransformerLayer {
            attention_self: DisentangledSelfAttention::on_device(config, device)?,
            attention_output: DebertaV2SelfOutput::on_device(config, device)?,
            intermediate: DebertaV2Intermediate::on_device(config, device)?,
            output: DebertaV2Output::on_device(config, device)?,
        })
    }

    /// Run one layer. See
    /// [`DisentangledSelfAttention::forward`](DisentangledSelfAttention::forward)
    /// for the signature of the attention block.
    pub fn forward(
        &self,
        hidden_states: &Variable,
        attention_mask: &Variable,
        relative_pos: &Tensor,
        rel_embeddings: &Variable,
    ) -> Result<Variable> {
        let attn = self.attention_self.forward(
            hidden_states,
            attention_mask,
            relative_pos,
            rel_embeddings,
        )?;
        let attn_out = self.attention_output.forward(&attn, hidden_states)?;
        let ffn_mid = self.intermediate.forward(&attn_out)?;
        self.output.forward(&ffn_mid, &attn_out)
    }

    /// Parameters in HF state_dict order, rooted at
    /// `attention.self.*` / `attention.output.*` / `intermediate.*` /
    /// `output.*`. Caller (encoder) prefixes these with `layer.{i}.`.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params(
            "attention.self",
            self.attention_self.parameters(),
        ));
        out.extend(prefix_params(
            "attention.output",
            self.attention_output.parameters(),
        ));
        out.extend(prefix_params(
            "intermediate",
            self.intermediate.parameters(),
        ));
        out.extend(prefix_params("output", self.output.parameters()));
        out
    }

    pub fn set_training(&self, training: bool) {
        self.attention_self.set_training(training);
        self.attention_output.set_training(training);
        self.output.set_training(training);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_config() -> DebertaV2LayerConfig {
        DebertaV2LayerConfig {
            hidden_size: 16,
            num_attention_heads: 4,
            intermediate_size: 32,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            layer_norm_eps: 1e-7,
            position_buckets: 4,
            max_relative_positions: 8,
            hidden_act: GeluApprox::Exact,
        }
    }

    /// Small bucket_size round-trip: for `bucket_size=4`, mid=2, so
    /// positions in (-2, 2) — i.e. {-1, 0, 1} — should pass through
    /// unchanged; positions outside get compressed.
    #[test]
    fn log_bucket_near_range_passthrough() {
        let dev = Device::CPU;
        let raw = Tensor::from_i64(&[-1, 0, 1], &[3], dev).unwrap();
        let bucketed = make_log_bucket_position(&raw, 4, 8).unwrap();
        let out = bucketed.to_i64_vec().unwrap();
        assert_eq!(out, vec![-1, 0, 1], "near-range values must pass through");
    }

    /// build_relative_position output shape + anti-symmetry: rel[q, k] = -rel[k, q].
    #[test]
    fn build_relative_position_antisymmetric() {
        let dev = Device::CPU;
        let rel = build_relative_position(3, 4, 8, dev).unwrap();
        assert_eq!(rel.shape(), vec![1, 3, 3]);
        let data = rel.to_i64_vec().unwrap();
        // Row-major [1, 3, 3]: rel[0, q, k]
        let at = |q: usize, k: usize| data[q * 3 + k];
        for q in 0..3 {
            for k in 0..3 {
                // Near-range passthrough (bucket_size=4, mid=2, so all
                // values in {-2, -1, 0, 1, 2} are in range — diff is
                // at most 2 for 3x3, so all pass through).
                assert_eq!(at(q, k), q as i64 - k as i64, "rel[{q}, {k}]");
            }
        }
    }

    /// Far-range positions get log-compressed into `[mid, 2*mid - 1]`.
    /// For bucket_size=4 → mid=2, so far-positive maps to {2, 3}.
    #[test]
    fn log_bucket_far_range_compressed() {
        let dev = Device::CPU;
        // Raw positions 2..7 — all >= mid=2 → compressed.
        let raw = Tensor::from_i64(&[2, 3, 4, 5, 6, 7], &[6], dev).unwrap();
        let bucketed = make_log_bucket_position(&raw, 4, 8).unwrap();
        let out = bucketed.to_i64_vec().unwrap();
        // rp=2 is at the boundary: |rp| <= mid → passes through as 2.
        assert_eq!(out[0], 2);
        // All others must be in [mid, 2*mid-1] = [2, 3] for this tiny config.
        for (i, &v) in out[1..].iter().enumerate() {
            assert!(
                (2..=3).contains(&v),
                "out[{}] = {} not in [2, 3] for bucket_size=4",
                i + 1,
                v,
            );
        }
    }

    /// A full transformer layer emits the expected 16 params with HF key suffixes.
    #[test]
    fn transformer_layer_param_keys() {
        let layer = DebertaV2TransformerLayer::on_device(&mini_config(), Device::CPU).unwrap();
        let names: Vec<String> = layer.parameters().into_iter().map(|p| p.name).collect();
        let expected = [
            "attention.self.query_proj.weight",
            "attention.self.query_proj.bias",
            "attention.self.key_proj.weight",
            "attention.self.key_proj.bias",
            "attention.self.value_proj.weight",
            "attention.self.value_proj.bias",
            "attention.output.dense.weight",
            "attention.output.dense.bias",
            "attention.output.LayerNorm.weight",
            "attention.output.LayerNorm.bias",
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            "output.dense.weight",
            "output.dense.bias",
            "output.LayerNorm.weight",
            "output.LayerNorm.bias",
        ];
        assert_eq!(names.len(), expected.len(), "got {names:?}");
        for key in expected {
            assert!(names.iter().any(|n| n == key), "missing {key} in {names:?}");
        }
    }

    /// End-to-end forward shape smoke: one layer on random inputs with
    /// all-attend mask and a precomputed rel_embeddings table.
    #[test]
    fn transformer_layer_forward_shape() {
        let cfg = mini_config();
        let dev = Device::CPU;
        let layer = DebertaV2TransformerLayer::on_device(&cfg, dev).unwrap();
        layer.set_training(false);

        let batch = 1;
        let seq = 3;
        let hidden = cfg.hidden_size;

        let hidden_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let x = Variable::new(
            Tensor::from_f32(&hidden_data, &[batch, seq, hidden], dev).unwrap(),
            false,
        );
        // All-attend additive mask: zeros, shape [B, 1, S, S].
        let mask = Variable::new(
            Tensor::zeros(
                &[batch, 1, seq, seq],
                TensorOptions {
                    dtype: DType::Float32,
                    device: dev,
                },
            )
            .unwrap(),
            false,
        );
        // rel_pos grid
        let rel_pos =
            build_relative_position(seq, cfg.position_buckets, cfg.max_relative_positions, dev)
                .unwrap();
        // rel_embeddings: [2*P, H] random (stand-in for encoder-owned table).
        let rel_emb_shape = [cfg.position_buckets * 2, hidden];
        let rel_emb_data: Vec<f32> = (0..(rel_emb_shape[0] * rel_emb_shape[1]) as usize)
            .map(|i| ((i as f32) * 0.003).sin())
            .collect();
        let rel_emb = Variable::new(
            Tensor::from_f32(&rel_emb_data, &rel_emb_shape, dev).unwrap(),
            false,
        );

        let out = layer.forward(&x, &mask, &rel_pos, &rel_emb).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, hidden]);
    }

    /// Regression guard: hidden_size must divide cleanly by
    /// num_attention_heads — a misconfigured layer config is a setup
    /// error, not a runtime condition.
    #[test]
    #[should_panic(expected = "must be divisible")]
    fn hidden_size_must_divide_num_heads() {
        let mut cfg = mini_config();
        cfg.num_attention_heads = 3; // 16 % 3 != 0
        let _ = DisentangledSelfAttention::on_device(&cfg, Device::CPU);
    }

    /// Sanity: changing an attention-mask entry must change the output,
    /// otherwise the mask is being silently dropped.
    #[test]
    fn attention_mask_is_applied() {
        let cfg = mini_config();
        let dev = Device::CPU;
        let layer = DebertaV2TransformerLayer::on_device(&cfg, dev).unwrap();
        layer.set_training(false);

        let batch = 1;
        let seq = 4;
        let hidden = cfg.hidden_size;
        let hidden_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let x = Variable::new(
            Tensor::from_f32(&hidden_data, &[batch, seq, hidden], dev).unwrap(),
            false,
        );
        let rel_pos =
            build_relative_position(seq, cfg.position_buckets, cfg.max_relative_positions, dev)
                .unwrap();
        let rel_emb_data: Vec<f32> = (0..((cfg.position_buckets * 2) * hidden) as usize)
            .map(|i| ((i as f32) * 0.003).sin())
            .collect();
        let rel_emb = Variable::new(
            Tensor::from_f32(&rel_emb_data, &[cfg.position_buckets * 2, hidden], dev).unwrap(),
            false,
        );

        let all_attend = Variable::new(
            Tensor::zeros(
                &[batch, 1, seq, seq],
                TensorOptions {
                    dtype: DType::Float32,
                    device: dev,
                },
            )
            .unwrap(),
            false,
        );
        // Mask out key position 3 from query position 0 (heavy -inf).
        let mut mask_data = vec![0.0_f32; (batch * seq * seq) as usize];
        mask_data[3] = -1e4; // index [0, 0, 0, 3]
        let partial = Variable::new(
            Tensor::from_f32(&mask_data, &[batch, 1, seq, seq], dev).unwrap(),
            false,
        );

        let out_all = layer.forward(&x, &all_attend, &rel_pos, &rel_emb).unwrap();
        let out_mask = layer.forward(&x, &partial, &rel_pos, &rel_emb).unwrap();
        let a: Vec<f32> = out_all.data().to_f32_vec().unwrap();
        let b: Vec<f32> = out_mask.data().to_f32_vec().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-5,
            "masking one key position must change the output; got max_diff = {max_diff}",
        );
    }
}
