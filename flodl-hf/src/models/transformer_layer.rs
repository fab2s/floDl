//! Cross-family transformer encoder layer.
//!
//! BERT, RoBERTa, and DistilBERT all use the same mathematical encoder
//! block — multi-head self-attention with residual + post-LN, followed
//! by a two-layer GELU feed-forward with residual + post-LN — but each
//! family ships its own weight-key naming scheme on the Hub. Rather than
//! duplicate the block in three files, [`TransformerLayer`] implements
//! the block once and [`LayerNaming`] carries the per-family key
//! suffixes.
//!
//! # Weight-key layout
//!
//! Each [`LayerNaming`] field is a dotted path *relative to the layer
//! tag* chosen by the caller (typically `{prefix}.encoder.layer.{i}` or
//! `{prefix}.transformer.layer.{i}`). Flat one-segment paths are fine —
//! DistilBERT's `sa_layer_norm` lives directly under the layer root,
//! whereas BERT's `attention.output.LayerNorm` is three segments deep.
//!
//! | Slot              | BERT / RoBERTa                 | DistilBERT             |
//! |-------------------|--------------------------------|------------------------|
//! | `query`           | `attention.self.query`         | `attention.q_lin`      |
//! | `key`             | `attention.self.key`           | `attention.k_lin`      |
//! | `value`           | `attention.self.value`         | `attention.v_lin`      |
//! | `attn_output`     | `attention.output.dense`       | `attention.out_lin`    |
//! | `attn_layer_norm` | `attention.output.LayerNorm`   | `sa_layer_norm`        |
//! | `ffn_up`          | `intermediate.dense`           | `ffn.lin1`             |
//! | `ffn_down`        | `output.dense`                 | `ffn.lin2`             |
//! | `ffn_layer_norm`  | `output.LayerNorm`             | `output_layer_norm`    |
//!
//! # Attention mask convention
//!
//! The layer is mask-agnostic: it forwards whatever additive f32 mask
//! the graph plumbs in under the `"attention_mask"` input, unchanged,
//! as the additive bias to `scaled_dot_product_attention`. Each model
//! file owns the mask-construction helper (e.g.
//! [`build_extended_attention_mask`](crate::models::bert::build_extended_attention_mask))
//! so the value used for masked positions (`-1e4` for BERT,
//! `f32::MIN` for DistilBERT, …) stays faithful to the published
//! implementation.

use std::cell::Cell;
use std::collections::HashMap;

use flodl::nn::{
    Dropout, GELU, GeluApprox, LayerNorm, Linear, Module, NamedInputModule, Parameter,
};
#[cfg(test)]
use flodl::{DType, Tensor, TensorOptions};
use flodl::{Device, Result, Variable, scaled_dot_product_attention};

use crate::path::prefix_params;

/// Weight-key suffixes for one transformer encoder layer.
///
/// Fields are static strings so a family's naming lives as a single
/// `const` (see [`LayerNaming::BERT`] and [`LayerNaming::DISTILBERT`])
/// and constructing a layer costs nothing.
///
/// See the module-level doc for the slot-by-slot mapping table.
#[derive(Debug, Clone, Copy)]
pub struct LayerNaming {
    pub query: &'static str,
    pub key: &'static str,
    pub value: &'static str,
    pub attn_output: &'static str,
    pub attn_layer_norm: &'static str,
    pub ffn_up: &'static str,
    pub ffn_down: &'static str,
    pub ffn_layer_norm: &'static str,
}

impl LayerNaming {
    /// BERT/RoBERTa encoder-layer key suffixes. The two families share
    /// this layout exactly — the family prefix (`bert.*` vs `roberta.*`)
    /// is applied outside the layer via `FlowBuilder::tag(...)`.
    pub const BERT: Self = Self {
        query: "attention.self.query",
        key: "attention.self.key",
        value: "attention.self.value",
        attn_output: "attention.output.dense",
        attn_layer_norm: "attention.output.LayerNorm",
        ffn_up: "intermediate.dense",
        ffn_down: "output.dense",
        ffn_layer_norm: "output.LayerNorm",
    };

    /// DistilBERT encoder-layer key suffixes.
    pub const DISTILBERT: Self = Self {
        query: "attention.q_lin",
        key: "attention.k_lin",
        value: "attention.v_lin",
        attn_output: "attention.out_lin",
        attn_layer_norm: "sa_layer_norm",
        ffn_up: "ffn.lin1",
        ffn_down: "ffn.lin2",
        ffn_layer_norm: "output_layer_norm",
    };

    /// ALBERT encoder-layer key suffixes. Differs from [`Self::BERT`]
    /// in two regular ways: the attention sub-module is flat (no
    /// `attention.self` / `attention.output` split — HF inlines both
    /// into `attention`), and the feed-forward uses `ffn` / `ffn_output`
    /// instead of `intermediate.dense` / `output.dense`.
    ///
    /// Full state_dict keys for one ALBERT inner layer start with the
    /// graph tag `albert.encoder.albert_layer_groups.{G}.albert_layers.{L}`
    /// (almost always `G=0`, `L=0`: every public ALBERT checkpoint
    /// uses `num_hidden_groups=1` and `inner_group_num=1`, sharing one
    /// transformer block across all `num_hidden_layers` applications).
    pub const ALBERT: Self = Self {
        query: "attention.query",
        key: "attention.key",
        value: "attention.value",
        attn_output: "attention.dense",
        attn_layer_norm: "attention.LayerNorm",
        ffn_up: "ffn",
        ffn_down: "ffn_output",
        ffn_layer_norm: "full_layer_layer_norm",
    };
}

/// Hyperparameters consumed by [`TransformerLayer::on_device`]. Bundles
/// the fields every BERT-family config exposes, translated to a common
/// vocabulary so the layer doesn't depend on any one config type.
///
/// Call sites in `bert.rs`, `roberta.rs`, and `distilbert.rs` each have
/// a tiny `From<&XxxConfig>`-shaped adapter.
#[derive(Debug, Clone, Copy)]
pub struct TransformerLayerConfig {
    /// Hidden dimension (BERT `hidden_size`, DistilBERT `dim`).
    pub hidden_size: i64,
    /// Attention heads per layer.
    pub num_attention_heads: i64,
    /// Feed-forward inner dimension (BERT `intermediate_size`,
    /// DistilBERT `hidden_dim`).
    pub intermediate_size: i64,
    /// Dropout on the two residual paths (after `attn_output` and
    /// after `ffn_down`). BERT `hidden_dropout_prob`, DistilBERT
    /// `dropout`.
    pub hidden_dropout_prob: f64,
    /// Dropout inside `scaled_dot_product_attention`. BERT
    /// `attention_probs_dropout_prob`, DistilBERT `attention_dropout`.
    pub attention_probs_dropout_prob: f64,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// FFN activation form. Each family parses this from its config —
    /// `hidden_act` for BERT/RoBERTa/ALBERT/XLM-R/DeBERTa-v2,
    /// `activation` for DistilBERT — via the crate-internal
    /// `optional_hidden_act` JSON helper.
    /// Default [`GeluApprox::Exact`] (erf form) matches the canonical
    /// `google-bert/bert-base-uncased` / `FacebookAI/roberta-base` / `distilbert/distilbert-base-uncased`
    /// checkpoints. ALBERT both v1 and v2 ship `"gelu_new"` and need
    /// [`GeluApprox::Tanh`].
    pub hidden_act: GeluApprox,
}

/// One transformer encoder layer — self-attention → residual → LN →
/// FFN → residual → LN. Post-LN ordering, GELU activation.
///
/// Structurally identical across BERT, RoBERTa, and DistilBERT. The
/// only per-family knob is [`LayerNaming`], which decides what
/// weight-key suffixes [`Self::parameters`] emits.
pub struct TransformerLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_output: Linear,
    attn_layer_norm: LayerNorm,
    attn_out_dropout: Dropout,
    attn_dropout_prob: f64,
    training: Cell<bool>,
    num_heads: i64,
    head_dim: i64,

    ffn_up: Linear,
    activation: GELU,
    ffn_down: Linear,
    ffn_layer_norm: LayerNorm,
    ffn_dropout: Dropout,

    naming: LayerNaming,
}

impl TransformerLayer {
    /// Build a fresh layer on `device`.
    ///
    /// Panics if `hidden_size` is not divisible by `num_attention_heads`
    /// — this is a config-authoring error, not a runtime condition.
    pub fn on_device(
        config: &TransformerLayerConfig,
        naming: LayerNaming,
        device: Device,
    ) -> Result<Self> {
        assert!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            config.hidden_size,
            config.num_attention_heads,
        );
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(TransformerLayer {
            query: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            key: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            value: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            attn_output: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            attn_layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            attn_out_dropout: Dropout::new(config.hidden_dropout_prob),
            attn_dropout_prob: config.attention_probs_dropout_prob,
            training: Cell::new(true),
            num_heads: config.num_attention_heads,
            head_dim,

            ffn_up: Linear::on_device(config.hidden_size, config.intermediate_size, device)?,
            activation: GELU::with_approximate(config.hidden_act),
            ffn_down: Linear::on_device(config.intermediate_size, config.hidden_size, device)?,
            ffn_layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            ffn_dropout: Dropout::new(config.hidden_dropout_prob),

            naming,
        })
    }

    fn forward_impl(
        &self,
        input: &Variable,
        attention_mask: Option<&Variable>,
    ) -> Result<Variable> {
        let shape = input.shape();
        let batch = shape[0];
        let seq = shape[1];

        let q = self
            .query
            .forward(input)?
            .reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = self
            .key
            .forward(input)?
            .reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = self
            .value
            .forward(input)?
            .reshape(&[batch, seq, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        let dropout_p = if self.training.get() {
            self.attn_dropout_prob
        } else {
            0.0
        };
        let mask_data = attention_mask.map(|m| m.data());
        let context = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            mask_data.as_ref(),
            dropout_p,
            /*is_causal=*/ false,
            /*scale=*/ None,
        )?;
        let attn_flat =
            context
                .transpose(1, 2)?
                .reshape(&[batch, seq, self.num_heads * self.head_dim])?;

        let attn_proj = self.attn_output.forward(&attn_flat)?;
        let attn_dropped = self.attn_out_dropout.forward(&attn_proj)?;
        let residual1 = self.attn_layer_norm.forward(&attn_dropped.add(input)?)?;

        let ffn_hidden = self.activation.forward(&self.ffn_up.forward(&residual1)?)?;
        let ffn_out = self.ffn_down.forward(&ffn_hidden)?;
        let ffn_dropped = self.ffn_dropout.forward(&ffn_out)?;
        self.ffn_layer_norm.forward(&ffn_dropped.add(&residual1)?)
    }
}

impl Module for TransformerLayer {
    fn name(&self) -> &str {
        "transformer_layer"
    }

    /// Unmasked forward, used by tests and diagnostics. The graph drives
    /// the masked path via [`NamedInputModule::forward_named`].
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(input, None)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let n = self.naming;
        let mut out = Vec::new();
        out.extend(prefix_params(n.query, self.query.parameters()));
        out.extend(prefix_params(n.key, self.key.parameters()));
        out.extend(prefix_params(n.value, self.value.parameters()));
        out.extend(prefix_params(n.attn_output, self.attn_output.parameters()));
        out.extend(prefix_params(
            n.attn_layer_norm,
            self.attn_layer_norm.parameters(),
        ));
        out.extend(prefix_params(n.ffn_up, self.ffn_up.parameters()));
        out.extend(prefix_params(n.ffn_down, self.ffn_down.parameters()));
        out.extend(prefix_params(
            n.ffn_layer_norm,
            self.ffn_layer_norm.parameters(),
        ));
        out
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.attn_out_dropout.set_training(training);
        self.ffn_dropout.set_training(training);
    }
}

impl NamedInputModule for TransformerLayer {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        self.forward_impl(input, refs.get("attention_mask"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_config() -> TransformerLayerConfig {
        TransformerLayerConfig {
            hidden_size: 8,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            layer_norm_eps: 1e-12,
            hidden_act: GeluApprox::Exact,
        }
    }

    #[test]
    fn bert_naming_emits_bert_suffixes() {
        let layer =
            TransformerLayer::on_device(&mini_config(), LayerNaming::BERT, Device::CPU).unwrap();
        let names: Vec<String> = layer.parameters().into_iter().map(|p| p.name).collect();
        // Spot-check a few — full layout is pinned via the
        // bert_parity.rs integration test.
        assert!(
            names.iter().any(|n| n == "attention.self.query.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "attention.self.query.bias"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "attention.output.dense.weight"),
            "got: {names:?}"
        );
        assert!(
            names
                .iter()
                .any(|n| n == "attention.output.LayerNorm.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "intermediate.dense.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "output.dense.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "output.LayerNorm.weight"),
            "got: {names:?}"
        );
    }

    #[test]
    fn distilbert_naming_emits_distilbert_suffixes() {
        let layer =
            TransformerLayer::on_device(&mini_config(), LayerNaming::DISTILBERT, Device::CPU)
                .unwrap();
        let names: Vec<String> = layer.parameters().into_iter().map(|p| p.name).collect();
        assert!(
            names.iter().any(|n| n == "attention.q_lin.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "attention.k_lin.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "attention.v_lin.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "attention.out_lin.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "sa_layer_norm.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ffn.lin1.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "ffn.lin2.weight"),
            "got: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "output_layer_norm.weight"),
            "got: {names:?}"
        );
    }

    #[test]
    fn parameter_count_identical_across_namings() {
        // Same structure, different suffixes — parameter count must match.
        let bert =
            TransformerLayer::on_device(&mini_config(), LayerNaming::BERT, Device::CPU).unwrap();
        let distil =
            TransformerLayer::on_device(&mini_config(), LayerNaming::DISTILBERT, Device::CPU)
                .unwrap();
        assert_eq!(bert.parameters().len(), distil.parameters().len());
        // 4 Linear × (w+b) + 2 LayerNorm × (w+b) + 2 Linear (FFN) × (w+b) = 16
        assert_eq!(bert.parameters().len(), 16);
    }

    #[test]
    fn forward_runs_end_to_end() {
        let layer =
            TransformerLayer::on_device(&mini_config(), LayerNaming::BERT, Device::CPU).unwrap();
        layer.set_training(false);
        let x = Variable::new(
            Tensor::zeros(
                &[2, 4, 8],
                TensorOptions {
                    dtype: DType::Float32,
                    device: Device::CPU,
                },
            )
            .unwrap(),
            /*requires_grad=*/ false,
        );
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.data().shape(), vec![2, 4, 8]);
    }

    /// Mask threading sanity: a zero-additive mask (all positions attend)
    /// must produce bitwise-identical output to the no-mask path. If the
    /// plumbing were wrong (shape-broadcast mismatch, stale ref lookup),
    /// the two would diverge despite the mask being a semantic no-op.
    #[test]
    fn zero_additive_mask_matches_unmasked() {
        let cfg = mini_config();
        let dev = Device::CPU;
        let layer = TransformerLayer::on_device(&cfg, LayerNaming::BERT, dev).unwrap();
        layer.set_training(false);

        let batch = 1;
        let seq = 3;
        let hidden = cfg.hidden_size;
        let x_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let x = Variable::new(
            Tensor::from_f32(&x_data, &[batch, seq, hidden], dev).unwrap(),
            false,
        );

        let zero_mask = Variable::new(
            Tensor::zeros(
                &[batch, 1, 1, seq],
                TensorOptions {
                    dtype: DType::Float32,
                    device: dev,
                },
            )
            .unwrap(),
            false,
        );
        let mut refs = HashMap::new();
        refs.insert("attention_mask".to_string(), zero_mask);

        let unmasked = layer.forward(&x).unwrap();
        let with_zero = layer.forward_named(&x, &refs).unwrap();

        let a: Vec<f32> = unmasked.data().to_f32_vec().unwrap();
        let b: Vec<f32> = with_zero.data().to_f32_vec().unwrap();
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "diverged at {i}: unmasked={x}, zero-masked={y}",
            );
        }
    }

    /// Masking a position must change encoder output at the remaining
    /// positions. A bitwise match would mean the mask was silently dropped.
    #[test]
    fn padding_mask_changes_output() {
        use crate::models::bert::build_extended_attention_mask;
        let cfg = mini_config();
        let dev = Device::CPU;
        let layer = TransformerLayer::on_device(&cfg, LayerNaming::BERT, dev).unwrap();
        layer.set_training(false);

        let batch = 1;
        let seq = 4;
        let hidden = cfg.hidden_size;
        let x_data: Vec<f32> = (0..(batch * seq * hidden) as usize)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect();
        let x = Variable::new(
            Tensor::from_f32(&x_data, &[batch, seq, hidden], dev).unwrap(),
            false,
        );

        // Attend to positions 0..2, mask position 3.
        let raw = Tensor::from_f32(&[1.0, 1.0, 1.0, 0.0], &[batch, seq], dev).unwrap();
        let additive = build_extended_attention_mask(&raw).unwrap();
        let mut refs = HashMap::new();
        refs.insert("attention_mask".to_string(), Variable::new(additive, false));

        let unmasked = layer.forward(&x).unwrap();
        let masked = layer.forward_named(&x, &refs).unwrap();

        let a: Vec<f32> = unmasked.data().to_f32_vec().unwrap();
        let b: Vec<f32> = masked.data().to_f32_vec().unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "masking a position must change attention output; max_diff={max_diff}",
        );
    }
}
