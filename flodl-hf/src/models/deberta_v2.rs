//! DeBERTa-v2 / DeBERTa-v3 encoder, compatible with HuggingFace
//! `microsoft/deberta-v3-base` and related checkpoints.
//!
//! DeBERTa-v3 is shipped under the `deberta-v2` architecture name in
//! HuggingFace transformers (the `v3` distinction is a config knob, not
//! a separate class). This port follows the v3-base convention and
//! hard-requires the exact config values that checkpoint ships with.
//! Deviations surface as loud parse-time errors from
//! [`DebertaV2Config::from_json_str`].
//!
//! # Architecture highlights
//!
//! DeBERTa-v2 differs from the BERT/RoBERTa/DistilBERT family in three
//! load-bearing ways:
//!
//! 1. **Disentangled self-attention.** Each layer computes
//!    content-to-content + content-to-position + position-to-content
//!    scores, scaled by `sqrt(head_dim * 3)`. See
//!    [`crate::models::deberta_transformer_layer`].
//!
//! 2. **No absolute positional embedding.** `position_biased_input = false`
//!    for every v3 checkpoint. The embeddings module is token +
//!    (optional) token-type only; position information is carried by
//!    the encoder's `rel_embeddings` table and threaded into every
//!    layer as a disentangled bias.
//!
//! 3. **Mask-gated embeddings.** Post-LayerNorm, the embedding output
//!    is multiplied element-wise by the padding mask — masked positions
//!    are zeroed before entering the encoder. BERT doesn't do this.
//!
//! # Supported configurations
//!
//! This port pins:
//!
//! - `model_type: deberta-v2`
//! - `share_att_key: true`
//! - `pos_att_type` contains both `c2p` and `p2c`
//! - `relative_attention: true`
//! - `position_biased_input: false`
//! - `norm_rel_ebd: "layer_norm"`
//! - `type_vocab_size: 0` (no token-type embeddings)
//! - `legacy: false` (non-legacy MLM head variant)
//! - no `conv_kernel_size` / `embedding_size != hidden_size`
//!
//! Matches `microsoft/deberta-v3-base`, `microsoft/deberta-v3-large`,
//! `microsoft/deberta-v3-small`, and `microsoft/deberta-v3-xsmall`.
//! DeBERTa-v1 and other variants are rejected at config-parse time
//! with a specific message identifying the unsupported knob.
//!
//! # Task heads
//!
//! - [`DebertaV2ForSequenceClassification`] — `ContextPooler` + dropout
//!   + dense; matches HF Python exactly.
//! - [`DebertaV2ForTokenClassification`] — dropout + dense, no pooler;
//!   NER / POS / any per-token tagger.
//! - [`DebertaV2ForQuestionAnswering`] — 2-wide `qa_outputs` dense;
//!   extractive SQuAD-style.
//! - [`DebertaV2ForMaskedLM`] — non-legacy head: `dense(H, H) → GELU →
//!   LN → h @ word_emb.T + bias`. The decoder projection is tied to
//!   the word-embedding table via [`flodl::nn::Linear::from_shared_weight`].

use std::cell::Cell;
use std::collections::HashMap;

use flodl::nn::{
    Dropout, Embedding, GeluApprox, LayerNorm, Linear, Module, NamedInputModule, Parameter, GELU,
};
use flodl::{
    DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, TensorOptions, Variable,
};

use crate::models::deberta_transformer_layer::{
    build_relative_position, DebertaV2LayerConfig, DebertaV2TransformerLayer,
};
use crate::path::prefix_params;

// ─── Config ─────────────────────────────────────────────────────────────

/// DeBERTa-v2 / DeBERTa-v3 hyperparameters.
///
/// Use [`DebertaV2Config::deberta_v3_base`] for the standard 12-layer /
/// 768-hidden preset. [`DebertaV2Config::from_json_str`] parses a Hub
/// `config.json` and rejects any value that would require building a
/// variant this port doesn't support.
#[derive(Debug, Clone)]
pub struct DebertaV2Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub pad_token_id: Option<i64>,
    /// Log-bucket count for relative positions (256 for v3-base).
    pub position_buckets: i64,
    /// Relative-position range — derived from `max_position_embeddings`
    /// when the config field is negative (the v3 convention). Always
    /// concrete in this struct.
    pub max_relative_positions: i64,
    /// Encoder + LM-head activation form (parsed from HF `hidden_act`).
    /// Default `GeluApprox::Exact` (erf form) matches `microsoft/deberta-v3-base`.
    pub hidden_act: GeluApprox,
    /// `ContextPooler` activation form (parsed from HF
    /// `pooler_hidden_act`). Default `GeluApprox::Exact` (erf form)
    /// matches the v3-base preset.
    pub pooler_hidden_act: GeluApprox,
    /// See [`crate::models::bert::BertConfig::num_labels`].
    pub num_labels: Option<i64>,
    /// See [`crate::models::bert::BertConfig::id2label`].
    pub id2label: Option<Vec<String>>,
    /// See [`crate::models::bert::BertConfig::architectures`].
    pub architectures: Option<Vec<String>>,
}

impl DebertaV2Config {
    /// Preset matching `microsoft/deberta-v3-base` on the HuggingFace Hub.
    pub fn deberta_v3_base() -> Self {
        DebertaV2Config {
            vocab_size: 128_100,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-7,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            pad_token_id: Some(0),
            position_buckets: 256,
            max_relative_positions: 512,
            hidden_act: GeluApprox::Exact,
            pooler_hidden_act: GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Parse a Hub `config.json` string.
    ///
    /// Rejects configs with any unsupported knob (see module-level docs
    /// for the supported set). Error messages name the offending field
    /// so a user targeting a DeBERTa-v1 or experimental fine-tune knows
    /// exactly what this port doesn't cover.
    pub fn from_json_str(s: &str) -> Result<Self> {
        use crate::config_json::{
            optional_bool, optional_f64, optional_hidden_act, optional_i64, optional_i64_or_none,
            parse_architectures, parse_id2label, parse_num_labels, required_i64,
        };
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;

        // Required: relative_attention = true. The whole arch depends on it.
        let relative_attention = optional_bool(&v, "relative_attention", false);
        if !relative_attention {
            return Err(TensorError::new(
                "DebertaV2Config: relative_attention = false is not supported — \
                 every published DeBERTa-v2/v3 checkpoint sets relative_attention=true.",
            ));
        }

        // Required: share_att_key = true (layer uses query_proj / key_proj for rel-pos too).
        let share_att_key = optional_bool(&v, "share_att_key", false);
        if !share_att_key {
            return Err(TensorError::new(
                "DebertaV2Config: share_att_key = false is not supported. \
                 flodl-hf requires share_att_key=true (the v3 convention — \
                 separate pos_query_proj / pos_key_proj weights are unsupported).",
            ));
        }

        // Required: position_biased_input = false (v3 convention; v1 had true).
        let position_biased_input = optional_bool(&v, "position_biased_input", false);
        if position_biased_input {
            return Err(TensorError::new(
                "DebertaV2Config: position_biased_input = true is not supported. \
                 flodl-hf covers v3-style checkpoints (position_biased_input=false). \
                 DeBERTa-v1 support is planned for a future release.",
            ));
        }

        // Required: pos_att_type contains both c2p and p2c. HF ships
        // this either as a pipe-separated string (`"p2c|c2p"`, the v3
        // base convention used by `microsoft/deberta-v3-base`) or as a
        // JSON array (`["p2c", "c2p"]`, what `transformers` re-emits
        // when re-saving fine-tuned heads — see e.g.
        // `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`,
        // `deepset/deberta-v3-base-squad2`).
        let pos_att_raw = match v.get("pos_att_type") {
            Some(s) if s.is_string() => s.as_str().unwrap_or("").to_string(),
            Some(arr) if arr.is_array() => arr
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|e| e.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            _ => String::new(),
        };
        let has_c2p = pos_att_raw.contains("c2p");
        let has_p2c = pos_att_raw.contains("p2c");
        if !(has_c2p && has_p2c) {
            return Err(TensorError::new(&format!(
                "DebertaV2Config: pos_att_type = {pos_att_raw:?} must contain both \"c2p\" \
                 and \"p2c\" — flodl-hf hard-codes scale_factor=3 for this path. \
                 Reduced-attention variants are planned for a future release.",
            )));
        }

        // Required: norm_rel_ebd = "layer_norm".
        let norm_raw = v
            .get("norm_rel_ebd")
            .and_then(|x| x.as_str())
            .unwrap_or("none");
        if !norm_raw.split('|').any(|p| p.trim() == "layer_norm") {
            return Err(TensorError::new(&format!(
                "DebertaV2Config: norm_rel_ebd = {norm_raw:?} not supported — \
                 flodl-hf requires norm_rel_ebd=\"layer_norm\" (the v3 convention).",
            )));
        }

        // Required: legacy = false (non-legacy MLM head).
        let legacy = optional_bool(&v, "legacy", false);
        if legacy {
            return Err(TensorError::new(
                "DebertaV2Config: legacy = true is not supported. \
                 flodl-hf uses the non-legacy MLM head (dense(H, H) + GELU + LN + \
                 tied-decoder matmul). Legacy-head checkpoints are planned for a \
                 future release.",
            ));
        }

        // Required: no ConvLayer.
        let conv_kernel = optional_i64(&v, "conv_kernel_size", 0);
        if conv_kernel > 0 {
            return Err(TensorError::new(&format!(
                "DebertaV2Config: conv_kernel_size = {conv_kernel} not supported — \
                 DeBERTa-v2 ConvLayer is not implemented. Most published v3 checkpoints \
                 don't use it.",
            )));
        }

        // Required: no token types (v3 uses type_vocab_size=0).
        let type_vocab_size = optional_i64(&v, "type_vocab_size", 0);
        if type_vocab_size != 0 {
            return Err(TensorError::new(&format!(
                "DebertaV2Config: type_vocab_size = {type_vocab_size} not supported — \
                 flodl-hf covers v3 (type_vocab_size=0). DeBERTa-v1 / legacy variants \
                 with segment embeddings are planned for a future release.",
            )));
        }

        // Required: no factorised embedding projection.
        let embedding_size = optional_i64(&v, "embedding_size", -1);
        let hidden_size = required_i64(&v, "hidden_size")?;
        if embedding_size > 0 && embedding_size != hidden_size {
            return Err(TensorError::new(&format!(
                "DebertaV2Config: embedding_size = {embedding_size} must equal \
                 hidden_size = {hidden_size} — flodl-hf does not yet support the \
                 factorised-embedding `embed_proj` path for DeBERTa-v2.",
            )));
        }

        let id2label = parse_id2label(&v)?;
        let num_labels = parse_num_labels(&v, id2label.as_deref());
        let architectures = parse_architectures(&v);

        // max_relative_positions: -1 means "use max_position_embeddings".
        let max_pos_emb = required_i64(&v, "max_position_embeddings")?;
        let raw_max_rel = optional_i64(&v, "max_relative_positions", -1);
        let max_relative_positions = if raw_max_rel < 1 { max_pos_emb } else { raw_max_rel };

        Ok(DebertaV2Config {
            vocab_size:                   required_i64(&v, "vocab_size")?,
            hidden_size,
            num_hidden_layers:            required_i64(&v, "num_hidden_layers")?,
            num_attention_heads:          required_i64(&v, "num_attention_heads")?,
            intermediate_size:            required_i64(&v, "intermediate_size")?,
            max_position_embeddings:      max_pos_emb,
            layer_norm_eps:               optional_f64(&v, "layer_norm_eps", 1e-7),
            hidden_dropout_prob:          optional_f64(&v, "hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob: optional_f64(&v, "attention_probs_dropout_prob", 0.1),
            pad_token_id:                 optional_i64_or_none(&v, "pad_token_id"),
            position_buckets:             optional_i64(&v, "position_buckets", -1),
            max_relative_positions,
            hidden_act:                   optional_hidden_act(&v, "hidden_act", "gelu")?,
            pooler_hidden_act:            optional_hidden_act(&v, "pooler_hidden_act", "gelu")?,
            num_labels,
            id2label,
            architectures,
        })
    }

    /// Replace the `architectures` field with `[arch_class]` and return
    /// `self`. See [`crate::models::bert::BertConfig::with_architectures`]
    /// for rationale — every family shares the same fix.
    pub fn with_architectures(mut self, arch_class: &str) -> Self {
        self.architectures = Some(vec![arch_class.to_string()]);
        self
    }

    /// Serialize to a HuggingFace-style `config.json` string.
    ///
    /// Inverse of [`Self::from_json_str`]. Emits the full set of knobs
    /// the parser validates: `relative_attention: true`,
    /// `share_att_key: true`, `position_biased_input: false`,
    /// `pos_att_type: "p2c|c2p"`, `norm_rel_ebd: "layer_norm"`,
    /// `legacy: false`, `type_vocab_size: 0`. Model type is emitted as
    /// `"deberta-v2"` (matching HF — v3 checkpoints use the same
    /// `model_type` as v2 under the hood) with
    /// `architectures: ["DebertaV2Model"]`.
    pub fn to_json_str(&self) -> String {
        use crate::config_json::{emit_architectures, emit_hidden_act, emit_id2label};
        let mut m = serde_json::Map::new();
        m.insert("model_type".into(), "deberta-v2".into());
        m.insert(
            "architectures".into(),
            emit_architectures(self.architectures.as_deref(), "DebertaV2Model"),
        );
        m.insert("vocab_size".into(), self.vocab_size.into());
        m.insert("hidden_size".into(), self.hidden_size.into());
        m.insert("num_hidden_layers".into(), self.num_hidden_layers.into());
        m.insert("num_attention_heads".into(), self.num_attention_heads.into());
        m.insert("intermediate_size".into(), self.intermediate_size.into());
        m.insert(
            "max_position_embeddings".into(),
            self.max_position_embeddings.into(),
        );
        m.insert("type_vocab_size".into(), 0i64.into());
        if let Some(pad) = self.pad_token_id {
            m.insert("pad_token_id".into(), pad.into());
        }
        m.insert("layer_norm_eps".into(), self.layer_norm_eps.into());
        m.insert("hidden_dropout_prob".into(), self.hidden_dropout_prob.into());
        m.insert(
            "attention_probs_dropout_prob".into(),
            self.attention_probs_dropout_prob.into(),
        );
        m.insert("position_buckets".into(), self.position_buckets.into());
        m.insert(
            "max_relative_positions".into(),
            self.max_relative_positions.into(),
        );
        m.insert("hidden_act".into(), emit_hidden_act(self.hidden_act).into());
        m.insert(
            "pooler_hidden_act".into(),
            emit_hidden_act(self.pooler_hidden_act).into(),
        );
        // Architecture invariants the parser validates — every published
        // v3 checkpoint sets these exact values.
        m.insert("relative_attention".into(), true.into());
        m.insert("share_att_key".into(), true.into());
        m.insert("position_biased_input".into(), false.into());
        m.insert("pos_att_type".into(), "p2c|c2p".into());
        m.insert("norm_rel_ebd".into(), "layer_norm".into());
        m.insert("legacy".into(), false.into());
        emit_id2label(&mut m, self.id2label.as_deref());
        if let Some(n) = self.num_labels {
            m.insert("num_labels".into(), n.into());
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(m))
            .expect("serde_json::Map serialization is infallible")
    }

    fn layer_config(&self) -> DebertaV2LayerConfig {
        // position_buckets defaults to max_relative_positions when
        // the config field is negative (mirrors HF's pos_ebd_size logic).
        let buckets = if self.position_buckets > 0 {
            self.position_buckets
        } else {
            self.max_relative_positions
        };
        DebertaV2LayerConfig {
            hidden_size:                  self.hidden_size,
            num_attention_heads:          self.num_attention_heads,
            intermediate_size:            self.intermediate_size,
            hidden_dropout_prob:          self.hidden_dropout_prob,
            attention_probs_dropout_prob: self.attention_probs_dropout_prob,
            layer_norm_eps:               self.layer_norm_eps,
            position_buckets:             buckets,
            max_relative_positions:       self.max_relative_positions,
            hidden_act:                   self.hidden_act,
        }
    }
}

// ─── Mask helper ────────────────────────────────────────────────────────

/// Build the DeBERTa-v2 attention mask `[B, 1, S, S]` from a flat
/// `[B, S]` mask.
///
/// DeBERTa masks both query and key dimensions (HF's `get_attention_mask`),
/// unlike BERT which only masks keys. Output is an additive bias of
/// dtype `target_dtype`: `0.0` where attention is permitted, the
/// dtype's most-negative finite value where blocked. The dtype is
/// chosen to match the attention scores it'll be added to — F32 for
/// f32 weights, F16 / BF16 for half-precision weights — so the
/// addition doesn't trip libtorch's same-dtype matmul check.
pub fn build_deberta_attention_mask(flat_mask: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let shape = flat_mask.shape();
    assert_eq!(shape.len(), 2, "flat_mask must be [B, S], got shape {shape:?}");
    let batch = shape[0];
    let seq = shape[1];

    let flat = flat_mask.to_dtype(target_dtype)?;
    // [B, 1, 1, S] and [B, 1, S, 1] then outer-product via multiplication.
    let k_mask = flat.reshape(&[batch, 1, 1, seq])?; // attending-to-key
    let q_mask = flat.reshape(&[batch, 1, seq, 1])?; // attending-from-query
    let grid = q_mask.mul(&k_mask)?; // [B, 1, S, S]  1 where both valid
    // Convert {0, 1} to {-large, 0}. Use the dtype's most-negative
    // finite value (matches HF's `torch.finfo(dtype).min`) — using
    // f32::MIN here would overflow f16's ±65504 range and libtorch
    // would refuse the masked_fill cast.
    let zero_base = Tensor::zeros(&[batch, 1, seq, seq], TensorOptions {
        dtype: target_dtype, device: flat.device(),
    })?;
    let zero_positions = grid.eq_scalar(0.0)?;
    zero_base.masked_fill(&zero_positions, dtype_min_finite(target_dtype))
}

/// Most-negative finite value representable in `dtype`, returned as
/// f64 for consumption by libtorch scalar ops. Mirrors HF Python's
/// `torch.finfo(dtype).min`. Used to populate attention mask bias
/// without overflowing the target dtype's range.
fn dtype_min_finite(dtype: DType) -> f64 {
    match dtype {
        DType::Float32 => f32::MIN as f64,
        // bf16 shares f32's exponent range; finite min is essentially the same.
        DType::BFloat16 => f32::MIN as f64,
        // f16 finite range is +/- 65504.
        DType::Float16 => -65504.0,
        // f64 finfo.min ~= -1.8e308.
        DType::Float64 => f64::MIN,
        // Integer dtypes have no use for this, but fall back to f32 min.
        DType::Int32 | DType::Int64 => f32::MIN as f64,
    }
}

// ─── DebertaV2Embeddings ─────────────────────────────────────────────────

/// Token embeddings + LayerNorm + dropout, with mask-gating.
///
/// For v3-base: `word_embeddings` only (no position, no token-type) →
/// LayerNorm → multiply by `mask.unsqueeze(-1)` to zero masked
/// positions → Dropout. Mirrors HF Python's `DebertaV2Embeddings` on
/// the `position_biased_input=false, type_vocab_size=0` path.
pub struct DebertaV2Embeddings {
    word_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DebertaV2Embeddings {
    pub fn on_device(config: &DebertaV2Config, device: Device) -> Result<Self> {
        Ok(DebertaV2Embeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                device,
            )?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    /// Tied-decoder handle. Matches
    /// [`BertEmbeddings::word_embeddings_weight`](crate::models::bert::BertEmbeddings::word_embeddings_weight).
    pub fn word_embeddings_weight(&self) -> Parameter {
        self.word_embeddings.weight.clone()
    }
}

impl Module for DebertaV2Embeddings {
    fn name(&self) -> &str { "deberta_v2_embeddings" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let emb = self.word_embeddings.forward(input)?;
        let ln = self.layer_norm.forward(&emb)?;
        self.dropout.forward(&ln)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("word_embeddings", self.word_embeddings.parameters());
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

impl NamedInputModule for DebertaV2Embeddings {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let emb = self.word_embeddings.forward(input)?;
        let ln = self.layer_norm.forward(&emb)?;

        // Mask-gate: multiply post-LN embeddings by [B, S, 1] mask so
        // padding positions are zeroed before entering the encoder.
        // The flat mask arrives via refs in whatever dtype the caller
        // chose (i64 from a tokenizer, f32 from a manual build); cast
        // to match the LayerNorm's output so the multiply doesn't trip
        // libtorch's same-dtype check when weights are f16 / bf16.
        let gated = if let Some(mask_var) = refs.get("attention_mask") {
            let target_dtype = ln.data().dtype();
            let mask_data = mask_var.data();
            let mask_cast = if mask_data.dtype() == target_dtype {
                mask_data
            } else {
                mask_data.to_dtype(target_dtype)?
            };
            let shape = mask_cast.shape();
            // Mask from the tokenizer is [B, S]; unsqueeze last dim for
            // broadcast with [B, S, H].
            let mask_unsq = Variable::new(
                mask_cast.reshape(&[shape[0], shape[1], 1])?,
                false,
            );
            ln.mul(&mask_unsq)?
        } else {
            ln
        };
        self.dropout.forward(&gated)
    }
}

// ─── DebertaV2Encoder ────────────────────────────────────────────────────

/// N-layer transformer stack that owns the shared relative-position
/// embedding table and LayerNorm.
///
/// Exposes a single graph-level input (`hidden_states`) and reads the
/// flat `[B, S]` padding mask via the graph's `attention_mask` ref.
/// Internally:
///
/// 1. LayerNorm-normalises the shared `rel_embeddings` weight once,
///    matching HF `get_rel_embedding`.
/// 2. Builds the `[1, S, S]` log-bucketed relative-position grid from
///    the current sequence length.
/// 3. Expands the `[B, S]` mask to the `[B, 1, S, S]` additive form
///    disentangled attention consumes (see
///    [`build_deberta_attention_mask`]).
/// 4. Threads each layer in sequence, handing every layer the same
///    `(attention_mask, relative_pos, rel_embeddings)` triple.
pub struct DebertaV2Encoder {
    layers: Vec<DebertaV2TransformerLayer>,
    rel_embeddings: Embedding,
    layer_norm: LayerNorm,
    position_buckets: i64,
    max_relative_positions: i64,
    training: Cell<bool>,
}

impl DebertaV2Encoder {
    pub fn on_device(config: &DebertaV2Config, device: Device) -> Result<Self> {
        let layer_cfg = config.layer_config();
        let num_layers = config.num_hidden_layers as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DebertaV2TransformerLayer::on_device(&layer_cfg, device)?);
        }

        // rel_embeddings table: [2 * pos_ebd_size, hidden].
        let pos_ebd_size = if config.position_buckets > 0 {
            config.position_buckets
        } else {
            config.max_relative_positions
        };
        let rel_embeddings = Embedding::on_device(
            pos_ebd_size * 2,
            config.hidden_size,
            device,
        )?;
        let layer_norm = LayerNorm::on_device_with_eps(
            config.hidden_size,
            config.layer_norm_eps,
            device,
        )?;

        Ok(DebertaV2Encoder {
            layers,
            rel_embeddings,
            layer_norm,
            position_buckets: layer_cfg.position_buckets,
            max_relative_positions: layer_cfg.max_relative_positions,
            training: Cell::new(true),
        })
    }
}

impl Module for DebertaV2Encoder {
    fn name(&self) -> &str { "deberta_v2_encoder" }

    /// Unmasked forward — used by diagnostics. The graph drives the
    /// masked path via [`NamedInputModule::forward_named`].
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let seq = input.shape()[1];
        let rel_pos = build_relative_position(
            seq, self.position_buckets, self.max_relative_positions, input.device(),
        )?;
        let rel_emb = self.layer_norm.forward(&self.rel_embeddings.weight.variable)?;
        let batch = input.shape()[0];
        let zero_mask = Variable::new(
            Tensor::zeros(
                &[batch, 1, seq, seq],
                TensorOptions { dtype: DType::Float32, device: input.device() },
            )?,
            false,
        );
        let mut h = input.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &zero_mask, &rel_pos, &rel_emb)?;
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("rel_embeddings", self.rel_embeddings.parameters());
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        for (i, layer) in self.layers.iter().enumerate() {
            let tag = format!("layer.{i}");
            out.extend(prefix_params(&tag, layer.parameters()));
        }
        out
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        for layer in &self.layers {
            layer.set_training(training);
        }
    }
}

impl NamedInputModule for DebertaV2Encoder {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let device = input.device();
        let seq = input.shape()[1];

        // Build [B, 1, S, S] additive mask from the flat [B, S] ref.
        let mask = refs.get("attention_mask").ok_or_else(|| {
            TensorError::new(
                "DebertaV2Encoder: missing \"attention_mask\" ref — the graph \
                 must plumb a [B, S] padding mask to the encoder input.",
            )
        })?;
        let flat = mask.data();
        // Match the additive bias dtype to the hidden states so attention
        // arithmetic stays single-dtype (libtorch refuses Float+Half mixes).
        let hidden_dtype = input.data().dtype();
        let extended = Variable::new(build_deberta_attention_mask(&flat, hidden_dtype)?, false);

        // Relative-position grid (deterministic given seq_len, no grad).
        let rel_pos = build_relative_position(
            seq, self.position_buckets, self.max_relative_positions, device,
        )?;

        // LayerNorm-normalise the shared rel_embeddings table once.
        let rel_emb = self.layer_norm.forward(&self.rel_embeddings.weight.variable)?;

        let mut h = input.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &extended, &rel_pos, &rel_emb)?;
        }
        Ok(h)
    }
}

// ─── DebertaV2Model ──────────────────────────────────────────────────────

/// Build the DeBERTa-v2 backbone graph on `device`.
///
/// Graph signature: 2 `forward_multi` inputs in order —
/// `input_ids` (positional, i64 `[B, S]`) and `attention_mask`
/// (f32 or i64 `[B, S]`, flat padding mask). The embeddings and
/// encoder both consume the mask via the graph's ref system.
///
/// Output: `last_hidden_state` `[B, S, hidden_size]`.
fn backbone_flow(config: &DebertaV2Config, device: Device) -> Result<FlowBuilder> {
    let fb = FlowBuilder::new()
        .input(&["attention_mask"])
        .through(DebertaV2Embeddings::on_device(config, device)?)
        .tag("deberta.embeddings")
        .using(&["attention_mask"])
        .through(DebertaV2Encoder::on_device(config, device)?)
        .tag("deberta.encoder")
        .using(&["attention_mask"]);
    Ok(fb)
}

/// Assembled DeBERTa-v2 backbone.
pub struct DebertaV2Model;

impl DebertaV2Model {
    /// Build on CPU.
    pub fn build(config: &DebertaV2Config) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build on `device`. Emits `last_hidden_state` of shape
    /// `[batch, seq_len, hidden]`.
    pub fn on_device(config: &DebertaV2Config, device: Device) -> Result<Graph> {
        backbone_flow(config, device)?.build()
    }
}

// ─── ContextPooler (used only by seq-cls) ────────────────────────────────

/// DeBERTa's `[CLS]`-then-dense-then-GELU pooler. Matches HF
/// `ContextPooler` on the defaults `microsoft/deberta-v3-*` configs
/// use (`pooler_hidden_act="gelu"`).
///
/// Parameter keys (relative to tag `pooler`):
/// - `pooler.dense.{weight,bias}`
pub struct ContextPooler {
    dense: Linear,
    dropout: Dropout,
    activation: GELU,
}

impl ContextPooler {
    pub fn on_device(config: &DebertaV2Config, device: Device) -> Result<Self> {
        Ok(ContextPooler {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            activation: GELU::with_approximate(config.pooler_hidden_act),
        })
    }
}

impl Module for ContextPooler {
    fn name(&self) -> &str { "context_pooler" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let cls = input.select(1, 0)?;          // [B, H]
        let dropped = self.dropout.forward(&cls)?;
        let dense = self.dense.forward(&dropped)?;
        self.activation.forward(&dense)
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ─── Task heads ──────────────────────────────────────────────────────────

use crate::task_heads::{
    check_num_labels, ClassificationHead, MaskedLmHead, QaHead, TaggingHead,
};

/// DeBERTa-v2 graphs take **2** `forward_multi` inputs: `input_ids`
/// and a flat `[B, S]` padding mask. The encoder handles the
/// `[B, 1, S, S]` expansion internally.
#[cfg(feature = "tokenizer")]
impl crate::task_heads::EncoderInputs for DebertaV2Config {
    const FAMILY_NAME: &'static str = "DebertaV2";
    const MASK_TOKEN: &'static str = "[MASK]";

    fn encoder_inputs(enc: &crate::tokenizer::EncodedBatch) -> Result<Vec<Variable>> {
        Ok(vec![enc.input_ids.clone(), enc.attention_mask.clone()])
    }
}

/// DeBERTa-v2 with a sequence-classification head:
/// `backbone → ContextPooler → dropout → Linear(hidden, num_labels)`.
///
/// Parameter keys for the head:
/// - `pooler.dense.{weight,bias}` — `[hidden, hidden]`
/// - `classifier.{weight,bias}`   — `[num_labels, hidden]`
///
/// Matches HF Python's `DebertaV2ForSequenceClassification`. Popular
/// checkpoints: `cross-encoder/nli-deberta-v3-base`,
/// `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.
pub type DebertaV2ForSequenceClassification = ClassificationHead<DebertaV2Config>;

impl ClassificationHead<DebertaV2Config> {
    pub fn on_device(
        config: &DebertaV2Config,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = backbone_flow(config, device)?
            .through(ContextPooler::on_device(config, device)?)
            .tag("pooler")
            .through(Dropout::new(config.hidden_dropout_prob))
            .through(Linear::on_device(config.hidden_size, num_labels, device)?)
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(graph, config, num_labels, config.id2label.clone()))
    }

    pub(crate) fn num_labels_from_config(config: &DebertaV2Config) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "DebertaV2ForSequenceClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// DeBERTa-v2 with a per-token classification head. Matches HF Python's
/// `DebertaV2ForTokenClassification`.
pub type DebertaV2ForTokenClassification = TaggingHead<DebertaV2Config>;

impl TaggingHead<DebertaV2Config> {
    pub fn on_device(
        config: &DebertaV2Config,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = backbone_flow(config, device)?
            .through(Dropout::new(config.hidden_dropout_prob))
            .through(Linear::on_device(config.hidden_size, num_labels, device)?)
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(graph, config, num_labels, config.id2label.clone()))
    }

    pub(crate) fn num_labels_from_config(config: &DebertaV2Config) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "DebertaV2ForTokenClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// DeBERTa-v2 with an extractive QA head: `last_hidden_state → Linear(hidden, 2)`.
pub type DebertaV2ForQuestionAnswering = QaHead<DebertaV2Config>;

impl QaHead<DebertaV2Config> {
    pub fn on_device(config: &DebertaV2Config, device: Device) -> Result<Self> {
        let graph = backbone_flow(config, device)?
            .through(Linear::on_device(config.hidden_size, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self::from_graph(graph, config))
    }
}

// ─── MLM head (non-legacy) ──────────────────────────────────────────────

/// DeBERTa-v2 non-legacy MLM head: `dense(H, H) → GELU → LN → h @
/// word_emb.T + bias`.
///
/// Differs structurally from BERT/RoBERTa/ALBERT:
///
/// - The decoder is a matmul against `word_embeddings.weight.T`, not
///   a Linear with a tied weight plus `embedding_size ↔ hidden_size`
///   factorisation. For v3-base-style configs `embedding_size == hidden_size`
///   so the shapes line up directly.
/// - The decoder bias is a separate `[vocab_size]` parameter; it
///   surfaces under `lm_predictions.lm_head.bias` (not `.decoder.bias`).
///
/// Parameter keys emitted under graph tag `lm_predictions.lm_head`:
/// - `dense.{weight,bias}` — `[hidden, hidden]` + `[hidden]`
/// - `LayerNorm.{weight,bias}` — `[hidden]`
/// - `bias` — `[vocab_size]`, the tied-decoder bias
///
/// Plus the tied `deberta.embeddings.word_embeddings.weight`
/// (`[vocab_size, hidden]`), deduplicated under the embeddings tag.
pub struct DebertaV2LMHead {
    dense: Linear,
    activation: GELU,
    layer_norm: LayerNorm,
    decoder: Linear, // from_shared_weight(word_emb, fresh_bias)
}

impl DebertaV2LMHead {
    fn on_device(
        config: &DebertaV2Config,
        tied_word_embedding: Parameter,
        device: Device,
    ) -> Result<Self> {
        let decoder_bias = Parameter {
            variable: Variable::new(
                Tensor::zeros(
                    &[config.vocab_size],
                    TensorOptions { dtype: DType::Float32, device },
                )?,
                true,
            ),
            name: "bias".into(),
        };
        Ok(DebertaV2LMHead {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            activation: GELU::with_approximate(config.hidden_act),
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
            decoder: Linear::from_shared_weight(tied_word_embedding, Some(decoder_bias)),
        })
    }
}

impl Module for DebertaV2LMHead {
    fn name(&self) -> &str { "deberta_v2_lm_head" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.dense.forward(input)?;
        let x = self.activation.forward(&x)?;
        let x = self.layer_norm.forward(&x)?;
        self.decoder.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("dense", self.dense.parameters());
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        // Decoder: emit only the fresh bias under name "bias" (no
        // "decoder." prefix — HF stores it as `lm_predictions.lm_head.bias`).
        // The weight is tied to `deberta.embeddings.word_embeddings.weight`
        // and the graph's Rc-based dedup will keep exactly one entry
        // under the embeddings tag.
        for p in self.decoder.parameters() {
            if p.name == "bias" {
                out.push(Parameter { variable: p.variable, name: "bias".into() });
            }
            // weight: skip (tied + dedup'd under embeddings tag)
        }
        out
    }
}

/// DeBERTa-v2 MLM. See [`DebertaV2LMHead`] for the head-layout details.
pub type DebertaV2ForMaskedLM = MaskedLmHead<DebertaV2Config>;

impl MaskedLmHead<DebertaV2Config> {
    pub fn on_device(config: &DebertaV2Config, device: Device) -> Result<Self> {
        // Build embeddings first so we can capture the tied weight.
        let embeddings = DebertaV2Embeddings::on_device(config, device)?;
        let tied_weight = embeddings.word_embeddings_weight();

        let mut fb = FlowBuilder::new()
            .input(&["attention_mask"])
            .through(embeddings)
            .tag("deberta.embeddings")
            .using(&["attention_mask"])
            .through(DebertaV2Encoder::on_device(config, device)?)
            .tag("deberta.encoder")
            .using(&["attention_mask"]);

        fb = fb
            .through(DebertaV2LMHead::on_device(config, tied_weight, device)?)
            .tag("lm_predictions.lm_head");

        let graph = fb.build()?;
        Ok(Self::from_graph(graph, config))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "deberta_v2_tests.rs"]
mod tests;
