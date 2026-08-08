//! ALBERT encoder, compatible with HuggingFace `albert/albert-base-v2`
//! checkpoints.
//!
//! ALBERT differs from the BERT/RoBERTa/DistilBERT family in two
//! load-bearing ways that every integration has to handle:
//!
//! 1. **Factorised embeddings.** The token / position / token-type
//!    embeddings live in a smaller `embedding_size` space (128 for
//!    `albert/albert-base-v2`), and a single `embedding_hidden_mapping_in`
//!    linear projection lifts them into the `hidden_size` space the
//!    transformer block expects. The embedding LayerNorm operates in
//!    embedding space, before the projection. The MLM decoder
//!    projects back down to embedding space before the tied-decoder
//!    linear so the decoder can share weight with
//!    `albert.embeddings.word_embeddings.weight`.
//!
//! 2. **Cross-layer parameter sharing.** Every public ALBERT
//!    checkpoint uses `num_hidden_groups=1` and `inner_group_num=1`:
//!    one transformer block, re-applied `num_hidden_layers` times.
//!    [`AlbertLayerStack`] wraps a single
//!    [`crate::models::transformer_layer::TransformerLayer`]
//!    and forwards `num_hidden_layers` times inside a single
//!    [`Module`]. Parameters surface once under the tag
//!    `albert.encoder.albert_layer_groups.0.albert_layers.0` — the
//!    key layout HF's state_dict uses. Configurations with
//!    `num_hidden_groups > 1` or `inner_group_num > 1` are rejected
//!    at `from_json_str` time; flodl-hf can grow that axis once a
//!    checkpoint using it appears in the wild.
//!
//! ALBERT's encoder block itself is mathematically identical to
//! BERT's (self-attention → residual → LayerNorm → FFN → residual →
//! LayerNorm, post-LN, GELU activation), so the shared
//! [`TransformerLayer`] carries the implementation. Only the weight-
//! key suffixes differ, captured by
//! [`LayerNaming::ALBERT`](crate::models::transformer_layer::LayerNaming::ALBERT):
//! the attention sub-module is flat (`attention.query` not
//! `attention.self.query`; `attention.dense` not
//! `attention.output.dense`), and the FFN uses `ffn` / `ffn_output` /
//! `full_layer_layer_norm` instead of BERT's
//! `intermediate.dense` / `output.dense` / `output.LayerNorm`.
//!
//! ## Activation note
//!
//! Both `albert-base-v1` and `albert/albert-base-v2` ship
//! `hidden_act: "gelu_new"` — the tanh-approximation form
//! (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`).
//! [`AlbertConfig::from_json_str`] parses `hidden_act` into a
//! [`flodl::nn::GeluApprox`] and the encoder layer plus
//! [`AlbertMLMHeadTransform`] dispatch to the matching libtorch op.
//! Picking the wrong form silently produces a ~1e-2 max-abs diff that
//! compounds across the encoder — large enough to fail any meaningful
//! parity test, which is exactly how this bug surfaced.

use std::cell::Cell;
use std::collections::HashMap;

use flodl::nn::{
    Dropout, Embedding, GELU, GeluApprox, LayerNorm, Linear, Module, NamedInputModule, Parameter,
};
use flodl::{
    DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, TensorOptions, Variable,
};

use crate::models::transformer_layer::{LayerNaming, TransformerLayer, TransformerLayerConfig};
use crate::path::{HfPath, prefix_params};

/// ALBERT hyperparameters. Matches the fields of a HuggingFace
/// `AlbertConfig` JSON that affect model shape.
///
/// Use [`AlbertConfig::albert_base_v2`] for the standard 12-layer /
/// 768-hidden / 128-embed preset.
#[derive(Debug, Clone)]
pub struct AlbertConfig {
    pub vocab_size: i64,
    /// Factorised embedding dimension. Must divide `hidden_size`
    /// implicitly — both are independent config knobs; HF defaults
    /// are 128 (embedding) and 768 (hidden) for `albert/albert-base-v2`.
    pub embedding_size: i64,
    pub hidden_size: i64,
    /// Total number of transformer-block applications. With
    /// `num_hidden_groups=1` and `inner_group_num=1`, the same
    /// [`AlbertLayerStack`] inner layer is re-used this many times.
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    pub pad_token_id: Option<i64>,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    /// FFN activation form (parsed from HF `hidden_act`). Default
    /// [`GeluApprox::Tanh`] matches both `albert-base-v1` and
    /// `albert/albert-base-v2` — both ship `hidden_act: "gelu_new"`.
    pub hidden_act: GeluApprox,
    /// See [`crate::models::bert::BertConfig::num_labels`].
    pub num_labels: Option<i64>,
    /// See [`crate::models::bert::BertConfig::id2label`].
    pub id2label: Option<Vec<String>>,
    /// See [`crate::models::bert::BertConfig::architectures`].
    pub architectures: Option<Vec<String>>,
}

impl AlbertConfig {
    /// Preset matching `albert/albert-base-v2` on the HuggingFace Hub.
    pub fn albert_base_v2() -> Self {
        AlbertConfig {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            hidden_act: GeluApprox::Tanh,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Parse a HuggingFace-style `config.json` string into an
    /// [`AlbertConfig`].
    ///
    /// Rejects configs with `num_hidden_groups > 1` or
    /// `inner_group_num > 1` — flodl-hf only supports the "one
    /// transformer block shared across all layers" layout every
    /// public ALBERT checkpoint uses. The error is loud and points
    /// at the offending field so a user who trained their own
    /// ALBERT variant with a different grouping knows exactly what
    /// to file against.
    pub fn from_json_str(s: &str) -> Result<Self> {
        use crate::config_json::{
            optional_f64, optional_hidden_act, optional_i64, optional_i64_or_none,
            parse_architectures, parse_id2label, parse_num_labels, required_i64,
        };
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let num_hidden_groups = optional_i64(&v, "num_hidden_groups", 1);
        if num_hidden_groups != 1 {
            return Err(TensorError::new(&format!(
                "AlbertConfig: num_hidden_groups = {num_hidden_groups} is not supported. \
                 flodl-hf only supports num_hidden_groups=1 (one shared transformer block). \
                 Every public ALBERT checkpoint uses this value.",
            )));
        }
        let inner_group_num = optional_i64(&v, "inner_group_num", 1);
        if inner_group_num != 1 {
            return Err(TensorError::new(&format!(
                "AlbertConfig: inner_group_num = {inner_group_num} is not supported. \
                 flodl-hf only supports inner_group_num=1 (one inner layer per group). \
                 Every public ALBERT checkpoint uses this value.",
            )));
        }
        let id2label = parse_id2label(&v)?;
        let num_labels = parse_num_labels(&v, id2label.as_deref());
        let architectures = parse_architectures(&v);
        Ok(AlbertConfig {
            vocab_size: required_i64(&v, "vocab_size")?,
            embedding_size: required_i64(&v, "embedding_size")?,
            hidden_size: required_i64(&v, "hidden_size")?,
            num_hidden_layers: required_i64(&v, "num_hidden_layers")?,
            num_attention_heads: required_i64(&v, "num_attention_heads")?,
            intermediate_size: required_i64(&v, "intermediate_size")?,
            max_position_embeddings: required_i64(&v, "max_position_embeddings")?,
            type_vocab_size: optional_i64(&v, "type_vocab_size", 2),
            pad_token_id: optional_i64_or_none(&v, "pad_token_id"),
            layer_norm_eps: optional_f64(&v, "layer_norm_eps", 1e-12),
            hidden_dropout_prob: optional_f64(&v, "hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob: optional_f64(&v, "attention_probs_dropout_prob", 0.0),
            // ALBERT default is "gelu_new" — both v1 and v2 ship that.
            // A v3 config (or community fine-tune) that overrides to
            // "gelu" gets the erf form via the same lookup.
            hidden_act: optional_hidden_act(&v, "hidden_act", "gelu_new")?,
            num_labels,
            id2label,
            architectures,
        })
    }

    /// Serialize to a HuggingFace-style `config.json` string.
    ///
    /// Replace the `architectures` field with `[arch_class]` and return
    /// `self`. See [`crate::models::bert::BertConfig::with_architectures`]
    /// for rationale — every family shares the same fix.
    pub fn with_architectures(mut self, arch_class: &str) -> Self {
        self.architectures = Some(vec![arch_class.to_string()]);
        self
    }

    /// Serialize to a HuggingFace-style `config.json` string.
    ///
    /// Inverse of [`Self::from_json_str`]. Emits
    /// `model_type: "albert"` + `architectures: ["AlbertModel"]` plus
    /// the two layout constants (`num_hidden_groups: 1`,
    /// `inner_group_num: 1`) that the parser requires — HF Python's
    /// `AlbertConfig` defaults both to `1`, so the emitted config loads
    /// as-is.
    pub fn to_json_str(&self) -> String {
        use crate::config_json::{emit_architectures, emit_hidden_act, emit_id2label};
        let mut m = serde_json::Map::new();
        m.insert("model_type".into(), "albert".into());
        m.insert(
            "architectures".into(),
            emit_architectures(self.architectures.as_deref(), "AlbertModel"),
        );
        m.insert("vocab_size".into(), self.vocab_size.into());
        m.insert("embedding_size".into(), self.embedding_size.into());
        m.insert("hidden_size".into(), self.hidden_size.into());
        m.insert("num_hidden_layers".into(), self.num_hidden_layers.into());
        m.insert("num_hidden_groups".into(), 1i64.into());
        m.insert("inner_group_num".into(), 1i64.into());
        m.insert(
            "num_attention_heads".into(),
            self.num_attention_heads.into(),
        );
        m.insert("intermediate_size".into(), self.intermediate_size.into());
        m.insert(
            "max_position_embeddings".into(),
            self.max_position_embeddings.into(),
        );
        m.insert("type_vocab_size".into(), self.type_vocab_size.into());
        if let Some(pad) = self.pad_token_id {
            m.insert("pad_token_id".into(), pad.into());
        }
        m.insert("layer_norm_eps".into(), self.layer_norm_eps.into());
        m.insert(
            "hidden_dropout_prob".into(),
            self.hidden_dropout_prob.into(),
        );
        m.insert(
            "attention_probs_dropout_prob".into(),
            self.attention_probs_dropout_prob.into(),
        );
        m.insert("hidden_act".into(), emit_hidden_act(self.hidden_act).into());
        emit_id2label(&mut m, self.id2label.as_deref());
        if let Some(n) = self.num_labels {
            m.insert("num_labels".into(), n.into());
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(m))
            .expect("serde_json::Map serialization is infallible")
    }
}

// ── AlbertEmbeddings ─────────────────────────────────────────────────────

/// Token + position + token-type embeddings in **embedding-dim**
/// space (not hidden-dim) with post-LayerNorm + Dropout.
///
/// Output shape: `[batch, seq_len, embedding_size]`. A downstream
/// `Linear(embedding_size, hidden_size)` lifts this into hidden space
/// before the transformer stack.
pub struct AlbertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl AlbertEmbeddings {
    pub fn on_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        Ok(AlbertEmbeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.embedding_size,
                config.pad_token_id,
                device,
            )?,
            position_embeddings: Embedding::on_device(
                config.max_position_embeddings,
                config.embedding_size,
                device,
            )?,
            token_type_embeddings: Embedding::on_device(
                config.type_vocab_size,
                config.embedding_size,
                device,
            )?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.embedding_size,
                config.layer_norm_eps,
                device,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    /// Clone the word-embedding weight `Parameter` for weight tying
    /// with the MLM decoder. The returned `Parameter` shares its
    /// underlying `Variable` by `Rc`; see
    /// [`BertEmbeddings::word_embeddings_weight`](crate::models::bert::BertEmbeddings::word_embeddings_weight)
    /// for the full contract.
    pub fn word_embeddings_weight(&self) -> Parameter {
        self.word_embeddings.weight.clone()
    }
}

impl Module for AlbertEmbeddings {
    fn name(&self) -> &str {
        "albert_embeddings"
    }

    /// Single-input forward: word ids only. The graph drives the full
    /// three-input path via [`NamedInputModule::forward_named`].
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let word = self.word_embeddings.forward(input)?;
        let ln = self.layer_norm.forward(&word)?;
        self.dropout.forward(&ln)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params(
            "word_embeddings",
            self.word_embeddings.parameters(),
        ));
        out.extend(prefix_params(
            "position_embeddings",
            self.position_embeddings.parameters(),
        ));
        out.extend(prefix_params(
            "token_type_embeddings",
            self.token_type_embeddings.parameters(),
        ));
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

impl NamedInputModule for AlbertEmbeddings {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let mut summed = self.word_embeddings.forward(input)?;
        if let Some(pos) = refs.get("position_ids") {
            let pe = self.position_embeddings.forward(pos)?;
            summed = summed.add(&pe)?;
        }
        if let Some(tt) = refs.get("token_type_ids") {
            let te = self.token_type_embeddings.forward(tt)?;
            summed = summed.add(&te)?;
        }
        let ln = self.layer_norm.forward(&summed)?;
        self.dropout.forward(&ln)
    }
}

// ── AlbertLayerStack ─────────────────────────────────────────────────────

/// One [`TransformerLayer`] applied `num_repeats` times in sequence —
/// ALBERT's cross-layer parameter sharing made explicit.
///
/// Holds a single inner layer, whose parameters are emitted once by
/// [`Self::parameters`]. Training / eval mode propagation also hits
/// just the inner layer. Forward threads the attention mask (and any
/// other refs) to every iteration.
pub struct AlbertLayerStack {
    layer: TransformerLayer,
    num_repeats: i64,
    training: Cell<bool>,
}

impl AlbertLayerStack {
    pub fn on_device(
        config: &TransformerLayerConfig,
        num_repeats: i64,
        device: Device,
    ) -> Result<Self> {
        assert!(
            num_repeats >= 1,
            "AlbertLayerStack: num_repeats must be >= 1, got {num_repeats}",
        );
        Ok(AlbertLayerStack {
            layer: TransformerLayer::on_device(config, LayerNaming::ALBERT, device)?,
            num_repeats,
            training: Cell::new(true),
        })
    }
}

impl Module for AlbertLayerStack {
    fn name(&self) -> &str {
        "albert_layer_stack"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let mut x = self.layer.forward(input)?;
        for _ in 1..self.num_repeats {
            x = self.layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        // One inner layer's params, surfacing once regardless of
        // num_repeats. The FlowBuilder tag supplied at the
        // call site (`albert.encoder.albert_layer_groups.0.albert_layers.0`)
        // lines these up with HF's state_dict keys.
        self.layer.parameters()
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.layer.set_training(training);
    }
}

impl NamedInputModule for AlbertLayerStack {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let mut x = self.layer.forward_named(input, refs)?;
        for _ in 1..self.num_repeats {
            x = self.layer.forward_named(&x, refs)?;
        }
        Ok(x)
    }
}

// ── AlbertPooler ─────────────────────────────────────────────────────────

/// Pooler: take the `[CLS]` hidden state, project through a linear
/// layer, then `tanh`.
///
/// Unlike `BertPooler`, HF's `AlbertModel.pooler` is a flat `nn.Linear`
/// directly on the model — no `.dense` attribute — so its checkpoint
/// keys are `albert.pooler.{weight,bias}` rather than
/// `albert.pooler.dense.{weight,bias}`. The pooler holds its `Linear`
/// without a `prefix_params` wrapper to match.
pub struct AlbertPooler {
    linear: Linear,
}

impl AlbertPooler {
    pub fn on_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        Ok(AlbertPooler {
            linear: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
        })
    }
}

impl Module for AlbertPooler {
    fn name(&self) -> &str {
        "albert_pooler"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let cls = input.select(1, 0)?;
        let pooled = self.linear.forward(&cls)?;
        pooled.tanh()
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.linear.parameters()
    }
}

// ── AlbertModel ──────────────────────────────────────────────────────────

/// Translate an [`AlbertConfig`] into the subset [`TransformerLayer`]
/// consumes.
fn albert_layer_config(config: &AlbertConfig) -> TransformerLayerConfig {
    TransformerLayerConfig {
        hidden_size: config.hidden_size,
        num_attention_heads: config.num_attention_heads,
        intermediate_size: config.intermediate_size,
        hidden_dropout_prob: config.hidden_dropout_prob,
        attention_probs_dropout_prob: config.attention_probs_dropout_prob,
        layer_norm_eps: config.layer_norm_eps,
        hidden_act: config.hidden_act,
    }
}

/// Assemble the ALBERT backbone onto a fresh [`FlowBuilder`], up to
/// and optionally including the pooler.
///
/// Graph signature is **4 named inputs** (matching BERT):
/// `input_ids` (implicit first), `position_ids`, `token_type_ids`,
/// `attention_mask`.
///
/// Graph shape: `albert.embeddings` →
/// `albert.encoder.embedding_hidden_mapping_in` →
/// `albert.encoder.albert_layer_groups.0.albert_layers.0` (shared layer,
/// applied `num_hidden_layers` times) → (`albert.pooler`?).
fn albert_backbone_flow(
    config: &AlbertConfig,
    device: Device,
    with_pooler: bool,
) -> Result<FlowBuilder> {
    let mut fb = FlowBuilder::new()
        .input(&["position_ids", "token_type_ids", "attention_mask"])
        .through(AlbertEmbeddings::on_device(config, device)?)
        .tag("albert.embeddings")
        .using(&["position_ids", "token_type_ids"])
        .through(Linear::on_device(
            config.embedding_size,
            config.hidden_size,
            device,
        )?)
        .tag("albert.encoder.embedding_hidden_mapping_in");

    let layer_cfg = albert_layer_config(config);
    let layer_tag = HfPath::new("albert")
        .sub("encoder")
        .sub("albert_layer_groups")
        .sub(0)
        .sub("albert_layers")
        .sub(0)
        .to_string();
    fb = fb
        .through(AlbertLayerStack::on_device(
            &layer_cfg,
            config.num_hidden_layers,
            device,
        )?)
        .tag(&layer_tag)
        .using(&["attention_mask"]);

    if with_pooler {
        fb = fb
            .through(AlbertPooler::on_device(config, device)?)
            .tag("albert.pooler");
    }
    Ok(fb)
}

/// Assembled ALBERT graph. See module docs for the factorised-
/// embedding + shared-layer architecture.
pub struct AlbertModel;

impl AlbertModel {
    /// Build an ALBERT graph on CPU with a pooler node.
    pub fn build(config: &AlbertConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build an ALBERT graph on `device` with a pooler node. Emits
    /// `pooler_output` (`[batch, hidden]`).
    pub fn on_device(config: &AlbertConfig, device: Device) -> Result<Graph> {
        albert_backbone_flow(config, device, true)?.build()
    }

    /// Build an ALBERT graph on `device` *without* the pooler. Emits
    /// `last_hidden_state` (`[batch, seq_len, hidden]`) — the shape
    /// token-classification and question-answering heads consume.
    pub fn on_device_without_pooler(config: &AlbertConfig, device: Device) -> Result<Graph> {
        albert_backbone_flow(config, device, false)?.build()
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

pub use crate::task_heads::{Answer, TokenPrediction};
use crate::task_heads::{
    ClassificationHead, EncoderInputs, MaskedLmHead, QaHead, TaggingHead, check_num_labels,
};

/// ALBERT graphs take four `forward_multi` inputs — `input_ids`,
/// `position_ids`, `token_type_ids`, and an extended attention mask —
/// in that order. Matches the BERT signature.
#[cfg(feature = "tokenizer")]
impl EncoderInputs for AlbertConfig {
    const FAMILY_NAME: &'static str = "Albert";
    const MASK_TOKEN: &'static str = "[MASK]";

    fn encoder_inputs(enc: &crate::tokenizer::EncodedBatch) -> Result<Vec<Variable>> {
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(
            crate::models::bert::build_extended_attention_mask(&mask_f32)?,
            false,
        );
        Ok(vec![
            enc.input_ids.clone(),
            enc.position_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])
    }
}

/// ALBERT with a sequence-classification head on the pooled `[CLS]`
/// output: `pooler_output → Dropout → Linear(hidden, num_labels)`.
///
/// Parameter keys for the head:
/// - `classifier.weight`  (`[num_labels, hidden]`)
/// - `classifier.bias`    (`[num_labels]`)
///
/// Matches HF Python's `AlbertForSequenceClassification`. Popular
/// checkpoints: `textattack/albert-base-v2-SST-2` (binary sentiment),
/// `textattack/albert-base-v2-MRPC`.
pub type AlbertForSequenceClassification = ClassificationHead<AlbertConfig>;

impl ClassificationHead<AlbertConfig> {
    pub fn on_device(config: &AlbertConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = albert_backbone_flow(config, device, /*with_pooler=*/ true)?
            .through(Dropout::new(config.hidden_dropout_prob))
            .through(Linear::on_device(config.hidden_size, num_labels, device)?)
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(
            graph,
            config,
            num_labels,
            config.id2label.clone(),
        ))
    }

    pub(crate) fn num_labels_from_config(config: &AlbertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "AlbertForSequenceClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// ALBERT with a per-token classification head. Matches HF Python's
/// `AlbertForTokenClassification`.
pub type AlbertForTokenClassification = TaggingHead<AlbertConfig>;

impl TaggingHead<AlbertConfig> {
    pub fn on_device(config: &AlbertConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = albert_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(Dropout::new(config.hidden_dropout_prob))
            .through(Linear::on_device(config.hidden_size, num_labels, device)?)
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(
            graph,
            config,
            num_labels,
            config.id2label.clone(),
        ))
    }

    pub(crate) fn num_labels_from_config(config: &AlbertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "AlbertForTokenClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// ALBERT with an extractive question-answering head:
/// `last_hidden_state → Linear(hidden, 2)`. Matches HF Python's
/// `AlbertForQuestionAnswering`. Popular checkpoints:
/// `twmkn9/albert-base-v2-squad2`.
pub type AlbertForQuestionAnswering = QaHead<AlbertConfig>;

impl QaHead<AlbertConfig> {
    pub fn on_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        let graph = albert_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(Linear::on_device(config.hidden_size, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self::from_graph(graph, config))
    }
}

// ── AlbertMLMHeadTransform ───────────────────────────────────────────────

/// `Linear(hidden, embedding) → GELU → LayerNorm(embedding)` stack
/// sitting between the encoder output and the tied decoder in the MLM
/// head. Shapes: `[B, S, hidden] → [B, S, embedding]`.
///
/// Parameter keys (post-tag `predictions`):
/// - `predictions.dense.{weight,bias}`
/// - `predictions.LayerNorm.{weight,bias}`
///
/// Matches HF Python's `AlbertMLMHead` pre-decoder layout.
pub struct AlbertMLMHeadTransform {
    dense: Linear,
    activation: GELU,
    layer_norm: LayerNorm,
}

impl AlbertMLMHeadTransform {
    pub fn on_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        Ok(AlbertMLMHeadTransform {
            dense: Linear::on_device(config.hidden_size, config.embedding_size, device)?,
            activation: GELU::with_approximate(config.hidden_act),
            layer_norm: LayerNorm::on_device_with_eps(
                config.embedding_size,
                config.layer_norm_eps,
                device,
            )?,
        })
    }
}

impl Module for AlbertMLMHeadTransform {
    fn name(&self) -> &str {
        "albert_mlm_head_transform"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.dense.forward(input)?;
        let x = self.activation.forward(&x)?;
        self.layer_norm.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("dense", self.dense.parameters());
        out.extend(prefix_params("LayerNorm", self.layer_norm.parameters()));
        out
    }
}

/// ALBERT with a masked-language-modelling head: transform stack
/// (`Linear(hidden, embedding) → GELU → LayerNorm`) followed by a
/// decoder `Linear(embedding, vocab_size)` whose weight is **tied**
/// to `albert.embeddings.word_embeddings.weight`.
///
/// Parameter keys emitted by the graph (post-dedup):
/// - `predictions.dense.{weight,bias}`
/// - `predictions.LayerNorm.{weight,bias}`
/// - `predictions.decoder.bias`  (`[vocab_size]`, fresh)
///
/// `predictions.decoder.weight` is **absent** — the decoder borrows
/// `albert.embeddings.word_embeddings.weight` via
/// [`Linear::from_shared_weight`](flodl::nn::Linear::from_shared_weight).
/// HF's historical save format also emits `predictions.bias` as a
/// top-level key tied to `decoder.bias`; the safetensors loader
/// silently ignores that extra key.
///
/// Canonical checkpoints: `albert/albert-base-v2`, `albert-large-v2`.
pub type AlbertForMaskedLM = MaskedLmHead<AlbertConfig>;

impl MaskedLmHead<AlbertConfig> {
    pub fn on_device(config: &AlbertConfig, device: Device) -> Result<Self> {
        // Build embeddings first, capture tied weight before
        // ownership moves into `.through(...)`.
        let embeddings = AlbertEmbeddings::on_device(config, device)?;
        let tied_weight = embeddings.word_embeddings_weight();

        let mut fb = FlowBuilder::new()
            .input(&["position_ids", "token_type_ids", "attention_mask"])
            .through(embeddings)
            .tag("albert.embeddings")
            .using(&["position_ids", "token_type_ids"])
            .through(Linear::on_device(
                config.embedding_size,
                config.hidden_size,
                device,
            )?)
            .tag("albert.encoder.embedding_hidden_mapping_in");

        let layer_cfg = albert_layer_config(config);
        let layer_tag = HfPath::new("albert")
            .sub("encoder")
            .sub("albert_layer_groups")
            .sub(0)
            .sub("albert_layers")
            .sub(0)
            .to_string();
        fb = fb
            .through(AlbertLayerStack::on_device(
                &layer_cfg,
                config.num_hidden_layers,
                device,
            )?)
            .tag(&layer_tag)
            .using(&["attention_mask"]);

        let decoder_bias = Parameter::new(
            Tensor::zeros(
                &[config.vocab_size],
                TensorOptions {
                    dtype: DType::Float32,
                    device,
                },
            )?,
            "bias",
        );
        let graph = fb
            .through(AlbertMLMHeadTransform::on_device(config, device)?)
            .tag("predictions")
            .through(Linear::from_shared_weight(tied_weight, Some(decoder_bias)))
            .tag("predictions.decoder")
            .build()?;

        Ok(Self::from_graph(graph, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_io::expected_from_graph;

    /// Round-trip: preset -> to_json_str -> from_json_str recovers the
    /// same config. ALBERT's writer emits `num_hidden_groups: 1` +
    /// `inner_group_num: 1` (the parser validates these); this test
    /// ensures those invariants are preserved, as well as the
    /// factorised `embedding_size` and the `"gelu_new"` default.
    #[test]
    fn albert_config_to_json_str_round_trip() {
        let preset = AlbertConfig::albert_base_v2();
        let s = preset.to_json_str();
        let recovered = AlbertConfig::from_json_str(&s).unwrap();
        assert_eq!(preset.to_json_str(), recovered.to_json_str());
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v.get("model_type").and_then(|x| x.as_str()), Some("albert"));
        // Layout invariants the parser validates.
        assert_eq!(v.get("num_hidden_groups").and_then(|x| x.as_i64()), Some(1));
        assert_eq!(v.get("inner_group_num").and_then(|x| x.as_i64()), Some(1));
        assert_eq!(v.get("embedding_size").and_then(|x| x.as_i64()), Some(128));
        // ALBERT's tanh activation preserved through emit_hidden_act.
        assert_eq!(
            v.get("hidden_act").and_then(|x| x.as_str()),
            Some("gelu_new"),
        );
    }

    fn mini_config() -> AlbertConfig {
        // Keep dims small so forward passes stay cheap while still
        // exercising the factorised-embedding + shared-layer layout.
        AlbertConfig {
            vocab_size: 128,
            embedding_size: 16,
            hidden_size: 32,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            intermediate_size: 64,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            hidden_act: GeluApprox::Tanh,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Factorisation must be real: word embeddings live in
    /// `embedding_size`, not `hidden_size`; a dedicated projection
    /// lifts them into hidden space.
    #[test]
    fn albert_factorised_embeddings_have_separate_dims() {
        let config = AlbertConfig::albert_base_v2();
        let graph = AlbertModel::on_device_without_pooler(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter()
            .map(|p| (p.key.as_str(), p.shape.as_slice()))
            .collect();

        let v = config.vocab_size;
        let e = config.embedding_size;
        let h = config.hidden_size;
        assert_eq!(by_key["albert.embeddings.word_embeddings.weight"], &[v, e]);
        assert_eq!(
            by_key["albert.embeddings.position_embeddings.weight"],
            &[config.max_position_embeddings, e]
        );
        assert_eq!(by_key["albert.embeddings.LayerNorm.weight"], &[e]);
        assert_eq!(
            by_key["albert.encoder.embedding_hidden_mapping_in.weight"],
            &[h, e],
        );
    }

    /// Cross-layer parameter sharing: regardless of
    /// `num_hidden_layers`, the encoder exposes exactly one set of
    /// transformer params under `albert_layer_groups.0.albert_layers.0`.
    #[test]
    fn albert_cross_layer_sharing_one_param_set_per_stack() {
        let mut config = mini_config();
        config.num_hidden_layers = 12;
        let graph = AlbertModel::on_device_without_pooler(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let layer_keys: Vec<&str> = expected
            .iter()
            .filter(|p| p.key.starts_with("albert.encoder.albert_layer_groups."))
            .map(|p| p.key.as_str())
            .collect();

        // 16 keys per TransformerLayer: 4 Linear × (w+b) + 2
        // LayerNorm × (w+b) = 16. One set only, despite 12
        // applications.
        assert_eq!(layer_keys.len(), 16, "got: {layer_keys:?}");
        for k in &layer_keys {
            assert!(
                k.starts_with("albert.encoder.albert_layer_groups.0.albert_layers.0."),
                "unexpected layer-key prefix: {k}",
            );
        }
    }

    /// ALBERT weight-key suffixes differ from BERT (`attention.query`
    /// not `attention.self.query`; `ffn` not `intermediate.dense`;
    /// `full_layer_layer_norm` not `output.LayerNorm`). A regression
    /// to BERT naming would silently break checkpoint loading.
    #[test]
    fn albert_weight_key_suffixes_match_hf() {
        let graph = AlbertModel::on_device_without_pooler(&mini_config(), Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        let prefix = "albert.encoder.albert_layer_groups.0.albert_layers.0";
        for suffix in &[
            "attention.query.weight",
            "attention.key.weight",
            "attention.value.weight",
            "attention.dense.weight",
            "attention.LayerNorm.weight",
            "ffn.weight",
            "ffn_output.weight",
            "full_layer_layer_norm.weight",
        ] {
            let full = format!("{prefix}.{suffix}");
            assert!(
                keys.iter().any(|k| *k == full),
                "missing expected ALBERT key: {full}; got keys starting with {prefix}.*: {:?}",
                keys.iter()
                    .filter(|k| k.starts_with(prefix))
                    .collect::<Vec<_>>(),
            );
        }

        assert!(
            !keys.iter().any(|k| k.contains("attention.self.")),
            "BERT-style 'attention.self.*' leaked into ALBERT keys: {keys:?}",
        );
    }

    /// Config rejection: `num_hidden_groups > 1` or
    /// `inner_group_num > 1` — unsupported layouts — must error at
    /// parse time so the user can file a clear bug rather than get
    /// a mis-built graph.
    #[test]
    fn albert_rejects_unsupported_grouping() {
        let json = r#"{
            "model_type": "albert",
            "vocab_size": 30000, "embedding_size": 128, "hidden_size": 768,
            "num_hidden_layers": 12, "num_attention_heads": 12,
            "intermediate_size": 3072, "max_position_embeddings": 512,
            "num_hidden_groups": 2
        }"#;
        let err = AlbertConfig::from_json_str(json).unwrap_err().to_string();
        assert!(err.contains("num_hidden_groups"), "got: {err}");

        let json = r#"{
            "model_type": "albert",
            "vocab_size": 30000, "embedding_size": 128, "hidden_size": 768,
            "num_hidden_layers": 12, "num_attention_heads": 12,
            "intermediate_size": 3072, "max_position_embeddings": 512,
            "inner_group_num": 2
        }"#;
        let err = AlbertConfig::from_json_str(json).unwrap_err().to_string();
        assert!(err.contains("inner_group_num"), "got: {err}");
    }

    /// MLM tied-decoder dedup: `[vocab, embedding]`-shaped weight
    /// surfaces once (under the embeddings tag), fresh decoder bias
    /// `[vocab]` surfaces as `predictions.decoder.bias`.
    #[test]
    fn albert_masked_lm_keeps_tied_weight_dedup() {
        let config = mini_config();
        let head = AlbertForMaskedLM::on_device(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert!(
            keys.contains(&"albert.embeddings.word_embeddings.weight"),
            "tied [V, E] weight must surface under albert.embeddings tag: {keys:?}",
        );
        assert!(
            !keys.contains(&"predictions.decoder.weight"),
            "predictions.decoder.weight must be absent (tied, dedup kept embeddings entry)",
        );
        assert!(
            keys.contains(&"predictions.decoder.bias"),
            "fresh decoder bias must appear as predictions.decoder.bias: {keys:?}",
        );

        let named = head.graph().named_parameters();
        let vocab_emb_shaped = named
            .iter()
            .filter(|(_, p)| p.variable.shape() == vec![config.vocab_size, config.embedding_size])
            .count();
        assert_eq!(
            vocab_emb_shaped, 1,
            "exactly one [V, E]-shaped Parameter expected under tying",
        );
    }

    /// Smoke: ALBERT MLM head emits `[batch, seq, vocab_size]` logits
    /// via the factorised decoder path (`hidden → embedding → vocab`).
    #[test]
    fn albert_masked_lm_forward_shape_smoke() {
        let config = mini_config();
        let dev = Device::CPU;
        let head = AlbertForMaskedLM::on_device(&config, dev).unwrap();
        head.graph().eval();

        let batch = 1;
        let seq = 3;
        let ids = Variable::new(
            Tensor::from_i64(&[1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let pos = Variable::new(
            Tensor::from_i64(&[0, 1, 2], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(
            Tensor::from_i64(&[0; 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(
            &[batch, seq],
            TensorOptions {
                dtype: DType::Float32,
                device: dev,
            },
        )
        .unwrap();
        let mask = Variable::new(
            crate::models::bert::build_extended_attention_mask(&mask_flat).unwrap(),
            false,
        );

        let out = head.graph().forward_multi(&[ids, pos, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, config.vocab_size]);
    }
}
