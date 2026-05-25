//! DistilBERT encoder, compatible with HuggingFace `distilbert-base-uncased`
//! checkpoints.
//!
//! DistilBERT is the 6-layer distilled BERT from Sanh et al. (2019). The
//! encoder shape is BERT-identical — self-attention + two-layer GELU
//! feed-forward, both residual-connected with post-LN — but the published
//! implementation diverges from BERT's in several load-bearing ways:
//!
//! 1. **No `token_type_ids`.** DistilBERT is single-segment. The
//!    embedding layer has no `token_type_embeddings` table, and the
//!    graph takes only two inputs (`input_ids` implicit + `attention_mask`).
//! 2. **No pooler.** Sequence-classification heads operate on the first
//!    token's hidden state directly; there is no `pooler.dense` layer
//!    and no tanh squash.
//! 3. **Position ids are sequential** (`0..seq_len`), not padding-aware
//!    like RoBERTa. Computed internally rather than threaded as a graph
//!    input.
//! 4. **Weight-key divergence.** BERT uses
//!    `attention.self.{query,key,value}` + `attention.output.dense` +
//!    `intermediate.dense` + `output.dense` +
//!    `attention.output.LayerNorm` + `output.LayerNorm`. DistilBERT uses
//!    `attention.{q_lin,k_lin,v_lin,out_lin}` + `ffn.{lin1,lin2}` +
//!    `sa_layer_norm` + `output_layer_norm`. A cross-family
//!    `LayerNaming` abstraction reconciles both at weight-loading time.
//! 5. **Config field names** differ from BERT's:
//!    `n_layers` / `n_heads` / `dim` / `hidden_dim` instead of
//!    `num_hidden_layers` / `num_attention_heads` / `hidden_size` /
//!    `intermediate_size`. Preserved as-is here to match HF Python's
//!    `DistilBertConfig` for API parity.
//!
//! Task heads deviate too. `DistilBertForSequenceClassification` is a
//! two-layer head (`pre_classifier` → ReLU → Dropout → `classifier`) with
//! its own `seq_classif_dropout` probability. QA uses an extra
//! `qa_dropout` before the output projection. Token classification
//! matches BERT/RoBERTa (Dropout + Linear).

use std::cell::Cell;

use flodl::nn::{Dropout, Embedding, GELU, GeluApprox, LayerNorm, Linear, Module, Parameter};
use flodl::{
    DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, TensorOptions, Variable,
};

use crate::models::transformer_layer::{LayerNaming, TransformerLayer, TransformerLayerConfig};
use crate::path::{prefix_params, HfPath};
use crate::task_heads::{
    check_num_labels, ClassificationHead, EncoderInputs, MaskedLmHead, QaHead, TaggingHead,
};
pub use crate::task_heads::{Answer, TokenPrediction};

/// DistilBERT graphs take two `forward_multi` inputs — `input_ids` and
/// an extended attention mask — in that order. DistilBERT drops BERT's
/// `token_type_ids` and `position_ids` entirely: there's no segment
/// embedding, and position ids are learned absolute positions looked
/// up inside the embedding layer from `[0, seq_len)`.
#[cfg(feature = "tokenizer")]
impl EncoderInputs for DistilBertConfig {
    const FAMILY_NAME: &'static str = "DistilBert";
    const MASK_TOKEN: &'static str = "[MASK]";

    fn encoder_inputs(enc: &crate::tokenizer::EncodedBatch) -> Result<Vec<Variable>> {
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(
            crate::models::bert::build_extended_attention_mask(&mask_f32)?,
            false,
        );
        Ok(vec![enc.input_ids.clone(), mask])
    }
}

/// DistilBERT hyperparameters. Matches the fields of a HuggingFace
/// `DistilBertConfig` JSON file that affect model shape.
///
/// Field names mirror HF Python exactly (`n_layers`, `dim`,
/// `hidden_dim`, `n_heads`) rather than BERT's (`num_hidden_layers`,
/// `hidden_size`, `intermediate_size`, `num_attention_heads`). This
/// keeps cross-referencing HF docs friction-free; the encoder
/// implementation pays the small cost of translating at construction
/// sites.
///
/// Use [`DistilBertConfig::distilbert_base_uncased`] for the standard
/// 6-layer / 768-dim preset.
#[derive(Debug, Clone)]
pub struct DistilBertConfig {
    pub vocab_size: i64,
    /// Hidden dimension (BERT's `hidden_size`). HF key: `dim`.
    pub dim: i64,
    /// Number of encoder layers (BERT's `num_hidden_layers`).
    /// HF key: `n_layers`.
    pub n_layers: i64,
    /// Attention heads per layer (BERT's `num_attention_heads`).
    /// HF key: `n_heads`.
    pub n_heads: i64,
    /// Feed-forward inner dimension (BERT's `intermediate_size`).
    /// HF key: `hidden_dim`.
    pub hidden_dim: i64,
    pub max_position_embeddings: i64,
    /// Padding token index. Freezes the gradient on row `pad_token_id`
    /// of the word-embedding table. Every public DistilBERT checkpoint
    /// uses `0`.
    pub pad_token_id: i64,
    /// Residual / output dropout (BERT's `hidden_dropout_prob`).
    /// HF key: `dropout`.
    pub dropout: f64,
    /// Attention-softmax dropout (BERT's
    /// `attention_probs_dropout_prob`). HF key: `attention_dropout`.
    pub attention_dropout: f64,
    /// Dropout applied before the QA output projection.
    /// DistilBERT-specific; BERT/RoBERTa reuse `hidden_dropout_prob`
    /// there.
    pub qa_dropout: f64,
    /// Dropout inside the two-layer sequence-classification head
    /// (`pre_classifier` → ReLU → Dropout → `classifier`).
    /// DistilBERT-specific; typical value `0.2`, distinct from the
    /// encoder-wide `dropout`.
    pub seq_classif_dropout: f64,
    /// Whether position embeddings should be initialized from a
    /// sinusoidal table rather than trained. HF Python uses this only
    /// at module `__init__`; `from_pretrained` still overwrites the
    /// table with the checkpoint's `position_embeddings.weight`, so in
    /// practice every public checkpoint ships learned positions and
    /// this flag has no runtime effect. Preserved here for fidelity
    /// but not consulted by flodl's load path.
    pub sinusoidal_pos_embds: bool,
    /// LayerNorm epsilon. DistilBERT configs do not ship this field;
    /// defaults to `1e-12` to match the BERT family.
    pub layer_norm_eps: f64,
    /// FFN activation form (parsed from HF `activation` — DistilBERT
    /// uses that key, not `hidden_act`). Default `GeluApprox::Exact`
    /// (erf form) matches `distilbert-base-uncased`.
    pub hidden_act: GeluApprox,
    /// See [`crate::models::bert::BertConfig::num_labels`].
    pub num_labels: Option<i64>,
    /// See [`crate::models::bert::BertConfig::id2label`].
    pub id2label: Option<Vec<String>>,
    /// See [`crate::models::bert::BertConfig::architectures`].
    pub architectures: Option<Vec<String>>,
}

impl DistilBertConfig {
    /// Preset matching `distilbert-base-uncased` on the HuggingFace Hub.
    pub fn distilbert_base_uncased() -> Self {
        DistilBertConfig {
            vocab_size: 30522,
            dim: 768,
            n_layers: 6,
            n_heads: 12,
            hidden_dim: 3072,
            max_position_embeddings: 512,
            pad_token_id: 0,
            dropout: 0.1,
            attention_dropout: 0.1,
            qa_dropout: 0.1,
            seq_classif_dropout: 0.2,
            sinusoidal_pos_embds: false,
            layer_norm_eps: 1e-12,
            hidden_act: GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Parse a HuggingFace-style `config.json` string into a
    /// [`DistilBertConfig`].
    ///
    /// Required integer fields (`vocab_size`, `dim`, `n_layers`,
    /// `n_heads`, `hidden_dim`, `max_position_embeddings`) error out
    /// if missing. Dropouts, `pad_token_id`, and `layer_norm_eps` fall
    /// back to the DistilBERT defaults. Unknown fields are ignored
    /// (architecture lists, torch dtype, `tie_weights_`, …).
    ///
    /// `activation` is parsed and dispatched: `"gelu"` → erf form,
    /// `"gelu_new"` / `"gelu_pytorch_tanh"` → tanh approximation. Other
    /// values error loudly.
    pub fn from_json_str(s: &str) -> Result<Self> {
        use crate::config_json::{
            optional_bool, optional_f64, optional_hidden_act, optional_i64, parse_architectures,
            parse_id2label, parse_num_labels, required_i64,
        };
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let id2label = parse_id2label(&v)?;
        let num_labels = parse_num_labels(&v, id2label.as_deref());
        let architectures = parse_architectures(&v);
        Ok(DistilBertConfig {
            vocab_size:              required_i64(&v, "vocab_size")?,
            dim:                     required_i64(&v, "dim")?,
            n_layers:                required_i64(&v, "n_layers")?,
            n_heads:                 required_i64(&v, "n_heads")?,
            hidden_dim:              required_i64(&v, "hidden_dim")?,
            max_position_embeddings: required_i64(&v, "max_position_embeddings")?,
            pad_token_id:            optional_i64(&v, "pad_token_id", 0),
            dropout:                 optional_f64(&v, "dropout", 0.1),
            attention_dropout:       optional_f64(&v, "attention_dropout", 0.1),
            qa_dropout:              optional_f64(&v, "qa_dropout", 0.1),
            seq_classif_dropout:     optional_f64(&v, "seq_classif_dropout", 0.2),
            sinusoidal_pos_embds:    optional_bool(&v, "sinusoidal_pos_embds", false),
            layer_norm_eps:          optional_f64(&v, "layer_norm_eps", 1e-12),
            hidden_act:              optional_hidden_act(&v, "activation", "gelu")?,
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
    /// Inverse of [`Self::from_json_str`]. DistilBERT uses HF's
    /// shorter field names (`dim`, `n_layers`, `n_heads`, `hidden_dim`,
    /// `activation`, `dropout`, `attention_dropout`) rather than the
    /// BERT-family names — the emitted JSON matches that shape so the
    /// Python `DistilBertConfig` parses it unchanged. Emits
    /// `model_type: "distilbert"` + `architectures: ["DistilBertModel"]`.
    pub fn to_json_str(&self) -> String {
        use crate::config_json::{emit_architectures, emit_hidden_act, emit_id2label};
        let mut m = serde_json::Map::new();
        m.insert("model_type".into(), "distilbert".into());
        m.insert(
            "architectures".into(),
            emit_architectures(self.architectures.as_deref(), "DistilBertModel"),
        );
        m.insert("vocab_size".into(), self.vocab_size.into());
        m.insert("dim".into(), self.dim.into());
        m.insert("n_layers".into(), self.n_layers.into());
        m.insert("n_heads".into(), self.n_heads.into());
        m.insert("hidden_dim".into(), self.hidden_dim.into());
        m.insert(
            "max_position_embeddings".into(),
            self.max_position_embeddings.into(),
        );
        m.insert("pad_token_id".into(), self.pad_token_id.into());
        m.insert("dropout".into(), self.dropout.into());
        m.insert("attention_dropout".into(), self.attention_dropout.into());
        m.insert("qa_dropout".into(), self.qa_dropout.into());
        m.insert("seq_classif_dropout".into(), self.seq_classif_dropout.into());
        m.insert(
            "sinusoidal_pos_embds".into(),
            self.sinusoidal_pos_embds.into(),
        );
        m.insert("layer_norm_eps".into(), self.layer_norm_eps.into());
        m.insert("activation".into(), emit_hidden_act(self.hidden_act).into());
        emit_id2label(&mut m, self.id2label.as_deref());
        if let Some(n) = self.num_labels {
            m.insert("num_labels".into(), n.into());
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(m))
            .expect("serde_json::Map serialization is infallible")
    }
}

// ── DistilBertEmbeddings ─────────────────────────────────────────────────

/// Token + position embeddings with post-LN and Dropout.
///
/// Distinct from [`BertEmbeddings`](crate::models::bert::BertEmbeddings)
/// in two ways: there is no `token_type_embeddings` table (DistilBERT is
/// single-segment), and position ids are computed internally from
/// `input_ids` shape as `0..seq_len` broadcast across the batch — matching
/// HF Python's `Embeddings.forward`.
///
/// No [`NamedInputModule`](flodl::nn::NamedInputModule) impl: the
/// graph feeds only `input_ids` (implicit first), so the single-arg
/// [`Module::forward`] path covers production use.
pub struct DistilBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DistilBertEmbeddings {
    pub fn on_device(config: &DistilBertConfig, device: Device) -> Result<Self> {
        Ok(DistilBertEmbeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.dim,
                Some(config.pad_token_id),
                device,
            )?,
            position_embeddings: Embedding::on_device(
                config.max_position_embeddings,
                config.dim,
                device,
            )?,
            layer_norm: LayerNorm::on_device_with_eps(
                config.dim,
                config.layer_norm_eps,
                device,
            )?,
            dropout: Dropout::new(config.dropout),
        })
    }

    /// Clone the word-embedding weight `Parameter` for weight tying.
    ///
    /// See [`crate::models::bert::BertEmbeddings::word_embeddings_weight`]
    /// for the full contract. The tied weight surfaces once under
    /// `distilbert.embeddings.word_embeddings.weight` when the MLM
    /// decoder shares it via [`flodl::nn::Linear::from_shared_weight`].
    ///
    /// Call this **before** moving the embeddings into the backbone's
    /// `FlowBuilder`, since `.through(...)` consumes ownership.
    pub fn word_embeddings_weight(&self) -> Parameter {
        self.word_embeddings.weight.clone()
    }

    /// Build sequential `0..seq_len` position ids matching `input_ids`
    /// shape. Runs on raw tensors (no autograd) — position ids are
    /// integer indices that never participate in backward.
    fn position_ids_from_input_ids(input_ids: &Tensor) -> Result<Tensor> {
        let shape = input_ids.shape();
        assert_eq!(shape.len(), 2, "input_ids must be [B, S], got {shape:?}");
        let batch = shape[0];
        let seq = shape[1];
        let pos = Tensor::arange(
            0.0,
            seq as f64,
            1.0,
            TensorOptions { dtype: DType::Int64, device: input_ids.device() },
        )?;
        pos.reshape(&[1, seq])?.expand(&[batch, seq])
    }
}

impl Module for DistilBertEmbeddings {
    fn name(&self) -> &str { "distilbert_embeddings" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let pos_ids = Self::position_ids_from_input_ids(&input.data())?;
        let pos_var = Variable::new(pos_ids, false);
        let word = self.word_embeddings.forward(input)?;
        let pe = self.position_embeddings.forward(&pos_var)?;
        let summed = word.add(&pe)?;
        let ln = self.layer_norm.forward(&summed)?;
        self.dropout.forward(&ln)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("word_embeddings",     self.word_embeddings.parameters()));
        out.extend(prefix_params("position_embeddings", self.position_embeddings.parameters()));
        out.extend(prefix_params("LayerNorm",           self.layer_norm.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ── DistilBertModel ──────────────────────────────────────────────────────

/// Translate a [`DistilBertConfig`] into the subset [`TransformerLayer`]
/// consumes. Maps DistilBERT's native field names (`dim`, `n_layers`,
/// `hidden_dim`, …) onto the cross-family vocabulary.
fn distilbert_layer_config(config: &DistilBertConfig) -> TransformerLayerConfig {
    TransformerLayerConfig {
        hidden_size:                  config.dim,
        num_attention_heads:          config.n_heads,
        intermediate_size:            config.hidden_dim,
        hidden_dropout_prob:          config.dropout,
        attention_probs_dropout_prob: config.attention_dropout,
        layer_norm_eps:               config.layer_norm_eps,
        hidden_act:                   config.hidden_act,
    }
}

/// Assemble the DistilBERT backbone onto a fresh [`FlowBuilder`].
/// There is no pooler — task heads that need a CLS-like summary select
/// index 0 of the last hidden state directly.
///
/// Graph signature: **2 named inputs** — `input_ids` (implicit first)
/// and `attention_mask`. No `position_ids` (sequential, computed
/// internally), no `token_type_ids` (DistilBERT is single-segment).
///
/// Graph shape: `distilbert.embeddings` →
/// `distilbert.transformer.layer.{0..N-1}`.
fn distilbert_backbone_flow(
    config: &DistilBertConfig,
    device: Device,
) -> Result<FlowBuilder> {
    let mut fb = FlowBuilder::new()
        .input(&["attention_mask"])
        .through(DistilBertEmbeddings::on_device(config, device)?)
        .tag("distilbert.embeddings");

    let layer_root = HfPath::new("distilbert").sub("transformer").sub("layer");
    let layer_cfg = distilbert_layer_config(config);
    for i in 0..config.n_layers {
        let tag = layer_root.sub(i).to_string();
        fb = fb
            .through(TransformerLayer::on_device(&layer_cfg, LayerNaming::DISTILBERT, device)?)
            .tag(&tag)
            .using(&["attention_mask"]);
    }
    Ok(fb)
}

/// Assembled DistilBERT graph.
///
/// The returned [`Graph`] accepts **two** inputs via `forward_multi`, in
/// declaration order:
///
/// 1. `input_ids` (i64, shape `[batch, seq_len]`)
/// 2. `attention_mask` (f32, shape `[batch, 1, 1, seq_len]`, additive —
///    build with [`crate::models::bert::build_extended_attention_mask`]
///    from a plain `[batch, seq_len]` 0/1 mask)
///
/// Output shape: `last_hidden_state` — `[batch, seq_len, dim]`. There
/// is no pooled output; task heads handle CLS extraction themselves.
pub struct DistilBertModel;

impl DistilBertModel {
    /// Build a DistilBERT graph on CPU.
    pub fn build(config: &DistilBertConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build a DistilBERT graph on `device`.
    pub fn on_device(config: &DistilBertConfig, device: Device) -> Result<Graph> {
        distilbert_backbone_flow(config, device)?.build()
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

// ── SeqCls head building blocks ──────────────────────────────────────────

/// Inner "select CLS then project" stage used by
/// [`DistilBertForSequenceClassification`]. Takes `[B, S, dim]`, selects
/// index 0 along the sequence axis, applies a learned linear
/// projection. Parameter keys are emitted as `weight`/`bias` so the
/// call site can tag this block with `"pre_classifier"` and have the
/// final keys land at `pre_classifier.weight` / `pre_classifier.bias`
/// — matching HF Python's state_dict layout.
struct SelectClsLinear {
    linear: Linear,
}

impl Module for SelectClsLinear {
    fn name(&self) -> &str { "select_cls_linear" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [B, S, dim] → [B, dim]
        let cls = input.select(1, 0)?;
        self.linear.forward(&cls)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.linear.parameters()
    }
}

/// Inner "activation → dropout → project" stage used by
/// [`DistilBertForSequenceClassification`]. Runs ReLU + configurable
/// dropout on its input, then a learned linear projection. Dropout is
/// skipped at eval time.
struct ActivationDropoutLinear {
    dropout: Dropout,
    linear: Linear,
    training: Cell<bool>,
}

impl Module for ActivationDropoutLinear {
    fn name(&self) -> &str { "activation_dropout_linear" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let acted = input.relu()?;
        let dropped = if self.training.get() {
            self.dropout.forward(&acted)?
        } else {
            acted
        };
        self.linear.forward(&dropped)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.linear.parameters()
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
        self.dropout.set_training(training);
    }
}

// ── DistilBertForSequenceClassification ──────────────────────────────────

/// DistilBERT with a sequence-classification head on the first token's
/// hidden state: `hidden[:, 0] → pre_classifier → ReLU → Dropout →
/// classifier`.
///
/// Parameter keys for the head:
/// - `pre_classifier.weight`  (`[dim, dim]`)
/// - `pre_classifier.bias`    (`[dim]`)
/// - `classifier.weight`      (`[num_labels, dim]`)
/// - `classifier.bias`        (`[num_labels]`)
///
/// Matches HF Python's `DistilBertForSequenceClassification`.
/// Pre-trained checkpoints:
/// `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
/// (3-class sentiment),
/// `distilbert-base-uncased-finetuned-sst-2-english` (2-class).
/// Type alias over the generic [`ClassificationHead`]; `predict`,
/// `classify`, `forward_encoded`, `compute_loss`, `labels`, `graph`,
/// `config`, and `with_tokenizer` are inherited. Only the
/// DistilBERT-specific `on_device` constructor lives below.
pub type DistilBertForSequenceClassification = ClassificationHead<DistilBertConfig>;

impl ClassificationHead<DistilBertConfig> {
    /// Build the full graph (backbone + 2-layer classification head) on
    /// `device` without loading any weights.
    pub fn on_device(
        config: &DistilBertConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = distilbert_backbone_flow(config, device)?
            .through(SelectClsLinear {
                linear: Linear::on_device(config.dim, config.dim, device)?,
            })
            .tag("pre_classifier")
            .through(ActivationDropoutLinear {
                dropout: Dropout::new(config.seq_classif_dropout),
                linear: Linear::on_device(config.dim, num_labels, device)?,
                training: Cell::new(true),
            })
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(graph, config, num_labels, config.id2label.clone()))
    }

    pub(crate) fn num_labels_from_config(config: &DistilBertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "DistilBertForSequenceClassification: config.json has no \
                 `num_labels` (nor `id2label`); cannot infer head size",
            )
        })
    }
}

// ── DistilBertForTokenClassification ─────────────────────────────────────

/// DistilBERT with a per-token classification head: `last_hidden_state
/// → Dropout → Linear(dim, num_labels)`. NER, POS tagging, any
/// sequence-labelling task.
///
/// Parameter keys for the head:
/// - `classifier.weight`  (`[num_labels, dim]`)
/// - `classifier.bias`    (`[num_labels]`)
///
/// Matches HF Python's `DistilBertForTokenClassification`. Pre-trained
/// checkpoints: `dslim/distilbert-NER` (PER/ORG/LOC/MISC, 4 entity
/// types × BIO = 9 labels including `O`).
/// Type alias over the generic [`TaggingHead`]; all per-token
/// machinery is inherited. Only the DistilBERT-specific `on_device`
/// constructor lives below.
pub type DistilBertForTokenClassification = TaggingHead<DistilBertConfig>;

impl TaggingHead<DistilBertConfig> {
    pub fn on_device(
        config: &DistilBertConfig,
        num_labels: i64,
        device: Device,
    ) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = distilbert_backbone_flow(config, device)?
            .through(Dropout::new(config.dropout))
            .through(Linear::on_device(config.dim, num_labels, device)?)
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(graph, config, num_labels, config.id2label.clone()))
    }

    pub(crate) fn num_labels_from_config(config: &DistilBertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "DistilBertForTokenClassification: config.json has no \
                 `num_labels` (nor `id2label`); cannot infer head size",
            )
        })
    }
}

// ── DistilBertForQuestionAnswering ───────────────────────────────────────

/// DistilBERT with an extractive QA head: `last_hidden_state →
/// Dropout(qa_dropout) → Linear(dim, 2)`, split into start / end logits
/// along the last dim.
///
/// Parameter keys for the head:
/// - `qa_outputs.weight`  (`[2, dim]`)
/// - `qa_outputs.bias`    (`[2]`)
///
/// Matches HF Python's `DistilBertForQuestionAnswering`. Canonical
/// checkpoint: `distilbert-base-cased-distilled-squad`.
/// Type alias over the generic [`QaHead`]; span-extraction,
/// `forward_encoded`, `compute_loss`, and tokenizer plumbing are
/// inherited. Only the DistilBERT-specific `on_device` constructor
/// lives below.
pub type DistilBertForQuestionAnswering = QaHead<DistilBertConfig>;

impl QaHead<DistilBertConfig> {
    pub fn on_device(config: &DistilBertConfig, device: Device) -> Result<Self> {
        let graph = distilbert_backbone_flow(config, device)?
            .through(Dropout::new(config.qa_dropout))
            .through(Linear::on_device(config.dim, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self::from_graph(graph, config))
    }
}

// ── DistilBertForMaskedLM ────────────────────────────────────────────────

/// DistilBERT with a masked-language-modelling head. HF Python's
/// `DistilBertForMaskedLM` lays out the LM head flat on top of the
/// backbone — no `cls.` / `lm_head.` subgroup prefix — so the graph
/// mirrors that with three top-level tags:
///
/// - `vocab_transform` — `Linear(dim, dim)`
/// - `vocab_layer_norm` — `LayerNorm(dim)`
/// - `vocab_projector` — `Linear(dim, vocab_size)`, weight tied to
///   `distilbert.embeddings.word_embeddings.weight`
///
/// A `GELU` node sits between `vocab_transform` and `vocab_layer_norm`;
/// it carries no parameters so it leaves no key in the state_dict.
///
/// Primary use case: **continued pretraining / domain adaptation** on
/// private corpora. Callers feed masked `input_ids` and labels shaped
/// `[batch, seq_len]` where loss-relevant positions carry the original
/// token id and the rest is `-100`. See [`crate::task_heads::masked_lm_loss`].
///
/// Parameter keys emitted by the graph (post-dedup):
/// - `vocab_transform.{weight,bias}`
/// - `vocab_layer_norm.{weight,bias}`
/// - `vocab_projector.bias`  (`[vocab_size]`, fresh)
///
/// `vocab_projector.weight` is **absent** — tied to the embedding
/// table via [`Linear::from_shared_weight`] and dedup'd by
/// `Graph::named_parameters()`. HF checkpoints that save the
/// redundant `vocab_projector.weight` load fine — the safetensors
/// loader silently ignores unused keys.
///
/// Matches HF Python's `DistilBertForMaskedLM`. Canonical checkpoints:
/// `distilbert-base-uncased`, `distilbert-base-cased`.
/// Type alias over the generic [`MaskedLmHead`]; `fill_mask`,
/// `forward_encoded`, `compute_loss`, and tokenizer plumbing are
/// inherited. Only the DistilBERT-specific `on_device` constructor
/// lives below.
pub type DistilBertForMaskedLM = MaskedLmHead<DistilBertConfig>;

impl MaskedLmHead<DistilBertConfig> {
    /// Build the full graph: backbone + vocab_transform + gelu +
    /// vocab_layer_norm + tied vocab_projector on `device`. Initializes
    /// all weights fresh; use
    /// [`from_pretrained`](crate::models::distilbert::DistilBertForMaskedLM::from_pretrained)
    /// to load a checkpoint.
    pub fn on_device(config: &DistilBertConfig, device: Device) -> Result<Self> {
        let embeddings = DistilBertEmbeddings::on_device(config, device)?;
        let tied_weight = embeddings.word_embeddings_weight();

        let mut fb = FlowBuilder::new()
            .input(&["attention_mask"])
            .through(embeddings)
            .tag("distilbert.embeddings");

        let layer_root = HfPath::new("distilbert").sub("transformer").sub("layer");
        let layer_cfg = distilbert_layer_config(config);
        for i in 0..config.n_layers {
            let tag = layer_root.sub(i).to_string();
            fb = fb
                .through(TransformerLayer::on_device(&layer_cfg, LayerNaming::DISTILBERT, device)?)
                .tag(&tag)
                .using(&["attention_mask"]);
        }

        // LM head: dense → gelu → layer_norm → tied projector (fresh bias).
        let projector_bias = Parameter::new(
            Tensor::zeros(
                &[config.vocab_size],
                TensorOptions { dtype: DType::Float32, device },
            )?,
            "bias",
        );
        let graph = fb
            .through(Linear::on_device(config.dim, config.dim, device)?)
            .tag("vocab_transform")
            .through(GELU::with_approximate(config.hidden_act))
            .through(LayerNorm::on_device_with_eps(
                config.dim,
                config.layer_norm_eps,
                device,
            )?)
            .tag("vocab_layer_norm")
            .through(Linear::from_shared_weight(tied_weight, Some(projector_bias)))
            .tag("vocab_projector")
            .build()?;

        Ok(Self::from_graph(graph, config))
    }
}

#[cfg(test)]
#[path = "distilbert_tests.rs"]
mod tests;
