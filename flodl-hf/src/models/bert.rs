//! BERT encoder, compatible with HuggingFace `bert-base-uncased` checkpoints.
//!
//! Structure: [`BertEmbeddings`] (token + position + token-type embeddings
//! with LayerNorm and Dropout), a stack of
//! [`crate::models::transformer_layer::TransformerLayer`]
//! instances (self-attention + two-layer GELU feed-forward, both wrapped
//! with residual + LayerNorm), and [`BertPooler`] on the `[CLS]` position.
//! [`BertModel::build`] assembles all of this into a flat [`Graph`].
//!
//! The encoder layer itself is shared with the RoBERTa and DistilBERT
//! ports via
//! [`LayerNaming::BERT`](crate::models::transformer_layer::LayerNaming::BERT)
//! — the Q/K/V projections, output projection, two LayerNorms, and
//! feed-forward block are identical, only the HF weight-key suffixes
//! differ.
//!
//! Padding is handled via an additive attention mask threaded into every
//! encoder layer as a named graph input (see [`build_extended_attention_mask`]).
//!
//! Parameter names are chosen so `Graph::named_parameters()` output, once
//! passed through [`hf_key_from_flodl_key`](crate::path::hf_key_from_flodl_key),
//! matches safetensors checkpoint keys exactly. No remapping needed at load
//! time.

use std::collections::HashMap;

use flodl::nn::{
    Dropout, Embedding, GELU, GeluApprox, LayerNorm, Linear, Module, NamedInputModule, Parameter,
};
use flodl::{
    DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, TensorOptions, Variable,
};

use crate::models::transformer_layer::{LayerNaming, TransformerLayer, TransformerLayerConfig};
use crate::path::{HfPath, prefix_params};

/// Convert a `[batch, seq_len]` attention mask (0 = mask, 1 = attend,
/// any numeric dtype) into a `[batch, 1, 1, seq_len]` additive f32 mask
/// suitable as the fourth input to the BERT graph.
///
/// Masked positions receive `-1e4`, attended positions `0.0`. The additive
/// mask is broadcast into the QKᵀ pre-softmax scores inside
/// `scaled_dot_product_attention`. `-1e4` (rather than `-inf`) matches
/// HuggingFace's `get_extended_attention_mask` convention and stays
/// numerically safe under fp16.
pub fn build_extended_attention_mask(mask: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    assert_eq!(shape.len(), 2, "expected [batch, seq_len], got {shape:?}");
    let mask_f = mask.to_dtype(DType::Float32)?;
    let additive = mask_f.mul_scalar(-1.0)?.add_scalar(1.0)?.mul_scalar(-1e4)?;
    additive.reshape(&[shape[0], 1, 1, shape[1]])
}

/// BERT hyperparameters. Matches the fields of a HuggingFace
/// `BertConfig` JSON file that affect model shape.
///
/// Use [`BertConfig::bert_base_uncased`] for the standard 12-layer / 768-dim
/// preset.
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    /// Padding token index. `None` when `pad_token_id == eos_token_id` or
    /// when padding is handled entirely via the attention mask; `Some(i)`
    /// freezes the gradient on row `i` of the word-embedding table.
    pub pad_token_id: Option<i64>,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    /// FFN activation form (parsed from HF `hidden_act`). Default
    /// `GeluApprox::Exact` (erf form) matches `bert-base-uncased`. Loud
    /// error from [`Self::from_json_str`] on unrecognised activation
    /// names.
    pub hidden_act: GeluApprox,
    /// Number of output labels for classification-style task heads. `None`
    /// on base `BertModel` configs; `Some(N)` when the checkpoint was fine-
    /// tuned as `BertForSequenceClassification`, `BertForTokenClassification`,
    /// etc. Derived from the HF `num_labels` field, or from the length of
    /// `id2label` if only the label map is present.
    pub num_labels: Option<i64>,
    /// Label strings indexed by class id (`id2label[k]` is the name of class
    /// `k`). `None` for base configs; `Some(vec)` for fine-tuned heads that
    /// shipped with an `id2label` / `label2id` mapping. Ordered by integer
    /// id so `vec[k]` reads like HF Python's `config.id2label[k]`.
    pub id2label: Option<Vec<String>>,
    /// HF Python class name list (e.g. `["BertForSequenceClassification"]`).
    /// `None` for configs that omit the field; otherwise the verbatim list
    /// from the source `config.json`. Read by
    /// [`crate::export::build_for_export`] to dispatch a checkpoint to the
    /// matching task-head builder, and round-tripped by
    /// [`Self::to_json_str`] so HF Python re-dispatches to the same class.
    pub architectures: Option<Vec<String>>,
}

impl BertConfig {
    /// Preset matching `bert-base-uncased` on the HuggingFace Hub.
    pub fn bert_base_uncased() -> Self {
        BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            hidden_act: GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Parse a HuggingFace-style `config.json` string into a [`BertConfig`].
    ///
    /// Reads the fields that affect model shape
    /// (`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
    /// `intermediate_size`, `max_position_embeddings`, `type_vocab_size`,
    /// `pad_token_id`, `layer_norm_eps`, `hidden_dropout_prob`,
    /// `attention_probs_dropout_prob`) plus the task-head metadata
    /// (`num_labels`, `id2label`) used by
    /// [`BertForSequenceClassification`] / [`BertForTokenClassification`] /
    /// [`BertForQuestionAnswering`]. Unknown fields are ignored, so adding
    /// new HF metadata (architecture lists, model type, torch dtype, …)
    /// doesn't break existing checkpoints.
    ///
    /// Required integer fields return a clear error if missing; dropout and
    /// layer-norm-eps fall back to the BERT defaults.
    ///
    /// `hidden_act` is parsed and dispatched: `"gelu"` → erf form,
    /// `"gelu_new"` / `"gelu_pytorch_tanh"` → tanh approximation. Other
    /// values error loudly.
    pub fn from_json_str(s: &str) -> Result<Self> {
        use crate::config_json::{
            optional_f64, optional_hidden_act, optional_i64_or_none, parse_architectures,
            parse_id2label, parse_num_labels, required_i64,
        };
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let id2label = parse_id2label(&v)?;
        let num_labels = parse_num_labels(&v, id2label.as_deref());
        let architectures = parse_architectures(&v);
        Ok(BertConfig {
            vocab_size: required_i64(&v, "vocab_size")?,
            hidden_size: required_i64(&v, "hidden_size")?,
            num_hidden_layers: required_i64(&v, "num_hidden_layers")?,
            num_attention_heads: required_i64(&v, "num_attention_heads")?,
            intermediate_size: required_i64(&v, "intermediate_size")?,
            max_position_embeddings: required_i64(&v, "max_position_embeddings")?,
            type_vocab_size: required_i64(&v, "type_vocab_size")?,
            pad_token_id: optional_i64_or_none(&v, "pad_token_id"),
            layer_norm_eps: optional_f64(&v, "layer_norm_eps", 1e-12),
            hidden_dropout_prob: optional_f64(&v, "hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob: optional_f64(&v, "attention_probs_dropout_prob", 0.1),
            hidden_act: optional_hidden_act(&v, "hidden_act", "gelu")?,
            num_labels,
            id2label,
            architectures,
        })
    }

    /// Replace the `architectures` field with `[arch_class]` and return
    /// `self`. Used by every `from_pretrained*` to pin the source-config
    /// sidecar to the class actually built, so a subsequent
    /// `save_checkpoint` → `--checkpoint` re-export round-trips through
    /// `classify_architecture` (private to `crate::export`) regardless of what the
    /// upstream Hub config advertised (e.g. `bert-base-uncased` ships
    /// `architectures: ["BertForPreTraining"]` but a user loading via
    /// `BertForMaskedLM::from_pretrained` is building an MLM head and the
    /// sidecar should reflect that).
    pub fn with_architectures(mut self, arch_class: &str) -> Self {
        self.architectures = Some(vec![arch_class.to_string()]);
        self
    }

    /// Serialize to a HuggingFace-style `config.json` string.
    ///
    /// Inverse of [`Self::from_json_str`]: the emitted JSON round-trips
    /// back to an equal `BertConfig` on every shape-affecting field.
    /// Includes `model_type: "bert"` + `architectures: ["BertModel"]` so
    /// HF `AutoConfig` / `AutoModel` can dispatch without extra hints.
    ///
    /// Intended for the `fdl flodl-hf export` path — pair with
    /// [`safetensors_io::save_safetensors_file_from_graph`](crate::safetensors_io::save_safetensors_file_from_graph)
    /// to produce a directory HF Python can load directly.
    pub fn to_json_str(&self) -> String {
        use crate::config_json::{emit_architectures, emit_hidden_act, emit_id2label};
        let mut m = serde_json::Map::new();
        m.insert("model_type".into(), "bert".into());
        m.insert(
            "architectures".into(),
            emit_architectures(self.architectures.as_deref(), "BertModel"),
        );
        m.insert("vocab_size".into(), self.vocab_size.into());
        m.insert("hidden_size".into(), self.hidden_size.into());
        m.insert("num_hidden_layers".into(), self.num_hidden_layers.into());
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

// ── BertEmbeddings ───────────────────────────────────────────────────────

/// Token + position + token-type embeddings with post-LN and Dropout.
///
/// Implements [`NamedInputModule`] so the graph can feed `position_ids` and
/// `token_type_ids` alongside the main `input_ids` stream via
/// `FlowBuilder::using(&["position_ids", "token_type_ids"])`.
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertEmbeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                device,
            )?,
            position_embeddings: Embedding::on_device(
                config.max_position_embeddings,
                config.hidden_size,
                device,
            )?,
            token_type_embeddings: Embedding::on_device(
                config.type_vocab_size,
                config.hidden_size,
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

    /// Clone the word-embedding weight `Parameter` for weight tying.
    ///
    /// The returned `Parameter` shares its underlying `Variable` (and the
    /// C++ tensor) with the embedding table by `Rc`. Feed it to
    /// [`Linear::from_shared_weight`] when building an MLM / LM output
    /// head — gradients from both paths accumulate on the same leaf, and
    /// `Graph::named_parameters()` deduplicates by pointer identity, so
    /// the tied weight surfaces once under
    /// `bert.embeddings.word_embeddings.weight` (the first-visited tag).
    ///
    /// Call this **before** moving the embeddings into the backbone's
    /// `FlowBuilder`, since `.through(...)` consumes ownership.
    pub fn word_embeddings_weight(&self) -> Parameter {
        self.word_embeddings.weight.clone()
    }
}

impl Module for BertEmbeddings {
    fn name(&self) -> &str {
        "bert_embeddings"
    }

    /// Single-input forward path: word ids only. Position and token-type
    /// embeddings are skipped, which is useful for narrow unit tests but
    /// does NOT produce HF-equivalent outputs. The graph drives the full
    /// three-input path via `forward_named`.
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

impl NamedInputModule for BertEmbeddings {
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

// ── BertPooler ───────────────────────────────────────────────────────────

/// Pooler: take the `[CLS]` token (index 0 along the sequence axis), pass
/// through a learned dense layer, then tanh.
///
/// Input shape: `[batch, seq_len, hidden]`. Output shape: `[batch, hidden]`.
pub struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertPooler {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
        })
    }
}

impl Module for BertPooler {
    fn name(&self) -> &str {
        "bert_pooler"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [batch, seq_len, hidden] → take index 0 along seq axis.
        let cls = input.select(1, 0)?; // [batch, hidden]
        let pooled = self.dense.forward(&cls)?;
        pooled.tanh()
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }
}

// ── BertPredictionHeadTransform ──────────────────────────────────────────

/// The two-layer MLP that sits between the encoder output and the MLM
/// decoder: `Linear(hidden, hidden) → GELU → LayerNorm`. Shapes are
/// preserved end-to-end (`[B, S, H] → [B, S, H]`).
///
/// Parameter keys (post-`prefix_params` and node tag):
/// - `cls.predictions.transform.dense.{weight,bias}`
/// - `cls.predictions.transform.LayerNorm.{weight,bias}`
///
/// Matches HF Python's `BertPredictionHeadTransform`. Used exclusively
/// by [`BertForMaskedLM`]; kept as its own composite Module so the tied
/// decoder stays a clean single-node `.through()` afterwards.
pub struct BertPredictionHeadTransform {
    dense: Linear,
    activation: GELU,
    layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(BertPredictionHeadTransform {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            activation: GELU::with_approximate(config.hidden_act),
            layer_norm: LayerNorm::on_device_with_eps(
                config.hidden_size,
                config.layer_norm_eps,
                device,
            )?,
        })
    }
}

impl Module for BertPredictionHeadTransform {
    fn name(&self) -> &str {
        "bert_prediction_head_transform"
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

// ── BertModel ────────────────────────────────────────────────────────────

/// Translate a [`BertConfig`] into the subset [`TransformerLayer`]
/// consumes. Localizes the field-name mapping in one place.
fn bert_layer_config(config: &BertConfig) -> TransformerLayerConfig {
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

/// Assemble the BERT backbone onto a fresh [`FlowBuilder`], up to and
/// optionally including the pooler.
///
/// Shared by [`BertModel`] and the task-head constructors — HF's
/// `add_pooling_layer=False` shortcut is just this helper with
/// `with_pooler=false`. Task heads that operate on `last_hidden_state`
/// (token classification, question answering) drop the pooler; those
/// that operate on the `[CLS]` vector (sequence classification) keep
/// it. Callers can `through()` their own head modules onto the returned
/// builder and then `.build()`.
///
/// Graph shape: `bert.embeddings` → `bert.encoder.layer.{0..N-1}` →
/// (`bert.pooler`?). Four named inputs are pre-declared:
/// `input_ids` (implicit first), `position_ids`, `token_type_ids`,
/// `attention_mask`. Every encoder layer pulls `attention_mask` via
/// `.using()`.
fn bert_backbone_flow(
    config: &BertConfig,
    device: Device,
    with_pooler: bool,
) -> Result<FlowBuilder> {
    let mut fb = FlowBuilder::new()
        .input(&["position_ids", "token_type_ids", "attention_mask"])
        .through(BertEmbeddings::on_device(config, device)?)
        .tag("bert.embeddings")
        .using(&["position_ids", "token_type_ids"]);

    let layer_root = HfPath::new("bert").sub("encoder").sub("layer");
    let layer_cfg = bert_layer_config(config);
    for i in 0..config.num_hidden_layers {
        let tag = layer_root.sub(i).to_string();
        fb = fb
            .through(TransformerLayer::on_device(
                &layer_cfg,
                LayerNaming::BERT,
                device,
            )?)
            .tag(&tag)
            .using(&["attention_mask"]);
    }
    if with_pooler {
        fb = fb
            .through(BertPooler::on_device(config, device)?)
            .tag("bert.pooler");
    }
    Ok(fb)
}

/// Assembled BERT graph.
///
/// The returned [`Graph`] accepts four inputs via `forward_multi`, in
/// declaration order:
///
/// 1. `input_ids` (i64, shape `[batch, seq_len]`)
/// 2. `position_ids` (i64, shape `[batch, seq_len]`)
/// 3. `token_type_ids` (i64, shape `[batch, seq_len]`)
/// 4. `attention_mask` (f32, shape `[batch, 1, 1, seq_len]`, additive —
///    build with [`build_extended_attention_mask`] from a plain
///    `[batch, seq_len]` 0/1 mask)
///
/// Graph layout: `bert.embeddings` → `bert.encoder.layer.{0..N-1}` →
/// `bert.pooler`, where `N = config.num_hidden_layers`. Every encoder
/// layer pulls `attention_mask` via `.using()` so the same mask tensor
/// is shared across layers without re-materialising.
pub struct BertModel;

impl BertModel {
    /// Build a BERT graph on CPU.
    pub fn build(config: &BertConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build a BERT graph on `device`. Includes the pooler node; the
    /// returned graph emits `pooler_output` (`[batch, hidden]`).
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Graph> {
        bert_backbone_flow(config, device, true)?.build()
    }

    /// Build a BERT graph on `device` *without* the pooler. The returned
    /// graph emits `last_hidden_state` (`[batch, seq_len, hidden]`) —
    /// the shape the token-classification and question-answering heads
    /// consume. Matches HF Python's `BertModel(config, add_pooling_layer=False)`.
    pub fn on_device_without_pooler(config: &BertConfig, device: Device) -> Result<Graph> {
        bert_backbone_flow(config, device, false)?.build()
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

pub use crate::task_heads::{Answer, TokenPrediction};
use crate::task_heads::{
    ClassificationHead, EncoderInputs, MaskedLmHead, QaHead, TaggingHead, check_num_labels,
};

/// BERT graphs take four `forward_multi` inputs — `input_ids`,
/// `position_ids`, `token_type_ids`, and an extended attention mask —
/// in that order. The backbone flow is built with
/// `.input(&["position_ids", "token_type_ids", "attention_mask"])` so
/// `input_ids` flows in via `.through(embeddings)` as the first arg.
#[cfg(feature = "tokenizer")]
impl EncoderInputs for BertConfig {
    const FAMILY_NAME: &'static str = "Bert";
    const MASK_TOKEN: &'static str = "[MASK]";

    fn encoder_inputs(enc: &crate::tokenizer::EncodedBatch) -> Result<Vec<Variable>> {
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(build_extended_attention_mask(&mask_f32)?, false);
        Ok(vec![
            enc.input_ids.clone(),
            enc.position_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])
    }
}

/// BERT with a sequence-classification head on top of the pooled
/// `[CLS]` output: `pooler_output → Dropout → Linear(hidden, num_labels)`.
///
/// Parameter keys for the head:
/// - `classifier.weight`  (`[num_labels, hidden]`)
/// - `classifier.bias`    (`[num_labels]`)
///
/// Matches HF Python's
/// [`BertForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification).
/// Pre-trained checkpoints: `nateraw/bert-base-uncased-emotion` (6 emotions,
/// requires `fdl flodl-hf convert` first for `.bin`-only repos),
/// `nlptown/bert-base-multilingual-uncased-sentiment` (5-star rating),
/// `unitary/toxic-bert` (6-label toxicity).
///
/// Type alias over the generic [`ClassificationHead`]; `predict`,
/// `classify`, `forward_encoded`, `compute_loss`, `labels`, `graph`,
/// `config`, and `with_tokenizer` are inherited from there. Only the
/// BERT-specific `on_device` constructor lives below.
pub type BertForSequenceClassification = ClassificationHead<BertConfig>;

impl ClassificationHead<BertConfig> {
    /// Build the full graph (backbone + classifier head) on `device`
    /// without loading any weights. `num_labels` determines the head's
    /// output dimension; `id2label` falls back to `["LABEL_0", ...]`.
    pub fn on_device(config: &BertConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = bert_backbone_flow(config, device, /*with_pooler=*/ true)?
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

    /// Resolve `num_labels` from config if present; error otherwise.
    /// Used by `from_pretrained` paths where the config must carry
    /// head metadata.
    pub(crate) fn num_labels_from_config(config: &BertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "BertForSequenceClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// BERT with a per-token classification head: `last_hidden_state →
/// Dropout → Linear(hidden, num_labels)`. Typical use is NER, POS
/// tagging, or any sequence labelling task.
///
/// Parameter keys for the head:
/// - `classifier.weight`  (`[num_labels, hidden]`)
/// - `classifier.bias`    (`[num_labels]`)
///
/// Matches HF Python's `BertForTokenClassification`. Pre-trained
/// checkpoints: `dslim/bert-base-NER`,
/// `dbmdz/bert-large-cased-finetuned-conll03-english`, etc.
/// Type alias over the generic [`TaggingHead`]; all per-token
/// machinery (`tag`, `predict`, `forward_encoded`, `compute_loss`,
/// `labels`, `graph`, `config`, `with_tokenizer`) is inherited. Only
/// the BERT-specific `on_device` constructor lives below.
pub type BertForTokenClassification = TaggingHead<BertConfig>;

impl TaggingHead<BertConfig> {
    /// Build the full graph (backbone without pooler + classifier head).
    pub fn on_device(config: &BertConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = bert_backbone_flow(config, device, /*with_pooler=*/ false)?
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

    pub(crate) fn num_labels_from_config(config: &BertConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "BertForTokenClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// BERT with an extractive question-answering head: `last_hidden_state →
/// Linear(hidden, 2)` splitting into `start_logits` and `end_logits`.
///
/// Parameter keys for the head:
/// - `qa_outputs.weight` (`[2, hidden]`)
/// - `qa_outputs.bias`   (`[2]`)
///
/// Matches HF Python's `BertForQuestionAnswering`. Pre-trained
/// checkpoints: `csarron/bert-base-uncased-squad-v1`,
/// `bert-large-uncased-whole-word-masking-finetuned-squad`, etc.
/// Type alias over the generic [`QaHead`]; span-extraction logic
/// (`answer`, `answer_batch`, `extract`, `forward_encoded`,
/// `compute_loss`, `graph`, `config`, `with_tokenizer`) is inherited.
/// Only the BERT-specific `on_device` constructor lives below.
pub type BertForQuestionAnswering = QaHead<BertConfig>;

impl QaHead<BertConfig> {
    /// Build the full graph (backbone without pooler + QA output head).
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        // QA is a fixed-width head: 2 outputs (start, end), independent
        // of num_labels. Hardcoding it here matches HF Python.
        let graph = bert_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(Linear::on_device(config.hidden_size, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self::from_graph(graph, config))
    }
}

/// BERT with a masked-language-modelling head: prediction-head
/// transform (`Linear → GELU → LayerNorm`) followed by a decoder
/// `Linear(hidden, vocab_size)` whose weight is **tied** to
/// `bert.embeddings.word_embeddings.weight`.
///
/// Primary use case: **continued pretraining / domain adaptation** on
/// private corpora. Callers feed masked `input_ids` (with `[MASK]`
/// tokens at chosen positions) and labels shaped `[batch, seq_len]`
/// where the loss-relevant positions carry the original token id and
/// everything else is `-100`. See [`crate::task_heads::masked_lm_loss`].
///
/// Parameter keys emitted by the graph (post-dedup):
/// - `cls.predictions.transform.dense.{weight,bias}`
/// - `cls.predictions.transform.LayerNorm.{weight,bias}`
/// - `cls.predictions.decoder.bias`  (`[vocab_size]`, fresh)
///
/// `cls.predictions.decoder.weight` is **absent** from the state_dict —
/// the decoder borrows `bert.embeddings.word_embeddings.weight` via
/// [`Linear::from_shared_weight`], and `Graph::named_parameters()`
/// dedupes shared parameters by pointer identity. This matches HF's
/// runtime `tie_weights()` semantics (one tensor, two uses, one
/// optimizer update) while avoiding HF Python's historical quirk of
/// saving both keys redundantly. Safetensors loaders built against
/// this head should accept the HF "both keys present" layout too,
/// silently ignoring `decoder.weight` when the config carries
/// `tie_word_embeddings=true`.
///
/// Matches HF Python's
/// [`BertForMaskedLM`](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM).
/// Pre-trained checkpoints ship with `bert-base-uncased` et al. out of
/// the box; for inference fill-mask demos, reach for
/// `bert-base-uncased` or `bert-base-cased`.
/// Type alias over the generic [`MaskedLmHead`]; `fill_mask`,
/// `forward_encoded`, `compute_loss`, `graph`, `config`, and
/// `with_tokenizer` are inherited. Only the BERT-specific `on_device`
/// constructor lives below.
pub type BertForMaskedLM = MaskedLmHead<BertConfig>;

impl MaskedLmHead<BertConfig> {
    /// Build the full graph: backbone (without pooler) + transform +
    /// tied decoder. Initializes all weights fresh; use
    /// [`from_pretrained`](crate::models::bert::BertForMaskedLM::from_pretrained)
    /// to load a checkpoint.
    pub fn on_device(config: &BertConfig, device: Device) -> Result<Self> {
        // Build embeddings first, grab the tied weight before ownership
        // moves into the flow's `.through(...)`.
        let embeddings = BertEmbeddings::on_device(config, device)?;
        let tied_weight = embeddings.word_embeddings_weight();

        let mut fb = FlowBuilder::new()
            .input(&["position_ids", "token_type_ids", "attention_mask"])
            .through(embeddings)
            .tag("bert.embeddings")
            .using(&["position_ids", "token_type_ids"]);

        let layer_root = HfPath::new("bert").sub("encoder").sub("layer");
        let layer_cfg = bert_layer_config(config);
        for i in 0..config.num_hidden_layers {
            let tag = layer_root.sub(i).to_string();
            fb = fb
                .through(TransformerLayer::on_device(
                    &layer_cfg,
                    LayerNaming::BERT,
                    device,
                )?)
                .tag(&tag)
                .using(&["attention_mask"]);
        }

        // MLM prediction head: transform stack → tied decoder.
        // The decoder borrows `tied_weight` (shared Rc); its bias is a
        // fresh `[vocab_size]` Parameter initialised to zero (HF default).
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
            .through(BertPredictionHeadTransform::on_device(config, device)?)
            .tag("cls.predictions.transform")
            .through(Linear::from_shared_weight(tied_weight, Some(decoder_bias)))
            .tag("cls.predictions.decoder")
            .build()?;

        Ok(Self::from_graph(graph, config))
    }
}

#[cfg(test)]
#[path = "bert_tests.rs"]
mod tests;
