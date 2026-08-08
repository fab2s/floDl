//! RoBERTa encoder, compatible with HuggingFace `FacebookAI/roberta-base` checkpoints.
//!
//! Same encoder shape as [BERT](crate::models::bert) — a stack of
//! pre-LN-free self-attention + feed-forward blocks — with four
//! load-bearing deltas from the BERT port:
//!
//! 1. **Position ids are computed internally** from `input_ids` using
//!    the RoBERTa convention: `padding_idx + cumsum(non-pad-mask) *
//!    non-pad-mask`. Real (non-pad) tokens start at `padding_idx + 1`,
//!    padding sits at `padding_idx`. Matches HF Python's
//!    `RobertaModel.forward`. Callers pass `[input_ids, token_type_ids,
//!    attention_mask]` into the graph's `forward_multi` — no position
//!    ids in the signature.
//! 2. **LayerNorm ε defaults to 1e-5** (BERT's is 1e-12).
//! 3. **`type_vocab_size = 1`** — `token_type_ids` is always zero, but
//!    the embedding table exists and is loaded from the checkpoint for
//!    fidelity with HF.
//! 4. **`RobertaForSequenceClassification` uses a custom 2-layer head**
//!    on the `<s>` hidden state (keys
//!    `classifier.dense.*` + `classifier.out_proj.*`), not the
//!    BERT-style `pooler → Dropout → Linear`.
//!
//! The encoder layer itself is shared with the BERT and DistilBERT
//! ports via
//! [`LayerNaming::BERT`](crate::models::transformer_layer::LayerNaming::BERT)
//! — BERT and RoBERTa use the same `attention.self.{query,key,value}`
//! / `attention.output.dense` / `intermediate.dense` / `output.dense`
//! layout, so one [`crate::models::transformer_layer::TransformerLayer`]
//! serves both families. Task heads remain here because RoBERTa's
//! sequence-classification head diverges from BERT's.

use std::collections::HashMap;

use flodl::nn::{
    Dropout, Embedding, GELU, GeluApprox, LayerNorm, Linear, Module, NamedInputModule, Parameter,
};
use flodl::{
    DType, Device, FlowBuilder, Graph, Result, Tensor, TensorError, TensorOptions, Variable,
};

use crate::models::transformer_layer::{LayerNaming, TransformerLayer, TransformerLayerConfig};
use crate::path::{HfPath, prefix_params};

/// RoBERTa hyperparameters. Matches the fields of a HuggingFace
/// `RobertaConfig` JSON file that affect model shape.
///
/// Use [`RobertaConfig::roberta_base`] for the standard 12-layer /
/// 768-dim preset.
#[derive(Debug, Clone)]
pub struct RobertaConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    /// Padding token index. RoBERTa uses `1` (for `<pad>`). Drives both
    /// the word-embedding padding row AND the position-id computation —
    /// real tokens start at `padding_idx + 1` in the position embedding
    /// lookup.
    pub pad_token_id: i64,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    /// FFN activation form (parsed from HF `hidden_act`). Default
    /// `GeluApprox::Exact` (erf form) matches `FacebookAI/roberta-base`.
    pub hidden_act: GeluApprox,
    /// See [`crate::models::bert::BertConfig::num_labels`].
    pub num_labels: Option<i64>,
    /// See [`crate::models::bert::BertConfig::id2label`].
    pub id2label: Option<Vec<String>>,
    /// See [`crate::models::bert::BertConfig::architectures`].
    pub architectures: Option<Vec<String>>,
}

impl RobertaConfig {
    /// Preset matching `FacebookAI/roberta-base` on the HuggingFace Hub.
    pub fn roberta_base() -> Self {
        RobertaConfig {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            // RoBERTa reserves position rows `[0 .. padding_idx]` so the
            // effective max sequence length is `514 - 2 = 512` real tokens.
            max_position_embeddings: 514,
            type_vocab_size: 1,
            pad_token_id: 1,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            hidden_act: GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Parse a HuggingFace-style `config.json` string into a [`RobertaConfig`].
    ///
    /// Reads the fields that affect model shape plus the task-head
    /// metadata (`num_labels`, `id2label`) used by
    /// [`RobertaForSequenceClassification`] /
    /// [`RobertaForTokenClassification`] /
    /// [`RobertaForQuestionAnswering`]. Unknown fields are ignored.
    ///
    /// Defaults mirror HF's `RobertaConfig`:
    /// `layer_norm_eps = 1e-5`, `type_vocab_size = 1`, `pad_token_id = 1`,
    /// dropout probabilities `0.1`.
    pub fn from_json_str(s: &str) -> Result<Self> {
        use crate::config_json::{
            optional_f64, optional_hidden_act, optional_i64, parse_architectures, parse_id2label,
            parse_num_labels, required_i64,
        };
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let id2label = parse_id2label(&v)?;
        let num_labels = parse_num_labels(&v, id2label.as_deref());
        let architectures = parse_architectures(&v);
        Ok(RobertaConfig {
            vocab_size: required_i64(&v, "vocab_size")?,
            hidden_size: required_i64(&v, "hidden_size")?,
            num_hidden_layers: required_i64(&v, "num_hidden_layers")?,
            num_attention_heads: required_i64(&v, "num_attention_heads")?,
            intermediate_size: required_i64(&v, "intermediate_size")?,
            max_position_embeddings: required_i64(&v, "max_position_embeddings")?,
            type_vocab_size: optional_i64(&v, "type_vocab_size", 1),
            pad_token_id: optional_i64(&v, "pad_token_id", 1),
            layer_norm_eps: optional_f64(&v, "layer_norm_eps", 1e-5),
            hidden_dropout_prob: optional_f64(&v, "hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob: optional_f64(&v, "attention_probs_dropout_prob", 0.1),
            hidden_act: optional_hidden_act(&v, "hidden_act", "gelu")?,
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
    /// Inverse of [`Self::from_json_str`]. Emits `model_type: "roberta"`
    /// and `architectures: ["RobertaModel"]` for HF `AutoConfig`
    /// dispatch, plus every field the parser reads so a round-trip is
    /// lossless on shape-affecting state.
    pub fn to_json_str(&self) -> String {
        use crate::config_json::{emit_architectures, emit_hidden_act, emit_id2label};
        let mut m = serde_json::Map::new();
        m.insert("model_type".into(), "roberta".into());
        m.insert(
            "architectures".into(),
            emit_architectures(self.architectures.as_deref(), "RobertaModel"),
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
        m.insert("pad_token_id".into(), self.pad_token_id.into());
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

// ── RobertaEmbeddings ────────────────────────────────────────────────────

/// Token + position + token-type embeddings with post-LN and Dropout.
///
/// Differs from [`BertEmbeddings`](crate::models::bert::BertEmbeddings)
/// in one crucial way: `position_ids` are computed internally from
/// `input_ids` rather than read from a graph input. The formula
/// matches HF Python's
/// `create_position_ids_from_input_ids`:
///
/// ```text
/// mask       = (input_ids != padding_idx).long()   // [B, S]
/// positions  = cumsum(mask, dim=1) * mask + padding_idx
/// ```
///
/// Real tokens land at indices `padding_idx + 1, padding_idx + 2, …`.
/// Padding slots stay at `padding_idx`. This mirrors how HF fine-tuned
/// every public RoBERTa checkpoint, so safetensors weights line up
/// directly.
pub struct RobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    padding_idx: i64,
}

impl RobertaEmbeddings {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        Ok(RobertaEmbeddings {
            word_embeddings: Embedding::on_device_with_padding_idx(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id),
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
            padding_idx: config.pad_token_id,
        })
    }

    /// Clone the word-embedding weight `Parameter` for weight tying.
    ///
    /// See [`crate::models::bert::BertEmbeddings::word_embeddings_weight`]
    /// for the full contract. In short: the returned `Parameter` shares
    /// the underlying `Variable` by `Rc`, gradients accumulate on the
    /// single leaf, and `Graph::named_parameters()` surfaces the tied
    /// weight once under the first-visited tag (which for RoBERTa MLM is
    /// `roberta.embeddings.word_embeddings.weight`).
    ///
    /// Call this **before** moving the embeddings into the backbone's
    /// `FlowBuilder`, since `.through(...)` consumes ownership.
    pub fn word_embeddings_weight(&self) -> Parameter {
        self.word_embeddings.weight.clone()
    }

    /// Compute RoBERTa-style position ids from `input_ids`.
    ///
    /// Runs entirely on raw `Tensor`s (no autograd) — position ids are
    /// integer indices into the embedding table, so they don't need to
    /// participate in backward.
    fn position_ids_from_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Float mask in {0.0, 1.0}; cumsum along seq dim stays integer-valued
        // in f32 for all realistic sequence lengths (f32 holds exact integers
        // up to 2^24, safely above any tokenizer truncation).
        let mask = input_ids.ne_scalar(self.padding_idx as f64)?;
        let cum = mask.cumsum(1)?;
        cum.mul(&mask)?
            .add_scalar(self.padding_idx as f64)?
            .to_dtype(DType::Int64)
    }
}

impl Module for RobertaEmbeddings {
    fn name(&self) -> &str {
        "roberta_embeddings"
    }

    /// Single-input forward: word ids only. Position ids are computed
    /// internally; token-type embedding adds a zero-row contribution
    /// (`type_vocab_size = 1` means all tokens map to row 0).
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let pos_ids = self.position_ids_from_input_ids(&input.data())?;
        let pos_var = Variable::new(pos_ids, false);
        let word = self.word_embeddings.forward(input)?;
        let pe = self.position_embeddings.forward(&pos_var)?;
        let summed = word.add(&pe)?;
        let ln = self.layer_norm.forward(&summed)?;
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

impl NamedInputModule for RobertaEmbeddings {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let pos_ids = self.position_ids_from_input_ids(&input.data())?;
        let pos_var = Variable::new(pos_ids, false);

        let word = self.word_embeddings.forward(input)?;
        let pe = self.position_embeddings.forward(&pos_var)?;
        let mut summed = word.add(&pe)?;
        if let Some(tt) = refs.get("token_type_ids") {
            let te = self.token_type_embeddings.forward(tt)?;
            summed = summed.add(&te)?;
        }
        let ln = self.layer_norm.forward(&summed)?;
        self.dropout.forward(&ln)
    }
}

// ── RobertaPooler ────────────────────────────────────────────────────────

/// Pooler: take the `<s>` (index 0) hidden state, project, then tanh.
///
/// Structurally identical to `BertPooler`. Most RoBERTa fine-tunes
/// don't use the pooler (the sequence-classification head does its own
/// two-layer projection on the `<s>` state), but the pooler weights
/// are still published with the base `FacebookAI/roberta-base` checkpoint, so the
/// backbone must be able to emit a pooled output when asked.
pub struct RobertaPooler {
    dense: Linear,
}

impl RobertaPooler {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        Ok(RobertaPooler {
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
        })
    }
}

impl Module for RobertaPooler {
    fn name(&self) -> &str {
        "roberta_pooler"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let cls = input.select(1, 0)?;
        let pooled = self.dense.forward(&cls)?;
        pooled.tanh()
    }

    fn parameters(&self) -> Vec<Parameter> {
        prefix_params("dense", self.dense.parameters())
    }
}

// ── RobertaModel ─────────────────────────────────────────────────────────

/// Translate a [`RobertaConfig`] into the subset [`TransformerLayer`]
/// consumes. BERT and RoBERTa share the encoder layer layout exactly;
/// only the embedding and task-head code differs.
pub(crate) fn roberta_layer_config(config: &RobertaConfig) -> TransformerLayerConfig {
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

/// Assemble the RoBERTa backbone onto a fresh [`FlowBuilder`], up to
/// and optionally including the pooler.
///
/// Graph signature is **3 named inputs** (no `position_ids`):
/// `input_ids` (implicit first), `token_type_ids`, `attention_mask`.
/// The embedding layer computes position ids internally from
/// `input_ids`.
///
/// Graph shape: `roberta.embeddings` →
/// `roberta.encoder.layer.{0..N-1}` → (`roberta.pooler`?).
///
/// Exposed `pub(crate)` so the sibling XLM-RoBERTa port can reuse the
/// exact builder — HF's XLM-R state_dict uses the `roberta.*` prefix
/// and shares this graph shape verbatim.
pub(crate) fn roberta_backbone_flow(
    config: &RobertaConfig,
    device: Device,
    with_pooler: bool,
) -> Result<FlowBuilder> {
    let mut fb = FlowBuilder::new()
        .input(&["token_type_ids", "attention_mask"])
        .through(RobertaEmbeddings::on_device(config, device)?)
        .tag("roberta.embeddings")
        .using(&["token_type_ids"]);

    let layer_root = HfPath::new("roberta").sub("encoder").sub("layer");
    let layer_cfg = roberta_layer_config(config);
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
            .through(RobertaPooler::on_device(config, device)?)
            .tag("roberta.pooler");
    }
    Ok(fb)
}

/// Assembled RoBERTa graph.
///
/// The returned [`Graph`] accepts **three** inputs via `forward_multi`,
/// in declaration order:
///
/// 1. `input_ids` (i64, shape `[batch, seq_len]`)
/// 2. `token_type_ids` (i64, shape `[batch, seq_len]`, all zeros for
///    most RoBERTa checkpoints)
/// 3. `attention_mask` (f32, shape `[batch, 1, 1, seq_len]`, additive
///    — build with
///    [`crate::models::bert::build_extended_attention_mask`]
///    from a plain `[batch, seq_len]` 0/1 mask)
///
/// Position ids are computed internally from `input_ids`.
///
/// The default Hub loader ([`RobertaModel::from_pretrained`]) builds
/// the graph *without* a pooler since `FacebookAI/roberta-base` and most
/// fine-tunes don't ship pooler weights (RoBERTa pretraining drops
/// NSP). Use [`RobertaModel::on_device`] explicitly when a specific
/// checkpoint is known to carry its own pooler.
pub struct RobertaModel;

impl RobertaModel {
    /// Build a RoBERTa graph on CPU with a pooler node.
    ///
    /// Prefer [`RobertaModel::on_device_without_pooler`] for the
    /// common case: `FacebookAI/roberta-base` and its fine-tunes skip the
    /// pooler, and the default [`from_pretrained`](Self::from_pretrained)
    /// path matches that convention.
    pub fn build(config: &RobertaConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build a RoBERTa graph on `device` with a pooler node. Emits
    /// `pooler_output` (`[batch, hidden]`).
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Graph> {
        roberta_backbone_flow(config, device, true)?.build()
    }

    /// Build a RoBERTa graph on `device` *without* the pooler. Emits
    /// `last_hidden_state` (`[batch, seq_len, hidden]`) — the shape
    /// every task head in this module consumes, and the shape
    /// [`from_pretrained`](Self::from_pretrained) loads into.
    pub fn on_device_without_pooler(config: &RobertaConfig, device: Device) -> Result<Graph> {
        roberta_backbone_flow(config, device, false)?.build()
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

pub use crate::task_heads::{Answer, TokenPrediction};
use crate::task_heads::{
    ClassificationHead, EncoderInputs, MaskedLmHead, QaHead, TaggingHead, check_num_labels,
};

/// RoBERTa graphs take three `forward_multi` inputs — `input_ids`,
/// `token_type_ids`, and an extended attention mask — in that order.
/// Position ids are computed inside the embedding layer (non-padding
/// cumulative count + `padding_idx + 1`), so they are not a graph
/// input. Token-type ids are accepted but typically all-zero — HF
/// matches this by keeping the type-embedding table and letting
/// every token index into row 0.
#[cfg(feature = "tokenizer")]
impl EncoderInputs for RobertaConfig {
    const FAMILY_NAME: &'static str = "Roberta";
    const MASK_TOKEN: &'static str = "<mask>";

    fn encoder_inputs(enc: &crate::tokenizer::EncodedBatch) -> Result<Vec<Variable>> {
        let mask_f32 = enc.attention_mask.data().to_dtype(DType::Float32)?;
        let mask = Variable::new(
            crate::models::bert::build_extended_attention_mask(&mask_f32)?,
            false,
        );
        Ok(vec![
            enc.input_ids.clone(),
            enc.token_type_ids.clone(),
            mask,
        ])
    }
}

// ── RobertaClassificationHead ────────────────────────────────────────────

/// The two-layer classification head HF Python calls
/// `RobertaClassificationHead`:
/// `Dropout → dense → tanh → Dropout → out_proj`, all applied to the
/// `<s>` hidden state (index 0 along the sequence axis).
///
/// Parameter keys (nested under the outer `classifier` tag):
/// - `classifier.dense.weight`    (`[hidden, hidden]`)
/// - `classifier.dense.bias`      (`[hidden]`)
/// - `classifier.out_proj.weight` (`[num_labels, hidden]`)
/// - `classifier.out_proj.bias`   (`[num_labels]`)
///
/// This replaces BERT's `pooler → Dropout → Linear(hidden, num_labels)`
/// pattern. RoBERTa fine-tunes typically ship with the pooler weights
/// too, but the SeqCls head ignores them — so when the backbone loads
/// here it runs *without* the pooler node.
pub struct RobertaClassificationHead {
    dropout: Dropout,
    dense: Linear,
    out_proj: Linear,
}

impl RobertaClassificationHead {
    pub fn on_device(config: &RobertaConfig, num_labels: i64, device: Device) -> Result<Self> {
        Ok(RobertaClassificationHead {
            dropout: Dropout::new(config.hidden_dropout_prob),
            dense: Linear::on_device(config.hidden_size, config.hidden_size, device)?,
            out_proj: Linear::on_device(config.hidden_size, num_labels, device)?,
        })
    }
}

impl Module for RobertaClassificationHead {
    fn name(&self) -> &str {
        "roberta_classification_head"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // input: [batch, seq_len, hidden] → take index 0 along seq axis.
        let cls = input.select(1, 0)?; // [batch, hidden]
        let x = self.dropout.forward(&cls)?;
        let x = self.dense.forward(&x)?;
        let x = x.tanh()?;
        let x = self.dropout.forward(&x)?;
        self.out_proj.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = Vec::new();
        out.extend(prefix_params("dense", self.dense.parameters()));
        out.extend(prefix_params("out_proj", self.out_proj.parameters()));
        out
    }

    fn set_training(&self, training: bool) {
        self.dropout.set_training(training);
    }
}

// ── RobertaForSequenceClassification ─────────────────────────────────────

/// RoBERTa with a sequence-classification head on the `<s>` hidden
/// state: a two-layer MLP with tanh activation and dropout on either
/// side (see [`RobertaClassificationHead`]).
///
/// Matches HF Python's `RobertaForSequenceClassification`.
/// Pre-trained checkpoints:
/// `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-label),
/// `roberta-large-mnli`, `SamLowe/roberta-base-go_emotions`.
/// Type alias over the generic [`ClassificationHead`]; `predict`,
/// `classify`, `forward_encoded`, `compute_loss`, `labels`, `graph`,
/// `config`, and `with_tokenizer` are inherited. Only the RoBERTa-
/// specific `on_device` constructor lives below.
pub type RobertaForSequenceClassification = ClassificationHead<RobertaConfig>;

impl ClassificationHead<RobertaConfig> {
    /// Build the full graph (backbone without pooler + two-layer
    /// classification head) on `device` without loading any weights.
    pub fn on_device(config: &RobertaConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = roberta_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(RobertaClassificationHead::on_device(
                config, num_labels, device,
            )?)
            .tag("classifier")
            .build()?;
        Ok(Self::from_graph(
            graph,
            config,
            num_labels,
            config.id2label.clone(),
        ))
    }

    pub(crate) fn num_labels_from_config(config: &RobertaConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "RobertaForSequenceClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

// ── RobertaForTokenClassification ────────────────────────────────────────

/// RoBERTa with a per-token classification head: `last_hidden_state →
/// Dropout → Linear(hidden, num_labels)`. Structurally identical to
/// [`BertForTokenClassification`](crate::models::bert::BertForTokenClassification).
///
/// Matches HF Python's `RobertaForTokenClassification`. Pre-trained
/// checkpoints: `Jean-Baptiste/roberta-large-ner-english`,
/// `obi/deid_roberta_i2b2`.
/// Type alias over the generic [`TaggingHead`]; all per-token
/// machinery is inherited. Only the RoBERTa-specific `on_device`
/// constructor lives below.
pub type RobertaForTokenClassification = TaggingHead<RobertaConfig>;

impl TaggingHead<RobertaConfig> {
    pub fn on_device(config: &RobertaConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let graph = roberta_backbone_flow(config, device, /*with_pooler=*/ false)?
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

    pub(crate) fn num_labels_from_config(config: &RobertaConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "RobertaForTokenClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

// ── RobertaForQuestionAnswering ──────────────────────────────────────────

/// RoBERTa with an extractive question-answering head:
/// `last_hidden_state → Linear(hidden, 2)` splitting into
/// `start_logits` and `end_logits`.
///
/// Parameter keys for the head:
/// - `qa_outputs.weight` (`[2, hidden]`)
/// - `qa_outputs.bias`   (`[2]`)
///
/// Matches HF Python's `RobertaForQuestionAnswering`. Pre-trained
/// checkpoints: `deepset/roberta-base-squad2`,
/// `csarron/roberta-base-squad-v1`.
/// Type alias over the generic [`QaHead`]; span-extraction and
/// forward machinery are inherited. Only the RoBERTa-specific
/// `on_device` constructor lives below.
pub type RobertaForQuestionAnswering = QaHead<RobertaConfig>;

impl QaHead<RobertaConfig> {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        let graph = roberta_backbone_flow(config, device, /*with_pooler=*/ false)?
            .through(Linear::on_device(config.hidden_size, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self::from_graph(graph, config))
    }
}

// ── RobertaLMHeadTransform ────────────────────────────────────────────────

/// Dense + GELU + LayerNorm stack that sits between the encoder output
/// and the MLM decoder in `RobertaLMHead`. Shapes preserved end-to-end
/// (`[B, S, H] → [B, S, H]`).
///
/// Parameter keys (post-`prefix_params` under the `lm_head` tag):
/// - `lm_head.dense.{weight,bias}`
/// - `lm_head.layer_norm.{weight,bias}`
///
/// Note the lowercase `layer_norm` — unlike BERT's `LayerNorm` inside
/// `cls.predictions.transform`, HF's `RobertaLMHead` spells it
/// lowercase. Matches HF Python's `RobertaLMHead`.
pub struct RobertaLMHeadTransform {
    dense: Linear,
    activation: GELU,
    layer_norm: LayerNorm,
}

impl RobertaLMHeadTransform {
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        Ok(RobertaLMHeadTransform {
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

impl Module for RobertaLMHeadTransform {
    fn name(&self) -> &str {
        "roberta_lm_head_transform"
    }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let x = self.dense.forward(input)?;
        let x = self.activation.forward(&x)?;
        self.layer_norm.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut out = prefix_params("dense", self.dense.parameters());
        out.extend(prefix_params("layer_norm", self.layer_norm.parameters()));
        out
    }
}

// ── RobertaForMaskedLM ───────────────────────────────────────────────────

/// RoBERTa with a masked-language-modelling head: `RobertaLMHead` =
/// `Linear → GELU → LayerNorm → tied-decoder Linear`, where the decoder
/// weight is tied to `roberta.embeddings.word_embeddings.weight`.
///
/// Primary use case: **continued pretraining / domain adaptation** on
/// private corpora. Callers feed masked `input_ids` and labels shaped
/// `[batch, seq_len]` where loss-relevant positions carry the original
/// token id and the rest is `-100`. See [`crate::task_heads::masked_lm_loss`].
///
/// Parameter keys emitted by the graph (post-dedup):
/// - `lm_head.dense.{weight,bias}`
/// - `lm_head.layer_norm.{weight,bias}`
/// - `lm_head.decoder.bias`  (`[vocab_size]`, fresh)
///
/// `lm_head.decoder.weight` is **absent** — the decoder borrows
/// `roberta.embeddings.word_embeddings.weight` via
/// [`Linear::from_shared_weight`], and `Graph::named_parameters()`
/// dedupes by pointer identity. HF Python's `RobertaLMHead` additionally
/// exposes a top-level `lm_head.bias` Parameter tied to
/// `decoder.bias`; we keep only the decoder's own bias and let the
/// safetensors loader ignore an extra `lm_head.bias` key when a
/// checkpoint carries one.
///
/// Matches HF Python's `RobertaForMaskedLM`. Canonical checkpoints:
/// `FacebookAI/roberta-base`, `roberta-large`, any `*-mlm` domain-adaptation
/// fine-tune.
/// Type alias over the generic [`MaskedLmHead`]; `fill_mask`,
/// `forward_encoded`, `compute_loss`, and tokenizer plumbing are
/// inherited. Only the RoBERTa-specific `on_device` constructor lives
/// below.
pub type RobertaForMaskedLM = MaskedLmHead<RobertaConfig>;

/// Build the full RoBERTa MLM graph: embeddings (with tied weight
/// captured) → encoder layers → LM-head transform → tied decoder.
///
/// Shared with the sibling XLM-RoBERTa port since HF's XLM-R state_dict
/// uses identical `roberta.*` keys and the same tied-decoder layout.
pub(crate) fn roberta_masked_lm_graph(config: &RobertaConfig, device: Device) -> Result<Graph> {
    // Build embeddings first, grab the tied weight before ownership
    // moves into `.through(...)`.
    let embeddings = RobertaEmbeddings::on_device(config, device)?;
    let tied_weight = embeddings.word_embeddings_weight();

    let mut fb = FlowBuilder::new()
        .input(&["token_type_ids", "attention_mask"])
        .through(embeddings)
        .tag("roberta.embeddings")
        .using(&["token_type_ids"]);

    let layer_root = HfPath::new("roberta").sub("encoder").sub("layer");
    let layer_cfg = roberta_layer_config(config);
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

    // LM head: transform stack → tied decoder with fresh [V] bias.
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
    fb.through(RobertaLMHeadTransform::on_device(config, device)?)
        .tag("lm_head")
        .through(Linear::from_shared_weight(tied_weight, Some(decoder_bias)))
        .tag("lm_head.decoder")
        .build()
}

impl MaskedLmHead<RobertaConfig> {
    /// Build the full graph: backbone (no pooler) + LM-head transform +
    /// tied decoder on `device`. Initializes all weights fresh; use
    /// [`from_pretrained`](crate::models::roberta::RobertaForMaskedLM::from_pretrained)
    /// to load a checkpoint.
    pub fn on_device(config: &RobertaConfig, device: Device) -> Result<Self> {
        let graph = roberta_masked_lm_graph(config, device)?;
        Ok(Self::from_graph(graph, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bert::build_extended_attention_mask;
    use crate::safetensors_io::expected_from_graph;
    use flodl::HasGraph;

    /// The 16 parameter keys every encoder layer exposes.
    fn expected_layer_keys(i: i64) -> Vec<String> {
        let suffixes = [
            "attention.output.LayerNorm.bias",
            "attention.output.LayerNorm.weight",
            "attention.output.dense.bias",
            "attention.output.dense.weight",
            "attention.self.key.bias",
            "attention.self.key.weight",
            "attention.self.query.bias",
            "attention.self.query.weight",
            "attention.self.value.bias",
            "attention.self.value.weight",
            "intermediate.dense.bias",
            "intermediate.dense.weight",
            "output.LayerNorm.bias",
            "output.LayerNorm.weight",
            "output.dense.bias",
            "output.dense.weight",
        ];
        suffixes
            .iter()
            .map(|s| format!("roberta.encoder.layer.{i}.{s}"))
            .collect()
    }

    /// Round-trip: preset -> to_json_str -> from_json_str recovers the
    /// same config. Guards against missing fields in the emitted JSON
    /// (parser errors on required_i64) and silent default drift.
    #[test]
    fn roberta_config_to_json_str_round_trip() {
        let preset = RobertaConfig::roberta_base();
        let s = preset.to_json_str();
        let recovered = RobertaConfig::from_json_str(&s).unwrap();
        assert_eq!(preset.to_json_str(), recovered.to_json_str());
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(
            v.get("model_type").and_then(|x| x.as_str()),
            Some("roberta")
        );
    }

    /// Full key set (199 keys: 5 embeddings + 16×12 layers + 2 pooler)
    /// matches HF safetensors dotted form exactly.
    #[test]
    fn roberta_parameter_keys_match_hf_dotted_form() {
        let config = RobertaConfig::roberta_base();
        let graph = RobertaModel::build(&config).unwrap();
        let expected = expected_from_graph(&graph);

        let mut keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        keys.sort();

        let mut want: Vec<String> = vec![
            "roberta.embeddings.LayerNorm.bias".into(),
            "roberta.embeddings.LayerNorm.weight".into(),
            "roberta.embeddings.position_embeddings.weight".into(),
            "roberta.embeddings.token_type_embeddings.weight".into(),
            "roberta.embeddings.word_embeddings.weight".into(),
        ];
        for i in 0..config.num_hidden_layers {
            want.extend(expected_layer_keys(i));
        }
        want.extend([
            "roberta.pooler.dense.bias".into(),
            "roberta.pooler.dense.weight".into(),
        ]);
        want.sort();

        assert_eq!(keys, want);
    }

    /// SeqCls head keys: the two-layer classifier ships
    /// `classifier.dense.*` + `classifier.out_proj.*`, NOT a single
    /// `classifier.{weight,bias}` like BERT.
    #[test]
    fn roberta_seqcls_head_has_two_layer_keys() {
        let config = RobertaConfig::roberta_base();
        let head = RobertaForSequenceClassification::on_device(&config, 3, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        assert!(keys.contains(&"classifier.dense.weight".to_string()));
        assert!(keys.contains(&"classifier.dense.bias".to_string()));
        assert!(keys.contains(&"classifier.out_proj.weight".to_string()));
        assert!(keys.contains(&"classifier.out_proj.bias".to_string()));
        // No single-linear `classifier.weight` at the top level.
        assert!(!keys.iter().any(|k| k == "classifier.weight"));
    }

    /// Position-id computation: real tokens start at `padding_idx + 1`
    /// (== 2 for `FacebookAI/roberta-base`), padding slots stay at `padding_idx`.
    #[test]
    fn roberta_position_ids_follow_hf_convention() {
        let config = RobertaConfig::roberta_base();
        let emb = RobertaEmbeddings::on_device(&config, Device::CPU).unwrap();
        // [<s>=0, real, real, </s>=2, <pad>=1, <pad>=1]  —  using
        // arbitrary non-pad ids for the real tokens.
        let ids = Tensor::from_i64(&[0, 100, 200, 2, 1, 1], &[1, 6], Device::CPU).unwrap();
        let pos = emb.position_ids_from_input_ids(&ids).unwrap();
        let flat: Vec<i64> = pos.to_i64_vec().unwrap();
        // Expected:
        //   mask        = [1, 1, 1, 1, 0, 0]
        //   cumsum*mask = [1, 2, 3, 4, 0, 0]
        //   + pad_idx=1 = [2, 3, 4, 5, 1, 1]
        assert_eq!(flat, vec![2, 3, 4, 5, 1, 1]);
    }

    // ── RobertaForMaskedLM ──────────────────────────────────────────

    /// `RobertaForMaskedLM` weight-ties its decoder. State_dict must
    /// carry `roberta.embeddings.word_embeddings.weight` but **not**
    /// `lm_head.decoder.weight`; the LM-head transform + fresh bias
    /// show up under `lm_head.{dense,layer_norm,decoder.bias}`.
    #[test]
    fn masked_lm_parameter_keys_match_hf_tied_layout() {
        let config = RobertaConfig::roberta_base();
        let head = RobertaForMaskedLM::on_device(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert!(
            keys.contains(&"roberta.embeddings.word_embeddings.weight"),
            "tied weight must surface under embeddings tag: {keys:?}",
        );
        assert!(
            !keys.contains(&"lm_head.decoder.weight"),
            "decoder.weight must be absent (tied, dedup kept embeddings entry)",
        );
        assert!(
            !keys.iter().any(|k| k.starts_with("roberta.pooler.")),
            "MLM must not carry pooler params",
        );

        let mut head_keys: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| k.starts_with("lm_head."))
            .collect();
        head_keys.sort();
        assert_eq!(
            head_keys,
            vec![
                "lm_head.decoder.bias",
                "lm_head.dense.bias",
                "lm_head.dense.weight",
                "lm_head.layer_norm.bias",
                "lm_head.layer_norm.weight",
            ],
        );

        let by_key: std::collections::HashMap<&str, &[i64]> = expected
            .iter()
            .map(|p| (p.key.as_str(), p.shape.as_slice()))
            .collect();
        let v = config.vocab_size;
        let h = config.hidden_size;
        assert_eq!(by_key["roberta.embeddings.word_embeddings.weight"], &[v, h]);
        assert_eq!(by_key["lm_head.dense.weight"], &[h, h]);
        assert_eq!(by_key["lm_head.dense.bias"], &[h]);
        assert_eq!(by_key["lm_head.layer_norm.weight"], &[h]);
        assert_eq!(by_key["lm_head.layer_norm.bias"], &[h]);
        assert_eq!(by_key["lm_head.decoder.bias"], &[v]);
    }

    /// Structural tying check: exactly one `[vocab, hidden]`-shaped
    /// Parameter in the graph. Uses `roberta_base` so the shape is
    /// unambiguous against the FFN dimensions.
    #[test]
    fn masked_lm_decoder_shares_embedding_rc() {
        let config = RobertaConfig::roberta_base();
        let head = RobertaForMaskedLM::on_device(&config, Device::CPU).unwrap();

        let named = head.graph().named_parameters();
        let embed_w = named
            .iter()
            .find(|(k, _)| k == "roberta.embeddings/word_embeddings.weight")
            .map(|(_, p)| p.clone())
            .expect("embeddings word_embeddings.weight must be present");
        assert_eq!(
            embed_w.variable.shape(),
            vec![config.vocab_size, config.hidden_size],
        );

        let vocab_shaped_count = named
            .iter()
            .filter(|(_, p)| p.variable.shape() == vec![config.vocab_size, config.hidden_size])
            .count();
        assert_eq!(
            vocab_shaped_count, 1,
            "exactly one [V, H]-shaped Parameter expected under tying",
        );
    }

    /// Smoke: RoBERTa MLM head emits `[batch, seq, vocab_size]` logits.
    /// Uses the full `roberta_base` config; seq_len and batch kept tiny
    /// so the forward pass stays cheap.
    #[test]
    fn masked_lm_forward_shape_smoke() {
        let config = RobertaConfig::roberta_base();
        let dev = Device::CPU;
        let head = RobertaForMaskedLM::on_device(&config, dev).unwrap();
        head.graph().eval();

        let batch = 1;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[0, 100, 200, 2], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(
            Tensor::from_i64(&[0; 4], &[batch, seq], dev).unwrap(),
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
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = head.graph().forward_multi(&[ids, tt, mask]).unwrap();
        assert_eq!(out.shape(), vec![batch, seq, config.vocab_size]);
    }

    /// `HasGraph` impl points to the MLM head's inner graph by reference.
    #[test]
    fn masked_lm_has_graph_returns_inner_graph_by_reference() {
        let config = RobertaConfig::roberta_base();
        let head = RobertaForMaskedLM::on_device(&config, Device::CPU).unwrap();
        assert!(std::ptr::eq(
            head.graph(),
            <RobertaForMaskedLM as HasGraph>::graph(&head)
        ));
    }

    /// Backward through the tied decoder must produce a gradient on
    /// the shared embedding weight.
    #[test]
    fn masked_lm_backward_accumulates_on_tied_weight() {
        let config = RobertaConfig::roberta_base();
        let dev = Device::CPU;
        let head = RobertaForMaskedLM::on_device(&config, dev).unwrap();
        head.graph().train();

        let batch = 1;
        let seq = 4;
        let ids = Variable::new(
            Tensor::from_i64(&[0, 100, 200, 2], &[batch, seq], dev).unwrap(),
            false,
        );
        let tt = Variable::new(
            Tensor::from_i64(&[0; 4], &[batch, seq], dev).unwrap(),
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
        let mask = Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let logits = head.graph().forward_multi(&[ids, tt, mask]).unwrap();
        let loss = logits.sum().unwrap();
        loss.backward().unwrap();

        let named = head.graph().named_parameters();
        let embed_w = named
            .iter()
            .find(|(k, _)| k == "roberta.embeddings/word_embeddings.weight")
            .map(|(_, p)| p.clone())
            .expect("tied weight must be present");
        assert!(
            embed_w.variable.grad().is_some(),
            "tied embedding/decoder weight must receive gradient",
        );
    }
}
