//! XLM-RoBERTa encoder, compatible with HF `xlm-roberta-base` checkpoints.
//!
//! Architecturally identical to [RoBERTa](crate::models::roberta):
//! same encoder layers, same state_dict prefix (`roberta.*`), same
//! tied-decoder MLM head, same position-id convention. HF's
//! `XLMRobertaModel` subclasses `RobertaModel` without structural
//! changes. The two genuine deltas from a framework-integration
//! perspective are:
//!
//! 1. **Tokenizer** — SentencePiece over a ~250k multilingual
//!    vocabulary, vs RoBERTa's BPE over 50k English-only.
//!    [`crate::tokenizer::HfTokenizer::from_pretrained`] handles both
//!    transparently; from the model's perspective, `input_ids` are
//!    `input_ids`.
//! 2. **`model_type: "xlm-roberta"`** in `config.json`, so
//!    [`AutoConfig`](crate::models::auto::AutoConfig) dispatches
//!    correctly.
//!
//! Because the state_dict prefix stays `roberta.*`, XLM-R graphs are
//! built with `roberta.*` tags — the backbone and MLM builders here
//! delegate to the corresponding `roberta.rs` helpers after a trivial
//! [`From<&XlmRobertaConfig>`](XlmRobertaConfig) for
//! [`crate::models::roberta::RobertaConfig`] conversion.
//! A loaded safetensors checkpoint lines up directly without any key
//! renaming.
//!
//! XLM-RoBERTa exists as a distinct module here, rather than a type
//! alias, for HF parity: callers reach for `XlmRobertaConfig`,
//! `XlmRobertaModel`, `XlmRobertaForSequenceClassification`, etc. by
//! the names they already know from HF Python.

use flodl::nn::{Dropout, GeluApprox, Linear};
use flodl::{DType, Device, Graph, Result, TensorError, Variable};

use crate::models::roberta::{
    RobertaClassificationHead, RobertaConfig, RobertaModel, roberta_backbone_flow,
    roberta_masked_lm_graph,
};

/// XLM-RoBERTa hyperparameters.
///
/// Field layout mirrors [`RobertaConfig`] exactly — HF's
/// `XLMRobertaConfig` adds no shape-affecting fields of its own. Kept
/// as a distinct struct (rather than a type alias) so
/// [`AutoConfig`](crate::models::auto::AutoConfig) can carry the HF
/// `model_type: "xlm-roberta"` signal through the Rust type system,
/// and so XLM-R-only features can grow here without churning
/// [`RobertaConfig`].
///
/// Use [`XlmRobertaConfig::xlm_roberta_base`] for the standard
/// 12-layer / 768-dim preset matching `xlm-roberta-base` on the
/// HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct XlmRobertaConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub num_attention_heads: i64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub type_vocab_size: i64,
    /// Padding token index. XLM-RoBERTa uses `1` (for `<pad>`), matching
    /// RoBERTa. Drives both the word-embedding padding row AND the
    /// position-id computation inside [`RobertaEmbeddings`](crate::models::roberta::RobertaEmbeddings).
    pub pad_token_id: i64,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    /// FFN activation form (parsed from HF `hidden_act`). Default
    /// `GeluApprox::Exact` (erf form) matches `xlm-roberta-base`.
    pub hidden_act: GeluApprox,
    /// See [`crate::models::bert::BertConfig::num_labels`].
    pub num_labels: Option<i64>,
    /// See [`crate::models::bert::BertConfig::id2label`].
    pub id2label: Option<Vec<String>>,
    /// See [`crate::models::bert::BertConfig::architectures`].
    pub architectures: Option<Vec<String>>,
}

impl XlmRobertaConfig {
    /// Preset matching `xlm-roberta-base` on the HuggingFace Hub.
    ///
    /// Differs from [`RobertaConfig::roberta_base`](crate::models::roberta::RobertaConfig::roberta_base)
    /// only in `vocab_size` (`250_002` vs `50_265`) — the multilingual
    /// SentencePiece vocabulary is the reason XLM-RoBERTa exists.
    pub fn xlm_roberta_base() -> Self {
        XlmRobertaConfig {
            vocab_size: 250_002,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
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

    /// Parse a HuggingFace-style `config.json` string into an
    /// [`XlmRobertaConfig`].
    ///
    /// Reads the shape-affecting fields plus task-head metadata
    /// (`num_labels`, `id2label`). Defaults mirror HF's
    /// `XLMRobertaConfig`: `layer_norm_eps = 1e-5`,
    /// `type_vocab_size = 1`, `pad_token_id = 1`, dropout probabilities
    /// `0.1`.
    ///
    /// Does not validate `model_type` — direct callers opt in to the
    /// XLM-R parse explicitly. Use
    /// [`AutoConfig::from_json_str`](crate::models::auto::AutoConfig::from_json_str)
    /// when the family needs to be chosen at runtime from the file's
    /// contents.
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
        Ok(XlmRobertaConfig {
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
    /// Inverse of [`Self::from_json_str`]. Emits
    /// `model_type: "xlm-roberta"` + `architectures: ["XLMRobertaModel"]`
    /// so HF `AutoConfig` routes to the right class.
    pub fn to_json_str(&self) -> String {
        use crate::config_json::{emit_architectures, emit_hidden_act, emit_id2label};
        let mut m = serde_json::Map::new();
        m.insert("model_type".into(), "xlm-roberta".into());
        m.insert(
            "architectures".into(),
            emit_architectures(self.architectures.as_deref(), "XLMRobertaModel"),
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

impl From<&XlmRobertaConfig> for RobertaConfig {
    /// Shape-preserving projection to [`RobertaConfig`], used to reuse
    /// the RoBERTa backbone builder and task-head graph constructors.
    /// Every field has a 1:1 counterpart.
    fn from(c: &XlmRobertaConfig) -> Self {
        RobertaConfig {
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            intermediate_size: c.intermediate_size,
            max_position_embeddings: c.max_position_embeddings,
            type_vocab_size: c.type_vocab_size,
            pad_token_id: c.pad_token_id,
            layer_norm_eps: c.layer_norm_eps,
            hidden_dropout_prob: c.hidden_dropout_prob,
            attention_probs_dropout_prob: c.attention_probs_dropout_prob,
            hidden_act: c.hidden_act,
            num_labels: c.num_labels,
            id2label: c.id2label.clone(),
            architectures: c.architectures.clone(),
        }
    }
}

// ── XlmRobertaModel ──────────────────────────────────────────────────────

/// Assembled XLM-RoBERTa graph — zero-sized marker type mirroring
/// [`crate::models::roberta::RobertaModel`].
///
/// The returned [`Graph`] has identical input signature and shape to
/// its RoBERTa equivalent; see that type's documentation for the full
/// contract. The default Hub loader ([`XlmRobertaModel::from_pretrained`],
/// defined under the `hub` feature) builds the backbone *without* a
/// pooler since XLM-R checkpoints drop the NSP objective just like
/// RoBERTa.
pub struct XlmRobertaModel;

impl XlmRobertaModel {
    /// Build an XLM-RoBERTa graph on CPU with a pooler node.
    pub fn build(config: &XlmRobertaConfig) -> Result<Graph> {
        Self::on_device(config, Device::CPU)
    }

    /// Build an XLM-RoBERTa graph on `device` with a pooler node. Emits
    /// `pooler_output` (`[batch, hidden]`).
    pub fn on_device(config: &XlmRobertaConfig, device: Device) -> Result<Graph> {
        let rc: RobertaConfig = config.into();
        RobertaModel::on_device(&rc, device)
    }

    /// Build an XLM-RoBERTa graph on `device` *without* the pooler.
    /// Emits `last_hidden_state` (`[batch, seq_len, hidden]`) — the
    /// shape every task head in this module consumes, and the shape
    /// [`from_pretrained`](Self::from_pretrained) loads into.
    ///
    /// [`from_pretrained`](Self::from_pretrained) lives under the
    /// `hub` feature in [`crate::hub`].
    pub fn on_device_without_pooler(config: &XlmRobertaConfig, device: Device) -> Result<Graph> {
        let rc: RobertaConfig = config.into();
        RobertaModel::on_device_without_pooler(&rc, device)
    }
}

// ── Task heads ───────────────────────────────────────────────────────────

pub use crate::task_heads::{Answer, TokenPrediction};
use crate::task_heads::{
    ClassificationHead, EncoderInputs, MaskedLmHead, QaHead, TaggingHead, check_num_labels,
};

/// XLM-RoBERTa graphs take three `forward_multi` inputs — `input_ids`,
/// `token_type_ids`, and an extended attention mask — in that order.
/// Identical to RoBERTa: position ids are computed inside the embedding
/// layer, `<mask>` is the masking token, and `token_type_ids` are
/// typically all-zero (`type_vocab_size = 1`).
#[cfg(feature = "tokenizer")]
impl EncoderInputs for XlmRobertaConfig {
    const FAMILY_NAME: &'static str = "XlmRoberta";
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

/// XLM-RoBERTa with a sequence-classification head. Graph layout is
/// identical to [`RobertaForSequenceClassification`](crate::models::roberta::RobertaForSequenceClassification):
/// two-layer classifier on the `<s>` hidden state, keys
/// `classifier.dense.*` + `classifier.out_proj.*`.
///
/// Popular checkpoints: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
/// (3-label multilingual sentiment),
/// `joeddav/xlm-roberta-large-xnli` (zero-shot NLI),
/// `papluca/xlm-roberta-base-language-detection` (20-label language id).
pub type XlmRobertaForSequenceClassification = ClassificationHead<XlmRobertaConfig>;

impl ClassificationHead<XlmRobertaConfig> {
    /// Build the full graph (backbone without pooler + two-layer
    /// classification head) on `device` without loading any weights.
    pub fn on_device(config: &XlmRobertaConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let rc: RobertaConfig = config.into();
        let graph = roberta_backbone_flow(&rc, device, /*with_pooler=*/ false)?
            .through(RobertaClassificationHead::on_device(
                &rc, num_labels, device,
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

    pub(crate) fn num_labels_from_config(config: &XlmRobertaConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "XlmRobertaForSequenceClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// XLM-RoBERTa with a per-token classification head:
/// `last_hidden_state → Dropout → Linear(hidden, num_labels)`. Matches
/// HF Python's `XLMRobertaForTokenClassification`.
///
/// Popular checkpoints:
/// `Davlan/xlm-roberta-base-ner-hrl` (multilingual NER),
/// `Davlan/xlm-roberta-base-finetuned-conll03-english`.
pub type XlmRobertaForTokenClassification = TaggingHead<XlmRobertaConfig>;

impl TaggingHead<XlmRobertaConfig> {
    pub fn on_device(config: &XlmRobertaConfig, num_labels: i64, device: Device) -> Result<Self> {
        let num_labels = check_num_labels(num_labels)?;
        let rc: RobertaConfig = config.into();
        let graph = roberta_backbone_flow(&rc, device, /*with_pooler=*/ false)?
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

    pub(crate) fn num_labels_from_config(config: &XlmRobertaConfig) -> Result<i64> {
        config.num_labels.ok_or_else(|| {
            TensorError::new(
                "XlmRobertaForTokenClassification: config.json has no `num_labels` \
                 (nor `id2label`); cannot infer head size",
            )
        })
    }
}

/// XLM-RoBERTa with an extractive question-answering head:
/// `last_hidden_state → Linear(hidden, 2)` splitting into
/// `start_logits` and `end_logits`. Matches HF Python's
/// `XLMRobertaForQuestionAnswering`.
///
/// Popular checkpoints: `deepset/xlm-roberta-base-squad2`,
/// `deepset/xlm-roberta-large-squad2`.
pub type XlmRobertaForQuestionAnswering = QaHead<XlmRobertaConfig>;

impl QaHead<XlmRobertaConfig> {
    pub fn on_device(config: &XlmRobertaConfig, device: Device) -> Result<Self> {
        let rc: RobertaConfig = config.into();
        let graph = roberta_backbone_flow(&rc, device, /*with_pooler=*/ false)?
            .through(Linear::on_device(config.hidden_size, 2, device)?)
            .tag("qa_outputs")
            .build()?;
        Ok(Self::from_graph(graph, config))
    }
}

/// XLM-RoBERTa with a masked-language-modelling head. Graph layout is
/// identical to [`RobertaForMaskedLM`](crate::models::roberta::RobertaForMaskedLM)
/// — `lm_head.dense` → GELU → `lm_head.layer_norm` → tied-decoder
/// `Linear` whose weight is shared with
/// `roberta.embeddings.word_embeddings.weight` via
/// [`Linear::from_shared_weight`](flodl::nn::Linear::from_shared_weight).
///
/// Primary use case: continued multilingual pretraining / domain
/// adaptation. Canonical checkpoints: `xlm-roberta-base`,
/// `xlm-roberta-large`.
pub type XlmRobertaForMaskedLM = MaskedLmHead<XlmRobertaConfig>;

impl MaskedLmHead<XlmRobertaConfig> {
    /// Build the full graph: backbone (no pooler) + LM-head transform +
    /// tied decoder. Delegates to RoBERTa's crate-internal MLM graph
    /// builder (`roberta_masked_lm_graph`) since the graph shape and
    /// state_dict keys are identical to RoBERTa MLM.
    pub fn on_device(config: &XlmRobertaConfig, device: Device) -> Result<Self> {
        let rc: RobertaConfig = config.into();
        let graph = roberta_masked_lm_graph(&rc, device)?;
        Ok(Self::from_graph(graph, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_io::expected_from_graph;

    /// Round-trip: preset -> to_json_str -> from_json_str recovers the
    /// same config, and the emitted JSON carries the HF dispatch keys
    /// (`model_type: "xlm-roberta"`).
    #[test]
    fn xlm_roberta_config_to_json_str_round_trip() {
        let preset = XlmRobertaConfig::xlm_roberta_base();
        let s = preset.to_json_str();
        let recovered = XlmRobertaConfig::from_json_str(&s).unwrap();
        assert_eq!(preset.to_json_str(), recovered.to_json_str());
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(
            v.get("model_type").and_then(|x| x.as_str()),
            Some("xlm-roberta"),
        );
    }

    /// Field-by-field round trip through the `RobertaConfig` projection.
    /// Catches future drift if either struct grows a new field without
    /// updating the `From` impl.
    #[test]
    fn xlm_roberta_config_conversion_preserves_shape_fields() {
        let c = XlmRobertaConfig::xlm_roberta_base();
        let rc: RobertaConfig = (&c).into();
        assert_eq!(rc.vocab_size, c.vocab_size);
        assert_eq!(rc.hidden_size, c.hidden_size);
        assert_eq!(rc.num_hidden_layers, c.num_hidden_layers);
        assert_eq!(rc.num_attention_heads, c.num_attention_heads);
        assert_eq!(rc.intermediate_size, c.intermediate_size);
        assert_eq!(rc.max_position_embeddings, c.max_position_embeddings);
        assert_eq!(rc.type_vocab_size, c.type_vocab_size);
        assert_eq!(rc.pad_token_id, c.pad_token_id);
        assert!((rc.layer_norm_eps - c.layer_norm_eps).abs() < 1e-12);
        assert!((rc.hidden_dropout_prob - c.hidden_dropout_prob).abs() < 1e-12);
        assert!((rc.attention_probs_dropout_prob - c.attention_probs_dropout_prob).abs() < 1e-12,);
    }

    /// `xlm-roberta-base` defaults: 250k vocab (the reason XLM-R
    /// exists), otherwise identical to `roberta-base`.
    #[test]
    fn xlm_roberta_base_preset_matches_hf_defaults() {
        let c = XlmRobertaConfig::xlm_roberta_base();
        assert_eq!(c.vocab_size, 250_002);
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.num_hidden_layers, 12);
        assert_eq!(c.max_position_embeddings, 514);
        assert_eq!(c.pad_token_id, 1);
    }

    /// Smoke-parse the minimal set of fields that show up in every
    /// public XLM-R config.json.
    #[test]
    fn xlm_roberta_config_from_json_parses_base() {
        let json = r#"{
            "model_type": "xlm-roberta",
            "vocab_size": 250002,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "type_vocab_size": 1,
            "pad_token_id": 1
        }"#;
        let c = XlmRobertaConfig::from_json_str(json).unwrap();
        assert_eq!(c.vocab_size, 250_002);
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.pad_token_id, 1);
    }

    /// Graph tags must stay `roberta.*` for HF checkpoint compatibility —
    /// XLM-R's state_dict uses the RoBERTa prefix verbatim.
    #[test]
    fn xlm_roberta_backbone_emits_roberta_prefix() {
        let config = XlmRobertaConfig::xlm_roberta_base();
        let graph = XlmRobertaModel::on_device_without_pooler(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(&graph);
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert!(
            keys.contains(&"roberta.embeddings.word_embeddings.weight"),
            "expected roberta.embeddings.word_embeddings.weight, got {keys:?}",
        );
        assert!(
            keys.iter()
                .any(|k| k.starts_with("roberta.encoder.layer.0.attention.self.query.")),
            "expected roberta.encoder.* layer keys, got {keys:?}",
        );
        assert!(
            !keys.iter().any(|k| k.starts_with("xlm_roberta.")),
            "no keys should use an xlm_roberta.* prefix (HF uses roberta.*): {keys:?}",
        );
    }

    /// MLM head keeps the tied-decoder dedup that RoBERTa established:
    /// one `[V, H]` weight surfaces under the embedding tag, no
    /// `lm_head.decoder.weight` key.
    #[test]
    fn xlm_roberta_masked_lm_keeps_tied_weight_dedup() {
        let config = XlmRobertaConfig::xlm_roberta_base();
        let head = XlmRobertaForMaskedLM::on_device(&config, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<&str> = expected.iter().map(|p| p.key.as_str()).collect();

        assert!(
            keys.contains(&"roberta.embeddings.word_embeddings.weight"),
            "tied weight must surface under roberta.embeddings tag: {keys:?}",
        );
        assert!(
            !keys.contains(&"lm_head.decoder.weight"),
            "lm_head.decoder.weight must be absent (tied, dedup kept embeddings entry)",
        );

        let named = head.graph().named_parameters();
        let vocab_shaped = named
            .iter()
            .filter(|(_, p)| p.variable.shape() == vec![config.vocab_size, config.hidden_size])
            .count();
        assert_eq!(
            vocab_shaped, 1,
            "exactly one [V, H]-shaped Parameter expected under tying",
        );
    }

    /// SeqCls on XLM-R must use the two-layer `RobertaClassificationHead`,
    /// not the BERT-style single `classifier.{weight,bias}` linear. The
    /// HF state_dict format is fixed — this guards against accidental
    /// regressions to BERT semantics during future refactors.
    #[test]
    fn xlm_roberta_seqcls_head_has_two_layer_keys() {
        let config = XlmRobertaConfig::xlm_roberta_base();
        let head = XlmRobertaForSequenceClassification::on_device(&config, 3, Device::CPU).unwrap();
        let expected = expected_from_graph(head.graph());
        let keys: Vec<String> = expected.iter().map(|p| p.key.clone()).collect();
        assert!(keys.contains(&"classifier.dense.weight".to_string()));
        assert!(keys.contains(&"classifier.dense.bias".to_string()));
        assert!(keys.contains(&"classifier.out_proj.weight".to_string()));
        assert!(keys.contains(&"classifier.out_proj.bias".to_string()));
        assert!(!keys.iter().any(|k| k == "classifier.weight"));
    }
}
