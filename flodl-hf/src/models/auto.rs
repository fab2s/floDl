//! Model-type dispatch over the supported BERT-family encoders.
//!
//! [`AutoConfig`] reads `config.json`'s `model_type` field and parses
//! the payload as the matching family-specific config. [`AutoModel`]
//! and the [`AutoModelForSequenceClassification`] /
//! [`AutoModelForTokenClassification`] /
//! [`AutoModelForQuestionAnswering`] /
//! [`AutoModelForMaskedLM`] enums provide one-liner Hub loading over
//! `bert` / `roberta` / `distilbert` without the caller knowing which
//! family the checkpoint belongs to.
//!
//! ## Output-shape convention
//!
//! [`AutoModel`] routes BERT through
//! [`BertModel::on_device_without_pooler`](crate::models::bert::BertModel::on_device_without_pooler)
//! so the returned [`Graph`](flodl::Graph) always emits
//! `last_hidden_state` of shape `[batch, seq_len, hidden]`, consistent
//! with RoBERTa and DistilBERT. This diverges from HF Python's
//! `AutoModel` (which exposes both `last_hidden_state` and
//! `pooler_output` on BERT), in favour of a uniform Rust API. If you
//! specifically need BERT's `pooler_output`, reach for
//! [`BertModel::from_pretrained`](crate::models::bert::BertModel::from_pretrained)
//! directly.
//!
//! ## Input-signature asymmetry
//!
//! The returned [`Graph`](flodl::Graph)'s `forward_multi` input count
//! differs by family:
//!
//! - BERT: `[input_ids, position_ids, token_type_ids, attention_mask]` (4)
//! - RoBERTa: `[input_ids, token_type_ids, attention_mask]` (3)
//! - DistilBERT: `[input_ids, attention_mask]` (2)
//!
//! Callers that run the graph directly need to know the family — use
//! [`AutoConfig::from_json_str`] or inspect
//! [`AutoModelForSequenceClassification`] and friends via their
//! enum variants. The task-head wrappers ([`AutoModelForSequenceClassification`]
//! etc.) encapsulate this asymmetry behind unified
//! `predict` / `tag` / `answer` methods.
//!
//! ## Supported model types
//!
//! | `model_type`    | Family                                                      |
//! |-----------------|-------------------------------------------------------------|
//! | `bert`          | [`crate::models::bert::BertConfig`]                         |
//! | `roberta`       | [`crate::models::roberta::RobertaConfig`]                   |
//! | `distilbert`    | [`crate::models::distilbert::DistilBertConfig`]             |
//! | `xlm-roberta`   | [`crate::models::xlm_roberta::XlmRobertaConfig`]            |
//! | `albert`        | [`crate::models::albert::AlbertConfig`]                     |
//! | `deberta-v2`    | [`crate::models::deberta_v2::DebertaV2Config`]              |
//!
//! Any other value (e.g. `modernbert`, `electra`) surfaces a loud
//! error listing the supported set.

use flodl::{Result, TensorError};

use crate::models::albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
    AlbertForTokenClassification,
};
use crate::models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification,
};
use crate::models::deberta_v2::{
    DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification,
};
use crate::models::distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification,
};
use crate::models::roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification,
};
use crate::models::xlm_roberta::{
    XlmRobertaConfig, XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering,
    XlmRobertaForSequenceClassification, XlmRobertaForTokenClassification,
};

/// Family-tagged parsed `config.json`.
///
/// Each variant wraps the fully-parsed family-specific config so callers
/// can match on the variant to know which architecture the checkpoint
/// belongs to. Parse via [`AutoConfig::from_json_str`].
///
/// Marked `#[non_exhaustive]` so future family additions do not break
/// caller `match` arms; write a `_ => …` arm to stay forward-compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AutoConfig {
    Bert(BertConfig),
    Roberta(RobertaConfig),
    DistilBert(DistilBertConfig),
    XlmRoberta(XlmRobertaConfig),
    Albert(AlbertConfig),
    DebertaV2(DebertaV2Config),
}

impl AutoConfig {
    /// Parse a `config.json` string, dispatching on its `model_type`
    /// field to the matching family config parser.
    ///
    /// Errors on:
    /// - invalid JSON,
    /// - missing / non-string `model_type`,
    /// - unsupported `model_type` (error names the supported set),
    /// - any downstream parse error from the family-specific
    ///   `from_json_str` (missing required field, malformed
    ///   `id2label`, …).
    pub fn from_json_str(s: &str) -> Result<Self> {
        use crate::config_json::required_string;
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| TensorError::new(&format!("config.json parse error: {e}")))?;
        let model_type = required_string(&v, "model_type")?;
        match model_type {
            "bert" => Ok(AutoConfig::Bert(BertConfig::from_json_str(s)?)),
            "roberta" => Ok(AutoConfig::Roberta(RobertaConfig::from_json_str(s)?)),
            "distilbert" => Ok(AutoConfig::DistilBert(DistilBertConfig::from_json_str(s)?)),
            "xlm-roberta" => Ok(AutoConfig::XlmRoberta(XlmRobertaConfig::from_json_str(s)?)),
            "albert" => Ok(AutoConfig::Albert(AlbertConfig::from_json_str(s)?)),
            "deberta-v2" => Ok(AutoConfig::DebertaV2(DebertaV2Config::from_json_str(s)?)),
            other => Err(TensorError::new(&format!(
                "AutoConfig: unsupported model_type {other:?}. \
                 Supported families: \"bert\", \"roberta\", \"distilbert\", \"xlm-roberta\", \
                 \"albert\", \"deberta-v2\". ModernBERT and other architectures are planned \
                 for a future release.",
            ))),
        }
    }

    /// The underlying `model_type` string as it appeared in
    /// `config.json` (e.g. `"bert"`, `"roberta"`, `"distilbert"`,
    /// `"xlm-roberta"`, `"albert"`, `"deberta-v2"`).
    pub fn model_type(&self) -> &'static str {
        match self {
            AutoConfig::Bert(_) => "bert",
            AutoConfig::Roberta(_) => "roberta",
            AutoConfig::DistilBert(_) => "distilbert",
            AutoConfig::XlmRoberta(_) => "xlm-roberta",
            AutoConfig::Albert(_) => "albert",
            AutoConfig::DebertaV2(_) => "deberta-v2",
        }
    }

    /// Dispatch to the inner family config's `to_json_str`. Produces
    /// an HF-compatible `config.json` string that round-trips through
    /// [`AutoConfig::from_json_str`]. Primary caller:
    /// [`crate::export::export_hf_dir`].
    pub fn to_json_str(&self) -> String {
        match self {
            AutoConfig::Bert(c) => c.to_json_str(),
            AutoConfig::Roberta(c) => c.to_json_str(),
            AutoConfig::DistilBert(c) => c.to_json_str(),
            AutoConfig::XlmRoberta(c) => c.to_json_str(),
            AutoConfig::Albert(c) => c.to_json_str(),
            AutoConfig::DebertaV2(c) => c.to_json_str(),
        }
    }

    /// HF Python class names from the underlying `config.json`'s
    /// `architectures` field. `None` when the config omitted the field
    /// or when the array was empty / non-string. Used by
    /// [`crate::export::build_for_export`] to dispatch a checkpoint to
    /// the matching task-head builder when re-exporting via
    /// `fdl flodl-hf export --checkpoint`.
    pub fn architectures(&self) -> Option<&[String]> {
        match self {
            AutoConfig::Bert(c) => c.architectures.as_deref(),
            AutoConfig::Roberta(c) => c.architectures.as_deref(),
            AutoConfig::DistilBert(c) => c.architectures.as_deref(),
            AutoConfig::XlmRoberta(c) => c.architectures.as_deref(),
            AutoConfig::Albert(c) => c.architectures.as_deref(),
            AutoConfig::DebertaV2(c) => c.architectures.as_deref(),
        }
    }

    /// HF Python class name for the bare backbone of this family
    /// (e.g. `"BertModel"`, `"XLMRobertaModel"`). Used to normalise the
    /// `architectures` field on the `source_config` stamped by base-only
    /// loaders like [`AutoModel::from_pretrained`](crate::hub) — the
    /// upstream Hub config typically advertises a head class
    /// (`BertForPreTraining`, `RobertaForMaskedLM`), but this loader
    /// builds the bare backbone and silently drops the head keys, so
    /// the sidecar must reflect what was actually built.
    pub fn base_class_name(&self) -> &'static str {
        match self {
            AutoConfig::Bert(_) => "BertModel",
            AutoConfig::Roberta(_) => "RobertaModel",
            AutoConfig::DistilBert(_) => "DistilBertModel",
            AutoConfig::XlmRoberta(_) => "XLMRobertaModel",
            AutoConfig::Albert(_) => "AlbertModel",
            AutoConfig::DebertaV2(_) => "DebertaV2Model",
        }
    }

    /// Replace the inner family config's `architectures` field with
    /// `[arch_class]` and return the serialised `config.json` string.
    /// Consuming variant of [`Self::to_json_str`] paired with
    /// [`Self::base_class_name`] for base-only loaders that need to
    /// stamp the source_config sidecar with the class actually built.
    pub fn into_normalized_config_json(self, arch_class: &str) -> String {
        match self {
            AutoConfig::Bert(c) => c.with_architectures(arch_class).to_json_str(),
            AutoConfig::Roberta(c) => c.with_architectures(arch_class).to_json_str(),
            AutoConfig::DistilBert(c) => c.with_architectures(arch_class).to_json_str(),
            AutoConfig::XlmRoberta(c) => c.with_architectures(arch_class).to_json_str(),
            AutoConfig::Albert(c) => c.with_architectures(arch_class).to_json_str(),
            AutoConfig::DebertaV2(c) => c.with_architectures(arch_class).to_json_str(),
        }
    }
}

/// One-liner Hub loader that dispatches to the matching family
/// backbone. Implementation lives in [`crate::hub`] behind the
/// `hub` feature; see
/// [`AutoModel::from_pretrained`](crate::hub) for the actual entry
/// point.
///
/// Zero-sized marker type — mirrors the pattern used by
/// [`BertModel`](crate::models::bert::BertModel),
/// [`RobertaModel`](crate::models::roberta::RobertaModel), and
/// [`DistilBertModel`](crate::models::distilbert::DistilBertModel).
pub struct AutoModel;

/// Family-dispatched sequence-classification head loaded from the
/// HuggingFace Hub. Call
/// [`AutoModelForSequenceClassification::from_pretrained`](crate::hub)
/// (behind the `hub` feature) to build one, then
/// [`predict`](Self::predict) to classify raw text.
///
/// Marked `#[non_exhaustive]` so future family additions do not break
/// caller `match` arms; write a `_ => …` arm to stay forward-compatible.
#[non_exhaustive]
pub enum AutoModelForSequenceClassification {
    Bert(BertForSequenceClassification),
    Roberta(RobertaForSequenceClassification),
    DistilBert(DistilBertForSequenceClassification),
    XlmRoberta(XlmRobertaForSequenceClassification),
    Albert(AlbertForSequenceClassification),
    DebertaV2(DebertaV2ForSequenceClassification),
}

/// Family-dispatched token-classification head. Canonical use is NER.
/// Build via
/// [`AutoModelForTokenClassification::from_pretrained`](crate::hub)
/// and call [`predict`](Self::predict) to tag raw text.
///
/// Marked `#[non_exhaustive]` so future family additions do not break
/// caller `match` arms; write a `_ => …` arm to stay forward-compatible.
#[non_exhaustive]
pub enum AutoModelForTokenClassification {
    Bert(BertForTokenClassification),
    Roberta(RobertaForTokenClassification),
    DistilBert(DistilBertForTokenClassification),
    XlmRoberta(XlmRobertaForTokenClassification),
    Albert(AlbertForTokenClassification),
    DebertaV2(DebertaV2ForTokenClassification),
}

/// Family-dispatched extractive question-answering head. Build via
/// [`AutoModelForQuestionAnswering::from_pretrained`](crate::hub)
/// and call [`answer`](Self::answer) for one `(question, context)`
/// pair.
///
/// Marked `#[non_exhaustive]` so future family additions do not break
/// caller `match` arms; write a `_ => …` arm to stay forward-compatible.
#[non_exhaustive]
pub enum AutoModelForQuestionAnswering {
    Bert(BertForQuestionAnswering),
    Roberta(RobertaForQuestionAnswering),
    DistilBert(DistilBertForQuestionAnswering),
    XlmRoberta(XlmRobertaForQuestionAnswering),
    Albert(AlbertForQuestionAnswering),
    DebertaV2(DebertaV2ForQuestionAnswering),
}

impl AutoModelForSequenceClassification {
    /// Borrow the underlying [`Graph`](flodl::Graph) of the inner
    /// concrete head. Useful for inspection, custom inference, or
    /// parameter export.
    pub fn graph(&self) -> &flodl::Graph {
        match self {
            Self::Bert(h) => h.graph(),
            Self::Roberta(h) => h.graph(),
            Self::DistilBert(h) => h.graph(),
            Self::XlmRoberta(h) => h.graph(),
            Self::Albert(h) => h.graph(),
            Self::DebertaV2(h) => h.graph(),
        }
    }

    /// Consume `self` and return the underlying [`Graph`](flodl::Graph)
    /// of the inner concrete head. Used by `fdl flodl-hf export --hub`
    /// after auto-dispatching on the upstream `architectures[0]`.
    pub fn into_graph(self) -> flodl::Graph {
        match self {
            Self::Bert(h) => h.into_graph(),
            Self::Roberta(h) => h.into_graph(),
            Self::DistilBert(h) => h.into_graph(),
            Self::XlmRoberta(h) => h.into_graph(),
            Self::Albert(h) => h.into_graph(),
            Self::DebertaV2(h) => h.into_graph(),
        }
    }

    /// Label names indexed by class id.
    pub fn labels(&self) -> &[String] {
        match self {
            Self::Bert(h) => h.labels(),
            Self::Roberta(h) => h.labels(),
            Self::DistilBert(h) => h.labels(),
            Self::XlmRoberta(h) => h.labels(),
            Self::Albert(h) => h.labels(),
            Self::DebertaV2(h) => h.labels(),
        }
    }

    /// Attach a tokenizer, replacing any previously attached one.
    /// Delegates to the inner concrete head.
    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(self, tok: crate::tokenizer::HfTokenizer) -> Self {
        match self {
            Self::Bert(h) => Self::Bert(h.with_tokenizer(tok)),
            Self::Roberta(h) => Self::Roberta(h.with_tokenizer(tok)),
            Self::DistilBert(h) => Self::DistilBert(h.with_tokenizer(tok)),
            Self::XlmRoberta(h) => Self::XlmRoberta(h.with_tokenizer(tok)),
            Self::Albert(h) => Self::Albert(h.with_tokenizer(tok)),
            Self::DebertaV2(h) => Self::DebertaV2(h.with_tokenizer(tok)),
        }
    }

    /// One-shot text → label distribution. Encodes with the attached
    /// tokenizer, runs the graph, and returns per-input label
    /// distributions sorted by descending probability.
    #[cfg(feature = "tokenizer")]
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<(String, f32)>>> {
        match self {
            Self::Bert(h) => h.predict(texts),
            Self::Roberta(h) => h.predict(texts),
            Self::DistilBert(h) => h.predict(texts),
            Self::XlmRoberta(h) => h.predict(texts),
            Self::Albert(h) => h.predict(texts),
            Self::DebertaV2(h) => h.predict(texts),
        }
    }
}

impl AutoModelForTokenClassification {
    pub fn graph(&self) -> &flodl::Graph {
        match self {
            Self::Bert(h) => h.graph(),
            Self::Roberta(h) => h.graph(),
            Self::DistilBert(h) => h.graph(),
            Self::XlmRoberta(h) => h.graph(),
            Self::Albert(h) => h.graph(),
            Self::DebertaV2(h) => h.graph(),
        }
    }

    /// Consume `self` and return the underlying [`Graph`](flodl::Graph)
    /// of the inner concrete head. Used by `fdl flodl-hf export --hub`
    /// after auto-dispatching on the upstream `architectures[0]`.
    pub fn into_graph(self) -> flodl::Graph {
        match self {
            Self::Bert(h) => h.into_graph(),
            Self::Roberta(h) => h.into_graph(),
            Self::DistilBert(h) => h.into_graph(),
            Self::XlmRoberta(h) => h.into_graph(),
            Self::Albert(h) => h.into_graph(),
            Self::DebertaV2(h) => h.into_graph(),
        }
    }

    pub fn labels(&self) -> &[String] {
        match self {
            Self::Bert(h) => h.labels(),
            Self::Roberta(h) => h.labels(),
            Self::DistilBert(h) => h.labels(),
            Self::XlmRoberta(h) => h.labels(),
            Self::Albert(h) => h.labels(),
            Self::DebertaV2(h) => h.labels(),
        }
    }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(self, tok: crate::tokenizer::HfTokenizer) -> Self {
        match self {
            Self::Bert(h) => Self::Bert(h.with_tokenizer(tok)),
            Self::Roberta(h) => Self::Roberta(h.with_tokenizer(tok)),
            Self::DistilBert(h) => Self::DistilBert(h.with_tokenizer(tok)),
            Self::XlmRoberta(h) => Self::XlmRoberta(h.with_tokenizer(tok)),
            Self::Albert(h) => Self::Albert(h.with_tokenizer(tok)),
            Self::DebertaV2(h) => Self::DebertaV2(h.with_tokenizer(tok)),
        }
    }

    /// One-shot text → per-token tags. Output shape mirrors the
    /// tokenizer's encoding: `result[b][s]` is the top-1 prediction
    /// for batch entry `b`, position `s`.
    #[cfg(feature = "tokenizer")]
    pub fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<crate::task_heads::TokenPrediction>>> {
        match self {
            Self::Bert(h) => h.predict(texts),
            Self::Roberta(h) => h.predict(texts),
            Self::DistilBert(h) => h.predict(texts),
            Self::XlmRoberta(h) => h.predict(texts),
            Self::Albert(h) => h.predict(texts),
            Self::DebertaV2(h) => h.predict(texts),
        }
    }
}

impl AutoModelForQuestionAnswering {
    pub fn graph(&self) -> &flodl::Graph {
        match self {
            Self::Bert(h) => h.graph(),
            Self::Roberta(h) => h.graph(),
            Self::DistilBert(h) => h.graph(),
            Self::XlmRoberta(h) => h.graph(),
            Self::Albert(h) => h.graph(),
            Self::DebertaV2(h) => h.graph(),
        }
    }

    /// Consume `self` and return the underlying [`Graph`](flodl::Graph)
    /// of the inner concrete head. Used by `fdl flodl-hf export --hub`
    /// after auto-dispatching on the upstream `architectures[0]`.
    pub fn into_graph(self) -> flodl::Graph {
        match self {
            Self::Bert(h) => h.into_graph(),
            Self::Roberta(h) => h.into_graph(),
            Self::DistilBert(h) => h.into_graph(),
            Self::XlmRoberta(h) => h.into_graph(),
            Self::Albert(h) => h.into_graph(),
            Self::DebertaV2(h) => h.into_graph(),
        }
    }

    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(self, tok: crate::tokenizer::HfTokenizer) -> Self {
        match self {
            Self::Bert(h) => Self::Bert(h.with_tokenizer(tok)),
            Self::Roberta(h) => Self::Roberta(h.with_tokenizer(tok)),
            Self::DistilBert(h) => Self::DistilBert(h.with_tokenizer(tok)),
            Self::XlmRoberta(h) => Self::XlmRoberta(h.with_tokenizer(tok)),
            Self::Albert(h) => Self::Albert(h.with_tokenizer(tok)),
            Self::DebertaV2(h) => Self::DebertaV2(h.with_tokenizer(tok)),
        }
    }

    /// Answer one `(question, context)` pair. Returns the
    /// highest-scoring span over the context tokens.
    #[cfg(feature = "tokenizer")]
    pub fn answer(&self, question: &str, context: &str) -> Result<crate::task_heads::Answer> {
        match self {
            Self::Bert(h) => h.answer(question, context),
            Self::Roberta(h) => h.answer(question, context),
            Self::DistilBert(h) => h.answer(question, context),
            Self::XlmRoberta(h) => h.answer(question, context),
            Self::Albert(h) => h.answer(question, context),
            Self::DebertaV2(h) => h.answer(question, context),
        }
    }

    /// Batched variant of [`answer`](Self::answer).
    #[cfg(feature = "tokenizer")]
    pub fn answer_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<crate::task_heads::Answer>> {
        match self {
            Self::Bert(h) => h.answer_batch(pairs),
            Self::Roberta(h) => h.answer_batch(pairs),
            Self::DistilBert(h) => h.answer_batch(pairs),
            Self::XlmRoberta(h) => h.answer_batch(pairs),
            Self::Albert(h) => h.answer_batch(pairs),
            Self::DebertaV2(h) => h.answer_batch(pairs),
        }
    }
}

/// Family-dispatched masked-language-modelling head. Build via
/// [`AutoModelForMaskedLM::from_pretrained`](crate::hub) and call
/// [`fill_mask`](Self::fill_mask) to predict the top-`k` replacements
/// at every mask-token position in the input.
///
/// The mask-token spelling (`[MASK]` for BERT/DistilBERT, `<mask>` for
/// RoBERTa) is handled per-variant via the inner head's
/// [`EncoderInputs::MASK_TOKEN`](crate::task_heads::EncoderInputs::MASK_TOKEN),
/// so callers write one `fill_mask` call regardless of the underlying
/// family.
///
/// Marked `#[non_exhaustive]` so future family additions do not break
/// caller `match` arms; write a `_ => …` arm to stay forward-compatible.
#[non_exhaustive]
pub enum AutoModelForMaskedLM {
    Bert(BertForMaskedLM),
    Roberta(RobertaForMaskedLM),
    DistilBert(DistilBertForMaskedLM),
    XlmRoberta(XlmRobertaForMaskedLM),
    Albert(AlbertForMaskedLM),
    DebertaV2(DebertaV2ForMaskedLM),
}

impl AutoModelForMaskedLM {
    /// Borrow the underlying [`Graph`](flodl::Graph) of the inner
    /// concrete head.
    pub fn graph(&self) -> &flodl::Graph {
        match self {
            Self::Bert(h) => h.graph(),
            Self::Roberta(h) => h.graph(),
            Self::DistilBert(h) => h.graph(),
            Self::XlmRoberta(h) => h.graph(),
            Self::Albert(h) => h.graph(),
            Self::DebertaV2(h) => h.graph(),
        }
    }

    /// Consume `self` and return the underlying [`Graph`](flodl::Graph)
    /// of the inner concrete head. Used by `fdl flodl-hf export --hub`
    /// after auto-dispatching on the upstream `architectures[0]`.
    pub fn into_graph(self) -> flodl::Graph {
        match self {
            Self::Bert(h) => h.into_graph(),
            Self::Roberta(h) => h.into_graph(),
            Self::DistilBert(h) => h.into_graph(),
            Self::XlmRoberta(h) => h.into_graph(),
            Self::Albert(h) => h.into_graph(),
            Self::DebertaV2(h) => h.into_graph(),
        }
    }

    /// Attach a tokenizer, replacing any previously attached one.
    #[cfg(feature = "tokenizer")]
    pub fn with_tokenizer(self, tok: crate::tokenizer::HfTokenizer) -> Self {
        match self {
            Self::Bert(h) => Self::Bert(h.with_tokenizer(tok)),
            Self::Roberta(h) => Self::Roberta(h.with_tokenizer(tok)),
            Self::DistilBert(h) => Self::DistilBert(h.with_tokenizer(tok)),
            Self::XlmRoberta(h) => Self::XlmRoberta(h.with_tokenizer(tok)),
            Self::Albert(h) => Self::Albert(h.with_tokenizer(tok)),
            Self::DebertaV2(h) => Self::DebertaV2(h.with_tokenizer(tok)),
        }
    }

    /// Fill every mask-token position in `text` with its top-`k`
    /// predicted replacements, sorted by descending softmax
    /// probability. Delegates to the inner concrete head's
    /// [`fill_mask`](crate::task_heads::MaskedLmHead::fill_mask) —
    /// the mask-token spelling comes from the head's config.
    #[cfg(feature = "tokenizer")]
    pub fn fill_mask(&self, text: &str, top_k: usize) -> Result<Vec<Vec<(String, f32)>>> {
        match self {
            Self::Bert(h) => h.fill_mask(text, top_k),
            Self::Roberta(h) => h.fill_mask(text, top_k),
            Self::DistilBert(h) => h.fill_mask(text, top_k),
            Self::XlmRoberta(h) => h.fill_mask(text, top_k),
            Self::Albert(h) => h.fill_mask(text, top_k),
            Self::DebertaV2(h) => h.fill_mask(text, top_k),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `bert-base-uncased`-style config.json dispatches to
    /// `AutoConfig::Bert` and round-trips through `BertConfig`.
    #[test]
    fn auto_config_dispatches_bert() {
        let json = r#"{
            "model_type": "bert",
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "pad_token_id": 0
        }"#;
        let c = AutoConfig::from_json_str(json).unwrap();
        assert_eq!(c.model_type(), "bert");
        match c {
            AutoConfig::Bert(b) => {
                assert_eq!(b.vocab_size, 30522);
                assert_eq!(b.hidden_size, 768);
            }
            other => panic!("expected Bert, got {:?}", other.model_type()),
        }
    }

    /// `roberta-base`-style config.json dispatches to
    /// `AutoConfig::Roberta`.
    #[test]
    fn auto_config_dispatches_roberta() {
        let json = r#"{
            "model_type": "roberta",
            "vocab_size": 50265,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 514,
            "type_vocab_size": 1,
            "pad_token_id": 1
        }"#;
        let c = AutoConfig::from_json_str(json).unwrap();
        assert_eq!(c.model_type(), "roberta");
        match c {
            AutoConfig::Roberta(r) => {
                assert_eq!(r.vocab_size, 50265);
                assert_eq!(r.pad_token_id, 1);
            }
            other => panic!("expected Roberta, got {:?}", other.model_type()),
        }
    }

    /// `distilbert-base-uncased`-style config.json dispatches to
    /// `AutoConfig::DistilBert` (note the DistilBERT-specific field
    /// names: `dim`, `n_layers`, etc.).
    #[test]
    fn auto_config_dispatches_distilbert() {
        let json = r#"{
            "model_type": "distilbert",
            "vocab_size": 30522,
            "dim": 768,
            "n_layers": 6,
            "n_heads": 12,
            "hidden_dim": 3072,
            "max_position_embeddings": 512,
            "pad_token_id": 0
        }"#;
        let c = AutoConfig::from_json_str(json).unwrap();
        assert_eq!(c.model_type(), "distilbert");
        match c {
            AutoConfig::DistilBert(d) => {
                assert_eq!(d.vocab_size, 30522);
                assert_eq!(d.n_layers, 6);
            }
            other => panic!("expected DistilBert, got {:?}", other.model_type()),
        }
    }

    /// `xlm-roberta-base`-style config.json dispatches to
    /// `AutoConfig::XlmRoberta`. Field layout is identical to RoBERTa;
    /// the multilingual vocabulary (250k) is the only distinguishing
    /// numeric field.
    #[test]
    fn auto_config_dispatches_xlm_roberta() {
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
        let c = AutoConfig::from_json_str(json).unwrap();
        assert_eq!(c.model_type(), "xlm-roberta");
        match c {
            AutoConfig::XlmRoberta(x) => {
                assert_eq!(x.vocab_size, 250_002);
                assert_eq!(x.pad_token_id, 1);
            }
            other => panic!("expected XlmRoberta, got {:?}", other.model_type()),
        }
    }

    /// Unsupported `model_type` must surface a loud error that names
    /// the supported set — silently misdispatching a ModernBERT or
    /// ELECTRA config as BERT would produce confusing shape errors
    /// deep inside the loader.
    #[test]
    fn auto_config_rejects_unknown_model_type() {
        let json = r#"{
            "model_type": "modernbert",
            "vocab_size": 50368,
            "hidden_size": 768
        }"#;
        let err = AutoConfig::from_json_str(json).unwrap_err().to_string();
        assert!(
            err.contains("modernbert"),
            "error names offending type: {err}"
        );
        assert!(err.contains("bert"), "error lists supported: {err}");
        assert!(err.contains("roberta"), "error lists supported: {err}");
        assert!(err.contains("distilbert"), "error lists supported: {err}");
        assert!(err.contains("xlm-roberta"), "error lists supported: {err}");
        assert!(err.contains("albert"), "error lists supported: {err}");
        assert!(err.contains("deberta-v2"), "error lists supported: {err}");
    }

    /// `microsoft/deberta-v3-base`-style config.json dispatches to
    /// `AutoConfig::DebertaV2`. Distinguishing fields:
    /// `relative_attention=true`, `pos_att_type="p2c|c2p"`,
    /// `share_att_key=true`, `type_vocab_size=0`.
    #[test]
    fn auto_config_dispatches_deberta_v2() {
        let json = r#"{
            "model_type": "deberta-v2",
            "vocab_size": 128100,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "relative_attention": true,
            "position_buckets": 256,
            "norm_rel_ebd": "layer_norm",
            "share_att_key": true,
            "pos_att_type": "p2c|c2p",
            "layer_norm_eps": 1e-7,
            "max_relative_positions": -1,
            "position_biased_input": false,
            "type_vocab_size": 0
        }"#;
        let c = AutoConfig::from_json_str(json).unwrap();
        assert_eq!(c.model_type(), "deberta-v2");
        match c {
            AutoConfig::DebertaV2(d) => {
                assert_eq!(d.vocab_size, 128_100);
                assert_eq!(d.hidden_size, 768);
                assert_eq!(d.position_buckets, 256);
                assert_eq!(d.max_relative_positions, 512);
            }
            other => panic!("expected DebertaV2, got {:?}", other.model_type()),
        }
    }

    /// `albert-base-v2`-style config.json dispatches to
    /// `AutoConfig::Albert`. Distinguishing fields: `embedding_size`
    /// (factorised embeddings) and the default
    /// `num_hidden_groups=1`, `inner_group_num=1` layout.
    #[test]
    fn auto_config_dispatches_albert() {
        let json = r#"{
            "model_type": "albert",
            "vocab_size": 30000,
            "embedding_size": 128,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512
        }"#;
        let c = AutoConfig::from_json_str(json).unwrap();
        assert_eq!(c.model_type(), "albert");
        match c {
            AutoConfig::Albert(a) => {
                assert_eq!(a.vocab_size, 30000);
                assert_eq!(a.embedding_size, 128);
                assert_eq!(a.hidden_size, 768);
            }
            other => panic!("expected Albert, got {:?}", other.model_type()),
        }
    }

    /// Missing `model_type` must error with a clear message, not
    /// silently default to BERT.
    #[test]
    fn auto_config_rejects_missing_model_type() {
        let json = r#"{
            "vocab_size": 30522,
            "hidden_size": 768
        }"#;
        let err = AutoConfig::from_json_str(json).unwrap_err().to_string();
        assert!(
            err.contains("model_type"),
            "error must name the missing field: {err}",
        );
    }

    /// Invalid JSON must produce a clear parse error rather than a
    /// confusing `model_type` missing error.
    #[test]
    fn auto_config_rejects_invalid_json() {
        let err = AutoConfig::from_json_str("not json")
            .unwrap_err()
            .to_string();
        assert!(err.contains("parse error"), "got: {err}");
    }
}
