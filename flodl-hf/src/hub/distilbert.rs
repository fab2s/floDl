//! DistilBERT-family `from_pretrained` impls (backbone + 4 task heads).
//!
//! Each DistilBERT head mirrors its BERT / RoBERTa counterparts. The
//! family-specific bits are the config type (`DistilBertConfig`) and the
//! fact that the backbone graph never ships a pooler — `DistilBertModel`
//! is pooler-free by construction, matching every public DistilBERT
//! checkpoint.

use flodl::{Device, Graph, Result};

use crate::models::distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertModel,
};

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, load_weights_with_logging};

impl DistilBertModel {
    /// Download a pretrained DistilBERT checkpoint from the HuggingFace
    /// Hub and return a fully-initialised pooler-free [`Graph`] on CPU.
    ///
    /// `repo_id` examples: `"distilbert/distilbert-base-uncased"`,
    /// `"distilbert/distilbert-base-cased"`. HF base checkpoints ship
    /// as `DistilBertForMaskedLM` with an extra MLM head
    /// (`vocab_transform.*`, `vocab_layer_norm.*`, `vocab_projector.*`)
    /// that a bare `DistilBertModel` has no slot for;
    /// `load_weights_with_logging` tolerates those and names them on
    /// stderr.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_config_and_weights(repo_id, DistilBertConfig::from_json_str)?;
        let graph = DistilBertModel::on_device(&config, device)?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("DistilBertModel").to_json_str());
        Ok(graph)
    }
}

impl DistilBertForSequenceClassification {
    /// Download a fine-tuned `DistilBertForSequenceClassification`
    /// checkpoint from the Hub and return a ready-to-use predictor on
    /// CPU.
    ///
    /// Popular checkpoints:
    /// `lxyuan/distilbert-base-multilingual-cased-sentiments-student`
    /// (3-label multilingual sentiment),
    /// `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (2-label
    /// sentiment — older, lacks `tokenizer.json`).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DistilBertConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DistilBertForSequenceClassification")
                .to_json_str(),
        );
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DistilBertForTokenClassification {
    /// Download a fine-tuned `DistilBertForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub. Popular
    /// checkpoint: `dslim/distilbert-NER` (PER/ORG/LOC/MISC, 9 labels
    /// including `O`).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DistilBertConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DistilBertForTokenClassification")
                .to_json_str(),
        );
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DistilBertForQuestionAnswering {
    /// Download a fine-tuned `DistilBertForQuestionAnswering`
    /// checkpoint (SQuAD, etc.) from the Hub. Canonical:
    /// `distilbert/distilbert-base-cased-distilled-squad`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DistilBertConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DistilBertForQuestionAnswering")
                .to_json_str(),
        );
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl DistilBertForMaskedLM {
    /// Download a DistilBERT MLM checkpoint (`distilbert/distilbert-base-uncased`,
    /// `distilbert-base-cased`, any `*-mlm` domain-adaptation fine-tune)
    /// from the Hub.
    ///
    /// The `vocab_projector` weight is tied to
    /// `distilbert.embeddings.word_embeddings.weight`; checkpoints that
    /// redundantly save `vocab_projector.weight` load cleanly — the
    /// loader silently ignores keys absent from the graph's deduplicated
    /// `named_parameters()`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DistilBertConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DistilBertForMaskedLM")
                .to_json_str(),
        );
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}
