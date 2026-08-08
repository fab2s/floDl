//! DeBERTa-v2/v3-family `from_pretrained` impls (backbone + 4 task heads).
//!
//! DeBERTa-v3 checkpoints ship under the `deberta-v2` architecture name
//! in HuggingFace transformers (the v3 distinction is a config knob, not
//! a separate model class). HF base checkpoints save as
//! `DebertaV2ForMaskedLM`, so the `lm_predictions.*` keys a bare
//! `DebertaV2Model` has no slot for are tolerated by
//! `load_weights_with_logging` and named on stderr.

use flodl::{Device, Graph, Result};

use crate::models::deberta_v2::{
    DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification, DebertaV2Model,
};

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, load_weights_with_logging};

impl DebertaV2Model {
    /// Download a pretrained DeBERTa-v2/v3 checkpoint from the
    /// HuggingFace Hub and return a fully-initialised [`Graph`] on
    /// CPU.
    ///
    /// `repo_id` examples: `"microsoft/deberta-v3-base"`,
    /// `"microsoft/deberta-v3-large"`, `"microsoft/deberta-v3-small"`.
    /// v1 DeBERTa checkpoints are rejected at config-parse time (see
    /// [`DebertaV2Config::from_json_str`](crate::models::deberta_v2::DebertaV2Config::from_json_str)).
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_config_and_weights(repo_id, DebertaV2Config::from_json_str)?;
        let graph = DebertaV2Model::on_device(&config, device)?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("DebertaV2Model").to_json_str());
        Ok(graph)
    }
}

impl DebertaV2ForSequenceClassification {
    /// Download a fine-tuned `DebertaV2ForSequenceClassification`
    /// checkpoint from the Hub. Popular checkpoints:
    /// `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI),
    /// `cross-encoder/nli-deberta-v3-base`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DebertaV2Config::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DebertaV2ForSequenceClassification")
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

impl DebertaV2ForTokenClassification {
    /// Download a fine-tuned `DebertaV2ForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DebertaV2Config::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DebertaV2ForTokenClassification")
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

impl DebertaV2ForQuestionAnswering {
    /// Download a fine-tuned `DebertaV2ForQuestionAnswering` checkpoint
    /// (SQuAD, …) from the Hub.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DebertaV2Config::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DebertaV2ForQuestionAnswering")
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

impl DebertaV2ForMaskedLM {
    /// Download a DeBERTa-v2/v3 MLM checkpoint from the Hub. Base
    /// checkpoints like `microsoft/deberta-v3-base` ship as the MLM
    /// variant, so this is the natural starting point for domain
    /// adaptation and continued pretraining.
    ///
    /// The decoder weight is tied to
    /// `deberta.embeddings.word_embeddings.weight`; the separate
    /// `lm_predictions.lm_head.bias` is a fresh `[vocab_size]`
    /// parameter.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, DebertaV2Config::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("DebertaV2ForMaskedLM")
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
