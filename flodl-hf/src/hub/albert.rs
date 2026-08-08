//! ALBERT-family `from_pretrained` impls (backbone + 4 task heads).
//!
//! Each head mirrors the BERT counterpart on the load side — the
//! family-specific bits (factorised embeddings, cross-layer sharing)
//! are all absorbed by the backbone builder. HF ALBERT base checkpoints
//! ship as `AlbertForMaskedLM` with `predictions.*` keys that a bare
//! `AlbertModel` has no slot for; `load_weights_with_logging`
//! tolerates and names those on stderr.

use flodl::{Device, Graph, Result};

use crate::models::albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
    AlbertForTokenClassification, AlbertModel,
};

use crate::safetensors_io::weights_have_pooler;

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, load_weights_with_logging};

impl AlbertModel {
    /// Download a pretrained ALBERT checkpoint from the HuggingFace
    /// Hub and return a fully-initialised [`Graph`] on CPU.
    ///
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't.
    /// `albert/albert-base-v2` ships its pooler as a flat
    /// `albert.pooler.{weight,bias}` (HF's `AlbertModel.pooler` is a
    /// bare `nn.Linear`, not a `BertPooler`-style `.dense` wrapper),
    /// and the dynamic detection picks both shapes.
    ///
    /// `repo_id` examples: `"albert/albert-base-v2"`, `"albert-large-v2"`,
    /// `"albert/albert-base-v2"`.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_config_and_weights(repo_id, AlbertConfig::from_json_str)?;
        let graph = if weights_have_pooler(&weights)? {
            AlbertModel::on_device(&config, device)?
        } else {
            AlbertModel::on_device_without_pooler(&config, device)?
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("AlbertModel").to_json_str());
        Ok(graph)
    }
}

impl AlbertForSequenceClassification {
    /// Download a fine-tuned `AlbertForSequenceClassification`
    /// checkpoint from the Hub. Popular checkpoints:
    /// `textattack/albert-base-v2-SST-2` (binary sentiment),
    /// `textattack/albert-base-v2-MRPC`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, AlbertConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("AlbertForSequenceClassification")
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

impl AlbertForTokenClassification {
    /// Download a fine-tuned `AlbertForTokenClassification` checkpoint
    /// (NER, POS tagging, …) from the Hub.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, AlbertConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("AlbertForTokenClassification")
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

impl AlbertForQuestionAnswering {
    /// Download a fine-tuned `AlbertForQuestionAnswering` checkpoint
    /// (SQuAD, …) from the Hub. Popular checkpoint:
    /// `twmkn9/albert-base-v2-squad2`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, AlbertConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("AlbertForQuestionAnswering")
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

impl AlbertForMaskedLM {
    /// Download an ALBERT MLM checkpoint (`albert/albert-base-v2`,
    /// `albert-large-v2`, any `*-mlm` domain-adaptation fine-tune)
    /// from the Hub.
    ///
    /// The decoder weight is tied to
    /// `albert.embeddings.word_embeddings.weight`; HF's historical
    /// save format also emits `predictions.bias` (tied to
    /// `decoder.bias`) as a top-level key. Both redundant keys are
    /// silently ignored by
    /// `load_safetensors_into_graph_with_rename_allow_unused`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, AlbertConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph()
            .set_source_config(config.with_architectures("AlbertForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}
