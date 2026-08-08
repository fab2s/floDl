//! RoBERTa-family `from_pretrained` impls (backbone + 4 task heads).

use flodl::{Device, Graph, Result};

use crate::models::roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel,
};

use crate::safetensors_io::weights_have_pooler;

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, load_weights_with_logging};

impl RobertaModel {
    /// Download a pretrained RoBERTa checkpoint from the HuggingFace
    /// Hub and return a fully-initialised [`Graph`] on CPU.
    ///
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't, so the
    /// graph shape matches the Hub repo regardless of whether the
    /// checkpoint kept BERT's pooler. Most RoBERTa checkpoints
    /// (including `roberta-base`) drop the pooler with the NSP
    /// objective, but some downstream-tuned variants keep it. HF
    /// Python silently random-initialises a missing pooler on load,
    /// producing non-reproducible `pooler_output`; flodl-hf instead
    /// builds a pooler-free backbone in that case so the load stays
    /// strict and reproducible. Reach for [`RobertaModel::on_device`]
    /// directly when a graph slot for the pooler is needed regardless
    /// of what the checkpoint ships.
    ///
    /// `repo_id` is the HF-style identifier, e.g. `"roberta-base"` or
    /// `"FacebookAI/roberta-large"`. HF base checkpoints ship as
    /// `RobertaForMaskedLM` with an `lm_head` that a bare
    /// `RobertaModel` has no slot for; `load_weights_with_logging`
    /// tolerates those and names them on stderr.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_config_and_weights(repo_id, RobertaConfig::from_json_str)?;
        let graph = if weights_have_pooler(&weights)? {
            RobertaModel::on_device(&config, device)?
        } else {
            RobertaModel::on_device_without_pooler(&config, device)?
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("RobertaModel").to_json_str());
        Ok(graph)
    }
}

impl RobertaForSequenceClassification {
    /// Download a fine-tuned `RobertaForSequenceClassification`
    /// checkpoint from the Hub and return a ready-to-use predictor on
    /// CPU.
    ///
    /// Popular checkpoints:
    /// `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-label
    /// sentiment), `roberta-large-mnli` (3-label NLI),
    /// `SamLowe/roberta-base-go_emotions` (28 emotions).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, RobertaConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("RobertaForSequenceClassification")
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

impl RobertaForTokenClassification {
    /// Download a fine-tuned `RobertaForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub. Popular
    /// checkpoints: `Jean-Baptiste/roberta-large-ner-english`,
    /// `obi/deid_roberta_i2b2`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, RobertaConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("RobertaForTokenClassification")
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

impl RobertaForQuestionAnswering {
    /// Download a fine-tuned `RobertaForQuestionAnswering` checkpoint
    /// (SQuAD, etc.) from the Hub. Popular checkpoints:
    /// `deepset/roberta-base-squad2`, `csarron/roberta-base-squad-v1`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, RobertaConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("RobertaForQuestionAnswering")
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

impl RobertaForMaskedLM {
    /// Download a RoBERTa MLM checkpoint (`roberta-base`,
    /// `roberta-large`, any `*-mlm` domain-adaptation fine-tune) from
    /// the Hub.
    ///
    /// The decoder weight is tied to
    /// `roberta.embeddings.word_embeddings.weight`; checkpoints that
    /// redundantly save `lm_head.decoder.weight` alongside it, or ship
    /// an extra `lm_head.bias` tied-to-decoder-bias, load cleanly —
    /// `load_safetensors_into_graph_with_rename_allow_unused` silently
    /// ignores keys absent from the graph's deduplicated
    /// `named_parameters()`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, RobertaConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("RobertaForMaskedLM")
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
