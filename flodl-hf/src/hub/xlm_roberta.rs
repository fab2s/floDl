//! XLM-RoBERTa-family `from_pretrained` impls (backbone + 4 task heads).
//!
//! Each head mirrors the RoBERTa counterpart — XLM-R's state_dict uses
//! the `roberta.*` prefix verbatim, so the shared safetensors loader
//! path loads cleanly with no key rewriting beyond the standard legacy
//! LayerNorm.gamma/beta rename.

use flodl::{Device, Graph, Result};

use crate::models::xlm_roberta::{
    XlmRobertaConfig, XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering,
    XlmRobertaForSequenceClassification, XlmRobertaForTokenClassification, XlmRobertaModel,
};

use crate::safetensors_io::weights_have_pooler;

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, load_weights_with_logging};

impl XlmRobertaModel {
    /// Download a pretrained XLM-RoBERTa checkpoint from the HuggingFace
    /// Hub and return a fully-initialised [`Graph`] on CPU.
    ///
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't, matching
    /// whatever the Hub repo carries. `FacebookAI/xlm-roberta-base`
    /// keeps the pooler; many encoder-only variants (e.g.
    /// `xlm-roberta-base`) don't. HF base checkpoints ship as
    /// `XLMRobertaForMaskedLM` and the `lm_head.*` keys are tolerated
    /// (logged, ignored) by `load_weights_with_logging`.
    ///
    /// `repo_id` examples: `"xlm-roberta-base"`, `"xlm-roberta-large"`,
    /// `"FacebookAI/xlm-roberta-base"`.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_config_and_weights(repo_id, XlmRobertaConfig::from_json_str)?;
        let graph = if weights_have_pooler(&weights)? {
            XlmRobertaModel::on_device(&config, device)?
        } else {
            XlmRobertaModel::on_device_without_pooler(&config, device)?
        };
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("XLMRobertaModel").to_json_str());
        Ok(graph)
    }
}

impl XlmRobertaForSequenceClassification {
    /// Download a fine-tuned `XLMRobertaForSequenceClassification`
    /// checkpoint from the Hub. Popular checkpoints:
    /// `cardiffnlp/twitter-xlm-roberta-base-sentiment` (3-label
    /// multilingual sentiment), `joeddav/xlm-roberta-large-xnli`
    /// (zero-shot NLI), `papluca/xlm-roberta-base-language-detection`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, XlmRobertaConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("XLMRobertaForSequenceClassification")
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

impl XlmRobertaForTokenClassification {
    /// Download a fine-tuned `XLMRobertaForTokenClassification`
    /// checkpoint (NER, POS tagging, …) from the Hub. Popular
    /// checkpoints: `Davlan/xlm-roberta-base-ner-hrl` (multilingual
    /// NER), `Davlan/xlm-roberta-base-finetuned-conll03-english`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, XlmRobertaConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("XLMRobertaForTokenClassification")
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

impl XlmRobertaForQuestionAnswering {
    /// Download a fine-tuned `XLMRobertaForQuestionAnswering`
    /// checkpoint (multilingual SQuAD, …) from the Hub. Popular
    /// checkpoints: `deepset/xlm-roberta-base-squad2`,
    /// `deepset/xlm-roberta-large-squad2`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, XlmRobertaConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("XLMRobertaForQuestionAnswering")
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

impl XlmRobertaForMaskedLM {
    /// Download an XLM-RoBERTa MLM checkpoint (`xlm-roberta-base`,
    /// `xlm-roberta-large`, any `*-mlm` domain-adaptation fine-tune)
    /// from the Hub.
    ///
    /// The decoder weight is tied to
    /// `roberta.embeddings.word_embeddings.weight`; checkpoints that
    /// redundantly save `lm_head.decoder.weight`, or ship an extra
    /// `lm_head.bias` tied-to-decoder-bias, load cleanly — the loader
    /// silently ignores keys absent from the graph's deduplicated
    /// `named_parameters()`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, XlmRobertaConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("XLMRobertaForMaskedLM")
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
