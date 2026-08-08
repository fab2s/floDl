//! BERT-family `from_pretrained` impls (backbone + 4 task heads).

use flodl::{Device, Graph, Result};

use crate::models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification, BertModel,
};

use crate::safetensors_io::weights_have_pooler;

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, load_weights_with_logging};

impl BertModel {
    /// Download a pretrained BERT checkpoint from the HuggingFace Hub and
    /// return a fully-initialised [`Graph`] on CPU.
    ///
    /// Convenience wrapper over [`BertModel::from_pretrained_on_device`]
    /// with `Device::CPU`.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Download a pretrained BERT checkpoint from the HuggingFace Hub and
    /// return a fully-initialised [`Graph`] on `device`.
    ///
    /// Pulls `config.json` and `model.safetensors` from `repo_id` via
    /// `hf_hub` (using its on-disk cache), parses the config, builds a
    /// matching graph, and loads the safetensors weights.
    ///
    /// Picks `on_device` (with pooler) when the checkpoint ships pooler
    /// weights and `on_device_without_pooler` when it doesn't, so an
    /// encoder-only BERT checkpoint loads strict without missing-key
    /// errors and a pooler-bearing one keeps its `pooler_output` slot.
    /// Reach for [`BertModel::on_device`] /
    /// [`BertModel::on_device_without_pooler`] directly when the call
    /// site needs a guaranteed shape regardless of what the Hub repo
    /// happens to ship.
    ///
    /// `repo_id` is the HF-style identifier, e.g. `"google-bert/bert-base-uncased"`
    /// or `"google-bert/bert-base-multilingual-cased"`.
    ///
    /// Errors on: hub API init failure, network / HTTP failure,
    /// config parse failure, shape/key mismatch against the built graph,
    /// and any I/O error reading the cached safetensors file. Nothing
    /// partial is returned — the graph is either fully loaded or the
    /// call errors out.
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_config_and_weights(repo_id, BertConfig::from_json_str)?;
        let graph = if weights_have_pooler(&weights)? {
            BertModel::on_device(&config, device)?
        } else {
            BertModel::on_device_without_pooler(&config, device)?
        };
        // HF base checkpoints (e.g. `google-bert/bert-base-uncased`) ship as
        // `BertForPreTraining`, which carries MLM + NSP heads that a
        // bare `BertModel` has no slot for. `load_weights_with_logging`
        // tolerates those and names them on stderr.
        load_weights_with_logging(repo_id, &graph, &weights)?;
        graph.set_source_config(config.with_architectures("BertModel").to_json_str());
        Ok(graph)
    }
}

impl BertForSequenceClassification {
    /// Download a fine-tuned `BertForSequenceClassification` checkpoint
    /// from the Hub and return a ready-to-use predictor on CPU.
    ///
    /// The config must carry `num_labels` (or `id2label`) so the head's
    /// output width is known. Popular checkpoints: `nateraw/bert-base-uncased-emotion`
    /// (6 emotions, `.bin`-only — needs `fdl flodl-hf convert` first),
    /// `nlptown/bert-base-multilingual-uncased-sentiment` (5-star rating,
    /// safetensors on main), `unitary/toxic-bert` (6-label toxicity).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, BertConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("BertForSequenceClassification")
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

impl BertForTokenClassification {
    /// Download a fine-tuned `BertForTokenClassification` checkpoint
    /// (NER, POS tagging, …) from the Hub. Popular checkpoints:
    /// `dslim/bert-base-NER`,
    /// `dbmdz/bert-large-cased-finetuned-conll03-english`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, BertConfig::from_json_str)?;
        let num_labels = Self::num_labels_from_config(&config)?;
        let head = Self::on_device(&config, num_labels, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("BertForTokenClassification")
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

impl BertForQuestionAnswering {
    /// Download a fine-tuned `BertForQuestionAnswering` checkpoint
    /// (SQuAD, etc.) from the Hub. Popular checkpoints:
    /// `csarron/bert-base-uncased-squad-v1`,
    /// `bert-large-uncased-whole-word-masking-finetuned-squad`.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, BertConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph().set_source_config(
            config
                .with_architectures("BertForQuestionAnswering")
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

impl BertForMaskedLM {
    /// Download a BERT MLM checkpoint (`google-bert/bert-base-uncased`,
    /// `bert-base-cased`, any `*-mlm` fine-tune) from the Hub.
    ///
    /// The decoder weight is tied to the word-embedding table, so
    /// checkpoints that redundantly save `cls.predictions.decoder.weight`
    /// alongside `bert.embeddings.word_embeddings.weight` (HF's
    /// historical save format) load cleanly — the decoder key is
    /// ignored by `load_safetensors_into_graph_with_rename_allow_unused`,
    /// and the embedding key populates the single tied Parameter.
    /// Checkpoints that skip the redundant decoder key (modern HF saves)
    /// also load cleanly.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_config_and_weights(repo_id, BertConfig::from_json_str)?;
        let head = Self::on_device(&config, device)?;
        load_weights_with_logging(repo_id, head.graph(), &weights)?;
        head.graph()
            .set_source_config(config.with_architectures("BertForMaskedLM").to_json_str());
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}
