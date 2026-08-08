//! AutoConfig / AutoModel Hub dispatch + the shared `HubExportHead` enum.
//!
//! Model-type dispatch: pull `config.json`, read `model_type`, then route
//! to the matching family. The backbone path ([`AutoModel`]) returns a
//! [`Graph`] whose shape is always `[batch, seq_len, hidden]` — BERT is
//! routed through [`BertModel::on_device_without_pooler`](crate::models::bert::BertModel::on_device_without_pooler) to keep that
//! invariant uniform across families. The task-head paths return the
//! corresponding `AutoModelFor*` enum
//! ([`AutoModelForSequenceClassification`] et al.), dispatching the
//! tokenizer attach + weight load through each concrete family's
//! loader.

use hf_hub::api::sync::ApiBuilder;

use flodl::{Device, Graph, Result, TensorError};

use crate::export::{HeadKind, build_for_export_with_head};
use crate::models::albert::{
    AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
    AlbertForTokenClassification,
};
use crate::models::auto::{
    AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
};
use crate::models::bert::{
    BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification,
};
use crate::models::deberta_v2::{
    DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering, DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
};
use crate::models::distilbert::{
    DistilBertForMaskedLM, DistilBertForQuestionAnswering, DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
};
use crate::models::roberta::{
    RobertaForMaskedLM, RobertaForQuestionAnswering, RobertaForSequenceClassification,
    RobertaForTokenClassification,
};
use crate::models::xlm_roberta::{
    XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering, XlmRobertaForSequenceClassification,
    XlmRobertaForTokenClassification,
};

#[cfg(feature = "tokenizer")]
use crate::tokenizer::HfTokenizer;

use crate::safetensors_io::weights_have_pooler;

#[cfg(feature = "tokenizer")]
use super::try_load_tokenizer;
use super::{fetch_config_and_weights, fetch_config_str, load_weights_with_logging};

/// Fetch a `config.json` string and safetensors blob, then parse the
/// config through [`AutoConfig::from_json_str`]. Shared by every
/// `AutoModelFor*::from_pretrained` path so model-type dispatch
/// happens exactly once per call.
fn fetch_auto_config_and_weights(repo_id: &str) -> Result<(AutoConfig, Vec<u8>)> {
    fetch_config_and_weights(repo_id, AutoConfig::from_json_str)
}

impl AutoConfig {
    /// Fetch `config.json` for `repo_id` from the HuggingFace Hub and
    /// parse it via [`AutoConfig::from_json_str`], dispatching on
    /// `model_type` to the matching family.
    ///
    /// Useful when a caller needs the parsed config independently of
    /// the weights — e.g. [`crate::export::export_hf_dir`] needs it to
    /// emit the output dir's `config.json`. hf-hub's on-disk cache
    /// means repeated calls (including alongside
    /// [`AutoModel::from_pretrained`]) don't re-download.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        let config_str = fetch_config_str(repo_id)?;
        AutoConfig::from_json_str(&config_str)
    }
}

impl AutoModel {
    /// Download a pretrained encoder from the HuggingFace Hub, auto-
    /// detecting the family (BERT / RoBERTa / DistilBERT) from the
    /// repo's `config.json`.
    ///
    /// The returned [`Graph`] emits `last_hidden_state` of shape
    /// `[batch, seq_len, hidden]` across all three families — BERT is
    /// routed through [`BertModel::on_device_without_pooler`](crate::models::bert::BertModel::on_device_without_pooler) so its
    /// output stays consistent with RoBERTa and DistilBERT. If the
    /// BERT pooler output is specifically required, call
    /// [`BertModel::from_pretrained`](crate::models::bert::BertModel) directly.
    ///
    /// `repo_id` examples:
    /// - `"bert-base-uncased"` (BERT)
    /// - `"roberta-base"` (RoBERTa)
    /// - `"distilbert/distilbert-base-uncased"` (DistilBERT)
    ///
    /// The returned graph's `forward_multi` input count differs by
    /// family (BERT: 4, RoBERTa: 3, DistilBERT: 2); see the
    /// [`auto`](crate::models::auto) module documentation.
    pub fn from_pretrained(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of [`from_pretrained`](Self::from_pretrained).
    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        // Force the no-pooler variant of pooler-bearing families so this
        // loader's output stays uniform `[batch, seq_len, hidden]` across
        // all six families (BERT's pooler would otherwise produce a
        // `[batch, hidden]` shape that diverges from RoBERTa/DistilBERT).
        // Pooler-less families ignore the flag.
        let graph = build_for_export_with_head(
            &config,
            /*has_pooler=*/ false,
            HeadKind::Base,
            device,
        )?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        // Normalise `architectures` to the base class name actually
        // being built (e.g. `bert-base-uncased` ships
        // `architectures: ["BertForPreTraining"]` but this loader
        // builds a bare `BertModel` with the head dropped). Without
        // normalisation, a subsequent `save_checkpoint` sidecar would
        // carry an arch name that `classify_architecture` rejects on
        // `--checkpoint` re-export.
        let arch = config.base_class_name();
        graph.set_source_config(config.into_normalized_config_json(arch));
        Ok(graph)
    }

    /// Download a pretrained model preserving the **full HF base layout**,
    /// including the pooler where the family defines one.
    ///
    /// Use this for round-trip-to-HF export workflows: HF's
    /// `AutoModel.from_pretrained(<exported_dir>)` expects every parameter
    /// the original Hub checkpoint shipped, and silently re-initializes
    /// any it can't find. The default
    /// [`from_pretrained`](Self::from_pretrained) drops family-specific
    /// pooler nodes for cross-family `last_hidden_state` consistency, so
    /// exports built from it are missing pooler weights and fail HF-side
    /// fidelity checks.
    ///
    /// Pooler-bearing families (BERT, RoBERTa, XLM-R, ALBERT) auto-pick
    /// `on_device` vs `on_device_without_pooler` by inspecting the
    /// checkpoint: with-pooler when pooler weights ship, without-pooler
    /// when the Hub repo is encoder-only (e.g. `roberta-base`,
    /// `FacebookAI/xlm-roberta-base`). Pooler-less families (DistilBERT,
    /// DeBERTa-v2) behave identically to
    /// [`from_pretrained`](Self::from_pretrained).
    ///
    /// **Auto-detect head from `architectures[0]`** (no `--head` flag).
    /// The upstream config's `architectures[0]` suffix decides which
    /// head this loader builds:
    ///
    /// | suffix                          | builder                                |
    /// |---------------------------------|-----------------------------------------|
    /// | `*ForSequenceClassification`    | `AutoModelForSequenceClassification`    |
    /// | `*ForTokenClassification`       | `AutoModelForTokenClassification`       |
    /// | `*ForQuestionAnswering`         | `AutoModelForQuestionAnswering`         |
    /// | `*ForMaskedLM`                  | `AutoModelForMaskedLM`                  |
    /// | `*Model` / no `For`             | base backbone (this method's old default) |
    /// | unrecognised `For{Other}`       | base backbone, with an info message     |
    ///
    /// Multi-head pretrained class names (`BertForPreTraining` etc.)
    /// fall back to the base backbone — mirrors HF Python's
    /// `AutoModel.from_pretrained` behaviour, where loading a
    /// pretraining checkpoint as `AutoModel` silently drops the heads.
    pub fn from_pretrained_for_export(repo_id: &str) -> Result<Graph> {
        Self::from_pretrained_for_export_on_device(repo_id, Device::CPU)
    }

    /// Device-aware variant of
    /// [`from_pretrained_for_export`](Self::from_pretrained_for_export).
    pub fn from_pretrained_for_export_on_device(repo_id: &str, device: Device) -> Result<Graph> {
        // Auto-detect the head from architectures[0]. Fetch only
        // config.json first (cheap) so we can decide the dispatch
        // without paying the full safetensors download for paths that
        // delegate to a head loader (which fetches its own).
        let config_str = fetch_config_str(repo_id)?;
        let probe = AutoConfig::from_json_str(&config_str)?;
        let arch_first = probe.architectures().and_then(|a| a.first().cloned());
        let head_kind = match arch_first.as_deref() {
            Some(arch) => classify_for_hub_export(arch),
            None => HubExportHead::Base,
        };
        Self::from_pretrained_for_export_with_head_on_device(repo_id, head_kind, device)
    }

    /// Force a specific head class instead of dispatching on
    /// `architectures[0]`. Useful for re-exporting a pretraining
    /// checkpoint as a feature-extraction encoder
    /// ([`HubExportHead::Base`]) or for cross-checking the base path
    /// against a checkpoint that advertises a head.
    pub fn from_pretrained_for_export_with_head(
        repo_id: &str,
        head: HubExportHead,
    ) -> Result<Graph> {
        Self::from_pretrained_for_export_with_head_on_device(repo_id, head, Device::CPU)
    }

    /// Device-aware variant of
    /// [`from_pretrained_for_export_with_head`](Self::from_pretrained_for_export_with_head).
    pub fn from_pretrained_for_export_with_head_on_device(
        repo_id: &str,
        head: HubExportHead,
        device: Device,
    ) -> Result<Graph> {
        match head {
            HubExportHead::Base => Self::base_from_pretrained_for_export(repo_id, device),
            HubExportHead::SeqCls => {
                let head =
                    AutoModelForSequenceClassification::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
            HubExportHead::TokCls => {
                let head =
                    AutoModelForTokenClassification::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
            HubExportHead::Qa => {
                let head =
                    AutoModelForQuestionAnswering::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
            HubExportHead::Mlm => {
                let head = AutoModelForMaskedLM::from_pretrained_on_device(repo_id, device)?;
                Ok(head.into_graph())
            }
        }
    }

    /// Base-backbone path of [`Self::from_pretrained_for_export_on_device`].
    /// Stays separate from the auto-dispatcher so the
    /// `<Family>Model::on_device` vs `<Family>Model::on_device_without_pooler`
    /// pooler-presence logic doesn't bleed into the head paths
    /// (heads carry their own pooler / classifier wiring already).
    fn base_from_pretrained_for_export(repo_id: &str, device: Device) -> Result<Graph> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;

        // Pick with-pooler vs without-pooler dynamically based on what
        // the checkpoint actually ships. Some Hub repos for pooler-
        // bearing families are encoder-only (e.g. `roberta-base`,
        // `FacebookAI/xlm-roberta-base`). Building with a pooler whose
        // weights aren't in the checkpoint trips the missing-keys
        // validation in `load_safetensors_into_graph_with_rename_allow_unused`.
        let has_pooler = weights_have_pooler(&weights)?;

        let graph = build_for_export_with_head(&config, has_pooler, HeadKind::Base, device)?;
        load_weights_with_logging(repo_id, &graph, &weights)?;
        // Normalise `architectures` to the base class name actually
        // being built. The Hub's source config typically tags a head
        // class (e.g. `bert-base-uncased` ships
        // `architectures: ["BertForPreTraining"]`) while this loader,
        // mirroring HF's `AutoModel.from_pretrained`, builds the base
        // backbone and silently drops head keys. The sidecar emitted
        // by a subsequent `save_checkpoint` must reflect what was
        // actually built, otherwise `build_for_export` re-dispatches
        // to the head class on `--checkpoint` re-export and produces
        // a graph whose structural hash no longer matches the saved
        // file (see `flodl-hf/tests/checkpoint_export_soak.rs`).
        let arch = config.base_class_name();
        graph.set_source_config(config.into_normalized_config_json(arch));
        Ok(graph)
    }
}

/// Head class to force when calling
/// [`AutoModel::from_pretrained_for_export_with_head`]. Use [`Self::Base`]
/// to bypass the `architectures[0]` auto-dispatch and load the bare
/// backbone (handy for re-exporting a pretraining checkpoint as a
/// feature-extraction encoder, or for stress-testing the base path
/// against a checkpoint that advertises a head).
///
/// Mirrors `export::HeadKind` but kept separate because the Hub-mode
/// auto-dispatch policy is more permissive: unrecognised `For{Other}`
/// suffixes fall back to base instead of erroring (a `bert-base-uncased`
/// checkpoint advertising `BertForPreTraining` is a real Hub case that
/// should still produce a base backbone, mirroring HF Python's
/// `AutoModel.from_pretrained`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubExportHead {
    Base,
    SeqCls,
    TokCls,
    Qa,
    Mlm,
}

impl HubExportHead {
    /// Parse the CLI-facing token form (`base`, `seqcls`, `tokcls`,
    /// `qa`, `mlm`). The auto-dispatch alias `auto` is intentionally
    /// not recognised here; callers route `auto` through the bare
    /// [`AutoModel::from_pretrained_for_export`] path which already
    /// reads `architectures[0]`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "base" => Ok(Self::Base),
            "seqcls" => Ok(Self::SeqCls),
            "tokcls" => Ok(Self::TokCls),
            "qa" => Ok(Self::Qa),
            "mlm" => Ok(Self::Mlm),
            other => Err(TensorError::new(&format!(
                "unsupported head `{other}`; expected one of \
                 auto|base|seqcls|tokcls|qa|mlm"
            ))),
        }
    }
}

/// Permissive variant of `export::classify_architecture` for Hub-mode
/// auto-dispatch. Unsupported `For{Other}` (multi-head pretraining
/// classes etc.) falls back to base instead of erroring; the
/// downstream `<Family>Model::on_device` path tolerates head-class
/// keys via `allow_unused` on weight load. Recognised heads dispatch
/// to the matching `AutoModelFor*` loader.
fn classify_for_hub_export(arch: &str) -> HubExportHead {
    if arch.ends_with("ForSequenceClassification") {
        HubExportHead::SeqCls
    } else if arch.ends_with("ForTokenClassification") {
        HubExportHead::TokCls
    } else if arch.ends_with("ForQuestionAnswering") {
        HubExportHead::Qa
    } else if arch.ends_with("ForMaskedLM") {
        HubExportHead::Mlm
    } else {
        // `*Model` / no-`For` / unsupported `For{Other}` → base.
        // Multi-head class names like `BertForPreTraining` land here,
        // matching HF Python's `AutoModel.from_pretrained` policy of
        // building the base backbone and silently dropping the heads.
        HubExportHead::Base
    }
}

impl AutoModelForSequenceClassification {
    /// Download a fine-tuned sequence-classification checkpoint from
    /// the Hub, auto-detecting the family from `config.json`. The
    /// config must carry `num_labels` (or `id2label`); otherwise the
    /// concrete family's `num_labels_from_config` surfaces a loud
    /// error.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let num_labels = BertForSequenceClassification::num_labels_from_config(&c)?;
                let h = BertForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("BertForSequenceClassification")
                    .to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let num_labels = RobertaForSequenceClassification::num_labels_from_config(&c)?;
                let h = RobertaForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("RobertaForSequenceClassification")
                    .to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let num_labels = DistilBertForSequenceClassification::num_labels_from_config(&c)?;
                let h = DistilBertForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("DistilBertForSequenceClassification")
                    .to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let num_labels = XlmRobertaForSequenceClassification::num_labels_from_config(&c)?;
                let h = XlmRobertaForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("XLMRobertaForSequenceClassification")
                    .to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let num_labels = AlbertForSequenceClassification::num_labels_from_config(&c)?;
                let h = AlbertForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("AlbertForSequenceClassification")
                    .to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let num_labels = DebertaV2ForSequenceClassification::num_labels_from_config(&c)?;
                let h = DebertaV2ForSequenceClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("DebertaV2ForSequenceClassification")
                    .to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AutoModelForTokenClassification {
    /// Download a fine-tuned token-classification checkpoint (NER,
    /// POS, …) from the Hub, auto-detecting the family.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let num_labels = BertForTokenClassification::num_labels_from_config(&c)?;
                let h = BertForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("BertForTokenClassification")
                    .to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let num_labels = RobertaForTokenClassification::num_labels_from_config(&c)?;
                let h = RobertaForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("RobertaForTokenClassification")
                    .to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let num_labels = DistilBertForTokenClassification::num_labels_from_config(&c)?;
                let h = DistilBertForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("DistilBertForTokenClassification")
                    .to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let num_labels = XlmRobertaForTokenClassification::num_labels_from_config(&c)?;
                let h = XlmRobertaForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("XLMRobertaForTokenClassification")
                    .to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let num_labels = AlbertForTokenClassification::num_labels_from_config(&c)?;
                let h = AlbertForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("AlbertForTokenClassification")
                    .to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let num_labels = DebertaV2ForTokenClassification::num_labels_from_config(&c)?;
                let h = DebertaV2ForTokenClassification::on_device(&c, num_labels, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("DebertaV2ForTokenClassification")
                    .to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AutoModelForQuestionAnswering {
    /// Download a fine-tuned extractive-QA checkpoint (SQuAD, …) from
    /// the Hub, auto-detecting the family. QA heads have a fixed
    /// 2-wide output (start, end logits) independent of `num_labels`,
    /// so the config carries no head-size metadata requirement.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let h = BertForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("BertForQuestionAnswering")
                    .to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let h = RobertaForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("RobertaForQuestionAnswering")
                    .to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let h = DistilBertForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("DistilBertForQuestionAnswering")
                    .to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let h = XlmRobertaForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("XLMRobertaForQuestionAnswering")
                    .to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let h = AlbertForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("AlbertForQuestionAnswering")
                    .to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let h = DebertaV2ForQuestionAnswering::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c
                    .with_architectures("DebertaV2ForQuestionAnswering")
                    .to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

impl AutoModelForMaskedLM {
    /// Download a masked-language-modelling checkpoint from the Hub,
    /// auto-detecting the family. MLM heads have no `num_labels`
    /// requirement — the decoder is tied to the word-embedding table
    /// and outputs `vocab_size` logits per position.
    ///
    /// Typical use is continued pretraining / domain adaptation on
    /// base checkpoints (`bert-base-uncased`, `roberta-base`,
    /// `distilbert-base-uncased`); each family's `from_pretrained`
    /// tolerates the redundant decoder-weight key some HF save
    /// formats ship, silently ignoring it during load.
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        Self::from_pretrained_on_device(repo_id, Device::CPU)
    }

    pub fn from_pretrained_on_device(repo_id: &str, device: Device) -> Result<Self> {
        let (config, weights) = fetch_auto_config_and_weights(repo_id)?;
        let (head, config_json) = match config {
            AutoConfig::Bert(c) => {
                let h = BertForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("BertForMaskedLM").to_json_str();
                (Self::Bert(h), cj)
            }
            AutoConfig::Roberta(c) => {
                let h = RobertaForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("RobertaForMaskedLM").to_json_str();
                (Self::Roberta(h), cj)
            }
            AutoConfig::DistilBert(c) => {
                let h = DistilBertForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DistilBertForMaskedLM").to_json_str();
                (Self::DistilBert(h), cj)
            }
            AutoConfig::XlmRoberta(c) => {
                let h = XlmRobertaForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("XLMRobertaForMaskedLM").to_json_str();
                (Self::XlmRoberta(h), cj)
            }
            AutoConfig::Albert(c) => {
                let h = AlbertForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("AlbertForMaskedLM").to_json_str();
                (Self::Albert(h), cj)
            }
            AutoConfig::DebertaV2(c) => {
                let h = DebertaV2ForMaskedLM::on_device(&c, device)?;
                load_weights_with_logging(repo_id, h.graph(), &weights)?;
                let cj = c.with_architectures("DebertaV2ForMaskedLM").to_json_str();
                (Self::DebertaV2(h), cj)
            }
        };
        head.graph().set_source_config(config_json);
        #[cfg(feature = "tokenizer")]
        let head = match try_load_tokenizer(repo_id) {
            Some(tok) => head.with_tokenizer(tok),
            None => head,
        };
        Ok(head)
    }
}

#[cfg(feature = "tokenizer")]
impl HfTokenizer {
    /// Download `tokenizer.json` from a HuggingFace Hub repo and wrap it.
    ///
    /// Uses the same `hf_hub` cache as
    /// [`BertModel::from_pretrained`](crate::models::bert::BertModel), so
    /// a model already pulled from a given repo won't re-download the
    /// tokenizer either (and vice versa).
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        let api = ApiBuilder::from_env()
            .build()
            .map_err(|e| TensorError::new(&format!("hf-hub init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo.get("tokenizer.json").map_err(|e| {
            TensorError::new(&format!("hf-hub fetch {repo_id}/tokenizer.json: {e}"))
        })?;
        Self::from_file(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hub_export_head_parse_round_trip() {
        assert_eq!(HubExportHead::parse("base").unwrap(), HubExportHead::Base);
        assert_eq!(
            HubExportHead::parse("seqcls").unwrap(),
            HubExportHead::SeqCls
        );
        assert_eq!(
            HubExportHead::parse("tokcls").unwrap(),
            HubExportHead::TokCls
        );
        assert_eq!(HubExportHead::parse("qa").unwrap(), HubExportHead::Qa);
        assert_eq!(HubExportHead::parse("mlm").unwrap(), HubExportHead::Mlm);
    }

    #[test]
    fn hub_export_head_parse_rejects_auto_alias() {
        // `auto` is the CLI-only sentinel; the typed enum demands an
        // explicit head and routes auto-dispatch through the bare
        // `from_pretrained_for_export` path.
        let err = HubExportHead::parse("auto").unwrap_err().to_string();
        assert!(err.contains("auto|base|seqcls"), "missing hint: {err}");
    }

    #[test]
    fn hub_export_head_parse_rejects_unknown() {
        let err = HubExportHead::parse("classifier").unwrap_err().to_string();
        assert!(err.contains("classifier"), "missing offending value: {err}");
        assert!(err.contains("auto|base|seqcls"), "missing hint: {err}");
    }
}
