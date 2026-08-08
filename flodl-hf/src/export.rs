//! HuggingFace-compatible export: write a trained flodl model as a
//! directory HF Python can load with `AutoModel.from_pretrained`.
//!
//! # What this produces
//!
//! A directory containing:
//! - `model.safetensors` — the graph's named parameters and buffers,
//!   serialised via [`crate::safetensors_io::save_safetensors_file_from_graph`].
//!   HF-dotted keys are applied (`gamma`/`beta` stay as modern
//!   `weight`/`bias`; no legacy rename is emitted).
//! - `config.json` — model architecture metadata. The caller provides
//!   this as a pre-built string, typically from the family's
//!   `to_json_str()` method (e.g.
//!   [`BertConfig::to_json_str`](crate::models::bert::BertConfig::to_json_str)).
//!
//! Tokenizer files are **not** emitted — `tokenizer.json`,
//! `vocab.txt`, `sentencepiece.bpe.model`, and the various `*_config.json`
//! files are orthogonal to model state and stay the caller's
//! responsibility (usually: copy them from the base checkpoint
//! directory).
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use flodl::Device;
//! use flodl_hf::export::export_hf_dir;
//! use flodl_hf::models::bert::{BertConfig, BertModel};
//!
//! # fn run() -> flodl::Result<()> {
//! let config = BertConfig::bert_base_uncased();
//! let graph  = BertModel::on_device(&config, Device::CPU)?;
//! export_hf_dir(&graph, &config.to_json_str(), Path::new("/tmp/my-bert"))?;
//! # Ok(())
//! # }
//! ```
//!
//! After this call, `/tmp/my-bert` is ready for
//! `AutoModel.from_pretrained("/tmp/my-bert")` in HF Python.

use std::path::Path;

use flodl::{Device, Graph, Result, TensorError};

use crate::models::albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
    AlbertForTokenClassification, AlbertModel,
};
use crate::models::auto::AutoConfig;
use crate::models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification, BertModel,
};
use crate::models::deberta_v2::{
    DebertaV2Config, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification, DebertaV2Model,
};
use crate::models::distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertModel,
};
use crate::models::roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel,
};
use crate::models::xlm_roberta::{
    XlmRobertaConfig, XlmRobertaForMaskedLM, XlmRobertaForQuestionAnswering,
    XlmRobertaForSequenceClassification, XlmRobertaForTokenClassification, XlmRobertaModel,
};
use crate::safetensors_io::save_safetensors_file_from_graph;

/// Write a HuggingFace-compatible export directory.
///
/// Creates `out_dir` (and any missing parents) if it doesn't exist,
/// then writes `model.safetensors` + `config.json` inside it. Both
/// files are overwritten on re-export — no staging or atomic rename
/// is attempted; the caller is expected to export into a fresh dir.
///
/// The `config_json` string is written verbatim. Build it with the
/// family's `to_json_str()` so the emitted `model_type` and field
/// names match what HF `AutoConfig` expects.
///
/// Returns an error if directory creation or file writes fail; I/O
/// error messages include the offending path.
pub fn export_hf_dir(graph: &Graph, config_json: &str, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir).map_err(|e| {
        TensorError::new(&format!(
            "export_hf_dir: create_dir_all {}: {e}",
            out_dir.display(),
        ))
    })?;

    let model_path = out_dir.join("model.safetensors");
    save_safetensors_file_from_graph(graph, &model_path)?;

    let config_path = out_dir.join("config.json");
    std::fs::write(&config_path, config_json).map_err(|e| {
        TensorError::new(&format!(
            "export_hf_dir: write {}: {e}",
            config_path.display(),
        ))
    })?;

    Ok(())
}

/// HF task-head category, parsed from a config's first `architectures`
/// entry by [`classify_architecture`]. Used to dispatch
/// [`build_for_export`] to the matching family head builder when a
/// checkpoint was saved from a fine-tuned model rather than a base
/// backbone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeadKind {
    /// Base backbone (`{Family}Model`) — no task-specific layers.
    Base,
    /// Sequence classification (`{Family}ForSequenceClassification`).
    SeqCls,
    /// Token classification / NER (`{Family}ForTokenClassification`).
    TokCls,
    /// Extractive question answering (`{Family}ForQuestionAnswering`).
    Qa,
    /// Masked language modelling (`{Family}ForMaskedLM`).
    Mlm,
}

/// 1:1 mapping from the Hub-mode head enum to the export-mode one.
///
/// The two enums carry identical variants on purpose: Hub-mode dispatch
/// ([`crate::hub::HubExportHead`]) and export-mode dispatch
/// ([`HeadKind`]) share the same five-way taxonomy. They live in
/// separate modules because their parsers differ — `HubExportHead`
/// accepts a permissive Hub auto-dispatch (unknown `For{Other}` → Base,
/// matching HF Python's `AutoModel.from_pretrained` policy), while
/// `classify_architecture` rejects unsupported `For{Other}` loudly.
///
/// This impl pins the variant correspondence in the type system: if
/// either enum gains a variant the other lacks, the exhaustive match
/// below stops compiling and the drift surfaces immediately.
impl From<crate::hub::HubExportHead> for HeadKind {
    fn from(h: crate::hub::HubExportHead) -> Self {
        use crate::hub::HubExportHead as H;
        match h {
            H::Base => HeadKind::Base,
            H::SeqCls => HeadKind::SeqCls,
            H::TokCls => HeadKind::TokCls,
            H::Qa => HeadKind::Qa,
            H::Mlm => HeadKind::Mlm,
        }
    }
}

/// Suffix-match an HF class name against the supported task heads.
///
/// Family-agnostic: every supported family follows the same naming
/// convention (`{Family}For...`), so a suffix match handles all six.
/// Returns:
/// - `Ok(HeadKind::SeqCls/TokCls/Qa/Mlm)` for the canonical task heads,
/// - `Ok(HeadKind::Base)` for `{Family}Model` and any non-`For*`
///   class name (permissive default for hand-edited or non-canonical
///   configs),
/// - `Err(...)` for `{Family}For{Other}` names that flodl-hf doesn't
///   yet build (NextSentencePrediction, MultipleChoice, Pretraining,
///   ...) — surfaces a loud message naming the supported set so the
///   user knows it's a flodl-hf scope issue, not a checkpoint issue.
fn classify_architecture(arch: &str) -> Result<HeadKind> {
    if arch.ends_with("ForSequenceClassification") {
        Ok(HeadKind::SeqCls)
    } else if arch.ends_with("ForTokenClassification") {
        Ok(HeadKind::TokCls)
    } else if arch.ends_with("ForQuestionAnswering") {
        Ok(HeadKind::Qa)
    } else if arch.ends_with("ForMaskedLM") {
        Ok(HeadKind::Mlm)
    } else if arch.ends_with("Model") || !arch.contains("For") {
        Ok(HeadKind::Base)
    } else {
        Err(TensorError::new(&format!(
            "build_for_export: unsupported architecture {arch:?}. \
             flodl-hf currently dispatches {{Model, ForSequenceClassification, \
             ForTokenClassification, ForQuestionAnswering, ForMaskedLM}}. \
             Other heads (NextSentencePrediction, MultipleChoice, Pretraining, \
             …) are planned for a future release.",
        )))
    }
}

/// Build a graph matching `config` for HF round-trip export, on `device`.
///
/// `has_pooler` selects between with-pooler and without-pooler variants
/// for pooler-bearing families (BERT, RoBERTa, XLM-R, ALBERT). Pass
/// `true` when the source checkpoint ships pooler weights — matches
/// HF `AutoModel.from_pretrained` parity. Pooler-less families
/// (DistilBERT, DeBERTa-v2) ignore the flag, and task-head dispatches
/// (which carry their own pooler / classifier wiring) ignore it too.
///
/// The first `config.architectures()` entry decides which builder runs.
/// Recognised suffixes:
/// `Model` (base backbone), `ForSequenceClassification`,
/// `ForTokenClassification`, `ForQuestionAnswering`, `ForMaskedLM`. An
/// absent or unrecognised non-`For*` value falls back to base. A
/// `For{Other}` value not in the list above errors loudly rather than
/// silently misdispatching.
///
/// Use this when you have a parsed `AutoConfig` (e.g. read from a
/// flodl checkpoint sidecar) and need to instantiate the matching
/// graph topology before loading weights via
/// [`flodl::Graph::load_checkpoint`]. For Hub-fetch flows reach for
/// [`crate::models::auto::AutoModel::from_pretrained_for_export`]
/// directly — it builds, fetches, and loads in one call.
pub fn build_for_export(config: &AutoConfig, has_pooler: bool, device: Device) -> Result<Graph> {
    let head = match config.architectures().and_then(|arr| arr.first()) {
        Some(name) => classify_architecture(name)?,
        None => HeadKind::Base,
    };
    build_for_export_with_head(config, has_pooler, head, device)
}

/// Explicit-head variant of [`build_for_export`]. Skips the
/// `architectures[0]` parse and dispatches the family + head match
/// directly. Hub-side base loaders
/// ([`AutoModel::from_pretrained`](crate::hub),
/// [`AutoModel::from_pretrained_for_export`](crate::hub)) call this
/// with [`HeadKind::Base`] so they don't trip
/// [`classify_architecture`] on multi-head pretraining class names
/// (`BertForPreTraining` etc.) that should silently fall through to
/// the bare backbone.
pub(crate) fn build_for_export_with_head(
    config: &AutoConfig,
    has_pooler: bool,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match config {
        AutoConfig::Bert(c) => build_bert_for_export(c, has_pooler, head, device),
        AutoConfig::Roberta(c) => build_roberta_for_export(c, has_pooler, head, device),
        AutoConfig::DistilBert(c) => build_distilbert_for_export(c, head, device),
        AutoConfig::XlmRoberta(c) => build_xlm_roberta_for_export(c, has_pooler, head, device),
        AutoConfig::Albert(c) => build_albert_for_export(c, has_pooler, head, device),
        AutoConfig::DebertaV2(c) => build_deberta_v2_for_export(c, head, device),
    }
}

fn build_bert_for_export(
    c: &BertConfig,
    has_pooler: bool,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match head {
        HeadKind::Base => {
            if has_pooler {
                BertModel::on_device(c, device)
            } else {
                BertModel::on_device_without_pooler(c, device)
            }
        }
        HeadKind::SeqCls => {
            let n = BertForSequenceClassification::num_labels_from_config(c)?;
            Ok(BertForSequenceClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::TokCls => {
            let n = BertForTokenClassification::num_labels_from_config(c)?;
            Ok(BertForTokenClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::Qa => Ok(BertForQuestionAnswering::on_device(c, device)?.graph),
        HeadKind::Mlm => Ok(BertForMaskedLM::on_device(c, device)?.graph),
    }
}

fn build_roberta_for_export(
    c: &RobertaConfig,
    has_pooler: bool,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match head {
        HeadKind::Base => {
            if has_pooler {
                RobertaModel::on_device(c, device)
            } else {
                RobertaModel::on_device_without_pooler(c, device)
            }
        }
        HeadKind::SeqCls => {
            let n = RobertaForSequenceClassification::num_labels_from_config(c)?;
            Ok(RobertaForSequenceClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::TokCls => {
            let n = RobertaForTokenClassification::num_labels_from_config(c)?;
            Ok(RobertaForTokenClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::Qa => Ok(RobertaForQuestionAnswering::on_device(c, device)?.graph),
        HeadKind::Mlm => Ok(RobertaForMaskedLM::on_device(c, device)?.graph),
    }
}

fn build_distilbert_for_export(
    c: &DistilBertConfig,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match head {
        HeadKind::Base => DistilBertModel::on_device(c, device),
        HeadKind::SeqCls => {
            let n = DistilBertForSequenceClassification::num_labels_from_config(c)?;
            Ok(DistilBertForSequenceClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::TokCls => {
            let n = DistilBertForTokenClassification::num_labels_from_config(c)?;
            Ok(DistilBertForTokenClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::Qa => Ok(DistilBertForQuestionAnswering::on_device(c, device)?.graph),
        HeadKind::Mlm => Ok(DistilBertForMaskedLM::on_device(c, device)?.graph),
    }
}

fn build_xlm_roberta_for_export(
    c: &XlmRobertaConfig,
    has_pooler: bool,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match head {
        HeadKind::Base => {
            if has_pooler {
                XlmRobertaModel::on_device(c, device)
            } else {
                XlmRobertaModel::on_device_without_pooler(c, device)
            }
        }
        HeadKind::SeqCls => {
            let n = XlmRobertaForSequenceClassification::num_labels_from_config(c)?;
            Ok(XlmRobertaForSequenceClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::TokCls => {
            let n = XlmRobertaForTokenClassification::num_labels_from_config(c)?;
            Ok(XlmRobertaForTokenClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::Qa => Ok(XlmRobertaForQuestionAnswering::on_device(c, device)?.graph),
        HeadKind::Mlm => Ok(XlmRobertaForMaskedLM::on_device(c, device)?.graph),
    }
}

fn build_albert_for_export(
    c: &AlbertConfig,
    has_pooler: bool,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match head {
        HeadKind::Base => {
            if has_pooler {
                AlbertModel::on_device(c, device)
            } else {
                AlbertModel::on_device_without_pooler(c, device)
            }
        }
        HeadKind::SeqCls => {
            let n = AlbertForSequenceClassification::num_labels_from_config(c)?;
            Ok(AlbertForSequenceClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::TokCls => {
            let n = AlbertForTokenClassification::num_labels_from_config(c)?;
            Ok(AlbertForTokenClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::Qa => Ok(AlbertForQuestionAnswering::on_device(c, device)?.graph),
        HeadKind::Mlm => Ok(AlbertForMaskedLM::on_device(c, device)?.graph),
    }
}

fn build_deberta_v2_for_export(
    c: &DebertaV2Config,
    head: HeadKind,
    device: Device,
) -> Result<Graph> {
    match head {
        HeadKind::Base => DebertaV2Model::on_device(c, device),
        HeadKind::SeqCls => {
            let n = DebertaV2ForSequenceClassification::num_labels_from_config(c)?;
            Ok(DebertaV2ForSequenceClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::TokCls => {
            let n = DebertaV2ForTokenClassification::num_labels_from_config(c)?;
            Ok(DebertaV2ForTokenClassification::on_device(c, n, device)?.graph)
        }
        HeadKind::Qa => Ok(DebertaV2ForQuestionAnswering::on_device(c, device)?.graph),
        HeadKind::Mlm => Ok(DebertaV2ForMaskedLM::on_device(c, device)?.graph),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bert::{BertConfig, BertModel};
    use flodl::Device;

    /// Unique tempdir root per test — `std::env::temp_dir()` + pid +
    /// tag matches the pattern used elsewhere in this crate (see
    /// `tests/roundtrip_common/mod.rs`) and avoids pulling in a
    /// `tempfile` dev-dep for two unit tests.
    fn unique_tempdir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("flodl_hf_export_{tag}_{}", std::process::id()))
    }

    /// Smoke test: a BERT preset exports into a fresh dir, producing
    /// exactly `model.safetensors` + `config.json` with non-empty
    /// contents and a config that parses back to the same preset.
    #[test]
    fn export_hf_dir_writes_both_files() {
        let out = unique_tempdir("writes_both").join("bert-export");
        let _ = std::fs::remove_dir_all(&out);

        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::on_device(&config, Device::CPU).unwrap();

        export_hf_dir(&graph, &config.to_json_str(), &out).unwrap();

        let model_path = out.join("model.safetensors");
        let config_path = out.join("config.json");

        assert!(model_path.exists(), "model.safetensors not written");
        assert!(config_path.exists(), "config.json not written");
        assert!(
            std::fs::metadata(&model_path).unwrap().len() > 0,
            "model.safetensors is empty",
        );

        // Config round-trips back through the parser.
        let written = std::fs::read_to_string(&config_path).unwrap();
        let recovered = BertConfig::from_json_str(&written).unwrap();
        assert_eq!(recovered.to_json_str(), config.to_json_str());

        let _ = std::fs::remove_dir_all(&out);
    }

    /// Creates missing parent directories — callers shouldn't need to
    /// pre-make the full path.
    #[test]
    fn export_hf_dir_creates_missing_parents() {
        let root = unique_tempdir("missing_parents");
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("b").join("c");

        let config = BertConfig::bert_base_uncased();
        let graph = BertModel::on_device(&config, Device::CPU).unwrap();

        export_hf_dir(&graph, &config.to_json_str(), &nested).unwrap();

        assert!(nested.join("model.safetensors").exists());
        assert!(nested.join("config.json").exists());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_for_export_dispatches_on_family() {
        // Each family returns Ok(_); pooler-toggle only affects bert/
        // roberta/xlm-roberta/albert.
        for has_pooler in [false, true] {
            let bert = AutoConfig::Bert(crate::models::bert::BertConfig::bert_base_uncased());
            assert!(build_for_export(&bert, has_pooler, Device::CPU).is_ok());
        }
        // DistilBERT is pooler-less; both bools should produce the same shape.
        let distil = AutoConfig::DistilBert(
            crate::models::distilbert::DistilBertConfig::distilbert_base_uncased(),
        );
        let g_a = build_for_export(&distil, false, Device::CPU).unwrap();
        let g_b = build_for_export(&distil, true, Device::CPU).unwrap();
        assert_eq!(g_a.structural_hash(), g_b.structural_hash());
    }

    #[test]
    fn hub_export_head_to_head_kind_round_trip() {
        use crate::hub::HubExportHead;
        assert_eq!(HeadKind::from(HubExportHead::Base), HeadKind::Base);
        assert_eq!(HeadKind::from(HubExportHead::SeqCls), HeadKind::SeqCls);
        assert_eq!(HeadKind::from(HubExportHead::TokCls), HeadKind::TokCls);
        assert_eq!(HeadKind::from(HubExportHead::Qa), HeadKind::Qa);
        assert_eq!(HeadKind::from(HubExportHead::Mlm), HeadKind::Mlm);
    }

    #[test]
    fn classify_architecture_suffix_match() {
        assert_eq!(classify_architecture("BertModel").unwrap(), HeadKind::Base,);
        assert_eq!(
            classify_architecture("BertForSequenceClassification").unwrap(),
            HeadKind::SeqCls,
        );
        assert_eq!(
            classify_architecture("RobertaForTokenClassification").unwrap(),
            HeadKind::TokCls,
        );
        assert_eq!(
            classify_architecture("DistilBertForQuestionAnswering").unwrap(),
            HeadKind::Qa,
        );
        assert_eq!(
            classify_architecture("AlbertForMaskedLM").unwrap(),
            HeadKind::Mlm,
        );
        // XLM uses uppercase XLMRoberta — suffix match handles it.
        assert_eq!(
            classify_architecture("XLMRobertaForSequenceClassification").unwrap(),
            HeadKind::SeqCls,
        );
        // DebertaV2 — same shape.
        assert_eq!(
            classify_architecture("DebertaV2ForMaskedLM").unwrap(),
            HeadKind::Mlm,
        );
    }

    #[test]
    fn classify_architecture_unrecognised_for_head_errors_loudly() {
        // NSP / MultipleChoice are real HF classes flodl-hf doesn't yet
        // build — error must name the supported set so users know it's a
        // scope issue, not a checkpoint issue.
        let err = classify_architecture("BertForNextSentencePrediction").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported architecture"), "got: {msg}");
        assert!(msg.contains("ForSequenceClassification"), "got: {msg}");
    }

    #[test]
    fn classify_architecture_unknown_non_for_falls_back_to_base() {
        // Hand-edited or non-canonical names without a "For" segment are
        // treated as base — load_checkpoint will surface a real error if
        // the structural hash is wrong.
        assert_eq!(
            classify_architecture("SomeHandRolledClass").unwrap(),
            HeadKind::Base,
        );
    }

    /// Architectures-driven dispatch produces a head graph whose
    /// structural hash differs from the base backbone — proves the
    /// dispatch actually rebuilds the right topology, not just the
    /// backbone twice.
    #[test]
    fn build_for_export_dispatches_to_seqcls_head_when_architectures_says_so() {
        // Tiny preset so the test stays cheap; real fields don't matter
        // for structural-hash dispatch, only that num_labels is set.
        let mut cfg = BertConfig::bert_base_uncased();
        cfg.num_labels = Some(3);
        cfg.architectures = Some(vec!["BertForSequenceClassification".to_string()]);
        let auto = AutoConfig::Bert(cfg.clone());

        let head_graph = build_for_export(&auto, true, Device::CPU).unwrap();

        // Compare against the same base config without the head hint.
        let mut base_cfg = cfg.clone();
        base_cfg.architectures = None;
        let base_graph = build_for_export(&AutoConfig::Bert(base_cfg), true, Device::CPU).unwrap();

        assert_ne!(
            head_graph.structural_hash(),
            base_graph.structural_hash(),
            "SeqCls head must produce a different structural hash than the base backbone",
        );
    }

    /// Round-trip: an unmodified config keeps `architectures = None`,
    /// the emitted JSON carries the canonical base name, re-parsing
    /// preserves None — i.e. the source was canonical and nothing got
    /// mangled.
    #[test]
    fn architectures_none_round_trips_as_base_default() {
        let cfg = BertConfig::bert_base_uncased();
        let s = cfg.to_json_str();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        let arr: Vec<&str> = v
            .get("architectures")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();
        assert_eq!(arr, vec!["BertModel"]);
    }

    /// Round-trip: when `architectures` carries a task-head name, both
    /// the emitted JSON and the re-parsed config preserve it verbatim.
    #[test]
    fn architectures_some_round_trips_verbatim() {
        let mut cfg = BertConfig::bert_base_uncased();
        cfg.num_labels = Some(2);
        cfg.architectures = Some(vec!["BertForSequenceClassification".to_string()]);
        let s = cfg.to_json_str();
        let r = BertConfig::from_json_str(&s).unwrap();
        assert_eq!(
            r.architectures.as_deref(),
            Some(&["BertForSequenceClassification".to_string()][..]),
        );
    }
}
