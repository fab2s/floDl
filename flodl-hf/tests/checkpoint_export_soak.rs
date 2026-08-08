//! End-to-end soak for the `--checkpoint` re-export pipeline.
//!
//! Validates that the queue #3 (sidecar emission via
//! `AutoModel::from_pretrained_for_export` → `set_source_config` →
//! `save_checkpoint` writes `<stem>.config.json`) and queue #4
//! (`build_for_export` + `load_checkpoint` + `export_hf_dir` reads the
//! sidecar to rebuild the matching topology) paths actually round-trip:
//!
//! 1. `AutoModel::from_pretrained_for_export("bert-base-uncased")`
//!    fetches the Hub config, sets it on the returned graph as
//!    `source_config`.
//! 2. `graph.save_checkpoint(<tmp>.fdl)` emits both the `.fdl` and the
//!    `<tmp>.config.json` sidecar (queue #3 contract).
//! 3. Read the sidecar, parse via `AutoConfig::from_json_str`, instantiate
//!    a fresh graph via `build_for_export`, then load the `.fdl` weights
//!    into it (queue #4 contract — same shape as the example binary's
//!    `run_checkpoint`).
//! 4. `export_hf_dir` writes `model.safetensors` + `config.json`.
//! 5. Compare `model.safetensors` bytes bit-exact against a parallel
//!    direct-from-Hub export — proves --hub and --checkpoint produce
//!    equivalent output.
//!
//! `_live` because step 1 hits the Hub. Run with `fdl test-live`.

use std::path::PathBuf;

use flodl::{Device, Graph, checkpoint_keys};
use flodl_hf::export::{build_for_export, export_hf_dir};
use flodl_hf::models::auto::{AutoConfig, AutoModel};
use flodl_hf::models::distilbert::DistilBertForSequenceClassification;
use flodl_hf::safetensors_io::keys_have_pooler;

fn unique_tmp_root(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "flodl_hf_checkpoint_soak_{tag}_{}",
        std::process::id(),
    ))
}

fn sidecar_for_checkpoint(checkpoint: &str) -> PathBuf {
    // Mirrors flodl's pub(crate) `sidecar_config_path`: strip optional
    // `.gz`, then replace the trailing extension with `config.json`.
    let mut p = PathBuf::from(checkpoint);
    if p.extension().and_then(|e| e.to_str()) == Some("gz") {
        p.set_extension("");
    }
    p.set_extension("config.json");
    p
}

#[test]
#[ignore = "live: HF checkpoint-mode soak; requires network + hf-hub cache"]
fn bert_checkpoint_export_pipeline_soak_live() {
    let repo = "bert-base-uncased";
    let root = unique_tmp_root("bert");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // Step 1: Hub-fetch with source_config attached (queue #3).
    let graph_hub: Graph = AutoModel::from_pretrained_for_export(repo)
        .unwrap_or_else(|e| panic!("AutoModel::from_pretrained_for_export({repo}): {e}"));
    assert!(
        graph_hub.source_config().is_some(),
        "queue #3 contract: AutoModel::from_pretrained_for_export must set source_config",
    );

    // Step 2: save_checkpoint emits the sidecar (queue #3 contract).
    let ckpt = root.join("bert.fdl");
    let ckpt_str = ckpt.to_string_lossy().to_string();
    graph_hub
        .save_checkpoint(&ckpt_str)
        .unwrap_or_else(|e| panic!("save_checkpoint({ckpt_str}): {e}"));
    assert!(ckpt.exists(), "save_checkpoint produced no .fdl");
    let sidecar = sidecar_for_checkpoint(&ckpt_str);
    assert!(
        sidecar.exists(),
        "queue #3 contract: save_checkpoint did not emit sidecar at {}",
        sidecar.display(),
    );

    // Step 3: Read sidecar → parse → build_for_export → load_checkpoint
    // (queue #4 contract — what `run_checkpoint` does in the example
    // binary, here exercised at the library level).
    let sidecar_str = std::fs::read_to_string(&sidecar)
        .unwrap_or_else(|e| panic!("read sidecar {}: {e}", sidecar.display()));
    let config = AutoConfig::from_json_str(&sidecar_str).unwrap();
    assert_eq!(config.model_type(), "bert");
    let keys =
        checkpoint_keys(&ckpt_str).unwrap_or_else(|e| panic!("checkpoint_keys({ckpt_str}): {e}"));
    let has_pooler = keys_have_pooler(&keys);
    let graph_rebuilt: Graph =
        build_for_export(&config, has_pooler, Device::CPU).unwrap_or_else(|e| {
            panic!("build_for_export from sidecar: {e}");
        });
    let report = graph_rebuilt
        .load_checkpoint(&ckpt_str)
        .unwrap_or_else(|e| panic!("load_checkpoint({ckpt_str}): {e}"));
    assert!(
        report.missing.is_empty(),
        "checkpoint-mode rebuild missed {} keys: {:?}",
        report.missing.len(),
        &report.missing[..report.missing.len().min(5)],
    );

    // Step 4: Export the rebuilt graph (checkpoint mode output).
    let ckpt_export = root.join("checkpoint_export");
    export_hf_dir(&graph_rebuilt, &config.to_json_str(), &ckpt_export).unwrap();

    // Parallel: export the original Hub graph (hub mode reference).
    let hub_export = root.join("hub_export");
    export_hf_dir(&graph_hub, &config.to_json_str(), &hub_export).unwrap();

    // Step 5: Bit-exact compare model.safetensors between the two
    // pipelines. Both should serialise identical parameter sets.
    let ckpt_bytes = std::fs::read(ckpt_export.join("model.safetensors")).unwrap();
    let hub_bytes = std::fs::read(hub_export.join("model.safetensors")).unwrap();
    assert_eq!(
        ckpt_bytes.len(),
        hub_bytes.len(),
        "checkpoint-mode and hub-mode exports differ in size: {} vs {} bytes",
        ckpt_bytes.len(),
        hub_bytes.len(),
    );
    assert!(
        ckpt_bytes == hub_bytes,
        "checkpoint-mode and hub-mode exports differ bit-exact (same length, different bytes)",
    );

    eprintln!(
        "checkpoint soak: {} keys round-tripped; --checkpoint and --hub produced \
         bit-identical {}-byte model.safetensors",
        report.loaded.len(),
        ckpt_bytes.len(),
    );

    let _ = std::fs::remove_dir_all(&root);
}

/// Head-class soak — the actual fine-tune demo path.
///
/// Validates:
/// 1. `<Family>For<Head>::from_pretrained` sets `source_config` with
///    `architectures: ["<Family>For<Head>"]` (the with_architectures
///    fix — without it, a multi-head upstream class like
///    `BertForPreTraining` would leak through and trip
///    `classify_architecture` on re-export).
/// 2. `save_checkpoint` emits the sidecar.
/// 3. Sidecar → `AutoConfig::from_json_str` → `build_for_export`
///    dispatches to the matching head class via `classify_architecture`
///    (the regression scenario gap 1 closes).
/// 4. `load_checkpoint` into the rebuilt graph reports zero missing
///    keys.
/// 5. `export_hf_dir` produces a non-empty `model.safetensors` +
///    `config.json` carrying the head class as architecture.
///
/// Picks DistilBERT-SST-2 since it's the family/head wired into
/// `examples/distilbert_finetune` — the actual fine-tune demo path.
/// Skip the bit-exact bytes comparison the base soak does: a Hub-mode
/// export of a head checkpoint drops the head (by design — see
/// `Block 4c` in `project_hf_arc_todo.md`), so the two pipelines
/// won't match. The `_live` head-roundtrip tests in
/// `flodl-hf/tests/distilbert_head_export_roundtrip.rs` already
/// validate bit-exact bytes via the direct `head.graph()` path.
#[test]
#[ignore = "live: HF head checkpoint-mode soak; requires network + hf-hub cache"]
fn distilbert_seqcls_checkpoint_export_pipeline_soak_live() {
    let repo = "distilbert-base-uncased-finetuned-sst-2-english";
    let root = unique_tmp_root("distilbert_seqcls");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    // Step 1: head from_pretrained — sets source_config with the
    // normalised architectures (gap 1 fix).
    let head = DistilBertForSequenceClassification::from_pretrained(repo).unwrap_or_else(|e| {
        panic!("DistilBertForSequenceClassification::from_pretrained({repo}): {e}")
    });
    let source = head
        .graph()
        .source_config()
        .expect("from_pretrained must set source_config");
    let parsed = AutoConfig::from_json_str(&source).unwrap();
    let arch = parsed
        .architectures()
        .expect("source_config must carry architectures");
    assert_eq!(
        arch,
        ["DistilBertForSequenceClassification"],
        "from_pretrained must normalise architectures to the head class actually built; \
         without with_architectures the upstream Hub config leaks through and trips \
         classify_architecture on --checkpoint re-export",
    );

    // Step 2: save_checkpoint emits the sidecar.
    let ckpt = root.join("model.fdl");
    let ckpt_str = ckpt.to_string_lossy().to_string();
    head.graph()
        .save_checkpoint(&ckpt_str)
        .unwrap_or_else(|e| panic!("save_checkpoint({ckpt_str}): {e}"));
    let sidecar = {
        // Mirrors flodl's pub(crate) `sidecar_config_path` (and the
        // helper in the base soak above).
        let mut p = PathBuf::from(&ckpt_str);
        p.set_extension("config.json");
        p
    };
    assert!(
        sidecar.exists(),
        "save_checkpoint did not emit sidecar at {}",
        sidecar.display(),
    );

    // Step 3: read sidecar → build_for_export → load_checkpoint.
    // build_for_export's classify_architecture must dispatch to
    // HeadKind::SeqCls and produce a SeqCls graph.
    let sidecar_str = std::fs::read_to_string(&sidecar).unwrap();
    let config = AutoConfig::from_json_str(&sidecar_str).unwrap();
    let keys =
        checkpoint_keys(&ckpt_str).unwrap_or_else(|e| panic!("checkpoint_keys({ckpt_str}): {e}"));
    let has_pooler = keys_have_pooler(&keys);
    let rebuilt: Graph = build_for_export(&config, has_pooler, Device::CPU)
        .unwrap_or_else(|e| panic!("build_for_export from sidecar: {e}"));
    let report = rebuilt
        .load_checkpoint(&ckpt_str)
        .unwrap_or_else(|e| panic!("load_checkpoint({ckpt_str}): {e}"));
    assert!(
        report.missing.is_empty(),
        "head checkpoint-mode rebuild missed {} keys: {:?}",
        report.missing.len(),
        &report.missing[..report.missing.len().min(5)],
    );

    // Step 4: export_hf_dir produces a populated dir whose
    // config.json reflects the head class.
    let export_dir = root.join("export");
    export_hf_dir(&rebuilt, &config.to_json_str(), &export_dir).unwrap();
    let model_path = export_dir.join("model.safetensors");
    let config_path = export_dir.join("config.json");
    assert!(model_path.exists(), "model.safetensors not written");
    assert!(config_path.exists(), "config.json not written");
    let exported_config_str = std::fs::read_to_string(&config_path).unwrap();
    let exported_config = AutoConfig::from_json_str(&exported_config_str).unwrap();
    let exported_arch = exported_config
        .architectures()
        .expect("exported config.json must carry architectures");
    assert_eq!(
        exported_arch,
        ["DistilBertForSequenceClassification"],
        "exported config.json must reflect the head class",
    );

    let exported_bytes = std::fs::read(&model_path).unwrap();
    assert!(!exported_bytes.is_empty(), "model.safetensors is empty");

    eprintln!(
        "distilbert_seqcls checkpoint soak: {} keys round-tripped via \
         save_checkpoint → sidecar → build_for_export → load_checkpoint → \
         export_hf_dir; exported model.safetensors {} bytes",
        report.loaded.len(),
        exported_bytes.len(),
    );

    let _ = std::fs::remove_dir_all(&root);
}
