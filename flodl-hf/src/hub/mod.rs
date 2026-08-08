//! HuggingFace Hub download and local cache integration.
//!
//! Thin façade over the [`hf_hub`] crate: pulls `config.json` and
//! `model.safetensors` from a Hub repo, parses the config, builds the
//! matching flodl model, and loads the weights. Downloads go through
//! `hf_hub`'s cache, so a second call on the same repo reuses the local
//! copy.
//!
//! Entry points live as inherent methods on the model types
//! ([`crate::models::bert::BertModel::from_pretrained`]) so user code
//! reads the same as HF Python's `BertModel.from_pretrained(...)`.
//!
//! Per-family `from_pretrained` impls live in their own submodules
//! (`bert`, `roberta`, `distilbert`, `xlm_roberta`, `albert`,
//! `deberta_v2`). The `Auto*` cross-family dispatch and the
//! [`HubExportHead`] enum live in `auto`. The infrastructure shared
//! across all families (Hub fetch, pooler detection, weight load with
//! logging, optional tokenizer attach) lives at the bottom of this
//! file.

use std::path::PathBuf;

use hf_hub::HFClientSync;

use flodl::{Graph, Result, TensorError};

use crate::safetensors_io::{
    bert_legacy_key_rename, load_safetensors_into_graph_with_rename_allow_unused,
};
#[cfg(feature = "tokenizer")]
use crate::tokenizer::HfTokenizer;

mod albert;
mod auto;
mod bert;
mod deberta_v2;
mod distilbert;
mod roberta;
mod xlm_roberta;

pub use auto::HubExportHead;

// ── shared infrastructure ────────────────────────────────────────────────

/// Environment variable `fetch_safetensors` honours when looking for a
/// locally-converted `model.safetensors` before hitting the Hub.
/// Matches the `HF_HOME` the Docker services are configured with
/// (`/workspace/.hf-cache` via docker-compose.yml).
const HF_HOME_ENV: &str = "HF_HOME";

/// Default cache root when `HF_HOME` is not set. Mirrors HF Python's
/// `~/.cache/huggingface/` convention.
fn default_hf_home() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("huggingface")
    } else {
        PathBuf::from("/tmp/huggingface")
    }
}

/// Locally-converted safetensors path for `repo_id`, if one exists.
///
/// flodl-hf writes converted weights to
/// `<HF_HOME>/flodl-converted/<repo_id>/model.safetensors` via
/// `fdl flodl-hf convert <repo_id>`. Checking this path first lets
/// `from_pretrained` transparently use the converted copy without
/// touching the network.
fn flodl_converted_path(repo_id: &str) -> PathBuf {
    let hf_home = std::env::var_os(HF_HOME_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(default_hf_home);
    hf_home
        .join("flodl-converted")
        .join(repo_id)
        .join("model.safetensors")
}

/// Resolve `model.safetensors` for `repo_id`, preferring a locally-
/// converted copy over the Hub.
///
/// Order:
/// 1. `<HF_HOME>/flodl-converted/<repo_id>/model.safetensors` — produced
///    by `fdl flodl-hf convert <repo_id>` for `.bin`-only repos.
/// 2. [`hub_download`] for `model.safetensors` — the normal Hub fetch,
///    goes through hf-hub's own on-disk cache.
///
/// On failure at step 2, the returned error explicitly points the user
/// at `fdl flodl-hf convert <repo_id>` for the common `.bin`-only case.
fn fetch_safetensors(client: &HFClientSync, repo_id: &str) -> Result<PathBuf> {
    let converted = flodl_converted_path(repo_id);
    if converted.exists() {
        eprintln!(
            "from_pretrained({repo_id}): using flodl-converted safetensors at {}",
            converted.display(),
        );
        return Ok(converted);
    }
    hub_download(client, repo_id, "model.safetensors").map_err(|e| {
        TensorError::new(&format!(
            "{e}\n\
             If this repo ships only `pytorch_model.bin`, convert it first:\n  \
             fdl flodl-hf convert {repo_id}",
        ))
    })
}

/// Build a Hub client.
///
/// The cache location follows `HF_HOME` (then `XDG_CACHE_HOME`, then
/// `~/.cache/huggingface`) inside hf-hub itself, which matters here: the
/// dev container's `$HOME` is ephemeral, so a client that ignored
/// `HF_HOME` would redownload every run.
fn hub_client() -> Result<HFClientSync> {
    HFClientSync::new().map_err(|e| TensorError::new(&format!("hf-hub init: {e}")))
}

/// Download one file from a Hub model repo, returning its cached path.
///
/// `repo_id` is the usual `owner/name`; hf-hub takes the two halves
/// separately and `split_id` does the split, yielding `("", id)` for a
/// bare name.
///
/// PREFER THE OWNER-QUALIFIED ID. The Hub answers a bare legacy name
/// (the pre-2024 short form, no owner segment) with a 307 to its
/// owner-qualified home; hf-hub 0.4's `ureq` transport followed that
/// redirect, hf-hub 1.0 does not, and reports `Repository not found`
/// instead. Every repo id in this crate is owner-qualified for that
/// reason, which is also the form the Hub itself now documents.
fn hub_download(client: &HFClientSync, repo_id: &str, filename: &str) -> Result<PathBuf> {
    let (owner, name) = hf_hub::split_id(repo_id);
    client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .send()
        .map_err(|e| TensorError::new(&format!("hf-hub fetch {repo_id}/{filename}: {e}")))
}

/// Fetch a Hub repo's `config.json` as a string, going through
/// hf-hub's on-disk cache. Extracted from
/// [`fetch_config_str_and_weights`] so config-only callers (e.g.
/// `AutoConfig::from_pretrained`) don't pay the safetensors read.
fn fetch_config_str(repo_id: &str) -> Result<String> {
    let client = hub_client()?;
    let config_path = hub_download(&client, repo_id, "config.json")?;
    std::fs::read_to_string(&config_path)
        .map_err(|e| TensorError::new(&format!("read {}: {e}", config_path.display())))
}

/// Pull `config.json` + `model.safetensors` from a Hub repo and return
/// `(config_string, weights_bytes)`. Config parsing is left to the
/// caller so the same fetch path serves every model family
/// (`BertConfig::from_json_str`, `RobertaConfig::from_json_str`, …).
fn fetch_config_str_and_weights(repo_id: &str) -> Result<(String, Vec<u8>)> {
    let config_str = fetch_config_str(repo_id)?;

    let client = hub_client()?;
    let weights_path = fetch_safetensors(&client, repo_id)?;
    let weights = std::fs::read(&weights_path)
        .map_err(|e| TensorError::new(&format!("read {}: {e}", weights_path.display())))?;
    Ok((config_str, weights))
}

/// Pull `config.json` + `model.safetensors` from a Hub repo, parse the
/// config via `parse`, and return `(config, weights)`. Generic over the
/// family's config type so each family's `from_pretrained` impl stays a
/// one-liner without 6 near-identical per-family wrappers.
///
/// `parse` is typically a family `from_json_str` associated function
/// (e.g. `BertConfig::from_json_str`); the closure form keeps callers
/// from needing turbofish syntax since `C` is inferred from `parse`'s
/// return type.
fn fetch_config_and_weights<C, F>(repo_id: &str, parse: F) -> Result<(C, Vec<u8>)>
where
    F: FnOnce(&str) -> Result<C>,
{
    let (config_str, weights) = fetch_config_str_and_weights(repo_id)?;
    let config = parse(&config_str)?;
    Ok((config, weights))
}

/// Best-effort tokenizer download for task-head `from_pretrained` paths.
///
/// `HfTokenizer::from_pretrained` requires the repo to ship a fast-
/// tokenizer `tokenizer.json`. Legacy checkpoints (pre-~2022, many
/// older fine-tunes, hand-uploaded models) only carry the slow-tokenizer
/// triple `tokenizer_config.json` + `vocab.txt` + `special_tokens_map.json`;
/// HF Python rebuilds a fast tokenizer from those on the fly, but the
/// Rust `tokenizers` crate does not. Failing `from_pretrained` over a
/// missing `tokenizer.json` breaks the HF-API parity that AutoModel (no
/// required tokenizer) ships. We log and continue — `predict()` /
/// `answer()` will then error with a clear "attach a tokenizer" message
/// at call time.
#[cfg(feature = "tokenizer")]
fn try_load_tokenizer(repo_id: &str) -> Option<HfTokenizer> {
    match HfTokenizer::from_pretrained(repo_id) {
        Ok(tok) => Some(tok),
        Err(e) => {
            // Common case: legacy repos ship only the slow-tokenizer
            // triple (vocab.txt + tokenizer_config.json + special_tokens_map.json),
            // no `tokenizer.json`. Surface one actionable line without
            // echoing the URL or the verbose HfTokenizer error.
            let terse = if e.to_string().contains("404") {
                "no tokenizer.json on Hub".to_string()
            } else {
                e.to_string()
            };
            eprintln!(
                "from_pretrained({repo_id}): tokenizer not attached ({terse}) \
                 — predict()/answer() need .with_tokenizer()",
            );
            None
        }
    }
}

/// Load safetensors into a graph, logging any discarded checkpoint keys
/// to stderr. Shared by every `from_pretrained` path.
fn load_weights_with_logging(repo_id: &str, graph: &Graph, bytes: &[u8]) -> Result<()> {
    let unused =
        load_safetensors_into_graph_with_rename_allow_unused(graph, bytes, bert_legacy_key_rename)?;
    if !unused.is_empty() {
        eprintln!(
            "from_pretrained({repo_id}): ignored {} checkpoint key(s) not used by the model:",
            unused.len(),
        );
        for k in unused.iter().take(20) {
            eprintln!("  - {k}");
        }
        if unused.len() > 20 {
            eprintln!("  ... and {} more", unused.len() - 20);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::models::bert::BertModel;
    use flodl::Device;

    /// Live-network integration test: pulls `google-bert/bert-base-uncased` from the
    /// HuggingFace Hub, builds the graph, loads the weights, and runs a
    /// forward pass. `#[ignore]` by default — run manually with
    /// `fdl test -- --ignored bert_from_pretrained_live` when the host
    /// has network access (and is happy to cache ~440MB of weights under
    /// `~/.cache/huggingface/`).
    #[test]
    #[ignore = "network + ~440MB cache write"]
    fn bert_from_pretrained_live() {
        use crate::models::bert::build_extended_attention_mask;
        use flodl::nn::Module;
        use flodl::{DType, Tensor, TensorOptions, Variable};

        let graph = BertModel::from_pretrained("google-bert/bert-base-uncased").unwrap();
        graph.eval();

        // Tiny forward pass to prove the loaded graph works end-to-end.
        let dev = Device::CPU;
        let batch = 1;
        let seq = 4;
        let input_ids = Variable::new(
            Tensor::from_i64(&[101, 7592, 2088, 102], &[batch, seq], dev).unwrap(),
            false,
        );
        let position_ids = Variable::new(
            Tensor::from_i64(&[0, 1, 2, 3], &[batch, seq], dev).unwrap(),
            false,
        );
        let token_type_ids = Variable::new(
            Tensor::from_i64(&[0, 0, 0, 0], &[batch, seq], dev).unwrap(),
            false,
        );
        let mask_flat = Tensor::ones(
            &[batch, seq],
            TensorOptions {
                dtype: DType::Float32,
                device: dev,
            },
        )
        .unwrap();
        let attention_mask =
            Variable::new(build_extended_attention_mask(&mask_flat).unwrap(), false);

        let out = graph
            .forward_multi(&[input_ids, position_ids, token_type_ids, attention_mask])
            .unwrap();
        assert_eq!(out.shape(), vec![batch, 768]);
    }

    /// Tiny synthetic BERT config — small enough that allocating a head
    /// graph in unit tests is millisecond-scale. Mirrors the
    /// `tiny_bert_config` private helper in `models/bert.rs` tests.
    fn tiny_bert_config() -> crate::models::bert::BertConfig {
        crate::models::bert::BertConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 32,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            pad_token_id: Some(0),
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            hidden_act: flodl::nn::GeluApprox::Exact,
            num_labels: None,
            id2label: None,
            architectures: None,
        }
    }

    /// Sidecar path naming mirror of flodl's `sidecar_config_path`:
    /// strip `.gz` if present, then replace the trailing extension with
    /// `config.json`. Used by `save_checkpoint` when `source_config`
    /// is set on the graph.
    fn sidecar_for_checkpoint(checkpoint: &str) -> std::path::PathBuf {
        let mut p = std::path::PathBuf::from(checkpoint);
        if p.extension().and_then(|e| e.to_str()) == Some("gz") {
            p.set_extension("");
        }
        p.set_extension("config.json");
        p
    }

    /// Regression test for the head-side architecture normalisation:
    /// every `<Family>For{Head}::from_pretrained_on_device` MUST stamp
    /// `source_config` with `architectures: ["<Family>For{Head}"]`,
    /// regardless of what the upstream Hub config advertised. Otherwise
    /// a subsequent `save_checkpoint → --checkpoint re-export` cycle
    /// trips `classify_architecture` on the multi-head class name some
    /// Hub repos ship (`google-bert/bert-base-uncased` → `BertForPreTraining`,
    /// which `classify_architecture` rejects loudly).
    ///
    /// The test simulates the from_pretrained_on_device contract
    /// without hitting the Hub: synthesizes a config whose upstream
    /// `architectures` is `BertForPreTraining`, then applies the same
    /// `with_architectures("BertForMaskedLM")` call the loader does
    /// before stamping `source_config`. save_checkpoint emits the
    /// sidecar; build_for_export reads it and dispatches to MLM.
    #[test]
    fn head_save_checkpoint_emits_normalised_architectures_sidecar() {
        use crate::export::build_for_export;
        use crate::models::auto::AutoConfig;
        use crate::models::bert::BertForMaskedLM;

        // Upstream-style config carrying the multi-head class name a
        // user pulling `google-bert/bert-base-uncased` from the Hub would see.
        let upstream = tiny_bert_config().with_architectures("BertForPreTraining");

        // Build the MLM head and stamp source_config exactly the way
        // BertForMaskedLM::from_pretrained_on_device does post-fix.
        let head = BertForMaskedLM::on_device(&upstream, Device::CPU).unwrap();
        head.graph()
            .set_source_config(upstream.with_architectures("BertForMaskedLM").to_json_str());

        // save_checkpoint emits the sidecar.
        let pid = std::process::id();
        let ckpt = std::env::temp_dir().join(format!("flodl_hf_mlm_norm_{pid}.fdl"));
        let ckpt_str = ckpt.to_string_lossy().to_string();
        head.graph().save_checkpoint(&ckpt_str).unwrap();
        let sidecar = sidecar_for_checkpoint(&ckpt_str);
        let sidecar_str = std::fs::read_to_string(&sidecar).unwrap();

        // Sidecar carries the head class, not the upstream multi-head class.
        let parsed = AutoConfig::from_json_str(&sidecar_str).unwrap();
        let arch = parsed.architectures().unwrap();
        assert_eq!(
            arch,
            ["BertForMaskedLM"],
            "save_checkpoint sidecar must reflect the head class actually built; \
             without the with_architectures call upstream's BertForPreTraining \
             would leak through and fail classify_architecture on re-export",
        );

        // build_for_export dispatches without an "unsupported architecture" error
        // and produces a graph with the same structural hash as the original.
        let rebuilt = build_for_export(&parsed, false, Device::CPU).unwrap();
        assert_eq!(
            rebuilt.structural_hash(),
            head.graph().structural_hash(),
            "build_for_export from sidecar must rebuild the same MLM topology",
        );

        // Cleanup.
        let _ = std::fs::remove_file(&ckpt);
        let _ = std::fs::remove_file(&sidecar);
    }
}
