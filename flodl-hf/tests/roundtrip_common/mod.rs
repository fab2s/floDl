//! Shared helpers for the `*_roundtrip.rs` integration tests.
//!
//! Each per-family roundtrip test:
//! 1. Loads the HF checkpoint into a flodl model via `from_pretrained`.
//! 2. Saves the model with [`save_safetensors_file_from_graph`].
//! 3. Reads the HF reference safetensors directly from the `hf-hub`
//!    cache.
//! 4. Asserts every flodl-saved key exists on the HF side (after
//!    [`bert_legacy_layernorm_rename`] canonicalisation), shapes agree, and
//!    bytes are bit-exact.
//!
//! Helpers below cover the parts that don't change between families.
//! Each test file's body wires the family-specific `from_pretrained`
//! call and the repo id, then defers to [`run_roundtrip`].

#![allow(dead_code)] // helpers are wired in via `mod roundtrip_common;`

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use hf_hub::HFClientSync;
use safetensors::{SafeTensors, tensor::TensorView};

use flodl::Graph;
use flodl_hf::export::export_hf_dir;
use flodl_hf::safetensors_io::{
    bert_legacy_layernorm_rename, save_safetensors_file_from_graph, tensor_view_to_f32_vec,
};

/// Download one file from a Hub model repo, mirroring
/// `flodl_hf::hub::hub_download`. hf-hub 1.0 takes owner and name
/// separately; `split_id` yields `("", id)` for a canonical repo.
fn hub_get(repo_id: &str, filename: &str) -> Result<PathBuf, String> {
    let client = HFClientSync::new().expect("hf-hub init (set HF_HOME for cache location)");
    let (owner, name) = hf_hub::split_id(repo_id);
    client
        .model(owner, name)
        .download_file()
        .filename(filename)
        .send()
        .map_err(|e| e.to_string())
}

/// Resolve the on-disk safetensors path that `from_pretrained` would
/// use for `repo_id`. Mirrors `flodl_hf::hub::fetch_safetensors`:
///
/// 1. `<HF_HOME>/flodl-converted/<repo_id>/model.safetensors` —
///    produced by `fdl flodl-hf convert <repo_id>` for `.bin`-only
///    repos like `microsoft/deberta-v3-base`. When present, this is
///    what flodl actually loaded, so the roundtrip's reference IS
///    this converted file.
/// 2. `client.model(owner, name).download_file()` — the normal Hub
///    fetch via `hf-hub`'s on-disk cache.
fn hf_safetensors_path(repo_id: &str) -> PathBuf {
    let hf_home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|h| h.join(".cache").join("huggingface"))
                .unwrap_or_else(|| PathBuf::from("/tmp/huggingface"))
        });
    let converted = hf_home
        .join("flodl-converted")
        .join(repo_id)
        .join("model.safetensors");
    if converted.exists() {
        return converted;
    }

    hub_get(repo_id, "model.safetensors")
        .unwrap_or_else(|e| {
            panic!(
                "fetch {repo_id}/model.safetensors: {e}\n\
                 If this repo ships only `pytorch_model.bin`, run `fdl flodl-hf convert {repo_id}` first.",
            )
        })
}

/// Fetch `config.json` for `repo_id` from the hf-hub cache (same path
/// `from_pretrained` uses). Useful when a test needs both the graph
/// (via `*Model::from_pretrained`) AND the family-specific `*Config`
/// — `from_pretrained` only returns the graph.
pub fn fetch_hf_config_json(repo_id: &str) -> String {
    let path = hub_get(repo_id, "config.json")
        .unwrap_or_else(|e| panic!("fetch {repo_id}/config.json: {e}"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Decode a safetensors `TensorView` into a flat `Vec<f32>` using the
/// exact same dtype rules flodl applies on load (f32 zero-cost; f16 /
/// bf16 / f64 widened on the host). DeBERTa-v3-base ships its
/// `intermediate.dense.bias` and a few other tensors as F16, so the
/// roundtrip's reference comparison must widen with the same kernel.
fn decode_f32(view: &TensorView<'_>, key: &str) -> Vec<f32> {
    tensor_view_to_f32_vec(view).unwrap_or_else(|e| {
        panic!("decode {key:?} as f32: {e}");
    })
}

/// Run the full roundtrip check for a loaded `graph` against the
/// `repo_id`'s on-disk HF reference. Asserts:
///
/// - every flodl-saved key is present on the HF side after applying
///   [`bert_legacy_layernorm_rename`] to the HF keys;
/// - shapes agree on every shared key;
/// - tensor bytes are bit-exact (f32 in / f32 out, no numerical work
///   between load and save).
///
/// Cleans up the temp save file before asserting so a failed run doesn't
/// leak it.
pub fn run_roundtrip(graph: &Graph, repo_id: &str, family_label: &str) {
    let saved_path = std::env::temp_dir().join(format!(
        "flodl_{family_label}_roundtrip_{}.safetensors",
        std::process::id(),
    ));
    save_safetensors_file_from_graph(graph, &saved_path)
        .unwrap_or_else(|e| panic!("save_safetensors_file_from_graph: {e}"));

    let flodl_bytes = std::fs::read(&saved_path)
        .unwrap_or_else(|e| panic!("read flodl-saved {}: {e}", saved_path.display()));
    let _ = std::fs::remove_file(&saved_path);
    let flodl_st = SafeTensors::deserialize(&flodl_bytes).expect("parse flodl-saved file");

    let hf_path = hf_safetensors_path(repo_id);
    let hf_bytes = std::fs::read(&hf_path)
        .unwrap_or_else(|e| panic!("read HF cache {}: {e}", hf_path.display()));
    let hf_st = SafeTensors::deserialize(&hf_bytes).expect("parse HF reference file");

    // Canonicalise HF keys via the same rename flodl applies on load,
    // so flodl-saved keys (already canonical) line up.
    let mut hf_canonical: HashMap<String, String> = HashMap::new();
    for name in hf_st.names() {
        let canonical = bert_legacy_layernorm_rename(name);
        if let Some(prev) = hf_canonical.insert(canonical.clone(), name.to_string()) {
            panic!(
                "{family_label}: HF-side rename collision: {prev:?} and {name:?} \
                 both map to canonical {canonical:?}",
            );
        }
    }

    let flodl_keys: HashSet<String> = flodl_st.names().iter().map(|s| s.to_string()).collect();
    let hf_canonical_keys: HashSet<&str> = hf_canonical.keys().map(|s| s.as_str()).collect();

    let missing: Vec<&str> = flodl_keys
        .iter()
        .filter(|k| !hf_canonical_keys.contains(k.as_str()))
        .map(|s| s.as_str())
        .collect();
    assert!(
        missing.is_empty(),
        "{family_label}: {} flodl-saved key(s) absent from HF reference: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)],
    );

    let mut diffs: Vec<(String, usize)> = Vec::new();
    let mut shape_mismatches: Vec<(String, Vec<usize>, Vec<usize>)> = Vec::new();
    for fk in &flodl_keys {
        let hf_orig = hf_canonical.get(fk).unwrap();
        let f_view = flodl_st.tensor(fk).unwrap();
        let h_view = hf_st.tensor(hf_orig).unwrap();
        assert_eq!(
            f_view.dtype(),
            h_view.dtype(),
            "{family_label}: flodl-saved tensor {fk:?} dtype {:?} != HF reference {:?}",
            f_view.dtype(),
            h_view.dtype(),
        );
        let f_shape = f_view.shape().to_vec();
        let h_shape = h_view.shape().to_vec();
        if f_shape != h_shape {
            shape_mismatches.push((fk.clone(), f_shape, h_shape));
            continue;
        }
        let f_data = decode_f32(&f_view, fk);
        let h_data = decode_f32(&h_view, hf_orig);
        if f_data != h_data {
            let n_mismatches = f_data
                .iter()
                .zip(h_data.iter())
                .filter(|(a, b)| a != b)
                .count();
            diffs.push((fk.clone(), n_mismatches));
        }
    }
    assert!(
        shape_mismatches.is_empty(),
        "{family_label}: shape mismatches: {:?}",
        &shape_mismatches[..shape_mismatches.len().min(10)],
    );
    assert!(
        diffs.is_empty(),
        "{family_label}: {} key(s) differ bit-exact: {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(10)],
    );

    eprintln!(
        "{family_label} roundtrip: {} keys verified bit-exact vs HF reference \
         ({} HF keys total, {} unused on flodl side)",
        flodl_keys.len(),
        hf_canonical_keys.len(),
        hf_canonical_keys.len() - flodl_keys.len(),
    );
}

/// End-to-end export roundtrip via
/// [`export_hf_dir`]: load HF into flodl, export to a fresh directory,
/// then verify:
///
/// - both `model.safetensors` and `config.json` land in the out-dir;
/// - `config.json` is non-empty valid JSON with a `model_type` field
///   (HF `AutoConfig` dispatch key);
/// - `model.safetensors` bytes are bit-exact vs the HF reference
///   (reusing the same comparator as [`run_roundtrip`]).
///
/// Complement to [`run_roundtrip`]: that tests the low-level save API;
/// this tests the full user-facing export path. Shares the bit-exact
/// comparison because `export_hf_dir` wraps `save_safetensors_file_from_graph`
/// — exercising both paths catches wiring regressions (e.g. the wrong
/// file name, missing `config.json` write) that a save-only test would
/// miss.
///
/// The `config_json` string is what the caller would write to
/// `config.json` — pass the family's `config.to_json_str()` output.
pub fn run_export_roundtrip(graph: &Graph, config_json: &str, repo_id: &str, family_label: &str) {
    let out_dir = std::env::temp_dir().join(format!(
        "flodl_{family_label}_export_{}",
        std::process::id(),
    ));
    // Defensive clean so a crashed prior run doesn't poison the test.
    let _ = std::fs::remove_dir_all(&out_dir);

    export_hf_dir(graph, config_json, &out_dir).unwrap_or_else(|e| panic!("export_hf_dir: {e}"));

    let model_path = out_dir.join("model.safetensors");
    let config_path = out_dir.join("config.json");
    assert!(
        model_path.exists(),
        "{family_label}: model.safetensors not written to {}",
        out_dir.display(),
    );
    assert!(
        config_path.exists(),
        "{family_label}: config.json not written to {}",
        out_dir.display(),
    );

    // Config is valid JSON with an HF dispatch key.
    let config_bytes = std::fs::read(&config_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", config_path.display()));
    let config_v: serde_json::Value = serde_json::from_slice(&config_bytes)
        .unwrap_or_else(|e| panic!("{family_label}: config.json is not valid JSON: {e}"));
    assert!(
        config_v
            .get("model_type")
            .and_then(|x| x.as_str())
            .is_some(),
        "{family_label}: config.json missing `model_type` (HF AutoConfig dispatch key)",
    );

    // Model bytes bit-exact vs HF reference — same comparator as
    // `run_roundtrip`, just reading from the exported dir instead of
    // a bare tempfile.
    let flodl_bytes =
        std::fs::read(&model_path).unwrap_or_else(|e| panic!("read {}: {e}", model_path.display()));
    let flodl_st =
        SafeTensors::deserialize(&flodl_bytes).expect("parse exported model.safetensors");

    let hf_path = hf_safetensors_path(repo_id);
    let hf_bytes = std::fs::read(&hf_path)
        .unwrap_or_else(|e| panic!("read HF cache {}: {e}", hf_path.display()));
    let hf_st = SafeTensors::deserialize(&hf_bytes).expect("parse HF reference file");

    let mut hf_canonical: HashMap<String, String> = HashMap::new();
    for name in hf_st.names() {
        let canonical = bert_legacy_layernorm_rename(name);
        hf_canonical.insert(canonical, name.to_string());
    }

    let flodl_keys: HashSet<String> = flodl_st.names().iter().map(|s| s.to_string()).collect();

    let mut diffs: Vec<String> = Vec::new();
    for fk in &flodl_keys {
        let hf_orig = hf_canonical.get(fk).unwrap_or_else(|| {
            panic!("{family_label}: exported key {fk:?} absent from HF reference")
        });
        let f_view = flodl_st.tensor(fk).unwrap();
        let h_view = hf_st.tensor(hf_orig).unwrap();
        assert_eq!(
            f_view.dtype(),
            h_view.dtype(),
            "{family_label}: exported tensor {fk:?} dtype {:?} != HF reference {:?}",
            f_view.dtype(),
            h_view.dtype(),
        );
        assert_eq!(
            f_view.shape(),
            h_view.shape(),
            "{family_label}: exported {fk:?} shape {:?} != HF {:?}",
            f_view.shape(),
            h_view.shape(),
        );
        let f_data = decode_f32(&f_view, fk);
        let h_data = decode_f32(&h_view, hf_orig);
        if f_data != h_data {
            diffs.push(fk.clone());
        }
    }
    assert!(
        diffs.is_empty(),
        "{family_label}: {} exported key(s) differ bit-exact from HF: {:?}",
        diffs.len(),
        &diffs[..diffs.len().min(10)],
    );

    eprintln!(
        "{family_label} export roundtrip: {} keys bit-exact, config.json written ({} bytes)",
        flodl_keys.len(),
        config_bytes.len(),
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}

/// True when a `from_pretrained` error is the "`.bin`-only repo" case.
///
/// Matches `Entry not found` PLUS the filename, never the whole message:
/// the best-effort tokenizer attach also reports `Entry not found`
/// (for `tokenizer.json`) on healthy runs, and a looser match would
/// swallow genuine failures as skips.
pub fn is_unconverted_bin_only(err: &str) -> bool {
    err.contains("Entry not found") && err.contains("model.safetensors")
}

/// Report a test skipped for want of a converted checkpoint, and say
/// exactly how to fix it. A silent early return reported as `ok` is the
/// false-green this project refuses elsewhere (`ci/rig-ladder.sh`).
pub fn skip_unconverted(model_id: &str) {
    eprintln!(
        "SKIPPED: {model_id} ships only `pytorch_model.bin`, so flodl has no \
         `model.safetensors` to load.\n  \
         This is a missing local artifact, not a parity failure. Convert it once:\n    \
         fdl flodl-hf convert {model_id}\n  \
         The result lands in `$HF_HOME/flodl-converted/{model_id}/` and \
         `from_pretrained` picks it up automatically."
    );
}
/// Unwrap a `from_pretrained` result, or LOUDLY skip when the repo ships
/// only `pytorch_model.bin`.
///
/// flodl loads `model.safetensors`; the Python parity/reference tooling
/// reads `.bin` happily. So for a `.bin`-only repo the fixture exists
/// while the Rust side cannot reach the weights -- a missing local
/// artifact, not a disagreement, and one that recurs on every fresh
/// clone or wiped cache. Left as a failure it is a permanent red that
/// teaches people to ignore red.
///
/// Returns `None` after printing; callers `return` on `None`. A repo
/// that does not resolve at all still PANICS -- that is a real error.
pub fn or_skip<T, E: std::fmt::Display>(r: Result<T, E>, model_id: &str) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) if is_unconverted_bin_only(&e.to_string()) => {
            skip_unconverted(model_id);
            None
        }
        Err(e) => panic!("from_pretrained({model_id}): {e}"),
    }
}
