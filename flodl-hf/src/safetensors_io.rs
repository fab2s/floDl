//! Safetensors format I/O and load-time validation.
//!
//! This module's primary public surface today is [`LoadValidation`] — a
//! structured diff between what a flodl model expects and what a HuggingFace
//! safetensors file actually contains. It surfaces three failure modes:
//!
//! 1. **Missing** — keys the model expects but the checkpoint lacks
//!    (typo in a `FlowBuilder::tag(...)` string, wrong model variant, etc.).
//! 2. **Unused** — keys the checkpoint contains but the model never asks for
//!    (architecture mismatch, stale pretrained file, etc.).
//! 3. **Shape mismatch** — a key matches by name but the tensor dimensions
//!    disagree (vocab size change, hidden dim change, head count change).
//!
//! The validator is the safety net behind flodl-hf's "string-named tag"
//! convention: typos and drift are caught at load time with a loud,
//! actionable error listing every key that disagrees.
//!
//! Once validation passes, [`load_safetensors_into_graph`] (and the file
//! variant, [`load_safetensors_file_into_graph`]) copy tensor data from
//! the checkpoint into the graph's `Parameter` and `Buffer` storage
//! in-place. Checkpoint dtypes other than f32 (f16, bf16, f64) are cast
//! to f32 on the host; integer dtypes are not supported (BERT-style
//! models only store floats).
//!
//! # Example (validator only)
//!
//! ```
//! use std::collections::HashMap;
//! use flodl_hf::safetensors_io::{ExpectedParam, validate_keys};
//!
//! let expected = vec![
//!     ExpectedParam { key: "bert.embeddings.word_embeddings.weight".into(), shape: vec![30522, 768] },
//!     ExpectedParam { key: "bert.pooler.dense.bias".into(),              shape: vec![768] },
//! ];
//! let mut actual: HashMap<String, Vec<i64>> = HashMap::new();
//! actual.insert("bert.embeddings.word_embeddings.weight".into(), vec![30522, 768]);
//! actual.insert("bert.pooler.dense.bias".into(),                  vec![768]);
//!
//! let v = validate_keys(&expected, &actual);
//! assert!(v.is_ok());
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;

use flodl::{DType, Device, Graph, Result, Tensor, TensorError};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use crate::path::hf_key_from_flodl_key;

/// A parameter the model expects to find in a checkpoint.
///
/// `key`: the HF-dotted key as it appears in a safetensors file
/// (e.g. `bert.encoder.layer.0.attention.self.query.weight`).
/// `shape`: the tensor shape flodl will try to assign to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedParam {
    pub key: String,
    pub shape: Vec<i64>,
}

/// A single shape disagreement between model and checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatch {
    pub key: String,
    pub expected: Vec<i64>,
    pub found: Vec<i64>,
}

/// The result of validating a checkpoint's key set against a model's
/// expected parameters.
///
/// Build this with [`validate_keys`]. Check [`is_ok`](LoadValidation::is_ok)
/// or convert to a loud `Result` via
/// [`into_result`](LoadValidation::into_result).
#[derive(Debug, Default, Clone)]
pub struct LoadValidation {
    pub missing: Vec<String>,
    pub unused: Vec<String>,
    pub shape_mismatches: Vec<ShapeMismatch>,
}

impl LoadValidation {
    /// True when there are no missing keys, no unused keys, and no shape
    /// mismatches.
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty() && self.unused.is_empty() && self.shape_mismatches.is_empty()
    }

    /// Convert to a flodl `Result` — returns `Ok(())` when the validation is
    /// clean, otherwise a [`TensorError`] whose message lists every
    /// disagreement (truncated to the first 20 entries per bucket).
    pub fn into_result(self) -> Result<()> {
        self.into_result_impl(false)
    }

    /// Like [`into_result`](Self::into_result), but unused checkpoint keys
    /// don't count as an error — only missing keys and shape mismatches
    /// do. Used when loading a base model out of a checkpoint that also
    /// contains task-specific heads (e.g. pulling `BertModel` weights
    /// out of a `BertForPreTraining` checkpoint — the MLM / NSP head
    /// tensors are "unused" from `BertModel`'s point of view but their
    /// presence is expected, not an error).
    pub fn into_result_allow_unused(self) -> Result<()> {
        self.into_result_impl(true)
    }

    fn into_result_impl(mut self, allow_unused: bool) -> Result<()> {
        if allow_unused {
            self.unused.clear();
        }
        if self.is_ok() {
            return Ok(());
        }
        let mut msg = String::from("safetensors checkpoint does not match model:\n");
        if !self.missing.is_empty() {
            msg.push_str(&format!(
                "  {} missing key(s) (model expects, checkpoint lacks):\n",
                self.missing.len(),
            ));
            for k in self.missing.iter().take(20) {
                msg.push_str(&format!("    - {k}\n"));
            }
            if self.missing.len() > 20 {
                msg.push_str(&format!("    ... and {} more\n", self.missing.len() - 20));
            }
        }
        if !self.unused.is_empty() {
            msg.push_str(&format!(
                "  {} unused key(s) (checkpoint has, model lacks):\n",
                self.unused.len(),
            ));
            for k in self.unused.iter().take(20) {
                msg.push_str(&format!("    - {k}\n"));
            }
            if self.unused.len() > 20 {
                msg.push_str(&format!("    ... and {} more\n", self.unused.len() - 20));
            }
        }
        if !self.shape_mismatches.is_empty() {
            msg.push_str(&format!(
                "  {} shape mismatch(es):\n",
                self.shape_mismatches.len(),
            ));
            for m in self.shape_mismatches.iter().take(20) {
                msg.push_str(&format!(
                    "    - {}: expected {:?}, found {:?}\n",
                    m.key, m.expected, m.found,
                ));
            }
            if self.shape_mismatches.len() > 20 {
                msg.push_str(&format!(
                    "    ... and {} more\n",
                    self.shape_mismatches.len() - 20,
                ));
            }
        }
        Err(TensorError::new(&msg))
    }
}

/// Validate model expectations against the `(key → shape)` map extracted
/// from a safetensors file.
///
/// Output is sorted (missing / unused / mismatches all ascending by key) so
/// error messages are stable across runs, which matters for diffing error
/// logs and writing tests.
pub fn validate_keys(
    expected: &[ExpectedParam],
    actual: &HashMap<String, Vec<i64>>,
) -> LoadValidation {
    let expected_keys: HashSet<&str> = expected.iter().map(|p| p.key.as_str()).collect();
    let mut v = LoadValidation::default();

    for p in expected {
        match actual.get(&p.key) {
            None => v.missing.push(p.key.clone()),
            Some(found) if found != &p.shape => {
                v.shape_mismatches.push(ShapeMismatch {
                    key: p.key.clone(),
                    expected: p.shape.clone(),
                    found: found.clone(),
                });
            }
            Some(_) => {}
        }
    }
    for k in actual.keys() {
        if !expected_keys.contains(k.as_str()) {
            v.unused.push(k.clone());
        }
    }
    v.missing.sort();
    v.unused.sort();
    v.shape_mismatches.sort_by(|a, b| a.key.cmp(&b.key));
    v
}

/// Collect expected parameters + buffers from a `Graph`, with keys already
/// converted to HF-dotted form via [`hf_key_from_flodl_key`].
///
/// Use this to drive [`validate_keys`] in the common case where the model
/// is a flodl `Graph` built via `FlowBuilder`.
pub fn expected_from_graph(graph: &Graph) -> Vec<ExpectedParam> {
    let mut out = Vec::new();
    for (k, p) in graph.named_parameters() {
        out.push(ExpectedParam {
            key: hf_key_from_flodl_key(&k),
            shape: p.variable.shape(),
        });
    }
    for (k, b) in graph.named_buffers() {
        out.push(ExpectedParam {
            key: hf_key_from_flodl_key(&k),
            shape: b.shape(),
        });
    }
    out
}

// ── Tensor data loading ──────────────────────────────────────────────────

/// Load a safetensors byte buffer's weights into the graph's parameters
/// and buffers.
///
/// Validates key set and shapes first — on any disagreement, returns a
/// [`LoadValidation::into_result`]-style error listing every mismatch so
/// the caller can fix tags / use the right checkpoint variant. On success,
/// copies each tensor's data (casting to f32 if the checkpoint is f16 /
/// bf16 / f64) into the live graph storage in-place. Parameters keep
/// their autograd identity; only the underlying buffer bytes change.
///
/// Integer-dtype checkpoint tensors are not supported — HF's
/// transformer zoo doesn't ship integer-weight tensors, so the common
/// path only needs floats.
pub fn load_safetensors_into_graph(graph: &Graph, bytes: &[u8]) -> Result<()> {
    load_safetensors_into_graph_with_rename(graph, bytes, |k| k.to_string())
}

/// Same as [`load_safetensors_into_graph`] but applies `rename` to every
/// checkpoint key before matching against the graph.
///
/// `rename(checkpoint_key) -> canonical_key` lets callers paper over
/// legacy HF naming (e.g. `LayerNorm.gamma` → `LayerNorm.weight`). Use
/// [`bert_legacy_key_rename`] for the standard BERT mapping.
pub fn load_safetensors_into_graph_with_rename<F>(
    graph: &Graph,
    bytes: &[u8],
    rename: F,
) -> Result<()>
where
    F: Fn(&str) -> String,
{
    load_safetensors_core(graph, bytes, &rename, false)?;
    Ok(())
}

/// Like [`load_safetensors_into_graph_with_rename`] but tolerates
/// checkpoint keys that the graph does not ask for — useful when
/// loading a base model out of a checkpoint that also ships task heads
/// (e.g. `BertForPreTraining` on the Hub carries MLM + NSP heads that
/// a bare `BertModel` has no slot for). Missing keys and shape
/// mismatches are still hard errors.
///
/// Returns the list of checkpoint keys that were present but not used,
/// sorted alphabetically, so callers can surface them to the user.
pub fn load_safetensors_into_graph_with_rename_allow_unused<F>(
    graph: &Graph,
    bytes: &[u8],
    rename: F,
) -> Result<Vec<String>>
where
    F: Fn(&str) -> String,
{
    load_safetensors_core(graph, bytes, &rename, true)
}

fn load_safetensors_core(
    graph: &Graph,
    bytes: &[u8],
    rename: &dyn Fn(&str) -> String,
    allow_unused: bool,
) -> Result<Vec<String>> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| TensorError::new(&format!("safetensors parse error: {e}")))?;

    // Index: canonical (renamed) key → original checkpoint key. The rename
    // must be injective across the checkpoint's key set, otherwise two
    // checkpoint tensors would collapse onto the same canonical slot —
    // surface that as a validation-shaped error so the caller knows.
    let mut canonical_to_original: HashMap<String, String> = HashMap::new();
    let mut actual_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    for name in st.names() {
        let canonical = rename(name);
        if let Some(prev) = canonical_to_original.insert(canonical.clone(), name.to_string()) {
            return Err(TensorError::new(&format!(
                "safetensors key rename collision: both {prev:?} and {name:?} \
                 map to canonical key {canonical:?}",
            )));
        }
        let view = st.tensor(name)
            .map_err(|e| TensorError::new(&format!("safetensors tensor lookup {name}: {e}")))?;
        actual_shapes.insert(
            canonical,
            view.shape().iter().map(|&s| s as i64).collect(),
        );
    }

    // Validate before touching any graph storage. If this fails, the graph
    // is left untouched so the caller can recover / report.
    let expected = expected_from_graph(graph);
    let validation = validate_keys(&expected, &actual_shapes);
    let unused = validation.unused.clone();
    if allow_unused {
        validation.into_result_allow_unused()?;
    } else {
        validation.into_result()?;
    }

    // Replace each parameter's tensor with the loaded one. `set_data`
    // preserves the Variable's `requires_grad` flag and its place in the
    // graph while swapping in the new backing storage. An in-place
    // `copy_` would trip libtorch's "leaf Variable used in in-place op"
    // check on parameters.
    for (flodl_key, param) in graph.named_parameters() {
        let hf_key = hf_key_from_flodl_key(&flodl_key);
        let original = canonical_to_original.get(&hf_key).ok_or_else(|| {
            TensorError::new(&format!(
                "canonical key {hf_key:?} missing from checkpoint after rename \
                 (validation should have caught this)",
            ))
        })?;
        let view = st.tensor(original)
            .map_err(|e| TensorError::new(&format!("safetensors tensor {original}: {e}")))?;
        let device = param.variable.data().device();
        let src = tensor_view_to_tensor(&view, device)?;
        param.variable.set_data(src);
    }

    // Same pattern for buffers (BERT has none today, but any future
    // model with BatchNorm-style running stats will).
    for (flodl_key, buffer) in graph.named_buffers() {
        let hf_key = hf_key_from_flodl_key(&flodl_key);
        let original = canonical_to_original.get(&hf_key).ok_or_else(|| {
            TensorError::new(&format!(
                "canonical buffer key {hf_key:?} missing after rename",
            ))
        })?;
        let view = st.tensor(original)
            .map_err(|e| TensorError::new(&format!("safetensors tensor {original}: {e}")))?;
        let src = tensor_view_to_tensor(&view, buffer.device())?;
        buffer.set(src);
    }

    Ok(unused)
}

/// Rewrite legacy HF BERT-family checkpoint keys to the form flodl's
/// MLM-head graphs expect.
///
/// 1. **LayerNorm gamma/beta** (TensorFlow-era): `LayerNorm.gamma` →
///    `LayerNorm.weight`, `LayerNorm.beta` → `LayerNorm.bias`.
///    `bert-base-*` and other pre-2020 checkpoints still ship with
///    `gamma`/`beta`. HF Python's `BertModel.from_pretrained` applies
///    the same remap at load time.
///
/// 2. **MLM decoder bias tying** (BERT / RoBERTa MLM):
///    `cls.predictions.bias` → `cls.predictions.decoder.bias`,
///    `lm_head.bias` → `lm_head.decoder.bias`. HF's `BertForMaskedLM`
///    and `RobertaForMaskedLM` both tie their decoder's bias to a
///    top-level `bias` Parameter via `self.decoder.bias = self.bias`.
///    PyTorch's `state_dict` dedupes tied Parameters on save, so
///    checkpoints ship only the top-level key. Our graphs store the
///    bias directly on the decoder `Linear` (one of the entry points
///    of weight tying via [`flodl::Linear::from_shared_weight`]), so
///    we rename the checkpoint's key onto the decoder at load time.
pub fn bert_legacy_key_rename(checkpoint_key: &str) -> String {
    // MLM decoder-bias tying: exact-match renames. Exact rather than
    // suffix so we don't accidentally eat `*.cls.predictions.bias`
    // sub-keys in some future nested head.
    if checkpoint_key == "cls.predictions.bias" {
        return "cls.predictions.decoder.bias".to_string();
    }
    if checkpoint_key == "lm_head.bias" {
        return "lm_head.decoder.bias".to_string();
    }
    if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.gamma") {
        format!("{prefix}LayerNorm.weight")
    } else if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.beta") {
        format!("{prefix}LayerNorm.bias")
    } else {
        checkpoint_key.to_string()
    }
}

/// HF-canonical LayerNorm rename: rewrite legacy `LayerNorm.gamma` /
/// `LayerNorm.beta` suffixes to the modern `LayerNorm.weight` /
/// `LayerNorm.bias` form. Pure suffix rename, leaves every other key
/// untouched. Used by the round-trip comparators in
/// `tests/roundtrip_common/mod.rs` to canonicalise HF-reference
/// safetensors against flodl exports — flodl always saves the modern
/// names; older HF checkpoints (e.g. `bert-base-uncased`) ship the
/// legacy names. Distinct from [`bert_legacy_key_rename`], which is
/// the *load-side* HF-to-flodl rename and additionally maps the MLM
/// decoder-bias tying alias (no longer needed by the comparator since
/// [`hf_canonical_save_key`] makes flodl saves match HF canonical
/// keys for that case).
pub fn bert_legacy_layernorm_rename(checkpoint_key: &str) -> String {
    if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.gamma") {
        format!("{prefix}LayerNorm.weight")
    } else if let Some(prefix) = checkpoint_key.strip_suffix("LayerNorm.beta") {
        format!("{prefix}LayerNorm.bias")
    } else {
        checkpoint_key.to_string()
    }
}

/// Inverse of [`bert_legacy_key_rename`] for the MLM decoder-bias tying
/// case: rewrite flodl's internal `cls.predictions.decoder.bias` /
/// `lm_head.decoder.bias` parameter name back to the canonical HF key
/// (`cls.predictions.bias` / `lm_head.bias`) at save time.
///
/// Why save-side renames matter: HF Python's `BertForMaskedLM` /
/// `RobertaForMaskedLM` declare `self.bias` as the storage parameter
/// and then alias it via `self.decoder.bias = self.bias`. When HF's
/// `from_pretrained` loads a state_dict that has only
/// `cls.predictions.decoder.bias` (flodl's internal name), the
/// owning `cls.predictions.bias` parameter ends up on the meta device
/// and `tie_weights()` doesn't always materialise it on torch 2.x —
/// forward then fails with "Tensor on device meta is not on the
/// expected device cpu" inside the decoder's `addmm`. Emitting the
/// canonical HF key makes the load route through the correct owning
/// parameter and keeps the alias materialised.
///
/// Applied by [`save_safetensors_from_graph`] after [`crate::path::hf_key_from_flodl_key`]
/// converts the flodl tag separator to dotted form. The
/// `LayerNorm.gamma`/`beta` legacy names are NOT inverted on save —
/// flodl emits the modern `weight`/`bias` form, which both HF Python
/// and the Rust `_live` head-roundtrip comparator already canonicalise
/// to.
pub fn hf_canonical_save_key(hf_key: &str) -> String {
    if hf_key == "cls.predictions.decoder.bias" {
        return "cls.predictions.bias".to_string();
    }
    if hf_key == "lm_head.decoder.bias" {
        return "lm_head.bias".to_string();
    }
    hf_key.to_string()
}

/// Predicate: does `key` end with one of the pooler suffixes?
///
/// Matches both BERT-style `pooler.dense.{weight,bias}` (a wrapper
/// around a `BertPooler { dense: Linear }`) and ALBERT-style flat
/// `pooler.{weight,bias}` (HF's `AlbertModel.pooler` is a bare
/// `nn.Linear`). Pooler-less families (DistilBERT, DeBERTa-v2) never
/// match either shape, so the predicate is a safe no-op for them.
///
/// Normalises the `/` tag separator that flodl checkpoints use between
/// qualified tag boundaries (e.g. `bert.pooler/dense.weight`) so the
/// same predicate works for both raw safetensors keys and flodl's
/// internal tag form.
fn is_pooler_key(key: &str) -> bool {
    let normalised = key.replace('/', ".");
    normalised.ends_with("pooler.dense.weight")
        || normalised.ends_with("pooler.dense.bias")
        || normalised.ends_with("pooler.weight")
        || normalised.ends_with("pooler.bias")
}

/// Inspect a safetensors blob and report whether it carries the pooler
/// `Linear` weights for any of the pooler-bearing families (BERT,
/// RoBERTa, XLM-R, ALBERT).
///
/// Used by every pooler-bearing family's `from_pretrained_on_device`
/// (and `AutoModel::from_pretrained_for_export_on_device`) to pick
/// `on_device` vs `on_device_without_pooler` based on what the
/// checkpoint actually ships, rather than baking a per-family default
/// that's always wrong for some Hub repos (e.g. `roberta-base` has no
/// pooler; `bert-base-uncased` does).
pub fn weights_have_pooler(weights: &[u8]) -> Result<bool> {
    let st = SafeTensors::deserialize(weights)
        .map_err(|e| TensorError::new(&format!("safetensors parse error: {e}")))?;
    Ok(st.names().iter().any(|n| is_pooler_key(n)))
}

/// Detect pooler presence from a list of checkpoint keys (e.g. the
/// output of [`flodl::checkpoint_keys`]). Mirrors [`weights_have_pooler`]
/// for the checkpoint-keys input shape; the flodl checkpoint key form
/// uses `/` as the tag-boundary separator and this helper normalises
/// that internally so both safetensors-style dotted keys and tagged
/// flodl keys are handled by the same predicate.
pub fn keys_have_pooler(keys: &[String]) -> bool {
    keys.iter().any(|k| is_pooler_key(k))
}

/// Read a safetensors file from disk and load it into `graph`.
///
/// Thin wrapper around [`load_safetensors_into_graph`]. I/O errors are
/// surfaced as `TensorError` with the path in the message for easier
/// debugging.
pub fn load_safetensors_file_into_graph(graph: &Graph, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::new(&format!("safetensors read {}: {e}", path.display()))
    })?;
    load_safetensors_into_graph(graph, &bytes)
}

/// Read a safetensors file from disk and load it into `graph`, applying
/// `rename` to every checkpoint key first. See
/// [`load_safetensors_into_graph_with_rename`].
pub fn load_safetensors_file_into_graph_with_rename<F>(
    graph: &Graph,
    path: &Path,
    rename: F,
) -> Result<()>
where
    F: Fn(&str) -> String,
{
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::new(&format!("safetensors read {}: {e}", path.display()))
    })?;
    load_safetensors_into_graph_with_rename(graph, &bytes, rename)
}

// ── Saving ────────────────────────────────────────────────────────────────

/// Serialise a graph's parameters and buffers as safetensors bytes.
///
/// Iterates [`Graph::named_parameters`] and [`Graph::named_buffers`],
/// converts each flodl key (slash form, e.g. `bert.pooler.dense/weight`)
/// to its HF-dotted equivalent via [`hf_key_from_flodl_key`], and writes
/// every tensor as f32 — the storage dtype flodl uses internally.
///
/// Tied parameters (the same `Variable` reachable through multiple tags)
/// are deduped upstream by `named_parameters`, so each weight ships
/// once. If two distinct tensors collide on the same HF key after
/// renaming, the function returns a loud error rather than silently
/// dropping one — that condition signals a tag-naming conflict in the
/// model, not a save-layer bug.
///
/// Output ordering is deterministic: keys serialise in HF-dotted
/// alphabetical order so the resulting file diffs cleanly across runs.
///
/// The bytes produced are byte-for-byte loadable by HF Python's
/// `safe_open(...).load_state_dict(...)` for any model whose `state_dict`
/// matches the saved key set.
pub fn save_safetensors_from_graph(graph: &Graph) -> Result<Vec<u8>> {
    use std::collections::BTreeMap;

    // BTreeMap orders keys for deterministic byte output; (dtype, shape, bytes)
    // payload owns the data so the TensorView slices below stay valid.
    let mut entries: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)> = BTreeMap::new();

    for (flodl_key, param) in graph.named_parameters() {
        let hf_key = hf_canonical_save_key(&hf_key_from_flodl_key(&flodl_key));
        let shape: Vec<usize> = param.variable.shape().iter().map(|&d| d as usize).collect();
        let dtype = param.variable.data().dtype();
        let bytes = param.variable.data().to_blob()?;
        if entries.contains_key(&hf_key) {
            return Err(TensorError::new(&format!(
                "save_safetensors: HF key {hf_key:?} collision \
                 — multiple distinct flodl tensors map to the same name; \
                 fix the conflicting `tag(...)` in the graph",
            )));
        }
        entries.insert(hf_key, (dtype_to_safetensors(dtype)?, shape, bytes));
    }

    for (flodl_key, buffer) in graph.named_buffers() {
        let hf_key = hf_canonical_save_key(&hf_key_from_flodl_key(&flodl_key));
        let shape: Vec<usize> = buffer.shape().iter().map(|&d| d as usize).collect();
        let dtype = buffer.get().dtype();
        let bytes = buffer.get().to_blob()?;
        if entries.contains_key(&hf_key) {
            return Err(TensorError::new(&format!(
                "save_safetensors: HF key {hf_key:?} collision \
                 — buffer collides with a parameter or another buffer",
            )));
        }
        entries.insert(hf_key, (dtype_to_safetensors(dtype)?, shape, bytes));
    }

    let views: HashMap<String, TensorView<'_>> = entries.iter()
        .map(|(k, (dtype, shape, bytes))| {
            let view = TensorView::new(*dtype, shape.clone(), bytes.as_slice())
                .map_err(|e| TensorError::new(&format!(
                    "safetensors view build for {k:?}: {e}",
                )))?;
            Ok::<(String, TensorView<'_>), TensorError>((k.clone(), view))
        })
        .collect::<std::result::Result<_, _>>()?;

    safetensors::serialize(&views, &None)
        .map_err(|e| TensorError::new(&format!("safetensors serialize: {e}")))
}

/// Map a flodl `DType` to the safetensors `Dtype` that uses the same
/// in-memory bit layout. Integer dtypes are rejected — flodl's exporter
/// only emits learnable parameter / buffer payloads, which are floats.
fn dtype_to_safetensors(dtype: DType) -> Result<Dtype> {
    match dtype {
        DType::Float32 => Ok(Dtype::F32),
        DType::Float64 => Ok(Dtype::F64),
        DType::Float16 => Ok(Dtype::F16),
        DType::BFloat16 => Ok(Dtype::BF16),
        DType::Int32 | DType::Int64 => Err(TensorError::new(&format!(
            "save_safetensors: integer dtype {dtype:?} not supported \
             — only floating-point parameters / buffers can be serialised",
        ))),
    }
}

/// Serialise a graph and write the bytes to `path`. Thin file wrapper
/// over [`save_safetensors_from_graph`]; I/O errors carry the path in
/// the message for easier debugging.
pub fn save_safetensors_file_from_graph(graph: &Graph, path: &Path) -> Result<()> {
    let bytes = save_safetensors_from_graph(graph)?;
    std::fs::write(path, &bytes).map_err(|e| {
        TensorError::new(&format!("safetensors write {}: {e}", path.display()))
    })
}

/// Materialise a safetensors `TensorView` as a `Tensor` on
/// `target_device`, **preserving the source dtype**. Raw bytes are
/// shuttled through libtorch's `from_blob` (which copies internally),
/// so the resulting tensor has the same dtype as the safetensors file
/// — F16 stays F16, BF16 stays BF16, F32 stays F32, F64 stays F64.
fn tensor_view_to_tensor(view: &TensorView, target_device: Device) -> Result<Tensor> {
    let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
    let dtype = match view.dtype() {
        Dtype::F32 => DType::Float32,
        Dtype::F64 => DType::Float64,
        Dtype::F16 => DType::Float16,
        Dtype::BF16 => DType::BFloat16,
        other => {
            return Err(TensorError::new(&format!(
                "unsupported safetensors dtype {other:?} — floats (F32/F64/BF16/F16) only",
            )));
        }
    };
    Tensor::from_blob(view.data(), &shape, dtype, target_device)
}

/// Decode a safetensors `TensorView`'s raw bytes as a flat `Vec<f32>`.
/// Supports f32 (zero conversion), f64 / bf16 / f16 (host-side cast).
/// Rejects integer / bool dtypes — BERT-style checkpoints don't use them
/// and silently accepting would mean casting integers to floats, which
/// is almost never the user's intent.
///
/// Public so external callers (e.g. roundtrip tests, custom load
/// pipelines) can decode safetensors values to f32 with the exact same
/// dtype rules flodl uses on its load path. Equivalent on the f16 path
/// to `f32::from(half::f16::from_bits(_))` without pulling in `half`.
pub fn tensor_view_to_f32_vec(view: &TensorView) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(TensorError::new(&format!(
                    "F32 tensor byte length {} is not a multiple of 4", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            if bytes.len() % 8 != 0 {
                return Err(TensorError::new(&format!(
                    "F64 tensor byte length {} is not a multiple of 8", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let bits = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                // Matches PyTorch's `.to(torch.float32)`: IEEE 754 narrowing,
                // overflow saturates silently to ±inf, precision rounds to
                // nearest-even. Transformer weights never hit the tails, so
                // the silent saturation is acceptable and PyTorch-compatible.
                out.push(bits as f32);
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(TensorError::new(&format!(
                    "BF16 tensor byte length {} is not a multiple of 2", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                // bf16 is the top 16 bits of a f32 (same exponent).
                out.push(f32::from_bits((bits as u32) << 16));
            }
            Ok(out)
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(TensorError::new(&format!(
                    "F16 tensor byte length {} is not a multiple of 2", bytes.len(),
                )));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_bits_to_f32(bits));
            }
            Ok(out)
        }
        other => Err(TensorError::new(&format!(
            "unsupported safetensors dtype {other:?} — floats (F32/F64/BF16/F16) only",
        ))),
    }
}

/// IEEE 754 half-precision (binary16) to single-precision conversion.
///
/// Handles zero, subnormals, normals, infinity, and NaN. No external
/// dependency. Equivalent to `f32::from(half::f16::from_bits(bits))`
/// but keeps flodl-hf dep-light.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32 & 0x1;
    let exp = (bits >> 10) as u32 & 0x1f;
    let mantissa = bits as u32 & 0x3ff;

    let out_bits: u32 = if exp == 0 {
        if mantissa == 0 {
            sign << 31
        } else {
            // Subnormal half → normal f32.
            let mut m = mantissa;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (e + 127) as u32 & 0xff;
            (sign << 31) | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Inf (mantissa == 0) or NaN (mantissa != 0) — preserve bits
        // shifted into the wider mantissa.
        (sign << 31) | (0xff << 23) | (mantissa << 13)
    } else {
        // Normal half.
        let f32_exp = (exp + 127 - 15) & 0xff;
        (sign << 31) | (f32_exp << 23) | (mantissa << 13)
    };
    f32::from_bits(out_bits)
}

#[cfg(test)]
#[path = "safetensors_io_tests.rs"]
mod tests;
