//! Shared `config.json` parsing helpers used by every `*Config::from_json_str`.
//!
//! HuggingFace model configs agree on a small vocabulary for the fields
//! that affect model shape and task-head metadata (`vocab_size`,
//! `num_labels`, `id2label`, …). Each model's `from_json_str` was
//! re-implementing the same per-field readers and the same
//! `id2label → Vec<String>` normalization; this module centralizes
//! both.

use serde_json::Value;

use flodl::nn::GeluApprox;
use flodl::{Result, TensorError};

/// Read a required integer field. Errors if missing, null, or not an
/// integer.
pub(crate) fn required_i64(v: &Value, key: &str) -> Result<i64> {
    v.get(key).and_then(|x| x.as_i64()).ok_or_else(|| {
        TensorError::new(&format!(
            "config.json missing required integer field: {key}",
        ))
    })
}

/// Read a required string field. Errors if missing, null, or not a
/// string. Used by [`crate::models::auto::AutoConfig`] to dispatch on
/// `model_type`.
pub(crate) fn required_string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key).and_then(|x| x.as_str()).ok_or_else(|| {
        TensorError::new(&format!("config.json missing required string field: {key}",))
    })
}

/// Read an integer field with a fallback default. Accepts both
/// explicit integers and absent / null values.
pub(crate) fn optional_i64(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(default)
}

/// Read an optional integer field. `None` when the key is absent or
/// explicitly null; used for `pad_token_id` where BERT-family configs
/// may legitimately omit it (e.g. LLaMA-derived `pad == eos` setups
/// prefer `None`).
pub(crate) fn optional_i64_or_none(v: &Value, key: &str) -> Option<i64> {
    v.get(key).and_then(|x| x.as_i64())
}

/// Read a float field with a fallback default.
pub(crate) fn optional_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

/// Read a boolean field with a fallback default.
pub(crate) fn optional_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(|x| x.as_bool()).unwrap_or(default)
}

/// Parse HuggingFace's `id2label` field into an ordered `Vec<String>`.
///
/// HF stores the mapping as `{"0": "LABEL_0", "1": "LABEL_1", ...}`
/// (string keys, string values). This helper sorts by integer key so
/// the returned vector is indexable by class id — `vec[k]` matches HF
/// Python's `config.id2label[k]`.
///
/// Errors on:
/// - a `k` that can't be parsed as an integer,
/// - a value that isn't a string,
/// - non-contiguous ids (gap or duplicate), which would silently
///   misalign names with logits rows.
///
/// Returns `Ok(None)` when the key is absent or not an object.
pub(crate) fn parse_id2label(v: &Value) -> Result<Option<Vec<String>>> {
    let obj = match v.get("id2label").and_then(|x| x.as_object()) {
        Some(obj) => obj,
        None => return Ok(None),
    };
    let mut pairs: Vec<(i64, String)> = Vec::with_capacity(obj.len());
    for (k, val) in obj {
        let id: i64 = k.parse().map_err(|_| {
            TensorError::new(&format!(
                "config.json: id2label key {k:?} is not an integer",
            ))
        })?;
        let label = val.as_str().ok_or_else(|| {
            TensorError::new(&format!("config.json: id2label[{k}] is not a string",))
        })?;
        pairs.push((id, label.to_string()));
    }
    pairs.sort_by_key(|(id, _)| *id);
    for (idx, (id, _)) in pairs.iter().enumerate() {
        if *id != idx as i64 {
            return Err(TensorError::new(&format!(
                "config.json: id2label must have contiguous ids 0..N, \
                 but index {idx} has id {id}",
            )));
        }
    }
    Ok(Some(pairs.into_iter().map(|(_, s)| s).collect()))
}

/// Map an HF `hidden_act` string to a [`GeluApprox`].
///
/// Picking the wrong form silently produces a small per-token diff
/// (~1e-2 max-abs after 12 layers) that compounds across the encoder —
/// large enough to fail any meaningful parity test. So unrecognised
/// strings error loudly with a message that names the supported set.
///
/// Mappings (case-sensitive — matches HF's `ACT2FN` keys):
/// - `"gelu"` → [`GeluApprox::Exact`] (erf form)
/// - `"gelu_new"` → [`GeluApprox::Tanh`] (tanh approximation,
///   ALBERT v1+v2, GPT-2)
/// - `"gelu_pytorch_tanh"` → [`GeluApprox::Tanh`] (HF's newer alias
///   for the same approximation)
fn map_hidden_act(s: &str) -> Result<GeluApprox> {
    match s {
        "gelu" => Ok(GeluApprox::Exact),
        "gelu_new" => Ok(GeluApprox::Tanh),
        "gelu_pytorch_tanh" => Ok(GeluApprox::Tanh),
        other => Err(TensorError::new(&format!(
            "config.json: unsupported hidden_act = {other:?}. \
             flodl-hf currently maps {{\"gelu\", \"gelu_new\", \
             \"gelu_pytorch_tanh\"}} to flodl::nn::GeluApprox; \
             other activations (e.g. \"relu\", \"silu\") are not \
             yet wired through the transformer layer. File against \
             flodl-hf with the failing checkpoint id."
        ))),
    }
}

/// Read the activation field from `config.json`, mapped to a
/// [`GeluApprox`]. `default` is used when the key is absent or null —
/// pass `"gelu"` for the BERT-family default (erf form). The presence
/// of the field is honoured even if it equals the default, so a
/// derivative checkpoint that explicitly writes `"hidden_act": "gelu"`
/// behaves the same as one that omits it.
///
/// Different families ship the field under different names —
/// most use `"hidden_act"`, DistilBERT uses `"activation"` — so the
/// caller passes the key.
pub(crate) fn optional_hidden_act(v: &Value, key: &str, default: &str) -> Result<GeluApprox> {
    let raw = v.get(key).and_then(|x| x.as_str()).unwrap_or(default);
    map_hidden_act(raw)
}

/// Derive `num_labels` from the config.
///
/// Priority: explicit `num_labels` field wins; otherwise fall back to
/// the length of the already-parsed `id2label` list. Returns `None`
/// when neither is present (base model configs that aren't fine-tuned
/// as task heads).
pub(crate) fn parse_num_labels(v: &Value, id2label: Option<&[String]>) -> Option<i64> {
    v.get("num_labels")
        .and_then(|x| x.as_i64())
        .or_else(|| id2label.map(|v| v.len() as i64))
}

/// Parse HuggingFace's `architectures` field (e.g.
/// `["BertForSequenceClassification"]`) into an ordered `Vec<String>`.
///
/// HF stores this as a JSON array of class names; the first entry is
/// the canonical class for dispatch (`AutoModel.from_pretrained` reads
/// it). Returns `None` when the key is absent, not an array, or empty;
/// non-string array entries are skipped silently — their presence
/// would be a malformed config rather than a flodl-hf concern.
///
/// Used by [`crate::export::build_for_export`] to dispatch a checkpoint
/// to the correct task-head builder when re-exporting via
/// `fdl flodl-hf export --checkpoint`.
pub(crate) fn parse_architectures(v: &Value) -> Option<Vec<String>> {
    let arr = v.get("architectures").and_then(|x| x.as_array())?;
    let names: Vec<String> = arr
        .iter()
        .filter_map(|x| x.as_str().map(|s| s.to_string()))
        .collect();
    if names.is_empty() { None } else { Some(names) }
}

// ─── Write helpers (inverse of the readers above) ────────────────────────

/// HF string to emit for a given [`GeluApprox`].
///
/// `None` → `"gelu"` (erf form), `Tanh` → `"gelu_new"`. Both HF strings
/// `"gelu_new"` and `"gelu_pytorch_tanh"` parse back to
/// [`GeluApprox::Tanh`], so a config written with `"gelu_pytorch_tanh"`
/// and round-tripped through flodl-hf will normalize to `"gelu_new"`.
/// Picking `"gelu_new"` keeps us in lockstep with every public Hub
/// checkpoint (ALBERT/GPT-2 variants and the BERT-family forks that use
/// the tanh form), and every HF transformers release routes it through
/// the same ACT2FN entry.
pub(crate) fn emit_hidden_act(act: GeluApprox) -> &'static str {
    match act {
        GeluApprox::Exact => "gelu",
        GeluApprox::Tanh => "gelu_new",
    }
}

/// Emit the `architectures` field as a JSON array.
///
/// When the parsed config carried an `architectures` list, write it
/// verbatim — preserves the canonical class name (e.g.
/// `["BertForSequenceClassification"]`) so HF Python re-dispatches to
/// the same task head, and so flodl-hf's own
/// [`crate::export::build_for_export`] can re-build the same head on
/// `--checkpoint` re-export. Otherwise emit `[default_class]`, which
/// is the family base name (`BertModel`, `RobertaModel`, …).
pub(crate) fn emit_architectures(stored: Option<&[String]>, default_class: &str) -> Value {
    let arr: Vec<Value> = match stored {
        Some(names) if !names.is_empty() => names.iter().map(|s| Value::from(s.as_str())).collect(),
        _ => vec![Value::from(default_class)],
    };
    Value::Array(arr)
}

/// Write the `id2label` + `label2id` pair into the given JSON object,
/// mirroring HF's convention (both maps present, string↔integer).
///
/// Inverse of [`parse_id2label`]. Keys are stringified integer ids
/// (`"0"`, `"1"`, …); `label2id` is `{name: id}`. Skipped entirely if
/// `labels` is `None`.
pub(crate) fn emit_id2label(out: &mut serde_json::Map<String, Value>, labels: Option<&[String]>) {
    let Some(labels) = labels else { return };
    let mut id2 = serde_json::Map::with_capacity(labels.len());
    let mut lab2 = serde_json::Map::with_capacity(labels.len());
    for (idx, name) in labels.iter().enumerate() {
        id2.insert(idx.to_string(), Value::from(name.as_str()));
        lab2.insert(name.clone(), Value::from(idx as i64));
    }
    out.insert("id2label".into(), Value::Object(id2));
    out.insert("label2id".into(), Value::Object(lab2));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_i64_reads_or_errors() {
        let v: Value = serde_json::from_str(r#"{"a": 42}"#).unwrap();
        assert_eq!(required_i64(&v, "a").unwrap(), 42);
        assert!(required_i64(&v, "missing").is_err());
    }

    #[test]
    fn required_string_reads_or_errors() {
        let v: Value = serde_json::from_str(r#"{"model_type": "bert", "num": 7}"#).unwrap();
        assert_eq!(required_string(&v, "model_type").unwrap(), "bert");
        assert!(required_string(&v, "missing").is_err());
        assert!(
            required_string(&v, "num").is_err(),
            "non-string values must error"
        );
    }

    #[test]
    fn optional_i64_falls_back() {
        let v: Value = serde_json::from_str(r#"{"a": 42}"#).unwrap();
        assert_eq!(optional_i64(&v, "a", 7), 42);
        assert_eq!(optional_i64(&v, "missing", 7), 7);
    }

    #[test]
    fn optional_i64_or_none_treats_absence_as_none() {
        let v: Value = serde_json::from_str(r#"{"a": 0, "b": null}"#).unwrap();
        assert_eq!(optional_i64_or_none(&v, "a"), Some(0));
        assert_eq!(optional_i64_or_none(&v, "b"), None);
        assert_eq!(optional_i64_or_none(&v, "missing"), None);
    }

    #[test]
    fn optional_f64_and_bool_defaults() {
        let v: Value = serde_json::from_str(r#"{"x": 0.5, "b": true}"#).unwrap();
        assert!((optional_f64(&v, "x", 1.0) - 0.5).abs() < 1e-12);
        assert!((optional_f64(&v, "missing", 1.0) - 1.0).abs() < 1e-12);
        assert!(optional_bool(&v, "b", false));
        assert!(!optional_bool(&v, "missing", false));
    }

    #[test]
    fn parse_id2label_orders_and_rejects_gaps() {
        let v: Value =
            serde_json::from_str(r#"{"id2label": {"2": "c", "0": "a", "1": "b"}}"#).unwrap();
        let out = parse_id2label(&v).unwrap().unwrap();
        assert_eq!(out, vec!["a", "b", "c"]);

        let gap: Value = serde_json::from_str(r#"{"id2label": {"0": "a", "2": "c"}}"#).unwrap();
        let err = parse_id2label(&gap).unwrap_err();
        assert!(format!("{err}").contains("contiguous"), "got: {err}");
    }

    #[test]
    fn parse_id2label_absent_is_none() {
        let v: Value = serde_json::from_str(r#"{}"#).unwrap();
        assert!(parse_id2label(&v).unwrap().is_none());
    }

    #[test]
    fn parse_num_labels_explicit_wins() {
        let v: Value = serde_json::from_str(r#"{"num_labels": 5}"#).unwrap();
        let labels = vec!["a".to_string(), "b".to_string()];
        assert_eq!(parse_num_labels(&v, Some(&labels)), Some(5));
    }

    #[test]
    fn parse_num_labels_falls_back_to_id2label_len() {
        let v: Value = serde_json::from_str(r#"{}"#).unwrap();
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(parse_num_labels(&v, Some(&labels)), Some(3));
    }

    #[test]
    fn parse_num_labels_none_when_both_missing() {
        let v: Value = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(parse_num_labels(&v, None), None);
    }

    #[test]
    fn emit_hidden_act_maps_both_variants() {
        assert_eq!(emit_hidden_act(GeluApprox::Exact), "gelu");
        assert_eq!(emit_hidden_act(GeluApprox::Tanh), "gelu_new");
    }

    #[test]
    fn emit_hidden_act_round_trips_through_map() {
        // Every emitted string must parse back to the same enum value.
        for act in [GeluApprox::Exact, GeluApprox::Tanh] {
            let parsed = map_hidden_act(emit_hidden_act(act)).unwrap();
            assert_eq!(parsed, act);
        }
    }

    #[test]
    fn emit_id2label_writes_both_maps() {
        let mut m = serde_json::Map::new();
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        emit_id2label(&mut m, Some(&labels));

        let id2 = m.get("id2label").unwrap().as_object().unwrap();
        assert_eq!(id2.get("0").unwrap().as_str().unwrap(), "A");
        assert_eq!(id2.get("1").unwrap().as_str().unwrap(), "B");
        assert_eq!(id2.get("2").unwrap().as_str().unwrap(), "C");

        let lab2 = m.get("label2id").unwrap().as_object().unwrap();
        assert_eq!(lab2.get("A").unwrap().as_i64().unwrap(), 0);
        assert_eq!(lab2.get("B").unwrap().as_i64().unwrap(), 1);
        assert_eq!(lab2.get("C").unwrap().as_i64().unwrap(), 2);
    }

    #[test]
    fn emit_id2label_skips_when_none() {
        let mut m = serde_json::Map::new();
        emit_id2label(&mut m, None);
        assert!(m.is_empty());
    }

    #[test]
    fn emit_id2label_round_trips_through_parse_id2label() {
        let mut m = serde_json::Map::new();
        let labels = vec!["LABEL_0".to_string(), "LABEL_1".to_string()];
        emit_id2label(&mut m, Some(&labels));
        let v = Value::Object(m);
        let parsed = parse_id2label(&v).unwrap().unwrap();
        assert_eq!(parsed, labels);
    }
}
