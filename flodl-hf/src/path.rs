//! Path-building helpers for HuggingFace-compatible module naming.
//!
//! HF safetensors checkpoints use dotted keys like
//! `bert.encoder.layer.0.attention.self.query.weight`. flodl's
//! [`Graph::named_parameters()`](flodl::Module) produces `"{tag}/{leaf}"`
//! with a single `/` separator, where `{tag}` is whatever was passed to
//! `FlowBuilder::tag(...)` and `{leaf}` is the parameter's own name
//! (typically `"weight"` or `"bias"`).
//!
//! Convention used throughout `flodl-hf`: tags passed to `FlowBuilder::tag`
//! literally encode the HF dotted path. At load time,
//! [`hf_key_from_flodl_key`] converts the single `/` separator back to `.`
//! to match safetensors keys exactly.
//!
//! [`HfPath`] is a small builder that assembles dotted paths segment by
//! segment, so authors write short identifiers and repetitive structures
//! (e.g. 12 transformer layers) without `format!` boilerplate.
//!
//! # Example
//!
//! ```
//! use flodl_hf::path::HfPath;
//!
//! let root = HfPath::new("bert");
//! let emb = root.sub("embeddings");
//! assert_eq!(emb.leaf("word_embeddings"), "bert.embeddings.word_embeddings");
//!
//! let layer = root.sub("encoder").sub("layer").sub(0);
//! let attn_self = layer.sub("attention").sub("self");
//! assert_eq!(attn_self.leaf("query"), "bert.encoder.layer.0.attention.self.query");
//! ```

use flodl::nn::Parameter;
use flodl::{Result, TensorError};

/// A dotted path rooted at some prefix, used as the tag string passed to
/// `FlowBuilder::tag(...)`.
///
/// See module docs for the overall convention. `HfPath` is immutable — `sub`
/// returns a new path rather than mutating self, which mirrors the consuming
/// `FlowBuilder` style in flodl.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfPath {
    path: String,
}

impl HfPath {
    /// Construct a new path rooted at `root`. The root must be a valid
    /// segment (non-empty, no dots, no slashes).
    ///
    /// Panics if validation fails — this is a programming error, not a
    /// runtime condition. Use [`HfPath::try_new`] for a fallible version.
    pub fn new(root: impl Into<String>) -> Self {
        let root = root.into();
        validate_segment(&root).expect("invalid HfPath root");
        HfPath { path: root }
    }

    /// Fallible constructor: returns an error instead of panicking on an
    /// invalid root segment.
    pub fn try_new(root: impl Into<String>) -> Result<Self> {
        let root = root.into();
        validate_segment(&root)?;
        Ok(HfPath { path: root })
    }

    /// Return a new path extended with `segment`. The segment becomes the
    /// new leaf part of the path. Accepts anything that can be converted
    /// via `to_string()` (e.g. `&str`, `String`, or integer layer indices).
    pub fn sub<S: ToString>(&self, segment: S) -> Self {
        let seg = segment.to_string();
        validate_segment(&seg).expect("invalid HfPath segment");
        HfPath {
            path: format!("{}.{}", self.path, seg),
        }
    }

    /// Fallible version of [`HfPath::sub`].
    pub fn try_sub<S: ToString>(&self, segment: S) -> Result<Self> {
        let seg = segment.to_string();
        validate_segment(&seg)?;
        Ok(HfPath {
            path: format!("{}.{}", self.path, seg),
        })
    }

    /// Return the full dotted tag string for a leaf module.
    ///
    /// The leaf name becomes the last segment of the returned string. Pass
    /// this directly to `FlowBuilder::tag(...)`.
    pub fn leaf(&self, name: &str) -> String {
        validate_segment(name).expect("invalid HfPath leaf");
        format!("{}.{}", self.path, name)
    }

    /// Fallible version of [`HfPath::leaf`].
    pub fn try_leaf(&self, name: &str) -> Result<String> {
        validate_segment(name)?;
        Ok(format!("{}.{}", self.path, name))
    }

    /// Return this path as a `&str` without the leaf suffix.
    pub fn as_str(&self) -> &str {
        &self.path
    }
}

impl std::fmt::Display for HfPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.path.fmt(f)
    }
}

/// Prepend `"{prefix}."` to every parameter's `name` field.
///
/// Composite modules whose internal parameter names must line up with
/// HuggingFace dotted keys use this to build the dotted-path prefix for
/// their sub-modules. For example, a `BertAttention` with a self-attention
/// child module returns parameters named `"query.weight"`, `"key.weight"`;
/// after `prefix_params("self", ...)` they become `"self.query.weight"`,
/// `"self.key.weight"`. Combined with the `FlowBuilder::tag(...)` string
/// (`"bert.encoder.layer.0.attention"`), the final safetensors key lands
/// at `"bert.encoder.layer.0.attention.self.query.weight"`.
///
/// ```
/// use flodl_hf::path::prefix_params;
/// use flodl::{Linear, Module};
///
/// let linear = Linear::new(4, 2).unwrap();
/// let prefixed = prefix_params("dense", linear.parameters());
/// assert_eq!(prefixed[0].name, "dense.weight");
/// assert_eq!(prefixed[1].name, "dense.bias");
/// ```
pub fn prefix_params(prefix: &str, params: Vec<Parameter>) -> Vec<Parameter> {
    params
        .into_iter()
        .map(|p| Parameter {
            variable: p.variable,
            name: format!("{prefix}.{}", p.name),
        })
        .collect()
}

/// Convert a flodl qualified parameter name to the HF safetensors key.
///
/// `Graph::named_parameters()` returns keys in the form `"{tag}/{leaf}"`
/// with a single `/` separator between the tag (HF-dotted path) and the
/// parameter's own name. This helper swaps the final `/` for `.` so the
/// full key is HF-compatible.
///
/// ```
/// use flodl_hf::path::hf_key_from_flodl_key;
/// assert_eq!(
///     hf_key_from_flodl_key("bert.encoder.layer.0.attention.self.query/weight"),
///     "bert.encoder.layer.0.attention.self.query.weight",
/// );
/// ```
pub fn hf_key_from_flodl_key(flodl_key: &str) -> String {
    match flodl_key.rsplit_once('/') {
        Some((prefix, leaf)) => format!("{prefix}.{leaf}"),
        None => flodl_key.to_string(),
    }
}

fn validate_segment(seg: &str) -> Result<()> {
    if seg.is_empty() {
        return Err(TensorError::new("HfPath segment must not be empty"));
    }
    if seg.contains('/') {
        return Err(TensorError::new(&format!(
            "HfPath segment {seg:?} must not contain '/'"
        )));
    }
    if seg.contains('.') {
        return Err(TensorError::new(&format!(
            "HfPath segment {seg:?} must not contain '.' (use .sub() to add segments)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_and_sub_compose_dotted() {
        let p = HfPath::new("bert").sub("embeddings");
        assert_eq!(p.as_str(), "bert.embeddings");
    }

    #[test]
    fn leaf_appends_final_segment() {
        let p = HfPath::new("bert").sub("encoder").sub("layer").sub(0);
        assert_eq!(p.leaf("attention"), "bert.encoder.layer.0.attention");
    }

    #[test]
    fn integer_segments_via_to_string() {
        let p = HfPath::new("bert").sub("layer");
        for i in 0..3 {
            let s = p.sub(i);
            assert_eq!(s.as_str(), format!("bert.layer.{i}"));
        }
    }

    #[test]
    fn sub_is_immutable_returns_new() {
        let root = HfPath::new("bert");
        let a = root.sub("a");
        let b = root.sub("b");
        assert_eq!(root.as_str(), "bert");
        assert_eq!(a.as_str(), "bert.a");
        assert_eq!(b.as_str(), "bert.b");
    }

    #[test]
    fn full_bert_self_attention_path() {
        let root = HfPath::new("bert");
        let attn_self = root
            .sub("encoder")
            .sub("layer")
            .sub(0)
            .sub("attention")
            .sub("self");
        assert_eq!(
            attn_self.leaf("query"),
            "bert.encoder.layer.0.attention.self.query",
        );
    }

    #[test]
    fn try_new_rejects_empty_dot_slash() {
        assert!(HfPath::try_new("").is_err());
        assert!(HfPath::try_new("a.b").is_err());
        assert!(HfPath::try_new("a/b").is_err());
    }

    #[test]
    fn try_sub_rejects_invalid_segments() {
        let root = HfPath::new("bert");
        assert!(root.try_sub("").is_err());
        assert!(root.try_sub("foo.bar").is_err());
        assert!(root.try_sub("foo/bar").is_err());
    }

    #[test]
    #[should_panic(expected = "invalid HfPath root")]
    fn new_panics_on_empty_root() {
        let _ = HfPath::new("");
    }

    #[test]
    fn try_leaf_rejects_dots_in_name() {
        let root = HfPath::new("bert");
        assert!(root.try_leaf("foo.bar").is_err());
    }

    #[test]
    fn hf_key_conversion_swaps_last_slash() {
        assert_eq!(
            hf_key_from_flodl_key("bert.embeddings.word_embeddings/weight"),
            "bert.embeddings.word_embeddings.weight",
        );
        assert_eq!(
            hf_key_from_flodl_key("bert.pooler.dense/bias"),
            "bert.pooler.dense.bias",
        );
    }

    #[test]
    fn hf_key_conversion_only_last_slash() {
        // If the tag itself ever contained a '/', only the final '/' gets
        // converted. In normal flodl-hf usage, tags don't contain '/'.
        assert_eq!(hf_key_from_flodl_key("a/b/c"), "a/b.c",);
    }

    #[test]
    fn hf_key_conversion_no_slash_passthrough() {
        assert_eq!(hf_key_from_flodl_key("plain_name"), "plain_name");
    }

    #[test]
    fn display_matches_as_str() {
        let p = HfPath::new("bert").sub("encoder");
        assert_eq!(format!("{p}"), "bert.encoder");
    }
}
