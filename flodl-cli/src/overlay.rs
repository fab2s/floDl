//! Multi-environment configuration overlays.
//!
//! An `fdl.yml` project manifest can be layered with per-environment files
//! (e.g. `fdl.local.yml`, `fdl.ci.yml`, `fdl.cloud.yml`). When an environment
//! is active, its file is deep-merged on top of the base config before the
//! strongly-typed [`ProjectConfig`](crate::config::ProjectConfig) /
//! [`CommandConfig`](crate::config::CommandConfig) deserialization runs.
//!
//! # Merge rules
//!
//! - **Maps**: deep-merge. Recurse into nested maps; overlay keys win.
//! - **Scalars**: replace. Overlay value takes over.
//! - **Lists**: replace entirely. (Order is contentious — append/prepend
//!   modes cause more debugging pain than they save.)
//! - **`null` deletes**: a key set to `null` in the overlay removes it from
//!   the merged map (not "write null"). Useful for "reset to defaults in
//!   this env."
//!
//! # Discovery
//!
//! Sibling files matching `fdl.<env>.{yml,yaml,json}` alongside the base
//! config. The `<env>` token is the first-arg env selector.

use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

// ── Deep-merge ──────────────────────────────────────────────────────────

/// Deep-merge `over` onto `base`. Maps recurse; scalars and lists replace;
/// `null` values in a map context delete the key from the result.
///
/// Non-Mapping destinations are replaced wholesale when the overlay is a
/// Mapping too — i.e. no cross-type merging, the newer value wins.
pub fn deep_merge(base: Value, over: Value) -> Value {
    match (base, over) {
        (Value::Mapping(base_map), Value::Mapping(over_map)) => {
            Value::Mapping(merge_mapping(base_map, over_map))
        }
        // Scalar, sequence, or type-change: overlay replaces base.
        (_, over) => over,
    }
}

fn merge_mapping(mut base: Mapping, over: Mapping) -> Mapping {
    for (k, v) in over {
        if matches!(v, Value::Null) {
            base.remove(&k);
            continue;
        }
        match base.remove(&k) {
            Some(existing) => {
                base.insert(k, deep_merge(existing, v));
            }
            None => {
                base.insert(k, v);
            }
        }
    }
    base
}

/// Merge a chain of layers left-to-right. The first is the base; each
/// subsequent layer is merged on top of the running result.
pub fn merge_layers<I>(layers: I) -> Value
where
    I: IntoIterator<Item = Value>,
{
    layers
        .into_iter()
        .reduce(deep_merge)
        .unwrap_or(Value::Null)
}

// ── Discovery ───────────────────────────────────────────────────────────

/// Config filename extensions in preference order. Mirrors `config::CONFIG_NAMES`
/// but exposed here so overlay lookup matches sibling base files.
const EXTENSIONS: &[&str] = &["yml", "yaml", "json"];

/// Find a sibling overlay for `env` next to `base_config`.
///
/// `base_config` should be the resolved path to the base `fdl.yml` (not a
/// directory). Returns `Some(path)` if `fdl.<env>.<ext>` exists for any
/// supported extension, `None` otherwise.
pub fn find_env_file(base_config: &Path, env: &str) -> Option<PathBuf> {
    let dir = base_config.parent()?;
    for ext in EXTENSIONS {
        let candidate = dir.join(format!("fdl.{env}.{ext}"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// List every environment overlay discoverable beside the base config.
///
/// Returns env names (without `fdl.` prefix or extension), sorted. Duplicate
/// names across extensions are de-duplicated — the first-found wins, matching
/// [`find_env_file`] precedence.
pub fn list_envs(base_config: &Path) -> Vec<String> {
    let Some(dir) = base_config.parent() else {
        return Vec::new();
    };
    let entries = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut envs = std::collections::BTreeSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Some(stripped) = name_str.strip_prefix("fdl.") else {
            continue;
        };
        // Must have at least one `.` separating env name from extension.
        let Some((env, ext)) = stripped.rsplit_once('.') else {
            continue;
        };
        if env.is_empty() || !EXTENSIONS.contains(&ext) {
            continue;
        }
        envs.insert(env.to_string());
    }
    envs.into_iter().collect()
}

// ── Provenance-tracking merge ───────────────────────────────────────────
//
// [`deep_merge`] is lossy: once values collapse together we lose track of
// which layer contributed each leaf. For `fdl config show`'s per-line
// source annotation we need the merged *and* the origin, so we carry a
// parallel tree that records a layer index at every leaf / sequence /
// replaced-wholesale value. Maps are recursive: each entry carries its
// own origin, the map itself has no single source. Sequences are
// replaced wholesale, so they behave as leaves — the whole list is
// attributed to whichever layer last wrote it.

/// A merged value plus the layer that produced each leaf.
///
/// Layer indices are 0-based and refer to the slice passed to
/// [`merge_layers_annotated`]: `0` is the base, `1` is the first overlay,
/// and so on. Callers map indices to display labels (filenames, usually)
/// at render time.
#[derive(Debug, Clone)]
pub enum AnnotatedNode {
    /// Terminal value: scalar, null, or sequence. `source` is the layer
    /// that last wrote this value.
    Leaf { value: Value, source: usize },
    /// Mapping node. `entries` preserves insertion order matching
    /// [`deep_merge`]'s re-key-to-end behaviour (overridden keys move to
    /// the tail of the map, matching the final `serde_yaml` serialisation).
    Map { entries: Vec<(Value, AnnotatedNode)> },
}

impl AnnotatedNode {
    /// Materialise the merged [`Value`] — useful for equality tests
    /// against [`deep_merge`] output.
    pub fn to_value(&self) -> Value {
        match self {
            AnnotatedNode::Leaf { value, .. } => value.clone(),
            AnnotatedNode::Map { entries } => {
                let mut m = Mapping::new();
                for (k, v) in entries {
                    m.insert(k.clone(), v.to_value());
                }
                Value::Mapping(m)
            }
        }
    }
}

/// Merge a chain of layers left-to-right with provenance tracking. Mirrors
/// [`merge_layers`] but returns an [`AnnotatedNode`] instead of a flat
/// [`Value`]. Layer indices in the result are positions into `layers`.
pub fn merge_layers_annotated(layers: &[Value]) -> AnnotatedNode {
    if layers.is_empty() {
        return AnnotatedNode::Leaf {
            value: Value::Null,
            source: 0,
        };
    }

    let mut result = to_annotated(&layers[0], 0);
    for (i, layer) in layers.iter().enumerate().skip(1) {
        result = deep_merge_annotated(result, layer, i);
    }
    result
}

/// Lift a raw [`Value`] into an [`AnnotatedNode`] tagged with one source.
fn to_annotated(v: &Value, source: usize) -> AnnotatedNode {
    match v {
        Value::Mapping(m) => {
            let entries = m
                .iter()
                .map(|(k, v)| (k.clone(), to_annotated(v, source)))
                .collect();
            AnnotatedNode::Map { entries }
        }
        other => AnnotatedNode::Leaf {
            value: other.clone(),
            source,
        },
    }
}

/// Merge `over` onto `base` with provenance. Mirrors [`deep_merge`] but
/// carries source indices; `over_source` is the layer index for any
/// leaves the overlay introduces or replaces.
fn deep_merge_annotated(
    base: AnnotatedNode,
    over: &Value,
    over_source: usize,
) -> AnnotatedNode {
    match (base, over) {
        (AnnotatedNode::Map { mut entries }, Value::Mapping(over_map)) => {
            for (k, v) in over_map {
                if matches!(v, Value::Null) {
                    entries.retain(|(ek, _)| ek != k);
                    continue;
                }
                let pos = entries.iter().position(|(ek, _)| ek == k);
                match pos {
                    Some(p) => {
                        // Match deep_merge's re-key-to-end behaviour: drop
                        // the existing entry and re-append under merge.
                        let (_, existing) = entries.remove(p);
                        let merged = deep_merge_annotated(existing, v, over_source);
                        entries.push((k.clone(), merged));
                    }
                    None => {
                        entries.push((k.clone(), to_annotated(v, over_source)));
                    }
                }
            }
            AnnotatedNode::Map { entries }
        }
        // Type change or scalar-over-anything: overlay replaces wholesale.
        (_, over) => to_annotated(over, over_source),
    }
}

// ── Rendering with inline source comments ───────────────────────────────

/// Emit an [`AnnotatedNode`] as YAML with a trailing `# <label>` on each
/// leaf line, column-aligned for legibility.
///
/// `source_labels[i]` is the label shown for layer index `i` (typically a
/// filename). Sequences are rendered inline when all items are scalars
/// and the resulting line fits the `INLINE_SEQ_LIMIT` threshold; otherwise
/// they drop to block style with the source tag on the key line.
pub fn render_annotated_yaml(node: &AnnotatedNode, source_labels: &[String]) -> String {
    // Three-pass render:
    // 1. Emit raw lines with `\0` between body and source tag.
    // 2. Pad bodies so `# tag` comments align.
    // 3. Colorize: green keys + dim-gray tags (no-op if color disabled).
    //
    // Color happens AFTER alignment so the ANSI escape bytes don't get
    // counted as body width.
    let mut raw = String::new();
    render_node(node, 0, source_labels, &mut raw);
    let aligned = align_comments(&raw);
    colorize_keys(&aligned)
}

/// Inline-sequence threshold: combined line length beyond which a
/// scalar-only sequence drops from `[a, b, c]` to block form.
const INLINE_SEQ_LIMIT: usize = 80;

fn render_node(node: &AnnotatedNode, indent: usize, labels: &[String], out: &mut String) {
    match node {
        AnnotatedNode::Leaf { value, source } => {
            // Top-level leaf (root is a bare scalar). Rare but support it.
            let tag = label(labels, *source);
            emit_line(out, indent, &format_scalar(value), Some(&tag));
        }
        AnnotatedNode::Map { entries } => {
            for (k, child) in entries {
                let key = format_key(k);
                match child {
                    AnnotatedNode::Leaf { value, source } => {
                        let tag = label(labels, *source);
                        render_leaf_entry(&key, value, &tag, indent, out);
                    }
                    AnnotatedNode::Map { .. } => {
                        // Header line for a nested map: no tag (the map
                        // itself has no single source).
                        emit_header(out, indent, &format!("{key}:"));
                        render_node(child, indent + 2, labels, out);
                    }
                }
            }
        }
    }
}

fn render_leaf_entry(key: &str, value: &Value, tag: &str, indent: usize, out: &mut String) {
    match value {
        Value::Sequence(items) if items.iter().all(is_inline_scalar) => {
            let inline = format!(
                "{key}: [{}]",
                items
                    .iter()
                    .map(format_scalar)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if indent + inline.len() <= INLINE_SEQ_LIMIT {
                emit_line(out, indent, &inline, Some(tag));
            } else {
                emit_line(out, indent, &format!("{key}:"), Some(tag));
                for item in items {
                    emit_header(out, indent + 2, &format!("- {}", format_scalar(item)));
                }
            }
        }
        Value::Sequence(items) => {
            emit_line(out, indent, &format!("{key}:"), Some(tag));
            for item in items {
                match item {
                    Value::Mapping(m) => {
                        // First entry on the `-` line, rest indented at the
                        // same column. Each entry recurses through
                        // `render_mapping_field` so nested sequences render
                        // correctly (was `ranks: - 0` from format_scalar's
                        // defensive fallback).
                        let mut it = m.iter();
                        if let Some((first_k, first_v)) = it.next() {
                            render_mapping_field(
                                first_k, first_v, indent + 2, Some("- "), out,
                            );
                            for (k, v) in it {
                                render_mapping_field(k, v, indent + 4, None, out);
                            }
                        }
                    }
                    other => {
                        emit_header(out, indent + 2, &format!("- {}", format_scalar(other)));
                    }
                }
            }
        }
        other => {
            emit_line(out, indent, &format!("{key}: {}", format_scalar(other)), Some(tag));
        }
    }
}

/// Render one `key: value` field inside a mapping that is itself a list
/// item. Same logic as [`render_leaf_entry`] but emits header lines (no
/// source tag) since the containing list already carried the source.
///
/// `prefix` is `Some("- ")` for the first key of a list item (printed
/// flush with the dash) and `None` for subsequent keys (printed at the
/// indent column for alignment with the first key).
fn render_mapping_field(
    k: &Value,
    v: &Value,
    indent: usize,
    prefix: Option<&str>,
    out: &mut String,
) {
    let key = format_key(k);
    let head = format!("{}{key}", prefix.unwrap_or(""));
    match v {
        Value::Sequence(items) if items.iter().all(is_inline_scalar) => {
            let inline = format!(
                "{head}: [{}]",
                items
                    .iter()
                    .map(format_scalar)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if indent + inline.len() <= INLINE_SEQ_LIMIT {
                emit_header(out, indent, &inline);
            } else {
                emit_header(out, indent, &format!("{head}:"));
                for item in items {
                    emit_header(out, indent + 2, &format!("- {}", format_scalar(item)));
                }
            }
        }
        Value::Sequence(items) => {
            emit_header(out, indent, &format!("{head}:"));
            for item in items {
                emit_header(out, indent + 2, &format!("- {}", format_scalar(item)));
            }
        }
        Value::Mapping(_) => {
            emit_header(out, indent, &format!("{head}:"));
            // Mapping values inside list items: walk recursively.
            if let Value::Mapping(m) = v {
                for (k2, v2) in m {
                    render_mapping_field(k2, v2, indent + 2, None, out);
                }
            }
        }
        other => {
            emit_header(out, indent, &format!("{head}: {}", format_scalar(other)));
        }
    }
}

/// Write a line that will participate in column alignment. `body` is the
/// YAML body (key: value); `tag` is the source label. Body and tag are
/// separated by a `\0` sentinel so [`align_comments`] can pad precisely.
fn emit_line(out: &mut String, indent: usize, body: &str, tag: Option<&str>) {
    for _ in 0..indent {
        out.push(' ');
    }
    out.push_str(body);
    if let Some(t) = tag {
        out.push('\0');
        out.push_str(t);
    }
    out.push('\n');
}

/// Write a header/structural line (no source tag). No `\0` sentinel so
/// alignment leaves it untouched.
fn emit_header(out: &mut String, indent: usize, body: &str) {
    for _ in 0..indent {
        out.push(' ');
    }
    out.push_str(body);
    out.push('\n');
}

/// Align `# <tag>` comments across lines that carry the `\0` sentinel.
/// Lines without the sentinel pass through unchanged. Comment column is
/// `max(body_width) + 2`, clamped to a minimum for single-line configs.
/// Maximum body width to track for comment alignment. Beyond this, a long
/// line (e.g. a multi-flag shell command) breaks alignment for that line
/// only -- its comment falls right after with a 2-space gutter. This stops
/// one 90-char clippy command from pushing every comment past the terminal
/// edge and triggering wrap.
const ALIGN_CAP: usize = 50;

fn align_comments(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let mut max_body = 0;
    for line in &lines {
        if let Some(idx) = line.find('\0') {
            // Only count lines that fit under the cap; outliers don't
            // drag everyone else's column rightward.
            if idx <= ALIGN_CAP {
                max_body = max_body.max(idx);
            }
        }
    }
    // 2-space gutter before the `#`. Minimum column so single-key files
    // still look deliberate rather than cramped.
    let col = max_body.max(12) + 2;

    let mut out = String::with_capacity(raw.len() + lines.len() * 4);
    for line in &lines {
        match line.find('\0') {
            Some(idx) => {
                let (body, rest) = line.split_at(idx);
                let tag = &rest[1..]; // skip the '\0'
                out.push_str(body);
                let body_width = body.chars().count();
                // If the body is too wide to align cleanly, fall back to a
                // 2-space gutter for that single line.
                let target_col = if body_width > ALIGN_CAP { body_width + 2 } else { col };
                for _ in body_width..target_col {
                    out.push(' ');
                }
                // Preserve a `\0` sentinel between padding and `# tag` so
                // the next pass (colorize_keys) can split unambiguously.
                // colorize_keys is mandatory and always strips it.
                out.push('\0');
                out.push_str("# ");
                out.push_str(tag);
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

/// Final render pass: colorize keys (green) and source tags (dark-gray),
/// and strip the `\0` body/tag sentinel emitted by [`align_comments`].
///
/// When color is disabled the function still runs (to remove `\0`) but
/// emits no ANSI escapes. `\x1b[32m` (green) matches `fdl -h`'s
/// `-h, --help` style for option names. `\x1b[90m` (bright-black) is the
/// most reliable "dim" effect across terminal themes; `\x1b[2m` actual-dim
/// is unimplemented or near-invisible in many setups.
fn colorize_keys(text: &str) -> String {
    let color = crate::style::color_enabled();
    let key_open = if color { "\x1b[32m" } else { "" };
    let key_close = if color { "\x1b[0m" } else { "" };
    let tag_open = if color { "\x1b[90m" } else { "" };
    let tag_close = if color { "\x1b[0m" } else { "" };

    let mut out = String::with_capacity(text.len() + text.lines().count() * 16);
    for line in text.lines() {
        // The `\0` sentinel marks the body / tag boundary (emitted by
        // align_comments). Unambiguous -- can't appear in user content.
        let (body, comment) = match line.find('\0') {
            Some(i) => (&line[..i], Some(&line[i + 1..])),
            None => (line, None),
        };

        // Key colorization on the body part.
        match find_key_segment(body) {
            Some((key_start, key_end)) => {
                out.push_str(&body[..key_start]);
                out.push_str(key_open);
                out.push_str(&body[key_start..key_end]);
                out.push_str(key_close);
                out.push_str(&body[key_end..]);
            }
            None => out.push_str(body),
        }

        if let Some(c) = comment {
            out.push_str(tag_open);
            out.push_str(c);
            out.push_str(tag_close);
        }
        out.push('\n');
    }
    out
}

/// Locate the `(start, end)` byte range of the YAML key on this line, or
/// None if there is no key (blank, list-scalar, etc.). Handles list-item
/// prefix `- ` and arbitrary indent.
fn find_key_segment(line: &str) -> Option<(usize, usize)> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    // Optional list-item dash.
    if i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] == b' ' {
        i += 2;
    }
    let key_start = i;
    // Scan for the first `:` followed by space / end / newline.
    while i < bytes.len() {
        if bytes[i] == b':' {
            let next = bytes.get(i + 1).copied();
            match next {
                None | Some(b' ') | Some(b'\n') => {
                    if i > key_start {
                        return Some((key_start, i));
                    }
                    return None;
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn label(labels: &[String], source: usize) -> String {
    labels
        .get(source)
        .cloned()
        .unwrap_or_else(|| format!("layer[{source}]"))
}

fn is_inline_scalar(v: &Value) -> bool {
    matches!(
        v,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

/// Format a scalar for display in a YAML line. Strings are quoted only
/// when they would otherwise parse ambiguously (start with a special
/// char, contain a `:` followed by space, etc.). Goal: look like the
/// user's source file when unambiguous, quote only when required.
fn format_scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format_string(s),
        Value::Sequence(_) | Value::Mapping(_) => {
            // Shouldn't be called with a container — defensive fallback.
            serde_yaml::to_string(v).unwrap_or_default().trim().to_string()
        }
        Value::Tagged(t) => serde_yaml::to_string(&**t)
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

fn format_key(k: &Value) -> String {
    match k {
        Value::String(s) => {
            // Most config keys are plain identifiers; keep them unquoted.
            if is_plain_key(s) {
                s.clone()
            } else {
                format_string(s)
            }
        }
        other => format_scalar(other),
    }
}

fn is_plain_key(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn format_string(s: &str) -> String {
    // Quote if the raw string would mis-parse as something else, or if
    // it contains characters that make unquoted YAML ambiguous.
    let needs_quote = s.is_empty()
        || s.contains(':')
        || s.contains('#')
        || s.contains('\n')
        || s.contains('"')
        || s.starts_with(|c: char| c.is_whitespace() || "!&*>|%@`[]{},-?".contains(c))
        || matches!(s, "true" | "false" | "null" | "yes" | "no" | "~")
        || s.parse::<f64>().is_ok();
    if needs_quote {
        // Double-quoted with JSON-style escapes.
        let escaped = s
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\t', "\\t");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

/// Load a YAML/JSON file as a [`Value`]. Extension-based dispatch on the
/// file suffix (`.yml`, `.yaml`, `.json`).
pub fn load_value(path: &Path) -> Result<Value, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("yaml");
    match ext {
        "json" => serde_json::from_str::<Value>(&content)
            .map_err(|e| format!("{}: {}", path.display(), e)),
        _ => serde_yaml::from_str::<Value>(&content)
            .map_err(|e| format!("{}: {}", path.display(), e)),
    }
}

// ── `inherit-from:` chain resolution ────────────────────────────────────
//
// A config file can declare a top-level `inherit-from: <path>` that names
// a parent to merge under. Chains are linear (single parent) so the
// effective layer list becomes [deepest-ancestor, ..., direct-parent, this].
// The `inherit-from` key is stripped from every returned value so it
// doesn't leak into the deserialised config.

/// YAML key used by [`resolve_chain`] to discover the parent layer.
const INHERIT_KEY: &str = "inherit-from";

/// Load `path` and every ancestor reachable via `inherit-from:`, returning
/// them in merge order (deepest ancestor first, `path` itself last). The
/// `inherit-from` key is removed from every returned [`Value`].
///
/// Relative ancestor paths are resolved against the directory of the file
/// that declared the `inherit-from:`. Cycles (including self-inheritance)
/// are detected via the recursion stack and surface as an error listing
/// the full cycle for fast diagnosis.
pub fn resolve_chain(path: &Path) -> Result<Vec<(PathBuf, Value)>, String> {
    let mut stack: Vec<PathBuf> = Vec::new();
    let mut out: Vec<(PathBuf, Value)> = Vec::new();
    resolve_chain_inner(path, &mut stack, &mut out)?;
    Ok(out)
}

fn resolve_chain_inner(
    path: &Path,
    stack: &mut Vec<PathBuf>,
    out: &mut Vec<(PathBuf, Value)>,
) -> Result<(), String> {
    let canonical = path.canonicalize().map_err(|e| {
        format!(
            "cannot resolve inherit-from target `{}`: {e}",
            path.display()
        )
    })?;

    if stack.contains(&canonical) {
        let mut chain: Vec<String> = stack.iter().map(|p| p.display().to_string()).collect();
        chain.push(canonical.display().to_string());
        return Err(format!("inherit-from cycle detected: {}", chain.join(" -> ")));
    }

    stack.push(canonical.clone());

    let mut value = load_value(path)?;
    let parent = extract_inherit_from(&mut value, path)?;

    if let Some(parent_rel) = parent {
        let parent_abs = if Path::new(&parent_rel).is_absolute() {
            PathBuf::from(&parent_rel)
        } else {
            canonical
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&parent_rel)
        };
        resolve_chain_inner(&parent_abs, stack, out)?;
    }

    stack.pop();
    out.push((canonical, value));
    Ok(())
}

/// Pop the top-level `inherit-from` key from a mapping and return its
/// string value. A missing or explicitly-null key returns `Ok(None)`.
/// A non-string value errors with the offending type named.
fn extract_inherit_from(value: &mut Value, path: &Path) -> Result<Option<String>, String> {
    let Value::Mapping(m) = value else {
        return Ok(None);
    };
    let key = Value::String(INHERIT_KEY.to_string());
    match m.remove(&key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.is_empty() => Err(format!(
            "{INHERIT_KEY} in {} must be a non-empty path",
            path.display()
        )),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(other) => Err(format!(
            "{INHERIT_KEY} in {} must be a string path, got {}",
            path.display(),
            type_name(&other)
        )),
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Sequence(_) => "sequence",
        Value::Mapping(_) => "mapping",
        Value::Tagged(_) => "tagged",
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn yaml(s: &str) -> Value {
        serde_yaml::from_str(s).expect("test fixture must parse")
    }

    /// Build `Vec<String>` from string literals — shorter than repeating
    /// `.to_string()` in every path assertion.
    fn p(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn scalar_over_scalar_replaces() {
        let base = yaml("42");
        let over = yaml("99");
        assert_eq!(deep_merge(base, over), yaml("99"));
    }

    #[test]
    fn map_keys_deep_merge() {
        let base = yaml(
            r"
            a: 1
            nested:
              x: one
              y: two
            ",
        );
        let over = yaml(
            r"
            nested:
              y: TWO
              z: three
            b: 2
            ",
        );
        let expected = yaml(
            r"
            a: 1
            b: 2
            nested:
              x: one
              y: TWO
              z: three
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn lists_replace_not_append() {
        let base = yaml(
            r"
            items: [a, b, c]
            ",
        );
        let over = yaml(
            r"
            items: [x, y]
            ",
        );
        let expected = yaml(
            r"
            items: [x, y]
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn null_in_overlay_deletes_key() {
        let base = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            training:
              epochs: 10
            ",
        );
        let over = yaml(
            r"
            ddp: ~
            training:
              epochs: 20
            ",
        );
        // `ddp: null` removes the whole block; training.epochs updates.
        let expected = yaml(
            r"
            training:
              epochs: 20
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn null_leaf_removes_single_key() {
        let base = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            ",
        );
        let over = yaml(
            r"
            ddp:
              anchor: ~
            ",
        );
        let expected = yaml(
            r"
            ddp:
              policy: cadence
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn overlay_adds_new_top_level_key() {
        let base = yaml("a: 1");
        let over = yaml("b: 2");
        let expected = yaml(
            r"
            a: 1
            b: 2
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn merge_chain_three_layers() {
        let l1 = yaml("a: 1\nb: 1");
        let l2 = yaml("b: 2\nc: 2");
        let l3 = yaml("c: 3");
        let got = merge_layers(vec![l1, l2, l3]);
        let expected = yaml(
            r"
            a: 1
            b: 2
            c: 3
            ",
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn type_change_overlay_replaces_wholesale() {
        let base = yaml(
            r"
            ddp:
              policy: cadence
            ",
        );
        let over = yaml(
            r"
            ddp: solo-0
            ",
        );
        let expected = yaml(
            r"
            ddp: solo-0
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn type_change_scalar_base_mapping_overlay_replaces() {
        // Symmetry with `type_change_overlay_replaces_wholesale`: when
        // the base is a scalar and the overlay is a mapping, the mapping
        // wins wholesale. No attempt at cross-type merging.
        let base = yaml(
            r"
            ddp: solo-0
            ",
        );
        let over = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            ",
        );
        let expected = yaml(
            r"
            ddp:
              policy: cadence
              anchor: 3
            ",
        );
        assert_eq!(deep_merge(base, over), expected);
    }

    #[test]
    fn list_envs_discovers_sibling_overlays() {
        let tmp = tempdir();
        std::fs::write(tmp.path().join("fdl.yml"), "description: base").unwrap();
        std::fs::write(tmp.path().join("fdl.ci.yml"), "description: ci").unwrap();
        std::fs::write(tmp.path().join("fdl.cloud.yaml"), "description: cloud").unwrap();
        std::fs::write(tmp.path().join("fdl.prod.json"), "{}").unwrap();
        // Decoys — must NOT be listed.
        std::fs::write(tmp.path().join("fdl.yml.example"), "").unwrap();
        std::fs::write(tmp.path().join("other.ci.yml"), "").unwrap();
        std::fs::write(tmp.path().join("fdl.yml.bak"), "").unwrap();

        let envs = list_envs(&tmp.path().join("fdl.yml"));
        assert_eq!(envs, vec!["ci".to_string(), "cloud".into(), "prod".into()]);
    }

    #[test]
    fn find_env_file_respects_extension_precedence() {
        let tmp = tempdir();
        std::fs::write(tmp.path().join("fdl.yml"), "").unwrap();
        std::fs::write(tmp.path().join("fdl.ci.yml"), "# yml wins").unwrap();
        std::fs::write(tmp.path().join("fdl.ci.yaml"), "# yaml loses").unwrap();

        let got = find_env_file(&tmp.path().join("fdl.yml"), "ci").unwrap();
        assert_eq!(got.file_name().unwrap().to_str(), Some("fdl.ci.yml"));
    }

    #[test]
    fn find_env_file_missing_returns_none() {
        let tmp = tempdir();
        std::fs::write(tmp.path().join("fdl.yml"), "").unwrap();
        assert!(find_env_file(&tmp.path().join("fdl.yml"), "nope").is_none());
    }

    // ── Annotated merge ──────────────────────────────────────────────────

    /// Collect every leaf's (key-path, source-index) from an AnnotatedNode.
    /// Key path elements are YAML `Value`s (almost always strings in our
    /// configs) for parity with [`AnnotatedNode::Map`]'s key type.
    fn leaves(node: &AnnotatedNode) -> Vec<(Vec<String>, usize)> {
        fn walk(node: &AnnotatedNode, path: &mut Vec<String>, out: &mut Vec<(Vec<String>, usize)>) {
            match node {
                AnnotatedNode::Leaf { source, .. } => out.push((path.clone(), *source)),
                AnnotatedNode::Map { entries } => {
                    for (k, v) in entries {
                        let key = match k {
                            Value::String(s) => s.clone(),
                            other => format!("{other:?}"),
                        };
                        path.push(key);
                        walk(v, path, out);
                        path.pop();
                    }
                }
            }
        }
        let mut out = Vec::new();
        walk(node, &mut Vec::new(), &mut out);
        out
    }

    #[test]
    fn annotated_single_layer_tags_every_leaf_with_zero() {
        let layers = vec![yaml("ddp:\n  policy: cadence\n  anchor: 3\ntraining:\n  epochs: 10\n")];
        let node = merge_layers_annotated(&layers);
        for (path, src) in leaves(&node) {
            assert_eq!(src, 0, "{path:?} should be tagged with layer 0");
        }
    }

    #[test]
    fn annotated_overlay_replaces_key_source() {
        let layers = vec![
            yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
            yaml("ddp:\n  anchor: 5\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
        assert_eq!(by_path[&p(&["ddp", "policy"])], 0);
        assert_eq!(by_path[&p(&["ddp", "anchor"])], 1);
    }

    #[test]
    fn annotated_added_key_tagged_with_overlay() {
        let layers = vec![
            yaml("ddp:\n  policy: cadence\n"),
            yaml("training:\n  epochs: 20\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
        assert_eq!(by_path[&p(&["training", "epochs"])], 1);
    }

    #[test]
    fn annotated_null_deletes_key_and_removes_leaf() {
        let layers = vec![
            yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
            yaml("ddp:\n  anchor: ~\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let paths: Vec<Vec<String>> = leaves(&node).into_iter().map(|(path, _)| path).collect();
        assert!(paths.contains(&p(&["ddp", "policy"])));
        assert!(!paths.iter().any(|path| path == &p(&["ddp", "anchor"])));
    }

    #[test]
    fn annotated_type_change_resets_source_to_overlay() {
        // Mapping in base → scalar in overlay: the whole subtree collapses
        // to a Leaf tagged with the overlay's index.
        let layers = vec![
            yaml("ddp:\n  policy: cadence\n"),
            yaml("ddp: solo-0\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
        assert_eq!(by_path[&p(&["ddp"])], 1);
        assert!(!by_path.contains_key(&p(&["ddp", "policy"])));
    }

    #[test]
    fn annotated_list_replaced_wholesale_tagged_with_setter() {
        // Lists are replace-not-append, so the whole sequence is attributed
        // to the layer that last wrote it.
        let layers = vec![
            yaml("regions: [eu-west]\n"),
            yaml("regions: [us-east, ap-south]\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
        assert_eq!(by_path[&p(&["regions"])], 1);
    }

    #[test]
    fn annotated_three_layer_chain() {
        let layers = vec![
            yaml("a: 1\nb: 1\nc: 1\n"),
            yaml("b: 2\nc: 2\n"),
            yaml("c: 3\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let by_path: BTreeMap<Vec<String>, usize> = leaves(&node).into_iter().collect();
        assert_eq!(by_path[&p(&["a"])], 0);
        assert_eq!(by_path[&p(&["b"])], 1);
        assert_eq!(by_path[&p(&["c"])], 2);
    }

    #[test]
    fn annotated_to_value_matches_deep_merge() {
        let l1 = yaml("ddp:\n  policy: cadence\n  anchor: 3\ntraining:\n  epochs: 10\n");
        let l2 = yaml("ddp:\n  anchor: 5\ntraining:\n  seed: 42\n");
        let annotated = merge_layers_annotated(&[l1.clone(), l2.clone()]);
        let plain = deep_merge(l1, l2);
        assert_eq!(annotated.to_value(), plain);
    }

    // ── Rendering ────────────────────────────────────────────────────────

    fn labels(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn render_tags_every_leaf_with_filename() {
        let layers = vec![yaml("ddp:\n  policy: cadence\n  anchor: 3\n")];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        for line in out.lines() {
            if line.contains(':') && !line.trim_end().ends_with(':') {
                assert!(line.contains("# fdl.yml"), "missing tag on: `{line}`");
            }
        }
    }

    #[test]
    fn render_tags_overlay_keys_with_overlay_filename() {
        let layers = vec![
            yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
            yaml("ddp:\n  anchor: 5\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml", "fdl.ci.yml"]));
        // policy unchanged → tagged with base.
        let policy_line = out.lines().find(|l| l.contains("policy:")).unwrap();
        assert!(policy_line.contains("# fdl.yml") && !policy_line.contains("# fdl.ci.yml"));
        // anchor overridden → tagged with overlay.
        let anchor_line = out.lines().find(|l| l.contains("anchor:")).unwrap();
        assert!(anchor_line.contains("# fdl.ci.yml"));
    }

    #[test]
    fn render_aligns_comment_column() {
        let layers = vec![yaml("a: 1\nbb: 22\nccc: 333\n")];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        // All `#` symbols must land in the same column.
        let cols: Vec<usize> = out
            .lines()
            .filter_map(|l| l.find('#'))
            .collect();
        assert!(cols.len() >= 3);
        let first = cols[0];
        assert!(cols.iter().all(|c| *c == first), "mismatched columns: {cols:?}");
    }

    #[test]
    fn render_inline_short_scalar_list() {
        // `serde_yaml::Number::to_string` preserves `1.0` as `1.0`.
        let layers = vec![yaml("ratios: [1.5, 1.0]\n")];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        assert!(out.contains("ratios: [1.5, 1.0]"), "got:\n{out}");
        assert!(out.lines().next().unwrap().contains("# fdl.yml"));
    }

    #[test]
    fn render_deleted_key_absent_from_output() {
        let layers = vec![
            yaml("ddp:\n  policy: cadence\n  anchor: 3\n"),
            yaml("ddp:\n  anchor: ~\n"),
        ];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml", "fdl.ci.yml"]));
        assert!(!out.contains("anchor"), "deleted key leaked: {out}");
        assert!(out.contains("policy"));
    }

    #[test]
    fn render_header_lines_have_no_comment() {
        // The `ddp:` header line is a nested-map opener — it has no single
        // source, so it gets no trailing `# <label>`.
        let layers = vec![yaml("ddp:\n  policy: cadence\n")];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        let header = out.lines().find(|l| l.trim() == "ddp:").unwrap();
        assert!(!header.contains('#'));
    }

    #[test]
    fn render_quotes_ambiguous_strings() {
        // `true` as a literal string must be quoted so it doesn't
        // round-trip as a boolean.
        let layers = vec![yaml("flag: \"true\"\n")];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        assert!(out.contains("flag: \"true\""), "got:\n{out}");
    }

    #[test]
    fn render_long_scalar_list_drops_to_block_form() {
        let long: Vec<String> = (0..30).map(|i| format!("item-number-{i}")).collect();
        let yaml_src = format!("items: [{}]\n", long.join(", "));
        let layers = vec![yaml(&yaml_src)];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        assert!(out.contains("items:  "), "expected header line with tag");
        assert!(out.contains("- item-number-0"));
    }

    #[test]
    fn render_sequence_in_mapping_list_item_inlines_scalars() {
        // Regression: previously rendered `ranks: - 0` because
        // format_scalar's defensive Sequence fallback emitted block YAML.
        let layers = vec![yaml(
            "cluster:\n  hosts:\n    - name: exa\n      ranks: [0, 1]\n      local_devices: [2, 3]\n",
        )];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        assert!(
            out.contains("ranks: [0, 1]"),
            "ranks should inline as `[0, 1]`, got:\n{out}"
        );
        assert!(
            out.contains("local_devices: [2, 3]"),
            "local_devices should inline as `[2, 3]`, got:\n{out}"
        );
        assert!(
            !out.contains("ranks: - "),
            "must not produce `ranks: - 0`, got:\n{out}"
        );
    }

    #[test]
    fn render_long_line_does_not_push_comments_past_cap() {
        // Regression: previously, one ~90-char clippy command pushed every
        // other comment to ~100+ cols, wrapping in the terminal.
        let layers = vec![yaml(
            "short: 1\nlong: \"some pretty long shell command with many flags and pipes\"\nshort2: 2\n",
        )];
        let node = merge_layers_annotated(&layers);
        let out = render_annotated_yaml(&node, &labels(&["fdl.yml"]));
        let short_col = out
            .lines()
            .find(|l| l.starts_with("short:"))
            .unwrap()
            .find('#')
            .unwrap();
        // ALIGN_CAP + small padding; assert short lines align to a sane
        // column (well under terminal width).
        assert!(
            short_col < 60,
            "short-line comment ended up at col {short_col}, expected < 60"
        );
    }

    // ── inherit-from chain resolution ────────────────────────────────────

    /// Canonicalise a path so tests can compare against `resolve_chain`'s
    /// returned paths (which are always canonical).
    fn canon(p: &Path) -> PathBuf {
        p.canonicalize().expect("canonicalize fixture path")
    }

    #[test]
    fn resolve_chain_single_file_no_inherit() {
        let tmp = tempdir();
        let f = tmp.path().join("fdl.yml");
        std::fs::write(&f, "description: test\nddp:\n  policy: cadence\n").unwrap();
        let chain = resolve_chain(&f).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].0, canon(&f));
    }

    #[test]
    fn resolve_chain_strips_inherit_from_key() {
        let tmp = tempdir();
        let parent = tmp.path().join("fdl.yml");
        let child = tmp.path().join("fdl.ci.yml");
        std::fs::write(&parent, "a: 1\n").unwrap();
        std::fs::write(&child, "inherit-from: fdl.yml\nb: 2\n").unwrap();
        let chain = resolve_chain(&child).unwrap();
        assert_eq!(chain.len(), 2);
        // First layer is the parent (deepest), second is the child.
        assert_eq!(chain[0].0, canon(&parent));
        assert_eq!(chain[1].0, canon(&child));
        // inherit-from must not appear in the returned values.
        for (_, v) in &chain {
            if let Value::Mapping(m) = v {
                assert!(!m.contains_key(Value::String("inherit-from".to_string())));
            }
        }
    }

    #[test]
    fn resolve_chain_three_level_ordering() {
        // c inherits from b, b inherits from a. Merge order must be [a, b, c].
        let tmp = tempdir();
        let a = tmp.path().join("a.yml");
        let b = tmp.path().join("b.yml");
        let c = tmp.path().join("c.yml");
        std::fs::write(&a, "x: from-a\n").unwrap();
        std::fs::write(&b, "inherit-from: a.yml\ny: from-b\n").unwrap();
        std::fs::write(&c, "inherit-from: b.yml\nz: from-c\n").unwrap();
        let chain = resolve_chain(&c).unwrap();
        let paths: Vec<PathBuf> = chain.iter().map(|(p, _)| p.clone()).collect();
        assert_eq!(paths, vec![canon(&a), canon(&b), canon(&c)]);
    }

    #[test]
    fn resolve_chain_relative_paths_resolve_from_declaring_file() {
        // Declaring file sits one dir down; inherit-from uses `../base.yml`.
        let tmp = tempdir();
        let base = tmp.path().join("base.yml");
        let nested_dir = tmp.path().join("nested");
        std::fs::create_dir_all(&nested_dir).unwrap();
        let child = nested_dir.join("child.yml");
        std::fs::write(&base, "shared: true\n").unwrap();
        std::fs::write(&child, "inherit-from: ../base.yml\nlocal: true\n").unwrap();
        let chain = resolve_chain(&child).unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].0, canon(&base));
        assert_eq!(chain[1].0, canon(&child));
    }

    #[test]
    fn resolve_chain_absolute_path_works() {
        let tmp = tempdir();
        let parent = tmp.path().join("parent.yml");
        let child = tmp.path().join("child.yml");
        std::fs::write(&parent, "a: 1\n").unwrap();
        // Use absolute path in inherit-from.
        let abs = canon(&parent);
        std::fs::write(
            &child,
            format!("inherit-from: {}\nb: 2\n", abs.display()),
        )
        .unwrap();
        let chain = resolve_chain(&child).unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].0, canon(&parent));
    }

    #[test]
    fn resolve_chain_self_inheritance_errors() {
        let tmp = tempdir();
        let f = tmp.path().join("fdl.yml");
        std::fs::write(&f, "inherit-from: fdl.yml\nx: 1\n").unwrap();
        let err = resolve_chain(&f).unwrap_err();
        assert!(err.contains("cycle"), "got: {err}");
        // Self-loop appears as the same path on both sides of the arrow.
        assert!(err.matches("fdl.yml").count() >= 2, "got: {err}");
    }

    #[test]
    fn resolve_chain_two_file_cycle_errors() {
        // a inherits from b, b inherits from a — classic cycle.
        let tmp = tempdir();
        let a = tmp.path().join("a.yml");
        let b = tmp.path().join("b.yml");
        std::fs::write(&a, "inherit-from: b.yml\nx: 1\n").unwrap();
        std::fs::write(&b, "inherit-from: a.yml\ny: 2\n").unwrap();
        let err = resolve_chain(&a).unwrap_err();
        assert!(err.contains("cycle"), "got: {err}");
        assert!(err.contains("a.yml"));
        assert!(err.contains("b.yml"));
    }

    #[test]
    fn resolve_chain_missing_parent_errors() {
        let tmp = tempdir();
        let f = tmp.path().join("fdl.yml");
        std::fs::write(&f, "inherit-from: missing.yml\nx: 1\n").unwrap();
        let err = resolve_chain(&f).unwrap_err();
        assert!(
            err.contains("cannot resolve inherit-from target"),
            "got: {err}"
        );
        assert!(err.contains("missing.yml"), "got: {err}");
    }

    #[test]
    fn resolve_chain_non_string_inherit_errors() {
        let tmp = tempdir();
        let f = tmp.path().join("fdl.yml");
        std::fs::write(&f, "inherit-from: 42\nx: 1\n").unwrap();
        let err = resolve_chain(&f).unwrap_err();
        assert!(err.contains("must be a string path"), "got: {err}");
        assert!(err.contains("got number"), "got: {err}");
    }

    #[test]
    fn resolve_chain_empty_string_inherit_errors() {
        let tmp = tempdir();
        let f = tmp.path().join("fdl.yml");
        std::fs::write(&f, "inherit-from: \"\"\nx: 1\n").unwrap();
        let err = resolve_chain(&f).unwrap_err();
        assert!(err.contains("non-empty"), "got: {err}");
    }

    #[test]
    fn resolve_chain_null_inherit_ignored() {
        // Explicit `inherit-from: null` == key absent. No error, no parent.
        let tmp = tempdir();
        let f = tmp.path().join("fdl.yml");
        std::fs::write(&f, "inherit-from: ~\nx: 1\n").unwrap();
        let chain = resolve_chain(&f).unwrap();
        assert_eq!(chain.len(), 1);
    }

    // Tiny tempdir helper — standalone so we don't pull in the tempfile crate.
    fn tempdir() -> TempDir {
        TempDir::new()
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let base = std::env::temp_dir();
            let unique = format!(
                "flodl-overlay-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let dir = base.join(unique);
            std::fs::create_dir_all(&dir).expect("tempdir creation");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}
