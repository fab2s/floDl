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
//! config. `<env>` is selected via the `@<env>` token, `--env <env>`, or
//! `FDL_ENV=<env>`.

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

/// Config filename extensions in preference order. Matches the order of
/// `config::CONFIG_NAMES` (`fdl.yaml` before `fdl.yml`) so overlay resolution
/// picks the same extension the base file would when both exist.
const EXTENSIONS: &[&str] = &["yaml", "yml", "json"];

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
#[path = "overlay_tests.rs"]
mod tests;
