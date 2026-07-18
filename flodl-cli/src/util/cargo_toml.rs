//! Minimal text-based Cargo.toml editor.
//!
//! flodl-cli keeps external deps minimal (the serde ecosystem only), so
//! we don't pull in the `toml` / `toml_edit` crates just to append a
//! dependency. The editor
//! is intentionally narrow: append a dep to `[dependencies]` if it
//! isn't already declared, preserving every other byte of the file.
//!
//! Anything more sophisticated (feature edits, version bumps, table
//! reshaping) is out of scope and should fall back to manual edits or
//! a real toml crate when the need arises.

use std::fs;
use std::path::Path;

/// Result of an [`add_dep`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddDepOutcome {
    /// The dependency line was appended.
    Added,
    /// `name` was already declared under `[dependencies]`; file untouched.
    AlreadyPresent,
}

/// Append `name = "<version>"` to `[dependencies]` in the Cargo.toml at
/// `path` if `name` isn't already declared there.
///
/// `version` is the bare version string (e.g. `"=0.5.2"`); quoting is
/// added by this function. For richer dep specs (table form, features),
/// extend this API rather than asking callers to pre-format strings.
///
/// Behaviour:
/// - `[dependencies]` table present, `name` absent → insert `name = "version"`
///   on a new line at the end of the table block, return [`AddDepOutcome::Added`].
/// - `[dependencies]` present and `name` already declared (any RHS shape:
///   plain string, inline table, workspace inheritance) → return
///   [`AddDepOutcome::AlreadyPresent`], file untouched.
/// - `[dependencies]` table absent → append `\n[dependencies]\nname = "version"\n`
///   at end of file, return [`AddDepOutcome::Added`].
///
/// Errors on IO failures or when the file isn't valid UTF-8.
pub fn add_dep(path: &Path, name: &str, version: &str) -> Result<AddDepOutcome, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (new_content, outcome) = insert_dep(&content, name, version)?;
    if outcome == AddDepOutcome::Added {
        fs::write(path, new_content)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(outcome)
}

/// Pure string transformation behind [`add_dep`]. Exposed for testing
/// without filesystem IO.
fn insert_dep(content: &str, name: &str, version: &str) -> Result<(String, AddDepOutcome), String> {
    if name.is_empty() {
        return Err("dep name cannot be empty".into());
    }

    let lines: Vec<&str> = content.lines().collect();

    // Find [dependencies] header and the line index where its block ends
    // (exclusive — first line of the next table, or lines.len()).
    let dep_header = lines.iter().position(|l| l.trim() == "[dependencies]");

    if let Some(header_idx) = dep_header {
        let block_end = lines[header_idx + 1..]
            .iter()
            .position(|l| l.trim_start().starts_with('['))
            .map(|i| header_idx + 1 + i)
            .unwrap_or(lines.len());

        // Already declared?
        for line in &lines[header_idx + 1..block_end] {
            if line_declares_dep(line, name) {
                return Ok((content.to_string(), AddDepOutcome::AlreadyPresent));
            }
        }

        // Find insertion point: last non-blank line within the block,
        // inserting AFTER it. Falls back to right after the header when
        // the block has nothing but blanks.
        let mut insert_at = header_idx + 1;
        for (offset, line) in lines[header_idx + 1..block_end].iter().enumerate() {
            if !line.trim().is_empty() {
                insert_at = header_idx + 1 + offset + 1;
            }
        }

        let new_line = format!("{name} = \"{version}\"");
        let mut out = lines[..insert_at].join("\n");
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&new_line);
        if insert_at < lines.len() {
            out.push('\n');
            out.push_str(&lines[insert_at..].join("\n"));
        }
        if content.ends_with('\n') && !out.ends_with('\n') {
            out.push('\n');
        }
        return Ok((out, AddDepOutcome::Added));
    }

    // No [dependencies] table — append one at EOF.
    let mut out = content.to_string();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    if !out.is_empty() && !out.ends_with("\n\n") {
        out.push('\n');
    }
    out.push_str(&format!("[dependencies]\n{name} = \"{version}\"\n"));
    Ok((out, AddDepOutcome::Added))
}

/// True when `line` declares the dependency `name` (any RHS shape).
/// Matches `name = ...` exactly so neighbours like `flodl-hf` don't
/// false-positive on `flodl`. Handles leading whitespace.
fn line_declares_dep(line: &str, name: &str) -> bool {
    let t = line.trim_start();
    let Some(after_key) = t.strip_prefix(name) else {
        return false;
    };
    let rest = after_key.trim_start();
    rest.starts_with('=')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_to_existing_dependencies_block() {
        let input = "\
[package]
name = \"x\"

[dependencies]
serde = \"1\"
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::Added);
        assert!(out.contains("serde = \"1\""), "preserves existing dep: {out}");
        assert!(
            out.contains("flodl-hf = \"=0.5.2\""),
            "appends new dep: {out}",
        );
        // Inserted within [dependencies] block, not at EOF.
        let header_pos = out.find("[dependencies]").unwrap();
        let new_pos = out.find("flodl-hf").unwrap();
        assert!(new_pos > header_pos);
    }

    #[test]
    fn already_present_plain_version_is_noop() {
        let input = "\
[dependencies]
flodl-hf = \"0.5.0\"
serde = \"1\"
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::AlreadyPresent);
        assert_eq!(out, input);
    }

    #[test]
    fn already_present_inline_table_is_noop() {
        let input = "\
[dependencies]
flodl-hf = { version = \"0.5.0\", features = [\"hub\"] }
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::AlreadyPresent);
        assert_eq!(out, input);
    }

    #[test]
    fn already_present_workspace_inheritance_is_noop() {
        let input = "\
[dependencies]
flodl-hf = { workspace = true }
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::AlreadyPresent);
        assert_eq!(out, input);
    }

    #[test]
    fn missing_table_is_appended_at_eof() {
        let input = "\
[package]
name = \"x\"
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::Added);
        assert!(out.contains("[package]"));
        assert!(out.contains("[dependencies]"));
        assert!(out.contains("flodl-hf = \"=0.5.2\""));
        // [dependencies] comes after [package].
        let pkg = out.find("[package]").unwrap();
        let dep = out.find("[dependencies]").unwrap();
        assert!(dep > pkg);
    }

    #[test]
    fn empty_dependencies_block_inserts_after_header() {
        let input = "\
[package]
name = \"x\"

[dependencies]

[dev-dependencies]
serde = \"1\"
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::Added);
        // New dep lands inside [dependencies], NOT [dev-dependencies].
        let dep = out.find("[dependencies]").unwrap();
        let dev = out.find("[dev-dependencies]").unwrap();
        let new_dep = out.find("flodl-hf").unwrap();
        assert!(
            new_dep > dep && new_dep < dev,
            "new dep must land inside [dependencies] block: {out}",
        );
    }

    #[test]
    fn neighbouring_crate_name_does_not_false_positive() {
        // Adding `flodl` must not see `flodl-hf` as already-present.
        let input = "\
[dependencies]
flodl-hf = \"=0.5.2\"
";
        let (out, outcome) = insert_dep(input, "flodl", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::Added);
        assert!(out.contains("flodl = \"=0.5.2\""));
        assert!(out.contains("flodl-hf = \"=0.5.2\""));
    }

    #[test]
    fn dep_in_other_table_does_not_count_as_present() {
        // `flodl-hf` under [dev-dependencies] should NOT block adding it
        // to [dependencies].
        let input = "\
[dependencies]
serde = \"1\"

[dev-dependencies]
flodl-hf = \"0.5.0\"
";
        let (out, outcome) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert_eq!(outcome, AddDepOutcome::Added);
        // Dep added to [dependencies], not [dev-dependencies] (the dev
        // entry stays untouched).
        let main_block_end = out.find("[dev-dependencies]").unwrap();
        let new_dep = out[..main_block_end].find("flodl-hf").unwrap();
        // And the [dev-dependencies] entry is still there.
        assert!(out[main_block_end..].contains("flodl-hf = \"0.5.0\""));
        let _ = new_dep;
    }

    #[test]
    fn preserves_trailing_newline() {
        let input = "[dependencies]\nserde = \"1\"\n";
        let (out, _) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert!(out.ends_with('\n'), "trailing newline preserved: {out:?}");
    }

    #[test]
    fn preserves_no_trailing_newline() {
        let input = "[dependencies]\nserde = \"1\"";
        let (out, _) = insert_dep(input, "flodl-hf", "=0.5.2").unwrap();
        assert!(!out.ends_with("\n\n"));
    }

    #[test]
    fn empty_name_errors() {
        let err = insert_dep("[dependencies]\n", "", "=0.5.2").unwrap_err();
        assert!(err.contains("name cannot be empty"));
    }
}
