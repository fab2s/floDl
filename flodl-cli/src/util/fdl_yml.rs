//! Minimal text-based fdl.yml editor.
//!
//! Same policy as [`super::cargo_toml`]: append-only, format-preserving,
//! no external yaml-edit crate. Scope is appending a top-level command
//! entry under `commands:` if the entry isn't already declared.
//!
//! By fdl.yml convention, a command with neither `run:` nor `path:` and
//! no preset fields falls through to a Path command with the default
//! `./<name>/` location, so the appended entry needs no explicit
//! `path:` to make `fdl <name> <subcmd>` route into `./<name>/fdl.yml`.

use std::fs;
use std::path::Path;

/// Result of an [`add_command`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddCommandOutcome {
    /// The command entry was appended.
    Added,
    /// `name` was already declared under `commands:`; file untouched.
    AlreadyPresent,
}

/// Append a top-level command entry under `commands:` in the fdl.yml at
/// `path` if the entry isn't already declared.
///
/// `description` is written as a `description:` subfield. Pass an empty
/// string to omit it (the entry then has nothing under it, falling back
/// to the convention-default `path: ./<name>/`).
///
/// Behaviour:
/// - `commands:` table present, `name` absent → append the entry at end
///   of the commands block, [`AddCommandOutcome::Added`].
/// - `commands:` present and `name` already declared → file untouched,
///   [`AddCommandOutcome::AlreadyPresent`].
/// - `commands:` absent → append `\ncommands:\n  name:\n    description: ...\n`
///   at end of file, [`AddCommandOutcome::Added`].
pub fn add_command(
    path: &Path,
    name: &str,
    description: &str,
) -> Result<AddCommandOutcome, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (new_content, outcome) = insert_command(&content, name, description)?;
    if outcome == AddCommandOutcome::Added {
        fs::write(path, new_content)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(outcome)
}

fn insert_command(
    content: &str,
    name: &str,
    description: &str,
) -> Result<(String, AddCommandOutcome), String> {
    if name.is_empty() {
        return Err("command name cannot be empty".into());
    }

    let lines: Vec<&str> = content.lines().collect();

    // Find top-level `commands:` (indent 0).
    let header_idx = lines
        .iter()
        .position(|l| l.trim_end() == "commands:" && !l.starts_with([' ', '\t']));

    let Some(header_idx) = header_idx else {
        // No commands: table — append a fresh one at EOF.
        let mut out = content.to_string();
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        if !out.is_empty() && !out.ends_with("\n\n") {
            out.push('\n');
        }
        out.push_str("commands:\n");
        out.push_str(&render_entry("  ", name, description));
        return Ok((out, AddCommandOutcome::Added));
    };

    // Block ends at the first line at indent 0 (excluding blanks).
    let block_end = lines[header_idx + 1..]
        .iter()
        .position(|l| !l.is_empty() && !l.starts_with([' ', '\t']))
        .map(|i| header_idx + 1 + i)
        .unwrap_or(lines.len());

    // Detect child indent from the first non-blank child; default to two
    // spaces when the block is empty (matches scaffold convention).
    let child_indent = lines[header_idx + 1..block_end]
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| {
            let n = l.chars().take_while(|c| *c == ' ').count();
            " ".repeat(n)
        })
        .unwrap_or_else(|| "  ".to_string());

    // Already declared?
    let key_token = format!("{name}:");
    for line in &lines[header_idx + 1..block_end] {
        if !line.starts_with(&child_indent) {
            continue;
        }
        let after_indent = &line[child_indent.len()..];
        // Must be a sibling key (no further leading spaces) and match
        // `name:` or `name :` exactly.
        if after_indent.starts_with(' ') {
            continue;
        }
        let trimmed = after_indent.trim_start();
        if trimmed == key_token
            || trimmed.starts_with(&format!("{key_token} "))
            || trimmed.starts_with(&format!("{name} :"))
        {
            return Ok((content.to_string(), AddCommandOutcome::AlreadyPresent));
        }
    }

    // Insert AFTER the last non-blank line in the block.
    let mut insert_at = header_idx + 1;
    for (offset, line) in lines[header_idx + 1..block_end].iter().enumerate() {
        if !line.trim().is_empty() {
            insert_at = header_idx + 1 + offset + 1;
        }
    }

    let entry = render_entry(&child_indent, name, description);

    let mut out = lines[..insert_at].join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    // Blank line before the entry when the previous content already
    // had a non-blank line (visual separator between sibling commands).
    // Skip when the immediately previous line is already blank.
    let prev_blank = insert_at == header_idx + 1
        || lines
            .get(insert_at - 1)
            .is_some_and(|l| l.trim().is_empty());
    if !prev_blank {
        out.push('\n');
    }
    out.push_str(&entry);
    if insert_at < lines.len() {
        out.push_str(&lines[insert_at..].join("\n"));
        if content.ends_with('\n') {
            out.push('\n');
        }
    }
    Ok((out, AddCommandOutcome::Added))
}

fn render_entry(child_indent: &str, name: &str, description: &str) -> String {
    let mut out = format!("{child_indent}{name}:\n");
    if !description.is_empty() {
        out.push_str(&format!(
            "{child_indent}{child_indent}description: {description}\n"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_to_existing_commands_block() {
        let input = "\
description: my project

commands:
  build:
    run: cargo build
    docker: dev
";
        let (out, outcome) = insert_command(input, "flodl-hf", "HF integration").unwrap();
        assert_eq!(outcome, AddCommandOutcome::Added);
        assert!(out.contains("build:"), "preserves existing: {out}");
        assert!(out.contains("flodl-hf:"), "appends: {out}");
        assert!(out.contains("description: HF integration"));
        // New entry comes after `build:`.
        let build = out.find("build:").unwrap();
        let new = out.find("flodl-hf:").unwrap();
        assert!(new > build);
    }

    #[test]
    fn already_present_is_noop() {
        let input = "\
commands:
  flodl-hf:
    description: existing entry
  build:
    run: cargo build
";
        let (out, outcome) = insert_command(input, "flodl-hf", "new desc").unwrap();
        assert_eq!(outcome, AddCommandOutcome::AlreadyPresent);
        assert_eq!(out, input);
    }

    #[test]
    fn missing_commands_block_appends_at_eof() {
        let input = "description: my project\n";
        let (out, outcome) = insert_command(input, "flodl-hf", "HF").unwrap();
        assert_eq!(outcome, AddCommandOutcome::Added);
        assert!(out.contains("commands:"));
        assert!(out.contains("  flodl-hf:"));
        assert!(out.contains("    description: HF"));
    }

    #[test]
    fn empty_commands_block_inserts_first_child() {
        let input = "commands:\n";
        let (out, outcome) = insert_command(input, "flodl-hf", "HF").unwrap();
        assert_eq!(outcome, AddCommandOutcome::Added);
        // Default 2-space indent kicks in.
        assert!(out.contains("  flodl-hf:"));
        assert!(out.contains("    description: HF"));
    }

    #[test]
    fn detects_existing_indent_and_matches_it() {
        // Existing block uses 4-space indent — new entry must follow.
        let input = "\
commands:
    build:
        run: cargo build
";
        let (out, _) = insert_command(input, "flodl-hf", "HF").unwrap();
        assert!(out.contains("    flodl-hf:"));
        assert!(out.contains("        description: HF"));
    }

    #[test]
    fn empty_description_omits_subfield() {
        let input = "commands:\n  build:\n    run: cargo build\n";
        let (out, _) = insert_command(input, "flodl-hf", "").unwrap();
        assert!(out.contains("  flodl-hf:"));
        assert!(
            !out.contains("description: \n"),
            "no empty description: {out}"
        );
    }

    #[test]
    fn neighbouring_command_name_does_not_false_positive() {
        // `flodl-hf` and `flodl` are distinct keys; presence of one must
        // not block adding the other.
        let input = "commands:\n  flodl-hf:\n    description: existing\n";
        let (out, outcome) = insert_command(input, "flodl", "new").unwrap();
        assert_eq!(outcome, AddCommandOutcome::Added);
        assert!(out.contains("flodl-hf:"));
        assert!(out.contains("flodl:"));
    }

    #[test]
    fn preserves_trailing_content_after_block() {
        // commands: is followed by another top-level key — new entry
        // must not bleed into it.
        let input = "\
commands:
  build:
    run: cargo build

other_top_level: foo
";
        let (out, _) = insert_command(input, "flodl-hf", "HF").unwrap();
        assert!(
            out.contains("other_top_level: foo"),
            "trailing key preserved: {out}"
        );
        // `flodl-hf:` lands BEFORE `other_top_level:` (still inside commands block).
        let new = out.find("flodl-hf:").unwrap();
        let other = out.find("other_top_level:").unwrap();
        assert!(new < other);
    }

    #[test]
    fn empty_name_errors() {
        let err = insert_command("commands:\n", "", "x").unwrap_err();
        assert!(err.contains("name cannot be empty"));
    }
}
