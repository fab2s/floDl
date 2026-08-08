//! `fdl add flodl-hf` -- two modes for wiring flodl-hf into a project.
//!
//! - **playground**: drops `./flodl-hf/` as a standalone cargo crate
//!   with a one-file `AutoModel` example, plus a `flodl-hf:` entry in
//!   the root `fdl.yml` so `fdl flodl-hf <cmd>` routes into the
//!   playground from the project root. Try-it-out path; the user's own
//!   `Cargo.toml` is untouched.
//! - **install**: appends `flodl-hf = "=X.Y.Z"` to root
//!   `Cargo.toml` `[dependencies]` (default features). Wires the crate
//!   into the user's own code; nothing else mutated.
//!
//! Modes are combinable on the same invocation. Without flags, an
//! interactive prompt asks; non-tty stdin errors loudly.
//!
//! Targets accepted: `flodl-hf` and its alias `hf`. Other targets
//! surface a loud error listing the supported set.

use std::fs;
use std::path::{Path, PathBuf};

use crate::util::{cargo_toml as cargo_edit, fdl_yml as yml_edit, prompt};

/// Scaffold templates baked into the binary at compile time. Live
/// under `flodl-cli/src/scaffold/` so they travel inside the
/// `flodl-cli` crate tarball on `cargo publish`.
// `.in` suffix avoids cargo treating this as a nested package manifest
// during `cargo package`; it is written out as `Cargo.toml` when the
// scaffold is generated.
const TEMPLATE_CARGO_TOML: &str = include_str!("scaffold/Cargo.toml.in");
const TEMPLATE_MAIN_RS: &str = include_str!("scaffold/src/main.rs");
const TEMPLATE_FDL_YML: &str = include_str!("scaffold/fdl.yml.example");
const TEMPLATE_README: &str = include_str!("scaffold/README.md");
const TEMPLATE_GITIGNORE: &str = include_str!("scaffold/.gitignore");

/// Description written into the root `fdl.yml` `flodl-hf:` entry.
const FDL_YML_HF_DESCRIPTION: &str = "HuggingFace integration (BERT, RoBERTa, DistilBERT, ...)";

pub fn run(target: Option<&str>, playground: bool, install: bool) -> Result<(), String> {
    let target = target.ok_or(
        "usage: fdl add <target> [--playground] [--install]\n\n\
         Supported targets:\n    \
         flodl-hf    HuggingFace integration (pre-built BERT / RoBERTa / DistilBERT, Hub loader, tokenizer)",
    )?;
    match target {
        "flodl-hf" | "hf" => {}
        other => {
            return Err(format!(
                "unknown target: {other:?}\n\n\
                 Supported targets:\n    \
                 flodl-hf    HuggingFace integration\n\n\
                 (More targets land as the flodl ecosystem grows.)",
            ));
        }
    }

    let cwd = std::env::current_dir().map_err(|e| format!("cannot read current directory: {e}"))?;

    // No flag: interactive prompt (or loud error on non-tty).
    let (do_playground, do_install) = if !playground && !install {
        resolve_interactive()?
    } else {
        (playground, install)
    };

    if do_install {
        install_flodl_hf_at(&cwd)?;
    }
    if do_playground {
        add_flodl_hf_at(&cwd)?;
    }
    Ok(())
}

/// Ask the user which mode(s) to run. Errors when no controlling
/// terminal is available — in CI / scripted contexts the caller must
/// pass `--playground` and/or `--install` explicitly.
fn resolve_interactive() -> Result<(bool, bool), String> {
    if !has_tty() {
        return Err(
            "fdl add flodl-hf needs an interactive terminal to prompt.\n\
             Pass --playground (sandbox at ./flodl-hf/) or --install \
             (add to Cargo.toml), or both."
                .into(),
        );
    }

    println!("Add flodl-hf to your project?");
    println!();
    let choice = prompt::ask_choice(
        "Choose",
        &[
            "playground   sandbox at ./flodl-hf/ (try it without touching your project)",
            "install      add flodl-hf to your root Cargo.toml as a dependency",
            "both         playground + install (try it, and wire it in)",
            "cancel",
        ],
        1,
    );
    println!();

    match choice {
        1 => Ok((true, false)),
        2 => Ok((false, true)),
        3 => Ok((true, true)),
        _ => Err("cancelled.".into()),
    }
}

/// Detect a usable controlling terminal — see [`prompt::has_tty`].
fn has_tty() -> bool {
    prompt::has_tty()
}

/// Append `flodl-hf` to the root `Cargo.toml` `[dependencies]` table.
/// Idempotent — already-present is a friendly no-op.
pub fn install_flodl_hf_at(cwd: &Path) -> Result<(), String> {
    let cargo_toml = cwd.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(format!(
            "no Cargo.toml in {}.\n\n\
             fdl add flodl-hf --install must run from a flodl project root.\n\
             Start with `fdl init <name>` if you don't have one yet.",
            cwd.display(),
        ));
    }

    let flodl_version = detect_flodl_version(&cargo_toml)?;
    let version_spec = format!("={flodl_version}");
    let outcome = cargo_edit::add_dep(&cargo_toml, "flodl-hf", &version_spec)?;

    match outcome {
        cargo_edit::AddDepOutcome::AlreadyPresent => {
            println!("flodl-hf is already declared in {}.", cargo_toml.display());
            println!("Edit the entry directly to change version or features.");
        }
        cargo_edit::AddDepOutcome::Added => {
            println!();
            println!(
                "Added flodl-hf = \"={flodl_version}\" to {} with default features (hub, tokenizer).",
                cargo_toml.display(),
            );
            println!();
            println!("Default features include the HuggingFace Hub loader and tokenizer.");
            println!("To switch to offline / vision-only flavors, edit the entry manually:");
            println!(
                "  flodl-hf = {{ version = \"={flodl_version}\", default-features = false, features = [...] }}"
            );
            println!();
            println!("Run `fdl build` (or `cargo build`) to pull and compile the new dependency.");
        }
    }
    Ok(())
}

/// Scaffold `flodl-hf/` playground under `base` and link it into the
/// root `fdl.yml`. Entry point for `fdl add flodl-hf --playground`
/// (with `base = cwd`) and `fdl init --with-hf` follow-up (with
/// `base = the freshly-scaffolded project dir`). The base dir must
/// contain a `Cargo.toml` with a pinnable `flodl` dependency.
pub fn add_flodl_hf_at(cwd: &Path) -> Result<(), String> {
    // Must be run from a flodl project root (Cargo.toml with flodl dep).
    let cargo_toml = cwd.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(format!(
            "no Cargo.toml in {}.\n\n\
             fdl add flodl-hf must run from a flodl project root.\n\
             Start with `fdl init <name>` if you don't have one yet.",
            cwd.display(),
        ));
    }

    // flodl-hf makes no sense without a functioning flodl project: every
    // runnable command assumes a Cargo.toml + fdl.yml pair are already
    // present. Enforce that invariant loudly so the user isn't left
    // with a dead sub-crate.
    if !has_fdl_config(cwd) {
        return Err(format!(
            "no fdl.yml (nor fdl.yml.example) in {}.\n\n\
             fdl add flodl-hf expects an initialised flodl project: \
             Docker or native mode already chosen, fdl.yml present. \
             Run `fdl init <name>` first, or cd into an existing flodl project.",
            cwd.display(),
        ));
    }

    let flodl_version = detect_flodl_version(&cargo_toml)?;
    let mode = detect_project_mode(cwd);

    // Refuse to overwrite an existing flodl-hf/ dir.
    let dest = cwd.join("flodl-hf");
    if dest.exists() {
        return Err(format!(
            "{} already exists.\n\n\
             Remove it first, or keep it. `fdl add flodl-hf` does not overwrite.",
            dest.display(),
        ));
    }

    // Scaffold.
    fs::create_dir_all(dest.join("src"))
        .map_err(|e| format!("cannot create {}: {e}", dest.join("src").display()))?;

    write_file(
        &dest.join("Cargo.toml"),
        &substitute_version(TEMPLATE_CARGO_TOML, &flodl_version),
    )?;
    write_file(&dest.join("src/main.rs"), TEMPLATE_MAIN_RS)?;
    let fdl_yml = render_fdl_yml(TEMPLATE_FDL_YML, mode);
    write_file(&dest.join("fdl.yml.example"), &fdl_yml)?;
    write_file(&dest.join("fdl.yml"), &fdl_yml)?;
    write_file(
        &dest.join("README.md"),
        &substitute_version(TEMPLATE_README, &flodl_version),
    )?;
    write_file(&dest.join(".gitignore"), TEMPLATE_GITIGNORE)?;

    // Link `flodl-hf:` into root fdl.yml so `fdl flodl-hf <cmd>` works
    // from project root. Idempotent — re-runs after a manual delete of
    // the playground dir do the right thing.
    link_into_root_fdl_yml(cwd)?;

    print_next_steps(&flodl_version, mode);
    Ok(())
}

/// Append a `flodl-hf:` entry under `commands:` in the root fdl.yml
/// (and fdl.yml.example when present) so `fdl flodl-hf` routes into
/// `./flodl-hf/fdl.yml` via the convention-default Path command.
fn link_into_root_fdl_yml(cwd: &Path) -> Result<(), String> {
    for filename in ["fdl.yml", "fdl.yml.example"] {
        let path = cwd.join(filename);
        if !path.exists() {
            continue;
        }
        yml_edit::add_command(&path, "flodl-hf", FDL_YML_HF_DESCRIPTION)?;
    }
    Ok(())
}

/// Host-project execution mode, inferred from file presence.
///
/// `fdl init` writes `docker-compose.yml` for its Mounted and Docker
/// modes, and omits it for Native. Scaffolded commands follow the
/// same convention: Docker modes dispatch to the `dev` service,
/// Native runs directly on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectMode {
    Docker,
    Native,
}

fn has_fdl_config(cwd: &Path) -> bool {
    cwd.join("fdl.yml").exists() || cwd.join("fdl.yml.example").exists()
}

fn detect_project_mode(cwd: &Path) -> ProjectMode {
    if cwd.join("docker-compose.yml").exists() {
        ProjectMode::Docker
    } else {
        ProjectMode::Native
    }
}

/// In Native mode, strip the `    docker: dev` lines from the scaffold
/// `fdl.yml` so cargo commands run directly on the host instead of
/// trying to dispatch into a non-existent Docker service. Matches the
/// indentation produced by the template exactly; anything else is left
/// alone.
fn render_fdl_yml(template: &str, mode: ProjectMode) -> String {
    match mode {
        ProjectMode::Docker => template.to_string(),
        ProjectMode::Native => {
            template
                .lines()
                .filter(|l| l.trim() != "docker: dev")
                .collect::<Vec<&str>>()
                .join("\n")
                + "\n"
        }
    }
}

/// Parse `Cargo.toml` for the `flodl` dependency version.
///
/// Recognises three forms:
/// - `flodl = "0.5.1"` — plain version string
/// - `flodl = { version = "0.5.1", ... }` — table form
/// - `flodl = { workspace = true }` — workspace inheritance (reads from
///   the workspace root's `Cargo.toml`)
///
/// Errors on: no flodl dep found, git-only dep (no pinnable version),
/// or path-only dep outside this repo (no version to pin against).
fn detect_flodl_version(cargo_toml: &Path) -> Result<String, String> {
    let content = fs::read_to_string(cargo_toml)
        .map_err(|e| format!("cannot read {}: {e}", cargo_toml.display()))?;

    if let Some(v) = parse_flodl_dep(&content)? {
        return Ok(v);
    }

    // Workspace inheritance: climb to find the workspace root.
    if let Some(ws_root) = find_workspace_root(cargo_toml) {
        let ws_content = fs::read_to_string(&ws_root)
            .map_err(|e| format!("cannot read workspace {}: {e}", ws_root.display()))?;
        if let Some(v) = parse_flodl_dep(&ws_content)? {
            return Ok(v);
        }
    }

    Err(format!(
        "no flodl dependency found in {}.\n\n\
         fdl add flodl-hf needs to pin flodl-hf to the same version as \
         flodl. Add `flodl = \"X.Y.Z\"` to [dependencies] first, or run \
         `fdl init <name>` to scaffold a flodl project.",
        cargo_toml.display(),
    ))
}

/// Extract the flodl version from a Cargo.toml's textual content.
///
/// Returns `Ok(Some(version))` on a pinnable version, `Ok(None)` when
/// no flodl dep is present, and `Err(...)` when the dep exists but is
/// git-only / path-only (no version to pin against).
fn parse_flodl_dep(content: &str) -> Result<Option<String>, String> {
    let lines: Vec<&str> = content.lines().collect();

    // Find a line that declares `flodl = ...` under a [dependencies]
    // or [workspace.dependencies] table. We accept any form whose LHS
    // matches `flodl`; the inline value on the RHS tells us the shape.
    let mut in_dep_table = false;
    for line in &lines {
        let t = line.trim();
        if t.starts_with('[') {
            // Only consider tables that declare dependencies.
            in_dep_table = matches!(
                t,
                "[dependencies]" | "[workspace.dependencies]" | "[dev-dependencies]",
            );
            continue;
        }
        if !in_dep_table {
            continue;
        }
        // Match `flodl = ...` exactly (not flodl-hf, flodl-sys, ...).
        let after_key = match t.strip_prefix("flodl") {
            Some(rest) => rest.trim_start(),
            None => continue,
        };
        let Some(rhs) = after_key.strip_prefix('=') else {
            continue;
        };
        let rhs = rhs.trim();

        // Three RHS shapes: "X.Y.Z", { version = "...", ... }, { workspace = true }
        if let Some(v) = rhs.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
            return Ok(Some(v.to_string()));
        }
        if let Some(v) = extract_version_from_table(rhs) {
            return Ok(Some(v));
        }
        if rhs.contains("workspace") && rhs.contains("true") {
            // Caller resolves workspace inheritance.
            return Ok(None);
        }
        if rhs.contains("git =") || rhs.contains("git=") {
            return Err("flodl is declared as a git dependency. \
                 fdl add flodl-hf needs a pinnable crates.io version. \
                 Switch to `flodl = \"X.Y.Z\"` first."
                .into());
        }
        if rhs.contains("path =") || rhs.contains("path=") {
            // Path-only dep: read version from the referenced Cargo.toml.
            // For MVP, error with guidance.
            return Err("flodl is declared as a path dependency only. \
                 Add an explicit `version = \"X.Y.Z\"` so fdl add can \
                 pin the matching flodl-hf release."
                .into());
        }
    }
    Ok(None)
}

/// Extract `version = "X.Y.Z"` from an inline table like
/// `{ version = "0.5.1", features = [...] }`. Returns `None` when the
/// string doesn't look like a table or carries no `version` key.
fn extract_version_from_table(rhs: &str) -> Option<String> {
    let rhs = rhs.strip_prefix('{')?.strip_suffix('}')?;
    for part in rhs.split(',') {
        let part = part.trim();
        let Some(after) = part.strip_prefix("version") else {
            continue;
        };
        let after = after.trim_start();
        let Some(after) = after.strip_prefix('=') else {
            continue;
        };
        let after = after.trim_start();
        let Some(v) = after.strip_prefix('"').and_then(|r| r.strip_suffix('"')) else {
            continue;
        };
        return Some(v.to_string());
    }
    None
}

/// Climb the directory tree looking for a Cargo.toml with a
/// `[workspace]` table. Returns the path when found, else None.
fn find_workspace_root(from: &Path) -> Option<PathBuf> {
    let mut dir = from.parent()?.parent()?.to_path_buf();
    loop {
        let candidate = dir.join("Cargo.toml");
        if candidate.exists()
            && let Ok(content) = fs::read_to_string(&candidate)
            && content.lines().any(|l| l.trim() == "[workspace]")
        {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn substitute_version(template: &str, version: &str) -> String {
    template.replace("{{FLODL_VERSION}}", version)
}

fn write_file(path: &Path, content: &str) -> Result<(), String> {
    fs::write(path, content).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

fn print_next_steps(version: &str, mode: ProjectMode) {
    println!();
    println!(
        "Scaffolded flodl-hf/ playground (flodl {version}, {} mode).",
        match mode {
            ProjectMode::Docker => "Docker",
            ProjectMode::Native => "native",
        },
    );
    println!();
    println!("Next steps:");
    println!("  fdl flodl-hf classify                 # default RoBERTa sentiment checkpoint");
    println!("  fdl flodl-hf classify -- bert-base-uncased   # any other BERT-family repo id");
    println!();
    println!("(Or `cd flodl-hf` and run `fdl classify` directly.)");
    println!();
    println!("See flodl-hf/README.md for feature flavors (offline / vision-only),");
    println!("`.bin` to safetensors conversion for older checkpoints, and how to wire");
    println!("flodl-hf into your main crate when you're ready (`fdl add flodl-hf --install`).");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_plain_version_string() {
        let c = r#"
[dependencies]
flodl = "0.6.0"
other = "1.0"
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), Some("0.6.0".into()));
    }

    #[test]
    fn parse_table_version() {
        let c = r#"
[dependencies]
flodl = { version = "0.5.1", features = ["cuda"] }
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), Some("0.5.1".into()));
    }

    #[test]
    fn parse_workspace_inheritance_returns_none() {
        let c = r#"
[dependencies]
flodl = { workspace = true }
"#;
        // Workspace inheritance returns None; caller climbs to workspace root.
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn parse_git_dep_errors() {
        let c = r#"
[dependencies]
flodl = { git = "https://github.com/flodl-labs/flodl" }
"#;
        let err = parse_flodl_dep(c).unwrap_err();
        assert!(err.contains("git dependency"), "got: {err}");
    }

    #[test]
    fn parse_no_flodl_returns_none() {
        let c = r#"
[dependencies]
other = "1.0"
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn parse_ignores_flodl_hf_and_flodl_sys() {
        // `flodl = ...` must match exactly — neighbouring crate names
        // (flodl-hf, flodl-sys) must not false-positive.
        let c = r#"
[dependencies]
flodl-hf = "0.6.0"
flodl-sys = "0.6.0"
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn parse_ignores_non_dep_tables() {
        let c = r#"
[package]
flodl = "0.6.0"   # not actually a dep; this is bogus but must not match
"#;
        assert_eq!(parse_flodl_dep(c).unwrap(), None);
    }

    #[test]
    fn substitute_version_replaces_all_occurrences() {
        let t = "flodl = \"={{FLODL_VERSION}}\"\nflodl-hf = \"={{FLODL_VERSION}}\"";
        let out = substitute_version(t, "0.6.0");
        assert_eq!(out, "flodl = \"=0.6.0\"\nflodl-hf = \"=0.6.0\"");
    }

    #[test]
    fn render_fdl_yml_docker_preserves_docker_lines() {
        let t = "commands:\n  classify:\n    run: cargo run --release\n    docker: dev\n";
        assert_eq!(render_fdl_yml(t, ProjectMode::Docker), t);
    }

    #[test]
    fn render_fdl_yml_native_strips_docker_lines() {
        let t = "commands:\n  classify:\n    run: cargo run --release\n    docker: dev\n  check:\n    run: cargo check\n    docker: dev\n";
        let out = render_fdl_yml(t, ProjectMode::Native);
        assert!(
            !out.contains("docker: dev"),
            "native output must not contain docker: dev lines: {out}"
        );
        // Non-docker lines stay in place — `cargo run --release` and
        // `cargo check` must both survive.
        assert!(out.contains("cargo run --release"));
        assert!(out.contains("cargo check"));
    }

    #[test]
    fn render_fdl_yml_native_only_strips_exact_docker_line() {
        // Indentation-sensitive: lines like `    docker: hf-parity` or
        // `description: docker: dev stuff` must NOT be stripped.
        let t = "\
commands:
  classify:
    run: cargo run
    docker: dev
  other:
    description: docker: dev isn't a literal directive here
    docker: hf-parity
";
        let out = render_fdl_yml(t, ProjectMode::Native);
        assert!(
            !out.contains("    docker: dev\n"),
            "exact match stripped: {out}"
        );
        assert!(out.contains("hf-parity"), "other services preserved: {out}");
        assert!(
            out.contains("docker: dev isn't a literal"),
            "description text preserved: {out}",
        );
    }

    /// Build a minimal flodl project tree under a unique temp dir for
    /// integration tests. Returns the project root.
    fn temp_project(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("fdl-add-test-{pid}-{n}-{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\nflodl = \"0.5.2\"\n",
        )
        .unwrap();
        fs::write(
            dir.join("fdl.yml"),
            "description: test project\n\ncommands:\n  build:\n    run: cargo build\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn install_appends_dep_and_is_idempotent() {
        let dir = temp_project("install-idem");
        install_flodl_hf_at(&dir).unwrap();
        let toml = fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        assert!(
            toml.contains("flodl-hf = \"=0.5.2\""),
            "first install: {toml}"
        );

        // Re-run: no-op, file unchanged.
        install_flodl_hf_at(&dir).unwrap();
        let toml2 = fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        assert_eq!(toml, toml2, "install is idempotent");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn install_errors_without_cargo_toml() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(9000);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("fdl-add-test-no-cargo-{pid}-{n}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let err = install_flodl_hf_at(&dir).unwrap_err();
        assert!(err.contains("no Cargo.toml"), "got: {err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn playground_links_root_fdl_yml() {
        let dir = temp_project("playground-link");
        add_flodl_hf_at(&dir).unwrap();
        let yml = fs::read_to_string(dir.join("fdl.yml")).unwrap();
        assert!(yml.contains("flodl-hf:"), "linked into root fdl.yml: {yml}");
        // Existing `build:` entry preserved.
        assert!(yml.contains("build:"));
        // Playground crate also exists.
        assert!(dir.join("flodl-hf/Cargo.toml").exists());
        assert!(dir.join("flodl-hf/fdl.yml").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
