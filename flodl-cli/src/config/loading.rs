//! Manifest discovery + load helpers: walks up from CWD to find the
//! project manifest, loads YAML/JSON, resolves env-overlay layers,
//! cluster-dispatch chain resolution.

use std::path::{Path, PathBuf};


use super::schema::ProjectConfig;

/// Resolve whether a leaf command should fan out across the cluster, given
/// the chain of [`super::schema::CommandSpec::cluster`] directives along its command path,
/// ordered root → leaf.
///
/// Walks the chain from leaf back to root; the first `Some` value wins.
/// This makes deeper overrides take precedence:
///
/// - root sets `cluster: true`, leaf unset → `true` (inherits)
/// - root sets `cluster: true`, leaf sets `cluster: false` → `false` (override)
/// - root unset, leaf sets `cluster: true` → `true`
/// - all unset → `false` (no clustering by default)
///
/// Intended caller pattern (from a future dispatch in `run.rs`):
///
/// ```ignore
/// let chain: Vec<Option<bool>> = ancestors.iter()
///     .map(|spec| spec.cluster)
///     .collect();
/// if cluster_dispatch_enabled(&project, &chain) {
///     // fan out across project.cluster.workers
/// }
/// ```
///
/// This is a pure function on the chain; cluster-block presence is checked
/// separately by [`cluster_dispatch_enabled`].
pub fn resolve_cluster_dispatch(chain: &[Option<bool>]) -> bool {
    chain.iter().rev().find_map(|x| *x).unwrap_or(false)
}

/// Whether the leaf command's effective `cluster:` value resolves to `true`
/// AND a `cluster:` topology is declared at the project root.
///
/// Without a project-root `cluster:` block, multi-host dispatch is
/// unavailable regardless of any per-command directives along the chain.
pub fn cluster_dispatch_enabled(project: &ProjectConfig, chain: &[Option<bool>]) -> bool {
    project.cluster.is_some() && resolve_cluster_dispatch(chain)
}


// ── Config discovery ────────────────────────────────────────────────────

pub(super) const CONFIG_NAMES: &[&str] = &["fdl.yaml", "fdl.yml", "fdl.json"];
pub(super) const EXAMPLE_SUFFIXES: &[&str] = &[".example", ".dist"];

/// Walk up from `start` looking for fdl.yaml.
///
/// If only an `.example` (or `.dist`) variant exists, offers to copy it
/// to the real config path. This lets the repo commit `fdl.yaml.example`
/// while `.gitignore`-ing `fdl.yaml` so users can customize locally.
pub fn find_config(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        // First pass: look for the real config.
        for name in CONFIG_NAMES {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        // Second pass: look for .example/.dist variants.
        for name in CONFIG_NAMES {
            for suffix in EXAMPLE_SUFFIXES {
                let example = dir.join(format!("{name}{suffix}"));
                if example.is_file() {
                    let target = dir.join(name);
                    if try_copy_example(&example, &target) {
                        return Some(target);
                    }
                    // User declined: use the example directly.
                    return Some(example);
                }
            }
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Prompt the user to copy an example config to the real path.
/// Returns true if the copy succeeded.
pub(super) fn try_copy_example(example: &Path, target: &Path) -> bool {
    let example_name = example.file_name().unwrap_or_default().to_string_lossy();
    let target_name = target.file_name().unwrap_or_default().to_string_lossy();
    eprintln!(
        "fdl: found {example_name} but no {target_name}. \
         Copy it to create your local config? [Y/n] "
    );
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let answer = input.trim().to_lowercase();
    if answer.is_empty() || answer == "y" || answer == "yes" {
        match std::fs::copy(example, target) {
            Ok(_) => {
                eprintln!("fdl: created {target_name} (edit to customize)");
                true
            }
            Err(e) => {
                eprintln!("fdl: failed to copy: {e}");
                false
            }
        }
    } else {
        false
    }
}

/// Load a project config from a specific path.
pub fn load_project(path: &Path) -> Result<ProjectConfig, String> {
    load_project_with_env(path, None)
}

/// Load a project config with an optional environment overlay.
///
/// When `env` is `Some`, looks for a sibling `fdl.<env>.{yml,yaml,json}` next
/// to `base_path` and deep-merges it over the base before deserialization.
/// Missing overlay files are a hard error — the user asked for this env, so
/// silently ignoring it would be worse than a clear message.
pub fn load_project_with_env(
    base_path: &Path,
    env: Option<&str>,
) -> Result<ProjectConfig, String> {
    let layers = resolve_config_layers(base_path, env)?;
    let merged = crate::overlay::merge_layers(
        layers.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>(),
    );
    // Re-serialize so `from_str`'s parser tracks line/col through deserialize.
    // `from_value` discards positional info, leaving errors location-less.
    let merged_str = serde_yaml::to_string(&merged).map_err(|e| {
        format!(
            "{}: failed to re-serialize merged YAML for diagnostics: {e}",
            base_path.display()
        )
    })?;
    serde_yaml::from_str::<ProjectConfig>(&merged_str).map_err(|e| {
        let names: Vec<String> = layers
            .iter()
            .map(|(p, _)| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect();
        let env_hint = env
            .map(|n| format!(" {n}"))
            .unwrap_or_default();
        let (loc_str, context) = match e.location() {
            Some(loc) => (
                format!(" at merged-view line {}, col {}", loc.line(), loc.column()),
                extract_context(&merged_str, loc.line()),
            ),
            None => (String::new(), String::new()),
        };
        format!(
            "{} (layers: {}){}: {}{}\n  inspect merged view: fdl{} config show",
            base_path.display(),
            names.join(" + "),
            loc_str,
            e,
            context,
            env_hint
        )
    })
}

/// Extract 3 lines of context around `line_no` (1-based) from the merged
/// YAML string, formatted as numbered indented lines for inclusion in error
/// messages. Returns empty if line_no is out of range.
fn extract_context(text: &str, line_no: usize) -> String {
    if line_no == 0 {
        return String::new();
    }
    let lines: Vec<&str> = text.lines().collect();
    if line_no > lines.len() {
        return String::new();
    }
    let start = line_no.saturating_sub(2).max(1);
    let end = (line_no + 1).min(lines.len());
    let mut out = String::from("\n");
    for n in start..=end {
        let marker = if n == line_no { ">>" } else { "  " };
        out.push_str(&format!("  {marker} {n:>4}: {}\n", lines[n - 1]));
    }
    out
}

/// Load the raw merged [`serde_yaml::Value`] for a config + optional env
/// overlay. Exposed so callers like `fdl config show` can inspect the
/// resolved view before it is deserialized into a strongly-typed struct.
pub fn load_merged_value(
    base_path: &Path,
    env: Option<&str>,
) -> Result<serde_yaml::Value, String> {
    let layers = resolve_config_layers(base_path, env)?;
    Ok(crate::overlay::merge_layers(
        layers.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
    ))
}

/// Resolve every layer contributing to a config, in merge order, with
/// `inherit-from:` chains expanded. Paired with the base file + optional
/// env overlay, the result is `[chain(base)..., chain(env_overlay)...]`
/// de-duplicated by canonical path (kept-first).
///
/// Used by `fdl config show` for per-leaf source annotation, and
/// internally by [`load_merged_value`] / [`super::command::load_command_with_env`] so
/// every consumer picks up `inherit-from:` uniformly.
pub fn resolve_config_layers(
    base_path: &Path,
    env: Option<&str>,
) -> Result<Vec<(PathBuf, serde_yaml::Value)>, String> {
    let mut layers = crate::overlay::resolve_chain(base_path)?;
    if let Some(name) = env {
        match crate::overlay::find_env_file(base_path, name) {
            Some(p) => {
                let env_chain = crate::overlay::resolve_chain(&p)?;
                layers.extend(env_chain);
            }
            None => {
                return Err(format!(
                    "environment `{name}` not found (expected fdl.{name}.yml next to {})",
                    base_path.display()
                ));
            }
        }
    }
    // Dedup by canonical path, keeping first occurrence. An env overlay
    // whose chain loops back to a file already in the base chain (same
    // file via a different inheritance route) collapses cleanly.
    let mut seen = std::collections::HashSet::new();
    layers.retain(|(path, _)| seen.insert(path.clone()));
    Ok(layers)
}

/// Source path list for a base config + env overlay, in merge order. Used
/// by `fdl config show` to annotate which layer a value came from.
pub fn config_layer_sources(base_path: &Path, env: Option<&str>) -> Vec<PathBuf> {
    resolve_config_layers(base_path, env)
        .map(|ls| ls.into_iter().map(|(p, _)| p).collect())
        .unwrap_or_else(|_| vec![base_path.to_path_buf()])
}
