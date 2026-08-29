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

/// Locate the project config inside `dir` only — no walk-up, no
/// example-copy prompt. For resolvers that already hold the project root
/// and must not block on interactive prompts.
pub fn find_config_in(dir: &Path) -> Option<PathBuf> {
    CONFIG_NAMES
        .iter()
        .map(|n| dir.join(n))
        .find(|c| c.is_file())
}

/// Walk up from `start` to the PROJECT-level config, stepping over
/// command-level fdl.ymls on the way. A command directory (e.g.
/// `ddp-bench/`) carries its own fdl.yml in [`CommandConfig`] shape —
/// `entry:`/`compile:`/`docker:`/`run:` at top level — which is not a
/// [`ProjectConfig`] and must not be mistaken for one: project-scoped
/// consumers (`fdl join`'s `join:` block, `fdl status`'s `cluster:`)
/// run happily from inside a command dir and need the config that OWNS
/// those blocks, one or more levels up. A file that is neither shape
/// (unreadable, malformed) is returned as the answer so the caller's
/// loader reports it loudly rather than this walk silently skipping a
/// broken project config.
///
/// [`CommandConfig`]: super::CommandConfig
/// [`ProjectConfig`]: super::ProjectConfig
pub fn find_project_config(start: &Path) -> Option<PathBuf> {
    const COMMAND_MARKERS: &[&str] = &["entry", "compile", "docker", "run", "append"];
    let mut dir = start.to_path_buf();
    loop {
        if let Some(candidate) = find_config_in(&dir) {
            let is_command_shaped = std::fs::read_to_string(&candidate)
                .ok()
                .and_then(|raw| serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&raw).ok())
                .and_then(|v| v.as_mapping().cloned())
                .is_some_and(|map| {
                    COMMAND_MARKERS
                        .iter()
                        .any(|k| map.contains_key(serde_yaml_ng::Value::String((*k).into())))
                });
            if !is_command_shaped {
                return Some(candidate);
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
    // Never prompt without a terminal: in CI, shell completions, or any
    // piped context the prompt is invisible and `read_line` hits EOF,
    // which the Y-default would treat as consent — silently adopting the
    // example as the live config. Non-interactive callers just use the
    // example file directly, copying nothing.
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return false;
    }
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
pub fn load_project_with_env(base_path: &Path, env: Option<&str>) -> Result<ProjectConfig, String> {
    let layers = resolve_config_layers(base_path, env)?;
    let merged =
        crate::overlay::merge_layers(layers.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>());
    // An empty (or comments-only) fdl.yml — and empty overlays — merge to a
    // null value. That is a valid "nothing configured" state, so return an
    // all-defaults config instead of letting `from_str::<ProjectConfig>("null")`
    // fail with a cryptic serde "invalid type: unit value" error.
    if merged.is_null() {
        return Ok(ProjectConfig::default());
    }
    // Re-serialize so `from_str`'s parser tracks line/col through deserialize.
    // `from_value` discards positional info, leaving errors location-less.
    let merged_str = serde_yaml_ng::to_string(&merged).map_err(|e| {
        format!(
            "{}: failed to re-serialize merged YAML for diagnostics: {e}",
            base_path.display()
        )
    })?;
    let cfg = serde_yaml_ng::from_str::<ProjectConfig>(&merged_str).map_err(|e| {
        let names: Vec<String> = layers
            .iter()
            .map(|(p, _)| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect();
        let env_hint = env.map(|n| format!(" {n}")).unwrap_or_default();
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
    })?;
    reject_user_ranks(&cfg, base_path)?;
    validate_gpu_ram_shares(&cfg, base_path)?;
    Ok(cfg)
}

/// Range-check every `gpu_ram_share` at load time: a non-negative,
/// finite fraction of host RAM. Values above 1.0 are deliberately legal
/// (the knob exists partly for platforms whose `MemTotal` under-states
/// what the APU can address), so only a negative or non-finite value is
/// refused — loud here, where the file and key can be named, not at
/// launch.
fn validate_gpu_ram_shares(cfg: &ProjectConfig, base_path: &Path) -> Result<(), String> {
    let bad = |s: Option<f64>| s.is_some_and(|f| !f.is_finite() || f < 0.0);
    let err = |key: &str, got: f64| {
        Err(format!(
            "{}: {key} must be a non-negative fraction of host RAM \
             (e.g. 0.5), got {got}",
            base_path.display(),
        ))
    };
    if let Some(cluster) = &cfg.cluster {
        if bad(cluster.gpu_ram_share) {
            return err("cluster.gpu_ram_share", cluster.gpu_ram_share.unwrap());
        }
        for (i, w) in cluster.workers.iter().enumerate() {
            if bad(w.gpu_ram_share) {
                return err(
                    &format!("cluster.workers[{i}] ({:?}) gpu_ram_share", w.host),
                    w.gpu_ram_share.unwrap(),
                );
            }
        }
    }
    if let Some(join) = &cfg.join
        && bad(join.gpu_ram_share)
    {
        return err("join.gpu_ram_share", join.gpu_ram_share.unwrap());
    }
    Ok(())
}

/// Reject `ranks:` in user-authored worker blocks. The key looks
/// load-bearing but never was: rank assignment is computed from probed
/// device counts (`ClusterConfig::populate_ranks`). The field must stay
/// deserializable for the canonical-JSON wire round-trip, so serde's
/// `deny_unknown_fields` cannot catch it; this check runs on the
/// user-YAML entry path only.
fn reject_user_ranks(cfg: &ProjectConfig, base_path: &Path) -> Result<(), String> {
    let Some(cluster) = &cfg.cluster else {
        return Ok(());
    };
    for (i, w) in cluster.workers.iter().enumerate() {
        if !w.ranks.is_empty() {
            return Err(format!(
                "{}: cluster.workers[{i}] ({:?}) declares `ranks:`, which is \
                 not user configuration; ranks are computed from probed device \
                 counts at launch. Remove the key.",
                base_path.display(),
                w.host,
            ));
        }
    }
    Ok(())
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

/// Load the raw merged [`serde_yaml_ng::Value`] for a config + optional env
/// overlay. Exposed so callers like `fdl config show` can inspect the
/// resolved view before it is deserialized into a strongly-typed struct.
pub fn load_merged_value(
    base_path: &Path,
    env: Option<&str>,
) -> Result<serde_yaml_ng::Value, String> {
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
) -> Result<Vec<(PathBuf, serde_yaml_ng::Value)>, String> {
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

/// Load the top-level `join:` block from the PROJECT config (base
/// fdl.yml merged with the active env overlay when one is selected),
/// plus the directory it lives in (the project root — where libtorch/
/// is anchored). The walk steps over command-level fdl.ymls
/// ([`find_project_config`]): `fdl join` typically runs from
/// the command dir the training binary expects as cwd (e.g.
/// `ddp-bench/`), whose own fdl.yml is a command config that neither
/// carries a `join:` block nor marks the libtorch root. `Ok(None)`
/// root/block when there is no project at all — flags carry everything
/// then; a present-but-broken project config is a loud error, not a
/// silent fallback (the operator may be relying on `join.bin`).
pub fn load_join_block() -> Result<(Option<super::WorkerJoin>, Option<PathBuf>), String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("cannot read the current directory: {e}"))?;
    let Some(config_path) = find_project_config(&cwd) else {
        return Ok((None, None));
    };
    let env_name = std::env::var("FDL_ENV")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let project = load_project_with_env(&config_path, env_name.as_deref())
        .map_err(|e| format!("cannot load {}: {e}", config_path.display()))?;
    let root = config_path.parent().map(Path::to_path_buf);
    Ok((project.join, root))
}
