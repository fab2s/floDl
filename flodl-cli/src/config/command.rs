//! Sub-command config loading from registered command directories.

use std::path::{Path, PathBuf};

use super::loading::{CONFIG_NAMES, EXAMPLE_SUFFIXES, try_copy_example};
use super::schema::{validate_schema, CommandConfig};

/// Load a command config from a sub-directory.
///
/// Applies the same `.example`/`.dist` fallback as [`super::loading::find_config`]. If a
/// `schema:` block is present, validates it before returning.
pub fn load_command(dir: &Path) -> Result<CommandConfig, String> {
    load_command_with_env(dir, None)
}

/// Load a sub-command config with an optional environment overlay.
///
/// Applies the same `.example`/`.dist` fallback as [`super::loading::find_config`] to locate
/// the base file, then deep-merges a sibling `fdl.<env>.yml` overlay if one
/// exists. A *missing* overlay is silently accepted here (different from
/// [`super::loading::load_project_with_env`]) — envs declared at the project root don't
/// have to exist for every sub-command.
pub fn load_command_with_env(dir: &Path, env: Option<&str>) -> Result<CommandConfig, String> {
    // Resolve the base config path (with .example fallback, same as before).
    let mut base_path: Option<PathBuf> = None;
    for name in CONFIG_NAMES {
        let path = dir.join(name);
        if path.is_file() {
            base_path = Some(path);
            break;
        }
    }
    if base_path.is_none() {
        for name in CONFIG_NAMES {
            for suffix in EXAMPLE_SUFFIXES {
                let example = dir.join(format!("{name}{suffix}"));
                if example.is_file() {
                    let target = dir.join(name);
                    let src = if try_copy_example(&example, &target) {
                        target
                    } else {
                        example
                    };
                    base_path = Some(src);
                    break;
                }
            }
            if base_path.is_some() {
                break;
            }
        }
    }
    let base_path = base_path
        .ok_or_else(|| format!("no fdl.yml found in {}", dir.display()))?;

    // Layered load: base chain + optional env overlay chain. Both sides
    // run through `resolve_chain` so `inherit-from:` composes the same
    // way for nested commands as for the project root.
    let mut layers = crate::overlay::resolve_chain(&base_path)?;
    if let Some(name) = env {
        if let Some(p) = crate::overlay::find_env_file(&base_path, name) {
            layers.extend(crate::overlay::resolve_chain(&p)?);
        }
    }
    let mut seen = std::collections::HashSet::new();
    layers.retain(|(path, _)| seen.insert(path.clone()));
    let merged = crate::overlay::merge_layers(
        layers.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
    );
    // Re-serialize so `from_str`'s parser tracks line/col through
    // deserialize (`from_value` discards positional info). With
    // `deny_unknown_fields` on the config structs, unknown-key errors
    // carry a location this way. Positions refer to the merged
    // document, not any single source file, when overlays are in play.
    let merged_str = serde_yaml::to_string(&merged).map_err(|e| {
        format!(
            "{}: failed to re-serialize merged YAML for diagnostics: {e}",
            base_path.display()
        )
    })?;
    let mut cfg: CommandConfig = serde_yaml::from_str(&merged_str)
        .map_err(|e| format!("{}: {}", base_path.display(), e))?;

    if let Some(schema) = &cfg.schema {
        validate_schema(schema)
            .map_err(|e| format!("schema error in {}/fdl.yml: {e}", dir.display()))?;
        // Preset validation (choice values + strict unknown-key rejection)
        // is intentionally deferred to the exec path. Load-time validation
        // would block `fdl <cmd> --help` whenever ANY preset in the config
        // has a typo — worse UX than letting help render and erroring only
        // when the broken preset is actually invoked.
    }

    // Cache precedence: a valid, fresh cached schema (written by `fdl <cmd>
    // --refresh-schema` or auto-probed below) wins over the inline YAML
    // schema. This lets a binary become the source of truth for its own
    // surface once it opts into the `--fdl-schema` contract. A cache that
    // is older than the command's fdl.yml is treated as stale and skipped
    // — the inline schema (if any) reasserts until a refresh happens.
    let cmd_name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("_");
    let cache = crate::schema_cache::cache_path(dir, cmd_name);
    // Reference mtimes: config files that, when edited, might invalidate
    // the cached schema (e.g. changing `entry:` to point somewhere else).
    let refs: Vec<std::path::PathBuf> = CONFIG_NAMES
        .iter()
        .map(|n| dir.join(n))
        .filter(|p| p.exists())
        .collect();
    if !crate::schema_cache::is_stale(&cache, &refs) {
        if let Some(cached) = crate::schema_cache::read_cache(&cache) {
            cfg.schema = Some(cached);
        }
    } else if let Some(entry) = cfg.entry.as_deref() {
        // Auto-probe non-cargo entries when the cache is stale or missing.
        // Cargo entries are skipped by default — `cargo run --fdl-schema`
        // triggers a full compile which is unacceptable latency for `-h`
        // — unless the yml explicitly opts in via `compile: true`.
        // Scripts and pre-built binaries are expected to handle the flag
        // cheaply (emit JSON and exit), so probing them on demand is safe.
        // Probe failures are swallowed: an entry that doesn't implement
        // `--fdl-schema` simply falls through to the inline schema (or no
        // schema) — help still renders.
        let opts_into_compile = cfg.compile.unwrap_or(false);
        let should_probe =
            !crate::schema_cache::is_cargo_entry(entry) || opts_into_compile;
        if should_probe {
            if let Ok(probed) =
                crate::schema_cache::probe(entry, dir, cfg.docker.as_deref())
            {
                // Best-effort cache write: if the dir is read-only, the
                // schema still applies to this invocation, we just re-probe
                // next time. Non-fatal.
                let _ = crate::schema_cache::write_cache(&cache, &probed);
                cfg.schema = Some(probed);
            }
        }
    }

    Ok(cfg)
}

// ── Strict-mode validation ──────────────────────────────────────────────

