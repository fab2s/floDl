//! Sub-command config loading from registered command directories.

use std::path::{Path, PathBuf};

use super::loading::{CONFIG_NAMES, EXAMPLE_SUFFIXES, try_copy_example};
use super::schema::{CommandConfig, validate_schema};

/// How much a load may spend to obtain the command's schema.
///
/// A schema comes from one of three places: the cache, a `--fdl-schema`
/// probe of the entry, or an inline `schema:` block. Probing is what costs,
/// and its cost is not uniform — which is why the caller picks rather than
/// the loader guessing.
///
/// The split exists because most loads do not want the schema at all. `fdl
/// --help` loads every child config just to read its `description:`, and
/// shell completion loads all of them to list flags. Neither can afford a
/// container spinning up a cargo build, and a tab-press least of all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeCost {
    /// Probe only entries that answer in milliseconds — a script or a
    /// pre-built binary. A `compile: true` cargo entry is served from its
    /// cache, stale included: an out-of-date surface beats no surface,
    /// since dropping the schema also drops `strict` and every `choices:`
    /// contract along with the help text.
    Cheap,
    /// Compile if that is what an authoritative surface costs. For the
    /// paths that actually consume the schema: rendering `--help` and
    /// validating an argv tail before handing it to the entry.
    Compile,
}

/// Load a command config from a sub-directory, [cheaply](ProbeCost::Cheap).
///
/// Applies the same `.example`/`.dist` fallback as [`super::loading::find_config`]. If a
/// `schema:` block is present, validates it before returning.
pub fn load_command(dir: &Path) -> Result<CommandConfig, String> {
    load_command_with_env(dir, None)
}

/// Load a sub-command config with an optional environment overlay,
/// [cheaply](ProbeCost::Cheap).
///
/// Applies the same `.example`/`.dist` fallback as [`super::loading::find_config`] to locate
/// the base file, then deep-merges a sibling `fdl.<env>.yml` overlay if one
/// exists. A *missing* overlay is silently accepted here (different from
/// [`super::loading::load_project_with_env`]) — envs declared at the project root don't
/// have to exist for every sub-command.
pub fn load_command_with_env(dir: &Path, env: Option<&str>) -> Result<CommandConfig, String> {
    load_command_full(dir, env, ProbeCost::Cheap)
}

/// [`load_command_with_env`], but willing to pay a `compile: true` probe to
/// get the entry's current surface. See [`ProbeCost`].
pub fn load_command_probed(dir: &Path, env: Option<&str>) -> Result<CommandConfig, String> {
    load_command_full(dir, env, ProbeCost::Compile)
}

fn load_command_full(
    dir: &Path,
    env: Option<&str>,
    cost: ProbeCost,
) -> Result<CommandConfig, String> {
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
    let base_path = base_path.ok_or_else(|| format!("no fdl.yml found in {}", dir.display()))?;

    // Layered load: base chain + optional env overlay chain. Both sides
    // run through `resolve_chain` so `inherit-from:` composes the same
    // way for nested commands as for the project root.
    let mut layers = crate::overlay::resolve_chain(&base_path)?;
    if let Some(name) = env
        && let Some(p) = crate::overlay::find_env_file(&base_path, name)
    {
        layers.extend(crate::overlay::resolve_chain(&p)?);
    }
    let mut seen = std::collections::HashSet::new();
    layers.retain(|(path, _)| seen.insert(path.clone()));
    let merged =
        crate::overlay::merge_layers(layers.into_iter().map(|(_, v)| v).collect::<Vec<_>>());
    // Re-serialize so `from_str`'s parser tracks line/col through
    // deserialize (`from_value` discards positional info). With
    // `deny_unknown_fields` on the config structs, unknown-key errors
    // carry a location this way. Positions refer to the merged
    // document, not any single source file, when overlays are in play.
    let merged_str = serde_yaml_ng::to_string(&merged).map_err(|e| {
        format!(
            "{}: failed to re-serialize merged YAML for diagnostics: {e}",
            base_path.display()
        )
    })?;
    let mut cfg: CommandConfig = serde_yaml_ng::from_str(&merged_str)
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

    apply_schema(&mut cfg, dir, cost);

    Ok(cfg)
}

/// Whether a stale-or-missing schema may be re-probed under `cost`.
///
/// Split out of [`apply_schema`] because it is the whole latency contract in
/// one expression, and the expression has to hold for callers that never see
/// a container: a cargo entry compiles, so it is off-limits to a
/// [`ProbeCost::Cheap`] load no matter what the yml says. Anything else
/// answers `--fdl-schema` in milliseconds and is always fair game.
pub(super) fn probe_allowed(entry: &str, compiles: bool, cost: ProbeCost) -> bool {
    if crate::schema_cache::is_cargo_entry(entry) {
        compiles && cost == ProbeCost::Compile
    } else {
        true
    }
}

/// Resolve `cfg.schema` from the cache, a `--fdl-schema` probe, or the
/// inline `schema:` block already parsed into `cfg`.
///
/// Cache precedence: a valid, fresh cached schema (written by `fdl <cmd>
/// --refresh-schema` or probed below) wins over the inline YAML schema. This
/// lets a binary become the source of truth for its own surface once it opts
/// into the `--fdl-schema` contract.
fn apply_schema(cfg: &mut CommandConfig, dir: &Path, cost: ProbeCost) {
    let cmd_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("_");
    let cache = crate::schema_cache::cache_path(dir, cmd_name);
    // Reference mtimes: everything whose edit could change the cached schema.
    //
    // The config file, because `entry:` might now point somewhere else — and,
    // when the binary declares its own surface, the sources that surface is
    // compiled from. Watching only the config meant editing a CLI struct left
    // the cache stale with NO signal: `-h` kept rendering the previous flags
    // until someone happened to touch the yml. Silent and repeatedly confusing,
    // since the binary itself was correct all along.
    let mut refs: Vec<std::path::PathBuf> = CONFIG_NAMES
        .iter()
        .map(|n| dir.join(n))
        .filter(|p| p.exists())
        .collect();
    let compiles = cfg.compile.unwrap_or(false);
    if compiles {
        refs.extend(crate::schema_cache::schema_source_refs(dir));
    }

    if !crate::schema_cache::is_stale(&cache, &refs) {
        if let Some(cached) = crate::schema_cache::read_cache(&cache) {
            cfg.schema = Some(cached);
        }
        return;
    }

    // Stale or missing. Probe when the entry can answer within the budget
    // this load was given: a script or pre-built binary emits JSON and exits
    // (cheap, always allowed), while a cargo entry compiles first — allowed
    // only where the yml opted in AND the caller asked for an authoritative
    // surface.
    if let Some(entry) = cfg.entry.as_deref() {
        let is_cargo = crate::schema_cache::is_cargo_entry(entry);
        if probe_allowed(entry, compiles, cost) {
            if is_cargo {
                // A compile behind `-h` is a multi-second wall at best and a
                // cold container build at worst. Say so before going quiet,
                // or it reads as a hang.
                eprintln!(
                    "fdl: reading {cmd_name}'s options from the binary \
                     (compiling once, then cached)"
                );
            }
            match crate::schema_cache::probe(entry, dir, cfg.docker.as_deref()) {
                Ok(probed) => {
                    // Best-effort cache write: if the dir is read-only, the
                    // schema still applies to this invocation, we just
                    // re-probe next time. Non-fatal.
                    let _ = crate::schema_cache::write_cache(&cache, &probed);
                    cfg.schema = Some(probed);
                    return;
                }
                // `compile: true` is an explicit claim that this entry
                // answers `--fdl-schema`, so a failure is a broken contract
                // and gets reported. Without it the fall-through is the
                // documented behaviour for an entry that simply doesn't
                // implement the flag, and saying so on every `-h` would be
                // noise.
                Err(e) if compiles => {
                    crate::cli_error!("could not read {cmd_name}'s schema: {e}");
                }
                Err(_) => {}
            }
        }
    }

    // Nothing fresh to be had. A stale cache still describes the entry far
    // better than nothing does, so prefer it over falling back to the inline
    // schema — help renders, and `strict`/`choices:` keep holding.
    if let Some(cached) = crate::schema_cache::read_cache(&cache) {
        cfg.schema = Some(cached);
    }
}

// ── Strict-mode validation ──────────────────────────────────────────────
