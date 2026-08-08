//! Schema → ArgsSpec conversion, tail/preset validation, and config
//! merging (ResolvedConfig + per-field merge helpers).

use std::collections::BTreeMap;

use super::cluster::{DdpConfig, OutputConfig, TrainingConfig};
use super::schema::{CommandConfig, CommandKind, CommandSpec, Schema};

/// Reserved flags that strict mode always tolerates in the user's tail.
/// These are fdl-level universals (help/version) or opt-ins every
/// FdlArgs-derived binary exposes (--fdl-schema) — keeping them out of
/// the `schema.options` map means strict mode has to allowlist them
/// separately or spuriously reject legal invocations.
const STRICT_UNIVERSAL_LONGS: &[(&str, Option<char>, bool)] = &[
    // (long, short, takes_value)
    ("help", Some('h'), false),
    ("version", Some('V'), false),
    ("fdl-schema", None, false),
    ("refresh-schema", None, false),
];

/// Convert a [`Schema`] into an [`ArgsSpec`](crate::args::parser::ArgsSpec) suitable for strict-mode
/// tail validation. Positional `required` flags are intentionally
/// dropped: the binary itself will enforce them after parsing, and
/// treating them as required here would turn "missing positional" into
/// a double-errored mess.
pub fn schema_to_args_spec(schema: &Schema) -> crate::args::parser::ArgsSpec {
    use crate::args::parser::{ArgsSpec, OptionDecl, PositionalDecl};

    let mut options: Vec<OptionDecl> = schema
        .options
        .iter()
        .map(|(long, spec)| OptionDecl {
            long: long.clone(),
            short: spec.short.as_deref().and_then(|s| s.chars().next()),
            takes_value: spec.ty != "bool",
            // Every value-taking option is allowed to appear bare in
            // strict mode. fdl does not second-guess whether the binary
            // would accept a bare `--foo`; that stays in the binary's
            // court.
            allows_bare: true,
            repeatable: spec.ty.starts_with("list["),
            choices: spec
                .choices
                .as_ref()
                .map(|cs| strict_choices_to_strings(cs)),
        })
        .collect();

    // Always-allowed universals — help/version/fdl-schema/refresh-schema
    // are not in the user's schema but must not trigger "unknown flag".
    for (long, short, takes_value) in STRICT_UNIVERSAL_LONGS {
        options.push(OptionDecl {
            long: (*long).to_string(),
            short: *short,
            takes_value: *takes_value,
            allows_bare: true,
            repeatable: false,
            choices: None,
        });
    }

    // Positionals: drop the `required` bit. Strict mode is scoped to
    // option names/values only; arity is the binary's concern.
    let mut positionals: Vec<PositionalDecl> = schema
        .args
        .iter()
        .map(|a| PositionalDecl {
            name: a.name.clone(),
            required: false,
            variadic: a.variadic,
            choices: a.choices.as_ref().map(|cs| strict_choices_to_strings(cs)),
        })
        .collect();
    // Catch-all so fdl-side validation never rejects positionals: like
    // arity, positional binding is the binary's concern — its own parse
    // errors loudly on excess. Keeps `-- <anything>` passthrough intact
    // under strict schemas.
    positionals.push(PositionalDecl {
        name: "rest".to_string(),
        required: false,
        variadic: true,
        choices: None,
    });

    ArgsSpec {
        options,
        positionals,
        // Non-strict schemas accept user-forwarded flags the author
        // didn't declare — the binary re-parses the tail anyway.
        // Strict schemas reject anything not declared.
        lenient_unknowns: !schema.strict,
    }
}

fn strict_choices_to_strings(cs: &[serde_json::Value]) -> Vec<String> {
    cs.iter()
        .map(|v| match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .collect()
}

/// Validate the user's extra argv tail against a schema. Always called
/// before `run::exec_command` — the parser's lenient-unknowns mode is
/// keyed off `schema.strict` so choice validation on declared flags
/// fires regardless, while unknown-flag rejection stays opt-in.
///
/// The tokenizer from [`crate::args::parser`] is reused so "did you
/// mean" suggestions, cluster, and equals handling come for free.
pub fn validate_tail(tail: &[String], schema: &Schema) -> Result<(), String> {
    // Variant-shaped CLI (tree schema): resolve to the invoked subcommand's
    // leaf, then validate the rest against it. The subcommand is `tail[0]`:
    // globals are peeled by the fdl wrapper and a branch node declares no
    // options of its own (a node is leaf XOR branch — see `Schema.commands`),
    // so any correct invocation has the subcommand first. If the tail is empty
    // (nothing typed yet) OR starts with a flag (a misplaced leaf option, or
    // its value), we can't confidently identify the subcommand here: a naive
    // scan for the first bare token would mistake `42` in `--seed 42 train`
    // for the subcommand and emit a bogus "unknown command `42`". Defer to the
    // binary's authoritative re-parse instead.
    if !schema.commands.is_empty() {
        let Some(sub) = tail.first().filter(|t| !t.starts_with('-')) else {
            return Ok(());
        };
        let Some(child) = schema.commands.get(sub) else {
            let names: Vec<&str> = schema.commands.keys().map(String::as_str).collect();
            return Err(match crate::args::parser::suggest(&names, sub) {
                Some(s) => format!("unknown command `{sub}`, did you mean `{s}`?"),
                None => format!(
                    "unknown command `{sub}`, expected one of: {}",
                    names.join(", ")
                ),
            });
        };
        return validate_tail(&tail[1..], child);
    }

    let spec = schema_to_args_spec(schema);
    let mut argv = Vec::with_capacity(tail.len() + 1);
    argv.push("fdl".to_string());
    argv.extend(tail.iter().cloned());
    crate::args::parser::parse(&spec, &argv).map(|_| ())
}

/// Validate a single preset that's about to be invoked. Combines the
/// always-on `choices:` check and, if `schema.strict`, the unknown-key
/// rejection — scoped to just this preset, not the whole `commands:`
/// map. Called from the exec path so typos in a sibling preset don't
/// block `--help` for a correct one.
pub fn validate_preset_for_exec(
    preset_name: &str,
    spec: &CommandSpec,
    schema: &Schema,
) -> Result<(), String> {
    for (key, value) in &spec.options {
        let Some(opt) = schema.options.get(key) else {
            if schema.strict {
                return Err(format!(
                    "preset `{preset_name}` pins option `{key}` which is not declared in schema.options"
                ));
            }
            continue;
        };
        let Some(choices) = &opt.choices else {
            continue;
        };
        if !choices.iter().any(|c| values_equal(c, value)) {
            let allowed: Vec<String> = choices
                .iter()
                .map(|c| match c {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect();
            return Err(format!(
                "preset `{preset_name}` sets option `{key}` to `{}` -- allowed: {}",
                display_json(value),
                allowed.join(", "),
            ));
        }
    }
    Ok(())
}

/// Always-on: validate preset YAML `options:` values against declared
/// `choices:` in the schema. An option YAML value whose key matches a
/// declared option with a `choices:` list must be one of those choices.
/// Keys not declared in the schema are ignored here — those are the
/// concern of [`validate_presets_strict`] (opt-in).
///
/// Used for whole-map validation (e.g. from a future `fdl config lint`
/// subcommand). The dispatch path uses [`validate_preset_for_exec`] so
/// sibling-preset typos don't block correct invocations.
pub fn validate_preset_values(
    commands: &BTreeMap<String, CommandSpec>,
    schema: &Schema,
) -> Result<(), String> {
    for (preset_name, spec) in commands {
        match spec.kind() {
            Ok(CommandKind::Preset) => {}
            _ => continue,
        }
        for (key, value) in &spec.options {
            let Some(opt) = schema.options.get(key) else {
                continue; // unknown key — strict's problem, not ours
            };
            let Some(choices) = &opt.choices else {
                continue; // no choices declared — anything goes
            };
            if !choices.iter().any(|c| values_equal(c, value)) {
                let allowed: Vec<String> = choices
                    .iter()
                    .map(|c| match c {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect();
                return Err(format!(
                    "preset `{preset_name}` sets option `{key}` to `{}` -- allowed: {}",
                    display_json(value),
                    allowed.join(", "),
                ));
            }
        }
    }
    Ok(())
}

/// Compare two JSON values for equality, treating YAML's loose-typed
/// representation (a preset might write `batch-size: 32` as an int
/// while the schema's choices list contains `"32"` as a string).
fn values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    if a == b {
        return true;
    }
    // Cross-type string ↔ number comparison for YAML-friendly matching.
    match (a, b) {
        (serde_json::Value::String(s), other) | (other, serde_json::Value::String(s)) => {
            s == &other.to_string()
        }
        _ => false,
    }
}

fn display_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// At load time, reject preset `options:` keys that are not declared in
/// the enclosing schema. Runs only when `schema.strict == true`, and
/// only against entries resolved to [`CommandKind::Preset`] — `run:` and
/// `path:` kinds don't share the parent schema.
pub fn validate_presets_strict(
    commands: &BTreeMap<String, CommandSpec>,
    schema: &Schema,
) -> Result<(), String> {
    for (preset_name, spec) in commands {
        match spec.kind() {
            Ok(CommandKind::Preset) => {}
            _ => continue,
        }
        for key in spec.options.keys() {
            if !schema.options.contains_key(key) {
                return Err(format!(
                    "preset `{preset_name}` pins option `{key}` which is not declared in schema.options"
                ));
            }
        }
    }
    Ok(())
}

// ── Merge ───────────────────────────────────────────────────────────────

/// Merge the enclosing `CommandConfig` defaults with a named preset's
/// overrides. Preset values win. Used when dispatching an inline preset
/// command (neither `run` nor `path`).
pub fn merge_preset(root: &CommandConfig, preset: &CommandSpec) -> ResolvedConfig {
    ResolvedConfig {
        ddp: merge_ddp(&root.ddp, &preset.ddp),
        training: merge_training(&root.training, &preset.training),
        output: merge_output(&root.output, &preset.output),
        options: preset.options.clone(),
    }
}

/// Resolved config from root defaults only (no job).
pub fn defaults_only(root: &CommandConfig) -> ResolvedConfig {
    ResolvedConfig {
        ddp: root.ddp.clone().unwrap_or_default(),
        training: root.training.clone().unwrap_or_default(),
        output: root.output.clone().unwrap_or_default(),
        options: BTreeMap::new(),
    }
}

/// Fully resolved configuration ready for arg translation.
pub struct ResolvedConfig {
    pub ddp: DdpConfig,
    pub training: TrainingConfig,
    pub output: OutputConfig,
    pub options: BTreeMap<String, serde_json::Value>,
}

macro_rules! merge_field {
    ($base:expr, $over:expr, $field:ident) => {
        $over
            .as_ref()
            .and_then(|o| o.$field.clone())
            .or_else(|| $base.as_ref().and_then(|b| b.$field.clone()))
    };
}

fn merge_ddp(base: &Option<DdpConfig>, over: &Option<DdpConfig>) -> DdpConfig {
    DdpConfig {
        mode: merge_field!(base, over, mode),
        policy: merge_field!(base, over, policy),
        backend: merge_field!(base, over, backend),
        anchor: merge_field!(base, over, anchor),
        max_anchor: merge_field!(base, over, max_anchor),
        overhead_target: merge_field!(base, over, overhead_target),
        divergence_threshold: merge_field!(base, over, divergence_threshold),
        max_batch_diff: merge_field!(base, over, max_batch_diff),
        speed_hint: merge_field!(base, over, speed_hint),
        partition_ratios: merge_field!(base, over, partition_ratios),
        progressive: merge_field!(base, over, progressive),
        max_grad_norm: merge_field!(base, over, max_grad_norm),
        lr_scale_ratio: merge_field!(base, over, lr_scale_ratio),
        snapshot_timeout: merge_field!(base, over, snapshot_timeout),
        checkpoint_every: merge_field!(base, over, checkpoint_every),
        timeline: merge_field!(base, over, timeline),
    }
}

fn merge_training(base: &Option<TrainingConfig>, over: &Option<TrainingConfig>) -> TrainingConfig {
    TrainingConfig {
        epochs: merge_field!(base, over, epochs),
        batch_size: merge_field!(base, over, batch_size),
        batches_per_epoch: merge_field!(base, over, batches_per_epoch),
        lr: merge_field!(base, over, lr),
        seed: merge_field!(base, over, seed),
    }
}

fn merge_output(base: &Option<OutputConfig>, over: &Option<OutputConfig>) -> OutputConfig {
    OutputConfig {
        dir: merge_field!(base, over, dir),
        timeline: merge_field!(base, over, timeline),
        monitor: merge_field!(base, over, monitor),
    }
}
