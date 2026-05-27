//! fdl.yaml schema types: ProjectConfig, CommandConfig, CommandSpec,
//! CommandKind, Schema, OptionSpec, ArgSpec, plus `validate_schema`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::cluster::{ClusterConfig, DdpConfig, OutputConfig, TrainingConfig};

/// Root fdl.yaml at project root.
#[derive(Debug, Default, Deserialize)]
pub struct ProjectConfig {
    #[serde(default)]
    pub description: Option<String>,
    /// Commands defined at this level. Each value is a [`CommandSpec`] that
    /// encodes the kind of command (inline `run` script, `path` pointer to
    /// a child fdl.yml, or inline preset reusing the parent entry).
    #[serde(default)]
    pub commands: BTreeMap<String, CommandSpec>,
    /// Multi-host cluster topology. When present, commands marked
    /// `cluster: true` are dispatched across every worker in
    /// [`ClusterConfig::workers`]. Lives at the project root because the
    /// topology is shared across all sub-command fdl.yml files; the
    /// canonical author pattern is to put the cluster block in a
    /// `fdl.<env>.yml` overlay (e.g. `fdl.vm.yml`) that deep-merges over
    /// the base `fdl.yml`.
    #[serde(default)]
    pub cluster: Option<ClusterConfig>,
}

// ── Sub-command config ──────────────────────────────────────────────────

/// Sub-command fdl.yaml (e.g., ddp-bench/fdl.yaml).
///
/// Identical shape to [`ProjectConfig`] but with an executable `entry:`
/// and optional structured config sections (ddp/training/output) that
/// inline preset commands can override.
#[derive(Debug, Default, Deserialize)]
pub struct CommandConfig {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub entry: Option<String>,
    /// Docker compose service name. When set, entry is wrapped in
    /// `docker compose run --rm <service> bash -c "cd <workdir> && <entry> <args>"`.
    #[serde(default)]
    pub docker: Option<String>,
    #[serde(default)]
    pub ddp: Option<DdpConfig>,
    #[serde(default)]
    pub training: Option<TrainingConfig>,
    #[serde(default)]
    pub output: Option<OutputConfig>,
    /// Nested commands — inline presets of this config's entry, standalone
    /// `run` scripts, or `path` pointers to child fdl.yml files.
    #[serde(default)]
    pub commands: BTreeMap<String, CommandSpec>,
    /// Help-only placeholder name for the first-positional slot when
    /// `commands:` holds presets. Defaults to "preset". Pure UX — it
    /// does not affect dispatch (presets are always looked up by name).
    /// Useful to match domain vocabulary, e.g. `arg-name: recipe` or
    /// `arg-name: target`.
    #[serde(default, rename = "arg-name")]
    pub arg_name: Option<String>,
    /// Inline interim schema (before `<entry> --fdl-schema` is implemented).
    /// Drives help rendering, validation, and completions.
    #[serde(default)]
    pub schema: Option<Schema>,
    /// Opt-in flag for cargo-entry schema probing. Cargo entries are
    /// auto-skipped from probing because `cargo run --fdl-schema` triggers
    /// a full compile (unacceptable latency for `-h`). Setting `compile:
    /// true` declares "I'm fine with the first-run compile cost — probe
    /// my binary for its real schema." Subsequent invocations use the
    /// mtime-keyed cache and pay no compile cost. Absent or `false` keeps
    /// the default skip behavior, so the inline yml schema (if any)
    /// stays the source of truth.
    #[serde(default)]
    pub compile: Option<bool>,
}

// ── Unified command specification ───────────────────────────────────────

/// A command at any nesting level. Three mutually-exclusive kinds are
/// recognised at resolve time:
///
/// - **Path** (`path` set, or by default when the map is empty/null): the
///   command is a pointer to a child `fdl.yml`. By convention the path is
///   `./<command-name>/` when omitted.
/// - **Run** (`run` set): the command is a self-contained shell script
///   that is executed as-is. Optional `docker:` service routes it through
///   `docker compose`.
/// - **Preset**: neither `path` nor `run` is set. The command merges its
///   `ddp` / `training` / `output` / `options` fields over the enclosing
///   `CommandConfig` defaults and invokes that config's `entry:`.
#[derive(Debug, Default, Clone)]
pub struct CommandSpec {
    pub description: Option<String>,
    /// Inline shell command. Mutex with `path`.
    pub run: Option<String>,
    /// Default trailing tokens for a `run:` command. Split on its own
    /// first `--` into pre/post halves; user args after fdl's first
    /// `--` are split similarly, then everything merges as
    /// `[append-pre] [user-pre] -- [append-post] [user-post]`. Append
    /// seeds defaults; user args last-win on each side. The legacy
    /// `append: -- --nocapture` shape (empty pre, libtest tokens post)
    /// keeps working as a degenerate case. Drop entirely with the
    /// global `--no-append` flag.
    pub append: Option<String>,
    /// Pointer to a child directory containing its own `fdl.yml`. Absolute
    /// or relative to the declaring config's directory. Mutex with `run`.
    /// `None` + no other fields = "use the convention path
    /// `./<command-name>/`".
    pub path: Option<String>,
    /// Docker compose service for `run`-kind commands.
    pub docker: Option<String>,
    /// Preset overrides. Only consulted when neither `run` nor `path` is set.
    pub ddp: Option<DdpConfig>,
    pub training: Option<TrainingConfig>,
    pub output: Option<OutputConfig>,
    pub options: BTreeMap<String, serde_json::Value>,
    /// Dispatch this command across every host in
    /// [`ProjectConfig::cluster`]'s `hosts` list. Set to `true` to opt the
    /// command into multi-host execution. Default `None` means single-host
    /// (today's behavior). Has no effect when no `cluster:` block is
    /// declared at the project root.
    pub cluster: Option<bool>,
}

/// What kind of command is this, resolved from a [`CommandSpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandKind {
    /// `run: "…"` — execute the inline shell command (optionally in Docker).
    Run,
    /// `path: "…"` or convention default — load `<path>/fdl.yml` and
    /// recurse.
    Path,
    /// Neither `run` nor `path`. Merges preset fields onto the enclosing
    /// `CommandConfig` defaults and invokes that config's `entry:`.
    Preset,
}

impl CommandSpec {
    /// Classify this command. Returns an error when both `run` and `path`
    /// are declared — always a mistake, caught loudly rather than silently
    /// picking one. Also rejects `docker:` without `run:`: the docker
    /// service wraps the inline run-script, so pairing it with a `path:`
    /// pointer or a preset entry is always silent-noop territory.
    pub fn kind(&self) -> Result<CommandKind, String> {
        if self.docker.is_some() && self.run.is_none() {
            return Err(
                "command declares `docker:` without `run:`; \
                 `docker:` only wraps inline run-scripts"
                    .to_string(),
            );
        }
        if self.append.is_some() && self.run.is_none() {
            return Err(
                "command declares `append:` without `run:`; \
                 `append:` only forwards trailing tokens for inline run-scripts"
                    .to_string(),
            );
        }
        match (self.run.as_deref(), self.path.as_deref()) {
            (Some(_), Some(_)) => Err(
                "command declares both `run:` and `path:`; \
                 only one is allowed"
                    .to_string(),
            ),
            (Some(_), None) => Ok(CommandKind::Run),
            (None, Some(_)) => Ok(CommandKind::Path),
            (None, None) => {
                // No kind-selecting field. If preset fields are present,
                // treat as Preset; otherwise, fall through to Path (the
                // convention-default: `./<name>/fdl.yml`).
                if self.ddp.is_some()
                    || self.training.is_some()
                    || self.output.is_some()
                    || !self.options.is_empty()
                {
                    Ok(CommandKind::Preset)
                } else {
                    Ok(CommandKind::Path)
                }
            }
        }
    }

    /// Resolve the effective directory for a `Path`-kind command declared
    /// in `parent_dir`. Applies the `./<name>/` convention when `path` is
    /// unset.
    pub fn resolve_path(&self, name: &str, parent_dir: &Path) -> PathBuf {
        match &self.path {
            Some(p) => parent_dir.join(p),
            None => parent_dir.join(name),
        }
    }
}

// Custom Deserialize so that `commands: { name: ~ }` (YAML null) and
// `commands: { name: }` (empty value) both deserialize to a default
// `CommandSpec`. Without this, serde_yaml errors on null because a
// struct expects a map.
impl<'de> Deserialize<'de> for CommandSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Inner {
            #[serde(default)]
            description: Option<String>,
            #[serde(default)]
            run: Option<String>,
            #[serde(default)]
            append: Option<String>,
            #[serde(default)]
            path: Option<String>,
            #[serde(default)]
            docker: Option<String>,
            #[serde(default)]
            ddp: Option<DdpConfig>,
            #[serde(default)]
            training: Option<TrainingConfig>,
            #[serde(default)]
            output: Option<OutputConfig>,
            #[serde(default)]
            options: BTreeMap<String, serde_json::Value>,
            #[serde(default)]
            cluster: Option<bool>,
        }

        let raw = serde_yaml::Value::deserialize(deserializer)?;
        if matches!(raw, serde_yaml::Value::Null) {
            return Ok(Self::default());
        }
        let inner: Inner =
            serde_yaml::from_value(raw).map_err(serde::de::Error::custom)?;
        Ok(Self {
            description: inner.description,
            run: inner.run,
            append: inner.append,
            path: inner.path,
            docker: inner.docker,
            ddp: inner.ddp,
            training: inner.training,
            output: inner.output,
            options: inner.options,
            cluster: inner.cluster,
        })
    }
}

// ── Schema (interim hand-written, future `<entry> --fdl-schema`) ────────

/// The schema declared inline in a sub-command's fdl.yaml. Maps 1:1 to
/// what `<entry> --fdl-schema` will later emit as JSON.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Schema {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<ArgSpec>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, OptionSpec>,
    /// When true, the fdl layer rejects options not declared in the
    /// schema before the sub-command's entry ever runs. Two validation
    /// points:
    ///
    /// 1. *Load time* — preset `options:` maps are checked against the
    ///    enclosing `schema.options` (see [`super::validation::validate_presets_strict`]).
    ///    A typo like `options: { batchsize: 32 }` when the schema
    ///    declares `batch-size` is a loud load error.
    /// 2. *Dispatch time* — the user's extra argv tail is tokenized
    ///    against the schema (see [`super::validation::validate_tail`]). Unknown flags
    ///    error out with a "did you mean" suggestion instead of being
    ///    silently forwarded.
    ///
    /// **Validation NOT gated by `strict`** — always-on for declared
    /// items, so positive assertions from the schema always hold:
    /// - `choices:` on options: the user's value and any preset YAML
    ///   value must be in the list.
    /// - `choices:` on positional args: the user's value must be in
    ///   the list (when strict is off, this may mis-fire if unknown
    ///   flags push orphan values into positional slots — opt into
    ///   strict for clean positional handling).
    ///
    /// `strict` is purely about **unknown** options/args, not about
    /// validating declared contracts.
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
}

/// A flag option, `--name` / `-x`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptionSpec {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<serde_json::Value>>,
    /// Single-letter short alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub short: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<String>,
    /// Shell snippet producing completion values.
    /// Consumed by `fdl completions <shell>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    pub completer: Option<String>,
}

/// A positional argument.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArgSpec {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_required")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub variadic: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<serde_json::Value>>,
    /// Shell snippet producing completion values.
    /// Consumed by `fdl completions <shell>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    pub completer: Option<String>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn default_required() -> bool {
    true
}

/// Flags reserved at the fdl level — no sub-command option may shadow them.
/// Kept in sync with main.rs dispatch.
const RESERVED_LONGS: &[&str] = &[
    "help", "version", "quiet", "env",
];
const RESERVED_SHORTS: &[&str] = &[
    "h", "V", "q", "v", "e",
];
const VALID_TYPES: &[&str] = &[
    "string", "int", "float", "bool", "path",
    "list[string]", "list[int]", "list[float]", "list[path]",
];

/// Check a schema for collisions and structural issues.
///
/// Loud-at-load-time: ambiguity caught here is cheaper to fix than mysterious
/// pass-through behavior at runtime.
pub fn validate_schema(schema: &Schema) -> Result<(), String> {
    // Options: check types, shorts, reserved flags.
    let mut short_seen: BTreeMap<String, String> = BTreeMap::new();
    for (long, spec) in &schema.options {
        if !VALID_TYPES.contains(&spec.ty.as_str()) {
            return Err(format!(
                "option --{}: unknown type '{}' (valid: {})",
                long,
                spec.ty,
                VALID_TYPES.join(", ")
            ));
        }
        if RESERVED_LONGS.contains(&long.as_str()) {
            return Err(format!(
                "option --{long} shadows a reserved fdl-level flag"
            ));
        }
        if let Some(s) = &spec.short {
            if s.chars().count() != 1 {
                return Err(format!(
                    "option --{long}: `short: \"{s}\"` must be a single character"
                ));
            }
            if RESERVED_SHORTS.contains(&s.as_str()) {
                return Err(format!(
                    "option --{long}: short -{s} shadows a reserved fdl-level flag"
                ));
            }
            if let Some(prev) = short_seen.insert(s.clone(), long.clone()) {
                return Err(format!(
                    "options --{prev} and --{long} both declare short -{s}"
                ));
            }
        }
    }

    // Args: check types, variadic-only-at-end, no-required-after-optional.
    let mut seen_optional = false;
    let mut name_seen: BTreeMap<String, ()> = BTreeMap::new();
    for (i, arg) in schema.args.iter().enumerate() {
        if !VALID_TYPES.contains(&arg.ty.as_str()) {
            return Err(format!(
                "arg <{}>: unknown type '{}' (valid: {})",
                arg.name,
                arg.ty,
                VALID_TYPES.join(", ")
            ));
        }
        if name_seen.insert(arg.name.clone(), ()).is_some() {
            return Err(format!("duplicate positional name <{}>", arg.name));
        }
        if arg.variadic && i != schema.args.len() - 1 {
            return Err(format!(
                "arg <{}>: variadic positional must be the last one",
                arg.name
            ));
        }
        let is_optional = !arg.required || arg.default.is_some();
        if arg.required && arg.default.is_some() {
            return Err(format!(
                "arg <{}>: `required: true` with a default is a contradiction",
                arg.name
            ));
        }
        if seen_optional && arg.required && arg.default.is_none() {
            return Err(format!(
                "arg <{}>: required positional cannot follow an optional one",
                arg.name
            ));
        }
        if is_optional {
            seen_optional = true;
        }
    }

    Ok(())
}

// ── Structured config sections ──────────────────────────────────────────
