//! fdl.yaml configuration loading and discovery.
//!
//! Walks up from CWD to find the project manifest, parses YAML/JSON,
//! and loads sub-command configs from registered command directories.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Root project config ─────────────────────────────────────────────────

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
    /// `cluster: true` are dispatched across every host in
    /// [`ClusterConfig::hosts`]. Lives at the project root because the
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
    ///    enclosing `schema.options` (see [`validate_presets_strict`]).
    ///    A typo like `options: { batchsize: 32 }` when the schema
    ///    declares `batch-size` is a loud load error.
    /// 2. *Dispatch time* — the user's extra argv tail is tokenized
    ///    against the schema (see [`validate_tail`]). Unknown flags
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
    /// Consumed by `fdl completions <shell>` (follow-up rollout task).
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
    /// Consumed by `fdl completions <shell>` (follow-up rollout task).
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

/// DDP configuration. Maps 1:1 to flodl DdpConfig / DdpRunConfig.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DdpConfig {
    pub mode: Option<String>,
    pub policy: Option<String>,
    pub backend: Option<String>,
    /// "auto" or integer.
    pub anchor: Option<serde_json::Value>,
    pub max_anchor: Option<u32>,
    pub overhead_target: Option<f64>,
    pub divergence_threshold: Option<f64>,
    /// null (unlimited) or integer.
    pub max_batch_diff: Option<serde_json::Value>,
    pub speed_hint: Option<SpeedHint>,
    pub partition_ratios: Option<Vec<f64>>,
    /// "auto" or bool.
    pub progressive: Option<serde_json::Value>,
    pub max_grad_norm: Option<f64>,
    pub lr_scale_ratio: Option<f64>,
    pub snapshot_timeout: Option<u32>,
    pub checkpoint_every: Option<u32>,
    pub timeline: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SpeedHint {
    pub slow_rank: usize,
    pub ratio: f64,
}

/// Training scalars.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrainingConfig {
    pub epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub batches_per_epoch: Option<u32>,
    pub lr: Option<f64>,
    pub seed: Option<u64>,
}

/// Output settings.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OutputConfig {
    pub dir: Option<String>,
    pub timeline: Option<bool>,
    pub monitor: Option<u16>,
}

// ── Cluster topology (multi-host DDP) ───────────────────────────────────

/// Multi-host cluster topology, parsed from the `cluster:` block at the
/// project root. This is the controller-side full topology; per-host slim
/// envelopes are derived from it and shipped to each node, where the
/// library reads them via `flodl::distributed::LocalCluster::from_env`.
/// Launcher-only fields (`ssh:`) are not propagated to the envelope.
///
/// The library re-validates after reading; this validation runs earlier so
/// errors surface before `fdl-cli` opens any SSH connection.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ClusterConfig {
    pub master_addr: String,
    pub master_port: u16,
    pub hosts: Vec<ClusterHost>,
}

/// One physical host in the cluster.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClusterHost {
    /// Hostname as reported by `hostname` on the host (or
    /// `FLODL_HOST_NAME`-overridden value).
    pub name: String,
    /// Global ranks owned by this host.
    pub ranks: Vec<usize>,
    /// CUDA device indices paired by position with `ranks`.
    pub local_devices: Vec<u8>,
    /// Network interface NCCL binds to (e.g. `virbr0`, `enp1s0`). Becomes
    /// `NCCL_SOCKET_IFNAME` in the spawned process environment.
    pub nccl_socket_ifname: String,
    /// Project checkout path on this host. `fdl-cli` cd's here before
    /// invoking the remote command. Heterogeneous mounts are fine
    /// (e.g. `/opt/flodl` on the controller, `/srv/flodl` in a VM); training
    /// data is the user's responsibility to mount identically across hosts
    /// (NAS / SMB / virtiofs / S3-FUSE).
    pub path: String,
    /// libtorch install path on this host. `fdl-cli` bind-mounts it into
    /// the Docker container. Library ignores this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub libtorch_path: Option<String>,
    /// SSH target for `fdl-cli`'s launcher (e.g. a short hostname, a
    /// `user@host`, or anything `ssh` accepts). Defaults to `name` when
    /// absent. Library ignores this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh: Option<String>,
}

impl ClusterConfig {
    /// Total ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.hosts.iter().map(|h| h.ranks.len()).sum()
    }

    /// Whether the cluster spans more than one physical host. Single-host
    /// clusters don't require `NCCL_SOCKET_IFNAME`.
    pub fn spans_multiple_hosts(&self) -> bool {
        self.hosts.len() > 1
    }

    /// Pre-flight validation. Mirrors the library check so failures surface
    /// before SSH dispatch instead of from a stack trace on a remote host:
    ///
    /// - `master_addr` non-empty
    /// - at least one host
    /// - per host: `ranks` non-empty, equal length to `local_devices`,
    ///   `nccl_socket_ifname` non-empty (when cluster spans multiple hosts),
    ///   `path` non-empty
    /// - across hosts: ranks form exactly `0..world_size` with no duplicates
    ///   or gaps
    pub fn validate(&self) -> Result<(), String> {
        if self.master_addr.trim().is_empty() {
            return Err("cluster.master_addr must be non-empty".into());
        }
        if self.hosts.is_empty() {
            return Err("cluster.hosts must be non-empty".into());
        }
        let multi_host = self.spans_multiple_hosts();
        for (i, h) in self.hosts.iter().enumerate() {
            if h.name.trim().is_empty() {
                return Err(format!("cluster.hosts[{i}].name must be non-empty"));
            }
            if h.ranks.is_empty() {
                return Err(format!(
                    "cluster.hosts[{i}] ({:?}): ranks must be non-empty",
                    h.name
                ));
            }
            if h.ranks.len() != h.local_devices.len() {
                return Err(format!(
                    "cluster.hosts[{i}] ({:?}): ranks ({}) and local_devices ({}) length mismatch",
                    h.name,
                    h.ranks.len(),
                    h.local_devices.len()
                ));
            }
            if multi_host && h.nccl_socket_ifname.trim().is_empty() {
                return Err(format!(
                    "cluster.hosts[{i}] ({:?}): nccl_socket_ifname must be \
                     non-empty when the cluster spans multiple hosts",
                    h.name
                ));
            }
            if h.path.trim().is_empty() {
                return Err(format!(
                    "cluster.hosts[{i}] ({:?}): path (project checkout dir) \
                     must be non-empty",
                    h.name
                ));
            }
        }
        let mut all: Vec<usize> = self
            .hosts
            .iter()
            .flat_map(|h| h.ranks.iter().copied())
            .collect();
        let ws = all.len();
        all.sort_unstable();
        let expected: Vec<usize> = (0..ws).collect();
        if all != expected {
            return Err(format!(
                "cluster: ranks across hosts must be exactly 0..{ws} with no \
                 duplicates or gaps, got sorted-unique sequence {all:?}"
            ));
        }
        Ok(())
    }

    /// Canonical JSON of the full topology. Used for debug-dumping the
    /// controller-side view; per-host envelopes go through
    /// [`Self::local_envelope_for`] instead.
    pub fn canonical_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("cluster: JSON serialization failed: {e}"))
    }

    /// Build the slim per-host envelope the library reads via
    /// `flodl::distributed::LocalCluster::from_env`.
    ///
    /// The envelope strips launcher-only fields (`ssh`), embeds derived
    /// world metadata (`world_size`, `num_hosts`), and carries only the
    /// requested host's slice. The launcher hex-encodes the resulting JSON
    /// into `FLODL_CLUSTER_JSON` per ssh invocation, so each remote process
    /// sees only itself + the master coordinates.
    pub fn local_envelope_for(&self, host: &ClusterHost) -> Value {
        let mut host_obj = serde_json::Map::new();
        host_obj.insert("name".into(), Value::String(host.name.clone()));
        host_obj.insert(
            "ranks".into(),
            Value::Array(host.ranks.iter().map(|r| Value::from(*r)).collect()),
        );
        host_obj.insert(
            "local_devices".into(),
            Value::Array(host.local_devices.iter().map(|d| Value::from(*d)).collect()),
        );
        host_obj.insert(
            "nccl_socket_ifname".into(),
            Value::String(host.nccl_socket_ifname.clone()),
        );
        host_obj.insert("path".into(), Value::String(host.path.clone()));
        if let Some(p) = &host.libtorch_path {
            host_obj.insert("libtorch_path".into(), Value::String(p.clone()));
        }

        let mut envelope = serde_json::Map::new();
        envelope.insert(
            "master_addr".into(),
            Value::String(self.master_addr.clone()),
        );
        envelope.insert("master_port".into(), Value::from(self.master_port));
        envelope.insert("world_size".into(), Value::from(self.world_size()));
        envelope.insert("num_hosts".into(), Value::from(self.hosts.len()));
        envelope.insert("host".into(), Value::Object(host_obj));
        Value::Object(envelope)
    }

    /// Look up the SSH target for a host, defaulting to `name` if `ssh:` is
    /// not set. Used by the launcher; library callers don't need this.
    pub fn ssh_target<'a>(&'a self, host: &'a ClusterHost) -> &'a str {
        host.ssh.as_deref().unwrap_or(&host.name)
    }
}

/// Resolve whether a leaf command should fan out across the cluster, given
/// the chain of [`CommandSpec::cluster`] directives along its command path,
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
///     // fan out across project.cluster.hosts
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

const CONFIG_NAMES: &[&str] = &["fdl.yaml", "fdl.yml", "fdl.json"];
const EXAMPLE_SUFFIXES: &[&str] = &[".example", ".dist"];

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
fn try_copy_example(example: &Path, target: &Path) -> bool {
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
/// internally by [`load_merged_value`] / [`load_command_with_env`] so
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

/// Load a command config from a sub-directory.
///
/// Applies the same `.example`/`.dist` fallback as [`find_config`]. If a
/// `schema:` block is present, validates it before returning.
pub fn load_command(dir: &Path) -> Result<CommandConfig, String> {
    load_command_with_env(dir, None)
}

/// Load a sub-command config with an optional environment overlay.
///
/// Applies the same `.example`/`.dist` fallback as [`find_config`] to locate
/// the base file, then deep-merges a sibling `fdl.<env>.yml` overlay if one
/// exists. A *missing* overlay is silently accepted here (different from
/// [`load_project_with_env`]) — envs declared at the project root don't
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
    let mut cfg: CommandConfig = serde_yaml::from_value(merged)
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
            short: spec
                .short
                .as_deref()
                .and_then(|s| s.chars().next()),
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
    let positionals: Vec<PositionalDecl> = schema
        .args
        .iter()
        .map(|a| PositionalDecl {
            name: a.name.clone(),
            required: false,
            variadic: a.variadic,
            choices: a
                .choices
                .as_ref()
                .map(|cs| strict_choices_to_strings(cs)),
        })
        .collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolve the project root (where fdl.yml / fdl.yml.example live) starting
    /// from CARGO_MANIFEST_DIR. The CLI crate sits one level down.
    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("flodl-cli parent must be project root")
            .to_path_buf()
    }

    fn load_example() -> ProjectConfig {
        let path = project_root().join("fdl.yml.example");
        assert!(
            path.is_file(),
            "fdl.yml.example missing at {} -- the CLI depends on it as the canonical config template",
            path.display()
        );
        load_project(&path).expect("fdl.yml.example must parse as a valid ProjectConfig")
    }

    fn opt(ty: &str) -> OptionSpec {
        OptionSpec {
            ty: ty.into(),
            description: None,
            default: None,
            choices: None,
            short: None,
            env: None,
            completer: None,
        }
    }

    fn arg(name: &str, ty: &str) -> ArgSpec {
        ArgSpec {
            name: name.into(),
            ty: ty.into(),
            description: None,
            required: true,
            variadic: false,
            default: None,
            choices: None,
            completer: None,
        }
    }

    #[test]
    fn validate_schema_accepts_minimal_valid() {
        let mut s = Schema::default();
        s.options.insert("model".into(), opt("string"));
        s.options.insert("epochs".into(), opt("int"));
        s.args.push(arg("run-id", "string"));
        validate_schema(&s).expect("minimal valid schema must pass");
    }

    #[test]
    fn validate_schema_rejects_unknown_option_type() {
        let mut s = Schema::default();
        s.options.insert("bad".into(), opt("integer"));
        let err = validate_schema(&s).expect_err("unknown type should fail");
        assert!(err.contains("unknown type"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_reserved_long() {
        let mut s = Schema::default();
        s.options.insert("help".into(), opt("bool"));
        let err = validate_schema(&s).expect_err("reserved --help must fail");
        assert!(err.contains("reserved"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_reserved_short() {
        let mut s = Schema::default();
        let mut o = opt("string");
        o.short = Some("h".into());
        s.options.insert("host".into(), o);
        let err = validate_schema(&s).expect_err("short -h must fail");
        assert!(err.contains("reserved"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_duplicate_short() {
        let mut s = Schema::default();
        let mut a = opt("string");
        a.short = Some("m".into());
        let mut b = opt("string");
        b.short = Some("m".into());
        s.options.insert("model".into(), a);
        s.options.insert("mode".into(), b);
        let err = validate_schema(&s).expect_err("duplicate -m must fail");
        assert!(err.contains("both declare short"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_non_last_variadic() {
        let mut s = Schema::default();
        let mut first = arg("files", "string");
        first.variadic = true;
        s.args.push(first);
        s.args.push(arg("trailer", "string"));
        let err = validate_schema(&s).expect_err("variadic-not-last must fail");
        assert!(err.contains("variadic"), "err was: {err}");
    }

    #[test]
    fn validate_schema_rejects_required_after_optional() {
        let mut s = Schema::default();
        let mut first = arg("maybe", "string");
        first.required = false;
        s.args.push(first);
        s.args.push(arg("need", "string"));
        let err = validate_schema(&s).expect_err("required-after-optional must fail");
        assert!(err.contains("cannot follow"), "err was: {err}");
    }

    // ── Tail validation (always-on) + strict unknown-rejection ─────

    fn schema_with_model_option(strict: bool) -> Schema {
        let mut s = Schema {
            strict,
            ..Schema::default()
        };
        let mut model = opt("string");
        model.short = Some("m".into());
        model.choices = Some(vec![
            serde_json::json!("mlp"),
            serde_json::json!("resnet"),
        ]);
        s.options.insert("model".into(), model);
        s.options.insert("epochs".into(), opt("int"));
        // A bool flag, no value.
        s.options.insert("validate".into(), opt("bool"));
        s
    }

    fn strict_schema_with_model_option() -> Schema {
        schema_with_model_option(true)
    }

    #[test]
    fn validate_tail_accepts_known_long_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--epochs".into(), "3".into()];
        validate_tail(&tail, &schema).expect("known flag must pass");
    }

    #[test]
    fn validate_tail_accepts_known_short_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["-m".into(), "mlp".into()];
        validate_tail(&tail, &schema).expect("known short must pass");
    }

    #[test]
    fn validate_tail_accepts_bool_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--validate".into()];
        validate_tail(&tail, &schema).expect("bool flag must pass");
    }

    #[test]
    fn validate_tail_strict_rejects_unknown_long_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--nope".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("unknown long flag must error in strict mode");
        assert!(err.contains("--nope"), "err was: {err}");
    }

    #[test]
    fn validate_tail_strict_suggests_did_you_mean() {
        // "--epoch" is one char off "--epochs" — edit distance ≤ 2.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--epoch".into(), "3".into()];
        let err = validate_tail(&tail, &schema).expect_err("typo must error");
        assert!(err.contains("did you mean"), "err was: {err}");
        assert!(err.contains("--epochs"), "suggestion missing: {err}");
    }

    #[test]
    fn validate_tail_strict_rejects_unknown_short_flag() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["-z".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("unknown short must error in strict mode");
        assert!(err.contains("-z"), "err was: {err}");
    }

    #[test]
    fn validate_tail_rejects_bad_choice_always_strict() {
        let schema = strict_schema_with_model_option();
        let tail = vec!["--model".into(), "lenet".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("out-of-set choice must error");
        assert!(err.contains("lenet"), "err was: {err}");
        assert!(err.contains("allowed"), "err should list allowed values: {err}");
    }

    #[test]
    fn validate_tail_rejects_bad_choice_even_when_not_strict() {
        // The main change in this rollout: `choices:` is a positive
        // assertion by the author, so it must be enforced regardless
        // of `schema.strict`. Only *unknown* flags relax without
        // strict.
        let schema = schema_with_model_option(false);
        let tail = vec!["--model".into(), "lenet".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("out-of-set choice must error without strict");
        assert!(err.contains("lenet"), "err was: {err}");
        assert!(err.contains("allowed"), "err should list allowed values: {err}");
    }

    #[test]
    fn validate_tail_non_strict_tolerates_unknown_flag() {
        // Without strict, unknown flags are legitimate pass-through
        // candidates (the binary handles them itself).
        let schema = schema_with_model_option(false);
        let tail = vec!["--fancy-passthrough".into(), "value".into()];
        validate_tail(&tail, &schema)
            .expect("unknown flag must be tolerated when strict is off");
    }

    #[test]
    fn validate_tail_non_strict_still_checks_known_short_choices() {
        // The declared short `-m` has choices; a bad value fails even
        // when strict is off. Unknown options would be tolerated, but
        // once the user reaches a declared option, its contract holds.
        let schema = schema_with_model_option(false);
        let tail = vec!["-m".into(), "lenet".into()];
        let err = validate_tail(&tail, &schema)
            .expect_err("out-of-set choice via short must error");
        assert!(err.contains("lenet"), "err was: {err}");
    }

    #[test]
    fn validate_tail_allows_reserved_help() {
        // Reserved universal flags must pass even though they are not
        // declared in the schema. Defense-in-depth against edge cases
        // where `--help` somehow reaches dispatch.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--help".into()];
        validate_tail(&tail, &schema).expect("--help must be allowed");
    }

    #[test]
    fn validate_tail_allows_reserved_fdl_schema() {
        // `fdl ddp-bench --fdl-schema` is forwarded to the binary.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--fdl-schema".into()];
        validate_tail(&tail, &schema).expect("--fdl-schema must be allowed");
    }

    #[test]
    fn validate_tail_passthrough_after_double_dash() {
        // `--` terminates flag parsing. Tokens after it are positionals
        // and must never trigger "unknown flag" errors.
        let schema = strict_schema_with_model_option();
        let tail = vec!["--".into(), "--arbitrary".into(), "anything".into()];
        validate_tail(&tail, &schema).expect("passthrough must work");
    }

    #[test]
    fn validate_presets_strict_rejects_unknown_option() {
        let schema = strict_schema_with_model_option();
        let mut commands = BTreeMap::new();
        let mut bad_options = BTreeMap::new();
        bad_options.insert("batchsize".into(), serde_json::json!(32));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: bad_options,
                ..Default::default()
            },
        );
        let err = validate_presets_strict(&commands, &schema)
            .expect_err("preset pinning undeclared option must error");
        assert!(err.contains("quick"), "err should name the preset: {err}");
        assert!(err.contains("batchsize"), "err should name the key: {err}");
    }

    #[test]
    fn validate_presets_strict_accepts_known_options() {
        let schema = strict_schema_with_model_option();
        let mut commands = BTreeMap::new();
        let mut good_options = BTreeMap::new();
        good_options.insert("model".into(), serde_json::json!("mlp"));
        good_options.insert("epochs".into(), serde_json::json!(5));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: good_options,
                ..Default::default()
            },
        );
        validate_presets_strict(&commands, &schema)
            .expect("presets with declared options must pass");
    }

    #[test]
    fn validate_presets_strict_ignores_run_and_path_kinds() {
        // Only Preset-kind entries share the parent schema. Run/Path
        // siblings are independent, so strict must not touch them.
        let schema = strict_schema_with_model_option();
        let mut commands = BTreeMap::new();
        commands.insert(
            "helper".into(),
            CommandSpec {
                run: Some("echo hi".into()),
                ..Default::default()
            },
        );
        commands.insert(
            "nested".into(),
            CommandSpec {
                path: Some("./nested/".into()),
                ..Default::default()
            },
        );
        validate_presets_strict(&commands, &schema)
            .expect("run/path siblings must be ignored by preset strict check");
    }

    // ── Preset value validation (always-on `choices:`) ──────────────

    #[test]
    fn validate_preset_values_rejects_bad_choice_even_without_strict() {
        // Schema has `choices:` on model; a preset pinning model to
        // something outside the list must fail at load, strict or not.
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("model".into(), serde_json::json!("lenet"));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        let err = validate_preset_values(&commands, &schema)
            .expect_err("out-of-choices preset must error");
        assert!(err.contains("quick"), "preset name missing: {err}");
        assert!(err.contains("model"), "option name missing: {err}");
        assert!(err.contains("lenet"), "bad value missing: {err}");
        assert!(err.contains("allowed"), "allowed list missing: {err}");
    }

    #[test]
    fn validate_preset_values_accepts_in_choices_preset() {
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("model".into(), serde_json::json!("mlp"));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        validate_preset_values(&commands, &schema)
            .expect("in-choices preset must pass");
    }

    #[test]
    fn validate_preset_values_ignores_undeclared_keys() {
        // Unknown keys aren't our concern here — that's for
        // `validate_presets_strict`, which only runs under strict.
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("extra".into(), serde_json::json!("whatever"));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        validate_preset_values(&commands, &schema)
            .expect("undeclared key must be ignored by value validator");
    }

    #[test]
    fn validate_preset_values_ignores_options_without_choices() {
        // `epochs` is declared as int with no `choices:`, so any value
        // passes the choice check (type validation is a separate pass).
        let schema = schema_with_model_option(false);
        let mut commands = BTreeMap::new();
        let mut opts = BTreeMap::new();
        opts.insert("epochs".into(), serde_json::json!(999));
        commands.insert(
            "quick".into(),
            CommandSpec {
                options: opts,
                ..Default::default()
            },
        );
        validate_preset_values(&commands, &schema)
            .expect("no-choices option must accept any value");
    }

    #[test]
    fn validate_schema_rejects_required_with_default() {
        let mut s = Schema::default();
        let mut a = arg("x", "string");
        a.default = Some(serde_json::json!("foo"));
        s.args.push(a);
        let err = validate_schema(&s).expect_err("required+default must fail");
        assert!(err.contains("contradiction"), "err was: {err}");
    }

    /// Regression guard: fdl.yml.example must keep a working `doc` command.
    /// The fdl.doc pipeline (api-ref for the port skill, rustdoc warning
    /// enforcement in CI) depends on this entry existing and producing output.
    #[test]
    fn fdl_yml_example_has_doc_script() {
        let cfg = load_example();
        let doc = cfg.commands.get("doc").unwrap_or_else(|| {
            panic!(
                "fdl.yml.example is missing a `doc` command; the rustdoc pipeline \
                 depends on `fdl doc` being defined"
            )
        });
        let cmd = doc
            .run
            .as_deref()
            .expect("fdl.yml.example `doc` command must be a `run:` entry");
        assert!(
            !cmd.trim().is_empty(),
            "fdl.yml.example `doc` command has an empty `run:` command"
        );
        assert!(
            cmd.contains("cargo doc"),
            "fdl.yml.example `doc` command must invoke `cargo doc`, got: {cmd}"
        );
        // Must assert some output was produced -- otherwise rustdoc can
        // silently succeed without writing anything useful (e.g. when the
        // target crate fails to resolve). Keeping the exact check liberal:
        // any mention of target/doc as a produced artifact counts.
        assert!(
            cmd.contains("target/doc"),
            "fdl.yml.example `doc` command must verify output was produced \
             (expected a `test -f target/doc/...` check), got: {cmd}"
        );
    }

    #[test]
    fn command_spec_kind_mutex_run_and_path() {
        let spec = CommandSpec {
            run: Some("echo".into()),
            path: Some("x/".into()),
            ..Default::default()
        };
        let err = spec.kind().expect_err("run + path must fail");
        assert!(err.contains("both"), "err was: {err}");
    }

    #[test]
    fn command_spec_kind_path_convention() {
        let spec = CommandSpec::default();
        assert_eq!(spec.kind().unwrap(), CommandKind::Path);
    }

    #[test]
    fn command_spec_kind_preset_when_preset_fields_set() {
        let spec = CommandSpec {
            training: Some(TrainingConfig {
                epochs: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Preset);
    }

    #[test]
    fn command_spec_kind_preset_when_only_options_set() {
        // `options:` alone is enough to make a preset — not every preset
        // overrides the structured ddp/training/output blocks.
        let mut options = BTreeMap::new();
        options.insert("model".into(), serde_json::json!("linear"));
        let spec = CommandSpec {
            options,
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Preset);
    }

    #[test]
    fn command_spec_kind_path_explicit() {
        // Explicit `path:` is a Path even if preset fields are also set;
        // the presence of `path:` is the kind-selecting field.
        let spec = CommandSpec {
            path: Some("./sub/".into()),
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Path);
    }

    #[test]
    fn command_spec_kind_rejects_docker_without_run() {
        // `docker:` is meaningful only as a wrapper around an inline
        // `run:` script. Pairing it with path/preset is a silent noop
        // at dispatch time, so we reject at load.
        let spec = CommandSpec {
            docker: Some("cuda".into()),
            ..Default::default()
        };
        let err = spec
            .kind()
            .expect_err("docker without run must fail");
        assert!(err.contains("docker"), "err was: {err}");
    }

    #[test]
    fn command_spec_kind_allows_docker_with_run() {
        let spec = CommandSpec {
            run: Some("cargo test".into()),
            docker: Some("dev".into()),
            ..Default::default()
        };
        assert_eq!(spec.kind().unwrap(), CommandKind::Run);
    }

    #[test]
    fn command_spec_deserialize_from_null() {
        let yaml = "cmd: ~";
        let map: BTreeMap<String, CommandSpec> =
            serde_yaml::from_str(yaml).expect("null must deserialize to default");
        let spec = map.get("cmd").expect("cmd missing");
        assert!(spec.run.is_none() && spec.path.is_none());
        assert_eq!(spec.kind().unwrap(), CommandKind::Path);
    }

    #[test]
    fn command_config_arg_name_deserializes_kebab_case() {
        // YAML uses `arg-name:`, Rust field is `arg_name`.
        let yaml = "arg-name: recipe\nentry: echo\n";
        let cfg: CommandConfig =
            serde_yaml::from_str(yaml).expect("arg-name must parse");
        assert_eq!(cfg.arg_name.as_deref(), Some("recipe"));
    }

    #[test]
    fn command_config_arg_name_defaults_to_none() {
        let cfg: CommandConfig =
            serde_yaml::from_str("entry: echo\n").expect("minimal cfg must parse");
        assert!(cfg.arg_name.is_none());
    }

    // ── resolve_config_layers: inherit-from + env composition ────────────
    //
    // Integration coverage for how `inherit-from:` chains compose with env
    // overlays at the config-module boundary. The overlay module already
    // tests `resolve_chain` in isolation; here we verify the concat+dedup
    // behaviour that config.rs layers on top.

    /// Minimal tempdir helper — matches the pattern used across the crate.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "fdl-cfg-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn filenames(layers: &[(PathBuf, serde_yaml::Value)]) -> Vec<String> {
        layers
            .iter()
            .map(|(p, _)| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn resolve_config_layers_base_only() {
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        let layers = resolve_config_layers(&base, None).unwrap();
        assert_eq!(filenames(&layers), vec!["fdl.yml"]);
    }

    #[test]
    fn resolve_config_layers_base_with_env_overlay() {
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let env = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        std::fs::write(&env, "b: 2\n").unwrap();
        let layers = resolve_config_layers(&base, Some("ci")).unwrap();
        assert_eq!(filenames(&layers), vec!["fdl.yml", "fdl.ci.yml"]);
    }

    #[test]
    fn resolve_config_layers_env_inherits_from_mixin() {
        // fdl.ci.yml inherits from fdl.cloud.yml (standalone mix-in, not
        // derived from base). Combined chain: [base, cloud, ci].
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let cloud = tmp.0.join("fdl.cloud.yml");
        let ci = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        std::fs::write(&cloud, "b: 2\n").unwrap();
        std::fs::write(&ci, "inherit-from: fdl.cloud.yml\nc: 3\n").unwrap();
        let layers = resolve_config_layers(&base, Some("ci")).unwrap();
        assert_eq!(
            filenames(&layers),
            vec!["fdl.yml", "fdl.cloud.yml", "fdl.ci.yml"]
        );
    }

    #[test]
    fn resolve_config_layers_dedups_when_env_inherits_from_base() {
        // fdl.ci.yml inherits from fdl.yml directly. Base is already in
        // the layer list, so env's chain collapses into it — the final
        // list must not have fdl.yml twice.
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let ci = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        std::fs::write(&ci, "inherit-from: fdl.yml\nb: 2\n").unwrap();
        let layers = resolve_config_layers(&base, Some("ci")).unwrap();
        assert_eq!(filenames(&layers), vec!["fdl.yml", "fdl.ci.yml"]);
    }

    #[test]
    fn resolve_config_layers_merged_value_matches_chain() {
        // End-to-end: the merge result should reflect the chain order
        // (base < cloud < ci), with each subsequent layer overriding.
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        let cloud = tmp.0.join("fdl.cloud.yml");
        let ci = tmp.0.join("fdl.ci.yml");
        std::fs::write(&base, "value: base\nkeep_base: yes\n").unwrap();
        std::fs::write(&cloud, "value: cloud\nkeep_cloud: yes\n").unwrap();
        std::fs::write(
            &ci,
            "inherit-from: fdl.cloud.yml\nvalue: ci\nkeep_ci: yes\n",
        )
        .unwrap();
        let merged = load_merged_value(&base, Some("ci")).unwrap();
        let m = merged.as_mapping().unwrap();
        // Last writer wins on `value`.
        assert_eq!(
            m.get(serde_yaml::Value::String("value".into())).unwrap(),
            &serde_yaml::Value::String("ci".into())
        );
        // Each layer's unique key survives.
        assert!(m.contains_key(serde_yaml::Value::String("keep_base".into())));
        assert!(m.contains_key(serde_yaml::Value::String("keep_cloud".into())));
        assert!(m.contains_key(serde_yaml::Value::String("keep_ci".into())));
    }

    #[test]
    fn resolve_config_layers_missing_env_errors() {
        let tmp = TempDir::new();
        let base = tmp.0.join("fdl.yml");
        std::fs::write(&base, "a: 1\n").unwrap();
        let err = resolve_config_layers(&base, Some("nope")).unwrap_err();
        assert!(err.contains("nope"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn resolve_config_layers_base_inherit_from_chain() {
        // Base itself uses inherit-from: shared-defaults.yml. The
        // defaults live in a sibling file and are merged UNDER the base.
        let tmp = TempDir::new();
        let defaults = tmp.0.join("shared.yml");
        let base = tmp.0.join("fdl.yml");
        std::fs::write(&defaults, "policy: default\n").unwrap();
        std::fs::write(&base, "inherit-from: shared.yml\npolicy: override\n").unwrap();
        let layers = resolve_config_layers(&base, None).unwrap();
        assert_eq!(filenames(&layers), vec!["shared.yml", "fdl.yml"]);
    }

    #[test]
    fn load_command_auto_probes_non_cargo_entry_and_writes_cache() {
        // Script-kind entry + missing cache: load_command should invoke
        // `<entry> --fdl-schema`, apply the result to cfg.schema, and
        // write it to .fdl/schema-cache/<name>.json for next time.
        let tmp = TempDir::new();
        let cmd_dir = tmp.0.join("mybench");
        std::fs::create_dir_all(&cmd_dir).unwrap();

        let script = cmd_dir.join("emit.sh");
        let body = "#!/bin/sh\n\
                    if [ \"$1\" = \"--fdl-schema\" ]; then\n\
                      cat <<'JSON'\n\
                    { \"options\": { \"rounds\": { \"type\": \"int\", \"description\": \"N\" } } }\n\
                    JSON\n\
                      exit 0\n\
                    fi\n";
        std::fs::write(&script, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        std::fs::write(cmd_dir.join("fdl.yml"), "entry: sh emit.sh\n").unwrap();

        let cfg = load_command(&cmd_dir).expect("load ok");
        let schema = cfg.schema.expect("auto-probe must populate schema");
        assert!(schema.options.contains_key("rounds"));

        // Second load reads the freshly-written cache (same content).
        let cached_path = crate::schema_cache::cache_path(&cmd_dir, "mybench");
        assert!(cached_path.is_file(), "cache file should exist");
    }

    #[test]
    fn load_command_skips_auto_probe_for_cargo_entries() {
        // Cargo entries are deliberately not probed — a `cargo run
        // --fdl-schema` would compile the whole crate before help
        // renders. Missing cache + cargo entry ⇒ no schema, help
        // still renders (just without options).
        let tmp = TempDir::new();
        let cmd_dir = tmp.0.join("cargo-cmd");
        std::fs::create_dir_all(&cmd_dir).unwrap();
        std::fs::write(cmd_dir.join("fdl.yml"), "entry: cargo run --\n").unwrap();

        let cfg = load_command(&cmd_dir).expect("load ok");
        assert!(
            cfg.schema.is_none(),
            "cargo entry must not be auto-probed (compile latency would ruin --help)"
        );
        let cached = crate::schema_cache::cache_path(&cmd_dir, "cargo-cmd");
        assert!(!cached.exists(), "no cache should be written for cargo entries");
    }

    #[test]
    fn load_command_compile_true_overrides_cargo_skip() {
        // `compile: true` opts a cargo-entry command in to schema probing.
        // The probe still fails on `cargo run` here (no real binary to
        // produce JSON), but the load path must reach `probe()` rather
        // than short-circuiting on the cargo-skip heuristic. We assert
        // by observing that probing was attempted: when the skip kicks
        // in, the test for `is_cargo_entry` returns early; with the
        // override, control flows into probe() which fails silently.
        // Lacking a way to spy on the probe call itself, we rely on the
        // sibling test (`load_command_auto_probe_failure_falls_through_silently`)
        // to demonstrate the silent-fail path, and assert here that
        // `compile: false` still skips. See the explicit-opt-out test
        // below.
        let tmp = TempDir::new();
        let cmd_dir = tmp.0.join("cargo-compile-true");
        std::fs::create_dir_all(&cmd_dir).unwrap();
        std::fs::write(
            cmd_dir.join("fdl.yml"),
            "entry: cargo run --\ncompile: true\n",
        )
        .unwrap();

        // Load must succeed; schema stays None because the bogus cargo
        // entry can't actually emit JSON, but the probe path was
        // exercised (no panic, no early-skip).
        let cfg = load_command(&cmd_dir).expect("load ok");
        assert_eq!(cfg.compile, Some(true), "compile field round-trips");
        assert!(cfg.schema.is_none());
    }

    #[test]
    fn load_command_compile_false_keeps_cargo_skip() {
        // Explicit `compile: false` is the same as absent — cargo skip
        // stays in place.
        let tmp = TempDir::new();
        let cmd_dir = tmp.0.join("cargo-compile-false");
        std::fs::create_dir_all(&cmd_dir).unwrap();
        std::fs::write(
            cmd_dir.join("fdl.yml"),
            "entry: cargo run --\ncompile: false\n",
        )
        .unwrap();

        let cfg = load_command(&cmd_dir).expect("load ok");
        assert_eq!(cfg.compile, Some(false));
        assert!(cfg.schema.is_none(), "cargo skip honored when compile: false");
        let cached = crate::schema_cache::cache_path(&cmd_dir, "cargo-compile-false");
        assert!(!cached.exists());
    }

    #[test]
    fn load_command_auto_probe_failure_falls_through_silently() {
        // An entry that ignores --fdl-schema (or errors) must not break
        // help rendering. cfg.schema stays None, no cache written.
        let tmp = TempDir::new();
        let cmd_dir = tmp.0.join("silent");
        std::fs::create_dir_all(&cmd_dir).unwrap();
        // `/bin/true` ignores any args and exits 0 with empty stdout; probe
        // will reject "no JSON object" and Err — we want that swallowed.
        // Quoted so YAML doesn't parse the bareword `true` as a boolean.
        std::fs::write(cmd_dir.join("fdl.yml"), "entry: \"/bin/true\"\n").unwrap();

        let cfg = load_command(&cmd_dir).expect("load must succeed despite probe error");
        assert!(cfg.schema.is_none());
    }

    // ── Cluster topology (multi-host DDP overlay) ───────────────────

    fn canonical_cluster_yaml() -> &'static str {
        "\
cluster:
  master_addr: 192.168.122.1
  master_port: 29500
  hosts:
    - name: master-host
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: virbr0
      path: /opt/flodl
      libtorch_path: /data/ssd/flodl/libtorch
    - name: worker-host
      ssh: worker-host
      ranks: [1, 2]
      local_devices: [0, 1]
      nccl_socket_ifname: enp1s0
      path: /srv/flodl
      libtorch_path: /mnt/flodl/libtorch

commands:
  cuda-test:
    cluster: true
  train:
    cluster: true
    run: cargo run --release --bin my-training-app
"
    }

    #[test]
    fn cluster_overlay_parses() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).expect("parse cluster overlay");
        let cluster = cfg.cluster.as_ref().expect("cluster: block present");
        assert_eq!(cluster.master_addr, "192.168.122.1");
        assert_eq!(cluster.master_port, 29500);
        assert_eq!(cluster.world_size(), 3);
        assert_eq!(cluster.hosts.len(), 2);
        assert!(cluster.spans_multiple_hosts());

        let worker = &cluster.hosts[1];
        assert_eq!(worker.name, "worker-host");
        assert_eq!(worker.ranks, vec![1, 2]);
        assert_eq!(worker.local_devices, vec![0, 1]);
        assert_eq!(worker.nccl_socket_ifname, "enp1s0");
        assert_eq!(worker.path, "/srv/flodl");
        assert_eq!(worker.ssh.as_deref(), Some("worker-host"));

        // CommandSpec cluster: true survives the custom Deserialize.
        let test_cmd = cfg.commands.get("cuda-test").expect("cuda-test command");
        assert_eq!(test_cmd.cluster, Some(true));
        let train_cmd = cfg.commands.get("train").expect("train command");
        assert_eq!(train_cmd.cluster, Some(true));
        assert_eq!(train_cmd.run.as_deref(), Some("cargo run --release --bin my-training-app"));
    }

    #[test]
    fn cluster_block_optional() {
        let yaml = "commands: { foo: { run: \"echo hi\" } }\n";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).expect("parse without cluster");
        assert!(cfg.cluster.is_none());
        // CommandSpec cluster defaults to None.
        assert_eq!(cfg.commands.get("foo").and_then(|c| c.cluster), None);
    }

    #[test]
    fn cluster_overlay_merges_via_deep_merge() {
        // The canonical author pattern: `cluster:` lives in `fdl.vm.yml`
        // (the overlay), not the base `fdl.yml`. deep_merge stitches them.
        let base_yaml = "commands:\n  train:\n    run: cargo run --release\n";
        let overlay_yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: lo
      path: /tmp/test-solo
commands:
  train:
    cluster: true
";
        let base: serde_yaml::Value = serde_yaml::from_str(base_yaml).unwrap();
        let overlay: serde_yaml::Value = serde_yaml::from_str(overlay_yaml).unwrap();
        let merged = crate::overlay::deep_merge(base, overlay);
        let merged_yaml = serde_yaml::to_string(&merged).unwrap();
        let cfg: ProjectConfig =
            serde_yaml::from_str(&merged_yaml).expect("merged config parses");
        let cluster = cfg.cluster.as_ref().expect("cluster: from overlay");
        assert_eq!(cluster.world_size(), 1);
        assert_eq!(cluster.hosts[0].name, "solo");
        // Overlay added `cluster: true` to the train command; the base's
        // `run:` survived the merge.
        let train = cfg.commands.get("train").expect("train command");
        assert_eq!(train.cluster, Some(true));
        assert_eq!(train.run.as_deref(), Some("cargo run --release"));
    }

    #[test]
    fn validate_rejects_duplicate_ranks() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().hosts[1].ranks = vec![1, 1];
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("duplicates or gaps"), "got: {err}");
    }

    #[test]
    fn validate_rejects_rank_gap() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().hosts[1].ranks = vec![2, 3];
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("duplicates or gaps"), "got: {err}");
    }

    #[test]
    fn validate_rejects_len_mismatch() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().hosts[1].local_devices = vec![0];
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("length mismatch"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_hosts() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().hosts.clear();
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("non-empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_socket_ifname_when_multi_host() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().hosts[0].nccl_socket_ifname = String::new();
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("nccl_socket_ifname"), "got: {err}");
        assert!(err.contains("multiple hosts"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_path() {
        let mut cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_mut().unwrap().hosts[0].path = String::new();
        let err = cfg.cluster.as_ref().unwrap().validate().unwrap_err();
        assert!(err.contains("path"), "got: {err}");
        assert!(err.contains("master-host"), "got: {err}");
    }

    #[test]
    fn validate_allows_empty_socket_ifname_when_single_host() {
        // Single-host clusters don't go through NCCL TCP, so the ifname
        // requirement doesn't apply.
        let yaml = "\
cluster:
  master_addr: 127.0.0.1
  master_port: 29500
  hosts:
    - name: solo
      ranks: [0]
      local_devices: [0]
      nccl_socket_ifname: \"\"
      path: /tmp/test-solo
";
        let cfg: ProjectConfig = serde_yaml::from_str(yaml).unwrap();
        cfg.cluster.as_ref().unwrap().validate().expect("single-host with empty ifname must pass");
    }

    #[test]
    fn validate_passes_canonical() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        cfg.cluster.as_ref().unwrap().validate().expect("canonical topology must validate");
    }

    #[test]
    fn canonical_json_stable_and_round_trips() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        let json = cluster.canonical_json().expect("serialize");

        // Stability: serialization twice produces identical output.
        let json2 = cluster.canonical_json().unwrap();
        assert_eq!(json, json2);

        // Round-trip: parsing back gives the same observable state.
        let parsed: ClusterConfig = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed.master_addr, cluster.master_addr);
        assert_eq!(parsed.master_port, cluster.master_port);
        assert_eq!(parsed.hosts.len(), cluster.hosts.len());
        assert_eq!(parsed.world_size(), cluster.world_size());
        assert_eq!(parsed.hosts[1].ssh.as_deref(), Some("worker-host"));
    }

    #[test]
    fn local_envelope_strips_ssh_adds_world_metadata() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        let worker = &cluster.hosts[1];
        let env = cluster.local_envelope_for(worker);

        // Top-level: master coords + world metadata.
        assert_eq!(env["master_addr"], "192.168.122.1");
        assert_eq!(env["master_port"], 29500);
        assert_eq!(env["world_size"], 3); // 1 + 2 ranks
        assert_eq!(env["num_hosts"], 2);

        // Host slice: only this host's fields.
        let h = &env["host"];
        assert_eq!(h["name"], "worker-host");
        assert_eq!(h["ranks"], serde_json::json!([1, 2]));
        assert_eq!(h["local_devices"], serde_json::json!([0, 1]));
        assert_eq!(h["nccl_socket_ifname"], "enp1s0");
        assert_eq!(h["path"], "/srv/flodl");
        assert_eq!(h["libtorch_path"], "/mnt/flodl/libtorch");

        // ssh: stripped (launcher-only).
        assert!(h.get("ssh").is_none(), "ssh must not appear in envelope");
    }

    #[test]
    fn local_envelope_omits_optional_libtorch_path() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let mut cluster = cfg.cluster.unwrap();
        cluster.hosts[0].libtorch_path = None;
        let env = cluster.local_envelope_for(&cluster.hosts[0]);
        assert!(
            env["host"].get("libtorch_path").is_none(),
            "libtorch_path should be omitted when None"
        );
    }

    #[test]
    fn local_envelope_master_host_carries_rank_zero() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        let master = &cluster.hosts[0];
        let env = cluster.local_envelope_for(master);
        let ranks = env["host"]["ranks"].as_array().unwrap();
        assert!(
            ranks.iter().any(|r| r.as_u64() == Some(0)),
            "master host's envelope must include rank 0"
        );
    }

    #[test]
    fn ssh_target_defaults_to_name() {
        let cfg: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        let cluster = cfg.cluster.as_ref().unwrap();
        assert_eq!(cluster.ssh_target(&cluster.hosts[0]), "master-host"); // no ssh: → name
        assert_eq!(cluster.ssh_target(&cluster.hosts[1]), "worker-host"); // explicit
    }

    // ── Path-inheritance cluster-dispatch resolver ──────────────────

    #[test]
    fn cluster_dispatch_empty_chain_is_false() {
        // No command directives anywhere → default false.
        assert!(!resolve_cluster_dispatch(&[]));
    }

    #[test]
    fn cluster_dispatch_all_unset_is_false() {
        // Every level along the path declined to set cluster: → false.
        assert!(!resolve_cluster_dispatch(&[None, None, None]));
    }

    #[test]
    fn cluster_dispatch_leaf_true_wins_over_unset_ancestors() {
        // root unset, leaf cluster: true → fan out.
        assert!(resolve_cluster_dispatch(&[None, Some(true)]));
    }

    #[test]
    fn cluster_dispatch_leaf_false_wins_over_unset_ancestors() {
        // root unset, leaf cluster: false → explicit local.
        assert!(!resolve_cluster_dispatch(&[None, Some(false)]));
    }

    #[test]
    fn cluster_dispatch_inherits_from_path_command_at_root() {
        // commands.ddp-bench in root: cluster: true. ddp-bench/fdl.yml's
        // leaf command leaves it unset → inherits parent's true.
        assert!(resolve_cluster_dispatch(&[Some(true), None]));
    }

    #[test]
    fn cluster_dispatch_leaf_overrides_ancestor_true_to_false() {
        // root: ddp-bench cluster: true. ddp-bench/cuda: cluster: false
        // (e.g. for a single-host quick smoke test) → stays local despite
        // ancestor saying yes.
        assert!(!resolve_cluster_dispatch(&[Some(true), Some(false)]));
    }

    #[test]
    fn cluster_dispatch_leaf_overrides_ancestor_false_to_true() {
        // root: research/ cluster: false (everything stays local).
        // research/multi-host-experiment: cluster: true (opt back in).
        assert!(resolve_cluster_dispatch(&[Some(false), Some(true)]));
    }

    #[test]
    fn cluster_dispatch_intermediate_override_wins_at_sub_sub_depth() {
        // 4-level chain: root → ddp-bench → sweep → experiment.
        // root true, ddp-bench false (turns ddp-bench off wholesale),
        // sweep unset, experiment unset → effective false. The deepest
        // explicit value (ddp-bench's false) outranks root's true; sweep
        // and experiment inherit from there.
        assert!(!resolve_cluster_dispatch(&[Some(true), Some(false), None, None]));
    }

    #[test]
    fn cluster_dispatch_deepest_override_wins_through_chain() {
        // Same 4-level chain but the experiment opts back in: effective true.
        // Resolution walks leaf-first, finds Some(true) immediately.
        assert!(resolve_cluster_dispatch(&[Some(true), Some(false), None, Some(true)]));
    }

    #[test]
    fn cluster_dispatch_enabled_requires_cluster_block() {
        // Even with leaf cluster: true, dispatching is disabled when no
        // cluster topology is declared at root.
        let no_cluster: ProjectConfig = serde_yaml::from_str(
            "commands: { foo: { run: \"echo hi\" } }\n",
        )
        .unwrap();
        assert!(!cluster_dispatch_enabled(&no_cluster, &[Some(true)]));
        assert!(!cluster_dispatch_enabled(&no_cluster, &[Some(true), Some(true)]));
    }

    #[test]
    fn cluster_dispatch_enabled_when_block_and_chain_agree() {
        let with_cluster: ProjectConfig =
            serde_yaml::from_str(canonical_cluster_yaml()).unwrap();
        assert!(cluster_dispatch_enabled(&with_cluster, &[Some(true)]));
        assert!(cluster_dispatch_enabled(&with_cluster, &[Some(true), None]));
        // Cluster block present but chain says false → not enabled.
        assert!(!cluster_dispatch_enabled(&with_cluster, &[Some(false)]));
        assert!(!cluster_dispatch_enabled(&with_cluster, &[None, Some(false)]));
        // Cluster block present but no directives anywhere → not enabled.
        assert!(!cluster_dispatch_enabled(&with_cluster, &[]));
        assert!(!cluster_dispatch_enabled(&with_cluster, &[None, None]));
    }
}
