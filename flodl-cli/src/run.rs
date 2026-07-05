//! Command resolution and execution.
//!
//! Merges structured config sections into CLI arguments, resolves named
//! command presets, and spawns the target process (directly or through
//! Docker when a `docker:` service is declared).

use std::collections::BTreeMap;
use std::path::Path;
use std::process::{ExitCode, Stdio};

use crate::builtins;
use crate::cli_error;
use crate::config::{self, ArgSpec, CommandConfig, OptionSpec, ResolvedConfig, Schema};
use crate::libtorch;
use crate::style;

// ── Config to CLI args ──────────────────────────────────────────────────

/// Translate a resolved config into CLI arguments for the entry point.
pub fn config_to_args(resolved: &ResolvedConfig) -> Vec<String> {
    let mut args = Vec::new();

    // DDP section
    let d = &resolved.ddp;
    push_opt(&mut args, "--mode", &d.mode);
    push_opt(&mut args, "--policy", &d.policy);
    push_opt(&mut args, "--backend", &d.backend);
    push_value(&mut args, "--anchor", &d.anchor);
    push_num(&mut args, "--max-anchor", &d.max_anchor);
    push_float(&mut args, "--overhead-target", &d.overhead_target);
    push_float(&mut args, "--divergence-threshold", &d.divergence_threshold);
    push_value(&mut args, "--max-batch-diff", &d.max_batch_diff);
    push_float(&mut args, "--max-grad-norm", &d.max_grad_norm);
    push_num(&mut args, "--snapshot-timeout", &d.snapshot_timeout);
    push_num(&mut args, "--checkpoint-every", &d.checkpoint_every);
    push_value(&mut args, "--progressive", &d.progressive);
    if let Some(hint) = &d.speed_hint {
        args.push("--speed-hint".into());
        args.push(format!("{}:{}", hint.slow_rank, hint.ratio));
    }
    if let Some(ratios) = &d.partition_ratios {
        let s: Vec<String> = ratios.iter().map(|r| format!("{r}")).collect();
        args.push("--partition-ratios".into());
        args.push(s.join(","));
    }
    if let Some(ratio) = d.lr_scale_ratio {
        args.push("--lr-scale-ratio".into());
        args.push(format!("{ratio}"));
    }
    if d.timeline == Some(true) {
        args.push("--timeline".into());
    }

    // Training section
    let t = &resolved.training;
    push_num(&mut args, "--epochs", &t.epochs);
    push_num(&mut args, "--batch-size", &t.batch_size);
    push_num(&mut args, "--batches", &t.batches_per_epoch);
    push_float(&mut args, "--lr", &t.lr);
    push_num(&mut args, "--seed", &t.seed);

    // Output section
    let o = &resolved.output;
    push_opt(&mut args, "--output", &o.dir);
    push_num(&mut args, "--monitor", &o.monitor);

    // Pass-through options
    for (key, val) in &resolved.options {
        let flag = format!("--{}", key.replace('_', "-"));
        match val {
            serde_json::Value::Bool(true) => args.push(flag),
            serde_json::Value::Bool(false) => {}
            serde_json::Value::Null => {}
            other => {
                args.push(flag);
                args.push(value_to_string(other));
            }
        }
    }

    args
}

fn push_opt(args: &mut Vec<String>, flag: &str, val: &Option<String>) {
    if let Some(v) = val {
        args.push(flag.into());
        args.push(v.clone());
    }
}

fn push_num<T: std::fmt::Display>(args: &mut Vec<String>, flag: &str, val: &Option<T>) {
    if let Some(v) = val {
        args.push(flag.into());
        args.push(v.to_string());
    }
}

fn push_float(args: &mut Vec<String>, flag: &str, val: &Option<f64>) {
    if let Some(v) = val {
        args.push(flag.into());
        args.push(format!("{v}"));
    }
}

fn push_value(args: &mut Vec<String>, flag: &str, val: &Option<serde_json::Value>) {
    if let Some(v) = val {
        match v {
            serde_json::Value::Null => {}
            other => {
                args.push(flag.into());
                args.push(value_to_string(other));
            }
        }
    }
}

fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

// ── Docker detection ────────────────────────────────────────────────────

/// Check if we're already running inside a Docker container.
fn inside_docker() -> bool {
    Path::new("/.dockerenv").exists()
}

/// Default container path we assume the host project root is mounted
/// at when `docker-compose.yml` is missing or can't be parsed.
/// Matches the convention `fdl init` writes into every generated
/// compose service (`.:/workspace`).
const DEFAULT_CONTAINER_PROJECT_ROOT: &str = "/workspace";

/// Per-process cache of the container-side project-root path, keyed by
/// docker-compose service name. Populated lazily on first lookup and
/// reused for the life of the `fdl` invocation. `docker-compose.yml` is
/// user-edited and version-controlled, so re-parsing once per
/// invocation is cheap — the cache only avoids re-parsing *within* a
/// single invocation.
static COMPOSE_MOUNT_CACHE: std::sync::OnceLock<
    std::collections::HashMap<String, String>,
> = std::sync::OnceLock::new();

/// Resolve the absolute container path where the host project root is
/// mounted inside `service`.
///
/// Reads `docker-compose.yml` at `project_root` once per process and
/// caches the `service → container_path` mapping. Falls back to
/// [`DEFAULT_CONTAINER_PROJECT_ROOT`] when the compose file is missing,
/// unparseable, or doesn't declare a matching bind mount for `.`.
///
/// This is what lets `exec_command` generate `cd <container-path>`
/// prefixes that work regardless of the service's `working_dir:` —
/// e.g. flodl-hf's `hf-parity` service uses
/// `working_dir: /workspace/flodl-hf` so Python parity scripts can
/// `import` sibling helpers, and a naive relative `cd flodl-hf/convert`
/// would resolve to the non-existent
/// `/workspace/flodl-hf/flodl-hf/convert`.
fn container_project_root(project_root: &Path, service: &str) -> String {
    let cache = COMPOSE_MOUNT_CACHE
        .get_or_init(|| parse_compose_project_mounts(project_root));
    cache
        .get(service)
        .cloned()
        .unwrap_or_else(|| DEFAULT_CONTAINER_PROJECT_ROOT.to_string())
}

/// Parse `<project_root>/docker-compose.yml` and return a map of
/// `service → container-mount-path` for every service that bind-mounts
/// the project root (host `.` or `./`).
///
/// Handles both short-form (`".:/workspace"`) and long-form
/// (`{ type: bind, source: ., target: /workspace }`) volume entries.
/// Read errors, parse errors, and missing sections all yield an empty
/// map — callers fall back to the convention.
fn parse_compose_project_mounts(
    project_root: &Path,
) -> std::collections::HashMap<String, String> {
    let compose_path = project_root.join("docker-compose.yml");
    let text = match std::fs::read_to_string(&compose_path) {
        Ok(t) => t,
        Err(_) => return std::collections::HashMap::new(),
    };
    let doc: serde_yaml::Value = match serde_yaml::from_str(&text) {
        Ok(d) => d,
        Err(_) => return std::collections::HashMap::new(),
    };
    let mut out = std::collections::HashMap::new();
    let services = match doc.get("services").and_then(|v| v.as_mapping()) {
        Some(s) => s,
        None => return out,
    };
    for (name, svc) in services {
        let svc_name = match name.as_str() {
            Some(s) => s,
            None => continue,
        };
        let volumes = match svc.get("volumes").and_then(|v| v.as_sequence()) {
            Some(v) => v,
            None => continue,
        };
        if let Some(container_path) = find_project_mount(volumes) {
            // Strip a trailing `/` so later `format!("{root}/{workdir}")`
            // never produces `//` in the middle of a path.
            let cleaned = container_path.trim_end_matches('/').to_string();
            let cleaned = if cleaned.is_empty() {
                "/".to_string()
            } else {
                cleaned
            };
            out.insert(svc_name.to_string(), cleaned);
        }
    }
    out
}

/// Inside a service's `volumes:` sequence, find the entry that
/// bind-mounts the project root (host path `.` or `./`) and return the
/// container-side target path.
fn find_project_mount(volumes: &[serde_yaml::Value]) -> Option<String> {
    for entry in volumes {
        if let Some(s) = entry.as_str() {
            // Short form: "host:container[:options]". Docker-compose's
            // short-form parser only splits on the first two `:` on
            // POSIX, but fdl-generated hosts are always `.` so naive
            // split-and-check works fine here.
            let mut parts = s.splitn(3, ':');
            let host = parts.next()?;
            let container = parts.next()?;
            if host == "." || host == "./" {
                return Some(container.to_string());
            }
        } else if let Some(m) = entry.as_mapping() {
            // Long form: { type: bind, source: ., target: /workspace }.
            let source = m
                .get(serde_yaml::Value::String("source".into()))
                .and_then(|v| v.as_str());
            let target = m
                .get(serde_yaml::Value::String("target".into()))
                .and_then(|v| v.as_str());
            if matches!(source, Some(".") | Some("./")) {
                if let Some(t) = target {
                    return Some(t.to_string());
                }
            }
        }
    }
    None
}

/// Resolve libtorch env vars from the project root, matching the Makefile logic:
///   LIBTORCH_HOST_PATH = ./libtorch/<active_variant>          (standalone)
///                      = <host.path>/libtorch/<host.arch>     (overlay)
///   LIBTORCH_CPU_PATH  = ./libtorch/precompiled/cpu
///   CUDA_VERSION, CUDA_TAG from .arch metadata
fn libtorch_env(project_root: &Path) -> Vec<(String, String)> {
    let mut env = Vec::new();

    // CPU path is always the same.
    env.push((
        "LIBTORCH_CPU_PATH".into(),
        "./libtorch/precompiled/cpu".into(),
    ));

    if let Some((info, host_path)) = resolve_libtorch(project_root) {
        env.push(("LIBTORCH_HOST_PATH".into(), host_path));

        // CUDA version from .arch metadata.
        if let Some(cuda) = &info.cuda_version {
            if cuda != "none" {
                let cuda_version = if cuda.matches('.').count() < 2 {
                    format!("{cuda}.0")
                } else {
                    cuda.clone()
                };
                let cuda_tag = cuda_version
                    .splitn(3, '.')
                    .take(2)
                    .collect::<Vec<_>>()
                    .join(".");
                env.push(("CUDA_VERSION".into(), cuda_version));
                env.push(("CUDA_TAG".into(), cuda_tag));
            }
        }
    }

    env
}

/// Resolve `(LibtorchInfo, host_path)` for `libtorch_env`.
///
/// Priority:
///   1. Cluster overlay's per-host `libtorch_path:` (when `FDL_ENV` is
///      set, the merged config has a `cluster:` block, AND the current
///      hostname matches an entry). Lets each host in a shared-checkout
///      heterogeneous rig pick its own libtorch without flipping the
///      global `.active`.
///   2. `project_root/libtorch/.active` (or `.active.<case>` via the
///      `FDL_LIBTORCH_CASE` env var). Standalone single-host default.
///
/// Returns `None` only when neither path resolves — the caller (env
/// builder) then omits `LIBTORCH_HOST_PATH`, which surfaces as a
/// libtorch-missing error from the downstream cargo/Docker invocation.
fn resolve_libtorch(
    project_root: &Path,
) -> Option<(libtorch::detect::LibtorchInfo, String)> {
    if let Some(resolved) = resolve_libtorch_from_overlay(project_root) {
        return Some(resolved);
    }
    let info = libtorch::detect::read_active(project_root)?;
    let host_path = format!("./libtorch/{}", info.path);
    Some((info, host_path))
}

/// Try to resolve libtorch from the active cluster overlay's current-
/// host entry. Returns `None` if no overlay is active, the overlay has
/// no `cluster:` block, the current host isn't listed, or the entry's
/// `arch:` is unset.
///
/// Convention: libtorch lives at `<host.path>/libtorch/<host.arch>`
/// on every host. The controller's view uses `<host.path>` directly
/// here because this function is the controller-side (local) path
/// resolver — when fdl runs locally as the current host, that host's
/// own `path:` IS the controller's view.
fn resolve_libtorch_from_overlay(
    project_root: &Path,
) -> Option<(libtorch::detect::LibtorchInfo, String)> {
    let env_name = std::env::var("FDL_ENV").ok()?;
    if env_name.trim().is_empty() {
        return None;
    }
    let cfg = config::load_project_with_env(
        &project_root.join("fdl.yml"),
        Some(env_name.trim()),
    ).ok()?;
    let cluster = cfg.cluster?;
    let host_name = crate::cluster::resolve_local_hostname();
    let entry = cluster.workers.iter().find(|w| w.host == host_name)?;
    let arch = entry.arch.as_ref()?;
    let variant_dir = std::path::PathBuf::from(&entry.path)
        .join("libtorch")
        .join(arch);
    resolve_libtorch_at(&variant_dir)
}

/// Resolve a `libtorch_path:` value (from cluster.yml) into
/// `(LibtorchInfo, absolute host path for Docker bind mount)`. Accepts
/// the same three shapes as `probe::check_libtorch_at`:
///   1. Pointer file `.active*` — read pointer, resolve variant against
///      the file's parent dir.
///   2. Directory containing `.active` — read its `.active`.
///   3. Direct variant dir (has `lib/`) — use as-is, parse `.arch` if
///      present.
pub(crate) fn resolve_libtorch_at(
    path: &Path,
) -> Option<(libtorch::detect::LibtorchInfo, String)> {
    if path.is_file()
        && path.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(".active"))
    {
        let libtorch_root = path.parent()?;
        let info = libtorch::detect::read_active_from(path, libtorch_root)?;
        let host_path = libtorch_root.join(&info.path).display().to_string();
        return Some((info, host_path));
    }
    if path.join(".active").exists() {
        let info = libtorch::detect::read_active_from(
            &path.join(".active"),
            path,
        )?;
        let host_path = path.join(&info.path).display().to_string();
        return Some((info, host_path));
    }
    if path.join("lib").is_dir() {
        let mut info = libtorch::detect::LibtorchInfo {
            path: path.display().to_string(),
            torch_version: None,
            cuda_version: None,
            archs: None,
            source: None,
        };
        if let Ok(content) = std::fs::read_to_string(path.join(".arch")) {
            for line in content.lines() {
                if let Some(v) = line.strip_prefix("torch=") {
                    info.torch_version = Some(v.into());
                } else if let Some(v) = line.strip_prefix("cuda=") {
                    info.cuda_version = Some(v.into());
                } else if let Some(v) = line.strip_prefix("archs=") {
                    info.archs = Some(v.into());
                } else if let Some(v) = line.strip_prefix("source=") {
                    info.source = Some(v.into());
                }
            }
        }
        let host_path = path.display().to_string();
        return Some((info, host_path));
    }
    None
}

/// Spawn a shell command with libtorch env vars set.
///
/// `FLODL_VERBOSITY` is forwarded to Docker containers via the
/// `environment:` section in docker-compose.yml (bare variable name
/// passes the host value through when set, ignored otherwise).
fn spawn_docker_shell(command: &str, project_root: &Path) -> ExitCode {
    let env_vars = libtorch_env(project_root);

    let mut cmd = std::process::Command::new("sh");
    cmd.args(["-c", command])
        .current_dir(project_root)
        // Export HOSTNAME so docker-compose's `hostname: ${HOSTNAME}`
        // interpolation resolves to the host's hostname. bash sets
        // HOSTNAME as a shell built-in but doesn't export it; docker
        // compose only reads exported env vars.
        .env(
            "HOSTNAME",
            crate::cluster::resolve_local_hostname(),
        )
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .stdin(Stdio::inherit());

    for (key, val) in &env_vars {
        cmd.env(key, val);
    }

    match cmd.status() {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

// ── Run-kind execution ──────────────────────────────────────────────────

/// POSIX-quote a single token so it round-trips through `sh -c` / `bash
/// -c` as one argument. Empty strings become `''`; tokens containing
/// only safe characters pass through unchanged; everything else is
/// wrapped in single quotes with embedded `'` escaped as `'\''`.
pub(crate) fn posix_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = s.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(c, '_' | '-' | '.' | '/' | ':' | '=' | '+' | '@' | ',')
    });
    if safe {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Split `s` on the first whitespace-bounded `--` token, returning the
/// halves with that token removed. Trim each half. When no such token is
/// found, the whole string is returned as the "before" half and the
/// "after" half is empty.
///
/// Whitespace-bounded means the `--` must be a standalone token: a
/// leading `--`, a trailing `--`, a sole `--`, or a ` -- ` between
/// other tokens. A bare `--foo` (no separator) is not a match. Quoted
/// content in `s` passes through verbatim — split scanning happens on
/// the raw string, not its shell-tokenised form.
fn split_append_dashdash(s: &str) -> (String, String) {
    let s = s.trim();
    if s == "--" {
        return (String::new(), String::new());
    }
    if let Some(rest) = s.strip_prefix("-- ") {
        return (String::new(), rest.trim().to_string());
    }
    if let Some(prefix) = s.strip_suffix(" --") {
        return (prefix.trim().to_string(), String::new());
    }
    if let Some(idx) = s.find(" -- ") {
        let before = &s[..idx];
        let after = &s[idx + 4..];
        return (before.trim().to_string(), after.trim().to_string());
    }
    (s.to_string(), String::new())
}

/// Split `args` on the first standalone `--` token, returning the
/// halves with that token removed. Returns `(before, Some(after))` when
/// a separator is present, `(args, None)` otherwise. The `None` case
/// keeps callers from emitting a stray `--` when the user did not pass
/// runner-side args.
fn split_user_args_dashdash(args: &[String]) -> (&[String], Option<&[String]>) {
    match args.iter().position(|a| a == "--") {
        Some(idx) => (&args[..idx], Some(&args[idx + 1..])),
        None => (args, None),
    }
}

/// Compose the final shell command from `run` + `append` + `user_args`.
///
/// Layout: `run [append-pre] [user-pre] -- [append-post] [user-post]`,
/// where `append-pre` / `append-post` are halves of the yml `append:`
/// field split on its first standalone `--`, and `user-pre` /
/// `user-post` are halves of the CLI tokens that followed fdl's own
/// first `--`, split again on a second `--`. The `--` separator only
/// emits when at least one of the post halves is non-empty (or the
/// `append` half explicitly carries one — preserving the legacy
/// `append: -- --nocapture` shape).
///
/// User args go after append on each side: `append` seeds defaults,
/// and last-wins for cargo-style flags lets the user override without
/// ceremony (e.g. `append: --no-ansi` + CLI `--ansi`).
pub(crate) fn compose_run_command(
    run: &str,
    user_args: &[String],
    append: Option<&str>,
) -> String {
    let suffix = append.map(str::trim).filter(|s| !s.is_empty());
    let (append_pre, append_post, append_has_dashdash) = match suffix {
        Some(s) => {
            let (pre, post) = split_append_dashdash(s);
            // Distinguish "append had a `--`" (legacy `-- --nocapture`
            // with empty pre, non-empty post) from "append had no `--`"
            // (post defaults empty). Without this we'd swallow the
            // separator on legacy yml.
            let has = s == "--"
                || s.starts_with("-- ")
                || s.ends_with(" --")
                || s.contains(" -- ");
            (pre, post, has)
        }
        None => (String::new(), String::new(), false),
    };
    let (user_pre, user_post_opt) = split_user_args_dashdash(user_args);

    let mut out = String::from(run.trim());
    if !append_pre.is_empty() {
        out.push(' ');
        out.push_str(&append_pre);
    }
    for a in user_pre {
        out.push(' ');
        out.push_str(&posix_quote(a));
    }

    let needs_separator = append_has_dashdash
        || !append_post.is_empty()
        || user_post_opt.is_some_and(|p| !p.is_empty());
    if needs_separator {
        out.push_str(" --");
        if !append_post.is_empty() {
            out.push(' ');
            out.push_str(&append_post);
        }
        if let Some(post) = user_post_opt {
            for a in post {
                out.push(' ');
                out.push_str(&posix_quote(a));
            }
        }
    }
    out
}

/// Run an inline `run:` script, optionally wrapped in Docker.
///
/// `user_args` (from CLI tokens after `--`) are POSIX-quoted and spliced
/// between `command` and `append`, so a script like `cargo test live`
/// with `append: -- --nocapture --ignored` still receives its libtest
/// flags after a user-supplied `-p flodl-hf`.
/// Forward the testing-cluster envelope into a docker-compose run
/// invocation. When `fdl @cluster-test-{nccl,cpu} <cmd>` activates an
/// overlay with a `cluster:` block, the dispatcher sets
/// `FLODL_TESTING_CLUSTER_JSON` in fdl-cli's own env (see
/// `dispatch_config` in main.rs). This helper reads that variable and
/// returns a ` -e NAME=VALUE` fragment so the inner cargo process can
/// see it; without it, the env var dies at the docker boundary and
/// `discover_test_cluster()` inside the container silently falls back
/// to local autodetect.
///
/// The hex-encoded value contains only [0-9a-f] so no shell quoting is
/// needed. Source of truth for the env-var name lives in
/// `flodl::distributed::testing::ENV_TESTING_CLUSTER_JSON`; mirrored
/// here as a literal because flodl-cli is zero-dep by policy.
fn testing_cluster_env_arg() -> String {
    match std::env::var("FLODL_TESTING_CLUSTER_JSON") {
        Ok(v) => format!(" -e FLODL_TESTING_CLUSTER_JSON={v}"),
        Err(_) => String::new(),
    }
}

pub fn exec_script(
    command: &str,
    append: Option<&str>,
    user_args: &[String],
    docker_service: Option<&str>,
    cwd: &Path,
) -> ExitCode {
    let inner_cmd = compose_run_command(command, user_args, append);

    match docker_service {
        Some(service) if !inside_docker() => {
            // Quote the whole composed command for the outer
            // `bash -c` so user args containing shell metacharacters
            // don't escape the inner shell.
            let overlay = crate::cluster::cluster_compose_overlay_arg(cwd);
            let testing_env_arg = testing_cluster_env_arg();
            let docker_cmd = format!(
                "docker compose{overlay} run --rm{testing_env_arg} {service} bash -c {}",
                posix_quote(&inner_cmd)
            );
            spawn_docker_shell(&docker_cmd, cwd)
        }
        _ => {
            let (shell, flag) = if cfg!(target_os = "windows") {
                ("cmd", "/C")
            } else {
                ("sh", "-c")
            };

            match std::process::Command::new(shell)
                .args([flag, inner_cmd.as_str()])
                .current_dir(cwd)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .stdin(Stdio::inherit())
                .status()
            {
                Ok(s) if s.success() => ExitCode::SUCCESS,
                Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}

// ── Command execution ───────────────────────────────────────────────────

/// Execute a sub-command, optionally with a named preset (inline command).
///
/// `project_root` is needed to resolve Docker compose context and
/// compute the relative workdir for containerized execution.
pub fn exec_command(
    cmd_config: &CommandConfig,
    preset_name: Option<&str>,
    extra_args: &[String],
    cmd_dir: &Path,
    project_root: &Path,
) -> ExitCode {
    let entry = match &cmd_config.entry {
        Some(e) => e.as_str(),
        None => {
            eprintln!(
                "error: no entry point defined in {}/fdl.yaml",
                cmd_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };

    // Tail validation pre-flight. Runs whenever a schema is present:
    // - `choices:` on declared options → always enforced.
    // - Unknown flags → rejected only when `schema.strict` is set
    //   (lenient mode tolerates pass-through flags the binary may
    //   consume directly).
    // fdl-generated args (from the structured ddp/training/output
    // blocks) are intentionally skipped — those are the binary's
    // surface, not the user's.
    if let Some(schema) = &cmd_config.schema {
        if let Err(e) = config::validate_tail(extra_args, schema) {
            cli_error!("{e}");
            return ExitCode::FAILURE;
        }
    }

    // Resolve config: preset overrides merged with root defaults.
    let resolved = match preset_name {
        Some(name) => match cmd_config.commands.get(name) {
            Some(preset) => {
                // Validate *this* preset only (choices + strict unknowns).
                // Whole-map validation is deferred so a broken sibling
                // preset doesn't block a correct one from running.
                if let Some(schema) = &cmd_config.schema {
                    if let Err(e) = config::validate_preset_for_exec(name, preset, schema) {
                        cli_error!("{e}");
                        return ExitCode::FAILURE;
                    }
                }
                config::merge_preset(cmd_config, preset)
            }
            None => {
                cli_error!("unknown command '{name}'");
                eprintln!();
                print_command_help(cmd_config, "");
                return ExitCode::FAILURE;
            }
        },
        None => config::defaults_only(cmd_config),
    };

    // Build argument list from config.
    let mut args = config_to_args(&resolved);

    // Append extra CLI args (these override config-derived args).
    args.extend(extra_args.iter().cloned());

    // Docker wrapping or direct execution.
    let use_docker = cmd_config.docker.is_some() && !inside_docker();

    if use_docker {
        let service = cmd_config.docker.as_deref().unwrap();
        let workdir = cmd_dir
            .strip_prefix(project_root)
            .unwrap_or(cmd_dir)
            .to_string_lossy();

        // Build the inner command: cd <container-root>/<workdir> && <entry> <args>
        //
        // `<container-root>` is discovered from docker-compose.yml's
        // bind mount for the project (host `.` → container target) via
        // [`container_project_root`], so the generated `cd` works
        // regardless of the service's own `working_dir:`. Falls back
        // to `/workspace` (the fdl init convention) when the compose
        // file is missing or silent on this service.
        let container_root = container_project_root(project_root, service);
        let args_str = shell_join(&args);
        let inner = if workdir.is_empty() || workdir == "." {
            format!("{entry} {args_str}")
        } else {
            format!("cd {container_root}/{workdir} && {entry} {args_str}")
        };

        if preset_name.is_some() {
            eprintln!("fdl: [{service}] {inner}");
        }

        // Surface the container-side workspace root to the inner
        // process so entry binaries can re-anchor argv path arguments
        // independently of the per-task `cd <root>/<workdir>` we
        // injected above. Mirrors the `HF_HOME` pattern: `fdl` owns
        // the env, the binary just reads it. Without this, a user
        // typing `flodl-hf/tests/.exports/bert` from the host repo
        // root resolves against the wrong cwd inside the container.
        let overlay = crate::cluster::cluster_compose_overlay_arg(project_root);
        let testing_env_arg = testing_cluster_env_arg();
        let docker_cmd = format!(
            "docker compose{overlay} run --rm -e FDL_PROJECT_ROOT={container_root}{testing_env_arg} {service} bash -c \"{inner}\"",
        );
        spawn_docker_shell(&docker_cmd, project_root)
    } else {
        // Direct execution (inside container or no docker configured).
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.is_empty() {
            cli_error!("empty entry point");
            return ExitCode::FAILURE;
        }
        let program = parts[0];
        let entry_args = &parts[1..];

        if preset_name.is_some() {
            let preview: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            eprintln!("fdl: {entry} {}", preview.join(" "));
        }

        match std::process::Command::new(program)
            .args(entry_args)
            .args(&args)
            .current_dir(cmd_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::inherit())
            .status()
        {
            Ok(s) if s.success() => ExitCode::SUCCESS,
            Ok(s) => ExitCode::from(s.code().unwrap_or(1) as u8),
            Err(e) => {
                cli_error!("failed to execute '{program}': {e}");
                ExitCode::FAILURE
            }
        }
    }
}

/// Join args into a single shell-safe string for the docker `bash -c "…"`
/// path. Each token is POSIX-quoted via [`posix_quote`], so shell
/// metacharacters in a value (`$`, backticks, globs, `;`, `|`, redirections,
/// …) are passed literally rather than expanded or interpreted by the inner
/// shell. The previous predicate only quoted tokens containing a space, `"`,
/// or empty — leaving `$HOME`, `` `cmd` ``, `*.py`, `a;b` to be interpreted.
/// Mirrors `exec_script`, which quotes through the same helper.
fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|a| posix_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Help output ─────────────────────────────────────────────────────────

/// Print help for a `run:`-kind command. Shows the inline script,
/// any `append:` suffix, the Docker service (if any), and the `--`
/// forwarding contract.
pub fn print_run_help(
    name: &str,
    description: Option<&str>,
    run: &str,
    append: Option<&str>,
    docker: Option<&str>,
) {
    if let Some(desc) = description {
        eprintln!("{} {desc}", style::bold(name));
    } else {
        eprintln!("{}", style::bold(name));
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!("    fdl {name} [-- <args>... [-- <runner-args>...]]");
    eprintln!();
    eprintln!("{}:", style::yellow("Runs"));
    let composed = match append.map(str::trim).filter(|s| !s.is_empty()) {
        Some(suffix) => {
            let (pre, post) = split_append_dashdash(suffix);
            // Show whichever halves the yml actually populated, with
            // the user slots interleaved at the right side per the
            // merge contract.
            let left = if pre.is_empty() {
                "[<args>]".to_string()
            } else {
                format!("{pre} [<args>]")
            };
            let right = if post.is_empty() {
                "[<runner-args>]".to_string()
            } else {
                format!("{post} [<runner-args>]")
            };
            format!("{run} {left} -- {right}")
        }
        None => format!("{run} [<args>] [-- <runner-args>]"),
    };
    if let Some(svc) = docker {
        eprintln!(
            "    {} {svc} -c {composed:?}",
            style::dim("docker compose run --rm")
        );
    } else {
        eprintln!("    {composed}");
    }
    eprintln!();
    eprintln!(
        "{} the first `--` separates fdl args from the run script; a second `--` splits cargo-side args from runner-side args.",
        style::dim("Note:"),
    );
    eprintln!(
        "{} `append:` is split on its own `--` and merged half-and-half; pass `--no-append` to drop it entirely.",
        style::dim("Note:"),
    );
}

/// Print help for a sub-command (its arguments, nested commands, and
/// entry). Orchestrates the per-section helpers below.
pub fn print_command_help(cmd_config: &CommandConfig, name: &str) {
    let (presets, sub_cmds) = split_commands_by_kind(&cmd_config.commands);
    let preset_slot = cmd_config.arg_name.as_deref().unwrap_or("preset");

    print_title(cmd_config, name);
    print_usage_line(cmd_config, name, &presets, &sub_cmds, preset_slot);
    print_arguments_section(cmd_config, &presets, preset_slot);
    print_sub_commands_section(&sub_cmds);
    print_schema_commands_section(cmd_config, name);
    print_options_section(cmd_config);
    print_entry_section(cmd_config);
    print_defaults_section(cmd_config);
}

fn print_title(cmd_config: &CommandConfig, name: &str) {
    if let Some(desc) = &cmd_config.description {
        eprintln!("{} {desc}", style::bold(name));
    } else {
        eprintln!("{}", style::bold(name));
    }
}

fn print_usage_line(
    cmd_config: &CommandConfig,
    name: &str,
    presets: &CommandGroup,
    sub_cmds: &CommandGroup,
    preset_slot: &str,
) {
    // The first-positional slot reflects what is actually accepted here:
    // preset name, sub-command name, or either.
    let usage_tail = build_usage_tail(
        cmd_config.schema.as_ref(),
        !presets.is_empty(),
        !sub_cmds.is_empty(),
        preset_slot,
    );
    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!("    fdl {name}{usage_tail}");
}

fn print_arguments_section(
    cmd_config: &CommandConfig,
    presets: &CommandGroup,
    preset_slot: &str,
) {
    // Schema-declared positionals (typed slots on the entry binary) and
    // the preset slot (dispatched by fdl before the binary sees argv)
    // both land in the first-positional position, so they share one
    // section. Schema args render first; the preset slot with its
    // value list follows.
    let has_schema_args = cmd_config
        .schema
        .as_ref()
        .is_some_and(|s| !s.args.is_empty());
    if !has_schema_args && presets.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Arguments"));
    if let Some(schema) = &cmd_config.schema {
        for a in &schema.args {
            eprintln!("    {}", format_arg(a));
        }
    }
    if !presets.is_empty() {
        let slot_label = format!("[<{preset_slot}>]");
        eprintln!(
            "    {}  Named preset, one of:",
            style::green(&format!("{:<20}", slot_label))
        );
        for (pname, spec) in presets {
            let desc = spec.description.as_deref().unwrap_or("-");
            eprintln!(
                "      {}  {}",
                style::green(&format!("{:<18}", pname)),
                desc
            );
        }
    }
}

fn print_sub_commands_section(sub_cmds: &CommandGroup) {
    // Run/Path kinds only — true sub-commands with their own behavior
    // (an inline script or a nested fdl.yml).
    if sub_cmds.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Commands"));
    for (sub_name, sub_spec) in sub_cmds {
        let desc = sub_spec.description.as_deref().unwrap_or("-");
        eprintln!(
            "    {}  {}",
            style::green(&format!("{:<20}", sub_name)),
            desc
        );
    }
}

/// List the entry binary's own subcommands when its schema is a tree
/// (a variant-shaped `#[derive(FdlArgs)]` CLI). Distinct from
/// [`print_sub_commands_section`], which lists fdl.yml-level Run/Path
/// commands — these come from the binary's `--fdl-schema` output, and each
/// has its own flag set (drill in with `fdl <name> <subcommand> --help`).
fn print_schema_commands_section(cmd_config: &CommandConfig, name: &str) {
    let Some(schema) = &cmd_config.schema else {
        return;
    };
    if schema.commands.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Commands"));
    for (sub_name, sub_schema) in &schema.commands {
        let desc = sub_schema.description.as_deref().unwrap_or("-");
        eprintln!(
            "    {}  {}",
            style::green(&format!("{:<20}", sub_name)),
            desc
        );
    }
    eprintln!();
    eprintln!(
        "    Run {} for a subcommand's options.",
        style::dim(&format!("fdl {name} <command> --help"))
    );
}

fn print_options_section(cmd_config: &CommandConfig) {
    // Schema-driven options. Renders only when a schema block is present
    // in fdl.yaml; the "Defaults" section covers ddp/training/output.
    let Some(schema) = &cmd_config.schema else {
        return;
    };
    if schema.options.is_empty() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Options"));
    for (long, spec) in &schema.options {
        for line in format_option(long, spec) {
            eprintln!("    {line}");
        }
    }
}

fn print_entry_section(cmd_config: &CommandConfig) {
    let Some(entry) = &cmd_config.entry else {
        return;
    };
    eprintln!();
    eprintln!("{}:", style::yellow("Entry"));
    eprintln!("    {entry}");
    if let Some(service) = &cmd_config.docker {
        eprintln!(
            "     {}",
            style::dim(&format!("[docker: {service}]"))
        );
    }
    eprintln!();
    eprintln!(
        "    Any extra {} are forwarded to the entry point.",
        style::dim("[options]")
    );
}

fn print_defaults_section(cmd_config: &CommandConfig) {
    if cmd_config.ddp.is_none() && cmd_config.training.is_none() {
        return;
    }
    eprintln!();
    eprintln!("{}:", style::yellow("Defaults"));
    if let Some(d) = &cmd_config.ddp {
        if let Some(mode) = &d.mode {
            eprintln!("    {}  {mode}", style::dim("ddp.mode"));
        }
        if let Some(anchor) = &d.anchor {
            eprintln!("    {}  {}", style::dim("ddp.anchor"), value_to_string(anchor));
        }
    }
    if let Some(t) = &cmd_config.training {
        if let Some(e) = t.epochs {
            eprintln!("    {}  {e}", style::dim("training.epochs"));
        }
        if let Some(bs) = t.batch_size {
            eprintln!("    {}  {bs}", style::dim("training.batch_size"));
        }
        if let Some(lr) = t.lr {
            eprintln!("    {}  {lr}", style::dim("training.lr"));
        }
        if let Some(seed) = t.seed {
            eprintln!("    {}  {seed}", style::dim("training.seed"));
        }
    }
}

/// Print help for a named preset command nested inside a sub-command.
pub fn print_preset_help(cmd_config: &CommandConfig, cmd_name: &str, preset_name: &str) {
    let preset = match cmd_config.commands.get(preset_name) {
        Some(s) => s,
        None => {
            eprintln!("unknown command: {preset_name}");
            return;
        }
    };

    // Title.
    let desc = preset.description.as_deref().unwrap_or("(no description)");
    eprintln!(
        "{} {} {}",
        style::bold(cmd_name),
        style::green(preset_name),
        desc
    );

    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!(
        "    fdl {cmd_name} {preset_name} {}",
        style::dim("[extra options]")
    );

    // Show the merged config that this preset produces.
    let resolved = config::merge_preset(cmd_config, preset);

    eprintln!();
    eprintln!("{}:", style::yellow("Effective config"));

    // DDP fields.
    let d = &resolved.ddp;
    print_config_field("ddp.mode", &d.mode);
    print_config_value("ddp.anchor", &d.anchor);
    print_config_field("ddp.max_anchor", &d.max_anchor);
    print_config_field("ddp.overhead_target", &d.overhead_target);
    print_config_field("ddp.divergence_threshold", &d.divergence_threshold);
    print_config_value("ddp.max_batch_diff", &d.max_batch_diff);
    print_config_field("ddp.max_grad_norm", &d.max_grad_norm);
    if d.timeline == Some(true) {
        eprintln!("    {}  true", style::dim("ddp.timeline"));
    }

    // Training fields.
    let t = &resolved.training;
    print_config_field("training.epochs", &t.epochs);
    print_config_field("training.batch_size", &t.batch_size);
    print_config_field("training.batches_per_epoch", &t.batches_per_epoch);
    print_config_field("training.lr", &t.lr);
    print_config_field("training.seed", &t.seed);

    // Output fields.
    let o = &resolved.output;
    print_config_field("output.dir", &o.dir);
    print_config_field("output.monitor", &o.monitor);

    // Pass-through options.
    if !resolved.options.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Options"));
        for (key, val) in &resolved.options {
            eprintln!(
                "    {}  {}",
                style::green(&format!("--{}", key.replace('_', "-"))),
                value_to_string(val)
            );
        }
    }

    // Show the effective command.
    if let Some(entry) = &cmd_config.entry {
        let args = config_to_args(&resolved);
        let args_str = args.join(" ");
        let docker_info = cmd_config
            .docker
            .as_ref()
            .map(|s| format!("[{s}] ", ))
            .unwrap_or_default();

        eprintln!();
        eprintln!("{}:", style::yellow("Effective command"));
        eprintln!(
            "    {}{}{}",
            style::dim(&docker_info),
            entry,
            if args_str.is_empty() {
                String::new()
            } else {
                format!(" {args_str}")
            }
        );
    }

    eprintln!();
    eprintln!(
        "Extra {} after the command name are appended to the entry.",
        style::dim("[options]")
    );
}

fn print_config_field<T: std::fmt::Display>(label: &str, val: &Option<T>) {
    if let Some(v) = val {
        eprintln!("    {}  {v}", style::dim(label));
    }
}

fn print_config_value(label: &str, val: &Option<serde_json::Value>) {
    if let Some(v) = val {
        if !v.is_null() {
            eprintln!("    {}  {}", style::dim(label), value_to_string(v));
        }
    }
}

/// Print the project help with scripts and commands.
pub fn print_project_help(
    project: &config::ProjectConfig,
    project_root: &Path,
    active_env: Option<&str>,
) {
    let visible_builtins = builtins::visible_top_level();
    if let Some(desc) = &project.description {
        eprintln!("{} {}", style::bold("fdl"), desc);
    } else {
        eprintln!("{} {}", style::bold("fdl"), env!("CARGO_PKG_VERSION"));
    }

    eprintln!();
    eprintln!("{}:", style::yellow("Usage"));
    eprintln!(
        "    fdl {} {}",
        style::dim("<command>"),
        style::dim("[options]")
    );

    eprintln!();
    eprintln!("{}:", style::yellow("Options"));
    eprintln!(
        "    {}  Show this help",
        style::green(&format!("{:<18}", "-h, --help"))
    );
    eprintln!(
        "    {}  Show version",
        style::green(&format!("{:<18}", "-V, --version"))
    );
    eprintln!(
        "    {}  Use fdl.<name>.yml overlay (also: --env <name>, FDL_ENV=<name>)",
        style::green(&format!("{:<18}", "@<name>"))
    );
    eprintln!(
        "    {}  Verbose output",
        style::green(&format!("{:<18}", "-v"))
    );
    eprintln!(
        "    {}  Debug output",
        style::green(&format!("{:<18}", "-vv"))
    );
    eprintln!(
        "    {}  Trace output (maximum detail)",
        style::green(&format!("{:<18}", "-vvv"))
    );
    eprintln!(
        "    {}  Suppress non-error output",
        style::green(&format!("{:<18}", "-q, --quiet"))
    );
    eprintln!(
        "    {}  Force ANSI color (bypass TTY / NO_COLOR detection)",
        style::green(&format!("{:<18}", "--ansi"))
    );
    eprintln!(
        "    {}  Disable ANSI color output",
        style::green(&format!("{:<18}", "--no-ansi"))
    );
    eprintln!(
        "    {}  Drop a run command's `append:` suffix",
        style::green(&format!("{:<18}", "--no-append"))
    );
    eprintln!(
        "    {}  Skip the cluster pre-flight build",
        style::green(&format!("{:<18}", "--no-prebuild"))
    );

    // Built-in commands.
    eprintln!();
    eprintln!("{}:", style::yellow("Built-in"));
    for (name, desc) in &visible_builtins {
        eprintln!("    {}  {desc}", style::green(&format!("{:<18}", name)));
    }

    // Commands: unified section. Each entry in `project.commands` is one
    // of: an inline `run:` script, a `path:` (or convention-default)
    // pointer to a child fdl.yml, or — at nested levels only — an inline
    // preset. Descriptions come from the `CommandSpec`; for `path:`
    // commands missing their own description, fall back to loading the
    // child fdl.yml's description.
    if !project.commands.is_empty() {
        eprintln!();
        eprintln!("{}:", style::yellow("Commands"));
        for (name, spec) in &project.commands {
            let desc: String = match spec.description.clone() {
                Some(d) => d,
                None => {
                    // For path-kind entries, fall back to the child config's
                    // own description so `commands: { ddp-bench: }` still
                    // shows a useful blurb.
                    let is_path_kind = spec.run.is_none();
                    if is_path_kind {
                        let child_dir = spec.resolve_path(name, project_root);
                        config::load_command_with_env(&child_dir, active_env)
                            .ok()
                            .and_then(|c| c.description)
                            .unwrap_or_else(|| "(sub-command)".into())
                    } else {
                        spec.run
                            .as_deref()
                            .unwrap_or("(command)")
                            .to_string()
                    }
                }
            };
            eprintln!("    {}  {desc}", style::green(&format!("{:<18}", name)));
        }
    }

    // Available environments (sibling fdl.<env>.yml files at project root).
    if let Some(base_config) = config::find_config(project_root) {
        let envs = crate::overlay::list_envs(&base_config);
        if !envs.is_empty() {
            eprintln!();
            eprintln!("{}:", style::yellow("Environments"));
            for e in &envs {
                let active_marker = if Some(e.as_str()) == active_env {
                    style::green(" (active)")
                } else {
                    String::new()
                };
                eprintln!(
                    "    {}  Overlay from fdl.{}.yml{active_marker}",
                    style::green(&format!("{:<18}", format!("@{e}"))),
                    e
                );
            }
            eprintln!();
            eprintln!(
                "Use {} to run a command with an environment overlay.",
                style::dim("fdl @<env> <command>")
            );
        }
    }

    eprintln!();
    eprintln!(
        "Use {} for more information on a command.",
        style::dim("fdl <command> -h")
    );
}

// ── Schema-driven help helpers ──────────────────────────────────────────

/// Build the part of `fdl <cmd>...` after the command name: positionals
/// rendered as `<name>` (required) or `[<name>]` (optional), plus a slot
/// for the first-positional picker — `[<preset>]` when only presets exist,
/// `[<command>]` when only sub-commands exist, `[<preset>|<command>]` when
/// both — and `[options]`. The preset placeholder is customisable per
/// sub-command via `arg-name:`.
fn build_usage_tail(
    schema: Option<&Schema>,
    has_presets: bool,
    has_sub_commands: bool,
    preset_slot: &str,
) -> String {
    let mut parts = String::new();
    let slot = match (has_presets, has_sub_commands) {
        (true, false) => Some(format!("[<{preset_slot}>]")),
        (false, true) => Some("[<command>]".to_string()),
        (true, true) => Some(format!("[<{preset_slot}>|<command>]")),
        (false, false) => None,
    };
    if let Some(s) = slot {
        parts.push(' ');
        parts.push_str(&style::dim(&s));
    }
    if let Some(s) = schema {
        for a in &s.args {
            parts.push(' ');
            parts.push_str(&format_arg_usage(a));
        }
    }
    parts.push(' ');
    parts.push_str(&style::dim("[options]"));
    parts
}

type CommandGroup = Vec<(String, crate::config::CommandSpec)>;

/// Partition a `commands:` map into (presets, sub-commands) by resolved
/// `CommandKind`. Entries whose `kind()` errors (both run and path set)
/// are treated as sub-commands so they still render somewhere — the
/// error surfaces when the user tries to dispatch them.
fn split_commands_by_kind(
    commands: &BTreeMap<String, crate::config::CommandSpec>,
) -> (CommandGroup, CommandGroup) {
    use crate::config::CommandKind;
    let mut presets = Vec::new();
    let mut sub_cmds = Vec::new();
    for (k, v) in commands {
        match v.kind() {
            Ok(CommandKind::Preset) => presets.push((k.clone(), v.clone())),
            _ => sub_cmds.push((k.clone(), v.clone())),
        }
    }
    (presets, sub_cmds)
}

fn format_arg_usage(a: &ArgSpec) -> String {
    let suffix = if a.variadic { "..." } else { "" };
    let core = format!("<{}>{suffix}", a.name);
    if a.required && a.default.is_none() {
        style::green(&core)
    } else {
        style::dim(&format!("[{core}]"))
    }
}

fn format_arg(a: &ArgSpec) -> String {
    let mut left = format_arg_usage(a);
    // Target ~22-char visual width for the label column.
    let visible = visible_width(&left);
    if visible < 22 {
        for _ in 0..(22 - visible) {
            left.push(' ');
        }
    } else {
        left.push(' ');
    }
    let mut line = left;
    line.push_str(a.description.as_deref().unwrap_or("-"));
    append_default_and_choices(&mut line, &a.default, &a.choices, &a.ty);
    line
}

/// Format an option row. Returns one or more lines; `choices` list wraps
/// onto a second indented line when present, to keep the main row readable.
fn format_option(long: &str, spec: &OptionSpec) -> Vec<String> {
    let flag = match &spec.short {
        Some(s) => format!("-{s}, --{long}"),
        None => format!("    --{long}"),
    };
    let placeholder = option_placeholder(&spec.ty);
    let left = if placeholder.is_empty() {
        style::green(&flag)
    } else {
        style::green(&format!("{flag} {placeholder}"))
    };
    let visible = visible_width_for(&flag, placeholder);

    // Pad to 30 columns for alignment.
    let pad = if visible < 30 { 30 - visible } else { 1 };
    let mut line = format!("{left}{}", " ".repeat(pad));
    line.push_str(spec.description.as_deref().unwrap_or("-"));
    append_default_and_choices(&mut line, &spec.default, &spec.choices, &spec.ty);

    let mut out = vec![line];
    if let Some(env) = &spec.env {
        out.push(format!("{}  {}", " ".repeat(32), style::dim(&format!("[env: {env}]"))));
    }
    out
}

fn option_placeholder(ty: &str) -> &'static str {
    match ty {
        "bool" => "",
        "int" => "<N>",
        "float" => "<F>",
        "path" => "<PATH>",
        "list[path]" => "<PATH>...",
        t if t.starts_with("list[") => "<VALUE>...",
        _ => "<VALUE>",
    }
}

fn append_default_and_choices(
    line: &mut String,
    default: &Option<serde_json::Value>,
    choices: &Option<Vec<serde_json::Value>>,
    ty: &str,
) {
    if let Some(d) = default {
        // Skip noisy defaults: bool false, empty list, null.
        let is_empty_list = matches!(d, serde_json::Value::Array(a) if a.is_empty());
        let is_false = matches!(d, serde_json::Value::Bool(false));
        if !d.is_null() && !is_false && !is_empty_list {
            line.push_str(&format!(" {}", style::dim(&format!("[default: {}]", format_value(d)))));
        }
    }
    if let Some(choices) = choices {
        if !choices.is_empty() {
            let list = choices
                .iter()
                .map(format_value)
                .collect::<Vec<_>>()
                .join(", ");
            line.push_str(&format!(" {}", style::dim(&format!("[possible: {list}]"))));
        }
    }
    // Annotate list types so users know about repeat/comma semantics.
    if ty.starts_with("list[") {
        line.push_str(&format!(" {}", style::dim("(repeat or comma-separate)")));
    }
}

fn format_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Rough visible width helper: styled strings wrap their visible content
/// in ANSI escapes, so we use the unstyled inputs we started from.
fn visible_width(s: &str) -> usize {
    // The inputs we pass here come from pre-styling helpers that already
    // know the raw length. Strip ANSI to be safe.
    strip_ansi(s).chars().count()
}

fn visible_width_for(flag: &str, placeholder: &str) -> usize {
    if placeholder.is_empty() {
        flag.chars().count()
    } else {
        flag.chars().count() + 1 + placeholder.chars().count()
    }
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posix_quote_passes_safe_strings_through() {
        assert_eq!(posix_quote("hello"), "hello");
        assert_eq!(posix_quote("-p"), "-p");
        assert_eq!(posix_quote("flodl-hf"), "flodl-hf");
        assert_eq!(posix_quote("a/b.c"), "a/b.c");
        assert_eq!(posix_quote("KEY=val"), "KEY=val");
    }

    #[test]
    fn posix_quote_wraps_unsafe_strings() {
        assert_eq!(posix_quote(""), "''");
        assert_eq!(posix_quote("foo bar"), "'foo bar'");
        assert_eq!(posix_quote("a$b"), "'a$b'");
        assert_eq!(posix_quote("a\"b"), "'a\"b'");
    }

    #[test]
    fn posix_quote_escapes_embedded_single_quotes() {
        assert_eq!(posix_quote("it's"), "'it'\\''s'");
        assert_eq!(posix_quote("'"), "''\\'''");
    }

    #[test]
    fn shell_join_quotes_shell_metacharacters() {
        // M23: safe tokens pass through bare and join with spaces; values with
        // shell metacharacters ($, glob, ;) are single-quoted so the inner
        // `bash -c "…"` treats them literally instead of expanding them.
        let args: Vec<String> = ["--model", "mlp", "--tag", "$HOME", "*.py", "a;b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(shell_join(&args), "--model mlp --tag '$HOME' '*.py' 'a;b'");
    }

    #[test]
    fn compose_run_command_no_extras_passes_run_through() {
        assert_eq!(compose_run_command("echo hello", &[], None), "echo hello");
    }

    #[test]
    fn compose_run_command_inserts_user_args_between_run_and_append() {
        let user = vec!["-p".to_string(), "flodl-hf".to_string()];
        let out = compose_run_command("cargo test live", &user, Some("-- --nocapture --ignored"));
        assert_eq!(out, "cargo test live -p flodl-hf -- --nocapture --ignored");
    }

    #[test]
    fn compose_run_command_quotes_user_args_with_spaces() {
        let user = vec!["--name".to_string(), "with space".to_string()];
        let out = compose_run_command("cmd", &user, None);
        assert_eq!(out, "cmd --name 'with space'");
    }

    #[test]
    fn compose_run_command_omits_empty_append() {
        let out = compose_run_command("cmd", &["arg".to_string()], Some(""));
        assert_eq!(out, "cmd arg");
        let out2 = compose_run_command("cmd", &["arg".to_string()], Some("   "));
        assert_eq!(out2, "cmd arg");
    }

    #[test]
    fn compose_run_command_user_double_dash_threads_runner_args() {
        let user = vec![
            "-p".to_string(),
            "foo".to_string(),
            "--".to_string(),
            "--ignored".to_string(),
        ];
        let out = compose_run_command("cargo test", &user, Some("-- --nocapture"));
        assert_eq!(out, "cargo test -p foo -- --nocapture --ignored");
    }

    #[test]
    fn compose_run_command_user_double_dash_without_append() {
        let user = vec![
            "-p".to_string(),
            "foo".to_string(),
            "--".to_string(),
            "--ignored".to_string(),
        ];
        let out = compose_run_command("cargo test", &user, None);
        assert_eq!(out, "cargo test -p foo -- --ignored");
    }

    #[test]
    fn compose_run_command_append_with_pre_and_post_halves() {
        let out = compose_run_command("cmd", &[], Some("--foo -- --bar"));
        assert_eq!(out, "cmd --foo -- --bar");
    }

    #[test]
    fn compose_run_command_append_pre_only_no_separator() {
        // append carries a default flag with no `--` token; user supplies
        // an override. Defaults seed first, user wins via last-flag-wins.
        let user = vec!["--ansi".to_string()];
        let out = compose_run_command("cmd", &user, Some("--no-ansi"));
        assert_eq!(out, "cmd --no-ansi --ansi");
    }

    #[test]
    fn compose_run_command_user_only_double_dash_emits_separator() {
        let user = vec!["--".to_string(), "--list".to_string()];
        let out = compose_run_command("cargo test", &user, None);
        assert_eq!(out, "cargo test -- --list");
    }

    #[test]
    fn compose_run_command_append_full_split_with_user_both_sides() {
        let user = vec![
            "-p".to_string(),
            "foo".to_string(),
            "--".to_string(),
            "--ignored".to_string(),
        ];
        let out = compose_run_command(
            "cargo test",
            &user,
            Some("--release -- --nocapture"),
        );
        assert_eq!(
            out,
            "cargo test --release -p foo -- --nocapture --ignored"
        );
    }

    #[test]
    fn split_append_dashdash_handles_edges() {
        assert_eq!(
            split_append_dashdash("-- --nocapture"),
            (String::new(), "--nocapture".to_string())
        );
        assert_eq!(
            split_append_dashdash("--foo -- --bar"),
            ("--foo".to_string(), "--bar".to_string())
        );
        assert_eq!(
            split_append_dashdash("--foo --"),
            ("--foo".to_string(), String::new())
        );
        assert_eq!(
            split_append_dashdash("--"),
            (String::new(), String::new())
        );
        assert_eq!(
            split_append_dashdash("--foo"),
            ("--foo".to_string(), String::new())
        );
        assert_eq!(
            split_append_dashdash(""),
            (String::new(), String::new())
        );
    }

    // ── resolve_libtorch_at: 3-shape libtorch_path: resolution ──────────
    //
    // Each test builds a synthetic libtorch dir under a per-test scratch
    // path (zero-deps policy precludes `tempfile`) and feeds the path
    // through `resolve_libtorch_at`. Variant names (`precompiled/v1`,
    // `builds/v2`) are deliberately abstract — the resolver is structural,
    // not rig-aware.

    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SCRATCH_SEQ: AtomicU64 = AtomicU64::new(0);

    struct Scratch(std::path::PathBuf);
    impl Scratch {
        fn new() -> Self {
            let nanos = SystemTime::now().duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos()).unwrap_or(0);
            let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("fdl-resolve-libtorch-{}-{}", nanos, seq));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &std::path::Path { &self.0 }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn populate_lt_root(root: &std::path::Path) {
        for (sub, torch, archs) in [
            ("precompiled/v1", "1.0", "0.0"),
            ("builds/v2", "2.0", "1.0"),
        ] {
            let d = root.join(sub);
            std::fs::create_dir_all(d.join("lib")).unwrap();
            std::fs::write(
                d.join(".arch"),
                format!("torch={torch}\ncuda=1.0\narchs={archs}\nsource=test\n"),
            ).unwrap();
        }
    }

    #[test]
    fn resolve_libtorch_at_pointer_file() {
        let s = Scratch::new();
        let lt = s.path().join("libtorch");
        populate_lt_root(&lt);
        let pointer = lt.join(".active.alt");
        std::fs::write(&pointer, "precompiled/v1\n").unwrap();

        let (info, host_path) = resolve_libtorch_at(&pointer)
            .expect("pointer file resolves");
        assert_eq!(info.path, "precompiled/v1");
        assert_eq!(info.torch_version.as_deref(), Some("1.0"));
        assert_eq!(host_path, lt.join("precompiled/v1").display().to_string());
    }

    #[test]
    fn resolve_libtorch_at_libtorch_root_dir() {
        let s = Scratch::new();
        let lt = s.path().join("libtorch");
        populate_lt_root(&lt);
        std::fs::write(lt.join(".active"), "builds/v2\n").unwrap();

        let (info, host_path) = resolve_libtorch_at(&lt)
            .expect("libtorch-root dir resolves");
        assert_eq!(info.path, "builds/v2");
        assert_eq!(info.torch_version.as_deref(), Some("2.0"));
        assert_eq!(host_path, lt.join("builds/v2").display().to_string());
    }

    #[test]
    fn resolve_libtorch_at_direct_variant_dir() {
        let s = Scratch::new();
        let variant = s.path().join("standalone-libtorch");
        std::fs::create_dir_all(variant.join("lib")).unwrap();
        std::fs::write(
            variant.join(".arch"),
            "torch=3.0\ncuda=2.0\narchs=1.0\nsource=test\n",
        ).unwrap();

        let (info, host_path) = resolve_libtorch_at(&variant)
            .expect("direct variant dir resolves");
        assert_eq!(info.path, variant.display().to_string());
        assert_eq!(info.torch_version.as_deref(), Some("3.0"));
        assert_eq!(host_path, variant.display().to_string());
    }

    #[test]
    fn resolve_libtorch_at_bogus_path_returns_none() {
        let s = Scratch::new();
        let bogus = s.path().join("no-lib-no-active-no-pointer");
        std::fs::create_dir_all(&bogus).unwrap();
        assert!(resolve_libtorch_at(&bogus).is_none(),
            "dir without lib/, .active, or pointer-shape filename → None");
    }
}
