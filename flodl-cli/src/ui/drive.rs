//! Driving the CLI: turn a request body into an `fdl` argv, run it, and
//! report the run verbatim.
//!
//! This is where the page's promise lives — it drives the CLI rather
//! than reimplementing it — so every function here either builds an
//! argv or executes one. The builders are pure and unit-tested, which is
//! what lets the exec leg stay out of the test suite (nothing here execs
//! a file a test wrote).
//!
//! All form values arrive through [`safe_value`]: bounded, no control
//! characters, and never flag-shaped unless the caller says dashes are
//! the point. A form field must not be able to become an option.

use std::io::Write;
use std::net::TcpStream;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use super::http::{error_json, json_ok};
use super::{MAX_VALUE_LEN, Request, UiServer};
use crate::join_config::validate_label;

pub(super) fn run_target_route(req: &Request, server: &UiServer) -> Vec<u8> {
    if req.method == "POST" {
        let body = match parse_body(&req.body) {
            Ok(b) => b,
            Err(why) => return error_json("400 Bad Request", &why),
        };
        let port = match body.get("port") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => match v.as_u64().filter(|p| (1..=65535).contains(p)) {
                Some(p) => Some(p as u16),
                None => return error_json("400 Bad Request", "port: expected 1..=65535 or null"),
            },
        };
        *server.run_target.lock().expect("run target lock") = port;
    }
    let port = *server.run_target.lock().expect("run target lock");
    let reachable = port.is_some_and(|p| {
        TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 1], p)),
            Duration::from_millis(300),
        )
        .is_ok()
    });
    json_ok(&serde_json::json!({ "port": port, "reachable": reachable }))
}

/// A farm's shareable artifacts, for the page's copy buttons. The
/// worker yml carries the admission token — same exposure as the
/// terminal render, behind the same token+Host gates as every route.
/// cloud-init is deliberately NOT served: it embeds the private key,
/// and a secret artifact stays a file path everywhere.
pub(super) fn farm_artifacts(req: &Request, server: &UiServer) -> Vec<u8> {
    let Some(label) = req.query.get("label") else {
        return error_json("400 Bad Request", "missing ?label=");
    };
    if let Err(why) = validate_label(label) {
        return error_json("400 Bad Request", &why);
    }
    let farm_dir = server.root.join(".fdl").join(label);
    if !farm_dir.is_dir() {
        return error_json("404 Not Found", "no such farm dir");
    }
    let read = |name: &str| std::fs::read_to_string(farm_dir.join(name)).ok();
    json_ok(&serde_json::json!({
        "label": label,
        "worker_yml": read("worker.yml"),
        "install_notes": read("install-notes.md"),
        "sshd_conf": read(&format!("sshd-{label}.conf")),
        "cloud_init_path": farm_dir
            .join("cloud-init.yml")
            .is_file()
            .then(|| farm_dir.join("cloud-init.yml").display().to_string()),
    }))
}

/// Parse a POST body as a JSON object (an empty body is an empty
/// object).
pub(super) fn parse_body(body: &[u8]) -> Result<serde_json::Value, String> {
    if body.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_slice(body).map_err(|e| format!("body is not JSON: {e}"))
}

/// One driven value: bounded, printable, and — unless the caller says
/// otherwise — not flag-shaped. A value opening with `-` would be read
/// by the child's parser as an option, which turns a form field into a
/// flag-injection road; the one place dashes are legal is a publish
/// args tail, which rides behind a standalone `--`.
pub(super) fn safe_value(name: &str, value: &str, allow_dash: bool) -> Result<(), String> {
    if value.len() > MAX_VALUE_LEN {
        return Err(format!("{name}: value too long"));
    }
    if value.chars().any(|c| c.is_control()) {
        return Err(format!("{name}: control characters are not a value"));
    }
    if !allow_dash && value.starts_with('-') {
        return Err(format!("{name}: a value cannot start with `-`"));
    }
    Ok(())
}

/// A body string field, validated. Absent and empty are both `None`.
pub(super) fn body_str<'a>(
    body: &'a serde_json::Value,
    key: &str,
    allow_dash: bool,
) -> Result<Option<&'a str>, String> {
    match body.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) if s.is_empty() => Ok(None),
        Some(serde_json::Value::String(s)) => {
            safe_value(key, s, allow_dash)?;
            Ok(Some(s))
        }
        Some(_) => Err(format!("{key}: expected a string")),
    }
}

/// The wizard form's argv. Always `--json` (the page reads the report)
/// and `--yes` (a subprocess has no tty; `--yes` accepts ordinary
/// defaults and still never consents to the authorized_keys install —
/// that stays the explicit `install_key` field, exactly the CLI's own
/// consent rule).
pub(super) fn join_config_argv(body: &serde_json::Value) -> Result<Vec<String>, String> {
    let Some(label) = body_str(body, "label", false)? else {
        return Err("label: required".to_string());
    };
    validate_label(label)?;
    let mut argv = vec![
        "join-config".to_string(),
        label.to_string(),
        "--json".to_string(),
        "--yes".to_string(),
    ];
    if let Some(door) = body_str(body, "door", false)? {
        if !matches!(door, "a" | "b" | "nologin") {
            return Err("door: one of `a`, `b`, `nologin`".to_string());
        }
        argv.extend(["--door".to_string(), door.to_string()]);
    }
    for (key, flag) in [
        ("controller", "--controller"),
        ("data_path", "--data-path"),
        ("cloud_init_user", "--cloud-init-user"),
    ] {
        if let Some(v) = body_str(body, key, false)? {
            argv.extend([flag.to_string(), v.to_string()]);
        }
    }
    if let Some(share) = body.get("gpu_ram_share").and_then(|v| v.as_f64()) {
        argv.extend(["--gpu-ram-share".to_string(), share.to_string()]);
    }
    for (key, flag) in [
        ("cloud_init", "--cloud-init"),
        ("regen", "--regen"),
        ("install_key", "--install-key"),
        ("dry_run", "--dry-run"),
    ] {
        if body.get(key).and_then(|v| v.as_bool()) == Some(true) {
            argv.push(flag.to_string());
        }
    }
    Ok(argv)
}

/// The publish form's argv. The args tail rides behind a standalone
/// `--` (the manifest's own args), which is also why it is the one
/// place flag-shaped values are legal.
pub(super) fn publish_argv(body: &serde_json::Value) -> Result<Vec<String>, String> {
    let mut argv = vec!["publish".to_string(), "--json".to_string()];
    if let Some(source) = body_str(body, "source", false)? {
        argv.push(source.to_string());
    }
    for (key, flag) in [("bin", "--bin"), ("cwd", "--cwd"), ("build", "--build")] {
        if let Some(v) = body_str(body, key, false)? {
            argv.extend([flag.to_string(), v.to_string()]);
        }
    }
    if body.get("no_build").and_then(|v| v.as_bool()) == Some(true) {
        argv.push("--no-build".to_string());
    }
    if let Some(args) = body.get("args") {
        let Some(args) = args.as_array() else {
            return Err("args: expected an array of strings".to_string());
        };
        if !args.is_empty() {
            argv.push("--".to_string());
            for a in args {
                let Some(a) = a.as_str() else {
                    return Err("args: expected an array of strings".to_string());
                };
                safe_value("args", a, true)?;
                argv.push(a.to_string());
            }
        }
    }
    Ok(argv)
}

/// The exact argv a route drives — pure, so tests pin it without
/// spawning anything. `--env` (position-independent) rather than the
/// `@env` sigil: the sigil is a human spelling.
pub(super) fn route_argv(path: &str, env: Option<&str>) -> Vec<String> {
    let mut argv: Vec<String> = Vec::new();
    if let Some(e) = env
        && path != "/api/config"
    {
        argv.extend(["--env".to_string(), e.to_string()]);
    }
    match path {
        "/api/probe" => argv.extend(["probe".to_string(), "--json".to_string()]),
        "/api/status" => argv.extend(["status".to_string(), "--json".to_string()]),
        "/api/config" => {
            // `config show` takes the env as a positional, not via
            // `--env`: it inspects a named overlay rather than running
            // under one.
            argv.extend(["config".to_string(), "show".to_string()]);
            if let Some(e) = env {
                argv.push(e.to_string());
            }
        }
        _ => {}
    }
    argv
}

/// Ask a driven command for the colours it would use on a terminal.
///
/// Every driven command gets this, not just the streamed ones: a pane
/// is a human reading output too, and `fdl config show` renders its keys
/// and per-key provenance in colour that a pipe would otherwise discard.
/// Safe for the `--json` routes because fdl never styles a JSON body —
/// the styling lives on the human paths and on stderr, both of which the
/// page renders through its ANSI reader.
///
/// `run.rs` forwards these across the docker boundary when they are set,
/// which is what keeps a containerized tool's colours (cargo's) alive.
pub(super) fn ask_for_color(cmd: &mut Command) -> &mut Command {
    cmd.env("FORCE_COLOR", "1")
        .env("CLICOLOR_FORCE", "1")
        .env("CARGO_TERM_COLOR", "always")
}

/// Drive one `fdl` subcommand and report it verbatim: argv (the
/// reproducible command line the page displays), exit code, both
/// streams. Never an error shape — a failed command IS the result.
pub(super) fn run_fdl(fdl_bin: &Path, root: &Path, args: &[&str]) -> serde_json::Value {
    let mut cmd = Command::new(fdl_bin);
    let out = ask_for_color(&mut cmd)
        .args(args)
        .current_dir(root)
        .output();
    let mut cmd = vec!["fdl".to_string()];
    cmd.extend(args.iter().map(|s| s.to_string()));
    match out {
        Ok(out) => serde_json::json!({
            "cmd": cmd,
            "exit": out.status.code(),
            "stdout": String::from_utf8_lossy(&out.stdout),
            "stderr": String::from_utf8_lossy(&out.stderr),
        }),
        Err(e) => serde_json::json!({
            "cmd": cmd,
            "exit": serde_json::Value::Null,
            "stdout": "",
            "stderr": format!("cannot spawn {}: {e}", fdl_bin.display()),
        }),
    }
}

// ── Launch: the configured commands, driven with their own schema ──────

/// The `commands:` tree of the merged config (base, or base + the
/// named overlay — a farm's run command usually lives in its overlay),
/// each carrying the schema that command actually resolves to.
///
/// **A schema belongs to a command's own directory, not to the root.**
/// A `path:` command (the convention `./<name>/fdl.yml`, which is how a
/// training vehicle is declared) resolves its surface through
/// `load_command_with_env` — cached `--fdl-schema` output when fresh,
/// the inline `schema:` block otherwise — and that cache lives under
/// `<name>/.fdl/schema-cache/`. Reading a root-level cache instead
/// found nothing for anything, so every command claimed "no schema"
/// while `ddp-bench/.fdl/schema-cache/ddp-bench.json` sat right there.
///
/// The load is deliberately the **cheap** one: a page load must never
/// trigger a cargo compile. A `compile: true` command whose cache is
/// stale therefore reports no schema rather than blocking the page —
/// `fdl <cmd> --refresh-schema` is the (named) way to refill it.
///
/// `kind` rides along so the page can be honest about *why* a form is
/// absent: a `run:` command is a shell line and never grows one.
pub(super) fn commands_route(env: Option<&str>, server: &UiServer) -> Vec<u8> {
    let Some(base) = crate::config::find_project_config(&server.root) else {
        return json_ok(&serde_json::json!({ "commands": [] }));
    };
    let project = match crate::config::load_project_with_env(&base, env) {
        Ok(p) => p,
        Err(e) => return error_json("400 Bad Request", &format!("config: {e}")),
    };
    let commands: Vec<serde_json::Value> = project
        .commands
        .iter()
        .map(|(name, spec)| {
            let kind = match spec.kind() {
                Ok(crate::config::CommandKind::Run) => "run",
                Ok(crate::config::CommandKind::Path) => "path",
                Ok(crate::config::CommandKind::Preset) => "preset",
                Err(_) => "invalid",
            };
            // Only a path command owns a directory, and therefore a
            // schema. Its own `description:` is better than the parent's
            // stub, so prefer it when the child config carries one.
            let (schema, description) = if kind == "path" {
                let dir = spec.resolve_path(name, &server.root);
                match crate::config::load_command_with_env(&dir, env) {
                    Ok(child) => (
                        child.schema.and_then(|s| serde_json::to_value(s).ok()),
                        child.description.or_else(|| spec.description.clone()),
                    ),
                    Err(_) => (None, spec.description.clone()),
                }
            } else {
                (None, spec.description.clone())
            };
            serde_json::json!({
                "name": name,
                "description": description,
                "cluster": spec.cluster.unwrap_or(false),
                "kind": kind,
                "schema": schema,
            })
        })
        .collect();
    json_ok(&serde_json::json!({ "commands": commands }))
}

/// fdl's own options, which sit BEFORE the command and apply to every
/// one of them (`fdl [options] <command> [command-options]`). They are
/// not in any command's schema — they belong to fdl — so the form has
/// to carry them separately or they are simply unreachable from the
/// page.
///
/// Taken as STRUCTURED fields rather than raw flags, like every other
/// driven form here: the allowlist is then implicit and a body cannot
/// smuggle an arbitrary pre-command argument.
///
/// `--ansi` / `--no-ansi` are deliberately absent: colour already
/// auto-disables off a tty, so a driven command's output is plain and
/// the only thing `--ansi` could add is escape sequences in the log.
pub(super) fn global_argv(body: &serde_json::Value) -> Result<Vec<String>, String> {
    let mut argv = Vec::new();
    if let Some(v) = body_str(body, "verbosity", true)? {
        if !matches!(v, "-q" | "-v" | "-vv" | "-vvv") {
            return Err("verbosity: one of `-q`, `-v`, `-vv`, `-vvv`".to_string());
        }
        argv.push(v.to_string());
    }
    if let Some(spec) = body_str(body, "gpus", false)? {
        // fdl's own parser is the authority on the spec; this only
        // keeps a value from posing as a flag.
        if spec != "all" && !spec.chars().all(|c| c.is_ascii_digit() || c == ',') {
            return Err("gpus: a device list like `0,1`, or `all`".to_string());
        }
        argv.extend(["--gpus".to_string(), spec.to_string()]);
    }
    for (key, flag) in [
        ("no_append", "--no-append"),
        ("no_prebuild", "--no-prebuild"),
    ] {
        if body.get(key).and_then(|v| v.as_bool()) == Some(true) {
            argv.push(flag.to_string());
        }
    }
    Ok(argv)
}

/// The launch argv: `[globals] [--env <farm>] <command> <args...>`. The
/// command must be declared in the merged config — the launch surface
/// drives the project's own commands, never arbitrary fdl subcommands
/// (those have their own routes and their own consent rules). Args are
/// the command's own flags, so dashes are the point; they are still
/// bounded and control-character-free.
pub(super) fn launch_argv(
    body: &serde_json::Value,
    server: &UiServer,
) -> Result<(Vec<String>, Option<String>), String> {
    let Some(command) = body_str(body, "command", false)? else {
        return Err("command: required".to_string());
    };
    let env = body_str(body, "env", false)?;
    if let Some(e) = env {
        validate_label(e)?;
    }
    let base = crate::config::find_project_config(&server.root)
        .ok_or("no fdl.yml here — nothing to launch")?;
    let project =
        crate::config::load_project_with_env(&base, env).map_err(|e| format!("config: {e}"))?;
    if !project.commands.contains_key(command) {
        return Err(format!(
            "command `{command}` is not declared in this project's `commands:`{}",
            env.map(|e| format!(" (env `{e}`)")).unwrap_or_default(),
        ));
    }
    // Everything fdl-level comes first, in the order its own usage
    // line states: `fdl [options] <command> [command-options]`.
    let mut argv: Vec<String> = global_argv(body)?;
    if let Some(e) = env {
        argv.extend(["--env".to_string(), e.to_string()]);
    }
    argv.push(command.to_string());
    if let Some(args) = body.get("args") {
        let Some(args) = args.as_array() else {
            return Err("args: expected an array of strings".to_string());
        };
        for a in args {
            let Some(a) = a.as_str() else {
                return Err("args: expected an array of strings".to_string());
            };
            safe_value("args", a, true)?;
            argv.push(a.to_string());
        }
    }
    Ok((argv, env.map(str::to_string)))
}

/// Append one launch record to the run ledger. Failures warn and never
/// break anything (`record_log`'s philosophy: a full disk must not
/// kill a run — nor, here, the stream reporting on it).
pub(super) fn append_ledger(path: &Path, record: &serde_json::Value) {
    let write = || -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f, "{record}")
    };
    if let Err(e) = write() {
        eprintln!(
            "fdl ui: cannot append the run ledger at {}: {e}",
            path.display(),
        );
    }
}
