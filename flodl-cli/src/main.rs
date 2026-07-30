//! flodl-cli: command-line tool for the floDl deep learning framework.
//!
//! Provides hardware diagnostics, libtorch management, and project scaffolding.
//! Pure Rust binary with no libtorch dependency (GPU detection via nvidia-smi).
//!
//! Works both inside a floDl project and standalone. When standalone, libtorch
//! is managed under `~/.flodl/` (override with `$FLODL_HOME`).

use flodl_cli::{
    add, api_ref, builtins, cli_error, cluster, completions, config, diagnose, gpus, init,
    overlay, parse_or_schema_from, probe, run, setup, skill, status, style, update_check,
};

use builtins::{
    AddArgs, ApiRefArgs, DiagnoseArgs, InitArgs, InstallArgs, ProbeArgs, SetupArgs,
    SkillInstallArgs, StartArgs, StatusArgs,
};

use std::env;
use std::process::ExitCode;

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    // RAII guard fires the daily crates.io update check from Drop, so
    // it runs after the user's command output regardless of which
    // match arm computes the ExitCode (and on early `return`s too).
    // The check is silent on every failure mode and respects
    // `FDL_NO_UPDATE_CHECK=1`, `CI=true`, in-container detection, and
    // the `update_check.enabled` config field.
    let _update_check_guard = update_check::Guard::new();

    let raw_args: Vec<String> = env::args().collect();

    // Extract global color flags before anything else — subsequent help
    // rendering and error messages must already honour the choice.
    let args = match extract_ansi_flags(&raw_args) {
        Ok((args, choice)) => {
            if let Some(c) = choice {
                style::set_color_choice(c);
                // Propagate to child processes (Docker, subprocess) so
                // nested `fdl` invocations inherit the choice.
                // SAFETY: called before any threads are spawned.
                unsafe {
                    env::set_var(
                        "FLODL_COLOR",
                        match c {
                            style::ColorChoice::Always => "always",
                            style::ColorChoice::Never => "never",
                            style::ColorChoice::Auto => "auto",
                        },
                    );
                }
            } else if let Ok(v) = env::var("FLODL_COLOR") {
                // Inherited from a parent `fdl` invocation.
                match v.as_str() {
                    "always" => style::set_color_choice(style::ColorChoice::Always),
                    "never" => style::set_color_choice(style::ColorChoice::Never),
                    _ => {}
                }
            }
            args
        }
        Err(msg) => {
            cli_error!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    // Extract global verbosity flags before command dispatch.
    // Sets FLODL_VERBOSITY so child processes (including Docker) inherit it.
    let (args, verbosity) = extract_verbosity(&args);
    if let Some(v) = verbosity {
        // SAFETY: called before any threads are spawned.
        unsafe {
            env::set_var("FLODL_VERBOSITY", v.to_string());
        }
    }

    // Export HOSTNAME so docker-compose's `hostname: ${HOSTNAME}` resolves to
    // the host's name (the cluster launcher matches
    // `cluster.hosts[i].name == hostname()`). Bash keeps HOSTNAME as a shell
    // variable but does NOT export it, so a non-interactive `fdl` invocation
    // otherwise leaves it unset -- compose warns ("The \"HOSTNAME\" variable
    // is not set. Defaulting to a blank string.") and the container hostname
    // comes up blank. Only fill when unset, so an explicit exported HOSTNAME
    // still wins.
    if env::var_os("HOSTNAME").is_none() {
        let host = crate::cluster::resolve_local_hostname();
        if !host.is_empty() {
            // SAFETY: called before any threads are spawned.
            unsafe {
                env::set_var("HOSTNAME", host);
            }
        }
    }

    // Extract `--no-append`, the escape hatch that suppresses any
    // `append:` suffix declared by a run-kind command. Scoped to this
    // invocation only — nested `fdl` calls re-evaluate their own flags.
    let (args, no_append) = extract_no_append(&args);

    // Extract `--no-prebuild`, which opts a single invocation out of
    // the cluster-mode pre-flight build (see [`prebuild`]). Lets users
    // skip the per-remote build phase when they know the binaries are
    // fresh (or when working around a build-only issue).
    let (args, no_prebuild) = extract_no_prebuild(&args);

    // Extract `--gpus` (global; accepted at any position). Cluster-aware
    // commands with N>=2 GPUs trigger single-host envelope synthesis and
    // multi-process spawn. Non-cluster commands map `--gpus` to
    // `CUDA_VISIBLE_DEVICES` on the single child process. Recursive
    // invocations (FLODL_INTERNAL_CLUSTER_JSON set) shouldn't normally see this
    // flag in their args -- the launcher strips it before re-exec'ing.
    let (args, gpus_spec) = match extract_gpus_flag(&args) {
        Ok(pair) => pair,
        Err(msg) => {
            cli_error!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    // Environment selection: `@env` / `--env X` (command-line, equivalent)
    // override `FDL_ENV=X`. Every form must resolve to an existing overlay
    // or it is a loud error — there is no positional-env convention, so the
    // first bare token is always a command.
    let fdl_env_var = env::var("FDL_ENV").ok();
    let (active_env, args) = match resolve_env(&args, fdl_env_var.as_deref()) {
        Ok(pair) => pair,
        Err(msg) => {
            cli_error!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    // Propagate the active env name to child processes so they can
    // discover the overlay-merged config at runtime. Previously only
    // exported inside `prepare_cluster_env` (cluster-launcher path);
    // unconditional propagation makes the env name available to any
    // spawned binary — load-bearing for test discovery (each test
    // resolves its cluster topology from
    // `fdl.<FDL_ENV>.yml`) and harmless for non-test commands.
    // No-op when no env is active (FDL_ENV stays unset).
    if let Some(env_name) = active_env.as_deref() {
        // SAFETY: main() has not spawned threads at this point;
        // matches the invariant documented for
        // `prepare_cluster_env` and `apply_cuda_visible_devices`.
        unsafe {
            env::set_var(cluster::ENV_FDL_ENV, env_name);
        }
    }

    // Bare `fdl` with no args behaves like `fdl --help`.
    let cmd = args.get(1).map(String::as_str).unwrap_or("--help");

    match cmd {
        "setup" => {
            let cli: SetupArgs = parse_sub("fdl setup", &args[1..]);
            let opts = setup::SetupOpts {
                non_interactive: cli.non_interactive,
                force: cli.force,
            };
            match setup::run(opts) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "libtorch" => dispatch_libtorch(&args),
        "nccl" => dispatch_nccl(&args),
        "diagnose" => {
            let cli: DiagnoseArgs = parse_sub("fdl diagnose", &args[1..]);
            diagnose::run(cli.json);
            ExitCode::SUCCESS
        }
        "probe" => {
            let cli: ProbeArgs = parse_sub("fdl probe", &args[1..]);
            let code = probe::run(
                cli.json,
                cli.skip_mount,
                cli.data_path,
                cli.libtorch_path,
                cli.docker,
            );
            if code == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        "status" => {
            let cli: StatusArgs = parse_sub("fdl status", &args[1..]);
            let code = status::run(cli.json, cli.addr.as_deref());
            if code == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        "start" => {
            let cli: StartArgs = parse_sub("fdl start", &args[1..]);
            let code = status::run_start(cli.addr.as_deref(), cli.token.as_deref());
            if code == 0 { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        "api-ref" => {
            let cli: ApiRefArgs = parse_sub("fdl api-ref", &args[1..]);
            match api_ref::run(cli.json, cli.path.as_deref()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "init" => {
            let cli: InitArgs = parse_sub("fdl init", &args[1..]);
            match init::run(cli.name.as_deref(), cli.docker, cli.native, cli.with_hf) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "add" => {
            let cli: AddArgs = parse_sub("fdl add", &args[1..]);
            match add::run(cli.target.as_deref(), cli.playground, cli.install) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "install" => {
            let cli: InstallArgs = parse_sub("fdl install", &args[1..]);
            cmd_install(cli.check, cli.dev)
        }
        "skill" => dispatch_skill(&args),
        "schema" => dispatch_schema(&args),
        "completions" => {
            let shell = args.get(2).map(String::as_str).unwrap_or("bash");
            let cwd = env::current_dir().unwrap_or_default();
            let project = load_project_config(&cwd, active_env.as_deref());
            completions::generate(
                shell,
                project.as_ref().map(|(p, r)| (p, r.as_path())),
                active_env.as_deref(),
            );
            ExitCode::SUCCESS
        }
        "autocomplete" => {
            let cwd = env::current_dir().unwrap_or_default();
            let project = load_project_config(&cwd, active_env.as_deref());
            completions::autocomplete(project.as_ref().map(|(p, r)| (p, r.as_path())));
            ExitCode::SUCCESS
        }
        "config" => cmd_config_show(&args[1..], active_env.as_deref()),
        "--help" | "-h" => {
            let cwd = env::current_dir().unwrap_or_default();
            // Never let a bogus env selector block --help: degrade to base
            // help when the overlay is missing (feedback_help_never_blocked).
            let help_env = help_env_or_note(&cwd, active_env.as_deref());
            if let Some((project, root)) = load_project_config(&cwd, help_env) {
                run::print_project_help(&project, &root, help_env);
            } else {
                print_usage();
            }
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => dispatch_config(
            other,
            &args,
            active_env.as_deref(),
            no_append,
            no_prebuild,
            gpus_spec.as_ref(),
        ),
    }
}

/// Resolve the active environment selector.
///
/// Three forms, no positional convention:
///   * `@env` sigil token (`fdl @cluster probe`), accepted anywhere before
///     a standalone `--`, exactly like `--env`.
///   * `--env X` / `--env=X` flag (scan-anywhere).
///   * `FDL_ENV=X` environment variable (`fdl_env`).
///
/// `@env` and `--env` are command-line selectors and rank equal; supplying
/// both with *different* values is a loud conflict error, and either one
/// overrides `FDL_ENV`. Because there is no positional-env convention, the
/// first bare token is unconditionally a command (no command/env name
/// collision is possible).
///
/// Overlay EXISTENCE is deliberately NOT validated here — that happens at the
/// config-load path ([`config::load_project_with_env`] →
/// `resolve_config_layers`), which errors loudly for any command that actually
/// consumes the config. Validating upfront would block `--help` and
/// env-agnostic builtins (e.g. `fdl setup` with a stale `FDL_ENV`), so
/// resolution and validation are kept separate (`feedback_help_never_blocked`).
///
/// `fdl_env` is injected for testability; `main` reads it from the process
/// environment once at startup.
fn resolve_env(
    args: &[String],
    fdl_env: Option<&str>,
) -> Result<(Option<String>, Vec<String>), String> {
    // Strip both command-line selectors up front.
    let (args, flag_env) = extract_env_flag(args)?;
    let (args, at_env) = extract_at_env(&args)?;

    // Reconcile `@env` and `--env` — equal rank, conflict if they disagree.
    let cli_env = match (flag_env, at_env) {
        (Some(f), Some(a)) if f != a => {
            return Err(format!(
                "conflicting env selectors: `--env {f}` and `@{a}` — pick one"
            ));
        }
        (Some(v), _) | (None, Some(v)) => Some(v),
        (None, None) => None,
    };

    // A command-line selector wins over the ambient `FDL_ENV`. Existence of
    // the overlay is validated later, at the config-load path (see the fn doc).
    if let Some(env_name) = cli_env {
        return Ok((Some(env_name), args));
    }
    if let Some(env_name) = fdl_env {
        if !env_name.is_empty() {
            return Ok((Some(env_name.to_string()), args));
        }
    }

    Ok((None, args))
}

/// Extract the global `--gpus` flag. Accepted at any position; both forms:
/// `--gpus 0,1` and `--gpus=0,1`. Errors on duplicate, missing value, or
/// value that looks like another flag.
///
/// Scan-anywhere, but stops at the first standalone `--`: a `--gpus` token
/// past the separator is bound for the inner command and forwarded untouched
/// (consistent with [`extract_env_flag`] and the other global-flag extractors).
///
/// Returns the args with `--gpus` (and its value) removed, plus the parsed
/// [`gpus::GpusSpec`] when set.
fn extract_gpus_flag(
    args: &[String],
) -> Result<(Vec<String>, Option<gpus::GpusSpec>), String> {
    let mut out = Vec::with_capacity(args.len());
    let mut spec: Option<gpus::GpusSpec> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            // Everything from here on is bound for the inner command;
            // forward it verbatim (consistent with the other global-flag
            // extractors). A `--gpus` past the separator is the script's.
            out.extend(args[i..].iter().cloned());
            break;
        }
        if a == "--gpus" {
            let value = args.get(i + 1).ok_or_else(|| {
                "--gpus requires a value (e.g. `--gpus 0,1` or `--gpus all`)"
                    .to_string()
            })?;
            if value.is_empty() || value.starts_with('-') {
                return Err(format!("--gpus requires a value, got `{value}`"));
            }
            if spec.is_some() {
                return Err("--gpus specified more than once".to_string());
            }
            spec = Some(gpus::GpusSpec::parse(value)?);
            i += 2;
            continue;
        }
        if let Some(value) = a.strip_prefix("--gpus=") {
            if spec.is_some() {
                return Err("--gpus specified more than once".to_string());
            }
            if value.is_empty() {
                return Err(
                    "--gpus= requires a value (e.g. `--gpus=0,1`)".to_string(),
                );
            }
            spec = Some(gpus::GpusSpec::parse(value)?);
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    Ok((out, spec))
}

/// Strip `--env <value>` / `--env=<value>` tokens from `args`.
///
/// Accepts either long-separated (`--env ci`) or equals-joined
/// (`--env=ci`) form. Errors on missing value, empty value, or duplicate
/// occurrence. Returns `(filtered_args, Some(value))` on success, or
/// `(filtered_args, None)` when the flag is absent.
///
/// Scan-anywhere, but stops at the first standalone `--`: a `--env` token
/// past the separator is bound for the inner command and forwarded
/// untouched (consistent with [`extract_at_env`] and the other global
/// flag extractors).
fn extract_env_flag(args: &[String]) -> Result<(Vec<String>, Option<String>), String> {
    let mut out = Vec::with_capacity(args.len());
    let mut env: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            // Everything from here on is forwarded verbatim.
            out.extend(args[i..].iter().cloned());
            break;
        }
        if a == "--env" {
            let value = args.get(i + 1).ok_or_else(|| {
                "--env requires a value (e.g. `--env ci`)".to_string()
            })?;
            if value.is_empty() || value.starts_with('-') {
                return Err(format!("--env requires a value, got `{value}`"));
            }
            if env.is_some() {
                return Err("--env specified more than once".to_string());
            }
            env = Some(value.clone());
            i += 2;
            continue;
        }
        if let Some(value) = a.strip_prefix("--env=") {
            if env.is_some() {
                return Err("--env specified more than once".to_string());
            }
            if value.is_empty() {
                return Err("--env= requires a value (e.g. `--env=ci`)".to_string());
            }
            env = Some(value.to_string());
            i += 1;
            continue;
        }
        out.push(a.clone());
        i += 1;
    }
    Ok((out, env))
}

/// Strip a single `@<env>` selector token from `args`, returning the env
/// name without the leading `@`.
///
/// The `@<env>` sigil is a PRE-COMMAND selector: it is recognized only among
/// the tokens *before* the command (`fdl @cluster probe`), never after it.
/// The first bare token (not a flag, not `@`-prefixed) is the command — from
/// there on nothing is treated as an env selector, so an `@`-prefixed option
/// value (`fdl train --tag @best`) or a positional the command owns is
/// forwarded verbatim rather than mistaken for an env selector. A standalone
/// `--` before the command also ends recognition. Index 0 (the program name)
/// is never inspected. By the time this runs every other global flag has been
/// stripped upstream (and `--env` inside `resolve_env`), so the first bare
/// token really is the command. Errors on a bare `@` (no name) or on more
/// than one `@`-token.
fn extract_at_env(args: &[String]) -> Result<(Vec<String>, Option<String>), String> {
    let mut out = Vec::with_capacity(args.len());
    let mut env: Option<String> = None;
    // Set once the command token (or a `--`) is reached: everything from there
    // on is the command and its arguments, forwarded verbatim.
    let mut command_seen = false;
    for (i, arg) in args.iter().enumerate() {
        if i == 0 || command_seen {
            out.push(arg.clone());
            continue;
        }
        if arg == "--" {
            command_seen = true;
            out.push(arg.clone());
            continue;
        }
        if let Some(name) = arg.strip_prefix('@') {
            if name.is_empty() {
                return Err(
                    "`@` requires an env name (e.g. `@cluster`)".to_string(),
                );
            }
            if env.is_some() {
                return Err(
                    "env selector (`@<env>`) specified more than once".to_string(),
                );
            }
            env = Some(name.to_string());
            continue;
        }
        // First non-flag, non-`@` token = the command; end recognition. A
        // leading global flag that survived upstream extraction starts with
        // `-` and is tolerated (skipped) until the command is reached.
        if !arg.starts_with('-') {
            command_seen = true;
        }
        out.push(arg.clone());
    }
    Ok((out, env))
}

/// For `--help` only: degrade to base help when the selected env's overlay is
/// missing, rather than hard-erroring at config load. `--help` must always
/// render (`feedback_help_never_blocked`); a typo'd `@env` / `FDL_ENV` should
/// still show help, with a note. Returns the env to load with: the original
/// `env` when its overlay exists (or there is no project), else `None` plus a
/// stderr note. Overlay *content* errors still surface normally at load.
fn help_env_or_note<'a>(cwd: &std::path::Path, env: Option<&'a str>) -> Option<&'a str> {
    let name = env?;
    match config::find_config(cwd) {
        // Overlay present → merge it as usual.
        Some(base) if overlay::find_env_file(&base, name).is_some() => Some(name),
        // In a project but the overlay is missing → base help + note.
        Some(_) => {
            eprintln!("fdl: env `{name}` not found; showing base help");
            None
        }
        // No project at all → load_project_config returns None → print_usage.
        None => None,
    }
}

/// Thin wrapper over `parse_or_schema_from` that sets the program name
/// shown in `--help` output so `fdl setup --help` looks like
/// "fdl setup" rather than the crate name.
pub(crate) fn parse_sub<T: flodl_cli::FdlArgsTrait>(program: &str, tail: &[String]) -> T {
    let mut argv = Vec::with_capacity(tail.len() + 1);
    argv.push(program.to_string());
    // tail[0] is the sub-command name (e.g. "setup"); skip it so the derive
    // only sees flags and positionals that belong to the sub-command.
    argv.extend(tail.iter().skip(1).cloned());
    parse_or_schema_from::<T>(&argv)
}


// ---------------------------------------------------------------------------
// skill dispatch
// ---------------------------------------------------------------------------

fn dispatch_skill(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("--help");
    match sub {
        "install" => {
            let cli: SkillInstallArgs = parse_sub("fdl skill install", &args[2..]);
            match skill::install(cli.tool.as_deref(), cli.skill.as_deref()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    cli_error!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "list" => {
            skill::list();
            ExitCode::SUCCESS
        }
        "--help" | "-h" => {
            skill::print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown skill command: {other}");
            skill::print_usage();
            ExitCode::FAILURE
        }
    }
}


mod cli;
use cli::libtorch::dispatch_libtorch;
use cli::nccl::dispatch_nccl;
use cli::schema::dispatch_schema;
use cli::install::cmd_install;
use cli::config::{cmd_config_show, dispatch_config, load_project_config};

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

pub(crate) fn print_usage() {
    println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("The floDl companion tool: setup, libtorch, diagnostics, API reference.");
    println!("Works anywhere. Uses project root when available, ~/.flodl/ otherwise.");
    println!();
    println!("USAGE:");
    println!("    fdl [options] <command> [command-options]");
    println!();
    println!("GLOBAL OPTIONS:");
    println!("    --env <name>       Use fdl.<name>.yml overlay (also: FDL_ENV=<name>)");
    println!("    --gpus <spec>      Scope visible GPUs, e.g. 0,1 or all (any position)");
    println!("    --ansi             Force ANSI color output");
    println!("    --no-ansi          Disable ANSI color output");
    println!("    -v                 Verbose output (DDP sync, data loading detail)");
    println!("    -vv                Debug output (per-batch timing, loop internals)");
    println!("    -vvv               Trace output (maximum detail)");
    println!("    -q, --quiet        Suppress all non-error output");
    println!("    --no-append        Drop a run command's `append:` suffix (cargo / runner defaults)");
    println!("    --no-prebuild      Skip the cluster pre-flight build (assumes binaries are fresh)");
    println!();
    println!("COMMANDS:");
    println!("    setup              Interactive guided setup");
    println!("    libtorch           Manage libtorch installations");
    println!("    init <name>        Scaffold a new floDl project");
    println!("        --docker       Generate Docker-based scaffold (libtorch baked in)");
    println!("    add <target>       Add a flodl ecosystem crate (currently: flodl-hf)");
    println!("    diagnose           System and GPU diagnostics");
    println!("        --json         Output as JSON");
    println!("    install             Install or update fdl globally (~/.local/bin)");
    println!("        --check        Check for updates without installing");
    println!("        --dev          Symlink to current binary (tracks local builds)");
    println!("    skill              Manage AI coding assistant skills");
    println!("        install        Install skills for detected tool (Claude, Cursor, ...)");
    println!("        list           Show available skills");
    println!("    api-ref            Generate flodl API reference");
    println!("        --json         Output as JSON");
    println!("        --path <dir>   Explicit flodl source path");
    println!("    version            Show version");
    println!();
    println!("Run `fdl --help` or `fdl <command> --help` for details.");
    println!();
    println!("INSTALL:");
    println!("    cargo install flodl-cli    # from crates.io");
    println!("    fdl install                # make current binary global (~/.local/bin/fdl)");
    println!();
    println!("EXAMPLES:");
    println!("    fdl setup                  # first-time setup");
    println!("    fdl libtorch download      # download pre-built libtorch");
    println!("    fdl libtorch list          # show installed variants");
    println!("    fdl init my-model          # scaffold with mounted libtorch");
    println!("    fdl diagnose               # hardware + compatibility report");
    println!("    fdl diagnose --json        # machine-readable output");
    println!("    fdl api-ref                # generate API reference");
    println!("    fdl api-ref --json         # structured JSON for tooling");
}

// ---------------------------------------------------------------------------
// Global verbosity flags
// ---------------------------------------------------------------------------

/// Extract verbosity flags from args, returning filtered args and the
/// `FLODL_VERBOSITY` value: Quiet=0, Normal=1, Verbose=2, Debug=3, Trace=4.
///
/// Supports `-v` (Verbose), `-vv` (Debug), `-vvv` (Trace), `--quiet`/`-q` (Quiet).
/// Flags can appear anywhere in the arg list and are stripped before dispatch.
fn extract_verbosity(args: &[String]) -> (Vec<String>, Option<u8>) {
    let mut level: Option<u8> = None;
    let mut filtered = Vec::with_capacity(args.len());
    let mut past_dashdash = false;

    for arg in args {
        // A verbosity flag after the first standalone `--` is bound for the
        // inner command; forward it (and everything after) verbatim, matching
        // the other global-flag extractors.
        if past_dashdash {
            filtered.push(arg.clone());
            continue;
        }
        if arg == "--" {
            past_dashdash = true;
            filtered.push(arg.clone());
            continue;
        }
        match arg.as_str() {
            "-vvv" => level = Some(4), // Trace
            "-vv" => level = Some(3),  // Debug
            "-v" => level = Some(2),   // Verbose
            "--quiet" | "-q" => level = Some(0), // Quiet
            _ => filtered.push(arg.clone()),
        }
    }

    (filtered, level)
}

/// Strip `--no-append` from `args`, returning a bool that is true
/// when the flag was present anywhere in the input. Scan-anywhere,
/// consistent with `-v`, `--env`, and the ANSI flags. The flag only
/// affects run-kind commands (it suppresses their `append:` suffix);
/// on commands with no append entry it is a silent no-op.
///
/// Stops scanning at the first standalone `--`, so a literal
/// `--no-append` passed *after* the user's `--` is forwarded to the
/// inner script untouched (an escape hatch for run-scripts whose own
/// proxy flags happen to collide).
fn extract_no_append(args: &[String]) -> (Vec<String>, bool) {
    let mut found = false;
    let mut filtered = Vec::with_capacity(args.len());
    let mut past_dashdash = false;

    for arg in args {
        if past_dashdash {
            filtered.push(arg.clone());
            continue;
        }
        if arg == "--" {
            past_dashdash = true;
            filtered.push(arg.clone());
            continue;
        }
        if arg == "--no-append" {
            found = true;
            continue;
        }
        filtered.push(arg.clone());
    }

    (filtered, found)
}

/// Strip `--no-prebuild` from `args`, returning a bool that's true
/// when the flag was present. Symmetric with [`extract_no_append`];
/// stops scanning at the first standalone `--` so a literal
/// `--no-prebuild` after the user's `--` reaches the inner script.
fn extract_no_prebuild(args: &[String]) -> (Vec<String>, bool) {
    let mut found = false;
    let mut filtered = Vec::with_capacity(args.len());
    let mut past_dashdash = false;

    for arg in args {
        if past_dashdash {
            filtered.push(arg.clone());
            continue;
        }
        if arg == "--" {
            past_dashdash = true;
            filtered.push(arg.clone());
            continue;
        }
        if arg == "--no-prebuild" {
            found = true;
            continue;
        }
        filtered.push(arg.clone());
    }

    (filtered, found)
}

/// Strip `--ansi` / `--no-ansi` from `args`, returning a
/// [`style::ColorChoice`] override when either was present. Errors if
/// both are given (ambiguous). Scan-anywhere, consistent with `-v`
/// and `--env` — global flags aren't position-locked. Stops at the first
/// standalone `--`, so an `--ansi` / `--no-ansi` past the separator is
/// forwarded to the inner command untouched.
fn extract_ansi_flags(
    args: &[String],
) -> Result<(Vec<String>, Option<style::ColorChoice>), String> {
    let mut ansi = false;
    let mut no_ansi = false;
    let mut filtered = Vec::with_capacity(args.len());
    let mut past_dashdash = false;

    for arg in args {
        if past_dashdash {
            filtered.push(arg.clone());
            continue;
        }
        if arg == "--" {
            past_dashdash = true;
            filtered.push(arg.clone());
            continue;
        }
        match arg.as_str() {
            "--ansi" => ansi = true,
            "--no-ansi" => no_ansi = true,
            _ => filtered.push(arg.clone()),
        }
    }

    let choice = match (ansi, no_ansi) {
        (true, true) => return Err(
            "--ansi and --no-ansi are mutually exclusive".to_string()
        ),
        (true, false) => Some(style::ColorChoice::Always),
        (false, true) => Some(style::ColorChoice::Never),
        (false, false) => None,
    };
    Ok((filtered, choice))
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
