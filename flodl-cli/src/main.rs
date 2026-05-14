//! flodl-cli: command-line tool for the floDl deep learning framework.
//!
//! Provides hardware diagnostics, libtorch management, and project scaffolding.
//! Pure Rust binary with no libtorch dependency (GPU detection via nvidia-smi).
//!
//! Works both inside a floDl project and standalone. When standalone, libtorch
//! is managed under `~/.flodl/` (override with `$FLODL_HOME`).

use flodl_cli::{
    add, api_ref, builtins, cli_error, cluster, completions, config, context, diagnose, dispatch,
    init, libtorch, overlay, parse_or_schema_from, run, schema, schema_cache, setup, skill, style,
    update_check, util,
};

use builtins::{
    AddArgs, ApiRefArgs, DiagnoseArgs, InitArgs, InstallArgs, LibtorchActivateArgs,
    LibtorchBuildArgs, LibtorchDownloadArgs, LibtorchListArgs, LibtorchRemoveArgs, SchemaClearArgs,
    SchemaListArgs, SchemaRefreshArgs, SetupArgs, SkillInstallArgs,
};
use dispatch::{walk_commands, WalkOutcome};

use std::env;
use std::process::ExitCode;

use context::Context;

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

    // Extract `--no-append`, the escape hatch that suppresses any
    // `append:` suffix declared by a run-kind command. Scoped to this
    // invocation only — nested `fdl` calls re-evaluate their own flags.
    let (args, no_append) = extract_no_append(&args);

    // Environment selection: `--env X` > `FDL_ENV=X` > first-arg convention.
    // Explicit selectors must resolve to an existing overlay; first-arg
    // detection falls through when the arg matches no overlay. Ambiguous
    // cases (a command name also matches a sibling env file) are a loud
    // error rather than silent precedence.
    let cwd = env::current_dir().unwrap_or_default();
    let fdl_env_var = env::var("FDL_ENV").ok();
    let (active_env, args) = match resolve_env(&args, &cwd, fdl_env_var.as_deref()) {
        Ok(pair) => pair,
        Err(msg) => {
            cli_error!("{msg}");
            return ExitCode::FAILURE;
        }
    };

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
        "diagnose" => {
            let cli: DiagnoseArgs = parse_sub("fdl diagnose", &args[1..]);
            diagnose::run(cli.json);
            ExitCode::SUCCESS
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
            completions::generate(shell, project.as_ref().map(|(p, r)| (p, r.as_path())));
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
            if let Some((project, root)) = load_project_config(&cwd, active_env.as_deref()) {
                run::print_project_help(&project, &root, active_env.as_deref());
            } else {
                print_usage();
            }
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flodl-cli {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        other => dispatch_config(other, &args, active_env.as_deref(), no_append),
    }
}

/// Resolve the active environment selector.
///
/// Precedence (highest wins):
///   1. Explicit `--env X` / `--env=X` flag (scan-anywhere, like `-v`).
///   2. `FDL_ENV=X` environment variable (`fdl_env`).
///   3. First-arg convention: `fdl ci test` where `fdl.ci.yml` exists.
///
/// Explicit selectors (#1, #2) must resolve to an existing overlay — missing
/// files error loudly rather than silently falling through. First-arg
/// detection still falls through when the arg matches no overlay (it may
/// just be a command).
///
/// `cwd` and `fdl_env` are injected for testability; `main` reads them from
/// the process environment once at startup.
fn resolve_env(
    args: &[String],
    cwd: &std::path::Path,
    fdl_env: Option<&str>,
) -> Result<(Option<String>, Vec<String>), String> {
    // 1. Explicit flag wins — strip it from args before anything else.
    let (args, flag_env) = extract_env_flag(args)?;
    if let Some(ref env_name) = flag_env {
        validate_env_exists(env_name, "--env", cwd)?;
        return Ok((flag_env, args));
    }

    // 2. Environment variable, if set and non-empty.
    if let Some(env_name) = fdl_env {
        if !env_name.is_empty() {
            validate_env_exists(env_name, "FDL_ENV", cwd)?;
            return Ok((Some(env_name.to_string()), args));
        }
    }

    // 3. First-arg convention — returns None if no overlay matches.
    resolve_env_first_arg(&args, cwd)
}

/// Strip `--env <value>` / `--env=<value>` tokens from `args`.
///
/// Accepts either long-separated (`--env ci`) or equals-joined
/// (`--env=ci`) form. Errors on missing value, empty value, or duplicate
/// occurrence. Returns `(filtered_args, Some(value))` on success, or
/// `(filtered_args, None)` when the flag is absent.
fn extract_env_flag(args: &[String]) -> Result<(Vec<String>, Option<String>), String> {
    let mut out = Vec::with_capacity(args.len());
    let mut env: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
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

/// Confirm that `fdl.<env>.yml` exists next to the nearest base config,
/// erroring with the source (`--env` or `FDL_ENV`) when it doesn't.
fn validate_env_exists(
    env_name: &str,
    source: &str,
    cwd: &std::path::Path,
) -> Result<(), String> {
    let base_config = config::find_config(cwd).ok_or_else(|| {
        format!(
            "{source} `{env_name}` set but no fdl.yml found in {} or parents",
            cwd.display()
        )
    })?;
    if overlay::find_env_file(&base_config, env_name).is_none() {
        return Err(format!(
            "{source} `{env_name}`: overlay not found \
             (expected fdl.{env_name}.yml next to {})",
            base_config.display()
        ));
    }
    Ok(())
}

/// First-arg environment resolution. Returns `(Some(env), args_without_env)`
/// when the first positional matches a sibling `fdl.<arg>.yml` overlay and
/// no built-in, script, or sub-command by that name exists. Returns
/// `(None, args)` when no env applies. Errors on ambiguity.
fn resolve_env_first_arg(
    args: &[String],
    cwd: &std::path::Path,
) -> Result<(Option<String>, Vec<String>), String> {
    let candidate = match args.get(1) {
        Some(a) if !a.starts_with('-') => a,
        _ => return Ok((None, args.to_vec())),
    };

    let base_config = match config::find_config(cwd) {
        Some(p) => p,
        None => return Ok((None, args.to_vec())),
    };
    let env_file = overlay::find_env_file(&base_config, candidate);
    if env_file.is_none() {
        return Ok((None, args.to_vec()));
    }

    // An overlay exists — check for collision with a command of the same name.
    let is_command = is_builtin_name(candidate) || is_project_command(&base_config, candidate);
    if is_command {
        return Err(format!(
            "ambiguous `{candidate}`: matches both a command and an env overlay \
             (fdl.{candidate}.yml).\nResolve by renaming one."
        ));
    }

    // Unambiguously an env; consume it and return the rest.
    let mut rest = Vec::with_capacity(args.len() - 1);
    rest.push(args[0].clone());
    rest.extend(args.iter().skip(2).cloned());
    Ok((Some(candidate.clone()), rest))
}

fn is_builtin_name(name: &str) -> bool {
    builtins::is_builtin_name(name)
}

fn is_project_command(base_config: &std::path::Path, name: &str) -> bool {
    // Must NOT merge env overlays here — that would be circular when the
    // env name also matches a command key. Inspect the raw base only.
    let Ok(project) = config::load_project_with_env(base_config, None) else {
        return false;
    };
    project.commands.contains_key(name)
}

/// Thin wrapper over `parse_or_schema_from` that sets the program name
/// shown in `--help` output so `fdl setup --help` looks like
/// "fdl setup" rather than the crate name.
fn parse_sub<T: flodl_cli::FdlArgsTrait>(program: &str, tail: &[String]) -> T {
    let mut argv = Vec::with_capacity(tail.len() + 1);
    argv.push(program.to_string());
    // tail[0] is the sub-command name (e.g. "setup"); skip it so the derive
    // only sees flags and positionals that belong to the sub-command.
    argv.extend(tail.iter().skip(1).cloned());
    parse_or_schema_from::<T>(&argv)
}

// ---------------------------------------------------------------------------
// libtorch dispatch
// ---------------------------------------------------------------------------

fn dispatch_libtorch(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("--help");
    match sub {
        "list" => {
            let cli: LibtorchListArgs = parse_sub("fdl libtorch list", &args[2..]);
            cmd_libtorch_list(cli.json)
        }
        "info" => cmd_libtorch_info(),
        "activate" => {
            let cli: LibtorchActivateArgs = parse_sub("fdl libtorch activate", &args[2..]);
            cmd_libtorch_activate(cli.variant.as_deref())
        }
        "download" => {
            let cli: LibtorchDownloadArgs = parse_sub("fdl libtorch download", &args[2..]);
            cmd_libtorch_download(cli)
        }
        "build" => {
            let cli: LibtorchBuildArgs = parse_sub("fdl libtorch build", &args[2..]);
            cmd_libtorch_build(cli)
        }
        "remove" => {
            let cli: LibtorchRemoveArgs = parse_sub("fdl libtorch remove", &args[2..]);
            cmd_libtorch_remove(cli.variant.as_deref())
        }
        "--help" | "-h" => {
            print_libtorch_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown libtorch command: {other}");
            eprintln!();
            print_libtorch_usage();
            ExitCode::FAILURE
        }
    }
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

// ---------------------------------------------------------------------------
// schema dispatch
// ---------------------------------------------------------------------------

fn dispatch_schema(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("--help");
    match sub {
        "list" => {
            let cli: SchemaListArgs = parse_sub("fdl schema list", &args[2..]);
            cmd_schema_list(cli.json)
        }
        "clear" => {
            let cli: SchemaClearArgs = parse_sub("fdl schema clear", &args[2..]);
            cmd_schema_clear(cli.cmd.as_deref())
        }
        "refresh" => {
            let cli: SchemaRefreshArgs = parse_sub("fdl schema refresh", &args[2..]);
            cmd_schema_refresh(cli.cmd.as_deref())
        }
        "--help" | "-h" => {
            print_schema_usage();
            ExitCode::SUCCESS
        }
        other => {
            cli_error!("unknown schema command: {other}");
            eprintln!();
            print_schema_usage();
            ExitCode::FAILURE
        }
    }
}

fn cmd_schema_list(json: bool) -> ExitCode {
    let Some(root) = project_root_for_schema() else {
        cli_error!("no fdl.yml found in {} or parent directories", env::current_dir().unwrap_or_default().display());
        return ExitCode::FAILURE;
    };
    let caches = schema::discover_caches(&root);

    if json {
        // Keep JSON minimal and stable: array of {name, path, status}.
        print!("[");
        for (i, c) in caches.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            let rel = c
                .cache_path
                .strip_prefix(&root)
                .unwrap_or(&c.cache_path);
            print!(
                "{{\"name\":\"{}\",\"path\":\"{}\",\"status\":\"{}\"}}",
                util::system::escape_json(&c.cmd_name),
                util::system::escape_json(&rel.to_string_lossy()),
                match c.status() {
                    schema::CacheStatus::Fresh => "fresh",
                    schema::CacheStatus::Stale => "stale",
                    schema::CacheStatus::Orphan => "orphan",
                }
            );
        }
        println!("]");
        return ExitCode::SUCCESS;
    }

    if caches.is_empty() {
        println!("No cached schemas under {}.", root.display());
        println!("Run `fdl <cmd> --refresh-schema` after building to populate.");
        return ExitCode::SUCCESS;
    }

    println!("{}:", style::yellow("Cached schemas"));
    for c in &caches {
        let rel = c.cache_path.strip_prefix(&root).unwrap_or(&c.cache_path);
        let status_label = match c.status() {
            schema::CacheStatus::Fresh => style::green("fresh"),
            schema::CacheStatus::Stale => style::yellow("stale"),
            schema::CacheStatus::Orphan => style::red("orphan"),
        };
        println!(
            "    {}  {}  [{status_label}]",
            style::green(&format!("{:<18}", c.cmd_name)),
            style::dim(&rel.display().to_string()),
        );
    }
    ExitCode::SUCCESS
}

fn cmd_schema_clear(filter: Option<&str>) -> ExitCode {
    let Some(root) = project_root_for_schema() else {
        cli_error!("no fdl.yml found in {} or parent directories", env::current_dir().unwrap_or_default().display());
        return ExitCode::FAILURE;
    };
    match schema::clear_caches(&root, filter) {
        Ok(removed) if removed.is_empty() => {
            match filter {
                Some(name) => println!("No cached schema for `{name}`."),
                None => println!("No cached schemas to clear."),
            }
            ExitCode::SUCCESS
        }
        Ok(removed) => {
            for p in &removed {
                let rel = p.strip_prefix(&root).unwrap_or(p);
                println!("Removed {}", rel.display());
            }
            println!();
            println!("Cleared {} cache file(s).", removed.len());
            ExitCode::SUCCESS
        }
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_schema_refresh(filter: Option<&str>) -> ExitCode {
    let Some(root) = project_root_for_schema() else {
        cli_error!("no fdl.yml found in {} or parent directories", env::current_dir().unwrap_or_default().display());
        return ExitCode::FAILURE;
    };
    let results = match schema::refresh_caches(&root, filter) {
        Ok(r) => r,
        Err(e) => {
            cli_error!("{e}");
            return ExitCode::FAILURE;
        }
    };

    if results.is_empty() {
        match filter {
            Some(name) => println!("No cached schema for `{name}`."),
            None => println!("No cached schemas to refresh."),
        }
        return ExitCode::SUCCESS;
    }

    let mut ok = 0usize;
    let mut failed = 0usize;
    for r in &results {
        let rel = r.cache_path.strip_prefix(&root).unwrap_or(&r.cache_path);
        match &r.outcome {
            Ok(()) => {
                ok += 1;
                println!(
                    "{}  {}  [{}]",
                    style::green(&format!("{:<18}", r.cmd_name)),
                    style::dim(&rel.display().to_string()),
                    style::green("refreshed"),
                );
            }
            Err(e) => {
                failed += 1;
                println!(
                    "{}  {}  [{}]",
                    style::green(&format!("{:<18}", r.cmd_name)),
                    style::dim(&rel.display().to_string()),
                    style::red("failed"),
                );
                println!("    {e}");
            }
        }
    }
    println!();
    println!("Refreshed {ok}, failed {failed}.");
    if failed > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Resolve the project root (dir of nearest `fdl.yml`) for schema ops.
/// All three sub-commands need it; factored here so each helper can
/// cli_error! + return on missing config with one call.
fn project_root_for_schema() -> Option<std::path::PathBuf> {
    let cwd = env::current_dir().unwrap_or_default();
    let base = config::find_config(&cwd)?;
    base.parent().map(|p| p.to_path_buf())
}

fn print_schema_usage() {
    println!("fdl schema -- inspect, clear, or refresh cached --fdl-schema outputs");
    println!();
    println!("{}:", style::yellow("Usage"));
    println!("    fdl schema list [--json]");
    println!("    fdl schema clear [<cmd>]");
    println!("    fdl schema refresh [<cmd>]");
    println!();
    println!("{}:", style::yellow("Commands"));
    println!(
        "    {}  Show every cached schema with fresh/stale/orphan status",
        style::green(&format!("{:<10}", "list"))
    );
    println!(
        "    {}  Delete cached schema(s). No arg clears all; `<cmd>` clears one",
        style::green(&format!("{:<10}", "clear"))
    );
    println!(
        "    {}  Re-probe each entry's --fdl-schema and overwrite the cache",
        style::green(&format!("{:<10}", "refresh"))
    );
    println!();
    println!("Cached schemas live at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`.");
    println!("Cargo entries must be built before `refresh` (`cargo build ...`).");
}

// ---------------------------------------------------------------------------
// libtorch subcommands
// ---------------------------------------------------------------------------

fn cmd_libtorch_list(json: bool) -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    let variants = libtorch::detect::list_variants(root);
    let active = libtorch::detect::read_active(root);
    let active_path = active.as_ref().map(|i| i.path.as_str());

    if json {
        print!("[");
        for (i, v) in variants.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            let is_active = active_path == Some(v.as_str());
            print!(
                "{{\"variant\":\"{}\",\"active\":{}}}",
                util::system::escape_json(v),
                is_active
            );
        }
        println!("]");
    } else if variants.is_empty() {
        println!("No libtorch variants installed.");
        println!("Run: fdl libtorch download");
    } else {
        for v in &variants {
            let marker = if active_path == Some(v.as_str()) {
                " (active)"
            } else {
                ""
            };
            println!("  {v}{marker}");
        }
    }

    ExitCode::SUCCESS
}

fn cmd_libtorch_info() -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    match libtorch::detect::read_active(root) {
        Some(info) => {
            println!("Active:   {}", info.path);
            if let Some(v) = &info.torch_version {
                println!("Version:  {v}");
            }
            if let Some(c) = &info.cuda_version {
                println!("CUDA:     {c}");
            }
            if let Some(a) = &info.archs {
                println!("Archs:    {a}");
            }
            if let Some(s) = &info.source {
                println!("Source:   {s}");
            }
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("No active libtorch variant.");
            eprintln!("Run: fdl libtorch download");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_activate(variant: Option<&str>) -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    let variant = match variant {
        Some(v) => v,
        None => {
            eprintln!("usage: fdl libtorch activate <variant>");
            eprintln!();
            eprintln!("Available variants:");
            for v in libtorch::detect::list_variants(root) {
                eprintln!("  {v}");
            }
            return ExitCode::FAILURE;
        }
    };

    if !libtorch::detect::is_valid_variant(root, variant) {
        cli_error!("'{variant}' is not a valid libtorch variant");
        eprintln!("  Expected: libtorch/{variant}/lib/ to exist");
        eprintln!();
        eprintln!("Available variants:");
        for v in libtorch::detect::list_variants(root) {
            eprintln!("  {v}");
        }
        return ExitCode::FAILURE;
    }

    match libtorch::detect::set_active(root, variant) {
        Ok(()) => {
            println!("Active variant set to: {variant}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_download(cli: LibtorchDownloadArgs) -> ExitCode {
    use libtorch::download::{DownloadOpts, Variant};
    use std::path::PathBuf;

    // --cpu and --cuda are mutually exclusive.
    if cli.cpu && cli.cuda.is_some() {
        cli_error!("--cpu and --cuda are mutually exclusive");
        return ExitCode::FAILURE;
    }

    let variant = if cli.cpu {
        Variant::Cpu
    } else {
        match cli.cuda.as_deref() {
            Some("12.6") => Variant::Cuda126,
            Some("12.8") => Variant::Cuda128,
            Some(_) => unreachable!("validated by #[option(choices = ...)]"),
            None => Variant::Auto,
        }
    };

    let opts = DownloadOpts {
        variant,
        custom_path: cli.path.map(PathBuf::from),
        activate: !cli.no_activate,
        dry_run: cli.dry_run,
    };

    match libtorch::download::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_build(cli: LibtorchBuildArgs) -> ExitCode {
    use libtorch::build::{BuildBackend, BuildOpts};

    if cli.jobs == 0 {
        cli_error!("--jobs must be a positive number");
        return ExitCode::FAILURE;
    }

    // --docker and --native are mutually exclusive; absent -> Auto.
    if cli.docker && cli.native {
        cli_error!("--docker and --native are mutually exclusive");
        return ExitCode::FAILURE;
    }

    let backend = if cli.docker {
        BuildBackend::Docker
    } else if cli.native {
        BuildBackend::Native
    } else {
        BuildBackend::Auto
    };

    let opts = BuildOpts {
        archs: cli.archs,
        max_jobs: cli.jobs,
        dry_run: cli.dry_run,
        backend,
    };

    match libtorch::build::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_libtorch_remove(variant: Option<&str>) -> ExitCode {
    let ctx = Context::resolve();
    let root = &ctx.root;
    let variant = match variant {
        Some(v) => v,
        None => {
            eprintln!("usage: fdl libtorch remove <variant>");
            eprintln!();
            eprintln!("Installed variants:");
            for v in libtorch::detect::list_variants(root) {
                eprintln!("  {v}");
            }
            return ExitCode::FAILURE;
        }
    };

    match libtorch::manage::remove_variant(root, variant) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

fn cmd_install(check_only: bool, dev: bool) -> ExitCode {
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    let current_version = env!("CARGO_PKG_VERSION");

    let self_path = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            cli_error!("cannot determine own binary path: {e}");
            return ExitCode::FAILURE;
        }
    };

    let home = match env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        Some(h) => PathBuf::from(h),
        None => {
            cli_error!("cannot determine home directory");
            return ExitCode::FAILURE;
        }
    };

    let bin_dir = home.join(".local/bin");
    let dest = bin_dir.join("fdl");

    // --check: compare versions
    if check_only {
        let latest = fetch_latest_github_tag();
        println!("Installed: {current_version}");
        // Check if current install is a symlink (dev mode)
        if dest.is_symlink() {
            if let Ok(target) = std::fs::read_link(&dest) {
                println!("Mode:      dev (symlink -> {})", target.display());
            }
        }
        match &latest {
            Some(tag) => {
                println!("Latest:    {tag}");
                if tag == current_version {
                    println!("Up to date.");
                } else {
                    println!("Update available. Run: fdl install");
                }
            }
            None => println!("Latest:    (could not check GitHub)"),
        }
        return ExitCode::SUCCESS;
    }

    // Create ~/.local/bin/ if needed
    if let Err(e) = std::fs::create_dir_all(&bin_dir) {
        cli_error!("cannot create {}: {}", bin_dir.display(), e);
        return ExitCode::FAILURE;
    }

    // --dev: symlink to the stable local-build location (~/.cargo/bin/fdl),
    // which is where `cargo install --path flodl-cli` (== `fdl self-build`)
    // drops the freshly-compiled binary. Every subsequent rebuild updates
    // that same path, so the symlink is kept pointing at today's build for
    // free. Falling back to `env::current_exe()` would footgun when the
    // user happens to be running the binary *from* `~/.local/bin/fdl`
    // itself — `canonicalize()` returns that same path and we'd create a
    // symlink to itself (Too many levels of symbolic links).
    if dev {
        #[cfg(unix)]
        {
            let cargo_bin = home.join(".cargo/bin/fdl");
            let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());

            // Prefer the cargo-install target; fall back to the running
            // binary only when cargo-install isn't present.
            let target = if cargo_bin.is_file() {
                cargo_bin.canonicalize().unwrap_or(cargo_bin)
            } else {
                self_canonical.clone()
            };

            // Self-loop guard: refuse to symlink dest to itself.
            //
            // Compare the fully-resolved `target` against the resolved
            // `dest` when it exists, and fall back to the raw `dest`
            // path otherwise. Raw-only would miss the case where
            // `~/.cargo/bin/fdl` already symlinks (transitively) back
            // to `~/.local/bin/fdl`; canonical-only would fail when
            // `dest` does not yet exist.
            let dest_resolved = dest.canonicalize().unwrap_or_else(|_| dest.clone());
            if target == dest_resolved || target == dest {
                eprintln!(
                    "error: --dev cannot symlink `{}` to itself.",
                    dest.display()
                );
                eprintln!();
                eprintln!(
                    "The currently-running `fdl` is installed at the dest path \
                     and no stable cargo build exists at `{}`.",
                    home.join(".cargo/bin/fdl").display()
                );
                eprintln!();
                eprintln!("Build one first:");
                eprintln!("    cargo install --path flodl-cli");
                eprintln!("    # or (from inside a flodl checkout):");
                eprintln!("    fdl self-build");
                eprintln!();
                eprintln!("Then rerun `fdl install --dev`.");
                return ExitCode::FAILURE;
            }

            // Remove existing (file or symlink) before linking.
            if dest.exists() || dest.is_symlink() {
                let _ = std::fs::remove_file(&dest);
            }

            match std::os::unix::fs::symlink(&target, &dest) {
                Ok(()) => {
                    println!("Linked fdl -> {}", target.display());
                    println!("Global fdl now tracks your local build.");
                    println!("Rebuild with: cargo install --path flodl-cli (or `fdl self-build`).");
                }
                Err(e) => {
                    cli_error!("symlink failed: {e}");
                    return ExitCode::FAILURE;
                }
            }

            return print_path_hint(&bin_dir);
        }

        #[cfg(not(unix))]
        {
            eprintln!("--dev mode requires Unix (symlinks). Use fdl install without --dev.");
            return ExitCode::FAILURE;
        }
    }

    // Normal install: download from GitHub if newer, else copy self
    let latest = fetch_latest_github_tag();

    // Check installed version
    let installed_version = if dest.exists() && !dest.is_symlink() {
        let self_canonical = self_path.canonicalize().unwrap_or(self_path.clone());
        let dest_canonical = dest.canonicalize().unwrap_or(dest.clone());
        if self_canonical == dest_canonical {
            Some(current_version.to_string())
        } else {
            Command::new(&dest)
                .arg("version")
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .strip_prefix("flodl-cli ")
                        .map(|v| v.to_string())
                })
        }
    } else {
        None
    };

    // Was it a dev symlink? Switching to copy mode.
    let was_dev = dest.is_symlink();
    if was_dev {
        let _ = std::fs::remove_file(&dest);
    }

    // Decide source: download if newer, else copy self
    let source_path;
    let source_version;
    let mut downloaded_path: Option<PathBuf> = None;

    if let Some(ref tag) = latest {
        if tag != current_version && is_newer(tag, current_version) {
            match download_release_binary(tag, &home) {
                Ok(path) => {
                    source_version = tag.clone();
                    source_path = path.clone();
                    downloaded_path = Some(path);
                }
                Err(e) => {
                    eprintln!("warning: could not download {tag}: {e}");
                    eprintln!("Installing current binary ({current_version}) instead.");
                    source_path = self_path.clone();
                    source_version = current_version.to_string();
                }
            }
        } else {
            source_path = self_path.clone();
            source_version = current_version.to_string();
        }
    } else {
        source_path = self_path.clone();
        source_version = current_version.to_string();
    }

    // Check if update is needed
    if !was_dev {
        if let Some(ref iv) = installed_version {
            if iv == &source_version {
                println!("fdl {} is already installed at {}", iv, dest.display());
                if let Some(ref dl) = downloaded_path {
                    let _ = std::fs::remove_file(dl);
                }
                return ExitCode::SUCCESS;
            }
            println!("Updating fdl {iv} -> {source_version}");
        } else {
            println!("Installing fdl {source_version}");
        }
    } else {
        println!("Switching from dev symlink to installed copy ({source_version})");
    }

    // Copy
    if let Err(e) = std::fs::copy(&source_path, &dest) {
        cli_error!("{e}");
        return ExitCode::FAILURE;
    }

    if let Some(ref dl) = downloaded_path {
        let _ = std::fs::remove_file(dl);
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755));
    }

    println!("Installed fdl {} to {}", source_version, dest.display());
    print_path_hint(&bin_dir)
}

fn print_path_hint(bin_dir: &std::path::Path) -> ExitCode {
    let path_var = env::var("PATH").unwrap_or_default();
    let bin_dir_str = bin_dir.to_string_lossy();
    let in_path = path_var.split(':').any(|p| p == bin_dir_str.as_ref());

    if !in_path {
        println!();
        println!("~/.local/bin is not in your PATH. Add it:");
        println!();
        let shell = env::var("SHELL").unwrap_or_default();
        if shell.contains("zsh") {
            println!("  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc && source ~/.zshrc");
        } else {
            println!(
                "  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && source ~/.bashrc"
            );
        }
    }

    ExitCode::SUCCESS
}

/// Fetch the latest release tag from GitHub.
fn fetch_latest_github_tag() -> Option<String> {
    use std::process::{Command, Stdio};
    let output = Command::new("curl")
        .args(["-sI", "https://github.com/flodl-labs/flodl/releases/latest"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.to_lowercase().starts_with("location:") {
            let tag = line.rsplit('/').next()?.trim();
            if !tag.is_empty() {
                return Some(tag.to_string());
            }
        }
    }
    None
}

/// Simple version comparison: "0.3.1" > "0.3.0".
fn is_newer(a: &str, b: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> { v.split('.').filter_map(|s| s.parse().ok()).collect() };
    let va = parse(a);
    let vb = parse(b);
    va > vb
}

/// Download the fdl binary for this platform from a GitHub release.
fn download_release_binary(tag: &str, home: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let os = if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        return Err("unsupported OS".into());
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        if cfg!(target_os = "macos") {
            "arm64"
        } else {
            "aarch64"
        }
    } else {
        return Err("unsupported architecture".into());
    };

    let ext = if cfg!(target_os = "windows") { ".exe" } else { "" };
    let artifact = format!("flodl-cli-{os}-{arch}{ext}");
    let url = format!("https://github.com/flodl-labs/flodl/releases/download/{tag}/{artifact}");

    let tmp = home.join(".flodl").join("tmp");
    std::fs::create_dir_all(&tmp).map_err(|e| format!("cannot create temp dir: {e}"))?;
    let dest = tmp.join(format!("fdl-{tag}{ext}"));

    println!("Downloading fdl {tag} from GitHub...");
    util::http::download_file(&url, &dest)?;

    Ok(dest)
}

// ---------------------------------------------------------------------------
// fdl.yaml dispatch
// ---------------------------------------------------------------------------

/// Locate and parse the project's `fdl.yml` (with optional env overlay).
///
/// Returns `None` only when there is no `fdl.yml` anywhere up the tree --
/// that's the normal "running outside a project" case. Parse errors are
/// loud: a malformed YAML or a `cluster:` block that fails strict
/// deserialization prints to stderr and aborts the process. Silently
/// falling through to "unknown command" hid real config typos
/// (`feedback_loud_errors_over_silent`).
fn load_project_config(
    cwd: &std::path::Path,
    env: Option<&str>,
) -> Option<(config::ProjectConfig, std::path::PathBuf)> {
    let config_path = config::find_config(cwd)?;
    let root = config_path.parent()?.to_path_buf();
    match config::load_project_with_env(&config_path, env) {
        Ok(project) => Some((project, root)),
        Err(e) => {
            cli_error!("{e}");
            std::process::exit(2);
        }
    }
}


/// Dispatch an unknown top-level token through the unified `commands:`
/// graph declared in fdl.yml. Handles arbitrary nesting: each step either
/// recurses into a child fdl.yml (Path), executes a self-contained shell
/// command (Run), or invokes the enclosing entry with merged preset
/// fields (Preset).
///
/// The graph walk itself lives in [`dispatch::walk_commands`] and is
/// pure — this wrapper performs the actual IO (process spawning, stdout
/// writes, exit code mapping) based on the returned [`WalkOutcome`].
fn dispatch_config(
    cmd: &str,
    args: &[String],
    env: Option<&str>,
    no_append: bool,
) -> ExitCode {
    let cwd = env::current_dir().unwrap_or_default();
    let (project, project_root) = match load_project_config(&cwd, env) {
        Some(pair) => pair,
        None => {
            eprintln!("unknown command: {cmd}");
            eprintln!();
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let tail: &[String] = args.get(2..).unwrap_or(&[]);
    let outcome = walk_commands(cmd, tail, &project.commands, &project_root, env);

    // Cluster dispatch is decided uniformly across RunScript / ExecCommand
    // outcomes: both flavors of `cluster:`-marked command fan out across
    // `cluster.hosts` via ssh, forwarding the original tail unchanged.
    // Recursion guard lives in `cluster::should_dispatch` (returns false
    // when `FLODL_CLUSTER_JSON` is already set in env -- we're a remote node).
    let cluster_chain: Option<&[Option<bool>]> = match &outcome {
        WalkOutcome::RunScript { cluster_chain, .. } => Some(cluster_chain.as_slice()),
        WalkOutcome::ExecCommand { cluster_chain, .. } => Some(cluster_chain.as_slice()),
        _ => None,
    };
    if let Some(chain) = cluster_chain
        && cluster::should_dispatch(&project, chain)
    {
        return cluster::dispatch(&project, env, cmd, tail);
    }

    match outcome {
        WalkOutcome::RunScript {
            command,
            append,
            user_args,
            docker,
            cwd,
            cluster_chain: _,
        } => {
            let effective_append = if no_append { None } else { append.as_deref() };
            run::exec_script(
                &command,
                effective_append,
                &user_args,
                docker.as_deref(),
                &cwd,
            )
        }
        WalkOutcome::ExecCommand {
            config: cmd_config,
            preset,
            tail,
            cmd_dir,
            cluster_chain: _,
        } => {
            run::exec_command(&cmd_config, preset.as_deref(), &tail, &cmd_dir, &project_root)
        }
        WalkOutcome::RefreshSchema {
            config,
            cmd_dir,
            cmd_name,
        } => cmd_refresh_schema(&config, &cmd_dir, &cmd_name),
        WalkOutcome::PrintCommandHelp { config, name } => {
            run::print_command_help(&config, &name);
            ExitCode::SUCCESS
        }
        WalkOutcome::PrintPresetHelp {
            config,
            parent_label,
            preset_name,
        } => {
            run::print_preset_help(&config, &parent_label, &preset_name);
            ExitCode::SUCCESS
        }
        WalkOutcome::PrintRunHelp {
            name,
            description,
            run,
            append,
            docker,
        } => {
            run::print_run_help(
                &name,
                description.as_deref(),
                &run,
                append.as_deref(),
                docker.as_deref(),
            );
            ExitCode::SUCCESS
        }
        WalkOutcome::UnknownCommand { name } => {
            eprintln!("unknown command: {name}");
            eprintln!();
            run::print_project_help(&project, &project_root, env);
            ExitCode::FAILURE
        }
        WalkOutcome::PresetAtTopLevel { name } => {
            eprintln!(
                "error: preset command `{name}` has no enclosing \
                 fdl.yml (top-level commands must be `run:` or `path:`)"
            );
            ExitCode::FAILURE
        }
        WalkOutcome::Error(msg) => {
            cli_error!("{msg}");
            ExitCode::FAILURE
        }
    }
}

/// `fdl config show [<env>]` — print the resolved merged config.
///
/// `tail` is `args[1..]`: `tail[0]` is always "config", `tail[1]` is the
/// sub-command ("show"), `tail[2..]` carry options (an optional explicit
/// `<env>` that overrides the first-arg env detection).
fn cmd_config_show(tail: &[String], active_env: Option<&str>) -> ExitCode {
    let sub = tail.get(1).map(String::as_str).unwrap_or("--help");
    match sub {
        "show" => {}
        "--help" | "-h" => {
            print_config_usage();
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("unknown config sub-command: {other}");
            eprintln!();
            print_config_usage();
            return ExitCode::FAILURE;
        }
    }

    // Optional explicit env override: `fdl config show prod`.
    let explicit_env = tail.get(2).map(String::as_str);
    let target_env = explicit_env.or(active_env);

    let cwd = env::current_dir().unwrap_or_default();
    let base = match config::find_config(&cwd) {
        Some(p) => p,
        None => {
            cli_error!("no fdl.yml found in {} or parent directories", cwd.display());
            return ExitCode::FAILURE;
        }
    };

    // Resolve every contributing layer (including `inherit-from:`
    // ancestors) so we can tag each leaf with its source file, not just
    // "base/overlay". Layer order matches `load_merged_value`: deepest
    // ancestor first, env overlay chain last.
    let layers = match config::resolve_config_layers(&base, target_env) {
        Ok(ls) => ls,
        Err(e) => {
            cli_error!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let labels: Vec<String> = layers
        .iter()
        .map(|(p, _)| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string()
        })
        .collect();
    let values: Vec<serde_yaml::Value> =
        layers.iter().map(|(_, v)| v.clone()).collect();

    let annotated = overlay::merge_layers_annotated(&values);
    print!("{}", overlay::render_annotated_yaml(&annotated, &labels));
    ExitCode::SUCCESS
}

fn print_config_usage() {
    println!("fdl config -- inspect resolved project configuration");
    println!();
    println!("USAGE:");
    println!("    fdl config show [<env>]");
    println!();
    println!("Without an env argument, prints the base fdl.yml. With an env argument");
    println!("(e.g. `fdl config show ci`), prints the base deep-merged with");
    println!("fdl.<env>.yml. When invoked through the first-arg form");
    println!("(`fdl ci config show`), the env is already active and no extra");
    println!("argument is needed.");
}

/// `fdl <cmd> --refresh-schema`: run `<entry> --fdl-schema`, validate, cache.
///
/// Required for cargo-based entries, which are never auto-probed (compile
/// latency would ruin `--help`). Users build once, then run this explicitly.
fn cmd_refresh_schema(
    cmd_config: &config::CommandConfig,
    cmd_dir: &std::path::Path,
    cmd_name: &str,
) -> ExitCode {
    let entry = match &cmd_config.entry {
        Some(e) => e.as_str(),
        None => {
            eprintln!(
                "error: no entry point defined in {}/fdl.yml",
                cmd_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Probing `{entry} --fdl-schema`...");
    let schema = match schema_cache::probe(entry, cmd_dir, cmd_config.docker.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            cli_error!("{e}");
            if schema_cache::is_cargo_entry(entry) {
                eprintln!();
                eprintln!("Hint: cargo-based entries must be built first.");
                eprintln!("Build with the right features, then rerun this command.");
            }
            return ExitCode::FAILURE;
        }
    };

    let cache = schema_cache::cache_path(cmd_dir, cmd_name);
    if let Err(e) = schema_cache::write_cache(&cache, &schema) {
        cli_error!("{e}");
        return ExitCode::FAILURE;
    }
    eprintln!("Cached schema for `{cmd_name}` at {}", cache.display());
    eprintln!(
        "  {} options, {} positional args",
        schema.options.len(),
        schema.args.len()
    );
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

fn print_usage() {
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
    println!("    --ansi             Force ANSI color output");
    println!("    --no-ansi          Disable ANSI color output");
    println!("    -v                 Verbose output (DDP sync, data loading detail)");
    println!("    -vv                Debug output (per-batch timing, loop internals)");
    println!("    -vvv               Trace output (maximum detail)");
    println!("    -q, --quiet        Suppress all non-error output");
    println!("    --no-append        Drop a run command's `append:` suffix (cargo / runner defaults)");
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

    for arg in args {
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

/// Strip `--ansi` / `--no-ansi` from `args`, returning a
/// [`style::ColorChoice`] override when either was present. Errors if
/// both are given (ambiguous). Scan-anywhere, consistent with `-v`
/// and `--env` — global flags aren't position-locked.
fn extract_ansi_flags(
    args: &[String],
) -> Result<(Vec<String>, Option<style::ColorChoice>), String> {
    let mut ansi = false;
    let mut no_ansi = false;
    let mut filtered = Vec::with_capacity(args.len());

    for arg in args {
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

fn print_libtorch_usage() {
    println!("fdl libtorch -- manage libtorch installations");
    println!();
    println!("USAGE:");
    println!("    fdl libtorch <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    download           Download pre-built libtorch");
    println!("        --cpu          Force CPU variant");
    println!("        --cuda <ver>   Specific CUDA version (12.6, 12.8)");
    println!("    build              Build libtorch from source");
    println!("        --docker       Force Docker build (isolated, reproducible)");
    println!("        --native       Force native build (faster, requires host toolchain)");
    println!("        --archs <list> Override CUDA architectures");
    println!("        --jobs <n>     Parallel compilation jobs (default: 6)");
    println!("    list               Show installed variants");
    println!("        --json         JSON output");
    println!("    activate <name>    Set active variant");
    println!("    remove <name>      Remove a variant");
    println!("    info               Show active variant details");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn args(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// Zero-dep tempdir helper — matches the pattern used in overlay.rs /
    /// dispatch.rs (no `tempfile` crate dependency in flodl-cli).
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("fdl-env-test-{pid}-{n}"));
            std::fs::create_dir_all(&dir).expect("tempdir creation");
            TempDir(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn touch(path: &Path, contents: &str) {
        std::fs::write(path, contents).expect("write fixture");
    }

    #[test]
    fn extract_env_flag_absent_returns_none() {
        let (out, env) = extract_env_flag(&args(&["fdl", "test"])).unwrap();
        assert_eq!(out, args(&["fdl", "test"]));
        assert!(env.is_none());
    }

    #[test]
    fn extract_env_flag_long_separated_form() {
        let (out, env) = extract_env_flag(&args(&["fdl", "--env", "ci", "test"])).unwrap();
        assert_eq!(out, args(&["fdl", "test"]));
        assert_eq!(env.as_deref(), Some("ci"));
    }

    #[test]
    fn extract_env_flag_equals_form() {
        let (out, env) = extract_env_flag(&args(&["fdl", "--env=ci", "test"])).unwrap();
        assert_eq!(out, args(&["fdl", "test"]));
        assert_eq!(env.as_deref(), Some("ci"));
    }

    #[test]
    fn extract_env_flag_scans_anywhere() {
        // Matches `-v`/`-q` global-flag convention: strippable from any position.
        let (out, env) = extract_env_flag(&args(&["fdl", "test", "--env", "prod"])).unwrap();
        assert_eq!(out, args(&["fdl", "test"]));
        assert_eq!(env.as_deref(), Some("prod"));
    }

    #[test]
    fn extract_env_flag_missing_value_errors() {
        let err = extract_env_flag(&args(&["fdl", "--env"])).unwrap_err();
        assert!(err.contains("--env requires a value"), "got: {err}");
    }

    #[test]
    fn extract_env_flag_empty_equals_errors() {
        let err = extract_env_flag(&args(&["fdl", "--env="])).unwrap_err();
        assert!(err.contains("requires a value"), "got: {err}");
    }

    #[test]
    fn extract_env_flag_value_looks_like_flag_errors() {
        // `fdl --env --help` almost certainly means the user forgot the value;
        // loud error beats silently treating `--help` as the env name.
        let err = extract_env_flag(&args(&["fdl", "--env", "--help"])).unwrap_err();
        assert!(err.contains("--env requires a value"), "got: {err}");
    }

    #[test]
    fn extract_env_flag_duplicate_errors() {
        let err = extract_env_flag(&args(&["fdl", "--env", "ci", "--env", "prod"])).unwrap_err();
        assert!(err.contains("more than once"), "got: {err}");
    }

    #[test]
    fn extract_env_flag_duplicate_mixed_forms_errors() {
        let err = extract_env_flag(&args(&["fdl", "--env=ci", "--env", "prod"])).unwrap_err();
        assert!(err.contains("more than once"), "got: {err}");
    }

    // --- resolve_env: precedence + error paths ----------------------------
    //
    // These exercise the full flag / env-var / first-arg composition with
    // real fixture files on disk. Injecting `cwd` + `fdl_env` keeps them
    // hermetic (no mutation of the process cwd or environment).

    #[test]
    fn resolve_env_flag_wins_over_env_var_and_first_arg() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");
        touch(&tmp.path().join("fdl.prod.yml"), "");
        touch(&tmp.path().join("fdl.stage.yml"), "");

        // --env prod is explicit → beats FDL_ENV=stage and the `ci` first-arg.
        let (env, rest) = resolve_env(
            &args(&["fdl", "ci", "--env", "prod", "test"]),
            tmp.path(),
            Some("stage"),
        )
        .unwrap();
        assert_eq!(env.as_deref(), Some("prod"));
        // --env/--env=value tokens stripped; `ci` stays as the first-arg candidate,
        // which resolve_env_first_arg is skipped for entirely.
        assert_eq!(rest, args(&["fdl", "ci", "test"]));
    }

    #[test]
    fn resolve_env_env_var_wins_over_first_arg() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");
        touch(&tmp.path().join("fdl.stage.yml"), "");

        let (env, rest) =
            resolve_env(&args(&["fdl", "ci", "test"]), tmp.path(), Some("stage")).unwrap();
        assert_eq!(env.as_deref(), Some("stage"));
        // First-arg `ci` left untouched when FDL_ENV takes over.
        assert_eq!(rest, args(&["fdl", "ci", "test"]));
    }

    #[test]
    fn resolve_env_empty_env_var_falls_through_to_first_arg() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");

        let (env, rest) = resolve_env(&args(&["fdl", "ci", "test"]), tmp.path(), Some("")).unwrap();
        assert_eq!(env.as_deref(), Some("ci"));
        assert_eq!(rest, args(&["fdl", "test"]));
    }

    #[test]
    fn resolve_env_first_arg_still_works_when_no_explicit_selector() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");

        let (env, rest) = resolve_env(&args(&["fdl", "ci", "test"]), tmp.path(), None).unwrap();
        assert_eq!(env.as_deref(), Some("ci"));
        assert_eq!(rest, args(&["fdl", "test"]));
    }

    #[test]
    fn resolve_env_flag_errors_on_missing_overlay() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");

        let err = resolve_env(&args(&["fdl", "--env", "nope", "test"]), tmp.path(), None)
            .unwrap_err();
        assert!(err.contains("--env"), "got: {err}");
        assert!(err.contains("nope"), "got: {err}");
        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn resolve_env_env_var_errors_on_missing_overlay() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");

        let err = resolve_env(&args(&["fdl", "test"]), tmp.path(), Some("nope")).unwrap_err();
        assert!(err.contains("FDL_ENV"), "got: {err}");
        assert!(err.contains("nope"), "got: {err}");
        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn resolve_env_equals_form_consumes_single_token() {
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");
        touch(&tmp.path().join("fdl.ci.yml"), "");

        let (env, rest) =
            resolve_env(&args(&["fdl", "test", "--env=ci"]), tmp.path(), None).unwrap();
        assert_eq!(env.as_deref(), Some("ci"));
        assert_eq!(rest, args(&["fdl", "test"]));
    }

    #[test]
    fn resolve_env_first_arg_unknown_falls_through() {
        // `deploy` isn't an env overlay — leave it as the first positional.
        let tmp = TempDir::new();
        touch(&tmp.path().join("fdl.yml"), "");

        let (env, rest) =
            resolve_env(&args(&["fdl", "deploy", "--now"]), tmp.path(), None).unwrap();
        assert!(env.is_none());
        assert_eq!(rest, args(&["fdl", "deploy", "--now"]));
    }

    // ── --ansi / --no-ansi extraction ────────────────────────────────────

    #[test]
    fn extract_ansi_flags_absent_returns_none() {
        let (rest, choice) = extract_ansi_flags(&args(&["fdl", "setup"])).unwrap();
        assert_eq!(rest, args(&["fdl", "setup"]));
        assert!(choice.is_none());
    }

    #[test]
    fn extract_ansi_flags_ansi_forces_always() {
        let (rest, choice) = extract_ansi_flags(&args(&["fdl", "--ansi", "setup"])).unwrap();
        assert_eq!(rest, args(&["fdl", "setup"]));
        assert_eq!(choice, Some(style::ColorChoice::Always));
    }

    #[test]
    fn extract_ansi_flags_no_ansi_forces_never() {
        let (rest, choice) = extract_ansi_flags(&args(&["fdl", "--no-ansi", "setup"])).unwrap();
        assert_eq!(rest, args(&["fdl", "setup"]));
        assert_eq!(choice, Some(style::ColorChoice::Never));
    }

    #[test]
    fn extract_ansi_flags_scans_anywhere() {
        // Position-independent, consistent with -v / --env.
        let (rest, choice) =
            extract_ansi_flags(&args(&["fdl", "setup", "--no-ansi"])).unwrap();
        assert_eq!(rest, args(&["fdl", "setup"]));
        assert_eq!(choice, Some(style::ColorChoice::Never));
    }

    #[test]
    fn extract_ansi_flags_both_set_errors() {
        let err = extract_ansi_flags(&args(&["fdl", "--ansi", "--no-ansi"])).unwrap_err();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    #[test]
    fn extract_no_append_absent_returns_false() {
        let (out, found) = extract_no_append(&args(&["fdl", "test"]));
        assert_eq!(out, args(&["fdl", "test"]));
        assert!(!found);
    }

    #[test]
    fn extract_no_append_strips_flag_from_anywhere_before_dashdash() {
        let (out, found) =
            extract_no_append(&args(&["fdl", "test", "--no-append", "--", "-p", "x"]));
        assert_eq!(out, args(&["fdl", "test", "--", "-p", "x"]));
        assert!(found);
    }

    #[test]
    fn extract_no_append_preserves_flag_after_dashdash() {
        // Past the first `--` the token is bound for the inner script,
        // so don't swallow it — escape hatch for run-scripts whose own
        // proxy flags collide.
        let (out, found) =
            extract_no_append(&args(&["fdl", "test", "--", "--no-append"]));
        assert_eq!(out, args(&["fdl", "test", "--", "--no-append"]));
        assert!(!found);
    }
}
