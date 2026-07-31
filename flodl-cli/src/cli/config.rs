//! `fdl config` subcommand surface + project config loader.

use std::env;
use std::process::ExitCode;

use flodl_cli::{
    cluster, config, dispatch, gpus, overlay, prebuild, run, schema_cache,
};
use flodl_cli::cli_error;
use dispatch::{walk_commands, WalkOutcome};

use crate::print_usage;


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
pub(crate) fn load_project_config(
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
pub(crate) fn dispatch_config(
    cmd: &str,
    args: &[String],
    env: Option<&str>,
    no_append: bool,
    no_prebuild: bool,
    gpus_spec: Option<&gpus::GpusSpec>,
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

    // Test-cluster envelope export: when an env overlay is active AND
    // its merged config has a `cluster:` block, surface the canonical
    // JSON via `FLODL_TESTING_CLUSTER_JSON` so test binaries (and any
    // other consumer that wants the topology WITHOUT entering launcher
    // mode) can read it. Distinct from `FLODL_INTERNAL_FULL_CLUSTER_JSON` which
    // is launcher-mode-only and gated on `cluster: true` commands —
    // this var is purely informational and never triggers fan-out.
    //
    // Gated on `env.is_some()` so `fdl <cmd>` (no overlay) leaves the
    // file inert, matching the convention "presence of fdl.<env>.yml
    // does nothing until invoked via --env <name>".
    //
    // Uses `prepare_test_cluster_env` (NOT the production
    // `prepare_cluster_env`) so the probe stays local (no SSH); tests
    // run in-process on whichever host the cluster-test invocation
    // lives on, and the local `nvidia-smi -L` is authoritative for
    // any worker declaring `local_devices: all`.
    if env.is_some()
        && let Some(cluster) = project.cluster.as_ref()
    {
        match cluster::prepare_test_cluster_env(cluster) {
            Ok(hex) => {
                // SAFETY: main() has not spawned threads at this point
                // in the dispatch flow; matches the invariant for
                // `prepare_cluster_env` and `apply_cuda_visible_devices`.
                unsafe {
                    env::set_var("FLODL_TESTING_CLUSTER_JSON", hex);
                }
            }
            Err(e) => {
                // Don't abort the command — the test binary may not
                // need the topology (e.g. running a non-cluster test
                // command under a cluster-aware overlay). Surface as a
                // warning so misconfigurations stay visible.
                eprintln!(
                    "warning: cluster-test envelope export failed: {e}"
                );
            }
        }
    }

    let tail: &[String] = args.get(2..).unwrap_or(&[]);
    let outcome = walk_commands(cmd, tail, &project.commands, &project_root, env);

    // The actual cwd the command would run from (per the walk into
    // sub-fdl.yml chains). Used by the pre-flight build below so
    // cargo runs from the right crate root for sub-command path
    // entries (e.g. `ddp-bench` is workspace-excluded — `cargo
    // build --bin ddp-bench` from project root fails; from
    // `ddp-bench/` it succeeds).
    let cmd_cwd: std::path::PathBuf = match &outcome {
        WalkOutcome::RunScript { cwd, .. } => cwd.clone(),
        WalkOutcome::ExecCommand { cmd_dir, .. } => cmd_dir.clone(),
        _ => project_root.clone(),
    };

    // Cluster dispatch sources: YAML `cluster:` block, or synthesized from
    // `--gpus` on a cluster-aware command (loopback, one host, N ranks).
    // Recursion guard skips both paths when `FLODL_INTERNAL_CLUSTER_JSON` is already
    // set in env -- we're a spawned child of a parent launcher.
    let cluster_chain: Option<&[Option<bool>]> = match &outcome {
        WalkOutcome::RunScript { cluster_chain, .. } => Some(cluster_chain.as_slice()),
        WalkOutcome::ExecCommand { cluster_chain, .. } => Some(cluster_chain.as_slice()),
        _ => None,
    };
    let wants_cluster = cluster_chain
        .map(config::resolve_cluster_dispatch)
        .unwrap_or(false)
        && !cluster::is_recursive_invocation();

    let cluster_to_dispatch: Option<config::ClusterConfig> = if wants_cluster {
        match (project.cluster.as_ref(), gpus_spec) {
            (Some(_), Some(_)) => {
                cli_error!(
                    "--gpus cannot be combined with a `cluster:` block in \
                     fdl.yml; remove one or the other"
                );
                return ExitCode::FAILURE;
            }
            (Some(c), None) => Some(c.clone()),
            (None, Some(spec)) => {
                let devs = match spec.resolve() {
                    Ok(d) => d,
                    Err(e) => {
                        cli_error!("{e}");
                        return ExitCode::FAILURE;
                    }
                };
                if devs.len() < 2 {
                    // Degenerate single-device: no synthesis, no spawn.
                    // Pin CUDA_VISIBLE_DEVICES and fall through to the
                    // regular RunScript / ExecCommand path.
                    // SAFETY: main has not spawned threads yet.
                    unsafe { gpus::apply_cuda_visible_devices(&devs) };
                    None
                } else {
                    match gpus::synthesize_local_cluster(&devs) {
                        Ok(c) => Some(c),
                        Err(e) => {
                            cli_error!("{e}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            }
            (None, None) => {
                // No YAML cluster, no --gpus, but the command opted into
                // cluster mode. Print a one-line hint if N>=2 GPUs visible.
                if let Ok(n) = gpus::local_gpu_count() {
                    if n >= 2 {
                        eprintln!(
                            "flodl: {n} GPUs visible but cluster mode is off; \
                             running single-device on GPU 0. Use --gpus all \
                             for multi-GPU."
                        );
                    }
                }
                None
            }
        }
    } else {
        // Non-cluster path (test/clippy/etc., or recursive child).
        // `--gpus`, if present, restricts CUDA_VISIBLE_DEVICES for the single
        // child process. No cluster dispatch.
        if let Some(spec) = gpus_spec {
            let devs = match spec.resolve() {
                Ok(d) => d,
                Err(e) => {
                    cli_error!("{e}");
                    return ExitCode::FAILURE;
                }
            };
            // SAFETY: main has not spawned threads yet.
            unsafe { gpus::apply_cuda_visible_devices(&devs) };
        }
        None
    };

    if let Some(cluster) = cluster_to_dispatch {
        // Loud early validation of the network-timeout scale: the
        // library reader deep in flodl warns-and-defaults on a bad
        // value (it cannot abort mid-run), so the fan-out path is
        // where an explicit-but-invalid value must error, before any
        // host is touched.
        if let Err(e) = cluster::validate_net_timeout_scale() {
            cli_error!("{e}");
            return ExitCode::FAILURE;
        }
        let controller = cluster::resolve_local_hostname();
        // Pre-flight host readiness (see flodl-cli::prebuild). One ssh
        // per remote host BEFORE any build, in both modes: always
        // verifies the shared `data_path` is mounted + readable; when
        // prebuilding, also runs the controller-vs-remote ABI gate. A
        // missing declared mount or an ABI mismatch aborts loudly here
        // rather than surfacing as a cryptic mid-run error on a remote
        // rank. Runs even under `--no-prebuild` (mount-readiness is
        // orthogonal to binary freshness).
        if let Err(e) = prebuild::preflight_hosts(&cluster, &controller, !no_prebuild) {
            cli_error!("{e}");
            return ExitCode::FAILURE;
        }
        // Pre-flight build (see flodl-cli::prebuild). Runs `cargo
        // build` locally for each remote host with that host's
        // libtorch + a per-host CARGO_TARGET_DIR, delivering the
        // binary via the shared project-root mount. Skipped when
        // `--no-prebuild` is set. The controller's own build is
        // handled by the normal dispatch path below (cargo run in
        // Docker against the local `.active`).
        if !no_prebuild {
            if let Err(e) = prebuild::prebuild_remotes(
                &project_root, &cmd_cwd, &cluster, cmd, &controller,
            ) {
                cli_error!("{e}");
                return ExitCode::FAILURE;
            }
        }
        // Cluster mode: set env vars on this process so the user binary
        // (spawned by the normal dispatch below) detects launcher role
        // and fans out via flodl::distributed::launcher. Fall through —
        // no early return. The launcher in the spawned subprocess owns
        // fan-out, log fan-in, ClusterController, exit propagation.
        match cluster::prepare_cluster_env(&cluster, env, cmd) {
            Ok(warnings) => {
                // Emit at the cluster-dispatch site (this branch only
                // runs when fdl really is fanning out via cluster mode).
                // Unit tests of prepare_cluster_env never reach here, so
                // their fixture hostnames stay silent without needing a
                // cfg!(test) gate inside the resolver.
                for w in warnings {
                    eprintln!("fdl: warning: {w}");
                }
            }
            Err(e) => {
                cli_error!("{e}");
                return ExitCode::FAILURE;
            }
        }
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
            // Likely the retired positional-env form (`fdl cluster probe`):
            // if a sibling `fdl.<name>.yml` overlay exists, steer to `@`.
            if let Some(base) = config::find_config(&cwd)
                && overlay::find_env_file(&base, &name).is_some()
            {
                eprintln!();
                eprintln!(
                    "`{name}` is an env overlay (fdl.{name}.yml), not a command. \
                     Select it with the `@` sigil: `fdl @{name} <command>`."
                );
            }
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
/// `<env>` that overrides the active `@env` / `--env` selector).
pub(crate) fn cmd_config_show(tail: &[String], active_env: Option<&str>) -> ExitCode {
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
    let values: Vec<serde_yaml_ng::Value> =
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
    println!("fdl.<env>.yml. When invoked through the `@` sigil form");
    println!("(`fdl @ci config show`), the env is already active and no extra");
    println!("argument is needed.");
}

/// `fdl <cmd> --refresh-schema`: run `<entry> --fdl-schema`, validate, cache.
///
/// Required for cargo-based entries, which are never auto-probed (compile
/// latency would ruin `--help`). Users build once, then run this explicitly.
pub(crate) fn cmd_refresh_schema(
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
    if schema.commands.is_empty() {
        eprintln!(
            "  {} options, {} positional args",
            schema.options.len(),
            schema.args.len()
        );
    } else {
        eprintln!("  {} subcommands", schema.commands.len());
    }
    ExitCode::SUCCESS
}
