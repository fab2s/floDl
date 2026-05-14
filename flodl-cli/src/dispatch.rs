//! Pure command-graph dispatch.
//!
//! `walk_commands` is the outer walker: it chases an arbitrarily nested
//! `commands:` graph starting from a top-level name + tail, and returns a
//! `WalkOutcome` describing what the caller should do (run a script,
//! spawn an entry, print help, error out, ...). The walker performs no
//! IO of its own: no process spawning, no stdout writes, no cwd reads.
//!
//! `classify_path_step` is the inner classifier used by `walk_commands`
//! for the `Path` arm: loads the child fdl.yml and inspects the tail to
//! decide whether to descend, render help, refresh the schema cache, or
//! forward to the entry.
//!
//! Keeping all impure actions (printing, spawning) in the caller makes
//! both functions straight-line and unit-testable against tempdir
//! fixtures.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::config::{self, CommandConfig, CommandKind, CommandSpec};

/// What a single `Path`-kind step resolved to. Every variant holds the
/// loaded `child` config when applicable, so the caller doesn't re-load.
pub enum PathOutcome {
    /// Failed to load the child `fdl.yml`. The string is the
    /// underlying error message.
    LoadFailed(String),
    /// Next tail token is a known sub-command of the child — descend.
    Descend {
        child: Box<CommandConfig>,
        new_dir: PathBuf,
        new_name: String,
    },
    /// Tail carries `--help` / `-h` at this level.
    ShowHelp { child: Box<CommandConfig> },
    /// Tail carries `--refresh-schema`.
    RefreshSchema {
        child: Box<CommandConfig>,
        child_dir: PathBuf,
    },
    /// Forward the tail to the child's entry.
    Exec {
        child: Box<CommandConfig>,
        child_dir: PathBuf,
    },
}

/// Classify a `Path`-kind step. Pure: loads the child config, inspects
/// the tail, and returns the matching [`PathOutcome`]. The caller owns
/// every side effect (printing, spawning).
pub fn classify_path_step(
    spec: &CommandSpec,
    name: &str,
    current_dir: &Path,
    tail: &[String],
    env: Option<&str>,
) -> PathOutcome {
    let child_dir = spec.resolve_path(name, current_dir);
    let child_cfg = match config::load_command_with_env(&child_dir, env) {
        Ok(c) => c,
        Err(e) => return PathOutcome::LoadFailed(e),
    };

    // Descent check runs first: `--help` / `--refresh-schema` apply to
    // the level the user is asking about, not to the parent. If the
    // next token names a nested entry, we descend before reading flags.
    if let Some(next) = tail.first() {
        if child_cfg.commands.contains_key(next) {
            return PathOutcome::Descend {
                child: Box::new(child_cfg),
                new_dir: child_dir,
                new_name: next.clone(),
            };
        }
    }

    if tail.iter().any(|a| a == "--help" || a == "-h") {
        return PathOutcome::ShowHelp {
            child: Box::new(child_cfg),
        };
    }

    if tail.iter().any(|a| a == "--refresh-schema") {
        return PathOutcome::RefreshSchema {
            child: Box::new(child_cfg),
            child_dir,
        };
    }

    // Bare project invocation with no entry but available sub-commands:
    // mirror the top-level `fdl` UX and print help instead of erroring on
    // a missing entry point. Only kicks in when tail is empty — any extra
    // tokens still flow through to the existing exec/error path.
    if tail.is_empty() && child_cfg.entry.is_none() && !child_cfg.commands.is_empty() {
        return PathOutcome::ShowHelp {
            child: Box::new(child_cfg),
        };
    }

    PathOutcome::Exec {
        child: Box::new(child_cfg),
        child_dir,
    }
}

// ── Outer walker ────────────────────────────────────────────────────────

/// What the outer walker resolved a user invocation to. The caller owns
/// every impure action (spawning, printing, exit code); the walker just
/// returns the terminal state.
pub enum WalkOutcome {
    /// Top-level or nested `Run` — caller runs the inline script,
    /// composing `command` + `user_args` (POSIX-quoted) + `append`.
    /// `user_args` carries everything the caller typed after `--` on
    /// the CLI (or the empty slice when `--` was absent).
    RunScript {
        command: String,
        append: Option<String>,
        user_args: Vec<String>,
        docker: Option<String>,
        cwd: PathBuf,
        /// `CommandSpec.cluster` values walked from root → leaf. Caller
        /// pairs with `ProjectConfig.cluster` and
        /// [`config::cluster_dispatch_enabled`] to decide whether to
        /// fan out across hosts instead of running locally.
        cluster_chain: Vec<Option<bool>>,
    },
    /// Path-or-Preset terminal → caller invokes the child's entry. For
    /// a Preset, `preset` is the preset name inside the enclosing
    /// `commands:` block; for a Path-Exec it is `None`.
    ExecCommand {
        config: Box<CommandConfig>,
        preset: Option<String>,
        tail: Vec<String>,
        cmd_dir: PathBuf,
        /// See [`WalkOutcome::RunScript::cluster_chain`].
        cluster_chain: Vec<Option<bool>>,
    },
    /// Path terminal with `--refresh-schema` in the tail.
    RefreshSchema {
        config: Box<CommandConfig>,
        cmd_dir: PathBuf,
        cmd_name: String,
    },
    /// Path terminal with `--help` / `-h` in the tail.
    PrintCommandHelp {
        config: Box<CommandConfig>,
        name: String,
    },
    /// Preset terminal with `--help` / `-h` in the tail.
    PrintPresetHelp {
        config: Box<CommandConfig>,
        parent_label: String,
        preset_name: String,
    },
    /// Run terminal with `--help` / `-h` in the tail.
    PrintRunHelp {
        name: String,
        description: Option<String>,
        run: String,
        append: Option<String>,
        docker: Option<String>,
    },
    /// The top-level or descended-into name doesn't exist in the current
    /// `commands:` map. Caller prints the project-help banner.
    UnknownCommand { name: String },
    /// A Preset-kind command at the top level has nothing to reuse an
    /// `entry:` from. Caller prints a pointer to the fix.
    PresetAtTopLevel { name: String },
    /// Structural error: spec declares both `run:` and `path:`, or a
    /// child fdl.yml failed to load / parse. String is the diagnostic.
    Error(String),
}

/// Walk the command graph from a top-level name and produce a
/// [`WalkOutcome`]. Every transition is pure: the walker never spawns a
/// process, prints to stdout, or reads the process cwd. Inputs carry all
/// the context needed.
///
/// - `cmd_name`: the top-level token the user typed (`fdl <cmd_name> ...`).
/// - `tail`: positional args following `cmd_name` (typically `&args[2..]`).
/// - `top_commands`: the root `commands:` block (usually
///   `&project.commands`).
/// - `project_root`: the directory containing the base `fdl.yml`; acts
///   as the initial `current_dir` for Path resolution.
/// - `env`: active overlay name, threaded to each `load_command_with_env`
///   call so descended configs pick up env-layered fields.
pub fn walk_commands(
    cmd_name: &str,
    tail: &[String],
    top_commands: &BTreeMap<String, CommandSpec>,
    project_root: &Path,
    env: Option<&str>,
) -> WalkOutcome {
    let mut commands: BTreeMap<String, CommandSpec> = top_commands.clone();
    let mut enclosing: Option<CommandConfig> = None;
    let mut current_dir: PathBuf = project_root.to_path_buf();
    let mut name: String = cmd_name.to_string();
    // `qualified` tracks the space-separated path the user typed
    // (`flodl-hf export`) so help renderers can show the correct
    // invocation. `name` is always the leaf used for command-map lookup.
    let mut qualified: String = cmd_name.to_string();
    // Accumulates `CommandSpec.cluster` at every step (root → leaf). The
    // caller pairs this with `ProjectConfig.cluster` to decide whether the
    // resolved terminal command should fan out across the cluster. Per-
    // command `cluster: false` deeper in the chain overrides an ancestor's
    // `cluster: true`; see [`config::resolve_cluster_dispatch`].
    let mut cluster_chain: Vec<Option<bool>> = Vec::new();
    let mut current_tail: Vec<String> = tail.to_vec();

    loop {
        let spec = match commands.get(&name) {
            Some(s) => s.clone(),
            None => return WalkOutcome::UnknownCommand { name },
        };
        cluster_chain.push(spec.cluster);

        let kind = match spec.kind() {
            Ok(k) => k,
            Err(e) => return WalkOutcome::Error(format!("command `{name}`: {e}")),
        };

        match kind {
            CommandKind::Run => {
                let command = spec
                    .run
                    .expect("Run kind guarantees `run` is set");
                if current_tail.iter().any(|a| a == "--help" || a == "-h") {
                    return WalkOutcome::PrintRunHelp {
                        name: qualified,
                        description: spec.description,
                        run: command,
                        append: spec.append,
                        docker: spec.docker,
                    };
                }
                // Split tail on the first `--`: anything before is
                // unexpected (no fdl-side flags exist for run-kind), and
                // anything after is forwarded to the script. Loud
                // rejection of stray args mirrors the loud-errors-over-
                // silent rule.
                let (before, after) = match current_tail.iter().position(|a| a == "--") {
                    Some(idx) => {
                        let after = current_tail[idx + 1..].to_vec();
                        let before = current_tail[..idx].to_vec();
                        (before, after)
                    }
                    None => (current_tail.clone(), Vec::new()),
                };
                if !before.is_empty() {
                    return WalkOutcome::Error(format!(
                        "command `{name}` does not accept extra args; \
                         use `fdl {name} -- {}` to forward them to the script",
                        before.join(" ")
                    ));
                }
                return WalkOutcome::RunScript {
                    command,
                    append: spec.append,
                    user_args: after,
                    docker: spec.docker,
                    cwd: current_dir,
                    cluster_chain,
                };
            }
            CommandKind::Path => {
                match classify_path_step(&spec, &name, &current_dir, &current_tail, env) {
                    PathOutcome::LoadFailed(msg) => return WalkOutcome::Error(msg),
                    PathOutcome::Descend {
                        child,
                        new_dir,
                        new_name,
                    } => {
                        commands = child.commands.clone();
                        enclosing = Some(*child);
                        current_dir = new_dir;
                        qualified.push(' ');
                        qualified.push_str(&new_name);
                        name = new_name;
                        // classify_path_step returned Descend because
                        // current_tail[0] named a nested command; consume
                        // that token before the next iteration.
                        if !current_tail.is_empty() {
                            current_tail.remove(0);
                        }
                    }
                    PathOutcome::ShowHelp { child } => {
                        return WalkOutcome::PrintCommandHelp {
                            config: child,
                            name: qualified,
                        };
                    }
                    PathOutcome::RefreshSchema { child, child_dir } => {
                        return WalkOutcome::RefreshSchema {
                            config: child,
                            cmd_dir: child_dir,
                            cmd_name: qualified,
                        };
                    }
                    PathOutcome::Exec { child, child_dir } => {
                        return WalkOutcome::ExecCommand {
                            config: child,
                            preset: None,
                            tail: current_tail,
                            cmd_dir: child_dir,
                            cluster_chain,
                        };
                    }
                }
            }
            CommandKind::Preset => {
                let Some(encl) = enclosing.take() else {
                    return WalkOutcome::PresetAtTopLevel { name };
                };

                if current_tail.iter().any(|a| a == "--help" || a == "-h") {
                    let parent_label = current_dir
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    return WalkOutcome::PrintPresetHelp {
                        config: Box::new(encl),
                        parent_label,
                        preset_name: name,
                    };
                }

                return WalkOutcome::ExecCommand {
                    config: Box::new(encl),
                    preset: Some(name),
                    tail: current_tail,
                    cmd_dir: current_dir,
                    cluster_chain,
                };
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal tempdir helper — avoids pulling in the `tempfile` crate.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let base = std::env::temp_dir();
            let unique = format!(
                "flodl-dispatch-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            );
            let dir = base.join(unique);
            std::fs::create_dir_all(&dir).expect("tempdir creation");
            Self(dir)
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

    /// Write a sub-command fdl.yml at `dir/sub/fdl.yml` with the given body.
    fn mkcmd(base: &Path, sub: &str, body: &str) -> PathBuf {
        let dir = base.join(sub);
        std::fs::create_dir_all(&dir).expect("mkcmd dir");
        std::fs::write(dir.join("fdl.yml"), body).expect("mkcmd write");
        dir
    }

    fn path_spec() -> CommandSpec {
        // Convention-default Path: no fields set, `kind()` returns Path.
        CommandSpec::default()
    }

    #[test]
    fn classify_descends_when_tail_names_nested_command() {
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "ddp-bench",
            "entry: echo\ncommands:\n  quick:\n    options: { model: linear }\n",
        );
        let spec = path_spec();
        let tail = vec!["quick".to_string()];
        let out = classify_path_step(&spec, "ddp-bench", tmp.path(), &tail, None);
        match out {
            PathOutcome::Descend { new_name, .. } => assert_eq!(new_name, "quick"),
            _ => panic!("expected Descend, got something else"),
        }
    }

    #[test]
    fn classify_show_help_when_tail_has_flag() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["--help".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::ShowHelp { .. }));
    }

    #[test]
    fn classify_show_help_short_flag() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["-h".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::ShowHelp { .. }));
    }

    #[test]
    fn classify_refresh_schema() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["--refresh-schema".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::RefreshSchema { .. }));
    }

    #[test]
    fn classify_exec_when_tail_has_no_known_token() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail = vec!["--model".to_string(), "linear".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }

    #[test]
    fn classify_exec_when_tail_is_empty() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "entry: echo\n");
        let spec = path_spec();
        let tail: Vec<String> = vec![];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }

    #[test]
    fn classify_descend_wins_over_help_at_same_level() {
        // `fdl sub quick --help` must render help for `quick` (handled
        // one level deeper), not for `sub`. Descent wins over help at
        // the current step.
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "sub",
            "entry: echo\ncommands:\n  quick:\n    options: { x: 1 }\n",
        );
        let spec = path_spec();
        let tail = vec!["quick".to_string(), "--help".to_string()];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Descend { .. }));
    }

    #[test]
    fn classify_bare_no_entry_with_subcommands_shows_help() {
        // Bare `fdl <project>` on a project that defines `commands:` but
        // no top-level `entry:` should print help, not error with
        // "no entry point defined". Mirrors the top-level `fdl` UX.
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "sub",
            "commands:\n  foo:\n    run: echo foo\n",
        );
        let spec = path_spec();
        let tail: Vec<String> = vec![];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::ShowHelp { .. }));
    }

    #[test]
    fn classify_no_entry_no_subcommands_still_falls_through() {
        // No entry, no sub-commands: keep the existing exec path so the
        // downstream "no entry point defined" error fires for genuinely
        // misconfigured projects.
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "sub", "description: empty\n");
        let spec = path_spec();
        let tail: Vec<String> = vec![];
        let out = classify_path_step(&spec, "sub", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }

    #[test]
    fn classify_load_failed_when_no_child_fdl_yml() {
        let tmp = TempDir::new();
        let spec = path_spec();
        let tail: Vec<String> = vec![];
        let out = classify_path_step(&spec, "missing", tmp.path(), &tail, None);
        match out {
            PathOutcome::LoadFailed(msg) => assert!(msg.contains("no fdl.yml")),
            _ => panic!("expected LoadFailed, got something else"),
        }
    }

    #[test]
    fn classify_uses_explicit_path() {
        // Explicit `path:` overrides the convention default. Drop the
        // child fdl.yml under `actual/` and point `path:` there.
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "actual", "entry: echo\n");
        let spec = CommandSpec {
            path: Some("actual".into()),
            ..Default::default()
        };
        let tail: Vec<String> = vec![];
        // `name` here is the command's label, not where we load from —
        // `actual/` is the real dir courtesy of `path:`.
        let out = classify_path_step(&spec, "label", tmp.path(), &tail, None);
        assert!(matches!(out, PathOutcome::Exec { .. }));
    }

    // ── walk_commands: outer walker ──────────────────────────────────────
    //
    // These drive the full walk from top-level down, asserting on the
    // terminal WalkOutcome variant. No processes are spawned — the walker
    // is pure, so tests stay fast and hermetic.

    /// Build a top-level `commands:` map by parsing a short YAML snippet.
    fn top_commands(yaml: &str) -> BTreeMap<String, CommandSpec> {
        #[derive(serde::Deserialize)]
        struct Root {
            #[serde(default)]
            commands: BTreeMap<String, CommandSpec>,
        }
        serde_yaml::from_str::<Root>(yaml)
            .expect("parse top-level commands")
            .commands
    }

    fn args(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn walk_top_level_run_returns_run_script() {
        let tmp = TempDir::new();
        let commands = top_commands("commands:\n  greet:\n    run: echo hello\n");
        let out = walk_commands("greet", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::RunScript {
                command,
                append,
                user_args,
                docker,
                cwd,
                cluster_chain,
            } => {
                assert_eq!(command, "echo hello");
                assert!(append.is_none());
                assert!(user_args.is_empty());
                assert!(docker.is_none());
                assert_eq!(cwd, tmp.path());
                // No cluster: directive anywhere → chain has one entry: None.
                assert_eq!(cluster_chain, vec![None]);
            }
            _ => panic!("expected RunScript"),
        }
    }

    #[test]
    fn walk_top_level_run_with_docker_preserves_service() {
        let tmp = TempDir::new();
        let commands = top_commands(
            "commands:\n  dev:\n    run: cargo test\n    docker: dev\n",
        );
        let out = walk_commands("dev", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::RunScript { docker, .. } => {
                assert_eq!(docker.as_deref(), Some("dev"));
            }
            _ => panic!("expected RunScript with docker"),
        }
    }

    #[test]
    fn walk_run_with_help_prints_help_not_script() {
        let tmp = TempDir::new();
        let commands = top_commands(
            "commands:\n  test:\n    description: Run all CPU tests\n    run: cargo test\n    docker: dev\n",
        );
        let tail = args(&["--help"]);
        let out = walk_commands("test", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::PrintRunHelp {
                name,
                description,
                run,
                append,
                docker,
            } => {
                assert_eq!(name, "test");
                assert_eq!(description.as_deref(), Some("Run all CPU tests"));
                assert_eq!(run, "cargo test");
                assert!(append.is_none());
                assert_eq!(docker.as_deref(), Some("dev"));
            }
            _ => panic!("expected PrintRunHelp"),
        }
    }

    #[test]
    fn walk_run_forwards_args_after_double_dash() {
        let tmp = TempDir::new();
        let commands = top_commands(
            "commands:\n  test:\n    run: cargo test live\n    append: -- --nocapture --ignored\n",
        );
        let tail = args(&["--", "-p", "flodl-hf"]);
        let out = walk_commands("test", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::RunScript {
                command,
                append,
                user_args,
                ..
            } => {
                assert_eq!(command, "cargo test live");
                assert_eq!(append.as_deref(), Some("-- --nocapture --ignored"));
                assert_eq!(user_args, vec!["-p".to_string(), "flodl-hf".to_string()]);
            }
            _ => panic!("expected RunScript"),
        }
    }

    #[test]
    fn walk_run_rejects_stray_args_before_double_dash() {
        let tmp = TempDir::new();
        let commands = top_commands("commands:\n  test:\n    run: cargo test\n");
        let tail = args(&["-p", "flodl-hf"]);
        let out = walk_commands("test", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::Error(msg) => {
                assert!(
                    msg.contains("does not accept extra args")
                        && msg.contains("fdl test -- -p flodl-hf"),
                    "got: {msg}"
                );
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn walk_run_rejects_stray_args_even_with_double_dash_after() {
        let tmp = TempDir::new();
        let commands = top_commands("commands:\n  test:\n    run: cargo test\n");
        // Stray `-p flodl-hf` BEFORE `--` is rejected even though `--`
        // appears later. The rule is structural: the tail must be empty
        // up to `--`.
        let tail = args(&["-p", "flodl-hf", "--", "extra"]);
        let out = walk_commands("test", &tail, &commands, tmp.path(), None);
        assert!(matches!(out, WalkOutcome::Error(_)));
    }

    #[test]
    fn walk_run_with_short_help_prints_help() {
        let tmp = TempDir::new();
        let commands = top_commands("commands:\n  test:\n    run: cargo test\n");
        let tail = args(&["-h"]);
        let out = walk_commands("test", &tail, &commands, tmp.path(), None);
        assert!(matches!(out, WalkOutcome::PrintRunHelp { .. }));
    }

    #[test]
    fn walk_unknown_top_level_returns_unknown() {
        let tmp = TempDir::new();
        let commands = top_commands("commands:\n  greet:\n    run: echo hello\n");
        let out = walk_commands("nope", &args(&["arg"]), &commands, tmp.path(), None);
        match out {
            WalkOutcome::UnknownCommand { name } => assert_eq!(name, "nope"),
            _ => panic!("expected UnknownCommand"),
        }
    }

    #[test]
    fn walk_top_level_preset_errors_without_enclosing() {
        // A top-level command with preset-shaped fields (`options:`) but
        // neither `run:` nor `path:` has no enclosing CommandConfig to
        // borrow an `entry:` from — must error loudly.
        let tmp = TempDir::new();
        let commands = top_commands(
            "commands:\n  orphan:\n    options: { model: linear }\n",
        );
        let out = walk_commands("orphan", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::PresetAtTopLevel { name } => assert_eq!(name, "orphan"),
            _ => panic!("expected PresetAtTopLevel"),
        }
    }

    #[test]
    fn walk_run_and_path_both_set_is_error() {
        let tmp = TempDir::new();
        let commands = top_commands(
            "commands:\n  bad:\n    run: echo hi\n    path: ./sub\n",
        );
        let out = walk_commands("bad", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::Error(msg) => {
                assert!(msg.contains("bad"), "got: {msg}");
                assert!(msg.contains("both `run:` and `path:`"), "got: {msg}");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn walk_path_exec_at_one_level() {
        // Top-level `ddp-bench` path-kind → no further descent → Exec.
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "ddp-bench", "entry: cargo run -p ddp-bench\n");
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let tail = args(&["--seed", "42"]);
        let out = walk_commands("ddp-bench", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand {
                preset,
                tail: returned_tail,
                cmd_dir,
                ..
            } => {
                assert!(preset.is_none());
                assert_eq!(returned_tail, args(&["--seed", "42"]));
                assert_eq!(cmd_dir, tmp.path().join("ddp-bench"));
            }
            _ => panic!("expected ExecCommand"),
        }
    }

    #[test]
    fn walk_path_then_preset_at_two_levels() {
        // fdl.yml: commands: { ddp-bench: {} }  → path kind, convention
        // ddp-bench/fdl.yml: commands: { quick: { options: { model: linear } } }
        // Invocation: `fdl ddp-bench quick --epochs 5`
        // Expected: descend into ddp-bench, resolve `quick` as preset,
        // emit ExecCommand with preset=Some("quick"), tail=["--epochs","5"].
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "ddp-bench",
            "entry: cargo run -p ddp-bench\n\
             commands:\n  quick:\n    options: { model: linear }\n",
        );
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let tail = args(&["quick", "--epochs", "5"]);
        let out = walk_commands("ddp-bench", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand {
                preset,
                tail: returned_tail,
                cmd_dir,
                ..
            } => {
                assert_eq!(preset.as_deref(), Some("quick"));
                assert_eq!(returned_tail, args(&["--epochs", "5"]));
                assert_eq!(cmd_dir, tmp.path().join("ddp-bench"));
            }
            _ => panic!("expected ExecCommand with preset"),
        }
    }

    #[test]
    fn walk_path_then_path_then_preset_at_three_levels() {
        // Three-level walk: `fdl a b quick`.
        // tmp/fdl.yml             → commands: { a: {} }
        // tmp/a/fdl.yml           → commands: { b: {} }   + entry (required for preset parent)
        // tmp/a/b/fdl.yml         → commands: { quick: { options: { x: 1 } } } + entry
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "a",
            "entry: echo a\ncommands:\n  b: {}\n",
        );
        // b is a sibling directory under a/
        let b_dir = tmp.path().join("a").join("b");
        std::fs::create_dir_all(&b_dir).unwrap();
        std::fs::write(
            b_dir.join("fdl.yml"),
            "entry: echo b\ncommands:\n  quick:\n    options: { x: 1 }\n",
        )
        .unwrap();
        let commands = top_commands("commands:\n  a: {}\n");
        let tail = args(&["b", "quick"]);
        let out = walk_commands("a", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand {
                preset, cmd_dir, ..
            } => {
                assert_eq!(preset.as_deref(), Some("quick"));
                assert_eq!(cmd_dir, b_dir);
            }
            _ => panic!("expected ExecCommand with preset at depth 3"),
        }
    }

    #[test]
    fn walk_path_child_missing_returns_error() {
        // Convention-default Path for `ghost`, but tmp/ghost/fdl.yml doesn't exist.
        let tmp = TempDir::new();
        let commands = top_commands("commands:\n  ghost: {}\n");
        let out = walk_commands("ghost", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::Error(msg) => assert!(msg.contains("no fdl.yml"), "got: {msg}"),
            _ => panic!("expected Error(LoadFailed)"),
        }
    }

    #[test]
    fn walk_path_help_prints_command_help() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "ddp-bench", "entry: echo\n");
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let tail = args(&["--help"]);
        let out = walk_commands("ddp-bench", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::PrintCommandHelp { name, .. } => assert_eq!(name, "ddp-bench"),
            _ => panic!("expected PrintCommandHelp"),
        }
    }

    #[test]
    fn walk_preset_help_prints_preset_help() {
        // `fdl ddp-bench quick --help` — help applies to the preset, not
        // the enclosing command (descent wins at the classify level, then
        // Preset-kind with `--help` in the tail emits PrintPresetHelp).
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "ddp-bench",
            "entry: echo\ncommands:\n  quick:\n    options: { x: 1 }\n",
        );
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let tail = args(&["quick", "--help"]);
        let out = walk_commands("ddp-bench", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::PrintPresetHelp {
                parent_label,
                preset_name,
                ..
            } => {
                assert_eq!(preset_name, "quick");
                assert_eq!(parent_label, "ddp-bench");
            }
            _ => panic!("expected PrintPresetHelp"),
        }
    }

    #[test]
    fn walk_path_refresh_schema() {
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "ddp-bench", "entry: echo\n");
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let tail = args(&["--refresh-schema"]);
        let out = walk_commands("ddp-bench", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::RefreshSchema { cmd_name, .. } => {
                assert_eq!(cmd_name, "ddp-bench");
            }
            _ => panic!("expected RefreshSchema"),
        }
    }

    #[test]
    fn walk_env_propagates_to_child_overlay() {
        // Base child says entry=echo-base; env overlay fdl.ci.yml
        // overrides entry=echo-ci. After descent with env=Some("ci"),
        // the ExecCommand carries the overlaid config.
        let tmp = TempDir::new();
        let child = mkcmd(tmp.path(), "ddp-bench", "entry: echo-base\n");
        std::fs::write(child.join("fdl.ci.yml"), "entry: echo-ci\n").unwrap();
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let out = walk_commands("ddp-bench", &[], &commands, tmp.path(), Some("ci"));
        match out {
            WalkOutcome::ExecCommand { config, .. } => {
                assert_eq!(config.entry.as_deref(), Some("echo-ci"));
            }
            _ => panic!("expected ExecCommand with env-overlaid entry"),
        }
    }

    #[test]
    fn walk_env_none_ignores_overlay() {
        // Same fixtures as above, but env=None — base must win.
        let tmp = TempDir::new();
        let child = mkcmd(tmp.path(), "ddp-bench", "entry: echo-base\n");
        std::fs::write(child.join("fdl.ci.yml"), "entry: echo-ci\n").unwrap();
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let out = walk_commands("ddp-bench", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand { config, .. } => {
                assert_eq!(config.entry.as_deref(), Some("echo-base"));
            }
            _ => panic!("expected ExecCommand with base entry"),
        }
    }

    // ── cluster_chain accumulation along the walk ───────────────────

    #[test]
    fn walk_run_with_cluster_true_carries_single_entry_chain() {
        let tmp = TempDir::new();
        let commands = top_commands(
            "commands:\n  train:\n    cluster: true\n    run: cargo run\n",
        );
        let out = walk_commands("train", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::RunScript { cluster_chain, .. } => {
                assert_eq!(cluster_chain, vec![Some(true)]);
            }
            _ => panic!("expected RunScript"),
        }
    }

    #[test]
    fn walk_path_carries_ancestor_cluster_into_chain() {
        // Root marks ddp-bench: cluster: true. Sub-fdl.yml's leaf command
        // leaves it unset → chain = [Some(true), None]. Per the resolver,
        // this would inherit cluster: true at dispatch.
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "ddp-bench", "entry: cargo run -p ddp-bench\n");
        let commands = top_commands(
            "commands:\n  ddp-bench:\n    cluster: true\n",
        );
        let out = walk_commands("ddp-bench", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand { cluster_chain, .. } => {
                assert_eq!(cluster_chain, vec![Some(true)]);
            }
            _ => panic!("expected ExecCommand"),
        }
    }

    #[test]
    fn walk_path_preset_chain_includes_both_levels() {
        // Root: ddp-bench (Path, cluster: true). ddp-bench/fdl.yml: quick
        // preset with cluster: false → chain = [Some(true), Some(false)].
        // Per the resolver, leaf override wins → effective false (stays
        // local despite ancestor saying cluster).
        let tmp = TempDir::new();
        mkcmd(
            tmp.path(),
            "ddp-bench",
            "entry: cargo run -p ddp-bench\n\
             commands:\n  quick:\n    cluster: false\n    options: { model: linear }\n",
        );
        let commands = top_commands(
            "commands:\n  ddp-bench:\n    cluster: true\n",
        );
        let tail = args(&["quick"]);
        let out = walk_commands("ddp-bench", &tail, &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand {
                preset,
                cluster_chain,
                ..
            } => {
                assert_eq!(preset.as_deref(), Some("quick"));
                assert_eq!(cluster_chain, vec![Some(true), Some(false)]);
            }
            _ => panic!("expected ExecCommand with preset"),
        }
    }

    #[test]
    fn walk_no_cluster_anywhere_yields_all_none_chain() {
        // Plain `fdl ddp-bench` with no cluster directives anywhere →
        // chain has one None entry per walked level. The resolver returns
        // false (no dispatch) for any all-None chain.
        let tmp = TempDir::new();
        mkcmd(tmp.path(), "ddp-bench", "entry: cargo run -p ddp-bench\n");
        let commands = top_commands("commands:\n  ddp-bench: {}\n");
        let out = walk_commands("ddp-bench", &[], &commands, tmp.path(), None);
        match out {
            WalkOutcome::ExecCommand { cluster_chain, .. } => {
                assert_eq!(cluster_chain, vec![None]);
            }
            _ => panic!("expected ExecCommand"),
        }
    }
}
