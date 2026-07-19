//! Shell completion generation and one-shot setup.
//!
//! `fdl completions bash`  outputs a bash completion script to stdout.
//! `fdl completions zsh`   outputs a zsh  completion script to stdout.
//! `fdl completions fish`  outputs a fish completion script to stdout.
//! `fdl autocomplete`      detects shell, installs completions, reloads.
//!
//! When a sub-command declares a `schema:` block in its fdl.yaml, completion
//! is option-aware: `choices:` drive value lists, `type: path` drives file
//! completion, `completer:` (opt-in) lets authors ship arbitrary shell
//! snippets (e.g. `ls runs/`).

use std::path::Path;

use crate::builtins;
use crate::cli_error;
use crate::config::{self, CommandConfig, CommandKind, OptionSpec, ProjectConfig};
use crate::style;

/// Reserved top-level flags, always offered at every position.
const TOP_FLAGS: &[&str] = &[
    "--help", "-h", "--version", "-V",
    "--env",
    "--ansi", "--no-ansi",
    "-v", "-vv", "-vvv", "--quiet", "-q",
];

/// The marker comment we use to detect our block in rc files.
const MARKER: &str = "# fdl shell completions";

// ── Intermediate model ──────────────────────────────────────────────────

/// Everything a completion emitter needs, shell-agnostic.
struct CompletionData {
    /// Top-level words: builtins + scripts + command names.
    top_level: Vec<String>,
    /// Per-command info for sub-commands declared in the root fdl.yaml.
    commands: Vec<CommandData>,
    /// Built-in commands and their nested sub-commands, derived from the
    /// [`builtins::registry`]. Carries both single-level entries
    /// (e.g. `install`) and nested leaves (e.g. `libtorch download`) so
    /// every level gets value-aware completion from the same pipeline.
    builtins: Vec<BuiltinCommandData>,
}

/// Completion data for one built-in path. `path.len() == 1` covers
/// top-level built-ins; `path.len() == 2` covers nested leaves.
/// Parents are represented as single-level entries whose `sub_commands`
/// lists the children — the emitter uses that to offer the sub-command
/// list at the next position.
struct BuiltinCommandData {
    path: Vec<String>,
    /// Direct children one level below `path`. Empty for leaves and for
    /// entries that have options but no sub-commands.
    sub_commands: Vec<String>,
    /// Flags declared by the `FdlArgs` struct, when the registry entry
    /// carries a `schema_fn`. Empty for parents and for entries parsed
    /// by hand (`config show`, `completions`, `autocomplete`, `version`).
    options: Vec<OptionCompletion>,
}

struct CommandData {
    name: String,
    /// Presets nested under this command — offered as first-positional
    /// values. Paired with their descriptions so zsh/fish can surface
    /// them through `_describe` / `complete -d`.
    presets: Vec<(String, Option<String>)>,
    /// Real sub-commands (Run / Path kinds). Also first-positional
    /// candidates, but without descriptions — their `fdl.yml` lives
    /// behind another lookup, so we keep this list lean.
    sub_commands: Vec<String>,
    options: Vec<OptionCompletion>,
    /// Sub-commands declared by the entry binary's own schema tree (a
    /// variant-shaped `#[derive(FdlArgs)]` CLI). Each carries its own
    /// option set, so completion can offer the right flags after the
    /// subcommand token — the project-command analogue of the nested
    /// built-in (`libtorch download`) machinery.
    tree_commands: Vec<TreeCommand>,
}

/// One subcommand of a variant-shaped entry binary, with its own flags.
struct TreeCommand {
    name: String,
    description: Option<String>,
    options: Vec<OptionCompletion>,
}

impl CommandData {
    /// Every first-positional candidate: presets, fdl.yml sub-commands,
    /// and the entry binary's own schema-tree subcommands. Used by shells
    /// that can't render descriptions (bash).
    fn first_positional_tokens(&self) -> Vec<String> {
        let mut out: Vec<String> = self.presets.iter().map(|(n, _)| n.clone()).collect();
        out.extend(self.sub_commands.iter().cloned());
        out.extend(self.tree_commands.iter().map(|t| t.name.clone()));
        out
    }
}

struct OptionCompletion {
    long: String,
    short: Option<String>,
    takes_value: bool,
    value: ValueKind,
    description: Option<String>,
}

enum ValueKind {
    /// bool flag, no value expected.
    None,
    /// Fixed list of choices.
    Choices(Vec<String>),
    /// File path completion.
    Path,
    /// Custom shell snippet (emitted verbatim).
    Completer(String),
    /// Free-form value, no hints.
    Any,
}

impl CompletionData {
    fn from_project(project: Option<(&ProjectConfig, &Path)>, env: Option<&str>) -> Self {
        let builtin_entries = Self::collect_builtins();

        // Top-level word list: every single-level built-in (visible +
        // hidden) in registry order, then project commands.
        let mut top_level: Vec<String> = builtin_entries
            .iter()
            .filter(|b| b.path.len() == 1)
            .map(|b| b.path[0].clone())
            .collect();
        let mut commands = Vec::new();

        if let Some((proj, root)) = project {
            for (name, spec) in &proj.commands {
                top_level.push(name.clone());

                // Only try to load a child fdl.yml for Path-kind commands.
                // `run:` commands are leaf scripts; presets at the root are
                // disallowed.
                let is_path_kind = spec.run.is_none();
                if !is_path_kind {
                    continue;
                }
                let child_dir = spec.resolve_path(name, root);
                if let Ok(cmd_config) = config::load_command_with_env(&child_dir, env) {
                    commands.push(CommandData::from_config(name.clone(), &cmd_config));
                } else {
                    commands.push(CommandData {
                        name: name.clone(),
                        presets: Vec::new(),
                        sub_commands: Vec::new(),
                        options: Vec::new(),
                        tree_commands: Vec::new(),
                    });
                }
            }
        }

        Self {
            top_level,
            commands,
            builtins: builtin_entries,
        }
    }

    /// Translate the static registry into completion-ready entries.
    /// Groups children under their top-level parent so single-level
    /// entries carry the sub-command list in registry order; nested
    /// entries with a `schema_fn` get their own entry for value-aware
    /// flag rules.
    fn collect_builtins() -> Vec<BuiltinCommandData> {
        let reg = builtins::registry();
        let mut out: Vec<BuiltinCommandData> = Vec::new();

        // Single-level entries carry the sub-command list for their
        // children, derived from registry order.
        for spec in reg.iter().filter(|s| s.path.len() == 1) {
            let name = spec.path[0];
            let sub_commands: Vec<String> = reg
                .iter()
                .filter(|s| s.path.len() == 2 && s.path[0] == name)
                .map(|s| s.path[1].to_string())
                .collect();
            let options = spec
                .schema_fn
                .map(|f| options_from_schema(&f()))
                .unwrap_or_default();
            out.push(BuiltinCommandData {
                path: vec![name.to_string()],
                sub_commands,
                options,
            });
        }

        // Nested entries that carry a schema get their own value-aware
        // block. Parents without a `schema_fn` at level 2 (e.g.
        // `libtorch info`) have no flags to offer beyond the sub-command
        // name already listed on the parent — skip them here.
        for spec in reg.iter().filter(|s| s.path.len() == 2) {
            let Some(schema_fn) = spec.schema_fn else {
                continue;
            };
            let options = options_from_schema(&schema_fn());
            out.push(BuiltinCommandData {
                path: spec.path.iter().map(|s| s.to_string()).collect(),
                sub_commands: Vec::new(),
                options,
            });
        }

        out
    }
}

fn options_from_schema(schema: &config::Schema) -> Vec<OptionCompletion> {
    schema
        .options
        .iter()
        .map(|(long, spec)| OptionCompletion::from_spec(long, spec))
        .collect()
}

impl BuiltinCommandData {
    fn joined_path(&self) -> String {
        self.path.join(" ")
    }

    fn option_flags_with_help(&self) -> Vec<String> {
        let mut v: Vec<String> = self.options.iter().flat_map(|o| o.flag_tokens()).collect();
        v.push("--help".into());
        v.push("-h".into());
        v
    }
}


impl CommandData {
    fn from_config(name: String, cfg: &CommandConfig) -> Self {
        // Split nested entries by kind so completions can treat presets
        // (named option bundles) separately from real sub-commands (an
        // inline run-script or another fdl.yml). Entries whose `kind()`
        // errors (e.g. both `run:` and `path:` set) are dropped — they
        // would fail at dispatch anyway; offering them as completions
        // would mislead the user.
        let mut presets: Vec<(String, Option<String>)> = Vec::new();
        let mut sub_commands: Vec<String> = Vec::new();
        for (sub_name, spec) in &cfg.commands {
            match spec.kind() {
                Ok(CommandKind::Preset) => {
                    presets.push((sub_name.clone(), spec.description.clone()));
                }
                Ok(CommandKind::Run) | Ok(CommandKind::Path) => {
                    sub_commands.push(sub_name.clone());
                }
                Err(_) => {}
            }
        }

        let options = cfg
            .schema
            .as_ref()
            .map(|s| {
                s.options
                    .iter()
                    .map(|(long, spec)| OptionCompletion::from_spec(long, spec))
                    .collect()
            })
            .unwrap_or_default();
        let tree_commands = cfg
            .schema
            .as_ref()
            .map(|s| {
                s.commands
                    .iter()
                    .map(|(sub_name, sub)| TreeCommand {
                        name: sub_name.clone(),
                        description: sub.description.clone(),
                        options: sub
                            .options
                            .iter()
                            .map(|(long, spec)| OptionCompletion::from_spec(long, spec))
                            .collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            name,
            presets,
            sub_commands,
            options,
            tree_commands,
        }
    }
}

impl OptionCompletion {
    fn from_spec(long: &str, spec: &OptionSpec) -> Self {
        let takes_value = spec.ty != "bool";
        let value = if !takes_value {
            ValueKind::None
        } else if let Some(choices) = &spec.choices {
            ValueKind::Choices(choices.iter().map(value_as_str).collect())
        } else if let Some(c) = &spec.completer {
            ValueKind::Completer(c.clone())
        } else if spec.ty == "path" || spec.ty == "list[path]" {
            ValueKind::Path
        } else {
            ValueKind::Any
        };
        Self {
            long: long.to_string(),
            short: spec.short.clone(),
            takes_value,
            value,
            description: spec.description.clone(),
        }
    }

    fn flag_tokens(&self) -> Vec<String> {
        let mut out = vec![format!("--{}", self.long)];
        if let Some(s) = &self.short {
            out.push(format!("-{s}"));
        }
        out
    }
}

fn value_as_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Generate a completion script for the given shell.
pub fn generate(shell: &str, project: Option<(&ProjectConfig, &Path)>, env: Option<&str>) {
    let data = CompletionData::from_project(project, env);
    match shell {
        "bash" => print!("{}", bash::emit_bash(&data)),
        "zsh" => print!("{}", zsh::emit_zsh(&data)),
        "fish" => print!("{}", fish::emit_fish(&data)),
        other => {
            eprintln!("unsupported shell: {other}");
            eprintln!("supported: bash, zsh, fish");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  eval \"$(fdl completions bash)\"   # bash");
            eprintln!("  eval \"$(fdl completions zsh)\"    # zsh");
            eprintln!("  fdl completions fish | source     # fish");
            eprintln!("  fdl autocomplete                  # auto-detect and install");
        }
    }
}

/// One-shot setup: detect shell, write to rc file, reload.
pub fn autocomplete(project: Option<(&ProjectConfig, &Path)>) {
    let shell = detect_shell();
    let (rc_path, shell_name) = match shell.as_str() {
        "fish" => {
            let home = std::env::var("HOME").unwrap_or_default();
            // Fish uses per-command files in ~/.config/fish/completions/.
            // A single rc-file block keeps the install story uniform here;
            // users who prefer the native approach can redirect manually.
            (format!("{home}/.config/fish/config.fish"), "fish")
        }
        "zsh" => {
            let home = std::env::var("HOME").unwrap_or_default();
            (format!("{home}/.zshrc"), "zsh")
        }
        _ => {
            let home = std::env::var("HOME").unwrap_or_default();
            (format!("{home}/.bashrc"), "bash")
        }
    };

    eprintln!(
        "{}  Detected shell: {}",
        style::green("*"),
        style::bold(shell_name)
    );
    eprintln!(
        "{}  Target: {}",
        style::green("*"),
        style::bold(&rc_path)
    );

    // Check if already installed.
    if let Ok(content) = std::fs::read_to_string(&rc_path) {
        if content.contains(MARKER) {
            eprintln!(
                "{}  Completions already installed. Updating...",
                style::yellow("*")
            );
            let cleaned = remove_completion_block(&content);
            if let Err(e) = std::fs::write(&rc_path, cleaned) {
                cli_error!("cannot write {rc_path}: {e}");
                return;
            }
        }
    }

    // Ensure target dir exists (needed for fish's config.fish).
    if let Some(parent) = std::path::Path::new(&rc_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let script = match shell_name {
        "fish" => "eval (fdl completions fish | string collect)\n".to_string(),
        "zsh" => "eval \"$(fdl completions zsh)\"\n".to_string(),
        _ => "eval \"$(fdl completions bash)\"\n".to_string(),
    };
    let _ = project; // project is only needed by generate(); install shells re-run fdl at load time

    let block = format!("\n{MARKER}\n{script}{MARKER} end\n");
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rc_path)
    {
        Ok(mut file) => {
            use std::io::Write;
            if let Err(e) = file.write_all(block.as_bytes()) {
                cli_error!("cannot append to {rc_path}: {e}");
                return;
            }
        }
        Err(e) => {
            cli_error!("cannot open {rc_path}: {e}");
            return;
        }
    }

    eprintln!(
        "{}  Completions installed.",
        style::green("*")
    );
    eprintln!();
    eprintln!(
        "  Reload with: {}",
        style::dim(&match shell_name {
            "fish" => format!("source {rc_path}"),
            _ => format!("source {rc_path}"),
        })
    );
    eprintln!("  Or restart your terminal for completions to take effect.");
}

fn detect_shell() -> String {
    if let Ok(shell) = std::env::var("SHELL") {
        if shell.contains("fish") {
            return "fish".into();
        }
        if shell.contains("zsh") {
            return "zsh".into();
        }
        if shell.contains("bash") {
            return "bash".into();
        }
    }
    "bash".into()
}

fn remove_completion_block(content: &str) -> String {
    let marker_end = format!("{MARKER} end");
    let mut result = String::new();
    let mut skipping = false;
    for line in content.lines() {
        if line.trim() == MARKER {
            skipping = true;
            continue;
        }
        if line.trim() == marker_end {
            skipping = false;
            continue;
        }
        if !skipping {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

mod bash;
mod fish;
mod zsh;

#[cfg(test)]
mod tests {
    use super::*;
    use super::{bash::emit_bash, fish::emit_fish, zsh::emit_zsh};
    use crate::config::{ArgSpec, OptionSpec, Schema};
    use std::collections::BTreeMap;

    fn make_cmd_with_schema() -> CommandData {
        let mut options = BTreeMap::new();
        options.insert(
            "model".into(),
            OptionSpec {
                ty: "string".into(),
                description: None,
                default: None,
                choices: Some(vec![
                    serde_json::json!("mlp"),
                    serde_json::json!("resnet"),
                ]),
                short: Some("m".into()),
                env: None,
                completer: None,
            },
        );
        options.insert(
            "baseline".into(),
            OptionSpec {
                ty: "path".into(),
                description: None,
                default: None,
                choices: None,
                short: None,
                env: None,
                completer: None,
            },
        );
        options.insert(
            "validate".into(),
            OptionSpec {
                ty: "bool".into(),
                description: None,
                default: None,
                choices: None,
                short: None,
                env: None,
                completer: None,
            },
        );
        let _: Vec<ArgSpec> = Vec::new(); // args intentionally empty here

        let schema = Schema {
            args: vec![],
            options,
            strict: false,
            ..Schema::default()
        };
        let cfg = CommandConfig {
            schema: Some(schema),
            ..Default::default()
        };
        CommandData::from_config("ddp-bench".into(), &cfg)
    }

    fn data_with_one_cmd() -> CompletionData {
        CompletionData {
            top_level: vec!["ddp-bench".into(), "libtorch".into()],
            commands: vec![make_cmd_with_schema()],
            builtins: CompletionData::collect_builtins(),
        }
    }

    #[test]
    fn option_completion_types_resolve() {
        let cmd = make_cmd_with_schema();
        let model = cmd
            .options
            .iter()
            .find(|o| o.long == "model")
            .expect("model option present");
        assert!(matches!(&model.value, ValueKind::Choices(cs) if cs == &vec!["mlp".to_string(), "resnet".into()]));
        let baseline = cmd
            .options
            .iter()
            .find(|o| o.long == "baseline")
            .expect("baseline option present");
        assert!(matches!(baseline.value, ValueKind::Path));
        let validate = cmd
            .options
            .iter()
            .find(|o| o.long == "validate")
            .expect("validate option present");
        assert!(!validate.takes_value);
        assert!(matches!(validate.value, ValueKind::None));
    }

    #[test]
    fn bash_contains_choices_and_path_completion() {
        let data = data_with_one_cmd();
        let out = emit_bash(&data);
        assert!(out.contains("--model|-m"), "model|short flag present");
        assert!(out.contains("mlp resnet"), "choice values inlined");
        assert!(out.contains("--baseline) COMPREPLY=($(compgen -f"), "path → file completion");
        assert!(!out.contains("--validate)"),
            "bool flags must not appear in value case (no value to complete)");
    }

    #[test]
    fn zsh_contains_choices_and_files() {
        let data = data_with_one_cmd();
        let out = emit_zsh(&data);
        assert!(out.contains("_values 'value' mlp resnet"));
        assert!(out.contains("_files"));
    }

    #[test]
    fn fish_contains_choices_and_path_flag() {
        let data = data_with_one_cmd();
        let out = emit_fish(&data);
        assert!(out.contains("-l 'model' -s 'm'"));
        assert!(out.contains("-a 'mlp resnet'"));
        assert!(out.contains("-l 'baseline' -r -F"));
    }

    /// Build a sub-command with a mix of preset, run, and path kinds
    /// so we can verify completions partition them correctly.
    fn make_cmd_with_mixed_kinds() -> CommandData {
        let mut commands: BTreeMap<String, crate::config::CommandSpec> = BTreeMap::new();
        let mut quick_opts = BTreeMap::new();
        quick_opts.insert("model".into(), serde_json::json!("linear"));
        commands.insert(
            "quick".into(),
            crate::config::CommandSpec {
                description: Some("fast smoke test".into()),
                options: quick_opts,
                ..Default::default()
            },
        );
        commands.insert(
            "helper".into(),
            crate::config::CommandSpec {
                description: Some("inline helper".into()),
                run: Some("echo hi".into()),
                ..Default::default()
            },
        );
        commands.insert(
            "nested".into(),
            crate::config::CommandSpec {
                path: Some("./nested/".into()),
                ..Default::default()
            },
        );

        let cfg = CommandConfig {
            commands,
            ..Default::default()
        };
        CommandData::from_config("parent".into(), &cfg)
    }

    #[test]
    fn from_config_splits_presets_from_sub_commands() {
        let cmd = make_cmd_with_mixed_kinds();
        assert_eq!(cmd.presets.len(), 1, "one preset expected");
        assert_eq!(cmd.presets[0].0, "quick");
        assert_eq!(cmd.presets[0].1.as_deref(), Some("fast smoke test"));
        // "helper" (Run) and "nested" (Path) are real sub-commands.
        let mut subs = cmd.sub_commands.clone();
        subs.sort();
        assert_eq!(subs, vec!["helper".to_string(), "nested".into()]);
    }

    #[test]
    fn zsh_emits_preset_descriptions() {
        let data = CompletionData {
            top_level: vec!["parent".into()],
            commands: vec![make_cmd_with_mixed_kinds()],
            builtins: CompletionData::collect_builtins(),
        };
        let out = emit_zsh(&data);
        assert!(
            out.contains("presets=('quick:fast smoke test')"),
            "zsh should surface preset descriptions via `name:desc` pairs; got:\n{out}"
        );
        assert!(
            out.contains("subcommands=(helper nested)"),
            "zsh should list real sub-commands separately"
        );
    }

    #[test]
    fn fish_emits_preset_descriptions() {
        let data = CompletionData {
            top_level: vec!["parent".into()],
            commands: vec![make_cmd_with_mixed_kinds()],
            builtins: CompletionData::collect_builtins(),
        };
        let out = emit_fish(&data);
        assert!(
            out.contains("-a 'quick' -d 'fast smoke test'"),
            "fish preset completion must carry description"
        );
        assert!(out.contains("-a 'helper' -d 'command'"));
        assert!(out.contains("-a 'nested' -d 'command'"));
    }

    #[test]
    fn bash_offers_both_kinds_as_positionals() {
        let data = CompletionData {
            top_level: vec!["parent".into()],
            commands: vec![make_cmd_with_mixed_kinds()],
            builtins: CompletionData::collect_builtins(),
        };
        let out = emit_bash(&data);
        assert!(
            out.contains("quick helper nested") || out.contains("helper nested")
                && out.contains("quick "),
            "bash must include preset + sub-command tokens in position-2 word list"
        );
    }

    #[test]
    fn top_level_flags_present_in_all_shells() {
        let data = data_with_one_cmd();
        // Each shell uses its own flag syntax: bash/zsh carry `--help` tokens
        // literally in word lists, while fish maps them to `-l 'help'`.
        let bash = emit_bash(&data);
        assert!(bash.contains("--help"), "bash: --help missing");
        assert!(bash.contains("--version"), "bash: --version missing");

        let zsh = emit_zsh(&data);
        assert!(zsh.contains("--help"), "zsh: --help missing");
        assert!(zsh.contains("--version"), "zsh: --version missing");

        let fish = emit_fish(&data);
        assert!(fish.contains("-l 'help'"), "fish: help long flag missing");
        assert!(fish.contains("-l 'version'"), "fish: version long flag missing");
        assert!(fish.contains("-s 'h'"), "fish: -h short missing");
    }

    /// Nested built-ins must emit their flag set at position-3+. The
    /// parent→child routing is the whole reason we unified the
    /// registry — before this refactor, `fdl libtorch download
    /// --<TAB>` offered nothing.
    #[test]
    fn nested_builtin_emits_flag_rule() {
        let data = CompletionData::from_project(None, None);
        let bash = emit_bash(&data);
        assert!(
            bash.contains(
                r#"if [[ "$cmd" == "libtorch" && "${COMP_WORDS[$((2 + env_offset))]}" == "download""#
            ),
            "bash must guard nested libtorch download block (env-offset-aware); got:\n{bash}"
        );
        assert!(
            bash.contains("--cpu --cuda --path --no-activate --dry-run --help -h")
                || bash.contains("--cpu")
                    && bash.contains("--cuda")
                    && bash.contains("--no-activate"),
            "bash nested block must list the download flag set; got:\n{bash}"
        );

        let fish = emit_fish(&data);
        assert!(
            fish.contains(
                "contains -- libtorch (__fdl_active_command); \
                 and contains -- download (__fdl_active_subcommand)"
            ),
            "fish must combine parent+child active-command predicates for nested rules; got:\n{fish}"
        );
    }

    /// A nested `--cuda` offers `12.6 12.8`, not just the flag name —
    /// the derive's `choices = &["12.6", "12.8"]` must flow all the
    /// way through to shell completion.
    #[test]
    fn nested_builtin_emits_value_rule_for_choices() {
        let data = CompletionData::from_project(None, None);
        let bash = emit_bash(&data);
        assert!(
            bash.contains("--cuda) COMPREPLY=($(compgen -W \"12.6 12.8\""),
            "bash must expand --cuda choices under the nested download block; got:\n{bash}"
        );

        let zsh = emit_zsh(&data);
        assert!(
            zsh.contains("--cuda) _values 'value' 12.6 12.8"),
            "zsh must expand --cuda choices under nested download arm; got:\n{zsh}"
        );

        let fish = emit_fish(&data);
        assert!(
            fish.contains("-l 'cuda'") && fish.contains("-a '12.6 12.8'"),
            "fish must emit --cuda choices for nested download; got:\n{fish}"
        );
    }

    /// Single-level built-ins with schemas still get their old
    /// completion coverage — no regression on today's behavior.
    #[test]
    fn single_level_builtin_still_offers_flags() {
        let data = CompletionData::from_project(None, None);
        let bash = emit_bash(&data);
        assert!(
            bash.contains(r#"if [[ "$cmd" == "install""#),
            "bash must still emit the install block; got:\n{bash}"
        );
        assert!(
            bash.contains("--check") && bash.contains("--dev") && bash.contains("-h"),
            "install flag set must survive the refactor"
        );
    }

    /// `@<env>` selector shift: `fdl @cluster <cmd>` must autocomplete.
    /// The generated bash script must (a) ship a runtime env discoverer
    /// that emits `@`-prefixed names, (b) complete `@<env>` at any
    /// position once the word starts with `@`, (c) compute env_offset
    /// from $COMP_WORDS[1], (d) shift cmd + cword, and (e) include
    /// detected envs in the position-1 word list.
    #[test]
    fn bash_emits_env_offset_shift_machinery() {
        let data = CompletionData::from_project(None, None);
        let bash = emit_bash(&data);
        // Runtime env discovery helper, emitting `@`-prefixed selectors.
        assert!(
            bash.contains("__fdl_find_envs()"),
            "bash must define __fdl_find_envs runtime helper; got:\n{bash}"
        );
        assert!(
            bash.contains("printf '@%s\\n'"),
            "bash env helper must emit `@`-prefixed env names; got:\n{bash}"
        );
        // Any-position `@<env>` completion branch.
        assert!(
            bash.contains(r#"if [[ "$cur" == @* ]]; then"#),
            "bash must complete `@<env>` when the word starts with @; got:\n{bash}"
        );
        // Glob pattern that catches fdl.<env>.yml without matching fdl.yml itself.
        assert!(
            bash.contains("\"$_dir\"/fdl.*.yml"),
            "bash env helper must glob fdl.*.yml siblings"
        );
        // env_offset computation.
        assert!(
            bash.contains("env_offset=0")
                && bash.contains("env_offset=1")
                && bash.contains(r#""$_fdl_envs" == *" ${COMP_WORDS[1]} "*"#),
            "bash must compute env_offset from COMP_WORDS[1]; got:\n{bash}"
        );
        // cmd + cword shift.
        assert!(
            bash.contains(r#"cmd="${COMP_WORDS[$((1 + env_offset))]}""#),
            "bash must shift cmd by env_offset"
        );
        assert!(
            bash.contains("cword=$((COMP_CWORD - env_offset))"),
            "bash must shift cword by env_offset"
        );
        // Position-1 word list pulls in envs when env_offset == 0.
        assert!(
            bash.contains("${_fdl_envs}"),
            "bash must include detected envs in the top-level word list"
        );
    }

    /// Same expectation on zsh: env_offset machinery + position-3
    /// dispatch on the env-shifted slot.
    #[test]
    fn zsh_emits_env_offset_shift_machinery() {
        let data = CompletionData::from_project(None, None);
        let zsh = emit_zsh(&data);
        assert!(
            zsh.contains("__fdl_find_envs()"),
            "zsh must define __fdl_find_envs"
        );
        assert!(
            zsh.contains("envs=(${(f)\"$(__fdl_find_envs)\"})"),
            "zsh must populate envs array from helper"
        );
        assert!(
            zsh.contains("local env_offset=0"),
            "zsh must declare env_offset"
        );
        assert!(
            zsh.contains("${envs[(I)$words[2]]}"),
            "zsh must test $words[2] against envs"
        );
        assert!(
            zsh.contains("local cword=$((CURRENT - env_offset))"),
            "zsh must compute cword from env_offset"
        );
        assert!(
            zsh.contains("case $words[$((2 + env_offset))] in"),
            "zsh top-level dispatch must use env-shifted slot"
        );
    }

    /// Fish goes all-in with approach (b): runtime helpers that walk
    /// the in-progress tokens, skip "fdl" and an optional env at
    /// position 2, and return the active command + sub-command tokens.
    /// Every per-command predicate switches from the loose
    /// `__fish_seen_subcommand_from` to the precise
    /// `test (__fdl_active_command) = '<name>'`, so rules fire
    /// identically with or without an env prefix and don't
    /// false-positive on weird command lines.
    #[test]
    fn fish_emits_env_aware_active_command_machinery() {
        let data = CompletionData::from_project(None, None);
        let fish = emit_fish(&data);

        // The three runtime helpers.
        assert!(
            fish.contains("function __fdl_find_envs"),
            "fish must define __fdl_find_envs function"
        );
        assert!(
            fish.contains("function __fdl_active_command"),
            "fish must define __fdl_active_command function"
        );
        assert!(
            fish.contains("function __fdl_active_subcommand"),
            "fish must define __fdl_active_subcommand function"
        );
        assert!(
            fish.contains("function __fdl_at_command_position"),
            "fish must define __fdl_at_command_position predicate"
        );

        // Env names surface at top level (fresh-start only).
        assert!(
            fish.contains(
                "complete -c fdl -n '__fish_use_subcommand' -a '(__fdl_find_envs)' -d 'env overlay'"
            ),
            "fish must surface env names at the top level via __fish_use_subcommand"
        );

        // Top-level commands use __fdl_at_command_position (fires both
        // at fresh start AND after an env was consumed).
        assert!(
            fish.contains("complete -c fdl -n '__fdl_at_command_position' -a 'install'")
                || fish.contains("complete -c fdl -n '__fdl_at_command_position' -a 'libtorch'"),
            "fish top-level commands must predicate on __fdl_at_command_position; got:\n{fish}"
        );

        // Per-command rules use `contains -- <name> (__fdl_active_command)`
        // (empty-safe vs `test ... = ...` when no command is typed yet).
        assert!(
            fish.contains("contains -- libtorch (__fdl_active_command)"),
            "fish per-command rules must use `contains` against __fdl_active_command"
        );

        // Old loose predicate must NOT appear anywhere.
        assert!(
            !fish.contains("__fish_seen_subcommand_from"),
            "fish must no longer use __fish_seen_subcommand_from (replaced by env-aware active-command helpers)"
        );
    }

    // ── Variant-shaped entry binary (tree schema) completions ──────────

    /// A command whose entry binary is a variant-shaped CLI: its schema is
    /// a tree of subcommands, each with its own flags.
    fn make_cmd_with_tree() -> CommandData {
        let mut train = Schema {
            description: Some("Train a model".into()),
            ..Schema::default()
        };
        let model = OptionSpec {
            ty: "string".into(),
            description: None,
            default: None,
            choices: Some(vec![serde_json::json!("mlp"), serde_json::json!("resnet")]),
            short: Some("m".into()),
            env: None,
            completer: None,
        };
        train.options.insert("model".into(), model);
        train.options.insert(
            "epochs".into(),
            OptionSpec {
                ty: "int".into(),
                description: None,
                default: None,
                choices: None,
                short: None,
                env: None,
                completer: None,
            },
        );

        let mut eval = Schema {
            description: Some("Evaluate a model".into()),
            ..Schema::default()
        };
        eval.options.insert(
            "checkpoint".into(),
            OptionSpec {
                ty: "path".into(),
                description: None,
                default: None,
                choices: None,
                short: None,
                env: None,
                completer: None,
            },
        );

        let mut schema = Schema::default();
        schema.commands.insert("train".into(), train);
        schema.commands.insert("eval".into(), eval);
        let cfg = CommandConfig {
            schema: Some(schema),
            ..Default::default()
        };
        CommandData::from_config("fbrl-letter".into(), &cfg)
    }

    fn tree_data() -> CompletionData {
        CompletionData {
            top_level: vec!["fbrl-letter".into()],
            commands: vec![make_cmd_with_tree()],
            builtins: CompletionData::collect_builtins(),
        }
    }

    #[test]
    fn tree_commands_become_first_positional_candidates() {
        let cmd = make_cmd_with_tree();
        let toks = cmd.first_positional_tokens();
        assert!(toks.contains(&"train".to_string()) && toks.contains(&"eval".to_string()));
    }

    #[test]
    fn bash_emits_per_subcommand_flag_block() {
        let out = emit_bash(&tree_data());
        // The env-offset-aware guard, mirroring nested builtins.
        assert!(
            out.contains(
                r#"if [[ "$cmd" == "fbrl-letter" && "${COMP_WORDS[$((2 + env_offset))]}" == "train""#
            ),
            "bash must guard a per-subcommand block for train; got:\n{out}"
        );
        // train's flags + its --model choices flow through.
        assert!(out.contains("--model|-m"), "train --model flag present");
        assert!(
            out.contains("--model|-m) COMPREPLY=($(compgen -W \"mlp resnet\""),
            "train --model choices must expand; got:\n{out}"
        );
        // eval's path option → file completion.
        assert!(
            out.contains(
                r#""${COMP_WORDS[$((2 + env_offset))]}" == "eval""#
            ) && out.contains("--checkpoint) COMPREPLY=($(compgen -f"),
            "eval --checkpoint must offer file completion; got:\n{out}"
        );
    }

    #[test]
    fn zsh_emits_tree_subcommands_and_per_sub_values() {
        let out = emit_zsh(&tree_data());
        assert!(
            out.contains("treecmds=('train:Train a model' 'eval:Evaluate a model')")
                || out.contains("'train:Train a model'"),
            "zsh must list tree subcommands with descriptions; got:\n{out}"
        );
        assert!(
            out.contains("case $words[$((3 + env_offset))] in"),
            "zsh must dispatch per-subcommand on the env-shifted slot"
        );
        assert!(
            out.contains("--model|-m) _values 'value' mlp resnet"),
            "zsh must expand train --model choices under its arm; got:\n{out}"
        );
    }

    #[test]
    fn fish_emits_tree_subcommands_and_per_sub_flags() {
        let out = emit_fish(&tree_data());
        assert!(
            out.contains("-a 'train' -d 'Train a model'"),
            "fish must offer the subcommand name with its description; got:\n{out}"
        );
        assert!(
            out.contains(
                "contains -- fbrl-letter (__fdl_active_command); \
                 and contains -- train (__fdl_active_subcommand)"
            ),
            "fish per-subcommand flag rules must gate on the active subcommand; got:\n{out}"
        );
        assert!(
            out.contains("-l 'model' -s 'm'") && out.contains("-a 'mlp resnet'"),
            "fish must emit train --model choices; got:\n{out}"
        );
    }
}
