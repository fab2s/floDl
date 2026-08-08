//! Bash completion script emitter.

use super::*;

pub(super) fn emit_bash(data: &CompletionData) -> String {
    let mut s = String::new();
    s.push_str("# fdl bash completion (generated)\n");
    s.push_str("# eval \"$(fdl completions bash)\"\n");
    // Helper: walk up from $PWD looking for fdl.yml, then enumerate
    // sibling fdl.<env>.yml overlays. Runtime detection (not embedded
    // at generation time) so the same eval'd script works across
    // projects and across project edits without re-installing.
    s.push_str("__fdl_find_envs() {\n");
    s.push_str("    local _dir=\"$PWD\" _f _n\n");
    s.push_str("    while :; do\n");
    s.push_str("        if [[ -f \"$_dir/fdl.yml\" ]]; then\n");
    s.push_str("            for _f in \"$_dir\"/fdl.*.yml; do\n");
    s.push_str("                [[ -f \"$_f\" ]] || continue\n");
    s.push_str("                _n=\"${_f##*/fdl.}\"\n");
    s.push_str("                _n=\"${_n%.yml}\"\n");
    // Skip empty (bare fdl.yml) and multi-dot names (e.g. fdl.foo.bar.yml).
    // Emit each overlay as a `@<env>` selector token.
    s.push_str("                [[ -n \"$_n\" && \"$_n\" != *.* ]] && printf '@%s\\n' \"$_n\"\n");
    s.push_str("            done\n");
    s.push_str("            return\n");
    s.push_str("        fi\n");
    s.push_str("        [[ \"$_dir\" == \"/\" ]] && return\n");
    s.push_str("        _dir=\"$(dirname \"$_dir\")\"\n");
    s.push_str("    done\n");
    s.push_str("}\n");
    s.push_str("_fdl_completions() {\n");
    s.push_str("    local cur prev cmd cword env_offset _fdl_envs\n");
    s.push_str("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    s.push_str("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n");
    s.push('\n');
    // `@<env>` selector: matches a sibling fdl.<env>.yml overlay. Offered
    // (with the `@` sigil) as a top-level candidate and completed at any
    // position the moment the current word starts with `@`.
    s.push_str("    _fdl_envs=\" $(__fdl_find_envs | tr '\\n' ' ') \"\n");
    s.push_str("    if [[ \"$cur\" == @* ]]; then\n");
    s.push_str("        COMPREPLY=($(compgen -W \"${_fdl_envs}\" -- \"$cur\"))\n");
    s.push_str("        return\n");
    s.push_str("    fi\n");
    // A `@<env>` token sitting in slot 1 shifts the command slot by one
    // (`fdl @cluster <cmd> ...`), so position checks below advance past it.
    s.push_str("    env_offset=0\n");
    s.push_str("    if [[ ${#COMP_WORDS[@]} -gt 1 && \"$_fdl_envs\" == *\" ${COMP_WORDS[1]} \"* ]]; then\n");
    s.push_str("        env_offset=1\n");
    s.push_str("    fi\n");
    s.push_str("    cmd=\"${COMP_WORDS[$((1 + env_offset))]}\"\n");
    s.push_str("    cword=$((COMP_CWORD - env_offset))\n");
    s.push('\n');

    // Position 1 (command slot, after the optional env): top-level
    // commands + flags. Envs are added too when env_offset == 0; once
    // an env is consumed, only commands are valid in this slot.
    let top = join_for_shell(&data.top_level);
    let top_with_flags = format!("{top} {}", TOP_FLAGS.join(" "));
    s.push_str("    if [[ $cword -eq 1 ]]; then\n");
    s.push_str("        if [[ $env_offset -eq 0 ]]; then\n");
    s.push_str(&format!(
        "            COMPREPLY=($(compgen -W \"{top_with_flags}${{_fdl_envs}}\" -- \"$cur\"))\n"
    ));
    s.push_str("        else\n");
    s.push_str(&format!(
        "            COMPREPLY=($(compgen -W \"{top_with_flags}\" -- \"$cur\"))\n"
    ));
    s.push_str("        fi\n");
    s.push_str("        return\n");
    s.push_str("    fi\n");

    // Variant-shaped entry binaries: per-subcommand value + flag blocks,
    // keyed on the env-shifted subcommand slot. Emitted BEFORE the generic
    // per-command blocks so the more specific rule wins (mirrors the
    // nested-builtin handling for `libtorch download`). Position-2 listing
    // of the subcommand names themselves comes from the per-command block's
    // first-positional word list (which now includes tree-command names).
    for cmd in &data.commands {
        for tc in &cmd.tree_commands {
            s.push_str(&format!(
                "\n    if [[ \"$cmd\" == \"{cmdname}\" && \"${{COMP_WORDS[$((2 + env_offset))]}}\" == \"{sub}\" && $cword -ge 3 ]]; then\n",
                cmdname = cmd.name,
                sub = tc.name
            ));
            s.push_str("        case \"$prev\" in\n");
            for opt in &tc.options {
                if !opt.takes_value {
                    continue;
                }
                let flags = opt.flag_tokens().join("|");
                let line = match &opt.value {
                    ValueKind::Choices(cs) => format!(
                        "            {flags}) COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")); return ;;\n",
                        cs.join(" ")
                    ),
                    ValueKind::Path => format!(
                        "            {flags}) COMPREPLY=($(compgen -f -- \"$cur\")); return ;;\n",
                    ),
                    ValueKind::Completer(c) => format!(
                        "            {flags}) COMPREPLY=($(compgen -W \"$({c})\" -- \"$cur\")); return ;;\n",
                    ),
                    ValueKind::Any => format!("            {flags}) return ;;\n",),
                    ValueKind::None => continue,
                };
                s.push_str(&line);
            }
            s.push_str("        esac\n");
            let mut flags: Vec<String> = tc.options.iter().flat_map(|o| o.flag_tokens()).collect();
            flags.push("--help".into());
            flags.push("-h".into());
            s.push_str(&format!(
                "        COMPREPLY=($(compgen -W \"{}\" -- \"$cur\"))\n",
                flags.join(" ")
            ));
            s.push_str("        return\n");
            s.push_str("    fi\n");
        }
    }

    // Sub-commands with schema / nested commands.
    for cmd in &data.commands {
        s.push_str(&format!(
            "\n    if [[ \"$cmd\" == \"{name}\" ]]; then\n",
            name = cmd.name
        ));

        // Value completion for options that take a value.
        s.push_str("        case \"$prev\" in\n");
        for opt in &cmd.options {
            if !opt.takes_value {
                continue;
            }
            let flags = opt.flag_tokens().join("|");
            let line = match &opt.value {
                ValueKind::Choices(cs) => format!(
                    "            {flags}) COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")); return ;;\n",
                    cs.join(" ")
                ),
                ValueKind::Path => format!(
                    "            {flags}) COMPREPLY=($(compgen -f -- \"$cur\")); return ;;\n",
                ),
                ValueKind::Completer(c) => format!(
                    "            {flags}) COMPREPLY=($(compgen -W \"$({c})\" -- \"$cur\")); return ;;\n",
                ),
                ValueKind::Any => format!("            {flags}) return ;;\n",),
                ValueKind::None => continue,
            };
            s.push_str(&line);
        }
        s.push_str("        esac\n");

        // At position 2, offer first-positional candidates (presets +
        // real sub-commands) plus option flags. Beyond position 2,
        // offer option flags only (prev-value already handled above).
        let option_flags: Vec<String> = cmd.options.iter().flat_map(|o| o.flag_tokens()).collect();
        let cmd_flags_str = {
            let mut v = option_flags.clone();
            v.push("--help".into());
            v.push("-h".into());
            v.join(" ")
        };
        let positional_tokens = cmd.first_positional_tokens();
        let positionals_str = positional_tokens.join(" ");

        s.push_str("        if [[ $cword -eq 2 ]]; then\n");
        if positional_tokens.is_empty() {
            s.push_str(&format!(
                "            COMPREPLY=($(compgen -W \"{cmd_flags_str}\" -- \"$cur\"))\n"
            ));
        } else {
            s.push_str(&format!(
                "            COMPREPLY=($(compgen -W \"{positionals_str} {cmd_flags_str}\" -- \"$cur\"))\n"
            ));
        }
        s.push_str("            return\n");
        s.push_str("        fi\n");

        s.push_str(&format!(
            "        COMPREPLY=($(compgen -W \"{cmd_flags_str}\" -- \"$cur\"))\n"
        ));
        s.push_str("        return\n");
        s.push_str("    fi\n");
    }

    // Nested built-ins first. Each block keys on `$cmd == <parent>` and
    // the post-env child-slot == <child>, so value-aware rules (e.g.
    // `--cuda` → `12.6 12.8`) fire before the single-level catch-all
    // below.
    for b in data.builtins.iter().filter(|b| b.path.len() == 2) {
        let parent = &b.path[0];
        let child = &b.path[1];
        s.push_str(&format!(
            "\n    if [[ \"$cmd\" == \"{parent}\" && \"${{COMP_WORDS[$((2 + env_offset))]}}\" == \"{child}\" && $cword -ge 3 ]]; then\n"
        ));
        s.push_str("        case \"$prev\" in\n");
        for opt in &b.options {
            if !opt.takes_value {
                continue;
            }
            let flags = opt.flag_tokens().join("|");
            let line = match &opt.value {
                ValueKind::Choices(cs) => format!(
                    "            {flags}) COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")); return ;;\n",
                    cs.join(" ")
                ),
                ValueKind::Path => format!(
                    "            {flags}) COMPREPLY=($(compgen -f -- \"$cur\")); return ;;\n",
                ),
                ValueKind::Completer(c) => format!(
                    "            {flags}) COMPREPLY=($(compgen -W \"$({c})\" -- \"$cur\")); return ;;\n",
                ),
                ValueKind::Any => format!("            {flags}) return ;;\n",),
                ValueKind::None => continue,
            };
            s.push_str(&line);
        }
        s.push_str("        esac\n");
        let flags_str = b.option_flags_with_help().join(" ");
        s.push_str(&format!(
            "        COMPREPLY=($(compgen -W \"{flags_str}\" -- \"$cur\"))\n"
        ));
        s.push_str("        return\n");
        s.push_str("    fi\n");
    }

    // Single-level built-ins. Parent entries (with sub_commands) offer
    // the sub-command list at position 2. Flag-carrying leaves offer
    // their options + `--help/-h` once past position 1.
    for b in data.builtins.iter().filter(|b| b.path.len() == 1) {
        let name = &b.path[0];
        let has_subs = !b.sub_commands.is_empty();
        let has_opts = !b.options.is_empty();
        if !has_subs && !has_opts {
            continue; // e.g. `version`, `completions`, `autocomplete`
        }

        // Value completion for option flags (applies at any position
        // beyond the command name).
        if b.options.iter().any(|o| o.takes_value) {
            s.push_str(&format!(
                "\n    if [[ \"$cmd\" == \"{name}\" && $cword -ge 2 ]]; then\n"
            ));
            s.push_str("        case \"$prev\" in\n");
            for opt in &b.options {
                if !opt.takes_value {
                    continue;
                }
                let flags = opt.flag_tokens().join("|");
                let line = match &opt.value {
                    ValueKind::Choices(cs) => format!(
                        "            {flags}) COMPREPLY=($(compgen -W \"{}\" -- \"$cur\")); return ;;\n",
                        cs.join(" ")
                    ),
                    ValueKind::Path => format!(
                        "            {flags}) COMPREPLY=($(compgen -f -- \"$cur\")); return ;;\n",
                    ),
                    ValueKind::Completer(c) => format!(
                        "            {flags}) COMPREPLY=($(compgen -W \"$({c})\" -- \"$cur\")); return ;;\n",
                    ),
                    ValueKind::Any => format!("            {flags}) return ;;\n",),
                    ValueKind::None => continue,
                };
                s.push_str(&line);
            }
            s.push_str("        esac\n    fi\n");
        }

        s.push_str(&format!(
            "\n    if [[ \"$cmd\" == \"{name}\" && $cword -eq 2 ]]; then\n"
        ));
        let mut position2_words: Vec<String> = b.sub_commands.clone();
        if has_opts {
            position2_words.extend(b.option_flags_with_help());
        } else if has_subs {
            // Parent-only — still offer --help/-h for `fdl <parent> --help`.
            position2_words.push("--help".into());
            position2_words.push("-h".into());
        }
        s.push_str(&format!(
            "        COMPREPLY=($(compgen -W \"{}\" -- \"$cur\"))\n",
            position2_words.join(" ")
        ));
        s.push_str("        return\n");
        s.push_str("    fi\n");

        if has_opts {
            s.push_str(&format!(
                "\n    if [[ \"$cmd\" == \"{name}\" && $cword -ge 3 ]]; then\n"
            ));
            s.push_str(&format!(
                "        COMPREPLY=($(compgen -W \"{}\" -- \"$cur\"))\n",
                b.option_flags_with_help().join(" ")
            ));
            s.push_str("        return\n");
            s.push_str("    fi\n");
        }
    }

    s.push_str("}\n");
    s.push_str("complete -F _fdl_completions fdl\n");
    s
}

fn join_for_shell(v: &[String]) -> String {
    v.join(" ")
}
