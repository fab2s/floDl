//! Zsh completion script emitter.

use super::*;

pub(super) fn emit_zsh(data: &CompletionData) -> String {
    let mut s = String::new();
    s.push_str("#compdef fdl\n");
    s.push_str("# fdl zsh completion (generated)\n");
    s.push_str("# eval \"$(fdl completions zsh)\"\n");
    // Helper: walk up from $PWD looking for fdl.yml, then enumerate
    // sibling fdl.<env>.yml overlays. Runtime detection so a single
    // installed completion script works across projects and edits.
    s.push_str("__fdl_find_envs() {\n");
    s.push_str("    local _dir=\"$PWD\" _f _n\n");
    s.push_str("    while :; do\n");
    s.push_str("        if [[ -f \"$_dir/fdl.yml\" ]]; then\n");
    s.push_str("            for _f in \"$_dir\"/fdl.*.yml(N); do\n");
    s.push_str("                _n=\"${_f##*/fdl.}\"\n");
    s.push_str("                _n=\"${_n%.yml}\"\n");
    s.push_str("                [[ -n \"$_n\" && \"$_n\" != *.* ]] && print -- \"@$_n\"\n");
    s.push_str("            done\n");
    s.push_str("            return\n");
    s.push_str("        fi\n");
    s.push_str("        [[ \"$_dir\" == \"/\" ]] && return\n");
    s.push_str("        _dir=\"${_dir:h}\"\n");
    s.push_str("    done\n");
    s.push_str("}\n");
    s.push_str("_fdl() {\n");
    s.push_str("    local -a commands envs\n");
    let top_level_with_flags = {
        let mut v: Vec<String> = data.top_level.clone();
        for f in TOP_FLAGS {
            v.push((*f).to_string());
        }
        v
    };
    s.push_str(&format!(
        "    commands=({})\n",
        top_level_with_flags.join(" ")
    ));
    s.push_str("    envs=(${(f)\"$(__fdl_find_envs)\"})\n");
    s.push('\n');
    // `@<env>` selector shift: `fdl @cluster <cmd> ...`. When
    // $words[2] matches a discovered `@env`, shift positions by one so
    // every subsequent check sees the same indices as the no-env case.
    s.push_str("    local env_offset=0\n");
    s.push_str("    if (( ${#words} >= 2 )) && (( ${envs[(I)$words[2]]} )); then\n");
    s.push_str("        env_offset=1\n");
    s.push_str("    fi\n");
    s.push_str("    local cword=$((CURRENT - env_offset))\n");
    s.push('\n');

    s.push_str("    if (( cword == 2 )); then\n");
    s.push_str("        if (( env_offset == 0 )) && (( ${#envs} > 0 )); then\n");
    s.push_str("            _describe 'env overlay' envs\n");
    s.push_str("        fi\n");
    s.push_str("        _describe 'command' commands\n");
    s.push_str("        return\n");
    s.push_str("    fi\n");

    s.push_str("\n    case $words[$((2 + env_offset))] in\n");
    for cmd in &data.commands {
        s.push_str(&format!("        {name})\n", name = cmd.name));

        // Value completion when the previous word is a flag with choices/path/completer.
        if cmd.options.iter().any(|o| o.takes_value) {
            s.push_str("            case $words[CURRENT-1] in\n");
            for opt in &cmd.options {
                if !opt.takes_value {
                    continue;
                }
                let flags = opt.flag_tokens().join("|");
                let body = match &opt.value {
                    ValueKind::Choices(cs) => format!(
                        "                {flags}) _values 'value' {}; return ;;\n",
                        cs.join(" ")
                    ),
                    ValueKind::Path => format!("                {flags}) _files; return ;;\n",),
                    ValueKind::Completer(c) => format!(
                        "                {flags}) local -a vals; vals=(${{(f)\"$({c})\"}}); _describe 'value' vals; return ;;\n",
                    ),
                    ValueKind::Any => format!("                {flags}) return ;;\n",),
                    ValueKind::None => continue,
                };
                s.push_str(&body);
            }
            s.push_str("            esac\n");
        }

        // Position 3 (after the command name): offer nested sub-commands + options.
        let option_flags: Vec<String> = cmd.options.iter().flat_map(|o| o.flag_tokens()).collect();
        let mut all_flags = option_flags.clone();
        all_flags.push("--help".into());
        all_flags.push("-h".into());
        let flags_joined = all_flags.join(" ");

        // Position 3 (env-shifted): first-positional candidates.
        // Presets carry descriptions (zsh can render them via
        // `name:desc` pairs), real sub-commands do not.
        if !cmd.presets.is_empty() || !cmd.sub_commands.is_empty() || !cmd.tree_commands.is_empty()
        {
            s.push_str("            if (( cword == 3 )); then\n");
            if !cmd.presets.is_empty() {
                s.push_str("                local -a presets\n");
                let pairs: Vec<String> = cmd
                    .presets
                    .iter()
                    .map(|(n, d)| {
                        let desc = d.as_deref().unwrap_or("preset");
                        // Escape colons and single quotes to keep zsh
                        // from splitting the name:desc pair.
                        let safe = desc.replace('\'', "'\\''").replace(':', "\\:");
                        format!("'{n}:{safe}'")
                    })
                    .collect();
                s.push_str(&format!("                presets=({})\n", pairs.join(" ")));
                s.push_str("                _describe 'preset' presets\n");
            }
            if !cmd.sub_commands.is_empty() {
                s.push_str("                local -a subcommands\n");
                s.push_str(&format!(
                    "                subcommands=({})\n",
                    cmd.sub_commands.join(" ")
                ));
                s.push_str("                _describe 'command' subcommands\n");
            }
            // Variant-shaped entry binary: its own subcommands, with
            // descriptions sourced from the schema tree.
            if !cmd.tree_commands.is_empty() {
                s.push_str("                local -a treecmds\n");
                let pairs: Vec<String> = cmd
                    .tree_commands
                    .iter()
                    .map(|t| {
                        let desc = t.description.as_deref().unwrap_or("command");
                        let safe = desc.replace('\'', "'\\''").replace(':', "\\:");
                        format!("'{}:{safe}'", t.name)
                    })
                    .collect();
                s.push_str(&format!("                treecmds=({})\n", pairs.join(" ")));
                s.push_str("                _describe 'command' treecmds\n");
            }
            s.push_str("            fi\n");
        }
        // Per-subcommand value + flag dispatch (env-shifted slot), so
        // `fdl <cmd> <sub> --<TAB>` offers that subcommand's flags.
        if !cmd.tree_commands.is_empty() {
            s.push_str("            case $words[$((3 + env_offset))] in\n");
            for tc in &cmd.tree_commands {
                s.push_str(&format!("                {})\n", tc.name));
                if tc.options.iter().any(|o| o.takes_value) {
                    s.push_str("                    case $words[CURRENT-1] in\n");
                    for opt in &tc.options {
                        if !opt.takes_value {
                            continue;
                        }
                        let flags = opt.flag_tokens().join("|");
                        let body = match &opt.value {
                            ValueKind::Choices(cs) => format!(
                                "                        {flags}) _values 'value' {}; return ;;\n",
                                cs.join(" ")
                            ),
                            ValueKind::Path => {
                                format!("                        {flags}) _files; return ;;\n",)
                            }
                            ValueKind::Completer(c) => format!(
                                "                        {flags}) local -a vals; vals=(${{(f)\"$({c})\"}}); _describe 'value' vals; return ;;\n",
                            ),
                            ValueKind::Any => {
                                format!("                        {flags}) return ;;\n",)
                            }
                            ValueKind::None => continue,
                        };
                        s.push_str(&body);
                    }
                    s.push_str("                    esac\n");
                }
                let mut sub_flags: Vec<String> =
                    tc.options.iter().flat_map(|o| o.flag_tokens()).collect();
                sub_flags.push("--help".into());
                sub_flags.push("-h".into());
                s.push_str(&format!(
                    "                    _values 'option' {}\n                    ;;\n",
                    sub_flags.join(" ")
                ));
            }
            s.push_str("            esac\n");
        }
        s.push_str(&format!("            _values 'option' {flags_joined}\n"));
        s.push_str("            ;;\n");
    }

    // Built-in top-level entries: one match arm per top-level path.
    // Parents with sub-commands offer the sub-command list at CURRENT ==
    // 3, then dispatch per-child value/flag rules via a nested `case`.
    for b in data.builtins.iter().filter(|b| b.path.len() == 1) {
        let name = &b.path[0];
        let has_subs = !b.sub_commands.is_empty();
        let has_opts = !b.options.is_empty();
        if !has_subs && !has_opts {
            continue;
        }

        s.push_str(&format!("        {name})\n"));

        // Flag value completion (single-level options).
        if b.options.iter().any(|o| o.takes_value) {
            s.push_str("            case $words[CURRENT-1] in\n");
            for opt in &b.options {
                if !opt.takes_value {
                    continue;
                }
                let flags = opt.flag_tokens().join("|");
                let body = match &opt.value {
                    ValueKind::Choices(cs) => format!(
                        "                {flags}) _values 'value' {}; return ;;\n",
                        cs.join(" ")
                    ),
                    ValueKind::Path => format!("                {flags}) _files; return ;;\n",),
                    ValueKind::Completer(c) => format!(
                        "                {flags}) local -a vals; vals=(${{(f)\"$({c})\"}}); _describe 'value' vals; return ;;\n",
                    ),
                    ValueKind::Any => format!("                {flags}) return ;;\n",),
                    ValueKind::None => continue,
                };
                s.push_str(&body);
            }
            s.push_str("            esac\n");
        }

        if has_subs {
            s.push_str("            if (( cword == 3 )); then\n");
            s.push_str("                local -a subcmds\n");
            s.push_str(&format!(
                "                subcmds=({})\n",
                b.sub_commands.join(" ")
            ));
            s.push_str("                _describe 'subcommand' subcmds\n");
            s.push_str("            fi\n");

            // Nested flag/value rules: dispatch on the env-shifted child slot.
            let nested: Vec<&BuiltinCommandData> = data
                .builtins
                .iter()
                .filter(|n| n.path.len() == 2 && n.path[0] == *name)
                .collect();
            if !nested.is_empty() {
                s.push_str("            case $words[$((3 + env_offset))] in\n");
                for nb in nested {
                    let child = &nb.path[1];
                    s.push_str(&format!("                {child})\n"));
                    if nb.options.iter().any(|o| o.takes_value) {
                        s.push_str("                    case $words[CURRENT-1] in\n");
                        for opt in &nb.options {
                            if !opt.takes_value {
                                continue;
                            }
                            let flags = opt.flag_tokens().join("|");
                            let body = match &opt.value {
                                ValueKind::Choices(cs) => format!(
                                    "                        {flags}) _values 'value' {}; return ;;\n",
                                    cs.join(" ")
                                ),
                                ValueKind::Path => {
                                    format!("                        {flags}) _files; return ;;\n",)
                                }
                                ValueKind::Completer(c) => format!(
                                    "                        {flags}) local -a vals; vals=(${{(f)\"$({c})\"}}); _describe 'value' vals; return ;;\n",
                                ),
                                ValueKind::Any => {
                                    format!("                        {flags}) return ;;\n",)
                                }
                                ValueKind::None => continue,
                            };
                            s.push_str(&body);
                        }
                        s.push_str("                    esac\n");
                    }
                    let flags_quoted: Vec<String> = nb
                        .option_flags_with_help()
                        .iter()
                        .map(|f| format!("'{f}'"))
                        .collect();
                    s.push_str(&format!(
                        "                    _values 'option' {}\n",
                        flags_quoted.join(" ")
                    ));
                    s.push_str("                    ;;\n");
                }
                s.push_str("            esac\n");
            }
        }

        if has_opts {
            let flags_quoted: Vec<String> = b
                .option_flags_with_help()
                .iter()
                .map(|f| format!("'{f}'"))
                .collect();
            s.push_str(&format!(
                "            _values 'option' {}\n",
                flags_quoted.join(" ")
            ));
        }
        s.push_str("            ;;\n");
    }
    s.push_str("    esac\n");

    s.push_str("}\n");
    s.push_str("_fdl\n");
    s
}
