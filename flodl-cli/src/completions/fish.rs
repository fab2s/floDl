//! Fish completion script emitter.

use super::*;

pub(super) fn emit_fish(data: &CompletionData) -> String {
    let mut s = String::new();
    s.push_str("# fdl fish completion (generated)\n");
    s.push_str("# fdl completions fish | source\n");
    s.push_str("complete -c fdl -f\n\n");

    // Helper: walk up from $PWD looking for fdl.yml, then list sibling
    // fdl.<env>.yml overlays. Runtime detection so the script works
    // across projects and project edits without re-installing.
    s.push_str("function __fdl_find_envs\n");
    s.push_str("    set -l _dir $PWD\n");
    s.push_str("    while true\n");
    s.push_str("        if test -f \"$_dir/fdl.yml\"\n");
    s.push_str("            for _f in \"$_dir\"/fdl.*.yml\n");
    s.push_str("                test -f \"$_f\"; or continue\n");
    s.push_str("                set -l _n (string replace -r '^.*/fdl\\.' '' -- $_f)\n");
    s.push_str("                set _n (string replace -r '\\.yml$' '' -- $_n)\n");
    s.push_str("                if test -n \"$_n\"; and not string match -q '*.*' -- \"$_n\"\n");
    s.push_str("                    echo \"@$_n\"\n");
    s.push_str("                end\n");
    s.push_str("            end\n");
    s.push_str("            return\n");
    s.push_str("        end\n");
    s.push_str("        test \"$_dir\" = '/'; and return\n");
    s.push_str("        set _dir (dirname \"$_dir\")\n");
    s.push_str("    end\n");
    s.push_str("end\n\n");

    // Helper: walk the pre-cursor tokens, skip "fdl" and an optional
    // `@env` selector at position 2 (`fdl @cluster <cmd>`), return the
    // command token. Empty when the user hasn't typed past the env /
    // fdl yet.
    s.push_str("function __fdl_active_command\n");
    s.push_str("    set -l toks (commandline -opc)\n");
    s.push_str("    set -l idx 2\n");
    s.push_str("    if test (count $toks) -ge 2\n");
    s.push_str("        set -l envs (__fdl_find_envs)\n");
    s.push_str("        if contains -- $toks[2] $envs\n");
    s.push_str("            set idx 3\n");
    s.push_str("        end\n");
    s.push_str("    end\n");
    s.push_str("    if test (count $toks) -ge $idx\n");
    s.push_str("        echo $toks[$idx]\n");
    s.push_str("    end\n");
    s.push_str("end\n\n");

    // Helper: return the sub-command token (one past the active
    // command). Used for nested-builtin rules like `libtorch download`.
    s.push_str("function __fdl_active_subcommand\n");
    s.push_str("    set -l toks (commandline -opc)\n");
    s.push_str("    set -l idx 3\n");
    s.push_str("    if test (count $toks) -ge 2\n");
    s.push_str("        set -l envs (__fdl_find_envs)\n");
    s.push_str("        if contains -- $toks[2] $envs\n");
    s.push_str("            set idx 4\n");
    s.push_str("        end\n");
    s.push_str("    end\n");
    s.push_str("    if test (count $toks) -ge $idx\n");
    s.push_str("        echo $toks[$idx]\n");
    s.push_str("    end\n");
    s.push_str("end\n\n");

    // Predicate: true when we're at the command slot (either fresh
    // start with no command typed, or right after an env name has been
    // consumed). The env-aware mirror of fish's built-in
    // `__fish_use_subcommand`.
    s.push_str("function __fdl_at_command_position\n");
    s.push_str("    test -z (__fdl_active_command)\n");
    s.push_str("end\n\n");

    // Top-level surface.
    // - Env names: position 1 only (no env-chaining), so keep on
    //   `__fish_use_subcommand`.
    // - Commands: fire at both fresh-start AND post-env via
    //   `__fdl_at_command_position`.
    // - Top-level flags (--help, --version, --env, ...): only valid
    //   before any command, so `__fish_use_subcommand`.
    s.push_str("# Top-level\n");
    s.push_str(
        "complete -c fdl -n '__fish_use_subcommand' -a '(__fdl_find_envs)' \
         -d 'env overlay'\n",
    );
    for word in &data.top_level {
        s.push_str(&format!(
            "complete -c fdl -n '__fdl_at_command_position' -a '{word}'\n"
        ));
    }
    for flag in TOP_FLAGS {
        if let Some(long) = flag.strip_prefix("--") {
            s.push_str(&format!(
                "complete -c fdl -n '__fish_use_subcommand' -l '{long}'\n"
            ));
        } else if let Some(short) = flag.strip_prefix('-') {
            s.push_str(&format!(
                "complete -c fdl -n '__fish_use_subcommand' -s '{short}'\n"
            ));
        }
    }
    s.push('\n');

    // Sub-command-specific completions. Every per-command rule
    // predicates on `__fdl_active_command` so it fires correctly under
    // both `fdl <cmd>` and `fdl @<env> <cmd>` (and won't false-positive
    // on weird command lines where the name appears later as an
    // option value, the way `__fish_seen_subcommand_from` would).
    //
    // Uses `contains` instead of `test ... = ...` so an empty
    // `__fdl_active_command` (no command typed yet) cleanly evaluates
    // to false instead of erroring with "missing argument" at the
    // `test` site.
    for cmd in &data.commands {
        s.push_str(&format!("# {name}\n", name = cmd.name));
        let cond = format!("contains -- {} (__fdl_active_command)", cmd.name);

        for (name, desc) in &cmd.presets {
            let safe = desc
                .as_deref()
                .unwrap_or("preset")
                .replace('\'', "\\'");
            s.push_str(&format!(
                "complete -c fdl -n '{cond}' -a '{name}' -d '{safe}'\n"
            ));
        }
        for sub in &cmd.sub_commands {
            s.push_str(&format!(
                "complete -c fdl -n '{cond}' -a '{sub}' -d 'command'\n"
            ));
        }

        for opt in &cmd.options {
            let long = &opt.long;

            // Base flag declaration.
            let mut line = format!("complete -c fdl -n '{cond}' -l '{long}'");
            if let Some(short_flag) = &opt.short {
                line.push_str(&format!(" -s '{short_flag}'"));
            }

            // Value completion.
            match &opt.value {
                ValueKind::None => {}
                ValueKind::Choices(cs) => {
                    line.push_str(" -r -f -a '");
                    line.push_str(&cs.join(" "));
                    line.push('\'');
                }
                ValueKind::Path => {
                    line.push_str(" -r -F");
                }
                ValueKind::Completer(c) => {
                    line.push_str(&format!(" -r -f -a '({c})'"));
                }
                ValueKind::Any => {
                    line.push_str(" -r");
                }
            }

            // Fish shows -d descriptions inline in the completion menu.
            if let Some(desc) = &opt.description {
                let safe = desc.replace('\'', "\\'");
                line.push_str(&format!(" -d '{safe}'"));
            }
            line.push('\n');
            s.push_str(&line);
        }

        // Variant-shaped entry binary: offer its subcommand names (with
        // descriptions), then per-subcommand flag rules gated on the
        // active subcommand (the project analogue of nested-builtin rules).
        for tc in &cmd.tree_commands {
            let safe = tc
                .description
                .as_deref()
                .unwrap_or("command")
                .replace('\'', "\\'");
            s.push_str(&format!(
                "complete -c fdl -n '{cond}' -a '{}' -d '{safe}'\n",
                tc.name
            ));
        }
        for tc in &cmd.tree_commands {
            let sub_cond = format!(
                "contains -- {} (__fdl_active_command); \
                 and contains -- {} (__fdl_active_subcommand)",
                cmd.name, tc.name
            );
            for opt in &tc.options {
                emit_fish_option_line(&mut s, &sub_cond, opt);
            }
        }
        s.push('\n');
    }

    // Built-in sub-command listings (parent entries) and their nested
    // flag rules. `__fdl_active_command` / `__fdl_active_subcommand`
    // pin the rule to the exact command + sub-command being typed,
    // regardless of whether an env prefix was used.
    for b in data.builtins.iter().filter(|b| b.path.len() == 1) {
        let name = &b.path[0];
        let has_subs = !b.sub_commands.is_empty();
        let has_opts = !b.options.is_empty();
        if !has_subs && !has_opts {
            continue;
        }
        s.push_str(&format!("# {}\n", b.joined_path()));
        let parent_cond = format!("contains -- {name} (__fdl_active_command)");

        if has_subs {
            for sub in &b.sub_commands {
                s.push_str(&format!(
                    "complete -c fdl -n '{parent_cond}' -a '{sub}'\n"
                ));
            }
        }

        if has_opts {
            for opt in &b.options {
                emit_fish_option_line(&mut s, &parent_cond, opt);
            }
        }
    }

    for b in data.builtins.iter().filter(|b| b.path.len() == 2) {
        if b.options.is_empty() {
            continue;
        }
        let parent = &b.path[0];
        let child = &b.path[1];
        s.push_str(&format!("# {} {}\n", parent, child));
        let cond = format!(
            "contains -- {parent} (__fdl_active_command); \
             and contains -- {child} (__fdl_active_subcommand)"
        );
        for opt in &b.options {
            emit_fish_option_line(&mut s, &cond, opt);
        }
    }

    s
}

fn emit_fish_option_line(s: &mut String, cond: &str, opt: &OptionCompletion) {
    let long = &opt.long;
    let mut line = format!("complete -c fdl -n '{cond}' -l '{long}'");
    if let Some(short_flag) = &opt.short {
        line.push_str(&format!(" -s '{short_flag}'"));
    }
    match &opt.value {
        ValueKind::None => {}
        ValueKind::Choices(cs) => {
            line.push_str(" -r -f -a '");
            line.push_str(&cs.join(" "));
            line.push('\'');
        }
        ValueKind::Path => {
            line.push_str(" -r -F");
        }
        ValueKind::Completer(c) => {
            line.push_str(&format!(" -r -f -a '({c})'"));
        }
        ValueKind::Any => {
            line.push_str(" -r");
        }
    }
    if let Some(desc) = &opt.description {
        let safe = desc.replace('\'', "\\'");
        line.push_str(&format!(" -d '{safe}'"));
    }
    line.push('\n');
    s.push_str(&line);
}
