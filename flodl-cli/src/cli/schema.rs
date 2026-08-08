//! `fdl schema` subcommand surface (list/clear/refresh cached --fdl-schema outputs).

use std::env;
use std::process::ExitCode;

use builtins::{SchemaClearArgs, SchemaListArgs, SchemaRefreshArgs};
use flodl_cli::cli_error;
use flodl_cli::{builtins, config, schema, style, util};

use crate::parse_sub;

// ---------------------------------------------------------------------------
// schema dispatch
// ---------------------------------------------------------------------------

pub(crate) fn dispatch_schema(args: &[String]) -> ExitCode {
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
        cli_error!(
            "no fdl.yml found in {} or parent directories",
            env::current_dir().unwrap_or_default().display()
        );
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
            let rel = c.cache_path.strip_prefix(&root).unwrap_or(&c.cache_path);
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
        cli_error!(
            "no fdl.yml found in {} or parent directories",
            env::current_dir().unwrap_or_default().display()
        );
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
        cli_error!(
            "no fdl.yml found in {} or parent directories",
            env::current_dir().unwrap_or_default().display()
        );
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
