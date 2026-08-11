//! `fdl cargo` subcommand surface: footprint report + per-tier `--clear`.
//!
//! Bare `fdl cargo` reports both tiers. Clearing always goes through a
//! tier sub-command (`fdl cargo target --clear` / `fdl cargo cache
//! --clear`) so the caller states which reclaim cost they accept:
//! compiled artifacts come back with a recompute (no network), registry
//! caches with a re-download (needs network).

use std::process::ExitCode;

use builtins::{CargoArgs, CargoCacheArgs, CargoTargetArgs};
use cargo::{ClearOutcome, DiskRoot, Tier, Usage, clear_contents, discover, format_bytes, usage};
use context::Context;
use flodl_cli::cli_error;
use flodl_cli::{builtins, cargo, context, util};

use crate::parse_sub;

// ---------------------------------------------------------------------------
// cargo dispatch
// ---------------------------------------------------------------------------

pub(crate) fn dispatch_cargo(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str);
    match sub {
        Some("target") => {
            let cli: CargoTargetArgs = parse_sub("fdl cargo target", &args[2..]);
            run_tier(Tier::Target, cli.clear, cli.json)
        }
        Some("cache") => {
            let cli: CargoCacheArgs = parse_sub("fdl cargo cache", &args[2..]);
            run_tier(Tier::Cache, cli.clear, cli.json)
        }
        Some("--help") | Some("-h") => {
            print_cargo_usage();
            ExitCode::SUCCESS
        }
        _ => {
            // Clearing must pick a tier: the two tiers have different
            // reclaim costs, and that choice is the whole point.
            if args[2..].iter().any(|a| a == "--clear") {
                cli_error!("`fdl cargo --clear` needs a tier");
                eprintln!("  fdl cargo target --clear   # compiled artifacts (recompute)");
                eprintln!("  fdl cargo cache --clear    # registry caches (re-download)");
                return ExitCode::FAILURE;
            }
            let cli: CargoArgs = parse_sub("fdl cargo", &args[1..]);
            run_report(cli.json)
        }
    }
}

fn resolve_project_root() -> Result<std::path::PathBuf, ExitCode> {
    let ctx = Context::resolve();
    if !ctx.is_project {
        cli_error!("no flodl project found (walked up from the current directory)");
        eprintln!("  `fdl cargo` inspects a project checkout's build footprint;");
        eprintln!("  run it from inside one.");
        return Err(ExitCode::FAILURE);
    }
    Ok(cargo::workspace_root(&ctx.root))
}

// ---------------------------------------------------------------------------
// report (bare `fdl cargo`, and tier without --clear)
// ---------------------------------------------------------------------------

struct MeasuredRoot {
    root: DiskRoot,
    usage: Usage,
}

fn measure(roots: Vec<DiskRoot>) -> Vec<MeasuredRoot> {
    roots
        .into_iter()
        .map(|root| {
            let usage = usage(&root.path);
            MeasuredRoot { root, usage }
        })
        .collect()
}

fn run_report(json: bool) -> ExitCode {
    let root = match resolve_project_root() {
        Ok(r) => r,
        Err(code) => return code,
    };
    let measured = measure(discover(&root));
    if json {
        print_report_json(&root, &measured, None);
    } else {
        print_report_human(&root, &measured, None);
    }
    ExitCode::SUCCESS
}

fn run_tier(tier: Tier, clear: bool, json: bool) -> ExitCode {
    let root = match resolve_project_root() {
        Ok(r) => r,
        Err(code) => return code,
    };
    let roots: Vec<DiskRoot> = discover(&root)
        .into_iter()
        .filter(|r| r.tier == tier)
        .collect();
    if !clear {
        let measured = measure(roots);
        if json {
            print_report_json(&root, &measured, Some(tier));
        } else {
            print_report_human(&root, &measured, Some(tier));
        }
        return ExitCode::SUCCESS;
    }
    run_clear(tier, roots, json)
}

fn print_report_human(root: &std::path::Path, measured: &[MeasuredRoot], only: Option<Tier>) {
    println!("cargo footprint at {}", root.display());
    let mut total = 0u64;
    for tier in [Tier::Target, Tier::Cache] {
        if only.is_some_and(|t| t != tier) {
            continue;
        }
        let rows: Vec<&MeasuredRoot> = measured.iter().filter(|m| m.root.tier == tier).collect();
        println!();
        println!("{} (reclaim = {})", tier.heading(), tier.reclaim());
        if rows.is_empty() {
            println!("  none found");
            continue;
        }
        let mut subtotal = 0u64;
        for m in &rows {
            let note = if m.usage.unreadable > 0 {
                format!("  ({} unreadable entries not counted)", m.usage.unreadable)
            } else {
                String::new()
            };
            println!(
                "  {:<24} {:>10}{}",
                m.root.label,
                format_bytes(m.usage.bytes),
                note
            );
            subtotal += m.usage.bytes;
        }
        println!(
            "  {:<24} {:>10}   reclaim: fdl cargo {} --clear",
            "subtotal",
            format_bytes(subtotal),
            tier.name()
        );
        total += subtotal;
    }
    if only.is_none() {
        println!();
        println!("  {:<24} {:>10}", "total", format_bytes(total));
    }
}

fn print_report_json(root: &std::path::Path, measured: &[MeasuredRoot], only: Option<Tier>) {
    let mut total = 0u64;
    print!(
        "{{\"root\":\"{}\",\"tiers\":[",
        esc(&root.display().to_string())
    );
    let mut first_tier = true;
    for tier in [Tier::Target, Tier::Cache] {
        if only.is_some_and(|t| t != tier) {
            continue;
        }
        if !first_tier {
            print!(",");
        }
        first_tier = false;
        print!(
            "{{\"tier\":\"{}\",\"reclaim\":\"{}\",\"paths\":[",
            tier.name(),
            esc(tier.reclaim())
        );
        let mut subtotal = 0u64;
        let rows = measured.iter().filter(|m| m.root.tier == tier);
        for (i, m) in rows.enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                "{{\"path\":\"{}\",\"bytes\":{},\"files\":{},\"unreadable\":{}}}",
                esc(&m.root.label),
                m.usage.bytes,
                m.usage.files,
                m.usage.unreadable
            );
            subtotal += m.usage.bytes;
        }
        print!("],\"subtotal_bytes\":{subtotal}}}");
        total += subtotal;
    }
    println!("],\"total_bytes\":{total}}}");
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

/// How many skipped entries the human report names before folding the
/// rest into a count (a root-owned tree can skip thousands).
const SKIP_DETAIL_CAP: usize = 5;

fn run_clear(tier: Tier, roots: Vec<DiskRoot>, json: bool) -> ExitCode {
    let cleared: Vec<(DiskRoot, ClearOutcome)> = roots
        .into_iter()
        .map(|root| {
            let outcome = clear_contents(&root.path);
            (root, outcome)
        })
        .collect();

    if json {
        print_clear_json(tier, &cleared);
        return ExitCode::SUCCESS;
    }

    println!("clearing {} ({})", tier.heading(), tier.reclaim());
    if cleared.is_empty() {
        println!("  none found");
        return ExitCode::SUCCESS;
    }
    let mut freed = 0u64;
    let mut skipped = 0usize;
    let mut left = 0u64;
    for (root, outcome) in &cleared {
        let skip_note = if outcome.skipped.is_empty() {
            String::new()
        } else {
            format!(
                "  skipped {} entries ({} left)",
                outcome.skipped.len(),
                format_bytes(outcome.skipped_bytes)
            )
        };
        println!(
            "  {:<24} freed {:>10}{}",
            root.label,
            format_bytes(outcome.freed),
            skip_note
        );
        for s in outcome.skipped.iter().take(SKIP_DETAIL_CAP) {
            println!("    {}: {}", s.path.display(), s.error);
        }
        if outcome.skipped.len() > SKIP_DETAIL_CAP {
            println!(
                "    ... and {} more",
                outcome.skipped.len() - SKIP_DETAIL_CAP
            );
        }
        if outcome.permission_skips() {
            // Absolute path, and a form that keeps the directory itself:
            // it may be a bind-mount source, and a glob would miss the
            // dotfiles cargo leaves at a target root.
            println!("    hint: root-owned content (docker container artifacts); reclaim with:");
            println!(
                "          sudo find {} -mindepth 1 -delete",
                root.path.display()
            );
        }
        freed += outcome.freed;
        skipped += outcome.skipped.len();
        left += outcome.skipped_bytes;
    }
    println!();
    // The binaries went with it, and a symlink pointing into the tree
    // (a remote host's `fdl`, a hand-made convenience link) resolves to
    // nothing until something rebuilds. Cheaper to say than to detect:
    // the link usually lives on another box.
    if tier == Tier::Target && freed > 0 {
        println!("note: binaries in this tree are gone; anything symlinked into it");
        println!("      needs a rebuild before it resolves again.");
        println!();
    }
    if skipped == 0 {
        println!("reclaimed {}", format_bytes(freed));
    } else {
        println!(
            "reclaimed {}; {} still held by {} skipped entries (see above)",
            format_bytes(freed),
            format_bytes(left),
            skipped
        );
    }
    ExitCode::SUCCESS
}

fn print_clear_json(tier: Tier, cleared: &[(DiskRoot, ClearOutcome)]) {
    let mut freed = 0u64;
    let mut skipped = 0usize;
    let mut left = 0u64;
    print!("{{\"tier\":\"{}\",\"cleared\":[", tier.name());
    for (i, (root, outcome)) in cleared.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        let first_error = outcome
            .skipped
            .first()
            .map(|s| format!("\"{}\"", esc(&s.error)))
            .unwrap_or_else(|| "null".to_string());
        print!(
            "{{\"path\":\"{}\",\"freed_bytes\":{},\"removed_files\":{},\"skipped\":{},\"skipped_bytes\":{},\"first_error\":{}}}",
            esc(&root.label),
            outcome.freed,
            outcome.removed_files,
            outcome.skipped.len(),
            outcome.skipped_bytes,
            first_error
        );
        freed += outcome.freed;
        skipped += outcome.skipped.len();
        left += outcome.skipped_bytes;
    }
    println!("],\"freed_bytes\":{freed},\"skipped_total\":{skipped},\"skipped_bytes\":{left}}}");
}

fn esc(s: &str) -> String {
    util::system::escape_json(s)
}

// ---------------------------------------------------------------------------
// usage
// ---------------------------------------------------------------------------

fn print_cargo_usage() {
    println!("fdl cargo -- report / reclaim cargo's on-disk footprint");
    println!();
    println!("USAGE:");
    println!("    fdl cargo [--json]             # both tiers, sizes only (always safe)");
    println!("    fdl cargo <tier> [--clear]     # one tier; --clear reclaims it");
    println!();
    println!("TIERS:");
    println!("    target             Compiled artifacts: target/, .target*, excluded");
    println!("                       crates' target/. Reclaim = recompute, no network.");
    println!("    cache              Registry caches: .cargo-cache*, .cargo-git*.");
    println!("                       Reclaim = re-download, needs network.");
    println!();
    println!("OPTIONS:");
    println!("    --clear            Delete the tier's contents (directories are kept:");
    println!("                       several are docker bind-mount sources)");
    println!("    --json             Machine-readable output");
    println!();
    println!("EXAMPLES:");
    println!("    fdl cargo                      # what is all this disk going to?");
    println!("    fdl cargo target --clear       # reclaim build artifacts (offline-safe)");
    println!("    fdl cargo cache --clear        # reclaim registry caches");
}
