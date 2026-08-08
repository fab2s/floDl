//! `fdl libtorch` subcommand surface (list/info/activate/download/build/remove).

use std::process::ExitCode;

use builtins::{
    LibtorchActivateArgs, LibtorchBuildArgs, LibtorchDownloadArgs, LibtorchListArgs,
    LibtorchRemoveArgs,
};
use context::Context;
use flodl_cli::cli_error;
use flodl_cli::{builtins, context, libtorch, util};

use crate::parse_sub;

// ---------------------------------------------------------------------------
// libtorch dispatch
// ---------------------------------------------------------------------------

pub(crate) fn dispatch_libtorch(args: &[String]) -> ExitCode {
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
// nccl dispatch
// ---------------------------------------------------------------------------

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

    // --cpu / --cuda / --rocm each name a different build; at most one.
    let picked = [cli.cpu, cli.cuda.is_some(), cli.rocm.is_some()]
        .iter()
        .filter(|p| **p)
        .count();
    if picked > 1 {
        cli_error!("--cpu, --cuda and --rocm are mutually exclusive");
        return ExitCode::FAILURE;
    }

    let variant = if cli.cpu {
        Variant::Cpu
    } else if cli.rocm.is_some() {
        match cli.rocm.as_deref() {
            Some("7.0") => Variant::Rocm70,
            Some("7.1") => Variant::Rocm71,
            _ => unreachable!("validated by #[option(choices = ...)]"),
        }
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
        force_linux: false,
    };

    match libtorch::download::run(opts) {
        Ok(_) => ExitCode::SUCCESS,
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
    println!("        --rocm <ver>   AMD ROCm build instead of CUDA (7.0, 7.1)");
    println!("        --path <dir>   Install here (default: project libtorch/)");
    println!("        --no-activate  Do not activate after download");
    println!("        --dry-run      Print the resolved URL and stop");
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
