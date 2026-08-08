//! `fdl nccl` subcommand surface (build NCCL from source).

use std::process::ExitCode;

use builtins::NcclBuildArgs;
use flodl_cli::cli_error;
use flodl_cli::{builtins, nccl};

use crate::parse_sub;

// ---------------------------------------------------------------------------
// nccl dispatch
// ---------------------------------------------------------------------------

pub(crate) fn dispatch_nccl(args: &[String]) -> ExitCode {
    let sub = args.get(2).map(String::as_str).unwrap_or("--help");
    match sub {
        "build" => {
            let cli: NcclBuildArgs = parse_sub("fdl nccl build", &args[2..]);
            cmd_nccl_build(cli)
        }
        "--help" | "-h" => {
            print_nccl_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown nccl command: {other}");
            eprintln!();
            print_nccl_usage();
            ExitCode::FAILURE
        }
    }
}

fn cmd_nccl_build(cli: NcclBuildArgs) -> ExitCode {
    use nccl::build::BuildOpts;

    if cli.jobs == 0 {
        cli_error!("--jobs must be a positive number");
        return ExitCode::FAILURE;
    }

    let opts = BuildOpts {
        tag: cli.tag,
        archs: cli.archs,
        max_jobs: cli.jobs,
        dry_run: cli.dry_run,
    };

    match nccl::build::run(opts) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            cli_error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn print_nccl_usage() {
    println!("fdl nccl -- build NCCL from source (heterogeneous-arch bridge)");
    println!();
    println!("USAGE:");
    println!("    fdl nccl <command> [options]");
    println!();
    println!("COMMANDS:");
    println!("    build              Compile libnccl for the local GPU arch");
    println!("        --tag <tag>      NCCL git tag (default: infer from active libtorch)");
    println!("        --archs <list>   Override CUDA architectures (e.g. \"6.1;12.0\")");
    println!("        --jobs <n>       Parallel compilation jobs (default: 6)");
    println!("        --dry-run        Print what would happen, do nothing");
    println!();
    println!("Output: libtorch/nccl/builds/v<version>-<archs>/");
    println!("Wire via cluster.yml: worker.env.LD_PRELOAD: <path>/lib/libnccl.so.2");
}
