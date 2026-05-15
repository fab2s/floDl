//! flodl-cli — library side of the `fdl` binary.
//!
//! This crate is both a library and a binary. The binary (`fdl`) is the
//! user-facing driver; the library exposes the pieces that other crates
//! (e.g. a flodl-based training binary) need to integrate with the
//! `fdl` ecosystem:
//!
//! - [`FdlArgs`] — derive macro + trait for argv parsing and schema emission
//! - [`parse_or_schema`] — intercepts `--fdl-schema` / `--help` and dispatches
//! - [`Schema`], [`OptionSpec`], [`ArgSpec`] — the canonical schema shape
//!
//! # Example
//!
//! ```no_run
//! use flodl_cli::{FdlArgs, parse_or_schema};
//!
//! /// My training binary.
//! #[derive(FdlArgs, Debug)]
//! struct Cli {
//!     /// Model to run.
//!     #[option(short = 'm', default = "all")]
//!     model: String,
//!
//!     /// Write a report instead of training.
//!     #[option(default = "runs/report.md")]
//!     report: Option<String>,
//! }
//!
//! fn main() {
//!     let cli: Cli = parse_or_schema();
//!     // ... use cli.model, cli.report, etc.
//! }
//! ```

// Self-alias: the `#[derive(FdlArgs)]` macro emits `::flodl_cli::...`
// paths. That resolves automatically when the derive is used from a
// downstream crate (or from `main.rs`, which sees the lib as an external
// dep), but inside the library itself the compiler only knows the
// crate by its `crate`-root name. The alias makes `::flodl_cli::...`
// resolve to ourselves so `builtins.rs` can derive the same trait.
extern crate self as flodl_cli;

// Internal modules — shared by lib consumers and the fdl binary.

/// Structured API reference for flodl itself (`fdl api-ref`), used by
/// AI porting tools and as a machine-readable surface index.
pub mod api_ref;

/// Argv parsing primitives and the [`FdlArgsTrait`](args::FdlArgsTrait)
/// contract that `#[derive(FdlArgs)]` implements.
pub mod args;

/// Built-in `fdl` sub-commands (setup, install, completions, schema,
/// config, libtorch, diagnose, init, skill, ...).
pub mod builtins;

/// Shell completion script generation and per-project completion
/// enrichment driven by cached schemas.
pub mod completions;

/// `fdl.yml` manifest loading, validation, and resolved-command types.
pub mod config;

/// Cluster-mode env preparation. fdl-cli sets `FLODL_FULL_CLUSTER_JSON`
/// + `FLODL_FDL_CMD` + `FDL_ENV` on its process env so the user binary
/// inherits them and detects launcher role via
/// `flodl::distributed::launcher::dispatch`. Fan-out, log fan-in, and
/// CpuAverager all live on the flodl side.
/// Entry point [`cluster::prepare_cluster_env`]; recursion guard via
/// [`cluster::should_dispatch`].
pub mod cluster;

/// `--gpus` flag parsing + single-host cluster envelope synthesis (loopback,
/// one host, N ranks). Used when `--gpus` is set on a cluster-aware command
/// and no `cluster:` block is configured in YAML.
pub mod gpus;

/// Cross-cutting context passed to sub-command handlers (resolved config,
/// verbosity, overlay selection, working directory, ...).
pub mod context;

/// Top-level command dispatch: routing argv to built-ins vs. manifest
/// entries, resolving the three command kinds (run / path / preset).
pub mod dispatch;

/// Hardware and compatibility diagnostics (`fdl diagnose`).
pub mod diagnose;

/// Project scaffolding (`fdl init`): generates Dockerfile, `fdl.yml`,
/// training template, `.gitignore`.
pub mod init;

/// Ecosystem-crate scaffolding (`fdl add <target>`): drops a
/// configured sub-project inside a flodl project for hands-on
/// discovery. Currently supports `flodl-hf`.
pub mod add;

/// libtorch variant management (download, build, list, activate, remove,
/// info) used by both `fdl libtorch` and the standalone-manager flow.
pub mod libtorch;

/// Environment overlay loader (`--env`, `FDL_ENV`, first-arg convention)
/// with per-field origin annotations for `fdl config show`.
pub mod overlay;

/// Runtime: invoking resolved commands, streaming their output, and
/// mapping exit codes through `fdl`.
pub mod run;

/// `fdl schema` sub-command: discover every cache under the project,
/// report fresh / stale / orphan states, and clear or refresh on
/// demand. The [`Schema`] type itself lives in [`config`].
pub mod schema;

/// `--fdl-schema` binary contract and per-command cache mechanics.
/// Caches live at `<cmd_dir>/.fdl/schema-cache/<cmd>.json` with
/// mtime + binary hash metadata for staleness detection.
pub mod schema_cache;

/// First-run and reconfiguration wizard (`fdl setup`).
pub mod setup;

/// Daily update check against crates.io for `flodl-cli` and
/// project-pinned `flodl` / `flodl-hf`. Opt out via
/// `FDL_NO_UPDATE_CHECK=1` or the global config file.
pub mod update_check;

/// AI-skill bundles: packaging and installing the `/port` skill and
/// similar assistant integrations.
pub mod skill;

/// ANSI styling primitives and the `--ansi` / `--no-ansi` / `NO_COLOR`
/// resolution chain used by the help renderer and CLI output.
pub mod style;

/// Miscellaneous helpers shared by the other modules.
pub mod util;

/// Print a red-prefixed `error: <formatted>` line to stderr.
///
/// Takes standard `format!` arguments. Coloring follows the `--ansi` /
/// `--no-ansi` / `NO_COLOR` / `FORCE_COLOR` chain via
/// [`style::color_enabled`], so pipes stay plain automatically.
#[macro_export]
macro_rules! cli_error {
    ($($arg:tt)*) => {
        $crate::style::print_cli_error(format_args!($($arg)*))
    };
}

// ── Public API for binary authors ──────────────────────────────────────

/// Parse argv into `T`, intercepting `--fdl-schema` and `--help`.
pub use args::parse_or_schema;

/// Slice-based variant of [`parse_or_schema`] — parses from an explicit
/// `&[String]` rather than `std::env::args()`. Used by the `fdl` driver to
/// dispatch per-sub-command arg tails.
pub use args::parse_or_schema_from;

/// Trait implemented by `#[derive(FdlArgs)]` structs. Binary authors do
/// not typically implement this manually — the derive emits it.
pub use args::FdlArgsTrait;

/// Derive macro for `FdlArgs`. Generates argv parsing, `--fdl-schema`
/// emission, and `--help` rendering from a single struct definition.
pub use flodl_cli_macros::FdlArgs;

/// Schema types — mirror the JSON shape emitted by `--fdl-schema` and
/// consumed by the fdl driver.
pub use config::{ArgSpec, OptionSpec, Schema};

/// Re-exported dependencies the derive macro needs to reference by path.
/// Users should not depend on these directly — they are only stable as
/// an implementation detail of the derive.
#[doc(hidden)]
pub use serde_json;
