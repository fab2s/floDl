//! Argv parser and `FdlArgs` trait — the library side of the
//! `#[derive(FdlArgs)]` machinery.
//!
//! The derive macro in `flodl-cli-macros` emits an `impl FdlArgsTrait for
//! Cli` that delegates to the parser exposed here. Binary authors do not
//! import this module directly — they use `#[derive(FdlArgs)]` and
//! `parse_or_schema::<Cli>()` from the top-level `flodl_cli` crate.

pub mod parser;

use crate::config::Schema;

/// Trait implemented by `#[derive(FdlArgs)]`. Carries the metadata needed
/// to parse argv into a concrete type and to emit the `--fdl-schema` JSON.
///
/// The name is `FdlArgsTrait` to avoid colliding with the re-exported
/// derive macro `FdlArgs` (which lives in the derive-macro namespace).
/// Users never refer to this trait directly — the derive implements it.
pub trait FdlArgsTrait: Sized {
    /// Parse argv into `Self`. Uses `std::env::args()` by default.
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        match Self::try_parse_from(&args) {
            Ok(t) => t,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(2);
            }
        }
    }

    /// Parse from an explicit argv slice. First element is the program
    /// name (ignored), following elements are flags/values/positionals.
    fn try_parse_from(args: &[String]) -> Result<Self, String>;

    /// Return the JSON schema for this CLI shape.
    fn schema() -> Schema;

    /// Render `--help` to a string.
    fn render_help() -> String;

    /// Render `--help` for a specific argv path (program name first, then
    /// the tokens typed). The default ignores the path and returns
    /// [`Self::render_help`] — correct for a single struct, whose help is
    /// context-free.
    ///
    /// The enum derive overrides this to peel the leading subcommand token
    /// and render that subcommand's help, recursing for nested trees. An
    /// absent or unknown subcommand falls back to the root (command-list)
    /// help. This is why `bin train --help` shows train's flags rather than
    /// the top-level command list.
    fn render_help_path(argv: &[String]) -> String {
        let _ = argv;
        Self::render_help()
    }
}

/// Intercept `--fdl-schema` and `--help`, otherwise parse argv.
///
/// - `--fdl-schema` anywhere in argv: print the JSON schema to stdout, exit 0.
/// - `--help` / `-h` anywhere in argv: print help to stdout, exit 0.
/// - Otherwise: parse via `T::try_parse_from`. On parse error (missing
///   required positional, unknown flag, invalid value, ...) the error
///   message AND the rendered help are printed to stderr; the binary
///   exits with code 2. Showing help on error keeps `<bin>` (no args)
///   and `<bin> --help` consistent.
pub fn parse_or_schema<T: FdlArgsTrait>() -> T {
    let argv: Vec<String> = std::env::args().collect();
    parse_or_schema_from::<T>(&argv)
}

/// Slice-based variant of [`parse_or_schema`]. The first element is the
/// program name (displayed in help text), the rest are arguments.
///
/// Used by the `fdl` driver itself when dispatching to sub-commands: each
/// sub-command parses its own `args[2..]` tail without re-reading `env::args`.
pub fn parse_or_schema_from<T: FdlArgsTrait>(argv: &[String]) -> T {
    // Only intercept `--fdl-schema` / `--help` when they appear BEFORE the
    // first standalone `--`. A token after `--` is bound for the inner
    // program (e.g. `bin train -- --help` asks the inner for its help), so
    // scanning the whole argv would hijack it.
    let scan_end = argv.iter().position(|a| a == "--").unwrap_or(argv.len());
    let before = &argv[..scan_end];
    if before.iter().any(|a| a == "--fdl-schema") {
        let schema = T::schema();
        let json = serde_json::to_string_pretty(&schema)
            .expect("Schema serializes cleanly by construction");
        println!("{json}");
        std::process::exit(0);
    }
    if before.iter().any(|a| a == "--help" || a == "-h") {
        // `render_help_path` is context-aware: for a variant-shaped CLI,
        // `bin train --help` renders train's help, not the command list.
        // For a single struct it is identical to `render_help`.
        println!("{}", T::render_help_path(argv));
        std::process::exit(0);
    }
    match T::try_parse_from(argv) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            eprintln!();
            eprintln!("{}", T::render_help_path(argv));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod env_tests {
    //! End-to-end coverage of `#[option(env = "...")]` fallback.
    //!
    //! These tests mutate process-global `std::env` state, so they must
    //! hold [`ENV_LOCK`] for the duration of set/parse/drop. Without the
    //! lock, `cargo test`'s default parallel execution races on shared
    //! env var names and produces flaky failures in CI.

    use std::sync::{Mutex, MutexGuard};

    use crate::args::FdlArgsTrait;
    use crate::FdlArgs;

    /// Serializes every test in this module. Poison is ignored because a
    /// panicking test that leaves the lock poisoned still left the env
    /// clean (`EnvGuard::drop` runs during unwind).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn mk_args(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// Scoped env-var guard — `Drop` unsets on the way out so assertions
    /// that panic mid-test can't leak state into the next one.
    struct EnvGuard(&'static str);
    impl EnvGuard {
        fn set(name: &'static str, value: &str) -> Self {
            // SAFETY: caller holds `ENV_LOCK` for the duration of this
            // test, so no other test thread writes env concurrently.
            unsafe { std::env::set_var(name, value); }
            EnvGuard(name)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            unsafe { std::env::remove_var(self.0); }
        }
    }

    /// Port the server binds to.
    #[derive(FdlArgs, Debug)]
    struct OptArgs {
        /// Port override.
        #[option(env = "FDL_TEST_PORT")]
        port: Option<u16>,
    }

    #[test]
    fn env_fills_absent_option() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_PORT", "8080");
        let cli: OptArgs = OptArgs::try_parse_from(&mk_args(&["prog"])).unwrap();
        assert_eq!(cli.port, Some(8080));
    }

    #[test]
    fn argv_flag_beats_env() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_PORT", "8080");
        let cli: OptArgs =
            OptArgs::try_parse_from(&mk_args(&["prog", "--port", "9999"])).unwrap();
        assert_eq!(cli.port, Some(9999));
    }

    #[test]
    fn equals_form_beats_env() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_PORT", "8080");
        let cli: OptArgs =
            OptArgs::try_parse_from(&mk_args(&["prog", "--port=9999"])).unwrap();
        assert_eq!(cli.port, Some(9999));
    }

    #[test]
    fn empty_env_falls_through() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_PORT", "");
        let cli: OptArgs = OptArgs::try_parse_from(&mk_args(&["prog"])).unwrap();
        assert_eq!(cli.port, None);
    }

    /// Retry count — scalar with default + env fallback.
    #[derive(FdlArgs, Debug)]
    struct ScalarArgs {
        /// Retries.
        #[option(default = "3", env = "FDL_TEST_RETRIES")]
        retries: u32,
    }

    #[test]
    fn env_overrides_default_on_scalar() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_RETRIES", "7");
        let cli: ScalarArgs = ScalarArgs::try_parse_from(&mk_args(&["prog"])).unwrap();
        assert_eq!(cli.retries, 7);
    }

    #[test]
    fn argv_beats_env_beats_default_on_scalar() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_RETRIES", "7");
        let cli: ScalarArgs =
            ScalarArgs::try_parse_from(&mk_args(&["prog", "--retries", "42"])).unwrap();
        assert_eq!(cli.retries, 42);
    }

    /// Env-sourced values must still satisfy `choices`.
    #[derive(FdlArgs, Debug)]
    struct ChoiceArgs {
        /// Pick.
        #[option(choices = &["a", "b"], env = "FDL_TEST_CHOICE")]
        pick: Option<String>,
    }

    #[test]
    fn env_value_is_validated_against_choices() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_CHOICE", "z"); // not in choices
        let err = ChoiceArgs::try_parse_from(&mk_args(&["prog"])).unwrap_err();
        assert!(
            err.contains("invalid value") && err.contains("z") && err.contains("allowed:"),
            "env-sourced invalid choice should error like an argv one; got: {err}"
        );
    }

    #[test]
    fn env_valid_choice_accepted() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_CHOICE", "a");
        let cli: ChoiceArgs = ChoiceArgs::try_parse_from(&mk_args(&["prog"])).unwrap();
        assert_eq!(cli.pick.as_deref(), Some("a"));
    }

    /// Short-form presence should suppress env fallback.
    #[derive(FdlArgs, Debug)]
    struct ShortArgs {
        /// Port.
        #[option(short = 'p', env = "FDL_TEST_SHORT")]
        port: Option<u16>,
    }

    #[test]
    fn short_form_suppresses_env_fallback() {
        let _lock = env_lock();
        let _g = EnvGuard::set("FDL_TEST_SHORT", "8080");
        let cli: ShortArgs =
            ShortArgs::try_parse_from(&mk_args(&["prog", "-p", "9999"])).unwrap();
        assert_eq!(cli.port, Some(9999));
    }
}

#[cfg(test)]
mod enum_tests {
    //! Variant-shaped CLI: `#[derive(FdlArgs)]` on an enum of newtype
    //! variants dispatches a subcommand to the wrapped type.

    use crate::args::FdlArgsTrait;
    use crate::FdlArgs;

    fn mk_args(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// Train a model on a dataset.
    #[derive(FdlArgs, Debug)]
    struct TrainArgs {
        /// Number of epochs.
        #[option(short = 'n', default = "10")]
        epochs: u32,
    }

    /// Evaluate a trained model.
    #[derive(FdlArgs, Debug)]
    struct EvalArgs {
        /// Checkpoint to load.
        #[arg]
        checkpoint: String,
    }

    /// flodl demo CLI.
    #[derive(FdlArgs, Debug)]
    enum Cli {
        /// Train a letter model on a dataset
        Train(TrainArgs),
        /// Evaluate a trained letter model
        Eval(EvalArgs),
        /// Generate samples (renamed)
        #[command(name = "gen")]
        Generate(TrainArgs),
    }

    #[test]
    fn dispatches_to_variant_and_parses_its_flags() {
        let cli = Cli::try_parse_from(&mk_args(&["prog", "train", "--epochs", "5"])).unwrap();
        match cli {
            Cli::Train(a) => assert_eq!(a.epochs, 5),
            other => panic!("expected Train, got {other:?}"),
        }
    }

    #[test]
    fn variant_default_applies_when_flag_absent() {
        let cli = Cli::try_parse_from(&mk_args(&["prog", "train"])).unwrap();
        match cli {
            Cli::Train(a) => assert_eq!(a.epochs, 10),
            other => panic!("expected Train, got {other:?}"),
        }
    }

    #[test]
    fn dispatches_positional_to_variant() {
        let cli = Cli::try_parse_from(&mk_args(&["prog", "eval", "model.fdl"])).unwrap();
        match cli {
            Cli::Eval(a) => assert_eq!(a.checkpoint, "model.fdl"),
            other => panic!("expected Eval, got {other:?}"),
        }
    }

    #[test]
    fn command_name_override_is_honored() {
        let cli = Cli::try_parse_from(&mk_args(&["prog", "gen"])).unwrap();
        match cli {
            // Reading the wrapped value also confirms the tail parsed.
            Cli::Generate(a) => assert_eq!(a.epochs, 10),
            other => panic!("`gen` must map to Generate, got {other:?}"),
        }
        // And the original kebab name no longer dispatches.
        let err = Cli::try_parse_from(&mk_args(&["prog", "generate"])).unwrap_err();
        assert!(err.contains("unknown command"), "got: {err}");
    }

    #[test]
    fn missing_command_errors_with_list() {
        let err = Cli::try_parse_from(&mk_args(&["prog"])).unwrap_err();
        assert!(
            err.contains("missing command") && err.contains("train") && err.contains("eval"),
            "got: {err}"
        );
    }

    #[test]
    fn unknown_command_suggests_close_match() {
        let err = Cli::try_parse_from(&mk_args(&["prog", "trian"])).unwrap_err();
        assert!(
            err.contains("did you mean `train`"),
            "near-miss must suggest; got: {err}"
        );
    }

    #[test]
    fn unknown_command_far_miss_lists_options() {
        let err = Cli::try_parse_from(&mk_args(&["prog", "zzzzz"])).unwrap_err();
        assert!(
            err.contains("expected one of") && err.contains("train"),
            "far miss must list commands; got: {err}"
        );
    }

    #[test]
    fn schema_is_a_branch_with_described_children() {
        let s = Cli::schema();
        assert!(s.args.is_empty() && s.options.is_empty(), "root is a branch, not a leaf");
        assert_eq!(s.commands.len(), 3);
        assert_eq!(
            s.commands["train"].description.as_deref(),
            Some("Train a letter model on a dataset")
        );
        // Child carries the wrapped struct's own leaf shape.
        assert!(s.commands["train"].options.contains_key("epochs"));
        // Renamed variant keys by its override.
        assert!(s.commands.contains_key("gen"));
        // The whole tree must clear validation.
        crate::config::validate_schema(&s).expect("derived tree schema must validate");
    }

    #[test]
    fn root_help_lists_commands() {
        let help = Cli::render_help();
        assert!(help.contains("Commands"), "root help has a Commands section");
        assert!(help.contains("train") && help.contains("eval") && help.contains("gen"));
        assert!(
            help.contains("Train a letter model on a dataset"),
            "command descriptions come from variant docs; got:\n{help}"
        );
    }

    #[test]
    fn help_path_renders_the_subcommands_help() {
        // `prog train --help` → train's help (mentions its own flag), not
        // the command list.
        let help = Cli::render_help_path(&mk_args(&["prog", "train", "--help"]));
        assert!(help.contains("epochs"), "train help must show its flags; got:\n{help}");
        assert!(!help.contains("Commands"), "must not fall back to the command list");
    }

    #[test]
    fn help_path_falls_back_to_root_when_no_subcommand() {
        let help = Cli::render_help_path(&mk_args(&["prog"]));
        assert!(help.contains("Commands"), "bare --help shows the command list");
    }

    // ── Nested enums: arbitrary subcommand depth, for free ─────────────

    /// A variant that wraps *another* `FdlArgs` enum nests the tree one
    /// level deeper via plain tail-recursive delegation.
    #[derive(FdlArgs, Debug)]
    enum WordCli {
        /// Train group
        Train(TrainGroup),
        /// Evaluate
        Eval(EvalArgs),
    }

    #[derive(FdlArgs, Debug)]
    enum TrainGroup {
        /// Plain training
        Full(TrainArgs),
        /// Subscan sweep
        Subscan(TrainArgs),
    }

    #[test]
    fn nested_enum_dispatches_two_levels() {
        let cli =
            WordCli::try_parse_from(&mk_args(&["prog", "train", "subscan", "--epochs", "3"]))
                .unwrap();
        match cli {
            WordCli::Train(TrainGroup::Subscan(a)) => assert_eq!(a.epochs, 3),
            other => panic!("expected Train>Subscan, got {other:?}"),
        }
        // Exercise the other two paths (inner-group default + outer leaf).
        match WordCli::try_parse_from(&mk_args(&["prog", "train", "full"])).unwrap() {
            WordCli::Train(TrainGroup::Full(a)) => assert_eq!(a.epochs, 10),
            other => panic!("expected Train>Full, got {other:?}"),
        }
        match WordCli::try_parse_from(&mk_args(&["prog", "eval", "ckpt.fdl"])).unwrap() {
            WordCli::Eval(a) => assert_eq!(a.checkpoint, "ckpt.fdl"),
            other => panic!("expected Eval, got {other:?}"),
        }
    }

    #[test]
    fn nested_enum_schema_is_a_two_level_tree() {
        let s = WordCli::schema();
        let train = &s.commands["train"];
        assert!(train.options.is_empty(), "the train node is itself a branch");
        assert!(train.commands.contains_key("subscan"));
        assert!(train.commands["full"].options.contains_key("epochs"));
        crate::config::validate_schema(&s).expect("nested tree must validate");
    }

    #[test]
    fn nested_enum_help_drills_to_leaf() {
        // `prog train subscan --help` reaches the innermost struct's help.
        let help =
            WordCli::render_help_path(&mk_args(&["prog", "train", "subscan", "--help"]));
        assert!(help.contains("epochs"), "must reach the leaf struct help; got:\n{help}");
    }
}

