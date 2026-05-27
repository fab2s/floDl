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
    if argv.iter().any(|a| a == "--fdl-schema") {
        let schema = T::schema();
        let json = serde_json::to_string_pretty(&schema)
            .expect("Schema serializes cleanly by construction");
        println!("{json}");
        std::process::exit(0);
    }
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", T::render_help());
        std::process::exit(0);
    }
    match T::try_parse_from(argv) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            eprintln!();
            eprintln!("{}", T::render_help());
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

