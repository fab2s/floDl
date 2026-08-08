//! Terminal colors and formatting.
//!
//! Default: auto-detect via stderr TTY + industry-standard env vars
//! (`NO_COLOR`, `FORCE_COLOR`). Explicit override via `--ansi`/`--no-ansi`
//! flags (set by `main` before any rendering). Falls back to plain text
//! when piped or redirected.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicU8, Ordering};

/// Explicit color preference. `Auto` means "pick based on TTY + env vars";
/// `Always` / `Never` force the answer regardless of environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

// Stored as u8: 0=Auto, 1=Always, 2=Never. Atomic so main's early set
// can't race with the first style call, even though in practice main
// sets the choice before any rendering happens.
const AUTO: u8 = 0;
const ALWAYS: u8 = 1;
const NEVER: u8 = 2;

static CHOICE: AtomicU8 = AtomicU8::new(AUTO);

/// Override the auto-detected choice. Called by `main` after parsing
/// `--ansi` / `--no-ansi`. Subsequent `color_enabled()` calls reflect the
/// override.
pub fn set_color_choice(choice: ColorChoice) {
    let v = match choice {
        ColorChoice::Auto => AUTO,
        ColorChoice::Always => ALWAYS,
        ColorChoice::Never => NEVER,
    };
    CHOICE.store(v, Ordering::Relaxed);
}

/// Current explicit choice, or `Auto` when none is set.
pub fn color_choice() -> ColorChoice {
    match CHOICE.load(Ordering::Relaxed) {
        ALWAYS => ColorChoice::Always,
        NEVER => ColorChoice::Never,
        _ => ColorChoice::Auto,
    }
}

/// Whether color output should be emitted right now.
///
/// Priority: explicit override (`--ansi`/`--no-ansi`) wins; then
/// `NO_COLOR` / `FORCE_COLOR` env vars (the industry conventions from
/// <https://no-color.org/>); finally fall back to `stderr().is_terminal()`.
/// Help output is written to stderr by the hand-rolled helps and to
/// stdout by the derive; both are checked so CI log viewers (which
/// render ANSI but have no stdout TTY) still get color when invoked
/// with `--ansi` or `FORCE_COLOR=1`.
pub fn color_enabled() -> bool {
    match color_choice() {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => {
            if env_flag_set("NO_COLOR") {
                return false;
            }
            if env_flag_set("FORCE_COLOR") {
                return true;
            }
            std::io::stderr().is_terminal()
        }
    }
}

/// An env var counts as "set" if it exists and is not empty. Matches
/// the convention used by `NO_COLOR` and `FORCE_COLOR` consumers
/// across the ecosystem.
fn env_flag_set(name: &str) -> bool {
    std::env::var_os(name)
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

// ANSI escape helpers. Return plain strings when color is disabled.

pub fn green(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[32m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn yellow(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[33m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn bold(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[1m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn dim(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[2m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn red(s: &str) -> String {
    if color_enabled() {
        format!("\x1b[31m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

/// Print a red-prefixed `error: <msg>` line to stderr.
///
/// Used via the [`crate::cli_error`] macro at call sites — the free
/// function is kept public so external tooling or tests can build the
/// same prefix without going through the macro.
pub fn print_cli_error(msg: impl std::fmt::Display) {
    eprintln!("{}: {msg}", red("error"));
}

/// Process-wide test mutex. Any test that mutates `CHOICE` or the
/// `NO_COLOR` / `FORCE_COLOR` env vars must take this lock, and so must
/// any test whose output is shaped by `color_enabled()` (e.g. the
/// overlay `render_*` tests, which would otherwise see stray ANSI keys
/// when stderr-in-docker is a TTY).
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::TEST_ENV_LOCK as LOCK;
    use super::*;

    fn reset() {
        set_color_choice(ColorChoice::Auto);
        // SAFETY: synchronised with LOCK; called only from tests.
        unsafe {
            std::env::remove_var("NO_COLOR");
            std::env::remove_var("FORCE_COLOR");
        }
    }

    #[test]
    fn explicit_always_forces_on() {
        let _g = LOCK.lock().unwrap();
        reset();
        set_color_choice(ColorChoice::Always);
        assert!(color_enabled());
        reset();
    }

    #[test]
    fn explicit_never_forces_off() {
        let _g = LOCK.lock().unwrap();
        reset();
        set_color_choice(ColorChoice::Never);
        assert!(!color_enabled());
        reset();
    }

    #[test]
    fn no_color_env_disables_in_auto_mode() {
        let _g = LOCK.lock().unwrap();
        reset();
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        assert!(!color_enabled());
        reset();
    }

    #[test]
    fn force_color_env_enables_in_auto_mode() {
        let _g = LOCK.lock().unwrap();
        reset();
        unsafe {
            std::env::set_var("FORCE_COLOR", "1");
        }
        assert!(color_enabled());
        reset();
    }

    #[test]
    fn no_color_beats_force_color_when_both_set() {
        // Industry precedent: NO_COLOR is documented as unconditional;
        // users who set both have bigger issues, but we pick the
        // safer default (no color).
        let _g = LOCK.lock().unwrap();
        reset();
        unsafe {
            std::env::set_var("NO_COLOR", "1");
            std::env::set_var("FORCE_COLOR", "1");
        }
        assert!(!color_enabled());
        reset();
    }

    #[test]
    fn explicit_override_beats_env_vars() {
        let _g = LOCK.lock().unwrap();
        reset();
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        set_color_choice(ColorChoice::Always);
        assert!(color_enabled());
        reset();
    }

    #[test]
    fn empty_env_var_treated_as_unset() {
        let _g = LOCK.lock().unwrap();
        reset();
        unsafe {
            std::env::set_var("NO_COLOR", "");
        }
        // Empty-string NO_COLOR must not disable color; the spec says
        // "any value other than the empty string". Auto-detect takes
        // over, which depends on stderr TTY.
        assert_eq!(color_choice(), ColorChoice::Auto);
        reset();
    }

    #[test]
    fn green_yellow_bold_dim_empty_when_disabled() {
        let _g = LOCK.lock().unwrap();
        reset();
        set_color_choice(ColorChoice::Never);
        assert_eq!(green("x"), "x");
        assert_eq!(yellow("x"), "x");
        assert_eq!(bold("x"), "x");
        assert_eq!(dim("x"), "x");
        reset();
    }

    #[test]
    fn green_wraps_with_ansi_when_enabled() {
        let _g = LOCK.lock().unwrap();
        reset();
        set_color_choice(ColorChoice::Always);
        assert_eq!(green("x"), "\x1b[32mx\x1b[0m");
        reset();
    }
}
