//! Verbosity-gated output for flodl.
//!
//! Normal and Verbose levels go to **stdout**.
//! Debug (`-vv`) and Trace (`-vvv`) go to **stderr** (unbuffered, visible
//! in Docker non-TTY mode where stdout is block-buffered).
//! Errors always go to **stderr** via standard `eprintln!` -- never gated.
//!
//! ```ignore
//! // Simple: just print (Normal level, suppressed by --quiet)
//! flodl::msg!("Epoch {}: loss={:.4}", epoch, loss);
//!
//! // With explicit level
//! flodl::msg!(@Verbose, "AllReduce: {:.1}ms", elapsed);
//! flodl::msg!(@Debug, "per-batch timing: {}ms", ms);
//! flodl::msg!(@Trace, "{:?}", my_tensor);
//! ```
//!
//! The level is a single global atomic -- no module filtering, no log targets.
//!
//! **Zero-code setup:** set the `FLODL_VERBOSITY` environment variable
//! (`0`..`4`, or `quiet`/`normal`/`verbose`/`debug`/`trace`).
//! The env var is read once on first access.
//!
//! ```bash
//! FLODL_VERBOSITY=verbose cargo run    # -v equivalent, no code needed
//! ```
//!
//! **Programmatic override:** [`set_verbosity`] takes precedence over the env var.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// Verbosity level for `flodl` output.
///
/// Higher levels include everything from lower levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Verbosity {
    /// `--quiet` -- errors only (via `eprintln!`), all macros suppressed.
    Quiet = 0,
    /// Default -- epoch summaries, progress, setup, milestones.
    Normal = 1,
    /// `-v` -- DDP sync, cadence changes, data loading detail.
    Verbose = 2,
    /// `-vv` -- heavy loop internals, per-batch timing.
    Debug = 3,
    /// `-vvv` -- extreme granularity.
    Trace = 4,
}

/// Environment variable used to set verbosity without code.
pub const ENV_VAR: &str = "FLODL_VERBOSITY";

/// Sentinel: 255 means "not yet initialized from env".
const UNINIT: u8 = 255;

static LEVEL: AtomicU8 = AtomicU8::new(UNINIT);

/// Read `FLODL_VERBOSITY` from the environment.
fn level_from_env() -> u8 {
    match std::env::var(ENV_VAR) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "quiet" => Verbosity::Quiet as u8,
            "1" | "normal" => Verbosity::Normal as u8,
            "2" | "verbose" => Verbosity::Verbose as u8,
            "3" | "debug" => Verbosity::Debug as u8,
            "4" | "trace" => Verbosity::Trace as u8,
            _ => Verbosity::Normal as u8,
        },
        Err(_) => Verbosity::Normal as u8,
    }
}

/// Load the level, initializing from env on first access.
#[inline]
fn load_or_init() -> u8 {
    let v = LEVEL.load(Ordering::Relaxed);
    if v != UNINIT {
        return v;
    }
    let from_env = level_from_env();
    // CAS: if still UNINIT, set to env value. If someone called
    // set_verbosity() concurrently, their value wins.
    let _ = LEVEL.compare_exchange(UNINIT, from_env, Ordering::Relaxed, Ordering::Relaxed);
    LEVEL.load(Ordering::Relaxed)
}

/// Set the global verbosity level.
///
/// Overrides the `FLODL_VERBOSITY` env var. Typically called once at startup.
pub fn set_verbosity(level: Verbosity) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Get the current global verbosity level.
pub fn verbosity() -> Verbosity {
    match load_or_init() {
        0 => Verbosity::Quiet,
        2 => Verbosity::Verbose,
        3 => Verbosity::Debug,
        4 => Verbosity::Trace,
        _ => Verbosity::Normal,
    }
}

/// Check whether a given verbosity level is enabled.
#[inline]
pub fn enabled(level: Verbosity) -> bool {
    load_or_init() >= level as u8
}

/// Optional cluster-wide host label, set once at startup.
static NODE_LABEL: OnceLock<String> = OnceLock::new();

thread_local! {
    /// Per-thread log prefix, computed once at [`set_thread_device`] time.
    ///
    /// Macros prepend this verbatim; the hot path is one thread-local read
    /// and no formatting work.
    pub static THREAD_PREFIX: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Build the prefix string from host label, local device, and optional global rank.
///
/// Pure helper, factored out so the four-case shape is testable in isolation
/// (the global `NODE_LABEL` is a `OnceLock`, so it can't be re-set per test).
fn build_thread_prefix(host: &str, local_dev: u8, global_rank: Option<usize>) -> String {
    match (host.is_empty(), global_rank) {
        (true, None) => String::new(),
        (true, Some(r)) => format!("[r{r}] "),
        (false, None) => format!("[{host}:{local_dev}] "),
        (false, Some(r)) => format!("[{host}:{local_dev}:r{r}] "),
    }
}

/// Register the cluster-wide host label (e.g. `"master-host"`, `"worker-host"`).
///
/// MUST run before worker threads spawn. The label is read by
/// [`set_thread_device`] when computing each worker's prefix. Idempotent:
/// later calls are silently ignored (single-source-of-truth invariant).
pub fn set_node_label(host: impl Into<String>) {
    let _ = NODE_LABEL.set(host.into());
}

/// Compute and store this thread's log prefix.
///
/// Called once per worker thread on entry. The resulting prefix is stored in
/// the [`THREAD_PREFIX`] thread-local and reused verbatim by every log macro
/// on this thread — no per-call formatting.
///
/// Prefix shape:
/// - No node label, no rank → empty (single-host, today's behavior)
/// - No node label, rank `r` → `"[rN] "`
/// - Node label `H`, dev `D`, no rank → `"[H:D] "`
/// - Node label `H`, dev `D`, rank `r` → `"[H:D:rN] "`
pub fn set_thread_device(local_dev: u8, global_rank: Option<usize>) {
    let host = NODE_LABEL.get().map(String::as_str).unwrap_or("");
    let prefix = build_thread_prefix(host, local_dev, global_rank);
    THREAD_PREFIX.with(|p| *p.borrow_mut() = prefix);
}

/// Print to stdout, gated by verbosity level.
///
/// ```ignore
/// // Default (Normal): suppressed by --quiet, shown otherwise
/// flodl::msg!("Epoch {}: loss={:.4}", epoch, loss);
///
/// // With explicit level (@ prefix)
/// flodl::msg!(@Verbose, "AllReduce: {:.1}ms", elapsed);
/// flodl::msg!(@Debug, "per-batch: {}ms", ms);
/// flodl::msg!(@Trace, "{:?}", tensor);
/// ```
#[macro_export]
macro_rules! msg {
    // With explicit level: flodl::msg!(@Verbose, "msg", arg)
    (@$level:expr, $($arg:tt)+) => {
        if $crate::log::enabled($level) {
            $crate::log::THREAD_PREFIX.with(|p|
                println!("{}{}", p.borrow().as_str(), format_args!($($arg)+))
            )
        }
    };
    // Default (Normal): flodl::msg!("msg", arg)
    ($($arg:tt)+) => {
        if $crate::log::enabled($crate::log::Verbosity::Normal) {
            $crate::log::THREAD_PREFIX.with(|p|
                println!("{}{}", p.borrow().as_str(), format_args!($($arg)+))
            )
        }
    };
}

/// Prints to stdout at `-v` (Verbose) and above.
///
/// Internal shortcut. Equivalent to `flodl::msg!(@Verbose, ...)`.
#[macro_export]
macro_rules! verbose {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Verbosity::Verbose) {
            $crate::log::THREAD_PREFIX.with(|p|
                println!("{}{}", p.borrow().as_str(), format_args!($($arg)*))
            )
        }
    };
}

/// Prints to stderr at `-vv` (Debug) and above.
///
/// Uses stderr so output is unbuffered and visible immediately in Docker
/// non-TTY mode (stdout is block-buffered there).
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Verbosity::Debug) {
            $crate::log::THREAD_PREFIX.with(|p|
                eprintln!("{}{}", p.borrow().as_str(), format_args!($($arg)*))
            )
        }
    };
}

/// Prints to stderr at `-vvv` (Trace) and above.
///
/// Uses stderr so output is unbuffered and visible immediately in Docker
/// non-TTY mode (stdout is block-buffered there).
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        if $crate::log::enabled($crate::log::Verbosity::Trace) {
            $crate::log::THREAD_PREFIX.with(|p|
                eprintln!("{}{}", p.borrow().as_str(), format_args!($($arg)*))
            )
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serializes tests that mutate the global `LEVEL` atomic; without this,
    // parallel runs race and observe each other's writes.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_verbosity_ordering() {
        assert!(Verbosity::Quiet < Verbosity::Normal);
        assert!(Verbosity::Normal < Verbosity::Verbose);
        assert!(Verbosity::Verbose < Verbosity::Debug);
        assert!(Verbosity::Debug < Verbosity::Trace);
    }

    #[test]
    fn test_set_and_get() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let orig = verbosity();

        set_verbosity(Verbosity::Debug);
        assert_eq!(verbosity(), Verbosity::Debug);

        set_verbosity(Verbosity::Trace);
        assert_eq!(verbosity(), Verbosity::Trace);

        set_verbosity(Verbosity::Quiet);
        assert_eq!(verbosity(), Verbosity::Quiet);

        set_verbosity(Verbosity::Normal);
        assert_eq!(verbosity(), Verbosity::Normal);

        set_verbosity(orig);
    }

    #[test]
    fn test_enabled() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let orig = verbosity();

        set_verbosity(Verbosity::Verbose);
        assert!(enabled(Verbosity::Normal));
        assert!(enabled(Verbosity::Verbose));
        assert!(!enabled(Verbosity::Debug));
        assert!(!enabled(Verbosity::Trace));

        set_verbosity(Verbosity::Trace);
        assert!(enabled(Verbosity::Normal));
        assert!(enabled(Verbosity::Verbose));
        assert!(enabled(Verbosity::Debug));
        assert!(enabled(Verbosity::Trace));

        set_verbosity(Verbosity::Quiet);
        assert!(!enabled(Verbosity::Normal));
        assert!(!enabled(Verbosity::Verbose));

        set_verbosity(orig);
    }

    #[test]
    fn test_msg_with_level() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let orig = verbosity();

        set_verbosity(Verbosity::Normal);
        // Verify the macro compiles with both forms
        crate::msg!("normal message");
        crate::msg!(@Verbosity::Verbose, "verbose message");
        crate::msg!(@Verbosity::Debug, "debug: {}", 42);

        set_verbosity(orig);
    }

    #[test]
    fn test_build_thread_prefix() {
        // Single-host mode: no node label, no rank → empty (today's behavior).
        assert_eq!(build_thread_prefix("", 0, None), "");
        assert_eq!(build_thread_prefix("", 3, None), "");

        // No node label, rank only → "[rN] ".
        assert_eq!(build_thread_prefix("", 0, Some(2)), "[r2] ");

        // Node label, dev only → "[host:dev] ".
        assert_eq!(build_thread_prefix("node-a", 0, None), "[node-a:0] ");

        // Full cluster mode: node label, dev, rank → "[host:dev:rN] ".
        assert_eq!(
            build_thread_prefix("node-b", 1, Some(2)),
            "[node-b:1:r2] "
        );
        assert_eq!(build_thread_prefix("node-a", 0, Some(0)), "[node-a:0:r0] ");
    }
}
