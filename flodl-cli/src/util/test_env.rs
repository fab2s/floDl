//! One process-global mutex for tests that mutate environment variables.
//!
//! `std::env::{set_var, remove_var}` mutate global process state, so tests
//! touching the same variable must run serially. A per-module `static
//! Mutex` only serializes *within* that module — tests in different modules
//! that touch the SAME variable (e.g. `run.rs` setting `FDL_ENV` while
//! `cluster.rs` asserts it is unset) run concurrently and race. Every
//! env-mutating test locks THIS single mutex so the exclusion is
//! process-wide.
//!
//! Poison is recovered rather than propagated: when a test panics while
//! holding the lock, the environment is left in whatever state that test
//! set, but the mutex itself is still usable. Recovering lets the next
//! test re-establish its own preconditions (each locker resets the vars it
//! cares about) instead of turning one real failure into a cascade of
//! `PoisonError` unwrap panics that bury the true culprit.

use std::sync::{Mutex, MutexGuard};

static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Acquire the process-global environment lock, recovering from poison.
pub fn env_lock() -> MutexGuard<'static, ()> {
    ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}
