//! Post-init / post-setup helper that offers to install `fdl` globally
//! (`~/.local/bin/fdl`) so the user can drop the `./` prefix in
//! subsequent invocations.
//!
//! Deliberately conservative: it skips itself when the current binary
//! already lives at the target path, when a different `fdl` already
//! exists there, or when no `HOME` is available. Users who decline
//! the prompt get a one-line reminder.

use std::path::PathBuf;
use std::process::Command;

use crate::util::prompt;

/// Offer to install the running `fdl` binary into `~/.local/bin/fdl`.
///
/// Returns silently (no prompt, no output) when the offer is not
/// applicable — see the module-level doc for the skip conditions.
pub fn offer_global_install() {
    let Some(home_os) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        return;
    };
    let target = PathBuf::from(home_os).join(".local/bin/fdl");

    let current = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };

    // Already running from the target path -> already installed.
    let current_canon = current.canonicalize().unwrap_or_else(|_| current.clone());
    let target_canon = target.canonicalize().unwrap_or_else(|_| target.clone());
    if current_canon == target_canon {
        return;
    }

    // Something else already occupies ~/.local/bin/fdl. Don't clobber it;
    // the user can `./fdl install` explicitly if they want to override.
    if target.exists() {
        return;
    }

    println!();
    let msg = format!("Install fdl globally to {}?", target.display());
    if !prompt::ask_yn(&msg, true) {
        println!("  (later: ./fdl install)");
        return;
    }

    let status = Command::new(&current).arg("install").status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("fdl install did not complete; rerun manually with `./fdl install`.");
        }
    }
}
