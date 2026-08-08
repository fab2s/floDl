//! Daily update check for `fdl` and the user-facing flodl crates.
//!
//! Probes crates.io once per day for newer versions of `flodl-cli`
//! (this binary), plus `flodl` and `flodl-hf` when found in the
//! current project's `Cargo.lock`. Caches results in
//! `<config_dir>/flodl/config.json` and prints one nudge line per
//! outdated crate at the end of the user's command.
//!
//! # Opt-out
//!
//! - `FDL_NO_UPDATE_CHECK=1` env var (wins over all else).
//! - `update_check.enabled = false` in `<config_dir>/flodl/config.json`.
//! - Auto-disabled when `CI=true` or running inside a Docker container
//!   (container filesystems are ephemeral so the cache resets every run,
//!   and CI runs already pin versions explicitly).
//!
//! # Network behaviour
//!
//! HTTP via `curl --max-time 2`, silent on every failure mode. The
//! probe never blocks the user's command output: it runs from a
//! `Guard` that fires at process exit
//! (Drop), after the user-visible work is done.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::util::system;

/// Throttle window between probes — once per day per machine.
const CHECK_INTERVAL_SECS: u64 = 24 * 3600;

/// Cap on each curl probe so a slow / hung crates.io can't block exit.
const HTTP_TIMEOUT_SECS: u64 = 2;

/// User-bumpable framework crates we probe when we find them in the
/// project's Cargo.lock. `flodl-cli` is checked separately (it's this
/// binary). `flodl-sys` and `flodl-cli-macros` are transitive deps that
/// ride along with `flodl` / `flodl-cli` upgrades — surfacing them adds
/// noise without action.
const FRAMEWORK_CRATES: &[&str] = &["flodl", "flodl-hf"];

// ---- Config schema --------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize)]
struct Config {
    #[serde(default)]
    update_check: UpdateCheck,
}

#[derive(Debug, Serialize, Deserialize)]
struct UpdateCheck {
    /// User-editable. `FDL_NO_UPDATE_CHECK=1` wins over this when set.
    #[serde(default = "default_enabled")]
    enabled: bool,
    /// Last successful probe, epoch seconds. fdl-managed.
    #[serde(default)]
    last_check: u64,
    /// Latest version per crate, as reported by crates.io. fdl-managed.
    #[serde(default)]
    latest_known: BTreeMap<String, String>,
    /// First-run disclosure banner shown once. fdl-managed.
    #[serde(default)]
    first_run_seen: bool,
}

impl Default for UpdateCheck {
    fn default() -> Self {
        Self {
            enabled: true,
            last_check: 0,
            latest_known: BTreeMap::new(),
            first_run_seen: false,
        }
    }
}

fn default_enabled() -> bool {
    true
}

// ---- Public surface -------------------------------------------------------

/// RAII guard whose `Drop` runs the update check at process exit.
///
/// Hold one in `main()`; the check fires after the user's command
/// output. Failures (network, parse, IO) are swallowed — the guard
/// never returns errors to the caller.
#[derive(Default)]
pub struct Guard;

impl Guard {
    pub fn new() -> Self {
        Self
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        run_silent();
    }
}

// ---- Orchestration --------------------------------------------------------

fn run_silent() {
    // Layered opt-outs: env var first (machine), CI second (automated
    // env), in-container third (ephemeral fs), config file last (user
    // policy).
    if env::var("FDL_NO_UPDATE_CHECK").is_ok() {
        return;
    }
    if env::var("CI").is_ok() {
        return;
    }
    if system::is_inside_docker() {
        return;
    }

    let cfg_path = match config_path() {
        Some(p) => p,
        None => return,
    };

    let mut cfg = load_config(&cfg_path);
    if !cfg.update_check.enabled {
        return;
    }

    // Decide what to probe and what to compare against.
    let project_versions = detect_project_crates();
    let mut crates_to_check: Vec<String> = vec!["flodl-cli".to_string()];
    crates_to_check.extend(project_versions.keys().cloned());

    // Refresh latest_known if 24h stale (or never probed).
    let now = unix_now();
    let mut probed = false;
    if now.saturating_sub(cfg.update_check.last_check) >= CHECK_INTERVAL_SECS
        && system::has_command("curl")
    {
        for name in &crates_to_check {
            if let Some(latest) = probe_crates_io(name) {
                cfg.update_check.latest_known.insert(name.clone(), latest);
            }
        }
        cfg.update_check.last_check = now;
        probed = true;
    }

    // First-run banner: print once, regardless of nudge presence.
    let mut printed_anything = false;
    if !cfg.update_check.first_run_seen {
        eprintln!();
        eprintln!("fdl checks for updates once a day.");
        eprintln!(
            "  Opt out: set `FDL_NO_UPDATE_CHECK=1` or edit `update_check.enabled`"
        );
        eprintln!("           in {}", cfg_path.display());
        cfg.update_check.first_run_seen = true;
        printed_anything = true;
    }

    // Compare and nudge per crate.
    let nudges = collect_nudges(
        &cfg.update_check.latest_known,
        env!("CARGO_PKG_VERSION"),
        &project_versions,
    );
    if !nudges.is_empty() {
        eprintln!();
        for n in &nudges {
            eprintln!("  {n}");
        }
        eprintln!();
        eprintln!("  Update fdl: `fdl install --check`");
        if nudges.iter().any(|n| !n.starts_with("flodl-cli ")) {
            eprintln!("  Update flodl deps in your project: `cargo update`");
        }
        printed_anything = true;
    }

    // Persist config if anything changed (probe ran or banner shown).
    if probed || printed_anything {
        let _ = save_config(&cfg_path, &cfg);
    }
}

// ---- Config IO ------------------------------------------------------------

fn config_path() -> Option<PathBuf> {
    let dir = config_dir()?;
    Some(dir.join("flodl").join("config.json"))
}

/// Platform-specific config root, mirroring the `dirs` crate's
/// `config_dir()` so we don't pull in an external crate for it.
fn config_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        env::var_os("HOME")
            .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
    } else if cfg!(target_os = "windows") {
        env::var_os("APPDATA").map(PathBuf::from)
    } else {
        // Linux / BSD / unknown unix: XDG.
        if let Some(xdg) = env::var_os("XDG_CONFIG_HOME") {
            let p = PathBuf::from(xdg);
            if p.is_absolute() {
                return Some(p);
            }
        }
        env::var_os("HOME").map(|h| PathBuf::from(h).join(".config"))
    }
}

fn load_config(path: &Path) -> Config {
    // Treat any failure (missing file, parse error, hand-edit broke
    // schema) as "use defaults". Never crash on user state.
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_config(path: &Path, cfg: &Config) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    fs::write(path, json).map_err(|e| e.to_string())
}

// ---- Project detection ----------------------------------------------------

/// Walk up from cwd looking for a `Cargo.lock`. Parse it for any of
/// [`FRAMEWORK_CRATES`] and return their resolved versions. Returns
/// empty map when we're not inside a cargo project, or when the
/// project doesn't depend on any of the user-facing flodl crates.
fn detect_project_crates() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();

    let cwd = match env::current_dir() {
        Ok(p) => p,
        Err(_) => return out,
    };

    let lock = match find_cargo_lock(&cwd) {
        Some(p) => p,
        None => return out,
    };

    let contents = match fs::read_to_string(&lock) {
        Ok(s) => s,
        Err(_) => return out,
    };

    // Cargo.lock is TOML with repeated `[[package]]` blocks. We do a
    // tiny line-based scan rather than pulling in a TOML crate.
    let mut current_name: Option<String> = None;
    let mut current_version: Option<String> = None;
    for line in contents.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            if let (Some(name), Some(version)) = (current_name.take(), current_version.take())
                && FRAMEWORK_CRATES.contains(&name.as_str()) {
                    out.insert(name, version);
                }
        } else if let Some(rest) = line.strip_prefix("name = ") {
            current_name = unquote(rest);
        } else if let Some(rest) = line.strip_prefix("version = ") {
            current_version = unquote(rest);
        }
    }
    // Trailing block.
    if let (Some(name), Some(version)) = (current_name, current_version)
        && FRAMEWORK_CRATES.contains(&name.as_str()) {
            out.insert(name, version);
        }

    out
}

fn unquote(s: &str) -> Option<String> {
    let s = s.trim();
    let s = s.strip_prefix('"')?.strip_suffix('"')?;
    Some(s.to_string())
}

fn find_cargo_lock(start: &Path) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        let candidate = dir.join("Cargo.lock");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
}

// ---- crates.io probe ------------------------------------------------------

#[derive(Deserialize)]
struct CratesIoResponse {
    #[serde(rename = "crate")]
    krate: CrateInfo,
}

#[derive(Deserialize)]
struct CrateInfo {
    max_stable_version: Option<String>,
    max_version: String,
}

/// Latest published version of `crate_name`, straight from the
/// crates.io API. Also `fdl init`'s source for the scaffold's flodl
/// pin. The `-A` header is load-bearing: crates.io's data-access
/// policy rejects unidentified clients, so a bare curl gets an error
/// body, not JSON.
pub(crate) fn probe_crates_io(crate_name: &str) -> Option<String> {
    let url = format!("https://crates.io/api/v1/crates/{crate_name}");
    let output = Command::new("curl")
        .arg("--silent")
        .arg("--fail")
        .arg("--max-time")
        .arg(HTTP_TIMEOUT_SECS.to_string())
        .arg("-A")
        .arg(concat!("flodl-cli/", env!("CARGO_PKG_VERSION")))
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let resp: CratesIoResponse = serde_json::from_slice(&output.stdout).ok()?;
    Some(resp.krate.max_stable_version.unwrap_or(resp.krate.max_version))
}

// ---- Comparison + nudges --------------------------------------------------

fn collect_nudges(
    latest_known: &BTreeMap<String, String>,
    self_version: &str,
    project_versions: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut out = Vec::new();

    if let Some(latest) = latest_known.get("flodl-cli")
        && semver_lt(self_version, latest) {
            out.push(format!(
                "flodl-cli {latest} is available (you have {self_version})"
            ));
        }

    for (name, current) in project_versions {
        if let Some(latest) = latest_known.get(name)
            && semver_lt(current, latest) {
                out.push(format!(
                    "{name} {latest} is available (your project pins {current})"
                ));
            }
    }

    out
}

/// Strict-less semver compare on the leading `MAJOR.MINOR.PATCH` parts.
/// Pre-release suffixes are dropped (we only nudge against stable
/// releases via `max_stable_version`).
fn semver_lt(a: &str, b: &str) -> bool {
    let parse = |s: &str| -> (u64, u64, u64) {
        let core = s.split(['-', '+']).next().unwrap_or(s);
        let mut it = core.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
        (
            it.next().unwrap_or(0),
            it.next().unwrap_or(0),
            it.next().unwrap_or(0),
        )
    };
    parse(a) < parse(b)
}

// ---- Misc -----------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_lt_basic() {
        assert!(semver_lt("0.5.2", "0.5.3"));
        assert!(semver_lt("0.5.2", "0.6.0"));
        assert!(semver_lt("0.5.2", "1.0.0"));
        assert!(!semver_lt("0.5.3", "0.5.3"));
        assert!(!semver_lt("0.5.4", "0.5.3"));
    }

    #[test]
    fn semver_lt_drops_prerelease_suffix() {
        // Pre-release suffix on either side gets stripped before
        // tuple compare. We only ever feed in stable versions, but be
        // defensive.
        assert!(!semver_lt("0.5.3", "0.5.3-alpha.1"));
        assert!(!semver_lt("0.5.3-rc.1", "0.5.3"));
    }

    #[test]
    fn semver_lt_handles_short_versions() {
        // "0.5" parses as (0,5,0).
        assert!(semver_lt("0.5", "0.5.1"));
        assert!(!semver_lt("0.5.0", "0.5"));
    }

    #[test]
    fn unquote_strips_double_quotes() {
        assert_eq!(unquote("\"foo\""), Some("foo".to_string()));
        assert_eq!(unquote("\"\""), Some("".to_string()));
        assert_eq!(unquote("foo"), None);
    }

    #[test]
    fn collect_nudges_self_outdated() {
        let mut latest = BTreeMap::new();
        latest.insert("flodl-cli".to_string(), "0.6.0".to_string());
        let nudges = collect_nudges(&latest, "0.5.2", &BTreeMap::new());
        assert_eq!(nudges.len(), 1);
        assert!(nudges[0].contains("0.6.0"));
        assert!(nudges[0].contains("0.5.2"));
    }

    #[test]
    fn collect_nudges_self_current_no_nudge() {
        let mut latest = BTreeMap::new();
        latest.insert("flodl-cli".to_string(), "0.5.2".to_string());
        let nudges = collect_nudges(&latest, "0.5.2", &BTreeMap::new());
        assert!(nudges.is_empty());
    }

    #[test]
    fn collect_nudges_project_dep_outdated() {
        let mut latest = BTreeMap::new();
        latest.insert("flodl-cli".to_string(), "0.5.2".to_string());
        latest.insert("flodl".to_string(), "0.6.0".to_string());
        let mut project = BTreeMap::new();
        project.insert("flodl".to_string(), "0.5.2".to_string());
        let nudges = collect_nudges(&latest, "0.5.2", &project);
        assert_eq!(nudges.len(), 1);
        assert!(nudges[0].starts_with("flodl 0.6.0"));
    }

    #[test]
    fn collect_nudges_no_latest_known_no_nudge() {
        // Empty latest_known (e.g. probe failed silently): no nudges.
        let nudges = collect_nudges(&BTreeMap::new(), "0.5.2", &BTreeMap::new());
        assert!(nudges.is_empty());
    }
}
