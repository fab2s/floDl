//! `fdl schema` — inspect, clear, and refresh cached `--fdl-schema`
//! outputs across the project.
//!
//! Caches live at `<cmd_dir>/.fdl/schema-cache/<cmd-name>.json` (see
//! [`crate::schema_cache`] for the per-command mechanics). This module
//! walks the project tree to find every cache, reports staleness, and
//! exposes clear / refresh operations. It's intentionally a filesystem
//! scan rather than a command-graph walk: any layout that ends up
//! writing a cache file gets discovered, regardless of how the
//! `commands:` tree is shaped.

use std::fs;
use std::path::{Path, PathBuf};

use crate::schema_cache;

/// Directories that never contain valid schema caches — skip them to
/// keep scans fast on large repos.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "libtorch",
    "runs",
    ".cargo",
    "site",
    "docs",
    ".claude",
];

/// One cached schema discovered on disk.
pub struct CacheEntry {
    /// Command name (filename stem).
    pub cmd_name: String,
    /// Directory that holds the command's `fdl.yml` and `.fdl/`.
    pub cmd_dir: PathBuf,
    /// Full path to the cache JSON file.
    pub cache_path: PathBuf,
    /// Path to the command's primary config file (the mtime anchor).
    /// `None` when no `fdl.yml` / `fdl.yaml` / `fdl.json` was found — the
    /// cache exists but has no reference to compare against, which is
    /// reported as a dedicated status.
    pub source_config: Option<PathBuf>,
}

/// Freshness of a cache file relative to its source `fdl.yml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// Cache file's mtime is newer than the source config.
    Fresh,
    /// Source config has been modified since the cache was written.
    Stale,
    /// Cache exists but no source config was found alongside it.
    Orphan,
}

impl CacheEntry {
    pub fn status(&self) -> CacheStatus {
        match &self.source_config {
            Some(src) => {
                if schema_cache::is_stale(&self.cache_path, std::slice::from_ref(src)) {
                    CacheStatus::Stale
                } else {
                    CacheStatus::Fresh
                }
            }
            None => CacheStatus::Orphan,
        }
    }
}

/// Scan `project_root` recursively for `.fdl/schema-cache/*.json` files.
/// Skips common noise dirs (`SKIP_DIRS`). Results are sorted by cache
/// path for stable `fdl schema list` output.
pub fn discover_caches(project_root: &Path) -> Vec<CacheEntry> {
    let mut out = Vec::new();
    walk(project_root, &mut out);
    out.sort_by(|a, b| a.cache_path.cmp(&b.cache_path));
    out
}

fn walk(dir: &Path, out: &mut Vec<CacheEntry>) {
    if let Some(name) = dir.file_name().and_then(|n| n.to_str())
        && SKIP_DIRS.contains(&name)
    {
        return;
    }

    let cache_dir = dir.join(".fdl").join("schema-cache");
    if cache_dir.is_dir()
        && let Ok(entries) = fs::read_dir(&cache_dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            out.push(CacheEntry {
                cmd_name: stem.to_string(),
                cmd_dir: dir.to_path_buf(),
                cache_path: path,
                source_config: find_source_config(dir),
            });
        }
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            }
        }
    }
}

/// Pick the primary config file (`fdl.yml` > `fdl.yaml` > `fdl.json`)
/// that sits next to a cache's command dir. Matches the preference
/// order used by `crate::overlay::EXTENSIONS` via
/// `crate::config::CONFIG_NAMES`.
fn find_source_config(cmd_dir: &Path) -> Option<PathBuf> {
    for name in &["fdl.yml", "fdl.yaml", "fdl.json"] {
        let p = cmd_dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Delete cache files. `filter` restricts the operation to a single
/// command name; `None` clears all discovered caches. Empty parent
/// `.fdl/schema-cache/` and `.fdl/` dirs are removed when nothing else
/// lives inside them. Returns the list of removed cache paths.
pub fn clear_caches(project_root: &Path, filter: Option<&str>) -> Result<Vec<PathBuf>, String> {
    let caches = discover_caches(project_root);
    let mut removed = Vec::new();
    let mut touched_dirs: Vec<PathBuf> = Vec::new();

    for entry in &caches {
        if let Some(name) = filter
            && entry.cmd_name != name
        {
            continue;
        }
        fs::remove_file(&entry.cache_path)
            .map_err(|e| format!("cannot remove {}: {e}", entry.cache_path.display()))?;
        removed.push(entry.cache_path.clone());
        touched_dirs.push(entry.cmd_dir.clone());
    }

    // Prune now-empty parent dirs. Best-effort: ignore errors, since
    // other processes could have written files in the meantime.
    touched_dirs.sort();
    touched_dirs.dedup();
    for d in touched_dirs {
        let cache_dir = d.join(".fdl").join("schema-cache");
        if is_empty_dir(&cache_dir) {
            let _ = fs::remove_dir(&cache_dir);
        }
        let fdl_dir = d.join(".fdl");
        if is_empty_dir(&fdl_dir) {
            let _ = fs::remove_dir(&fdl_dir);
        }
    }

    Ok(removed)
}

fn is_empty_dir(p: &Path) -> bool {
    p.is_dir()
        && fs::read_dir(p)
            .map(|mut it| it.next().is_none())
            .unwrap_or(false)
}

/// Probe each cached command's entry and rewrite its cache file.
/// `filter` scopes to a single command name. Returns per-cache results
/// so the caller can print a summary.
///
/// Cargo entries that haven't been built will surface their probe
/// failure — the user is expected to build first, same contract as the
/// per-command `fdl <cmd> --refresh-schema` flag.
pub fn refresh_caches(
    project_root: &Path,
    filter: Option<&str>,
) -> Result<Vec<RefreshResult>, String> {
    let caches = discover_caches(project_root);
    let mut results = Vec::new();

    for entry in &caches {
        if let Some(name) = filter
            && entry.cmd_name != name
        {
            continue;
        }

        let outcome = refresh_one(entry);
        results.push(RefreshResult {
            cmd_name: entry.cmd_name.clone(),
            cache_path: entry.cache_path.clone(),
            outcome,
        });
    }

    Ok(results)
}

pub struct RefreshResult {
    pub cmd_name: String,
    pub cache_path: PathBuf,
    pub outcome: Result<(), String>,
}

fn refresh_one(entry: &CacheEntry) -> Result<(), String> {
    let config = crate::config::load_command(&entry.cmd_dir)?;
    let entry_cmd = config.entry.as_deref().ok_or_else(|| {
        format!(
            "no `entry:` declared in {}/fdl.yml",
            entry.cmd_dir.display()
        )
    })?;
    let schema = schema_cache::probe(entry_cmd, &entry.cmd_dir, config.docker.as_deref())?;
    schema_cache::write_cache(&entry.cache_path, &schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "fdl-schema-test-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_cache(dir: &Path, cmd_name: &str, json: &str) -> PathBuf {
        let cache_dir = dir.join(".fdl").join("schema-cache");
        fs::create_dir_all(&cache_dir).unwrap();
        let path = cache_dir.join(format!("{cmd_name}.json"));
        fs::write(&path, json).unwrap();
        path
    }

    const VALID_SCHEMA_JSON: &str = r#"{"options":{},"args":[]}"#;

    #[test]
    fn discover_finds_single_cache() {
        let tmp = TempDir::new();
        let train = tmp.0.join("train");
        fs::create_dir_all(&train).unwrap();
        fs::write(train.join("fdl.yml"), "entry: echo\n").unwrap();
        write_cache(&train, "train", VALID_SCHEMA_JSON);

        let caches = discover_caches(&tmp.0);
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].cmd_name, "train");
        assert_eq!(caches[0].cmd_dir, train);
        assert!(caches[0].source_config.is_some());
    }

    #[test]
    fn discover_finds_multiple_nested_caches() {
        let tmp = TempDir::new();
        for name in &["train", "bench", "eval"] {
            let d = tmp.0.join(name);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("fdl.yml"), "entry: echo\n").unwrap();
            write_cache(&d, name, VALID_SCHEMA_JSON);
        }
        let caches = discover_caches(&tmp.0);
        let names: Vec<_> = caches.iter().map(|c| c.cmd_name.as_str()).collect();
        assert_eq!(names, vec!["bench", "eval", "train"]); // sorted by path
    }

    #[test]
    fn discover_skips_target_and_git() {
        let tmp = TempDir::new();
        // Decoy caches under skipped dirs.
        for noise in &["target", ".git", "node_modules"] {
            let d = tmp.0.join(noise);
            fs::create_dir_all(&d).unwrap();
            write_cache(&d, "decoy", VALID_SCHEMA_JSON);
        }
        // Real cache.
        let train = tmp.0.join("train");
        fs::create_dir_all(&train).unwrap();
        fs::write(train.join("fdl.yml"), "entry: echo\n").unwrap();
        write_cache(&train, "train", VALID_SCHEMA_JSON);

        let caches = discover_caches(&tmp.0);
        assert_eq!(caches.len(), 1);
        assert_eq!(caches[0].cmd_name, "train");
    }

    #[test]
    fn status_fresh_when_cache_newer_than_source() {
        let tmp = TempDir::new();
        let train = tmp.0.join("train");
        fs::create_dir_all(&train).unwrap();
        fs::write(train.join("fdl.yml"), "entry: echo\n").unwrap();
        // Sleep briefly then write cache so its mtime is strictly newer.
        std::thread::sleep(std::time::Duration::from_millis(10));
        write_cache(&train, "train", VALID_SCHEMA_JSON);
        let caches = discover_caches(&tmp.0);
        assert_eq!(caches[0].status(), CacheStatus::Fresh);
    }

    #[test]
    fn status_stale_when_source_newer_than_cache() {
        let tmp = TempDir::new();
        let train = tmp.0.join("train");
        fs::create_dir_all(&train).unwrap();
        write_cache(&train, "train", VALID_SCHEMA_JSON);
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(train.join("fdl.yml"), "entry: echo\n").unwrap();
        let caches = discover_caches(&tmp.0);
        assert_eq!(caches[0].status(), CacheStatus::Stale);
    }

    #[test]
    fn status_orphan_when_no_source_config() {
        let tmp = TempDir::new();
        let dir = tmp.0.join("lonely");
        fs::create_dir_all(&dir).unwrap();
        write_cache(&dir, "lonely", VALID_SCHEMA_JSON);
        let caches = discover_caches(&tmp.0);
        assert_eq!(caches[0].status(), CacheStatus::Orphan);
    }

    #[test]
    fn clear_removes_all_caches_when_no_filter() {
        let tmp = TempDir::new();
        for name in &["a", "b"] {
            let d = tmp.0.join(name);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("fdl.yml"), "entry: echo\n").unwrap();
            write_cache(&d, name, VALID_SCHEMA_JSON);
        }
        let removed = clear_caches(&tmp.0, None).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(discover_caches(&tmp.0).is_empty());
        // Empty `.fdl/` parents cleaned up too.
        assert!(!tmp.0.join("a").join(".fdl").exists());
        assert!(!tmp.0.join("b").join(".fdl").exists());
    }

    #[test]
    fn clear_respects_filter() {
        let tmp = TempDir::new();
        for name in &["keep", "drop"] {
            let d = tmp.0.join(name);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("fdl.yml"), "entry: echo\n").unwrap();
            write_cache(&d, name, VALID_SCHEMA_JSON);
        }
        let removed = clear_caches(&tmp.0, Some("drop")).unwrap();
        assert_eq!(removed.len(), 1);
        assert!(removed[0].to_string_lossy().contains("drop"));
        let remaining: Vec<_> = discover_caches(&tmp.0)
            .into_iter()
            .map(|c| c.cmd_name)
            .collect();
        assert_eq!(remaining, vec!["keep".to_string()]);
    }

    #[test]
    fn clear_filter_matching_nothing_is_a_noop() {
        let tmp = TempDir::new();
        let d = tmp.0.join("a");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("fdl.yml"), "entry: echo\n").unwrap();
        write_cache(&d, "a", VALID_SCHEMA_JSON);
        let removed = clear_caches(&tmp.0, Some("nonexistent")).unwrap();
        assert!(removed.is_empty());
        assert_eq!(discover_caches(&tmp.0).len(), 1);
    }
}
