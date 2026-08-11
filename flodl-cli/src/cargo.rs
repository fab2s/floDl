//! `fdl cargo` engine: discover, size, and clear cargo's on-disk footprint.
//!
//! Two tiers, split along the reclaim-cost axis:
//!
//! - **Target** (compiled artifacts): `target/`, `.target*` (per-container
//!   target dirs such as `.target-docsrs`), and the workspace-excluded
//!   crates' own `target/` dirs. Reclaiming costs a recompute, no network.
//! - **Cache** (registry + git caches): `.cargo-cache*`, `.cargo-git*`.
//!   Reclaiming costs a re-download, which needs network.
//!
//! Clearing empties a root's CONTENTS and always keeps the root directory
//! itself: several of these dirs are docker bind-mount sources
//! (`.cargo-cache*`, `.cargo-git*`, `.target-docsrs`), and a removed
//! source gets recreated root-owned by docker on the next compose run,
//! breaking container builds. Contents-only is the uniform safe rule for
//! the rest too.
//!
//! Both sizing and clearing are error-tolerant by construction: an
//! unreadable or undeletable entry (e.g. root-owned files written by the
//! docs.rs container) is counted/reported, never fatal.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Which of the two footprint tiers a root belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Compiled artifacts: reclaim = recompute, no network needed.
    Target,
    /// Registry/git caches: reclaim = re-download, needs network.
    Cache,
}

impl Tier {
    /// The sub-command name (`fdl cargo <name>`).
    pub fn name(self) -> &'static str {
        match self {
            Tier::Target => "target",
            Tier::Cache => "cache",
        }
    }

    /// Human heading for the report section.
    pub fn heading(self) -> &'static str {
        match self {
            Tier::Target => "compiled artifacts",
            Tier::Cache => "registry caches",
        }
    }

    /// What getting the bytes back again costs.
    pub fn reclaim(self) -> &'static str {
        match self {
            Tier::Target => "recompute, no network needed",
            Tier::Cache => "re-download, needs network",
        }
    }
}

/// One discovered footprint root.
#[derive(Debug)]
pub struct DiskRoot {
    /// Absolute path.
    pub path: PathBuf,
    /// Project-root-relative display label (`target`, `ddp-bench/target`).
    pub label: String,
    pub tier: Tier,
}

/// Discover footprint roots under a project root, by naming convention:
/// `target` / `.target*` and `.cargo-cache*` / `.cargo-git*` at the top
/// level, plus `<dir>/target` for any top-level crate dir carrying its
/// own `Cargo.toml` (workspace members share the root `target/`, so only
/// workspace-excluded crates ever match). Sorted tier-first, then by
/// label, so report order is stable.
pub fn discover(root: &Path) -> Vec<DiskRoot> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        // `is_dir` follows symlinks on purpose: a `target` symlinked
        // onto a bigger disk is a real layout, and lstat would drop it
        // from the report while the bytes stayed. Clearing follows it
        // too, which is what the user asking for `target` meant; the
        // link itself survives, like any other root.
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let tier = if name == "target" || name.starts_with(".target") {
            Some(Tier::Target)
        } else if name.starts_with(".cargo-cache") || name.starts_with(".cargo-git") {
            Some(Tier::Cache)
        } else {
            None
        };
        if let Some(tier) = tier {
            out.push(DiskRoot {
                path: entry.path(),
                label: name,
                tier,
            });
            continue;
        }
        if !name.starts_with('.') && entry.path().join("Cargo.toml").is_file() {
            let target = entry.path().join("target");
            if target.is_dir() {
                out.push(DiskRoot {
                    path: target,
                    label: format!("{name}/target"),
                    tier: Tier::Target,
                });
            }
        }
    }
    out.sort_by(|a, b| (a.tier, &a.label).cmp(&(b.tier, &b.label)));
    out
}

/// The workspace root, which is where cargo puts `target/` and where
/// the container bind-mount caches live.
///
/// This is a different question from "which project am I in", and the
/// difference bites: [`crate::context::Context::resolve`] accepts any
/// `Cargo.toml` mentioning flodl as a project signal, so starting from
/// `flodl/src` it stops at the member crate and the footprint of a
/// 90 GiB checkout reads as empty. Walk up from `start` through the
/// contiguous cargo tree and take the outermost manifest declaring a
/// `[workspace]`; with none (a standalone single-crate project) `start`
/// is already right.
pub fn workspace_root(start: &Path) -> PathBuf {
    let mut best = start.to_path_buf();
    let mut dir = start;
    loop {
        let manifest = dir.join("Cargo.toml");
        if !manifest.is_file() {
            break;
        }
        if fs::read_to_string(&manifest)
            .map(|s| s.lines().any(|l| l.trim_end() == "[workspace]"))
            .unwrap_or(false)
        {
            best = dir.to_path_buf();
        }
        match dir.parent() {
            Some(p) => dir = p,
            None => break,
        }
    }
    best
}

/// Recursive disk usage of one root.
#[derive(Debug, Default)]
pub struct Usage {
    /// Bytes on disk (block-based on unix, apparent size elsewhere).
    pub bytes: u64,
    /// File count (regular files + symlinks, not directories).
    pub files: u64,
    /// Entries that could not be read (permissions, races). Their size
    /// is missing from `bytes`.
    pub unreadable: u64,
}

/// Measure a root. Never fails: unreadable entries are counted in
/// [`Usage::unreadable`] instead.
pub fn usage(path: &Path) -> Usage {
    let mut u = Usage::default();
    walk_usage(path, &mut u);
    u
}

fn walk_usage(dir: &Path, u: &mut Usage) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => {
            u.unreadable += 1;
            return;
        }
    };
    for entry in entries {
        let Ok(entry) = entry else {
            u.unreadable += 1;
            continue;
        };
        // DirEntry::metadata does not traverse symlinks, so a symlink is
        // sized as the link itself and never followed (a followed link
        // could double-count or escape the root).
        let Ok(meta) = entry.metadata() else {
            u.unreadable += 1;
            continue;
        };
        u.bytes += disk_size(&meta);
        if meta.is_dir() {
            walk_usage(&entry.path(), u);
        } else {
            u.files += 1;
        }
    }
}

/// Physical size on unix (`st_blocks`, what `du` reports and what a
/// delete actually frees); apparent size elsewhere.
#[cfg(unix)]
fn disk_size(meta: &fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    meta.blocks() * 512
}

#[cfg(not(unix))]
fn disk_size(meta: &fs::Metadata) -> u64 {
    meta.len()
}

/// One entry that survived a clear attempt.
#[derive(Debug)]
pub struct Skipped {
    pub path: PathBuf,
    pub error: String,
    /// Permission-denied failures get a dedicated flag so the CLI can
    /// print the root-owned hint (docs.rs container artifacts) once.
    pub permission: bool,
}

/// Outcome of clearing one root's contents.
#[derive(Debug, Default)]
pub struct ClearOutcome {
    /// Bytes actually freed (successfully removed entries only).
    pub freed: u64,
    /// Regular files + symlinks removed.
    pub removed_files: u64,
    /// Bytes still on disk behind a skipped entry, where the size was
    /// known (a stat'd entry that would not unlink). "1 entry skipped"
    /// alone hides whether 4 KiB or 2 GiB stayed; a directory we could
    /// not even list contributes nothing here.
    pub skipped_bytes: u64,
    pub skipped: Vec<Skipped>,
}

impl ClearOutcome {
    pub fn permission_skips(&self) -> bool {
        self.skipped.iter().any(|s| s.permission)
    }
}

/// Delete the CONTENTS of `root`, keeping `root` itself (bind-mount
/// sources must survive; see module docs). Error-tolerant: every failed
/// entry is recorded in [`ClearOutcome::skipped`] and the walk continues.
pub fn clear_contents(root: &Path) -> ClearOutcome {
    let mut out = ClearOutcome::default();
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(e) => {
            push_skip(&mut out, root, &e);
            return out;
        }
    };
    for entry in entries {
        match entry {
            Ok(entry) => remove_tree(&entry.path(), &mut out),
            Err(e) => push_skip(&mut out, root, &e),
        }
    }
    out
}

fn remove_tree(path: &Path, out: &mut ClearOutcome) {
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) => {
            push_skip(out, path, &e);
            return;
        }
    };
    if meta.is_dir() {
        let skips_before = out.skipped.len();
        match fs::read_dir(path) {
            Ok(entries) => {
                for entry in entries {
                    match entry {
                        Ok(entry) => remove_tree(&entry.path(), out),
                        Err(e) => push_skip(out, path, &e),
                    }
                }
            }
            Err(e) => {
                push_skip(out, path, &e);
                return;
            }
        }
        // A dir whose children were skipped fails remove_dir with
        // NotEmpty; the children already carry the real error, so only
        // report the dir's own failure when its subtree was clean.
        if out.skipped.len() == skips_before {
            match fs::remove_dir(path) {
                Ok(()) => out.freed += disk_size(&meta),
                Err(e) => {
                    push_skip(out, path, &e);
                    out.skipped_bytes += disk_size(&meta);
                }
            }
        }
    } else {
        let size = disk_size(&meta);
        match fs::remove_file(path) {
            Ok(()) => {
                out.freed += size;
                out.removed_files += 1;
            }
            Err(e) => {
                push_skip(out, path, &e);
                out.skipped_bytes += size;
            }
        }
    }
}

fn push_skip(out: &mut ClearOutcome, path: &Path, e: &std::io::Error) {
    out.skipped.push(Skipped {
        path: path.to_path_buf(),
        error: e.to_string(),
        permission: e.kind() == ErrorKind::PermissionDenied,
    });
}

/// Human byte formatting: `79.0 GiB`, `986.2 MiB`, `12 B`.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{:.1} {}", value, UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero-dep tempdir helper, same pattern as cli_tests.rs.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("fdl-cargo-test-{pid}-{n}"));
            fs::create_dir_all(&dir).expect("tempdir creation");
            TempDir(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn mk(root: &Path, rel: &str) {
        fs::create_dir_all(root.join(rel)).expect("mkdir fixture");
    }

    fn touch(root: &Path, rel: &str, contents: &str) {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("mkdir parent");
        }
        fs::write(p, contents).expect("write fixture");
    }

    #[test]
    fn discover_classifies_by_convention() {
        let tmp = TempDir::new();
        let root = tmp.path();
        mk(root, "target");
        mk(root, ".target-docsrs");
        mk(root, ".cargo-cache");
        mk(root, ".cargo-cache-cuda");
        mk(root, ".cargo-git");
        // Workspace-excluded crate with its own target dir.
        touch(root, "ddp-bench/Cargo.toml", "[package]");
        mk(root, "ddp-bench/target");
        // Crate dir WITHOUT a target dir: not a root.
        touch(root, "hf-ddp/Cargo.toml", "[package]");
        // Plain dir with a target subdir but no Cargo.toml: not a root.
        mk(root, "docs/target");
        // Top-level FILE named like a cache: not a root.
        touch(root, ".cargo-cache-notes", "scratch");

        let roots = discover(root);
        let labels: Vec<(&str, Tier)> = roots.iter().map(|r| (r.label.as_str(), r.tier)).collect();
        assert_eq!(
            labels,
            vec![
                (".target-docsrs", Tier::Target),
                ("ddp-bench/target", Tier::Target),
                ("target", Tier::Target),
                (".cargo-cache", Tier::Cache),
                (".cargo-cache-cuda", Tier::Cache),
                (".cargo-git", Tier::Cache),
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn discover_follows_a_symlinked_target() {
        let tmp = TempDir::new();
        let root = tmp.path();
        // Builds parked on another disk, linked in as `target`.
        touch(root, "elsewhere/debug/app", "compiled");
        std::os::unix::fs::symlink(root.join("elsewhere"), root.join("target")).unwrap();

        let roots = discover(root);
        let labels: Vec<&str> = roots.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, vec!["target"], "a symlinked target must be seen");
        // And sized through the link, not as a 0-byte link.
        assert_eq!(usage(&roots[0].path).files, 1);
    }

    #[test]
    fn workspace_root_climbs_out_of_a_member_crate() {
        let tmp = TempDir::new();
        let root = tmp.path();
        touch(root, "Cargo.toml", "[workspace]\nmembers = [\"flodl\"]\n");
        touch(root, "flodl/Cargo.toml", "[package]\nname = \"flodl\"\n");
        // A member crate is where Context::resolve() stops; target/ is not
        // there, so the report would read empty.
        assert_eq!(workspace_root(&root.join("flodl")), root);
        assert_eq!(workspace_root(root), root);
    }

    #[test]
    fn workspace_root_keeps_a_standalone_crate() {
        let tmp = TempDir::new();
        let root = tmp.path();
        // No [workspace] anywhere: a single-crate project owns its target/.
        touch(root, "Cargo.toml", "[package]\nname = \"solo\"\n");
        assert_eq!(workspace_root(root), root);
    }

    #[test]
    fn workspace_root_stops_at_the_cargo_tree_edge() {
        let tmp = TempDir::new();
        let root = tmp.path();
        // An unrelated ancestor manifest must not be climbed into: the
        // walk only follows a contiguous chain of Cargo.toml files.
        touch(root, "outer/Cargo.toml", "[workspace]\n");
        touch(
            root,
            "outer/gap/proj/Cargo.toml",
            "[package]\nname = \"p\"\n",
        );
        let proj = root.join("outer/gap/proj");
        assert_eq!(workspace_root(&proj), proj);
    }

    #[test]
    fn discover_missing_root_is_empty() {
        let tmp = TempDir::new();
        let gone = tmp.path().join("nope");
        assert!(discover(&gone).is_empty());
    }

    #[test]
    fn usage_counts_files_recursively() {
        let tmp = TempDir::new();
        let root = tmp.path();
        touch(root, "a.bin", "aaaa");
        touch(root, "deep/b.bin", "bbbbbbbb");
        touch(root, "deep/er/c.bin", "cc");

        let u = usage(root);
        assert_eq!(u.files, 3);
        assert_eq!(u.unreadable, 0);
        // Block-based on unix, apparent elsewhere; either way three
        // non-empty files plus two dirs occupy at least the content.
        assert!(u.bytes >= 14, "bytes = {}", u.bytes);
    }

    #[test]
    fn clear_contents_empties_but_keeps_root() {
        let tmp = TempDir::new();
        let root = tmp.path();
        touch(root, "a.bin", "aaaa");
        touch(root, "deep/er/c.bin", "cc");
        mk(root, "empty");

        let out = clear_contents(root);
        assert_eq!(out.removed_files, 2);
        assert!(out.skipped.is_empty(), "skipped: {:?}", out.skipped);
        assert!(out.freed > 0);
        assert!(root.is_dir(), "root must survive a clear");
        assert_eq!(fs::read_dir(root).unwrap().count(), 0);
    }

    #[test]
    fn clear_contents_survives_missing_root() {
        let tmp = TempDir::new();
        let gone = tmp.path().join("nope");
        let out = clear_contents(&gone);
        assert_eq!(out.removed_files, 0);
        assert_eq!(out.skipped.len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn clear_contents_reports_undeletable_and_continues() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = TempDir::new();
        let root = tmp.path();
        touch(root, "locked/pinned.bin", "can't touch this");
        touch(root, "free.bin", "gone");
        // Read+exec but no write: children can be listed, not unlinked.
        // Same shape as a root-owned docs.rs target dir.
        let locked = root.join("locked");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o555)).unwrap();

        let out = clear_contents(root);

        // Restore before asserting so Drop can clean up either way.
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755)).unwrap();

        assert_eq!(out.removed_files, 1, "free.bin still removed");
        assert!(!out.skipped.is_empty());
        assert!(out.permission_skips());
        // The bytes left behind are reported, not merely counted as an
        // entry: "1 skipped" cannot distinguish 4 KiB from 2 GiB.
        assert!(out.skipped_bytes > 0, "skipped bytes must be accounted");
        // The locked dir itself must NOT be reported (children carry
        // the real error; its NotEmpty would be noise).
        assert!(
            out.skipped.iter().all(|s| s.path != locked),
            "skipped: {:?}",
            out.skipped
        );
    }

    #[test]
    fn format_bytes_scales() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(986 * 1024 * 1024), "986.0 MiB");
        assert_eq!(format_bytes(79 * 1024 * 1024 * 1024), "79.0 GiB");
    }
}
