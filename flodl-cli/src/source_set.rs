//! What a published tree CONTAINS, and how a worker knows it got all of it.
//!
//! A source root is rarely just source: the checkout `fdl publish` is run
//! from carries build output, registry caches, datasets, editor state and,
//! on a rig, container secrets, all gitignored and none of it the build's
//! business. Copying "everything minus three guessed names" shipped 28 GB
//! of that to the served tree once, private ssh host keys included. So the
//! tree is a whitelist, and the authority for it is cargo, which already
//! knows exactly which files a crate consists of:
//!
//! - `cargo metadata` in the project dir names every package the build
//!   resolves; the ones with no registry `source` are path crates, the
//!   dependency closure that has to travel.
//! - `cargo package --list` per path crate is cargo's own file list for
//!   it: tracked and untracked-but-not-ignored files inside a git repo,
//!   the crate directory minus `package.exclude` outside one. It is what
//!   `cargo publish` would ship, so a file it omits is one the crate has
//!   already declared not its own.
//! - The workspace-level files no crate lists but every build reads:
//!   the root manifest, the lockfile (gitignored in many trees, and the
//!   one gitignored file that MUST travel: it is the verified pin), the
//!   toolchain pin, cargo's config.
//!
//! A tree with no `Cargo.toml` at its project dir (a script build, a make
//! target) keeps the whole-tree copy; the whitelist is a cargo fact, not a
//! guess fdl makes about other build systems.
//!
//! The same list feeds the tree's digest: a `sha256sum`-format sidecar
//! beside the manifest, whose own hash the manifest carries. A worker
//! recomputes it before building and refuses a tree that differs, which
//! turns "pulled across a re-publish" from a convention (the manifest's
//! absence mid-publish) into a check that names the file.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::prepare::Fail;
use crate::source::MANIFEST_FILE;
use crate::util::sha256;

/// File name of the digest sidecar, at the root of a published tree.
/// `sha256sum -c .fdl-run.sha256` verifies it by hand.
pub const DIGEST_FILE: &str = ".fdl-run.sha256";

/// The files that make up a source tree, as paths relative to `root`.
///
/// Entries are `/`-joined strings, not `PathBuf`s: the list crosses
/// frames (rsync's `--files-from`, the `sha256sum` sidecar, a digest a
/// worker on another OS recomputes), and a path rendered by THIS
/// platform (`app\Cargo.toml` on Windows) is a different string with a
/// different hash. `Path::join` accepts `/` on every platform for the
/// way back to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSet {
    pub root: PathBuf,
    /// Sorted, deduplicated, every entry an existing regular file.
    pub files: Vec<String>,
    /// How many path crates contributed (0 for a whole-tree set).
    pub crates: usize,
}

/// What the manifest records about the tree's content.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeDigest {
    /// SHA-256 of the sidecar's bytes, hex.
    pub sha256: String,
    pub files: usize,
    pub bytes: u64,
}

/// Workspace-level files a build reads that no crate lists.
const WORKSPACE_FILES: [&str; 6] = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "rust-toolchain",
    ".cargo/config.toml",
    ".cargo/config",
];

/// Names `cargo package --list` prints that do not exist on disk: cargo
/// synthesizes them into the archive.
const SYNTHESIZED: [&str; 2] = ["Cargo.toml.orig", ".cargo_vcs_info.json"];

/// The cargo-derived file set for the project at `root/cwd`, or `None`
/// when there is no `Cargo.toml` there (a non-cargo build keeps the
/// whole tree).
///
/// Permanent failures: cargo missing (the gate build needs it anyway), a
/// manifest cargo cannot resolve, a path dependency outside `root` (the
/// tree would not be self-contained; publish from the dependency root).
pub fn cargo_file_set(root: &Path, cwd: Option<&str>) -> Result<Option<FileSet>, Fail> {
    let project = match cwd {
        Some(sub) => root.join(sub),
        None => root.to_path_buf(),
    };
    let manifest = project.join("Cargo.toml");
    if !manifest.is_file() {
        return Ok(None);
    }
    if !crate::util::system::has_command("cargo") {
        return Err(Fail::Permanent(
            "listing the source needs cargo, which is not installed: install \
             a toolchain (https://rustup.rs); the gate build needs it too"
                .to_string(),
        ));
    }
    let root = root
        .canonicalize()
        .map_err(|e| Fail::Permanent(format!("cannot resolve {}: {e}", root.display())))?;

    let meta = cargo_output(
        &["metadata", "--format-version", "1", "--manifest-path"],
        &manifest,
    )?;
    let meta: serde_json::Value = serde_json::from_str(&meta)
        .map_err(|e| Fail::Permanent(format!("cargo metadata is not JSON: {e}")))?;

    let mut crate_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    for pkg in meta["packages"].as_array().into_iter().flatten() {
        if !pkg["source"].is_null() {
            continue;
        }
        let Some(mp) = pkg["manifest_path"].as_str() else {
            continue;
        };
        let dir = Path::new(mp)
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let dir = dir.canonicalize().unwrap_or(dir);
        if dir.strip_prefix(&root).is_err() {
            return Err(Fail::Permanent(format!(
                "path dependency `{}` at {} lies outside the source root {}: the \
                 published tree would not be self-contained. Publish from the \
                 dependency root and name the project with --cwd",
                pkg["name"].as_str().unwrap_or("?"),
                dir.display(),
                root.display(),
            )));
        }
        crate_dirs.insert(dir);
    }

    let mut files: BTreeSet<String> = BTreeSet::new();
    for dir in &crate_dirs {
        let rel_dir = dir.strip_prefix(&root).unwrap_or(Path::new(""));
        let listing = cargo_output(
            &["package", "--list", "--allow-dirty", "--manifest-path"],
            &dir.join("Cargo.toml"),
        )?;
        for line in listing.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if SYNTHESIZED.contains(&line) {
                continue;
            }
            let rel = rel_dir.join(line);
            if root.join(&rel).is_file() {
                files.insert(slash(&rel));
            }
        }
    }

    // Workspace files: at the project dir (an excluded crate with its own
    // lockfile and toolchain pin) and at the workspace root of EVERY path
    // crate, not only the project's. A crate that inherits `edition` or
    // a version from `[workspace.package]` reads its workspace's root
    // manifest, and a dependency's workspace is usually not the
    // project's (ddp-bench is excluded from the workspace whose members
    // it depends on). Found by the first real publish: the tree built
    // nothing, `failed to find a workspace root`.
    let mut anchors: BTreeSet<PathBuf> = BTreeSet::new();
    anchors.insert(project.canonicalize().unwrap_or(project.clone()));
    for dir in &crate_dirs {
        let ws = cargo_output(
            &[
                "locate-project",
                "--workspace",
                "--message-format",
                "plain",
                "--manifest-path",
            ],
            &dir.join("Cargo.toml"),
        )?;
        if let Some(ws_dir) = Path::new(ws.trim()).parent() {
            anchors.insert(ws_dir.canonicalize().unwrap_or(ws_dir.to_path_buf()));
        }
    }
    for anchor in anchors {
        let Ok(rel_anchor) = anchor.strip_prefix(&root) else {
            continue;
        };
        for name in WORKSPACE_FILES {
            let rel = rel_anchor.join(name);
            if root.join(&rel).is_file() {
                files.insert(slash(&rel));
            }
        }
    }

    Ok(Some(FileSet {
        root,
        files: files.into_iter().collect(),
        crates: crate_dirs.len(),
    }))
}

/// A relative path as the `/`-joined string every frame agrees on.
fn slash(rel: &Path) -> String {
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn cargo_output(args: &[&str], manifest: &Path) -> Result<String, Fail> {
    let out = Command::new("cargo")
        .args(args)
        .arg(manifest)
        .output()
        .map_err(|e| Fail::Permanent(format!("spawn cargo: {e}")))?;
    if !out.status.success() {
        return Err(Fail::Permanent(format!(
            "`cargo {}` on {} failed ({}): {}",
            args.join(" "),
            manifest.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim(),
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Every regular file under `tree` that a worker would receive: the
/// pull excludes `target/` at any depth, and the two run files describe
/// the tree rather than belong to it. Relative, sorted.
pub fn tree_files(tree: &Path) -> Result<Vec<String>, Fail> {
    let mut out = Vec::new();
    walk(tree, tree, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(tree: &Path, dir: &Path, out: &mut Vec<String>) -> Result<(), Fail> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| Fail::Permanent(format!("cannot list {}: {e}", dir.display())))?;
    for entry in entries {
        let entry =
            entry.map_err(|e| Fail::Permanent(format!("cannot list {}: {e}", dir.display())))?;
        let path = entry.path();
        let name = entry.file_name();
        let ft = entry
            .file_type()
            .map_err(|e| Fail::Permanent(format!("cannot stat {}: {e}", path.display())))?;
        if ft.is_dir() {
            if name == "target" {
                continue;
            }
            walk(tree, &path, out)?;
        } else if ft.is_file() {
            if dir == tree && (name == MANIFEST_FILE || name == DIGEST_FILE) {
                continue;
            }
            out.push(slash(path.strip_prefix(tree).unwrap_or(&path)));
        }
    }
    Ok(())
}

/// Copy exactly `set` into `dest` and remove what is there and not in
/// it, leaving `target/` directories (the gate build's, or a worker's
/// own) and the run files alone.
pub fn sync(set: &FileSet, dest: &Path, notes: &mut Vec<String>) -> Result<(), Fail> {
    std::fs::create_dir_all(dest)
        .map_err(|e| Fail::Permanent(format!("cannot create {}: {e}", dest.display())))?;
    let list_path = dest.with_extension("files");
    let mut list = String::new();
    for f in &set.files {
        list.push_str(f);
        list.push('\n');
    }
    std::fs::write(&list_path, list)
        .map_err(|e| Fail::Permanent(format!("cannot write {}: {e}", list_path.display())))?;
    let argv = [
        "rsync".to_string(),
        "-a".to_string(),
        format!("--files-from={}", list_path.display()),
        format!("{}/", set.root.display()),
        format!("{}/", dest.display()),
    ];
    let result = crate::source::run_rsync(&argv, dest, true);
    let _ = std::fs::remove_file(&list_path);
    result?;

    let wanted: BTreeSet<&String> = set.files.iter().collect();
    let mut pruned = 0usize;
    for rel in tree_files(dest)? {
        if !wanted.contains(&rel) {
            let _ = std::fs::remove_file(dest.join(&rel));
            pruned += 1;
        }
    }
    remove_empty_dirs(dest, dest);
    if pruned > 0 {
        notes.push(format!(
            "source: removed {pruned} file(s) from the served tree that the \
             crates no longer list"
        ));
    }
    Ok(())
}

fn remove_empty_dirs(tree: &Path, dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    let mut empty = true;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && entry.file_name() != "target" {
            if !remove_empty_dirs(tree, &path) {
                empty = false;
            }
        } else {
            empty = false;
        }
    }
    if empty && dir != tree {
        return std::fs::remove_dir(dir).is_ok();
    }
    false
}

/// Hash `files` under `tree` into the sidecar's text (`sha256sum`
/// format) and the digest the manifest records.
pub fn digest(tree: &Path, files: &[String]) -> Result<(String, TreeDigest), Fail> {
    let mut sidecar = String::new();
    let mut bytes = 0u64;
    for rel in files {
        let path = tree.join(rel);
        let hex = sha256::file_hex(&path)
            .map_err(|e| Fail::Permanent(format!("cannot hash {}: {e}", path.display())))?;
        bytes += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        sidecar.push_str(&hex);
        sidecar.push_str("  ");
        sidecar.push_str(rel);
        sidecar.push('\n');
    }
    let d = TreeDigest {
        sha256: sha256::hex(sidecar.as_bytes()),
        files: files.len(),
        bytes,
    };
    Ok((sidecar, d))
}

/// Write the sidecar at the tree root. Ordered BEFORE the manifest by
/// the caller: the manifest's presence stays the commit point.
pub fn write_sidecar(tree: &Path, sidecar: &str) -> Result<(), Fail> {
    let path = tree.join(DIGEST_FILE);
    std::fs::write(&path, sidecar)
        .map_err(|e| Fail::Permanent(format!("cannot write {}: {e}", path.display())))
}

/// Check a fetched tree against the digest its manifest carries.
///
/// Transient on every mismatch: the usual cause is a pull that straddled
/// a re-publish, and the next dial gets a coherent tree. The message
/// names what differs, so a persistent one reads as what it is.
pub fn verify(tree: &Path, expected: &TreeDigest) -> Result<(), Fail> {
    let sidecar = std::fs::read_to_string(tree.join(DIGEST_FILE)).map_err(|e| {
        Fail::Transient(format!(
            "the fetched tree carries a manifest but no {DIGEST_FILE} ({e}): \
             pulled mid-publish? re-dialing fetches a coherent tree"
        ))
    })?;
    let got = sha256::hex(sidecar.as_bytes());
    if got != expected.sha256 {
        return Err(Fail::Transient(format!(
            "{DIGEST_FILE} does not match the manifest (sidecar {}…, manifest \
             {}…): the tree and its manifest are from different publishes; \
             re-dialing fetches a coherent one",
            &got[..8],
            &expected.sha256[..expected.sha256.len().min(8)],
        )));
    }
    let mut listed: BTreeSet<String> = BTreeSet::new();
    let mut problems: Vec<String> = Vec::new();
    for line in sidecar.lines() {
        let Some((hex, rel)) = line.split_once("  ") else {
            continue;
        };
        listed.insert(rel.to_string());
        match sha256::file_hex(&tree.join(rel)) {
            Ok(h) if h == hex => {}
            Ok(_) => problems.push(format!("{rel} differs")),
            Err(_) => problems.push(format!("{rel} is missing")),
        }
    }
    for rel in tree_files(tree)? {
        if !listed.contains(&rel) {
            problems.push(format!("{rel} is not in the published set"));
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    let shown = problems
        .iter()
        .take(5)
        .cloned()
        .collect::<Vec<_>>()
        .join("; ");
    let more = problems.len().saturating_sub(5);
    Err(Fail::Transient(format!(
        "the fetched tree does not match what was published ({} file(s)): {shown}{}",
        problems.len(),
        if more > 0 {
            format!("; and {more} more")
        } else {
            String::new()
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("fdl-set-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        base
    }

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    /// Two path crates and a heap of non-source beside them: the set is
    /// the crates' own files plus the workspace-level ones, and nothing
    /// under the junk directories, whatever their size.
    #[test]
    fn the_cargo_set_is_the_path_closure_and_nothing_else() {
        if !crate::util::system::has_command("cargo") {
            return;
        }
        let root = scratch("cargo");
        // The dependency lives in a workspace the project is NOT a member
        // of, and inherits its edition from it: the root manifest is a
        // build input no crate lists.
        write(
            &root.join("Cargo.toml"),
            "[workspace]\nmembers=[\"lib\"]\nexclude=[\"app\"]\n\
             [workspace.package]\nedition=\"2021\"\n",
        );
        write(
            &root.join("lib/Cargo.toml"),
            "[package]\nname=\"lib\"\nversion=\"0.1.0\"\nedition.workspace=true\n",
        );
        write(&root.join("lib/src/lib.rs"), "pub fn f() {}\n");
        write(
            &root.join("app/Cargo.toml"),
            "[package]\nname=\"app\"\nversion=\"0.1.0\"\nedition=\"2021\"\npublish=false\n\
             [dependencies]\nlib={path=\"../lib\"}\n",
        );
        write(&root.join("app/src/main.rs"), "fn main() { lib::f() }\n");
        write(
            &root.join("app/rust-toolchain.toml"),
            "[toolchain]\nchannel=\"stable\"\n",
        );
        write(&root.join(".hf-cache/hub/blob"), "22G of models");
        write(&root.join(".cargo-cache/registry/x"), "crates");
        write(&root.join(".ssh-keys/host_key"), "secret");
        write(&root.join("data/train.bin"), "dataset");

        let set = cargo_file_set(&root, Some("app"))
            .unwrap()
            .expect("a Cargo.toml at cwd yields a set");
        let files = &set.files;
        assert_eq!(set.crates, 2, "{files:?}");
        for must in [
            "Cargo.toml",
            "app/Cargo.toml",
            "app/src/main.rs",
            "app/rust-toolchain.toml",
            "lib/Cargo.toml",
            "lib/src/lib.rs",
        ] {
            assert!(
                files.contains(&must.to_string()),
                "{must} missing from {files:?}"
            );
        }
        for never in [
            ".hf-cache",
            ".cargo-cache",
            ".ssh-keys",
            "data/",
            "Cargo.toml.orig",
        ] {
            assert!(
                !files.iter().any(|f| f.contains(never)),
                "{never} must not ship: {files:?}"
            );
        }
        // The synced tree must be self-contained: cargo resolves the
        // project there with nothing from the original root.
        if crate::util::system::has_command("rsync") {
            let tree = root.with_file_name(format!(
                "{}-tree",
                root.file_name().unwrap().to_string_lossy()
            ));
            sync(&set, &tree, &mut Vec::new()).unwrap();
            let out = std::process::Command::new("cargo")
                .args([
                    "metadata",
                    "--format-version",
                    "1",
                    "--offline",
                    "--manifest-path",
                ])
                .arg(tree.join("app/Cargo.toml"))
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "the published tree does not resolve on its own: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let _ = std::fs::remove_dir_all(&tree);
        }
        // No Cargo.toml at cwd: not a cargo project, no set.
        assert_eq!(cargo_file_set(&root, Some("data")).unwrap(), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A path dependency above the root cannot travel; the error names it
    /// and the fix rather than publishing a tree that will not build.
    #[test]
    fn a_dependency_outside_the_root_is_permanent_and_named() {
        if !crate::util::system::has_command("cargo") {
            return;
        }
        let base = scratch("outside");
        write(
            &base.join("lib/Cargo.toml"),
            "[package]\nname=\"lib\"\nversion=\"0.1.0\"\nedition=\"2021\"\n",
        );
        write(&base.join("lib/src/lib.rs"), "");
        write(
            &base.join("proj/app/Cargo.toml"),
            "[package]\nname=\"app\"\nversion=\"0.1.0\"\nedition=\"2021\"\n\
             [dependencies]\nlib={path=\"../../lib\"}\n",
        );
        write(&base.join("proj/app/src/main.rs"), "fn main() {}\n");
        let err = cargo_file_set(&base.join("proj"), Some("app")).unwrap_err();
        assert!(err.is_permanent(), "{err:?}");
        assert!(err.message().contains("`lib`"), "{}", err.message());
        assert!(err.message().contains("--cwd"), "{}", err.message());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Sync lands exactly the set, prunes what the set dropped, and keeps
    /// a `target/` the gate built; the digest round-trips, and a tampered,
    /// missing or extra file is named as a transient mismatch.
    #[test]
    fn sync_prunes_to_the_set_and_the_digest_names_every_deviation() {
        if !crate::util::system::has_command("rsync") {
            return;
        }
        let base = scratch("sync");
        let (src, tree) = (base.join("src"), base.join("tree"));
        write(&src.join("a/keep.rs"), "keep");
        write(&src.join("a/b/deep.rs"), "deep");
        write(&src.join("junk.bin"), "junk");
        write(&tree.join("stale.rs"), "stale");
        write(&tree.join("target/release/bin"), "built");
        let set = FileSet {
            root: src.clone(),
            files: vec!["a/b/deep.rs".to_string(), "a/keep.rs".to_string()],
            crates: 1,
        };
        let mut notes = Vec::new();
        sync(&set, &tree, &mut notes).unwrap();
        assert!(tree.join("a/keep.rs").is_file());
        assert!(tree.join("a/b/deep.rs").is_file());
        assert!(!tree.join("junk.bin").exists(), "not in the set");
        assert!(!tree.join("stale.rs").exists(), "pruned");
        assert!(
            tree.join("target/release/bin").is_file(),
            "target/ is left alone"
        );
        assert!(
            notes.iter().any(|n| n.contains("removed 1 file")),
            "{notes:?}"
        );
        assert_eq!(tree_files(&tree).unwrap(), set.files);

        let (sidecar, d) = digest(&tree, &set.files).unwrap();
        assert_eq!(d.files, 2);
        assert_eq!(d.bytes, 8);
        assert!(
            sidecar.lines().all(|l| l.len() > 66 && &l[64..66] == "  "),
            "{sidecar}"
        );
        write_sidecar(&tree, &sidecar).unwrap();
        verify(&tree, &d).unwrap();

        write(&tree.join("a/keep.rs"), "tampered");
        let err = verify(&tree, &d).unwrap_err();
        assert!(!err.is_permanent(), "{err:?}");
        assert!(
            err.message().contains("a/keep.rs differs"),
            "{}",
            err.message()
        );
        write(&tree.join("a/keep.rs"), "keep");

        std::fs::remove_file(tree.join("a/b/deep.rs")).unwrap();
        assert!(
            verify(&tree, &d)
                .unwrap_err()
                .message()
                .contains("deep.rs is missing")
        );
        write(&tree.join("a/b/deep.rs"), "deep");

        write(&tree.join("extra.rs"), "x");
        assert!(
            verify(&tree, &d)
                .unwrap_err()
                .message()
                .contains("extra.rs is not in the published set")
        );
        std::fs::remove_file(tree.join("extra.rs")).unwrap();

        // A sidecar from another publish beside this manifest.
        write(&tree.join(DIGEST_FILE), "0000  nothing\n");
        assert!(
            verify(&tree, &d)
                .unwrap_err()
                .message()
                .contains("different publishes")
        );
        std::fs::remove_file(tree.join(DIGEST_FILE)).unwrap();
        assert!(
            verify(&tree, &d)
                .unwrap_err()
                .message()
                .contains("pulled mid-publish")
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
