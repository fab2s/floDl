//! Host-scoped dataset cache: the persistent, across-run tier below a
//! shared source root.
//!
//! Distinct from the staging cascade in [`super::loader`], which is
//! within-run and ephemeral (`DiskStage` removes its pack on drop). This
//! tier is where a dataset's raw bytes LIVE on a box, across runs.
//!
//! # The invariant
//!
//! A dataset has two storage tiers and they are not interchangeable:
//!
//! - the **source root** is READ-ONLY by contract. On a cluster it is
//!   commonly a shared mount exported read-only to some hosts, and it
//!   holds whatever a provisioning step put there. Its per-host value
//!   arrives via [`crate::distributed::cluster_data_path`].
//! - the **host cache** is node-local and writable. Anything the source
//!   root does not have is acquired here.
//!
//! So a training process never writes the source root. Checkpoints are
//! the only thing that legitimately does, because multi-host resume needs
//! every piece of the bundle on one shared layer.
//!
//! Getting this wrong is not a hypothetical: collapsing both roles onto
//! one path is what made ranks try to write a read-only project mount,
//! and it fails as a bare `EROFS` from deep inside a download.
//!
//! # Why this is framework surface and not each binary's business
//!
//! Every flodl program that reads a dataset faces the same two tiers, and
//! a convention that each binary re-derives is not a convention. flodl
//! still never learns how to fetch anything: [`resolve_cached`] takes the
//! acquisition as a closure, so URLs, byte ranges and archive formats stay
//! entirely with the caller.

use std::fs;
use std::path::{Path, PathBuf};

use crate::tensor::{Result, TensorError};

/// Node-local dataset cache root: `~/.flodl/data`.
///
/// Home-relative on purpose, by the same rule the cluster paths follow. A
/// source root is named across hosts, so it must be absolute (a `~` means
/// different things to different users, and mounts live at absolute
/// paths). A cache is one box's own bookkeeping, so it needs no
/// privileges to create and nothing outside the box ever names it.
///
/// Falls back to a temp-dir subtree when `HOME` is unset, and says so:
/// `/tmp` is frequently tmpfs, where a large corpus would be spent as RAM
/// rather than disk.
pub fn data_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".flodl").join("data");
    }
    let tmp = std::env::temp_dir().join("flodl-data");
    eprintln!(
        "flodl data: HOME unset, caching datasets under {} — on a tmpfs /tmp \
         this spends RAM, not disk. Set HOME, or pre-provision the source root.",
        tmp.display(),
    );
    tmp
}

/// Resolve one dataset file across the two tiers, acquiring it only if
/// neither has it.
///
/// Returns the path to read from, which is under `source_root` when that
/// tier already has a valid copy and under [`data_cache_dir`] otherwise.
/// `source_root` is never written.
///
/// `valid` decides what counts as already-present, so a caller can demand
/// more than existence. Checking an exact byte length, for instance, makes
/// a changed corpus size re-acquire instead of silently reading the
/// previous one. Note that a length check cannot detect interleaved writes
/// of the same length, which is why `fetch` should publish through
/// [`publish_atomically`] rather than stream to its destination.
///
/// `fetch` receives the destination path inside the cache and is called at
/// most once.
///
/// ```no_run
/// # use std::path::Path;
/// # fn download(_url: &str, _dst: &Path) -> flodl::tensor::Result<()> { Ok(()) }
/// # fn main() -> flodl::tensor::Result<()> {
/// let shard = flodl::data::resolve_cached(
///     Path::new("/flodl/data"),
///     "olmo",
///     "part-0.npy",
///     |p| p.exists(),
///     |dst| download("https://example.invalid/part-0.npy", dst),
/// )?;
/// # let _ = shard;
/// # Ok(())
/// # }
/// ```
pub fn resolve_cached(
    source_root: &Path,
    subdir: &str,
    file_name: &str,
    valid: impl Fn(&Path) -> bool,
    fetch: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
    let from_source = source_root.join(subdir).join(file_name);
    if valid(&from_source) {
        return Ok(from_source);
    }
    let dir = data_cache_dir().join(subdir);
    let cached = dir.join(file_name);
    if valid(&cached) {
        return Ok(cached);
    }
    ensure_dir(&dir)?;
    fetch(&cached)?;
    Ok(cached)
}

/// Write `dest` through a pid-unique sibling temp, then rename onto it.
///
/// Rename within a directory is atomic, so a reader sees either the old
/// file or the complete new one, never a half-written one. That is what
/// makes several processes acquiring the same file concurrently correct
/// rather than merely likely to work, and co-hosted ranks do exactly that:
/// they share one cache directory.
///
/// The temp is a SIBLING so the rename stays inside one filesystem; across
/// filesystems `rename` gives `EXDEV` and a copy fallback would not be
/// atomic. It is removed on every error path, so a failed acquisition
/// leaves nothing for the next attempt to trip over.
///
/// The staged bytes are flushed before the rename. Rename publishes the
/// NAME, and after a crash a visible name whose bytes never reached the
/// platter is a cache file that passes its own length check holding zeros.
pub fn publish_atomically(
    dest: &Path,
    write: impl FnOnce(&mut fs::File) -> Result<()>,
) -> Result<()> {
    let name = dest
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            TensorError::new(&format!(
                "flodl data: publish target has no file name: {}",
                dest.display()
            ))
        })?;
    let tmp = dest.with_file_name(format!("{name}.{}.part", std::process::id()));

    let staged = (|| {
        let mut f = fs::File::create(&tmp).map_err(|e| write_error(&tmp, &e))?;
        write(&mut f)?;
        f.sync_all()
            .map_err(|e| TensorError::new(&format!("sync {}: {e}", tmp.display())))
    })();
    if let Err(e) = staged {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }

    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        write_error(dest, &e)
    })
}

/// A write into a dataset directory failed. Say which fix applies.
///
/// The bare errno is not actionable on its own: `EROFS` on a dataset path
/// usually means "this host is a reader, not a provisioner" rather than
/// "something is broken", and `EACCES` / `ENOSPC` want the same two
/// answers, so the guidance is unconditional rather than errno-matched.
pub fn write_error(path: &Path, e: &std::io::Error) -> TensorError {
    TensorError::new(&format!(
        "write {}: {e}\n  \
         this dataset path cannot be written. Either provision it from a host \
         that can write it (and leave this one reading), or point the dataset \
         source at a writable path with room for the data.",
        path.display(),
    ))
}

/// `mkdir -p` that tolerates an already-present directory on a read-only
/// mount.
///
/// `fs::create_dir_all` issues `mkdir(2)` first and only catches `EEXIST`,
/// so on a read-only mount it fails with `EROFS` even when the directory
/// is already there and no write is actually needed. A host reading a
/// populated shared source hits exactly that.
fn ensure_dir(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(dir).map_err(|e| write_error(dir, &e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A present, valid source file is read IN PLACE and never copied,
    /// which is the whole point: the source root may be read-only.
    #[test]
    fn a_valid_source_file_is_used_where_it_is() {
        let root = std::env::temp_dir().join(format!("flodl-hc-src-{}", std::process::id()));
        let dir = root.join("sub");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("f.bin"), b"abc").unwrap();

        let got = resolve_cached(
            &root,
            "sub",
            "f.bin",
            |p| p.exists(),
            |_| panic!("must not fetch when the source root already has it"),
        )
        .unwrap();

        assert_eq!(got, dir.join("f.bin"));
        fs::remove_dir_all(&root).unwrap();
    }

    /// An invalid source copy does not satisfy the lookup: `valid` is the
    /// gate, not existence, so a stale corpus re-acquires.
    #[test]
    fn an_invalid_source_file_falls_through_to_the_fetch() {
        let root = std::env::temp_dir().join(format!("flodl-hc-inv-{}", std::process::id()));
        let dir = root.join("sub");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("f.bin"), b"short").unwrap();

        let sub = format!("flodl-hc-test-{}", std::process::id());
        let cache = data_cache_dir().join(&sub);
        let _ = fs::remove_dir_all(&cache);

        let right_size = |p: &Path| fs::metadata(p).map(|m| m.len() == 3).unwrap_or(false);
        let got = resolve_cached(&root, &sub, "f.bin", right_size, |dst| {
            publish_atomically(dst, |f| {
                use std::io::Write;
                f.write_all(b"abc").map_err(|e| write_error(dst, &e))
            })
        })
        .unwrap();

        assert!(
            got.starts_with(data_cache_dir()),
            "fetched copy must land in the cache: {got:?}"
        );
        assert_eq!(fs::read(&got).unwrap(), b"abc");
        fs::remove_dir_all(&root).unwrap();
        let _ = fs::remove_dir_all(&cache);
    }

    /// THE invariant, asserted directly: acquisition never creates
    /// anything under the source root.
    ///
    /// This is the regression guard for the failure that motivated the
    /// module. A rank running off a read-only project mount used to try to
    /// download into it and die on a bare `EROFS` from deep inside a
    /// stream copy. Stated as "nothing appears under the source root"
    /// rather than as a permissions test, because a suite running as root
    /// ignores a mode-0555 directory and would pass vacuously.
    #[test]
    fn acquisition_never_writes_under_the_source_root() {
        let root = std::env::temp_dir().join(format!("flodl-hc-ro-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();

        let sub = format!("flodl-hc-ro-{}", std::process::id());
        let cache = data_cache_dir().join(&sub);
        let _ = fs::remove_dir_all(&cache);

        let got = resolve_cached(
            &root,
            &sub,
            "f.bin",
            |p| p.exists(),
            |dst| {
                publish_atomically(dst, |f| {
                    use std::io::Write;
                    f.write_all(b"fetched").map_err(|e| write_error(dst, &e))
                })
            },
        )
        .unwrap();

        assert_eq!(fs::read(&got).unwrap(), b"fetched");
        let under_source: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(
            under_source.is_empty(),
            "the source root must be untouched, found {under_source:?}"
        );

        fs::remove_dir_all(&root).unwrap();
        let _ = fs::remove_dir_all(&cache);
    }

    /// A failed write leaves NOTHING at the destination and no temp
    /// behind. Without the temp-then-rename shape the destination would
    /// hold a truncated file that a later existence check accepts.
    #[test]
    fn a_failed_publish_leaves_neither_destination_nor_temp() {
        let dir = std::env::temp_dir().join(format!("flodl-hc-fail-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("f.bin");

        let err = publish_atomically(&dest, |f| {
            use std::io::Write;
            f.write_all(b"partial").unwrap();
            Err(TensorError::new("simulated mid-stream failure"))
        })
        .unwrap_err();
        assert!(err.to_string().contains("simulated"), "got: {err}");

        assert!(
            !dest.exists(),
            "destination must not exist after a failed publish"
        );
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp must be cleaned up, found {leftovers:?}"
        );

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_completed_publish_is_byte_exact() {
        let dir = std::env::temp_dir().join(format!("flodl-hc-ok-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("f.bin");

        publish_atomically(&dest, |f| {
            use std::io::Write;
            f.write_all(&[7u8; 4096])
                .map_err(|e| write_error(&dest, &e))
        })
        .unwrap();

        assert_eq!(fs::read(&dest).unwrap(), vec![7u8; 4096]);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The guidance is what makes an `EROFS` from inside a download
    /// actionable, so it must name both fixes.
    #[test]
    fn write_error_names_both_fixes() {
        let e = std::io::Error::other("read-only file system");
        let msg = write_error(Path::new("/flodl/data/x"), &e).to_string();
        assert!(msg.contains("provision it from a host"), "got: {msg}");
        assert!(msg.contains("writable path"), "got: {msg}");
    }
}
