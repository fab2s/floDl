//! Telemetry shipper: mirrors a node-local telemetry dir to shared storage.
//!
//! Stage two of the telemetry plane. [`super::spill::TimelineSpill`] (and,
//! by the same road, any producer writing the local telemetry dir) keeps
//! full-resolution history on node-local disk; a [`TelemetryShipper`]
//! tail-follows that directory on its own clock and appends what is new to
//! a destination directory — the shared run dir in cluster use. The wire
//! cost is the producers' trickle (KB/s), so a run's telemetry arrives
//! continuously instead of as an every-host burst at teardown, and an
//! ephemeral host that dies mid-run has already delivered everything but
//! the last tick.
//!
//! ## What a mirror is here
//!
//! `*.log` files are followed as **streams**: the shipper tracks a byte
//! offset per file and appends only new bytes to `<name>.partial` at the
//! destination. The local ring's rotation is invisible on the shared side —
//! when the active segment rotates, the unshipped tail is recovered from
//! the `.log.1` sibling before the offset resets — so the destination
//! accumulates the run's FULL history while the local copy stays bounded.
//! Every other file (`meta.json`) is copied whole when it changes, through
//! a temp name then rename. At teardown [`TelemetryShipper::finish`] ships
//! the residual and renames every `.partial` to its final name: a
//! `.partial` left behind is the honest signal of an unpublished mirror,
//! and a later harvest can adopt it.
//!
//! ## Containment
//!
//! The shipper must be structurally unable to harm training or the spill.
//! It reads the local dir only (never the timeline's RAM), destination
//! errors warn once per file and retry next tick, and `finish` waits a
//! bounded time for the final ship — a stalled shared mount (sshfs/NFS in
//! D-state, where no userspace timeout can interrupt the write) strands
//! the shipper thread alone: `finish` warns, detaches it, and teardown
//! proceeds.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Default mirror tick. The sources trickle at KB/s, so seconds of lag is
/// invisible; a faster tick would only add destination round trips.
pub const DEFAULT_SHIP_INTERVAL_MS: u64 = 5_000;

/// How long `finish` waits for the final ship + publish before declaring
/// the destination stalled and detaching the thread.
const FINISH_TIMEOUT: Duration = Duration::from_secs(10);

/// Per-file byte cap per tick, so one tick stays short even against a
/// backlog (a cold destination, a just-attached shipper). The residual
/// simply rides later ticks; the teardown drain is uncapped.
const TICK_BYTES_PER_FILE: u64 = 4 * 1024 * 1024;

/// In-flight suffix at the destination.
const PARTIAL_SUFFIX: &str = ".partial";

/// Compose the conventional destination for one producer's telemetry dir:
/// `<dest_root>/<host>/<src basename>`. The host component uses the
/// cluster world-map name (same resolution the roster uses), so the shared
/// tree's host names line up with `rank_samples` and cluster.yml.
pub fn shipping_dest(dest_root: impl Into<PathBuf>, src: &Path) -> PathBuf {
    let host = crate::distributed::cluster::resolve_hostname().unwrap_or_default();
    let host = super::spill::sanitize_component(&host, "unknown-host");
    let base = src
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let base = super::spill::sanitize_component(&base, "telemetry");
    dest_root.into().join(host).join(base)
}

/// Follow state for one `*.log` stream: how far into the ACTIVE segment
/// the destination is, plus the warn-once latch for destination errors.
struct FollowState {
    offset: u64,
    warned: bool,
}

/// Change detection for whole-copied aux files.
struct AuxState {
    len: u64,
    mtime: Option<std::time::SystemTime>,
    warned: bool,
}

/// Background mirror of a node-local telemetry dir to a destination dir.
/// Construct with [`TelemetryShipper::start`]; call
/// [`TelemetryShipper::finish`] after the producers finished (for the
/// timeline: after `TimelineSpill::finish`) so the final lines are shipped
/// and the `.partial` files publish.
pub struct TelemetryShipper {
    dest: PathBuf,
    stop: Arc<AtomicBool>,
    done: mpsc::Receiver<()>,
    handle: Option<JoinHandle<()>>,
}

impl TelemetryShipper {
    /// Mirror `src` into `dest` every `interval_ms` until [`Self::finish`].
    /// Never fails: an unusable destination warns once per file and keeps
    /// retrying; the sources are never touched beyond reads.
    pub fn start(src: impl Into<PathBuf>, dest: impl Into<PathBuf>, interval_ms: u64) -> Self {
        let src = src.into();
        let dest = dest.into();
        let stop = Arc::new(AtomicBool::new(false));
        let (done_tx, done) = mpsc::channel();
        let stop2 = Arc::clone(&stop);
        let dest2 = dest.clone();
        let handle = thread::Builder::new()
            .name("telemetry-shipper".into())
            .spawn(move || {
                run_loop(&src, &dest2, interval_ms, &stop2);
                let _ = done_tx.send(());
            })
            .ok();
        TelemetryShipper {
            dest,
            stop,
            done,
            handle,
        }
    }

    /// The destination directory this shipper mirrors into.
    pub fn dest(&self) -> &Path {
        &self.dest
    }

    /// Ship the residual, publish (`.partial` → final names), and stop.
    /// Bounded: a destination that stalls the final ship (a dead shared
    /// mount) gets a warning and a detached thread, never a hung teardown.
    pub fn finish(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        let Some(handle) = self.handle.take() else {
            return;
        };
        self.stop.store(true, Ordering::Relaxed);
        match self.done.recv_timeout(FINISH_TIMEOUT) {
            // Sent its done marker (or already had): the join is immediate.
            Ok(()) => {
                let _ = handle.join();
            }
            // Thread gone without a marker (panicked): join reaps it.
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = handle.join();
            }
            // Still ships after the deadline: a stalled destination write
            // cannot be interrupted from here, so leave the thread behind
            // rather than hang teardown on it.
            Err(mpsc::RecvTimeoutError::Timeout) => {
                crate::msg!(
                    "warning: telemetry shipper: final ship to {} did not \
                     complete within {}s (stalled shared mount?); detaching — \
                     .partial files there mark the unpublished mirror",
                    self.dest.display(),
                    FINISH_TIMEOUT.as_secs(),
                );
                drop(handle);
            }
        }
    }
}

impl Drop for TelemetryShipper {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run_loop(src: &Path, dest: &Path, interval_ms: u64, stop: &AtomicBool) {
    let interval = Duration::from_millis(interval_ms.max(1));
    let mut logs: HashMap<PathBuf, FollowState> = HashMap::new();
    let mut aux: HashMap<PathBuf, AuxState> = HashMap::new();
    loop {
        let stopping = stop.load(Ordering::Relaxed);
        tick(src, dest, &mut logs, &mut aux, stopping);
        if stopping {
            publish(dest);
            return;
        }
        let wake = Instant::now() + interval;
        while Instant::now() < wake {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
    }
}

/// One mirror pass. `drain_fully` (the teardown pass) lifts the per-file
/// byte cap so the residual ships completely.
fn tick(
    src: &Path,
    dest: &Path,
    logs: &mut HashMap<PathBuf, FollowState>,
    aux: &mut HashMap<PathBuf, AuxState>,
    drain_fully: bool,
) {
    let mut files = Vec::new();
    collect_files(src, Path::new(""), &mut files);
    let cap = if drain_fully {
        u64::MAX
    } else {
        TICK_BYTES_PER_FILE
    };
    for rel in files {
        let name = rel.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".log") {
            let st = logs.entry(rel.clone()).or_insert(FollowState {
                offset: 0,
                warned: false,
            });
            follow_log(src, dest, &rel, st, cap);
        } else if name.ends_with(".log.1") {
            // The rotated sibling is part of its active file's stream
            // (recovered inside follow_log), never a file of its own.
        } else {
            let st = aux.entry(rel.clone()).or_insert(AuxState {
                len: u64::MAX,
                mtime: None,
                warned: false,
            });
            copy_aux(src, dest, &rel, st);
        }
    }
}

/// Recursively list files under `root/rel`, as root-relative paths.
/// Unreadable directories are skipped: the mirror ships what it can see.
fn collect_files(root: &Path, rel: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root.join(rel)) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        let child = rel.join(entry.file_name());
        if ft.is_dir() {
            collect_files(root, &child, out);
        } else if ft.is_file() {
            out.push(child);
        }
    }
}

/// Advance one `*.log` stream: recover a rotation, then append new bytes
/// to the destination `.partial`. The offset only advances by bytes that
/// actually reached the destination, so a failed write retries next tick.
fn follow_log(src: &Path, dest: &Path, rel: &Path, st: &mut FollowState, cap: u64) {
    let src_path = src.join(rel);
    let Ok(md) = fs::metadata(&src_path) else {
        return;
    };
    let len = md.len();
    if len < st.offset {
        // The active segment rotated under us: what it held now lives in
        // the `.1` sibling. Ship the tail this stream had not yet read —
        // uncapped, since rotation is segment-rare and the tail is at most
        // one segment — then restart at the new active file's head.
        let mut rotated = src_path.clone().into_os_string();
        rotated.push(".1");
        let rotated = PathBuf::from(rotated);
        if let Ok(rmd) = fs::metadata(&rotated)
            && rmd.len() > st.offset
        {
            let _ = copy_range(&rotated, st.offset, u64::MAX, dest, rel, st);
        }
        st.offset = 0;
    }
    if len > st.offset {
        let shipped = copy_range(
            &src_path,
            st.offset,
            (len - st.offset).min(cap),
            dest,
            rel,
            st,
        );
        st.offset += shipped;
    }
}

/// Append up to `max` bytes of `src_path` starting at `from` onto the
/// destination `.partial` for `rel`. Returns the bytes that reached the
/// destination; destination errors warn once per stream and stop the copy
/// (retried next tick from the same offset).
fn copy_range(
    src_path: &Path,
    from: u64,
    max: u64,
    dest: &Path,
    rel: &Path,
    st: &mut FollowState,
) -> u64 {
    let Ok(mut src) = fs::File::open(src_path) else {
        return 0;
    };
    if src.seek(SeekFrom::Start(from)).is_err() {
        return 0;
    }
    let dest_path = partial_path(dest, rel);
    if let Some(parent) = dest_path.parent()
        && let Err(e) = fs::create_dir_all(parent)
    {
        warn_once(st, &dest_path, &e);
        return 0;
    }
    let mut out = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&dest_path)
    {
        Ok(f) => f,
        Err(e) => {
            warn_once(st, &dest_path, &e);
            return 0;
        }
    };
    let mut remaining = max;
    let mut shipped = 0u64;
    let mut buf = [0u8; 64 * 1024];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let n = match src.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        if let Err(e) = out.write_all(&buf[..n]) {
            warn_once(st, &dest_path, &e);
            break;
        }
        shipped += n as u64;
        remaining -= n as u64;
    }
    shipped
}

/// Copy a non-log file whole when its (len, mtime) changed, through a
/// `.partial` temp then rename so a reader never sees a half-written copy.
fn copy_aux(src: &Path, dest: &Path, rel: &Path, st: &mut AuxState) {
    let src_path = src.join(rel);
    let Ok(md) = fs::metadata(&src_path) else {
        return;
    };
    let mtime = md.modified().ok();
    if md.len() == st.len && mtime == st.mtime {
        return;
    }
    let final_path = dest.join(rel);
    let tmp_path = partial_path(dest, rel);
    let copied = final_path
        .parent()
        .map(fs::create_dir_all)
        .unwrap_or(Ok(()))
        .and_then(|()| fs::copy(&src_path, &tmp_path).map(|_| ()))
        .and_then(|()| fs::rename(&tmp_path, &final_path));
    match copied {
        Ok(()) => {
            st.len = md.len();
            st.mtime = mtime;
        }
        Err(e) => {
            if !st.warned {
                st.warned = true;
                crate::msg!(
                    "warning: telemetry shipper: cannot mirror {} ({e}); \
                     will keep retrying (training continues)",
                    final_path.display(),
                );
            }
        }
    }
}

/// Rename every `.partial` under `dest` to its final name.
fn publish(dest: &Path) {
    let mut files = Vec::new();
    collect_files(dest, Path::new(""), &mut files);
    for rel in files {
        let Some(name) = rel.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(PARTIAL_SUFFIX) else {
            continue;
        };
        let from = dest.join(&rel);
        let to = dest.join(rel.with_file_name(stem));
        if let Err(e) = fs::rename(&from, &to) {
            crate::msg!(
                "warning: telemetry shipper: cannot publish {} ({e}); \
                 leaving the .partial behind",
                from.display(),
            );
        }
    }
}

fn partial_path(dest: &Path, rel: &Path) -> PathBuf {
    let mut p = dest.join(rel).into_os_string();
    p.push(PARTIAL_SUFFIX);
    PathBuf::from(p)
}

fn warn_once(st: &mut FollowState, path: &Path, e: &std::io::Error) {
    if !st.warned {
        st.warned = true;
        crate::msg!(
            "warning: telemetry shipper: write to {} failed ({e}); \
             will keep retrying (training continues)",
            path.display(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique temp dir per test (pid + name), removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p =
                std::env::temp_dir().join(format!("flodl-shipper-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            TempDir(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn append(path: &Path, content: &str) {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    /// Live appends reach the mirror, and finish() publishes: the
    /// destination ends with the exact stream under the final name, no
    /// `.partial` left, aux files copied beside it.
    #[test]
    fn mirror_appends_and_publishes() {
        let src = TempDir::new("mirror-src");
        let dst = TempDir::new("mirror-dst");
        append(&src.0.join("timeline.log"), "a\nb\n");
        std::fs::write(src.0.join("meta.json"), "{\"pid\":1}\n").unwrap();

        let sh = TelemetryShipper::start(&src.0, &dst.0, 25);
        std::thread::sleep(Duration::from_millis(150));
        append(&src.0.join("timeline.log"), "c\n");
        std::thread::sleep(Duration::from_millis(150));
        sh.finish();

        let shipped = std::fs::read_to_string(dst.0.join("timeline.log")).unwrap();
        assert_eq!(shipped, "a\nb\nc\n");
        assert!(!dst.0.join("timeline.log.partial").exists(), "must publish");
        let meta = std::fs::read_to_string(dst.0.join("meta.json")).unwrap();
        assert_eq!(meta, "{\"pid\":1}\n");
    }

    /// The local ring's rotation is invisible on the shared side: bytes
    /// appended after the last tick land in the rotated sibling and are
    /// recovered from it, so the destination holds the FULL uninterrupted
    /// stream — including history the bounded local ring dropped.
    #[test]
    fn rotation_preserves_the_full_stream() {
        let src = TempDir::new("rot-src");
        let dst = TempDir::new("rot-dst");
        let active = src.0.join("timeline.log");
        append(&active, "1\n2\n3\n");

        let sh = TelemetryShipper::start(&src.0, &dst.0, 25);
        std::thread::sleep(Duration::from_millis(150));

        // Appended-then-rotated: these bytes may or may not have been
        // shipped before the rotation lands, and both schedules must
        // produce the same stream.
        append(&active, "4\n5\n");
        std::fs::rename(&active, src.0.join("timeline.log.1")).unwrap();
        append(&active, "6\n7\n");
        std::thread::sleep(Duration::from_millis(150));
        sh.finish();

        let shipped = std::fs::read_to_string(dst.0.join("timeline.log")).unwrap();
        assert_eq!(shipped, "1\n2\n3\n4\n5\n6\n7\n");
    }

    /// Files appearing after start are picked up (the record-log tree
    /// creates node files on first record).
    #[test]
    fn late_files_are_picked_up() {
        let src = TempDir::new("late-src");
        let dst = TempDir::new("late-dst");
        std::fs::create_dir_all(src.0.join("records/root")).unwrap();

        let sh = TelemetryShipper::start(&src.0, &dst.0, 25);
        std::thread::sleep(Duration::from_millis(80));
        append(&src.0.join("records/root/rank0.log"), "{\"tick\":1}\n");
        std::thread::sleep(Duration::from_millis(150));
        sh.finish();

        let shipped = std::fs::read_to_string(dst.0.join("records/root/rank0.log")).unwrap();
        assert_eq!(shipped, "{\"tick\":1}\n");
    }

    /// An unusable destination degrades to warnings: finish() returns
    /// promptly, the source is untouched, nothing panics. (A FILE where
    /// the dest dir must go — the same trick record_log's test uses, and
    /// unlike a chmod it also fails under root.)
    #[test]
    fn unusable_dest_never_blocks() {
        let src = TempDir::new("bad-src");
        let holder = TempDir::new("bad-dst-holder");
        let dest = holder.0.join("dest");
        std::fs::write(&dest, b"i am a file, not a dir").unwrap();
        append(&src.0.join("timeline.log"), "a\n");

        let sh = TelemetryShipper::start(&src.0, &dest, 25);
        std::thread::sleep(Duration::from_millis(120));
        sh.finish();

        let src_content = std::fs::read_to_string(src.0.join("timeline.log")).unwrap();
        assert_eq!(src_content, "a\n", "source must be untouched");
        assert!(dest.is_file(), "dest file must not be replaced");
    }

    /// `shipping_dest` composes `<root>/<host>/<src basename>` with
    /// sanitized components — never empty, never path-shaped.
    #[test]
    fn shipping_dest_shape() {
        let d = shipping_dest(
            "/shared/run",
            Path::new("/home/u/.flodl/telemetry/lenet-x-42"),
        );
        let mut parts = d.components().rev();
        let base = parts
            .next()
            .unwrap()
            .as_os_str()
            .to_string_lossy()
            .into_owned();
        let host = parts
            .next()
            .unwrap()
            .as_os_str()
            .to_string_lossy()
            .into_owned();
        assert_eq!(base, "lenet-x-42");
        assert!(!host.is_empty() && !host.contains('/'), "host: {host}");
        assert!(d.starts_with("/shared/run"));
    }
}
