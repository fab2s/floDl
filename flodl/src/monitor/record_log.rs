//! Append-only JSONL persistence for the monitor record stream
//! (`.design/monitoring-portal-b3.md`, "Persistence").
//!
//! Each node's records append to its own file, and a record's `path` **is**
//! the filesystem path: `root` → `root.log`, `root/exa/rank0` →
//! `root/exa/rank0.log`, with a node's children living in the sibling
//! directory `root/exa/`. One producer, two sinks — the same records that
//! stream live are the lines on disk.
//!
//! Three properties make this the durable half of the stream:
//!
//! - **Bounded, drop-oldest.** Each node's log is a ring of at most
//!   `SEGMENTS` segments; when the active segment fills, it rotates and
//!   the oldest is dropped. Total bytes per node never exceed the
//!   configured cap. Long runs cannot fill the disk.
//! - **Training never fails on I/O.** Every error here is swallowed (warned
//!   once per node). A monitoring sink must not be able to kill a training
//!   run — a full disk degrades observability, nothing else.
//! - **Tail-read resume.** Every `node` record is an absolute snapshot, so
//!   catching up means reading the last N lines ([`RecordLog::tail`]) — no
//!   seek index, no checkpoint replay.
//!
//! Records stay plain JSONL while live: gzip cannot be appended to or
//! cheaply tailed. Compressing rotated segments is a later, additive step.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::Value;

/// Segments retained per node: the active one plus one rotated. Two is the
/// minimum that bounds total bytes while still keeping history across a
/// rotation (with one, a rotation would drop *everything* older than the
/// current segment).
const SEGMENTS: u64 = 2;

/// Default per-node byte cap. At ~300 B a record this holds ~100k records
/// per node — far past any plausible sub-epoch report count — while a
/// thousand-node cluster still bounds to a few GB.
pub const DEFAULT_MAX_LOG_BYTES: u64 = 32 * 1024 * 1024;

/// One node's open log file plus the byte count that drives rotation.
struct NodeFile {
    file: File,
    /// Bytes in the ACTIVE segment (not the rotated one).
    bytes: u64,
    /// Active segment path, e.g. `<dir>/root/exa/rank0.log`.
    path: PathBuf,
    /// Set once an I/O error has been reported for this node, so a broken
    /// disk warns once instead of once per record.
    warned: bool,
}

/// Append-only JSONL record log: one bounded, rotating file per node.
///
/// Cloneable by `Arc`; all methods take `&self` and serialize internally.
pub struct RecordLog {
    /// Root directory holding the node tree.
    dir: PathBuf,
    /// Per-node byte cap (active + rotated segments).
    max_bytes: u64,
    /// Open handles keyed by record `path`.
    nodes: Mutex<HashMap<String, NodeFile>>,
}

impl RecordLog {
    /// A log rooted at `dir`, capping each node at `max_bytes` total.
    /// `max_bytes` is **per node**: a tree of `N` nodes bounds to
    /// `N * max_bytes`. A cap below `SEGMENTS` bytes is raised to it so a
    /// segment can always hold at least one byte.
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        RecordLog {
            dir: dir.into(),
            max_bytes: max_bytes.max(SEGMENTS),
            nodes: Mutex::new(HashMap::new()),
        }
    }

    /// Bytes a single segment may reach before rotating.
    fn segment_limit(&self) -> u64 {
        (self.max_bytes / SEGMENTS).max(1)
    }

    /// Append records to their nodes' logs, one JSON object per line.
    /// Records without a usable `path` are skipped. Never fails: I/O errors
    /// warn once per node and the run continues.
    pub fn append(&self, records: &[Value]) {
        let mut nodes = self.nodes.lock().unwrap();
        for rec in records {
            let Some(path) = rec.get("path").and_then(Value::as_str) else {
                continue;
            };
            let Some(rel) = safe_relative_path(path) else {
                continue;
            };
            let line = rec.to_string();
            self.append_one(&mut nodes, path, &rel, &line);
        }
    }

    /// Append one serialized record to `path`'s log, opening and rotating
    /// as needed.
    fn append_one(
        &self,
        nodes: &mut HashMap<String, NodeFile>,
        path: &str,
        rel: &Path,
        line: &str,
    ) {
        if !nodes.contains_key(path) {
            let Some(nf) = self.open_node(rel) else {
                return;
            };
            nodes.insert(path.to_string(), nf);
        }
        let limit = self.segment_limit();
        let nf = nodes.get_mut(path).expect("just inserted");

        // Rotate BEFORE writing when this record would overflow the active
        // segment, so a segment never exceeds the limit mid-record.
        let need = line.len() as u64 + 1;
        if nf.bytes > 0 && nf.bytes + need > limit {
            rotate(nf);
        }
        let res = nf
            .file
            .write_all(line.as_bytes())
            .and_then(|()| nf.file.write_all(b"\n"));
        match res {
            Ok(()) => nf.bytes += need,
            Err(e) => {
                if !nf.warned {
                    nf.warned = true;
                    crate::msg!(
                        "warning: monitor record log: write failed for {} ({e}); \
                         this node's log stops here (training continues)",
                        nf.path.display(),
                    );
                }
            }
        }
    }

    /// Open (creating parents) the active segment for a node, resuming its
    /// byte count from the existing file so a restart does not blow the cap.
    fn open_node(&self, rel: &Path) -> Option<NodeFile> {
        let path = self.dir.join(rel).with_extension("log");
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            crate::msg!(
                "warning: monitor record log: cannot create {} ({e}); \
                 record persistence disabled for this node",
                parent.display(),
            );
            return None;
        }
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => {
                let bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
                Some(NodeFile {
                    file,
                    bytes,
                    path,
                    warned: false,
                })
            }
            Err(e) => {
                crate::msg!(
                    "warning: monitor record log: cannot open {} ({e}); \
                     record persistence disabled for this node",
                    path.display(),
                );
                None
            }
        }
    }

    /// The last `n` records for `path`, oldest-first — the tail-read that
    /// replaces a checkpoint replay (each record is an absolute snapshot).
    /// Reads the rotated segment first, then the active one, keeping only
    /// the final `n` lines in memory. Empty when the node has no log.
    pub fn tail(&self, path: &str, n: usize) -> Vec<Value> {
        if n == 0 {
            return Vec::new();
        }
        // Flush whatever is buffered for this node so a tail taken mid-run
        // sees the records already appended.
        if let Ok(mut nodes) = self.nodes.lock()
            && let Some(nf) = nodes.get_mut(path)
        {
            let _ = nf.file.flush();
        }
        let Some(rel) = safe_relative_path(path) else {
            return Vec::new();
        };
        let active = self.dir.join(rel).with_extension("log");
        let mut lines: std::collections::VecDeque<String> =
            std::collections::VecDeque::with_capacity(n);
        for seg in [rotated_path(&active), active] {
            let Ok(file) = File::open(&seg) else { continue };
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                if line.is_empty() {
                    continue;
                }
                if lines.len() == n {
                    lines.pop_front();
                }
                lines.push_back(line);
            }
        }
        lines
            .into_iter()
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect()
    }

    /// Flush every open node log. Called at teardown so the final records
    /// are on disk.
    pub fn flush(&self) {
        if let Ok(mut nodes) = self.nodes.lock() {
            for nf in nodes.values_mut() {
                let _ = nf.file.flush();
            }
        }
    }
}

/// Rotate a node's active segment: the current file becomes the single
/// retained rotated segment (replacing whatever was there — that is the
/// drop-oldest), and a fresh empty active file takes its place. On any I/O
/// failure the node keeps writing to the existing handle; an oversized log
/// is strictly better than a lost one.
fn rotate(nf: &mut NodeFile) {
    let _ = nf.file.flush();
    let rotated = rotated_path(&nf.path);
    if std::fs::rename(&nf.path, &rotated).is_err() {
        return;
    }
    match OpenOptions::new().create(true).append(true).open(&nf.path) {
        Ok(file) => {
            nf.file = file;
            nf.bytes = 0;
        }
        Err(e) => {
            if !nf.warned {
                nf.warned = true;
                crate::msg!(
                    "warning: monitor record log: reopen after rotate failed for {} ({e})",
                    nf.path.display(),
                );
            }
        }
    }
}

/// Path of the single retained rotated segment for an active segment.
fn rotated_path(active: &Path) -> PathBuf {
    let mut s = active.as_os_str().to_os_string();
    s.push(".1");
    PathBuf::from(s)
}

/// Map a record `path` to a relative filesystem path, refusing anything
/// that could escape the log directory. Our own builder only ever emits
/// `root/host/rankN`, but a record is data — an absolute component, a `..`,
/// or a root/prefix component must never be joined onto the log dir.
fn safe_relative_path(path: &str) -> Option<PathBuf> {
    if path.is_empty() {
        return None;
    }
    let mut out = PathBuf::new();
    for seg in path.split('/') {
        if seg.is_empty() || seg == "." || seg == ".." {
            return None;
        }
        // Reject anything the OS would read as a root, prefix, or separator.
        let p = Path::new(seg);
        if p.components().count() != 1
            || !matches!(
                p.components().next(),
                Some(std::path::Component::Normal(_))
            )
        {
            return None;
        }
        out.push(seg);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Unique temp dir per test (pid + name), removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir()
                .join(format!("flodl-recordlog-{}-{name}", std::process::id()));
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

    fn rec(path: &str, tick: u64) -> Value {
        json!({ "v": 1, "kind": "node", "path": path, "tick": tick })
    }

    #[test]
    fn path_is_the_filesystem_path() {
        let d = TempDir::new("paths");
        let log = RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES);
        log.append(&[
            rec("root", 1),
            rec("root/exa", 1),
            rec("root/exa/rank0", 1),
        ]);
        log.flush();
        assert!(d.0.join("root.log").is_file());
        assert!(d.0.join("root/exa.log").is_file());
        assert!(d.0.join("root/exa/rank0.log").is_file());
    }

    #[test]
    fn appends_one_json_per_line_per_node() {
        let d = TempDir::new("append");
        let log = RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES);
        log.append(&[rec("root", 1), rec("root/rank0", 1)]);
        log.append(&[rec("root", 2), rec("root/rank0", 2)]);
        log.flush();
        let root = std::fs::read_to_string(d.0.join("root.log")).unwrap();
        assert_eq!(root.lines().count(), 2);
        // Each line is a standalone JSON object (the JSONL contract).
        for line in root.lines() {
            let v: Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["path"], "root");
        }
        let r0 = std::fs::read_to_string(d.0.join("root/rank0.log")).unwrap();
        assert_eq!(r0.lines().count(), 2);
    }

    #[test]
    fn tail_returns_last_n_oldest_first() {
        let d = TempDir::new("tail");
        let log = RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES);
        for t in 1..=10 {
            log.append(&[rec("root", t)]);
        }
        let got = log.tail("root", 3);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["tick"], 8);
        assert_eq!(got[2]["tick"], 10);
        // Asking for more than exists yields everything.
        assert_eq!(log.tail("root", 100).len(), 10);
        // n = 0 and unknown nodes are empty, not errors.
        assert!(log.tail("root", 0).is_empty());
        assert!(log.tail("root/nope", 5).is_empty());
    }

    #[test]
    fn rotation_bounds_total_bytes_and_drops_oldest() {
        let d = TempDir::new("rotate");
        // Tiny cap: 400 B total => 200 B per segment, a few records each.
        let log = RecordLog::new(&d.0, 400);
        for t in 1..=200 {
            log.append(&[rec("root", t)]);
        }
        log.flush();

        let active = d.0.join("root.log");
        let rotated = d.0.join("root.log.1");
        let a = std::fs::metadata(&active).unwrap().len();
        let r = std::fs::metadata(&rotated).map(|m| m.len()).unwrap_or(0);
        assert!(a <= 200, "active segment {a} over the 200 B limit");
        assert!(r <= 200, "rotated segment {r} over the 200 B limit");
        assert!(a + r <= 400, "total {} over the cap", a + r);

        // Drop-oldest: the earliest records are gone, the latest survive.
        let kept = log.tail("root", 1000);
        assert!(!kept.is_empty());
        let ticks: Vec<u64> = kept.iter().map(|v| v["tick"].as_u64().unwrap()).collect();
        assert_eq!(*ticks.last().unwrap(), 200, "newest record retained");
        assert!(ticks.len() < 200, "old records dropped (kept {})", ticks.len());
        assert!(!ticks.contains(&1), "oldest record dropped");
        // Retained ticks stay contiguous and ordered (a ring, not a shuffle).
        for w in ticks.windows(2) {
            assert_eq!(w[1], w[0] + 1, "ticks contiguous: {ticks:?}");
        }
    }

    #[test]
    fn reopen_resumes_byte_count_so_the_cap_still_holds() {
        let d = TempDir::new("reopen");
        {
            let log = RecordLog::new(&d.0, 400);
            for t in 1..=20 {
                log.append(&[rec("root", t)]);
            }
            log.flush();
        }
        // A fresh log over the same dir continues the existing segment
        // rather than treating it as empty.
        let log = RecordLog::new(&d.0, 400);
        for t in 21..=200 {
            log.append(&[rec("root", t)]);
        }
        log.flush();
        let a = std::fs::metadata(d.0.join("root.log")).unwrap().len();
        let r = std::fs::metadata(d.0.join("root.log.1"))
            .map(|m| m.len())
            .unwrap_or(0);
        assert!(a + r <= 400, "total {} over the cap after reopen", a + r);
        assert_eq!(
            log.tail("root", 1).first().unwrap()["tick"],
            200,
            "newest record still retained across reopen",
        );
    }

    #[test]
    fn traversal_and_malformed_paths_are_refused() {
        // A record's `path` is data; it must never escape the log dir.
        assert!(safe_relative_path("root/../../etc/passwd").is_none());
        assert!(safe_relative_path("/etc/passwd").is_none());
        assert!(safe_relative_path("..").is_none());
        assert!(safe_relative_path("root//rank0").is_none());
        assert!(safe_relative_path("").is_none());
        assert_eq!(
            safe_relative_path("root/exa/rank0"),
            Some(PathBuf::from("root/exa/rank0")),
        );

        // End to end: an escaping record writes nothing, and does not
        // disturb the well-formed records beside it.
        let d = TempDir::new("traversal");
        let log = RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES);
        log.append(&[rec("../escape", 1), rec("root", 1)]);
        log.flush();
        assert!(d.0.join("root.log").is_file());
        assert!(!d.0.parent().unwrap().join("escape.log").exists());
    }

    #[test]
    fn records_without_a_path_are_skipped() {
        let d = TempDir::new("nopath");
        let log = RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES);
        log.append(&[json!({ "kind": "meta" }), rec("root", 1)]);
        log.flush();
        let root = std::fs::read_to_string(d.0.join("root.log")).unwrap();
        assert_eq!(root.lines().count(), 1);
    }

    #[test]
    fn unwritable_dir_never_panics() {
        // A path that cannot be created (a FILE where a dir must go):
        // persistence degrades, training continues.
        let d = TempDir::new("unwritable");
        std::fs::write(d.0.join("root"), b"i am a file, not a dir").unwrap();
        let log = RecordLog::new(&d.0, DEFAULT_MAX_LOG_BYTES);
        log.append(&[rec("root/rank0", 1)]); // parent "root" is a file
        log.flush();
        assert!(log.tail("root/rank0", 5).is_empty());
    }
}
