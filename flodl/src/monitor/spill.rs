//! Node-local timeline spill: the durable half of the timeline broadcast.
//!
//! [`super::Timeline`]'s archive is RAM-only, so a crashed run loses its
//! timeline exactly when the forensic artifact is most needed. The spill
//! closes that hole: a [`TimelineSpill`] subscribes to the timeline's
//! batched broadcast (1 s cadence by default) and appends every sample and
//! event to a bounded JSONL ring on node-local disk. Disk never appears on
//! the training thread — `Timeline::event` stays a mutex push into RAM,
//! and the spill consumes the broadcast on its own thread.
//!
//! The ring reuses [`super::record_log::RecordLog`]'s discipline (bounded
//! drop-oldest segments, I/O errors swallowed and warned once): a full or
//! broken disk degrades observability, never training.
//!
//! ## Stream contract
//!
//! One interleaved JSONL stream per producer, `timeline.log` (active) plus
//! `timeline.log.1` (rotated). Line shapes are byte-identical to the
//! corresponding array elements in [`super::Timeline::save_json`], so any
//! consumer of `timeline.json` can parse spill lines with the same code.
//! Discriminant, in order: a `"k"` key marks an event, a `"rank"` key
//! marks a rank-reported sample, neither marks a local poller sample.
//!
//! ## Directory naming
//!
//! [`telemetry_dir`] composes the conventional location:
//! `~/.flodl/telemetry/<label>-<UTC stamp>-<pid>/`, home-relative by the
//! same rule as the dataset cache (one box's own bookkeeping; nothing
//! outside the box ever names it). A `meta.json` beside the log carries
//! host / pid / start time for attributing an orphaned spill to its run —
//! in the directory rather than the stream, so ring rotation can never
//! drop it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::record_log::RecordLog;
use super::timeline::{
    Timeline, TimelineBroadcast, write_event_json, write_rank_sample_json, write_sample_json,
};

/// Default spill ring cap. The stream trickles at ~1-2 KB/s (10 samples/s
/// plus training events), so this holds multiple days of full-resolution
/// history while bounding what an abandoned run dir can cost the disk.
pub const DEFAULT_SPILL_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// The single node the spill writes under: `<dir>/timeline.log`.
const SPILL_NODE: &str = "timeline";

/// Node-local telemetry root: `~/.flodl/telemetry`.
///
/// Falls back to a temp-dir subtree when `HOME` is unset, and says so: a
/// tmpfs `/tmp` keeps the spill in RAM and loses it on reboot, which is
/// exactly the durability the spill exists to provide.
pub fn telemetry_root() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".flodl").join("telemetry");
    }
    let tmp = std::env::temp_dir().join("flodl-telemetry");
    eprintln!(
        "flodl monitor: HOME unset, spilling telemetry under {} — on a tmpfs \
         /tmp this lives in RAM and vanishes on reboot (crash forensics \
         degraded). Set HOME to keep spills on disk.",
        tmp.display(),
    );
    tmp
}

/// A run-scoped directory under [`telemetry_root`]:
/// `<root>/<label>-<UTC stamp>-<pid>`. Readable in a directory listing and
/// unique per process (label sanitized, stamp to the second, pid breaks
/// same-second collisions on one box).
pub fn telemetry_dir(label: &str) -> PathBuf {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    telemetry_root().join(run_id(label, secs, std::process::id()))
}

/// `<label>-<UTC stamp>-<pid>`, label reduced to `[A-Za-z0-9._-]` so a
/// caller-supplied string can never change the directory shape.
fn run_id(label: &str, unix_secs: u64, pid: u32) -> String {
    let mut clean: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    if clean.is_empty() {
        clean.push_str("run");
    }
    format!(
        "{clean}-{}-{pid}",
        super::format::format_utc_stamp(unix_secs)
    )
}

/// Background consumer spilling a [`Timeline`]'s broadcast to node-local
/// disk. Construct with [`TimelineSpill::start`]; call
/// [`TimelineSpill::finish`] after `Timeline::stop` so the final broadcast
/// (flushed by the poller on its way out) is drained to disk. Dropping
/// without `finish` still joins the thread and flushes what was received.
pub struct TimelineSpill {
    dir: PathBuf,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl TimelineSpill {
    /// Subscribe to `timeline` and spill its broadcast under `dir` (created
    /// if needed), ring-capped at `max_bytes`. Never fails: an unusable
    /// `dir` warns and the spill degrades to a no-op, training untouched.
    pub fn start(timeline: &Arc<Timeline>, dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        let dir = dir.into();
        write_meta(&dir);
        let rx = timeline.subscribe();
        let log = RecordLog::new(&dir, max_bytes);
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("timeline-spill".into())
            .spawn(move || {
                let mut line = String::with_capacity(256);
                loop {
                    match rx.recv_timeout(Duration::from_millis(200)) {
                        Ok(batch) => write_batch(&log, &batch, &mut line),
                        Err(RecvTimeoutError::Timeout) => {
                            if stop2.load(Ordering::Relaxed) {
                                // Drain what the timeline flushed on stop
                                // before exiting.
                                while let Ok(batch) = rx.try_recv() {
                                    write_batch(&log, &batch, &mut line);
                                }
                                break;
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
                log.flush();
            })
            .ok();
        TimelineSpill { dir, stop, handle }
    }

    /// The spill directory (holds `timeline.log` and `meta.json`).
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Stop the spill thread, draining pending broadcasts and flushing the
    /// log. Call after `Timeline::stop` to capture the final batch.
    pub fn finish(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for TimelineSpill {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Append a broadcast batch as JSONL lines, one buffer reused throughout.
fn write_batch(log: &RecordLog, batch: &TimelineBroadcast, line: &mut String) {
    for s in &batch.samples {
        line.clear();
        write_sample_json(line, s);
        log.append_line(SPILL_NODE, line);
    }
    for r in &batch.rank_samples {
        line.clear();
        write_rank_sample_json(line, r);
        log.append_line(SPILL_NODE, line);
    }
    for e in &batch.events {
        line.clear();
        write_event_json(line, e);
        log.append_line(SPILL_NODE, line);
    }
}

/// Best-effort `meta.json` for orphan attribution. Failures warn and the
/// spill proceeds — the log itself may still be writable.
fn write_meta(dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        crate::msg!(
            "warning: timeline spill: cannot create {} ({e}); \
             spill disabled for this run (training continues)",
            dir.display(),
        );
        return;
    }
    let host = crate::distributed::cluster::resolve_hostname()
        .unwrap_or_default()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let body = format!(
        "{{\"host\":\"{host}\",\"pid\":{},\"started_unix_ms\":{started}}}\n",
        std::process::id(),
    );
    let path = dir.join("meta.json");
    if let Err(e) = std::fs::write(&path, body) {
        crate::msg!(
            "warning: timeline spill: cannot write {} ({e})",
            path.display(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::EventKind;

    /// Unique temp dir per test (pid + name), removed on drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("flodl-spill-{}-{name}", std::process::id()));
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

    #[test]
    fn run_id_is_sanitized_and_stamped() {
        assert_eq!(
            run_id("lenet nccl/sync", 0, 42),
            "lenet-nccl-sync-19700101-000000-42"
        );
        assert_eq!(run_id("", 0, 1), "run-19700101-000000-1");
    }

    /// End-to-end: samples, events, and rank samples all land as JSONL
    /// lines carrying their discriminants, every line is valid JSON, and
    /// meta.json is written beside the log. Timing bar matches
    /// `test_subscribe_receives_batches`: at least one of each suffices on
    /// a loaded host.
    #[test]
    fn spill_writes_interleaved_stream() {
        let d = TempDir::new("stream");
        let tl = Timeline::with_intervals(50, 100);
        let spill = TimelineSpill::start(&tl, &d.0, DEFAULT_SPILL_MAX_BYTES);

        tl.event(EventKind::EpochStart { epoch: 0 });
        tl.rank_sample(
            2,
            "pascal",
            &crate::monitor::ResourceSample {
                cpu_percent: Some(12.0),
                ..Default::default()
            },
        );
        tl.start();
        std::thread::sleep(Duration::from_millis(350));
        tl.stop();
        spill.finish();

        let content = std::fs::read_to_string(d.0.join("timeline.log")).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert!(
            lines.iter().any(|l| l.contains("\"k\":\"epoch_start\"")),
            "event line missing: {content}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("\"rank\":2") && l.contains("\"host\":\"pascal\"")),
            "rank sample line missing: {content}"
        );
        assert!(
            lines.iter().any(|l| !l.contains("\"k\":")
                && !l.contains("\"rank\":")
                && l.contains("\"cpu\":")),
            "poller sample line missing: {content}"
        );
        for l in &lines {
            let _: serde_json::Value =
                serde_json::from_str(l).unwrap_or_else(|e| panic!("invalid JSON line ({e}): {l}"));
        }

        let meta = std::fs::read_to_string(d.0.join("meta.json")).unwrap();
        assert!(meta.contains("\"pid\":"), "meta missing pid: {meta}");
        assert!(meta.contains("\"started_unix_ms\":"), "meta: {meta}");
    }

    /// A spill on a never-started timeline shuts down promptly and cleanly:
    /// meta.json exists, no panic, no hang on the 200ms stop tick.
    #[test]
    fn finish_without_poller_is_clean() {
        let d = TempDir::new("noop");
        let tl = Timeline::new(100);
        let spill = TimelineSpill::start(&tl, &d.0, DEFAULT_SPILL_MAX_BYTES);
        spill.finish();
        assert!(d.0.join("meta.json").is_file());
    }
}
