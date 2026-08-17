//! The job slot: one long-running command, streamed and replayable.
//!
//! At most one job at a time, because two concurrent publishes race the
//! manifest commit point. The buffer is a drop-oldest ring (the tail is
//! what diagnoses a run) and every line carries its absolute stream
//! index, which is what lets a client whose transport died resume from
//! exactly where it stopped rather than replaying or gapping.
//!
//! A closed browser tab never kills a job: a publish must reach — or
//! cleanly fail before — its commit point regardless of who is watching.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::drive::{append_ledger, ask_for_color};
use super::http::error_json;
use super::{JOB_MAX_LINES, LedgerCtx, STREAM_HEARTBEAT, UiServer};

// ── The job slot: one long-running command, streamed and replayable ────

/// The buffer one job accumulates: NDJSON lines, pushed by the reader
/// threads, drained by however many sockets are following.
#[derive(Debug)]
pub(super) struct JobState {
    /// Monotonic per-server job identity. A resume cursor (`?from=`) is
    /// only meaningful against the stream that minted it; the id is what
    /// lets a reconnect that raced a NEW job say so instead of silently
    /// skipping the new job's head.
    pub(super) id: u64,
    pub(super) buf: Mutex<JobBuf>,
    pub(super) done: AtomicBool,
}

/// The drop-oldest ring plus how much of the stream's history it no
/// longer holds — followers use `base` to say what they missed instead
/// of silently starting late.
#[derive(Debug, Default)]
pub(super) struct JobBuf {
    pub(super) lines: std::collections::VecDeque<String>,
    /// Absolute stream index of `lines[0]` — the count dropped from
    /// the front so far.
    pub(super) base: usize,
}

impl JobState {
    /// Buffer one event, stamped with its ABSOLUTE stream index `i` —
    /// what lets a client that lost its transport reconnect with
    /// `?from=` and resume exactly where it died, instead of replaying
    /// or gapping. Synthetic lines a follower writes (heartbeats, gap
    /// markers) are never buffered and carry no index.
    pub(super) fn push(&self, mut line: serde_json::Value) {
        let mut buf = self.buf.lock().expect("job buffer lock");
        let idx = buf.base + buf.lines.len();
        if let Some(obj) = line.as_object_mut() {
            obj.insert("i".to_string(), idx.into());
        }
        buf.lines.push_back(line.to_string());
        while buf.lines.len() > JOB_MAX_LINES {
            buf.lines.pop_front();
            buf.base += 1;
        }
    }

    pub(super) fn push_final(&self, line: serde_json::Value) {
        self.push(line);
        self.done.store(true, Ordering::Release);
    }
}

/// At most one job at a time. Two concurrent publishes would race the
/// manifest commit point, so the second caller is told to wait rather
/// than silently queued.
#[derive(Default)]
pub(super) struct JobSlot {
    current: Mutex<Option<Arc<JobState>>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl JobSlot {
    /// Claim the slot, or say what is still running.
    pub(super) fn try_start(&self) -> Result<Arc<JobState>, String> {
        let mut current = self.current.lock().expect("job slot lock");
        if let Some(job) = current.as_ref()
            && !job.done.load(Ordering::Acquire)
        {
            return Err("a job is already running — follow it at /api/jobs/last".to_string());
        }
        let job = Arc::new(JobState {
            id: self.next_id.fetch_add(1, Ordering::Relaxed) + 1,
            buf: Mutex::new(JobBuf::default()),
            done: AtomicBool::new(false),
        });
        *current = Some(Arc::clone(&job));
        Ok(job)
    }

    pub(super) fn last(&self) -> Option<Arc<JobState>> {
        self.current.lock().expect("job slot lock").clone()
    }
}

/// Spawn the job's command and stream its buffer to this socket. The
/// child is never killed on client loss: a publish must reach (or
/// cleanly fail before) its manifest commit point regardless of a
/// closed tab, so the readers keep buffering and `/api/jobs/last`
/// replays what the tab missed.
pub(super) fn stream_job(
    mut stream: TcpStream,
    server: &UiServer,
    argv: Vec<String>,
    ledger: Option<LedgerCtx>,
) {
    let job = match server.job.try_start() {
        Ok(j) => j,
        Err(why) => {
            let _ = stream.write_all(&error_json("409 Conflict", &why));
            return;
        }
    };
    let mut cmd_line = vec!["fdl".to_string()];
    cmd_line.extend(argv.iter().cloned());
    job.push(serde_json::json!({ "cmd": cmd_line }));

    let mut cmd = Command::new(&server.fdl_bin);
    let spawned = ask_for_color(&mut cmd)
        .args(&argv)
        .current_dir(&server.root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    match spawned {
        Ok(mut child) => {
            let readers: Vec<_> = [
                child
                    .stdout
                    .take()
                    .map(|o| ("out", BufReader::new(Box::new(o) as Box<dyn Read + Send>))),
                child
                    .stderr
                    .take()
                    .map(|e| ("err", BufReader::new(Box::new(e) as Box<dyn Read + Send>))),
            ]
            .into_iter()
            .flatten()
            .map(|(tag, reader)| {
                let job = Arc::clone(&job);
                std::thread::spawn(move || {
                    for line in reader.lines() {
                        let Ok(line) = line else { break };
                        job.push(serde_json::json!({ "s": tag, "t": line }));
                    }
                })
            })
            .collect();
            let job_done = Arc::clone(&job);
            let waiter_cmd = cmd_line.clone();
            let started = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let clock = std::time::Instant::now();
            std::thread::spawn(move || {
                // Output first, exit last: join the readers before the
                // exit event so nothing lands after it.
                for r in readers {
                    let _ = r.join();
                }
                let exit = child.wait().ok().and_then(|s| s.code());
                // A launch's completion is what makes it history: the
                // ledger records invocations that actually ran, with
                // whatever artifact pointers are knowable here.
                if let Some(ctx) = ledger {
                    append_ledger(
                        &ctx.path,
                        &serde_json::json!({
                            "v": 1,
                            "ts": started,
                            "dur_s": clock.elapsed().as_secs(),
                            "farm": ctx.farm,
                            "argv": waiter_cmd,
                            "exit": exit,
                            "port": ctx.port,
                        }),
                    );
                }
                job_done.push_final(serde_json::json!({ "exit": exit }));
            });
        }
        Err(e) => {
            job.push(serde_json::json!({
                "s": "ui",
                "t": format!("cannot spawn {}: {e}", server.fdl_bin.display()),
            }));
            job.push_final(serde_json::json!({ "exit": serde_json::Value::Null }));
        }
    }
    follow(&mut stream, &job, 0, None);
}

/// `/api/jobs/last`: replay the current or finished job from its first
/// line and follow while it runs. A resume cursor is honored only
/// against the job that minted it: `job` is the client's recorded id
/// (from the stream preamble), and a mismatch — or an id-less cursor
/// pointing past this stream's end, which no same-job reconnect can
/// produce — replays from the start with a note, never a silent gap.
pub(super) fn follow_job(
    mut stream: TcpStream,
    server: &UiServer,
    from: usize,
    job_id: Option<u64>,
) {
    match server.job.last() {
        Some(job) => {
            let mut from = from;
            let mut note = None;
            let stale = match job_id {
                Some(id) => id != job.id,
                None => {
                    let buf = job.buf.lock().expect("job buffer lock");
                    from > buf.base + buf.lines.len()
                }
            };
            if stale && from > 0 {
                note = Some(format!(
                    "(the resume cursor belongs to an earlier job — \
                     replaying job {} from its start)",
                    job.id,
                ));
                from = 0;
            }
            follow(&mut stream, &job, from, note)
        }
        None => {
            let _ = stream.write_all(&error_json("404 Not Found", "no job has run yet"));
        }
    }
}

/// Stream a job's buffer as NDJSON from the start, then poll for new
/// lines until the job is done and drained. A dead socket ends the
/// following, never the job. The stream opens with a synthetic
/// `{"job": id}` preamble (unbuffered, no index) so every follower —
/// including one that connected after the ring dropped the head —
/// learns which stream its resume cursor will belong to.
pub(super) fn follow(stream: &mut TcpStream, job: &JobState, from: usize, note: Option<String>) {
    // Streaming legs drop the write timeout: a reader throttled by its
    // own rendering cost (a coverage run floods tens of thousands of
    // lines — found live, tab in the foreground) or by a backgrounded
    // tab builds backpressure, and a 10s budget then cuts a perfectly
    // healthy stream mid-run. A slow consumer is not a dead one; a
    // dead one still errors the write when TCP gives up.
    let _ = stream.set_write_timeout(None);
    let header = "HTTP/1.1 200 OK\r\n\
         Content-Type: application/x-ndjson\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Connection: close\r\n\r\n";
    if stream.write_all(header.as_bytes()).is_err() {
        return;
    }
    let preamble = format!("{{\"job\":{}}}\n", job.id);
    if stream.write_all(preamble.as_bytes()).is_err() {
        return;
    }
    if let Some(note) = note {
        let line = serde_json::json!({ "s": "ui", "t": note });
        if stream.write_all(format!("{line}\n").as_bytes()).is_err() {
            return;
        }
    }
    // `sent` is an ABSOLUTE stream position; the ring's `base` says
    // how much history is gone, so a follower that fell behind (or a
    // replay of a run that out-talked the ring) reports the gap
    // instead of silently starting late.
    let mut sent = from;
    let mut last_write = std::time::Instant::now();
    loop {
        // Batch and done-ness read under one lock. The final line is
        // pushed before `done` is stored (release), so seeing `done`
        // here guarantees the exit event is in the batch just taken —
        // this iteration drains everything and can end the stream.
        let (batch, dropped, done) = {
            let buf = job.buf.lock().expect("job buffer lock");
            let dropped = buf.base.saturating_sub(sent);
            let from = sent.max(buf.base) - buf.base;
            let batch: Vec<String> = buf.lines.iter().skip(from).cloned().collect();
            sent = buf.base + buf.lines.len();
            (batch, dropped, job.done.load(Ordering::Acquire))
        };
        if dropped > 0 {
            let gap = serde_json::json!({
                "s": "ui",
                "t": format!(
                    "({dropped} earlier lines fell out of the buffer — it keeps \
                     the most recent {JOB_MAX_LINES})",
                ),
            });
            if stream.write_all(format!("{gap}\n").as_bytes()).is_err() {
                return;
            }
        }
        if !batch.is_empty() {
            let mut chunk = batch.join("\n");
            chunk.push('\n');
            if stream.write_all(chunk.as_bytes()).is_err() {
                return;
            }
            last_write = std::time::Instant::now();
        } else if !done && last_write.elapsed() >= STREAM_HEARTBEAT {
            // A no-op object the page ignores: bytes on the wire are
            // what keep an idle stream alive through whatever sits
            // between here and the reader.
            if stream.write_all(b"{}\n").is_err() {
                return;
            }
            last_write = std::time::Instant::now();
        }
        if done {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}
