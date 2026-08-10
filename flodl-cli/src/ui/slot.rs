//! The dashboard slot: a loopback proxy to a live run's dashboard, and
//! the persisted dashboards a finished run left on disk.
//!
//! One slot, whose backing follows the run's lifecycle — admission view,
//! then the live dashboard proxied through this server's own port (so a
//! headless controller needs ONE ssh forward), then an archive from
//! disk. The proxy target is a PORT: the host is hardwired to loopback,
//! so it cannot be aimed off-box.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::http::{error_json, http};
use super::{Request, UiServer};

// ── The dashboard slot: loopback proxy + archives ───────────────────────

/// Forward one GET to the dashboard's loopback port and relay the raw
/// response until either side closes. No rewriting: the dashboard's
/// routes are forwarded under their own paths, so its root-relative
/// fetches resolve correctly from inside the iframe. The upstream read
/// gets NO timeout on purpose — `/events` and `/stream` are SSE legs
/// that legitimately sit idle between window ticks.
pub(super) fn proxy_dashboard(mut stream: TcpStream, server: &UiServer, target: &str) {
    // Same rule as `follow`: an SSE leg to a backgrounded tab may
    // legitimately stall past any fixed budget.
    let _ = stream.set_write_timeout(None);
    let Some(port) = *server.run_target.lock().expect("run target lock") else {
        let _ = stream.write_all(&error_json(
            "502 Bad Gateway",
            "no dashboard target set — set the port on the run tab",
        ));
        return;
    };
    let upstream = TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(500),
    );
    let Ok(mut upstream) = upstream else {
        let _ = stream.write_all(&error_json(
            "502 Bad Gateway",
            &format!("nothing answering on 127.0.0.1:{port} — is the run up?"),
        ));
        return;
    };
    if upstream
        .write_all(
            format!("GET {target} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .is_err()
    {
        let _ = stream.write_all(&error_json("502 Bad Gateway", "dashboard hung up"));
        return;
    }
    let mut buf = [0u8; 8192];
    loop {
        match upstream.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                if stream.write_all(&buf[..n]).is_err() {
                    return;
                }
            }
        }
    }
}

/// What counts as a servable run artifact — the single predicate both
/// the scan and `/archive` apply, so the list can never offer a file
/// the endpoint then refuses (or vice versa). Exactly the names the
/// harness writes: `dashboard.html` / `timeline.html`, bare or with a
/// `_<timestamp>` suffix. A dotted stem is refused — rustdoc source
/// pages are named `timeline.rs.html` and must never list.
pub(super) fn is_archive_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".html") else {
        return false;
    };
    if stem.contains('.') {
        return false;
    }
    ["dashboard", "timeline"].iter().any(|prefix| {
        stem == *prefix
            || stem
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('_'))
    })
}

/// Walk the project for persisted dashboards/timelines, newest first.
/// Bounded: heavy build/vendor trees are skipped, depth is capped, and
/// the result is truncated (newest wins) with the truncation visible in
/// the count.
pub(super) fn scan_archives(root: &Path) -> Vec<serde_json::Value> {
    // Dot-dirs are skipped wholesale (.git, .fdl, .target-docsrs,
    // .cargo-cache*, ...): no run artifact ever lives in hidden state,
    // and rustdoc trees under them are exactly the false-positive farm.
    // `src` is skipped because a crate's sources are where the
    // dashboard/timeline TEMPLATES live (flodl/src/monitor/), and a
    // template is an empty page, not a run.
    const SKIP: &[&str] = &["target", "libtorch", "node_modules", "_site", "src"];
    const MAX_DEPTH: usize = 6;
    const MAX_RESULTS: usize = 200;
    let mut found: Vec<(std::time::SystemTime, PathBuf, u64)> = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if path.is_dir() {
                if depth < MAX_DEPTH && !name.starts_with('.') && !SKIP.contains(&name) {
                    stack.push((path, depth + 1));
                }
            } else if is_archive_name(name)
                && let Ok(meta) = entry.metadata()
            {
                found.push((
                    meta.modified().unwrap_or(std::time::UNIX_EPOCH),
                    path,
                    meta.len(),
                ));
            }
        }
    }
    found.sort_by_key(|(mtime, _, _)| std::cmp::Reverse(*mtime));
    found.truncate(MAX_RESULTS);
    found
        .into_iter()
        .map(|(mtime, path, size)| {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            serde_json::json!({
                "path": rel,
                "mtime": mtime
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                "size": size,
            })
        })
        .collect()
}

/// Serve one archived dashboard file. The ?path= is project-relative;
/// absolute paths and any `..` component are refused before the
/// filesystem is touched, the resolved file must still live under the
/// project root, and it must pass the same name predicate the scan
/// applies.
pub(super) fn serve_archive(req: &Request, server: &UiServer) -> Vec<u8> {
    let Some(rel) = req.query.get("path") else {
        return error_json("400 Bad Request", "missing ?path=");
    };
    let rel_path = Path::new(rel);
    if rel_path.is_absolute()
        || rel_path
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return error_json("400 Bad Request", "path: project-relative, no `..`");
    }
    let name = rel_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if !is_archive_name(name) {
        return error_json("400 Bad Request", "path: not a run artifact");
    }
    let Ok(full) = server.root.join(rel_path).canonicalize() else {
        return error_json("404 Not Found", "no such archive");
    };
    let Ok(root) = server.root.canonicalize() else {
        return error_json("404 Not Found", "project root vanished");
    };
    if !full.starts_with(&root) {
        return error_json("400 Bad Request", "path: escapes the project root");
    }
    match std::fs::read(&full) {
        Ok(bytes) => http("200 OK", "text/html; charset=utf-8", &bytes),
        Err(_) => error_json("404 Not Found", "no such archive"),
    }
}

/// The run ledger, if the launch slice has written one yet. Bad lines
/// are skipped, not fatal — an append-only file's tail can be mid-write.
pub(super) fn read_runs_ledger(root: &Path) -> Vec<serde_json::Value> {
    let Ok(content) = std::fs::read_to_string(root.join(".fdl/ui/runs.jsonl")) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}
