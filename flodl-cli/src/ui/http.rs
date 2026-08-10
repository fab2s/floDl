//! HTTP plumbing: parse a request, build a response, and the two
//! codecs the query string needs.
//!
//! Deliberately minimal — this serves one embedded page and a handful of
//! JSON routes on loopback, so a hand-rolled reader beats a dependency
//! (and `flodl-cli` carries none). Everything security-relevant lives
//! one level up in the router: this module reads bytes and writes bytes.

use std::collections::HashMap;
use std::io::Read;
use std::net::TcpStream;

use super::{MAX_BODY_BYTES, MAX_REQUEST_BYTES};

pub(super) struct Request {
    pub(super) method: String,
    /// Path without the query string.
    pub(super) path: String,
    pub(super) query: HashMap<String, String>,
    pub(super) host: Option<String>,
    pub(super) token: Option<String>,
    /// POST body, empty on GET.
    pub(super) body: Vec<u8>,
}

/// Read and parse the request line + headers (any body is ignored:
/// every route here is a GET). `None` on anything malformed — a
/// non-HTTP client gets a hangup, not a parse attempt.
pub(super) fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buf = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        if buf.len() >= MAX_REQUEST_BYTES {
            return None;
        }
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let head = String::from_utf8_lossy(&buf);
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?;
    let (path, query_str) = match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    };
    let mut query = HashMap::new();
    for pair in query_str.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        query.insert(percent_decode(k), percent_decode(v));
    }
    let mut host = None;
    let mut token = None;
    let mut content_length = 0usize;
    for line in lines {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim();
            if name.eq_ignore_ascii_case("host") {
                host = Some(value.to_string());
            } else if name.eq_ignore_ascii_case("x-fdl-token") {
                token = Some(value.to_string());
            } else if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().ok()?;
            }
        }
    }
    if content_length > MAX_BODY_BYTES {
        return None;
    }
    // Whatever of the body already arrived sits after the header
    // terminator; read the rest.
    let header_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)?;
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Some(Request {
        method,
        path: path.to_string(),
        query,
        host,
        token,
        body,
    })
}

/// Re-encode one query value for an upstream request line (the parse
/// decoded it). Conservative: everything but unreserved characters is
/// escaped.
pub(super) fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Percent-decode one query component (browsers encode `/`, `@`, `:`
/// in values like archive paths). `+` stays literal — these are path
/// components, not form encoding. Malformed escapes pass through
/// verbatim and fail whatever validation comes next.
pub(super) fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (
                bytes.get(i + 1).and_then(|b| (*b as char).to_digit(16)),
                bytes.get(i + 2).and_then(|b| (*b as char).to_digit(16)),
            )
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Both spellings a loopback browser sends, port included — we never
/// serve on 80, so a portless Host is nothing we produced.
pub(super) fn host_is_local(host: Option<&str>, port: u16) -> bool {
    let Some(host) = host else { return false };
    host == format!("127.0.0.1:{port}") || host == format!("localhost:{port}")
}

pub(super) fn http(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Connection: close\r\n\r\n",
        body.len(),
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

pub(super) fn json_ok(value: &serde_json::Value) -> Vec<u8> {
    http(
        "200 OK",
        "application/json",
        serde_json::to_string(value)
            .expect("api payloads serialize")
            .as_bytes(),
    )
}

pub(super) fn error_json(status: &str, message: &str) -> Vec<u8> {
    http(
        status,
        "application/json",
        serde_json::json!({ "error": message })
            .to_string()
            .as_bytes(),
    )
}
