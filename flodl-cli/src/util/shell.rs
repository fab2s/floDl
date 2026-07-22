//! Shell-quoting helpers shared by every place fdl composes an `sh -c` /
//! `bash -c` command line (docker dispatch, cluster SSH fan-out, schema
//! probes, prebuild).

/// POSIX-quote a single token so it round-trips through `sh -c` / `bash
/// -c` as one argument. Empty strings become `''`; tokens containing
/// only safe characters pass through unchanged; everything else is
/// wrapped in single quotes with embedded `'` escaped as `'\''`.
///
/// This is THE quoting implementation — an earlier era had three local
/// copies whose safe-character sets silently diverged. Local copies
/// dodge review; don't reintroduce one.
pub(crate) fn posix_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = s.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(c, '_' | '-' | '.' | '/' | ':' | '=' | '+' | '@' | ',')
    });
    if safe {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}
