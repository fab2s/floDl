//! The scheme-plus-path grammar the artifact specs share.
//!
//! Everything a box needs before it can train answers one question, and
//! the answers differ only in size, mutability and public availability:
//! the dataset source root, libtorch, the training source. That makes
//! the *grammar* real shared structure while the resolvers stay
//! genuinely separate — mounting a tree, unpacking an archive and
//! fetching a checkout have no common shape worth abstracting. This
//! module is the shared half and nothing else.
//!
//! Errors here carry only the reason. The caller names the field and
//! spells out its accepted forms, because both differ per artifact and a
//! message that says `data_source` when the operator mistyped `source`
//! sends them to the wrong line.

/// Split `<scheme>://<rest>`. A value with no `://` carries no
/// transport, which each field answers for itself: for `data_path` a
/// bare value is a path already mounted, for a source spec it is an
/// error naming the transports.
pub fn split_scheme(spec: &str) -> (Option<&str>, &str) {
    match spec.split_once("://") {
        Some((scheme, rest)) => (Some(scheme), rest),
        None => (None, spec),
    }
}

/// An ssh endpoint, in the spelling `sshfs` and `rsync` both take on the
/// command line — and the one `/proc/mounts` reports back, which is what
/// lets the already-mounted check be a string compare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshTarget {
    /// `[user@]host:/abs/path`.
    pub remote: String,
    /// Non-default ssh port, when the spec named one.
    pub port: Option<u16>,
}

/// Parse `[user@]host[:port]/abs/path`, and the scp spelling
/// `[user@]host:/abs/path` for the same thing — both sshfs and rsync
/// take the second, so refusing it would be a gratuitous trap.
pub fn parse_ssh_target(rest: &str) -> Result<SshTarget, &'static str> {
    let (user, hostpart) = match rest.split_once('@') {
        Some(("", _)) => return Err("empty user before `@`"),
        Some((u, h)) => (Some(u), h),
        None => (None, rest),
    };

    // Split host from path at whichever delimiter comes first. A `:`
    // followed by digits is a port; a `:` followed by `/` is the scp
    // separator — and a port may itself be followed by the scp colon
    // (`host:2222:/abs/path`), which is exactly what the documented
    // grammar `[user@]host[:port]:/abs/path` produces when both parts
    // are used at once. Refusing that spelling would make the docs'
    // own grammar a parse error precisely on the guardrail recipe's
    // advice (a non-standard external port).
    let colon = hostpart.find(':');
    let slash = hostpart.find('/');
    let (host, port, path) = match (colon, slash) {
        (Some(c), s) if s.is_none_or(|s| c < s) => {
            let after = &hostpart[c + 1..];
            if after.starts_with('/') {
                (&hostpart[..c], None, after)
            } else {
                let end = after.find('/').ok_or("no remote path")?;
                let port_str = after[..end].strip_suffix(':').unwrap_or(&after[..end]);
                let port = port_str.parse::<u16>().map_err(|_| "port is not a number")?;
                (&hostpart[..c], Some(port), &after[end..])
            }
        }
        (_, Some(s)) => (&hostpart[..s], None, &hostpart[s..]),
        (_, None) => return Err("no remote path"),
    };
    if host.is_empty() {
        return Err("empty host");
    }
    if path.len() < 2 {
        return Err("the remote path must be absolute");
    }
    Ok(SshTarget {
        remote: match user {
            Some(u) => format!("{u}@{host}:{path}"),
            None => format!("{host}:{path}"),
        },
        port,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scheme_is_only_a_scheme_with_its_separator() {
        assert_eq!(split_scheme("sshfs://exa/data"), (Some("sshfs"), "exa/data"));
        assert_eq!(split_scheme("/flodl/data"), (None, "/flodl/data"));
        // A colon alone is not a scheme: `host:/path` is the scp
        // spelling, and reading it as one would swallow the host.
        assert_eq!(split_scheme("exa:/flodl/data"), (None, "exa:/flodl/data"));
    }

    #[test]
    fn an_ssh_target_parses_all_four_spellings() {
        assert_eq!(
            parse_ssh_target("flodl@exa:/flodl/data").unwrap(),
            SshTarget { remote: "flodl@exa:/flodl/data".into(), port: None },
        );
        assert_eq!(
            parse_ssh_target("exa/flodl/data").unwrap(),
            SshTarget { remote: "exa:/flodl/data".into(), port: None },
        );
        assert_eq!(
            parse_ssh_target("flodl@exa:2222/flodl/data").unwrap(),
            SshTarget { remote: "flodl@exa:/flodl/data".into(), port: Some(2222) },
        );
        // What the documented grammar `[user@]host[:port]:/abs/path`
        // literally produces with both parts in play — the spelling an
        // operator on a non-standard port will type first.
        assert_eq!(
            parse_ssh_target("flodl@exa:2222:/flodl/data").unwrap(),
            SshTarget { remote: "flodl@exa:/flodl/data".into(), port: Some(2222) },
        );
    }

    #[test]
    fn an_ssh_target_says_why_it_refused() {
        for (spec, why) in [
            ("exa", "no remote path"),
            ("exa:2222", "no remote path"),
            ("exa:banana/data", "port is not a number"),
            ("@exa:/flodl/data", "empty user before `@`"),
            (":/flodl/data", "empty host"),
            ("/flodl/data", "empty host"),
            ("exa:/", "the remote path must be absolute"),
        ] {
            assert_eq!(parse_ssh_target(spec), Err(why), "for {spec}");
        }
    }
}
