//! Who holds this port, and how to get rid of them.
//!
//! `Address already in use` is accurate and useless. Every cluster role
//! that binds a port can collide with a launcher left over from a killed
//! run, and the operator's next question is always the same two: which
//! process, and what kills it. Both answers are a `/proc` read away.
//!
//! The second one is not obvious, which is the reason this module exists.
//! `ddp-bench` runs under a `docker:` service, so a leftover launcher is
//! **PID 1 inside its own PID namespace**. Signals from outside do not
//! reach it, `sudo kill -9` fails, and the failure reads as a permissions
//! problem while being a namespace one. The remedy is `docker rm -f`, and
//! nothing in the error said so.
//!
//! # Advisory, never fatal
//!
//! Every entry point returns `Option` and every failure is `None`: another
//! user's socket, a stripped `/proc`, a non-Linux host. A diagnostic that
//! can itself fail must never turn a clear error into a confusing one, so
//! the caller appends the hint when there is one and prints exactly what
//! it printed before when there is not.
//!
//! # Linux only
//!
//! `/proc/net/tcp` and `/proc/<pid>/cgroup` are Linux interfaces. macOS
//! and Windows compile the stub, which always answers `None` (the OS
//! matrix runs this suite on both, and a platform-local API applied
//! unconditionally is a scheduled failure there).

/// A human-readable hint naming the process holding `port`, or `None`.
///
/// Append to a bind error; never a substitute for it.
pub(crate) fn port_holder_hint(port: u16) -> Option<String> {
    imp::port_holder_hint(port)
}

/// [`port_holder_hint`] as a printable suffix, gated on the bind error
/// actually being a collision.
///
/// The gate is not cosmetic. A bind can fail for reasons that have no
/// holder at all (`PermissionDenied` on a privileged port,
/// `AddrNotAvailable` for an address this host does not own), and
/// answering those with "held by ..." would send the operator hunting a
/// process that is not the problem. Only `AddrInUse` means somebody has
/// it.
///
/// Empty string when there is nothing to add, so a message with no
/// identifiable holder is byte-identical to what it was before.
pub(crate) fn hint_suffix(port: u16, kind: std::io::ErrorKind) -> String {
    if kind != std::io::ErrorKind::AddrInUse {
        return String::new();
    }
    match port_holder_hint(port) {
        Some(h) => format!("\n  {h}"),
        None => String::new(),
    }
}

/// The listening socket inode bound to `port`, from a `/proc/net/tcp`
/// body.
///
/// `st` `0A` is `TCP_LISTEN`; a connected socket to the same port is a
/// different thing and must not match. The address column is
/// `HEXADDR:HEXPORT`, so the port is parsed from hex, not decimal.
#[cfg(any(target_os = "linux", test))]
fn listen_inode(body: &str, port: u16) -> Option<u64> {
    for line in body.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        // sl local rem st tx:rx tr:when retrnsmt uid timeout inode
        if f.len() < 10 || f[3] != "0A" {
            continue;
        }
        let Some((_, hex_port)) = f[1].rsplit_once(':') else {
            continue;
        };
        if u16::from_str_radix(hex_port, 16).ok() != Some(port) {
            continue;
        }
        if let Ok(inode) = f[9].parse::<u64>() {
            return Some(inode);
        }
    }
    None
}

/// The 12-hex-digit container id from a `/proc/<pid>/cgroup` body.
///
/// Matches cgroup v2's systemd shape (`.../docker-<64hex>.scope`) and
/// v1's (`/docker/<64hex>`), taking the short id `docker rm` accepts.
#[cfg(any(target_os = "linux", test))]
fn container_id(cgroup: &str) -> Option<String> {
    let at = cgroup.find("docker-").map(|i| i + 7).or_else(|| {
        cgroup
            .find("/docker/")
            .map(|i| i + 8)
            .filter(|_| !cgroup.contains("docker-"))
    })?;
    let id: String = cgroup[at..]
        .chars()
        .take_while(|c| c.is_ascii_hexdigit())
        .take(12)
        .collect();
    (id.len() == 12).then_some(id)
}

/// The port is held, but by a process this namespace cannot see.
///
/// Not a failure to diagnose: it IS the diagnosis. `/proc/net/tcp` is
/// per **network** namespace and `/proc/<pid>` is per **PID** namespace,
/// and flodl's own compose services run `network_mode: host` without
/// sharing PIDs. So a launcher inside a `docker:` service sees the
/// squatter's socket and can never see its process, which is exactly the
/// collision an operator hits with a leftover run. Saying so beats
/// saying nothing, which is what a bare `None` would do in the one case
/// this module exists for.
#[cfg(any(target_os = "linux", test))]
fn describe_unseen(port: u16) -> String {
    format!(
        "port {port} is held, but by a process outside this PID namespace. \
         A containerised run (a `docker:` service) shares the host network \
         and not its processes, so the holder is almost certainly another \
         container or another user. From the HOST: \
         `ss -ltnp | grep :{port}`, then `docker rm -f <id>` if the PID's \
         `/proc/<pid>/cgroup` names one (a container's PID 1 ignores signals \
         from outside, so `kill` will fail)"
    )
}

/// Render the hint once the facts are known.
#[cfg(any(target_os = "linux", test))]
fn describe(pid: u32, name: &str, container: Option<&str>) -> String {
    match container {
        Some(cid) => format!(
            "held by PID {pid} ({name}) in container {cid}. \
             That process is PID 1 inside its own namespace, so no signal \
             from outside reaches it (`sudo kill` fails too): `docker rm -f {cid}`"
        ),
        None => format!("held by PID {pid} ({name}): `kill -9 {pid}`"),
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{container_id, describe, describe_unseen, listen_inode};

    pub(super) fn port_holder_hint(port: u16) -> Option<String> {
        // Both families: a v4 listener and a v6 one are equally likely to
        // be the squatter, and they live in separate tables.
        let inode = ["/proc/net/tcp", "/proc/net/tcp6"]
            .iter()
            .filter_map(|p| std::fs::read_to_string(p).ok())
            .find_map(|body| listen_inode(&body, port))?;
        // Socket found but no PID: the holder is in another PID namespace.
        // That is the containerised-collision case, and it gets its own
        // message rather than a silent None.
        let Some(pid) = pid_owning(inode) else {
            return Some(describe_unseen(port));
        };
        let name = process_name(pid).unwrap_or_else(|| "unknown".into());
        let cgroup = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).unwrap_or_default();
        Some(describe(pid, &name, container_id(&cgroup).as_deref()))
    }

    /// Scan `/proc/*/fd` for the socket. Only processes this uid may
    /// inspect are visible, which is the common case (the squatter is
    /// almost always another run by the same operator) and a clean `None`
    /// otherwise.
    fn pid_owning(inode: u64) -> Option<u32> {
        let want = format!("socket:[{inode}]");
        for entry in std::fs::read_dir("/proc").ok()?.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(fds) = std::fs::read_dir(entry.path().join("fd")) else {
                continue;
            };
            for fd in fds.flatten() {
                if std::fs::read_link(fd.path()).is_ok_and(|t| t.to_string_lossy() == want) {
                    return Some(pid);
                }
            }
        }
        None
    }

    /// The basename of `argv[0]`. `cmdline` is NUL-separated.
    fn process_name(pid: u32) -> Option<String> {
        let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
        let argv0 = raw.split(|b| *b == 0).next()?;
        let s = String::from_utf8_lossy(argv0);
        Some(s.rsplit('/').next().unwrap_or(&s).to_string())
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    pub(super) fn port_holder_hint(_port: u16) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim `/proc/net/tcp` shape, with a LISTEN row on 1337
    /// (0x0539) and an ESTABLISHED row on the same port that must not
    /// match: a connection TO the port is not the thing holding it.
    const TCP: &str = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:0539 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4212345 1 0000
   1: 0100007F:0539 0100007F:B3C4 01 00000000:00000000 00:00000000 00000000  1000        0 9999999 1 0000
   2: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4200001 1 0000
";

    #[test]
    fn finds_the_listening_socket_and_ignores_a_connection_to_it() {
        assert_eq!(listen_inode(TCP, 1337), Some(4_212_345));
    }

    #[test]
    fn matches_the_port_in_hex_not_decimal() {
        // 0x1F90 is 8080. A decimal reading of "1F90" parses as nothing,
        // and a naive reading of "0539" as decimal would be 539.
        assert_eq!(listen_inode(TCP, 8080), Some(4_200_001));
        assert_eq!(listen_inode(TCP, 539), None);
    }

    #[test]
    fn an_unheld_port_is_none() {
        assert_eq!(listen_inode(TCP, 1339), None);
    }

    #[test]
    fn the_header_row_is_not_parsed_as_a_socket() {
        assert_eq!(
            listen_inode("sl local_address rem_address st\n", 1337),
            None
        );
    }

    #[test]
    fn container_id_from_cgroup_v2_systemd_scope() {
        let v2 = "0::/system.slice/docker-f060e0cd3b0db67371447224e9b42a57dd366875c0bee086f6f1303c20ab67a2.scope\n";
        assert_eq!(container_id(v2).as_deref(), Some("f060e0cd3b0d"));
    }

    #[test]
    fn container_id_from_cgroup_v1() {
        let v1 = "11:cpu,cpuacct:/docker/aaca59dfb5ad9e1c2f3b4a5d6e7f8091a2b3c4d5\n";
        assert_eq!(container_id(v1).as_deref(), Some("aaca59dfb5ad"));
    }

    #[test]
    fn a_host_process_has_no_container_id() {
        assert_eq!(container_id("0::/user.slice/user-1000.slice\n"), None);
        assert_eq!(container_id(""), None);
    }

    /// The container branch must name `docker rm -f`, because that is the
    /// half an operator cannot guess: `kill` on a namespace's PID 1 fails
    /// in a way that looks like a permissions problem.
    #[test]
    fn the_container_hint_names_docker_not_kill() {
        let h = describe(32129, "ddp-bench", Some("f060e0cd3b0d"));
        assert!(h.contains("docker rm -f f060e0cd3b0d"), "{h}");
        assert!(h.contains("PID 1"), "must explain WHY kill fails: {h}");
        assert!(
            !h.contains("kill -9"),
            "must not suggest a kill that fails: {h}"
        );
    }

    /// A bind that failed for a reason with no holder must not claim one.
    #[test]
    fn only_a_collision_gets_a_hint() {
        use std::io::ErrorKind;
        assert_eq!(hint_suffix(1337, ErrorKind::PermissionDenied), "");
        assert_eq!(hint_suffix(1337, ErrorKind::AddrNotAvailable), "");
    }

    /// An unidentifiable holder leaves the message exactly as it was.
    #[test]
    fn an_unknown_holder_adds_nothing() {
        // Port 0 is never a listening socket, so the lookup finds nothing
        // even on Linux, and the non-Linux stub returns None regardless.
        assert_eq!(hint_suffix(0, std::io::ErrorKind::AddrInUse), "");
    }

    /// End-to-end against a socket this process really holds.
    ///
    /// Every other test here feeds synthetic `/proc` text, which proves
    /// the parsers and nothing about the chain: `/proc/net/tcp` ->  inode
    /// -> scan `/proc/*/fd` -> pid -> cmdline -> cgroup. Any link can be
    /// wrong (a kernel that formats a column differently, an fd scan that
    /// cannot read its own process) and every unit test would still pass.
    ///
    /// `#[ignore]` because it walks all of `/proc` and depends on the
    /// host allowing that. Run it when touching this module:
    ///
    /// ```text
    /// cargo test -p flodl identifies_a_socket_this_process_holds -- --ignored --nocapture
    /// ```
    /// The motivating scenario, end to end: a host-networked container
    /// looking at a port held by a DIFFERENT container. `cuda` is
    /// `network_mode: host` with its own PID namespace, so the socket is
    /// visible and the process is not, and this must produce the
    /// unseen-holder message rather than nothing. Port 2222 is
    /// `cuda-rank`'s sshd.
    ///
    /// ```text
    /// docker compose run --rm cuda sh -c \
    ///   'cargo test -p flodl --features cuda names_a_holder_in_another_container -- --ignored --nocapture'
    /// ```
    // The probes read /proc; on any other platform the stub answers None
    // and the assertions would fail for a reason that is not a defect.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "environment probe: needs a host-networked container + a foreign holder"]
    fn names_a_holder_in_another_container() {
        let hint = port_holder_hint(2222);
        println!("port 2222 -> {hint:?}");
        let hint = hint.expect("2222 is held by cuda-rank's sshd; the socket must be seen");
        assert!(
            hint.contains("outside this PID namespace") || hint.contains("held by PID"),
            "must diagnose something: {hint}"
        );
    }

    // The probes read /proc; on any other platform the stub answers None
    // and the assertions would fail for a reason that is not a defect.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "environment probe: scans /proc, needs a permissive host"]
    fn identifies_a_socket_this_process_holds() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let hint = port_holder_hint(port);
        println!("port {port} -> {hint:?}");
        let hint = hint.expect("this process holds the port, so it must be found");
        let me = std::process::id();
        assert!(
            hint.contains(&format!("PID {me}")),
            "must name THIS pid ({me}): {hint}"
        );
    }

    /// The unseen-holder message must still be ACTIONABLE: it is emitted
    /// exactly in the containerised collision this module exists for.
    #[test]
    fn the_unseen_hint_points_at_the_host_and_explains_why_kill_fails() {
        let h = describe_unseen(1337);
        assert!(h.contains("ss -ltnp | grep :1337"), "{h}");
        assert!(h.contains("docker rm -f"), "{h}");
        assert!(h.contains("PID namespace"), "{h}");
    }

    #[test]
    fn the_host_hint_names_kill() {
        let h = describe(32129, "ddp-bench", None);
        assert!(h.contains("kill -9 32129"), "{h}");
        assert!(!h.contains("docker"), "{h}");
    }
}
