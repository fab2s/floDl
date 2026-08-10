//! What THIS box is still missing for the chosen door.
//!
//! The wizard used to compose a beautiful setup for a machine it had
//! asked nothing of, so every gap surfaced one at a time and in the
//! worst order: after credentials were minted, after a publish, and
//! sometimes only after a worker had already dialed. Every check here
//! was hit for real on this rig.

use std::path::Path;

use crate::util::platform;
use crate::util::system;

use super::Door;
use super::authorized_keys::sshd_listening;

// ── Preflight ───────────────────────────────────────────────────────────

/// One thing this setup needs, whether the box has it, and the command
/// that supplies it here.
#[derive(Debug, Clone)]
pub(super) struct Check {
    pub(super) what: String,
    pub(super) ok: bool,
    /// Platform-translated fix, absent when there is nothing to run
    /// (either it is already satisfied, or no command would be honest).
    pub(super) fix: Option<String>,
}

/// What the chosen door needs before any of this works, checked before
/// the artifacts are written.
///
/// The wizard used to compose a beautiful setup for a machine it never
/// asked a single question of, so its gaps surfaced one at a time and in
/// the worst order: after credentials were minted, after a publish, and
/// sometimes only after a worker had already dialed. Every gap below was
/// hit for real on this rig.
pub(super) fn preflight(
    door: Door,
    port: u16,
    served: &Path,
    plat: platform::Platform,
) -> Vec<Check> {
    let mut checks = Vec::new();

    // The daemon itself. On macOS it ships and is toggled, not installed.
    let have_sshd = system::has_command("sshd")
        || Path::new("/usr/sbin/sshd").exists()
        || Path::new("/usr/libexec/sshd-keygen-wrapper").exists();
    checks.push(Check {
        what: "an ssh daemon on this box".into(),
        ok: have_sshd,
        fix: (!have_sshd)
            .then(|| plat.sshd_package().and_then(|pkg| plat.install(&[pkg])))
            .flatten(),
    });

    // Something answering where workers will knock. This is the check
    // that catches the socket-activation trap below in its effect.
    // Debian hands the ssh listener to a socket unit, and while it holds
    // it the `Port` directive in sshd_config is IGNORED outright, so a
    // correct drop-in looks like it did nothing at all. That is the same
    // gap as "nothing is listening", so it is the same check with the
    // cause named rather than a second one repeating the fix.
    let listening = sshd_listening(port);
    let socket_owns = plat == platform::Platform::Debian && port != 22 && socket_activated();
    checks.push(Check {
        what: if socket_owns {
            format!(
                "something listening on port {port} — ssh.socket owns the \
                 listener, so sshd_config's `Port` is ignored until it is \
                 handed back"
            )
        } else {
            format!("something listening on port {port}")
        },
        ok: listening && !socket_owns,
        fix: (!listening || socket_owns)
            .then(|| plat.enable_sshd().join(" && "))
            .filter(|s| !s.is_empty()),
    });

    // SELinux refuses a non-standard ssh port with an error that never
    // says SELinux.
    if let Some(fix) = plat.allow_ssh_port(port) {
        checks.push(Check {
            what: format!("SELinux permits sshd on port {port}"),
            ok: false,
            fix: Some(fix),
        });
    }

    // The door's own tool runs on THIS side, as the forced command.
    match door {
        Door::B => {
            let have = system::has_command("rrsync");
            checks.push(Check {
                what: "`rrsync` (door b runs it as the forced command)".into(),
                ok: have,
                // Not simply "install rsync": RHEL ships rrsync as
                // non-executable documentation, so there the fix has to
                // place it too (see Platform::rrsync_fix).
                fix: (!have).then(|| plat.rrsync_fix()).flatten(),
            });
            let served_ok = served.is_dir();
            checks.push(Check {
                what: format!("the served directory {} exists", served.display()),
                ok: served_ok,
                fix: (!served_ok).then(|| {
                    "fdl publish <source> --bin <artifact>   # creates and fills it".to_string()
                }),
            });
        }
        Door::A => {
            let have = [
                "/usr/lib/openssh/sftp-server",
                "/usr/libexec/openssh/sftp-server",
                "/usr/libexec/sftp-server",
            ]
            .iter()
            .any(|p| Path::new(p).exists());
            checks.push(Check {
                what: "an sftp server (door a serves the data mount over it)".into(),
                ok: have,
                fix: (!have)
                    .then(|| plat.sshd_package().and_then(|p| plat.install(&[p])))
                    .flatten(),
            });
        }
        Door::Nologin => {}
    }

    checks
}

/// Whether systemd's ssh socket unit currently owns the listener.
/// Linux-only and best-effort: a missing systemctl answers "no".
fn socket_activated() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    std::process::Command::new("systemctl")
        .args(["is-active", "ssh.socket"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active")
        .unwrap_or(false)
}
