//! The confirm-gated `authorized_keys` install.
//!
//! Only the wizard's own line is ever touched (identity is the public
//! key material itself), `/etc/ssh` never is, and the rewrite is atomic.
//! Consent is explicit: the prompt or `--install-key`. `--yes` does not
//! count — it accepts ordinary defaults, and a security-relevant
//! mutation is not one.

use std::fs;
use std::path::{Path, PathBuf};

use crate::builtins::JoinConfigArgs;
use crate::context::home_dir;
use crate::style;
use crate::util::prompt;

use super::{ChangeKind, Changes, PLACEHOLDER_PUB};

// ── authorized_keys install ─────────────────────────────────────────────

/// What the install offer decided and did. `Skipped` carries the why so
/// the report can say it; refusals are ordinary `Err`s (the wizard has
/// already produced every artifact, but a half-touched authorized_keys
/// must be loud).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum InstallAction {
    Installed,
    Replaced,
    AlreadyPresent,
    Skipped(String),
}

/// The confirm-gated install: append (or replace) the wizard's own line
/// in the INVOKING user's `~/.ssh/authorized_keys`. Only that line is
/// ever touched — identity is the public key material itself — and
/// `/etc/ssh` never is (the dedicated-user hardening stays in the
/// notes). The friction this removes is real: the composed guardrail
/// line is the artifact most likely to be mangled by hand, by exactly
/// the audience least equipped to debug an sshd refusal.
pub(super) fn install_authorized_line(
    cli: &JoinConfigArgs,
    changes: &mut Changes,
    line: &str,
    sshd_port: u16,
) -> Result<InstallAction, String> {
    if cli.install_key && cli.no_install_key {
        return Err("--install-key and --no-install-key contradict each other".into());
    }
    if cli.dry_run {
        return dry_install_verdict(cli, changes, line);
    }
    // Consent to touch authorized_keys is EXPLICIT: the tty prompt or
    // `--install-key`. `--yes` deliberately does not count — it accepts
    // ordinary defaults, and a security-relevant mutation is not one.
    let wanted = if cli.install_key {
        true
    } else if cli.no_install_key {
        false
    } else if cli.yes || !prompt::has_tty() {
        return Ok(InstallAction::Skipped(
            "installing needs explicit consent: the prompt, or --install-key".to_string(),
        ));
    } else {
        prompt::ask_yn(
            "install the guardrailed line into this user's \
             ~/.ssh/authorized_keys now? (only the wizard's own line is \
             ever touched)",
            true,
        )
    };
    if !wanted {
        return Ok(InstallAction::Skipped("declined".to_string()));
    }

    // Default: the invoking user's own file. `--authorized-keys` names a
    // different door, for the setups the default cannot reach at all —
    // an sshd in a container (its key file is a bind mount, so the
    // in-container run hits a read-only filesystem and the host run
    // writes a file no sshd reads), or a host with
    // `AuthorizedKeysFile /etc/ssh/authorized_keys.d/%u`. The uid
    // already bounds what this can touch; what the refusal below keeps
    // is the promise that the wizard never edits system sshd config.
    let ak_path = match cli.authorized_keys.as_deref() {
        Some(p) => {
            // `~/` names the invoking user's home, matching how
            // `inherit-from:` reads a path in overlay.rs.
            let p = match p.strip_prefix("~/") {
                Some(rest) => home_dir().join(rest),
                None => PathBuf::from(p),
            };
            if p.starts_with("/etc/ssh") {
                return Err(format!(
                    "{} is system sshd configuration — the wizard installs \
                     door keys, never /etc/ssh. Install it there by hand \
                     (the line is in the install notes)",
                    p.display(),
                ));
            }
            p
        }
        None => home_dir().join(".ssh").join("authorized_keys"),
    };
    let ssh_dir = ak_path
        .parent()
        .ok_or("the authorized_keys path has no parent directory")?
        .to_path_buf();
    // The 0700 rule is about the invoking user's own `~/.ssh`, which
    // sshd's StrictModes checks and which the wizard may have to create.
    // A named door lives in a layout the operator already owns — it may
    // be a bind-mount source, or a shared directory like /tmp that must
    // not be narrowed — so its parent is left exactly as found.
    if cli.authorized_keys.is_none() {
        if !ssh_dir.is_dir() {
            fs::create_dir_all(&ssh_dir)
                .map_err(|e| format!("cannot create {}: {e}", ssh_dir.display()))?;
            set_mode(&ssh_dir, 0o700)?;
        } else {
            fix_perms_confirmed(cli, &ssh_dir, 0o700)?;
        }
    } else if !ssh_dir.is_dir() {
        return Err(format!(
            "{} does not exist — create the directory holding the \
             authorized_keys file first",
            ssh_dir.display(),
        ));
    }
    // sshd refuses to follow surprises here and so does the wizard: a
    // symlinked authorized_keys is someone's deliberate setup, not a
    // file to rewrite through.
    if ak_path.is_symlink() {
        return Err(format!(
            "{} is a symlink — install the line manually (it is in the \
             install notes)",
            ak_path.display(),
        ));
    }

    let ak_existed = ak_path.is_file();
    let content = if ak_existed {
        fix_perms_confirmed(cli, &ak_path, 0o600)?;
        fs::read_to_string(&ak_path)
            .map_err(|e| format!("cannot read {}: {e}", ak_path.display()))?
    } else {
        String::new()
    };

    let (new_content, outcome) = upsert_authorized_line(&content, line)?;
    if outcome == UpsertOutcome::Identical {
        changes.note(
            &ak_path,
            ChangeKind::Unchanged,
            "authorized_keys (wizard's line)",
        );
        return Ok(InstallAction::AlreadyPresent);
    }
    if outcome == UpsertOutcome::Replaced {
        let confirmed = cli.install_key
            || (prompt::has_tty()
                && prompt::ask_yn(
                    "the key is already installed with DIFFERENT options — \
                     replace that line with the wizard's?",
                    true,
                ));
        if !confirmed {
            return Ok(InstallAction::Skipped(
                "the key is present with different options; left alone".to_string(),
            ));
        }
    }

    // Atomic in place: temp file beside it (same filesystem), 0600
    // before any bytes land, rename over. A crash cannot leave a
    // half-written authorized_keys behind.
    let tmp = ssh_dir.join(".authorized_keys.fdl-tmp");
    fs::write(&tmp, &new_content).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    set_mode(&tmp, 0o600)?;
    fs::rename(&tmp, &ak_path)
        .map_err(|e| format!("cannot move the new authorized_keys into place: {e}"))?;
    changes.note(
        &ak_path,
        if ak_existed {
            ChangeKind::Update
        } else {
            ChangeKind::Create
        },
        "authorized_keys (wizard's line)",
    );

    // Best-effort floor check: the door is installed, but is anyone
    // listening where workers will knock?
    if !sshd_listening(sshd_port) {
        let hint = if cfg!(target_os = "macos") {
            " (on macOS: System Settings > General > Sharing > Remote Login)"
        } else {
            ""
        };
        eprintln!(
            "{}",
            style::dim(&format!(
                "fdl join-config: nothing seems to be listening on port \
                 {sshd_port} — the line is installed, but workers cannot \
                 dial until sshd is up{hint}",
            )),
        );
    }

    Ok(match outcome {
        UpsertOutcome::Appended => InstallAction::Installed,
        UpsertOutcome::Replaced => InstallAction::Replaced,
        UpsertOutcome::Identical => unreachable!("handled above"),
    })
}

/// The `--dry-run` half of the install offer: decide consent from the
/// flags alone (a dry run never prompts) and classify what an apply
/// would do to the file, touching nothing — no mkdir, no perms fixes.
pub(super) fn dry_install_verdict(
    cli: &JoinConfigArgs,
    changes: &mut Changes,
    line: &str,
) -> Result<InstallAction, String> {
    if cli.no_install_key {
        return Ok(InstallAction::Skipped(
            "declined (--no-install-key)".to_string(),
        ));
    }
    if !cli.install_key {
        return Ok(InstallAction::Skipped(
            "installing needs explicit consent: the prompt on apply, or --install-key".to_string(),
        ));
    }
    let ak_path = match cli.authorized_keys.as_deref() {
        Some(p) => {
            let p = match p.strip_prefix("~/") {
                Some(rest) => home_dir().join(rest),
                None => PathBuf::from(p),
            };
            if p.starts_with("/etc/ssh") {
                return Err(format!(
                    "{} is system sshd configuration — the wizard installs \
                     door keys, never /etc/ssh. Install it there by hand \
                     (the line is in the install notes)",
                    p.display(),
                ));
            }
            p
        }
        None => home_dir().join(".ssh").join("authorized_keys"),
    };
    let exists = ak_path.is_file();
    let kind = if exists {
        ChangeKind::Update
    } else {
        ChangeKind::Create
    };
    // A key an apply would mint cannot be compared against the file
    // yet; the apply would append its fresh line.
    if line.contains(PLACEHOLDER_PUB) {
        changes.note(&ak_path, kind, "authorized_keys (wizard's line)");
        return Ok(InstallAction::Installed);
    }
    let content = if exists {
        fs::read_to_string(&ak_path)
            .map_err(|e| format!("cannot read {}: {e}", ak_path.display()))?
    } else {
        String::new()
    };
    let (_, outcome) = upsert_authorized_line(&content, line)?;
    Ok(match outcome {
        UpsertOutcome::Identical => {
            changes.note(
                &ak_path,
                ChangeKind::Unchanged,
                "authorized_keys (wizard's line)",
            );
            InstallAction::AlreadyPresent
        }
        UpsertOutcome::Replaced => {
            changes.note(
                &ak_path,
                ChangeKind::Update,
                "authorized_keys (wizard's line)",
            );
            InstallAction::Replaced
        }
        UpsertOutcome::Appended => {
            changes.note(&ak_path, kind, "authorized_keys (wizard's line)");
            InstallAction::Installed
        }
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UpsertOutcome {
    Appended,
    Replaced,
    Identical,
}

/// Append `line`, or replace the one existing line that carries the
/// same public key material. Identity is `keytype + base64` — options
/// and comment may differ, the key bytes cannot. Every other line is
/// preserved byte for byte.
pub(super) fn upsert_authorized_line(
    content: &str,
    line: &str,
) -> Result<(String, UpsertOutcome), String> {
    let wanted =
        key_material(line).ok_or("the composed authorized_keys line carries no key material")?;
    let mut out = String::with_capacity(content.len() + line.len() + 2);
    let mut outcome = UpsertOutcome::Appended;
    for existing in content.lines() {
        if key_material(existing) == Some(wanted) {
            if existing.trim() == line.trim() {
                return Ok((content.to_string(), UpsertOutcome::Identical));
            }
            out.push_str(line);
            outcome = UpsertOutcome::Replaced;
        } else {
            out.push_str(existing);
        }
        out.push('\n');
    }
    if outcome == UpsertOutcome::Appended {
        out.push_str(line);
        out.push('\n');
    }
    Ok((out, outcome))
}

/// The `keytype base64` pair of an authorized_keys line, skipping any
/// leading options field. Options may contain quoted commas
/// (`command="a,b"`), so the scan is quote-aware: the key type is the
/// first whitespace-separated token outside quotes that looks like one.
pub(super) fn key_material(line: &str) -> Option<(&str, &str)> {
    let mut rest = line.trim();
    if rest.is_empty() || rest.starts_with('#') {
        return None;
    }
    loop {
        let mut fields = rest.splitn(2, char::is_whitespace);
        let first = fields.next()?;
        let tail = fields.next().unwrap_or("").trim_start();
        if first.starts_with("ssh-") || first.starts_with("ecdsa-") || first.starts_with("sk-") {
            let key = tail.split_whitespace().next()?;
            return Some((first, key));
        }
        // `first` is the options field — but splitn cut it at the first
        // space, which may sit INSIDE quotes. Walk to the real end of
        // the options (first unquoted whitespace) and retry from there.
        let mut in_quotes = false;
        let mut cut = None;
        for (i, c) in rest.char_indices() {
            match c {
                '"' => in_quotes = !in_quotes,
                c if c.is_whitespace() && !in_quotes => {
                    cut = Some(i);
                    break;
                }
                _ => {}
            }
        }
        rest = rest[cut?..].trim_start();
    }
}

/// Report a wrong mode with the exact fix and apply it on confirm —
/// never silently. `--install-key` is standing consent for what the
/// install needs; a non-tty run without it refuses loudly rather than
/// leaving a lax authorized_keys in play.
#[cfg(unix)]
pub(super) fn fix_perms_confirmed(
    cli: &JoinConfigArgs,
    path: &Path,
    mode: u32,
) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let current = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .permissions()
        .mode()
        & 0o777;
    if current == mode {
        return Ok(());
    }
    let question = format!(
        "{} is mode {current:03o}, sshd wants {mode:03o} — apply `chmod \
         {mode:o} {}`?",
        path.display(),
        path.display(),
    );
    let apply = cli.install_key || (prompt::has_tty() && prompt::ask_yn(&question, true));
    if !apply {
        return Err(format!(
            "{} stays mode {current:03o} — sshd will refuse the key until \
             it is {mode:03o}",
            path.display(),
        ));
    }
    set_mode(path, mode)
}

#[cfg(not(unix))]
pub(super) fn fix_perms_confirmed(
    _cli: &JoinConfigArgs,
    _path: &Path,
    _mode: u32,
) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
pub(super) fn set_mode(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .map_err(|e| format!("cannot chmod {}: {e}", path.display()))
}

#[cfg(not(unix))]
pub(super) fn set_mode(_path: &Path, _mode: u32) -> Result<(), String> {
    Ok(())
}

/// Is anything accepting on the sshd port workers will dial? Loopback
/// suffices as a heuristic — sshd binds all interfaces by default.
pub(super) fn sshd_listening(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        std::time::Duration::from_millis(300),
    )
    .is_ok()
}
