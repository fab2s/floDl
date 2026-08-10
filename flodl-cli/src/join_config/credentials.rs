//! Credentials and the farm's own config: the join key, the admission
//! token, the overlay that carries it, and reading a farm's shape back.
//!
//! Token surgery is byte-preserving on purpose: a user's overlay is
//! mostly comments, and a serde round-trip would delete every one of
//! them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::builtins::JoinConfigArgs;
use crate::util::prompt;

use super::publish_recipe::package_name;
use super::render::render_overlay_scaffold;
use super::{ChangeKind, Changes, Door, KeyAction, OverlayAction, PLACEHOLDER_TOKEN};

pub(super) fn resolve_label(cli: &JoinConfigArgs) -> Result<String, String> {
    if let Some(l) = &cli.label {
        return Ok(l.clone());
    }
    if let Ok(env) = std::env::var("FDL_ENV")
        && !env.trim().is_empty()
    {
        return Ok(env.trim().to_string());
    }
    Err(
        "a farm needs a label: `fdl join-config <label>` (or target an \
         existing overlay: `fdl @<label> join-config`)"
            .to_string(),
    )
}

/// Labels become filenames (`fdl.<label>.yml`, `.fdl/<label>/`), so the
/// charset is the portable-filename one. Also the validation `fdl ui`
/// applies to env names arriving in a query string.
pub(crate) fn validate_label(label: &str) -> Result<(), String> {
    let ok = !label.is_empty()
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "farm label `{label}` — letters, digits, `-` and `_` only (it \
             names fdl.<label>.yml and .fdl/<label>/)"
        ))
    }
}

/// Ask, honoring the flag twins: `--regen` answers yes to regeneration
/// prompts, `--yes` accepts every default, and a non-tty run without
/// the deciding flag errors loudly instead of hanging.
pub(super) fn confirm(cli: &JoinConfigArgs, question: &str) -> Result<bool, String> {
    if cli.yes {
        return Ok(true);
    }
    if !prompt::has_tty() {
        return Err(format!(
            "non-interactive run needs a decision: {question} \
             (pass --yes to accept, or run in a terminal)"
        ));
    }
    Ok(prompt::ask_yn(question, true))
}

pub(super) fn ensure_key(
    cli: &JoinConfigArgs,
    changes: &mut Changes,
    key_path: &Path,
    label: &str,
) -> Result<KeyAction, String> {
    let exists = key_path.is_file() && key_path.with_extension("pub").is_file();
    if exists {
        // --yes reuses (regeneration is opt-in, never a default), and
        // neither a non-tty run nor a dry run can be asked, so all
        // three fall through to reuse.
        let regen = if cli.regen {
            true
        } else if cli.yes || cli.dry_run || !prompt::has_tty() {
            false
        } else {
            prompt::ask_yn(
                &format!(
                    "a join key already exists for `{label}` — regenerate it \
                     for a new farm instantiation? (workers holding the old \
                     key stop being admitted)"
                ),
                false,
            )
        };
        if !regen {
            changes.note(key_path, ChangeKind::Unchanged, "join key pair");
            return Ok(KeyAction::Reused);
        }
        changes.note(key_path, ChangeKind::Update, "join key pair (regenerated)");
        if cli.dry_run {
            return Ok(KeyAction::Regenerated);
        }
        fs::remove_file(key_path).ok();
        fs::remove_file(key_path.with_extension("pub")).ok();
        generate_key(key_path, label)?;
        return Ok(KeyAction::Regenerated);
    }
    changes.note(key_path, ChangeKind::Create, "join key pair");
    if cli.dry_run {
        return Ok(KeyAction::Generated);
    }
    generate_key(key_path, label)?;
    Ok(KeyAction::Generated)
}

pub(super) fn generate_key(key_path: &Path, label: &str) -> Result<(), String> {
    let out = Command::new("ssh-keygen")
        .args(["-t", "ed25519", "-N", "", "-C"])
        .arg(format!("flodl-join-{label}"))
        .arg("-f")
        .arg(key_path)
        .output()
        .map_err(|e| {
            format!(
                "cannot run ssh-keygen ({e}) — it ships with OpenSSH, which \
                 the join sshd needs anyway"
            )
        })?;
    if !out.status.success() {
        return Err(format!(
            "ssh-keygen failed: {}",
            String::from_utf8_lossy(&out.stderr).trim(),
        ));
    }
    // ssh-keygen writes 0600/0644 itself; assert the private side anyway
    // so a weird umask cannot leave a lax key behind.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(key_path, fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Token + overlay state machine. Returns the effective token and what
/// happened to the overlay file.
pub(super) fn ensure_overlay(
    cli: &JoinConfigArgs,
    changes: &mut Changes,
    overlay_path: &Path,
    label: &str,
    root: &Path,
    cmd_hint: Option<&str>,
) -> Result<(String, OverlayAction), String> {
    if !overlay_path.is_file() {
        if cli.dry_run {
            changes.note(overlay_path, ChangeKind::Create, "farm overlay");
            return Ok((PLACEHOLDER_TOKEN.to_string(), OverlayAction::Scaffolded));
        }
        let token = fresh_token()?;
        let scaffold = render_overlay_scaffold(label, &token, root, cmd_hint);
        changes.write(overlay_path, &scaffold, "farm overlay")?;
        return Ok((token, OverlayAction::Scaffolded));
    }
    let content = fs::read_to_string(overlay_path)
        .map_err(|e| format!("cannot read {}: {e}", overlay_path.display()))?;
    match find_token_line(&content) {
        Some(old) => {
            // Same non-question policy as the key: a dry run reads the
            // consent flags as given and never asks.
            let regen = if cli.regen {
                true
            } else if cli.yes || cli.dry_run || !prompt::has_tty() {
                false
            } else {
                prompt::ask_yn(
                    &format!(
                        "`{label}` already carries a token — mint a fresh one \
                         for a new farm instantiation? (workers holding the \
                         old one stop being admitted)"
                    ),
                    false,
                )
            };
            if !regen {
                changes.note(overlay_path, ChangeKind::Unchanged, "farm overlay");
                return Ok((old, OverlayAction::TokenReused));
            }
            if cli.dry_run {
                changes.note(
                    overlay_path,
                    ChangeKind::Update,
                    "farm overlay (token regenerated)",
                );
                return Ok((PLACEHOLDER_TOKEN.to_string(), OverlayAction::TokenReplaced));
            }
            let token = fresh_token()?;
            let replaced = replace_token_line(&content, &token)
                .ok_or("token line vanished between read and replace")?;
            changes.write(overlay_path, &replaced, "farm overlay (token regenerated)")?;
            Ok((token, OverlayAction::TokenReplaced))
        }
        None => {
            // A user-authored overlay without a token: the wizard does
            // not edit prose it does not own — nested surgical inserts
            // into hand-written yml guess too much. The snippet is in
            // the report.
            changes.note(
                overlay_path,
                ChangeKind::Unchanged,
                "farm overlay (user-authored, not edited)",
            );
            let token = if cli.dry_run {
                PLACEHOLDER_TOKEN.to_string()
            } else {
                fresh_token()?
            };
            Ok((token, OverlayAction::SnippetPrinted))
        }
    }
}

/// A fresh 32-hex credential from OS entropy. Same shape as the publish
/// nonce, but this one IS secret (it is the admission token), so there
/// is no fallback: a credential mint with no entropy has nothing honest
/// to fall back to, and the syscall failing is a refusal, not a
/// degradation. Also mints `fdl ui`'s per-session token.
pub(crate) fn fresh_token() -> Result<String, String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|e| format!("cannot draw OS entropy for the token: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// First `token:` line's value, however deep it sits. The wizard only
/// ever needs the join token and a farm overlay carries exactly one.
pub(super) fn find_token_line(content: &str) -> Option<String> {
    for line in content.lines() {
        let t = line.trim_start();
        if t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("token:") {
            let v = rest.trim().trim_matches('"').trim_matches('\'');
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Replace the first uncommented `token:` value, byte-preserving
/// everything else (indentation, comments, the rest of the file).
pub(super) fn replace_token_line(content: &str, new_token: &str) -> Option<String> {
    let mut out = String::with_capacity(content.len());
    let mut replaced = false;
    for line in content.lines() {
        let t = line.trim_start();
        if !replaced
            && !t.starts_with('#')
            && t.strip_prefix("token:")
                .is_some_and(|r| !r.trim().is_empty())
        {
            let indent = &line[..line.len() - t.len()];
            out.push_str(indent);
            out.push_str("token: ");
            out.push_str(new_token);
            replaced = true;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    replaced.then_some(out)
}

/// The scaffolded farm overlay: the cluster-join recipe with this
/// farm's token inline. Deliberately compact — per-key documentation
/// lives in fdl.cluster-join.yml.example and the guide.
/// The door and controller a previous pass wrote, read back from the
/// farm's own worker yml.
///
/// The wizard reuses credentials across runs but re-rendered every other
/// decision from flag defaults, so `fdl join-config <label>` with no
/// flags silently rewrote an existing farm: the door reverted to `b` and
/// the controller to this box's hostname, leaving the printed
/// authorized_keys line describing a different farm than the one on
/// disk. Reprinting that line is the obvious reason to run the wizard
/// twice, so the previous answers are the right defaults.
///
/// The worker yml is the wizard's own deterministic output, which is why
/// it can be read back: door B writes an `rsync://` source, door A an
/// `sshfs://` data source, and `nologin` neither.
pub(super) fn recover_shape(farm_dir: &Path) -> Option<(Door, String)> {
    let yml = fs::read_to_string(farm_dir.join("worker.yml")).ok()?;
    let door = if yml.contains("from: rsync://") {
        Door::B
    } else if yml.contains("data_source: sshfs://") {
        Door::A
    } else {
        Door::Nologin
    };
    let field = |key: &str| -> Option<String> {
        yml.lines()
            .map(str::trim)
            .find_map(|l| l.strip_prefix(key))
            .map(|v| v.trim().to_string())
    };
    // `target:` is the only required piece; a farm yml without one is
    // not this wizard's output, so recover nothing rather than guess.
    let host = field("target:")?;
    let user = field("user:")?;
    let spec = match field("port:") {
        Some(p) => format!("{user}@{host}:{p}"),
        None => format!("{user}@{host}"),
    };
    Some((door, spec))
}

/// The command name the scaffold should wire for launcher mode: the
/// training crate's package name, by the convention that a run command
/// carries its binary's name. `None` when there is no crate here, which
/// scaffolds a commented placeholder instead of inventing a name.
pub(super) fn command_hint(crate_dir: PathBuf) -> Option<String> {
    let manifest = fs::read_to_string(crate_dir.join("Cargo.toml")).ok()?;
    package_name(&manifest)
}

/// Warn when the merged config's join identity points outside the farm
/// dir — a key shared with something else, which per-farm keys exist to
/// prevent.
pub(super) fn foreign_identity_warning(
    root: &Path,
    label: &str,
    farm_dir: &Path,
) -> Option<String> {
    let base = crate::config::find_project_config(root)?;
    let project = crate::config::load_project_with_env(&base, Some(label)).ok()?;
    let identity = project.join.as_ref()?.ssh.as_ref()?.identity_file.clone()?;
    let p = PathBuf::from(&identity);
    if p.starts_with(farm_dir) || identity.contains(&format!(".fdl/{label}/")) {
        return None;
    }
    Some(format!(
        "the merged config's join.ssh.identity_file ({identity}) lives \
         outside .fdl/{label}/ — a key reused across farms (or anything \
         else) widens what one leaked worker can reach. Per-farm keys are \
         the point of this wizard.",
    ))
}
