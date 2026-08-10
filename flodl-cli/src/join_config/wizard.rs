//! The pass itself: one invocation, every artifact, in the order an
//! operator meets them.
//!
//! Everything here goes through [`Changes`], the
//! write-through recorder, which is what makes `--dry-run` the same pass
//! with the writes withheld rather than a second code path pretending to
//! agree with the first.

use std::fs;
use std::path::{Path, PathBuf};

use crate::builtins::JoinConfigArgs;
use crate::context::home_dir;
use crate::util::platform;

use super::authorized_keys::{install_authorized_line, set_mode};
use super::cloud_init::{docker_services, render_cloud_init};
use super::credentials::{
    command_hint, confirm, ensure_key, ensure_overlay, foreign_identity_warning, recover_shape,
    resolve_label, validate_label,
};
use super::preflight::preflight;
use super::publish_recipe::{derive_publish, freshness_report, render_publish_block};
use super::render::{
    authorized_keys_line, render_notes, render_sshd_conf, render_worker_yml, sshd_match_block,
};
use super::{
    ChangeKind, Changes, Door, Endpoint, KEY_NAME, KeyAction, PLACEHOLDER_PRIVATE, PLACEHOLDER_PUB,
    Report, SERVED_SUBDIR,
};

pub(super) fn wizard(cli: &JoinConfigArgs) -> Result<Report, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read cwd: {e}"))?;
    wizard_at(cli, &cwd)
}

/// The whole pass from an explicit working directory — what [`wizard`]
/// resolves for real invocations and what tests pin.
pub(super) fn wizard_at(cli: &JoinConfigArgs, cwd: &Path) -> Result<Report, String> {
    let label = resolve_label(cli)?;
    validate_label(&label)?;
    let mut changes = Changes::new(cli.dry_run);
    // Door and controller are resolved after the farm dir is known: an
    // existing farm's own answers are better defaults than the flag
    // defaults (see `recover_shape`).

    // The farm sits beside a base fdl.yml (overlays are siblings). No
    // project at all gets a minimal base, confirm-gated — the /training
    // shape, an operator working next to their script.
    let root = match crate::config::find_project_config(cwd) {
        Some(p) => p
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.to_path_buf()),
        None => {
            let target = cwd.join("fdl.yml");
            if cli.dry_run {
                changes.note(&target, ChangeKind::Create, "minimal base fdl.yml");
            } else {
                if !confirm(
                    cli,
                    &format!(
                        "no fdl.yml here — create a minimal one at {} so the farm \
                         overlay has a base?",
                        target.display()
                    ),
                )? {
                    return Err("a farm overlay needs a base fdl.yml to merge onto".into());
                }
                fs::write(
                    &target,
                    "# fdl.yml — created by `fdl join-config` so farm overlays\n\
                     # (fdl.<label>.yml) have a base to merge onto. Project\n\
                     # config goes here as it grows.\n",
                )
                .map_err(|e| format!("cannot write {}: {e}", target.display()))?;
                changes.note(&target, ChangeKind::Create, "minimal base fdl.yml");
            }
            cwd.to_path_buf()
        }
    };

    let farm_dir = root.join(".fdl").join(&label);
    // What a previous pass decided, so a re-run does not silently
    // re-render the farm from flag defaults. A flag still wins: naming
    // one is how you intend a change.
    let prior = recover_shape(&farm_dir);
    let door = match cli.door.as_deref() {
        Some(d) => Door::parse(Some(d))?,
        None => prior.as_ref().map(|p| p.0).unwrap_or(Door::B),
    };
    let controller = match cli.controller.as_deref() {
        Some(c) => Endpoint::parse(Some(c))?,
        None => match prior.as_ref().map(|p| p.1.clone()) {
            Some(spec) => Endpoint::parse(Some(&spec))?,
            None => Endpoint::parse(None)?,
        },
    };
    let keys_dir = farm_dir.join("keys");
    if !cli.dry_run {
        fs::create_dir_all(&keys_dir)
            .map_err(|e| format!("cannot create {}: {e}", keys_dir.display()))?;
    }
    // The farm dir holds private keys, so it removes ITSELF from git:
    // a `*` gitignore inside covers everything including the ignore. An
    // existing one is the user's file and stays untouched.
    let self_ignore = root.join(".fdl").join(".gitignore");
    if !self_ignore.is_file() {
        changes.write(&self_ignore, "*\n", ".fdl self-gitignore")?;
    }

    // ── Keys ────────────────────────────────────────────────────────────
    let key_path = keys_dir.join(KEY_NAME);
    let key_action = ensure_key(cli, &mut changes, &key_path, &label)?;
    // A dry run mints nothing, so wherever the (re)generated key would
    // appear, a placeholder appears instead.
    let pub_line =
        if cli.dry_run && matches!(key_action, KeyAction::Generated | KeyAction::Regenerated) {
            PLACEHOLDER_PUB.to_string()
        } else {
            fs::read_to_string(key_path.with_extension("pub"))
                .map_err(|e| format!("cannot read the generated public key: {e}"))?
                .trim()
                .to_string()
        };

    // ── Token + overlay ─────────────────────────────────────────────────
    let overlay_path = root.join(format!("fdl.{label}.yml"));
    // The command whose entry the scaffold names. Read from the crate's
    // own manifest by the repo convention that a run command carries its
    // binary's name; `None` scaffolds a commented placeholder rather than
    // inventing one.
    let cmd_hint = command_hint(match &cli.crate_dir {
        Some(d) => {
            let p = PathBuf::from(d);
            if p.is_absolute() { p } else { cwd.join(p) }
        }
        None => cwd.to_path_buf(),
    });
    let (token, overlay_action) = ensure_overlay(
        cli,
        &mut changes,
        &overlay_path,
        &label,
        &root,
        cmd_hint.as_deref(),
    )?;

    // A yml-referenced identity outside the farm dir is a key shared
    // with something else — legal, but exactly the reuse a per-farm key
    // exists to prevent, so it gets said out loud.
    let reuse_warning = foreign_identity_warning(&root, &label, &farm_dir);

    // ── Guardrail artifacts ─────────────────────────────────────────────
    let served_abs = home_dir().join(SERVED_SUBDIR);
    let authorized_line = authorized_keys_line(door, &served_abs, cli, &pub_line);
    let match_block = sshd_match_block(&controller.user);

    // ── The training crate: publish recipe + freshness ──────────────────
    let crate_dir = match &cli.crate_dir {
        Some(d) => {
            let p = PathBuf::from(d);
            if p.is_absolute() { p } else { cwd.join(p) }
        }
        None => cwd.to_path_buf(),
    };
    let (publish_block, bin_caveat, freshness) = match derive_publish(&crate_dir) {
        Ok(Some(d)) => (
            Some(render_publish_block(&d)),
            d.bin_caveat.clone(),
            Some(freshness_report(&d.from_root)),
        ),
        Ok(None) => (None, None, None),
        Err(e) => (None, Some(e), None),
    };

    // ── Worker yml ──────────────────────────────────────────────────────
    let worker_yml = render_worker_yml(&label, &controller, &token, door, cli);
    let worker_yml_path = farm_dir.join("worker.yml");
    changes.write(&worker_yml_path, &worker_yml, "worker fdl.yml")?;

    // ── Install notes ───────────────────────────────────────────────────
    let notes_path = farm_dir.join("install-notes.md");
    let notes = render_notes(&label, &authorized_line, &match_block, &controller, door);
    changes.write(&notes_path, &notes, "install notes")?;

    // ── The sshd drop-in ────────────────────────────────────────────────
    // Written as a FILE rather than printed, so the install step is a
    // copy of something reviewable rather than a heredoc pasted blind.
    let plat = platform::Platform::detect();
    let sshd_conf_path = farm_dir.join(format!("sshd-{label}.conf"));
    let sshd_conf = render_sshd_conf(&label, door, controller.port, plat);
    changes.write(&sshd_conf_path, &sshd_conf, "sshd drop-in")?;

    // ── Preflight ───────────────────────────────────────────────────────
    let checks = preflight(door, controller.port, &served_abs, plat);
    let services = docker_services(&root, &label);

    // ── The install offer ───────────────────────────────────────────────
    let install = install_authorized_line(cli, &mut changes, &authorized_line, controller.port)?;

    // ── cloud-init (opt-in) ─────────────────────────────────────────────
    let cloud_init_path = if cli.cloud_init {
        let user = cli.cloud_init_user.as_deref().unwrap_or("ubuntu");
        let private_key =
            if cli.dry_run && matches!(key_action, KeyAction::Generated | KeyAction::Regenerated) {
                PLACEHOLDER_PRIVATE.to_string()
            } else {
                fs::read_to_string(&key_path)
                    .map_err(|e| format!("cannot read the private key for cloud-init: {e}"))?
            };
        let content = render_cloud_init(&label, user, door, &worker_yml, &private_key);
        let path = farm_dir.join("cloud-init.yml");
        let kind = changes.write(&path, &content, "cloud-init user-data (SECRET)")?;
        if !cli.dry_run && kind != ChangeKind::Unchanged {
            set_mode(&path, 0o600)?;
        }
        Some(path)
    } else {
        None
    };

    Ok(Report {
        label,
        farm_dir,
        overlay_path,
        overlay_action,
        key_path,
        pub_line,
        key_action,
        reuse_warning,
        authorized_line,
        match_block,
        door,
        worker_yml_path,
        worker_yml,
        publish_block,
        bin_caveat,
        freshness,
        notes_path,
        controller,
        install,
        cloud_init_path,
        checks,
        sshd_conf_path,
        plat,
        in_container: platform::Platform::in_container(),
        docker_services: services,
        dry_run: cli.dry_run,
        changes: changes.entries,
    })
}
