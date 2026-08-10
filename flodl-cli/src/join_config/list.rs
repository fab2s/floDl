//! `--list`: the project's farms, as the union of what declares one.
//!
//! Overlay names come from the resolver's own discovery, so this can
//! never disagree with what `fdl @<env>` accepts; farm dirs are
//! recognised by the wizard's own output (`worker.yml` or `keys/`), so
//! other `.fdl/` state — schema caches, the ui's run ledger — is not
//! mistaken for a farm.

use std::fs;
use std::path::{Path, PathBuf};

use crate::builtins::JoinConfigArgs;
use crate::overlay;
use crate::style;

use super::credentials::{find_token_line, recover_shape, validate_label};
use super::{Door, KEY_NAME, door_key};

// ── Farm enumeration (--list) ───────────────────────────────────────────

/// `--list`: every farm of this project — the union of `fdl.<label>.*`
/// overlays and `.fdl/<label>/` farm dirs — with door, controller and
/// credential state. Read-only. Env overlays that are not farms (no
/// farm dir) are reported separately rather than dressed as broken
/// farms.
pub(super) fn run_list(cli: &JoinConfigArgs) -> i32 {
    let cwd = match std::env::current_dir() {
        Ok(c) => c,
        Err(e) => {
            crate::cli_error!("cannot read cwd: {e}");
            return 1;
        }
    };
    let root = crate::config::find_project_config(&cwd)
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or(cwd);
    if cli.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&farms_json(&root)).expect("a farm list serializes"),
        );
    } else {
        let (farms, other_envs) = enumerate_farms(&root);
        print!("{}", render_farm_list(&root, &farms, &other_envs));
    }
    0
}

/// The `--list --json` payload for a project root — also what `fdl ui`'s
/// farms endpoint serves, so the page and the CLI cannot disagree.
pub(crate) fn farms_json(root: &Path) -> serde_json::Value {
    let (farms, other_envs) = enumerate_farms(root);
    serde_json::json!({
        "root": root.display().to_string(),
        "farms": farms.iter().map(FarmInfo::to_json).collect::<Vec<_>>(),
        "other_envs": other_envs,
    })
}

/// One farm's on-disk state, as `--list` reports it.
pub(super) struct FarmInfo {
    pub(super) label: String,
    pub(super) overlay_path: PathBuf,
    pub(super) overlay_exists: bool,
    pub(super) has_token: bool,
    pub(super) farm_dir: PathBuf,
    pub(super) key_present: bool,
    pub(super) worker_yml: bool,
    pub(super) cloud_init: bool,
    pub(super) sshd_conf: bool,
    pub(super) door: Option<Door>,
    pub(super) controller: Option<String>,
}

/// A directory under `.fdl/` is a farm dir when it looks like the
/// wizard's output — `worker.yml` or a `keys/` dir — so schema caches
/// and future non-farm state under `.fdl/` never masquerade as farms.
pub(super) fn is_farm_dir(dir: &Path) -> bool {
    dir.join("worker.yml").is_file() || dir.join("keys").is_dir()
}

/// The project's farms plus the env overlays that are not farms.
pub(super) fn enumerate_farms(root: &Path) -> (Vec<FarmInfo>, Vec<String>) {
    // Overlay names come from the resolver's own discovery, so `--list`
    // cannot disagree with what `fdl @<env>` accepts (.yaml/.json too).
    let base = crate::config::find_project_config(root);
    let envs = base.as_deref().map(overlay::list_envs).unwrap_or_default();

    let mut labels: Vec<String> = Vec::new();
    let mut other_envs: Vec<String> = Vec::new();
    for env in envs {
        if is_farm_dir(&root.join(".fdl").join(&env)) {
            labels.push(env);
        } else {
            other_envs.push(env);
        }
    }
    // Farm dirs without an overlay (half-provisioned, or the overlay
    // was deleted) still list — their gap is the finding.
    if let Ok(entries) = fs::read_dir(root.join(".fdl")) {
        for e in entries.flatten() {
            let name = e.file_name();
            let Some(name) = name.to_str() else { continue };
            if validate_label(name).is_ok()
                && is_farm_dir(&e.path())
                && !labels.iter().any(|l| l == name)
            {
                labels.push(name.to_string());
            }
        }
    }
    labels.sort();
    let farms = labels
        .iter()
        .map(|label| farm_info(root, base.as_deref(), label))
        .collect();
    (farms, other_envs)
}

pub(super) fn farm_info(root: &Path, base: Option<&Path>, label: &str) -> FarmInfo {
    let overlay_path = base
        .and_then(|b| overlay::find_env_file(b, label))
        .unwrap_or_else(|| root.join(format!("fdl.{label}.yml")));
    let overlay_exists = overlay_path.is_file();
    let has_token = overlay_exists
        && fs::read_to_string(&overlay_path)
            .ok()
            .and_then(|c| find_token_line(&c))
            .is_some();
    let farm_dir = root.join(".fdl").join(label);
    let key_path = farm_dir.join("keys").join(KEY_NAME);
    let shape = recover_shape(&farm_dir);
    FarmInfo {
        label: label.to_string(),
        overlay_path,
        overlay_exists,
        has_token,
        key_present: key_path.is_file() && key_path.with_extension("pub").is_file(),
        worker_yml: farm_dir.join("worker.yml").is_file(),
        cloud_init: farm_dir.join("cloud-init.yml").is_file(),
        sshd_conf: farm_dir.join(format!("sshd-{label}.conf")).is_file(),
        door: shape.as_ref().map(|s| s.0),
        controller: shape.map(|s| s.1),
        farm_dir,
    }
}

impl FarmInfo {
    pub(super) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "label": self.label,
            "overlay": {
                "path": self.overlay_path.display().to_string(),
                "exists": self.overlay_exists,
                "token": self.has_token,
            },
            "farm_dir": self.farm_dir.display().to_string(),
            "key_present": self.key_present,
            "worker_yml": self.worker_yml,
            "cloud_init": self.cloud_init,
            "sshd_conf": self.sshd_conf,
            "door": self.door.map(door_key),
            "controller": self.controller,
        })
    }
}

pub(super) fn render_farm_list(root: &Path, farms: &[FarmInfo], other_envs: &[String]) -> String {
    let mut out = String::new();
    if farms.is_empty() {
        out.push_str(&format!(
            "no farms at {} — create one with `fdl join-config <label>`\n",
            root.display(),
        ));
    } else {
        out.push_str(&format!("farms at {}:\n\n", root.display()));
        for f in farms {
            let door = match f.door {
                Some(Door::B) => "door b (rrsync source pull)",
                Some(Door::A) => "door a (sftp data mount)",
                Some(Door::Nologin) => "door nologin (tunnel only)",
                None => "door unknown (no wizard worker.yml)",
            };
            let controller = f
                .controller
                .as_deref()
                .map(|c| format!(", controller {c}"))
                .unwrap_or_default();
            let state = |present: bool| if present { "present" } else { "MISSING" };
            out.push_str(&format!("  {}\n", style::bold(&f.label)));
            out.push_str(&format!("    {door}{controller}\n"));
            out.push_str(&format!(
                "    overlay: {}   key: {}   token: {}\n",
                state(f.overlay_exists),
                state(f.key_present),
                state(f.has_token),
            ));
            let extras: Vec<&str> = [
                f.worker_yml.then_some("worker.yml"),
                f.cloud_init.then_some("cloud-init.yml (SECRET)"),
                f.sshd_conf.then_some("sshd drop-in"),
            ]
            .into_iter()
            .flatten()
            .collect();
            if !extras.is_empty() {
                out.push_str(&format!("    artifacts: {}\n", extras.join(", ")));
            }
            out.push('\n');
        }
    }
    if !other_envs.is_empty() {
        out.push_str(&format!(
            "env overlays that are not farms: {}\n",
            other_envs.join(", "),
        ));
    }
    out
}
