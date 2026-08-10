//! `fdl join-config` — the once-per-farm wizard.
//!
//! The controller side of provisioning a walk-in fleet, assembled in one
//! pass instead of from guide prose: the farm overlay (`fdl.<label>.yml`,
//! token stamped), an ed25519 join key born in `./.fdl/<label>/` so it
//! cannot be shared across farms by construction, the guardrailed
//! `authorized_keys` line for the chosen door, the paste-ready worker
//! yml, a publish recipe derived from the training crate's own manifest,
//! and a build-freshness report.
//!
//! A farm IS an env overlay: `fdl @<label> <cmd>` targets it with the
//! machinery that already exists (deep-merge onto the base fdl.yml,
//! `inherit-from:` for shared bases, `fdl config show` provenance).
//! The wizard only ever *scaffolds* an overlay or replaces a `token:`
//! value textually — a user's yml is mostly comments, and a serde
//! round-trip would delete every one of them.
//!
//! Secret hygiene: the private key goes ONLY to 0600 files under the
//! farm dir (which self-gitignores); stdout gets the public line and the
//! worker yml. `--json` reports secrets as file paths, never payloads.
//!
//! Two read-only companions serve scripted and GUI callers: `--list`
//! enumerates the project's farms with their credential state, and
//! `--dry-run` runs the full pass with every write withheld, reporting
//! what an apply would create, update or leave alone. Neither ever
//! prompts.

use std::fs;
use std::path::{Path, PathBuf};

use crate::builtins::JoinConfigArgs;
use crate::util::platform;

/// Key file basename inside `<farm>/keys/`.
const KEY_NAME: &str = "flodl-join";
/// Where the worker yml tells the box to keep its private key. The
/// wizard cannot know the worker's home, so the yml uses `~` and the
/// operator (or cloud-init) lands the file there.
const WORKER_KEY_PATH: &str = "~/.ssh/flodl-join";
/// Default served dir on the controller (`fdl publish`'s default),
/// relative to the controller user's home.
const SERVED_SUBDIR: &str = ".flodl/run";

pub fn run(cli: &JoinConfigArgs) -> i32 {
    if cli.list {
        return run_list(cli);
    }
    match wizard(cli) {
        Ok(report) => {
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report.to_json())
                        .expect("a report value serializes"),
                );
            } else {
                print!("{}", report.render_human());
            }
            0
        }
        Err(e) => {
            crate::cli_error!("{e}");
            1
        }
    }
}

/// Which forced-command door the join key opens. Mirrors the guide's
/// recipe lettering; `C` (a second source-only key for cross-host
/// routing) stays a manual composition and is named in the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Door {
    /// `rrsync -ro <served>`: tunnel + source pull — the publish-then-
    /// join flagship, and the default.
    B,
    /// `internal-sftp -R -d <data>`: tunnel + read-only data mount;
    /// the source is then provisioned or pulled through another road.
    A,
    /// `/usr/sbin/nologin`: tunnel only; source and data both
    /// provisioned.
    Nologin,
}

impl Door {
    fn parse(s: Option<&str>) -> Result<Self, String> {
        match s.unwrap_or("b") {
            "b" | "B" => Ok(Door::B),
            "a" | "A" => Ok(Door::A),
            "nologin" => Ok(Door::Nologin),
            "c" | "C" => Err("door `c` (a second, source-only key) serves cross-host \
                 routing that fdl join's single identity cannot express — \
                 compose it manually from the guide's recipe \
                 (docs/ddp/02-cluster-guide.md), or use door `b`"
                .to_string()),
            other => Err(format!(
                "unknown door `{other}` — one of `b` (rrsync source pull, \
                 the publish-then-join default), `a` (sftp data mount), \
                 `nologin` (tunnel only)"
            )),
        }
    }
}

/// Everything one wizard pass decided and produced — the pure render
/// core's input, so the human text and the JSON twin cannot drift.
struct Report {
    label: String,
    farm_dir: PathBuf,
    overlay_path: PathBuf,
    overlay_action: OverlayAction,
    key_path: PathBuf,
    pub_line: String,
    key_action: KeyAction,
    reuse_warning: Option<String>,
    authorized_line: String,
    match_block: String,
    door: Door,
    worker_yml_path: PathBuf,
    worker_yml: String,
    publish_block: Option<String>,
    bin_caveat: Option<String>,
    freshness: Option<String>,
    notes_path: PathBuf,
    controller: Endpoint,
    install: InstallAction,
    cloud_init_path: Option<PathBuf>,
    /// What this box still needs for the chosen door to work.
    checks: Vec<Check>,
    /// The ready-to-install sshd drop-in written into the farm dir.
    sshd_conf_path: PathBuf,
    plat: platform::Platform,
    /// Whether the box just inspected was a container. What preflight
    /// found is true of THAT environment, and a package installed into a
    /// running container does not survive the next `--rm`.
    in_container: bool,
    /// Compose services this project dispatches through, when fdl runs on
    /// the host but the work happens in a container.
    docker_services: Vec<String>,
    /// `--dry-run`: every action reported is prospective — nothing was
    /// written or installed, and credentials an apply would mint appear
    /// as placeholders.
    dry_run: bool,
    /// Every file the pass wrote or (dry) would write.
    changes: Vec<Change>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverlayAction {
    Scaffolded,
    TokenReplaced,
    TokenReused,
    /// The overlay exists but is user-authored with no token: the
    /// wizard prints the snippet instead of editing prose it does not
    /// own.
    SnippetPrinted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyAction {
    Generated,
    Regenerated,
    Reused,
}

/// `[user@]host[:port]` — how workers reach the join sshd.
#[derive(Debug, Clone)]
struct Endpoint {
    user: String,
    host: String,
    port: u16,
}

impl Endpoint {
    fn parse(spec: Option<&str>) -> Result<Self, String> {
        let default_user = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "op".to_string());
        let Some(spec) = spec else {
            return Ok(Endpoint {
                user: default_user,
                host: crate::cluster::resolve_local_hostname(),
                port: 22,
            });
        };
        let (user, rest) = match spec.split_once('@') {
            Some((u, r)) if !u.is_empty() => (u.to_string(), r),
            Some(_) => return Err(format!("--controller `{spec}`: empty user before `@`")),
            None => (default_user, spec),
        };
        let (host, port) = match rest.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse::<u16>()
                    .map_err(|_| format!("--controller `{spec}`: bad port `{p}`"))?,
            ),
            None => (rest.to_string(), 22),
        };
        if host.is_empty() {
            return Err(format!("--controller `{spec}`: empty host"));
        }
        Ok(Endpoint { user, host, port })
    }
}

/// What a dry run shows where a credential an apply would mint belongs:
/// displaying a real value the apply will not reproduce would be a lie.
const PLACEHOLDER_PUB: &str = "<ed25519 public key — minted on apply>";
const PLACEHOLDER_TOKEN: &str = "<token — minted on apply>";
const PLACEHOLDER_PRIVATE: &str = "<private key — minted on apply>";

/// How a wizard write relates to what is already on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Create,
    Update,
    Unchanged,
}

impl ChangeKind {
    fn as_str(self) -> &'static str {
        match self {
            ChangeKind::Create => "create",
            ChangeKind::Update => "update",
            ChangeKind::Unchanged => "unchanged",
        }
    }
}

/// One file the pass touched — or, under `--dry-run`, would touch.
#[derive(Debug, Clone)]
struct Change {
    path: PathBuf,
    kind: ChangeKind,
    what: &'static str,
}

/// Write-through recorder: every artifact write goes through here, so
/// `--dry-run` is the same pass with the writes withheld and the report
/// carries a faithful change list either way. Identical content is
/// never rewritten, so an idle re-run leaves every mtime alone.
struct Changes {
    dry: bool,
    entries: Vec<Change>,
}

impl Changes {
    fn new(dry: bool) -> Self {
        Changes {
            dry,
            entries: Vec::new(),
        }
    }

    fn write(
        &mut self,
        path: &Path,
        content: &str,
        what: &'static str,
    ) -> Result<ChangeKind, String> {
        let kind = match fs::read_to_string(path) {
            Ok(existing) if existing == content => ChangeKind::Unchanged,
            Ok(_) => ChangeKind::Update,
            Err(_) => ChangeKind::Create,
        };
        if !self.dry && kind != ChangeKind::Unchanged {
            fs::write(path, content)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        self.note(path, kind, what);
        Ok(kind)
    }

    /// Record a mutation that does not flow through [`Changes::write`]
    /// (key generation, the authorized_keys upsert).
    fn note(&mut self, path: &Path, kind: ChangeKind, what: &'static str) {
        self.entries.push(Change {
            path: path.to_path_buf(),
            kind,
            what,
        });
    }
}

fn door_key(door: Door) -> &'static str {
    match door {
        Door::B => "b",
        Door::A => "a",
        Door::Nologin => "nologin",
    }
}

use authorized_keys::InstallAction;
use preflight::Check;

mod authorized_keys;
mod cloud_init;
mod credentials;
mod list;
mod preflight;
mod publish_recipe;
mod render;
mod wizard;

pub(crate) use credentials::{fresh_token, validate_label};
pub(crate) use list::farms_json;
use list::run_list;
use wizard::wizard;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
