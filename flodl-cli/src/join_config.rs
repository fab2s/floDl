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

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::builtins::JoinConfigArgs;
use crate::context::home_dir;
use crate::style;
use crate::util::platform;
use crate::util::prompt;
use crate::util::system;

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
            "c" | "C" => Err(
                "door `c` (a second, source-only key) serves cross-host \
                 routing that fdl join's single identity cannot express — \
                 compose it manually from the guide's recipe \
                 (docs/ddp/02-cluster-guide.md), or use door `b`"
                    .to_string(),
            ),
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

fn wizard(cli: &JoinConfigArgs) -> Result<Report, String> {
    let label = resolve_label(cli)?;
    validate_label(&label)?;
    // Door and controller are resolved after the farm dir is known: an
    // existing farm's own answers are better defaults than the flag
    // defaults (see `recover_shape`).

    // The farm sits beside a base fdl.yml (overlays are siblings). No
    // project at all gets a minimal base, confirm-gated — the /training
    // shape, an operator working next to their script.
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read cwd: {e}"))?;
    let root = match crate::config::find_project_config(&cwd) {
        Some(p) => p.parent().map(Path::to_path_buf).unwrap_or_else(|| cwd.clone()),
        None => {
            let target = cwd.join("fdl.yml");
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
            cwd.clone()
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
    fs::create_dir_all(&keys_dir)
        .map_err(|e| format!("cannot create {}: {e}", keys_dir.display()))?;
    // The farm dir holds private keys, so it removes ITSELF from git:
    // a `*` gitignore inside covers everything including the ignore.
    let self_ignore = root.join(".fdl").join(".gitignore");
    if !self_ignore.is_file() {
        fs::write(&self_ignore, "*\n")
            .map_err(|e| format!("cannot write {}: {e}", self_ignore.display()))?;
    }

    // ── Keys ────────────────────────────────────────────────────────────
    let key_path = keys_dir.join(KEY_NAME);
    let key_action = ensure_key(cli, &key_path, &label)?;
    let pub_line = fs::read_to_string(key_path.with_extension("pub"))
        .map_err(|e| format!("cannot read the generated public key: {e}"))?
        .trim()
        .to_string();

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
        None => cwd.clone(),
    });
    let (token, overlay_action) =
        ensure_overlay(cli, &overlay_path, &label, &root, cmd_hint.as_deref())?;

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
        None => cwd.clone(),
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
    fs::write(&worker_yml_path, &worker_yml)
        .map_err(|e| format!("cannot write {}: {e}", worker_yml_path.display()))?;

    // ── Install notes ───────────────────────────────────────────────────
    let notes_path = farm_dir.join("install-notes.md");
    let notes = render_notes(&label, &authorized_line, &match_block, &controller, door);
    fs::write(&notes_path, notes)
        .map_err(|e| format!("cannot write {}: {e}", notes_path.display()))?;

    // ── The sshd drop-in ────────────────────────────────────────────────
    // Written as a FILE rather than printed, so the install step is a
    // copy of something reviewable rather than a heredoc pasted blind.
    let plat = platform::Platform::detect();
    let sshd_conf_path = farm_dir.join(format!("sshd-{label}.conf"));
    fs::write(
        &sshd_conf_path,
        render_sshd_conf(&label, door, controller.port, plat),
    )
    .map_err(|e| format!("cannot write {}: {e}", sshd_conf_path.display()))?;

    // ── Preflight ───────────────────────────────────────────────────────
    let checks = preflight(door, controller.port, &served_abs, plat);

    // ── The install offer ───────────────────────────────────────────────
    let install = install_authorized_line(cli, &authorized_line, controller.port)?;

    // ── cloud-init (opt-in) ─────────────────────────────────────────────
    let cloud_init_path = if cli.cloud_init {
        let user = cli.cloud_init_user.as_deref().unwrap_or("ubuntu");
        let private_key = fs::read_to_string(&key_path)
            .map_err(|e| format!("cannot read the private key for cloud-init: {e}"))?;
        let content = render_cloud_init(&label, user, door, &worker_yml, &private_key);
        let path = farm_dir.join("cloud-init.yml");
        fs::write(&path, content)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        set_mode(&path, 0o600)?;
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
    })
}

/// Where a provisioned user's files go. `root` is the exception every
/// cloud image with a root login hits: its home is `/root`, not
/// `/home/root`, so composing the path from the name alone writes the
/// key somewhere sshd will never read.
fn home_of(user: &str) -> String {
    if user == "root" {
        "/root".to_string()
    } else {
        format!("/home/{user}")
    }
}

/// The cloud-init user-data: worker yml, private key, the tools the
/// declared door needs and the systemd recipe, so an instance boots
/// straight into a persistent `fdl join`. A SECRET artifact from the
/// moment it exists — it carries the key and the token — which is why
/// it lands 0600 in the farm dir and never on stdout.
///
/// The unit encodes the failure taxonomy end to end: `Restart=always`
/// re-dials transient exits, `RestartPreventExitStatus=2` stops the hot
/// loop on a permanent one, and `FailureAction=poweroff` (a `[Unit]`
/// option) then halts the instance.
///
/// **Halting is not always deprovisioning.** On providers that keep
/// billing a powered-off instance (DigitalOcean, and the AMD Developer
/// Cloud that runs on it, reserve disk/CPU/RAM/IP until the instance is
/// destroyed), the unit stops the work but not the meter. There, pair
/// it with a provider-side destroy.
///
/// What the instance is assumed to have is only a shell, systemd and
/// network: `fdl` is fetched here, and the door's own tooling with it.
/// Anything already baked into the image wins, since every step is
/// guarded by a `command -v`.
fn render_cloud_init(
    label: &str,
    user: &str,
    door: Door,
    worker_yml: &str,
    private_key: &str,
) -> String {
    let indent = |s: &str| -> String {
        s.lines()
            .map(|l| if l.is_empty() { String::new() } else { format!("      {l}") })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let home = home_of(user);

    // Packages the DECLARED door will actually reach for. Door B pulls a
    // source tree and builds it here, so it needs a compiler and the
    // transports; door A mounts the data root over sshfs, and prepare
    // classes a missing sshfs as permanent — which under this very unit
    // means exit 2 and a halt, on a box that only lacked a package.
    let mut packages: Vec<&str> = vec!["curl"];
    match door {
        Door::B => packages.extend(["build-essential", "pkg-config", "unzip", "rsync", "git"]),
        Door::A => packages.push("sshfs"),
        Door::Nologin => {}
    }
    let packages = packages
        .iter()
        .map(|p| format!("\x20 - {p}\n"))
        .collect::<String>();

    // Door B builds, so it needs a toolchain. Installed AS THE SERVICE
    // USER: cargo writes its registry cache into CARGO_HOME, so a
    // system-wide install root-owned and world-readable is a build that
    // fails on its first fetch. The unit then carries the matching PATH
    // rather than relying on a login shell it never gets.
    let (rust_step, rust_path) = match door {
        Door::B => (
            format!(
                "\x20 - [ sh, -c, \"command -v cargo >/dev/null || \
                 su -l {user} -c 'curl -fsSL https://sh.rustup.rs | \
                 sh -s -- -y --profile minimal --no-modify-path'\" ]\n"
            ),
            format!("{home}/.cargo/bin:"),
        ),
        _ => (String::new(), String::new()),
    };

    format!(
        "#cloud-config\n\
         # Farm `{label}` worker user-data — generated by `fdl join-config`.\n\
         # SECRET ARTIFACT: carries the join key and the admission token.\n\
         # On a provider that bills powered-off instances (DigitalOcean and\n\
         # the AMD Developer Cloud on top of it), the unit's poweroff stops\n\
         # the work but NOT the meter: destroy the instance to stop billing.\n\
         packages:\n{packages}\
         write_files:\n\
         \x20 - path: {home}/.ssh/flodl-join\n\
         \x20   owner: {user}:{user}\n\
         \x20   permissions: \"0600\"\n\
         \x20   defer: true\n\
         \x20   content: |\n{key}\n\
         \x20 - path: {home}/training/fdl.yml\n\
         \x20   owner: {user}:{user}\n\
         \x20   permissions: \"0644\"\n\
         \x20   defer: true\n\
         \x20   content: |\n{yml}\n\
         \x20 - path: /etc/systemd/system/flodl-join.service\n\
         \x20   permissions: \"0644\"\n\
         \x20   content: |\n\
         \x20     [Unit]\n\
         \x20     Description=flodl walk-in worker (farm {label})\n\
         \x20     After=network-online.target\n\
         \x20     Wants=network-online.target\n\
         \x20     FailureAction=poweroff\n\
         \n\
         \x20     [Service]\n\
         \x20     Type=simple\n\
         \x20     User={user}\n\
         \x20     WorkingDirectory={home}/training\n\
         \x20     Environment=PATH={rust_path}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n\
         \x20     ExecStart=/usr/bin/env fdl join\n\
         \x20     Restart=always\n\
         \x20     RestartSec=5\n\
         \x20     RestartPreventExitStatus=2\n\
         \n\
         \x20     [Install]\n\
         \x20     WantedBy=multi-user.target\n\
         runcmd:\n\
         \x20 - [ sh, -c, \"command -v fdl >/dev/null || \
         (curl -fsSL https://flodl.dev/fdl -o /usr/local/bin/fdl && \
         chmod 0755 /usr/local/bin/fdl)\" ]\n\
         {rust_step}\
         \x20 - systemctl daemon-reload\n\
         \x20 - systemctl enable --now flodl-join.service\n",
        label = label,
        user = user,
        home = home,
        packages = packages,
        rust_step = rust_step,
        rust_path = rust_path,
        key = indent(private_key),
        yml = indent(worker_yml),
    )
}

// ── Preflight ───────────────────────────────────────────────────────────

/// One thing this setup needs, whether the box has it, and the command
/// that supplies it here.
#[derive(Debug, Clone)]
struct Check {
    what: String,
    ok: bool,
    /// Platform-translated fix, absent when there is nothing to run
    /// (either it is already satisfied, or no command would be honest).
    fix: Option<String>,
}

/// What the chosen door needs before any of this works, checked before
/// the artifacts are written.
///
/// The wizard used to compose a beautiful setup for a machine it never
/// asked a single question of, so its gaps surfaced one at a time and in
/// the worst order: after credentials were minted, after a publish, and
/// sometimes only after a worker had already dialed. Every gap below was
/// hit for real on this rig.
fn preflight(door: Door, port: u16, served: &Path, plat: platform::Platform) -> Vec<Check> {
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
    let socket_owns =
        plat == platform::Platform::Debian && port != 22 && socket_activated();
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
            let have = ["/usr/lib/openssh/sftp-server", "/usr/libexec/openssh/sftp-server",
                        "/usr/libexec/sftp-server"]
                .iter()
                .any(|p| Path::new(p).exists());
            checks.push(Check {
                what: "an sftp server (door a serves the data mount over it)".into(),
                ok: have,
                fix: (!have).then(|| plat.sshd_package().and_then(|p| plat.install(&[p]))).flatten(),
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

// ── authorized_keys install ─────────────────────────────────────────────

/// What the install offer decided and did. `Skipped` carries the why so
/// the report can say it; refusals are ordinary `Err`s (the wizard has
/// already produced every artifact, but a half-touched authorized_keys
/// must be loud).
#[derive(Debug, Clone, PartialEq, Eq)]
enum InstallAction {
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
fn install_authorized_line(
    cli: &JoinConfigArgs,
    line: &str,
    sshd_port: u16,
) -> Result<InstallAction, String> {
    if cli.install_key && cli.no_install_key {
        return Err("--install-key and --no-install-key contradict each other".into());
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
            "installing needs explicit consent: the prompt, or --install-key"
                .to_string(),
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

    let content = if ak_path.is_file() {
        fix_perms_confirmed(cli, &ak_path, 0o600)?;
        fs::read_to_string(&ak_path)
            .map_err(|e| format!("cannot read {}: {e}", ak_path.display()))?
    } else {
        String::new()
    };

    let (new_content, outcome) = upsert_authorized_line(&content, line)?;
    if outcome == UpsertOutcome::Identical {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpsertOutcome {
    Appended,
    Replaced,
    Identical,
}

/// Append `line`, or replace the one existing line that carries the
/// same public key material. Identity is `keytype + base64` — options
/// and comment may differ, the key bytes cannot. Every other line is
/// preserved byte for byte.
fn upsert_authorized_line(content: &str, line: &str) -> Result<(String, UpsertOutcome), String> {
    let wanted = key_material(line)
        .ok_or("the composed authorized_keys line carries no key material")?;
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
fn key_material(line: &str) -> Option<(&str, &str)> {
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
fn fix_perms_confirmed(cli: &JoinConfigArgs, path: &Path, mode: u32) -> Result<(), String> {
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
fn fix_perms_confirmed(_cli: &JoinConfigArgs, _path: &Path, _mode: u32) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .map_err(|e| format!("cannot chmod {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> Result<(), String> {
    Ok(())
}

/// Is anything accepting on the sshd port workers will dial? Loopback
/// suffices as a heuristic — sshd binds all interfaces by default.
fn sshd_listening(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        std::time::Duration::from_millis(300),
    )
    .is_ok()
}

fn resolve_label(cli: &JoinConfigArgs) -> Result<String, String> {
    if let Some(l) = &cli.label {
        return Ok(l.clone());
    }
    if let Ok(env) = std::env::var("FDL_ENV") {
        if !env.trim().is_empty() {
            return Ok(env.trim().to_string());
        }
    }
    Err("a farm needs a label: `fdl join-config <label>` (or target an \
         existing overlay: `fdl @<label> join-config`)"
        .to_string())
}

/// Labels become filenames (`fdl.<label>.yml`, `.fdl/<label>/`), so the
/// charset is the portable-filename one.
fn validate_label(label: &str) -> Result<(), String> {
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
fn confirm(cli: &JoinConfigArgs, question: &str) -> Result<bool, String> {
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

fn ensure_key(cli: &JoinConfigArgs, key_path: &Path, label: &str) -> Result<KeyAction, String> {
    let exists = key_path.is_file() && key_path.with_extension("pub").is_file();
    if exists {
        // --yes reuses (regeneration is opt-in, never a default) and a
        // non-tty run cannot be asked, so both fall through to reuse.
        let regen = if cli.regen {
            true
        } else if cli.yes || !prompt::has_tty() {
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
            return Ok(KeyAction::Reused);
        }
        fs::remove_file(key_path).ok();
        fs::remove_file(key_path.with_extension("pub")).ok();
        generate_key(key_path, label)?;
        return Ok(KeyAction::Regenerated);
    }
    generate_key(key_path, label)?;
    Ok(KeyAction::Generated)
}

fn generate_key(key_path: &Path, label: &str) -> Result<(), String> {
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
fn ensure_overlay(
    cli: &JoinConfigArgs,
    overlay_path: &Path,
    label: &str,
    root: &Path,
    cmd_hint: Option<&str>,
) -> Result<(String, OverlayAction), String> {
    if !overlay_path.is_file() {
        let token = fresh_token()?;
        let scaffold = render_overlay_scaffold(label, &token, root, cmd_hint);
        fs::write(overlay_path, scaffold)
            .map_err(|e| format!("cannot write {}: {e}", overlay_path.display()))?;
        return Ok((token, OverlayAction::Scaffolded));
    }
    let content = fs::read_to_string(overlay_path)
        .map_err(|e| format!("cannot read {}: {e}", overlay_path.display()))?;
    match find_token_line(&content) {
        Some(old) => {
            let regen = if cli.regen {
                true
            } else if cli.yes || !prompt::has_tty() {
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
                return Ok((old, OverlayAction::TokenReused));
            }
            let token = fresh_token()?;
            let replaced = replace_token_line(&content, &token)
                .ok_or("token line vanished between read and replace")?;
            fs::write(overlay_path, replaced)
                .map_err(|e| format!("cannot write {}: {e}", overlay_path.display()))?;
            Ok((token, OverlayAction::TokenReplaced))
        }
        None => {
            // A user-authored overlay without a token: the wizard does
            // not edit prose it does not own — nested surgical inserts
            // into hand-written yml guess too much. The snippet is in
            // the report.
            Ok((fresh_token()?, OverlayAction::SnippetPrinted))
        }
    }
}

/// A fresh 32-hex credential from OS entropy. Same shape as the publish
/// nonce, but this one IS secret (it is the admission token), so there
/// is no fallback: a credential mint with no entropy has nothing honest
/// to fall back to, and the syscall failing is a refusal, not a
/// degradation.
fn fresh_token() -> Result<String, String> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|e| format!("cannot draw OS entropy for the token: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// First `token:` line's value, however deep it sits. The wizard only
/// ever needs the join token and a farm overlay carries exactly one.
fn find_token_line(content: &str) -> Option<String> {
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
fn replace_token_line(content: &str, new_token: &str) -> Option<String> {
    let mut out = String::with_capacity(content.len());
    let mut replaced = false;
    for line in content.lines() {
        let t = line.trim_start();
        if !replaced && !t.starts_with('#') && t.strip_prefix("token:").is_some_and(|r| !r.trim().is_empty()) {
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
fn recover_shape(farm_dir: &Path) -> Option<(Door, String)> {
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
fn command_hint(crate_dir: PathBuf) -> Option<String> {
    let manifest = fs::read_to_string(crate_dir.join("Cargo.toml")).ok()?;
    package_name(&manifest)
}

fn render_overlay_scaffold(
    label: &str,
    token: &str,
    root: &Path,
    cmd_hint: Option<&str>,
) -> String {
    // Active when the manifest named it, commented when it would be a
    // guess: a wrong active entry defines a command that does not exist,
    // which is a worse failure than one the reader has to uncomment.
    let cmd = match cmd_hint {
        Some(name) => format!("\x20 {name}:\n\x20   cluster: true\n"),
        None => "\x20 # <your-run-command>:\n\x20 #   cluster: true\n".to_string(),
    };
    format!(
        "# fdl.{label}.yml — farm overlay, generated by `fdl join-config`.\n\
         # Activate with `fdl @{label} <cmd>`. Regenerate credentials with\n\
         # `fdl @{label} join-config --regen`. Keys + worker yml live in\n\
         # .fdl/{label}/. Deep-merges onto fdl.yml; see\n\
         # fdl.cluster-join.yml.example for per-key docs, and `inherit-from:`\n\
         # for sharing a base between farms.\n\
         \n\
         cluster:\n\
         \x20 controller:\n\
         \x20   host: 127.0.0.1                # loopback; tunnel_only forces it anyway\n\
         \x20   port: 1337\n\
         \x20   path: {root}\n\
         \x20   join:\n\
         \x20     discovery: true              # the window defines the world\n\
         \x20     min_rank_start: 1            # RAISE to your fleet's quorum (in ranks)\n\
         \x20     start: manual                # hold at quorum; `fdl start` fires it\n\
         \x20     join_timeout: 600\n\
         \x20     max_join_timeout: 1200\n\
         \x20     tunnel_only: true            # sshd forward is the only road in (CPU modes)\n\
         \x20     token: {token}\n\
         \n\
         \x20 workers: []                      # walk-ins fill it\n\
         \n\
         # A join window only opens for a command that runs in launcher\n\
         # mode, and `cluster:` is what puts it there. Without an entry\n\
         # here `fdl @{label} <cmd>` resolves the base command and runs it\n\
         # LOCALLY: no window, no walk-ins, and nothing says so, because\n\
         # training on this box is a legitimate thing to do. Name the\n\
         # command that starts your run.\n\
         commands:\n\
         {cmd}",
        label = label,
        token = token,
        root = root.display(),
        cmd = cmd,
    )
}

/// Warn when the merged config's join identity points outside the farm
/// dir — a key shared with something else, which per-farm keys exist to
/// prevent.
fn foreign_identity_warning(root: &Path, label: &str, farm_dir: &Path) -> Option<String> {
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

fn authorized_keys_line(
    door: Door,
    served_abs: &Path,
    cli: &JoinConfigArgs,
    pub_line: &str,
) -> String {
    let forced = match door {
        Door::B => format!("command=\"rrsync -ro {}\"", served_abs.display()),
        Door::A => format!(
            "command=\"internal-sftp -R -d {}\"",
            cli.data_path.as_deref().unwrap_or(crate::config::DEFAULT_DATA_PATH),
        ),
        Door::Nologin => "command=\"/usr/sbin/nologin\"".to_string(),
    };
    format!(
        "restrict,port-forwarding,permitopen=\"127.0.0.1:1337\",{forced} {pub_line}"
    )
}

/// The ready-to-install sshd drop-in for this farm's door.
///
/// A file rather than a printed block: the install step then copies
/// something the operator can read first, and re-running the wizard
/// updates it in place instead of asking anyone to re-paste.
///
/// The guardrail is scoped to the PORT, not to a user. Scoping it to a
/// user means either a dedicated no-shell account (whose authorized_keys
/// only root can write, so the wizard cannot install its own line) or
/// crippling the operator's own logins. Binding it to the exposed port
/// restricts every key that arrives there, including ones added later,
/// and leaves port 22 alone.
fn render_sshd_conf(label: &str, door: Door, port: u16, plat: platform::Platform) -> String {
    let door_note = match door {
        Door::Nologin => "tunnel only",
        Door::A => "tunnel + read-only sftp data mount (the key carries the command)",
        Door::B => "tunnel + rrsync source pull (the key carries the command)",
    };
    let mut l: Vec<String> = vec![
        format!("# floDl join door for farm `{label}` — generated by `fdl join-config`."),
        format!("# Door: {door_note}."),
        "#".into(),
        format!(
            "# Port {port} is the join door and the only one to expose; 22 stays"
        ),
        "# for ordinary logins and should NOT be forwarded from a router.".into(),
    ];
    if plat == platform::Platform::Debian && port != 22 {
        l.push("#".into());
        l.push("# NOTE: while ssh.socket owns the listener, the Port line below is".into());
        l.push("# IGNORED. The install step disables it and enables ssh.service.".into());
    }
    if plat == platform::Platform::Rhel && port != 22 {
        l.push("#".into());
        l.push("# NOTE: SELinux must be told this port is for ssh, or the daemon".into());
        l.push("# fails to bind with an error that never mentions SELinux.".into());
    }
    l.push(String::new());
    l.push("Port 22".into());
    l.push(format!("Port {port}"));
    l.push("PermitRootLogin no".into());
    l.push("PasswordAuthentication no".into());
    l.push("KbdInteractiveAuthentication no".into());
    l.push("PubkeyAuthentication yes".into());
    l.push(String::new());
    l.push("# The daemon half of the guardrail, bound to the exposed port so".into());
    l.push("# ordinary logins on 22 are untouched and ANY key arriving here is".into());
    l.push("# confined to the controller mux forward.".into());
    l.push(format!("Match LocalPort {port}"));
    l.push("    AllowTcpForwarding local".into());
    l.push("    PermitOpen 127.0.0.1:1337".into());
    // ForceCommand belongs ONLY to the tunnel-only door. Doors a and b
    // carry their command in the key line, and a daemon-level forced
    // command would override it: the tunnel would keep working while the
    // mount or the source pull failed, which is the confusing half.
    if door == Door::Nologin {
        l.push("    ForceCommand /usr/sbin/nologin".into());
    }
    l.push("    PermitTTY no".into());
    l.push("    X11Forwarding no".into());
    l.push("    AllowAgentForwarding no".into());
    l.push(String::new());
    l.join("\n")
}

fn sshd_match_block(user: &str) -> String {
    format!(
        "Match User {user}\n    \
         AllowTcpForwarding local\n    \
         PermitOpen 127.0.0.1:1337\n    \
         PermitTTY no\n    \
         X11Forwarding no\n    \
         AllowAgentForwarding no"
    )
}

// ── The training crate: publish recipe derivation ───────────────────────

/// What one manifest scan decided.
struct PublishDerivation {
    from_root: PathBuf,
    cwd_rel: Option<String>,
    bin: String,
    build: String,
    bin_caveat: Option<String>,
}

/// Derive the publish recipe from the crate's own Cargo.toml. `Ok(None)`
/// when there is no crate here — a farm can be config-only, so absence
/// is a note, not an error.
fn derive_publish(crate_dir: &Path) -> Result<Option<PublishDerivation>, String> {
    let manifest_path = crate_dir.join("Cargo.toml");
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("cannot read {}: {e}", manifest_path.display()))?;
    let name = package_name(&manifest).ok_or_else(|| {
        format!(
            "{} has no [package] name — a workspace root? Point --crate at \
             the training crate itself",
            manifest_path.display()
        )
    })?;

    // A path dep on flodl decides the fetched root: the tree must
    // contain the dep or the worker's build dangles on a path outside
    // what it fetched.
    let crate_abs = crate_dir
        .canonicalize()
        .map_err(|e| format!("cannot resolve {}: {e}", crate_dir.display()))?;
    let (from_root, cwd_rel) = match flodl_path_dep(&manifest) {
        Some(rel) => {
            let dep_abs = normalize(&crate_abs.join(&rel));
            let root = common_ancestor(&crate_abs, &dep_abs);
            let cwd_rel = crate_abs
                .strip_prefix(&root)
                .ok()
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| p.display().to_string());
            (root, cwd_rel)
        }
        None => (crate_abs.clone(), None),
    };

    // The artifact convention: `cargo build` in the crate dir writes to
    // the crate's own target/ unless a WORKSPACE above claims it. The
    // wizard states the guess and the caveat instead of over-guessing —
    // membership is glob-shaped and a wrong silent answer costs a
    // permanent walk-in failure.
    let bin = format!("target/release/{name}");
    let bin_caveat = workspace_above(&crate_abs, &from_root).map(|ws| {
        format!(
            "{} declares [workspace] above the crate — if it claims the \
             crate as a member, the artifact lands in the WORKSPACE \
             target/, so `bin:` must point there (e.g. \
             `{}target/release/{name}`)",
            ws.join("Cargo.toml").display(),
            "../".repeat(
                crate_abs.strip_prefix(&ws).map(|p| p.components().count()).unwrap_or(1),
            ),
        )
    });

    let features = declares_gpu_features(&manifest);
    let build = if features {
        format!("cargo build --release --features \"$FDL_GPU_FEATURE\" --bin {name}")
    } else {
        format!("cargo build --release --bin {name}")
    };

    Ok(Some(PublishDerivation { from_root, cwd_rel, bin, build, bin_caveat }))
}

/// `[package] name = "..."` — first `name =` line inside the
/// `[package]` table.
fn package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_package = t == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = t.strip_prefix("name") {
                let rest = rest.trim_start();
                if let Some(v) = rest.strip_prefix('=') {
                    return Some(v.trim().trim_matches('"').to_string());
                }
            }
        }
    }
    None
}

/// The `path = "..."` of a `flodl` dependency, if any — the line-level
/// scan handles both the inline-table form (`flodl = { path = ".." }`)
/// and the `[dependencies.flodl]` table form.
fn flodl_path_dep(manifest: &str) -> Option<String> {
    let mut in_flodl_table = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_flodl_table = t == "[dependencies.flodl]"
                || t == "[dev-dependencies.flodl]"
                || t == "[dependencies.flodl-hf]";
            continue;
        }
        let inline = t
            .strip_prefix("flodl")
            .map(|r| r.trim_start())
            .and_then(|r| r.strip_prefix('='))
            .map(|r| r.trim());
        if let Some(spec) = inline {
            if let Some(p) = extract_path_value(spec) {
                return Some(p);
            }
            continue;
        }
        if in_flodl_table {
            if let Some(rest) = t.strip_prefix("path") {
                if let Some(v) = rest.trim_start().strip_prefix('=') {
                    return Some(v.trim().trim_matches('"').to_string());
                }
            }
        }
    }
    None
}

/// `path = "..."` inside an inline table value.
fn extract_path_value(spec: &str) -> Option<String> {
    let idx = spec.find("path")?;
    let rest = spec[idx + 4..].trim_start().strip_prefix('=')?;
    let rest = rest.trim_start();
    let quoted = rest.strip_prefix('"')?;
    let end = quoted.find('"')?;
    Some(quoted[..end].to_string())
}

/// Whether the crate declares the vendor features the recipe would
/// forward — `cuda` or `rocm` keys under `[features]`.
fn declares_gpu_features(manifest: &str) -> bool {
    let mut in_features = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_features = t == "[features]";
            continue;
        }
        if in_features {
            let key = t.split('=').next().unwrap_or("").trim();
            if key == "cuda" || key == "rocm" {
                return true;
            }
        }
    }
    false
}

/// Resolve `..` components without touching the filesystem (the dep
/// path may or may not exist yet on this box).
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

fn common_ancestor(a: &Path, b: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for (ca, cb) in a.components().zip(b.components()) {
        if ca == cb {
            out.push(ca);
        } else {
            break;
        }
    }
    out
}

/// Nearest ancestor between the crate and the fetched root (exclusive
/// of the crate itself) whose Cargo.toml declares `[workspace]`.
fn workspace_above(crate_abs: &Path, from_root: &Path) -> Option<PathBuf> {
    let mut dir = crate_abs.parent()?;
    loop {
        let m = dir.join("Cargo.toml");
        if m.is_file() {
            if let Ok(content) = fs::read_to_string(&m) {
                if content.lines().any(|l| l.trim() == "[workspace]") {
                    return Some(dir.to_path_buf());
                }
            }
        }
        if dir == from_root {
            return None;
        }
        dir = dir.parent()?;
    }
}

fn render_publish_block(d: &PublishDerivation) -> String {
    let mut out = String::from("publish:\n");
    out.push_str(&format!("  source: file://{}\n", d.from_root.display()));
    if let Some(cwd) = &d.cwd_rel {
        out.push_str(&format!("  cwd: {cwd}\n"));
    }
    out.push_str(&format!("  build: {}\n", d.build));
    out.push_str(&format!("  bin: {}\n", d.bin));
    out.push_str("  # args: [--model, ..., --epochs, ...]   # the RUN's args\n");
    out
}

/// Does the lockfile still describe the source about to ship? The
/// honest form of "is the compiled version right".
fn freshness_report(from_root: &Path) -> String {
    let lock = from_root.join("Cargo.lock");
    let Ok(lock_meta) = fs::metadata(&lock) else {
        return format!(
            "no Cargo.lock at {} yet — the publish gate build will create \
             the verified pin",
            from_root.display(),
        );
    };
    let lock_mtime = lock_meta.modified().ok();
    let newest = newest_source_mtime(from_root);
    match (lock_mtime, newest) {
        (Some(l), Some((n, path))) if n > l => format!(
            "Cargo.lock predates the newest source edit ({}) — the next \
             gate build refreshes the pin; publish before pointing workers \
             at this tree",
            path.display(),
        ),
        (Some(_), Some(_)) => "Cargo.lock is current with the source".to_string(),
        _ => "freshness undetermined (no readable source mtimes)".to_string(),
    }
}

fn newest_source_mtime(root: &Path) -> Option<(std::time::SystemTime, PathBuf)> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if matches!(name.as_ref(), "target" | ".git" | ".fdl" | "libtorch") {
                    continue;
                }
                stack.push(path);
            } else if name != "Cargo.lock" {
                if let Ok(m) = entry.metadata() {
                    if let Ok(t) = m.modified() {
                        if newest.as_ref().is_none_or(|(n, _)| t > *n) {
                            newest = Some((t, path));
                        }
                    }
                }
            }
        }
    }
    newest
}

// ── Renders ─────────────────────────────────────────────────────────────

/// The paste-ready worker fdl.yml.
fn render_worker_yml(
    label: &str,
    controller: &Endpoint,
    token: &str,
    door: Door,
    cli: &JoinConfigArgs,
) -> String {
    let mut out = format!(
        "# fdl.yml for a `{label}` farm worker — generated by `fdl join-config`.\n\
         # Land the private key at {WORKER_KEY_PATH} (0600) and run:\n\
         #   fdl join\n\
         # (persist: true makes exits re-dial; the systemd recipe in\n\
         # fdl.yml.example turns exit 2 into self-deprovisioning.)\n\
         \n\
         join:\n\
         \x20 controller: 127.0.0.1:1337       # the tunnel's loopback end\n\
         \x20 ssh:\n\
         \x20   target: {host}\n",
        label = label,
        host = controller.host,
    );
    if controller.port != 22 {
        out.push_str(&format!("    port: {}\n", controller.port));
    }
    out.push_str(&format!(
        "    user: {user}\n\
         \x20   identity_file: {WORKER_KEY_PATH}\n\
         \x20 token: {token}\n\
         \x20 libtorch: auto                   # routes on THIS box's devices\n",
        user = controller.user,
        token = token,
    ));
    match door {
        Door::B => {
            out.push_str(&format!(
                "  source:\n\
                 \x20   from: rsync://{user}@{host}:/tree   # rrsync re-roots under the served dir\n",
                user = controller.user,
                host = controller.host,
            ));
        }
        Door::A => {
            let data_path = cli.data_path.as_deref().unwrap_or(crate::config::DEFAULT_DATA_PATH);
            out.push_str(&format!(
                "  data_path: {data_path}\n\
                 \x20 data_source: sshfs://{user}@{host}:{data_path}\n\
                 \x20 # door `a` serves the DATA mount; the training binary must be\n\
                 \x20 # provisioned (`bin:`) or pulled through another road.\n\
                 \x20 # bin: /path/to/train\n",
                user = controller.user,
                host = controller.host,
            ));
        }
        Door::Nologin => {
            out.push_str(
                "  # door `nologin` is tunnel-only: provision the binary and any\n\
                 \x20 # data root, then declare them here.\n\
                 \x20 # bin: /path/to/train\n\
                 \x20 # data_path: /flodl/data\n",
            );
        }
    }
    if door != Door::A {
        if let Some(dp) = &cli.data_path {
            out.push_str(&format!("  data_path: {dp}\n"));
        }
    }
    if let Some(share) = cli.gpu_ram_share {
        out.push_str(&format!(
            "  gpu_ram_share: {share}            # this box's APU aperture share\n"
        ));
    }
    out.push_str("  persist: true\n");
    out
}

fn render_notes(
    label: &str,
    authorized_line: &str,
    match_block: &str,
    controller: &Endpoint,
    door: Door,
) -> String {
    let door_name = match door {
        Door::B => "B (rrsync source pull — publish-then-join)",
        Door::A => "A (read-only sftp data mount)",
        Door::Nologin => "nologin (tunnel only)",
    };
    format!(
        "# Farm `{label}` — controller-side install notes\n\n\
         Door: {door_name}\n\n\
         ## 1. authorized_keys ({user}@{host})\n\n\
         Append to `~{user}/.ssh/authorized_keys` (one line):\n\n\
         ```\n{authorized_line}\n```\n\n\
         ## 2. Hardening (optional, recommended for a permanent setup)\n\n\
         A dedicated no-shell user plus the daemon-level mirror of the key\n\
         restrictions, so a mistake in either layer is caught by the other:\n\n\
         ```\n{match_block}\n```\n\n\
         ## 3. The worker side\n\n\
         Copy `keys/{KEY_NAME}` to each worker at `{WORKER_KEY_PATH}` (0600)\n\
         and `worker.yml` to its `fdl.yml`, then `fdl join`. Workers reach\n\
         this box at `{host}:{port}`.\n\n\
         Full recipe rationale: docs/ddp/02-cluster-guide.md.\n",
        label = label,
        door_name = door_name,
        user = controller.user,
        host = controller.host,
        port = controller.port,
        authorized_line = authorized_line,
        match_block = match_block,
    )
}

impl Report {
    fn render_human(&self) -> String {
        let mut out = String::new();
        let push = |out: &mut String, s: &str| {
            out.push_str(s);
            out.push('\n');
        };
        push(&mut out, "");
        push(&mut out, &format!("  farm:      {}", self.label));
        push(&mut out, &format!("  dir:       {}", self.farm_dir.display()));
        let overlay = match self.overlay_action {
            OverlayAction::Scaffolded => "created",
            OverlayAction::TokenReplaced => "token regenerated",
            OverlayAction::TokenReused => "token reused",
            OverlayAction::SnippetPrinted => "NOT edited (user-authored, no token)",
        };
        push(
            &mut out,
            &format!("  overlay:   {} ({overlay})", self.overlay_path.display()),
        );
        let key = match self.key_action {
            KeyAction::Generated => "generated",
            KeyAction::Regenerated => "REGENERATED (old key no longer admits)",
            KeyAction::Reused => "reused",
        };
        push(&mut out, &format!("  join key:  {} ({key})", self.key_path.display()));
        push(&mut out, &format!("  worker yml: {}", self.worker_yml_path.display()));
        push(&mut out, &format!("  notes:     {}", self.notes_path.display()));
        let install = match &self.install {
            InstallAction::Installed => "line appended to ~/.ssh/authorized_keys".to_string(),
            InstallAction::Replaced => {
                "line REPLACED in ~/.ssh/authorized_keys (options updated)".to_string()
            }
            InstallAction::AlreadyPresent => "already in ~/.ssh/authorized_keys".to_string(),
            InstallAction::Skipped(why) => format!("NOT installed ({why}) — see notes"),
        };
        push(&mut out, &format!("  sshd:      {install}"));
        if let Some(ci) = &self.cloud_init_path {
            push(
                &mut out,
                &format!("  cloud-init: {} (SECRET: key + token inside)", ci.display()),
            );
        }
        if let Some(w) = &self.reuse_warning {
            push(&mut out, "");
            push(&mut out, &format!("  WARNING: {w}"));
        }
        if self.overlay_action == OverlayAction::SnippetPrinted {
            push(&mut out, "");
            push(&mut out, "  Your overlay carries no token; add under `cluster.controller.join:`:");
            push(&mut out, "");
            push(&mut out, "      token: <generated — see worker.yml, they must match>");
        }
        push(&mut out, "");
        push(&mut out, "  ── authorized_keys line (controller sshd, one line) ──");
        push(&mut out, "");
        push(&mut out, &format!("  {}", self.authorized_line));
        push(&mut out, "");
        push(&mut out, &format!("  ── worker fdl.yml ({}) ──", self.worker_yml_path.display()));
        push(&mut out, "");
        for line in self.worker_yml.lines() {
            push(&mut out, &format!("  {line}"));
        }
        if let Some(p) = &self.publish_block {
            push(&mut out, "");
            push(&mut out, "  ── publish recipe for the base fdl.yml (then: `fdl publish`) ──");
            push(&mut out, "");
            for line in p.lines() {
                push(&mut out, &format!("  {line}"));
            }
        }
        if let Some(c) = &self.bin_caveat {
            push(&mut out, "");
            push(&mut out, &format!("  note: {c}"));
        }
        if let Some(f) = &self.freshness {
            push(&mut out, "");
            push(&mut out, &format!("  freshness: {f}"));
        }
        push(&mut out, "");
        for line in self.steps() {
            push(&mut out, &line);
        }
        push(&mut out, "");
        push(
            &mut out,
            &style::dim(&format!(
                "  Rationale + hardening notes: {}. The private key is the \
                 worker-bound secret; it never prints here.",
                self.notes_path.display(),
            )),
        );
        push(&mut out, "");
        out
    }

    /// The setup as an ordered list of things to run, each command
    /// reading an artifact the wizard just wrote rather than repeating
    /// its content inline. A first-timer should be able to work down
    /// this list without reading anything else.
    fn steps(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut n = 0;
        let mut step = |out: &mut Vec<String>, title: &str, cmds: &[String]| {
            n += 1;
            out.push(format!("  {n}. {title}"));
            for c in cmds {
                out.push(format!("       {c}"));
            }
            out.push(String::new());
        };

        out.push(format!(
            "  ── setup, in order ({}) ──",
            self.plat.name()
        ));
        out.push(String::new());

        // Anything preflight found missing comes first: the later steps
        // silently do nothing useful without it.
        let gaps: Vec<&Check> = self.checks.iter().filter(|c| !c.ok).collect();
        if !gaps.is_empty() {
            let mut cmds: Vec<String> = Vec::new();
            for c in &gaps {
                cmds.push(format!("# {}", c.what));
                match &c.fix {
                    Some(f) => cmds.push(f.clone()),
                    None => cmds.push("#   (no command for this one here)".to_string()),
                }
            }
            step(&mut out, "this box is missing what the door needs:", &cmds);
        }

        step(
            &mut out,
            "install the sshd drop-in (read it first — it is yours to edit):",
            &{
                let mut c = vec![format!(
                    "sudo install -m 644 {} /etc/ssh/sshd_config.d/flodl-{}.conf",
                    self.sshd_conf_path.display(),
                    self.label,
                )];
                c.extend(self.plat.enable_sshd());
                c.push("sudo sshd -t && echo 'sshd config OK'".to_string());
                if let Some(fw) = self.plat.open_port(self.controller.port) {
                    c.push(fw);
                }
                c
            },
        );

        if !matches!(self.install, InstallAction::Installed | InstallAction::Replaced | InstallAction::AlreadyPresent) {
            step(
                &mut out,
                "authorize the join key (the wizard can do this for you):",
                &[format!("fdl join-config {} --install-key", self.label)],
            );
        }

        // Proving the door BEFORE handing keys to workers is what turns a
        // silent misconfiguration into a two-command answer: under every
        // door a command must be refused while the forward is allowed,
        // and port 22 must still behave normally.
        // Proving the door BEFORE handing keys to workers turns a silent
        // misconfiguration into a two-command answer. Every door must
        // refuse a shell and permit the forward; what else it allows is
        // the door's whole definition, so each gets its own positive
        // test rather than a generic one that would pass on the wrong
        // door.
        let key = self.key_path.display().to_string();
        let (p, u, h) = (self.controller.port, &self.controller.user, &self.controller.host);
        let mut verify = vec![format!(
            "ssh -i {key} -p {p} {u}@{h} true                 # must NOT give a shell"
        )];
        match self.door {
            Door::B => verify.push(format!(
                "rsync --list-only -e 'ssh -i {key} -p {p}' {u}@{h}:/tree/   # must LIST the served tree"
            )),
            Door::A => verify.push(format!(
                "sftp -i {key} -P {p} {u}@{h}                     # must OPEN (read-only)"
            )),
            Door::Nologin => {}
        }
        verify.push(format!(
            "ssh -i {key} -p {p} {u}@{h} -N -L 19337:127.0.0.1:1337   # must CONNECT"
        ));
        verify.push(format!(
            "sudo sshd -T -C user={u},host=x,addr=127.0.0.1,laddr=127.0.0.1,lport={p} \
             | grep -E 'permitopen|forcecommand'"
        ));
        step(&mut out, "prove the door does exactly one thing:", &verify);

        step(
            &mut out,
            "land the worker's config and key on each box:",
            &[
                format!(
                    "scp {} <worker>:{}",
                    self.key_path.display(),
                    WORKER_KEY_PATH,
                ),
                format!("ssh <worker> 'chmod 600 {WORKER_KEY_PATH}'"),
                format!(
                    "scp {} <worker>:<project>/fdl.yml",
                    self.worker_yml_path.display(),
                ),
            ],
        );

        step(
            &mut out,
            "open the window here, then let the boxes dial in:",
            &[
                format!("fdl @{} <your-run-command>      # holds a join window", self.label),
                "ssh <worker> 'cd <project> && fdl join'".to_string(),
                format!("fdl @{} status                  # then: fdl @{} start", self.label, self.label),
            ],
        );

        out
    }

    /// The machine twin: paths and states, secrets as file paths only.
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "label": self.label,
            "farm_dir": self.farm_dir.display().to_string(),
            "overlay": {
                "path": self.overlay_path.display().to_string(),
                "action": match self.overlay_action {
                    OverlayAction::Scaffolded => "scaffolded",
                    OverlayAction::TokenReplaced => "token_replaced",
                    OverlayAction::TokenReused => "token_reused",
                    OverlayAction::SnippetPrinted => "snippet_printed",
                },
            },
            "key": {
                "private_path": self.key_path.display().to_string(),
                "public_line": self.pub_line,
                "action": match self.key_action {
                    KeyAction::Generated => "generated",
                    KeyAction::Regenerated => "regenerated",
                    KeyAction::Reused => "reused",
                },
            },
            "door": match self.door {
                Door::B => "b",
                Door::A => "a",
                Door::Nologin => "nologin",
            },
            "authorized_keys_line": self.authorized_line,
            "platform": self.plat.name(),
            "sshd_conf_path": self.sshd_conf_path.display().to_string(),
            "preflight": self.checks.iter().map(|c| serde_json::json!({
                "what": c.what,
                "ok": c.ok,
                "fix": c.fix,
            })).collect::<Vec<_>>(),
            "steps": self.steps(),
            "sshd_match_block": self.match_block,
            "controller": format!(
                "{}@{}:{}",
                self.controller.user, self.controller.host, self.controller.port,
            ),
            "worker_yml_path": self.worker_yml_path.display().to_string(),
            "notes_path": self.notes_path.display().to_string(),
            "install": match &self.install {
                InstallAction::Installed => serde_json::json!({"action": "installed"}),
                InstallAction::Replaced => serde_json::json!({"action": "replaced"}),
                InstallAction::AlreadyPresent => {
                    serde_json::json!({"action": "already_present"})
                }
                InstallAction::Skipped(why) => {
                    serde_json::json!({"action": "skipped", "why": why})
                }
            },
            "cloud_init_path": self
                .cloud_init_path
                .as_ref()
                .map(|p| p.display().to_string()),
            "publish_block": self.publish_block,
            "bin_caveat": self.bin_caveat,
            "freshness": self.freshness,
            "reuse_warning": self.reuse_warning,
        })
    }
}

#[cfg(test)]
#[path = "join_config_tests.rs"]
mod tests;
