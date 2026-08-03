//! Training preparation — get this box ready before it dials in.
//!
//! `fdl join` runs this once per attempt, strictly BEFORE the tunnel and
//! the dial: admission starts a window deadline, so anything acquired
//! after it burns the deadline. Every step is idempotent, which is what
//! makes `--persist` a provisioning loop for free — a box picks up a
//! changed source on its next re-dial, with no reprovisioning.
//!
//! What it does today: gate on the GPU stack, put the dataset source
//! root where the ranks will look for it, and confirm the node-local
//! directories the data plane writes are actually writable. libtorch and
//! the training binary join the list when they become specs too.
//!
//! Failures split two ways, and the split is the point. `--persist`
//! re-dials forever with backoff, which is right for a controller that
//! is not up yet and wrong for a box with no GPU: without the
//! distinction a misprovisioned instance hot-loops instead of stopping.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::{SshConfig, DEFAULT_DATA_PATH};
use crate::style;

/// Why preparation stopped, and whether trying again could help.
#[derive(Debug, PartialEq, Eq)]
pub enum Fail {
    /// Retrying cannot help: no usable GPU, a spec that does not parse,
    /// a directory that cannot be created. Report and stop, even under
    /// `--persist`.
    Permanent(String),
    /// The next attempt may well work: the far side of a mount is down,
    /// the controller has not opened its window yet. Back off, re-dial.
    Transient(String),
}

impl Fail {
    /// Message without the class tag.
    pub fn message(&self) -> &str {
        match self {
            Fail::Permanent(m) | Fail::Transient(m) => m,
        }
    }

    /// True for [`Fail::Permanent`] — the caller must not re-dial.
    pub fn is_permanent(&self) -> bool {
        matches!(self, Fail::Permanent(_))
    }
}

/// What preparation settled, for the join that follows.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Prepared {
    /// Local dataset source root to hand this host's ranks. `None` when
    /// the box declares no data path — the training binary then keeps
    /// its own default, which is what a run that never mentions data
    /// expects.
    pub data_path: Option<PathBuf>,
}

/// The data half of the join recipe, as resolved from flags over the
/// `join:` block.
#[derive(Debug, Default)]
pub struct DataSpec<'a> {
    /// Local source root (the mountpoint, when `source` is set).
    pub path: Option<&'a str>,
    /// `<scheme>://<target>` transport that establishes `path`.
    pub source: Option<&'a str>,
    /// The join block's tunnel credentials. Reused for the data mount:
    /// in the shape this exists for, the data host IS the controller box
    /// the tunnel already authenticates against, so a second key field
    /// would be surface that only ever repeats the first one. A box that
    /// needs different credentials mounts its source in provisioning and
    /// declares a bare `path`.
    pub ssh: Option<&'a SshConfig>,
}

/// Node-local dataset cache, one level up from what it holds. Mirrors
/// flodl's `data::host_cache::data_cache_dir()` — the `$HOME/.flodl`
/// convention is fdl's, and flodl-cli is zero-dep on flodl by design, so
/// the two spell it separately. Only the parent is checked here: the
/// per-dataset subdirectory below it is the application's business.
const CACHE_SUBPATH: &str = ".flodl/data";

/// Free space below which a local directory gets a warning: smaller than
/// any real corpus, so it is almost certainly not the volume the
/// operator meant to train from.
const LOW_SPACE_KIB: u64 = 1 << 20;

/// Prepare this box. `notes` collects everything worth telling the
/// operator that is not a failure (a reused mount, a tmpfs stage, a
/// nearly-full volume); the caller prints them.
pub fn prepare(spec: &DataSpec, notes: &mut Vec<String>) -> Result<Prepared, Fail> {
    check_gpu_stack()?;
    let data_path = resolve_data_root(spec, notes)?;
    check_local_dirs(notes)?;
    Ok(Prepared { data_path })
}

// ---------------------------------------------------------------------------
// The GPU gate
// ---------------------------------------------------------------------------

/// Refuse to dial from a box that has no rank to offer.
///
/// The agent already rejects an empty device list, but only after
/// admission — by then this host has been counted into a quorum and its
/// failure takes the cohort's formation with it. Same verdict, before the
/// window instead of inside it.
///
/// The bar is deliberately "any usable device", not "nothing to report".
/// `fdl probe` answers a broader question — is everything on this box
/// configured correctly — and flags an unusable card even when other
/// cards work; a mixed box (an AMD iGPU with no ROCm runtime beside two
/// working NVIDIA cards) is a normal, trainable box, and treating
/// probe's findings as a verdict here would refuse it. So the findings
/// stay what they are: the *explanation* when there is genuinely nothing,
/// which is what `require_devices` quotes.
///
/// Masks are applied (`CUDA_VISIBLE_DEVICES=` means zero devices for the
/// rank, whatever is installed) but the vendor filter is not: `fdl` is
/// built for no GPU backend, and the training binary is built for exactly
/// one. So this gate is a superset — it blocks a box with nothing at all,
/// and leaves "these devices are the wrong vendor for me" to the process
/// that knows its own backend.
fn check_gpu_stack() -> Result<(), Fail> {
    flodl_hw::survey_visible()
        .require_devices()
        .map(|_| ())
        .map_err(|why| {
            Fail::Permanent(format!(
                "{why} This box has no rank to offer; `fdl probe` has the \
                 full picture. (A driver still coming up at boot belongs \
                 before `fdl join`, not inside its re-dial loop.)"
            ))
        })
}

// ---------------------------------------------------------------------------
// The dataset source root
// ---------------------------------------------------------------------------

/// Put the source root where the ranks will look, and return that path.
///
/// Three shapes, and the empty one is the common case:
/// - neither field: ship nothing, check nothing.
/// - `path` alone: a root provisioning already placed. Verified, shipped.
/// - `source` (with `path` as the mountpoint, default
///   [`DEFAULT_DATA_PATH`]): established here when it is not already up.
fn resolve_data_root(
    spec: &DataSpec,
    notes: &mut Vec<String>,
) -> Result<Option<PathBuf>, Fail> {
    let Some(source) = spec.source else {
        let Some(path) = spec.path else {
            return Ok(None);
        };
        let path = absolute(path)?;
        verify_source_root(&path)?;
        return Ok(Some(path));
    };

    let mountpoint = absolute(spec.path.unwrap_or(DEFAULT_DATA_PATH))?;
    let target = parse_source(source)?;
    ensure_mountpoint(&mountpoint)?;

    match crate::probe::mounted_at(&mountpoint) {
        Some((mounted_source, fs_type)) => {
            if mounted_source != target.remote {
                notes.push(format!(
                    "{} already carries a mount from `{mounted_source}` \
                     ({fs_type}), not the configured `{}` — leaving it \
                     alone; the ranks will read whatever is mounted \
                     there. Unmount it (`fusermount -u {}`) to let fdl \
                     mount the configured source.",
                    mountpoint.display(),
                    target.remote,
                    mountpoint.display(),
                ));
            } else {
                notes.push(format!(
                    "source root {} already mounted from `{mounted_source}` \
                     ({fs_type})",
                    mountpoint.display(),
                ));
            }
        }
        None => {
            mount_sshfs(&target, &mountpoint, spec.ssh)?;
            notes.push(format!(
                "mounted `{}` read-only at {}",
                target.remote,
                mountpoint.display(),
            ));
        }
    }
    verify_source_root(&mountpoint)?;
    Ok(Some(mountpoint))
}

/// Absolutize a declared path without resolving symlinks: the value is
/// shipped to this host's ranks, and a relative one would resolve against
/// each reader's working directory. Lexical on purpose —
/// `canonicalize` would silently replace what the operator declared with
/// whatever its symlinks point at.
fn absolute(path: &str) -> Result<PathBuf, Fail> {
    std::path::absolute(path).map_err(|e| {
        Fail::Permanent(format!("cannot resolve data path `{path}`: {e}"))
    })
}

/// A readable directory is all a source root has to be — a rank reads it
/// and never writes it, so an unwritable one is the NORMAL case, not a
/// fault (see `flodl::data::host_cache`).
fn verify_source_root(path: &Path) -> Result<(), Fail> {
    if !path.is_dir() {
        return Err(Fail::Permanent(format!(
            "dataset source root {} is not a readable directory — \
             provision it (mount or create it), point `data_path:` \
             somewhere that exists, or set `data_source:` so fdl mounts \
             it here",
            path.display(),
        )));
    }
    std::fs::read_dir(path).map_err(|e| {
        Fail::Permanent(format!(
            "dataset source root {} cannot be listed: {e}",
            path.display(),
        ))
    })?;
    Ok(())
}

/// The mountpoint must exist before anything can be mounted on it.
/// Creating it is provisioning's job when it sits outside `$HOME` (the
/// `/flodl/data` convention needs one `sudo mkdir` per box), so failure
/// names both ways out.
fn ensure_mountpoint(dir: &Path) -> Result<(), Fail> {
    if dir.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(|e| {
        Fail::Permanent(format!(
            "mountpoint {} does not exist and cannot be created: {e} — \
             create it once during provisioning (`sudo mkdir -p {} && \
             sudo chown $USER {}`), or set `data_path:` to a directory \
             this user owns",
            dir.display(),
            dir.display(),
            dir.display(),
        ))
    })
}

// ---------------------------------------------------------------------------
// Source specs
// ---------------------------------------------------------------------------

/// A parsed transport target.
#[derive(Debug, PartialEq, Eq)]
struct SourceTarget {
    /// `[user@]host:/abs/path`, exactly as sshfs takes it — and exactly
    /// as `/proc/mounts` reports it back, which is what makes the
    /// already-mounted comparison a string compare.
    remote: String,
    /// Non-default ssh port, when the spec named one.
    port: Option<u16>,
}

/// Split `<scheme>://<rest>`. A value with no `://` has no scheme —
/// which for a data source is an error, because a path that is already
/// mounted belongs in `data_path:`. The three artifacts a box needs
/// (data, libtorch, the flodl source) share this grammar and nothing
/// else; their resolvers have no common shape worth abstracting.
fn split_scheme(spec: &str) -> (Option<&str>, &str) {
    match spec.split_once("://") {
        Some((scheme, rest)) => (Some(scheme), rest),
        None => (None, spec),
    }
}

/// Parse a `data_source:` value. Unsupported schemes error loudly rather
/// than being stubbed: a box that cannot reach its data must say so here,
/// not fail mid-epoch.
fn parse_source(spec: &str) -> Result<SourceTarget, Fail> {
    match split_scheme(spec) {
        (Some("sshfs"), rest) => parse_sshfs(rest),
        (Some(scheme), _) => Err(Fail::Permanent(format!(
            "unsupported data_source scheme `{scheme}://` — fdl ships \
             `sshfs://` today. A source another tool already mounted \
             needs no scheme: name its path in `data_path:` instead"
        ))),
        (None, _) => Err(Fail::Permanent(format!(
            "data_source `{spec}` names no transport — a source that is \
             already mounted goes in `data_path:`; a source fdl should \
             mount needs a scheme, e.g. \
             `sshfs://user@host:/flodl/data`"
        ))),
    }
}

/// `[user@]host[:port]/abs/path`, and the scp spelling
/// `[user@]host:/abs/path` for the same thing — sshfs itself takes the
/// second, so refusing it would be a gratuitous trap.
fn parse_sshfs(rest: &str) -> Result<SourceTarget, Fail> {
    let bad = |why: &str| {
        Fail::Permanent(format!(
            "invalid data_source `sshfs://{rest}` — {why}. Expected \
             `sshfs://[user@]host[:port]/abs/path` (or the scp spelling \
             `sshfs://[user@]host:/abs/path`)"
        ))
    };
    let (user, hostpart) = match rest.split_once('@') {
        Some(("", _)) => return Err(bad("empty user before `@`")),
        Some((u, h)) => (Some(u), h),
        None => (None, rest),
    };

    // Split host from path at whichever delimiter comes first. A `:`
    // followed by digits is a port; a `:` followed by `/` is the scp
    // separator.
    let colon = hostpart.find(':');
    let slash = hostpart.find('/');
    let (host, port, path) = match (colon, slash) {
        (Some(c), s) if s.is_none_or(|s| c < s) => {
            let after = &hostpart[c + 1..];
            if after.starts_with('/') {
                (&hostpart[..c], None, after)
            } else {
                let end = after.find('/').ok_or_else(|| bad("no remote path"))?;
                let port = after[..end]
                    .parse::<u16>()
                    .map_err(|_| bad("port is not a number"))?;
                (&hostpart[..c], Some(port), &after[end..])
            }
        }
        (_, Some(s)) => (&hostpart[..s], None, &hostpart[s..]),
        (_, None) => return Err(bad("no remote path")),
    };
    if host.is_empty() {
        return Err(bad("empty host"));
    }
    if path.len() < 2 {
        return Err(bad("the remote path must be absolute"));
    }
    Ok(SourceTarget {
        remote: match user {
            Some(u) => format!("{u}@{host}:{path}"),
            None => format!("{host}:{path}"),
        },
        port,
    })
}

// ---------------------------------------------------------------------------
// The mount
// ---------------------------------------------------------------------------

/// Establish the source mount. Read-only, and that is load-bearing: a
/// rank reads the source root and never writes it (anything missing is
/// acquired into the node-local cache instead), so `ro` puts the kernel
/// behind an invariant that was previously only a convention. A box that
/// needs a writable share mounts it during provisioning and declares a
/// bare `data_path:`.
///
/// The mount outlives the attempt, and the run: it is provisioning state,
/// which is what makes a re-dial cheap (the next attempt finds it and
/// reuses it). `fusermount -u <mountpoint>` drops it.
fn mount_sshfs(
    target: &SourceTarget,
    mountpoint: &Path,
    ssh: Option<&SshConfig>,
) -> Result<(), Fail> {
    if !crate::util::system::has_command("sshfs") {
        return Err(Fail::Permanent(format!(
            "data_source needs sshfs, which is not installed — \
             `sudo apt install sshfs` (or mount `{}` during provisioning \
             and declare a bare `data_path:`)",
            target.remote,
        )));
    }
    let argv = sshfs_argv(target, mountpoint, ssh);
    let out = Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .map_err(|e| Fail::Permanent(format!("spawn sshfs: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(Fail::Transient(format!(
            "mounting `{}` at {} failed ({}): {}",
            target.remote,
            mountpoint.display(),
            out.status,
            stderr.trim(),
        )));
    }
    // sshfs backgrounds itself only once the mount is established, so a
    // zero exit that left nothing mounted means the far side went away
    // mid-handshake.
    if crate::probe::mounted_at(mountpoint).is_none() {
        return Err(Fail::Transient(format!(
            "sshfs reported success but nothing is mounted at {} — the \
             far side likely dropped the connection",
            mountpoint.display(),
        )));
    }
    Ok(())
}

/// Assemble the sshfs command: user options first (OpenSSH takes the
/// first value it sees per key, so the operator's win), then flodl's
/// defaults. Returned as argv for testability.
fn sshfs_argv(
    target: &SourceTarget,
    mountpoint: &Path,
    ssh: Option<&SshConfig>,
) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "sshfs".into(),
        target.remote.clone(),
        mountpoint.display().to_string(),
    ];
    let mut opt = |v: String| {
        argv.push("-o".into());
        argv.push(v);
    };
    if let Some(ssh) = ssh {
        // The tunnel's own options and key: same box, same trust path.
        if let Some(warning) = crate::cluster::batchmode_override_warning(
            &ssh.options,
            &target.remote,
        ) {
            eprintln!("{warning}");
        }
        for o in &ssh.options {
            opt(o.clone());
        }
        if let Some(id) = &ssh.identity_file {
            opt(format!("IdentityFile={id}"));
        }
    }
    if let Some(port) = target.port {
        opt(format!("port={port}"));
    }
    // ro: the source root is read-only by invariant. reconnect +
    // ServerAlive: a dropped link comes back instead of wedging every
    // read behind a dead channel. BatchMode: never hang on a prompt (a
    // passphrase prompt inside a systemd unit wedges forever).
    for o in [
        "ro",
        "reconnect",
        "ServerAliveInterval=15",
        "ServerAliveCountMax=3",
        "BatchMode=yes",
    ] {
        opt(o.to_string());
    }
    argv
}

// ---------------------------------------------------------------------------
// Node-local directories
// ---------------------------------------------------------------------------

/// Confirm the box can actually write where the data plane will write:
/// the across-run dataset cache, and the within-run disk stage.
///
/// Neither is optional and neither is the source root — a read-only
/// source is normal, an unwritable cache is fatal, and discovering that
/// mid-epoch wastes the whole window.
fn check_local_dirs(notes: &mut Vec<String>) -> Result<(), Fail> {
    match std::env::var_os("HOME") {
        Some(home) => {
            let cache = PathBuf::from(home).join(CACHE_SUBPATH);
            check_writable("dataset cache", &cache, true, notes)?;
        }
        None => notes.push(
            "HOME is unset, so flodl will cache datasets under the temp \
             directory — on a tmpfs that spends RAM, not disk. Set HOME, \
             or pre-provision the source root."
                .to_string(),
        ),
    }
    check_writable("disk stage", &std::env::temp_dir(), false, notes)
}

/// One directory: create it if it is ours to create, prove it is
/// writable by writing, then flag what would merely hurt.
///
/// Writability is proven, never inferred: permissions, ACLs, a
/// read-only mount and a full filesystem all present differently in
/// metadata and identically to a write.
fn check_writable(
    label: &str,
    dir: &Path,
    create: bool,
    notes: &mut Vec<String>,
) -> Result<(), Fail> {
    if create {
        std::fs::create_dir_all(dir).map_err(|e| {
            Fail::Permanent(format!(
                "{label} directory {} cannot be created: {e}",
                dir.display(),
            ))
        })?;
    } else if !dir.is_dir() {
        return Err(Fail::Permanent(format!(
            "{label} directory {} does not exist",
            dir.display(),
        )));
    }
    let probe = dir.join(format!(
        ".fdl-prepare-{}-{}",
        std::process::id(),
        next_probe_id(),
    ));
    let written = std::fs::write(&probe, b"fdl prepare\n");
    let _ = std::fs::remove_file(&probe);
    written.map_err(|e| {
        Fail::Permanent(format!(
            "{label} directory {} is not writable: {e} — training stages \
             data there, so it must be",
            dir.display(),
        ))
    })?;

    if let Some(fs_type) = crate::probe::detect_fs_type(dir) {
        if fs_type == "tmpfs" || fs_type == "ramfs" {
            notes.push(format!(
                "{label} directory {} is on {fs_type} (RAM-backed) — \
                 staging there spends RAM, not disk",
                dir.display(),
            ));
        }
    }
    if let Some(kib) = available_kib(dir) {
        if kib < LOW_SPACE_KIB {
            notes.push(format!(
                "{label} directory {} has {} MiB free — smaller than any \
                 real corpus",
                dir.display(),
                kib / 1024,
            ));
        }
    }
    Ok(())
}

/// Free space in KiB via `df -Pk` (POSIX output: one line per
/// filesystem, never wrapped). `None` when `df` is unavailable or its
/// output does not parse — a missing number is not worth failing a join
/// over.
fn available_kib(dir: &Path) -> Option<u64> {
    let out = Command::new("df").arg("-Pk").arg(dir).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .nth(1)?
        .split_whitespace()
        .nth(3)?
        .parse::<u64>()
        .ok()
}

/// Per-process counter for probe-file names: the pid alone collides
/// between two threads checking the same directory.
fn next_probe_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Print what preparation found. Notes are advisory by construction —
/// every fatal condition already returned a [`Fail`].
pub fn print_notes(notes: &[String]) {
    for note in notes {
        eprintln!("{}", style::dim(&format!("fdl join: {note}")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hardware-independent, and it caught a real bug: the gate first
    /// treated every absence-explaining survey note as a verdict, so a
    /// box with an unusable AMD iGPU beside two working NVIDIA cards
    /// refused to dial. Findings explain an absence; they do not create
    /// one. (No spoofing: `FLODL_TESTING_GPU_JSON` is process-global and
    /// this test binary reads the survey from several tests at once —
    /// spoof-driven cases belong in flodl-hw.)
    #[test]
    fn the_gpu_gate_blocks_exactly_when_there_is_no_usable_device() {
        let usable = !flodl_hw::survey_visible().devices.is_empty();
        assert_eq!(
            check_gpu_stack().is_ok(),
            usable,
            "the gate must follow the device list, not the findings",
        );
    }

    #[test]
    fn a_source_spec_parses_all_three_spellings() {
        // scp spelling — what sshfs itself takes.
        assert_eq!(
            parse_source("sshfs://flodl@exa:/flodl/data").unwrap(),
            SourceTarget { remote: "flodl@exa:/flodl/data".into(), port: None },
        );
        // URL spelling, no user.
        assert_eq!(
            parse_source("sshfs://exa/flodl/data").unwrap(),
            SourceTarget { remote: "exa:/flodl/data".into(), port: None },
        );
        // With a port.
        assert_eq!(
            parse_source("sshfs://flodl@exa:2222/flodl/data").unwrap(),
            SourceTarget { remote: "flodl@exa:/flodl/data".into(), port: Some(2222) },
        );
    }

    #[test]
    fn a_source_spec_rejects_every_broken_shape_permanently() {
        for spec in [
            "/flodl/data",                 // no scheme: belongs in data_path
            "smb://server/share",          // scheme we do not ship
            "sshfs://exa",                 // no path
            "sshfs://exa:2222",            // port, still no path
            "sshfs://exa:banana/data",     // port that is not a number
            "sshfs://@exa:/flodl/data",    // empty user
            "sshfs://:/flodl/data",        // empty host
            "sshfs:///flodl/data",         // empty host, URL spelling
            "sshfs://exa:/",               // root is not a source root
        ] {
            let err = parse_source(spec).unwrap_err();
            assert!(err.is_permanent(), "{spec} should be permanent: {err:?}");
        }
    }

    #[test]
    fn a_bare_path_names_the_field_it_belongs_in() {
        // The one-field grammar would have read this as "already
        // mounted"; the two-field split says where that goes.
        let err = parse_source("/flodl/data").unwrap_err();
        assert!(err.message().contains("data_path:"), "got: {err:?}");
    }

    #[test]
    fn sshfs_argv_puts_user_options_before_the_defaults() {
        let ssh = SshConfig {
            target: Some("ctrl".into()),
            port: Some(2222),
            user: Some("join-user".into()),
            identity_file: Some("/etc/flodl/join_key".into()),
            options: vec!["ServerAliveInterval=5".into()],
        };
        let target = parse_source("sshfs://flodl@exa:2222/flodl/data").unwrap();
        let argv = sshfs_argv(&target, Path::new("/flodl/data"), Some(&ssh));
        assert_eq!(argv[0], "sshfs");
        assert_eq!(argv[1], "flodl@exa:/flodl/data");
        assert_eq!(argv[2], "/flodl/data");
        // First -o value wins in OpenSSH: the user's override must
        // appear before flodl's default of the same key.
        let user_pos = argv.iter().position(|a| a == "ServerAliveInterval=5").unwrap();
        let default_pos = argv.iter().position(|a| a == "ServerAliveInterval=15").unwrap();
        assert!(user_pos < default_pos);
        assert!(argv.contains(&"IdentityFile=/etc/flodl/join_key".to_string()));
        // The port rides the SOURCE spec, not the tunnel block: they can
        // be different hosts.
        assert!(argv.contains(&"port=2222".to_string()));
        assert!(argv.contains(&"BatchMode=yes".to_string()));
        // Read-only is not optional — it is the source-root invariant.
        assert!(argv.contains(&"ro".to_string()));
    }

    #[test]
    fn sshfs_argv_without_an_ssh_block_still_carries_the_defaults() {
        let target = parse_source("sshfs://exa/data").unwrap();
        let argv = sshfs_argv(&target, Path::new("/mnt/d"), None);
        assert!(argv.contains(&"ro".to_string()));
        assert!(argv.contains(&"reconnect".to_string()));
        assert!(!argv.iter().any(|a| a.starts_with("IdentityFile")));
        assert!(!argv.iter().any(|a| a.starts_with("port=")));
    }

    #[test]
    fn no_data_fields_prepares_nothing() {
        let mut notes = Vec::new();
        let got = resolve_data_root(&DataSpec::default(), &mut notes).unwrap();
        assert_eq!(got, None, "a run that never mentions data must ship nothing");
        assert!(notes.is_empty());
    }

    #[test]
    fn a_relative_declared_path_is_shipped_absolute() {
        // The value reaches this host's ranks, so a relative path would
        // resolve against whatever each reader's cwd happens to be.
        let cwd = std::env::current_dir().unwrap();
        let name = format!("fdl-prep-rel-{}-{}", std::process::id(), next_probe_id());
        let dir = cwd.join(&name);
        std::fs::create_dir_all(&dir).unwrap();
        let spec = DataSpec { path: Some(&name), ..Default::default() };
        let got = resolve_data_root(&spec, &mut Vec::new()).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, Some(dir));
    }

    #[test]
    fn a_declared_path_is_verified_and_returned() {
        let dir = std::env::temp_dir().join(format!(
            "fdl-prep-src-{}-{}",
            std::process::id(),
            next_probe_id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.display().to_string();
        let mut notes = Vec::new();
        let spec = DataSpec { path: Some(&path), ..Default::default() };
        assert_eq!(
            resolve_data_root(&spec, &mut notes).unwrap(),
            Some(dir.clone()),
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_declared_path_that_is_not_there_is_permanent() {
        let missing = std::env::temp_dir()
            .join("fdl-prep-absent-do-not-create")
            .display()
            .to_string();
        let spec = DataSpec { path: Some(&missing), ..Default::default() };
        let err = resolve_data_root(&spec, &mut Vec::new()).unwrap_err();
        assert!(err.is_permanent(), "got: {err:?}");
        // Every fix, named: provision it, repoint it, or let fdl mount it.
        assert!(err.message().contains("data_source:"), "got: {err:?}");
    }

    #[test]
    fn a_readable_source_root_needs_no_write_permission() {
        // The invariant under test: ranks read the source root and never
        // write it, so read-only must pass. A root-running suite can
        // write anywhere, so this asserts the CHECK's shape rather than
        // trying to build an unwritable directory.
        let dir = std::env::temp_dir().join(format!(
            "fdl-prep-ro-{}-{}",
            std::process::id(),
            next_probe_id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            perms.set_mode(0o555);
        }
        std::fs::set_permissions(&dir, perms).unwrap();
        assert!(verify_source_root(&dir).is_ok());
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            perms.set_mode(0o755);
        }
        std::fs::set_permissions(&dir, perms).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_writable_directory_passes_and_leaves_no_probe_file_behind() {
        let dir = std::env::temp_dir().join(format!(
            "fdl-prep-w-{}-{}",
            std::process::id(),
            next_probe_id(),
        ));
        let mut notes = Vec::new();
        check_writable("test", &dir, true, &mut notes).unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(leftovers.is_empty(), "probe file left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_directory_we_must_not_create_is_permanent() {
        let dir = std::env::temp_dir().join("fdl-prep-absent-stage-dir");
        let err = check_writable("disk stage", &dir, false, &mut Vec::new())
            .unwrap_err();
        assert!(err.is_permanent(), "got: {err:?}");
    }

    #[test]
    fn free_space_reads_back_for_a_directory_that_exists() {
        // Skip where `df` is absent rather than assert a number: the
        // point is that the parse lines up with real output.
        if !crate::util::system::has_command("df") {
            return;
        }
        let kib = available_kib(&std::env::temp_dir());
        assert!(kib.is_some_and(|k| k > 0), "got: {kib:?}");
    }
}
