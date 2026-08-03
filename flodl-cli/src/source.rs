//! The training source: a spec, a local tree, a build.
//!
//! A box that compiles its own training binary links against the exact
//! libtorch it holds, which is what makes the ABI match by construction
//! rather than by manifest discipline. It needs the tree on LOCAL disk
//! first: cargo fingerprints by stat'ing every source file on every
//! invocation, so building over a network mount pays that latency
//! thousands of times before a line compiles, and the attribute caching
//! that would fix the latency makes cargo miss real changes and hand
//! back a stale binary.
//!
//! So a mount is a transport for the fetch, never a compile location,
//! and every spec lands the same way: materialise into a local
//! directory, then build there. One code path, which is why the dev loop
//! is exercised by the production path instead of being a second mode.
//!
//! **The fetch must preserve mtimes.** A copy that stamps every file
//! fresh makes cargo rebuild everything, so the loop silently degrades
//! to cold builds while still looking incremental. `rsync -a` preserves
//! them; a git fetch plus checkout only writes what changed. Both
//! behave, and nothing else here is allowed to.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::config::SshConfig;
use crate::prepare::Fail;
use crate::spec::{parse_ssh_target, split_scheme, SshTarget};

/// Paths the fetch skips, and `--delete` leaves an excluded path on the
/// receiver alone, which is what keeps a local build alive across a
/// refetch and so keeps the loop incremental.
///
/// `target/` and `.git/` are deliberately NOT anchored to the transfer
/// root: cargo writes into the target dir of whichever manifest it built,
/// so a `cwd:` naming a subdirectory (a workspace-excluded crate, say)
/// puts the build under `<cwd>/target/` and an anchored `/target/` would
/// protect the wrong one — the refetch then deletes the build every dial
/// and every dial is a cold one, wearing an incremental costume. Found
/// exactly that way, on a box, with a two-line rehearsal that a passing
/// unit suite had nothing to say about.
///
/// `libtorch/` stays anchored, because it is the fdl project convention
/// for one specific directory and a user's tree may legitimately carry
/// its own `vendor/libtorch/` full of source.
const RSYNC_EXCLUDES: [&str; 3] = ["target/", "libtorch/", ".git/"];

/// The one exclude that means the project root and not any directory of
/// that name. Kept beside its list so the asymmetry is visible.
const ROOT_ANCHORED: [&str; 1] = ["libtorch/"];

/// The default build recipe. A crate that builds locally already carries
/// its `Cargo.toml`, its lockfile and its `rust-toolchain.toml` if the
/// operator pinned one, so the recipe is usually just this — and a
/// project that needs more (a feature flag, a workspace member, a
/// script) says so rather than having fdl guess.
const DEFAULT_BUILD: &str = "cargo build --release";

/// Where a source tree comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A directory on this box: a mount, a second disk, a checkout the
    /// operator placed. Copied to local disk rather than built in place.
    Local(PathBuf),
    /// A working tree pulled over ssh. The one transport that carries
    /// uncommitted work, which is what a training crate that lives in no
    /// repo at all needs.
    Rsync(SshTarget),
    /// A checkout at a pinned ref.
    Git { url: String, git_ref: String },
}

/// A built training binary and the directory it expects as cwd.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    pub bin: PathBuf,
    pub cwd: PathBuf,
}

/// File name of the run manifest, at the root of a published tree.
pub const MANIFEST_FILE: &str = ".fdl-run.yml";

/// The controller's answer to "what is this run", written beside the
/// source it published and read by every box that fetches it.
///
/// It exists because a worker's own config is the wrong place for
/// anything that changes per run. `args` is the sharp case: they must
/// match the run, since rank children re-enter the binary with them, so a
/// standing fleet carrying its own copy trains the next run with the
/// previous one's hyperparameters. Everything stable for a box (its
/// credentials, its libtorch policy, where to pull from) stays local;
/// everything that belongs to the *run* comes from here.
///
/// **Its presence is the publish's commit point.** `fdl publish` removes
/// it before it touches the tree and writes it only once the build has
/// passed, so a box that dials mid-publish, or after a publish whose
/// build failed, finds no manifest and waits for the next dial rather
/// than training something unvalidated.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Project directory inside the tree; `None` = its root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Build recipe; `None` = the default cargo release build.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
    /// Artifact, relative to `cwd`.
    pub bin: String,
    /// The binary's own arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Where the controller got this tree, for provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// `rustc -V` on the controller when it built this, ADVISORY. A
    /// mismatch is worth reporting and not worth enforcing: every box
    /// compiles its own binary, cohort agreement is about model
    /// structure, and a toolchain too old fails loudly at compile time
    /// anyway. Enforcing it would cost a toolchain install per box and
    /// buy what a warning already gives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rustc: Option<String>,
    /// Unix seconds at publish, so a box can say how old its run is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_epoch: Option<u64>,
    /// False when the publish skipped its build gate (`--no-build`), so a
    /// worker can say out loud that nothing has compiled this tree yet.
    #[serde(default)]
    pub built: bool,
}

impl Manifest {
    /// Read the manifest at the root of `tree`. `Ok(None)` when there is
    /// none, which is a state with meaning rather than an error: nobody
    /// has published a run into this tree.
    pub fn read(tree: &Path) -> Result<Option<Manifest>, Fail> {
        let path = tree.join(MANIFEST_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(Fail::Permanent(format!(
                    "cannot read the run manifest {}: {e}",
                    path.display(),
                )));
            }
        };
        serde_yaml_ng::from_str(&text)
            .map(Some)
            .map_err(|e| Fail::Permanent(format!("{} is not a run manifest: {e}", path.display())))
    }

    /// Write the manifest at the root of `tree`.
    pub fn write(&self, tree: &Path) -> Result<(), Fail> {
        let path = tree.join(MANIFEST_FILE);
        let body = serde_yaml_ng::to_string(self)
            .map_err(|e| Fail::Permanent(format!("cannot serialize the run manifest: {e}")))?;
        std::fs::write(
            &path,
            format!(
                "# Written by `fdl publish`. The controller is the authority \
                 for a run:\n# a worker merges this over its own config, \
                 because args must match the run\n# (rank children re-enter \
                 the binary with them). Do not hand-edit — the next\n# \
                 publish overwrites it, and its presence is what tells a \
                 worker the run\n# is ready.\n{body}"
            ),
        )
        .map_err(|e| {
            Fail::Permanent(format!("cannot write the run manifest {}: {e}", path.display()))
        })
    }

    /// Remove the manifest, which is how a publish says "not ready yet".
    pub fn remove(tree: &Path) -> Result<(), Fail> {
        match std::fs::remove_file(tree.join(MANIFEST_FILE)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Fail::Permanent(format!("cannot clear the run manifest: {e}"))),
        }
    }
}

/// Parse a source spec.
///
/// ```text
/// file:///abs/path                      a directory on this box
/// rsync://[user@]host[:port]:/abs/path  a working tree over ssh
/// git+https://host/owner/repo#<ref>     a pinned checkout
/// git+ssh://git@host/owner/repo#<ref>
/// git+file:///abs/repo#<ref>            a local repository or mirror
/// ```
///
/// The scheme names the TOOL rather than a wire protocol (`rsync://` the
/// protocol is the port-873 daemon, which is not what this means), the
/// same convention `data_source:` already uses for `sshfs://`.
pub fn parse(spec: &str) -> Result<Source, Fail> {
    let forms = "Expected `file:///abs/path`, \
                 `rsync://[user@]host[:port]:/abs/path`, or \
                 `git+https://host/owner/repo#<tag|branch|sha>`";
    match split_scheme(spec) {
        (Some("file"), rest) => {
            if !rest.starts_with('/') {
                return Err(Fail::Permanent(format!(
                    "invalid source `{spec}` — a `file://` path must be \
                     absolute (three slashes: `file:///srv/train`). {forms}"
                )));
            }
            Ok(Source::Local(PathBuf::from(rest)))
        }
        (Some("rsync"), rest) => parse_ssh_target(rest).map(Source::Rsync).map_err(|why| {
            Fail::Permanent(format!("invalid source `{spec}` — {why}. {forms}"))
        }),
        // `git+file://` is a local repository or mirror, and it is also
        // what makes this resolver testable without a network.
        (Some(scheme), rest)
            if scheme == "git+https" || scheme == "git+ssh" || scheme == "git+file" =>
        {
            parse_git(scheme, rest, spec, forms)
        }
        (Some(scheme), _) => Err(Fail::Permanent(format!(
            "unsupported source scheme `{scheme}://` — {forms}"
        ))),
        (None, _) => Err(Fail::Permanent(format!(
            "source `{spec}` names no transport — a directory already on \
             this box is `file://` plus its absolute path. {forms}"
        ))),
    }
}

/// `<url>#<ref>`, and the ref is not optional.
///
/// `#` separates it rather than `@` because both alternatives are
/// ambiguous in real specs: a ref may contain `/` (`refs/heads/x`,
/// `feature/y`) and an ssh URL carries `git@host` before the path, so an
/// `@` split picks the wrong side of one or the other. A missing ref
/// would mean the remote's default branch, which floats, and a floating
/// ref is not a pin.
fn parse_git(scheme: &str, rest: &str, spec: &str, forms: &str) -> Result<Source, Fail> {
    let transport = scheme.trim_start_matches("git+");
    let (path, git_ref) = rest.split_once('#').ok_or_else(|| {
        Fail::Permanent(format!(
            "source `{spec}` names no ref — add `#<tag|branch|sha>`. \
             Without one the remote's default branch decides what a box \
             builds, which is not a pin: two boxes provisioned an hour \
             apart would not agree. {forms}"
        ))
    })?;
    if git_ref.is_empty() {
        return Err(Fail::Permanent(format!(
            "source `{spec}` ends at `#` with no ref. {forms}"
        )));
    }
    if path.is_empty() {
        return Err(Fail::Permanent(format!("source `{spec}` names no repository. {forms}")));
    }
    Ok(Source::Git { url: format!("{transport}://{path}"), git_ref: git_ref.to_string() })
}

/// Put the tree at `dest`, preserving mtimes. Idempotent by
/// construction: both resolvers are incremental refreshes, so a re-dial
/// costs the changed files and nothing else.
///
/// `dest` is fdl's directory to manage — `rsync --delete` and `git
/// checkout --force` both make the tree match the spec, so an edit made
/// there does not survive.
pub fn materialize(
    source: &Source,
    dest: &Path,
    ssh: Option<&SshConfig>,
    notes: &mut Vec<String>,
) -> Result<(), Fail> {
    std::fs::create_dir_all(dest).map_err(|e| {
        Fail::Permanent(format!("cannot create source directory {}: {e}", dest.display()))
    })?;
    match source {
        Source::Local(path) => {
            if !path.is_dir() {
                return Err(Fail::Permanent(format!(
                    "source {} is not a readable directory — provision it, \
                     or point `from:` somewhere that exists",
                    path.display(),
                )));
            }
            run_rsync(&rsync_argv(&format!("{}/", path.display()), dest, None, None), dest)?;
            notes.push(format!("source: copied {} into {}", path.display(), dest.display()));
        }
        Source::Rsync(target) => {
            let argv = rsync_argv(&format!("{}/", target.remote), dest, Some(target), ssh);
            run_rsync(&argv, dest)?;
            notes.push(format!("source: pulled {} into {}", target.remote, dest.display()));
        }
        Source::Git { url, git_ref } => {
            run_git(url, git_ref, dest)?;
            notes.push(format!("source: checked out {url} at {git_ref} in {}", dest.display()));
        }
    }
    Ok(())
}

/// Assemble the rsync command. `-a` is what preserves mtimes (and so
/// what keeps cargo incremental); `--delete` is what makes a removed
/// file actually disappear from the node instead of lingering as a stale
/// module. Returned as argv for testability.
fn rsync_argv(
    src: &str,
    dest: &Path,
    target: Option<&SshTarget>,
    ssh: Option<&SshConfig>,
) -> Vec<String> {
    let mut argv: Vec<String> = vec!["rsync".into(), "-a".into(), "--delete".into()];
    for ex in RSYNC_EXCLUDES {
        let anchor = if ROOT_ANCHORED.contains(&ex) { "/" } else { "" };
        argv.push(format!("--exclude={anchor}{ex}"));
    }
    if let Some(target) = target {
        // One `-e` string, which rsync word-splits itself, so the ssh
        // hop's own port / key / options ride along on the same trust
        // path the tunnel uses. Word-split means a key path containing a
        // space cannot travel this way; ssh_config on the box is the
        // answer there, not more quoting.
        let mut ssh_cmd = String::from("ssh");
        if let Some(port) = target.port {
            ssh_cmd.push_str(&format!(" -p {port}"));
        }
        if let Some(ssh) = ssh {
            if let Some(id) = &ssh.identity_file {
                ssh_cmd.push_str(&format!(" -i {id}"));
            }
            for opt in &ssh.options {
                ssh_cmd.push_str(&format!(" -o {opt}"));
            }
        }
        // Never hang on a prompt: a passphrase prompt inside a systemd
        // unit wedges forever.
        ssh_cmd.push_str(" -o BatchMode=yes");
        argv.push("-e".into());
        argv.push(ssh_cmd);
    }
    argv.push(src.to_string());
    argv.push(format!("{}/", dest.display()));
    argv
}

fn run_rsync(argv: &[String], dest: &Path) -> Result<(), Fail> {
    if !crate::util::system::has_command("rsync") {
        return Err(Fail::Permanent(
            "a source spec needs rsync, which is not installed — \
             `sudo apt install rsync` (it is what preserves mtimes, so \
             cargo stays incremental instead of rebuilding everything \
             every dial)"
                .to_string(),
        ));
    }
    let out = Command::new(&argv[0])
        .args(&argv[1..])
        .output()
        .map_err(|e| Fail::Permanent(format!("spawn rsync: {e}")))?;
    if !out.status.success() {
        // Transient on purpose, the same call slice C's tunnel makes: a
        // far side that is down and a path or key that is wrong are not
        // distinguishable from here, and a wrong one keeps saying so
        // loudly once a backoff.
        return Err(Fail::Transient(format!(
            "fetching the source into {} failed ({}): {} — check the \
             remote path, the key, and whether that key's forced command \
             permits rsync (a join key guardrailed with \
             `command=\"/usr/sbin/nologin\"` does not)",
            dest.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim(),
        )));
    }
    Ok(())
}

/// Fetch a pinned ref into `dest`, shallow.
///
/// `git init` + `git fetch <url> <ref>` + `git checkout FETCH_HEAD`
/// rather than `clone --branch`: one path covers a tag, a branch AND a
/// bare commit, every step is idempotent, and no named remote means no
/// remote bookkeeping to keep in sync when the spec changes.
fn run_git(url: &str, git_ref: &str, dest: &Path) -> Result<(), Fail> {
    if !crate::util::system::has_command("git") {
        return Err(Fail::Permanent(
            "a `git+` source spec needs git, which is not installed — \
             `sudo apt install git`"
                .to_string(),
        ));
    }
    let dest_s = dest.display().to_string();
    git(&["init", "--quiet", &dest_s], "initialise")?;
    // Shallow: a node builds a tree, it does not browse history.
    let fetch = git_output(&[
        "-C",
        &dest_s,
        "fetch",
        "--quiet",
        "--depth",
        "1",
        url,
        git_ref,
    ]);
    match fetch {
        Ok(()) => {}
        Err(stderr) => {
            // A bare commit sha can only be fetched when the server
            // allows unadvertised objects (`uploadpack.allowReachableSHA1InWant`).
            // Naming that beats falling back to a full clone, which on a
            // metered box is the cost this whole path exists to avoid.
            if stderr.contains("unadvertised object") || stderr.contains("allow request for") {
                return Err(Fail::Permanent(format!(
                    "the server refused a shallow fetch of `{git_ref}` \
                     ({url}): fetching a bare commit needs \
                     `uploadpack.allowReachableSHA1InWant` on the remote. \
                     Name a tag or branch instead, or push the commit to \
                     a ref. ({stderr})"
                )));
            }
            return Err(Fail::Transient(format!(
                "fetching {url} at `{git_ref}` failed: {stderr}"
            )));
        }
    }
    // --force: the tree is fdl's to manage, so a previous spec's files
    // give way. It does not touch what is untracked, which is what keeps
    // the local `target/` (and its incremental state) alive.
    git(&["-C", &dest_s, "checkout", "--quiet", "--detach", "--force", "FETCH_HEAD"], "check out")?;
    Ok(())
}

fn git(args: &[&str], what: &str) -> Result<(), Fail> {
    git_output(args).map_err(|stderr| {
        Fail::Permanent(format!("git failed to {what} the source tree: {stderr}"))
    })
}

/// Run git, returning its stderr on failure so callers can class it.
fn git_output(args: &[&str]) -> Result<(), String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("spawn git: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
    Err(if stderr.is_empty() { format!("exited {}", out.status) } else { stderr })
}

/// The environment a build recipe gets: the same names an `fdl.yml`
/// `commands.run` line already relies on, so a recipe that works there
/// works here.
///
/// What it deliberately does NOT contain is a feature flag fdl chose.
/// `cuda` and `rocm` are this repo's feature names; a user's crate pins
/// its flodl features in its own manifest and may expose neither, so the
/// vendor is handed over as `$FDL_GPU_FEATURE` for a recipe to use or
/// ignore.
pub fn build_env(libtorch: Option<&(PathBuf, String)>) -> Vec<(String, String)> {
    let Some((dir, variant)) = libtorch else {
        return Vec::new();
    };
    let lib = dir.join("lib").display().to_string();
    let vendor = crate::libtorch::detect::variant_vendor(variant);
    vec![
        // What flodl-sys/build.rs reads to find headers and libraries.
        ("LIBTORCH_PATH".to_string(), dir.display().to_string()),
        // The vendor's cargo feature, so a recipe can say
        // `--features $FDL_GPU_FEATURE` instead of naming a vendor.
        // Falls back to `cuda` exactly as `fdl run` does for the same
        // case, so the variable means one thing everywhere in fdl.
        (
            "FDL_GPU_FEATURE".to_string(),
            vendor.map(|v| v.cargo_feature().to_string()).unwrap_or_else(|| "cuda".to_string()),
        ),
        // Build scripts and the linker both want to find the libs, and
        // on ROCm the ordering is the difference between a working
        // process and a segfault at the first GPU op.
        (
            "LD_LIBRARY_PATH".to_string(),
            crate::libtorch::detect::ld_library_path_value(
                vendor,
                &lib,
                &std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string()),
            ),
        ),
    ]
}

/// Build the tree and hand back the binary.
///
/// `cwd` is the project directory inside the tree (the default is the
/// tree root) and governs both the build and the run, so it answers
/// "where is the project in this tree" once. `cmd` is a shell line, so
/// it can be a script committed beside the code: the recipe then travels
/// with the source while its invocation stays in the box's config. `env`
/// is what fdl resolved for it (libtorch, the vendor's cargo feature,
/// the loader path).
pub fn build(
    tree: &Path,
    cwd: Option<&str>,
    cmd: Option<&str>,
    bin: &str,
    env: &[(String, String)],
    notes: &mut Vec<String>,
) -> Result<Built, Fail> {
    let dir = match cwd {
        Some(sub) => tree.join(sub),
        None => tree.to_path_buf(),
    };
    if !dir.is_dir() {
        return Err(Fail::Permanent(format!(
            "`cwd: {}` names no directory in the fetched source ({}) — it \
             is a path inside the tree, not on this box",
            cwd.unwrap_or(""),
            dir.display(),
        )));
    }
    let recipe = cmd.unwrap_or(DEFAULT_BUILD);
    // Only the default recipe is known to need cargo. A custom one may
    // be a script, a make target, anything.
    if cmd.is_none() && !crate::util::system::has_command("cargo") {
        return Err(Fail::Permanent(
            "building the source needs cargo, which is not installed — \
             install a toolchain (https://rustup.rs), or set `build:` to \
             a recipe that does not need one"
                .to_string(),
        ));
    }

    notes.push(format!("source: building in {} — {recipe}", dir.display()));
    let mut command = Command::new("sh");
    command.args(["-c", recipe]).current_dir(&dir);
    for (k, v) in env {
        command.env(k, v);
    }
    let status = command
        .status()
        .map_err(|e| Fail::Permanent(format!("spawn `{recipe}`: {e}")))?;
    if !status.success() {
        // The fact, with no audience assumed: a worker and a publishing
        // controller both land here and owe the operator different next
        // steps, so each adds its own. Transient by default because the
        // worker is the caller that re-dials, and a compile error is the
        // most transient thing in that system — the fix is a push away,
        // while exiting permanently would let the systemd recipe power a
        // box off over a typo.
        return Err(Fail::Transient(format!(
            "the source does not build ({status}) — see the compiler \
             output above"
        )));
    }

    let path = dir.join(bin);
    if !path.is_file() {
        return Err(Fail::Permanent(format!(
            "the build succeeded but `bin: {bin}` is not there ({}) — it \
             is the artifact path relative to `cwd:`, e.g. \
             `target/release/<name>`",
            path.display(),
        )));
    }
    Ok(Built { bin: path, cwd: dir })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_spelling_parses() {
        assert_eq!(parse("file:///srv/train").unwrap(), Source::Local("/srv/train".into()));
        assert_eq!(
            parse("rsync://flodl@exa:/home/op/train").unwrap(),
            Source::Rsync(SshTarget { remote: "flodl@exa:/home/op/train".into(), port: None }),
        );
        assert_eq!(
            parse("rsync://exa:2222/home/op/train").unwrap(),
            Source::Rsync(SshTarget { remote: "exa:/home/op/train".into(), port: Some(2222) }),
        );
        assert_eq!(
            parse("git+https://github.com/flodl-labs/flodl#0.7.0").unwrap(),
            Source::Git {
                url: "https://github.com/flodl-labs/flodl".into(),
                git_ref: "0.7.0".into(),
            },
        );
        // An ssh URL carries `git@host` and a ref may carry `/`, which is
        // exactly why the separator is `#` and not `@`.
        assert_eq!(
            parse("git+ssh://git@github.com/me/train#feature/wip").unwrap(),
            Source::Git {
                url: "ssh://git@github.com/me/train".into(),
                git_ref: "feature/wip".into(),
            },
        );
    }

    /// The git resolver against a real repository: a shallow fetch of a
    /// ref, a checkout, and then the same tree refetched at a second ref
    /// to prove the incremental path works rather than only the first
    /// clone. `git+file://` is what makes this reachable without a
    /// network, which is the reason that scheme exists.
    #[test]
    fn the_git_resolver_fetches_a_ref_and_then_moves_to_another() {
        if !crate::util::system::has_command("git") {
            return;
        }
        let base = std::env::temp_dir().join(format!("fdl-src-git-{}", std::process::id()));
        let (origin, dest) = (base.join("origin"), base.join("dest"));
        std::fs::create_dir_all(&origin).unwrap();
        let git = |args: &[&str]| {
            let out = Command::new("git")
                .args(args)
                .current_dir(&origin)
                .env("GIT_AUTHOR_NAME", "fdl")
                .env("GIT_AUTHOR_EMAIL", "fdl@example.com")
                .env("GIT_COMMITTER_NAME", "fdl")
                .env("GIT_COMMITTER_EMAIL", "fdl@example.com")
                .output()
                .unwrap();
            assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        };
        git(&["init", "--quiet"]);
        std::fs::write(origin.join("main.rs"), "// one").unwrap();
        git(&["add", "."]);
        git(&["commit", "--quiet", "-m", "one"]);
        git(&["tag", "v1"]);
        std::fs::write(origin.join("main.rs"), "// two").unwrap();
        git(&["add", "."]);
        git(&["commit", "--quiet", "-m", "two"]);
        git(&["tag", "v2"]);

        let url = format!("git+file://{}", origin.display());
        let at_v1 = parse(&format!("{url}#v1")).unwrap();
        materialize(&at_v1, &dest, None, &mut Vec::new()).unwrap();
        assert_eq!(std::fs::read_to_string(dest.join("main.rs")).unwrap(), "// one");

        // A build output is untracked, so moving refs must not sweep it
        // away — that is what keeps the loop incremental here too.
        std::fs::create_dir_all(dest.join("target/release")).unwrap();
        std::fs::write(dest.join("target/release/train"), "binary").unwrap();

        let at_v2 = parse(&format!("{url}#v2")).unwrap();
        materialize(&at_v2, &dest, None, &mut Vec::new()).unwrap();
        assert_eq!(std::fs::read_to_string(dest.join("main.rs")).unwrap(), "// two");
        assert!(dest.join("target/release/train").is_file(), "the checkout swept the build");

        // A ref that does not exist must fail rather than land on
        // whatever the remote's default branch happens to be.
        let missing = parse(&format!("{url}#v9")).unwrap();
        assert!(materialize(&missing, &dest, None, &mut Vec::new()).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_broken_spec_is_permanent_and_names_the_forms() {
        for spec in [
            "/srv/train",                          // no transport
            "file://srv/train",                    // relative file url
            "smb://server/share",                  // scheme we do not ship
            "rsync://exa",                         // no remote path
            "git+https://github.com/me/train",     // no ref: a floating default branch
            "git+https://github.com/me/train#",    // empty ref
            "git+ssh://#0.7.0",                    // no repository
        ] {
            let err = parse(spec).unwrap_err();
            assert!(err.is_permanent(), "{spec} should be permanent: {err:?}");
            assert!(
                err.message().contains("file:///") || err.message().contains("`#<"),
                "{spec} should name the accepted forms: {err:?}",
            );
        }
    }

    #[test]
    fn a_missing_ref_explains_why_a_default_branch_is_not_a_pin() {
        let err = parse("git+https://github.com/me/train").unwrap_err();
        assert!(err.message().contains("not a pin"), "got: {err:?}");
    }

    #[test]
    fn rsync_argv_preserves_times_and_protects_the_target_dir() {
        let argv = rsync_argv("/mnt/rdl/", Path::new("/home/op/.flodl/source"), None, None);
        assert_eq!(argv[0], "rsync");
        // -a is the mtime guarantee, and the whole loop rests on it.
        assert!(argv.contains(&"-a".to_string()));
        assert!(argv.contains(&"--delete".to_string()));
        // UNANCHORED, and that is the whole point: a `cwd:` subdirectory
        // holds its own target dir, and `/target/` would protect only the
        // root one — every refetch would then delete the build.
        assert!(argv.contains(&"--exclude=target/".to_string()));
        assert!(!argv.contains(&"--exclude=/target/".to_string()));
        // Anchored, because it names the project convention rather than
        // any directory called libtorch.
        assert!(argv.contains(&"--exclude=/libtorch/".to_string()));
        // Trailing slashes on both ends: contents into contents, not a
        // nested `source/rdl/`.
        assert_eq!(argv[argv.len() - 2], "/mnt/rdl/");
        assert_eq!(argv[argv.len() - 1], "/home/op/.flodl/source/");
        // No `-e` without a remote: a local copy needs no ssh at all.
        assert!(!argv.contains(&"-e".to_string()));
    }

    #[test]
    fn rsync_argv_carries_the_ssh_hops_port_key_and_options() {
        let ssh = SshConfig {
            target: Some("exa".into()),
            port: Some(22),
            user: None,
            identity_file: Some("/etc/flodl/join_key".into()),
            options: vec!["StrictHostKeyChecking=accept-new".into()],
        };
        let target = SshTarget { remote: "op@exa:/srv/train".into(), port: Some(2222) };
        let argv = rsync_argv("op@exa:/srv/train/", Path::new("/t"), Some(&target), Some(&ssh));
        let e = argv.iter().position(|a| a == "-e").expect("-e for a remote source");
        let cmd = &argv[e + 1];
        // The port comes from the SOURCE spec, not the tunnel block: they
        // can be different hosts.
        assert!(cmd.contains("-p 2222"), "got: {cmd}");
        assert!(cmd.contains("-i /etc/flodl/join_key"), "got: {cmd}");
        assert!(cmd.contains("-o StrictHostKeyChecking=accept-new"), "got: {cmd}");
        assert!(cmd.contains("-o BatchMode=yes"), "got: {cmd}");
    }

    /// The two properties the whole loop rests on, exercised against the
    /// real tool: an old mtime stays old (or cargo rebuilds the world
    /// every dial), and a build under a `cwd:` subdirectory survives the
    /// refetch (or every dial is a cold one wearing an incremental
    /// costume). Skipped where rsync is absent rather than asserted
    /// around, the same call the free-space test makes about `df`.
    #[test]
    fn a_refetch_keeps_an_old_mtime_and_a_nested_build() {
        if !crate::util::system::has_command("rsync") {
            return;
        }
        let base = std::env::temp_dir().join(format!("fdl-src-refetch-{}", std::process::id()));
        let (src, dest) = (base.join("src"), base.join("dest"));
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("sub/lib.rs"), "fn main() {}").unwrap();
        std::fs::write(src.join("gone.txt"), "temporary").unwrap();
        // Stamp the source old, so "did not touch it" is observable
        // rather than inferred from two fresh timestamps agreeing.
        let old = std::time::SystemTime::now() - std::time::Duration::from_secs(86_400);
        std::fs::File::options()
            .write(true)
            .open(src.join("sub/lib.rs"))
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();

        let source = Source::Local(src.clone());
        materialize(&source, &dest, None, &mut Vec::new()).unwrap();
        // A build lands under the subdirectory, where cargo puts it for a
        // manifest that is not the tree root.
        std::fs::create_dir_all(dest.join("sub/target/release")).unwrap();
        std::fs::write(dest.join("sub/target/release/train"), "binary").unwrap();
        std::fs::remove_file(src.join("gone.txt")).unwrap();

        materialize(&source, &dest, None, &mut Vec::new()).unwrap();

        assert!(
            dest.join("sub/target/release/train").is_file(),
            "the refetch deleted a nested build",
        );
        assert!(!dest.join("gone.txt").exists(), "--delete must drop a removed file");
        let copied = std::fs::metadata(dest.join("sub/lib.rs")).unwrap().modified().unwrap();
        let drift = copied.duration_since(old).unwrap_or_default();
        assert!(
            drift < std::time::Duration::from_secs(2),
            "the fetch restamped the file ({drift:?} newer than the source)",
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_cwd_outside_the_fetched_tree_is_permanent() {
        let tree = std::env::temp_dir().join("fdl-src-no-such-tree");
        let err = build(&tree, Some("nope"), Some("true"), "x", &[], &mut Vec::new())
            .unwrap_err();
        assert!(err.is_permanent(), "got: {err:?}");
        assert!(err.message().contains("inside the tree"), "got: {err:?}");
    }

    #[test]
    fn a_build_that_fails_is_transient_so_the_fleet_survives_a_typo() {
        let dir = std::env::temp_dir().join(format!("fdl-src-build-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = build(&dir, None, Some("exit 3"), "bin", &[], &mut Vec::new()).unwrap_err();
        assert!(!err.is_permanent(), "a compile error must not stop the box: {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_build_that_produces_nothing_at_bin_is_permanent() {
        // Config error, not a transient one: retrying cannot make the
        // recipe write somewhere else.
        let dir = std::env::temp_dir().join(format!("fdl-src-nobin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = build(&dir, None, Some("true"), "target/release/x", &[], &mut Vec::new())
            .unwrap_err();
        assert!(err.is_permanent(), "got: {err:?}");
        assert!(err.message().contains("bin:"), "got: {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_env_reaches_the_recipe_and_the_binary_is_returned() {
        let dir = std::env::temp_dir().join(format!("fdl-src-env-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        let env = vec![("FDL_TEST_MARKER".to_string(), "ok".to_string())];
        // The recipe writes the marker's value where `bin:` says, so a
        // pass proves both the env and the cwd arrived.
        let built = build(
            &dir,
            Some("sub"),
            Some("printf %s \"$FDL_TEST_MARKER\" > out"),
            "out",
            &env,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(built.cwd, dir.join("sub"));
        assert_eq!(built.bin, dir.join("sub").join("out"));
        assert_eq!(std::fs::read_to_string(&built.bin).unwrap(), "ok");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
