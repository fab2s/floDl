//! `fdl publish` — put a run where the fleet can pull it.
//!
//! The controller side of compiling on the node. It resolves a source
//! spec into a served directory, builds it once, and writes the run
//! manifest workers read. Chaining trainings on a standing fleet is then
//! one command: publish again and every box picks the new run up on its
//! next dial, with nothing to edit on any worker.
//!
//! **The build is validation, not an artifact.** One build gates the
//! publish; each worker still compiles its own, because a controller
//! producing binaries for N worker variants is the build matrix this
//! design deleted. A gate needs no GPU libtorch either — compiling
//! without a GPU feature against the cheap CPU variant catches user-code
//! errors just as well — so the cost of having it on by default is
//! rustup plus `fdl libtorch download --cpu`. What it buys is that a tree
//! which cannot compile never reaches the fleet, where N boxes would each
//! discover it separately, in logs nobody is watching.
//!
//! It proves the tree for the CONTROLLER's variant only. A break that
//! exists solely under `--features rocm` passes a CUDA gate and lands on
//! a worker; superset check, not a proof.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::builtins::PublishArgs;
use crate::context::Context;
use crate::prepare::Fail;
use crate::source::{self, Manifest};
use crate::style;

/// Served-directory name under the root, and the tree inside it. The
/// manifest sits at the tree's root, so one rsync of `<served>/tree`
/// carries both.
const SERVED_SUBDIR: &str = "run";
const TREE_SUBDIR: &str = "tree";

/// Run `fdl publish`. `args_tail` is everything after a standalone `--`:
/// the training binary's own arguments, which belong to the RUN and
/// therefore to the manifest rather than to any worker's config.
///
/// A top-level `publish:` block in fdl.yml (or the active env overlay)
/// supplies standing answers so re-publishing a run is one bare
/// command; flags win field by field, and a `--` tail replaces the
/// block's `args:` outright.
pub fn run(cli: &PublishArgs, args_tail: Option<&[String]>) -> i32 {
    let block = match load_publish_block() {
        Ok(block) => block,
        Err(e) => {
            crate::cli_error!("{e}");
            return 1;
        }
    };
    let (cli, tail) = with_block_defaults(cli, args_tail, block);
    match publish(&cli, tail.as_deref()) {
        Ok((served, tree, manifest)) => {
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report_value(&served, &tree, &manifest))
                        .expect("a report value serializes"),
                );
            } else {
                report(&served, &tree, &manifest);
            }
            0
        }
        Err(fail) => {
            crate::cli_error!("{}", fail.message());
            1
        }
    }
}

/// The top-level `publish:` block from the project config, honoring the
/// active env overlay — the same walk `fdl join` does for its block.
/// `Ok(None)` when no project (or no block) exists: flags then carry
/// everything. A present-but-broken config is a loud error, never a
/// silent fallback.
fn load_publish_block() -> Result<Option<crate::config::PublishBlock>, String> {
    let cwd =
        std::env::current_dir().map_err(|e| format!("cannot read the current directory: {e}"))?;
    let Some(config_path) = crate::config::find_project_config(&cwd) else {
        return Ok(None);
    };
    let env_name = std::env::var("FDL_ENV")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let project = crate::config::load_project_with_env(&config_path, env_name.as_deref())
        .map_err(|e| format!("cannot load {}: {e}", config_path.display()))?;
    Ok(project.publish)
}

/// Fill flag gaps from the block. Flags win field by field; the `--`
/// tail replaces `args:` outright, EVEN WHEN EMPTY — the args belong to
/// the run, so "explicitly none" must be sayable. `--no-build` stays
/// flag-only on purpose: a standing config that skips the gate would
/// ship every future typo to the fleet.
fn with_block_defaults(
    cli: &PublishArgs,
    args_tail: Option<&[String]>,
    block: Option<crate::config::PublishBlock>,
) -> (PublishArgs, Option<Vec<String>>) {
    let block = block.unwrap_or_default();
    let merged = PublishArgs {
        source: cli.source.clone().or(block.source),
        bin: cli.bin.clone().or(block.bin),
        cwd: cli.cwd.clone().or(block.cwd),
        build: cli.build.clone().or(block.build),
        to: cli.to.clone().or(block.to),
        no_build: cli.no_build,
        identity: cli.identity.clone().or(block.identity),
        json: cli.json,
        gate: cli.gate.clone(),
    };
    let tail = match args_tail {
        Some(t) => Some(t.to_vec()),
        None => (!block.args.is_empty()).then_some(block.args),
    };
    (merged, tail)
}

fn publish(
    cli: &PublishArgs,
    args_tail: Option<&[String]>,
) -> Result<(PathBuf, PathBuf, Manifest), Fail> {
    let Some(spec) = cli.source.as_deref() else {
        return Err(Fail::Permanent(
            "fdl publish needs a source to publish, e.g. `fdl publish \
             file:///home/op/my-train --bin target/release/my-train` — or \
             a standing `publish:` block in fdl.yml carrying both"
                .to_string(),
        ));
    };
    let Some(bin) = cli.bin.as_deref() else {
        return Err(Fail::Permanent(
            "fdl publish needs `--bin <path relative to the project dir>` \
             (or `publish.bin:` in fdl.yml) — it is what workers run, and \
             it cannot be guessed (a workspace member's build lands in the \
             WORKSPACE target/, not the member's)"
                .to_string(),
        ));
    };

    // Parse before touching anything. Clearing the manifest takes the
    // fleet out of service until the build passes, and a spec with a typo
    // in it must not cost that.
    let source = source::parse(spec)?;

    let served = match &cli.to {
        Some(dir) => PathBuf::from(dir),
        None => Context::global().root.join(SERVED_SUBDIR),
    };
    let tree = served.join(TREE_SUBDIR);
    std::fs::create_dir_all(&tree).map_err(|e| {
        Fail::Permanent(format!(
            "cannot create the served directory {}: {e}",
            tree.display()
        ))
    })?;

    // Now clear it. Its presence is this command's commit point, so from
    // here until the build passes a worker sees a tree with no run in it
    // and waits for the next dial instead of training something nobody
    // has compiled.
    Manifest::remove(&tree)?;

    let mut notes = Vec::new();
    let result = (|| -> Result<Manifest, Fail> {
        source::materialize(&source, &tree, cli.ssh_config().as_ref(), &mut notes)?;
        let built = if cli.no_build {
            notes.push(
                "--no-build: nothing has compiled this tree, so the first \
                 worker to fetch it is where a broken build will surface"
                    .to_string(),
            );
            if !cli.gate.is_empty() {
                notes.push(
                    "--no-build also skips the --gate check-builds — they \
                     are builds"
                        .to_string(),
                );
            }
            false
        } else {
            build_gate(&tree, cli, bin, &mut notes)?;
            let root = Context::resolve().root;
            for variant in &cli.gate {
                check_gate_variant(&root, &tree, cli, variant, &mut notes)?;
            }
            true
        };
        Ok(Manifest {
            cwd: cli.cwd.clone(),
            build: cli.build.clone(),
            bin: bin.to_string(),
            args: args_tail.map(<[String]>::to_vec).unwrap_or_default(),
            origin: Some(spec.to_string()),
            rustc: rustc_version(),
            published_epoch: unix_seconds(),
            run: Some(run_nonce()),
            built,
        })
    })();
    crate::prepare::print_notes("publish", &notes);
    let manifest = result?;
    manifest.write(&tree)?;
    Ok((served, tree, manifest))
}

/// Compile the tree once, against this box's own libtorch.
fn build_gate(
    tree: &Path,
    cli: &PublishArgs,
    bin: &str,
    notes: &mut Vec<String>,
) -> Result<(), Fail> {
    let ctx = Context::resolve();
    let libtorch = crate::libtorch::detect::active_variant(&ctx.root);
    if libtorch.is_none() {
        notes.push(format!(
            "no active libtorch under {} — the gate builds without \
             LIBTORCH_PATH, which anything linking flodl will refuse. \
             `fdl libtorch download --cpu` is enough for a gate (it \
             validates the tree, it does not ship the binary).",
            ctx.root.display(),
        ));
    }
    let env = source::build_env(libtorch.as_ref());
    // A gate failure is not the worker-side "wait for the next publish":
    // the operator is standing right here, so it is theirs to fix now.
    source::build(
        tree,
        cli.cwd.as_deref(),
        cli.build.as_deref(),
        bin,
        &env,
        notes,
    )
    .map(|_| ())
    .map_err(|e| {
        Fail::Permanent(format!(
            "{}. Nothing was published, so the fleet keeps running \
                 whatever it had",
            e.message(),
        ))
    })
}

/// One `--gate <variant>` check-build: the same recipe against a named
/// libtorch variant, so a break that exists only under the other
/// vendor's feature dies here instead of on a worker. Linking needs no
/// GPU — a CPU-only controller proves both vendors this way — but a
/// flodl-linking crate still needs the vendor's dev headers on this
/// box: libtorch bundles runtime libraries, not headers, and flodl-sys'
/// pre-flight fails the gate loudly with the exact package line.
///
/// The compile runs under its own `CARGO_TARGET_DIR` so each variant's
/// incremental cache stays warm (a shared target/ would rebuild the
/// world on every `LIBTORCH_PATH` flip) — which also moves the artifact
/// away from the `bin:` convention, so success alone is the verdict
/// (`source::check_build`).
fn check_gate_variant(
    root: &Path,
    tree: &Path,
    cli: &PublishArgs,
    variant: &str,
    notes: &mut Vec<String>,
) -> Result<(), Fail> {
    let dir = root.join("libtorch").join(variant);
    if !dir.join("lib").is_dir() {
        return Err(Fail::Permanent(format!(
            "--gate {variant}: no libtorch at {} — `fdl libtorch download` \
             can fetch it (a check-build needs no GPU, only the libraries \
             to link against)",
            dir.display(),
        )));
    }
    let mut env = source::build_env(Some(&(dir, variant.to_string())));
    env.push((
        "CARGO_TARGET_DIR".to_string(),
        format!("target/gate/{}", variant.replace('/', "-")),
    ));
    notes.push(format!("gate: check-building against {variant}"));
    source::check_build(tree, cli.cwd.as_deref(), cli.build.as_deref(), &env, notes).map_err(|e| {
        Fail::Permanent(format!(
            "--gate {variant}: {}. Nothing was published, so the fleet \
                 keeps running whatever it had",
            e.message(),
        ))
    })
}

/// The report as data — the single source both renderers draw from, so
/// the JSON twin cannot drift from the human text. `worker_specs`
/// carries BOTH source-spec spellings because which one is right is a
/// property of the serving key (plain ssh vs rrsync-guardrailed), which
/// only the reader knows.
fn report_value(served: &Path, tree: &Path, manifest: &Manifest) -> serde_json::Value {
    let host = crate::cluster::resolve_local_hostname();
    serde_json::json!({
        "tree": tree.display().to_string(),
        "served": served.display().to_string(),
        "built": manifest.built,
        "toolchain": manifest.rustc,
        "args": manifest.args,
        "run": manifest.run,
        "origin": manifest.origin,
        "published_epoch": manifest.published_epoch,
        "host": host,
        "worker_specs": {
            "plain": format!("rsync://{}:{}", host, tree.display()),
            "rrsync": format!("rsync://{host}:/{TREE_SUBDIR}"),
        },
    })
}

/// What the operator needs to hand a worker, and what the run now is.
fn report(served: &Path, tree: &Path, manifest: &Manifest) {
    println!();
    println!("  published: {}", tree.display());
    if !manifest.built {
        println!("  build:     SKIPPED (--no-build)");
    }
    if let Some(rustc) = &manifest.rustc {
        println!("  toolchain: {rustc} (advisory)");
    }
    if !manifest.args.is_empty() {
        println!("  args:      {}", manifest.args.join(" "));
    }
    let host = crate::cluster::resolve_local_hostname();
    if let Some(run) = &manifest.run {
        println!("  run:       {run} (the join window refuses a cohort mixing ids)");
    }
    println!();
    println!("  Workers pull it with a source spec pointing here. TWO spellings,");
    println!("  and the key that serves the pull decides which — they do not mix:");
    println!();
    println!("    # a plain ssh key (no forced command): the absolute path");
    println!("    join:");
    println!("      source:");
    println!("        from: rsync://{}:{}", host, tree.display());
    println!();
    println!(
        "    # a guardrailed key, forced command=\"rrsync -ro {}\":",
        served.display(),
    );
    println!("    # rrsync re-roots every requested path under its directory,");
    println!("    # so the worker asks for /{TREE_SUBDIR} — the absolute path would");
    println!("    # double-root and fail");
    println!("    join:");
    println!("      source:");
    println!("        from: rsync://{host}:/{TREE_SUBDIR}");
    println!();
    println!(
        "  {}",
        style::dim(
            "cwd / bin / build / args come from the manifest beside the \
             tree, so a worker's own config carries only what is stable \
             for that box. Re-publish to change the run; every box picks \
             it up on its next dial."
        ),
    );
    println!(
        "  {}",
        style::dim(&format!(
            "Adjust `{host}` to how workers reach this box, and add \
             `user@` when the serving key lives on a dedicated user \
             (docs/ddp/02-cluster-guide.md has the key recipes)."
        )),
    );
}

/// `rustc -V`, for the manifest's advisory line. `None` when there is no
/// toolchain here, which `--no-build` makes legitimate.
fn rustc_version() -> Option<String> {
    let out = Command::new("rustc").arg("-V").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!v.is_empty()).then_some(v)
}

fn unix_seconds() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// A fresh 16-byte hex nonce per publish — the run's identity at the
/// join window. Not a credential (it travels in a world-readable
/// manifest), so the entropy bar is "two publishes never collide", not
/// secrecy: OS entropy, and time+pid when even that fails — uniqueness
/// survives the fallback, which is why the nonce keeps one while the
/// wizard's token (a secret) refuses instead.
fn run_nonce() -> String {
    let mut bytes = [0u8; 16];
    if getrandom::fill(&mut bytes).is_err() {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
            ^ (std::process::id() as u128);
        bytes[..16].copy_from_slice(&seed.to_le_bytes());
    }
    let mut s = String::with_capacity(32);
    use std::fmt::Write as _;
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(source: &str, bin: &str, to: &str) -> PublishArgs {
        PublishArgs {
            source: Some(source.to_string()),
            bin: Some(bin.to_string()),
            to: Some(to.to_string()),
            cwd: None,
            build: None,
            no_build: true,
            identity: None,
            json: false,
            gate: Vec::new(),
        }
    }

    /// A tree with a manifest, and the manifest is the commit point: it
    /// appears only at the end, and a failing gate leaves none behind.
    #[test]
    fn publishing_lands_a_tree_and_its_manifest() {
        if !crate::util::system::has_command("rsync") {
            return;
        }
        let base = std::env::temp_dir().join(format!("fdl-publish-{}", std::process::id()));
        let (src, served) = (base.join("src"), base.join("served"));
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "// code").unwrap();

        let mut cli = args(
            &format!("file://{}", src.display()),
            "out",
            &served.display().to_string(),
        );
        let tail = vec!["--model".to_string(), "olmo".to_string()];
        publish(&cli, Some(&tail)).unwrap();

        let tree = served.join(TREE_SUBDIR);
        assert!(tree.join("main.rs").is_file(), "the tree was not published");
        let manifest = Manifest::read(&tree).unwrap().expect("a manifest");
        assert_eq!(manifest.bin, "out");
        assert_eq!(manifest.args, tail);
        assert!(
            !manifest.built,
            "--no-build must be recorded, not glossed over"
        );
        assert_eq!(
            manifest.origin.as_deref(),
            Some(&*format!("file://{}", src.display()))
        );

        // Now a real gate that fails: the manifest must be GONE, because a
        // worker reading one would be told a broken tree is ready.
        cli.no_build = false;
        cli.build = Some("exit 7".to_string());
        let err = publish(&cli, None).unwrap_err();
        assert!(
            err.message().contains("Nothing was published"),
            "got: {err:?}"
        );
        assert_eq!(
            Manifest::read(&tree).unwrap(),
            None,
            "a failed gate left a manifest"
        );

        // And a gate that passes writes it again, with the build recorded.
        cli.build = Some("printf x > out".to_string());
        publish(&cli, None).unwrap();
        let manifest = Manifest::read(&tree).unwrap().expect("a manifest");
        assert!(manifest.built);
        assert!(
            manifest.args.is_empty(),
            "no tail means no args, not the previous ones"
        );

        // Every publish is a NEW run identity — chaining "same args, new
        // code" is the common re-publish, which any content hash would
        // call identical. The nonce is what lets the join window refuse
        // a cohort straddling this boundary.
        let first_run = manifest.run.clone().expect("a publish stamps a run id");
        publish(&cli, None).unwrap();
        let manifest = Manifest::read(&tree).unwrap().expect("a manifest");
        assert_ne!(
            manifest.run,
            Some(first_run),
            "a re-publish must mint a fresh id"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// A spec that does not parse must not cost the fleet its manifest:
    /// clearing it takes every box out of service until a build passes, so
    /// a typo would idle the fleet over something that never reached the
    /// tree.
    #[test]
    fn a_bad_spec_leaves_the_published_run_alone() {
        let base = std::env::temp_dir().join(format!("fdl-publish-typo-{}", std::process::id()));
        let tree = base.join(TREE_SUBDIR);
        std::fs::create_dir_all(&tree).unwrap();
        let live = Manifest {
            bin: "target/release/train".into(),
            built: true,
            ..Default::default()
        };
        live.write(&tree).unwrap();

        let cli = args(
            "nonsense-with-no-scheme",
            "out",
            &base.display().to_string(),
        );
        assert!(publish(&cli, None).is_err());
        assert_eq!(
            Manifest::read(&tree).unwrap(),
            Some(live),
            "the live run was cleared"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The report's pairing, proven against the real tool: behind
    /// `command="rrsync -ro <served>"` a worker's spec path is `/tree`,
    /// and the absolute spelling double-roots under the served dir and
    /// fails. This is a COMPOSITION failure — each printed line was
    /// individually right while the pair was unfollowable — so the
    /// guard runs the composed recipe, not the pieces. Skipped where
    /// rrsync is absent (it ships in the rsync package).
    #[test]
    #[cfg(unix)]
    fn the_rrsync_pairing_serves_tree_and_refuses_the_absolute_path() {
        use std::os::unix::fs::PermissionsExt;
        if !crate::util::system::has_command("rsync") {
            return;
        }
        // Presence is not runnability: Ubuntu's rrsync is a python3
        // script, so on a box holding rrsync but not python3 the spawn
        // "succeeds" and /usr/bin/env exits 127. Probe by running it —
        // argument errors prove the interpreter is there.
        match Command::new("rrsync").output() {
            Err(_) => return,
            Ok(o) if o.status.code() == Some(127) => return,
            Ok(_) => {}
        }
        let base = std::env::temp_dir().join(format!("fdl-publish-rr-{}", std::process::id()));
        let served = base.join("served");
        let src = base.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("main.rs"), "// code").unwrap();
        let cli = args(
            &format!("file://{}", src.display()),
            "out",
            &served.display().to_string(),
        );
        publish(&cli, None).unwrap();
        let tree = served.join(TREE_SUBDIR);

        // An ssh stand-in that hands the client's command to rrsync the
        // way a forced authorized_keys command would.
        let rsh = base.join("rsh.sh");
        std::fs::write(
            &rsh,
            format!(
                "#!/bin/sh\nshift\nSSH_ORIGINAL_COMMAND=\"$*\" exec rrsync -ro {}\n",
                served.display(),
            ),
        )
        .unwrap();
        std::fs::set_permissions(&rsh, std::fs::Permissions::from_mode(0o755)).unwrap();
        // `sh <script>` rather than the script alone: rsync EXECS what
        // `-e` names, and exec'ing a file this multithreaded test binary
        // has just written races any other test's fork (the inherited
        // write fd makes it ETXTBSY). Reading it as sh's argument cannot.
        let rsh_cmd = format!("sh {}", rsh.display());
        let fetch = |path: &str, dest: &str| {
            Command::new("rsync")
                .args([
                    "-a",
                    "-e",
                    &rsh_cmd,
                    &format!("fake:{path}/"),
                    &base.join(dest).display().to_string(),
                ])
                .output()
                .unwrap()
        };

        let ok = fetch(&format!("/{TREE_SUBDIR}"), "out-tree");
        assert!(
            ok.status.success(),
            "{}",
            String::from_utf8_lossy(&ok.stderr)
        );
        assert!(
            base.join("out-tree").join(source::MANIFEST_FILE).is_file(),
            "the /tree spelling must deliver the manifest with the source",
        );

        let refused = fetch(&tree.display().to_string(), "out-abs");
        assert!(
            !refused.status.success(),
            "the absolute path must NOT resolve behind rrsync — if this \
             starts passing, the report's pairing text is stale",
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// The flags-over-block contract, on a literal block (no file IO —
    /// the loader is `fdl join`'s own walk, already covered there).
    #[test]
    fn the_publish_block_fills_gaps_and_flags_win() {
        let block = crate::config::PublishBlock {
            source: Some("file:///srv/train".into()),
            bin: Some("target/release/train".into()),
            cwd: Some("member".into()),
            build: Some("./ci/build.sh".into()),
            to: Some("/srv/run".into()),
            identity: Some("/etc/flodl/pub_key".into()),
            args: vec!["--model".into(), "olmo".into()],
        };

        // Bare `fdl publish`: the block carries everything.
        let bare = PublishArgs {
            source: None,
            bin: None,
            to: None,
            cwd: None,
            build: None,
            no_build: false,
            identity: None,
            json: false,
            gate: Vec::new(),
        };
        let (merged, tail) = with_block_defaults(&bare, None, Some(block.clone()));
        assert_eq!(merged.source.as_deref(), Some("file:///srv/train"));
        assert_eq!(merged.bin.as_deref(), Some("target/release/train"));
        assert_eq!(merged.cwd.as_deref(), Some("member"));
        assert_eq!(merged.build.as_deref(), Some("./ci/build.sh"));
        assert_eq!(merged.to.as_deref(), Some("/srv/run"));
        assert_eq!(merged.identity.as_deref(), Some("/etc/flodl/pub_key"));
        assert_eq!(
            tail.as_deref(),
            Some(&["--model".to_string(), "olmo".to_string()][..])
        );

        // Flags win field by field, and an EMPTY `--` tail replaces the
        // block's args — explicitly none is a sayable answer.
        let flags = PublishArgs {
            source: Some("rsync://exa:/home/op/tree".into()),
            bin: None,
            to: None,
            cwd: None,
            build: None,
            no_build: false,
            identity: None,
            json: false,
            gate: Vec::new(),
        };
        let empty: Vec<String> = Vec::new();
        let (merged, tail) = with_block_defaults(&flags, Some(&empty), Some(block));
        assert_eq!(merged.source.as_deref(), Some("rsync://exa:/home/op/tree"));
        assert_eq!(merged.bin.as_deref(), Some("target/release/train"));
        assert_eq!(
            tail.as_deref(),
            Some(&[][..]),
            "an empty tail must replace, not defer"
        );

        // No block at all: flags and tail pass through untouched.
        let (merged, tail) = with_block_defaults(&bare, None, None);
        assert_eq!(merged.source, None);
        assert_eq!(tail, None);
    }

    /// The report's two renderings draw from one value; this pins the
    /// machine twin's shape (the dashboard contract).
    #[test]
    fn the_json_report_carries_both_worker_spellings() {
        let manifest = Manifest {
            bin: "target/release/train".into(),
            args: vec!["--model".into(), "olmo".into()],
            run: Some("a1b2c3d4e5f60718".into()),
            rustc: Some("rustc 1.90.0".into()),
            built: true,
            ..Default::default()
        };
        let v = report_value(Path::new("/srv/run"), Path::new("/srv/run/tree"), &manifest);
        assert_eq!(v["tree"], "/srv/run/tree");
        assert_eq!(v["served"], "/srv/run");
        assert_eq!(v["built"], true);
        assert_eq!(v["run"], "a1b2c3d4e5f60718");
        assert_eq!(v["args"], serde_json::json!(["--model", "olmo"]));
        let host = crate::cluster::resolve_local_hostname();
        assert_eq!(
            v["worker_specs"]["plain"],
            format!("rsync://{host}:/srv/run/tree")
        );
        assert_eq!(v["worker_specs"]["rrsync"], format!("rsync://{host}:/tree"));
    }

    /// `--gate <variant>`: a missing variant names the fetch, a present
    /// one runs the recipe with that variant's env (the rocm feature
    /// derivation and the per-variant CARGO_TARGET_DIR are what the
    /// probe recipe asserts), and a recipe failure publishes nothing.
    #[test]
    fn a_gate_variant_check_builds_with_that_variants_env() {
        let base = std::env::temp_dir().join(format!("fdl-publish-gate-{}", std::process::id()));
        let (root, tree) = (base.join("root"), base.join("tree"));
        std::fs::create_dir_all(&tree).unwrap();
        let mut cli = args("file:///unused", "out", "/unused");

        // Absent variant: permanent, names the fetch.
        let err =
            check_gate_variant(&root, &tree, &cli, "precompiled/rocm7.0", &mut vec![]).unwrap_err();
        assert!(
            err.message().contains("fdl libtorch download"),
            "got: {err:?}"
        );

        // Present variant: the recipe sees the rocm feature and its own
        // target dir, and success needs no artifact anywhere.
        std::fs::create_dir_all(root.join("libtorch/precompiled/rocm7.0/lib")).unwrap();
        cli.build = Some(
            "test \"$FDL_GPU_FEATURE\" = rocm && \
             test \"$CARGO_TARGET_DIR\" = target/gate/precompiled-rocm7.0"
                .to_string(),
        );
        check_gate_variant(&root, &tree, &cli, "precompiled/rocm7.0", &mut vec![])
            .expect("the env probe recipe must pass");

        // A failing check-build reports as "nothing was published".
        cli.build = Some("exit 3".to_string());
        let err =
            check_gate_variant(&root, &tree, &cli, "precompiled/rocm7.0", &mut vec![]).unwrap_err();
        assert!(
            err.message().contains("Nothing was published"),
            "got: {err:?}"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_missing_source_or_bin_says_which() {
        let mut cli = args("file:///nowhere", "out", "/tmp/fdl-publish-none");
        cli.source = None;
        assert!(
            publish(&cli, None)
                .unwrap_err()
                .message()
                .contains("needs a source")
        );
        let mut cli = args("file:///nowhere", "out", "/tmp/fdl-publish-none");
        cli.bin = None;
        assert!(publish(&cli, None).unwrap_err().message().contains("--bin"));
    }

    #[test]
    fn the_manifest_round_trips_through_yaml() {
        let dir = std::env::temp_dir().join(format!("fdl-publish-m-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = Manifest {
            cwd: Some("ddp-bench".into()),
            build: Some("cargo build --release".into()),
            bin: "target/release/ddp-bench".into(),
            args: vec!["--epochs".into(), "3".into()],
            origin: Some("git+https://example.com/o/r#v1".into()),
            rustc: Some("rustc 1.90.0".into()),
            published_epoch: Some(1_780_000_000),
            run: Some("a1b2c3d4e5f60718".into()),
            built: true,
        };
        manifest.write(&dir).unwrap();
        assert_eq!(Manifest::read(&dir).unwrap(), Some(manifest));
        // Absent is a state with meaning, not an error.
        Manifest::remove(&dir).unwrap();
        assert_eq!(Manifest::read(&dir).unwrap(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
