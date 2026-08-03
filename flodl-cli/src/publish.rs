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
pub fn run(cli: &PublishArgs, args_tail: Option<&[String]>) -> i32 {
    match publish(cli, args_tail) {
        Ok(()) => 0,
        Err(fail) => {
            crate::cli_error!("{}", fail.message());
            1
        }
    }
}

fn publish(cli: &PublishArgs, args_tail: Option<&[String]>) -> Result<(), Fail> {
    let Some(spec) = cli.source.as_deref() else {
        return Err(Fail::Permanent(
            "fdl publish needs a source to publish, e.g. `fdl publish \
             file:///home/op/my-train --bin target/release/my-train`"
                .to_string(),
        ));
    };
    let Some(bin) = cli.bin.as_deref() else {
        return Err(Fail::Permanent(
            "fdl publish needs `--bin <path relative to the project dir>` \
             — it is what workers run, and it cannot be guessed (a \
             workspace member's build lands in the WORKSPACE target/, not \
             the member's)"
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
        Fail::Permanent(format!("cannot create the served directory {}: {e}", tree.display()))
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
            false
        } else {
            build_gate(&tree, cli, bin, &mut notes)?;
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
            built,
        })
    })();
    crate::prepare::print_notes("publish", &notes);
    let manifest = result?;
    manifest.write(&tree)?;

    report(&served, &tree, &manifest);
    Ok(())
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
    source::build(tree, cli.cwd.as_deref(), cli.build.as_deref(), bin, &env, notes)
        .map(|_| ())
        .map_err(|e| {
            Fail::Permanent(format!(
                "{}. Nothing was published, so the fleet keeps running \
                 whatever it had",
                e.message(),
            ))
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
    println!();
    println!("  Workers pull it with a source spec pointing here:");
    println!();
    println!("    join:");
    println!("      source:");
    println!(
        "        from: rsync://{}:{}",
        crate::cluster::resolve_local_hostname(),
        tree.display(),
    );
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
            "The key that serves this needs a forced command scoped to \
             it: command=\"rrsync -ro {}\"",
            served.display(),
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
        assert!(!manifest.built, "--no-build must be recorded, not glossed over");
        assert_eq!(manifest.origin.as_deref(), Some(&*format!("file://{}", src.display())));

        // Now a real gate that fails: the manifest must be GONE, because a
        // worker reading one would be told a broken tree is ready.
        cli.no_build = false;
        cli.build = Some("exit 7".to_string());
        let err = publish(&cli, None).unwrap_err();
        assert!(err.message().contains("nothing was published"), "got: {err:?}");
        assert_eq!(Manifest::read(&tree).unwrap(), None, "a failed gate left a manifest");

        // And a gate that passes writes it again, with the build recorded.
        cli.build = Some("printf x > out".to_string());
        publish(&cli, None).unwrap();
        let manifest = Manifest::read(&tree).unwrap().expect("a manifest");
        assert!(manifest.built);
        assert!(manifest.args.is_empty(), "no tail means no args, not the previous ones");
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
        let live = Manifest { bin: "target/release/train".into(), built: true, ..Default::default() };
        live.write(&tree).unwrap();

        let cli = args("nonsense-with-no-scheme", "out", &base.display().to_string());
        assert!(publish(&cli, None).is_err());
        assert_eq!(Manifest::read(&tree).unwrap(), Some(live), "the live run was cleared");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_missing_source_or_bin_says_which() {
        let mut cli = args("file:///nowhere", "out", "/tmp/fdl-publish-none");
        cli.source = None;
        assert!(publish(&cli, None).unwrap_err().message().contains("needs a source"));
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
