//! Training preparation — get this box ready before it dials in.
//!
//! `fdl join` runs this once per attempt, strictly BEFORE the tunnel and
//! the dial: admission starts a window deadline, so anything acquired
//! after it burns the deadline. Every step is idempotent, which is what
//! makes `--persist` a provisioning loop for free — a box picks up a
//! changed source on its next re-dial, with no reprovisioning.
//!
//! Five steps, in an order that is itself load-bearing. First the cheap
//! ones: gate on the GPU stack, put the dataset source root where the
//! ranks will look for it, prove the node-local directories the data
//! plane writes are writable. Then what a box may not have yet — the
//! training source, a libtorch variant, and the binary built from both —
//! because those take minutes, and discovering an unwritable stage
//! directory after a cold build has wasted the build.
//!
//! Failures split two ways, and the split is the point. `--persist`
//! re-dials forever with backoff, which is right for a controller that
//! is not up yet and wrong for a box with no GPU: without the
//! distinction a misprovisioned instance hot-loops instead of stopping.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::config::{DEFAULT_DATA_PATH, SshConfig};
use crate::context::Context;
use crate::source::{Built, Manifest};
use crate::spec::{SshTarget, parse_ssh_target, split_scheme};
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
    /// The libtorch this box will train against: `(variant directory,
    /// variant label)`. Acquired when a spec asked for one, otherwise
    /// whatever was already active here. One field rather than two
    /// because the answer has one authority: the build links against it
    /// and the ranks load from it.
    pub libtorch: Option<(PathBuf, String)>,
    /// The training binary this box built, and the directory it expects
    /// as cwd. `None` when the operator named a binary instead of a
    /// source.
    pub bin: Option<Built>,
    /// The run's arguments, when a controller published them. They
    /// replace whatever this box carried: args must match the run,
    /// because rank children re-enter the binary with them.
    pub args: Option<Vec<String>>,
    /// The published run's identity nonce, when the manifest carries
    /// one. Rides the join hello so the window can refuse a cohort
    /// straddling a publish boundary; `None` gates nothing.
    pub run_id: Option<String>,
}

/// Everything this box has to settle before it dials.
#[derive(Debug, Default)]
pub struct PrepareSpec<'a> {
    pub data: DataSpec<'a>,
    /// libtorch variant to acquire: `auto`, `cpu`, `cu126`, `cu128`,
    /// `rocm7.0`, `rocm7.1`. `None` leaves this box on whatever it
    /// already has.
    pub libtorch: Option<&'a str>,
    /// This box's already-active libtorch, when fdl found one. Used when
    /// no variant is acquired, and it is what the build links against.
    pub active_libtorch: Option<&'a (PathBuf, String)>,
    /// Training source to fetch and build. `None` when the operator
    /// named an existing binary.
    pub source: Option<SourceSpec<'a>>,
    /// Device ids this box will offer (`--devices`); `None` = all
    /// visible. The arch-coverage gate scopes to these: a half-covered
    /// box explicitly offering only its covered card is a working box.
    pub devices: Option<&'a [u8]>,
}

/// The source half of the join recipe.
#[derive(Debug, Default)]
pub struct SourceSpec<'a> {
    /// Transport plus location (see [`crate::source::parse`]).
    pub from: &'a str,
    /// Project directory inside the fetched tree. Governs the build AND
    /// the run, so it answers "where is the project in this tree" once.
    pub cwd: Option<&'a str>,
    /// Build recipe, a shell line. `None` uses cargo's release build.
    pub build: Option<&'a str>,
    /// Built artifact, relative to `cwd`. `None` when this box leaves it
    /// to the controller's run manifest.
    pub bin: Option<&'a str>,
    /// The join block's ssh credentials, for a transport that needs
    /// them. Same reuse (and same caveat) as [`DataSpec::ssh`].
    pub ssh: Option<&'a SshConfig>,
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
    /// would be surface that only ever repeats the first one.
    ///
    /// One consequence belongs in the operator's hands rather than in
    /// their debugging: a forced `command=` on that key covers subsystem
    /// requests, so a join key guardrailed with `/usr/sbin/nologin`
    /// turns sftp away and the mount never comes up while the tunnel
    /// keeps working (`ssh -N` requests no command at all). Either the
    /// key permits sftp (`internal-sftp -R`), or the source root is
    /// mounted during provisioning and `path` is declared bare — which
    /// is also the answer for a box whose mount needs different
    /// credentials entirely. Both spellings are in the guardrail recipe
    /// in docs/ddp/02-cluster-guide.md.
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

/// Where a fetched source tree lands, under the global root. Sibling of
/// the `libtorch/` and `data/` the same root already carries, per the
/// convention that anything fdl manages locally on one box lives there
/// while only paths a config names across hosts are absolute.
const SOURCE_SUBDIR: &str = "source";

/// Prepare this box. `notes` collects everything worth telling the
/// operator that is not a failure (a reused mount, a tmpfs stage, a
/// nearly-full volume); the caller prints them.
///
/// Cheap before expensive: the gate, the mount and the write proofs all
/// finish in about the time it takes to say so, while a source fetch and
/// a cold build take minutes. The source comes before libtorch for the
/// same reason at a smaller scale — a tree is megabytes and a libtorch
/// variant is gigabytes, so a broken spec should fail before the
/// download rather than after it.
pub fn prepare(spec: &PrepareSpec, notes: &mut Vec<String>) -> Result<Prepared, Fail> {
    check_gpu_stack()?;
    let data_path = resolve_data_root(&spec.data, notes)?;
    check_local_dirs(notes)?;

    let fetched = match &spec.source {
        Some(source) => Some(fetch_source(source, notes)?),
        None => None,
    };
    let libtorch = match spec.libtorch {
        Some(token) => Some(acquire_libtorch(token, notes)?),
        None => spec.active_libtorch.cloned(),
    };
    if let Some(lt) = &libtorch {
        check_arch_coverage(lt, spec.devices)?;
    }
    let (bin, args, run_id) = match (&spec.source, &fetched) {
        (Some(source), Some((tree, manifest))) => {
            let recipe = merge_manifest(source, manifest.as_ref())?;
            let built = build_source(&recipe, tree, libtorch.as_ref(), notes)?;
            (
                Some(built),
                manifest.as_ref().map(|m| m.args.clone()),
                manifest.as_ref().and_then(|m| m.run.clone()),
            )
        }
        _ => (None, None, None),
    };
    Ok(Prepared {
        data_path,
        libtorch,
        bin,
        args,
        run_id,
    })
}

/// What to build, once the controller has had its say.
///
/// The manifest wins over the box's own config, and that is the point of
/// it: everything here belongs to the RUN, and a cohort where one box
/// disagrees about the binary or its arguments is not a cohort. The local
/// values stay as the answer when nobody has published — a rig where the
/// operator drives `fdl join` by hand needs no publish at all.
fn merge_manifest<'a>(
    local: &'a SourceSpec<'a>,
    manifest: Option<&'a Manifest>,
) -> Result<Recipe<'a>, Fail> {
    let Some(m) = manifest else {
        let Some(bin) = local.bin else {
            // Nothing published, and nothing declared here either: the
            // box cannot know what to run. Transient, because the far
            // side publishing is exactly what fixes it — including the
            // window where a publish has cleared the manifest and not
            // yet written the new one.
            return Err(Fail::Transient(
                "the fetched source carries no run manifest and this box \
                 declares no artifact — publish a run on the controller \
                 (`fdl publish`), or name it locally with `--source-bin`"
                    .to_string(),
            ));
        };
        return Ok(Recipe {
            cwd: local.cwd,
            build: local.build,
            bin,
        });
    };
    Ok(Recipe {
        cwd: m.cwd.as_deref().or(local.cwd),
        build: m.build.as_deref().or(local.build),
        bin: &m.bin,
    })
}

/// The resolved build recipe: the manifest's answers where it has them,
/// the box's own where it does not.
#[derive(Debug)]
struct Recipe<'a> {
    cwd: Option<&'a str>,
    build: Option<&'a str>,
    bin: &'a str,
}

// ---------------------------------------------------------------------------
// libtorch
// ---------------------------------------------------------------------------

/// Acquire a libtorch variant and return `(variant directory, label)`.
///
/// Into the GLOBAL root, never the project one: a walk-in is consuming
/// artifacts, and the project root it happens to stand in is frequently a
/// read-only shared mount (see [`Context::global`]). Idempotent — the
/// downloader recognises a variant already installed at the pinned
/// version and returns without touching the network.
///
/// `auto` is the fleet-friendly value: it routes on the devices this box
/// actually has, so one golden image serves NVIDIA and AMD instances.
fn acquire_libtorch(token: &str, notes: &mut Vec<String>) -> Result<(PathBuf, String), Fail> {
    let variant = parse_libtorch_token(token)?;
    let ctx = Context::global();
    let id = crate::libtorch::download::run_with_context(
        crate::libtorch::download::DownloadOpts {
            variant,
            custom_path: None,
            // Write `.active` under the global root so anything else fdl
            // does on this box afterwards agrees with what trains here.
            activate: true,
            dry_run: false,
            force_linux: false,
        },
        &ctx,
    )
    // A download that fails is the network, or a mirror having a bad
    // day: worth another dial rather than stopping the box.
    .map_err(Fail::Transient)?;

    let dir = ctx.root.join("libtorch").join(&id);
    if !dir.join("lib").is_dir() {
        return Err(Fail::Permanent(format!(
            "libtorch `{id}` is not usable at {} (no lib/) — remove it and \
             let fdl fetch it again",
            dir.display(),
        )));
    }
    notes.push(format!("libtorch: {id} at {}", dir.display()));
    Ok((dir, id))
}

/// Map a `libtorch:` value onto a downloadable variant. The accepted
/// values are `fdl libtorch download`'s own flags spelled as one token,
/// so the two surfaces cannot drift into naming different things.
fn parse_libtorch_token(token: &str) -> Result<crate::libtorch::download::Variant, Fail> {
    use crate::libtorch::download::Variant;
    match token.trim() {
        "auto" => Ok(Variant::Auto),
        "cpu" => Ok(Variant::Cpu),
        "cu126" | "12.6" => Ok(Variant::Cuda126),
        "cu128" | "12.8" => Ok(Variant::Cuda128),
        "rocm7.0" | "rocm70" | "7.0" => Ok(Variant::Rocm70),
        "rocm7.1" | "rocm71" | "7.1" => Ok(Variant::Rocm71),
        other => Err(Fail::Permanent(format!(
            "unknown libtorch variant `{other}` — fdl ships `auto`, `cpu`, \
             `cu126`, `cu128`, `rocm7.0` and `rocm7.1`. `auto` picks from the \
             devices this box has, which is what lets one image serve both \
             vendors"
        ))),
    }
}

// ---------------------------------------------------------------------------
// The training source
// ---------------------------------------------------------------------------

/// Materialise the source tree on local disk and hand back its root plus
/// whatever run manifest came with it.
fn fetch_source(
    spec: &SourceSpec,
    notes: &mut Vec<String>,
) -> Result<(PathBuf, Option<Manifest>), Fail> {
    let source = crate::source::parse(spec.from)?;
    let dest = Context::global().root.join(SOURCE_SUBDIR);
    crate::source::materialize(&source, &dest, spec.ssh, notes)?;
    let manifest = Manifest::read(&dest)?;
    if let Some(m) = &manifest {
        notes.push(format!(
            "run manifest: {}bin {}{}{}{}",
            m.run
                .as_deref()
                .map(|r| format!("run {}… ", &r[..r.len().min(8)]))
                .unwrap_or_default(),
            m.bin,
            m.cwd
                .as_deref()
                .map(|c| format!(" in {c}"))
                .unwrap_or_default(),
            m.published_epoch
                .and_then(age_hint)
                .map(|age| format!(", published {age}"))
                .unwrap_or_default(),
            if m.built {
                ""
            } else {
                " — NOT built by the controller"
            },
        ));
        if let (Some(theirs), Some(ours)) = (&m.rustc, local_rustc())
            && theirs != &ours
        {
            notes.push(format!(
                "the controller built this with {theirs}, this box has \
                     {ours} — advisory only, every box compiles its own \
                     binary and a toolchain too old fails loudly at compile \
                     time",
            ));
        }
    }
    Ok((dest, manifest))
}

/// How long ago a publish happened, in the roughest terms that are still
/// useful. `None` when the clock disagrees with the manifest (a box whose
/// time has not synced yet says nothing rather than something wrong).
fn age_hint(then: u64) -> Option<String> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let secs = now.checked_sub(then)?;
    Some(match secs {
        0..=90 => "just now".to_string(),
        s if s < 5400 => format!("{}m ago", s / 60),
        s if s < 172_800 => format!("{}h ago", s / 3600),
        s => format!("{}d ago", s / 86_400),
    })
}

/// `rustc -V` here, for the manifest's advisory comparison.
fn local_rustc() -> Option<String> {
    let out = Command::new("rustc").arg("-V").output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Build the fetched tree.
fn build_source(
    recipe: &Recipe,
    tree: &Path,
    libtorch: Option<&(PathBuf, String)>,
    notes: &mut Vec<String>,
) -> Result<Built, Fail> {
    if libtorch.is_none() {
        notes.push(
            "no libtorch is active on this box and none was requested, so \
             the build gets no LIBTORCH_PATH — set `libtorch:` (`auto` \
             picks for this box) unless the recipe supplies its own"
                .to_string(),
        );
    }
    let env = crate::source::build_env(libtorch);
    crate::source::build(tree, recipe.cwd, recipe.build, recipe.bin, &env, notes).map_err(|e| {
        if e.is_permanent() {
            return e;
        }
        // A failed build while the vendor's toolkit headers are missing
        // is a PROVISIONING fault wearing a compile error: waiting
        // cannot install a package, and ROCm needs seven -dev packages
        // with no metapackage, so this is the predicted first-contact
        // failure on a golden AMD image. Classed at failure time rather
        // than as a pre-flight, deliberately — fdl passes no feature
        // flag of its own and cannot know whether the recipe needed the
        // toolkit (a cpu-feature crate builds fine without it), but a
        // build that FAILED while the toolkit is demonstrably absent is
        // the case re-dialing provably cannot fix.
        if let Some((_, variant)) = libtorch
            && let flodl_hw::VariantClass::Vendor(vendor) =
                flodl_hw::classify_variant_label(variant)
            && let Some(gap) = crate::util::requirements::toolkit_gap(vendor)
        {
            return Fail::Permanent(format!(
                "{} — and this box is missing the {vendor} toolkit \
                         headers under {} ({}), which a `--features {}` \
                         compile needs. Re-dialing cannot install a package: \
                         {}",
                e.message(),
                gap.root.display(),
                gap.headers.join(", "),
                vendor.cargo_feature(),
                gap.install,
            ));
        }
        // Still transient, with the worker's next step spelled out: this
        // box cannot fix a compile error, and it must not stop over one.
        Fail::Transient(format!(
            "{} — fix it at the source; this box picks the fix up on its \
             next dial",
            e.message(),
        ))
    })
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

/// Refuse a libtorch that ships no kernel for a card this box offers.
///
/// [`check_gpu_stack`] proves devices EXIST; this proves the resolved
/// variant can address them. Without it a Pascal-class box holding a
/// cu128-only build passes every gate, is admitted into a quorum,
/// builds successfully, and dies at its FIRST GPU op with `no kernel
/// image is available` — after the window was spent, taking the
/// cohort's formation with it. This is the arch-coherence check the
/// membership design promised, landed where the information lives: the
/// variant's `.arch` metadata and the device list are both local facts.
///
/// Scope, deliberately narrow: only devices of the variant's OWN vendor
/// are consulted (an unusable other-vendor iGPU beside working cards is
/// a trainable box — the same lesson as the GPU gate), only devices
/// this box offers (`--devices` scopes a half-covered box onto its
/// covered card), and a variant with no `.arch` metadata gates nothing
/// here — `fdl probe` flags missing metadata as its own issue, and
/// refusing to dial over it would stop working setups.
fn check_arch_coverage(libtorch: &(PathBuf, String), offered: Option<&[u8]>) -> Result<(), Fail> {
    let (dir, label) = libtorch;
    let flodl_hw::VariantClass::Vendor(vendor) = flodl_hw::classify_variant_label(label) else {
        return Ok(());
    };
    let info = crate::libtorch::detect::libtorch_info_from_dir(label.clone(), dir);
    let Some(archs) = info.archs.clone() else {
        return Ok(());
    };
    let devices: Vec<_> = flodl_hw::survey_visible()
        .devices
        .into_iter()
        .filter(|d| d.vendor == vendor)
        .filter(|d| offered.is_none_or(|ids| ids.contains(&d.index)))
        .collect();
    if devices.is_empty() {
        // The variant's vendor has nothing here — whether that is fine
        // is the training binary's question, not this gate's.
        return Ok(());
    }
    let mut details = Vec::new();
    let coverage = crate::libtorch::detect::arch_coverage(&info, &devices, &mut details);
    if coverage.iter().all(|(_, ok)| *ok) {
        return Ok(());
    }
    Err(Fail::Permanent(format!(
        "libtorch `{label}` (archs `{archs}`) ships no kernel for part of \
         what this box offers: {} The first GPU op would die with `no \
         kernel image is available` — after admission counted this host \
         into a quorum. `libtorch: auto` picks a covering variant when \
         one exists; `--devices` can scope the offer to covered cards",
        details.join(" "),
    )))
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
fn resolve_data_root(spec: &DataSpec, notes: &mut Vec<String>) -> Result<Option<PathBuf>, Fail> {
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
    std::path::absolute(path)
        .map_err(|e| Fail::Permanent(format!("cannot resolve data path `{path}`: {e}")))
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

/// Parse a `data_source:` value. Unsupported schemes error loudly rather
/// than being stubbed: a box that cannot reach its data must say so here,
/// not fail mid-epoch.
fn parse_source(spec: &str) -> Result<SshTarget, Fail> {
    match split_scheme(spec) {
        (Some("sshfs"), rest) => parse_ssh_target(rest).map_err(|why| {
            Fail::Permanent(format!(
                "invalid data_source `sshfs://{rest}` — {why}. Expected \
                 `sshfs://[user@]host[:port]/abs/path` (or the scp spelling \
                 `sshfs://[user@]host:/abs/path`)"
            ))
        }),
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
fn mount_sshfs(target: &SshTarget, mountpoint: &Path, ssh: Option<&SshConfig>) -> Result<(), Fail> {
    if !crate::util::system::has_command("sshfs") {
        return Err(Fail::Permanent(format!(
            "data_source needs sshfs, which is not installed — {} (or mount `{}` \
             during provisioning and declare a bare `data_path:`)",
            crate::util::requirements::install_hint(&["sshfs".to_string()]),
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
fn sshfs_argv(target: &SshTarget, mountpoint: &Path, ssh: Option<&SshConfig>) -> Vec<String> {
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
        // The tunnel's own options and key: same box, same trust path,
        // which that key has to actually permit (see `DataSpec::ssh`).
        if let Some(warning) =
            crate::cluster::batchmode_override_warning(&ssh.options, &target.remote)
        {
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

    if let Some(fs_type) = crate::probe::detect_fs_type(dir)
        && (fs_type == "tmpfs" || fs_type == "ramfs")
    {
        notes.push(format!(
            "{label} directory {} is on {fs_type} (RAM-backed) — \
                 staging there spends RAM, not disk",
            dir.display(),
        ));
    }
    if let Some(kib) = available_kib(dir)
        && kib < LOW_SPACE_KIB
    {
        notes.push(format!(
            "{label} directory {} has {} MiB free — smaller than any \
                 real corpus",
            dir.display(),
            kib / 1024,
        ));
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

/// Print what preparation found, under the name of the command that
/// found it. Notes are advisory by construction — every fatal condition
/// already returned a [`Fail`].
pub fn print_notes(command: &str, notes: &[String]) {
    for note in notes {
        eprintln!("{}", style::dim(&format!("fdl {command}: {note}")));
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

    /// Hardware-independent the same way the GPU-gate test is: the
    /// expectation is computed from the real box, so a GPU-less CI
    /// runner asserts the pass-through and a real rig asserts the
    /// refusal.
    #[test]
    fn a_variant_covering_none_of_the_offered_cards_is_refused() {
        let dir = std::env::temp_dir().join(format!(
            "fdl-prep-arch-{}-{}",
            std::process::id(),
            next_probe_id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // `0.0` matches no real device arch, so any NVIDIA card on this
        // box is uncovered by construction.
        std::fs::write(dir.join(".arch"), "archs=0.0\n").unwrap();
        let lt = (dir.clone(), "precompiled/cu128".to_string());
        let nvidia_present = flodl_hw::survey_visible()
            .devices
            .iter()
            .any(|d| d.vendor == flodl_hw::GpuVendor::Nvidia);
        match check_arch_coverage(&lt, None) {
            Err(err) => {
                assert!(nvidia_present, "refused with no matching device: {err:?}");
                assert!(
                    err.is_permanent(),
                    "kernels do not grow by waiting: {err:?}"
                );
                assert!(err.message().contains("no kernel image"), "got: {err:?}");
            }
            Ok(()) => assert!(
                !nvidia_present,
                "an NVIDIA card offered against archs `0.0` must be refused",
            ),
        }
        // A CPU variant, and a variant with no `.arch` metadata, gate
        // nothing — probe owns the missing-metadata complaint.
        assert!(check_arch_coverage(&(dir.clone(), "precompiled/cpu".into()), None).is_ok());
        std::fs::remove_file(dir.join(".arch")).unwrap();
        assert!(check_arch_coverage(&(dir.clone(), "precompiled/cu128".into()), None).is_ok());
        // An empty offer gates nothing either: the device scope means
        // this box deliberately offers none of that vendor's cards.
        std::fs::write(dir.join(".arch"), "archs=0.0\n").unwrap();
        assert!(check_arch_coverage(&(dir.clone(), "precompiled/cu128".into()), Some(&[])).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_sshfs_scheme_reaches_the_shared_grammar() {
        // The spellings themselves are `crate::spec`'s tests; this is the
        // wrapper's job — dispatch on the scheme, hand back a target.
        assert_eq!(
            parse_source("sshfs://flodl@exa:2222/flodl/data").unwrap(),
            SshTarget {
                remote: "flodl@exa:/flodl/data".into(),
                port: Some(2222)
            },
        );
    }

    #[test]
    fn a_published_manifest_outranks_the_boxs_own_recipe() {
        // The point of a manifest: everything in it belongs to the RUN, and
        // a cohort where one box disagrees about the binary is not a
        // cohort.
        let local = SourceSpec {
            from: "rsync://ctrl:/srv/run/tree",
            cwd: Some("stale"),
            build: Some("stale-build"),
            bin: Some("stale-bin"),
            ssh: None,
        };
        let manifest = Manifest {
            cwd: Some("ddp-bench".into()),
            build: Some("cargo build --release".into()),
            bin: "target/release/ddp-bench".into(),
            ..Manifest::default()
        };
        let recipe = merge_manifest(&local, Some(&manifest)).unwrap();
        assert_eq!(recipe.cwd, Some("ddp-bench"));
        assert_eq!(recipe.build, Some("cargo build --release"));
        assert_eq!(recipe.bin, "target/release/ddp-bench");
    }

    #[test]
    fn a_manifest_that_says_nothing_leaves_the_local_answer_standing() {
        // A hand-driven box needs no publish at all, and a manifest that
        // omits a field is not an instruction to forget the local one.
        let local = SourceSpec {
            from: "file:///mnt/rdl",
            cwd: Some("ddp-bench"),
            build: Some("./ci/node-build.sh"),
            bin: Some("target/release/x"),
            ssh: None,
        };
        let bare = Manifest {
            bin: "target/release/y".into(),
            ..Manifest::default()
        };
        let recipe = merge_manifest(&local, Some(&bare)).unwrap();
        assert_eq!(recipe.cwd, Some("ddp-bench"));
        assert_eq!(recipe.build, Some("./ci/node-build.sh"));
        assert_eq!(recipe.bin, "target/release/y");

        let recipe = merge_manifest(&local, None).unwrap();
        assert_eq!(recipe.bin, "target/release/x");
    }

    #[test]
    fn no_manifest_and_no_local_artifact_waits_rather_than_stopping() {
        // This is also the window a publish opens on purpose: it clears the
        // manifest before it touches the tree, so a box dialing mid-publish
        // must come back rather than train something unvalidated.
        let local = SourceSpec {
            from: "rsync://ctrl:/srv/run/tree",
            ..Default::default()
        };
        let err = merge_manifest(&local, None).unwrap_err();
        assert!(!err.is_permanent(), "the fix is a publish away: {err:?}");
        assert!(err.message().contains("fdl publish"), "got: {err:?}");
    }

    #[test]
    fn a_failed_build_is_classed_by_whether_the_toolkit_could_explain_it() {
        let dir = std::env::temp_dir().join(format!(
            "fdl-prep-toolkit-{}-{}",
            std::process::id(),
            next_probe_id(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fail = |variant: &str| {
            let libtorch = (dir.clone(), variant.to_string());
            build_source(
                &Recipe {
                    cwd: None,
                    build: Some("exit 3"),
                    bin: "x",
                },
                &dir,
                Some(&libtorch),
                &mut Vec::new(),
            )
            .unwrap_err()
        };
        // A GPU variant: the verdict follows whether this box actually
        // has the toolkit (the check probes the real filesystem, so the
        // expectation is computed, not assumed — a ROCm rig running this
        // suite has the headers and must stay on the transient side).
        let err = fail("precompiled/rocm70");
        match crate::util::requirements::toolkit_gap(flodl_hw::GpuVendor::Amd) {
            Some(gap) => {
                assert!(
                    err.is_permanent(),
                    "waiting cannot install a package: {err:?}"
                );
                assert!(
                    err.message().contains(&gap.install),
                    "the fix must be named: {err:?}"
                );
            }
            None => assert!(
                !err.is_permanent(),
                "toolkit present, so a compile error stays a push away: {err:?}"
            ),
        }
        // A CPU variant wants no toolkit, so the failure stays transient
        // whatever this box has installed.
        let err = fail("precompiled/cpu");
        assert!(!err.is_permanent(), "got: {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_source_spec_rejects_every_broken_shape_permanently() {
        for spec in [
            "/flodl/data",             // no scheme: belongs in data_path
            "smb://server/share",      // scheme we do not ship
            "sshfs://exa",             // no path
            "sshfs://exa:banana/data", // port that is not a number
            "sshfs://:/flodl/data",    // empty host
            "sshfs://exa:/",           // root is not a source root
        ] {
            let err = parse_source(spec).unwrap_err();
            assert!(err.is_permanent(), "{spec} should be permanent: {err:?}");
            // Whatever the shape, the message names the field the
            // operator has to go and fix.
            assert!(err.message().contains("data_source"), "{spec}: {err:?}");
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
        let user_pos = argv
            .iter()
            .position(|a| a == "ServerAliveInterval=5")
            .unwrap();
        let default_pos = argv
            .iter()
            .position(|a| a == "ServerAliveInterval=15")
            .unwrap();
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
        assert_eq!(
            got, None,
            "a run that never mentions data must ship nothing"
        );
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
        let spec = DataSpec {
            path: Some(&name),
            ..Default::default()
        };
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
        let spec = DataSpec {
            path: Some(&path),
            ..Default::default()
        };
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
        let spec = DataSpec {
            path: Some(&missing),
            ..Default::default()
        };
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
        // `mut` is used only by the cfg(unix) arm below; on Windows
        // nothing writes it, and an unused_mut warning there is noise
        // rather than a finding.
        #[allow(unused_mut)]
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            perms.set_mode(0o555);
        }
        std::fs::set_permissions(&dir, perms).unwrap();
        assert!(verify_source_root(&dir).is_ok());
        #[allow(unused_mut)]
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
        assert!(
            leftovers.is_empty(),
            "probe file left behind: {leftovers:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_directory_we_must_not_create_is_permanent() {
        let dir = std::env::temp_dir().join("fdl-prep-absent-stage-dir");
        let err = check_writable("disk stage", &dir, false, &mut Vec::new()).unwrap_err();
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
