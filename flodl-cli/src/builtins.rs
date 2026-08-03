//! Registry of built-in commands. Single source of truth for dispatch,
//! help listing, collision detection, and shell completion.
//!
//! Each leaf sub-command owns a `#[derive(FdlArgs)]` struct that carries
//! the canonical flag set. `BuiltinSpec::schema_fn` returns the
//! `Schema` derived from that struct, so completion rules
//! (`--cuda <TAB>` → `12.6 12.8`, etc.) flow through the same pipeline
//! as project commands rather than a hand-mirrored flag table.

use crate::args::FdlArgsTrait;
use crate::config::Schema;

// ---------------------------------------------------------------------------
// FdlArgs structs (one per leaf sub-command)
//
// These dogfood the derive macro across flodl-cli itself. Each is parsed
// with `parse_or_schema_from(&argv)` from a sliced argv tail; the derive
// handles argv, `--help`, and `--fdl-schema` uniformly.
// ---------------------------------------------------------------------------

/// Interactive guided setup wizard.
#[derive(crate::FdlArgs, Debug)]
pub struct SetupArgs {
    /// Skip all prompts and use auto-detected defaults.
    #[option(short = 'y')]
    pub non_interactive: bool,
    /// Re-download or rebuild even if libtorch exists.
    #[option]
    pub force: bool,
}

/// System and GPU diagnostics.
#[derive(crate::FdlArgs, Debug)]
pub struct DiagnoseArgs {
    /// Emit machine-readable JSON.
    #[option]
    pub json: bool,
}

/// Cluster readiness probe.
///
/// Default (single-host): probes the local box for GPU + libtorch arch
/// match + shared-data path + NCCL availability. Cluster context
/// (`fdl @cluster probe` / `FDL_ENV=cluster`): probes every host in
/// `fdl.cluster.yml` via SSH and aggregates the report.
///
/// Exit code: 0 when every checked component is green; 1 when any
/// issue was surfaced. `fdl deploy` and CI consume the `--json` shape.
#[derive(crate::FdlArgs, Debug)]
pub struct ProbeArgs {
    /// Emit machine-readable JSON.
    #[option]
    pub json: bool,
    /// Skip the shared-data-mount visibility check. Useful for
    /// single-host setups without a shared filesystem configured.
    #[option]
    pub skip_mount: bool,
    /// Override the shared-data path (default: cluster.yml's
    /// per-host `data_path:`, or the convention default
    /// `/flodl/data` when unset).
    #[option]
    pub data_path: Option<std::path::PathBuf>,
    /// Override the libtorch directory. Default walks up from cwd
    /// for `libtorch/.active`. Use this when the libtorch install
    /// lives outside the project tree (e.g. a separate virtiofs
    /// share mounted at a known path on a worker node).
    #[option]
    pub libtorch_path: Option<std::path::PathBuf>,
    /// Treat NCCL as provided by a Docker image (compose service name
    /// from `fdl.yml`, e.g. `cuda`). Suppresses host-level NCCL
    /// discovery and reports "via Docker image `<svc>`" instead. In
    /// cluster mode, this is auto-derived from each host's `docker:`
    /// field in `fdl.cluster.yml`.
    #[option]
    pub docker: Option<String>,
}

/// Live cluster run status.
///
/// Fetches the controller's `state.json` (membership + lifecycle
/// phase, served on the training port itself) and pretty-prints it.
/// Live for the whole run, join window included: shows who has joined
/// while the world is still forming.
///
/// Exit code: 0 when the state was fetched; 1 when no endpoint
/// answered (usually: no run is up).
#[derive(crate::FdlArgs, Debug)]
pub struct StatusArgs {
    /// Emit the raw state.json body instead of the human summary.
    #[option]
    pub json: bool,
    /// Controller address to query, `host[:port]` (default port 1337).
    /// Overrides the active env's `cluster.controller`. This is all a
    /// self-deployed worker's operator needs to watch a run.
    #[option]
    pub addr: Option<String>,
}

/// Fire the operator start switch of a staging cluster run.
///
/// A join window opened with `controller.join.start: manual` (or
/// `hybrid`) holds the roster once quorum is met instead of forming on
/// the clock — inspect it with `fdl status`, then fire the topology
/// freeze with this command. Refusals name their reason (auto mode,
/// quorum not met, window already closed).
///
/// Trust mirrors join admission: fired from the controller host (or
/// through the sshd tunnel) no credential is needed; from anywhere
/// else pass `--token` (the run's `controller.join.token`).
///
/// Exit code: 0 when the start was armed; 1 otherwise.
#[derive(crate::FdlArgs, Debug)]
pub struct StartArgs {
    /// Controller address, `host[:port]` (default port 1337).
    /// Overrides the active env's `cluster.controller`.
    #[option]
    pub addr: Option<String>,
    /// Session credential for a non-loopback fire (the run's
    /// `controller.join.token`).
    #[option]
    pub token: Option<String>,
}

/// Join a cluster run's window as a self-deployed worker.
///
/// Dials the controller's join channel, offers this box's GPUs, and
/// runs the training binary (`--bin`) in agent role: the binary joins,
/// then spawns and supervises this host's relay and rank children
/// itself. Every flag defaults from the `join:` block of fdl.yml when
/// present; flags win. Arguments after a standalone `--` go to the
/// training binary verbatim (they must match the run — rank children
/// re-enter the binary with them).
///
/// Trust, mirroring join admission: `--token` presents the run's
/// pre-shared credential; `--ssh` reaches a loopback-bound controller
/// through its guardrailed sshd (reachability = authentication); with
/// neither, the controller must run open admission.
///
/// Before dialing, the box is prepared: the GPU stack is checked, the
/// dataset source root is put where the ranks will look for it
/// (`--data-source` mounts it read-only when it is not already there),
/// the directories the data plane writes are proven writable, and
/// anything this box does not have yet is acquired — a libtorch variant
/// (`--libtorch`) and the training binary itself (`--source`, fetched to
/// local disk and built here, which is what makes its ABI match the
/// libtorch it holds). Preparation re-runs per attempt, so a `--persist`
/// box picks up a changed source on its next re-dial.
///
/// Exit code: the agent's exit code (0 = this host finished cleanly);
/// 2 for a failure retrying cannot fix (no GPU, a spec that does not
/// parse, a missing binary or toolchain), which `--persist` does not
/// re-dial; 1 for a transient one, which it does — the systemd /
/// golden-image mode. Source that does not compile counts as transient
/// on purpose: the fix is a push away at the source, and a box that
/// stopped permanently over a typo would be powered off by the systemd
/// recipe.
#[derive(crate::FdlArgs, Debug)]
pub struct JoinArgs {
    /// Controller mux address, `host[:port]` (default port 1337).
    /// With `--ssh`, the address as seen FROM the ssh host — the
    /// default `127.0.0.1:1337` is the sshd-on-the-controller-box
    /// convention.
    #[arg]
    pub controller: Option<String>,
    /// SSH tunnel hop, `[user@]host[:port]`: brings up a local `-L`
    /// forward of the controller port and dials through it.
    #[option]
    pub ssh: Option<String>,
    /// Identity file for the tunnel (`ssh -i`).
    #[option]
    pub identity: Option<String>,
    /// Pre-shared session credential (the run's
    /// `controller.join.token`).
    #[option]
    pub token: Option<String>,
    /// Training binary to run in agent role, as a path on this box.
    /// Mutually exclusive with `--source`.
    #[option]
    pub bin: Option<String>,
    /// Build the training binary here instead: a source spec, one of
    /// `file:///abs/path`, `rsync://[user@]host[:port]:/abs/path` or
    /// `git+https://host/owner/repo#<tag|branch|sha>`. Fetched to local
    /// disk, then built against this box's libtorch.
    #[option]
    pub source: Option<String>,
    /// Project directory inside the fetched source tree (default: its
    /// root). Governs the build and the run both.
    #[option]
    pub source_cwd: Option<String>,
    /// Build recipe for the fetched source, a shell line (default:
    /// `cargo build --release`). Gets `LIBTORCH_PATH`,
    /// `FDL_GPU_FEATURE` and `LD_LIBRARY_PATH`.
    #[option]
    pub source_build: Option<String>,
    /// Built artifact, relative to the project directory, e.g.
    /// `target/release/train`.
    #[option]
    pub source_bin: Option<String>,
    /// libtorch variant to acquire into `~/.flodl/libtorch/` before
    /// building or running: `auto`, `cpu`, `cu126`, `cu128`, `rocm7.0`.
    /// `auto` routes on this box's own devices, so one image serves
    /// both vendors. Default: whatever is already active here.
    #[option]
    pub libtorch: Option<String>,
    /// Logical host name in the roster (default: this machine's
    /// hostname).
    #[option]
    pub host: Option<String>,
    /// CUDA device ids to offer, comma-separated (default: all GPUs
    /// on this host).
    #[option]
    pub devices: Option<String>,
    /// Keep re-dialing across runs with backoff instead of exiting
    /// when the agent does.
    #[option]
    pub persist: bool,
    /// Dataset source root on this box, shipped to this host's ranks
    /// (with `--data-source`, the mountpoint; default `/flodl/data`).
    #[option]
    pub data_path: Option<String>,
    /// Transport that establishes the source root when it is not
    /// already mounted, e.g. `sshfs://user@ctrl:/flodl/data`.
    #[option]
    pub data_source: Option<String>,
}

/// Publish a training run for a fleet to pull.
///
/// The controller side of compiling on the node: resolves a source spec
/// into a served directory, builds it once as a gate, and writes the run
/// manifest workers read. Chaining runs on a standing fleet is then one
/// command — publish again and every box picks the new run up on its next
/// dial, with nothing to edit on any worker.
///
/// Arguments after a standalone `--` are the training binary's own, and
/// they go in the manifest rather than into any worker's config: they must
/// match the run, because rank children re-enter the binary with them, so
/// a fleet carrying its own copy would train the next run with the
/// previous one's hyperparameters.
///
/// The build proves the tree for THIS box's libtorch variant, and one
/// build is all it is: every worker still compiles its own, since a
/// controller producing binaries for N variants is a build matrix. A gate
/// needs no GPU libtorch — `fdl libtorch download --cpu` is enough —
/// because it is catching user-code errors, not shipping an artifact.
///
/// Exit code: 0 when the run is published; 1 otherwise, and a failed gate
/// publishes nothing, so the fleet keeps running whatever it had.
#[derive(crate::FdlArgs, Debug)]
pub struct PublishArgs {
    /// Source to publish: `file:///abs/path`,
    /// `rsync://[user@]host[:port]:/abs/path`, or
    /// `git+https://host/owner/repo#<tag|branch|sha>`.
    #[arg]
    pub source: Option<String>,
    /// Built artifact, relative to the project directory — what workers
    /// run. Required: a workspace member's build lands in the WORKSPACE
    /// `target/`, so no rule fdl invented would be right for everyone.
    #[option]
    pub bin: Option<String>,
    /// Project directory inside the tree (default: its root). Governs
    /// the build and the run both.
    #[option]
    pub cwd: Option<String>,
    /// Build recipe, a shell line (default: `cargo build --release`).
    /// Gets `LIBTORCH_PATH`, `FDL_GPU_FEATURE` and `LD_LIBRARY_PATH`.
    #[option]
    pub build: Option<String>,
    /// Served directory (default: `~/.flodl/run`). The tree lands in
    /// `<dir>/tree`, which is what a worker's source spec points at.
    #[option]
    pub to: Option<String>,
    /// Skip the build gate. Publishes source nothing has compiled, so
    /// the first worker to fetch it is where a broken build surfaces.
    #[option]
    pub no_build: bool,
    /// Identity file for a source pulled over ssh (`rsync -e ssh -i`).
    #[option]
    pub identity: Option<String>,
}

impl PublishArgs {
    /// Credentials for a source pulled over ssh. The spec itself carries
    /// user, host and port, so only the key can be missing.
    pub fn ssh_config(&self) -> Option<crate::config::SshConfig> {
        self.identity.as_ref().map(|id| crate::config::SshConfig {
            identity_file: Some(id.clone()),
            ..Default::default()
        })
    }
}

/// Generate flodl API reference.
#[derive(crate::FdlArgs, Debug)]
pub struct ApiRefArgs {
    /// Emit machine-readable JSON.
    #[option]
    pub json: bool,
    /// Explicit flodl source path (defaults to detected project root).
    #[option]
    pub path: Option<String>,
}

/// Scaffold a new floDl project.
///
/// Three modes, mutually exclusive:
///   default (no flag) — Docker with host-mounted libtorch (recommended)
///   --docker          — Docker with libtorch baked into the image
///   --native          — no Docker, host-provided libtorch + cargo
#[derive(crate::FdlArgs, Debug)]
pub struct InitArgs {
    /// New project directory name.
    #[arg]
    pub name: Option<String>,
    /// Generate a Docker scaffold with libtorch baked into the image.
    #[option]
    pub docker: bool,
    /// Generate a native scaffold (no Docker; libtorch provided on the host).
    #[option]
    pub native: bool,
    /// Also scaffold the flodl-hf HuggingFace playground (skips the prompt).
    #[option]
    pub with_hf: bool,
}

/// Add a flodl ecosystem crate to the current flodl project.
///
/// Currently supports `flodl-hf` (alias: `hf`). Two modes (combinable):
///
/// - `--playground`: drops a standalone cargo crate under `./flodl-hf/`
///   with pinned deps and a one-file AutoModel example, plus a
///   `flodl-hf:` entry in the root `fdl.yml` so `fdl flodl-hf <cmd>`
///   routes into it. Try-it-out path; doesn't touch `Cargo.toml`.
/// - `--install`: appends `flodl-hf = "=X.Y.Z"` (default features) to
///   the root `Cargo.toml` `[dependencies]`. Wires the crate into the
///   user's own code; doesn't create a subdir.
///
/// With neither flag, an interactive prompt asks. Non-tty stdin errors
/// loudly rather than silently picking a default.
#[derive(crate::FdlArgs, Debug)]
pub struct AddArgs {
    /// Target to scaffold (currently: `flodl-hf` or the alias `hf`).
    #[arg]
    pub target: Option<String>,
    /// Drop a sandbox playground under `./flodl-hf/`.
    #[option]
    pub playground: bool,
    /// Add as a dependency in the root `Cargo.toml`.
    #[option]
    pub install: bool,
}

/// Install or update fdl globally (~/.local/bin/fdl).
#[derive(crate::FdlArgs, Debug)]
pub struct InstallArgs {
    /// Check for updates without installing.
    #[option]
    pub check: bool,
    /// Symlink to the current binary (tracks local builds).
    #[option]
    pub dev: bool,
}

/// List installed libtorch variants.
#[derive(crate::FdlArgs, Debug)]
pub struct LibtorchListArgs {
    /// Emit machine-readable JSON.
    #[option]
    pub json: bool,
}

/// Activate a libtorch variant.
#[derive(crate::FdlArgs, Debug)]
pub struct LibtorchActivateArgs {
    /// Variant to activate (as shown by `fdl libtorch list`).
    #[arg]
    pub variant: Option<String>,
}

/// Remove a libtorch variant.
#[derive(crate::FdlArgs, Debug)]
pub struct LibtorchRemoveArgs {
    /// Variant to remove (as shown by `fdl libtorch list`).
    #[arg]
    pub variant: Option<String>,
}

/// Download a pre-built libtorch variant.
#[derive(crate::FdlArgs, Debug)]
pub struct LibtorchDownloadArgs {
    /// Force the CPU variant.
    #[option]
    pub cpu: bool,
    /// Pick a specific CUDA version (instead of auto-detect).
    #[option(choices = &["12.6", "12.8"])]
    pub cuda: Option<String>,
    /// Pick an AMD ROCm build instead of CUDA.
    #[option(choices = &["7.0"])]
    pub rocm: Option<String>,
    /// Install libtorch to this directory (default: project libtorch/).
    #[option]
    pub path: Option<String>,
    /// Do not activate after download.
    #[option]
    pub no_activate: bool,
    /// Show what would happen without downloading.
    #[option]
    pub dry_run: bool,
}

/// Build libtorch from source.
#[derive(crate::FdlArgs, Debug)]
pub struct LibtorchBuildArgs {
    /// Override CUDA architectures (semicolon-separated, e.g. "6.1;12.0").
    #[option]
    pub archs: Option<String>,
    /// Parallel compilation jobs.
    #[option(default = "6")]
    pub jobs: usize,
    /// Force Docker build (isolated, reproducible).
    #[option]
    pub docker: bool,
    /// Force native build (faster, requires host toolchain).
    #[option]
    pub native: bool,
    /// Show what would happen without building.
    #[option]
    pub dry_run: bool,
}

/// Build NCCL from NVIDIA source for a heterogeneous-arch rig.
#[derive(crate::FdlArgs, Debug)]
pub struct NcclBuildArgs {
    /// NCCL git tag to build (e.g. "v2.27.5-1"). Default: infer from the
    /// active libtorch's bundled NCCL version string (the version we must
    /// match for cross-rank handshake).
    #[option]
    pub tag: Option<String>,
    /// Override CUDA architectures (semicolon-separated, e.g. "6.1;12.0").
    /// Default: auto-detect from local GPUs.
    #[option]
    pub archs: Option<String>,
    /// Parallel compilation jobs.
    #[option(default = "6")]
    pub jobs: usize,
    /// Show what would happen without building.
    #[option]
    pub dry_run: bool,
}

/// Install AI coding assistant skills.
#[derive(crate::FdlArgs, Debug)]
pub struct SkillInstallArgs {
    /// Target tool (defaults to auto-detect).
    #[option]
    pub tool: Option<String>,
    /// Specific skill name (defaults to all detected skills).
    #[option]
    pub skill: Option<String>,
}

/// List cached `--fdl-schema` outputs.
#[derive(crate::FdlArgs, Debug)]
pub struct SchemaListArgs {
    /// Emit machine-readable JSON.
    #[option]
    pub json: bool,
}

/// Clear cached schemas. No command name clears all.
#[derive(crate::FdlArgs, Debug)]
pub struct SchemaClearArgs {
    /// Command name to clear (defaults to all).
    #[arg]
    pub cmd: Option<String>,
}

/// Re-probe each entry and rewrite the cache.
#[derive(crate::FdlArgs, Debug)]
pub struct SchemaRefreshArgs {
    /// Command name to refresh (defaults to all).
    #[arg]
    pub cmd: Option<String>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// One built-in command (or sub-command) slot.
pub struct BuiltinSpec {
    /// Path from the top-level command name. `["install"]`,
    /// `["libtorch", "download"]`.
    pub path: &'static [&'static str],
    /// One-line description for `fdl -h` listing. `None` = hidden
    /// (reserved for collision detection but not shown in help).
    pub description: Option<&'static str>,
    /// Constructor for the command's schema. `None` for parent commands
    /// that only group sub-commands (e.g. `libtorch` itself has no args)
    /// or for leaves whose argv is parsed by hand (`config show`,
    /// `completions`, `autocomplete`).
    pub schema_fn: Option<fn() -> Schema>,
}

/// Ordered registry of every built-in. Order drives `fdl -h` and the
/// top-level completion word list, so it mirrors today's `BUILTINS`
/// const in `main.rs`.
pub fn registry() -> &'static [BuiltinSpec] {
    static REG: &[BuiltinSpec] = &[
        BuiltinSpec {
            path: &["setup"],
            description: Some("Interactive guided setup"),
            schema_fn: Some(SetupArgs::schema),
        },
        BuiltinSpec {
            path: &["libtorch"],
            description: Some("Manage libtorch installations"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["libtorch", "download"],
            description: Some("Download pre-built libtorch"),
            schema_fn: Some(LibtorchDownloadArgs::schema),
        },
        BuiltinSpec {
            path: &["libtorch", "build"],
            description: Some("Build libtorch from source"),
            schema_fn: Some(LibtorchBuildArgs::schema),
        },
        BuiltinSpec {
            path: &["libtorch", "list"],
            description: Some("Show installed variants"),
            schema_fn: Some(LibtorchListArgs::schema),
        },
        BuiltinSpec {
            path: &["libtorch", "activate"],
            description: Some("Set active variant"),
            schema_fn: Some(LibtorchActivateArgs::schema),
        },
        BuiltinSpec {
            path: &["libtorch", "remove"],
            description: Some("Remove a variant"),
            schema_fn: Some(LibtorchRemoveArgs::schema),
        },
        BuiltinSpec {
            path: &["libtorch", "info"],
            description: Some("Show active variant details"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["nccl"],
            description: Some("Build NCCL from source (heterogeneous-arch bridge)"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["nccl", "build"],
            description: Some("Compile libnccl for the local GPU arch"),
            schema_fn: Some(NcclBuildArgs::schema),
        },
        BuiltinSpec {
            path: &["init"],
            description: Some("Scaffold a new floDl project"),
            schema_fn: Some(InitArgs::schema),
        },
        BuiltinSpec {
            path: &["add"],
            description: Some("Add a flodl ecosystem crate (currently: flodl-hf)"),
            schema_fn: Some(AddArgs::schema),
        },
        BuiltinSpec {
            path: &["diagnose"],
            description: Some("System and GPU diagnostics"),
            schema_fn: Some(DiagnoseArgs::schema),
        },
        BuiltinSpec {
            path: &["probe"],
            description: Some(
                "Cluster readiness probe (GPU + libtorch + data mount)",
            ),
            schema_fn: Some(ProbeArgs::schema),
        },
        BuiltinSpec {
            path: &["status"],
            description: Some(
                "Live cluster run status (membership, lifecycle phase)",
            ),
            schema_fn: Some(StatusArgs::schema),
        },
        BuiltinSpec {
            path: &["start"],
            description: Some(
                "Fire the start switch of a staging cluster run",
            ),
            schema_fn: Some(StartArgs::schema),
        },
        BuiltinSpec {
            path: &["publish"],
            description: Some("Publish a training run for a fleet to pull"),
            schema_fn: Some(PublishArgs::schema),
        },
        BuiltinSpec {
            path: &["join"],
            description: Some(
                "Join a cluster run's window as a self-deployed worker",
            ),
            schema_fn: Some(JoinArgs::schema),
        },
        BuiltinSpec {
            path: &["install"],
            description: Some("Install or update fdl globally"),
            schema_fn: Some(InstallArgs::schema),
        },
        BuiltinSpec {
            path: &["skill"],
            description: Some("Manage AI coding assistant skills"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["skill", "install"],
            description: Some("Install skills for the detected tool"),
            schema_fn: Some(SkillInstallArgs::schema),
        },
        BuiltinSpec {
            path: &["skill", "list"],
            description: Some("Show available skills"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["api-ref"],
            description: Some("Generate flodl API reference"),
            schema_fn: Some(ApiRefArgs::schema),
        },
        BuiltinSpec {
            path: &["config"],
            description: Some("Inspect resolved project configuration"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["config", "show"],
            description: Some("Print the resolved merged config"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["schema"],
            description: Some("Inspect, clear, or refresh cached --fdl-schema outputs"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["schema", "list"],
            description: Some("Show every cached schema with status"),
            schema_fn: Some(SchemaListArgs::schema),
        },
        BuiltinSpec {
            path: &["schema", "clear"],
            description: Some("Delete cached schema(s)"),
            schema_fn: Some(SchemaClearArgs::schema),
        },
        BuiltinSpec {
            path: &["schema", "refresh"],
            description: Some("Re-probe each entry and rewrite the cache"),
            schema_fn: Some(SchemaRefreshArgs::schema),
        },
        BuiltinSpec {
            path: &["completions"],
            description: Some("Emit shell completion script (bash|zsh|fish)"),
            schema_fn: None,
        },
        BuiltinSpec {
            path: &["autocomplete"],
            description: Some("Install completions into the detected shell"),
            schema_fn: None,
        },
        // Hidden: `version` is covered by `-V` / `--version` but still
        // reserved as a top-level built-in name.
        BuiltinSpec {
            path: &["version"],
            description: None,
            schema_fn: None,
        },
    ];
    REG
}

/// True when `name` is a reserved top-level built-in (visible or hidden).
pub fn is_builtin_name(name: &str) -> bool {
    registry()
        .iter()
        .any(|s| s.path.len() == 1 && s.path[0] == name)
}

/// Visible top-level built-ins as `(name, description)` pairs, in
/// registry order. Feeds `run::print_project_help` and the fallback
/// `print_usage`.
pub fn visible_top_level() -> Vec<(&'static str, &'static str)> {
    registry()
        .iter()
        .filter(|s| s.path.len() == 1)
        .filter_map(|s| s.description.map(|d| (s.path[0], d)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;



    #[test]
    fn registry_has_no_duplicate_paths() {
        let mut seen = HashSet::new();
        for s in registry() {
            let key = s.path.join(" ");
            assert!(
                seen.insert(key.clone()),
                "duplicate registry path: {key}"
            );
        }
    }

    #[test]
    fn hidden_entries_have_no_description() {
        for s in registry() {
            if s.path == ["version"] {
                assert!(s.description.is_none(),
                    "`version` is hidden but carries a description");
            }
        }
    }

    #[test]
    fn every_parent_has_at_least_one_child() {
        let parents: HashSet<&str> = registry()
            .iter()
            .filter(|s| s.path.len() == 1 && s.schema_fn.is_none()
                && s.description.is_some())
            .map(|s| s.path[0])
            .collect();

        // `completions`, `autocomplete` are leaves with no schema — exclude
        // them by checking that parents have at least one 2-path child.
        for parent in &parents {
            let has_child = registry().iter().any(|s| s.path.len() == 2 && s.path[0] == *parent);
            if !has_child {
                // `completions` / `autocomplete` / `version` end up here by
                // virtue of having no children; they are leaf built-ins.
                continue;
            }
            assert!(has_child, "parent `{parent}` has no child entries");
        }
    }

    #[test]
    fn top_level_dispatched_by_main_is_in_registry() {
        // Compile-time guard: every match arm target in main.rs is listed
        // here. Keeping the list local (rather than introspecting main.rs)
        // documents the coupling explicitly.
        let dispatched = [
            "setup", "libtorch", "nccl", "diagnose", "probe", "status",
            "start", "publish", "join", "api-ref", "init", "add", "install",
            "skill", "schema", "completions", "autocomplete", "config",
            "version",
        ];
        for name in &dispatched {
            assert!(
                is_builtin_name(name),
                "`{name}` dispatched by main.rs but missing from registry"
            );
        }
    }

    #[test]
    fn visible_top_level_matches_help_ordering() {
        let top = visible_top_level();
        let names: Vec<&str> = top.iter().map(|(n, _)| *n).collect();
        // Lock in the order that `fdl -h` depends on.
        assert_eq!(
            names,
            vec![
                "setup", "libtorch", "nccl", "init", "add", "diagnose",
                "probe", "status", "start", "publish", "join", "install",
                "skill", "api-ref", "config", "schema", "completions",
                "autocomplete",
            ]
        );
    }

    #[test]
    fn libtorch_download_schema_carries_cuda_choices() {
        let spec = registry()
            .iter()
            .find(|s| s.path == ["libtorch", "download"])
            .expect("libtorch download entry present");
        let schema = (spec.schema_fn.expect("download has schema"))();
        let cuda = schema
            .options
            .get("cuda")
            .expect("`--cuda` option declared");
        let choices = cuda.choices.as_ref().expect("--cuda has choices");
        let values: Vec<String> = choices
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        assert_eq!(values, vec!["12.6".to_string(), "12.8".into()]);
    }
}
