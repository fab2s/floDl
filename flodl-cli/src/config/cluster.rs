//! DDP + cluster configuration types: DdpConfig, SpeedHint,
//! TrainingConfig, OutputConfig, ClusterConfig, ClusterController,
//! LocalDevices, ClusterWorker + the `ClusterConfig` impl block.


use serde::{Deserialize, Serialize};
use serde_json::Value;

/// DDP configuration. Maps 1:1 to flodl DdpConfig / DdpRunConfig.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DdpConfig {
    pub mode: Option<String>,
    pub policy: Option<String>,
    pub backend: Option<String>,
    /// "auto" or integer.
    pub anchor: Option<serde_json::Value>,
    pub max_anchor: Option<u32>,
    pub overhead_target: Option<f64>,
    pub divergence_threshold: Option<f64>,
    /// null (unlimited) or integer.
    pub max_batch_diff: Option<serde_json::Value>,
    pub speed_hint: Option<SpeedHint>,
    pub partition_ratios: Option<Vec<f64>>,
    /// "auto" or bool.
    pub progressive: Option<serde_json::Value>,
    pub max_grad_norm: Option<f64>,
    pub lr_scale_ratio: Option<f64>,
    pub snapshot_timeout: Option<u32>,
    pub checkpoint_every: Option<u32>,
    pub timeline: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeedHint {
    pub slow_rank: usize,
    pub ratio: f64,
}

/// Training scalars.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub epochs: Option<u32>,
    pub batch_size: Option<u32>,
    pub batches_per_epoch: Option<u32>,
    pub lr: Option<f64>,
    pub seed: Option<u64>,
}

/// Output settings.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    pub dir: Option<String>,
    pub timeline: Option<bool>,
    pub monitor: Option<u16>,
}

// ── Cluster topology (multi-host DDP) ───────────────────────────────────

/// Default port for the controller's rendezvous bind. Overridable via
/// `cluster.controller.port:` in fdl.cluster.yml.
pub const DEFAULT_CONTROLLER_PORT: u16 = 1337;

fn default_controller_port() -> u16 {
    DEFAULT_CONTROLLER_PORT
}

/// Cluster topology, parsed from the `cluster:` block at the project
/// root. Two role-separated sub-blocks:
///
/// - `controller`: the orchestrator host fdl-cli runs on. Holds the
///   rendezvous bind point (`host`/`port`) plus the controller-local
///   fields needed for pre-flight build and the ClusterController TCP
///   listener. The controller is NOT a NCCL rank.
/// - `workers`: every rank-carrying host. Each entry binds one or more
///   global ranks and their CUDA device indices.
///
/// This is the controller-side full topology; per-worker slim
/// envelopes are derived from it and shipped to each node, where the
/// library reads them via `flodl::distributed::LocalCluster::from_env`.
/// Launcher-only fields (`ssh:`) are not propagated to the envelope.
///
/// The library re-validates after reading; this validation runs earlier
/// so errors surface before `fdl-cli` opens any SSH connection.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterConfig {
    pub controller: ClusterController,
    pub workers: Vec<ClusterWorker>,
    /// Cluster-scope env vars exported into every rank child on every
    /// worker. Mapping `NAME: VALUE` (string→string). Use for tuning
    /// the launcher itself shouldn't hardcode — e.g. on the Pascal-
    /// under-VFIO rig, set `NCCL_P2P_DISABLE: "1"` +
    /// `NCCL_SHM_DISABLE: "1"` so NCCL falls back to socket transport
    /// (the direct-IPC transports fail under VFIO on consumer Pascal).
    /// Worker-scope [`ClusterWorker::env`] takes precedence for matching
    /// keys.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub env: std::collections::BTreeMap<String, String>,
    /// Cluster-scope default for the integrated-GPU host-RAM share: the
    /// fraction of `MemTotal` an APU host's GPU aperture claims (the
    /// library's `gpu_ram_share` knob). Discrete-GPU hosts ignore it,
    /// which is what makes a fleet-wide default legal on a mixed fleet;
    /// the case it exists for is a farm of identical APU boxes, where
    /// one line here covers walk-ins the controller never enumerates.
    /// Per-worker [`ClusterWorker::gpu_ram_share`] overrides it, and a
    /// walk-in's own `join.gpu_ram_share:` overrides both (host truth
    /// wins). Non-negative fraction of `MemTotal` (above 1.0 is legal
    /// where the platform under-states the aperture); unset ships
    /// nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_ram_share: Option<f64>,
}

/// Controller-side cluster config. Holds the rendezvous bind point and
/// pre-flight build context. The controller is the orchestrator host
/// fdl-cli runs on; it is NOT a NCCL rank itself.
///
/// Rejected fields (via `deny_unknown_fields`): `ranks`,
/// `local_devices`, `ssh*` (controller is local to fdl-cli),
/// per-controller `env` (use cluster-scope `env:` instead) — and any
/// mistyped key.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterController {
    /// Bind address. Used as the rendezvous endpoint workers dial.
    /// What was `cluster.controller.host` in the pre-Refactor-2 schema.
    pub host: String,
    /// Bind port. What was `cluster.controller.port` in the pre-Refactor-2
    /// schema. Defaults to [`DEFAULT_CONTROLLER_PORT`] when omitted.
    #[serde(default = "default_controller_port")]
    pub port: u16,
    /// Controller's view of the shared project root. Used by the pre-
    /// flight build phase to drive cargo invocations from the
    /// controller's filesystem perspective; the remote-side path (each
    /// worker's [`ClusterWorker::path`]) is used at runtime via SSH.
    ///
    /// On homogeneous-mount rigs the controller's view equals every
    /// worker's view; on heterogeneous rigs (different mount points
    /// per host), the controller's value diverges (e.g. the controller
    /// sees `/opt/flodl` while a VM worker sees `/mnt/flodl`).
    pub path: String,
    /// Docker compose service the controller runs the pre-flight build
    /// inside (e.g. `cuda`, `dev`). Optional; affects build context
    /// only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docker: Option<String>,
    /// libtorch variant subpath under `<path>/libtorch/` for the
    /// pre-flight build (e.g. `precompiled/cu128`,
    /// `builds/sm61-sm120`). When unset, fdl-cli falls back to the
    /// controller's project-root `.active` file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// Dataset source root visible to the controller: training data,
    /// model checkpoints and per-rank logs. Usually a filesystem
    /// reachable at the same logical path on every node, which is the
    /// recommended shape; a node-local directory is legal, and then
    /// each host holds its own copy. When absent, the convention
    /// default [`DEFAULT_DATA_PATH`] applies to the checks (`fdl probe`,
    /// pre-flight) but is NOT shipped to a run — see
    /// [`ClusterWorker::data_path`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_path: Option<String>,
    /// Join-window quorum knobs (dial-in membership). fdl-cli carries
    /// the block through the launcher envelope verbatim; flodl owns
    /// validation and the capacity-derived defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join: Option<ClusterJoin>,
}

/// `controller.join:` sub-block — per-field overrides for the
/// membership window. All fields optional; unset keeps flodl's fan-out
/// defaults (quorum = early-close target = configured capacity, window
/// 300s, hard cap 600s, pre-shared-salt admission).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterJoin {
    /// Quorum in ranks — the run cannot start below it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_rank_start: Option<usize>,
    /// Join window in seconds; quorum reached early does NOT close it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_timeout: Option<u64>,
    /// Early-close target in ranks: the window closes the moment this
    /// many are in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_ranks: Option<usize>,
    /// Hard cap in seconds: quorum still unmet when it expires fails
    /// the run loudly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_join_timeout: Option<u64>,
    /// Accept joins without pre-shared-salt authentication on a
    /// non-loopback bind (loudly warned by flodl).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_admission: Option<bool>,
    /// Roster-free formation: the join window alone defines the world,
    /// so `workers:` may be empty (walk-ins self-register via `fdl
    /// join`). Requires an explicit `min_rank_start` — there is no
    /// configured capacity to derive the quorum from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discovery: Option<bool>,
    /// Pre-shared session salt, hex (32 chars / 16 bytes). Injected at
    /// fleet-create time and presented by walk-ins as their join
    /// credential; forces authenticated admission even behind sshd.
    /// Mutually exclusive with `open_admission: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Discovery-only: bind the controller loopback-only so every
    /// walk-in must arrive through an sshd-carried forward
    /// (reachability = authentication). Requires a CPU averaging mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tunnel_only: Option<bool>,
    /// Who closes the window once quorum is met: `auto` (clock —
    /// target/expiry, the default), `manual` (only the operator, via
    /// `fdl start`; refuses `target_ranks`), or `hybrid` (clock, and
    /// the operator may fire earlier). flodl owns the semantics; fdl
    /// validates the value so a typo dies before any host is touched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
}

impl ClusterController {
    /// Effective shared-data path: `data_path` if set, else
    /// [`DEFAULT_DATA_PATH`].
    pub fn effective_data_path(&self) -> &str {
        self.data_path.as_deref().unwrap_or(DEFAULT_DATA_PATH)
    }
}

/// Top-level `join:` block — the worker-side counterpart of
/// `controller.join:`. Carries defaults for `fdl join` so a golden
/// image (or a systemd unit) can bake the whole dial-in recipe into
/// config and run bare `fdl join`. Every field is overridable on the
/// command line; flags win over this block.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerJoin {
    /// Controller mux address, `host[:port]` (default port 1337).
    /// Under `ssh:` this is the address as seen FROM the ssh host —
    /// usually loopback, the guardrailed-sshd-on-the-controller-box
    /// case.
    #[serde(default)]
    pub controller: Option<String>,
    /// SSH tunnel hop: `fdl join` brings up a local `-L` forward of
    /// the controller mux port through this host and dials loopback.
    /// Same shape as a worker's `ssh:` sub-block (target / port /
    /// user / identity_file / options).
    #[serde(default)]
    pub ssh: Option<SshConfig>,
    /// Pre-shared session credential (the run's
    /// `controller.join.token`), hex. Unset = open admission (the
    /// controller hands the salt out in the accept reply).
    #[serde(default)]
    pub token: Option<String>,
    /// Training binary to run in agent role, as a path on this box: use
    /// it as given. Mutually exclusive with `source:`, and one of the
    /// two is required — the binary IS the protocol (it dials, joins,
    /// and spawns this host's relay + rank children itself), so `fdl
    /// join` cannot default it.
    ///
    /// This is also the escape hatch for a box whose owner wants the
    /// last word on what it compiles and runs: a declared binary wins
    /// over any source a controller would hand it.
    #[serde(default)]
    pub bin: Option<String>,
    /// Build the training binary here instead of naming one. The box
    /// fetches the tree to local disk and compiles it against its OWN
    /// libtorch, which is what makes the ABI match by construction
    /// rather than by manifest discipline.
    #[serde(default)]
    pub source: Option<WorkerSource>,
    /// libtorch variant to acquire before building or running:
    /// `auto`, `cpu`, `cu126`, `cu128`, `rocm7.0`. It lands under
    /// `~/.flodl/libtorch/` (never the project tree, which on a walk-in
    /// is often a read-only mount) and becomes this box's active
    /// variant. Unset leaves the box on whatever it already has.
    ///
    /// `auto` is the fleet value: it routes on the devices THIS box has,
    /// so one golden image serves NVIDIA and AMD instances.
    #[serde(default)]
    pub libtorch: Option<String>,
    /// Logical host name presented in the join hello (default: this
    /// machine's hostname).
    #[serde(default)]
    pub host: Option<String>,
    /// GPU device indices to offer, one rank each (default: every
    /// device detection sees for the training build's vendor).
    #[serde(default)]
    pub devices: Option<Vec<u8>>,
    /// Keep dialing across runs: when the agent exits (run finished,
    /// controller gone, no window open yet) `fdl join` backs off and
    /// re-dials instead of exiting. The systemd / golden-image mode.
    #[serde(default)]
    pub persist: bool,
    /// Dataset source root on THIS box — a local path, the same role
    /// `cluster.workers[].data_path` plays for a fan-out host. `fdl
    /// join` verifies it before dialing and ships it to this host's
    /// ranks, so the training binary needs no data flag. Unset ships
    /// nothing: the binary keeps its own default.
    ///
    /// With `data_source:` set this is the MOUNTPOINT, defaulting to
    /// [`DEFAULT_DATA_PATH`].
    #[serde(default)]
    pub data_path: Option<String>,
    /// Transport that establishes `data_path:` when it is not already
    /// there, `<scheme>://<target>`. One scheme ships today:
    ///
    /// ```text
    /// sshfs://[user@]host[:port]/abs/path
    /// sshfs://[user@]host:/abs/path        # the scp spelling, same thing
    /// ```
    ///
    /// A source already mounted by provisioning needs nothing here —
    /// name its path in `data_path:` and leave this unset. The mount
    /// goes up READ-ONLY: a rank reads the source root and never
    /// writes it, so the kernel, not a convention, enforces that.
    #[serde(default)]
    pub data_source: Option<String>,
    /// Integrated-GPU host-RAM share for THIS box: the fraction of
    /// `MemTotal` its GPU aperture claims (the library's
    /// `gpu_ram_share` knob; discrete GPUs ignore it). Same role as a
    /// fan-out host's `cluster.workers[].gpu_ram_share`, riding the
    /// same envelope road: the agent writes it into this host's block
    /// at join time, where it overrides any cluster-scope default the
    /// controller stamped. Non-negative fraction of `MemTotal` (above
    /// 1.0 is legal where the platform under-states the aperture);
    /// unset ships nothing.
    #[serde(default)]
    pub gpu_ram_share: Option<f64>,
    /// Probe the training binary for its model signature before each
    /// dial (default: on). The probe re-runs the binary with
    /// `FLODL_INTERNAL_MODEL_SIG_PROBE` set; it builds its model on
    /// CPU, prints the signature and exits, and admission can then
    /// refuse a box building a different model without costing the
    /// run. `false` skips it — the formation-time check remains — for
    /// a binary whose startup is too heavy to run twice per dial, or
    /// one built against a flodl that predates the probe (which would
    /// otherwise run its whole `main` until the probe timeout kills
    /// it).
    #[serde(default)]
    pub sig_probe: Option<bool>,
    /// Arguments for the training binary (the binary's own training
    /// flags). A `--` tail on the command line replaces this list.
    #[serde(default)]
    pub args: Vec<String>,
}

/// `join.source:` — build the training binary on this box.
///
/// A tree always lands on LOCAL disk before it is built, whatever the
/// transport: cargo fingerprints by stat'ing every source file on every
/// invocation, so compiling over a network mount pays that latency
/// thousands of times, and the attribute caching that would hide the
/// latency makes cargo serve a stale binary. The fetch preserves mtimes,
/// which is what keeps the loop incremental rather than a cold rebuild
/// wearing an incremental costume.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSource {
    /// Transport plus location. Three ship:
    ///
    /// ```text
    /// file:///abs/path                      a directory on this box (a mount, a disk)
    /// rsync://[user@]host[:port]:/abs/path  a working tree over ssh
    /// git+https://host/owner/repo#<ref>     a pinned checkout (also git+ssh://)
    /// ```
    ///
    /// rsync is the one that carries UNCOMMITTED work, which is what a
    /// training crate living in no repo at all needs. `git+` wants a ref
    /// (`#<tag|branch|sha>`) because a default branch floats, and a
    /// floating ref is not a pin.
    pub from: String,
    /// Project directory inside the fetched tree (default: its root).
    /// Governs the build and the run both, so it answers "where is the
    /// project in this tree" exactly once. A path dep pointing outside
    /// this directory still resolves, which is why the fetched tree has
    /// to be the dep root and not just the crate.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Build recipe, run as a shell line in `cwd`. Default: `cargo build
    /// --release`, which is right whenever the crate carries its own
    /// `Cargo.toml`, lockfile and `rust-toolchain.toml` — the usual case
    /// for a crate that builds on the operator's own box.
    ///
    /// It can be a script committed beside the code (`./ci/node-build.sh`),
    /// so the recipe travels with the source while its invocation stays
    /// here. fdl exports what it resolved: `LIBTORCH_PATH`,
    /// `FDL_GPU_FEATURE` (say `--features "$FDL_GPU_FEATURE"` rather than
    /// naming a vendor) and `LD_LIBRARY_PATH`.
    ///
    /// It re-runs on every dial, so it must be cheap when nothing
    /// changed. cargo is; a recipe that rebuilds unconditionally is not.
    #[serde(default)]
    pub build: Option<String>,
    /// Built artifact, relative to `cwd` (e.g. `target/release/train`).
    /// Not guessed: `cargo build` in a workspace member writes to the
    /// WORKSPACE `target/`, not the member's, so any rule fdl invented
    /// would be wrong for someone.
    ///
    /// Optional because a published tree carries a run manifest that
    /// names it, and that manifest is the authority when present. Declare
    /// it here for a box the operator drives by hand.
    #[serde(default)]
    pub bin: Option<String>,
}

/// Top-level `publish:` block — the controller's standing answers for
/// `fdl publish`, so re-publishing a run is one bare command. Every
/// field mirrors a flag 1:1 and the FLAG WINS when both are given; a
/// `--` tail replaces `args:` outright (even an empty tail, because
/// "explicitly none" must be sayable — the args belong to the RUN).
///
/// `--no-build` has no field here ON PURPOSE: a standing config that
/// silently skips the gate would ship every future typo to the fleet.
/// The escape hatch stays a per-invocation decision.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishBlock {
    /// Source to publish, same grammar as `join.source.from:`
    /// (`file://`, `rsync://`, `git+https://…#<ref>`).
    #[serde(default)]
    pub source: Option<String>,
    /// Built artifact, relative to the project directory — what workers
    /// run (`--bin`).
    #[serde(default)]
    pub bin: Option<String>,
    /// Project directory inside the tree (`--cwd`; default: its root).
    #[serde(default)]
    pub cwd: Option<String>,
    /// Build recipe for the gate, a shell line (`--build`; default
    /// `cargo build --release`).
    #[serde(default)]
    pub build: Option<String>,
    /// Served directory (`--to`; default `~/.flodl/run`).
    #[serde(default)]
    pub to: Option<String>,
    /// Identity file for a source pulled over ssh (`--identity`).
    #[serde(default)]
    pub identity: Option<String>,
    /// The run's own arguments — what rank children re-enter the binary
    /// with. A command-line `--` tail replaces this list.
    #[serde(default)]
    pub args: Vec<String>,
}

/// CUDA device indices on a host. Either explicit (a list of indices) or
/// the `"all"` shorthand (auto-detect at startup on that host).
///
/// `local_devices: all` defers device-index resolution to startup on
/// whichever host the value lives on. The host uses `gpu_device_count()`
/// and binds devices `0..ranks.len()` (sequential from index 0). The
/// detected count must be at least as large as `ranks.len()`, otherwise
/// rendezvous errors loudly.
///
/// `local_devices: [0, 1]` is the explicit form; same as before this
/// shorthand existed.
///
/// Symmetric: works for both the controller host and remote nodes. Pair
/// with explicit `ranks: [N, N+1, ...]` to define the world topology;
/// `local_devices` only selects which device indices on this host map to
/// those ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalDevices {
    /// Auto-detect at startup on this host. Indices `0..ranks.len()` bound.
    All,
    /// Explicit list of CUDA device indices, paired positionally with ranks.
    Explicit(Vec<u8>),
}

impl LocalDevices {
    /// True for the auto-detect form (unresolved).
    pub fn is_all(&self) -> bool {
        matches!(self, LocalDevices::All)
    }

    /// Explicit indices as a slice, or `None` for `All`.
    pub fn as_explicit(&self) -> Option<&[u8]> {
        match self {
            LocalDevices::All => None,
            LocalDevices::Explicit(v) => Some(v.as_slice()),
        }
    }
}

impl Serialize for LocalDevices {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            LocalDevices::All => s.serialize_str("all"),
            LocalDevices::Explicit(v) => v.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for LocalDevices {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let v = Value::deserialize(d)?;
        match v {
            Value::String(s) if s == "all" => Ok(LocalDevices::All),
            Value::String(s) => Err(D::Error::custom(format!(
                "local_devices: expected \"all\" or array of device indices, got string {s:?}"
            ))),
            Value::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for (i, item) in arr.iter().enumerate() {
                    let n = item.as_u64().ok_or_else(|| {
                        D::Error::custom(format!(
                            "local_devices[{i}]: expected integer device index, got {item}"
                        ))
                    })?;
                    let d = u8::try_from(n).map_err(|_| {
                        D::Error::custom(format!(
                            "local_devices[{i}]: value {n} does not fit in u8"
                        ))
                    })?;
                    out.push(d);
                }
                Ok(LocalDevices::Explicit(out))
            }
            _ => Err(D::Error::custom(
                "local_devices: expected \"all\" or array of device indices",
            )),
        }
    }
}

/// SSH endpoint configuration for a remote worker host.
///
/// fdl-cli parser side; mirrors `flodl::distributed::launcher::SshConfig`
/// shape so the slim envelope round-trips cleanly into the launcher. All
/// fields optional; absent fields fall back to system ssh defaults or
/// `~/.ssh/config` rules.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SshConfig {
    /// SSH target hostname / IP / alias. Defaults to the worker's
    /// `host` field when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// SSH port (`ssh -p <port>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// SSH login user (`ssh -l <user>`). Defaults to the current user
    /// (or `FLODL_INTERNAL_HOST_USER` from the controller env).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Identity file / private key (`ssh -i <path>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity_file: Option<String>,
    /// Pass-through `-o Key=Value` SSH options (e.g.
    /// `"ProxyJump=bastion"`, `"StrictHostKeyChecking=no"`). Each entry
    /// becomes one `-o ...` arg on the spawned `ssh` command, in the
    /// declared order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
}

/// One worker (a physical host running one or more NCCL ranks).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterWorker {
    /// Hostname / identifier (was `name:` in the pre-Refactor-2 schema).
    /// Used for /etc/hosts resolution, `--add-host` injection, and as
    /// the default SSH target (override via `ssh:`). On the worker
    /// itself, this is the value `hostname` returns (or the
    /// `FLODL_HOST_NAME`-overridden value).
    pub host: String,
    /// Global ranks owned by this worker.
    ///
    /// NOT part of the user YAML schema — populated internally by
    /// [`ClusterConfig::populate_ranks`] from probed device counts
    /// (called by `prepare_cluster_env` before envelope emission).
    /// Sequential assignment by worker order: worker 0 owns
    /// `[0..counts[0])`, worker 1 owns
    /// `[counts[0]..counts[0]+counts[1])`, etc.
    ///
    /// Serialized into the wire format so the rank-side library reads
    /// the post-probe assignment via `FullCluster::from_value`, and
    /// deserializable so the canonical JSON round-trips. A `ranks:` key
    /// in USER yaml is rejected loudly at load
    /// (`loading::reject_user_ranks`): a user writing it expects it to
    /// pin ranks, and it never did (probe is authoritative).
    #[serde(default)]
    pub ranks: Vec<usize>,
    /// CUDA device indices paired by position with `ranks`, or `"all"`
    /// shorthand for auto-detect at startup. See [`LocalDevices`] for
    /// semantics.
    pub local_devices: LocalDevices,
    /// Network interface NCCL binds to (e.g. `virbr0`, `enp1s0`). Becomes
    /// `NCCL_SOCKET_IFNAME` in the spawned process environment.
    pub nccl_socket_ifname: String,
    /// Project checkout path on this host. `fdl-cli` cd's here before
    /// invoking the remote command. Heterogeneous mounts are fine
    /// (e.g. `/opt/flodl` on the controller, `/srv/flodl` in a VM); training
    /// data is the user's responsibility to mount identically across hosts
    /// (NAS / SMB / virtiofs / S3-FUSE).
    pub path: String,
    /// SSH endpoint for `fdl-cli`'s launcher. `None` means the host
    /// runs on the same machine as the launcher (fork/exec, no ssh).
    /// When `Some`, all fields inside are optional and fall back to
    /// system ssh defaults (or `~/.ssh/config` rules) when unset.
    ///
    /// In YAML, the sub-block accepts:
    ///
    /// ```yaml
    /// ssh:
    ///   target: node-b.lan          # ssh hostname/IP, default: host
    ///   port: 2222                  # -p <port>
    ///   user: ubuntu                # -l <user>
    ///   identity_file: ~/.ssh/key   # -i <path>
    ///   options:                    # -o <opt> (repeatable)
    ///     - ProxyJump=bastion
    ///     - StrictHostKeyChecking=no
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh: Option<SshConfig>,
    /// Route this worker's training traffic through its fan-out SSH
    /// session instead of a direct TCP connection to the controller:
    /// the launcher adds a remote forward to the host's relay SSH
    /// session and points the host at `127.0.0.1:<controller.port>`.
    /// Requires a CPU ElChe mode (NCCL's peer-to-peer data plane cannot
    /// ride a controller tunnel) and a remote host — validated loudly
    /// at launch. When EVERY remote worker is tunneled, the controller
    /// binds loopback only. Consumed by the library's launcher; fdl-cli
    /// just carries it through the envelope.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tunnel: bool,
    /// libtorch variant subpath under `<path>/libtorch/` on this host.
    /// E.g. `precompiled/cu128` for a Blackwell host on PT 2.10 cu128,
    /// `builds/sm61-sm120` for a Pascal host on a from-source build.
    /// Convention: the convention path
    /// `<host.path>/libtorch/<arch>` is what the rank exec reads at
    /// runtime; the controller-side build mirrors it via
    /// `<cluster.controller.path or controller-host.path>/libtorch/<arch>`.
    /// Both paths point at the same physical libtorch via the shared
    /// project-root mount.
    ///
    /// Optional; when unset, fdl-cli falls back to the host's
    /// project-root `.active` file (single-host default behaviour
    /// for non-cluster runs).
    ///
    /// `fdl probe`'s GPU compat check derives the supported sm
    /// architectures by parsing the basename of this value (e.g.
    /// `builds/sm61-sm120` → `6.1 12.0`) AND/OR reading the variant's
    /// `.arch` metadata file at
    /// `<path>/libtorch/<arch>/.arch`. The former wins when the
    /// basename encodes archs; the latter covers `precompiled/cuXXX`
    /// where the basename names the CUDA version, not GPU archs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// Dataset source root on this host: where its ranks READ training
    /// data from, alongside model checkpoints and per-rank logs.
    ///
    /// Usually a shared filesystem (NAS / SMB / virtiofs / S3-FUSE /
    /// SSHFS) reachable at the same logical path on every node, which
    /// is the recommended shape and the one multi-host checkpointing
    /// requires. A node-local directory is legal too, and then each
    /// host holds its own copy. `fdl probe` verifies the path exists +
    /// is readable on each host before training can fan out.
    ///
    /// **Declaring it changes a run.** The value travels to every rank
    /// through the launcher envelope and supplies the training binary's
    /// data directory (`LocalCluster::data_path` on the flodl side; an
    /// explicit `--data-dir` still wins). When absent, nothing travels
    /// and the binary keeps its own default: the convention default
    /// [`DEFAULT_DATA_PATH`] governs the CHECKS only, because shipping
    /// it would point every cluster that never mentioned data at a
    /// directory most hosts do not have.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_path: Option<String>,
    /// Integrated-GPU host-RAM share for THIS host: the fraction of
    /// `MemTotal` its GPU aperture claims (the library's
    /// `gpu_ram_share` knob; discrete GPUs ignore it). Host-hardware
    /// truth like `data_path`: it travels to this host's ranks through
    /// the launcher envelope and fills the training binary's config
    /// when that left the knob unset (an explicit `with_gpu_ram_share`
    /// in code still wins). Overrides the cluster-scope
    /// [`ClusterConfig::gpu_ram_share`] default. Non-negative fraction
    /// of `MemTotal` (above 1.0 is legal where the platform
    /// under-states the aperture).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_ram_share: Option<f64>,
    /// Names the docker compose service that provides this host's
    /// runtime environment (e.g. `cuda`, `dev`). It does NOT wrap the
    /// training exec: cluster fan-out runs each rank's binary directly
    /// on the host over SSH (the pre-flight build is what runs in a
    /// container, in the *controller's* [`ClusterController`] `docker`).
    /// This field is a probe-time signal only: when set, `fdl probe`
    /// skips host-level NCCL discovery — NCCL ships inside the image,
    /// not on the host — and reports "provided via Docker image
    /// `<svc>`" instead of erroring on a missing `libnccl.so`. The
    /// host's libtorch (resolved via the `<path>/libtorch/<arch>`
    /// convention) is still validated because it's the bind-mount
    /// target, not container state. Per-host (not global) because mixed
    /// deployments are common: controller in Docker, worker bare-metal
    /// (or vice-versa). Library ignores this field; consumed only by
    /// fdl-cli's probe / deploy paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docker: Option<String>,
    /// Host-scoped env vars exported into every rank child spawned on
    /// this host. Mapping `NAME: VALUE` (string→string). Useful for
    /// host-specific tuning that doesn't belong at cluster scope
    /// (e.g. one host needs a different `NCCL_SOCKET_IFNAME` override,
    /// or a custom `LD_LIBRARY_PATH` due to a non-standard CUDA
    /// install). Overrides matching keys from
    /// [`ClusterConfig::env`].
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub env: std::collections::BTreeMap<String, String>,
}

/// Convention default for [`ClusterWorker::data_path`] /
/// [`ClusterController::data_path`] when the entry does not declare
/// one. Maps to the cross-node mount that holds training data +
/// checkpoints + per-rank logs.
pub const DEFAULT_DATA_PATH: &str = "/flodl/data";

impl ClusterWorker {
    /// Effective shared-data path: `data_path` if set, else
    /// [`DEFAULT_DATA_PATH`].
    pub fn effective_data_path(&self) -> &str {
        self.data_path.as_deref().unwrap_or(DEFAULT_DATA_PATH)
    }
}

impl ClusterConfig {
    /// Total ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.workers.iter().map(|w| w.ranks.len()).sum()
    }

    /// Whether the cluster spans more than one physical worker.
    /// Single-worker clusters don't require `NCCL_SOCKET_IFNAME`.
    pub fn spans_multiple_hosts(&self) -> bool {
        self.workers.len() > 1
    }

    /// Pre-flight validation. Mirrors the library check so failures
    /// surface before SSH dispatch instead of from a stack trace on a
    /// remote host:
    ///
    /// - `controller.host` non-empty, `controller.path` non-empty
    /// - `workers` non-empty
    /// - per worker: `host` non-empty, `nccl_socket_ifname` non-empty
    ///   (when cluster spans multiple workers), `path` non-empty
    ///
    /// Ranks are NOT user input — they're computed by
    /// [`Self::populate_ranks`] from probed device counts. When this
    /// runs pre-probe (ranks empty), the rank-shape check is skipped;
    /// when called post-probe (ranks populated), the
    /// `0..world_size` + length-match-vs-local_devices invariant is
    /// enforced.
    pub fn validate(&self) -> Result<(), String> {
        if self.controller.host.trim().is_empty() {
            return Err("cluster.controller.host must be non-empty".into());
        }
        if self.controller.path.trim().is_empty() {
            return Err("cluster.controller.path must be non-empty".into());
        }
        let discovery = self
            .controller
            .join
            .as_ref()
            .is_some_and(|j| j.discovery == Some(true));
        if self.workers.is_empty() && !discovery {
            return Err("cluster.workers must be non-empty (a roster-free \
                        window needs `controller.join.discovery: true`)"
                .into());
        }
        if let Some(start) = self
            .controller
            .join
            .as_ref()
            .and_then(|j| j.start.as_deref())
            && !matches!(start, "auto" | "manual" | "hybrid") {
                return Err(format!(
                    "cluster.controller.join.start must be one of \
                     auto | manual | hybrid, got {start:?}"
                ));
            }
        // Reserved-env-key check: a user env map must not carry a key the
        // launcher owns per-rank. The launcher applies user env after its
        // own built-ins (shell last-wins), so an override would silently
        // break device mapping / rank identity / the HMAC envelope. The
        // reserved predicate lives in the library (single source of truth
        // for the launcher-owned names).
        for k in self.env.keys() {
            if crate::cluster::is_reserved_cluster_env_key(k) {
                return Err(format!(
                    "cluster.env: key {k:?} is reserved (launcher-owned) and \
                     cannot be set via env — it would override the launcher's \
                     per-rank value"
                ));
            }
        }
        for (i, w) in self.workers.iter().enumerate() {
            for k in w.env.keys() {
                if crate::cluster::is_reserved_cluster_env_key(k) {
                    return Err(format!(
                        "cluster.workers[{i}] ({:?}): env key {k:?} is reserved \
                         (launcher-owned) and cannot be set via env — it would \
                         override the launcher's per-rank value",
                        w.host,
                    ));
                }
            }
        }
        let multi_host = self.spans_multiple_hosts();
        for (i, w) in self.workers.iter().enumerate() {
            if w.host.trim().is_empty() {
                return Err(format!("cluster.workers[{i}].host must be non-empty"));
            }
            if multi_host && w.nccl_socket_ifname.trim().is_empty() {
                return Err(format!(
                    "cluster.workers[{i}] ({:?}): nccl_socket_ifname must be \
                     non-empty when the cluster spans multiple workers",
                    w.host
                ));
            }
            if w.path.trim().is_empty() {
                return Err(format!(
                    "cluster.workers[{i}] ({:?}): path (project checkout dir) \
                     must be non-empty",
                    w.host
                ));
            }
        }
        // Post-probe shape checks: only when ranks have been populated.
        // Pre-probe (all empty) skips this branch.
        if self.workers.iter().all(|w| !w.ranks.is_empty()) {
            for (i, w) in self.workers.iter().enumerate() {
                if let Some(devs) = w.local_devices.as_explicit()
                    && w.ranks.len() != devs.len() {
                        return Err(format!(
                            "cluster.workers[{i}] ({:?}): ranks ({}) and local_devices ({}) length mismatch",
                            w.host,
                            w.ranks.len(),
                            devs.len()
                        ));
                    }
            }
            let mut all: Vec<usize> = self
                .workers
                .iter()
                .flat_map(|w| w.ranks.iter().copied())
                .collect();
            let ws = all.len();
            all.sort_unstable();
            let expected: Vec<usize> = (0..ws).collect();
            if all != expected {
                return Err(format!(
                    "cluster: ranks across workers must be exactly 0..{ws} with no \
                     duplicates or gaps, got sorted-unique sequence {all:?}"
                ));
            }
        }
        Ok(())
    }

    /// Populate `workers[i].ranks` from probed device counts.
    /// Sequential assignment by worker order: worker 0 owns
    /// `[0..counts[0])`, worker 1 owns
    /// `[counts[0]..counts[0]+counts[1])`, etc.
    ///
    /// Errors when `device_counts.len() != workers.len()` or any
    /// count is 0. Workers' existing `ranks` are unconditionally
    /// overwritten — they're not user input, the probe is
    /// authoritative.
    ///
    /// Caller orchestration (see `prepare_cluster_env`):
    /// 1. parse YAML → ClusterConfig (ranks empty by serde default)
    /// 2. probe device counts per worker
    /// 3. call `populate_ranks` to fill in
    /// 4. validate (now ranks are non-empty → shape checks run)
    /// 5. serialize and ship via FLODL_INTERNAL_FULL_CLUSTER_JSON
    pub fn populate_ranks(&mut self, device_counts: &[usize]) -> Result<(), String> {
        if device_counts.len() != self.workers.len() {
            return Err(format!(
                "populate_ranks: device_counts len {} != workers len {}",
                device_counts.len(),
                self.workers.len(),
            ));
        }
        let mut next_rank = 0usize;
        for (i, w) in self.workers.iter_mut().enumerate() {
            let count = device_counts[i];
            if count == 0 {
                return Err(format!(
                    "populate_ranks: worker[{i}] ({:?}) reported 0 devices",
                    w.host,
                ));
            }
            w.ranks = (next_rank..next_rank + count).collect();
            next_rank += count;
        }
        Ok(())
    }

    /// Canonical JSON of the full topology. Used for debug-dumping the
    /// controller-side view; per-worker envelopes go through
    /// [`Self::local_envelope_for`] instead.
    pub fn canonical_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("cluster: JSON serialization failed: {e}"))
    }

    /// Build the slim per-worker envelope the library reads via
    /// `flodl::distributed::LocalCluster::from_env`.
    ///
    /// The envelope strips launcher-only fields (`ssh*`), embeds derived
    /// world metadata (`world_size`, `num_workers`), and carries only
    /// the requested worker's slice. The launcher hex-encodes the
    /// resulting JSON into `FLODL_INTERNAL_CLUSTER_JSON` per ssh invocation, so
    /// each remote process sees only itself + the controller
    /// coordinates.
    pub fn local_envelope_for(&self, worker: &ClusterWorker) -> Value {
        let mut worker_obj = serde_json::Map::new();
        worker_obj.insert("host".into(), Value::String(worker.host.clone()));
        worker_obj.insert(
            "ranks".into(),
            Value::Array(worker.ranks.iter().map(|r| Value::from(*r)).collect()),
        );
        worker_obj.insert(
            "local_devices".into(),
            match &worker.local_devices {
                LocalDevices::All => Value::String("all".into()),
                LocalDevices::Explicit(v) => {
                    Value::Array(v.iter().map(|d| Value::from(*d)).collect())
                }
            },
        );
        worker_obj.insert(
            "nccl_socket_ifname".into(),
            Value::String(worker.nccl_socket_ifname.clone()),
        );
        worker_obj.insert("path".into(), Value::String(worker.path.clone()));
        if let Some(a) = &worker.arch {
            worker_obj.insert("arch".into(), Value::String(a.clone()));
        }
        // SSH fields (port/user/identity_file/options) are launcher-only.
        // The slim per-rank envelope read by `LocalCluster::from_env`
        // doesn't need them — the rank-side library has nothing to do
        // with SSH. They're already on the FULL envelope via serde on
        // `ClusterConfig` so the launcher picks them up there.
        //
        // Dataset source root: DECLARED values only, never
        // `effective_data_path()`. The convention default names a path
        // that most hosts do not actually have, and the rank consumes
        // this one (it reaches `--data-dir` through
        // `LocalCluster::data_path`), so shipping the default would
        // point every run that never mentioned data at a directory that
        // is not there. Absent = the training binary keeps its own
        // default. Same split `check_remote_data_path` already makes:
        // a declared path that is missing is an error, a convention
        // default that is missing is a warning.
        if let Some(dp) = &worker.data_path {
            worker_obj.insert("data_path".into(), Value::String(dp.clone()));
        }

        let mut controller_obj = serde_json::Map::new();
        controller_obj.insert("host".into(), Value::String(self.controller.host.clone()));
        controller_obj.insert("port".into(), Value::from(self.controller.port));

        let mut envelope = serde_json::Map::new();
        envelope.insert("controller".into(), Value::Object(controller_obj));
        envelope.insert("world_size".into(), Value::from(self.world_size()));
        envelope.insert("num_workers".into(), Value::from(self.workers.len()));
        envelope.insert("worker".into(), Value::Object(worker_obj));
        Value::Object(envelope)
    }

    /// Look up the SSH target for a worker, defaulting to `host` if
    /// `ssh:` is not set. Used by the launcher; library callers don't
    /// need this.
    pub fn ssh_target<'a>(&'a self, worker: &'a ClusterWorker) -> &'a str {
        worker
            .ssh
            .as_ref()
            .and_then(|s| s.target.as_deref())
            .unwrap_or(&worker.host)
    }
}
