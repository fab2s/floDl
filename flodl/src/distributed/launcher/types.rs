//! Cluster topology model + JSON deserialization.
//!
//! Mirrors the `cluster.yml` schema that fdl-cli serializes into
//! `FLODL_INTERNAL_FULL_CLUSTER_JSON`; `FullCluster::from_env()` is the launcher-side
//! entry point.

use std::env;

use crate::tensor::{Result, TensorError};

use super::ENV_FULL_CLUSTER_JSON;

// ---------------------------------------------------------------------------
// SshConfig: per-worker SSH endpoint knobs.
// ---------------------------------------------------------------------------

/// SSH endpoint configuration for a remote worker host.
///
/// Carries the per-host SSH knobs used by the launcher when fanning
/// out to remote ranks. All fields are optional; when absent, the
/// corresponding flag is omitted from the spawned `ssh` command and
/// system ssh defaults (or `~/.ssh/config` rules) apply.
///
/// In YAML, this lives under each worker's `ssh:` sub-block, e.g.:
///
/// ```yaml
/// workers:
///   - host: node-b
///     ssh:
///       target: node-b.lan
///       port: 2222
///       user: ubuntu
///       identity_file: ~/.ssh/id_ed25519
///       options:
///         - ProxyJump=bastion
/// ```
///
/// The `host:` (logical name) and `ssh.target:` (network endpoint)
/// split lets the worker's logical identity differ from its SSH
/// target. When `target` is unset, the launcher falls back to the
/// worker's `host` name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SshConfig {
    /// SSH target hostname / IP / alias. Defaults to the worker's
    /// `host` when `None`.
    pub target: Option<String>,
    /// SSH port. Maps to `ssh -p <port>`.
    pub port: Option<u16>,
    /// SSH login user. Maps to `ssh -l <user>`. Falls back to the
    /// current user (or `FLODL_INTERNAL_HOST_USER` from env) when `None`.
    pub user: Option<String>,
    /// Identity file (private key) path. Maps to `ssh -i <path>`.
    pub identity_file: Option<String>,
    /// Pass-through `-o Key=Value` SSH options (e.g.
    /// `"ProxyJump=bastion"`, `"StrictHostKeyChecking=no"`). Each
    /// entry becomes one `-o ...` arg on the spawned `ssh` command,
    /// in the order declared.
    pub options: Vec<String>,
}

// ---------------------------------------------------------------------------
// FullCluster: launcher-side parser for the multi-host topology.
// ---------------------------------------------------------------------------

/// Full cluster topology as seen by the launcher process.
///
/// Mirrors flodl-cli's `ClusterConfig` shape; lives on the flodl side so
/// the framework owns cluster orchestration end-to-end. The slim
/// per-rank envelopes parsed by [`LocalCluster`] are derived from this
/// view at fan-out time.
///
/// Like [`LocalCluster`], the rank-side `local_devices: "all"` shorthand
/// is resolved at parse time when applicable (host-side resolution
/// happens later, after envelope ship — see [`crate::distributed::cluster`]
/// for the slim path).
///
/// [`LocalCluster`]: crate::distributed::cluster::LocalCluster
#[derive(Debug, Clone)]
pub struct FullCluster {
    /// Controller's rendezvous bind point + pre-flight build context.
    pub controller: FullController,
    /// All rank-carrying entries.
    pub workers: Vec<FullWorker>,
    /// 128-bit session salt the launcher generates fresh per training
    /// session and propagates to every rank's slim envelope. Used as the
    /// HMAC key for the cross-process control + data channels. All
    /// zeros until [`FullCluster::with_session_salt`] (or
    /// [`super::run_launcher_with_config`]) populates it.
    pub salt: crate::distributed::wire::SessionSalt,
    /// Cluster-scope env vars exported into every rank child's
    /// environment. Cluster-yml `env:` block (mapping `NAME: VALUE`).
    /// Used for cluster-specific tuning that the launcher itself
    /// shouldn't hardcode — e.g. setting `NCCL_P2P_DISABLE=1` +
    /// `NCCL_SHM_DISABLE=1` for the Pascal-under-VFIO rig where NCCL's
    /// direct-IPC transports fail but socket transport works.
    ///
    /// Empty by default. Per-worker envs (see [`FullWorker::env`])
    /// override per-cluster ones for the matching worker.
    pub env: std::collections::BTreeMap<String, String>,
    /// Cluster-scope default for the integrated-GPU host-RAM share
    /// (fraction of `MemTotal` the GPU aperture claims; see
    /// `DataLoaderBuilder::gpu_ram_share`). Meaningful only on APU
    /// hosts; discrete-GPU hosts ignore it, which is what makes a
    /// fleet-wide default legal on a mixed fleet. A per-worker
    /// [`FullWorker::gpu_ram_share`] overrides it, and a walk-in's own
    /// `join.gpu_ram_share:` overrides both (host truth wins). `None`
    /// emits nothing, so a cluster that never mentions APUs is
    /// unaffected by this field existing.
    pub gpu_ram_share: Option<f64>,
}

/// Controller-side fields, launcher view.
#[derive(Debug, Clone)]
pub struct FullController {
    pub host: String,
    pub port: u16,
    pub path: String,
    pub docker: Option<String>,
    pub arch: Option<String>,
    pub data_path: Option<String>,
    /// User overrides for the join-window quorum knobs (`join:`
    /// sub-block). `None` / unset fields keep the fan-out defaults:
    /// quorum = early-close target = the configured capacity (the
    /// window closes the instant everything fan-out started is in, and
    /// the run cannot start below it — the direct-spawn era's
    /// all-or-nothing, unchanged unless asked for).
    pub join: Option<JoinKnobs>,
}

/// Optional per-field overrides for the membership window, mirroring
/// the `controller.join:` cluster.yml sub-block. Every field is
/// optional; the launcher fills the gaps from its capacity-derived
/// defaults. See the membership module for the window semantics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JoinKnobs {
    /// Quorum in ranks — the run cannot start below it. Default: the
    /// configured capacity.
    pub min_rank_start: Option<usize>,
    /// Join window in seconds (`join_timeout:`). Default 300.
    pub join_timeout_secs: Option<u64>,
    /// Early-close target in ranks. Default: the configured capacity;
    /// set it higher (with a matching window) to leave room for
    /// self-deployed workers to dial in alongside the managed rig.
    pub target_ranks: Option<usize>,
    /// Hard cap in seconds (`max_join_timeout:`). Default 600, raised
    /// to `join_timeout` when that is set higher.
    pub max_join_timeout_secs: Option<u64>,
    /// Accept joins without pre-shared-salt authentication on a
    /// non-loopback bind. Loudly warned; see the trust model.
    pub open_admission: Option<bool>,
    /// Roster-free formation: the window alone defines the world. The
    /// `workers:` list may be empty (walk-ins self-register against
    /// `bind addr + credential`); an enumerated roster is still fanned
    /// out as usual and walk-ins join alongside it. Requires an
    /// explicit `min_rank_start` — with no roster there is no capacity
    /// to derive the quorum from.
    pub discovery: Option<bool>,
    /// Pre-shared session salt, hex (32 chars / 16 bytes). Replaces the
    /// launcher's per-run generated salt so a credential can be injected
    /// at fleet-create time and presented by walk-ins (`fdl join
    /// --token`). Forces authenticated admission even on a loopback
    /// bind (layered with the sshd guardrails, not an alternative to
    /// them); mutually exclusive with `open_admission: true`.
    pub token: Option<String>,
    /// Bind the controller mux loopback-only with no enumerated tunnel
    /// workers to infer it from: every walk-in must arrive through an
    /// sshd-carried forward, making reachability itself the
    /// authentication. Discovery-only knob (a roster infers the bind
    /// scope from its `tunnel:` flags); requires a CPU averaging mode,
    /// exactly like per-worker `tunnel: true`.
    pub tunnel_only: Option<bool>,
    /// Who closes the window once quorum is met (`start:` in YAML):
    /// `auto` (clock — target/expiry, the default), `manual` (only the
    /// operator, via `fdl start`), or `hybrid` (clock, operator may
    /// fire earlier).
    pub start: Option<crate::distributed::membership::StartMode>,
}

impl FullCluster {
    /// Replace the session salt and return self for chaining. Called
    /// by [`super::run_launcher_with_config`] once it has generated a fresh salt.
    pub fn with_session_salt(mut self, salt: crate::distributed::wire::SessionSalt) -> Self {
        self.salt = salt;
        self
    }
}

/// One worker's entry in the full topology, launcher-side.
///
/// Differs from [`WorkerBlock`] by carrying `ssh:` (launcher-only field
/// stripped from slim envelopes) and the unresolved `local_devices:
/// "all"` form (which is only resolved on the host that will use it).
///
/// [`WorkerBlock`]: crate::distributed::cluster::WorkerBlock
#[derive(Debug, Clone)]
pub struct FullWorker {
    pub host: String,
    pub ranks: Vec<usize>,
    /// Either an explicit list of CUDA indices or `None` for the `"all"`
    /// shorthand (resolved at startup on the host that owns this entry).
    pub local_devices: Option<Vec<u8>>,
    pub nccl_socket_ifname: String,
    pub path: String,
    /// libtorch variant subpath under `<path>/libtorch/`. The runtime
    /// libtorch lives at `<path>/libtorch/<arch>/` by convention; the
    /// launcher uses this to build the remote-side LD_LIBRARY_PATH
    /// when no pre-flight envelope overrides it.
    pub arch: Option<String>,
    /// Dataset source root on this worker: where a rank READS its
    /// training data from. May be a shared mount visible to every host
    /// or a node-local directory; the library does not care which, it
    /// only forwards the path to the rank.
    ///
    /// `None` when the host did not declare `data_path:`, in which case
    /// the run keeps whatever default its own CLI defines. Only an
    /// EXPLICIT declaration travels, so a cluster that never mentions
    /// data paths is unaffected by this field existing.
    pub data_path: Option<String>,
    /// Integrated-GPU host-RAM share for THIS host (fraction of
    /// `MemTotal`; see `DataLoaderBuilder::gpu_ram_share`). Overrides
    /// the cluster-scope [`FullCluster::gpu_ram_share`] default.
    /// Host-hardware truth like `data_path`: only an explicit
    /// declaration travels.
    pub gpu_ram_share: Option<f64>,
    /// SSH endpoint for remote dispatch. `None` means the host runs
    /// on the same machine as the launcher (fork/exec path, no ssh).
    /// When `Some`, all fields inside are optional and fall back to
    /// system ssh defaults (or `~/.ssh/config` rules) when unset.
    pub ssh: Option<SshConfig>,
    /// Route this host's training traffic through the fan-out SSH
    /// session instead of a direct TCP connection to the controller.
    /// The launcher adds a remote forward (`-R port:127.0.0.1:port`) to
    /// the host's relay SSH session and points the host at
    /// `127.0.0.1:<controller.port>` — its loopback end of the tunnel.
    /// Requires a CPU ElChe mode (NCCL's peer-to-peer data plane cannot
    /// ride a controller tunnel) and a remote host. When EVERY remote
    /// worker is tunneled, the controller binds loopback only — the
    /// port is then unreachable except through sshd.
    pub tunnel: bool,
    /// Per-host env vars exported into this host's rank children.
    /// Override the cluster-scope [`FullCluster::env`] for matching
    /// keys. Use for host-specific tuning (e.g. an interface override
    /// only one host needs).
    pub env: std::collections::BTreeMap<String, String>,
}

impl FullWorker {
    /// SSH target for this worker, defaulting to `host` when
    /// `ssh.target` is unset or `ssh` itself is `None`. Used by the
    /// launcher's `build_ssh_spawn_command` and by `fdl probe`.
    pub fn ssh_target(&self) -> &str {
        self.ssh
            .as_ref()
            .and_then(|s| s.target.as_deref())
            .unwrap_or(&self.host)
    }
}

impl FullCluster {
    /// Read + parse the full topology from [`ENV_FULL_CLUSTER_JSON`].
    ///
    /// Loud errors on missing var, hex/JSON decode failure, or schema
    /// violations. The launcher-only path; not relevant on rank children.
    pub fn from_env() -> Result<Self> {
        let raw = env::var(ENV_FULL_CLUSTER_JSON).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: reading {ENV_FULL_CLUSTER_JSON} failed: {e}"
            ))
        })?;
        let bytes = crate::distributed::cluster::hex_decode(raw.trim()).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: {ENV_FULL_CLUSTER_JSON} hex-decode failed: {e}"
            ))
        })?;
        let val: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            TensorError::new(&format!(
                "cluster launcher: {ENV_FULL_CLUSTER_JSON} JSON parse failed: {e}"
            ))
        })?;
        Self::from_value(&val)
    }

    /// Parse from a pre-decoded JSON value. Test entry point + future
    /// programmatic callers.
    pub fn from_value(val: &serde_json::Value) -> Result<Self> {
        let obj = val.as_object().ok_or_else(|| {
            TensorError::new("cluster launcher: top-level JSON must be an object")
        })?;

        let controller_val = obj
            .get("controller")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TensorError::new("cluster launcher: controller (object) required"))?;
        let controller_host = controller_val
            .get("host")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensorError::new("cluster launcher: controller.host (string) required"))?
            .to_string();
        if controller_host.trim().is_empty() {
            return Err(TensorError::new(
                "cluster launcher: controller.host must be non-empty",
            ));
        }
        let controller_port_u64 = controller_val
            .get("port")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| TensorError::new("cluster launcher: controller.port (u16) required"))?;
        let controller_port = u16::try_from(controller_port_u64).map_err(|_| {
            TensorError::new(&format!(
                "cluster launcher: controller.port must fit in u16 (got {controller_port_u64})"
            ))
        })?;
        let controller_path = controller_val
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| TensorError::new("cluster launcher: controller.path (string) required"))?
            .to_string();
        let controller_docker = controller_val
            .get("docker")
            .and_then(|v| v.as_str())
            .map(String::from);
        let controller_arch = controller_val
            .get("arch")
            .and_then(|v| v.as_str())
            .map(String::from);
        let controller_data_path = controller_val
            .get("data_path")
            .and_then(|v| v.as_str())
            .map(String::from);
        let controller_join = parse_join_block(controller_val.get("join"))?;

        let workers_val = obj
            .get("workers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| TensorError::new("cluster launcher: workers (array) required"))?;
        let discovery = controller_join
            .as_ref()
            .is_some_and(|j| j.discovery == Some(true));
        if workers_val.is_empty() && !discovery {
            return Err(TensorError::new(
                "cluster launcher: workers must be non-empty (a roster-free \
                 window needs `controller.join.discovery: true`)",
            ));
        }

        let workers: Vec<FullWorker> = workers_val
            .iter()
            .enumerate()
            .map(|(i, w)| parse_full_worker(w, i))
            .collect::<Result<_>>()?;

        // Cross-worker rank check: union must be exactly 0..world_size.
        let mut all: Vec<usize> = workers
            .iter()
            .flat_map(|w| w.ranks.iter().copied())
            .collect();
        let ws = all.len();
        all.sort_unstable();
        let expected: Vec<usize> = (0..ws).collect();
        if all != expected {
            return Err(TensorError::new(&format!(
                "cluster launcher: ranks across workers must be exactly 0..{ws} \
                 with no duplicates or gaps, got sorted-unique sequence {all:?}"
            )));
        }

        // Optional cluster-scope `env:` block: mapping of NAME → VALUE
        // exported into every rank child. Missing → empty map.
        let env = parse_env_block(obj.get("env"), "cluster.env")?;

        let gpu_ram_share = parse_gpu_ram_share(obj.get("gpu_ram_share"), "cluster.gpu_ram_share")?;

        Ok(FullCluster {
            controller: FullController {
                host: controller_host,
                port: controller_port,
                path: controller_path,
                docker: controller_docker,
                arch: controller_arch,
                data_path: controller_data_path,
                join: controller_join,
            },
            workers,
            // ENV_FULL_CLUSTER_JSON is the config snapshot fdl-cli ships;
            // the session salt is generated freshly by `run_launcher` per
            // training session (override via [`Self::with_session_salt`]).
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
            env,
            gpu_ram_share,
        })
    }

    /// Total ranks across the cluster.
    pub fn world_size(&self) -> usize {
        self.workers.iter().map(|w| w.ranks.len()).sum()
    }

    /// Whether the cluster spans more than one physical worker.
    pub fn spans_multiple_workers(&self) -> bool {
        self.workers.len() > 1
    }

    /// Serialize to the JSON shape [`Self::from_value`] parses. Symmetric
    /// round-trip: `FullCluster::from_value(&cluster.to_json()) == cluster`.
    /// Used by [`crate::distributed::Trainer::run`] to convert a
    /// programmatic [`crate::distributed::ClusterBuilder`] result into the
    /// `FLODL_INTERNAL_FULL_CLUSTER_JSON` env-var contract the launcher path
    /// reads. `salt` is intentionally NOT serialized — the launcher
    /// generates a fresh session salt per run.
    pub fn to_json(&self) -> serde_json::Value {
        let workers: Vec<serde_json::Value> = self
            .workers
            .iter()
            .map(|h| {
                let mut o = serde_json::Map::new();
                o.insert("host".into(), serde_json::Value::String(h.host.clone()));
                o.insert(
                    "ranks".into(),
                    serde_json::Value::Array(
                        h.ranks
                            .iter()
                            .map(|r| serde_json::Value::from(*r))
                            .collect(),
                    ),
                );
                let ld = match &h.local_devices {
                    None => serde_json::Value::String("all".into()),
                    Some(v) => serde_json::Value::Array(
                        v.iter().map(|d| serde_json::Value::from(*d)).collect(),
                    ),
                };
                o.insert("local_devices".into(), ld);
                o.insert(
                    "nccl_socket_ifname".into(),
                    serde_json::Value::String(h.nccl_socket_ifname.clone()),
                );
                o.insert("path".into(), serde_json::Value::String(h.path.clone()));
                if let Some(a) = &h.arch {
                    o.insert("arch".into(), serde_json::Value::String(a.clone()));
                }
                if let Some(d) = &h.data_path {
                    o.insert("data_path".into(), serde_json::Value::String(d.clone()));
                }
                if let Some(g) = h.gpu_ram_share {
                    o.insert("gpu_ram_share".into(), serde_json::Value::from(g));
                }
                if let Some(s) = &h.ssh {
                    let mut ssh_obj = serde_json::Map::new();
                    if let Some(t) = &s.target {
                        ssh_obj.insert("target".into(), serde_json::Value::String(t.clone()));
                    }
                    if let Some(p) = s.port {
                        ssh_obj.insert("port".into(), serde_json::Value::from(p));
                    }
                    if let Some(u) = &s.user {
                        ssh_obj.insert("user".into(), serde_json::Value::String(u.clone()));
                    }
                    if let Some(i) = &s.identity_file {
                        ssh_obj
                            .insert("identity_file".into(), serde_json::Value::String(i.clone()));
                    }
                    if !s.options.is_empty() {
                        ssh_obj.insert(
                            "options".into(),
                            serde_json::Value::Array(
                                s.options
                                    .iter()
                                    .map(|opt| serde_json::Value::String(opt.clone()))
                                    .collect(),
                            ),
                        );
                    }
                    o.insert("ssh".into(), serde_json::Value::Object(ssh_obj));
                }
                if h.tunnel {
                    o.insert("tunnel".into(), serde_json::Value::Bool(true));
                }
                if !h.env.is_empty() {
                    let mut env_obj = serde_json::Map::new();
                    for (k, v) in &h.env {
                        env_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
                    }
                    o.insert("env".into(), serde_json::Value::Object(env_obj));
                }
                serde_json::Value::Object(o)
            })
            .collect();
        let mut top = serde_json::Map::new();
        let mut controller_obj = serde_json::Map::new();
        controller_obj.insert(
            "host".into(),
            serde_json::Value::String(self.controller.host.clone()),
        );
        controller_obj.insert("port".into(), serde_json::Value::from(self.controller.port));
        controller_obj.insert(
            "path".into(),
            serde_json::Value::String(self.controller.path.clone()),
        );
        if let Some(s) = &self.controller.docker {
            controller_obj.insert("docker".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(s) = &self.controller.arch {
            controller_obj.insert("arch".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(s) = &self.controller.data_path {
            controller_obj.insert("data_path".into(), serde_json::Value::String(s.clone()));
        }
        if let Some(j) = &self.controller.join {
            let mut join_obj = serde_json::Map::new();
            if let Some(n) = j.min_rank_start {
                join_obj.insert("min_rank_start".into(), serde_json::Value::from(n));
            }
            if let Some(n) = j.join_timeout_secs {
                join_obj.insert("join_timeout".into(), serde_json::Value::from(n));
            }
            if let Some(n) = j.target_ranks {
                join_obj.insert("target_ranks".into(), serde_json::Value::from(n));
            }
            if let Some(n) = j.max_join_timeout_secs {
                join_obj.insert("max_join_timeout".into(), serde_json::Value::from(n));
            }
            if let Some(b) = j.open_admission {
                join_obj.insert("open_admission".into(), serde_json::Value::Bool(b));
            }
            if let Some(b) = j.discovery {
                join_obj.insert("discovery".into(), serde_json::Value::Bool(b));
            }
            if let Some(t) = &j.token {
                join_obj.insert("token".into(), serde_json::Value::String(t.clone()));
            }
            if let Some(b) = j.tunnel_only {
                join_obj.insert("tunnel_only".into(), serde_json::Value::Bool(b));
            }
            if let Some(m) = j.start {
                use crate::distributed::membership::StartMode;
                let s = match m {
                    StartMode::Auto => "auto",
                    StartMode::Manual => "manual",
                    StartMode::Hybrid => "hybrid",
                };
                join_obj.insert("start".into(), serde_json::Value::String(s.into()));
            }
            if !join_obj.is_empty() {
                controller_obj.insert("join".into(), serde_json::Value::Object(join_obj));
            }
        }
        top.insert(
            "controller".into(),
            serde_json::Value::Object(controller_obj),
        );
        top.insert("workers".into(), serde_json::Value::Array(workers));
        if let Some(g) = self.gpu_ram_share {
            top.insert("gpu_ram_share".into(), serde_json::Value::from(g));
        }
        if !self.env.is_empty() {
            let mut env_obj = serde_json::Map::new();
            for (k, v) in &self.env {
                env_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            top.insert("env".into(), serde_json::Value::Object(env_obj));
        }
        serde_json::Value::Object(top)
    }
}

fn parse_full_worker(v: &serde_json::Value, i: usize) -> Result<FullWorker> {
    let obj = v.as_object().ok_or_else(|| {
        TensorError::new(&format!("cluster launcher: workers[{i}] must be an object"))
    })?;

    let host = obj
        .get("host")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}].host (string) required"
            ))
        })?
        .to_string();
    if host.trim().is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: workers[{i}].host must be non-empty"
        )));
    }
    let name = host;

    let ranks_arr = obj.get("ranks").and_then(|v| v.as_array()).ok_or_else(|| {
        TensorError::new(&format!(
            "cluster launcher: workers[{i}] ({name:?}): ranks (array) required"
        ))
    })?;
    // Empty ranks: orchestrator-only host entry. Declared in cluster.yml
    // solely so fdl-cli's pre-flight build can read its `docker:` /
    // `arch:` for controller-side build context; the launcher itself
    // skips it (no rank spawn for this host). Distinct from "host
    // absent from cluster.workers" — both result in orchestrator-only
    // launcher behavior, but the explicit entry surfaces config to
    // fdl-cli.
    let ranks: Vec<usize> = ranks_arr
        .iter()
        .enumerate()
        .map(|(j, e)| {
            let n = e.as_u64().ok_or_else(|| {
                TensorError::new(&format!(
                    "cluster launcher: workers[{i}].ranks[{j}]: non-integer entry"
                ))
            })?;
            usize::try_from(n).map_err(|_| {
                TensorError::new(&format!(
                    "cluster launcher: workers[{i}].ranks[{j}]: value {n} out of range"
                ))
            })
        })
        .collect::<Result<_>>()?;

    let local_devices = match obj.get("local_devices") {
        None => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): local_devices required"
            )));
        }
        Some(serde_json::Value::String(s)) if s == "all" => None,
        Some(serde_json::Value::String(s)) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): local_devices: \
                 expected \"all\" or array, got string {s:?}"
            )));
        }
        Some(serde_json::Value::Array(arr)) => {
            let v: Vec<u8> = arr
                .iter()
                .enumerate()
                .map(|(j, e)| {
                    let n = e.as_u64().ok_or_else(|| {
                        TensorError::new(&format!(
                            "cluster launcher: workers[{i}].local_devices[{j}]: \
                             non-integer entry"
                        ))
                    })?;
                    u8::try_from(n).map_err(|_| {
                        TensorError::new(&format!(
                            "cluster launcher: workers[{i}].local_devices[{j}]: \
                             value {n} does not fit in u8"
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            if v.len() != ranks.len() {
                return Err(TensorError::new(&format!(
                    "cluster launcher: workers[{i}] ({name:?}): ranks ({}) and \
                     local_devices ({}) length mismatch",
                    ranks.len(),
                    v.len()
                )));
            }
            Some(v)
        }
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): local_devices: \
                 expected \"all\" or array, got {other}"
            )));
        }
    };

    let nccl_socket_ifname = obj
        .get("nccl_socket_ifname")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): nccl_socket_ifname (string) required"
            ))
        })?
        .to_string();

    let path = obj
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): path (string) required"
            ))
        })?
        .to_string();
    if path.trim().is_empty() {
        return Err(TensorError::new(&format!(
            "cluster launcher: workers[{i}] ({name:?}): path must be non-empty"
        )));
    }

    let arch = obj.get("arch").and_then(|v| v.as_str()).map(String::from);

    let data_path = obj
        .get("data_path")
        .and_then(|v| v.as_str())
        .map(String::from);

    let gpu_ram_share = parse_gpu_ram_share(
        obj.get("gpu_ram_share"),
        &format!("workers[{i}] ({name:?}).gpu_ram_share"),
    )?;

    let ssh = parse_ssh_block(obj.get("ssh"), &format!("workers[{i}] ({name:?})"))?;

    let tunnel = match obj.get("tunnel") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: workers[{i}] ({name:?}): tunnel must be a \
                 boolean, got {other}"
            )));
        }
    };

    let env = parse_env_block(obj.get("env"), &format!("workers[{i}] ({name:?}).env"))?;

    Ok(FullWorker {
        host: name,
        ranks,
        local_devices,
        nccl_socket_ifname,
        path,
        arch,
        data_path,
        gpu_ram_share,
        ssh,
        tunnel,
        env,
    })
}

/// Parse a `gpu_ram_share` field: a non-negative fraction of `MemTotal`.
/// Values above 1.0 are deliberately legal — the knob exists partly for
/// platforms whose `MemTotal` under-states what the APU can address —
/// so the only refusals are a negative, a non-finite, or a non-number
/// (the same envelope `with_gpu_ram_share`'s `max(0.0)` clamp draws,
/// made loud because yml is user input).
fn parse_gpu_ram_share(v: Option<&serde_json::Value>, label: &str) -> Result<Option<f64>> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f >= 0.0 => Ok(Some(f)),
            _ => Err(TensorError::new(&format!(
                "cluster launcher: {label} must be a non-negative fraction \
                 of host RAM (e.g. 0.5), got {v}"
            ))),
        },
    }
}

/// Parse a `controller.join:` sub-block into [`JoinKnobs`]. Missing /
/// null yields `None` (all defaults); present fields are validated for
/// type but NOT cross-field consistency — that happens once, against
/// the capacity-filled config, in the membership gate.
fn parse_join_block(v: Option<&serde_json::Value>) -> Result<Option<JoinKnobs>> {
    let obj = match v {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::Object(m)) => m,
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: controller.join must be a map \
                 (min_rank_start, join_timeout, target_ranks, \
                 max_join_timeout, open_admission), got {other}"
            )));
        }
    };
    let uint = |key: &str| -> Result<Option<u64>> {
        match obj.get(key) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(v) => match v.as_u64() {
                Some(n) => Ok(Some(n)),
                None => Err(TensorError::new(&format!(
                    "cluster launcher: controller.join.{key} must be a \
                     non-negative integer, got {v}"
                ))),
            },
        }
    };
    let boolean = |key: &str| -> Result<Option<bool>> {
        match obj.get(key) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Bool(b)) => Ok(Some(*b)),
            Some(other) => Err(TensorError::new(&format!(
                "cluster launcher: controller.join.{key} must be a \
                 boolean, got {other}"
            ))),
        }
    };
    let token = match obj.get("token") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: controller.join.token must be a hex \
                 string, got {other}"
            )));
        }
    };
    let start = match obj.get("start") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(match s.as_str() {
            "auto" => crate::distributed::membership::StartMode::Auto,
            "manual" => crate::distributed::membership::StartMode::Manual,
            "hybrid" => crate::distributed::membership::StartMode::Hybrid,
            other => {
                return Err(TensorError::new(&format!(
                    "cluster launcher: controller.join.start must be one of \
                     \"auto\" | \"manual\" | \"hybrid\", got {other:?}"
                )));
            }
        }),
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: controller.join.start must be a string \
                 (\"auto\" | \"manual\" | \"hybrid\"), got {other}"
            )));
        }
    };
    // Unknown keys are a config typo waiting to silently no-op a knob.
    const KNOWN: [&str; 9] = [
        "min_rank_start",
        "join_timeout",
        "target_ranks",
        "max_join_timeout",
        "open_admission",
        "discovery",
        "token",
        "tunnel_only",
        "start",
    ];
    for k in obj.keys() {
        if !KNOWN.contains(&k.as_str()) {
            return Err(TensorError::new(&format!(
                "cluster launcher: controller.join.{k}: unknown field \
                 (expected one of {KNOWN:?})"
            )));
        }
    }
    Ok(Some(JoinKnobs {
        min_rank_start: uint("min_rank_start")?.map(|n| n as usize),
        join_timeout_secs: uint("join_timeout")?,
        target_ranks: uint("target_ranks")?.map(|n| n as usize),
        max_join_timeout_secs: uint("max_join_timeout")?,
        open_admission: boolean("open_admission")?,
        discovery: boolean("discovery")?,
        token,
        tunnel_only: boolean("tunnel_only")?,
        start,
    }))
}

/// Parse an `ssh:` sub-block. Expects a JSON object with optional
/// `target`, `port`, `user`, `identity_file`, and `options` fields;
/// missing/null produces `None` (meaning "no SSH overrides, fall back
/// to host name + system ssh defaults"). Loud errors on type
/// mismatches per field so typos surface immediately rather than
/// silently dropping a value.
fn parse_ssh_block(v: Option<&serde_json::Value>, label: &str) -> Result<Option<SshConfig>> {
    let obj = match v {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::Object(m)) => m,
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: {label}.ssh must be a map (target, port, \
                 user, identity_file, options), got {other}"
            )));
        }
    };

    let target = obj.get("target").and_then(|v| v.as_str()).map(String::from);

    let port = match obj.get("port") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| {
                TensorError::new(&format!(
                    "cluster launcher: {label}.ssh.port must be integer"
                ))
            })?;
            Some(u16::try_from(n).map_err(|_| {
                TensorError::new(&format!(
                    "cluster launcher: {label}.ssh.port {n} does not fit in u16"
                ))
            })?)
        }
    };

    let user = obj.get("user").and_then(|v| v.as_str()).map(String::from);

    let identity_file = obj
        .get("identity_file")
        .and_then(|v| v.as_str())
        .map(String::from);

    let options: Vec<String> = match obj.get("options") {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .enumerate()
            .map(|(j, e)| {
                e.as_str().map(String::from).ok_or_else(|| {
                    TensorError::new(&format!(
                        "cluster launcher: {label}.ssh.options[{j}]: must be string"
                    ))
                })
            })
            .collect::<Result<_>>()?,
        Some(other) => {
            return Err(TensorError::new(&format!(
                "cluster launcher: {label}.ssh.options must be array of strings, got {other}"
            )));
        }
    };

    Ok(Some(SshConfig {
        target,
        port,
        user,
        identity_file,
        options,
    }))
}

/// Parse an `env:` block from either a launcher-level or host-level
/// position. Expects a JSON object whose values are all strings
/// (`{"NAME": "value", ...}`); missing/null produces an empty map.
/// Loud-errors on anything else so a typo can't silently produce an
/// empty env that hides a real config error.
fn parse_env_block(
    v: Option<&serde_json::Value>,
    label: &str,
) -> Result<std::collections::BTreeMap<String, String>> {
    use std::collections::BTreeMap;
    match v {
        None | Some(serde_json::Value::Null) => Ok(BTreeMap::new()),
        Some(serde_json::Value::Object(map)) => {
            let mut out = BTreeMap::new();
            for (k, val) in map {
                // Reserved keys: the launcher owns rank identity. User env
                // is applied after the launcher's built-ins on both spawn
                // mediums (Command::env re-insertion locally, last shell
                // assignment remotely) and last-write-wins, so a reserved
                // key here would silently clobber rank↔device identity —
                // e.g. `CUDA_VISIBLE_DEVICES: "0"` pins every local rank
                // to GPU 0 (NCCL duplicate-device failure, or silently
                // permuted ranks). Rejected loudly at parse, never
                // filtered at spawn. The rule is the canonical
                // `is_reserved_cluster_env_key` (mirrored in fdl-cli):
                // `FLODL_INTERNAL_*` + the launcher-owned specials.
                // User-facing `FLODL_*` knobs (`FLODL_VERBOSITY`,
                // `FLODL_STAGER`, NCCL tuning) are deliberately allowed.
                if crate::distributed::cluster::is_reserved_cluster_env_key(k) {
                    return Err(TensorError::new(&format!(
                        "cluster launcher: {label}[{k:?}] is reserved \
                         (launcher-owned rank identity). GPU scoping belongs \
                         in `local_devices:`; FLODL_INTERNAL_* vars are set \
                         by the launcher itself."
                    )));
                }
                // Keys are interpolated unquoted into the remote shell's
                // K=V assignment prefix — restrict to the portable
                // identifier charset so a stray space or metacharacter
                // cannot break (or inject into) the remote command line.
                let valid_key = !k.is_empty()
                    && k.chars()
                        .next()
                        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                    && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                if !valid_key {
                    return Err(TensorError::new(&format!(
                        "cluster launcher: {label}[{k:?}] is not a valid env var \
                         name ([A-Za-z_][A-Za-z0-9_]*)"
                    )));
                }
                let s = val.as_str().ok_or_else(|| {
                    TensorError::new(&format!(
                        "cluster launcher: {label}[{k:?}] must be a string, got {val}"
                    ))
                })?;
                out.insert(k.clone(), s.to_string());
            }
            Ok(out)
        }
        Some(other) => Err(TensorError::new(&format!(
            "cluster launcher: {label} must be an object (NAME → string VALUE), \
             got {other}"
        ))),
    }
}
