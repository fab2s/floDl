//! Programmatic construction of [`FullCluster`] / [`FullWorker`].
//!
//! Mirrors the yml schema 1:1 — same fields, same validation, same
//! launcher consumption. Two construction paths exist (yml via
//! `flodl-cli` overlay parsing, or this typed builder); they produce
//! the SAME [`FullCluster`] shape so the launcher doesn't care which
//! was used. Useful for:
//!
//! - Single-host multi-GPU runs that don't want to check in an
//!   `fdl.cluster.yml` (see [`ClusterBuilder::all_local_gpus`]).
//! - Tests that exercise `cluster_coordinator` without a yml file
//!   (paired with the `cargo test`-aware fork+exec strategy).
//! - Dynamic cluster shapes (job-scheduler-discovered hosts,
//!   ephemeral worker pools) where serializing a yml first would be
//!   round-trip overhead.
//!
//! [`FullCluster`]: super::launcher::FullCluster
//! [`FullWorker`]: super::launcher::FullWorker

use crate::tensor::{Result, TensorError};

use super::launcher::{FullCluster, FullWorker};

// ---------------------------------------------------------------------------
// FullCluster builder
// ---------------------------------------------------------------------------

/// Fluent builder for [`FullCluster`]. Compose the controller via
/// [`Self::controller`] (returns a [`ControllerBuilder`]) and each
/// worker via [`Self::host`] (returns a [`HostBuilder`]), finalize
/// with [`Self::build`]. Mirrors the YAML schema's sibling
/// `controller:` / `workers[]:` blocks.
///
/// ```ignore
/// let cluster = ClusterBuilder::new()
///     .controller("192.168.122.1")
///         .port(1337)
///         .path("/opt/flodl")
///     .done()
///     .host("node-a")
///         .ranks([0])
///         .devices([0])
///         .nccl_socket_ifname("virbr0")
///         .path("/opt/flodl")
///     .done()
///     .host("node-b")
///         .ranks([1, 2])
///         .all_devices()
///         .nccl_socket_ifname("enp1s0")
///         .path("/srv/flodl")
///         .ssh_port(22)
///         .ssh_identity_file("/home/ubuntu/.ssh/cluster_key")
///     .done()
///     .build()?;
/// ```
pub struct ClusterBuilder {
    controller: super::launcher::FullController,
    workers: Vec<FullWorker>,
    /// Cluster-scope env vars exported into every rank child on every
    /// host. Mirrors the YAML cluster-scope `env:` block. Per-host
    /// [`HostBuilder::env`] overrides matching keys.
    env: std::collections::BTreeMap<String, String>,
    /// Cluster-scope default for the integrated-GPU host-RAM share.
    /// Mirrors the YAML cluster-scope `gpu_ram_share:` field; per-host
    /// [`HostBuilder::gpu_ram_share`] overrides it.
    gpu_ram_share: Option<f64>,
    /// Host-finalization errors deferred from [`HostBuilder::done`] so the
    /// fluent chain stays infallible; surfaced as `Err` by
    /// [`ClusterBuilder::build`]. Required-field mistakes are user-input
    /// errors, not programmer invariants — they must not panic.
    deferred_errors: Vec<String>,
}

impl Default for ClusterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterBuilder {
    /// Begin construction. [`Self::controller`] must be called before
    /// [`Self::build`] — `build` errors loudly if the controller host
    /// is empty.
    pub fn new() -> Self {
        Self {
            controller: super::launcher::FullController {
                host: String::new(),
                port: 1337,
                path: String::new(),
                docker: None,
                arch: None,
                data_path: None,
                join: None,
            },
            workers: Vec::new(),
            env: std::collections::BTreeMap::new(),
            gpu_ram_share: None,
            deferred_errors: Vec::new(),
        }
    }

    /// Cluster-scope default for the integrated-GPU host-RAM share: the
    /// fraction of `MemTotal` an APU host's GPU aperture claims (same
    /// knob as `DataLoaderBuilder::gpu_ram_share`). Discrete-GPU hosts
    /// ignore it, which is what makes a fleet-wide default legal on a
    /// mixed fleet. Per-host [`HostBuilder::gpu_ram_share`] overrides
    /// it, and a walk-in's own `join.gpu_ram_share:` overrides both.
    /// Non-negative (above 1.0 is legal where `MemTotal` under-states
    /// the aperture) — validated at [`Self::build`].
    pub fn gpu_ram_share(mut self, share: f64) -> Self {
        self.gpu_ram_share = Some(share);
        self
    }

    /// Start configuring the controller. `host` is the rendezvous bind
    /// hostname or IP — DNS resolution happens at TCP-connect time. Use
    /// `"localhost"` (or `"127.0.0.1"`) when every worker is local.
    /// Returns a [`ControllerBuilder`]; call
    /// [`ControllerBuilder::done`] to finalize and return to the
    /// cluster builder.
    pub fn controller(self, host: impl Into<String>) -> ControllerBuilder {
        ControllerBuilder::new(self, host.into())
    }

    /// Start configuring a new worker. Returns a [`HostBuilder`]; call
    /// [`HostBuilder::done`] to finalize and return to the cluster
    /// builder. The `name` argument identifies the worker.
    pub fn host(self, name: impl Into<String>) -> HostBuilder {
        HostBuilder::new(self, name.into())
    }

    /// Set a cluster-scope env var exported into every rank child on
    /// every host. Mirrors the YAML cluster-scope `env:` block; the
    /// common use is NCCL transport overrides on rigs where the default
    /// transports fail (e.g. `.env("NCCL_P2P_DISABLE", "1")` +
    /// `.env("NCCL_SHM_DISABLE", "1")` on a PCIe-passthrough VM). Call
    /// repeatedly to accumulate; a later call with the same key wins.
    /// Per-host [`HostBuilder::env`] overrides matching keys.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Finalize. Validates that ranks across workers form
    /// `0..world_size` with no duplicates or gaps; validates non-empty
    /// controller.host / controller.path, non-empty workers list,
    /// non-empty per-worker fields.
    pub fn build(self) -> Result<FullCluster> {
        if !self.deferred_errors.is_empty() {
            return Err(TensorError::new(&format!(
                "ClusterBuilder: incomplete host definitions — {}",
                self.deferred_errors.join("; "),
            )));
        }
        if self.controller.host.trim().is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: controller(...).done() must be called \
                 with a non-empty host before build()",
            ));
        }
        if self.controller.path.trim().is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: controller.path must be non-empty",
            ));
        }
        let discovery = self
            .controller
            .join
            .as_ref()
            .is_some_and(|j| j.discovery == Some(true));
        if self.workers.is_empty() && !discovery {
            return Err(TensorError::new(
                "ClusterBuilder: at least one worker required (a roster-free \
                 window needs `.discovery(true)` on the controller)",
            ));
        }
        // Explicit-devices length check: a host with explicit device
        // indices must supply exactly one per rank (paired by position).
        // Mirrors the YAML validator (config/cluster.rs) so the builder
        // stays 1:1 with the file schema, and honors the promise in
        // `HostBuilder::devices`'s doc that the length is checked here.
        // `all_devices()` (unresolved "all") is exempt — its count is
        // resolved at startup on the owning host.
        for w in &self.workers {
            if let Some(devs) = w.local_devices.as_deref()
                && devs.len() != w.ranks.len()
            {
                return Err(TensorError::new(&format!(
                    "ClusterBuilder: host {:?}: devices ({}) and ranks ({}) \
                         length mismatch — supply exactly one device index per \
                         rank, or use all_devices()",
                    w.host,
                    devs.len(),
                    w.ranks.len(),
                )));
            }
        }
        // Reserved-env-key check: a user env map (cluster- or host-scope)
        // must not carry a key the launcher owns per-rank — the launcher
        // applies user env after its own built-ins (shell last-wins), so
        // an override would silently break device mapping / rank identity /
        // the HMAC envelope. Mirrors the YAML validator so the builder
        // stays 1:1 with the file schema. See
        // [`crate::distributed::is_reserved_cluster_env_key`].
        for k in self.env.keys() {
            if crate::distributed::is_reserved_cluster_env_key(k) {
                return Err(TensorError::new(&format!(
                    "ClusterBuilder: cluster-scope env key {k:?} is reserved \
                     (launcher-owned) and cannot be set via env — it would \
                     override the launcher's per-rank value"
                )));
            }
        }
        for w in &self.workers {
            for k in w.env.keys() {
                if crate::distributed::is_reserved_cluster_env_key(k) {
                    return Err(TensorError::new(&format!(
                        "ClusterBuilder: host {:?}: env key {k:?} is reserved \
                         (launcher-owned) and cannot be set via env — it would \
                         override the launcher's per-rank value",
                        w.host,
                    )));
                }
            }
        }
        // Fraction-of-RAM checks, one rule with the yml parser's:
        // non-negative and finite, loud otherwise. Above 1.0 stays
        // legal on purpose — the knob exists partly for platforms whose
        // MemTotal under-states what the APU can address.
        let share_ok = |s: Option<f64>| s.is_none_or(|f| f.is_finite() && f >= 0.0);
        if !share_ok(self.gpu_ram_share) {
            return Err(TensorError::new(&format!(
                "ClusterBuilder: gpu_ram_share must be a non-negative \
                 fraction of host RAM (e.g. 0.5), got {:?}",
                self.gpu_ram_share,
            )));
        }
        if let Some(w) = self.workers.iter().find(|w| !share_ok(w.gpu_ram_share)) {
            return Err(TensorError::new(&format!(
                "ClusterBuilder: host {:?}: gpu_ram_share must be a \
                 non-negative fraction of host RAM (e.g. 0.5), got {:?}",
                w.host, w.gpu_ram_share,
            )));
        }
        // Cross-worker rank check: union must be exactly 0..world_size.
        let mut all: Vec<usize> = self
            .workers
            .iter()
            .flat_map(|w| w.ranks.iter().copied())
            .collect();
        let ws = all.len();
        all.sort_unstable();
        let expected: Vec<usize> = (0..ws).collect();
        if all != expected {
            return Err(TensorError::new(&format!(
                "ClusterBuilder: ranks across workers must form 0..{ws} with no \
                 duplicates or gaps, got sorted-unique sequence {all:?}"
            )));
        }
        Ok(FullCluster {
            controller: self.controller,
            workers: self.workers,
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
            env: self.env,
            gpu_ram_share: self.gpu_ram_share,
        })
    }

    /// Synthesize a single-host cluster from [`crate::sys::detect_gpus`].
    ///
    /// The ergonomic one-liner for the common "single machine,
    /// multi-GPU" case. Reads visible GPUs through the build's own vendor
    /// probe (`nvidia-smi`, or the KFD topology on ROCm), filtered by
    /// that vendor's visibility mask, builds a one-host topology with
    /// `world_size = num_visible_gpus`, defaults master to `localhost`,
    /// nccl_socket_ifname to `lo`.
    ///
    /// Errors when no GPUs are visible — caller should fall back to
    /// the single-device path in that case.
    ///
    /// ```ignore
    /// let cfg = TrainerConfig::new(load_data()?)
    ///     .cluster(ClusterBuilder::all_local_gpus()?)
    ///     .elche(ElCheConfig::nccl_cadence());
    /// Trainer::run(model_factory, opt_factory, train_fn, cfg)?;
    /// ```
    pub fn all_local_gpus() -> Result<FullCluster> {
        let gpus = crate::sys::detect_gpus();
        if gpus.is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder::all_local_gpus: no usable GPU visible for this \
                 build's vendor (none detected, or a visibility mask \
                 narrowed the list to empty -- CUDA_VISIBLE_DEVICES, and \
                 HIP/ROCR_VISIBLE_DEVICES on ROCm). Use the single-device \
                 path instead.",
            ));
        }
        let hostname = crate::distributed::cluster::resolve_hostname()?;
        let n = gpus.len();
        let ranks: Vec<usize> = (0..n).collect();
        let local_devices: Vec<u8> = gpus.iter().map(|g| g.index).collect();
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default();
        Ok(FullCluster {
            controller: super::launcher::FullController {
                // "127.0.0.1" rather than "localhost": Rust's
                // `SocketAddr::from_str` requires a numeric IP —
                // passing the hostname string downstream fails the
                // coord-addr parse in orchestrator with "invalid
                // socket address syntax".
                host: "127.0.0.1".to_string(),
                port: 1337,
                path: cwd.clone(),
                docker: None,
                arch: None,
                data_path: None,
                join: None,
            },
            workers: vec![FullWorker {
                host: hostname,
                ranks,
                local_devices: Some(local_devices),
                nccl_socket_ifname: "lo".to_string(),
                path: cwd,
                arch: None,
                data_path: None,
                gpu_ram_share: None,
                ssh: None,
                tunnel: false,
                env: std::collections::BTreeMap::new(),
            }],
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
            env: std::collections::BTreeMap::new(),
            gpu_ram_share: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Controller builder
// ---------------------------------------------------------------------------

/// Fluent builder for the cluster's controller block. Borrows the
/// parent [`ClusterBuilder`] so [`Self::done`] returns to it for
/// chaining `host(...)` calls. Mirrors the YAML `controller:` block.
pub struct ControllerBuilder {
    parent: ClusterBuilder,
    host: String,
    port: u16,
    path: Option<String>,
    docker: Option<String>,
    arch: Option<String>,
    data_path: Option<String>,
    join: super::launcher::JoinKnobs,
}

impl ControllerBuilder {
    fn new(parent: ClusterBuilder, host: String) -> Self {
        Self {
            parent,
            host,
            port: 1337,
            path: None,
            docker: None,
            arch: None,
            data_path: None,
            join: super::launcher::JoinKnobs::default(),
        }
    }

    /// Controller rendezvous port. Default 1337 (matches
    /// `flodl-cli`'s `DEFAULT_CONTROLLER_PORT`).
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Controller's view of the shared project root. Defaults to the
    /// current working directory at [`Self::done`] time. Override when
    /// the controller's mount path differs from each worker's
    /// (heterogeneous-mount rigs).
    pub fn path(mut self, p: impl Into<String>) -> Self {
        self.path = Some(p.into());
        self
    }

    /// Optional pre-flight build context — the docker-compose service
    /// name fdl-cli should use when building the rank binary for this
    /// cluster (mirrors the YAML `controller.docker` field).
    pub fn docker(mut self, d: impl Into<String>) -> Self {
        self.docker = Some(d.into());
        self
    }

    /// Optional libtorch variant subpath for the controller's
    /// pre-flight build (e.g. `"precompiled/cu128"`).
    pub fn arch(mut self, a: impl Into<String>) -> Self {
        self.arch = Some(a.into());
        self
    }

    /// Optional shared-data path the controller side of the launcher
    /// should resolve. Mirrors the YAML `controller.data_path` field.
    pub fn data_path(mut self, p: impl Into<String>) -> Self {
        self.data_path = Some(p.into());
        self
    }

    /// Join-window quorum: the run cannot start below this many ranks.
    /// Mirrors YAML `controller.join.min_rank_start`. Default: the
    /// configured capacity (all-or-nothing).
    pub fn min_rank_start(mut self, ranks: usize) -> Self {
        self.join.min_rank_start = Some(ranks);
        self
    }

    /// Join window in seconds; quorum reached early does NOT close it.
    /// Mirrors YAML `controller.join.join_timeout`. Default 300.
    pub fn join_timeout_secs(mut self, secs: u64) -> Self {
        self.join.join_timeout_secs = Some(secs);
        self
    }

    /// Early-close target in ranks: the window closes the moment this
    /// many are in. Mirrors YAML `controller.join.target_ranks`.
    /// Default: the configured capacity; set it higher (with a matching
    /// window) to leave room for self-deployed workers to dial in
    /// alongside the managed rig.
    pub fn target_ranks(mut self, ranks: usize) -> Self {
        self.join.target_ranks = Some(ranks);
        self
    }

    /// Hard cap in seconds: quorum still unmet when it expires fails
    /// the run loudly. Mirrors YAML `controller.join.max_join_timeout`.
    /// Default 600 (or the window length when that is set higher).
    pub fn max_join_timeout_secs(mut self, secs: u64) -> Self {
        self.join.max_join_timeout_secs = Some(secs);
        self
    }

    /// Accept joins without pre-shared-salt authentication on a
    /// non-loopback bind (loudly warned — any peer that can reach the
    /// port can then join, and therefore influence, the run). Mirrors
    /// YAML `controller.join.open_admission`.
    pub fn open_admission(mut self, open: bool) -> Self {
        self.join.open_admission = Some(open);
        self
    }

    /// Roster-free formation: the join window alone defines the world
    /// (the `workers:` list may be empty; walk-ins self-register).
    /// Requires an explicit [`Self::min_rank_start`]. Mirrors YAML
    /// `controller.join.discovery`.
    pub fn discovery(mut self, on: bool) -> Self {
        self.join.discovery = Some(on);
        self
    }

    /// Pre-shared session salt, hex (32 chars / 16 bytes), replacing the
    /// per-run generated salt so walk-ins can present it as their join
    /// credential. Forces authenticated admission even on a loopback
    /// bind. Mirrors YAML `controller.join.token`.
    pub fn join_token(mut self, token_hex: impl Into<String>) -> Self {
        self.join.token = Some(token_hex.into());
        self
    }

    /// Bind the controller mux loopback-only in discovery mode: every
    /// walk-in must arrive through an sshd-carried forward, making
    /// reachability itself the authentication. Requires a CPU averaging
    /// mode. Mirrors YAML `controller.join.tunnel_only`.
    pub fn tunnel_only(mut self, on: bool) -> Self {
        self.join.tunnel_only = Some(on);
        self
    }

    /// Who closes the join window once quorum is met: the clock
    /// ([`StartMode::Auto`], default), the operator via `fdl start`
    /// ([`StartMode::Manual`]), or either ([`StartMode::Hybrid`]).
    /// Mirrors YAML `controller.join.start`.
    ///
    /// [`StartMode::Auto`]: crate::distributed::StartMode::Auto
    /// [`StartMode::Manual`]: crate::distributed::StartMode::Manual
    /// [`StartMode::Hybrid`]: crate::distributed::StartMode::Hybrid
    pub fn start_mode(mut self, mode: crate::distributed::StartMode) -> Self {
        self.join.start = Some(mode);
        self
    }

    /// Finalize the controller and return to the cluster builder.
    pub fn done(self) -> ClusterBuilder {
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default();
        let mut parent = self.parent;
        let join = if self.join == super::launcher::JoinKnobs::default() {
            None
        } else {
            Some(self.join)
        };
        parent.controller = super::launcher::FullController {
            host: self.host,
            port: self.port,
            path: self.path.unwrap_or(cwd),
            docker: self.docker,
            arch: self.arch,
            data_path: self.data_path,
            join,
        };
        parent
    }
}

// ---------------------------------------------------------------------------
// FullWorker builder
// ---------------------------------------------------------------------------

/// Fluent builder for one [`FullWorker`]. Borrows the parent
/// [`ClusterBuilder`] so [`Self::done`] can return to chaining
/// additional hosts.
pub struct HostBuilder {
    parent: ClusterBuilder,
    name: String,
    ranks: Option<Vec<usize>>,
    local_devices: Option<Option<Vec<u8>>>, // outer None=unset, inner None="all"
    nccl_socket_ifname: Option<String>,
    path: Option<String>,
    arch: Option<String>,
    data_path: Option<String>,
    gpu_ram_share: Option<f64>,
    ssh: Option<crate::distributed::launcher::SshConfig>,
    tunnel: bool,
    env: std::collections::BTreeMap<String, String>,
}

impl HostBuilder {
    fn new(parent: ClusterBuilder, name: String) -> Self {
        Self {
            parent,
            name,
            ranks: None,
            local_devices: None,
            nccl_socket_ifname: None,
            path: None,
            arch: None,
            data_path: None,
            gpu_ram_share: None,
            ssh: None,
            tunnel: false,
            env: std::collections::BTreeMap::new(),
        }
    }

    /// Lazily materialize the inner `SshConfig` so the per-field
    /// setters can mutate it without `Option`-juggling at each call
    /// site. Calling any of the `ssh_*` methods promotes the worker
    /// to "remote with overrides" even if no other field is set.
    fn ssh_mut(&mut self) -> &mut crate::distributed::launcher::SshConfig {
        self.ssh
            .get_or_insert_with(crate::distributed::launcher::SshConfig::default)
    }

    /// Global rank indices owned by this host.
    pub fn ranks<I: IntoIterator<Item = usize>>(mut self, ranks: I) -> Self {
        self.ranks = Some(ranks.into_iter().collect());
        self
    }

    /// Explicit CUDA device indices for this host's ranks, paired by
    /// position with `ranks`. Length must match `ranks` at
    /// [`ClusterBuilder::build`] time.
    pub fn devices<I: IntoIterator<Item = u8>>(mut self, devices: I) -> Self {
        self.local_devices = Some(Some(devices.into_iter().collect()));
        self
    }

    /// Use ALL visible CUDA devices on this host (resolved at startup
    /// on the host that owns this entry). Equivalent to the yml
    /// shorthand `local_devices: "all"`.
    pub fn all_devices(mut self) -> Self {
        self.local_devices = Some(None);
        self
    }

    /// Network interface NCCL binds to (e.g. `"virbr0"`, `"enp1s0"`,
    /// `"lo"` for single-host loopback).
    pub fn nccl_socket_ifname(mut self, name: impl Into<String>) -> Self {
        self.nccl_socket_ifname = Some(name.into());
        self
    }

    /// Project checkout path on this host. Launcher cd's here before
    /// invoking the remote command.
    pub fn path(mut self, p: impl Into<String>) -> Self {
        self.path = Some(p.into());
        self
    }

    /// libtorch variant subpath under `<path>/libtorch/` on this host
    /// (e.g. `"precompiled/cu128"`, `"builds/sm61-sm120"`). Convention
    /// resolves the runtime libtorch at `<path>/libtorch/<arch>/`.
    pub fn arch(mut self, p: impl Into<String>) -> Self {
        self.arch = Some(p.into());
        self
    }

    /// Dataset source root on this host: where its ranks READ training
    /// data from. A shared mount or a node-local directory, whichever
    /// the deployment provides. Mirrors the YAML field
    /// `workers[].data_path`.
    ///
    /// Left unset, nothing travels to the rank and the training binary
    /// keeps its own default.
    pub fn data_path(mut self, p: impl Into<String>) -> Self {
        self.data_path = Some(p.into());
        self
    }

    /// SSH target (e.g. `"user@host"` or a short alias from
    /// `~/.ssh/config`). Default: the host's `name`. Set to a
    /// different value when the SSH connect string differs from the
    /// cluster name (e.g. `~/.ssh/config` host aliases). Mirrors the
    /// YAML field `ssh.target`.
    pub fn ssh(mut self, target: impl Into<String>) -> Self {
        self.ssh_mut().target = Some(target.into());
        self
    }

    /// SSH port (`-p <port>`). Default: 22. Mirrors `ssh.port`.
    pub fn ssh_port(mut self, port: u16) -> Self {
        self.ssh_mut().port = Some(port);
        self
    }

    /// SSH login user (`-l <user>`). Default: current user. Mirrors
    /// `ssh.user`.
    pub fn ssh_user(mut self, user: impl Into<String>) -> Self {
        self.ssh_mut().user = Some(user.into());
        self
    }

    /// SSH identity file (`-i <path>`). Mirrors `ssh.identity_file`.
    pub fn ssh_identity_file(mut self, path: impl Into<String>) -> Self {
        self.ssh_mut().identity_file = Some(path.into());
        self
    }

    /// Append one `-o Key=Value` SSH option. Call multiple times to
    /// accumulate options in order (e.g. `.ssh_option("ProxyJump=bastion")`,
    /// `.ssh_option("StrictHostKeyChecking=no")`). Mirrors
    /// `ssh.options[]`.
    pub fn ssh_option(mut self, opt: impl Into<String>) -> Self {
        self.ssh_mut().options.push(opt.into());
        self
    }

    /// Route this host's training traffic through its fan-out SSH
    /// session (remote forward) instead of a direct TCP connection to
    /// the controller. Requires a CPU ElChe mode and a remote host —
    /// validated loudly at launch. Mirrors the YAML field `tunnel:`.
    pub fn tunnel(mut self, tunnel: bool) -> Self {
        self.tunnel = tunnel;
        self
    }

    /// Integrated-GPU host-RAM share for THIS host (fraction of
    /// `MemTotal`; same knob as `DataLoaderBuilder::gpu_ram_share`).
    /// Overrides the cluster-scope [`ClusterBuilder::gpu_ram_share`]
    /// default. Mirrors the YAML per-worker `gpu_ram_share:` field.
    /// Non-negative (above 1.0 is legal where `MemTotal` under-states
    /// the aperture) — validated at [`ClusterBuilder::build`].
    pub fn gpu_ram_share(mut self, share: f64) -> Self {
        self.gpu_ram_share = Some(share);
        self
    }

    /// Set a host-scoped env var exported into every rank child spawned
    /// on this host. Mirrors the YAML per-worker `env:` block; overrides
    /// matching keys from the cluster-scope [`ClusterBuilder::env`].
    /// Useful for host-specific tuning (e.g. a different
    /// `NCCL_SOCKET_IFNAME` or a custom `LD_LIBRARY_PATH` for a
    /// non-standard CUDA install). Call repeatedly to accumulate; a
    /// later call with the same key wins.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Finalize this host and return to the cluster builder.
    ///
    /// Missing required fields (`ranks`, `local_devices`,
    /// `nccl_socket_ifname`, `path`) do NOT panic: they are recorded and
    /// surfaced as a single `Err` by [`ClusterBuilder::build`], keeping
    /// the fluent chain infallible while user-input mistakes stay loud.
    pub fn done(self) -> ClusterBuilder {
        let mut parent = self.parent;
        let mut missing: Vec<&str> = Vec::new();
        if self.ranks.is_none() {
            missing.push("ranks(...)");
        }
        if self.local_devices.is_none() {
            missing.push("devices(...) or all_devices()");
        }
        if self.nccl_socket_ifname.is_none() {
            missing.push("nccl_socket_ifname(...)");
        }
        if self.path.is_none() {
            missing.push("path(...)");
        }
        if !missing.is_empty() {
            parent.deferred_errors.push(format!(
                "host '{}': missing {}",
                self.name,
                missing.join(", "),
            ));
            return parent;
        }
        let host = FullWorker {
            host: self.name,
            ranks: self.ranks.expect("checked above"),
            local_devices: self.local_devices.expect("checked above"),
            nccl_socket_ifname: self.nccl_socket_ifname.expect("checked above"),
            path: self.path.expect("checked above"),
            arch: self.arch,
            data_path: self.data_path,
            gpu_ram_share: self.gpu_ram_share,
            ssh: self.ssh,
            tunnel: self.tunnel,
            env: self.env,
        };
        parent.workers.push(host);
        parent
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_two_host_cluster() {
        let cluster = ClusterBuilder::new()
            .controller("192.168.122.1")
            .port(29500)
            .done()
            .host("exa")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("virbr0")
            .path("/opt/flodl")
            .done()
            .host("flodl-pascal")
            .ranks([1, 2])
            .all_devices()
            .nccl_socket_ifname("enp1s0")
            .path("/mnt/rdl")
            .ssh_port(2222)
            .ssh_identity_file("/keys/cluster")
            .ssh_option("StrictHostKeyChecking=no")
            .done()
            .build()
            .expect("build succeeds");

        assert_eq!(cluster.controller.host, "192.168.122.1");
        assert_eq!(cluster.controller.port, 29500);
        assert_eq!(cluster.workers.len(), 2);
        assert_eq!(cluster.world_size(), 3);

        let exa = &cluster.workers[0];
        assert_eq!(exa.host, "exa");
        assert_eq!(exa.local_devices.as_deref(), Some(&[0u8][..]));

        let pascal = &cluster.workers[1];
        assert!(pascal.local_devices.is_none(), "all_devices() → None");
        let ssh = pascal.ssh.as_ref().expect("ssh fields set the sub-block");
        assert_eq!(ssh.port, Some(2222));
        assert_eq!(ssh.identity_file.as_deref(), Some("/keys/cluster"));
        assert_eq!(ssh.options, vec!["StrictHostKeyChecking=no".to_string()]);
    }

    #[test]
    fn build_rejects_rank_gap() {
        let err = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .host("h0")
            .ranks([0, 2]) // skips 1 → gap
            .devices([0, 1])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect_err("gap must error");
        assert!(err.to_string().contains("gaps"), "err: {err}");
    }

    #[test]
    fn build_rejects_missing_controller() {
        let err = ClusterBuilder::new()
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect_err("missing controller(...) call must error");
        assert!(err.to_string().contains("controller"), "err: {err}");
    }

    #[test]
    fn build_rejects_empty_controller_host() {
        let err = ClusterBuilder::new()
            .controller("")
            .done()
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect_err("empty controller host must error");
        assert!(err.to_string().contains("controller"), "err: {err}");
    }

    #[test]
    fn build_rejects_no_workers() {
        let err = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .build()
            .expect_err("no workers must error");
        assert!(err.to_string().contains("worker"), "err: {err}");
    }

    #[test]
    fn controller_path_defaults_to_cwd_when_unset() {
        let cluster = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect("build succeeds");
        assert!(
            !cluster.controller.path.is_empty(),
            "controller.path falls back to cwd when path() is not called"
        );
    }

    #[test]
    fn controller_path_override() {
        let cluster = ClusterBuilder::new()
            .controller("localhost")
            .port(2222)
            .path("/opt/flodl")
            .docker("cuda")
            .arch("precompiled/cu128")
            .done()
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect("build succeeds");
        assert_eq!(cluster.controller.port, 2222);
        assert_eq!(cluster.controller.path, "/opt/flodl");
        assert_eq!(cluster.controller.docker.as_deref(), Some("cuda"));
        assert_eq!(
            cluster.controller.arch.as_deref(),
            Some("precompiled/cu128")
        );
    }

    #[test]
    fn cluster_scope_env_flows_into_full_cluster() {
        let cluster = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .env("NCCL_P2P_DISABLE", "1")
            .env("NCCL_SHM_DISABLE", "1")
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect("build succeeds");
        assert_eq!(
            cluster.env.get("NCCL_P2P_DISABLE").map(String::as_str),
            Some("1")
        );
        assert_eq!(
            cluster.env.get("NCCL_SHM_DISABLE").map(String::as_str),
            Some("1")
        );
    }

    #[test]
    fn host_scope_env_flows_into_worker() {
        let cluster = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .env("LD_LIBRARY_PATH", "/opt/custom/lib")
            .done()
            .build()
            .expect("build succeeds");
        assert_eq!(
            cluster.workers[0]
                .env
                .get("LD_LIBRARY_PATH")
                .map(String::as_str),
            Some("/opt/custom/lib"),
        );
    }

    #[test]
    fn repeated_env_key_last_wins() {
        let cluster = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .env("K", "first")
            .env("K", "second")
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect("build succeeds");
        assert_eq!(cluster.env.get("K").map(String::as_str), Some("second"));
    }

    #[test]
    fn build_rejects_reserved_cluster_env_key() {
        let err = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .env("CUDA_VISIBLE_DEVICES", "3") // reserved (launcher-owned)
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect_err("reserved cluster-scope env key must error");
        assert!(err.to_string().contains("reserved"), "err: {err}");
    }

    #[test]
    fn build_rejects_reserved_host_env_key() {
        let err = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .env("FLODL_INTERNAL_LOCAL_RANK", "0") // reserved prefix
            .done()
            .build()
            .expect_err("reserved host-scope env key must error");
        assert!(err.to_string().contains("reserved"), "err: {err}");
    }

    #[test]
    fn build_allows_non_reserved_env_keys() {
        ClusterBuilder::new()
            .controller("localhost")
            .done()
            .env("NCCL_P2P_DISABLE", "1")
            .env("FLODL_DASHBOARD_BIND", "0.0.0.0")
            .host("h0")
            .ranks([0])
            .devices([0])
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .env("LD_LIBRARY_PATH", "/opt/custom/lib")
            .done()
            .build()
            .expect("non-reserved env keys must pass");
    }

    #[test]
    fn build_rejects_devices_ranks_length_mismatch() {
        let err = ClusterBuilder::new()
            .controller("localhost")
            .done()
            .host("h0")
            .ranks([0, 1]) // 2 ranks
            .devices([0]) // 1 device → mismatch
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect_err("devices/ranks length mismatch must error");
        assert!(err.to_string().contains("length mismatch"), "err: {err}");
    }

    #[test]
    fn build_allows_all_devices_without_length_check() {
        // all_devices() is unresolved ("all") → exempt from the
        // explicit-devices length check; count is resolved at startup.
        ClusterBuilder::new()
            .controller("localhost")
            .done()
            .host("h0")
            .ranks([0, 1])
            .all_devices()
            .nccl_socket_ifname("lo")
            .path("/tmp")
            .done()
            .build()
            .expect("all_devices() bypasses the explicit-length check");
    }
}
