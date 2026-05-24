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

/// Fluent builder for [`FullCluster`]. Compose hosts via
/// [`Self::host`] (which returns a [`HostBuilder`]) and finalize with
/// [`Self::build`].
///
/// ```ignore
/// let cluster = ClusterBuilder::new("192.168.122.1")
///     .controller_port(29500)
///     .host("exa")
///         .ranks([0])
///         .devices([0])
///         .nccl_socket_ifname("virbr0")
///         .path("/home/fab2s/src/fab2s/ai/rdl")
///     .done()
///     .host("flodl-pascal")
///         .ranks([1, 2])
///         .all_devices()
///         .nccl_socket_ifname("enp1s0")
///         .path("/mnt/rdl")
///         .ssh_port(22)
///         .ssh_identity_file("/home/fab2s/.ssh/cluster_key")
///     .done()
///     .build()?;
/// ```
pub struct ClusterBuilder {
    controller: super::launcher::FullController,
    workers: Vec<FullWorker>,
}

impl ClusterBuilder {
    /// Begin construction with the controller's rendezvous bind host.
    /// Accepts hostname or IP — DNS resolution happens at TCP-connect
    /// time. Use `"localhost"` (or `"127.0.0.1"`) when every worker is
    /// local. `controller.port` defaults to 1337 (matches
    /// `flodl-cli`'s `DEFAULT_CONTROLLER_PORT`); override with
    /// [`Self::controller_port`]. `controller.path` defaults to the
    /// current working directory; override with
    /// [`Self::controller_path`].
    pub fn new(controller_host: impl Into<String>) -> Self {
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_default();
        Self {
            controller: super::launcher::FullController {
                host: controller_host.into(),
                port: 1337,
                path: cwd,
                nccl_socket_ifname: None,
                docker: None,
                arch: None,
                data_path: None,
            },
            workers: Vec::new(),
        }
    }

    /// Override the controller's rendezvous port. Default 1337.
    pub fn controller_port(mut self, port: u16) -> Self {
        self.controller.port = port;
        self
    }

    /// Set the controller's project-root path (defaults to current
    /// working directory at builder construction time). Override when
    /// the controller's view of the shared project root differs from
    /// each worker's view (heterogeneous-mount rigs).
    pub fn controller_path(mut self, path: impl Into<String>) -> Self {
        self.controller.path = path.into();
        self
    }

    /// Set the network interface NCCL binds to on the controller side.
    /// Required when more than one worker is declared.
    pub fn controller_nccl_socket_ifname(
        mut self,
        ifname: impl Into<String>,
    ) -> Self {
        self.controller.nccl_socket_ifname = Some(ifname.into());
        self
    }

    /// Start configuring a new worker. Returns a [`HostBuilder`]; call
    /// [`HostBuilder::done`] to finalize and return to the cluster
    /// builder. (Method name `host` is kept for source-compat; the
    /// argument names the worker.)
    pub fn host(self, name: impl Into<String>) -> HostBuilder {
        HostBuilder::new(self, name.into())
    }

    /// Finalize. Validates that ranks across workers form
    /// `0..world_size` with no duplicates or gaps; validates non-empty
    /// controller.host / controller.path, non-empty workers list,
    /// non-empty per-worker fields.
    pub fn build(self) -> Result<FullCluster> {
        if self.controller.host.trim().is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: controller.host must be non-empty",
            ));
        }
        if self.controller.path.trim().is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: controller.path must be non-empty",
            ));
        }
        if self.workers.is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: at least one worker required",
            ));
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
            env: std::collections::BTreeMap::new(),
        })
    }

    /// Synthesize a single-host cluster from [`crate::sys::detect_gpus`].
    ///
    /// The ergonomic one-liner for the common "single machine,
    /// multi-GPU" case. Reads visible GPUs via `nvidia-smi` (filtered
    /// by `CUDA_VISIBLE_DEVICES`), builds a one-host topology with
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
                "ClusterBuilder::all_local_gpus: no CUDA GPUs visible \
                 (nvidia-smi reported none, or CUDA_VISIBLE_DEVICES \
                 narrowed to empty). Use the single-device path instead.",
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
                nccl_socket_ifname: None,
                docker: None,
                arch: None,
                data_path: None,
            },
            workers: vec![FullWorker {
                host: hostname,
                ranks,
                local_devices: Some(local_devices),
                nccl_socket_ifname: "lo".to_string(),
                path: cwd,
                arch: None,
                ssh: None,
                env: std::collections::BTreeMap::new(),
            }],
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
            env: std::collections::BTreeMap::new(),
        })
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
    ssh: Option<crate::distributed::launcher::SshConfig>,
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
            ssh: None,
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

    /// Finalize this host and return to the cluster builder.
    ///
    /// # Panics
    ///
    /// Panics if required fields (`ranks`, `local_devices`,
    /// `nccl_socket_ifname`, `path`) were not set. Validate via
    /// [`ClusterBuilder::build`].
    pub fn done(self) -> ClusterBuilder {
        let host = FullWorker {
            host: self.name,
            ranks: self.ranks.expect("HostBuilder: ranks(...) required"),
            local_devices: self
                .local_devices
                .expect("HostBuilder: devices(...) or all_devices() required"),
            nccl_socket_ifname: self
                .nccl_socket_ifname
                .expect("HostBuilder: nccl_socket_ifname(...) required"),
            path: self.path.expect("HostBuilder: path(...) required"),
            arch: self.arch,
            ssh: self.ssh,
            env: std::collections::BTreeMap::new(),
        };
        let mut parent = self.parent;
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
        let cluster = ClusterBuilder::new("192.168.122.1")
            .controller_port(29500)
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
        let err = ClusterBuilder::new("localhost")
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
    fn build_rejects_empty_controller_host() {
        let err = ClusterBuilder::new("")
            .host("h0")
                .ranks([0])
                .devices([0])
                .nccl_socket_ifname("lo")
                .path("/tmp")
            .done()
            .build()
            .expect_err("empty controller.host must error");
        assert!(err.to_string().contains("controller.host"), "err: {err}");
    }

    #[test]
    fn build_rejects_no_workers() {
        let err = ClusterBuilder::new("localhost")
            .build()
            .expect_err("no workers must error");
        assert!(err.to_string().contains("worker"), "err: {err}");
    }
}
