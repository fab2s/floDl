//! Programmatic construction of [`FullCluster`] / [`FullHost`].
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
//! [`FullHost`]: super::launcher::FullHost

use crate::tensor::{Result, TensorError};

use super::launcher::{FullCluster, FullHost};

// ---------------------------------------------------------------------------
// FullCluster builder
// ---------------------------------------------------------------------------

/// Fluent builder for [`FullCluster`]. Compose hosts via
/// [`Self::host`] (which returns a [`HostBuilder`]) and finalize with
/// [`Self::build`].
///
/// ```ignore
/// let cluster = ClusterBuilder::new("192.168.122.1")
///     .master_port(29500)
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
    master_addr: String,
    master_port: u16,
    hosts: Vec<FullHost>,
}

impl ClusterBuilder {
    /// Begin construction with the rendezvous master address. Accepts
    /// hostname or IP — DNS resolution happens at TCP-connect time. Use
    /// `"localhost"` (or `"127.0.0.1"`) when every host is local.
    /// `master_port` defaults to `29500`; override with
    /// [`Self::master_port`].
    pub fn new(master_addr: impl Into<String>) -> Self {
        Self {
            master_addr: master_addr.into(),
            master_port: 29500,
            hosts: Vec::new(),
        }
    }

    /// Override the rendezvous master port. Default `29500`.
    pub fn master_port(mut self, port: u16) -> Self {
        self.master_port = port;
        self
    }

    /// Start configuring a new host. Returns a [`HostBuilder`]; call
    /// [`HostBuilder::done`] to finalize and return to the cluster
    /// builder.
    pub fn host(self, name: impl Into<String>) -> HostBuilder {
        HostBuilder::new(self, name.into())
    }

    /// Finalize. Validates that ranks across hosts form `0..world_size`
    /// with no duplicates or gaps; validates non-empty master_addr,
    /// non-empty hosts list, non-empty per-host fields.
    pub fn build(self) -> Result<FullCluster> {
        if self.master_addr.trim().is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: master_addr must be non-empty",
            ));
        }
        if self.hosts.is_empty() {
            return Err(TensorError::new(
                "ClusterBuilder: at least one host required",
            ));
        }
        // Cross-host rank check: union must be exactly 0..world_size.
        let mut all: Vec<usize> = self
            .hosts
            .iter()
            .flat_map(|h| h.ranks.iter().copied())
            .collect();
        let ws = all.len();
        all.sort_unstable();
        let expected: Vec<usize> = (0..ws).collect();
        if all != expected {
            return Err(TensorError::new(&format!(
                "ClusterBuilder: ranks across hosts must form 0..{ws} with no \
                 duplicates or gaps, got sorted-unique sequence {all:?}"
            )));
        }
        Ok(FullCluster {
            master_addr: self.master_addr,
            master_port: self.master_port,
            hosts: self.hosts,
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
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
        Ok(FullCluster {
            master_addr: "localhost".to_string(),
            master_port: 29500,
            hosts: vec![FullHost {
                name: hostname,
                ranks,
                local_devices: Some(local_devices),
                nccl_socket_ifname: "lo".to_string(),
                path: std::env::current_dir()
                    .ok()
                    .and_then(|p| p.to_str().map(String::from))
                    .unwrap_or_default(),
                libtorch_path: None,
                ssh: None,
                ssh_port: None,
                ssh_user: None,
                ssh_identity_file: None,
                ssh_options: Vec::new(),
            }],
            salt: [0u8; crate::distributed::wire::SESSION_SALT_BYTES],
        })
    }
}

// ---------------------------------------------------------------------------
// FullHost builder
// ---------------------------------------------------------------------------

/// Fluent builder for one [`FullHost`]. Borrows the parent
/// [`ClusterBuilder`] so [`Self::done`] can return to chaining
/// additional hosts.
pub struct HostBuilder {
    parent: ClusterBuilder,
    name: String,
    ranks: Option<Vec<usize>>,
    local_devices: Option<Option<Vec<u8>>>, // outer None=unset, inner None="all"
    nccl_socket_ifname: Option<String>,
    path: Option<String>,
    libtorch_path: Option<String>,
    ssh: Option<String>,
    ssh_port: Option<u16>,
    ssh_user: Option<String>,
    ssh_identity_file: Option<String>,
    ssh_options: Vec<String>,
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
            libtorch_path: None,
            ssh: None,
            ssh_port: None,
            ssh_user: None,
            ssh_identity_file: None,
            ssh_options: Vec::new(),
        }
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

    /// libtorch install path on this host (bind-mount target for
    /// containerized hosts; informational otherwise).
    pub fn libtorch_path(mut self, p: impl Into<String>) -> Self {
        self.libtorch_path = Some(p.into());
        self
    }

    /// SSH target (e.g. `"user@host"` or a short alias from
    /// `~/.ssh/config`). Default: the host's `name`. Set to a
    /// different value when the SSH connect string differs from the
    /// cluster name (e.g. `~/.ssh/config` host aliases).
    pub fn ssh(mut self, target: impl Into<String>) -> Self {
        self.ssh = Some(target.into());
        self
    }

    /// SSH port (`-p <port>`). Default: 22.
    pub fn ssh_port(mut self, port: u16) -> Self {
        self.ssh_port = Some(port);
        self
    }

    /// SSH login user (`-l <user>`). Default: current user.
    pub fn ssh_user(mut self, user: impl Into<String>) -> Self {
        self.ssh_user = Some(user.into());
        self
    }

    /// SSH identity file (`-i <path>`).
    pub fn ssh_identity_file(mut self, path: impl Into<String>) -> Self {
        self.ssh_identity_file = Some(path.into());
        self
    }

    /// Append one `-o Key=Value` SSH option. Call multiple times to
    /// accumulate options in order (e.g. `.ssh_option("ProxyJump=bastion")`,
    /// `.ssh_option("StrictHostKeyChecking=no")`).
    pub fn ssh_option(mut self, opt: impl Into<String>) -> Self {
        self.ssh_options.push(opt.into());
        self
    }

    /// Finalize this host and return to the cluster builder.
    ///
    /// # Panics
    ///
    /// Panics if required fields (`ranks`, `local_devices`,
    /// `nccl_socket_ifname`, `path`) were not set. Validate via
    /// [`ClusterBuilder::build`].
    pub fn done(mut self) -> ClusterBuilder {
        let host = FullHost {
            name: self.name,
            ranks: self.ranks.expect("HostBuilder: ranks(...) required"),
            local_devices: self
                .local_devices
                .expect("HostBuilder: devices(...) or all_devices() required"),
            nccl_socket_ifname: self
                .nccl_socket_ifname
                .expect("HostBuilder: nccl_socket_ifname(...) required"),
            path: self.path.expect("HostBuilder: path(...) required"),
            libtorch_path: self.libtorch_path,
            ssh: self.ssh,
            ssh_port: self.ssh_port,
            ssh_user: self.ssh_user,
            ssh_identity_file: self.ssh_identity_file,
            ssh_options: std::mem::take(&mut self.ssh_options),
        };
        self.parent.hosts.push(host);
        self.parent
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
            .master_port(29500)
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

        assert_eq!(cluster.master_addr, "192.168.122.1");
        assert_eq!(cluster.master_port, 29500);
        assert_eq!(cluster.hosts.len(), 2);
        assert_eq!(cluster.world_size(), 3);

        let exa = &cluster.hosts[0];
        assert_eq!(exa.name, "exa");
        assert_eq!(exa.local_devices.as_deref(), Some(&[0u8][..]));

        let pascal = &cluster.hosts[1];
        assert!(pascal.local_devices.is_none(), "all_devices() → None");
        assert_eq!(pascal.ssh_port, Some(2222));
        assert_eq!(pascal.ssh_identity_file.as_deref(), Some("/keys/cluster"));
        assert_eq!(pascal.ssh_options, vec!["StrictHostKeyChecking=no".to_string()]);
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
    fn build_rejects_empty_master_addr() {
        let err = ClusterBuilder::new("")
            .host("h0")
                .ranks([0])
                .devices([0])
                .nccl_socket_ifname("lo")
                .path("/tmp")
            .done()
            .build()
            .expect_err("empty master_addr must error");
        assert!(err.to_string().contains("master_addr"), "err: {err}");
    }

    #[test]
    fn build_rejects_no_hosts() {
        let err = ClusterBuilder::new("localhost")
            .build()
            .expect_err("no hosts must error");
        assert!(err.to_string().contains("host"), "err: {err}");
    }
}
