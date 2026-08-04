//! `--gpus` flag parsing + single-host cluster envelope synthesis.
//!
//! The `--gpus` flag has uniform semantics ("use these GPUs") but the
//! mechanism depends on the command kind:
//!
//! - **Cluster-aware commands** (`cluster: true`): N >= 2 GPUs trigger
//!   synthesis of a single-host cluster envelope (master=127.0.0.1, lo
//!   transport, one host with N ranks) and spawn-per-rank via the existing
//!   launcher (see [`crate::cluster::prepare_cluster_env`]). The library
//!   inside each spawned process reads the envelope from `FLODL_INTERNAL_CLUSTER_JSON`
//!   and uses the same code path as multi-host. N = 1 is degenerate — no
//!   synthesis, just runs single-process on that device.
//!
//! - **Non-cluster commands** (`test`, `clippy`, etc.): `--gpus` sets
//!   `CUDA_VISIBLE_DEVICES` on the single child process. No envelope, no
//!   spawning. Tests internally manage their own multi-rank coordination
//!   (typically via the threaded `NcclRankComm` pattern in unit tests).
//!
//! Caller (`main.rs`) decides which mechanism applies based on whether the
//! resolved command's `cluster:` chain enables dispatch.

use crate::cluster::resolve_local_hostname;
use crate::config::{
    ClusterConfig, ClusterController, ClusterWorker, LocalDevices,
    DEFAULT_CONTROLLER_PORT,
};

/// Parsed `--gpus` argument value.
///
/// Two forms accepted by [`GpusSpec::parse`]:
/// - `--gpus all`: resolve to all visible CUDA devices via `nvidia-smi -L`.
/// - `--gpus 0,1,2`: explicit comma-separated physical device indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpusSpec {
    /// Use every visible CUDA device. Resolved against `nvidia-smi -L` at
    /// [`GpusSpec::resolve`] time.
    All,
    /// Explicit list of physical CUDA device indices.
    List(Vec<u8>),
}

impl GpusSpec {
    /// Parse a `--gpus` value. Loud errors on empty, malformed, or duplicate
    /// device indices.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(
                "--gpus requires a value (e.g. `--gpus 0,1` or `--gpus all`)".to_string(),
            );
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return Ok(GpusSpec::All);
        }
        let mut out = Vec::new();
        for part in trimmed.split(',') {
            let p = part.trim();
            if p.is_empty() {
                return Err(format!("--gpus: empty entry in {trimmed:?}"));
            }
            let idx: u8 = p.parse().map_err(|e| {
                format!("--gpus: cannot parse {p:?} as device index: {e}")
            })?;
            out.push(idx);
        }
        let mut sorted = out.clone();
        sorted.sort_unstable();
        for win in sorted.windows(2) {
            if win[0] == win[1] {
                return Err(format!(
                    "--gpus: duplicate device index {} in {trimmed:?}",
                    win[0]
                ));
            }
        }
        Ok(GpusSpec::List(out))
    }

    /// Resolve to a concrete list of physical CUDA device indices.
    ///
    /// `List` returns its entries verbatim. `All` shells out to
    /// `nvidia-smi -L` and counts the result -- loud error if nvidia-smi
    /// is missing or returns 0 GPUs.
    pub fn resolve(&self) -> Result<Vec<u8>, String> {
        match self {
            GpusSpec::List(v) => Ok(v.clone()),
            GpusSpec::All => {
                // `require_devices` turns an empty sweep into the best
                // available explanation: a driver that failed to
                // enumerate, hardware present without its stack
                // installed, or genuinely no GPU. An explicit `--gpus
                // all` must fail loudly rather than resolve to zero.
                let devices = local_gpu_count()
                    .map_err(|e| format!("--gpus all: {e}"))?;
                if devices > u8::MAX as usize {
                    return Err(format!(
                        "--gpus all: {devices} GPUs detected, which exceeds \
                         the supported device-index range (0..255). Specify \
                         devices explicitly via --gpus."
                    ));
                }
                Ok((0u8..devices as u8).collect())
            }
        }
    }
}

/// Number of GPUs on this box, or a caller-facing reason there are none.
///
/// One entry point for every "how many GPUs are here" question in fdl,
/// across every vendor. The error is the point: an empty device list has
/// several causes (no driver, driver present but its tool broken,
/// hardware present without its stack installed, genuinely no card) and
/// a command that was *asked* for GPUs must say which one it hit rather
/// than silently resolving to zero.
///
/// Counts **physical** devices: `--gpus`/`local_devices` select from the
/// full set, and applying a visibility mask here would make the
/// selection depend on a mask the selection itself is about to set.
pub fn local_gpu_count() -> Result<usize, String> {
    flodl_hw::survey().require_devices().map(|d| d.len())
}

/// Build a `ClusterConfig` for single-host loopback from a list of physical
/// CUDA device indices.
///
/// Used when `--gpus` is set on a cluster-aware command and no `cluster:`
/// block is in YAML. Returns a config with one host (this machine), N ranks
/// (`0..devices.len()`), NCCL loopback transport (`lo`).
///
/// `controller.port` defaults to [`DEFAULT_CONTROLLER_PORT`] (1337),
/// overridable via `FLODL_CONTROLLER_PORT`. Concurrent `fdl` cluster
/// commands on the same host must use distinct ports to avoid
/// rendezvous collisions.
pub fn synthesize_local_cluster(devices: &[u8]) -> Result<ClusterConfig, String> {
    if devices.is_empty() {
        return Err("synthesize_local_cluster: device list is empty".to_string());
    }
    let hostname = resolve_local_hostname();
    let path = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| {
            format!("synthesize_local_cluster: cannot read current_dir: {e}")
        })?;
    let port = std::env::var("FLODL_CONTROLLER_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(DEFAULT_CONTROLLER_PORT);

    Ok(ClusterConfig {
        controller: ClusterController {
            host: "127.0.0.1".to_string(),
            port,
            path: path.clone(),
            docker: None,
            arch: None,
            data_path: None,
            join: None,
        },
        workers: vec![ClusterWorker {
            host: hostname,
            ranks: (0..devices.len()).collect(),
            local_devices: LocalDevices::Explicit(devices.to_vec()),
            nccl_socket_ifname: "lo".to_string(),
            path,
            ssh: None,
            tunnel: false,
            arch: None,
            data_path: None,
            docker: None,
            env: std::collections::BTreeMap::new(),
        }],
        env: std::collections::BTreeMap::new(),
    })
}

/// Set `CUDA_VISIBLE_DEVICES` to restrict the spawned process to the given
/// physical CUDA device indices.
///
/// Used on the non-cluster path (`--gpus 0,1` on `fdl test`, `clippy`, etc.)
/// so the single child process sees only the requested GPUs. NVIDIA Docker
/// forwards `CUDA_VISIBLE_DEVICES` to containers automatically.
///
/// Empty slice removes the var. The caller normally avoids calling with an
/// empty slice (a loud error earlier in the resolution path).
///
/// # Safety
///
/// Calls `std::env::set_var` which is unsafe in multi-threaded programs.
/// Must be called from `main` before any threads are spawned, which is the
/// case for the fdl-cli dispatch flow.
pub unsafe fn apply_cuda_visible_devices(devices: &[u8]) {
    let joined = devices
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    // Both vendors' spellings: HIP prefers its own variable over
    // CUDA_VISIBLE_DEVICES (first one set wins), so setting only the
    // CUDA one would leave an AMD box unmasked whenever HIP_VISIBLE_DEVICES
    // is already in the environment. Inert where the other vendor's
    // runtime never looks.
    for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"] {
        if joined.is_empty() {
            unsafe { std::env::remove_var(key) };
        } else {
            unsafe { std::env::set_var(key, &joined) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_case_insensitive() {
        assert_eq!(GpusSpec::parse("all").unwrap(), GpusSpec::All);
        assert_eq!(GpusSpec::parse("ALL").unwrap(), GpusSpec::All);
        assert_eq!(GpusSpec::parse("All").unwrap(), GpusSpec::All);
    }

    #[test]
    fn parse_single_index() {
        assert_eq!(GpusSpec::parse("0").unwrap(), GpusSpec::List(vec![0]));
        assert_eq!(GpusSpec::parse("3").unwrap(), GpusSpec::List(vec![3]));
    }

    #[test]
    fn parse_multiple_indices() {
        assert_eq!(
            GpusSpec::parse("0,1,2").unwrap(),
            GpusSpec::List(vec![0, 1, 2])
        );
        assert_eq!(GpusSpec::parse("3,1").unwrap(), GpusSpec::List(vec![3, 1]));
    }

    #[test]
    fn parse_tolerates_whitespace() {
        assert_eq!(
            GpusSpec::parse(" 0 , 1 ").unwrap(),
            GpusSpec::List(vec![0, 1])
        );
        assert_eq!(GpusSpec::parse("  all  ").unwrap(), GpusSpec::All);
    }

    #[test]
    fn parse_rejects_empty() {
        let err = GpusSpec::parse("").unwrap_err();
        assert!(err.contains("--gpus requires a value"), "got: {err}");
        let err = GpusSpec::parse("   ").unwrap_err();
        assert!(err.contains("--gpus requires a value"), "got: {err}");
    }

    #[test]
    fn parse_rejects_empty_entry() {
        let err = GpusSpec::parse("0,,1").unwrap_err();
        assert!(err.contains("empty entry"), "got: {err}");
        let err = GpusSpec::parse(",0").unwrap_err();
        assert!(err.contains("empty entry"), "got: {err}");
    }

    #[test]
    fn parse_rejects_non_numeric() {
        let err = GpusSpec::parse("0,abc").unwrap_err();
        assert!(err.contains("cannot parse"), "got: {err}");
        assert!(err.contains("abc"), "got: {err}");
    }

    #[test]
    fn parse_rejects_duplicates() {
        let err = GpusSpec::parse("0,1,0").unwrap_err();
        assert!(err.contains("duplicate"), "got: {err}");
        assert!(err.contains("0"), "got: {err}");
    }

    #[test]
    fn resolve_list_returns_verbatim() {
        let r = GpusSpec::List(vec![3, 1]).resolve().unwrap();
        assert_eq!(r, vec![3, 1]);
    }

    #[test]
    fn synthesize_local_cluster_basic_shape() {
        // We don't control hostname/cwd here, so just assert structural invariants.
        let c = synthesize_local_cluster(&[0, 1]).unwrap();
        assert_eq!(c.controller.host, "127.0.0.1");
        assert_eq!(c.workers.len(), 1);
        let w = &c.workers[0];
        assert_eq!(w.ranks, vec![0, 1]);
        assert_eq!(w.local_devices, LocalDevices::Explicit(vec![0, 1]));
        assert_eq!(w.nccl_socket_ifname, "lo");
        assert!(w.arch.is_none());
        assert!(w.ssh.is_none());
        assert!(!w.host.trim().is_empty(), "hostname must be non-empty");
        assert!(!w.path.trim().is_empty(), "path must be non-empty");
    }

    #[test]
    fn synthesize_local_cluster_validates() {
        // The synthesized config must pass ClusterConfig::validate (so the
        // launcher accepts it without special-casing).
        let c = synthesize_local_cluster(&[0, 1]).unwrap();
        c.validate().expect("synthesized cluster must pass validate");
    }

    #[test]
    fn synthesize_local_cluster_single_device() {
        // N=1 is structurally valid (validate enforces 0..world_size with
        // ranks=[0], devices=[0]). Caller decides whether to use it.
        let c = synthesize_local_cluster(&[2]).unwrap();
        c.validate().expect("single-device synthesized config validates");
        assert_eq!(c.workers[0].ranks, vec![0]);
        assert_eq!(c.workers[0].local_devices, LocalDevices::Explicit(vec![2]));
    }

    #[test]
    fn synthesize_local_cluster_rejects_empty() {
        let err = synthesize_local_cluster(&[]).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn synthesize_local_cluster_respects_controller_port_env() {
        // SAFETY: cargo test parallelism. Use a unique env var name probe
        // pattern instead of FLODL_CONTROLLER_PORT to avoid clobbering other
        // tests -- but here we DO want to test the env reading path, so we
        // accept the race. Single-threaded mod tests would be cleaner.
        // For now, accept that this test runs serially-enough.
        unsafe {
            std::env::set_var("FLODL_CONTROLLER_PORT", "31415");
        }
        let c = synthesize_local_cluster(&[0]).unwrap();
        unsafe {
            std::env::remove_var("FLODL_CONTROLLER_PORT");
        }
        assert_eq!(c.controller.port, 31415);
    }
}
