//! `--gpus` flag parsing + single-host cluster envelope synthesis.
//!
//! The `--gpus` flag has uniform semantics ("use these GPUs") but the
//! mechanism depends on the command kind:
//!
//! - **Cluster-aware commands** (`cluster: true`): N >= 2 GPUs trigger
//!   synthesis of a single-host cluster envelope (master=127.0.0.1, lo
//!   transport, one host with N ranks) and spawn-per-rank via the existing
//!   launcher in [`crate::cluster::dispatch`]. The library inside each
//!   spawned process reads the envelope from `FLODL_CLUSTER_JSON` and uses
//!   the same code path as multi-host. N = 1 is degenerate -- no synthesis,
//!   just runs single-process on that device.
//!
//! - **Non-cluster commands** (`test`, `clippy`, etc.): `--gpus` sets
//!   `CUDA_VISIBLE_DEVICES` on the single child process. No envelope, no
//!   spawning. Tests internally manage their own multi-rank coordination
//!   (typically via the threaded `NcclRankComm` pattern in unit tests).
//!
//! Caller (`main.rs`) decides which mechanism applies based on whether the
//! resolved command's `cluster:` chain enables dispatch.

use std::process::Command;

use crate::cluster::resolve_local_hostname;
use crate::config::{ClusterConfig, ClusterHost};

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
                let count = count_visible_gpus_via_nvidia_smi()?;
                if count == 0 {
                    return Err(
                        "--gpus all: nvidia-smi reports 0 GPUs visible. Install \
                         NVIDIA drivers or specify devices explicitly (e.g. \
                         --gpus 0)."
                            .to_string(),
                    );
                }
                if count > u8::MAX as usize {
                    return Err(format!(
                        "--gpus all: nvidia-smi reports {count} GPUs which \
                         exceeds the supported device-index range (0..255). \
                         Specify devices explicitly via --gpus."
                    ));
                }
                Ok((0u8..count as u8).collect())
            }
        }
    }
}

/// Count visible CUDA devices via `nvidia-smi -L`.
///
/// Each GPU is one line starting with `GPU <idx>:`. Returns the number of
/// such lines. Loud error if nvidia-smi is missing or exits non-zero.
pub fn count_visible_gpus_via_nvidia_smi() -> Result<usize, String> {
    let out = Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map_err(|e| {
            format!(
                "failed to run `nvidia-smi -L`: {e}. Install NVIDIA drivers \
                 or specify devices explicitly (e.g. --gpus 0)."
            )
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "`nvidia-smi -L` exited non-zero: {}",
            stderr.trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(stdout.lines().filter(|l| l.starts_with("GPU ")).count())
}

/// Build a `ClusterConfig` for single-host loopback from a list of physical
/// CUDA device indices.
///
/// Used when `--gpus` is set on a cluster-aware command and no `cluster:`
/// block is in YAML. Returns a config with one host (this machine), N ranks
/// (`0..devices.len()`), NCCL loopback transport (`lo`).
///
/// `master_port` defaults to 29500, overridable via `FLODL_MASTER_PORT`.
/// Concurrent `fdl` cluster commands on the same host must use distinct
/// ports to avoid rendezvous collisions.
pub fn synthesize_local_cluster(devices: &[u8]) -> Result<ClusterConfig, String> {
    if devices.is_empty() {
        return Err("synthesize_local_cluster: device list is empty".to_string());
    }
    let name = resolve_local_hostname();
    let path = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| {
            format!("synthesize_local_cluster: cannot read current_dir: {e}")
        })?;
    let master_port = std::env::var("FLODL_MASTER_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(29500);

    Ok(ClusterConfig {
        master_addr: "127.0.0.1".to_string(),
        master_port,
        hosts: vec![ClusterHost {
            name,
            ranks: (0..devices.len()).collect(),
            local_devices: devices.to_vec(),
            nccl_socket_ifname: "lo".to_string(),
            path,
            libtorch_path: None,
            ssh: None,
        }],
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
    if joined.is_empty() {
        unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES") };
    } else {
        unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", &joined) };
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
        assert_eq!(c.master_addr, "127.0.0.1");
        assert_eq!(c.hosts.len(), 1);
        let h = &c.hosts[0];
        assert_eq!(h.ranks, vec![0, 1]);
        assert_eq!(h.local_devices, vec![0, 1]);
        assert_eq!(h.nccl_socket_ifname, "lo");
        assert!(h.libtorch_path.is_none());
        assert!(h.ssh.is_none());
        assert!(!h.name.trim().is_empty(), "hostname must be non-empty");
        assert!(!h.path.trim().is_empty(), "path must be non-empty");
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
        assert_eq!(c.hosts[0].ranks, vec![0]);
        assert_eq!(c.hosts[0].local_devices, vec![2]);
    }

    #[test]
    fn synthesize_local_cluster_rejects_empty() {
        let err = synthesize_local_cluster(&[]).unwrap_err();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn synthesize_local_cluster_respects_master_port_env() {
        // SAFETY: cargo test parallelism. Use a unique env var name probe
        // pattern instead of FLODL_MASTER_PORT to avoid clobbering other
        // tests -- but here we DO want to test the env reading path, so we
        // accept the race. Single-threaded mod tests would be cleaner.
        // For now, accept that this test runs serially-enough.
        unsafe {
            std::env::set_var("FLODL_MASTER_PORT", "31415");
        }
        let c = synthesize_local_cluster(&[0]).unwrap();
        unsafe {
            std::env::remove_var("FLODL_MASTER_PORT");
        }
        assert_eq!(c.master_port, 31415);
    }
}
