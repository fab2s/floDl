//! System-level detection that does NOT touch libtorch / CUDA.
//!
//! flodl's main CUDA APIs (e.g. [`crate::tensor::cuda_device_count`])
//! initialize libtorch on first call. Once libtorch latches onto a
//! device list, `CUDA_VISIBLE_DEVICES` is ignored and — critically for
//! cluster mode — the spawned children inherit a corrupted CUDA
//! context on heterogeneous-GPU rigs.
//!
//! [`detect_gpus`] avoids both problems by shelling out to `nvidia-smi`
//! and parsing its CSV output. Use this when you need GPU info for
//! pre-`Trainer::run` decisions (mode filtering, log banners,
//! CLI-flag validation) — see the "no CUDA before `Trainer::run`"
//! invariant in the [`crate::distributed::Trainer`] docs.
//!
//! For *runtime* GPU queries (after dispatch), the libtorch-backed
//! APIs in [`crate::tensor`] remain the right tool.
//!
//! Returns an empty `Vec` when `nvidia-smi` is missing or fails —
//! callers can treat that as "no CUDA visible" without a separate
//! "did we have a driver" branch.

use std::process::Command;

/// One GPU's identity + capability + VRAM, as reported by `nvidia-smi`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// `nvidia-smi`'s device index (0-based). Matches the index libtorch
    /// would assign with `CUDA_VISIBLE_DEVICES` unset.
    pub index: u8,
    /// Marketing name (e.g. `"NVIDIA GeForce RTX 5060 Ti"`).
    pub name: String,
    /// CUDA compute capability major (e.g. `12` for Blackwell sm_120).
    pub sm_major: u32,
    /// CUDA compute capability minor (e.g. `0` for sm_120, `1` for sm_61).
    pub sm_minor: u32,
    /// Total VRAM in MiB (as reported by `nvidia-smi --query-gpu=memory.total`).
    pub total_memory_mb: u64,
}

impl GpuInfo {
    /// Canonical `sm_NN` form (e.g. `"sm_120"`, `"sm_61"`).
    pub fn sm_version(&self) -> String {
        format!("sm_{}{}", self.sm_major, self.sm_minor)
    }

    /// Total VRAM in bytes.
    pub fn vram_bytes(&self) -> u64 {
        self.total_memory_mb * 1024 * 1024
    }

    /// `name` with the common `"NVIDIA "` / `"GeForce "` prefixes
    /// stripped — kinder on `eprintln!`-style banners.
    pub fn short_name(&self) -> String {
        self.name.replace("NVIDIA ", "").replace("GeForce ", "")
    }
}

/// Enumerate visible GPUs WITHOUT initializing libtorch / CUDA.
///
/// Shells out to `nvidia-smi --query-gpu=index,name,compute_cap,memory.total`.
/// Returns an empty `Vec` when `nvidia-smi` is missing, fails to run,
/// or returns a non-zero exit (treat as "no CUDA visible").
///
/// Use for pre-`Trainer::run` decisions where calling
/// [`crate::tensor::cuda_device_count`] would prematurely init libtorch
/// and break the cluster launcher's "no CUDA touch before fan-out"
/// invariant on heterogeneous rigs.
pub fn detect_gpus() -> Vec<GpuInfo> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(4, ", ").collect();
            if parts.len() < 4 {
                return None;
            }
            let index: u8 = parts[0].trim().parse().ok()?;
            let name = parts[1].trim().to_string();
            let cap_parts: Vec<&str> = parts[2].trim().split('.').collect();
            let sm_major: u32 = cap_parts.first()?.parse().ok()?;
            let sm_minor: u32 = cap_parts.get(1)?.parse().ok()?;
            let total_memory_mb: u64 = parts[3].trim().parse().ok()?;
            Some(GpuInfo {
                index,
                name,
                sm_major,
                sm_minor,
                total_memory_mb,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_gpus_returns_empty_or_valid() {
        // On CI without GPUs: empty Vec. On a GPU box: parseable info.
        // Either is fine — function must NOT panic and must NOT touch libtorch.
        let gpus = detect_gpus();
        for g in &gpus {
            assert!(!g.name.is_empty(), "name parsed");
            assert!(g.total_memory_mb > 0, "VRAM parsed");
            assert!(g.sm_version().starts_with("sm_"), "sm version formatted");
        }
    }

    #[test]
    fn gpu_info_sm_version_format() {
        let g = GpuInfo {
            index: 0,
            name: "NVIDIA Test".into(),
            sm_major: 12,
            sm_minor: 0,
            total_memory_mb: 16000,
        };
        assert_eq!(g.sm_version(), "sm_120");
        assert_eq!(g.short_name(), "Test");
        assert_eq!(g.vram_bytes(), 16000 * 1024 * 1024);
    }
}
