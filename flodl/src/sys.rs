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
///
/// NOTE: `GpuInfo` + the nvidia-smi enumeration (`detect_gpus_raw` /
/// `parse_gpu_csv_row`) are intentionally duplicated in
/// `flodl_cli::util::system`. flodl-cli cannot depend on flodl (it must build
/// without libtorch), so the two can't share a module — keep the query column
/// order + parse in sync with that copy.
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
///
/// # `CUDA_VISIBLE_DEVICES`
///
/// `nvidia-smi` reports every physical GPU regardless of
/// `CUDA_VISIBLE_DEVICES`. To match what the libtorch runtime would
/// actually see (and to keep the auto-promote logic in
/// `DdpHandle::launch` consistent), this function filters the raw
/// `nvidia-smi` output by `CUDA_VISIBLE_DEVICES` when set:
///
/// - `CUDA_VISIBLE_DEVICES=0,2` → returns only the GPUs with indices
///   0 and 2 from the physical enumeration.
/// - `CUDA_VISIBLE_DEVICES=` (empty) → returns an empty `Vec`
///   (matches libtorch behavior: "explicitly no CUDA").
/// - Unset → returns every physical GPU.
///
/// This lets tests scope down to single-GPU view via
/// `CUDA_VISIBLE_DEVICES=0 cargo test` and prevents auto-promote from
/// surprising the test harness on a multi-GPU box.
pub fn detect_gpus() -> Vec<GpuInfo> {
    let all = detect_gpus_raw();
    let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") else {
        return all;
    };
    let trimmed = visible.trim();
    if trimmed.is_empty() {
        // Explicit "no CUDA" — libtorch treats this as zero devices.
        return Vec::new();
    }
    let mut allowed: std::collections::HashSet<u8> = std::collections::HashSet::new();
    for entry in trimmed.split(',') {
        let entry = entry.trim();
        match entry.parse::<u8>() {
            Ok(idx) => {
                allowed.insert(idx);
            }
            Err(_) => {
                // CUDA also accepts GPU-<uuid> / MIG-<...> forms that this
                // index-based filter can't resolve. Silently dropping them
                // reported "no GPUs" while libtorch would happily see one —
                // exactly the runtime divergence detect_gpus exists to
                // prevent. Loud per the explicit-selector rule.
                eprintln!(
                    "flodl sys: CUDA_VISIBLE_DEVICES entry '{entry}' is not a \
                     numeric index (UUID/MIG forms are not resolved by \
                     detect_gpus); GPU detection may under-count"
                );
            }
        }
    }
    all.into_iter()
        .filter(|g| allowed.contains(&g.index))
        .collect()
}

/// Host RAM snapshot: total and currently-available bytes.
///
/// `available_bytes` is the kernel's `MemAvailable` estimate: memory a
/// new workload can take without pushing the system into swap,
/// reclaimable page cache included. That makes it the honest baseline
/// for sizing staging buffers — it already accounts for every other
/// process on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemInfo {
    /// Total physical RAM in bytes (`MemTotal`).
    pub total_bytes: u64,
    /// Kernel estimate of allocatable RAM in bytes (`MemAvailable`).
    pub available_bytes: u64,
}

/// Read host RAM totals from `/proc/meminfo`.
///
/// Returns `None` when `/proc/meminfo` is missing (non-Linux) or does
/// not parse — callers fall back to conservative fixed sizing.
pub fn mem_info() -> Option<MemInfo> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// Parse `MemTotal:` / `MemAvailable:` lines (values are in kB).
fn parse_meminfo(text: &str) -> Option<MemInfo> {
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_kb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_kb(rest);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }
    Some(MemInfo {
        total_bytes: total? * 1024,
        available_bytes: available? * 1024,
    })
}

fn parse_meminfo_kb(rest: &str) -> Option<u64> {
    rest.trim().strip_suffix("kB")?.trim().parse().ok()
}

/// Unfiltered nvidia-smi enumeration. Internal — public callers go
/// through [`detect_gpus`] which honors `CUDA_VISIBLE_DEVICES`.
fn detect_gpus_raw() -> Vec<GpuInfo> {
    let output = match Command::new("nvidia-smi")
        // `name` is queried LAST: it is the only field that can contain the
        // `", "` separator, so with it last `splitn(4, ", ")` keeps the whole
        // name (commas and all) in the final cell — no CSV parser needed.
        .args([
            "--query-gpu=index,compute_cap,memory.total,name",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            // nvidia-smi present but errored — most commonly an old driver
            // bundle that doesn't support the `compute_cap` query field.
            // Distinguish from "no nvidia-smi" so a real multi-GPU box
            // reporting zero GPUs leaves a trace instead of a mystery.
            eprintln!(
                "flodl sys: nvidia-smi exited non-zero ({}); reporting 0 GPUs. \
                 stderr: {}",
                o.status,
                String::from_utf8_lossy(&o.stderr).trim(),
            );
            return Vec::new();
        }
        Err(_) => return Vec::new(), // nvidia-smi absent: genuinely no CUDA
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let parsed = parse_gpu_csv_row(line);
            if parsed.is_none() {
                // A malformed row = a GPU silently dropped. Leave a trace.
                eprintln!("flodl sys: could not parse an nvidia-smi GPU row, skipping: {line:?}");
            }
            parsed
        })
        .collect()
}

/// Parse one `index, compute_cap, memory.total, name` CSV row (nvidia-smi
/// `--format=csv,noheader,nounits`; `name` is last so an embedded ", " stays
/// intact — the first three cells are comma-free). None on any bad field.
fn parse_gpu_csv_row(line: &str) -> Option<GpuInfo> {
    let parts: Vec<&str> = line.splitn(4, ", ").collect();
    if parts.len() < 4 {
        return None;
    }
    let index: u8 = parts[0].trim().parse().ok()?;
    let cap_parts: Vec<&str> = parts[1].trim().split('.').collect();
    let sm_major: u32 = cap_parts.first()?.parse().ok()?;
    let sm_minor: u32 = cap_parts.get(1)?.parse().ok()?;
    let total_memory_mb: u64 = parts[2].trim().parse().ok()?;
    let name = parts[3].trim().to_string();
    Some(GpuInfo {
        index,
        name,
        sm_major,
        sm_minor,
        total_memory_mb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env mutations must be serialized — cargo test runs in parallel
    // and `CUDA_VISIBLE_DEVICES` is process-global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII helper that snapshots `CUDA_VISIBLE_DEVICES` on construction
    /// and restores it on drop. Pair with `ENV_LOCK` to make env tests
    /// thread-safe in the parallel harness.
    struct CudaVisibleGuard {
        prev: Option<String>,
    }

    impl CudaVisibleGuard {
        fn set(value: &str) -> Self {
            let prev = std::env::var("CUDA_VISIBLE_DEVICES").ok();
            // SAFETY: `ENV_LOCK` serializes env mutations across tests
            // in this module. Outside-module readers may still race,
            // but cargo test in flodl doesn't read this env var
            // outside the GPU path which doesn't run under `fdl test`.
            unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", value); }
            Self { prev }
        }
        fn unset() -> Self {
            let prev = std::env::var("CUDA_VISIBLE_DEVICES").ok();
            unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES"); }
            Self { prev }
        }
    }

    impl Drop for CudaVisibleGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var("CUDA_VISIBLE_DEVICES", v),
                    None => std::env::remove_var("CUDA_VISIBLE_DEVICES"),
                }
            }
        }
    }

    #[test]
    fn detect_gpus_returns_empty_or_valid() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = CudaVisibleGuard::unset();
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

    #[test]
    fn detect_gpus_empty_cuda_visible_devices_returns_empty() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g = CudaVisibleGuard::set("");
        // Empty CUDA_VISIBLE_DEVICES means "explicitly no CUDA" — libtorch
        // treats it as zero devices, so detect_gpus must match.
        assert!(detect_gpus().is_empty());
    }

    #[test]
    fn detect_gpus_cuda_visible_devices_filters_by_index() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _g_unset = CudaVisibleGuard::unset();
        // Get the physical baseline (whatever this host has).
        let physical = detect_gpus();
        if physical.is_empty() {
            // No GPUs on this CI box — nothing to filter. Test is a no-op.
            return;
        }
        drop(_g_unset);
        // Pick an index that exists; restrict to just that one.
        let pick = physical[0].index;
        let _g_set = CudaVisibleGuard::set(&pick.to_string());
        let filtered = detect_gpus();
        assert_eq!(filtered.len(), 1, "single-index filter narrows to one");
        assert_eq!(filtered[0].index, pick);
    }

    #[test]
    fn mem_info_parses_meminfo_format() {
        let text = "MemTotal:       131781120 kB\n\
                    MemFree:         8123456 kB\n\
                    MemAvailable:   98765432 kB\n\
                    Buffers:          123456 kB\n";
        let m = parse_meminfo(text).unwrap();
        assert_eq!(m.total_bytes, 131_781_120 * 1024);
        assert_eq!(m.available_bytes, 98_765_432 * 1024);

        // Missing MemAvailable (pre-3.14 kernels) → None, not a guess.
        assert!(parse_meminfo("MemTotal: 100 kB\nMemFree: 50 kB\n").is_none());
        assert!(parse_meminfo("").is_none());
    }

    #[test]
    fn mem_info_reads_live_host() {
        // On Linux this must parse; elsewhere None is the contract.
        if let Some(m) = mem_info() {
            assert!(m.total_bytes > 0);
            assert!(m.available_bytes <= m.total_bytes);
        }
    }

    #[test]
    fn detect_gpus_cuda_visible_devices_excludes_nonexistent_index() {
        let _lock = ENV_LOCK.lock().unwrap();
        // Index 99 should never exist on any realistic rig — verify filter
        // drops missing indices instead of inventing them.
        let _g = CudaVisibleGuard::set("99");
        assert!(detect_gpus().is_empty());
    }

    #[test]
    fn parse_gpu_csv_row_parses_a_well_formed_row() {
        // Column order: index, compute_cap, memory.total, name.
        let g = parse_gpu_csv_row("1, 6.1, 6078, NVIDIA GeForce GTX 1060 6GB").unwrap();
        assert_eq!(g.index, 1);
        assert_eq!(g.name, "NVIDIA GeForce GTX 1060 6GB");
        assert_eq!(g.sm_major, 6);
        assert_eq!(g.sm_minor, 1);
        assert_eq!(g.total_memory_mb, 6078);
    }

    #[test]
    fn parse_gpu_csv_row_keeps_comma_in_name() {
        // `name` last => an embedded ", " stays in the final cell (splitn(4)
        // stops after 3 separators) rather than truncating the row.
        let g = parse_gpu_csv_row("0, 8.0, 81920, NVIDIA A100, 80GB").unwrap();
        assert_eq!(g.name, "NVIDIA A100, 80GB");
        assert_eq!(g.total_memory_mb, 81920);
    }

    #[test]
    fn parse_gpu_csv_row_rejects_malformed() {
        assert!(parse_gpu_csv_row("0, 8.9, three").is_none());
        assert!(parse_gpu_csv_row("x, 8.9, 24564, name").is_none());
    }
}
