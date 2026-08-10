//! Deprecated names, kept so existing code keeps compiling.
//!
//! Two renaming arcs live here. The bulk is the `cuda_*` / `Cuda*` →
//! `gpu_*` / `Gpu*` sweep documented below; the tail is the convergence-guard
//! rename (`TrendGuard` →
//! [`LevelGuard`](crate::distributed::ddp_run::LevelGuard), `MsfGuard` →
//! [`GrowthGuard`](crate::distributed::ddp_run::GrowthGuard)).
//!
//! The GPU vendor is a build-time property of the libtorch you link
//! against, not something these entry points select: on a ROCm build
//! `cuda_device_count()` returned an AMD device count, and the name said
//! otherwise. Everything here forwards to the `gpu_*` / `Gpu*` name that
//! tells the truth on both vendors.
//!
//! Two names were deliberately NOT renamed and so do not appear here:
//! [`crate::cuda_compute_capability`], because compute capability is an
//! NVIDIA concept with no AMD counterpart (it errors on a ROCm build
//! rather than dressing a gfx architecture up as an `sm_` pair), and
//! `set_cudnn_benchmark`, which keeps PyTorch's own spelling — upstream
//! routes `torch.backends.cudnn.benchmark` to MIOpen on ROCm.
//!
//! This module is a single unit of removal: delete the file, drop the
//! `mod compat;` line and its re-export block in `lib.rs`.

use crate::nn::cuda_graph::{GpuGraph, MemPoolId, gpu_graph_capture, gpu_graph_pool_handle};
use crate::tensor::cuda_event::GpuEvent as GpuEventInner;
use crate::tensor::cuda_event::GpuEventFlags as GpuEventFlagsInner;
use crate::tensor::cuda_stream::GpuStream as GpuStreamInner;
use crate::tensor::{
    Device, DeviceInfo, Result, current_gpu_device, gpu_active_bytes, gpu_active_bytes_idx,
    gpu_allocated_bytes, gpu_allocated_bytes_idx, gpu_available, gpu_device_count, gpu_device_name,
    gpu_device_name_idx, gpu_devices, gpu_empty_cache, gpu_has_primary_context,
    gpu_manual_seed_all, gpu_memory_info, gpu_memory_info_idx, gpu_peak_active_bytes,
    gpu_peak_active_bytes_idx, gpu_peak_reserved_bytes, gpu_peak_reserved_bytes_idx,
    gpu_reset_peak_stats, gpu_reset_peak_stats_idx, gpu_smi_memory_info_idx, gpu_synchronize,
    gpu_utilization, gpu_utilization_idx, set_current_gpu_device, usable_gpu_devices,
};

/// Deprecated alias for [`crate::GpuStream`].
#[deprecated(note = "renamed to `GpuStream` — the type is not NVIDIA-specific")]
pub type CudaStream = GpuStreamInner;

/// Deprecated alias for [`crate::GpuEvent`].
#[deprecated(note = "renamed to `GpuEvent` — the type is not NVIDIA-specific")]
pub type CudaEvent = GpuEventInner;

/// Deprecated alias for [`crate::GpuEventFlags`].
#[deprecated(note = "renamed to `GpuEventFlags` — the type is not NVIDIA-specific")]
pub type CudaEventFlags = GpuEventFlagsInner;

/// Deprecated alias for [`crate::GpuGraph`].
#[deprecated(note = "renamed to `GpuGraph` — ROCm calls the same construct a HIP graph")]
pub type CudaGraph = GpuGraph;

/// Deprecated alias for [`crate::gpu_available`].
#[deprecated(note = "renamed to `gpu_available`")]
pub fn cuda_available() -> bool {
    gpu_available()
}
/// Deprecated alias for [`crate::gpu_device_count`].
#[deprecated(note = "renamed to `gpu_device_count`")]
pub fn cuda_device_count() -> i32 {
    gpu_device_count()
}
/// Deprecated alias for [`crate::gpu_memory_info`].
#[deprecated(note = "renamed to `gpu_memory_info`")]
pub fn cuda_memory_info() -> Result<(u64, u64)> {
    gpu_memory_info()
}
/// Deprecated alias for [`crate::gpu_memory_info_idx`].
#[deprecated(note = "renamed to `gpu_memory_info_idx`")]
pub fn cuda_memory_info_idx(device_index: i32) -> Result<(u64, u64)> {
    gpu_memory_info_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_allocated_bytes`].
#[deprecated(note = "renamed to `gpu_allocated_bytes`")]
pub fn cuda_allocated_bytes() -> Result<u64> {
    gpu_allocated_bytes()
}
/// Deprecated alias for [`crate::gpu_allocated_bytes_idx`].
#[deprecated(note = "renamed to `gpu_allocated_bytes_idx`")]
pub fn cuda_allocated_bytes_idx(device_index: i32) -> Result<u64> {
    gpu_allocated_bytes_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_active_bytes`].
#[deprecated(note = "renamed to `gpu_active_bytes`")]
pub fn cuda_active_bytes() -> Result<u64> {
    gpu_active_bytes()
}
/// Deprecated alias for [`crate::gpu_active_bytes_idx`].
#[deprecated(note = "renamed to `gpu_active_bytes_idx`")]
pub fn cuda_active_bytes_idx(device_index: i32) -> Result<u64> {
    gpu_active_bytes_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_peak_active_bytes`].
#[deprecated(note = "renamed to `gpu_peak_active_bytes`")]
pub fn cuda_peak_active_bytes() -> Result<u64> {
    gpu_peak_active_bytes()
}
/// Deprecated alias for [`crate::gpu_peak_active_bytes_idx`].
#[deprecated(note = "renamed to `gpu_peak_active_bytes_idx`")]
pub fn cuda_peak_active_bytes_idx(device_index: i32) -> Result<u64> {
    gpu_peak_active_bytes_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_peak_reserved_bytes`].
#[deprecated(note = "renamed to `gpu_peak_reserved_bytes`")]
pub fn cuda_peak_reserved_bytes() -> Result<u64> {
    gpu_peak_reserved_bytes()
}
/// Deprecated alias for [`crate::gpu_peak_reserved_bytes_idx`].
#[deprecated(note = "renamed to `gpu_peak_reserved_bytes_idx`")]
pub fn cuda_peak_reserved_bytes_idx(device_index: i32) -> Result<u64> {
    gpu_peak_reserved_bytes_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_reset_peak_stats`].
#[deprecated(note = "renamed to `gpu_reset_peak_stats`")]
pub fn cuda_reset_peak_stats() {
    gpu_reset_peak_stats()
}
/// Deprecated alias for [`crate::gpu_reset_peak_stats_idx`].
#[deprecated(note = "renamed to `gpu_reset_peak_stats_idx`")]
pub fn cuda_reset_peak_stats_idx(device_index: i32) {
    gpu_reset_peak_stats_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_empty_cache`].
#[deprecated(note = "renamed to `gpu_empty_cache`")]
pub fn cuda_empty_cache() {
    gpu_empty_cache()
}
/// Deprecated alias for [`crate::gpu_utilization`].
#[deprecated(note = "renamed to `gpu_utilization`")]
pub fn cuda_utilization() -> Option<u32> {
    gpu_utilization()
}
/// Deprecated alias for [`crate::gpu_utilization_idx`].
#[deprecated(note = "renamed to `gpu_utilization_idx`")]
pub fn cuda_utilization_idx(device_index: i32) -> Option<u32> {
    gpu_utilization_idx(device_index)
}
/// Deprecated alias for [`crate::gpu_device_name`].
#[deprecated(note = "renamed to `gpu_device_name`")]
pub fn cuda_device_name() -> Option<String> {
    gpu_device_name()
}
/// Deprecated alias for [`crate::gpu_device_name_idx`].
#[deprecated(note = "renamed to `gpu_device_name_idx`")]
pub fn cuda_device_name_idx(device: i32) -> Option<String> {
    gpu_device_name_idx(device)
}
/// Deprecated alias for [`crate::gpu_devices`].
#[deprecated(note = "renamed to `gpu_devices`")]
pub fn cuda_devices() -> Vec<DeviceInfo> {
    gpu_devices()
}
/// Deprecated alias for [`crate::usable_gpu_devices`].
#[deprecated(note = "renamed to `usable_gpu_devices`")]
pub fn usable_cuda_devices() -> Vec<Device> {
    usable_gpu_devices()
}
/// Deprecated alias for [`crate::set_current_gpu_device`].
#[deprecated(note = "renamed to `set_current_gpu_device`")]
pub fn set_current_cuda_device(device_index: u8) {
    set_current_gpu_device(device_index)
}
/// Deprecated alias for [`crate::current_gpu_device`].
#[deprecated(note = "renamed to `current_gpu_device`")]
pub fn current_cuda_device() -> u8 {
    current_gpu_device()
}
/// Deprecated alias for [`crate::gpu_synchronize`].
#[deprecated(note = "renamed to `gpu_synchronize`")]
pub fn cuda_synchronize(device_index: u8) {
    gpu_synchronize(device_index)
}
/// Deprecated alias for [`crate::gpu_manual_seed_all`].
#[deprecated(note = "renamed to `gpu_manual_seed_all`")]
pub fn cuda_manual_seed_all(seed: u64) {
    gpu_manual_seed_all(seed)
}
/// Deprecated alias for [`crate::gpu_has_primary_context`].
#[deprecated(note = "renamed to `gpu_has_primary_context`")]
pub fn cuda_has_primary_context(device_index: i32) -> bool {
    gpu_has_primary_context(device_index)
}
/// Deprecated alias for [`crate::gpu_graph_pool_handle`].
#[deprecated(note = "renamed to `gpu_graph_pool_handle`")]
pub fn cuda_graph_pool_handle() -> MemPoolId {
    gpu_graph_pool_handle()
}

/// Deprecated alias for [`crate::gpu_smi_memory_info_idx`].
///
/// Renamed off `nvml` because the concept — device-wide VRAM read
/// outside the caching allocator — is vendor-neutral even though NVML is
/// the only backend wired up today.
#[deprecated(note = "renamed to `gpu_smi_memory_info_idx`")]
pub fn cuda_nvml_memory_info_idx(physical_index: i32) -> Option<(u64, u64)> {
    gpu_smi_memory_info_idx(physical_index)
}

/// Deprecated alias for [`crate::gpu_graph_capture`].
#[deprecated(note = "renamed to `gpu_graph_capture`")]
pub fn cuda_graph_capture<F>(warmup_runs: usize, pool: Option<MemPoolId>, f: F) -> Result<GpuGraph>
where
    F: FnMut() -> Result<()>,
{
    gpu_graph_capture(warmup_runs, pool, f)
}

// ── Convergence guards ──────────────────────────────────────────────────
//
// The guards were named after what inspired them rather than what they
// measure. `MsfGuard`'s "MSF" was Master Stability Function, borrowed from
// synchronization theory — a lineage the research review declined to defend,
// since flodl's ranks are non-identical, stochastic and independently
// perturbed where the theory assumes identical deterministic oscillators. It
// was also expanded nowhere in the codebase, so `--guard msf` was undecodable
// from the inside. `TrendGuard` was merely imprecise, but it is the other half
// of a pair whose axis is level-versus-rate, and that axis only reads once
// both names state it.

/// Deprecated alias for [`crate::distributed::ddp_run::LevelGuard`].
#[deprecated(note = "renamed to `LevelGuard` — it watches the divergence level, \
                     as against `GrowthGuard`'s growth rate")]
pub type TrendGuard = crate::distributed::ddp_run::convergence::LevelGuard;

/// Deprecated alias for [`crate::distributed::ddp_run::GrowthGuard`].
#[deprecated(note = "renamed to `GrowthGuard` — it watches the rate at which \
                     divergence compounds; the old name stood for Master \
                     Stability Function, a borrowed framing this does not claim")]
pub type MsfGuard = crate::distributed::ddp_run::convergence::GrowthGuard;

#[cfg(test)]
mod tests {
    //! The BC promise is that these names still resolve and forward.
    //! Nothing else in the tree calls them (that is the point), so
    //! without this they could silently rot into a broken re-export.
    #![allow(deprecated)]
    use super::*;

    #[test]
    fn deprecated_aliases_forward_to_their_replacements() {
        assert_eq!(cuda_available(), gpu_available());
        assert_eq!(cuda_device_count(), gpu_device_count());
        // `current_gpu_device` asks the driver directly; on a GPU-feature
        // build running without a driver that is a fatal C++ exception,
        // not an error return. Everything else here degrades to 0/empty.
        if gpu_available() {
            assert_eq!(current_cuda_device(), current_gpu_device());
        }
        assert_eq!(cuda_devices().len(), gpu_devices().len());
        assert_eq!(usable_cuda_devices().len(), usable_gpu_devices().len());
    }

    #[test]
    fn deprecated_guard_aliases_resolve() {
        // Compile-time only: the old guard names must still name the new
        // types, so a `MsfGuard::default()` in someone's builder call keeps
        // compiling.
        fn _assert_level(g: TrendGuard) -> crate::distributed::ddp_run::LevelGuard {
            g
        }
        fn _assert_growth(g: MsfGuard) -> crate::distributed::ddp_run::GrowthGuard {
            g
        }
    }

    #[test]
    fn deprecated_type_aliases_resolve() {
        // Compile-time only: each alias must still name its replacement.
        fn _assert_same(s: CudaStream) -> crate::GpuStream {
            s
        }
        fn _assert_event(e: CudaEvent) -> crate::GpuEvent {
            e
        }
        fn _assert_flags(f: CudaEventFlags) -> crate::GpuEventFlags {
            f
        }
        fn _assert_graph(g: CudaGraph) -> crate::GpuGraph {
            g
        }
    }
}
