//! CUDA utility functions — device queries, memory stats, hardware probing.

use std::ffi::CStr;

use flodl_sys as ffi;

use super::{check_err, Device, DType, Result, Tensor, TensorError, TensorOptions, LIVE_TENSOR_COUNT};
use std::sync::atomic::Ordering;

/// Returns true if a GPU is available.
///
/// On Linux, this also ensures CUDA libraries are loaded (they can be
/// dropped by the linker's `--as-needed` flag since no Rust code
/// directly references symbols in `libtorch_cuda.so`).
pub fn gpu_available() -> bool {
    // flodl_force_gpu_link references c10::cuda::device_count(),
    // creating a real symbol dependency on c10_cuda.so. This prevents
    // --as-needed from dropping CUDA libs. The call is cheap (no-op on
    // non-CUDA builds since the symbol resolves to a stub returning 0).
    unsafe { let _ = ffi::flodl_force_gpu_link(); }
    unsafe { ffi::flodl_gpu_is_available() != 0 }
}

/// Returns the number of GPU devices.
pub fn gpu_device_count() -> i32 {
    unsafe { ffi::flodl_gpu_device_count() }
}

/// Query GPU memory usage for a specific device.
/// Returns `(used_bytes, total_bytes)` or an error if no GPU is available.
pub fn gpu_memory_info_idx(device_index: i32) -> Result<(u64, u64)> {
    let mut used: u64 = 0;
    let mut total: u64 = 0;
    check_err(unsafe { ffi::flodl_gpu_mem_info(device_index, &mut used, &mut total) })?;
    Ok((used, total))
}

/// Query GPU memory usage for device 0.
/// Returns `(used_bytes, total_bytes)` or an error if no GPU is available.
pub fn gpu_memory_info() -> Result<(u64, u64)> {
    gpu_memory_info_idx(0)
}

/// Query bytes reserved by the GPU caching allocator on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.memory_reserved()`. It can exceed
/// physical VRAM when unified memory spills to host RAM.
pub fn gpu_allocated_bytes_idx(device_index: i32) -> Result<u64> {
    let mut allocated: u64 = 0;
    check_err(unsafe { ffi::flodl_gpu_alloc_bytes(device_index, &mut allocated) })?;
    Ok(allocated)
}

/// Query bytes reserved by the GPU caching allocator on device 0.
pub fn gpu_allocated_bytes() -> Result<u64> {
    gpu_allocated_bytes_idx(0)
}

/// Query bytes actively used by tensors on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.memory_allocated()`. Unlike
/// `gpu_allocated_bytes` (which reports the allocator's total reservation),
/// this only counts sub-blocks currently backing live tensors.
pub fn gpu_active_bytes_idx(device_index: i32) -> Result<u64> {
    let mut active: u64 = 0;
    check_err(unsafe { ffi::flodl_gpu_active_bytes(device_index, &mut active) })?;
    Ok(active)
}

/// Query bytes actively used by tensors on device 0.
pub fn gpu_active_bytes() -> Result<u64> {
    gpu_active_bytes_idx(0)
}

/// Peak bytes allocated to tensors since last `gpu_reset_peak_stats()` on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.max_memory_allocated()`.
pub fn gpu_peak_active_bytes_idx(device_index: i32) -> Result<u64> {
    let mut peak: u64 = 0;
    check_err(unsafe { ffi::flodl_gpu_peak_active_bytes(device_index, &mut peak) })?;
    Ok(peak)
}

/// Peak bytes allocated to tensors since last `gpu_reset_peak_stats()` on device 0.
pub fn gpu_peak_active_bytes() -> Result<u64> {
    gpu_peak_active_bytes_idx(0)
}

/// Peak bytes reserved by the GPU caching allocator since last `gpu_reset_peak_stats()` on a specific device.
///
/// This is the Rust equivalent of `torch.cuda.max_memory_reserved()`.
pub fn gpu_peak_reserved_bytes_idx(device_index: i32) -> Result<u64> {
    let mut peak: u64 = 0;
    check_err(unsafe { ffi::flodl_gpu_peak_reserved_bytes(device_index, &mut peak) })?;
    Ok(peak)
}

/// Peak bytes reserved by the GPU caching allocator since last `gpu_reset_peak_stats()` on device 0.
pub fn gpu_peak_reserved_bytes() -> Result<u64> {
    gpu_peak_reserved_bytes_idx(0)
}

/// Reset peak memory statistics for a specific device.
/// Equivalent to `torch.cuda.reset_peak_memory_stats()`.
pub fn gpu_reset_peak_stats_idx(device_index: i32) {
    unsafe { ffi::flodl_gpu_reset_peak_stats(device_index) }
}

/// Reset peak memory statistics for device 0.
pub fn gpu_reset_peak_stats() {
    gpu_reset_peak_stats_idx(0)
}

/// Release all unused cached memory from the GPU caching allocator.
/// Equivalent to `torch.cuda.empty_cache()`.
pub fn gpu_empty_cache() {
    unsafe { ffi::flodl_gpu_empty_cache() }
}

/// Query GPU utilization percentage (0-100) via NVML.
/// Returns `None` if NVML is not available or the query fails.
pub fn gpu_utilization() -> Option<u32> {
    gpu_utilization_idx(0)
}

/// Query GPU utilization percentage for a specific device (0-100) via NVML.
///
/// `device_index` is the PHYSICAL device index (nvidia-smi / NVML
/// enumeration): NVML ignores `CUDA_VISIBLE_DEVICES`, so passing a CUDA
/// runtime index queries the wrong card in scoped-down processes. Use
/// [`crate::sys::GpuInfo::index`] to resolve physical indices.
///
/// Never initializes the CUDA runtime in the calling process; safe to
/// call before `Trainer::run`.
pub fn gpu_utilization_idx(device_index: i32) -> Option<u32> {
    let val = unsafe { ffi::flodl_gpu_utilization(device_index) };
    if val >= 0 { Some(val as u32) } else { None }
}

/// Query device-wide VRAM usage `(used_bytes, total_bytes)` via NVML.
///
/// Reports the same device-global numbers as [`gpu_memory_info_idx`]
/// but WITHOUT creating a CUDA context in the calling process
/// (`cudaMemGetInfo` initializes a primary context on the queried
/// device, pinning VRAM for the life of the process; NVML does not).
/// Use this from monitoring paths and any process that must not touch
/// the CUDA runtime (launcher, pre-`Trainer::run`).
///
/// `physical_index` is the PHYSICAL device index (nvidia-smi / NVML
/// enumeration, independent of `CUDA_VISIBLE_DEVICES`); see
/// [`crate::sys::GpuInfo::index`].
///
/// Returns `None` if NVML is unavailable or the index is invalid.
pub fn gpu_smi_memory_info_idx(physical_index: i32) -> Option<(u64, u64)> {
    let mut used: u64 = 0;
    let mut total: u64 = 0;
    let rc = unsafe { ffi::flodl_gpu_smi_mem_info(physical_index, &mut used, &mut total) };
    if rc == 0 { Some((used, total)) } else { None }
}

/// Whether this process already holds a CUDA primary context on the
/// device (CUDA runtime index).
///
/// Queries driver context state without creating one. Context-dependent
/// reads (caching-allocator stats such as [`gpu_allocated_bytes_idx`])
/// are only meaningful in a process that has actually used the device;
/// gate them on this so a monitoring call never initializes CUDA as a
/// side effect. Always `false` on CPU-only builds and before the first
/// CUDA operation on the device.
pub fn gpu_has_primary_context(device_index: i32) -> bool {
    unsafe { ffi::flodl_gpu_has_primary_context(device_index) != 0 }
}

/// Set the current GPU device.
pub fn set_current_gpu_device(device_index: u8) {
    unsafe { ffi::flodl_set_current_device(device_index as i32) };
}

/// Get the current GPU device index.
pub fn current_gpu_device() -> u8 {
    unsafe { ffi::flodl_get_current_device() as u8 }
}

/// Synchronize a GPU device (wait for all pending work to complete).
pub fn gpu_synchronize(device_index: u8) {
    unsafe { ffi::flodl_gpu_synchronize(device_index as i32) };
}

/// Returns the GPU device name for the given index (e.g. "NVIDIA GeForce GTX 1060 6GB").
pub fn gpu_device_name_idx(device: i32) -> Option<String> {
    // `c_char` (i8 on x86_64, u8 on Linux aarch64) so the FFI buffer
    // type matches on every platform.
    let mut buf = [0 as std::ffi::c_char; 256];
    let err = unsafe { ffi::flodl_gpu_device_name(device, buf.as_mut_ptr(), 256) };
    if err.is_null() {
        let name = unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Some(name)
    } else {
        unsafe { ffi::flodl_free_string(err) };
        None
    }
}

/// Returns the GPU device name for device 0 (e.g. "NVIDIA GeForce GTX 1060 6GB").
pub fn gpu_device_name() -> Option<String> {
    gpu_device_name_idx(0)
}

/// Information about a CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index (0-based).
    pub index: u8,
    /// Device name (e.g. "NVIDIA GeForce GTX 1060 6GB").
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Compute capability major version (e.g. 6 for sm_61).
    pub sm_major: u32,
    /// Compute capability minor version (e.g. 1 for sm_61).
    pub sm_minor: u32,
}

impl DeviceInfo {
    /// Compute capability as a string (e.g. "sm_61").
    pub fn sm_version(&self) -> String {
        format!("sm_{}{}", self.sm_major, self.sm_minor)
    }
}

/// Architecture of a GPU device, as the vendor names it: `"sm_120"` on
/// NVIDIA, `"gfx1100"` (possibly with feature qualifiers, e.g.
/// `"gfx90a:sramecc+:xnack-"`) on AMD.
///
/// Read from the GPU runtime, so it describes the device *libtorch*
/// sees. [`crate::sys::GpuInfo::arch_label`] answers the same question
/// without initializing the runtime, which is the right call before
/// `Trainer::run`; the two can disagree under visibility masks and index
/// remapping, so prefer this one when the question is "why did a kernel
/// launch fail on this device".
///
/// Returns `None` if no GPU is available or the index is invalid.
///
/// This is the portable replacement for [`cuda_compute_capability`],
/// which stays NVIDIA-only on purpose: a gfx target is major/minor/step
/// with the step in hex, and only major+minor survive a numeric pair, so
/// gfx906/gfx908/gfx90a (and gfx1100/1101/1102, and gfx1200/1201) all
/// collapse together despite needing different code objects.
pub fn gpu_arch_name(device_index: i32) -> Option<String> {
    let mut buf = [0 as std::ffi::c_char; 256];
    let err = unsafe { ffi::flodl_gpu_arch_name(device_index, buf.as_mut_ptr(), 256) };
    if err.is_null() {
        Some(
            unsafe { CStr::from_ptr(buf.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        )
    } else {
        unsafe { ffi::flodl_free_string(err) };
        None
    }
}

/// Query compute capability (major, minor) for a CUDA device.
///
/// Returns `None` if CUDA is unavailable or the device index is invalid.
pub fn cuda_compute_capability(device_index: i32) -> Option<(u32, u32)> {
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    let err = unsafe {
        ffi::flodl_cuda_compute_capability(device_index, &mut major, &mut minor)
    };
    if err.is_null() {
        Some((major as u32, minor as u32))
    } else {
        unsafe { ffi::flodl_free_string(err) };
        None
    }
}

/// Enumerate all available GPU devices.
pub fn gpu_devices() -> Vec<DeviceInfo> {
    let n = gpu_device_count();
    (0..n).filter_map(|i| {
        let name = gpu_device_name_idx(i)?;
        let total_memory = gpu_memory_info_idx(i).map(|(_, t)| t).unwrap_or(0);
        let (sm_major, sm_minor) = cuda_compute_capability(i).unwrap_or((0, 0));
        Some(DeviceInfo { index: i as u8, name, total_memory, sm_major, sm_minor })
    }).collect()
}

/// Probe whether a CUDA device can execute compute kernels under the
/// current libtorch build. Returns `Ok(())` if the device works, or an
/// error describing why it cannot (e.g. missing kernel image for sm_61).
pub fn probe_device(device: Device) -> Result<()> {
    let idx = match device {
        Device::CUDA(i) => i,
        Device::CPU => return Ok(()),
    };
    let opts = TensorOptions { dtype: DType::Float32, device };
    match Tensor::zeros(&[1], opts) {
        Ok(t) => {
            // Also try a simple op to verify kernels load
            let _ = t.add(&t)?;
            Ok(())
        }
        Err(e) => {
            let msg = format!("{}", e);
            if msg.contains("no kernel image") {
                let name = gpu_device_name_idx(idx as i32)
                    .unwrap_or_else(|| format!("CUDA({})", idx));
                // `cuda_compute_capability` is NVIDIA-only and returns
                // `None` on a ROCm build, so the sm_/cu### advice below
                // is emitted only where it actually means something.
                // Claiming "sm_00 ... switch to cu0" on an AMD card would
                // send the reader after the wrong fix entirely.
                match cuda_compute_capability(idx as i32) {
                    Some((sm_maj, sm_min)) => Err(TensorError::new(&format!(
                        "CUDA({}) {} (sm_{}{}) cannot run kernels in this libtorch build. \
                         Recommended: switch to libtorch {} \
                         (in Dockerfile, change the cu### variant)",
                        idx, name, sm_maj, sm_min, recommended_cuda_variant(sm_maj)
                    ))),
                    // No compute capability means a non-NVIDIA build, so
                    // name the architecture the vendor's own way instead
                    // of leaving the reader to go find it.
                    None => Err(TensorError::new(&format!(
                        "CUDA({}) {} ({}) cannot run kernels in this libtorch build. \
                         Recommended: install or rebuild a libtorch carrying kernels \
                         for this architecture (`fdl probe` reports it, and \
                         libtorch/<variant>/.arch lists what the build covers)",
                        idx,
                        name,
                        gpu_arch_name(idx as i32).unwrap_or_else(|| "unknown arch".into())
                    ))),
                }
            } else {
                Err(e)
            }
        }
    }
}

/// Return all GPU devices that can run compute kernels, with warnings
/// for any excluded devices printed to stderr.
///
/// This is the primary entry point for multi-GPU setup. It probes each
/// GPU and returns only the working ones, giving actionable diagnostics
/// for any that fail.
pub fn usable_gpu_devices() -> Vec<Device> {
    if !gpu_available() {
        return vec![];
    }
    let devices = gpu_devices();
    let mut usable = Vec::new();

    for info in &devices {
        let dev = Device::CUDA(info.index);
        match probe_device(dev) {
            Ok(()) => usable.push(dev),
            Err(e) => {
                eprintln!("[flodl] WARNING: {}", e);
            }
        }
    }

    if usable.len() < devices.len() {
        let names: Vec<String> = usable.iter().map(|d| format!("{}", d)).collect();
        crate::verbose!(
            "[flodl] Proceeding with {}/{} devices: [{}]",
            usable.len(), devices.len(), names.join(", ")
        );
    }

    usable
}

/// Recommend the best libtorch CUDA variant for a given compute capability.
fn recommended_cuda_variant(sm_major: u32) -> &'static str {
    match sm_major {
        0..=6 => "cu126",  // Maxwell/Pascal (sm_50-sm_61): cu126 for broadest compat
        _ => "cu128",      // Volta+ (sm_70+): cu128 for best performance
    }
}

/// One-line hardware summary for dashboard headers.
///
/// Returns something like:
/// `"CPU: AMD Ryzen 9 5900X (64GB) | GPU: NVIDIA GeForce GTX 1060 (6GB)"`
///
/// GPU identity comes from [`crate::sys::detect_gpus`] (nvidia-smi,
/// honors `CUDA_VISIBLE_DEVICES`), so this never initializes the CUDA
/// runtime in the calling process. `Monitor::new` calls it before
/// `Trainer::run` resolves roles; a CUDA touch here would violate the
/// no-CUDA-before-`Trainer::run` invariant in every user binary that
/// constructs a monitor at the top of its training code.
pub fn hardware_summary() -> String {
    let cpu = cpu_model_name().unwrap_or_else(|| "Unknown CPU".into());
    let threads = cpu_thread_count();
    let ram = total_ram_gb();
    let mut s = format!("{} ({} threads, {}GB)", cpu, threads, ram);

    if cfg!(feature = "gpu") {
        for gpu in crate::sys::detect_gpus() {
            let _ = std::fmt::Write::write_fmt(&mut s, format_args!(
                " | {} ({}GB)", gpu.name, gpu.total_memory_mb / 1024
            ));
        }
    }
    s
}

/// Count logical CPU threads from /proc/cpuinfo (Linux).
fn cpu_thread_count() -> usize {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .map(|s| s.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(1)
}

/// Read CPU model name from /proc/cpuinfo (Linux).
fn cpu_model_name() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in info.lines() {
        if line.starts_with("model name") && let Some(val) = line.split(':').nth(1) {
            return Some(val.trim().to_string());
        }
    }
    None
}

/// Total physical RAM in GB (Linux).
fn total_ram_gb() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            for line in s.lines() {
                if line.starts_with("MemTotal:") {
                    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
                    return Some(kb / (1024 * 1024));
                }
            }
            None
        })
        .unwrap_or(0)
}

/// Enable or disable cuDNN benchmark mode.
///
/// When enabled, cuDNN will benchmark multiple convolution algorithms
/// on the first call and cache the fastest. Benefits fixed-size workloads
/// (FBRL, fixed image dims) with 5-10% speedup. Can hurt dynamic-shape
/// workloads due to warmup cost. Off by default — users opt in.
pub fn set_cudnn_benchmark(enable: bool) {
    unsafe { ffi::flodl_set_cudnn_benchmark(enable as i32) }
}

/// Seed all libtorch RNGs (CPU + CUDA) for reproducible tensor ops.
///
/// This sets the global seed for `Tensor::rand`, `Tensor::randn`,
/// dropout masks, and all other libtorch random operations.
/// Call before model creation and training for full reproducibility.
pub fn manual_seed(seed: u64) {
    unsafe { ffi::flodl_manual_seed(seed) }
}

/// Seed all CUDA device RNGs. No-op when built without CUDA.
///
/// Usually you want `manual_seed()` instead, which seeds both CPU
/// and CUDA. Use this only when you need to re-seed CUDA independently.
pub fn gpu_manual_seed_all(seed: u64) {
    unsafe { ffi::flodl_gpu_manual_seed_all(seed) }
}

/// Ask glibc to return free memory to the OS (Linux only).
///
/// Returns `true` if memory was actually released. Useful for
/// distinguishing allocator fragmentation from real leaks:
/// if RSS drops after calling this, the growth was fragmentation.
pub fn malloc_trim() -> bool {
    unsafe { ffi::flodl_malloc_trim() != 0 }
}

/// Number of live C++ Tensor handles (created but not yet dropped).
/// If this grows over time during training, there is a handle leak.
/// If it stays stable but RSS grows, the leak is inside libtorch.
pub fn live_tensor_count() -> u64 {
    LIVE_TENSOR_COUNT.load(Ordering::Relaxed)
}

/// Read current process RSS in kilobytes (Linux only).
/// Returns 0 on non-Linux or if /proc/self/statm is unreadable.
pub fn rss_kb() -> usize {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<usize>().ok())
        .map(|pages| pages * 4)
        .unwrap_or(0)
}
