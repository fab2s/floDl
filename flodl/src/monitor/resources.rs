//! System resource sampling: CPU, RAM, GPU memory, GPU utilization.
//!
//! CPU and RAM are read from `/proc/stat` and `/proc/meminfo` (Linux only).
//! GPU metrics use the FFI layer (CUDA + NVML).

use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Per-device GPU snapshot.
#[derive(Debug, Clone, Default)]
pub struct GpuSnapshot {
    /// CUDA device index.
    pub device_index: u8,
    /// Device name (e.g., "NVIDIA GeForce RTX 5060 Ti").
    pub name: String,
    /// GPU utilization percentage (0-100), None if NVML unavailable.
    pub util_percent: Option<f32>,
    /// Bytes reserved by the CUDA caching allocator.
    pub vram_allocated_bytes: Option<u64>,
    /// Total physical VRAM in bytes.
    pub vram_total_bytes: Option<u64>,
}

/// A snapshot of system resource usage.
#[derive(Debug, Clone, Default)]
pub struct ResourceSample {
    /// CPU utilization percentage (0-100), None if unavailable.
    pub cpu_percent: Option<f32>,
    /// RAM used by the system in bytes.
    pub ram_used_bytes: Option<u64>,
    /// Total system RAM in bytes.
    pub ram_total_bytes: Option<u64>,
    /// GPU utilization percentage (0-100) for the rank in `aggregate_rank`,
    /// None if NVML unavailable.
    pub gpu_util_percent: Option<f32>,
    /// Total physical VRAM in bytes for the rank in `aggregate_rank`.
    pub vram_total_bytes: Option<u64>,
    /// Bytes reserved by the CUDA caching allocator on the rank in
    /// `aggregate_rank`.
    pub vram_allocated_bytes: Option<u64>,
    /// CUDA device index whose data populates the aggregate fields above.
    /// Picked uniformly at random per `sample()` call so that, over the
    /// run, each rank is observed roughly equally without paying O(N)
    /// per-tick aggregation cost. Per-GPU detail is preserved in `gpus`.
    pub aggregate_rank: Option<u8>,
    /// Per-GPU snapshots (empty on CPU builds).
    pub gpus: Vec<GpuSnapshot>,
}

impl ResourceSample {
    /// Format a compact resource summary string.
    ///
    /// Example: `"CPU: 45% | RAM: 3.2/7.8 GB | GPU: 82% | VRAM: 2.1 GB / 0 KB"`
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();

        if let Some(cpu) = self.cpu_percent {
            parts.push(format!("CPU: {:.0}%", cpu));
        }
        if let (Some(used), Some(total)) = (self.ram_used_bytes, self.ram_total_bytes) {
            parts.push(format!(
                "RAM: {}/{}",
                super::format::format_bytes(used),
                super::format::format_bytes(total),
            ));
        }

        if self.gpus.len() > 1 {
            // Multi-GPU: show per-device VRAM
            for gpu in &self.gpus {
                if let Some(alloc) = gpu.vram_allocated_bytes {
                    let spill = match gpu.vram_total_bytes {
                        Some(total) if alloc > total => alloc - total,
                        _ => 0,
                    };
                    let util = gpu.util_percent.map(|u| format!(" ({:.0}%)", u)).unwrap_or_default();
                    parts.push(format!(
                        "GPU{}: {} / {}{}",
                        gpu.device_index,
                        super::format::format_bytes(alloc),
                        super::format::format_bytes(spill),
                        util,
                    ));
                }
            }
        } else {
            // Single GPU or CPU
            if let Some(gpu) = self.gpu_util_percent {
                parts.push(format!("GPU: {:.0}%", gpu));
            }
            if let Some(alloc) = self.vram_allocated_bytes {
                let spill = match self.vram_total_bytes {
                    Some(total) if alloc > total => alloc - total,
                    _ => 0,
                };
                parts.push(format!(
                    "VRAM: {} / {}",
                    super::format::format_bytes(alloc),
                    super::format::format_bytes(spill),
                ));
            }
        }

        parts.join(" | ")
    }
}

/// Accumulated CPU jiffies from `/proc/stat`.
#[derive(Clone)]
pub(super) struct CpuTimes {
    pub(super) total: u64,
    pub(super) idle: u64,
}

/// Per-device GPU utilization accumulator for background polling.
struct GpuUtilAccum {
    sum: Vec<f64>,
    count: Vec<u32>,
}

/// Handle for the background GPU utilization poller thread.
struct GpuPollerHandle {
    accum: Arc<Mutex<GpuUtilAccum>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for GpuPollerHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

/// Stateful resource sampler. Maintains previous CPU reading for delta
/// computation and a background thread for GPU utilization averaging.
///
/// GPU utilization is polled via NVML every ~1 second in a background
/// thread. When `sample()` is called, it returns the average over the
/// interval since the last call. This prevents point-in-time sampling
/// artifacts in heterogeneous DDP (where the fast GPU may be idle at
/// epoch boundaries, giving misleadingly low single-sample readings).
pub struct ResourceSampler {
    prev_cpu: Option<CpuTimes>,
    gpu_poller: Option<GpuPollerHandle>,
}

impl Default for ResourceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSampler {
    /// Create a new sampler, capturing an initial CPU reading for the
    /// first delta and starting a background GPU utilization poller.
    pub fn new() -> Self {
        let prev_cpu = read_cpu_times();
        let gpu_poller = Self::start_gpu_poller();
        Self { prev_cpu, gpu_poller }
    }

    /// Start a background thread that polls NVML utilization every ~1 second.
    /// Returns `None` if no CUDA devices are available.
    fn start_gpu_poller() -> Option<GpuPollerHandle> {
        let n = crate::tensor::cuda_device_count();
        if n <= 0 {
            return None;
        }
        let n = n as usize;
        let accum = Arc::new(Mutex::new(GpuUtilAccum {
            sum: vec![0.0; n],
            count: vec![0; n],
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let accum2 = accum.clone();
        let stop2 = stop.clone();
        let thread = thread::Builder::new()
            .name("gpu-util-poller".into())
            .spawn(move || {
                while !stop2.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_secs(1));
                    if stop2.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok(mut acc) = accum2.lock() {
                        for i in 0..n {
                            if let Some(util) = crate::tensor::cuda_utilization_idx(i as i32) {
                                acc.sum[i] += util as f64;
                                acc.count[i] += 1;
                            }
                        }
                    }
                }
            })
            .ok()?;
        Some(GpuPollerHandle {
            accum,
            stop,
            thread: Some(thread),
        })
    }

    /// Take a snapshot of current system resources (CPU, RAM, GPU, VRAM).
    ///
    /// CPU utilization is computed as a delta since the previous call.
    /// GPU utilization is averaged over background samples since the last
    /// call (falls back to an instant NVML sample if no background data).
    /// Fields that cannot be read on this platform are `None`.
    pub fn sample(&mut self) -> ResourceSample {
        let mut s = ResourceSample::default();

        // CPU utilization (delta between two readings)
        if let Some(current) = read_cpu_times() {
            if let Some(ref prev) = self.prev_cpu {
                let d_total = current.total.saturating_sub(prev.total);
                let d_idle = current.idle.saturating_sub(prev.idle);
                if d_total > 0 {
                    s.cpu_percent = Some(
                        (d_total.saturating_sub(d_idle) as f32 / d_total as f32) * 100.0,
                    );
                }
            }
            self.prev_cpu = Some(current);
        }

        // RAM from /proc/meminfo
        if let Some((used, total)) = read_meminfo() {
            s.ram_used_bytes = Some(used);
            s.ram_total_bytes = Some(total);
        }

        // Drain background GPU utilization averages
        let n = crate::tensor::cuda_device_count();
        let util_averages: Vec<Option<f32>> = if let Some(ref poller) = self.gpu_poller {
            if let Ok(mut acc) = poller.accum.lock() {
                let avgs: Vec<Option<f32>> = acc.sum.iter().zip(acc.count.iter())
                    .map(|(&s, &c)| if c > 0 { Some((s / c as f64) as f32) } else { None })
                    .collect();
                // Reset for next interval
                for s in acc.sum.iter_mut() { *s = 0.0; }
                for c in acc.count.iter_mut() { *c = 0; }
                avgs
            } else {
                vec![None; n as usize]
            }
        } else {
            vec![None; n as usize]
        };

        // Per-GPU snapshots
        for i in 0..n {
            let mut gpu = GpuSnapshot {
                device_index: i as u8,
                ..Default::default()
            };
            if let Some(name) = crate::tensor::cuda_device_name_idx(i) {
                gpu.name = name;
            }
            if let Ok((_, total)) = crate::tensor::cuda_memory_info_idx(i) {
                gpu.vram_total_bytes = Some(total);
            }
            if let Ok(alloc) = crate::tensor::cuda_allocated_bytes_idx(i) {
                gpu.vram_allocated_bytes = Some(alloc);
            }
            // Background average if available, else instant NVML sample
            gpu.util_percent = util_averages.get(i as usize).copied().flatten()
                .or_else(|| crate::tensor::cuda_utilization_idx(i).map(|u| u as f32));
            s.gpus.push(gpu);
        }

        // Aggregate fields: pick one GPU uniformly at random and copy its
        // VRAM + util into the top-level fields. Over many ticks the
        // displayed values cover every rank ergodically; each tick pays
        // O(1) instead of O(world_size). Per-GPU detail stays in `s.gpus`
        // for the timeline CSV / dashboard so this only affects the
        // compact summary surface (epoch log line, summary string).
        if !s.gpus.is_empty() {
            let pick = crate::rng::Rng::from_entropy().usize(s.gpus.len());
            let g = &s.gpus[pick];
            s.aggregate_rank = Some(g.device_index);
            s.vram_total_bytes = g.vram_total_bytes;
            s.vram_allocated_bytes = g.vram_allocated_bytes;
            s.gpu_util_percent = g.util_percent;
        }

        s
    }
}

/// Parse `/proc/stat` first line for CPU jiffies.
pub(super) fn read_cpu_times() -> Option<CpuTimes> {
    let content = fs::read_to_string("/proc/stat").ok()?;
    let line = content.lines().next()?;
    if !line.starts_with("cpu ") {
        return None;
    }
    let fields: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    if fields.len() < 4 {
        return None;
    }
    // Fields: user, nice, system, idle, iowait, irq, softirq, steal, ...
    let total: u64 = fields.iter().sum();
    let idle = fields[3] + fields.get(4).copied().unwrap_or(0); // idle + iowait
    Some(CpuTimes { total, idle })
}

/// Parse `/proc/meminfo` for total and available memory.
pub(super) fn read_meminfo() -> Option<(u64, u64)> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    let mut total: Option<u64> = None;
    let mut available: Option<u64> = None;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_kb_value(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_kb_value(rest);
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }

    match (total, available) {
        (Some(t), Some(a)) => Some((t.saturating_sub(a), t)),
        _ => None,
    }
}

/// Parse a value like "  16384000 kB" into bytes.
pub(super) fn parse_kb_value(s: &str) -> Option<u64> {
    let val: u64 = s.split_whitespace().next()?.parse().ok()?;
    Some(val * 1024) // kB to bytes
}
