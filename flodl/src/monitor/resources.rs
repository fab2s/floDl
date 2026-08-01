//! System resource sampling: CPU, RAM, GPU memory, GPU utilization.
//!
//! CPU and RAM are read from `/proc/stat` and `/proc/meminfo` (Linux only).
//! GPU identity (count, names, VRAM totals) comes from
//! [`crate::sys::detect_gpus`] (nvidia-smi) and live metrics from NVML;
//! neither initializes the CUDA runtime. The only CUDA-context-dependent
//! read, caching-allocator reserved bytes, is gated on
//! [`crate::tensor::gpu_has_primary_context`] so that constructing or
//! polling a sampler never creates a CUDA context as a side effect.
//! `Monitor::new` runs in every process, including the pre-fan-out
//! launcher, where touching CUDA would break the
//! no-CUDA-before-`Trainer::run` invariant and pin VRAM on every GPU
//! for the life of the process.

use std::collections::VecDeque;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Background NVML poll interval. Higher rate means denser samples per
/// epoch, so a single barrier-period sample (which can read 0%) gets
/// diluted by surrounding compute samples in the rolling-window mean
/// returned from [`ResourceSampler::sample`]. 250 ms keeps NVML
/// overhead negligible while giving ~13 samples per 3 s epoch.
const GPU_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Rolling-window size for per-device GPU utilization samples. At
/// [`GPU_POLL_INTERVAL`] = 250 ms, 32 entries covers ~8 s of history
/// — long enough to smooth out brief sync-barrier dips, short enough
/// that the displayed average still tracks real workload shifts
/// (mode changes, batch-size sweeps, schedule transitions) within a
/// few seconds rather than blurring the entire run.
const GPU_UTIL_WINDOW: usize = 32;

/// Per-device GPU snapshot.
#[derive(Debug, Clone, Default)]
pub struct GpuSnapshot {
    /// Physical device index (nvidia-smi / NVML enumeration). Matches
    /// the CUDA runtime index when `CUDA_VISIBLE_DEVICES` is unset; in
    /// scoped-down processes it identifies the actual card rather than
    /// the remapped runtime slot.
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

/// Per-device GPU utilization rolling-window accumulator. Background
/// poller pushes a sample per device every [`GPU_POLL_INTERVAL`];
/// [`ResourceSampler::sample`] reads the per-device mean over the last
/// [`GPU_UTIL_WINDOW`] entries without draining the buffer. This
/// dilutes single-sample dips when one poll lands during a sync
/// barrier; with ~32 samples in the window, the per-device mean tracks
/// the between-syncs compute load rather than any individual barrier-
/// period zero.
struct GpuUtilAccum {
    samples: Vec<VecDeque<f32>>,
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

/// Static per-device identity, captured once at sampler construction
/// from [`crate::sys::detect_gpus`] (nvidia-smi). No CUDA runtime
/// involvement, and no per-sample process spawn.
struct GpuStatic {
    /// Physical index (nvidia-smi / NVML enumeration). NVML queries
    /// must use this: NVML ignores `CUDA_VISIBLE_DEVICES`.
    physical_index: u8,
    name: String,
    total_bytes: Option<u64>,
}

/// Enumerate GPUs without touching the CUDA runtime. Position in the
/// returned Vec is used as the CUDA runtime index for context-gated
/// allocator reads (exact when `CUDA_VISIBLE_DEVICES` is unset or an
/// ascending index list; a reordering list like `1,0` would mislabel
/// allocator stats, which nothing in flodl's launch paths produces).
fn detect_gpu_statics() -> Vec<GpuStatic> {
    if !cfg!(feature = "gpu") {
        return Vec::new();
    }
    crate::sys::detect_gpus()
        .into_iter()
        .map(|g| GpuStatic {
            physical_index: g.index,
            total_bytes: Some(g.vram_bytes()),
            name: g.name,
        })
        .collect()
}

/// Stateful resource sampler. Maintains previous CPU reading for delta
/// computation and a background thread for GPU utilization averaging.
///
/// GPU utilization is polled via NVML every ~1 second in a background
/// thread. When `sample()` is called, it returns the average over the
/// interval since the last call. This prevents point-in-time sampling
/// artifacts in heterogeneous DDP (where the fast GPU may be idle at
/// epoch boundaries, giving misleadingly low single-sample readings).
///
/// Construction and sampling never initialize the CUDA runtime (see
/// the module docs); a sampler is safe in any process, launcher
/// included.
pub struct ResourceSampler {
    prev_cpu: Option<CpuTimes>,
    gpus: Vec<GpuStatic>,
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
        let gpus = detect_gpu_statics();
        let gpu_poller = Self::start_gpu_poller(&gpus);
        Self { prev_cpu, gpus, gpu_poller }
    }

    /// Start a background thread that polls NVML utilization at
    /// [`GPU_POLL_INTERVAL`] and feeds a per-device rolling window
    /// (window position i tracks `gpus[i]`, polled by physical index).
    /// Returns `None` if no GPUs are visible.
    fn start_gpu_poller(gpus: &[GpuStatic]) -> Option<GpuPollerHandle> {
        if gpus.is_empty() {
            return None;
        }
        let physical: Vec<u8> = gpus.iter().map(|g| g.physical_index).collect();
        let accum = Arc::new(Mutex::new(GpuUtilAccum {
            samples: physical.iter().map(|_| VecDeque::with_capacity(GPU_UTIL_WINDOW)).collect(),
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let accum2 = accum.clone();
        let stop2 = stop.clone();
        let thread = thread::Builder::new()
            .name("gpu-util-poller".into())
            .spawn(move || {
                while !stop2.load(Ordering::Relaxed) {
                    thread::sleep(GPU_POLL_INTERVAL);
                    if stop2.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok(mut acc) = accum2.lock() {
                        for (i, &phys) in physical.iter().enumerate() {
                            if let Some(util) = crate::tensor::gpu_utilization_idx(phys as i32) {
                                let buf = &mut acc.samples[i];
                                if buf.len() == GPU_UTIL_WINDOW {
                                    buf.pop_front();
                                }
                                buf.push_back(util as f32);
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

        // Read the per-device rolling-window mean WITHOUT draining
        // the buffer: samples persist across `sample()` calls so a
        // short epoch with one sync-period dip averages out against
        // the rest of the window's compute samples. See [`GPU_UTIL_WINDOW`]
        // / [`GPU_POLL_INTERVAL`] for window-sizing rationale.
        let n = self.gpus.len();
        let util_averages: Vec<Option<f32>> = if let Some(ref poller) = self.gpu_poller {
            if let Ok(acc) = poller.accum.lock() {
                acc.samples
                    .iter()
                    .map(|buf| {
                        if buf.is_empty() {
                            None
                        } else {
                            let sum: f32 = buf.iter().sum();
                            Some(sum / buf.len() as f32)
                        }
                    })
                    .collect()
            } else {
                vec![None; n]
            }
        } else {
            vec![None; n]
        };

        // Per-GPU snapshots. Identity from the construction-time
        // nvidia-smi statics, utilization via NVML, allocator stats
        // only where this process already holds a CUDA context.
        for (i, g) in self.gpus.iter().enumerate() {
            let mut gpu = GpuSnapshot {
                device_index: g.physical_index,
                name: g.name.clone(),
                vram_total_bytes: g.total_bytes,
                ..Default::default()
            };
            // Caching-allocator reserved bytes are per-process and take
            // the CUDA runtime index (= position in the visible list).
            // Querying them from a process without a context would
            // CREATE one (pinning VRAM); gate on context presence.
            if crate::tensor::gpu_has_primary_context(i as i32) {
                if let Ok(alloc) = crate::tensor::gpu_allocated_bytes_idx(i as i32) {
                    gpu.vram_allocated_bytes = Some(alloc);
                }
            }
            // Background average if available, else instant NVML sample
            gpu.util_percent = util_averages.get(i).copied().flatten()
                .or_else(|| {
                    crate::tensor::gpu_utilization_idx(g.physical_index as i32)
                        .map(|u| u as f32)
                });
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructing and polling a sampler must never initialize the
    /// CUDA runtime: `Monitor::new` runs in the fan-out launcher, where
    /// a CUDA touch breaks the no-CUDA-before-`Trainer::run` invariant
    /// and a primary context pins VRAM on every device for the life of
    /// the process.
    ///
    /// Only verifiable in a process that has done no CUDA work yet; in
    /// the parallel CUDA test harness another test may already hold a
    /// context, in which case this degrades to a no-op. Run it alone
    /// (`--exact`) for the meaningful negative check.
    #[test]
    fn test_resource_sampler_never_initializes_cuda() {
        if crate::tensor::gpu_has_primary_context(0) {
            eprintln!("skipped: CUDA context already present in this process");
            return;
        }
        let mut sampler = ResourceSampler::new();
        let s = sampler.sample();
        // Give the NVML poller a couple of ticks too.
        std::thread::sleep(Duration::from_millis(550));
        let _ = sampler.sample();
        assert!(
            !crate::tensor::gpu_has_primary_context(0),
            "ResourceSampler must not create a CUDA context"
        );
        // Identity still works without a context: totals come from
        // nvidia-smi, allocator stats stay None.
        for gpu in &s.gpus {
            assert!(gpu.vram_allocated_bytes.is_none());
        }
    }
}
