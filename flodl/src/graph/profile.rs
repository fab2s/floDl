use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use super::Graph;
use super::trend::{Trend, TrendGroup};
use crate::tensor::{Device, GpuEvent, GpuEventFlags};

/// Which clock produced a [`Profile`]'s timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileSource {
    /// Host `Instant` around each node call. Honest on CPU; on a GPU it
    /// would time the kernel *launch*, not the execution (launches are
    /// async), which is why CUDA forwards use [`ProfileSource::GpuEvents`].
    HostWallClock,
    /// Device-side boundary events (CUDA/ROCm): the elapsed device time
    /// between the events bracketing each node, i.e. actual execution.
    GpuEvents,
}

impl ProfileSource {
    /// Short label for display surfaces (DOT legend, `Display`).
    pub fn label(&self) -> &'static str {
        match self {
            ProfileSource::HostWallClock => "host wall clock",
            ProfileSource::GpuEvents => "gpu events",
        }
    }
}

/// Per-node execution time from a single Forward pass.
#[derive(Clone, Debug)]
pub struct NodeTiming {
    pub id: String,
    pub tag: String,
    pub duration: Duration,
    pub level: usize,
}

/// Per-level execution time. Multi-node levels could theoretically
/// benefit from parallelism — `parallelism()` measures efficiency.
#[derive(Clone, Debug)]
pub struct LevelTiming {
    pub index: usize,
    pub wall_clock: Duration,
    pub sum_nodes: Duration,
    pub num_nodes: usize,
}

impl LevelTiming {
    /// Ratio of sequential node time to wall-clock time.
    /// Values above 1.0 indicate effective parallelism.
    /// Returns 1.0 for single-node levels.
    pub fn parallelism(&self) -> f64 {
        if self.wall_clock.is_zero() || self.num_nodes <= 1 {
            return 1.0;
        }
        self.sum_nodes.as_secs_f64() / self.wall_clock.as_secs_f64()
    }
}

/// Timing data from a single Forward pass.
#[derive(Clone, Debug)]
pub struct Profile {
    pub total: Duration,
    pub levels: Vec<LevelTiming>,
    pub nodes: Vec<NodeTiming>,
    /// Which clock produced these timings.
    pub source: ProfileSource,
}

impl Profile {
    /// Duration of a tagged node, or zero if not found.
    pub fn timing(&self, tag: &str) -> Duration {
        for n in &self.nodes {
            if n.tag == tag {
                return n.duration;
            }
        }
        Duration::ZERO
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Forward: {:?} ({} levels, {} nodes; {})",
            self.total,
            self.levels.len(),
            self.nodes.len(),
            self.source.label()
        )?;

        let mut node_idx = 0;
        for level in &self.levels {
            write!(f, "\n  Level {}  {:?}", level.index, level.wall_clock)?;
            if level.num_nodes > 1 {
                write!(
                    f,
                    "  {} nodes  x{:.1}",
                    level.num_nodes,
                    level.parallelism()
                )?;
            }
            writeln!(f)?;

            while node_idx < self.nodes.len() && self.nodes[node_idx].level == level.index {
                let n = &self.nodes[node_idx];
                let mut label = n.id.clone();
                if !n.tag.is_empty() {
                    label += &format!(" {:?}", n.tag);
                }
                writeln!(f, "    {:<40} {:?}", label, n.duration)?;
                node_idx += 1;
            }
        }

        Ok(())
    }
}

// --- GPU-event profiling (device-side timing) ---

/// GPU profiling state, created lazily on the first profiled CUDA forward
/// and torn down by `disable_profiling`. `Failed` records that event setup
/// or readback errored (no driver, device mismatch, ...) so the host-clock
/// fallback warns once instead of every pass.
pub(crate) enum GpuProfState {
    Unused,
    Active(GpuProfilePool),
    Failed,
}

/// Pooled boundary events for device-side profiling of one graph.
///
/// One marker before the first node plus one after each node: node `i`'s
/// duration is the elapsed device time between events `i` and `i+1`. The
/// deltas telescope, so the node sum equals the pass total (device-idle
/// gaps land inside the adjacent node's span: the launch-bound signal).
/// Events re-record freely, so one set serves every pass: a pass is
/// recorded, left `pending` while the device drains, and resolved into a
/// [`Profile`] before the next recording (or on an explicit read).
pub(crate) struct GpuProfilePool {
    device: Device,
    events: Vec<GpuEvent>,
    /// `(id, tag, level)` per executed node, in execution order. The
    /// execution plan is immutable after build, so this is captured once.
    node_meta: Vec<(String, String, usize)>,
    /// Executed node count per level, mapping levels onto event ranges.
    level_sizes: Vec<usize>,
    /// The events hold a recorded, not-yet-resolved pass.
    pending: bool,
}

impl GpuProfilePool {
    fn new(graph: &Graph, device: Device) -> crate::tensor::Result<Self> {
        let tags_by_node = graph.tags_by_node();
        let mut node_meta = Vec::new();
        let mut level_sizes = Vec::with_capacity(graph.levels.len());
        for (level_idx, level) in graph.levels.iter().enumerate() {
            level_sizes.push(level.len());
            for &ni in level {
                node_meta.push((
                    graph.nodes[ni].id.clone(),
                    tags_by_node.get(&ni).cloned().unwrap_or_default(),
                    level_idx,
                ));
            }
        }
        let mut events = Vec::with_capacity(node_meta.len() + 1);
        for _ in 0..node_meta.len() + 1 {
            events.push(GpuEvent::new(GpuEventFlags::Default)?);
        }
        Ok(GpuProfilePool {
            device,
            events,
            node_meta,
            level_sizes,
            pending: false,
        })
    }

    /// Record boundary event `idx` on the current stream.
    pub(crate) fn record(&self, idx: usize) -> crate::tensor::Result<()> {
        self.events[idx].record()
    }

    /// Mark the just-recorded pass as awaiting resolution.
    pub(crate) fn mark_pending(&mut self) {
        self.pending = true;
    }

    /// Whether the pending pass has fully executed (non-blocking).
    fn complete(&self) -> bool {
        self.events[self.node_meta.len()].is_complete()
    }

    /// Read the pending pass's deltas into a Profile. Waits on the final
    /// boundary event, a no-op when the pass has already drained.
    fn resolve(&mut self) -> crate::tensor::Result<Profile> {
        let n = self.node_meta.len();
        self.events[n].synchronize()?;
        self.pending = false;

        let mut nodes = Vec::with_capacity(n);
        for (i, (id, tag, level)) in self.node_meta.iter().enumerate() {
            let ms = GpuEvent::elapsed_time(&self.events[i], &self.events[i + 1])?;
            nodes.push(NodeTiming {
                id: id.clone(),
                tag: tag.clone(),
                duration: Duration::from_secs_f64(f64::from(ms.max(0.0)) / 1_000.0),
                level: *level,
            });
        }

        let mut levels = Vec::with_capacity(self.level_sizes.len());
        let mut start = 0usize;
        for (index, &len) in self.level_sizes.iter().enumerate() {
            let sum: Duration = nodes[start..start + len].iter().map(|nt| nt.duration).sum();
            levels.push(LevelTiming {
                index,
                // Nodes run sequentially on one stream, so a level's wall
                // clock IS its node sum (parallelism() reads 1.0).
                wall_clock: sum,
                sum_nodes: sum,
                num_nodes: len,
            });
            start += len;
        }

        let total_ms = GpuEvent::elapsed_time(&self.events[0], &self.events[n])?;
        Ok(Profile {
            total: Duration::from_secs_f64(f64::from(total_ms.max(0.0)) / 1_000.0),
            levels,
            nodes,
            source: ProfileSource::GpuEvents,
        })
    }
}

// --- Graph profiling methods ---

impl Graph {
    /// Turn on per-node and per-level timing for subsequent forward calls.
    pub fn enable_profiling(&self) {
        self.profiling.set(true);
    }

    /// Turn off timing. Subsequent forward calls have zero profiling overhead.
    pub fn disable_profiling(&self) {
        self.profiling.set(false);
        *self.last_profile.borrow_mut() = None;
        *self.gpu_prof.borrow_mut() = GpuProfState::Unused;
    }

    /// Reverse tag lookup: node_idx → first tag name.
    pub(crate) fn tags_by_node(&self) -> HashMap<usize, String> {
        let mut m = HashMap::new();
        for (name, &(ni, _)) in &self.tag_names {
            m.entry(ni).or_insert_with(|| name.clone());
        }
        m
    }

    /// Make the event pool ready for `device`, returning whether the GPU
    /// path is usable. Any pending pass must be resolved first (a device
    /// change rebuilds the pool, dropping its events). A failed setup
    /// falls back to the host clock and warns once.
    pub(crate) fn ensure_gpu_profile_pool(&self, device: Device) -> bool {
        let mut state = self.gpu_prof.borrow_mut();
        match &*state {
            GpuProfState::Active(pool) if pool.device == device => true,
            GpuProfState::Failed => false,
            _ => match GpuProfilePool::new(self, device) {
                Ok(pool) => {
                    *state = GpuProfState::Active(pool);
                    true
                }
                Err(e) => {
                    eprintln!(
                        "  warning: graph profiling could not set up GPU events \
                         ({e}); timings fall back to the host wall clock"
                    );
                    *state = GpuProfState::Failed;
                    false
                }
            },
        }
    }

    /// Resolve a pending GPU-event pass into `last_profile`.
    ///
    /// `block` forces the read (waits on the final boundary event);
    /// otherwise the pass is resolved only if the device has already
    /// drained it, keeping reads on the training path sync-free.
    pub(crate) fn resolve_gpu_profile(&self, block: bool) {
        let mut state = self.gpu_prof.borrow_mut();
        if let GpuProfState::Active(pool) = &mut *state {
            if !pool.pending || (!block && !pool.complete()) {
                return;
            }
            match pool.resolve() {
                Ok(p) => {
                    *self.last_profile.borrow_mut() = Some(p);
                    self.profile_collected.set(false);
                }
                Err(e) => {
                    eprintln!(
                        "  warning: graph profiling could not read GPU events \
                         ({e}); timings fall back to the host wall clock"
                    );
                    *state = GpuProfState::Failed;
                }
            }
        }
    }

    /// Whether profiling is currently enabled.
    pub fn profiling(&self) -> bool {
        self.profiling.get()
    }

    /// Timing data from the most recent forward call, or None.
    ///
    /// On CUDA, timings come from device-side events and resolve one pass
    /// behind the forward calls; a read serves the freshest drained pass
    /// and only waits on the device when there is nothing else to serve
    /// (the first read after a single profiled pass).
    pub fn profile(&self) -> Option<Profile> {
        let nothing_to_serve = self.last_profile.borrow().is_none();
        self.resolve_gpu_profile(nothing_to_serve);
        self.last_profile.borrow().clone()
    }

    /// Duration of a tagged node from the most recent forward call.
    pub fn timing(&self, tag: &str) -> Duration {
        let nothing_to_serve = self.last_profile.borrow().is_none();
        self.resolve_gpu_profile(nothing_to_serve);
        self.last_profile
            .borrow()
            .as_ref()
            .map(|p| p.timing(tag))
            .unwrap_or(Duration::ZERO)
    }

    /// Snapshot tagged node durations into the timing batch buffer.
    /// If tags is empty, all tagged nodes with timing data are collected.
    pub fn collect_timings(&self, tags: &[&str]) {
        let profile = self.last_profile.borrow();
        let profile = match profile.as_ref() {
            Some(p) => p,
            None => return,
        };
        let mut buffer = self.timing_buffer.borrow_mut();

        if tags.is_empty() {
            for n in &profile.nodes {
                if !n.tag.is_empty() {
                    buffer
                        .entry(n.tag.clone())
                        .or_default()
                        .push(n.duration.as_secs_f64());
                }
            }
        } else {
            for &tag in tags {
                let d = profile.timing(tag);
                if !d.is_zero() {
                    buffer
                        .entry(tag.to_string())
                        .or_default()
                        .push(d.as_secs_f64());
                }
            }
        }
    }

    /// Compute batch mean, append to timing epoch history, clear buffer.
    /// If tags is empty, flushes all buffered tags.
    pub fn flush_timings(&self, tags: &[&str]) {
        let mut buffer = self.timing_buffer.borrow_mut();
        let mut history = self.timing_history.borrow_mut();

        let keys: Vec<String> = if tags.is_empty() {
            buffer.keys().cloned().collect()
        } else {
            tags.iter().map(|t| t.to_string()).collect()
        };

        for key in &keys {
            if let Some(values) = buffer.remove(key)
                && !values.is_empty()
            {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                history.entry(key.clone()).or_default().push(mean);
            }
        }
    }

    /// Epoch-level trend over the timing history of a tagged node.
    /// Values are mean execution times in seconds.
    pub fn timing_trend(&self, tag: &str) -> Trend {
        let history = self.timing_history.borrow();
        Trend::new(history.get(tag).cloned().unwrap_or_default())
    }

    /// TrendGroup for timing trends of the given tags (expands groups).
    pub fn timing_trends(&self, tags: &[&str]) -> TrendGroup {
        let expanded = self.expand_groups(tags);
        let history = self.timing_history.borrow();
        let trends = expanded
            .iter()
            .map(|tag| Trend::new(history.get(tag).cloned().unwrap_or_default()))
            .collect();
        TrendGroup(trends)
    }

    /// Clear timing epoch history. If tags is empty, clears all.
    pub fn reset_timing_trend(&self, tags: &[&str]) {
        let mut history = self.timing_history.borrow_mut();
        if tags.is_empty() {
            history.clear();
        } else {
            for tag in tags {
                history.remove(*tag);
            }
        }
    }
}
