use std::fmt;
use std::time::Duration;

use super::Graph;
use super::trend::{Trend, TrendGroup};

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
            "Forward: {:?} ({} levels, {} nodes)",
            self.total,
            self.levels.len(),
            self.nodes.len()
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
    }

    /// Whether profiling is currently enabled.
    pub fn profiling(&self) -> bool {
        self.profiling.get()
    }

    /// Timing data from the most recent forward call, or None.
    pub fn profile(&self) -> Option<Profile> {
        self.last_profile.borrow().clone()
    }

    /// Duration of a tagged node from the most recent forward call.
    pub fn timing(&self, tag: &str) -> Duration {
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
