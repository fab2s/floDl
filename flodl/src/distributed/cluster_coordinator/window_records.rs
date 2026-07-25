//! Per-window monitor records: the coordinator's sub-epoch view of the
//! cohort, shaped as the path-keyed record tree
//! (`.design/monitoring-portal-b3.md`).
//!
//! Metrics historically reach the dashboard once per **epoch**, which is a
//! single point for a one-epoch LLM run. At each reduce boundary the
//! coordinator already holds everything a per-rank leaf needs — steps (the
//! realized-work weight), the window's mean loss, the marginal delivered
//! rate — so a sub-epoch record costs no new wire traffic.
//!
//! This module is the **pure** shaping step: per-rank window stats in, one
//! [`NodeRecord`] tree out (`root → host → rank`, or `root → rank` when the
//! cohort is single-host, or root-only for a lone rank). Gathering the stats
//! off the live ledger and emitting the tree stays in the coordinator.
//!
//! Absent ≠ zero: a rank that reported no batch this window contributes no
//! `loss` / `throughput` key at all rather than a false `0.0`, so the
//! work-weighted rollup skips it instead of dragging the cohort mean down.

use std::collections::BTreeMap;

use crate::monitor::record::{build_tree, Leaf, NodeRecord, Reductions, Res};

/// One rank's window slice, as read off the coordinator's ledger.
#[derive(Debug, Clone, Default)]
pub(super) struct WindowRankStat {
    /// Global rank index.
    pub rank: usize,
    /// Host this rank runs on (empty when the topology is unknown).
    pub host: String,
    /// Physical device index, when known.
    pub device: Option<u8>,
    /// Whether the rank is currently alive (dead ranks still appear, so the
    /// tree shows the loss rather than silently shrinking).
    pub alive: bool,
    /// Steps completed this window — the linear realized-work weight.
    pub steps: usize,
    /// Mean training loss over the window's batches; `None` if it reported none.
    pub mean_loss: Option<f64>,
    /// Marginal delivered throughput (samples/ms); `None` without a sample.
    pub throughput: Option<f64>,
    /// Compute-only wall (ms) accumulated this window.
    pub compute_only_ms: f64,
}

/// Shape per-rank window stats into the path-keyed record tree.
///
/// Work = `steps` (samples-proportional under a uniform batch size, so it is
/// an exact weight for the work-weighted `Mean`). `batch_share` is each rank's
/// fraction of the cohort's steps, so it sums back to ~1 at the root.
///
/// Tiering: a host level appears only when the cohort spans **more than one**
/// host — a single-host rig gets `root → rank`, and a lone rank collapses to
/// root-only (the root *is* the leaf). Every shape renders identically because
/// each node carries the same field set.
pub(super) fn build_window_tree(stats: &[WindowRankStat]) -> NodeRecord {
    let total_steps: usize = stats.iter().map(|s| s.steps).sum();
    let multi_host = {
        let mut hosts: Vec<&str> =
            stats.iter().map(|s| s.host.as_str()).filter(|h| !h.is_empty()).collect();
        hosts.sort_unstable();
        hosts.dedup();
        hosts.len() > 1
    };
    let root_only = stats.len() == 1 && !multi_host;

    let leaves: Vec<Leaf> = stats
        .iter()
        .map(|s| {
            let mut metrics = BTreeMap::new();
            // Absent stays absent — never zero-filled.
            if let Some(l) = s.mean_loss {
                metrics.insert("loss".to_string(), l);
            }
            if let Some(t) = s.throughput {
                metrics.insert("throughput".to_string(), t);
            }
            if s.compute_only_ms > 0.0 {
                metrics.insert("compute_only_ms".to_string(), s.compute_only_ms);
            }
            if total_steps > 0 {
                metrics.insert(
                    "batch_share".to_string(),
                    s.steps as f64 / total_steps as f64,
                );
            }
            let path = if root_only {
                Vec::new()
            } else {
                let mut p = Vec::new();
                if multi_host {
                    p.push(s.host.clone());
                }
                p.push(format!("rank{}", s.rank));
                p
            };
            Leaf {
                path,
                work: s.steps as f64,
                metrics,
                res: Res::default(),
                device: s.device,
                alive: s.alive,
            }
        })
        .collect();

    build_tree(&leaves, &Reductions::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(rank: usize, host: &str, steps: usize, loss: Option<f64>) -> WindowRankStat {
        WindowRankStat {
            rank,
            host: host.to_string(),
            device: Some(rank as u8),
            alive: true,
            steps,
            mean_loss: loss,
            throughput: Some(10.0),
            compute_only_ms: 5.0,
        }
    }

    fn metric(n: &NodeRecord, k: &str) -> Option<f64> {
        n.metrics.get(k).copied()
    }

    #[test]
    fn single_rank_single_host_is_root_only() {
        let root = build_window_tree(&[stat(0, "h1", 4, Some(0.5))]);
        assert!(root.is_leaf());
        assert_eq!(root.path, vec!["root"]);
        assert_eq!(metric(&root, "loss"), Some(0.5));
        assert_eq!(root.work, 4.0);
    }

    #[test]
    fn single_host_multi_rank_skips_the_host_tier() {
        let root = build_window_tree(&[
            stat(0, "h1", 3, Some(0.2)),
            stat(1, "h1", 1, Some(0.6)),
        ]);
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].path, vec!["root", "rank0"]);
        // Work-weighted loss: (0.2*3 + 0.6*1)/4 = 0.3
        assert!((metric(&root, "loss").unwrap() - 0.3).abs() < 1e-12);
        // batch_share sums back to 1 at the root.
        assert!((metric(&root, "batch_share").unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(root.work, 4.0);
    }

    #[test]
    fn multi_host_gets_a_host_tier_and_stays_exact() {
        let root = build_window_tree(&[
            stat(0, "h1", 2, Some(0.10)),
            stat(1, "h1", 3, Some(0.40)),
            stat(2, "h2", 5, Some(0.90)),
        ]);
        assert_eq!(root.children.len(), 2); // h1, h2
        let h1 = root
            .children
            .iter()
            .find(|c| c.path.last().unwrap() == "h1")
            .unwrap();
        assert_eq!(h1.children.len(), 2);
        assert_eq!(h1.work, 5.0);
        // Hierarchical == flat: (0.10*2 + 0.40*3 + 0.90*5) / 10 = 0.59
        assert!((metric(&root, "loss").unwrap() - 0.59).abs() < 1e-12);
        assert_eq!(root.work, 10.0);
    }

    #[test]
    fn silent_rank_is_absent_not_zero() {
        // rank1 reported no batch this window (loss absent, zero steps).
        let mut silent = stat(1, "h1", 0, None);
        silent.throughput = None;
        silent.compute_only_ms = 0.0;
        let root = build_window_tree(&[stat(0, "h1", 4, Some(0.8)), silent]);
        // The cohort loss is rank0's alone — NOT halved by a phantom zero.
        assert_eq!(metric(&root, "loss"), Some(0.8));
        // The silent leaf carries no loss key at all.
        let r1 = root
            .children
            .iter()
            .find(|c| c.path.last().unwrap() == "rank1")
            .unwrap();
        assert_eq!(metric(r1, "loss"), None);
        assert_eq!(r1.work, 0.0);
    }

    #[test]
    fn dead_rank_still_appears_and_marks_the_tree() {
        let mut dead = stat(1, "h1", 2, Some(0.4));
        dead.alive = false;
        let root = build_window_tree(&[stat(0, "h1", 2, Some(0.2)), dead]);
        assert_eq!(root.children.len(), 2);
        let r1 = root
            .children
            .iter()
            .find(|c| c.path.last().unwrap() == "rank1")
            .unwrap();
        assert!(!r1.alive);
        // Root is alive while any child is.
        assert!(root.alive);
    }

    #[test]
    fn idle_window_has_no_batch_share_and_zero_work() {
        // Every rank silent (a fully idle window): no division by zero, and
        // batch_share is absent rather than NaN.
        let mut a = stat(0, "h1", 0, None);
        a.throughput = None;
        a.compute_only_ms = 0.0;
        let mut b = stat(1, "h1", 0, None);
        b.throughput = None;
        b.compute_only_ms = 0.0;
        let root = build_window_tree(&[a, b]);
        assert_eq!(root.work, 0.0);
        assert_eq!(metric(&root, "batch_share"), None);
        assert_eq!(metric(&root, "loss"), None);
    }

    #[test]
    fn throughput_sums_and_device_rides_the_leaf() {
        let root = build_window_tree(&[
            stat(0, "h1", 2, Some(0.2)),
            stat(1, "h1", 2, Some(0.2)),
        ]);
        // throughput is a core Sum.
        assert_eq!(metric(&root, "throughput"), Some(20.0));
        let r1 = root
            .children
            .iter()
            .find(|c| c.path.last().unwrap() == "rank1")
            .unwrap();
        assert_eq!(r1.device, Some(1));
    }
}
