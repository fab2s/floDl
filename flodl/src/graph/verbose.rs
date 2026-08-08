//! Verbose build-time output for graph tree inspection.

use super::{Graph, GraphExt};
use crate::nn::Module;
use std::fmt::Write;

impl Graph {
    /// Generate a tree summary showing subgraph structure, tags, and parameter counts.
    pub fn tree_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== Graph Tree ===");
        self.write_tree_node(&mut out, "", true);
        let _ = writeln!(out);
        let _ = writeln!(out, "{}", self.param_summary());
        out
    }

    fn write_tree_node(&self, out: &mut String, indent: &str, is_root: bool) {
        let name = self.label().unwrap_or("(root)");
        let hash = &self.structural_hash()[..8];
        let param_count: usize = self.parameters().len();
        let frozen_count: usize = self.parameters().iter().filter(|p| p.is_frozen()).count();

        if is_root {
            let _ = writeln!(out, "{} [hash: {}]", name, hash);
        }

        // List tags (non-internal only)
        let mut tags: Vec<&str> = self
            .tag_names
            .keys()
            .filter(|t| !self.internal_tags.contains(*t))
            .map(|s| s.as_str())
            .collect();
        tags.sort();

        if !tags.is_empty() {
            let _ = writeln!(out, "{}+-- tags: {}", indent, tags.join(", "));
        }

        // Show param count with frozen info
        if frozen_count > 0 && frozen_count == param_count {
            let _ = writeln!(out, "{}+-- params: {} [frozen]", indent, param_count);
        } else if frozen_count > 0 {
            let _ = writeln!(
                out,
                "{}+-- params: {} ({} frozen)",
                indent, param_count, frozen_count
            );
        } else if param_count > 0 {
            let _ = writeln!(out, "{}+-- params: {}", indent, param_count);
        }

        // Recurse into children
        let mut child_entries: Vec<(String, usize)> = self
            .children
            .iter()
            .map(|(label, &ni)| (label.clone(), ni))
            .collect();
        child_entries.sort_by_key(|(_, ni)| *ni);

        for (i, (label, ni)) in child_entries.iter().enumerate() {
            let is_last = i == child_entries.len() - 1;
            let child_indent = if is_last {
                format!("{}    ", indent)
            } else {
                format!("{}|   ", indent)
            };

            if let Some(ref module) = self.nodes[*ni].module
                && let Some(child_graph) = module.as_graph()
            {
                let child_hash = &child_graph.structural_hash()[..8];
                let child_params = child_graph.parameters().len();
                let child_frozen = child_graph
                    .parameters()
                    .iter()
                    .filter(|p| p.is_frozen())
                    .count();

                let frozen_str = if child_frozen > 0 && child_frozen == child_params {
                    " * frozen"
                } else {
                    ""
                };
                let _ = writeln!(
                    out,
                    "{}+-- {} [hash: {}]{}",
                    indent, label, child_hash, frozen_str
                );
                child_graph.write_tree_node(out, &child_indent, false);
            }
        }
    }

    /// Per-subgraph parameter count breakdown.
    pub fn param_summary(&self) -> String {
        let mut out = String::new();
        let total: usize = self.parameters().len();
        let _ = writeln!(out, "=== Parameter Summary ===");
        let _ = writeln!(out, "Total: {} parameters", total);

        if !self.children.is_empty() {
            let mut child_entries: Vec<(String, usize)> = self
                .children
                .iter()
                .map(|(label, &ni)| (label.clone(), ni))
                .collect();
            child_entries.sort_by_key(|(_, ni)| *ni);

            let mut accounted = 0usize;
            for (label, ni) in &child_entries {
                if let Some(ref module) = self.nodes[*ni].module
                    && let Some(child_graph) = module.as_graph()
                {
                    let count = child_graph.parameters().len();
                    let frozen = child_graph
                        .parameters()
                        .iter()
                        .filter(|p| p.is_frozen())
                        .count();
                    let pct = if total > 0 {
                        count as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    };
                    let frozen_str = if frozen == count && count > 0 {
                        "  frozen".to_string()
                    } else if frozen > 0 {
                        format!("  {}/{} frozen", frozen, count)
                    } else {
                        "  trainable".to_string()
                    };
                    let _ = writeln!(out, "  {}: {} ({:.1}%){}", label, count, pct, frozen_str);
                    accounted += count;
                }
            }
            let own = total.saturating_sub(accounted);
            if own > 0 {
                let pct = if total > 0 {
                    own as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                let _ = writeln!(out, "  (own): {} ({:.1}%)  trainable", own, pct);
            }
        }
        out
    }
}
