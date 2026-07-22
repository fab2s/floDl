//! Graph tree: hierarchical subgraph composition with label-path addressing.
//!
//! When a labeled [`Graph`] is used inside a [`FlowBuilder`](super::FlowBuilder),
//! the parent registers it as a child subgraph. Dot-separated label paths
//! (`"encoder.scan.hidden"`) address subgraphs and tags across boundaries.
//!
//! All operations are build-time or explicit-query-time. The forward path is untouched.
//!
//! # Key methods on [`Graph`]
//!
//! - **Navigation**: [`tree_children()`](Graph::tree_children), [`child_graph()`](Graph::child_graph),
//!   [`subgraph()`](Graph::subgraph), [`is_composed()`](Graph::is_composed)
//! - **Parameters**: [`parameters_at()`](Graph::parameters_at), [`named_parameters_at()`](Graph::named_parameters_at)
//! - **Freeze/thaw**: [`freeze()`](Graph::freeze), [`thaw()`](Graph::thaw), [`is_frozen()`](Graph::is_frozen)
//! - **Checkpoints**: [`load_subgraph_checkpoint()`](Graph::load_subgraph_checkpoint)
//! - **Observation**: [`tagged_at()`](Graph::tagged_at), [`collect_at()`](Graph::collect_at),
//!   [`record_at()`](Graph::record_at), [`trend_at()`](Graph::trend_at)

use std::collections::HashMap;
use crate::autograd::Variable;
use crate::nn::{self, Buffer, Module, Parameter};
use crate::tensor::{Result, TensorError};
use super::{Graph, GraphExt};
use super::trend::Trend;

/// What a label path resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// An entire child subgraph.
    Subgraph,
    /// A named tag within a graph.
    Tag,
}

/// Internal resolution result (borrowed references, no ownership).
#[allow(dead_code)]
pub(crate) enum ResolvedPath<'a> {
    /// Path resolves to an entire child subgraph.
    Subgraph(&'a Graph),
    /// Path resolves to a tag within a specific graph.
    Tag { graph: &'a Graph, tag: String },
}

impl Graph {
    // ── Path resolution ──────────────────────────────────────────────

    /// Resolve a dot-separated label path to a subgraph or tag.
    ///
    /// **Strict dot semantics:**
    /// - `"scan"` — local: check children first (Subgraph), then tags (Tag).
    /// - `"letter.scan"` — child `"letter"`, then `"scan"` within it.
    /// - `"letter.scan.location"` — child `"letter"`, child/tag `"scan"`, then `"location"`.
    ///
    /// Returns `Err` if any segment doesn't resolve.
    pub(crate) fn resolve(&self, path: &str) -> Result<ResolvedPath<'_>> {
        if path.is_empty() {
            return Err(TensorError::new("empty label path"));
        }
        let segments: Vec<&str> = path.split('.').collect();
        self.resolve_segments(&segments, path, false)
    }

    fn resolve_segments<'a>(
        &'a self,
        segments: &[&str],
        full_path: &str,
        cross_boundary: bool,
    ) -> Result<ResolvedPath<'a>> {
        debug_assert!(!segments.is_empty());
        let first = segments[0];

        if segments.len() == 1 {
            // Single segment: children take priority over tags
            if let Some(g) = self.child_graph(first) {
                return Ok(ResolvedPath::Subgraph(g));
            }
            if self.tag_names.contains_key(first) {
                // Block internal tags when accessed from outside
                if cross_boundary && self.internal_tags.contains(first) {
                    return Err(TensorError::new(&format!(
                        "tag {:?} is internal and cannot be accessed from a parent graph (path: {:?})",
                        first, full_path
                    )));
                }
                return Ok(ResolvedPath::Tag { graph: self, tag: first.to_string() });
            }
            return Err(TensorError::new(&format!(
                "{:?} is not a subgraph or tag of this graph (path: {:?})",
                first, full_path
            )));
        }

        // Multi-segment: first MUST be a child label
        let child = self.child_graph(first).ok_or_else(|| {
            TensorError::new(&format!(
                "{:?} is not a subgraph of this graph (path: {:?})",
                first, full_path
            ))
        })?;

        // Once we cross into a child, all subsequent resolution is cross-boundary
        child.resolve_segments(&segments[1..], full_path, true)
    }

    // ── Public navigation ────────────────────────────────────────────

    /// Direct children: label -> child graph.
    pub fn tree_children(&self) -> HashMap<&str, &Graph> {
        self.children.iter()
            .filter_map(|(label, &ni)| {
                self.nodes[ni].module.as_ref()
                    .and_then(|m| m.as_graph())
                    .map(|g| (label.as_str(), g))
            })
            .collect()
    }

    /// Get a direct child graph by label (one level only).
    pub fn child_graph(&self, label: &str) -> Option<&Graph> {
        self.children.get(label)
            .and_then(|&ni| self.nodes[ni].module.as_ref())
            .and_then(|m| m.as_graph())
    }

    /// Get a subgraph at any depth via dot-path.
    pub fn subgraph(&self, path: &str) -> Result<&Graph> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g),
            ResolvedPath::Tag { .. } => Err(TensorError::new(&format!(
                "path {:?} resolves to a tag, not a subgraph", path
            ))),
        }
    }

    /// Whether this graph has been composed into a parent graph.
    pub fn is_composed(&self) -> bool {
        self.composed.get()
    }

    /// Tags marked as internal (hidden from parent resolution).
    pub fn internal_tags(&self) -> &std::collections::HashSet<String> {
        &self.internal_tags
    }

    /// Validate that a path resolves, returning what it resolves to.
    pub fn validate_path(&self, path: &str) -> Result<PathKind> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(_) => Ok(PathKind::Subgraph),
            ResolvedPath::Tag { .. } => Ok(PathKind::Tag),
        }
    }

    // ── Parameter operations ─────────────────────────────────────────

    /// All parameters at a label path.
    pub fn parameters_at(&self, path: &str) -> Result<Vec<Parameter>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g.parameters()),
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        Ok(module.parameters())
                    } else {
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Named parameters at a label path, using the target's own namespace.
    /// For subgraphs: delegates to the child graph's `named_parameters()`.
    /// For tags: qualifies with the tag name as prefix.
    pub fn named_parameters_at(&self, path: &str) -> Result<Vec<(String, Parameter)>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g.named_parameters()),
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        Ok(module.parameters().into_iter()
                            .map(|p| (format!("{}/{}", tag, p.name), p))
                            .collect())
                    } else {
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    /// Named buffers at a label path, using the target's own namespace.
    pub fn named_buffers_at(&self, path: &str) -> Result<Vec<(String, Buffer)>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => Ok(g.named_buffers()),
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        Ok(module.buffers().into_iter()
                            .map(|b| (format!("{}/{}", tag, b.name), b))
                            .collect())
                    } else {
                        Ok(vec![])
                    }
                } else {
                    Ok(vec![])
                }
            }
        }
    }

    // ── Freeze / thaw ────────────────────────────────────────────────

    /// Freeze all parameters at the given label path.
    pub fn freeze(&self, path: &str) -> Result<()> {
        for p in self.parameters_at(path)? {
            p.freeze()?;
        }
        Ok(())
    }

    /// Thaw (unfreeze) all parameters at the given label path.
    pub fn thaw(&self, path: &str) -> Result<()> {
        for p in self.parameters_at(path)? {
            p.unfreeze()?;
        }
        Ok(())
    }

    /// Check if all parameters at the path are frozen.
    /// Returns true only if there are parameters and ALL are frozen.
    pub fn is_frozen(&self, path: &str) -> Result<bool> {
        let params = self.parameters_at(path)?;
        if params.is_empty() {
            return Ok(false);
        }
        Ok(params.iter().all(|p| p.is_frozen()))
    }

    // ── Training mode ────────────────────────────────────────────────

    // ── Checkpoint composition ────────────────────────────────────────

    /// Load a checkpoint into a specific subgraph.
    ///
    /// The checkpoint's structural hash is validated against the target
    /// subgraph's hash. Named parameters/buffers are matched within the
    /// subgraph's own namespace.
    pub fn load_subgraph_checkpoint(&self, path: &str, file: &str) -> Result<nn::LoadReport> {
        let target = self.subgraph(path)?;
        let params = target.named_parameters();
        let buffers = target.named_buffers();
        let hash = target.structural_hash();
        nn::load_checkpoint_file(file, &params, &buffers, Some(hash))
    }

    // ── Training mode ────────────────────────────────────────────────

    /// Set training mode on a specific subgraph or tagged module.
    pub fn set_training_at(&self, path: &str, training: bool) -> Result<()> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(g) => {
                g.set_training(training);
            }
            ResolvedPath::Tag { graph, ref tag } => {
                if let Some(&(ni, _)) = graph.tag_names.get(tag.as_str()) {
                    if let Some(ref module) = graph.nodes[ni].module {
                        crate::nn::walk_modules(module.as_ref(), &mut |m| {
                            m.set_training(training);
                        });
                    }
                }
            }
        }
        Ok(())
    }

    // ── Cross-boundary observation ───────────────────────────────────

    /// Get a tagged output by label path.
    /// Returns `Err` if the path doesn't exist (null -- wiring bug).
    /// Returns `Ok(None)` if the path exists but hasn't been computed yet (nil).
    /// Returns `Ok(Some(v))` if the value is available.
    pub fn tagged_at(&self, path: &str) -> Result<Option<Variable>> {
        match self.resolve(path)? {
            ResolvedPath::Subgraph(_) => Err(TensorError::new(&format!(
                "path {:?} resolves to a subgraph, not a tag", path
            ))),
            ResolvedPath::Tag { graph, ref tag } => Ok(graph.tagged(tag)),
        }
    }

    /// Collect metrics from label paths into observation buffers.
    /// Each path must resolve to a tag (not a subgraph).
    /// Metrics are stored in the target graph's batch buffer.
    pub fn collect_at(&self, paths: &[&str]) -> Result<()> {
        for &path in paths {
            match self.resolve(path)? {
                ResolvedPath::Subgraph(_) => {
                    return Err(TensorError::new(&format!(
                        "collect_at: {:?} resolves to a subgraph, not a tag", path
                    )));
                }
                ResolvedPath::Tag { graph, ref tag } => {
                    graph.collect(&[tag.as_str()])?;
                }
            }
        }
        Ok(())
    }

    /// Record a scalar metric at a label path.
    /// For dotted paths, the metric is stored in the target graph's buffer
    /// under the final segment name.
    pub fn record_at(&self, path: &str, value: f64) -> Result<()> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.len() < 2 {
            // Single segment: record into self
            self.record_scalar(path, value);
            return Ok(());
        }
        // Multi-segment: resolve parent graph, record under last segment
        let parent_path = segments[..segments.len() - 1].join(".");
        let tag = segments[segments.len() - 1];
        let target = self.subgraph(&parent_path)?;
        target.record_scalar(tag, value);
        Ok(())
    }

    /// Get trend for a label-path metric.
    /// For dotted paths, reads from the target graph's epoch history.
    pub fn trend_at(&self, path: &str) -> Result<Trend> {
        let segments: Vec<&str> = path.split('.').collect();
        if segments.len() < 2 {
            return Ok(self.trend(path));
        }
        let parent_path = segments[..segments.len() - 1].join(".");
        let tag = segments[segments.len() - 1];
        let target = self.subgraph(&parent_path)?;
        Ok(target.trend(tag))
    }
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
