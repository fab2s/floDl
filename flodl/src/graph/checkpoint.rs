//! Graph checkpoint save/load, structural hashing, and source-config sidecar.
//!
//! Split from `graph.rs`: parameter/buffer persistence, the lazily-cached
//! structural hash used for architecture validation on load, the opaque
//! source-config passthrough, and the sidecar-path helper. `Graph`'s fields
//! are `pub(crate)`, so this `impl Graph` block needs no visibility change.
//! `named_parameters`/`named_buffers` (the introspection these call) stay in
//! `graph.rs` next to the struct.

use std::path::PathBuf;

use hmac_sha256::Hash as Sha256;

use super::graph::Graph;
use crate::tensor::{Result, TensorError};

impl Graph {
    /// Save all parameters and buffers to a checkpoint file.
    ///
    /// Embeds the structural hash for architecture validation on load.
    /// Supports `.gz` extension for gzip compression.
    ///
    /// When [`source_config`](Self::source_config) is set on the graph
    /// (e.g. populated by `flodl_hf::models::auto::AutoModel::from_pretrained`),
    /// a sidecar `<stem>.config.json` file is written next to the
    /// checkpoint with the source config verbatim. The stem strips both
    /// `.fdl` and an optional `.gz` so `model.fdl.gz` produces
    /// `model.config.json`. Downstream tools (e.g. `fdl flodl-hf export
    /// --checkpoint`) use this to rebuild the right family without an
    /// explicit `--config` argument.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let params = self.named_parameters();
        let buffers = self.named_buffers();
        let hash = self.structural_hash();
        crate::nn::save_checkpoint_file(path, &params, &buffers, Some(hash))?;

        if let Some(content) = self.source_config.borrow().as_ref() {
            let sidecar = sidecar_config_path(path);
            std::fs::write(&sidecar, content).map_err(|e| {
                TensorError::new(&format!(
                    "save_checkpoint: cannot write sidecar {}: {e}",
                    sidecar.display()
                ))
            })?;
        }
        Ok(())
    }

    /// Attach an opaque source-config string to the graph.
    ///
    /// Typically called by loaders that build a graph from an external
    /// definition (HF `config.json`, ONNX manifest, …) so subsequent
    /// `save_checkpoint` calls drop a `<stem>.config.json` sidecar.
    /// Pass an empty string to attach a non-meaningful sentinel; pass
    /// `clear_source_config` to drop attachment entirely.
    pub fn set_source_config(&self, content: String) {
        *self.source_config.borrow_mut() = Some(content);
    }

    /// Read the currently-attached source config, if any.
    pub fn source_config(&self) -> Option<String> {
        self.source_config.borrow().clone()
    }

    /// Drop any attached source config so subsequent saves emit no
    /// sidecar.
    pub fn clear_source_config(&self) {
        *self.source_config.borrow_mut() = None;
    }

    /// Load parameters and buffers from a checkpoint file.
    ///
    /// Validates the structural hash against this graph's architecture.
    /// Returns a [`LoadReport`](crate::nn::LoadReport) describing what was
    /// loaded, skipped, or missing.
    pub fn load_checkpoint(&self, path: &str) -> Result<crate::nn::LoadReport> {
        let params = self.named_parameters();
        let buffers = self.named_buffers();
        let hash = self.structural_hash();
        crate::nn::load_checkpoint_file(path, &params, &buffers, Some(hash))
    }

    // pub(crate): the `structural_hash()` accessor in `graph.rs` calls this.
    pub(crate) fn compute_structural_hash(&self) -> String {
        let mut hasher = Sha256::new();

        // 1. Nodes in topological order
        for &ni in &self.order {
            let node = &self.nodes[ni];
            hasher.update(node.id.as_bytes());
            hasher.update(b"\0");

            if let Some(ref module) = node.module {
                hasher.update(module.name().as_bytes());
                hasher.update(b"\0");

                // Sorted parameters
                let mut params: Vec<_> = module
                    .parameters()
                    .into_iter()
                    .map(|p| (p.name.clone(), p.variable.shape()))
                    .collect();
                params.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, shape) in &params {
                    hasher.update(b"P");
                    hasher.update(name.as_bytes());
                    hasher.update(b"\0");
                    for &dim in shape {
                        hasher.update(dim.to_le_bytes());
                    }
                }

                // Sorted buffers
                let mut bufs: Vec<_> = module
                    .buffers()
                    .into_iter()
                    .map(|b| (b.name.clone(), b.shape()))
                    .collect();
                bufs.sort_by(|a, b| a.0.cmp(&b.0));
                for (name, shape) in &bufs {
                    hasher.update(b"B");
                    hasher.update(name.as_bytes());
                    hasher.update(b"\0");
                    for &dim in shape {
                        hasher.update(dim.to_le_bytes());
                    }
                }

                // Nested graph hash
                if let Some(nested_hash) = module.structural_hash() {
                    hasher.update(b"G");
                    hasher.update(nested_hash.as_bytes());
                }
            }
        }

        // 2. Edges
        hasher.update(b"EDGES");
        for edge in &self.edges {
            hasher.update(edge.from_node.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.from_port.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.to_node.as_bytes());
            hasher.update(b"\0");
            hasher.update(edge.to_port.as_bytes());
            hasher.update(b"\0");
        }

        // 3. Tags (sorted)
        hasher.update(b"TAGS");
        let mut tags: Vec<_> = self.tag_names.iter().collect();
        tags.sort_by(|a, b| a.0.cmp(b.0));
        for (name, (node_idx, port_idx)) in &tags {
            hasher.update(name.as_bytes());
            hasher.update(b"\0");
            hasher.update((*node_idx as u64).to_le_bytes());
            hasher.update((*port_idx as u64).to_le_bytes());
        }

        // 4. Input/output ports
        hasher.update(b"INPUTS");
        for port in &self.inputs {
            hasher.update(port.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.node_id.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.port.as_bytes());
            hasher.update(b"\0");
        }
        hasher.update(b"OUTPUTS");
        for port in &self.outputs {
            hasher.update(port.name.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.node_id.as_bytes());
            hasher.update(b"\0");
            hasher.update(port.port.as_bytes());
            hasher.update(b"\0");
        }

        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// Derive the sidecar config-json path for a checkpoint path.
///
/// Strips a trailing `.gz` (compression marker) if present, then
/// replaces the extension with `config.json`. Stem is preserved
/// verbatim so paths like `/some/dir/v3.fdl` map to
/// `/some/dir/v3.config.json`.
pub(crate) fn sidecar_config_path(checkpoint: &str) -> PathBuf {
    let mut p = PathBuf::from(checkpoint);
    if p.extension().and_then(|e| e.to_str()) == Some("gz") {
        // Drop the .gz to expose the inner extension (.fdl, .ckpt, ...).
        p.set_extension("");
    }
    p.set_extension("config.json");
    p
}
