use std::collections::HashMap;
use std::fmt::Write;
use std::process::Command;

use crate::tensor::TensorError;

use super::Graph;
use super::profile::Profile;

impl Graph {
    /// Graphviz DOT representation of the graph structure.
    ///
    /// Nodes are grouped by execution level, shaped by type (input/output/activation/norm),
    /// and annotated with parameter counts. Render with:
    /// `dot -Tsvg graph.dot -o graph.svg`
    ///
    /// ```ignore
    /// std::fs::write("graph.dot", g.dot()).unwrap();
    /// ```
    pub fn dot(&self) -> String {
        self.build_dot(None)
    }

    /// Timing-annotated DOT using the most recent [`Profile`].
    /// Nodes are color-coded green → yellow → red by relative execution time,
    /// with per-node microsecond timings and per-level wall clock + parallelism.
    pub fn dot_with_profile(&self) -> String {
        // Serve the freshest drained GPU pass; block only when there is
        // nothing else to serve (e.g. teardown right after a single pass).
        let nothing_to_serve = self.last_profile.borrow().is_none();
        self.resolve_gpu_profile(nothing_to_serve);
        let profile = self.last_profile.borrow().clone();
        self.build_dot(profile.as_ref())
    }

    /// Render the structural graph as SVG via the Graphviz `dot` binary.
    ///
    /// Requires Graphviz installed (`apt install graphviz` / `brew install graphviz`).
    /// Optionally writes to `path` if provided. Returns the raw SVG bytes.
    ///
    /// ```ignore
    /// g.svg(Some("graph.svg"))?;
    /// ```
    pub fn svg(&self, path: Option<&str>) -> crate::tensor::Result<Vec<u8>> {
        render_svg(&self.dot(), path)
    }

    /// Render the timing-annotated graph as SVG via the Graphviz `dot` binary.
    ///
    /// Requires a prior profiled forward pass. Optionally writes to `path`.
    pub fn svg_with_profile(&self, path: Option<&str>) -> crate::tensor::Result<Vec<u8>> {
        render_svg(&self.dot_with_profile(), path)
    }

    fn build_dot(&self, profile: Option<&Profile>) -> String {
        let mut b = String::with_capacity(4096);
        b.push_str("digraph G {\n");
        b.push_str("  rankdir=TB;\n");
        b.push_str("  fontname=\"Helvetica\";\n");
        b.push_str("  node [fontname=\"Helvetica\" fontsize=11 style=filled];\n");
        b.push_str("  edge [fontname=\"Helvetica\" fontsize=9];\n");
        b.push_str("  compound=true;\n\n");

        // Build timing lookups.
        let mut node_timings: HashMap<&str, f64> = HashMap::new();
        let mut max_duration: f64 = 0.0;
        if let Some(p) = profile {
            for n in &p.nodes {
                let secs = n.duration.as_secs_f64();
                node_timings.insert(&n.id, secs);
                if secs > max_duration {
                    max_duration = secs;
                }
            }
        }

        // Build reverse tag lookup: node_idx → tag names.
        let mut node_tags: HashMap<usize, Vec<String>> = HashMap::new();
        for (name, &(ni, _)) in &self.tag_names {
            node_tags.entry(ni).or_default().push(name.clone());
        }

        // Identify input/output nodes.
        let mut input_nodes: HashMap<usize, bool> = HashMap::new();
        let mut output_nodes: HashMap<usize, bool> = HashMap::new();
        for ep in &self.inputs {
            if let Some(&ni) = self.node_index.get(&ep.node_id) {
                input_nodes.insert(ni, true);
            }
        }
        for ep in &self.outputs {
            if let Some(&ni) = self.node_index.get(&ep.node_id) {
                output_nodes.insert(ni, true);
            }
        }

        // Emit nodes grouped by level.
        for (i, level) in self.levels.iter().enumerate() {
            let _ = writeln!(b, "  subgraph cluster_level_{} {{", i);

            // Level label with optional timing.
            if let Some(p) = profile
                && let Some(lt) = p.levels.get(i)
            {
                let mut label = format!("level {}  {:.0?}", i, lt.wall_clock);
                if lt.num_nodes > 1 {
                    let _ = write!(label, "  x{:.1}", lt.parallelism());
                }
                let _ = writeln!(b, "    label=\"{}\";", label);
            } else {
                let _ = writeln!(b, "    label=\"level {}\";", i);
            }
            b.push_str("    style=dashed; color=\"#999999\"; fontcolor=\"#999999\";\n");
            b.push_str("    rank=same;\n");

            for &ni in level {
                let node = &self.nodes[ni];
                let tags = node_tags.get(&ni).cloned().unwrap_or_default();
                let mut label = node_label(&node.id, &tags);
                let is_input = input_nodes.contains_key(&ni);
                let is_output = output_nodes.contains_key(&ni);
                let (shape, mut fill) = node_style(&node.id, is_input, is_output);

                // Annotate with timing.
                if let Some(&secs) = node_timings.get(node.id.as_str()) {
                    let _ = write!(label, "\\n{}", format_duration_us(secs));
                    if max_duration > 0.0 {
                        fill = heat_color(secs / max_duration);
                    }
                }

                // Parameter count.
                if let Some(ref module) = node.module {
                    let params = module.parameters();
                    let count: i64 = params
                        .iter()
                        .map(|p| p.variable.shape().iter().product::<i64>())
                        .sum();
                    if count > 0 {
                        let _ = write!(label, "\\n[{}]", format_count(count));
                    }
                }

                let _ = writeln!(
                    b,
                    "    \"{}\" [label=\"{}\" shape={} fillcolor=\"{}\"];",
                    node.id, label, shape, fill
                );
            }
            b.push_str("  }\n\n");
        }

        // Emit edges.
        for edge in &self.edges {
            let (style, color, elabel) = edge_style(edge);
            let mut attrs = format!("style={} color=\"{}\"", style, color);
            if !elabel.is_empty() {
                let _ = write!(attrs, " label=\"{}\" fontcolor=\"{}\"", elabel, color);
            }
            let _ = writeln!(
                b,
                "  \"{}\" -> \"{}\" [{}];",
                edge.from_node, edge.to_node, attrs
            );
        }

        // Emit forward-ref state loops.
        for entry in &self.state {
            let writer_id = &self.nodes[entry.writer_ni].id;
            let _ = writeln!(
                b,
                "  \"{}\" -> \"state_read_{}\" [style=dotted color=\"#e67e22\" label=\"state\" fontcolor=\"#e67e22\" constraint=false];",
                writer_id, writer_id
            );
        }

        // Total timing as graph label, with the clock that produced it.
        if let Some(p) = profile {
            let _ = writeln!(b);
            let _ = writeln!(
                b,
                "  label=\"Forward: {:.0?} ({})\";",
                p.total,
                p.source.label()
            );
            let _ = writeln!(b, "  labelloc=t;");
            let _ = writeln!(b, "  fontsize=14;");
        }

        b.push_str("}\n");
        b
    }
}

/// Strip trailing _N counter from node ID for display.
fn clean_id(id: &str) -> &str {
    if let Some(pos) = id.rfind('_') {
        let suffix = &id[pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return &id[..pos];
        }
    }
    id
}

/// Build a human-readable label for a node.
fn node_label(id: &str, tags: &[String]) -> String {
    let mut label = clean_id(id).to_string();
    if !tags.is_empty() {
        label += "\\n";
        label += &tags
            .iter()
            .map(|t| format!("#{}", t))
            .collect::<Vec<_>>()
            .join(" ");
    }
    label
}

/// Returns (shape, fill_color) based on node type.
fn node_style(id: &str, is_input: bool, is_output: bool) -> (&'static str, String) {
    match (is_input, is_output) {
        (true, true) => return ("doubleoctagon", "#aed6f1".into()),
        (true, false) => return ("invhouse", "#aed6f1".into()),
        (false, true) => return ("house", "#a9dfbf".into()),
        _ => {}
    }

    let prefix = clean_id(id);
    if id.starts_with("state_read_") {
        ("diamond", "#f9e79f".into())
    } else if prefix == "add" || prefix == "merge" {
        ("circle", "#d5dbdb".into())
    } else if prefix == "map" {
        ("parallelogram", "#a9cce3".into())
    } else if prefix == "loop" {
        ("box3d", "#d7bde2".into())
    } else if prefix == "switch" || prefix == "gate" {
        ("diamond", "#f5cba7".into())
    } else if is_activation(prefix) {
        ("ellipse", "#fdebd0".into())
    } else if is_norm(prefix) {
        ("box", "#d5f5e3".into())
    } else {
        ("box", "#eaecee".into())
    }
}

fn edge_style(edge: &super::node::Edge) -> (&'static str, &'static str, String) {
    if edge.to_port.starts_with("ref_") {
        let name = edge.to_port.strip_prefix("ref_").unwrap_or("");
        return ("dashed", "#2980b9", name.to_string());
    }
    ("solid", "#2c3e50", String::new())
}

fn is_activation(prefix: &str) -> bool {
    matches!(prefix, "relu" | "sigmoid" | "tanh" | "gelu" | "silu")
}

fn is_norm(prefix: &str) -> bool {
    matches!(prefix, "layernorm" | "batchnorm" | "rmsnorm")
}

/// Interpolate green (#27ae60) → yellow (#f39c12) → red (#e74c3c).
/// Shared with the cluster heat-map bake (`graph::heatmap`), so the
/// solo profiled SVG and the aggregated one read on the same scale.
pub(crate) fn heat_color(ratio: f64) -> String {
    let ratio = ratio.clamp(0.0, 1.0);
    let (r, g, b) = if ratio < 0.5 {
        let t = ratio * 2.0;
        (
            0x27 as f64 + t * (0xf3 - 0x27) as f64,
            0xae as f64 + t * (0x9c_u8 as f64 - 0xae as f64),
            0x60 as f64 + t * (0x12 as f64 - 0x60 as f64),
        )
    } else {
        let t = (ratio - 0.5) * 2.0;
        (
            0xf3 as f64 + t * (0xe7 as f64 - 0xf3 as f64),
            0x9c as f64 + t * (0x4c as f64 - 0x9c as f64),
            0x12 as f64 + t * (0x3c as f64 - 0x12 as f64),
        )
    };
    format!("#{:02x}{:02x}{:02x}", r as u8, g as u8, b as u8)
}

/// Format parameter count with K/M suffixes.
fn format_count(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M params", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K params", n as f64 / 1_000.0)
    } else {
        format!("{} params", n)
    }
}

/// Format seconds as human-readable microsecond duration.
fn format_duration_us(secs: f64) -> String {
    let us = secs * 1_000_000.0;
    if us < 1000.0 {
        format!("{:.0}us", us)
    } else {
        format!("{:.2}ms", us / 1000.0)
    }
}

/// Pipe a DOT string through `dot -Tsvg` and optionally write to a file.
fn render_svg(dot: &str, path: Option<&str>) -> crate::tensor::Result<Vec<u8>> {
    let dot_bin = which_dot()?;

    let mut child = Command::new(&dot_bin)
        .arg("-Tsvg")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| TensorError::new(&format!("failed to spawn {}: {}", dot_bin, e)))?;

    // Write DOT to stdin
    {
        use std::io::Write as IoWrite;
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(dot.as_bytes())
            .map_err(|e| TensorError::new(&format!("failed to write DOT to stdin: {}", e)))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| TensorError::new(&format!("dot process failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TensorError::new(&format!(
            "dot exited with {}: {}",
            output.status, stderr
        )));
    }

    if let Some(p) = path {
        std::fs::write(p, &output.stdout)
            .map_err(|e| TensorError::new(&format!("failed to write {}: {}", p, e)))?;
    }

    Ok(output.stdout)
}

/// Locate the `dot` binary, returning an error with install instructions if missing.
fn which_dot() -> crate::tensor::Result<String> {
    if let Ok(output) = Command::new("which").arg("dot").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    Err(TensorError::new(
        "Graphviz `dot` binary not found. Install with:\n  \
         apt install graphviz   (Debian/Ubuntu)\n  \
         brew install graphviz  (macOS)\n\n\
         Or use .dot() to get the DOT string and render online at:\n  \
         https://dreampuf.github.io/GraphvizOnline",
    ))
}
