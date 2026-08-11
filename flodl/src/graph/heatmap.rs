//! Bake accumulated node timings into an already-rendered graph SVG.
//!
//! The cluster dashboard's heat map cannot re-run `dot` (ranks shipped
//! the structural SVG precisely because the renderer is not a runtime
//! dependency), but it does not need to: graphviz output is regular
//! enough that recolouring is XML surgery. Every node is a
//! `<g class="node"><title>{id}</title><shape fill="..."/>...</g>`
//! group whose `<title>` is the graph node id, the same id the
//! timing frames are keyed by. The transform sets each node shape's
//! fill from its relative mean, extends the `<title>` so the native
//! browser tooltip carries mean/min, and appends a legend band below
//! the drawing. The output is a finished, self-contained artifact:
//! what the dashboard displays is byte-for-byte what a download saves.

use std::fmt::Write as _;

use super::dot::heat_color;

/// One node's aggregated timing, keyed by the graph node id (the
/// `<title>` of its SVG group).
pub(crate) struct HeatNode {
    pub id: String,
    pub min_ms: f64,
    pub mean_ms: f64,
}

/// Legend metadata for one baked heat map.
pub(crate) struct HeatLegend {
    /// GPU model the timings aggregate over (one heat map per model).
    pub gpu_model: String,
    /// Clock provenance (`ProfileSource::label()` wording).
    pub source: String,
    /// Accumulated passes across the ranks in this group.
    pub samples: u64,
    /// Ranks aggregated into this map.
    pub ranks: usize,
    /// Mean pass total (ms) across the group.
    pub total_mean_ms: f64,
}

/// Minimal XML text escaping for values interpolated into the legend.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn format_ms(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.0}us", ms * 1000.0)
    } else {
        format!("{ms:.2}ms")
    }
}

/// Gutter added to the canvas right edge so rightmost-column nodes have
/// room for their side annotation.
const ANNOTATION_GUTTER: f64 = 76.0;

/// Bake `nodes` into `svg`. Returns `None` when the SVG does not look
/// like graphviz output (no parseable `viewBox`); nodes without a
/// timing entry keep their structural colour, which is itself signal
/// (a node the profile never saw).
pub(crate) fn bake_heatmap(svg: &str, nodes: &[HeatNode], legend: &HeatLegend) -> Option<String> {
    // --- canvas geometry, from the viewBox ------------------------------
    let vb_start = svg.find("viewBox=\"")? + "viewBox=\"".len();
    let vb_end = vb_start + svg[vb_start..].find('"')?;
    let vb: Vec<f64> = svg[vb_start..vb_end]
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    let [min_x, min_y, width, height] = vb.as_slice() else {
        return None;
    };
    let new_width = width + ANNOTATION_GUTTER;

    let max_mean = nodes.iter().fold(0.0f64, |m, n| m.max(n.mean_ms));

    // --- pass 1: node shapes + the dashed level clusters around them ----
    let mut boxes: Vec<(String, BBox)> = Vec::new();
    for_each_node_group(svg, |id, group| {
        if let Some(b) = shape_bbox(group) {
            boxes.push((id.to_string(), b));
        }
    });
    let mut clusters: Vec<BBox> = Vec::new();
    let mut rest_c = svg;
    while let Some(pos) = rest_c.find("class=\"cluster\">") {
        let from = pos + "class=\"cluster\">".len();
        let Some(len) = rest_c[from..].find("</g>") else {
            break;
        };
        if let Some(b) = shape_bbox(&rest_c[from..from + len]) {
            clusters.push(b);
        }
        rest_c = &rest_c[from + len..];
    }

    // --- pass 2: recolour, extend tooltips, place annotations -----------
    let mut out = String::with_capacity(svg.len() + 4096);
    let mut rest = svg;
    while let Some(pos) = rest.find("class=\"node\">") {
        let group_start = pos + "class=\"node\">".len();
        let Some(group_len) = rest[group_start..].find("</g>") else {
            break;
        };
        out.push_str(&rest[..group_start]);
        let group = &rest[group_start..group_start + group_len];
        out.push_str(&recolour_node_group(
            group,
            nodes,
            max_mean,
            &boxes,
            &clusters,
            min_x + new_width,
            *min_x,
        ));
        rest = &rest[group_start + group_len..];
    }
    out.push_str(rest);

    // --- legend band below, gutter to the right -------------------------
    let band_h = 56.0;
    let new_height = height + band_h;
    // graphviz emits matching `width="{W}pt"`/`height="{H}pt"` + viewBox;
    // grow all so the band and gutter are inside the canvas, not clipped.
    let old_width_attr = format!("width=\"{}pt\"", *width as i64);
    let new_width_attr = format!("width=\"{}pt\"", new_width as i64);
    let old_height_attr = format!("height=\"{}pt\"", *height as i64);
    let new_height_attr = format!("height=\"{}pt\"", new_height as i64);
    let old_viewbox = &svg[vb_start..vb_end];
    let new_viewbox = format!("{min_x:.2} {min_y:.2} {new_width:.2} {new_height:.2}");
    let mut out = out
        .replace(&old_width_attr, &new_width_attr)
        .replace(&old_height_attr, &new_height_attr)
        .replacen(old_viewbox, &new_viewbox, 1);

    let band_y = min_y + height;
    let text_x = min_x + 8.0;
    let mut band = String::new();
    let _ = write!(
        band,
        r##"<g id="flodl_heat_legend">
<defs><linearGradient id="flodl_heat_scale" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#27ae60"/><stop offset="50%" stop-color="#f39c12"/><stop offset="100%" stop-color="#e74c3c"/></linearGradient></defs>
<rect x="{rx:.2}" y="{ry:.2}" width="{rw:.2}" height="{rh:.2}" fill="white" stroke="none"/>
<text x="{tx:.2}" y="{ty1:.2}" font-family="Helvetica,sans-Serif" font-weight="bold" font-size="12.00" fill="#333333">Timing heat map &#183; {model}</text>
<text x="{tx:.2}" y="{ty2:.2}" font-family="Helvetica,sans-Serif" font-size="10.00" fill="#555555">node labels: mean &#183; min &#183; {ranks} rank(s) &#183; {samples} passes &#183; {source} &#183; pass mean {total}</text>
<rect x="{tx:.2}" y="{gy:.2}" width="120" height="8" fill="url(#flodl_heat_scale)" stroke="#999999" stroke-width="0.5"/>
<text x="{gl:.2}" y="{gty:.2}" font-family="Helvetica,sans-Serif" font-size="9.00" fill="#555555">fast</text>
<text x="{gr:.2}" y="{gty:.2}" font-family="Helvetica,sans-Serif" font-size="9.00" fill="#555555">slow ({peak})</text>
</g>
"##,
        rx = min_x,
        ry = band_y,
        rw = new_width,
        rh = band_h,
        tx = text_x,
        ty1 = band_y + 18.0,
        ty2 = band_y + 33.0,
        gy = band_y + 40.0,
        gl = text_x + 126.0,
        gr = text_x + 152.0,
        gty = band_y + 48.0,
        model = xml_escape(&legend.gpu_model),
        ranks = legend.ranks,
        samples = legend.samples,
        source = xml_escape(&legend.source),
        total = format_ms(legend.total_mean_ms),
        peak = format_ms(max_mean),
    );
    let close = out.rfind("</svg>")?;
    out.insert_str(close, &band);
    Some(out)
}

/// A node shape's bounding box, in the drawing's coordinate space.
#[derive(Clone, Copy)]
struct BBox {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

/// Visit every `class="node"` group as `(title id, group body)`.
fn for_each_node_group<'a>(svg: &'a str, mut f: impl FnMut(&'a str, &'a str)) {
    let mut rest = svg;
    while let Some(pos) = rest.find("class=\"node\">") {
        let group_start = pos + "class=\"node\">".len();
        let Some(group_len) = rest[group_start..].find("</g>") else {
            return;
        };
        let group = &rest[group_start..group_start + group_len];
        if let Some(id) = group_title(group) {
            f(id, group);
        }
        rest = &rest[group_start + group_len..];
    }
}

/// The `<title>` text of one group body (the graph node id).
fn group_title(group: &str) -> Option<&str> {
    let from = group.find("<title>")? + "<title>".len();
    let len = group[from..].find("</title>")?;
    Some(&group[from..from + len])
}

/// Recolour one `class="node"` group body (between the group's opening
/// tag and its `</g>`): heat-fill the shape, extend the tooltip, and
/// place the visible `mean · min` label beside the shape.
fn recolour_node_group(
    group: &str,
    nodes: &[HeatNode],
    max_mean: f64,
    boxes: &[(String, BBox)],
    clusters: &[BBox],
    canvas_right: f64,
    canvas_left: f64,
) -> String {
    let Some(id) = group_title(group) else {
        return group.to_string();
    };
    let title_end = group.find("</title>").unwrap_or(0);
    let Some(node) = nodes.iter().find(|n| n.id == id) else {
        return group.to_string();
    };

    // Tooltip: id then timings, newline-separated (`&#10;` renders as a
    // multi-line native tooltip).
    let mut out = String::with_capacity(group.len() + 160);
    out.push_str(&group[..title_end]);
    let _ = write!(
        out,
        "&#10;mean {}&#10;min {}",
        format_ms(node.mean_ms),
        format_ms(node.min_ms)
    );
    let mut rest = &group[title_end..];

    // First fill after the title is the node shape's.
    let ratio = if max_mean > 0.0 {
        node.mean_ms / max_mean
    } else {
        0.0
    };
    if let Some(fill_pos) = rest.find("fill=\"") {
        let val_from = fill_pos + "fill=\"".len();
        if let Some(val_len) = rest[val_from..].find('"') {
            out.push_str(&rest[..val_from]);
            out.push_str(&heat_color(ratio));
            rest = &rest[val_from + val_len..];
        }
    }
    out.push_str(rest);

    // Visible `mean · min` beside the node, anchored OUTSIDE the dashed
    // level cluster: a label inside the cluster sits on the dashed
    // border and smudges in print. Levels stack vertically without
    // y-overlap, so the space beside a cluster box is structurally
    // free (the canvas gained a right gutter for the widest ones).
    if let Some(bbox) = boxes.iter().find(|(bid, _)| bid == id).map(|(_, b)| *b) {
        let label = format!(
            "{} \u{b7} {}",
            format_ms(node.mean_ms),
            format_ms(node.min_ms)
        );
        // ~0.56em average advance for 10px Helvetica, plus breathing room.
        let need = label.chars().count() as f64 * 5.6 + 10.0;
        let mid_y = (bbox.y0 + bbox.y1) / 2.0 + 3.5;
        let contains =
            |c: &BBox, b: &BBox| c.x0 <= b.x0 && c.x1 >= b.x1 && c.y0 <= b.y0 && c.y1 >= b.y1;
        let enclosing = clusters
            .iter()
            .filter(|c| contains(c, &bbox))
            .min_by(|a, b| {
                let area = |r: &BBox| (r.x1 - r.x0) * (r.y1 - r.y0);
                area(a).total_cmp(&area(b))
            });
        let (x, y, anchor) = if let Some(c) = enclosing {
            let has_mates = boxes.iter().any(|(bid, b)| bid != id && contains(c, b));
            if has_mates {
                // Side-by-side nodes in one level: each label takes the
                // cluster edge nearest its node, so a pair reads
                // unambiguously without leader lines.
                if (bbox.x0 + bbox.x1) / 2.0 <= (c.x0 + c.x1) / 2.0 {
                    (c.x0 - 4.0, mid_y, "end")
                } else {
                    (c.x1 + 4.0, mid_y, "start")
                }
            } else if c.x1 + 4.0 + need <= canvas_right - 4.0 {
                (c.x1 + 4.0, mid_y, "start")
            } else {
                (c.x0 - 4.0, mid_y, "end")
            }
        } else if bbox.x1 + 5.0 + need <= canvas_right - 4.0 || bbox.x0 - need < canvas_left {
            // No enclosing cluster (a render without level boxes):
            // the node shape's own right flank, left when it clips.
            (bbox.x1 + 5.0, mid_y, "start")
        } else {
            (bbox.x0 - 5.0, mid_y, "end")
        };
        let _ = write!(
            out,
            "\n<text text-anchor=\"{anchor}\" x=\"{x:.2}\" y=\"{y:.2}\" \
             font-family=\"Helvetica,sans-Serif\" font-size=\"10.00\" \
             fill=\"#555555\">{label}</text>",
        );
    }
    out
}

/// Bounding box of the first shape element in a node group body.
fn shape_bbox(group: &str) -> Option<BBox> {
    if let Some(pos) = group.find("<polygon") {
        let tag = &group[pos..pos + group[pos..].find("/>")?];
        let points = attr(tag, "points=\"")?;
        let (mut x0, mut x1, mut y0, mut y1) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for pair in points.split_whitespace() {
            let (x, y) = pair.split_once(',')?;
            let (x, y): (f64, f64) = (x.parse().ok()?, y.parse().ok()?);
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        return Some(BBox { x0, x1, y0, y1 });
    }
    if let Some(pos) = group.find("<ellipse") {
        let tag = &group[pos..pos + group[pos..].find("/>")?];
        let cx: f64 = attr(tag, "cx=\"")?.parse().ok()?;
        let cy: f64 = attr(tag, "cy=\"")?.parse().ok()?;
        let rx: f64 = attr(tag, "rx=\"")?.parse().ok()?;
        let ry: f64 = attr(tag, "ry=\"")?.parse().ok()?;
        return Some(BBox {
            x0: cx - rx,
            x1: cx + rx,
            y0: cy - ry,
            y1: cy + ry,
        });
    }
    None
}

/// Value of `key` (up to the closing quote) inside one element tag.
fn attr<'a>(tag: &'a str, key: &str) -> Option<&'a str> {
    let from = tag.find(key)? + key.len();
    let len = tag[from..].find('"')?;
    Some(&tag[from..from + len])
}

/// A two-node graphviz `-Tsvg` output, structurally faithful to what
/// `dot` emits for this crate's own `Graph::dot()` (fixture, since the
/// renderer is deliberately not a test dependency). Shared with the
/// dashboard-sink tests, which exercise the aggregation on top of it.
#[cfg(test)]
pub(crate) const GRAPHVIZ_FIXTURE: &str = r##"<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="134pt" height="116pt"
 viewBox="0.00 0.00 134.00 116.00" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(4 112)">
<title>G</title>
<polygon fill="white" stroke="none" points="-4,4 -4,-112 130,-112 130,4 -4,4"/>
<g id="clust1" class="cluster">
<title>cluster_level_0</title>
<polygon fill="none" stroke="#999999" stroke-dasharray="5,2" points="19,-64 19,-116 107,-116 107,-64 19,-64"/>
</g>
<g id="clust2" class="cluster">
<title>cluster_level_1</title>
<polygon fill="none" stroke="#999999" stroke-dasharray="5,2" points="28,8 28,-44 98,-44 98,8 28,8"/>
</g>
<g id="node1" class="node">
<title>linear_1</title>
<polygon fill="#e8f4fd" stroke="black" points="99,-108 27,-108 27,-72 99,-72"/>
<text text-anchor="middle" x="63" y="-86.3" font-family="Helvetica,sans-Serif" font-size="11.00">linear</text>
</g>
<g id="node2" class="node">
<title>relu_2</title>
<ellipse fill="#f5f5f5" stroke="black" cx="63" cy="-18" rx="27" ry="18"/>
<text text-anchor="middle" x="63" y="-14.3" font-family="Helvetica,sans-Serif" font-size="11.00">relu</text>
</g>
<g id="edge1" class="edge">
<title>linear_1&#45;&gt;relu_2</title>
<path fill="none" stroke="#7f8c8d" d="M63,-71.7C63,-64.41 63,-55.73 63,-47.54"/>
<polygon fill="#7f8c8d" stroke="#7f8c8d" points="66.5,-47.62 63,-37.62 59.5,-47.62 66.5,-47.62"/>
</g>
</g>
</svg>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = GRAPHVIZ_FIXTURE;

    fn legend() -> HeatLegend {
        HeatLegend {
            gpu_model: "NVIDIA GeForce RTX 5060 Ti".to_string(),
            source: "gpu events".to_string(),
            samples: 240,
            ranks: 2,
            total_mean_ms: 3.5,
        }
    }

    #[test]
    fn bakes_colors_tooltips_and_legend() {
        let nodes = vec![
            HeatNode {
                id: "linear_1".to_string(),
                min_ms: 2.0,
                mean_ms: 2.5,
            },
            HeatNode {
                id: "relu_2".to_string(),
                min_ms: 0.2,
                mean_ms: 0.25,
            },
        ];
        let baked = bake_heatmap(FIXTURE, &nodes, &legend()).unwrap();

        // The hottest node is full red, the cool one keeps a green hue.
        assert!(
            baked.contains(r##"<polygon fill="#e74c3c""##),
            "hot node recoloured"
        );
        assert!(!baked.contains("#e8f4fd"), "structural fill replaced");
        assert!(
            baked.contains(r##"<ellipse fill="#"##),
            "cool node recoloured"
        );

        // Tooltips carry the timings.
        assert!(baked.contains("linear_1&#10;mean 2.50ms&#10;min 2.00ms"));
        assert!(baked.contains("relu_2&#10;mean 250us&#10;min 200us"));

        // Visible `mean · min` beside each shape: nothing blocks the
        // right flank on this fixture, so the box (x1=99, mid y -90)
        // and the ellipse (x1=90, mid y -18) annotate rightward.
        // Anchored outside the dashed level clusters (x1 107 and 98),
        // not at the node shapes inside them.
        assert!(
            baked.contains(
                r##"<text text-anchor="start" x="111.00" y="-86.50" font-family="Helvetica,sans-Serif" font-size="10.00" fill="#555555">2.50ms · 2.00ms</text>"##
            ),
            "polygon node timing annotation missing",
        );
        assert!(
            baked.contains(
                r##"<text text-anchor="start" x="102.00" y="-14.50" font-family="Helvetica,sans-Serif" font-size="10.00" fill="#555555">250us · 200us</text>"##
            ),
            "ellipse node timing annotation missing",
        );
        // The dashed cluster borders themselves stay untouched.
        assert_eq!(baked.matches("stroke-dasharray").count(), 2);

        // The canvas gained the annotation gutter on top of the legend
        // band's height growth.
        assert!(baked.contains(r#"width="210pt""#), "gutter not applied");
        assert!(baked.contains("viewBox=\"0.00 0.00 210.00 172.00\""));

        // Edge groups untouched (their fills stay).
        assert!(baked.contains(r##"<polygon fill="#7f8c8d""##));

        // Canvas grew for the legend, in both height attr and viewBox.
        assert!(baked.contains(r#"height="172pt""#));
        assert!(baked.contains("Timing heat map"));
        assert!(baked.contains("NVIDIA GeForce RTX 5060 Ti"));
        assert!(baked.contains("2 rank(s)"));
        assert!(baked.contains("240 passes"));
        assert!(baked.contains("gpu events"));
        // Legend sits before the closing tag.
        assert!(baked.trim_end().ends_with("</svg>"));
    }

    #[test]
    fn unknown_nodes_keep_their_colour() {
        let nodes = vec![HeatNode {
            id: "linear_1".to_string(),
            min_ms: 1.0,
            mean_ms: 1.0,
        }];
        let baked = bake_heatmap(FIXTURE, &nodes, &legend()).unwrap();
        assert!(baked.contains("#f5f5f5"), "unprofiled node untouched");
        assert!(!baked.contains("relu_2&#10;"), "no tooltip invented");
    }

    #[test]
    fn garbage_svg_is_refused() {
        assert!(bake_heatmap("<svg>nope</svg>", &[], &legend()).is_none());
        assert!(bake_heatmap("not svg at all", &[], &legend()).is_none());
    }
}
