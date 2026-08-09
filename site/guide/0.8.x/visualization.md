---
permalink: /guide/0.8.x/visualization
channel: 0.8.x
sitemap: true
layout: guide
title: "Visualization"
prev_url: /guide/0.8.x/advanced-graphs
prev_title: "Advanced Graphs"
next_url: /guide/0.8.x/utilities
next_title: "Utilities"
anchors:
  - id: "generating-output"
    title: "Generating output"
  - id: "reading-the-diagrams"
    title: "Reading the diagrams"
  - id: "profiling-overlay"
    title: "Profiling overlay"
  - id: "tips"
    title: "Tips"
  - id: "whats-next"
    title: "What's next"
last_modified_at: 2026-07-30T23:28:37+02:00
---

# Visualizing Graphs

Every `Graph` can export its structure as Graphviz DOT or SVG. This is the
fastest way to verify that your architecture is wired correctly, especially
when references, loops, and switches are involved.

> **Prerequisites**: [The Graph Builder](/guide/0.8.x/graph-builder) and
> [Advanced Graphs](/guide/0.8.x/advanced-graphs) introduce the constructs
> shown in the diagrams.

> **Runnable examples**: [`showcase`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/showcase) and
> [`flowbuilder_residual`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/flowbuilder_residual) both emit SVG.

## Generating output

Two methods on `Graph`:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?)
    .through(GELU)
    .also(Linear::new(8, 8)?)
    .through(Linear::new(8, 2)?)
    .build()?;

// DOT string - print, pipe, or paste into an online viewer.
println!("{}", g.dot());

// SVG file - requires the `dot` binary from Graphviz.
let svg = g.svg(Some("model.svg"))?;
```

`dot()` always works. `svg()` shells out to the Graphviz `dot` command,
so it returns an error if the binary is not found.

### Installing Graphviz

| OS     | Command                     |
|--------|-----------------------------|
| Ubuntu | `apt install graphviz`      |
| macOS  | `brew install graphviz`     |
| Alpine | `apk add graphviz`          |

If you cannot install Graphviz, paste the `dot()` output into an online
viewer such as [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline).

## Reading the diagrams

### Node shapes

| Shape           | Meaning                                         |
|-----------------|-------------------------------------------------|
| invhouse        | Input node (graph entry point)                  |
| house           | Output node (graph exit point)                  |
| doubleoctagon   | Node that is both input and output              |
| box             | Standard module (Linear, LayerNorm, Conv2d, ...) |
| ellipse         | Activation (GELU, ReLU, Sigmoid, Tanh, ...)     |
| box3d           | Loop (For, While, Until)                        |
| parallelogram   | Map (per-element processing)                    |
| diamond         | Switch router or state-read node                |
| circle          | Merge / add node                                |

### Colors

| Fill color   | Meaning                              |
|--------------|--------------------------------------|
| Blue         | Input nodes                          |
| Green        | Output nodes, normalization layers   |
| Yellow       | State-read nodes (forward refs)      |
| Purple       | Loop nodes                           |
| Orange       | Switch clusters                      |
| Light grey   | Standard modules (Linear, etc.)      |
| Peach        | Activations                          |
| Pink         | Dropout                              |
| Light blue   | Sub-graph (Graph-as-Module) nodes    |

### Node labels

Each node label shows:
1. The module type name (e.g. `Linear`, `GELU`, `Graph (sub)`)
2. Parameter count in brackets, formatted as K/M for readability
   (`[1.2K params]`, `[3.1M params]`)
3. Tag annotations as `#tagName` when the node has been tagged

### Edge styles

| Style                | Color   | Meaning                              |
|----------------------|---------|--------------------------------------|
| Solid                | Dark    | Normal data flow                     |
| Dashed               | Blue    | Using reference (backward ref)       |
| Dotted               | Orange  | Forward-ref state loop               |

### Switch clusters

Switch nodes are expanded into sub-clusters showing their internal
structure: a diamond for the router, a box per branch, and a circle
as the exit merge point.

### Execution levels

Nodes are grouped into levels that show which nodes can execute in parallel.
Nodes in the same level have no data dependencies on each other.

## Profiling overlay

Enable profiling to see per-node execution times:

```rust
g.enable_profiling();
g.forward(&input)?;

let profile = g.profile().unwrap();
for node in &profile.nodes {
    println!("{}: {}", node.id, format_duration(node.duration.as_secs_f64()));
}
```

Profiled SVG output colors nodes green->yellow->red by relative execution
time:

```rust
g.svg_with_profile(Some("profile.svg"))?;
```

### Timing trends

Track timing across epochs to detect performance regression:

```rust
g.enable_profiling();

for epoch in 0..num_epochs {
    for batch in &batches {
        g.forward(&input)?;
        g.collect_timings(&["encoder", "decoder"]);
    }
    g.flush_timings(&["encoder", "decoder"]);

    let encoder_trend = g.timing_trend("encoder");
    println!("encoder avg: {}", format_duration(encoder_trend.latest()));
}
```

## Tips

- **Iterate with DOT first.** Paste into an online viewer while
  prototyping - no need to install Graphviz locally during design.
- **Check levels for parallelism.** If branches you expect to be at the
  same level end up in different levels, there is an unintended dependency.
- **Verify Using wires.** Dashed blue edges should connect the tagged
  node to the consuming node.
- **State loops.** Dotted orange edges show forward-ref state. They
  should form cycles - from writer back to reader.
- **Large models.** Write SVG and open in a browser - SVG scales cleanly.

## What's next

[Utilities](/guide/0.8.x/utilities) covers gradient clipping, checkpoints,
parameter freezing, and weight initialization.

