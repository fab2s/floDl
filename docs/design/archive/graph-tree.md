# Graph Tree: Composable Subgraph Training

> **ARCHIVED - Historical design doc.** This feature shipped in v0.2.0
> and is documented for users in
> [Tutorial 10: Graph Tree](../../tutorials/10-graph-tree.md). Internal
> links below were written from the original `docs/design/` location;
> add a `../` if reread.

A design for hierarchical graph composition with label-path addressing and
per-subgraph training control.

This feature enables the "hierarchical composition" and "multi-strategy training"
described in the trajectory thesis - trained graphs nested as modules inside
parent graphs, with fine-grained control over which parts train, freeze, or
share gradients.

**Status:** Implemented in v0.2.0. See [Tutorial 10: Graph Tree](../../tutorials/10-graph-tree.md)
for the current API and usage examples.

---

## Motivation

flodl supports Graph-as-Module composition: a built `Graph` implements
`Module` and can be placed inside another `FlowBuilder`. Training a composed
system requires these capabilities (all now implemented):

1. **Selective freezing** - freeze the read phase of a letter model while its
   scan phase adapts to word-level context.
2. **Cross-boundary observation** - a parent graph needs to read a child's
   tagged outputs (e.g., confidence scores) to make routing decisions.
3. **Subgraph checkpointing** - load a pre-trained letter model into a subtree
   of a larger word model, without disturbing surrounding weights.
4. **Per-subgraph optimizer groups** - different learning rates for different
   subgraphs, collected by label path rather than manual parameter wrangling.
5. **Training phase control** - declarative freeze/thaw schedules expressed as
   label paths, not imperative `requires_grad` surgery.

### Core principle: layered models trained independently

The graph tree is designed for **models of models** - layered compositions
where each layer can be trained, checkpointed, and loaded into a parent as
a building block. Training is progressive: lower layers are proven first,
then frozen (fully or partially) and composed into higher layers.

Each composition step produces a self-contained checkpoint. Layers can be
swapped, retrained, or remixed independently. A better letter model can be
dropped into an existing word model without retraining the word-level routing.

This is fundamentally different from monolithic end-to-end training. Each
layer is a tested, standalone model before it becomes a component.

### Driving use case: FBRL hierarchical word recognition

A three-level model for reading words from visual input. Each level spawns
instances of the next based on what it observes:

```
WordGraph "word"
|
+-- MetaScan "meta"
|   Wide blurred glimpses -> word boundaries + letter count N
|   Output: N approximate letter positions
|
+-- each(N) -> SubScan "subscan"
|   Narrower, less blurred -> refine letter center per position
|   Output: N precise letter centers
|
+-- each(N) -> LetterModel "letter"
    +-- Scan "scan"  [learnable, slightly wider focused]
    |   Adapts to word context -- gradients flow from word loss
    +-- Read "read"  [frozen, focused]
        Already trained on isolated letters -- classification
```

**N is data-dependent** - meta scan estimates the letter count, and `each`
fans out dynamically. All N instances share weights (one model called N times).

**Progressive training pipeline:**

```
Step 1: Train letter model (scan + read) on isolated letter data
        -> checkpoint: letter_v1.fdl.gz

Step 2: Compose subscan + letter model
        Freeze "letter.read", keep "letter.scan" trainable
        Train on letter-in-word data (subscan learns to locate, letter.scan adapts)
        -> checkpoint: subscan_v1.fdl.gz

Step 3: Compose full word model
        Load subscan checkpoint (which contains its letter subgraph)
        Optionally freeze "subscan" entirely, or keep "letter.scan" trainable
        Train meta scan on full word data
        -> checkpoint: word_v1.fdl.gz
```

Each step builds on proven layers. Step 2 doesn't touch letter.read weights -
they're proven on isolated letters. Step 3 doesn't touch subscan's letter
spotting - it's proven on letter-in-word data. The word model only needs to
learn boundary detection and letter count estimation.

**What the graph tree supports:**

- Load `letter_v1.fdl.gz` into `"letter"` subtree without touching `"subscan"`
- Load `subscan_v1.fdl.gz` into `"subscan"` subtree (carrying its letter subgraph)
- Freeze `"letter.read"` while training everything else
- Freeze `"subscan"` entirely for step 3, or selectively thaw `"letter.scan"`
- Read `"letter.read.confidence"` across boundaries for retry decisions
- Assign per-layer LRs: `"meta"` at 0.001, `"subscan"` at 0.0005, `"letter.scan"` at 0.0001
- `each` over N positions with shared weights and correct gradient flow

Before graph tree, none of this was possible without manual parameter bookkeeping.

---

## Design principles

### 1. Zero forward-path impact

The forward path (`forward_impl`) is untouched. All graph tree features are
build-time, setup-time, or explicit-query-time. The routing loop never touches
tree metadata. The pre-computed `Vec<Vec<Route>>`, reused `exec_slots`, and
topological level execution remain exactly as they are.

### 2. Children are oblivious (but aware)

A `Graph` does not know *who* its parent is. It does not store a parent
pointer. Being used as a child is something that *can happen* to a graph,
not something it's built for.

However, when a graph is composed into a parent, the parent sets a
`composed: Cell<bool>` flag on it. The child can query `is_composed()` to
adapt its behavior - e.g., skip its own loss computation when a parent
handles the loss, or suppress standalone-only observation. The flag is a
one-bit signal: "you are part of something larger." Nothing more.

The parent discovers its children at build time and registers them in its own
metadata. All tree operations are top-down - you call them on an ancestor,
never on a child reaching up.

### 3. Strict dot semantics

Dots in paths always mean subgraph boundaries. No fuzzy resolution, no
"walk up to parent", no ambiguity heuristics.

- `"scan"` - local tag in the current graph (existing behavior, unchanged)
- `"letter.scan"` - child subgraph `"letter"`, then `"scan"` within it
- `"letter.scan.location"` - child `"letter"`, then `"scan"` within it, then `"location"` within that

If a dot-path segment doesn't resolve to a registered child subgraph, you get
a clear error: *`"letter" is not a subgraph of this graph`*. Period.

This means:
- No ambiguity detection needed (bare names are always local)
- No upward resolution (children don't know about parents)
- One rule, one behavior, one error message

---

## Design

### 1. Graph tree structure

The parent `Graph` gains a children map. The child `Graph` gains a composed flag.

```rust
pub struct Graph {
    // ... existing fields unchanged ...

    /// Human-readable label (set via FlowBuilder::label).
    /// Stable across development -- "letter" stays "letter" even if
    /// internal modules change (which would change the structural hash).
    label: Option<String>,

    /// Child subgraphs, keyed by label.
    /// Maps label -> node_index in self.nodes.
    /// The actual Graph is accessed via self.nodes[idx].module.as_graph().
    children: HashMap<String, usize>,

    /// True when this graph has been composed into a parent graph.
    /// Set by the parent's build(), never by the child itself.
    /// Allows the child to adapt behavior (e.g., skip standalone loss).
    composed: Cell<bool>,
}
```

Children are stored as **node indices**, not `Rc<Graph>`. The child graph
already lives inside `self.nodes[idx].module` as an `Rc<dyn Module>`. The
children map is just an index for label-based lookup. No new ownership, no
`Rc<Graph>`, no `Weak` references, no circular dependencies.

Children are registered automatically when a `Graph` (with a label) is used
as a `Module` inside a `FlowBuilder`. The builder detects that the module is
a `Graph` via `Module::as_graph()`, reads its label, records the mapping,
and sets `composed = true` on the child.

**Build-time validation:**
- Label collision at the same tree level is a build error.
- A graph used as a child subgraph *must* have a label (build error otherwise).
- Labels must be valid identifiers (alphanumeric + underscore, no dots).

### 2. Label-path addressing

All tree operations use dot-separated label paths with optional loop indexing:

```
"letter"                    -> entire letter subgraph
"letter.scan"               -> scan tag/subgraph within letter
"letter.scan[3]"            -> scan loop iteration 3
"letter.scan[3].location"   -> tag "location" within scan step 3
"meta"                      -> meta-scan subgraph (or local tag "meta")
"subscan.hit_rate"          -> hit_rate tag within subscan subgraph
```

**Resolution rules (strict, top-down):**

1. Split path on `.` into segments.
2. First segment: look up in `self.children`. If found, descend into that
   child graph. If not found, treat the entire path as a local tag name
   (only valid for single-segment paths - a dotted path that doesn't start
   with a child label is an error).
3. Subsequent segments: within the resolved child graph, look up the next
   segment in *its* `children` first, then in its `tag_names`. Recurse.
4. Final segment: resolves to either a child subgraph or a tag.

**Loop indexing:**
- `"scan"` resolves to the loop's collected output (all iterations).
- `"scan[i]"` resolves to iteration `i`'s output specifically.
- Negative indexing: `"scan[-1]"` for last iteration.

**Error messages are explicit:**
- `"letter" is not a subgraph of this graph` - first segment doesn't match any child
- `"scan" is not a subgraph or tag within "letter"` - intermediate segment fails
- `path "letter.scan" resolves to a tag, not a subgraph -- cannot descend further` - dotting into a non-graph

### 3. Tag visibility: shared by default, opt-in internal

Every tag in every subgraph is accessible from any ancestor via its full
label path. No port declarations, no explicit exports.

**Opt-in internal:** for tags that are truly implementation details:

```rust
FlowBuilder::from(module)
    .through(intermediate)
    .tag("_plumbing")           // convention: underscore prefix = internal
    .internal("_plumbing")      // explicit: hide from parent resolution
    .through(next)
    .tag("output")              // visible everywhere
    .build()
```

Internal tags:
- Cannot be resolved from outside their graph.
- Do not appear in verbose wiring output.

**Auto-internal inference:** some tags are obviously internal:

| Pattern | Reason | Auto-internal? |
|---|---|---|
| `.using("ref")` forward-ref wiring | Plumbing between modules | Yes |
| Loop iteration counter | Bookkeeping | Yes |
| Tags starting with `_` | Convention | Yes |
| `.tag("loss")`, `.tag("location")` | Semantic output | No |
| Loop trace outputs | Diagnostic/reuse | No |

Auto-internal can be overridden with `.shared()` if needed.

### 4. Tree-aware parameter collection

```rust
impl Graph {
    /// Parameters matching a label path.
    /// "letter.scan" -> all parameters in the scan subgraph of letter.
    /// "letter" -> all parameters in the entire letter subgraph.
    ///
    /// Returns Err if the path doesn't resolve.
    /// Returns Ok(vec![]) if the path resolves but contains no parameters.
    pub fn parameters_at(&self, path: &str) -> Result<Vec<Parameter>> { ... }

    /// Named parameters at a label path, using the target's own namespace.
    /// Useful for checkpoint operations on subtrees.
    pub fn named_parameters_at(&self, path: &str) -> Result<Vec<(String, Parameter)>> { ... }

    /// Named buffers at a label path, using the target's own namespace.
    pub fn named_buffers_at(&self, path: &str) -> Result<Vec<(String, Buffer)>> { ... }
}
```

This resolves the path, then delegates to the target graph/module's existing
`parameters()` or `named_parameters()` method. Deduplication by pointer
(existing behavior) prevents double-counting shared parameters.

The existing `parameters()` method remains unchanged (returns all parameters
in the graph, including those in child subgraphs - this already works because
child graphs are modules in nodes).

### 5. Tree-aware freeze/thaw

```rust
impl Graph {
    /// Freeze all parameters at the given label path.
    /// Sets requires_grad = false on all parameters in the subtree.
    /// Returns Err if the path doesn't resolve.
    pub fn freeze(&self, path: &str) -> Result<()> { ... }

    /// Thaw (unfreeze) all parameters at the given label path.
    /// Returns Err if the path doesn't resolve.
    pub fn thaw(&self, path: &str) -> Result<()> { ... }

    /// Check if a label path is fully frozen.
    /// Returns Err if the path doesn't resolve.
    pub fn is_frozen(&self, path: &str) -> Result<bool> { ... }
}
```

These delegate to `Parameter::freeze()` / `Parameter::unfreeze()` which already
exist. Invalid paths return `Err` - a typo in `"lettre.scan"` is caught
immediately rather than silently freezing nothing.

**Training phase definitions become declarative:**

```rust
// Phase A: train subscan only, letter frozen
graph.freeze("letter")?;

// Phase B: thaw letter scan, read stays frozen
graph.thaw("letter.scan")?;

// Phase C: fine-tune everything except read
graph.thaw("meta")?;
graph.thaw("subscan")?;
// "letter.read" remains frozen from Phase A
```

### 6. Subgraph checkpoint loading

```rust
impl Graph {
    /// Load a checkpoint into a specific subgraph.
    ///
    /// The checkpoint's structural hash is validated against the target
    /// subgraph's hash (not the root graph's hash). Named parameters are
    /// matched within the subgraph's own namespace.
    ///
    /// Returns LoadReport with loaded/skipped/missing parameter names.
    pub fn load_subgraph_checkpoint(
        &self,
        path: &str,          // label path to target subgraph
        checkpoint: &str,    // file path to checkpoint
    ) -> Result<LoadReport> { ... }
}
```

This is the key to transfer learning in composed systems:

```rust
// Build word model
let word = FlowBuilder::from(Identity)
    .through(meta_scan).tag("meta")
    .through(sub_scan).tag("subscan")
    .through(letter_graph)  // labeled "letter" internally
    .build()?;

// Load pre-trained letter weights into the subgraph
let report = word.load_subgraph_checkpoint("letter", "letter_v2_best.fdl.gz")?;
eprintln!("Loaded {}/{} params into letter subgraph", report.loaded, report.total);

// Freeze the read phase (pre-trained, proven)
word.freeze("letter.read")?;
```

The checkpoint uses the **child's own namespace** (`"scan/weight"`, `"read/weight"`)
because it was saved from the child graph directly. The structural hash check
ensures the letter checkpoint matches the letter subgraph's architecture,
regardless of what the parent graph looks like.

### 7. Two-level access semantics: null vs nil

All tree-aware access methods distinguish between two kinds of absence:

- **Null** (`Err`) - the path doesn't exist in the topology. Typo, wrong
  structure, or the subgraph was never registered. This is a wiring bug.
- **Nil** (`Ok(None)`) - the path resolves to a real tag/subgraph, but no
  value has been computed yet in the current forward pass. This is a timing
  issue - you're reading before the producer executed.
- **Value** (`Ok(Some(v))`) - the path resolves and the value is available.

This maps to `Result<Option<T>>` - idiomatic Rust for two-level fallibility.

**Why this matters:**

1. **Debuggable.** `Ok(None)` means execution order. `Err` means wiring.
   Without this distinction, both collapse into `None` and you stare at it.

2. **Build-time validation.** After `build()`, the graph can walk all declared
   cross-boundary paths and verify they resolve structurally. Invalid paths
   (null) are caught before any forward pass.

3. **Conditional dispatch.** A parent graph can probe a child tag to decide
   whether to run the child first: `Ok(None)` means "run it", `Ok(Some(v))`
   means "already have a value", `Err` means "bug in my wiring".

**Applies uniformly across all `_at` methods:**

| Method | Null (`Err`) | Nil (`Ok(None)`) | Value (`Ok(Some)`) |
|---|---|---|---|
| `tagged_at` | path invalid | tag not yet computed | `Variable` |
| `trend_at` | path invalid | no data collected yet | `Trend` |
| `parameters_at` | path invalid | - (always `Ok`) | `Vec<Parameter>` |
| `collect_at` | path invalid | - (always `Ok`) | `()` |

**Build-time path validation:**

```rust
impl Graph {
    /// Verify that a label path resolves to something in the topology.
    /// Call after build() to catch wiring bugs early.
    pub fn validate_path(&self, path: &str) -> Result<PathKind> { ... }
}

pub enum PathKind {
    Tag,        // resolves to a named tag
    Subgraph,   // resolves to an entire child graph
    LoopIndex,  // resolves to a specific loop iteration
}
```

### 8. Cross-boundary tag access

```rust
impl Graph {
    /// Get a tagged output by label path.
    /// Returns Err if the path doesn't exist (null -- wiring bug).
    /// Returns Ok(None) if the path exists but hasn't been computed yet (nil).
    /// Returns Ok(Some(v)) if the value is available.
    pub fn tagged_at(&self, path: &str) -> Result<Option<Variable>> { ... }

    /// Collect metrics from a label path into observation buffers.
    /// Returns Err if the path doesn't exist.
    pub fn collect_at(&self, path: &str) -> Result<()> { ... }

    /// Record a scalar metric at a label path.
    /// Returns Err if the path doesn't exist.
    pub fn record_at(&self, path: &str, value: f64) -> Result<()> { ... }

    /// Get trend for a label-path metric.
    /// Returns Err if the path doesn't exist (null).
    /// Returns Ok(None) if no data has been collected yet (nil).
    pub fn trend_at(&self, path: &str) -> Result<Option<Trend>> { ... }
}
```

The existing `tagged()`, `collect()`, `record_scalar()`, `trend()` methods
remain unchanged for local access. The `_at` variants resolve the label path,
then delegate to the target graph's existing observation methods.

**Use case - conditional retry in word model:**

```rust
// After letter model forward pass:
let confidence = word.tagged_at("letter.read.confidence")?  // Err = wiring bug
    .map(|v| v.item().unwrap_or(0.0))
    .unwrap_or(0.0);  // None = not computed yet, treat as 0

if confidence < threshold {
    let hidden = word.tagged_at("letter.read.hidden")?
        .expect("hidden should be computed after forward");
    let delta = refine_head.forward(&hidden)?;
    // ... retry logic ...
}
```

### 9. Verbose build-time wiring output

```rust
let graph = FlowBuilder::from(Identity)
    // ... build graph ...
    .verbose(true)
    .build()?;
```

Output format:

```
=== Graph Tree ===
word [hash: a3f8c2d1]
+-- meta [hash: 7b2e9f4a]
|   +-- tags: location, content_logit
|   +-- params: 12,544 (3 modules)
+-- subscan [hash: 1c5d8e3b]
|   +-- tags: positions, hit_rate
|   +-- params: 98,816 (5 modules)
+-- letter [hash: 9e4f2a7c]    <- matches checkpoint letter_v2_best.fdl.gz
    +-- scan [hash: 4d1a8c5e]
    |   +-- tags: location, content_logit
    |   +-- params: 45,312 (2 modules)
    +-- read [hash: 6f3b9d2a]  * frozen
        +-- tags: hidden, confidence, classification
        +-- params: 45,312 (2 modules) [frozen]

=== Tag Resolution ===
"meta.location"     -> meta/node_3:0
"positions"         -> subscan/node_2:0
"letter.scan[0]"    -> letter/scan/loop_body:iter_0
"confidence"        -> letter.read.confidence -> letter/read/node_5:0

=== Internal (auto-detected) ===
meta._using_image         auto-internal (forward ref wiring)
letter.scan._step_idx     auto-internal (loop counter)

=== Parameter Summary ===
Total: 201,984 parameters
  meta:          12,544  (6.2%)
  subscan:       98,816  (48.9%)
  letter.scan:   45,312  (22.4%)  trainable
  letter.read:   45,312  (22.4%)  frozen
```

Only produced when `.verbose(true)` is set.

### 10. Optimizer integration

The existing `Adam::with_groups()` builder works with label-path parameter
collection:

```rust
let mut optimizer = Adam::with_groups()
    .group(&graph.parameters_at("meta")?, 0.001)
    .group(&graph.parameters_at("subscan")?, 0.001)
    .group(&graph.parameters_at("letter.scan")?, 0.0001)
    // letter.read is frozen -- not in any group
    .build();
```

No changes needed to the optimizer itself. The new capability is
`parameters_at()` which makes group construction declarative.

### 11. Training mode propagation

`set_training(bool)` already propagates to child modules. Since `Graph`
implements `Module`, calling `word_graph.train()` sets training mode on all
subgraphs recursively. For selective eval (e.g., letter read in eval mode
during word training):

```rust
impl Graph {
    /// Set training mode on a specific subgraph.
    /// Returns Err if the path doesn't resolve.
    pub fn set_training_at(&self, path: &str, training: bool) -> Result<()> { ... }
}
```

This matters for BatchNorm - frozen subgraphs should use running stats (eval
mode), not batch stats (training mode).

---

## Performance analysis

### Forward path: zero impact

The graph tree adds metadata to `Graph` but `forward_impl()` only accesses:
`nodes`, `levels`, `order`, `routes_from`, `input_routes`, `exec_slots`,
`state`, `state_writers`, `tag_capture`, `tagged_outputs`. None of these
change. The `children` HashMap is never touched during forward.

### Memory: negligible

`children: HashMap<String, usize>` adds ~3 words to the Graph struct (pointer +
len + capacity on the stack side). The Graph struct is heap-allocated behind
`Rc<dyn Module>` and already contains multiple HashMaps and Vecs. No
measurable impact.

### Build time: proportional to tree depth

Child registration, label validation, and path pre-validation add work
proportional to the number of subgraphs. This is a one-time cost at
`build()`, not per-forward.

### Structural hash: unchanged

Child graphs already contribute to the parent's structural hash implicitly
through their parameter shapes and topology (they're modules in nodes). The
`children` HashMap is **not** added to hash computation - that would change
existing hashes and invalidate checkpoints for no functional gain.

### Parameter collection: same complexity

`parameters()` already walks all nodes including child subgraphs (they're
modules). `parameters_at()` is additive - it walks a subtree instead of
the full graph. Same deduplication by `Rc` pointer.

---

## Implementation plan

### Phase A: Foundation (tree registration + path resolution)

**Files:** `graph/mod.rs`, `graph/flow.rs`, `nn/mod.rs` (Module trait)

1. Add `as_graph(&self) -> Option<&Graph>` to Module trait (default `None`).
2. Override in Graph.
3. Add `children: HashMap<String, usize>` to Graph.
4. In FlowBuilder::build(), detect Graph modules via `as_graph()`, read
   their label, validate (no collisions, no dots in labels, label required),
   and populate the children map.
5. Add `resolve(&self, path: &str) -> Result<ResolvedPath>` to Graph.
   Parse dot-separated segments, walk children map, return target.
6. Add `validate_path(&self, path: &str) -> Result<PathKind>`.

```rust
/// Result of resolving a label path.
pub(crate) enum ResolvedPath<'a> {
    /// Resolves to an entire child subgraph.
    Subgraph(&'a Graph),
    /// Resolves to a tag within a specific graph.
    Tag { graph: &'a Graph, tag: String, index: Option<usize> },
}
```

No Rc, no ownership - just borrowed references into the existing node array.

### Phase B: Training control

**Files:** `graph/mod.rs`

1. `parameters_at(path) -> Result<Vec<Parameter>>`
2. `named_parameters_at(path) -> Result<Vec<(String, Parameter)>>`
3. `named_buffers_at(path) -> Result<Vec<(String, Buffer)>>`
4. `freeze(path) -> Result<()>`
5. `thaw(path) -> Result<()>`
6. `is_frozen(path) -> Result<bool>`
7. `set_training_at(path, bool) -> Result<()>`

All resolve the path, then delegate to existing methods on the target.

### Phase C: Checkpoint composition

**Files:** `nn/checkpoint.rs`, `graph/mod.rs`

1. `load_subgraph_checkpoint(path, file) -> Result<LoadReport>`
2. Resolve path to target subgraph.
3. Use target's `named_parameters()` / `named_buffers()` (its own namespace).
4. Validate checkpoint hash against target's structural hash.
5. Load parameters/buffers via existing checkpoint machinery.

### Phase D: Cross-boundary observation

**Files:** `graph/observe.rs`, `graph/mod.rs`

1. `tagged_at(path) -> Result<Option<Variable>>`
2. `collect_at(path) -> Result<()>`
3. `record_at(path, value) -> Result<()>`
4. `trend_at(path) -> Result<Option<Trend>>`

All resolve the path, then delegate to the target graph's existing
observation methods.

### Phase E: Developer experience

**Files:** `graph/flow.rs`, `graph/verbose.rs`

1. `.verbose(bool)` on FlowBuilder.
2. Tree structure output with hashes, param counts, frozen state.
3. Tag resolution map.
4. Internal tag marking (`.internal()` builder method, auto-inference).
5. Parameter summary.

---

## API summary

New methods on `Graph`:

```rust
// Tree navigation
fn children(&self) -> &HashMap<String, usize>;
fn child_graph(&self, label: &str) -> Option<&Graph>;
fn subgraph(&self, path: &str) -> Result<&Graph>;
fn is_composed(&self) -> bool;  // true when nested inside a parent graph

// Path validation (build-time, no forward pass needed)
fn validate_path(&self, path: &str) -> Result<PathKind>;

// Label-path parameter operations (Err = path doesn't exist)
fn parameters_at(&self, path: &str) -> Result<Vec<Parameter>>;
fn named_parameters_at(&self, path: &str) -> Result<Vec<(String, Parameter)>>;
fn named_buffers_at(&self, path: &str) -> Result<Vec<(String, Buffer)>>;
fn freeze(&self, path: &str) -> Result<()>;
fn thaw(&self, path: &str) -> Result<()>;
fn is_frozen(&self, path: &str) -> Result<bool>;

// Label-path observation (Err = null/path invalid, None = nil/not yet computed)
fn tagged_at(&self, path: &str) -> Result<Option<Variable>>;
fn collect_at(&self, path: &str) -> Result<()>;
fn record_at(&self, path: &str, value: f64) -> Result<()>;
fn trend_at(&self, path: &str) -> Result<Option<Trend>>;

// Subgraph checkpoint
fn load_subgraph_checkpoint(&self, path: &str, file: &str) -> Result<LoadReport>;

// Training mode
fn set_training_at(&self, path: &str, training: bool) -> Result<()>;
```

New methods on `FlowBuilder`:

```rust
fn internal(&mut self, tag: &str) -> &mut Self;
fn verbose(&mut self, enabled: bool) -> &mut Self;
```

New method on `Module` trait:

```rust
fn as_graph(&self) -> Option<&Graph> { None }
```

---

## Connection to trajectory thesis

This design directly enables the "Mixture of Strategies" and "Hierarchical
Composition" sections of the trajectory thesis:

> Each sub-graph carries its training context (optimizer, loss, schedule).
> Gradient flow between sub-graphs is declared in the graph topology.
> The meta-controller's branching and looping are native Rust - zero overhead.
> Training the meta-controller is just another backward pass through a graph
> that happens to contain other trained graphs as nodes.

With graph tree + label-path addressing:
- "Each sub-graph carries its training context" = `parameters_at()` + per-group optimizer
- "Gradient flow declared in topology" = tags visible across boundaries + `freeze()`/`thaw()`
- "Trained graphs as nodes" = subgraph checkpoint loading + selective freezing

**Progressive composition is the key differentiator.** Unlike monolithic
end-to-end training, each layer is a proven, checkpointed model before it
becomes a component. The training pipeline is a sequence of composition steps,
not a single optimization run. This means:

- **Reproducibility** - each layer's training is isolated. A regression in
  the letter model is debugged at the letter level, not by staring at word-
  level loss curves.
- **Remixability** - swap in a better letter model, retrain only the layers
  above it. The checkpoint boundary is the composition boundary.
- **Incremental cost** - adding a new level (e.g., sentence -> word) doesn't
  retrain the proven lower layers. You pay training cost only for the new
  routing and the adapting scan phases.
- **Testability** - each layer has its own test suite and success criteria
  before it enters a composition. The letter model is proven on letter data;
  it doesn't need to be re-proven at the word level.

---

## Open questions

1. **Shared-weight subgraphs.** If the same `LetterGraph` is used at 4 word
   positions (shared weights, called in a loop), it's one tree node called
   N times. If we want independent weights per position, it's N tree nodes
   (`"letter_0"` through `"letter_3"`). Both patterns should work. The loop
   case is already handled by flodl's `Loop`; the independent case just needs
   distinct labels.

2. **Cross-graph gradient blocking.** `freeze()` blocks gradients through
   frozen parameters. But what about blocking gradients *between* subgraphs
   while keeping both trainable? A `detach_at()` or gradient barrier at a
   label path boundary. The parent-only tree design makes this natural: the
   parent inserts the barrier, the child doesn't participate. Not needed for
   the initial use case.

3. **Dynamic tree modification.** Can you add/remove subgraphs after build?
   No. The tree is fixed at build time. If you need different compositions,
   build different graphs. This keeps the implementation simple and the
   structural hash meaningful.

4. **Monitoring integration.** The dashboard could show a tree view with
   per-subgraph metrics. This is a Monitor feature, not a Graph feature,
   but the `_at` observation methods provide the data it needs.

5. **`each` + tree tag access.** When `each` runs N forward passes of a
   subgraph (shared weights), `tagged_at("letter.read.confidence")` returns
   the last iteration's value. For per-instance access, loop-style trace
   indexing (`tagged_at("letter.read.confidence[k]")`) would be needed.
   The trace mechanism already exists for loops; `each` may need the same
   pattern.

6. **Nested checkpoint loading.** `subscan_v1.fdl.gz` contains the letter
   subgraph's weights. Loading it into `"subscan"` should recursively
   restore the letter weights within it. The checkpoint format already
   stores named parameters with qualified paths, so this should work
   naturally - needs explicit testing.
