# Graph Tree: Composable Subgraph Training

The graph tree extends Graph-as-Module composition with label-path addressing,
selective freeze/thaw, subgraph checkpointing, and cross-boundary observation.

> **Prerequisites**: [The Graph Builder](05-graph-builder.md) and
> [Advanced Graphs](06-advanced-graphs.md). You should be comfortable with
> `FlowBuilder`, `tag`, `using`, and Graph-as-Module nesting.

## Labeling subgraphs

Any `Graph` can be labeled with `.label()`:

```rust
let encoder = FlowBuilder::from(Linear::new(4, 8)?)
    .through(GELU)
    .through(Linear::new(8, 4)?)
    .label("encoder")
    .build()?;
```

When a labeled graph is used inside another `FlowBuilder`, the parent
automatically registers it as a **child subgraph**. Unlabeled graphs work
exactly as before - they just don't get tree features.

```rust
let model = FlowBuilder::from(encoder)  // child "encoder" registered
    .through(Linear::new(4, 2)?)
    .build()?;

assert_eq!(model.tree_children().len(), 1);
assert!(model.child_graph("encoder").is_some());
```

Labels must be valid identifiers (no dots - dots are path separators).
Duplicate labels at the same level produce a build error.

## The composed flag

A child graph knows it's been nested:

```rust
let child = model.child_graph("encoder").unwrap();
assert!(child.is_composed());  // true -- someone is using us
```

Use `is_composed()` to adapt behavior. For example, skip standalone loss
computation when a parent will handle the loss.

## Label-path addressing

Dots in paths mean subgraph boundaries. The rule is simple:

- `"encoder"` - child subgraph "encoder" (or local tag if no child with that name)
- `"encoder.hidden"` - tag "hidden" inside child "encoder"
- `"letter.read.confidence"` - tag "confidence" inside "read" inside "letter"

No fuzzy resolution, no walking up to parents. If a segment doesn't match
a child or tag, you get a clear error.

```rust
// Validate paths at build time
assert_eq!(model.validate_path("encoder")?, PathKind::Subgraph);
assert_eq!(model.validate_path("encoder.hidden")?, PathKind::Tag);
```

## Selective freeze and thaw

Freeze or thaw any subtree by path:

```rust
// Freeze the entire encoder
model.freeze("encoder")?;
assert!(model.is_frozen("encoder")?);

// Thaw just the scan phase within the encoder
model.thaw("encoder.scan")?;

// Read phase stays frozen
assert!(model.is_frozen("encoder.read")?);
```

This makes training phase definitions declarative:

```rust
// Phase 1: train only the routing layer, everything else frozen
model.freeze("encoder")?;
// ... train ...

// Phase 2: thaw encoder.scan, keep encoder.read frozen
model.thaw("encoder.scan")?;
// ... train with lower LR on scan ...
```

## Parameter groups by path

`parameters_at()` collects parameters from a subtree, ready for optimizer
groups:

```rust
let mut optimizer = Adam::with_groups()
    .group(&model.parameters_at("meta")?, 0.001)
    .group(&model.parameters_at("encoder.scan")?, 0.0001)
    // encoder.read is frozen -- not in any group
    .build();
```

For checkpoint operations, `named_parameters_at()` returns names in the
**target's own namespace** - not the parent's:

```rust
let named = model.named_parameters_at("encoder")?;
// Names like "hidden/weight", "hidden/bias" -- the encoder's own names
```

## Subgraph checkpoint loading

Train a component standalone, save it, then load it into a larger model:

```rust
// Step 1: Train the encoder standalone
let encoder = FlowBuilder::from(scan_module)
    .tag("scan")
    .through(read_module)
    .tag("read")
    .label("encoder")
    .build()?;

// ... train encoder ...
encoder.save_checkpoint("encoder_v1.fdl.gz")?;

// Step 2: Build a larger model with a fresh encoder
let fresh_encoder = FlowBuilder::from(scan_module_new)
    .tag("scan")
    .through(read_module_new)
    .tag("read")
    .label("encoder")
    .build()?;

let model = FlowBuilder::from(fresh_encoder)
    .through(classifier)
    .build()?;

// Load pre-trained weights into the encoder subgraph
let report = model.load_subgraph_checkpoint("encoder", "encoder_v1.fdl.gz")?;
eprintln!("Loaded {}/{} params", report.loaded.len(), report.loaded.len() + report.missing.len());

// Freeze the read phase (pre-trained, proven)
model.freeze("encoder.read")?;
```

The checkpoint uses the child's own namespace and structural hash.
Architecture mismatches are caught at load time.

## Cross-boundary observation

Read tagged outputs across graph boundaries:

```rust
// After forward pass
model.forward(&input)?;

// Read a child's tagged output (null/nil semantics)
match model.tagged_at("encoder.hidden")? {
    Some(v) => println!("hidden shape: {:?}", v.shape()),
    None => println!("not computed yet"),
}
// Err = path doesn't exist (wiring bug)
```

Record and track metrics across boundaries:

```rust
// Record into child's observation buffer
model.record_at("encoder.loss", loss_value)?;
model.record_at("encoder.accuracy", acc)?;

// Single flush on the parent flushes the entire tree
model.flush(&[]);

// Read trend from child
let trend = model.trend_at("encoder.loss")?;
println!("encoder loss trend: {:?}", trend.last());
```

### Tree-aware flush and metrics

`flush()` **automatically recurses** into all labeled child subgraphs. A single
`model.flush(&[])` on the root graph flushes the entire tree - no need to walk
children manually. If a child's buffer is already empty (flushed separately),
it's safely skipped (no double epoch entries).

`latest_metrics()` collects from the entire tree with **dotted prefixes**. A
child labeled `"encoder"` with a metric `"loss"` appears as `"encoder.loss"`.
Deep nesting works too: `"letter.read.confidence"`.

This means `Monitor::log()` sees the whole tree automatically:

```rust
model.record_at("subscan.ce", ce_value)?;
model.record_at("letter.accuracy", acc)?;
model.record_scalar("total_loss", total);

model.flush(&[]);  // flushes parent + subscan + letter

// Monitor sees: total_loss, subscan.ce, letter.accuracy
monitor.log(epoch, t.elapsed(), &model);
```

The dashboard displays each metric as a separate curve - the dotted names
provide natural grouping in the legend.

### Independent flush cadences

Sometimes child subgraphs train on a different schedule (e.g. a slow auxiliary
loss that's only meaningful every N epochs). Use `flush_local()` and
`latest_metrics_local()` to manage each graph's observation cycle independently:

```rust
// Every epoch: flush parent only
model.flush_local(&[]);

// Every 10 epochs: flush the slow child
if epoch % 10 == 0 {
    model.child_graph("auxiliary").unwrap().flush_local(&[]);
}

// For monitoring, choose what to show:
// - latest_metrics_local() = only this graph's own metrics
// - latest_metrics()       = this graph + all children (tree-recursive)
monitor.log(epoch, t.elapsed(), &model);  // uses latest_metrics() by default
```

When using independent cadences, the parent's `latest_metrics()` still collects
from children - it reads whatever the child last flushed. So the dashboard
shows the child's most recent epoch value, updated at the child's own pace.

## Internal tags

Tags starting with `_` are automatically internal - hidden from parent
graph resolution:

```rust
let encoder = FlowBuilder::from(module)
    .tag("_plumbing")       // auto-internal (underscore prefix)
    .through(next)
    .tag("output")          // visible from parent
    .label("encoder")
    .build()?;

let model = FlowBuilder::from(encoder)
    .through(Linear::new(4, 2)?)
    .build()?;

// This fails: _plumbing is internal
assert!(model.tagged_at("encoder._plumbing").is_err());

// This works
assert!(model.tagged_at("encoder.output").is_ok());
```

You can also mark tags explicitly:

```rust
FlowBuilder::from(module)
    .tag("intermediate")
    .internal("intermediate")  // explicitly hide from parent
    .through(next)
    .build()?;
```

## Training mode propagation

Set training/eval mode on specific subgraphs:

```rust
// Put encoder in eval mode (BatchNorm uses running stats)
model.set_training_at("encoder", false)?;

// Rest of model stays in training mode
```

This matters for BatchNorm - frozen subgraphs should use running stats
(eval mode), not batch stats.

## Verbose build output

Enable `.verbose(true)` to print the tree structure on build:

```rust
let model = FlowBuilder::from(encoder)
    .through(classifier)
    .verbose(true)
    .build()?;
```

Prints to stderr:

```
=== Graph Tree ===
(root) [hash: a3f8c2d1]
+-- tags: output
+-- params: 6
+-- encoder [hash: 7b2e9f4a]
    +-- tags: hidden, output
    +-- params: 4

=== Parameter Summary ===
Total: 6 parameters
  encoder: 4 (66.7%)  trainable
  (own): 2 (33.3%)  trainable
```

You can also call `tree_summary()` and `param_summary()` directly.

## Performance guarantee

The graph tree adds **zero overhead to the forward path**. All tree metadata
(`children`, `composed`, `internal_tags`) is stored in the `Graph` struct but
never accessed during `forward_impl()`. The pre-computed Vec routing, reused
execution buffers, and topological level execution remain exactly as they are.

Tree operations (`parameters_at`, `freeze`, `tagged_at`, etc.) are explicit
calls - they only run when you call them, never during forward/backward.

## Quick reference

| Method | Returns | Description |
|---|---|---|
| `tree_children()` | `&HashMap<String, usize>` | Direct children map |
| `child_graph(label)` | `Option<&Graph>` | One-level child lookup |
| `subgraph(path)` | `Result<&Graph>` | Multi-level subgraph lookup |
| `is_composed()` | `bool` | Whether nested in a parent |
| `validate_path(path)` | `Result<PathKind>` | Check if path resolves |
| `parameters_at(path)` | `Result<Vec<Parameter>>` | Params at path |
| `named_parameters_at(path)` | `Result<Vec<(String, Parameter)>>` | Named params (target namespace) |
| `named_buffers_at(path)` | `Result<Vec<(String, Buffer)>>` | Named buffers (target namespace) |
| `freeze(path)` | `Result<()>` | Freeze all params at path |
| `thaw(path)` | `Result<()>` | Unfreeze all params at path |
| `is_frozen(path)` | `Result<bool>` | All params frozen? |
| `set_training_at(path, bool)` | `Result<()>` | Training/eval mode at path |
| `load_subgraph_checkpoint(path, file)` | `Result<LoadReport>` | Load checkpoint into subgraph |
| `tagged_at(path)` | `Result<Option<Variable>>` | Tagged output across boundaries |
| `collect_at(paths)` | `Result<()>` | Collect metrics across boundaries |
| `record_at(path, value)` | `Result<()>` | Record scalar into child's buffer |
| `trend_at(path)` | `Result<Trend>` | Epoch trend from child's history |
| `flush(tags)` | `()` | Flush batch buffer (recurses into children) |
| `flush_local(tags)` | `()` | Flush this graph only (no recursion) |
| `latest_metrics()` | `Vec<(String, f64)>` | Latest epoch values (children with dotted prefixes) |
| `latest_metrics_local()` | `Vec<(String, f64)>` | Latest epoch values (this graph only) |
| `tree_summary()` | `String` | Tree structure visualization |
| `param_summary()` | `String` | Per-subgraph param breakdown |
| `internal_tags()` | `&HashSet<String>` | Tags hidden from parent |

### FlowBuilder methods

| Method | Description |
|---|---|
| `.label(name)` | Set graph label (enables tree features when nested) |
| `.internal(tag)` | Mark a tag as internal (hidden from parent) |
| `.verbose(true)` | Print tree structure on build |

## Migrating checkpoints from earlier versions

If you trained a model with flodl 0.1.x and renamed tags or restructured the
graph for 0.2.0, use `migrate_checkpoint_file()` to remap parameter names
without retraining:

```rust
use flodl::nn::{checkpoint_version, migrate_checkpoint_file};

if checkpoint_version("encoder_v1.fdl")? < 2 {
    let report = migrate_checkpoint_file(
        "encoder_v1.fdl",
        "encoder_v2.fdl",
        &encoder.named_parameters(),
        &encoder.named_buffers(),
    )?;
    println!("{}", report);
    assert!(report.is_complete());
}

// Load into the subgraph as usual
model.load_subgraph_checkpoint("encoder", "encoder_v2.fdl")?;
```

The migrated checkpoint is written as v2 with a zeroed structural hash, so it
loads without architecture validation. Same architecture required - if you
changed layer sizes, retrain instead.

## What's next

The graph tree is the foundation for progressive model composition -
training layers independently, checkpointing them, and composing them
into larger models with fine-grained training control. See the
[design document](../design/graph-tree.md) for the full architecture.

---

Previous: [Tutorial 9: Training Monitor](09-monitor.md) |
Next: [Tutorial 11: Multi-GPU Training](11-multi-gpu.md)
