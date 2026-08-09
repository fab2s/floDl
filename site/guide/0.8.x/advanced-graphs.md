---
permalink: /guide/0.8.x/advanced-graphs
channel: 0.8.x
sitemap: true
layout: guide
title: "Advanced Graphs"
prev_url: /guide/0.8.x/graph-builder
prev_title: "The Graph Builder"
next_url: /guide/0.8.x/visualization
next_title: "Visualization"
anchors:
  - id: "tag-and-using---backward-references"
    title: "Tag and Using - backward references"
  - id: "forward-references---recurrent-state"
    title: "Forward references - recurrent state"
  - id: "loops"
    title: "Loops"
  - id: "gate---soft-routing"
    title: "Gate - soft routing"
  - id: "switch---hard-routing"
    title: "Switch - hard routing"
  - id: "performance-internals"
    title: "Performance internals"
  - id: "putting-it-together"
    title: "Putting it together"
  - id: "quick-reference"
    title: "Quick reference"
  - id: "whats-next"
    title: "What's next"
last_modified_at: 2026-07-30T23:28:37+02:00
---

# Advanced Graphs

This tutorial covers the graph builder's advanced constructs: backward
and forward references, loops, gated routing, and conditional branching.

> **Prerequisites**: [The Graph Builder](/guide/0.8.x/graph-builder) covers the
> basics - from, through, build, also, split/merge, tag, map, and
> Graph-as-Module. Everything here builds on those primitives.

> **Runnable example**: [`showcase`](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/showcase) exercises every
> method of the fluent builder in one coherent graph.

## Tag and Using - backward references

The Graph Builder introduced `tag` for naming points in the flow. `using`
consumes those names. When the tag appears *before* the using call in
the builder chain, the value is wired directly - it is available in the
same forward pass with no extra machinery:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?).tag("hidden")
    .through(GELU)
    .through(cross_attention).using(&["hidden"])
    .build()?;
```

Here `cross_attention` must implement `NamedInputModule` - it receives
the stream and the tagged `"hidden"` tensor via `forward_named(stream, refs)`.

You can wire multiple tags at once:

```rust
.through(fusion).using(&["audio", "video"])
```

The module receives `(stream, {"audio": audio, "video": video})`.

## Forward references - recurrent state

When `using` appears *before* the matching `tag`, the builder creates a
**forward reference**. The value does not exist yet during the current
forward pass, so it is carried in a state buffer between calls to
`g.forward()`.

This is how you build recurrent connections:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?)
    .through(StateAdd).using(&["memory"]).tag("memory")
    .through(Linear::new(8, 2)?)
    .build()?;
```

Walk through what happens:

1. `using("memory")` appears before `tag("memory")` - the builder
   detects this automatically and creates a state buffer.
2. On the **first** `g.forward(&input)` call, the `"memory"` state is
   nil. The graph auto-fills with zeros, so `StateAdd` computes
   `stream + zeros = stream` (clean pass-through).
3. After execution, the output of the tagged node is captured into the
   state buffer.
4. On the **second** `g.forward(&input)` call, `StateAdd` receives the
   real previous output as the `"memory"` argument.

### Managing state

Two methods control the state buffers:

**reset_state** clears all buffers. Call this when starting a new sequence:

```rust
g.reset_state();
let out = g.forward(&first_input)?; // state starts fresh
```

**detach_state** breaks the gradient chain on state buffers, tagged outputs,
and module internal state without clearing values. Call this between
training steps to prevent the autograd graph from growing without bound:

```rust
for (input, target) in &batches {
    let output = g.forward(&Variable::new(input.clone(), true))?;
    let loss = mse_loss(&output, &Variable::new(target.clone(), false))?;
    loss.backward()?;
    g.detach_state(); // cut gradients, keep values
    optimizer.step()?;
    optimizer.zero_grad();
}
```

### Built-in state primitive

`StateAdd` is a nil-safe additive cell: it sums all non-nil inputs. On
the first pass it acts as a pass-through. On subsequent passes it adds
the previous state to the current stream.

## Loops

The `loop_body` builder repeats a body module, feeding each iteration's
output as the next iteration's input. Three termination modes cover
different use cases.

### Fixed iteration with for_n

```rust
let g = FlowBuilder::from(encoder)
    .loop_body(refinement_step).for_n(5)
    .through(decoder)
    .build()?;
```

The body runs exactly 5 times. Each iteration builds its own computation
graph, so the backward pass unrolls automatically - backpropagation
through time (BPTT) with no special handling.

`for_n` detects at call time whether refs are needed and skips
indirection when they are not. Loops without `.using()` run a tight
`body.forward()` loop with no HashMap construction.

### Conditional loops with while_cond and until_cond

Both take a condition module and a maximum iteration count. The condition
module receives the current state and returns a scalar.
**Positive (> 0) means halt.**

**while_cond** checks the condition *before* each body execution (0 to
max iterations):

```rust
.loop_body(refine).while_cond(ThresholdHalt::new(100.0), 20)
```

**until_cond** runs the body first, then checks (1 to max iterations):

```rust
.loop_body(refine).until_cond(LearnedHalt::new(hidden_dim)?, 20)
```

### Built-in halt conditions

**ThresholdHalt(val)** - signals halt when the max element of the state
exceeds the threshold. Parameter-free.

**LearnedHalt(dim)** - a learnable linear probe that projects the state
to a scalar. The network learns when to stop (ACT pattern). Has trainable
parameters.

### Loops with Using - external references

Loop bodies often need access to data that does not change between
iterations:

```rust
let g = FlowBuilder::from(identity).tag("image")
    .through(h0_init)
    .loop_body(attention_step).for_n(n_glimpses).using(&["image"]).tag("attention")
    .through(decoder)
    .build()?;
```

The loop body receives the tagged ref at every iteration via
`forward_named(state, refs)`.

### Auto-reset before loop iteration

Loop bodies with internal mutable state override `reset()` on Module:

```rust
impl Module for AttentionStep {
    fn reset(&self) {
        self.location.set(None); // clear stale state
    }
    // ...
}
```

Loops call `reset()` on the body before iterating, preventing stale
tensors whose grad_fns reference freed saved tensors from crashing
backward.

### Composing loop interfaces

A loop body can override any combination:

| Method | Effect |
|--------|--------|
| `reset()` | Auto-reset before each forward |
| `as_named_input()` | Named ref forwarding from `using()` |
| `detach_state()` | Gradient chain breaking between training steps |

### Per-iteration traces

Loop bodies often need to publish auxiliary values per iteration: the
running loss, an intermediate quality score, a confidence number, a
halting probability. These are not the body's main output (which the
runner already feeds to the next iteration) but observation signals
for the loss closure or for downstream analysis.

The canonical pattern is `LoopBody` + `TraceEmit`. The body implements
`step(input, refs, emit)` and calls `emit.publish("name", value)` for
each per-iteration output it wants to expose; the runner harvests
each step's emit map and appends entries into per-name vectors on
the loop node. Same API across single-GPU, multi-GPU, and the El Che
DDP path.

```rust
use flodl::{LoopBody, TraceEmit, Variable, forward_via_step};

struct RefinementBlock {
    quality_head: Linear,
    inner: Graph,
}

impl LoopBody for RefinementBlock {
    fn step(&self, x: &Variable, _refs: &Refs, emit: &mut TraceEmit)
        -> Result<Variable>
    {
        let refined = self.inner.forward(x)?;
        let quality = self.quality_head.forward(&refined)?;
        emit.publish("quality", quality.clone())?;
        emit.publish("step_loss", refined.norm()?)?;
        Ok(refined)
    }
}

impl Module for RefinementBlock {
    fn forward(&self, x: &Variable) -> Result<Variable> {
        forward_via_step(self, x)
    }
    fn as_loop_body(&self) -> Option<&dyn LoopBody> { Some(self) }
    fn parameters(&self) -> Vec<Parameter> {
        // ... usual delegation
    }
}
```

Read the published traces back from the graph after a forward pass:

```rust
let _ = graph.forward(&x)?;

let qualities = graph.traces("quality");   // Vec<Variable>, one per iteration
let losses    = graph.traces("step_loss"); // Vec<Variable>, one per iteration

// `graph.traces(name)` returns the published `Variable`s directly.
// Backward through these is fully supported -- gradients flow through
// the per-iteration outputs into the loop body's parameters.
```

Sparse emits are allowed: a step that does not call
`emit.publish("name", ...)` simply does not grow that name's vector,
so `traces["name"].len() <= n_iter`. `TraceEmit::publish` panics on
duplicate names within a single step (per-step dedup, always-on);
cross-loop name reuse and tag-vs-trace key collisions are validated
once per graph the first time traces are observed.

`forward_via_step` is the shorthand for "this body has no standalone
forward semantics, run it via the loop runner". Pair it with
`as_loop_body()` returning `Some(self)` and the body works
identically inside `loop_body(...).for_n(n)` and outside (where the
emitter is a no-op).

#### Single-stream legacy: `Module::trace()`

For loop bodies that publish exactly one per-iteration value and do
not need DDP-safe state (single-GPU only), the older
`Module::trace() -> Option<Variable>` shortcut still compiles. It
returns the most recently produced trace, the runner appends it
under the implicit name set by the loop's `tag(...)`. Reach for
`LoopBody` + `TraceEmit` whenever you need more than one stream,
sparse emits, or DDP support.

## Gate - soft routing

`gate` implements mixture-of-experts style routing. A router module
produces weights, all expert modules execute, and their outputs are
combined using the router's weights:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?).tag("features")
    .through(GELU)
    .gate(SoftmaxRouter::new(8, 3)?, modules![expert_a, expert_b, expert_c])
        .using(&["features"])
    .through(Linear::new(8, 2)?)
    .build()?;
```

Key properties:

- **All experts execute** on every forward pass. For sparse execution,
  use switch instead.
- **The router owns normalization.** SoftmaxRouter produces weights that
  sum to 1.
- **Using wires to the router**, not the experts. The router can make
  routing decisions based on earlier representations.
- **Vectorized combination.** Gate routing is vectorized internally: all
  expert outputs are stacked into a single tensor, then combined via
  broadcast multiply + sum in approximately 3 kernel launches regardless
  of expert count (compared to 3N with naive per-expert accumulation).
  This is transparent to the user - just use `.gate()` as before.

### Built-in routers

**SoftmaxRouter(dim, n)** - linear projection to n logits, then softmax:

```rust
gate(SoftmaxRouter::new(hidden, 3)?, modules![...])
```

**SigmoidRouter(dim, n)** - sigmoid gating, each expert independent:

```rust
gate(SigmoidRouter::new(hidden, 2)?, modules![...])
```

Both routers implement `NamedInputModule` - they sum Using refs into the
input before projection, so extra context does not change the input
dimension.

## Switch - hard routing

`switch` selects a single branch to execute based on the router's output.
Only the selected branch runs:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?).tag("features")
    .through(GELU)
    .switch(ArgmaxSelector::new(8, 2)?, modules![light_path, heavy_path])
        .using(&["features"])
    .through(Linear::new(8, 2)?)
    .build()?;
```

Key properties:

- **The router returns 0-based branch indices** — either a single scalar
  index for the whole stream, or one index per row of dim 0.
- **Per-sample routing is the batched default.** With one index per row,
  each sample goes to its own branch, only the branches that received rows
  run, and rows come back in their original order. `ArgmaxSelector` routes
  per sample; `FixedSelector` emits a scalar.
- **Selection is non-differentiable.** Gradients flow through whichever
  branch each sample was routed to.
- **Using refs go to the router**, not the branches.

A selector that emits neither one index nor one-per-row is a loud error, as
is a per-row index naming a branch that does not exist (the error names the
offending row).

One consequence of per-sample routing to keep in mind: a branch sees only the
rows routed to it, so batch-dependent layers inside a branch (BatchNorm above
all) compute their statistics over that sub-batch, not the full batch - and a
branch that received no rows does not run at all, so its buffers do not update
and its parameters get no gradient that step. This is inherent to conditional
computation, not a flodl quirk, but it means a branch that wins only a couple
of rows per batch trains on very noisy statistics. Prefer LayerNorm inside
switch branches, or keep the routing distribution balanced.

### Built-in selectors

**FixedSelector(idx)** - always picks the same branch:

```rust
switch(FixedSelector::new(0), modules![branch_a, branch_b])
```

**ArgmaxSelector(dim, n)** - learnable projection, picks the highest logit
per sample:

```rust
switch(ArgmaxSelector::new(hidden, 3)?, modules![...])
```

Routing is per row of dim 0. A stream with extra leading dims (for example
`[Batch, Time, features]`) yields one index per (row, step), which switch
rejects rather than misroute - flatten the leading dims into rows first if
per-step routing is what you want.

### Build-time validation

The builder validates that `using()` refs are only wired to modules
that implement `NamedInputModule`. If a router does not support named
inputs, the builder returns a clear error at `build()` time - not a
runtime crash.

## Performance internals

`Graph::build()` pre-computes a routing table at build time. Every node's
successor list and reference wiring is resolved into Vec-indexed routes
instead of HashMap lookups, and execution buffers are cached across
forward calls. There is zero HashMap allocation during inference. This
means graphs have near-zero framework overhead after build - the cost
is dominated by the modules themselves.

## Putting it together

```rust
let h = 8;

// Reusable sub-graph.
let block = FlowBuilder::from(Linear::new(h, h)?)
    .through(GELU)
    .through(LayerNorm::new(h)?)
    .build()?;

// Main model.
let model = FlowBuilder::from(Linear::new(4, h)?).tag("input")
    .through(GELU)
    .split(modules![Linear::new(h, h)?, Linear::new(h, h)?])
        .merge(MergeOp::Mean)
    .also(Linear::new(h, h)?)
    .loop_body(block).for_n(2).tag("refined")
    .gate(SoftmaxRouter::new(h, 2)?, modules![Linear::new(h, h)?, Linear::new(h, h)?])
        .using(&["input"])
    .switch(ArgmaxSelector::new(h, 2)?, modules![Linear::new(h, h)?, Linear::new(h, h)?])
        .using(&["refined"])
    .through(StateAdd).using(&["memory"]).tag("memory")
    .loop_body(Linear::new(h, h)?).while_cond(ThresholdHalt::new(100.0), 5)
    .through(Linear::new(h, 2)?)
    .build()?;

// Train.
let params = model.parameters();
let optimizer = Adam::new(&params, 0.001);
model.train();

for step in 0..num_steps {
    let output = model.forward(&input)?;
    let loss = mse_loss(&output, &target)?;
    loss.backward()?;
    model.detach_state();
    optimizer.step()?;
    optimizer.zero_grad();
}

// Evaluate on a new sequence.
model.eval();
model.reset_state();
let output = model.forward(&test_input)?;
```

## Quick reference

| Construct | Builder call | Behavior |
|-----------|-------------|----------|
| Auxiliary input | `input(&["name"])` | Named entry point, consumed via using |
| Backward ref | `tag("x")` ... `using(&["x"])` | Direct wire, same pass |
| Forward ref | `using(&["x"])` ... `tag("x")` | State buffer, cross-pass |
| Fixed loop | `loop_body(body).for_n(n)` | Exactly n iterations |
| While loop | `loop_body(body).while_cond(cond, max)` | 0..max, check before body |
| Until loop | `loop_body(body).until_cond(cond, max)` | 1..max, check after body |
| Loop + ref | `loop_body(body).for_n(n).using(&["x"])` | External ref at each iteration |
| Auto-reset | override `reset()` on Module | Loop resets body before iterating |
| Soft routing | `gate(router, modules![...])` | All execute, weighted sum |
| Hard routing | `switch(router, modules![...])` | One executes, index select |
| Device placement | `g.move_to_device(Device::CUDA(0))` | Move params + state |

## What's next

For hierarchical model composition - freezing subgraphs, loading
checkpoints into subtrees, cross-boundary observation - see the
[Graph Tree](/guide/0.8.x/graph-tree) tutorial.

For DOT/SVG output of your graphs, see
[Visualization](/guide/0.8.x/visualization).

