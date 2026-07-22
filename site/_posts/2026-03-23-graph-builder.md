---
title: "Building a graph builder that doesn't feel like a graph builder"
subtitle: "How a consuming self pattern and forward references turned architecture description into data flow"
date: 2026-03-23
description: "PyTorch gives you Sequential or boilerplate. floDl's FlowBuilder finds the middle path: declarative, composable, and inspectable."
---

There are two ways to define a neural network in PyTorch.

The first is `nn.Sequential`:

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.GELU(),
    nn.Linear(16, 2),
)
```

Clean, readable, instant. But try adding a residual connection. Or a loop.
Or a branch that routes to different experts based on the input. You can't.
Sequential is a list. Data goes in one end and comes out the other.

The second is a custom `nn.Module`:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 16)
        self.decoder = nn.Linear(16, 2)
        self.residual = nn.Linear(16, 16)

    def forward(self, x):
        h = F.gelu(self.encoder(x))
        h = h + self.residual(h)
        return self.decoder(h)
```

Now you can express anything. But the architecture lives in imperative Python.
The framework can't see it. You can't automatically visualize the graph,
profile individual nodes, or collect metrics at intermediate points without
instrumenting `forward()` by hand. And every new architecture means a new
class with new wiring.

There's a gap between "too simple" and "too manual." floDl's FlowBuilder
tries to live in that gap.

## Data flow as description

The design constraint was: the builder chain should *read* like a description
of how data flows through the model. Not a set of construction commands.
You should be able to look at the chain and see the architecture.

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(Linear::new(16, 2)?)
    .build()?;
```

That's the Sequential equivalent. Data flows from Linear through GELU
through Linear. Same concepts, same readability. But the builder doesn't
stop here.

## Residual connections: the first thing Sequential can't do

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .also(Linear::new(16, 16)?)
    .through(Linear::new(16, 2)?)
    .build()?;
```

`also` means: run this module on the current tensor, then add the result to
the input. One word. That's a residual connection. In PyTorch, this requires
either a custom `forward()` or a helper class. Here it's part of the
vocabulary.

Read it aloud: "data flows through GELU, also through a Linear (with a
residual add), then through the output projection." The architecture is the
description.

## Parallel branches

```rust
let model = FlowBuilder::from(encoder)
    .split(modules![head_a, head_b, head_c])
    .merge(MergeOp::Mean)
    .through(output)
    .build()?;
```

`split` fans the stream to multiple modules. `merge` combines their outputs.
This is multi-head attention, or an ensemble, or any architecture where the
same input needs to go through parallel paths.

## Named references

Here's where it gets interesting. Tags let you name points in the flow and
reference them later:

```rust
let model = FlowBuilder::from(encoder).tag("encoded")
    .through(transform)
    .through(cross_attention).using(&["encoded"])
    .through(output)
    .build()?;
```

`tag("encoded")` names the encoder's output. `using(&["encoded"])` makes it
available to the cross-attention module as a named input. The data doesn't
need to flow linearly anymore. You can reach back to any tagged point.

This is how you build skip connections, cross-attention, U-Net architectures -
any pattern where a downstream module needs to see an earlier activation.

## Loops

```rust
let model = FlowBuilder::from(initial_state)
    .loop_body(refinement_block).for_n(3)
    .through(output_head)
    .build()?;
```

`loop_body` takes a module and repeats it, feeding each iteration's output
as the next iteration's input. `for_n(3)` runs it three times. The gradient
flows through all iterations - full backpropagation through time.

There's also `while_cond(halt_fn, max)` for adaptive computation: keep
iterating until a learned halt condition is met, up to a maximum.

## Routing

```rust
// Soft routing: all experts run, outputs weighted by router
.gate(router, modules![expert_a, expert_b])

// Hard routing: only selected branch executes
.switch(selector, modules![light_path, heavy_path])
```

`gate` is mixture-of-experts: the router produces weights, all experts
execute, outputs are combined. `switch` is conditional computation: the
selector picks one branch, only that branch runs.

Both support `.using()` to give the router access to earlier context.

## Side branches

```rust
.fork(classifier_head).tag("class_logits")
```

`fork` runs a module on the current stream but doesn't change the main flow.
The output is captured by the tag. The stream continues as if the fork wasn't
there. This is for observation points, auxiliary classifiers, or anything that
reads the stream without modifying it.

## The consuming self pattern

Every builder method consumes `self` and returns a new builder. This isn't
an accident. It means you can't do this:

```rust
let a = FlowBuilder::from(encoder);
let b = a.through(decoder_1);  // a is consumed
let c = a.through(decoder_2);  // compile error: a was moved
```

The compiler prevents you from branching from intermediate states. The
builder chain is a linear narrative: each step follows the previous one.
There's exactly one path through the description, and the type system
enforces it.

This is what makes the builder read as data flow rather than as a bag of
mutations. You're writing a description, not modifying state.

## Forward references: the trick that enables recurrence

Everything so far has been about referring to the past - modules that need
earlier activations. But some architectures need to refer to their own
output from the *previous* forward pass. Recurrent networks. Attention
with memory. Anything where state persists across calls.

The builder handles this with a simple rule: if you `using()` a tag
before it exists, it becomes a forward reference.

```rust
.through(StateAdd).using(&["memory"]).tag("memory")
```

On the first forward pass, `"memory"` doesn't exist yet - the reference
returns a zero tensor. On subsequent passes, it returns the value from
the previous call. The graph owns the state buffer. The module doesn't
need to know it's recurrent.

One line. That's an RNN-style feedback loop. No hidden state threading,
no manual `detach()` between steps, no state containers. The graph
manages it.

## Graphs are modules

A `Graph` built by the builder implements `Module` - the same trait as
`Linear` or `GELU`. This means a graph is just another building block:

```rust
let ffn = FlowBuilder::from(Linear::new(16, 32)?)
    .through(GELU)
    .through(Linear::new(32, 16)?)
    .build()?;

let model = FlowBuilder::from(encoder)
    .through(ffn)             // graph nested inside graph
    .through(output_head)
    .build()?;
```

Parameters, buffers, named references - everything composes. Build
sub-graphs as functions, nest them arbitrarily. The same API works at
every scale.

## What you get for free

Because the architecture is declared rather than buried in imperative
`forward()` code, the framework can inspect it. Without any user code:

- **Visualization**: `graph.dot()` produces a Graphviz diagram with
  parameter counts and node types. `graph.svg()` renders it.

- **Profiling**: `graph.enable_profiling()` times each node. The SVG
  colors nodes from green to red by execution time.

- **Observation**: `graph.collect(&["hidden"])` captures a tagged output
  during training. `graph.trend("loss")` computes epoch-level trends.
  Stalled? Improving? Converged? The graph tracks it.

- **Structural identity**: `graph.structural_hash()` produces a SHA-256
  hash of the architecture. Change a layer size and the hash changes.
  Checkpoints validate against it.

None of this is possible when the architecture lives in a Python `forward()`
method. It's only possible because the builder knows the graph structure
at build time.

## A real architecture

This isn't a framework that only handles toy examples. [FBRL](https://github.com/fab2s/fbrl)
uses the builder for a foveal attention model - a recurrent architecture
that learns where to look in an image. Here's a simplified view of the
actual builder chain:

```rust
FlowBuilder::from(Identity).tag("image")
    .input(&["case"])
    .through(h0_init)
    .loop_body(scan_step).for_n(n_scan).using(&["image"])
    .loop_body(read_step).for_n(n_read).using(&["image"]).tag("latent")
    .fork(letter_head)
    .fork(case_head)
    .through(decoder).using(&["latent", "case"])
    .build()?
```

Two recurrent loop phases (scan and read), each accessing the original
image through a named reference. Fork branches for classification heads.
A decoder that pulls from the latent bottleneck and an auxiliary input.
Nine loss components drive attention guidance, reconstruction,
classification, and fixation diversity.

This architecture trains at ~40 seconds per epoch on a GTX 1060.
Restructuring it - from separated phases to a unified graph for transfer
to word-level recognition - meant changing the builder chain. Not
rewriting `forward()`. Not a new class. The restructured version trained
faster.

That's the point. The builder makes architecture changes cheap. You
describe the new flow, and the training infrastructure - checkpoints,
visualization, observation, profiling - follows automatically.

## The gap

Sequential is beautiful for what it handles. Custom modules are necessary
for what it can't. The FlowBuilder tries to push the boundary of what's
declarative: residuals, branches, loops, routing, recurrence, and
composition, all as vocabulary in a fluent chain.

The architecture is the description. The description is the architecture.
Everything else follows.

---

*flodl is open source:
[GitHub](https://github.com/flodl-labs/flodl) |
[crates.io](https://crates.io/crates/flodl) |
[docs](https://docs.rs/flodl) |
[showcase example](https://github.com/flodl-labs/flodl/tree/main/flodl/examples/showcase/)*

*The graph builder tutorial:
[step-by-step guide](https://github.com/flodl-labs/flodl/blob/main/docs/tutorials/05-graph-builder.md)*
