# The Graph Builder

The fluent graph builder is how you describe model architectures in floDl.
Instead of manually wiring layers together, you write data flow - what
happens to the tensor as it moves through the model.

By the end of this tutorial you'll be able to build models with linear
chains, parallel branches, residual connections, and per-element mapping.

> **Prerequisites**: familiarity with [Modules](03-modules.md) and
> [Training](04-training.md). You don't need to have read them - the
> code here is self-contained - but they explain the building blocks.

## Your first graph

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?)
    .through(GELU)
    .through(Linear::new(8, 2)?)
    .build()?;
```

`from` starts the flow. `through` appends a module. `build` finalizes the
graph and returns a `Graph` that implements `Module` - it has `forward` and
`parameters` just like any other module.

> **Note for PyTorch users**: In Python you write
> `model = nn.Sequential(...)` and errors raise exceptions implicitly.
> In Rust, `build()` returns `Result<Graph>` - errors are explicit values
> you handle with `?`. Throughout these tutorials you will see
> `let g = ... .build()?;` which propagates errors to the caller.

```rust
let input = Variable::new(input_tensor, true);
let output = g.forward(&input)?;
let loss = mse_loss(&output, &target)?;
loss.backward()?;
```

Gradients flow through the entire graph automatically.

## Residual connections with also

`also` adds a skip connection. The input passes through the module *and*
gets added to the module's output:

```rust
let g = FlowBuilder::from(Linear::new(8, 8)?)
    .through(GELU)
    .also(Linear::new(8, 8)?)          // output = input + Linear(input)
    .through(Linear::new(8, 2)?)
    .build()?;
```

This is the standard residual pattern from ResNet.

When the skip path also needs a transform - e.g. a 1×1 conv + BN to match
channel or stride changes in ResNet downsample blocks - use `also_with`.
It takes an explicit `skip` and `main` pair: `output = skip(x) + main(x)`.

```rust
let block = FlowBuilder::from(prev)
    .also_with(
        downsample_1x1_bn,      // skip path transform
        conv_bn_relu_conv_bn,   // main path
    )
    .through(ReLU)
    .build()?;
```

See `ddp-bench/src/models/resnet_graph.rs` for a full ResNet-20 using
this pattern.

## Parallel branches with split/merge

`split` sends the same input to multiple modules in parallel.
`merge` combines their outputs:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?)
    .split(modules![
        Linear::new(8, 8)?,   // branch A
        Linear::new(8, 8)?,   // branch B
        Linear::new(8, 8)?,   // branch C
    ])
    .merge(MergeOp::Mean)      // average the three outputs
    .through(Linear::new(8, 2)?)
    .build()?;
```

Each branch has independent parameters. Built-in merge operations:
- `MergeOp::Add` - element-wise sum
- `MergeOp::Mean` - element-wise average

## Naming points with tag

`tag` names a point in the flow so you can reference it later:

```rust
let g = FlowBuilder::from(encoder).tag("encoded")
    .through(transformer)
    .through(decoder)
    .build()?;
```

Tags are used by `using`, `gate`, `switch`, and `map.over` to access values
from earlier in the graph. See
[Advanced Graphs](06-advanced-graphs.md) for the full story.

## Naming parallel branches with tag_group

When you have parallel branches from `split`, `tag_group` names them all
at once with auto-suffixed tags:

```rust
let g = FlowBuilder::from(Linear::new(4, 8)?)
    .split(modules![head_a, head_b, head_c])
    .tag_group("head")
    .merge(MergeOp::Mean)
    .build()?;
// Creates tags: "head_0", "head_1", "head_2"
```

The suffixed tags work with all existing APIs - `tagged`, `collect`,
`trends`. `g.trends(&["head"])` expands the group and returns a
`TrendGroup` for aggregate queries.

## Multiple inputs with input

Some models need more than one external input. `input` adds a named
auxiliary entry point to the graph:

```rust
let g = FlowBuilder::from(encoder).tag("features")
    .input(&["condition"])
    .through(decoder).using(&["features", "condition"])
    .build()?;

g.forward_multi(&[image, condition_label])?; // inputs in declaration order
```

## Per-element processing with map

`map` applies a module to each element along dimension 0:

```rust
let g = FlowBuilder::from(encoder)
    .map(Linear::new(8, 8)?).each()
    .through(decoder)
    .build()?;
```

Three iteration modes:
- `.each()` - iterate over current stream (dim 0)
- `.over(tag)` - iterate over a tagged tensor
- `.slices(n)` - decompose last dim into n slices, map, recompose

For stateless bodies, add `.batched()` to skip element-by-element iteration:

```rust
.map(Linear::new(8, 8)?).batched().each()  // much faster
```

## Sub-graphs as modules

Since `Graph` implements `Module`, you can use graphs as building blocks
inside other graphs:

```rust
// Define a reusable block.
let block = FlowBuilder::from(Linear::new(8, 8)?)
    .through(GELU)
    .through(LayerNorm::new(8)?)
    .build()?;

// Use it like any module.
let model = FlowBuilder::from(Linear::new(4, 8)?)
    .through(block)               // sub-graph
    .through(Linear::new(8, 2)?)
    .build()?;
```

This is **Graph-as-Module** - the same pattern scales from small blocks
to entire model components. Add `.label("encoder")` to enable
[graph tree](10-graph-tree.md) features: selective freeze/thaw, subgraph
checkpointing, and cross-boundary observation.

## Putting it together

Here's a complete model that uses everything from this tutorial:

```rust
// Reusable feed-forward block.
fn ffn(dim: i64) -> flodl::Result<Graph> {
    FlowBuilder::from(Linear::new(dim, dim)?)
        .through(GELU)
        .through(LayerNorm::new(dim)?)
        .build()
}

// Main model.
let model = FlowBuilder::from(Linear::new(4, 16)?)
    .through(GELU)
    .split(modules![ffn(16)?, ffn(16)?]).merge(MergeOp::Mean)  // multi-head
    .also(Linear::new(16, 16)?)                                 // residual
    .through(Dropout::new(0.1))
    .through(Linear::new(16, 2)?)
    .build()?;

// Train it.
let params = model.parameters();
let optimizer = Adam::new(&params, 0.001);
model.train();

// ... training loop (see Tutorial 04) ...

// Evaluate.
model.eval();
let output = model.forward(&input)?;
```

## What's next

This tutorial covered the core builder methods. The
[Advanced Graphs](06-advanced-graphs.md) tutorial covers:
- **Forward references** - recurrent state across calls
- **Loops** - fixed, while, and until with BPTT
- **Gates** - soft routing with learned weights
- **Switches** - hard routing with selectors

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Training](04-training.md) | Next: [Advanced Graphs](06-advanced-graphs.md)
