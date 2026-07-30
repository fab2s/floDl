# Rust for PyTorch Users

You don't need to learn all of Rust to use floDl. This tutorial covers the 10
patterns you'll actually encounter in training scripts. If you've written
PyTorch, you already have the mental model - Rust just makes a few things
explicit that Python leaves implicit.

> **Time:** ~15 minutes. After this you can follow every floDl tutorial.

> **Don't want to translate by hand?** floDl ships with an AI porting skill
> that reads your PyTorch script and generates a complete floDl project.
> Run `/port my_model.py` in Claude Code, or see the
> [Porting Guide](../porting.md). For a side-by-side API reference, see the
> [PyTorch Migration Guide](../pytorch_migration.md). You can also run
> `fdl api-ref` to get the full, up-to-date API surface in your terminal.

---

## 1. Variables: `let` and `let mut`

```rust
let x = 5;         // immutable - like Python, most things don't change
let mut y = 10;    // mutable - opt-in, like declaring "I will modify this"
y += 1;            // ok
// x += 1;         // compile error - x is immutable
```

**PyTorch equivalent:** Python variables are always mutable. Rust defaults to
immutable and makes you say `mut` when you need mutation. This prevents
accidental overwrites - the compiler catches them for you.

Optimizers need `mut` because `step()` updates internal state:

```rust
let mut optimizer = Adam::new(&params, 0.001);
```

## 2. References: `&x`

```rust
let a = Tensor::zeros(&[2, 3], opts)?;
let b = Tensor::ones(&[2, 3], opts)?;
let c = a.add(&b)?;  // &b = "borrow b without taking ownership"
// b is still usable here
```

**PyTorch equivalent:** In Python, everything is passed by reference
automatically. Rust makes you write `&` to say "I'm borrowing this, not
consuming it." Most floDl methods take `&self` or `&Tensor` - you'll see `&`
everywhere, and it always means the same thing: "use it, don't move it."

## 3. The `?` Operator and `Result<T>`

GPU operations can fail (shape mismatch, out of memory, etc.). Python raises
exceptions. Rust returns `Result<T>` - either `Ok(value)` or `Err(error)`:

```rust
// The ? operator: "if this failed, return the error immediately"
let y = x.matmul(&w)?.add(&b)?.relu()?;
```

This is equivalent to:

```python
# Python - exceptions propagate implicitly
y = (x @ w + b).relu()
```

The only difference: Rust makes you acknowledge each fallible call with `?`.
Chain them freely - a single `?` per call is all you need.

Your `main` function returns `Result<()>` to enable `?`:

```rust
fn main() -> Result<()> {
    // ... use ? freely ...
    Ok(())  // "everything succeeded"
}
```

## 4. Closures: `|| {}`

```rust
// Rust closure - like Python's lambda, but multi-line
let make_batch = || {
    let x = Tensor::randn(&[16, 2], opts).unwrap();
    let y = Tensor::randn(&[16, 2], opts).unwrap();
    (x, y)
};

let (x, y) = make_batch();
```

You'll see closures in `no_grad`, iterators, and data generation:

```rust
// No-grad inference (like `with torch.no_grad():`)
let pred = no_grad(|| model.forward(&input))?;

// Generate batches
let batches: Vec<_> = (0..32).map(|_| make_batch()).collect();
```

## 5. Vectors, Slices, and Iteration

```rust
// Vec<T> - like Python's list, but typed
let losses: Vec<f64> = Vec::new();

// Slices &[T] - a view into contiguous data
let shape: &[i64] = &[2, 3];
let t = Tensor::zeros(shape, opts)?;

// Iteration - like Python's for loop
for (input, target) in &batches {
    let pred = model.forward(&input)?;
    // ...
}

// Ranges
for epoch in 0..100 {
    // epoch goes 0, 1, 2, ..., 99
}
```

**Shape arguments** are always `&[i64]` slices: `&[2, 3]`, `&[batch, features]`.

## 6. Traits: Like Abstract Classes

A trait defines behavior that types can implement. `Module` is the key trait:

```rust
// floDl's Module trait (simplified)
trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable>;
    fn parameters(&self) -> Vec<Parameter>;
}
```

Every layer (`Linear`, `GELU`, `LayerNorm`) and every `Graph` implements
`Module`. You call `.forward()` on any of them the same way:

```rust
let pred = model.forward(&input)?;   // works for Linear, Graph, anything
```

**PyTorch equivalent:** `nn.Module` - same concept, Rust just enforces the
interface at compile time instead of duck-typing it.

## 7. Types You'll See But Don't Need to Understand

These appear in signatures and error messages. You don't need to construct them:

| Type | What it means | When you see it |
|------|---------------|-----------------|
| `Result<T>` | "Might fail" - use `?` | Every tensor/module operation |
| `Option<T>` | "Might be absent" - `Some(v)` or `None` | `x.grad()` returns `Option<Tensor>` |
| `Vec<T>` | Growable array | `parameters()`, shape data |
| `&[T]` | Slice (borrowed view) | Shape arguments: `&[2, 3]` |
| `Box<dyn Module>` | Any module, heap-allocated | Inside `modules![...]` macro |
| `Rc<RefCell<...>>` | Shared mutable state | Variable internals (you never touch this) |
| `f64`, `f32`, `i64` | Number types | Loss values, shapes, indices |
| `usize` | Unsigned index | Loop counters, `.len()` |
| `()` | "Nothing" (like Python's None) | `Result<()>` = "succeeds or fails, no value" |

## 8. Common Gotchas

### Move semantics

```rust
let a = Tensor::zeros(&[2, 3], opts)?;
let b = a;       // a is "moved" into b - a is no longer valid
// println!("{:?}", a);  // compile error: a was moved

// Fix: clone if you need both
let a = Tensor::zeros(&[2, 3], opts)?;
let b = a.clone();  // shallow: b shares a's storage (an in-place op on one
                    // is seen by the other). UNLIKE PyTorch's deep .clone();
                    // use a.copy() for an independent, owned copy.
// both a and b are valid
```

In practice, floDl methods take references (`&self`, `&Tensor`), so moves are
rare in training code. The compiler tells you when it happens.

### Mutable borrow rules

```rust
let mut v = vec![1, 2, 3];
// Can't borrow mutably and immutably at the same time:
// let r = &v[0];
// v.push(4);    // compile error - v is borrowed by r
// println!("{}", r);

// Fix: finish using the immutable borrow first
let r = v[0];    // copy the value
v.push(4);       // now fine
```

You'll rarely hit this in floDl code. When you do, the compiler error message
tells you exactly what's conflicting.

### Semicolons matter

```rust
fn add_one(x: i64) -> i64 {
    x + 1      // no semicolon = this is the return value
}

fn add_one_v2(x: i64) -> i64 {
    return x + 1;  // explicit return also works
}
```

The last expression without a semicolon is the return value. Add a semicolon
and it becomes a statement that returns `()` - the compiler will tell you if
you get this wrong.

### Turbofish `::<Type>`

Occasionally Rust needs a type hint:

```rust
let data: Vec<f32> = t.to_f32_vec()?;
```

In floDl code, type annotations on `let` bindings are usually enough.

## 9. String Formatting

```rust
// println! with {} placeholders (like Python's f-strings)
println!("Epoch {}: loss = {:.4}", epoch, loss_val);
//                          ^^^^ 4 decimal places

// Format strings
let msg = format!("loss: {:.6}", loss_val);
```

| Format | Output | Python equivalent |
|--------|--------|-------------------|
| `{}` | Default | `{}` |
| `{:.4}` | 4 decimal places | `{:.4f}` |
| `{:>10}` | Right-align, width 10 | `{:>10}` |
| `{:?}` | Debug output | `repr()` |

## 10. The Full Picture: PyTorch vs floDl Training Loop

```python
# PyTorch
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.GELU(),
    nn.Linear(32, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epoch in range(200):
    for x, y in batches:
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

```rust
// floDl
use flodl::*;

fn main() -> Result<()> {
    let model = FlowBuilder::from(Linear::new(1, 32)?)
        .through(GELU)
        .through(Linear::new(32, 1)?)
        .build()?;

    let params = model.parameters();
    let mut optimizer = Adam::new(&params, 0.01);
    model.train();

    for epoch in 0..200 {
        let mut last_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), true);
            let target = Variable::new(y.clone(), false);

            optimizer.zero_grad();
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;
            last_loss = loss.item()?;
            loss.backward()?;
            optimizer.step()?;
        }
        println!("Epoch {}: loss={:.4}", epoch, last_loss);
    }
    Ok(())
}
```

**What's different:**
- `use flodl::*` instead of `import torch`
- Shapes are slices: `&[1, 32]` not `(1, 32)`
- Every fallible call has `?`
- Tensors become `Variable` for gradient tracking
- `model.forward(&input)` not `model(input)`
- `fn main() -> Result<()>` + `Ok(())` at the end

**What's the same:**
- Build model, create optimizer, training loop, zero_grad/forward/backward/step
- The structure is identical - Rust just makes error handling and ownership explicit

---

**You're ready.** The patterns above cover 95% of what you'll write in floDl.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Porting from PyTorch](../porting.md) | Next: [PyTorch Migration Reference](../pytorch_migration.md)
