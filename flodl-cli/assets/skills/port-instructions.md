# Port Agent Instructions

You are porting a PyTorch script to flodl (Rust deep learning framework).
Follow this process exactly.

## Step 0: Check the Environment

Before writing any code, determine what the user has available:

```!
which fdl 2>/dev/null && fdl version || echo "fdl: not installed"
which cargo 2>/dev/null && cargo --version || echo "cargo: not installed"
which docker 2>/dev/null && docker --version 2>/dev/null || echo "docker: not installed"
```

Based on results:

- **fdl + cargo available**: User can build natively. Scaffold with
  `fdl init <project-name>` (mounted libtorch mode).
- **fdl + docker available (no cargo)**: Scaffold with
  `fdl init <project-name> --docker` (libtorch baked into Docker image,
  no Rust needed on host).
- **Neither fdl nor cargo**: Guide the user to install fdl first:
  `cargo install flodl-cli` (if Rust available) or download the
  pre-compiled binary: `curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl`
  then `./fdl install` to make it global.

If the user does not already have a project scaffolded, offer to create
one before porting. The scaffold provides Cargo.toml, Dockerfile,
docker-compose.yml, Makefile, and a training template, so you only need
to replace src/main.rs with the ported code.

If the user has GPU(s), suggest running `fdl setup` to auto-detect
hardware and download the right libtorch variant.

## Step 1: Get the API Reference

Run this command to get the current flodl API surface:

```!
fdl api-ref 2>/dev/null || echo "FALLBACK: fdl not available"
```

If fdl is not available, read `flodl/src/lib.rs` and explore the source
tree to understand what's available. Focus on `flodl/src/nn/` for modules
and `flodl/src/graph/flow.rs` for the FlowBuilder.

For the CLI surface (project setup, libtorch, fdl.yml dispatch, the
`fdl flodl-hf` sub-commands, etc.), use `fdl --help` recursively:
`fdl --help` lists top-level commands, `fdl <cmd> --help` drills into
one, `fdl <cmd> <sub> --help` drills further. Help text is
auto-generated from the source via `#[derive(FdlArgs)]`, so it reflects
the user's installed version and is the most reliable source for the
current CLI surface.

## Step 2: Read the Porting Guide

Read the porting guide at `ai/skills/port/guide.md` for the complete
PyTorch-to-flodl mapping. This is your reference for all translations.

## Step 3: Analyze the PyTorch Source

Read the entire PyTorch script. Classify every block:

- **Model**: classes inheriting nn.Module
- **Data**: datasets, dataloaders, transforms
- **Training**: epoch loops, backward, optimizer step
- **Loss**: criterion setup and computation
- **Optimizer**: optimizer and scheduler setup
- **Checkpoint**: save/load patterns
- **Inference**: eval mode, no_grad blocks
- **Distributed**: `torch.distributed.*`, `DistributedDataParallel`,
  `DataParallel`, `torchrun` launcher, `mp.spawn`, `init_process_group`,
  `dist.barrier`, `dist.all_reduce`. Flag this explicitly -- it changes
  the target entry point in Step 4.
- **Utility**: logging, metrics, visualization

List what you found before starting to port.

## Step 4: Design the flodl Architecture

Before writing code, decide:

1. **Graph or manual?** If the model is a feed-forward pipeline (even with
   branches, residuals, loops), use FlowBuilder. If it has complex control
   flow (GAN with two models, RL with environment interaction), use manual
   Module implementations.

2. **Tags**: Identify outputs that are reused (skip connections, losses on
   intermediate layers, observation points). Each becomes a `.tag("name")`.

3. **Data flow**: Map the PyTorch forward() data flow to FlowBuilder
   operations: `.through()` for sequential, `.also()` for residual,
   `.split()/.merge()` for parallel, `.tag()/.using()` for cross-connections,
   `.loop_body()` for iteration.

4. **Project structure**: One `src/main.rs` for simple scripts. Separate
   modules for complex projects.

5. **Distributed?** If Step 3 flagged a Distributed block, route the
   training loop through flodl's one training entry instead of the manual
   `forward / backward / step` loop. flodl unifies data loading and
   training under DDP:
   - `Trainer::builder(model_factory, optim_factory, train_fn)
     .dataset(...).batch_size(...).num_epochs(...).run()?` (or the
     config-bag `Trainer::run(model_factory, optim_factory, train_fn, cfg)`).
     Any model works: a `Graph` is just a `Module`, so return it from
     `model_factory`. It auto-detects GPUs: 0-1 trains inline, 2+
     auto-promotes to process-per-rank. Tune DDP cadence with
     `.elche(ElCheConfig::nccl_cadence())` (five modes).
   - For explicit per-rank gradient-sync / broadcast control, use
     `Ddp::wrap(&model, device, rank, &rendezvous)?` (bypass tier). To
     own the loop body while the controller schedules, use
     `Trainer::builder(...).into_worker()?` (cooperative tier).
   - See `ai/skills/port/guide.md` Phase 3 "Distributed Training" for the
     full mapping and `docs/ddp/01-reference.md` for the reference.

## Step 5: Generate the Port

Write the complete Rust project:

1. `Cargo.toml` with flodl dependency
2. `src/main.rs` (or split into modules)
3. DataSet implementation if custom data loading
4. Training loop

Use `flodl::*` for imports. Every constructor returns Result, use `?`.

## Step 6: Validate

Run `cargo check`. Read errors carefully:

- **"expected Result"**: add `?` to constructor calls
- **"borrowed value does not live long enough"**: restructure to keep
  temporaries alive
- **"no method named X"**: check the API reference, the method may have
  a different name or be on a different type
- **"trait Module is not implemented"**: wrap in FlowBuilder or implement
  Module manually

Fix errors one by one. Re-run `cargo check` after each fix until clean.

## Rules

- Do NOT invent flodl APIs that don't exist. If the API reference doesn't
  list a function, it doesn't exist. Note it as a gap.
- Do NOT use `unsafe` code.
- Prefer FlowBuilder over manual Module implementations when possible.
- Use `.tag()` generously. Tags are cheap and make models observable.
- Use `on_device()` constructors in tests. Never hardcode `Device::CPU`.
- Loss functions are free functions (not structs): `mse_loss(&pred, &target)?`
- Always add error handling: `fn main() -> flodl::tensor::Result<()>`
