---
title: "One struct, one manifest, one overlay"
subtitle: "v0.5.0: #[derive(FdlArgs)] for any Rust binary, a consolidated fdl.yml, and explainable per-environment config with origin annotations"
date: 2026-04-18
description: "flodl v0.5.0 ships the fdl CLI maturity pass. New proc-macro crate flodl-cli-macros adds #[derive(FdlArgs)] so any Rust binary gets typed argv, JSON schema, shell completions, and env-var fallback from a single struct. fdl.yml consolidates to one commands: map with three clean kinds. Environment overlays with fdl config show surface merged config with per-field origin annotations."
---

`fdl` has shipped standalone since 0.3.0. It has been on crates.io
since 0.4.0. What 0.5.0 delivers is the next layer up: the CLI
becomes something you can **program against** from your own Rust
binaries, with a manifest that consolidates into one clean shape and
configuration you can actually see before you run anything.

Three deliverables, one theme: **one source of truth for every
surface**.

## One struct for argv, help, schema, completion, and env fallback

The new proc-macro crate
[`flodl-cli-macros`](https://crates.io/crates/flodl-cli-macros)
exposes `#[derive(FdlArgs)]`, re-exported by `flodl-cli` as
`flodl_cli::FdlArgs`. You write the struct; the derive generates the
argv parser, the JSON schema emitter, the ANSI-coloured help
renderer, and the environment-variable fallback.

```rust
use flodl_cli::{FdlArgs, parse_or_schema};

/// Run the training benchmark suite.
#[derive(FdlArgs, Debug)]
struct BenchArgs {
    /// Model to train (or `all` for the full suite).
    #[option(short = 'm',
             choices = &["all", "resnet", "gpt-nano", "char-rnn"],
             default = "all")]
    model: String,

    /// Epochs to run.
    #[option(short = 'e', default = "10")]
    epochs: u32,

    /// API key (falls back to env when absent).
    #[option(env = "WANDB_API_KEY")]
    wandb_key: Option<String>,

    /// Extra dataset paths.
    #[arg(variadic)]
    datasets: Vec<String>,
}

fn main() {
    let args: BenchArgs = parse_or_schema();
    // args.model, args.epochs, args.wandb_key, args.datasets are typed.
}
```

With this in place:

- `--help` renders an ANSI-coloured help page assembled from the
  doc-comments. Descriptions, choices, defaults, types - all there,
  no hand-written banner.
- `--fdl-schema` emits JSON describing every flag. `fdl` probes the
  binary on first use and caches the result at
  `<cmd_dir>/.fdl/schema-cache/<cmd>.json`.
- `fdl bench --model <TAB>` offers the declared `choices` in any
  completion-enabled shell.
- `fdl bench --wandb-key foo` works, and so does leaving the flag
  off with `WANDB_API_KEY=foo` in the environment.
- Unknown flags and invalid choices fail with a clear error before
  your binary runs.

Reserved flag names (`--help`, `--version`, `--quiet`, `--env`) can
no longer be shadowed by accident - collisions error at derive time
with a message pointing at the offending field. This is one of the
two breaking changes in 0.5.0.

We dogfooded the whole thing: `ddp-bench` in 0.5.0 is an `FdlArgs`
binary. The hand-rolled argv handling from 0.4.0 is gone. The help
you see when you type `fdl ddp-bench --help` is rendered from the
same struct the parser validates against. One source of truth, end
to end.

## One manifest shape

`fdl.yml` in 0.4.0 had two top-level maps (`scripts:` for shell
commands, `commands:` for docker-wrapped entries with structured
config). In 0.5.0 they merge into one `commands:` map, and each
entry comes in exactly one of three kinds:

- **`run:`** - inline shell (optionally wrapped in `docker compose`).
- **`path:`** - pointer to a nested sub-project with its own
  `fdl.yml`.
- **preset** - deep-merge structured `ddp:` / `training:` /
  `options:` config over an enclosing `entry:`.

The three-kind model is the long-term stable surface. No further
breaking changes to its shape are scheduled. If you had a 0.4.0
`fdl.yml`, the upgrade is boring - wrap your `scripts:` values in
`run:` and move on. Step-by-step in
[UPGRADE.md](https://github.com/flodl-labs/flodl/blob/main/UPGRADE.md).

Load-time validation tells you exactly which file, which key, and
which rule failed. `docker:` on non-`run:` entries is rejected.
Kind-mismatches (both `run:` and `path:` set) error loudly. Unknown
keys are called out by name.

## One overlay, explained

This is the feature I wanted for a year and never wrote because
scaffolding didn't deserve it. Now it does.

```sh
fdl --env ci test           # loads fdl.ci.yml on top of fdl.yml
FDL_ENV=ci fdl test         # same, via env var
fdl ci test                 # first-arg convention, when "ci" doesn't
                            # collide with a real command
```

The overlay is a deep merge, field by field. And then `fdl config
show` prints the result with per-field origin annotations:

```
$ fdl --env ci config show
# resolved manifest, base + overlay
docker: dev                     # fdl.yml:2
ddp:
  policy: cadence               # fdl.yml:8
  backend: cpu                  # fdl.ci.yml:3 (override)
  divergence_threshold: 0.05    # fdl.yml:10
training:
  epochs: 1                     # fdl.ci.yml:6 (override)
  seed: 42                      # fdl.yml:13
  batch_size: 32                # fdl.yml:14
```

Every field tagged with the file and line that contributed it. You
see the merged reality before running a job that might take two
hours. Override annotations surface accidental shadowing loudly. If
you've ever burned a weekend training run because your CI overlay
silently picked up the wrong `lr_scale_ratio`, this is the sanity
check you wanted.

Explicit selectors (`--env`, `FDL_ENV`) fail loudly on missing
files. The first-arg convention (`fdl ci test`) silently falls
through to normal dispatch when no matching file exists, so
existing commands are never shadowed.

## Why the derive matters

The derive is the biggest of the three, and the least obvious.

`fdl` already knew how to dispatch commands, run scripts, probe
hardware, manage libtorch. What it didn't know how to do was **let
your binary play the same way its built-ins play**. Built-in
sub-commands in 0.4.0 had ANSI help, typed flags, schema-driven
completion. Your `cargo run --bin my-training` didn't, because there
was no public contract for participating.

`#[derive(FdlArgs)]` is that contract. Your binary exports its flag
surface through `--fdl-schema`; `fdl` caches it; help, validation,
and completion are driven from the same struct the parser reads
from. Writing a custom training harness doesn't mean rebuilding the
CLI ergonomics from scratch - it means adding `#[derive(FdlArgs)]`
above a struct.

The derive is ~1100 lines of proc-macro code, zero runtime
dependencies on any argv-parsing crate. `flodl-cli` itself doesn't
pull in `clap` or `structopt` and neither do binaries that use the
derive - the generated parser is self-contained.

## What's still the same

The framework API. Nothing in `flodl` itself broke in 0.5.0. The
tensor, autograd, nn, graph, data, monitor, and distributed modules
are all unchanged at their public surfaces. If you have 0.4.0 code
that doesn't touch `fdl.yml` or `#[derive(FdlArgs)]`, upgrading is a
version bump.

## Where to go next

- **Read the derive reference**:
  [declaring flags in Rust](https://github.com/flodl-labs/flodl/blob/main/docs/cli/05-manifest.md#declaring-flags-in-rust)
  covers the attribute surface with a worked example.
- **Upgrade your 0.4.0 project**:
  [UPGRADE.md](https://github.com/flodl-labs/flodl/blob/main/UPGRADE.md)
  is short. Breaking changes are `scripts:` to `commands:` and a
  handful of reserved flag names.
- **Preview your config**: once upgraded, run
  `fdl config show` (and `fdl --env <name> config show` if you use
  overlays) to see what actually merges.
- **Read the full CLI reference**:
  [docs/cli/01-install.md](https://github.com/flodl-labs/flodl/blob/main/docs/cli/01-install.md)
  covers all three contexts - standalone, in-project, flodl source
  checkout.

And if the whole framing of a CLI you can extend rather than wrap
sounds strange, read [out of the cave](/out-of-the-cave). That's
where this came from.
