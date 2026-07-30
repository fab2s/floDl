# The fdl.yml Manifest

Any directory (or ancestor) that contains `fdl.yml`, `fdl.yaml`, or
`fdl.json` is a floDl project in the manifest sense. In that context,
`fdl` doubles as a project task runner: `fdl <name>` dispatches into
the manifest, and a small set of meta-commands (`fdl config`,
`fdl schema`, plus the manifest sub-commands themselves) become
available.

If only `fdl.yml.example` (or `.dist`) exists, `fdl` offers to copy it
to the real (gitignored) `fdl.yml` so users can customise locally.

## The general principle

`fdl.yml` gives you four composable building blocks:

1. **Link any script as a sub-command** via `run:`.
2. **Declare arguments and options in Rust** on binaries via
   `#[derive(FdlArgs)]`. `fdl` probes them with `--fdl-schema` and
   inherits typed help + completion for free.
3. **Layer environments** with `fdl.<env>.yml` overlays (dev / ci /
   prod variations of the same command tree).
4. **Fall back to shell environment variables** per-option with
   `#[option(env = "…")]` - argv wins, then env var, then default.

End-to-end example. A cargo-backed sub-command with Rust-declared flags,
an env overlay, and an env-var fallback for a secret:

```rust
// src/bin/train.rs in your project
use flodl_cli::FdlArgs;

/// Train the model.
#[derive(FdlArgs, Debug)]
pub struct TrainArgs {
    /// Device to train on.
    #[option(choices = &["cpu", "cuda"], default = "cuda")]
    pub device: String,

    /// Number of epochs.
    #[option(short = 'e', default = "10")]
    pub epochs: u32,

    /// Weights & Biases API key (argv > env > absent).
    #[option(env = "WANDB_API_KEY")]
    pub wandb_api_key: Option<String>,

    /// Dataset path.
    #[arg]
    pub dataset: std::path::PathBuf,
}
```

```yaml
# fdl.yml -- base manifest
description: My training project

commands:
  # Path-kind: loads ./train/fdl.yml, which declares an `entry:` pointing
  # at the cargo binary. Extra argv after `fdl train ...` flows through
  # to the entry, validated against the FdlArgs schema.
  train:
```

```yaml
# train/fdl.yml -- sub-command configuration
description: Train the model
docker: dev
entry: cargo run --release --bin train --
```

```yaml
# fdl.ci.yml -- CI overlay, deep-merged over fdl.yml
commands:
  train:
    # Overlay the child's entry for CI: CPU, one epoch.
    entry: cargo run --release --bin train -- --device cpu --epochs 1
```

Usage:

```bash
# Base config: GPU training, 10 epochs. Extra args flow to the binary.
fdl train ./data/train.bin --epochs 50

# CI overlay: CPU, one epoch. Secret picked up from the environment.
WANDB_API_KEY=xxx fdl --env ci train ./data/train.bin

# Introspect the fully-resolved config (base + overlay).
fdl config show ci

# Help flows through #[derive(FdlArgs)] -> --fdl-schema -> render_help,
# so values, choices, and env fallbacks are all visible here.
fdl train --help
```

Path-kind sub-commands with an `entry:` forward every extra argv token to
the underlying binary, where the derived parser validates it. `run:`-kind
commands (shown in the next section) forward argv only after an explicit
`--` separator - `fdl test-live -- -p flodl-hf` splices `-p flodl-hf`
into the script, while `fdl test-live -p flodl-hf` errors loudly. Stray
args before `--` are rejected with a hint pointing at the right form.

`#[derive(FdlArgs)]` is re-exported as `flodl_cli::FdlArgs`. See the
[`flodl-cli-macros`
README](https://crates.io/crates/flodl-cli-macros) for the full
attribute surface (`short`, `default`, `choices`, `env`, `completer`,
`variadic`).

## Command kinds: `run` / `path` / preset

`fdl.yml` declares a unified `commands:` map. Each entry is exactly one
of three kinds, chosen by which fields are set:

- **Run** - `run:` is set. Executes the inline shell script, optionally
  wrapped in `docker compose run --rm <service>` when `docker:` is set.
  An optional `append:` field declares literal trailing tokens
  (typically the libtest `-- --nocapture --ignored` portion) that should
  follow any user-supplied args.
- **Path** - `path:` is set (or, by convention, the entry is empty/null
  and a sibling directory named `<command>/` with its own `fdl.yml`
  exists). Loads the nested manifest and recurses.
- **Preset** - neither `run:` nor `path:` is set. Inline `ddp:` /
  `training:` / `output:` / `options:` fields merge over the enclosing
  config and invoke its `entry:`. Only legal inside a sub-command
  (path-kind entry's own `fdl.yml`).

```yaml
description: flodl - Rust deep learning framework

commands:
  test:
    description: Run all CPU tests
    run: cargo test
    append: -- --nocapture
    docker: dev
  cuda-test:
    description: Run CUDA tests (parallel)
    run: cargo test --features cuda
    append: -- --nocapture
    docker: cuda
  shell:
    run: bash
    docker: dev
  ddp-bench:          # convention default: loads ./ddp-bench/fdl.yml
```

```bash
fdl test                              # runs "test" in the "dev" docker service
fdl cuda-test                         # runs in the "cuda" service
fdl test -- -p flodl-hf --test foo    # forwards `-p flodl-hf --test foo` to cargo
fdl shell                             # opens an interactive shell
fdl ddp-bench --list                  # dispatches into the ddp-bench sub-command
```

When a `run:` command declares `docker: <service>`, `fdl` wraps it in
`docker compose run --rm <service> bash -c "…"`. Without `docker:`, it
runs on the host. `docker:` is only valid on `run:` commands -
declaring it on a `path:` or preset entry is rejected at load time.

#### Forwarding extra args with `--` and `append:`

`run:`-kind commands accept user args after an explicit `--` separator
on the CLI. The composed shell command is:

```
[run:]  +  [user args after --]  +  [append:]
```

Args before `--` are rejected loudly (with a hint showing the right
form). Args after `--` are POSIX-quoted and spliced between the run
line and the append suffix. So:

```yaml
test-live:
  run: cargo test live
  append: -- --nocapture --ignored
  docker: dev
```

```bash
fdl test-live                                       # cargo test live -- --nocapture --ignored
fdl test-live -- -p flodl-hf                        # cargo test live -p flodl-hf -- --nocapture --ignored
fdl test-live -- --test xlm_roberta_parity          # cargo test live --test xlm_roberta_parity -- --nocapture --ignored
fdl test-live -p flodl-hf                           # error: use `fdl test-live -- -p flodl-hf`
```

`append:` is purely structural: it lets the script author reserve
trailing tokens (libtest harness flags, fixed test-name filters, etc.)
that should always follow any user-supplied args. There is no opt-in or
opt-out flag; the user typing `--` *is* the explicit forwarding signal.
Commands without an `append:` simply receive the user args at the tail.

`append:` without `run:` is rejected at load time: it only forwards
tokens for inline run-scripts.

## Declaring flags in Rust

Binaries can declare their argv surface with `#[derive(FdlArgs)]`. The
derive wires a hidden `--fdl-schema` flag that emits JSON describing
every option and positional; `fdl` runs the entry with that flag
(explicitly via `fdl schema refresh` for cargo entries, automatically
for script/pre-built-binary entries), caches the JSON under
`<cmd-dir>/.fdl/schema-cache/<cmd>.json`, and uses it to drive:

- `fdl <cmd> --help` - typed, color-annotated help rendered from the
  doc-comments and attributes.
- Shell completion - choices, short/long forms, value types.
- Validation - unknown flags error with a clear message.

One struct is the single source of truth. The doc-comments become help
text. The attribute metadata becomes schema. The struct fields become
typed values in your `main()`.

```rust
use flodl_cli::{FdlArgs, parse_or_schema};

/// Run the training benchmark suite.
#[derive(FdlArgs, Debug)]
pub struct BenchArgs {
    /// Model to train (or `all` for the full suite).
    #[option(short = 'm', choices = &["all", "linear", "mlp", "lenet",
                                      "resnet", "char-rnn", "gpt-nano"],
             default = "all")]
    pub model: String,

    /// DDP mode to exercise.
    #[option(choices = &["solo-0", "nccl-cadence",
                         "cpu-cadence", "cpu-async"],
             default = "nccl-cadence")]
    pub mode: String,

    /// Epochs to run (overrides the preset default).
    #[option(short = 'e', default = "10")]
    pub epochs: u32,

    /// Write a Markdown convergence report to this path.
    #[option]
    pub report: Option<String>,

    /// Weights & Biases API key (read from env if flag absent).
    #[option(env = "WANDB_API_KEY")]
    pub wandb_key: Option<String>,

    /// Extra dataset paths to include.
    #[arg(variadic)]
    pub datasets: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: BenchArgs = parse_or_schema();
    // args.model, args.mode, args.epochs, ... are typed values.
    Ok(())
}
```

With this struct in place:

- `cargo run --bin bench -- --help` renders an ANSI-coloured help page
  with the doc-comments as descriptions.
- `cargo run --bin bench -- --fdl-schema` emits JSON describing every
  flag. `fdl` calls this on first use and caches the result.
- `fdl bench --model <TAB>` in a completion-enabled shell offers
  `all linear mlp lenet resnet char-rnn gpt-nano`.
- `fdl bench --wandb-key <value>` works, and so does leaving the flag
  off with `WANDB_API_KEY=...` in the environment.
- Unknown flags and invalid choices fail with a clear error before your
  binary starts.

#### Attribute reference

Each field must carry exactly one of `#[option(...)]` (named flag,
kebab-cased from the field name) or `#[arg(...)]` (positional). The
field's Rust type determines cardinality.

| Shape       | Meaning                                               |
|-------------|-------------------------------------------------------|
| `bool`      | Flag is present or absent; no value. Absent = `false`. |
| `T`         | Scalar, required. `#[option]` must supply `default`.  |
| `Option<T>` | Scalar, optional. Absent = `None`.                    |
| `Vec<T>`    | `#[option]`: repeatable. `#[arg]`: variadic (last).    |

`#[option]` keys:

| Key         | Example          | Notes                                                     |
|-------------|------------------|-----------------------------------------------------------|
| `short`     | `'c'`            | Single-char short flag.                                   |
| `default`   | `"string"`       | Parsed via `FromStr` at run time; required on bare `T`.   |
| `choices`   | `&["a", "b"]`    | Accepted values; enforced by the parser.                  |
| `env`       | `"VAR_NAME"`     | Env fallback when the flag is absent; skipped on `bool`.  |
| `completer` | `"name"`         | Named completer for shell completion scripts.             |

`#[arg]` keys:

| Key         | Example          | Notes                                                     |
|-------------|------------------|-----------------------------------------------------------|
| `default`   | `"string"`       | Makes the positional optional.                            |
| `choices`   | `&["a", "b"]`    | Accepted values.                                          |
| `variadic`  | bare or `= true` | Requires `Vec<T>`; must be the last positional.           |
| `completer` | `"name"`         | Named completer for shell completion scripts.             |

Validation runs at derive time: required positionals cannot follow
optional ones, variadic must be last, reserved flags cannot be
shadowed, and duplicate long/short flags error out. Errors point at
the offending field, not at a run-time parser message.

See the [`fdl schema`](#fdl-schema-and---fdl-schema) section for how to
refresh the cache after rebuilding, and the
[`flodl-cli-macros`](https://crates.io/crates/flodl-cli-macros) README
and [`flodl-cli` docs.rs page](https://docs.rs/flodl-cli) for the full
attribute surface and internals.

## Environment overlays

An environment selector tells `fdl` to deep-merge `fdl.<name>.yml` on
top of the base `fdl.yml` before resolving any command. Three equivalent
forms are supported; a command-line selector (`@` or `--env`) overrides
`FDL_ENV`:

1. `fdl @<name> <cmd>` - `@` sigil token. Accepted anywhere before a
   standalone `--` (`fdl @ci test` ≡ `fdl test @ci`), exactly like `--env`.
2. `fdl --env <name> <cmd>` - explicit flag.
3. `FDL_ENV=<name> fdl <cmd>` - environment variable.

```bash
fdl @ci test                         # sigil form
fdl --env ci test                    # flag form
FDL_ENV=ci fdl test                  # env var form
```

Every form must resolve to an existing `fdl.<name>.yml` overlay or it
fails loudly. There is no bare-positional form: `fdl ci test` treats `ci`
as a command, so an env overlay may freely share a name with a command.
Supplying `@` and `--env` with different values is a conflict error. A
literal `@`-prefixed argument for an inner command goes after `--`.

Typical overlay files:

- `fdl.dev.yml` - fast iteration (shorter epochs, smaller batches).
- `fdl.ci.yml` - CPU-only, minimal epochs, strict validation.
- `fdl.prod.yml` - full runs, checkpoint to cloud storage.

Use `fdl config show <env>` to preview the resolved merged config.

## Preset sub-commands

A sub-command directory (e.g. `ddp-bench/`) has its own `fdl.yaml` with
an `entry:`, optional `docker:`, structured `ddp` / `training` /
`output` sections, and a `commands:` map whose entries are **presets** -
inline overrides of this config's `entry`:

```yaml
description: DDP validation and benchmark suite
docker: cuda
entry: cargo run --release --features cuda --

training:
  epochs: 5
  seed: 42

# Each `mode:` value below maps 1:1 to flodl's ElCheMode variants:
# nccl-sync, nccl-cadence (default), cpu-sync, cpu-cadence, cpu-async.
# Plus solo-N for single-GPU baselines (N = physical device index).

commands:
  quick:
    description: Fast smoke test
    training: { epochs: 1 }
    options: { model: linear, mode: solo-0, batches: 100 }
  validate:
    options: { model: all, mode: all, validate: true }
```

Then:

```bash
fdl ddp-bench quick                   # runs the "quick" preset
fdl ddp-bench validate --report out   # preset + extra flags
fdl ddp-bench --help                  # description + presets + defaults
fdl ddp-bench validate --help         # resolved options
```

A sub-command's `commands:` may mix kinds freely: a preset sits
alongside a nested `path:` (another directory) or a standalone `run:`
helper. `fdl <cmd> --help` splits them into an **Arguments** section
(the single preset slot, with values indented underneath - override the
placeholder via `arg-name:`) and a **Commands** section (real
sub-commands with their own behaviour).

Preset `options:` entries map 1:1 to the binary's `#[derive(FdlArgs)]`
shape - `mode`, `model`, `epochs`, `batch-size`, etc. The set of
accepted `mode` values mirrors `ElCheMode` (five modes:
`nccl-sync` / `nccl-cadence` / `cpu-sync` / `cpu-cadence` /
`cpu-async`) plus the `solo-N` single-GPU baselines. See
[DDP Reference: ElCheMode](../ddp/01-reference.md#elchemode---cadence--backend-in-one-name)
for the mode semantics.

## `fdl config`

Inspect the resolved project configuration, with or without an overlay
applied.

```bash
fdl config show              # base fdl.yml
fdl config show ci           # base deep-merged with fdl.ci.yml
fdl --env ci config show     # same result, via the flag form
fdl @ci config show          # same result, via the @ sigil
```

The output is the fully-merged YAML with per-layer annotations, so you
can see which file contributed which field. Useful for debugging
overlay behaviour before running a long job.

## `fdl schema` and `--fdl-schema`

Any entry that responds to a hidden `--fdl-schema` flag by emitting a
JSON description of its arguments and options becomes a self-describing
sub-command. `fdl` uses the result to power help, completion, and
validation, caching the output per-command.

Two ways to opt in:

- **Rust binaries** - `#[derive(FdlArgs)]` wires `--fdl-schema`
  automatically (see [Declaring flags in Rust](#declaring-flags-in-rust)).
- **Scripts and pre-built tools** - emit the JSON yourself. A few lines
  of shell/Python/whatever at the top of the entry, exit 0 before any
  real work. The shape is the same JSON object that the derive macro
  emits (`{"options": {...}, "args": [...], "strict": bool}`). See
  `benchmarks/run.sh` for a reference implementation.

```bash
fdl schema list              # every cached schema with fresh/stale/orphan status
fdl schema list --json       # machine-readable
fdl schema clear             # delete all cached schemas
fdl schema clear ddp-bench   # delete one
fdl schema refresh           # re-probe every entry and rewrite the cache
fdl schema refresh ddp-bench # refresh one
```

Cached schemas live at `<cmd-dir>/.fdl/schema-cache/<cmd>.json`.

**Non-cargo entries auto-probe** on first use (or when the cache goes
stale after an `fdl.yml` edit). Scripts and pre-built binaries get
their schema into the cache without any manual step - `fdl <cmd>
--help` on a fresh clone just works.

**Cargo entries must be built before `refresh`** - `fdl` runs the
entry's `--fdl-schema` as a subprocess, which requires the binary to
exist. To avoid the compile latency ruining `--help`, cargo entries
are **never** auto-probed unless the command opts in with
`compile: true`.

## What makes a cached schema stale

The cache is compared against every file whose edit could change the
schema: the command's `fdl.yml`, and - for a `compile: true` command -
every `.rs` and `Cargo.toml` under the command directory.

That second half matters, because the schema of a cargo entry is
*compiled from* those sources. Watching only `fdl.yml` meant adding a
flag left the cache stale with no signal at all: `-h` kept rendering
the previous surface until someone happened to touch the yml or delete
the cache by hand. Now a new flag shows up on its own.

The scan is deliberately coarse - the whole command directory rather
than an attempt to find the files that *define* the schema. Watching
too much costs one extra probe, which is exactly what editing the yml
already costs; watching too little is the silent-stale bug. It is also
cheap enough to sit in front of `-h` (23 files in ~0.1 ms for
`ddp-bench`), skips `target/`, and is bounded so a pathological tree
degrades to config-only invalidation rather than stalling help.

Dependency crates are **not** followed. A command's schema comes from
its own CLI struct, and following its dependencies would invalidate the
cache on every edit anywhere in the workspace - recompiling on nearly
every `-h`, which is the cost the cache exists to avoid.

```bash
cargo build --release --features cuda
fdl schema refresh ddp-bench
fdl ddp-bench --help         # now picks up the new schema
```

An individual command can also refresh its own cache on the next
invocation by passing `--refresh-schema`:

```bash
fdl ddp-bench --refresh-schema
```

This is handy during development: rebuild, run with the refresh flag,
and the cache updates automatically without calling `fdl schema
refresh` explicitly.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Introspection and Tooling](04-tooling-commands.md) | Next: [Working in the Source Checkout](06-source-checkout.md)
