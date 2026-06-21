# fdl.yaml spec

`fdl.yaml` is the project manifest for flodl projects. It declares a single
`commands:` map at every level; each entry is either a path to a child
`fdl.yml`, an inline `run:` script, or (inside a sub-command) a preset that
reuses the enclosing `entry:` with merged config fields. `fdl` discovers
the root manifest automatically, like `package.json` or `composer.json`.

## Quick start

```
fdl --help                             # show commands + built-ins
fdl build                              # run a `run:` command (shell script)
fdl ddp-bench --model mlp              # pass args through to a sub-command's entry
fdl ddp-bench quick                    # run a named preset inside ddp-bench
fdl ddp-bench quick --epochs 3         # preset + extra pass-through args
fdl ddp-bench --help                   # list nested commands + options
```

## Format

YAML primary. JSON accepted too (auto-detected by file extension).

## One concept: `commands:`

Every `fdl.yml` - root and sub - declares a single `commands:` map. The
kind of each entry is inferred from its fields:

- **Path** (child directory): the entry has `path:` set, or is empty/null
  and the command name maps by convention to `./<name>/fdl.yml`.
- **Run** (inline script): the entry has `run:` set. Optional `docker:`
  routes the script through `docker compose run --rm <service>`.
- **Preset** (inside a sub-command only): neither `path:` nor `run:` is set;
  preset fields (`ddp:`, `training:`, `output:`, `options:`) are merged
  onto the enclosing config's defaults and the enclosing `entry:` is
  invoked with the merged values.

`run:` and `path:` are mutually exclusive; declaring both is a load-time
error.

## Project manifest

### Root fdl.yml

```yaml
description: flodl - Rust deep learning framework

commands:
  # Convention: `commands: { <name>: }` resolves to ./<name>/fdl.yml
  ddp-bench:

  # Explicit path override
  bench:
    description: flodl vs PyTorch benchmarks
    path: benchmarks/

  # Inline run scripts replace the old `scripts:` block
  build:
    description: Build (debug)
    run: cargo build
    docker: dev
  cuda-test:
    description: Run CUDA tests (parallel, excludes NCCL/Graph)
    run: cargo test --features cuda
    append: -- --nocapture
    docker: cuda
  self-build:
    description: Rebuild fdl CLI after changes
    run: cargo install --path flodl-cli
```

Top-level commands must be `Run` or `Path` - `Preset` is disallowed because
there is no enclosing `entry:` at the project root.

## Sub-command manifest

Each sub-command has its own `fdl.yml` with an `entry` point, optional
structured config sections, and nested `commands:` (which may be presets
of this entry, deeper child paths, or further `run:` scripts).

```yaml
# ddp-bench/fdl.yml
description: DDP validation and benchmark suite
docker: cuda
entry: cargo run --release --features cuda --

ddp:
  mode: nccl-cadence
  timeline: true

training:
  epochs: 5
  batch_size: 256
  seed: 42

output:
  dir: runs/

commands:
  quick:
    description: Fast smoke test
    training: { epochs: 1, batches_per_epoch: 100 }
    options: { model: linear }

  full-sweep:
    description: All models, all modes
    options: { model: all, mode: all }

  anchor-3:
    description: ElChe with tight sync
    ddp: { anchor: 3 }

  # A `run:` command inside a sub-command is also fine - it replaces
  # the parent entry invocation with a self-contained shell script.
  report:
    description: Regenerate the convergence report
    run: python scripts/report.py

  # A path can point at further nesting if you want a grandchild tree.
  # demo:
  #   path: examples/demo/
```

### Entry

The binary that bare arguments (and preset invocations) pass through to.
`fdl ddp-bench --model mlp` becomes
`cargo run --release --features cuda -- --model mlp`.

### Presets

A preset merges its `ddp:` / `training:` / `output:` / `options:` fields
over the enclosing sub-command's defaults, then invokes the enclosing
`entry:` with the merged values appended to any CLI pass-through args.

```
fdl ddp-bench quick                    # preset match -> merge + exec entry
fdl ddp-bench quick --model mlp        # preset + extra CLI args (CLI wins)
fdl ddp-bench --model mlp --epochs 10  # no preset, pure pass-through
```

## CLI resolution

```
fdl [@env] <name> [args...]
  0. strip env selector (@env / --env / FDL_ENV), validate overlay exists
  1. built-in?                                     -> execute (setup, init, …)
  2. commands[<name>] in root fdl.yml?             -> resolve by kind:
       Run  -> execute `run:` (optionally in docker)
       Path -> load child fdl.yml, recurse with args[1..]
  4. not found                                     -> error + help listing

Inside a recursion (child fdl.yml loaded):
  a. commands[<next-arg>] present?                 -> recurse again
       Preset -> merge fields, exec enclosing entry with tail args
       Run    -> execute `run:`
       Path   -> recurse into grandchild fdl.yml
  b. else                                          -> pass args to child entry
```

Recursion is unbounded: `fdl a b c d ...` walks the command tree as long
as each token matches a nested `commands:` entry, and once a non-match
token is reached (or there are no more tokens), the current level either
runs the matched command or passes the remaining tail through to the
entry.

## Help conventions

One way to ask for help: `--help` / `-h`. No `fdl help` subcommand -
keeps the top-level namespace clean and frees `help` from being a
reserved name.

- `fdl`                         -> same as `fdl --help`
- `fdl --help` / `fdl -h`       -> project overview: built-ins, commands, envs
- `fdl <cmd> --help`            -> command overview: arguments, nested commands, options
- `fdl <cmd> <sub> --help`      -> resolved preset or recursive command help

`-h` is reserved at the fdl level and cannot be shadowed by a sub-command
or entry option (see "Collision rules" below).

## Config sections

Structured sections that `fdl` understands natively. They appear at the
root level of a sub-command's `fdl.yml` (defaults for all nested preset
commands) and can be overridden per preset.

### ddp

Maps 1:1 to `DdpConfig` and `DdpRunConfig`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| mode | string | - | Shorthand: `solo-{n}`, `sync`, `{nccl\|cpu}-{sync\|cadence\|async}` |
| policy | string | - | Decomposed: `sync`, `cadence`, `async` (mutually exclusive with mode) |
| backend | string | - | Decomposed: `nccl`, `cpu` (mutually exclusive with mode) |
| anchor | auto/int | auto | Initial slow-GPU batches between syncs |
| max_anchor | int | 200 | Gradient staleness ceiling |
| overhead_target | float | 0.10 | AllReduce overhead fraction target |
| divergence_threshold | float | 0.05 | Param norm drift trigger for the trend guard (Cadence and Async; Sync is a no-op by construction) |
| max_batch_diff | int/null | null | Max batch lead fast over slow. null=unlimited, 0=lockstep |
| speed_hint | object | null | `{slow_rank: int, ratio: float}`, optional warm-start |
| partition_ratios | list/null | null | Explicit e.g. [0.7, 0.3], disables auto-balance |
| progressive | auto/bool | auto | Chunk dispatch mode |
| max_grad_norm | float/null | null | Per-rank L2 gradient clip before AllReduce |
| lr_scale_ratio | float | 1.0 | LR scaling ratio: factor = 1 + (world_size-1)*ratio |
| snapshot_timeout | int | 5 | Seconds, CPU backend only |
| checkpoint_every | int/null | null | Epochs between DDP checkpoints |
| timeline | bool | false | Record timeline events |

### training

Pure scalars that map directly to CLI args or RunConfig fields.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| epochs | int | - | Number of epochs |
| batch_size | int | - | Batch size per GPU |
| batches_per_epoch | int/null | null | Override (null = derive from dataset) |
| lr | float | - | Learning rate |
| seed | int | 42 | RNG seed |

### output

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| dir | string | runs/ | Base output directory. Preset name appended. |
| timeline | bool | false | Save timeline JSON/CSV/HTML |
| monitor | int/null | null | Dashboard port (null = disabled) |

### options

Pass-through key-value pairs rendered as `--key value` on the command
line. Used in preset commands for arguments that don't have a structured
section yet.

```yaml
commands:
  full-sweep:
    options: { model: all, mode: all }
```

### model (future)

Declarative alternative to entry/target. Follows conventions that define
how to build, train, and deploy a model. Not yet implemented.

```yaml
model:
  name: letter-subscan
  entry: models/subscan        # convention: look for model.yaml, build.rs, etc.
```

## Inheritance rules

1. Structured sections at the root of a sub-command's fdl.yml are defaults
   for all nested preset commands.
2. Preset-level sections deep-merge into root defaults.
3. Preset-level scalar wins over root-level scalar.
4. Preset-level map extends root-level map (matching keys overwritten).
5. CLI args override everything (appended after config-derived args).
6. `null` means "use default", not "clear the field" (within a sub-command
   config - the env-overlay layer above *does* use `null` to delete).

## Execution model

```
fdl ddp-bench quick --model mlp
  -> load ddp-bench/fdl.yml
  -> resolve commands["quick"]  (a Preset)
  -> deep-merge: root ddp + root training + preset overrides
  -> translate structured sections to CLI args
  -> append explicit CLI args (--model mlp)
  -> exec: cargo run --release --features cuda -- [merged args] --model mlp
```

## Help assembly

`fdl --help` reads the root fdl.yml and each path-kind child's description:

```
fdl - flodl project toolkit

Built-in:
  setup          Auto-detect hardware, download libtorch, build Docker
  init           Scaffold a new flodl project
  diagnose       Check GPU, libtorch, Docker health
  config         Inspect resolved project configuration

Commands:
  build          Build (debug)
  clippy         Lint (including test code, workspace + ddp-bench)
  cuda-test      Run CUDA tests (parallel, excludes NCCL/Graph)
  ddp-bench      DDP validation and benchmark suite
  shell          Interactive shell in dev container
  test           Run all CPU tests
```

`fdl ddp-bench --help`:

```
ddp-bench DDP validation and benchmark suite

Usage:
    fdl ddp-bench [<preset>] [options]

Arguments:
    [<preset>]  Named preset, one of:
      full-sweep     All models, all DDP modes
      nccl-cadence   NCCL cadence for all models
      quick          Fast smoke test (linear, 1 epoch)
      solo-0         Solo baseline on fast GPU (all models)
      solo-1         Solo baseline on slow GPU (all models)

Options:
    --model <VALUE>   Run a specific model  [default: all]
    --mode <VALUE>    Run a specific DDP mode  [default: solo-0]
    ...

Entry:
    cargo run --release --features cuda --  [docker: cuda]
    Any extra [options] are forwarded to the entry point.
```

This reflects what is actually happening: `ddp-bench` takes one
optional positional argument (a named preset) plus options. The preset
slot renders once with its possible values indented underneath - not
N distinct argument rows. Preset entries share the enclosing `entry:`;
each value just pins some option defaults and invokes the same binary.

A command with `run:` or `path:` set would render under a separate
`Commands:` section instead (it *is* a different script). When both
kinds coexist at the same level, the usage line reads
`[<preset>|<command>]` and two sections render side by side.

### `arg-name:` override

By default the preset slot is labelled `[<preset>]`. A sub-command can
override the placeholder to match its domain vocabulary without
affecting dispatch (presets are always looked up by name):

```yaml
# bake-off/fdl.yml
description: Cake bake-off
entry: cargo run --release --bin bake --
arg-name: recipe

commands:
  chocolate: { options: { flavor: chocolate, sugar: 120 } }
  vanilla:   { options: { flavor: vanilla,   sugar: 80  } }
```

Renders as:

```
Usage:
    fdl bake-off [<recipe>] [options]

Arguments:
    [<recipe>]  Named preset, one of:
      chocolate  …
      vanilla    …
```

This is purely help cosmetics - the user still types `fdl bake-off
chocolate` to invoke the preset.

## Design principles

- **One concept, `commands:`.** No separate `scripts:`, `jobs:`, or root
  command list. `run:` replaces scripts, `path:` replaces the old command
  pointer list, and preset fields replace jobs.
- **Convention over configuration.** `commands: { ddp-bench: }` implies
  `./ddp-bench/fdl.yml`; explicit `path:` overrides.
- **Recursive uniformity.** Every level works the same way. `fdl a b c`
  walks `commands` maps left-to-right until a leaf (`run:` or a terminal
  preset) is reached.
- **No premature abstraction.** Structured sections (ddp, training) exist
  because fdl understands them. Everything else is pass-through options.
- **Self-documenting.** `--help` assembles from descriptions at every
  level. New contributors see everything immediately.
- **Replaces Makefile.** Unified commands + Docker awareness = complete
  project CLI.

---

# Future Work: fdl.yaml as a Mature Config Layer

Three related design sketches that together extend fdl.yaml from "project
manifest" to "full project frontend." Each is independently useful and
independently implementable; together they make the scale-up story
(laptop -> workstation -> cluster -> cloud) a matter of editing YAML
rather than rewriting code.

**Status**: shipped. Derive, `--fdl-schema` contract, ddp-bench and
flodl-cli migration, and multi-env overlay are all in tree as of
2026-04-17. The text below is retained as the canonical reference for
how the layer is meant to behave.

## 1. Multi-environment fdl.yaml inheritance

### Goal

One project, many environments. The same `fdl.yaml` layered with
per-environment overlays: `fdl.local.yml`, `fdl.ci.yml`, `fdl.cloud.yml`,
`fdl.prod.yml`. Research code that works on a laptop scales to a cluster
or a cross-datacenter cloud deployment by editing one YAML line, not by
rewriting the training script.

### Discovery and targeting

Three equivalent forms, all explicit; a command-line selector (`@` or
`--env`) overrides the ambient `FDL_ENV`:

```bash
fdl @ci test                 # @ sigil - scan-anywhere before --, also `fdl test @ci`
fdl --env ci test            # flag - scan-anywhere: also `fdl test --env ci` / `--env=ci`
FDL_ENV=ci fdl test          # environment variable, for CI runners + shell rc
fdl local ddp-bench validate # env=local: `fdl @local ddp-bench validate`
```

Precedence: (`@env` ≡ `--env X`) > `FDL_ENV=X`. Every form must resolve to
an existing `fdl.X.yml` overlay; a miss errors loudly. `@` and `--env` are
mutually exclusive unless they name the same env (different values are a
conflict error); duplicate `--env` or duplicate `@` also errors.

### Why a sigil, not a bare positional

There is **no** bare-positional convention (`fdl ci test` does *not* select
the `ci` env). An earlier design tried `fdl <name> <cmd>` with first-arg
detection, but that forced the env namespace and the command namespace to
be disjoint - naming an env `ci`, `test`, or `setup` collided with a
command and had to error. Parsing was *semantic*: deciding whether the
first token was an env or a command required consulting the full overlay
and command sets.

The `@` sigil makes resolution *lexical* - one character of lookahead, no
tables consulted. The two namespaces stop interfering: an env may freely
share a name with a command. The first bare token is unconditionally a
command.

```
fdl [@env] <name> [...]

0. strip @env / --env / FDL_ENV; each must resolve to fdl.<env>.yml or error
1. Does a command named <name> exist?          -> dispatch it
2. Otherwise                                    -> unknown command
```

A `@`-token after a standalone `--` is left verbatim (escape hatch for an
inner command that genuinely wants a `@`-prefixed argument).

### `inherit-from:` for non-obvious topologies

Convention handles the 90% case. For cross-project inheritance or multi-
level chains, an explicit key:

```yaml
# fdl.prod.yml
inherit-from: fdl.cloud.yml   # which itself inherits from fdl.yml
ddp:
  hosts: { regions: [eu-west, us-east, ap-south] }
```

Resolution rules:

- **Linear chain**: each file names one parent; the effective layer
  order is `[deepest-ancestor, ..., direct-parent, this]`.
- **Relative paths** resolve against the directory of the declaring
  file (not the cwd). Absolute paths work as-is.
- **Env overlays compose**: `fdl.ci.yml` with `inherit-from: fdl.cloud.yml`
  produces `[fdl.yml, fdl.cloud.yml, fdl.ci.yml]`. Duplicate files
  (reached via two routes) are deduplicated at first occurrence.
- **Cycles error loudly**: `A → B → A` (or self-inheritance) surfaces
  the full cycle in the error message.
- **`fdl config show`** tags each leaf with the file that set it, so
  a three-level chain's provenance is visible at a glance.

### Merge rules

- **Maps: deep-merge.** Recurse into nested maps; overlay keys win.
- **Scalars: replace.** Overlay value takes over.
- **Lists: replace entirely.** Order is always contentious; append/prepend
  modes cause more debugging pain than they save. Users who want to
  extend a list copy the full new list - unambiguous when reading.
- **`null` deletes.** `ddp: null` in an overlay removes the entire block,
  useful for "reset to defaults in this env." Avoids needing a separate
  `!unset` syntax.

Mental model: *any key you name in the overlay wins; any key you don't
mention stays.*

### Critical UX: `fdl config show`

Layered config is opaque without introspection. `fdl config show [<env>]`
prints the resolved config with source annotations:

```yaml
$ fdl config show prod
description: flodl
ddp:
  policy: diloco                # from fdl.prod.yml
  max_anchor: 500               # from fdl.cloud.yml
  overhead_target: 0.10         # from fdl.yml (default)
  hosts:                        # from fdl.prod.yml
    regions: [eu-west, us-east, ap-south]
training:
  epochs: 10                    # from $FDL_TRAINING__EPOCHS
  batch_size: 64                # from fdl.yml
```

Without this, every mysterious value becomes a multi-file debugging hunt.
With it, "why is it using that value" is one command.

### `fdl help` lists available envs

```
$ fdl help
...
Available environments:
  @local    (fdl.local.yml)       [gitignored]
  @ci       (fdl.ci.yml)
  @cloud    (fdl.cloud.yml)
  @prod     (fdl.prod.yml)

Use:  fdl @<env> <command>   or   fdl --env <env> <command>
```

### MVP increment

~300 LOC in `flodl-cli/src/config.rs`:

1. `@env` / `--env` / `FDL_ENV` selector parsing + validate-overlay-exists.
2. Deep-merge maps, replace lists, `null` deletes - post-parse on
   existing `ProjectConfig` / `CommandConfig` structs.
3. `fdl config show [<env>]` with source annotations.
4. `fdl help` lists available envs.

Everything else (`--env` flag, `inherit-from:`, env-var single-key
overrides like `FDL_DDP__POLICY`, `!append` escape hatches) is incremental
on top.

---

## 2. Option schemas and the `--fdl-schema` contract

### The duplication trap

Without a shared source of truth, option definitions end up in three
places: the binary's argv parser, the binary's `--help` output, and the
fdl.yaml schema. That's maintenance friction and drift waiting to happen.
ddp-bench's current `main.rs` has ~65 lines of argv parsing and ~35 lines
of help text encoding the same 17 flags. Mirroring that in YAML would
triple the cost.

The fix is one source of truth, with the other layers derived.

### The contract

A binary that wants fdl-side help exposes a single flag:

```
<binary> --fdl-schema
```

which prints its options as JSON to stdout and exits:

```json
{
  "description": "DDP validation and benchmark suite for flodl",
  "args": [
    {
      "name": "run-id",
      "type": "string",
      "description": "Run identifier to analyze",
      "required": false,
      "variadic": false,
      "completer": "ls runs/"
    }
  ],
  "options": {
    "model": {
      "type": "string",
      "short": "m",
      "description": "Run specific model",
      "default": "all",
      "choices": ["logistic", "mlp", "lenet", "resnet", "resnet-graph",
                  "char-rnn", "gpt-nano", "conv-ae", "all"]
    },
    "mode": {
      "type": "string",
      "description": "Run specific DDP mode",
      "default": "solo-0",
      "choices": ["solo-0", "solo-1", "nccl-sync", "nccl-cadence",
                  "cpu-sync", "cpu-cadence", "cpu-async", "all"]
    },
    "tags": {
      "type": "list[string]",
      "description": "Filter runs by tag (repeat or comma-separate)",
      "default": []
    },
    "epochs":      { "type": "int",   "description": "Override epoch count" },
    "lr-scale":    { "type": "float", "description": "Multiply default LR" },
    "validate":    { "type": "bool",  "short": "V", "description": "Check against baseline" },
    "baseline":    { "type": "path",  "default": "baselines/baseline.json" },
    "tolerance":   { "type": "float", "default": 0.15 },
    "seed":        { "type": "int",   "default": 42 },
    "report":      { "type": "path",  "description": "Analyze runs and write report" }
  }
}
```

### Positional args

`args:` is an ordered list of positional arguments the binary (or a
preset) expects. Each entry:

| Field        | Type    | Description                                           |
|--------------|---------|-------------------------------------------------------|
| `name`       | string  | Kebab-case identifier, rendered as `<name>` in usage  |
| `type`       | string  | Same type set as options                              |
| `description`| string  | Shown in `--help`                                     |
| `required`   | bool    | Default `true`; `false` makes it optional            |
| `variadic`   | bool    | Default `false`; collects all remaining args          |
| `default`    | literal | Only valid when `required: false`                     |
| `choices`    | list    | Drives completion + validation                        |
| `completer`  | string  | Shell snippet producing completion values             |

Rules:
- Only the *last* positional may be `variadic: true`.
- A required positional cannot follow an optional one.
- Presets may pin an arg value (`args: { run-id: latest }`) - the
  positional becomes effectively fixed for that preset.

### Long and short option variants

Options are always long-form (`--name`). Adding `"short": "x"` gives a
short alias (`-x`). Short forms are single ASCII letters. Long forms are
derived from the JSON key and always kebab-case.

Short aliases are optional and always declared explicitly - no implicit
first-letter mapping (that path leads to drift when a new option shares
a prefix).

### Collision rules (checked at schema-cache build time)

Loud errors on any of:
- two options share the same long name
- two options share the same short letter
- any option (long or short) shadows a reserved fdl-level flag
  (`--help`, `-h`, `--env`, `-e`, `--list`, `--version`, `-V` when fdl
  reserves it, etc.)
- a preset pins an option or arg that isn't declared in the schema
  (only when `schema.strict: true`, otherwise silently forwarded)

Loud-at-build-time is the pattern from the env-overlay section: ambiguity
is cheapest to fix the moment it's introduced.

### Supported types

| Type             | Shell form                                   | Completion hint               |
|------------------|----------------------------------------------|-------------------------------|
| `string`         | `--key value`                                | `choices:` list if declared   |
| `int`            | `--key 42`                                   | none                          |
| `float`          | `--key 0.5`                                  | none                          |
| `bool`           | `--key` / `-k` (flag, no value)              | none                          |
| `path`           | `--key ./foo`                                | file path completion          |
| `list[T]`        | `--key a,b,c` or repeated `--key a --key b`  | T's completion per item       |

List semantics:
- Repeated flags *append*; comma-separated values *extend*. Both forms
  are equivalent and may be mixed in one invocation.
- Empty default is `[]`. A preset-level list replaces the root-level
  list (same rule as env-overlay merging - lists replace, never append).
- `list[bool]` is not allowed (meaningless).

Kept deliberately small. Richer types belong in the binary's own
validation, not in the schema.

### Defaults and precedence

Defaults can appear on both options and positionals in the schema, and on
both `#[option]` and `#[arg]` in the derive. A few rules keep the
interactions predictable:

- **Default implies optional.** A field with `default` is never
  "required." In the derive, pair `default` with plain `T` (not
  `Option<T>`) - the two are redundant and the derive rejects the
  combination at compile time.
- **`Option<T>` without `default` means "None when absent."** Use it
  when the binary needs to distinguish "user didn't set it" from "user
  set it to the default value."
- **List defaults.** `Vec<T>` defaults to `[]`. Declare a non-empty
  default explicitly (`default = &["a", "b"]`). Overlays *replace* the
  list, same rule as everywhere else.
- **Positional defaults.** Only valid on optional positionals
  (`required: false` in schema, `Option<T>` or `default = ...` in
  derive). Required positionals with a default are a contradiction and
  fail at schema-cache build time.

**Precedence chain** (highest wins):

```
argv  >  env var (via `env = "..."`)  >  preset-level YAML  >  root-level
   YAML in sub-command fdl.yaml  >  env-overlay YAML (fdl.ci.yml, ...)
      >  schema / struct default
```

The chain is collapsed at resolve time into a single value per field
before exec. `fdl config show` annotates each resolved value with its
source - which is how you debug "why is it using that value."

### fdl-side behavior

1. At project-load time, invoke `<entry> --fdl-schema` if the binary
   supports it. Cache output under `.fdl/schema-cache/<cmd>.json`.
2. Cache invalidates on binary mtime change.
3. `fdl <cmd> --help` renders from the schema (command description,
   options table, preset list).
4. `fdl <cmd> <preset> --help` renders the resolved options (schema +
   preset defaults + env overlay + env vars + argv).
5. `fdl completions` integrates: `choices:` drive completion values,
   `type: path` drives file path completion.

### Fallback chain

If `<binary> --fdl-schema` fails or isn't supported:

1. **`--help`-parsing fallback**: invoke `<binary> --help`, parse the
   text (works for any `clap`-based binary; best-effort for hand-rolled).
2. **Pass-through**: if parsing fails, show "Options (from underlying
   command):" with raw `--help` text verbatim. No validation.

Adopting the schema is opt-in and purely additive. Existing commands
keep working.

### Validation (opt-in)

```yaml
# sub-command fdl.yaml
schema:
  strict: true     # reject unknown flags at the fdl level before invoking
  options: { ... } # the declared surface
  args: [ ... ]
```

Default behavior stays pass-through. `strict: true` turns on two
complementary checks:

- **Load time.** Every preset's `options:` keys must exist in
  `schema.options`. Typos like `options: { batchsize: 32 }` when the
  schema declares `batch-size` error out the moment the `fdl.yml` is
  loaded, not later when the user runs the preset.
- **Dispatch time.** The user's extra `argv` tail is tokenized against
  the schema before the binary is invoked. Unknown flags are rejected
  with a "did you mean `--...`?" suggestion (edit-distance ≤ 2).
  Positional count and type coercion stay the binary's responsibility
  - strict is scoped to option-level typos.

Reserved universals (`--help`, `--version`, `--fdl-schema`,
`--refresh-schema`) are always allowed through, even when they are not
declared in `schema.options`.

This preserves "don't break working usage" - adding a schema to an
existing command surfaces help without enforcing anything until
`strict: true` is added explicitly.

### Rendered help - the payoff

```
$ fdl ddp-bench --help
DDP validation and benchmark suite for flodl

USAGE:
    fdl ddp-bench [<preset>] [OPTIONS]

ARGUMENTS:
    [<preset>]          Named preset, one of:
      quick             Fast smoke test (linear, 1 epoch)
      full-sweep        All models, all DDP modes
      validate          Check convergence against structured baselines
      nccl-cadence      NCCL cadence for all models

OPTIONS:
    --model <MODEL>         Run specific model  [default: all]
                            [possible: logistic, mlp, lenet, resnet,
                             resnet-graph, char-rnn, gpt-nano, conv-ae, all]
    --mode <MODE>           Run specific DDP mode  [default: solo-0]
                            [possible: solo-0, solo-1, nccl-sync,
                             nccl-cadence, cpu-sync,
                             cpu-cadence, cpu-async, all]
    --epochs <N>            Override epoch count
    --lr-scale <F>          Multiply default LR
    --validate              Check against baseline
    --baseline <PATH>       [default: baselines/baseline.json]
    --tolerance <F>         [default: 0.15]
    --seed <N>              [default: 42]
    --report <PATH>         Analyze runs and write report

Run 'fdl ddp-bench <preset> --help' for preset-specific defaults.
```

Tab-completion:

```
$ fdl ddp-bench --model <TAB>
logistic  mlp  lenet  resnet  resnet-graph  char-rnn  gpt-nano  conv-ae  all

$ fdl ddp-bench --mode <TAB>
solo-0  solo-1  nccl-sync  nccl-cadence  cpu-sync  cpu-cadence  cpu-async  all

$ fdl ddp-bench --baseline <TAB>
# (file path completion because type: path)
```

---

## 3. The `FdlArgs` derive - one struct, everything derived

> **Consolidation note (2026-04-17):** An earlier revision of this doc
> proposed a separate `flodl-args` crate, initially clap-backed. After
> shipping the clap-backed MVP and attempting ddp-bench migration, the
> clap attribute vocabulary (`default_value` vs `default_missing_value`,
> `num_args = 0..=1`) leaked through to user code and clashed with the
> "option are optional, only their value can have a default" principle
> below. We consolidated into **flodl-cli as a lib+bin** (exposes the
> runtime) + **flodl-cli-macros** as the proc-macro crate (one extra
> crate is unavoidable - Rust language rule). Clap was dropped entirely;
> flodl-cli owns the argv parser + schema emission. The `#[option]` /
> `#[arg]` split proposed below is what the derive actually uses.

### Goal

A binary declares its options **once**, as a Rust struct. The derive
macro generates:

- argv parser
- `--help` handler
- `--fdl-schema` handler
- field-level validation

No YAML mirror. No manual parsing loop. No second `print_help` function.
The struct definition *is* the source of truth.

### API shape

```rust
use flodl_args::{Args, parse_or_schema};

/// DDP validation and benchmark suite for flodl.
#[derive(Args, Debug)]
struct Cli {
    /// Optional run identifier to analyze (positional)
    #[arg(completer = "ls runs/")]
    run_id: Option<String>,

    /// Extra tags to filter by (repeat or comma-separate)
    #[option(short = 't')]
    tags: Vec<String>,

    /// Run specific model (or "all")
    #[option(short = 'm', default = "all",
             choices = &["logistic", "mlp", "lenet", "resnet", "resnet-graph",
                         "char-rnn", "gpt-nano", "conv-ae", "all"])]
    model: String,

    /// Run specific DDP mode (or "all")
    #[option(default = "solo-0", choices = &DdpMode::ALL_NAMES)]
    mode: String,

    /// Override epoch count
    epochs: Option<usize>,

    /// Override batches per epoch
    batches: Option<usize>,

    /// Override batch size
    batch_size: Option<usize>,

    /// Multiply default LR (auto-scales when epochs > default)
    lr_scale: Option<f64>,

    /// Output directory
    #[option(default = "runs")]
    output: String,

    /// Dataset cache directory
    #[option(default = "data")]
    data_dir: std::path::PathBuf,

    /// Live dashboard port
    monitor: Option<u16>,

    /// Check results against baselines
    validate: bool,

    /// Save results as baseline
    save_baseline: bool,

    /// Baseline file
    #[option(default = "baselines/baseline.json")]
    baseline: std::path::PathBuf,

    /// Validation tolerance (0.0-1.0)
    #[option(default = 0.15)]
    tolerance: f64,

    /// RNG seed
    #[option(default = 42)]
    seed: u64,

    /// Analyze runs and print report (optional FILE to save)
    report: Option<Option<String>>,

    /// Show available models and modes
    list: bool,
}

fn main() {
    // Handles --fdl-schema and --help automatically;
    // returns parsed struct or exits cleanly.
    let cli: Cli = parse_or_schema();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> flodl::tensor::Result<()> {
    // business logic using `cli.model`, `cli.mode`, etc.
    // No arg parsing. No help text. Nothing to forget.
}
```

### Conventions

- Doc comments (`///`) become descriptions.
- A plain field (no attribute) is an option with a long name derived
  from the field name. This is the common case, so it stays
  boilerplate-free.
- `Option<T>` means "not required."
- `bool` means "switch" (no value).
- `Vec<T>` is a `list[T]` option (repeat `--key` or comma-separate).
- `Option<Option<T>>` means "optional with optional value"
  (e.g. `--report` alone vs `--report foo.md`).
- Field names snake_case in Rust, rendered as kebab-case in argv
  (`batch_size` -> `--batch-size`).

#### `#[option(...)]` - a flag (`--name` / `-n`)

Maps to the JSON schema `options:` section. Attrs:

- `default = <literal>` - explicit default
- `choices = &[...]` - allowed values (drives completion + validation)
- `name = "..."` - override the CLI name (rare)
- `short = 'x'` - add a short alias (`-x`). Explicit only, no implicit
  first-letter mapping.
- `completer = "..."` - shell snippet producing completion values
  (escape hatch beyond `choices` and `type: path`)
- `env = "..."` - read from environment variable when not in argv

#### `#[arg(...)]` - a positional

Maps to the JSON schema `args:` section. Field order in the struct
defines positional order. Attrs:

- `default = <literal>` - explicit default (requires `Option<T>` or
  equivalent to also be optional)
- `choices = &[...]` - allowed values
- `name = "..."` - override the `<name>` rendered in usage
- `completer = "..."` - shell snippet producing completion values

Guardrails (compile errors):
- `short` is not valid on `#[arg]` (positionals have no short).
- Only the last positional may be variadic (`Vec<T>`).
- A required positional (`T`) cannot follow an optional one
  (`Option<T>` or one with `default`).

The derive emits the same collision checks as the JSON schema (reserved
fdl flags, duplicate longs/shorts) - build fails at compile time rather
than at runtime.

### What ddp-bench shrinks by

- 65 lines of `while i < args.len() { match ... }` - gone
- 35 lines of `print_help` - gone
- Two places that had to stay in sync - consolidated into one struct

`main.rs` becomes ~20 lines (struct definition + `fn main` + `fn run`).

### Dependency policy

`flodl-cli` already follows a zero-external-crate principle (GPU detection
via `nvidia-smi`, HTTP via `curl`, unzip via `unzip`). `flodl-args`
should fit the same constraint where feasible.

Two deployment options:

- **Zero-dep**: hand-rolled `proc-macro` + argv parsing, ~500-800 LOC.
  More work but preserves the no-external-crate story.
- **Clap-backed (opt-in feature)**: thin wrapper over `clap` + `clap_derive`
  that adds the `--fdl-schema` handler. Much smaller to implement but
  pulls in `clap`, `syn`, `quote`.

Start with the clap-backed variant to de-risk the ergonomics, then
evaluate whether the zero-dep rewrite is worth the effort once the API
is settled.

### Clap-based binaries (third-party adoption)

Binaries that already use `clap` get fdl support via a tiny adapter:

```rust
use flodl_args_clap::fdl_schema_from;

fn main() {
    let cmd = Cli::command();   // clap's Command
    if std::env::args().any(|a| a == "--fdl-schema") {
        println!("{}", fdl_schema_from(&cmd));
        return;
    }
    let cli = Cli::parse();
    // ...
}
```

This walks `clap`'s `Command` tree and emits our JSON schema. No fdl
dependency in the binary beyond this one adapter call.

---

## Rollout order

1. **Document the schema format.** This section, written down. Includes
   positional args, long/short variants, list values, collision rules.
   Done.
2. **fdl-side rendering + completions.** `flodl-cli/src/run.rs` renders
   the JSON schema as `--help` output (command description, presets,
   args, options); `completions.rs` emits `fdl completions <shell>` for bash,
   zsh, and fish, driven by the schema (`choices`, `type: path`,
   `completer`). Help unification lands here too: drop `fdl help`, bare
   `fdl` defaults to `--help`. Schema written by hand in YAML as the
   interim source. ~300 LOC.
3. **Schema-from-binary contract.** `flodl-cli` tries `<entry>
   --fdl-schema` before falling back to the YAML schema or parsed
   `--help`. Cache under `.fdl/schema-cache/`, mtime-invalidated.
   Collision check runs at cache build time. ~100 LOC.
4. **`flodl-args` crate.** New workspace member. Derive macro +
   `parse_or_schema()`. The biggest chunk (~500-800 LOC depending on
   dep policy).
5. **Migrate ddp-bench.** First real user. Proves both the crate and
   the schema format. Shrinks `main.rs` by ~100 lines.
6. **Migrate `flodl-cli` itself.** Dogfood the crate. Validates the
   story all the way down.
7. **Multi-environment inheritance.** Orthogonal to the schema work,
   can ship in parallel or after. ~300 LOC. `fdl config show` is the
   critical UX piece.

Steps 1-3 are pure win with no new crate to maintain. Steps 4-6 are
the ergonomic endgame. Step 7 is the scale-up story.

**2026-04-17 status:** Steps 1-7 are all implemented. Source-annotated
`fdl config show` is limited to layer-file headers (not per-line
provenance); richer annotation, `inherit-from:`, env-var single-key
overrides, and the explicit `--env` flag remain future-work polish.

---

## Why this matters

The scale-up story - laptop to datacenter in one YAML file - only works
if the config layer can carry the weight. Four things have to be true:

- **Options are self-documenting.** Otherwise every new scale gets a new
  "what do these flags mean" learning cost. The `struct Cli` approach
  makes that impossible to skip: add a field, add a doc comment, done.
- **Environments compose without duplication.** Otherwise `fdl.cloud.yml`
  and `fdl.prod.yml` become divergent codebases. Inheritance + `config
  show` keep them honest.
- **The frontend stays flat.** Otherwise new features leak out as new
  command-line surface, and the YAML-as-frontend promise dies. The
  schema contract means any capability added to the underlying binary
  surfaces in `fdl -h` automatically.
- **The right path is the easiest path.** Adopting `struct Cli` is
  strictly cheaper than hand-rolling argv parsing - fewer lines, no
  `print_help` to maintain, no YAML mirror to keep in sync. And the
  payoff is immediate: early validation (unknown values and bad types
  caught at fdl's boundary before Docker spin-up), "did you mean"
  suggestions from `choices:`, typed tab-completion, and a declaration
  that AI agents can consume in one read. No "silent this-value-doesn't-
  exist" failures surviving into the binary. Design principle: any
  contract that relies on "please keep these two places in sync"
  decays; this one has one place.

Together these make `fdl.yaml` a *stable* project interface - one that
research code, infra code, and AI agents can all target without
coordinating on which flag moved where.
