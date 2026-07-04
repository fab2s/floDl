# Upgrading floDl

Two upgrades are documented here, newest first: the unreleased
process-model distributed rewrite, then the 0.5.0 CLI maturity pass.

---

## Upgrading past 0.5.x (unreleased): the process-model distributed rewrite

This is the largest pre-1.0 change to the `flodl` crate's public
surface. Single-device and single-host code is unaffected; the breaks
are all in the multi-GPU / cluster distributed layer. Removal targets
are not version-pinned (the release is unversioned as of writing); this
guide describes the shape of each change.

### The in-process multi-GPU engine is gone; multi-GPU is process-per-rank

`Trainer::builder(...).run()` and `Trainer::run(...)` now transparently
auto-promote to process-per-rank fan-out when 2+ GPUs are visible (or a
cluster overlay is active). The training entry point is unchanged — the
same code runs single-device, single-host multi-GPU, and multi-host
cluster. You do not launch anything differently.

`Ddp::wrap` remains as the explicit per-rank bypass primitive, but its
signature changed from the old multi-model/multi-device thread form to
one rank per call:

```rust
// before: one call wrapped every local replica
// Ddp::wrap(&[&model0, &model1], &[dev0, dev1])?

// now: one call per rank (one model, one device, this rank's global id,
// the shared rendezvous)
let ddp = Ddp::wrap(&model, device, global_rank, &rendezvous)?;
```

### Removed: the 0.3.0-deprecated compatibility surface

These aliased `Trainer::builder()` for two minor releases and are now
gone:

| Removed | Replacement |
|---|---|
| `AsyncDdp` / `AsyncDdpBuilder` / `AsyncDdpConfig` | `DdpHandle` / `DdpBuilder` / `DdpRunConfig` (create via `Trainer::builder()`) |
| `DdpHandle::auto()` / `auto_with()` / `builder()` | `Trainer::builder(model_factory, optim_factory, train_fn)` |

### Deprecated: the self-driven `Trainer::setup` tier

`Trainer::setup` / `setup_with` / `setup_head` / `setup_head_with` and
their `DdpConfig` config bag are deprecated (they still compile). They
are the only path that schedules without the controller, so they miss
the convergence guard, LR-aware meta-controller, outer optimizer,
elastic membership, and checkpoint orchestration. Move to
`Trainer::builder(...)` (own the loop-body via the step closure) or
`Trainer::run(...)` (framework owns the loop). The user-owned-loop
ergonomics return later as a cooperative tier on the controller engine
(see `docs/design/trainer-execution-tiers.md`).

### Removed: `DataLoader::distributed`

Data sharding is now owned by the cluster coordinator (it dispatches
each rank's per-epoch partition from the shared dataset). Build a plain
`DataLoader` / pass the dataset to `Trainer::builder(...).dataset(...)`;
the framework shards it.

### cluster.yml: structured `controller:` / `workers:` schema

The flat top-level keys are replaced by nested blocks:

```yaml
# before
master_addr: 192.168.1.10
master_port: 29500
workers:
  - host: gpu-1
    ssh_user: ubuntu
    ssh_port: 22

# now
cluster:
  controller:
    host: 192.168.1.10          # was master_addr
    port: 1337                   # was master_port
    path: /home/me/project       # controller's view of the shared root
  workers:
    - host: gpu-1
      ssh:                       # ssh_* knobs move into an ssh: sub-block
        user: ubuntu
        port: 22
```

`fdl probe` flags the legacy flat keys with migration hints.

---

## Upgrading to floDl 0.5.0 (the fdl CLI maturity pass)

floDl 0.5.0 is the **fdl CLI maturity pass**. The framework API stays
compatible with 0.4.0; the only breaking change lives in the `fdl.yml`
manifest and the `#[derive(FdlArgs)]` attribute contract.

## TL;DR

1. Rename `scripts:` → `commands:` in your `fdl.yml`, wrapping each
   value in a `run:` field.
2. If any `#[derive(FdlArgs)]` struct has a field named `help`,
   `version`, `quiet`, or `env` (or short-flagged `h`, `V`, `q`, `v`,
   `e`), rename it.
3. Optional: rename `fdl.dev.yml` / `fdl.ci.yml` style files you had
   been selecting manually - `fdl --env <name>` now loads them
   automatically.

That's it. Everything else is additive.

---

## 1. `scripts:` → `commands:` in `fdl.yml`

In 0.4.0, `fdl.yml` had two top-level maps:

- `scripts:` - shell-string commands, no docker wrapping.
- `commands:` - docker-wrapped entries with structured config.

In 0.5.0 these are merged into one **`commands:` map** with three
kinds, chosen by which fields the entry sets: `run:` (shell), `path:`
(nested project), or preset (`ddp:` / `training:` / `output:` /
`options:` merging over an enclosing `entry:`).

### Minimal migration

```yaml
# 0.4.0 ---------------------------------------------------
scripts:
  fmt: cargo fmt --all
  lint: cargo clippy -- -D warnings

commands:
  test:
    docker: dev
    run: cargo test --features cuda

# 0.5.0 ---------------------------------------------------
commands:
  fmt:
    run: cargo fmt --all
  lint:
    run: cargo clippy -- -D warnings
  test:
    docker: dev
    run: cargo test --features cuda
```

### Rules of the three kinds

| Kind    | Set                              | Argv forwarded? | Notes                                                           |
|---------|----------------------------------|-----------------|-----------------------------------------------------------------|
| `run:`  | `run:` (optionally `docker:`)    | **no**          | Closed script; use shell `$VAR` inside. `docker:` is allowed here only. |
| `path:` | `path:` (or empty + sibling dir) | yes             | Nested project with its own `fdl.yml`; forwarded argv validated against its `entry:` schema. |
| preset  | neither `run:` nor `path:`       | yes             | Only legal inside a `path:`-kind sub-command's own `fdl.yml`. Deep-merges `ddp:` / `training:` / `output:` / `options:` over the enclosing defaults. |

Common gotchas:

- **`docker:` on a non-`run:` entry** now errors at load time. Move the
  `docker:` field onto the `run:` entry it belongs to, or onto the
  sub-command's own `fdl.yml` at the top level.
- **Extra argv after `fdl <cmd> ...`** is **not** forwarded to a `run:`
  entry. If you relied on `fdl my-script foo bar` passing `foo bar` to
  the shell, switch to either `$FDL_EXTRA_ARGS` inside the script, or
  migrate the entry to a `path:` kind with a typed `entry:` binary.
- **Auto-bootstrap**: if only `fdl.yml.example` is checked in, `fdl`
  now offers to copy it to a real (gitignored) `fdl.yml` on first run.

Load-time errors tell you exactly which file, which key, and which
rule failed.

---

## 2. Reserved CLI flags in `#[derive(FdlArgs)]`

In 0.4.0, a struct field named `help` silently overrode `--help`. In
0.5.0 the following longs and shorts are **reserved** and cannot be
shadowed; collisions error at derive time:

- Longs: `--help`, `--version`, `--quiet`, `--env`
- Shorts: `-h`, `-V`, `-q`, `-v`, `-e`

If you have a struct like this:

```rust
// 0.4.0
#[derive(FdlArgs)]
struct Args {
    #[option]
    help: Option<String>,   // will fail to compile in 0.5.0
}
```

rename the field:

```rust
// 0.5.0
#[derive(FdlArgs)]
struct Args {
    #[option(short = 'H')]   // a non-reserved short, if you need one
    help_text: Option<String>,
}
```

The short-flag derivation is automatic from the long name's first
letter; if that first letter is reserved, pass `short = '...'`
explicitly or let the derive skip the short.

---

## 3. Environment overlays (optional, new)

If you already maintained per-environment `fdl.yml` files manually
(e.g. `fdl.local.yml`, `fdl.ci.yml`), 0.5.0 now loads them on top of
the base via:

```bash
fdl @ci test              # @ sigil (scan-anywhere before --)
fdl --env ci test         # explicit flag
FDL_ENV=ci fdl test       # env var
```

Nothing breaks if you don't use this - overlays are purely additive.
`fdl config show [env]` prints the resolved merged config with
per-layer origin annotations, which is the fastest way to verify a
new overlay before running a long job.

---

## 4. New top-level commands (informational)

None of these replace existing commands; they are new conveniences
that existed as no-ops or were simply absent in 0.4.0:

- `fdl config show [env]` - resolved YAML with origin annotations.
- `fdl schema list | clear [<cmd>] | refresh [<cmd>]` - manage the
  per-command schema cache.
- `fdl autocomplete` - one-shot installer for shell completions.
- `--refresh-schema` per-invocation flag to refresh one entry's cache
  without a manual `fdl schema refresh`.

---

## 5. `flodl-cli-macros` on crates.io

0.5.0 adds one new published crate:

- [`flodl-cli-macros`](https://crates.io/crates/flodl-cli-macros) -
  the proc-macro derive for `FdlArgs`, re-exported by
  [`flodl-cli`](https://crates.io/crates/flodl-cli) as
  `flodl_cli::FdlArgs`. Downstream binaries depend on `flodl-cli`,
  not on this crate directly.

`flodl-cli` itself was already published on crates.io in earlier
versions; 0.5.0 bumps it along with the rest of the workspace.

You can install the CLI with `cargo install flodl-cli` or via the
pre-compiled bootstrap: `curl -sL https://flodl.dev/fdl -o fdl`.

---

## 6. Framework changes

No breaking changes to the `flodl` crate in 0.5.0. The CHANGELOG has
no `### Removed` or `### Changed (breaking)` entries outside of the
CLI / manifest scope above.

If you're upgrading from 0.3.0 or earlier, read through CHANGELOG.md
from your version forward - the 0.4.0 entry is the larger one on the
framework side.

---

## Reporting issues

Please file [GitHub issues](https://github.com/flodl-labs/flodl/issues)
with a minimal reproducing `fdl.yml` and the exact error message if
anything in this guide leaves you stuck.
