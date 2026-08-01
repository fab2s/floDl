# The floDl CLI: Install and Global Flags

`fdl` is floDl's command-line tool. It handles hardware detection, libtorch
management, project scaffolding, guided setup, and doubles as a project
task runner driven by a declarative `fdl.yml` manifest. It is a pure Rust
binary with **zero native dependencies** (no libtorch needed to run),
compiles in under a second, and works on any machine with Rust or Docker.

`fdl` is useful in three contexts, and this reference is structured around
them:

1. **Standalone** - just the binary, no project around. Hardware probing,
   libtorch install, scaffolding, skill bundles, `fdl install`. Split across
   [project and libtorch commands](02-setup-commands.md),
   [cluster and hardware commands](03-cluster-commands.md), and
   [introspection and tooling](04-tooling-commands.md).
2. **[Inside a floDl project](05-manifest.md)** -
   any directory (or ancestor) that contains an `fdl.yml`. Manifest-driven
   task dispatch, environment overlays, schema introspection, preset
   sub-commands, value-aware completions.
3. **[In the flodl source checkout](06-source-checkout.md)** -
   the cloned repo's `fdl.yml` ships the concrete command set used to
   develop flodl itself (`fdl test`, `fdl gpu-test`, `fdl ddp-bench …`,
   `fdl self-build`, etc.).

Standalone, libtorch is managed under `~/.flodl/` (override with
`$FLODL_HOME`). In a project, it is managed under `./libtorch/` in the
project root.

---

## Install

The fastest path is the pre-compiled binary (no Rust toolchain
required):

```bash
curl -sL https://flodl.dev/fdl -o fdl && chmod +x fdl
./fdl install                # copies to ~/.local/bin/fdl
```

The `fdl` bootstrap script downloads the right pre-compiled binary from
GitHub Releases on first use, then `./fdl install` puts it on your PATH.
It detects your shell and prints PATH instructions if `~/.local/bin`
is not yet on your PATH. The bootstrap falls back to `cargo build`
if no binary is available for your platform.

If you have a Rust toolchain handy, the equivalent one-liner is:

```bash
cargo install flodl-cli
```

For developers working on flodl itself:

```bash
cargo build --release -p flodl-cli
./target/release/fdl --help
```

## Install flags and updates

```bash
fdl install                  # copy to ~/.local/bin/fdl
fdl install --dev            # symlink instead (developers: tracks local builds)
fdl install --check          # compare installed vs latest GitHub release
```

Use `--dev` to symlink instead of copy, so
`cargo build --release -p flodl-cli` instantly updates the global
`fdl`. `fdl install --check` compares the installed version with the
latest GitHub release and is the primary way to update an existing
install.

---

## Global flags

Every `fdl` invocation accepts the following flags before the command
name (or in some positions, after):

| Flag             | Effect                                                         |
|------------------|----------------------------------------------------------------|
| `-h`, `--help`   | Show help for the current command scope.                       |
| `-V`, `--version`| Print the CLI version.                                         |
| `--env <name>`   | Apply `fdl.<name>.yml` overlay on top of `fdl.yml`.            |
| `--gpus <spec>`  | Scope GPU visibility for the dispatched command. `all` (every visible device) or comma-separated indices (`0,1`). On cluster-aware commands, synthesizes a single-host cluster envelope; on non-cluster commands, sets `CUDA_VISIBLE_DEVICES` for the child. See [`fdl --gpus`](03-cluster-commands.md#fdl---gpus). |
| `--no-prebuild`  | Skip the cluster fan-out pre-flight build (`fdl @cluster <cmd>` and any `cluster: true` command). Use when binaries are known fresh, or when iterating on a build-only issue. |
| `--no-append`    | Drop a `run:` command's `append:` suffix.                      |
| `-v`             | Verbose output.                                                |
| `-vv`            | Debug output.                                                  |
| `-vvv`           | Trace output (maximum detail).                                 |
| `-q`, `--quiet`  | Suppress non-error output.                                     |
| `--ansi`         | Force ANSI color (bypass TTY / `NO_COLOR` detection).          |
| `--no-ansi`      | Disable ANSI color output.                                     |

Verbosity flags propagate into the framework's logging system
(`flodl::log`) and into Docker child commands via `FLODL_VERBOSITY`.
Equivalent without the CLI: `FLODL_VERBOSITY=verbose cargo run`. The
variable accepts integers `0`-`4` or names
`quiet`/`normal`/`verbose`/`debug`/`trace`. Level `normal` (1) is the
default when no verbosity flag is passed.

```bash
fdl -v ddp-bench quick    # verbose: DDP sync, cadence changes, prefetch detail
fdl -vv cuda-test         # debug: per-batch timing, internal loops
fdl -vvv shell            # trace: extreme granularity
fdl --quiet test          # errors only
fdl --no-ansi config show # plain output for pipes and CI
```

Some flag names are **reserved** by the CLI and cannot be shadowed by
derived argument structs (see [Declaring flags in Rust](05-manifest.md#declaring-flags-in-rust)):
`--help`, `--version`, `--quiet`, `--env`, and the shorts `-h`, `-V`,
`-q`, `-v`, `-e`.

---

## Update checks

`fdl` probes crates.io once per day for newer versions of itself
(`flodl-cli`) and, when run inside a Cargo project, the user-facing
flodl crates the project depends on (`flodl`, `flodl-hf`). Outdated
crates are surfaced as one-line nudges at the end of the user's
command, after their normal output, so they never block work.

The first run prints a one-time disclosure pointing at the opt-outs;
subsequent runs are silent unless an update is found.

| Behavior              | Trigger                                                                   |
|-----------------------|---------------------------------------------------------------------------|
| Disabled this run     | `FDL_NO_UPDATE_CHECK=1` env var (wins over everything else)               |
| Disabled in CI        | `CI=true` env var (auto-detected from any standard CI runner)             |
| Disabled in container | `/.dockerenv` present (avoids ephemeral cache + redundant probes)         |
| Disabled persistently | Set `update_check.enabled = false` in `<config-dir>/flodl/config.json`    |

`<config-dir>` follows platform conventions:
- Linux / BSD: `$XDG_CONFIG_HOME` if set, else `~/.config`
- macOS: `~/Library/Application Support`
- Windows: `%APPDATA%`

Config-file shape (auto-managed except `enabled`):

```json
{
  "update_check": {
    "enabled": true,
    "last_check": 1714138800,
    "latest_known": {
      "flodl-cli": "0.5.3",
      "flodl": "0.5.3",
      "flodl-hf": "0.5.3"
    },
    "first_run_seen": true
  }
}
```

The probe uses `curl --max-time 2` and silently skips on every failure
mode - the user's command is never delayed past the timeout, and a
broken or offline network just means today's check didn't update the
cache. Pre-release versions are ignored; nudges only fire against the
crate's `max_stable_version`.

Updating from a nudge:

```bash
fdl install --check               # update fdl itself
cargo update                      # update flodl/flodl-hf in your project
```

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Troubleshooting](../ddp/04-troubleshooting.md) | Next: [Project and libtorch Commands](02-setup-commands.md)
