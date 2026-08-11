# Introspection and Tooling Commands

## `fdl api-ref`

Generate a structured API reference from the flodl source. Extracts all
public types, constructors, methods, builder patterns, trait
implementations, and doc examples.

```bash
fdl api-ref                # human-readable (170+ types, 1700+ lines)
fdl api-ref --json         # structured JSON for tooling
fdl api-ref --path ~/src   # explicit source path
```

Source discovery (in order):

1. Walk up from cwd for a flodl checkout.
2. Cargo registry (`~/.cargo/registry/src/`).
3. Cached GitHub download (`~/.flodl/api-ref-cache/<version>/`).
4. Download the latest release source from GitHub (cached for next time).

This means `fdl api-ref` works anywhere, even without a local checkout.
First run on a fresh machine downloads ~2MB of source; subsequent runs
use the cache.

Example output (abbreviated):

```
flodl API Reference v0.5.0
========================================

## Modules (nn)

### Linear
  Fully connected layer: `y = x @ W^T + b`.
  file: nn/linear.rs
  constructors:
    pub fn new(in_features: i64, out_features: i64) -> Result<Self>
    pub fn on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self>

### Conv2d  (implements: Module)
  file: nn/conv2d.rs
  constructors:
    pub fn configure(in_ch: i64, out_ch: i64, kernel: impl Into<KernelSize>) -> Conv2dBuilder
  builder:
    .with_stride()  .with_padding()  .with_dilation()  .done()
```

The JSON output is designed for AI-assisted porting tools. An agent can
read the full API surface, match PyTorch patterns to flodl equivalents,
and generate a working port.

## `fdl skill`

Manage AI coding assistant skills. Detects your tool, installs the right
skill files.

```bash
fdl skill list                       # show available skills and detected tools
fdl skill install                    # auto-detect tool, install all skills
fdl skill install --tool claude      # force Claude Code
fdl skill install --tool cursor      # force Cursor
fdl skill install --skill port       # install a single skill only
```

**Supported tools:**

| Tool        | Detection                       | Install target                  |
|-------------|---------------------------------|---------------------------------|
| Claude Code | `.claude/` directory            | `.claude/skills/<skill>/SKILL.md` |
| Cursor      | `.cursor/` or `.cursorrules`    | `.cursorrules` (appended)       |

**Available skills** (as of v0.5.0):

| Skill  | Description                                                                                     |
|--------|-------------------------------------------------------------------------------------------------|
| `port` | Port PyTorch scripts to flodl. Reads source, maps patterns, generates Rust project, validates with `cargo check`. |

After installing, use `/port my_model.py` in Claude Code, or ask
"Port this PyTorch code to flodl" in Cursor. Skill files are embedded
in the `fdl` binary, so this works anywhere, even without a flodl
checkout. Inside the repo, it uses the latest `ai/skills/` files from
the source tree.

## `fdl cargo`

Report and reclaim cargo's on-disk footprint. A long-lived checkout
accumulates tens of GB of compiled artifacts across `target/`,
per-container target dirs, and the docker-mounted registry caches; this
command answers "where did my disk go" and gives it back.

```bash
fdl cargo                  # both tiers, per-path sizes (always safe)
fdl cargo target           # compiled-artifact tier only
fdl cargo target --clear   # reclaim it (cost: recompute, no network)
fdl cargo cache --clear    # reclaim registry caches (cost: re-download)
fdl cargo --json           # machine-readable report
```

Two tiers, split by what getting the bytes back costs:

| Tier     | Covers                                                      | Reclaim cost              |
|----------|-------------------------------------------------------------|---------------------------|
| `target` | `target/`, `.target*`, workspace-excluded crates' `target/` | recompute, offline-safe   |
| `cache`  | `.cargo-cache*`, `.cargo-git*`                              | re-download, needs network |

`--clear` deletes a tier's *contents* and always keeps the directories
themselves: several are docker bind-mount sources, and a removed source
would be recreated root-owned by the next compose run, breaking container
builds. Entries the current user cannot delete are skipped, reported with
the bytes they still hold, and paired with a command that works, never
fatal. Those entries come from a container that ran as root, so the
durable fix is a `user:` host-identity mapping on the compose service
that wrote them rather than a habit of reaching for `sudo` (and note that
`fdl` installs per user, so `sudo fdl` does not resolve). Bare
`fdl cargo --clear` is refused: clearing names a tier, because the two
reclaim costs differ.

Clearing the `target` tier takes its binaries with it, so a symlink
pointing into that tree resolves to nothing until something rebuilds (the
clear says so). `fdl install --dev` is unaffected: it links to
`~/.cargo/bin/fdl`, which is outside the tree. A hand-made link, such as
a cluster host pointing `fdl` at a shared checkout's
`target/release/fdl`, does need the rebuild.

## `fdl completions` / `fdl autocomplete`

Generate shell completion scripts. Completions are project-aware: they
reflect the current `fdl.yml`'s `commands:` (all three kinds) plus every
sub-command's own nested entries, and are **value-aware** for flags
declared with `choices:`.

```bash
fdl completions bash > ~/.local/share/bash-completion/completions/fdl
fdl completions zsh  > "${fpath[1]}/_fdl"
fdl completions fish > ~/.config/fish/completions/fdl.fish

fdl autocomplete    # auto-detect and install into the right shell
```

Example of value-aware completion:

```bash
fdl libtorch download --cuda <TAB>   # offers: 12.6  12.8
fdl ddp-bench quick --model <TAB>    # offers values from fdl.yml `choices:`
```

Re-running `fdl completions` picks up new entries as `fdl.yml` evolves.

<!-- nav: generated by site/build_guide.py — do not edit below -->

---

Previous: [Cluster and Hardware Commands](03-cluster-commands.md) | Next: [The fdl.yml Manifest](05-manifest.md)
