# Release process

Cutting a floDl release is a sequence of small steps, most of which are
automated. The manual checklist fits on one screen; the automated gate
is `make release-check`.

## Pre-flight

1. Bump the workspace version in `Cargo.toml` (`version = "X.Y.Z"`) —
   **plus the four hardcoded dep-version companions** (the `=` pins make a
   miss fail loudly at resolve, but they are part of the bump, not a
   surprise): `flodl-sys = { version = ... }` in `flodl/Cargo.toml`,
   `flodl-cli-macros "=..."` in `flodl-cli/Cargo.toml`, and `flodl "=..."` +
   `flodl-cli "=..."` in `flodl-hf/Cargo.toml`.
2. Rename the `[Unreleased]` CHANGELOG heading to `[X.Y.Z] - YYYY-MM-DD`
   and add a fresh empty `[Unreleased]` above.
3. Commit both edits on `main`.

## Gate: `make release-check`

Runs every script under `ci/release/`. Each one is self-contained and
prints `PASS` / `FAIL` / `WARN`; the orchestrator prints a summary and
exits non-zero on any failure.

```
make release-check
```

Scripts, in order:

| # | Script              | Verifies                                                        |
|---|---------------------|-----------------------------------------------------------------|
| 01 | `01-git.sh`         | No uncommitted changes; target tag doesn't exist; branch sanity. |
| 02 | `02-version-sync.sh`| `Cargo.toml` version matches a dated `## [X.Y.Z] - YYYY-MM-DD` CHANGELOG header. |
| 03 | `03-lint-docs.sh`   | No stale `make <target>` refs, no hardcoded user paths, every `` `fdl <cmd>` `` in docs resolves, and every doc link, heading anchor, code fence and `/guide/` URL is valid. |
| 04 | `04-shell.sh`       | `sh -n` clean on every tracked `.sh` outside `ci/release/` (those run anyway); `shellcheck` advisory. |
| 05 | `05-ci.sh`          | Delegates to `fdl ci` (cargo build + test + clippy + strict rustdoc). |
| 06 | `06-scaffold.sh`    | `make test-init`: `fdl init` generates expected files, `docker compose config` parses. |
| 07 | `07-docs-rs.sh`     | `make docs-rs`: nightly rustdoc build simulating docs.rs. |
| 08 | `08-publish-dry.sh` | `cargo publish --dry-run` per workspace crate in dep order. |
| 09 | `09-skill-assets.sh`| `flodl-cli/assets/skills/` matches its `ai/` sources, so a released `fdl` does not ship a stale `/port` skill. |

To iterate on a single check without running the whole suite:

```
sh ci/release/03-lint-docs.sh
```

## Common failures

- **`02-version-sync` fails** - you bumped `Cargo.toml` but the
  `[Unreleased]` header in `CHANGELOG.md` still says `[Unreleased]`.
  Rename it to `[X.Y.Z] - YYYY-MM-DD`.
- **`03-lint-docs` A (make refs)** - a command was removed from the
  root Makefile but docs still reference it. Update the doc or add a
  new Makefile target.
- **`03-lint-docs` B (hardcoded paths)** - someone pasted their local
  checkout path into a script. Swap for `"$(dirname "$0")/.."` in
  shell, `(Resolve-Path "$PSScriptRoot\..").Path` in PowerShell, or
  `env::current_dir()` in Rust.
- **`03-lint-docs` C (fdl cmd)** - docs reference a subcommand that no
  longer exists. The historical case was `bench-cpu`, superseded by the
  `cpu` preset of `fdl bench`. Update or drop the mention. Extraction is
  fence-aware, so a `fdl <cmd>` inside a fenced `fdl.yml` sample - a
  user-defined project command, not a built-in - is correctly ignored.
- **`03-lint-docs` D (links/anchors)** - a link target, heading anchor,
  code fence or `/guide/` URL broke. Anchors are the common one: renaming
  a heading silently invalidates every link into it, and splitting a doc
  does it wholesale. Guide URLs are checked against the permalinks in
  `site/_stubs/`, so moving a page needs those references updated too -
  including the ones outside `docs/` (crate READMEs, blog posts, the
  `fdl init` scaffold).
- **`08-publish-dry` missing `version =`** - a `path = "../foo"` dep
  without a `version = "X.Y.Z"` companion - crates.io requires both.

## Tagging and publishing

After `make release-check` is all green:

```bash
git tag -a X.Y.Z -m "X.Y.Z -- <short description>"
git push origin main
git push origin X.Y.Z
```

The tag push fires `.github/workflows/release-cli.yml`, which builds
pre-compiled `flodl-cli` binaries for Linux / macOS / Windows and
uploads them to the GitHub release. `init.sh` and the scaffolded
`./fdl` bootstrap both grab these artifacts on first use.

Then publish to crates.io in dependency order:

```bash
cargo publish -p flodl-sys
cargo publish -p flodl-cli-macros
cargo publish -p flodl
cargo publish -p flodl-cli
```

Wait for each to index on crates.io (typically a few seconds) before
running the next - `flodl` depends on `flodl-sys`, so the latter must
be indexed first.

## After the release

- Post the release link on `@flodl_dev` (X) and `r/rust`.
- If the release changes install instructions, refresh
  `docs/cli/01-install.md` and `flodl-cli/README.md` on the same commit.
- Open a `post-0.X.Y` todo note for anything deferred during the cut.
