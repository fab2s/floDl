# Release process

Cutting a floDl release is a sequence of small steps, most of which are
automated. The manual checklist fits on one screen; the automated gate
is `make release-check`.

## Pre-flight

1. Bump the workspace version in `Cargo.toml`, in **two** places, both in
   that one file:
   - `[workspace.package] version = "X.Y.Z"`, which every member inherits.
   - the five internal `=X.Y.Z` pins in `[workspace.dependencies]`
     (`flodl`, `flodl-cli`, `flodl-cli-macros`, `flodl-hw`, `flodl-sys`).

   The pins are exact on purpose: a consumer resolving `flodl` X.Y against
   `flodl-sys` X.(Y-1) would pair a safe wrapper with a shim it was not
   built for. Missing one fails loudly at resolve rather than silently, but
   it is part of the bump, not a surprise. These used to be six hardcoded
   companions spread across `flodl/`, `flodl-cli/` and `flodl-hf/`
   manifests; they were hoisted into the workspace table, so the member
   manifests now say `dep.workspace = true` and hold no version at all.
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
| 02 | `02-version-sync.sh`| Everything that names the version agrees with it: a dated `## [X.Y.Z] - YYYY-MM-DD` CHANGELOG header, the five internal `=X.Y.Z` pins in `[workspace.dependencies]`, the version pins quoted in tracked docs, and the minimum Rust version those docs state. |
| 03 | `03-lint-docs.sh`   | No stale `make <target>` refs, no hardcoded user paths, every `` `fdl <cmd>` `` in docs resolves, and every doc link, heading anchor, code fence and `/guide/` URL is valid. |
| 04 | `04-shell.sh`       | `sh -n` clean on every tracked `.sh` outside `ci/release/` (those run anyway); `shellcheck` advisory. |
| 05 | `05-ci.sh`          | Delegates to `fdl ci` (cargo build + test + clippy + strict rustdoc). |
| 06 | `06-scaffold.sh`    | `make test-init`: `fdl init` generates expected files, `docker compose config` parses. |
| 07 | `07-docs-rs.sh`     | `make docs-rs`: nightly rustdoc build simulating docs.rs. |
| 08 | `08-publish-dry.sh` | `cargo publish --dry-run` per workspace crate in dep order. |
| 09 | `09-skill-assets.sh`| `flodl-cli/assets/skills/` matches its `ai/` sources, so a released `fdl` does not ship a stale `/port` skill. |
| 10 | `10-crate-coverage.sh`| Every publishable workspace member is listed in this doc's publish block AND the `Makefile` `docs-rs` target (and neither names a crate that is no longer a member). |
| 11 | `11-semver-checks.sh`| Public-API breakage, per crate, against the last release on crates.io. Reads the version from `Cargo.toml`, so it must run *after* `02`: before the bump it correctly fails. |
| 12 | `12-guide-snapshot.sh`| The release series' guide is committed under `site/guide/X.Y.x/` with its sidebar, so flodl.dev serves documentation for the version being cut. |

To iterate on a single check without running the whole suite:

```
sh ci/release/03-lint-docs.sh
```

## Common failures

- **`02-version-sync` fails** - three separate causes, and the output says
  which. The `[Unreleased]` CHANGELOG header still says `[Unreleased]`
  (rename it to `[X.Y.Z] - YYYY-MM-DD`); the `[workspace.dependencies]`
  pins still name the previous version; or a doc still quotes it. The
  last two are the bump's own checklist - the failure lists every file
  and line, so run it first and work the list.
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
- **`12-guide-snapshot` missing tree** - the release's guide has not been
  snapshotted. Run it once, before the release commit, naming the SERIES
  rather than the version:

  ```sh
  python3 site/build_guide.py --channel 0.8.x
  git add site/guide/0.8.x site/_includes/sidebar-0.8.x.html site/guide/index.html
  ```

  The site publishes the guide under a permanent version segment
  (`/guide/0.8.x/...`) so a link written against a release keeps resolving
  after `docs/` moves on; bare `/guide/...` paths redirect to whichever
  series is newest, which is what makes cutting a release move the
  documentation. This gate exists because a forgotten snapshot is silent:
  the site still builds and every link still works, the docs are simply a
  version behind. The committed tree is then patchable in place, so fixing
  a typo in shipped docs is an ordinary edit rather than a wait for the
  next release - which is why it must not be regenerated afterwards. Each
  tree carries a README saying so.
- **`10-crate-coverage` fails after adding a crate** - add it to the
  publish block above and to the `Makefile` `docs-rs` target. The failure
  names the crate and the list it is missing from. If it fires the other
  way (a list names a non-member), the crate was renamed, removed, or
  marked `publish = false`; drop the stale entry.

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
cargo publish -p flodl-hw
cargo publish -p flodl-sys
cargo publish -p flodl-cli-macros
cargo publish -p flodl
cargo publish -p flodl-cli
cargo publish -p flodl-hf
```

Wait for each to index on crates.io (typically a few seconds) before
running the next - `flodl` depends on `flodl-sys` and `flodl-hw`, so
those must be indexed first, and `flodl-hf` depends on both `flodl` and
`flodl-cli`.

The order is leaves-first. Membership is gated: `10-crate-coverage.sh`
checks this block and the `Makefile` `docs-rs` target against the
workspace `members` list in the root `Cargo.toml`, in both directions, so
a new crate cannot reach a release absent from either. The gate exists
because `08-publish-dry.sh` uses `--workspace` and therefore picks a new
crate up *automatically* -- the check that covers every crate is exactly
the one that can never notice a hand-maintained list going stale.

The **order** within this block is not gated, deliberately: getting it
wrong fails loudly at the registry on the first `cargo publish` and costs
a re-run, whereas a missing crate is silent.

## After the release

- Post the release link on `@flodl_dev` (X) and `r/rust`.
- If the release changes install instructions, refresh
  `docs/cli/01-install.md` and `flodl-cli/README.md` on the same commit.
- Open a `post-0.X.Y` todo note for anything deferred during the cut.
