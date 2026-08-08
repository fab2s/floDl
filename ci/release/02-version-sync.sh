#!/bin/sh
# Verify everything that names the workspace version agrees with it.
#
#   A. CHANGELOG.md has a dated `## [X.Y.Z] - YYYY-MM-DD` entry. Catches
#      the two classic mistakes: bumped Cargo.toml but forgot to move
#      `[Unreleased]`, or added the header but forgot the date.
#   B. The internal `=X.Y.Z` pins in `[workspace.dependencies]` match the
#      version every member inherits from `[workspace.package]`.
#   C. Doc version pins match. Docs quote the exact line `fdl add` writes
#      (`flodl-hf = "=X.Y.Z"`), and flodl-hf depends on `flodl "=X.Y.Z"`,
#      so the two genuinely move together. Nothing else notices when they
#      don't: the number is prose, it compiles nowhere, and it sat three
#      releases behind before this check existed.

set -u
cd "$(git rev-parse --show-toplevel)"

VERSION=$(awk -F '"' '/^version *=/ { print $2; exit }' Cargo.toml)
FAIL=0

# --- A. CHANGELOG has a dated entry ---
if ! grep -qE "^## \[$VERSION\] - [0-9]{4}-[0-9]{2}-[0-9]{2}\b" CHANGELOG.md; then
    echo "FAIL: CHANGELOG.md has no '## [$VERSION] - YYYY-MM-DD' header"
    echo "  Cargo.toml version: $VERSION"
    echo "  CHANGELOG headers found (top 3):"
    grep -E '^## \[' CHANGELOG.md | head -3 | sed 's/^/    /'
    FAIL=1
fi

# --- B. Internal workspace pins ---
# The pins are exact on purpose: a consumer resolving `flodl` X.Y against
# `flodl-sys` X.(Y-1) would pair a safe wrapper with a shim it was not
# built for. Cargo catches a MISSED bump at resolve; it cannot catch one
# bumped to the wrong value.
BAD_PIN=$(grep -nE '^(flodl|flodl-cli|flodl-cli-macros|flodl-hw|flodl-sys) *= *\{ *version *= *"=' Cargo.toml |
    grep -vE "\"=$VERSION\"" || true)

if [ -n "$BAD_PIN" ]; then
    echo "FAIL: [workspace.dependencies] pins do not match version $VERSION"
    echo "$BAD_PIN" | sed 's/^/  /'
    FAIL=1
fi

# --- C. Doc version pins ---
# Scoped to tracked markdown. CHANGELOG.md is excluded: it records
# historical state, so an old release's entry legitimately quotes an old
# pin.
STALE_DOC=$(git grep -nE '(flodl|flodl-cli|flodl-cli-macros|flodl-hw|flodl-hf|flodl-sys) ?= ?"=[0-9]+\.[0-9]+\.[0-9]+"' \
    -- '*.md' ':!CHANGELOG.md' ':!site/_site' ':!site/.jekyll-cache' ':!site/_posts' 2>/dev/null |
    grep -vE "\"=$VERSION\"" || true)

if [ -n "$STALE_DOC" ]; then
    echo "FAIL: doc version pins do not match version $VERSION"
    echo "$STALE_DOC" | sed 's/^/  /'
    FAIL=1
fi

# --- D. Stated MSRV in docs ---
# The `msrv` CI job proves `rust-version` is the floor that actually
# compiles. Nothing proved the docs said the same number, and they
# didn't: 0.7.0 shipped `rust-version = "1.85"`, the floor was corrected
# twice on the way to 1.91, and README kept telling users 1.85 the whole
# time. Any tracked markdown stating a minimum has to track the manifest.
MSRV=$(awk -F '"' '/^rust-version *=/ { print $2; exit }' Cargo.toml)

STALE_MSRV=$(git grep -nE 'Rust\]\(https://rustup\.rs/\) [0-9]+\.[0-9]+\+|Rust [0-9]+\.[0-9]+\+' \
    -- '*.md' ':!CHANGELOG.md' ':!site/_site' ':!site/.jekyll-cache' ':!site/_posts' 2>/dev/null |
    grep -vE "Rust[^0-9]*$MSRV\+" || true)

if [ -n "$STALE_MSRV" ]; then
    echo "FAIL: docs state a minimum Rust version other than $MSRV"
    echo "$STALE_MSRV" | sed 's/^/  /'
    FAIL=1
fi

[ "$FAIL" = 0 ] && echo "PASS: CHANGELOG, workspace pins, doc pins say $VERSION; docs state MSRV $MSRV"
exit "$FAIL"
