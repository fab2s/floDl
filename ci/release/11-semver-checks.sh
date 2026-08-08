#!/bin/sh
# Public-API breakage check against what is actually published.
#
# This automates a step that was being done by eye: scanning for breaking
# changes before locking a version. It catches a class the test suite
# structurally cannot -- 3000 tests all compile against the CURRENT source,
# so none of them can notice that a name a downstream crate imports has
# gone. cargo-semver-checks compares this tree's rustdoc against the
# rustdoc of the last release on crates.io.
#
# RUN ORDER MATTERS: this must come AFTER 02-version-sync.sh. The check is
# "is the DECLARED version sufficient for the changes made", so it reads
# the version out of Cargo.toml. Run before the bump and it correctly
# fails, because the tree has changes the current version does not cover.
#
# No libtorch: DOCS_RS=1 is flodl-sys's build script skipping its C++
# compile, and rustdoc never links. Nightly is required for rustdoc JSON.
#
# Per crate rather than `--workspace`, deliberately: the workspace run
# stops at the first failing crate, so one break hides every other. Same
# reasoning as `fail-fast: false` on the OS matrix.

set -u
cd "$(git rev-parse --show-toplevel)"

# The publishable set, in the order docs/release.md publishes them.
# 10-crate-coverage.sh is what keeps this list from going stale.
CRATES="flodl-hw flodl-sys flodl-cli-macros flodl flodl-cli flodl-hf"

if ! command -v cargo-semver-checks >/dev/null 2>&1; then
    echo "SKIP: cargo-semver-checks not installed"
    echo "  cargo install cargo-semver-checks --locked"
    # Loud, and non-fatal: a developer box without the tool should not
    # fail the suite, but it must not look like the check passed either.
    echo "UNVERIFIED: public API breakage was NOT checked"
    exit 0
fi

FAIL=0
SKIPPED=""
for c in $CRATES; do
    OUT=$(DOCS_RS=1 cargo semver-checks check-release -p "$c" 2>&1)
    RC=$?
    if [ "$RC" -eq 0 ]; then
        echo "PASS: $c"
        continue
    fi
    # A crate that has never been published has no baseline to compare
    # against, which is not a failure: its first release cannot break
    # anyone. flodl-hw is in this state until it ships.
    if printf '%s' "$OUT" | grep -q "not found in registry"; then
        echo "SKIP: $c is not on crates.io yet (no baseline; first release breaks nobody)"
        SKIPPED="$SKIPPED $c"
        continue
    fi
    echo "FAIL: $c requires a larger version bump than Cargo.toml declares"
    printf '%s\n' "$OUT" | grep -E "^--- failure|^ *Summary|^ *function |^ *struct |^ *enum |^ *method " | head -20 | sed 's/^/    /'
    FAIL=1
done

[ -n "$SKIPPED" ] && echo "NOTE: unpublished, not checked:$SKIPPED"

if [ "$FAIL" = 0 ]; then
    echo "PASS: no public API breakage beyond the declared version"
else
    echo
    echo "Under 0.x, a breaking change needs a MINOR bump (0.7 -> 0.8);"
    echo "after 1.0 it needs a major one. Either raise the version or"
    echo "restore the removed item, e.g. as a deprecated alias."
fi
exit "$FAIL"
