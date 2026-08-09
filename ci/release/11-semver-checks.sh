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
UNCHECKABLE=""
# Count what was actually COMPARED, not just what did not fail. A crate
# with no crates.io baseline is skipped, and a summary that says PASS
# after skipping everything is a verdict about zero comparisons -- the
# same shape as a resume marker satisfied without the work.
CHECKED=0
for c in $CRATES; do
    OUT=$(DOCS_RS=1 cargo semver-checks check-release -p "$c" 2>&1)
    RC=$?
    if [ "$RC" -eq 0 ]; then
        echo "PASS: $c"
        CHECKED=$((CHECKED + 1))
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
    # A proc-macro crate exports no library API surface, so there is
    # nothing for rustdoc JSON to compare and the tool refuses outright.
    # Without this branch the catch-all below reports "requires a larger
    # version bump" for a crate that CANNOT be checked at all -- which is
    # what it did for flodl-cli-macros on this gate's first real run,
    # a false alarm that would have blocked the 0.8.0 release.
    if printf '%s' "$OUT" | grep -qE "no library target|nothing to semver-check"; then
        echo "SKIP: $c has no library target (proc-macro: no API surface to compare)"
        UNCHECKABLE="$UNCHECKABLE $c"
        continue
    fi
    # Catch-all. Anything reaching here is a non-zero exit that is NOT a
    # known non-failure, so it is reported as breakage -- but the message
    # names the assumption, because a build error or a network failure
    # would also land here.
    echo "FAIL: $c requires a larger version bump than Cargo.toml declares (or the check itself errored -- see output)"
    printf '%s\n' "$OUT" | grep -E "^--- failure|^ *Summary|^ *function |^ *struct |^ *enum |^ *method " | head -20 | sed 's/^/    /'
    CHECKED=$((CHECKED + 1))
    FAIL=1
done

[ -n "$SKIPPED" ] && echo "NOTE: unpublished, not checked:$SKIPPED"
[ -n "$UNCHECKABLE" ] && echo "NOTE: no library target, uncheckable by construction:$UNCHECKABLE"

if [ "$FAIL" = 0 ] && [ "$CHECKED" -eq 0 ]; then
    # Every crate skipped: nothing was compared, so there is nothing to
    # pass. Non-fatal like the missing-tool branch above, but it must not
    # read as coverage.
    echo "UNVERIFIED: no crate had a crates.io baseline, so NOTHING was compared"
elif [ "$FAIL" = 0 ]; then
    # Scope the claim to what was actually examined.
    echo "PASS: no public API breakage in the $CHECKED crate(s) with a baseline"
else
    echo
    echo "Under 0.x, a breaking change needs a MINOR bump (0.7 -> 0.8);"
    echo "after 1.0 it needs a major one. Either raise the version or"
    echo "restore the removed item, e.g. as a deprecated alias."
fi
exit "$FAIL"
