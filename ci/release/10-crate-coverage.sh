#!/bin/sh
# Verify every publishable workspace crate appears in the release
# process's hand-maintained crate lists.
#
# WHY THIS EXISTS: 08-publish-dry.sh runs `cargo publish --dry-run
# --workspace`, so it picks a new crate up automatically -- and will
# therefore NEVER tell you the hand-maintained lists went stale. The
# check that "covers every crate" is exactly the one that cannot notice
# a missing entry. Two lists drift behind it:
#
#   - docs/release.md   the `cargo publish -p X` block an operator
#                       follows by hand at publish time. A crate missing
#                       here is simply never published.
#   - Makefile docs-rs  the per-crate docs.rs simulation. A crate
#                       missing here has its docs.rs rendering validated
#                       for the first time by the real publish, when
#                       crates.io is already immutable.
#
# Both failures are SILENT. flodl-hf joined the publishable set at 0.5.2
# and was absent from docs/release.md until 0.7.x; nothing caught it.
#
# Checks BOTH directions: a member missing from a list, and a list
# naming a crate that is no longer a member (stale after a rename).
#
# Deliberately does NOT check publish ORDER. Getting the order wrong
# fails loudly at the registry on the first `cargo publish` and costs a
# re-run; getting the membership wrong is the silent one, and silent is
# what a gate is for.

set -eu
cd "$(git rev-parse --show-toplevel)"

fail=0

# --- Source of truth: publishable workspace members --------------------
# The `members = [...]` array in the root manifest. Handles the array
# spanning lines; stops at the closing bracket so `exclude = [...]`
# (benchmarks, ddp-bench, hf-ddp) is never picked up.
members=$(awk '
    /^members *=/ { inlist = 1 }
    inlist { print; if (/\]/) exit }
' Cargo.toml | grep -o '"[^"]*"' | tr -d '"')

if [ -z "$members" ]; then
    echo "FAIL: could not parse [workspace] members from Cargo.toml"
    echo "  An empty crate set must never read as green."
    exit 1
fi

# A member carrying `publish = false` is intentionally unpublished, so
# it belongs in neither list.
publishable=""
for m in $members; do
    if [ -f "$m/Cargo.toml" ] && grep -qE '^publish *= *false' "$m/Cargo.toml"; then
        continue
    fi
    publishable="$publishable $m"
done

# --- What each hand-maintained list actually names ---------------------
# Guard the inputs explicitly. Without this a moved or renamed file
# yields an empty set, which reports as "every crate is missing" and
# sends the reader looking at the crate list instead of the path.
for f in docs/release.md Makefile; do
    if [ ! -f "$f" ]; then
        echo "FAIL: $f not found — this check cannot verify crate coverage"
        exit 1
    fi
done

release_md=$(grep -oE '^cargo publish -p [A-Za-z0-9_-]+' docs/release.md \
    | awk '{ print $4 }' | sort -u)

# The docs-rs recipe, from the target line to the next column-0 line.
makefile=$(awk '
    /^docs-rs:/ { indocs = 1; next }
    indocs && /^[^\t ]/ { exit }
    indocs { print }
' Makefile | grep -oE -- '-p [A-Za-z0-9_-]+' | awk '{ print $2 }' | sort -u)

# --- Compare ------------------------------------------------------------
# $1 label, $2 how to fix a missing entry, $3 the list to check.
check_list() {
    label="$1"
    fixhint="$2"
    have="$3"
    for c in $publishable; do
        if ! printf '%s\n' $have | grep -qx "$c"; then
            echo "FAIL: $label does not list workspace crate '$c'"
            echo "  $fixhint"
            fail=1
        fi
    done
    for c in $have; do
        case " $publishable " in
            *" $c "*) ;;
            *)
                echo "FAIL: $label names '$c', which is not a publishable workspace member"
                echo "  Renamed, removed, or marked publish = false? Drop the stale entry."
                fail=1
                ;;
        esac
    done
}

check_list "docs/release.md publish block" \
    "Add a 'cargo publish -p <crate>' line, leaves first (deps before dependents)." \
    "$release_md"

check_list "Makefile docs-rs target" \
    "Add a 'cargo +nightly rustdoc --lib -p <crate>' line (with --all-features if it has feature-gated modules)." \
    "$makefile"

if [ "$fail" != 0 ]; then
    echo ""
    echo "Publishable workspace members:$publishable"
    exit 1
fi

count=$(printf '%s\n' $publishable | wc -l | tr -d ' ')
echo "PASS: all $count publishable crates listed in docs/release.md and the Makefile docs-rs target"
