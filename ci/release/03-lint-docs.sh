#!/bin/sh
# Documentation drift detector. Three independent checks:
#
#   A. Stale `make <target>` references    -- any `make FOO` in tracked
#      files where FOO is not declared in the root Makefile and not on
#      the prose skip-list ("make sure", "make it", etc.).
#   B. Hardcoded user paths                -- `/home/<user>/`, `/Users/<user>/`,
#      `C:\Users\<user>\` patterns that leak developer-local checkouts
#      into committed files.
#   C. `fdl <cmd>` references resolve      -- every ``fdl <cmd>`` token in
#      docs/README must be a command `fdl` currently recognizes.
#
# CHANGELOG.md is excluded from all three -- it records historical
# state, which may legitimately reference removed targets.

set -u
cd "$(git rev-parse --show-toplevel)"

FAIL=0

# --- A. Stale `make <target>` ---
# Match only command references: backticked `make foo`, shell prompt
# `$ make foo`, error output `make: ***`, or `make <target>` in
# executable script contexts. Prose ("make sure", "make participation")
# contains `make <word>` but never in those frames, so we skip it.
MAKE_OK=$(awk -F: '/^[a-z][a-zA-Z0-9_-]+ *:/ { gsub(/[ \t]/, "", $1); print $1 }' Makefile | sort -u)

STALE_MAKE=$(git grep -nE '`make [a-z][a-zA-Z0-9_-]+`|\$ make [a-z][a-zA-Z0-9_-]+|^make [a-z][a-zA-Z0-9_-]+|Run .make [a-z][a-zA-Z0-9_-]+.|run with: make [a-z][a-zA-Z0-9_-]+' \
    -- ':!ci/release' ':!CHANGELOG.md' ':!flodl-cli/src/init.rs' \
       ':!Cargo.lock' ':!site/_site' ':!site/.jekyll-cache' \
       ':!site/_posts' ':!docs/design' \
    2>/dev/null |
    awk -v ok="$MAKE_OK" '
    BEGIN {
        split(ok, ok_arr, "\n")
        for (i in ok_arr) OK[ok_arr[i]] = 1
    }
    {
        # Extract the first `make <target>` target from the line.
        if (match($0, /make [a-z][a-zA-Z0-9_-]+/)) {
            target = substr($0, RSTART + 5, RLENGTH - 5)
            if (!(target in OK)) print
        }
    }')

if [ -n "$STALE_MAKE" ]; then
    echo "FAIL: stale \`make <target>\` references (not declared in root Makefile):"
    echo "$STALE_MAKE" | sed 's/^/  /'
    FAIL=1
fi

# --- B. Hardcoded user paths ---
# `/home/me/` is the documented placeholder convention (docs, examples)
# and `/home/ubuntu/` is the container-internal user (Dockerfile.cuda,
# docker-compose ssh rank setup) — both are deliberate, not leaks.
HARDCODED=$(git grep -nE '/home/[a-z][a-zA-Z0-9_-]+/|/Users/[a-zA-Z][a-zA-Z0-9_-]+/|C:\\\\Users\\\\[a-zA-Z][a-zA-Z0-9_-]+' \
    -- ':!ci/release' ':!CHANGELOG.md' ':!Cargo.lock' \
       ':!site/_site' ':!site/.jekyll-cache' \
    2>/dev/null | grep -vE '/home/(me|ubuntu)/' || true)

if [ -n "$HARDCODED" ]; then
    echo "FAIL: hardcoded user-specific paths:"
    echo "$HARDCODED" | sed 's/^/  /'
    FAIL=1
fi

# --- C. `fdl <cmd>` references resolve ---
if command -v fdl >/dev/null 2>&1; then
    REFS=$(git grep -hoE '`fdl [a-z][a-z0-9-]*' \
        -- 'docs/**/*.md' 'README.md' 'flodl-cli/README.md' 'ROADMAP.md' \
           ':!docs/design' \
        2>/dev/null | awk '{print $2}' | tr -d '`' | sort -u)

    BROKEN=""
    for cmd in $REFS; do
        [ -z "$cmd" ] && continue
        if ! fdl "$cmd" -h >/dev/null 2>&1; then
            BROKEN="$BROKEN $cmd"
        fi
    done

    if [ -n "$BROKEN" ]; then
        echo "FAIL: \`fdl <cmd>\` references in docs do not resolve:"
        for cmd in $BROKEN; do echo "  fdl $cmd"; done
        FAIL=1
    fi
else
    echo "WARN: fdl not on PATH; skipping fdl-cmd-ref check"
fi

[ "$FAIL" = 0 ] && echo "PASS: docs lint clean"
exit "$FAIL"
