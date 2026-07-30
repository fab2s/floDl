#!/bin/sh
# Documentation drift detector. Four independent checks:
#
#   A. Stale `make <target>` references    -- any `make FOO` in tracked
#      files where FOO is not declared in the root Makefile and not on
#      the prose skip-list ("make sure", "make it", etc.).
#   B. Hardcoded user paths                -- `/home/<user>/`, `/Users/<user>/`,
#      `C:\Users\<user>\` patterns that leak developer-local checkouts
#      into committed files.
#   C. `fdl <cmd>` references resolve      -- every ``fdl <cmd>`` token in
#      docs/README must be a command `fdl` currently recognizes.
#   D. Links, anchors, fences, guide URLs  -- delegated to lint_doc_links.py,
#      which needs a real markdown parse (fence tracking + GitHub slugging)
#      that shell cannot do honestly. See that file's header for why each
#      check exists and which shipped failure it guards.
#
# CHANGELOG.md is excluded from all four -- it records historical
# state, which may legitimately reference removed targets or old file names.

set -u
# Resolve our own directory BEFORE the cd: run-all.sh invokes us as
# `./03-lint-docs.sh` from ci/release, so a $0-relative path computed after
# chdir'ing to the repo root would point at the wrong place.
CHECK_DIR="$(cd "$(dirname "$0")" && pwd)"
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
# Extraction is delegated to lint_doc_links.py --fdl-refs, which strips code
# fences. It has to: docs show illustrative `fdl.yml` manifests whose comments
# reference USER-DEFINED project commands (`fdl train`), and those are not
# flodl built-ins. A fence-blind grep reads them as broken built-ins.
#
# That grep also carried a silent coverage hole worth remembering: its pathspec
# was `docs/**/*.md`, and git's `**` still requires the literal `/`, so the 8
# top-level `docs/*.md` files were never checked at all. The python scope is
# `docs/` recursive.
if ! command -v python3 >/dev/null 2>&1; then
    echo "WARN: python3 not on PATH; skipping fdl-cmd-ref check"
elif command -v fdl >/dev/null 2>&1; then
    REFS=$(python3 "$CHECK_DIR/lint_doc_links.py" --fdl-refs)

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

# --- D. Links, anchors, code fences, guide URLs ---
if command -v python3 >/dev/null 2>&1; then
    if ! python3 "$CHECK_DIR/lint_doc_links.py"; then
        FAIL=1
    fi
else
    echo "WARN: python3 not on PATH; skipping doc link/anchor check"
fi

[ "$FAIL" = 0 ] && echo "PASS: docs lint clean"
exit "$FAIL"
