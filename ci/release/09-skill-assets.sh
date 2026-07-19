#!/bin/sh
# Verify the embedded skill assets match their ai/ sources.
#
# flodl-cli/assets/skills/ is the copy include_str!'d into the fdl binary
# as the out-of-repo fallback for `fdl skill install` (crates.io packages
# only the crate dir, so include_str! cannot reach ../../ai/). If it drifts
# from ai/, a released fdl ships a stale /port skill. `make sync-skills`
# (or `make site`) refreshes it.

set -eu
cd "$(git rev-parse --show-toplevel)"

fail=0
check() {
    if ! diff -q "$1" "$2" >/dev/null 2>&1; then
        echo "  DRIFT: $2 != $1"
        fail=1
    fi
}

check ai/skills/port/guide.md          flodl-cli/assets/skills/port-guide.md
check ai/skills/port/instructions.md   flodl-cli/assets/skills/port-instructions.md
check ai/adapters/claude/port-skill.md flodl-cli/assets/skills/claude-port.md

if [ "$fail" -ne 0 ]; then
    echo "FAIL: embedded skill assets drifted from ai/ sources"
    echo "  fix: make sync-skills"
    exit 1
fi

echo "PASS: embedded skill assets in sync with ai/"
