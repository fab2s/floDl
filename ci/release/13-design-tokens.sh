#!/bin/sh
# 13 — design-token drift gate.
#
# One palette serves the site, the live dashboard and fdl ui. The
# canonical file lives with the site; every consumer vendors it
# verbatim between `flodl design tokens -- begin/end` markers (a
# runtime include is impossible: the pages are embedded, self-contained
# artifacts, and a file outside a crate dir does not ship to
# crates.io). This check is what makes the copy discipline real —
# without it, a drifted copy looks exactly like a working one.
set -eu
cd "$(dirname "$0")/../.."

canon="site/assets/css/flodl-tokens.css"
consumers="flodl-cli/src/ui/page.html flodl/src/monitor/timeline.html"

if [ ! -f "$canon" ]; then
    echo "FAIL: canonical token file $canon is missing"
    exit 1
fi

status=0
for consumer in $consumers; do
    extracted=$(awk '/flodl design tokens -- begin/{f=1;next}/flodl design tokens -- end/{f=0}f' "$consumer")
    if [ -z "$extracted" ]; then
        echo "FAIL: $consumer carries no 'flodl design tokens -- begin/end' block"
        status=1
        continue
    fi
    if [ "$extracted" != "$(cat "$canon")" ]; then
        echo "FAIL: $consumer token block drifted from $canon — re-vendor the canonical file"
        printf '%s\n' "$extracted" | diff -u "$canon" - | head -40 || true
        status=1
    fi
done

if [ "$status" -eq 0 ]; then
    echo "OK: design tokens in sync across $(echo "$consumers" | wc -w | tr -d ' ') consumer(s)"
fi
exit $status
