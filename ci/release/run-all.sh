#!/bin/sh
# Release-readiness orchestrator.
#
# Runs every ci/release/NN-*.sh in numeric order, captures each
# script's output to a log, and ends with an actionable checklist:
# every FAIL:/WARN: line the checks emitted, grouped by script, so
# the remaining work is readable without scrolling back through the
# full output.
#
# Usage:
#   sh ci/release/run-all.sh          # strict: run after the release commit
#   sh ci/release/run-all.sh --prep   # release prep: uncommitted changes are
#                                     # tolerated (WARN) and the publish
#                                     # dry-run gets --allow-dirty, so the
#                                     # whole suite can go green BEFORE the
#                                     # release commit exists. A strict run
#                                     # is still required before tagging.
#
# Individual scripts can also be invoked directly (they each chdir
# to the repo root), so `sh ci/release/03-lint-docs.sh` is fine for
# iterating on a single check. Export RELEASE_PREP=1 to get prep-mode
# behavior on a direct invocation.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RELEASE_PREP=${RELEASE_PREP:-}
for arg in "$@"; do
    case "$arg" in
        --prep) RELEASE_PREP=1 ;;
        *)
            echo "unknown argument: $arg (supported: --prep)"
            exit 2
            ;;
    esac
done
export RELEASE_PREP

LOG_DIR="${TMPDIR:-/tmp}/fdl-release-$$"
mkdir -p "$LOG_DIR"

if [ -n "$RELEASE_PREP" ]; then
    echo "mode: release prep (dirty tree tolerated; run again WITHOUT --prep after the release commit)"
fi

PASS_LIST=""
FAIL_LIST=""
for script in $(ls -1 [0-9][0-9]-*.sh 2>/dev/null | sort); do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $script"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log="$LOG_DIR/${script%.sh}.log"
    # POSIX sh has no PIPESTATUS; smuggle the script's exit code out
    # through a file so `tee` doesn't mask it.
    { sh "./$script" 2>&1; echo "$?" > "$LOG_DIR/rc"; } | tee "$log"
    rc=$(cat "$LOG_DIR/rc")
    if [ "$rc" = 0 ]; then
        PASS_LIST="$PASS_LIST $script"
    else
        FAIL_LIST="$FAIL_LIST $script"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for s in $PASS_LIST; do
    if grep -q '^WARN' "$LOG_DIR/${s%.sh}.log" 2>/dev/null; then
        printf '  \033[32mPASS\033[0m  %s  \033[33m(with warnings)\033[0m\n' "$s"
    else
        printf '  \033[32mPASS\033[0m  %s\n' "$s"
    fi
done
for s in $FAIL_LIST; do printf '  \033[31mFAIL\033[0m  %s\n' "$s"; done

# --- Actionable checklist: what remains, without scrolling back ---
if [ -n "$FAIL_LIST" ]; then
    echo ""
    echo "Remaining before tag:"
    for s in $FAIL_LIST; do
        log="$LOG_DIR/${s%.sh}.log"
        echo "  [ ] $s"
        if grep -q '^FAIL' "$log" 2>/dev/null; then
            grep '^FAIL' "$log" | sed 's/^/        /'
        else
            # set -e death without a FAIL: line -- show the tail.
            echo "        (no FAIL: marker; last output lines:)"
            tail -5 "$log" | sed 's/^/        | /'
        fi
    done
fi

WARN_SCRIPTS=$(grep -l '^WARN' "$LOG_DIR"/*.log 2>/dev/null || true)
if [ -n "$WARN_SCRIPTS" ]; then
    echo ""
    echo "Warnings (non-blocking, review before tagging):"
    for log in $WARN_SCRIPTS; do
        s=$(basename "$log" .log)
        grep '^WARN' "$log" | sed "s/^/  $s: /"
    done
fi

echo ""
echo "logs: $LOG_DIR"

if [ -n "$FAIL_LIST" ]; then
    echo "Release NOT ready."
    exit 1
fi

if [ -n "$RELEASE_PREP" ]; then
    echo "All checks passed in --prep mode. Commit the release, then re-run WITHOUT --prep to clear for tagging."
else
    echo "All checks passed. Ready to tag."
fi
