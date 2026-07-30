#!/bin/sh
# Shell-script hygiene:
#   - `sh -n` (syntax check) on every tracked `.sh` OUTSIDE ci/release/.
#   - `shellcheck -S warning` if installed (advisory, never fails).
#
# The ci/release/ scripts are excluded: run-all.sh executes each of them on
# every suite run, so a syntax error there fails the suite directly rather
# than needing a separate check.

set -u
cd "$(git rev-parse --show-toplevel)"

FAIL=0
SCRIPTS=$(git ls-files '*.sh' | grep -v '^ci/release/')

COUNT=0
for s in $SCRIPTS; do
    COUNT=$((COUNT + 1))
    # Pick the interpreter from the shebang so bash-specific syntax
    # (arrays, [[ ]], $(( )) extensions) in bash scripts doesn't trip
    # plain sh -n. Falls back to sh when no bash shebang is present.
    head1=$(head -1 "$s")
    case "$head1" in
        *bash*) interp="bash" ;;
        *)      interp="sh"   ;;
    esac
    if ! $interp -n "$s" 2>/tmp/fdl-shell-err; then
        echo "FAIL: syntax error in $s (checked with $interp -n)"
        sed 's/^/  /' /tmp/fdl-shell-err
        FAIL=1
    fi
done
rm -f /tmp/fdl-shell-err

if command -v shellcheck >/dev/null 2>&1; then
    FINDINGS=$(shellcheck -S warning $SCRIPTS 2>&1 || true)
    if [ -n "$FINDINGS" ]; then
        echo "WARN: shellcheck warnings (advisory, not failing):"
        echo "$FINDINGS" | sed 's/^/  /'
    fi
fi

[ "$FAIL" = 0 ] && echo "PASS: sh -n clean on $COUNT scripts"
exit "$FAIL"
