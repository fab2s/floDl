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

# Second pass, under bash 3.2 -- the oldest shell any supported platform
# ships, and the one macOS still has at /bin/bash (and behind /bin/sh).
# The pass above uses the LOCAL bash, so on a modern box it is blind to
# everything 3.2 cannot parse, and that blindness has cost a red macOS
# leg: `FEATURE=$(case "$x" in a*) echo y ;; esac)` parses on bash 5 and
# dies on 3.2, where the `)` of the first pattern closes the `$(`.
#
# Every script, not a macOS-bound subset. init.sh and download-libtorch.sh
# are user-facing installers that run there, os-matrix.sh runs there in
# CI, and none of the rest needs syntax 3.2 lacks. When one legitimately
# does, that is the moment to add an exclusion, not before.
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    # One line, not the newline-separated list: newlines inside a `for`
    # word list terminate the statement, so the loop lost its `do`.
    SCRIPTS_1L=$(echo $SCRIPTS)
    if docker run --rm -v "$(pwd):/w:ro" -w /w bash:3.2 \
            bash -c 'for s in '"$SCRIPTS_1L"'; do bash -n "$s" || exit 1; done' \
            >/tmp/fdl-shell32-err 2>&1; then
        echo "PASS: bash 3.2 parses all $COUNT scripts"
    else
        echo "FAIL: a script does not parse under bash 3.2 (what macOS ships)"
        sed 's/^/  /' /tmp/fdl-shell32-err
        FAIL=1
    fi
    rm -f /tmp/fdl-shell32-err
else
    # Loud, because a silent skip here reads as coverage. CI has docker,
    # so this branch is a developer-box convenience, never the CI result.
    echo "UNVERIFIED: no usable docker, so the bash 3.2 pass did NOT run"
fi

if command -v shellcheck >/dev/null 2>&1; then
    FINDINGS=$(shellcheck -S warning $SCRIPTS 2>&1 || true)
    if [ -n "$FINDINGS" ]; then
        echo "WARN: shellcheck warnings (advisory, not failing):"
        echo "$FINDINGS" | sed 's/^/  /'
    fi
fi

[ "$FAIL" = 0 ] && echo "PASS: sh -n clean on $COUNT scripts"
exit "$FAIL"
