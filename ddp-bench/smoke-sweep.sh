#!/bin/bash
# Release-gating regression smoke for the 3-GPU heterogeneous rig.
#
# This is NOT a benchmark. It answers one question before a release:
# did the changes since the last published sweep break DISTRIBUTED
# training? Wall-times here are informational; convergence is the gate.
#
#   ./smoke-sweep.sh          # run it
#   ./smoke-sweep.sh compare  # re-print the comparison table only
#
# WHAT IT IS SHAPED TO CATCH. The risk since 0.7.0 concentrates in two
# places, so every cell exercises both:
#   - the CPU averaging plane: structural-zero elision (PROTOCOL_VERSION
#     2->3), the RAM-neutral consensus decode, the bf16 writeback bounce.
#     All three rewrote how bytes move through the reduce path.
#   - formation: CONTROL_PROTOCOL_VERSION 2->4, the model-signature
#     check, the vendor gate. Every cell forms a cohort, so every cell
#     tests it.
# Small models are sufficient for both: the plane does not care how big
# the tensors are, only that they move.
#
# BLOCKS (~44 cells, ~43 min measured against publish-3gpu timings):
#   A. six small models x six modes, at the PUBLISHED epochs and seed,
#      so every cell is a direct A/B against runs/publish-3gpu.
#   B. resnet-graph at full 200 epochs on cpu-async + cpu-async-diloco.
#      These are the highest-value cells in the sweep: they are the only
#      ones carrying BatchNorm BUFFERS through the reduce path, they run
#      exactly the code that changed most, and both are published, so
#      they are comparable rather than merely "it ran".
#   C. resnet-graph at 20 epochs on the remaining four modes. Not
#      comparable (different epoch count); the bar is runs-clean.
#   D. two orphan paths that exist NOWHERE in publish-3gpu, so nothing
#      else here touches them: --epoch-splits and --bf16-wire. Run on
#      char-rnn rather than a toy model so the wire actually carries
#      something. Bar is runs-clean.
#
# WHY resnet-graph AND NOT resnet: eager resnet has only nccl-cadence
# published; resnet-graph carries the full mode suite, so comparability
# lives on the graph engine.
#
# Deliberately NOT here: olmo / olmo-graph. That is its own arc with its
# own provenance discipline, and it wants a RELEASED version to cite.
set -u
cd "$(dirname "$0")/.."

OUT=${SMOKE_OUT:-runs/smoke-0.8.0}
BASE=${SMOKE_BASE:-ddp-bench/runs/publish-3gpu}
ABS_OUT=ddp-bench/$OUT
SEED=${SMOKE_SEED:-42}
mkdir -p "$ABS_OUT"

SMALL="logistic:5 mlp:5 lenet:5 conv-ae:5 char-rnn:50 gpt-nano:50"
MODES="nccl-sync nccl-cadence cpu-sync cpu-cadence cpu-async cpu-async-diloco"
RESNET_FULL_MODES="cpu-async cpu-async-diloco"
RESNET_SMOKE_MODES="nccl-sync nccl-cadence cpu-sync cpu-cadence"
RESNET_FULL_EPOCHS=200
RESNET_SMOKE_EPOCHS=20

ts()   { date '+%F %T'; }
say()  { printf '%s %s\n' "$(ts)" "$*"; }

# Evidence of WORK, not evidence the process reached its exit path. The
# `# total:` footer is written at teardown whether or not anything
# trained; gating on it alone lets an empty cell read as passed and be
# skipped by every resume. Same predicate as sweep.sh.
cell_done() {
    log="$ABS_OUT/$1/training.log"
    grep -q '^epoch ' "$log" 2>/dev/null && grep -q '# total:' "$log" 2>/dev/null
}

strays() { pgrep -af 'release/ddp-benc[h]' >/dev/null 2>&1; }

mode_args() {
    case "$1" in
        cpu-async-diloco) echo "--mode cpu-async --outer-optimizer diloco" ;;
        *)                echo "--mode $1" ;;
    esac
}

# Per-cell provenance. The olmo cells taught this: runs/olmo (481.7s, one
# epoch) and runs/smoke-splits/olmo (149.2s, four splits) cannot be
# compared at all, because the token budget explaining the 3x spread was
# never written down. A reference that cannot say what produced it is an
# anecdote with good formatting. sweep.sh echoes rev= and seed= to its
# own stdout, which survives only wherever the operator piped it.
# Called from the SUCCESS path only, so provenance exists if and only if
# the cell holds results. Stamping before the run would leave a fresh SHA
# and timestamp on a cell that failed, or re-stamp a resumed cell with a
# tree that did not produce it -- the exact lie this file exists to stop.
stamp() {
    dir="$ABS_OUT/$1"; shift
    mkdir -p "$dir"
    {
        echo "invocation: $*"
        echo "git_sha:    $(git rev-parse HEAD 2>/dev/null)"
        echo "git_dirty:  $(git status --porcelain 2>/dev/null | wc -l) file(s) modified"
        echo "seed:       $SEED"
        echo "utc:        $(date -u '+%F %T UTC')"
        echo "host:       $(hostname)"
        echo "libtorch:   $(cat libtorch/.active 2>/dev/null || echo unknown)"
        echo "baseline:   $BASE"
        for h in $(grep -oE '^\s+- host: [a-z0-9-]+' fdl.cluster.yml 2>/dev/null | awk '{print $3}'); do
            echo "worker:     $h"
        done
    } > "$dir/provenance.txt"
}

# run_cell <celldir> <model> <epochs> <extra args...>
run_cell() {
    cell=$1; model=$2; epochs=$3; shift 3
    if cell_done "$cell"; then say "SKIP  $cell (done)"; return 0; fi
    say "START $cell (model=$model epochs=$epochs $*)"

    # `--output` takes the run root; ddp-bench appends <model>/<mode>/.
    # Cells whose dir would collide (the orphan-path pair reuses
    # char-rnn/cpu-async) get their own root, passed by the caller.
    log=$(mktemp)
    timeout 5400 ./fdl @cluster ddp-bench --model "$model" "$@" \
        --epochs "$epochs" --seed "$SEED" --save-dashboard 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}

    # A DEGRADED run exits 0 -- correct for training under elastic
    # membership, invalid for a comparison cell. Reject it, and reject a
    # cell that emitted a libtorch warning: those do not fail a run and
    # are exactly what a green summary hides.
    degraded=$(grep -c "finished DEGRADED\|child exit(s) tolerated\|device-side assert" "$log")
    warns=$(grep -cE '\[W[0-9]* ' "$log")
    rm -f "$log"

    if [ "$rc" -eq 0 ] && [ "$degraded" -eq 0 ] && [ "$warns" -eq 0 ] && cell_done "$cell"; then
        stamp "$cell" "fdl @cluster ddp-bench --model $model $* --epochs $epochs --seed $SEED --save-dashboard"
        say "OK    $cell"
        return 0
    fi

    say "FAIL  $cell rc=$rc degraded=$degraded libtorch_warnings=$warns"
    rm -rf "${ABS_OUT:?}/$cell"
    # The agent wrapper's kill-trap needs up to 10s to KILL its child
    # after a rank crash; probing sooner aborts on strays that clear
    # themselves. 15s separates those from a genuinely wedged launcher.
    sleep 15
    if strays; then
        say "ABORT-DIRTY: leftover ddp-bench processes need manual cleanup:"
        pgrep -af 'release/ddp-benc[h]'
        exit 2
    fi
    return 1
}

# --- comparison -------------------------------------------------------
# Convergence is the gate. Wall-clock is printed but never fails a cell:
# ~/src moved to a faster NVMe between the baseline and now, so any
# I/O-sensitive cell (resnet reads CIFAR, char-rnn and gpt-nano read
# corpora) is expected to shift. That shift is itself a datum for the
# open "virtiofs pread latency vs param-plane transport" question, which
# is why it is reported rather than suppressed.
final_eval() { grep -oE 'final eval=[0-9.]+' "$1" 2>/dev/null | tail -1 | cut -d= -f2; }
last_loss()  { grep -oE 'loss=[0-9.]+'      "$1" 2>/dev/null | tail -1 | cut -d= -f2; }
total_s()    { grep -oE '# total: [0-9.]+'  "$1" 2>/dev/null | tail -1 | awk '{print $3}'; }

# Metric DIRECTION is per model and it is not guessable from the value:
# logistic/mlp/lenet/resnet-graph report accuracy (higher better) while
# conv-ae/char-rnn/gpt-nano/olmo report a loss (lower better). Read it
# from the model's own definition rather than hardcoding a table here,
# so this cannot drift from the binary. Each src/models/<name>.rs carries
# one `eval_higher_is_better:`; the file name is the model with - as _.
higher_is_better() {
    grep -q 'eval_higher_is_better: true' \
        "ddp-bench/src/models/$(echo "$1" | tr - _).rs" 2>/dev/null
}

# The relative delta is only meaningful once the absolute difference
# clears the log's print resolution. training.log carries eval to 4dp, so
# a 1-2 ulp difference is quantisation, not signal. conv-ae proved it on
# the 2026-08-09 run: 0.0009 vs 0.0010 is ONE unit in the last printed
# digit and rendered as -10.00%, flagging a cell that had not moved.
ULP=0.0002

# verdict <new> <base> <higher_is_better 1|0>  ->  "<delta> <verdict>"
verdict() {
    awk -v a="$1" -v b="$2" -v hib="$3" -v ulp="$ULP" 'BEGIN{
        if (a == "" || b == "") { print "n/a  ?"; exit }
        d = a - b; ad = (d < 0 ? -d : d)
        if (ad < ulp)  { printf "%+.2f noise\n", 0; exit }
        if (b + 0 == 0){ print "n/a  ?"; exit }
        rel = d / b * 100
        better = (hib ? (d > 0) : (d < 0))
        if (rel > -5 && rel < 5) { printf "%+.2f ok\n", rel; exit }
        printf "%+.2f %s\n", rel, (better ? "BETTER" : "WORSE")
    }'
}

row() {  # <model> <mode> <newlog> <oldlog>
    m=$1; mode=$2; new=$3; old=$4
    [ -f "$new" ] && [ -f "$old" ] || return 0
    ne=$(final_eval "$new"); oe=$(final_eval "$old")
    [ -n "$ne" ] && [ -n "$oe" ] || { ne=$(last_loss "$new"); oe=$(last_loss "$old"); }
    if higher_is_better "$m"; then hib=1; dir="acc"; else hib=0; dir="loss"; fi
    set -- $(verdict "$ne" "$oe" "$hib")
    d=$1; v=$2
    case "$v" in WORSE|BETTER) flagged=$((flagged+1)); mark=" <-- $v" ;; *) mark="" ;; esac
    printf '%-14s %-18s %-5s %9s %9s %8s   %8s %8s%s\n' \
        "$m" "$mode" "$dir" "${ne:-?}" "${oe:-?}" "$d" \
        "$(total_s "$new")" "$(total_s "$old")" "$mark"
}

compare() {
    printf '\n%s\n' "=== convergence vs $BASE (the gate) ==="
    printf '%-14s %-18s %-5s %9s %9s %8s   %8s %8s\n' MODEL MODE DIR EVAL BASE DELTA% WALL BASE
    flagged=0
    for entry in $SMALL; do
        m=${entry%%:*}
        for mode in $MODES; do
            row "$m" "$mode" "$ABS_OUT/$m/$mode/training.log" "$BASE/$m/$mode/training.log"
        done
    done
    for mode in $RESNET_FULL_MODES; do
        row resnet-graph "$mode" "$ABS_OUT/resnet-graph/$mode/training.log" "$BASE/resnet-graph/$mode/training.log"
    done
    printf '\n  DIR   = the model'"'"'s own eval_higher_is_better (acc: up is good, loss: down is good).\n'
    printf '  noise = absolute difference below %s, the log'"'"'s 4dp print resolution.\n' "$ULP"
    printf '  BETTER/WORSE flag a >5%% move that clears that floor -- for a human to judge,\n'
    printf '  not auto-failed: these are short runs and the modes differ by design.\n'
    printf '  cells flagged: %s\n' "$flagged"
}

# --- run --------------------------------------------------------------
[ "${1:-}" = "compare" ] && { compare; exit 0; }

say "SMOKE BEGIN rev=$(git rev-parse --short HEAD) seed=$SEED out=$OUT base=$BASE"
FAILED=0

say "--- block A: small models, published epochs, all modes (comparable) ---"
for entry in $SMALL; do
    m=${entry%%:*}; e=${entry##*:}
    for mode in $MODES; do
        run_cell "$m/$mode" "$m" "$e" $(mode_args "$mode") --output "$OUT" || FAILED=$((FAILED+1))
    done
done

say "--- block B: resnet-graph full $RESNET_FULL_EPOCHS ep, async pair (comparable, buffers) ---"
for mode in $RESNET_FULL_MODES; do
    run_cell "resnet-graph/$mode" resnet-graph "$RESNET_FULL_EPOCHS" $(mode_args "$mode") --output "$OUT" || FAILED=$((FAILED+1))
done

say "--- block C: resnet-graph $RESNET_SMOKE_EPOCHS ep, remaining modes (runs-clean only) ---"
for mode in $RESNET_SMOKE_MODES; do
    # Cell dir is <output>/<model>/<mode>, so the stamp path must carry
    # the model segment too or provenance lands beside the cell, not in it.
    run_cell "resnet-graph-smoke/resnet-graph/$mode" resnet-graph "$RESNET_SMOKE_EPOCHS" \
        $(mode_args "$mode") --output "$OUT/resnet-graph-smoke" || FAILED=$((FAILED+1))
done

say "--- block D: orphan paths absent from the baseline (runs-clean only) ---"
# Own --output root each: the dir is <model>/<mode>, so both would land
# on char-rnn/cpu-async and overwrite each other.
run_cell "paths-epoch-splits/char-rnn/cpu-async" char-rnn 50 \
    --mode cpu-async --epoch-splits 4 --output "$OUT/paths-epoch-splits" || FAILED=$((FAILED+1))

run_cell "paths-bf16-wire/char-rnn/cpu-async" char-rnn 50 \
    --mode cpu-async --bf16-wire --output "$OUT/paths-bf16-wire" || FAILED=$((FAILED+1))

say "SMOKE END failed_cells=$FAILED"
compare
[ "$FAILED" -eq 0 ] || exit 1
