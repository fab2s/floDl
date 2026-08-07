#!/usr/bin/env bash
# Multi-GPU / multi-host validation ladder for a real rig.
#
# The suites `fdl gpu-test-nccl` runs are gated on device count: every
# NCCL test opens with `if !require_multi_gpu() { return; }`, which is a
# SILENT early return reported as `ok`. On a one-GPU box the whole NCCL
# suite therefore passes in ~0.1s having tested nothing, and the summary
# says 16 passed. That is not a bug in the tests (a suite must not fail
# for lack of hardware), but it means a green NCCL run is evidence of
# nothing until you know how many GPUs were visible.
#
# So this script exists to answer one question honestly: what did this
# rig actually EXECUTE? Every rung reports RAN or SKIPPED with the
# reason, never a bare ok, and the summary counts them separately.
#
# The rungs climb in blast radius, each adding one thing:
#
#   1  in-VM / single-host   process-per-rank NCCL, one host, N GPUs
#   2  fan-out               controller outside the box, ranks inside
#   3  heterogeneous         mixed arch across hosts, per-host libnccl
#
# Rung 1 is the one that proves the rank control handshake: launcher,
# relay and rank processes all form a real cohort through it. Rung 2
# adds a controller that is not on the rank host. Rung 3 adds a second
# host with a different GPU architecture, which is what exercises the
# relay forwarding per-rank data upward.
#
# Local use:  bash ci/rig-ladder.sh [rung ...]     (default: all)
# Env:        FDL_RIG_MODE=nccl-sync|cpu-async     averaging backend
#             FDL_RIG_EPOCHS=1
#
# Not a CI job: it needs a rig. Overlays (`fdl.cluster*.yml`) are
# user-local, so a missing one is a SKIP, not a failure.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

MODE="${FDL_RIG_MODE:-nccl-sync}"
EPOCHS="${FDL_RIG_EPOCHS:-1}"
FDL=./fdl
LOGDIR="${FDL_RIG_LOGDIR:-target/rig-ladder}"
mkdir -p "$LOGDIR"

if [ -n "${NO_COLOR:-}" ] || [ ! -t 1 ]; then
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_OFF=""
else
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_OFF=$'\033[0m'
fi

RAN=0; SKIPPED=0; FAILED=0
ran()     { RAN=$((RAN+1));         echo "${C_GREEN}RAN${C_OFF}: $*"; }
skipped() { SKIPPED=$((SKIPPED+1)); echo "${C_YELLOW}SKIPPED${C_OFF}: $*"; }
failed()  { FAILED=$((FAILED+1));   echo "${C_RED}FAILED${C_OFF}: $*"; }

# A run counts as real only on all three: a zero exit, a `done:` line
# (the trainer's own end-of-run statement, so the cohort reached the
# end rather than exiting early), and no libtorch `[W...]` warnings,
# which never fail a run on their own and are how a silent correctness
# problem announces itself.
verify() {
    local name="$1" rc="$2" log="$3"
    if [ "$rc" -ne 0 ]; then
        failed "$name: exit $rc"
        tail -15 "$log" | sed 's/^/    /'
        return 1
    fi
    if ! grep -aq "done:" "$log"; then
        failed "$name: exit 0 but no 'done:' line -- the run did not finish training"
        tail -15 "$log" | sed 's/^/    /'
        return 1
    fi
    local warns
    warns=$(grep -acE "\[W[0-9]" "$log")
    if [ "$warns" -ne 0 ]; then
        failed "$name: $warns libtorch [W] warning(s) -- green summary, unhappy runtime"
        grep -aE "\[W[0-9]" "$log" | head -5 | sed 's/^/    /'
        return 1
    fi
    ran "$name -- $(grep -a 'done:' "$log" | tail -1 | sed 's/^ *//')"
    return 0
}

have_overlay() { [ -f "fdl.$1.yml" ]; }

# --- rung 1: process-per-rank NCCL on one host ------------------------
rung_1() {
    echo; echo "===== rung 1: process-per-rank NCCL, single host ====="
    if ! have_overlay cluster-test; then
        skipped "rung 1: no fdl.cluster-test.yml (user-local overlay)"
        return
    fi
    local log="$LOGDIR/rung1.log"
    "$FDL" @cluster-test cuda-test-nccl > "$log" 2>&1
    verify "rung 1 (single-host process-per-rank)" "$?" "$log"
}

# --- rung 2/3: cluster fan-out ----------------------------------------
# One overlay drives both: `fdl.cluster.yml`'s roster decides whether
# the cohort is one remote host or several of different architectures.
# The rung numbering describes the rig, not the command.
rung_3() {
    echo; echo "===== rung 3: heterogeneous fan-out cohort ====="
    if ! have_overlay cluster; then
        skipped "rung 3: no fdl.cluster.yml (user-local overlay)"
        return
    fi
    local log="$LOGDIR/rung3.log"
    "$FDL" @cluster ddp-bench --model mlp --mode "$MODE" \
        --epochs "$EPOCHS" --batch-size 256 > "$log" 2>&1
    verify "rung 3 (heterogeneous, mode=$MODE)" "$?" "$log"
}

# --- rung 4: walk-in through a guardrailed door -----------------------
# The cloud shape: controller here, worker dials in. Three processes and
# an operator gesture, so it is scripted rather than a single command --
# the controller holds a join window in the background, the worker walks
# in, and `fdl start` fires the topology freeze once quorum is met.
#
# Needs, and SKIPS loudly without: a farm overlay, a reachable worker,
# and a door whose authorized_keys already carries the farm's guardrailed
# line (`fdl join-config` composes it; installing it into a CONTAINER's
# key file is manual, since the installer only writes the invoking
# user's own ~/.ssh/authorized_keys).
rung_4() {
    echo; echo "===== rung 4: walk-in through the guardrailed door ====="
    local farm="${FDL_RIG_FARM:-rig}"
    local worker="${FDL_RIG_WORKER:-}"
    if ! have_overlay "$farm"; then
        skipped "rung 4: no fdl.$farm.yml (run: fdl join-config $farm)"
        return
    fi
    if [ -z "$worker" ]; then
        skipped "rung 4: set FDL_RIG_WORKER=<ssh-host> to name the walk-in box"
        return
    fi
    if [ -z "${FDL_RIG_WALKIN_CMD:-}" ]; then
        skipped "rung 4: set FDL_RIG_WALKIN_CMD to the walk-in box's own fdl join line"
        return
    fi
    local clog="$LOGDIR/rung4-controller.log" wlog="$LOGDIR/rung4-walkin.log"

    "$FDL" "@$farm" ddp-bench --model mlp --mode cpu-async \
        --epochs "$EPOCHS" --batch-size 256 > "$clog" 2>&1 &
    local cpid=$!
    # The window has to be open before the worker dials, or the dial is
    # refused and the rung fails for a reason that is not the one under
    # test.
    local waited=0
    until grep -aq "join: window open" "$clog" 2>/dev/null; do
        sleep 2; waited=$((waited+2))
        if [ "$waited" -ge 120 ] || ! kill -0 "$cpid" 2>/dev/null; then
            failed "rung 4: controller never opened a join window"
            tail -15 "$clog" | sed 's/^/    /'
            kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
            return 1
        fi
    done

    ssh "$worker" "$FDL_RIG_WALKIN_CMD" > "$wlog" 2>&1 &
    local wpid=$!

    waited=0
    until "$FDL" "@$farm" status 2>/dev/null | grep -q "roster startable"; do
        sleep 3; waited=$((waited+3))
        if [ "$waited" -ge 300 ]; then
            failed "rung 4: quorum never reached"
            tail -15 "$wlog" | sed 's/^/    /'
            kill "$cpid" "$wpid" 2>/dev/null; wait 2>/dev/null
            return 1
        fi
    done
    "$FDL" "@$farm" start >> "$clog" 2>&1

    wait "$cpid"; local crc=$?
    wait "$wpid" 2>/dev/null
    # The worker's own exit matters as much as the controller's: an agent
    # that died still lets the controller finish with the ranks it had.
    if ! grep -aq "finished cleanly" "$wlog"; then
        failed "rung 4: the walk-in agent did not finish cleanly"
        tail -15 "$wlog" | sed 's/^/    /'
        return 1
    fi
    verify "rung 4 (walk-in, farm=$farm, worker=$worker)" "$crc" "$clog"
}

# --- dispatch ---------------------------------------------------------
echo "rig ladder: mode=$MODE epochs=$EPOCHS logs=$LOGDIR"
echo "local GPUs visible: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"

if [ "$#" -eq 0 ]; then
    set -- 1 3 4
fi
for rung in "$@"; do
    case "$rung" in
        1) rung_1 ;;
        3) rung_3 ;;
        4) rung_4 ;;
        *) echo "unknown rung: $rung"; exit 1 ;;
    esac
done

echo
echo "ladder: $RAN ran, $SKIPPED skipped, $FAILED failed"
[ "$FAILED" -eq 0 ] || exit 1
# A ladder that skipped everything is not a pass. Say so in the exit
# code, because the whole point of this script is that silence about
# what did not run is how the NCCL suite fooled us in the first place.
if [ "$RAN" -eq 0 ]; then
    echo "${C_YELLOW}nothing ran${C_OFF} -- this rig validated nothing"
    exit 2
fi
