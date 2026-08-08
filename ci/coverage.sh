#!/usr/bin/env bash
# Coverage across every suite this box can run, plus an explicit account
# of the ones it could not.
#
# The number a single `cargo llvm-cov` prints is not wrong, it is
# PARTIAL, and partial in a way that reads like a verdict. Measured on
# this workspace, CPU-only against CPU+GPU merged:
#
#     tensor/cuda_stream.rs    9.96%  ->  89.32%
#     tensor/cuda_event.rs    17.19%  ->  92.71%
#     nn/cuda_graph.rs        12.11%  ->  88.66%
#     tensor/cuda.rs          21.41%  ->  65.88%
#
# Those files were never poorly tested, they were UNREACHABLE. So a
# coverage tool that runs one suite quietly slanders whatever the suite
# cannot reach, and the fix is not a better threshold, it is running the
# other suites and naming the ones that did not run.
#
# WHY A SCRIPT AND NOT AN fdl `run:` LINE
# The phases execute in DIFFERENT containers (CPU in `dev`, GPU in
# `cuda`/`rocm`) and an fdl command names exactly one `docker:` service.
# So this drives compose directly, and is itself invoked by a
# `docker:`-less `fdl coverage-all`, which is what exports
# FDL_GPU_FEATURE (see `libtorch_env` in flodl-cli/src/run.rs).
#
# HOW THE MERGE WORKS
# Each phase runs `cargo llvm-cov --no-report`, which runs the tests,
# keeps the .profraw and generates nothing; a final `report` merges all
# of them. That one flag IS the accumulate mechanism, including not
# wiping the previous phase's data, which is why only phase 1 cleans
# (explicitly, up front) and why `--no-clean` must NOT be added here:
# cargo-llvm-cov rejects the pair outright with `--no-report may not be
# used together with --no-clean`, and every GPU phase fails.
#
# What makes the merge sound ACROSS cfgs is separate: llvm-cov needs both
# the profraw and the instrumented binary that produced it, and cargo
# names test binaries with a metadata hash that includes the feature set.
# So the CPU phase's binary and the GPU phase's binary coexist under
# deps/ and each phase's data maps to its own rather than to a binary
# that has since been overwritten.
#
# Verified 2026-08-08 on a purpose-built crate whose one function had
# each branch covered by a DIFFERENT phase: merged regions came out as
# the max (23) not the sum (43), so nothing is double-counted; merged
# missed (11) was lower than either phase alone (12, 15), so coverage
# genuinely unions; and `report --text` showed a hit marker on both
# branches, one from each phase. A function excluded from both phases
# correctly stayed at 0, so the merge does not invent coverage either.
#
# WHAT THE TOTAL MEANS
# The denominator is the SUPERSET of regions across cfgs, so GPU-only
# code counts against the total even when the GPU phases were skipped.
# That UNDER-reports rather than flatters, which is the direction a
# coverage number should err. It also means totals from two different
# runs of this script are only comparable when the same phases ran,
# which is why the summary prints the phase roster every time.
#
# `--skip leakcheck` is in every phase and is required, not tidiness:
# those tests assert RSS growth stays under a bound, and the coverage
# instrumentation allocates counters that count as growth.
#
# Local use:  fdl coverage-all            (preferred: exports FDL_GPU_FEATURE)
#             bash ci/coverage.sh
# Env:        FDL_COV_HTML=1              also write target/llvm-cov/html
#             FDL_COV_SKIP_GPU=1          CPU phase only, on purpose

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -n "${NO_COLOR:-}" ] || [ ! -t 1 ]; then
    C_GREEN=""; C_RED=""; C_YELLOW=""; C_OFF=""
else
    C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'; C_OFF=$'\033[0m'
fi

LOGDIR="${FDL_COV_LOGDIR:-target/coverage}"
mkdir -p "$LOGDIR"

RAN=0; SKIPPED=0; FAILED=0
ROSTER=""

ran()     { RAN=$((RAN+1));         ROSTER="$ROSTER  ${C_GREEN}RAN${C_OFF}      $*"$'\n'; echo "${C_GREEN}RAN${C_OFF}: $*"; }
skipped() { SKIPPED=$((SKIPPED+1)); ROSTER="$ROSTER  ${C_YELLOW}SKIPPED${C_OFF}  $*"$'\n'; echo "${C_YELLOW}SKIPPED${C_OFF}: $*"; }
failed()  { FAILED=$((FAILED+1));   ROSTER="$ROSTER  ${C_RED}FAILED${C_OFF}   $*"$'\n'; echo "${C_RED}FAILED${C_OFF}: $*"; }

# --- what this box can do ---------------------------------------------
# Device count comes from `fdl probe`, which reads the same flodl-hw
# survey the framework itself uses, rather than a second nvidia-smi parse
# that could disagree with it. Counting `"index":` is gpu-array-specific
# in that JSON (`archs_match` entries key on "gpu", not "index").
#
# This is the HOST's view. A GPU container sees what was passed through
# to it, so a box that hides devices from the container will over-count
# here; the phases themselves still behave correctly, the roster line is
# what would read optimistically.
GPUS=0
if [ -x ./fdl ]; then
    GPUS=$(./fdl probe --json 2>/dev/null | grep -o '"index":' | wc -l | tr -d ' ')
fi
[ -n "$GPUS" ] || GPUS=0

FEATURE="${FDL_GPU_FEATURE:-cuda}"
case "$FEATURE" in
    rocm) GPU_SERVICE=rocm ;;
    *)    GPU_SERVICE=cuda ;;
esac

echo "===== coverage-all ====="
echo "GPUs visible (host): $GPUS"
echo "GPU feature/service: $FEATURE / $GPU_SERVICE"
echo "logs:                $LOGDIR"
echo

# --- phase runner ------------------------------------------------------
# Every phase is `cargo llvm-cov --no-report`; only the first may clean.
# A phase failure is recorded and does NOT abort the run: a red NCCL
# phase still leaves usable data from the phases that passed, and the
# summary says which was which.
phase() {
    local name="$1" service="$2" log="$LOGDIR/$3.log"; shift 3
    echo "--- $name ---"
    docker compose run --rm --no-deps "$service" "$@" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        failed "$name (exit $rc, see $log)"
        tail -12 "$log" | sed 's/^/    /'
        return 1
    fi
    ran "$name"
    return 0
}

# --- phase 1: CPU workspace -------------------------------------------
# The only phase that cleans, and the only one that is --workspace: it is
# what instruments flodl-cli/flodl-hw/flodl-hf at all. Everything after
# adds GPU-reachable regions to the same target dir.
echo "--- phase 1: CPU workspace (clean start) ---"
docker compose run --rm --no-deps dev cargo llvm-cov clean --workspace \
    > "$LOGDIR/clean.log" 2>&1 || true

phase "phase 1: CPU workspace" dev cpu \
    cargo llvm-cov --no-report --workspace -- --skip leakcheck

# --- phases 2-4: GPU ---------------------------------------------------
if [ "${FDL_COV_SKIP_GPU:-0}" = "1" ]; then
    skipped "phase 2: GPU parallel (FDL_COV_SKIP_GPU=1)"
    skipped "phase 3: GPU serial (FDL_COV_SKIP_GPU=1)"
    skipped "phase 4: NCCL / multi-GPU (FDL_COV_SKIP_GPU=1)"
elif [ "$GPUS" -lt 1 ]; then
    skipped "phase 2: GPU parallel (no GPU visible)"
    skipped "phase 3: GPU serial (no GPU visible)"
    skipped "phase 4: NCCL / multi-GPU (no GPU visible)"
else
    phase "phase 2: GPU parallel" "$GPU_SERVICE" gpu-parallel \
        cargo llvm-cov --no-report -p flodl -p flodl-hf \
        --features "$FEATURE" -- --skip leakcheck

    # `--ignored` + single-threaded: Graphs, manual_seed and the probes
    # cannot share the device with anything else.
    phase "phase 3: GPU serial" "$GPU_SERVICE" gpu-serial \
        cargo llvm-cov --no-report -p flodl -p flodl-hf \
        --features "$FEATURE" -- --ignored --test-threads=1 \
        --skip nccl --skip graph_distribute --skip _live --skip leakcheck

    # Every NCCL test opens with `if !require_multi_gpu() { return; }`,
    # a SILENT early return reported as ok. On a one-GPU box the suite
    # would pass in ~0.1s having tested nothing and contributed no
    # coverage, so gate on the device count instead of letting a green
    # phase stand in for a run that never happened.
    if [ "$GPUS" -lt 2 ]; then
        skipped "phase 4: NCCL / multi-GPU ($GPUS GPU visible, needs 2 -- the suite would self-skip and report ok)"
    else
        phase "phase 4a: NCCL" "$GPU_SERVICE" gpu-nccl \
            cargo llvm-cov --no-report -p flodl -p flodl-hf \
            --features "$FEATURE" -- --ignored --test-threads=1 nccl
        phase "phase 4b: graph_distribute" "$GPU_SERVICE" gpu-graph-dist \
            cargo llvm-cov --no-report -p flodl -p flodl-hf \
            --features "$FEATURE" -- --ignored --test-threads=1 graph_distribute
    fi
fi

# --- phase 5: merged report -------------------------------------------
echo
echo "--- phase 5: merged report ---"
docker compose run --rm --no-deps dev \
    cargo llvm-cov report --summary-only 2>&1 | tee "$LOGDIR/report.txt"
REPORT_RC=${PIPESTATUS[0]}

if [ "${FDL_COV_HTML:-0}" = "1" ]; then
    docker compose run --rm --no-deps dev cargo llvm-cov report --html \
        > "$LOGDIR/html.log" 2>&1 \
        && echo "html: target/llvm-cov/html/index.html"
fi

# --- the honest part ---------------------------------------------------
echo
echo "===== what this number covers ====="
printf '%s' "$ROSTER"
echo
echo "phases ran: $RAN   skipped: $SKIPPED   failed: $FAILED"

if [ "$SKIPPED" -gt 0 ]; then
    cat <<'EOF'

CAVEAT: at least one suite did not run. Regions only that suite can
reach are counted in the denominator and scored as missed, so the total
below is a FLOOR, not this workspace's coverage. Files to distrust when
the GPU phases are absent: flodl/src/tensor/cuda*.rs,
flodl/src/distributed/**, flodl/src/nn/graph*.rs.
EOF
fi

if [ "$FAILED" -gt 0 ]; then
    cat <<'EOF'

A phase FAILED, so its data is partial or absent. Note before chasing it
that coverage instrumentation is itself a plausible cause: it slows every
rank, and the cluster coordinator's heartbeat deadline is 30s, so an NCCL
phase that is green under `fdl gpu-test-nccl` and red here is more likely
an artifact of instrumentation than a finding. Re-run that suite without
coverage before believing it.
EOF
fi

[ "$FAILED" -eq 0 ] && [ "$REPORT_RC" -eq 0 ]
