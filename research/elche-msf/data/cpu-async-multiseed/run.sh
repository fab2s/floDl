#!/usr/bin/env bash
# Gate A — multi-seed EASGD α-sweep (200 epochs)
#
# Phase 2's single-seed EASGD smoke (seed 0, cpu-async × {msf,trend} ×
# --easgd-alpha 0.5) showed:
#   - msf+α=0.5: 91.91% eval / 408 syncs (matches α=1.0 msf baseline 91.86%
#     at 619 syncs → 34% sync reduction at parity eval).
#   - trend+α=0.5: 91.39% eval / 726 syncs (degraded by 0.79pp; impulsive-
#     coupling perturbation breaks trend's 3-rises-in-D detector).
#
# The Pareto aggregation (ddp-bench/runs/pareto-frontier-200ep/) shows the
# msf+α=0.5 cell DOMINATES the production default (nccl-async default + msf,
# 91.83% / 882 syncs). But it's n=1. Multi-seed confirmation is required to
# enter the paper's Pareto-frontier characterization.
#
# Sharp falsifiable predictions (per v2 doc Gate A spec, line ~498):
#   - msf+α=0.5 cross-seed mean within ±0.15pp of msf+α=1.0 baseline (91.86%)
#   - sync reduction ≥ 25% across all 4 seeds vs α=1.0 baseline
#   - trend+α=0.5 degrades by 0.5–1.0pp consistently
#
# Design: 4 seeds × 2 guards × cpu-async × --easgd-alpha 0.5
#   seeds: 1, 2, 3, 4 (seed 0 already in 2026-05-05 relaxed-easgd sweep)
#   = 8 runs ≈ 4h half-overnight
#
# Tree at staging: 0806f84 on ddp-scale (clean, post --min-anchor commit;
# v2 doc edits + pareto script uncommitted but don't affect the binary).
#
# To monitor:
#   tail -f ddp-bench/runs/overnight-2026-05-06-easgd-multiseed/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-06-easgd-multiseed"
RUNLOG="${BASE}/_runlog.txt"

cd "$(dirname "$0")/../../.."

mkdir -p "${BASE}"
: > "${RUNLOG}"

log() {
    printf "%s  %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

# Run one bench invocation via cuda-shell (bypasses the fdl preset clobber).
# Args: <run-name> <seed> <guard>
do_run() {
    local name="$1" seed="$2" guard="$3"
    local out_dir="${BASE}/${name}"
    mkdir -p "${out_dir}"

    local guard_flag=""
    [[ "${guard}" == "msf" ]] && guard_flag="--guard msf"

    log "START  ${name}  seed=${seed} guard=${guard}"
    local t0=$(date +%s)
    # cuda-shell cd's to /workspace then runs the binary directly. --output
    # path is relative to /workspace/ddp-bench (the binary's cwd inside the
    # container), so we strip the ddp-bench/ prefix from BASE.
    local out_in_container="${out_dir#ddp-bench/}/"
    fdl cuda-shell -- -c "cd /workspace/ddp-bench && ./target/release/ddp-bench \
        --model resnet-graph \
        --mode cpu-async \
        --gpus all \
        --epochs 200 \
        --per-epoch-eval \
        --seed ${seed} \
        --output ${out_in_container} \
        ${guard_flag} \
        --easgd-alpha 0.5" \
        > "${out_dir}/run.stdout.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    if [[ ${rc} -eq 0 ]]; then
        log "OK     ${name}  ${elapsed}s  exit=0"
    else
        log "FAIL   ${name}  ${elapsed}s  exit=${rc}  see ${out_dir}/run.stdout.log"
        : > "${out_dir}/.failed"
    fi
}

log "OVERNIGHT SWEEP START (Gate A — multi-seed EASGD α=0.5 confirmation)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "design: 4 seeds (1-4) × 2 guards (msf, trend) × cpu-async × --easgd-alpha 0.5 × 200 epochs"
log ""

# Outer loop on guard so msf cells (the load-bearing ones for the parity
# claim) come first — if hardware blows up partway through, we have the
# msf data to interpret, not the trend negative-control data.
for guard in msf trend; do
    log "PHASE: guard=${guard}"
    for seed in 1 2 3 4; do
        do_run "seed-${seed}-cpu-async-${guard}-easgd05" "${seed}" "${guard}"
    done
done

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
