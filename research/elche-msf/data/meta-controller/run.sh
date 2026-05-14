#!/usr/bin/env bash
# 0.6 LR-aware meta-controller — validation sweep (cpu-async + nccl-async)
#
# Mirrors passive-observation's recipe (the bare meta-OFF baseline) and
# adds --meta-controller. Pair this sweep cell-for-cell against
# research/elche-msf/data/passive-observation/seed-{1..4}-{mode}-{guard}/.
#
# Decision criteria (from project_06_controller_arc.md, locked 2026-05-09):
#   - meta-on preserves sync-count saving at parity eval (paired-seed
#     Δ_eval < 0.15 pp).
#   - meta-on reduces seed sd vs meta-off (load-bearing claim — phase-aware
#     reactive correction handles LR-decay variance).
#   - Convergence watcher does NOT fire spuriously on the msf-silent cohort
#     (verified via "meta-nudge" Custom-event count in the timeline; expected
#     0 fires across the msf cells).
#
# Design: 4 seeds × 2 guards × 2 modes × meta-on × 200 epochs
#   seeds: 1, 2, 3, 4 (matches cpu-async-multiseed's seed numbering)
#   modes: cpu-async (with α=0.5), nccl-async (default)
#   = 16 runs ≈ 9–10h single overnight
#
# To monitor:
#   tail -f ddp-bench/runs/overnight-2026-05-09-meta-controller/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-09-meta-controller"
RUNLOG="${BASE}/_runlog.txt"

cd "$(dirname "$0")/../../../.."

mkdir -p "${BASE}"
: > "${RUNLOG}"

log() {
    printf "%s  %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

# Run one bench invocation via cuda-shell (bypasses the fdl preset clobber).
# Args: <run-name> <seed> <mode> <guard>
do_run() {
    local name="$1" seed="$2" mode="$3" guard="$4"
    local out_dir="${BASE}/${name}"
    mkdir -p "${out_dir}"

    # cpu-async honors EASGD α (load_averaged blend); nccl-async uses
    # in-place AllReduce(Avg) and ignores --easgd-alpha. Match the
    # passive-observation recipe so the paired-seed comparison is clean.
    local alpha_flag=""
    [[ "${mode}" == "cpu-async" ]] && alpha_flag="--easgd-alpha 0.5"

    log "START  ${name}  seed=${seed} mode=${mode} guard=${guard}"
    local t0=$(date +%s)
    local out_in_container="${out_dir#ddp-bench/}/"
    fdl cuda-shell -- -c "cd /workspace/ddp-bench && ./target/release/ddp-bench \
        --model resnet-graph \
        --mode ${mode} \
        --gpus all \
        --epochs 200 \
        --per-epoch-eval \
        --seed ${seed} \
        --output ${out_in_container} \
        --guard ${guard} \
        ${alpha_flag} \
        --meta-controller" \
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

log "OVERNIGHT SWEEP START (0.6 meta-controller validation)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "design: 4 seeds (1-4) × 2 guards (msf, trend) × 2 modes (cpu-async α=0.5, nccl-async default) × 200 epochs × meta-on"
log ""

# Seed-major outer loop (matches passive-observation order). Within each
# seed: nccl first (cheaper per epoch → fail-fast signal), then cpu-async.
# Both guards each.
for seed in 1 2 3 4; do
    do_run "seed-${seed}-nccl-async-trend-meta" "${seed}" nccl-async trend
    do_run "seed-${seed}-nccl-async-msf-meta"   "${seed}" nccl-async msf
    do_run "seed-${seed}-cpu-async-trend-meta"  "${seed}" cpu-async  trend
    do_run "seed-${seed}-cpu-async-msf-meta"    "${seed}" cpu-async  msf
done

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
