#!/usr/bin/env bash
# MSF passive-observation overnight sweep
# Tree: ddp-scale (load_averaged + eval-streaming fixes committed)
# Workload: resnet-graph (ResNet-20 / CIFAR-10), 200 epochs, 3-GPU heterogeneous
# Total: 1 validation + 20 sweep runs ≈ 10.5h
#
# Per-run output goes to <BASE>/<run-name>/resnet-graph/<mode>/
# Per-run runlog appends to <BASE>/_runlog.txt
# Per-run captured stdout/stderr goes to <BASE>/<run-name>/run.stdout.log
#
# Failures don't halt the sweep. Each invocation's exit code is logged.
# To monitor progress:  tail -f ddp-bench/runs/overnight-2026-05-04/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-04"
RUNLOG="${BASE}/_runlog.txt"

# Ensure cwd is project root (fdl requires it)
cd "$(dirname "$0")/../../.."

mkdir -p "${BASE}"
: > "${RUNLOG}"  # truncate

log() {
    printf "%s  %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

# Run one bench invocation. Args: <run-name> <seed> <mode> <guard>
# guard ∈ {trend, msf}
do_run() {
    local name="$1" seed="$2" mode="$3" guard="$4"
    local out_dir="${BASE}/${name}"
    mkdir -p "${out_dir}"

    local guard_flag=""
    [[ "${guard}" == "msf" ]] && guard_flag="--guard msf"

    log "START  ${name}  seed=${seed} mode=${mode} guard=${guard}"
    local t0=$(date +%s)
    fdl ddp-bench \
        --model resnet-graph \
        --mode "${mode}" \
        --gpus all \
        --epochs 200 \
        --per-epoch-eval \
        --seed "${seed}" \
        --output "${out_dir}/" \
        ${guard_flag} \
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

log "OVERNIGHT SWEEP START"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log ""

# ---------------------------------------------------------------------------
# Validation: cpu-async + MSF + seed 42 (matches today's pre-fix run)
# ---------------------------------------------------------------------------
log "validation: load_averaged fix verification"
do_run "validation" 42 cpu-async msf

# ---------------------------------------------------------------------------
# N=5 sweep, seed-major across {nccl,cpu}-async × {trend, msf}
# ---------------------------------------------------------------------------
log ""
log "N=5 sweep (seed-major)"

for seed in 0 1 2 3 4; do
    do_run "seed-${seed}-nccl-async-trend" "${seed}" nccl-async trend
    do_run "seed-${seed}-nccl-async-msf"   "${seed}" nccl-async msf
    do_run "seed-${seed}-cpu-async-trend"  "${seed}" cpu-async  trend
    do_run "seed-${seed}-cpu-async-msf"    "${seed}" cpu-async  msf
done

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
