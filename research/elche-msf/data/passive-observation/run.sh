#!/usr/bin/env bash
# passive-observation full re-run (clean-tree consolidation)
#
# Workload: resnet-graph (ResNet-20 / CIFAR-10), 200 epochs, 3-GPU heterogeneous
# Axes: 5 seeds (0-4) x 2 modes (cpu-async, nccl-async) x 2 guards (msf, trend)
# cpu-async cells use --easgd-alpha 0.5 (post-EASGD canonical alpha).
# nccl-async cells use default anchor (no --easgd-alpha; NCCL path uses in-place AllReduce).
# Total: 20 cells, ~10h.
#
# Per-run output goes to <BASE>/<run-name>/resnet-graph/<mode>/
# Per-run runlog appends to <BASE>/_runlog.txt
# Per-run captured stdout/stderr goes to <BASE>/<run-name>/run.stdout.log
#
# Failures don't halt the sweep. Each invocation's exit code is logged.
# To monitor:  tail -f ddp-bench/runs/overnight-2026-05-07-passive-observation/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-07-passive-observation"
RUNLOG="${BASE}/_runlog.txt"

cd "$(dirname "$0")/../../.."

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
        ${alpha_flag}" \
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

log "OVERNIGHT SWEEP START (passive-observation re-run, post-reorg clean tree)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "design: 5 seeds (0-4) x 2 modes (cpu-async, nccl-async) x 2 guards (msf, trend) x 200 epochs"
log "        cpu-async cells use --easgd-alpha 0.5; nccl-async cells use default anchor"
log ""

# Seed-major ordering. Within each seed: nccl first (cheaper per epoch -> fail-fast
# signal early), then cpu-async, both guards each.
for seed in 0 1 2 3 4; do
    do_run "seed-${seed}-nccl-async-trend" "${seed}" nccl-async trend
    do_run "seed-${seed}-nccl-async-msf"   "${seed}" nccl-async msf
    do_run "seed-${seed}-cpu-async-trend"  "${seed}" cpu-async  trend
    do_run "seed-${seed}-cpu-async-msf"    "${seed}" cpu-async  msf
done

log ""
log "OVERNIGHT.SWEEP.DONE"
