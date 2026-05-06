#!/usr/bin/env bash
# Relaxed-anchor + EASGD smoke overnight
#
# Tree: ddp-scale (post analyzer scale labels + EASGD blend, uncommitted)
# Workload: resnet-graph (ResNet-20 / CIFAR-10), 200 epochs, 3-GPU heterogeneous
#
# Primary (10 runs): 5 seeds × {msf, trend} × nccl-async × --elche-relax-up
#   Tests the v2 doc's relaxed-anchor prediction: relaxing cadence
#   preserves convergence while saving syncs (spiral framing).
#   Anti-prediction: >0.3pp degradation.
#
# EASGD side (2 runs): seed 0 × {msf, trend} × cpu-async × --easgd-alpha 0.5
#   Tests Zhang/Choromanska/LeCun 2015 elastic blending vs current cpu-async
#   discard semantics. Single-seed smoke test, directional signal only.
#   Compare against existing 2026-05-04 seed-0 cpu-async-{msf,trend} as α=1.0
#   baseline (current overwrite behavior).
#
# Total: 12 runs ≈ 6 hours
#
# CRITICAL: bypass the fdl preset clobber via fdl cuda-shell. The preset
# injects --epochs 5 --seed 42 --output runs/ BEFORE forwarded args, which
# corrupts non-default invocations. cuda-shell calls the binary directly.
#
# To monitor progress:  tail -f ddp-bench/runs/overnight-2026-05-05-relaxed-easgd/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-05-relaxed-easgd"
RUNLOG="${BASE}/_runlog.txt"

# Ensure cwd is project root (fdl requires it). __dirname/../../.. lands here.
cd "$(dirname "$0")/../../.."

mkdir -p "${BASE}"
: > "${RUNLOG}"  # truncate

log() {
    printf "%s  %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

# Run one bench invocation via cuda-shell (bypasses the fdl preset clobber).
# Args: <run-name> <seed> <mode> <guard> <extra-flags>
do_run() {
    local name="$1" seed="$2" mode="$3" guard="$4" extra="$5"
    local out_dir="${BASE}/${name}"
    mkdir -p "${out_dir}"

    local guard_flag=""
    [[ "${guard}" == "msf" ]] && guard_flag="--guard msf"

    log "START  ${name}  seed=${seed} mode=${mode} guard=${guard} extra='${extra}'"
    local t0=$(date +%s)
    # cuda-shell cd's to /workspace then runs the binary directly. --output
    # path is relative to /workspace/ddp-bench (the binary's cwd inside the
    # container), so we strip the ddp-bench/ prefix from BASE.
    local out_in_container="${out_dir#ddp-bench/}/"
    fdl cuda-shell -- -c "cd /workspace/ddp-bench && ./target/release/ddp-bench \
        --model resnet-graph \
        --mode ${mode} \
        --gpus all \
        --epochs 200 \
        --per-epoch-eval \
        --seed ${seed} \
        --output ${out_in_container} \
        ${guard_flag} \
        ${extra}" \
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

log "OVERNIGHT SWEEP START (relaxed-anchor + EASGD smoke)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "uncommitted state: analyzer scale labels + EASGD blend"
log ""

# ---------------------------------------------------------------------------
# Primary: nccl-async × {msf, trend} × 5 seeds × --elche-relax-up
# ---------------------------------------------------------------------------
log "primary: relaxed-anchor on nccl-async, 5 seeds × 2 guards"

for seed in 0 1 2 3 4; do
    do_run "seed-${seed}-nccl-async-trend-relaxed" "${seed}" nccl-async trend "--elche-relax-up"
    do_run "seed-${seed}-nccl-async-msf-relaxed"   "${seed}" nccl-async msf   "--elche-relax-up"
done

# ---------------------------------------------------------------------------
# EASGD side: cpu-async × {msf, trend} × seed 0 × --easgd-alpha 0.5
# ---------------------------------------------------------------------------
log ""
log "EASGD smoke: cpu-async α=0.5, seed 0, 2 guards"

do_run "seed-0-cpu-async-trend-easgd05" 0 cpu-async trend "--easgd-alpha 0.5"
do_run "seed-0-cpu-async-msf-easgd05"   0 cpu-async msf   "--easgd-alpha 0.5"

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
