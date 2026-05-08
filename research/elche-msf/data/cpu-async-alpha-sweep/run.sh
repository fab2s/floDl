#!/usr/bin/env bash
# P1 — EASGD α-axis sweep (cpu-async × msf × ResNet-20)
#
# Purpose
# -------
# The msf paper's Gate A multi-seed established α=0.5 + msf at 91.67% ± 0.19
# (cpu-async-multiseed/, n=4 seeds 1-4). The "α=1.0 baseline" referenced in
# that sweep's run.sh (91.86%) was actually the passive-observation nccl-async
# msf cohort — mode-confounded. There is no R-20 cpu-async α=1.0 cohort on
# disk anywhere.
#
# This sweep walks the α axis at fixed cpu-async × msf × ResNet-20:
#   α ∈ {0.3, 0.5, 0.7, 1.0} × 4 seeds = 16 cells × ~33 min ≈ 8.8h
#
# Two questions get answered cleanly:
#   1. (gap fill) cpu-async α=1.0 R-20 cohort — was missing from the paper.
#   2. (P1 thesis) does α<1 acting as elastic-blending regularization show a
#      Pareto-improving optimum away from α=0.5? User framing 2026-05-08:
#      "EASGD α<1 IS regularization — partial blending preserves per-rank
#      Lyapunov signal between syncs". α=0.7 is the leading candidate to
#      dominate α=0.5; α=0.3 probes the deep-blending end.
#
# Sharp predictions to falsify
# ----------------------------
# Regularization-optimum (positive, P1 thesis):
#   - α=0.7 mean ≥ α=0.5 mean by ≥ +0.15 pp at n=4 AND sync count within ±10 %
#   - α monotone in eval over {1.0, 0.7, 0.5, 0.3} or single-peaked at 0.7
#
# Null (Gate A behavior):
#   - eval differences across α span ≤ 1× pooled seed sd (~0.25 pp); α=0.5
#     remains a defensible default; pick α purely on sync count.
#
# Either outcome is informative. The eval-ceiling read at R-20 (recipe-bound
# at 91.6–92.0 %, seed sd 0.20–0.31 pp) means n=4 distinguishes ~0.25 pp at
# 2σ — sufficient for the regularization-optimum prediction, marginal for the
# null.
#
# Loop ordering: α-major, seed-minor
# ----------------------------------
# Outer loop on α with order [1.0, 0.7, 0.5, 0.3]. Load-bearing cohorts
# complete first:
#   - after ~2.2h: α=1.0 cohort done — fills the missing gap.
#   - after ~4.4h: α=0.7 added — P1 regularization-optimum read.
#   - after ~6.6h: α=0.5 added — reproducibility cross-check vs
#                  cpu-async-multiseed (different day, same code).
#   - after ~8.8h: α=0.3 added — full axis.
# Partial completion still yields the most decision-relevant data first.
#
# Seeds 1-4 match cpu-async-multiseed cohort exactly so the α=0.5 cells in
# this sweep are paired against the existing α=0.5 cells (cross-day repro
# check). Seed 0 not run (no analog in cpu-async-multiseed).
#
# Tree at staging
# ---------------
# branch: ddp-scale, tip: 54bcfe6 ("refine"). Working tree clean.
#
# To monitor:
#   tail -f ddp-bench/runs/overnight-2026-05-08-easgd-alpha-sweep/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-08-easgd-alpha-sweep"
RUNLOG="${BASE}/_runlog.txt"

cd "$(dirname "$0")/../../../.."

mkdir -p "${BASE}"
: > "${RUNLOG}"

log() {
    printf "%s  %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

# Run one bench invocation via cuda-shell (bypasses the fdl preset clobber).
# Args: <run-name> <seed> <alpha>
do_run() {
    local name="$1" seed="$2" alpha="$3"
    local out_dir="${BASE}/${name}"
    mkdir -p "${out_dir}"

    log "START  ${name}  seed=${seed} alpha=${alpha}"
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
        --guard msf \
        --easgd-alpha ${alpha}" \
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

# α value → directory suffix (no decimal point in dir names).
alpha_suffix() {
    case "$1" in
        0.3) echo "alpha03" ;;
        0.5) echo "alpha05" ;;
        0.7) echo "alpha07" ;;
        1.0) echo "alpha10" ;;
        *)   echo "alpha-unknown" ;;
    esac
}

log "OVERNIGHT SWEEP START (P1 — EASGD α-axis sweep, cpu-async × msf × ResNet-20)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "design: 4 α-values [1.0, 0.7, 0.5, 0.3] × 4 seeds [1-4] × cpu-async × msf × ResNet-20 × 200 epochs"
log "        α-major outer loop; load-bearing α=1.0 baseline completes first"
log ""

# Outer α order: 1.0 → 0.7 → 0.5 → 0.3 (most decision-relevant first).
for alpha in 1.0 0.7 0.5 0.3; do
    suffix=$(alpha_suffix "${alpha}")
    log "PHASE: alpha=${alpha} (suffix=${suffix})"
    for seed in 1 2 3 4; do
        do_run "seed-${seed}-cpu-async-msf-${suffix}" "${seed}" "${alpha}"
    done
done

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
