#!/usr/bin/env bash
# Gate D — ResNet-56 bytes-axis confirmation (Batch 1, narrow)
#
# Purpose
# -------
# At ResNet-20 / 3-GPU, the Gate A multi-seed (overnight-2026-05-06-easgd-
# multiseed) found that EASGD α=0.5 does NOT add a Pareto-improving direction:
# msf+α=0.5 mean 91.67% ± 0.19 vs α=1.0 baseline 91.86% ± 0.27 (-0.19pp at
# marginal sync savings). The two-scale framing predicts the coupling-mechanism
# axis (α<1) becomes Pareto-relevant only when AllReduce cost is non-trivial
# relative to per-step compute — which ResNet-20 does not reach.
#
# Gate D walks the bytes axis: same dataset, same eval metric, same
# architecture family, just 3.1× the parameter count (n=9 → ResNet-56,
# ~850K params vs R-20's ~270K). Confirms-or-falsifies the directional
# structural prediction.
#
# Progressive design — Batch 1 (narrow, load-bearing only)
# --------------------------------------------------------
# The single contrast that answers the bytes-axis question is:
#   "does α=0.5 + msf beat α=1.0 + msf at 3× params?"
#
# Drops trend (Gate A showed trend+α=0.5 was worse at R-20, not load-bearing).
# Drops nccl-async (different axis — we're isolating the α effect on
# cpu-async where it actually applies; nccl-async is fast-path overwrite).
#
# Decision rule for Batch 2:
#   - α=0.5 still ≈ α=1.0 → bytes-axis null confirmed at R-56. Optionally
#     add 4 trend runs (1-4 × cpu-async × trend × α=0.5) to firm the
#     negative; otherwise call Gate D landed.
#   - α=0.5 starts dominating α=1.0 → expand to trend + maybe nccl-async
#     to characterize the rotation properly.
#
# Design: 4 seeds × 2 α-values × cpu-async × msf × ResNet-56 (n=9)
#   seeds: 1, 2, 3, 4 (matches Gate A seed range; seed 0 not run for
#                      direct cross-experiment comparability)
#   α ∈ {0.5, 1.0}
#   = 8 runs ≈ 12.5h (~93 min/run from 3-epoch smoke extrapolation:
#                     28s/epoch × 200 epochs = 5600s = 93 min)
#
# Sharp predictions to falsify
# ----------------------------
# Bytes-axis-rotation prediction (positive direction):
#   - msf+α=0.5 cross-seed mean ≥ msf+α=1.0 cross-seed mean within seed sd
#     AND sync count reduction ≥ 15% under α=0.5.
#
# Null prediction (Gate A R-20 behavior generalizes to R-56):
#   - msf+α=0.5 mean within ±0.3pp of msf+α=1.0 mean; sync count delta < 10%.
#
# Either outcome is informative for the paper.
#
# Tree state at staging
# ---------------------
# branch: ddp-scale, tree: 0806f84 + uncommitted --depth-n flag plumbing:
#   ddp-bench/src/main.rs           (--depth-n CLI flag)
#   ddp-bench/src/models/resnet_graph.rs (depth-flexible build_model + def())
# Smoke validated 2026-05-06: 3 epochs, n=9, cpu-async msf α=0.5, 3 GPU
# heterogeneous. VRAM 494 MB on cuda1 (plenty of headroom on 6GB 1060).
# Loss 2.41 → 2.11 → 1.98 across 3 epochs, train_acc 10.5% → 23.2%.
#
# To monitor:
#   tail -f ddp-bench/runs/overnight-2026-05-06-resnet56-easgd/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-06-resnet56-easgd"
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

    # α=1.0: full overwrite (matches default behavior; explicit for parity).
    # α=0.5: EASGD elastic blending.
    local alpha_flag="--easgd-alpha ${alpha}"

    log "START  ${name}  seed=${seed} alpha=${alpha}"
    local t0=$(date +%s)
    # cuda-shell cd's to /workspace then runs the binary directly. --output
    # path is relative to /workspace/ddp-bench (the binary's cwd inside the
    # container), so we strip the ddp-bench/ prefix from BASE.
    local out_in_container="${out_dir#ddp-bench/}/"
    fdl cuda-shell -- -c "cd /workspace/ddp-bench && ./target/release/ddp-bench \
        --model resnet-graph \
        --depth-n 9 \
        --mode cpu-async \
        --gpus all \
        --epochs 200 \
        --per-epoch-eval \
        --seed ${seed} \
        --output ${out_in_container} \
        --guard msf \
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

log "OVERNIGHT SWEEP START (Gate D Batch 1 — ResNet-56 bytes-axis confirmation)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "design: 4 seeds (1-4) × 2 α-values ({0.5, 1.0}) × cpu-async × msf × ResNet-56 (n=9) × 200 epochs"
log ""

# Outer loop on alpha so the load-bearing α=0.5 cells (test arm) come
# first — if hardware blows up partway through, we have α=0.5 data to
# interpret against the Gate A R-20 baseline, not just the α=1.0 control.
for alpha in 0.5 1.0; do
    # Output dir suffix: easgd05 (α=0.5) vs easgd10 (α=1.0).
    suffix=""
    case "${alpha}" in
        0.5) suffix="easgd05" ;;
        1.0) suffix="easgd10" ;;
    esac
    log "PHASE: alpha=${alpha}"
    for seed in 1 2 3 4; do
        do_run "seed-${seed}-cpu-async-msf-${suffix}" "${seed}" "${alpha}"
    done
done

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
