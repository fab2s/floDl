#!/usr/bin/env bash
# Cliff bracket — locate the synchronization-collapse cliff (200 epochs)
#
# A 50-epoch reconnaissance sweep showed flat eval across k ∈ {100..3200}.
# The k=100000 zero-sync smoke (1 final AllReduce, no within-training
# syncs) collapsed to 10.02% eval — random chance on 10-class CIFAR. So
# the threshold sits between syncs=4 (k=3200, 87.12% at 50ep) and syncs=1
# (k=100000, 10.02% at 50ep). This sweep brackets it densely at the FULL
# 200-epoch schedule
# (the LR-low phase may have stricter coupling needs than the 50ep cut
# could reveal).
#
# Cells (200 epochs ≈ 52,000 slow-rank batches; within-training syncs ≈
# 52000/k):
#   k =  3200  → ~16 within-training syncs (safe baseline)
#   k =  6400  → ~8  syncs                  (likely safe)
#   k = 12800  → ~4  syncs                  (50ep recon regime; 200ep test)
#   k = 16000  → ~3  syncs                  (entering bracket)
#   k = 25600  → ~2  syncs                  (likely past cliff)
#   k = 51200  → ~1  sync                   (cliff-confirmation; matches 50ep k=100000 collapse)
#
# 6 k values × 3 seeds × 200 epochs × nccl-async × `--guard none`
# × `--min-anchor=k --max-anchor=k` = 18 runs ≈ 10h overnight.
#
# Sharp falsifiable predictions:
#  - k=51200 should collapse to ~10% eval (matches 50ep k=100000 anchor)
#  - k=25600 likely collapses too (2 within-training syncs may not be
#    enough for replicas to find a common basin under full LR schedule)
#  - The cliff sits between two adjacent cells where eval drops by >50pp
#  - cross-rank Pearson r should drop dramatically across the cliff
#  - Side-question: does the eval-vs-k curve have a peak ABOVE ElChe's
#    default operating point (~200) but below the cliff? If so, the
#    "ride the limit but don't cross it" regime is real.
#
# Tree: 0806f84 on ddp-scale (post --min-anchor commit; clean state).
#
# To monitor:
#   tail -f ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff/_runlog.txt

set -u
BASE="ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff"
RUNLOG="${BASE}/_runlog.txt"

cd "$(dirname "$0")/../../.."

mkdir -p "${BASE}"
: > "${RUNLOG}"

log() {
    printf "%s  %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RUNLOG}"
}

# Args: <run-name> <seed> <k>
do_run() {
    local name="$1" seed="$2" k="$3"
    local out_dir="${BASE}/${name}"
    mkdir -p "${out_dir}"

    log "START  ${name}  seed=${seed} k=${k}"
    local t0=$(date +%s)
    local out_in_container="${out_dir#ddp-bench/}/"
    fdl cuda-shell -- -c "cd /workspace/ddp-bench && ./target/release/ddp-bench \
        --model resnet-graph \
        --mode nccl-async \
        --gpus all \
        --epochs 200 \
        --per-epoch-eval \
        --seed ${seed} \
        --output ${out_in_container} \
        --guard none \
        --min-anchor ${k} \
        --max-anchor ${k}" \
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

log "OVERNIGHT SWEEP START (cliff bracket — synchronization-collapse at 200ep)"
log "tree: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
log "design: 3 seeds × 6 k values (3200, 6400, 12800, 16000, 25600, 51200) × --guard none × pinned anchor × 200 epochs"
log ""

# Outer loop on k (increasing) so any catastrophic failure at high k
# doesn't take out the lower-k transition-region data first.
for k in 3200 6400 12800 16000 25600 51200; do
    log "k=${k}"
    for seed in 0 1 2; do
        do_run "seed-${seed}-fixed-k-${k}" "${seed}" "${k}"
    done
done

log ""
log "OVERNIGHT SWEEP DONE"
log "Summary: $(grep -c '^.* OK ' "${RUNLOG}") OK, $(grep -c '^.* FAIL ' "${RUNLOG}") FAIL"
