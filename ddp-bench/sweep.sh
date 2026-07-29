#!/bin/bash
# ddp-benchmark publish sweep — 3-GPU heterogeneous rig (2026-07-22).
# Replaces the retired run-missing.sh (which targeted a 2-GPU matrix and
# modes that no longer exist).
#
# Env knobs: SWEEP_OUT (default runs/publish-3gpu), PASCAL_DEFER
# (models whose pascal solos wait for a later pass; default
# "resnet-graph" — re-run with PASCAL_DEFER="" to pick them up),
# PASCAL_SKIP (models with no pascal solos at all; default "resnet"),
# PASCAL_BIN / PASCAL_LIB (remote binary + libtorch lib dir).
#
# Every cell also saves its self-contained portal to
# <cell>/dashboard.html (`--save-dashboard`). Timing-safe: the cluster
# path builds its dashboard sink unconditionally, so the flag adds only
# the run-scoped cards (built before the timer starts) and one file
# write at teardown — the published wall-times are unchanged. For the
# same reason `--reports-per-epoch` is deliberately NOT set here: it
# emits on the hot path at the reduce boundary, and a benchmark whose
# numbers get published must not carry monitoring overhead. Run a
# dedicated showcase cell with it when a richer portal is wanted.
# The archives follow the reader's `prefers-color-scheme`; pin one for
# publication by editing its baked `ARCHIVE_THEME` rather than re-running.
#
# Phase 1: 7 full-suite models x 6 DDP modes (incl. cpu-async-diloco) +
#          resnet parity cell, via `fdl @cluster` (sequential cells, one
#          fdl invocation per cell — the rendezvous-lifetime bug rules
#          out `--model all`).
# Phase 2a: solo-0 on exa (Blackwell, local docker path).
# Phase 2b: pascal solos LAST — devices 0/1 staged to pascal-LOCAL disk
#          (the /mnt/rdl mount is read-only there) and tar'd back as
#          solo-1/solo-2 (rig-global GPU numbering). Deliberately serial
#          after 2a (user call, 2026-07-29): the GTX cells are the
#          slowest (x1 riser), so trailing them means every cluster +
#          Blackwell result is complete and inspectable early while the
#          Pascals grind out the tail.
# Phase 3: report regen into the publish dir.
#
# Resume-safe: a cell whose training.log carries the `# total:` footer is
# skipped, so re-running this script continues where it left off.
# Hygiene: after a failed cluster cell, leftover ddp-bench processes on
# exa abort the sweep (a wedged launcher holds ports 1337/1339 and needs
# the user's kill — see TODO.md rig hygiene).
set -u
cd "$(dirname "$0")/.."
OUT=${SWEEP_OUT:-runs/publish-3gpu}
ABS_OUT=ddp-bench/$OUT
mkdir -p "$ABS_OUT"

MODELS="logistic:5 mlp:5 lenet:5 conv-ae:5 char-rnn:50 gpt-nano:50 resnet:200 resnet-graph:200"
DDP_MODES="nccl-sync nccl-cadence cpu-sync cpu-cadence cpu-async cpu-async-diloco"

# resnet vs resnet-graph are the same model through two engines (eager vs
# graph); running the full suite on both doubles the longest cells for an
# engine-parity claim two cells can make. resnet-graph carries the full
# suite; resnet keeps solo-0 (phase 2a, clean single-engine parity) plus
# one flagship cluster cell (parity under distribution).
modes_for() {
  case "$1" in
    resnet) echo "nccl-cadence" ;;
    *)      echo "$DDP_MODES" ;;
  esac
}

# Pseudo-modes expand to a base mode + extra flags; the harness suffixes
# the artifact dir to match (cpu-async + diloco -> <model>/cpu-async-diloco/).
mode_args() {
  case "$1" in
    cpu-async-diloco) echo "--mode cpu-async --outer-optimizer diloco" ;;
    *)                echo "--mode $1" ;;
  esac
}

ts() { date '+%F %T'; }
cell_done() { grep -q '# total:' "$ABS_OUT/$1/$2/training.log" 2>/dev/null; }
strays() { pgrep -af 'release/ddp-benc[h]' >/dev/null 2>&1; }

echo "$(ts) SWEEP BEGIN rev=$(git rev-parse --short HEAD) seed=42 out=$OUT"
echo "$(ts) models: $MODELS"
echo "$(ts) ddp modes: $DDP_MODES ; solos: exa solo-0, then pascal dev0/dev1 -> solo-1/solo-2 (serial, pascal last)"

# ── Phase 1 — cluster DDP cells ─────────────────────────────────────────
for entry in $MODELS; do
  m=${entry%%:*}; e=${entry##*:}
  for mode in $(modes_for "$m"); do
    if cell_done "$m" "$mode"; then echo "$(ts) SKIP $m/$mode (done)"; continue; fi
    echo "$(ts) START $m/$mode (epochs=$e)"
    # Per-cell capture (tee: stream into the sweep log AND keep the
    # cell's own copy for the verdict): a DEGRADED run (rank death
    # tolerated by elastic membership) exits 0 — correct for training,
    # invalid for a benchmark cell. Reject it and delete the cell dir
    # so a resume re-runs it.
    cell_log=$(mktemp)
    timeout 5400 ./fdl @cluster ddp-bench --model "$m" $(mode_args "$mode") --epochs "$e" --seed 42 --output "$OUT" --save-dashboard 2>&1 | tee "$cell_log"
    rc=${PIPESTATUS[0]}
    degraded=$(grep -c "finished DEGRADED\|child exit(s) tolerated\|device-side assert" "$cell_log")
    rm -f "$cell_log"
    if [ $rc -eq 0 ] && [ "$degraded" -eq 0 ] && cell_done "$m" "$mode"; then
      echo "$(ts) OK $m/$mode"
    else
      echo "$(ts) FAIL $m/$mode rc=$rc degraded=$degraded"
      rm -rf "${ABS_OUT:?}/$m/$mode"
      sleep 5
      if strays; then
        echo "$(ts) ABORT-DIRTY: leftover ddp-bench processes after failed cell — manual cleanup needed:"
        pgrep -af 'release/ddp-benc[h]'
        exit 1
      fi
    fi
  done
done
echo "$(ts) PHASE1 DONE"

# ── Phase 2a — exa solo-0 (Blackwell, local docker path) ────────────────
for entry in $MODELS; do
  m=${entry%%:*}; e=${entry##*:}
  if cell_done "$m" "solo-0"; then echo "$(ts) SKIP $m/solo-0 (done)"; continue; fi
  echo "$(ts) START $m/solo-0 (epochs=$e)"
  timeout 7200 ./fdl ddp-bench --model "$m" --mode solo-0 --epochs "$e" --seed 42 --output "$OUT" --save-dashboard
  rc=$?
  if [ $rc -eq 0 ] && cell_done "$m" "solo-0"; then
    echo "$(ts) OK $m/solo-0"
  else
    echo "$(ts) FAIL $m/solo-0 rc=$rc"
  fi
done
echo "$(ts) PHASE2A DONE"

# ── Phase 2b — pascal solos LAST (dev 0 -> solo-1, dev 1 -> solo-2) ─────
# The 200-epoch models take hours per GPU on the Pascals (x1 riser
# especially).
PASCAL_DEFER=${PASCAL_DEFER-"resnet-graph"}
# resnet pascal solos are permanently out (parity needs solo-0 only);
# resnet-graph pascal solos stay DEFERRED (hours on the x1 riser) — run a
# later pass with PASCAL_DEFER="" to pick them up.
PASCAL_SKIP=${PASCAL_SKIP-"resnet"}
PBIN=${PASCAL_BIN:-/mnt/rdl/target/cluster/flodl-pascal/builds-sm61-sm120/release/ddp-bench}
PLIB=${PASCAL_LIB:-/mnt/rdl/libtorch/builds/sm61-sm120/lib}
for entry in $MODELS; do
  m=${entry%%:*}; e=${entry##*:}
  case " $PASCAL_SKIP " in
    *" $m "*) echo "$(ts) SKIP $m pascal solos (parity model, solo-0 only)"; continue ;;
  esac
  case " $PASCAL_DEFER " in
    *" $m "*) echo "$(ts) DEFER $m/solo-1 + $m/solo-2 (pascal long-run deferred)"; continue ;;
  esac
  for pdev in 0 1; do
    pub=solo-$((pdev+1))
    if cell_done "$m" "$pub"; then echo "$(ts) SKIP $m/$pub (done)"; continue; fi
    echo "$(ts) START $m/$pub (pascal solo-$pdev, epochs=$e)"
    ssh -o BatchMode=yes flodl-pascal \
      "mkdir -p ~/solo-sweep && cd ~/solo-sweep && \
       LD_LIBRARY_PATH=$PLIB timeout 10800 $PBIN --model $m --mode solo-$pdev \
       --epochs $e --seed 42 --output runs --data-dir /mnt/rdl/ddp-bench/data \
       --save-dashboard"
    rc=$?
    if [ $rc -ne 0 ]; then echo "$(ts) FAIL $m/$pub rc=$rc"; continue; fi
    mkdir -p "$ABS_OUT/$m/$pub"
    ssh -o BatchMode=yes flodl-pascal "tar -C ~/solo-sweep/runs/$m/solo-$pdev -cf - ." \
      | tar -C "$ABS_OUT/$m/$pub" -xf -
    if cell_done "$m" "$pub"; then
      echo "$(ts) OK $m/$pub"
    else
      echo "$(ts) FAIL $m/$pub (no training.log after copy-back)"
    fi
  done
done
echo "$(ts) PHASE2 DONE"

# ── Phase 3 — report ────────────────────────────────────────────────────
timeout 300 ./fdl ddp-bench --report "$OUT/report.md" --output "$OUT" --charts resnet-graph
echo "$(ts) SWEEP COMPLETE"
