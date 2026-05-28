#!/bin/bash
# Run all missing ddp-bench cases sequentially.
# gpt-nano/sync and resnet/sync are skipped (not Graph-based models).
set -e
cd "$(dirname "$0")/.."

echo "=== Starting missing runs at $(date) ==="

# resnet (200 epochs) - 5 missing modes
for mode in nccl-sync cpu-sync cpu-cadence cpu-async; do
  echo ""
  echo "=== resnet / $mode (200 epochs) at $(date) ==="
  fdl ddp-bench --model resnet --mode $mode --epochs 200
done

# resnet-graph (200 epochs) - 3 missing/incomplete modes
for mode in sync nccl-sync cpu-sync; do
  echo ""
  echo "=== resnet-graph / $mode (200 epochs) at $(date) ==="
  fdl ddp-bench --model resnet-graph --mode $mode --epochs 200
done

echo ""
echo "=== ALL DONE at $(date) ==="

# Regenerate report
fdl ddp-bench report
