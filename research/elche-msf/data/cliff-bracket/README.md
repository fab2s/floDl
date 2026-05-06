# Cliff bracket (fixed-k probe) — paper-evidence extracts

## What this sweep tested

Pins cadence at exactly `k` batches per cycle via
`--min-anchor=k --max-anchor=k --guard none` (the only path that
bypasses the auto-tune's natural setpoint). Probes whether final eval
collapses past a synchronization-threshold value of `k`. Locates the
cliff between `k=16000` (last fully safe, all 3 seeds within 1.3pp
of safe-regime mean) and `k=25600` (first bimodal seed split,
within-cell range 35.1pp).

| Axis | Values |
|---|---|
| seeds | 0, 1, 2 |
| k (fixed) | 3200, 6400, 12800, 16000, 25600, 51200 |
| guard | `none` |
| mode | `nccl-async` |
| epochs | 200 |
| model | `resnet-graph` (ResNet-20, n=3) |
| dataset | CIFAR-10 |

18 cells total = 3 seeds × 6 k-values.

## Files

```
.
├── README.md
├── run.sh
├── _runlog.txt
├── extract.py
├── aggregate.py                   reads ./, produces the cliff bracket
│                                  table with bimodality flag
├── aggregate.txt                  canonical aggregator output
└── seed-N-fixed-k-K/
    ├── timeline.csv.gz
    ├── training.log
    └── report.md
```

## Reproducibility + provenance

Same extract policy as `../passive-observation/README.md`. Per-sweep reduction:
**184 MB → 4.1 MB**.

The aggregator surfaces per-seed evals + cell range + a bimodality
flag (`range > 30pp`); past-threshold cells with ≤1 within-training
sync skip OLS / Pearson silently rather than crashing on degenerate
input.

The original raw output dir was
`ddp-bench/runs/overnight-2026-05-05-sweep-b2-cliff/` (gitignored).
