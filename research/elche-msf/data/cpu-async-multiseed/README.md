# cpu-async multi-seed — paper-evidence extracts

## What this sweep tested

The 4-seed cpu-async cohort that confirms (or falsifies) the seed-0
single-shot probe of EASGD α=0.5 elastic blending as a coupling
mechanism. cpu-async is the only backend where α<1 is meaningful —
nccl-async uses in-place AllReduce and has no α knob. α=0.5 is now
the canonical default for cpu-async; raw cell names carried an
`-easgd05` suffix that was load-bearing pre-default but is no longer
informative, so the extract drops it.

| Axis | Values |
|---|---|
| seeds | 1, 2, 3, 4 |
| guard | `msf`, `trend` |
| mode | `cpu-async` (EASGD α=0.5) |
| epochs | 200 |
| model | `resnet-graph` (ResNet-20, n=3) |
| dataset | CIFAR-10 |
| hardware | 1× RTX 5060 Ti 16 GB + 2× GTX 1060 6 GB |

8 cells = 4 seeds × 2 guards.

## Relationship to `../passive-observation/`

Numerically and design-wise, this is the same cohort as
`../passive-observation/seed-{1..4}-cpu-async-{msf,trend}` (4 of the
5 seeds at α=0.5 cpu-async with both guards). The two sweeps differ
only in run history:

- This dir is the **2026-05-06 Gate A confirmation pass** of the
  seed-0 single-shot α=0.5 probe (8 cells, 4 seeds, half-overnight
  ~4h).
- `../passive-observation/` is the **2026-05-07 clean-rerun**
  (20 cells: 5 seeds × 2 modes × 2 guards) that supersedes this one
  for cross-config comparisons because it adds seed-0 and the
  matched-pair nccl-async cohort.

The clean-rerun numbers differ slightly cell-to-cell (RNG / scheduling
variance); both sweeps tell the same qualitative story.
[`../passive-observation/`](../passive-observation/) is the canonical
cpu-async α=0.5 cohort for the Pareto frontier figure. This dir
remains as the design-doc Gate-A reference (separate sweep, different
RNG, same conclusion).

## Files

```
.
├── README.md
├── run.sh                          original launcher (verbatim)
├── _runlog.txt                     sweep timing + START/OK/FAIL log
├── extract.py                      raw → committed extracts
├── analyze.py                      cohort + per-rank aggregator
├── analysis/
│   ├── README.md                   analysis walkthrough + key observations
│   ├── per_cell.csv                8 rows × 18 cols
│   ├── per_rank.csv                24 rows = 8 cells × 3 ranks
│   └── gate_a_alpha_predictions.png
└── seed-N-cpu-async-{msf|trend}/
    ├── timeline.csv.gz
    ├── training.log
    └── report.md
```

## Reproducibility + provenance

Same extract policy as `../passive-observation/README.md`. The
original raw output dir was
`ddp-bench/runs/overnight-2026-05-06-easgd-multiseed/` (gitignored).
Tree at staging: `0806f84` on `ddp-scale`. The bench at run time
exposed the `--easgd-alpha` flag (introduced post-2026-05-04), and
`run.sh` invokes it as `--easgd-alpha 0.5` for every cell.
