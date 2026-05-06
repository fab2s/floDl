# Passive observation — paper-evidence extracts

## What this sweep tested

The first multi-seed pass establishing the meta-oscillator framing
empirically. No controller intervention beyond default behavior;
observes natural dynamics under both backends and both guard choices.

| Axis | Values |
|---|---|
| seeds | 0, 1, 2, 3, 4 |
| guard | `msf`, `trend` |
| mode | `cpu-async`, `nccl-async` |
| epochs | 200 |
| model | `resnet-graph` (ResNet-20, n=3) |
| dataset | CIFAR-10 |
| hardware | 1× RTX 5060 Ti 16 GB + 2× GTX 1060 6 GB |

20 cells total = 5 seeds × 2 modes × 2 guards.

## Backend coverage

Both backends are MSF-framing-valid in this dataset:

- `nccl-async` cells use default-anchor with no elastic blending
  (impulsive AllReduce(Avg)). 10 of the 20 cells.
- `cpu-async` cells use **EASGD α=0.5** elastic blending. The
  3-phase Idle/Collecting/Computing pipeline pre-EASGD broke the
  impulsive-coupling assumption that anchors the meta-oscillator
  framing under α=1.0; the α=0.5 elastic blending restores it (R1
  by-k slope, cross-rank Pearson r ≥ 0.95, R1' per-rank gate at
  warmup all confirmed; see `aggregate.txt`). 10 of the 20 cells.


## Files

```
.
├── README.md                      this file
├── run.sh                         original sweep launcher (verbatim snapshot)
├── _runlog.txt                    sweep timing + START/OK/FAIL log
├── extract.py                     raw → committed extracts
└── seed-N-{cpu|nccl}-async-{msf|trend}/
    ├── timeline.csv.gz            system-resource trace
    ├── training.log               event log: per-epoch metrics, per-rank
    │                              shares, sync events, guard fires
    └── report.md                  flodl bench-generated per-cell report
```

`timeline.csv.gz` is gzipped (compressed level 9, ~12× ratio on numeric
trace). pandas reads transparently: `pd.read_csv("timeline.csv.gz")`.

## What was deliberately dropped

Same policy as the other sweeps: drop `timeline.json`,
`timeline.html`, `run.stdout.log`. Keep `training.log` + `report.md`
verbatim, gzip the timeline CSV.

## Reproducibility

This directory is the **canonical paper-evidence dataset** for the
passive-observation phase. Aggregators in
`research/elche-msf/figures/` and `research/elche-msf/tables/` read
from here directly.

To re-run on a different machine:

1. `fdl setup` (libtorch + dev image), then build the bench
   (`cargo build --release --features cuda` from `ddp-bench/`).
2. Run `bash run.sh` from the repo root. Default raw output goes to
   `ddp-bench/runs/overnight-2026-05-04/`; adjust the `BASE` variable
   inside `run.sh` to redirect.
3. `python3 extract.py [--raw-base /path/to/raw]` — overwrites the
   gzipped CSVs + logs + reports in this directory.

## Provenance

The bench binary at run time was built from flodl source between the
0.5.3 release and 0.5.4 (unreleased), with the following CLI surfaces
present and exercised by `run.sh`:
`--guard {msf,trend,none}`, `--per-epoch-eval`, `--mode`, `--gpus`,
`--seed`, `--epochs`, `--output`.

The arc's code changes were strictly additive — new CLI options and
the multi-guard strategy split — and **do not affect default
behavior**. A fresh build at any post-arc release that exposes these
flags should reproduce equivalent numerics from the recipe in
`run.sh`. The original raw output dir was at
`ddp-bench/runs/overnight-2026-05-04/` (gitignored).
