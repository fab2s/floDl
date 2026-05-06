# Relaxed-anchor — paper-evidence extracts

## What this sweep tested

The relaxed-anchor configuration (`--elche-relax-up`) on nccl-async
with both guards across 5 seeds. Compared head-to-head against the
default-anchor baseline in `../passive-observation/`.

| Axis | Values |
|---|---|
| seeds | 0, 1, 2, 3, 4 |
| guard | `msf`, `trend` |
| mode | `nccl-async` |
| anchor | `--elche-relax-up` (relaxed) |
| epochs | 200 |
| model | `resnet-graph` (ResNet-20, n=3) |
| dataset | CIFAR-10 |
| hardware | 1× RTX 5060 Ti 16 GB + 2× GTX 1060 6 GB |

10 cells total = 5 seeds × 2 guards.

## Files

```
.
├── README.md
├── run.sh                         original sweep launcher (verbatim)
├── _runlog.txt                    sweep timing log
├── extract.py                     raw → committed extracts
├── aggregate.py                   reads `passive-observation/` (default)
│                                  + `./` (relaxed) and produces the
│                                  4-cell comparison table
├── aggregate.txt                  canonical aggregator output
└── seed-N-nccl-async-{msf|trend}-relaxed/
    ├── timeline.csv.gz
    ├── training.log
    └── report.md
```

## Reproducibility + provenance

Same extract policy as `../passive-observation/README.md`: drop
`json/html/stdout`, gzip CSV, keep `training.log` + `report.md`
verbatim. Provenance is the flag-set + invariant claim, not a commit
hash.

`aggregate.py` reads `report.md` from both `passive-observation/`
(default-anchor cells, for the comparison baseline) and this
directory (relaxed cells), so both dirs must be extracted before the
aggregator runs.

The original raw output dir was
`ddp-bench/runs/overnight-2026-05-05-relaxed-easgd/` (gitignored). The
`-easgd` suffix on that dir name is historical — that overnight launch
also bundled the seed-0 EASGD α=0.5 smoke test for cpu-async, which
has since been migrated into `../passive-observation/` as the seed-0
cpu-async cells (post-EASGD α=0.5 became the canonical cpu-async
cohort once the multi-seed Gate A confirmation landed).
