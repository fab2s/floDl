# ResNet-56 cpu-async — paper-evidence extracts

## What this sweep tested

Bytes-axis confirmation of the two-scale framing's structural
prediction: at 3.1× ResNet-20's parameter count (~850K params,
n=9 in the He et al. CIFAR family), does EASGD α<1 elastic blending
become a Pareto-improving coupling mechanism, or does the
ResNet-20 / 3-GPU null result generalize?

Decision rule (from the launcher header):
- α=0.5 ≈ α=1.0 → bytes-axis null confirmed at R-56.
- α=0.5 dominates α=1.0 → expand to trend + maybe nccl-async.

| Axis | Values |
|---|---|
| seeds | 1, 2, 3, 4 (α=0.5); 1 only (α=1.0, single-seed) |
| guard | `msf` |
| mode | `cpu-async` |
| EASGD α | `0.5`, `1.0` |
| epochs | 200 |
| model | `resnet-graph` with `--depth-n 9` (ResNet-56) |
| dataset | CIFAR-10 |
| hardware | 1× RTX 5060 Ti 16 GB + 2× GTX 1060 6 GB |

5 cells: 4 seeds × α=0.5 + 1 seed × α=1.0 (the α=1.0 cohort is
incomplete — sweep launcher was killed after seed-1 α=1.0 finished;
seeds 2/3/4 α=1.0 not run).

## Cell-name policy

- `-easgd05` suffix is dropped from raw cell names (α=0.5 is
  canonical cpu-async; uniform across the cohort would not
  differentiate).
- `-alpha10` suffix is preserved on the α=1.0 cell because that IS
  the differentiator this sweep probes.

So the four α=0.5 cells become `seed-{N}-cpu-async-msf` and the one
α=1.0 cell becomes `seed-1-cpu-async-msf-alpha10`.

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
│   ├── per_cell.csv                5 rows × 18 cols
│   ├── per_rank.csv                15 rows = 5 cells × 3 ranks
│   └── gate_d_resnet56_alpha.png
├── seed-{1..4}-cpu-async-msf/      α=0.5 cohort (canonical)
│   ├── timeline.csv.gz
│   ├── training.log
│   └── report.md
└── seed-1-cpu-async-msf-alpha10/   α=1.0 single-seed baseline
    ├── timeline.csv.gz
    ├── training.log
    └── report.md
```

## Limitations

The α=1.0 cohort is **single-seed** (n=1). Per single-seed
differential-claims policy: a 1-seed datapoint in a noisy estimator
(~0.2pp seed sd at this rig and dataset) gives near-zero-information
differential measurement. The α=0.5 vs α=1.0 cross-cohort comparison
should be read as directional only; a 4-seed α=1.0 cohort is required
before claiming Gate D's null result holds at R-56. This dir
preserves the asymmetric data as-is so a follow-up batch can fill in
seeds 2–4 α=1.0 without rerunning what already succeeded.

## Reproducibility + provenance

Same extract policy as `../passive-observation/README.md`. Original
raw output dir: `ddp-bench/runs/overnight-2026-05-06-resnet56-easgd/`
(gitignored). Tree at staging: `0806f84` on `ddp-scale` plus
uncommitted `--depth-n` flag plumbing in `ddp-bench/src/main.rs` +
`ddp-bench/src/models/resnet_graph.rs` (depth-flexible build_model).

### Post-hoc report regeneration

The sweep launcher bash was killed mid-α=1.0-phase. As a side effect,
4 of the 5 cells (`seed-{1..4}-cpu-async-msf-easgd05`) and 1 cell
(`seed-1-cpu-async-msf-easgd10`) were missing their `report.md` —
the bench `--report` write happens after training completes and was
interrupted. Reports were regenerated from the saved `timeline.csv`
via `ddp-bench --report` analyze-mode (does not rerun training; only
re-aggregates the timeline + parses training.log). This is functionally
equivalent to the in-line bench-generated report and uses the same
formatter.
