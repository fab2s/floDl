# cpu-async EASGD α-axis sweep — paper-evidence extracts

## What this sweep tested

Walks the EASGD α axis at fixed cpu-async × msf × ResNet-20 to (a) fill
the previously-missing α=1.0 R-20 cpu-async cohort and (b) test whether
α<1 elastic blending introduces a Pareto-improving regularization
direction at the recipe ceiling.

Background: the prior cpu-async-multiseed sweep cited a "α=1.0 baseline
91.86 %" but that came from `passive-observation/seed-N-nccl-async-msf` —
mode-confounded. There was no R-20 cpu-async α=1.0 cohort on disk
anywhere. This sweep produces it (α=1.0 column) and walks 0.3 / 0.5 /
0.7 in the same self-contained run for cross-day reproducibility against
the prior α=0.5 cohort.

| Axis | Values |
|---|---|
| seeds | 1, 2, 3, 4 (matches cpu-async-multiseed for paired contrast) |
| guard | `msf` |
| mode | `cpu-async` |
| EASGD α | `0.3`, `0.5`, `0.7`, `1.0` |
| epochs | 200 |
| model | `resnet-graph` (ResNet-20, n=3) |
| dataset | CIFAR-10 |
| hardware | 1× RTX 5060 Ti 16 GB + 2× GTX 1060 6 GB |

16 cells = 4 α × 4 seeds.

## Sharp falsifiable predictions (pre-registered in launcher)

**Regularization-optimum (positive, P1 thesis):**
- α=0.7 mean ≥ α=0.5 mean by ≥ +0.15 pp at n=4 AND sync count within ±10 %
- α monotone in eval over {1.0, 0.7, 0.5, 0.3} or single-peaked at 0.7

**Null (Gate A behavior):**
- eval differences across α span ≤ 1× pooled seed sd (~0.25 pp)
- α=0.5 remains a defensible default; pick α purely on sync count

## Results

| α | n | eval mean | sd | range | per-seed (1, 2, 3, 4) | sync mean |
|---|---|---|---|---|---|---|
| 1.0 | 4 | **91.77 %** | 0.187 | 91.51–91.94 | 91.94 / 91.76 / 91.86 / 91.51 | 502 |
| 0.7 | 4 | **91.88 %** | 0.213 | 91.59–92.04 | 92.04 / 91.59 / 92.04 / 91.86 | 571 |
| 0.5 | 4 | **91.89 %** | 0.230 | 91.65–92.19 | 91.93 / 91.65 / 91.79 / 92.19 | 548 |
| 0.3 | 4 | **91.57 %** | 0.231 | 91.26–91.82 | 91.60 / 91.60 / 91.82 / 91.26 | 581 |

### Paired-seed contrasts vs α=1.0 (n=4, df=3)

| contrast | Δ mean | sd of diff | paired t | verdict |
|---|---|---|---|---|
| α=0.7 − α=1.0 | +0.115 pp | 0.217 | **1.06** | NS |
| α=0.5 − α=1.0 | +0.123 pp | 0.374 | **0.66** | NS |
| α=0.3 − α=1.0 | −0.197 pp | 0.128 | **−3.08** | **p≈0.027 one-sided** |

### Verdicts

1. **Regularization-optimum prediction FALSIFIED.** α=0.7 leads α=1.0 by
   only +0.115 pp at n=4, paired t=1.06 — well below the pre-registered
   +0.15 pp threshold. No clean α<1 dominance.
2. **α=0.5 vs α=1.0 parity confirmed at full n=4 paired.** Δ=+0.123 pp,
   t=0.66, NS. Gate A's "≈α=1.0 within seed sd" generalizes across the α
   axis from 0.5 to 1.0.
3. **α=0.3 is a NEW falsifying boundary.** Significant degradation
   (−0.197 pp, p<0.05). The deep-blending end is where partial-overwrite
   becomes too aggressive — local Lyapunov trajectories lose enough signal
   between syncs that the meta-oscillator coupling weakens.
4. **Sync counts are NOT monotone in α.** Cohort means 502 / 571 / 548 / 581 —
   α=1.0 actually has the lowest sync mean. The "α<1 → fewer syncs"
   framing-prediction does not survive.
5. **R-20 cpu-async α=1.0 baseline = 91.77 ± 0.19 (n=4).** First clean
   measurement; supersedes the previously-cited 91.86 % which was nccl-async.

### Cross-day reproducibility against `cpu-async-multiseed`

The prior `cpu-async-multiseed` α=0.5 msf cohort (2026-05-06, 4 seeds)
has actual mean 91.61 % (per `../cpu-async-multiseed/analyze.py`; the
launcher-comment value of 91.67 % was the pre-run prediction). This
sweep's α=0.5 cohort (2026-05-08, same 4 seeds) reports 91.89 %. The
+0.28 pp shift is within pooled seed noise (sd ≈ 0.20 pp); repro
passes loosely.

## Cell-name policy

Every cell carries an explicit `-alpha{03,05,07,10}` suffix. α IS the axis
under test, so the suffix is the differentiator (unlike
`cpu-async-multiseed` where α=0.5 is uniform across the cohort and the
suffix was dropped). `alpha10` matches the convention used in
`resnet56-cpu-async/seed-1-cpu-async-msf-alpha10`.

## Files

```
.
├── README.md
├── run.sh                                  original launcher (verbatim)
├── _runlog.txt                             sweep timing + START/OK/FAIL log
├── extract.py                              raw → committed extracts
├── analyze.py                              cohort + paired-contrast + per-rank aggregator
├── analysis/
│   ├── README.md                           analysis walkthrough + key observations
│   ├── per_cell.csv                        16 rows × 18 cols (eval / syncs / GPU util / VRAM)
│   ├── per_rank.csv                        48 rows = 16 cells × 3 ranks
│   └── p1_easgd_alpha_axis.png             3-panel headline figure
└── seed-{1..4}-cpu-async-msf-alpha{03,05,07,10}/    16 cells
    ├── timeline.csv.gz
    ├── training.log
    └── report.md
```

## Limitations

- n=4 per α is the minimum that distinguishes ~0.25 pp at 2σ given the
  R-20 / 3-GPU recipe-ceiling seed sd of 0.18–0.23 pp. Marginal for the
  null prediction; sufficient for the +0.15 pp regularization-optimum
  prediction (which falsified anyway).
- Single rig (1× RTX 5060 Ti + 2× GTX 1060). The α-axis read is
  recipe-ceiling-bound; whether α<1 introduces gains at recipes that lift
  the ceiling (AutoAugment / Cutout / Mixup) is out of scope.
- α=0.3 degradation is empirically solid at p<0.05 but n=4. A wider
  bracket (α=0.2, 0.4) would localize the boundary more precisely.

## Reproducibility + provenance

Same extract policy as `../resnet56-cpu-async/README.md`. Original raw
output dir: `ddp-bench/runs/overnight-2026-05-08-easgd-alpha-sweep/`
(gitignored).

Tree at staging: `54bcfe6` on `ddp-scale` (clean working tree).

### Post-hoc report regeneration

All 16 `report.md` files in this dir were regenerated via
`ddp-bench --report` analyze-mode after a template fix in
`ddp-bench/src/report.rs` made the GPU columns dynamic (the original
template hardcoded `GPU0 | GPU1` and silently dropped the third GPU's
utilization stats on this 3-GPU rig). The regenerated reports show
`GPU0 | GPU1 | GPU2`. Training data was not affected — `training.log`
and `timeline.csv` always retained the full per-rank record.

The same regeneration was applied to all sibling sweep dirs
(`cpu-async-multiseed`, `cliff-bracket`, `passive-observation`,
`relaxed-anchor`, `resnet56-cpu-async`).
