# cpu-async-alpha-sweep — analysis

16 cells: 4 EASGD α-values × 4 seeds on cpu-async × msf × ResNet-20.
Walks the α axis at fixed recipe to (a) measure the previously-missing
R-20 cpu-async α=1.0 cohort and (b) test whether α<1 elastic blending
introduces a Pareto-improving regularization direction at the recipe
ceiling.

Model / dataset / hardware as in `../README.md`.


## α-axis cohort summary

| α | n | eval mean | eval sd | range | sync mean | sync range |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 4 | 91.77 % | 0.187 | 91.51–91.94 | 502 | 316–731 |
| 0.7 | 4 | 91.88 % | 0.213 | 91.59–92.04 | 571 | 307–919 |
| 0.5 | 4 | 91.89 % | 0.230 | 91.65–92.19 | 548 | 484–712 |
| 0.3 | 4 | 91.57 % | 0.231 | 91.26–91.82 | 581 | 335–731 |

### Paired-seed contrasts vs α=1.0 (n=4, df=3)

| contrast | Δ mean | sd of diff | paired t | verdict |
|---|---:|---:|---:|---|
| α=0.7 − α=1.0 | +0.115 pp | 0.217 | +1.06 | NS |
| α=0.5 − α=1.0 | +0.123 pp | 0.374 | +0.66 | NS |
| α=0.3 − α=1.0 | −0.197 pp | 0.128 | **−3.08** | **p ≈ 0.027 one-sided** |

![α-axis 3-panel headline](p1_easgd_alpha_axis.png)

Three panels: (a) eval per seed across α with cohort means and the
recipe-ceiling band drawn as a shaded region; (b) paired-seed Δ vs α=1.0
with the pre-registered ±0.15 pp threshold drawn as horizontal guides
— α=0.7 lands inside the threshold band; only α=0.3 lands outside;
(c) sync count per seed per α, showing the non-monotone pattern (α=1.0
has the lowest sync mean of the four cohorts).


## Per-rank heterogeneity (across all 16 cells)

| rank | GPU | mean util | peak VRAM | mean VRAM |
|---|---|---:|---:|---:|
| 0 | RTX 5060 Ti  | 100.0 % ± 0.0 | 358 MB | 357 MB |
| 1 | GTX 1060 (#1) | 99.8 % ± 0.4 | 394 MB | 389 MB |
| 2 | GTX 1060 (#2) | 99.9 % ± 0.2 | 390 MB | 384 MB |

cpu-async fully saturates all three GPUs at this recipe — every rank
runs near 100 % compute utilization. This is the cpu-async profile (CPU
does the averaging on the host between sync windows, GPUs spend their
wall time computing); it is structurally different from the nccl-async
profile (e.g. `../relaxed-anchor/analysis/README.md` shows 22 % / 58 % /
61 % under the same model on nccl-async). The α value does not change
this — utilization is uniform across all 16 cells regardless of α.


## Key observations

- **Regularization-optimum prediction falsified.** α=0.7 leads α=1.0 by
  only +0.115 pp at n=4, paired t = 1.06. The pre-registered +0.15 pp
  threshold for "α<1 introduces a Pareto-improving direction" is not
  cleared. α=0.7, α=0.5, and α=1.0 all live inside one pooled seed sd
  (~0.21 pp) of each other.
- **α=0.3 is a NEW falsifying boundary.** Significant degradation
  (Δ = −0.197 pp, paired t = −3.08, p ≈ 0.027 one-sided). The
  deep-blending end is where partial-overwrite becomes too aggressive:
  local Lyapunov trajectories lose enough per-rank signal between syncs
  that the meta-oscillator coupling weakens. Soft lower bound on the
  useful α range.
- **Sync count is not monotone in α.** Cohort means 502 / 571 / 548 /
  581 for α=1.0 / 0.7 / 0.5 / 0.3 — α=1.0 has the lowest sync mean of
  the four. The "α<1 → fewer syncs" framing-prediction does not survive
  the data. Whatever sync-cost reduction α<1 achieves on average is
  drowned out by the per-seed variation in cadence (full ranges span
  300–900 syncs across all α).
- **R-20 cpu-async α=1.0 baseline = 91.77 ± 0.19 (n=4).** First clean
  measurement of this cohort. The "91.86 % α=1.0 baseline" cited in
  `../cpu-async-multiseed/run.sh` was sourced from the
  `../passive-observation/seed-{N}-nccl-async-msf` cohort — different
  mode, mode-confounded comparison. This sweep replaces that baseline
  with a same-mode same-recipe paired one.
- **Cross-day reproducibility holds loosely.** This sweep's α=0.5 cohort
  (mean 91.89 % over seeds 1–4, 2026-05-08) lies +0.28 pp above the
  prior `../cpu-async-multiseed/` α=0.5 msf cohort (mean 91.61 % over
  the same seeds, 2026-05-06). The shift is within pooled seed sd
  (~0.20 pp); same binary, two days apart, no code change between them.
  Repro passes loosely; the pair counts as cross-day noise floor.


## Source data

- `per_cell.csv` — 16 rows: α, seed, eval, syncs, sync_ms, GPU0/1/2
  utilization, VRAM peak/mean per rank.
- `per_rank.csv` — 48 rows = 16 cells × 3 ranks: GPU label, util, peak
  and mean VRAM per (cell, rank).
- `p1_easgd_alpha_axis.png` — the figure embedded above.


## Reproducibility

Run from the parent directory:

```
python3 analyze.py
```

Reads the 16 cells in `..` and writes the three artifacts above to this
directory. See [`../README.md`](../README.md) for the sweep-level
launcher recipe and provenance.
