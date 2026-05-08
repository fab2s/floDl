# Table — Relaxed-anchor 4-cell head-to-head

ResNet-20 / CIFAR-10 / 200 epochs / 3-GPU heterogeneous /
nccl-async. Compares the default-anchor cohort (5 seeds × 2 guards
from `passive-observation/`) against the relaxed-anchor cohort
(`--elche-relax-up`, 5 seeds × 2 guards from `relaxed-anchor/`).

Source: `data/relaxed-anchor/aggregate.txt`,
`data/relaxed-anchor/analysis/per_cell.csv`,
`data/passive-observation/analysis/per_cell.csv`.

## Headline 4-cell table (n = 5 per cell)

| guard | anchor | eval (mean ± sd) | syncs | cross-rank r̄ (3-pair) | rate-based fires |
|---|---|---:|---:|---:|---:|
| `trend` | default | 91.80 % ± 0.22 | 676 ± 104 | 0.996 | 0.6 ± 1.3 |
| `trend` | relaxed | 91.64 % ± 0.32 | 402 ± 160 | 0.995 | 0.0 ± 0.0 |
| `msf` | default | 91.70 % ± 0.25 | 671 ± 242 | 0.995 | 1.2 ± 1.1 |
| **`msf`** | **relaxed** | **91.82 % ± 0.31** | **431 ± 178** | 0.991 | **0.0 ± 0.0** |

Relax-up flips the production default `nccl-async msf` cohort onto
the Pareto frontier: **+0.12 pp eval** at **−36 % syncs** vs the
default-anchor msf cohort. Same backend, same guard, single CLI flag
(`--elche-relax-up`).

![4-cell comparison (eval, syncs, Pearson r̄, rate-based fires)](../data/relaxed-anchor/analysis/four_cell_comparison.png)

Four panels: final eval, total syncs, mean cross-rank Pearson r̄,
and rate-based detector fires per run. Color encodes the anchor
mode (default vs relaxed). Error bars are seed-to-seed standard
deviation.

## Asymmetric Pearson decoupling under relax-up

Cross-rank Pearson r drop under `--elche-relax-up`, mean across all
5 seeds × 2 guards = 10 cells per anchor:

| pair | default mean ± sd | relaxed mean ± sd | Δ |
|---|---:|---:|---:|
| rank 0 ↔ 1 (fast ↔ slow) | +0.9950 ± 0.0043 | +0.9914 ± 0.0072 | **−0.0036** |
| rank 0 ↔ 2 (fast ↔ slow) | +0.9946 ± 0.0051 | +0.9912 ± 0.0060 | **−0.0034** |
| rank 1 ↔ 2 (slow ↔ slow) | +0.9980 ± 0.0024 | +0.9967 ± 0.0031 | −0.0013 |

![Asymmetric Pearson decoupling under relax-up](../data/relaxed-anchor/analysis/pearson_asymmetric.png)

Per-pair Pearson r̄ before vs after enabling `--elche-relax-up`. The
two fast↔slow pairs lose ~0.0035 of correlation; the slow↔slow pair
holds.

The decoupling is **asymmetric by hardware heterogeneity**:
fast↔slow pairs (rank 0 = RTX 5060 Ti against ranks 1, 2 = GTX 1060)
shed correlation under relax-up; slow↔slow pairs (the two GTX 1060s)
stay essentially perfectly locked. Mechanism: the fast GPU runs
farthest ahead between syncs, so it drifts most when cadence loosens;
the two slow GPUs are pinned together by their matched throughput.

This is a publishable empirical fact about heterogeneous-DDP
synchronization with **no homogeneous-cluster analog** — the
HetSeq / Cannikin framing (mixed-GPU clusters) is exactly the regime
where this matters.

## Critical mechanism — guard silence breaks relax-up

Per 200-epoch run:

| cell | trend-rule fires | msf-rule fires |
|---|---:|---:|
| msf default | 44.4 ± 18.7 | 1.2 ± 1.1 |
| msf relaxed | 64.4 ± 16.6 | **0.0 ± 0.0** |

**The msf guard never fires across all 5 relaxed-anchor seeds.**
ElChe's anchor relax-up policy grows cadence on every "Stable"
verdict; with msf silent, every verdict is Stable, anchor grows
unbounded, cycles get very long, cycles saturate to D*(LR) before
sync, R² collapses. trend-relaxed avoids this because trend keeps
firing (~64×/run), bounding the anchor growth.

**Implication for production:** a threshold-aware controller
(C5' in the design doc) that targets `μ · k*(LR)` directly is
required for safe deployment. Silence ≠ stability; the relax-up
policy currently composes badly with a quiet guard. v1's
"loose by default" intuition fails when the safety mechanism never
reports.

## Reproducibility

```
python3 research/elche-msf/data/relaxed-anchor/aggregate.py
python3 research/elche-msf/data/relaxed-anchor/analyze.py
```

Reads own cells plus `../passive-observation/` for the default-anchor
baseline; writes `aggregate.txt` + `analysis/{per_cell.csv,
per_rank.csv, *.png}`.
