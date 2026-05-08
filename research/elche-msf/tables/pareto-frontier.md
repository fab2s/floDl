# Table — Eval-vs-cost Pareto frontier

ResNet-20 / CIFAR-10 / 200 epochs / 3-GPU heterogeneous
(1× RTX 5060 Ti + 2× GTX 1060 6 GB). Held-out test eval. Cost axis is
AllReduce events per 200-epoch run (network-volume proxy at fixed
model size; rotates toward wall time as parameter count grows).

Source: `data/pareto-frontier/pareto.txt`, with per-cell evidence in
`data/{passive-observation,relaxed-anchor,cliff-bracket}/analysis/per_cell.csv`.

## Frontier across 12 configurations (48 cells)

| Config | n | eval (%) | syncs / 200 ep | wall (s) | Pareto status |
|---|---:|---:|---:|---:|---|
| `fixed k=51200` | 3 | 27.75 ± 30.72 | 1 ± 1 | 1705 ± 4 | **frontier** (collapsed) |
| `fixed k=25600` | 3 | 77.09 ± 18.71 | 2 ± 0 | 1706 ± 5 | **frontier** (bimodal cliff edge) |
| `fixed k=16000` | 3 | 90.05 ± 0.64 | 3 ± 0 | 1706 ± 4 | **frontier** (post-knee) |
| `fixed k=12800` | 3 | 91.29 ± 0.08 | 4 ± 1 | 1703 ± 9 | **frontier** |
| `fixed k=6400`  | 3 | 91.45 ± 0.17 | 7 ± 0 | 1705 ± 3 | **frontier** |
| `fixed k=3200`  | 3 | 91.60 ± 0.28 | 13 ± 1 | 1712 ± 9 | **frontier** (knee) |
| `nccl-async relaxed trend` | 5 | 91.64 ± 0.32 | 402 ± 160 | 1863 ± 20 | **frontier** (lowest-sync near-parity) |
| `nccl-async relaxed msf`   | 5 | 91.82 ± 0.31 | 431 ± 178 | 1857 ± 14 | **frontier** (mid) |
| `cpu-async default trend`  | 5 | 91.70 ± 0.10 | 468 ± 149 | 1890 ± 46 | dominated by nccl-async relaxed msf |
| `cpu-async default msf` (EASGD α=0.5) | 5 | **92.03 ± 0.31** | 613 ± 128 | 1917 ± 23 | **frontier** (eval max) |
| `nccl-async default msf`   | 5 | 91.70 ± 0.25 | 671 ± 242 | 1891 ± 84 | dominated by nccl-async relaxed msf |
| `nccl-async default trend` | 5 | 91.80 ± 0.22 | 676 ± 104 | 1902 ± 92 | dominated by nccl-async relaxed msf |

Sorted by mean sync count, ascending. mean ± standard deviation
across seeds. A configuration is dominated when another is strictly
better (or equal) on eval at strictly lower sync count.

![Pareto frontier — full view (eval vs syncs/200ep, log-x)](../data/pareto-frontier/pareto.png)

Full Pareto plot. Fixed-k cells span the low-sync end (1–13 syncs);
auto-tuned cells cluster in the high-sync band (~400–900 syncs).
Past the cliff (k ≥ 25600), fixed-k cells collapse far below
random-chance accuracy.

![Frontier-ranked bar pairs (eval + syncs, sorted by sync count)](../data/pareto-frontier/analysis/frontier_ranked.png)

Direct table-companion: each row of the table above corresponds to a
horizontal bar pair. Frontier configurations in green; dominated in grey.

## High-sync end of the frontier (production-relevant)

Three non-dominated configurations; the production default is not
one of them.

| Frontier point | eval | syncs | one-line summary |
|---|---:|---:|---|
| `cpu-async default msf` (EASGD α=0.5) | **92.03 ± 0.31** | 613 ± 128 | eval maximum; coupling-mechanism axis (EASGD elastic blending, α<1) |
| `nccl-async relaxed msf` | 91.82 ± 0.31 | 431 ± 178 | mid-frontier; flips production default `nccl-async default msf` to frontier with a single flag (`--elche-relax-up`) |
| `nccl-async relaxed trend` | 91.64 ± 0.32 | 402 ± 160 | lowest-sync near-parity; trades −0.18 pp eval for the lowest-sync auto-tuned config |

Production default `nccl-async default msf` (91.70 ± 0.25 / 671 ± 242)
is **dominated by `nccl-async relaxed msf`**: Δ +0.12 pp eval at
−36 % sync count. The Pareto improvement against the current
production default is a single CLI flag.

## Cadence-axis frontier (fixed-k cliff probe)

The fixed-k cells dominate the low-sync end three orders of magnitude
below the auto-tuned regime. Knee is at k ≈ 3200–6400 (eval
91.45–91.60 %); the bend is sharp past k = 12800.

![Safe-regime zoom (high-sync band)](../data/pareto-frontier/pareto-safe-zoom.png)

Safe-regime zoom — the auto-tune cluster in the high-sync band, with
the frontier knee visible at the low-sync end of the safe regime.

| k | eval | syncs | regime |
|---:|---:|---:|---|
| 3200 | 91.60 ± 0.28 | 13 ± 1 | safe baseline (frontier knee) |
| 6400 | 91.45 ± 0.17 | 7 ± 0 | safe |
| 12800 | 91.29 ± 0.08 | 4 ± 1 | safe |
| 16000 | 90.05 ± 0.64 | 3 ± 0 | post-knee (soft drop, all 3 seeds within 1.3 pp) |
| 25600 | 77.09 ± 18.71 | 2 ± 0 | bimodal cliff edge (range 35.1 pp) |
| 51200 | 27.75 ± 30.72 | 1 ± 1 | past cliff (2 of 3 seeds at random chance) |

ElChe's auto-tuned cadence saturates at k ≈ 200 in this regime;
operating ~80–125× below the synchronization threshold.

## Verdict

- The eval-vs-cost frontier resolves to **9 non-dominated configurations
  out of 12**.
- The production default (`nccl-async default msf`) is **dominated** by
  `nccl-async relaxed msf` — single-flag improvement.
- The eval maximum (`cpu-async default msf`, EASGD α=0.5) sits on the
  frontier but the +0.21 pp margin to `nccl-async relaxed msf` is
  within seed-noise sd (0.31 pp); coupling-mechanism axis is **not a
  clearly differentiable Pareto direction at this scale** — see the
  design doc Gate A null-result discussion for prediction-vs-outcome
  detail.
- Wall-time is essentially flat at this rig and model size (≤ 215 s
  spread across all auto-tuned configs out of ~1900 s total): the
  frontier rotation onto wall-time requires a model where AllReduce
  cost is non-trivial relative to per-step compute (see manuscript
  §6/§7 for the bytes-axis ResNet-56 prediction).

## Reproducibility

```
python3 research/elche-msf/data/pareto-frontier/pareto.py
```

Reads `report.md` from the four upstream sweep extracts and
regenerates `pareto.txt` + `pareto.png` + `pareto-safe-zoom.png`.
