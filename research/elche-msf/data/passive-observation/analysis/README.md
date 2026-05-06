# passive-observation — analysis

20 cells: 5 seeds (0–4) × 2 modes (`cpu-async`, `nccl-async`) × 2
guards (`msf`, `trend`), on ResNet-20 / CIFAR-10 / 200 epochs.
Hardware: 1× RTX 5060 Ti (16 GB) + 2× GTX 1060 (6 GB each). Default
anchor; no relax-up; no EASGD blending.

The `cpu-async` cohort (10 cells) is retained as a foil rather than
primary data: the 3-phase pipelined averaging on this backend breaks
the impulsive-coupling assumption that anchors the meta-oscillator
framing. The `nccl-async` cohort (10 cells) is the primary
subset for the verdicts on `R3` (final eval), `R5` (guard
fires), and the meta-oscillator anchor (cross-rank Pearson r).


## Meta unified view

### Summary (nccl-async, guard comparison)

| guard | n cells | mean eval | mean syncs | mean fires (3-rises) | mean fires (rate-based) |
|---|---:|---:|---:|---:|---:|
| `msf`   | 5 | 91.83% ± 0.20 pp | 882 ± 299 | 55.4 ± 7.1 | 2.2 ± 2.9 |
| `trend` | 5 | 91.71% ± 0.21 pp | 539 ± 205 | 40.6 ± 9.7 | 0.8 ± 1.1 |

### Eval and guard fires per cell

![Guard comparison](guard_comparison.png)

Left panel: per-cell final eval, color-coded by guard, with both
guards' sweep-mean shown as dashed lines. Right panel: guard fires per
run for both detectors on the same cells.

### Cross-rank Pearson r (meta-oscillator anchor)

| pair | mean r ± sd | min | max |
|---|---:|---:|---:|
| rank 0 ↔ 1 | +0.9923 ± 0.0077 | +0.9728 | +0.9988 |
| rank 0 ↔ 2 | +0.9943 ± 0.0037 | +0.9898 | +0.9990 |
| rank 1 ↔ 2 | +0.9949 ± 0.0081 | +0.9726 | +0.9992 |

![Meta-oscillator anchor](meta_oscillator_pearson.png)

Cross-rank Pearson r per pair across all 10 nccl-async cells.
Reference lines: r = 0.99 (empirical anchor across the sweep) and
r = 0.95 (framing-validity gate; below this the meta-oscillator
framing breaks and per-rank treatment is required).

## Individual GPU view

### Per-rank averages (across all 20 cells)

| rank | GPU | mean share | mean throughput (samples/ms) | mean util | peak VRAM |
|---|---|---:|---:|---:|---:|
| 0 | RTX 5060 Ti | 0.401 ± 0.002 | 2.63 ± 0.16 | 21.4% | 357 MB |
| 1 | GTX 1060 (#1) | 0.304 ± 0.002 | 3.54 ± 0.22 | 56.5% | 396 MB |
| 2 | GTX 1060 (#2) | 0.295 ± 0.001 | 3.47 ± 0.24 | 59.1% | 390 MB |

Mean ± standard deviation across cells. Peak VRAM is the maximum
allocated by libtorch over the run, sampled at ~100 ms intervals from
`timeline.csv.gz`.

### Per-rank heterogeneity per cell

![Per-rank heterogeneity](per_rank_heterogeneity.png)

Three panels (share, throughput, GPU utilization). Within each panel,
every cell has three adjacent bars (one per rank). Color encodes the
GPU. Cell labels: `s{seed}-{c|n}-{m|t}` where `c` = cpu-async,
`n` = nccl-async, `m` = msf, `t` = trend.

## Key observations

- **Cross-rank Pearson r anchors the meta-oscillator framing**. Across
  all 10 `nccl-async` cells, every rank pair stays at
  r > 0.9726 (mean r ≥ 0.9923 on the lowest-mean
  pair). The framing-validity gate at r = 0.95 is comfortably
  non-binding — ranks behave as coupled views of one D-trajectory,
  not as independent oscillators.
- **The rate-based detector fires ~25× less
  than the 3-rises rule at no eval cost**. On the
  5 msf-guard cells (where the rate-based detector is
  the active gate), mean fires per 200-epoch run:
  55.4 (3-rises rule) vs
  2.2 (rate-based). Final eval is
  statistically equivalent across guards (Δ = +0.11 pp,
  within seed sd ≈ 0.20 pp). The aggregator output in
  [`aggregate.txt`](aggregate.txt) reports a slightly higher reduction
  ratio (98.2%) because its detector-table parser excludes one cell
  whose "both-fired" column carried a multi-event list rather than a
  single count.
- **Heterogeneous load balancing visible**. The fast GPU (rank 0,
  RTX 5060 Ti) carries a mean batch share of
  0.40 but runs at
  21% mean
  GPU utilization; the slow GPUs (ranks 1, 2, GTX 1060) take
  0.30 /
0.30 and run at
  57% /
59% utilization.


## Source data

- `per_cell.csv` — one row per cell (20 rows): includes mode, guard,
  eval, syncs, guard-fire counts, cross-rank Pearson r, per-rank
  shares + throughputs + GPU utilization + VRAM peaks.
- `per_rank.csv` — one row per (cell, rank), 60 rows total.
- `guard_comparison.png`, `meta_oscillator_pearson.png`,
  `per_rank_heterogeneity.png` — the figures embedded above.

## Reproducibility

Run from the sweep directory: `python3 analyze.py`. Reads the cell
extracts in this directory; writes outputs to `analysis/`. See
[`../README.md`](../README.md) for the sweep-level reproducibility
recipe.
