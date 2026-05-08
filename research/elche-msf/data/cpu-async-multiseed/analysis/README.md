# cpu-async-multiseed — analysis

8 cells: 4 seeds × 2 guards (`msf`, `trend`) on cpu-async × `--easgd-alpha 0.5`
× ResNet-20. Multi-seed Gate A confirmation pass — checks whether the
seed-0 single-shot smoke prediction generalizes.

Model / dataset / hardware as in `../README.md`.


## Cohort summary

| guard | n | eval mean | eval sd | range | sync mean | sync range |
|---|---:|---:|---:|---:|---:|---:|
| `msf`   | 4 | 91.61 % | 0.154 | 91.40–91.77 | 652 | 408–874 |
| `trend` | 4 | 91.84 % | 0.103 | 91.73–91.98 | 561 | 463–642 |

Per-seed eval (msf / trend): 91.40 / 91.98, 91.62 / 91.83, 91.65 / 91.83,
91.77 / 91.73. Both guards land tightly inside one pooled seed sd of
each other — the multi-seed mean is essentially flat against the
seed-0-driven design-doc predictions.

![Gate A α-prediction figure](gate_a_alpha_predictions.png)

Two panels: (left) eval per seed across both guards, with the seed-0
single-shot smoke star, the α=1.0 baseline bands, and cohort means as
dashed lines; (right) sync count per seed, exposing the seed-0 outlier
position (408 syncs at msf was the 2nd-lowest of any cell — well below
the multi-seed mean of 652).


## Sharp Gate A predictions vs reality

The launcher header set the following predictions from the seed-0 smoke:

| prediction | actual outcome | verdict |
|---|---|---|
| msf+α=0.5 cross-seed mean within ±0.15 pp of msf+α=1.0 baseline (91.86 %) | 91.61 % vs 91.86 % cited baseline → −0.25 pp; vs same-mode α=1.0 cohort 91.77 % (`../cpu-async-alpha-sweep/`) → −0.16 pp | **borderline** |
| sync reduction ≥ 25 % across all 4 seeds vs α=1.0 baseline | sync mean 652 vs cited baseline 882 → 26 % reduction (mean) but range 408–874 spans the baseline | **fails per-seed** |
| trend+α=0.5 degrades by 0.5–1.0 pp consistently | trend cohort mean 91.84 % vs cited trend baseline 91.96 % → only −0.12 pp degradation | **falsified** (no consistent degradation) |

Net read: the Gate A α=0.5 confirmation pass **does not** clear the
Pareto-improving threshold the design doc set for it. The msf result is
within one pooled seed sd of α=1.0 (and the α=1.0 baseline cited in the
launcher was nccl-async — mode-confounded; the in-mode α=1.0 baseline
from `../cpu-async-alpha-sweep/` is 91.77 ± 0.19, even closer). The
trend result outright falsifies the predicted α=0.5 degradation
direction. Strengthens the structural-scaling argument: the α knob has
no Pareto-improving direction at R-20 / 3-GPU.

The seed-0 smoke (msf 91.91 %, 408 syncs) sits on the favorable tail of
both axes; the multi-seed mean is flatter and slower.


## Per-rank heterogeneity (across all 8 cells)

| rank | GPU | mean util | peak VRAM | mean VRAM |
|---|---|---:|---:|---:|
| 0 | RTX 5060 Ti  | 100.0 % ± 0.0 | 358 MB | 357 MB |
| 1 | GTX 1060 (#1) | 99.9 % ± 0.4 | 396 MB | 389 MB |
| 2 | GTX 1060 (#2) | 99.8 % ± 0.5 | 390 MB | 385 MB |

cpu-async fully saturates all three GPUs — every rank runs near 100 %
compute utilization across all 8 cells. Profile is structurally
different from nccl-async (e.g. `../relaxed-anchor/analysis/README.md`
shows 22 % / 58 % / 61 % under the same model on nccl-async). Same
utilization profile observed in `../cpu-async-alpha-sweep/` (the
α-axis variant of this sweep).


## Key observations

- **Multi-seed mean falls below the seed-0-driven predictions on both
  axes.** Eval drops from 91.91 % (smoke) to 91.61 % (n=4 mean); sync
  count rises from 408 to 652. Sharp Gate A predictions for
  α=0.5 ≈ α=1.0 within ±0.15 pp at parity sync-cost reduction do not
  hold cleanly.
- **Trend cohort does not show the predicted α=0.5 degradation.** The
  −0.5 to −1.0 pp prediction (from the seed-0 trend smoke at 91.39 %)
  is falsified at n=4: the cohort mean is 91.84 %, only −0.12 pp from
  the cited trend α=1.0 baseline. The seed-0 trend smoke was a low
  outlier.
- **Sync-cost reduction is not consistent per-seed.** Mean reduction
  is 26 % vs the cited baseline, but per-seed range (408–874) spans
  the baseline value. The seed-0 408-sync number was the favorable
  tail, not the typical cost.
- **The originally-cited "α=1.0 baseline 91.86 %" was nccl-async.** The
  in-mode cpu-async α=1.0 cohort (`../cpu-async-alpha-sweep/`,
  measured cleanly 2026-05-08) sits at 91.77 ± 0.19. The Gate A
  cross-mode comparison overstated the contrast by ~0.10 pp.
- **cpu-async fully saturates 3 GPUs.** ~100 % utilization across all
  cells; the bottleneck on this rig is compute, not communication
  (matching the cpu-async path's design — averaging happens host-side
  between sync windows).


## Cross-day same-seed observation (cpu-async α=0.5 msf)

Three independent cohorts of the same recipe (cpu-async × msf × α=0.5
× R-20) ran on different days:

| sweep | date | n | eval mean (msf cohort) | sync mean |
|---|---|---:|---:|---:|
| this sweep (`cpu-async-multiseed/`) | 2026-05-06 | 4 | 91.61 % | 652 |
| `../passive-observation/` (cpu-async msf cells) | 2026-05-07 | 5 | 92.03 % | 613 |
| `../cpu-async-alpha-sweep/` α=0.5 cohort | 2026-05-08 | 4 | 91.89 % | 548 |

Per-seed eval at fixed RNG seed swings 0.55–0.87 pp across these days
(seed-2: 91.62 → 92.49 → 91.65, range 0.87 pp). Within-day cohort sd
is 0.15–0.31 pp — so the cross-day fixed-seed swing is ~3× the
within-day spread.

A commit-walk between the three trees the sweeps ran on (`0806f84`,
`4544408`, `54bcfe6`) ruled out code drift as the source. The only
training-path delta is a refactor of the ResNet builder that is
byte-equivalent at depth n = 3 (the default), with all new behavior
gated behind an unused `--depth-n` CLI flag. `flodl/`, `flodl-sys/`,
`Cargo.lock`, and `libtorch/.active` are untouched across the range.
The non-determinism is rig-level, not code-level.

The most plausible dominant source is **cadence-controller timing
variability** — sync-window boundaries on cpu-async are placed by a
controller that consults wall-time and overhead measurements, both of
which vary slightly across runs (warm-up state, thermal envelope,
host scheduling). Once sync points shift, the per-rank parameter
snapshots being averaged shift with them, and the trajectory
diverges. This is at a different scale than the micro-level CUDA /
NCCL kernel-ordering jitter both DDP modes share.

We do not yet have an nccl-async cross-day same-seed cohort at
matched recipe to compare (would require a small targeted rerun);
within-day cohort sd suggests nccl-async cohorts (0.22–0.32 sd) are
not tighter than cpu-async cohorts (0.10–0.31 sd) at this rig. The
observation is recorded here as an empirical fact about realization
noise on this rig + recipe; see manuscript §7.2 for the discussion-
level framing.


## Source data

- `per_cell.csv` — 8 rows: seed, guard, eval, syncs, sync_ms, GPU0/1/2
  utilization, VRAM peak/mean per rank.
- `per_rank.csv` — 24 rows = 8 cells × 3 ranks: GPU label, util, peak
  and mean VRAM per (cell, rank).
- `gate_a_alpha_predictions.png` — the figure embedded above.


## Reproducibility

Run from the parent directory:

```
python3 analyze.py
```

Reads the 8 cells in `..` and writes the three artifacts above to this
directory. See [`../README.md`](../README.md) for the sweep-level
launcher recipe and provenance.
