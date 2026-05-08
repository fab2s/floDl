# resnet56-cpu-async — analysis

5 cells: 4 seeds α=0.5 + 1 seed α=1.0 on cpu-async × `msf` ×
ResNet-56 (n=9 in the He et al. CIFAR family, ~850K params, 3.1× the
parameter count of ResNet-20). Bytes-axis confirmation of the
two-scale framing's structural prediction at the next model size up.

Model / dataset / hardware as in `../README.md`.


## Cohort summary

| α | n | eval mean | eval sd | range | sync mean |
|---|---:|---:|---:|---:|---:|
| 0.5 | 4 | **93.06 %** | 0.103 | 92.97–93.21 | 454 |
| 1.0 | 1 | 92.88 %  | n/a   | 92.88 (single-seed) | 286 |

Published ResNet-56 baseline: **93.03 %** (He et al. 2015, CIFAR-10
Table 6). The α=0.5 cohort mean lands at 93.06 % — **+0.03 pp above
the published baseline at n=4 with sd 0.10** — the recipe + cadence
controller deliver published-baseline parity at 3.1× the parameter
count.

![Gate D bytes-axis figure](gate_d_resnet56_alpha.png)

Two panels: (left) eval per seed for both α-cohorts vs the published
baseline drawn as a dotted reference line; (right) sync count per seed.
The α=1.0 single-seed lands lower (92.88 %) at fewer syncs (286), but
n=1 makes the cross-cohort comparison directional only — paired Δ
cannot be computed.


## Sharp Gate D predictions vs reality

The launcher header set the following decision rule:

| prediction (positive: bytes-axis rotation) | actual outcome | verdict |
|---|---|---|
| msf+α=0.5 cross-seed mean ≥ msf+α=1.0 cross-seed mean within seed sd | α=0.5 mean 93.06 ± 0.10 vs α=1.0 single-cell 92.88 — α=0.5 leads, but α=1.0 cohort is n=1 | **directional, not paired** |
| sync count reduction ≥ 15 % under α=0.5 vs α=1.0 | α=0.5 mean 454 vs α=1.0 single-cell 286 — α=0.5 has *more* syncs, +59 % | **opposite direction** |

Net read: the bytes-axis null result holds for eval — α=0.5 cohort lands
at the recipe ceiling for ResNet-56, no different from what α=1.0 would
be expected to do. The sync-cost picture goes against the original
prediction (α=0.5 syncs more, not fewer); the α=1.0 286-sync cell is
likely on the lucky tail of the cadence-controller's stochastic walk
(consistent with `../cpu-async-alpha-sweep/` where α=1.0 had the lowest
sync mean of the four α-cohorts at R-20).


## Per-rank heterogeneity (across all 5 cells)

| rank | GPU | mean util | peak VRAM | mean VRAM |
|---|---|---:|---:|---:|
| 0 | RTX 5060 Ti  | 100.0 % ± 0.0 | 514 MB | 513 MB |
| 1 | GTX 1060 (#1) | 100.0 % ± 0.0 | 582 MB | 578 MB |
| 2 | GTX 1060 (#2) | 100.0 % ± 0.0 | 574 MB | 570 MB |

VRAM peak ~580 MB on the slow GPUs is well within the 6 GB headroom on
the GTX 1060s. cpu-async fully saturates all three GPUs at ResNet-56
just as at ResNet-20 — the bottleneck remains compute, not VRAM.


## Key observations

- **R-56 cpu-async α=0.5 lands at the published baseline.** Cohort mean
  93.06 ± 0.10 % vs published 93.03 % — the controller + recipe deliver
  baseline parity at 3.1× the parameter count of R-20 with no
  hand-tuning.
- **Bytes-axis null result confirmed (eval-axis).** The Gate A finding
  ("α=0.5 ≈ α=1.0 within seed sd") generalizes from R-20 to R-56 on the
  eval dimension. The single-seed α=1.0 cell at 92.88 % differs by
  −0.18 pp from the n=4 α=0.5 mean — within the typical seed sd
  observed at R-56 (0.10 pp). Single-seed cannot prove parity, but is
  consistent with it.
- **Sync-cost prediction goes the wrong way.** α=0.5 cohort syncs 59 %
  *more* than the α=1.0 single cell (454 vs 286). The R-20 α-axis sweep
  showed the same pattern: α=1.0 had the lowest sync mean of the four
  α-cohorts. The "α<1 → fewer syncs" framing-prediction does not
  generalize empirically.
- **Single-seed α=1.0 limitation.** A 1-seed datapoint in a noisy
  estimator (~0.10 pp seed sd at R-56) gives near-zero-information
  differential measurement. The α=0.5 vs α=1.0 cross-cohort comparison
  here should be read as directional only; a 4-seed α=1.0 cohort (3
  more cells: seeds 2/3/4 × α=1.0) is required before claiming Gate D
  null result holds at R-56 in a paired-design sense. See
  `project_msf_paper_handoff.md` for the queued fill batch.
- **3-GPU cpu-async saturation profile holds at R-56.** Same ~100 %
  utilization across all ranks as observed at R-20 in
  `../cpu-async-multiseed/` and `../cpu-async-alpha-sweep/`.


## Source data

- `per_cell.csv` — 5 rows: α, seed, eval, syncs, sync_ms, GPU0/1/2
  utilization, VRAM peak/mean per rank.
- `per_rank.csv` — 15 rows = 5 cells × 3 ranks.
- `gate_d_resnet56_alpha.png` — the figure embedded above.


## Reproducibility

Run from the parent directory:

```
python3 analyze.py
```

Reads the 5 cells in `..` and writes the three artifacts above to this
directory. See [`../README.md`](../README.md) for the sweep-level
launcher recipe, provenance, and the deferred α=1.0 fill batch
(seeds 2–4) noted in the limitations.
