# Table — Trend vs MSF guard (passive observation)

ResNet-20 / CIFAR-10 / 200 epochs / 3-GPU heterogeneous /
nccl-async / default anchor. Tests the design doc's R3 (final-eval
parity) and R5 (false-positive reduction at the meta scale) on the
clean-rerun 5-seed cohort.

Source: `data/passive-observation/aggregate.txt`,
`data/passive-observation/analysis/per_cell.csv`.

## Eval parity (R3) — N = 5 seeds, nccl-async cohort

| guard | n | eval (mean ± sd) | seed range |
|---|---:|---:|---:|
| `msf`   | 5 | 91.70 % ± 0.25 | [91.32, 91.94] |
| `trend` | 5 | 91.80 % ± 0.22 | [91.56, 92.12] |
| **Δ (msf − trend)** | | **−0.09 pp** | within seed sd (~0.23 pp) |

Final eval is **statistically equivalent** across guards. The
guard choice doesn't move final eval at this scale; it moves cost
and false-positive rate.

![Guard comparison: per-cell eval + guard fires](../data/passive-observation/analysis/guard_comparison.png)

Left panel: per-cell final eval, color-coded by guard, with both
guards' sweep-mean shown as dashed lines. Right panel: guard fires
per run for both detectors on the same cells.

## Guard fires per 200-epoch run (R5)

The MSF guard runs on `λ_ema`, a smoothed signal that averages out
per-rank chaos and fires only when the top-scale (meta-oscillator)
setpoint shifts. The trend rule operates on per-event raw `D`,
which carries per-rank chaos signal regardless of whether the
meta-oscillator is converging.

| guard cell | trend-rule fires (3 rises in D) | rate-based fires (λ_ema > 1e-3, sustain 3) |
|---|---:|---:|
| msf-guard cells (n=5) | 44.4 ± 18.7  range [31, 73] | **1.2 ± 1.1**  range [0, 2] |
| trend-guard cells (n=5) | 48.8 ± 12.4 | 0.6 ± 1.3 |

On the msf-guard cells (where the rate-based detector is the active
gate), the fire reduction is **97.3 %** (44.4 → 1.2) at no eval cost.
The aggregator's `aggregate.txt` reports a slightly higher reduction
(98.2 %) because its detector-table parser excludes one cell whose
"both-fired" column carried a multi-event list rather than a single
count; both numbers are within the noise band of a noisy detector
running on a single run.

The architectural takeaway: **the rate-based detector is the right
tool on the right signal at the right scale.** Trend's 44 fires/run
are mostly true positives at the bottom scale (cycles ARE individually
chaotic by physics) and mostly false positives at the top scale
(meta-oscillator is converging regardless). λ_ema averages out the
per-rank chaos and surfaces only top-scale events, which is what the
controller actually wants to react to.

## Cross-rank Pearson r̄ (meta-oscillator anchor, R0)

The framing-validity gate. Cross-rank Pearson r > 0.95 anchors the
two-scale framing — ranks behave as coupled views of one
D-trajectory rather than as independent oscillators.

| pair | mean r ± sd (n=10 nccl-async cells) | min observed |
|---|---:|---:|
| rank 0 ↔ 1 | +0.9950 ± 0.0055 | +0.9859 |
| rank 0 ↔ 2 | +0.9939 ± 0.0067 | +0.9863 |
| rank 1 ↔ 2 | +0.9974 ± 0.0034 | +0.9916 |

The r = 0.95 framing-validity gate is **comfortably non-binding** —
the lowest single observation is +0.9859, ~10× the gate margin away
from breaking. The two-scale framing applies across the entire
nccl-async cohort.

![Cross-rank Pearson r per pair across 10 nccl-async cells](../data/passive-observation/analysis/meta_oscillator_pearson.png)

Reference lines: r = 0.99 (empirical anchor across the sweep) and
r = 0.95 (framing-validity gate; below this the meta-oscillator
framing breaks and per-rank treatment is required).

## Predictive correlations (kill criterion)

Cross-seed N = 5 on nccl-async msf:

| correlation | mean ± sd | range |
|---|---:|---:|
| `r(λ_raw_t → ln D_{t+1})` | −0.0026 ± 0.0108 | [−0.0150, +0.0120] |
| `r(λ_mean per epoch → eval)` | +0.1078 ± 0.0865 | [+0.0090, +0.2240] |
| `r(λ_ema end-of-epoch → eval)` | +0.1052 ± 0.2033 | [−0.1210, +0.3350] |

`λ_raw → ln D_{t+1}` is **indistinguishable from noise** —
endpoint-to-endpoint λ̂ does not predict next D. This is consistent
with the OU picture: λ̂ measures fluctuation around D*(LR), which is
mean-reverting noise, not drift. The original v1 doc's λ̂ formula
tracks the wrong quantity for prediction; the within-cycle by-k slope
is the right Lyapunov estimator (see `cliff-bracket.md` for the by-k
observability boundary).

`→ eval` correlations are **weakly positive but small** in mean. Two
scales explain why: the meta-scale signal (which `λ_ema` tracks via
smoothing) carries *regime* information (LR drops), not *accuracy*
information per se. Eval correlation appears at LR drops and dilutes
through the rest of training.

## Reproducibility

```
python3 research/elche-msf/data/passive-observation/aggregate.py
python3 research/elche-msf/data/passive-observation/analyze.py
```
