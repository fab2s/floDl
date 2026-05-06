# DDP Benchmark Report

## Hardware

- gpu0: NVIDIA GeForce RTX 5060 Ti (15GB, sm_120)
- gpu1: NVIDIA GeForce GTX 1060 6GB (5GB, sm_61)
- gpu2: NVIDIA GeForce GTX 1060 6GB (5GB, sm_61)

- **Models**: 1
- **Modes**: 1

## Notes

DDP modes are expected to show slightly lower eval than solo on small models with few epochs. Distributed training converges slower in early epochs due to gradient averaging across devices with different data views, and ElChe (cadence/async) modes need calibration time to find the optimal sync interval, which further penalizes short runs. On longer training (200 epochs), every DDP mode surpasses solo convergence while completing faster -- the whole point of multi-GPU training.

## Incomplete Runs

- resnet-graph/solo-0
- resnet-graph/solo-1
- resnet-graph/solo-2
- resnet-graph/nccl-sync
- resnet-graph/nccl-cadence
- resnet-graph/cpu-sync
- resnet-graph/cpu-cadence
- resnet-graph/cpu-async

## Per-Model Results

GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|----------|
| nccl-async | 0.061840 | 0.9177 | +0.0052 | 1829.6 | 307 | 42.0 | 100% | 100% | 4.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | - | - | - | - |

## Eval Quality (vs solo-0)

| Model | nccl-async |
|-------|------------|
| resnet-graph | - |

## Convergence Quality (loss ratio vs solo-0)

| Model | nccl-async |
|-------|------------|
| resnet-graph | - |

## Per-Epoch Loss Trajectory

### resnet-graph (sampled, 200 epochs)

| Mode | E0 | E10 | E21 | E31 | E42 | E52 | E63 | E73 | E84 | E94 | E105 | E115 | E126 | E136 | E147 | E157 | E168 | E178 | E189 | E199 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| nccl-async | 1.9896 | 0.7963 | 0.6371 | 0.5566 | 0.5206 | 0.5033 | 0.4894 | 0.4723 | 0.4576 | 0.4441 | 0.2055 | 0.1749 | 0.1584 | 0.1451 | 0.1449 | 0.0798 | 0.0742 | 0.0691 | 0.0652 | 0.0618 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3991 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3054 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2955 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 390 | 386 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1828.6 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu2 | 1828.6 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu0 | 1828.5 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 1.4s |
| resnet-graph | nccl-async | gpu1 | 0.9s | 0.0s | 0.0s | 0.0s | 1.7s |
| resnet-graph | nccl-async | gpu2 | 0.9s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 264 | 0 | 307 | 42.0 | 6463/10356 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 182.8 | 10.0% |

## MSF Passive Observation

Per the v2 framing (`docs/design/msf-cadence-control-v2.md`), DDP is a synchronization-of-coupled-chaotic-oscillators problem at **two scales** linked by AllReduce. Each subsection below is tagged by the scale it operates at:

- **Top scale (meta-oscillator)**: the cross-rank-collapsed observable `D_mean(t)`, the OU process the system spirals toward. The model we ship is the centroid that sits on the synchronization manifold; convergence is exclusively a top-scale phenomenon.
- **Bottom scale (per-GPU)**: per-rank `D_i(τ)` within a cycle, chaotic by construction with positive within-cycle Lyapunov `λ_T(LR)`. Per-replica trajectories don't converge — that's by design.
- **Cross-scale consistency**: cross-rank Pearson `r` and per-rank vs meta slope agreement. The gate that validates the meta-oscillator framing — when `r < 0.95` for any rank pair, the framing has broken and bottom-scale per-rank treatment is required (e.g. cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons).

Historical proxy `λ̂ = (1/k) * log(D_t / D_{t-1})` from v1 doc survives only as a coarse phase indicator; the v2 estimators are the by-k OLS slope (within-cycle Lyapunov, bottom-scale) and CUSUM-on-OU-residual (regime detection, top-scale).

Phase candidates flag epochs where `λ_min < -1e-2` AND `D_end / prev_D_end < 1/3` (collapse signature, e.g. LR drop boundary).

### Summary (top scale)

| Model | Mode | Epochs | Div Events | Phase Candidates | Final D | Final λ_ema |
|-------|------|--------|-----------:|-----------------:|--------:|------------:|
| resnet-graph | nccl-async | 188 | 307 | 0 | 5.39e-3 | +1.29e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 307 | 9.52e-2 | 7.90e-2 | 0.00e0 | 4.14e-1 | 22.5 | -1.72e-4 | 2.36e-3 |
| resnet-graph | nccl-async | 1 | 307 | 9.72e-2 | 8.12e-2 | 0.00e0 | 4.12e-1 | 56.0 | -1.71e-4 | 2.75e-3 |
| resnet-graph | nccl-async | 2 | 307 | 9.62e-2 | 8.04e-2 | 0.00e0 | 3.81e-1 | 21.5 | -1.65e-4 | 2.68e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9986 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9980 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9992 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 29 (0,1,2,3,21,22,26,35…141,146) | 0 (—) | — | 0,1,2,3,21,22,26,35…141,146 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 30 | 30 |
| resnet-graph | nccl-async | 0e0 | 5 | 15 | 15 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 5 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-3 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-2 | 10 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 5 | 0 | 0 |
| resnet-graph | nccl-async | 1e-1 | 10 | 0 | 0 |

### Predictive Value by LR Window (top scale)

Per-LR-window Pearson `r(λ_raw_t, ln(D_{t+1}))`. Pairs that straddle a window boundary are excluded so the LR-drop collapse cannot leak in as artefactual signal. Reading: a clean R1 (exponential growth at fixed LR) shows up as *non-zero* `r` within each window, with sign and magnitude that may differ between warmup, post-drop transient, and late-training phases.

| Model | Mode | LR | Epochs | n_pairs | r(λ → ln D_{t+1}) |
|-------|------|---:|--------|--------:|----------------:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 174 | +0.083 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 55 | +0.234 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 73 | -0.008 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 304 | -0.002 | 187 | +0.095 | +0.011 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 305 | 3.34e1–7.88e1 | 6.38e1 | 4.01e-3 | 9.57e-3 | 1.19e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 176 | 64–77902 | +1.085e-5 | 0.391 | +1.114e-5 | 0.394 | 94 | +1.780e-6 | 0.028 | 32–980 | +1.345e-3 | 0.694 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 159 | 885–77902 | +9.253e-6 | 0.365 | +9.459e-6 | 0.364 | 93 | +1.054e-6 | 0.011 | 40–980 | +1.267e-3 | 0.747 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 56 | 78277–116600 | +2.399e-5 | 0.331 | +2.414e-5 | 0.333 | 45 | +2.569e-5 | 0.637 | 280–958 | +9.921e-4 | 0.156 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 74 | 117253–156104 | -1.166e-5 | 0.120 | -1.175e-5 | 0.121 | 49 | -8.701e-6 | 0.086 | 310–754 | +6.020e-4 | 0.026 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.345e-3 | r0: +1.313e-3, r1: +1.351e-3, r2: +1.374e-3 | r0: 0.707, r1: 0.684, r2: 0.687 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.267e-3 | r0: +1.248e-3, r1: +1.274e-3, r2: +1.284e-3 | r0: 0.769, r1: 0.732, r2: 0.737 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +9.921e-4 | r0: +9.972e-4, r1: +9.827e-4, r2: +9.977e-4 | r0: 0.160, r1: 0.151, r2: 0.158 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +6.020e-4 | r0: +6.299e-4, r1: +5.981e-4, r2: +5.797e-4 | r0: 0.029, r1: 0.026, r2: 0.024 | 1.09× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇█████████████████████▆▃▄▄▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁` | `▂▇██▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▁▁▄▆▆▇▇▇▇▇▇▇▂▄▅▆▆▇▇▇▇▇▇█` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 18 | 0.00e0 | 4.14e-1 | 8.63e-2 | 6.11e-2 | 15 | -5.60e-2 | +1.71e-2 | -8.77e-3 | -3.58e-3 |
| 1 | 3.00e-1 | 12 | 5.46e-2 | 1.20e-1 | 7.33e-2 | 7.66e-2 | 19 | -2.85e-2 | +3.95e-2 | +1.47e-3 | -6.14e-5 |
| 2 | 3.00e-1 | 14 | 6.07e-2 | 1.23e-1 | 7.45e-2 | 7.98e-2 | 22 | -3.39e-2 | +3.49e-2 | +7.08e-4 | +6.28e-4 |
| 3 | 3.00e-1 | 13 | 7.82e-2 | 1.40e-1 | 8.73e-2 | 8.12e-2 | 18 | -2.43e-2 | +2.12e-2 | -1.78e-4 | -3.73e-5 |
| 4 | 3.00e-1 | 1 | 8.03e-2 | 8.03e-2 | 8.03e-2 | 8.03e-2 | 18 | -6.58e-4 | -6.58e-4 | -6.58e-4 | -9.95e-5 |
| 5 | 3.00e-1 | 1 | 7.82e-2 | 7.82e-2 | 7.82e-2 | 7.82e-2 | 318 | -8.15e-5 | -8.15e-5 | -8.15e-5 | -9.77e-5 |
| 6 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 340 | +3.34e-3 | +3.34e-3 | +3.34e-3 | +2.47e-4 |
| 7 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 312 | -2.69e-4 | -2.69e-4 | -2.69e-4 | +1.95e-4 |
| 8 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 346 | -1.52e-4 | -1.52e-4 | -1.52e-4 | +1.61e-4 |
| 9 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 320 | +9.24e-5 | +9.24e-5 | +9.24e-5 | +1.54e-4 |
| 11 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 359 | -1.34e-4 | -1.34e-4 | -1.34e-4 | +1.25e-4 |
| 12 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 285 | +1.61e-5 | +1.61e-5 | +1.61e-5 | +1.14e-4 |
| 13 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 284 | -2.00e-4 | -2.00e-4 | -2.00e-4 | +8.26e-5 |
| 14 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 280 | +6.88e-5 | +6.88e-5 | +6.88e-5 | +8.13e-5 |
| 15 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 319 | -8.01e-5 | -8.01e-5 | -8.01e-5 | +6.51e-5 |
| 16 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 307 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +7.38e-5 |
| 17 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 294 | -7.03e-5 | -7.03e-5 | -7.03e-5 | +5.94e-5 |
| 18 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 284 | -2.14e-5 | -2.14e-5 | -2.14e-5 | +5.13e-5 |
| 19 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 294 | -1.26e-5 | -1.26e-5 | -1.26e-5 | +4.49e-5 |
| 20 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 303 | +1.66e-5 | +1.66e-5 | +1.66e-5 | +4.21e-5 |
| 21 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 314 | +1.92e-5 | +1.92e-5 | +1.92e-5 | +3.98e-5 |
| 22 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 278 | +5.68e-5 | +5.68e-5 | +5.68e-5 | +4.15e-5 |
| 23 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 277 | -1.43e-4 | -1.43e-4 | -1.43e-4 | +2.30e-5 |
| 25 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 359 | +4.58e-5 | +4.58e-5 | +4.58e-5 | +2.53e-5 |
| 26 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 291 | +2.74e-4 | +2.74e-4 | +2.74e-4 | +5.02e-5 |
| 27 | 3.00e-1 | 2 | 2.01e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 245 | -2.74e-4 | +1.39e-5 | -1.30e-4 | +1.74e-5 |
| 29 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 300 | -1.39e-4 | -1.39e-4 | -1.39e-4 | +1.70e-6 |
| 30 | 3.00e-1 | 2 | 1.97e-1 | 2.04e-1 | 2.01e-1 | 1.97e-1 | 227 | -1.63e-4 | +2.01e-4 | +1.89e-5 | +3.14e-6 |
| 31 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 244 | -2.44e-4 | -2.44e-4 | -2.44e-4 | -2.16e-5 |
| 32 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 261 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +9.53e-7 |
| 33 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 264 | -6.12e-7 | -6.12e-7 | -6.12e-7 | +7.97e-7 |
| 34 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 288 | +9.75e-5 | +9.75e-5 | +9.75e-5 | +1.05e-5 |
| 35 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 265 | +5.04e-5 | +5.04e-5 | +5.04e-5 | +1.45e-5 |
| 36 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 275 | -6.45e-5 | -6.45e-5 | -6.45e-5 | +6.56e-6 |
| 37 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 270 | +6.09e-5 | +6.09e-5 | +6.09e-5 | +1.20e-5 |
| 38 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 267 | -4.56e-5 | -4.56e-5 | -4.56e-5 | +6.24e-6 |
| 39 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 301 | -2.65e-5 | -2.65e-5 | -2.65e-5 | +2.97e-6 |
| 40 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 284 | +1.40e-4 | +1.40e-4 | +1.40e-4 | +1.66e-5 |
| 41 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 293 | -7.89e-5 | -7.89e-5 | -7.89e-5 | +7.08e-6 |
| 42 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 249 | +5.28e-5 | +5.28e-5 | +5.28e-5 | +1.17e-5 |
| 43 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 272 | -1.60e-4 | -1.60e-4 | -1.60e-4 | -5.55e-6 |
| 44 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 265 | +6.87e-5 | +6.87e-5 | +6.87e-5 | +1.87e-6 |
| 45 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 246 | +2.66e-5 | +2.66e-5 | +2.66e-5 | +4.35e-6 |
| 46 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 252 | -1.56e-4 | -1.56e-4 | -1.56e-4 | -1.17e-5 |
| 47 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 317 | +4.12e-5 | +4.12e-5 | +4.12e-5 | -6.41e-6 |
| 48 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 270 | +2.65e-4 | +2.65e-4 | +2.65e-4 | +2.07e-5 |
| 49 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 249 | -2.43e-4 | -2.43e-4 | -2.43e-4 | -5.61e-6 |
| 50 | 3.00e-1 | 2 | 1.96e-1 | 1.99e-1 | 1.97e-1 | 1.99e-1 | 209 | -5.72e-5 | +7.05e-5 | +6.61e-6 | -2.65e-6 |
| 51 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 251 | -2.79e-4 | -2.79e-4 | -2.79e-4 | -3.03e-5 |
| 52 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 233 | +3.17e-4 | +3.17e-4 | +3.17e-4 | +4.47e-6 |
| 53 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 214 | -7.95e-5 | -7.95e-5 | -7.95e-5 | -3.93e-6 |
| 54 | 3.00e-1 | 2 | 1.89e-1 | 1.97e-1 | 1.93e-1 | 1.97e-1 | 209 | -1.69e-4 | +2.09e-4 | +1.96e-5 | +2.44e-6 |
| 56 | 3.00e-1 | 2 | 1.89e-1 | 2.11e-1 | 2.00e-1 | 2.11e-1 | 210 | -1.35e-4 | +5.25e-4 | +1.95e-4 | +4.23e-5 |
| 57 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 241 | -4.39e-4 | -4.39e-4 | -4.39e-4 | -5.83e-6 |
| 58 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 214 | +2.23e-4 | +2.23e-4 | +2.23e-4 | +1.70e-5 |
| 59 | 3.00e-1 | 2 | 1.87e-1 | 1.93e-1 | 1.90e-1 | 1.93e-1 | 227 | -2.86e-4 | +1.57e-4 | -6.44e-5 | +3.77e-6 |
| 61 | 3.00e-1 | 2 | 1.90e-1 | 2.02e-1 | 1.96e-1 | 2.02e-1 | 196 | -6.83e-5 | +3.24e-4 | +1.28e-4 | +2.93e-5 |
| 62 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 246 | -3.86e-4 | -3.86e-4 | -3.86e-4 | -1.22e-5 |
| 63 | 3.00e-1 | 2 | 1.98e-1 | 2.03e-1 | 2.00e-1 | 2.03e-1 | 180 | +1.21e-4 | +2.79e-4 | +2.00e-4 | +2.74e-5 |
| 65 | 3.00e-1 | 2 | 1.78e-1 | 1.95e-1 | 1.86e-1 | 1.95e-1 | 178 | -5.44e-4 | +5.10e-4 | -1.73e-5 | +2.41e-5 |
| 66 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 220 | -4.43e-4 | -4.43e-4 | -4.43e-4 | -2.26e-5 |
| 67 | 3.00e-1 | 2 | 1.91e-1 | 1.92e-1 | 1.91e-1 | 1.92e-1 | 204 | +8.25e-6 | +3.59e-4 | +1.84e-4 | +1.48e-5 |
| 68 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 254 | -7.66e-5 | -7.66e-5 | -7.66e-5 | +5.67e-6 |
| 69 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 213 | +4.02e-4 | +4.02e-4 | +4.02e-4 | +4.54e-5 |
| 70 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 210 | -3.66e-4 | -3.66e-4 | -3.66e-4 | +4.24e-6 |
| 71 | 3.00e-1 | 2 | 1.85e-1 | 1.90e-1 | 1.88e-1 | 1.85e-1 | 190 | -1.44e-4 | +1.41e-5 | -6.48e-5 | -9.67e-6 |
| 72 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 219 | -7.41e-5 | -7.41e-5 | -7.41e-5 | -1.61e-5 |
| 73 | 3.00e-1 | 2 | 1.89e-1 | 1.93e-1 | 1.91e-1 | 1.89e-1 | 187 | -1.30e-4 | +2.91e-4 | +8.02e-5 | +9.24e-8 |
| 74 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 229 | -2.06e-4 | -2.06e-4 | -2.06e-4 | -2.05e-5 |
| 75 | 3.00e-1 | 2 | 1.95e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 185 | +1.69e-5 | +3.64e-4 | +1.91e-4 | +1.78e-5 |
| 76 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 222 | -4.21e-4 | -4.21e-4 | -4.21e-4 | -2.60e-5 |
| 77 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 209 | +4.35e-4 | +4.35e-4 | +4.35e-4 | +2.01e-5 |
| 78 | 3.00e-1 | 2 | 1.79e-1 | 1.87e-1 | 1.83e-1 | 1.79e-1 | 150 | -3.09e-4 | -2.48e-4 | -2.78e-4 | -3.69e-5 |
| 79 | 3.00e-1 | 1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 179 | -4.20e-4 | -4.20e-4 | -4.20e-4 | -7.52e-5 |
| 80 | 3.00e-1 | 2 | 1.82e-1 | 1.89e-1 | 1.85e-1 | 1.89e-1 | 157 | +2.19e-4 | +4.55e-4 | +3.37e-4 | +1.94e-6 |
| 81 | 3.00e-1 | 2 | 1.71e-1 | 1.89e-1 | 1.80e-1 | 1.89e-1 | 158 | -4.88e-4 | +6.21e-4 | +6.66e-5 | +1.98e-5 |
| 82 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 228 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -1.63e-5 |
| 83 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 187 | +7.12e-4 | +7.12e-4 | +7.12e-4 | +5.65e-5 |
| 84 | 3.00e-1 | 2 | 1.85e-1 | 1.86e-1 | 1.86e-1 | 1.85e-1 | 147 | -3.78e-4 | -2.90e-5 | -2.04e-4 | +8.85e-6 |
| 85 | 3.00e-1 | 2 | 1.66e-1 | 1.86e-1 | 1.76e-1 | 1.86e-1 | 147 | -6.00e-4 | +7.80e-4 | +9.04e-5 | +3.12e-5 |
| 86 | 3.00e-1 | 2 | 1.65e-1 | 1.88e-1 | 1.77e-1 | 1.88e-1 | 155 | -5.90e-4 | +8.25e-4 | +1.18e-4 | +5.47e-5 |
| 87 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 185 | -4.76e-4 | -4.76e-4 | -4.76e-4 | +1.60e-6 |
| 88 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 187 | +4.46e-4 | +4.46e-4 | +4.46e-4 | +4.61e-5 |
| 89 | 3.00e-1 | 2 | 1.86e-1 | 1.96e-1 | 1.91e-1 | 1.96e-1 | 156 | -3.99e-5 | +3.42e-4 | +1.51e-4 | +6.80e-5 |
| 90 | 3.00e-1 | 2 | 1.68e-1 | 1.80e-1 | 1.74e-1 | 1.80e-1 | 133 | -9.25e-4 | +5.29e-4 | -1.98e-4 | +2.46e-5 |
| 91 | 3.00e-1 | 2 | 1.60e-1 | 1.85e-1 | 1.72e-1 | 1.85e-1 | 139 | -6.59e-4 | +1.07e-3 | +2.04e-4 | +6.73e-5 |
| 92 | 3.00e-1 | 2 | 1.69e-1 | 1.74e-1 | 1.71e-1 | 1.74e-1 | 122 | -5.66e-4 | +2.46e-4 | -1.60e-4 | +2.83e-5 |
| 93 | 3.00e-1 | 2 | 1.63e-1 | 1.81e-1 | 1.72e-1 | 1.81e-1 | 114 | -3.89e-4 | +9.48e-4 | +2.79e-4 | +8.27e-5 |
| 94 | 3.00e-1 | 2 | 1.53e-1 | 1.76e-1 | 1.64e-1 | 1.76e-1 | 114 | -1.07e-3 | +1.23e-3 | +7.77e-5 | +9.32e-5 |
| 95 | 3.00e-1 | 2 | 1.55e-1 | 1.75e-1 | 1.65e-1 | 1.75e-1 | 120 | -8.02e-4 | +1.02e-3 | +1.11e-4 | +1.06e-4 |
| 96 | 3.00e-1 | 2 | 1.60e-1 | 1.71e-1 | 1.65e-1 | 1.71e-1 | 114 | -6.04e-4 | +6.21e-4 | +8.54e-6 | +9.34e-5 |
| 97 | 3.00e-1 | 3 | 1.58e-1 | 1.74e-1 | 1.63e-1 | 1.58e-1 | 114 | -8.46e-4 | +8.59e-4 | -1.72e-4 | +1.80e-5 |
| 98 | 3.00e-1 | 1 | 1.53e-1 | 1.53e-1 | 1.53e-1 | 1.53e-1 | 147 | -2.11e-4 | -2.11e-4 | -2.11e-4 | -4.94e-6 |
| 99 | 3.00e-1 | 2 | 1.72e-1 | 1.78e-1 | 1.75e-1 | 1.78e-1 | 138 | +2.82e-4 | +6.88e-4 | +4.85e-4 | +8.62e-5 |
| 100 | 3.00e-2 | 3 | 1.57e-2 | 1.70e-1 | 1.18e-1 | 1.57e-2 | 108 | -2.20e-2 | -3.25e-5 | -7.45e-3 | -2.17e-3 |
| 101 | 3.00e-2 | 2 | 1.50e-2 | 1.72e-2 | 1.61e-2 | 1.72e-2 | 114 | -3.39e-4 | +1.21e-3 | +4.36e-4 | -1.66e-3 |
| 102 | 3.00e-2 | 2 | 1.62e-2 | 1.94e-2 | 1.78e-2 | 1.94e-2 | 127 | -3.62e-4 | +1.41e-3 | +5.22e-4 | -1.24e-3 |
| 103 | 3.00e-2 | 2 | 1.81e-2 | 1.94e-2 | 1.87e-2 | 1.94e-2 | 117 | -4.60e-4 | +5.96e-4 | +6.76e-5 | -9.86e-4 |
| 104 | 3.00e-2 | 2 | 1.79e-2 | 1.88e-2 | 1.84e-2 | 1.88e-2 | 100 | -5.62e-4 | +4.63e-4 | -4.97e-5 | -8.03e-4 |
| 105 | 3.00e-2 | 2 | 1.74e-2 | 1.76e-2 | 1.75e-2 | 1.76e-2 | 122 | -7.56e-4 | +9.57e-5 | -3.30e-4 | -7.09e-4 |
| 106 | 3.00e-2 | 1 | 1.89e-2 | 1.89e-2 | 1.89e-2 | 1.89e-2 | 304 | +2.46e-4 | +2.46e-4 | +2.46e-4 | -6.13e-4 |
| 107 | 3.00e-2 | 1 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 2.99e-2 | 284 | +1.61e-3 | +1.61e-3 | +1.61e-3 | -3.91e-4 |
| 108 | 3.00e-2 | 1 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 255 | -8.66e-5 | -8.66e-5 | -8.66e-5 | -3.60e-4 |
| 109 | 3.00e-2 | 1 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 260 | -4.19e-5 | -4.19e-5 | -4.19e-5 | -3.29e-4 |
| 110 | 3.00e-2 | 1 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 261 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -2.80e-4 |
| 111 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 300 | +8.30e-5 | +8.30e-5 | +8.30e-5 | -2.44e-4 |
| 112 | 3.00e-2 | 1 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 3.35e-2 | 296 | +2.69e-4 | +2.69e-4 | +2.69e-4 | -1.92e-4 |
| 113 | 3.00e-2 | 1 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 310 | +9.21e-6 | +9.21e-6 | +9.21e-6 | -1.72e-4 |
| 114 | 3.00e-2 | 1 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 308 | +2.06e-4 | +2.06e-4 | +2.06e-4 | -1.34e-4 |
| 115 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 322 | +1.28e-5 | +1.28e-5 | +1.28e-5 | -1.20e-4 |
| 117 | 3.00e-2 | 2 | 3.69e-2 | 3.92e-2 | 3.80e-2 | 3.92e-2 | 248 | +7.63e-5 | +2.48e-4 | +1.62e-4 | -6.51e-5 |
| 118 | 3.00e-2 | 1 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 279 | -4.14e-4 | -4.14e-4 | -4.14e-4 | -1.00e-4 |
| 120 | 3.00e-2 | 2 | 3.78e-2 | 4.24e-2 | 4.01e-2 | 4.24e-2 | 248 | +2.28e-4 | +4.60e-4 | +3.44e-4 | -1.45e-5 |
| 122 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 346 | -3.64e-4 | -3.64e-4 | -3.64e-4 | -4.94e-5 |
| 123 | 3.00e-2 | 2 | 4.16e-2 | 4.32e-2 | 4.24e-2 | 4.16e-2 | 250 | -1.52e-4 | +4.84e-4 | +1.66e-4 | -1.17e-5 |
| 125 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 325 | -3.61e-5 | -3.61e-5 | -3.61e-5 | -1.41e-5 |
| 126 | 3.00e-2 | 2 | 4.36e-2 | 4.46e-2 | 4.41e-2 | 4.36e-2 | 248 | -9.43e-5 | +2.97e-4 | +1.01e-4 | +5.87e-6 |
| 128 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 364 | -8.46e-5 | -8.46e-5 | -8.46e-5 | -3.18e-6 |
| 129 | 3.00e-2 | 1 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 284 | +5.23e-4 | +5.23e-4 | +5.23e-4 | +4.95e-5 |
| 130 | 3.00e-2 | 1 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 4.58e-2 | 282 | -2.41e-4 | -2.41e-4 | -2.41e-4 | +2.04e-5 |
| 131 | 3.00e-2 | 1 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 262 | +1.03e-4 | +1.03e-4 | +1.03e-4 | +2.87e-5 |
| 132 | 3.00e-2 | 1 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 300 | +4.24e-6 | +4.24e-6 | +4.24e-6 | +2.62e-5 |
| 133 | 3.00e-2 | 1 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 321 | +1.79e-4 | +1.79e-4 | +1.79e-4 | +4.15e-5 |
| 134 | 3.00e-2 | 1 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 5.28e-2 | 286 | +1.94e-4 | +1.94e-4 | +1.94e-4 | +5.67e-5 |
| 135 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 278 | -2.08e-4 | -2.08e-4 | -2.08e-4 | +3.02e-5 |
| 136 | 3.00e-2 | 1 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 278 | +2.06e-5 | +2.06e-5 | +2.06e-5 | +2.92e-5 |
| 137 | 3.00e-2 | 1 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 301 | -1.27e-6 | -1.27e-6 | -1.27e-6 | +2.62e-5 |
| 138 | 3.00e-2 | 1 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 284 | +2.55e-4 | +2.55e-4 | +2.55e-4 | +4.91e-5 |
| 139 | 3.00e-2 | 1 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 290 | -7.28e-5 | -7.28e-5 | -7.28e-5 | +3.69e-5 |
| 140 | 3.00e-2 | 1 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 269 | +2.95e-5 | +2.95e-5 | +2.95e-5 | +3.62e-5 |
| 141 | 3.00e-2 | 1 | 5.32e-2 | 5.32e-2 | 5.32e-2 | 5.32e-2 | 251 | +2.66e-6 | +2.66e-6 | +2.66e-6 | +3.28e-5 |
| 142 | 3.00e-2 | 1 | 5.17e-2 | 5.17e-2 | 5.17e-2 | 5.17e-2 | 269 | -1.05e-4 | -1.05e-4 | -1.05e-4 | +1.90e-5 |
| 143 | 3.00e-2 | 1 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 5.50e-2 | 276 | +2.24e-4 | +2.24e-4 | +2.24e-4 | +3.95e-5 |
| 144 | 3.00e-2 | 1 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 272 | -6.69e-6 | -6.69e-6 | -6.69e-6 | +3.49e-5 |
| 145 | 3.00e-2 | 1 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 5.57e-2 | 269 | +5.39e-5 | +5.39e-5 | +5.39e-5 | +3.68e-5 |
| 146 | 3.00e-2 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 263 | +4.90e-5 | +4.90e-5 | +4.90e-5 | +3.80e-5 |
| 147 | 3.00e-2 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 272 | -8.90e-7 | -8.90e-7 | -8.90e-7 | +3.41e-5 |
| 148 | 3.00e-2 | 1 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 259 | +1.78e-5 | +1.78e-5 | +1.78e-5 | +3.25e-5 |
| 149 | 3.00e-2 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 266 | -1.91e-5 | -1.91e-5 | -1.91e-5 | +2.73e-5 |
| 150 | 3.00e-3 | 2 | 5.54e-2 | 5.89e-2 | 5.72e-2 | 5.54e-2 | 197 | -3.08e-4 | +1.83e-4 | -6.26e-5 | +7.80e-6 |
| 151 | 3.00e-3 | 1 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 5.26e-3 | 235 | -1.00e-2 | -1.00e-2 | -1.00e-2 | -9.95e-4 |
| 152 | 3.00e-3 | 1 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 225 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -8.71e-4 |
| 153 | 3.00e-3 | 1 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 5.36e-3 | 252 | -1.48e-4 | -1.48e-4 | -1.48e-4 | -7.98e-4 |
| 154 | 3.00e-3 | 2 | 5.30e-3 | 5.55e-3 | 5.43e-3 | 5.30e-3 | 200 | -2.30e-4 | +1.57e-4 | -3.60e-5 | -6.56e-4 |
| 155 | 3.00e-3 | 1 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 255 | -1.54e-4 | -1.54e-4 | -1.54e-4 | -6.05e-4 |
| 156 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 257 | +5.41e-4 | +5.41e-4 | +5.41e-4 | -4.91e-4 |
| 157 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 240 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -4.53e-4 |
| 158 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 233 | -1.26e-4 | -1.26e-4 | -1.26e-4 | -4.20e-4 |
| 159 | 3.00e-3 | 2 | 5.48e-3 | 5.67e-3 | 5.58e-3 | 5.67e-3 | 210 | -4.27e-5 | +1.61e-4 | +5.94e-5 | -3.28e-4 |
| 160 | 3.00e-3 | 1 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 260 | -9.89e-5 | -9.89e-5 | -9.89e-5 | -3.05e-4 |
| 161 | 3.00e-3 | 1 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 5.92e-3 | 216 | +3.18e-4 | +3.18e-4 | +3.18e-4 | -2.43e-4 |
| 162 | 3.00e-3 | 1 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 242 | -3.12e-4 | -3.12e-4 | -3.12e-4 | -2.50e-4 |
| 163 | 3.00e-3 | 1 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 260 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -2.08e-4 |
| 164 | 3.00e-3 | 2 | 5.60e-3 | 5.91e-3 | 5.76e-3 | 5.60e-3 | 195 | -2.77e-4 | +1.38e-4 | -6.94e-5 | -1.84e-4 |
| 165 | 3.00e-3 | 1 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 221 | -3.01e-4 | -3.01e-4 | -3.01e-4 | -1.95e-4 |
| 166 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 203 | +3.25e-4 | +3.25e-4 | +3.25e-4 | -1.43e-4 |
| 167 | 3.00e-3 | 2 | 5.40e-3 | 5.53e-3 | 5.47e-3 | 5.40e-3 | 181 | -1.37e-4 | -5.86e-5 | -9.77e-5 | -1.35e-4 |
| 168 | 3.00e-3 | 1 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 206 | -1.99e-4 | -1.99e-4 | -1.99e-4 | -1.41e-4 |
| 169 | 3.00e-3 | 2 | 5.53e-3 | 5.55e-3 | 5.54e-3 | 5.55e-3 | 185 | +2.28e-5 | +2.96e-4 | +1.59e-4 | -8.57e-5 |
| 170 | 3.00e-3 | 1 | 5.20e-3 | 5.20e-3 | 5.20e-3 | 5.20e-3 | 207 | -3.15e-4 | -3.15e-4 | -3.15e-4 | -1.09e-4 |
| 171 | 3.00e-3 | 2 | 5.58e-3 | 5.64e-3 | 5.61e-3 | 5.64e-3 | 185 | +5.93e-5 | +3.20e-4 | +1.90e-4 | -5.32e-5 |
| 173 | 3.00e-3 | 2 | 5.27e-3 | 6.43e-3 | 5.85e-3 | 6.43e-3 | 198 | -2.36e-4 | +1.01e-3 | +3.86e-4 | +3.65e-5 |
| 174 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 249 | -5.58e-4 | -5.58e-4 | -5.58e-4 | -2.30e-5 |
| 175 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 205 | +4.53e-4 | +4.53e-4 | +4.53e-4 | +2.47e-5 |
| 176 | 3.00e-3 | 2 | 5.48e-3 | 5.57e-3 | 5.52e-3 | 5.48e-3 | 169 | -5.29e-4 | -9.03e-5 | -3.10e-4 | -3.67e-5 |
| 177 | 3.00e-3 | 1 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 201 | -3.35e-4 | -3.35e-4 | -3.35e-4 | -6.66e-5 |
| 178 | 3.00e-3 | 2 | 5.65e-3 | 5.69e-3 | 5.67e-3 | 5.69e-3 | 180 | +4.21e-5 | +4.77e-4 | +2.60e-4 | -6.76e-6 |
| 179 | 3.00e-3 | 2 | 5.40e-3 | 6.32e-3 | 5.86e-3 | 6.32e-3 | 174 | -2.27e-4 | +9.05e-4 | +3.39e-4 | +6.46e-5 |
| 180 | 3.00e-3 | 1 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 203 | -8.01e-4 | -8.01e-4 | -8.01e-4 | -2.19e-5 |
| 181 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 232 | +3.08e-4 | +3.08e-4 | +3.08e-4 | +1.11e-5 |
| 182 | 3.00e-3 | 2 | 5.45e-3 | 5.96e-3 | 5.71e-3 | 5.45e-3 | 147 | -6.05e-4 | +1.78e-4 | -2.13e-4 | -3.55e-5 |
| 183 | 3.00e-3 | 1 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 187 | -3.98e-4 | -3.98e-4 | -3.98e-4 | -7.17e-5 |
| 184 | 3.00e-3 | 2 | 5.24e-3 | 5.46e-3 | 5.35e-3 | 5.24e-3 | 156 | -2.60e-4 | +4.63e-4 | +1.01e-4 | -4.25e-5 |
| 185 | 3.00e-3 | 2 | 4.92e-3 | 5.58e-3 | 5.25e-3 | 5.58e-3 | 165 | -3.16e-4 | +7.57e-4 | +2.20e-4 | +1.28e-5 |
| 186 | 3.00e-3 | 1 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 186 | -1.76e-4 | -1.76e-4 | -1.76e-4 | -6.09e-6 |
| 187 | 3.00e-3 | 2 | 5.72e-3 | 5.79e-3 | 5.75e-3 | 5.79e-3 | 153 | +7.13e-5 | +3.05e-4 | +1.88e-4 | +2.97e-5 |
| 188 | 3.00e-3 | 2 | 5.19e-3 | 5.63e-3 | 5.41e-3 | 5.63e-3 | 142 | -5.53e-4 | +5.73e-4 | +9.98e-6 | +3.16e-5 |
| 189 | 3.00e-3 | 1 | 4.87e-3 | 4.87e-3 | 4.87e-3 | 4.87e-3 | 176 | -8.25e-4 | -8.25e-4 | -8.25e-4 | -5.41e-5 |
| 190 | 3.00e-3 | 2 | 5.47e-3 | 5.66e-3 | 5.56e-3 | 5.47e-3 | 142 | -2.33e-4 | +8.93e-4 | +3.30e-4 | +1.32e-5 |
| 191 | 3.00e-3 | 2 | 4.85e-3 | 5.54e-3 | 5.20e-3 | 5.54e-3 | 155 | -6.98e-4 | +8.60e-4 | +8.07e-5 | +3.38e-5 |
| 192 | 3.00e-3 | 1 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 191 | -3.04e-4 | -3.04e-4 | -3.04e-4 | +4.34e-9 |
| 193 | 3.00e-3 | 2 | 5.63e-3 | 5.96e-3 | 5.79e-3 | 5.96e-3 | 134 | +3.50e-4 | +4.20e-4 | +3.85e-4 | +7.35e-5 |
| 194 | 3.00e-3 | 2 | 4.94e-3 | 5.57e-3 | 5.26e-3 | 5.57e-3 | 121 | -1.14e-3 | +9.85e-4 | -7.52e-5 | +5.59e-5 |
| 195 | 3.00e-3 | 2 | 4.60e-3 | 5.13e-3 | 4.87e-3 | 5.13e-3 | 128 | -1.26e-3 | +8.46e-4 | -2.08e-4 | +1.64e-5 |
| 196 | 3.00e-3 | 2 | 4.71e-3 | 5.42e-3 | 5.07e-3 | 5.42e-3 | 129 | -5.49e-4 | +1.08e-3 | +2.65e-4 | +7.17e-5 |
| 197 | 3.00e-3 | 2 | 5.06e-3 | 5.64e-3 | 5.35e-3 | 5.64e-3 | 136 | -3.97e-4 | +7.95e-4 | +1.99e-4 | +1.02e-4 |
| 198 | 3.00e-3 | 2 | 5.06e-3 | 5.65e-3 | 5.35e-3 | 5.65e-3 | 124 | -6.14e-4 | +8.87e-4 | +1.37e-4 | +1.16e-4 |
| 199 | 3.00e-3 | 2 | 4.62e-3 | 5.39e-3 | 5.01e-3 | 5.39e-3 | 109 | -1.18e-3 | +1.41e-3 | +1.14e-4 | +1.29e-4 |

