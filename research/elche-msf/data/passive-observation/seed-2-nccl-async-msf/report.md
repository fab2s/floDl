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
| nccl-async | 0.062810 | 0.9208 | +0.0083 | 1993.7 | 537 | 40.8 | 100% | 100% | 6.1 |

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
| nccl-async | 1.9494 | 0.6823 | 0.6155 | 0.5553 | 0.5291 | 0.5130 | 0.4877 | 0.4795 | 0.4697 | 0.4759 | 0.2055 | 0.1727 | 0.1474 | 0.1538 | 0.1449 | 0.0808 | 0.0709 | 0.0684 | 0.0662 | 0.0628 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3997 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3053 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2950 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 396 | 393 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1992.6 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu1 | 1992.7 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu0 | 1992.6 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.9s |
| resnet-graph | nccl-async | gpu1 | 0.9s | 0.0s | 0.0s | 0.0s | 2.4s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 353 | 0 | 537 | 40.8 | 1541/10502 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 218.1 | 10.9% |

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
| resnet-graph | nccl-async | 192 | 537 | 0 | 5.88e-3 | +5.02e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 537 | 7.45e-2 | 6.18e-2 | 0.00e0 | 4.18e-1 | 43.8 | -1.56e-4 | 3.00e-3 |
| resnet-graph | nccl-async | 1 | 537 | 7.56e-2 | 6.60e-2 | 0.00e0 | 5.93e-1 | 38.4 | -1.75e-4 | 4.56e-3 |
| resnet-graph | nccl-async | 2 | 537 | 7.45e-2 | 6.45e-2 | 0.00e0 | 5.12e-1 | 17.9 | -1.83e-4 | 4.52e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9903 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9939 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9978 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 43 (0,1,2,3,4,5,6,7…145,146) | 0 (—) | — | 0,1,2,3,4,5,6,7…145,146 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 31 | 31 |
| resnet-graph | nccl-async | 0e0 | 5 | 17 | 17 |
| resnet-graph | nccl-async | 0e0 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 3 | 12 | 12 |
| resnet-graph | nccl-async | 1e-4 | 5 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 10 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 341 | +0.032 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 129 | +0.065 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 62 | +0.022 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 534 | +0.001 | 191 | +0.190 | +0.195 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 535 | 3.45e1–7.91e1 | 6.27e1 | 2.24e-3 | 3.39e-3 | 4.65e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 343 | 69–78042 | +1.700e-5 | 0.586 | +1.750e-5 | 0.598 | 93 | +1.147e-5 | 0.413 | 27–1013 | +1.446e-3 | 0.747 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 327 | 878–78042 | +1.703e-5 | 0.627 | +1.752e-5 | 0.634 | 92 | +1.117e-5 | 0.397 | 36–1013 | +1.443e-3 | 0.799 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 130 | 78559–117004 | -3.328e-7 | 0.000 | -8.913e-7 | 0.000 | 50 | +1.623e-5 | 0.120 | 40–903 | +1.486e-3 | 0.526 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 63 | 117699–156166 | -9.793e-6 | 0.083 | -9.963e-6 | 0.086 | 49 | -1.234e-5 | 0.103 | 385–829 | +5.449e-4 | 0.016 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.446e-3 | r0: +1.410e-3, r1: +1.457e-3, r2: +1.476e-3 | r0: 0.786, r1: 0.724, r2: 0.726 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.443e-3 | r0: +1.405e-3, r1: +1.456e-3, r2: +1.472e-3 | r0: 0.838, r1: 0.778, r2: 0.773 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.486e-3 | r0: +1.479e-3, r1: +1.481e-3, r2: +1.503e-3 | r0: 0.543, r1: 0.514, r2: 0.514 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +5.449e-4 | r0: +5.677e-4, r1: +5.692e-4, r2: +4.995e-4 | r0: 0.017, r1: 0.017, r2: 0.013 | 1.14× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇██████████████████▆▄▄▄▄▄▄▄▄▄▅▅▆▂▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▇▇████▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▅▅▆▇▇▇▇▆▆▆▇▇▇▅▅▆▇▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 17 | 0.00e0 | 5.93e-1 | 9.25e-2 | 5.16e-2 | 17 | -1.03e-1 | +1.05e-2 | -1.31e-2 | -6.76e-3 |
| 1 | 3.00e-1 | 15 | 5.41e-2 | 1.05e-1 | 6.29e-2 | 6.06e-2 | 18 | -3.04e-2 | +3.90e-2 | +4.91e-4 | -1.09e-3 |
| 2 | 3.00e-1 | 15 | 5.67e-2 | 1.10e-1 | 6.75e-2 | 7.07e-2 | 16 | -4.01e-2 | +3.57e-2 | +3.99e-4 | +1.89e-4 |
| 3 | 3.00e-1 | 12 | 6.44e-2 | 1.23e-1 | 8.55e-2 | 8.01e-2 | 19 | -4.07e-2 | +4.37e-2 | +7.87e-4 | +4.06e-5 |
| 4 | 3.00e-1 | 14 | 7.08e-2 | 1.31e-1 | 8.00e-2 | 7.08e-2 | 16 | -3.01e-2 | +2.64e-2 | -5.08e-4 | -6.55e-4 |
| 5 | 3.00e-1 | 15 | 6.35e-2 | 1.36e-1 | 7.76e-2 | 7.42e-2 | 17 | -3.99e-2 | +4.98e-2 | +1.22e-3 | +2.37e-4 |
| 6 | 3.00e-1 | 16 | 6.14e-2 | 1.32e-1 | 7.61e-2 | 7.55e-2 | 23 | -5.48e-2 | +4.07e-2 | -2.80e-4 | -2.26e-5 |
| 7 | 3.00e-1 | 18 | 6.38e-2 | 1.47e-1 | 8.19e-2 | 7.94e-2 | 17 | -2.96e-2 | +2.37e-2 | -6.46e-4 | -2.83e-4 |
| 8 | 3.00e-1 | 8 | 5.87e-2 | 1.37e-1 | 7.83e-2 | 9.82e-2 | 28 | -5.67e-2 | +4.30e-2 | +7.79e-4 | +6.62e-4 |
| 9 | 3.00e-1 | 10 | 8.25e-2 | 1.50e-1 | 9.20e-2 | 8.30e-2 | 24 | -2.11e-2 | +1.99e-2 | -4.80e-4 | -1.35e-4 |
| 10 | 3.00e-1 | 15 | 5.96e-2 | 1.32e-1 | 7.25e-2 | 7.02e-2 | 18 | -4.50e-2 | +2.90e-2 | -7.60e-4 | -2.96e-4 |
| 11 | 3.00e-1 | 22 | 6.06e-2 | 1.30e-1 | 7.18e-2 | 6.80e-2 | 16 | -4.66e-2 | +4.03e-2 | -8.25e-5 | -4.74e-4 |
| 12 | 3.00e-1 | 6 | 6.16e-2 | 1.34e-1 | 8.14e-2 | 7.90e-2 | 37 | -4.33e-2 | +4.51e-2 | +2.21e-3 | +5.47e-4 |
| 13 | 3.00e-1 | 10 | 6.59e-2 | 1.58e-1 | 9.14e-2 | 7.90e-2 | 21 | -1.78e-2 | +9.88e-3 | -1.38e-3 | -4.71e-4 |
| 14 | 3.00e-1 | 6 | 7.52e-2 | 1.50e-1 | 1.08e-1 | 9.79e-2 | 34 | -6.83e-3 | +1.64e-2 | +8.03e-4 | -9.02e-5 |
| 15 | 3.00e-1 | 9 | 8.40e-2 | 1.36e-1 | 9.37e-2 | 8.67e-2 | 26 | -1.27e-2 | +1.18e-2 | -4.46e-4 | -3.17e-4 |
| 16 | 3.00e-1 | 19 | 6.20e-2 | 1.30e-1 | 7.20e-2 | 7.06e-2 | 18 | -2.57e-2 | +1.87e-2 | -4.90e-4 | -5.22e-5 |
| 17 | 3.00e-1 | 11 | 5.42e-2 | 1.35e-1 | 6.83e-2 | 7.06e-2 | 18 | -5.85e-2 | +3.93e-2 | -6.83e-4 | -1.99e-5 |
| 18 | 3.00e-1 | 14 | 6.00e-2 | 1.29e-1 | 7.23e-2 | 8.27e-2 | 20 | -4.75e-2 | +4.02e-2 | +3.80e-4 | +6.92e-4 |
| 19 | 3.00e-1 | 1 | 7.49e-2 | 7.49e-2 | 7.49e-2 | 7.49e-2 | 19 | -5.18e-3 | -5.18e-3 | -5.18e-3 | +1.04e-4 |
| 20 | 3.00e-1 | 1 | 6.89e-2 | 6.89e-2 | 6.89e-2 | 6.89e-2 | 380 | -2.23e-4 | -2.23e-4 | -2.23e-4 | +7.17e-5 |
| 21 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 322 | +3.92e-3 | +3.92e-3 | +3.92e-3 | +4.57e-4 |
| 22 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 321 | -3.67e-4 | -3.67e-4 | -3.67e-4 | +3.74e-4 |
| 23 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 326 | -8.86e-5 | -8.86e-5 | -8.86e-5 | +3.28e-4 |
| 24 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 337 | +1.67e-5 | +1.67e-5 | +1.67e-5 | +2.97e-4 |
| 26 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 367 | +1.02e-5 | +1.02e-5 | +1.02e-5 | +2.68e-4 |
| 27 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 287 | +5.05e-5 | +5.05e-5 | +5.05e-5 | +2.47e-4 |
| 28 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 326 | -1.83e-4 | -1.83e-4 | -1.83e-4 | +2.04e-4 |
| 29 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 329 | +8.75e-5 | +8.75e-5 | +8.75e-5 | +1.92e-4 |
| 30 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 320 | -1.35e-5 | -1.35e-5 | -1.35e-5 | +1.71e-4 |
| 31 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 277 | -5.37e-5 | -5.37e-5 | -5.37e-5 | +1.49e-4 |
| 32 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 299 | -1.15e-4 | -1.15e-4 | -1.15e-4 | +1.22e-4 |
| 33 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 293 | +8.93e-5 | +8.93e-5 | +8.93e-5 | +1.19e-4 |
| 34 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 312 | -1.49e-5 | -1.49e-5 | -1.49e-5 | +1.06e-4 |
| 36 | 3.00e-1 | 2 | 2.07e-1 | 2.08e-1 | 2.07e-1 | 2.08e-1 | 257 | +2.82e-5 | +6.70e-5 | +4.76e-5 | +9.45e-5 |
| 38 | 3.00e-1 | 2 | 1.93e-1 | 2.08e-1 | 2.00e-1 | 2.08e-1 | 257 | -2.38e-4 | +2.78e-4 | +1.98e-5 | +8.29e-5 |
| 40 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 340 | -2.30e-4 | -2.30e-4 | -2.30e-4 | +5.16e-5 |
| 41 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 289 | +4.09e-4 | +4.09e-4 | +4.09e-4 | +8.73e-5 |
| 42 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 286 | -2.51e-4 | -2.51e-4 | -2.51e-4 | +5.35e-5 |
| 43 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 280 | +5.42e-5 | +5.42e-5 | +5.42e-5 | +5.35e-5 |
| 44 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 266 | +3.67e-5 | +3.67e-5 | +3.67e-5 | +5.19e-5 |
| 45 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 258 | -1.67e-4 | -1.67e-4 | -1.67e-4 | +2.99e-5 |
| 46 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 269 | +3.64e-5 | +3.64e-5 | +3.64e-5 | +3.06e-5 |
| 47 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 271 | -1.73e-5 | -1.73e-5 | -1.73e-5 | +2.58e-5 |
| 48 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 289 | +6.48e-5 | +6.48e-5 | +6.48e-5 | +2.97e-5 |
| 49 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 263 | -1.48e-5 | -1.48e-5 | -1.48e-5 | +2.53e-5 |
| 50 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 279 | -2.75e-5 | -2.75e-5 | -2.75e-5 | +2.00e-5 |
| 51 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 292 | +4.35e-5 | +4.35e-5 | +4.35e-5 | +2.23e-5 |
| 52 | 3.00e-1 | 2 | 2.01e-1 | 2.06e-1 | 2.03e-1 | 2.01e-1 | 239 | -1.14e-4 | +6.74e-5 | -2.31e-5 | +1.28e-5 |
| 54 | 3.00e-1 | 2 | 1.88e-1 | 2.10e-1 | 1.99e-1 | 2.10e-1 | 239 | -2.02e-4 | +4.57e-4 | +1.28e-4 | +3.79e-5 |
| 56 | 3.00e-1 | 2 | 1.91e-1 | 2.20e-1 | 2.05e-1 | 2.20e-1 | 258 | -2.71e-4 | +5.50e-4 | +1.40e-4 | +6.13e-5 |
| 58 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 314 | -3.30e-4 | -3.30e-4 | -3.30e-4 | +2.22e-5 |
| 59 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 259 | +2.61e-4 | +2.61e-4 | +2.61e-4 | +4.60e-5 |
| 60 | 3.00e-1 | 2 | 1.93e-1 | 1.96e-1 | 1.94e-1 | 1.93e-1 | 204 | -3.35e-4 | -6.63e-5 | -2.01e-4 | +4.79e-7 |
| 61 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 273 | -1.46e-4 | -1.46e-4 | -1.46e-4 | -1.42e-5 |
| 62 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 230 | +3.78e-4 | +3.78e-4 | +3.78e-4 | +2.51e-5 |
| 63 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 257 | -2.06e-4 | -2.06e-4 | -2.06e-4 | +1.91e-6 |
| 64 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 264 | +7.39e-5 | +7.39e-5 | +7.39e-5 | +9.11e-6 |
| 65 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 250 | +1.12e-4 | +1.12e-4 | +1.12e-4 | +1.94e-5 |
| 66 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 232 | -1.13e-4 | -1.13e-4 | -1.13e-4 | +6.23e-6 |
| 67 | 3.00e-1 | 2 | 1.91e-1 | 1.97e-1 | 1.94e-1 | 1.91e-1 | 204 | -1.54e-4 | +2.57e-5 | -6.43e-5 | -8.07e-6 |
| 68 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 250 | -1.45e-4 | -1.45e-4 | -1.45e-4 | -2.18e-5 |
| 69 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 264 | +2.59e-4 | +2.59e-4 | +2.59e-4 | +6.31e-6 |
| 70 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 250 | +6.75e-5 | +6.75e-5 | +6.75e-5 | +1.24e-5 |
| 71 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 248 | -3.88e-5 | -3.88e-5 | -3.88e-5 | +7.31e-6 |
| 72 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 248 | -2.36e-5 | -2.36e-5 | -2.36e-5 | +4.22e-6 |
| 73 | 3.00e-1 | 2 | 1.96e-1 | 1.97e-1 | 1.96e-1 | 1.96e-1 | 220 | -3.50e-5 | -8.63e-6 | -2.18e-5 | -8.56e-7 |
| 74 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 246 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -1.28e-5 |
| 75 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 222 | +1.25e-4 | +1.25e-4 | +1.25e-4 | +9.86e-7 |
| 76 | 3.00e-1 | 2 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 191 | -8.95e-5 | +4.42e-6 | -4.25e-5 | -6.81e-6 |
| 77 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 228 | -2.16e-4 | -2.16e-4 | -2.16e-4 | -2.78e-5 |
| 78 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 205 | +2.82e-4 | +2.82e-4 | +2.82e-4 | +3.17e-6 |
| 79 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 232 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -9.27e-6 |
| 80 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 233 | +1.33e-4 | +1.33e-4 | +1.33e-4 | +4.94e-6 |
| 81 | 3.00e-1 | 2 | 1.94e-1 | 1.96e-1 | 1.95e-1 | 1.96e-1 | 202 | +1.11e-5 | +6.25e-5 | +3.68e-5 | +1.13e-5 |
| 82 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 220 | -2.02e-4 | -2.02e-4 | -2.02e-4 | -1.01e-5 |
| 83 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 213 | +1.16e-4 | +1.16e-4 | +1.16e-4 | +2.50e-6 |
| 84 | 3.00e-1 | 2 | 1.89e-1 | 1.92e-1 | 1.90e-1 | 1.92e-1 | 175 | -9.17e-5 | +1.04e-4 | +6.24e-6 | +4.19e-6 |
| 85 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 201 | -4.38e-4 | -4.38e-4 | -4.38e-4 | -4.00e-5 |
| 86 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 204 | +3.21e-4 | +3.21e-4 | +3.21e-4 | -3.94e-6 |
| 87 | 3.00e-1 | 2 | 1.87e-1 | 1.92e-1 | 1.89e-1 | 1.92e-1 | 175 | -2.48e-5 | +1.48e-4 | +6.17e-5 | +9.39e-6 |
| 88 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 208 | -3.39e-4 | -3.39e-4 | -3.39e-4 | -2.55e-5 |
| 89 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 233 | +2.54e-4 | +2.54e-4 | +2.54e-4 | +2.49e-6 |
| 90 | 3.00e-1 | 2 | 1.89e-1 | 1.95e-1 | 1.92e-1 | 1.89e-1 | 175 | -1.95e-4 | +1.44e-4 | -2.54e-5 | -4.51e-6 |
| 91 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 200 | -2.84e-4 | -2.84e-4 | -2.84e-4 | -3.25e-5 |
| 92 | 3.00e-1 | 2 | 1.84e-1 | 1.90e-1 | 1.87e-1 | 1.84e-1 | 175 | -1.90e-4 | +3.22e-4 | +6.57e-5 | -1.64e-5 |
| 93 | 3.00e-1 | 2 | 1.77e-1 | 1.91e-1 | 1.84e-1 | 1.91e-1 | 177 | -2.01e-4 | +4.57e-4 | +1.28e-4 | +1.43e-5 |
| 94 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 240 | -3.25e-4 | -3.25e-4 | -3.25e-4 | -1.96e-5 |
| 95 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 220 | +5.35e-4 | +5.35e-4 | +5.35e-4 | +3.59e-5 |
| 96 | 3.00e-1 | 2 | 1.85e-1 | 1.90e-1 | 1.88e-1 | 1.85e-1 | 148 | -2.27e-4 | -1.78e-4 | -2.02e-4 | -9.16e-6 |
| 97 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 170 | -5.66e-4 | -5.66e-4 | -5.66e-4 | -6.48e-5 |
| 98 | 3.00e-1 | 2 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 148 | +1.84e-5 | +3.60e-4 | +1.89e-4 | -1.83e-5 |
| 99 | 3.00e-1 | 2 | 1.65e-1 | 1.85e-1 | 1.75e-1 | 1.85e-1 | 148 | -4.77e-4 | +7.61e-4 | +1.42e-4 | +1.84e-5 |
| 100 | 3.00e-2 | 1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 193 | -6.38e-4 | -6.38e-4 | -6.38e-4 | -4.73e-5 |
| 101 | 3.00e-2 | 2 | 1.99e-2 | 1.07e-1 | 6.33e-2 | 1.99e-2 | 150 | -1.12e-2 | -2.06e-3 | -6.62e-3 | -1.34e-3 |
| 102 | 3.00e-2 | 1 | 1.88e-2 | 1.88e-2 | 1.88e-2 | 1.88e-2 | 187 | -3.20e-4 | -3.20e-4 | -3.20e-4 | -1.24e-3 |
| 103 | 3.00e-2 | 3 | 1.86e-2 | 2.08e-2 | 1.99e-2 | 1.86e-2 | 130 | -8.38e-4 | +4.34e-4 | -8.18e-5 | -9.38e-4 |
| 104 | 3.00e-2 | 1 | 1.95e-2 | 1.95e-2 | 1.95e-2 | 1.95e-2 | 171 | +2.57e-4 | +2.57e-4 | +2.57e-4 | -8.19e-4 |
| 105 | 3.00e-2 | 2 | 2.15e-2 | 2.22e-2 | 2.18e-2 | 2.15e-2 | 129 | -2.31e-4 | +7.46e-4 | +2.57e-4 | -6.19e-4 |
| 106 | 3.00e-2 | 2 | 1.99e-2 | 2.28e-2 | 2.14e-2 | 2.28e-2 | 137 | -4.46e-4 | +1.00e-3 | +2.79e-4 | -4.41e-4 |
| 107 | 3.00e-2 | 1 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 187 | -4.52e-4 | -4.52e-4 | -4.52e-4 | -4.42e-4 |
| 108 | 3.00e-2 | 2 | 2.48e-2 | 2.57e-2 | 2.52e-2 | 2.57e-2 | 154 | +2.39e-4 | +8.09e-4 | +5.24e-4 | -2.62e-4 |
| 109 | 3.00e-2 | 2 | 2.29e-2 | 2.40e-2 | 2.34e-2 | 2.40e-2 | 126 | -6.84e-4 | +3.57e-4 | -1.63e-4 | -2.38e-4 |
| 110 | 3.00e-2 | 2 | 2.11e-2 | 2.53e-2 | 2.32e-2 | 2.53e-2 | 120 | -7.33e-4 | +1.51e-3 | +3.86e-4 | -1.08e-4 |
| 111 | 3.00e-2 | 2 | 2.21e-2 | 2.45e-2 | 2.33e-2 | 2.45e-2 | 120 | -8.81e-4 | +8.41e-4 | -2.00e-5 | -8.27e-5 |
| 112 | 3.00e-2 | 3 | 2.23e-2 | 2.56e-2 | 2.39e-2 | 2.39e-2 | 128 | -5.72e-4 | +1.11e-3 | -2.85e-6 | -6.14e-5 |
| 113 | 3.00e-2 | 1 | 2.34e-2 | 2.34e-2 | 2.34e-2 | 2.34e-2 | 163 | -1.46e-4 | -1.46e-4 | -1.46e-4 | -6.98e-5 |
| 114 | 3.00e-2 | 2 | 2.64e-2 | 2.77e-2 | 2.70e-2 | 2.77e-2 | 119 | +4.09e-4 | +7.27e-4 | +5.68e-4 | +4.98e-5 |
| 115 | 3.00e-2 | 2 | 2.43e-2 | 2.76e-2 | 2.60e-2 | 2.76e-2 | 111 | -8.10e-4 | +1.16e-3 | +1.73e-4 | +8.30e-5 |
| 116 | 3.00e-2 | 3 | 2.42e-2 | 2.63e-2 | 2.50e-2 | 2.44e-2 | 111 | -9.31e-4 | +7.36e-4 | -2.94e-4 | -1.73e-5 |
| 117 | 3.00e-2 | 2 | 2.40e-2 | 2.70e-2 | 2.55e-2 | 2.70e-2 | 111 | -1.03e-4 | +1.05e-3 | +4.72e-4 | +8.13e-5 |
| 118 | 3.00e-2 | 3 | 2.50e-2 | 2.91e-2 | 2.66e-2 | 2.55e-2 | 102 | -1.29e-3 | +1.32e-3 | -1.47e-4 | +1.11e-5 |
| 119 | 3.00e-2 | 1 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 168 | -7.36e-5 | -7.36e-5 | -7.36e-5 | +2.60e-6 |
| 120 | 3.00e-2 | 3 | 2.37e-2 | 3.05e-2 | 2.81e-2 | 2.37e-2 | 98 | -2.40e-3 | +1.23e-3 | -4.55e-4 | -1.56e-4 |
| 121 | 3.00e-2 | 2 | 2.51e-2 | 2.90e-2 | 2.71e-2 | 2.90e-2 | 83 | +4.26e-4 | +1.71e-3 | +1.07e-3 | +8.26e-5 |
| 122 | 3.00e-2 | 4 | 2.38e-2 | 2.83e-2 | 2.52e-2 | 2.47e-2 | 83 | -1.84e-3 | +2.06e-3 | -2.47e-4 | -2.26e-5 |
| 123 | 3.00e-2 | 2 | 2.41e-2 | 2.94e-2 | 2.67e-2 | 2.94e-2 | 94 | -1.88e-4 | +2.11e-3 | +9.58e-4 | +1.75e-4 |
| 124 | 3.00e-2 | 4 | 2.38e-2 | 2.90e-2 | 2.57e-2 | 2.38e-2 | 72 | -2.48e-3 | +1.53e-3 | -5.38e-4 | -7.53e-5 |
| 125 | 3.00e-2 | 3 | 2.35e-2 | 2.91e-2 | 2.55e-2 | 2.39e-2 | 71 | -2.76e-3 | +2.96e-3 | +3.30e-5 | -7.27e-5 |
| 126 | 3.00e-2 | 3 | 2.35e-2 | 2.97e-2 | 2.57e-2 | 2.40e-2 | 71 | -3.01e-3 | +3.33e-3 | +5.52e-5 | -6.69e-5 |
| 127 | 3.00e-2 | 4 | 2.25e-2 | 2.98e-2 | 2.52e-2 | 2.25e-2 | 62 | -3.74e-3 | +2.68e-3 | -3.90e-4 | -2.22e-4 |
| 128 | 3.00e-2 | 5 | 2.12e-2 | 2.96e-2 | 2.40e-2 | 2.25e-2 | 57 | -5.85e-3 | +3.87e-3 | -8.54e-5 | -2.00e-4 |
| 129 | 3.00e-2 | 4 | 2.18e-2 | 2.83e-2 | 2.42e-2 | 2.18e-2 | 52 | -2.64e-3 | +4.08e-3 | -2.78e-4 | -2.87e-4 |
| 130 | 3.00e-2 | 5 | 2.03e-2 | 2.96e-2 | 2.28e-2 | 2.18e-2 | 47 | -7.23e-3 | +6.84e-3 | +4.57e-5 | -1.81e-4 |
| 131 | 3.00e-2 | 7 | 1.97e-2 | 2.63e-2 | 2.15e-2 | 2.13e-2 | 42 | -5.09e-3 | +5.39e-3 | +2.54e-5 | -7.29e-5 |
| 132 | 3.00e-2 | 5 | 1.86e-2 | 2.78e-2 | 2.09e-2 | 1.86e-2 | 32 | -1.01e-2 | +8.21e-3 | -6.10e-4 | -3.57e-4 |
| 133 | 3.00e-2 | 16 | 1.17e-2 | 2.68e-2 | 1.54e-2 | 1.59e-2 | 22 | -1.33e-2 | +1.19e-2 | -7.25e-4 | -2.72e-5 |
| 134 | 3.00e-2 | 9 | 9.93e-3 | 2.81e-2 | 1.34e-2 | 1.09e-2 | 18 | -4.69e-2 | +4.18e-2 | -1.54e-3 | -1.03e-3 |
| 135 | 3.00e-2 | 2 | 1.13e-2 | 1.15e-2 | 1.14e-2 | 1.13e-2 | 17 | -1.20e-3 | +3.49e-3 | +1.14e-3 | -6.38e-4 |
| 136 | 3.00e-2 | 1 | 1.06e-2 | 1.06e-2 | 1.06e-2 | 1.06e-2 | 276 | -2.05e-4 | -2.05e-4 | -2.05e-4 | -5.95e-4 |
| 137 | 3.00e-2 | 1 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 295 | +5.57e-3 | +5.57e-3 | +5.57e-3 | +2.20e-5 |
| 138 | 3.00e-2 | 1 | 5.52e-2 | 5.52e-2 | 5.52e-2 | 5.52e-2 | 334 | +4.55e-6 | +4.55e-6 | +4.55e-6 | +2.02e-5 |
| 139 | 3.00e-2 | 1 | 5.73e-2 | 5.73e-2 | 5.73e-2 | 5.73e-2 | 318 | +1.20e-4 | +1.20e-4 | +1.20e-4 | +3.03e-5 |
| 140 | 3.00e-2 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 288 | -5.59e-5 | -5.59e-5 | -5.59e-5 | +2.16e-5 |
| 141 | 3.00e-2 | 1 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 5.49e-2 | 278 | -9.75e-5 | -9.75e-5 | -9.75e-5 | +9.72e-6 |
| 142 | 3.00e-2 | 1 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 5.37e-2 | 283 | -7.82e-5 | -7.82e-5 | -7.82e-5 | +9.23e-7 |
| 143 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 278 | +1.54e-4 | +1.54e-4 | +1.54e-4 | +1.62e-5 |
| 144 | 3.00e-2 | 1 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 280 | +1.76e-5 | +1.76e-5 | +1.76e-5 | +1.63e-5 |
| 145 | 3.00e-2 | 1 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 318 | +3.04e-5 | +3.04e-5 | +3.04e-5 | +1.77e-5 |
| 146 | 3.00e-2 | 1 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 263 | +7.47e-5 | +7.47e-5 | +7.47e-5 | +2.34e-5 |
| 147 | 3.00e-2 | 1 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 279 | -3.95e-5 | -3.95e-5 | -3.95e-5 | +1.71e-5 |
| 148 | 3.00e-2 | 1 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 314 | -2.94e-5 | -2.94e-5 | -2.94e-5 | +1.25e-5 |
| 149 | 3.00e-2 | 1 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 284 | +2.93e-4 | +2.93e-4 | +2.93e-4 | +4.05e-5 |
| 150 | 3.00e-3 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 249 | -2.68e-4 | -2.68e-4 | -2.68e-4 | +9.65e-6 |
| 151 | 3.00e-3 | 1 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 239 | -9.42e-4 | -9.42e-4 | -9.42e-4 | -8.55e-5 |
| 152 | 3.00e-3 | 1 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 5.50e-3 | 242 | -8.79e-3 | -8.79e-3 | -8.79e-3 | -9.56e-4 |
| 153 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 273 | -8.11e-5 | -8.11e-5 | -8.11e-5 | -8.69e-4 |
| 154 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 258 | +3.12e-4 | +3.12e-4 | +3.12e-4 | -7.51e-4 |
| 155 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 264 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -6.88e-4 |
| 156 | 3.00e-3 | 1 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 272 | -4.60e-5 | -4.60e-5 | -4.60e-5 | -6.24e-4 |
| 157 | 3.00e-3 | 2 | 5.62e-3 | 5.78e-3 | 5.70e-3 | 5.62e-3 | 209 | -1.42e-4 | +1.56e-4 | +6.78e-6 | -5.06e-4 |
| 158 | 3.00e-3 | 1 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 5.17e-3 | 235 | -3.53e-4 | -3.53e-4 | -3.53e-4 | -4.91e-4 |
| 159 | 3.00e-3 | 1 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 5.53e-3 | 230 | +2.92e-4 | +2.92e-4 | +2.92e-4 | -4.12e-4 |
| 160 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 246 | +8.90e-6 | +8.90e-6 | +8.90e-6 | -3.70e-4 |
| 161 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 250 | +6.14e-5 | +6.14e-5 | +6.14e-5 | -3.27e-4 |
| 162 | 3.00e-3 | 2 | 5.78e-3 | 6.31e-3 | 6.04e-3 | 6.31e-3 | 219 | +9.61e-5 | +4.02e-4 | +2.49e-4 | -2.16e-4 |
| 164 | 3.00e-3 | 2 | 5.44e-3 | 6.34e-3 | 5.89e-3 | 6.34e-3 | 219 | -4.99e-4 | +6.99e-4 | +1.00e-4 | -1.50e-4 |
| 165 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 249 | -6.21e-4 | -6.21e-4 | -6.21e-4 | -1.97e-4 |
| 166 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 296 | +2.86e-4 | +2.86e-4 | +2.86e-4 | -1.49e-4 |
| 167 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 312 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -1.10e-4 |
| 168 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 237 | +1.56e-4 | +1.56e-4 | +1.56e-4 | -8.32e-5 |
| 169 | 3.00e-3 | 1 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 238 | -5.92e-4 | -5.92e-4 | -5.92e-4 | -1.34e-4 |
| 170 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 230 | +2.21e-5 | +2.21e-5 | +2.21e-5 | -1.18e-4 |
| 171 | 3.00e-3 | 2 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 189 | -2.99e-6 | +4.47e-5 | +2.09e-5 | -9.22e-5 |
| 172 | 3.00e-3 | 1 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 239 | -3.61e-4 | -3.61e-4 | -3.61e-4 | -1.19e-4 |
| 173 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 240 | +4.51e-4 | +4.51e-4 | +4.51e-4 | -6.21e-5 |
| 174 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 250 | -1.25e-5 | -1.25e-5 | -1.25e-5 | -5.71e-5 |
| 175 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 248 | +1.42e-4 | +1.42e-4 | +1.42e-4 | -3.72e-5 |
| 176 | 3.00e-3 | 2 | 6.23e-3 | 6.30e-3 | 6.26e-3 | 6.30e-3 | 201 | +4.92e-5 | +5.31e-5 | +5.12e-5 | -2.04e-5 |
| 177 | 3.00e-3 | 1 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 5.72e-3 | 237 | -4.06e-4 | -4.06e-4 | -4.06e-4 | -5.90e-5 |
| 178 | 3.00e-3 | 1 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 197 | +2.67e-4 | +2.67e-4 | +2.67e-4 | -2.64e-5 |
| 179 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 213 | -3.83e-4 | -3.83e-4 | -3.83e-4 | -6.21e-5 |
| 180 | 3.00e-3 | 2 | 5.85e-3 | 5.98e-3 | 5.91e-3 | 5.98e-3 | 186 | +1.18e-4 | +2.36e-4 | +1.77e-4 | -1.72e-5 |
| 181 | 3.00e-3 | 1 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 213 | -2.51e-4 | -2.51e-4 | -2.51e-4 | -4.06e-5 |
| 182 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 232 | +2.39e-4 | +2.39e-4 | +2.39e-4 | -1.27e-5 |
| 183 | 3.00e-3 | 2 | 6.03e-3 | 6.08e-3 | 6.05e-3 | 6.08e-3 | 194 | +2.75e-5 | +3.97e-5 | +3.36e-5 | -3.82e-6 |
| 184 | 3.00e-3 | 1 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 5.67e-3 | 258 | -2.67e-4 | -2.67e-4 | -2.67e-4 | -3.01e-5 |
| 185 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 216 | +5.15e-4 | +5.15e-4 | +5.15e-4 | +2.44e-5 |
| 186 | 3.00e-3 | 2 | 5.91e-3 | 6.07e-3 | 5.99e-3 | 6.07e-3 | 170 | -3.26e-4 | +1.60e-4 | -8.27e-5 | +6.47e-6 |
| 187 | 3.00e-3 | 1 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 5.42e-3 | 199 | -5.71e-4 | -5.71e-4 | -5.71e-4 | -5.13e-5 |
| 188 | 3.00e-3 | 2 | 5.83e-3 | 6.03e-3 | 5.93e-3 | 6.03e-3 | 180 | +1.94e-4 | +3.43e-4 | +2.69e-4 | +8.76e-6 |
| 189 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 234 | -3.63e-4 | -3.63e-4 | -3.63e-4 | -2.84e-5 |
| 190 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 240 | +4.46e-4 | +4.46e-4 | +4.46e-4 | +1.90e-5 |
| 191 | 3.00e-3 | 2 | 5.81e-3 | 6.36e-3 | 6.09e-3 | 5.81e-3 | 180 | -5.11e-4 | +1.66e-4 | -1.72e-4 | -2.07e-5 |
| 192 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 197 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -3.87e-5 |
| 193 | 3.00e-3 | 2 | 5.75e-3 | 5.90e-3 | 5.83e-3 | 5.75e-3 | 180 | -1.44e-4 | +2.80e-4 | +6.82e-5 | -2.05e-5 |
| 194 | 3.00e-3 | 1 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 215 | -5.42e-5 | -5.42e-5 | -5.42e-5 | -2.39e-5 |
| 195 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 199 | +3.17e-4 | +3.17e-4 | +3.17e-4 | +1.02e-5 |
| 196 | 3.00e-3 | 2 | 5.86e-3 | 6.01e-3 | 5.94e-3 | 6.01e-3 | 170 | -1.54e-4 | +1.51e-4 | -1.31e-6 | +9.55e-6 |
| 197 | 3.00e-3 | 2 | 5.67e-3 | 5.82e-3 | 5.75e-3 | 5.82e-3 | 156 | -3.09e-4 | +1.71e-4 | -6.89e-5 | -2.96e-6 |
| 198 | 3.00e-3 | 1 | 5.27e-3 | 5.27e-3 | 5.27e-3 | 5.27e-3 | 220 | -4.56e-4 | -4.56e-4 | -4.56e-4 | -4.82e-5 |
| 199 | 3.00e-3 | 2 | 5.88e-3 | 6.22e-3 | 6.05e-3 | 5.88e-3 | 140 | -4.01e-4 | +9.35e-4 | +2.67e-4 | +5.02e-6 |

