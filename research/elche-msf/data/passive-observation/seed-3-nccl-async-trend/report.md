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
| nccl-async | 0.062922 | 0.9155 | +0.0030 | 1988.7 | 710 | 42.3 | 100% | 100% | 8.3 |

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
| nccl-async | 1.9726 | 0.7682 | 0.6041 | 0.5512 | 0.5139 | 0.4947 | 0.4920 | 0.4729 | 0.4661 | 0.4698 | 0.2145 | 0.1784 | 0.1659 | 0.1436 | 0.1316 | 0.0785 | 0.0745 | 0.0680 | 0.0680 | 0.0629 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4009 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3062 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2929 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 400 | 390 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1987.2 | 1.5 | epoch-boundary(199) |
| nccl-async | gpu2 | 1987.2 | 1.5 | epoch-boundary(199) |
| nccl-async | gpu0 | 1987.6 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.5s |
| resnet-graph | nccl-async | gpu1 | 1.5s | 0.0s | 0.0s | 0.0s | 2.9s |
| resnet-graph | nccl-async | gpu2 | 1.5s | 0.0s | 0.0s | 0.0s | 2.9s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 407 | 0 | 710 | 42.3 | 849/9237 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 196.6 | 9.9% |

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
| resnet-graph | nccl-async | 192 | 710 | 0 | 5.95e-3 | +3.68e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 710 | 4.15e-2 | 6.87e-2 | 0.00e0 | 3.68e-1 | 44.6 | -1.29e-4 | 3.54e-3 |
| resnet-graph | nccl-async | 1 | 710 | 4.25e-2 | 7.09e-2 | 0.00e0 | 4.38e-1 | 35.8 | -1.57e-4 | 5.54e-3 |
| resnet-graph | nccl-async | 2 | 710 | 4.18e-2 | 6.94e-2 | 0.00e0 | 3.44e-1 | 19.6 | -1.62e-4 | 5.55e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9987 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9990 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9985 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 39 (0,1,2,3,4,11,22,23…149,150) | 2 (157,171) | — | 0,1,2,3,4,11,22,23…149,150 | 157,171 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 56 | 56 |
| resnet-graph | nccl-async | 0e0 | 5 | 26 | 26 |
| resnet-graph | nccl-async | 0e0 | 10 | 9 | 9 |
| resnet-graph | nccl-async | 1e-4 | 3 | 27 | 27 |
| resnet-graph | nccl-async | 1e-4 | 5 | 13 | 13 |
| resnet-graph | nccl-async | 1e-4 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-3 | 3 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 163 | +0.111 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 104 | +0.127 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 438 | +0.007 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 707 | -0.009 | 191 | +0.113 | +0.147 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 708 | 3.34e1–7.94e1 | 6.12e1 | 1.74e-3 | 5.74e-3 | 9.24e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 165 | 62–77672 | +1.329e-5 | 0.462 | +1.370e-5 | 0.472 | 92 | +3.981e-6 | 0.165 | 28–983 | +1.440e-3 | 0.768 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 150 | 909–77672 | +1.247e-5 | 0.470 | +1.278e-5 | 0.474 | 91 | +3.388e-6 | 0.139 | 32–983 | +1.466e-3 | 0.843 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 105 | 78256–117042 | +2.259e-6 | 0.008 | +1.841e-6 | 0.005 | 50 | +5.895e-6 | 0.047 | 124–671 | -1.818e-4 | 0.008 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 439 | 117249–155877 | +1.749e-5 | 0.113 | +1.801e-5 | 0.118 | 50 | +3.960e-5 | 0.520 | 33–650 | +2.884e-3 | 0.479 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.440e-3 | r0: +1.417e-3, r1: +1.423e-3, r2: +1.484e-3 | r0: 0.780, r1: 0.745, r2: 0.774 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.466e-3 | r0: +1.443e-3, r1: +1.463e-3, r2: +1.496e-3 | r0: 0.854, r1: 0.828, r2: 0.842 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -1.818e-4 | r0: -2.032e-4, r1: -1.505e-4, r2: -1.876e-4 | r0: 0.010, r1: 0.005, r2: 0.008 | 1.35× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.884e-3 | r0: +2.834e-3, r1: +2.902e-3, r2: +2.931e-3 | r0: 0.523, r1: 0.452, r2: 0.450 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `███████████████████████▅▄▅▅▅▅▅▅▅▅▅▅▄▁▁▁▁▁▁▁▁▁▂▂▂` | `▂▇▇▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▄▅▅▆▆▆▆▆▆▆▆▆▁▅▇▅▆█▆▇▅▆▇▆▆` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 16 | 0.00e0 | 4.38e-1 | 9.68e-2 | 6.96e-2 | 15 | -6.44e-2 | +1.80e-2 | -1.22e-2 | -6.26e-3 |
| 1 | 3.00e-1 | 18 | 4.56e-2 | 1.11e-1 | 6.61e-2 | 8.38e-2 | 22 | -4.55e-2 | +5.39e-2 | +1.56e-3 | +6.64e-4 |
| 2 | 3.00e-1 | 8 | 7.58e-2 | 1.36e-1 | 8.88e-2 | 8.61e-2 | 22 | -2.46e-2 | +1.82e-2 | -6.62e-4 | -3.42e-5 |
| 3 | 3.00e-1 | 14 | 6.57e-2 | 1.34e-1 | 8.06e-2 | 6.90e-2 | 17 | -2.79e-2 | +2.49e-2 | -7.89e-4 | -8.20e-4 |
| 4 | 3.00e-1 | 2 | 7.89e-2 | 8.02e-2 | 7.95e-2 | 7.89e-2 | 236 | -6.59e-5 | +8.32e-3 | +4.13e-3 | +8.07e-5 |
| 5 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 287 | +3.59e-3 | +3.59e-3 | +3.59e-3 | +4.33e-4 |
| 6 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 296 | -1.43e-4 | -1.43e-4 | -1.43e-4 | +3.75e-4 |
| 7 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 252 | -7.74e-5 | -7.74e-5 | -7.74e-5 | +3.30e-4 |
| 8 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 239 | -2.54e-4 | -2.54e-4 | -2.54e-4 | +2.71e-4 |
| 9 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 268 | -5.68e-5 | -5.68e-5 | -5.68e-5 | +2.38e-4 |
| 10 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 256 | +6.12e-5 | +6.12e-5 | +6.12e-5 | +2.21e-4 |
| 11 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 274 | +4.39e-5 | +4.39e-5 | +4.39e-5 | +2.03e-4 |
| 12 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 274 | -9.38e-5 | -9.38e-5 | -9.38e-5 | +1.73e-4 |
| 13 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 260 | +9.37e-5 | +9.37e-5 | +9.37e-5 | +1.65e-4 |
| 14 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 270 | -4.24e-5 | -4.24e-5 | -4.24e-5 | +1.44e-4 |
| 15 | 3.00e-1 | 2 | 1.93e-1 | 1.96e-1 | 1.95e-1 | 1.93e-1 | 233 | -6.62e-5 | +8.75e-6 | -2.87e-5 | +1.11e-4 |
| 17 | 3.00e-1 | 2 | 1.88e-1 | 2.05e-1 | 1.97e-1 | 2.05e-1 | 230 | -7.90e-5 | +3.72e-4 | +1.47e-4 | +1.20e-4 |
| 19 | 3.00e-1 | 2 | 1.85e-1 | 2.04e-1 | 1.95e-1 | 2.04e-1 | 230 | -3.40e-4 | +4.32e-4 | +4.63e-5 | +1.10e-4 |
| 20 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 243 | -4.17e-4 | -4.17e-4 | -4.17e-4 | +5.72e-5 |
| 21 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 250 | +1.66e-4 | +1.66e-4 | +1.66e-4 | +6.81e-5 |
| 22 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 280 | +2.55e-6 | +2.55e-6 | +2.55e-6 | +6.16e-5 |
| 23 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 261 | +1.71e-4 | +1.71e-4 | +1.71e-4 | +7.25e-5 |
| 24 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 298 | -4.69e-5 | -4.69e-5 | -4.69e-5 | +6.06e-5 |
| 25 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 247 | +1.04e-4 | +1.04e-4 | +1.04e-4 | +6.50e-5 |
| 26 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 241 | -3.15e-4 | -3.15e-4 | -3.15e-4 | +2.70e-5 |
| 27 | 3.00e-1 | 2 | 1.90e-1 | 1.92e-1 | 1.91e-1 | 1.90e-1 | 206 | -3.42e-5 | +5.86e-5 | +1.22e-5 | +2.37e-5 |
| 28 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 265 | -1.95e-4 | -1.95e-4 | -1.95e-4 | +1.80e-6 |
| 29 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 270 | +3.61e-4 | +3.61e-4 | +3.61e-4 | +3.78e-5 |
| 30 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 299 | +6.47e-6 | +6.47e-6 | +6.47e-6 | +3.46e-5 |
| 31 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 240 | +8.23e-5 | +8.23e-5 | +8.23e-5 | +3.94e-5 |
| 32 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 250 | -2.28e-4 | -2.28e-4 | -2.28e-4 | +1.27e-5 |
| 33 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 246 | +3.06e-5 | +3.06e-5 | +3.06e-5 | +1.45e-5 |
| 34 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 236 | +3.10e-5 | +3.10e-5 | +3.10e-5 | +1.61e-5 |
| 35 | 3.00e-1 | 2 | 1.90e-1 | 1.94e-1 | 1.92e-1 | 1.90e-1 | 196 | -1.10e-4 | -3.54e-5 | -7.26e-5 | -1.12e-6 |
| 36 | 3.00e-1 | 2 | 1.78e-1 | 1.86e-1 | 1.82e-1 | 1.86e-1 | 177 | -3.06e-4 | +2.55e-4 | -2.55e-5 | -2.96e-6 |
| 37 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 200 | -3.25e-4 | -3.25e-4 | -3.25e-4 | -3.51e-5 |
| 38 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 210 | +3.29e-4 | +3.29e-4 | +3.29e-4 | +1.26e-6 |
| 39 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 243 | -2.26e-6 | -2.26e-6 | -2.26e-6 | +9.11e-7 |
| 40 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 264 | +2.12e-4 | +2.12e-4 | +2.12e-4 | +2.20e-5 |
| 41 | 3.00e-1 | 2 | 1.91e-1 | 1.97e-1 | 1.94e-1 | 1.91e-1 | 202 | -1.75e-4 | -1.04e-5 | -9.28e-5 | -6.14e-7 |
| 42 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 238 | -2.37e-4 | -2.37e-4 | -2.37e-4 | -2.43e-5 |
| 43 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 239 | +2.94e-4 | +2.94e-4 | +2.94e-4 | +7.52e-6 |
| 44 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 243 | +7.67e-6 | +7.67e-6 | +7.67e-6 | +7.53e-6 |
| 45 | 3.00e-1 | 2 | 1.89e-1 | 1.94e-1 | 1.92e-1 | 1.89e-1 | 202 | -1.28e-4 | +2.08e-5 | -5.34e-5 | -4.79e-6 |
| 46 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 233 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -1.76e-5 |
| 47 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 255 | +2.05e-4 | +2.05e-4 | +2.05e-4 | +4.64e-6 |
| 48 | 3.00e-1 | 2 | 1.85e-1 | 2.00e-1 | 1.93e-1 | 1.85e-1 | 185 | -4.02e-4 | +1.49e-4 | -1.27e-4 | -2.31e-5 |
| 49 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 213 | -2.33e-4 | -2.33e-4 | -2.33e-4 | -4.41e-5 |
| 50 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 213 | +3.69e-4 | +3.69e-4 | +3.69e-4 | -2.77e-6 |
| 51 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 222 | -2.34e-5 | -2.34e-5 | -2.34e-5 | -4.83e-6 |
| 52 | 3.00e-1 | 2 | 1.91e-1 | 1.97e-1 | 1.94e-1 | 1.97e-1 | 168 | +1.44e-5 | +1.87e-4 | +1.00e-4 | +1.60e-5 |
| 53 | 3.00e-1 | 2 | 1.71e-1 | 1.84e-1 | 1.77e-1 | 1.84e-1 | 168 | -6.84e-4 | +4.26e-4 | -1.29e-4 | -5.98e-6 |
| 54 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 206 | -2.18e-4 | -2.18e-4 | -2.18e-4 | -2.72e-5 |
| 55 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 189 | +3.24e-4 | +3.24e-4 | +3.24e-4 | +7.97e-6 |
| 56 | 3.00e-1 | 2 | 1.76e-1 | 1.83e-1 | 1.80e-1 | 1.76e-1 | 168 | -2.47e-4 | -1.07e-4 | -1.77e-4 | -2.79e-5 |
| 57 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 292 | +2.67e-5 | +2.67e-5 | +2.67e-5 | -2.24e-5 |
| 58 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 313 | +5.79e-4 | +5.79e-4 | +5.79e-4 | +3.77e-5 |
| 60 | 3.00e-1 | 2 | 2.11e-1 | 2.21e-1 | 2.16e-1 | 2.21e-1 | 266 | -1.44e-5 | +1.68e-4 | +7.69e-5 | +4.61e-5 |
| 62 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 367 | -2.69e-4 | -2.69e-4 | -2.69e-4 | +1.45e-5 |
| 63 | 3.00e-1 | 2 | 1.98e-1 | 2.20e-1 | 2.09e-1 | 1.98e-1 | 246 | -4.19e-4 | +3.71e-4 | -2.43e-5 | +3.22e-6 |
| 65 | 3.00e-1 | 2 | 1.98e-1 | 2.17e-1 | 2.08e-1 | 2.17e-1 | 246 | -9.15e-6 | +3.83e-4 | +1.87e-4 | +4.01e-5 |
| 67 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 296 | -3.54e-4 | -3.54e-4 | -3.54e-4 | +7.05e-7 |
| 68 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 259 | +2.70e-4 | +2.70e-4 | +2.70e-4 | +2.76e-5 |
| 69 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 300 | -1.18e-4 | -1.18e-4 | -1.18e-4 | +1.31e-5 |
| 70 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 302 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +2.24e-5 |
| 71 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 270 | -5.37e-7 | -5.37e-7 | -5.37e-7 | +2.01e-5 |
| 72 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 284 | -1.24e-4 | -1.24e-4 | -1.24e-4 | +5.69e-6 |
| 73 | 3.00e-1 | 2 | 1.99e-1 | 2.07e-1 | 2.03e-1 | 1.99e-1 | 229 | -1.87e-4 | +1.09e-4 | -3.91e-5 | -4.31e-6 |
| 75 | 3.00e-1 | 2 | 1.94e-1 | 2.14e-1 | 2.04e-1 | 2.14e-1 | 229 | -7.31e-5 | +4.15e-4 | +1.71e-4 | +3.14e-5 |
| 76 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 269 | -3.67e-4 | -3.67e-4 | -3.67e-4 | -8.36e-6 |
| 77 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 250 | +2.31e-4 | +2.31e-4 | +2.31e-4 | +1.56e-5 |
| 78 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 249 | -8.94e-5 | -8.94e-5 | -8.94e-5 | +5.11e-6 |
| 79 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 254 | -2.81e-5 | -2.81e-5 | -2.81e-5 | +1.79e-6 |
| 80 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 268 | +6.27e-5 | +6.27e-5 | +6.27e-5 | +7.87e-6 |
| 81 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 292 | +7.38e-5 | +7.38e-5 | +7.38e-5 | +1.45e-5 |
| 82 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 257 | +7.76e-5 | +7.76e-5 | +7.76e-5 | +2.08e-5 |
| 83 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 260 | -1.93e-4 | -1.93e-4 | -1.93e-4 | -5.66e-7 |
| 84 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 260 | +2.46e-5 | +2.46e-5 | +2.46e-5 | +1.95e-6 |
| 85 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 262 | +1.05e-5 | +1.05e-5 | +1.05e-5 | +2.81e-6 |
| 86 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 248 | +1.17e-5 | +1.17e-5 | +1.17e-5 | +3.70e-6 |
| 87 | 3.00e-1 | 2 | 2.02e-1 | 2.06e-1 | 2.04e-1 | 2.06e-1 | 254 | -2.51e-5 | +7.21e-5 | +2.35e-5 | +7.94e-6 |
| 88 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 250 | -7.20e-5 | -7.20e-5 | -7.20e-5 | -5.69e-8 |
| 89 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 252 | -4.24e-5 | -4.24e-5 | -4.24e-5 | -4.29e-6 |
| 90 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 217 | -4.17e-5 | -4.17e-5 | -4.17e-5 | -8.03e-6 |
| 91 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 225 | -8.97e-5 | -8.97e-5 | -8.97e-5 | -1.62e-5 |
| 92 | 3.00e-1 | 2 | 1.89e-1 | 1.96e-1 | 1.92e-1 | 1.89e-1 | 192 | -1.72e-4 | +3.98e-5 | -6.62e-5 | -2.68e-5 |
| 93 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 266 | -1.40e-4 | -1.40e-4 | -1.40e-4 | -3.81e-5 |
| 94 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 299 | +4.39e-4 | +4.39e-4 | +4.39e-4 | +9.59e-6 |
| 95 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 288 | +8.41e-5 | +8.41e-5 | +8.41e-5 | +1.70e-5 |
| 96 | 3.00e-1 | 2 | 1.99e-1 | 2.06e-1 | 2.03e-1 | 1.99e-1 | 192 | -1.77e-4 | -1.44e-4 | -1.61e-4 | -1.69e-5 |
| 98 | 3.00e-1 | 2 | 1.85e-1 | 2.14e-1 | 1.99e-1 | 2.14e-1 | 170 | -2.51e-4 | +8.61e-4 | +3.05e-4 | +4.99e-5 |
| 99 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 194 | -1.13e-3 | -1.13e-3 | -1.13e-3 | -6.82e-5 |
| 100 | 3.00e-2 | 2 | 1.85e-1 | 1.90e-1 | 1.88e-1 | 1.90e-1 | 170 | +1.72e-4 | +3.58e-4 | +2.65e-4 | -5.88e-6 |
| 101 | 3.00e-2 | 1 | 1.85e-2 | 1.85e-2 | 1.85e-2 | 1.85e-2 | 209 | -1.12e-2 | -1.12e-2 | -1.12e-2 | -1.12e-3 |
| 102 | 3.00e-2 | 1 | 2.01e-2 | 2.01e-2 | 2.01e-2 | 2.01e-2 | 213 | +4.01e-4 | +4.01e-4 | +4.01e-4 | -9.69e-4 |
| 103 | 3.00e-2 | 2 | 2.13e-2 | 2.21e-2 | 2.17e-2 | 2.21e-2 | 179 | +2.03e-4 | +2.75e-4 | +2.39e-4 | -7.40e-4 |
| 104 | 3.00e-2 | 1 | 2.15e-2 | 2.15e-2 | 2.15e-2 | 2.15e-2 | 216 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -6.79e-4 |
| 105 | 3.00e-2 | 2 | 2.37e-2 | 2.38e-2 | 2.38e-2 | 2.37e-2 | 179 | -2.22e-5 | +4.88e-4 | +2.33e-4 | -5.09e-4 |
| 106 | 3.00e-2 | 1 | 2.28e-2 | 2.28e-2 | 2.28e-2 | 2.28e-2 | 230 | -1.67e-4 | -1.67e-4 | -1.67e-4 | -4.74e-4 |
| 107 | 3.00e-2 | 1 | 2.54e-2 | 2.54e-2 | 2.54e-2 | 2.54e-2 | 233 | +4.61e-4 | +4.61e-4 | +4.61e-4 | -3.81e-4 |
| 108 | 3.00e-2 | 2 | 2.64e-2 | 2.79e-2 | 2.72e-2 | 2.79e-2 | 164 | +1.56e-4 | +3.29e-4 | +2.42e-4 | -2.62e-4 |
| 109 | 3.00e-2 | 1 | 2.41e-2 | 2.41e-2 | 2.41e-2 | 2.41e-2 | 195 | -7.57e-4 | -7.57e-4 | -7.57e-4 | -3.11e-4 |
| 110 | 3.00e-2 | 2 | 2.57e-2 | 2.70e-2 | 2.64e-2 | 2.70e-2 | 187 | +2.65e-4 | +3.42e-4 | +3.04e-4 | -1.95e-4 |
| 111 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 227 | -9.61e-5 | -9.61e-5 | -9.61e-5 | -1.85e-4 |
| 112 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 220 | +4.87e-4 | +4.87e-4 | +4.87e-4 | -1.18e-4 |
| 113 | 3.00e-2 | 2 | 2.96e-2 | 2.98e-2 | 2.97e-2 | 2.96e-2 | 144 | -3.89e-5 | +5.26e-5 | +6.85e-6 | -9.45e-5 |
| 114 | 3.00e-2 | 2 | 2.63e-2 | 2.66e-2 | 2.64e-2 | 2.66e-2 | 155 | -7.56e-4 | +7.03e-5 | -3.43e-4 | -1.38e-4 |
| 115 | 3.00e-2 | 1 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 184 | +2.05e-5 | +2.05e-5 | +2.05e-5 | -1.22e-4 |
| 116 | 3.00e-2 | 2 | 2.96e-2 | 3.11e-2 | 3.03e-2 | 3.11e-2 | 164 | +2.86e-4 | +5.35e-4 | +4.10e-4 | -2.19e-5 |
| 117 | 3.00e-2 | 1 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 2.90e-2 | 217 | -3.15e-4 | -3.15e-4 | -3.15e-4 | -5.12e-5 |
| 118 | 3.00e-2 | 2 | 3.13e-2 | 3.19e-2 | 3.16e-2 | 3.13e-2 | 153 | -1.31e-4 | +5.02e-4 | +1.85e-4 | -9.41e-6 |
| 119 | 3.00e-2 | 1 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 185 | -6.09e-4 | -6.09e-4 | -6.09e-4 | -6.93e-5 |
| 120 | 3.00e-2 | 2 | 3.09e-2 | 3.31e-2 | 3.20e-2 | 3.31e-2 | 157 | +4.35e-4 | +5.10e-4 | +4.72e-4 | +3.32e-5 |
| 121 | 3.00e-2 | 2 | 3.10e-2 | 3.36e-2 | 3.23e-2 | 3.36e-2 | 153 | -3.34e-4 | +5.15e-4 | +9.05e-5 | +4.84e-5 |
| 122 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 194 | -4.74e-4 | -4.74e-4 | -4.74e-4 | -3.84e-6 |
| 123 | 3.00e-2 | 2 | 3.46e-2 | 3.49e-2 | 3.48e-2 | 3.49e-2 | 142 | +6.62e-5 | +6.27e-4 | +3.47e-4 | +5.99e-5 |
| 124 | 3.00e-2 | 2 | 3.09e-2 | 3.37e-2 | 3.23e-2 | 3.37e-2 | 142 | -7.52e-4 | +6.11e-4 | -7.02e-5 | +4.20e-5 |
| 125 | 3.00e-2 | 1 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 170 | -3.95e-4 | -3.95e-4 | -3.95e-4 | -1.65e-6 |
| 126 | 3.00e-2 | 2 | 3.41e-2 | 3.62e-2 | 3.51e-2 | 3.62e-2 | 146 | +4.14e-4 | +4.21e-4 | +4.17e-4 | +7.79e-5 |
| 127 | 3.00e-2 | 2 | 3.30e-2 | 3.37e-2 | 3.34e-2 | 3.37e-2 | 125 | -6.02e-4 | +1.87e-4 | -2.07e-4 | +2.76e-5 |
| 128 | 3.00e-2 | 2 | 3.16e-2 | 3.43e-2 | 3.30e-2 | 3.43e-2 | 125 | -4.33e-4 | +6.51e-4 | +1.09e-4 | +4.84e-5 |
| 129 | 3.00e-2 | 3 | 3.10e-2 | 3.49e-2 | 3.27e-2 | 3.22e-2 | 125 | -7.02e-4 | +9.42e-4 | -1.34e-4 | -9.17e-7 |
| 130 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 148 | +6.95e-5 | +6.95e-5 | +6.95e-5 | +6.12e-6 |
| 131 | 3.00e-2 | 2 | 3.60e-2 | 3.82e-2 | 3.71e-2 | 3.82e-2 | 126 | +4.58e-4 | +5.86e-4 | +5.22e-4 | +1.04e-4 |
| 132 | 3.00e-2 | 3 | 3.35e-2 | 3.80e-2 | 3.51e-2 | 3.35e-2 | 125 | -1.02e-3 | +9.88e-4 | -2.91e-4 | -5.76e-6 |
| 133 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 169 | +1.01e-4 | +1.01e-4 | +1.01e-4 | +4.86e-6 |
| 134 | 3.00e-2 | 2 | 3.99e-2 | 4.01e-2 | 4.00e-2 | 4.01e-2 | 117 | +4.49e-5 | +9.03e-4 | +4.74e-4 | +8.97e-5 |
| 135 | 3.00e-2 | 3 | 3.37e-2 | 3.64e-2 | 3.48e-2 | 3.37e-2 | 109 | -1.18e-3 | +5.82e-4 | -4.34e-4 | -4.84e-5 |
| 136 | 3.00e-2 | 2 | 3.26e-2 | 3.92e-2 | 3.59e-2 | 3.92e-2 | 109 | -2.24e-4 | +1.68e-3 | +7.30e-4 | +1.09e-4 |
| 137 | 3.00e-2 | 2 | 3.33e-2 | 4.05e-2 | 3.69e-2 | 4.05e-2 | 120 | -1.09e-3 | +1.65e-3 | +2.78e-4 | +1.55e-4 |
| 138 | 3.00e-2 | 3 | 3.41e-2 | 4.04e-2 | 3.71e-2 | 3.41e-2 | 99 | -1.71e-3 | +9.41e-4 | -4.67e-4 | -2.45e-5 |
| 139 | 3.00e-2 | 2 | 3.34e-2 | 3.73e-2 | 3.54e-2 | 3.73e-2 | 94 | -1.79e-4 | +1.19e-3 | +5.07e-4 | +8.34e-5 |
| 140 | 3.00e-2 | 4 | 3.36e-2 | 4.01e-2 | 3.60e-2 | 3.57e-2 | 103 | -1.55e-3 | +1.83e-3 | -4.36e-5 | +3.92e-5 |
| 141 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 143 | +7.85e-5 | +7.85e-5 | +7.85e-5 | +4.31e-5 |
| 142 | 3.00e-2 | 4 | 3.34e-2 | 4.11e-2 | 3.69e-2 | 3.40e-2 | 80 | -1.86e-3 | +1.03e-3 | -2.87e-4 | -8.50e-5 |
| 143 | 3.00e-2 | 2 | 3.05e-2 | 3.74e-2 | 3.40e-2 | 3.74e-2 | 80 | -1.03e-3 | +2.56e-3 | +7.64e-4 | +9.42e-5 |
| 144 | 3.00e-2 | 3 | 3.18e-2 | 4.31e-2 | 3.64e-2 | 3.44e-2 | 78 | -2.89e-3 | +3.51e-3 | -1.82e-4 | +1.09e-6 |
| 145 | 3.00e-2 | 4 | 2.83e-2 | 3.74e-2 | 3.27e-2 | 2.83e-2 | 57 | -3.00e-3 | +1.45e-3 | -9.13e-4 | -3.56e-4 |
| 146 | 3.00e-2 | 5 | 2.85e-2 | 3.65e-2 | 3.10e-2 | 3.08e-2 | 65 | -3.76e-3 | +4.11e-3 | +2.53e-4 | -1.31e-4 |
| 147 | 3.00e-2 | 3 | 3.01e-2 | 3.83e-2 | 3.30e-2 | 3.01e-2 | 66 | -3.64e-3 | +3.55e-3 | -6.52e-5 | -1.49e-4 |
| 148 | 3.00e-2 | 5 | 2.67e-2 | 3.99e-2 | 3.16e-2 | 2.90e-2 | 51 | -5.49e-3 | +4.04e-3 | -3.21e-4 | -2.49e-4 |
| 149 | 3.00e-2 | 5 | 2.57e-2 | 3.63e-2 | 2.87e-2 | 2.57e-2 | 47 | -6.85e-3 | +4.63e-3 | -5.20e-4 | -4.03e-4 |
| 150 | 3.00e-3 | 6 | 2.33e-3 | 3.72e-2 | 1.23e-2 | 2.33e-3 | 40 | -5.96e-2 | +7.39e-3 | -9.57e-3 | -4.54e-3 |
| 151 | 3.00e-3 | 9 | 1.89e-3 | 3.37e-3 | 2.14e-3 | 1.89e-3 | 26 | -1.87e-2 | +1.19e-2 | -7.46e-4 | -2.22e-3 |
| 152 | 3.00e-3 | 10 | 1.45e-3 | 3.29e-3 | 1.82e-3 | 1.62e-3 | 24 | -1.98e-2 | +2.27e-2 | -5.17e-4 | -1.15e-3 |
| 153 | 3.00e-3 | 14 | 1.13e-3 | 2.71e-3 | 1.47e-3 | 1.37e-3 | 20 | -3.18e-2 | +2.08e-2 | -8.28e-4 | -5.44e-4 |
| 154 | 3.00e-3 | 15 | 1.10e-3 | 2.67e-3 | 1.42e-3 | 1.42e-3 | 19 | -5.82e-2 | +4.49e-2 | +6.14e-5 | -1.09e-5 |
| 155 | 3.00e-3 | 20 | 1.13e-3 | 2.75e-3 | 1.39e-3 | 1.36e-3 | 13 | -4.48e-2 | +4.20e-2 | -6.32e-5 | +1.42e-4 |
| 156 | 3.00e-3 | 10 | 1.05e-3 | 2.67e-3 | 1.36e-3 | 1.61e-3 | 24 | -5.48e-2 | +6.24e-2 | +1.62e-3 | +1.23e-3 |
| 157 | 3.00e-3 | 15 | 1.41e-3 | 3.30e-3 | 1.74e-3 | 1.69e-3 | 27 | -1.77e-2 | +2.15e-2 | -2.46e-4 | +2.67e-4 |
| 158 | 3.00e-3 | 7 | 1.31e-3 | 2.88e-3 | 1.70e-3 | 1.90e-3 | 27 | -4.02e-2 | +2.61e-2 | -3.59e-4 | +2.26e-4 |
| 159 | 3.00e-3 | 12 | 1.42e-3 | 3.05e-3 | 1.71e-3 | 1.42e-3 | 18 | -3.00e-2 | +2.40e-2 | -1.06e-3 | -8.51e-4 |
| 160 | 3.00e-3 | 11 | 1.26e-3 | 2.96e-3 | 1.67e-3 | 2.96e-3 | 53 | -4.08e-2 | +3.54e-2 | +1.03e-3 | +9.22e-4 |
| 161 | 3.00e-3 | 7 | 2.07e-3 | 3.37e-3 | 2.49e-3 | 2.32e-3 | 34 | -8.79e-3 | +3.40e-3 | -1.06e-3 | -1.40e-6 |
| 162 | 3.00e-3 | 10 | 1.61e-3 | 3.15e-3 | 1.92e-3 | 1.63e-3 | 20 | -1.32e-2 | +1.33e-2 | -1.31e-3 | -8.04e-4 |
| 163 | 3.00e-3 | 16 | 1.30e-3 | 2.59e-3 | 1.80e-3 | 1.45e-3 | 22 | -3.62e-2 | +2.16e-2 | -9.42e-4 | -1.22e-3 |
| 164 | 3.00e-3 | 10 | 1.16e-3 | 3.05e-3 | 1.52e-3 | 1.31e-3 | 15 | -5.76e-2 | +4.25e-2 | -9.38e-4 | -1.14e-3 |
| 165 | 3.00e-3 | 13 | 1.10e-3 | 2.69e-3 | 1.51e-3 | 1.58e-3 | 23 | -4.98e-2 | +5.50e-2 | +1.81e-3 | +6.48e-4 |
| 166 | 3.00e-3 | 21 | 1.07e-3 | 3.16e-3 | 1.44e-3 | 1.37e-3 | 15 | -3.46e-2 | +3.34e-2 | -3.82e-4 | +2.93e-4 |
| 167 | 3.00e-3 | 11 | 1.07e-3 | 2.90e-3 | 1.38e-3 | 1.24e-3 | 17 | -4.93e-2 | +5.93e-2 | -8.66e-5 | -1.56e-4 |
| 168 | 3.00e-3 | 15 | 9.68e-4 | 3.15e-3 | 1.36e-3 | 1.42e-3 | 17 | -4.82e-2 | +5.46e-2 | +5.34e-4 | +6.21e-4 |
| 169 | 3.00e-3 | 15 | 1.21e-3 | 3.19e-3 | 1.50e-3 | 1.47e-3 | 16 | -6.02e-2 | +4.89e-2 | +1.25e-4 | +3.73e-4 |
| 170 | 3.00e-3 | 15 | 1.09e-3 | 3.24e-3 | 1.47e-3 | 1.67e-3 | 17 | -5.67e-2 | +5.15e-2 | +2.49e-4 | +8.95e-4 |
| 171 | 3.00e-3 | 10 | 1.34e-3 | 3.05e-3 | 1.89e-3 | 2.07e-3 | 27 | -4.31e-2 | +4.77e-2 | +1.47e-3 | +1.01e-3 |
| 172 | 3.00e-3 | 18 | 1.19e-3 | 3.34e-3 | 1.64e-3 | 1.31e-3 | 16 | -2.29e-2 | +2.35e-2 | -1.18e-3 | -8.36e-4 |
| 173 | 3.00e-3 | 10 | 1.30e-3 | 3.11e-3 | 1.57e-3 | 1.43e-3 | 15 | -4.96e-2 | +5.14e-2 | +6.04e-4 | -1.67e-4 |
| 174 | 3.00e-3 | 14 | 1.26e-3 | 2.93e-3 | 1.60e-3 | 1.50e-3 | 17 | -5.56e-2 | +5.61e-2 | +6.03e-4 | +3.92e-5 |
| 175 | 3.00e-3 | 20 | 1.12e-3 | 3.00e-3 | 1.50e-3 | 1.54e-3 | 20 | -5.24e-2 | +4.46e-2 | +1.29e-4 | +3.43e-4 |
| 176 | 3.00e-3 | 5 | 1.57e-3 | 3.83e-3 | 2.35e-3 | 2.10e-3 | 23 | -1.93e-2 | +2.97e-2 | +1.77e-3 | +6.77e-4 |
| 177 | 3.00e-3 | 11 | 1.60e-3 | 3.18e-3 | 1.90e-3 | 1.93e-3 | 19 | -2.69e-2 | +2.53e-2 | -1.17e-4 | +3.65e-4 |
| 178 | 3.00e-3 | 9 | 1.39e-3 | 3.29e-3 | 2.16e-3 | 2.78e-3 | 48 | -3.97e-2 | +3.88e-2 | +5.79e-4 | +5.69e-4 |
| 179 | 3.00e-3 | 4 | 2.69e-3 | 3.84e-3 | 3.12e-3 | 2.69e-3 | 51 | -5.85e-3 | +4.56e-3 | -4.66e-4 | +1.31e-4 |
| 180 | 3.00e-3 | 6 | 2.82e-3 | 4.11e-3 | 3.20e-3 | 2.82e-3 | 40 | -4.83e-3 | +5.47e-3 | -1.76e-4 | -1.14e-4 |
| 181 | 3.00e-3 | 7 | 2.38e-3 | 3.59e-3 | 2.68e-3 | 2.54e-3 | 34 | -1.11e-2 | +6.40e-3 | -4.26e-4 | -2.57e-4 |
| 182 | 3.00e-3 | 11 | 1.66e-3 | 3.57e-3 | 2.05e-3 | 1.66e-3 | 19 | -1.89e-2 | +1.36e-2 | -1.69e-3 | -1.21e-3 |
| 183 | 3.00e-3 | 16 | 1.09e-3 | 3.21e-3 | 1.46e-3 | 1.51e-3 | 18 | -4.86e-2 | +3.96e-2 | -7.02e-4 | +8.34e-5 |
| 184 | 3.00e-3 | 14 | 1.30e-3 | 3.34e-3 | 1.95e-3 | 2.12e-3 | 27 | -5.53e-2 | +4.64e-2 | -5.43e-4 | -3.16e-4 |
| 185 | 3.00e-3 | 2 | 2.20e-3 | 2.24e-3 | 2.22e-3 | 2.24e-3 | 28 | +7.90e-4 | +1.11e-3 | +9.51e-4 | -7.71e-5 |
| 186 | 3.00e-3 | 1 | 2.16e-3 | 2.16e-3 | 2.16e-3 | 2.16e-3 | 230 | -1.56e-4 | -1.56e-4 | -1.56e-4 | -8.50e-5 |
| 187 | 3.00e-3 | 2 | 6.59e-3 | 6.64e-3 | 6.61e-3 | 6.64e-3 | 191 | +3.80e-5 | +4.72e-3 | +2.38e-3 | +3.60e-4 |
| 188 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 216 | -2.91e-4 | -2.91e-4 | -2.91e-4 | +2.95e-4 |
| 189 | 3.00e-3 | 2 | 6.37e-3 | 6.95e-3 | 6.66e-3 | 6.95e-3 | 179 | +8.93e-5 | +4.91e-4 | +2.90e-4 | +2.96e-4 |
| 190 | 3.00e-3 | 1 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 229 | -5.85e-4 | -5.85e-4 | -5.85e-4 | +2.08e-4 |
| 191 | 3.00e-3 | 1 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 6.52e-3 | 214 | +3.22e-4 | +3.22e-4 | +3.22e-4 | +2.19e-4 |
| 192 | 3.00e-3 | 2 | 6.27e-3 | 6.38e-3 | 6.32e-3 | 6.27e-3 | 166 | -1.11e-4 | -9.99e-5 | -1.06e-4 | +1.57e-4 |
| 193 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 208 | -2.67e-4 | -2.67e-4 | -2.67e-4 | +1.15e-4 |
| 194 | 3.00e-3 | 2 | 6.33e-3 | 6.39e-3 | 6.36e-3 | 6.39e-3 | 166 | +4.97e-5 | +3.40e-4 | +1.95e-4 | +1.29e-4 |
| 195 | 3.00e-3 | 1 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 5.82e-3 | 197 | -4.69e-4 | -4.69e-4 | -4.69e-4 | +6.90e-5 |
| 196 | 3.00e-3 | 2 | 6.18e-3 | 6.23e-3 | 6.21e-3 | 6.23e-3 | 166 | +4.96e-5 | +3.13e-4 | +1.81e-4 | +8.90e-5 |
| 197 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 205 | -1.42e-4 | -1.42e-4 | -1.42e-4 | +6.59e-5 |
| 198 | 3.00e-3 | 2 | 6.14e-3 | 6.47e-3 | 6.31e-3 | 6.47e-3 | 170 | +6.99e-5 | +3.08e-4 | +1.89e-4 | +9.04e-5 |
| 199 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 190 | -4.45e-4 | -4.45e-4 | -4.45e-4 | +3.68e-5 |

