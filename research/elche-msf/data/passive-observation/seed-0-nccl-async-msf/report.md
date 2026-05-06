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
| nccl-async | 0.047386 | 0.9196 | +0.0071 | 1913.0 | 876 | 39.8 | 100% | 100% | 4.9 |

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
| nccl-async | 1.9764 | 0.6871 | 0.5216 | 0.4651 | 0.5315 | 0.5012 | 0.4767 | 0.4740 | 0.4613 | 0.4516 | 0.1965 | 0.1602 | 0.1376 | 0.1243 | 0.1150 | 0.0606 | 0.0584 | 0.0526 | 0.0501 | 0.0474 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3963 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3063 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2973 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 402 | 396 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1911.8 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu1 | 1911.9 | 1.0 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.3s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |
| resnet-graph | nccl-async | gpu2 | 1.1s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 412 | 0 | 876 | 39.8 | 661/9228 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 182.8 | 9.6% |

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
| resnet-graph | nccl-async | 194 | 876 | 0 | 6.84e-3 | +2.01e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 876 | 6.33e-2 | 5.22e-2 | 0.00e0 | 4.24e-1 | 41.4 | -1.52e-4 | 4.05e-3 |
| resnet-graph | nccl-async | 1 | 876 | 6.45e-2 | 5.56e-2 | 0.00e0 | 5.37e-1 | 38.8 | -1.64e-4 | 6.21e-3 |
| resnet-graph | nccl-async | 2 | 876 | 6.37e-2 | 5.47e-2 | 0.00e0 | 4.90e-1 | 19.7 | -1.71e-4 | 6.14e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9896 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9905 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9981 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 60 (0,1,2,3,4,5,6,7…147,148) | 1 (23) | 23 | 0,1,2,3,4,5,6,7…147,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 55 | 55 |
| resnet-graph | nccl-async | 0e0 | 5 | 29 | 29 |
| resnet-graph | nccl-async | 0e0 | 10 | 7 | 7 |
| resnet-graph | nccl-async | 1e-4 | 3 | 22 | 22 |
| resnet-graph | nccl-async | 1e-4 | 5 | 13 | 13 |
| resnet-graph | nccl-async | 1e-4 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-3 | 3 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 587 | +0.042 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 98 | +0.075 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 186 | +0.014 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 873 | +0.009 | 193 | +0.166 | +0.175 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 874 | 3.42e1–8.07e1 | 6.49e1 | 1.51e-3 | 3.00e-3 | 4.87e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 589 | 69–78005 | +1.568e-5 | 0.393 | +1.600e-5 | 0.407 | 96 | +1.934e-5 | 0.674 | 27–1046 | +1.445e-3 | 0.655 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 572 | 929–78005 | +1.610e-5 | 0.424 | +1.642e-5 | 0.437 | 95 | +1.941e-5 | 0.669 | 31–1046 | +1.446e-3 | 0.696 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 99 | 78605–117113 | +4.753e-6 | 0.036 | +4.256e-6 | 0.029 | 50 | +4.904e-6 | 0.031 | 154–643 | -3.352e-4 | 0.026 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 187 | 117364–156127 | +5.018e-5 | 0.401 | +5.135e-5 | 0.406 | 48 | +4.200e-5 | 0.491 | 36–1018 | +2.014e-3 | 0.689 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.445e-3 | r0: +1.427e-3, r1: +1.455e-3, r2: +1.458e-3 | r0: 0.719, r1: 0.621, r2: 0.619 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.446e-3 | r0: +1.429e-3, r1: +1.456e-3, r2: +1.458e-3 | r0: 0.767, r1: 0.659, r2: 0.655 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -3.352e-4 | r0: -3.023e-4, r1: -3.510e-4, r2: -3.467e-4 | r0: 0.022, r1: 0.027, r2: 0.027 | 1.16× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.014e-3 | r0: +1.989e-3, r1: +2.017e-3, r2: +2.044e-3 | r0: 0.706, r1: 0.679, r2: 0.674 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇██████████████▆▅▅▅▅▅▅▅▅▅▅▅▄▁▁▁▃▃▃▃▃▃▃▃▃` | `▁█▆▆█▇▅▇▇▇██▇▇▇▇▇▇▇▇▇▇▇▇▅▆▆▇▇▇▇▇▇█▇▇▃▄▇▅▇▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 18 | 0.00e0 | 5.37e-1 | 9.11e-2 | 5.14e-2 | 15 | -1.02e-1 | +1.33e-2 | -1.47e-2 | -7.44e-3 |
| 1 | 3.00e-1 | 16 | 5.56e-2 | 1.22e-1 | 6.79e-2 | 6.01e-2 | 16 | -2.41e-2 | +3.42e-2 | +9.30e-5 | -1.59e-3 |
| 2 | 3.00e-1 | 11 | 5.83e-2 | 1.15e-1 | 7.11e-2 | 7.19e-2 | 17 | -4.24e-2 | +4.32e-2 | +1.24e-3 | +1.56e-4 |
| 3 | 3.00e-1 | 17 | 5.83e-2 | 1.29e-1 | 7.33e-2 | 6.44e-2 | 13 | -5.68e-2 | +4.45e-2 | -5.74e-4 | -9.72e-4 |
| 4 | 3.00e-1 | 17 | 5.99e-2 | 1.40e-1 | 7.21e-2 | 7.89e-2 | 15 | -7.08e-2 | +7.09e-2 | +1.32e-3 | +1.08e-3 |
| 5 | 3.00e-1 | 15 | 6.23e-2 | 1.37e-1 | 7.52e-2 | 7.27e-2 | 17 | -5.03e-2 | +4.64e-2 | -2.33e-4 | +1.52e-4 |
| 6 | 3.00e-1 | 14 | 5.79e-2 | 1.38e-1 | 7.62e-2 | 7.13e-2 | 24 | -4.20e-2 | +4.28e-2 | +2.98e-5 | -6.04e-5 |
| 7 | 3.00e-1 | 10 | 6.35e-2 | 1.43e-1 | 9.24e-2 | 9.36e-2 | 26 | -4.27e-2 | +2.59e-2 | +2.32e-4 | +1.01e-4 |
| 8 | 3.00e-1 | 11 | 7.17e-2 | 1.43e-1 | 8.80e-2 | 7.17e-2 | 20 | -1.74e-2 | +1.83e-2 | -1.03e-3 | -8.88e-4 |
| 9 | 3.00e-1 | 9 | 7.60e-2 | 1.50e-1 | 9.75e-2 | 8.68e-2 | 24 | -1.34e-2 | +2.07e-2 | +3.07e-4 | -4.51e-4 |
| 10 | 3.00e-1 | 10 | 7.16e-2 | 1.52e-1 | 8.60e-2 | 7.49e-2 | 19 | -2.90e-2 | +2.35e-2 | -6.84e-4 | -6.96e-4 |
| 11 | 3.00e-1 | 8 | 7.10e-2 | 1.51e-1 | 1.03e-1 | 9.24e-2 | 30 | -1.22e-2 | +2.16e-2 | +8.51e-4 | -1.51e-4 |
| 12 | 3.00e-1 | 10 | 7.66e-2 | 1.44e-1 | 9.03e-2 | 7.66e-2 | 19 | -1.96e-2 | +1.53e-2 | -1.00e-3 | -8.14e-4 |
| 13 | 3.00e-1 | 14 | 6.43e-2 | 1.39e-1 | 7.41e-2 | 7.09e-2 | 14 | -3.78e-2 | +3.83e-2 | +5.60e-5 | -1.82e-4 |
| 14 | 3.00e-1 | 17 | 5.38e-2 | 1.32e-1 | 6.57e-2 | 5.71e-2 | 14 | -6.42e-2 | +6.00e-2 | -2.34e-4 | -7.17e-4 |
| 15 | 3.00e-1 | 14 | 5.43e-2 | 1.40e-1 | 7.34e-2 | 7.69e-2 | 19 | -6.66e-2 | +6.62e-2 | +1.08e-3 | +4.35e-4 |
| 16 | 3.00e-1 | 17 | 5.25e-2 | 1.36e-1 | 7.10e-2 | 8.82e-2 | 21 | -7.04e-2 | +4.61e-2 | +1.14e-4 | +1.11e-3 |
| 17 | 3.00e-1 | 11 | 6.52e-2 | 1.46e-1 | 7.96e-2 | 7.39e-2 | 18 | -3.67e-2 | +2.73e-2 | -1.14e-3 | -2.11e-4 |
| 18 | 3.00e-1 | 14 | 6.36e-2 | 1.43e-1 | 7.59e-2 | 8.05e-2 | 19 | -3.92e-2 | +3.16e-2 | -3.53e-4 | +1.56e-4 |
| 19 | 3.00e-1 | 15 | 5.52e-2 | 1.39e-1 | 7.27e-2 | 7.36e-2 | 18 | -4.35e-2 | +3.44e-2 | -3.87e-4 | +3.50e-4 |
| 20 | 3.00e-1 | 13 | 6.33e-2 | 1.43e-1 | 7.78e-2 | 7.42e-2 | 19 | -3.86e-2 | +4.18e-2 | +5.13e-4 | +2.61e-4 |
| 21 | 3.00e-1 | 14 | 5.60e-2 | 1.41e-1 | 7.56e-2 | 5.60e-2 | 18 | -3.46e-2 | +3.25e-2 | -8.43e-4 | -1.17e-3 |
| 22 | 3.00e-1 | 17 | 5.82e-2 | 1.41e-1 | 7.24e-2 | 9.36e-2 | 26 | -5.88e-2 | +4.39e-2 | +8.07e-4 | +1.20e-3 |
| 23 | 3.00e-1 | 8 | 7.31e-2 | 1.58e-1 | 9.68e-2 | 7.31e-2 | 20 | -2.03e-2 | +1.86e-2 | -1.38e-3 | -6.15e-4 |
| 24 | 3.00e-1 | 12 | 6.47e-2 | 1.43e-1 | 8.16e-2 | 8.24e-2 | 20 | -3.91e-2 | +3.25e-2 | -5.80e-5 | -1.30e-4 |
| 25 | 3.00e-1 | 20 | 5.33e-2 | 1.49e-1 | 7.17e-2 | 5.33e-2 | 11 | -3.73e-2 | +3.40e-2 | -1.29e-3 | -2.02e-3 |
| 26 | 3.00e-1 | 8 | 4.94e-2 | 1.37e-1 | 7.81e-2 | 9.18e-2 | 25 | -8.72e-2 | +8.48e-2 | +1.97e-3 | +2.69e-6 |
| 27 | 3.00e-1 | 12 | 6.77e-2 | 1.44e-1 | 8.10e-2 | 6.92e-2 | 17 | -2.75e-2 | +2.07e-2 | -1.29e-3 | -9.87e-4 |
| 28 | 3.00e-1 | 16 | 5.65e-2 | 1.41e-1 | 6.84e-2 | 6.65e-2 | 15 | -5.04e-2 | +4.57e-2 | -4.69e-5 | +7.96e-5 |
| 29 | 3.00e-1 | 12 | 5.78e-2 | 1.39e-1 | 8.41e-2 | 7.84e-2 | 21 | -5.14e-2 | +5.24e-2 | +5.69e-4 | -9.33e-5 |
| 30 | 3.00e-1 | 13 | 6.79e-2 | 1.45e-1 | 7.80e-2 | 7.46e-2 | 17 | -3.12e-2 | +2.90e-2 | -2.80e-4 | -8.07e-5 |
| 31 | 3.00e-1 | 15 | 5.65e-2 | 1.57e-1 | 7.19e-2 | 5.79e-2 | 25 | -6.63e-2 | +5.47e-2 | -6.00e-4 | -7.78e-4 |
| 32 | 3.00e-1 | 12 | 6.59e-2 | 1.42e-1 | 8.15e-2 | 8.50e-2 | 23 | -4.15e-2 | +2.58e-2 | +2.75e-4 | +1.46e-4 |
| 33 | 3.00e-1 | 14 | 6.20e-2 | 1.44e-1 | 7.67e-2 | 7.72e-2 | 18 | -3.77e-2 | +2.61e-2 | -5.35e-4 | +1.25e-4 |
| 34 | 3.00e-1 | 14 | 5.97e-2 | 1.43e-1 | 7.53e-2 | 7.06e-2 | 16 | -3.92e-2 | +3.30e-2 | -7.55e-4 | -4.04e-4 |
| 35 | 3.00e-1 | 20 | 5.34e-2 | 1.41e-1 | 7.25e-2 | 7.03e-2 | 18 | -5.54e-2 | +4.76e-2 | -2.54e-4 | -3.74e-4 |
| 36 | 3.00e-1 | 9 | 6.57e-2 | 1.52e-1 | 8.15e-2 | 7.19e-2 | 16 | -4.41e-2 | +4.10e-2 | +1.61e-4 | -2.82e-4 |
| 37 | 3.00e-1 | 14 | 5.59e-2 | 1.42e-1 | 7.29e-2 | 7.43e-2 | 17 | -5.54e-2 | +4.92e-2 | +2.12e-4 | +2.15e-4 |
| 38 | 3.00e-1 | 15 | 5.29e-2 | 1.44e-1 | 7.36e-2 | 6.56e-2 | 16 | -5.63e-2 | +4.87e-2 | -4.97e-4 | -6.07e-4 |
| 39 | 3.00e-1 | 2 | 6.85e-2 | 7.14e-2 | 7.00e-2 | 7.14e-2 | 19 | +2.21e-3 | +2.53e-3 | +2.37e-3 | -4.28e-5 |
| 40 | 3.00e-1 | 1 | 7.60e-2 | 7.60e-2 | 7.60e-2 | 7.60e-2 | 379 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -2.22e-5 |
| 41 | 3.00e-1 | 1 | 2.66e-1 | 2.66e-1 | 2.66e-1 | 2.66e-1 | 345 | +3.63e-3 | +3.63e-3 | +3.63e-3 | +3.43e-4 |
| 42 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 324 | -3.48e-4 | -3.48e-4 | -3.48e-4 | +2.74e-4 |
| 44 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 361 | -1.32e-4 | -1.32e-4 | -1.32e-4 | +2.34e-4 |
| 45 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 312 | +5.81e-5 | +5.81e-5 | +5.81e-5 | +2.16e-4 |
| 46 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 318 | -1.67e-4 | -1.67e-4 | -1.67e-4 | +1.78e-4 |
| 47 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 289 | +3.89e-5 | +3.89e-5 | +3.89e-5 | +1.64e-4 |
| 48 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 332 | -1.17e-4 | -1.17e-4 | -1.17e-4 | +1.36e-4 |
| 49 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 310 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +1.33e-4 |
| 50 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 302 | +2.36e-5 | +2.36e-5 | +2.36e-5 | +1.22e-4 |
| 52 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 366 | -5.74e-5 | -5.74e-5 | -5.74e-5 | +1.04e-4 |
| 53 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 319 | +1.54e-4 | +1.54e-4 | +1.54e-4 | +1.09e-4 |
| 54 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 294 | -1.64e-4 | -1.64e-4 | -1.64e-4 | +8.20e-5 |
| 55 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 306 | -5.36e-5 | -5.36e-5 | -5.36e-5 | +6.85e-5 |
| 56 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 319 | +3.95e-5 | +3.95e-5 | +3.95e-5 | +6.56e-5 |
| 57 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 292 | +3.80e-5 | +3.80e-5 | +3.80e-5 | +6.28e-5 |
| 58 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 309 | -6.53e-5 | -6.53e-5 | -6.53e-5 | +5.00e-5 |
| 59 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 301 | +8.90e-5 | +8.90e-5 | +8.90e-5 | +5.39e-5 |
| 60 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 288 | -3.03e-5 | -3.03e-5 | -3.03e-5 | +4.55e-5 |
| 61 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 280 | -1.11e-4 | -1.11e-4 | -1.11e-4 | +2.98e-5 |
| 62 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 260 | +4.83e-5 | +4.83e-5 | +4.83e-5 | +3.16e-5 |
| 63 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 283 | -9.15e-5 | -9.15e-5 | -9.15e-5 | +1.93e-5 |
| 64 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 267 | +1.34e-4 | +1.34e-4 | +1.34e-4 | +3.08e-5 |
| 65 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 264 | -5.70e-5 | -5.70e-5 | -5.70e-5 | +2.20e-5 |
| 66 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 291 | -3.58e-5 | -3.58e-5 | -3.58e-5 | +1.62e-5 |
| 67 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 272 | +7.03e-5 | +7.03e-5 | +7.03e-5 | +2.17e-5 |
| 68 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 266 | -1.22e-5 | -1.22e-5 | -1.22e-5 | +1.83e-5 |
| 69 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 278 | -1.00e-4 | -1.00e-4 | -1.00e-4 | +6.42e-6 |
| 70 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 260 | +8.53e-5 | +8.53e-5 | +8.53e-5 | +1.43e-5 |
| 71 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 248 | -1.31e-4 | -1.31e-4 | -1.31e-4 | -2.65e-7 |
| 72 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 264 | +1.87e-5 | +1.87e-5 | +1.87e-5 | +1.63e-6 |
| 73 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 265 | +6.35e-5 | +6.35e-5 | +6.35e-5 | +7.82e-6 |
| 74 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 264 | -4.59e-5 | -4.59e-5 | -4.59e-5 | +2.44e-6 |
| 75 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 286 | +5.63e-5 | +5.63e-5 | +5.63e-5 | +7.83e-6 |
| 76 | 3.00e-1 | 2 | 2.10e-1 | 2.17e-1 | 2.13e-1 | 2.10e-1 | 262 | -1.15e-4 | +7.98e-5 | -1.74e-5 | +2.07e-6 |
| 78 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 332 | +4.12e-5 | +4.12e-5 | +4.12e-5 | +5.98e-6 |
| 79 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 278 | +1.72e-4 | +1.72e-4 | +1.72e-4 | +2.26e-5 |
| 80 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 285 | -2.49e-4 | -2.49e-4 | -2.49e-4 | -4.56e-6 |
| 81 | 3.00e-1 | 2 | 2.04e-1 | 2.14e-1 | 2.09e-1 | 2.04e-1 | 210 | -2.35e-4 | +1.18e-4 | -5.87e-5 | -1.66e-5 |
| 82 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 233 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -2.84e-5 |
| 83 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 242 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -3.10e-6 |
| 84 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 244 | -1.35e-6 | -1.35e-6 | -1.35e-6 | -2.93e-6 |
| 85 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 263 | +7.07e-6 | +7.07e-6 | +7.07e-6 | -1.93e-6 |
| 86 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 265 | +9.92e-5 | +9.92e-5 | +9.92e-5 | +8.19e-6 |
| 87 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 260 | +2.98e-5 | +2.98e-5 | +2.98e-5 | +1.03e-5 |
| 88 | 3.00e-1 | 2 | 2.04e-1 | 2.10e-1 | 2.07e-1 | 2.04e-1 | 206 | -1.39e-4 | -1.30e-4 | -1.34e-4 | -1.72e-5 |
| 89 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 296 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -2.60e-5 |
| 90 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 231 | +4.30e-4 | +4.30e-4 | +4.30e-4 | +1.96e-5 |
| 91 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 226 | -3.03e-4 | -3.03e-4 | -3.03e-4 | -1.27e-5 |
| 92 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 233 | -1.59e-5 | -1.59e-5 | -1.59e-5 | -1.30e-5 |
| 93 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 233 | -6.90e-5 | -6.90e-5 | -6.90e-5 | -1.86e-5 |
| 94 | 3.00e-1 | 2 | 2.03e-1 | 2.05e-1 | 2.04e-1 | 2.03e-1 | 190 | -5.96e-5 | +1.21e-4 | +3.09e-5 | -1.01e-5 |
| 95 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 224 | -3.24e-4 | -3.24e-4 | -3.24e-4 | -4.15e-5 |
| 96 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 248 | +3.18e-4 | +3.18e-4 | +3.18e-4 | -5.60e-6 |
| 97 | 3.00e-1 | 2 | 2.02e-1 | 2.09e-1 | 2.06e-1 | 2.02e-1 | 208 | -1.67e-4 | +1.04e-4 | -3.15e-5 | -1.19e-5 |
| 99 | 3.00e-1 | 2 | 1.97e-1 | 2.29e-1 | 2.13e-1 | 2.29e-1 | 193 | -7.46e-5 | +7.86e-4 | +3.56e-4 | +6.23e-5 |
| 100 | 3.00e-2 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 216 | -8.54e-4 | -8.54e-4 | -8.54e-4 | -2.94e-5 |
| 101 | 3.00e-2 | 2 | 1.96e-2 | 1.41e-1 | 8.05e-2 | 1.96e-2 | 179 | -1.10e-2 | -1.45e-3 | -6.25e-3 | -1.26e-3 |
| 102 | 3.00e-2 | 1 | 1.93e-2 | 1.93e-2 | 1.93e-2 | 1.93e-2 | 232 | -6.52e-5 | -6.52e-5 | -6.52e-5 | -1.14e-3 |
| 103 | 3.00e-2 | 1 | 2.25e-2 | 2.25e-2 | 2.25e-2 | 2.25e-2 | 222 | +6.88e-4 | +6.88e-4 | +6.88e-4 | -9.57e-4 |
| 104 | 3.00e-2 | 2 | 2.28e-2 | 2.30e-2 | 2.29e-2 | 2.28e-2 | 179 | -4.87e-5 | +1.11e-4 | +3.13e-5 | -7.70e-4 |
| 105 | 3.00e-2 | 1 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 232 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -7.05e-4 |
| 106 | 3.00e-2 | 1 | 2.53e-2 | 2.53e-2 | 2.53e-2 | 2.53e-2 | 225 | +5.88e-4 | +5.88e-4 | +5.88e-4 | -5.76e-4 |
| 107 | 3.00e-2 | 1 | 2.56e-2 | 2.56e-2 | 2.56e-2 | 2.56e-2 | 225 | +5.58e-5 | +5.58e-5 | +5.58e-5 | -5.13e-4 |
| 108 | 3.00e-2 | 2 | 2.55e-2 | 2.69e-2 | 2.62e-2 | 2.69e-2 | 181 | -1.87e-5 | +3.09e-4 | +1.45e-4 | -3.86e-4 |
| 109 | 3.00e-2 | 1 | 2.51e-2 | 2.51e-2 | 2.51e-2 | 2.51e-2 | 192 | -3.71e-4 | -3.71e-4 | -3.71e-4 | -3.85e-4 |
| 110 | 3.00e-2 | 2 | 2.56e-2 | 2.72e-2 | 2.64e-2 | 2.72e-2 | 178 | +9.91e-5 | +3.46e-4 | +2.23e-4 | -2.68e-4 |
| 111 | 3.00e-2 | 1 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 247 | -6.31e-5 | -6.31e-5 | -6.31e-5 | -2.48e-4 |
| 112 | 3.00e-2 | 1 | 3.03e-2 | 3.03e-2 | 3.03e-2 | 3.03e-2 | 210 | +5.87e-4 | +5.87e-4 | +5.87e-4 | -1.64e-4 |
| 113 | 3.00e-2 | 2 | 2.83e-2 | 3.02e-2 | 2.92e-2 | 3.02e-2 | 164 | -3.07e-4 | +3.94e-4 | +4.34e-5 | -1.21e-4 |
| 114 | 3.00e-2 | 1 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 197 | -5.60e-4 | -5.60e-4 | -5.60e-4 | -1.65e-4 |
| 115 | 3.00e-2 | 2 | 2.98e-2 | 3.04e-2 | 3.01e-2 | 3.04e-2 | 145 | +1.37e-4 | +5.08e-4 | +3.22e-4 | -7.43e-5 |
| 116 | 3.00e-2 | 2 | 2.64e-2 | 2.86e-2 | 2.75e-2 | 2.86e-2 | 158 | -7.77e-4 | +5.04e-4 | -1.36e-4 | -7.96e-5 |
| 117 | 3.00e-2 | 2 | 2.89e-2 | 3.08e-2 | 2.98e-2 | 3.08e-2 | 151 | +5.21e-5 | +4.25e-4 | +2.39e-4 | -1.73e-5 |
| 118 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 194 | -2.32e-4 | -2.32e-4 | -2.32e-4 | -3.87e-5 |
| 119 | 3.00e-2 | 1 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 213 | +2.99e-4 | +2.99e-4 | +2.99e-4 | -4.96e-6 |
| 120 | 3.00e-2 | 2 | 3.21e-2 | 3.34e-2 | 3.27e-2 | 3.21e-2 | 148 | -2.83e-4 | +3.38e-4 | +2.76e-5 | -1.88e-6 |
| 121 | 3.00e-2 | 2 | 2.95e-2 | 3.28e-2 | 3.12e-2 | 3.28e-2 | 147 | -4.73e-4 | +7.12e-4 | +1.20e-4 | +2.71e-5 |
| 122 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 196 | -2.98e-4 | -2.98e-4 | -2.98e-4 | -5.40e-6 |
| 123 | 3.00e-2 | 3 | 3.25e-2 | 3.53e-2 | 3.41e-2 | 3.25e-2 | 147 | -4.13e-4 | +7.46e-4 | +5.82e-5 | +9.35e-7 |
| 124 | 3.00e-2 | 1 | 3.25e-2 | 3.25e-2 | 3.25e-2 | 3.25e-2 | 179 | +5.82e-6 | +5.82e-6 | +5.82e-6 | +1.42e-6 |
| 125 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 178 | +5.85e-4 | +5.85e-4 | +5.85e-4 | +5.98e-5 |
| 126 | 3.00e-2 | 2 | 3.54e-2 | 3.98e-2 | 3.76e-2 | 3.98e-2 | 137 | -8.50e-5 | +8.55e-4 | +3.85e-4 | +1.26e-4 |
| 127 | 3.00e-2 | 2 | 3.31e-2 | 3.70e-2 | 3.51e-2 | 3.70e-2 | 129 | -1.02e-3 | +8.54e-4 | -8.46e-5 | +9.56e-5 |
| 128 | 3.00e-2 | 2 | 3.17e-2 | 3.71e-2 | 3.44e-2 | 3.71e-2 | 129 | -8.83e-4 | +1.22e-3 | +1.68e-4 | +1.20e-4 |
| 129 | 3.00e-2 | 2 | 3.28e-2 | 3.93e-2 | 3.60e-2 | 3.93e-2 | 129 | -6.44e-4 | +1.41e-3 | +3.82e-4 | +1.80e-4 |
| 130 | 3.00e-2 | 2 | 3.37e-2 | 3.70e-2 | 3.53e-2 | 3.70e-2 | 123 | -9.31e-4 | +7.62e-4 | -8.42e-5 | +1.38e-4 |
| 131 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 151 | -5.26e-4 | -5.26e-4 | -5.26e-4 | +7.19e-5 |
| 132 | 3.00e-2 | 3 | 3.50e-2 | 3.77e-2 | 3.65e-2 | 3.50e-2 | 122 | -5.99e-4 | +4.97e-4 | +2.86e-5 | +4.97e-5 |
| 133 | 3.00e-2 | 1 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 166 | +7.86e-5 | +7.86e-5 | +7.86e-5 | +5.26e-5 |
| 134 | 3.00e-2 | 2 | 4.04e-2 | 4.16e-2 | 4.10e-2 | 4.16e-2 | 113 | +2.60e-4 | +6.88e-4 | +4.74e-4 | +1.30e-4 |
| 135 | 3.00e-2 | 3 | 3.44e-2 | 3.79e-2 | 3.59e-2 | 3.55e-2 | 113 | -1.39e-3 | +8.32e-4 | -3.84e-4 | -1.90e-6 |
| 136 | 3.00e-2 | 2 | 3.49e-2 | 4.12e-2 | 3.80e-2 | 4.12e-2 | 113 | -1.01e-4 | +1.46e-3 | +6.81e-4 | +1.36e-4 |
| 137 | 3.00e-2 | 2 | 3.65e-2 | 4.15e-2 | 3.90e-2 | 4.15e-2 | 113 | -7.51e-4 | +1.14e-3 | +1.94e-4 | +1.56e-4 |
| 138 | 3.00e-2 | 2 | 3.65e-2 | 4.11e-2 | 3.88e-2 | 4.11e-2 | 117 | -7.77e-4 | +1.00e-3 | +1.13e-4 | +1.57e-4 |
| 139 | 3.00e-2 | 2 | 3.74e-2 | 4.37e-2 | 4.05e-2 | 4.37e-2 | 103 | -5.99e-4 | +1.51e-3 | +4.54e-4 | +2.24e-4 |
| 140 | 3.00e-2 | 4 | 3.34e-2 | 4.20e-2 | 3.62e-2 | 3.34e-2 | 90 | -2.01e-3 | +2.02e-3 | -5.64e-4 | -4.97e-5 |
| 141 | 3.00e-2 | 2 | 3.45e-2 | 4.09e-2 | 3.77e-2 | 4.09e-2 | 90 | +2.39e-4 | +1.91e-3 | +1.08e-3 | +1.73e-4 |
| 142 | 3.00e-2 | 2 | 3.50e-2 | 4.12e-2 | 3.81e-2 | 4.12e-2 | 92 | -1.20e-3 | +1.77e-3 | +2.80e-4 | +2.08e-4 |
| 143 | 3.00e-2 | 4 | 3.41e-2 | 4.34e-2 | 3.74e-2 | 3.78e-2 | 89 | -2.48e-3 | +2.67e-3 | -4.78e-5 | +1.30e-4 |
| 144 | 3.00e-2 | 2 | 3.44e-2 | 4.51e-2 | 3.97e-2 | 4.51e-2 | 90 | -6.34e-4 | +3.02e-3 | +1.19e-3 | +3.50e-4 |
| 145 | 3.00e-2 | 3 | 3.21e-2 | 3.98e-2 | 3.58e-2 | 3.21e-2 | 67 | -3.20e-3 | +1.55e-3 | -1.22e-3 | -8.69e-5 |
| 146 | 3.00e-2 | 4 | 3.24e-2 | 3.80e-2 | 3.42e-2 | 3.29e-2 | 72 | -1.80e-3 | +2.07e-3 | +1.81e-5 | -7.32e-5 |
| 147 | 3.00e-2 | 3 | 3.14e-2 | 4.51e-2 | 3.65e-2 | 3.14e-2 | 57 | -6.34e-3 | +4.84e-3 | -4.92e-4 | -2.50e-4 |
| 148 | 3.00e-2 | 5 | 2.82e-2 | 3.79e-2 | 3.21e-2 | 3.53e-2 | 57 | -5.21e-3 | +4.59e-3 | +4.66e-4 | +8.00e-5 |
| 149 | 3.00e-2 | 4 | 2.89e-2 | 4.02e-2 | 3.28e-2 | 2.89e-2 | 52 | -4.63e-3 | +4.62e-3 | -7.08e-4 | -2.35e-4 |
| 150 | 3.00e-3 | 4 | 2.62e-3 | 3.12e-2 | 1.52e-2 | 2.62e-3 | 50 | -4.23e-2 | +7.60e-4 | -1.20e-2 | -4.45e-3 |
| 151 | 3.00e-3 | 6 | 2.34e-3 | 3.70e-3 | 2.68e-3 | 2.34e-3 | 43 | -9.38e-3 | +7.51e-3 | -4.90e-4 | -2.67e-3 |
| 152 | 3.00e-3 | 9 | 1.87e-3 | 3.30e-3 | 2.24e-3 | 1.92e-3 | 26 | -1.25e-2 | +8.20e-3 | -7.79e-4 | -1.56e-3 |
| 153 | 3.00e-3 | 7 | 1.53e-3 | 2.91e-3 | 1.81e-3 | 1.60e-3 | 25 | -2.08e-2 | +2.48e-2 | +2.25e-4 | -7.94e-4 |
| 154 | 3.00e-3 | 12 | 1.23e-3 | 2.73e-3 | 1.59e-3 | 1.36e-3 | 23 | -3.04e-2 | +2.37e-2 | -1.01e-3 | -1.06e-3 |
| 155 | 3.00e-3 | 12 | 1.16e-3 | 2.67e-3 | 1.49e-3 | 1.31e-3 | 20 | -2.15e-2 | +2.33e-2 | -9.73e-4 | -1.02e-3 |
| 156 | 3.00e-3 | 18 | 1.08e-3 | 2.50e-3 | 1.31e-3 | 1.62e-3 | 21 | -4.00e-2 | +3.10e-2 | +1.47e-4 | +8.90e-4 |
| 157 | 3.00e-3 | 9 | 1.05e-3 | 2.48e-3 | 1.48e-3 | 1.50e-3 | 20 | -4.10e-2 | +3.53e-2 | +1.30e-4 | +5.19e-4 |
| 158 | 3.00e-3 | 15 | 1.01e-3 | 2.68e-3 | 1.30e-3 | 1.08e-3 | 17 | -6.52e-2 | +4.54e-2 | -1.23e-3 | -1.12e-3 |
| 159 | 3.00e-3 | 18 | 1.09e-3 | 2.88e-3 | 1.40e-3 | 1.16e-3 | 16 | -5.64e-2 | +5.89e-2 | +3.61e-4 | -9.30e-4 |
| 160 | 3.00e-3 | 10 | 9.87e-4 | 2.81e-3 | 1.28e-3 | 1.12e-3 | 15 | -7.46e-2 | +6.38e-2 | -3.80e-4 | -8.13e-4 |
| 161 | 3.00e-3 | 16 | 8.93e-4 | 2.73e-3 | 1.22e-3 | 8.93e-4 | 16 | -6.00e-2 | +6.00e-2 | -1.03e-3 | -2.04e-3 |
| 162 | 3.00e-3 | 14 | 1.18e-3 | 2.47e-3 | 1.46e-3 | 1.24e-3 | 18 | -3.69e-2 | +4.93e-2 | +8.74e-4 | -8.16e-4 |
| 163 | 3.00e-3 | 1 | 1.28e-3 | 1.28e-3 | 1.28e-3 | 1.28e-3 | 17 | +1.87e-3 | +1.87e-3 | +1.87e-3 | -5.48e-4 |
| 164 | 3.00e-3 | 2 | 1.14e-3 | 6.43e-3 | 3.79e-3 | 6.43e-3 | 275 | -3.58e-4 | +6.28e-3 | +2.96e-3 | +1.52e-4 |
| 166 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 382 | -7.28e-5 | -7.28e-5 | -7.28e-5 | +1.30e-4 |
| 167 | 3.00e-3 | 1 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 7.11e-3 | 303 | +4.22e-4 | +4.22e-4 | +4.22e-4 | +1.59e-4 |
| 168 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 307 | -3.46e-4 | -3.46e-4 | -3.46e-4 | +1.08e-4 |
| 169 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 305 | +7.37e-5 | +7.37e-5 | +7.37e-5 | +1.05e-4 |
| 170 | 3.00e-3 | 1 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 287 | +7.13e-5 | +7.13e-5 | +7.13e-5 | +1.01e-4 |
| 171 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 290 | -9.89e-6 | -9.89e-6 | -9.89e-6 | +9.03e-5 |
| 172 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 255 | -1.79e-4 | -1.79e-4 | -1.79e-4 | +6.34e-5 |
| 173 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 267 | -1.19e-4 | -1.19e-4 | -1.19e-4 | +4.51e-5 |
| 174 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 273 | +4.46e-5 | +4.46e-5 | +4.46e-5 | +4.51e-5 |
| 175 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 259 | +1.90e-5 | +1.90e-5 | +1.90e-5 | +4.25e-5 |
| 176 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 276 | +9.15e-6 | +9.15e-6 | +9.15e-6 | +3.91e-5 |
| 177 | 3.00e-3 | 1 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 6.50e-3 | 330 | +1.07e-4 | +1.07e-4 | +1.07e-4 | +4.59e-5 |
| 179 | 3.00e-3 | 1 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 6.80e-3 | 349 | +1.27e-4 | +1.27e-4 | +1.27e-4 | +5.40e-5 |
| 180 | 3.00e-3 | 1 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 375 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +5.91e-5 |
| 181 | 3.00e-3 | 1 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 7.30e-3 | 326 | +9.63e-5 | +9.63e-5 | +9.63e-5 | +6.28e-5 |
| 182 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 260 | -1.79e-4 | -1.79e-4 | -1.79e-4 | +3.86e-5 |
| 183 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 263 | -4.30e-4 | -4.30e-4 | -4.30e-4 | -8.27e-6 |
| 184 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 250 | +1.37e-4 | +1.37e-4 | +1.37e-4 | +6.24e-6 |
| 185 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 262 | -7.17e-5 | -7.17e-5 | -7.17e-5 | -1.55e-6 |
| 186 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 273 | +2.81e-5 | +2.81e-5 | +2.81e-5 | +1.41e-6 |
| 187 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 278 | +1.14e-5 | +1.14e-5 | +1.14e-5 | +2.41e-6 |
| 188 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 288 | +8.20e-5 | +8.20e-5 | +8.20e-5 | +1.04e-5 |
| 189 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 288 | +5.83e-5 | +5.83e-5 | +5.83e-5 | +1.52e-5 |
| 190 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 286 | +5.12e-6 | +5.12e-6 | +5.12e-6 | +1.42e-5 |
| 191 | 3.00e-3 | 1 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 270 | -4.86e-5 | -4.86e-5 | -4.86e-5 | +7.88e-6 |
| 192 | 3.00e-3 | 1 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 276 | -1.92e-6 | -1.92e-6 | -1.92e-6 | +6.90e-6 |
| 193 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 272 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -4.66e-6 |
| 194 | 3.00e-3 | 2 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 232 | -1.43e-6 | +8.18e-5 | +4.02e-5 | +3.45e-6 |
| 195 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 254 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -9.28e-6 |
| 196 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 250 | -1.21e-6 | -1.21e-6 | -1.21e-6 | -8.48e-6 |
| 197 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 271 | +3.08e-5 | +3.08e-5 | +3.08e-5 | -4.55e-6 |
| 198 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 268 | +1.47e-4 | +1.47e-4 | +1.47e-4 | +1.06e-5 |
| 199 | 3.00e-3 | 1 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 319 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +2.01e-5 |

