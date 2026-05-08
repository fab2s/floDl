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

GPU columns = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | GPU2 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|------|----------|
| nccl-async | 0.063339 | 0.9187 | +0.0062 | 1852.1 | 598 | 38.0 | 100% | 100% | 100% | 11.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9187 | nccl-async | - | - |

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
| nccl-async | 1.9982 | 0.7454 | 0.5574 | 0.5039 | 0.4713 | 0.4517 | 0.4973 | 0.4810 | 0.4728 | 0.4685 | 0.2160 | 0.1791 | 0.1601 | 0.1483 | 0.1442 | 0.0820 | 0.0761 | 0.0723 | 0.0687 | 0.0633 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4067 | 2.8 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.2973 | 3.6 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2960 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 386 | 384 | 384 | 381 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1848.2 | 3.8 | epoch-boundary(199) |
| nccl-async | gpu2 | 1848.2 | 3.8 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.6s |
| resnet-graph | nccl-async | gpu1 | 3.8s | 0.0s | 0.0s | 0.0s | 5.2s |
| resnet-graph | nccl-async | gpu2 | 3.8s | 0.0s | 0.0s | 0.0s | 4.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 174 | 0 | 598 | 38.0 | 1052/10381 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 165.8 | 8.9% |

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
| resnet-graph | nccl-async | 184 | 598 | 0 | 7.54e-3 | +4.29e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 598 | 9.07e-2 | 4.63e-2 | 0.00e0 | 4.28e-1 | 51.3 | -8.18e-5 | 1.75e-3 |
| resnet-graph | nccl-async | 1 | 598 | 9.11e-2 | 4.86e-2 | 0.00e0 | 4.40e-1 | 31.6 | -9.25e-5 | 2.71e-3 |
| resnet-graph | nccl-async | 2 | 598 | 9.04e-2 | 4.85e-2 | 0.00e0 | 4.50e-1 | 17.1 | -9.32e-5 | 2.69e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9860 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9867 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9973 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 72 (0,1,3,4,5,8,9,10…146,147) | 0 (—) | — | 0,1,3,4,5,8,9,10…146,147 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 30 | 30 |
| resnet-graph | nccl-async | 0e0 | 5 | 13 | 13 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 7 | 7 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 496 | +0.046 |
| resnet-graph | nccl-async | 3.00e-2 | 101–149 | 47 | +0.328 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 50 | +0.141 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 595 | +0.011 | 183 | +0.186 | +0.324 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 596 | 3.40e1–8.05e1 | 6.74e1 | 2.21e-3 | 4.25e-3 | 7.08e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 498 | 70–78095 | +8.923e-6 | 0.264 | +9.132e-6 | 0.272 | 95 | +1.511e-5 | 0.700 | 38–1026 | +1.010e-3 | 0.560 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 486 | 900–78095 | +9.041e-6 | 0.300 | +9.231e-6 | 0.308 | 94 | +1.521e-5 | 0.697 | 59–1026 | +1.009e-3 | 0.649 |
| resnet-graph | nccl-async | 3.00e-2 | 101–149 | 48 | 79078–117072 | +1.529e-5 | 0.245 | +1.523e-5 | 0.245 | 43 | +1.493e-5 | 0.224 | 634–1014 | -2.834e-4 | 0.005 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 51 | 117742–155611 | -6.341e-6 | 0.034 | -6.328e-6 | 0.033 | 46 | -7.512e-6 | 0.043 | 629–951 | -8.525e-4 | 0.022 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.010e-3 | r0: +1.000e-3, r1: +1.015e-3, r2: +1.019e-3 | r0: 0.634, r1: 0.517, r2: 0.519 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.009e-3 | r0: +9.992e-4, r1: +1.013e-3, r2: +1.019e-3 | r0: 0.747, r1: 0.594, r2: 0.597 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 101–149 | -2.834e-4 | r0: -2.410e-4, r1: -2.841e-4, r2: -3.239e-4 | r0: 0.003, r1: 0.005, r2: 0.006 | 1.34× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -8.525e-4 | r0: -8.474e-4, r1: -7.985e-4, r2: -9.149e-4 | r0: 0.022, r1: 0.019, r2: 0.025 | 1.15× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇█████████▄▄▄▄▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁` | `▁█▇▇█▇▆█▇▇▆██▇▇█████████▅▆▇▇▇██████▅▆▇▇▇▇█████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 4.50e-1 | 1.08e-1 | 6.59e-2 | 22 | -4.21e-2 | +8.77e-3 | -1.02e-2 | -6.31e-3 |
| 1 | 3.00e-1 | 13 | 6.11e-2 | 1.16e-1 | 7.30e-2 | 8.17e-2 | 27 | -2.42e-2 | +3.20e-2 | +1.07e-3 | -3.62e-4 |
| 2 | 3.00e-1 | 6 | 7.61e-2 | 1.18e-1 | 8.57e-2 | 7.61e-2 | 30 | -1.18e-2 | +1.31e-2 | -3.61e-4 | -5.07e-4 |
| 3 | 3.00e-1 | 8 | 8.68e-2 | 1.30e-1 | 1.01e-1 | 9.67e-2 | 34 | -9.90e-3 | +1.13e-2 | +5.50e-4 | -5.34e-5 |
| 4 | 3.00e-1 | 6 | 9.43e-2 | 1.40e-1 | 1.06e-1 | 1.07e-1 | 39 | -1.17e-2 | +1.11e-2 | +3.69e-4 | +1.23e-4 |
| 5 | 3.00e-1 | 7 | 1.05e-1 | 1.40e-1 | 1.13e-1 | 1.11e-1 | 39 | -7.28e-3 | +6.03e-3 | +7.94e-5 | +7.69e-5 |
| 6 | 3.00e-1 | 7 | 1.03e-1 | 1.41e-1 | 1.10e-1 | 1.03e-1 | 33 | -7.59e-3 | +7.59e-3 | -2.67e-4 | -1.45e-4 |
| 7 | 3.00e-1 | 8 | 9.39e-2 | 1.42e-1 | 1.04e-1 | 1.10e-1 | 31 | -1.16e-2 | +1.22e-2 | +4.71e-4 | +2.69e-4 |
| 8 | 3.00e-1 | 8 | 8.81e-2 | 1.42e-1 | 1.01e-1 | 8.81e-2 | 29 | -1.10e-2 | +9.73e-3 | -7.02e-4 | -3.56e-4 |
| 9 | 3.00e-1 | 8 | 9.20e-2 | 1.34e-1 | 1.03e-1 | 9.39e-2 | 32 | -1.14e-2 | +1.21e-2 | +1.99e-4 | -2.29e-4 |
| 10 | 3.00e-1 | 11 | 8.76e-2 | 1.39e-1 | 9.78e-2 | 8.88e-2 | 29 | -1.42e-2 | +1.22e-2 | -1.47e-4 | -2.94e-4 |
| 11 | 3.00e-1 | 6 | 8.93e-2 | 1.37e-1 | 9.81e-2 | 9.05e-2 | 28 | -1.42e-2 | +1.30e-2 | -8.54e-5 | -2.90e-4 |
| 12 | 3.00e-1 | 9 | 8.75e-2 | 1.34e-1 | 9.62e-2 | 8.88e-2 | 29 | -1.24e-2 | +1.43e-2 | +7.67e-5 | -2.10e-4 |
| 13 | 3.00e-1 | 11 | 8.80e-2 | 1.44e-1 | 9.75e-2 | 8.93e-2 | 27 | -1.64e-2 | +1.58e-2 | +2.62e-5 | -2.22e-4 |
| 14 | 3.00e-1 | 5 | 8.73e-2 | 1.33e-1 | 1.01e-1 | 9.60e-2 | 34 | -1.13e-2 | +1.40e-2 | +7.00e-4 | +5.39e-5 |
| 15 | 3.00e-1 | 9 | 8.54e-2 | 1.36e-1 | 9.62e-2 | 9.77e-2 | 32 | -1.63e-2 | +1.12e-2 | -5.89e-5 | +5.70e-5 |
| 16 | 3.00e-1 | 11 | 8.03e-2 | 1.31e-1 | 9.50e-2 | 9.79e-2 | 28 | -1.15e-2 | +1.08e-2 | +6.09e-5 | +1.90e-4 |
| 17 | 3.00e-1 | 6 | 8.24e-2 | 1.36e-1 | 9.63e-2 | 9.33e-2 | 32 | -1.61e-2 | +1.78e-2 | +3.22e-4 | +1.89e-4 |
| 18 | 3.00e-1 | 10 | 9.00e-2 | 1.40e-1 | 1.00e-1 | 9.45e-2 | 31 | -1.06e-2 | +1.22e-2 | +4.83e-5 | -2.89e-7 |
| 19 | 3.00e-1 | 6 | 8.78e-2 | 1.44e-1 | 9.89e-2 | 9.06e-2 | 29 | -1.56e-2 | +1.45e-2 | -8.98e-5 | -1.21e-4 |
| 20 | 3.00e-1 | 8 | 8.25e-2 | 1.45e-1 | 9.53e-2 | 9.25e-2 | 34 | -1.89e-2 | +1.65e-2 | +5.05e-5 | -4.90e-5 |
| 21 | 3.00e-1 | 8 | 8.67e-2 | 1.41e-1 | 9.78e-2 | 9.43e-2 | 32 | -1.36e-2 | +9.31e-3 | -3.43e-4 | -1.77e-4 |
| 22 | 3.00e-1 | 8 | 8.42e-2 | 1.44e-1 | 9.68e-2 | 9.17e-2 | 32 | -1.60e-2 | +1.35e-2 | -1.64e-4 | -1.90e-4 |
| 23 | 3.00e-1 | 11 | 8.52e-2 | 1.39e-1 | 9.70e-2 | 9.40e-2 | 35 | -1.69e-2 | +1.31e-2 | -5.77e-5 | -1.27e-4 |
| 24 | 3.00e-1 | 5 | 8.91e-2 | 1.46e-1 | 1.07e-1 | 8.91e-2 | 32 | -9.97e-3 | +9.19e-3 | -9.18e-4 | -6.01e-4 |
| 25 | 3.00e-1 | 8 | 8.78e-2 | 1.33e-1 | 9.88e-2 | 9.67e-2 | 35 | -1.18e-2 | +1.08e-2 | +1.92e-4 | -2.01e-4 |
| 26 | 3.00e-1 | 7 | 8.03e-2 | 1.44e-1 | 1.01e-1 | 8.03e-2 | 28 | -8.08e-3 | +9.87e-3 | -1.28e-3 | -9.47e-4 |
| 27 | 3.00e-1 | 9 | 8.04e-2 | 1.38e-1 | 9.53e-2 | 9.18e-2 | 27 | -1.99e-2 | +1.61e-2 | +2.37e-4 | -3.41e-4 |
| 28 | 3.00e-1 | 8 | 8.39e-2 | 1.36e-1 | 9.68e-2 | 1.01e-1 | 34 | -1.37e-2 | +1.49e-2 | +2.39e-4 | +1.07e-5 |
| 29 | 3.00e-1 | 8 | 8.29e-2 | 1.33e-1 | 9.87e-2 | 1.03e-1 | 34 | -1.71e-2 | +1.06e-2 | -1.30e-4 | +1.46e-5 |
| 30 | 3.00e-1 | 10 | 8.98e-2 | 1.37e-1 | 9.94e-2 | 9.55e-2 | 31 | -1.40e-2 | +1.09e-2 | -1.91e-4 | -1.24e-4 |
| 31 | 3.00e-1 | 6 | 8.73e-2 | 1.49e-1 | 1.01e-1 | 9.52e-2 | 32 | -1.77e-2 | +1.76e-2 | +3.19e-4 | -8.25e-6 |
| 32 | 3.00e-1 | 7 | 8.94e-2 | 1.45e-1 | 1.02e-1 | 9.77e-2 | 35 | -1.16e-2 | +1.29e-2 | +1.40e-4 | +8.45e-6 |
| 33 | 3.00e-1 | 8 | 9.19e-2 | 1.46e-1 | 1.03e-1 | 1.02e-1 | 34 | -1.32e-2 | +1.20e-2 | +2.03e-4 | +1.04e-4 |
| 34 | 3.00e-1 | 11 | 8.08e-2 | 1.36e-1 | 9.49e-2 | 8.79e-2 | 31 | -1.33e-2 | +1.09e-2 | -3.83e-4 | -2.55e-4 |
| 35 | 3.00e-1 | 6 | 8.63e-2 | 1.40e-1 | 1.00e-1 | 8.80e-2 | 31 | -1.10e-2 | +1.02e-2 | -6.89e-4 | -5.72e-4 |
| 36 | 3.00e-1 | 8 | 8.58e-2 | 1.40e-1 | 9.80e-2 | 8.62e-2 | 26 | -1.17e-2 | +1.24e-2 | -1.99e-4 | -5.21e-4 |
| 37 | 3.00e-1 | 9 | 8.29e-2 | 1.35e-1 | 9.56e-2 | 1.02e-1 | 28 | -1.75e-2 | +1.60e-2 | +6.67e-4 | +2.51e-4 |
| 38 | 3.00e-1 | 9 | 7.51e-2 | 1.38e-1 | 9.10e-2 | 9.50e-2 | 35 | -1.78e-2 | +1.47e-2 | -5.46e-4 | -4.05e-5 |
| 39 | 3.00e-1 | 8 | 8.64e-2 | 1.38e-1 | 9.97e-2 | 9.14e-2 | 26 | -1.12e-2 | +8.26e-3 | -5.16e-4 | -3.51e-4 |
| 40 | 3.00e-1 | 8 | 8.51e-2 | 1.43e-1 | 9.97e-2 | 1.01e-1 | 36 | -1.70e-2 | +1.75e-2 | +5.42e-4 | +9.09e-5 |
| 41 | 3.00e-1 | 8 | 8.14e-2 | 1.45e-1 | 9.97e-2 | 8.14e-2 | 27 | -1.13e-2 | +9.95e-3 | -1.07e-3 | -7.25e-4 |
| 42 | 3.00e-1 | 9 | 7.44e-2 | 1.50e-1 | 9.31e-2 | 8.04e-2 | 24 | -1.91e-2 | +1.78e-2 | -3.59e-4 | -6.81e-4 |
| 43 | 3.00e-1 | 11 | 8.02e-2 | 1.41e-1 | 9.09e-2 | 8.67e-2 | 26 | -2.02e-2 | +2.04e-2 | +2.78e-4 | -1.86e-4 |
| 44 | 3.00e-1 | 8 | 8.22e-2 | 1.44e-1 | 9.57e-2 | 9.24e-2 | 28 | -1.68e-2 | +1.94e-2 | +3.96e-4 | +4.33e-5 |
| 45 | 3.00e-1 | 9 | 7.23e-2 | 1.44e-1 | 9.07e-2 | 8.71e-2 | 29 | -2.72e-2 | +1.84e-2 | -7.47e-4 | -3.50e-4 |
| 46 | 3.00e-1 | 10 | 7.42e-2 | 1.33e-1 | 8.78e-2 | 9.34e-2 | 29 | -2.46e-2 | +2.12e-2 | +4.30e-4 | +2.57e-4 |
| 47 | 3.00e-1 | 7 | 8.88e-2 | 1.42e-1 | 1.08e-1 | 1.08e-1 | 45 | -1.42e-2 | +1.45e-2 | +6.25e-4 | +3.44e-4 |
| 48 | 3.00e-1 | 7 | 1.04e-1 | 1.43e-1 | 1.15e-1 | 1.15e-1 | 40 | -7.70e-3 | +6.15e-3 | +1.81e-4 | +2.52e-4 |
| 49 | 3.00e-1 | 5 | 1.02e-1 | 1.49e-1 | 1.17e-1 | 1.03e-1 | 40 | -5.69e-3 | +8.67e-3 | -1.27e-4 | -7.45e-6 |
| 50 | 3.00e-1 | 8 | 9.20e-2 | 1.45e-1 | 1.13e-1 | 1.16e-1 | 37 | -1.14e-2 | +8.75e-3 | +3.86e-4 | +1.93e-4 |
| 51 | 3.00e-1 | 5 | 9.33e-2 | 1.40e-1 | 1.05e-1 | 9.95e-2 | 37 | -1.17e-2 | +1.06e-2 | -3.48e-4 | -4.81e-5 |
| 52 | 3.00e-1 | 7 | 8.71e-2 | 1.48e-1 | 1.02e-1 | 9.31e-2 | 35 | -1.42e-2 | +1.05e-2 | -6.14e-4 | -3.70e-4 |
| 53 | 3.00e-1 | 9 | 8.79e-2 | 1.42e-1 | 9.78e-2 | 9.32e-2 | 29 | -1.47e-2 | +1.14e-2 | -2.14e-4 | -2.83e-4 |
| 54 | 3.00e-1 | 8 | 8.73e-2 | 1.39e-1 | 9.84e-2 | 9.65e-2 | 29 | -1.24e-2 | +1.28e-2 | +9.67e-5 | -8.93e-5 |
| 55 | 3.00e-1 | 9 | 7.89e-2 | 1.35e-1 | 9.34e-2 | 9.04e-2 | 28 | -1.97e-2 | +1.36e-2 | -2.91e-4 | -1.91e-4 |
| 56 | 3.00e-1 | 2 | 8.77e-2 | 9.03e-2 | 8.90e-2 | 8.77e-2 | 27 | -1.08e-3 | -6.37e-5 | -5.71e-4 | -2.68e-4 |
| 57 | 3.00e-1 | 1 | 8.30e-2 | 8.30e-2 | 8.30e-2 | 8.30e-2 | 315 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -2.59e-4 |
| 58 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 321 | +3.36e-3 | +3.36e-3 | +3.36e-3 | +1.04e-4 |
| 59 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 297 | -2.64e-4 | -2.64e-4 | -2.64e-4 | +6.70e-5 |
| 60 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 315 | -1.46e-4 | -1.46e-4 | -1.46e-4 | +4.56e-5 |
| 61 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 318 | +9.08e-6 | +9.08e-6 | +9.08e-6 | +4.20e-5 |
| 62 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 294 | +8.75e-6 | +8.75e-6 | +8.75e-6 | +3.87e-5 |
| 64 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 348 | -4.29e-5 | -4.29e-5 | -4.29e-5 | +3.05e-5 |
| 65 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 365 | +9.98e-5 | +9.98e-5 | +9.98e-5 | +3.74e-5 |
| 66 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 341 | +2.78e-5 | +2.78e-5 | +2.78e-5 | +3.65e-5 |
| 67 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 285 | -9.59e-5 | -9.59e-5 | -9.59e-5 | +2.32e-5 |
| 68 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 334 | -1.08e-4 | -1.08e-4 | -1.08e-4 | +1.02e-5 |
| 69 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 294 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +1.97e-5 |
| 71 | 3.00e-1 | 2 | 2.09e-1 | 2.20e-1 | 2.15e-1 | 2.20e-1 | 259 | -1.01e-4 | +1.86e-4 | +4.25e-5 | +2.55e-5 |
| 73 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 329 | -2.90e-4 | -2.90e-4 | -2.90e-4 | -6.09e-6 |
| 74 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 298 | +2.61e-4 | +2.61e-4 | +2.61e-4 | +2.07e-5 |
| 75 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 262 | -1.32e-4 | -1.32e-4 | -1.32e-4 | +5.42e-6 |
| 76 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 303 | -9.13e-5 | -9.13e-5 | -9.13e-5 | -4.25e-6 |
| 77 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 290 | +1.13e-4 | +1.13e-4 | +1.13e-4 | +7.46e-6 |
| 78 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 296 | +1.12e-4 | +1.12e-4 | +1.12e-4 | +1.79e-5 |
| 79 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 335 | -9.42e-5 | -9.42e-5 | -9.42e-5 | +6.69e-6 |
| 80 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 337 | +1.72e-4 | +1.72e-4 | +1.72e-4 | +2.32e-5 |
| 81 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 348 | -5.73e-5 | -5.73e-5 | -5.73e-5 | +1.52e-5 |
| 83 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 375 | +3.02e-5 | +3.02e-5 | +3.02e-5 | +1.67e-5 |
| 84 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 297 | +4.24e-5 | +4.24e-5 | +4.24e-5 | +1.93e-5 |
| 85 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 294 | -1.86e-4 | -1.86e-4 | -1.86e-4 | -1.29e-6 |
| 86 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 284 | -9.02e-6 | -9.02e-6 | -9.02e-6 | -2.06e-6 |
| 87 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 286 | +1.19e-6 | +1.19e-6 | +1.19e-6 | -1.73e-6 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 336 | -2.64e-5 | -2.64e-5 | -2.64e-5 | -4.20e-6 |
| 90 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 350 | +1.67e-4 | +1.67e-4 | +1.67e-4 | +1.29e-5 |
| 91 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 335 | +1.12e-5 | +1.12e-5 | +1.12e-5 | +1.27e-5 |
| 92 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 302 | -4.48e-5 | -4.48e-5 | -4.48e-5 | +6.99e-6 |
| 93 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 281 | -8.73e-5 | -8.73e-5 | -8.73e-5 | -2.44e-6 |
| 94 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 291 | -5.38e-5 | -5.38e-5 | -5.38e-5 | -7.58e-6 |
| 95 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 290 | +1.07e-5 | +1.07e-5 | +1.07e-5 | -5.75e-6 |
| 96 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 301 | -1.35e-5 | -1.35e-5 | -1.35e-5 | -6.52e-6 |
| 97 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 280 | +3.26e-5 | +3.26e-5 | +3.26e-5 | -2.61e-6 |
| 98 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 288 | +1.44e-5 | +1.44e-5 | +1.44e-5 | -9.08e-7 |
| 99 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 297 | -4.72e-5 | -4.72e-5 | -4.72e-5 | -5.54e-6 |
| 101 | 3.00e-2 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 373 | +6.68e-5 | +6.68e-5 | +6.68e-5 | +1.69e-6 |
| 102 | 3.00e-2 | 1 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 3.02e-2 | 288 | -6.83e-3 | -6.83e-3 | -6.83e-3 | -6.82e-4 |
| 103 | 3.00e-2 | 1 | 2.34e-2 | 2.34e-2 | 2.34e-2 | 2.34e-2 | 320 | -8.01e-4 | -8.01e-4 | -8.01e-4 | -6.94e-4 |
| 104 | 3.00e-2 | 1 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 2.52e-2 | 293 | +2.56e-4 | +2.56e-4 | +2.56e-4 | -5.99e-4 |
| 105 | 3.00e-2 | 1 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 319 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -5.29e-4 |
| 106 | 3.00e-2 | 1 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 306 | +2.64e-4 | +2.64e-4 | +2.64e-4 | -4.49e-4 |
| 107 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 264 | -4.73e-5 | -4.73e-5 | -4.73e-5 | -4.09e-4 |
| 108 | 3.00e-2 | 1 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 299 | -4.57e-5 | -4.57e-5 | -4.57e-5 | -3.73e-4 |
| 110 | 3.00e-2 | 1 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 315 | +1.98e-4 | +1.98e-4 | +1.98e-4 | -3.16e-4 |
| 111 | 3.00e-2 | 2 | 3.11e-2 | 3.15e-2 | 3.13e-2 | 3.15e-2 | 264 | +4.94e-5 | +2.01e-4 | +1.25e-4 | -2.33e-4 |
| 113 | 3.00e-2 | 1 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 3.07e-2 | 331 | -7.72e-5 | -7.72e-5 | -7.72e-5 | -2.17e-4 |
| 114 | 3.00e-2 | 1 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 299 | +3.28e-4 | +3.28e-4 | +3.28e-4 | -1.63e-4 |
| 115 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 288 | +9.84e-6 | +9.84e-6 | +9.84e-6 | -1.45e-4 |
| 116 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 286 | +2.06e-5 | +2.06e-5 | +2.06e-5 | -1.29e-4 |
| 117 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 304 | +4.28e-5 | +4.28e-5 | +4.28e-5 | -1.12e-4 |
| 118 | 3.00e-2 | 1 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 288 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -8.77e-5 |
| 119 | 3.00e-2 | 1 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 284 | +5.79e-5 | +5.79e-5 | +5.79e-5 | -7.31e-5 |
| 120 | 3.00e-2 | 1 | 3.68e-2 | 3.68e-2 | 3.68e-2 | 3.68e-2 | 265 | +3.21e-5 | +3.21e-5 | +3.21e-5 | -6.26e-5 |
| 121 | 3.00e-2 | 1 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 3.69e-2 | 318 | +1.47e-5 | +1.47e-5 | +1.47e-5 | -5.49e-5 |
| 122 | 3.00e-2 | 1 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 272 | +2.75e-4 | +2.75e-4 | +2.75e-4 | -2.19e-5 |
| 123 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 290 | -7.49e-5 | -7.49e-5 | -7.49e-5 | -2.72e-5 |
| 125 | 3.00e-2 | 1 | 4.08e-2 | 4.08e-2 | 4.08e-2 | 4.08e-2 | 374 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -1.22e-5 |
| 126 | 3.00e-2 | 2 | 4.22e-2 | 4.54e-2 | 4.38e-2 | 4.22e-2 | 245 | -2.95e-4 | +3.61e-4 | +3.31e-5 | -6.83e-6 |
| 128 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 327 | -1.44e-4 | -1.44e-4 | -1.44e-4 | -2.06e-5 |
| 129 | 3.00e-2 | 2 | 4.26e-2 | 4.55e-2 | 4.41e-2 | 4.26e-2 | 232 | -2.77e-4 | +4.81e-4 | +1.02e-4 | -1.13e-6 |
| 131 | 3.00e-2 | 2 | 4.07e-2 | 4.69e-2 | 4.38e-2 | 4.69e-2 | 232 | -1.48e-4 | +6.11e-4 | +2.31e-4 | +4.69e-5 |
| 132 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 280 | -3.72e-4 | -3.72e-4 | -3.72e-4 | +4.95e-6 |
| 133 | 3.00e-2 | 1 | 4.67e-2 | 4.67e-2 | 4.67e-2 | 4.67e-2 | 272 | +3.63e-4 | +3.63e-4 | +3.63e-4 | +4.08e-5 |
| 134 | 3.00e-2 | 1 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 243 | +7.26e-5 | +7.26e-5 | +7.26e-5 | +4.40e-5 |
| 135 | 3.00e-2 | 1 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 234 | -1.42e-4 | -1.42e-4 | -1.42e-4 | +2.54e-5 |
| 136 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 280 | +3.59e-5 | +3.59e-5 | +3.59e-5 | +2.64e-5 |
| 137 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 252 | +1.73e-4 | +1.73e-4 | +1.73e-4 | +4.11e-5 |
| 138 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 282 | +5.97e-5 | +5.97e-5 | +5.97e-5 | +4.30e-5 |
| 139 | 3.00e-2 | 1 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 5.14e-2 | 265 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +5.41e-5 |
| 140 | 3.00e-2 | 1 | 5.21e-2 | 5.21e-2 | 5.21e-2 | 5.21e-2 | 265 | +5.74e-5 | +5.74e-5 | +5.74e-5 | +5.45e-5 |
| 141 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 265 | -3.85e-5 | -3.85e-5 | -3.85e-5 | +4.52e-5 |
| 142 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 273 | -4.63e-5 | -4.63e-5 | -4.63e-5 | +3.60e-5 |
| 143 | 3.00e-2 | 1 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 289 | +8.71e-5 | +8.71e-5 | +8.71e-5 | +4.11e-5 |
| 144 | 3.00e-2 | 1 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 284 | +2.40e-4 | +2.40e-4 | +2.40e-4 | +6.10e-5 |
| 146 | 3.00e-2 | 1 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 377 | +5.73e-5 | +5.73e-5 | +5.73e-5 | +6.06e-5 |
| 147 | 3.00e-2 | 1 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 6.20e-2 | 263 | +3.06e-4 | +3.06e-4 | +3.06e-4 | +8.52e-5 |
| 148 | 3.00e-2 | 2 | 5.44e-2 | 5.73e-2 | 5.59e-2 | 5.73e-2 | 220 | -4.57e-4 | +2.32e-4 | -1.12e-4 | +5.11e-5 |
| 149 | 3.00e-2 | 1 | 5.25e-2 | 5.25e-2 | 5.25e-2 | 5.25e-2 | 259 | -3.34e-4 | -3.34e-4 | -3.34e-4 | +1.25e-5 |
| 150 | 3.00e-3 | 1 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 227 | +2.32e-4 | +2.32e-4 | +2.32e-4 | +3.44e-5 |
| 151 | 3.00e-3 | 1 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 3.30e-2 | 246 | -2.10e-3 | -2.10e-3 | -2.10e-3 | -1.79e-4 |
| 152 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 266 | -6.71e-3 | -6.71e-3 | -6.71e-3 | -8.33e-4 |
| 153 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 267 | +3.88e-5 | +3.88e-5 | +3.88e-5 | -7.45e-4 |
| 154 | 3.00e-3 | 1 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 295 | -2.17e-5 | -2.17e-5 | -2.17e-5 | -6.73e-4 |
| 155 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 300 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -5.75e-4 |
| 156 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 259 | -1.55e-5 | -1.55e-5 | -1.55e-5 | -5.19e-4 |
| 157 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 261 | -3.12e-4 | -3.12e-4 | -3.12e-4 | -4.99e-4 |
| 158 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 254 | +4.56e-5 | +4.56e-5 | +4.56e-5 | -4.44e-4 |
| 159 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 227 | +9.92e-5 | +9.92e-5 | +9.92e-5 | -3.90e-4 |
| 160 | 3.00e-3 | 2 | 5.59e-3 | 5.75e-3 | 5.67e-3 | 5.75e-3 | 227 | -1.29e-4 | +1.21e-4 | -3.93e-6 | -3.15e-4 |
| 162 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 282 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -3.01e-4 |
| 163 | 3.00e-3 | 2 | 5.75e-3 | 6.15e-3 | 5.95e-3 | 5.75e-3 | 239 | -2.81e-4 | +4.67e-4 | +9.30e-5 | -2.30e-4 |
| 164 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 245 | -7.31e-5 | -7.31e-5 | -7.31e-5 | -2.14e-4 |
| 165 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 272 | +1.33e-4 | +1.33e-4 | +1.33e-4 | -1.79e-4 |
| 166 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 298 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -1.44e-4 |
| 167 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 276 | +9.41e-5 | +9.41e-5 | +9.41e-5 | -1.20e-4 |
| 169 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 360 | -6.79e-5 | -6.79e-5 | -6.79e-5 | -1.15e-4 |
| 170 | 3.00e-3 | 1 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 331 | +3.28e-4 | +3.28e-4 | +3.28e-4 | -7.06e-5 |
| 171 | 3.00e-3 | 2 | 6.04e-3 | 6.73e-3 | 6.39e-3 | 6.04e-3 | 216 | -4.97e-4 | -9.03e-5 | -2.94e-4 | -1.15e-4 |
| 172 | 3.00e-3 | 1 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 224 | -2.10e-4 | -2.10e-4 | -2.10e-4 | -1.25e-4 |
| 173 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 255 | +4.25e-5 | +4.25e-5 | +4.25e-5 | -1.08e-4 |
| 174 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 240 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -8.12e-5 |
| 175 | 3.00e-3 | 1 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 248 | -3.50e-5 | -3.50e-5 | -3.50e-5 | -7.66e-5 |
| 176 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 235 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -5.18e-5 |
| 177 | 3.00e-3 | 1 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 315 | -8.80e-5 | -8.80e-5 | -8.80e-5 | -5.54e-5 |
| 178 | 3.00e-3 | 1 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 310 | +3.05e-4 | +3.05e-4 | +3.05e-4 | -1.94e-5 |
| 179 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 274 | +5.19e-5 | +5.19e-5 | +5.19e-5 | -1.23e-5 |
| 180 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 250 | -1.81e-4 | -1.81e-4 | -1.81e-4 | -2.91e-5 |
| 181 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 244 | -9.92e-5 | -9.92e-5 | -9.92e-5 | -3.61e-5 |
| 182 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 261 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -4.54e-5 |
| 183 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 270 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -2.55e-5 |
| 184 | 3.00e-3 | 1 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 6.60e-3 | 288 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -1.05e-5 |
| 185 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 270 | -1.37e-5 | -1.37e-5 | -1.37e-5 | -1.08e-5 |
| 186 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 269 | -1.80e-5 | -1.80e-5 | -1.80e-5 | -1.15e-5 |
| 187 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 266 | +6.27e-5 | +6.27e-5 | +6.27e-5 | -4.11e-6 |
| 188 | 3.00e-3 | 2 | 6.59e-3 | 6.66e-3 | 6.62e-3 | 6.66e-3 | 247 | -4.04e-5 | +4.37e-5 | +1.64e-6 | -2.59e-6 |
| 190 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 304 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -1.44e-5 |
| 191 | 3.00e-3 | 1 | 7.00e-3 | 7.00e-3 | 7.00e-3 | 7.00e-3 | 254 | +3.40e-4 | +3.40e-4 | +3.40e-4 | +2.10e-5 |
| 192 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 256 | -3.17e-4 | -3.17e-4 | -3.17e-4 | -1.28e-5 |
| 193 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 260 | +3.34e-5 | +3.34e-5 | +3.34e-5 | -8.20e-6 |
| 194 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 253 | +1.92e-6 | +1.92e-6 | +1.92e-6 | -7.19e-6 |
| 195 | 3.00e-3 | 1 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 6.58e-3 | 297 | +3.80e-5 | +3.80e-5 | +3.80e-5 | -2.67e-6 |
| 196 | 3.00e-3 | 2 | 6.69e-3 | 6.81e-3 | 6.75e-3 | 6.69e-3 | 247 | -7.37e-5 | +1.36e-4 | +3.12e-5 | +2.73e-6 |
| 198 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 352 | +3.75e-5 | +3.75e-5 | +3.75e-5 | +6.21e-6 |
| 199 | 3.00e-3 | 1 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 7.54e-3 | 285 | +3.73e-4 | +3.73e-4 | +3.73e-4 | +4.29e-5 |

