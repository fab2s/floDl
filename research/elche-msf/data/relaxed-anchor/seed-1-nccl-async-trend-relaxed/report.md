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
| nccl-async | 0.051590 | 0.9180 | +0.0055 | 2062.1 | 598 | 44.8 | 100% | 100% | 7.3 |

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
| nccl-async | 1.9739 | 0.7521 | 0.5791 | 0.5013 | 0.4685 | 0.5242 | 0.4902 | 0.4740 | 0.4571 | 0.4552 | 0.2007 | 0.1614 | 0.1418 | 0.1368 | 0.1285 | 0.0683 | 0.0614 | 0.0586 | 0.0550 | 0.0516 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3928 | 2.3 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3068 | 3.1 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.3004 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 394 | 392 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 2060.8 | 1.3 | epoch-boundary(199) |
| nccl-async | gpu2 | 2060.9 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu0 | 2060.9 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 2.2s |
| resnet-graph | nccl-async | gpu1 | 1.3s | 0.0s | 0.0s | 0.0s | 2.9s |
| resnet-graph | nccl-async | gpu2 | 1.2s | 0.0s | 0.0s | 0.0s | 2.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 170 | 0 | 598 | 44.8 | 1207/11289 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 224.5 | 10.9% |

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
| resnet-graph | nccl-async | 188 | 598 | 0 | 6.63e-3 | -1.95e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 598 | 9.32e-2 | 4.71e-2 | 0.00e0 | 3.89e-1 | 40.5 | -8.27e-5 | 1.85e-3 |
| resnet-graph | nccl-async | 1 | 598 | 9.52e-2 | 4.98e-2 | 0.00e0 | 3.53e-1 | 39.3 | -8.94e-5 | 2.87e-3 |
| resnet-graph | nccl-async | 2 | 598 | 9.44e-2 | 4.97e-2 | 0.00e0 | 3.69e-1 | 20.2 | -8.72e-5 | 2.83e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9855 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9867 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9976 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 64 (0,1,2,3,4,5,6,7…147,148) | 0 (—) | — | 0,1,2,3,4,5,6,7…147,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 36 | 36 |
| resnet-graph | nccl-async | 0e0 | 5 | 16 | 16 |
| resnet-graph | nccl-async | 0e0 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 3 | 11 | 11 |
| resnet-graph | nccl-async | 1e-4 | 5 | 6 | 6 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 494 | +0.043 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 55 | +0.172 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 44 | +0.087 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 595 | +0.013 | 187 | +0.138 | +0.334 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 596 | 3.35e1–8.08e1 | 6.77e1 | 2.22e-3 | 3.63e-3 | 5.78e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 496 | 69–77946 | +1.149e-5 | 0.450 | +1.173e-5 | 0.470 | 100 | +1.375e-5 | 0.770 | 37–649 | +1.529e-3 | 0.612 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 483 | 939–77946 | +1.149e-5 | 0.490 | +1.172e-5 | 0.512 | 99 | +1.374e-5 | 0.764 | 56–649 | +1.519e-3 | 0.687 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 56 | 78497–116726 | +2.355e-5 | 0.307 | +2.347e-5 | 0.304 | 43 | +2.640e-5 | 0.451 | 430–1085 | +1.346e-3 | 0.269 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 45 | 117551–155722 | -1.093e-5 | 0.068 | -1.091e-5 | 0.068 | 45 | -1.091e-5 | 0.068 | 767–992 | -8.079e-4 | 0.012 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.529e-3 | r0: +1.504e-3, r1: +1.538e-3, r2: +1.549e-3 | r0: 0.680, r1: 0.573, r2: 0.576 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.519e-3 | r0: +1.496e-3, r1: +1.527e-3, r2: +1.540e-3 | r0: 0.779, r1: 0.636, r2: 0.641 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.346e-3 | r0: +1.339e-3, r1: +1.349e-3, r2: +1.350e-3 | r0: 0.267, r1: 0.269, r2: 0.270 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -8.079e-4 | r0: -8.614e-4, r1: -7.724e-4, r2: -7.908e-4 | r0: 0.014, r1: 0.011, r2: 0.011 | 1.12× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇▇▇▇▇▇▇▇▇▇▇▇▇████████████▄▄▄▄▄▅▅▅▅▆▆▂▁▁▁▁▁▁▁▁▁▁` | `▁▆▅▅▆▆▆▆▇▆▆▆▇█▇▇▇▇▇▇▇▇▇▇▇▄▅▆▆▇▇▇▇▇▇▇▅▅▅▆▆▆▆▆▆▆▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 0.00e0 | 3.89e-1 | 9.56e-2 | 5.62e-2 | 25 | -5.03e-2 | +5.35e-3 | -9.72e-3 | -5.88e-3 |
| 1 | 3.00e-1 | 9 | 6.29e-2 | 1.13e-1 | 7.38e-2 | 7.37e-2 | 23 | -2.16e-2 | +2.45e-2 | +1.20e-3 | -1.24e-3 |
| 2 | 3.00e-1 | 10 | 6.69e-2 | 1.07e-1 | 7.93e-2 | 7.80e-2 | 27 | -1.53e-2 | +1.88e-2 | +5.87e-4 | -1.81e-4 |
| 3 | 3.00e-1 | 9 | 8.04e-2 | 1.17e-1 | 8.92e-2 | 9.50e-2 | 28 | -1.29e-2 | +1.28e-2 | +5.97e-4 | +3.35e-4 |
| 4 | 3.00e-1 | 9 | 7.94e-2 | 1.33e-1 | 9.32e-2 | 9.08e-2 | 31 | -1.45e-2 | +1.21e-2 | -3.75e-4 | -5.47e-5 |
| 5 | 3.00e-1 | 8 | 8.28e-2 | 1.33e-1 | 9.59e-2 | 9.64e-2 | 30 | -1.54e-2 | +1.13e-2 | +1.33e-5 | +1.65e-5 |
| 6 | 3.00e-1 | 13 | 7.25e-2 | 1.35e-1 | 9.10e-2 | 9.10e-2 | 28 | -2.17e-2 | +1.29e-2 | -6.24e-4 | -2.14e-4 |
| 7 | 3.00e-1 | 6 | 9.30e-2 | 1.47e-1 | 1.06e-1 | 1.02e-1 | 32 | -1.27e-2 | +1.39e-2 | +3.86e-4 | -1.33e-5 |
| 8 | 3.00e-1 | 7 | 9.47e-2 | 1.40e-1 | 1.04e-1 | 1.01e-1 | 35 | -1.30e-2 | +1.06e-2 | -1.39e-4 | -9.54e-5 |
| 9 | 3.00e-1 | 11 | 8.88e-2 | 1.48e-1 | 1.03e-1 | 8.88e-2 | 30 | -1.18e-2 | +1.21e-2 | -4.07e-4 | -4.99e-4 |
| 10 | 3.00e-1 | 5 | 9.09e-2 | 1.32e-1 | 1.01e-1 | 9.35e-2 | 32 | -1.23e-2 | +1.14e-2 | +1.47e-4 | -3.30e-4 |
| 11 | 3.00e-1 | 10 | 7.39e-2 | 1.33e-1 | 9.06e-2 | 7.39e-2 | 24 | -1.66e-2 | +1.01e-2 | -1.46e-3 | -1.27e-3 |
| 12 | 3.00e-1 | 10 | 7.37e-2 | 1.43e-1 | 8.94e-2 | 7.98e-2 | 21 | -3.02e-2 | +2.40e-2 | -1.58e-4 | -8.32e-4 |
| 13 | 3.00e-1 | 10 | 7.27e-2 | 1.40e-1 | 8.86e-2 | 8.10e-2 | 25 | -2.99e-2 | +2.90e-2 | +3.07e-4 | -3.62e-4 |
| 14 | 3.00e-1 | 10 | 7.69e-2 | 1.40e-1 | 8.94e-2 | 8.56e-2 | 27 | -2.52e-2 | +2.17e-2 | +5.21e-5 | -1.73e-4 |
| 15 | 3.00e-1 | 12 | 7.66e-2 | 1.45e-1 | 9.13e-2 | 8.99e-2 | 27 | -1.87e-2 | +1.75e-2 | -4.80e-5 | -2.98e-5 |
| 16 | 3.00e-1 | 6 | 8.69e-2 | 1.48e-1 | 9.97e-2 | 9.54e-2 | 27 | -1.76e-2 | +1.81e-2 | +4.49e-4 | +1.18e-4 |
| 17 | 3.00e-1 | 9 | 8.13e-2 | 1.36e-1 | 9.22e-2 | 9.16e-2 | 29 | -2.04e-2 | +1.80e-2 | +9.55e-5 | +1.08e-4 |
| 18 | 3.00e-1 | 9 | 7.71e-2 | 1.55e-1 | 1.08e-1 | 1.06e-1 | 42 | -2.39e-2 | +1.70e-2 | -1.96e-4 | -1.22e-4 |
| 19 | 3.00e-1 | 3 | 1.04e-1 | 1.51e-1 | 1.23e-1 | 1.04e-1 | 42 | -8.71e-3 | +6.31e-3 | -4.73e-4 | -3.13e-4 |
| 20 | 3.00e-1 | 7 | 1.01e-1 | 1.48e-1 | 1.15e-1 | 1.11e-1 | 47 | -8.80e-3 | +7.17e-3 | +1.69e-4 | -1.12e-4 |
| 21 | 3.00e-1 | 5 | 9.65e-2 | 1.55e-1 | 1.14e-1 | 1.08e-1 | 37 | -1.28e-2 | +7.38e-3 | -4.10e-4 | -2.47e-4 |
| 22 | 3.00e-1 | 7 | 9.34e-2 | 1.46e-1 | 1.05e-1 | 9.88e-2 | 35 | -1.25e-2 | +1.17e-2 | +8.72e-6 | -1.68e-4 |
| 23 | 3.00e-1 | 6 | 9.11e-2 | 1.46e-1 | 1.09e-1 | 1.05e-1 | 40 | -1.25e-2 | +1.23e-2 | +5.79e-4 | +1.05e-4 |
| 24 | 3.00e-1 | 8 | 9.18e-2 | 1.50e-1 | 1.05e-1 | 9.20e-2 | 29 | -9.70e-3 | +7.43e-3 | -7.15e-4 | -4.18e-4 |
| 25 | 3.00e-1 | 7 | 8.91e-2 | 1.52e-1 | 1.00e-1 | 9.08e-2 | 29 | -1.47e-2 | +1.47e-2 | -3.10e-4 | -4.56e-4 |
| 26 | 3.00e-1 | 8 | 9.12e-2 | 1.44e-1 | 1.03e-1 | 1.01e-1 | 29 | -1.25e-2 | +1.27e-2 | +3.30e-4 | -4.52e-5 |
| 27 | 3.00e-1 | 8 | 8.65e-2 | 1.47e-1 | 9.93e-2 | 9.11e-2 | 31 | -1.76e-2 | +1.76e-2 | +2.92e-5 | -1.39e-4 |
| 28 | 3.00e-1 | 7 | 8.82e-2 | 1.52e-1 | 1.01e-1 | 9.35e-2 | 33 | -1.87e-2 | +1.73e-2 | +8.15e-5 | -1.27e-4 |
| 29 | 3.00e-1 | 9 | 8.74e-2 | 1.46e-1 | 9.95e-2 | 9.53e-2 | 30 | -1.76e-2 | +1.25e-2 | -1.73e-4 | -1.77e-4 |
| 30 | 3.00e-1 | 8 | 8.29e-2 | 1.45e-1 | 9.45e-2 | 9.04e-2 | 30 | -1.68e-2 | +1.67e-2 | -1.97e-4 | -1.94e-4 |
| 31 | 3.00e-1 | 12 | 8.67e-2 | 1.46e-1 | 9.79e-2 | 9.97e-2 | 30 | -1.50e-2 | +1.24e-2 | +5.89e-5 | +7.35e-5 |
| 32 | 3.00e-1 | 5 | 8.30e-2 | 1.40e-1 | 1.01e-1 | 9.36e-2 | 34 | -1.73e-2 | +1.48e-2 | +1.16e-4 | +1.94e-5 |
| 33 | 3.00e-1 | 8 | 9.29e-2 | 1.54e-1 | 1.06e-1 | 1.01e-1 | 30 | -1.54e-2 | +1.36e-2 | +1.64e-4 | +3.01e-5 |
| 34 | 3.00e-1 | 7 | 8.73e-2 | 1.37e-1 | 1.03e-1 | 1.17e-1 | 35 | -1.50e-2 | +1.37e-2 | +8.31e-4 | +5.38e-4 |
| 35 | 3.00e-1 | 9 | 7.88e-2 | 1.40e-1 | 9.41e-2 | 8.84e-2 | 30 | -2.30e-2 | +1.48e-2 | -9.82e-4 | -3.02e-4 |
| 36 | 3.00e-1 | 13 | 8.00e-2 | 1.55e-1 | 9.27e-2 | 9.75e-2 | 30 | -2.13e-2 | +1.54e-2 | -1.61e-4 | +5.74e-5 |
| 37 | 3.00e-1 | 6 | 6.87e-2 | 1.59e-1 | 9.37e-2 | 8.45e-2 | 25 | -3.13e-2 | +1.98e-2 | -1.65e-3 | -6.84e-4 |
| 38 | 3.00e-1 | 8 | 8.21e-2 | 1.44e-1 | 9.91e-2 | 8.64e-2 | 26 | -1.26e-2 | +1.81e-2 | +1.16e-4 | -5.04e-4 |
| 39 | 3.00e-1 | 8 | 8.71e-2 | 1.46e-1 | 9.99e-2 | 9.83e-2 | 33 | -1.85e-2 | +1.66e-2 | +3.30e-4 | -9.93e-5 |
| 40 | 3.00e-1 | 11 | 7.82e-2 | 1.60e-1 | 9.85e-2 | 9.14e-2 | 33 | -1.73e-2 | +1.49e-2 | -3.00e-4 | -2.34e-4 |
| 41 | 3.00e-1 | 5 | 9.02e-2 | 1.48e-1 | 1.06e-1 | 9.83e-2 | 30 | -1.58e-2 | +1.30e-2 | +1.47e-4 | -1.55e-4 |
| 42 | 3.00e-1 | 8 | 8.84e-2 | 1.65e-1 | 1.01e-1 | 8.97e-2 | 29 | -1.83e-2 | +1.73e-2 | -3.00e-4 | -3.32e-4 |
| 43 | 3.00e-1 | 8 | 8.83e-2 | 1.38e-1 | 1.03e-1 | 1.04e-1 | 35 | -1.36e-2 | +1.60e-2 | +7.58e-4 | +1.78e-4 |
| 44 | 3.00e-1 | 6 | 9.55e-2 | 1.52e-1 | 1.11e-1 | 9.55e-2 | 30 | -9.14e-3 | +1.05e-2 | -4.65e-4 | -2.82e-4 |
| 45 | 3.00e-1 | 7 | 9.28e-2 | 1.41e-1 | 1.08e-1 | 1.09e-1 | 33 | -1.24e-2 | +1.21e-2 | +5.13e-4 | +7.75e-5 |
| 46 | 3.00e-1 | 8 | 8.83e-2 | 1.47e-1 | 1.04e-1 | 9.54e-2 | 35 | -1.65e-2 | +1.28e-2 | -3.10e-4 | -2.13e-4 |
| 47 | 3.00e-1 | 8 | 8.80e-2 | 1.47e-1 | 9.90e-2 | 9.12e-2 | 30 | -1.67e-2 | +1.27e-2 | -4.08e-4 | -3.62e-4 |
| 48 | 3.00e-1 | 9 | 8.40e-2 | 1.53e-1 | 9.54e-2 | 9.36e-2 | 27 | -2.08e-2 | +1.74e-2 | -7.63e-5 | -1.36e-4 |
| 49 | 3.00e-1 | 9 | 8.14e-2 | 1.47e-1 | 9.92e-2 | 9.50e-2 | 30 | -1.97e-2 | +2.05e-2 | +5.64e-4 | +1.29e-4 |
| 50 | 3.00e-1 | 10 | 8.97e-2 | 1.47e-1 | 9.98e-2 | 9.03e-2 | 30 | -1.49e-2 | +1.40e-2 | -1.68e-4 | -1.96e-4 |
| 51 | 3.00e-1 | 2 | 9.64e-2 | 1.05e-1 | 1.01e-1 | 1.05e-1 | 37 | +1.86e-3 | +2.20e-3 | +2.03e-3 | +2.29e-4 |
| 52 | 3.00e-1 | 2 | 1.05e-1 | 2.28e-1 | 1.66e-1 | 2.28e-1 | 174 | +1.43e-5 | +4.45e-3 | +2.23e-3 | +6.32e-4 |
| 53 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 204 | -8.78e-4 | -8.78e-4 | -8.78e-4 | +4.81e-4 |
| 54 | 3.00e-1 | 2 | 1.94e-1 | 2.07e-1 | 2.00e-1 | 1.94e-1 | 161 | -3.89e-4 | +4.15e-4 | +1.30e-5 | +3.88e-4 |
| 55 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 211 | -2.59e-4 | -2.59e-4 | -2.59e-4 | +3.23e-4 |
| 56 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 209 | +4.86e-4 | +4.86e-4 | +4.86e-4 | +3.39e-4 |
| 57 | 3.00e-1 | 2 | 1.95e-1 | 1.99e-1 | 1.97e-1 | 1.95e-1 | 182 | -1.16e-4 | -8.94e-5 | -1.03e-4 | +2.56e-4 |
| 58 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 208 | -1.37e-4 | -1.37e-4 | -1.37e-4 | +2.16e-4 |
| 59 | 3.00e-1 | 2 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 182 | +3.33e-6 | +1.37e-4 | +7.02e-5 | +1.88e-4 |
| 60 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 241 | -1.06e-4 | -1.06e-4 | -1.06e-4 | +1.58e-4 |
| 61 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 223 | +3.38e-4 | +3.38e-4 | +3.38e-4 | +1.76e-4 |
| 62 | 3.00e-1 | 2 | 1.95e-1 | 1.96e-1 | 1.96e-1 | 1.95e-1 | 158 | -2.44e-4 | -1.87e-5 | -1.32e-4 | +1.19e-4 |
| 63 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 209 | -4.44e-4 | -4.44e-4 | -4.44e-4 | +6.27e-5 |
| 64 | 3.00e-1 | 2 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 158 | +2.10e-5 | +4.37e-4 | +2.29e-4 | +9.23e-5 |
| 65 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 212 | -4.05e-4 | -4.05e-4 | -4.05e-4 | +4.25e-5 |
| 66 | 3.00e-1 | 2 | 1.93e-1 | 1.98e-1 | 1.95e-1 | 1.93e-1 | 158 | -1.64e-4 | +5.09e-4 | +1.73e-4 | +6.39e-5 |
| 67 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 208 | -2.97e-4 | -2.97e-4 | -2.97e-4 | +2.77e-5 |
| 68 | 3.00e-1 | 2 | 1.94e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 164 | +2.51e-5 | +3.26e-4 | +1.75e-4 | +5.43e-5 |
| 69 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 222 | -3.02e-4 | -3.02e-4 | -3.02e-4 | +1.86e-5 |
| 70 | 3.00e-1 | 2 | 1.99e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 148 | +1.96e-5 | +3.97e-4 | +2.08e-4 | +5.28e-5 |
| 71 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 179 | -8.36e-4 | -8.36e-4 | -8.36e-4 | -3.61e-5 |
| 72 | 3.00e-1 | 2 | 1.88e-1 | 2.03e-1 | 1.95e-1 | 2.03e-1 | 157 | +3.86e-4 | +4.98e-4 | +4.42e-4 | +5.53e-5 |
| 73 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 207 | -7.01e-4 | -7.01e-4 | -7.01e-4 | -2.03e-5 |
| 74 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 210 | +5.80e-4 | +5.80e-4 | +5.80e-4 | +3.97e-5 |
| 75 | 3.00e-1 | 2 | 1.97e-1 | 2.00e-1 | 1.98e-1 | 2.00e-1 | 169 | -3.09e-5 | +7.52e-5 | +2.21e-5 | +3.69e-5 |
| 76 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 210 | -2.95e-4 | -2.95e-4 | -2.95e-4 | +3.65e-6 |
| 77 | 3.00e-1 | 2 | 1.96e-1 | 1.97e-1 | 1.96e-1 | 1.96e-1 | 160 | -3.83e-5 | +2.30e-4 | +9.57e-5 | +1.98e-5 |
| 78 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 209 | -4.15e-4 | -4.15e-4 | -4.15e-4 | -2.37e-5 |
| 79 | 3.00e-1 | 2 | 1.95e-1 | 1.96e-1 | 1.95e-1 | 1.95e-1 | 171 | -3.80e-5 | +4.32e-4 | +1.97e-4 | +1.59e-5 |
| 80 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 192 | -2.17e-4 | -2.17e-4 | -2.17e-4 | -7.43e-6 |
| 81 | 3.00e-1 | 2 | 1.93e-1 | 1.99e-1 | 1.96e-1 | 1.99e-1 | 158 | +1.60e-4 | +1.80e-4 | +1.70e-4 | +2.64e-5 |
| 82 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 200 | -4.37e-4 | -4.37e-4 | -4.37e-4 | -1.99e-5 |
| 83 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 205 | +4.59e-4 | +4.59e-4 | +4.59e-4 | +2.80e-5 |
| 84 | 3.00e-1 | 2 | 1.94e-1 | 2.00e-1 | 1.97e-1 | 2.00e-1 | 160 | -1.31e-4 | +1.78e-4 | +2.36e-5 | +2.87e-5 |
| 85 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 216 | -3.95e-4 | -3.95e-4 | -3.95e-4 | -1.36e-5 |
| 86 | 3.00e-1 | 2 | 1.94e-1 | 2.00e-1 | 1.97e-1 | 1.94e-1 | 158 | -2.07e-4 | +4.35e-4 | +1.14e-4 | +7.37e-6 |
| 87 | 3.00e-1 | 2 | 1.83e-1 | 1.98e-1 | 1.90e-1 | 1.98e-1 | 170 | -2.97e-4 | +4.70e-4 | +8.65e-5 | +2.62e-5 |
| 88 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 218 | -2.43e-4 | -2.43e-4 | -2.43e-4 | -7.06e-7 |
| 89 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 188 | +3.48e-4 | +3.48e-4 | +3.48e-4 | +3.42e-5 |
| 90 | 3.00e-1 | 2 | 1.92e-1 | 2.04e-1 | 1.98e-1 | 2.04e-1 | 175 | -2.04e-4 | +3.50e-4 | +7.30e-5 | +4.43e-5 |
| 91 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 199 | -4.37e-4 | -4.37e-4 | -4.37e-4 | -3.83e-6 |
| 92 | 3.00e-1 | 2 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 158 | -7.58e-6 | +2.48e-4 | +1.20e-4 | +1.85e-5 |
| 93 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 177 | -5.37e-4 | -5.37e-4 | -5.37e-4 | -3.71e-5 |
| 94 | 3.00e-1 | 2 | 1.88e-1 | 1.92e-1 | 1.90e-1 | 1.92e-1 | 170 | +1.16e-4 | +2.87e-4 | +2.02e-4 | +7.43e-6 |
| 95 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 221 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -5.70e-6 |
| 96 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 230 | +3.12e-4 | +3.12e-4 | +3.12e-4 | +2.61e-5 |
| 97 | 3.00e-1 | 2 | 2.05e-1 | 2.08e-1 | 2.06e-1 | 2.05e-1 | 166 | -8.50e-5 | +1.51e-4 | +3.31e-5 | +2.62e-5 |
| 98 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 204 | -4.89e-4 | -4.89e-4 | -4.89e-4 | -2.53e-5 |
| 99 | 3.00e-1 | 2 | 1.93e-1 | 1.95e-1 | 1.94e-1 | 1.93e-1 | 159 | -5.59e-5 | +2.57e-4 | +1.00e-4 | -2.99e-6 |
| 100 | 3.00e-2 | 2 | 1.82e-1 | 1.83e-1 | 1.82e-1 | 1.82e-1 | 160 | -2.79e-4 | -4.04e-5 | -1.60e-4 | -3.15e-5 |
| 101 | 3.00e-2 | 1 | 1.76e-2 | 1.76e-2 | 1.76e-2 | 1.76e-2 | 197 | -1.18e-2 | -1.18e-2 | -1.18e-2 | -1.21e-3 |
| 102 | 3.00e-2 | 1 | 1.94e-2 | 1.94e-2 | 1.94e-2 | 1.94e-2 | 206 | +4.62e-4 | +4.62e-4 | +4.62e-4 | -1.05e-3 |
| 103 | 3.00e-2 | 2 | 2.07e-2 | 2.10e-2 | 2.09e-2 | 2.10e-2 | 162 | +8.78e-5 | +3.44e-4 | +2.16e-4 | -8.07e-4 |
| 104 | 3.00e-2 | 1 | 2.07e-2 | 2.07e-2 | 2.07e-2 | 2.07e-2 | 194 | -9.33e-5 | -9.33e-5 | -9.33e-5 | -7.36e-4 |
| 105 | 3.00e-2 | 2 | 2.25e-2 | 2.30e-2 | 2.28e-2 | 2.30e-2 | 159 | +1.53e-4 | +4.23e-4 | +2.88e-4 | -5.43e-4 |
| 106 | 3.00e-2 | 1 | 2.16e-2 | 2.16e-2 | 2.16e-2 | 2.16e-2 | 207 | -3.09e-4 | -3.09e-4 | -3.09e-4 | -5.19e-4 |
| 107 | 3.00e-2 | 2 | 2.44e-2 | 2.45e-2 | 2.45e-2 | 2.44e-2 | 159 | -2.61e-5 | +6.40e-4 | +3.07e-4 | -3.66e-4 |
| 108 | 3.00e-2 | 2 | 2.25e-2 | 2.53e-2 | 2.39e-2 | 2.53e-2 | 159 | -4.17e-4 | +7.34e-4 | +1.59e-4 | -2.60e-4 |
| 109 | 3.00e-2 | 1 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 220 | -1.89e-4 | -1.89e-4 | -1.89e-4 | -2.53e-4 |
| 110 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 212 | +6.35e-4 | +6.35e-4 | +6.35e-4 | -1.64e-4 |
| 111 | 3.00e-2 | 2 | 2.74e-2 | 2.79e-2 | 2.76e-2 | 2.79e-2 | 148 | -6.79e-5 | +1.19e-4 | +2.57e-5 | -1.27e-4 |
| 112 | 3.00e-2 | 2 | 2.53e-2 | 2.71e-2 | 2.62e-2 | 2.71e-2 | 148 | -5.22e-4 | +4.69e-4 | -2.67e-5 | -1.03e-4 |
| 113 | 3.00e-2 | 1 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 210 | -2.47e-4 | -2.47e-4 | -2.47e-4 | -1.18e-4 |
| 114 | 3.00e-2 | 2 | 2.90e-2 | 3.00e-2 | 2.95e-2 | 3.00e-2 | 156 | +2.01e-4 | +6.13e-4 | +4.07e-4 | -1.99e-5 |
| 115 | 3.00e-2 | 1 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 225 | -3.08e-4 | -3.08e-4 | -3.08e-4 | -4.87e-5 |
| 116 | 3.00e-2 | 2 | 3.06e-2 | 3.19e-2 | 3.13e-2 | 3.06e-2 | 156 | -2.65e-4 | +6.61e-4 | +1.98e-4 | -6.50e-6 |
| 117 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 156 | -4.95e-4 | -4.95e-4 | -4.95e-4 | -5.53e-5 |
| 118 | 3.00e-2 | 1 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 310 | -5.77e-5 | -5.77e-5 | -5.77e-5 | -5.55e-5 |
| 119 | 3.00e-2 | 1 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 286 | +1.23e-3 | +1.23e-3 | +1.23e-3 | +7.25e-5 |
| 120 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 282 | +5.78e-6 | +5.78e-6 | +5.78e-6 | +6.59e-5 |
| 121 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 294 | +1.80e-6 | +1.80e-6 | +1.80e-6 | +5.95e-5 |
| 122 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 300 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +6.78e-5 |
| 123 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 284 | +7.20e-6 | +7.20e-6 | +7.20e-6 | +6.17e-5 |
| 125 | 3.00e-2 | 2 | 4.20e-2 | 4.32e-2 | 4.26e-2 | 4.32e-2 | 264 | +4.91e-5 | +1.03e-4 | +7.62e-5 | +6.48e-5 |
| 126 | 3.00e-2 | 1 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 312 | -7.12e-5 | -7.12e-5 | -7.12e-5 | +5.12e-5 |
| 128 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 403 | +2.32e-4 | +2.32e-4 | +2.32e-4 | +6.93e-5 |
| 129 | 3.00e-2 | 1 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 364 | +3.28e-4 | +3.28e-4 | +3.28e-4 | +9.51e-5 |
| 130 | 3.00e-2 | 1 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 5.27e-2 | 332 | +2.52e-5 | +2.52e-5 | +2.52e-5 | +8.81e-5 |
| 132 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 364 | -1.35e-4 | -1.35e-4 | -1.35e-4 | +6.58e-5 |
| 133 | 3.00e-2 | 2 | 4.85e-2 | 5.31e-2 | 5.08e-2 | 4.85e-2 | 251 | -3.57e-4 | +2.11e-4 | -7.28e-5 | +3.67e-5 |
| 135 | 3.00e-2 | 2 | 4.70e-2 | 5.31e-2 | 5.00e-2 | 5.31e-2 | 263 | -1.01e-4 | +4.63e-4 | +1.81e-4 | +6.68e-5 |
| 137 | 3.00e-2 | 2 | 5.13e-2 | 5.74e-2 | 5.44e-2 | 5.74e-2 | 259 | -9.92e-5 | +4.32e-4 | +1.66e-4 | +8.84e-5 |
| 139 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 344 | -3.42e-4 | -3.42e-4 | -3.42e-4 | +4.54e-5 |
| 140 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 310 | +3.99e-4 | +3.99e-4 | +3.99e-4 | +8.07e-5 |
| 141 | 3.00e-2 | 1 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 308 | -5.93e-5 | -5.93e-5 | -5.93e-5 | +6.67e-5 |
| 142 | 3.00e-2 | 1 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 280 | +8.09e-5 | +8.09e-5 | +8.09e-5 | +6.81e-5 |
| 143 | 3.00e-2 | 1 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 294 | -1.02e-4 | -1.02e-4 | -1.02e-4 | +5.11e-5 |
| 144 | 3.00e-2 | 1 | 5.86e-2 | 5.86e-2 | 5.86e-2 | 5.86e-2 | 306 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +5.92e-5 |
| 145 | 3.00e-2 | 1 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 322 | +3.91e-5 | +3.91e-5 | +3.91e-5 | +5.72e-5 |
| 147 | 3.00e-2 | 1 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 389 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +6.20e-5 |
| 148 | 3.00e-2 | 1 | 6.68e-2 | 6.68e-2 | 6.68e-2 | 6.68e-2 | 312 | +2.49e-4 | +2.49e-4 | +2.49e-4 | +8.07e-5 |
| 149 | 3.00e-2 | 1 | 6.16e-2 | 6.16e-2 | 6.16e-2 | 6.16e-2 | 288 | -2.80e-4 | -2.80e-4 | -2.80e-4 | +4.46e-5 |
| 150 | 3.00e-3 | 1 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 291 | +8.23e-5 | +8.23e-5 | +8.23e-5 | +4.84e-5 |
| 151 | 3.00e-3 | 1 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 295 | -1.58e-5 | -1.58e-5 | -1.58e-5 | +4.20e-5 |
| 152 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 275 | -8.41e-3 | -8.41e-3 | -8.41e-3 | -8.03e-4 |
| 153 | 3.00e-3 | 1 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 310 | -2.83e-4 | -2.83e-4 | -2.83e-4 | -7.51e-4 |
| 154 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 317 | +3.94e-5 | +3.94e-5 | +3.94e-5 | -6.72e-4 |
| 155 | 3.00e-3 | 1 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 6.08e-3 | 312 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -5.88e-4 |
| 157 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 375 | -2.66e-5 | -2.66e-5 | -2.66e-5 | -5.32e-4 |
| 158 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 327 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -4.61e-4 |
| 159 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 260 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -4.34e-4 |
| 160 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 266 | -2.36e-4 | -2.36e-4 | -2.36e-4 | -4.14e-4 |
| 161 | 3.00e-3 | 1 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 289 | +2.80e-5 | +2.80e-5 | +2.80e-5 | -3.70e-4 |
| 162 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 287 | +1.84e-5 | +1.84e-5 | +1.84e-5 | -3.31e-4 |
| 163 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 295 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -2.88e-4 |
| 164 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 291 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -2.46e-4 |
| 165 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 310 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -2.32e-4 |
| 166 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 310 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -1.96e-4 |
| 167 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 370 | -2.75e-5 | -2.75e-5 | -2.75e-5 | -1.79e-4 |
| 169 | 3.00e-3 | 1 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 349 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -1.38e-4 |
| 170 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 299 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -1.35e-4 |
| 171 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 308 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -1.33e-4 |
| 172 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 335 | +6.14e-5 | +6.14e-5 | +6.14e-5 | -1.13e-4 |
| 173 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 340 | +3.41e-5 | +3.41e-5 | +3.41e-5 | -9.86e-5 |
| 174 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 340 | -1.55e-6 | -1.55e-6 | -1.55e-6 | -8.89e-5 |
| 176 | 3.00e-3 | 1 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 6.71e-3 | 360 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -6.92e-5 |
| 177 | 3.00e-3 | 1 | 6.91e-3 | 6.91e-3 | 6.91e-3 | 6.91e-3 | 302 | +9.73e-5 | +9.73e-5 | +9.73e-5 | -5.26e-5 |
| 178 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 335 | -2.79e-4 | -2.79e-4 | -2.79e-4 | -7.52e-5 |
| 179 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 320 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -5.07e-5 |
| 180 | 3.00e-3 | 1 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 298 | -8.05e-6 | -8.05e-6 | -8.05e-6 | -4.64e-5 |
| 181 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 306 | -3.22e-5 | -3.22e-5 | -3.22e-5 | -4.50e-5 |
| 182 | 3.00e-3 | 1 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 6.38e-3 | 295 | -9.39e-5 | -9.39e-5 | -9.39e-5 | -4.99e-5 |
| 183 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 301 | +4.98e-5 | +4.98e-5 | +4.98e-5 | -3.99e-5 |
| 184 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 323 | -1.38e-5 | -1.38e-5 | -1.38e-5 | -3.73e-5 |
| 185 | 3.00e-3 | 1 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 352 | +9.38e-5 | +9.38e-5 | +9.38e-5 | -2.42e-5 |
| 187 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 363 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -9.56e-6 |
| 188 | 3.00e-3 | 1 | 7.00e-3 | 7.00e-3 | 7.00e-3 | 7.00e-3 | 300 | +1.66e-5 | +1.66e-5 | +1.66e-5 | -6.94e-6 |
| 189 | 3.00e-3 | 1 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 6.51e-3 | 284 | -2.59e-4 | -2.59e-4 | -2.59e-4 | -3.22e-5 |
| 190 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 266 | -5.23e-5 | -5.23e-5 | -5.23e-5 | -3.42e-5 |
| 191 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 270 | -2.48e-5 | -2.48e-5 | -2.48e-5 | -3.33e-5 |
| 192 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 305 | -2.89e-5 | -2.89e-5 | -2.89e-5 | -3.28e-5 |
| 193 | 3.00e-3 | 1 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 6.59e-3 | 329 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -1.66e-5 |
| 194 | 3.00e-3 | 1 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 303 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -1.86e-6 |
| 195 | 3.00e-3 | 1 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 6.75e-3 | 333 | -4.82e-5 | -4.82e-5 | -4.82e-5 | -6.50e-6 |
| 197 | 3.00e-3 | 1 | 7.09e-3 | 7.09e-3 | 7.09e-3 | 7.09e-3 | 369 | +1.33e-4 | +1.33e-4 | +1.33e-4 | +7.42e-6 |
| 198 | 3.00e-3 | 1 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 7.38e-3 | 286 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +2.10e-5 |
| 199 | 3.00e-3 | 1 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 279 | -3.84e-4 | -3.84e-4 | -3.84e-4 | -1.95e-5 |

