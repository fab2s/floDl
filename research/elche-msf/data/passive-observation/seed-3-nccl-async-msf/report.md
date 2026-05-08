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
| nccl-async | 0.052934 | 0.9189 | +0.0064 | 1927.4 | 639 | 40.7 | 100% | 100% | 5.5 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9189 | nccl-async | - | - |

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
| nccl-async | 1.9968 | 0.6865 | 0.5261 | 0.5428 | 0.5192 | 0.4974 | 0.4888 | 0.4612 | 0.4643 | 0.4474 | 0.2026 | 0.1613 | 0.1405 | 0.1162 | 0.1349 | 0.0732 | 0.0647 | 0.0600 | 0.0569 | 0.0529 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4009 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3018 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2973 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 400 | 396 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1926.3 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu2 | 1926.4 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1926.2 | 0.8 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.8s | 0.0s | 0.0s | 0.0s | 1.9s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 2.6s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 362 | 0 | 639 | 40.7 | 741/10062 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 183.1 | 9.5% |

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
| resnet-graph | nccl-async | 191 | 639 | 0 | 5.91e-3 | +2.99e-7 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 639 | 7.02e-2 | 5.84e-2 | 0.00e0 | 4.43e-1 | 46.8 | -1.40e-4 | 3.10e-3 |
| resnet-graph | nccl-async | 1 | 639 | 7.09e-2 | 6.14e-2 | 0.00e0 | 5.35e-1 | 36.3 | -1.60e-4 | 4.72e-3 |
| resnet-graph | nccl-async | 2 | 639 | 7.03e-2 | 6.22e-2 | 0.00e0 | 5.08e-1 | 16.9 | -1.49e-4 | 4.84e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9937 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9868 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9916 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 54 (0,1,2,3,4,5,6,7…142,147) | 2 (139,143) | — | 0,1,2,3,4,5,6,7…142,147 | 139,143 |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 39 | 39 |
| resnet-graph | nccl-async | 0e0 | 5 | 19 | 19 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 20 | 20 |
| resnet-graph | nccl-async | 1e-4 | 5 | 8 | 8 |
| resnet-graph | nccl-async | 1e-4 | 10 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 425 | +0.066 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 156 | -0.027 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 53 | +0.029 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 636 | +0.004 | 190 | +0.224 | +0.335 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 637 | 3.44e1–8.01e1 | 6.44e1 | 1.94e-3 | 6.85e-3 | 1.40e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 427 | 63–77737 | +1.912e-5 | 0.582 | +1.974e-5 | 0.607 | 95 | +1.449e-5 | 0.507 | 32–947 | +1.579e-3 | 0.728 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 411 | 878–77737 | +1.947e-5 | 0.670 | +2.000e-5 | 0.682 | 94 | +1.432e-5 | 0.495 | 32–947 | +1.582e-3 | 0.815 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 157 | 78246–116608 | -1.375e-5 | 0.078 | -1.472e-5 | 0.086 | 49 | +1.147e-5 | 0.065 | 32–902 | +1.615e-3 | 0.477 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 54 | 117537–155774 | -1.458e-5 | 0.132 | -1.460e-5 | 0.133 | 47 | -1.551e-5 | 0.134 | 502–1007 | +1.196e-3 | 0.089 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.579e-3 | r0: +1.550e-3, r1: +1.590e-3, r2: +1.601e-3 | r0: 0.770, r1: 0.709, r2: 0.698 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.582e-3 | r0: +1.551e-3, r1: +1.589e-3, r2: +1.609e-3 | r0: 0.853, r1: 0.790, r2: 0.791 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.615e-3 | r0: +1.596e-3, r1: +1.623e-3, r2: +1.636e-3 | r0: 0.495, r1: 0.465, r2: 0.467 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +1.196e-3 | r0: +1.217e-3, r1: +1.203e-3, r2: +1.166e-3 | r0: 0.092, r1: 0.090, r2: 0.084 | 1.04× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇▇▇▇▇▇▇█████████████████▃▄▄▄▄▄▄▄▄▄▆▆▃▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▅▅▅▅▆▆▆▆▆▆▅▅▅▅▅▅▅▅▅▅▅▅▃▄▅▅▅▆▅▅▅▆█▇▆▅▅▅▅▅▅▅▅▅▅▅` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 17 | 0.00e0 | 5.35e-1 | 1.17e-1 | 4.87e-2 | 20 | -6.81e-2 | +1.05e-2 | -1.13e-2 | -5.93e-3 |
| 1 | 3.00e-1 | 16 | 4.68e-2 | 9.84e-2 | 5.41e-2 | 5.25e-2 | 16 | -3.38e-2 | +3.72e-2 | +8.11e-5 | -8.97e-4 |
| 2 | 3.00e-1 | 14 | 5.23e-2 | 1.07e-1 | 6.43e-2 | 6.02e-2 | 20 | -3.88e-2 | +3.37e-2 | -8.65e-5 | -4.43e-4 |
| 3 | 3.00e-1 | 17 | 4.98e-2 | 1.03e-1 | 6.14e-2 | 5.85e-2 | 15 | -3.60e-2 | +2.67e-2 | -3.12e-4 | -2.72e-4 |
| 4 | 3.00e-1 | 16 | 6.15e-2 | 1.14e-1 | 7.04e-2 | 6.95e-2 | 17 | -3.84e-2 | +3.58e-2 | +4.10e-4 | +8.17e-5 |
| 5 | 3.00e-1 | 17 | 5.83e-2 | 1.30e-1 | 6.93e-2 | 6.65e-2 | 17 | -4.08e-2 | +3.52e-2 | -5.20e-4 | -1.06e-4 |
| 6 | 3.00e-1 | 21 | 6.08e-2 | 1.26e-1 | 7.25e-2 | 7.01e-2 | 17 | -3.71e-2 | +3.70e-2 | +3.41e-4 | +1.09e-4 |
| 7 | 3.00e-1 | 8 | 6.40e-2 | 1.42e-1 | 8.14e-2 | 7.99e-2 | 22 | -5.27e-2 | +4.38e-2 | +3.13e-4 | +1.39e-4 |
| 8 | 3.00e-1 | 18 | 6.59e-2 | 1.31e-1 | 7.81e-2 | 7.69e-2 | 20 | -2.27e-2 | +2.33e-2 | -9.60e-5 | +1.27e-4 |
| 9 | 3.00e-1 | 9 | 6.59e-2 | 1.31e-1 | 7.76e-2 | 6.89e-2 | 17 | -3.37e-2 | +3.00e-2 | -7.56e-4 | -5.29e-4 |
| 10 | 3.00e-1 | 14 | 6.46e-2 | 1.33e-1 | 7.37e-2 | 6.78e-2 | 18 | -4.12e-2 | +3.95e-2 | -2.69e-5 | -3.81e-4 |
| 11 | 3.00e-1 | 16 | 5.64e-2 | 1.28e-1 | 6.78e-2 | 6.02e-2 | 19 | -4.25e-2 | +3.94e-2 | -3.53e-4 | -6.55e-4 |
| 12 | 3.00e-1 | 18 | 5.86e-2 | 1.35e-1 | 7.09e-2 | 6.64e-2 | 18 | -4.57e-2 | +3.64e-2 | -2.24e-4 | -3.53e-4 |
| 13 | 3.00e-1 | 10 | 5.78e-2 | 1.40e-1 | 7.54e-2 | 8.51e-2 | 31 | -5.17e-2 | +4.00e-2 | -9.72e-6 | +2.05e-4 |
| 14 | 3.00e-1 | 11 | 7.63e-2 | 1.35e-1 | 8.81e-2 | 7.63e-2 | 21 | -1.88e-2 | +1.13e-2 | -1.02e-3 | -6.97e-4 |
| 15 | 3.00e-1 | 17 | 5.33e-2 | 1.31e-1 | 6.62e-2 | 7.48e-2 | 17 | -5.64e-2 | +3.64e-2 | -7.53e-4 | +3.86e-4 |
| 16 | 3.00e-1 | 20 | 5.12e-2 | 1.27e-1 | 7.18e-2 | 8.42e-2 | 23 | -3.57e-2 | +3.29e-2 | +1.14e-4 | +7.65e-4 |
| 17 | 3.00e-1 | 8 | 5.65e-2 | 1.37e-1 | 7.82e-2 | 6.59e-2 | 22 | -2.55e-2 | +2.30e-2 | -2.23e-3 | -1.03e-3 |
| 18 | 3.00e-1 | 10 | 7.41e-2 | 1.31e-1 | 8.72e-2 | 8.07e-2 | 24 | -2.09e-2 | +1.73e-2 | -1.48e-4 | -6.32e-4 |
| 19 | 3.00e-1 | 17 | 5.46e-2 | 1.41e-1 | 7.57e-2 | 7.88e-2 | 23 | -5.03e-2 | +3.08e-2 | -8.48e-4 | -1.92e-4 |
| 20 | 3.00e-1 | 8 | 6.56e-2 | 1.41e-1 | 8.23e-2 | 8.14e-2 | 21 | -3.34e-2 | +2.48e-2 | -9.34e-4 | -4.09e-4 |
| 21 | 3.00e-1 | 16 | 5.73e-2 | 1.33e-1 | 7.55e-2 | 8.15e-2 | 23 | -4.97e-2 | +2.77e-2 | -1.45e-4 | +2.27e-4 |
| 22 | 3.00e-1 | 9 | 6.45e-2 | 1.29e-1 | 7.97e-2 | 8.07e-2 | 22 | -2.50e-2 | +2.22e-2 | -2.51e-4 | +1.07e-4 |
| 23 | 3.00e-1 | 14 | 6.24e-2 | 1.30e-1 | 7.30e-2 | 6.41e-2 | 22 | -3.25e-2 | +2.47e-2 | -9.72e-4 | -6.13e-4 |
| 24 | 3.00e-1 | 2 | 7.88e-2 | 8.76e-2 | 8.32e-2 | 7.88e-2 | 22 | -4.84e-3 | +1.42e-2 | +4.71e-3 | +3.02e-4 |
| 25 | 3.00e-1 | 1 | 7.47e-2 | 7.47e-2 | 7.47e-2 | 7.47e-2 | 289 | -1.84e-4 | -1.84e-4 | -1.84e-4 | +2.54e-4 |
| 26 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 311 | +3.70e-3 | +3.70e-3 | +3.70e-3 | +5.99e-4 |
| 27 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 267 | -2.85e-4 | -2.85e-4 | -2.85e-4 | +5.10e-4 |
| 28 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 277 | -2.23e-4 | -2.23e-4 | -2.23e-4 | +4.37e-4 |
| 29 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 302 | +4.91e-6 | +4.91e-6 | +4.91e-6 | +3.94e-4 |
| 30 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 287 | +1.09e-5 | +1.09e-5 | +1.09e-5 | +3.55e-4 |
| 31 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 291 | -4.27e-5 | -4.27e-5 | -4.27e-5 | +3.16e-4 |
| 32 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 275 | +4.19e-5 | +4.19e-5 | +4.19e-5 | +2.88e-4 |
| 33 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 303 | -5.15e-5 | -5.15e-5 | -5.15e-5 | +2.54e-4 |
| 34 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 278 | +4.12e-5 | +4.12e-5 | +4.12e-5 | +2.33e-4 |
| 35 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 310 | -4.44e-5 | -4.44e-5 | -4.44e-5 | +2.05e-4 |
| 36 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 286 | +1.25e-4 | +1.25e-4 | +1.25e-4 | +1.97e-4 |
| 37 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 319 | -1.18e-4 | -1.18e-4 | -1.18e-4 | +1.66e-4 |
| 38 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 293 | +1.57e-4 | +1.57e-4 | +1.57e-4 | +1.65e-4 |
| 40 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 342 | -1.03e-4 | -1.03e-4 | -1.03e-4 | +1.38e-4 |
| 41 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 288 | +1.93e-4 | +1.93e-4 | +1.93e-4 | +1.44e-4 |
| 42 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 297 | -2.13e-4 | -2.13e-4 | -2.13e-4 | +1.08e-4 |
| 43 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 273 | +1.55e-5 | +1.55e-5 | +1.55e-5 | +9.87e-5 |
| 44 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 285 | -2.67e-5 | -2.67e-5 | -2.67e-5 | +8.62e-5 |
| 45 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 280 | +5.81e-5 | +5.81e-5 | +5.81e-5 | +8.33e-5 |
| 46 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 261 | -3.40e-5 | -3.40e-5 | -3.40e-5 | +7.16e-5 |
| 47 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 278 | -1.29e-4 | -1.29e-4 | -1.29e-4 | +5.16e-5 |
| 48 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 266 | +1.68e-4 | +1.68e-4 | +1.68e-4 | +6.32e-5 |
| 49 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 276 | -7.95e-5 | -7.95e-5 | -7.95e-5 | +4.89e-5 |
| 50 | 3.00e-1 | 2 | 2.00e-1 | 2.03e-1 | 2.02e-1 | 2.00e-1 | 234 | -7.26e-5 | +3.18e-5 | -2.04e-5 | +3.52e-5 |
| 52 | 3.00e-1 | 2 | 1.94e-1 | 2.12e-1 | 2.03e-1 | 2.12e-1 | 252 | -1.03e-4 | +3.53e-4 | +1.25e-4 | +5.45e-5 |
| 54 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 326 | -2.06e-4 | -2.06e-4 | -2.06e-4 | +2.84e-5 |
| 55 | 3.00e-1 | 2 | 1.97e-1 | 2.13e-1 | 2.05e-1 | 1.97e-1 | 218 | -3.51e-4 | +2.88e-4 | -3.12e-5 | +1.39e-5 |
| 56 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 237 | -2.06e-4 | -2.06e-4 | -2.06e-4 | -8.09e-6 |
| 57 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 225 | +1.46e-4 | +1.46e-4 | +1.46e-4 | +7.33e-6 |
| 58 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 263 | +2.34e-5 | +2.34e-5 | +2.34e-5 | +8.94e-6 |
| 59 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 252 | +1.80e-4 | +1.80e-4 | +1.80e-4 | +2.61e-5 |
| 60 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 239 | -1.15e-4 | -1.15e-4 | -1.15e-4 | +1.19e-5 |
| 61 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 256 | -1.55e-5 | -1.55e-5 | -1.55e-5 | +9.19e-6 |
| 62 | 3.00e-1 | 2 | 1.99e-1 | 2.05e-1 | 2.02e-1 | 2.05e-1 | 229 | +2.75e-5 | +1.14e-4 | +7.07e-5 | +2.13e-5 |
| 64 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 298 | -2.01e-4 | -2.01e-4 | -2.01e-4 | -9.27e-7 |
| 65 | 3.00e-1 | 2 | 1.95e-1 | 2.09e-1 | 2.02e-1 | 1.95e-1 | 240 | -2.88e-4 | +3.20e-4 | +1.60e-5 | -7.54e-7 |
| 66 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 248 | +3.20e-5 | +3.20e-5 | +3.20e-5 | +2.52e-6 |
| 67 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 236 | +2.55e-5 | +2.55e-5 | +2.55e-5 | +4.82e-6 |
| 68 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 258 | -5.60e-6 | -5.60e-6 | -5.60e-6 | +3.78e-6 |
| 69 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 240 | +1.14e-4 | +1.14e-4 | +1.14e-4 | +1.48e-5 |
| 70 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 285 | -9.10e-5 | -9.10e-5 | -9.10e-5 | +4.20e-6 |
| 71 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 276 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +1.42e-5 |
| 72 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 224 | +1.02e-4 | +1.02e-4 | +1.02e-4 | +2.30e-5 |
| 73 | 3.00e-1 | 2 | 1.93e-1 | 1.98e-1 | 1.95e-1 | 1.98e-1 | 216 | -3.11e-4 | +1.33e-4 | -8.90e-5 | +3.93e-6 |
| 74 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 245 | -1.94e-4 | -1.94e-4 | -1.94e-4 | -1.59e-5 |
| 75 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 240 | +2.20e-4 | +2.20e-4 | +2.20e-4 | +7.71e-6 |
| 76 | 3.00e-1 | 2 | 1.93e-1 | 1.98e-1 | 1.95e-1 | 1.93e-1 | 205 | -1.38e-4 | -2.48e-5 | -8.16e-5 | -9.83e-6 |
| 78 | 3.00e-1 | 2 | 1.83e-1 | 2.06e-1 | 1.95e-1 | 2.06e-1 | 193 | -1.90e-4 | +6.27e-4 | +2.18e-4 | +3.76e-5 |
| 79 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 220 | -5.48e-4 | -5.48e-4 | -5.48e-4 | -2.09e-5 |
| 80 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 255 | +2.43e-4 | +2.43e-4 | +2.43e-4 | +5.52e-6 |
| 81 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 241 | +1.04e-4 | +1.04e-4 | +1.04e-4 | +1.53e-5 |
| 82 | 3.00e-1 | 2 | 1.93e-1 | 2.00e-1 | 1.97e-1 | 1.93e-1 | 194 | -2.06e-4 | +1.88e-5 | -9.36e-5 | -6.49e-6 |
| 83 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 202 | -3.54e-4 | -3.54e-4 | -3.54e-4 | -4.13e-5 |
| 84 | 3.00e-1 | 2 | 1.88e-1 | 1.92e-1 | 1.90e-1 | 1.88e-1 | 194 | -9.64e-5 | +3.22e-4 | +1.13e-4 | -1.41e-5 |
| 85 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 220 | -3.60e-5 | -3.60e-5 | -3.60e-5 | -1.63e-5 |
| 86 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 219 | +1.54e-4 | +1.54e-4 | +1.54e-4 | +8.08e-7 |
| 87 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 219 | +3.10e-5 | +3.10e-5 | +3.10e-5 | +3.82e-6 |
| 88 | 3.00e-1 | 2 | 1.94e-1 | 2.05e-1 | 1.99e-1 | 2.05e-1 | 177 | -1.24e-5 | +3.14e-4 | +1.51e-4 | +3.34e-5 |
| 89 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 206 | -5.74e-4 | -5.74e-4 | -5.74e-4 | -2.73e-5 |
| 90 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 229 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -1.28e-5 |
| 91 | 3.00e-1 | 2 | 1.90e-1 | 1.95e-1 | 1.92e-1 | 1.90e-1 | 177 | -1.50e-4 | +2.07e-4 | +2.85e-5 | -6.73e-6 |
| 92 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 220 | -3.70e-4 | -3.70e-4 | -3.70e-4 | -4.30e-5 |
| 93 | 3.00e-1 | 2 | 1.89e-1 | 1.98e-1 | 1.93e-1 | 1.89e-1 | 173 | -2.81e-4 | +5.91e-4 | +1.55e-4 | -9.81e-6 |
| 94 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 188 | -3.04e-4 | -3.04e-4 | -3.04e-4 | -3.92e-5 |
| 95 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 232 | +1.02e-4 | +1.02e-4 | +1.02e-4 | -2.51e-5 |
| 96 | 3.00e-1 | 2 | 1.96e-1 | 1.98e-1 | 1.97e-1 | 1.96e-1 | 161 | -4.77e-5 | +3.49e-4 | +1.51e-4 | +6.33e-6 |
| 97 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 222 | -5.42e-4 | -5.42e-4 | -5.42e-4 | -4.86e-5 |
| 98 | 3.00e-1 | 2 | 1.78e-1 | 1.95e-1 | 1.86e-1 | 1.78e-1 | 154 | -5.77e-4 | +6.08e-4 | +1.56e-5 | -4.23e-5 |
| 99 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 199 | -2.20e-4 | -2.20e-4 | -2.20e-4 | -6.00e-5 |
| 100 | 3.00e-2 | 2 | 1.82e-1 | 1.91e-1 | 1.87e-1 | 1.82e-1 | 150 | -3.25e-4 | +6.37e-4 | +1.56e-4 | -2.38e-5 |
| 101 | 3.00e-2 | 2 | 1.71e-2 | 1.79e-2 | 1.75e-2 | 1.79e-2 | 150 | -1.41e-2 | +2.84e-4 | -6.90e-3 | -1.26e-3 |
| 102 | 3.00e-2 | 2 | 1.75e-2 | 1.95e-2 | 1.85e-2 | 1.95e-2 | 150 | -1.06e-4 | +7.06e-4 | +3.00e-4 | -9.58e-4 |
| 103 | 3.00e-2 | 1 | 1.87e-2 | 1.87e-2 | 1.87e-2 | 1.87e-2 | 178 | -2.32e-4 | -2.32e-4 | -2.32e-4 | -8.85e-4 |
| 104 | 3.00e-2 | 2 | 2.06e-2 | 2.15e-2 | 2.11e-2 | 2.15e-2 | 159 | +2.78e-4 | +5.24e-4 | +4.01e-4 | -6.42e-4 |
| 105 | 3.00e-2 | 1 | 2.06e-2 | 2.06e-2 | 2.06e-2 | 2.06e-2 | 200 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -6.00e-4 |
| 106 | 3.00e-2 | 2 | 2.31e-2 | 2.40e-2 | 2.36e-2 | 2.40e-2 | 186 | +1.98e-4 | +5.51e-4 | +3.74e-4 | -4.17e-4 |
| 107 | 3.00e-2 | 1 | 2.30e-2 | 2.30e-2 | 2.30e-2 | 2.30e-2 | 190 | -2.10e-4 | -2.10e-4 | -2.10e-4 | -3.96e-4 |
| 108 | 3.00e-2 | 3 | 2.04e-2 | 2.42e-2 | 2.28e-2 | 2.04e-2 | 125 | -1.21e-3 | +2.67e-4 | -3.61e-4 | -4.01e-4 |
| 109 | 3.00e-2 | 2 | 2.13e-2 | 2.23e-2 | 2.18e-2 | 2.23e-2 | 125 | +2.81e-4 | +3.58e-4 | +3.19e-4 | -2.64e-4 |
| 110 | 3.00e-2 | 2 | 2.09e-2 | 2.37e-2 | 2.23e-2 | 2.37e-2 | 137 | -4.04e-4 | +8.92e-4 | +2.44e-4 | -1.61e-4 |
| 111 | 3.00e-2 | 2 | 2.24e-2 | 2.45e-2 | 2.35e-2 | 2.45e-2 | 137 | -3.66e-4 | +6.54e-4 | +1.44e-4 | -9.77e-5 |
| 112 | 3.00e-2 | 1 | 2.28e-2 | 2.28e-2 | 2.28e-2 | 2.28e-2 | 192 | -3.84e-4 | -3.84e-4 | -3.84e-4 | -1.26e-4 |
| 113 | 3.00e-2 | 2 | 2.60e-2 | 2.71e-2 | 2.66e-2 | 2.60e-2 | 129 | -3.19e-4 | +1.10e-3 | +3.90e-4 | -3.53e-5 |
| 114 | 3.00e-2 | 2 | 2.41e-2 | 2.68e-2 | 2.54e-2 | 2.68e-2 | 114 | -4.42e-4 | +9.48e-4 | +2.53e-4 | +2.64e-5 |
| 115 | 3.00e-2 | 3 | 2.31e-2 | 2.60e-2 | 2.43e-2 | 2.31e-2 | 111 | -1.06e-3 | +7.92e-4 | -3.58e-4 | -8.07e-5 |
| 116 | 3.00e-2 | 2 | 2.34e-2 | 2.60e-2 | 2.47e-2 | 2.60e-2 | 114 | +7.68e-5 | +9.26e-4 | +5.01e-4 | +3.41e-5 |
| 117 | 3.00e-2 | 2 | 2.44e-2 | 2.62e-2 | 2.53e-2 | 2.62e-2 | 127 | -4.25e-4 | +5.78e-4 | +7.63e-5 | +4.72e-5 |
| 118 | 3.00e-2 | 2 | 2.64e-2 | 2.75e-2 | 2.69e-2 | 2.75e-2 | 116 | +2.72e-5 | +3.69e-4 | +1.98e-4 | +7.75e-5 |
| 119 | 3.00e-2 | 2 | 2.53e-2 | 2.88e-2 | 2.70e-2 | 2.88e-2 | 123 | -5.52e-4 | +1.07e-3 | +2.57e-4 | +1.20e-4 |
| 120 | 3.00e-2 | 3 | 2.46e-2 | 2.74e-2 | 2.61e-2 | 2.46e-2 | 108 | -9.98e-4 | +4.14e-4 | -4.34e-4 | -3.34e-5 |
| 121 | 3.00e-2 | 2 | 2.54e-2 | 2.72e-2 | 2.63e-2 | 2.72e-2 | 97 | +2.56e-4 | +7.01e-4 | +4.78e-4 | +6.60e-5 |
| 122 | 3.00e-2 | 3 | 2.49e-2 | 2.87e-2 | 2.67e-2 | 2.66e-2 | 115 | -6.62e-4 | +1.32e-3 | +4.10e-6 | +4.87e-5 |
| 123 | 3.00e-2 | 2 | 2.77e-2 | 3.11e-2 | 2.94e-2 | 3.11e-2 | 110 | +2.51e-4 | +1.06e-3 | +6.58e-4 | +1.68e-4 |
| 124 | 3.00e-2 | 2 | 2.81e-2 | 2.90e-2 | 2.85e-2 | 2.90e-2 | 109 | -8.19e-4 | +2.83e-4 | -2.68e-4 | +9.10e-5 |
| 125 | 3.00e-2 | 3 | 2.34e-2 | 2.90e-2 | 2.69e-2 | 2.34e-2 | 78 | -2.74e-3 | +3.01e-4 | -8.79e-4 | -1.97e-4 |
| 126 | 3.00e-2 | 3 | 2.36e-2 | 2.89e-2 | 2.60e-2 | 2.54e-2 | 81 | -1.60e-3 | +2.49e-3 | +3.28e-4 | -7.15e-5 |
| 127 | 3.00e-2 | 4 | 2.64e-2 | 3.04e-2 | 2.77e-2 | 2.69e-2 | 80 | -1.42e-3 | +1.58e-3 | +7.69e-5 | -3.99e-5 |
| 128 | 3.00e-2 | 3 | 2.52e-2 | 2.95e-2 | 2.68e-2 | 2.58e-2 | 80 | -1.66e-3 | +1.97e-3 | -8.27e-5 | -6.29e-5 |
| 129 | 3.00e-2 | 3 | 2.58e-2 | 3.15e-2 | 2.81e-2 | 2.58e-2 | 75 | -2.65e-3 | +1.85e-3 | -1.44e-4 | -1.15e-4 |
| 130 | 3.00e-2 | 5 | 2.55e-2 | 3.03e-2 | 2.66e-2 | 2.59e-2 | 66 | -2.62e-3 | +2.34e-3 | -1.44e-5 | -8.65e-5 |
| 131 | 3.00e-2 | 2 | 2.45e-2 | 3.05e-2 | 2.75e-2 | 3.05e-2 | 61 | -5.05e-4 | +3.60e-3 | +1.55e-3 | +2.44e-4 |
| 132 | 3.00e-2 | 5 | 2.31e-2 | 3.06e-2 | 2.51e-2 | 2.38e-2 | 55 | -4.92e-3 | +4.19e-3 | -4.98e-4 | -5.39e-5 |
| 133 | 3.00e-2 | 5 | 2.24e-2 | 2.93e-2 | 2.44e-2 | 2.40e-2 | 51 | -4.94e-3 | +4.35e-3 | +7.45e-5 | -1.45e-5 |
| 134 | 3.00e-2 | 6 | 2.06e-2 | 2.77e-2 | 2.25e-2 | 2.06e-2 | 37 | -6.22e-3 | +4.03e-3 | -4.77e-4 | -2.60e-4 |
| 135 | 3.00e-2 | 7 | 1.64e-2 | 2.77e-2 | 1.92e-2 | 1.88e-2 | 35 | -1.45e-2 | +1.32e-2 | -6.66e-5 | -1.43e-4 |
| 136 | 3.00e-2 | 9 | 1.43e-2 | 2.75e-2 | 1.75e-2 | 1.47e-2 | 29 | -1.56e-2 | +1.27e-2 | -1.19e-3 | -8.22e-4 |
| 137 | 3.00e-2 | 18 | 1.14e-2 | 2.77e-2 | 1.43e-2 | 1.21e-2 | 16 | -1.52e-2 | +1.52e-2 | -1.32e-3 | -1.13e-3 |
| 138 | 3.00e-2 | 10 | 9.28e-3 | 2.65e-2 | 1.37e-2 | 1.52e-2 | 15 | -6.56e-2 | +6.21e-2 | +2.78e-3 | +1.40e-3 |
| 139 | 3.00e-2 | 13 | 1.05e-2 | 2.35e-2 | 1.36e-2 | 1.28e-2 | 17 | -5.44e-2 | +5.62e-2 | +4.58e-4 | +3.47e-4 |
| 140 | 3.00e-2 | 3 | 1.33e-2 | 1.48e-2 | 1.38e-2 | 1.33e-2 | 247 | -4.19e-4 | +5.52e-3 | +2.36e-3 | +8.68e-4 |
| 141 | 3.00e-2 | 1 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 275 | +5.30e-3 | +5.30e-3 | +5.30e-3 | +1.31e-3 |
| 142 | 3.00e-2 | 1 | 6.04e-2 | 6.04e-2 | 6.04e-2 | 6.04e-2 | 273 | +1.95e-4 | +1.95e-4 | +1.95e-4 | +1.20e-3 |
| 143 | 3.00e-2 | 1 | 5.76e-2 | 5.76e-2 | 5.76e-2 | 5.76e-2 | 256 | -1.82e-4 | -1.82e-4 | -1.82e-4 | +1.06e-3 |
| 144 | 3.00e-2 | 1 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 297 | -4.88e-5 | -4.88e-5 | -4.88e-5 | +9.50e-4 |
| 146 | 3.00e-2 | 1 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 6.05e-2 | 320 | +1.98e-4 | +1.98e-4 | +1.98e-4 | +8.75e-4 |
| 147 | 3.00e-2 | 1 | 6.33e-2 | 6.33e-2 | 6.33e-2 | 6.33e-2 | 292 | +1.54e-4 | +1.54e-4 | +1.54e-4 | +8.03e-4 |
| 148 | 3.00e-2 | 1 | 5.93e-2 | 5.93e-2 | 5.93e-2 | 5.93e-2 | 306 | -2.13e-4 | -2.13e-4 | -2.13e-4 | +7.01e-4 |
| 149 | 3.00e-2 | 1 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 304 | +1.58e-4 | +1.58e-4 | +1.58e-4 | +6.47e-4 |
| 150 | 3.00e-3 | 1 | 6.21e-2 | 6.21e-2 | 6.21e-2 | 6.21e-2 | 329 | -4.18e-6 | -4.18e-6 | -4.18e-6 | +5.82e-4 |
| 151 | 3.00e-3 | 1 | 6.62e-2 | 6.62e-2 | 6.62e-2 | 6.62e-2 | 286 | +2.21e-4 | +2.21e-4 | +2.21e-4 | +5.46e-4 |
| 152 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 296 | -8.12e-3 | -8.12e-3 | -8.12e-3 | -3.20e-4 |
| 153 | 3.00e-3 | 1 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 318 | -7.79e-5 | -7.79e-5 | -7.79e-5 | -2.96e-4 |
| 154 | 3.00e-3 | 1 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 5.97e-3 | 301 | +6.82e-5 | +6.82e-5 | +6.82e-5 | -2.60e-4 |
| 156 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 374 | +3.79e-5 | +3.79e-5 | +3.79e-5 | -2.30e-4 |
| 157 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 300 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -1.84e-4 |
| 158 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 321 | -3.10e-4 | -3.10e-4 | -3.10e-4 | -1.97e-4 |
| 159 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 302 | +1.62e-4 | +1.62e-4 | +1.62e-4 | -1.61e-4 |
| 160 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 278 | -1.92e-5 | -1.92e-5 | -1.92e-5 | -1.47e-4 |
| 161 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 280 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -1.50e-4 |
| 162 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 345 | +5.12e-5 | +5.12e-5 | +5.12e-5 | -1.30e-4 |
| 163 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 321 | +2.00e-4 | +2.00e-4 | +2.00e-4 | -9.68e-5 |
| 164 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 273 | +2.86e-5 | +2.86e-5 | +2.86e-5 | -8.43e-5 |
| 165 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 288 | -2.23e-4 | -2.23e-4 | -2.23e-4 | -9.81e-5 |
| 166 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 258 | +9.55e-5 | +9.55e-5 | +9.55e-5 | -7.88e-5 |
| 167 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 260 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -8.34e-5 |
| 168 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 258 | -1.46e-5 | -1.46e-5 | -1.46e-5 | -7.65e-5 |
| 169 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 242 | +1.11e-4 | +1.11e-4 | +1.11e-4 | -5.78e-5 |
| 170 | 3.00e-3 | 1 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 5.78e-3 | 236 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -7.23e-5 |
| 171 | 3.00e-3 | 1 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 5.80e-3 | 238 | +1.19e-5 | +1.19e-5 | +1.19e-5 | -6.39e-5 |
| 172 | 3.00e-3 | 2 | 5.74e-3 | 6.05e-3 | 5.90e-3 | 6.05e-3 | 233 | -4.09e-5 | +2.29e-4 | +9.40e-5 | -3.25e-5 |
| 174 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 315 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -4.84e-5 |
| 175 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 279 | +4.36e-4 | +4.36e-4 | +4.36e-4 | +7.01e-8 |
| 176 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 279 | -8.32e-5 | -8.32e-5 | -8.32e-5 | -8.26e-6 |
| 177 | 3.00e-3 | 2 | 6.05e-3 | 6.14e-3 | 6.10e-3 | 6.05e-3 | 213 | -9.38e-5 | -6.62e-5 | -8.00e-5 | -2.18e-5 |
| 178 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 257 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -3.67e-5 |
| 179 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 302 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -1.23e-5 |
| 180 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 270 | +2.56e-4 | +2.56e-4 | +2.56e-4 | +1.45e-5 |
| 181 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 257 | -3.20e-4 | -3.20e-4 | -3.20e-4 | -1.89e-5 |
| 182 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 254 | -1.57e-5 | -1.57e-5 | -1.57e-5 | -1.86e-5 |
| 183 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 236 | +9.42e-5 | +9.42e-5 | +9.42e-5 | -7.32e-6 |
| 184 | 3.00e-3 | 2 | 5.72e-3 | 6.01e-3 | 5.86e-3 | 5.72e-3 | 197 | -2.48e-4 | -1.43e-4 | -1.96e-4 | -4.36e-5 |
| 185 | 3.00e-3 | 1 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 217 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -5.18e-5 |
| 186 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 227 | +1.76e-4 | +1.76e-4 | +1.76e-4 | -2.91e-5 |
| 187 | 3.00e-3 | 2 | 5.92e-3 | 5.93e-3 | 5.92e-3 | 5.92e-3 | 219 | -1.40e-5 | +1.06e-4 | +4.58e-5 | -1.54e-5 |
| 189 | 3.00e-3 | 2 | 5.85e-3 | 6.60e-3 | 6.22e-3 | 6.60e-3 | 219 | -3.88e-5 | +5.52e-4 | +2.56e-4 | +3.92e-5 |
| 190 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 250 | -3.90e-4 | -3.90e-4 | -3.90e-4 | -3.78e-6 |
| 191 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 251 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +1.21e-5 |
| 192 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 240 | -7.93e-5 | -7.93e-5 | -7.93e-5 | +2.92e-6 |
| 193 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 236 | +3.40e-5 | +3.40e-5 | +3.40e-5 | +6.03e-6 |
| 194 | 3.00e-3 | 2 | 6.07e-3 | 6.08e-3 | 6.07e-3 | 6.07e-3 | 181 | -5.67e-5 | -3.58e-6 | -3.01e-5 | -5.75e-7 |
| 195 | 3.00e-3 | 1 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 213 | -5.10e-4 | -5.10e-4 | -5.10e-4 | -5.15e-5 |
| 196 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 215 | +4.19e-4 | +4.19e-4 | +4.19e-4 | -4.46e-6 |
| 197 | 3.00e-3 | 2 | 5.68e-3 | 6.07e-3 | 5.88e-3 | 6.07e-3 | 181 | -2.08e-4 | +3.63e-4 | +7.75e-5 | +1.40e-5 |
| 198 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 210 | -4.37e-4 | -4.37e-4 | -4.37e-4 | -3.11e-5 |
| 199 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 231 | +2.83e-4 | +2.83e-4 | +2.83e-4 | +2.99e-7 |

