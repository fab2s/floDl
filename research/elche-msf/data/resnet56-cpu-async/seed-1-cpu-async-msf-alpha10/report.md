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
- resnet-graph/nccl-async
- resnet-graph/cpu-sync
- resnet-graph/cpu-cadence

## Per-Model Results

GPU columns = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | GPU2 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|------|----------|
| cpu-async | 0.015590 | 0.9288 | +0.0163 | 5095.8 | 286 | 254.9 | 100% | 100% | 100% | 7.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9288 | cpu-async | - | - |

## Eval Quality (vs solo-0)

| Model | cpu-async |
|-------|-----------|
| resnet-graph | - |

## Convergence Quality (loss ratio vs solo-0)

| Model | cpu-async |
|-------|-----------|
| resnet-graph | - |

## Per-Epoch Loss Trajectory

### resnet-graph (sampled, 200 epochs)

| Mode | E0 | E10 | E21 | E31 | E42 | E52 | E63 | E73 | E84 | E94 | E105 | E115 | E126 | E136 | E147 | E157 | E168 | E178 | E189 | E199 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| cpu-async | 2.3507 | 1.1553 | 0.6999 | 0.5883 | 0.5267 | 0.4918 | 0.4597 | 0.4535 | 0.4509 | 0.4228 | 0.1672 | 0.1233 | 0.0981 | 0.0869 | 0.0894 | 0.0304 | 0.0227 | 0.0197 | 0.0167 | 0.0156 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3524 | 1.1 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3283 | 1.2 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.3194 | 1.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 514 | 513 | 578 | 574 | 572 | 569 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 5093.9 | 1.9 | epoch-boundary(199) |
| cpu-async | gpu2 | 5094.0 | 1.8 | epoch-boundary(199) |
| cpu-async | gpu0 | 0.3 | 0.9 | unexplained |
| cpu-async | gpu1 | 2113.2 | 0.7 | epoch-boundary(83) |
| cpu-async | gpu2 | 2113.1 | 0.6 | epoch-boundary(83) |
| cpu-async | gpu2 | 2163.7 | 0.5 | epoch-boundary(85) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.9s | 0.9s |
| resnet-graph | cpu-async | gpu1 | 2.6s | 0.0s | 0.0s | 0.0s | 3.7s |
| resnet-graph | cpu-async | gpu2 | 3.0s | 0.0s | 0.0s | 0.0s | 3.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 265 | 0 | 286 | 254.9 | 19073/24159 | 286 | 254.9 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 304.5 | 6.0% |

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
| resnet-graph | cpu-async | 199 | 286 | 0 | 4.14e-3 | -8.86e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 286 | 9.23e-2 | 9.59e-2 | 3.72e-3 | 9.35e-1 | 17.8 | -4.71e-4 | 4.22e-3 |
| resnet-graph | cpu-async | 1 | 286 | 9.40e-2 | 9.98e-2 | 3.96e-3 | 1.02e0 | 39.5 | -5.00e-4 | 4.33e-3 |
| resnet-graph | cpu-async | 2 | 286 | 9.49e-2 | 1.05e-1 | 3.93e-3 | 1.15e0 | 42.7 | -5.24e-4 | 4.50e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9979 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9937 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9957 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 53 (0,1,2,4,8,9,10,11…149,150) | 0 (—) | — | 0,1,2,4,8,9,10,11…149,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 4 | 4 |
| resnet-graph | cpu-async | 0e0 | 5 | 1 | 1 |
| resnet-graph | cpu-async | 0e0 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-4 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-4 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-4 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-3 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-3 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-3 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-2 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-2 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-2 | 10 | 0 | 0 |
| resnet-graph | cpu-async | 1e-1 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 1e-1 | 5 | 0 | 0 |
| resnet-graph | cpu-async | 1e-1 | 10 | 0 | 0 |

### Predictive Value by LR Window (top scale)

Per-LR-window Pearson `r(λ_raw_t, ln(D_{t+1}))`. Pairs that straddle a window boundary are excluded so the LR-drop collapse cannot leak in as artefactual signal. Reading: a clean R1 (exponential growth at fixed LR) shows up as *non-zero* `r` within each window, with sign and magnitude that may differ between warmup, post-drop transient, and late-training phases.

| Model | Mode | LR | Epochs | n_pairs | r(λ → ln D_{t+1}) |
|-------|------|---:|--------|--------:|----------------:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 149 | +0.278 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 67 | +0.208 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 66 | +0.117 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 284 | +0.048 | 198 | +0.249 | +0.518 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 285 | 6.07e1–1.08e2 | 7.50e1 | 4.62e-3 | 1.58e-2 | 2.39e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 151 | 30–77553 | +2.412e-5 | 0.420 | +2.470e-5 | 0.428 | 99 | +1.101e-5 | 0.354 | 30–874 | +2.984e-3 | 0.607 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 137 | 888–77553 | +2.013e-5 | 0.459 | +2.042e-5 | 0.461 | 98 | +9.908e-6 | 0.342 | 79–874 | +2.954e-3 | 0.706 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 68 | 78224–116672 | +2.012e-5 | 0.389 | +2.023e-5 | 0.394 | 50 | +2.299e-5 | 0.747 | 390–746 | +1.364e-4 | 0.001 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 67 | 117447–156020 | -7.597e-6 | 0.069 | -7.994e-6 | 0.077 | 50 | -9.562e-6 | 0.084 | 414–775 | +1.417e-3 | 0.136 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +2.984e-3 | r0: +2.943e-3, r1: +3.017e-3, r2: +3.005e-3 | r0: 0.603, r1: 0.613, r2: 0.602 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +2.954e-3 | r0: +2.956e-3, r1: +2.967e-3, r2: +2.944e-3 | r0: 0.700, r1: 0.704, r2: 0.709 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +1.364e-4 | r0: +8.526e-5, r1: +1.745e-4, r2: +1.480e-4 | r0: 0.001, r1: 0.002, r2: 0.001 | 2.05× | ⚠ framing breaking |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.417e-3 | r0: +1.405e-3, r1: +1.370e-3, r2: +1.477e-3 | r0: 0.133, r1: 0.129, r2: 0.144 | 1.08× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `▆▇▇▇█████████████████████▄▄▄▅▅▅▅▅▅▅▆▆▃▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇██████████████████████▇▇███████████▇███████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 6.92e-3 | 1.15e0 | 1.37e-1 | 1.77e-2 | 25 | -1.23e-1 | +4.91e-2 | -2.98e-2 | -1.65e-2 |
| 1 | 3.00e-1 | 8 | 1.87e-2 | 3.52e-2 | 2.28e-2 | 2.10e-2 | 30 | -2.04e-2 | +8.91e-3 | -1.06e-3 | -6.52e-3 |
| 2 | 3.00e-1 | 8 | 2.26e-2 | 3.65e-2 | 2.61e-2 | 2.49e-2 | 28 | -1.66e-2 | +8.16e-3 | -6.18e-4 | -2.92e-3 |
| 3 | 3.00e-1 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 46 | +7.40e-3 | +7.40e-3 | +7.40e-3 | -1.85e-3 |
| 4 | 3.00e-1 | 2 | 7.68e-2 | 7.95e-2 | 7.82e-2 | 7.68e-2 | 233 | -1.48e-4 | +2.65e-3 | +1.25e-3 | -1.25e-3 |
| 6 | 3.00e-1 | 2 | 9.39e-2 | 9.98e-2 | 9.69e-2 | 9.39e-2 | 233 | -2.58e-4 | +7.90e-4 | +2.66e-4 | -9.59e-4 |
| 7 | 3.00e-1 | 1 | 1.08e-1 | 1.08e-1 | 1.08e-1 | 1.08e-1 | 272 | +4.96e-4 | +4.96e-4 | +4.96e-4 | -8.10e-4 |
| 8 | 3.00e-1 | 1 | 1.18e-1 | 1.18e-1 | 1.18e-1 | 1.18e-1 | 256 | +3.72e-4 | +3.72e-4 | +3.72e-4 | -6.89e-4 |
| 9 | 3.00e-1 | 1 | 1.28e-1 | 1.28e-1 | 1.28e-1 | 1.28e-1 | 256 | +3.20e-4 | +3.20e-4 | +3.20e-4 | -5.86e-4 |
| 10 | 3.00e-1 | 1 | 1.34e-1 | 1.34e-1 | 1.34e-1 | 1.34e-1 | 250 | +1.82e-4 | +1.82e-4 | +1.82e-4 | -5.08e-4 |
| 11 | 3.00e-1 | 1 | 1.46e-1 | 1.46e-1 | 1.46e-1 | 1.46e-1 | 269 | +3.08e-4 | +3.08e-4 | +3.08e-4 | -4.25e-4 |
| 12 | 3.00e-1 | 1 | 1.55e-1 | 1.55e-1 | 1.55e-1 | 1.55e-1 | 265 | +2.35e-4 | +2.35e-4 | +2.35e-4 | -3.58e-4 |
| 13 | 3.00e-1 | 1 | 1.57e-1 | 1.57e-1 | 1.57e-1 | 1.57e-1 | 261 | +5.16e-5 | +5.16e-5 | +5.16e-5 | -3.16e-4 |
| 14 | 3.00e-1 | 1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 252 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -2.69e-4 |
| 15 | 3.00e-1 | 1 | 1.60e-1 | 1.60e-1 | 1.60e-1 | 1.60e-1 | 232 | -9.50e-5 | -9.50e-5 | -9.50e-5 | -2.51e-4 |
| 16 | 3.00e-1 | 2 | 1.58e-1 | 1.67e-1 | 1.63e-1 | 1.58e-1 | 202 | -2.75e-4 | +1.79e-4 | -4.76e-5 | -2.15e-4 |
| 17 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 242 | +2.40e-4 | +2.40e-4 | +2.40e-4 | -1.69e-4 |
| 18 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 240 | +1.77e-4 | +1.77e-4 | +1.77e-4 | -1.34e-4 |
| 19 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 248 | +4.52e-5 | +4.52e-5 | +4.52e-5 | -1.16e-4 |
| 20 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 244 | -1.21e-5 | -1.21e-5 | -1.21e-5 | -1.06e-4 |
| 21 | 3.00e-1 | 2 | 1.68e-1 | 1.81e-1 | 1.74e-1 | 1.68e-1 | 203 | -3.81e-4 | +1.10e-4 | -1.35e-4 | -1.14e-4 |
| 22 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 241 | +3.43e-4 | +3.43e-4 | +3.43e-4 | -6.79e-5 |
| 23 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 246 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -5.05e-5 |
| 24 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 238 | -7.19e-5 | -7.19e-5 | -7.19e-5 | -5.26e-5 |
| 25 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 248 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -3.05e-5 |
| 26 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 245 | -3.53e-5 | -3.53e-5 | -3.53e-5 | -3.10e-5 |
| 27 | 3.00e-1 | 2 | 1.74e-1 | 1.85e-1 | 1.79e-1 | 1.74e-1 | 190 | -3.20e-4 | -1.11e-4 | -2.16e-4 | -6.72e-5 |
| 28 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 241 | +4.24e-4 | +4.24e-4 | +4.24e-4 | -1.80e-5 |
| 29 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 222 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -2.95e-5 |
| 30 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 244 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -1.53e-5 |
| 31 | 3.00e-1 | 2 | 1.76e-1 | 1.84e-1 | 1.80e-1 | 1.76e-1 | 194 | -2.30e-4 | -1.87e-4 | -2.09e-4 | -5.23e-5 |
| 32 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 220 | +3.80e-4 | +3.80e-4 | +3.80e-4 | -9.03e-6 |
| 33 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 228 | -1.06e-6 | -1.06e-6 | -1.06e-6 | -8.24e-6 |
| 34 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 225 | -8.52e-5 | -8.52e-5 | -8.52e-5 | -1.59e-5 |
| 35 | 3.00e-1 | 2 | 1.80e-1 | 1.93e-1 | 1.87e-1 | 1.80e-1 | 190 | -3.86e-4 | +1.23e-4 | -1.32e-4 | -4.05e-5 |
| 36 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 229 | +3.62e-4 | +3.62e-4 | +3.62e-4 | -2.44e-7 |
| 37 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 223 | -1.65e-5 | -1.65e-5 | -1.65e-5 | -1.87e-6 |
| 38 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 222 | -7.50e-7 | -7.50e-7 | -7.50e-7 | -1.76e-6 |
| 39 | 3.00e-1 | 2 | 1.82e-1 | 1.86e-1 | 1.84e-1 | 1.82e-1 | 190 | -1.99e-4 | -1.32e-4 | -1.66e-4 | -3.26e-5 |
| 40 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 233 | +3.59e-4 | +3.59e-4 | +3.59e-4 | +6.64e-6 |
| 41 | 3.00e-1 | 2 | 1.75e-1 | 1.86e-1 | 1.81e-1 | 1.75e-1 | 169 | -3.59e-4 | -2.83e-4 | -3.21e-4 | -5.60e-5 |
| 42 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 207 | +2.83e-4 | +2.83e-4 | +2.83e-4 | -2.21e-5 |
| 43 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 205 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -8.03e-6 |
| 44 | 3.00e-1 | 2 | 1.76e-1 | 1.93e-1 | 1.84e-1 | 1.76e-1 | 170 | -5.43e-4 | +5.83e-5 | -2.42e-4 | -5.55e-5 |
| 45 | 3.00e-1 | 1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 207 | +4.08e-4 | +4.08e-4 | +4.08e-4 | -9.16e-6 |
| 46 | 3.00e-1 | 2 | 1.76e-1 | 1.98e-1 | 1.87e-1 | 1.76e-1 | 169 | -7.18e-4 | +1.64e-4 | -2.77e-4 | -6.45e-5 |
| 47 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 193 | +2.55e-4 | +2.55e-4 | +2.55e-4 | -3.25e-5 |
| 48 | 3.00e-1 | 2 | 1.76e-1 | 1.87e-1 | 1.82e-1 | 1.76e-1 | 160 | -3.81e-4 | +7.24e-5 | -1.54e-4 | -5.79e-5 |
| 49 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 196 | +3.74e-4 | +3.74e-4 | +3.74e-4 | -1.46e-5 |
| 50 | 3.00e-1 | 2 | 1.76e-1 | 1.90e-1 | 1.83e-1 | 1.76e-1 | 163 | -4.82e-4 | +1.91e-5 | -2.32e-4 | -5.84e-5 |
| 51 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 195 | +4.41e-4 | +4.41e-4 | +4.41e-4 | -8.45e-6 |
| 52 | 3.00e-1 | 2 | 1.75e-1 | 1.98e-1 | 1.87e-1 | 1.75e-1 | 160 | -7.64e-4 | +1.48e-4 | -3.08e-4 | -6.99e-5 |
| 53 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 192 | +3.12e-4 | +3.12e-4 | +3.12e-4 | -3.17e-5 |
| 54 | 3.00e-1 | 2 | 1.77e-1 | 1.89e-1 | 1.83e-1 | 1.77e-1 | 162 | -4.28e-4 | +9.74e-5 | -1.65e-4 | -5.97e-5 |
| 55 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 191 | +4.60e-4 | +4.60e-4 | +4.60e-4 | -7.71e-6 |
| 56 | 3.00e-1 | 2 | 1.71e-1 | 1.90e-1 | 1.81e-1 | 1.71e-1 | 150 | -7.07e-4 | -7.95e-5 | -3.93e-4 | -8.41e-5 |
| 57 | 3.00e-1 | 2 | 1.73e-1 | 1.81e-1 | 1.77e-1 | 1.73e-1 | 150 | -2.94e-4 | +3.04e-4 | +4.75e-6 | -7.02e-5 |
| 58 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 209 | +6.39e-4 | +6.39e-4 | +6.39e-4 | +7.64e-7 |
| 59 | 3.00e-1 | 2 | 1.70e-1 | 1.99e-1 | 1.84e-1 | 1.70e-1 | 148 | -1.07e-3 | +2.86e-5 | -5.19e-4 | -1.03e-4 |
| 60 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 188 | +5.83e-4 | +5.83e-4 | +5.83e-4 | -3.47e-5 |
| 61 | 3.00e-1 | 2 | 1.75e-1 | 1.89e-1 | 1.82e-1 | 1.75e-1 | 148 | -5.44e-4 | -7.11e-6 | -2.76e-4 | -8.32e-5 |
| 62 | 3.00e-1 | 2 | 1.74e-1 | 1.82e-1 | 1.78e-1 | 1.74e-1 | 153 | -3.12e-4 | +2.52e-4 | -3.00e-5 | -7.59e-5 |
| 63 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 194 | +6.95e-4 | +6.95e-4 | +6.95e-4 | +1.25e-6 |
| 64 | 3.00e-1 | 2 | 1.73e-1 | 2.00e-1 | 1.86e-1 | 1.73e-1 | 150 | -9.74e-4 | +2.42e-5 | -4.75e-4 | -9.43e-5 |
| 65 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 188 | +5.59e-4 | +5.59e-4 | +5.59e-4 | -2.90e-5 |
| 66 | 3.00e-1 | 2 | 1.65e-1 | 2.06e-1 | 1.86e-1 | 2.06e-1 | 271 | -1.11e-3 | +8.15e-4 | -1.48e-4 | -4.19e-5 |
| 67 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 252 | +8.30e-5 | +8.30e-5 | +8.30e-5 | -2.94e-5 |
| 68 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 270 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -1.60e-5 |
| 69 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 249 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -3.26e-5 |
| 70 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 239 | +7.97e-5 | +7.97e-5 | +7.97e-5 | -2.13e-5 |
| 71 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 278 | +1.46e-4 | +1.46e-4 | +1.46e-4 | -4.65e-6 |
| 72 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 241 | -3.11e-4 | -3.11e-4 | -3.11e-4 | -3.53e-5 |
| 73 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 260 | +9.25e-5 | +9.25e-5 | +9.25e-5 | -2.25e-5 |
| 74 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 253 | +8.31e-5 | +8.31e-5 | +8.31e-5 | -1.19e-5 |
| 75 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 270 | +5.99e-5 | +5.99e-5 | +5.99e-5 | -4.75e-6 |
| 76 | 3.00e-1 | 2 | 1.98e-1 | 2.13e-1 | 2.06e-1 | 1.98e-1 | 208 | -3.40e-4 | -6.83e-5 | -2.04e-4 | -4.40e-5 |
| 77 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 252 | +2.22e-4 | +2.22e-4 | +2.22e-4 | -1.74e-5 |
| 78 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 244 | +1.17e-5 | +1.17e-5 | +1.17e-5 | -1.45e-5 |
| 79 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 236 | -9.87e-5 | -9.87e-5 | -9.87e-5 | -2.29e-5 |
| 80 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 262 | +1.42e-4 | +1.42e-4 | +1.42e-4 | -6.40e-6 |
| 81 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 227 | -1.73e-4 | -1.73e-4 | -1.73e-4 | -2.31e-5 |
| 82 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 241 | +7.00e-5 | +7.00e-5 | +7.00e-5 | -1.38e-5 |
| 83 | 3.00e-1 | 2 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 254 | -2.02e-4 | +5.96e-6 | -9.82e-5 | -2.88e-5 |
| 84 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 289 | +3.71e-4 | +3.71e-4 | +3.71e-4 | +1.12e-5 |
| 85 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 278 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -1.15e-6 |
| 86 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 264 | -2.39e-5 | -2.39e-5 | -2.39e-5 | -3.43e-6 |
| 87 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 261 | -2.86e-5 | -2.86e-5 | -2.86e-5 | -5.95e-6 |
| 88 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 252 | -1.69e-5 | -1.69e-5 | -1.69e-5 | -7.04e-6 |
| 89 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 240 | -5.36e-5 | -5.36e-5 | -5.36e-5 | -1.17e-5 |
| 90 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 242 | -3.69e-6 | -3.69e-6 | -3.69e-6 | -1.09e-5 |
| 91 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 250 | +4.51e-6 | +4.51e-6 | +4.51e-6 | -9.36e-6 |
| 92 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 243 | +6.90e-5 | +6.90e-5 | +6.90e-5 | -1.52e-6 |
| 93 | 3.00e-1 | 2 | 1.99e-1 | 2.14e-1 | 2.07e-1 | 1.99e-1 | 212 | -3.49e-4 | +2.09e-5 | -1.64e-4 | -3.43e-5 |
| 94 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 247 | +2.38e-4 | +2.38e-4 | +2.38e-4 | -7.10e-6 |
| 95 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 250 | +3.02e-5 | +3.02e-5 | +3.02e-5 | -3.37e-6 |
| 96 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 250 | +3.85e-5 | +3.85e-5 | +3.85e-5 | +8.12e-7 |
| 97 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 295 | +2.13e-4 | +2.13e-4 | +2.13e-4 | +2.21e-5 |
| 98 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 232 | -4.93e-4 | -4.93e-4 | -4.93e-4 | -2.94e-5 |
| 99 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 232 | -5.59e-5 | -5.59e-5 | -5.59e-5 | -3.21e-5 |
| 100 | 3.00e-2 | 2 | 1.88e-2 | 2.08e-1 | 1.13e-1 | 1.88e-2 | 207 | -1.16e-2 | +1.33e-4 | -5.74e-3 | -1.18e-3 |
| 101 | 3.00e-2 | 1 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 2.10e-2 | 250 | +4.49e-4 | +4.49e-4 | +4.49e-4 | -1.01e-3 |
| 102 | 3.00e-2 | 1 | 2.14e-2 | 2.14e-2 | 2.14e-2 | 2.14e-2 | 243 | +8.26e-5 | +8.26e-5 | +8.26e-5 | -9.03e-4 |
| 103 | 3.00e-2 | 1 | 2.39e-2 | 2.39e-2 | 2.39e-2 | 2.39e-2 | 258 | +4.14e-4 | +4.14e-4 | +4.14e-4 | -7.72e-4 |
| 104 | 3.00e-2 | 1 | 2.49e-2 | 2.49e-2 | 2.49e-2 | 2.49e-2 | 260 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -6.78e-4 |
| 105 | 3.00e-2 | 1 | 2.16e-2 | 2.16e-2 | 2.16e-2 | 2.16e-2 | 185 | -7.82e-4 | -7.82e-4 | -7.82e-4 | -6.88e-4 |
| 106 | 3.00e-2 | 2 | 2.42e-2 | 2.67e-2 | 2.54e-2 | 2.42e-2 | 206 | -4.93e-4 | +7.99e-4 | +1.53e-4 | -5.35e-4 |
| 107 | 3.00e-2 | 1 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 2.60e-2 | 241 | +3.03e-4 | +3.03e-4 | +3.03e-4 | -4.51e-4 |
| 108 | 3.00e-2 | 1 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 2.79e-2 | 255 | +2.73e-4 | +2.73e-4 | +2.73e-4 | -3.79e-4 |
| 109 | 3.00e-2 | 1 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 2.68e-2 | 218 | -1.75e-4 | -1.75e-4 | -1.75e-4 | -3.58e-4 |
| 110 | 3.00e-2 | 2 | 2.69e-2 | 2.82e-2 | 2.75e-2 | 2.69e-2 | 195 | -2.41e-4 | +2.14e-4 | -1.36e-5 | -2.95e-4 |
| 111 | 3.00e-2 | 1 | 3.05e-2 | 3.05e-2 | 3.05e-2 | 3.05e-2 | 238 | +5.27e-4 | +5.27e-4 | +5.27e-4 | -2.13e-4 |
| 112 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 211 | -3.31e-4 | -3.31e-4 | -3.31e-4 | -2.25e-4 |
| 113 | 3.00e-2 | 2 | 2.78e-2 | 2.89e-2 | 2.84e-2 | 2.78e-2 | 176 | -2.32e-4 | +9.04e-5 | -7.07e-5 | -1.97e-4 |
| 114 | 3.00e-2 | 1 | 3.05e-2 | 3.05e-2 | 3.05e-2 | 3.05e-2 | 216 | +4.31e-4 | +4.31e-4 | +4.31e-4 | -1.34e-4 |
| 115 | 3.00e-2 | 1 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 216 | +2.61e-4 | +2.61e-4 | +2.61e-4 | -9.47e-5 |
| 116 | 3.00e-2 | 2 | 3.00e-2 | 3.28e-2 | 3.14e-2 | 3.00e-2 | 175 | -5.07e-4 | +7.06e-5 | -2.18e-4 | -1.21e-4 |
| 117 | 3.00e-2 | 1 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 3.34e-2 | 227 | +4.74e-4 | +4.74e-4 | +4.74e-4 | -6.16e-5 |
| 118 | 3.00e-2 | 1 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 3.36e-2 | 213 | +2.87e-5 | +2.87e-5 | +2.87e-5 | -5.25e-5 |
| 119 | 3.00e-2 | 2 | 3.15e-2 | 3.51e-2 | 3.33e-2 | 3.15e-2 | 176 | -6.07e-4 | +1.94e-4 | -2.07e-4 | -8.58e-5 |
| 120 | 3.00e-2 | 1 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 224 | +6.26e-4 | +6.26e-4 | +6.26e-4 | -1.47e-5 |
| 121 | 3.00e-2 | 2 | 3.40e-2 | 3.54e-2 | 3.47e-2 | 3.40e-2 | 174 | -2.27e-4 | -1.15e-4 | -1.71e-4 | -4.49e-5 |
| 122 | 3.00e-2 | 1 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 219 | +4.56e-4 | +4.56e-4 | +4.56e-4 | +5.13e-6 |
| 123 | 3.00e-2 | 1 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 3.74e-2 | 213 | -2.10e-5 | -2.10e-5 | -2.10e-5 | +2.51e-6 |
| 124 | 3.00e-2 | 2 | 3.48e-2 | 3.88e-2 | 3.68e-2 | 3.48e-2 | 168 | -6.57e-4 | +1.66e-4 | -2.45e-4 | -4.87e-5 |
| 125 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 196 | +4.74e-4 | +4.74e-4 | +4.74e-4 | +3.60e-6 |
| 126 | 3.00e-2 | 2 | 3.57e-2 | 3.94e-2 | 3.75e-2 | 3.57e-2 | 156 | -6.31e-4 | +1.59e-4 | -2.36e-4 | -4.59e-5 |
| 127 | 3.00e-2 | 1 | 3.88e-2 | 3.88e-2 | 3.88e-2 | 3.88e-2 | 202 | +4.14e-4 | +4.14e-4 | +4.14e-4 | +1.63e-7 |
| 128 | 3.00e-2 | 2 | 3.71e-2 | 4.19e-2 | 3.95e-2 | 3.71e-2 | 166 | -7.36e-4 | +3.63e-4 | -1.87e-4 | -4.09e-5 |
| 129 | 3.00e-2 | 1 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 221 | +7.36e-4 | +7.36e-4 | +7.36e-4 | +3.68e-5 |
| 130 | 3.00e-2 | 2 | 3.75e-2 | 4.23e-2 | 3.99e-2 | 3.75e-2 | 152 | -7.94e-4 | -1.47e-4 | -4.70e-4 | -6.28e-5 |
| 131 | 3.00e-2 | 1 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 197 | +4.92e-4 | +4.92e-4 | +4.92e-4 | -7.38e-6 |
| 132 | 3.00e-2 | 2 | 3.95e-2 | 4.54e-2 | 4.24e-2 | 3.95e-2 | 154 | -9.10e-4 | +4.07e-4 | -2.51e-4 | -6.03e-5 |
| 133 | 3.00e-2 | 1 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 190 | +3.37e-4 | +3.37e-4 | +3.37e-4 | -2.06e-5 |
| 134 | 3.00e-2 | 2 | 4.15e-2 | 4.64e-2 | 4.39e-2 | 4.15e-2 | 153 | -7.24e-4 | +4.64e-4 | -1.30e-4 | -4.73e-5 |
| 135 | 3.00e-2 | 1 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 190 | +2.84e-4 | +2.84e-4 | +2.84e-4 | -1.41e-5 |
| 136 | 3.00e-2 | 2 | 4.21e-2 | 4.46e-2 | 4.34e-2 | 4.21e-2 | 151 | -3.81e-4 | +9.65e-5 | -1.42e-4 | -4.08e-5 |
| 137 | 3.00e-2 | 2 | 4.19e-2 | 4.54e-2 | 4.36e-2 | 4.19e-2 | 149 | -5.32e-4 | +3.90e-4 | -7.08e-5 | -5.11e-5 |
| 138 | 3.00e-2 | 2 | 4.32e-2 | 4.66e-2 | 4.49e-2 | 4.32e-2 | 151 | -5.05e-4 | +5.62e-4 | +2.87e-5 | -4.13e-5 |
| 139 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 195 | +4.67e-4 | +4.67e-4 | +4.67e-4 | +9.48e-6 |
| 140 | 3.00e-2 | 2 | 4.14e-2 | 4.72e-2 | 4.43e-2 | 4.14e-2 | 133 | -9.81e-4 | -1.64e-5 | -4.98e-4 | -9.18e-5 |
| 141 | 3.00e-2 | 1 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 171 | +7.27e-4 | +7.27e-4 | +7.27e-4 | -9.98e-6 |
| 142 | 3.00e-2 | 2 | 4.20e-2 | 5.40e-2 | 4.80e-2 | 5.40e-2 | 248 | -8.23e-4 | +1.02e-3 | +9.80e-5 | +1.97e-5 |
| 143 | 3.00e-2 | 1 | 6.11e-2 | 6.11e-2 | 6.11e-2 | 6.11e-2 | 266 | +4.63e-4 | +4.63e-4 | +4.63e-4 | +6.40e-5 |
| 144 | 3.00e-2 | 1 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 257 | -1.13e-4 | -1.13e-4 | -1.13e-4 | +4.63e-5 |
| 145 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 261 | -1.04e-4 | -1.04e-4 | -1.04e-4 | +3.12e-5 |
| 146 | 3.00e-2 | 1 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 265 | +1.03e-4 | +1.03e-4 | +1.03e-4 | +3.84e-5 |
| 147 | 3.00e-2 | 1 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 5.95e-2 | 243 | +5.66e-6 | +5.66e-6 | +5.66e-6 | +3.51e-5 |
| 148 | 3.00e-2 | 1 | 6.06e-2 | 6.06e-2 | 6.06e-2 | 6.06e-2 | 250 | +7.77e-5 | +7.77e-5 | +7.77e-5 | +3.94e-5 |
| 149 | 3.00e-2 | 1 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 6.28e-2 | 269 | +1.32e-4 | +1.32e-4 | +1.32e-4 | +4.87e-5 |
| 150 | 3.00e-3 | 1 | 6.48e-2 | 6.48e-2 | 6.48e-2 | 6.48e-2 | 283 | +1.08e-4 | +1.08e-4 | +1.08e-4 | +5.46e-5 |
| 151 | 3.00e-3 | 2 | 4.27e-3 | 5.08e-3 | 4.68e-3 | 4.27e-3 | 213 | -9.79e-3 | -8.25e-4 | -5.31e-3 | -9.19e-4 |
| 152 | 3.00e-3 | 1 | 4.87e-3 | 4.87e-3 | 4.87e-3 | 4.87e-3 | 253 | +5.27e-4 | +5.27e-4 | +5.27e-4 | -7.74e-4 |
| 153 | 3.00e-3 | 1 | 4.53e-3 | 4.53e-3 | 4.53e-3 | 4.53e-3 | 257 | -2.81e-4 | -2.81e-4 | -2.81e-4 | -7.25e-4 |
| 154 | 3.00e-3 | 1 | 4.89e-3 | 4.89e-3 | 4.89e-3 | 4.89e-3 | 269 | +2.84e-4 | +2.84e-4 | +2.84e-4 | -6.24e-4 |
| 155 | 3.00e-3 | 1 | 4.57e-3 | 4.57e-3 | 4.57e-3 | 4.57e-3 | 251 | -2.74e-4 | -2.74e-4 | -2.74e-4 | -5.89e-4 |
| 156 | 3.00e-3 | 1 | 4.30e-3 | 4.30e-3 | 4.30e-3 | 4.30e-3 | 234 | -2.53e-4 | -2.53e-4 | -2.53e-4 | -5.56e-4 |
| 157 | 3.00e-3 | 1 | 4.42e-3 | 4.42e-3 | 4.42e-3 | 4.42e-3 | 240 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -4.89e-4 |
| 158 | 3.00e-3 | 1 | 4.59e-3 | 4.59e-3 | 4.59e-3 | 4.59e-3 | 251 | +1.58e-4 | +1.58e-4 | +1.58e-4 | -4.25e-4 |
| 159 | 3.00e-3 | 1 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 265 | +5.25e-5 | +5.25e-5 | +5.25e-5 | -3.77e-4 |
| 160 | 3.00e-3 | 2 | 4.32e-3 | 4.43e-3 | 4.37e-3 | 4.32e-3 | 209 | -2.16e-4 | -1.15e-4 | -1.66e-4 | -3.36e-4 |
| 161 | 3.00e-3 | 1 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 247 | +3.63e-4 | +3.63e-4 | +3.63e-4 | -2.66e-4 |
| 162 | 3.00e-3 | 1 | 4.55e-3 | 4.55e-3 | 4.55e-3 | 4.55e-3 | 234 | -1.64e-4 | -1.64e-4 | -1.64e-4 | -2.56e-4 |
| 163 | 3.00e-3 | 1 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 239 | +9.88e-5 | +9.88e-5 | +9.88e-5 | -2.21e-4 |
| 164 | 3.00e-3 | 1 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 248 | +1.20e-4 | +1.20e-4 | +1.20e-4 | -1.87e-4 |
| 165 | 3.00e-3 | 1 | 4.97e-3 | 4.97e-3 | 4.97e-3 | 4.97e-3 | 275 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -1.55e-4 |
| 166 | 3.00e-3 | 2 | 4.27e-3 | 4.58e-3 | 4.42e-3 | 4.27e-3 | 189 | -3.62e-4 | -3.53e-4 | -3.57e-4 | -1.94e-4 |
| 167 | 3.00e-3 | 1 | 4.69e-3 | 4.69e-3 | 4.69e-3 | 4.69e-3 | 230 | +4.05e-4 | +4.05e-4 | +4.05e-4 | -1.34e-4 |
| 168 | 3.00e-3 | 1 | 4.57e-3 | 4.57e-3 | 4.57e-3 | 4.57e-3 | 218 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -1.32e-4 |
| 169 | 3.00e-3 | 2 | 4.39e-3 | 4.60e-3 | 4.49e-3 | 4.39e-3 | 182 | -2.62e-4 | +2.78e-5 | -1.17e-4 | -1.31e-4 |
| 170 | 3.00e-3 | 1 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 237 | +3.15e-4 | +3.15e-4 | +3.15e-4 | -8.61e-5 |
| 171 | 3.00e-3 | 1 | 4.84e-3 | 4.84e-3 | 4.84e-3 | 4.84e-3 | 243 | +9.37e-5 | +9.37e-5 | +9.37e-5 | -6.82e-5 |
| 172 | 3.00e-3 | 1 | 4.59e-3 | 4.59e-3 | 4.59e-3 | 4.59e-3 | 240 | -2.18e-4 | -2.18e-4 | -2.18e-4 | -8.31e-5 |
| 173 | 3.00e-3 | 2 | 4.18e-3 | 4.84e-3 | 4.51e-3 | 4.18e-3 | 191 | -7.69e-4 | +2.21e-4 | -2.74e-4 | -1.24e-4 |
| 174 | 3.00e-3 | 1 | 4.57e-3 | 4.57e-3 | 4.57e-3 | 4.57e-3 | 224 | +4.02e-4 | +4.02e-4 | +4.02e-4 | -7.17e-5 |
| 175 | 3.00e-3 | 1 | 4.82e-3 | 4.82e-3 | 4.82e-3 | 4.82e-3 | 223 | +2.38e-4 | +2.38e-4 | +2.38e-4 | -4.07e-5 |
| 176 | 3.00e-3 | 2 | 4.16e-3 | 4.50e-3 | 4.33e-3 | 4.16e-3 | 172 | -4.60e-4 | -3.23e-4 | -3.91e-4 | -1.08e-4 |
| 177 | 3.00e-3 | 1 | 4.60e-3 | 4.60e-3 | 4.60e-3 | 4.60e-3 | 208 | +4.90e-4 | +4.90e-4 | +4.90e-4 | -4.81e-5 |
| 178 | 3.00e-3 | 1 | 4.30e-3 | 4.30e-3 | 4.30e-3 | 4.30e-3 | 207 | -3.27e-4 | -3.27e-4 | -3.27e-4 | -7.60e-5 |
| 179 | 3.00e-3 | 2 | 4.22e-3 | 4.65e-3 | 4.43e-3 | 4.22e-3 | 173 | -5.72e-4 | +3.85e-4 | -9.37e-5 | -8.42e-5 |
| 180 | 3.00e-3 | 1 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 214 | +5.41e-4 | +5.41e-4 | +5.41e-4 | -2.17e-5 |
| 181 | 3.00e-3 | 2 | 4.36e-3 | 4.84e-3 | 4.60e-3 | 4.36e-3 | 176 | -6.04e-4 | +1.02e-4 | -2.51e-4 | -6.88e-5 |
| 182 | 3.00e-3 | 1 | 4.75e-3 | 4.75e-3 | 4.75e-3 | 4.75e-3 | 223 | +3.91e-4 | +3.91e-4 | +3.91e-4 | -2.27e-5 |
| 183 | 3.00e-3 | 2 | 4.17e-3 | 4.70e-3 | 4.43e-3 | 4.17e-3 | 165 | -7.24e-4 | -5.76e-5 | -3.91e-4 | -9.60e-5 |
| 184 | 3.00e-3 | 1 | 4.61e-3 | 4.61e-3 | 4.61e-3 | 4.61e-3 | 205 | +4.91e-4 | +4.91e-4 | +4.91e-4 | -3.73e-5 |
| 185 | 3.00e-3 | 2 | 4.30e-3 | 4.70e-3 | 4.50e-3 | 4.30e-3 | 166 | -5.39e-4 | +9.40e-5 | -2.22e-4 | -7.56e-5 |
| 186 | 3.00e-3 | 1 | 4.88e-3 | 4.88e-3 | 4.88e-3 | 4.88e-3 | 221 | +5.80e-4 | +5.80e-4 | +5.80e-4 | -1.01e-5 |
| 187 | 3.00e-3 | 1 | 4.64e-3 | 4.64e-3 | 4.64e-3 | 4.64e-3 | 193 | -2.68e-4 | -2.68e-4 | -2.68e-4 | -3.59e-5 |
| 188 | 3.00e-3 | 2 | 4.41e-3 | 4.80e-3 | 4.61e-3 | 4.41e-3 | 170 | -5.02e-4 | +1.63e-4 | -1.70e-4 | -6.46e-5 |
| 189 | 3.00e-3 | 1 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 192 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -2.93e-5 |
| 190 | 3.00e-3 | 2 | 4.18e-3 | 4.38e-3 | 4.28e-3 | 4.18e-3 | 155 | -3.14e-4 | -2.99e-4 | -3.06e-4 | -8.19e-5 |
| 191 | 3.00e-3 | 1 | 4.64e-3 | 4.64e-3 | 4.64e-3 | 4.64e-3 | 194 | +5.35e-4 | +5.35e-4 | +5.35e-4 | -2.02e-5 |
| 192 | 3.00e-3 | 2 | 4.17e-3 | 4.55e-3 | 4.36e-3 | 4.17e-3 | 153 | -5.74e-4 | -1.03e-4 | -3.39e-4 | -8.31e-5 |
| 193 | 3.00e-3 | 1 | 4.67e-3 | 4.67e-3 | 4.67e-3 | 4.67e-3 | 200 | +5.61e-4 | +5.61e-4 | +5.61e-4 | -1.86e-5 |
| 194 | 3.00e-3 | 2 | 4.19e-3 | 4.77e-3 | 4.48e-3 | 4.19e-3 | 153 | -8.42e-4 | +1.04e-4 | -3.69e-4 | -8.99e-5 |
| 195 | 3.00e-3 | 1 | 4.29e-3 | 4.29e-3 | 4.29e-3 | 4.29e-3 | 193 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -6.88e-5 |
| 196 | 3.00e-3 | 2 | 4.07e-3 | 4.53e-3 | 4.30e-3 | 4.07e-3 | 145 | -7.36e-4 | +2.99e-4 | -2.19e-4 | -1.02e-4 |
| 197 | 3.00e-3 | 2 | 4.44e-3 | 4.72e-3 | 4.58e-3 | 4.44e-3 | 146 | -4.22e-4 | +7.51e-4 | +1.64e-4 | -5.76e-5 |
| 198 | 3.00e-3 | 2 | 4.31e-3 | 4.61e-3 | 4.46e-3 | 4.31e-3 | 152 | -4.41e-4 | +1.98e-4 | -1.22e-4 | -7.29e-5 |
| 199 | 3.00e-3 | 1 | 4.14e-3 | 4.14e-3 | 4.14e-3 | 4.14e-3 | 178 | -2.30e-4 | -2.30e-4 | -2.30e-4 | -8.86e-5 |

