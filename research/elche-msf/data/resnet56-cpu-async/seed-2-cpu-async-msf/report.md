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
| cpu-async | 0.015120 | 0.9321 | +0.0196 | 4925.7 | 459 | 222.7 | 100% | 100% | 100% | 7.3 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9321 | cpu-async | - | - |

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
| cpu-async | 2.4569 | 1.0859 | 0.6323 | 0.5456 | 0.5214 | 0.4986 | 0.4738 | 0.4597 | 0.4390 | 0.4320 | 0.1652 | 0.1267 | 0.1001 | 0.0929 | 0.0839 | 0.0277 | 0.0222 | 0.0182 | 0.0159 | 0.0151 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3553 | 1.1 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3257 | 1.3 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.3190 | 1.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 514 | 513 | 580 | 576 | 574 | 569 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 3442.8 | 1.5 | epoch-boundary(140) |
| cpu-async | gpu1 | 4924.2 | 1.5 | epoch-boundary(199) |
| cpu-async | gpu2 | 4924.2 | 1.5 | epoch-boundary(199) |
| cpu-async | gpu1 | 3687.1 | 0.9 | epoch-boundary(150) |
| cpu-async | gpu2 | 3687.5 | 0.7 | epoch-boundary(150) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 2.4s | 0.0s | 0.0s | 0.0s | 3.0s |
| resnet-graph | cpu-async | gpu2 | 3.8s | 0.0s | 0.0s | 0.0s | 4.3s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 304 | 0 | 459 | 222.7 | 7820/23106 | 459 | 222.7 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 321.6 | 6.5% |

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
| resnet-graph | cpu-async | 198 | 459 | 0 | 3.27e-3 | -3.17e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 459 | 9.39e-2 | 9.47e-2 | 3.01e-3 | 1.08e0 | 23.7 | -2.70e-4 | 1.97e-3 |
| resnet-graph | cpu-async | 1 | 459 | 9.46e-2 | 9.42e-2 | 2.83e-3 | 1.08e0 | 33.3 | -2.80e-4 | 1.96e-3 |
| resnet-graph | cpu-async | 2 | 459 | 9.47e-2 | 9.12e-2 | 2.96e-3 | 9.78e-1 | 42.9 | -2.82e-4 | 1.94e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9965 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9950 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9943 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 61 (0,1,2,3,4,5,6,7…136,142) | 0 (—) | — | 0,1,2,3,4,5,6,7…136,142 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 9 | 9 |
| resnet-graph | cpu-async | 0e0 | 5 | 4 | 4 |
| resnet-graph | cpu-async | 0e0 | 10 | 1 | 1 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 301 | +0.077 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 60 | +0.120 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 94 | -0.297 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 457 | -0.044 | 197 | +0.382 | +0.521 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 458 | 6.20e1–1.13e2 | 7.53e1 | 3.08e-3 | 1.32e-2 | 2.59e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 303 | 30–77604 | +2.175e-5 | 0.511 | +2.218e-5 | 0.514 | 100 | +1.752e-5 | 0.627 | 30–822 | +1.654e-3 | 0.438 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 291 | 785–77604 | +2.266e-5 | 0.661 | +2.284e-5 | 0.667 | 99 | +1.757e-5 | 0.621 | 81–822 | +1.689e-3 | 0.572 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 61 | 78209–116710 | +1.641e-5 | 0.247 | +1.649e-5 | 0.249 | 48 | +1.857e-5 | 0.366 | 460–885 | +6.791e-4 | 0.027 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 95 | 117301–156159 | -1.971e-5 | 0.321 | -2.011e-5 | 0.331 | 50 | -2.043e-5 | 0.327 | 200–656 | +1.830e-3 | 0.272 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.654e-3 | r0: +1.643e-3, r1: +1.662e-3, r2: +1.663e-3 | r0: 0.424, r1: 0.431, r2: 0.450 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.689e-3 | r0: +1.671e-3, r1: +1.694e-3, r2: +1.703e-3 | r0: 0.571, r1: 0.569, r2: 0.569 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +6.791e-4 | r0: +6.489e-4, r1: +6.869e-4, r2: +7.010e-4 | r0: 0.024, r1: 0.028, r2: 0.028 | 1.08× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.830e-3 | r0: +1.780e-3, r1: +1.868e-3, r2: +1.841e-3 | r0: 0.257, r1: 0.278, r2: 0.275 | 1.05× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `▇▆▇▇▇▇▇▇█████████████████▆▄▄▅▅▅▅▅▅▆▆▆▃▁▁▁▁▁▁▁▁▁▁▁▁` | `▁██▇▇▇▇▇█████████████████▇▇██████████▇▇███████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 2.23e-2 | 1.08e0 | 2.63e-1 | 3.36e-2 | 35 | -5.54e-2 | +6.59e-3 | -2.22e-2 | -1.67e-2 |
| 1 | 3.00e-1 | 7 | 2.65e-2 | 4.76e-2 | 3.19e-2 | 3.26e-2 | 34 | -1.37e-2 | +5.65e-3 | -1.19e-3 | -6.93e-3 |
| 2 | 3.00e-1 | 8 | 2.90e-2 | 4.87e-2 | 3.33e-2 | 3.21e-2 | 36 | -1.34e-2 | +5.22e-3 | -9.34e-4 | -3.11e-3 |
| 3 | 3.00e-1 | 7 | 3.38e-2 | 4.35e-2 | 3.69e-2 | 3.75e-2 | 30 | -5.61e-3 | +4.29e-3 | +4.50e-5 | -1.36e-3 |
| 4 | 3.00e-1 | 8 | 3.74e-2 | 5.52e-2 | 4.28e-2 | 4.51e-2 | 38 | -7.97e-3 | +5.53e-3 | -2.61e-4 | -5.73e-4 |
| 5 | 3.00e-1 | 6 | 5.05e-2 | 6.08e-2 | 5.43e-2 | 5.65e-2 | 44 | -5.15e-3 | +3.77e-3 | +2.09e-4 | -1.83e-4 |
| 6 | 3.00e-1 | 6 | 5.87e-2 | 7.49e-2 | 6.21e-2 | 5.96e-2 | 36 | -5.40e-3 | +3.48e-3 | -3.35e-4 | -2.52e-4 |
| 7 | 3.00e-1 | 7 | 6.51e-2 | 9.04e-2 | 7.06e-2 | 6.62e-2 | 33 | -8.48e-3 | +4.26e-3 | -6.96e-4 | -4.61e-4 |
| 8 | 3.00e-1 | 6 | 7.05e-2 | 9.73e-2 | 7.76e-2 | 7.62e-2 | 38 | -8.20e-3 | +4.70e-3 | -4.16e-4 | -3.97e-4 |
| 9 | 3.00e-1 | 10 | 7.46e-2 | 1.08e-1 | 8.23e-2 | 8.52e-2 | 33 | -1.00e-2 | +4.07e-3 | -4.59e-4 | -2.52e-4 |
| 10 | 3.00e-1 | 5 | 8.59e-2 | 1.15e-1 | 9.22e-2 | 8.64e-2 | 32 | -8.78e-3 | +4.08e-3 | -9.05e-4 | -5.11e-4 |
| 11 | 3.00e-1 | 8 | 8.47e-2 | 1.21e-1 | 9.42e-2 | 9.81e-2 | 43 | -7.90e-3 | +4.49e-3 | -2.90e-4 | -2.64e-4 |
| 12 | 3.00e-1 | 6 | 9.85e-2 | 1.33e-1 | 1.07e-1 | 1.04e-1 | 41 | -5.56e-3 | +3.65e-3 | -3.84e-4 | -2.84e-4 |
| 13 | 3.00e-1 | 9 | 9.35e-2 | 1.39e-1 | 1.05e-1 | 9.76e-2 | 37 | -7.70e-3 | +3.09e-3 | -7.51e-4 | -4.94e-4 |
| 14 | 3.00e-1 | 4 | 9.71e-2 | 1.34e-1 | 1.07e-1 | 9.71e-2 | 37 | -9.50e-3 | +4.24e-3 | -1.50e-3 | -8.51e-4 |
| 15 | 3.00e-1 | 8 | 9.89e-2 | 1.39e-1 | 1.08e-1 | 9.89e-2 | 33 | -6.03e-3 | +4.58e-3 | -5.48e-4 | -6.91e-4 |
| 16 | 3.00e-1 | 5 | 1.04e-1 | 1.35e-1 | 1.12e-1 | 1.04e-1 | 35 | -7.96e-3 | +4.26e-3 | -7.81e-4 | -7.55e-4 |
| 17 | 3.00e-1 | 7 | 9.77e-2 | 1.52e-1 | 1.10e-1 | 1.00e-1 | 31 | -8.76e-3 | +3.85e-3 | -1.20e-3 | -9.01e-4 |
| 18 | 3.00e-1 | 9 | 9.77e-2 | 1.42e-1 | 1.07e-1 | 1.07e-1 | 34 | -8.77e-3 | +4.64e-3 | -4.63e-4 | -4.81e-4 |
| 19 | 3.00e-1 | 6 | 1.01e-1 | 1.50e-1 | 1.12e-1 | 1.05e-1 | 32 | -9.53e-3 | +4.23e-3 | -1.14e-3 | -7.20e-4 |
| 20 | 3.00e-1 | 8 | 9.99e-2 | 1.55e-1 | 1.12e-1 | 1.00e-1 | 31 | -1.20e-2 | +4.62e-3 | -1.27e-3 | -9.72e-4 |
| 21 | 3.00e-1 | 6 | 1.05e-1 | 1.44e-1 | 1.14e-1 | 1.15e-1 | 41 | -1.12e-2 | +5.12e-3 | -6.38e-4 | -7.38e-4 |
| 22 | 3.00e-1 | 9 | 1.03e-1 | 1.61e-1 | 1.18e-1 | 1.09e-1 | 36 | -5.54e-3 | +3.86e-3 | -6.35e-4 | -6.06e-4 |
| 23 | 3.00e-1 | 4 | 1.13e-1 | 1.47e-1 | 1.25e-1 | 1.22e-1 | 44 | -7.86e-3 | +4.08e-3 | -5.53e-4 | -5.91e-4 |
| 24 | 3.00e-1 | 8 | 1.13e-1 | 1.56e-1 | 1.25e-1 | 1.13e-1 | 38 | -4.18e-3 | +3.04e-3 | -5.73e-4 | -5.98e-4 |
| 25 | 3.00e-1 | 5 | 1.08e-1 | 1.56e-1 | 1.22e-1 | 1.08e-1 | 32 | -4.47e-3 | +4.17e-3 | -1.09e-3 | -8.35e-4 |
| 26 | 3.00e-1 | 7 | 1.07e-1 | 1.52e-1 | 1.16e-1 | 1.11e-1 | 37 | -8.77e-3 | +4.87e-3 | -7.45e-4 | -7.24e-4 |
| 27 | 3.00e-1 | 6 | 1.16e-1 | 1.48e-1 | 1.26e-1 | 1.28e-1 | 48 | -5.46e-3 | +4.03e-3 | -4.87e-5 | -3.85e-4 |
| 28 | 3.00e-1 | 6 | 1.07e-1 | 1.63e-1 | 1.25e-1 | 1.07e-1 | 33 | -5.36e-3 | +2.73e-3 | -1.23e-3 | -8.07e-4 |
| 29 | 3.00e-1 | 8 | 1.13e-1 | 1.51e-1 | 1.21e-1 | 1.25e-1 | 43 | -8.39e-3 | +4.86e-3 | -2.04e-4 | -3.75e-4 |
| 30 | 3.00e-1 | 5 | 1.08e-1 | 1.72e-1 | 1.28e-1 | 1.21e-1 | 44 | -8.03e-3 | +3.38e-3 | -1.23e-3 | -6.71e-4 |
| 31 | 3.00e-1 | 6 | 1.25e-1 | 1.63e-1 | 1.35e-1 | 1.25e-1 | 45 | -4.78e-3 | +3.46e-3 | -4.33e-4 | -5.82e-4 |
| 32 | 3.00e-1 | 4 | 1.27e-1 | 1.65e-1 | 1.37e-1 | 1.27e-1 | 49 | -5.62e-3 | +3.20e-3 | -6.79e-4 | -6.31e-4 |
| 33 | 3.00e-1 | 2 | 1.28e-1 | 2.20e-1 | 1.74e-1 | 2.20e-1 | 247 | +2.21e-4 | +2.17e-3 | +1.20e-3 | -2.74e-4 |
| 34 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 244 | +1.73e-4 | +1.73e-4 | +1.73e-4 | -2.29e-4 |
| 35 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 251 | +4.24e-5 | +4.24e-5 | +4.24e-5 | -2.02e-4 |
| 36 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 275 | +5.49e-5 | +5.49e-5 | +5.49e-5 | -1.77e-4 |
| 37 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 269 | -4.37e-5 | -4.37e-5 | -4.37e-5 | -1.63e-4 |
| 38 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 265 | -6.58e-5 | -6.58e-5 | -6.58e-5 | -1.54e-4 |
| 39 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 295 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -1.21e-4 |
| 40 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 298 | +5.75e-5 | +5.75e-5 | +5.75e-5 | -1.03e-4 |
| 41 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 271 | -8.51e-5 | -8.51e-5 | -8.51e-5 | -1.02e-4 |
| 42 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 235 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -1.26e-4 |
| 43 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 238 | -8.14e-6 | -8.14e-6 | -8.14e-6 | -1.14e-4 |
| 44 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 233 | -1.21e-5 | -1.21e-5 | -1.21e-5 | -1.04e-4 |
| 45 | 3.00e-1 | 2 | 2.14e-1 | 2.27e-1 | 2.21e-1 | 2.14e-1 | 215 | -2.71e-4 | +1.34e-4 | -6.85e-5 | -9.90e-5 |
| 46 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 264 | +3.41e-4 | +3.41e-4 | +3.41e-4 | -5.49e-5 |
| 47 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 254 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -6.16e-5 |
| 48 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 255 | +1.95e-5 | +1.95e-5 | +1.95e-5 | -5.35e-5 |
| 49 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 257 | +1.94e-5 | +1.94e-5 | +1.94e-5 | -4.62e-5 |
| 50 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 280 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -2.31e-5 |
| 51 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 249 | -2.46e-4 | -2.46e-4 | -2.46e-4 | -4.54e-5 |
| 52 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 260 | +9.80e-5 | +9.80e-5 | +9.80e-5 | -3.10e-5 |
| 53 | 3.00e-1 | 2 | 2.14e-1 | 2.28e-1 | 2.21e-1 | 2.14e-1 | 199 | -3.08e-4 | -9.48e-5 | -2.01e-4 | -6.45e-5 |
| 54 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 247 | +2.80e-4 | +2.80e-4 | +2.80e-4 | -3.00e-5 |
| 55 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 250 | +5.40e-5 | +5.40e-5 | +5.40e-5 | -2.16e-5 |
| 56 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 227 | -1.73e-4 | -1.73e-4 | -1.73e-4 | -3.67e-5 |
| 57 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 251 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -1.97e-5 |
| 58 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 249 | +1.32e-5 | +1.32e-5 | +1.32e-5 | -1.64e-5 |
| 59 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 199 | -4.13e-4 | -4.13e-4 | -4.13e-4 | -5.60e-5 |
| 60 | 3.00e-1 | 2 | 2.21e-1 | 2.32e-1 | 2.26e-1 | 2.21e-1 | 226 | -2.04e-4 | +2.55e-4 | +2.54e-5 | -4.29e-5 |
| 61 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 253 | +2.31e-4 | +2.31e-4 | +2.31e-4 | -1.55e-5 |
| 62 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 256 | -1.09e-5 | -1.09e-5 | -1.09e-5 | -1.50e-5 |
| 63 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 268 | +2.53e-5 | +2.53e-5 | +2.53e-5 | -1.10e-5 |
| 64 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 265 | +1.83e-5 | +1.83e-5 | +1.83e-5 | -8.06e-6 |
| 65 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 268 | +1.38e-5 | +1.38e-5 | +1.38e-5 | -5.87e-6 |
| 66 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 288 | +7.88e-5 | +7.88e-5 | +7.88e-5 | +2.59e-6 |
| 67 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 264 | -7.33e-5 | -7.33e-5 | -7.33e-5 | -5.00e-6 |
| 68 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 242 | -1.55e-4 | -1.55e-4 | -1.55e-4 | -2.00e-5 |
| 69 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 243 | +9.88e-6 | +9.88e-6 | +9.88e-6 | -1.70e-5 |
| 70 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 245 | -2.82e-5 | -2.82e-5 | -2.82e-5 | -1.82e-5 |
| 71 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 252 | +8.78e-5 | +8.78e-5 | +8.78e-5 | -7.57e-6 |
| 72 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 273 | +1.00e-4 | +1.00e-4 | +1.00e-4 | +3.21e-6 |
| 73 | 3.00e-1 | 2 | 2.24e-1 | 2.32e-1 | 2.28e-1 | 2.24e-1 | 213 | -1.65e-4 | -1.42e-4 | -1.53e-4 | -2.67e-5 |
| 74 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 239 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -1.35e-5 |
| 75 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 237 | -2.83e-5 | -2.83e-5 | -2.83e-5 | -1.50e-5 |
| 76 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 250 | +1.92e-5 | +1.92e-5 | +1.92e-5 | -1.16e-5 |
| 77 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 262 | +1.74e-4 | +1.74e-4 | +1.74e-4 | +7.02e-6 |
| 78 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 270 | +4.31e-5 | +4.31e-5 | +4.31e-5 | +1.06e-5 |
| 79 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 252 | -1.40e-4 | -1.40e-4 | -1.40e-4 | -4.39e-6 |
| 80 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 257 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +7.13e-6 |
| 81 | 3.00e-1 | 2 | 2.19e-1 | 2.37e-1 | 2.28e-1 | 2.19e-1 | 200 | -3.97e-4 | -5.37e-5 | -2.25e-4 | -3.87e-5 |
| 82 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 233 | +1.72e-4 | +1.72e-4 | +1.72e-4 | -1.76e-5 |
| 83 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 227 | +3.15e-5 | +3.15e-5 | +3.15e-5 | -1.27e-5 |
| 84 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 237 | -5.83e-5 | -5.83e-5 | -5.83e-5 | -1.73e-5 |
| 85 | 3.00e-1 | 2 | 2.18e-1 | 2.37e-1 | 2.28e-1 | 2.18e-1 | 200 | -4.07e-4 | +1.85e-4 | -1.11e-4 | -3.80e-5 |
| 86 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 226 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -2.41e-5 |
| 87 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 232 | +1.53e-4 | +1.53e-4 | +1.53e-4 | -6.43e-6 |
| 88 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 241 | +6.80e-5 | +6.80e-5 | +6.80e-5 | +1.02e-6 |
| 89 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 254 | +4.35e-5 | +4.35e-5 | +4.35e-5 | +5.27e-6 |
| 90 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 247 | +3.44e-5 | +3.44e-5 | +3.44e-5 | +8.18e-6 |
| 91 | 3.00e-1 | 2 | 2.18e-1 | 2.27e-1 | 2.22e-1 | 2.18e-1 | 195 | -2.53e-4 | -2.15e-4 | -2.34e-4 | -3.77e-5 |
| 92 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 236 | +2.58e-4 | +2.58e-4 | +2.58e-4 | -8.08e-6 |
| 93 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 204 | -2.44e-4 | -2.44e-4 | -2.44e-4 | -3.16e-5 |
| 94 | 3.00e-1 | 2 | 2.12e-1 | 2.28e-1 | 2.20e-1 | 2.12e-1 | 176 | -4.13e-4 | +1.57e-4 | -1.28e-4 | -5.28e-5 |
| 95 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 216 | +3.39e-4 | +3.39e-4 | +3.39e-4 | -1.36e-5 |
| 96 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 212 | -7.28e-5 | -7.28e-5 | -7.28e-5 | -1.96e-5 |
| 97 | 3.00e-1 | 2 | 2.12e-1 | 2.28e-1 | 2.20e-1 | 2.12e-1 | 176 | -4.13e-4 | +7.28e-5 | -1.70e-4 | -5.06e-5 |
| 98 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 214 | +3.65e-4 | +3.65e-4 | +3.65e-4 | -9.04e-6 |
| 99 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 215 | -6.96e-5 | -6.96e-5 | -6.96e-5 | -1.51e-5 |
| 100 | 3.00e-2 | 2 | 1.14e-1 | 2.27e-1 | 1.70e-1 | 1.14e-1 | 179 | -3.87e-3 | +2.05e-5 | -1.92e-3 | -3.97e-4 |
| 101 | 3.00e-2 | 1 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 5.69e-2 | 198 | -3.49e-3 | -3.49e-3 | -3.49e-3 | -7.06e-4 |
| 102 | 3.00e-2 | 2 | 2.54e-2 | 3.61e-2 | 3.07e-2 | 2.54e-2 | 176 | -2.00e-3 | -1.95e-3 | -1.97e-3 | -9.47e-4 |
| 103 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 228 | -6.19e-5 | -6.19e-5 | -6.19e-5 | -8.59e-4 |
| 104 | 3.00e-2 | 2 | 2.42e-2 | 2.47e-2 | 2.44e-2 | 2.42e-2 | 181 | -1.23e-4 | -5.92e-5 | -9.11e-5 | -7.13e-4 |
| 105 | 3.00e-2 | 1 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 227 | +4.41e-4 | +4.41e-4 | +4.41e-4 | -5.98e-4 |
| 106 | 3.00e-2 | 1 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 221 | +1.46e-4 | +1.46e-4 | +1.46e-4 | -5.23e-4 |
| 107 | 3.00e-2 | 1 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 201 | -7.83e-5 | -7.83e-5 | -7.83e-5 | -4.79e-4 |
| 108 | 3.00e-2 | 2 | 2.74e-2 | 2.86e-2 | 2.80e-2 | 2.74e-2 | 173 | -2.46e-4 | +2.41e-4 | -2.13e-6 | -3.91e-4 |
| 109 | 3.00e-2 | 1 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 193 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -3.40e-4 |
| 110 | 3.00e-2 | 2 | 2.82e-2 | 2.97e-2 | 2.90e-2 | 2.82e-2 | 158 | -3.28e-4 | +2.76e-4 | -2.59e-5 | -2.83e-4 |
| 111 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 192 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -2.34e-4 |
| 112 | 3.00e-2 | 1 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 2.84e-2 | 163 | -2.02e-4 | -2.02e-4 | -2.02e-4 | -2.31e-4 |
| 113 | 3.00e-2 | 2 | 3.52e-2 | 3.54e-2 | 3.53e-2 | 3.54e-2 | 236 | +1.94e-5 | +6.87e-4 | +3.53e-4 | -1.23e-4 |
| 115 | 3.00e-2 | 2 | 3.74e-2 | 4.06e-2 | 3.90e-2 | 3.74e-2 | 236 | -3.44e-4 | +4.21e-4 | +3.83e-5 | -9.64e-5 |
| 117 | 3.00e-2 | 2 | 3.92e-2 | 4.31e-2 | 4.11e-2 | 3.92e-2 | 240 | -3.94e-4 | +4.24e-4 | +1.49e-5 | -7.93e-5 |
| 118 | 3.00e-2 | 1 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 4.02e-2 | 256 | +9.92e-5 | +9.92e-5 | +9.92e-5 | -6.15e-5 |
| 119 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 261 | +1.97e-4 | +1.97e-4 | +1.97e-4 | -3.57e-5 |
| 120 | 3.00e-2 | 1 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 264 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -2.08e-5 |
| 121 | 3.00e-2 | 1 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 4.39e-2 | 269 | +2.95e-5 | +2.95e-5 | +2.95e-5 | -1.57e-5 |
| 122 | 3.00e-2 | 1 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 4.54e-2 | 272 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -1.90e-6 |
| 123 | 3.00e-2 | 1 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 4.63e-2 | 277 | +6.42e-5 | +6.42e-5 | +6.42e-5 | +4.72e-6 |
| 124 | 3.00e-2 | 1 | 4.47e-2 | 4.47e-2 | 4.47e-2 | 4.47e-2 | 228 | -1.49e-4 | -1.49e-4 | -1.49e-4 | -1.06e-5 |
| 125 | 3.00e-2 | 1 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 4.71e-2 | 248 | +2.07e-4 | +2.07e-4 | +2.07e-4 | +1.12e-5 |
| 126 | 3.00e-2 | 1 | 4.88e-2 | 4.88e-2 | 4.88e-2 | 4.88e-2 | 257 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +2.43e-5 |
| 127 | 3.00e-2 | 1 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 279 | +1.68e-4 | +1.68e-4 | +1.68e-4 | +3.87e-5 |
| 128 | 3.00e-2 | 1 | 5.17e-2 | 5.17e-2 | 5.17e-2 | 5.17e-2 | 273 | +3.67e-5 | +3.67e-5 | +3.67e-5 | +3.85e-5 |
| 129 | 3.00e-2 | 2 | 4.84e-2 | 5.16e-2 | 5.00e-2 | 4.84e-2 | 199 | -3.20e-4 | -1.01e-5 | -1.65e-4 | -1.72e-6 |
| 130 | 3.00e-2 | 1 | 4.96e-2 | 4.96e-2 | 4.96e-2 | 4.96e-2 | 234 | +1.10e-4 | +1.10e-4 | +1.10e-4 | +9.49e-6 |
| 131 | 3.00e-2 | 1 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 5.26e-2 | 250 | +2.32e-4 | +2.32e-4 | +2.32e-4 | +3.17e-5 |
| 132 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 220 | -1.68e-4 | -1.68e-4 | -1.68e-4 | +1.17e-5 |
| 133 | 3.00e-2 | 1 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 4.99e-2 | 214 | -7.03e-5 | -7.03e-5 | -7.03e-5 | +3.48e-6 |
| 134 | 3.00e-2 | 2 | 5.25e-2 | 5.25e-2 | 5.25e-2 | 5.25e-2 | 208 | -7.78e-7 | +2.26e-4 | +1.12e-4 | +2.30e-5 |
| 135 | 3.00e-2 | 1 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 234 | +1.33e-4 | +1.33e-4 | +1.33e-4 | +3.41e-5 |
| 136 | 3.00e-2 | 1 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 259 | +2.66e-4 | +2.66e-4 | +2.66e-4 | +5.73e-5 |
| 137 | 3.00e-2 | 1 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 5.68e-2 | 238 | -8.62e-5 | -8.62e-5 | -8.62e-5 | +4.30e-5 |
| 138 | 3.00e-2 | 1 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 249 | +1.48e-4 | +1.48e-4 | +1.48e-4 | +5.35e-5 |
| 139 | 3.00e-2 | 1 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 232 | -4.06e-5 | -4.06e-5 | -4.06e-5 | +4.41e-5 |
| 140 | 3.00e-2 | 2 | 5.57e-2 | 6.51e-2 | 6.04e-2 | 5.57e-2 | 192 | -8.08e-4 | +4.05e-4 | -2.01e-4 | -8.64e-6 |
| 141 | 3.00e-2 | 1 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 210 | +1.25e-4 | +1.25e-4 | +1.25e-4 | +4.74e-6 |
| 142 | 3.00e-2 | 1 | 6.24e-2 | 6.24e-2 | 6.24e-2 | 6.24e-2 | 240 | +3.63e-4 | +3.63e-4 | +3.63e-4 | +4.06e-5 |
| 143 | 3.00e-2 | 1 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 248 | -6.22e-5 | -6.22e-5 | -6.22e-5 | +3.03e-5 |
| 144 | 3.00e-2 | 2 | 5.93e-2 | 6.18e-2 | 6.06e-2 | 5.93e-2 | 191 | -2.17e-4 | +2.25e-5 | -9.74e-5 | +4.83e-6 |
| 145 | 3.00e-2 | 1 | 6.06e-2 | 6.06e-2 | 6.06e-2 | 6.06e-2 | 218 | +9.58e-5 | +9.58e-5 | +9.58e-5 | +1.39e-5 |
| 146 | 3.00e-2 | 1 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 200 | -1.12e-4 | -1.12e-4 | -1.12e-4 | +1.35e-6 |
| 147 | 3.00e-2 | 2 | 5.95e-2 | 6.15e-2 | 6.05e-2 | 5.95e-2 | 187 | -1.79e-4 | +1.76e-4 | -1.63e-6 | -9.93e-7 |
| 148 | 3.00e-2 | 1 | 6.48e-2 | 6.48e-2 | 6.48e-2 | 6.48e-2 | 227 | +3.81e-4 | +3.81e-4 | +3.81e-4 | +3.72e-5 |
| 149 | 3.00e-2 | 1 | 6.47e-2 | 6.47e-2 | 6.47e-2 | 6.47e-2 | 224 | -4.98e-6 | -4.98e-6 | -4.98e-6 | +3.30e-5 |
| 150 | 3.00e-3 | 2 | 3.16e-2 | 6.35e-2 | 4.76e-2 | 3.16e-2 | 187 | -3.74e-3 | -9.24e-5 | -1.92e-3 | -3.55e-4 |
| 151 | 3.00e-3 | 1 | 1.66e-2 | 1.66e-2 | 1.66e-2 | 1.66e-2 | 204 | -3.15e-3 | -3.15e-3 | -3.15e-3 | -6.35e-4 |
| 152 | 3.00e-3 | 1 | 9.10e-3 | 9.10e-3 | 9.10e-3 | 9.10e-3 | 201 | -2.99e-3 | -2.99e-3 | -2.99e-3 | -8.71e-4 |
| 153 | 3.00e-3 | 2 | 4.92e-3 | 6.16e-3 | 5.54e-3 | 4.92e-3 | 180 | -1.80e-3 | -1.25e-3 | -1.52e-3 | -9.92e-4 |
| 154 | 3.00e-3 | 1 | 4.82e-3 | 4.82e-3 | 4.82e-3 | 4.82e-3 | 221 | -9.13e-5 | -9.13e-5 | -9.13e-5 | -9.02e-4 |
| 155 | 3.00e-3 | 1 | 5.04e-3 | 5.04e-3 | 5.04e-3 | 5.04e-3 | 247 | +1.82e-4 | +1.82e-4 | +1.82e-4 | -7.94e-4 |
| 156 | 3.00e-3 | 2 | 4.41e-3 | 5.08e-3 | 4.74e-3 | 4.41e-3 | 164 | -8.58e-4 | +2.78e-5 | -4.15e-4 | -7.26e-4 |
| 157 | 3.00e-3 | 1 | 4.72e-3 | 4.72e-3 | 4.72e-3 | 4.72e-3 | 220 | +3.03e-4 | +3.03e-4 | +3.03e-4 | -6.23e-4 |
| 158 | 3.00e-3 | 2 | 4.35e-3 | 4.45e-3 | 4.40e-3 | 4.35e-3 | 166 | -2.85e-4 | -1.35e-4 | -2.10e-4 | -5.44e-4 |
| 159 | 3.00e-3 | 1 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 204 | +3.19e-4 | +3.19e-4 | +3.19e-4 | -4.58e-4 |
| 160 | 3.00e-3 | 1 | 4.69e-3 | 4.69e-3 | 4.69e-3 | 4.69e-3 | 222 | +4.27e-5 | +4.27e-5 | +4.27e-5 | -4.08e-4 |
| 161 | 3.00e-3 | 2 | 4.33e-3 | 4.54e-3 | 4.44e-3 | 4.33e-3 | 162 | -2.91e-4 | -1.48e-4 | -2.20e-4 | -3.73e-4 |
| 162 | 3.00e-3 | 1 | 4.52e-3 | 4.52e-3 | 4.52e-3 | 4.52e-3 | 202 | +2.12e-4 | +2.12e-4 | +2.12e-4 | -3.14e-4 |
| 163 | 3.00e-3 | 2 | 4.39e-3 | 4.59e-3 | 4.49e-3 | 4.39e-3 | 155 | -2.99e-4 | +7.71e-5 | -1.11e-4 | -2.77e-4 |
| 164 | 3.00e-3 | 2 | 4.40e-3 | 4.66e-3 | 4.53e-3 | 4.40e-3 | 151 | -3.70e-4 | +3.11e-4 | -2.93e-5 | -2.34e-4 |
| 165 | 3.00e-3 | 1 | 4.51e-3 | 4.51e-3 | 4.51e-3 | 4.51e-3 | 192 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -1.98e-4 |
| 166 | 3.00e-3 | 2 | 4.12e-3 | 4.54e-3 | 4.33e-3 | 4.12e-3 | 156 | -6.27e-4 | +3.33e-5 | -2.97e-4 | -2.20e-4 |
| 167 | 3.00e-3 | 1 | 4.60e-3 | 4.60e-3 | 4.60e-3 | 4.60e-3 | 191 | +5.81e-4 | +5.81e-4 | +5.81e-4 | -1.40e-4 |
| 168 | 3.00e-3 | 2 | 4.17e-3 | 5.16e-3 | 4.67e-3 | 4.17e-3 | 140 | -1.51e-3 | +5.16e-4 | -4.96e-4 | -2.18e-4 |
| 169 | 3.00e-3 | 1 | 4.43e-3 | 4.43e-3 | 4.43e-3 | 4.43e-3 | 191 | +3.16e-4 | +3.16e-4 | +3.16e-4 | -1.64e-4 |
| 170 | 3.00e-3 | 2 | 4.29e-3 | 4.53e-3 | 4.41e-3 | 4.29e-3 | 149 | -3.65e-4 | +1.11e-4 | -1.27e-4 | -1.60e-4 |
| 171 | 3.00e-3 | 2 | 4.39e-3 | 4.39e-3 | 4.39e-3 | 4.39e-3 | 158 | +1.28e-6 | +1.31e-4 | +6.60e-5 | -1.17e-4 |
| 172 | 3.00e-3 | 2 | 4.17e-3 | 4.30e-3 | 4.23e-3 | 4.17e-3 | 141 | -2.12e-4 | -1.29e-4 | -1.71e-4 | -1.28e-4 |
| 173 | 3.00e-3 | 1 | 4.39e-3 | 4.39e-3 | 4.39e-3 | 4.39e-3 | 170 | +3.08e-4 | +3.08e-4 | +3.08e-4 | -8.44e-5 |
| 174 | 3.00e-3 | 2 | 4.46e-3 | 4.49e-3 | 4.47e-3 | 4.46e-3 | 153 | -4.66e-5 | +1.21e-4 | +3.73e-5 | -6.21e-5 |
| 175 | 3.00e-3 | 2 | 4.46e-3 | 4.56e-3 | 4.51e-3 | 4.46e-3 | 147 | -1.40e-4 | +1.23e-4 | -8.09e-6 | -5.32e-5 |
| 176 | 3.00e-3 | 1 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 167 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -2.38e-5 |
| 177 | 3.00e-3 | 2 | 4.03e-3 | 4.49e-3 | 4.26e-3 | 4.03e-3 | 132 | -8.22e-4 | -1.86e-4 | -5.04e-4 | -1.18e-4 |
| 178 | 3.00e-3 | 2 | 4.26e-3 | 4.60e-3 | 4.43e-3 | 4.26e-3 | 133 | -5.80e-4 | +7.82e-4 | +1.01e-4 | -8.34e-5 |
| 179 | 3.00e-3 | 2 | 4.03e-3 | 4.18e-3 | 4.10e-3 | 4.03e-3 | 125 | -3.07e-4 | -1.18e-4 | -2.12e-4 | -1.09e-4 |
| 180 | 3.00e-3 | 2 | 3.98e-3 | 4.48e-3 | 4.23e-3 | 3.98e-3 | 131 | -8.97e-4 | +6.27e-4 | -1.35e-4 | -1.21e-4 |
| 181 | 3.00e-3 | 2 | 4.06e-3 | 4.73e-3 | 4.40e-3 | 4.06e-3 | 127 | -1.20e-3 | +8.63e-4 | -1.70e-4 | -1.41e-4 |
| 182 | 3.00e-3 | 2 | 4.19e-3 | 4.21e-3 | 4.20e-3 | 4.21e-3 | 119 | +3.83e-5 | +2.06e-4 | +1.22e-4 | -9.17e-5 |
| 183 | 3.00e-3 | 2 | 3.88e-3 | 4.23e-3 | 4.05e-3 | 3.88e-3 | 119 | -7.27e-4 | +2.67e-5 | -3.50e-4 | -1.45e-4 |
| 184 | 3.00e-3 | 2 | 3.79e-3 | 4.08e-3 | 3.94e-3 | 3.79e-3 | 116 | -6.17e-4 | +3.04e-4 | -1.56e-4 | -1.51e-4 |
| 185 | 3.00e-3 | 2 | 4.02e-3 | 4.25e-3 | 4.14e-3 | 4.02e-3 | 124 | -4.36e-4 | +7.37e-4 | +1.50e-4 | -1.00e-4 |
| 186 | 3.00e-3 | 2 | 3.60e-3 | 4.36e-3 | 3.98e-3 | 3.60e-3 | 109 | -1.74e-3 | +5.29e-4 | -6.07e-4 | -2.08e-4 |
| 187 | 3.00e-3 | 2 | 3.85e-3 | 4.35e-3 | 4.10e-3 | 3.85e-3 | 114 | -1.07e-3 | +1.31e-3 | +1.20e-4 | -1.57e-4 |
| 188 | 3.00e-3 | 2 | 3.67e-3 | 4.46e-3 | 4.07e-3 | 3.67e-3 | 105 | -1.84e-3 | +8.85e-4 | -4.76e-4 | -2.32e-4 |
| 189 | 3.00e-3 | 3 | 3.67e-3 | 4.44e-3 | 3.98e-3 | 3.67e-3 | 110 | -1.51e-3 | +1.26e-3 | -2.22e-4 | -2.44e-4 |
| 190 | 3.00e-3 | 2 | 3.67e-3 | 4.14e-3 | 3.91e-3 | 3.67e-3 | 102 | -1.18e-3 | +9.04e-4 | -1.37e-4 | -2.34e-4 |
| 191 | 3.00e-3 | 3 | 3.73e-3 | 4.21e-3 | 3.89e-3 | 3.73e-3 | 95 | -1.20e-3 | +9.11e-4 | -1.10e-4 | -2.09e-4 |
| 192 | 3.00e-3 | 2 | 3.72e-3 | 4.09e-3 | 3.90e-3 | 3.72e-3 | 97 | -9.55e-4 | +6.99e-4 | -1.28e-4 | -2.02e-4 |
| 193 | 3.00e-3 | 3 | 3.56e-3 | 3.99e-3 | 3.71e-3 | 3.56e-3 | 99 | -1.08e-3 | +4.99e-4 | -2.27e-4 | -2.14e-4 |
| 194 | 3.00e-3 | 2 | 3.62e-3 | 4.50e-3 | 4.06e-3 | 3.62e-3 | 99 | -2.18e-3 | +1.41e-3 | -3.88e-4 | -2.65e-4 |
| 195 | 3.00e-3 | 3 | 3.30e-3 | 4.18e-3 | 3.68e-3 | 3.30e-3 | 90 | -1.79e-3 | +1.03e-3 | -5.37e-4 | -3.56e-4 |
| 196 | 3.00e-3 | 3 | 3.39e-3 | 3.80e-3 | 3.55e-3 | 3.39e-3 | 84 | -1.08e-3 | +1.09e-3 | -8.86e-5 | -2.96e-4 |
| 197 | 3.00e-3 | 2 | 3.33e-3 | 3.89e-3 | 3.61e-3 | 3.33e-3 | 92 | -1.68e-3 | +1.06e-3 | -3.13e-4 | -3.13e-4 |
| 198 | 3.00e-3 | 3 | 3.24e-3 | 3.90e-3 | 3.47e-3 | 3.24e-3 | 76 | -2.07e-3 | +1.27e-3 | -3.10e-4 | -3.24e-4 |
| 199 | 3.00e-3 | 4 | 3.07e-3 | 4.07e-3 | 3.42e-3 | 3.27e-3 | 72 | -2.85e-3 | +1.77e-3 | -3.00e-4 | -3.17e-4 |

