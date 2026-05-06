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
| nccl-async | 0.065369 | 0.9174 | +0.0049 | 1930.0 | 229 | 38.9 | 100% | 100% | 5.5 |

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
| nccl-async | 2.0115 | 0.8599 | 0.6466 | 0.5829 | 0.5456 | 0.5157 | 0.4841 | 0.4933 | 0.4822 | 0.4745 | 0.2184 | 0.1778 | 0.1593 | 0.1476 | 0.1436 | 0.0841 | 0.0748 | 0.0712 | 0.0672 | 0.0654 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3998 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3059 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2943 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 390 | 387 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1929.0 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu2 | 1929.0 | 1.0 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.0s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 2.3s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 47 | 0 | 229 | 38.9 | 8895/10538 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 164.2 | 8.5% |

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
| resnet-graph | nccl-async | 187 | 229 | 0 | 6.49e-3 | +1.64e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 229 | 1.04e-1 | 9.24e-2 | 0.00e0 | 5.25e-1 | 13.1 | -2.59e-4 | 2.24e-3 |
| resnet-graph | nccl-async | 1 | 229 | 1.07e-1 | 9.44e-2 | 0.00e0 | 5.11e-1 | 58.5 | -2.61e-4 | 2.29e-3 |
| resnet-graph | nccl-async | 2 | 229 | 1.07e-1 | 9.96e-2 | 0.00e0 | 7.04e-1 | 28.4 | -2.91e-4 | 2.50e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9992 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9882 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9874 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 36 (0,3,4,5,6,14,17,24…147,148) | 0 (—) | — | 0,3,4,5,6,14,17,24…147,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 16 | 16 |
| resnet-graph | nccl-async | 0e0 | 5 | 7 | 7 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 0 | 0 |
| resnet-graph | nccl-async | 1e-4 | 5 | 0 | 0 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 111 | +0.403 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 54 | +0.194 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 59 | +0.031 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 226 | -0.003 | 186 | +0.254 | +0.754 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 227 | 3.58e1–8.04e1 | 6.73e1 | 5.16e-3 | 1.23e-2 | 1.35e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 113 | 69–77556 | +7.815e-6 | 0.202 | +8.319e-6 | 0.226 | 92 | +4.011e-6 | 0.174 | 37–987 | +1.214e-3 | 0.474 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 101 | 812–77556 | +3.835e-6 | 0.165 | +3.894e-6 | 0.161 | 91 | +3.375e-6 | 0.147 | 62–987 | +1.144e-3 | 0.382 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 55 | 78322–116490 | +1.182e-5 | 0.101 | +1.183e-5 | 0.102 | 45 | +9.153e-6 | 0.055 | 554–867 | -4.675e-4 | 0.006 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 60 | 117312–155871 | -7.520e-6 | 0.041 | -7.437e-6 | 0.040 | 50 | -2.763e-6 | 0.009 | 501–822 | +3.703e-4 | 0.004 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.214e-3 | r0: +1.239e-3, r1: +1.260e-3, r2: +1.160e-3 | r0: 0.517, r1: 0.510, r2: 0.408 | 1.09× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.144e-3 | r0: +1.195e-3, r1: +1.163e-3, r2: +1.079e-3 | r0: 0.425, r1: 0.377, r2: 0.344 | 1.11× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | -4.675e-4 | r0: -3.893e-4, r1: -4.727e-4, r2: -5.395e-4 | r0: 0.004, r1: 0.006, r2: 0.008 | 1.39× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +3.703e-4 | r0: +3.995e-4, r1: +4.150e-4, r2: +2.961e-4 | r0: 0.004, r1: 0.004, r2: 0.002 | 1.40× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇██████████████████████▆▄▄▄▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▄▆▇▇██████████████████████████████▇███████████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 7.04e-1 | 1.37e-1 | 6.41e-2 | 31 | -8.37e-2 | +1.04e-2 | -1.55e-2 | -9.35e-3 |
| 1 | 3.00e-1 | 2 | 5.26e-2 | 7.68e-2 | 6.47e-2 | 5.26e-2 | 25 | -1.51e-2 | +7.51e-3 | -3.80e-3 | -8.09e-3 |
| 2 | 3.00e-1 | 1 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 277 | +1.86e-4 | +1.86e-4 | +1.86e-4 | -7.02e-3 |
| 3 | 3.00e-1 | 1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 1.64e-1 | 295 | +3.67e-3 | +3.67e-3 | +3.67e-3 | -5.67e-3 |
| 4 | 3.00e-1 | 1 | 1.67e-1 | 1.67e-1 | 1.67e-1 | 1.67e-1 | 278 | +6.71e-5 | +6.71e-5 | +6.71e-5 | -4.97e-3 |
| 5 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 278 | +3.56e-4 | +3.56e-4 | +3.56e-4 | -4.33e-3 |
| 6 | 3.00e-1 | 2 | 1.96e-1 | 1.97e-1 | 1.96e-1 | 1.97e-1 | 249 | +4.33e-6 | +2.23e-4 | +1.14e-4 | -3.35e-3 |
| 8 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 327 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -2.99e-3 |
| 9 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 282 | +3.84e-4 | +3.84e-4 | +3.84e-4 | -2.61e-3 |
| 10 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 307 | -1.43e-4 | -1.43e-4 | -1.43e-4 | -2.33e-3 |
| 11 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 259 | +1.86e-6 | +1.86e-6 | +1.86e-6 | -2.08e-3 |
| 12 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 283 | -2.09e-4 | -2.09e-4 | -2.09e-4 | -1.87e-3 |
| 13 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 307 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -1.66e-3 |
| 14 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 283 | +7.86e-5 | +7.86e-5 | +7.86e-5 | -1.48e-3 |
| 15 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 286 | -6.02e-5 | -6.02e-5 | -6.02e-5 | -1.32e-3 |
| 16 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 303 | +2.06e-5 | +2.06e-5 | +2.06e-5 | -1.18e-3 |
| 17 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 280 | +6.66e-6 | +6.66e-6 | +6.66e-6 | -1.06e-3 |
| 18 | 3.00e-1 | 2 | 1.92e-1 | 1.96e-1 | 1.94e-1 | 1.92e-1 | 219 | -9.24e-5 | -6.69e-5 | -7.97e-5 | -8.65e-4 |
| 20 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 291 | -7.93e-5 | -7.93e-5 | -7.93e-5 | -7.84e-4 |
| 21 | 3.00e-1 | 2 | 1.97e-1 | 2.03e-1 | 2.00e-1 | 1.97e-1 | 219 | -1.49e-4 | +3.19e-4 | +8.47e-5 | -6.16e-4 |
| 22 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 251 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -5.71e-4 |
| 23 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 278 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -4.99e-4 |
| 24 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 250 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -4.36e-4 |
| 25 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 265 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -4.04e-4 |
| 26 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 292 | +7.12e-5 | +7.12e-5 | +7.12e-5 | -3.56e-4 |
| 27 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 300 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -3.08e-4 |
| 28 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 293 | +6.07e-5 | +6.07e-5 | +6.07e-5 | -2.70e-4 |
| 29 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 303 | -7.84e-5 | -7.84e-5 | -7.84e-5 | -2.51e-4 |
| 30 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 286 | +7.05e-5 | +7.05e-5 | +7.05e-5 | -2.18e-4 |
| 31 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 280 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -2.10e-4 |
| 32 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 299 | +4.72e-5 | +4.72e-5 | +4.72e-5 | -1.84e-4 |
| 33 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 278 | +2.72e-5 | +2.72e-5 | +2.72e-5 | -1.63e-4 |
| 34 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 256 | -7.68e-5 | -7.68e-5 | -7.68e-5 | -1.54e-4 |
| 35 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 290 | -4.92e-5 | -4.92e-5 | -4.92e-5 | -1.43e-4 |
| 36 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 270 | +1.48e-4 | +1.48e-4 | +1.48e-4 | -1.14e-4 |
| 37 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 292 | -2.69e-5 | -2.69e-5 | -2.69e-5 | -1.05e-4 |
| 38 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 284 | +9.35e-5 | +9.35e-5 | +9.35e-5 | -8.54e-5 |
| 40 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 362 | -8.00e-5 | -8.00e-5 | -8.00e-5 | -8.48e-5 |
| 41 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 290 | +2.67e-4 | +2.67e-4 | +2.67e-4 | -4.95e-5 |
| 42 | 3.00e-1 | 2 | 2.06e-1 | 2.12e-1 | 2.09e-1 | 2.12e-1 | 275 | -2.31e-4 | +9.37e-5 | -6.85e-5 | -5.15e-5 |
| 44 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 335 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -5.69e-5 |
| 45 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 274 | +2.43e-4 | +2.43e-4 | +2.43e-4 | -2.68e-5 |
| 46 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 303 | -2.35e-4 | -2.35e-4 | -2.35e-4 | -4.77e-5 |
| 47 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 302 | +1.19e-4 | +1.19e-4 | +1.19e-4 | -3.10e-5 |
| 48 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 296 | +1.76e-5 | +1.76e-5 | +1.76e-5 | -2.61e-5 |
| 49 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 301 | -4.85e-5 | -4.85e-5 | -4.85e-5 | -2.83e-5 |
| 50 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 321 | +3.02e-5 | +3.02e-5 | +3.02e-5 | -2.25e-5 |
| 51 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 305 | +3.89e-5 | +3.89e-5 | +3.89e-5 | -1.63e-5 |
| 52 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 270 | +3.59e-5 | +3.59e-5 | +3.59e-5 | -1.11e-5 |
| 53 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 284 | -1.72e-4 | -1.72e-4 | -1.72e-4 | -2.73e-5 |
| 54 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 302 | -1.19e-5 | -1.19e-5 | -1.19e-5 | -2.57e-5 |
| 55 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 292 | +1.29e-4 | +1.29e-4 | +1.29e-4 | -1.02e-5 |
| 56 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 270 | +8.81e-6 | +8.81e-6 | +8.81e-6 | -8.33e-6 |
| 57 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 277 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -1.92e-5 |
| 58 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 266 | +6.40e-5 | +6.40e-5 | +6.40e-5 | -1.09e-5 |
| 59 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 274 | -4.39e-5 | -4.39e-5 | -4.39e-5 | -1.42e-5 |
| 60 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 263 | +2.27e-5 | +2.27e-5 | +2.27e-5 | -1.05e-5 |
| 61 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 285 | -9.58e-5 | -9.58e-5 | -9.58e-5 | -1.90e-5 |
| 62 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 273 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -4.73e-6 |
| 63 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 251 | -7.37e-6 | -7.37e-6 | -7.37e-6 | -4.99e-6 |
| 64 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 280 | -1.97e-4 | -1.97e-4 | -1.97e-4 | -2.42e-5 |
| 65 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 310 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -4.22e-6 |
| 66 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 268 | +1.24e-4 | +1.24e-4 | +1.24e-4 | +8.63e-6 |
| 67 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 260 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -5.64e-6 |
| 68 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 246 | -2.74e-5 | -2.74e-5 | -2.74e-5 | -7.82e-6 |
| 69 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 275 | -1.02e-4 | -1.02e-4 | -1.02e-4 | -1.72e-5 |
| 70 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 273 | +2.08e-4 | +2.08e-4 | +2.08e-4 | +5.27e-6 |
| 71 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 265 | -1.40e-4 | -1.40e-4 | -1.40e-4 | -9.26e-6 |
| 72 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 324 | +4.05e-5 | +4.05e-5 | +4.05e-5 | -4.28e-6 |
| 73 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 284 | +2.07e-4 | +2.07e-4 | +2.07e-4 | +1.69e-5 |
| 74 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 267 | -1.49e-4 | -1.49e-4 | -1.49e-4 | +2.84e-7 |
| 75 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 251 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -1.33e-5 |
| 76 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 257 | -4.57e-5 | -4.57e-5 | -4.57e-5 | -1.66e-5 |
| 77 | 3.00e-1 | 2 | 1.96e-1 | 2.04e-1 | 2.00e-1 | 1.96e-1 | 215 | -1.92e-4 | +2.86e-5 | -8.19e-5 | -3.01e-5 |
| 79 | 3.00e-1 | 2 | 1.95e-1 | 2.25e-1 | 2.10e-1 | 2.25e-1 | 240 | -8.89e-6 | +5.98e-4 | +2.95e-4 | +3.47e-5 |
| 80 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 280 | -4.16e-4 | -4.16e-4 | -4.16e-4 | -1.04e-5 |
| 82 | 3.00e-1 | 2 | 2.12e-1 | 2.25e-1 | 2.19e-1 | 2.25e-1 | 215 | +1.64e-4 | +2.69e-4 | +2.17e-4 | +3.33e-5 |
| 83 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 258 | -6.14e-4 | -6.14e-4 | -6.14e-4 | -3.14e-5 |
| 84 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 294 | +2.54e-4 | +2.54e-4 | +2.54e-4 | -2.89e-6 |
| 85 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 253 | +1.27e-4 | +1.27e-4 | +1.27e-4 | +1.01e-5 |
| 86 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 257 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -4.12e-6 |
| 87 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 256 | -3.99e-6 | -3.99e-6 | -3.99e-6 | -4.11e-6 |
| 88 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 252 | -3.77e-5 | -3.77e-5 | -3.77e-5 | -7.47e-6 |
| 89 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 236 | +7.60e-5 | +7.60e-5 | +7.60e-5 | +8.73e-7 |
| 90 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 269 | -4.79e-5 | -4.79e-5 | -4.79e-5 | -4.00e-6 |
| 91 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 271 | +9.11e-5 | +9.11e-5 | +9.11e-5 | +5.51e-6 |
| 92 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 270 | +4.41e-5 | +4.41e-5 | +4.41e-5 | +9.36e-6 |
| 93 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 293 | -1.82e-5 | -1.82e-5 | -1.82e-5 | +6.60e-6 |
| 94 | 3.00e-1 | 2 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 238 | -5.41e-6 | +2.56e-5 | +1.01e-5 | +7.11e-6 |
| 96 | 3.00e-1 | 2 | 2.01e-1 | 2.20e-1 | 2.11e-1 | 2.20e-1 | 251 | -1.82e-4 | +3.54e-4 | +8.61e-5 | +2.48e-5 |
| 98 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 341 | -2.73e-4 | -2.73e-4 | -2.73e-4 | -4.96e-6 |
| 99 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 299 | +3.87e-4 | +3.87e-4 | +3.87e-4 | +3.42e-5 |
| 100 | 3.00e-2 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 260 | -8.87e-5 | -8.87e-5 | -8.87e-5 | +2.19e-5 |
| 101 | 3.00e-2 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 282 | -1.55e-4 | -1.55e-4 | -1.55e-4 | +4.20e-6 |
| 102 | 3.00e-2 | 2 | 2.17e-2 | 2.25e-2 | 2.21e-2 | 2.25e-2 | 236 | -8.14e-3 | +1.47e-4 | -4.00e-3 | -7.15e-4 |
| 104 | 3.00e-2 | 1 | 2.20e-2 | 2.20e-2 | 2.20e-2 | 2.20e-2 | 332 | -6.77e-5 | -6.77e-5 | -6.77e-5 | -6.50e-4 |
| 105 | 3.00e-2 | 1 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 278 | +6.66e-4 | +6.66e-4 | +6.66e-4 | -5.19e-4 |
| 106 | 3.00e-2 | 1 | 2.61e-2 | 2.61e-2 | 2.61e-2 | 2.61e-2 | 311 | -3.55e-5 | -3.55e-5 | -3.55e-5 | -4.70e-4 |
| 107 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 274 | +2.23e-4 | +2.23e-4 | +2.23e-4 | -4.01e-4 |
| 108 | 3.00e-2 | 1 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 276 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -3.71e-4 |
| 109 | 3.00e-2 | 2 | 2.81e-2 | 2.83e-2 | 2.82e-2 | 2.83e-2 | 220 | +3.12e-5 | +1.54e-4 | +9.27e-5 | -2.84e-4 |
| 110 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 228 | -2.85e-4 | -2.85e-4 | -2.85e-4 | -2.84e-4 |
| 111 | 3.00e-2 | 1 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 248 | +2.46e-4 | +2.46e-4 | +2.46e-4 | -2.31e-4 |
| 112 | 3.00e-2 | 1 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 249 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -1.93e-4 |
| 113 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 246 | +5.55e-5 | +5.55e-5 | +5.55e-5 | -1.68e-4 |
| 114 | 3.00e-2 | 1 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 3.11e-2 | 251 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -1.32e-4 |
| 115 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 261 | +6.80e-5 | +6.80e-5 | +6.80e-5 | -1.12e-4 |
| 116 | 3.00e-2 | 1 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 264 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -8.75e-5 |
| 117 | 3.00e-2 | 2 | 3.32e-2 | 3.41e-2 | 3.36e-2 | 3.41e-2 | 234 | +4.35e-5 | +1.14e-4 | +7.89e-5 | -5.55e-5 |
| 119 | 3.00e-2 | 2 | 3.30e-2 | 3.76e-2 | 3.53e-2 | 3.76e-2 | 234 | -1.01e-4 | +5.61e-4 | +2.30e-4 | +2.01e-6 |
| 121 | 3.00e-2 | 2 | 3.45e-2 | 4.05e-2 | 3.75e-2 | 4.05e-2 | 249 | -2.62e-4 | +6.44e-4 | +1.91e-4 | +4.25e-5 |
| 122 | 3.00e-2 | 1 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 3.65e-2 | 242 | -4.30e-4 | -4.30e-4 | -4.30e-4 | -4.74e-6 |
| 123 | 3.00e-2 | 1 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 264 | +6.38e-5 | +6.38e-5 | +6.38e-5 | +2.12e-6 |
| 124 | 3.00e-2 | 1 | 3.92e-2 | 3.92e-2 | 3.92e-2 | 3.92e-2 | 252 | +2.13e-4 | +2.13e-4 | +2.13e-4 | +2.32e-5 |
| 125 | 3.00e-2 | 1 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 3.90e-2 | 252 | -1.76e-5 | -1.76e-5 | -1.76e-5 | +1.91e-5 |
| 126 | 3.00e-2 | 1 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 3.99e-2 | 250 | +9.19e-5 | +9.19e-5 | +9.19e-5 | +2.64e-5 |
| 127 | 3.00e-2 | 2 | 3.93e-2 | 4.03e-2 | 3.98e-2 | 3.93e-2 | 203 | -1.24e-4 | +3.58e-5 | -4.43e-5 | +1.22e-5 |
| 128 | 3.00e-2 | 1 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 233 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -6.44e-6 |
| 129 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 259 | +4.20e-4 | +4.20e-4 | +4.20e-4 | +3.62e-5 |
| 130 | 3.00e-2 | 1 | 4.35e-2 | 4.35e-2 | 4.35e-2 | 4.35e-2 | 235 | +1.44e-4 | +1.44e-4 | +1.44e-4 | +4.71e-5 |
| 131 | 3.00e-2 | 1 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 4.26e-2 | 245 | -8.58e-5 | -8.58e-5 | -8.58e-5 | +3.38e-5 |
| 132 | 3.00e-2 | 1 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 4.38e-2 | 249 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +4.15e-5 |
| 133 | 3.00e-2 | 1 | 4.48e-2 | 4.48e-2 | 4.48e-2 | 4.48e-2 | 264 | +9.00e-5 | +9.00e-5 | +9.00e-5 | +4.63e-5 |
| 134 | 3.00e-2 | 1 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 269 | +1.75e-4 | +1.75e-4 | +1.75e-4 | +5.92e-5 |
| 135 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 266 | +1.13e-4 | +1.13e-4 | +1.13e-4 | +6.46e-5 |
| 136 | 3.00e-2 | 2 | 4.82e-2 | 4.86e-2 | 4.84e-2 | 4.86e-2 | 242 | -1.77e-5 | +3.55e-5 | +8.90e-6 | +5.42e-5 |
| 138 | 3.00e-2 | 2 | 4.66e-2 | 5.14e-2 | 4.90e-2 | 5.14e-2 | 205 | -1.47e-4 | +4.79e-4 | +1.66e-4 | +7.86e-5 |
| 139 | 3.00e-2 | 1 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 255 | -4.60e-4 | -4.60e-4 | -4.60e-4 | +2.47e-5 |
| 140 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 232 | +3.58e-4 | +3.58e-4 | +3.58e-4 | +5.81e-5 |
| 141 | 3.00e-2 | 1 | 5.06e-2 | 5.06e-2 | 5.06e-2 | 5.06e-2 | 236 | +7.89e-5 | +7.89e-5 | +7.89e-5 | +6.02e-5 |
| 142 | 3.00e-2 | 2 | 4.97e-2 | 5.08e-2 | 5.03e-2 | 5.08e-2 | 205 | -7.70e-5 | +1.07e-4 | +1.49e-5 | +5.25e-5 |
| 143 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 290 | -1.67e-4 | -1.67e-4 | -1.67e-4 | +3.05e-5 |
| 144 | 3.00e-2 | 1 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 5.72e-2 | 220 | +7.58e-4 | +7.58e-4 | +7.58e-4 | +1.03e-4 |
| 145 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 227 | -4.55e-4 | -4.55e-4 | -4.55e-4 | +4.73e-5 |
| 146 | 3.00e-2 | 1 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 5.18e-2 | 244 | +1.91e-5 | +1.91e-5 | +1.91e-5 | +4.45e-5 |
| 147 | 3.00e-2 | 1 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 255 | +1.60e-4 | +1.60e-4 | +1.60e-4 | +5.61e-5 |
| 148 | 3.00e-2 | 2 | 5.61e-2 | 5.68e-2 | 5.64e-2 | 5.68e-2 | 220 | +5.70e-5 | +1.51e-4 | +1.04e-4 | +6.47e-5 |
| 150 | 3.00e-3 | 2 | 5.37e-2 | 6.31e-2 | 5.84e-2 | 6.31e-2 | 187 | -1.70e-4 | +8.67e-4 | +3.48e-4 | +1.24e-4 |
| 151 | 3.00e-3 | 1 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 5.32e-3 | 234 | -1.06e-2 | -1.06e-2 | -1.06e-2 | -9.45e-4 |
| 152 | 3.00e-3 | 1 | 5.35e-3 | 5.35e-3 | 5.35e-3 | 5.35e-3 | 236 | +2.08e-5 | +2.08e-5 | +2.08e-5 | -8.49e-4 |
| 153 | 3.00e-3 | 1 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 216 | +4.10e-5 | +4.10e-5 | +4.10e-5 | -7.60e-4 |
| 154 | 3.00e-3 | 2 | 5.25e-3 | 5.27e-3 | 5.26e-3 | 5.25e-3 | 212 | -1.01e-4 | -2.22e-5 | -6.17e-5 | -6.27e-4 |
| 155 | 3.00e-3 | 1 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 5.12e-3 | 247 | -9.88e-5 | -9.88e-5 | -9.88e-5 | -5.74e-4 |
| 156 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 254 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -4.97e-4 |
| 157 | 3.00e-3 | 1 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 5.45e-3 | 250 | +5.15e-5 | +5.15e-5 | +5.15e-5 | -4.42e-4 |
| 158 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 260 | +7.24e-5 | +7.24e-5 | +7.24e-5 | -3.91e-4 |
| 159 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 264 | +6.95e-5 | +6.95e-5 | +6.95e-5 | -3.45e-4 |
| 160 | 3.00e-3 | 2 | 5.74e-3 | 5.76e-3 | 5.75e-3 | 5.74e-3 | 223 | -1.33e-5 | +7.32e-5 | +2.99e-5 | -2.74e-4 |
| 161 | 3.00e-3 | 1 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 270 | -2.50e-4 | -2.50e-4 | -2.50e-4 | -2.72e-4 |
| 162 | 3.00e-3 | 1 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 6.04e-3 | 270 | +4.36e-4 | +4.36e-4 | +4.36e-4 | -2.01e-4 |
| 163 | 3.00e-3 | 1 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 301 | -8.46e-5 | -8.46e-5 | -8.46e-5 | -1.89e-4 |
| 164 | 3.00e-3 | 1 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 6.30e-3 | 240 | +2.81e-4 | +2.81e-4 | +2.81e-4 | -1.42e-4 |
| 165 | 3.00e-3 | 1 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 257 | -3.97e-4 | -3.97e-4 | -3.97e-4 | -1.68e-4 |
| 166 | 3.00e-3 | 1 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 280 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -1.40e-4 |
| 167 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 268 | +5.54e-5 | +5.54e-5 | +5.54e-5 | -1.20e-4 |
| 168 | 3.00e-3 | 2 | 5.60e-3 | 5.93e-3 | 5.77e-3 | 5.60e-3 | 197 | -2.96e-4 | -1.40e-5 | -1.55e-4 | -1.28e-4 |
| 169 | 3.00e-3 | 1 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 5.18e-3 | 250 | -3.08e-4 | -3.08e-4 | -3.08e-4 | -1.46e-4 |
| 170 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 212 | +5.19e-4 | +5.19e-4 | +5.19e-4 | -7.98e-5 |
| 171 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 214 | -2.56e-4 | -2.56e-4 | -2.56e-4 | -9.74e-5 |
| 172 | 3.00e-3 | 2 | 5.54e-3 | 5.74e-3 | 5.64e-3 | 5.74e-3 | 182 | +4.54e-5 | +2.00e-4 | +1.23e-4 | -5.47e-5 |
| 173 | 3.00e-3 | 1 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 5.24e-3 | 221 | -4.16e-4 | -4.16e-4 | -4.16e-4 | -9.09e-5 |
| 174 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 222 | +2.87e-4 | +2.87e-4 | +2.87e-4 | -5.32e-5 |
| 175 | 3.00e-3 | 2 | 5.55e-3 | 5.73e-3 | 5.64e-3 | 5.73e-3 | 192 | -2.28e-5 | +1.59e-4 | +6.83e-5 | -2.92e-5 |
| 176 | 3.00e-3 | 1 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 5.23e-3 | 286 | -3.17e-4 | -3.17e-4 | -3.17e-4 | -5.80e-5 |
| 177 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 248 | +8.57e-4 | +8.57e-4 | +8.57e-4 | +3.35e-5 |
| 178 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 242 | -2.04e-4 | -2.04e-4 | -2.04e-4 | +9.68e-6 |
| 179 | 3.00e-3 | 2 | 5.82e-3 | 6.08e-3 | 5.95e-3 | 6.08e-3 | 202 | -2.44e-4 | +2.12e-4 | -1.63e-5 | +7.02e-6 |
| 180 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 222 | -3.76e-4 | -3.76e-4 | -3.76e-4 | -3.13e-5 |
| 181 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 230 | +2.05e-4 | +2.05e-4 | +2.05e-4 | -7.65e-6 |
| 182 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 266 | +9.29e-5 | +9.29e-5 | +9.29e-5 | +2.41e-6 |
| 183 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 239 | +2.59e-4 | +2.59e-4 | +2.59e-4 | +2.80e-5 |
| 184 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 223 | -2.10e-4 | -2.10e-4 | -2.10e-4 | +4.20e-6 |
| 185 | 3.00e-3 | 2 | 6.01e-3 | 6.03e-3 | 6.02e-3 | 6.01e-3 | 202 | -4.96e-5 | -1.16e-5 | -3.06e-5 | -2.22e-6 |
| 186 | 3.00e-3 | 1 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 272 | -2.04e-4 | -2.04e-4 | -2.04e-4 | -2.24e-5 |
| 187 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 265 | +4.26e-4 | +4.26e-4 | +4.26e-4 | +2.25e-5 |
| 188 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 252 | +9.74e-5 | +9.74e-5 | +9.74e-5 | +2.99e-5 |
| 189 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 225 | -1.37e-4 | -1.37e-4 | -1.37e-4 | +1.33e-5 |
| 190 | 3.00e-3 | 2 | 5.61e-3 | 6.18e-3 | 5.89e-3 | 5.61e-3 | 179 | -5.46e-4 | -1.11e-4 | -3.28e-4 | -5.38e-5 |
| 191 | 3.00e-3 | 1 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 5.65e-3 | 228 | +3.54e-5 | +3.54e-5 | +3.54e-5 | -4.49e-5 |
| 192 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 245 | +3.50e-4 | +3.50e-4 | +3.50e-4 | -5.45e-6 |
| 193 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 249 | -3.67e-6 | -3.67e-6 | -3.67e-6 | -5.27e-6 |
| 194 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 238 | +1.14e-4 | +1.14e-4 | +1.14e-4 | +6.65e-6 |
| 195 | 3.00e-3 | 2 | 6.26e-3 | 6.37e-3 | 6.31e-3 | 6.37e-3 | 220 | -4.21e-5 | +8.09e-5 | +1.94e-5 | +9.69e-6 |
| 196 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 260 | -2.59e-4 | -2.59e-4 | -2.59e-4 | -1.72e-5 |
| 197 | 3.00e-3 | 1 | 6.72e-3 | 6.72e-3 | 6.72e-3 | 6.72e-3 | 250 | +4.83e-4 | +4.83e-4 | +4.83e-4 | +3.28e-5 |
| 198 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 260 | -2.78e-4 | -2.78e-4 | -2.78e-4 | +1.68e-6 |
| 199 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 254 | +1.49e-4 | +1.49e-4 | +1.49e-4 | +1.64e-5 |

