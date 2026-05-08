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
| cpu-async | 0.050737 | 0.9194 | +0.0069 | 1817.3 | 316 | 82.5 | 100% | 100% | 100% | 8.0 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9194 | cpu-async | - | - |

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
| cpu-async | 2.0066 | 0.7757 | 0.6203 | 0.5634 | 0.5327 | 0.5167 | 0.4893 | 0.4817 | 0.4708 | 0.4731 | 0.1954 | 0.1642 | 0.1488 | 0.1387 | 0.1290 | 0.0701 | 0.0632 | 0.0562 | 0.0530 | 0.0507 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3991 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3040 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2969 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 390 | 379 | 384 | 382 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1028.7 | 1.4 | epoch-boundary(112) |
| cpu-async | gpu2 | 1028.7 | 1.2 | epoch-boundary(112) |
| cpu-async | gpu2 | 512.0 | 1.1 | epoch-boundary(55) |
| cpu-async | gpu1 | 1380.1 | 1.0 | epoch-boundary(151) |
| cpu-async | gpu2 | 1380.3 | 0.8 | epoch-boundary(151) |
| cpu-async | gpu0 | 0.4 | 0.7 | cpu-avg |
| cpu-async | gpu2 | 866.2 | 0.6 | epoch-boundary(94) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.7s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 2.4s | 0.0s | 0.0s | 0.0s | 3.0s |
| resnet-graph | cpu-async | gpu2 | 3.7s | 0.0s | 0.0s | 0.0s | 4.3s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 262 | 0 | 316 | 82.5 | 6153/10087 | 316 | 82.5 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 212.5 | 11.7% |

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
| resnet-graph | cpu-async | 186 | 316 | 0 | 4.65e-3 | -1.13e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 316 | 9.65e-2 | 7.83e-2 | 4.50e-3 | 4.86e-1 | 25.0 | -2.81e-4 | 2.44e-3 |
| resnet-graph | cpu-async | 1 | 316 | 9.69e-2 | 7.70e-2 | 4.44e-3 | 3.67e-1 | 37.7 | -2.76e-4 | 2.26e-3 |
| resnet-graph | cpu-async | 2 | 316 | 9.68e-2 | 7.71e-2 | 4.35e-3 | 3.91e-1 | 37.3 | -2.65e-4 | 2.19e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9895 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9912 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9970 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 44 (0,1,2,3,4,5,6,7…143,146) | 0 (—) | — | 0,1,2,3,4,5,6,7…143,146 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 13 | 13 |
| resnet-graph | cpu-async | 0e0 | 5 | 7 | 7 |
| resnet-graph | cpu-async | 0e0 | 10 | 3 | 3 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 184 | +0.297 |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | 59 | +0.246 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 69 | +0.091 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 314 | -0.008 | 185 | +0.389 | +0.542 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 315 | 3.38e1–7.99e1 | 6.45e1 | 3.87e-3 | 6.19e-3 | 6.70e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 186 | 33–78054 | +1.023e-5 | 0.421 | +1.049e-5 | 0.442 | 93 | +5.316e-6 | 0.243 | 31–945 | +1.194e-3 | 0.708 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 173 | 922–78054 | +9.383e-6 | 0.489 | +9.459e-6 | 0.496 | 92 | +4.773e-6 | 0.217 | 77–945 | +1.137e-3 | 0.860 |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | 60 | 78544–115704 | +3.204e-5 | 0.664 | +3.243e-5 | 0.686 | 45 | +2.832e-5 | 0.589 | 285–1052 | +1.624e-3 | 0.658 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 70 | 116542–156114 | -1.329e-5 | 0.143 | -1.337e-5 | 0.145 | 48 | -9.950e-6 | 0.108 | 369–843 | +1.317e-3 | 0.136 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.194e-3 | r0: +1.166e-3, r1: +1.220e-3, r2: +1.201e-3 | r0: 0.663, r1: 0.726, r2: 0.723 | 1.05× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.137e-3 | r0: +1.121e-3, r1: +1.149e-3, r2: +1.142e-3 | r0: 0.851, r1: 0.858, r2: 0.854 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | +1.624e-3 | r0: +1.634e-3, r1: +1.621e-3, r2: +1.619e-3 | r0: 0.668, r1: 0.654, r2: 0.646 | 1.01× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +1.317e-3 | r0: +1.266e-3, r1: +1.283e-3, r2: +1.400e-3 | r0: 0.126, r1: 0.131, r2: 0.151 | 1.11× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇████████████████████▅▄▄▅▅▅▅▅▅▅▆▅▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▆▇▇██████████████████▇▇▇█████████▇▇██████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 4.04e-2 | 4.86e-1 | 1.18e-1 | 5.89e-2 | 26 | -7.31e-2 | +1.06e-2 | -1.54e-2 | -9.29e-3 |
| 1 | 3.00e-1 | 10 | 6.05e-2 | 1.13e-1 | 7.80e-2 | 7.88e-2 | 32 | -2.41e-2 | +8.47e-3 | -7.28e-4 | -2.86e-3 |
| 2 | 3.00e-1 | 5 | 7.93e-2 | 1.16e-1 | 8.94e-2 | 7.93e-2 | 30 | -8.35e-3 | +4.84e-3 | -1.29e-3 | -2.20e-3 |
| 3 | 3.00e-1 | 5 | 1.01e-1 | 1.43e-1 | 1.12e-1 | 1.08e-1 | 45 | -7.28e-3 | +6.74e-3 | +7.78e-5 | -1.26e-3 |
| 4 | 3.00e-1 | 9 | 9.32e-2 | 1.60e-1 | 1.05e-1 | 1.03e-1 | 42 | -1.17e-2 | +3.73e-3 | -7.77e-4 | -7.58e-4 |
| 5 | 3.00e-1 | 4 | 8.99e-2 | 1.57e-1 | 1.09e-1 | 9.79e-2 | 41 | -1.55e-2 | +4.00e-3 | -2.32e-3 | -1.26e-3 |
| 6 | 3.00e-1 | 5 | 9.23e-2 | 1.43e-1 | 1.11e-1 | 1.03e-1 | 42 | -1.19e-2 | +4.20e-3 | -1.12e-3 | -1.18e-3 |
| 7 | 3.00e-1 | 6 | 1.01e-1 | 1.41e-1 | 1.12e-1 | 1.04e-1 | 45 | -7.83e-3 | +3.64e-3 | -6.02e-4 | -9.06e-4 |
| 8 | 3.00e-1 | 6 | 9.74e-2 | 1.39e-1 | 1.07e-1 | 9.75e-2 | 37 | -7.72e-3 | +3.46e-3 | -9.27e-4 | -9.05e-4 |
| 9 | 3.00e-1 | 9 | 9.08e-2 | 1.45e-1 | 1.01e-1 | 9.28e-2 | 32 | -1.04e-2 | +4.33e-3 | -8.68e-4 | -7.90e-4 |
| 10 | 3.00e-1 | 4 | 8.96e-2 | 1.46e-1 | 1.09e-1 | 1.07e-1 | 46 | -1.47e-2 | +5.30e-3 | -1.34e-3 | -9.27e-4 |
| 11 | 3.00e-1 | 5 | 1.04e-1 | 1.43e-1 | 1.15e-1 | 1.06e-1 | 41 | -6.57e-3 | +3.54e-3 | -6.67e-4 | -8.34e-4 |
| 12 | 3.00e-1 | 1 | 1.01e-1 | 1.01e-1 | 1.01e-1 | 1.01e-1 | 41 | -1.17e-3 | -1.17e-3 | -1.17e-3 | -8.68e-4 |
| 13 | 3.00e-1 | 2 | 1.98e-1 | 2.22e-1 | 2.10e-1 | 1.98e-1 | 260 | -4.32e-4 | +2.31e-3 | +9.41e-4 | -5.38e-4 |
| 15 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 333 | +2.51e-4 | +2.51e-4 | +2.51e-4 | -4.59e-4 |
| 16 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 313 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -4.25e-4 |
| 17 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 313 | -6.78e-5 | -6.78e-5 | -6.78e-5 | -3.89e-4 |
| 18 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 307 | -6.57e-6 | -6.57e-6 | -6.57e-6 | -3.51e-4 |
| 19 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 320 | +7.92e-5 | +7.92e-5 | +7.92e-5 | -3.08e-4 |
| 20 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 303 | -5.93e-5 | -5.93e-5 | -5.93e-5 | -2.83e-4 |
| 21 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 262 | -2.74e-4 | -2.74e-4 | -2.74e-4 | -2.82e-4 |
| 22 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 277 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -2.34e-4 |
| 24 | 3.00e-1 | 2 | 1.88e-1 | 2.14e-1 | 2.01e-1 | 1.88e-1 | 252 | -5.04e-4 | +1.83e-4 | -1.60e-4 | -2.23e-4 |
| 26 | 3.00e-1 | 2 | 1.93e-1 | 2.16e-1 | 2.05e-1 | 1.93e-1 | 254 | -4.26e-4 | +3.93e-4 | -1.64e-5 | -1.88e-4 |
| 28 | 3.00e-1 | 2 | 1.95e-1 | 2.09e-1 | 2.02e-1 | 1.95e-1 | 252 | -2.76e-4 | +2.52e-4 | -1.20e-5 | -1.57e-4 |
| 30 | 3.00e-1 | 2 | 1.94e-1 | 2.05e-1 | 2.00e-1 | 1.94e-1 | 261 | -2.23e-4 | +1.60e-4 | -3.15e-5 | -1.35e-4 |
| 32 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 355 | +3.83e-4 | +3.83e-4 | +3.83e-4 | -8.34e-5 |
| 33 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 287 | -3.89e-4 | -3.89e-4 | -3.89e-4 | -1.14e-4 |
| 34 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 268 | -6.34e-5 | -6.34e-5 | -6.34e-5 | -1.09e-4 |
| 35 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 270 | +4.23e-5 | +4.23e-5 | +4.23e-5 | -9.38e-5 |
| 36 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 270 | +1.07e-5 | +1.07e-5 | +1.07e-5 | -8.33e-5 |
| 37 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 269 | +2.76e-5 | +2.76e-5 | +2.76e-5 | -7.22e-5 |
| 38 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 277 | +5.24e-6 | +5.24e-6 | +5.24e-6 | -6.45e-5 |
| 39 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 281 | +1.73e-5 | +1.73e-5 | +1.73e-5 | -5.63e-5 |
| 40 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 302 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -3.64e-5 |
| 41 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 299 | -9.55e-5 | -9.55e-5 | -9.55e-5 | -4.23e-5 |
| 42 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 290 | +6.47e-5 | +6.47e-5 | +6.47e-5 | -3.16e-5 |
| 43 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 269 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -4.50e-5 |
| 44 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 281 | +9.80e-5 | +9.80e-5 | +9.80e-5 | -3.07e-5 |
| 45 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 248 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -4.17e-5 |
| 46 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 278 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -2.14e-5 |
| 47 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 272 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -3.32e-5 |
| 48 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 265 | +3.66e-5 | +3.66e-5 | +3.66e-5 | -2.62e-5 |
| 49 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 275 | +7.58e-5 | +7.58e-5 | +7.58e-5 | -1.60e-5 |
| 50 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 275 | +7.50e-5 | +7.50e-5 | +7.50e-5 | -6.91e-6 |
| 51 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 290 | +1.07e-5 | +1.07e-5 | +1.07e-5 | -5.15e-6 |
| 52 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 251 | -2.59e-4 | -2.59e-4 | -2.59e-4 | -3.05e-5 |
| 53 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 293 | +2.22e-4 | +2.22e-4 | +2.22e-4 | -5.25e-6 |
| 54 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 250 | -2.33e-4 | -2.33e-4 | -2.33e-4 | -2.80e-5 |
| 55 | 3.00e-1 | 2 | 1.92e-1 | 2.08e-1 | 2.00e-1 | 1.92e-1 | 224 | -3.62e-4 | +1.91e-4 | -8.56e-5 | -4.17e-5 |
| 56 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 240 | +9.43e-5 | +9.43e-5 | +9.43e-5 | -2.81e-5 |
| 57 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 280 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -2.88e-6 |
| 58 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 232 | -3.84e-4 | -3.84e-4 | -3.84e-4 | -4.10e-5 |
| 59 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 225 | +9.39e-6 | +9.39e-6 | +9.39e-6 | -3.59e-5 |
| 60 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 246 | +9.80e-5 | +9.80e-5 | +9.80e-5 | -2.26e-5 |
| 61 | 3.00e-1 | 2 | 1.88e-1 | 1.93e-1 | 1.90e-1 | 1.88e-1 | 205 | -1.38e-4 | -7.65e-5 | -1.07e-4 | -3.89e-5 |
| 62 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 255 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -6.12e-6 |
| 63 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 238 | -1.49e-4 | -1.49e-4 | -1.49e-4 | -2.04e-5 |
| 64 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 242 | +9.91e-5 | +9.91e-5 | +9.91e-5 | -8.44e-6 |
| 65 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 263 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +2.86e-6 |
| 66 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 259 | +5.23e-5 | +5.23e-5 | +5.23e-5 | +7.81e-6 |
| 67 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 270 | -9.21e-6 | -9.21e-6 | -9.21e-6 | +6.11e-6 |
| 68 | 3.00e-1 | 2 | 1.81e-1 | 1.91e-1 | 1.86e-1 | 1.81e-1 | 190 | -3.60e-4 | -2.91e-4 | -3.25e-4 | -5.66e-5 |
| 69 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 223 | +3.45e-4 | +3.45e-4 | +3.45e-4 | -1.64e-5 |
| 70 | 3.00e-1 | 2 | 1.84e-1 | 2.01e-1 | 1.92e-1 | 1.84e-1 | 190 | -4.69e-4 | +1.19e-4 | -1.75e-4 | -4.95e-5 |
| 71 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 217 | +2.01e-4 | +2.01e-4 | +2.01e-4 | -2.44e-5 |
| 72 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 227 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -2.75e-6 |
| 73 | 3.00e-1 | 2 | 1.88e-1 | 2.02e-1 | 1.95e-1 | 1.88e-1 | 190 | -3.87e-4 | +3.32e-5 | -1.77e-4 | -3.79e-5 |
| 75 | 3.00e-1 | 2 | 1.83e-1 | 2.12e-1 | 1.98e-1 | 1.83e-1 | 190 | -7.81e-4 | +4.70e-4 | -1.55e-4 | -6.65e-5 |
| 76 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 212 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -4.24e-5 |
| 77 | 3.00e-1 | 2 | 1.86e-1 | 2.00e-1 | 1.93e-1 | 1.86e-1 | 190 | -3.92e-4 | +2.27e-4 | -8.22e-5 | -5.30e-5 |
| 78 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 204 | +3.52e-5 | +3.52e-5 | +3.52e-5 | -4.42e-5 |
| 79 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 210 | +2.10e-4 | +2.10e-4 | +2.10e-4 | -1.88e-5 |
| 80 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 176 | -4.97e-4 | -4.97e-4 | -4.97e-4 | -6.66e-5 |
| 81 | 3.00e-1 | 2 | 1.79e-1 | 2.09e-1 | 1.94e-1 | 1.79e-1 | 176 | -8.73e-4 | +6.16e-4 | -1.28e-4 | -8.58e-5 |
| 82 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 225 | +4.43e-4 | +4.43e-4 | +4.43e-4 | -3.29e-5 |
| 83 | 3.00e-1 | 2 | 1.79e-1 | 1.95e-1 | 1.87e-1 | 1.79e-1 | 176 | -5.03e-4 | -6.31e-5 | -2.83e-4 | -8.26e-5 |
| 84 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 224 | +6.28e-4 | +6.28e-4 | +6.28e-4 | -1.15e-5 |
| 85 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 238 | -1.62e-5 | -1.62e-5 | -1.62e-5 | -1.20e-5 |
| 86 | 3.00e-1 | 2 | 1.73e-1 | 1.86e-1 | 1.79e-1 | 1.73e-1 | 162 | -4.96e-4 | -4.69e-4 | -4.83e-4 | -1.01e-4 |
| 87 | 3.00e-1 | 2 | 1.72e-1 | 1.97e-1 | 1.85e-1 | 1.72e-1 | 152 | -8.79e-4 | +6.47e-4 | -1.16e-4 | -1.12e-4 |
| 88 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 194 | +5.64e-4 | +5.64e-4 | +5.64e-4 | -4.42e-5 |
| 89 | 3.00e-1 | 2 | 1.75e-1 | 1.85e-1 | 1.80e-1 | 1.75e-1 | 152 | -3.69e-4 | -2.22e-4 | -2.96e-4 | -9.27e-5 |
| 90 | 3.00e-1 | 2 | 1.76e-1 | 1.86e-1 | 1.81e-1 | 1.76e-1 | 163 | -3.44e-4 | +3.59e-4 | +7.87e-6 | -7.71e-5 |
| 91 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 196 | +5.12e-4 | +5.12e-4 | +5.12e-4 | -1.82e-5 |
| 92 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 194 | -5.86e-5 | -5.86e-5 | -5.86e-5 | -2.23e-5 |
| 93 | 3.00e-1 | 2 | 1.79e-1 | 1.88e-1 | 1.84e-1 | 1.79e-1 | 163 | -2.98e-4 | -1.22e-4 | -2.10e-4 | -5.88e-5 |
| 94 | 3.00e-1 | 2 | 1.68e-1 | 1.98e-1 | 1.83e-1 | 1.68e-1 | 145 | -1.13e-3 | +4.51e-4 | -3.38e-4 | -1.20e-4 |
| 95 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 203 | +7.29e-4 | +7.29e-4 | +7.29e-4 | -3.49e-5 |
| 96 | 3.00e-1 | 2 | 1.74e-1 | 1.84e-1 | 1.79e-1 | 1.74e-1 | 156 | -3.68e-4 | -3.18e-4 | -3.43e-4 | -9.37e-5 |
| 97 | 3.00e-1 | 2 | 1.61e-1 | 1.91e-1 | 1.76e-1 | 1.61e-1 | 125 | -1.39e-3 | +4.87e-4 | -4.53e-4 | -1.71e-4 |
| 98 | 3.00e-1 | 2 | 1.58e-1 | 1.83e-1 | 1.70e-1 | 1.58e-1 | 125 | -1.20e-3 | +8.02e-4 | -1.99e-4 | -1.87e-4 |
| 99 | 3.00e-1 | 2 | 1.62e-1 | 1.80e-1 | 1.71e-1 | 1.62e-1 | 126 | -8.43e-4 | +8.48e-4 | +2.45e-6 | -1.59e-4 |
| 100 | 3.00e-2 | 1 | 1.03e-1 | 1.03e-1 | 1.03e-1 | 1.03e-1 | 186 | -2.40e-3 | -2.40e-3 | -2.40e-3 | -3.84e-4 |
| 101 | 3.00e-2 | 3 | 1.73e-2 | 1.93e-2 | 1.80e-2 | 1.75e-2 | 135 | -8.99e-3 | +5.95e-5 | -3.23e-3 | -1.07e-3 |
| 102 | 3.00e-2 | 1 | 2.04e-2 | 2.04e-2 | 2.04e-2 | 2.04e-2 | 174 | +8.97e-4 | +8.97e-4 | +8.97e-4 | -8.75e-4 |
| 103 | 3.00e-2 | 2 | 1.83e-2 | 2.01e-2 | 1.92e-2 | 1.83e-2 | 122 | -8.02e-4 | -9.09e-5 | -4.46e-4 | -7.97e-4 |
| 104 | 3.00e-2 | 2 | 1.91e-2 | 2.07e-2 | 1.99e-2 | 1.91e-2 | 122 | -6.51e-4 | +7.68e-4 | +5.84e-5 | -6.42e-4 |
| 105 | 3.00e-2 | 2 | 1.96e-2 | 2.30e-2 | 2.13e-2 | 1.96e-2 | 114 | -1.39e-3 | +1.04e-3 | -1.76e-4 | -5.65e-4 |
| 106 | 3.00e-2 | 2 | 1.87e-2 | 2.08e-2 | 1.98e-2 | 1.87e-2 | 114 | -9.45e-4 | +4.33e-4 | -2.56e-4 | -5.13e-4 |
| 107 | 3.00e-2 | 2 | 1.95e-2 | 2.32e-2 | 2.13e-2 | 1.95e-2 | 119 | -1.49e-3 | +1.31e-3 | -8.87e-5 | -4.47e-4 |
| 108 | 3.00e-2 | 3 | 2.06e-2 | 2.28e-2 | 2.14e-2 | 2.08e-2 | 114 | -9.03e-4 | +1.02e-3 | +6.66e-5 | -3.16e-4 |
| 109 | 3.00e-2 | 2 | 2.01e-2 | 2.14e-2 | 2.07e-2 | 2.01e-2 | 107 | -6.00e-4 | +2.20e-4 | -1.90e-4 | -2.96e-4 |
| 110 | 3.00e-2 | 2 | 2.06e-2 | 2.25e-2 | 2.15e-2 | 2.06e-2 | 107 | -8.29e-4 | +8.03e-4 | -1.30e-5 | -2.50e-4 |
| 111 | 3.00e-2 | 2 | 2.07e-2 | 2.42e-2 | 2.25e-2 | 2.07e-2 | 107 | -1.46e-3 | +1.15e-3 | -1.54e-4 | -2.45e-4 |
| 112 | 3.00e-2 | 2 | 2.17e-2 | 3.39e-2 | 2.78e-2 | 3.39e-2 | 297 | +4.32e-4 | +1.49e-3 | +9.63e-4 | -1.04e-5 |
| 114 | 3.00e-2 | 1 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 383 | +4.71e-4 | +4.71e-4 | +4.71e-4 | +3.77e-5 |
| 115 | 3.00e-2 | 1 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 3.95e-2 | 352 | -7.46e-5 | -7.46e-5 | -7.46e-5 | +2.65e-5 |
| 116 | 3.00e-2 | 1 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 325 | -6.87e-5 | -6.87e-5 | -6.87e-5 | +1.70e-5 |
| 117 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 298 | -9.93e-5 | -9.93e-5 | -9.93e-5 | +5.37e-6 |
| 118 | 3.00e-2 | 1 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 286 | +1.41e-4 | +1.41e-4 | +1.41e-4 | +1.89e-5 |
| 119 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 262 | -9.91e-5 | -9.91e-5 | -9.91e-5 | +7.13e-6 |
| 120 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 286 | +2.03e-4 | +2.03e-4 | +2.03e-4 | +2.67e-5 |
| 121 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 284 | +6.82e-5 | +6.82e-5 | +6.82e-5 | +3.09e-5 |
| 122 | 3.00e-2 | 1 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 4.22e-2 | 300 | +8.68e-5 | +8.68e-5 | +8.68e-5 | +3.65e-5 |
| 124 | 3.00e-2 | 2 | 4.21e-2 | 4.61e-2 | 4.41e-2 | 4.21e-2 | 274 | -3.36e-4 | +2.66e-4 | -3.52e-5 | +1.99e-5 |
| 126 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 352 | +3.99e-4 | +3.99e-4 | +3.99e-4 | +5.77e-5 |
| 127 | 3.00e-2 | 1 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 311 | -1.31e-4 | -1.31e-4 | -1.31e-4 | +3.88e-5 |
| 128 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 305 | +6.27e-5 | +6.27e-5 | +6.27e-5 | +4.12e-5 |
| 129 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 287 | -6.03e-6 | -6.03e-6 | -6.03e-6 | +3.65e-5 |
| 130 | 3.00e-2 | 1 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 4.97e-2 | 312 | +1.56e-4 | +1.56e-4 | +1.56e-4 | +4.85e-5 |
| 131 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 285 | -1.67e-4 | -1.67e-4 | -1.67e-4 | +2.69e-5 |
| 132 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 270 | +7.02e-6 | +7.02e-6 | +7.02e-6 | +2.49e-5 |
| 133 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 280 | +8.25e-5 | +8.25e-5 | +8.25e-5 | +3.07e-5 |
| 134 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 257 | +1.69e-4 | +1.69e-4 | +1.69e-4 | +4.46e-5 |
| 135 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 290 | +5.91e-5 | +5.91e-5 | +5.91e-5 | +4.60e-5 |
| 136 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 287 | -7.87e-6 | -7.87e-6 | -7.87e-6 | +4.06e-5 |
| 137 | 3.00e-2 | 1 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 290 | +9.77e-5 | +9.77e-5 | +9.77e-5 | +4.63e-5 |
| 138 | 3.00e-2 | 1 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 291 | +4.25e-5 | +4.25e-5 | +4.25e-5 | +4.60e-5 |
| 139 | 3.00e-2 | 1 | 5.19e-2 | 5.19e-2 | 5.19e-2 | 5.19e-2 | 257 | -1.26e-4 | -1.26e-4 | -1.26e-4 | +2.88e-5 |
| 140 | 3.00e-2 | 1 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 259 | +1.26e-4 | +1.26e-4 | +1.26e-4 | +3.85e-5 |
| 141 | 3.00e-2 | 1 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 268 | +4.78e-5 | +4.78e-5 | +4.78e-5 | +3.95e-5 |
| 142 | 3.00e-2 | 1 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 268 | +1.00e-4 | +1.00e-4 | +1.00e-4 | +4.55e-5 |
| 143 | 3.00e-2 | 1 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 5.62e-2 | 267 | +2.66e-5 | +2.66e-5 | +2.66e-5 | +4.36e-5 |
| 144 | 3.00e-2 | 1 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 5.51e-2 | 268 | -7.06e-5 | -7.06e-5 | -7.06e-5 | +3.22e-5 |
| 145 | 3.00e-2 | 1 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 5.78e-2 | 274 | +1.69e-4 | +1.69e-4 | +1.69e-4 | +4.59e-5 |
| 146 | 3.00e-2 | 1 | 5.99e-2 | 5.99e-2 | 5.99e-2 | 5.99e-2 | 271 | +1.37e-4 | +1.37e-4 | +1.37e-4 | +5.50e-5 |
| 147 | 3.00e-2 | 2 | 5.38e-2 | 5.70e-2 | 5.54e-2 | 5.38e-2 | 237 | -2.50e-4 | -1.97e-4 | -2.23e-4 | +1.86e-6 |
| 149 | 3.00e-3 | 2 | 5.62e-2 | 6.31e-2 | 5.97e-2 | 5.62e-2 | 237 | -4.88e-4 | +5.29e-4 | +2.04e-5 | +2.99e-7 |
| 151 | 3.00e-3 | 2 | 5.13e-3 | 6.35e-3 | 5.74e-3 | 5.13e-3 | 237 | -6.73e-3 | -9.01e-4 | -3.82e-3 | -6.96e-4 |
| 152 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 255 | +3.37e-4 | +3.37e-4 | +3.37e-4 | -5.93e-4 |
| 153 | 3.00e-3 | 1 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 242 | -5.75e-5 | -5.75e-5 | -5.75e-5 | -5.39e-4 |
| 154 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 260 | +5.34e-5 | +5.34e-5 | +5.34e-5 | -4.80e-4 |
| 155 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 261 | -1.42e-4 | -1.42e-4 | -1.42e-4 | -4.46e-4 |
| 156 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 271 | +1.53e-4 | +1.53e-4 | +1.53e-4 | -3.86e-4 |
| 157 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 258 | -7.96e-6 | -7.96e-6 | -7.96e-6 | -3.48e-4 |
| 158 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 257 | +1.81e-5 | +1.81e-5 | +1.81e-5 | -3.12e-4 |
| 159 | 3.00e-3 | 2 | 5.01e-3 | 5.18e-3 | 5.10e-3 | 5.01e-3 | 208 | -3.72e-4 | -1.53e-4 | -2.63e-4 | -3.01e-4 |
| 160 | 3.00e-3 | 1 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 239 | +3.12e-4 | +3.12e-4 | +3.12e-4 | -2.40e-4 |
| 161 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 246 | +1.40e-4 | +1.40e-4 | +1.40e-4 | -2.02e-4 |
| 162 | 3.00e-3 | 1 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 252 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -1.71e-4 |
| 163 | 3.00e-3 | 2 | 4.97e-3 | 5.62e-3 | 5.30e-3 | 4.97e-3 | 193 | -6.37e-4 | -9.10e-5 | -3.64e-4 | -2.10e-4 |
| 165 | 3.00e-3 | 2 | 4.94e-3 | 5.90e-3 | 5.42e-3 | 4.94e-3 | 194 | -9.16e-4 | +6.05e-4 | -1.55e-4 | -2.07e-4 |
| 166 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 234 | +4.08e-4 | +4.08e-4 | +4.08e-4 | -1.46e-4 |
| 167 | 3.00e-3 | 1 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 233 | +9.50e-6 | +9.50e-6 | +9.50e-6 | -1.30e-4 |
| 168 | 3.00e-3 | 2 | 5.37e-3 | 5.79e-3 | 5.58e-3 | 5.37e-3 | 222 | -3.32e-4 | +2.31e-4 | -5.09e-5 | -1.18e-4 |
| 170 | 3.00e-3 | 2 | 5.05e-3 | 5.95e-3 | 5.50e-3 | 5.05e-3 | 200 | -8.19e-4 | +3.57e-4 | -2.31e-4 | -1.45e-4 |
| 171 | 3.00e-3 | 1 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 245 | +4.85e-4 | +4.85e-4 | +4.85e-4 | -8.24e-5 |
| 172 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 231 | +7.55e-5 | +7.55e-5 | +7.55e-5 | -6.66e-5 |
| 173 | 3.00e-3 | 1 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 5.69e-3 | 220 | -7.59e-5 | -7.59e-5 | -7.59e-5 | -6.75e-5 |
| 174 | 3.00e-3 | 2 | 5.10e-3 | 5.75e-3 | 5.42e-3 | 5.10e-3 | 173 | -6.88e-4 | +4.29e-5 | -3.22e-4 | -1.20e-4 |
| 175 | 3.00e-3 | 1 | 5.31e-3 | 5.31e-3 | 5.31e-3 | 5.31e-3 | 225 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -8.98e-5 |
| 176 | 3.00e-3 | 2 | 4.98e-3 | 5.44e-3 | 5.21e-3 | 4.98e-3 | 173 | -5.06e-4 | +1.09e-4 | -1.98e-4 | -1.14e-4 |
| 177 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 202 | +5.35e-4 | +5.35e-4 | +5.35e-4 | -4.86e-5 |
| 178 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 264 | +2.91e-4 | +2.91e-4 | +2.91e-4 | -1.47e-5 |
| 179 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 234 | -2.61e-4 | -2.61e-4 | -2.61e-4 | -3.93e-5 |
| 180 | 3.00e-3 | 2 | 5.03e-3 | 5.80e-3 | 5.42e-3 | 5.03e-3 | 183 | -7.84e-4 | +1.19e-4 | -3.33e-4 | -9.96e-5 |
| 181 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 212 | +5.25e-4 | +5.25e-4 | +5.25e-4 | -3.71e-5 |
| 182 | 3.00e-3 | 2 | 4.62e-3 | 5.08e-3 | 4.85e-3 | 4.62e-3 | 149 | -6.36e-4 | -5.40e-4 | -5.88e-4 | -1.42e-4 |
| 183 | 3.00e-3 | 1 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 201 | +8.12e-4 | +8.12e-4 | +8.12e-4 | -4.68e-5 |
| 184 | 3.00e-3 | 2 | 4.93e-3 | 5.77e-3 | 5.35e-3 | 4.93e-3 | 159 | -9.98e-4 | +2.94e-4 | -3.52e-4 | -1.11e-4 |
| 185 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 235 | +7.20e-4 | +7.20e-4 | +7.20e-4 | -2.81e-5 |
| 186 | 3.00e-3 | 2 | 4.77e-3 | 5.30e-3 | 5.04e-3 | 4.77e-3 | 149 | -7.10e-4 | -4.73e-4 | -5.92e-4 | -1.36e-4 |
| 187 | 3.00e-3 | 2 | 5.00e-3 | 5.34e-3 | 5.17e-3 | 5.00e-3 | 158 | -4.10e-4 | +5.95e-4 | +9.25e-5 | -9.79e-5 |
| 188 | 3.00e-3 | 1 | 5.11e-3 | 5.11e-3 | 5.11e-3 | 5.11e-3 | 181 | +1.21e-4 | +1.21e-4 | +1.21e-4 | -7.61e-5 |
| 189 | 3.00e-3 | 2 | 4.89e-3 | 5.38e-3 | 5.14e-3 | 4.89e-3 | 149 | -6.35e-4 | +2.52e-4 | -1.92e-4 | -1.02e-4 |
| 190 | 3.00e-3 | 1 | 5.02e-3 | 5.02e-3 | 5.02e-3 | 5.02e-3 | 169 | +1.56e-4 | +1.56e-4 | +1.56e-4 | -7.66e-5 |
| 191 | 3.00e-3 | 2 | 4.88e-3 | 5.26e-3 | 5.07e-3 | 4.88e-3 | 150 | -5.04e-4 | +2.68e-4 | -1.18e-4 | -8.83e-5 |
| 192 | 3.00e-3 | 2 | 4.83e-3 | 5.22e-3 | 5.02e-3 | 5.22e-3 | 177 | -7.48e-5 | +4.38e-4 | +1.82e-4 | -3.44e-5 |
| 193 | 3.00e-3 | 1 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 5.43e-3 | 201 | +2.00e-4 | +2.00e-4 | +2.00e-4 | -1.10e-5 |
| 194 | 3.00e-3 | 2 | 4.53e-3 | 5.30e-3 | 4.92e-3 | 4.53e-3 | 140 | -1.12e-3 | -1.36e-4 | -6.31e-4 | -1.34e-4 |
| 195 | 3.00e-3 | 2 | 5.10e-3 | 5.47e-3 | 5.28e-3 | 5.10e-3 | 154 | -4.54e-4 | +1.03e-3 | +2.90e-4 | -6.06e-5 |
| 196 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 172 | +4.14e-4 | +4.14e-4 | +4.14e-4 | -1.31e-5 |
| 197 | 3.00e-3 | 3 | 4.89e-3 | 5.30e-3 | 5.06e-3 | 4.99e-3 | 139 | -5.35e-4 | +1.42e-4 | -1.95e-4 | -5.91e-5 |
| 198 | 3.00e-3 | 1 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 178 | +4.85e-4 | +4.85e-4 | +4.85e-4 | -4.64e-6 |
| 199 | 3.00e-3 | 2 | 4.65e-3 | 5.20e-3 | 4.93e-3 | 4.65e-3 | 128 | -8.66e-4 | -2.52e-4 | -5.59e-4 | -1.13e-4 |

