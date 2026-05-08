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
| nccl-async | 0.054398 | 0.9176 | +0.0051 | 1874.1 | 619 | 38.8 | 100% | 100% | 6.5 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9176 | nccl-async | - | - |

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
| nccl-async | 2.0038 | 0.7585 | 0.5716 | 0.5058 | 0.4788 | 0.4608 | 0.4417 | 0.4363 | 0.4667 | 0.4712 | 0.2038 | 0.1659 | 0.1471 | 0.1350 | 0.1325 | 0.0728 | 0.0677 | 0.0606 | 0.0573 | 0.0544 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4043 | 2.8 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3023 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2934 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 394 | 392 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1873.0 | 1.2 | epoch-boundary(199) |
| nccl-async | gpu2 | 1873.2 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1873.1 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.4s |
| resnet-graph | nccl-async | gpu1 | 1.2s | 0.0s | 0.0s | 0.0s | 2.7s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 22 | 0 | 619 | 38.8 | 1315/10071 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 158.1 | 8.4% |

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
| resnet-graph | nccl-async | 187 | 619 | 0 | 6.69e-3 | -1.71e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 619 | 9.49e-2 | 4.21e-2 | 0.00e0 | 4.72e-1 | 49.4 | -7.38e-5 | 1.40e-3 |
| resnet-graph | nccl-async | 1 | 619 | 9.55e-2 | 4.36e-2 | 0.00e0 | 4.27e-1 | 33.8 | -7.25e-5 | 2.09e-3 |
| resnet-graph | nccl-async | 2 | 619 | 9.49e-2 | 4.32e-2 | 0.00e0 | 4.11e-1 | 16.8 | -6.93e-5 | 2.06e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9815 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9812 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9973 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 77 (0,1,2,3,5,7,8,9…142,148) | 0 (—) | — | 0,1,2,3,5,7,8,9…142,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 31 | 31 |
| resnet-graph | nccl-async | 0e0 | 5 | 13 | 13 |
| resnet-graph | nccl-async | 0e0 | 10 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 3 | 3 | 3 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 518 | +0.031 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 52 | +0.225 |
| resnet-graph | nccl-async | 3.00e-3 | 151–199 | 44 | +0.097 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 616 | +0.006 | 186 | +0.144 | +0.291 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 617 | 3.38e1–8.13e1 | 6.92e1 | 2.19e-3 | 3.67e-3 | 5.51e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 520 | 68–77502 | +6.024e-6 | 0.246 | +6.139e-6 | 0.259 | 99 | +1.086e-5 | 0.673 | 35–851 | +1.055e-3 | 0.419 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 507 | 935–77502 | +5.899e-6 | 0.287 | +5.971e-6 | 0.300 | 98 | +1.088e-5 | 0.667 | 71–851 | +1.043e-3 | 0.528 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 53 | 78192–117079 | +1.363e-5 | 0.129 | +1.352e-5 | 0.128 | 45 | +1.901e-5 | 0.313 | 606–1129 | +4.244e-5 | 0.000 |
| resnet-graph | nccl-async | 3.00e-3 | 151–199 | 45 | 118213–155775 | -7.948e-6 | 0.047 | -7.880e-6 | 0.046 | 43 | -8.310e-6 | 0.050 | 743–1134 | +2.700e-3 | 0.247 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.055e-3 | r0: +1.055e-3, r1: +1.060e-3, r2: +1.052e-3 | r0: 0.493, r1: 0.378, r2: 0.379 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.043e-3 | r0: +1.046e-3, r1: +1.046e-3, r2: +1.039e-3 | r0: 0.669, r1: 0.460, r2: 0.460 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +4.244e-5 | r0: +1.018e-4, r1: -3.256e-6, r2: +2.916e-5 | r0: 0.001, r1: 0.000, r2: 0.000 | 31.25× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-3 | 151–199 | +2.700e-3 | r0: +2.775e-3, r1: +2.640e-3, r2: +2.685e-3 | r0: 0.259, r1: 0.238, r2: 0.243 | 1.05× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇██████▄▄▄▄▅▅▅▅▅▅▆▃▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▆▇▇▇▇▇██▇▇▇██████▄▆▆▇▇██████▆▅▆▇▇▇▇▇███` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 0.00e0 | 4.72e-1 | 1.05e-1 | 7.43e-2 | 27 | -3.74e-2 | +7.13e-3 | -9.45e-3 | -4.85e-3 |
| 1 | 3.00e-1 | 11 | 6.81e-2 | 1.13e-1 | 8.02e-2 | 8.61e-2 | 31 | -1.30e-2 | +1.31e-2 | +6.34e-4 | -6.74e-4 |
| 2 | 3.00e-1 | 5 | 8.68e-2 | 1.19e-1 | 9.61e-2 | 9.21e-2 | 38 | -7.31e-3 | +9.17e-3 | +4.51e-4 | -2.67e-4 |
| 3 | 3.00e-1 | 7 | 9.54e-2 | 1.21e-1 | 1.00e-1 | 9.54e-2 | 32 | -4.30e-3 | +6.00e-3 | +3.99e-5 | -1.75e-4 |
| 4 | 3.00e-1 | 8 | 9.31e-2 | 1.28e-1 | 1.01e-1 | 9.31e-2 | 32 | -6.70e-3 | +9.26e-3 | -6.94e-5 | -2.20e-4 |
| 5 | 3.00e-1 | 8 | 8.66e-2 | 1.32e-1 | 9.58e-2 | 9.40e-2 | 30 | -1.39e-2 | +1.17e-2 | +3.31e-5 | -8.05e-5 |
| 6 | 3.00e-1 | 7 | 9.31e-2 | 1.26e-1 | 1.03e-1 | 1.05e-1 | 40 | -7.62e-3 | +9.57e-3 | +5.31e-4 | +2.02e-4 |
| 7 | 3.00e-1 | 7 | 9.46e-2 | 1.37e-1 | 1.06e-1 | 9.46e-2 | 38 | -7.92e-3 | +6.84e-3 | -4.27e-4 | -2.03e-4 |
| 8 | 3.00e-1 | 7 | 9.90e-2 | 1.34e-1 | 1.06e-1 | 1.02e-1 | 40 | -7.91e-3 | +7.64e-3 | +2.00e-4 | -6.23e-5 |
| 9 | 3.00e-1 | 7 | 9.23e-2 | 1.38e-1 | 1.05e-1 | 9.76e-2 | 34 | -6.68e-3 | +7.32e-3 | -1.75e-4 | -1.62e-4 |
| 10 | 3.00e-1 | 7 | 9.00e-2 | 1.28e-1 | 1.03e-1 | 1.07e-1 | 37 | -1.05e-2 | +8.29e-3 | +2.86e-4 | +7.87e-5 |
| 11 | 3.00e-1 | 7 | 8.81e-2 | 1.31e-1 | 9.96e-2 | 1.03e-1 | 38 | -1.38e-2 | +8.96e-3 | -2.24e-4 | -1.15e-6 |
| 12 | 3.00e-1 | 7 | 9.26e-2 | 1.37e-1 | 1.02e-1 | 9.84e-2 | 38 | -1.09e-2 | +9.12e-3 | -1.05e-4 | -7.39e-5 |
| 13 | 3.00e-1 | 6 | 9.70e-2 | 1.34e-1 | 1.06e-1 | 1.02e-1 | 38 | -8.52e-3 | +7.52e-3 | +1.19e-4 | -2.20e-5 |
| 14 | 3.00e-1 | 8 | 8.97e-2 | 1.37e-1 | 9.91e-2 | 9.90e-2 | 39 | -1.33e-2 | +7.98e-3 | -3.79e-4 | -1.44e-4 |
| 15 | 3.00e-1 | 6 | 9.32e-2 | 1.31e-1 | 1.06e-1 | 9.32e-2 | 34 | -5.01e-3 | +5.90e-3 | -2.98e-4 | -3.42e-4 |
| 16 | 3.00e-1 | 8 | 8.62e-2 | 1.33e-1 | 9.85e-2 | 1.01e-1 | 44 | -1.27e-2 | +9.52e-3 | +2.20e-4 | -6.63e-6 |
| 17 | 3.00e-1 | 9 | 8.38e-2 | 1.35e-1 | 9.88e-2 | 9.65e-2 | 39 | -1.06e-2 | +5.09e-3 | -3.93e-4 | -1.59e-4 |
| 18 | 3.00e-1 | 5 | 9.72e-2 | 1.41e-1 | 1.07e-1 | 9.79e-2 | 34 | -9.84e-3 | +8.74e-3 | -9.69e-5 | -2.16e-4 |
| 19 | 3.00e-1 | 8 | 8.86e-2 | 1.34e-1 | 9.91e-2 | 9.30e-2 | 40 | -1.01e-2 | +9.97e-3 | -1.23e-4 | -2.18e-4 |
| 20 | 3.00e-1 | 5 | 9.57e-2 | 1.37e-1 | 1.10e-1 | 1.06e-1 | 40 | -9.04e-3 | +6.39e-3 | +3.34e-4 | -4.15e-5 |
| 21 | 3.00e-1 | 7 | 9.44e-2 | 1.33e-1 | 1.06e-1 | 1.03e-1 | 44 | -8.58e-3 | +7.01e-3 | -3.53e-5 | -6.53e-5 |
| 22 | 3.00e-1 | 9 | 9.39e-2 | 1.39e-1 | 1.07e-1 | 1.06e-1 | 35 | -5.56e-3 | +5.10e-3 | -6.12e-5 | -1.56e-7 |
| 23 | 3.00e-1 | 4 | 9.14e-2 | 1.40e-1 | 1.05e-1 | 9.17e-2 | 38 | -1.29e-2 | +1.05e-2 | -7.99e-4 | -3.66e-4 |
| 24 | 3.00e-1 | 7 | 8.35e-2 | 1.37e-1 | 9.81e-2 | 1.02e-1 | 40 | -1.65e-2 | +1.15e-2 | +1.90e-4 | -3.70e-5 |
| 25 | 3.00e-1 | 7 | 9.46e-2 | 1.40e-1 | 1.05e-1 | 1.01e-1 | 39 | -7.89e-3 | +7.36e-3 | -1.93e-4 | -1.25e-4 |
| 26 | 3.00e-1 | 6 | 1.00e-1 | 1.41e-1 | 1.09e-1 | 1.04e-1 | 40 | -7.54e-3 | +8.14e-3 | +1.48e-4 | -5.30e-5 |
| 27 | 3.00e-1 | 6 | 9.94e-2 | 1.41e-1 | 1.08e-1 | 1.06e-1 | 39 | -8.64e-3 | +8.02e-3 | +4.57e-5 | -2.30e-5 |
| 28 | 3.00e-1 | 6 | 1.01e-1 | 1.33e-1 | 1.09e-1 | 1.07e-1 | 44 | -5.89e-3 | +5.93e-3 | +4.87e-5 | -1.35e-5 |
| 29 | 3.00e-1 | 7 | 9.64e-2 | 1.39e-1 | 1.07e-1 | 1.01e-1 | 34 | -7.96e-3 | +6.17e-3 | -2.32e-4 | -1.49e-4 |
| 30 | 3.00e-1 | 7 | 8.98e-2 | 1.41e-1 | 1.04e-1 | 1.06e-1 | 41 | -1.32e-2 | +1.17e-2 | +2.62e-4 | +5.23e-5 |
| 31 | 3.00e-1 | 8 | 9.69e-2 | 1.38e-1 | 1.08e-1 | 9.69e-2 | 37 | -7.37e-3 | +6.34e-3 | -2.19e-4 | -1.83e-4 |
| 32 | 3.00e-1 | 5 | 8.96e-2 | 1.41e-1 | 1.04e-1 | 9.36e-2 | 37 | -1.33e-2 | +1.03e-2 | -2.59e-4 | -2.97e-4 |
| 33 | 3.00e-1 | 10 | 8.95e-2 | 1.40e-1 | 1.03e-1 | 8.95e-2 | 30 | -7.09e-3 | +8.22e-3 | -3.11e-4 | -4.62e-4 |
| 34 | 3.00e-1 | 4 | 8.76e-2 | 1.50e-1 | 1.05e-1 | 8.76e-2 | 34 | -1.53e-2 | +1.49e-2 | -6.20e-4 | -6.74e-4 |
| 35 | 3.00e-1 | 7 | 8.42e-2 | 1.47e-1 | 1.00e-1 | 9.26e-2 | 34 | -1.40e-2 | +1.26e-2 | +3.32e-5 | -3.98e-4 |
| 36 | 3.00e-1 | 6 | 9.76e-2 | 1.34e-1 | 1.09e-1 | 1.08e-1 | 43 | -6.58e-3 | +8.08e-3 | +6.07e-4 | +5.19e-6 |
| 37 | 3.00e-1 | 7 | 9.92e-2 | 1.40e-1 | 1.09e-1 | 1.09e-1 | 40 | -8.90e-3 | +7.14e-3 | +3.23e-5 | +2.75e-5 |
| 38 | 3.00e-1 | 6 | 9.26e-2 | 1.41e-1 | 1.06e-1 | 9.96e-2 | 39 | -1.19e-2 | +7.60e-3 | -4.95e-4 | -2.20e-4 |
| 39 | 3.00e-1 | 7 | 8.92e-2 | 1.43e-1 | 1.08e-1 | 8.92e-2 | 33 | -7.23e-3 | +7.22e-3 | -6.25e-4 | -6.14e-4 |
| 40 | 3.00e-1 | 7 | 9.38e-2 | 1.39e-1 | 1.07e-1 | 1.13e-1 | 39 | -1.08e-2 | +9.53e-3 | +4.52e-4 | -8.12e-5 |
| 41 | 3.00e-1 | 6 | 8.83e-2 | 1.53e-1 | 1.07e-1 | 1.05e-1 | 41 | -1.48e-2 | +1.33e-2 | +2.13e-4 | +3.77e-5 |
| 42 | 3.00e-1 | 8 | 9.48e-2 | 1.49e-1 | 1.11e-1 | 1.03e-1 | 39 | -7.97e-3 | +7.60e-3 | -1.70e-4 | -1.28e-4 |
| 43 | 3.00e-1 | 4 | 9.93e-2 | 1.44e-1 | 1.14e-1 | 1.08e-1 | 42 | -8.83e-3 | +8.74e-3 | +4.91e-4 | +3.31e-5 |
| 44 | 3.00e-1 | 6 | 9.98e-2 | 1.44e-1 | 1.10e-1 | 1.04e-1 | 41 | -8.99e-3 | +8.12e-3 | -8.06e-5 | -6.07e-5 |
| 45 | 3.00e-1 | 6 | 1.02e-1 | 1.40e-1 | 1.11e-1 | 1.06e-1 | 44 | -7.22e-3 | +7.54e-3 | +1.87e-4 | +3.64e-7 |
| 46 | 3.00e-1 | 6 | 1.03e-1 | 1.47e-1 | 1.14e-1 | 1.09e-1 | 44 | -7.99e-3 | +6.57e-3 | +2.97e-5 | -3.23e-5 |
| 47 | 3.00e-1 | 6 | 1.00e-1 | 1.44e-1 | 1.11e-1 | 1.00e-1 | 37 | -8.25e-3 | +6.15e-3 | -4.70e-4 | -3.03e-4 |
| 48 | 3.00e-1 | 9 | 9.62e-2 | 1.47e-1 | 1.07e-1 | 1.04e-1 | 35 | -1.14e-2 | +1.11e-2 | +1.21e-4 | -1.21e-4 |
| 49 | 3.00e-1 | 4 | 9.03e-2 | 1.48e-1 | 1.08e-1 | 9.03e-2 | 37 | -1.31e-2 | +1.17e-2 | -6.78e-4 | -4.32e-4 |
| 50 | 3.00e-1 | 7 | 8.99e-2 | 1.43e-1 | 1.06e-1 | 1.07e-1 | 37 | -1.25e-2 | +9.69e-3 | +4.79e-4 | +1.21e-5 |
| 51 | 3.00e-1 | 7 | 9.62e-2 | 1.38e-1 | 1.07e-1 | 1.10e-1 | 44 | -9.65e-3 | +8.49e-3 | +1.50e-4 | +9.74e-5 |
| 52 | 3.00e-1 | 5 | 9.95e-2 | 1.53e-1 | 1.15e-1 | 1.09e-1 | 43 | -1.07e-2 | +8.73e-3 | -1.55e-5 | +5.36e-6 |
| 53 | 3.00e-1 | 7 | 8.77e-2 | 1.40e-1 | 1.04e-1 | 1.05e-1 | 39 | -1.33e-2 | +7.73e-3 | -2.64e-4 | -7.32e-5 |
| 54 | 3.00e-1 | 6 | 9.62e-2 | 1.46e-1 | 1.11e-1 | 9.84e-2 | 39 | -7.39e-3 | +9.89e-3 | -5.90e-5 | -1.88e-4 |
| 55 | 3.00e-1 | 9 | 9.20e-2 | 1.57e-1 | 1.09e-1 | 9.20e-2 | 38 | -7.16e-3 | +8.64e-3 | -4.30e-4 | -4.91e-4 |
| 56 | 3.00e-1 | 4 | 9.49e-2 | 1.45e-1 | 1.15e-1 | 1.17e-1 | 46 | -8.17e-3 | +8.67e-3 | +1.64e-3 | +2.09e-4 |
| 57 | 3.00e-1 | 6 | 9.45e-2 | 1.46e-1 | 1.10e-1 | 9.86e-2 | 38 | -9.44e-3 | +7.34e-3 | -4.00e-4 | -1.29e-4 |
| 58 | 3.00e-1 | 6 | 1.00e-1 | 1.43e-1 | 1.12e-1 | 1.11e-1 | 44 | -6.99e-3 | +7.48e-3 | +2.64e-4 | +1.75e-5 |
| 59 | 3.00e-1 | 9 | 1.00e-1 | 1.40e-1 | 1.12e-1 | 1.14e-1 | 38 | -7.22e-3 | +4.86e-3 | +2.62e-5 | +6.71e-5 |
| 60 | 3.00e-1 | 4 | 9.16e-2 | 1.47e-1 | 1.13e-1 | 1.09e-1 | 40 | -1.18e-2 | +9.63e-3 | +2.88e-4 | +1.19e-4 |
| 61 | 3.00e-1 | 6 | 9.31e-2 | 1.43e-1 | 1.10e-1 | 1.06e-1 | 41 | -1.08e-2 | +8.15e-3 | +1.01e-4 | +7.91e-5 |
| 62 | 3.00e-1 | 6 | 9.68e-2 | 1.51e-1 | 1.11e-1 | 1.07e-1 | 40 | -9.45e-3 | +8.94e-3 | -5.53e-5 | +5.61e-7 |
| 63 | 3.00e-1 | 6 | 1.02e-1 | 1.39e-1 | 1.13e-1 | 1.10e-1 | 49 | -6.45e-3 | +7.33e-3 | +2.44e-4 | +6.24e-5 |
| 64 | 3.00e-1 | 5 | 1.05e-1 | 1.56e-1 | 1.20e-1 | 1.07e-1 | 34 | -8.56e-3 | +6.49e-3 | -2.43e-4 | -1.36e-4 |
| 65 | 3.00e-1 | 8 | 9.39e-2 | 1.59e-1 | 1.07e-1 | 1.04e-1 | 41 | -1.41e-2 | +1.35e-2 | +4.75e-5 | -6.14e-5 |
| 66 | 3.00e-1 | 5 | 1.01e-1 | 1.45e-1 | 1.13e-1 | 1.12e-1 | 44 | -8.36e-3 | +8.79e-3 | +4.02e-4 | +9.94e-5 |
| 67 | 3.00e-1 | 5 | 1.08e-1 | 1.48e-1 | 1.19e-1 | 1.08e-1 | 41 | -6.54e-3 | +5.93e-3 | -2.19e-4 | -9.16e-5 |
| 68 | 3.00e-1 | 8 | 1.05e-1 | 1.42e-1 | 1.13e-1 | 1.08e-1 | 39 | -6.11e-3 | +7.18e-3 | +7.50e-6 | -8.36e-5 |
| 69 | 3.00e-1 | 4 | 1.03e-1 | 1.39e-1 | 1.15e-1 | 1.12e-1 | 47 | -6.48e-3 | +7.41e-3 | +4.19e-4 | +5.27e-5 |
| 70 | 3.00e-1 | 6 | 1.07e-1 | 1.52e-1 | 1.20e-1 | 1.07e-1 | 42 | -5.63e-3 | +5.46e-3 | -2.39e-4 | -1.75e-4 |
| 71 | 3.00e-1 | 6 | 9.48e-2 | 1.49e-1 | 1.10e-1 | 1.02e-1 | 40 | -1.22e-2 | +7.73e-3 | -3.04e-4 | -2.73e-4 |
| 72 | 3.00e-1 | 6 | 9.90e-2 | 1.47e-1 | 1.11e-1 | 1.09e-1 | 39 | -9.91e-3 | +7.56e-3 | +2.11e-5 | -1.44e-4 |
| 73 | 3.00e-1 | 7 | 9.83e-2 | 1.57e-1 | 1.12e-1 | 1.09e-1 | 44 | -1.11e-2 | +9.94e-3 | -7.22e-5 | -1.22e-4 |
| 74 | 3.00e-1 | 2 | 1.11e-1 | 1.13e-1 | 1.12e-1 | 1.11e-1 | 255 | -4.45e-5 | +6.62e-4 | +3.09e-4 | -4.36e-5 |
| 76 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 320 | +2.25e-3 | +2.25e-3 | +2.25e-3 | +1.86e-4 |
| 77 | 3.00e-1 | 2 | 2.07e-1 | 2.24e-1 | 2.16e-1 | 2.07e-1 | 228 | -3.44e-4 | -7.10e-5 | -2.08e-4 | +1.10e-4 |
| 78 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 272 | -2.16e-4 | -2.16e-4 | -2.16e-4 | +7.70e-5 |
| 79 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 241 | +2.83e-4 | +2.83e-4 | +2.83e-4 | +9.75e-5 |
| 80 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 254 | -1.22e-4 | -1.22e-4 | -1.22e-4 | +7.56e-5 |
| 81 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 249 | +2.69e-5 | +2.69e-5 | +2.69e-5 | +7.07e-5 |
| 82 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 254 | -5.15e-5 | -5.15e-5 | -5.15e-5 | +5.85e-5 |
| 83 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 270 | +2.37e-5 | +2.37e-5 | +2.37e-5 | +5.50e-5 |
| 84 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 253 | +7.27e-5 | +7.27e-5 | +7.27e-5 | +5.68e-5 |
| 85 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 265 | -7.36e-5 | -7.36e-5 | -7.36e-5 | +4.37e-5 |
| 86 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 260 | +1.05e-4 | +1.05e-4 | +1.05e-4 | +4.98e-5 |
| 87 | 3.00e-1 | 2 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 216 | -4.71e-5 | +6.37e-7 | -2.32e-5 | +3.62e-5 |
| 88 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 248 | -3.39e-4 | -3.39e-4 | -3.39e-4 | -1.31e-6 |
| 89 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 265 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +1.93e-5 |
| 90 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 277 | +9.29e-5 | +9.29e-5 | +9.29e-5 | +2.66e-5 |
| 91 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 284 | +1.09e-5 | +1.09e-5 | +1.09e-5 | +2.51e-5 |
| 92 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 238 | +1.21e-4 | +1.21e-4 | +1.21e-4 | +3.47e-5 |
| 93 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 238 | -2.23e-4 | -2.23e-4 | -2.23e-4 | +8.94e-6 |
| 94 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 265 | -2.39e-5 | -2.39e-5 | -2.39e-5 | +5.65e-6 |
| 95 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 265 | +9.07e-5 | +9.07e-5 | +9.07e-5 | +1.42e-5 |
| 96 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 243 | +3.64e-5 | +3.64e-5 | +3.64e-5 | +1.64e-5 |
| 97 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 290 | -7.42e-5 | -7.42e-5 | -7.42e-5 | +7.32e-6 |
| 98 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 237 | +2.01e-4 | +2.01e-4 | +2.01e-4 | +2.67e-5 |
| 99 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 263 | -2.09e-4 | -2.09e-4 | -2.09e-4 | +3.10e-6 |
| 100 | 3.00e-2 | 2 | 1.98e-1 | 2.08e-1 | 2.03e-1 | 1.98e-1 | 227 | -2.16e-4 | +1.57e-4 | -2.96e-5 | -4.98e-6 |
| 101 | 3.00e-2 | 1 | 1.93e-2 | 1.93e-2 | 1.93e-2 | 1.93e-2 | 258 | -9.02e-3 | -9.02e-3 | -9.02e-3 | -9.07e-4 |
| 102 | 3.00e-2 | 1 | 2.13e-2 | 2.13e-2 | 2.13e-2 | 2.13e-2 | 285 | +3.55e-4 | +3.55e-4 | +3.55e-4 | -7.81e-4 |
| 103 | 3.00e-2 | 1 | 2.31e-2 | 2.31e-2 | 2.31e-2 | 2.31e-2 | 260 | +3.10e-4 | +3.10e-4 | +3.10e-4 | -6.72e-4 |
| 104 | 3.00e-2 | 1 | 2.42e-2 | 2.42e-2 | 2.42e-2 | 2.42e-2 | 272 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -5.88e-4 |
| 105 | 3.00e-2 | 1 | 2.47e-2 | 2.47e-2 | 2.47e-2 | 2.47e-2 | 286 | +7.19e-5 | +7.19e-5 | +7.19e-5 | -5.22e-4 |
| 106 | 3.00e-2 | 1 | 2.66e-2 | 2.66e-2 | 2.66e-2 | 2.66e-2 | 265 | +2.77e-4 | +2.77e-4 | +2.77e-4 | -4.42e-4 |
| 107 | 3.00e-2 | 1 | 2.63e-2 | 2.63e-2 | 2.63e-2 | 2.63e-2 | 257 | -3.88e-5 | -3.88e-5 | -3.88e-5 | -4.01e-4 |
| 108 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 285 | +2.42e-5 | +2.42e-5 | +2.42e-5 | -3.59e-4 |
| 109 | 3.00e-2 | 1 | 2.91e-2 | 2.91e-2 | 2.91e-2 | 2.91e-2 | 239 | +3.97e-4 | +3.97e-4 | +3.97e-4 | -2.83e-4 |
| 110 | 3.00e-2 | 1 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 238 | -4.16e-4 | -4.16e-4 | -4.16e-4 | -2.97e-4 |
| 111 | 3.00e-2 | 1 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 2.82e-2 | 272 | +2.38e-4 | +2.38e-4 | +2.38e-4 | -2.43e-4 |
| 112 | 3.00e-2 | 1 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 272 | +2.36e-4 | +2.36e-4 | +2.36e-4 | -1.95e-4 |
| 113 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 284 | +6.92e-5 | +6.92e-5 | +6.92e-5 | -1.69e-4 |
| 114 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 300 | +1.11e-4 | +1.11e-4 | +1.11e-4 | -1.41e-4 |
| 115 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 281 | +1.78e-4 | +1.78e-4 | +1.78e-4 | -1.09e-4 |
| 116 | 3.00e-2 | 2 | 3.22e-2 | 3.24e-2 | 3.23e-2 | 3.24e-2 | 223 | -1.35e-4 | +2.62e-5 | -5.45e-5 | -9.77e-5 |
| 118 | 3.00e-2 | 2 | 3.22e-2 | 3.66e-2 | 3.44e-2 | 3.66e-2 | 223 | -2.57e-5 | +5.79e-4 | +2.77e-4 | -2.35e-5 |
| 119 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 286 | -4.24e-4 | -4.24e-4 | -4.24e-4 | -6.35e-5 |
| 120 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 261 | +4.48e-4 | +4.48e-4 | +4.48e-4 | -1.24e-5 |
| 121 | 3.00e-2 | 1 | 3.63e-2 | 3.63e-2 | 3.63e-2 | 3.63e-2 | 249 | -1.16e-5 | -1.16e-5 | -1.16e-5 | -1.23e-5 |
| 122 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 259 | +5.59e-6 | +5.59e-6 | +5.59e-6 | -1.05e-5 |
| 123 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 251 | +1.18e-4 | +1.18e-4 | +1.18e-4 | +2.31e-6 |
| 124 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 300 | +5.01e-5 | +5.01e-5 | +5.01e-5 | +7.08e-6 |
| 125 | 3.00e-2 | 1 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 250 | +3.29e-4 | +3.29e-4 | +3.29e-4 | +3.93e-5 |
| 126 | 3.00e-2 | 1 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 3.86e-2 | 262 | -2.59e-4 | -2.59e-4 | -2.59e-4 | +9.40e-6 |
| 127 | 3.00e-2 | 1 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 4.05e-2 | 252 | +1.93e-4 | +1.93e-4 | +1.93e-4 | +2.78e-5 |
| 128 | 3.00e-2 | 2 | 4.03e-2 | 4.07e-2 | 4.05e-2 | 4.07e-2 | 221 | -2.48e-5 | +4.13e-5 | +8.28e-6 | +2.44e-5 |
| 130 | 3.00e-2 | 2 | 4.05e-2 | 4.66e-2 | 4.35e-2 | 4.66e-2 | 234 | -1.43e-5 | +6.00e-4 | +2.93e-4 | +7.85e-5 |
| 132 | 3.00e-2 | 2 | 4.16e-2 | 4.79e-2 | 4.47e-2 | 4.79e-2 | 234 | -3.66e-4 | +5.96e-4 | +1.15e-4 | +9.03e-5 |
| 133 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 271 | -4.58e-4 | -4.58e-4 | -4.58e-4 | +3.54e-5 |
| 134 | 3.00e-2 | 1 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 4.61e-2 | 248 | +3.49e-4 | +3.49e-4 | +3.49e-4 | +6.68e-5 |
| 135 | 3.00e-2 | 1 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 242 | -1.19e-5 | -1.19e-5 | -1.19e-5 | +5.89e-5 |
| 136 | 3.00e-2 | 1 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 270 | +4.34e-5 | +4.34e-5 | +4.34e-5 | +5.74e-5 |
| 137 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 277 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +6.71e-5 |
| 138 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 254 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +7.40e-5 |
| 139 | 3.00e-2 | 1 | 4.86e-2 | 4.86e-2 | 4.86e-2 | 4.86e-2 | 255 | -1.31e-4 | -1.31e-4 | -1.31e-4 | +5.35e-5 |
| 140 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 240 | +1.00e-4 | +1.00e-4 | +1.00e-4 | +5.82e-5 |
| 141 | 3.00e-2 | 1 | 5.09e-2 | 5.09e-2 | 5.09e-2 | 5.09e-2 | 259 | +8.84e-5 | +8.84e-5 | +8.84e-5 | +6.12e-5 |
| 142 | 3.00e-2 | 2 | 5.10e-2 | 5.11e-2 | 5.11e-2 | 5.11e-2 | 234 | +6.74e-6 | +9.85e-6 | +8.30e-6 | +5.11e-5 |
| 144 | 3.00e-2 | 1 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 280 | -1.39e-4 | -1.39e-4 | -1.39e-4 | +3.21e-5 |
| 145 | 3.00e-2 | 2 | 5.13e-2 | 5.60e-2 | 5.36e-2 | 5.13e-2 | 260 | -3.42e-4 | +5.30e-4 | +9.40e-5 | +3.95e-5 |
| 147 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 422 | +9.04e-5 | +9.04e-5 | +9.04e-5 | +4.46e-5 |
| 148 | 3.00e-2 | 1 | 6.63e-2 | 6.63e-2 | 6.63e-2 | 6.63e-2 | 384 | +5.69e-4 | +5.69e-4 | +5.69e-4 | +9.71e-5 |
| 149 | 3.00e-2 | 1 | 6.52e-2 | 6.52e-2 | 6.52e-2 | 6.52e-2 | 318 | -5.07e-5 | -5.07e-5 | -5.07e-5 | +8.23e-5 |
| 151 | 3.00e-3 | 1 | 6.17e-2 | 6.17e-2 | 6.17e-2 | 6.17e-2 | 430 | -1.30e-4 | -1.30e-4 | -1.30e-4 | +6.11e-5 |
| 152 | 3.00e-3 | 1 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 2.83e-2 | 316 | -2.47e-3 | -2.47e-3 | -2.47e-3 | -1.92e-4 |
| 153 | 3.00e-3 | 1 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 5.79e-3 | 273 | -5.81e-3 | -5.81e-3 | -5.81e-3 | -7.54e-4 |
| 154 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 278 | -2.07e-4 | -2.07e-4 | -2.07e-4 | -6.99e-4 |
| 155 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 295 | +4.66e-5 | +4.66e-5 | +4.66e-5 | -6.24e-4 |
| 156 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 292 | +7.10e-5 | +7.10e-5 | +7.10e-5 | -5.55e-4 |
| 157 | 3.00e-3 | 1 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 316 | +1.13e-5 | +1.13e-5 | +1.13e-5 | -4.98e-4 |
| 158 | 3.00e-3 | 1 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 5.86e-3 | 297 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -4.38e-4 |
| 159 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 303 | -1.30e-4 | -1.30e-4 | -1.30e-4 | -4.07e-4 |
| 160 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 283 | +2.00e-4 | +2.00e-4 | +2.00e-4 | -3.46e-4 |
| 161 | 3.00e-3 | 1 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 5.55e-3 | 325 | -2.19e-4 | -2.19e-4 | -2.19e-4 | -3.34e-4 |
| 162 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 262 | +4.10e-4 | +4.10e-4 | +4.10e-4 | -2.59e-4 |
| 164 | 3.00e-3 | 2 | 5.65e-3 | 6.20e-3 | 5.93e-3 | 6.20e-3 | 262 | -2.67e-4 | +3.56e-4 | +4.41e-5 | -1.99e-4 |
| 166 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 330 | -2.50e-4 | -2.50e-4 | -2.50e-4 | -2.04e-4 |
| 167 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 304 | +2.95e-4 | +2.95e-4 | +2.95e-4 | -1.54e-4 |
| 168 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 319 | -5.70e-5 | -5.70e-5 | -5.70e-5 | -1.44e-4 |
| 169 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 342 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -1.16e-4 |
| 170 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 307 | +5.98e-5 | +5.98e-5 | +5.98e-5 | -9.87e-5 |
| 171 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 287 | -7.72e-5 | -7.72e-5 | -7.72e-5 | -9.65e-5 |
| 172 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 306 | -2.24e-4 | -2.24e-4 | -2.24e-4 | -1.09e-4 |
| 173 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 253 | +1.63e-4 | +1.63e-4 | +1.63e-4 | -8.21e-5 |
| 174 | 3.00e-3 | 1 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 5.73e-3 | 285 | -2.90e-4 | -2.90e-4 | -2.90e-4 | -1.03e-4 |
| 176 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 353 | +1.26e-4 | +1.26e-4 | +1.26e-4 | -8.00e-5 |
| 177 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 304 | +4.72e-4 | +4.72e-4 | +4.72e-4 | -2.49e-5 |
| 178 | 3.00e-3 | 1 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 6.23e-3 | 304 | -3.44e-4 | -3.44e-4 | -3.44e-4 | -5.67e-5 |
| 179 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 335 | +6.72e-5 | +6.72e-5 | +6.72e-5 | -4.43e-5 |
| 180 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 318 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -2.25e-5 |
| 181 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 277 | -6.71e-5 | -6.71e-5 | -6.71e-5 | -2.69e-5 |
| 182 | 3.00e-3 | 1 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 6.18e-3 | 270 | -2.52e-4 | -2.52e-4 | -2.52e-4 | -4.94e-5 |
| 183 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 303 | -6.51e-5 | -6.51e-5 | -6.51e-5 | -5.10e-5 |
| 184 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 284 | +1.90e-4 | +1.90e-4 | +1.90e-4 | -2.69e-5 |
| 185 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 286 | -5.92e-5 | -5.92e-5 | -5.92e-5 | -3.01e-5 |
| 187 | 3.00e-3 | 2 | 6.48e-3 | 6.84e-3 | 6.66e-3 | 6.84e-3 | 282 | +8.26e-5 | +1.88e-4 | +1.35e-4 | +1.81e-6 |
| 189 | 3.00e-3 | 1 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 348 | -1.84e-4 | -1.84e-4 | -1.84e-4 | -1.68e-5 |
| 190 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 310 | +2.11e-4 | +2.11e-4 | +2.11e-4 | +6.00e-6 |
| 191 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 300 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -6.11e-6 |
| 192 | 3.00e-3 | 1 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 6.76e-3 | 288 | +7.88e-5 | +7.88e-5 | +7.88e-5 | +2.38e-6 |
| 193 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 297 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -8.51e-6 |
| 194 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 349 | -4.52e-5 | -4.52e-5 | -4.52e-5 | -1.22e-5 |
| 196 | 3.00e-3 | 1 | 7.03e-3 | 7.03e-3 | 7.03e-3 | 7.03e-3 | 320 | +2.68e-4 | +2.68e-4 | +2.68e-4 | +1.59e-5 |
| 197 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 269 | -1.40e-4 | -1.40e-4 | -1.40e-4 | +2.75e-7 |
| 198 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 298 | -1.59e-4 | -1.59e-4 | -1.59e-4 | -1.57e-5 |
| 199 | 3.00e-3 | 1 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 292 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -1.71e-6 |

