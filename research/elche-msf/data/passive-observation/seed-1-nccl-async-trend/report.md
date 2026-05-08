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
| nccl-async | 0.059584 | 0.9167 | +0.0042 | 1934.5 | 543 | 41.9 | 100% | 100% | 8.4 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9167 | nccl-async | - | - |

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
| nccl-async | 1.9880 | 0.6902 | 0.6160 | 0.5597 | 0.5448 | 0.5186 | 0.4766 | 0.4810 | 0.4628 | 0.4684 | 0.2089 | 0.1676 | 0.1413 | 0.1534 | 0.1440 | 0.0800 | 0.0721 | 0.0687 | 0.0641 | 0.0596 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4002 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3045 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2953 | 3.2 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 410 | 407 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1932.2 | 2.3 | epoch-boundary(199) |
| nccl-async | gpu2 | 1932.3 | 2.2 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 1.3s |
| resnet-graph | nccl-async | gpu1 | 2.3s | 0.0s | 0.0s | 0.0s | 4.0s |
| resnet-graph | nccl-async | gpu2 | 2.2s | 0.0s | 0.0s | 0.0s | 3.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 379 | 0 | 543 | 41.9 | 2462/9351 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 184.4 | 9.5% |

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
| resnet-graph | nccl-async | 190 | 543 | 0 | 3.19e-3 | -7.36e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 543 | 6.67e-2 | 6.75e-2 | 0.00e0 | 3.81e-1 | 40.1 | -1.43e-4 | 2.72e-3 |
| resnet-graph | nccl-async | 1 | 543 | 6.79e-2 | 7.00e-2 | 0.00e0 | 4.79e-1 | 41.4 | -1.44e-4 | 4.03e-3 |
| resnet-graph | nccl-async | 2 | 543 | 6.66e-2 | 6.89e-2 | 0.00e0 | 3.86e-1 | 18.4 | -1.44e-4 | 4.05e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9964 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9975 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9975 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 34 (1,2,3,4,5,7,8,9…145,148) | 0 (—) | — | 1,2,3,4,5,7,8,9…145,148 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 20 | 20 |
| resnet-graph | nccl-async | 0e0 | 5 | 9 | 9 |
| resnet-graph | nccl-async | 0e0 | 10 | 2 | 2 |
| resnet-graph | nccl-async | 1e-4 | 3 | 10 | 10 |
| resnet-graph | nccl-async | 1e-4 | 5 | 4 | 4 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 270 | +0.052 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 162 | +0.035 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 106 | +0.014 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 540 | -0.003 | 189 | +0.224 | +0.324 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 541 | 3.37e1–7.87e1 | 6.26e1 | 2.22e-3 | 3.73e-3 | 4.65e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 272 | 66–78302 | +1.525e-5 | 0.570 | +1.575e-5 | 0.582 | 92 | +7.074e-6 | 0.226 | 31–1063 | +1.562e-3 | 0.792 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 259 | 890–78302 | +1.539e-5 | 0.609 | +1.586e-5 | 0.615 | 91 | +6.675e-6 | 0.205 | 34–1063 | +1.576e-3 | 0.845 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 163 | 78764–117080 | +1.946e-5 | 0.082 | +1.934e-5 | 0.078 | 49 | +2.838e-5 | 0.276 | 37–821 | +2.146e-3 | 0.729 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 107 | 117708–156133 | -1.851e-5 | 0.359 | -1.918e-5 | 0.377 | 49 | -2.202e-5 | 0.297 | 112–685 | +1.431e-3 | 0.355 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.562e-3 | r0: +1.527e-3, r1: +1.552e-3, r2: +1.611e-3 | r0: 0.813, r1: 0.770, r2: 0.786 | 1.06× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.576e-3 | r0: +1.540e-3, r1: +1.568e-3, r2: +1.624e-3 | r0: 0.870, r1: 0.830, r2: 0.830 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +2.146e-3 | r0: +2.096e-3, r1: +2.163e-3, r2: +2.185e-3 | r0: 0.751, r1: 0.708, r2: 0.720 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +1.431e-3 | r0: +1.432e-3, r1: +1.422e-3, r2: +1.445e-3 | r0: 0.370, r1: 0.344, r2: 0.348 | 1.02× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇████████████████████▅▄▄▄▄▄▄▅▅▆▆▆▄▂▁▂▂▁▁▂▂▂▁▁▁` | `▁▇▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▄▆▇▇▇▅███▇▇▇▆▅▆▆▇▇▇▇▇▇▇▆▆` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 0.00e0 | 4.79e-1 | 1.00e-1 | 6.23e-2 | 20 | -7.47e-2 | +2.37e-2 | -1.30e-2 | -7.74e-3 |
| 1 | 3.00e-1 | 17 | 5.53e-2 | 1.03e-1 | 6.43e-2 | 6.79e-2 | 18 | -3.39e-2 | +2.57e-2 | +1.69e-4 | -5.20e-4 |
| 2 | 3.00e-1 | 10 | 6.02e-2 | 1.04e-1 | 7.13e-2 | 8.11e-2 | 20 | -3.06e-2 | +2.73e-2 | +1.04e-3 | +7.11e-4 |
| 3 | 3.00e-1 | 14 | 6.51e-2 | 1.27e-1 | 7.68e-2 | 7.39e-2 | 18 | -2.99e-2 | +2.67e-2 | -1.95e-4 | +8.14e-5 |
| 4 | 3.00e-1 | 15 | 6.16e-2 | 1.28e-1 | 7.26e-2 | 6.44e-2 | 15 | -3.52e-2 | +3.52e-2 | -3.00e-4 | -4.87e-4 |
| 5 | 3.00e-1 | 17 | 6.09e-2 | 1.29e-1 | 7.92e-2 | 8.43e-2 | 20 | -3.06e-2 | +3.73e-2 | +7.62e-4 | +4.84e-4 |
| 6 | 3.00e-1 | 11 | 5.97e-2 | 1.38e-1 | 7.87e-2 | 7.15e-2 | 16 | -3.93e-2 | +3.02e-2 | -5.27e-4 | +3.33e-6 |
| 7 | 3.00e-1 | 14 | 6.41e-2 | 1.29e-1 | 7.70e-2 | 7.02e-2 | 17 | -3.24e-2 | +3.29e-2 | -8.89e-5 | -3.40e-4 |
| 8 | 3.00e-1 | 16 | 6.08e-2 | 1.38e-1 | 7.01e-2 | 6.89e-2 | 22 | -5.15e-2 | +4.54e-2 | -2.51e-4 | -8.52e-6 |
| 9 | 3.00e-1 | 21 | 6.29e-2 | 1.37e-1 | 7.50e-2 | 8.71e-2 | 17 | -3.82e-2 | +2.85e-2 | -8.38e-5 | +7.10e-4 |
| 10 | 3.00e-1 | 10 | 5.78e-2 | 1.26e-1 | 7.24e-2 | 6.50e-2 | 14 | -5.56e-2 | +4.03e-2 | -1.16e-3 | -5.54e-4 |
| 11 | 3.00e-1 | 3 | 5.87e-2 | 6.24e-2 | 6.06e-2 | 6.24e-2 | 240 | -5.94e-3 | +2.51e-4 | -2.58e-3 | -1.05e-3 |
| 12 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 222 | +5.58e-3 | +5.58e-3 | +5.58e-3 | -3.83e-4 |
| 13 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 243 | -2.79e-4 | -2.79e-4 | -2.79e-4 | -3.73e-4 |
| 14 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 280 | -2.82e-5 | -2.82e-5 | -2.82e-5 | -3.38e-4 |
| 15 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 360 | +5.23e-5 | +5.23e-5 | +5.23e-5 | -2.99e-4 |
| 17 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 399 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -2.57e-4 |
| 18 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 297 | +6.91e-5 | +6.91e-5 | +6.91e-5 | -2.24e-4 |
| 19 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 280 | -3.16e-4 | -3.16e-4 | -3.16e-4 | -2.33e-4 |
| 20 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 260 | -6.44e-5 | -6.44e-5 | -6.44e-5 | -2.17e-4 |
| 21 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 241 | -6.02e-5 | -6.02e-5 | -6.02e-5 | -2.01e-4 |
| 22 | 3.00e-1 | 2 | 1.84e-1 | 2.01e-1 | 1.93e-1 | 2.01e-1 | 257 | -1.69e-4 | +3.46e-4 | +8.86e-5 | -1.43e-4 |
| 24 | 3.00e-1 | 2 | 1.95e-1 | 2.08e-1 | 2.01e-1 | 2.08e-1 | 257 | -1.03e-4 | +2.49e-4 | +7.29e-5 | -1.00e-4 |
| 26 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 318 | -2.78e-4 | -2.78e-4 | -2.78e-4 | -1.18e-4 |
| 27 | 3.00e-1 | 2 | 2.01e-1 | 2.07e-1 | 2.04e-1 | 2.01e-1 | 254 | -1.20e-4 | +2.95e-4 | +8.72e-5 | -8.13e-5 |
| 29 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 297 | -1.49e-4 | -1.49e-4 | -1.49e-4 | -8.81e-5 |
| 30 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 283 | +1.61e-4 | +1.61e-4 | +1.61e-4 | -6.31e-5 |
| 31 | 3.00e-1 | 2 | 1.99e-1 | 2.00e-1 | 1.99e-1 | 1.99e-1 | 231 | -3.34e-5 | -1.84e-5 | -2.59e-5 | -5.60e-5 |
| 33 | 3.00e-1 | 2 | 1.87e-1 | 1.99e-1 | 1.93e-1 | 1.99e-1 | 231 | -2.15e-4 | +2.71e-4 | +2.81e-5 | -3.76e-5 |
| 34 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 236 | -2.07e-4 | -2.07e-4 | -2.07e-4 | -5.45e-5 |
| 35 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 289 | +6.78e-5 | +6.78e-5 | +6.78e-5 | -4.22e-5 |
| 36 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 295 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -2.50e-5 |
| 37 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 263 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -1.09e-5 |
| 39 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 350 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -2.33e-5 |
| 40 | 3.00e-1 | 2 | 2.01e-1 | 2.13e-1 | 2.07e-1 | 2.01e-1 | 239 | -2.50e-4 | +2.74e-4 | +1.18e-5 | -1.92e-5 |
| 41 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 268 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -3.71e-5 |
| 43 | 3.00e-1 | 2 | 1.99e-1 | 2.13e-1 | 2.06e-1 | 2.13e-1 | 227 | +1.22e-4 | +3.19e-4 | +2.20e-4 | +1.28e-5 |
| 44 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 275 | -5.14e-4 | -5.14e-4 | -5.14e-4 | -3.99e-5 |
| 45 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 246 | +2.63e-4 | +2.63e-4 | +2.63e-4 | -9.62e-6 |
| 46 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 232 | -3.02e-5 | -3.02e-5 | -3.02e-5 | -1.17e-5 |
| 47 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 260 | -8.37e-5 | -8.37e-5 | -8.37e-5 | -1.89e-5 |
| 48 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 252 | +1.62e-4 | +1.62e-4 | +1.62e-4 | -7.60e-7 |
| 49 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 271 | -4.88e-6 | -4.88e-6 | -4.88e-6 | -1.17e-6 |
| 50 | 3.00e-1 | 2 | 1.99e-1 | 2.03e-1 | 2.01e-1 | 2.03e-1 | 211 | -2.05e-5 | +9.65e-5 | +3.80e-5 | +6.86e-6 |
| 52 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 264 | -2.76e-4 | -2.76e-4 | -2.76e-4 | -2.14e-5 |
| 53 | 3.00e-1 | 2 | 1.99e-1 | 2.00e-1 | 2.00e-1 | 1.99e-1 | 212 | -1.77e-5 | +2.20e-4 | +1.01e-4 | +6.51e-7 |
| 54 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 264 | -3.13e-4 | -3.13e-4 | -3.13e-4 | -3.07e-5 |
| 55 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 253 | +3.91e-4 | +3.91e-4 | +3.91e-4 | +1.15e-5 |
| 56 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 233 | -1.06e-4 | -1.06e-4 | -1.06e-4 | -2.01e-7 |
| 57 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 231 | -8.22e-5 | -8.22e-5 | -8.22e-5 | -8.40e-6 |
| 58 | 3.00e-1 | 2 | 1.93e-1 | 1.95e-1 | 1.94e-1 | 1.95e-1 | 192 | -1.22e-5 | +4.06e-5 | +1.42e-5 | -3.84e-6 |
| 59 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 225 | -3.27e-4 | -3.27e-4 | -3.27e-4 | -3.62e-5 |
| 60 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 246 | +2.02e-4 | +2.02e-4 | +2.02e-4 | -1.24e-5 |
| 61 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 244 | +1.33e-4 | +1.33e-4 | +1.33e-4 | +2.19e-6 |
| 62 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 246 | -3.09e-5 | -3.09e-5 | -3.09e-5 | -1.12e-6 |
| 63 | 3.00e-1 | 2 | 1.92e-1 | 1.96e-1 | 1.94e-1 | 1.92e-1 | 188 | -9.46e-5 | +1.99e-5 | -3.73e-5 | -8.58e-6 |
| 64 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 232 | -2.65e-4 | -2.65e-4 | -2.65e-4 | -3.42e-5 |
| 65 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 221 | +3.18e-4 | +3.18e-4 | +3.18e-4 | +1.03e-6 |
| 66 | 3.00e-1 | 2 | 1.88e-1 | 1.92e-1 | 1.90e-1 | 1.88e-1 | 181 | -1.13e-4 | -5.93e-5 | -8.60e-5 | -1.58e-5 |
| 67 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 202 | -2.54e-4 | -2.54e-4 | -2.54e-4 | -3.96e-5 |
| 68 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 233 | +1.77e-4 | +1.77e-4 | +1.77e-4 | -1.80e-5 |
| 69 | 3.00e-1 | 2 | 1.92e-1 | 1.93e-1 | 1.93e-1 | 1.92e-1 | 181 | -3.82e-5 | +1.77e-4 | +6.95e-5 | -2.45e-6 |
| 70 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 195 | -3.11e-4 | -3.11e-4 | -3.11e-4 | -3.33e-5 |
| 71 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 207 | +9.53e-5 | +9.53e-5 | +9.53e-5 | -2.04e-5 |
| 72 | 3.00e-1 | 2 | 1.85e-1 | 1.87e-1 | 1.86e-1 | 1.85e-1 | 181 | -5.55e-5 | +7.32e-5 | +8.85e-6 | -1.55e-5 |
| 73 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 222 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -2.75e-5 |
| 74 | 3.00e-1 | 2 | 1.89e-1 | 1.92e-1 | 1.90e-1 | 1.89e-1 | 181 | -8.89e-5 | +2.94e-4 | +1.02e-4 | -4.72e-6 |
| 75 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 194 | -2.88e-4 | -2.88e-4 | -2.88e-4 | -3.30e-5 |
| 76 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 216 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -1.44e-5 |
| 77 | 3.00e-1 | 2 | 1.88e-1 | 1.90e-1 | 1.89e-1 | 1.88e-1 | 186 | -5.80e-5 | +1.39e-4 | +4.03e-5 | -4.96e-6 |
| 78 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 200 | -8.80e-5 | -8.80e-5 | -8.80e-5 | -1.33e-5 |
| 79 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 204 | +6.93e-5 | +6.93e-5 | +6.93e-5 | -5.01e-6 |
| 80 | 3.00e-1 | 2 | 1.86e-1 | 1.90e-1 | 1.88e-1 | 1.90e-1 | 182 | -1.62e-5 | +1.11e-4 | +4.76e-5 | +5.61e-6 |
| 81 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 194 | -2.50e-4 | -2.50e-4 | -2.50e-4 | -2.00e-5 |
| 82 | 3.00e-1 | 2 | 1.80e-1 | 1.83e-1 | 1.82e-1 | 1.80e-1 | 160 | -1.00e-4 | +5.59e-5 | -2.21e-5 | -2.11e-5 |
| 83 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 198 | -2.26e-4 | -2.26e-4 | -2.26e-4 | -4.16e-5 |
| 84 | 3.00e-1 | 2 | 1.85e-1 | 1.86e-1 | 1.85e-1 | 1.85e-1 | 160 | -2.39e-5 | +3.83e-4 | +1.80e-4 | -1.59e-6 |
| 85 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 200 | -3.35e-4 | -3.35e-4 | -3.35e-4 | -3.49e-5 |
| 86 | 3.00e-1 | 2 | 1.80e-1 | 1.85e-1 | 1.83e-1 | 1.80e-1 | 157 | -1.97e-4 | +3.64e-4 | +8.33e-5 | -1.53e-5 |
| 87 | 3.00e-1 | 2 | 1.69e-1 | 1.88e-1 | 1.78e-1 | 1.88e-1 | 157 | -3.22e-4 | +6.78e-4 | +1.78e-4 | +2.64e-5 |
| 88 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 185 | -4.77e-4 | -4.77e-4 | -4.77e-4 | -2.39e-5 |
| 89 | 3.00e-1 | 2 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 149 | -6.68e-6 | +3.45e-4 | +1.69e-4 | +1.10e-5 |
| 90 | 3.00e-1 | 2 | 1.71e-1 | 1.81e-1 | 1.76e-1 | 1.81e-1 | 152 | -3.77e-4 | +3.73e-4 | -1.81e-6 | +1.23e-5 |
| 91 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 170 | -1.88e-4 | -1.88e-4 | -1.88e-4 | -7.73e-6 |
| 92 | 3.00e-1 | 2 | 1.78e-1 | 1.80e-1 | 1.79e-1 | 1.78e-1 | 145 | -8.93e-5 | +1.75e-4 | +4.27e-5 | +5.27e-7 |
| 93 | 3.00e-1 | 2 | 1.69e-1 | 1.79e-1 | 1.74e-1 | 1.79e-1 | 140 | -2.98e-4 | +4.06e-4 | +5.44e-5 | +1.43e-5 |
| 94 | 3.00e-1 | 1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 1.66e-1 | 215 | -3.62e-4 | -3.62e-4 | -3.62e-4 | -2.34e-5 |
| 95 | 3.00e-1 | 2 | 1.79e-1 | 1.95e-1 | 1.87e-1 | 1.79e-1 | 131 | -6.33e-4 | +8.89e-4 | +1.28e-4 | -2.19e-6 |
| 96 | 3.00e-1 | 2 | 1.60e-1 | 1.71e-1 | 1.66e-1 | 1.71e-1 | 131 | -7.34e-4 | +5.24e-4 | -1.05e-4 | -1.55e-5 |
| 97 | 3.00e-1 | 2 | 1.60e-1 | 1.85e-1 | 1.72e-1 | 1.85e-1 | 139 | -3.73e-4 | +1.06e-3 | +3.44e-4 | +5.99e-5 |
| 98 | 3.00e-1 | 2 | 1.64e-1 | 1.74e-1 | 1.69e-1 | 1.74e-1 | 123 | -7.38e-4 | +5.20e-4 | -1.09e-4 | +3.42e-5 |
| 99 | 3.00e-1 | 3 | 1.57e-1 | 1.72e-1 | 1.64e-1 | 1.64e-1 | 132 | -6.77e-4 | +7.00e-4 | -1.22e-4 | -5.78e-6 |
| 100 | 3.00e-2 | 1 | 1.63e-1 | 1.63e-1 | 1.63e-1 | 1.63e-1 | 184 | -1.50e-6 | -1.50e-6 | -1.50e-6 | -5.35e-6 |
| 101 | 3.00e-2 | 2 | 1.74e-2 | 1.89e-2 | 1.81e-2 | 1.74e-2 | 130 | -1.37e-2 | -6.01e-4 | -7.14e-3 | -1.29e-3 |
| 102 | 3.00e-2 | 2 | 1.71e-2 | 1.87e-2 | 1.79e-2 | 1.87e-2 | 123 | -1.47e-4 | +7.48e-4 | +3.00e-4 | -9.87e-4 |
| 103 | 3.00e-2 | 2 | 1.79e-2 | 1.99e-2 | 1.89e-2 | 1.99e-2 | 123 | -3.04e-4 | +8.84e-4 | +2.90e-4 | -7.39e-4 |
| 104 | 3.00e-2 | 2 | 1.81e-2 | 2.17e-2 | 1.99e-2 | 2.17e-2 | 112 | -5.80e-4 | +1.65e-3 | +5.34e-4 | -4.86e-4 |
| 105 | 3.00e-2 | 3 | 1.79e-2 | 2.02e-2 | 1.89e-2 | 1.79e-2 | 106 | -1.14e-3 | +7.77e-4 | -4.79e-4 | -4.85e-4 |
| 106 | 3.00e-2 | 2 | 1.96e-2 | 2.11e-2 | 2.03e-2 | 2.11e-2 | 112 | +6.15e-4 | +6.62e-4 | +6.39e-4 | -2.71e-4 |
| 107 | 3.00e-2 | 2 | 1.95e-2 | 2.23e-2 | 2.09e-2 | 2.23e-2 | 108 | -5.32e-4 | +1.24e-3 | +3.54e-4 | -1.44e-4 |
| 108 | 3.00e-2 | 2 | 2.03e-2 | 2.34e-2 | 2.19e-2 | 2.34e-2 | 110 | -5.86e-4 | +1.25e-3 | +3.34e-4 | -4.39e-5 |
| 109 | 3.00e-2 | 3 | 1.86e-2 | 2.25e-2 | 2.05e-2 | 1.86e-2 | 87 | -2.18e-3 | +1.16e-3 | -6.56e-4 | -2.22e-4 |
| 110 | 3.00e-2 | 4 | 1.88e-2 | 2.27e-2 | 2.05e-2 | 2.04e-2 | 89 | -1.35e-3 | +2.09e-3 | +2.64e-4 | -6.85e-5 |
| 111 | 3.00e-2 | 2 | 1.97e-2 | 2.28e-2 | 2.13e-2 | 2.28e-2 | 89 | -2.92e-4 | +1.62e-3 | +6.66e-4 | +8.06e-5 |
| 112 | 3.00e-2 | 3 | 2.01e-2 | 2.32e-2 | 2.12e-2 | 2.02e-2 | 89 | -1.53e-3 | +1.59e-3 | -3.23e-4 | -3.47e-5 |
| 113 | 3.00e-2 | 2 | 2.15e-2 | 2.45e-2 | 2.30e-2 | 2.45e-2 | 85 | +4.71e-4 | +1.56e-3 | +1.01e-3 | +1.70e-4 |
| 114 | 3.00e-2 | 4 | 1.93e-2 | 2.46e-2 | 2.14e-2 | 2.07e-2 | 74 | -3.23e-3 | +2.08e-3 | -3.64e-4 | -8.26e-6 |
| 115 | 3.00e-2 | 4 | 1.99e-2 | 2.30e-2 | 2.11e-2 | 2.07e-2 | 78 | -1.94e-3 | +1.26e-3 | -1.21e-6 | -1.40e-5 |
| 116 | 3.00e-2 | 2 | 2.06e-2 | 2.50e-2 | 2.28e-2 | 2.50e-2 | 75 | -5.84e-5 | +2.60e-3 | +1.27e-3 | +2.44e-4 |
| 117 | 3.00e-2 | 4 | 1.92e-2 | 2.54e-2 | 2.13e-2 | 1.92e-2 | 68 | -4.11e-3 | +2.81e-3 | -7.24e-4 | -1.02e-4 |
| 118 | 3.00e-2 | 4 | 1.93e-2 | 2.48e-2 | 2.11e-2 | 1.93e-2 | 61 | -4.06e-3 | +2.90e-3 | -1.06e-4 | -1.44e-4 |
| 119 | 3.00e-2 | 6 | 1.89e-2 | 2.40e-2 | 2.02e-2 | 1.91e-2 | 52 | -3.74e-3 | +3.91e-3 | -2.80e-5 | -1.35e-4 |
| 120 | 3.00e-2 | 4 | 1.82e-2 | 2.44e-2 | 2.00e-2 | 1.82e-2 | 40 | -5.86e-3 | +5.50e-3 | -1.80e-4 | -2.04e-4 |
| 121 | 3.00e-2 | 7 | 1.47e-2 | 2.32e-2 | 1.65e-2 | 1.48e-2 | 34 | -1.17e-2 | +9.15e-3 | -7.63e-4 | -5.26e-4 |
| 122 | 3.00e-2 | 9 | 1.22e-2 | 2.16e-2 | 1.48e-2 | 1.27e-2 | 27 | -1.01e-2 | +1.01e-2 | -6.63e-4 | -7.21e-4 |
| 123 | 3.00e-2 | 12 | 1.05e-2 | 1.99e-2 | 1.25e-2 | 1.15e-2 | 18 | -1.67e-2 | +1.50e-2 | -6.38e-4 | -5.21e-4 |
| 124 | 3.00e-2 | 17 | 9.10e-3 | 2.08e-2 | 1.13e-2 | 1.32e-2 | 20 | -4.85e-2 | +4.15e-2 | +4.24e-4 | +7.47e-4 |
| 125 | 3.00e-2 | 11 | 9.49e-3 | 2.22e-2 | 1.20e-2 | 1.31e-2 | 19 | -4.54e-2 | +3.37e-2 | -8.33e-5 | +6.11e-4 |
| 126 | 3.00e-2 | 19 | 8.91e-3 | 2.15e-2 | 1.14e-2 | 1.15e-2 | 18 | -5.06e-2 | +4.03e-2 | -4.30e-4 | +6.02e-5 |
| 127 | 3.00e-2 | 2 | 1.11e-2 | 1.15e-2 | 1.13e-2 | 1.15e-2 | 18 | -2.15e-3 | +1.89e-3 | -1.31e-4 | +4.41e-5 |
| 128 | 3.00e-2 | 1 | 1.14e-2 | 1.14e-2 | 1.14e-2 | 1.14e-2 | 293 | -2.75e-5 | -2.75e-5 | -2.75e-5 | +3.70e-5 |
| 129 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 282 | +5.30e-3 | +5.30e-3 | +5.30e-3 | +5.63e-4 |
| 130 | 3.00e-2 | 1 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 295 | -1.02e-4 | -1.02e-4 | -1.02e-4 | +4.97e-4 |
| 131 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 294 | -5.21e-5 | -5.21e-5 | -5.21e-5 | +4.42e-4 |
| 132 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 286 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +4.11e-4 |
| 133 | 3.00e-2 | 2 | 4.80e-2 | 5.03e-2 | 4.91e-2 | 4.80e-2 | 245 | -1.88e-4 | -1.40e-5 | -1.01e-4 | +3.13e-4 |
| 135 | 3.00e-2 | 2 | 4.81e-2 | 5.14e-2 | 4.98e-2 | 5.14e-2 | 213 | +5.57e-6 | +3.20e-4 | +1.63e-4 | +2.86e-4 |
| 136 | 3.00e-2 | 1 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 4.65e-2 | 257 | -3.93e-4 | -3.93e-4 | -3.93e-4 | +2.18e-4 |
| 137 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 234 | +3.71e-4 | +3.71e-4 | +3.71e-4 | +2.34e-4 |
| 138 | 3.00e-2 | 1 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 5.05e-2 | 240 | -2.01e-5 | -2.01e-5 | -2.01e-5 | +2.08e-4 |
| 139 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 248 | -5.65e-5 | -5.65e-5 | -5.65e-5 | +1.82e-4 |
| 140 | 3.00e-2 | 1 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 5.23e-2 | 256 | +1.92e-4 | +1.92e-4 | +1.92e-4 | +1.83e-4 |
| 141 | 3.00e-2 | 1 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 5.31e-2 | 247 | +6.05e-5 | +6.05e-5 | +6.05e-5 | +1.71e-4 |
| 142 | 3.00e-2 | 2 | 5.35e-2 | 5.55e-2 | 5.45e-2 | 5.55e-2 | 187 | +3.19e-5 | +1.93e-4 | +1.13e-4 | +1.60e-4 |
| 143 | 3.00e-2 | 1 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 4.85e-2 | 194 | -6.92e-4 | -6.92e-4 | -6.92e-4 | +7.50e-5 |
| 144 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 210 | +1.22e-4 | +1.22e-4 | +1.22e-4 | +7.97e-5 |
| 145 | 3.00e-2 | 2 | 5.05e-2 | 5.20e-2 | 5.12e-2 | 5.20e-2 | 187 | +6.40e-5 | +1.56e-4 | +1.10e-4 | +8.59e-5 |
| 146 | 3.00e-2 | 1 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 4.91e-2 | 227 | -2.48e-4 | -2.48e-4 | -2.48e-4 | +5.25e-5 |
| 147 | 3.00e-2 | 1 | 5.39e-2 | 5.39e-2 | 5.39e-2 | 5.39e-2 | 237 | +3.89e-4 | +3.89e-4 | +3.89e-4 | +8.62e-5 |
| 148 | 3.00e-2 | 2 | 5.39e-2 | 5.46e-2 | 5.42e-2 | 5.39e-2 | 203 | -6.41e-5 | +5.81e-5 | -2.98e-6 | +6.86e-5 |
| 149 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 235 | -4.80e-5 | -4.80e-5 | -4.80e-5 | +5.69e-5 |
| 150 | 3.00e-3 | 1 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 5.64e-2 | 215 | +2.63e-4 | +2.63e-4 | +2.63e-4 | +7.75e-5 |
| 151 | 3.00e-3 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 243 | -2.47e-3 | -2.47e-3 | -2.47e-3 | -1.77e-4 |
| 152 | 3.00e-3 | 2 | 5.50e-3 | 5.60e-3 | 5.55e-3 | 5.60e-3 | 192 | -6.80e-3 | +9.67e-5 | -3.35e-3 | -7.46e-4 |
| 153 | 3.00e-3 | 1 | 5.03e-3 | 5.03e-3 | 5.03e-3 | 5.03e-3 | 237 | -4.55e-4 | -4.55e-4 | -4.55e-4 | -7.17e-4 |
| 154 | 3.00e-3 | 1 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 5.52e-3 | 234 | +3.97e-4 | +3.97e-4 | +3.97e-4 | -6.06e-4 |
| 155 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 217 | +5.36e-5 | +5.36e-5 | +5.36e-5 | -5.40e-4 |
| 156 | 3.00e-3 | 2 | 5.23e-3 | 5.45e-3 | 5.34e-3 | 5.23e-3 | 166 | -2.48e-4 | -1.15e-4 | -1.82e-4 | -4.72e-4 |
| 157 | 3.00e-3 | 1 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 188 | -6.20e-4 | -6.20e-4 | -6.20e-4 | -4.87e-4 |
| 158 | 3.00e-3 | 2 | 5.18e-3 | 5.22e-3 | 5.20e-3 | 5.22e-3 | 172 | +5.11e-5 | +5.21e-4 | +2.86e-4 | -3.43e-4 |
| 159 | 3.00e-3 | 1 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 5.10e-3 | 207 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -3.20e-4 |
| 160 | 3.00e-3 | 1 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 5.40e-3 | 197 | +2.87e-4 | +2.87e-4 | +2.87e-4 | -2.59e-4 |
| 161 | 3.00e-3 | 3 | 5.13e-3 | 5.35e-3 | 5.20e-3 | 5.13e-3 | 152 | -2.80e-4 | +2.45e-4 | -9.50e-5 | -2.15e-4 |
| 163 | 3.00e-3 | 2 | 4.75e-3 | 5.95e-3 | 5.35e-3 | 5.95e-3 | 138 | -2.96e-4 | +1.63e-3 | +6.67e-4 | -3.77e-5 |
| 164 | 3.00e-3 | 2 | 4.75e-3 | 4.82e-3 | 4.78e-3 | 4.82e-3 | 138 | -1.49e-3 | +1.03e-4 | -6.92e-4 | -1.54e-4 |
| 165 | 3.00e-3 | 2 | 4.54e-3 | 5.24e-3 | 4.89e-3 | 5.24e-3 | 151 | -3.13e-4 | +9.53e-4 | +3.20e-4 | -5.76e-5 |
| 166 | 3.00e-3 | 2 | 4.85e-3 | 5.30e-3 | 5.08e-3 | 5.30e-3 | 171 | -4.06e-4 | +5.20e-4 | +5.70e-5 | -3.12e-5 |
| 167 | 3.00e-3 | 1 | 5.13e-3 | 5.13e-3 | 5.13e-3 | 5.13e-3 | 198 | -1.65e-4 | -1.65e-4 | -1.65e-4 | -4.46e-5 |
| 168 | 3.00e-3 | 1 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 5.64e-3 | 190 | +4.94e-4 | +4.94e-4 | +4.94e-4 | +9.27e-6 |
| 169 | 3.00e-3 | 3 | 5.11e-3 | 5.34e-3 | 5.26e-3 | 5.11e-3 | 143 | -3.02e-4 | -1.28e-5 | -2.05e-4 | -4.90e-5 |
| 170 | 3.00e-3 | 1 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 4.65e-3 | 187 | -4.96e-4 | -4.96e-4 | -4.96e-4 | -9.37e-5 |
| 171 | 3.00e-3 | 1 | 5.47e-3 | 5.47e-3 | 5.47e-3 | 5.47e-3 | 191 | +8.49e-4 | +8.49e-4 | +8.49e-4 | +5.27e-7 |
| 172 | 3.00e-3 | 2 | 5.11e-3 | 5.46e-3 | 5.29e-3 | 5.11e-3 | 143 | -4.64e-4 | -1.38e-5 | -2.39e-4 | -4.73e-5 |
| 173 | 3.00e-3 | 3 | 4.69e-3 | 4.93e-3 | 4.81e-3 | 4.82e-3 | 143 | -5.05e-4 | +3.47e-4 | -1.03e-4 | -5.92e-5 |
| 174 | 3.00e-3 | 1 | 4.75e-3 | 4.75e-3 | 4.75e-3 | 4.75e-3 | 175 | -8.72e-5 | -8.72e-5 | -8.72e-5 | -6.20e-5 |
| 175 | 3.00e-3 | 2 | 5.27e-3 | 5.32e-3 | 5.30e-3 | 5.32e-3 | 144 | +5.89e-5 | +6.21e-4 | +3.40e-4 | +1.16e-5 |
| 176 | 3.00e-3 | 1 | 4.76e-3 | 4.76e-3 | 4.76e-3 | 4.76e-3 | 174 | -6.33e-4 | -6.33e-4 | -6.33e-4 | -5.29e-5 |
| 177 | 3.00e-3 | 3 | 4.98e-3 | 5.44e-3 | 5.23e-3 | 4.98e-3 | 144 | -4.01e-4 | +7.68e-4 | +5.12e-5 | -3.56e-5 |
| 178 | 3.00e-3 | 1 | 4.90e-3 | 4.90e-3 | 4.90e-3 | 4.90e-3 | 165 | -8.96e-5 | -8.96e-5 | -8.96e-5 | -4.10e-5 |
| 179 | 3.00e-3 | 2 | 5.24e-3 | 5.55e-3 | 5.40e-3 | 5.55e-3 | 124 | +3.60e-4 | +4.56e-4 | +4.08e-4 | +4.48e-5 |
| 180 | 3.00e-3 | 2 | 4.54e-3 | 5.22e-3 | 4.88e-3 | 5.22e-3 | 124 | -1.34e-3 | +1.12e-3 | -1.07e-4 | +2.83e-5 |
| 181 | 3.00e-3 | 2 | 4.81e-3 | 5.60e-3 | 5.20e-3 | 5.60e-3 | 127 | -4.62e-4 | +1.19e-3 | +3.63e-4 | +1.00e-4 |
| 182 | 3.00e-3 | 2 | 4.83e-3 | 5.31e-3 | 5.07e-3 | 5.31e-3 | 127 | -8.96e-4 | +7.52e-4 | -7.22e-5 | +7.57e-5 |
| 183 | 3.00e-3 | 2 | 4.83e-3 | 5.33e-3 | 5.08e-3 | 5.33e-3 | 116 | -6.02e-4 | +8.60e-4 | +1.29e-4 | +9.30e-5 |
| 184 | 3.00e-3 | 2 | 4.46e-3 | 4.94e-3 | 4.70e-3 | 4.94e-3 | 116 | -1.27e-3 | +8.79e-4 | -1.94e-4 | +4.92e-5 |
| 185 | 3.00e-3 | 2 | 4.46e-3 | 5.03e-3 | 4.75e-3 | 5.03e-3 | 122 | -6.58e-4 | +9.91e-4 | +1.66e-4 | +7.97e-5 |
| 186 | 3.00e-3 | 2 | 4.59e-3 | 5.19e-3 | 4.89e-3 | 5.19e-3 | 129 | -5.68e-4 | +9.42e-4 | +1.87e-4 | +1.08e-4 |
| 187 | 3.00e-3 | 3 | 4.52e-3 | 5.35e-3 | 4.94e-3 | 4.52e-3 | 102 | -1.65e-3 | +7.16e-4 | -4.15e-4 | -4.74e-5 |
| 188 | 3.00e-3 | 2 | 4.54e-3 | 5.20e-3 | 4.87e-3 | 5.20e-3 | 94 | +2.74e-5 | +1.46e-3 | +7.43e-4 | +1.10e-4 |
| 189 | 3.00e-3 | 3 | 4.10e-3 | 4.89e-3 | 4.36e-3 | 4.10e-3 | 85 | -2.07e-3 | +1.95e-3 | -6.19e-4 | -9.21e-5 |
| 190 | 3.00e-3 | 3 | 4.13e-3 | 4.85e-3 | 4.38e-3 | 4.15e-3 | 83 | -1.89e-3 | +1.94e-3 | +3.70e-5 | -7.65e-5 |
| 191 | 3.00e-3 | 3 | 3.94e-3 | 4.89e-3 | 4.37e-3 | 3.94e-3 | 84 | -2.58e-3 | +1.60e-3 | -2.44e-4 | -1.50e-4 |
| 192 | 3.00e-3 | 4 | 3.83e-3 | 4.62e-3 | 4.18e-3 | 3.83e-3 | 77 | -1.55e-3 | +1.17e-3 | -1.89e-4 | -1.94e-4 |
| 193 | 3.00e-3 | 2 | 3.85e-3 | 4.70e-3 | 4.28e-3 | 4.70e-3 | 77 | +4.81e-5 | +2.59e-3 | +1.32e-3 | +1.07e-4 |
| 194 | 3.00e-3 | 4 | 3.63e-3 | 4.77e-3 | 4.04e-3 | 3.63e-3 | 73 | -2.66e-3 | +2.84e-3 | -6.33e-4 | -1.67e-4 |
| 195 | 3.00e-3 | 3 | 3.72e-3 | 4.56e-3 | 4.02e-3 | 3.72e-3 | 80 | -2.56e-3 | +2.63e-3 | +1.40e-4 | -1.12e-4 |
| 196 | 3.00e-3 | 4 | 3.50e-3 | 4.75e-3 | 3.98e-3 | 3.50e-3 | 58 | -4.25e-3 | +2.60e-3 | -5.26e-4 | -3.07e-4 |
| 197 | 3.00e-3 | 4 | 3.25e-3 | 4.40e-3 | 3.61e-3 | 3.25e-3 | 54 | -5.17e-3 | +4.41e-3 | -3.40e-4 | -3.66e-4 |
| 198 | 3.00e-3 | 5 | 3.07e-3 | 3.93e-3 | 3.33e-3 | 3.07e-3 | 47 | -4.13e-3 | +3.72e-3 | -2.27e-4 | -3.57e-4 |
| 199 | 3.00e-3 | 6 | 2.69e-3 | 3.77e-3 | 3.08e-3 | 3.19e-3 | 44 | -6.10e-3 | +5.12e-3 | +1.76e-4 | -7.36e-5 |

