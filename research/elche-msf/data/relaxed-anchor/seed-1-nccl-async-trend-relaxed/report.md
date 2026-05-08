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
| nccl-async | 0.063926 | 0.9124 | -0.0001 | 1855.0 | 291 | 39.8 | 100% | 100% | 100% | 4.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9124 | nccl-async | - | - |

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
| nccl-async | 1.9864 | 0.7716 | 0.6102 | 0.5474 | 0.5276 | 0.5119 | 0.4923 | 0.4727 | 0.4593 | 0.4665 | 0.2140 | 0.1786 | 0.1575 | 0.1492 | 0.1413 | 0.0805 | 0.0737 | 0.0698 | 0.0649 | 0.0639 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4019 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3013 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2968 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 386 | 384 | 388 | 381 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1854.0 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu2 | 1854.0 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1854.0 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 2.1s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 71 | 0 | 291 | 39.8 | 7080/9406 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 197.0 | 10.6% |

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
| resnet-graph | nccl-async | 192 | 291 | 0 | 6.74e-3 | +2.12e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 291 | 1.04e-1 | 7.61e-2 | 0.00e0 | 3.82e-1 | 23.7 | -1.31e-4 | 1.35e-3 |
| resnet-graph | nccl-async | 1 | 291 | 1.06e-1 | 7.84e-2 | 0.00e0 | 4.12e-1 | 46.7 | -1.37e-4 | 1.62e-3 |
| resnet-graph | nccl-async | 2 | 291 | 1.06e-1 | 7.87e-2 | 0.00e0 | 4.40e-1 | 29.6 | -1.48e-4 | 1.66e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9979 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9961 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9988 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 41 (0,1,2,3,4,12,13,19…148,149) | 0 (—) | — | 0,1,2,3,4,12,13,19…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 30 | 30 |
| resnet-graph | nccl-async | 0e0 | 5 | 13 | 13 |
| resnet-graph | nccl-async | 0e0 | 10 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 3 | 1 | 1 |
| resnet-graph | nccl-async | 1e-4 | 5 | 1 | 1 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 176 | +0.124 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 52 | +0.191 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 58 | +0.081 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 288 | -0.004 | 191 | +0.205 | +0.296 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 289 | 3.37e1–7.97e1 | 6.49e1 | 4.17e-3 | 7.31e-3 | 6.57e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | 178 | 71–77491 | +1.019e-5 | 0.428 | +1.051e-5 | 0.440 | 99 | +4.141e-6 | 0.244 | 38–829 | +1.345e-3 | 0.708 |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 166 | 899–77491 | +9.633e-6 | 0.485 | +9.868e-6 | 0.488 | 98 | +3.742e-6 | 0.220 | 60–829 | +1.372e-3 | 0.852 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 53 | 78338–115973 | +1.131e-5 | 0.094 | +1.115e-5 | 0.092 | 45 | +1.145e-5 | 0.133 | 563–916 | -7.335e-5 | 0.000 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 59 | 116667–155830 | -1.223e-5 | 0.080 | -1.227e-5 | 0.081 | 48 | -5.946e-6 | 0.034 | 561–797 | +7.229e-4 | 0.008 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–98 | +1.345e-3 | r0: +1.317e-3, r1: +1.343e-3, r2: +1.379e-3 | r0: 0.733, r1: 0.701, r2: 0.689 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | +1.372e-3 | r0: +1.342e-3, r1: +1.366e-3, r2: +1.411e-3 | r0: 0.868, r1: 0.843, r2: 0.839 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | -7.335e-5 | r0: +5.742e-6, r1: -1.114e-4, r2: -1.118e-4 | r0: 0.000, r1: 0.000, r2: 0.000 | 19.48× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | +7.229e-4 | r0: +8.231e-4, r1: +7.095e-4, r2: +6.367e-4 | r0: 0.010, r1: 0.008, r2: 0.006 | 1.29× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇███████████████████████▅▄▄▄▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁▁` | `▁████████████████████████▆▇▇▇███████▆▆▆▇▇▇██████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 4.40e-1 | 1.16e-1 | 6.56e-2 | 30 | -3.97e-2 | +6.93e-3 | -9.06e-3 | -5.90e-3 |
| 1 | 3.00e-1 | 9 | 7.07e-2 | 1.16e-1 | 8.16e-2 | 8.80e-2 | 28 | -1.74e-2 | +1.62e-2 | +7.52e-4 | -1.24e-3 |
| 2 | 3.00e-1 | 10 | 6.58e-2 | 1.18e-1 | 8.36e-2 | 7.82e-2 | 25 | -2.54e-2 | +1.78e-2 | -2.01e-4 | -6.52e-4 |
| 3 | 3.00e-1 | 13 | 8.07e-2 | 1.34e-1 | 9.23e-2 | 8.07e-2 | 23 | -1.42e-2 | +1.48e-2 | -2.69e-4 | -6.48e-4 |
| 4 | 3.00e-1 | 7 | 9.07e-2 | 1.32e-1 | 9.90e-2 | 9.93e-2 | 30 | -1.37e-2 | +1.39e-2 | +7.05e-4 | -2.83e-5 |
| 5 | 3.00e-1 | 2 | 9.53e-2 | 9.70e-2 | 9.62e-2 | 9.53e-2 | 229 | -7.62e-4 | -7.68e-5 | -4.20e-4 | -9.95e-5 |
| 6 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 240 | +3.30e-3 | +3.30e-3 | +3.30e-3 | +2.41e-4 |
| 7 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 270 | -1.31e-4 | -1.31e-4 | -1.31e-4 | +2.04e-4 |
| 8 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 254 | -4.12e-5 | -4.12e-5 | -4.12e-5 | +1.79e-4 |
| 9 | 3.00e-1 | 2 | 1.88e-1 | 1.98e-1 | 1.93e-1 | 1.88e-1 | 189 | -2.64e-4 | -6.27e-5 | -1.63e-4 | +1.13e-4 |
| 10 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 212 | -3.41e-4 | -3.41e-4 | -3.41e-4 | +6.77e-5 |
| 11 | 3.00e-1 | 1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 1.86e-1 | 247 | +2.36e-4 | +2.36e-4 | +2.36e-4 | +8.45e-5 |
| 12 | 3.00e-1 | 1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 1.89e-1 | 234 | +7.15e-5 | +7.15e-5 | +7.15e-5 | +8.32e-5 |
| 13 | 3.00e-1 | 2 | 1.86e-1 | 1.89e-1 | 1.88e-1 | 1.86e-1 | 198 | -7.35e-5 | +3.26e-6 | -3.51e-5 | +6.03e-5 |
| 14 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 200 | -1.59e-4 | -1.59e-4 | -1.59e-4 | +3.84e-5 |
| 15 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 211 | -9.93e-5 | -9.93e-5 | -9.93e-5 | +2.46e-5 |
| 16 | 3.00e-1 | 2 | 1.79e-1 | 1.83e-1 | 1.81e-1 | 1.79e-1 | 179 | -1.36e-4 | +1.75e-4 | +1.95e-5 | +2.21e-5 |
| 17 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 210 | -1.61e-4 | -1.61e-4 | -1.61e-4 | +3.81e-6 |
| 18 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 211 | +1.59e-4 | +1.59e-4 | +1.59e-4 | +1.93e-5 |
| 19 | 3.00e-1 | 2 | 1.83e-1 | 1.88e-1 | 1.86e-1 | 1.88e-1 | 197 | +1.10e-4 | +1.37e-4 | +1.24e-4 | +3.93e-5 |
| 20 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 203 | -2.76e-4 | -2.76e-4 | -2.76e-4 | +7.82e-6 |
| 21 | 3.00e-1 | 1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 1.80e-1 | 186 | +5.31e-5 | +5.31e-5 | +5.31e-5 | +1.24e-5 |
| 22 | 3.00e-1 | 2 | 1.74e-1 | 1.77e-1 | 1.75e-1 | 1.74e-1 | 179 | -1.04e-4 | -8.10e-5 | -9.24e-5 | -7.44e-6 |
| 23 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 192 | -7.98e-5 | -7.98e-5 | -7.98e-5 | -1.47e-5 |
| 24 | 3.00e-1 | 2 | 1.78e-1 | 1.84e-1 | 1.81e-1 | 1.84e-1 | 194 | +1.63e-4 | +1.75e-4 | +1.69e-4 | +2.02e-5 |
| 25 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 258 | -1.01e-4 | -1.01e-4 | -1.01e-4 | +8.09e-6 |
| 26 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 218 | +4.21e-4 | +4.21e-4 | +4.21e-4 | +4.94e-5 |
| 27 | 3.00e-1 | 1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 224 | -2.22e-4 | -2.22e-4 | -2.22e-4 | +2.23e-5 |
| 28 | 3.00e-1 | 2 | 1.87e-1 | 1.88e-1 | 1.88e-1 | 1.87e-1 | 172 | -1.46e-5 | +2.83e-5 | +6.88e-6 | +1.91e-5 |
| 29 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 207 | -5.35e-4 | -5.35e-4 | -5.35e-4 | -3.63e-5 |
| 30 | 3.00e-1 | 2 | 1.83e-1 | 1.89e-1 | 1.86e-1 | 1.89e-1 | 185 | +1.82e-4 | +3.77e-4 | +2.79e-4 | +2.28e-5 |
| 31 | 3.00e-1 | 1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 189 | -3.16e-4 | -3.16e-4 | -3.16e-4 | -1.11e-5 |
| 32 | 3.00e-1 | 2 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 1.78e-1 | 172 | -1.03e-5 | +3.12e-5 | +1.04e-5 | -6.81e-6 |
| 33 | 3.00e-1 | 1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 1.71e-1 | 213 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -2.52e-5 |
| 34 | 3.00e-1 | 2 | 1.81e-1 | 1.85e-1 | 1.83e-1 | 1.81e-1 | 183 | -1.14e-4 | +4.10e-4 | +1.48e-4 | +5.10e-6 |
| 35 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 218 | -5.90e-5 | -5.90e-5 | -5.90e-5 | -1.31e-6 |
| 36 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 196 | +2.57e-4 | +2.57e-4 | +2.57e-4 | +2.46e-5 |
| 37 | 3.00e-1 | 2 | 1.81e-1 | 1.88e-1 | 1.85e-1 | 1.88e-1 | 171 | -1.63e-4 | +2.11e-4 | +2.44e-5 | +2.64e-5 |
| 38 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 202 | -4.06e-4 | -4.06e-4 | -4.06e-4 | -1.68e-5 |
| 39 | 3.00e-1 | 2 | 1.83e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 190 | +2.68e-5 | +2.72e-4 | +1.49e-4 | +1.35e-5 |
| 40 | 3.00e-1 | 1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 1.82e-1 | 235 | -6.42e-5 | -6.42e-5 | -6.42e-5 | +5.72e-6 |
| 41 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 236 | +3.12e-4 | +3.12e-4 | +3.12e-4 | +3.63e-5 |
| 42 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 213 | -6.98e-5 | -6.98e-5 | -6.98e-5 | +2.57e-5 |
| 43 | 3.00e-1 | 2 | 1.79e-1 | 1.82e-1 | 1.80e-1 | 1.79e-1 | 162 | -2.90e-4 | -1.20e-4 | -2.05e-4 | -1.73e-5 |
| 44 | 3.00e-1 | 1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 1.69e-1 | 195 | -2.81e-4 | -2.81e-4 | -2.81e-4 | -4.37e-5 |
| 45 | 3.00e-1 | 2 | 1.82e-1 | 1.85e-1 | 1.83e-1 | 1.85e-1 | 182 | +9.25e-5 | +3.67e-4 | +2.30e-4 | +6.90e-6 |
| 46 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 237 | -1.91e-4 | -1.91e-4 | -1.91e-4 | -1.29e-5 |
| 47 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 232 | +4.44e-4 | +4.44e-4 | +4.44e-4 | +3.28e-5 |
| 48 | 3.00e-1 | 2 | 1.88e-1 | 1.92e-1 | 1.90e-1 | 1.88e-1 | 182 | -1.18e-4 | -9.53e-5 | -1.06e-4 | +6.23e-6 |
| 49 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 192 | -3.07e-4 | -3.07e-4 | -3.07e-4 | -2.51e-5 |
| 50 | 3.00e-1 | 2 | 1.82e-1 | 1.83e-1 | 1.82e-1 | 1.83e-1 | 182 | +4.10e-5 | +1.33e-4 | +8.68e-5 | -4.27e-6 |
| 51 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 190 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -1.71e-5 |
| 52 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 251 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -2.89e-6 |
| 53 | 3.00e-1 | 2 | 1.91e-1 | 1.98e-1 | 1.95e-1 | 1.91e-1 | 170 | -2.12e-4 | +3.20e-4 | +5.38e-5 | +5.22e-6 |
| 54 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 208 | -4.80e-4 | -4.80e-4 | -4.80e-4 | -4.33e-5 |
| 55 | 3.00e-1 | 2 | 1.84e-1 | 1.85e-1 | 1.85e-1 | 1.84e-1 | 161 | -3.51e-5 | +3.59e-4 | +1.62e-4 | -6.31e-6 |
| 56 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 193 | -3.50e-4 | -3.50e-4 | -3.50e-4 | -4.07e-5 |
| 57 | 3.00e-1 | 2 | 1.86e-1 | 1.87e-1 | 1.86e-1 | 1.87e-1 | 184 | +4.05e-5 | +3.49e-4 | +1.95e-4 | +2.50e-6 |
| 58 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 251 | -8.78e-5 | -8.78e-5 | -8.78e-5 | -6.53e-6 |
| 59 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 236 | +4.17e-4 | +4.17e-4 | +4.17e-4 | +3.58e-5 |
| 60 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 230 | -1.20e-4 | -1.20e-4 | -1.20e-4 | +2.02e-5 |
| 61 | 3.00e-1 | 2 | 1.79e-1 | 1.91e-1 | 1.85e-1 | 1.79e-1 | 151 | -4.59e-4 | -1.34e-4 | -2.97e-4 | -4.16e-5 |
| 62 | 3.00e-1 | 2 | 1.67e-1 | 1.77e-1 | 1.72e-1 | 1.77e-1 | 154 | -3.58e-4 | +3.44e-4 | -6.72e-6 | -3.15e-5 |
| 63 | 3.00e-1 | 2 | 1.71e-1 | 1.92e-1 | 1.82e-1 | 1.92e-1 | 166 | -1.38e-4 | +6.75e-4 | +2.68e-4 | +2.95e-5 |
| 64 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 214 | -3.96e-4 | -3.96e-4 | -3.96e-4 | -1.30e-5 |
| 65 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 226 | +3.72e-4 | +3.72e-4 | +3.72e-4 | +2.55e-5 |
| 66 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 205 | +7.89e-5 | +7.89e-5 | +7.89e-5 | +3.08e-5 |
| 67 | 3.00e-1 | 2 | 1.88e-1 | 1.90e-1 | 1.89e-1 | 1.90e-1 | 167 | -1.75e-4 | +7.09e-5 | -5.18e-5 | +1.64e-5 |
| 68 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 200 | -4.26e-4 | -4.26e-4 | -4.26e-4 | -2.79e-5 |
| 69 | 3.00e-1 | 2 | 1.88e-1 | 1.96e-1 | 1.92e-1 | 1.96e-1 | 170 | +2.23e-4 | +3.51e-4 | +2.87e-4 | +3.13e-5 |
| 70 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 186 | -5.40e-4 | -5.40e-4 | -5.40e-4 | -2.59e-5 |
| 71 | 3.00e-1 | 2 | 1.82e-1 | 1.83e-1 | 1.82e-1 | 1.83e-1 | 155 | +2.45e-5 | +1.46e-4 | +8.54e-5 | -5.33e-6 |
| 72 | 3.00e-1 | 2 | 1.72e-1 | 1.84e-1 | 1.78e-1 | 1.84e-1 | 182 | -3.35e-4 | +3.77e-4 | +2.13e-5 | +3.30e-6 |
| 73 | 3.00e-1 | 1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 1.84e-1 | 218 | -1.92e-6 | -1.92e-6 | -1.92e-6 | +2.77e-6 |
| 74 | 3.00e-1 | 2 | 1.83e-1 | 1.90e-1 | 1.87e-1 | 1.83e-1 | 163 | -2.40e-4 | +1.85e-4 | -2.74e-5 | -5.09e-6 |
| 75 | 3.00e-1 | 1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 1.72e-1 | 182 | -3.38e-4 | -3.38e-4 | -3.38e-4 | -3.84e-5 |
| 76 | 3.00e-1 | 2 | 1.81e-1 | 1.88e-1 | 1.84e-1 | 1.88e-1 | 172 | +2.10e-4 | +2.70e-4 | +2.40e-4 | +1.42e-5 |
| 77 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 211 | -3.38e-4 | -3.38e-4 | -3.38e-4 | -2.10e-5 |
| 78 | 3.00e-1 | 2 | 1.86e-1 | 1.92e-1 | 1.89e-1 | 1.86e-1 | 172 | -1.91e-4 | +4.60e-4 | +1.35e-4 | +5.35e-6 |
| 79 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 192 | -2.55e-4 | -2.55e-4 | -2.55e-4 | -2.07e-5 |
| 80 | 3.00e-1 | 2 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 1.87e-1 | 172 | -4.34e-6 | +2.69e-4 | +1.32e-4 | +6.95e-6 |
| 81 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 207 | -2.63e-4 | -2.63e-4 | -2.63e-4 | -2.01e-5 |
| 82 | 3.00e-1 | 1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 1.88e-1 | 202 | +3.10e-4 | +3.10e-4 | +3.10e-4 | +1.30e-5 |
| 83 | 3.00e-1 | 2 | 1.84e-1 | 1.89e-1 | 1.87e-1 | 1.84e-1 | 176 | -1.64e-4 | +2.51e-5 | -6.95e-5 | -3.64e-6 |
| 84 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 160 | -1.96e-5 | -1.96e-5 | -1.96e-5 | -5.24e-6 |
| 85 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 258 | -2.31e-4 | -2.31e-4 | -2.31e-4 | -2.78e-5 |
| 86 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 299 | +6.62e-4 | +6.62e-4 | +6.62e-4 | +4.12e-5 |
| 87 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 290 | -1.78e-5 | -1.78e-5 | -1.78e-5 | +3.53e-5 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 269 | -1.06e-5 | -1.06e-5 | -1.06e-5 | +3.07e-5 |
| 89 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 267 | -7.80e-5 | -7.80e-5 | -7.80e-5 | +1.98e-5 |
| 90 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 294 | +1.65e-5 | +1.65e-5 | +1.65e-5 | +1.95e-5 |
| 91 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 287 | +5.40e-5 | +5.40e-5 | +5.40e-5 | +2.29e-5 |
| 92 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 284 | +2.01e-6 | +2.01e-6 | +2.01e-6 | +2.08e-5 |
| 93 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 240 | -6.97e-6 | -6.97e-6 | -6.97e-6 | +1.81e-5 |
| 94 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 261 | -1.69e-4 | -1.69e-4 | -1.69e-4 | -6.13e-7 |
| 95 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 270 | +6.32e-5 | +6.32e-5 | +6.32e-5 | +5.77e-6 |
| 96 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 273 | +4.56e-5 | +4.56e-5 | +4.56e-5 | +9.75e-6 |
| 97 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 257 | +3.17e-5 | +3.17e-5 | +3.17e-5 | +1.19e-5 |
| 98 | 3.00e-1 | 2 | 2.02e-1 | 2.03e-1 | 2.03e-1 | 2.02e-1 | 232 | -7.10e-5 | -3.13e-5 | -5.12e-5 | +1.51e-7 |
| 100 | 3.00e-2 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 315 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -1.17e-5 |
| 101 | 3.00e-2 | 2 | 2.09e-2 | 2.14e-1 | 1.17e-1 | 2.09e-2 | 246 | -9.46e-3 | +3.48e-4 | -4.56e-3 | -9.25e-4 |
| 103 | 3.00e-2 | 2 | 2.14e-2 | 2.50e-2 | 2.32e-2 | 2.50e-2 | 232 | +7.23e-5 | +6.78e-4 | +3.75e-4 | -6.75e-4 |
| 105 | 3.00e-2 | 2 | 2.35e-2 | 2.64e-2 | 2.49e-2 | 2.64e-2 | 232 | -2.10e-4 | +5.08e-4 | +1.49e-4 | -5.14e-4 |
| 106 | 3.00e-2 | 1 | 2.46e-2 | 2.46e-2 | 2.46e-2 | 2.46e-2 | 257 | -2.80e-4 | -2.80e-4 | -2.80e-4 | -4.91e-4 |
| 107 | 3.00e-2 | 1 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 2.62e-2 | 262 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -4.18e-4 |
| 108 | 3.00e-2 | 1 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 2.75e-2 | 272 | +1.85e-4 | +1.85e-4 | +1.85e-4 | -3.57e-4 |
| 109 | 3.00e-2 | 1 | 2.85e-2 | 2.85e-2 | 2.85e-2 | 2.85e-2 | 262 | +1.26e-4 | +1.26e-4 | +1.26e-4 | -3.09e-4 |
| 110 | 3.00e-2 | 1 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 252 | +4.18e-5 | +4.18e-5 | +4.18e-5 | -2.74e-4 |
| 111 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 278 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -2.36e-4 |
| 112 | 3.00e-2 | 1 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 3.09e-2 | 258 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -1.96e-4 |
| 113 | 3.00e-2 | 1 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 3.10e-2 | 283 | +1.32e-5 | +1.32e-5 | +1.32e-5 | -1.76e-4 |
| 114 | 3.00e-2 | 1 | 3.29e-2 | 3.29e-2 | 3.29e-2 | 3.29e-2 | 255 | +2.33e-4 | +2.33e-4 | +2.33e-4 | -1.35e-4 |
| 115 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 252 | -5.96e-5 | -5.96e-5 | -5.96e-5 | -1.27e-4 |
| 116 | 3.00e-2 | 1 | 3.21e-2 | 3.21e-2 | 3.21e-2 | 3.21e-2 | 273 | -3.00e-5 | -3.00e-5 | -3.00e-5 | -1.17e-4 |
| 117 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 255 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -8.50e-5 |
| 118 | 3.00e-2 | 1 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 3.28e-2 | 263 | -1.22e-4 | -1.22e-4 | -1.22e-4 | -8.87e-5 |
| 119 | 3.00e-2 | 1 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 3.54e-2 | 251 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -4.94e-5 |
| 120 | 3.00e-2 | 2 | 3.39e-2 | 3.71e-2 | 3.55e-2 | 3.71e-2 | 250 | -1.56e-4 | +3.58e-4 | +1.01e-4 | -1.83e-5 |
| 122 | 3.00e-2 | 1 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 3.59e-2 | 350 | -9.34e-5 | -9.34e-5 | -9.34e-5 | -2.58e-5 |
| 123 | 3.00e-2 | 1 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 260 | +5.92e-4 | +5.92e-4 | +5.92e-4 | +3.59e-5 |
| 124 | 3.00e-2 | 1 | 3.84e-2 | 3.84e-2 | 3.84e-2 | 3.84e-2 | 285 | -3.02e-4 | -3.02e-4 | -3.02e-4 | +2.11e-6 |
| 125 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 249 | +1.89e-4 | +1.89e-4 | +1.89e-4 | +2.08e-5 |
| 126 | 3.00e-2 | 2 | 3.92e-2 | 3.96e-2 | 3.94e-2 | 3.96e-2 | 246 | -1.11e-4 | +4.67e-5 | -3.21e-5 | +1.15e-5 |
| 128 | 3.00e-2 | 2 | 3.94e-2 | 4.39e-2 | 4.17e-2 | 4.39e-2 | 213 | -1.57e-5 | +5.00e-4 | +2.42e-4 | +5.80e-5 |
| 129 | 3.00e-2 | 1 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 3.91e-2 | 255 | -4.52e-4 | -4.52e-4 | -4.52e-4 | +6.96e-6 |
| 130 | 3.00e-2 | 1 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 221 | +3.77e-4 | +3.77e-4 | +3.77e-4 | +4.39e-5 |
| 131 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 209 | -1.29e-4 | -1.29e-4 | -1.29e-4 | +2.66e-5 |
| 132 | 3.00e-2 | 1 | 4.00e-2 | 4.00e-2 | 4.00e-2 | 4.00e-2 | 250 | -1.34e-4 | -1.34e-4 | -1.34e-4 | +1.06e-5 |
| 133 | 3.00e-2 | 1 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 260 | +4.07e-4 | +4.07e-4 | +4.07e-4 | +5.03e-5 |
| 134 | 3.00e-2 | 1 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 260 | +6.02e-5 | +6.02e-5 | +6.02e-5 | +5.13e-5 |
| 135 | 3.00e-2 | 2 | 4.69e-2 | 4.72e-2 | 4.71e-2 | 4.72e-2 | 242 | +2.41e-5 | +1.42e-4 | +8.30e-5 | +5.67e-5 |
| 136 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 252 | -6.92e-5 | -6.92e-5 | -6.92e-5 | +4.41e-5 |
| 137 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 267 | +1.50e-4 | +1.50e-4 | +1.50e-4 | +5.47e-5 |
| 138 | 3.00e-2 | 1 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 5.03e-2 | 280 | +1.43e-4 | +1.43e-4 | +1.43e-4 | +6.36e-5 |
| 139 | 3.00e-2 | 1 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 5.07e-2 | 264 | +3.16e-5 | +3.16e-5 | +3.16e-5 | +6.04e-5 |
| 140 | 3.00e-2 | 1 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 5.13e-2 | 275 | +4.80e-5 | +4.80e-5 | +4.80e-5 | +5.92e-5 |
| 141 | 3.00e-2 | 1 | 5.24e-2 | 5.24e-2 | 5.24e-2 | 5.24e-2 | 259 | +7.65e-5 | +7.65e-5 | +7.65e-5 | +6.09e-5 |
| 142 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 245 | -1.56e-4 | -1.56e-4 | -1.56e-4 | +3.93e-5 |
| 143 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 194 | +1.19e-4 | +1.19e-4 | +1.19e-4 | +4.72e-5 |
| 144 | 3.00e-2 | 2 | 4.68e-2 | 5.02e-2 | 4.85e-2 | 5.02e-2 | 203 | -4.52e-4 | +3.40e-4 | -5.62e-5 | +3.15e-5 |
| 145 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 230 | -1.66e-4 | -1.66e-4 | -1.66e-4 | +1.18e-5 |
| 146 | 3.00e-2 | 1 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 5.15e-2 | 245 | +2.64e-4 | +2.64e-4 | +2.64e-4 | +3.70e-5 |
| 147 | 3.00e-2 | 1 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 258 | +1.11e-4 | +1.11e-4 | +1.11e-4 | +4.44e-5 |
| 148 | 3.00e-2 | 1 | 5.46e-2 | 5.46e-2 | 5.46e-2 | 5.46e-2 | 253 | +1.21e-4 | +1.21e-4 | +1.21e-4 | +5.21e-5 |
| 149 | 3.00e-3 | 2 | 5.51e-2 | 5.64e-2 | 5.57e-2 | 5.64e-2 | 221 | +2.90e-5 | +1.07e-4 | +6.78e-5 | +5.55e-5 |
| 151 | 3.00e-3 | 2 | 6.12e-3 | 5.32e-2 | 2.97e-2 | 6.12e-3 | 234 | -9.24e-3 | -1.93e-4 | -4.72e-3 | -8.96e-4 |
| 152 | 3.00e-3 | 1 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 5.14e-3 | 266 | -6.54e-4 | -6.54e-4 | -6.54e-4 | -8.72e-4 |
| 153 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 276 | +3.07e-4 | +3.07e-4 | +3.07e-4 | -7.54e-4 |
| 154 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 256 | +2.14e-5 | +2.14e-5 | +2.14e-5 | -6.77e-4 |
| 155 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 249 | -2.60e-5 | -2.60e-5 | -2.60e-5 | -6.12e-4 |
| 156 | 3.00e-3 | 1 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 5.48e-3 | 250 | -8.46e-5 | -8.46e-5 | -8.46e-5 | -5.59e-4 |
| 157 | 3.00e-3 | 1 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 5.38e-3 | 252 | -6.66e-5 | -6.66e-5 | -6.66e-5 | -5.10e-4 |
| 158 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 292 | +2.03e-4 | +2.03e-4 | +2.03e-4 | -4.38e-4 |
| 159 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 260 | +1.56e-4 | +1.56e-4 | +1.56e-4 | -3.79e-4 |
| 160 | 3.00e-3 | 2 | 5.32e-3 | 5.86e-3 | 5.59e-3 | 5.32e-3 | 210 | -4.61e-4 | -6.91e-5 | -2.65e-4 | -3.59e-4 |
| 161 | 3.00e-3 | 1 | 5.33e-3 | 5.33e-3 | 5.33e-3 | 5.33e-3 | 224 | +7.60e-6 | +7.60e-6 | +7.60e-6 | -3.23e-4 |
| 162 | 3.00e-3 | 1 | 5.33e-3 | 5.33e-3 | 5.33e-3 | 5.33e-3 | 210 | +3.87e-6 | +3.87e-6 | +3.87e-6 | -2.90e-4 |
| 163 | 3.00e-3 | 1 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 5.37e-3 | 251 | +3.28e-5 | +3.28e-5 | +3.28e-5 | -2.58e-4 |
| 164 | 3.00e-3 | 2 | 5.72e-3 | 5.90e-3 | 5.81e-3 | 5.72e-3 | 211 | -1.49e-4 | +3.99e-4 | +1.25e-4 | -1.88e-4 |
| 166 | 3.00e-3 | 2 | 5.46e-3 | 6.05e-3 | 5.76e-3 | 6.05e-3 | 213 | -1.64e-4 | +4.84e-4 | +1.60e-4 | -1.18e-4 |
| 167 | 3.00e-3 | 1 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 211 | -4.67e-4 | -4.67e-4 | -4.67e-4 | -1.53e-4 |
| 168 | 3.00e-3 | 1 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 5.34e-3 | 210 | -1.30e-4 | -1.30e-4 | -1.30e-4 | -1.51e-4 |
| 169 | 3.00e-3 | 1 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 220 | +1.45e-4 | +1.45e-4 | +1.45e-4 | -1.21e-4 |
| 170 | 3.00e-3 | 2 | 5.55e-3 | 5.97e-3 | 5.76e-3 | 5.97e-3 | 200 | +2.69e-5 | +3.66e-4 | +1.97e-4 | -5.92e-5 |
| 171 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 225 | -3.32e-4 | -3.32e-4 | -3.32e-4 | -8.65e-5 |
| 172 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 227 | +1.31e-4 | +1.31e-4 | +1.31e-4 | -6.48e-5 |
| 173 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 223 | +5.07e-5 | +5.07e-5 | +5.07e-5 | -5.32e-5 |
| 174 | 3.00e-3 | 2 | 5.55e-3 | 5.87e-3 | 5.71e-3 | 5.87e-3 | 200 | -1.74e-4 | +2.78e-4 | +5.21e-5 | -3.09e-5 |
| 175 | 3.00e-3 | 1 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 5.54e-3 | 237 | -2.43e-4 | -2.43e-4 | -2.43e-4 | -5.22e-5 |
| 176 | 3.00e-3 | 1 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 224 | +2.58e-4 | +2.58e-4 | +2.58e-4 | -2.11e-5 |
| 177 | 3.00e-3 | 1 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 247 | -7.45e-5 | -7.45e-5 | -7.45e-5 | -2.65e-5 |
| 178 | 3.00e-3 | 1 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 6.07e-3 | 267 | +1.97e-4 | +1.97e-4 | +1.97e-4 | -4.15e-6 |
| 179 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 250 | +1.72e-4 | +1.72e-4 | +1.72e-4 | +1.35e-5 |
| 180 | 3.00e-3 | 1 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 6.22e-3 | 270 | -6.77e-5 | -6.77e-5 | -6.77e-5 | +5.38e-6 |
| 181 | 3.00e-3 | 2 | 6.33e-3 | 6.39e-3 | 6.36e-3 | 6.33e-3 | 209 | -4.55e-5 | +1.08e-4 | +3.12e-5 | +9.51e-6 |
| 182 | 3.00e-3 | 1 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 5.74e-3 | 248 | -3.95e-4 | -3.95e-4 | -3.95e-4 | -3.10e-5 |
| 183 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 242 | +2.44e-4 | +2.44e-4 | +2.44e-4 | -3.51e-6 |
| 184 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 233 | +3.71e-5 | +3.71e-5 | +3.71e-5 | +5.52e-7 |
| 185 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 262 | -1.25e-5 | -1.25e-5 | -1.25e-5 | -7.48e-7 |
| 186 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 272 | +2.37e-4 | +2.37e-4 | +2.37e-4 | +2.30e-5 |
| 187 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 286 | -3.48e-5 | -3.48e-5 | -3.48e-5 | +1.72e-5 |
| 188 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 225 | +9.86e-5 | +9.86e-5 | +9.86e-5 | +2.54e-5 |
| 189 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 270 | -2.63e-4 | -2.63e-4 | -2.63e-4 | -3.45e-6 |
| 190 | 3.00e-3 | 2 | 6.22e-3 | 6.63e-3 | 6.42e-3 | 6.22e-3 | 203 | -3.17e-4 | +3.04e-4 | -6.50e-6 | -7.13e-6 |
| 191 | 3.00e-3 | 1 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 5.68e-3 | 250 | -3.59e-4 | -3.59e-4 | -3.59e-4 | -4.23e-5 |
| 192 | 3.00e-3 | 1 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 6.47e-3 | 264 | +4.92e-4 | +4.92e-4 | +4.92e-4 | +1.11e-5 |
| 193 | 3.00e-3 | 2 | 6.16e-3 | 6.30e-3 | 6.23e-3 | 6.16e-3 | 203 | -1.24e-4 | -1.08e-4 | -1.16e-4 | -1.29e-5 |
| 195 | 3.00e-3 | 2 | 6.17e-3 | 6.85e-3 | 6.51e-3 | 6.85e-3 | 214 | +3.04e-6 | +4.90e-4 | +2.46e-4 | +3.88e-5 |
| 196 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 236 | -5.18e-4 | -5.18e-4 | -5.18e-4 | -1.69e-5 |
| 197 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 270 | +1.77e-4 | +1.77e-4 | +1.77e-4 | +2.50e-6 |
| 198 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 270 | +2.36e-4 | +2.36e-4 | +2.36e-4 | +2.58e-5 |
| 199 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 236 | -2.02e-5 | -2.02e-5 | -2.02e-5 | +2.12e-5 |

