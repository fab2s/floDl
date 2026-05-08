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
| cpu-async | 0.052505 | 0.9176 | +0.0051 | 1837.5 | 533 | 79.7 | 100% | 100% | 100% | 13.1 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9176 | cpu-async | - | - |

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
| cpu-async | 1.9837 | 0.7453 | 0.5651 | 0.4891 | 0.4558 | 0.5001 | 0.4838 | 0.4759 | 0.4562 | 0.4699 | 0.2028 | 0.1663 | 0.1464 | 0.1341 | 0.1256 | 0.0705 | 0.0641 | 0.0581 | 0.0545 | 0.0525 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.3997 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3029 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2974 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 388 | 385 | 378 | 373 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 518.8 | 1.0 | epoch-boundary(56) |
| cpu-async | gpu1 | 972.5 | 0.9 | epoch-boundary(105) |
| cpu-async | gpu2 | 518.9 | 0.8 | epoch-boundary(56) |
| cpu-async | gpu0 | 702.9 | 0.8 | unexplained |
| cpu-async | gpu2 | 777.1 | 0.7 | epoch-boundary(84) |
| cpu-async | gpu1 | 991.2 | 0.7 | epoch-boundary(107) |
| cpu-async | gpu1 | 1476.2 | 0.7 | epoch-boundary(160) |
| cpu-async | gpu1 | 1366.4 | 0.7 | unexplained |
| cpu-async | gpu2 | 1366.4 | 0.7 | unexplained |
| cpu-async | gpu1 | 777.2 | 0.6 | epoch-boundary(84) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.6s | 0.8s | 1.4s |
| resnet-graph | cpu-async | gpu1 | 5.4s | 0.0s | 0.0s | 1.3s | 7.7s |
| resnet-graph | cpu-async | gpu2 | 2.1s | 0.0s | 0.0s | 1.3s | 4.0s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 306 | 0 | 533 | 79.7 | 1578/9660 | 533 | 79.7 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 224.6 | 12.2% |

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
| resnet-graph | cpu-async | 188 | 533 | 0 | 2.10e-3 | -1.04e-3 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 533 | 8.12e-2 | 6.11e-2 | 1.80e-3 | 4.53e-1 | 27.6 | -2.81e-4 | 2.38e-3 |
| resnet-graph | cpu-async | 1 | 533 | 8.16e-2 | 6.16e-2 | 1.76e-3 | 4.24e-1 | 33.0 | -2.87e-4 | 2.28e-3 |
| resnet-graph | cpu-async | 2 | 533 | 8.22e-2 | 6.23e-2 | 1.88e-3 | 4.27e-1 | 39.4 | -3.02e-4 | 2.38e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9972 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9964 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9967 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 63 (0,1,2,3,4,5,6,7…149,150) | 0 (—) | — | 0,1,2,3,4,5,6,7…149,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 7 | 7 |
| resnet-graph | cpu-async | 0e0 | 5 | 3 | 3 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–97 | 345 | +0.209 |
| resnet-graph | cpu-async | 3.00e-2 | 99–149 | 57 | +0.271 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 127 | +0.139 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 531 | -0.004 | 187 | +0.405 | +0.549 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 532 | 3.39e1–8.07e1 | 6.57e1 | 2.44e-3 | 3.73e-3 | 2.68e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–97 | 347 | 40–76570 | +1.260e-5 | 0.469 | +1.275e-5 | 0.494 | 91 | +1.467e-5 | 0.753 | 30–964 | +1.123e-3 | 0.753 |
| resnet-graph | cpu-async | 3.00e-1 | 1–97 (post-transient, skipped 1) | 335 | 950–76570 | +1.269e-5 | 0.529 | +1.281e-5 | 0.557 | 90 | +1.467e-5 | 0.747 | 70–964 | +1.114e-3 | 0.870 |
| resnet-graph | cpu-async | 3.00e-2 | 99–149 | 58 | 77432–116907 | +1.061e-5 | 0.096 | +1.055e-5 | 0.096 | 47 | +1.363e-5 | 0.198 | 511–869 | -8.483e-4 | 0.034 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 128 | 117559–156144 | -2.451e-5 | 0.552 | -2.475e-5 | 0.561 | 50 | -2.405e-5 | 0.407 | 86–703 | +1.924e-3 | 0.671 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–97 | +1.123e-3 | r0: +1.102e-3, r1: +1.127e-3, r2: +1.140e-3 | r0: 0.754, r1: 0.747, r2: 0.743 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–97 (post-transient, skipped 1) | +1.114e-3 | r0: +1.095e-3, r1: +1.117e-3, r2: +1.130e-3 | r0: 0.874, r1: 0.865, r2: 0.853 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 99–149 | -8.483e-4 | r0: -8.624e-4, r1: -8.780e-4, r2: -8.052e-4 | r0: 0.036, r1: 0.037, r2: 0.030 | 1.09× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +1.924e-3 | r0: +1.891e-3, r1: +1.954e-3, r2: +1.928e-3 | r0: 0.664, r1: 0.675, r2: 0.664 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇███████████▄▅▅▅▅▅▅▅▅▆▆▅▂▂▂▂▂▂▂▁▁▁▁▁` | `▁▆▇▇▆▇▆▇▇▇▇▇▇██████████▇▇██████████▇▇▇█████▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 3.74e-2 | 4.53e-1 | 1.12e-1 | 7.73e-2 | 32 | -9.43e-2 | +2.82e-2 | -1.62e-2 | -9.66e-3 |
| 1 | 3.00e-1 | 8 | 6.98e-2 | 1.20e-1 | 8.36e-2 | 8.81e-2 | 34 | -1.80e-2 | +5.75e-3 | -6.36e-4 | -3.44e-3 |
| 2 | 3.00e-1 | 8 | 7.53e-2 | 1.32e-1 | 9.05e-2 | 8.98e-2 | 30 | -1.81e-2 | +5.79e-3 | -1.07e-3 | -1.78e-3 |
| 3 | 3.00e-1 | 9 | 7.50e-2 | 1.33e-1 | 9.53e-2 | 1.12e-1 | 50 | -2.19e-2 | +5.41e-3 | -6.89e-4 | -7.15e-4 |
| 4 | 3.00e-1 | 4 | 9.41e-2 | 1.55e-1 | 1.13e-1 | 1.02e-1 | 41 | -1.24e-2 | +3.08e-3 | -1.82e-3 | -1.07e-3 |
| 5 | 3.00e-1 | 7 | 9.61e-2 | 1.35e-1 | 1.05e-1 | 1.00e-1 | 41 | -7.69e-3 | +3.44e-3 | -5.73e-4 | -7.62e-4 |
| 6 | 3.00e-1 | 5 | 9.55e-2 | 1.39e-1 | 1.09e-1 | 1.05e-1 | 46 | -9.12e-3 | +4.05e-3 | -5.97e-4 | -6.76e-4 |
| 7 | 3.00e-1 | 9 | 9.07e-2 | 1.44e-1 | 1.02e-1 | 9.13e-2 | 36 | -8.77e-3 | +3.30e-3 | -8.92e-4 | -7.52e-4 |
| 8 | 3.00e-1 | 3 | 1.12e-1 | 1.40e-1 | 1.23e-1 | 1.17e-1 | 55 | -4.10e-3 | +4.95e-3 | +5.46e-4 | -4.37e-4 |
| 9 | 3.00e-1 | 5 | 1.01e-1 | 1.43e-1 | 1.12e-1 | 1.01e-1 | 47 | -6.26e-3 | +2.18e-3 | -9.85e-4 | -6.61e-4 |
| 10 | 3.00e-1 | 5 | 9.68e-2 | 1.43e-1 | 1.08e-1 | 1.00e-1 | 44 | -8.66e-3 | +3.81e-3 | -8.61e-4 | -7.24e-4 |
| 11 | 3.00e-1 | 8 | 9.57e-2 | 1.47e-1 | 1.07e-1 | 9.99e-2 | 41 | -7.15e-3 | +4.50e-3 | -5.75e-4 | -6.15e-4 |
| 12 | 3.00e-1 | 4 | 9.56e-2 | 1.36e-1 | 1.08e-1 | 1.01e-1 | 44 | -8.61e-3 | +3.72e-3 | -8.72e-4 | -7.02e-4 |
| 13 | 3.00e-1 | 6 | 9.32e-2 | 1.37e-1 | 1.06e-1 | 1.09e-1 | 45 | -9.33e-3 | +3.57e-3 | -3.61e-4 | -4.49e-4 |
| 14 | 3.00e-1 | 6 | 9.74e-2 | 1.33e-1 | 1.06e-1 | 9.74e-2 | 39 | -5.64e-3 | +2.54e-3 | -7.56e-4 | -5.82e-4 |
| 15 | 3.00e-1 | 8 | 8.83e-2 | 1.42e-1 | 1.01e-1 | 1.02e-1 | 42 | -1.39e-2 | +4.27e-3 | -7.74e-4 | -5.24e-4 |
| 16 | 3.00e-1 | 5 | 8.60e-2 | 1.31e-1 | 1.03e-1 | 8.60e-2 | 32 | -9.46e-3 | +3.59e-3 | -1.83e-3 | -1.13e-3 |
| 17 | 3.00e-1 | 7 | 8.22e-2 | 1.34e-1 | 9.44e-2 | 9.44e-2 | 36 | -1.58e-2 | +5.91e-3 | -8.30e-4 | -8.23e-4 |
| 18 | 3.00e-1 | 7 | 8.95e-2 | 1.46e-1 | 9.95e-2 | 9.04e-2 | 33 | -1.15e-2 | +4.94e-3 | -1.08e-3 | -8.86e-4 |
| 19 | 3.00e-1 | 7 | 8.56e-2 | 1.41e-1 | 1.03e-1 | 1.02e-1 | 44 | -1.41e-2 | +5.56e-3 | -7.47e-4 | -6.99e-4 |
| 20 | 3.00e-1 | 6 | 9.21e-2 | 1.46e-1 | 1.04e-1 | 9.41e-2 | 36 | -1.09e-2 | +4.21e-3 | -1.16e-3 | -8.77e-4 |
| 21 | 3.00e-1 | 7 | 8.84e-2 | 1.41e-1 | 1.02e-1 | 1.02e-1 | 42 | -1.23e-2 | +5.15e-3 | -5.18e-4 | -5.94e-4 |
| 22 | 3.00e-1 | 5 | 9.82e-2 | 1.46e-1 | 1.10e-1 | 9.82e-2 | 41 | -7.53e-3 | +3.81e-3 | -1.00e-3 | -7.69e-4 |
| 23 | 3.00e-1 | 6 | 8.69e-2 | 1.39e-1 | 1.01e-1 | 9.89e-2 | 41 | -1.46e-2 | +4.01e-3 | -1.20e-3 | -8.56e-4 |
| 24 | 3.00e-1 | 8 | 9.15e-2 | 1.51e-1 | 1.08e-1 | 1.05e-1 | 43 | -9.06e-3 | +4.88e-3 | -4.54e-4 | -5.38e-4 |
| 25 | 3.00e-1 | 4 | 8.96e-2 | 1.43e-1 | 1.09e-1 | 8.96e-2 | 36 | -7.72e-3 | +3.77e-3 | -1.99e-3 | -1.09e-3 |
| 26 | 3.00e-1 | 8 | 8.53e-2 | 1.39e-1 | 9.63e-2 | 8.61e-2 | 33 | -1.45e-2 | +5.33e-3 | -1.18e-3 | -1.09e-3 |
| 27 | 3.00e-1 | 9 | 8.83e-2 | 1.38e-1 | 9.77e-2 | 9.69e-2 | 33 | -1.19e-2 | +6.65e-3 | -3.76e-4 | -5.29e-4 |
| 28 | 3.00e-1 | 5 | 8.10e-2 | 1.42e-1 | 1.00e-1 | 9.27e-2 | 37 | -1.17e-2 | +4.72e-3 | -1.65e-3 | -9.21e-4 |
| 29 | 3.00e-1 | 9 | 9.30e-2 | 1.37e-1 | 1.02e-1 | 9.33e-2 | 39 | -9.84e-3 | +5.13e-3 | -5.95e-4 | -7.06e-4 |
| 30 | 3.00e-1 | 3 | 1.08e-1 | 1.43e-1 | 1.20e-1 | 1.08e-1 | 45 | -5.48e-3 | +5.21e-3 | -2.12e-4 | -6.23e-4 |
| 31 | 3.00e-1 | 7 | 8.80e-2 | 1.48e-1 | 1.01e-1 | 9.19e-2 | 34 | -1.06e-2 | +3.32e-3 | -1.23e-3 | -8.34e-4 |
| 32 | 3.00e-1 | 6 | 9.19e-2 | 1.46e-1 | 1.05e-1 | 1.01e-1 | 40 | -1.22e-2 | +5.76e-3 | -7.81e-4 | -7.51e-4 |
| 33 | 3.00e-1 | 8 | 8.79e-2 | 1.44e-1 | 9.91e-2 | 9.62e-2 | 35 | -1.41e-2 | +4.93e-3 | -8.73e-4 | -6.95e-4 |
| 34 | 3.00e-1 | 7 | 8.36e-2 | 1.41e-1 | 9.98e-2 | 1.06e-1 | 43 | -1.31e-2 | +4.75e-3 | -6.43e-4 | -4.72e-4 |
| 35 | 3.00e-1 | 6 | 9.32e-2 | 1.45e-1 | 1.10e-1 | 9.97e-2 | 38 | -1.11e-2 | +3.88e-3 | -9.98e-4 | -7.20e-4 |
| 36 | 3.00e-1 | 5 | 8.79e-2 | 1.37e-1 | 1.12e-1 | 1.21e-1 | 54 | -1.31e-2 | +5.39e-3 | -4.23e-4 | -5.10e-4 |
| 37 | 3.00e-1 | 5 | 1.12e-1 | 1.59e-1 | 1.27e-1 | 1.13e-1 | 54 | -3.58e-3 | +2.70e-3 | -5.68e-4 | -5.58e-4 |
| 38 | 3.00e-1 | 5 | 1.06e-1 | 1.53e-1 | 1.18e-1 | 1.09e-1 | 45 | -7.76e-3 | +3.47e-3 | -7.57e-4 | -6.30e-4 |
| 39 | 3.00e-1 | 7 | 1.00e-1 | 1.49e-1 | 1.10e-1 | 1.02e-1 | 41 | -9.30e-3 | +3.44e-3 | -7.82e-4 | -6.62e-4 |
| 40 | 3.00e-1 | 4 | 1.01e-1 | 1.49e-1 | 1.18e-1 | 1.13e-1 | 47 | -8.52e-3 | +4.30e-3 | -4.36e-4 | -5.78e-4 |
| 41 | 3.00e-1 | 6 | 8.84e-2 | 1.47e-1 | 1.03e-1 | 8.84e-2 | 34 | -1.33e-2 | +2.81e-3 | -2.00e-3 | -1.19e-3 |
| 42 | 3.00e-1 | 8 | 8.76e-2 | 1.32e-1 | 1.01e-1 | 1.03e-1 | 42 | -1.20e-2 | +5.77e-3 | -2.77e-4 | -5.89e-4 |
| 43 | 3.00e-1 | 5 | 1.06e-1 | 1.45e-1 | 1.15e-1 | 1.09e-1 | 46 | -7.48e-3 | +4.09e-3 | -5.58e-4 | -5.75e-4 |
| 44 | 3.00e-1 | 5 | 1.00e-1 | 1.42e-1 | 1.14e-1 | 1.10e-1 | 42 | -4.72e-3 | +3.12e-3 | -4.07e-4 | -4.80e-4 |
| 45 | 3.00e-1 | 9 | 8.95e-2 | 1.48e-1 | 9.98e-2 | 9.26e-2 | 34 | -1.43e-2 | +3.59e-3 | -1.08e-3 | -6.92e-4 |
| 46 | 3.00e-1 | 1 | 8.84e-2 | 8.84e-2 | 8.84e-2 | 8.84e-2 | 33 | -1.41e-3 | -1.41e-3 | -1.41e-3 | -7.64e-4 |
| 47 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 330 | +2.92e-3 | +2.92e-3 | +2.92e-3 | -3.96e-4 |
| 48 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 293 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -3.74e-4 |
| 49 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 308 | -2.74e-5 | -2.74e-5 | -2.74e-5 | -3.40e-4 |
| 50 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 280 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -3.21e-4 |
| 51 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 295 | +4.97e-5 | +4.97e-5 | +4.97e-5 | -2.84e-4 |
| 52 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 263 | -1.62e-4 | -1.62e-4 | -1.62e-4 | -2.72e-4 |
| 53 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 277 | +7.27e-5 | +7.27e-5 | +7.27e-5 | -2.37e-4 |
| 54 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 294 | +5.71e-5 | +5.71e-5 | +5.71e-5 | -2.08e-4 |
| 55 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 291 | -1.62e-5 | -1.62e-5 | -1.62e-5 | -1.89e-4 |
| 56 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 277 | -1.32e-4 | -1.32e-4 | -1.32e-4 | -1.83e-4 |
| 58 | 3.00e-1 | 2 | 2.09e-1 | 2.25e-1 | 2.17e-1 | 2.09e-1 | 277 | -2.67e-4 | +3.08e-4 | +2.04e-5 | -1.47e-4 |
| 60 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 340 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -1.16e-4 |
| 61 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 275 | -2.88e-4 | -2.88e-4 | -2.88e-4 | -1.33e-4 |
| 62 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 311 | +2.08e-4 | +2.08e-4 | +2.08e-4 | -9.88e-5 |
| 63 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 258 | -3.07e-4 | -3.07e-4 | -3.07e-4 | -1.20e-4 |
| 64 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 302 | +2.52e-4 | +2.52e-4 | +2.52e-4 | -8.25e-5 |
| 65 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 272 | -2.08e-4 | -2.08e-4 | -2.08e-4 | -9.50e-5 |
| 66 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 293 | +1.20e-4 | +1.20e-4 | +1.20e-4 | -7.35e-5 |
| 67 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 304 | -1.05e-5 | -1.05e-5 | -1.05e-5 | -6.72e-5 |
| 68 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 283 | -1.10e-5 | -1.10e-5 | -1.10e-5 | -6.16e-5 |
| 69 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 287 | -6.67e-5 | -6.67e-5 | -6.67e-5 | -6.21e-5 |
| 70 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 269 | -3.39e-5 | -3.39e-5 | -3.39e-5 | -5.93e-5 |
| 71 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 284 | +1.03e-4 | +1.03e-4 | +1.03e-4 | -4.30e-5 |
| 72 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 303 | +6.01e-5 | +6.01e-5 | +6.01e-5 | -3.27e-5 |
| 73 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 260 | -2.17e-4 | -2.17e-4 | -2.17e-4 | -5.11e-5 |
| 74 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 272 | +1.28e-4 | +1.28e-4 | +1.28e-4 | -3.31e-5 |
| 75 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 267 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -4.15e-5 |
| 76 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 257 | +4.11e-5 | +4.11e-5 | +4.11e-5 | -3.33e-5 |
| 77 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 281 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -1.94e-5 |
| 78 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 272 | -1.39e-4 | -1.39e-4 | -1.39e-4 | -3.13e-5 |
| 79 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 260 | +4.37e-5 | +4.37e-5 | +4.37e-5 | -2.38e-5 |
| 80 | 3.00e-1 | 2 | 2.09e-1 | 2.11e-1 | 2.10e-1 | 2.09e-1 | 262 | -4.54e-5 | +7.37e-5 | +1.42e-5 | -1.72e-5 |
| 82 | 3.00e-1 | 2 | 1.99e-1 | 2.21e-1 | 2.10e-1 | 1.99e-1 | 222 | -4.71e-4 | +1.98e-4 | -1.37e-4 | -4.32e-5 |
| 84 | 3.00e-1 | 2 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 254 | +2.00e-6 | +5.83e-5 | +3.01e-5 | -2.96e-5 |
| 85 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 282 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -7.95e-6 |
| 87 | 3.00e-1 | 2 | 2.07e-1 | 2.30e-1 | 2.18e-1 | 2.07e-1 | 267 | -3.86e-4 | +2.05e-4 | -9.01e-5 | -2.65e-5 |
| 89 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 341 | +3.04e-4 | +3.04e-4 | +3.04e-4 | +6.49e-6 |
| 90 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 341 | -2.67e-5 | -2.67e-5 | -2.67e-5 | +3.17e-6 |
| 91 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 280 | -2.31e-4 | -2.31e-4 | -2.31e-4 | -2.03e-5 |
| 92 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 284 | -4.01e-5 | -4.01e-5 | -4.01e-5 | -2.22e-5 |
| 93 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 294 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -6.99e-6 |
| 94 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 312 | +4.47e-5 | +4.47e-5 | +4.47e-5 | -1.83e-6 |
| 95 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 301 | -5.13e-5 | -5.13e-5 | -5.13e-5 | -6.77e-6 |
| 97 | 3.00e-1 | 2 | 2.00e-1 | 2.24e-1 | 2.12e-1 | 2.00e-1 | 252 | -4.51e-4 | +7.63e-5 | -1.87e-4 | -4.37e-5 |
| 99 | 3.00e-2 | 2 | 2.01e-1 | 2.25e-1 | 2.13e-1 | 2.01e-1 | 244 | -4.56e-4 | +3.69e-4 | -4.35e-5 | -4.78e-5 |
| 101 | 3.00e-2 | 2 | 2.08e-2 | 2.23e-2 | 2.15e-2 | 2.08e-2 | 239 | -7.29e-3 | -2.85e-4 | -3.79e-3 | -7.24e-4 |
| 103 | 3.00e-2 | 2 | 2.30e-2 | 2.48e-2 | 2.39e-2 | 2.30e-2 | 239 | -3.12e-4 | +5.46e-4 | +1.17e-4 | -5.68e-4 |
| 105 | 3.00e-2 | 2 | 2.51e-2 | 2.67e-2 | 2.59e-2 | 2.51e-2 | 239 | -2.68e-4 | +4.78e-4 | +1.05e-4 | -4.44e-4 |
| 106 | 3.00e-2 | 1 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 291 | +2.89e-4 | +2.89e-4 | +2.89e-4 | -3.71e-4 |
| 107 | 3.00e-2 | 1 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 2.67e-2 | 255 | -8.73e-5 | -8.73e-5 | -8.73e-5 | -3.42e-4 |
| 108 | 3.00e-2 | 1 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 287 | +2.65e-4 | +2.65e-4 | +2.65e-4 | -2.82e-4 |
| 109 | 3.00e-2 | 1 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 275 | +5.65e-5 | +5.65e-5 | +5.65e-5 | -2.48e-4 |
| 110 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 270 | +5.37e-5 | +5.37e-5 | +5.37e-5 | -2.18e-4 |
| 111 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 248 | +7.82e-7 | +7.82e-7 | +7.82e-7 | -1.96e-4 |
| 112 | 3.00e-2 | 1 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 3.08e-2 | 267 | +1.37e-4 | +1.37e-4 | +1.37e-4 | -1.62e-4 |
| 113 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 261 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -1.34e-4 |
| 114 | 3.00e-2 | 1 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 259 | +3.42e-5 | +3.42e-5 | +3.42e-5 | -1.18e-4 |
| 115 | 3.00e-2 | 1 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 3.23e-2 | 271 | +3.49e-5 | +3.49e-5 | +3.49e-5 | -1.02e-4 |
| 116 | 3.00e-2 | 1 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 288 | +3.09e-4 | +3.09e-4 | +3.09e-4 | -6.13e-5 |
| 117 | 3.00e-2 | 1 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 3.46e-2 | 271 | -7.92e-5 | -7.92e-5 | -7.92e-5 | -6.31e-5 |
| 118 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 287 | +1.81e-4 | +1.81e-4 | +1.81e-4 | -3.87e-5 |
| 119 | 3.00e-2 | 1 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 280 | +7.95e-5 | +7.95e-5 | +7.95e-5 | -2.69e-5 |
| 120 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 251 | -1.30e-4 | -1.30e-4 | -1.30e-4 | -3.72e-5 |
| 121 | 3.00e-2 | 1 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 222 | -1.92e-4 | -1.92e-4 | -1.92e-4 | -5.27e-5 |
| 122 | 3.00e-2 | 2 | 3.51e-2 | 3.60e-2 | 3.56e-2 | 3.51e-2 | 221 | -1.12e-4 | +1.77e-4 | +3.25e-5 | -3.80e-5 |
| 124 | 3.00e-2 | 2 | 3.71e-2 | 4.22e-2 | 3.97e-2 | 3.71e-2 | 221 | -5.85e-4 | +5.98e-4 | +6.64e-6 | -3.54e-5 |
| 125 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 244 | +2.65e-4 | +2.65e-4 | +2.65e-4 | -5.37e-6 |
| 126 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 239 | +5.31e-5 | +5.31e-5 | +5.31e-5 | +4.74e-7 |
| 127 | 3.00e-2 | 1 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 4.20e-2 | 263 | +1.78e-4 | +1.78e-4 | +1.78e-4 | +1.82e-5 |
| 128 | 3.00e-2 | 1 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 246 | -1.38e-4 | -1.38e-4 | -1.38e-4 | +2.63e-6 |
| 129 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 235 | +5.27e-5 | +5.27e-5 | +5.27e-5 | +7.64e-6 |
| 130 | 3.00e-2 | 1 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 4.21e-2 | 233 | +9.83e-5 | +9.83e-5 | +9.83e-5 | +1.67e-5 |
| 131 | 3.00e-2 | 2 | 3.95e-2 | 4.21e-2 | 4.08e-2 | 3.95e-2 | 208 | -3.09e-4 | +6.07e-6 | -1.51e-4 | -1.68e-5 |
| 132 | 3.00e-2 | 1 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 4.81e-2 | 280 | +7.00e-4 | +7.00e-4 | +7.00e-4 | +5.49e-5 |
| 133 | 3.00e-2 | 1 | 4.88e-2 | 4.88e-2 | 4.88e-2 | 4.88e-2 | 279 | +5.35e-5 | +5.35e-5 | +5.35e-5 | +5.47e-5 |
| 134 | 3.00e-2 | 1 | 4.56e-2 | 4.56e-2 | 4.56e-2 | 4.56e-2 | 229 | -2.94e-4 | -2.94e-4 | -2.94e-4 | +1.99e-5 |
| 135 | 3.00e-2 | 1 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 219 | -1.00e-4 | -1.00e-4 | -1.00e-4 | +7.87e-6 |
| 136 | 3.00e-2 | 1 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 239 | +2.17e-4 | +2.17e-4 | +2.17e-4 | +2.88e-5 |
| 137 | 3.00e-2 | 2 | 4.46e-2 | 4.73e-2 | 4.59e-2 | 4.46e-2 | 198 | -2.99e-4 | +2.29e-5 | -1.38e-4 | -4.49e-6 |
| 138 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 242 | +4.18e-4 | +4.18e-4 | +4.18e-4 | +3.77e-5 |
| 139 | 3.00e-2 | 1 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 4.93e-2 | 245 | +1.92e-7 | +1.92e-7 | +1.92e-7 | +3.40e-5 |
| 140 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 218 | -8.87e-5 | -8.87e-5 | -8.87e-5 | +2.17e-5 |
| 141 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 216 | +4.51e-6 | +4.51e-6 | +4.51e-6 | +2.00e-5 |
| 142 | 3.00e-2 | 2 | 4.56e-2 | 5.35e-2 | 4.95e-2 | 4.56e-2 | 187 | -8.54e-4 | +3.83e-4 | -2.36e-4 | -3.47e-5 |
| 143 | 3.00e-2 | 1 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 5.12e-2 | 223 | +5.18e-4 | +5.18e-4 | +5.18e-4 | +2.05e-5 |
| 144 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 222 | -9.57e-6 | -9.57e-6 | -9.57e-6 | +1.75e-5 |
| 145 | 3.00e-2 | 2 | 4.77e-2 | 5.01e-2 | 4.89e-2 | 4.77e-2 | 187 | -2.69e-4 | -8.14e-5 | -1.75e-4 | -2.01e-5 |
| 146 | 3.00e-2 | 1 | 5.00e-2 | 5.00e-2 | 5.00e-2 | 5.00e-2 | 212 | +2.24e-4 | +2.24e-4 | +2.24e-4 | +4.35e-6 |
| 147 | 3.00e-2 | 2 | 4.96e-2 | 5.16e-2 | 5.06e-2 | 4.96e-2 | 187 | -2.13e-4 | +1.48e-4 | -3.22e-5 | -4.39e-6 |
| 148 | 3.00e-2 | 1 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 5.30e-2 | 225 | +3.00e-4 | +3.00e-4 | +3.00e-4 | +2.60e-5 |
| 149 | 3.00e-2 | 1 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 5.40e-2 | 227 | +7.98e-5 | +7.98e-5 | +7.98e-5 | +3.14e-5 |
| 150 | 3.00e-3 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 241 | +1.61e-4 | +1.61e-4 | +1.61e-4 | +4.44e-5 |
| 151 | 3.00e-3 | 2 | 4.75e-3 | 5.57e-3 | 5.16e-3 | 4.75e-3 | 175 | -9.74e-3 | -9.17e-4 | -5.33e-3 | -9.33e-4 |
| 152 | 3.00e-3 | 1 | 5.35e-3 | 5.35e-3 | 5.35e-3 | 5.35e-3 | 234 | +5.11e-4 | +5.11e-4 | +5.11e-4 | -7.88e-4 |
| 153 | 3.00e-3 | 2 | 4.50e-3 | 5.20e-3 | 4.85e-3 | 4.50e-3 | 175 | -8.19e-4 | -1.37e-4 | -4.78e-4 | -7.33e-4 |
| 154 | 3.00e-3 | 1 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 190 | +3.33e-4 | +3.33e-4 | +3.33e-4 | -6.26e-4 |
| 155 | 3.00e-3 | 1 | 4.75e-3 | 4.75e-3 | 4.75e-3 | 4.75e-3 | 202 | -4.50e-5 | -4.50e-5 | -4.50e-5 | -5.68e-4 |
| 156 | 3.00e-3 | 1 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 4.66e-3 | 175 | -1.12e-4 | -1.12e-4 | -1.12e-4 | -5.22e-4 |
| 157 | 3.00e-3 | 2 | 4.59e-3 | 5.45e-3 | 5.02e-3 | 4.59e-3 | 172 | -1.00e-3 | +5.90e-4 | -2.05e-4 | -4.70e-4 |
| 158 | 3.00e-3 | 1 | 5.21e-3 | 5.21e-3 | 5.21e-3 | 5.21e-3 | 225 | +5.59e-4 | +5.59e-4 | +5.59e-4 | -3.67e-4 |
| 159 | 3.00e-3 | 2 | 4.39e-3 | 4.59e-3 | 4.49e-3 | 4.39e-3 | 155 | -7.20e-4 | -2.85e-4 | -5.03e-4 | -3.91e-4 |
| 160 | 3.00e-3 | 2 | 4.18e-3 | 4.79e-3 | 4.48e-3 | 4.18e-3 | 152 | -8.86e-4 | +4.61e-4 | -2.12e-4 | -3.64e-4 |
| 161 | 3.00e-3 | 1 | 4.79e-3 | 4.79e-3 | 4.79e-3 | 4.79e-3 | 188 | +7.24e-4 | +7.24e-4 | +7.24e-4 | -2.55e-4 |
| 162 | 3.00e-3 | 2 | 4.26e-3 | 5.14e-3 | 4.70e-3 | 4.26e-3 | 152 | -1.23e-3 | +3.58e-4 | -4.34e-4 | -2.97e-4 |
| 163 | 3.00e-3 | 2 | 4.56e-3 | 4.64e-3 | 4.60e-3 | 4.56e-3 | 153 | -1.16e-4 | +4.79e-4 | +1.81e-4 | -2.09e-4 |
| 164 | 3.00e-3 | 1 | 4.94e-3 | 4.94e-3 | 4.94e-3 | 4.94e-3 | 198 | +4.10e-4 | +4.10e-4 | +4.10e-4 | -1.47e-4 |
| 165 | 3.00e-3 | 2 | 4.51e-3 | 5.14e-3 | 4.82e-3 | 4.51e-3 | 153 | -8.50e-4 | +1.93e-4 | -3.28e-4 | -1.87e-4 |
| 166 | 3.00e-3 | 1 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 5.06e-3 | 197 | +5.86e-4 | +5.86e-4 | +5.86e-4 | -1.10e-4 |
| 167 | 3.00e-3 | 2 | 4.33e-3 | 4.88e-3 | 4.60e-3 | 4.33e-3 | 142 | -8.42e-4 | -2.02e-4 | -5.22e-4 | -1.91e-4 |
| 168 | 3.00e-3 | 2 | 4.47e-3 | 4.97e-3 | 4.72e-3 | 4.47e-3 | 142 | -7.56e-4 | +7.38e-4 | -9.09e-6 | -1.64e-4 |
| 169 | 3.00e-3 | 1 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 4.80e-3 | 178 | +4.02e-4 | +4.02e-4 | +4.02e-4 | -1.07e-4 |
| 170 | 3.00e-3 | 2 | 4.47e-3 | 4.99e-3 | 4.73e-3 | 4.47e-3 | 142 | -7.77e-4 | +1.93e-4 | -2.92e-4 | -1.47e-4 |
| 171 | 3.00e-3 | 2 | 4.24e-3 | 4.83e-3 | 4.54e-3 | 4.24e-3 | 131 | -1.01e-3 | +4.62e-4 | -2.71e-4 | -1.78e-4 |
| 172 | 3.00e-3 | 2 | 4.31e-3 | 4.64e-3 | 4.47e-3 | 4.31e-3 | 131 | -5.62e-4 | +5.43e-4 | -9.66e-6 | -1.52e-4 |
| 173 | 3.00e-3 | 2 | 4.27e-3 | 4.59e-3 | 4.43e-3 | 4.27e-3 | 131 | -5.53e-4 | +4.12e-4 | -7.03e-5 | -1.41e-4 |
| 174 | 3.00e-3 | 2 | 4.19e-3 | 4.65e-3 | 4.42e-3 | 4.19e-3 | 131 | -7.84e-4 | +5.21e-4 | -1.31e-4 | -1.46e-4 |
| 175 | 3.00e-3 | 1 | 4.64e-3 | 4.64e-3 | 4.64e-3 | 4.64e-3 | 156 | +6.50e-4 | +6.50e-4 | +6.50e-4 | -6.62e-5 |
| 176 | 3.00e-3 | 3 | 4.24e-3 | 5.01e-3 | 4.53e-3 | 4.24e-3 | 122 | -1.37e-3 | +8.12e-4 | -3.61e-4 | -1.55e-4 |
| 177 | 3.00e-3 | 1 | 4.74e-3 | 4.74e-3 | 4.74e-3 | 4.74e-3 | 161 | +6.90e-4 | +6.90e-4 | +6.90e-4 | -7.04e-5 |
| 178 | 3.00e-3 | 2 | 4.00e-3 | 4.69e-3 | 4.35e-3 | 4.00e-3 | 111 | -1.45e-3 | -6.01e-5 | -7.55e-4 | -2.07e-4 |
| 179 | 3.00e-3 | 3 | 3.94e-3 | 4.57e-3 | 4.25e-3 | 3.94e-3 | 105 | -1.42e-3 | +5.37e-4 | -1.32e-4 | -2.05e-4 |
| 180 | 3.00e-3 | 2 | 3.82e-3 | 4.62e-3 | 4.22e-3 | 4.62e-3 | 150 | -2.98e-4 | +1.28e-3 | +4.90e-4 | -6.55e-5 |
| 181 | 3.00e-3 | 2 | 3.74e-3 | 4.88e-3 | 4.31e-3 | 3.74e-3 | 99 | -2.70e-3 | +3.10e-4 | -1.19e-3 | -2.95e-4 |
| 182 | 3.00e-3 | 3 | 3.56e-3 | 4.24e-3 | 3.82e-3 | 3.66e-3 | 100 | -1.88e-3 | +9.57e-4 | -2.17e-4 | -2.79e-4 |
| 183 | 3.00e-3 | 3 | 3.87e-3 | 4.25e-3 | 4.01e-3 | 3.87e-3 | 108 | -7.26e-4 | +1.17e-3 | +1.02e-4 | -1.88e-4 |
| 184 | 3.00e-3 | 2 | 3.70e-3 | 4.61e-3 | 4.16e-3 | 3.70e-3 | 101 | -2.19e-3 | +1.31e-3 | -4.41e-4 | -2.54e-4 |
| 185 | 3.00e-3 | 3 | 3.83e-3 | 4.58e-3 | 4.11e-3 | 3.83e-3 | 101 | -1.53e-3 | +1.40e-3 | -1.22e-4 | -2.33e-4 |
| 186 | 3.00e-3 | 2 | 3.95e-3 | 4.33e-3 | 4.14e-3 | 3.95e-3 | 94 | -9.59e-4 | +9.22e-4 | -1.85e-5 | -2.01e-4 |
| 187 | 3.00e-3 | 3 | 3.36e-3 | 4.18e-3 | 3.78e-3 | 3.36e-3 | 76 | -1.61e-3 | +4.23e-4 | -7.37e-4 | -3.66e-4 |
| 188 | 3.00e-3 | 3 | 3.32e-3 | 4.09e-3 | 3.58e-3 | 3.32e-3 | 76 | -2.69e-3 | +1.72e-3 | -3.39e-4 | -3.74e-4 |
| 189 | 3.00e-3 | 5 | 3.00e-3 | 4.22e-3 | 3.52e-3 | 3.10e-3 | 68 | -3.64e-3 | +2.00e-3 | -4.93e-4 | -4.55e-4 |
| 190 | 3.00e-3 | 2 | 3.08e-3 | 4.25e-3 | 3.66e-3 | 3.08e-3 | 67 | -4.81e-3 | +2.77e-3 | -1.02e-3 | -6.00e-4 |
| 191 | 3.00e-3 | 5 | 3.15e-3 | 4.22e-3 | 3.40e-3 | 3.15e-3 | 63 | -3.73e-3 | +2.54e-3 | -3.73e-4 | -5.16e-4 |
| 192 | 3.00e-3 | 3 | 2.65e-3 | 3.94e-3 | 3.16e-3 | 2.65e-3 | 58 | -5.32e-3 | +2.21e-3 | -1.54e-3 | -8.26e-4 |
| 193 | 3.00e-3 | 4 | 2.75e-3 | 3.88e-3 | 3.13e-3 | 3.00e-3 | 55 | -5.32e-3 | +3.85e-3 | -2.13e-4 | -6.20e-4 |
| 194 | 3.00e-3 | 6 | 2.61e-3 | 3.84e-3 | 2.89e-3 | 2.66e-3 | 52 | -5.92e-3 | +2.69e-3 | -6.70e-4 | -6.23e-4 |
| 195 | 3.00e-3 | 4 | 2.68e-3 | 3.63e-3 | 3.04e-3 | 2.68e-3 | 43 | -2.58e-3 | +3.30e-3 | -5.75e-4 | -6.55e-4 |
| 196 | 3.00e-3 | 6 | 2.31e-3 | 3.45e-3 | 2.63e-3 | 2.31e-3 | 43 | -7.49e-3 | +3.23e-3 | -1.06e-3 | -8.52e-4 |
| 197 | 3.00e-3 | 6 | 2.28e-3 | 3.40e-3 | 2.56e-3 | 2.28e-3 | 36 | -6.73e-3 | +4.70e-3 | -8.02e-4 | -8.56e-4 |
| 198 | 3.00e-3 | 9 | 1.94e-3 | 3.36e-3 | 2.22e-3 | 2.48e-3 | 40 | -1.55e-2 | +6.15e-3 | -7.25e-4 | -3.98e-4 |
| 199 | 3.00e-3 | 5 | 1.97e-3 | 3.37e-3 | 2.44e-3 | 2.10e-3 | 32 | -1.02e-2 | +4.04e-3 | -2.00e-3 | -1.04e-3 |

