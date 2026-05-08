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

GPU0/GPU1 = compute utilization % (not load). Idle = total time with <5% utilization.

### resnet-graph

> Published: CIFAR-10 91.25% acc, Graph builder ([He et al. 2015](https://arxiv.org/abs/1512.03385), Table 6)

| Mode | Loss | Eval | vs Ref | Total (s) | Syncs | Avg Sync (ms) | GPU0 | GPU1 | Idle (s) |
|------|------|------|--------|-----------|-------|--------------|------|------|----------|
| cpu-async | 0.056484 | 0.9183 | +0.0058 | 1748.8 | 382 | 81.0 | 100% | 99% | 24.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9183 | cpu-async | - | - |

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
| cpu-async | 2.0104 | 0.8001 | 0.5938 | 0.5659 | 0.5297 | 0.5102 | 0.5018 | 0.4813 | 0.4761 | 0.4668 | 0.2063 | 0.1665 | 0.1440 | 0.1389 | 0.1354 | 0.0730 | 0.0666 | 0.0614 | 0.0587 | 0.0565 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4011 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3027 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2962 | 3.9 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 394 | 386 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1142.1 | 4.4 | epoch-boundary(130) |
| cpu-async | gpu1 | 1745.2 | 3.6 | epoch-boundary(199) |
| cpu-async | gpu2 | 1745.2 | 3.6 | epoch-boundary(199) |
| cpu-async | gpu1 | 264.7 | 1.4 | epoch-boundary(29) |
| cpu-async | gpu2 | 264.7 | 1.4 | epoch-boundary(29) |
| cpu-async | gpu2 | 412.3 | 1.2 | epoch-boundary(46) |
| cpu-async | gpu1 | 412.4 | 1.1 | epoch-boundary(46) |
| cpu-async | gpu2 | 535.2 | 1.0 | epoch-boundary(60) |
| cpu-async | gpu2 | 1266.4 | 0.9 | epoch-boundary(144) |
| cpu-async | gpu1 | 1266.5 | 0.7 | epoch-boundary(144) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.7s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 8.8s | 0.0s | 0.0s | 0.0s | 9.4s |
| resnet-graph | cpu-async | gpu2 | 13.9s | 0.0s | 0.0s | 0.0s | 14.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 259 | 0 | 382 | 81.0 | 3845/9965 | 382 | 81.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 199.9 | 11.4% |

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
| resnet-graph | cpu-async | 180 | 382 | 0 | 7.34e-3 | -1.24e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 382 | 1.07e-1 | 7.04e-2 | 6.06e-3 | 4.41e-1 | 30.9 | -1.97e-4 | 1.34e-3 |
| resnet-graph | cpu-async | 1 | 382 | 1.07e-1 | 7.08e-2 | 6.30e-3 | 4.44e-1 | 34.0 | -2.22e-4 | 1.49e-3 |
| resnet-graph | cpu-async | 2 | 382 | 1.08e-1 | 7.36e-2 | 6.06e-3 | 5.83e-1 | 35.1 | -2.48e-4 | 1.63e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9926 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9886 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9882 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 46 (0,2,3,4,5,7,8,9…141,143) | 0 (—) | — | 0,2,3,4,5,7,8,9…141,143 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 8 | 8 |
| resnet-graph | cpu-async | 0e0 | 5 | 3 | 3 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 256 | +0.033 |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | 72 | +0.030 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 50 | -0.145 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 380 | -0.074 | 179 | +0.439 | +0.588 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 381 | 3.43e1–8.04e1 | 6.60e1 | 3.13e-3 | 3.87e-3 | 2.63e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 258 | 39–77721 | +1.019e-5 | 0.477 | +1.049e-5 | 0.526 | 91 | +9.402e-6 | 0.593 | 25–1034 | +9.704e-4 | 0.749 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 247 | 867–77721 | +1.141e-5 | 0.658 | +1.155e-5 | 0.683 | 90 | +9.689e-6 | 0.611 | 79–1034 | +1.038e-3 | 0.934 |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | 73 | 78311–115690 | +1.953e-5 | 0.248 | +1.947e-5 | 0.247 | 44 | +2.146e-5 | 0.332 | 271–982 | +1.286e-3 | 0.396 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 51 | 116514–156086 | -1.857e-5 | 0.201 | -1.828e-5 | 0.199 | 45 | -1.151e-5 | 0.119 | 605–1034 | +7.933e-5 | 0.000 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.704e-4 | r0: +9.443e-4, r1: +9.832e-4, r2: +9.860e-4 | r0: 0.737, r1: 0.759, r2: 0.725 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.038e-3 | r0: +1.014e-3, r1: +1.046e-3, r2: +1.055e-3 | r0: 0.933, r1: 0.917, r2: 0.919 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–147 | +1.286e-3 | r0: +1.274e-3, r1: +1.272e-3, r2: +1.312e-3 | r0: 0.392, r1: 0.380, r2: 0.410 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +7.933e-5 | r0: -2.741e-6, r1: -1.392e-5, r2: +2.535e-4 | r0: 0.000, r1: 0.000, r2: 0.002 | 92.46× | ⚠ framing breaking |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇████████████████████▇▄▄▄▄▄▄▄▄▄▅▅▅▅▆▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▇▇▇███████████████████▇▇▇████████████▇▇▇████████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.07e-1 | 5.83e-1 | 2.21e-1 | 1.32e-1 | 39 | -4.42e-2 | +1.15e-2 | -1.46e-2 | -1.05e-2 |
| 1 | 3.00e-1 | 10 | 9.07e-2 | 1.49e-1 | 1.05e-1 | 9.58e-2 | 36 | -6.40e-3 | +1.55e-3 | -1.15e-3 | -3.35e-3 |
| 2 | 3.00e-1 | 4 | 1.02e-1 | 1.36e-1 | 1.13e-1 | 1.08e-1 | 44 | -8.13e-3 | +4.24e-3 | -6.09e-4 | -2.32e-3 |
| 3 | 3.00e-1 | 6 | 1.10e-1 | 1.44e-1 | 1.18e-1 | 1.14e-1 | 41 | -5.20e-3 | +3.34e-3 | -4.06e-4 | -1.37e-3 |
| 4 | 3.00e-1 | 6 | 1.13e-1 | 1.48e-1 | 1.22e-1 | 1.17e-1 | 46 | -5.51e-3 | +3.23e-3 | -4.29e-4 | -9.05e-4 |
| 5 | 3.00e-1 | 5 | 1.16e-1 | 1.50e-1 | 1.25e-1 | 1.16e-1 | 48 | -4.03e-3 | +2.83e-3 | -4.91e-4 | -7.48e-4 |
| 6 | 3.00e-1 | 5 | 1.12e-1 | 1.53e-1 | 1.25e-1 | 1.12e-1 | 42 | -4.73e-3 | +2.94e-3 | -7.75e-4 | -7.86e-4 |
| 7 | 3.00e-1 | 7 | 1.08e-1 | 1.48e-1 | 1.15e-1 | 1.08e-1 | 38 | -8.83e-3 | +3.38e-3 | -8.19e-4 | -7.51e-4 |
| 8 | 3.00e-1 | 6 | 1.05e-1 | 1.47e-1 | 1.16e-1 | 1.16e-1 | 41 | -1.01e-2 | +3.93e-3 | -6.61e-4 | -6.23e-4 |
| 9 | 3.00e-1 | 7 | 1.05e-1 | 1.54e-1 | 1.19e-1 | 1.22e-1 | 49 | -8.07e-3 | +3.18e-3 | -5.83e-4 | -4.76e-4 |
| 10 | 3.00e-1 | 5 | 1.12e-1 | 1.56e-1 | 1.23e-1 | 1.12e-1 | 40 | -6.12e-3 | +2.55e-3 | -1.05e-3 | -7.08e-4 |
| 11 | 3.00e-1 | 7 | 1.08e-1 | 1.46e-1 | 1.17e-1 | 1.10e-1 | 38 | -5.89e-3 | +3.33e-3 | -5.60e-4 | -6.14e-4 |
| 12 | 3.00e-1 | 7 | 9.80e-2 | 1.46e-1 | 1.12e-1 | 9.80e-2 | 30 | -4.85e-3 | +3.85e-3 | -1.06e-3 | -8.70e-4 |
| 13 | 3.00e-1 | 7 | 9.45e-2 | 1.47e-1 | 1.07e-1 | 1.02e-1 | 31 | -8.58e-3 | +4.91e-3 | -9.39e-4 | -8.12e-4 |
| 14 | 3.00e-1 | 6 | 1.16e-1 | 1.50e-1 | 1.25e-1 | 1.22e-1 | 48 | -4.10e-3 | +4.64e-3 | +2.13e-5 | -4.40e-4 |
| 15 | 3.00e-1 | 6 | 1.03e-1 | 1.57e-1 | 1.17e-1 | 1.03e-1 | 33 | -7.68e-3 | +2.69e-3 | -1.36e-3 | -8.49e-4 |
| 16 | 3.00e-1 | 7 | 1.06e-1 | 1.40e-1 | 1.14e-1 | 1.07e-1 | 36 | -5.22e-3 | +3.97e-3 | -4.86e-4 | -6.57e-4 |
| 17 | 3.00e-1 | 4 | 1.15e-1 | 1.49e-1 | 1.29e-1 | 1.35e-1 | 70 | -6.34e-3 | +3.65e-3 | -1.19e-4 | -4.56e-4 |
| 18 | 3.00e-1 | 6 | 1.31e-1 | 1.63e-1 | 1.41e-1 | 1.34e-1 | 64 | -1.68e-3 | +1.79e-3 | -2.00e-4 | -3.44e-4 |
| 19 | 3.00e-1 | 3 | 1.19e-1 | 1.58e-1 | 1.35e-1 | 1.19e-1 | 46 | -4.29e-3 | +1.70e-3 | -1.33e-3 | -6.40e-4 |
| 20 | 3.00e-1 | 5 | 1.20e-1 | 1.43e-1 | 1.26e-1 | 1.23e-1 | 50 | -3.89e-3 | +2.33e-3 | -2.08e-4 | -4.64e-4 |
| 21 | 3.00e-1 | 5 | 1.16e-1 | 1.54e-1 | 1.28e-1 | 1.16e-1 | 45 | -4.22e-3 | +2.52e-3 | -6.67e-4 | -5.71e-4 |
| 22 | 3.00e-1 | 6 | 1.06e-1 | 1.52e-1 | 1.20e-1 | 1.06e-1 | 39 | -6.45e-3 | +2.99e-3 | -1.11e-3 | -8.40e-4 |
| 23 | 3.00e-1 | 8 | 1.11e-1 | 1.45e-1 | 1.19e-1 | 1.17e-1 | 43 | -4.24e-3 | +3.96e-3 | -1.37e-4 | -4.20e-4 |
| 24 | 3.00e-1 | 4 | 1.07e-1 | 1.52e-1 | 1.21e-1 | 1.07e-1 | 37 | -7.11e-3 | +3.11e-3 | -1.62e-3 | -8.54e-4 |
| 25 | 3.00e-1 | 8 | 1.06e-1 | 1.49e-1 | 1.15e-1 | 1.06e-1 | 34 | -7.66e-3 | +4.34e-3 | -6.37e-4 | -7.25e-4 |
| 26 | 3.00e-1 | 6 | 1.03e-1 | 1.48e-1 | 1.14e-1 | 1.11e-1 | 37 | -8.38e-3 | +4.52e-3 | -6.71e-4 | -6.39e-4 |
| 27 | 3.00e-1 | 9 | 1.03e-1 | 1.52e-1 | 1.14e-1 | 1.06e-1 | 36 | -7.57e-3 | +3.82e-3 | -6.73e-4 | -6.05e-4 |
| 28 | 3.00e-1 | 4 | 1.07e-1 | 1.49e-1 | 1.19e-1 | 1.07e-1 | 34 | -8.45e-3 | +4.32e-3 | -1.33e-3 | -8.79e-4 |
| 29 | 3.00e-1 | 2 | 1.04e-1 | 2.21e-1 | 1.63e-1 | 2.21e-1 | 281 | -7.74e-4 | +2.68e-3 | +9.53e-4 | -5.14e-4 |
| 31 | 3.00e-1 | 2 | 2.18e-1 | 2.47e-1 | 2.33e-1 | 2.18e-1 | 261 | -4.75e-4 | +2.83e-4 | -9.60e-5 | -4.38e-4 |
| 33 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 320 | +1.25e-4 | +1.25e-4 | +1.25e-4 | -3.82e-4 |
| 34 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 286 | -1.87e-4 | -1.87e-4 | -1.87e-4 | -3.62e-4 |
| 35 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 294 | +5.51e-5 | +5.51e-5 | +5.51e-5 | -3.21e-4 |
| 36 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 286 | -1.94e-5 | -1.94e-5 | -1.94e-5 | -2.91e-4 |
| 37 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 269 | -9.00e-5 | -9.00e-5 | -9.00e-5 | -2.71e-4 |
| 38 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 279 | +6.47e-5 | +6.47e-5 | +6.47e-5 | -2.37e-4 |
| 39 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 275 | -3.59e-5 | -3.59e-5 | -3.59e-5 | -2.17e-4 |
| 40 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 292 | +2.41e-5 | +2.41e-5 | +2.41e-5 | -1.93e-4 |
| 41 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 291 | +1.00e-5 | +1.00e-5 | +1.00e-5 | -1.73e-4 |
| 42 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 289 | +7.64e-6 | +7.64e-6 | +7.64e-6 | -1.54e-4 |
| 43 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 270 | -4.62e-5 | -4.62e-5 | -4.62e-5 | -1.44e-4 |
| 44 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 291 | +4.12e-5 | +4.12e-5 | +4.12e-5 | -1.25e-4 |
| 46 | 3.00e-1 | 2 | 2.19e-1 | 2.25e-1 | 2.22e-1 | 2.19e-1 | 262 | -1.18e-4 | +1.20e-4 | +1.16e-6 | -1.02e-4 |
| 47 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 272 | -5.47e-5 | -5.47e-5 | -5.47e-5 | -9.76e-5 |
| 48 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 279 | +5.29e-6 | +5.29e-6 | +5.29e-6 | -8.73e-5 |
| 49 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 260 | -6.76e-5 | -6.76e-5 | -6.76e-5 | -8.53e-5 |
| 50 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 259 | +1.48e-5 | +1.48e-5 | +1.48e-5 | -7.53e-5 |
| 51 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 305 | +1.44e-4 | +1.44e-4 | +1.44e-4 | -5.33e-5 |
| 52 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 265 | -7.34e-5 | -7.34e-5 | -7.34e-5 | -5.54e-5 |
| 54 | 3.00e-1 | 2 | 2.19e-1 | 2.38e-1 | 2.29e-1 | 2.19e-1 | 287 | -2.92e-4 | +2.48e-4 | -2.20e-5 | -5.17e-5 |
| 56 | 3.00e-1 | 2 | 2.13e-1 | 2.30e-1 | 2.22e-1 | 2.13e-1 | 241 | -3.14e-4 | +1.56e-4 | -7.90e-5 | -5.92e-5 |
| 58 | 3.00e-1 | 2 | 2.12e-1 | 2.21e-1 | 2.16e-1 | 2.12e-1 | 241 | -1.72e-4 | +1.21e-4 | -2.58e-5 | -5.43e-5 |
| 60 | 3.00e-1 | 2 | 2.17e-1 | 2.32e-1 | 2.24e-1 | 2.17e-1 | 241 | -2.79e-4 | +2.80e-4 | +5.11e-7 | -4.67e-5 |
| 62 | 3.00e-1 | 2 | 2.14e-1 | 2.20e-1 | 2.17e-1 | 2.14e-1 | 242 | -1.14e-4 | +5.60e-5 | -2.91e-5 | -4.42e-5 |
| 63 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 295 | +2.18e-4 | +2.18e-4 | +2.18e-4 | -1.80e-5 |
| 64 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 254 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -3.64e-5 |
| 65 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 242 | -1.18e-4 | -1.18e-4 | -1.18e-4 | -4.46e-5 |
| 66 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 249 | +6.25e-5 | +6.25e-5 | +6.25e-5 | -3.39e-5 |
| 67 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 247 | -3.24e-5 | -3.24e-5 | -3.24e-5 | -3.37e-5 |
| 68 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 279 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -1.51e-5 |
| 69 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 303 | +1.79e-4 | +1.79e-4 | +1.79e-4 | +4.28e-6 |
| 70 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 285 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -9.75e-6 |
| 71 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 260 | -7.60e-5 | -7.60e-5 | -7.60e-5 | -1.64e-5 |
| 72 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 243 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -2.67e-5 |
| 73 | 3.00e-1 | 2 | 2.08e-1 | 2.26e-1 | 2.17e-1 | 2.08e-1 | 223 | -3.70e-4 | +1.94e-4 | -8.81e-5 | -4.12e-5 |
| 74 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 227 | +5.48e-5 | +5.48e-5 | +5.48e-5 | -3.16e-5 |
| 75 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 212 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -4.17e-5 |
| 76 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 222 | +7.85e-5 | +7.85e-5 | +7.85e-5 | -2.97e-5 |
| 77 | 3.00e-1 | 2 | 2.10e-1 | 2.11e-1 | 2.11e-1 | 2.10e-1 | 213 | -3.70e-5 | +6.91e-5 | +1.60e-5 | -2.16e-5 |
| 78 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 252 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -5.93e-6 |
| 79 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 261 | +4.27e-5 | +4.27e-5 | +4.27e-5 | -1.07e-6 |
| 80 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 283 | +9.57e-5 | +9.57e-5 | +9.57e-5 | +8.61e-6 |
| 81 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 280 | +5.32e-5 | +5.32e-5 | +5.32e-5 | +1.31e-5 |
| 82 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 276 | +6.11e-7 | +6.11e-7 | +6.11e-7 | +1.18e-5 |
| 83 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 231 | -3.00e-4 | -3.00e-4 | -3.00e-4 | -1.94e-5 |
| 84 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 227 | -1.14e-4 | -1.14e-4 | -1.14e-4 | -2.89e-5 |
| 85 | 3.00e-1 | 2 | 2.01e-1 | 2.06e-1 | 2.03e-1 | 2.01e-1 | 184 | -1.40e-4 | -5.20e-5 | -9.58e-5 | -4.20e-5 |
| 86 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 201 | -4.18e-5 | -4.18e-5 | -4.18e-5 | -4.20e-5 |
| 87 | 3.00e-1 | 2 | 2.05e-1 | 2.10e-1 | 2.07e-1 | 2.10e-1 | 197 | +1.19e-4 | +1.40e-4 | +1.30e-4 | -9.48e-6 |
| 88 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 235 | +1.03e-4 | +1.03e-4 | +1.03e-4 | +1.74e-6 |
| 89 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 249 | +7.24e-5 | +7.24e-5 | +7.24e-5 | +8.80e-6 |
| 90 | 3.00e-1 | 2 | 2.08e-1 | 2.18e-1 | 2.13e-1 | 2.08e-1 | 207 | -2.26e-4 | -1.45e-5 | -1.20e-4 | -1.68e-5 |
| 92 | 3.00e-1 | 2 | 2.12e-1 | 2.13e-1 | 2.12e-1 | 2.13e-1 | 221 | +3.14e-5 | +7.84e-5 | +5.49e-5 | -3.37e-6 |
| 93 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 211 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -1.51e-5 |
| 94 | 3.00e-1 | 2 | 2.03e-1 | 2.11e-1 | 2.07e-1 | 2.03e-1 | 180 | -2.30e-4 | +6.84e-5 | -8.07e-5 | -2.91e-5 |
| 95 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 213 | +3.59e-5 | +3.59e-5 | +3.59e-5 | -2.26e-5 |
| 96 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 217 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -2.47e-6 |
| 97 | 3.00e-1 | 2 | 2.03e-1 | 2.07e-1 | 2.05e-1 | 2.07e-1 | 192 | -2.17e-4 | +1.04e-4 | -5.67e-5 | -1.12e-5 |
| 98 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 210 | +2.82e-6 | +2.82e-6 | +2.82e-6 | -9.77e-6 |
| 99 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 257 | +3.22e-4 | +3.22e-4 | +3.22e-4 | +2.34e-5 |
| 100 | 3.00e-2 | 2 | 1.04e-1 | 2.12e-1 | 1.58e-1 | 1.04e-1 | 172 | -4.14e-3 | -2.92e-4 | -2.22e-3 | -4.22e-4 |
| 101 | 3.00e-2 | 1 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 5.42e-2 | 195 | -3.34e-3 | -3.34e-3 | -3.34e-3 | -7.13e-4 |
| 102 | 3.00e-2 | 2 | 2.46e-2 | 3.21e-2 | 2.83e-2 | 2.46e-2 | 158 | -2.87e-3 | -1.69e-3 | -2.28e-3 | -1.01e-3 |
| 103 | 3.00e-2 | 2 | 2.21e-2 | 2.25e-2 | 2.23e-2 | 2.21e-2 | 166 | -5.03e-4 | -1.29e-4 | -3.16e-4 | -8.72e-4 |
| 104 | 3.00e-2 | 1 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 203 | +4.79e-4 | +4.79e-4 | +4.79e-4 | -7.37e-4 |
| 105 | 3.00e-2 | 1 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 2.43e-2 | 186 | -2.31e-6 | -2.31e-6 | -2.31e-6 | -6.64e-4 |
| 106 | 3.00e-2 | 3 | 2.40e-2 | 2.58e-2 | 2.47e-2 | 2.40e-2 | 155 | -3.90e-4 | +2.87e-4 | -5.93e-5 | -5.03e-4 |
| 107 | 3.00e-2 | 1 | 2.56e-2 | 2.56e-2 | 2.56e-2 | 2.56e-2 | 178 | +3.68e-4 | +3.68e-4 | +3.68e-4 | -4.16e-4 |
| 108 | 3.00e-2 | 2 | 2.54e-2 | 2.61e-2 | 2.57e-2 | 2.54e-2 | 163 | -1.85e-4 | +1.07e-4 | -3.90e-5 | -3.46e-4 |
| 109 | 3.00e-2 | 1 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 2.80e-2 | 197 | +5.00e-4 | +5.00e-4 | +5.00e-4 | -2.61e-4 |
| 110 | 3.00e-2 | 1 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 190 | +2.41e-4 | +2.41e-4 | +2.41e-4 | -2.11e-4 |
| 111 | 3.00e-2 | 2 | 2.84e-2 | 3.08e-2 | 2.96e-2 | 2.84e-2 | 157 | -5.14e-4 | +2.39e-4 | -1.37e-4 | -2.01e-4 |
| 112 | 3.00e-2 | 2 | 2.68e-2 | 2.92e-2 | 2.80e-2 | 2.92e-2 | 172 | -3.98e-4 | +4.94e-4 | +4.83e-5 | -1.49e-4 |
| 113 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 152 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -1.48e-4 |
| 114 | 3.00e-2 | 2 | 2.75e-2 | 2.91e-2 | 2.83e-2 | 2.75e-2 | 139 | -4.08e-4 | +1.26e-4 | -1.41e-4 | -1.49e-4 |
| 115 | 3.00e-2 | 2 | 2.92e-2 | 3.01e-2 | 2.97e-2 | 2.92e-2 | 146 | -1.93e-4 | +5.45e-4 | +1.76e-4 | -9.14e-5 |
| 116 | 3.00e-2 | 2 | 3.04e-2 | 3.10e-2 | 3.07e-2 | 3.04e-2 | 149 | -1.16e-4 | +3.32e-4 | +1.08e-4 | -5.57e-5 |
| 117 | 3.00e-2 | 2 | 2.92e-2 | 3.36e-2 | 3.14e-2 | 2.92e-2 | 121 | -1.16e-3 | +5.46e-4 | -3.06e-4 | -1.12e-4 |
| 118 | 3.00e-2 | 2 | 2.83e-2 | 2.98e-2 | 2.90e-2 | 2.83e-2 | 121 | -4.19e-4 | +1.30e-4 | -1.45e-4 | -1.21e-4 |
| 119 | 3.00e-2 | 2 | 2.97e-2 | 3.16e-2 | 3.06e-2 | 2.97e-2 | 121 | -5.23e-4 | +7.49e-4 | +1.13e-4 | -8.27e-5 |
| 120 | 3.00e-2 | 2 | 3.17e-2 | 3.28e-2 | 3.22e-2 | 3.17e-2 | 140 | -2.44e-4 | +6.31e-4 | +1.93e-4 | -3.46e-5 |
| 121 | 3.00e-2 | 2 | 3.28e-2 | 3.38e-2 | 3.33e-2 | 3.28e-2 | 140 | -2.15e-4 | +4.02e-4 | +9.34e-5 | -1.34e-5 |
| 122 | 3.00e-2 | 2 | 3.24e-2 | 3.51e-2 | 3.37e-2 | 3.24e-2 | 132 | -6.00e-4 | +4.14e-4 | -9.30e-5 | -3.36e-5 |
| 123 | 3.00e-2 | 2 | 3.36e-2 | 3.48e-2 | 3.42e-2 | 3.36e-2 | 127 | -2.77e-4 | +4.61e-4 | +9.23e-5 | -1.34e-5 |
| 124 | 3.00e-2 | 2 | 3.45e-2 | 3.53e-2 | 3.49e-2 | 3.45e-2 | 130 | -1.75e-4 | +3.24e-4 | +7.44e-5 | +8.17e-7 |
| 125 | 3.00e-2 | 2 | 3.33e-2 | 3.54e-2 | 3.44e-2 | 3.33e-2 | 106 | -5.89e-4 | +1.75e-4 | -2.07e-4 | -4.25e-5 |
| 126 | 3.00e-2 | 2 | 3.17e-2 | 3.57e-2 | 3.37e-2 | 3.17e-2 | 98 | -1.22e-3 | +5.02e-4 | -3.57e-4 | -1.11e-4 |
| 127 | 3.00e-2 | 3 | 3.31e-2 | 3.52e-2 | 3.40e-2 | 3.38e-2 | 109 | -5.93e-4 | +7.51e-4 | +1.11e-4 | -5.60e-5 |
| 128 | 3.00e-2 | 2 | 3.36e-2 | 3.59e-2 | 3.48e-2 | 3.36e-2 | 109 | -6.08e-4 | +4.22e-4 | -9.31e-5 | -6.82e-5 |
| 129 | 3.00e-2 | 2 | 3.43e-2 | 3.93e-2 | 3.68e-2 | 3.43e-2 | 109 | -1.25e-3 | +9.92e-4 | -1.29e-4 | -9.10e-5 |
| 130 | 3.00e-2 | 2 | 3.42e-2 | 5.48e-2 | 4.45e-2 | 5.48e-2 | 403 | -2.10e-5 | +1.17e-3 | +5.75e-4 | +4.16e-5 |
| 132 | 3.00e-2 | 1 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 357 | +9.38e-5 | +9.38e-5 | +9.38e-5 | +4.68e-5 |
| 133 | 3.00e-2 | 1 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 318 | -1.98e-5 | -1.98e-5 | -1.98e-5 | +4.02e-5 |
| 134 | 3.00e-2 | 1 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 5.94e-2 | 340 | +1.55e-4 | +1.55e-4 | +1.55e-4 | +5.17e-5 |
| 135 | 3.00e-2 | 1 | 6.04e-2 | 6.04e-2 | 6.04e-2 | 6.04e-2 | 338 | +5.02e-5 | +5.02e-5 | +5.02e-5 | +5.15e-5 |
| 136 | 3.00e-2 | 1 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 5.43e-2 | 265 | -4.04e-4 | -4.04e-4 | -4.04e-4 | +5.95e-6 |
| 137 | 3.00e-2 | 1 | 5.82e-2 | 5.82e-2 | 5.82e-2 | 5.82e-2 | 308 | +2.30e-4 | +2.30e-4 | +2.30e-4 | +2.84e-5 |
| 138 | 3.00e-2 | 1 | 5.52e-2 | 5.52e-2 | 5.52e-2 | 5.52e-2 | 256 | -2.09e-4 | -2.09e-4 | -2.09e-4 | +4.62e-6 |
| 139 | 3.00e-2 | 1 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 275 | +8.66e-5 | +8.66e-5 | +8.66e-5 | +1.28e-5 |
| 140 | 3.00e-2 | 1 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 5.67e-2 | 271 | +7.53e-6 | +7.53e-6 | +7.53e-6 | +1.23e-5 |
| 141 | 3.00e-2 | 1 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 283 | +2.07e-4 | +2.07e-4 | +2.07e-4 | +3.18e-5 |
| 143 | 3.00e-2 | 2 | 6.09e-2 | 6.77e-2 | 6.43e-2 | 6.09e-2 | 271 | -3.86e-4 | +3.30e-4 | -2.78e-5 | +1.69e-5 |
| 145 | 3.00e-2 | 2 | 6.24e-2 | 6.71e-2 | 6.47e-2 | 6.24e-2 | 241 | -3.02e-4 | +2.91e-4 | -5.44e-6 | +9.69e-6 |
| 147 | 3.00e-2 | 2 | 6.00e-2 | 6.35e-2 | 6.17e-2 | 6.00e-2 | 241 | -2.36e-4 | +6.09e-5 | -8.76e-5 | -1.03e-5 |
| 149 | 3.00e-3 | 2 | 6.24e-2 | 6.68e-2 | 6.46e-2 | 6.24e-2 | 241 | -2.80e-4 | +3.46e-4 | +3.28e-5 | -5.25e-6 |
| 151 | 3.00e-3 | 2 | 1.58e-2 | 3.22e-2 | 2.40e-2 | 1.58e-2 | 270 | -2.63e-3 | -2.04e-3 | -2.34e-3 | -4.51e-4 |
| 153 | 3.00e-3 | 2 | 7.23e-3 | 9.93e-3 | 8.58e-3 | 7.23e-3 | 260 | -1.43e-3 | -1.22e-3 | -1.33e-3 | -6.16e-4 |
| 155 | 3.00e-3 | 1 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 324 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -5.67e-4 |
| 156 | 3.00e-3 | 1 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 6.31e-3 | 241 | -3.99e-4 | -3.99e-4 | -3.99e-4 | -5.50e-4 |
| 157 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 315 | +3.86e-5 | +3.86e-5 | +3.86e-5 | -4.91e-4 |
| 158 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 277 | +3.75e-5 | +3.75e-5 | +3.75e-5 | -4.38e-4 |
| 159 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 268 | -1.74e-5 | -1.74e-5 | -1.74e-5 | -3.96e-4 |
| 160 | 3.00e-3 | 1 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 6.63e-3 | 284 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -3.46e-4 |
| 161 | 3.00e-3 | 1 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 7.07e-3 | 335 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -2.92e-4 |
| 162 | 3.00e-3 | 1 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 315 | -4.86e-5 | -4.86e-5 | -4.86e-5 | -2.67e-4 |
| 163 | 3.00e-3 | 1 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 6.86e-3 | 285 | -5.19e-5 | -5.19e-5 | -5.19e-5 | -2.46e-4 |
| 165 | 3.00e-3 | 2 | 6.86e-3 | 7.41e-3 | 7.14e-3 | 6.86e-3 | 283 | -2.68e-4 | +2.15e-4 | -2.64e-5 | -2.07e-4 |
| 167 | 3.00e-3 | 1 | 7.62e-3 | 7.62e-3 | 7.62e-3 | 7.62e-3 | 376 | +2.77e-4 | +2.77e-4 | +2.77e-4 | -1.58e-4 |
| 168 | 3.00e-3 | 1 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 7.28e-3 | 336 | -1.34e-4 | -1.34e-4 | -1.34e-4 | -1.56e-4 |
| 169 | 3.00e-3 | 1 | 7.25e-3 | 7.25e-3 | 7.25e-3 | 7.25e-3 | 321 | -1.17e-5 | -1.17e-5 | -1.17e-5 | -1.41e-4 |
| 170 | 3.00e-3 | 1 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 290 | -1.35e-4 | -1.35e-4 | -1.35e-4 | -1.41e-4 |
| 171 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 269 | -1.31e-4 | -1.31e-4 | -1.31e-4 | -1.40e-4 |
| 172 | 3.00e-3 | 1 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 284 | +1.06e-4 | +1.06e-4 | +1.06e-4 | -1.15e-4 |
| 173 | 3.00e-3 | 1 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 272 | -5.33e-5 | -5.33e-5 | -5.33e-5 | -1.09e-4 |
| 174 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 291 | +7.40e-5 | +7.40e-5 | +7.40e-5 | -9.08e-5 |
| 175 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 285 | -8.91e-7 | -8.91e-7 | -8.91e-7 | -8.18e-5 |
| 176 | 3.00e-3 | 1 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 6.85e-3 | 269 | -7.35e-5 | -7.35e-5 | -7.35e-5 | -8.10e-5 |
| 177 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 276 | +3.92e-5 | +3.92e-5 | +3.92e-5 | -6.89e-5 |
| 178 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 284 | -2.97e-6 | -2.97e-6 | -2.97e-6 | -6.23e-5 |
| 179 | 3.00e-3 | 1 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 281 | +1.73e-5 | +1.73e-5 | +1.73e-5 | -5.44e-5 |
| 180 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 252 | -1.28e-4 | -1.28e-4 | -1.28e-4 | -6.18e-5 |
| 181 | 3.00e-3 | 1 | 7.48e-3 | 7.48e-3 | 7.48e-3 | 7.48e-3 | 303 | +3.48e-4 | +3.48e-4 | +3.48e-4 | -2.08e-5 |
| 182 | 3.00e-3 | 1 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 7.06e-3 | 284 | -2.04e-4 | -2.04e-4 | -2.04e-4 | -3.92e-5 |
| 183 | 3.00e-3 | 1 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 6.73e-3 | 259 | -1.85e-4 | -1.85e-4 | -1.85e-4 | -5.38e-5 |
| 184 | 3.00e-3 | 1 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 6.88e-3 | 261 | +8.67e-5 | +8.67e-5 | +8.67e-5 | -3.97e-5 |
| 185 | 3.00e-3 | 1 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 6.89e-3 | 252 | +6.64e-6 | +6.64e-6 | +6.64e-6 | -3.51e-5 |
| 186 | 3.00e-3 | 1 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 6.96e-3 | 256 | +3.69e-5 | +3.69e-5 | +3.69e-5 | -2.79e-5 |
| 187 | 3.00e-3 | 1 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 7.01e-3 | 266 | +2.73e-5 | +2.73e-5 | +2.73e-5 | -2.24e-5 |
| 188 | 3.00e-3 | 1 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 7.12e-3 | 261 | +6.21e-5 | +6.21e-5 | +6.21e-5 | -1.39e-5 |
| 189 | 3.00e-3 | 1 | 7.58e-3 | 7.58e-3 | 7.58e-3 | 7.58e-3 | 302 | +2.04e-4 | +2.04e-4 | +2.04e-4 | +7.85e-6 |
| 190 | 3.00e-3 | 1 | 7.40e-3 | 7.40e-3 | 7.40e-3 | 7.40e-3 | 293 | -8.20e-5 | -8.20e-5 | -8.20e-5 | -1.14e-6 |
| 191 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 291 | -9.08e-5 | -9.08e-5 | -9.08e-5 | -1.01e-5 |
| 192 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 250 | -2.43e-4 | -2.43e-4 | -2.43e-4 | -3.34e-5 |
| 193 | 3.00e-3 | 1 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 243 | +1.01e-4 | +1.01e-4 | +1.01e-4 | -1.99e-5 |
| 194 | 3.00e-3 | 1 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 6.78e-3 | 227 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -2.84e-5 |
| 195 | 3.00e-3 | 2 | 6.55e-3 | 6.74e-3 | 6.64e-3 | 6.55e-3 | 221 | -1.30e-4 | -2.73e-5 | -7.86e-5 | -3.85e-5 |
| 197 | 3.00e-3 | 2 | 7.00e-3 | 7.87e-3 | 7.44e-3 | 7.00e-3 | 240 | -4.89e-4 | +5.20e-4 | +1.55e-5 | -3.33e-5 |
| 198 | 3.00e-3 | 1 | 7.36e-3 | 7.36e-3 | 7.36e-3 | 7.36e-3 | 286 | +1.72e-4 | +1.72e-4 | +1.72e-4 | -1.27e-5 |
| 199 | 3.00e-3 | 1 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 7.34e-3 | 255 | -9.78e-6 | -9.78e-6 | -9.78e-6 | -1.24e-5 |

