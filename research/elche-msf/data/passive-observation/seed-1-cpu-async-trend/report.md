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
| cpu-async | 0.065652 | 0.9176 | +0.0051 | 1924.4 | 393 | 85.5 | 100% | 99% | 16.7 |

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
| cpu-async | 2.0069 | 0.7368 | 0.6219 | 0.5581 | 0.5325 | 0.5217 | 0.4981 | 0.4882 | 0.4727 | 0.4583 | 0.2141 | 0.1759 | 0.1549 | 0.1585 | 0.1544 | 0.0856 | 0.0786 | 0.0732 | 0.0685 | 0.0657 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4030 | 2.5 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3011 | 3.4 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2960 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 392 | 384 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1920.0 | 4.4 | epoch-boundary(199) |
| cpu-async | gpu2 | 1920.1 | 4.3 | epoch-boundary(199) |
| cpu-async | gpu1 | 307.0 | 1.2 | epoch-boundary(31) |
| cpu-async | gpu2 | 307.0 | 1.2 | epoch-boundary(31) |
| cpu-async | gpu1 | 403.3 | 0.8 | epoch-boundary(41) |
| cpu-async | gpu2 | 403.3 | 0.8 | epoch-boundary(41) |
| cpu-async | gpu1 | 326.6 | 0.7 | epoch-boundary(33) |
| cpu-async | gpu2 | 326.7 | 0.6 | epoch-boundary(33) |
| cpu-async | gpu1 | 287.9 | 0.6 | epoch-boundary(29) |
| cpu-async | gpu2 | 287.9 | 0.6 | epoch-boundary(29) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 7.8s | 0.0s | 0.0s | 0.0s | 8.4s |
| resnet-graph | cpu-async | gpu2 | 7.6s | 0.0s | 0.0s | 0.0s | 8.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 290 | 0 | 393 | 85.5 | 4211/10192 | 393 | 85.5 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 226.4 | 11.8% |

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
| resnet-graph | cpu-async | 188 | 393 | 0 | 6.71e-3 | -2.53e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 393 | 9.95e-2 | 7.62e-2 | 5.92e-3 | 3.80e-1 | 30.8 | -1.72e-4 | 1.04e-3 |
| resnet-graph | cpu-async | 1 | 393 | 1.00e-1 | 7.70e-2 | 6.10e-3 | 3.70e-1 | 36.6 | -1.76e-4 | 1.23e-3 |
| resnet-graph | cpu-async | 2 | 393 | 1.00e-1 | 7.75e-2 | 6.24e-3 | 4.18e-1 | 32.6 | -1.87e-4 | 1.14e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9960 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9952 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9946 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 43 (0,2,3,4,6,7,8,9…137,144) | 0 (—) | — | 0,2,3,4,6,7,8,9…137,144 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 0 | 0 |
| resnet-graph | cpu-async | 0e0 | 5 | 0 | 0 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–98 | 220 | +0.123 |
| resnet-graph | cpu-async | 3.00e-2 | 99–148 | 113 | +0.090 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 56 | -0.312 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 391 | -0.062 | 187 | +0.365 | +0.568 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 392 | 3.34e1–7.91e1 | 6.52e1 | 3.07e-3 | 4.23e-3 | 2.73e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–98 | 222 | 30–77201 | +8.370e-6 | 0.426 | +8.675e-6 | 0.454 | 93 | +6.127e-6 | 0.335 | 30–972 | +9.141e-4 | 0.742 |
| resnet-graph | cpu-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | 212 | 964–77201 | +9.326e-6 | 0.566 | +9.589e-6 | 0.585 | 92 | +6.337e-6 | 0.347 | 72–972 | +9.937e-4 | 0.932 |
| resnet-graph | cpu-async | 3.00e-2 | 99–148 | 114 | 77676–116546 | +8.083e-6 | 0.034 | +8.414e-6 | 0.038 | 47 | +1.403e-5 | 0.099 | 101–1012 | +1.266e-3 | 0.471 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 57 | 117510–156083 | -1.361e-5 | 0.158 | -1.357e-5 | 0.160 | 48 | -1.579e-5 | 0.181 | 486–964 | +2.030e-3 | 0.234 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–98 | +9.141e-4 | r0: +8.967e-4, r1: +9.289e-4, r2: +9.191e-4 | r0: 0.743, r1: 0.740, r2: 0.719 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–98 (post-transient, skipped 1) | +9.937e-4 | r0: +9.691e-4, r1: +1.010e-3, r2: +1.005e-3 | r0: 0.918, r1: 0.924, r2: 0.919 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 99–148 | +1.266e-3 | r0: +1.232e-3, r1: +1.291e-3, r2: +1.274e-3 | r0: 0.452, r1: 0.486, r2: 0.467 | 1.05× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +2.030e-3 | r0: +1.959e-3, r1: +2.066e-3, r2: +2.063e-3 | r0: 0.223, r1: 0.244, r2: 0.234 | 1.05× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇██████████████████▇▄▄▄▄▄▄▄▄▅▅▆▄▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▆▇▇▇█████████████████▇▆▇████▇▇▇██▇▇▇█████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 10 | 1.08e-1 | 4.18e-1 | 2.16e-1 | 1.62e-1 | 51 | -2.60e-2 | +9.11e-3 | -9.81e-3 | -7.63e-3 |
| 1 | 3.00e-1 | 8 | 1.07e-1 | 1.75e-1 | 1.20e-1 | 1.10e-1 | 41 | -8.28e-3 | +1.14e-3 | -1.30e-3 | -3.11e-3 |
| 2 | 3.00e-1 | 4 | 1.20e-1 | 1.48e-1 | 1.31e-1 | 1.29e-1 | 50 | -4.47e-3 | +3.98e-3 | +2.10e-4 | -1.86e-3 |
| 3 | 3.00e-1 | 5 | 1.22e-1 | 1.65e-1 | 1.34e-1 | 1.24e-1 | 46 | -4.75e-3 | +2.69e-3 | -6.86e-4 | -1.34e-3 |
| 4 | 3.00e-1 | 8 | 1.28e-1 | 1.59e-1 | 1.34e-1 | 1.28e-1 | 42 | -3.41e-3 | +3.01e-3 | -2.37e-4 | -7.05e-4 |
| 5 | 3.00e-1 | 4 | 1.17e-1 | 1.62e-1 | 1.32e-1 | 1.17e-1 | 37 | -6.83e-3 | +2.99e-3 | -1.45e-3 | -9.90e-4 |
| 6 | 3.00e-1 | 7 | 1.16e-1 | 1.57e-1 | 1.26e-1 | 1.21e-1 | 41 | -5.68e-3 | +4.13e-3 | -3.75e-4 | -6.40e-4 |
| 7 | 3.00e-1 | 7 | 1.09e-1 | 1.73e-1 | 1.22e-1 | 1.11e-1 | 35 | -8.39e-3 | +3.55e-3 | -1.21e-3 | -8.51e-4 |
| 8 | 3.00e-1 | 7 | 1.01e-1 | 1.60e-1 | 1.17e-1 | 1.01e-1 | 26 | -8.90e-3 | +4.46e-3 | -1.48e-3 | -1.21e-3 |
| 9 | 3.00e-1 | 8 | 1.01e-1 | 1.48e-1 | 1.13e-1 | 1.14e-1 | 36 | -1.46e-2 | +5.48e-3 | -8.23e-4 | -8.15e-4 |
| 10 | 3.00e-1 | 9 | 1.02e-1 | 1.58e-1 | 1.18e-1 | 1.02e-1 | 29 | -6.57e-3 | +3.54e-3 | -9.33e-4 | -8.99e-4 |
| 11 | 3.00e-1 | 5 | 1.01e-1 | 1.50e-1 | 1.14e-1 | 1.05e-1 | 31 | -1.16e-2 | +5.34e-3 | -1.47e-3 | -1.10e-3 |
| 12 | 3.00e-1 | 7 | 1.02e-1 | 1.44e-1 | 1.14e-1 | 1.02e-1 | 31 | -6.50e-3 | +4.51e-3 | -7.65e-4 | -9.66e-4 |
| 13 | 3.00e-1 | 7 | 1.14e-1 | 1.54e-1 | 1.22e-1 | 1.16e-1 | 40 | -5.42e-3 | +5.32e-3 | -2.19e-4 | -5.86e-4 |
| 14 | 3.00e-1 | 7 | 1.18e-1 | 1.56e-1 | 1.26e-1 | 1.20e-1 | 42 | -4.52e-3 | +3.59e-3 | -3.22e-4 | -4.41e-4 |
| 15 | 3.00e-1 | 5 | 1.10e-1 | 1.49e-1 | 1.20e-1 | 1.14e-1 | 38 | -6.41e-3 | +2.89e-3 | -8.36e-4 | -5.78e-4 |
| 16 | 3.00e-1 | 7 | 1.05e-1 | 1.54e-1 | 1.17e-1 | 1.13e-1 | 42 | -7.61e-3 | +3.59e-3 | -7.46e-4 | -5.86e-4 |
| 17 | 3.00e-1 | 9 | 1.10e-1 | 1.51e-1 | 1.17e-1 | 1.12e-1 | 38 | -6.64e-3 | +3.61e-3 | -4.71e-4 | -4.46e-4 |
| 18 | 3.00e-1 | 5 | 1.04e-1 | 1.52e-1 | 1.19e-1 | 1.04e-1 | 32 | -7.18e-3 | +3.76e-3 | -1.46e-3 | -8.87e-4 |
| 19 | 3.00e-1 | 1 | 1.09e-1 | 1.09e-1 | 1.09e-1 | 1.09e-1 | 33 | +1.27e-3 | +1.27e-3 | +1.27e-3 | -6.71e-4 |
| 20 | 3.00e-1 | 2 | 2.21e-1 | 2.22e-1 | 2.22e-1 | 2.21e-1 | 263 | -2.89e-5 | +2.31e-3 | +1.14e-3 | -3.39e-4 |
| 22 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 349 | +2.18e-4 | +2.18e-4 | +2.18e-4 | -2.83e-4 |
| 23 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 307 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -2.70e-4 |
| 24 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 291 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -2.55e-4 |
| 25 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 323 | +6.47e-5 | +6.47e-5 | +6.47e-5 | -2.23e-4 |
| 26 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 279 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -2.13e-4 |
| 27 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 281 | -3.78e-5 | -3.78e-5 | -3.78e-5 | -1.95e-4 |
| 28 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 270 | +2.36e-5 | +2.36e-5 | +2.36e-5 | -1.73e-4 |
| 29 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 270 | -1.02e-5 | -1.02e-5 | -1.02e-5 | -1.57e-4 |
| 30 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 289 | +3.30e-5 | +3.30e-5 | +3.30e-5 | -1.38e-4 |
| 31 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 270 | -7.27e-5 | -7.27e-5 | -7.27e-5 | -1.31e-4 |
| 32 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 276 | +2.62e-5 | +2.62e-5 | +2.62e-5 | -1.16e-4 |
| 33 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 270 | -4.01e-5 | -4.01e-5 | -4.01e-5 | -1.08e-4 |
| 35 | 3.00e-1 | 2 | 2.19e-1 | 2.36e-1 | 2.27e-1 | 2.19e-1 | 257 | -2.88e-4 | +2.90e-4 | +8.99e-7 | -9.03e-5 |
| 37 | 3.00e-1 | 2 | 2.12e-1 | 2.24e-1 | 2.18e-1 | 2.12e-1 | 253 | -2.25e-4 | +7.91e-5 | -7.30e-5 | -8.86e-5 |
| 39 | 3.00e-1 | 2 | 2.09e-1 | 2.24e-1 | 2.17e-1 | 2.09e-1 | 239 | -2.95e-4 | +1.94e-4 | -5.04e-5 | -8.37e-5 |
| 40 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 302 | +1.93e-4 | +1.93e-4 | +1.93e-4 | -5.61e-5 |
| 41 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 251 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -6.28e-5 |
| 42 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 278 | +2.70e-5 | +2.70e-5 | +2.70e-5 | -5.39e-5 |
| 43 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 310 | +1.14e-4 | +1.14e-4 | +1.14e-4 | -3.71e-5 |
| 44 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 263 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -5.36e-5 |
| 45 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 260 | +1.51e-5 | +1.51e-5 | +1.51e-5 | -4.67e-5 |
| 46 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 273 | +7.20e-5 | +7.20e-5 | +7.20e-5 | -3.49e-5 |
| 47 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 283 | -1.37e-5 | -1.37e-5 | -1.37e-5 | -3.28e-5 |
| 48 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 245 | -5.28e-5 | -5.28e-5 | -5.28e-5 | -3.48e-5 |
| 49 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 257 | -1.39e-5 | -1.39e-5 | -1.39e-5 | -3.27e-5 |
| 50 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 276 | +1.59e-4 | +1.59e-4 | +1.59e-4 | -1.35e-5 |
| 51 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 303 | +4.56e-5 | +4.56e-5 | +4.56e-5 | -7.62e-6 |
| 52 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 272 | -8.73e-5 | -8.73e-5 | -8.73e-5 | -1.56e-5 |
| 53 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 267 | -9.18e-5 | -9.18e-5 | -9.18e-5 | -2.32e-5 |
| 54 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 262 | -3.84e-5 | -3.84e-5 | -3.84e-5 | -2.47e-5 |
| 55 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 252 | +2.57e-6 | +2.57e-6 | +2.57e-6 | -2.20e-5 |
| 56 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 278 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -5.73e-6 |
| 57 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 263 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -1.56e-5 |
| 58 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 243 | -7.77e-5 | -7.77e-5 | -7.77e-5 | -2.18e-5 |
| 59 | 3.00e-1 | 2 | 2.04e-1 | 2.10e-1 | 2.07e-1 | 2.04e-1 | 214 | -1.44e-4 | -2.89e-5 | -8.66e-5 | -3.47e-5 |
| 60 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 217 | +1.61e-5 | +1.61e-5 | +1.61e-5 | -2.96e-5 |
| 61 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 242 | +1.10e-4 | +1.10e-4 | +1.10e-4 | -1.57e-5 |
| 62 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 253 | +3.46e-5 | +3.46e-5 | +3.46e-5 | -1.06e-5 |
| 63 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 271 | +9.64e-5 | +9.64e-5 | +9.64e-5 | +7.61e-8 |
| 64 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 273 | -8.38e-6 | -8.38e-6 | -8.38e-6 | -7.69e-7 |
| 65 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 284 | +5.62e-5 | +5.62e-5 | +5.62e-5 | +4.92e-6 |
| 66 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 256 | -9.10e-5 | -9.10e-5 | -9.10e-5 | -4.67e-6 |
| 67 | 3.00e-1 | 2 | 2.08e-1 | 2.10e-1 | 2.09e-1 | 2.08e-1 | 226 | -1.05e-4 | -4.46e-5 | -7.50e-5 | -1.77e-5 |
| 69 | 3.00e-1 | 2 | 2.06e-1 | 2.25e-1 | 2.16e-1 | 2.06e-1 | 213 | -4.17e-4 | +2.90e-4 | -6.35e-5 | -3.00e-5 |
| 70 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 237 | +3.42e-5 | +3.42e-5 | +3.42e-5 | -2.36e-5 |
| 71 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 222 | -4.83e-5 | -4.83e-5 | -4.83e-5 | -2.60e-5 |
| 72 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 221 | +1.49e-5 | +1.49e-5 | +1.49e-5 | -2.19e-5 |
| 73 | 3.00e-1 | 2 | 2.09e-1 | 2.14e-1 | 2.12e-1 | 2.09e-1 | 228 | -1.10e-4 | +1.58e-4 | +2.43e-5 | -1.45e-5 |
| 75 | 3.00e-1 | 2 | 2.10e-1 | 2.14e-1 | 2.12e-1 | 2.10e-1 | 228 | -9.33e-5 | +9.22e-5 | -5.35e-7 | -1.28e-5 |
| 76 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 214 | -6.72e-5 | -6.72e-5 | -6.72e-5 | -1.82e-5 |
| 77 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 217 | -3.43e-5 | -3.43e-5 | -3.43e-5 | -1.98e-5 |
| 78 | 3.00e-1 | 2 | 1.99e-1 | 2.02e-1 | 2.00e-1 | 1.99e-1 | 188 | -7.92e-5 | -7.65e-5 | -7.78e-5 | -3.08e-5 |
| 79 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 227 | +2.92e-4 | +2.92e-4 | +2.92e-4 | +1.50e-6 |
| 80 | 3.00e-1 | 2 | 2.00e-1 | 2.17e-1 | 2.09e-1 | 2.00e-1 | 188 | -4.35e-4 | +9.00e-5 | -1.72e-4 | -3.41e-5 |
| 81 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 205 | +1.57e-4 | +1.57e-4 | +1.57e-4 | -1.50e-5 |
| 82 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 207 | -1.00e-4 | -1.00e-4 | -1.00e-4 | -2.36e-5 |
| 83 | 3.00e-1 | 2 | 1.96e-1 | 2.01e-1 | 1.98e-1 | 1.96e-1 | 175 | -1.50e-4 | -4.34e-5 | -9.66e-5 | -3.80e-5 |
| 84 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 191 | +7.66e-5 | +7.66e-5 | +7.66e-5 | -2.65e-5 |
| 85 | 3.00e-1 | 2 | 1.98e-1 | 2.01e-1 | 2.00e-1 | 1.98e-1 | 189 | -8.24e-5 | +6.47e-5 | -8.83e-6 | -2.39e-5 |
| 86 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 201 | +9.26e-5 | +9.26e-5 | +9.26e-5 | -1.22e-5 |
| 87 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 231 | +2.24e-4 | +2.24e-4 | +2.24e-4 | +1.14e-5 |
| 88 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 270 | +2.35e-4 | +2.35e-4 | +2.35e-4 | +3.37e-5 |
| 89 | 3.00e-1 | 2 | 1.92e-1 | 2.10e-1 | 2.01e-1 | 1.92e-1 | 164 | -5.61e-4 | -3.57e-4 | -4.59e-4 | -6.09e-5 |
| 90 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 207 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -2.44e-5 |
| 91 | 3.00e-1 | 2 | 1.86e-1 | 1.99e-1 | 1.93e-1 | 1.86e-1 | 153 | -4.36e-4 | -1.32e-4 | -2.84e-4 | -7.53e-5 |
| 92 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 213 | +4.14e-4 | +4.14e-4 | +4.14e-4 | -2.64e-5 |
| 93 | 3.00e-1 | 2 | 1.90e-1 | 2.00e-1 | 1.95e-1 | 1.90e-1 | 164 | -3.26e-4 | -8.25e-5 | -2.04e-4 | -6.14e-5 |
| 94 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 178 | +1.44e-4 | +1.44e-4 | +1.44e-4 | -4.09e-5 |
| 95 | 3.00e-1 | 2 | 1.86e-1 | 1.98e-1 | 1.92e-1 | 1.86e-1 | 154 | -3.95e-4 | +1.02e-4 | -1.46e-4 | -6.34e-5 |
| 96 | 3.00e-1 | 2 | 1.91e-1 | 1.99e-1 | 1.95e-1 | 1.91e-1 | 162 | -2.56e-4 | +3.49e-4 | +4.63e-5 | -4.56e-5 |
| 97 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 199 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -1.63e-5 |
| 98 | 3.00e-1 | 2 | 1.88e-1 | 2.04e-1 | 1.96e-1 | 1.88e-1 | 154 | -5.29e-4 | +8.68e-5 | -2.21e-4 | -5.83e-5 |
| 99 | 3.00e-2 | 2 | 1.84e-1 | 1.92e-1 | 1.88e-1 | 1.84e-1 | 144 | -2.72e-4 | +1.13e-4 | -7.96e-5 | -6.42e-5 |
| 100 | 3.00e-2 | 1 | 1.25e-1 | 1.25e-1 | 1.25e-1 | 1.25e-1 | 204 | -1.90e-3 | -1.90e-3 | -1.90e-3 | -2.48e-4 |
| 101 | 3.00e-2 | 2 | 3.39e-2 | 6.23e-2 | 4.81e-2 | 3.39e-2 | 152 | -4.00e-3 | -3.80e-3 | -3.90e-3 | -9.44e-4 |
| 102 | 3.00e-2 | 2 | 2.18e-2 | 2.76e-2 | 2.47e-2 | 2.18e-2 | 127 | -1.83e-3 | -1.07e-3 | -1.45e-3 | -1.04e-3 |
| 103 | 3.00e-2 | 2 | 2.08e-2 | 2.22e-2 | 2.15e-2 | 2.08e-2 | 122 | -5.34e-4 | +1.07e-4 | -2.14e-4 | -8.90e-4 |
| 104 | 3.00e-2 | 1 | 2.29e-2 | 2.29e-2 | 2.29e-2 | 2.29e-2 | 170 | +5.62e-4 | +5.62e-4 | +5.62e-4 | -7.45e-4 |
| 105 | 3.00e-2 | 3 | 2.22e-2 | 2.43e-2 | 2.31e-2 | 2.27e-2 | 133 | -6.72e-4 | +3.39e-4 | -4.81e-5 | -5.57e-4 |
| 106 | 3.00e-2 | 1 | 2.47e-2 | 2.47e-2 | 2.47e-2 | 2.47e-2 | 165 | +5.10e-4 | +5.10e-4 | +5.10e-4 | -4.50e-4 |
| 107 | 3.00e-2 | 2 | 2.39e-2 | 2.50e-2 | 2.45e-2 | 2.39e-2 | 133 | -3.54e-4 | +7.57e-5 | -1.39e-4 | -3.93e-4 |
| 108 | 3.00e-2 | 3 | 2.42e-2 | 2.67e-2 | 2.52e-2 | 2.42e-2 | 133 | -6.19e-4 | +6.92e-4 | -1.64e-5 | -2.99e-4 |
| 109 | 3.00e-2 | 1 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 2.81e-2 | 187 | +8.02e-4 | +8.02e-4 | +8.02e-4 | -1.89e-4 |
| 110 | 3.00e-2 | 2 | 2.54e-2 | 2.76e-2 | 2.65e-2 | 2.54e-2 | 124 | -6.69e-4 | -1.10e-4 | -3.90e-4 | -2.30e-4 |
| 111 | 3.00e-2 | 2 | 2.54e-2 | 2.73e-2 | 2.63e-2 | 2.54e-2 | 124 | -5.82e-4 | +4.65e-4 | -5.87e-5 | -2.02e-4 |
| 112 | 3.00e-2 | 2 | 2.61e-2 | 2.85e-2 | 2.73e-2 | 2.61e-2 | 122 | -7.17e-4 | +7.27e-4 | +5.03e-6 | -1.70e-4 |
| 113 | 3.00e-2 | 2 | 2.60e-2 | 2.73e-2 | 2.67e-2 | 2.60e-2 | 106 | -4.66e-4 | +3.27e-4 | -6.96e-5 | -1.55e-4 |
| 114 | 3.00e-2 | 3 | 2.53e-2 | 2.75e-2 | 2.61e-2 | 2.54e-2 | 106 | -7.99e-4 | +4.18e-4 | -1.04e-4 | -1.44e-4 |
| 115 | 3.00e-2 | 2 | 2.62e-2 | 2.81e-2 | 2.72e-2 | 2.62e-2 | 106 | -6.41e-4 | +7.21e-4 | +3.98e-5 | -1.16e-4 |
| 116 | 3.00e-2 | 2 | 2.77e-2 | 2.98e-2 | 2.87e-2 | 2.77e-2 | 106 | -6.83e-4 | +8.60e-4 | +8.86e-5 | -8.49e-5 |
| 117 | 3.00e-2 | 3 | 2.66e-2 | 2.97e-2 | 2.79e-2 | 2.66e-2 | 106 | -8.07e-4 | +5.04e-4 | -1.80e-4 | -1.17e-4 |
| 118 | 3.00e-2 | 2 | 2.81e-2 | 3.19e-2 | 3.00e-2 | 2.81e-2 | 110 | -1.16e-3 | +1.07e-3 | -4.56e-5 | -1.15e-4 |
| 119 | 3.00e-2 | 3 | 2.72e-2 | 3.09e-2 | 2.85e-2 | 2.72e-2 | 95 | -1.21e-3 | +7.38e-4 | -1.99e-4 | -1.45e-4 |
| 120 | 3.00e-2 | 3 | 2.61e-2 | 3.24e-2 | 2.83e-2 | 2.64e-2 | 87 | -2.97e-3 | +1.23e-3 | -5.23e-4 | -2.57e-4 |
| 121 | 3.00e-2 | 3 | 2.63e-2 | 2.86e-2 | 2.76e-2 | 2.63e-2 | 85 | -7.03e-4 | +7.75e-4 | -5.73e-5 | -2.17e-4 |
| 122 | 3.00e-2 | 2 | 2.89e-2 | 3.10e-2 | 3.00e-2 | 2.89e-2 | 95 | -7.31e-4 | +1.37e-3 | +3.18e-4 | -1.25e-4 |
| 123 | 3.00e-2 | 4 | 2.77e-2 | 3.26e-2 | 2.92e-2 | 2.78e-2 | 86 | -1.28e-3 | +9.03e-4 | -2.07e-4 | -1.61e-4 |
| 124 | 3.00e-2 | 3 | 2.63e-2 | 3.09e-2 | 2.84e-2 | 2.80e-2 | 72 | -1.40e-3 | +1.53e-3 | -2.09e-4 | -1.81e-4 |
| 125 | 3.00e-2 | 3 | 2.63e-2 | 3.09e-2 | 2.79e-2 | 2.63e-2 | 70 | -2.35e-3 | +9.87e-4 | -4.88e-4 | -2.73e-4 |
| 126 | 3.00e-2 | 4 | 2.69e-2 | 3.14e-2 | 2.82e-2 | 2.74e-2 | 70 | -2.11e-3 | +1.76e-3 | -4.31e-5 | -2.03e-4 |
| 127 | 3.00e-2 | 4 | 2.67e-2 | 3.32e-2 | 2.87e-2 | 2.71e-2 | 65 | -2.82e-3 | +1.89e-3 | -3.05e-4 | -2.47e-4 |
| 128 | 3.00e-2 | 5 | 2.55e-2 | 3.19e-2 | 2.73e-2 | 2.55e-2 | 56 | -2.30e-3 | +1.70e-3 | -3.83e-4 | -3.13e-4 |
| 129 | 3.00e-2 | 4 | 2.45e-2 | 3.18e-2 | 2.70e-2 | 2.45e-2 | 51 | -3.57e-3 | +2.19e-3 | -6.99e-4 | -4.69e-4 |
| 130 | 3.00e-2 | 5 | 2.32e-2 | 2.94e-2 | 2.53e-2 | 2.65e-2 | 56 | -5.41e-3 | +2.15e-3 | -1.94e-4 | -3.07e-4 |
| 131 | 3.00e-2 | 7 | 2.51e-2 | 3.51e-2 | 2.74e-2 | 2.51e-2 | 48 | -3.99e-3 | +2.57e-3 | -4.75e-4 | -3.98e-4 |
| 132 | 3.00e-2 | 3 | 2.53e-2 | 3.26e-2 | 2.80e-2 | 2.53e-2 | 46 | -5.01e-3 | +3.02e-3 | -9.00e-4 | -5.68e-4 |
| 133 | 3.00e-2 | 6 | 2.29e-2 | 3.19e-2 | 2.53e-2 | 2.49e-2 | 44 | -6.52e-3 | +2.83e-3 | -6.13e-4 | -5.24e-4 |
| 134 | 3.00e-2 | 1 | 2.44e-2 | 2.44e-2 | 2.44e-2 | 2.44e-2 | 41 | -4.83e-4 | -4.83e-4 | -4.83e-4 | -5.20e-4 |
| 135 | 3.00e-2 | 1 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 5.48e-2 | 346 | +2.34e-3 | +2.34e-3 | +2.34e-3 | -2.34e-4 |
| 136 | 3.00e-2 | 1 | 5.76e-2 | 5.76e-2 | 5.76e-2 | 5.76e-2 | 285 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -1.93e-4 |
| 137 | 3.00e-2 | 1 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 323 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -1.55e-4 |
| 138 | 3.00e-2 | 1 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 292 | -1.33e-4 | -1.33e-4 | -1.33e-4 | -1.52e-4 |
| 139 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 273 | -1.28e-5 | -1.28e-5 | -1.28e-5 | -1.39e-4 |
| 140 | 3.00e-2 | 1 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 322 | +2.23e-4 | +2.23e-4 | +2.23e-4 | -1.02e-4 |
| 141 | 3.00e-2 | 1 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 6.13e-2 | 291 | -1.00e-4 | -1.00e-4 | -1.00e-4 | -1.02e-4 |
| 142 | 3.00e-2 | 1 | 6.35e-2 | 6.35e-2 | 6.35e-2 | 6.35e-2 | 315 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -8.10e-5 |
| 144 | 3.00e-2 | 2 | 6.15e-2 | 6.99e-2 | 6.57e-2 | 6.15e-2 | 250 | -5.07e-4 | +2.56e-4 | -1.25e-4 | -9.33e-5 |
| 146 | 3.00e-2 | 2 | 6.23e-2 | 6.72e-2 | 6.48e-2 | 6.23e-2 | 250 | -3.02e-4 | +2.63e-4 | -1.98e-5 | -8.21e-5 |
| 148 | 3.00e-2 | 2 | 6.46e-2 | 6.94e-2 | 6.70e-2 | 6.46e-2 | 261 | -2.74e-4 | +3.10e-4 | +1.82e-5 | -6.60e-5 |
| 150 | 3.00e-3 | 1 | 7.26e-2 | 7.26e-2 | 7.26e-2 | 7.26e-2 | 371 | +3.15e-4 | +3.15e-4 | +3.15e-4 | -2.79e-5 |
| 151 | 3.00e-3 | 1 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 282 | -2.65e-3 | -2.65e-3 | -2.65e-3 | -2.90e-4 |
| 152 | 3.00e-3 | 1 | 1.78e-2 | 1.78e-2 | 1.78e-2 | 1.78e-2 | 280 | -2.36e-3 | -2.36e-3 | -2.36e-3 | -4.97e-4 |
| 153 | 3.00e-3 | 1 | 1.03e-2 | 1.03e-2 | 1.03e-2 | 1.03e-2 | 272 | -2.00e-3 | -2.00e-3 | -2.00e-3 | -6.48e-4 |
| 154 | 3.00e-3 | 1 | 7.56e-3 | 7.56e-3 | 7.56e-3 | 7.56e-3 | 241 | -1.28e-3 | -1.28e-3 | -1.28e-3 | -7.11e-4 |
| 155 | 3.00e-3 | 1 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 6.49e-3 | 252 | -6.03e-4 | -6.03e-4 | -6.03e-4 | -7.01e-4 |
| 156 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 278 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -6.17e-4 |
| 157 | 3.00e-3 | 2 | 6.50e-3 | 6.55e-3 | 6.52e-3 | 6.50e-3 | 247 | -1.04e-4 | -3.40e-5 | -6.92e-5 | -5.13e-4 |
| 159 | 3.00e-3 | 1 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 7.10e-3 | 331 | +2.66e-4 | +2.66e-4 | +2.66e-4 | -4.35e-4 |
| 160 | 3.00e-3 | 1 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 7.32e-3 | 328 | +9.68e-5 | +9.68e-5 | +9.68e-5 | -3.82e-4 |
| 161 | 3.00e-3 | 1 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 6.61e-3 | 262 | -3.91e-4 | -3.91e-4 | -3.91e-4 | -3.82e-4 |
| 162 | 3.00e-3 | 1 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 6.64e-3 | 265 | +1.48e-5 | +1.48e-5 | +1.48e-5 | -3.43e-4 |
| 163 | 3.00e-3 | 2 | 6.38e-3 | 6.59e-3 | 6.49e-3 | 6.38e-3 | 227 | -1.48e-4 | -2.51e-5 | -8.63e-5 | -2.95e-4 |
| 164 | 3.00e-3 | 1 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 6.55e-3 | 252 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -2.54e-4 |
| 165 | 3.00e-3 | 1 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 6.68e-3 | 261 | +7.35e-5 | +7.35e-5 | +7.35e-5 | -2.22e-4 |
| 166 | 3.00e-3 | 1 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 7.02e-3 | 282 | +1.74e-4 | +1.74e-4 | +1.74e-4 | -1.82e-4 |
| 167 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 267 | -1.36e-5 | -1.36e-5 | -1.36e-5 | -1.65e-4 |
| 168 | 3.00e-3 | 1 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 7.13e-3 | 278 | +7.31e-5 | +7.31e-5 | +7.31e-5 | -1.41e-4 |
| 169 | 3.00e-3 | 1 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 6.93e-3 | 257 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -1.38e-4 |
| 170 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 285 | +1.62e-4 | +1.62e-4 | +1.62e-4 | -1.08e-4 |
| 171 | 3.00e-3 | 1 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 251 | -1.76e-4 | -1.76e-4 | -1.76e-4 | -1.15e-4 |
| 172 | 3.00e-3 | 1 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 264 | +4.96e-5 | +4.96e-5 | +4.96e-5 | -9.85e-5 |
| 173 | 3.00e-3 | 1 | 7.05e-3 | 7.05e-3 | 7.05e-3 | 7.05e-3 | 271 | +4.34e-6 | +4.34e-6 | +4.34e-6 | -8.83e-5 |
| 174 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 262 | +1.13e-4 | +1.13e-4 | +1.13e-4 | -6.82e-5 |
| 175 | 3.00e-3 | 1 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 6.95e-3 | 249 | -1.77e-4 | -1.77e-4 | -1.77e-4 | -7.91e-5 |
| 176 | 3.00e-3 | 1 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 258 | +5.45e-5 | +5.45e-5 | +5.45e-5 | -6.57e-5 |
| 177 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 256 | -3.15e-5 | -3.15e-5 | -3.15e-5 | -6.23e-5 |
| 178 | 3.00e-3 | 2 | 6.81e-3 | 6.81e-3 | 6.81e-3 | 6.81e-3 | 216 | -1.12e-4 | +1.52e-6 | -5.55e-5 | -6.04e-5 |
| 179 | 3.00e-3 | 1 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 6.94e-3 | 238 | +7.86e-5 | +7.86e-5 | +7.86e-5 | -4.65e-5 |
| 180 | 3.00e-3 | 1 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 231 | -6.35e-5 | -6.35e-5 | -6.35e-5 | -4.82e-5 |
| 181 | 3.00e-3 | 1 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 6.92e-3 | 230 | +5.50e-5 | +5.50e-5 | +5.50e-5 | -3.79e-5 |
| 182 | 3.00e-3 | 1 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 6.84e-3 | 247 | -5.17e-5 | -5.17e-5 | -5.17e-5 | -3.93e-5 |
| 183 | 3.00e-3 | 2 | 7.08e-3 | 7.11e-3 | 7.10e-3 | 7.08e-3 | 226 | -2.42e-5 | +1.63e-4 | +6.93e-5 | -1.96e-5 |
| 185 | 3.00e-3 | 2 | 7.03e-3 | 7.55e-3 | 7.29e-3 | 7.03e-3 | 220 | -3.24e-4 | +2.36e-4 | -4.41e-5 | -2.71e-5 |
| 186 | 3.00e-3 | 1 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 6.99e-3 | 242 | -2.47e-5 | -2.47e-5 | -2.47e-5 | -2.68e-5 |
| 187 | 3.00e-3 | 1 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 6.98e-3 | 226 | -7.90e-6 | -7.90e-6 | -7.90e-6 | -2.49e-5 |
| 188 | 3.00e-3 | 1 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 6.83e-3 | 211 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -3.29e-5 |
| 189 | 3.00e-3 | 1 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 7.17e-3 | 224 | +2.18e-4 | +2.18e-4 | +2.18e-4 | -7.75e-6 |
| 190 | 3.00e-3 | 2 | 6.79e-3 | 6.87e-3 | 6.83e-3 | 6.79e-3 | 196 | -1.92e-4 | -6.57e-5 | -1.29e-4 | -3.01e-5 |
| 191 | 3.00e-3 | 1 | 7.46e-3 | 7.46e-3 | 7.46e-3 | 7.46e-3 | 260 | +3.63e-4 | +3.63e-4 | +3.63e-4 | +9.24e-6 |
| 192 | 3.00e-3 | 1 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 7.26e-3 | 246 | -1.11e-4 | -1.11e-4 | -1.11e-4 | -2.78e-6 |
| 193 | 3.00e-3 | 1 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 6.97e-3 | 219 | -1.86e-4 | -1.86e-4 | -1.86e-4 | -2.11e-5 |
| 194 | 3.00e-3 | 2 | 6.60e-3 | 6.97e-3 | 6.79e-3 | 6.60e-3 | 191 | -2.82e-4 | +1.83e-6 | -1.40e-4 | -4.52e-5 |
| 195 | 3.00e-3 | 1 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 6.79e-3 | 199 | +1.39e-4 | +1.39e-4 | +1.39e-4 | -2.67e-5 |
| 196 | 3.00e-3 | 2 | 6.51e-3 | 6.83e-3 | 6.67e-3 | 6.51e-3 | 183 | -2.68e-4 | +3.29e-5 | -1.18e-4 | -4.55e-5 |
| 197 | 3.00e-3 | 1 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 6.69e-3 | 194 | +1.41e-4 | +1.41e-4 | +1.41e-4 | -2.68e-5 |
| 198 | 3.00e-3 | 1 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 6.77e-3 | 190 | +6.59e-5 | +6.59e-5 | +6.59e-5 | -1.76e-5 |
| 199 | 3.00e-3 | 2 | 6.71e-3 | 7.14e-3 | 6.93e-3 | 6.71e-3 | 197 | -3.17e-4 | +2.30e-4 | -4.38e-5 | -2.53e-5 |

