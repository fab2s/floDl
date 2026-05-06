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
| nccl-async | 0.044611 | 0.9227 | +0.0102 | 1949.8 | 754 | 38.5 | 100% | 100% | 5.5 |

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
| nccl-async | 1.9973 | 0.7226 | 0.5478 | 0.4828 | 0.4634 | 0.4446 | 0.4288 | 0.4269 | 0.4204 | 0.4148 | 0.1715 | 0.1456 | 0.1324 | 0.1263 | 0.1200 | 0.0611 | 0.0543 | 0.0494 | 0.0465 | 0.0446 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3996 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3059 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2945 | 3.4 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 404 | 400 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu2 | 1948.7 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu1 | 1948.8 | 0.9 | epoch-boundary(199) |
| nccl-async | gpu0 | 1948.6 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 1.4s |
| resnet-graph | nccl-async | gpu1 | 0.9s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.8s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 25 | 0 | 754 | 38.5 | 1517/9621 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 187.7 | 9.6% |

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
| resnet-graph | nccl-async | 188 | 754 | 0 | 6.57e-3 | +5.49e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 754 | 8.73e-2 | 3.93e-2 | 0.00e0 | 3.64e-1 | 43.4 | -6.93e-5 | 1.44e-3 |
| resnet-graph | nccl-async | 1 | 754 | 8.94e-2 | 4.23e-2 | 0.00e0 | 3.71e-1 | 39.8 | -6.61e-5 | 2.06e-3 |
| resnet-graph | nccl-async | 2 | 754 | 8.82e-2 | 4.15e-2 | 0.00e0 | 3.34e-1 | 16.8 | -6.45e-5 | 2.07e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9813 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9799 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9967 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 92 (0,1,2,3,5,6,8,12…140,144) | 0 (—) | — | 0,1,2,3,5,6,8,12…140,144 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 40 | 40 |
| resnet-graph | nccl-async | 0e0 | 5 | 15 | 15 |
| resnet-graph | nccl-async | 0e0 | 10 | 4 | 4 |
| resnet-graph | nccl-async | 1e-4 | 3 | 5 | 5 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 589 | +0.025 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 109 | +0.098 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 51 | +0.073 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 751 | +0.020 | 187 | +0.255 | +0.359 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 752 | 3.33e1–7.83e1 | 7.01e1 | 1.89e-3 | 2.89e-3 | 3.78e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 591 | 74–78079 | +2.934e-6 | 0.153 | +2.972e-6 | 0.171 | 100 | +2.805e-6 | 0.658 | 38–264 | +4.843e-4 | 0.014 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 579 | 893–78079 | +2.734e-6 | 0.169 | +2.741e-6 | 0.194 | 99 | +2.672e-6 | 0.663 | 61–264 | +3.070e-4 | 0.007 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 110 | 78295–115719 | +5.205e-5 | 0.685 | +5.258e-5 | 0.689 | 45 | +5.018e-5 | 0.859 | 107–1049 | +1.811e-3 | 0.747 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 52 | 116466–155976 | -1.578e-5 | 0.114 | -1.568e-5 | 0.113 | 43 | -9.943e-6 | 0.082 | 620–985 | -1.578e-4 | 0.001 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +4.843e-4 | r0: +7.318e-4, r1: +3.839e-4, r2: +3.506e-4 | r0: 0.045, r1: 0.007, r2: 0.006 | 2.09× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +3.070e-4 | r0: +6.019e-4, r1: +1.774e-4, r2: +1.549e-4 | r0: 0.046, r1: 0.002, r2: 0.001 | 3.88× | ⚠ framing breaking |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | +1.811e-3 | r0: +1.805e-3, r1: +1.822e-3, r2: +1.808e-3 | r0: 0.759, r1: 0.741, r2: 0.737 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | -1.578e-4 | r0: -1.182e-4, r1: -1.648e-4, r2: -1.894e-4 | r0: 0.000, r1: 0.001, r2: 0.001 | 1.60× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█████████████████████████▄▃▄▄▅▅▅▆▆▆▆▅▁▁▁▁▁▁▁▁▁▁` | `▁▇█▇█▇▇▇█▇▆█▇█▇▇█▇▇▇▇█▇▇▇▁▇█▇███████▆▆▇▇▇▇█▇▇██` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 3.71e-1 | 9.70e-2 | 7.92e-2 | 30 | -4.79e-2 | +7.19e-3 | -9.64e-3 | -5.21e-3 |
| 1 | 3.00e-1 | 8 | 7.29e-2 | 1.06e-1 | 8.40e-2 | 8.37e-2 | 27 | -1.17e-2 | +9.91e-3 | +3.14e-4 | -1.62e-3 |
| 2 | 3.00e-1 | 10 | 7.85e-2 | 1.30e-1 | 9.14e-2 | 8.14e-2 | 28 | -1.47e-2 | +1.47e-2 | -7.63e-5 | -7.02e-4 |
| 3 | 3.00e-1 | 8 | 8.31e-2 | 1.22e-1 | 9.58e-2 | 1.02e-1 | 35 | -1.32e-2 | +1.16e-2 | +5.19e-4 | +4.32e-6 |
| 4 | 3.00e-1 | 7 | 8.62e-2 | 1.39e-1 | 9.94e-2 | 8.88e-2 | 33 | -9.23e-3 | +9.46e-3 | -5.74e-4 | -3.69e-4 |
| 5 | 3.00e-1 | 7 | 8.28e-2 | 1.34e-1 | 9.71e-2 | 9.59e-2 | 38 | -1.30e-2 | +1.07e-2 | +1.66e-4 | -1.12e-4 |
| 6 | 3.00e-1 | 8 | 9.17e-2 | 1.27e-1 | 1.03e-1 | 1.08e-1 | 43 | -9.03e-3 | +8.13e-3 | +4.11e-4 | +1.79e-4 |
| 7 | 3.00e-1 | 5 | 9.60e-2 | 1.41e-1 | 1.12e-1 | 9.60e-2 | 36 | -5.27e-3 | +6.02e-3 | -8.37e-4 | -3.44e-4 |
| 8 | 3.00e-1 | 6 | 9.26e-2 | 1.41e-1 | 1.04e-1 | 9.95e-2 | 36 | -1.04e-2 | +1.05e-2 | +2.15e-4 | -1.36e-4 |
| 9 | 3.00e-1 | 7 | 8.94e-2 | 1.33e-1 | 1.01e-1 | 1.00e-1 | 32 | -9.64e-3 | +8.78e-3 | +5.45e-5 | -1.06e-5 |
| 10 | 3.00e-1 | 9 | 8.64e-2 | 1.39e-1 | 9.96e-2 | 9.95e-2 | 35 | -1.39e-2 | +1.42e-2 | +3.35e-4 | +1.58e-4 |
| 11 | 3.00e-1 | 5 | 9.30e-2 | 1.38e-1 | 1.05e-1 | 1.01e-1 | 41 | -9.14e-3 | +1.01e-2 | +1.54e-4 | +1.06e-4 |
| 12 | 3.00e-1 | 6 | 8.59e-2 | 1.45e-1 | 1.03e-1 | 8.59e-2 | 33 | -1.26e-2 | +9.30e-3 | -8.15e-4 | -4.45e-4 |
| 13 | 3.00e-1 | 7 | 8.84e-2 | 1.29e-1 | 1.02e-1 | 1.06e-1 | 41 | -1.01e-2 | +8.95e-3 | +6.61e-4 | +1.00e-4 |
| 14 | 3.00e-1 | 8 | 9.61e-2 | 1.31e-1 | 1.05e-1 | 9.96e-2 | 40 | -6.31e-3 | +6.76e-3 | -1.09e-4 | -6.55e-5 |
| 15 | 3.00e-1 | 4 | 9.88e-2 | 1.43e-1 | 1.10e-1 | 9.88e-2 | 35 | -9.20e-3 | +7.92e-3 | -3.14e-4 | -2.28e-4 |
| 16 | 3.00e-1 | 8 | 8.64e-2 | 1.40e-1 | 9.91e-2 | 1.02e-1 | 40 | -1.42e-2 | +1.34e-2 | +3.49e-4 | +1.09e-4 |
| 17 | 3.00e-1 | 5 | 9.45e-2 | 1.32e-1 | 1.05e-1 | 9.92e-2 | 43 | -7.84e-3 | +8.11e-3 | +9.69e-5 | +4.95e-5 |
| 18 | 3.00e-1 | 6 | 9.64e-2 | 1.35e-1 | 1.06e-1 | 1.01e-1 | 40 | -8.48e-3 | +6.51e-3 | -3.29e-5 | -3.05e-5 |
| 19 | 3.00e-1 | 6 | 9.37e-2 | 1.36e-1 | 1.06e-1 | 1.01e-1 | 46 | -9.23e-3 | +7.03e-3 | -3.36e-5 | -6.92e-5 |
| 20 | 3.00e-1 | 6 | 9.72e-2 | 1.35e-1 | 1.07e-1 | 9.94e-2 | 43 | -8.11e-3 | +7.11e-3 | -1.98e-5 | -1.15e-4 |
| 21 | 3.00e-1 | 6 | 9.54e-2 | 1.43e-1 | 1.09e-1 | 9.81e-2 | 36 | -6.57e-3 | +7.63e-3 | -4.33e-5 | -1.76e-4 |
| 22 | 3.00e-1 | 6 | 9.33e-2 | 1.36e-1 | 1.07e-1 | 1.09e-1 | 46 | -7.75e-3 | +9.34e-3 | +5.46e-4 | +1.31e-4 |
| 23 | 3.00e-1 | 6 | 9.35e-2 | 1.42e-1 | 1.08e-1 | 1.06e-1 | 45 | -7.48e-3 | +6.90e-3 | -9.39e-5 | +3.26e-5 |
| 24 | 3.00e-1 | 6 | 9.47e-2 | 1.46e-1 | 1.08e-1 | 9.47e-2 | 40 | -8.81e-3 | +6.95e-3 | -6.37e-4 | -3.53e-4 |
| 25 | 3.00e-1 | 5 | 1.00e-1 | 1.34e-1 | 1.12e-1 | 1.06e-1 | 44 | -4.87e-3 | +6.25e-3 | +4.46e-4 | -1.02e-4 |
| 26 | 3.00e-1 | 9 | 9.82e-2 | 1.40e-1 | 1.10e-1 | 1.08e-1 | 32 | -7.76e-3 | +6.54e-3 | +1.89e-4 | +8.62e-5 |
| 27 | 3.00e-1 | 4 | 8.74e-2 | 1.36e-1 | 1.04e-1 | 8.74e-2 | 35 | -8.99e-3 | +9.21e-3 | -1.54e-3 | -5.82e-4 |
| 28 | 3.00e-1 | 6 | 9.23e-2 | 1.37e-1 | 1.03e-1 | 1.00e-1 | 40 | -1.08e-2 | +1.06e-2 | +4.38e-4 | -1.69e-4 |
| 29 | 3.00e-1 | 8 | 1.00e-1 | 1.38e-1 | 1.10e-1 | 1.08e-1 | 45 | -7.51e-3 | +7.40e-3 | +1.98e-4 | -1.70e-5 |
| 30 | 3.00e-1 | 4 | 9.20e-2 | 1.48e-1 | 1.10e-1 | 9.75e-2 | 37 | -1.21e-2 | +9.44e-3 | -3.71e-4 | -2.08e-4 |
| 31 | 3.00e-1 | 6 | 9.43e-2 | 1.35e-1 | 1.06e-1 | 1.04e-1 | 41 | -8.18e-3 | +9.70e-3 | +4.99e-4 | +6.67e-5 |
| 32 | 3.00e-1 | 8 | 9.73e-2 | 1.48e-1 | 1.08e-1 | 1.09e-1 | 38 | -7.84e-3 | +9.46e-3 | +2.64e-4 | +1.85e-4 |
| 33 | 3.00e-1 | 4 | 9.39e-2 | 1.39e-1 | 1.07e-1 | 9.88e-2 | 39 | -1.07e-2 | +1.09e-2 | -2.83e-5 | +5.52e-5 |
| 34 | 3.00e-1 | 6 | 9.62e-2 | 1.36e-1 | 1.10e-1 | 9.90e-2 | 41 | -7.27e-3 | +8.41e-3 | +1.06e-5 | -9.86e-5 |
| 35 | 3.00e-1 | 8 | 9.99e-2 | 1.32e-1 | 1.09e-1 | 1.05e-1 | 40 | -5.84e-3 | +6.78e-3 | +1.42e-4 | -3.20e-5 |
| 36 | 3.00e-1 | 4 | 9.24e-2 | 1.38e-1 | 1.06e-1 | 9.24e-2 | 47 | -9.30e-3 | +9.16e-3 | -4.51e-4 | -2.54e-4 |
| 37 | 3.00e-1 | 6 | 9.84e-2 | 1.40e-1 | 1.08e-1 | 1.02e-1 | 40 | -8.47e-3 | +5.70e-3 | +5.38e-6 | -1.83e-4 |
| 38 | 3.00e-1 | 6 | 1.02e-1 | 1.38e-1 | 1.11e-1 | 1.06e-1 | 45 | -6.63e-3 | +6.88e-3 | +1.55e-4 | -7.73e-5 |
| 39 | 3.00e-1 | 7 | 1.05e-1 | 1.49e-1 | 1.14e-1 | 1.06e-1 | 50 | -8.02e-3 | +7.41e-3 | -2.13e-5 | -1.20e-4 |
| 40 | 3.00e-1 | 4 | 9.36e-2 | 1.47e-1 | 1.18e-1 | 9.36e-2 | 37 | -7.46e-3 | +4.11e-3 | -1.77e-3 | -8.23e-4 |
| 41 | 3.00e-1 | 6 | 8.83e-2 | 1.46e-1 | 1.02e-1 | 9.87e-2 | 35 | -1.25e-2 | +1.23e-2 | +3.97e-4 | -2.92e-4 |
| 42 | 3.00e-1 | 8 | 9.58e-2 | 1.38e-1 | 1.04e-1 | 9.84e-2 | 40 | -9.00e-3 | +9.11e-3 | +1.62e-5 | -1.87e-4 |
| 43 | 3.00e-1 | 5 | 9.28e-2 | 1.41e-1 | 1.08e-1 | 1.04e-1 | 39 | -1.16e-2 | +7.69e-3 | +1.78e-6 | -1.37e-4 |
| 44 | 3.00e-1 | 6 | 9.96e-2 | 1.40e-1 | 1.08e-1 | 1.03e-1 | 40 | -7.08e-3 | +8.03e-3 | -1.18e-5 | -1.27e-4 |
| 45 | 3.00e-1 | 5 | 9.62e-2 | 1.44e-1 | 1.13e-1 | 1.08e-1 | 43 | -8.11e-3 | +8.26e-3 | +4.86e-4 | +6.79e-5 |
| 46 | 3.00e-1 | 6 | 9.97e-2 | 1.41e-1 | 1.10e-1 | 1.05e-1 | 43 | -7.87e-3 | +6.49e-3 | -1.82e-4 | -7.64e-5 |
| 47 | 3.00e-1 | 8 | 1.03e-1 | 1.39e-1 | 1.13e-1 | 1.14e-1 | 44 | -5.58e-3 | +5.84e-3 | +1.62e-4 | +3.69e-5 |
| 48 | 3.00e-1 | 4 | 9.90e-2 | 1.43e-1 | 1.12e-1 | 9.93e-2 | 39 | -9.49e-3 | +7.45e-3 | -6.93e-4 | -2.78e-4 |
| 49 | 3.00e-1 | 7 | 9.99e-2 | 1.48e-1 | 1.12e-1 | 1.13e-1 | 44 | -9.63e-3 | +9.18e-3 | +4.41e-4 | +6.70e-5 |
| 50 | 3.00e-1 | 4 | 9.83e-2 | 1.50e-1 | 1.14e-1 | 1.02e-1 | 36 | -1.06e-2 | +9.27e-3 | -2.67e-4 | -1.09e-4 |
| 51 | 3.00e-1 | 6 | 9.89e-2 | 1.43e-1 | 1.10e-1 | 1.04e-1 | 41 | -9.00e-3 | +8.74e-3 | +1.65e-4 | -4.16e-5 |
| 52 | 3.00e-1 | 6 | 9.95e-2 | 1.42e-1 | 1.12e-1 | 1.15e-1 | 39 | -7.04e-3 | +8.27e-3 | +4.00e-4 | +1.56e-4 |
| 53 | 3.00e-1 | 6 | 9.09e-2 | 1.42e-1 | 1.05e-1 | 1.03e-1 | 41 | -1.27e-2 | +9.28e-3 | -2.92e-4 | -2.80e-5 |
| 54 | 3.00e-1 | 6 | 9.97e-2 | 1.48e-1 | 1.12e-1 | 1.08e-1 | 44 | -8.93e-3 | +7.74e-3 | +1.49e-4 | +1.26e-5 |
| 55 | 3.00e-1 | 5 | 1.10e-1 | 1.43e-1 | 1.21e-1 | 1.19e-1 | 49 | -5.14e-3 | +4.86e-3 | +3.42e-4 | +1.13e-4 |
| 56 | 3.00e-1 | 8 | 1.05e-1 | 1.60e-1 | 1.16e-1 | 1.08e-1 | 44 | -7.25e-3 | +6.29e-3 | -1.74e-4 | -6.62e-5 |
| 57 | 3.00e-1 | 3 | 1.10e-1 | 1.44e-1 | 1.21e-1 | 1.10e-1 | 50 | -5.49e-3 | +5.77e-3 | +1.75e-4 | -5.82e-5 |
| 58 | 3.00e-1 | 5 | 1.05e-1 | 1.51e-1 | 1.18e-1 | 1.05e-1 | 44 | -7.41e-3 | +5.91e-3 | -2.25e-4 | -1.95e-4 |
| 59 | 3.00e-1 | 5 | 1.03e-1 | 1.46e-1 | 1.16e-1 | 1.12e-1 | 47 | -5.73e-3 | +7.12e-3 | +2.53e-4 | -6.07e-5 |
| 60 | 3.00e-1 | 6 | 9.95e-2 | 1.50e-1 | 1.12e-1 | 1.06e-1 | 44 | -1.03e-2 | +7.35e-3 | -2.83e-4 | -1.80e-4 |
| 61 | 3.00e-1 | 5 | 1.05e-1 | 1.46e-1 | 1.16e-1 | 1.05e-1 | 44 | -5.39e-3 | +6.32e-3 | -1.88e-4 | -2.60e-4 |
| 62 | 3.00e-1 | 6 | 9.89e-2 | 1.52e-1 | 1.13e-1 | 1.02e-1 | 40 | -8.39e-3 | +8.02e-3 | +1.61e-4 | -1.53e-4 |
| 63 | 3.00e-1 | 6 | 9.63e-2 | 1.43e-1 | 1.10e-1 | 1.04e-1 | 43 | -8.99e-3 | +7.90e-3 | +1.45e-4 | -7.80e-5 |
| 64 | 3.00e-1 | 6 | 1.05e-1 | 1.52e-1 | 1.16e-1 | 1.14e-1 | 46 | -7.86e-3 | +8.01e-3 | +2.71e-4 | +5.04e-5 |
| 65 | 3.00e-1 | 5 | 1.06e-1 | 1.50e-1 | 1.20e-1 | 1.17e-1 | 46 | -6.00e-3 | +7.06e-3 | +2.49e-4 | +9.99e-5 |
| 66 | 3.00e-1 | 5 | 1.03e-1 | 1.43e-1 | 1.17e-1 | 1.17e-1 | 49 | -5.42e-3 | +6.96e-3 | +1.97e-4 | +1.27e-4 |
| 67 | 3.00e-1 | 5 | 1.09e-1 | 1.46e-1 | 1.19e-1 | 1.09e-1 | 46 | -5.36e-3 | +4.81e-3 | -4.51e-4 | -1.54e-4 |
| 68 | 3.00e-1 | 5 | 1.11e-1 | 1.59e-1 | 1.25e-1 | 1.11e-1 | 39 | -5.47e-3 | +5.99e-3 | -1.72e-4 | -2.63e-4 |
| 69 | 3.00e-1 | 6 | 9.51e-2 | 1.46e-1 | 1.07e-1 | 1.03e-1 | 40 | -1.10e-2 | +9.28e-3 | -7.70e-5 | -1.80e-4 |
| 70 | 3.00e-1 | 6 | 1.01e-1 | 1.58e-1 | 1.13e-1 | 1.01e-1 | 40 | -9.27e-3 | +9.23e-3 | -1.88e-4 | -2.76e-4 |
| 71 | 3.00e-1 | 6 | 1.03e-1 | 1.53e-1 | 1.17e-1 | 1.09e-1 | 41 | -8.36e-3 | +9.06e-3 | +2.81e-4 | -1.13e-4 |
| 72 | 3.00e-1 | 5 | 1.04e-1 | 1.54e-1 | 1.18e-1 | 1.15e-1 | 44 | -7.81e-3 | +8.87e-3 | +4.66e-4 | +7.93e-5 |
| 73 | 3.00e-1 | 5 | 1.07e-1 | 1.50e-1 | 1.18e-1 | 1.10e-1 | 47 | -7.49e-3 | +7.63e-3 | -5.34e-5 | -2.54e-5 |
| 74 | 3.00e-1 | 5 | 1.09e-1 | 1.60e-1 | 1.25e-1 | 1.12e-1 | 46 | -5.72e-3 | +7.10e-3 | +5.30e-5 | -8.04e-5 |
| 75 | 3.00e-1 | 6 | 9.38e-2 | 1.55e-1 | 1.10e-1 | 1.03e-1 | 42 | -1.25e-2 | +8.98e-3 | -2.71e-4 | -1.92e-4 |
| 76 | 3.00e-1 | 6 | 1.02e-1 | 1.48e-1 | 1.16e-1 | 1.20e-1 | 55 | -8.59e-3 | +7.56e-3 | +4.26e-4 | +7.83e-5 |
| 77 | 3.00e-1 | 5 | 1.09e-1 | 1.59e-1 | 1.24e-1 | 1.17e-1 | 47 | -7.93e-3 | +5.37e-3 | -1.91e-4 | -5.78e-5 |
| 78 | 3.00e-1 | 5 | 1.07e-1 | 1.41e-1 | 1.17e-1 | 1.16e-1 | 50 | -5.37e-3 | +4.89e-3 | +5.46e-5 | -1.57e-5 |
| 79 | 3.00e-1 | 5 | 1.06e-1 | 1.54e-1 | 1.21e-1 | 1.06e-1 | 42 | -6.04e-3 | +4.95e-3 | -5.49e-4 | -3.04e-4 |
| 80 | 3.00e-1 | 7 | 1.04e-1 | 1.58e-1 | 1.14e-1 | 1.07e-1 | 43 | -8.56e-3 | +8.81e-3 | +8.51e-5 | -1.69e-4 |
| 81 | 3.00e-1 | 4 | 1.07e-1 | 1.64e-1 | 1.21e-1 | 1.08e-1 | 43 | -8.98e-3 | +9.78e-3 | +2.33e-4 | -1.13e-4 |
| 82 | 3.00e-1 | 7 | 1.05e-1 | 1.50e-1 | 1.13e-1 | 1.05e-1 | 45 | -7.94e-3 | +7.19e-3 | -1.34e-4 | -1.80e-4 |
| 83 | 3.00e-1 | 4 | 1.05e-1 | 1.52e-1 | 1.20e-1 | 1.13e-1 | 48 | -7.71e-3 | +7.06e-3 | +3.12e-4 | -6.03e-5 |
| 84 | 3.00e-1 | 5 | 1.12e-1 | 1.51e-1 | 1.22e-1 | 1.25e-1 | 44 | -6.26e-3 | +6.42e-3 | +4.67e-4 | +1.48e-4 |
| 85 | 3.00e-1 | 5 | 1.04e-1 | 1.53e-1 | 1.16e-1 | 1.06e-1 | 47 | -8.67e-3 | +7.16e-3 | -4.93e-4 | -1.52e-4 |
| 86 | 3.00e-1 | 5 | 1.13e-1 | 1.51e-1 | 1.22e-1 | 1.17e-1 | 55 | -5.88e-3 | +5.69e-3 | +2.88e-4 | -2.29e-5 |
| 87 | 3.00e-1 | 7 | 1.05e-1 | 1.49e-1 | 1.19e-1 | 1.20e-1 | 63 | -7.25e-3 | +4.12e-3 | +5.95e-6 | +6.04e-6 |
| 88 | 3.00e-1 | 3 | 1.15e-1 | 1.55e-1 | 1.35e-1 | 1.15e-1 | 49 | -6.09e-3 | +2.74e-3 | -7.65e-4 | -2.73e-4 |
| 89 | 3.00e-1 | 5 | 1.10e-1 | 1.50e-1 | 1.21e-1 | 1.16e-1 | 49 | -5.40e-3 | +6.34e-3 | +1.38e-4 | -1.44e-4 |
| 90 | 3.00e-1 | 7 | 1.03e-1 | 1.54e-1 | 1.20e-1 | 1.03e-1 | 44 | -7.40e-3 | +5.56e-3 | -4.35e-4 | -4.00e-4 |
| 91 | 3.00e-1 | 4 | 1.03e-1 | 1.53e-1 | 1.20e-1 | 1.16e-1 | 46 | -6.87e-3 | +9.05e-3 | +8.20e-4 | -3.64e-5 |
| 92 | 3.00e-1 | 5 | 1.01e-1 | 1.54e-1 | 1.19e-1 | 1.17e-1 | 46 | -9.70e-3 | +6.62e-3 | +1.07e-5 | -2.24e-5 |
| 93 | 3.00e-1 | 6 | 1.05e-1 | 1.64e-1 | 1.18e-1 | 1.05e-1 | 40 | -1.01e-2 | +7.75e-3 | -4.47e-4 | -2.78e-4 |
| 94 | 3.00e-1 | 5 | 1.02e-1 | 1.45e-1 | 1.19e-1 | 1.16e-1 | 47 | -5.35e-3 | +7.55e-3 | +4.83e-4 | -2.64e-5 |
| 95 | 3.00e-1 | 5 | 1.12e-1 | 1.46e-1 | 1.22e-1 | 1.20e-1 | 58 | -5.23e-3 | +4.90e-3 | +1.34e-4 | +1.65e-5 |
| 96 | 3.00e-1 | 6 | 1.07e-1 | 1.48e-1 | 1.20e-1 | 1.19e-1 | 41 | -7.67e-3 | +2.94e-3 | -2.34e-4 | -5.76e-5 |
| 97 | 3.00e-1 | 5 | 1.02e-1 | 1.48e-1 | 1.17e-1 | 1.25e-1 | 59 | -8.31e-3 | +8.62e-3 | +4.10e-4 | +1.46e-4 |
| 98 | 3.00e-1 | 5 | 1.02e-1 | 1.59e-1 | 1.23e-1 | 1.02e-1 | 40 | -7.35e-3 | +4.88e-3 | -1.08e-3 | -4.39e-4 |
| 99 | 3.00e-1 | 6 | 9.88e-2 | 1.46e-1 | 1.16e-1 | 1.15e-1 | 54 | -8.11e-3 | +9.13e-3 | +6.51e-4 | +7.00e-6 |
| 100 | 3.00e-2 | 7 | 1.02e-2 | 1.24e-1 | 3.53e-2 | 1.03e-2 | 45 | -3.72e-2 | +8.95e-4 | -6.83e-3 | -3.09e-3 |
| 101 | 3.00e-2 | 3 | 1.03e-2 | 1.39e-2 | 1.19e-2 | 1.16e-2 | 45 | -4.04e-3 | +6.04e-3 | +6.45e-4 | -2.12e-3 |
| 102 | 3.00e-2 | 7 | 1.06e-2 | 1.44e-2 | 1.19e-2 | 1.14e-2 | 47 | -6.66e-3 | +5.05e-3 | -5.40e-5 | -1.08e-3 |
| 103 | 3.00e-2 | 4 | 1.16e-2 | 1.53e-2 | 1.26e-2 | 1.17e-2 | 52 | -5.69e-3 | +6.12e-3 | +1.22e-4 | -7.26e-4 |
| 104 | 3.00e-2 | 5 | 1.31e-2 | 1.67e-2 | 1.42e-2 | 1.40e-2 | 48 | -4.40e-3 | +4.60e-3 | +5.03e-4 | -2.69e-4 |
| 105 | 3.00e-2 | 6 | 1.20e-2 | 1.73e-2 | 1.32e-2 | 1.30e-2 | 39 | -8.65e-3 | +7.77e-3 | -1.37e-5 | -1.35e-4 |
| 106 | 3.00e-2 | 5 | 1.15e-2 | 1.75e-2 | 1.33e-2 | 1.18e-2 | 45 | -7.26e-3 | +9.58e-3 | -2.49e-4 | -2.75e-4 |
| 107 | 3.00e-2 | 6 | 1.30e-2 | 1.75e-2 | 1.47e-2 | 1.51e-2 | 51 | -5.84e-3 | +5.48e-3 | +6.87e-4 | +1.24e-4 |
| 108 | 3.00e-2 | 4 | 1.35e-2 | 1.97e-2 | 1.57e-2 | 1.44e-2 | 45 | -8.41e-3 | +5.66e-3 | -2.91e-4 | -6.13e-5 |
| 109 | 3.00e-2 | 6 | 1.28e-2 | 2.06e-2 | 1.46e-2 | 1.45e-2 | 44 | -1.08e-2 | +1.06e-2 | +1.72e-4 | +4.22e-5 |
| 110 | 3.00e-2 | 5 | 1.41e-2 | 1.91e-2 | 1.55e-2 | 1.51e-2 | 48 | -5.63e-3 | +6.36e-3 | +2.31e-4 | +8.69e-5 |
| 111 | 3.00e-2 | 7 | 1.52e-2 | 2.14e-2 | 1.66e-2 | 1.58e-2 | 51 | -5.69e-3 | +6.67e-3 | +1.32e-4 | +4.91e-5 |
| 112 | 3.00e-2 | 3 | 1.59e-2 | 2.19e-2 | 1.80e-2 | 1.61e-2 | 48 | -6.40e-3 | +6.62e-3 | +1.04e-4 | -8.67e-7 |
| 113 | 3.00e-2 | 7 | 1.54e-2 | 2.15e-2 | 1.68e-2 | 1.59e-2 | 48 | -6.96e-3 | +5.79e-3 | -1.72e-4 | -1.27e-4 |
| 114 | 3.00e-2 | 2 | 1.43e-2 | 1.60e-2 | 1.52e-2 | 1.43e-2 | 39 | -2.80e-3 | +2.21e-4 | -1.29e-3 | -3.63e-4 |
| 115 | 3.00e-2 | 2 | 1.55e-2 | 3.47e-2 | 2.51e-2 | 3.47e-2 | 228 | +3.53e-4 | +3.54e-3 | +1.95e-3 | +9.15e-5 |
| 117 | 3.00e-2 | 2 | 3.61e-2 | 4.03e-2 | 3.82e-2 | 4.03e-2 | 235 | +1.30e-4 | +4.72e-4 | +3.01e-4 | +1.33e-4 |
| 118 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 276 | -2.80e-4 | -2.80e-4 | -2.80e-4 | +9.16e-5 |
| 119 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 292 | +2.44e-4 | +2.44e-4 | +2.44e-4 | +1.07e-4 |
| 121 | 3.00e-2 | 1 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 4.23e-2 | 350 | +1.56e-4 | +1.56e-4 | +1.56e-4 | +1.12e-4 |
| 122 | 3.00e-2 | 2 | 4.20e-2 | 4.58e-2 | 4.39e-2 | 4.20e-2 | 262 | -3.24e-4 | +2.90e-4 | -1.68e-5 | +8.43e-5 |
| 124 | 3.00e-2 | 1 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 350 | +7.21e-5 | +7.21e-5 | +7.21e-5 | +8.31e-5 |
| 125 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 302 | +4.22e-4 | +4.22e-4 | +4.22e-4 | +1.17e-4 |
| 126 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 258 | -1.23e-4 | -1.23e-4 | -1.23e-4 | +9.30e-5 |
| 127 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 268 | -1.76e-4 | -1.76e-4 | -1.76e-4 | +6.61e-5 |
| 128 | 3.00e-2 | 1 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 4.62e-2 | 287 | +7.38e-5 | +7.38e-5 | +7.38e-5 | +6.69e-5 |
| 129 | 3.00e-2 | 1 | 4.82e-2 | 4.82e-2 | 4.82e-2 | 4.82e-2 | 278 | +1.52e-4 | +1.52e-4 | +1.52e-4 | +7.54e-5 |
| 130 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 286 | -6.48e-5 | -6.48e-5 | -6.48e-5 | +6.14e-5 |
| 131 | 3.00e-2 | 1 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 4.87e-2 | 270 | +1.03e-4 | +1.03e-4 | +1.03e-4 | +6.56e-5 |
| 132 | 3.00e-2 | 1 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 278 | +3.81e-5 | +3.81e-5 | +3.81e-5 | +6.28e-5 |
| 133 | 3.00e-2 | 1 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 5.02e-2 | 285 | +6.84e-5 | +6.84e-5 | +6.84e-5 | +6.34e-5 |
| 134 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 306 | +5.09e-5 | +5.09e-5 | +5.09e-5 | +6.21e-5 |
| 135 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 279 | +1.60e-4 | +1.60e-4 | +1.60e-4 | +7.20e-5 |
| 136 | 3.00e-2 | 1 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 5.33e-2 | 264 | +2.70e-6 | +2.70e-6 | +2.70e-6 | +6.50e-5 |
| 137 | 3.00e-2 | 1 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 5.41e-2 | 292 | +4.89e-5 | +4.89e-5 | +4.89e-5 | +6.34e-5 |
| 138 | 3.00e-2 | 1 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 5.58e-2 | 315 | +9.94e-5 | +9.94e-5 | +9.94e-5 | +6.70e-5 |
| 139 | 3.00e-2 | 1 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 311 | +6.77e-5 | +6.77e-5 | +6.77e-5 | +6.71e-5 |
| 140 | 3.00e-2 | 1 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 316 | +7.64e-5 | +7.64e-5 | +7.64e-5 | +6.80e-5 |
| 141 | 3.00e-2 | 1 | 5.82e-2 | 5.82e-2 | 5.82e-2 | 5.82e-2 | 341 | -7.57e-6 | -7.57e-6 | -7.57e-6 | +6.05e-5 |
| 143 | 3.00e-2 | 1 | 6.35e-2 | 6.35e-2 | 6.35e-2 | 6.35e-2 | 396 | +2.20e-4 | +2.20e-4 | +2.20e-4 | +7.64e-5 |
| 144 | 3.00e-2 | 1 | 6.74e-2 | 6.74e-2 | 6.74e-2 | 6.74e-2 | 310 | +1.92e-4 | +1.92e-4 | +1.92e-4 | +8.79e-5 |
| 145 | 3.00e-2 | 1 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 245 | -3.78e-4 | -3.78e-4 | -3.78e-4 | +4.14e-5 |
| 146 | 3.00e-2 | 1 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 5.63e-2 | 258 | -3.38e-4 | -3.38e-4 | -3.38e-4 | +3.45e-6 |
| 147 | 3.00e-2 | 1 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 269 | +1.86e-4 | +1.86e-4 | +1.86e-4 | +2.17e-5 |
| 148 | 3.00e-2 | 1 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 5.88e-2 | 260 | -2.54e-5 | -2.54e-5 | -2.54e-5 | +1.70e-5 |
| 149 | 3.00e-3 | 2 | 5.87e-2 | 6.15e-2 | 6.01e-2 | 6.15e-2 | 247 | -9.35e-6 | +1.88e-4 | +8.94e-5 | +3.17e-5 |
| 151 | 3.00e-3 | 2 | 6.17e-3 | 6.03e-2 | 3.32e-2 | 6.17e-3 | 247 | -9.23e-3 | -6.55e-5 | -4.65e-3 | -9.03e-4 |
| 153 | 3.00e-3 | 2 | 5.37e-3 | 6.20e-3 | 5.78e-3 | 6.20e-3 | 247 | -3.86e-4 | +5.82e-4 | +9.81e-5 | -7.08e-4 |
| 155 | 3.00e-3 | 2 | 5.26e-3 | 5.89e-3 | 5.57e-3 | 5.89e-3 | 247 | -5.18e-4 | +4.62e-4 | -2.81e-5 | -5.74e-4 |
| 157 | 3.00e-3 | 1 | 5.20e-3 | 5.20e-3 | 5.20e-3 | 5.20e-3 | 330 | -3.78e-4 | -3.78e-4 | -3.78e-4 | -5.54e-4 |
| 158 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 286 | +5.51e-4 | +5.51e-4 | +5.51e-4 | -4.44e-4 |
| 159 | 3.00e-3 | 2 | 5.53e-3 | 5.62e-3 | 5.57e-3 | 5.53e-3 | 247 | -3.11e-4 | -7.08e-5 | -1.91e-4 | -3.94e-4 |
| 160 | 3.00e-3 | 1 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 251 | -2.38e-5 | -2.38e-5 | -2.38e-5 | -3.57e-4 |
| 161 | 3.00e-3 | 1 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 290 | +8.76e-7 | +8.76e-7 | +8.76e-7 | -3.21e-4 |
| 162 | 3.00e-3 | 1 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 5.90e-3 | 321 | +2.24e-4 | +2.24e-4 | +2.24e-4 | -2.67e-4 |
| 164 | 3.00e-3 | 2 | 6.16e-3 | 6.22e-3 | 6.19e-3 | 6.16e-3 | 247 | -3.75e-5 | +1.57e-4 | +5.97e-5 | -2.06e-4 |
| 166 | 3.00e-3 | 1 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 5.62e-3 | 362 | -2.56e-4 | -2.56e-4 | -2.56e-4 | -2.11e-4 |
| 167 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 256 | +4.69e-4 | +4.69e-4 | +4.69e-4 | -1.43e-4 |
| 168 | 3.00e-3 | 1 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 5.63e-3 | 251 | -4.68e-4 | -4.68e-4 | -4.68e-4 | -1.75e-4 |
| 169 | 3.00e-3 | 2 | 5.64e-3 | 5.77e-3 | 5.70e-3 | 5.77e-3 | 226 | +5.11e-6 | +9.75e-5 | +5.13e-5 | -1.32e-4 |
| 170 | 3.00e-3 | 1 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 5.44e-3 | 283 | -2.03e-4 | -2.03e-4 | -2.03e-4 | -1.39e-4 |
| 172 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 329 | +3.21e-4 | +3.21e-4 | +3.21e-4 | -9.30e-5 |
| 173 | 3.00e-3 | 2 | 6.01e-3 | 6.42e-3 | 6.21e-3 | 6.01e-3 | 239 | -2.75e-4 | +2.18e-4 | -2.86e-5 | -8.32e-5 |
| 174 | 3.00e-3 | 1 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 239 | -3.66e-4 | -3.66e-4 | -3.66e-4 | -1.11e-4 |
| 175 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 254 | +3.70e-4 | +3.70e-4 | +3.70e-4 | -6.33e-5 |
| 176 | 3.00e-3 | 1 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 5.85e-3 | 265 | -1.28e-4 | -1.28e-4 | -1.28e-4 | -6.98e-5 |
| 177 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 255 | +1.12e-4 | +1.12e-4 | +1.12e-4 | -5.16e-5 |
| 178 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 299 | -5.88e-5 | -5.88e-5 | -5.88e-5 | -5.24e-5 |
| 179 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 305 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -3.37e-5 |
| 180 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 304 | +9.17e-5 | +9.17e-5 | +9.17e-5 | -2.12e-5 |
| 181 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 303 | -4.79e-7 | -4.79e-7 | -4.79e-7 | -1.91e-5 |
| 183 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 380 | -2.16e-5 | -2.16e-5 | -2.16e-5 | -1.94e-5 |
| 184 | 3.00e-3 | 2 | 6.36e-3 | 6.96e-3 | 6.66e-3 | 6.36e-3 | 224 | -4.02e-4 | +3.58e-4 | -2.20e-5 | -2.37e-5 |
| 185 | 3.00e-3 | 1 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 5.57e-3 | 255 | -5.21e-4 | -5.21e-4 | -5.21e-4 | -7.34e-5 |
| 186 | 3.00e-3 | 1 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 6.11e-3 | 300 | +3.10e-4 | +3.10e-4 | +3.10e-4 | -3.51e-5 |
| 187 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 255 | +2.32e-4 | +2.32e-4 | +2.32e-4 | -8.34e-6 |
| 188 | 3.00e-3 | 1 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 6.02e-3 | 257 | -2.90e-4 | -2.90e-4 | -2.90e-4 | -3.65e-5 |
| 189 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 265 | +6.96e-5 | +6.96e-5 | +6.96e-5 | -2.59e-5 |
| 190 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 237 | -1.22e-6 | -1.22e-6 | -1.22e-6 | -2.35e-5 |
| 191 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 250 | -2.41e-4 | -2.41e-4 | -2.41e-4 | -4.52e-5 |
| 192 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 269 | +2.29e-4 | +2.29e-4 | +2.29e-4 | -1.78e-5 |
| 193 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 269 | +1.15e-4 | +1.15e-4 | +1.15e-4 | -4.52e-6 |
| 194 | 3.00e-3 | 1 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 6.05e-3 | 266 | -1.67e-4 | -1.67e-4 | -1.67e-4 | -2.08e-5 |
| 195 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 298 | +2.09e-4 | +2.09e-4 | +2.09e-4 | +2.21e-6 |
| 196 | 3.00e-3 | 1 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 6.65e-3 | 294 | +1.08e-4 | +1.08e-4 | +1.08e-4 | +1.27e-5 |
| 197 | 3.00e-3 | 1 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 6.66e-3 | 289 | +8.00e-6 | +8.00e-6 | +8.00e-6 | +1.23e-5 |
| 198 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 280 | -7.41e-5 | -7.41e-5 | -7.41e-5 | +3.63e-6 |
| 199 | 3.00e-3 | 1 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 272 | +2.21e-5 | +2.21e-5 | +2.21e-5 | +5.49e-6 |

