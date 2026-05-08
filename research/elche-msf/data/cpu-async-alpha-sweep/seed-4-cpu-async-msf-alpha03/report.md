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
| cpu-async | 0.066195 | 0.9126 | +0.0001 | 1862.6 | 719 | 79.3 | 100% | 100% | 100% | 10.2 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9126 | cpu-async | - | - |

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
| cpu-async | 2.0323 | 0.8043 | 0.6145 | 0.5475 | 0.5438 | 0.5286 | 0.5161 | 0.4972 | 0.4890 | 0.4847 | 0.2251 | 0.1850 | 0.1680 | 0.1544 | 0.1398 | 0.0846 | 0.0805 | 0.0715 | 0.0685 | 0.0662 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4001 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3043 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2956 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 392 | 388 | 400 | 389 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 341.8 | 3.2 | epoch-boundary(36) |
| cpu-async | gpu1 | 498.2 | 1.9 | epoch-boundary(53) |
| cpu-async | gpu2 | 498.1 | 1.9 | epoch-boundary(53) |
| cpu-async | gpu1 | 619.2 | 1.1 | epoch-boundary(66) |
| cpu-async | gpu2 | 766.5 | 0.5 | epoch-boundary(82) |
| cpu-async | gpu1 | 1862.1 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 3.5s | 0.0s | 0.0s | 0.0s | 4.0s |
| resnet-graph | cpu-async | gpu2 | 5.7s | 0.0s | 0.0s | 0.0s | 6.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 344 | 0 | 719 | 79.3 | 1400/8703 | 719 | 79.3 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 191.4 | 10.3% |

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
| resnet-graph | cpu-async | 191 | 719 | 0 | 3.77e-3 | -5.18e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 719 | 7.52e-2 | 8.29e-2 | 2.78e-3 | 5.54e-1 | 33.1 | -1.35e-4 | 7.16e-4 |
| resnet-graph | cpu-async | 1 | 719 | 7.60e-2 | 8.65e-2 | 2.84e-3 | 7.35e-1 | 32.1 | -1.83e-4 | 1.21e-3 |
| resnet-graph | cpu-async | 2 | 719 | 7.63e-2 | 8.89e-2 | 2.82e-3 | 8.52e-1 | 34.8 | -2.14e-4 | 1.30e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9840 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9822 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9791 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 40 (4,6,7,10,11,12,13,15…139,150) | 0 (—) | — | 4,6,7,10,11,12,13,15…139,150 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 299 | -0.093 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 113 | -0.090 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 303 | -0.107 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 717 | -0.010 | 190 | +0.242 | +0.421 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 718 | 3.75e1–8.05e1 | 6.44e1 | 1.70e-3 | 5.59e-3 | 1.25e-1 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 301 | 36–77657 | +8.343e-6 | 0.266 | +8.842e-6 | 0.329 | 92 | +9.143e-6 | 0.552 | 27–1065 | +8.365e-4 | 0.556 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 292 | 994–77657 | +1.042e-5 | 0.600 | +1.071e-5 | 0.644 | 91 | +9.880e-6 | 0.652 | 74–1065 | +9.187e-4 | 0.920 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 114 | 78327–117128 | -3.795e-6 | 0.026 | -4.170e-6 | 0.032 | 49 | -5.382e-6 | 0.039 | 111–670 | +3.983e-4 | 0.050 |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | 304 | 117338–156162 | +1.768e-6 | 0.007 | +1.696e-6 | 0.008 | 50 | +1.911e-6 | 0.012 | 80–270 | +2.428e-3 | 0.167 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +8.365e-4 | r0: +8.312e-4, r1: +8.382e-4, r2: +8.447e-4 | r0: 0.575, r1: 0.537, r2: 0.530 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.187e-4 | r0: +9.101e-4, r1: +9.206e-4, r2: +9.271e-4 | r0: 0.902, r1: 0.891, r2: 0.895 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +3.983e-4 | r0: +3.647e-4, r1: +4.055e-4, r2: +4.259e-4 | r0: 0.042, r1: 0.053, r2: 0.055 | 1.17× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–199 | +2.428e-3 | r0: +2.024e-3, r1: +2.477e-3, r2: +2.718e-3 | r0: 0.120, r1: 0.159, r2: 0.196 | 1.34× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇██████████████▆▅▅▅▅▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇▇▇▇██████████████▇▇█████████▇▄▇▇▇▇▇▇▇▇▇▇▇▇` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 9 | 1.99e-1 | 8.52e-1 | 4.55e-1 | 1.99e-1 | 63 | -2.94e-2 | -6.66e-4 | -1.05e-2 | -9.47e-3 |
| 1 | 3.00e-1 | 5 | 1.64e-1 | 2.19e-1 | 1.84e-1 | 1.64e-1 | 47 | -3.13e-3 | +8.88e-4 | -1.06e-3 | -4.87e-3 |
| 2 | 3.00e-1 | 5 | 1.48e-1 | 1.88e-1 | 1.62e-1 | 1.48e-1 | 44 | -2.05e-3 | +1.44e-3 | -7.60e-4 | -2.90e-3 |
| 3 | 3.00e-1 | 6 | 1.26e-1 | 1.56e-1 | 1.36e-1 | 1.28e-1 | 41 | -1.65e-3 | +7.09e-4 | -6.83e-4 | -1.76e-3 |
| 4 | 3.00e-1 | 9 | 1.23e-1 | 1.66e-1 | 1.32e-1 | 1.24e-1 | 38 | -4.86e-3 | +2.92e-3 | -5.36e-4 | -9.40e-4 |
| 5 | 3.00e-1 | 4 | 1.35e-1 | 1.67e-1 | 1.44e-1 | 1.38e-1 | 45 | -5.26e-3 | +3.88e-3 | -3.18e-4 | -7.41e-4 |
| 6 | 3.00e-1 | 9 | 1.19e-1 | 1.74e-1 | 1.34e-1 | 1.33e-1 | 43 | -5.05e-3 | +2.49e-3 | -5.74e-4 | -5.15e-4 |
| 7 | 3.00e-1 | 3 | 1.43e-1 | 1.66e-1 | 1.51e-1 | 1.43e-1 | 50 | -3.39e-3 | +2.65e-3 | -2.58e-4 | -4.69e-4 |
| 8 | 3.00e-1 | 6 | 1.31e-1 | 1.79e-1 | 1.44e-1 | 1.33e-1 | 45 | -4.30e-3 | +2.22e-3 | -7.46e-4 | -5.84e-4 |
| 9 | 3.00e-1 | 6 | 1.22e-1 | 1.69e-1 | 1.33e-1 | 1.24e-1 | 37 | -5.91e-3 | +2.82e-3 | -9.53e-4 | -7.16e-4 |
| 10 | 3.00e-1 | 7 | 1.20e-1 | 1.65e-1 | 1.34e-1 | 1.20e-1 | 34 | -4.36e-3 | +3.46e-3 | -7.52e-4 | -7.65e-4 |
| 11 | 3.00e-1 | 7 | 1.18e-1 | 1.60e-1 | 1.30e-1 | 1.32e-1 | 41 | -6.97e-3 | +3.69e-3 | -4.48e-4 | -5.19e-4 |
| 12 | 3.00e-1 | 8 | 1.13e-1 | 1.64e-1 | 1.27e-1 | 1.25e-1 | 39 | -7.97e-3 | +2.82e-3 | -8.39e-4 | -5.58e-4 |
| 13 | 3.00e-1 | 9 | 1.20e-1 | 1.58e-1 | 1.27e-1 | 1.21e-1 | 34 | -6.14e-3 | +3.22e-3 | -5.48e-4 | -4.95e-4 |
| 14 | 3.00e-1 | 6 | 1.22e-1 | 1.62e-1 | 1.33e-1 | 1.24e-1 | 37 | -4.71e-3 | +3.56e-3 | -7.33e-4 | -6.03e-4 |
| 15 | 3.00e-1 | 7 | 1.21e-1 | 1.70e-1 | 1.32e-1 | 1.23e-1 | 35 | -7.50e-3 | +3.66e-3 | -8.70e-4 | -6.96e-4 |
| 16 | 3.00e-1 | 7 | 1.18e-1 | 1.65e-1 | 1.31e-1 | 1.18e-1 | 31 | -6.01e-3 | +3.79e-3 | -9.46e-4 | -8.34e-4 |
| 17 | 3.00e-1 | 12 | 1.10e-1 | 1.63e-1 | 1.20e-1 | 1.19e-1 | 34 | -9.65e-3 | +4.06e-3 | -6.84e-4 | -4.86e-4 |
| 18 | 3.00e-1 | 5 | 1.20e-1 | 1.57e-1 | 1.30e-1 | 1.21e-1 | 33 | -6.66e-3 | +3.98e-3 | -9.38e-4 | -6.77e-4 |
| 19 | 3.00e-1 | 7 | 1.25e-1 | 1.58e-1 | 1.32e-1 | 1.30e-1 | 41 | -5.72e-3 | +3.76e-3 | -3.58e-4 | -4.76e-4 |
| 20 | 3.00e-1 | 6 | 1.30e-1 | 1.70e-1 | 1.41e-1 | 1.32e-1 | 37 | -4.40e-3 | +3.00e-3 | -5.22e-4 | -5.02e-4 |
| 21 | 3.00e-1 | 7 | 1.24e-1 | 1.72e-1 | 1.36e-1 | 1.33e-1 | 41 | -7.48e-3 | +2.90e-3 | -7.98e-4 | -5.68e-4 |
| 22 | 3.00e-1 | 7 | 1.21e-1 | 1.70e-1 | 1.37e-1 | 1.36e-1 | 45 | -6.74e-3 | +2.87e-3 | -6.98e-4 | -5.62e-4 |
| 23 | 3.00e-1 | 4 | 1.39e-1 | 1.71e-1 | 1.49e-1 | 1.39e-1 | 43 | -3.92e-3 | +2.84e-3 | -4.91e-4 | -5.65e-4 |
| 24 | 3.00e-1 | 6 | 1.35e-1 | 1.74e-1 | 1.45e-1 | 1.36e-1 | 46 | -4.16e-3 | +2.61e-3 | -5.35e-4 | -5.47e-4 |
| 25 | 3.00e-1 | 5 | 1.38e-1 | 1.75e-1 | 1.49e-1 | 1.38e-1 | 46 | -3.45e-3 | +2.64e-3 | -5.15e-4 | -5.53e-4 |
| 26 | 3.00e-1 | 6 | 1.33e-1 | 1.78e-1 | 1.45e-1 | 1.36e-1 | 43 | -4.25e-3 | +2.87e-3 | -5.70e-4 | -5.52e-4 |
| 27 | 3.00e-1 | 8 | 1.32e-1 | 1.65e-1 | 1.40e-1 | 1.32e-1 | 43 | -3.30e-3 | +2.53e-3 | -3.21e-4 | -4.20e-4 |
| 28 | 3.00e-1 | 4 | 1.40e-1 | 1.67e-1 | 1.49e-1 | 1.40e-1 | 38 | -2.86e-3 | +2.75e-3 | -2.54e-4 | -3.97e-4 |
| 29 | 3.00e-1 | 7 | 1.21e-1 | 1.76e-1 | 1.34e-1 | 1.27e-1 | 37 | -7.52e-3 | +2.69e-3 | -9.89e-4 | -6.12e-4 |
| 30 | 3.00e-1 | 7 | 1.34e-1 | 1.58e-1 | 1.43e-1 | 1.40e-1 | 48 | -4.14e-3 | +3.00e-3 | -7.45e-5 | -3.42e-4 |
| 31 | 3.00e-1 | 4 | 1.35e-1 | 1.73e-1 | 1.50e-1 | 1.35e-1 | 44 | -3.61e-3 | +2.65e-3 | -7.41e-4 | -5.15e-4 |
| 32 | 3.00e-1 | 6 | 1.35e-1 | 1.72e-1 | 1.46e-1 | 1.35e-1 | 35 | -4.02e-3 | +2.76e-3 | -5.12e-4 | -5.42e-4 |
| 33 | 3.00e-1 | 7 | 1.19e-1 | 1.74e-1 | 1.33e-1 | 1.23e-1 | 34 | -6.32e-3 | +2.91e-3 | -1.02e-3 | -7.28e-4 |
| 34 | 3.00e-1 | 8 | 1.21e-1 | 1.60e-1 | 1.28e-1 | 1.27e-1 | 34 | -7.90e-3 | +3.63e-3 | -5.43e-4 | -5.29e-4 |
| 35 | 3.00e-1 | 5 | 1.42e-1 | 1.72e-1 | 1.51e-1 | 1.47e-1 | 50 | -3.13e-3 | +3.45e-3 | -2.74e-5 | -3.44e-4 |
| 36 | 3.00e-1 | 2 | 1.42e-1 | 2.52e-1 | 1.97e-1 | 2.52e-1 | 343 | -7.19e-4 | +1.68e-3 | +4.83e-4 | -1.75e-4 |
| 38 | 3.00e-1 | 1 | 2.60e-1 | 2.60e-1 | 2.60e-1 | 2.60e-1 | 390 | +7.52e-5 | +7.52e-5 | +7.52e-5 | -1.50e-4 |
| 39 | 3.00e-1 | 1 | 2.54e-1 | 2.54e-1 | 2.54e-1 | 2.54e-1 | 335 | -7.17e-5 | -7.17e-5 | -7.17e-5 | -1.42e-4 |
| 40 | 3.00e-1 | 1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 319 | -4.89e-5 | -4.89e-5 | -4.89e-5 | -1.33e-4 |
| 42 | 3.00e-1 | 2 | 2.46e-1 | 2.68e-1 | 2.57e-1 | 2.46e-1 | 289 | -3.07e-4 | +1.85e-4 | -6.13e-5 | -1.22e-4 |
| 44 | 3.00e-1 | 1 | 2.59e-1 | 2.59e-1 | 2.59e-1 | 2.59e-1 | 345 | +1.53e-4 | +1.53e-4 | +1.53e-4 | -9.42e-5 |
| 45 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 279 | -2.34e-4 | -2.34e-4 | -2.34e-4 | -1.08e-4 |
| 46 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 292 | +1.07e-5 | +1.07e-5 | +1.07e-5 | -9.63e-5 |
| 47 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 273 | -7.62e-5 | -7.62e-5 | -7.62e-5 | -9.43e-5 |
| 48 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 277 | -5.90e-6 | -5.90e-6 | -5.90e-6 | -8.55e-5 |
| 49 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 302 | +7.63e-5 | +7.63e-5 | +7.63e-5 | -6.93e-5 |
| 50 | 3.00e-1 | 1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 336 | +8.16e-5 | +8.16e-5 | +8.16e-5 | -5.42e-5 |
| 51 | 3.00e-1 | 1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 306 | -3.73e-5 | -3.73e-5 | -3.73e-5 | -5.25e-5 |
| 52 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 285 | -9.84e-5 | -9.84e-5 | -9.84e-5 | -5.71e-5 |
| 53 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 281 | +4.10e-5 | +4.10e-5 | +4.10e-5 | -4.73e-5 |
| 55 | 3.00e-1 | 2 | 2.41e-1 | 2.67e-1 | 2.54e-1 | 2.41e-1 | 263 | -3.96e-4 | +2.45e-4 | -7.55e-5 | -5.59e-5 |
| 57 | 3.00e-1 | 1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 307 | +9.64e-5 | +9.64e-5 | +9.64e-5 | -4.06e-5 |
| 58 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 266 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -5.03e-5 |
| 59 | 3.00e-1 | 1 | 2.52e-1 | 2.52e-1 | 2.52e-1 | 2.52e-1 | 301 | +1.71e-4 | +1.71e-4 | +1.71e-4 | -2.82e-5 |
| 60 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 263 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -3.95e-5 |
| 61 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 273 | -4.34e-5 | -4.34e-5 | -4.34e-5 | -3.99e-5 |
| 62 | 3.00e-1 | 2 | 2.29e-1 | 2.36e-1 | 2.32e-1 | 2.29e-1 | 228 | -1.21e-4 | -6.96e-5 | -9.54e-5 | -5.07e-5 |
| 64 | 3.00e-1 | 2 | 2.36e-1 | 2.54e-1 | 2.45e-1 | 2.36e-1 | 244 | -2.99e-4 | +3.27e-4 | +1.41e-5 | -4.15e-5 |
| 66 | 3.00e-1 | 2 | 2.37e-1 | 2.58e-1 | 2.47e-1 | 2.37e-1 | 241 | -3.41e-4 | +2.66e-4 | -3.73e-5 | -4.37e-5 |
| 67 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 269 | +5.26e-5 | +5.26e-5 | +5.26e-5 | -3.41e-5 |
| 68 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 262 | -5.77e-5 | -5.77e-5 | -5.77e-5 | -3.65e-5 |
| 69 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 249 | -6.65e-5 | -6.65e-5 | -6.65e-5 | -3.95e-5 |
| 70 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 257 | +5.40e-5 | +5.40e-5 | +5.40e-5 | -3.01e-5 |
| 71 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 294 | +1.04e-4 | +1.04e-4 | +1.04e-4 | -1.67e-5 |
| 72 | 3.00e-1 | 1 | 2.51e-1 | 2.51e-1 | 2.51e-1 | 2.51e-1 | 310 | +9.48e-5 | +9.48e-5 | +9.48e-5 | -5.53e-6 |
| 73 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 263 | -1.57e-4 | -1.57e-4 | -1.57e-4 | -2.06e-5 |
| 74 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 253 | -6.67e-5 | -6.67e-5 | -6.67e-5 | -2.53e-5 |
| 75 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 272 | +1.08e-4 | +1.08e-4 | +1.08e-4 | -1.19e-5 |
| 76 | 3.00e-1 | 1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 2.43e-1 | 271 | -2.11e-5 | -2.11e-5 | -2.11e-5 | -1.29e-5 |
| 77 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 246 | -9.73e-5 | -9.73e-5 | -9.73e-5 | -2.13e-5 |
| 78 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 258 | +5.41e-5 | +5.41e-5 | +5.41e-5 | -1.38e-5 |
| 79 | 3.00e-1 | 1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 2.49e-1 | 270 | +1.29e-4 | +1.29e-4 | +1.29e-4 | +5.56e-7 |
| 80 | 3.00e-1 | 2 | 2.32e-1 | 2.36e-1 | 2.34e-1 | 2.32e-1 | 219 | -2.26e-4 | -7.88e-5 | -1.52e-4 | -2.78e-5 |
| 82 | 3.00e-1 | 2 | 2.33e-1 | 2.55e-1 | 2.44e-1 | 2.33e-1 | 218 | -4.14e-4 | +3.15e-4 | -4.96e-5 | -3.55e-5 |
| 83 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 244 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -2.01e-5 |
| 84 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 251 | -5.77e-6 | -5.77e-6 | -5.77e-6 | -1.87e-5 |
| 85 | 3.00e-1 | 1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 276 | +1.50e-4 | +1.50e-4 | +1.50e-4 | -1.84e-6 |
| 86 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 240 | -2.08e-4 | -2.08e-4 | -2.08e-4 | -2.25e-5 |
| 87 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 232 | -9.80e-5 | -9.80e-5 | -9.80e-5 | -3.00e-5 |
| 88 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 247 | +9.82e-5 | +9.82e-5 | +9.82e-5 | -1.72e-5 |
| 89 | 3.00e-1 | 2 | 2.31e-1 | 2.32e-1 | 2.31e-1 | 2.31e-1 | 204 | -1.14e-4 | -1.75e-5 | -6.56e-5 | -2.59e-5 |
| 90 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 235 | +8.44e-5 | +8.44e-5 | +8.44e-5 | -1.49e-5 |
| 91 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 233 | +2.62e-5 | +2.62e-5 | +2.62e-5 | -1.08e-5 |
| 92 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 231 | -2.77e-5 | -2.77e-5 | -2.77e-5 | -1.25e-5 |
| 93 | 3.00e-1 | 2 | 2.27e-1 | 2.46e-1 | 2.36e-1 | 2.27e-1 | 195 | -4.06e-4 | +1.61e-4 | -1.23e-4 | -3.62e-5 |
| 94 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 229 | +1.02e-4 | +1.02e-4 | +1.02e-4 | -2.24e-5 |
| 95 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 213 | -7.65e-5 | -7.65e-5 | -7.65e-5 | -2.78e-5 |
| 96 | 3.00e-1 | 2 | 2.23e-1 | 2.27e-1 | 2.25e-1 | 2.23e-1 | 190 | -9.66e-5 | -3.27e-5 | -6.46e-5 | -3.51e-5 |
| 97 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 213 | +6.11e-5 | +6.11e-5 | +6.11e-5 | -2.55e-5 |
| 98 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 230 | +1.15e-4 | +1.15e-4 | +1.15e-4 | -1.15e-5 |
| 99 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 242 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +3.02e-7 |
| 100 | 3.00e-2 | 2 | 1.64e-1 | 2.37e-1 | 2.00e-1 | 1.64e-1 | 204 | -1.81e-3 | -1.51e-5 | -9.15e-4 | -1.83e-4 |
| 101 | 3.00e-2 | 1 | 1.15e-1 | 1.15e-1 | 1.15e-1 | 1.15e-1 | 236 | -1.49e-3 | -1.49e-3 | -1.49e-3 | -3.13e-4 |
| 102 | 3.00e-2 | 1 | 8.11e-2 | 8.11e-2 | 8.11e-2 | 8.11e-2 | 227 | -1.55e-3 | -1.55e-3 | -1.55e-3 | -4.37e-4 |
| 103 | 3.00e-2 | 2 | 4.61e-2 | 5.91e-2 | 5.26e-2 | 4.61e-2 | 180 | -1.46e-3 | -1.38e-3 | -1.42e-3 | -6.23e-4 |
| 104 | 3.00e-2 | 1 | 3.83e-2 | 3.83e-2 | 3.83e-2 | 3.83e-2 | 204 | -9.08e-4 | -9.08e-4 | -9.08e-4 | -6.52e-4 |
| 105 | 3.00e-2 | 2 | 3.16e-2 | 3.47e-2 | 3.31e-2 | 3.16e-2 | 181 | -5.24e-4 | -4.24e-4 | -4.74e-4 | -6.18e-4 |
| 107 | 3.00e-2 | 2 | 3.12e-2 | 3.37e-2 | 3.24e-2 | 3.12e-2 | 186 | -4.10e-4 | +2.66e-4 | -7.22e-5 | -5.18e-4 |
| 108 | 3.00e-2 | 1 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 3.20e-2 | 208 | +1.14e-4 | +1.14e-4 | +1.14e-4 | -4.55e-4 |
| 109 | 3.00e-2 | 2 | 3.11e-2 | 3.22e-2 | 3.16e-2 | 3.11e-2 | 160 | -2.07e-4 | +3.16e-5 | -8.79e-5 | -3.86e-4 |
| 110 | 3.00e-2 | 1 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 3.27e-2 | 191 | +2.60e-4 | +2.60e-4 | +2.60e-4 | -3.22e-4 |
| 111 | 3.00e-2 | 2 | 3.27e-2 | 3.42e-2 | 3.35e-2 | 3.27e-2 | 164 | -2.80e-4 | +2.24e-4 | -2.78e-5 | -2.68e-4 |
| 112 | 3.00e-2 | 1 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 207 | +2.46e-4 | +2.46e-4 | +2.46e-4 | -2.17e-4 |
| 113 | 3.00e-2 | 2 | 3.44e-2 | 3.45e-2 | 3.45e-2 | 3.44e-2 | 160 | -3.08e-5 | +2.09e-5 | -4.91e-6 | -1.77e-4 |
| 114 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 226 | +3.86e-4 | +3.86e-4 | +3.86e-4 | -1.21e-4 |
| 115 | 3.00e-2 | 2 | 3.47e-2 | 3.56e-2 | 3.51e-2 | 3.47e-2 | 147 | -2.84e-4 | -1.83e-4 | -2.33e-4 | -1.42e-4 |
| 116 | 3.00e-2 | 2 | 3.59e-2 | 3.61e-2 | 3.60e-2 | 3.59e-2 | 168 | -3.35e-5 | +2.17e-4 | +9.18e-5 | -9.85e-5 |
| 117 | 3.00e-2 | 1 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 167 | +4.72e-5 | +4.72e-5 | +4.72e-5 | -8.39e-5 |
| 118 | 3.00e-2 | 2 | 3.64e-2 | 3.82e-2 | 3.73e-2 | 3.64e-2 | 152 | -3.26e-4 | +2.98e-4 | -1.40e-5 | -7.37e-5 |
| 119 | 3.00e-2 | 2 | 3.71e-2 | 3.88e-2 | 3.79e-2 | 3.71e-2 | 136 | -3.36e-4 | +3.76e-4 | +1.98e-5 | -5.95e-5 |
| 120 | 3.00e-2 | 1 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 171 | +2.82e-4 | +2.82e-4 | +2.82e-4 | -2.54e-5 |
| 121 | 3.00e-2 | 3 | 3.66e-2 | 3.87e-2 | 3.74e-2 | 3.70e-2 | 135 | -3.90e-4 | +8.09e-5 | -1.14e-4 | -4.81e-5 |
| 122 | 3.00e-2 | 1 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 4.06e-2 | 185 | +5.03e-4 | +5.03e-4 | +5.03e-4 | +7.02e-6 |
| 123 | 3.00e-2 | 2 | 3.89e-2 | 4.20e-2 | 4.04e-2 | 3.89e-2 | 124 | -6.19e-4 | +1.77e-4 | -2.21e-4 | -4.03e-5 |
| 124 | 3.00e-2 | 2 | 3.86e-2 | 4.13e-2 | 4.00e-2 | 3.86e-2 | 121 | -5.73e-4 | +3.60e-4 | -1.06e-4 | -5.75e-5 |
| 125 | 3.00e-2 | 2 | 3.93e-2 | 4.18e-2 | 4.05e-2 | 3.93e-2 | 122 | -5.06e-4 | +4.91e-4 | -7.29e-6 | -5.29e-5 |
| 126 | 3.00e-2 | 2 | 4.14e-2 | 4.42e-2 | 4.28e-2 | 4.14e-2 | 122 | -5.38e-4 | +6.66e-4 | +6.44e-5 | -3.67e-5 |
| 127 | 3.00e-2 | 2 | 4.10e-2 | 4.41e-2 | 4.25e-2 | 4.10e-2 | 112 | -6.49e-4 | +3.53e-4 | -1.48e-4 | -6.28e-5 |
| 128 | 3.00e-2 | 2 | 4.03e-2 | 4.19e-2 | 4.11e-2 | 4.03e-2 | 126 | -3.15e-4 | +1.61e-4 | -7.69e-5 | -6.78e-5 |
| 129 | 3.00e-2 | 2 | 4.09e-2 | 4.32e-2 | 4.20e-2 | 4.09e-2 | 112 | -4.76e-4 | +4.55e-4 | -1.05e-5 | -6.16e-5 |
| 130 | 3.00e-2 | 2 | 4.10e-2 | 4.34e-2 | 4.22e-2 | 4.10e-2 | 112 | -4.94e-4 | +3.93e-4 | -5.05e-5 | -6.39e-5 |
| 131 | 3.00e-2 | 2 | 4.18e-2 | 4.56e-2 | 4.37e-2 | 4.18e-2 | 113 | -7.68e-4 | +6.50e-4 | -5.85e-5 | -7.00e-5 |
| 132 | 3.00e-2 | 2 | 4.20e-2 | 4.55e-2 | 4.37e-2 | 4.20e-2 | 112 | -7.13e-4 | +5.63e-4 | -7.53e-5 | -7.74e-5 |
| 133 | 3.00e-2 | 4 | 4.04e-2 | 4.74e-2 | 4.27e-2 | 4.04e-2 | 91 | -1.37e-3 | +7.68e-4 | -2.15e-4 | -1.32e-4 |
| 134 | 3.00e-2 | 1 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 4.53e-2 | 133 | +8.58e-4 | +8.58e-4 | +8.58e-4 | -3.33e-5 |
| 135 | 3.00e-2 | 3 | 4.08e-2 | 4.66e-2 | 4.31e-2 | 4.08e-2 | 89 | -1.12e-3 | +2.12e-4 | -4.11e-4 | -1.41e-4 |
| 136 | 3.00e-2 | 3 | 4.15e-2 | 4.47e-2 | 4.26e-2 | 4.15e-2 | 87 | -8.01e-4 | +6.97e-4 | -3.03e-5 | -1.17e-4 |
| 137 | 3.00e-2 | 2 | 4.40e-2 | 4.62e-2 | 4.51e-2 | 4.40e-2 | 91 | -5.25e-4 | +7.48e-4 | +1.12e-4 | -7.97e-5 |
| 138 | 3.00e-2 | 3 | 4.37e-2 | 4.62e-2 | 4.47e-2 | 4.41e-2 | 103 | -5.42e-4 | +3.66e-4 | -2.72e-5 | -6.78e-5 |
| 139 | 3.00e-2 | 3 | 4.30e-2 | 4.78e-2 | 4.51e-2 | 4.30e-2 | 90 | -8.05e-4 | +6.28e-4 | -1.83e-4 | -1.08e-4 |
| 140 | 3.00e-2 | 3 | 4.17e-2 | 4.64e-2 | 4.38e-2 | 4.17e-2 | 84 | -7.89e-4 | +6.52e-4 | -2.05e-4 | -1.45e-4 |
| 141 | 3.00e-2 | 2 | 4.34e-2 | 5.04e-2 | 4.69e-2 | 4.34e-2 | 77 | -1.92e-3 | +1.32e-3 | -3.02e-4 | -1.91e-4 |
| 142 | 3.00e-2 | 5 | 3.91e-2 | 4.82e-2 | 4.26e-2 | 3.91e-2 | 71 | -1.39e-3 | +9.30e-4 | -3.56e-4 | -2.73e-4 |
| 143 | 3.00e-2 | 2 | 4.46e-2 | 4.48e-2 | 4.47e-2 | 4.46e-2 | 72 | -5.47e-5 | +1.40e-3 | +6.71e-4 | -1.01e-4 |
| 144 | 3.00e-2 | 5 | 4.10e-2 | 4.73e-2 | 4.33e-2 | 4.10e-2 | 66 | -1.01e-3 | +5.06e-4 | -2.84e-4 | -1.83e-4 |
| 145 | 3.00e-2 | 3 | 3.96e-2 | 4.62e-2 | 4.24e-2 | 3.96e-2 | 60 | -1.78e-3 | +1.22e-3 | -4.44e-4 | -2.72e-4 |
| 146 | 3.00e-2 | 4 | 3.77e-2 | 4.69e-2 | 4.11e-2 | 3.77e-2 | 51 | -2.43e-3 | +1.59e-3 | -5.56e-4 | -3.93e-4 |
| 147 | 3.00e-2 | 5 | 3.54e-2 | 4.53e-2 | 3.89e-2 | 3.54e-2 | 48 | -2.99e-3 | +1.97e-3 | -6.28e-4 | -5.12e-4 |
| 148 | 3.00e-2 | 5 | 3.72e-2 | 4.75e-2 | 4.04e-2 | 3.72e-2 | 47 | -2.87e-3 | +2.83e-3 | -3.69e-4 | -4.84e-4 |
| 149 | 3.00e-2 | 6 | 3.42e-2 | 4.76e-2 | 3.79e-2 | 3.59e-2 | 39 | -4.67e-3 | +2.48e-3 | -7.01e-4 | -5.45e-4 |
| 150 | 3.00e-3 | 6 | 5.84e-3 | 3.71e-2 | 1.63e-2 | 5.84e-3 | 35 | -1.27e-2 | +3.65e-4 | -7.98e-3 | -4.13e-3 |
| 151 | 3.00e-3 | 7 | 3.25e-3 | 5.76e-3 | 3.82e-3 | 3.25e-3 | 39 | -8.86e-3 | -1.51e-4 | -2.30e-3 | -3.00e-3 |
| 152 | 3.00e-3 | 6 | 3.00e-3 | 4.33e-3 | 3.41e-3 | 3.16e-3 | 39 | -5.41e-3 | +3.31e-3 | -9.22e-4 | -2.00e-3 |
| 153 | 3.00e-3 | 7 | 2.93e-3 | 4.21e-3 | 3.29e-3 | 3.16e-3 | 39 | -5.57e-3 | +3.67e-3 | -6.72e-4 | -1.24e-3 |
| 154 | 3.00e-3 | 6 | 3.33e-3 | 4.25e-3 | 3.56e-3 | 3.33e-3 | 43 | -5.67e-3 | +3.41e-3 | -5.67e-4 | -9.36e-4 |
| 155 | 3.00e-3 | 6 | 3.43e-3 | 4.20e-3 | 3.62e-3 | 3.43e-3 | 43 | -3.44e-3 | +2.68e-3 | -3.22e-4 | -6.63e-4 |
| 156 | 3.00e-3 | 8 | 3.03e-3 | 4.27e-3 | 3.41e-3 | 3.11e-3 | 37 | -3.76e-3 | +2.43e-3 | -7.23e-4 | -6.65e-4 |
| 157 | 3.00e-3 | 5 | 3.18e-3 | 3.94e-3 | 3.46e-3 | 3.18e-3 | 35 | -2.87e-3 | +3.06e-3 | -4.92e-4 | -6.43e-4 |
| 158 | 3.00e-3 | 6 | 3.21e-3 | 3.97e-3 | 3.41e-3 | 3.37e-3 | 41 | -4.11e-3 | +3.20e-3 | -2.26e-4 | -4.30e-4 |
| 159 | 3.00e-3 | 7 | 3.16e-3 | 4.25e-3 | 3.47e-3 | 3.25e-3 | 38 | -3.60e-3 | +2.64e-3 | -5.93e-4 | -4.95e-4 |
| 160 | 3.00e-3 | 7 | 2.99e-3 | 4.20e-3 | 3.35e-3 | 3.07e-3 | 32 | -4.95e-3 | +3.34e-3 | -8.21e-4 | -6.46e-4 |
| 161 | 3.00e-3 | 6 | 3.19e-3 | 4.15e-3 | 3.44e-3 | 3.36e-3 | 39 | -6.38e-3 | +3.81e-3 | -5.33e-4 | -5.58e-4 |
| 162 | 3.00e-3 | 8 | 3.31e-3 | 4.30e-3 | 3.57e-3 | 3.31e-3 | 41 | -3.11e-3 | +3.06e-3 | -4.15e-4 | -4.88e-4 |
| 163 | 3.00e-3 | 4 | 3.53e-3 | 4.27e-3 | 3.81e-3 | 3.53e-3 | 41 | -3.09e-3 | +3.26e-3 | -2.79e-4 | -4.60e-4 |
| 164 | 3.00e-3 | 7 | 3.45e-3 | 4.47e-3 | 3.70e-3 | 3.45e-3 | 39 | -4.61e-3 | +2.83e-3 | -6.14e-4 | -5.36e-4 |
| 165 | 3.00e-3 | 7 | 3.58e-3 | 4.26e-3 | 3.78e-3 | 3.66e-3 | 45 | -1.68e-3 | +2.44e-3 | -1.31e-4 | -3.32e-4 |
| 166 | 3.00e-3 | 4 | 3.62e-3 | 4.50e-3 | 3.92e-3 | 3.67e-3 | 45 | -3.20e-3 | +2.64e-3 | -4.62e-4 | -3.97e-4 |
| 167 | 3.00e-3 | 6 | 3.50e-3 | 4.40e-3 | 3.74e-3 | 3.50e-3 | 39 | -3.38e-3 | +2.25e-3 | -5.59e-4 | -4.78e-4 |
| 168 | 3.00e-3 | 9 | 3.22e-3 | 4.38e-3 | 3.57e-3 | 3.27e-3 | 37 | -3.82e-3 | +2.61e-3 | -5.42e-4 | -5.05e-4 |
| 169 | 3.00e-3 | 5 | 3.47e-3 | 4.15e-3 | 3.64e-3 | 3.52e-3 | 37 | -3.82e-3 | +3.25e-3 | -1.94e-4 | -3.91e-4 |
| 170 | 3.00e-3 | 7 | 3.18e-3 | 4.32e-3 | 3.49e-3 | 3.33e-3 | 41 | -7.45e-3 | +3.02e-3 | -9.00e-4 | -5.90e-4 |
| 171 | 3.00e-3 | 6 | 3.56e-3 | 4.39e-3 | 3.74e-3 | 3.60e-3 | 41 | -3.77e-3 | +3.56e-3 | -2.09e-4 | -4.20e-4 |
| 172 | 3.00e-3 | 7 | 3.17e-3 | 4.63e-3 | 3.55e-3 | 3.17e-3 | 34 | -5.79e-3 | +3.10e-3 | -1.06e-3 | -7.29e-4 |
| 173 | 3.00e-3 | 8 | 3.27e-3 | 4.39e-3 | 3.52e-3 | 3.60e-3 | 47 | -6.49e-3 | +4.39e-3 | -2.57e-4 | -3.80e-4 |
| 174 | 3.00e-3 | 5 | 3.19e-3 | 4.87e-3 | 3.81e-3 | 3.19e-3 | 32 | -4.60e-3 | +3.06e-3 | -1.58e-3 | -9.27e-4 |
| 175 | 3.00e-3 | 6 | 3.46e-3 | 4.50e-3 | 3.77e-3 | 3.63e-3 | 44 | -4.54e-3 | +4.29e-3 | -2.72e-4 | -6.26e-4 |
| 176 | 3.00e-3 | 7 | 3.68e-3 | 4.67e-3 | 3.97e-3 | 3.68e-3 | 47 | -2.44e-3 | +3.06e-3 | -2.89e-4 | -4.80e-4 |
| 177 | 3.00e-3 | 5 | 3.50e-3 | 4.80e-3 | 4.05e-3 | 3.50e-3 | 34 | -2.53e-3 | +2.84e-3 | -9.67e-4 | -7.49e-4 |
| 178 | 3.00e-3 | 6 | 3.60e-3 | 4.26e-3 | 3.80e-3 | 3.83e-3 | 44 | -3.16e-3 | +2.71e-3 | -3.61e-5 | -3.99e-4 |
| 179 | 3.00e-3 | 6 | 3.57e-3 | 4.60e-3 | 3.85e-3 | 3.67e-3 | 40 | -4.13e-3 | +2.25e-3 | -5.72e-4 | -4.57e-4 |
| 180 | 3.00e-3 | 6 | 3.41e-3 | 4.57e-3 | 3.82e-3 | 3.41e-3 | 37 | -2.86e-3 | +2.75e-3 | -7.92e-4 | -6.48e-4 |
| 181 | 3.00e-3 | 6 | 3.46e-3 | 4.48e-3 | 3.97e-3 | 4.11e-3 | 55 | -4.95e-3 | +3.19e-3 | +6.52e-5 | -3.23e-4 |
| 182 | 3.00e-3 | 5 | 4.15e-3 | 5.02e-3 | 4.40e-3 | 4.26e-3 | 52 | -3.13e-3 | +2.09e-3 | -2.17e-4 | -2.82e-4 |
| 183 | 3.00e-3 | 5 | 4.00e-3 | 4.73e-3 | 4.23e-3 | 4.15e-3 | 49 | -2.55e-3 | +1.29e-3 | -2.70e-4 | -2.64e-4 |
| 184 | 3.00e-3 | 6 | 3.38e-3 | 5.49e-3 | 4.07e-3 | 3.38e-3 | 32 | -4.56e-3 | +2.58e-3 | -1.57e-3 | -9.03e-4 |
| 185 | 3.00e-3 | 7 | 3.64e-3 | 4.50e-3 | 3.86e-3 | 3.80e-3 | 41 | -3.88e-3 | +3.76e-3 | -9.54e-5 | -4.76e-4 |
| 186 | 3.00e-3 | 4 | 3.74e-3 | 4.94e-3 | 4.22e-3 | 4.10e-3 | 62 | -5.16e-3 | +3.03e-3 | -7.56e-4 | -5.75e-4 |
| 187 | 3.00e-3 | 4 | 4.62e-3 | 5.27e-3 | 4.85e-3 | 4.62e-3 | 53 | -1.48e-3 | +2.29e-3 | +6.11e-5 | -3.86e-4 |
| 188 | 3.00e-3 | 5 | 4.13e-3 | 5.61e-3 | 4.55e-3 | 4.19e-3 | 52 | -4.36e-3 | +1.94e-3 | -8.58e-4 | -5.72e-4 |
| 189 | 3.00e-3 | 6 | 3.78e-3 | 5.24e-3 | 4.26e-3 | 3.78e-3 | 44 | -3.44e-3 | +2.49e-3 | -8.00e-4 | -6.97e-4 |
| 190 | 3.00e-3 | 5 | 3.95e-3 | 4.97e-3 | 4.26e-3 | 3.95e-3 | 49 | -3.82e-3 | +3.17e-3 | -3.80e-4 | -5.97e-4 |
| 191 | 3.00e-3 | 5 | 4.21e-3 | 5.12e-3 | 4.47e-3 | 4.24e-3 | 54 | -2.38e-3 | +2.62e-3 | -2.40e-4 | -4.71e-4 |
| 192 | 3.00e-3 | 6 | 3.92e-3 | 5.29e-3 | 4.29e-3 | 4.09e-3 | 48 | -4.48e-3 | +2.41e-3 | -6.12e-4 | -5.06e-4 |
| 193 | 3.00e-3 | 5 | 3.99e-3 | 5.06e-3 | 4.29e-3 | 4.00e-3 | 42 | -3.71e-3 | +2.43e-3 | -5.24e-4 | -5.24e-4 |
| 194 | 3.00e-3 | 6 | 3.68e-3 | 4.93e-3 | 4.05e-3 | 3.75e-3 | 44 | -3.53e-3 | +2.47e-3 | -7.52e-4 | -6.19e-4 |
| 195 | 3.00e-3 | 6 | 3.74e-3 | 5.01e-3 | 4.11e-3 | 3.84e-3 | 41 | -3.88e-3 | +3.09e-3 | -5.34e-4 | -5.77e-4 |
| 196 | 3.00e-3 | 8 | 3.72e-3 | 4.96e-3 | 4.06e-3 | 3.72e-3 | 36 | -3.90e-3 | +3.17e-3 | -5.16e-4 | -5.62e-4 |
| 197 | 3.00e-3 | 5 | 3.68e-3 | 4.63e-3 | 3.96e-3 | 3.78e-3 | 42 | -4.61e-3 | +2.83e-3 | -5.92e-4 | -5.71e-4 |
| 198 | 3.00e-3 | 9 | 3.39e-3 | 4.66e-3 | 3.75e-3 | 3.39e-3 | 30 | -4.02e-3 | +2.92e-3 | -7.51e-4 | -6.82e-4 |
| 199 | 3.00e-3 | 5 | 3.77e-3 | 4.52e-3 | 4.05e-3 | 3.77e-3 | 43 | -2.60e-3 | +3.79e-3 | -1.48e-4 | -5.18e-4 |

