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
| cpu-async | 0.052240 | 0.9204 | +0.0079 | 1821.8 | 919 | 77.0 | 100% | 100% | 100% | 13.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9204 | cpu-async | - | - |

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
| cpu-async | 2.0042 | 0.7378 | 0.5693 | 0.5023 | 0.4863 | 0.4623 | 0.4345 | 0.4312 | 0.4219 | 0.4125 | 0.1817 | 0.1471 | 0.1265 | 0.1382 | 0.1294 | 0.0698 | 0.0626 | 0.0591 | 0.0526 | 0.0522 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4006 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3037 | 3.6 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2956 | 3.7 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 398 | 395 | 390 | 385 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu2 | 1246.2 | 2.6 | epoch-boundary(136) |
| cpu-async | gpu1 | 1246.2 | 2.5 | epoch-boundary(136) |
| cpu-async | gpu1 | 1676.4 | 1.3 | epoch-boundary(183) |
| cpu-async | gpu0 | 1080.9 | 0.8 | epoch-boundary(118) |
| cpu-async | gpu2 | 1080.9 | 0.8 | epoch-boundary(118) |
| cpu-async | gpu0 | 131.1 | 0.7 | cpu-avg |
| cpu-async | gpu1 | 1350.1 | 0.7 | epoch-boundary(147) |
| cpu-async | gpu0 | 0.3 | 0.6 | cpu-avg |
| cpu-async | gpu1 | 1080.9 | 0.6 | epoch-boundary(118) |
| cpu-async | gpu1 | 131.1 | 0.6 | cpu-avg |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.8s | 0.0s | 1.4s | 0.0s | 2.1s |
| resnet-graph | cpu-async | gpu1 | 5.5s | 0.0s | 0.6s | 0.0s | 6.1s |
| resnet-graph | cpu-async | gpu2 | 3.9s | 0.0s | 0.6s | 0.0s | 5.4s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 356 | 0 | 919 | 77.0 | 1285/8066 | 919 | 77.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 185.4 | 10.2% |

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
| resnet-graph | cpu-async | 193 | 919 | 0 | 6.04e-3 | -4.42e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 919 | 8.10e-2 | 4.74e-2 | 5.52e-3 | 4.31e-1 | 32.0 | -2.43e-4 | 1.55e-3 |
| resnet-graph | cpu-async | 1 | 919 | 8.15e-2 | 4.85e-2 | 5.51e-3 | 4.40e-1 | 32.0 | -2.73e-4 | 1.78e-3 |
| resnet-graph | cpu-async | 2 | 919 | 8.19e-2 | 4.87e-2 | 5.70e-3 | 4.33e-1 | 36.0 | -2.84e-4 | 1.75e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9917 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9925 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9903 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 112 (0,1,2,3,4,5,6,8…143,146) | 0 (—) | — | 0,1,2,3,4,5,6,8…143,146 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 625 | +0.148 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 237 | +0.154 |
| resnet-graph | cpu-async | 3.00e-3 | 151–199 | 53 | -0.526 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 917 | +0.008 | 192 | +0.366 | +0.481 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 918 | 3.37e1–7.85e1 | 7.03e1 | 1.67e-3 | 2.81e-3 | 3.45e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 627 | 33–78010 | +6.381e-7 | 0.007 | +7.474e-7 | 0.011 | 100 | +7.478e-7 | 0.062 | 33–242 | +3.163e-3 | 0.580 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 613 | 928–78010 | +6.576e-7 | 0.009 | +7.112e-7 | 0.014 | 99 | +7.218e-7 | 0.056 | 72–242 | +3.483e-3 | 0.932 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 238 | 78239–117139 | +3.644e-5 | 0.587 | +3.657e-5 | 0.600 | 49 | +4.883e-5 | 0.861 | 74–1034 | +1.956e-3 | 0.569 |
| resnet-graph | cpu-async | 3.00e-3 | 151–199 | 54 | 118027–156095 | -4.133e-6 | 0.064 | -4.145e-6 | 0.068 | 44 | -2.185e-6 | 0.039 | 547–945 | +6.877e-4 | 0.116 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +3.163e-3 | r0: +2.924e-3, r1: +3.218e-3, r2: +3.326e-3 | r0: 0.543, r1: 0.562, r2: 0.571 | 1.14× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +3.483e-3 | r0: +3.194e-3, r1: +3.526e-3, r2: +3.693e-3 | r0: 0.867, r1: 0.897, r2: 0.899 | 1.16× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +1.956e-3 | r0: +1.934e-3, r1: +1.969e-3, r2: +1.965e-3 | r0: 0.563, r1: 0.576, r2: 0.557 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 151–199 | +6.877e-4 | r0: +6.713e-4, r1: +6.581e-4, r2: +7.336e-4 | r0: 0.112, r1: 0.108, r2: 0.123 | 1.11× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█████████████████████████▄▃▃▃▄▄▄▄▄▆▆▆▃▁▁▁▁▁▁▁▁▁▁▁` | `▁▆▆▆▆▆▆▆▆▇▆▆▆▆▆▆▅▆▆▆▆▆▆▆▇▁▆▇▆▆▇▆▇▆███▇▇▇▇████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 14 | 7.04e-2 | 4.40e-1 | 1.41e-1 | 8.25e-2 | 28 | -4.49e-2 | +4.19e-3 | -8.97e-3 | -5.23e-3 |
| 1 | 3.00e-1 | 8 | 8.01e-2 | 1.36e-1 | 8.97e-2 | 8.49e-2 | 32 | -1.47e-2 | +6.65e-3 | -1.07e-3 | -2.39e-3 |
| 2 | 3.00e-1 | 7 | 9.42e-2 | 1.33e-1 | 1.06e-1 | 1.02e-1 | 32 | -9.84e-3 | +6.06e-3 | -3.03e-4 | -1.26e-3 |
| 3 | 3.00e-1 | 7 | 9.24e-2 | 1.49e-1 | 1.04e-1 | 1.04e-1 | 36 | -1.82e-2 | +4.65e-3 | -1.46e-3 | -1.12e-3 |
| 4 | 3.00e-1 | 6 | 1.05e-1 | 1.49e-1 | 1.18e-1 | 1.13e-1 | 43 | -5.87e-3 | +4.46e-3 | -3.11e-4 | -7.71e-4 |
| 5 | 3.00e-1 | 6 | 1.06e-1 | 1.61e-1 | 1.20e-1 | 1.14e-1 | 39 | -5.67e-3 | +4.04e-3 | -7.01e-4 | -7.18e-4 |
| 6 | 3.00e-1 | 8 | 9.86e-2 | 1.52e-1 | 1.10e-1 | 1.05e-1 | 35 | -1.10e-2 | +3.11e-3 | -1.04e-3 | -7.45e-4 |
| 7 | 3.00e-1 | 8 | 1.01e-1 | 1.49e-1 | 1.10e-1 | 1.02e-1 | 35 | -8.63e-3 | +4.26e-3 | -7.95e-4 | -7.28e-4 |
| 8 | 3.00e-1 | 5 | 1.06e-1 | 1.43e-1 | 1.16e-1 | 1.15e-1 | 43 | -7.94e-3 | +4.72e-3 | -2.68e-4 | -5.19e-4 |
| 9 | 3.00e-1 | 9 | 9.68e-2 | 1.58e-1 | 1.10e-1 | 1.12e-1 | 41 | -1.33e-2 | +3.51e-3 | -8.63e-4 | -4.99e-4 |
| 10 | 3.00e-1 | 5 | 9.87e-2 | 1.50e-1 | 1.18e-1 | 9.87e-2 | 34 | -5.42e-3 | +3.65e-3 | -1.32e-3 | -9.01e-4 |
| 11 | 3.00e-1 | 7 | 9.45e-2 | 1.41e-1 | 1.07e-1 | 9.65e-2 | 31 | -1.18e-2 | +4.98e-3 | -1.17e-3 | -1.04e-3 |
| 12 | 3.00e-1 | 5 | 9.48e-2 | 1.50e-1 | 1.20e-1 | 1.20e-1 | 47 | -1.58e-2 | +5.23e-3 | -1.12e-3 | -1.01e-3 |
| 13 | 3.00e-1 | 6 | 1.13e-1 | 1.52e-1 | 1.24e-1 | 1.21e-1 | 50 | -5.69e-3 | +2.61e-3 | -4.42e-4 | -7.14e-4 |
| 14 | 3.00e-1 | 5 | 1.11e-1 | 1.46e-1 | 1.20e-1 | 1.16e-1 | 49 | -4.94e-3 | +2.21e-3 | -5.74e-4 | -6.32e-4 |
| 15 | 3.00e-1 | 5 | 1.16e-1 | 1.57e-1 | 1.26e-1 | 1.16e-1 | 49 | -5.44e-3 | +3.33e-3 | -4.82e-4 | -5.82e-4 |
| 16 | 3.00e-1 | 5 | 1.14e-1 | 1.49e-1 | 1.22e-1 | 1.15e-1 | 46 | -5.57e-3 | +3.03e-3 | -5.39e-4 | -5.67e-4 |
| 17 | 3.00e-1 | 6 | 1.06e-1 | 1.48e-1 | 1.17e-1 | 1.12e-1 | 42 | -6.81e-3 | +2.82e-3 | -6.15e-4 | -5.44e-4 |
| 18 | 3.00e-1 | 6 | 9.94e-2 | 1.61e-1 | 1.14e-1 | 1.08e-1 | 42 | -1.07e-2 | +3.62e-3 | -1.15e-3 | -7.46e-4 |
| 19 | 3.00e-1 | 5 | 1.11e-1 | 1.49e-1 | 1.21e-1 | 1.15e-1 | 50 | -6.71e-3 | +3.68e-3 | -4.59e-4 | -6.35e-4 |
| 20 | 3.00e-1 | 6 | 1.15e-1 | 1.52e-1 | 1.25e-1 | 1.18e-1 | 50 | -3.62e-3 | +3.13e-3 | -2.77e-4 | -4.74e-4 |
| 21 | 3.00e-1 | 4 | 1.10e-1 | 1.53e-1 | 1.23e-1 | 1.16e-1 | 46 | -7.32e-3 | +2.87e-3 | -8.37e-4 | -5.91e-4 |
| 22 | 3.00e-1 | 6 | 1.02e-1 | 1.49e-1 | 1.13e-1 | 1.02e-1 | 39 | -9.06e-3 | +2.94e-3 | -1.16e-3 | -8.38e-4 |
| 23 | 3.00e-1 | 6 | 1.03e-1 | 1.42e-1 | 1.12e-1 | 1.08e-1 | 41 | -8.24e-3 | +4.41e-3 | -4.23e-4 | -6.28e-4 |
| 24 | 3.00e-1 | 8 | 1.01e-1 | 1.49e-1 | 1.12e-1 | 1.07e-1 | 39 | -6.81e-3 | +3.76e-3 | -5.54e-4 | -5.20e-4 |
| 25 | 3.00e-1 | 6 | 9.73e-2 | 1.55e-1 | 1.11e-1 | 1.04e-1 | 36 | -1.05e-2 | +4.44e-3 | -1.10e-3 | -7.05e-4 |
| 26 | 3.00e-1 | 6 | 9.70e-2 | 1.43e-1 | 1.10e-1 | 1.00e-1 | 35 | -7.57e-3 | +4.18e-3 | -8.91e-4 | -7.89e-4 |
| 27 | 3.00e-1 | 7 | 9.22e-2 | 1.49e-1 | 1.04e-1 | 1.05e-1 | 36 | -1.39e-2 | +4.73e-3 | -8.77e-4 | -6.54e-4 |
| 28 | 3.00e-1 | 7 | 1.04e-1 | 1.52e-1 | 1.14e-1 | 1.09e-1 | 40 | -9.30e-3 | +4.56e-3 | -6.53e-4 | -6.24e-4 |
| 29 | 3.00e-1 | 6 | 9.54e-2 | 1.53e-1 | 1.10e-1 | 1.03e-1 | 36 | -1.09e-2 | +3.93e-3 | -1.17e-3 | -8.05e-4 |
| 30 | 3.00e-1 | 7 | 9.48e-2 | 1.45e-1 | 1.08e-1 | 9.48e-2 | 32 | -6.70e-3 | +4.15e-3 | -1.14e-3 | -1.00e-3 |
| 31 | 3.00e-1 | 7 | 9.78e-2 | 1.45e-1 | 1.07e-1 | 1.03e-1 | 37 | -1.04e-2 | +5.51e-3 | -6.02e-4 | -7.28e-4 |
| 32 | 3.00e-1 | 10 | 9.15e-2 | 1.47e-1 | 1.01e-1 | 9.30e-2 | 29 | -1.35e-2 | +4.55e-3 | -1.11e-3 | -8.53e-4 |
| 33 | 3.00e-1 | 4 | 1.03e-1 | 1.50e-1 | 1.15e-1 | 1.03e-1 | 35 | -9.90e-3 | +6.36e-3 | -9.75e-4 | -9.37e-4 |
| 34 | 3.00e-1 | 6 | 1.03e-1 | 1.62e-1 | 1.18e-1 | 1.07e-1 | 38 | -1.09e-2 | +4.75e-3 | -1.09e-3 | -9.86e-4 |
| 35 | 3.00e-1 | 6 | 1.02e-1 | 1.52e-1 | 1.20e-1 | 1.17e-1 | 46 | -7.04e-3 | +4.29e-3 | -3.64e-4 | -6.67e-4 |
| 36 | 3.00e-1 | 6 | 1.05e-1 | 1.52e-1 | 1.15e-1 | 1.09e-1 | 43 | -9.24e-3 | +3.14e-3 | -8.46e-4 | -6.91e-4 |
| 37 | 3.00e-1 | 6 | 1.07e-1 | 1.59e-1 | 1.20e-1 | 1.15e-1 | 43 | -6.96e-3 | +4.38e-3 | -4.52e-4 | -5.52e-4 |
| 38 | 3.00e-1 | 8 | 1.05e-1 | 1.43e-1 | 1.13e-1 | 1.10e-1 | 43 | -5.03e-3 | +3.05e-3 | -3.77e-4 | -3.94e-4 |
| 39 | 3.00e-1 | 4 | 1.10e-1 | 1.49e-1 | 1.20e-1 | 1.12e-1 | 43 | -6.92e-3 | +3.64e-3 | -7.44e-4 | -5.22e-4 |
| 40 | 3.00e-1 | 8 | 8.87e-2 | 1.48e-1 | 1.06e-1 | 8.87e-2 | 29 | -9.40e-3 | +3.47e-3 | -1.45e-3 | -1.05e-3 |
| 41 | 3.00e-1 | 6 | 9.96e-2 | 1.47e-1 | 1.11e-1 | 1.01e-1 | 37 | -8.46e-3 | +6.67e-3 | -5.99e-4 | -8.62e-4 |
| 42 | 3.00e-1 | 7 | 1.00e-1 | 1.51e-1 | 1.22e-1 | 1.25e-1 | 57 | -9.87e-3 | +5.12e-3 | -2.57e-4 | -5.00e-4 |
| 43 | 3.00e-1 | 3 | 1.26e-1 | 1.65e-1 | 1.40e-1 | 1.26e-1 | 55 | -4.50e-3 | +2.96e-3 | -6.19e-4 | -5.62e-4 |
| 44 | 3.00e-1 | 6 | 1.10e-1 | 1.64e-1 | 1.21e-1 | 1.15e-1 | 44 | -8.51e-3 | +2.85e-3 | -8.64e-4 | -6.41e-4 |
| 45 | 3.00e-1 | 6 | 9.60e-2 | 1.54e-1 | 1.12e-1 | 1.15e-1 | 47 | -1.19e-2 | +3.17e-3 | -9.59e-4 | -6.42e-4 |
| 46 | 3.00e-1 | 6 | 1.09e-1 | 1.43e-1 | 1.18e-1 | 1.12e-1 | 41 | -6.14e-3 | +2.61e-3 | -6.10e-4 | -6.27e-4 |
| 47 | 3.00e-1 | 7 | 9.40e-2 | 1.43e-1 | 1.08e-1 | 1.08e-1 | 40 | -9.31e-3 | +3.30e-3 | -6.31e-4 | -5.10e-4 |
| 48 | 3.00e-1 | 6 | 1.03e-1 | 1.48e-1 | 1.11e-1 | 1.03e-1 | 38 | -9.25e-3 | +3.82e-3 | -1.03e-3 | -7.14e-4 |
| 49 | 3.00e-1 | 7 | 9.50e-2 | 1.48e-1 | 1.08e-1 | 1.08e-1 | 37 | -1.27e-2 | +4.36e-3 | -9.22e-4 | -6.74e-4 |
| 50 | 3.00e-1 | 9 | 9.68e-2 | 1.56e-1 | 1.06e-1 | 9.87e-2 | 35 | -1.24e-2 | +4.39e-3 | -9.57e-4 | -7.08e-4 |
| 51 | 3.00e-1 | 5 | 1.11e-1 | 1.56e-1 | 1.24e-1 | 1.20e-1 | 46 | -6.88e-3 | +5.78e-3 | -1.01e-4 | -4.63e-4 |
| 52 | 3.00e-1 | 8 | 9.87e-2 | 1.61e-1 | 1.13e-1 | 1.12e-1 | 42 | -8.91e-3 | +3.21e-3 | -7.60e-4 | -4.73e-4 |
| 53 | 3.00e-1 | 4 | 9.93e-2 | 1.44e-1 | 1.12e-1 | 1.04e-1 | 38 | -9.04e-3 | +3.65e-3 | -1.26e-3 | -7.35e-4 |
| 54 | 3.00e-1 | 7 | 1.16e-1 | 1.62e-1 | 1.27e-1 | 1.18e-1 | 47 | -5.30e-3 | +4.70e-3 | -1.92e-4 | -4.75e-4 |
| 55 | 3.00e-1 | 3 | 1.10e-1 | 1.67e-1 | 1.31e-1 | 1.10e-1 | 43 | -7.70e-3 | +3.52e-3 | -1.83e-3 | -8.87e-4 |
| 56 | 3.00e-1 | 6 | 1.00e-1 | 1.54e-1 | 1.15e-1 | 1.08e-1 | 42 | -7.45e-3 | +4.02e-3 | -7.72e-4 | -8.01e-4 |
| 57 | 3.00e-1 | 8 | 1.08e-1 | 1.55e-1 | 1.16e-1 | 1.09e-1 | 42 | -8.54e-3 | +4.32e-3 | -5.05e-4 | -5.96e-4 |
| 58 | 3.00e-1 | 5 | 9.48e-2 | 1.54e-1 | 1.13e-1 | 1.13e-1 | 39 | -1.23e-2 | +3.74e-3 | -1.01e-3 | -6.56e-4 |
| 59 | 3.00e-1 | 8 | 9.61e-2 | 1.65e-1 | 1.07e-1 | 1.01e-1 | 38 | -1.31e-2 | +3.82e-3 | -1.30e-3 | -8.33e-4 |
| 60 | 3.00e-1 | 4 | 1.10e-1 | 1.46e-1 | 1.22e-1 | 1.16e-1 | 49 | -6.19e-3 | +4.84e-3 | -5.32e-5 | -5.90e-4 |
| 61 | 3.00e-1 | 6 | 9.64e-2 | 1.58e-1 | 1.14e-1 | 9.82e-2 | 33 | -6.95e-3 | +3.19e-3 | -1.31e-3 | -9.28e-4 |
| 62 | 3.00e-1 | 10 | 9.51e-2 | 1.40e-1 | 1.04e-1 | 1.06e-1 | 36 | -9.40e-3 | +5.00e-3 | -3.81e-4 | -4.17e-4 |
| 63 | 3.00e-1 | 5 | 9.25e-2 | 1.56e-1 | 1.10e-1 | 1.04e-1 | 37 | -1.19e-2 | +4.64e-3 | -1.47e-3 | -7.63e-4 |
| 64 | 3.00e-1 | 7 | 1.01e-1 | 1.56e-1 | 1.12e-1 | 1.01e-1 | 37 | -1.14e-2 | +4.95e-3 | -9.65e-4 | -8.35e-4 |
| 65 | 3.00e-1 | 7 | 9.02e-2 | 1.60e-1 | 1.08e-1 | 9.02e-2 | 28 | -1.50e-2 | +5.27e-3 | -1.97e-3 | -1.42e-3 |
| 66 | 3.00e-1 | 8 | 8.83e-2 | 1.43e-1 | 1.00e-1 | 8.83e-2 | 28 | -1.49e-2 | +6.27e-3 | -1.34e-3 | -1.34e-3 |
| 67 | 3.00e-1 | 7 | 1.07e-1 | 1.52e-1 | 1.18e-1 | 1.19e-1 | 44 | -7.07e-3 | +6.85e-3 | +1.33e-4 | -5.70e-4 |
| 68 | 3.00e-1 | 5 | 1.04e-1 | 1.57e-1 | 1.18e-1 | 1.04e-1 | 37 | -9.52e-3 | +3.30e-3 | -1.46e-3 | -9.24e-4 |
| 69 | 3.00e-1 | 6 | 9.92e-2 | 1.50e-1 | 1.15e-1 | 1.11e-1 | 41 | -6.25e-3 | +4.58e-3 | -4.52e-4 | -6.83e-4 |
| 70 | 3.00e-1 | 6 | 1.01e-1 | 1.41e-1 | 1.13e-1 | 1.16e-1 | 45 | -6.66e-3 | +3.25e-3 | -3.42e-4 | -4.41e-4 |
| 71 | 3.00e-1 | 8 | 1.01e-1 | 1.73e-1 | 1.18e-1 | 1.05e-1 | 39 | -7.82e-3 | +4.06e-3 | -9.05e-4 | -6.58e-4 |
| 72 | 3.00e-1 | 4 | 1.12e-1 | 1.47e-1 | 1.24e-1 | 1.23e-1 | 46 | -5.15e-3 | +4.15e-3 | +7.33e-5 | -4.09e-4 |
| 73 | 3.00e-1 | 9 | 9.96e-2 | 1.58e-1 | 1.11e-1 | 1.10e-1 | 39 | -9.05e-3 | +2.83e-3 | -7.06e-4 | -4.03e-4 |
| 74 | 3.00e-1 | 3 | 1.00e-1 | 1.50e-1 | 1.20e-1 | 1.00e-1 | 36 | -8.35e-3 | +4.02e-3 | -2.18e-3 | -9.41e-4 |
| 75 | 3.00e-1 | 8 | 1.10e-1 | 1.53e-1 | 1.23e-1 | 1.13e-1 | 47 | -4.06e-3 | +5.00e-3 | -2.38e-4 | -6.08e-4 |
| 76 | 3.00e-1 | 3 | 1.11e-1 | 1.53e-1 | 1.25e-1 | 1.12e-1 | 48 | -6.65e-3 | +3.74e-3 | -9.32e-4 | -7.27e-4 |
| 77 | 3.00e-1 | 5 | 1.14e-1 | 1.60e-1 | 1.26e-1 | 1.14e-1 | 43 | -5.46e-3 | +4.19e-3 | -5.37e-4 | -6.80e-4 |
| 78 | 3.00e-1 | 6 | 1.09e-1 | 1.51e-1 | 1.18e-1 | 1.09e-1 | 43 | -6.83e-3 | +3.46e-3 | -7.06e-4 | -6.82e-4 |
| 79 | 3.00e-1 | 6 | 1.06e-1 | 1.41e-1 | 1.17e-1 | 1.06e-1 | 43 | -4.85e-3 | +3.69e-3 | -4.94e-4 | -6.39e-4 |
| 80 | 3.00e-1 | 6 | 1.07e-1 | 1.52e-1 | 1.18e-1 | 1.13e-1 | 40 | -8.78e-3 | +4.33e-3 | -5.40e-4 | -5.68e-4 |
| 81 | 3.00e-1 | 7 | 9.79e-2 | 1.62e-1 | 1.11e-1 | 1.07e-1 | 42 | -1.38e-2 | +3.91e-3 | -1.17e-3 | -7.27e-4 |
| 82 | 3.00e-1 | 6 | 1.01e-1 | 1.48e-1 | 1.16e-1 | 1.01e-1 | 37 | -7.03e-3 | +4.15e-3 | -8.60e-4 | -8.40e-4 |
| 83 | 3.00e-1 | 7 | 9.67e-2 | 1.53e-1 | 1.11e-1 | 1.08e-1 | 39 | -1.08e-2 | +4.99e-3 | -7.38e-4 | -6.87e-4 |
| 84 | 3.00e-1 | 7 | 9.92e-2 | 1.53e-1 | 1.17e-1 | 1.21e-1 | 45 | -1.08e-2 | +4.39e-3 | -5.03e-4 | -4.74e-4 |
| 85 | 3.00e-1 | 5 | 1.07e-1 | 1.52e-1 | 1.19e-1 | 1.07e-1 | 42 | -6.20e-3 | +2.57e-3 | -1.03e-3 | -6.92e-4 |
| 86 | 3.00e-1 | 6 | 1.10e-1 | 1.52e-1 | 1.19e-1 | 1.10e-1 | 42 | -6.94e-3 | +4.21e-3 | -5.57e-4 | -6.31e-4 |
| 87 | 3.00e-1 | 6 | 1.02e-1 | 1.58e-1 | 1.14e-1 | 1.04e-1 | 35 | -7.19e-3 | +4.41e-3 | -9.73e-4 | -7.65e-4 |
| 88 | 3.00e-1 | 7 | 8.74e-2 | 1.49e-1 | 1.03e-1 | 9.89e-2 | 33 | -1.39e-2 | +4.90e-3 | -1.21e-3 | -8.56e-4 |
| 89 | 3.00e-1 | 8 | 9.80e-2 | 1.65e-1 | 1.16e-1 | 1.31e-1 | 53 | -1.57e-2 | +5.57e-3 | -6.33e-4 | -4.80e-4 |
| 90 | 3.00e-1 | 3 | 1.22e-1 | 1.63e-1 | 1.36e-1 | 1.22e-1 | 53 | -5.33e-3 | +2.33e-3 | -1.06e-3 | -6.59e-4 |
| 91 | 3.00e-1 | 7 | 1.16e-1 | 1.64e-1 | 1.24e-1 | 1.19e-1 | 50 | -7.42e-3 | +3.17e-3 | -5.21e-4 | -5.33e-4 |
| 92 | 3.00e-1 | 4 | 1.14e-1 | 1.53e-1 | 1.27e-1 | 1.22e-1 | 50 | -5.90e-3 | +2.98e-3 | -3.87e-4 | -4.80e-4 |
| 93 | 3.00e-1 | 7 | 1.09e-1 | 1.56e-1 | 1.18e-1 | 1.13e-1 | 44 | -7.44e-3 | +2.72e-3 | -6.60e-4 | -5.00e-4 |
| 94 | 3.00e-1 | 4 | 1.02e-1 | 1.52e-1 | 1.20e-1 | 1.02e-1 | 37 | -7.48e-3 | +3.92e-3 | -1.61e-3 | -9.43e-4 |
| 95 | 3.00e-1 | 7 | 1.00e-1 | 1.43e-1 | 1.11e-1 | 1.17e-1 | 41 | -8.74e-3 | +4.56e-3 | -2.25e-4 | -4.40e-4 |
| 96 | 3.00e-1 | 7 | 8.89e-2 | 1.51e-1 | 1.09e-1 | 1.12e-1 | 42 | -1.40e-2 | +3.15e-3 | -1.08e-3 | -5.62e-4 |
| 97 | 3.00e-1 | 6 | 9.17e-2 | 1.51e-1 | 1.15e-1 | 1.23e-1 | 45 | -7.65e-3 | +5.19e-3 | -5.31e-4 | -4.23e-4 |
| 98 | 3.00e-1 | 5 | 1.06e-1 | 1.53e-1 | 1.27e-1 | 1.29e-1 | 55 | -9.37e-3 | +2.98e-3 | -5.68e-4 | -4.29e-4 |
| 99 | 3.00e-1 | 5 | 1.20e-1 | 1.58e-1 | 1.32e-1 | 1.20e-1 | 52 | -3.75e-3 | +2.37e-3 | -5.28e-4 | -4.92e-4 |
| 100 | 3.00e-2 | 7 | 9.52e-3 | 1.57e-1 | 3.74e-2 | 1.03e-2 | 44 | -3.13e-2 | +2.74e-3 | -9.87e-3 | -4.67e-3 |
| 101 | 3.00e-2 | 5 | 1.04e-2 | 1.38e-2 | 1.16e-2 | 1.15e-2 | 44 | -4.77e-3 | +3.55e-3 | -1.16e-4 | -2.82e-3 |
| 102 | 3.00e-2 | 6 | 1.11e-2 | 1.51e-2 | 1.20e-2 | 1.16e-2 | 41 | -5.56e-3 | +3.28e-3 | -4.49e-4 | -1.68e-3 |
| 103 | 3.00e-2 | 9 | 1.06e-2 | 1.50e-2 | 1.20e-2 | 1.06e-2 | 32 | -6.19e-3 | +3.53e-3 | -6.61e-4 | -1.14e-3 |
| 104 | 3.00e-2 | 4 | 1.04e-2 | 1.56e-2 | 1.22e-2 | 1.11e-2 | 36 | -8.56e-3 | +5.04e-3 | -1.44e-3 | -1.26e-3 |
| 105 | 3.00e-2 | 8 | 1.00e-2 | 1.64e-2 | 1.16e-2 | 1.12e-2 | 32 | -1.03e-2 | +5.05e-3 | -8.69e-4 | -8.98e-4 |
| 106 | 3.00e-2 | 7 | 1.10e-2 | 1.58e-2 | 1.22e-2 | 1.17e-2 | 35 | -9.90e-3 | +4.71e-3 | -6.79e-4 | -7.26e-4 |
| 107 | 3.00e-2 | 8 | 1.09e-2 | 1.70e-2 | 1.25e-2 | 1.17e-2 | 35 | -5.90e-3 | +5.14e-3 | -6.66e-4 | -6.42e-4 |
| 108 | 3.00e-2 | 7 | 1.18e-2 | 1.69e-2 | 1.33e-2 | 1.32e-2 | 36 | -9.42e-3 | +5.12e-3 | -3.40e-4 | -4.26e-4 |
| 109 | 3.00e-2 | 7 | 1.19e-2 | 1.83e-2 | 1.35e-2 | 1.35e-2 | 38 | -1.40e-2 | +3.91e-3 | -9.56e-4 | -5.52e-4 |
| 110 | 3.00e-2 | 7 | 1.30e-2 | 1.88e-2 | 1.42e-2 | 1.38e-2 | 39 | -8.18e-3 | +4.55e-3 | -5.62e-4 | -4.91e-4 |
| 111 | 3.00e-2 | 8 | 1.32e-2 | 2.12e-2 | 1.46e-2 | 1.37e-2 | 36 | -1.15e-2 | +4.53e-3 | -9.36e-4 | -6.30e-4 |
| 112 | 3.00e-2 | 5 | 1.38e-2 | 1.96e-2 | 1.58e-2 | 1.56e-2 | 42 | -8.09e-3 | +4.73e-3 | -4.06e-4 | -5.11e-4 |
| 113 | 3.00e-2 | 7 | 1.38e-2 | 1.95e-2 | 1.50e-2 | 1.50e-2 | 37 | -9.61e-3 | +2.83e-3 | -6.75e-4 | -4.85e-4 |
| 114 | 3.00e-2 | 7 | 1.36e-2 | 2.11e-2 | 1.53e-2 | 1.45e-2 | 35 | -9.66e-3 | +4.54e-3 | -8.42e-4 | -5.94e-4 |
| 115 | 3.00e-2 | 6 | 1.34e-2 | 2.09e-2 | 1.61e-2 | 1.65e-2 | 43 | -1.70e-2 | +5.05e-3 | -1.18e-3 | -7.37e-4 |
| 116 | 3.00e-2 | 6 | 1.55e-2 | 2.32e-2 | 1.76e-2 | 1.55e-2 | 38 | -8.25e-3 | +3.67e-3 | -1.16e-3 | -9.55e-4 |
| 117 | 3.00e-2 | 7 | 1.45e-2 | 2.34e-2 | 1.82e-2 | 2.07e-2 | 57 | -1.01e-2 | +4.83e-3 | -2.22e-4 | -4.13e-4 |
| 118 | 3.00e-2 | 4 | 2.04e-2 | 2.50e-2 | 2.19e-2 | 2.07e-2 | 61 | -2.54e-3 | +2.01e-3 | -2.98e-4 | -3.87e-4 |
| 119 | 3.00e-2 | 5 | 1.78e-2 | 2.67e-2 | 2.02e-2 | 1.83e-2 | 44 | -5.16e-3 | +2.38e-3 | -1.04e-3 | -6.38e-4 |
| 120 | 3.00e-2 | 5 | 1.85e-2 | 2.54e-2 | 2.02e-2 | 1.96e-2 | 45 | -7.07e-3 | +3.47e-3 | -4.53e-4 | -5.39e-4 |
| 121 | 3.00e-2 | 5 | 1.84e-2 | 2.55e-2 | 2.04e-2 | 2.01e-2 | 50 | -7.28e-3 | +3.24e-3 | -4.40e-4 | -4.64e-4 |
| 122 | 3.00e-2 | 6 | 1.83e-2 | 2.68e-2 | 2.04e-2 | 1.83e-2 | 42 | -5.73e-3 | +3.14e-3 | -7.89e-4 | -6.17e-4 |
| 123 | 3.00e-2 | 6 | 1.92e-2 | 2.55e-2 | 2.06e-2 | 2.02e-2 | 45 | -5.79e-3 | +3.98e-3 | -2.25e-4 | -4.12e-4 |
| 124 | 3.00e-2 | 8 | 1.80e-2 | 2.55e-2 | 2.07e-2 | 1.99e-2 | 40 | -6.86e-3 | +2.87e-3 | -5.63e-4 | -4.32e-4 |
| 125 | 3.00e-2 | 4 | 1.83e-2 | 2.67e-2 | 2.09e-2 | 1.83e-2 | 35 | -8.45e-3 | +3.63e-3 | -1.74e-3 | -9.03e-4 |
| 126 | 3.00e-2 | 6 | 1.85e-2 | 2.51e-2 | 2.03e-2 | 2.07e-2 | 42 | -7.52e-3 | +4.30e-3 | -1.98e-4 | -5.16e-4 |
| 127 | 3.00e-2 | 9 | 1.78e-2 | 3.09e-2 | 2.08e-2 | 1.80e-2 | 38 | -1.00e-2 | +4.03e-3 | -1.15e-3 | -8.34e-4 |
| 128 | 3.00e-2 | 4 | 2.07e-2 | 2.77e-2 | 2.31e-2 | 2.29e-2 | 51 | -6.45e-3 | +5.59e-3 | +1.33e-4 | -5.17e-4 |
| 129 | 3.00e-2 | 7 | 1.82e-2 | 2.84e-2 | 2.03e-2 | 1.93e-2 | 34 | -1.16e-2 | +2.50e-3 | -1.34e-3 | -7.95e-4 |
| 130 | 3.00e-2 | 7 | 1.65e-2 | 2.51e-2 | 2.13e-2 | 2.51e-2 | 51 | -7.32e-3 | +5.72e-3 | -1.64e-5 | -2.28e-4 |
| 131 | 3.00e-2 | 4 | 2.43e-2 | 3.07e-2 | 2.63e-2 | 2.44e-2 | 49 | -3.08e-3 | +2.25e-3 | -5.08e-4 | -3.43e-4 |
| 132 | 3.00e-2 | 9 | 2.13e-2 | 3.36e-2 | 2.39e-2 | 2.25e-2 | 43 | -7.56e-3 | +3.27e-3 | -6.81e-4 | -4.48e-4 |
| 133 | 3.00e-2 | 3 | 2.31e-2 | 3.37e-2 | 2.67e-2 | 2.31e-2 | 41 | -9.05e-3 | +4.42e-3 | -1.61e-3 | -8.03e-4 |
| 134 | 3.00e-2 | 6 | 2.25e-2 | 3.02e-2 | 2.44e-2 | 2.47e-2 | 48 | -6.39e-3 | +3.11e-3 | -3.56e-4 | -5.36e-4 |
| 135 | 3.00e-2 | 5 | 2.13e-2 | 3.18e-2 | 2.56e-2 | 2.50e-2 | 50 | -1.14e-2 | +3.06e-3 | -1.08e-3 | -7.09e-4 |
| 136 | 3.00e-2 | 2 | 2.70e-2 | 5.85e-2 | 4.27e-2 | 5.85e-2 | 310 | +1.30e-3 | +2.50e-3 | +1.90e-3 | -2.07e-4 |
| 138 | 3.00e-2 | 1 | 6.59e-2 | 6.59e-2 | 6.59e-2 | 6.59e-2 | 387 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -1.56e-4 |
| 139 | 3.00e-2 | 1 | 6.17e-2 | 6.17e-2 | 6.17e-2 | 6.17e-2 | 312 | -2.09e-4 | -2.09e-4 | -2.09e-4 | -1.61e-4 |
| 140 | 3.00e-2 | 1 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 6.18e-2 | 335 | +3.17e-6 | +3.17e-6 | +3.17e-6 | -1.45e-4 |
| 141 | 3.00e-2 | 1 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 6.01e-2 | 290 | -9.25e-5 | -9.25e-5 | -9.25e-5 | -1.39e-4 |
| 142 | 3.00e-2 | 1 | 6.03e-2 | 6.03e-2 | 6.03e-2 | 6.03e-2 | 279 | +6.92e-6 | +6.92e-6 | +6.92e-6 | -1.25e-4 |
| 143 | 3.00e-2 | 1 | 6.16e-2 | 6.16e-2 | 6.16e-2 | 6.16e-2 | 293 | +7.56e-5 | +7.56e-5 | +7.56e-5 | -1.05e-4 |
| 144 | 3.00e-2 | 1 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 6.15e-2 | 296 | -2.93e-6 | -2.93e-6 | -2.93e-6 | -9.45e-5 |
| 145 | 3.00e-2 | 1 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 6.22e-2 | 287 | +3.83e-5 | +3.83e-5 | +3.83e-5 | -8.13e-5 |
| 146 | 3.00e-2 | 1 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 305 | +4.74e-5 | +4.74e-5 | +4.74e-5 | -6.84e-5 |
| 147 | 3.00e-2 | 1 | 5.99e-2 | 5.99e-2 | 5.99e-2 | 5.99e-2 | 265 | -1.98e-4 | -1.98e-4 | -1.98e-4 | -8.13e-5 |
| 148 | 3.00e-2 | 1 | 6.34e-2 | 6.34e-2 | 6.34e-2 | 6.34e-2 | 276 | +2.07e-4 | +2.07e-4 | +2.07e-4 | -5.25e-5 |
| 149 | 3.00e-2 | 1 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 6.31e-2 | 265 | -2.25e-5 | -2.25e-5 | -2.25e-5 | -4.95e-5 |
| 151 | 3.00e-3 | 2 | 8.55e-3 | 2.30e-2 | 1.58e-2 | 8.55e-3 | 265 | -3.74e-3 | -3.18e-3 | -3.46e-3 | -7.00e-4 |
| 153 | 3.00e-3 | 2 | 5.77e-3 | 6.55e-3 | 6.16e-3 | 5.77e-3 | 265 | -7.92e-4 | -4.78e-4 | -6.35e-4 | -6.86e-4 |
| 155 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 345 | +3.64e-4 | +3.64e-4 | +3.64e-4 | -5.81e-4 |
| 156 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 279 | -3.44e-4 | -3.44e-4 | -3.44e-4 | -5.57e-4 |
| 157 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 281 | -1.97e-5 | -1.97e-5 | -1.97e-5 | -5.04e-4 |
| 158 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 289 | +2.29e-4 | +2.29e-4 | +2.29e-4 | -4.30e-4 |
| 159 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 296 | -3.34e-5 | -3.34e-5 | -3.34e-5 | -3.91e-4 |
| 160 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 273 | -2.27e-5 | -2.27e-5 | -2.27e-5 | -3.54e-4 |
| 161 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 287 | +6.92e-5 | +6.92e-5 | +6.92e-5 | -3.12e-4 |
| 162 | 3.00e-3 | 1 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 6.32e-3 | 285 | -8.05e-6 | -8.05e-6 | -8.05e-6 | -2.81e-4 |
| 163 | 3.00e-3 | 1 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 6.13e-3 | 275 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -2.65e-4 |
| 164 | 3.00e-3 | 1 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 5.95e-3 | 251 | -1.15e-4 | -1.15e-4 | -1.15e-4 | -2.50e-4 |
| 165 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 258 | +3.60e-5 | +3.60e-5 | +3.60e-5 | -2.21e-4 |
| 166 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 269 | +3.16e-5 | +3.16e-5 | +3.16e-5 | -1.96e-4 |
| 167 | 3.00e-3 | 1 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 6.67e-3 | 315 | +3.06e-4 | +3.06e-4 | +3.06e-4 | -1.46e-4 |
| 168 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 300 | -6.54e-5 | -6.54e-5 | -6.54e-5 | -1.38e-4 |
| 169 | 3.00e-3 | 1 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 6.40e-3 | 288 | -7.48e-5 | -7.48e-5 | -7.48e-5 | -1.31e-4 |
| 170 | 3.00e-3 | 1 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 6.42e-3 | 275 | +8.55e-6 | +8.55e-6 | +8.55e-6 | -1.17e-4 |
| 171 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 265 | +4.70e-6 | +4.70e-6 | +4.70e-6 | -1.05e-4 |
| 172 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 268 | -9.00e-5 | -9.00e-5 | -9.00e-5 | -1.04e-4 |
| 173 | 3.00e-3 | 1 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 260 | +4.75e-5 | +4.75e-5 | +4.75e-5 | -8.85e-5 |
| 174 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 271 | +4.72e-5 | +4.72e-5 | +4.72e-5 | -7.49e-5 |
| 175 | 3.00e-3 | 2 | 6.08e-3 | 6.74e-3 | 6.41e-3 | 6.08e-3 | 227 | -4.59e-4 | +1.65e-4 | -1.47e-4 | -9.17e-5 |
| 177 | 3.00e-3 | 2 | 6.22e-3 | 6.24e-3 | 6.23e-3 | 6.24e-3 | 240 | +1.01e-5 | +1.01e-4 | +5.54e-5 | -6.42e-5 |
| 178 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 255 | +2.97e-5 | +2.97e-5 | +2.97e-5 | -5.48e-5 |
| 179 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 241 | -5.92e-5 | -5.92e-5 | -5.92e-5 | -5.53e-5 |
| 180 | 3.00e-3 | 1 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 225 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -6.22e-5 |
| 181 | 3.00e-3 | 2 | 6.13e-3 | 6.20e-3 | 6.16e-3 | 6.13e-3 | 226 | -5.14e-5 | +1.14e-4 | +3.15e-5 | -4.52e-5 |
| 183 | 3.00e-3 | 2 | 6.23e-3 | 6.73e-3 | 6.48e-3 | 6.23e-3 | 226 | -3.43e-4 | +3.19e-4 | -1.18e-5 | -4.22e-5 |
| 184 | 3.00e-3 | 1 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 6.56e-3 | 264 | +1.95e-4 | +1.95e-4 | +1.95e-4 | -1.85e-5 |
| 185 | 3.00e-3 | 1 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 6.48e-3 | 276 | -4.49e-5 | -4.49e-5 | -4.49e-5 | -2.11e-5 |
| 186 | 3.00e-3 | 1 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 6.12e-3 | 228 | -2.52e-4 | -2.52e-4 | -2.52e-4 | -4.42e-5 |
| 187 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 230 | +1.51e-4 | +1.51e-4 | +1.51e-4 | -2.47e-5 |
| 188 | 3.00e-3 | 2 | 6.12e-3 | 6.25e-3 | 6.19e-3 | 6.12e-3 | 211 | -9.83e-5 | -5.43e-5 | -7.63e-5 | -3.47e-5 |
| 190 | 3.00e-3 | 2 | 5.98e-3 | 6.68e-3 | 6.33e-3 | 5.98e-3 | 208 | -5.25e-4 | +3.10e-4 | -1.08e-4 | -5.27e-5 |
| 191 | 3.00e-3 | 1 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 240 | +2.44e-4 | +2.44e-4 | +2.44e-4 | -2.30e-5 |
| 192 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 232 | +6.95e-5 | +6.95e-5 | +6.95e-5 | -1.38e-5 |
| 193 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 224 | -1.74e-4 | -1.74e-4 | -1.74e-4 | -2.98e-5 |
| 194 | 3.00e-3 | 1 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 6.43e-3 | 249 | +1.42e-4 | +1.42e-4 | +1.42e-4 | -1.26e-5 |
| 195 | 3.00e-3 | 2 | 6.14e-3 | 6.49e-3 | 6.31e-3 | 6.14e-3 | 208 | -2.68e-4 | +3.76e-5 | -1.15e-4 | -3.36e-5 |
| 196 | 3.00e-3 | 1 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 6.90e-3 | 273 | +4.30e-4 | +4.30e-4 | +4.30e-4 | +1.28e-5 |
| 197 | 3.00e-3 | 1 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 6.24e-3 | 221 | -4.57e-4 | -4.57e-4 | -4.57e-4 | -3.42e-5 |
| 198 | 3.00e-3 | 1 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 6.41e-3 | 221 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -1.84e-5 |
| 199 | 3.00e-3 | 2 | 6.04e-3 | 6.29e-3 | 6.17e-3 | 6.04e-3 | 186 | -2.19e-4 | -8.26e-5 | -1.51e-4 | -4.42e-5 |

