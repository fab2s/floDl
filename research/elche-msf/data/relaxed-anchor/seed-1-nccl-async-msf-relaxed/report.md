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
| nccl-async | 0.048578 | 0.9200 | +0.0075 | 1870.7 | 592 | 37.6 | 100% | 100% | 100% | 6.8 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9200 | nccl-async | - | - |

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
| nccl-async | 1.9785 | 0.7434 | 0.5520 | 0.4912 | 0.4673 | 0.4529 | 0.4400 | 0.4370 | 0.4732 | 0.4548 | 0.1929 | 0.1573 | 0.1366 | 0.1322 | 0.1227 | 0.0672 | 0.0607 | 0.0541 | 0.0505 | 0.0486 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4036 | 2.7 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3039 | 3.4 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2925 | 3.5 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 394 | 392 | 380 | 373 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1869.2 | 1.4 | epoch-boundary(199) |
| nccl-async | gpu2 | 1869.2 | 1.4 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.0s | 0.0s | 0.0s | 0.0s | 2.0s |
| resnet-graph | nccl-async | gpu1 | 1.4s | 0.0s | 0.0s | 0.0s | 2.6s |
| resnet-graph | nccl-async | gpu2 | 1.4s | 0.0s | 0.0s | 0.0s | 2.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 24 | 0 | 592 | 37.6 | 1606/9589 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 150.3 | 8.0% |

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
| resnet-graph | nccl-async | 199 | 592 | 0 | 6.34e-3 | +7.24e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 592 | 1.00e-1 | 4.48e-2 | 0.00e0 | 4.76e-1 | 44.8 | -7.45e-5 | 1.35e-3 |
| resnet-graph | nccl-async | 1 | 592 | 1.02e-1 | 4.59e-2 | 0.00e0 | 3.96e-1 | 35.1 | -7.46e-5 | 1.84e-3 |
| resnet-graph | nccl-async | 2 | 592 | 1.01e-1 | 4.64e-2 | 0.00e0 | 4.61e-1 | 20.1 | -8.28e-5 | 1.91e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9802 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9836 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9948 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 85 (0,1,2,3,4,5,7,8…146,149) | 0 (—) | — | 0,1,2,3,4,5,7,8…146,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 24 | 24 |
| resnet-graph | nccl-async | 0e0 | 5 | 8 | 8 |
| resnet-graph | nccl-async | 0e0 | 10 | 3 | 3 |
| resnet-graph | nccl-async | 1e-4 | 3 | 3 | 3 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 488 | +0.041 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 49 | +0.327 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 50 | +0.040 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 589 | +0.008 | 198 | +0.134 | +0.343 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 590 | 3.36e1–8.16e1 | 6.93e1 | 2.35e-3 | 3.57e-3 | 4.81e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 490 | 70–78035 | +6.369e-6 | 0.338 | +6.508e-6 | 0.361 | 100 | +9.360e-6 | 0.683 | 34–953 | +1.002e-3 | 0.349 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 478 | 907–78035 | +6.420e-6 | 0.404 | +6.523e-6 | 0.429 | 99 | +9.415e-6 | 0.679 | 62–953 | +9.977e-4 | 0.426 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 50 | 78842–116781 | +1.414e-5 | 0.173 | +1.406e-5 | 0.170 | 50 | +1.406e-5 | 0.170 | 676–876 | +1.438e-3 | 0.028 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 51 | 117514–155849 | -1.011e-5 | 0.065 | -1.006e-5 | 0.064 | 49 | -1.042e-5 | 0.069 | 618–901 | -7.246e-4 | 0.007 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.002e-3 | r0: +1.009e-3, r1: +1.001e-3, r2: +9.998e-4 | r0: 0.408, r1: 0.320, r2: 0.308 | 1.01× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.977e-4 | r0: +1.010e-3, r1: +9.901e-4, r2: +9.963e-4 | r0: 0.527, r1: 0.374, r2: 0.372 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | +1.438e-3 | r0: +1.475e-3, r1: +1.376e-3, r2: +1.463e-3 | r0: 0.030, r1: 0.026, r2: 0.030 | 1.07× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | -7.246e-4 | r0: -6.086e-4, r1: -7.591e-4, r2: -8.007e-4 | r0: 0.005, r1: 0.007, r2: 0.008 | 1.32× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇████▅▄▄▄▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇█▇▇▇▇▇▇▇▇▇▇▇▇▇▇████▆▆▆▇▇▇███████▅▆▆▇▇▇█▇████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 13 | 0.00e0 | 4.76e-1 | 1.16e-1 | 6.96e-2 | 25 | -4.39e-2 | +1.19e-2 | -8.98e-3 | -5.71e-3 |
| 1 | 3.00e-1 | 10 | 6.69e-2 | 1.12e-1 | 7.53e-2 | 7.42e-2 | 24 | -2.04e-2 | +1.62e-2 | -9.46e-5 | -1.57e-3 |
| 2 | 3.00e-1 | 8 | 7.14e-2 | 1.14e-1 | 9.21e-2 | 9.51e-2 | 31 | -8.36e-3 | +1.52e-2 | +1.17e-3 | -7.43e-5 |
| 3 | 3.00e-1 | 7 | 9.11e-2 | 1.21e-1 | 1.00e-1 | 1.03e-1 | 36 | -7.18e-3 | +8.37e-3 | +4.40e-4 | +1.90e-4 |
| 4 | 3.00e-1 | 10 | 9.78e-2 | 1.46e-1 | 1.07e-1 | 1.03e-1 | 33 | -1.24e-2 | +9.29e-3 | -2.18e-4 | -9.99e-5 |
| 5 | 3.00e-1 | 5 | 9.48e-2 | 1.38e-1 | 1.04e-1 | 9.65e-2 | 30 | -1.23e-2 | +1.26e-2 | -1.07e-4 | -1.82e-4 |
| 6 | 3.00e-1 | 8 | 8.89e-2 | 1.38e-1 | 1.00e-1 | 8.89e-2 | 37 | -8.54e-3 | +1.07e-2 | -3.12e-4 | -3.87e-4 |
| 7 | 3.00e-1 | 7 | 9.79e-2 | 1.44e-1 | 1.08e-1 | 1.03e-1 | 39 | -1.13e-2 | +9.53e-3 | +2.88e-4 | -1.28e-4 |
| 8 | 3.00e-1 | 7 | 9.49e-2 | 1.41e-1 | 1.10e-1 | 9.49e-2 | 35 | -4.87e-3 | +5.81e-3 | -6.45e-4 | -5.05e-4 |
| 9 | 3.00e-1 | 7 | 9.91e-2 | 1.39e-1 | 1.10e-1 | 1.08e-1 | 30 | -6.28e-3 | +8.47e-3 | +6.74e-4 | +4.77e-5 |
| 10 | 3.00e-1 | 7 | 9.39e-2 | 1.40e-1 | 1.05e-1 | 1.01e-1 | 37 | -1.19e-2 | +1.21e-2 | +3.41e-5 | -1.21e-5 |
| 11 | 3.00e-1 | 8 | 9.86e-2 | 1.37e-1 | 1.07e-1 | 1.01e-1 | 39 | -7.89e-3 | +8.46e-3 | +2.91e-5 | -6.21e-5 |
| 12 | 3.00e-1 | 5 | 9.73e-2 | 1.36e-1 | 1.09e-1 | 1.11e-1 | 42 | -9.05e-3 | +7.53e-3 | +3.44e-4 | +9.09e-5 |
| 13 | 3.00e-1 | 7 | 9.76e-2 | 1.42e-1 | 1.08e-1 | 1.00e-1 | 35 | -7.37e-3 | +6.64e-3 | -4.43e-4 | -2.02e-4 |
| 14 | 3.00e-1 | 7 | 9.76e-2 | 1.43e-1 | 1.07e-1 | 1.11e-1 | 39 | -1.02e-2 | +1.02e-2 | +4.48e-4 | +1.41e-4 |
| 15 | 3.00e-1 | 7 | 9.66e-2 | 1.44e-1 | 1.07e-1 | 9.94e-2 | 33 | -1.16e-2 | +9.68e-3 | -3.93e-4 | -1.78e-4 |
| 16 | 3.00e-1 | 7 | 8.68e-2 | 1.39e-1 | 9.95e-2 | 9.61e-2 | 38 | -1.51e-2 | +1.20e-2 | -1.51e-4 | -1.80e-4 |
| 17 | 3.00e-1 | 6 | 1.03e-1 | 1.41e-1 | 1.14e-1 | 1.09e-1 | 38 | -5.57e-3 | +7.11e-3 | +2.96e-4 | -3.08e-5 |
| 18 | 3.00e-1 | 7 | 9.98e-2 | 1.41e-1 | 1.09e-1 | 1.02e-1 | 38 | -8.72e-3 | +8.04e-3 | -1.51e-4 | -1.38e-4 |
| 19 | 3.00e-1 | 6 | 1.01e-1 | 1.38e-1 | 1.12e-1 | 1.13e-1 | 46 | -7.71e-3 | +6.73e-3 | +2.95e-4 | +3.99e-5 |
| 20 | 3.00e-1 | 6 | 1.11e-1 | 1.44e-1 | 1.17e-1 | 1.13e-1 | 45 | -5.18e-3 | +5.56e-3 | +1.76e-5 | -4.19e-6 |
| 21 | 3.00e-1 | 6 | 9.61e-2 | 1.50e-1 | 1.10e-1 | 1.02e-1 | 40 | -1.22e-2 | +7.39e-3 | -7.00e-4 | -3.44e-4 |
| 22 | 3.00e-1 | 9 | 9.03e-2 | 1.49e-1 | 1.08e-1 | 1.02e-1 | 38 | -6.26e-3 | +8.73e-3 | +1.94e-5 | -1.94e-4 |
| 23 | 3.00e-1 | 4 | 1.03e-1 | 1.44e-1 | 1.16e-1 | 1.08e-1 | 40 | -8.34e-3 | +7.32e-3 | +2.09e-4 | -1.17e-4 |
| 24 | 3.00e-1 | 6 | 9.96e-2 | 1.44e-1 | 1.11e-1 | 1.04e-1 | 37 | -6.93e-3 | +7.66e-3 | +2.80e-5 | -1.08e-4 |
| 25 | 3.00e-1 | 6 | 9.54e-2 | 1.44e-1 | 1.10e-1 | 1.09e-1 | 44 | -9.65e-3 | +1.06e-2 | +4.52e-4 | +1.13e-4 |
| 26 | 3.00e-1 | 7 | 1.04e-1 | 1.43e-1 | 1.19e-1 | 1.10e-1 | 49 | -4.08e-3 | +6.35e-3 | +6.12e-5 | -8.90e-6 |
| 27 | 3.00e-1 | 4 | 1.13e-1 | 1.48e-1 | 1.23e-1 | 1.16e-1 | 50 | -5.52e-3 | +5.23e-3 | +1.68e-4 | +3.54e-6 |
| 28 | 3.00e-1 | 5 | 1.04e-1 | 1.56e-1 | 1.19e-1 | 1.08e-1 | 43 | -9.48e-3 | +5.44e-3 | -5.06e-4 | -2.42e-4 |
| 29 | 3.00e-1 | 5 | 1.00e-1 | 1.45e-1 | 1.20e-1 | 1.21e-1 | 54 | -4.86e-3 | +7.45e-3 | +5.37e-4 | +4.45e-5 |
| 30 | 3.00e-1 | 6 | 1.12e-1 | 1.54e-1 | 1.23e-1 | 1.16e-1 | 43 | -6.42e-3 | +4.45e-3 | -1.99e-4 | -9.41e-5 |
| 31 | 3.00e-1 | 6 | 9.80e-2 | 1.46e-1 | 1.11e-1 | 1.06e-1 | 42 | -1.08e-2 | +7.50e-3 | -3.32e-4 | -2.11e-4 |
| 32 | 3.00e-1 | 5 | 1.05e-1 | 1.48e-1 | 1.18e-1 | 1.16e-1 | 43 | -7.00e-3 | +7.48e-3 | +4.06e-4 | +2.99e-7 |
| 33 | 3.00e-1 | 7 | 1.01e-1 | 1.62e-1 | 1.12e-1 | 1.01e-1 | 40 | -1.09e-2 | +9.95e-3 | -4.04e-4 | -2.58e-4 |
| 34 | 3.00e-1 | 6 | 1.05e-1 | 1.50e-1 | 1.15e-1 | 1.07e-1 | 34 | -7.20e-3 | +7.93e-3 | -5.02e-6 | -2.23e-4 |
| 35 | 3.00e-1 | 6 | 9.58e-2 | 1.53e-1 | 1.08e-1 | 9.90e-2 | 36 | -1.35e-2 | +1.24e-2 | -2.77e-4 | -3.12e-4 |
| 36 | 3.00e-1 | 7 | 9.84e-2 | 1.51e-1 | 1.11e-1 | 9.84e-2 | 36 | -8.06e-3 | +7.79e-3 | -4.02e-4 | -4.58e-4 |
| 37 | 3.00e-1 | 6 | 9.53e-2 | 1.43e-1 | 1.15e-1 | 1.11e-1 | 40 | -1.01e-2 | +8.13e-3 | +2.72e-4 | -2.03e-4 |
| 38 | 3.00e-1 | 7 | 1.01e-1 | 1.55e-1 | 1.16e-1 | 1.09e-1 | 51 | -1.06e-2 | +8.75e-3 | -7.19e-5 | -1.96e-4 |
| 39 | 3.00e-1 | 4 | 1.07e-1 | 1.55e-1 | 1.24e-1 | 1.12e-1 | 48 | -8.28e-3 | +5.52e-3 | -9.33e-5 | -2.26e-4 |
| 40 | 3.00e-1 | 5 | 1.09e-1 | 1.52e-1 | 1.22e-1 | 1.17e-1 | 48 | -6.48e-3 | +6.12e-3 | +2.07e-4 | -9.88e-5 |
| 41 | 3.00e-1 | 8 | 1.04e-1 | 1.48e-1 | 1.16e-1 | 1.16e-1 | 43 | -6.20e-3 | +5.78e-3 | -2.40e-5 | -3.03e-5 |
| 42 | 3.00e-1 | 5 | 9.43e-2 | 1.51e-1 | 1.12e-1 | 9.43e-2 | 38 | -1.01e-2 | +7.23e-3 | -1.15e-3 | -5.71e-4 |
| 43 | 3.00e-1 | 4 | 1.03e-1 | 1.52e-1 | 1.19e-1 | 1.14e-1 | 59 | -7.52e-3 | +9.09e-3 | +8.68e-4 | -1.53e-4 |
| 44 | 3.00e-1 | 6 | 1.15e-1 | 1.50e-1 | 1.25e-1 | 1.17e-1 | 52 | -5.19e-3 | +2.56e-3 | -1.20e-4 | -1.70e-4 |
| 45 | 3.00e-1 | 4 | 1.19e-1 | 1.58e-1 | 1.31e-1 | 1.19e-1 | 47 | -4.44e-3 | +4.17e-3 | -1.86e-4 | -2.34e-4 |
| 46 | 3.00e-1 | 5 | 1.12e-1 | 1.52e-1 | 1.23e-1 | 1.20e-1 | 51 | -5.51e-3 | +6.18e-3 | +6.87e-5 | -1.38e-4 |
| 47 | 3.00e-1 | 6 | 1.08e-1 | 1.51e-1 | 1.22e-1 | 1.21e-1 | 50 | -7.16e-3 | +4.39e-3 | -2.47e-5 | -8.22e-5 |
| 48 | 3.00e-1 | 5 | 1.10e-1 | 1.51e-1 | 1.25e-1 | 1.10e-1 | 54 | -3.69e-3 | +3.65e-3 | -5.90e-4 | -3.51e-4 |
| 49 | 3.00e-1 | 4 | 1.15e-1 | 1.58e-1 | 1.31e-1 | 1.30e-1 | 59 | -5.77e-3 | +5.13e-3 | +5.99e-4 | -5.80e-5 |
| 50 | 3.00e-1 | 8 | 1.12e-1 | 1.64e-1 | 1.22e-1 | 1.13e-1 | 45 | -7.10e-3 | +4.42e-3 | -4.00e-4 | -2.49e-4 |
| 51 | 3.00e-1 | 3 | 1.13e-1 | 1.52e-1 | 1.27e-1 | 1.13e-1 | 52 | -5.61e-3 | +5.24e-3 | -9.88e-6 | -2.43e-4 |
| 52 | 3.00e-1 | 5 | 1.09e-1 | 1.51e-1 | 1.24e-1 | 1.11e-1 | 49 | -6.57e-3 | +3.66e-3 | -2.63e-4 | -3.20e-4 |
| 53 | 3.00e-1 | 4 | 1.12e-1 | 1.58e-1 | 1.28e-1 | 1.25e-1 | 57 | -6.05e-3 | +5.61e-3 | +5.52e-4 | -5.32e-5 |
| 54 | 3.00e-1 | 5 | 1.25e-1 | 1.59e-1 | 1.34e-1 | 1.31e-1 | 49 | -3.75e-3 | +3.99e-3 | +1.52e-4 | +4.23e-6 |
| 55 | 3.00e-1 | 5 | 1.13e-1 | 1.51e-1 | 1.22e-1 | 1.13e-1 | 48 | -5.28e-3 | +5.72e-3 | -4.62e-4 | -2.16e-4 |
| 56 | 3.00e-1 | 7 | 1.08e-1 | 1.54e-1 | 1.21e-1 | 1.08e-1 | 48 | -5.49e-3 | +5.67e-3 | -1.75e-4 | -2.79e-4 |
| 57 | 3.00e-1 | 3 | 1.16e-1 | 1.53e-1 | 1.28e-1 | 1.17e-1 | 48 | -5.52e-3 | +5.77e-3 | +3.37e-4 | -1.74e-4 |
| 58 | 3.00e-1 | 5 | 1.08e-1 | 1.60e-1 | 1.22e-1 | 1.16e-1 | 48 | -7.35e-3 | +7.10e-3 | +2.99e-5 | -1.21e-4 |
| 59 | 3.00e-1 | 5 | 1.15e-1 | 1.48e-1 | 1.29e-1 | 1.34e-1 | 50 | -3.82e-3 | +4.93e-3 | +5.62e-4 | +1.42e-4 |
| 60 | 3.00e-1 | 5 | 1.09e-1 | 1.58e-1 | 1.24e-1 | 1.18e-1 | 55 | -6.81e-3 | +5.99e-3 | -3.17e-4 | -5.73e-5 |
| 61 | 3.00e-1 | 5 | 1.10e-1 | 1.76e-1 | 1.29e-1 | 1.13e-1 | 44 | -1.02e-2 | +6.99e-3 | -4.15e-4 | -2.75e-4 |
| 62 | 3.00e-1 | 5 | 1.12e-1 | 1.47e-1 | 1.25e-1 | 1.29e-1 | 47 | -4.43e-3 | +5.43e-3 | +5.79e-4 | +6.66e-5 |
| 63 | 3.00e-1 | 5 | 1.11e-1 | 1.55e-1 | 1.22e-1 | 1.16e-1 | 43 | -6.65e-3 | +6.34e-3 | -3.01e-4 | -1.03e-4 |
| 64 | 3.00e-1 | 6 | 1.07e-1 | 1.54e-1 | 1.20e-1 | 1.11e-1 | 53 | -7.69e-3 | +7.97e-3 | +7.48e-6 | -1.18e-4 |
| 65 | 3.00e-1 | 4 | 1.18e-1 | 1.54e-1 | 1.29e-1 | 1.22e-1 | 58 | -4.84e-3 | +4.07e-3 | +2.77e-4 | -2.91e-5 |
| 66 | 3.00e-1 | 5 | 1.24e-1 | 1.60e-1 | 1.33e-1 | 1.27e-1 | 55 | -3.79e-3 | +4.19e-3 | +8.87e-5 | -1.80e-5 |
| 67 | 3.00e-1 | 6 | 1.21e-1 | 1.58e-1 | 1.30e-1 | 1.27e-1 | 56 | -4.39e-3 | +4.80e-3 | +1.14e-4 | +2.00e-5 |
| 68 | 3.00e-1 | 3 | 1.23e-1 | 1.60e-1 | 1.36e-1 | 1.25e-1 | 60 | -4.13e-3 | +4.37e-3 | -4.53e-5 | -3.56e-5 |
| 69 | 3.00e-1 | 6 | 1.17e-1 | 1.52e-1 | 1.28e-1 | 1.23e-1 | 52 | -4.16e-3 | +2.42e-3 | -1.61e-4 | -1.03e-4 |
| 70 | 3.00e-1 | 3 | 1.14e-1 | 1.54e-1 | 1.33e-1 | 1.14e-1 | 52 | -5.88e-3 | +3.41e-3 | -6.25e-4 | -3.08e-4 |
| 71 | 3.00e-1 | 5 | 1.18e-1 | 1.53e-1 | 1.30e-1 | 1.29e-1 | 61 | -4.49e-3 | +4.51e-3 | +4.20e-4 | -4.29e-5 |
| 72 | 3.00e-1 | 4 | 1.28e-1 | 1.60e-1 | 1.39e-1 | 1.33e-1 | 55 | -3.71e-3 | +2.82e-3 | +7.13e-5 | -2.89e-5 |
| 73 | 3.00e-1 | 5 | 1.18e-1 | 1.54e-1 | 1.28e-1 | 1.22e-1 | 59 | -4.39e-3 | +4.40e-3 | -1.93e-4 | -1.11e-4 |
| 74 | 3.00e-1 | 5 | 1.05e-1 | 1.68e-1 | 1.27e-1 | 1.05e-1 | 44 | -7.52e-3 | +5.59e-3 | -9.06e-4 | -5.31e-4 |
| 75 | 3.00e-1 | 5 | 1.09e-1 | 1.53e-1 | 1.24e-1 | 1.21e-1 | 53 | -5.33e-3 | +6.92e-3 | +5.74e-4 | -1.40e-4 |
| 76 | 3.00e-1 | 5 | 1.15e-1 | 1.49e-1 | 1.28e-1 | 1.26e-1 | 53 | -4.87e-3 | +2.92e-3 | +7.39e-5 | -6.48e-5 |
| 77 | 3.00e-1 | 4 | 1.29e-1 | 1.67e-1 | 1.39e-1 | 1.29e-1 | 53 | -4.22e-3 | +4.14e-3 | -4.18e-5 | -1.05e-4 |
| 78 | 3.00e-1 | 6 | 1.14e-1 | 1.46e-1 | 1.25e-1 | 1.20e-1 | 58 | -4.89e-3 | +3.96e-3 | -1.38e-4 | -1.36e-4 |
| 79 | 3.00e-1 | 3 | 1.21e-1 | 1.64e-1 | 1.38e-1 | 1.21e-1 | 51 | -6.07e-3 | +4.62e-3 | -2.53e-4 | -2.35e-4 |
| 80 | 3.00e-1 | 6 | 1.18e-1 | 1.49e-1 | 1.28e-1 | 1.23e-1 | 52 | -2.81e-3 | +4.17e-3 | +1.33e-4 | -1.05e-4 |
| 81 | 3.00e-1 | 2 | 1.25e-1 | 1.26e-1 | 1.25e-1 | 1.25e-1 | 49 | -1.63e-4 | +4.73e-4 | +1.55e-4 | -5.89e-5 |
| 82 | 3.00e-1 | 1 | 1.11e-1 | 1.11e-1 | 1.11e-1 | 1.11e-1 | 280 | -4.13e-4 | -4.13e-4 | -4.13e-4 | -9.43e-5 |
| 83 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 290 | +2.61e-3 | +2.61e-3 | +2.61e-3 | +1.76e-4 |
| 84 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 305 | -1.89e-4 | -1.89e-4 | -1.89e-4 | +1.39e-4 |
| 85 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 273 | +2.37e-5 | +2.37e-5 | +2.37e-5 | +1.28e-4 |
| 86 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 283 | -8.97e-5 | -8.97e-5 | -8.97e-5 | +1.06e-4 |
| 87 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 246 | -4.04e-5 | -4.04e-5 | -4.04e-5 | +9.13e-5 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 232 | -1.66e-4 | -1.66e-4 | -1.66e-4 | +6.56e-5 |
| 89 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 249 | -1.19e-4 | -1.19e-4 | -1.19e-4 | +4.71e-5 |
| 90 | 3.00e-1 | 2 | 2.09e-1 | 2.20e-1 | 2.14e-1 | 2.20e-1 | 224 | +9.66e-5 | +2.28e-4 | +1.62e-4 | +6.96e-5 |
| 91 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 272 | -3.09e-4 | -3.09e-4 | -3.09e-4 | +3.17e-5 |
| 92 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 245 | +2.55e-4 | +2.55e-4 | +2.55e-4 | +5.40e-5 |
| 93 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 251 | -1.65e-4 | -1.65e-4 | -1.65e-4 | +3.21e-5 |
| 94 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 235 | +1.89e-5 | +1.89e-5 | +1.89e-5 | +3.08e-5 |
| 95 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 267 | -1.11e-4 | -1.11e-4 | -1.11e-4 | +1.66e-5 |
| 96 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 350 | +1.75e-4 | +1.75e-4 | +1.75e-4 | +3.24e-5 |
| 97 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 314 | +2.53e-4 | +2.53e-4 | +2.53e-4 | +5.44e-5 |
| 98 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 298 | -8.81e-5 | -8.81e-5 | -8.81e-5 | +4.02e-5 |
| 99 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 272 | -1.00e-4 | -1.00e-4 | -1.00e-4 | +2.61e-5 |
| 100 | 3.00e-2 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 293 | -4.11e-5 | -4.11e-5 | -4.11e-5 | +1.94e-5 |
| 101 | 3.00e-2 | 1 | 1.06e-1 | 1.06e-1 | 1.06e-1 | 1.06e-1 | 280 | -2.55e-3 | -2.55e-3 | -2.55e-3 | -2.38e-4 |
| 102 | 3.00e-2 | 1 | 2.28e-2 | 2.28e-2 | 2.28e-2 | 2.28e-2 | 262 | -5.86e-3 | -5.86e-3 | -5.86e-3 | -8.00e-4 |
| 103 | 3.00e-2 | 1 | 2.31e-2 | 2.31e-2 | 2.31e-2 | 2.31e-2 | 279 | +4.98e-5 | +4.98e-5 | +4.98e-5 | -7.15e-4 |
| 104 | 3.00e-2 | 1 | 2.53e-2 | 2.53e-2 | 2.53e-2 | 2.53e-2 | 265 | +3.38e-4 | +3.38e-4 | +3.38e-4 | -6.10e-4 |
| 105 | 3.00e-2 | 1 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 2.57e-2 | 282 | +4.67e-5 | +4.67e-5 | +4.67e-5 | -5.44e-4 |
| 106 | 3.00e-2 | 1 | 2.69e-2 | 2.69e-2 | 2.69e-2 | 2.69e-2 | 300 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -4.73e-4 |
| 107 | 3.00e-2 | 1 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 2.88e-2 | 252 | +2.60e-4 | +2.60e-4 | +2.60e-4 | -4.00e-4 |
| 108 | 3.00e-2 | 1 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 2.73e-2 | 265 | -2.04e-4 | -2.04e-4 | -2.04e-4 | -3.80e-4 |
| 109 | 3.00e-2 | 1 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 272 | +2.10e-4 | +2.10e-4 | +2.10e-4 | -3.21e-4 |
| 110 | 3.00e-2 | 1 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 2.93e-2 | 251 | +5.77e-5 | +5.77e-5 | +5.77e-5 | -2.83e-4 |
| 111 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 284 | +1.96e-5 | +1.96e-5 | +1.96e-5 | -2.53e-4 |
| 112 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 242 | +3.04e-4 | +3.04e-4 | +3.04e-4 | -1.97e-4 |
| 113 | 3.00e-2 | 1 | 3.03e-2 | 3.03e-2 | 3.03e-2 | 3.03e-2 | 297 | -1.52e-4 | -1.52e-4 | -1.52e-4 | -1.93e-4 |
| 114 | 3.00e-2 | 1 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 3.38e-2 | 270 | +4.08e-4 | +4.08e-4 | +4.08e-4 | -1.33e-4 |
| 115 | 3.00e-2 | 1 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 3.39e-2 | 298 | +1.19e-5 | +1.19e-5 | +1.19e-5 | -1.18e-4 |
| 116 | 3.00e-2 | 1 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 3.53e-2 | 273 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -9.22e-5 |
| 117 | 3.00e-2 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 325 | -2.60e-5 | -2.60e-5 | -2.60e-5 | -8.56e-5 |
| 118 | 3.00e-2 | 1 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 300 | +2.76e-4 | +2.76e-4 | +2.76e-4 | -4.95e-5 |
| 119 | 3.00e-2 | 1 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 290 | -1.26e-6 | -1.26e-6 | -1.26e-6 | -4.46e-5 |
| 120 | 3.00e-2 | 1 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 3.81e-2 | 234 | +8.97e-6 | +8.97e-6 | +8.97e-6 | -3.93e-5 |
| 121 | 3.00e-2 | 1 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 3.58e-2 | 251 | -2.47e-4 | -2.47e-4 | -2.47e-4 | -6.00e-5 |
| 122 | 3.00e-2 | 1 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 3.79e-2 | 285 | +1.99e-4 | +1.99e-4 | +1.99e-4 | -3.42e-5 |
| 123 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 282 | +2.02e-4 | +2.02e-4 | +2.02e-4 | -1.06e-5 |
| 124 | 3.00e-2 | 1 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 3.98e-2 | 264 | -2.37e-5 | -2.37e-5 | -2.37e-5 | -1.19e-5 |
| 125 | 3.00e-2 | 1 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 4.14e-2 | 272 | +1.38e-4 | +1.38e-4 | +1.38e-4 | +3.07e-6 |
| 126 | 3.00e-2 | 1 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 4.11e-2 | 240 | -2.54e-5 | -2.54e-5 | -2.54e-5 | +2.25e-7 |
| 127 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 263 | -9.21e-5 | -9.21e-5 | -9.21e-5 | -9.00e-6 |
| 128 | 3.00e-2 | 1 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 4.13e-2 | 285 | +9.94e-5 | +9.94e-5 | +9.94e-5 | +1.84e-6 |
| 129 | 3.00e-2 | 1 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 277 | +3.06e-4 | +3.06e-4 | +3.06e-4 | +3.22e-5 |
| 130 | 3.00e-2 | 1 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 4.43e-2 | 300 | -4.95e-5 | -4.95e-5 | -4.95e-5 | +2.41e-5 |
| 131 | 3.00e-2 | 1 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 4.75e-2 | 288 | +2.42e-4 | +2.42e-4 | +2.42e-4 | +4.59e-5 |
| 132 | 3.00e-2 | 1 | 4.80e-2 | 4.80e-2 | 4.80e-2 | 4.80e-2 | 266 | +3.98e-5 | +3.98e-5 | +3.98e-5 | +4.53e-5 |
| 133 | 3.00e-2 | 1 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 4.69e-2 | 266 | -8.87e-5 | -8.87e-5 | -8.87e-5 | +3.19e-5 |
| 134 | 3.00e-2 | 1 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 4.74e-2 | 281 | +4.34e-5 | +4.34e-5 | +4.34e-5 | +3.30e-5 |
| 135 | 3.00e-2 | 1 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 5.01e-2 | 300 | +1.86e-4 | +1.86e-4 | +1.86e-4 | +4.83e-5 |
| 136 | 3.00e-2 | 1 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 5.16e-2 | 309 | +9.05e-5 | +9.05e-5 | +9.05e-5 | +5.25e-5 |
| 137 | 3.00e-2 | 1 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 5.29e-2 | 264 | +9.46e-5 | +9.46e-5 | +9.46e-5 | +5.67e-5 |
| 138 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 274 | -2.79e-4 | -2.79e-4 | -2.79e-4 | +2.32e-5 |
| 139 | 3.00e-2 | 1 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 4.92e-2 | 275 | +1.87e-5 | +1.87e-5 | +1.87e-5 | +2.27e-5 |
| 140 | 3.00e-2 | 1 | 5.21e-2 | 5.21e-2 | 5.21e-2 | 5.21e-2 | 290 | +1.94e-4 | +1.94e-4 | +1.94e-4 | +3.98e-5 |
| 141 | 3.00e-2 | 1 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 5.36e-2 | 273 | +1.07e-4 | +1.07e-4 | +1.07e-4 | +4.65e-5 |
| 142 | 3.00e-2 | 1 | 5.34e-2 | 5.34e-2 | 5.34e-2 | 5.34e-2 | 274 | -1.33e-5 | -1.33e-5 | -1.33e-5 | +4.05e-5 |
| 143 | 3.00e-2 | 1 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 292 | +1.91e-4 | +1.91e-4 | +1.91e-4 | +5.56e-5 |
| 144 | 3.00e-2 | 1 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 5.65e-2 | 308 | +1.63e-6 | +1.63e-6 | +1.63e-6 | +5.02e-5 |
| 145 | 3.00e-2 | 1 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 5.90e-2 | 300 | +1.45e-4 | +1.45e-4 | +1.45e-4 | +5.97e-5 |
| 146 | 3.00e-2 | 1 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 5.92e-2 | 251 | +7.64e-6 | +7.64e-6 | +7.64e-6 | +5.45e-5 |
| 147 | 3.00e-2 | 1 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 5.53e-2 | 270 | -2.53e-4 | -2.53e-4 | -2.53e-4 | +2.38e-5 |
| 148 | 3.00e-2 | 1 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 5.70e-2 | 270 | +1.15e-4 | +1.15e-4 | +1.15e-4 | +3.29e-5 |
| 149 | 3.00e-2 | 1 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 5.84e-2 | 267 | +9.29e-5 | +9.29e-5 | +9.29e-5 | +3.89e-5 |
| 150 | 3.00e-3 | 1 | 5.77e-2 | 5.77e-2 | 5.77e-2 | 5.77e-2 | 249 | -5.12e-5 | -5.12e-5 | -5.12e-5 | +2.99e-5 |
| 151 | 3.00e-3 | 1 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 5.74e-2 | 251 | -1.67e-5 | -1.67e-5 | -1.67e-5 | +2.52e-5 |
| 152 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 256 | -9.09e-3 | -9.09e-3 | -9.09e-3 | -8.86e-4 |
| 153 | 3.00e-3 | 1 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 5.51e-3 | 268 | -6.61e-5 | -6.61e-5 | -6.61e-5 | -8.04e-4 |
| 154 | 3.00e-3 | 1 | 5.41e-3 | 5.41e-3 | 5.41e-3 | 5.41e-3 | 340 | -5.23e-5 | -5.23e-5 | -5.23e-5 | -7.29e-4 |
| 155 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 252 | +4.72e-4 | +4.72e-4 | +4.72e-4 | -6.09e-4 |
| 156 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 260 | -4.21e-4 | -4.21e-4 | -4.21e-4 | -5.90e-4 |
| 157 | 3.00e-3 | 1 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 5.56e-3 | 260 | +7.20e-5 | +7.20e-5 | +7.20e-5 | -5.24e-4 |
| 158 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 286 | +1.62e-5 | +1.62e-5 | +1.62e-5 | -4.70e-4 |
| 159 | 3.00e-3 | 1 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 273 | +1.11e-4 | +1.11e-4 | +1.11e-4 | -4.12e-4 |
| 160 | 3.00e-3 | 1 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 5.66e-3 | 308 | -5.95e-5 | -5.95e-5 | -5.95e-5 | -3.77e-4 |
| 161 | 3.00e-3 | 1 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 273 | +9.88e-5 | +9.88e-5 | +9.88e-5 | -3.29e-4 |
| 162 | 3.00e-3 | 1 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 5.61e-3 | 284 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -3.08e-4 |
| 163 | 3.00e-3 | 1 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 5.89e-3 | 245 | +1.92e-4 | +1.92e-4 | +1.92e-4 | -2.58e-4 |
| 164 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 257 | -2.92e-4 | -2.92e-4 | -2.92e-4 | -2.62e-4 |
| 165 | 3.00e-3 | 1 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 5.60e-3 | 263 | +9.46e-5 | +9.46e-5 | +9.46e-5 | -2.26e-4 |
| 166 | 3.00e-3 | 2 | 5.68e-3 | 5.91e-3 | 5.79e-3 | 5.91e-3 | 223 | +5.11e-5 | +1.77e-4 | +1.14e-4 | -1.61e-4 |
| 168 | 3.00e-3 | 2 | 5.19e-3 | 5.99e-3 | 5.59e-3 | 5.99e-3 | 223 | -4.60e-4 | +6.43e-4 | +9.15e-5 | -1.07e-4 |
| 169 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 260 | -3.53e-4 | -3.53e-4 | -3.53e-4 | -1.32e-4 |
| 170 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 244 | +9.33e-5 | +9.33e-5 | +9.33e-5 | -1.09e-4 |
| 171 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 269 | +2.58e-6 | +2.58e-6 | +2.58e-6 | -9.82e-5 |
| 172 | 3.00e-3 | 1 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 5.99e-3 | 259 | +2.66e-4 | +2.66e-4 | +2.66e-4 | -6.17e-5 |
| 173 | 3.00e-3 | 1 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 272 | -1.10e-4 | -1.10e-4 | -1.10e-4 | -6.66e-5 |
| 174 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 246 | +6.63e-5 | +6.63e-5 | +6.63e-5 | -5.33e-5 |
| 175 | 3.00e-3 | 1 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 5.70e-3 | 248 | -1.45e-4 | -1.45e-4 | -1.45e-4 | -6.24e-5 |
| 176 | 3.00e-3 | 1 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 5.76e-3 | 299 | +3.57e-5 | +3.57e-5 | +3.57e-5 | -5.26e-5 |
| 177 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 264 | +2.75e-4 | +2.75e-4 | +2.75e-4 | -1.99e-5 |
| 178 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 296 | -7.78e-5 | -7.78e-5 | -7.78e-5 | -2.57e-5 |
| 179 | 3.00e-3 | 1 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 6.27e-3 | 293 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -1.14e-5 |
| 180 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 244 | +1.52e-5 | +1.52e-5 | +1.52e-5 | -8.69e-6 |
| 181 | 3.00e-3 | 1 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 5.77e-3 | 245 | -3.54e-4 | -3.54e-4 | -3.54e-4 | -4.32e-5 |
| 182 | 3.00e-3 | 1 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 5.87e-3 | 251 | +7.14e-5 | +7.14e-5 | +7.14e-5 | -3.17e-5 |
| 183 | 3.00e-3 | 1 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 5.94e-3 | 260 | +4.13e-5 | +4.13e-5 | +4.13e-5 | -2.44e-5 |
| 184 | 3.00e-3 | 1 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 6.03e-3 | 277 | +5.44e-5 | +5.44e-5 | +5.44e-5 | -1.65e-5 |
| 185 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 315 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -3.06e-6 |
| 186 | 3.00e-3 | 1 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 6.45e-3 | 282 | +1.07e-4 | +1.07e-4 | +1.07e-4 | +7.91e-6 |
| 187 | 3.00e-3 | 1 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 6.33e-3 | 293 | -6.06e-5 | -6.06e-5 | -6.06e-5 | +1.06e-6 |
| 188 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 266 | +6.30e-5 | +6.30e-5 | +6.30e-5 | +7.25e-6 |
| 189 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 264 | -1.82e-4 | -1.82e-4 | -1.82e-4 | -1.16e-5 |
| 190 | 3.00e-3 | 1 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 6.19e-3 | 281 | +2.95e-5 | +2.95e-5 | +2.95e-5 | -7.52e-6 |
| 191 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 288 | +3.61e-5 | +3.61e-5 | +3.61e-5 | -3.15e-6 |
| 192 | 3.00e-3 | 1 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 6.26e-3 | 284 | +3.91e-6 | +3.91e-6 | +3.91e-6 | -2.45e-6 |
| 193 | 3.00e-3 | 1 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 6.28e-3 | 247 | +1.47e-5 | +1.47e-5 | +1.47e-5 | -7.33e-7 |
| 194 | 3.00e-3 | 1 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 6.00e-3 | 262 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -1.85e-5 |
| 195 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 262 | +9.04e-6 | +9.04e-6 | +9.04e-6 | -1.58e-5 |
| 196 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 247 | +1.58e-4 | +1.58e-4 | +1.58e-4 | +1.64e-6 |
| 197 | 3.00e-3 | 1 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 5.98e-3 | 260 | -1.72e-4 | -1.72e-4 | -1.72e-4 | -1.57e-5 |
| 198 | 3.00e-3 | 1 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 6.10e-3 | 270 | +7.56e-5 | +7.56e-5 | +7.56e-5 | -6.55e-6 |
| 199 | 3.00e-3 | 1 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 6.34e-3 | 296 | +1.31e-4 | +1.31e-4 | +1.31e-4 | +7.24e-6 |

