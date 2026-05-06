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
| nccl-async | 0.054631 | 0.9182 | +0.0057 | 1946.1 | 408 | 40.2 | 100% | 100% | 6.1 |

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
| nccl-async | 1.9812 | 0.6826 | 0.5479 | 0.5731 | 0.5232 | 0.5036 | 0.4948 | 0.4857 | 0.4737 | 0.4577 | 0.2073 | 0.1701 | 0.1487 | 0.1349 | 0.1295 | 0.0709 | 0.0636 | 0.0605 | 0.0560 | 0.0546 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.3994 | 2.5 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3064 | 3.3 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2942 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 392 | 390 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1945.1 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu2 | 1945.1 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1945.1 | 0.6 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.6s | 0.0s | 0.0s | 0.0s | 2.2s |
| resnet-graph | nccl-async | gpu1 | 1.0s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.7s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 106 | 0 | 408 | 40.2 | 1340/11595 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 202.8 | 10.4% |

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
| resnet-graph | nccl-async | 176 | 408 | 0 | 6.57e-3 | +2.04e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 408 | 9.84e-2 | 6.13e-2 | 0.00e0 | 3.72e-1 | 32.1 | -9.91e-5 | 1.57e-3 |
| resnet-graph | nccl-async | 1 | 408 | 1.00e-1 | 6.33e-2 | 0.00e0 | 3.51e-1 | 48.3 | -1.00e-4 | 2.49e-3 |
| resnet-graph | nccl-async | 2 | 408 | 9.90e-2 | 6.33e-2 | 0.00e0 | 3.88e-1 | 19.6 | -1.11e-4 | 2.43e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9925 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9934 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9982 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 41 (0,1,2,4,5,6,7,8…141,149) | 0 (—) | — | 0,1,2,4,5,6,7,8…141,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 29 | 29 |
| resnet-graph | nccl-async | 0e0 | 5 | 15 | 15 |
| resnet-graph | nccl-async | 0e0 | 10 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 3 | 5 | 5 |
| resnet-graph | nccl-async | 1e-4 | 5 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 311 | +0.038 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 44 | +0.170 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 48 | +0.069 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 405 | +0.004 | 175 | +0.239 | +0.432 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 406 | 3.33e1–8.12e1 | 6.55e1 | 3.08e-3 | 4.23e-3 | 4.48e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 313 | 74–77752 | +1.475e-5 | 0.552 | +1.520e-5 | 0.571 | 88 | +1.427e-5 | 0.686 | 42–1108 | +1.042e-3 | 0.738 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 302 | 897–77752 | +1.498e-5 | 0.606 | +1.541e-5 | 0.624 | 87 | +1.425e-5 | 0.679 | 58–1108 | +1.044e-3 | 0.810 |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | 45 | 78656–115820 | +9.371e-6 | 0.057 | +9.410e-6 | 0.057 | 42 | +7.028e-6 | 0.030 | 679–966 | -7.187e-4 | 0.013 |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | 49 | 116596–155668 | -1.626e-5 | 0.122 | -1.643e-5 | 0.124 | 46 | -1.122e-5 | 0.075 | 699–982 | -3.573e-5 | 0.000 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.042e-3 | r0: +1.028e-3, r1: +1.045e-3, r2: +1.056e-3 | r0: 0.782, r1: 0.712, r2: 0.710 | 1.03× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.044e-3 | r0: +1.033e-3, r1: +1.046e-3, r2: +1.057e-3 | r0: 0.864, r1: 0.773, r2: 0.780 | 1.02× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–148 | -7.187e-4 | r0: -6.944e-4, r1: -7.440e-4, r2: -7.181e-4 | r0: 0.013, r1: 0.015, r2: 0.013 | 1.07× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 149–199 | -3.573e-5 | r0: +2.650e-5, r1: -5.683e-5, r2: -7.623e-5 | r0: 0.000, r1: 0.000, r2: 0.000 | 2.88× | ⚠ framing breaking |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇▇▇▇▇▇▇▆███████████████████▄▄▄▄▅▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` | `▁▇▇▇▇▇█▇▇▇▇███████████████████▆▇▇▇▇█████████▆▆▇▇▇▇▇▇▇▇█████` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 0.00e0 | 3.88e-1 | 1.09e-1 | 7.23e-2 | 28 | -2.80e-2 | +5.70e-3 | -8.89e-3 | -5.99e-3 |
| 1 | 3.00e-1 | 9 | 6.95e-2 | 1.15e-1 | 8.44e-2 | 8.35e-2 | 28 | -1.77e-2 | +1.47e-2 | -1.67e-5 | -1.84e-3 |
| 2 | 3.00e-1 | 12 | 7.79e-2 | 1.30e-1 | 9.67e-2 | 9.50e-2 | 27 | -1.50e-2 | +1.91e-2 | +6.04e-4 | -2.71e-4 |
| 3 | 3.00e-1 | 5 | 9.31e-2 | 1.41e-1 | 1.06e-1 | 1.03e-1 | 40 | -1.30e-2 | +1.43e-2 | +4.63e-4 | -5.06e-5 |
| 4 | 3.00e-1 | 8 | 9.50e-2 | 1.42e-1 | 1.10e-1 | 1.06e-1 | 34 | -8.37e-3 | +4.44e-3 | -4.80e-4 | -2.73e-4 |
| 5 | 3.00e-1 | 10 | 9.46e-2 | 1.46e-1 | 1.08e-1 | 9.69e-2 | 32 | -9.59e-3 | +8.63e-3 | -3.60e-4 | -4.08e-4 |
| 6 | 3.00e-1 | 5 | 1.02e-1 | 1.45e-1 | 1.13e-1 | 1.09e-1 | 26 | -1.10e-2 | +9.95e-3 | +3.76e-4 | -1.54e-4 |
| 7 | 3.00e-1 | 12 | 7.64e-2 | 1.52e-1 | 9.62e-2 | 1.03e-1 | 35 | -1.94e-2 | +1.63e-2 | -3.26e-4 | +3.41e-6 |
| 8 | 3.00e-1 | 6 | 9.31e-2 | 1.52e-1 | 1.08e-1 | 9.90e-2 | 26 | -1.44e-2 | +1.10e-2 | -4.45e-4 | -2.42e-4 |
| 9 | 3.00e-1 | 7 | 8.25e-2 | 1.50e-1 | 9.67e-2 | 9.26e-2 | 32 | -2.29e-2 | +2.13e-2 | -1.67e-5 | -1.71e-4 |
| 10 | 3.00e-1 | 11 | 8.77e-2 | 1.46e-1 | 9.75e-2 | 8.77e-2 | 29 | -1.69e-2 | +1.23e-2 | -3.30e-4 | -3.77e-4 |
| 11 | 3.00e-1 | 7 | 8.32e-2 | 1.38e-1 | 9.47e-2 | 8.32e-2 | 27 | -1.91e-2 | +1.49e-2 | -5.10e-4 | -5.79e-4 |
| 12 | 3.00e-1 | 8 | 8.29e-2 | 1.48e-1 | 9.86e-2 | 9.90e-2 | 33 | -2.07e-2 | +2.11e-2 | +8.32e-4 | +1.15e-4 |
| 13 | 3.00e-1 | 9 | 8.08e-2 | 1.35e-1 | 9.35e-2 | 8.87e-2 | 26 | -2.11e-2 | +1.41e-2 | -5.01e-4 | -2.52e-4 |
| 14 | 3.00e-1 | 11 | 7.90e-2 | 1.51e-1 | 9.46e-2 | 9.59e-2 | 29 | -1.99e-2 | +1.88e-2 | -1.91e-4 | -1.27e-4 |
| 15 | 3.00e-1 | 6 | 8.27e-2 | 1.39e-1 | 9.54e-2 | 8.67e-2 | 40 | -1.90e-2 | +1.62e-2 | -2.84e-4 | -2.71e-4 |
| 16 | 3.00e-1 | 6 | 9.83e-2 | 1.52e-1 | 1.12e-1 | 9.83e-2 | 39 | -9.21e-3 | +8.29e-3 | -1.77e-5 | -2.95e-4 |
| 17 | 3.00e-1 | 8 | 9.40e-2 | 1.41e-1 | 1.08e-1 | 1.10e-1 | 43 | -1.11e-2 | +8.05e-3 | +2.65e-4 | +1.54e-5 |
| 18 | 3.00e-1 | 5 | 1.09e-1 | 1.43e-1 | 1.17e-1 | 1.15e-1 | 38 | -6.32e-3 | +6.42e-3 | +2.86e-4 | +1.03e-4 |
| 19 | 3.00e-1 | 9 | 9.16e-2 | 1.48e-1 | 1.08e-1 | 1.08e-1 | 41 | -1.13e-2 | +9.44e-3 | -8.91e-5 | +3.28e-5 |
| 20 | 3.00e-1 | 4 | 9.63e-2 | 1.47e-1 | 1.14e-1 | 1.06e-1 | 37 | -1.21e-2 | +9.62e-3 | -7.54e-5 | -6.12e-5 |
| 21 | 3.00e-1 | 7 | 8.80e-2 | 1.48e-1 | 1.05e-1 | 8.80e-2 | 30 | -1.08e-2 | +1.01e-2 | -7.55e-4 | -5.77e-4 |
| 22 | 3.00e-1 | 9 | 8.93e-2 | 1.47e-1 | 1.01e-1 | 1.01e-1 | 32 | -1.54e-2 | +1.60e-2 | +4.51e-4 | +5.90e-6 |
| 23 | 3.00e-1 | 6 | 8.68e-2 | 1.52e-1 | 1.03e-1 | 9.36e-2 | 31 | -1.65e-2 | +1.63e-2 | +1.60e-4 | -2.97e-5 |
| 24 | 3.00e-1 | 7 | 8.90e-2 | 1.38e-1 | 1.04e-1 | 1.01e-1 | 41 | -1.01e-2 | +1.28e-2 | +4.36e-4 | +1.14e-4 |
| 25 | 3.00e-1 | 7 | 9.67e-2 | 1.42e-1 | 1.08e-1 | 1.01e-1 | 35 | -1.01e-2 | +6.27e-3 | -2.16e-4 | -8.56e-5 |
| 26 | 3.00e-1 | 8 | 8.93e-2 | 1.44e-1 | 1.00e-1 | 8.97e-2 | 31 | -1.24e-2 | +9.83e-3 | -6.53e-4 | -4.45e-4 |
| 27 | 3.00e-1 | 10 | 8.96e-2 | 1.40e-1 | 1.01e-1 | 9.96e-2 | 32 | -1.24e-2 | +1.19e-2 | +2.18e-4 | -7.05e-5 |
| 28 | 3.00e-1 | 6 | 8.19e-2 | 1.49e-1 | 9.71e-2 | 8.47e-2 | 26 | -2.07e-2 | +1.67e-2 | -5.49e-4 | -3.90e-4 |
| 29 | 3.00e-1 | 9 | 8.16e-2 | 1.37e-1 | 9.53e-2 | 8.16e-2 | 25 | -1.68e-2 | +1.64e-2 | -1.76e-4 | -5.88e-4 |
| 30 | 3.00e-1 | 10 | 7.34e-2 | 1.35e-1 | 8.59e-2 | 7.84e-2 | 22 | -2.77e-2 | +2.22e-2 | -3.29e-4 | -5.48e-4 |
| 31 | 3.00e-1 | 2 | 8.31e-2 | 8.60e-2 | 8.46e-2 | 8.60e-2 | 28 | +1.20e-3 | +2.18e-3 | +1.69e-3 | -1.28e-4 |
| 32 | 3.00e-1 | 1 | 8.92e-2 | 8.92e-2 | 8.92e-2 | 8.92e-2 | 305 | +1.23e-4 | +1.23e-4 | +1.23e-4 | -1.03e-4 |
| 33 | 3.00e-1 | 2 | 2.14e-1 | 2.44e-1 | 2.29e-1 | 2.14e-1 | 237 | -5.63e-4 | +3.80e-3 | +1.62e-3 | +2.02e-4 |
| 34 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 275 | -1.34e-4 | -1.34e-4 | -1.34e-4 | +1.69e-4 |
| 35 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 261 | +1.01e-4 | +1.01e-4 | +1.01e-4 | +1.62e-4 |
| 37 | 3.00e-1 | 2 | 2.08e-1 | 2.19e-1 | 2.13e-1 | 2.19e-1 | 270 | -5.95e-5 | +1.92e-4 | +6.62e-5 | +1.45e-4 |
| 39 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 369 | -1.67e-4 | -1.67e-4 | -1.67e-4 | +1.14e-4 |
| 40 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 324 | +2.77e-4 | +2.77e-4 | +2.77e-4 | +1.30e-4 |
| 41 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 297 | -4.40e-5 | -4.40e-5 | -4.40e-5 | +1.13e-4 |
| 42 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 316 | -1.01e-4 | -1.01e-4 | -1.01e-4 | +9.13e-5 |
| 43 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 300 | +5.82e-5 | +5.82e-5 | +5.82e-5 | +8.80e-5 |
| 44 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 296 | -4.75e-5 | -4.75e-5 | -4.75e-5 | +7.44e-5 |
| 46 | 3.00e-1 | 2 | 2.13e-1 | 2.22e-1 | 2.18e-1 | 2.22e-1 | 270 | -3.77e-5 | +1.56e-4 | +5.90e-5 | +7.25e-5 |
| 48 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 387 | -1.62e-4 | -1.62e-4 | -1.62e-4 | +4.90e-5 |
| 49 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 309 | +2.75e-4 | +2.75e-4 | +2.75e-4 | +7.16e-5 |
| 50 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 302 | -2.12e-4 | -2.12e-4 | -2.12e-4 | +4.33e-5 |
| 51 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 269 | +2.35e-5 | +2.35e-5 | +2.35e-5 | +4.13e-5 |
| 52 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 276 | -4.96e-5 | -4.96e-5 | -4.96e-5 | +3.22e-5 |
| 53 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 276 | +9.73e-6 | +9.73e-6 | +9.73e-6 | +3.00e-5 |
| 54 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 272 | -1.74e-6 | -1.74e-6 | -1.74e-6 | +2.68e-5 |
| 55 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 312 | -5.35e-5 | -5.35e-5 | -5.35e-5 | +1.88e-5 |
| 56 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 339 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +3.05e-5 |
| 57 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 292 | +1.00e-4 | +1.00e-4 | +1.00e-4 | +3.74e-5 |
| 59 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 353 | -1.37e-4 | -1.37e-4 | -1.37e-4 | +2.00e-5 |
| 60 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 274 | +2.30e-4 | +2.30e-4 | +2.30e-4 | +4.10e-5 |
| 61 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 265 | -3.16e-4 | -3.16e-4 | -3.16e-4 | +5.35e-6 |
| 62 | 3.00e-1 | 2 | 2.08e-1 | 2.10e-1 | 2.09e-1 | 2.10e-1 | 248 | -3.32e-5 | +3.35e-5 | +1.26e-7 | +4.69e-6 |
| 64 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 352 | -9.21e-5 | -9.21e-5 | -9.21e-5 | -4.99e-6 |
| 65 | 3.00e-1 | 2 | 2.11e-1 | 2.27e-1 | 2.19e-1 | 2.11e-1 | 273 | -2.66e-4 | +4.14e-4 | +7.36e-5 | +6.55e-6 |
| 67 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 330 | +6.42e-5 | +6.42e-5 | +6.42e-5 | +1.23e-5 |
| 68 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 297 | +1.45e-4 | +1.45e-4 | +1.45e-4 | +2.56e-5 |
| 69 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 320 | -1.70e-4 | -1.70e-4 | -1.70e-4 | +5.97e-6 |
| 70 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 309 | +1.39e-4 | +1.39e-4 | +1.39e-4 | +1.93e-5 |
| 71 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 304 | -4.62e-5 | -4.62e-5 | -4.62e-5 | +1.28e-5 |
| 72 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 290 | +2.96e-6 | +2.96e-6 | +2.96e-6 | +1.18e-5 |
| 73 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 298 | -4.18e-5 | -4.18e-5 | -4.18e-5 | +6.43e-6 |
| 75 | 3.00e-1 | 2 | 2.16e-1 | 2.24e-1 | 2.20e-1 | 2.24e-1 | 272 | -5.62e-6 | +1.30e-4 | +6.24e-5 | +1.77e-5 |
| 77 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 363 | -1.60e-4 | -1.60e-4 | -1.60e-4 | -4.36e-9 |
| 78 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 293 | +3.25e-4 | +3.25e-4 | +3.25e-4 | +3.25e-5 |
| 79 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 306 | -2.73e-4 | -2.73e-4 | -2.73e-4 | +1.90e-6 |
| 80 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 294 | +1.58e-4 | +1.58e-4 | +1.58e-4 | +1.75e-5 |
| 81 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 300 | -4.12e-5 | -4.12e-5 | -4.12e-5 | +1.16e-5 |
| 82 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 309 | -3.06e-5 | -3.06e-5 | -3.06e-5 | +7.41e-6 |
| 83 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 288 | +5.81e-5 | +5.81e-5 | +5.81e-5 | +1.25e-5 |
| 85 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 327 | -7.12e-5 | -7.12e-5 | -7.12e-5 | +4.11e-6 |
| 86 | 3.00e-1 | 2 | 2.16e-1 | 2.28e-1 | 2.22e-1 | 2.16e-1 | 267 | -2.03e-4 | +1.45e-4 | -2.89e-5 | -3.90e-6 |
| 88 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 346 | -9.53e-5 | -9.53e-5 | -9.53e-5 | -1.30e-5 |
| 89 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 278 | +3.11e-4 | +3.11e-4 | +3.11e-4 | +1.94e-5 |
| 90 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 277 | -2.21e-4 | -2.21e-4 | -2.21e-4 | -4.63e-6 |
| 91 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 305 | +6.15e-5 | +6.15e-5 | +6.15e-5 | +1.98e-6 |
| 92 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 339 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +1.24e-5 |
| 94 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 419 | -4.44e-6 | -4.44e-6 | -4.44e-6 | +1.07e-5 |
| 95 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 317 | +1.65e-4 | +1.65e-4 | +1.65e-4 | +2.61e-5 |
| 96 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 316 | -2.16e-4 | -2.16e-4 | -2.16e-4 | +1.86e-6 |
| 97 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 262 | +3.20e-6 | +3.20e-6 | +3.20e-6 | +1.99e-6 |
| 98 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 309 | -1.70e-4 | -1.70e-4 | -1.70e-4 | -1.52e-5 |
| 99 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 291 | +1.70e-4 | +1.70e-4 | +1.70e-4 | +3.30e-6 |
| 100 | 3.00e-2 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 318 | -8.52e-5 | -8.52e-5 | -8.52e-5 | -5.55e-6 |
| 101 | 3.00e-2 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 320 | +1.28e-4 | +1.28e-4 | +1.28e-4 | +7.81e-6 |
| 103 | 3.00e-2 | 2 | 2.35e-2 | 2.52e-2 | 2.43e-2 | 2.52e-2 | 278 | -6.19e-3 | +2.53e-4 | -2.97e-3 | -5.25e-4 |
| 105 | 3.00e-2 | 1 | 2.40e-2 | 2.40e-2 | 2.40e-2 | 2.40e-2 | 341 | -1.46e-4 | -1.46e-4 | -1.46e-4 | -4.87e-4 |
| 106 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 309 | +4.83e-4 | +4.83e-4 | +4.83e-4 | -3.90e-4 |
| 107 | 3.00e-2 | 1 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 2.72e-2 | 330 | -7.51e-5 | -7.51e-5 | -7.51e-5 | -3.59e-4 |
| 108 | 3.00e-2 | 1 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 2.89e-2 | 310 | +1.97e-4 | +1.97e-4 | +1.97e-4 | -3.03e-4 |
| 109 | 3.00e-2 | 1 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 2.94e-2 | 320 | +5.63e-5 | +5.63e-5 | +5.63e-5 | -2.67e-4 |
| 111 | 3.00e-2 | 2 | 3.01e-2 | 3.26e-2 | 3.13e-2 | 3.26e-2 | 262 | +6.51e-5 | +3.10e-4 | +1.88e-4 | -1.79e-4 |
| 113 | 3.00e-2 | 1 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 2.96e-2 | 363 | -2.65e-4 | -2.65e-4 | -2.65e-4 | -1.88e-4 |
| 114 | 3.00e-2 | 1 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 3.47e-2 | 329 | +4.80e-4 | +4.80e-4 | +4.80e-4 | -1.21e-4 |
| 115 | 3.00e-2 | 1 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 3.44e-2 | 334 | -2.12e-5 | -2.12e-5 | -2.12e-5 | -1.11e-4 |
| 116 | 3.00e-2 | 1 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 300 | +1.04e-4 | +1.04e-4 | +1.04e-4 | -8.97e-5 |
| 117 | 3.00e-2 | 1 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 3.49e-2 | 280 | -6.82e-5 | -6.82e-5 | -6.82e-5 | -8.75e-5 |
| 118 | 3.00e-2 | 1 | 3.40e-2 | 3.40e-2 | 3.40e-2 | 3.40e-2 | 304 | -8.14e-5 | -8.14e-5 | -8.14e-5 | -8.69e-5 |
| 119 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 342 | +2.03e-4 | +2.03e-4 | +2.03e-4 | -5.80e-5 |
| 120 | 3.00e-2 | 1 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 282 | +2.75e-4 | +2.75e-4 | +2.75e-4 | -2.47e-5 |
| 122 | 3.00e-2 | 1 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 3.75e-2 | 354 | -1.36e-4 | -1.36e-4 | -1.36e-4 | -3.58e-5 |
| 123 | 3.00e-2 | 1 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 4.18e-2 | 290 | +3.71e-4 | +3.71e-4 | +3.71e-4 | +4.92e-6 |
| 124 | 3.00e-2 | 1 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 3.94e-2 | 267 | -2.18e-4 | -2.18e-4 | -2.18e-4 | -1.74e-5 |
| 125 | 3.00e-2 | 1 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 3.96e-2 | 311 | +1.14e-5 | +1.14e-5 | +1.14e-5 | -1.45e-5 |
| 126 | 3.00e-2 | 1 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 322 | +1.75e-4 | +1.75e-4 | +1.75e-4 | +4.49e-6 |
| 127 | 3.00e-2 | 1 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 4.40e-2 | 328 | +1.51e-4 | +1.51e-4 | +1.51e-4 | +1.91e-5 |
| 128 | 3.00e-2 | 1 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 4.49e-2 | 280 | +7.59e-5 | +7.59e-5 | +7.59e-5 | +2.48e-5 |
| 129 | 3.00e-2 | 1 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 4.36e-2 | 289 | -1.03e-4 | -1.03e-4 | -1.03e-4 | +1.20e-5 |
| 130 | 3.00e-2 | 1 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 4.46e-2 | 282 | +7.66e-5 | +7.66e-5 | +7.66e-5 | +1.84e-5 |
| 131 | 3.00e-2 | 1 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 4.45e-2 | 300 | -4.49e-6 | -4.49e-6 | -4.49e-6 | +1.61e-5 |
| 132 | 3.00e-2 | 1 | 4.67e-2 | 4.67e-2 | 4.67e-2 | 4.67e-2 | 267 | +1.80e-4 | +1.80e-4 | +1.80e-4 | +3.25e-5 |
| 133 | 3.00e-2 | 1 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 4.64e-2 | 282 | -2.28e-5 | -2.28e-5 | -2.28e-5 | +2.70e-5 |
| 134 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 278 | +1.41e-4 | +1.41e-4 | +1.41e-4 | +3.84e-5 |
| 135 | 3.00e-2 | 1 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 4.73e-2 | 283 | -6.95e-5 | -6.95e-5 | -6.95e-5 | +2.76e-5 |
| 136 | 3.00e-2 | 1 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 4.90e-2 | 291 | +1.16e-4 | +1.16e-4 | +1.16e-4 | +3.65e-5 |
| 137 | 3.00e-2 | 1 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 5.04e-2 | 276 | +1.06e-4 | +1.06e-4 | +1.06e-4 | +4.35e-5 |
| 138 | 3.00e-2 | 1 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 5.10e-2 | 353 | +3.48e-5 | +3.48e-5 | +3.48e-5 | +4.26e-5 |
| 140 | 3.00e-2 | 1 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 5.61e-2 | 333 | +2.87e-4 | +2.87e-4 | +2.87e-4 | +6.70e-5 |
| 141 | 3.00e-2 | 2 | 5.55e-2 | 5.67e-2 | 5.61e-2 | 5.55e-2 | 240 | -8.32e-5 | +2.74e-5 | -2.79e-5 | +4.84e-5 |
| 143 | 3.00e-2 | 1 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 4.98e-2 | 313 | -3.48e-4 | -3.48e-4 | -3.48e-4 | +8.79e-6 |
| 144 | 3.00e-2 | 1 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 5.80e-2 | 278 | +5.50e-4 | +5.50e-4 | +5.50e-4 | +6.29e-5 |
| 145 | 3.00e-2 | 1 | 5.56e-2 | 5.56e-2 | 5.56e-2 | 5.56e-2 | 269 | -1.61e-4 | -1.61e-4 | -1.61e-4 | +4.05e-5 |
| 146 | 3.00e-2 | 1 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 5.47e-2 | 271 | -6.07e-5 | -6.07e-5 | -6.07e-5 | +3.04e-5 |
| 147 | 3.00e-2 | 1 | 5.75e-2 | 5.75e-2 | 5.75e-2 | 5.75e-2 | 263 | +1.91e-4 | +1.91e-4 | +1.91e-4 | +4.65e-5 |
| 148 | 3.00e-2 | 1 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 5.59e-2 | 277 | -1.03e-4 | -1.03e-4 | -1.03e-4 | +3.16e-5 |
| 149 | 3.00e-3 | 2 | 5.78e-2 | 5.85e-2 | 5.81e-2 | 5.85e-2 | 254 | +4.94e-5 | +1.20e-4 | +8.47e-5 | +4.13e-5 |
| 151 | 3.00e-3 | 1 | 5.76e-2 | 5.76e-2 | 5.76e-2 | 5.76e-2 | 343 | -4.31e-5 | -4.31e-5 | -4.31e-5 | +3.29e-5 |
| 152 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 279 | -7.99e-3 | -7.99e-3 | -7.99e-3 | -7.69e-4 |
| 153 | 3.00e-3 | 1 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 5.75e-3 | 261 | -2.96e-4 | -2.96e-4 | -2.96e-4 | -7.22e-4 |
| 154 | 3.00e-3 | 1 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 5.46e-3 | 271 | -1.89e-4 | -1.89e-4 | -1.89e-4 | -6.69e-4 |
| 155 | 3.00e-3 | 1 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 5.59e-3 | 289 | +8.34e-5 | +8.34e-5 | +8.34e-5 | -5.93e-4 |
| 156 | 3.00e-3 | 1 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 5.71e-3 | 280 | +7.40e-5 | +7.40e-5 | +7.40e-5 | -5.27e-4 |
| 157 | 3.00e-3 | 1 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 5.58e-3 | 302 | -7.29e-5 | -7.29e-5 | -7.29e-5 | -4.81e-4 |
| 158 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 332 | +1.80e-4 | +1.80e-4 | +1.80e-4 | -4.15e-4 |
| 159 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 308 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -3.61e-4 |
| 160 | 3.00e-3 | 1 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 6.06e-3 | 330 | -4.53e-5 | -4.53e-5 | -4.53e-5 | -3.30e-4 |
| 161 | 3.00e-3 | 1 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 6.09e-3 | 266 | +1.95e-5 | +1.95e-5 | +1.95e-5 | -2.95e-4 |
| 162 | 3.00e-3 | 1 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 5.83e-3 | 270 | -1.67e-4 | -1.67e-4 | -1.67e-4 | -2.82e-4 |
| 163 | 3.00e-3 | 1 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 5.81e-3 | 294 | -1.20e-5 | -1.20e-5 | -1.20e-5 | -2.55e-4 |
| 164 | 3.00e-3 | 1 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 5.88e-3 | 333 | +3.72e-5 | +3.72e-5 | +3.72e-5 | -2.26e-4 |
| 166 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 338 | +1.64e-4 | +1.64e-4 | +1.64e-4 | -1.87e-4 |
| 167 | 3.00e-3 | 1 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 6.25e-3 | 277 | +2.16e-5 | +2.16e-5 | +2.16e-5 | -1.66e-4 |
| 168 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 270 | -1.92e-4 | -1.92e-4 | -1.92e-4 | -1.69e-4 |
| 169 | 3.00e-3 | 1 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 5.93e-3 | 288 | -2.88e-6 | -2.88e-6 | -2.88e-6 | -1.52e-4 |
| 170 | 3.00e-3 | 1 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 5.91e-3 | 305 | -1.15e-5 | -1.15e-5 | -1.15e-5 | -1.38e-4 |
| 171 | 3.00e-3 | 1 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 6.35e-3 | 290 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -9.95e-5 |
| 172 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 296 | -1.04e-4 | -1.04e-4 | -1.04e-4 | -1.00e-4 |
| 173 | 3.00e-3 | 1 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 6.20e-3 | 274 | +2.68e-5 | +2.68e-5 | +2.68e-5 | -8.73e-5 |
| 174 | 3.00e-3 | 1 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 6.01e-3 | 295 | -1.07e-4 | -1.07e-4 | -1.07e-4 | -8.93e-5 |
| 175 | 3.00e-3 | 1 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 6.36e-3 | 296 | +1.95e-4 | +1.95e-4 | +1.95e-4 | -6.09e-5 |
| 176 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 277 | -1.29e-4 | -1.29e-4 | -1.29e-4 | -6.77e-5 |
| 177 | 3.00e-3 | 1 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 6.14e-3 | 293 | -9.22e-7 | -9.22e-7 | -9.22e-7 | -6.10e-5 |
| 178 | 3.00e-3 | 1 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 6.21e-3 | 271 | +4.22e-5 | +4.22e-5 | +4.22e-5 | -5.07e-5 |
| 179 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 315 | -3.05e-5 | -3.05e-5 | -3.05e-5 | -4.87e-5 |
| 180 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 267 | +2.27e-4 | +2.27e-4 | +2.27e-4 | -2.11e-5 |
| 181 | 3.00e-3 | 1 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 5.96e-3 | 277 | -3.33e-4 | -3.33e-4 | -3.33e-4 | -5.23e-5 |
| 182 | 3.00e-3 | 1 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 6.15e-3 | 275 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -3.55e-5 |
| 183 | 3.00e-3 | 1 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 6.16e-3 | 280 | +5.33e-6 | +5.33e-6 | +5.33e-6 | -3.14e-5 |
| 184 | 3.00e-3 | 1 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 6.46e-3 | 309 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -1.31e-5 |
| 185 | 3.00e-3 | 1 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 6.54e-3 | 288 | +4.61e-5 | +4.61e-5 | +4.61e-5 | -7.13e-6 |
| 187 | 3.00e-3 | 1 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 6.39e-3 | 374 | -6.34e-5 | -6.34e-5 | -6.34e-5 | -1.28e-5 |
| 188 | 3.00e-3 | 1 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 7.20e-3 | 295 | +4.04e-4 | +4.04e-4 | +4.04e-4 | +2.89e-5 |
| 189 | 3.00e-3 | 1 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 6.37e-3 | 282 | -4.32e-4 | -4.32e-4 | -4.32e-4 | -1.72e-5 |
| 190 | 3.00e-3 | 1 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 6.44e-3 | 269 | +3.86e-5 | +3.86e-5 | +3.86e-5 | -1.16e-5 |
| 191 | 3.00e-3 | 2 | 6.22e-3 | 6.32e-3 | 6.27e-3 | 6.32e-3 | 267 | -1.17e-4 | +6.13e-5 | -2.78e-5 | -1.38e-5 |
| 193 | 3.00e-3 | 1 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 6.17e-3 | 306 | -8.26e-5 | -8.26e-5 | -8.26e-5 | -2.07e-5 |
| 194 | 3.00e-3 | 2 | 6.33e-3 | 6.76e-3 | 6.54e-3 | 6.33e-3 | 266 | -2.47e-4 | +3.31e-4 | +4.18e-5 | -1.17e-5 |
| 196 | 3.00e-3 | 1 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 6.29e-3 | 319 | -1.93e-5 | -1.93e-5 | -1.93e-5 | -1.25e-5 |
| 197 | 3.00e-3 | 1 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 6.74e-3 | 304 | +2.30e-4 | +2.30e-4 | +2.30e-4 | +1.18e-5 |
| 198 | 3.00e-3 | 1 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 6.53e-3 | 282 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -1.04e-6 |
| 199 | 3.00e-3 | 1 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 6.57e-3 | 240 | +2.98e-5 | +2.98e-5 | +2.98e-5 | +2.04e-6 |

