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
| nccl-async | 0.063205 | 0.9174 | +0.0049 | 1909.9 | 649 | 40.5 | 100% | 100% | 5.4 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9174 | nccl-async | - | - |

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
| nccl-async | 1.9923 | 0.6830 | 0.6015 | 0.5460 | 0.5084 | 0.4945 | 0.4738 | 0.4400 | 0.4775 | 0.4690 | 0.2137 | 0.1731 | 0.1559 | 0.1434 | 0.1380 | 0.0802 | 0.0711 | 0.0659 | 0.0627 | 0.0632 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | nccl-async | 0 | cuda:0 | 0.4025 | 2.6 |
| resnet-graph | nccl-async | 1 | cuda:1 | 0.3019 | 3.5 |
| resnet-graph | nccl-async | 2 | cuda:2 | 0.2956 | 3.3 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | nccl-async | 356 | 355 | 398 | 395 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| nccl-async | gpu1 | 1908.8 | 1.1 | epoch-boundary(199) |
| nccl-async | gpu2 | 1908.9 | 1.0 | epoch-boundary(199) |
| nccl-async | gpu0 | 1908.9 | 0.5 | epoch-boundary(199) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | nccl-async | gpu0 | 0.5s | 0.0s | 0.0s | 0.0s | 1.5s |
| resnet-graph | nccl-async | gpu1 | 1.1s | 0.0s | 0.0s | 0.0s | 2.3s |
| resnet-graph | nccl-async | gpu2 | 1.0s | 0.0s | 0.0s | 0.0s | 1.5s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | nccl-async | 395 | 0 | 649 | 40.5 | 1308/9198 | 0 | 0.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | nccl-async | 201.6 | 10.6% |

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
| resnet-graph | nccl-async | 199 | 649 | 0 | 1.42e-3 | +1.53e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | nccl-async | 0 | 649 | 7.05e-2 | 6.25e-2 | 0.00e0 | 3.60e-1 | 47.6 | -1.21e-4 | 3.15e-3 |
| resnet-graph | nccl-async | 1 | 649 | 7.08e-2 | 6.38e-2 | 0.00e0 | 3.32e-1 | 32.8 | -9.72e-5 | 4.87e-3 |
| resnet-graph | nccl-async | 2 | 649 | 7.03e-2 | 6.37e-2 | 0.00e0 | 3.45e-1 | 19.6 | -1.11e-4 | 4.88e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | nccl-async | rank0 ↔ rank1 | +0.9941 |
| resnet-graph | nccl-async | rank0 ↔ rank2 | +0.9945 |
| resnet-graph | nccl-async | rank1 ↔ rank2 | +0.9991 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | nccl-async | 44 (0,1,2,3,4,5,6,7…147,150) | 0 (—) | — | 0,1,2,3,4,5,6,7…147,150 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | nccl-async | 0e0 | 3 | 36 | 36 |
| resnet-graph | nccl-async | 0e0 | 5 | 20 | 20 |
| resnet-graph | nccl-async | 0e0 | 10 | 6 | 6 |
| resnet-graph | nccl-async | 1e-4 | 3 | 10 | 10 |
| resnet-graph | nccl-async | 1e-4 | 5 | 7 | 7 |
| resnet-graph | nccl-async | 1e-4 | 10 | 2 | 2 |
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
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 396 | +0.038 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 56 | +0.204 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 192 | -0.002 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | nccl-async | 646 | +0.002 | 198 | +0.101 | +0.119 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | nccl-async | 647 | 3.33e1–8.07e1 | 6.36e1 | 2.10e-3 | 4.00e-3 | 4.70e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | 398 | 67–77768 | +7.409e-6 | 0.171 | +7.768e-6 | 0.180 | 100 | +6.185e-6 | 0.127 | 32–917 | +1.636e-3 | 0.668 |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 382 | 929–77768 | +7.161e-6 | 0.168 | +7.489e-6 | 0.176 | 99 | +5.759e-6 | 0.111 | 33–917 | +1.624e-3 | 0.714 |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | 57 | 78544–116805 | +9.258e-6 | 0.070 | +9.107e-6 | 0.068 | 49 | +8.535e-6 | 0.054 | 468–893 | -1.113e-3 | 0.073 |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | 193 | 117332–156192 | -4.631e-5 | 0.681 | -4.770e-5 | 0.689 | 50 | -3.608e-5 | 0.560 | 38–726 | +2.761e-3 | 0.616 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | nccl-async | 3.00e-1 | 0–99 | +1.636e-3 | r0: +1.588e-3, r1: +1.660e-3, r2: +1.665e-3 | r0: 0.687, r1: 0.659, r2: 0.649 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.624e-3 | r0: +1.575e-3, r1: +1.649e-3, r2: +1.654e-3 | r0: 0.750, r1: 0.697, r2: 0.688 | 1.05× | ✓ |
| resnet-graph | nccl-async | 3.00e-2 | 100–149 | -1.113e-3 | r0: -1.090e-3, r1: -1.135e-3, r2: -1.113e-3 | r0: 0.070, r1: 0.077, r2: 0.073 | 1.04× | ✓ |
| resnet-graph | nccl-async | 3.00e-3 | 150–199 | +2.761e-3 | r0: +2.718e-3, r1: +2.797e-3, r2: +2.778e-3 | r0: 0.644, r1: 0.602, r2: 0.594 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | nccl-async | `█▇▇▇██████████████▇▇▇████▆▅▅▅▅▅▅▅▅▆▆▆▄▂▂▂▂▂▂▂▂▂▂▁▁` | `▁▇▇███▇▇▇▇▇▇▇███▇▇▆▅▇███▇▅▅▆▆▇▇▇▇▇███▄▃▅▆▇▇▇▇▇▆▄▆▇` |

### Per-Epoch Detail: resnet-graph / nccl-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 17 | 0.00e0 | 3.60e-1 | 8.67e-2 | 6.53e-2 | 20 | -5.79e-2 | +2.56e-2 | -8.80e-3 | -3.71e-3 |
| 1 | 3.00e-1 | 13 | 5.34e-2 | 1.20e-1 | 7.02e-2 | 6.64e-2 | 19 | -2.89e-2 | +3.12e-2 | +2.84e-4 | -7.47e-4 |
| 2 | 3.00e-1 | 12 | 6.93e-2 | 1.22e-1 | 8.35e-2 | 8.15e-2 | 21 | -1.38e-2 | +2.11e-2 | +5.34e-4 | +8.20e-6 |
| 3 | 3.00e-1 | 22 | 7.02e-2 | 1.41e-1 | 7.85e-2 | 7.27e-2 | 15 | -3.71e-2 | +3.12e-2 | -3.77e-4 | -3.85e-4 |
| 4 | 3.00e-1 | 10 | 7.37e-2 | 1.41e-1 | 8.20e-2 | 7.69e-2 | 16 | -3.87e-2 | +4.31e-2 | +5.29e-4 | -4.73e-5 |
| 5 | 3.00e-1 | 15 | 7.01e-2 | 1.47e-1 | 8.49e-2 | 9.48e-2 | 18 | -4.62e-2 | +4.35e-2 | +9.40e-4 | +8.66e-4 |
| 6 | 3.00e-1 | 14 | 6.23e-2 | 1.55e-1 | 8.26e-2 | 6.83e-2 | 17 | -5.05e-2 | +4.72e-2 | -3.06e-4 | -7.62e-4 |
| 7 | 3.00e-1 | 21 | 6.29e-2 | 1.50e-1 | 7.35e-2 | 7.03e-2 | 18 | -4.49e-2 | +4.50e-2 | +2.34e-4 | +5.95e-5 |
| 8 | 3.00e-1 | 13 | 5.90e-2 | 1.38e-1 | 7.00e-2 | 6.85e-2 | 15 | -5.65e-2 | +4.44e-2 | -3.88e-4 | +6.11e-5 |
| 9 | 3.00e-1 | 20 | 5.67e-2 | 1.36e-1 | 7.41e-2 | 7.73e-2 | 18 | -6.42e-2 | +5.93e-2 | +4.86e-4 | +2.98e-4 |
| 10 | 3.00e-1 | 11 | 5.89e-2 | 1.43e-1 | 7.47e-2 | 6.64e-2 | 17 | -5.91e-2 | +4.58e-2 | -5.90e-4 | -4.50e-4 |
| 11 | 3.00e-1 | 13 | 5.86e-2 | 1.36e-1 | 7.71e-2 | 7.45e-2 | 20 | -3.52e-2 | +3.76e-2 | +9.73e-5 | -2.35e-4 |
| 12 | 3.00e-1 | 17 | 6.78e-2 | 1.40e-1 | 7.85e-2 | 7.72e-2 | 18 | -3.27e-2 | +2.82e-2 | +3.79e-6 | +1.01e-4 |
| 13 | 3.00e-1 | 2 | 7.13e-2 | 7.64e-2 | 7.39e-2 | 7.13e-2 | 19 | -3.67e-3 | -5.57e-4 | -2.11e-3 | -3.35e-4 |
| 14 | 3.00e-1 | 2 | 7.41e-2 | 2.37e-1 | 1.56e-1 | 2.37e-1 | 229 | +1.40e-4 | +5.08e-3 | +2.61e-3 | +2.49e-4 |
| 15 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 265 | -5.44e-4 | -5.44e-4 | -5.44e-4 | +1.70e-4 |
| 16 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 237 | +1.31e-4 | +1.31e-4 | +1.31e-4 | +1.66e-4 |
| 17 | 3.00e-1 | 1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 2.01e-1 | 240 | -2.10e-4 | -2.10e-4 | -2.10e-4 | +1.28e-4 |
| 18 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 234 | -5.35e-5 | -5.35e-5 | -5.35e-5 | +1.10e-4 |
| 19 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 232 | -2.31e-5 | -2.31e-5 | -2.31e-5 | +9.69e-5 |
| 20 | 3.00e-1 | 2 | 1.88e-1 | 1.93e-1 | 1.90e-1 | 1.88e-1 | 201 | -1.23e-4 | -1.20e-4 | -1.21e-4 | +5.55e-5 |
| 21 | 3.00e-1 | 1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 1.85e-1 | 266 | -5.72e-5 | -5.72e-5 | -5.72e-5 | +4.42e-5 |
| 22 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 226 | +4.40e-4 | +4.40e-4 | +4.40e-4 | +8.38e-5 |
| 23 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 219 | -3.32e-4 | -3.32e-4 | -3.32e-4 | +4.22e-5 |
| 24 | 3.00e-1 | 2 | 1.91e-1 | 1.92e-1 | 1.91e-1 | 1.91e-1 | 169 | -1.06e-5 | +2.98e-5 | +9.56e-6 | +3.58e-5 |
| 25 | 3.00e-1 | 1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 1.77e-1 | 225 | -3.51e-4 | -3.51e-4 | -3.51e-4 | -2.91e-6 |
| 26 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 225 | +3.68e-4 | +3.68e-4 | +3.68e-4 | +3.42e-5 |
| 27 | 3.00e-1 | 2 | 1.81e-1 | 1.90e-1 | 1.86e-1 | 1.81e-1 | 169 | -2.68e-4 | -5.80e-5 | -1.63e-4 | -4.35e-6 |
| 28 | 3.00e-1 | 1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 1.76e-1 | 194 | -1.68e-4 | -1.68e-4 | -1.68e-4 | -2.07e-5 |
| 29 | 3.00e-1 | 2 | 1.82e-1 | 1.83e-1 | 1.83e-1 | 1.82e-1 | 169 | -6.37e-6 | +1.97e-4 | +9.53e-5 | +3.48e-7 |
| 30 | 3.00e-1 | 1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 1.74e-1 | 219 | -2.19e-4 | -2.19e-4 | -2.19e-4 | -2.15e-5 |
| 31 | 3.00e-1 | 2 | 1.93e-1 | 1.95e-1 | 1.94e-1 | 1.93e-1 | 185 | -5.13e-5 | +5.01e-4 | +2.25e-4 | +2.25e-5 |
| 32 | 3.00e-1 | 1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 1.81e-1 | 193 | -3.45e-4 | -3.45e-4 | -3.45e-4 | -1.42e-5 |
| 33 | 3.00e-1 | 1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 1.83e-1 | 183 | +7.88e-5 | +7.88e-5 | +7.88e-5 | -4.94e-6 |
| 34 | 3.00e-1 | 2 | 1.80e-1 | 1.86e-1 | 1.83e-1 | 1.86e-1 | 172 | -9.35e-5 | +2.04e-4 | +5.54e-5 | +8.01e-6 |
| 35 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 195 | -2.00e-4 | -2.00e-4 | -2.00e-4 | -1.28e-5 |
| 36 | 3.00e-1 | 2 | 1.85e-1 | 1.90e-1 | 1.87e-1 | 1.90e-1 | 172 | +1.43e-4 | +1.61e-4 | +1.52e-4 | +1.85e-5 |
| 37 | 3.00e-1 | 1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 1.79e-1 | 228 | -2.56e-4 | -2.56e-4 | -2.56e-4 | -8.95e-6 |
| 38 | 3.00e-1 | 2 | 1.87e-1 | 1.95e-1 | 1.91e-1 | 1.87e-1 | 161 | -2.64e-4 | +4.12e-4 | +7.43e-5 | +3.49e-6 |
| 39 | 3.00e-1 | 1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 1.70e-1 | 180 | -5.49e-4 | -5.49e-4 | -5.49e-4 | -5.18e-5 |
| 40 | 3.00e-1 | 2 | 1.80e-1 | 1.82e-1 | 1.81e-1 | 1.80e-1 | 161 | -7.96e-5 | +3.87e-4 | +1.54e-4 | -1.50e-5 |
| 41 | 3.00e-1 | 1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 1.75e-1 | 215 | -1.25e-4 | -1.25e-4 | -1.25e-4 | -2.61e-5 |
| 42 | 3.00e-1 | 2 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 1.91e-1 | 161 | -6.03e-6 | +4.03e-4 | +1.98e-4 | +1.45e-5 |
| 43 | 3.00e-1 | 2 | 1.73e-1 | 1.84e-1 | 1.78e-1 | 1.84e-1 | 150 | -5.32e-4 | +3.81e-4 | -7.54e-5 | +2.03e-6 |
| 44 | 3.00e-1 | 1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 1.73e-1 | 210 | -2.86e-4 | -2.86e-4 | -2.86e-4 | -2.68e-5 |
| 45 | 3.00e-1 | 2 | 1.79e-1 | 1.89e-1 | 1.84e-1 | 1.79e-1 | 150 | -3.64e-4 | +4.84e-4 | +5.97e-5 | -1.46e-5 |
| 46 | 3.00e-1 | 1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 1.68e-1 | 199 | -3.08e-4 | -3.08e-4 | -3.08e-4 | -4.39e-5 |
| 47 | 3.00e-1 | 2 | 1.83e-1 | 1.88e-1 | 1.85e-1 | 1.83e-1 | 159 | -1.73e-4 | +5.86e-4 | +2.07e-4 | -1.15e-7 |
| 48 | 3.00e-1 | 2 | 1.75e-1 | 1.86e-1 | 1.80e-1 | 1.86e-1 | 137 | -2.52e-4 | +4.62e-4 | +1.05e-4 | +2.35e-5 |
| 49 | 3.00e-1 | 2 | 1.60e-1 | 1.82e-1 | 1.71e-1 | 1.82e-1 | 137 | -8.44e-4 | +9.30e-4 | +4.29e-5 | +3.61e-5 |
| 50 | 3.00e-1 | 1 | 1.65e-1 | 1.65e-1 | 1.65e-1 | 1.65e-1 | 148 | -6.56e-4 | -6.56e-4 | -6.56e-4 | -3.32e-5 |
| 51 | 3.00e-1 | 2 | 1.71e-1 | 1.77e-1 | 1.74e-1 | 1.77e-1 | 137 | +2.34e-4 | +2.45e-4 | +2.40e-4 | +1.87e-5 |
| 52 | 3.00e-1 | 2 | 1.64e-1 | 1.81e-1 | 1.73e-1 | 1.81e-1 | 147 | -4.43e-4 | +6.82e-4 | +1.19e-4 | +4.35e-5 |
| 53 | 3.00e-1 | 2 | 1.71e-1 | 1.77e-1 | 1.74e-1 | 1.77e-1 | 141 | -3.51e-4 | +2.52e-4 | -4.96e-5 | +2.88e-5 |
| 54 | 3.00e-1 | 2 | 1.70e-1 | 1.75e-1 | 1.73e-1 | 1.75e-1 | 118 | -2.52e-4 | +2.66e-4 | +6.77e-6 | +2.72e-5 |
| 55 | 3.00e-1 | 2 | 1.57e-1 | 1.72e-1 | 1.64e-1 | 1.72e-1 | 138 | -7.33e-4 | +6.52e-4 | -4.04e-5 | +2.13e-5 |
| 56 | 3.00e-1 | 2 | 1.68e-1 | 1.78e-1 | 1.73e-1 | 1.78e-1 | 110 | -1.40e-4 | +5.16e-4 | +1.88e-4 | +5.63e-5 |
| 57 | 3.00e-1 | 3 | 1.51e-1 | 1.70e-1 | 1.59e-1 | 1.51e-1 | 103 | -1.16e-3 | +9.42e-4 | -3.87e-4 | -6.66e-5 |
| 58 | 3.00e-1 | 2 | 1.50e-1 | 1.71e-1 | 1.60e-1 | 1.71e-1 | 103 | -7.83e-5 | +1.31e-3 | +6.17e-4 | +7.01e-5 |
| 59 | 3.00e-1 | 2 | 1.50e-1 | 1.68e-1 | 1.59e-1 | 1.68e-1 | 103 | -1.01e-3 | +1.10e-3 | +4.18e-5 | +7.53e-5 |
| 60 | 3.00e-1 | 2 | 1.48e-1 | 1.81e-1 | 1.65e-1 | 1.81e-1 | 103 | -7.87e-4 | +1.95e-3 | +5.82e-4 | +1.85e-4 |
| 61 | 3.00e-1 | 3 | 1.51e-1 | 1.70e-1 | 1.59e-1 | 1.57e-1 | 103 | -1.23e-3 | +1.08e-3 | -3.20e-4 | +5.18e-5 |
| 62 | 3.00e-1 | 3 | 1.48e-1 | 1.70e-1 | 1.56e-1 | 1.48e-1 | 103 | -1.30e-3 | +1.28e-3 | -1.35e-4 | -8.15e-6 |
| 63 | 3.00e-1 | 2 | 1.55e-1 | 1.69e-1 | 1.62e-1 | 1.69e-1 | 103 | +3.06e-4 | +8.43e-4 | +5.75e-4 | +1.05e-4 |
| 64 | 3.00e-1 | 2 | 1.50e-1 | 1.70e-1 | 1.60e-1 | 1.70e-1 | 96 | -8.37e-4 | +1.33e-3 | +2.48e-4 | +1.43e-4 |
| 65 | 3.00e-1 | 4 | 1.45e-1 | 1.59e-1 | 1.51e-1 | 1.50e-1 | 98 | -1.18e-3 | +7.35e-4 | -2.61e-4 | +1.73e-5 |
| 66 | 3.00e-1 | 1 | 1.50e-1 | 1.50e-1 | 1.50e-1 | 1.50e-1 | 122 | -1.25e-5 | -1.25e-5 | -1.25e-5 | +1.43e-5 |
| 67 | 3.00e-1 | 3 | 1.44e-1 | 1.63e-1 | 1.56e-1 | 1.44e-1 | 81 | -1.35e-3 | +6.60e-4 | -2.92e-4 | -8.80e-5 |
| 68 | 3.00e-1 | 4 | 1.36e-1 | 1.60e-1 | 1.43e-1 | 1.39e-1 | 81 | -1.92e-3 | +1.96e-3 | -3.54e-5 | -7.79e-5 |
| 69 | 3.00e-1 | 3 | 1.34e-1 | 1.59e-1 | 1.43e-1 | 1.35e-1 | 82 | -1.96e-3 | +2.12e-3 | -7.06e-5 | -9.22e-5 |
| 70 | 3.00e-1 | 4 | 1.25e-1 | 1.56e-1 | 1.38e-1 | 1.25e-1 | 66 | -2.83e-3 | +1.41e-3 | -3.85e-4 | -2.24e-4 |
| 71 | 3.00e-1 | 3 | 1.25e-1 | 1.49e-1 | 1.34e-1 | 1.25e-1 | 70 | -2.55e-3 | +2.17e-3 | -2.34e-5 | -1.98e-4 |
| 72 | 3.00e-1 | 5 | 1.11e-1 | 1.49e-1 | 1.26e-1 | 1.11e-1 | 50 | -3.84e-3 | +1.58e-3 | -6.09e-4 | -4.13e-4 |
| 73 | 3.00e-1 | 4 | 1.14e-1 | 1.48e-1 | 1.24e-1 | 1.14e-1 | 53 | -4.77e-3 | +4.53e-3 | +3.11e-6 | -3.23e-4 |
| 74 | 3.00e-1 | 6 | 1.03e-1 | 1.49e-1 | 1.17e-1 | 1.09e-1 | 44 | -5.75e-3 | +5.09e-3 | -1.87e-4 | -2.94e-4 |
| 75 | 3.00e-1 | 6 | 1.01e-1 | 1.37e-1 | 1.13e-1 | 1.06e-1 | 39 | -7.05e-3 | +5.86e-3 | -1.27e-4 | -2.77e-4 |
| 76 | 3.00e-1 | 9 | 7.86e-2 | 1.48e-1 | 9.26e-2 | 8.99e-2 | 25 | -2.23e-2 | +1.06e-2 | -1.17e-3 | -5.56e-4 |
| 77 | 3.00e-1 | 12 | 6.84e-2 | 1.39e-1 | 8.05e-2 | 6.89e-2 | 18 | -2.88e-2 | +2.18e-2 | -1.10e-3 | -1.12e-3 |
| 78 | 3.00e-1 | 15 | 5.70e-2 | 1.29e-1 | 6.78e-2 | 6.74e-2 | 16 | -4.43e-2 | +4.06e-2 | -1.15e-4 | -1.28e-4 |
| 79 | 3.00e-1 | 15 | 5.32e-2 | 1.34e-1 | 7.00e-2 | 6.81e-2 | 16 | -6.18e-2 | +5.26e-2 | +2.61e-4 | +1.00e-5 |
| 80 | 3.00e-1 | 2 | 6.45e-2 | 6.63e-2 | 6.54e-2 | 6.45e-2 | 18 | -1.64e-3 | -1.51e-3 | -1.58e-3 | -2.90e-4 |
| 81 | 3.00e-1 | 1 | 6.76e-2 | 6.76e-2 | 6.76e-2 | 6.76e-2 | 327 | +1.43e-4 | +1.43e-4 | +1.43e-4 | -2.47e-4 |
| 82 | 3.00e-1 | 1 | 2.55e-1 | 2.55e-1 | 2.55e-1 | 2.55e-1 | 313 | +4.24e-3 | +4.24e-3 | +4.24e-3 | +2.02e-4 |
| 83 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 336 | -3.61e-4 | -3.61e-4 | -3.61e-4 | +1.45e-4 |
| 84 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 294 | -2.14e-5 | -2.14e-5 | -2.14e-5 | +1.29e-4 |
| 85 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 242 | -2.44e-4 | -2.44e-4 | -2.44e-4 | +9.14e-5 |
| 86 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 254 | -1.73e-4 | -1.73e-4 | -1.73e-4 | +6.50e-5 |
| 87 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 297 | +2.12e-5 | +2.12e-5 | +2.12e-5 | +6.06e-5 |
| 88 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 288 | +1.62e-4 | +1.62e-4 | +1.62e-4 | +7.08e-5 |
| 89 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 304 | -4.17e-5 | -4.17e-5 | -4.17e-5 | +5.95e-5 |
| 90 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 260 | +2.04e-5 | +2.04e-5 | +2.04e-5 | +5.56e-5 |
| 91 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 284 | -7.72e-5 | -7.72e-5 | -7.72e-5 | +4.23e-5 |
| 92 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 275 | +1.84e-5 | +1.84e-5 | +1.84e-5 | +4.00e-5 |
| 93 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 285 | -1.21e-5 | -1.21e-5 | -1.21e-5 | +3.47e-5 |
| 94 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 318 | +1.71e-5 | +1.71e-5 | +1.71e-5 | +3.30e-5 |
| 95 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 274 | +1.96e-4 | +1.96e-4 | +1.96e-4 | +4.93e-5 |
| 96 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 256 | -2.74e-4 | -2.74e-4 | -2.74e-4 | +1.69e-5 |
| 97 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 277 | -3.02e-6 | -3.02e-6 | -3.02e-6 | +1.49e-5 |
| 98 | 3.00e-1 | 1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 2.10e-1 | 278 | +7.81e-5 | +7.81e-5 | +7.81e-5 | +2.13e-5 |
| 99 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 263 | -4.51e-6 | -4.51e-6 | -4.51e-6 | +1.87e-5 |
| 100 | 3.00e-2 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 274 | -2.13e-5 | -2.13e-5 | -2.13e-5 | +1.47e-5 |
| 101 | 3.00e-2 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 273 | -2.55e-6 | -2.55e-6 | -2.55e-6 | +1.30e-5 |
| 102 | 3.00e-2 | 1 | 2.05e-2 | 2.05e-2 | 2.05e-2 | 2.05e-2 | 268 | -8.64e-3 | -8.64e-3 | -8.64e-3 | -8.52e-4 |
| 103 | 3.00e-2 | 1 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 279 | +2.68e-4 | +2.68e-4 | +2.68e-4 | -7.40e-4 |
| 104 | 3.00e-2 | 1 | 2.35e-2 | 2.35e-2 | 2.35e-2 | 2.35e-2 | 282 | +2.11e-4 | +2.11e-4 | +2.11e-4 | -6.45e-4 |
| 105 | 3.00e-2 | 1 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 2.50e-2 | 267 | +2.31e-4 | +2.31e-4 | +2.31e-4 | -5.58e-4 |
| 106 | 3.00e-2 | 1 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 2.48e-2 | 309 | -2.88e-5 | -2.88e-5 | -2.88e-5 | -5.05e-4 |
| 107 | 3.00e-2 | 1 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 270 | +4.20e-4 | +4.20e-4 | +4.20e-4 | -4.12e-4 |
| 108 | 3.00e-2 | 1 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 2.64e-2 | 262 | -1.92e-4 | -1.92e-4 | -1.92e-4 | -3.90e-4 |
| 109 | 3.00e-2 | 1 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 2.70e-2 | 270 | +8.69e-5 | +8.69e-5 | +8.69e-5 | -3.43e-4 |
| 110 | 3.00e-2 | 1 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 2.86e-2 | 283 | +2.03e-4 | +2.03e-4 | +2.03e-4 | -2.88e-4 |
| 111 | 3.00e-2 | 1 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 2.92e-2 | 315 | +6.60e-5 | +6.60e-5 | +6.60e-5 | -2.53e-4 |
| 112 | 3.00e-2 | 1 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 3.15e-2 | 311 | +2.39e-4 | +2.39e-4 | +2.39e-4 | -2.03e-4 |
| 113 | 3.00e-2 | 1 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 3.24e-2 | 262 | +1.07e-4 | +1.07e-4 | +1.07e-4 | -1.72e-4 |
| 114 | 3.00e-2 | 1 | 3.13e-2 | 3.13e-2 | 3.13e-2 | 3.13e-2 | 260 | -1.23e-4 | -1.23e-4 | -1.23e-4 | -1.67e-4 |
| 115 | 3.00e-2 | 1 | 3.19e-2 | 3.19e-2 | 3.19e-2 | 3.19e-2 | 250 | +7.75e-5 | +7.75e-5 | +7.75e-5 | -1.43e-4 |
| 116 | 3.00e-2 | 1 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 3.14e-2 | 274 | -5.75e-5 | -5.75e-5 | -5.75e-5 | -1.34e-4 |
| 117 | 3.00e-2 | 1 | 3.32e-2 | 3.32e-2 | 3.32e-2 | 3.32e-2 | 257 | +2.14e-4 | +2.14e-4 | +2.14e-4 | -9.96e-5 |
| 118 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 266 | -7.29e-5 | -7.29e-5 | -7.29e-5 | -9.69e-5 |
| 119 | 3.00e-2 | 2 | 3.44e-2 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 252 | +1.25e-5 | +2.18e-4 | +1.15e-4 | -5.76e-5 |
| 121 | 3.00e-2 | 1 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 3.60e-2 | 330 | +1.24e-4 | +1.24e-4 | +1.24e-4 | -3.95e-5 |
| 122 | 3.00e-2 | 2 | 3.74e-2 | 4.01e-2 | 3.87e-2 | 3.74e-2 | 210 | -3.32e-4 | +4.14e-4 | +4.11e-5 | -2.79e-5 |
| 123 | 3.00e-2 | 1 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 3.41e-2 | 258 | -3.52e-4 | -3.52e-4 | -3.52e-4 | -6.04e-5 |
| 124 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 290 | +2.76e-4 | +2.76e-4 | +2.76e-4 | -2.68e-5 |
| 125 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 236 | +4.34e-4 | +4.34e-4 | +4.34e-4 | +1.93e-5 |
| 126 | 3.00e-2 | 1 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 3.80e-2 | 216 | -3.41e-4 | -3.41e-4 | -3.41e-4 | -1.67e-5 |
| 127 | 3.00e-2 | 1 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 3.77e-2 | 217 | -3.59e-5 | -3.59e-5 | -3.59e-5 | -1.87e-5 |
| 128 | 3.00e-2 | 2 | 3.76e-2 | 4.00e-2 | 3.88e-2 | 4.00e-2 | 199 | -2.01e-5 | +3.13e-4 | +1.47e-4 | +1.44e-5 |
| 129 | 3.00e-2 | 1 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 3.76e-2 | 279 | -2.21e-4 | -2.21e-4 | -2.21e-4 | -9.10e-6 |
| 130 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 300 | +4.79e-4 | +4.79e-4 | +4.79e-4 | +3.97e-5 |
| 131 | 3.00e-2 | 1 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 4.57e-2 | 222 | +2.34e-4 | +2.34e-4 | +2.34e-4 | +5.92e-5 |
| 132 | 3.00e-2 | 1 | 4.08e-2 | 4.08e-2 | 4.08e-2 | 4.08e-2 | 244 | -4.70e-4 | -4.70e-4 | -4.70e-4 | +6.26e-6 |
| 133 | 3.00e-2 | 1 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 4.31e-2 | 205 | +2.69e-4 | +2.69e-4 | +2.69e-4 | +3.25e-5 |
| 134 | 3.00e-2 | 2 | 4.08e-2 | 4.18e-2 | 4.13e-2 | 4.18e-2 | 188 | -2.57e-4 | +1.24e-4 | -6.64e-5 | +1.56e-5 |
| 135 | 3.00e-2 | 1 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 4.03e-2 | 208 | -1.71e-4 | -1.71e-4 | -1.71e-4 | -3.05e-6 |
| 136 | 3.00e-2 | 1 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 4.30e-2 | 217 | +2.94e-4 | +2.94e-4 | +2.94e-4 | +2.67e-5 |
| 137 | 3.00e-2 | 2 | 4.39e-2 | 4.74e-2 | 4.57e-2 | 4.74e-2 | 202 | +8.74e-5 | +3.79e-4 | +2.33e-4 | +6.74e-5 |
| 138 | 3.00e-2 | 1 | 4.44e-2 | 4.44e-2 | 4.44e-2 | 4.44e-2 | 236 | -2.76e-4 | -2.76e-4 | -2.76e-4 | +3.30e-5 |
| 139 | 3.00e-2 | 1 | 4.68e-2 | 4.68e-2 | 4.68e-2 | 4.68e-2 | 223 | +2.32e-4 | +2.32e-4 | +2.32e-4 | +5.29e-5 |
| 140 | 3.00e-2 | 1 | 4.66e-2 | 4.66e-2 | 4.66e-2 | 4.66e-2 | 232 | -1.82e-5 | -1.82e-5 | -1.82e-5 | +4.58e-5 |
| 141 | 3.00e-2 | 2 | 4.62e-2 | 4.78e-2 | 4.70e-2 | 4.62e-2 | 190 | -1.82e-4 | +1.18e-4 | -3.20e-5 | +2.95e-5 |
| 142 | 3.00e-2 | 1 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 4.52e-2 | 216 | -1.03e-4 | -1.03e-4 | -1.03e-4 | +1.62e-5 |
| 143 | 3.00e-2 | 1 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 4.83e-2 | 212 | +3.15e-4 | +3.15e-4 | +3.15e-4 | +4.61e-5 |
| 144 | 3.00e-2 | 2 | 4.78e-2 | 4.97e-2 | 4.88e-2 | 4.97e-2 | 200 | -4.20e-5 | +1.89e-4 | +7.34e-5 | +5.25e-5 |
| 145 | 3.00e-2 | 1 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 4.84e-2 | 250 | -1.07e-4 | -1.07e-4 | -1.07e-4 | +3.65e-5 |
| 146 | 3.00e-2 | 1 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 5.38e-2 | 265 | +4.02e-4 | +4.02e-4 | +4.02e-4 | +7.31e-5 |
| 147 | 3.00e-2 | 1 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 205 | +1.38e-4 | +1.38e-4 | +1.38e-4 | +7.96e-5 |
| 148 | 3.00e-2 | 2 | 4.86e-2 | 5.01e-2 | 4.93e-2 | 5.01e-2 | 166 | -6.48e-4 | +1.84e-4 | -2.32e-4 | +2.46e-5 |
| 149 | 3.00e-2 | 1 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 202 | -4.75e-4 | -4.75e-4 | -4.75e-4 | -2.54e-5 |
| 150 | 3.00e-3 | 2 | 4.95e-2 | 5.03e-2 | 4.99e-2 | 5.03e-2 | 166 | +9.81e-5 | +4.52e-4 | +2.75e-4 | +2.99e-5 |
| 151 | 3.00e-3 | 1 | 4.83e-3 | 4.83e-3 | 4.83e-3 | 4.83e-3 | 176 | -1.33e-2 | -1.33e-2 | -1.33e-2 | -1.30e-3 |
| 152 | 3.00e-3 | 2 | 4.69e-3 | 4.87e-3 | 4.78e-3 | 4.87e-3 | 166 | -1.50e-4 | +2.20e-4 | +3.53e-5 | -1.05e-3 |
| 153 | 3.00e-3 | 1 | 4.34e-3 | 4.34e-3 | 4.34e-3 | 4.34e-3 | 200 | -5.77e-4 | -5.77e-4 | -5.77e-4 | -1.00e-3 |
| 154 | 3.00e-3 | 2 | 4.74e-3 | 4.86e-3 | 4.80e-3 | 4.74e-3 | 186 | -1.35e-4 | +5.77e-4 | +2.21e-4 | -7.72e-4 |
| 155 | 3.00e-3 | 1 | 4.91e-3 | 4.91e-3 | 4.91e-3 | 4.91e-3 | 209 | +1.68e-4 | +1.68e-4 | +1.68e-4 | -6.78e-4 |
| 156 | 3.00e-3 | 2 | 5.02e-3 | 5.17e-3 | 5.09e-3 | 5.17e-3 | 186 | +9.28e-5 | +1.58e-4 | +1.25e-4 | -5.25e-4 |
| 157 | 3.00e-3 | 1 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 4.73e-3 | 248 | -3.57e-4 | -3.57e-4 | -3.57e-4 | -5.08e-4 |
| 158 | 3.00e-3 | 1 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 5.49e-3 | 227 | +6.57e-4 | +6.57e-4 | +6.57e-4 | -3.92e-4 |
| 159 | 3.00e-3 | 2 | 4.98e-3 | 5.17e-3 | 5.08e-3 | 4.98e-3 | 148 | -3.11e-4 | -2.54e-4 | -2.83e-4 | -3.71e-4 |
| 160 | 3.00e-3 | 1 | 4.32e-3 | 4.32e-3 | 4.32e-3 | 4.32e-3 | 187 | -7.61e-4 | -7.61e-4 | -7.61e-4 | -4.10e-4 |
| 161 | 3.00e-3 | 2 | 4.74e-3 | 5.02e-3 | 4.88e-3 | 5.02e-3 | 155 | +3.64e-4 | +4.56e-4 | +4.10e-4 | -2.55e-4 |
| 162 | 3.00e-3 | 2 | 4.54e-3 | 4.99e-3 | 4.77e-3 | 4.99e-3 | 149 | -5.37e-4 | +6.27e-4 | +4.53e-5 | -1.92e-4 |
| 163 | 3.00e-3 | 2 | 4.59e-3 | 4.99e-3 | 4.79e-3 | 4.99e-3 | 123 | -4.19e-4 | +6.93e-4 | +1.37e-4 | -1.24e-4 |
| 164 | 3.00e-3 | 2 | 4.20e-3 | 4.67e-3 | 4.43e-3 | 4.67e-3 | 143 | -9.99e-4 | +7.44e-4 | -1.27e-4 | -1.16e-4 |
| 165 | 3.00e-3 | 1 | 4.34e-3 | 4.34e-3 | 4.34e-3 | 4.34e-3 | 155 | -4.63e-4 | -4.63e-4 | -4.63e-4 | -1.51e-4 |
| 166 | 3.00e-3 | 2 | 4.47e-3 | 4.69e-3 | 4.58e-3 | 4.69e-3 | 123 | +1.77e-4 | +3.94e-4 | +2.85e-4 | -6.66e-5 |
| 167 | 3.00e-3 | 2 | 4.21e-3 | 4.78e-3 | 4.49e-3 | 4.78e-3 | 123 | -6.82e-4 | +1.04e-3 | +1.79e-4 | -1.13e-5 |
| 168 | 3.00e-3 | 3 | 4.07e-3 | 4.57e-3 | 4.25e-3 | 4.07e-3 | 131 | -1.00e-3 | +8.29e-4 | -3.54e-4 | -1.04e-4 |
| 169 | 3.00e-3 | 1 | 4.18e-3 | 4.18e-3 | 4.18e-3 | 4.18e-3 | 165 | +1.75e-4 | +1.75e-4 | +1.75e-4 | -7.59e-5 |
| 170 | 3.00e-3 | 2 | 4.72e-3 | 4.83e-3 | 4.78e-3 | 4.72e-3 | 131 | -1.65e-4 | +9.39e-4 | +3.87e-4 | +6.60e-6 |
| 171 | 3.00e-3 | 2 | 4.25e-3 | 4.88e-3 | 4.56e-3 | 4.88e-3 | 131 | -6.38e-4 | +1.06e-3 | +2.11e-4 | +5.39e-5 |
| 172 | 3.00e-3 | 3 | 4.35e-3 | 4.58e-3 | 4.44e-3 | 4.40e-3 | 131 | -7.37e-4 | +3.91e-4 | -2.19e-4 | -1.63e-5 |
| 173 | 3.00e-3 | 1 | 4.50e-3 | 4.50e-3 | 4.50e-3 | 4.50e-3 | 168 | +1.30e-4 | +1.30e-4 | +1.30e-4 | -1.63e-6 |
| 174 | 3.00e-3 | 2 | 4.59e-3 | 4.84e-3 | 4.71e-3 | 4.59e-3 | 124 | -4.35e-4 | +4.87e-4 | +2.60e-5 | -9.79e-7 |
| 175 | 3.00e-3 | 3 | 4.08e-3 | 4.99e-3 | 4.51e-3 | 4.08e-3 | 114 | -1.77e-3 | +9.98e-4 | -3.23e-4 | -1.04e-4 |
| 176 | 3.00e-3 | 2 | 4.14e-3 | 4.51e-3 | 4.33e-3 | 4.51e-3 | 119 | +1.04e-4 | +7.26e-4 | +4.15e-4 | -2.08e-6 |
| 177 | 3.00e-3 | 2 | 4.15e-3 | 4.78e-3 | 4.46e-3 | 4.78e-3 | 114 | -5.40e-4 | +1.23e-3 | +3.47e-4 | +7.30e-5 |
| 178 | 3.00e-3 | 3 | 4.00e-3 | 4.35e-3 | 4.18e-3 | 4.00e-3 | 96 | -9.96e-4 | +4.23e-4 | -4.86e-4 | -7.80e-5 |
| 179 | 3.00e-3 | 2 | 3.81e-3 | 4.35e-3 | 4.08e-3 | 4.35e-3 | 100 | -3.93e-4 | +1.32e-3 | +4.65e-4 | +3.37e-5 |
| 180 | 3.00e-3 | 3 | 4.03e-3 | 4.46e-3 | 4.18e-3 | 4.03e-3 | 96 | -1.05e-3 | +1.01e-3 | -2.07e-4 | -3.67e-5 |
| 181 | 3.00e-3 | 3 | 3.80e-3 | 4.76e-3 | 4.12e-3 | 3.81e-3 | 89 | -2.51e-3 | +2.55e-3 | -1.13e-4 | -7.91e-5 |
| 182 | 3.00e-3 | 2 | 3.82e-3 | 4.42e-3 | 4.12e-3 | 4.42e-3 | 89 | +3.06e-5 | +1.64e-3 | +8.36e-4 | +1.03e-4 |
| 183 | 3.00e-3 | 3 | 3.61e-3 | 4.24e-3 | 3.88e-3 | 3.79e-3 | 103 | -1.81e-3 | +1.79e-3 | -3.70e-4 | -1.97e-5 |
| 184 | 3.00e-3 | 3 | 3.67e-3 | 4.51e-3 | 4.10e-3 | 3.67e-3 | 74 | -2.76e-3 | +1.09e-3 | -3.68e-4 | -1.46e-4 |
| 185 | 3.00e-3 | 4 | 3.30e-3 | 4.09e-3 | 3.58e-3 | 3.30e-3 | 66 | -2.94e-3 | +1.90e-3 | -3.40e-4 | -2.35e-4 |
| 186 | 3.00e-3 | 3 | 3.19e-3 | 3.94e-3 | 3.45e-3 | 3.19e-3 | 66 | -3.19e-3 | +3.05e-3 | -1.26e-4 | -2.35e-4 |
| 187 | 3.00e-3 | 4 | 3.14e-3 | 4.07e-3 | 3.41e-3 | 3.22e-3 | 66 | -3.92e-3 | +3.60e-3 | +3.11e-5 | -1.72e-4 |
| 188 | 3.00e-3 | 5 | 3.12e-3 | 3.89e-3 | 3.35e-3 | 3.12e-3 | 62 | -2.43e-3 | +3.20e-3 | -8.30e-5 | -1.75e-4 |
| 189 | 3.00e-3 | 3 | 2.91e-3 | 4.03e-3 | 3.35e-3 | 2.91e-3 | 53 | -6.15e-3 | +4.43e-3 | -5.85e-4 | -3.47e-4 |
| 190 | 3.00e-3 | 5 | 2.58e-3 | 3.78e-3 | 2.96e-3 | 2.58e-3 | 49 | -5.83e-3 | +5.59e-3 | -5.09e-4 | -4.83e-4 |
| 191 | 3.00e-3 | 9 | 2.30e-3 | 3.75e-3 | 2.67e-3 | 2.30e-3 | 36 | -9.18e-3 | +5.69e-3 | -4.73e-4 | -5.66e-4 |
| 192 | 3.00e-3 | 6 | 1.79e-3 | 3.24e-3 | 2.23e-3 | 1.79e-3 | 23 | -1.34e-2 | +1.28e-2 | -1.42e-3 | -1.16e-3 |
| 193 | 3.00e-3 | 11 | 1.54e-3 | 3.21e-3 | 1.80e-3 | 1.71e-3 | 17 | -3.48e-2 | +2.97e-2 | -2.97e-4 | -6.06e-4 |
| 194 | 3.00e-3 | 18 | 1.07e-3 | 2.98e-3 | 1.45e-3 | 1.51e-3 | 20 | -4.32e-2 | +4.46e-2 | +9.05e-5 | +2.22e-4 |
| 195 | 3.00e-3 | 11 | 1.39e-3 | 2.69e-3 | 1.63e-3 | 1.39e-3 | 16 | -2.87e-2 | +2.38e-2 | -7.16e-4 | -6.94e-4 |
| 196 | 3.00e-3 | 15 | 1.13e-3 | 2.98e-3 | 1.47e-3 | 1.40e-3 | 19 | -5.71e-2 | +5.07e-2 | -1.61e-4 | -2.09e-4 |
| 197 | 3.00e-3 | 17 | 1.31e-3 | 3.05e-3 | 1.60e-3 | 1.40e-3 | 16 | -4.11e-2 | +4.62e-2 | +1.83e-4 | -3.56e-4 |
| 198 | 3.00e-3 | 10 | 1.28e-3 | 3.42e-3 | 1.63e-3 | 1.46e-3 | 18 | -3.52e-2 | +4.27e-2 | +5.58e-5 | -2.46e-4 |
| 199 | 3.00e-3 | 3 | 1.37e-3 | 1.56e-3 | 1.45e-3 | 1.42e-3 | 265 | -3.48e-3 | +7.23e-3 | +1.13e-3 | +1.53e-4 |

