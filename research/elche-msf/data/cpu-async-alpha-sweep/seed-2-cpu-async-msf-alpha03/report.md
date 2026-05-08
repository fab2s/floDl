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
| cpu-async | 0.065670 | 0.9160 | +0.0035 | 1800.7 | 538 | 79.0 | 100% | 99% | 99% | 16.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9160 | cpu-async | - | - |

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
| cpu-async | 2.0416 | 0.7639 | 0.5811 | 0.5196 | 0.5379 | 0.5203 | 0.4991 | 0.4907 | 0.4800 | 0.4775 | 0.2236 | 0.1848 | 0.1661 | 0.1536 | 0.1397 | 0.0813 | 0.0766 | 0.0717 | 0.0675 | 0.0657 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4039 | 2.7 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3004 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2957 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 390 | 387 | 394 | 392 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1798.9 | 1.8 | epoch-boundary(199) |
| cpu-async | gpu2 | 1799.0 | 1.7 | epoch-boundary(199) |
| cpu-async | gpu1 | 1438.6 | 1.6 | epoch-boundary(159) |
| cpu-async | gpu2 | 1438.6 | 1.5 | epoch-boundary(159) |
| cpu-async | gpu1 | 1689.6 | 1.0 | epoch-boundary(187) |
| cpu-async | gpu2 | 1689.7 | 1.0 | epoch-boundary(187) |
| cpu-async | gpu1 | 514.3 | 0.8 | epoch-boundary(56) |
| cpu-async | gpu2 | 514.3 | 0.7 | epoch-boundary(56) |
| cpu-async | gpu0 | 491.2 | 0.7 | unexplained |
| cpu-async | gpu1 | 951.2 | 0.7 | epoch-boundary(105) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.7s | 0.7s | 1.4s |
| resnet-graph | cpu-async | gpu1 | 7.1s | 0.0s | 0.0s | 0.6s | 8.3s |
| resnet-graph | cpu-async | gpu2 | 6.0s | 0.0s | 0.0s | 0.6s | 7.2s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 286 | 0 | 538 | 79.0 | 1507/9562 | 538 | 79.0 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 204.8 | 11.4% |

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
| resnet-graph | cpu-async | 185 | 538 | 0 | 8.75e-3 | -2.08e-5 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 538 | 1.06e-1 | 7.38e-2 | 2.79e-3 | 4.85e-1 | 30.5 | -1.64e-4 | 9.30e-4 |
| resnet-graph | cpu-async | 1 | 538 | 1.07e-1 | 7.62e-2 | 2.83e-3 | 5.27e-1 | 33.8 | -2.00e-4 | 1.09e-3 |
| resnet-graph | cpu-async | 2 | 538 | 1.06e-1 | 7.60e-2 | 2.99e-3 | 5.12e-1 | 35.7 | -1.86e-4 | 1.03e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9907 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9914 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9915 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 55 (0,1,3,5,6,8,9,10…148,149) | 0 (—) | — | 0,1,3,5,6,8,9,10…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 2 | 2 |
| resnet-graph | cpu-async | 0e0 | 5 | 1 | 1 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 334 | -0.022 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 103 | -0.182 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 97 | +0.014 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 536 | -0.019 | 184 | +0.357 | +0.514 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 537 | 3.45e1–7.97e1 | 6.58e1 | 2.24e-3 | 3.23e-3 | 2.59e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 336 | 42–77685 | +7.356e-6 | 0.272 | +7.704e-6 | 0.304 | 88 | +9.250e-6 | 0.580 | 28–981 | +8.335e-4 | 0.621 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 324 | 915–77685 | +8.989e-6 | 0.477 | +9.349e-6 | 0.526 | 87 | +9.774e-6 | 0.637 | 71–981 | +9.023e-4 | 0.887 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 104 | 78345–116215 | -2.420e-6 | 0.010 | -3.063e-6 | 0.017 | 49 | -2.837e-6 | 0.011 | 140–660 | +1.736e-4 | 0.008 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 98 | 116458–155894 | +1.458e-5 | 0.065 | +1.565e-5 | 0.074 | 48 | +1.498e-5 | 0.175 | 97–1067 | +8.523e-4 | 0.182 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +8.335e-4 | r0: +8.121e-4, r1: +8.450e-4, r2: +8.457e-4 | r0: 0.619, r1: 0.593, r2: 0.604 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.023e-4 | r0: +8.772e-4, r1: +9.148e-4, r2: +9.170e-4 | r0: 0.865, r1: 0.843, r2: 0.870 | 1.05× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | +1.736e-4 | r0: +1.293e-4, r1: +1.962e-4, r2: +1.977e-4 | r0: 0.005, r1: 0.011, r2: 0.010 | 1.53× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +8.523e-4 | r0: +8.532e-4, r1: +8.524e-4, r2: +8.550e-4 | r0: 0.183, r1: 0.179, r2: 0.182 | 1.00× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇████████████▆▅▄▅▅▅▅▅▅▅▅▅▄▁▁▂▂▂▂▂▂▂▂▂▂` | `▁▆▇▇▇▇▇▇▇▇▇███████████▇▇▇████████▇▄▅▇██████████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 12 | 1.83e-1 | 5.27e-1 | 2.73e-1 | 2.03e-1 | 37 | -2.57e-2 | +2.64e-3 | -8.38e-3 | -5.83e-3 |
| 1 | 3.00e-1 | 6 | 1.91e-1 | 2.55e-1 | 2.11e-1 | 1.91e-1 | 41 | -4.70e-3 | +2.80e-3 | -7.50e-4 | -3.00e-3 |
| 2 | 3.00e-1 | 9 | 1.21e-1 | 1.94e-1 | 1.49e-1 | 1.23e-1 | 25 | -5.40e-3 | +8.36e-4 | -1.97e-3 | -2.24e-3 |
| 3 | 3.00e-1 | 8 | 1.31e-1 | 1.64e-1 | 1.40e-1 | 1.39e-1 | 42 | -7.52e-3 | +4.28e-3 | -3.48e-4 | -1.08e-3 |
| 4 | 3.00e-1 | 5 | 1.44e-1 | 1.75e-1 | 1.52e-1 | 1.45e-1 | 46 | -3.86e-3 | +2.84e-3 | -3.22e-4 | -7.78e-4 |
| 5 | 3.00e-1 | 6 | 1.38e-1 | 1.71e-1 | 1.48e-1 | 1.39e-1 | 45 | -3.03e-3 | +2.12e-3 | -4.17e-4 | -6.15e-4 |
| 6 | 3.00e-1 | 7 | 1.21e-1 | 1.77e-1 | 1.37e-1 | 1.21e-1 | 30 | -6.83e-3 | +2.99e-3 | -1.28e-3 | -9.54e-4 |
| 7 | 3.00e-1 | 7 | 1.21e-1 | 1.62e-1 | 1.29e-1 | 1.21e-1 | 33 | -6.97e-3 | +4.07e-3 | -7.52e-4 | -8.25e-4 |
| 8 | 3.00e-1 | 7 | 1.26e-1 | 1.64e-1 | 1.35e-1 | 1.32e-1 | 40 | -6.67e-3 | +4.01e-3 | -4.17e-4 | -5.74e-4 |
| 9 | 3.00e-1 | 6 | 1.31e-1 | 1.64e-1 | 1.40e-1 | 1.34e-1 | 43 | -4.09e-3 | +2.86e-3 | -3.83e-4 | -4.86e-4 |
| 10 | 3.00e-1 | 6 | 1.30e-1 | 1.66e-1 | 1.42e-1 | 1.38e-1 | 46 | -3.41e-3 | +2.71e-3 | -2.93e-4 | -3.81e-4 |
| 11 | 3.00e-1 | 7 | 1.22e-1 | 1.78e-1 | 1.38e-1 | 1.31e-1 | 46 | -5.20e-3 | +2.76e-3 | -8.41e-4 | -5.50e-4 |
| 12 | 3.00e-1 | 6 | 1.31e-1 | 1.61e-1 | 1.40e-1 | 1.32e-1 | 43 | -2.27e-3 | +2.51e-3 | -3.28e-4 | -4.66e-4 |
| 13 | 3.00e-1 | 8 | 1.29e-1 | 1.67e-1 | 1.39e-1 | 1.32e-1 | 40 | -3.81e-3 | +2.82e-3 | -3.42e-4 | -3.86e-4 |
| 14 | 3.00e-1 | 5 | 1.20e-1 | 1.62e-1 | 1.37e-1 | 1.20e-1 | 32 | -2.64e-3 | +2.44e-3 | -1.08e-3 | -7.16e-4 |
| 15 | 3.00e-1 | 6 | 1.22e-1 | 1.56e-1 | 1.30e-1 | 1.26e-1 | 40 | -5.81e-3 | +3.74e-3 | -5.25e-4 | -6.07e-4 |
| 16 | 3.00e-1 | 8 | 1.21e-1 | 1.62e-1 | 1.29e-1 | 1.23e-1 | 34 | -4.72e-3 | +3.47e-3 | -5.13e-4 | -5.10e-4 |
| 17 | 3.00e-1 | 7 | 1.18e-1 | 1.61e-1 | 1.27e-1 | 1.21e-1 | 34 | -6.39e-3 | +3.70e-3 | -7.56e-4 | -5.87e-4 |
| 18 | 3.00e-1 | 7 | 1.22e-1 | 1.70e-1 | 1.33e-1 | 1.22e-1 | 35 | -6.37e-3 | +3.72e-3 | -8.03e-4 | -6.84e-4 |
| 19 | 3.00e-1 | 9 | 1.26e-1 | 1.65e-1 | 1.33e-1 | 1.29e-1 | 40 | -3.99e-3 | +3.57e-3 | -3.23e-4 | -4.21e-4 |
| 20 | 3.00e-1 | 5 | 1.28e-1 | 1.64e-1 | 1.37e-1 | 1.32e-1 | 42 | -5.67e-3 | +3.22e-3 | -6.25e-4 | -4.92e-4 |
| 21 | 3.00e-1 | 7 | 1.18e-1 | 1.70e-1 | 1.28e-1 | 1.19e-1 | 33 | -8.42e-3 | +3.01e-3 | -1.15e-3 | -7.50e-4 |
| 22 | 3.00e-1 | 6 | 1.24e-1 | 1.59e-1 | 1.36e-1 | 1.40e-1 | 51 | -5.99e-3 | +3.79e-3 | -2.16e-4 | -4.58e-4 |
| 23 | 3.00e-1 | 6 | 1.26e-1 | 1.80e-1 | 1.42e-1 | 1.26e-1 | 38 | -4.74e-3 | +2.67e-3 | -9.57e-4 | -7.03e-4 |
| 24 | 3.00e-1 | 6 | 1.27e-1 | 1.65e-1 | 1.38e-1 | 1.27e-1 | 40 | -4.74e-3 | +3.29e-3 | -5.72e-4 | -6.60e-4 |
| 25 | 3.00e-1 | 6 | 1.29e-1 | 1.66e-1 | 1.38e-1 | 1.29e-1 | 40 | -3.80e-3 | +3.24e-3 | -5.05e-4 | -5.96e-4 |
| 26 | 3.00e-1 | 7 | 1.26e-1 | 1.70e-1 | 1.39e-1 | 1.28e-1 | 37 | -3.88e-3 | +2.91e-3 | -5.36e-4 | -5.75e-4 |
| 27 | 3.00e-1 | 6 | 1.32e-1 | 1.59e-1 | 1.39e-1 | 1.33e-1 | 45 | -4.04e-3 | +2.89e-3 | -3.27e-4 | -4.65e-4 |
| 28 | 3.00e-1 | 6 | 1.30e-1 | 1.67e-1 | 1.42e-1 | 1.30e-1 | 37 | -3.00e-3 | +2.85e-3 | -4.99e-4 | -5.13e-4 |
| 29 | 3.00e-1 | 10 | 1.16e-1 | 1.72e-1 | 1.28e-1 | 1.25e-1 | 37 | -7.66e-3 | +3.28e-3 | -7.53e-4 | -4.91e-4 |
| 30 | 3.00e-1 | 6 | 1.17e-1 | 1.64e-1 | 1.31e-1 | 1.19e-1 | 32 | -6.30e-3 | +3.19e-3 | -1.16e-3 | -7.82e-4 |
| 31 | 3.00e-1 | 8 | 1.19e-1 | 1.54e-1 | 1.28e-1 | 1.28e-1 | 41 | -4.29e-3 | +3.97e-3 | -2.41e-4 | -4.23e-4 |
| 32 | 3.00e-1 | 6 | 1.32e-1 | 1.63e-1 | 1.39e-1 | 1.32e-1 | 38 | -3.96e-3 | +3.17e-3 | -3.38e-4 | -4.02e-4 |
| 33 | 3.00e-1 | 7 | 1.21e-1 | 1.64e-1 | 1.31e-1 | 1.22e-1 | 36 | -6.05e-3 | +2.91e-3 | -8.51e-4 | -5.88e-4 |
| 34 | 3.00e-1 | 10 | 1.23e-1 | 1.57e-1 | 1.30e-1 | 1.29e-1 | 39 | -3.94e-3 | +3.35e-3 | -2.33e-4 | -2.88e-4 |
| 35 | 3.00e-1 | 4 | 1.29e-1 | 1.71e-1 | 1.46e-1 | 1.29e-1 | 34 | -4.38e-3 | +3.45e-3 | -1.03e-3 | -6.06e-4 |
| 36 | 3.00e-1 | 8 | 1.23e-1 | 1.61e-1 | 1.30e-1 | 1.25e-1 | 35 | -6.50e-3 | +2.98e-3 | -6.28e-4 | -5.50e-4 |
| 37 | 3.00e-1 | 7 | 1.20e-1 | 1.60e-1 | 1.32e-1 | 1.20e-1 | 34 | -3.84e-3 | +3.13e-3 | -7.28e-4 | -6.70e-4 |
| 38 | 3.00e-1 | 8 | 1.30e-1 | 1.64e-1 | 1.40e-1 | 1.34e-1 | 40 | -2.90e-3 | +3.80e-3 | -1.55e-4 | -4.08e-4 |
| 39 | 3.00e-1 | 1 | 1.27e-1 | 1.27e-1 | 1.27e-1 | 1.27e-1 | 35 | -1.37e-3 | -1.37e-3 | -1.37e-3 | -5.04e-4 |
| 40 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 314 | +1.84e-3 | +1.84e-3 | +1.84e-3 | -2.70e-4 |
| 41 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 231 | +6.22e-5 | +6.22e-5 | +6.22e-5 | -2.37e-4 |
| 42 | 3.00e-1 | 2 | 2.31e-1 | 2.35e-1 | 2.33e-1 | 2.31e-1 | 237 | -5.94e-5 | +7.94e-5 | +1.00e-5 | -1.90e-4 |
| 44 | 3.00e-1 | 2 | 2.40e-1 | 2.46e-1 | 2.43e-1 | 2.40e-1 | 275 | -8.01e-5 | +1.95e-4 | +5.76e-5 | -1.45e-4 |
| 46 | 3.00e-1 | 1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 2.48e-1 | 322 | +9.91e-5 | +9.91e-5 | +9.91e-5 | -1.20e-4 |
| 47 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 286 | -9.43e-5 | -9.43e-5 | -9.43e-5 | -1.18e-4 |
| 48 | 3.00e-1 | 1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 2.45e-1 | 307 | +4.71e-5 | +4.71e-5 | +4.71e-5 | -1.01e-4 |
| 49 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 287 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -1.05e-4 |
| 50 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 297 | +5.59e-5 | +5.59e-5 | +5.59e-5 | -8.89e-5 |
| 51 | 3.00e-1 | 1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 2.42e-1 | 299 | +4.11e-5 | +4.11e-5 | +4.11e-5 | -7.59e-5 |
| 52 | 3.00e-1 | 1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 2.39e-1 | 289 | -5.55e-5 | -5.55e-5 | -5.55e-5 | -7.38e-5 |
| 53 | 3.00e-1 | 1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 2.41e-1 | 305 | +3.24e-5 | +3.24e-5 | +3.24e-5 | -6.32e-5 |
| 55 | 3.00e-1 | 2 | 2.41e-1 | 2.56e-1 | 2.48e-1 | 2.41e-1 | 275 | -2.27e-4 | +1.69e-4 | -2.91e-5 | -5.87e-5 |
| 57 | 3.00e-1 | 2 | 2.37e-1 | 2.59e-1 | 2.48e-1 | 2.37e-1 | 251 | -3.60e-4 | +2.09e-4 | -7.55e-5 | -6.48e-5 |
| 59 | 3.00e-1 | 1 | 2.53e-1 | 2.53e-1 | 2.53e-1 | 2.53e-1 | 343 | +1.94e-4 | +1.94e-4 | +1.94e-4 | -3.89e-5 |
| 60 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 262 | -2.53e-4 | -2.53e-4 | -2.53e-4 | -6.02e-5 |
| 61 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 249 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -6.61e-5 |
| 62 | 3.00e-1 | 1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 2.30e-1 | 257 | +3.64e-6 | +3.64e-6 | +3.64e-6 | -5.92e-5 |
| 63 | 3.00e-1 | 2 | 2.31e-1 | 2.32e-1 | 2.31e-1 | 2.32e-1 | 244 | +8.01e-6 | +1.59e-5 | +1.20e-5 | -4.56e-5 |
| 65 | 3.00e-1 | 2 | 2.37e-1 | 2.47e-1 | 2.42e-1 | 2.37e-1 | 259 | -1.47e-4 | +2.04e-4 | +2.83e-5 | -3.33e-5 |
| 67 | 3.00e-1 | 2 | 2.42e-1 | 2.49e-1 | 2.45e-1 | 2.42e-1 | 259 | -1.21e-4 | +1.55e-4 | +1.70e-5 | -2.51e-5 |
| 69 | 3.00e-1 | 2 | 2.33e-1 | 2.45e-1 | 2.39e-1 | 2.33e-1 | 246 | -1.97e-4 | +4.12e-5 | -7.79e-5 | -3.63e-5 |
| 71 | 3.00e-1 | 2 | 2.33e-1 | 2.46e-1 | 2.39e-1 | 2.33e-1 | 254 | -2.14e-4 | +1.66e-4 | -2.42e-5 | -3.60e-5 |
| 73 | 3.00e-1 | 2 | 2.32e-1 | 2.46e-1 | 2.39e-1 | 2.32e-1 | 230 | -2.51e-4 | +1.78e-4 | -3.62e-5 | -3.81e-5 |
| 75 | 3.00e-1 | 2 | 2.29e-1 | 2.46e-1 | 2.37e-1 | 2.29e-1 | 218 | -3.16e-4 | +1.88e-4 | -6.39e-5 | -4.56e-5 |
| 76 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 225 | -6.52e-5 | -6.52e-5 | -6.52e-5 | -4.75e-5 |
| 77 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 247 | +2.74e-5 | +2.74e-5 | +2.74e-5 | -4.00e-5 |
| 78 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 253 | +7.97e-5 | +7.97e-5 | +7.97e-5 | -2.81e-5 |
| 79 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 252 | +8.69e-5 | +8.69e-5 | +8.69e-5 | -1.66e-5 |
| 80 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 286 | +9.91e-5 | +9.91e-5 | +9.91e-5 | -5.00e-6 |
| 81 | 3.00e-1 | 1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 2.40e-1 | 274 | -6.33e-5 | -6.33e-5 | -6.33e-5 | -1.08e-5 |
| 82 | 3.00e-1 | 1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 2.50e-1 | 314 | +1.36e-4 | +1.36e-4 | +1.36e-4 | +3.88e-6 |
| 83 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 233 | -3.17e-4 | -3.17e-4 | -3.17e-4 | -2.82e-5 |
| 84 | 3.00e-1 | 2 | 2.26e-1 | 2.31e-1 | 2.28e-1 | 2.26e-1 | 206 | -9.21e-5 | -3.62e-5 | -6.41e-5 | -3.53e-5 |
| 85 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 242 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -1.83e-5 |
| 86 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 247 | -1.18e-6 | -1.18e-6 | -1.18e-6 | -1.66e-5 |
| 87 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 247 | +2.53e-5 | +2.53e-5 | +2.53e-5 | -1.24e-5 |
| 88 | 3.00e-1 | 1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 2.35e-1 | 253 | -6.29e-6 | -6.29e-6 | -6.29e-6 | -1.18e-5 |
| 89 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 215 | -2.12e-4 | -2.12e-4 | -2.12e-4 | -3.18e-5 |
| 90 | 3.00e-1 | 2 | 2.20e-1 | 2.30e-1 | 2.25e-1 | 2.30e-1 | 229 | -1.04e-4 | +1.96e-4 | +4.57e-5 | -1.56e-5 |
| 91 | 3.00e-1 | 1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 2.33e-1 | 239 | +5.63e-5 | +5.63e-5 | +5.63e-5 | -8.43e-6 |
| 92 | 3.00e-1 | 2 | 2.21e-1 | 2.31e-1 | 2.26e-1 | 2.21e-1 | 193 | -2.27e-4 | -3.30e-5 | -1.30e-4 | -3.25e-5 |
| 94 | 3.00e-1 | 2 | 2.21e-1 | 2.48e-1 | 2.34e-1 | 2.21e-1 | 183 | -6.29e-4 | +4.05e-4 | -1.12e-4 | -5.27e-5 |
| 95 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 232 | +2.17e-4 | +2.17e-4 | +2.17e-4 | -2.58e-5 |
| 96 | 3.00e-1 | 2 | 2.17e-1 | 2.30e-1 | 2.24e-1 | 2.17e-1 | 184 | -3.03e-4 | -4.36e-5 | -1.73e-4 | -5.51e-5 |
| 97 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 210 | +9.96e-5 | +9.96e-5 | +9.96e-5 | -3.96e-5 |
| 98 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 213 | +1.05e-4 | +1.05e-4 | +1.05e-4 | -2.51e-5 |
| 99 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 231 | +9.97e-5 | +9.97e-5 | +9.97e-5 | -1.26e-5 |
| 100 | 3.00e-2 | 2 | 1.57e-1 | 2.37e-1 | 1.97e-1 | 1.57e-1 | 171 | -2.41e-3 | +9.14e-5 | -1.16e-3 | -2.43e-4 |
| 101 | 3.00e-2 | 1 | 1.10e-1 | 1.10e-1 | 1.10e-1 | 1.10e-1 | 213 | -1.67e-3 | -1.67e-3 | -1.67e-3 | -3.86e-4 |
| 102 | 3.00e-2 | 2 | 5.81e-2 | 7.70e-2 | 6.75e-2 | 5.81e-2 | 171 | -1.88e-3 | -1.65e-3 | -1.77e-3 | -6.47e-4 |
| 103 | 3.00e-2 | 1 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 4.34e-2 | 194 | -1.50e-3 | -1.50e-3 | -1.50e-3 | -7.33e-4 |
| 104 | 3.00e-2 | 1 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 3.64e-2 | 208 | -8.41e-4 | -8.41e-4 | -8.41e-4 | -7.43e-4 |
| 105 | 3.00e-2 | 2 | 3.11e-2 | 3.28e-2 | 3.19e-2 | 3.11e-2 | 183 | -4.98e-4 | -2.89e-4 | -3.94e-4 | -6.76e-4 |
| 106 | 3.00e-2 | 1 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 216 | +1.46e-5 | +1.46e-5 | +1.46e-5 | -6.07e-4 |
| 107 | 3.00e-2 | 2 | 3.05e-2 | 3.14e-2 | 3.09e-2 | 3.05e-2 | 172 | -1.67e-4 | +2.38e-5 | -7.14e-5 | -5.06e-4 |
| 108 | 3.00e-2 | 1 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 3.17e-2 | 215 | +1.82e-4 | +1.82e-4 | +1.82e-4 | -4.37e-4 |
| 109 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 207 | +1.35e-4 | +1.35e-4 | +1.35e-4 | -3.80e-4 |
| 110 | 3.00e-2 | 2 | 3.21e-2 | 3.22e-2 | 3.21e-2 | 3.21e-2 | 172 | -6.08e-5 | -1.91e-5 | -3.99e-5 | -3.15e-4 |
| 111 | 3.00e-2 | 1 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 3.33e-2 | 202 | +1.77e-4 | +1.77e-4 | +1.77e-4 | -2.66e-4 |
| 112 | 3.00e-2 | 2 | 3.36e-2 | 3.42e-2 | 3.39e-2 | 3.36e-2 | 163 | -1.01e-4 | +1.42e-4 | +2.07e-5 | -2.13e-4 |
| 113 | 3.00e-2 | 1 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 3.73e-2 | 228 | +4.56e-4 | +4.56e-4 | +4.56e-4 | -1.46e-4 |
| 114 | 3.00e-2 | 2 | 3.50e-2 | 3.52e-2 | 3.51e-2 | 3.50e-2 | 152 | -3.04e-4 | -5.02e-5 | -1.77e-4 | -1.51e-4 |
| 115 | 3.00e-2 | 2 | 3.50e-2 | 3.60e-2 | 3.55e-2 | 3.50e-2 | 152 | -1.86e-4 | +1.47e-4 | -1.96e-5 | -1.27e-4 |
| 116 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 190 | +2.90e-4 | +2.90e-4 | +2.90e-4 | -8.57e-5 |
| 117 | 3.00e-2 | 2 | 3.70e-2 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 161 | +4.17e-6 | +2.18e-5 | +1.30e-5 | -6.68e-5 |
| 118 | 3.00e-2 | 1 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 3.89e-2 | 187 | +2.44e-4 | +2.44e-4 | +2.44e-4 | -3.58e-5 |
| 119 | 3.00e-2 | 2 | 3.84e-2 | 3.93e-2 | 3.89e-2 | 3.84e-2 | 175 | -1.36e-4 | +6.17e-5 | -3.72e-5 | -3.70e-5 |
| 120 | 3.00e-2 | 2 | 3.85e-2 | 4.00e-2 | 3.92e-2 | 3.85e-2 | 152 | -2.65e-4 | +2.26e-4 | -1.94e-5 | -3.61e-5 |
| 121 | 3.00e-2 | 1 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 200 | +5.05e-4 | +5.05e-4 | +5.05e-4 | +1.80e-5 |
| 122 | 3.00e-2 | 2 | 3.89e-2 | 4.05e-2 | 3.97e-2 | 3.89e-2 | 137 | -2.93e-4 | -2.86e-4 | -2.89e-4 | -4.04e-5 |
| 123 | 3.00e-2 | 1 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 4.17e-2 | 177 | +3.88e-4 | +3.88e-4 | +3.88e-4 | +2.40e-6 |
| 124 | 3.00e-2 | 3 | 3.99e-2 | 4.08e-2 | 4.05e-2 | 4.08e-2 | 141 | -1.68e-4 | +1.73e-4 | -3.95e-5 | -6.08e-6 |
| 125 | 3.00e-2 | 1 | 4.32e-2 | 4.32e-2 | 4.32e-2 | 4.32e-2 | 183 | +3.10e-4 | +3.10e-4 | +3.10e-4 | +2.55e-5 |
| 126 | 3.00e-2 | 2 | 4.17e-2 | 4.42e-2 | 4.30e-2 | 4.17e-2 | 130 | -4.50e-4 | +1.19e-4 | -1.65e-4 | -1.36e-5 |
| 127 | 3.00e-2 | 2 | 4.13e-2 | 4.36e-2 | 4.25e-2 | 4.13e-2 | 130 | -4.09e-4 | +2.60e-4 | -7.44e-5 | -2.85e-5 |
| 128 | 3.00e-2 | 2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 4.12e-2 | 130 | -2.68e-5 | +7.27e-6 | -9.75e-6 | -2.48e-5 |
| 129 | 3.00e-2 | 2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 4.25e-2 | 130 | +6.35e-7 | +2.11e-4 | +1.06e-4 | -9.72e-7 |
| 130 | 3.00e-2 | 2 | 4.38e-2 | 4.57e-2 | 4.48e-2 | 4.38e-2 | 130 | -3.26e-4 | +4.25e-4 | +4.98e-5 | +4.92e-6 |
| 131 | 3.00e-2 | 1 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 4.70e-2 | 181 | +3.86e-4 | +3.86e-4 | +3.86e-4 | +4.30e-5 |
| 132 | 3.00e-2 | 3 | 4.29e-2 | 4.66e-2 | 4.45e-2 | 4.29e-2 | 104 | -7.98e-4 | +4.12e-4 | -3.11e-4 | -5.56e-5 |
| 133 | 3.00e-2 | 2 | 4.31e-2 | 4.68e-2 | 4.49e-2 | 4.31e-2 | 107 | -7.76e-4 | +5.68e-4 | -1.04e-4 | -7.15e-5 |
| 134 | 3.00e-2 | 2 | 4.31e-2 | 4.63e-2 | 4.47e-2 | 4.31e-2 | 112 | -6.45e-4 | +4.99e-4 | -7.28e-5 | -7.75e-5 |
| 135 | 3.00e-2 | 3 | 4.05e-2 | 4.67e-2 | 4.32e-2 | 4.05e-2 | 96 | -1.04e-3 | +5.70e-4 | -3.04e-4 | -1.48e-4 |
| 136 | 3.00e-2 | 2 | 4.42e-2 | 4.46e-2 | 4.44e-2 | 4.42e-2 | 98 | -1.04e-4 | +7.62e-4 | +3.29e-4 | -6.18e-5 |
| 137 | 3.00e-2 | 4 | 4.41e-2 | 4.79e-2 | 4.56e-2 | 4.41e-2 | 105 | -6.09e-4 | +5.44e-4 | -6.03e-5 | -6.93e-5 |
| 138 | 3.00e-2 | 1 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 4.59e-2 | 118 | +3.31e-4 | +3.31e-4 | +3.31e-4 | -2.93e-5 |
| 139 | 3.00e-2 | 4 | 4.13e-2 | 5.02e-2 | 4.40e-2 | 4.13e-2 | 81 | -1.96e-3 | +6.47e-4 | -4.38e-4 | -1.71e-4 |
| 140 | 3.00e-2 | 2 | 4.28e-2 | 4.68e-2 | 4.48e-2 | 4.28e-2 | 81 | -1.10e-3 | +1.06e-3 | -2.21e-5 | -1.54e-4 |
| 141 | 3.00e-2 | 3 | 4.09e-2 | 4.60e-2 | 4.31e-2 | 4.09e-2 | 81 | -9.82e-4 | +6.25e-4 | -2.73e-4 | -1.96e-4 |
| 142 | 3.00e-2 | 5 | 3.99e-2 | 4.90e-2 | 4.32e-2 | 3.99e-2 | 72 | -1.40e-3 | +1.41e-3 | -2.52e-4 | -2.40e-4 |
| 143 | 3.00e-2 | 2 | 4.40e-2 | 4.65e-2 | 4.52e-2 | 4.40e-2 | 72 | -7.69e-4 | +1.34e-3 | +2.86e-4 | -1.51e-4 |
| 144 | 3.00e-2 | 3 | 4.24e-2 | 4.85e-2 | 4.45e-2 | 4.24e-2 | 72 | -1.86e-3 | +8.66e-4 | -3.37e-4 | -2.09e-4 |
| 145 | 3.00e-2 | 5 | 4.02e-2 | 4.70e-2 | 4.25e-2 | 4.02e-2 | 63 | -1.39e-3 | +9.54e-4 | -2.75e-4 | -2.47e-4 |
| 146 | 3.00e-2 | 3 | 3.79e-2 | 4.85e-2 | 4.25e-2 | 4.10e-2 | 58 | -2.87e-3 | +2.16e-3 | -5.66e-4 | -3.53e-4 |
| 147 | 3.00e-2 | 5 | 3.66e-2 | 4.58e-2 | 3.98e-2 | 3.75e-2 | 56 | -2.42e-3 | +1.13e-3 | -5.27e-4 | -4.22e-4 |
| 148 | 3.00e-2 | 4 | 3.84e-2 | 4.61e-2 | 4.06e-2 | 3.87e-2 | 56 | -2.91e-3 | +2.32e-3 | -2.03e-4 | -3.62e-4 |
| 149 | 3.00e-3 | 7 | 2.52e-2 | 4.70e-2 | 3.78e-2 | 2.52e-2 | 50 | -7.34e-3 | +2.07e-3 | -1.46e-3 | -1.13e-3 |
| 150 | 3.00e-3 | 4 | 6.85e-3 | 2.09e-2 | 1.23e-2 | 6.85e-3 | 44 | -1.18e-2 | -2.08e-3 | -6.85e-3 | -3.13e-3 |
| 151 | 3.00e-3 | 6 | 3.38e-3 | 6.58e-3 | 4.30e-3 | 3.38e-3 | 44 | -8.55e-3 | -4.46e-4 | -2.82e-3 | -2.88e-3 |
| 152 | 3.00e-3 | 7 | 3.37e-3 | 4.47e-3 | 3.63e-3 | 3.37e-3 | 41 | -4.68e-3 | +3.32e-3 | -4.43e-4 | -1.62e-3 |
| 153 | 3.00e-3 | 5 | 3.31e-3 | 4.12e-3 | 3.55e-3 | 3.31e-3 | 41 | -3.59e-3 | +2.50e-3 | -5.77e-4 | -1.21e-3 |
| 154 | 3.00e-3 | 6 | 3.08e-3 | 4.31e-3 | 3.54e-3 | 3.08e-3 | 38 | -3.56e-3 | +3.18e-3 | -8.46e-4 | -1.09e-3 |
| 155 | 3.00e-3 | 7 | 3.06e-3 | 4.00e-3 | 3.43e-3 | 3.53e-3 | 48 | -5.57e-3 | +3.19e-3 | -1.91e-4 | -5.70e-4 |
| 156 | 3.00e-3 | 5 | 3.42e-3 | 4.25e-3 | 3.65e-3 | 3.42e-3 | 38 | -3.72e-3 | +2.59e-3 | -7.24e-4 | -6.59e-4 |
| 157 | 3.00e-3 | 6 | 3.31e-3 | 4.37e-3 | 3.57e-3 | 3.31e-3 | 44 | -4.99e-3 | +2.75e-3 | -7.27e-4 | -6.84e-4 |
| 158 | 3.00e-3 | 5 | 3.51e-3 | 4.13e-3 | 3.68e-3 | 3.55e-3 | 43 | -2.94e-3 | +2.57e-3 | -1.57e-4 | -4.83e-4 |
| 159 | 3.00e-3 | 2 | 3.49e-3 | 6.85e-3 | 5.17e-3 | 6.85e-3 | 315 | -4.02e-4 | +2.14e-3 | +8.71e-4 | -2.13e-4 |
| 161 | 3.00e-3 | 1 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 8.38e-3 | 388 | +5.18e-4 | +5.18e-4 | +5.18e-4 | -1.40e-4 |
| 162 | 3.00e-3 | 1 | 8.28e-3 | 8.28e-3 | 8.28e-3 | 8.28e-3 | 316 | -3.62e-5 | -3.62e-5 | -3.62e-5 | -1.29e-4 |
| 163 | 3.00e-3 | 1 | 8.62e-3 | 8.62e-3 | 8.62e-3 | 8.62e-3 | 327 | +1.22e-4 | +1.22e-4 | +1.22e-4 | -1.04e-4 |
| 165 | 3.00e-3 | 2 | 8.70e-3 | 9.44e-3 | 9.07e-3 | 8.70e-3 | 281 | -2.89e-4 | +2.31e-4 | -2.90e-5 | -9.26e-5 |
| 167 | 3.00e-3 | 1 | 8.90e-3 | 8.90e-3 | 8.90e-3 | 8.90e-3 | 354 | +6.37e-5 | +6.37e-5 | +6.37e-5 | -7.70e-5 |
| 168 | 3.00e-3 | 1 | 8.78e-3 | 8.78e-3 | 8.78e-3 | 8.78e-3 | 319 | -4.47e-5 | -4.47e-5 | -4.47e-5 | -7.38e-5 |
| 169 | 3.00e-3 | 1 | 9.34e-3 | 9.34e-3 | 9.34e-3 | 9.34e-3 | 376 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -4.99e-5 |
| 170 | 3.00e-3 | 1 | 8.99e-3 | 8.99e-3 | 8.99e-3 | 8.99e-3 | 311 | -1.22e-4 | -1.22e-4 | -1.22e-4 | -5.71e-5 |
| 171 | 3.00e-3 | 1 | 8.65e-3 | 8.65e-3 | 8.65e-3 | 8.65e-3 | 277 | -1.41e-4 | -1.41e-4 | -1.41e-4 | -6.55e-5 |
| 172 | 3.00e-3 | 1 | 8.78e-3 | 8.78e-3 | 8.78e-3 | 8.78e-3 | 279 | +5.49e-5 | +5.49e-5 | +5.49e-5 | -5.35e-5 |
| 173 | 3.00e-3 | 1 | 8.52e-3 | 8.52e-3 | 8.52e-3 | 8.52e-3 | 270 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -5.91e-5 |
| 174 | 3.00e-3 | 1 | 8.54e-3 | 8.54e-3 | 8.54e-3 | 8.54e-3 | 273 | +5.55e-6 | +5.55e-6 | +5.55e-6 | -5.26e-5 |
| 175 | 3.00e-3 | 1 | 8.61e-3 | 8.61e-3 | 8.61e-3 | 8.61e-3 | 260 | +3.32e-5 | +3.32e-5 | +3.32e-5 | -4.40e-5 |
| 176 | 3.00e-3 | 1 | 8.52e-3 | 8.52e-3 | 8.52e-3 | 8.52e-3 | 257 | -4.04e-5 | -4.04e-5 | -4.04e-5 | -4.37e-5 |
| 177 | 3.00e-3 | 1 | 9.09e-3 | 9.09e-3 | 9.09e-3 | 9.09e-3 | 302 | +2.12e-4 | +2.12e-4 | +2.12e-4 | -1.80e-5 |
| 178 | 3.00e-3 | 1 | 8.75e-3 | 8.75e-3 | 8.75e-3 | 8.75e-3 | 262 | -1.44e-4 | -1.44e-4 | -1.44e-4 | -3.07e-5 |
| 179 | 3.00e-3 | 1 | 8.72e-3 | 8.72e-3 | 8.72e-3 | 8.72e-3 | 291 | -1.01e-5 | -1.01e-5 | -1.01e-5 | -2.86e-5 |
| 180 | 3.00e-3 | 1 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 8.63e-3 | 273 | -3.83e-5 | -3.83e-5 | -3.83e-5 | -2.96e-5 |
| 181 | 3.00e-3 | 1 | 8.61e-3 | 8.61e-3 | 8.61e-3 | 8.61e-3 | 260 | -1.11e-5 | -1.11e-5 | -1.11e-5 | -2.77e-5 |
| 182 | 3.00e-3 | 1 | 8.56e-3 | 8.56e-3 | 8.56e-3 | 8.56e-3 | 256 | -2.04e-5 | -2.04e-5 | -2.04e-5 | -2.70e-5 |
| 183 | 3.00e-3 | 1 | 9.01e-3 | 9.01e-3 | 9.01e-3 | 9.01e-3 | 291 | +1.74e-4 | +1.74e-4 | +1.74e-4 | -6.90e-6 |
| 184 | 3.00e-3 | 1 | 8.93e-3 | 8.93e-3 | 8.93e-3 | 8.93e-3 | 287 | -3.09e-5 | -3.09e-5 | -3.09e-5 | -9.30e-6 |
| 185 | 3.00e-3 | 1 | 8.77e-3 | 8.77e-3 | 8.77e-3 | 8.77e-3 | 257 | -7.10e-5 | -7.10e-5 | -7.10e-5 | -1.55e-5 |
| 186 | 3.00e-3 | 1 | 9.19e-3 | 9.19e-3 | 9.19e-3 | 9.19e-3 | 306 | +1.53e-4 | +1.53e-4 | +1.53e-4 | +1.40e-6 |
| 187 | 3.00e-3 | 1 | 9.00e-3 | 9.00e-3 | 9.00e-3 | 9.00e-3 | 271 | -7.70e-5 | -7.70e-5 | -7.70e-5 | -6.44e-6 |
| 188 | 3.00e-3 | 1 | 8.84e-3 | 8.84e-3 | 8.84e-3 | 8.84e-3 | 260 | -6.71e-5 | -6.71e-5 | -6.71e-5 | -1.25e-5 |
| 189 | 3.00e-3 | 1 | 8.93e-3 | 8.93e-3 | 8.93e-3 | 8.93e-3 | 271 | +3.62e-5 | +3.62e-5 | +3.62e-5 | -7.64e-6 |
| 190 | 3.00e-3 | 1 | 8.89e-3 | 8.89e-3 | 8.89e-3 | 8.89e-3 | 282 | -1.65e-5 | -1.65e-5 | -1.65e-5 | -8.53e-6 |
| 191 | 3.00e-3 | 1 | 9.10e-3 | 9.10e-3 | 9.10e-3 | 9.10e-3 | 274 | +8.51e-5 | +8.51e-5 | +8.51e-5 | +8.27e-7 |
| 192 | 3.00e-3 | 1 | 9.15e-3 | 9.15e-3 | 9.15e-3 | 9.15e-3 | 285 | +1.87e-5 | +1.87e-5 | +1.87e-5 | +2.62e-6 |
| 193 | 3.00e-3 | 1 | 9.27e-3 | 9.27e-3 | 9.27e-3 | 9.27e-3 | 307 | +4.46e-5 | +4.46e-5 | +4.46e-5 | +6.81e-6 |
| 194 | 3.00e-3 | 1 | 9.63e-3 | 9.63e-3 | 9.63e-3 | 9.63e-3 | 313 | +1.21e-4 | +1.21e-4 | +1.21e-4 | +1.82e-5 |
| 195 | 3.00e-3 | 1 | 9.47e-3 | 9.47e-3 | 9.47e-3 | 9.47e-3 | 293 | -5.49e-5 | -5.49e-5 | -5.49e-5 | +1.09e-5 |
| 196 | 3.00e-3 | 1 | 9.12e-3 | 9.12e-3 | 9.12e-3 | 9.12e-3 | 258 | -1.47e-4 | -1.47e-4 | -1.47e-4 | -4.89e-6 |
| 197 | 3.00e-3 | 1 | 9.22e-3 | 9.22e-3 | 9.22e-3 | 9.22e-3 | 263 | +4.09e-5 | +4.09e-5 | +4.09e-5 | -3.17e-7 |
| 198 | 3.00e-3 | 1 | 9.01e-3 | 9.01e-3 | 9.01e-3 | 9.01e-3 | 248 | -9.50e-5 | -9.50e-5 | -9.50e-5 | -9.79e-6 |
| 199 | 3.00e-3 | 1 | 8.75e-3 | 8.75e-3 | 8.75e-3 | 8.75e-3 | 238 | -1.19e-4 | -1.19e-4 | -1.19e-4 | -2.08e-5 |

