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
| cpu-async | 0.061986 | 0.9173 | +0.0048 | 1754.4 | 638 | 77.4 | 100% | 100% | 99% | 14.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9173 | cpu-async | - | - |

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
| cpu-async | 1.9972 | 0.7529 | 0.5755 | 0.5178 | 0.5251 | 0.5069 | 0.4856 | 0.4852 | 0.4769 | 0.4753 | 0.2183 | 0.1804 | 0.1613 | 0.1486 | 0.1308 | 0.0769 | 0.0741 | 0.0675 | 0.0651 | 0.0620 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4013 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3018 | 3.8 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2969 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 391 | 396 | 392 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 331.7 | 2.8 | epoch-boundary(37) |
| cpu-async | gpu2 | 331.8 | 2.8 | epoch-boundary(37) |
| cpu-async | gpu2 | 558.3 | 1.6 | epoch-boundary(63) |
| cpu-async | gpu2 | 351.5 | 1.4 | epoch-boundary(39) |
| cpu-async | gpu2 | 983.1 | 0.9 | epoch-boundary(112) |
| cpu-async | gpu2 | 740.7 | 0.9 | epoch-boundary(84) |
| cpu-async | gpu1 | 983.2 | 0.8 | epoch-boundary(112) |
| cpu-async | gpu1 | 1753.8 | 0.5 | epoch-boundary(199) |
| cpu-async | gpu2 | 1753.8 | 0.5 | epoch-boundary(199) |
| cpu-async | gpu2 | 1037.3 | 0.5 | epoch-boundary(118) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.5s | 0.0s | 0.5s |
| resnet-graph | cpu-async | gpu1 | 4.2s | 0.0s | 0.0s | 0.0s | 4.8s |
| resnet-graph | cpu-async | gpu2 | 8.7s | 0.0s | 0.0s | 0.0s | 9.3s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 334 | 0 | 638 | 77.4 | 1379/8789 | 638 | 77.4 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 174.4 | 9.9% |

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
| resnet-graph | cpu-async | 192 | 638 | 0 | 8.15e-3 | -8.87e-6 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 638 | 7.42e-2 | 7.17e-2 | 2.17e-3 | 4.63e-1 | 30.9 | -2.08e-4 | 1.23e-3 |
| resnet-graph | cpu-async | 1 | 638 | 7.48e-2 | 7.26e-2 | 2.20e-3 | 4.31e-1 | 32.8 | -2.35e-4 | 1.40e-3 |
| resnet-graph | cpu-async | 2 | 638 | 7.45e-2 | 7.21e-2 | 2.25e-3 | 3.91e-1 | 36.4 | -2.16e-4 | 1.36e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9952 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9948 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9953 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 47 (0,1,2,3,4,5,8,9…145,147) | 0 (—) | — | 0,1,2,3,4,5,8,9…145,147 | — |

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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 301 | +0.088 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 113 | -0.005 |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | 220 | +0.064 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 636 | +0.010 | 192 | +0.316 | +0.458 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 637 | 3.39e1–7.99e1 | 6.46e1 | 1.92e-3 | 3.53e-3 | 4.11e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 303 | 31–77853 | +9.412e-6 | 0.441 | +9.601e-6 | 0.474 | 93 | +9.968e-6 | 0.649 | 30–942 | +9.232e-4 | 0.756 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 292 | 948–77853 | +1.045e-5 | 0.595 | +1.062e-5 | 0.630 | 92 | +1.027e-5 | 0.668 | 81–942 | +9.717e-4 | 0.938 |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | 114 | 78486–117072 | -5.822e-8 | 0.000 | -3.696e-7 | 0.000 | 50 | -5.137e-7 | 0.000 | 118–690 | +2.477e-4 | 0.023 |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | 221 | 117289–155381 | +1.773e-5 | 0.180 | +1.816e-5 | 0.191 | 49 | +3.154e-5 | 0.610 | 78–930 | +1.528e-3 | 0.528 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +9.232e-4 | r0: +9.121e-4, r1: +9.240e-4, r2: +9.341e-4 | r0: 0.751, r1: 0.723, r2: 0.757 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +9.717e-4 | r0: +9.613e-4, r1: +9.739e-4, r2: +9.801e-4 | r0: 0.932, r1: 0.907, r2: 0.926 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–149 | +2.477e-4 | r0: +2.001e-4, r1: +2.773e-4, r2: +2.670e-4 | r0: 0.015, r1: 0.028, r2: 0.025 | 1.39× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 150–198 | +1.528e-3 | r0: +1.499e-3, r1: +1.535e-3, r2: +1.550e-3 | r0: 0.505, r1: 0.529, r2: 0.535 | 1.03× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇██████████████▇▄▄▅▅▅▅▅▅▅▅▅▅▁▁▁▁▁▁▁▂▂▂▂▂` | `▁▇▇▇▇▇▇▇▇▇█████████████▇▇▇████████▇▅▅▇▇▇▇▇▇█████` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 1.04e-1 | 4.63e-1 | 2.00e-1 | 1.34e-1 | 46 | -3.84e-2 | +1.16e-2 | -1.21e-2 | -8.69e-3 |
| 1 | 3.00e-1 | 8 | 1.03e-1 | 1.59e-1 | 1.13e-1 | 1.04e-1 | 41 | -7.20e-3 | +1.71e-3 | -9.24e-4 | -3.36e-3 |
| 2 | 3.00e-1 | 3 | 1.02e-1 | 1.51e-1 | 1.19e-1 | 1.02e-1 | 36 | -1.10e-2 | +4.02e-3 | -2.45e-3 | -3.13e-3 |
| 3 | 3.00e-1 | 7 | 1.10e-1 | 1.40e-1 | 1.22e-1 | 1.26e-1 | 49 | -5.23e-3 | +4.08e-3 | +1.08e-4 | -1.32e-3 |
| 4 | 3.00e-1 | 5 | 1.22e-1 | 1.63e-1 | 1.35e-1 | 1.32e-1 | 48 | -5.06e-3 | +2.96e-3 | -3.76e-4 | -9.03e-4 |
| 5 | 3.00e-1 | 6 | 1.18e-1 | 1.64e-1 | 1.34e-1 | 1.18e-1 | 36 | -3.71e-3 | +2.68e-3 | -8.11e-4 | -9.04e-4 |
| 6 | 3.00e-1 | 6 | 1.16e-1 | 1.58e-1 | 1.26e-1 | 1.21e-1 | 40 | -6.75e-3 | +3.78e-3 | -5.56e-4 | -7.10e-4 |
| 7 | 3.00e-1 | 7 | 1.12e-1 | 1.58e-1 | 1.24e-1 | 1.16e-1 | 39 | -5.51e-3 | +3.34e-3 | -7.00e-4 | -6.75e-4 |
| 8 | 3.00e-1 | 7 | 1.11e-1 | 1.55e-1 | 1.21e-1 | 1.18e-1 | 40 | -8.25e-3 | +3.92e-3 | -6.67e-4 | -5.90e-4 |
| 9 | 3.00e-1 | 7 | 1.07e-1 | 1.55e-1 | 1.18e-1 | 1.09e-1 | 33 | -4.86e-3 | +3.43e-3 | -7.73e-4 | -6.46e-4 |
| 10 | 3.00e-1 | 7 | 1.12e-1 | 1.52e-1 | 1.22e-1 | 1.19e-1 | 42 | -9.05e-3 | +4.31e-3 | -5.48e-4 | -5.39e-4 |
| 11 | 3.00e-1 | 6 | 1.18e-1 | 1.52e-1 | 1.26e-1 | 1.20e-1 | 45 | -5.01e-3 | +3.05e-3 | -4.26e-4 | -4.79e-4 |
| 12 | 3.00e-1 | 5 | 1.20e-1 | 1.58e-1 | 1.30e-1 | 1.20e-1 | 43 | -4.89e-3 | +3.15e-3 | -6.03e-4 | -5.45e-4 |
| 13 | 3.00e-1 | 8 | 1.14e-1 | 1.54e-1 | 1.22e-1 | 1.20e-1 | 44 | -6.91e-3 | +3.04e-3 | -4.48e-4 | -4.19e-4 |
| 14 | 3.00e-1 | 5 | 1.14e-1 | 1.50e-1 | 1.23e-1 | 1.14e-1 | 40 | -5.57e-3 | +2.92e-3 | -7.90e-4 | -5.70e-4 |
| 15 | 3.00e-1 | 6 | 1.11e-1 | 1.50e-1 | 1.21e-1 | 1.18e-1 | 41 | -5.79e-3 | +3.54e-3 | -4.39e-4 | -4.81e-4 |
| 16 | 3.00e-1 | 7 | 1.13e-1 | 1.54e-1 | 1.22e-1 | 1.18e-1 | 40 | -6.88e-3 | +3.38e-3 | -5.64e-4 | -4.64e-4 |
| 17 | 3.00e-1 | 6 | 1.13e-1 | 1.51e-1 | 1.21e-1 | 1.13e-1 | 40 | -5.87e-3 | +3.00e-3 | -6.93e-4 | -5.54e-4 |
| 18 | 3.00e-1 | 6 | 1.15e-1 | 1.64e-1 | 1.26e-1 | 1.16e-1 | 40 | -6.93e-3 | +4.18e-3 | -7.57e-4 | -6.41e-4 |
| 19 | 3.00e-1 | 7 | 1.07e-1 | 1.47e-1 | 1.17e-1 | 1.07e-1 | 35 | -5.25e-3 | +3.53e-3 | -6.73e-4 | -6.62e-4 |
| 20 | 3.00e-1 | 7 | 1.05e-1 | 1.49e-1 | 1.17e-1 | 1.05e-1 | 30 | -5.92e-3 | +4.23e-3 | -7.35e-4 | -7.38e-4 |
| 21 | 3.00e-1 | 5 | 1.15e-1 | 1.39e-1 | 1.29e-1 | 1.34e-1 | 56 | -5.16e-3 | +4.17e-3 | +3.37e-4 | -3.12e-4 |
| 22 | 3.00e-1 | 5 | 1.24e-1 | 1.64e-1 | 1.37e-1 | 1.24e-1 | 51 | -2.96e-3 | +2.18e-3 | -5.55e-4 | -4.37e-4 |
| 23 | 3.00e-1 | 7 | 1.15e-1 | 1.58e-1 | 1.27e-1 | 1.21e-1 | 44 | -4.15e-3 | +2.66e-3 | -4.16e-4 | -4.02e-4 |
| 24 | 3.00e-1 | 4 | 1.20e-1 | 1.58e-1 | 1.32e-1 | 1.20e-1 | 44 | -4.03e-3 | +3.03e-3 | -6.97e-4 | -5.29e-4 |
| 25 | 3.00e-1 | 6 | 1.13e-1 | 1.61e-1 | 1.23e-1 | 1.13e-1 | 35 | -7.36e-3 | +3.15e-3 | -9.65e-4 | -7.08e-4 |
| 26 | 3.00e-1 | 8 | 1.10e-1 | 1.53e-1 | 1.18e-1 | 1.13e-1 | 37 | -8.81e-3 | +3.66e-3 | -6.64e-4 | -6.26e-4 |
| 27 | 3.00e-1 | 5 | 1.08e-1 | 1.46e-1 | 1.17e-1 | 1.10e-1 | 37 | -6.66e-3 | +3.42e-3 | -8.83e-4 | -7.19e-4 |
| 28 | 3.00e-1 | 7 | 1.06e-1 | 1.51e-1 | 1.17e-1 | 1.10e-1 | 38 | -6.93e-3 | +4.12e-3 | -7.46e-4 | -7.00e-4 |
| 29 | 3.00e-1 | 8 | 1.02e-1 | 1.49e-1 | 1.13e-1 | 1.12e-1 | 35 | -1.24e-2 | +4.36e-3 | -8.02e-4 | -6.08e-4 |
| 30 | 3.00e-1 | 6 | 1.14e-1 | 1.50e-1 | 1.22e-1 | 1.14e-1 | 38 | -6.51e-3 | +3.59e-3 | -6.52e-4 | -6.24e-4 |
| 31 | 3.00e-1 | 6 | 1.19e-1 | 1.55e-1 | 1.29e-1 | 1.27e-1 | 47 | -5.60e-3 | +3.82e-3 | -1.91e-4 | -4.04e-4 |
| 32 | 3.00e-1 | 6 | 1.14e-1 | 1.56e-1 | 1.25e-1 | 1.14e-1 | 40 | -5.52e-3 | +2.46e-3 | -8.08e-4 | -5.87e-4 |
| 33 | 3.00e-1 | 7 | 1.09e-1 | 1.59e-1 | 1.22e-1 | 1.13e-1 | 38 | -5.35e-3 | +4.24e-3 | -6.39e-4 | -5.95e-4 |
| 34 | 3.00e-1 | 8 | 1.11e-1 | 1.50e-1 | 1.25e-1 | 1.38e-1 | 55 | -5.32e-3 | +3.79e-3 | +5.44e-5 | -1.31e-4 |
| 35 | 3.00e-1 | 4 | 1.09e-1 | 1.63e-1 | 1.26e-1 | 1.09e-1 | 34 | -7.54e-3 | +1.85e-3 | -2.39e-3 | -9.10e-4 |
| 36 | 3.00e-1 | 6 | 1.08e-1 | 1.56e-1 | 1.19e-1 | 1.14e-1 | 41 | -9.82e-3 | +4.39e-3 | -8.79e-4 | -8.39e-4 |
| 37 | 3.00e-1 | 2 | 1.13e-1 | 2.29e-1 | 1.71e-1 | 2.29e-1 | 293 | -1.12e-4 | +2.40e-3 | +1.15e-3 | -4.49e-4 |
| 39 | 3.00e-1 | 2 | 2.27e-1 | 2.37e-1 | 2.32e-1 | 2.27e-1 | 276 | -1.51e-4 | +1.20e-4 | -1.58e-5 | -3.68e-4 |
| 41 | 3.00e-1 | 2 | 2.22e-1 | 2.35e-1 | 2.28e-1 | 2.22e-1 | 276 | -2.11e-4 | +9.20e-5 | -5.93e-5 | -3.11e-4 |
| 43 | 3.00e-1 | 1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 2.36e-1 | 346 | +1.79e-4 | +1.79e-4 | +1.79e-4 | -2.62e-4 |
| 44 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 298 | -1.80e-4 | -1.80e-4 | -1.80e-4 | -2.54e-4 |
| 45 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 297 | +5.94e-6 | +5.94e-6 | +5.94e-6 | -2.28e-4 |
| 46 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 300 | +1.67e-5 | +1.67e-5 | +1.67e-5 | -2.04e-4 |
| 47 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 268 | -1.38e-4 | -1.38e-4 | -1.38e-4 | -1.97e-4 |
| 48 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 301 | +6.54e-5 | +6.54e-5 | +6.54e-5 | -1.71e-4 |
| 49 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 268 | -1.03e-4 | -1.03e-4 | -1.03e-4 | -1.64e-4 |
| 50 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 275 | +9.39e-5 | +9.39e-5 | +9.39e-5 | -1.38e-4 |
| 51 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 254 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -1.35e-4 |
| 52 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 273 | +7.76e-5 | +7.76e-5 | +7.76e-5 | -1.14e-4 |
| 53 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 283 | -1.48e-5 | -1.48e-5 | -1.48e-5 | -1.04e-4 |
| 54 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 272 | -4.61e-5 | -4.61e-5 | -4.61e-5 | -9.80e-5 |
| 55 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 261 | +5.96e-6 | +5.96e-6 | +5.96e-6 | -8.76e-5 |
| 56 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 296 | +6.43e-5 | +6.43e-5 | +6.43e-5 | -7.24e-5 |
| 57 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 319 | +9.69e-5 | +9.69e-5 | +9.69e-5 | -5.55e-5 |
| 58 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 282 | -1.46e-4 | -1.46e-4 | -1.46e-4 | -6.45e-5 |
| 59 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 271 | -2.82e-5 | -2.82e-5 | -2.82e-5 | -6.09e-5 |
| 60 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 294 | +9.87e-5 | +9.87e-5 | +9.87e-5 | -4.49e-5 |
| 61 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 315 | +8.45e-5 | +8.45e-5 | +8.45e-5 | -3.20e-5 |
| 63 | 3.00e-1 | 2 | 2.23e-1 | 2.37e-1 | 2.30e-1 | 2.23e-1 | 266 | -2.20e-4 | +9.81e-5 | -6.07e-5 | -3.90e-5 |
| 65 | 3.00e-1 | 2 | 2.15e-1 | 2.37e-1 | 2.26e-1 | 2.15e-1 | 251 | -3.86e-4 | +1.82e-4 | -1.02e-4 | -5.38e-5 |
| 67 | 3.00e-1 | 2 | 2.10e-1 | 2.35e-1 | 2.23e-1 | 2.10e-1 | 231 | -4.94e-4 | +2.78e-4 | -1.08e-4 | -6.79e-5 |
| 68 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 244 | +2.22e-5 | +2.22e-5 | +2.22e-5 | -5.89e-5 |
| 69 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 288 | +2.31e-4 | +2.31e-4 | +2.31e-4 | -2.99e-5 |
| 70 | 3.00e-1 | 1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 2.16e-1 | 254 | -1.67e-4 | -1.67e-4 | -1.67e-4 | -4.36e-5 |
| 71 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 277 | +1.18e-4 | +1.18e-4 | +1.18e-4 | -2.74e-5 |
| 72 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 261 | -1.09e-4 | -1.09e-4 | -1.09e-4 | -3.56e-5 |
| 73 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 239 | -9.39e-5 | -9.39e-5 | -9.39e-5 | -4.14e-5 |
| 74 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 252 | +8.01e-5 | +8.01e-5 | +8.01e-5 | -2.93e-5 |
| 75 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 264 | +2.13e-5 | +2.13e-5 | +2.13e-5 | -2.42e-5 |
| 76 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 304 | +1.33e-4 | +1.33e-4 | +1.33e-4 | -8.52e-6 |
| 77 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 244 | -3.05e-4 | -3.05e-4 | -3.05e-4 | -3.81e-5 |
| 78 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 224 | -1.05e-4 | -1.05e-4 | -1.05e-4 | -4.48e-5 |
| 79 | 3.00e-1 | 2 | 2.05e-1 | 2.09e-1 | 2.07e-1 | 2.05e-1 | 218 | -7.90e-5 | +6.38e-5 | -7.60e-6 | -3.85e-5 |
| 80 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 243 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -1.59e-5 |
| 81 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 258 | +5.07e-5 | +5.07e-5 | +5.07e-5 | -9.27e-6 |
| 82 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 253 | -4.60e-5 | -4.60e-5 | -4.60e-5 | -1.29e-5 |
| 83 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 216 | -1.87e-4 | -1.87e-4 | -1.87e-4 | -3.03e-5 |
| 84 | 3.00e-1 | 2 | 2.03e-1 | 2.12e-1 | 2.08e-1 | 2.03e-1 | 207 | -2.08e-4 | +1.14e-4 | -4.71e-5 | -3.51e-5 |
| 85 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 244 | +2.35e-4 | +2.35e-4 | +2.35e-4 | -8.10e-6 |
| 86 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 241 | -5.99e-5 | -5.99e-5 | -5.99e-5 | -1.33e-5 |
| 87 | 3.00e-1 | 2 | 2.03e-1 | 2.14e-1 | 2.08e-1 | 2.03e-1 | 191 | -2.85e-4 | +3.29e-5 | -1.26e-4 | -3.63e-5 |
| 89 | 3.00e-1 | 2 | 2.06e-1 | 2.23e-1 | 2.15e-1 | 2.06e-1 | 216 | -3.79e-4 | +3.59e-4 | -9.89e-6 | -3.50e-5 |
| 90 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 238 | +9.84e-5 | +9.84e-5 | +9.84e-5 | -2.16e-5 |
| 91 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 241 | +7.23e-5 | +7.23e-5 | +7.23e-5 | -1.22e-5 |
| 92 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 251 | -3.29e-6 | -3.29e-6 | -3.29e-6 | -1.13e-5 |
| 93 | 3.00e-1 | 2 | 1.99e-1 | 2.21e-1 | 2.10e-1 | 1.99e-1 | 199 | -5.46e-4 | +1.22e-4 | -2.12e-4 | -5.28e-5 |
| 94 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 232 | +2.69e-4 | +2.69e-4 | +2.69e-4 | -2.06e-5 |
| 95 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 201 | -2.02e-4 | -2.02e-4 | -2.02e-4 | -3.88e-5 |
| 96 | 3.00e-1 | 1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 2.11e-1 | 237 | +1.63e-4 | +1.63e-4 | +1.63e-4 | -1.86e-5 |
| 97 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 232 | -4.38e-5 | -4.38e-5 | -4.38e-5 | -2.11e-5 |
| 98 | 3.00e-1 | 2 | 2.07e-1 | 2.16e-1 | 2.11e-1 | 2.07e-1 | 196 | -2.01e-4 | +1.23e-4 | -3.91e-5 | -2.61e-5 |
| 99 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 243 | +9.59e-6 | +9.59e-6 | +9.59e-6 | -2.26e-5 |
| 100 | 3.00e-2 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 226 | +1.70e-5 | +1.70e-5 | +1.70e-5 | -1.86e-5 |
| 101 | 3.00e-2 | 2 | 5.48e-2 | 1.05e-1 | 7.98e-2 | 5.48e-2 | 196 | -3.31e-3 | -2.98e-3 | -3.14e-3 | -6.14e-4 |
| 102 | 3.00e-2 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 251 | -1.79e-3 | -1.79e-3 | -1.79e-3 | -7.31e-4 |
| 103 | 3.00e-2 | 1 | 2.66e-2 | 2.66e-2 | 2.66e-2 | 2.66e-2 | 208 | -1.32e-3 | -1.32e-3 | -1.32e-3 | -7.90e-4 |
| 104 | 3.00e-2 | 2 | 2.32e-2 | 2.47e-2 | 2.39e-2 | 2.32e-2 | 168 | -3.78e-4 | -3.63e-4 | -3.71e-4 | -7.10e-4 |
| 105 | 3.00e-2 | 1 | 2.44e-2 | 2.44e-2 | 2.44e-2 | 2.44e-2 | 200 | +2.47e-4 | +2.47e-4 | +2.47e-4 | -6.15e-4 |
| 106 | 3.00e-2 | 2 | 2.43e-2 | 2.44e-2 | 2.44e-2 | 2.43e-2 | 168 | -3.35e-5 | +1.03e-5 | -1.16e-5 | -5.00e-4 |
| 107 | 3.00e-2 | 1 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 2.76e-2 | 221 | +5.83e-4 | +5.83e-4 | +5.83e-4 | -3.92e-4 |
| 108 | 3.00e-2 | 2 | 2.54e-2 | 2.74e-2 | 2.64e-2 | 2.54e-2 | 156 | -4.89e-4 | -3.42e-5 | -2.62e-4 | -3.70e-4 |
| 109 | 3.00e-2 | 1 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 2.78e-2 | 205 | +4.42e-4 | +4.42e-4 | +4.42e-4 | -2.88e-4 |
| 110 | 3.00e-2 | 2 | 2.69e-2 | 2.83e-2 | 2.76e-2 | 2.69e-2 | 156 | -3.38e-4 | +8.76e-5 | -1.25e-4 | -2.60e-4 |
| 111 | 3.00e-2 | 1 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 2.77e-2 | 182 | +1.65e-4 | +1.65e-4 | +1.65e-4 | -2.17e-4 |
| 112 | 3.00e-2 | 2 | 2.81e-2 | 3.10e-2 | 2.96e-2 | 2.81e-2 | 156 | -6.20e-4 | +5.10e-4 | -5.49e-5 | -1.92e-4 |
| 113 | 3.00e-2 | 1 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 | 198 | +4.32e-4 | +4.32e-4 | +4.32e-4 | -1.29e-4 |
| 114 | 3.00e-2 | 2 | 2.82e-2 | 3.06e-2 | 2.94e-2 | 2.82e-2 | 133 | -6.07e-4 | -5.87e-6 | -3.07e-4 | -1.66e-4 |
| 115 | 3.00e-2 | 2 | 2.88e-2 | 2.96e-2 | 2.92e-2 | 2.96e-2 | 148 | +1.24e-4 | +1.94e-4 | +1.59e-4 | -1.04e-4 |
| 116 | 3.00e-2 | 1 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 3.12e-2 | 192 | +2.73e-4 | +2.73e-4 | +2.73e-4 | -6.63e-5 |
| 117 | 3.00e-2 | 2 | 3.21e-2 | 3.25e-2 | 3.23e-2 | 3.21e-2 | 170 | -8.12e-5 | +2.09e-4 | +6.40e-5 | -4.30e-5 |
| 118 | 3.00e-2 | 2 | 3.25e-2 | 3.43e-2 | 3.34e-2 | 3.25e-2 | 159 | -3.24e-4 | +3.13e-4 | -5.57e-6 | -3.90e-5 |
| 119 | 3.00e-2 | 1 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 3.45e-2 | 192 | +3.11e-4 | +3.11e-4 | +3.11e-4 | -4.07e-6 |
| 120 | 3.00e-2 | 2 | 3.23e-2 | 3.38e-2 | 3.31e-2 | 3.23e-2 | 145 | -3.09e-4 | -1.19e-4 | -2.14e-4 | -4.49e-5 |
| 121 | 3.00e-2 | 2 | 3.30e-2 | 3.52e-2 | 3.41e-2 | 3.30e-2 | 145 | -4.44e-4 | +4.45e-4 | +3.71e-7 | -4.08e-5 |
| 122 | 3.00e-2 | 1 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 3.61e-2 | 196 | +4.60e-4 | +4.60e-4 | +4.60e-4 | +9.35e-6 |
| 123 | 3.00e-2 | 2 | 3.47e-2 | 3.62e-2 | 3.54e-2 | 3.47e-2 | 155 | -2.73e-4 | +9.00e-6 | -1.32e-4 | -1.89e-5 |
| 124 | 3.00e-2 | 1 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 3.70e-2 | 179 | +3.67e-4 | +3.67e-4 | +3.67e-4 | +1.96e-5 |
| 125 | 3.00e-2 | 3 | 3.24e-2 | 3.83e-2 | 3.46e-2 | 3.24e-2 | 116 | -1.18e-3 | +1.74e-4 | -4.00e-4 | -9.73e-5 |
| 126 | 3.00e-2 | 1 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 3.71e-2 | 170 | +7.91e-4 | +7.91e-4 | +7.91e-4 | -8.48e-6 |
| 127 | 3.00e-2 | 3 | 3.35e-2 | 3.54e-2 | 3.43e-2 | 3.35e-2 | 116 | -3.73e-4 | -9.32e-5 | -2.56e-4 | -7.36e-5 |
| 128 | 3.00e-2 | 2 | 3.48e-2 | 3.68e-2 | 3.58e-2 | 3.48e-2 | 118 | -4.73e-4 | +6.01e-4 | +6.39e-5 | -5.28e-5 |
| 129 | 3.00e-2 | 2 | 3.33e-2 | 3.63e-2 | 3.48e-2 | 3.33e-2 | 109 | -7.90e-4 | +3.08e-4 | -2.41e-4 | -9.41e-5 |
| 130 | 3.00e-2 | 2 | 3.43e-2 | 3.68e-2 | 3.55e-2 | 3.43e-2 | 109 | -6.65e-4 | +6.82e-4 | +8.69e-6 | -8.13e-5 |
| 131 | 3.00e-2 | 3 | 3.39e-2 | 3.64e-2 | 3.49e-2 | 3.45e-2 | 109 | -6.37e-4 | +4.56e-4 | -1.51e-5 | -6.61e-5 |
| 132 | 3.00e-2 | 2 | 3.66e-2 | 3.96e-2 | 3.81e-2 | 3.66e-2 | 109 | -7.29e-4 | +8.78e-4 | +7.46e-5 | -4.74e-5 |
| 133 | 3.00e-2 | 2 | 3.73e-2 | 4.03e-2 | 3.88e-2 | 3.73e-2 | 115 | -6.81e-4 | +6.21e-4 | -2.99e-5 | -5.06e-5 |
| 134 | 3.00e-2 | 2 | 3.72e-2 | 3.98e-2 | 3.85e-2 | 3.72e-2 | 115 | -5.90e-4 | +4.50e-4 | -6.97e-5 | -5.94e-5 |
| 135 | 3.00e-2 | 3 | 3.59e-2 | 4.06e-2 | 3.77e-2 | 3.59e-2 | 108 | -1.00e-3 | +5.68e-4 | -1.97e-4 | -1.03e-4 |
| 136 | 3.00e-2 | 2 | 3.83e-2 | 4.08e-2 | 3.95e-2 | 3.83e-2 | 108 | -6.01e-4 | +9.21e-4 | +1.60e-4 | -6.08e-5 |
| 137 | 3.00e-2 | 2 | 3.90e-2 | 4.11e-2 | 4.01e-2 | 3.90e-2 | 108 | -4.80e-4 | +5.05e-4 | +1.26e-5 | -5.18e-5 |
| 138 | 3.00e-2 | 4 | 3.59e-2 | 4.11e-2 | 3.78e-2 | 3.64e-2 | 95 | -8.44e-4 | +3.73e-4 | -2.13e-4 | -1.08e-4 |
| 139 | 3.00e-2 | 2 | 3.68e-2 | 3.95e-2 | 3.82e-2 | 3.68e-2 | 90 | -7.89e-4 | +6.95e-4 | -4.69e-5 | -1.04e-4 |
| 140 | 3.00e-2 | 2 | 3.74e-2 | 4.06e-2 | 3.90e-2 | 3.74e-2 | 90 | -8.93e-4 | +8.19e-4 | -3.66e-5 | -9.97e-5 |
| 141 | 3.00e-2 | 5 | 3.52e-2 | 4.13e-2 | 3.75e-2 | 3.52e-2 | 78 | -1.31e-3 | +7.65e-4 | -2.32e-4 | -1.65e-4 |
| 142 | 3.00e-2 | 2 | 3.34e-2 | 4.02e-2 | 3.68e-2 | 3.34e-2 | 68 | -2.75e-3 | +1.30e-3 | -7.22e-4 | -2.91e-4 |
| 143 | 3.00e-2 | 4 | 3.15e-2 | 3.76e-2 | 3.38e-2 | 3.15e-2 | 64 | -2.16e-3 | +1.19e-3 | -4.56e-4 | -3.63e-4 |
| 144 | 3.00e-2 | 4 | 3.31e-2 | 3.85e-2 | 3.50e-2 | 3.51e-2 | 68 | -2.17e-3 | +1.93e-3 | +1.35e-4 | -1.94e-4 |
| 145 | 3.00e-2 | 3 | 3.36e-2 | 4.32e-2 | 3.74e-2 | 3.36e-2 | 57 | -3.03e-3 | +1.62e-3 | -7.51e-4 | -3.68e-4 |
| 146 | 3.00e-2 | 5 | 3.03e-2 | 3.84e-2 | 3.22e-2 | 3.04e-2 | 51 | -4.06e-3 | +1.38e-3 | -5.91e-4 | -4.47e-4 |
| 147 | 3.00e-2 | 5 | 3.09e-2 | 3.89e-2 | 3.31e-2 | 3.09e-2 | 50 | -3.54e-3 | +2.78e-3 | -3.00e-4 | -4.09e-4 |
| 148 | 3.00e-2 | 7 | 2.77e-2 | 3.85e-2 | 3.09e-2 | 2.77e-2 | 44 | -4.24e-3 | +2.46e-3 | -6.64e-4 | -5.52e-4 |
| 149 | 3.00e-2 | 4 | 3.04e-2 | 3.65e-2 | 3.24e-2 | 3.04e-2 | 47 | -2.83e-3 | +3.52e-3 | -9.53e-5 | -4.34e-4 |
| 150 | 3.00e-3 | 6 | 3.05e-3 | 4.05e-2 | 1.37e-2 | 3.05e-3 | 38 | -1.66e-2 | +3.22e-3 | -9.40e-3 | -4.68e-3 |
| 151 | 3.00e-3 | 6 | 2.77e-3 | 3.55e-3 | 2.93e-3 | 2.85e-3 | 45 | -6.33e-3 | +1.98e-3 | -6.55e-4 | -2.74e-3 |
| 152 | 3.00e-3 | 6 | 2.61e-3 | 3.75e-3 | 2.92e-3 | 2.81e-3 | 41 | -5.80e-3 | +3.40e-3 | -6.03e-4 | -1.69e-3 |
| 153 | 3.00e-3 | 9 | 2.47e-3 | 3.84e-3 | 2.77e-3 | 2.66e-3 | 38 | -7.03e-3 | +3.45e-3 | -7.83e-4 | -1.00e-3 |
| 154 | 3.00e-3 | 5 | 2.29e-3 | 3.46e-3 | 2.70e-3 | 2.29e-3 | 31 | -5.60e-3 | +3.17e-3 | -1.79e-3 | -1.37e-3 |
| 155 | 3.00e-3 | 6 | 2.42e-3 | 3.11e-3 | 2.74e-3 | 2.96e-3 | 41 | -9.62e-3 | +4.78e-3 | -8.38e-6 | -6.59e-4 |
| 156 | 3.00e-3 | 8 | 2.77e-3 | 3.62e-3 | 2.94e-3 | 2.79e-3 | 40 | -5.21e-3 | +2.40e-3 | -4.46e-4 | -5.10e-4 |
| 157 | 3.00e-3 | 4 | 2.81e-3 | 3.47e-3 | 3.03e-3 | 2.92e-3 | 43 | -4.50e-3 | +2.62e-3 | -4.22e-4 | -4.82e-4 |
| 158 | 3.00e-3 | 5 | 2.86e-3 | 3.69e-3 | 3.16e-3 | 3.05e-3 | 45 | -4.03e-3 | +2.71e-3 | -3.97e-4 | -4.50e-4 |
| 159 | 3.00e-3 | 6 | 2.78e-3 | 3.97e-3 | 3.07e-3 | 2.82e-3 | 40 | -7.30e-3 | +2.89e-3 | -1.04e-3 | -6.93e-4 |
| 160 | 3.00e-3 | 8 | 2.61e-3 | 3.82e-3 | 2.92e-3 | 2.76e-3 | 38 | -5.12e-3 | +3.52e-3 | -5.74e-4 | -5.71e-4 |
| 161 | 3.00e-3 | 5 | 2.69e-3 | 3.62e-3 | 2.99e-3 | 2.70e-3 | 39 | -3.15e-3 | +3.28e-3 | -7.95e-4 | -6.93e-4 |
| 162 | 3.00e-3 | 7 | 2.54e-3 | 3.61e-3 | 2.77e-3 | 2.67e-3 | 37 | -7.46e-3 | +3.40e-3 | -8.34e-4 | -6.72e-4 |
| 163 | 3.00e-3 | 6 | 2.65e-3 | 3.49e-3 | 2.96e-3 | 3.08e-3 | 42 | -5.99e-3 | +3.42e-3 | -9.95e-5 | -3.42e-4 |
| 164 | 3.00e-3 | 7 | 2.52e-3 | 3.77e-3 | 2.85e-3 | 2.52e-3 | 32 | -7.48e-3 | +2.56e-3 | -1.26e-3 | -7.89e-4 |
| 165 | 3.00e-3 | 8 | 2.34e-3 | 3.56e-3 | 2.63e-3 | 2.63e-3 | 32 | -1.01e-2 | +4.63e-3 | -7.41e-4 | -5.85e-4 |
| 166 | 3.00e-3 | 10 | 2.41e-3 | 3.64e-3 | 2.74e-3 | 2.69e-3 | 35 | -7.24e-3 | +4.59e-3 | -4.63e-4 | -3.86e-4 |
| 167 | 3.00e-3 | 5 | 2.50e-3 | 3.58e-3 | 2.88e-3 | 2.94e-3 | 38 | -9.42e-3 | +3.41e-3 | -8.35e-4 | -4.78e-4 |
| 168 | 3.00e-3 | 6 | 2.62e-3 | 4.10e-3 | 3.05e-3 | 2.72e-3 | 34 | -7.17e-3 | +4.22e-3 | -1.11e-3 | -7.59e-4 |
| 169 | 3.00e-3 | 9 | 2.53e-3 | 3.88e-3 | 2.85e-3 | 2.53e-3 | 36 | -7.78e-3 | +4.69e-3 | -8.43e-4 | -7.87e-4 |
| 170 | 3.00e-3 | 5 | 2.65e-3 | 3.70e-3 | 3.00e-3 | 2.85e-3 | 39 | -6.03e-3 | +5.05e-3 | -4.87e-4 | -6.70e-4 |
| 171 | 3.00e-3 | 8 | 2.32e-3 | 3.68e-3 | 2.62e-3 | 2.45e-3 | 31 | -1.24e-2 | +3.44e-3 | -1.41e-3 | -9.28e-4 |
| 172 | 3.00e-3 | 7 | 2.64e-3 | 3.91e-3 | 3.09e-3 | 3.05e-3 | 46 | -5.59e-3 | +4.86e-3 | +1.18e-4 | -4.41e-4 |
| 173 | 3.00e-3 | 6 | 2.76e-3 | 4.46e-3 | 3.24e-3 | 2.76e-3 | 35 | -7.76e-3 | +3.92e-3 | -1.32e-3 | -8.68e-4 |
| 174 | 3.00e-3 | 8 | 2.85e-3 | 3.77e-3 | 3.08e-3 | 2.87e-3 | 36 | -5.68e-3 | +4.05e-3 | -4.72e-4 | -6.60e-4 |
| 175 | 3.00e-3 | 4 | 3.01e-3 | 3.85e-3 | 3.29e-3 | 3.01e-3 | 44 | -6.51e-3 | +4.05e-3 | -7.27e-4 | -7.23e-4 |
| 176 | 3.00e-3 | 6 | 2.58e-3 | 4.30e-3 | 3.31e-3 | 3.63e-3 | 50 | -1.08e-2 | +4.01e-3 | -7.64e-4 | -5.81e-4 |
| 177 | 3.00e-3 | 5 | 3.25e-3 | 4.23e-3 | 3.57e-3 | 3.25e-3 | 47 | -3.37e-3 | +1.87e-3 | -6.93e-4 | -6.44e-4 |
| 178 | 3.00e-3 | 7 | 3.12e-3 | 4.23e-3 | 3.52e-3 | 4.11e-3 | 85 | -3.57e-3 | +3.30e-3 | +4.82e-5 | -1.92e-4 |
| 179 | 3.00e-3 | 2 | 4.58e-3 | 5.06e-3 | 4.82e-3 | 4.58e-3 | 85 | -1.17e-3 | +1.70e-3 | +2.67e-4 | -1.19e-4 |
| 180 | 3.00e-3 | 3 | 4.42e-3 | 5.12e-3 | 4.70e-3 | 4.42e-3 | 82 | -1.42e-3 | +9.54e-4 | -2.80e-4 | -1.75e-4 |
| 181 | 3.00e-3 | 3 | 4.10e-3 | 5.66e-3 | 4.73e-3 | 4.10e-3 | 71 | -3.41e-3 | +1.85e-3 | -9.02e-4 | -3.99e-4 |
| 182 | 3.00e-3 | 4 | 4.19e-3 | 4.78e-3 | 4.45e-3 | 4.35e-3 | 81 | -9.57e-4 | +1.49e-3 | +4.00e-5 | -2.61e-4 |
| 183 | 3.00e-3 | 3 | 3.93e-3 | 4.61e-3 | 4.21e-3 | 3.93e-3 | 62 | -1.71e-3 | +5.51e-4 | -5.93e-4 | -3.61e-4 |
| 184 | 3.00e-3 | 4 | 3.80e-3 | 4.88e-3 | 4.19e-3 | 4.07e-3 | 68 | -3.10e-3 | +2.11e-3 | -2.34e-4 | -3.19e-4 |
| 185 | 3.00e-3 | 1 | 4.03e-3 | 4.03e-3 | 4.03e-3 | 4.03e-3 | 62 | -1.64e-4 | -1.64e-4 | -1.64e-4 | -3.04e-4 |
| 186 | 3.00e-3 | 1 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 7.04e-3 | 325 | +1.72e-3 | +1.72e-3 | +1.72e-3 | -1.02e-4 |
| 187 | 3.00e-3 | 1 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 7.49e-3 | 260 | +2.42e-4 | +2.42e-4 | +2.42e-4 | -6.74e-5 |
| 188 | 3.00e-3 | 1 | 8.15e-3 | 8.15e-3 | 8.15e-3 | 8.15e-3 | 344 | +2.43e-4 | +2.43e-4 | +2.43e-4 | -3.64e-5 |
| 189 | 3.00e-3 | 1 | 7.89e-3 | 7.89e-3 | 7.89e-3 | 7.89e-3 | 283 | -1.13e-4 | -1.13e-4 | -1.13e-4 | -4.40e-5 |
| 190 | 3.00e-3 | 1 | 7.79e-3 | 7.79e-3 | 7.79e-3 | 7.79e-3 | 269 | -4.61e-5 | -4.61e-5 | -4.61e-5 | -4.42e-5 |
| 191 | 3.00e-3 | 1 | 8.18e-3 | 8.18e-3 | 8.18e-3 | 8.18e-3 | 292 | +1.67e-4 | +1.67e-4 | +1.67e-4 | -2.31e-5 |
| 192 | 3.00e-3 | 1 | 7.91e-3 | 7.91e-3 | 7.91e-3 | 7.91e-3 | 292 | -1.16e-4 | -1.16e-4 | -1.16e-4 | -3.24e-5 |
| 193 | 3.00e-3 | 1 | 7.91e-3 | 7.91e-3 | 7.91e-3 | 7.91e-3 | 300 | +1.86e-6 | +1.86e-6 | +1.86e-6 | -2.90e-5 |
| 194 | 3.00e-3 | 1 | 8.09e-3 | 8.09e-3 | 8.09e-3 | 8.09e-3 | 288 | +7.66e-5 | +7.66e-5 | +7.66e-5 | -1.84e-5 |
| 195 | 3.00e-3 | 1 | 8.21e-3 | 8.21e-3 | 8.21e-3 | 8.21e-3 | 289 | +4.93e-5 | +4.93e-5 | +4.93e-5 | -1.16e-5 |
| 196 | 3.00e-3 | 1 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 7.95e-3 | 289 | -1.08e-4 | -1.08e-4 | -1.08e-4 | -2.13e-5 |
| 197 | 3.00e-3 | 1 | 8.06e-3 | 8.06e-3 | 8.06e-3 | 8.06e-3 | 271 | +5.09e-5 | +5.09e-5 | +5.09e-5 | -1.41e-5 |
| 198 | 3.00e-3 | 1 | 8.15e-3 | 8.15e-3 | 8.15e-3 | 8.15e-3 | 292 | +3.80e-5 | +3.80e-5 | +3.80e-5 | -8.87e-6 |

