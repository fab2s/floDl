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
| cpu-async | 0.054548 | 0.9198 | +0.0073 | 1741.1 | 755 | 75.3 | 100% | 100% | 8.9 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | - | - | - | - |

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
| cpu-async | 2.0000 | 0.7631 | 0.5791 | 0.5177 | 0.4798 | 0.5100 | 0.4920 | 0.4778 | 0.4736 | 0.4646 | 0.2028 | 0.1677 | 0.1482 | 0.1368 | 0.1251 | 0.0692 | 0.0631 | 0.0609 | 0.0550 | 0.0545 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4000 | 2.8 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3025 | 3.8 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2975 | 3.8 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 398 | 393 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1738.9 | 2.2 | epoch-boundary(199) |
| cpu-async | gpu2 | 1739.0 | 2.1 | epoch-boundary(199) |
| cpu-async | gpu1 | 946.1 | 1.0 | epoch-boundary(108) |
| cpu-async | gpu2 | 946.1 | 1.0 | epoch-boundary(108) |
| cpu-async | gpu0 | 0.3 | 0.7 | cpu-avg |
| cpu-async | gpu2 | 998.9 | 0.7 | epoch-boundary(114) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu0 | 0.0s | 0.0s | 0.7s | 0.0s | 0.7s |
| resnet-graph | cpu-async | gpu1 | 3.2s | 0.0s | 0.0s | 0.0s | 3.8s |
| resnet-graph | cpu-async | gpu2 | 3.7s | 0.0s | 0.0s | 0.0s | 4.4s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 331 | 0 | 755 | 75.3 | 1203/8035 | 755 | 75.3 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 189.5 | 10.9% |

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
| resnet-graph | cpu-async | 194 | 755 | 0 | 7.35e-3 | -2.16e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 755 | 7.00e-2 | 6.92e-2 | 2.02e-3 | 4.93e-1 | 28.9 | -2.13e-4 | 1.30e-3 |
| resnet-graph | cpu-async | 1 | 755 | 7.06e-2 | 6.97e-2 | 1.99e-3 | 4.50e-1 | 37.9 | -2.27e-4 | 1.30e-3 |
| resnet-graph | cpu-async | 2 | 755 | 7.09e-2 | 7.05e-2 | 2.02e-3 | 4.41e-1 | 33.2 | -2.44e-4 | 1.44e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9961 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9956 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9952 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 64 (1,2,3,4,5,6,7,9…140,146) | 0 (—) | — | 1,2,3,4,5,6,7,9…140,146 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 2 | 2 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 366 | +0.072 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 82 | +0.007 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 303 | +0.071 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 753 | +0.004 | 193 | +0.274 | +0.387 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 754 | 3.40e1–8.08e1 | 6.48e1 | 1.68e-3 | 3.16e-3 | 4.01e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 368 | 32–77617 | +1.005e-5 | 0.426 | +1.023e-5 | 0.461 | 96 | +1.218e-5 | 0.737 | 28–973 | +1.003e-3 | 0.721 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 357 | 898–77617 | +1.132e-5 | 0.599 | +1.143e-5 | 0.635 | 95 | +1.259e-5 | 0.765 | 78–973 | +1.044e-3 | 0.898 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 83 | 78220–116356 | +5.856e-6 | 0.058 | +5.680e-6 | 0.055 | 47 | +6.495e-6 | 0.073 | 237–734 | -5.461e-4 | 0.063 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 304 | 116677–155812 | -8.129e-6 | 0.060 | -8.084e-6 | 0.061 | 51 | -1.220e-5 | 0.090 | 74–1015 | +2.911e-3 | 0.353 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.003e-3 | r0: +9.836e-4, r1: +1.007e-3, r2: +1.019e-3 | r0: 0.714, r1: 0.713, r2: 0.701 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.044e-3 | r0: +1.024e-3, r1: +1.048e-3, r2: +1.063e-3 | r0: 0.899, r1: 0.872, r2: 0.876 | 1.04× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | -5.461e-4 | r0: -5.920e-4, r1: -5.140e-4, r2: -5.323e-4 | r0: 0.075, r1: 0.056, r2: 0.058 | 1.15× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +2.911e-3 | r0: +2.827e-3, r1: +2.891e-3, r2: +3.008e-3 | r0: 0.337, r1: 0.350, r2: 0.360 | 1.06× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇▇▇▇▇▇▇▇▇████████████▆▅▅▅▅▅▅▅▅▅▅▅▂▁▁▁▁▁▁▁▁▁▁▁▂` | `▁▇▇▇▇▇▇▇▇▇▇▇████████████▇▇██████████▅▇▇▇▆▇▇▇▇▇▆▇▇` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 9.36e-2 | 4.93e-1 | 2.09e-1 | 9.36e-2 | 31 | -5.10e-2 | +9.04e-3 | -1.26e-2 | -1.01e-2 |
| 1 | 3.00e-1 | 11 | 8.39e-2 | 1.42e-1 | 9.44e-2 | 9.39e-2 | 30 | -8.89e-3 | +5.68e-3 | -5.46e-4 | -2.48e-3 |
| 2 | 3.00e-1 | 5 | 9.13e-2 | 1.39e-1 | 1.06e-1 | 9.71e-2 | 32 | -8.74e-3 | +4.68e-3 | -1.20e-3 | -1.89e-3 |
| 3 | 3.00e-1 | 7 | 9.92e-2 | 1.41e-1 | 1.08e-1 | 1.08e-1 | 34 | -1.03e-2 | +5.03e-3 | -6.16e-4 | -1.09e-3 |
| 4 | 3.00e-1 | 7 | 1.14e-1 | 1.43e-1 | 1.21e-1 | 1.21e-1 | 44 | -4.91e-3 | +3.56e-3 | -1.73e-4 | -5.74e-4 |
| 5 | 3.00e-1 | 6 | 1.07e-1 | 1.53e-1 | 1.18e-1 | 1.11e-1 | 36 | -7.31e-3 | +2.89e-3 | -9.82e-4 | -7.11e-4 |
| 6 | 3.00e-1 | 7 | 1.10e-1 | 1.46e-1 | 1.21e-1 | 1.24e-1 | 45 | -8.13e-3 | +3.67e-3 | -2.31e-4 | -3.94e-4 |
| 7 | 3.00e-1 | 6 | 1.14e-1 | 1.56e-1 | 1.28e-1 | 1.14e-1 | 38 | -3.69e-3 | +2.76e-3 | -8.04e-4 | -6.30e-4 |
| 8 | 3.00e-1 | 7 | 1.02e-1 | 1.52e-1 | 1.13e-1 | 1.03e-1 | 32 | -9.67e-3 | +3.80e-3 | -1.19e-3 | -8.61e-4 |
| 9 | 3.00e-1 | 8 | 1.06e-1 | 1.48e-1 | 1.15e-1 | 1.13e-1 | 39 | -8.62e-3 | +4.98e-3 | -4.39e-4 | -5.46e-4 |
| 10 | 3.00e-1 | 6 | 1.04e-1 | 1.54e-1 | 1.16e-1 | 1.04e-1 | 33 | -7.35e-3 | +3.61e-3 | -1.35e-3 | -8.87e-4 |
| 11 | 3.00e-1 | 9 | 1.04e-1 | 1.43e-1 | 1.13e-1 | 1.12e-1 | 39 | -8.43e-3 | +4.65e-3 | -3.34e-4 | -4.80e-4 |
| 12 | 3.00e-1 | 6 | 1.02e-1 | 1.48e-1 | 1.14e-1 | 1.09e-1 | 38 | -7.88e-3 | +3.80e-3 | -8.90e-4 | -6.14e-4 |
| 13 | 3.00e-1 | 9 | 1.03e-1 | 1.50e-1 | 1.14e-1 | 1.08e-1 | 37 | -5.49e-3 | +4.01e-3 | -5.96e-4 | -5.28e-4 |
| 14 | 3.00e-1 | 5 | 9.69e-2 | 1.59e-1 | 1.13e-1 | 9.69e-2 | 29 | -1.31e-2 | +4.08e-3 | -2.59e-3 | -1.34e-3 |
| 15 | 3.00e-1 | 7 | 1.03e-1 | 1.51e-1 | 1.17e-1 | 1.18e-1 | 41 | -8.70e-3 | +5.64e-3 | -3.84e-4 | -7.71e-4 |
| 16 | 3.00e-1 | 7 | 1.06e-1 | 1.63e-1 | 1.20e-1 | 1.11e-1 | 40 | -9.10e-3 | +3.50e-3 | -1.08e-3 | -8.60e-4 |
| 17 | 3.00e-1 | 5 | 1.16e-1 | 1.45e-1 | 1.24e-1 | 1.19e-1 | 44 | -3.55e-3 | +3.43e-3 | -2.84e-4 | -6.39e-4 |
| 18 | 3.00e-1 | 9 | 1.07e-1 | 1.59e-1 | 1.20e-1 | 1.16e-1 | 37 | -7.19e-3 | +3.22e-3 | -5.48e-4 | -4.80e-4 |
| 19 | 3.00e-1 | 4 | 1.12e-1 | 1.54e-1 | 1.25e-1 | 1.12e-1 | 37 | -6.47e-3 | +3.48e-3 | -1.16e-3 | -7.38e-4 |
| 20 | 3.00e-1 | 9 | 1.03e-1 | 1.50e-1 | 1.15e-1 | 1.04e-1 | 33 | -6.63e-3 | +4.01e-3 | -6.59e-4 | -6.82e-4 |
| 21 | 3.00e-1 | 4 | 1.08e-1 | 1.58e-1 | 1.22e-1 | 1.08e-1 | 32 | -1.02e-2 | +4.56e-3 | -1.88e-3 | -1.11e-3 |
| 22 | 3.00e-1 | 7 | 1.13e-1 | 1.50e-1 | 1.22e-1 | 1.18e-1 | 42 | -5.13e-3 | +3.99e-3 | -2.78e-4 | -6.60e-4 |
| 23 | 3.00e-1 | 5 | 1.17e-1 | 1.55e-1 | 1.27e-1 | 1.21e-1 | 42 | -5.01e-3 | +3.40e-3 | -4.80e-4 | -5.85e-4 |
| 24 | 3.00e-1 | 7 | 1.03e-1 | 1.60e-1 | 1.17e-1 | 1.03e-1 | 30 | -7.60e-3 | +3.12e-3 | -1.26e-3 | -9.08e-4 |
| 25 | 3.00e-1 | 8 | 1.03e-1 | 1.43e-1 | 1.17e-1 | 1.19e-1 | 41 | -8.88e-3 | +4.58e-3 | -2.50e-4 | -4.62e-4 |
| 26 | 3.00e-1 | 5 | 1.16e-1 | 1.63e-1 | 1.28e-1 | 1.20e-1 | 44 | -6.17e-3 | +3.38e-3 | -7.65e-4 | -5.74e-4 |
| 27 | 3.00e-1 | 6 | 1.15e-1 | 1.53e-1 | 1.24e-1 | 1.17e-1 | 41 | -5.73e-3 | +2.94e-3 | -6.00e-4 | -5.71e-4 |
| 28 | 3.00e-1 | 6 | 1.09e-1 | 1.55e-1 | 1.21e-1 | 1.09e-1 | 35 | -7.49e-3 | +3.38e-3 | -1.04e-3 | -7.98e-4 |
| 29 | 3.00e-1 | 6 | 1.10e-1 | 1.59e-1 | 1.22e-1 | 1.13e-1 | 36 | -7.77e-3 | +4.24e-3 | -8.69e-4 | -8.07e-4 |
| 30 | 3.00e-1 | 7 | 1.14e-1 | 1.47e-1 | 1.24e-1 | 1.23e-1 | 48 | -5.23e-3 | +3.42e-3 | -1.57e-4 | -4.46e-4 |
| 31 | 3.00e-1 | 5 | 1.20e-1 | 1.59e-1 | 1.31e-1 | 1.22e-1 | 45 | -4.43e-3 | +2.93e-3 | -5.99e-4 | -5.15e-4 |
| 32 | 3.00e-1 | 6 | 1.08e-1 | 1.59e-1 | 1.21e-1 | 1.14e-1 | 41 | -6.51e-3 | +3.12e-3 | -8.90e-4 | -6.39e-4 |
| 33 | 3.00e-1 | 6 | 1.21e-1 | 1.53e-1 | 1.29e-1 | 1.21e-1 | 45 | -5.51e-3 | +3.61e-3 | -3.88e-4 | -5.41e-4 |
| 34 | 3.00e-1 | 6 | 1.16e-1 | 1.55e-1 | 1.28e-1 | 1.16e-1 | 39 | -4.67e-3 | +3.01e-3 | -6.61e-4 | -6.21e-4 |
| 35 | 3.00e-1 | 6 | 1.09e-1 | 1.60e-1 | 1.22e-1 | 1.20e-1 | 39 | -7.91e-3 | +3.81e-3 | -7.00e-4 | -5.79e-4 |
| 36 | 3.00e-1 | 8 | 9.96e-2 | 1.53e-1 | 1.15e-1 | 1.09e-1 | 33 | -7.16e-3 | +3.28e-3 | -8.34e-4 | -6.24e-4 |
| 37 | 3.00e-1 | 6 | 1.08e-1 | 1.61e-1 | 1.22e-1 | 1.18e-1 | 41 | -8.23e-3 | +4.75e-3 | -7.61e-4 | -6.24e-4 |
| 38 | 3.00e-1 | 9 | 1.15e-1 | 1.50e-1 | 1.24e-1 | 1.17e-1 | 37 | -3.36e-3 | +3.20e-3 | -2.99e-4 | -4.28e-4 |
| 39 | 3.00e-1 | 4 | 1.14e-1 | 1.51e-1 | 1.27e-1 | 1.14e-1 | 40 | -5.11e-3 | +3.59e-3 | -8.71e-4 | -6.23e-4 |
| 40 | 3.00e-1 | 9 | 1.09e-1 | 1.57e-1 | 1.20e-1 | 1.13e-1 | 35 | -5.41e-3 | +4.26e-3 | -4.65e-4 | -4.81e-4 |
| 41 | 3.00e-1 | 5 | 1.08e-1 | 1.54e-1 | 1.25e-1 | 1.08e-1 | 35 | -6.27e-3 | +3.85e-3 | -1.22e-3 | -8.44e-4 |
| 42 | 3.00e-1 | 6 | 1.11e-1 | 1.51e-1 | 1.21e-1 | 1.19e-1 | 43 | -8.32e-3 | +4.84e-3 | -4.61e-4 | -6.31e-4 |
| 43 | 3.00e-1 | 7 | 1.15e-1 | 1.48e-1 | 1.28e-1 | 1.36e-1 | 52 | -4.44e-3 | +2.16e-3 | +1.30e-4 | -1.94e-4 |
| 44 | 3.00e-1 | 6 | 1.11e-1 | 1.60e-1 | 1.22e-1 | 1.11e-1 | 35 | -5.25e-3 | +1.93e-3 | -1.16e-3 | -6.07e-4 |
| 45 | 3.00e-1 | 7 | 1.03e-1 | 1.62e-1 | 1.18e-1 | 1.21e-1 | 47 | -7.92e-3 | +4.44e-3 | -7.02e-4 | -5.57e-4 |
| 46 | 3.00e-1 | 5 | 1.25e-1 | 1.66e-1 | 1.36e-1 | 1.28e-1 | 45 | -4.57e-3 | +3.37e-3 | -4.20e-4 | -5.11e-4 |
| 47 | 3.00e-1 | 1 | 1.21e-1 | 1.21e-1 | 1.21e-1 | 1.21e-1 | 45 | -1.17e-3 | -1.17e-3 | -1.17e-3 | -5.77e-4 |
| 48 | 3.00e-1 | 2 | 2.34e-1 | 2.37e-1 | 2.35e-1 | 2.34e-1 | 287 | -3.75e-5 | +2.01e-3 | +9.85e-4 | -2.91e-4 |
| 50 | 3.00e-1 | 1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 2.47e-1 | 345 | +1.54e-4 | +1.54e-4 | +1.54e-4 | -2.46e-4 |
| 51 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 319 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -2.34e-4 |
| 52 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 303 | -1.14e-4 | -1.14e-4 | -1.14e-4 | -2.22e-4 |
| 53 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 293 | -4.22e-5 | -4.22e-5 | -4.22e-5 | -2.04e-4 |
| 54 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 282 | -1.50e-5 | -1.50e-5 | -1.50e-5 | -1.85e-4 |
| 56 | 3.00e-1 | 2 | 2.29e-1 | 2.45e-1 | 2.37e-1 | 2.29e-1 | 272 | -2.49e-4 | +2.36e-4 | -6.63e-6 | -1.54e-4 |
| 58 | 3.00e-1 | 2 | 2.28e-1 | 2.30e-1 | 2.29e-1 | 2.28e-1 | 280 | -3.95e-5 | +1.29e-5 | -1.33e-5 | -1.27e-4 |
| 60 | 3.00e-1 | 1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 2.44e-1 | 360 | +1.87e-4 | +1.87e-4 | +1.87e-4 | -9.57e-5 |
| 61 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 336 | -6.71e-5 | -6.71e-5 | -6.71e-5 | -9.29e-5 |
| 62 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 299 | -6.22e-5 | -6.22e-5 | -6.22e-5 | -8.98e-5 |
| 63 | 3.00e-1 | 1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 2.31e-1 | 296 | -4.07e-5 | -4.07e-5 | -4.07e-5 | -8.49e-5 |
| 64 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 296 | +2.03e-5 | +2.03e-5 | +2.03e-5 | -7.44e-5 |
| 65 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 270 | -1.64e-4 | -1.64e-4 | -1.64e-4 | -8.33e-5 |
| 66 | 3.00e-1 | 1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 2.38e-1 | 311 | +2.20e-4 | +2.20e-4 | +2.20e-4 | -5.30e-5 |
| 67 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 282 | -1.63e-4 | -1.63e-4 | -1.63e-4 | -6.40e-5 |
| 68 | 3.00e-1 | 1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 2.34e-1 | 291 | +9.22e-5 | +9.22e-5 | +9.22e-5 | -4.84e-5 |
| 69 | 3.00e-1 | 1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 2.17e-1 | 240 | -3.09e-4 | -3.09e-4 | -3.09e-4 | -7.44e-5 |
| 70 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 246 | +5.25e-5 | +5.25e-5 | +5.25e-5 | -6.17e-5 |
| 71 | 3.00e-1 | 1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 2.24e-1 | 265 | +7.14e-5 | +7.14e-5 | +7.14e-5 | -4.84e-5 |
| 72 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 260 | -1.93e-5 | -1.93e-5 | -1.93e-5 | -4.55e-5 |
| 73 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 263 | -2.25e-6 | -2.25e-6 | -2.25e-6 | -4.12e-5 |
| 74 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 255 | +1.28e-5 | +1.28e-5 | +1.28e-5 | -3.58e-5 |
| 75 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 273 | +3.49e-5 | +3.49e-5 | +3.49e-5 | -2.87e-5 |
| 76 | 3.00e-1 | 1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 2.29e-1 | 279 | +5.09e-5 | +5.09e-5 | +5.09e-5 | -2.08e-5 |
| 77 | 3.00e-1 | 2 | 2.15e-1 | 2.27e-1 | 2.21e-1 | 2.15e-1 | 224 | -2.56e-4 | -2.56e-5 | -1.41e-4 | -4.47e-5 |
| 78 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 248 | +7.45e-5 | +7.45e-5 | +7.45e-5 | -3.28e-5 |
| 79 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 263 | +5.37e-5 | +5.37e-5 | +5.37e-5 | -2.42e-5 |
| 80 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 253 | -2.39e-5 | -2.39e-5 | -2.39e-5 | -2.41e-5 |
| 81 | 3.00e-1 | 1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 2.20e-1 | 250 | +2.71e-7 | +2.71e-7 | +2.71e-7 | -2.17e-5 |
| 82 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 247 | +4.40e-6 | +4.40e-6 | +4.40e-6 | -1.91e-5 |
| 83 | 3.00e-1 | 1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 2.37e-1 | 293 | +2.51e-4 | +2.51e-4 | +2.51e-4 | +7.94e-6 |
| 84 | 3.00e-1 | 1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 2.27e-1 | 270 | -1.66e-4 | -1.66e-4 | -1.66e-4 | -9.51e-6 |
| 85 | 3.00e-1 | 1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 2.23e-1 | 261 | -7.15e-5 | -7.15e-5 | -7.15e-5 | -1.57e-5 |
| 86 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 269 | +5.71e-5 | +5.71e-5 | +5.71e-5 | -8.43e-6 |
| 87 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 252 | -8.00e-5 | -8.00e-5 | -8.00e-5 | -1.56e-5 |
| 88 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 263 | +6.75e-5 | +6.75e-5 | +6.75e-5 | -7.28e-6 |
| 89 | 3.00e-1 | 1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 2.26e-1 | 260 | +4.84e-6 | +4.84e-6 | +4.84e-6 | -6.07e-6 |
| 90 | 3.00e-1 | 1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 2.28e-1 | 255 | +3.42e-5 | +3.42e-5 | +3.42e-5 | -2.03e-6 |
| 91 | 3.00e-1 | 2 | 2.12e-1 | 2.22e-1 | 2.17e-1 | 2.12e-1 | 208 | -2.28e-4 | -1.03e-4 | -1.66e-4 | -3.38e-5 |
| 92 | 3.00e-1 | 1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 2.19e-1 | 226 | +1.52e-4 | +1.52e-4 | +1.52e-4 | -1.52e-5 |
| 93 | 3.00e-1 | 1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 2.32e-1 | 267 | +2.08e-4 | +2.08e-4 | +2.08e-4 | +7.13e-6 |
| 94 | 3.00e-1 | 1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 2.22e-1 | 250 | -1.64e-4 | -1.64e-4 | -1.64e-4 | -9.99e-6 |
| 95 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 238 | -7.22e-5 | -7.22e-5 | -7.22e-5 | -1.62e-5 |
| 96 | 3.00e-1 | 2 | 2.07e-1 | 2.25e-1 | 2.16e-1 | 2.07e-1 | 196 | -4.20e-4 | +1.09e-4 | -1.56e-4 | -4.54e-5 |
| 97 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 217 | +1.70e-4 | +1.70e-4 | +1.70e-4 | -2.39e-5 |
| 98 | 3.00e-1 | 1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 2.25e-1 | 238 | +2.05e-4 | +2.05e-4 | +2.05e-4 | -1.02e-6 |
| 99 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 227 | -2.46e-4 | -2.46e-4 | -2.46e-4 | -2.55e-5 |
| 100 | 3.00e-2 | 2 | 1.09e-1 | 2.11e-1 | 1.60e-1 | 1.09e-1 | 196 | -3.38e-3 | -5.43e-5 | -1.72e-3 | -3.64e-4 |
| 101 | 3.00e-2 | 1 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 5.54e-2 | 226 | -2.98e-3 | -2.98e-3 | -2.98e-3 | -6.25e-4 |
| 102 | 3.00e-2 | 2 | 2.60e-2 | 3.44e-2 | 3.02e-2 | 2.60e-2 | 196 | -2.09e-3 | -1.42e-3 | -1.76e-3 | -8.37e-4 |
| 104 | 3.00e-2 | 2 | 2.53e-2 | 2.77e-2 | 2.65e-2 | 2.53e-2 | 208 | -4.30e-4 | +2.34e-4 | -9.83e-5 | -7.00e-4 |
| 105 | 3.00e-2 | 1 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 2.65e-2 | 232 | +2.04e-4 | +2.04e-4 | +2.04e-4 | -6.09e-4 |
| 106 | 3.00e-2 | 2 | 2.68e-2 | 2.74e-2 | 2.71e-2 | 2.68e-2 | 208 | -1.15e-4 | +1.45e-4 | +1.46e-5 | -4.92e-4 |
| 108 | 3.00e-2 | 2 | 2.86e-2 | 3.08e-2 | 2.97e-2 | 2.86e-2 | 208 | -3.52e-4 | +4.87e-4 | +6.73e-5 | -3.90e-4 |
| 109 | 3.00e-2 | 1 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 2.97e-2 | 227 | +1.69e-4 | +1.69e-4 | +1.69e-4 | -3.34e-4 |
| 110 | 3.00e-2 | 1 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 3.00e-2 | 220 | +4.02e-5 | +4.02e-5 | +4.02e-5 | -2.97e-4 |
| 111 | 3.00e-2 | 2 | 2.98e-2 | 3.12e-2 | 3.05e-2 | 2.98e-2 | 197 | -2.37e-4 | +1.77e-4 | -3.03e-5 | -2.48e-4 |
| 112 | 3.00e-2 | 1 | 3.19e-2 | 3.19e-2 | 3.19e-2 | 3.19e-2 | 223 | +3.10e-4 | +3.10e-4 | +3.10e-4 | -1.92e-4 |
| 113 | 3.00e-2 | 1 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 3.26e-2 | 219 | +9.92e-5 | +9.92e-5 | +9.92e-5 | -1.63e-4 |
| 114 | 3.00e-2 | 2 | 3.27e-2 | 3.30e-2 | 3.29e-2 | 3.30e-2 | 197 | +1.46e-5 | +4.35e-5 | +2.90e-5 | -1.27e-4 |
| 115 | 3.00e-2 | 1 | 3.68e-2 | 3.68e-2 | 3.68e-2 | 3.68e-2 | 270 | +4.00e-4 | +4.00e-4 | +4.00e-4 | -7.39e-5 |
| 116 | 3.00e-2 | 1 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 3.48e-2 | 212 | -2.52e-4 | -2.52e-4 | -2.52e-4 | -9.17e-5 |
| 117 | 3.00e-2 | 1 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 3.50e-2 | 209 | +2.71e-5 | +2.71e-5 | +2.71e-5 | -7.98e-5 |
| 118 | 3.00e-2 | 2 | 3.29e-2 | 3.50e-2 | 3.39e-2 | 3.50e-2 | 207 | -3.61e-4 | +2.88e-4 | -3.69e-5 | -6.84e-5 |
| 119 | 3.00e-2 | 1 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 3.55e-2 | 194 | +7.91e-5 | +7.91e-5 | +7.91e-5 | -5.37e-5 |
| 120 | 3.00e-2 | 2 | 3.44e-2 | 3.60e-2 | 3.52e-2 | 3.44e-2 | 162 | -2.89e-4 | +8.09e-5 | -1.04e-4 | -6.51e-5 |
| 121 | 3.00e-2 | 2 | 3.43e-2 | 3.59e-2 | 3.51e-2 | 3.43e-2 | 162 | -2.86e-4 | +2.24e-4 | -3.10e-5 | -6.12e-5 |
| 122 | 3.00e-2 | 1 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 3.72e-2 | 207 | +3.97e-4 | +3.97e-4 | +3.97e-4 | -1.53e-5 |
| 123 | 3.00e-2 | 1 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 3.62e-2 | 172 | -1.57e-4 | -1.57e-4 | -1.57e-4 | -2.95e-5 |
| 124 | 3.00e-2 | 2 | 3.92e-2 | 4.24e-2 | 4.08e-2 | 3.92e-2 | 171 | -4.65e-4 | +6.39e-4 | +8.70e-5 | -1.28e-5 |
| 125 | 3.00e-2 | 2 | 3.76e-2 | 3.89e-2 | 3.82e-2 | 3.76e-2 | 157 | -2.05e-4 | -4.04e-5 | -1.22e-4 | -3.45e-5 |
| 126 | 3.00e-2 | 2 | 3.80e-2 | 4.07e-2 | 3.93e-2 | 3.80e-2 | 157 | -4.35e-4 | +4.08e-4 | -1.34e-5 | -3.47e-5 |
| 127 | 3.00e-2 | 1 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 4.10e-2 | 186 | +4.07e-4 | +4.07e-4 | +4.07e-4 | +9.44e-6 |
| 128 | 3.00e-2 | 1 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 4.01e-2 | 170 | -1.24e-4 | -1.24e-4 | -1.24e-4 | -3.89e-6 |
| 129 | 3.00e-2 | 2 | 4.09e-2 | 4.11e-2 | 4.10e-2 | 4.09e-2 | 157 | -3.11e-5 | +1.36e-4 | +5.23e-5 | +5.95e-6 |
| 130 | 3.00e-2 | 2 | 3.98e-2 | 4.26e-2 | 4.12e-2 | 3.98e-2 | 146 | -4.70e-4 | +2.04e-4 | -1.33e-4 | -2.39e-5 |
| 131 | 3.00e-2 | 1 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 4.19e-2 | 164 | +3.16e-4 | +3.16e-4 | +3.16e-4 | +1.02e-5 |
| 132 | 3.00e-2 | 3 | 4.06e-2 | 4.24e-2 | 4.18e-2 | 4.23e-2 | 152 | -2.89e-4 | +2.71e-4 | +1.89e-5 | +1.45e-5 |
| 133 | 3.00e-2 | 1 | 4.50e-2 | 4.50e-2 | 4.50e-2 | 4.50e-2 | 182 | +3.39e-4 | +3.39e-4 | +3.39e-4 | +4.70e-5 |
| 134 | 3.00e-2 | 2 | 4.15e-2 | 4.36e-2 | 4.25e-2 | 4.15e-2 | 135 | -3.63e-4 | -1.96e-4 | -2.80e-4 | -1.59e-5 |
| 135 | 3.00e-2 | 1 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 4.55e-2 | 176 | +5.32e-4 | +5.32e-4 | +5.32e-4 | +3.89e-5 |
| 136 | 3.00e-2 | 3 | 4.14e-2 | 4.77e-2 | 4.42e-2 | 4.14e-2 | 125 | -6.61e-4 | +2.42e-4 | -2.71e-4 | -5.09e-5 |
| 137 | 3.00e-2 | 1 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 4.60e-2 | 164 | +6.34e-4 | +6.34e-4 | +6.34e-4 | +1.76e-5 |
| 138 | 3.00e-2 | 3 | 4.04e-2 | 4.68e-2 | 4.26e-2 | 4.04e-2 | 115 | -1.23e-3 | +1.01e-4 | -3.87e-4 | -9.30e-5 |
| 139 | 3.00e-2 | 2 | 4.24e-2 | 4.25e-2 | 4.24e-2 | 4.25e-2 | 109 | +2.50e-5 | +3.53e-4 | +1.89e-4 | -4.11e-5 |
| 140 | 3.00e-2 | 2 | 4.20e-2 | 4.77e-2 | 4.49e-2 | 4.20e-2 | 115 | -1.12e-3 | +6.54e-4 | -2.35e-4 | -8.68e-5 |
| 141 | 3.00e-2 | 2 | 4.32e-2 | 4.54e-2 | 4.43e-2 | 4.32e-2 | 113 | -4.42e-4 | +5.47e-4 | +5.26e-5 | -6.53e-5 |
| 142 | 3.00e-2 | 2 | 4.35e-2 | 4.63e-2 | 4.49e-2 | 4.35e-2 | 120 | -5.26e-4 | +5.07e-4 | -9.47e-6 | -5.98e-5 |
| 143 | 3.00e-2 | 2 | 4.40e-2 | 4.65e-2 | 4.52e-2 | 4.40e-2 | 109 | -5.05e-4 | +4.54e-4 | -2.57e-5 | -5.81e-5 |
| 144 | 3.00e-2 | 3 | 3.96e-2 | 4.90e-2 | 4.37e-2 | 3.96e-2 | 87 | -1.44e-3 | +7.06e-4 | -5.10e-4 | -1.94e-4 |
| 145 | 3.00e-2 | 3 | 4.05e-2 | 4.22e-2 | 4.11e-2 | 4.06e-2 | 93 | -4.29e-4 | +5.38e-4 | +5.06e-5 | -1.33e-4 |
| 146 | 3.00e-2 | 2 | 4.34e-2 | 4.90e-2 | 4.62e-2 | 4.34e-2 | 98 | -1.25e-3 | +1.31e-3 | +3.09e-5 | -1.14e-4 |
| 147 | 3.00e-2 | 4 | 4.10e-2 | 4.56e-2 | 4.30e-2 | 4.10e-2 | 93 | -6.55e-4 | +4.12e-4 | -1.71e-4 | -1.41e-4 |
| 148 | 3.00e-2 | 2 | 4.56e-2 | 4.85e-2 | 4.71e-2 | 4.56e-2 | 101 | -5.95e-4 | +1.26e-3 | +3.34e-4 | -6.03e-5 |
| 149 | 3.00e-3 | 3 | 4.25e-2 | 4.70e-2 | 4.41e-2 | 4.27e-2 | 91 | -1.18e-3 | +2.34e-4 | -3.03e-4 | -1.27e-4 |
| 150 | 3.00e-3 | 2 | 1.14e-2 | 2.35e-2 | 1.74e-2 | 1.14e-2 | 91 | -7.99e-3 | -4.97e-3 | -6.48e-3 | -1.35e-3 |
| 151 | 3.00e-3 | 4 | 3.67e-3 | 7.21e-3 | 4.78e-3 | 3.67e-3 | 78 | -6.42e-3 | -6.41e-4 | -3.11e-3 | -1.89e-3 |
| 152 | 3.00e-3 | 4 | 3.55e-3 | 4.17e-3 | 3.76e-3 | 3.55e-3 | 73 | -1.67e-3 | +1.11e-3 | -2.52e-4 | -1.34e-3 |
| 153 | 3.00e-3 | 3 | 3.36e-3 | 3.92e-3 | 3.57e-3 | 3.43e-3 | 67 | -2.46e-3 | +9.41e-4 | -4.03e-4 | -1.09e-3 |
| 154 | 3.00e-3 | 4 | 3.19e-3 | 3.82e-3 | 3.43e-3 | 3.19e-3 | 63 | -2.08e-3 | +1.14e-3 | -4.29e-4 | -8.79e-4 |
| 155 | 3.00e-3 | 4 | 3.16e-3 | 3.99e-3 | 3.47e-3 | 3.16e-3 | 63 | -2.47e-3 | +2.20e-3 | -3.77e-4 | -7.36e-4 |
| 156 | 3.00e-3 | 4 | 3.16e-3 | 3.75e-3 | 3.33e-3 | 3.16e-3 | 59 | -2.45e-3 | +1.86e-3 | -2.18e-4 | -5.74e-4 |
| 157 | 3.00e-3 | 6 | 2.80e-3 | 3.91e-3 | 3.16e-3 | 2.99e-3 | 49 | -3.38e-3 | +2.16e-3 | -4.27e-4 | -4.86e-4 |
| 158 | 3.00e-3 | 3 | 3.10e-3 | 3.86e-3 | 3.41e-3 | 3.10e-3 | 48 | -3.05e-3 | +2.66e-3 | -5.31e-4 | -5.33e-4 |
| 159 | 3.00e-3 | 6 | 2.52e-3 | 3.77e-3 | 2.86e-3 | 2.52e-3 | 40 | -5.98e-3 | +2.02e-3 | -1.41e-3 | -9.22e-4 |
| 160 | 3.00e-3 | 7 | 2.45e-3 | 3.50e-3 | 2.75e-3 | 2.47e-3 | 39 | -4.82e-3 | +4.28e-3 | -6.65e-4 | -8.06e-4 |
| 161 | 3.00e-3 | 8 | 2.41e-3 | 3.76e-3 | 2.77e-3 | 2.50e-3 | 40 | -6.33e-3 | +4.62e-3 | -7.68e-4 | -7.69e-4 |
| 162 | 3.00e-3 | 5 | 2.32e-3 | 3.32e-3 | 2.70e-3 | 2.32e-3 | 33 | -5.84e-3 | +4.09e-3 | -1.21e-3 | -1.01e-3 |
| 163 | 3.00e-3 | 7 | 2.34e-3 | 3.51e-3 | 2.67e-3 | 2.34e-3 | 33 | -6.38e-3 | +4.94e-3 | -1.01e-3 | -1.03e-3 |
| 164 | 3.00e-3 | 10 | 2.21e-3 | 3.26e-3 | 2.50e-3 | 2.34e-3 | 37 | -7.01e-3 | +4.33e-3 | -6.14e-4 | -7.10e-4 |
| 165 | 3.00e-3 | 5 | 2.41e-3 | 3.24e-3 | 2.69e-3 | 2.75e-3 | 42 | -5.85e-3 | +4.67e-3 | -3.78e-5 | -4.00e-4 |
| 166 | 3.00e-3 | 7 | 2.03e-3 | 3.78e-3 | 2.64e-3 | 2.03e-3 | 25 | -4.35e-3 | +3.55e-3 | -2.02e-3 | -1.31e-3 |
| 167 | 3.00e-3 | 7 | 2.04e-3 | 3.10e-3 | 2.58e-3 | 2.84e-3 | 40 | -9.53e-3 | +6.83e-3 | -4.54e-5 | -5.48e-4 |
| 168 | 3.00e-3 | 7 | 2.31e-3 | 3.58e-3 | 2.63e-3 | 2.48e-3 | 39 | -1.11e-2 | +3.06e-3 | -1.32e-3 | -8.19e-4 |
| 169 | 3.00e-3 | 7 | 2.58e-3 | 3.72e-3 | 2.79e-3 | 2.64e-3 | 40 | -9.39e-3 | +4.95e-3 | -6.91e-4 | -7.06e-4 |
| 170 | 3.00e-3 | 9 | 2.35e-3 | 3.63e-3 | 2.75e-3 | 2.65e-3 | 36 | -4.92e-3 | +4.17e-3 | -4.25e-4 | -4.42e-4 |
| 171 | 3.00e-3 | 5 | 2.52e-3 | 3.59e-3 | 2.82e-3 | 2.56e-3 | 36 | -6.60e-3 | +4.15e-3 | -1.12e-3 | -7.21e-4 |
| 172 | 3.00e-3 | 10 | 2.39e-3 | 3.43e-3 | 2.62e-3 | 2.56e-3 | 37 | -6.83e-3 | +4.14e-3 | -3.87e-4 | -4.02e-4 |
| 173 | 3.00e-3 | 4 | 2.59e-3 | 3.73e-3 | 3.00e-3 | 2.74e-3 | 42 | -6.34e-3 | +4.16e-3 | -1.14e-3 | -6.74e-4 |
| 174 | 3.00e-3 | 6 | 2.75e-3 | 3.84e-3 | 3.06e-3 | 2.75e-3 | 41 | -5.80e-3 | +3.92e-3 | -7.19e-4 | -7.17e-4 |
| 175 | 3.00e-3 | 6 | 2.75e-3 | 3.93e-3 | 3.08e-3 | 2.75e-3 | 37 | -5.30e-3 | +3.93e-3 | -7.20e-4 | -7.48e-4 |
| 176 | 3.00e-3 | 6 | 2.54e-3 | 3.89e-3 | 2.95e-3 | 2.76e-3 | 42 | -5.96e-3 | +3.88e-3 | -8.83e-4 | -7.71e-4 |
| 177 | 3.00e-3 | 6 | 2.77e-3 | 4.05e-3 | 3.11e-3 | 2.80e-3 | 42 | -6.14e-3 | +4.33e-3 | -7.49e-4 | -7.71e-4 |
| 178 | 3.00e-3 | 8 | 2.75e-3 | 3.57e-3 | 2.99e-3 | 2.75e-3 | 39 | -3.11e-3 | +3.28e-3 | -3.79e-4 | -5.79e-4 |
| 179 | 3.00e-3 | 5 | 2.54e-3 | 3.60e-3 | 2.84e-3 | 2.66e-3 | 35 | -7.56e-3 | +3.58e-3 | -1.02e-3 | -7.34e-4 |
| 180 | 3.00e-3 | 6 | 2.68e-3 | 3.71e-3 | 2.94e-3 | 2.83e-3 | 45 | -9.11e-3 | +4.32e-3 | -6.76e-4 | -6.70e-4 |
| 181 | 3.00e-3 | 7 | 2.47e-3 | 3.91e-3 | 2.88e-3 | 2.47e-3 | 33 | -7.69e-3 | +3.53e-3 | -1.33e-3 | -1.01e-3 |
| 182 | 3.00e-3 | 9 | 2.53e-3 | 3.55e-3 | 2.74e-3 | 2.53e-3 | 32 | -9.11e-3 | +5.03e-3 | -6.21e-4 | -7.56e-4 |
| 183 | 3.00e-3 | 5 | 2.58e-3 | 3.83e-3 | 3.00e-3 | 2.72e-3 | 36 | -7.38e-3 | +5.85e-3 | -8.89e-4 | -8.31e-4 |
| 184 | 3.00e-3 | 9 | 2.46e-3 | 3.70e-3 | 2.78e-3 | 2.90e-3 | 37 | -5.88e-3 | +4.45e-3 | -2.13e-4 | -2.81e-4 |
| 185 | 3.00e-3 | 5 | 2.62e-3 | 3.51e-3 | 2.98e-3 | 2.62e-3 | 36 | -3.03e-3 | +2.68e-3 | -9.50e-4 | -6.06e-4 |
| 186 | 3.00e-3 | 9 | 2.39e-3 | 3.75e-3 | 2.78e-3 | 2.51e-3 | 33 | -1.13e-2 | +4.69e-3 | -1.01e-3 | -8.09e-4 |
| 187 | 3.00e-3 | 5 | 2.66e-3 | 3.40e-3 | 2.94e-3 | 3.03e-3 | 43 | -6.99e-3 | +4.58e-3 | +3.99e-5 | -4.39e-4 |
| 188 | 3.00e-3 | 9 | 2.63e-3 | 3.97e-3 | 3.04e-3 | 2.63e-3 | 39 | -5.02e-3 | +3.15e-3 | -7.56e-4 | -6.57e-4 |
| 189 | 3.00e-3 | 4 | 2.78e-3 | 3.76e-3 | 3.13e-3 | 2.78e-3 | 38 | -5.37e-3 | +4.36e-3 | -7.60e-4 | -7.48e-4 |
| 190 | 3.00e-3 | 7 | 2.43e-3 | 4.25e-3 | 2.82e-3 | 2.50e-3 | 31 | -1.01e-2 | +4.93e-3 | -1.51e-3 | -1.05e-3 |
| 191 | 3.00e-3 | 7 | 2.44e-3 | 3.95e-3 | 2.89e-3 | 2.44e-3 | 31 | -8.62e-3 | +5.59e-3 | -1.39e-3 | -1.27e-3 |
| 192 | 3.00e-3 | 8 | 2.39e-3 | 3.31e-3 | 2.71e-3 | 2.62e-3 | 36 | -7.52e-3 | +4.76e-3 | -4.61e-4 | -7.83e-4 |
| 193 | 3.00e-3 | 9 | 2.68e-3 | 3.85e-3 | 2.91e-3 | 2.75e-3 | 39 | -8.02e-3 | +4.88e-3 | -4.99e-4 | -5.66e-4 |
| 194 | 3.00e-3 | 4 | 2.78e-3 | 3.92e-3 | 3.19e-3 | 2.78e-3 | 38 | -5.59e-3 | +4.08e-3 | -1.25e-3 | -8.49e-4 |
| 195 | 3.00e-3 | 7 | 2.73e-3 | 3.59e-3 | 2.96e-3 | 2.92e-3 | 43 | -5.75e-3 | +3.22e-3 | -4.22e-4 | -5.78e-4 |
| 196 | 3.00e-3 | 7 | 2.49e-3 | 4.24e-3 | 3.02e-3 | 2.52e-3 | 31 | -6.16e-3 | +4.21e-3 | -1.34e-3 | -9.91e-4 |
| 197 | 3.00e-3 | 7 | 2.62e-3 | 4.02e-3 | 3.02e-3 | 3.01e-3 | 46 | -1.01e-2 | +5.48e-3 | -6.61e-4 | -7.23e-4 |
| 198 | 3.00e-3 | 1 | 3.27e-3 | 3.27e-3 | 3.27e-3 | 3.27e-3 | 51 | +1.63e-3 | +1.63e-3 | +1.63e-3 | -4.88e-4 |
| 199 | 3.00e-3 | 1 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 7.35e-3 | 363 | +2.23e-3 | +2.23e-3 | +2.23e-3 | -2.16e-4 |

