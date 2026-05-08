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
| cpu-async | 0.053912 | 0.9151 | +0.0026 | 1813.3 | 731 | 75.2 | 100% | 100% | 100% | 7.6 |

## Best Mode per Model

| Model | Best Eval | Mode | Fastest (within 2% of solo-0) | Mode |
|-------|-----------|------|-------------------------------|------|
| resnet-graph | 0.9151 | cpu-async | - | - |

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
| cpu-async | 2.0149 | 0.7393 | 0.6218 | 0.5566 | 0.5353 | 0.5018 | 0.4867 | 0.4798 | 0.4792 | 0.4617 | 0.2109 | 0.1703 | 0.1487 | 0.1252 | 0.1195 | 0.0743 | 0.0675 | 0.0617 | 0.0592 | 0.0539 |

## Per-Rank Schedule

`share` is fraction of batches consumed by each rank (sums to ~1). `tput` is samples/ms. Heterogeneous topology shows up here: in cadence/async modes the fast GPU consumes a proportionally larger share to keep pace with the slow ones.

| Model | Mode | Rank | Device | Share | Tput (samp/ms) |
|-------|------|------|--------|-------|----------------|
| resnet-graph | cpu-async | 0 | cuda:0 | 0.4027 | 2.6 |
| resnet-graph | cpu-async | 1 | cuda:1 | 0.3036 | 3.7 |
| resnet-graph | cpu-async | 2 | cuda:2 | 0.2937 | 3.6 |

## VRAM Usage

| Model | Mode | GPU0 Peak (MB) | GPU0 Mean (MB) | GPU1 Peak (MB) | GPU1 Mean (MB) | GPU2 Peak (MB) | GPU2 Mean (MB) |
|-------|------|---------------|---------------|---------------|---------------|---------------|---------------|
| resnet-graph | cpu-async | 358 | 357 | 396 | 391 | 384 | 378 |

## GPU Idle Analysis

Idle gaps >= 500ms, classified by nearest event.

### resnet-graph

| Mode | GPU | Start (s) | Duration (s) | Cause |
|------|-----|-----------|-------------|-------|
| cpu-async | gpu1 | 1811.2 | 2.1 | epoch-boundary(199) |
| cpu-async | gpu2 | 1811.3 | 2.0 | epoch-boundary(199) |
| cpu-async | gpu1 | 407.1 | 0.9 | epoch-boundary(44) |
| cpu-async | gpu1 | 370.1 | 0.7 | epoch-boundary(40) |
| cpu-async | gpu2 | 370.1 | 0.7 | epoch-boundary(40) |
| cpu-async | gpu1 | 666.8 | 0.5 | epoch-boundary(73) |

## Idle Breakdown by Cause

| Model | Mode | GPU | Epoch Boundary | Sync | CPU Avg | Unexplained | Total Idle |
|-------|------|-----|---------------|------|---------|-------------|------------|
| resnet-graph | cpu-async | gpu1 | 4.2s | 0.0s | 0.0s | 0.0s | 5.0s |
| resnet-graph | cpu-async | gpu2 | 2.7s | 0.0s | 0.0s | 0.0s | 2.7s |

## ElChe Calibration

| Model | Mode | Anchors | Throttles | Syncs | Avg Sync (ms) | Sync Interval P50/P95 (ms) | CPU Avgs | Avg CPU (ms) |
|-------|------|---------|-----------|-------|--------------|---------------------------|---------|-------------|
| resnet-graph | cpu-async | 328 | 0 | 731 | 75.2 | 1320/8207 | 731 | 75.2 |

## Streaming Epoch Overlap

| Model | Mode | Overlap (s) | % of Total |
|-------|------|------------|------------|
| resnet-graph | cpu-async | 201.1 | 11.1% |

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
| resnet-graph | cpu-async | 192 | 731 | 0 | 3.14e-3 | -5.29e-4 |

### Per-Rank Breakdown (bottom scale + cross-scale consistency)

Rank 0 is the fast GPU under heterogeneous dispatch (largest `batch_share`). `win%` is the fraction of AllReduce events where this rank had the highest D across ranks (uniform = 100/world_size). NCCL backends typically expose per-rank-step asymmetry (rank 0 wins ~57% on heterogeneous 3-GPU rigs); CPU averaging hides it (~33% per rank).

| Model | Mode | Rank | n | D_mean | D_sd | D_min | D_max | win% | λ_mean | λ_sd |
|-------|------|-----:|--:|-------:|-----:|------:|------:|-----:|-------:|-----:|
| resnet-graph | cpu-async | 0 | 731 | 4.93e-2 | 6.75e-2 | 1.71e-3 | 4.12e-1 | 29.1 | -3.18e-4 | 2.29e-3 |
| resnet-graph | cpu-async | 1 | 731 | 5.00e-2 | 6.90e-2 | 1.69e-3 | 4.60e-1 | 35.6 | -3.26e-4 | 2.26e-3 |
| resnet-graph | cpu-async | 2 | 731 | 4.99e-2 | 6.84e-2 | 1.69e-3 | 3.85e-1 | 35.3 | -3.29e-4 | 2.30e-3 |

Cross-rank Pearson correlation of D trajectories. Values consistently near 1.0 (>0.99 empirically) support the meta-oscillator framing: ranks are coupled views of one process, not independent oscillators.

| Model | Mode | Pair | Pearson r |
|-------|------|------|----------:|
| resnet-graph | cpu-async | rank0 ↔ rank1 | +0.9980 |
| resnet-graph | cpu-async | rank0 ↔ rank2 | +0.9983 |
| resnet-graph | cpu-async | rank1 ↔ rank2 | +0.9975 |

### Convergence Guard Comparison (per-event simulators) (top scale)

Both guards replayed against the per-AllReduce divergence trajectory (matching the production guard's temporal grain). **Current**: ConvergenceGuard::check_trend — fires when 3 consecutive `d_raw` events rise AND the latest exceeds 0.01 (production default). **MSF**: fires when the recomputed bias-corrected `λ_ema` exceeds 1e-3 for 3 consecutive events (R5-style innovation rule, threshold and sustain are the centre point of the sweep below). Lambda is recomputed in analyze.rs from `(d_raw, k_max)` so old timelines and new ones produce the same numbers. Counts are distinct firing epochs.

| Model | Mode | Current fires | MSF fires | Both | Current only | MSF only |
|-------|------|--------------:|----------:|-----:|-------------:|---------:|
| resnet-graph | cpu-async | 52 (1,2,5,6,7,9,10,11…148,149) | 0 (—) | — | 1,2,5,6,7,9,10,11…148,149 | — |

### MSF Guard Threshold Sweep (top scale)

Per-event MSF guard fires across (threshold × sustain) grid. `fires` = total firing events, `epochs` = distinct epochs touched by ≥1 fire. Reading: a useful detector should have monotone fall-off as `threshold` rises (signal-driven, not threshold-driven), and `epochs` should concentrate near the phase boundaries the design doc predicts (LR drops, warmup stabilization).

| Model | Mode | threshold | sustain | fires | epochs |
|-------|------|----------:|--------:|------:|------:|
| resnet-graph | cpu-async | 0e0 | 3 | 1 | 1 |
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
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 208 | +0.269 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 194 | +0.175 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 325 | +0.056 |

### Predictive Value (Phase-1 kill criterion) (top scale)

Pearson correlations testing whether λ̂ carries forward-looking signal independent of D itself. **`λ_raw_t → ln(D_{t+1})`**: does the rate at event t predict the next D? **`λ_mean_per_ep → eval`**: does the epoch's mean λ̂ correlate with held-out accuracy? **`λ_ema_end_of_ep → eval`**: does the EMA value at epoch boundary correlate with eval? All Pearson, scale-invariant under the `k_used` ↔ `k_max` rescale, so values are robust to the pipeline correction.

| Model | Mode | n_evt | r(λ→ln D_{t+1}) | n_ep | r(λ_mean→eval) | r(λ_ema→eval) |
|-------|------|------:|---------------:|-----:|---------------:|--------------:|
| resnet-graph | cpu-async | 729 | +0.033 | 191 | +0.109 | +0.183 |

### Longitudinal Meta-Velocity (top scale)

Per-event consensus magnitude motion `|Δ||W̄|||/||W̄||_prev`. Independent of `D_t` (transversal): tracks LR schedule + gradient size, not inter-rank synchronization. Phase-transition signal complementary to λ̂. Only available on backends that report `post_norm` (CPU averaging always; NCCL after post_norm wiring).

| Model | Mode | n | ||W̄|| range | ||W̄||_mean | v_mean | v_sd | v_max |
|-------|------|--:|-------------|----------:|------:|----:|------:|
| resnet-graph | cpu-async | 730 | 3.37e1–7.98e1 | 6.26e1 | 1.68e-3 | 3.54e-3 | 3.57e-2 |

### R1 informal: log(D) vs step per LR window (top scale)

LR windows auto-detected from `EpochEnd` LR transitions (>5% change starts a new window). Within each window, OLS fit of `ln(D)` vs cumulative step on two bases: `D_max` (per-event max across ranks; legacy — sensitive to per-rank step asymmetry) and `D_mean` (per-event mean across ranks; the meta-oscillator amplitude — averages out per-rank noise. Cross-rank Pearson r ≥ 0.99 empirically, so the two bases trace one process up to scale, but `D_mean` is the cleaner estimator). Slope is in units of ln(D)/step. R² > 0.9 supports R1 (exponential growth at the slope rate); R² ≈ 0 with slope ≈ 0 is the marginal-stability prediction in noise-dominated equilibria. Rows tagged `(post-transient, skipped N)` are sub-window fits emitted alongside the first LR window when an initialization transient is detected — separating the warmup spike from the stable-LR steady state. The third basis `epoch d_mean` aggregates per-event `ln(D_mean)` to one log-mean per epoch and fits those aggregates — denoises intra-epoch SGD variance, which is the dominant remaining noise source after the cross-rank swap. The fourth basis `by k_used` reframes the x-axis: instead of cumulative step, fit `ln(D_mean)` vs `k_used` (the cycle length: steps since the last AllReduce). Sync is a reset — D ≈ 0 immediately after AllReduce — so the natural drift clock restarts at every coupling event. If pure exponential growth holds *within* a cycle, then `ln(D_t) ≈ const + λ_T · k_used` and the by-k slope is the within-cycle Lyapunov exponent. If the system is in the OU / spiral-to-consensus regime, D_t saturates toward a setpoint D*(LR) and the by-k slope flattens for large k_used (R² collapses).

| Model | Mode | LR | Epochs | n_events | Step range | Slope/step (max) | R² (max) | Slope/step (mean) | R² (mean) | n_epochs | Slope/step (epoch d_mean) | R² (epoch d_mean) | k_used range | Slope/k (mean) | R² (k, mean) |
|-------|------|---:|--------|--:|-----------:|----------:|----:|----------:|----:|--:|----------:|----:|------:|----------:|----:|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | 210 | 34–78059 | +1.010e-5 | 0.485 | +1.022e-5 | 0.499 | 92 | +6.290e-6 | 0.336 | 33–944 | +1.060e-3 | 0.772 |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | 199 | 902–78059 | +9.826e-6 | 0.553 | +9.884e-6 | 0.565 | 91 | +5.972e-6 | 0.315 | 83–944 | +1.046e-3 | 0.909 |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | 195 | 78523–116333 | -1.133e-6 | 0.003 | -1.203e-6 | 0.004 | 49 | -7.369e-7 | 0.004 | 78–515 | +6.029e-4 | 0.138 |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | 326 | 116537–156062 | -2.703e-6 | 0.007 | -2.918e-6 | 0.008 | 51 | -1.508e-6 | 0.003 | 75–246 | +4.977e-3 | 0.243 |

### R1' per-rank by-k slopes (bottom scale + cross-scale consistency)

Bottom-scale within-cycle Lyapunov estimate computed independently from each rank's `D_i` trajectory (instead of the cross-rank-collapsed `D_mean`). The meta-oscillator framing predicts these per-rank slopes match the meta `D_mean` by-k slope (already in the R1 table above) within seed-to-seed sd, because cross-rank Pearson `r > 0.99` says ranks are colinear at the bottom scale up to a per-rank scaling factor. Used as a falsifier: if a per-rank slope diverges from the meta slope by more than 2× on any LR window, the meta-oscillator framing has broken for this run and bottom-scale per-rank treatment is required (cpu-async backend's pipelined averaging is a special case of this gate firing for backend reasons rather than dynamics reasons).

| Model | Mode | LR | Epochs | Meta slope/k | Per-rank slopes (k_used basis) | Per-rank R² | Max ratio (max/min per-rank) | Framing OK? |
|-------|------|---:|--------|----------:|---|---|---|---|
| resnet-graph | cpu-async | 3.00e-1 | 0–99 | +1.060e-3 | r0: +1.048e-3, r1: +1.064e-3, r2: +1.068e-3 | r0: 0.765, r1: 0.757, r2: 0.780 | 1.02× | ✓ |
| resnet-graph | cpu-async | 3.00e-1 | 1–99 (post-transient, skipped 1) | +1.046e-3 | r0: +1.026e-3, r1: +1.057e-3, r2: +1.054e-3 | r0: 0.908, r1: 0.899, r2: 0.903 | 1.03× | ✓ |
| resnet-graph | cpu-async | 3.00e-2 | 100–148 | +6.029e-4 | r0: +5.627e-4, r1: +6.448e-4, r2: +6.012e-4 | r0: 0.126, r1: 0.151, r2: 0.130 | 1.15× | ✓ |
| resnet-graph | cpu-async | 3.00e-3 | 149–199 | +4.977e-3 | r0: +4.751e-3, r1: +5.023e-3, r2: +5.147e-3 | r0: 0.223, r1: 0.244, r2: 0.256 | 1.08× | ✓ |

### Trajectories

Sparklines span all epochs in the run. `log10(D_max)` shows the magnitude-decay structure (LR drops appear as steps). `λ_ema` shows the smoothed Lyapunov proxy (zero-crossings = phase transitions; sharp negatives = collapse).

| Model | Mode | log10(D_max) | λ_ema |
|-------|------|--------------|-------|
| resnet-graph | cpu-async | `█▇▇▇███████████████████▅▄▄▅▅▅▅▅▅▅▅▅▃▁▁▁▁▁▁▁▁▁▁▁▂` | `▁▆▇▇▇██████████████████▆▇▇████▇▆▇▇▇▂▆▆▇▇▇▆▇▇▆▇▇▇` |

### Per-Epoch Detail: resnet-graph / cpu-async

| Epoch | LR | Syncs | D_min | D_max | D_mean | D_end | k_end | λ_min | λ_max | λ_mean | λ_ema_end |
|------:|---:|------:|------:|------:|-------:|------:|------:|------:|------:|-------:|----------:|
| 0 | 3.00e-1 | 11 | 4.16e-2 | 4.60e-1 | 1.25e-1 | 8.96e-2 | 35 | -6.39e-2 | +2.17e-2 | -1.44e-2 | -9.28e-3 |
| 1 | 3.00e-1 | 7 | 8.21e-2 | 1.34e-1 | 9.42e-2 | 1.05e-1 | 51 | -1.38e-2 | +5.53e-3 | -5.90e-4 | -3.58e-3 |
| 2 | 3.00e-1 | 5 | 1.06e-1 | 1.51e-1 | 1.19e-1 | 1.06e-1 | 45 | -6.01e-3 | +4.26e-3 | -5.49e-4 | -2.24e-3 |
| 3 | 3.00e-1 | 6 | 1.13e-1 | 1.53e-1 | 1.21e-1 | 1.16e-1 | 45 | -6.37e-3 | +4.17e-3 | -3.25e-4 | -1.29e-3 |
| 4 | 3.00e-1 | 6 | 1.00e-1 | 1.77e-1 | 1.19e-1 | 1.00e-1 | 31 | -6.52e-3 | +4.04e-3 | -1.52e-3 | -1.38e-3 |
| 5 | 3.00e-1 | 8 | 1.02e-1 | 1.47e-1 | 1.13e-1 | 1.02e-1 | 35 | -9.42e-3 | +4.95e-3 | -6.41e-4 | -9.86e-4 |
| 6 | 3.00e-1 | 6 | 9.73e-2 | 1.49e-1 | 1.12e-1 | 1.03e-1 | 35 | -1.25e-2 | +5.34e-3 | -9.35e-4 | -9.49e-4 |
| 7 | 3.00e-1 | 9 | 9.56e-2 | 1.53e-1 | 1.08e-1 | 1.00e-1 | 34 | -1.37e-2 | +4.84e-3 | -8.97e-4 | -8.51e-4 |
| 8 | 3.00e-1 | 4 | 8.88e-2 | 1.46e-1 | 1.13e-1 | 1.08e-1 | 40 | -1.66e-2 | +5.20e-3 | -1.64e-3 | -1.09e-3 |
| 9 | 3.00e-1 | 7 | 1.01e-1 | 1.56e-1 | 1.13e-1 | 1.10e-1 | 41 | -7.95e-3 | +4.12e-3 | -6.11e-4 | -7.51e-4 |
| 10 | 3.00e-1 | 5 | 1.05e-1 | 1.42e-1 | 1.14e-1 | 1.09e-1 | 42 | -6.44e-3 | +3.41e-3 | -5.96e-4 | -6.69e-4 |
| 11 | 3.00e-1 | 9 | 9.90e-2 | 1.53e-1 | 1.08e-1 | 1.00e-1 | 38 | -9.51e-3 | +3.55e-3 | -7.71e-4 | -6.47e-4 |
| 12 | 3.00e-1 | 4 | 1.00e-1 | 1.51e-1 | 1.14e-1 | 1.02e-1 | 38 | -1.08e-2 | +4.75e-3 | -1.40e-3 | -9.10e-4 |
| 13 | 3.00e-1 | 7 | 9.44e-2 | 1.43e-1 | 1.05e-1 | 1.03e-1 | 41 | -1.09e-2 | +4.48e-3 | -5.75e-4 | -6.48e-4 |
| 14 | 3.00e-1 | 7 | 8.89e-2 | 1.42e-1 | 1.00e-1 | 8.90e-2 | 34 | -1.12e-2 | +4.30e-3 | -1.30e-3 | -9.35e-4 |
| 15 | 3.00e-1 | 6 | 9.37e-2 | 1.44e-1 | 1.09e-1 | 9.92e-2 | 37 | -1.19e-2 | +5.84e-3 | -8.07e-4 | -8.86e-4 |
| 16 | 3.00e-1 | 8 | 9.68e-2 | 1.40e-1 | 1.09e-1 | 1.08e-1 | 45 | -7.30e-3 | +4.13e-3 | -4.27e-4 | -5.78e-4 |
| 17 | 3.00e-1 | 1 | 1.12e-1 | 1.12e-1 | 1.12e-1 | 1.12e-1 | 46 | +7.92e-4 | +7.92e-4 | +7.92e-4 | -4.41e-4 |
| 18 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 270 | +2.40e-3 | +2.40e-3 | +2.40e-3 | -1.57e-4 |
| 19 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 271 | -2.59e-5 | -2.59e-5 | -2.59e-5 | -1.44e-4 |
| 20 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 283 | -5.02e-5 | -5.02e-5 | -5.02e-5 | -1.34e-4 |
| 21 | 3.00e-1 | 1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 2.21e-1 | 351 | +1.57e-4 | +1.57e-4 | +1.57e-4 | -1.05e-4 |
| 22 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 332 | -8.19e-5 | -8.19e-5 | -8.19e-5 | -1.03e-4 |
| 23 | 3.00e-1 | 1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 2.12e-1 | 331 | -4.91e-5 | -4.91e-5 | -4.91e-5 | -9.74e-5 |
| 25 | 3.00e-1 | 2 | 1.98e-1 | 2.20e-1 | 2.09e-1 | 1.98e-1 | 272 | -3.84e-4 | +1.16e-4 | -1.34e-4 | -1.07e-4 |
| 27 | 3.00e-1 | 1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 2.18e-1 | 341 | +2.82e-4 | +2.82e-4 | +2.82e-4 | -6.79e-5 |
| 28 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 301 | -1.65e-4 | -1.65e-4 | -1.65e-4 | -7.77e-5 |
| 29 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 284 | -2.21e-5 | -2.21e-5 | -2.21e-5 | -7.21e-5 |
| 30 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 314 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -5.22e-5 |
| 31 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 300 | -1.51e-4 | -1.51e-4 | -1.51e-4 | -6.20e-5 |
| 32 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 271 | -6.51e-5 | -6.51e-5 | -6.51e-5 | -6.23e-5 |
| 33 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 297 | +1.09e-4 | +1.09e-4 | +1.09e-4 | -4.52e-5 |
| 34 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 288 | -2.71e-5 | -2.71e-5 | -2.71e-5 | -4.34e-5 |
| 35 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 285 | -1.84e-5 | -1.84e-5 | -1.84e-5 | -4.09e-5 |
| 36 | 3.00e-1 | 1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 2.08e-1 | 292 | +3.54e-5 | +3.54e-5 | +3.54e-5 | -3.33e-5 |
| 37 | 3.00e-1 | 1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 2.13e-1 | 291 | +7.66e-5 | +7.66e-5 | +7.66e-5 | -2.23e-5 |
| 38 | 3.00e-1 | 1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 2.04e-1 | 280 | -1.46e-4 | -1.46e-4 | -1.46e-4 | -3.47e-5 |
| 40 | 3.00e-1 | 2 | 2.01e-1 | 2.15e-1 | 2.08e-1 | 2.01e-1 | 258 | -2.67e-4 | +1.66e-4 | -5.07e-5 | -3.99e-5 |
| 41 | 3.00e-1 | 1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 2.14e-1 | 296 | +2.21e-4 | +2.21e-4 | +2.21e-4 | -1.38e-5 |
| 43 | 3.00e-1 | 2 | 2.08e-1 | 2.27e-1 | 2.17e-1 | 2.08e-1 | 288 | -3.08e-4 | +1.61e-4 | -7.34e-5 | -2.75e-5 |
| 45 | 3.00e-1 | 1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 2.15e-1 | 300 | +1.16e-4 | +1.16e-4 | +1.16e-4 | -1.32e-5 |
| 46 | 3.00e-1 | 2 | 1.96e-1 | 2.10e-1 | 2.03e-1 | 1.96e-1 | 234 | -3.05e-4 | -7.34e-5 | -1.89e-4 | -4.78e-5 |
| 48 | 3.00e-1 | 2 | 1.96e-1 | 2.19e-1 | 2.07e-1 | 1.96e-1 | 234 | -4.73e-4 | +3.55e-4 | -5.92e-5 | -5.41e-5 |
| 50 | 3.00e-1 | 2 | 1.95e-1 | 2.25e-1 | 2.10e-1 | 1.95e-1 | 234 | -6.25e-4 | +4.37e-4 | -9.39e-5 | -6.70e-5 |
| 52 | 3.00e-1 | 2 | 1.96e-1 | 2.00e-1 | 1.98e-1 | 1.96e-1 | 235 | -9.26e-5 | +1.10e-4 | +8.87e-6 | -5.36e-5 |
| 53 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 254 | +1.27e-4 | +1.27e-4 | +1.27e-4 | -3.55e-5 |
| 54 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 252 | -7.02e-6 | -7.02e-6 | -7.02e-6 | -3.26e-5 |
| 55 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 244 | -6.73e-5 | -6.73e-5 | -6.73e-5 | -3.61e-5 |
| 56 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 263 | +1.38e-4 | +1.38e-4 | +1.38e-4 | -1.86e-5 |
| 57 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 264 | +1.24e-6 | +1.24e-6 | +1.24e-6 | -1.67e-5 |
| 58 | 3.00e-1 | 1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 2.09e-1 | 269 | +4.77e-5 | +4.77e-5 | +4.77e-5 | -1.02e-5 |
| 59 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 250 | -7.36e-5 | -7.36e-5 | -7.36e-5 | -1.66e-5 |
| 60 | 3.00e-1 | 1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 2.05e-1 | 244 | -8.40e-6 | -8.40e-6 | -8.40e-6 | -1.57e-5 |
| 61 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 238 | -1.17e-4 | -1.17e-4 | -1.17e-4 | -2.59e-5 |
| 62 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 263 | +1.34e-4 | +1.34e-4 | +1.34e-4 | -9.94e-6 |
| 63 | 3.00e-1 | 2 | 1.89e-1 | 2.12e-1 | 2.01e-1 | 1.89e-1 | 210 | -5.63e-4 | +1.04e-4 | -2.30e-4 | -5.50e-5 |
| 64 | 3.00e-1 | 1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 2.07e-1 | 251 | +3.64e-4 | +3.64e-4 | +3.64e-4 | -1.31e-5 |
| 65 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 226 | -1.88e-4 | -1.88e-4 | -1.88e-4 | -3.06e-5 |
| 66 | 3.00e-1 | 1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 1.93e-1 | 222 | -1.21e-4 | -1.21e-4 | -1.21e-4 | -3.96e-5 |
| 67 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 231 | +6.62e-5 | +6.62e-5 | +6.62e-5 | -2.90e-5 |
| 68 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 248 | +1.36e-4 | +1.36e-4 | +1.36e-4 | -1.25e-5 |
| 69 | 3.00e-1 | 2 | 1.85e-1 | 2.00e-1 | 1.93e-1 | 1.85e-1 | 205 | -3.80e-4 | -4.42e-5 | -2.12e-4 | -5.21e-5 |
| 70 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 217 | +2.65e-4 | +2.65e-4 | +2.65e-4 | -2.04e-5 |
| 71 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 230 | +5.59e-5 | +5.59e-5 | +5.59e-5 | -1.28e-5 |
| 72 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 232 | -7.23e-5 | -7.23e-5 | -7.23e-5 | -1.87e-5 |
| 73 | 3.00e-1 | 2 | 1.86e-1 | 2.02e-1 | 1.94e-1 | 1.86e-1 | 205 | -3.95e-4 | +1.27e-4 | -1.34e-4 | -4.32e-5 |
| 74 | 3.00e-1 | 1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 1.98e-1 | 224 | +2.80e-4 | +2.80e-4 | +2.80e-4 | -1.09e-5 |
| 75 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 227 | +1.29e-5 | +1.29e-5 | +1.29e-5 | -8.51e-6 |
| 76 | 3.00e-1 | 2 | 1.89e-1 | 1.97e-1 | 1.93e-1 | 1.89e-1 | 205 | -2.04e-4 | -3.49e-5 | -1.20e-4 | -3.05e-5 |
| 77 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 243 | +2.89e-4 | +2.89e-4 | +2.89e-4 | +1.44e-6 |
| 78 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 224 | +1.05e-5 | +1.05e-5 | +1.05e-5 | +2.34e-6 |
| 79 | 3.00e-1 | 1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 2.03e-1 | 236 | -1.68e-6 | -1.68e-6 | -1.68e-6 | +1.94e-6 |
| 80 | 3.00e-1 | 1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 1.95e-1 | 227 | -1.79e-4 | -1.79e-4 | -1.79e-4 | -1.62e-5 |
| 81 | 3.00e-1 | 2 | 1.87e-1 | 2.03e-1 | 1.95e-1 | 1.87e-1 | 194 | -4.27e-4 | +1.58e-4 | -1.35e-4 | -4.16e-5 |
| 82 | 3.00e-1 | 1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 1.94e-1 | 214 | +1.95e-4 | +1.95e-4 | +1.95e-4 | -1.80e-5 |
| 83 | 3.00e-1 | 1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 2.06e-1 | 244 | +2.44e-4 | +2.44e-4 | +2.44e-4 | +8.19e-6 |
| 84 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 240 | -1.50e-4 | -1.50e-4 | -1.50e-4 | -7.59e-6 |
| 85 | 3.00e-1 | 2 | 1.85e-1 | 2.07e-1 | 1.96e-1 | 1.85e-1 | 194 | -5.60e-4 | +1.47e-4 | -2.06e-4 | -4.89e-5 |
| 86 | 3.00e-1 | 1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 1.99e-1 | 236 | +3.06e-4 | +3.06e-4 | +3.06e-4 | -1.34e-5 |
| 87 | 3.00e-1 | 1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 2.02e-1 | 236 | +5.48e-5 | +5.48e-5 | +5.48e-5 | -6.56e-6 |
| 88 | 3.00e-1 | 2 | 1.85e-1 | 1.98e-1 | 1.91e-1 | 1.85e-1 | 179 | -3.59e-4 | -9.22e-5 | -2.26e-4 | -4.95e-5 |
| 89 | 3.00e-1 | 1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 2.00e-1 | 213 | +3.58e-4 | +3.58e-4 | +3.58e-4 | -8.76e-6 |
| 90 | 3.00e-1 | 2 | 1.76e-1 | 1.93e-1 | 1.84e-1 | 1.76e-1 | 156 | -6.02e-4 | -1.84e-4 | -3.93e-4 | -8.38e-5 |
| 91 | 3.00e-1 | 1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 1.92e-1 | 192 | +4.57e-4 | +4.57e-4 | +4.57e-4 | -2.97e-5 |
| 92 | 3.00e-1 | 2 | 1.71e-1 | 1.90e-1 | 1.80e-1 | 1.71e-1 | 156 | -6.88e-4 | -5.55e-5 | -3.72e-4 | -9.78e-5 |
| 93 | 3.00e-1 | 1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 1.90e-1 | 191 | +5.62e-4 | +5.62e-4 | +5.62e-4 | -3.19e-5 |
| 94 | 3.00e-1 | 2 | 1.80e-1 | 1.91e-1 | 1.86e-1 | 1.80e-1 | 165 | -3.63e-4 | +4.15e-5 | -1.61e-4 | -5.84e-5 |
| 95 | 3.00e-1 | 1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 1.96e-1 | 202 | +4.07e-4 | +4.07e-4 | +4.07e-4 | -1.19e-5 |
| 96 | 3.00e-1 | 2 | 1.78e-1 | 1.87e-1 | 1.83e-1 | 1.78e-1 | 165 | -3.09e-4 | -2.29e-4 | -2.69e-4 | -6.12e-5 |
| 97 | 3.00e-1 | 1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 1.97e-1 | 216 | +4.59e-4 | +4.59e-4 | +4.59e-4 | -9.12e-6 |
| 98 | 3.00e-1 | 2 | 1.79e-1 | 1.97e-1 | 1.88e-1 | 1.79e-1 | 157 | -6.31e-4 | +9.66e-6 | -3.10e-4 | -6.96e-5 |
| 99 | 3.00e-1 | 2 | 1.71e-1 | 1.95e-1 | 1.83e-1 | 1.71e-1 | 145 | -9.23e-4 | +4.27e-4 | -2.48e-4 | -1.10e-4 |
| 100 | 3.00e-2 | 2 | 1.73e-2 | 8.90e-2 | 5.32e-2 | 1.73e-2 | 145 | -1.13e-2 | -4.01e-3 | -7.66e-3 | -1.58e-3 |
| 101 | 3.00e-2 | 1 | 1.98e-2 | 1.98e-2 | 1.98e-2 | 1.98e-2 | 186 | +7.37e-4 | +7.37e-4 | +7.37e-4 | -1.35e-3 |
| 102 | 3.00e-2 | 2 | 1.90e-2 | 2.12e-2 | 2.01e-2 | 1.90e-2 | 145 | -7.62e-4 | +3.70e-4 | -1.96e-4 | -1.14e-3 |
| 103 | 3.00e-2 | 1 | 2.12e-2 | 2.12e-2 | 2.12e-2 | 2.12e-2 | 183 | +5.89e-4 | +5.89e-4 | +5.89e-4 | -9.63e-4 |
| 104 | 3.00e-2 | 3 | 1.99e-2 | 2.26e-2 | 2.08e-2 | 2.01e-2 | 134 | -9.37e-4 | +3.26e-4 | -1.81e-4 | -7.53e-4 |
| 105 | 3.00e-2 | 1 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 2.21e-2 | 165 | +5.77e-4 | +5.77e-4 | +5.77e-4 | -6.20e-4 |
| 106 | 3.00e-2 | 2 | 2.13e-2 | 2.35e-2 | 2.24e-2 | 2.13e-2 | 142 | -7.14e-4 | +3.48e-4 | -1.83e-4 | -5.42e-4 |
| 107 | 3.00e-2 | 2 | 2.10e-2 | 2.26e-2 | 2.18e-2 | 2.10e-2 | 128 | -5.88e-4 | +4.06e-4 | -9.10e-5 | -4.62e-4 |
| 108 | 3.00e-2 | 2 | 2.14e-2 | 2.34e-2 | 2.24e-2 | 2.14e-2 | 125 | -7.20e-4 | +7.11e-4 | -4.50e-6 | -3.82e-4 |
| 109 | 3.00e-2 | 2 | 2.22e-2 | 2.44e-2 | 2.33e-2 | 2.22e-2 | 125 | -7.55e-4 | +7.41e-4 | -6.73e-6 | -3.18e-4 |
| 110 | 3.00e-2 | 2 | 2.22e-2 | 2.38e-2 | 2.30e-2 | 2.22e-2 | 125 | -5.74e-4 | +4.55e-4 | -5.94e-5 | -2.74e-4 |
| 111 | 3.00e-2 | 2 | 2.32e-2 | 2.66e-2 | 2.49e-2 | 2.32e-2 | 132 | -1.02e-3 | +1.08e-3 | +3.00e-5 | -2.27e-4 |
| 112 | 3.00e-2 | 2 | 2.51e-2 | 2.66e-2 | 2.58e-2 | 2.51e-2 | 132 | -4.39e-4 | +7.98e-4 | +1.79e-4 | -1.56e-4 |
| 113 | 3.00e-2 | 2 | 2.46e-2 | 2.65e-2 | 2.56e-2 | 2.46e-2 | 123 | -6.06e-4 | +3.41e-4 | -1.32e-4 | -1.56e-4 |
| 114 | 3.00e-2 | 2 | 2.38e-2 | 2.66e-2 | 2.52e-2 | 2.38e-2 | 114 | -9.48e-4 | +4.80e-4 | -2.34e-4 | -1.78e-4 |
| 115 | 3.00e-2 | 2 | 2.40e-2 | 2.70e-2 | 2.55e-2 | 2.40e-2 | 112 | -1.03e-3 | +8.30e-4 | -1.02e-4 | -1.73e-4 |
| 116 | 3.00e-2 | 3 | 2.26e-2 | 2.71e-2 | 2.43e-2 | 2.26e-2 | 96 | -1.61e-3 | +8.01e-4 | -3.57e-4 | -2.32e-4 |
| 117 | 3.00e-2 | 2 | 2.27e-2 | 2.57e-2 | 2.42e-2 | 2.27e-2 | 96 | -1.27e-3 | +1.03e-3 | -1.21e-4 | -2.23e-4 |
| 118 | 3.00e-2 | 3 | 2.44e-2 | 2.65e-2 | 2.52e-2 | 2.46e-2 | 107 | -7.80e-4 | +1.19e-3 | +1.66e-4 | -1.28e-4 |
| 119 | 3.00e-2 | 2 | 2.64e-2 | 2.95e-2 | 2.79e-2 | 2.64e-2 | 107 | -1.03e-3 | +1.30e-3 | +1.36e-4 | -8.93e-5 |
| 120 | 3.00e-2 | 4 | 2.36e-2 | 2.73e-2 | 2.50e-2 | 2.41e-2 | 89 | -8.62e-4 | +2.64e-4 | -2.58e-4 | -1.46e-4 |
| 121 | 3.00e-2 | 2 | 2.36e-2 | 2.74e-2 | 2.55e-2 | 2.36e-2 | 76 | -1.97e-3 | +1.15e-3 | -4.12e-4 | -2.12e-4 |
| 122 | 3.00e-2 | 3 | 2.27e-2 | 2.68e-2 | 2.41e-2 | 2.27e-2 | 76 | -2.14e-3 | +1.13e-3 | -3.44e-4 | -2.58e-4 |
| 123 | 3.00e-2 | 3 | 2.46e-2 | 2.82e-2 | 2.61e-2 | 2.54e-2 | 83 | -1.66e-3 | +1.77e-3 | +1.59e-4 | -1.57e-4 |
| 124 | 3.00e-2 | 3 | 2.45e-2 | 2.99e-2 | 2.63e-2 | 2.46e-2 | 79 | -2.51e-3 | +1.34e-3 | -3.74e-4 | -2.27e-4 |
| 125 | 3.00e-2 | 3 | 2.40e-2 | 2.98e-2 | 2.62e-2 | 2.49e-2 | 74 | -2.86e-3 | +1.46e-3 | -3.14e-4 | -2.59e-4 |
| 126 | 3.00e-2 | 5 | 2.37e-2 | 2.94e-2 | 2.51e-2 | 2.39e-2 | 69 | -2.92e-3 | +1.57e-3 | -2.45e-4 | -2.57e-4 |
| 127 | 3.00e-2 | 3 | 2.33e-2 | 3.00e-2 | 2.56e-2 | 2.35e-2 | 64 | -3.90e-3 | +2.11e-3 | -5.50e-4 | -3.53e-4 |
| 128 | 3.00e-2 | 4 | 2.11e-2 | 2.96e-2 | 2.41e-2 | 2.11e-2 | 53 | -3.54e-3 | +2.37e-3 | -7.78e-4 | -5.29e-4 |
| 129 | 3.00e-2 | 5 | 2.09e-2 | 2.80e-2 | 2.31e-2 | 2.28e-2 | 53 | -5.55e-3 | +3.09e-3 | -1.69e-4 | -3.68e-4 |
| 130 | 3.00e-2 | 7 | 1.77e-2 | 3.16e-2 | 2.13e-2 | 1.77e-2 | 38 | -7.48e-3 | +3.07e-3 | -1.38e-3 | -8.87e-4 |
| 131 | 3.00e-2 | 4 | 1.96e-2 | 2.74e-2 | 2.20e-2 | 1.96e-2 | 40 | -7.38e-3 | +5.30e-3 | -6.93e-4 | -8.70e-4 |
| 132 | 3.00e-2 | 6 | 1.68e-2 | 2.60e-2 | 2.05e-2 | 2.08e-2 | 46 | -1.24e-2 | +4.05e-3 | -5.09e-4 | -6.23e-4 |
| 133 | 3.00e-2 | 9 | 1.77e-2 | 2.89e-2 | 2.11e-2 | 1.77e-2 | 31 | -9.69e-3 | +3.51e-3 | -1.05e-3 | -9.34e-4 |
| 134 | 3.00e-2 | 4 | 1.66e-2 | 2.68e-2 | 1.93e-2 | 1.67e-2 | 29 | -1.55e-2 | +5.53e-3 | -2.70e-3 | -1.54e-3 |
| 135 | 3.00e-2 | 8 | 1.65e-2 | 2.77e-2 | 1.96e-2 | 1.78e-2 | 31 | -1.63e-2 | +6.86e-3 | -8.88e-4 | -1.13e-3 |
| 136 | 3.00e-2 | 7 | 1.64e-2 | 2.69e-2 | 1.91e-2 | 2.35e-2 | 49 | -1.75e-2 | +6.47e-3 | -6.38e-4 | -5.42e-4 |
| 137 | 3.00e-2 | 6 | 2.04e-2 | 3.15e-2 | 2.39e-2 | 2.49e-2 | 49 | -8.07e-3 | +3.53e-3 | -2.85e-4 | -3.25e-4 |
| 138 | 3.00e-2 | 7 | 1.85e-2 | 3.20e-2 | 2.18e-2 | 2.20e-2 | 43 | -7.33e-3 | +4.12e-3 | -1.05e-3 | -5.12e-4 |
| 139 | 3.00e-2 | 6 | 2.41e-2 | 3.16e-2 | 2.62e-2 | 2.61e-2 | 50 | -5.42e-3 | +4.21e-3 | +5.67e-5 | -2.44e-4 |
| 140 | 3.00e-2 | 4 | 2.35e-2 | 3.62e-2 | 2.70e-2 | 2.44e-2 | 46 | -9.28e-3 | +3.41e-3 | -1.38e-3 | -6.24e-4 |
| 141 | 3.00e-2 | 6 | 2.25e-2 | 3.22e-2 | 2.50e-2 | 2.37e-2 | 42 | -8.08e-3 | +3.34e-3 | -5.94e-4 | -5.84e-4 |
| 142 | 3.00e-2 | 6 | 2.01e-2 | 3.21e-2 | 2.34e-2 | 2.40e-2 | 43 | -1.01e-2 | +3.87e-3 | -7.17e-4 | -5.14e-4 |
| 143 | 3.00e-2 | 6 | 2.25e-2 | 3.62e-2 | 2.51e-2 | 2.27e-2 | 40 | -1.19e-2 | +4.32e-3 | -1.24e-3 | -7.99e-4 |
| 144 | 3.00e-2 | 8 | 2.19e-2 | 3.42e-2 | 2.45e-2 | 2.46e-2 | 42 | -7.02e-3 | +4.81e-3 | -3.50e-4 | -4.31e-4 |
| 145 | 3.00e-2 | 5 | 2.21e-2 | 3.47e-2 | 2.53e-2 | 2.27e-2 | 36 | -1.23e-2 | +4.00e-3 | -1.58e-3 | -9.16e-4 |
| 146 | 3.00e-2 | 9 | 2.02e-2 | 3.36e-2 | 2.36e-2 | 2.02e-2 | 32 | -8.83e-3 | +4.87e-3 | -1.03e-3 | -1.01e-3 |
| 147 | 3.00e-2 | 5 | 2.25e-2 | 3.19e-2 | 2.49e-2 | 2.33e-2 | 34 | -9.82e-3 | +6.55e-3 | -5.58e-4 | -8.39e-4 |
| 148 | 3.00e-2 | 7 | 1.94e-2 | 3.71e-2 | 2.49e-2 | 2.61e-2 | 44 | -1.59e-2 | +5.57e-3 | -9.31e-4 | -6.71e-4 |
| 149 | 3.00e-3 | 7 | 2.41e-3 | 3.65e-2 | 2.45e-2 | 2.41e-3 | 37 | -6.54e-2 | +4.03e-3 | -9.74e-3 | -7.03e-3 |
| 150 | 3.00e-3 | 5 | 2.19e-3 | 3.56e-3 | 2.56e-3 | 2.31e-3 | 40 | -1.32e-2 | +5.20e-3 | -1.30e-3 | -4.71e-3 |
| 151 | 3.00e-3 | 9 | 2.03e-3 | 3.33e-3 | 2.46e-3 | 2.03e-3 | 34 | -7.20e-3 | +4.88e-3 | -8.87e-4 | -2.50e-3 |
| 152 | 3.00e-3 | 5 | 1.75e-3 | 2.88e-3 | 2.03e-3 | 1.85e-3 | 30 | -1.82e-2 | +4.97e-3 | -2.34e-3 | -2.35e-3 |
| 153 | 3.00e-3 | 8 | 1.95e-3 | 2.92e-3 | 2.23e-3 | 2.04e-3 | 34 | -8.04e-3 | +7.02e-3 | -4.89e-4 | -1.32e-3 |
| 154 | 3.00e-3 | 7 | 1.90e-3 | 3.11e-3 | 2.27e-3 | 2.80e-3 | 55 | -1.35e-2 | +6.16e-3 | -3.23e-4 | -5.48e-4 |
| 155 | 3.00e-3 | 7 | 2.26e-3 | 3.27e-3 | 2.56e-3 | 2.36e-3 | 44 | -4.09e-3 | +3.18e-3 | -7.91e-4 | -6.37e-4 |
| 156 | 3.00e-3 | 4 | 2.44e-3 | 3.77e-3 | 2.84e-3 | 2.60e-3 | 48 | -8.45e-3 | +4.68e-3 | -9.22e-4 | -7.40e-4 |
| 157 | 3.00e-3 | 5 | 2.24e-3 | 3.64e-3 | 2.72e-3 | 2.24e-3 | 41 | -5.48e-3 | +3.73e-3 | -1.31e-3 | -1.03e-3 |
| 158 | 3.00e-3 | 8 | 2.09e-3 | 3.38e-3 | 2.57e-3 | 2.39e-3 | 42 | -1.17e-2 | +5.02e-3 | -5.02e-4 | -7.21e-4 |
| 159 | 3.00e-3 | 4 | 2.14e-3 | 3.30e-3 | 2.60e-3 | 2.14e-3 | 41 | -6.18e-3 | +4.04e-3 | -1.56e-3 | -1.08e-3 |
| 160 | 3.00e-3 | 7 | 2.06e-3 | 3.10e-3 | 2.37e-3 | 2.25e-3 | 38 | -4.85e-3 | +4.93e-3 | -4.47e-4 | -7.32e-4 |
| 161 | 3.00e-3 | 6 | 2.16e-3 | 3.33e-3 | 2.45e-3 | 2.21e-3 | 33 | -7.90e-3 | +4.99e-3 | -9.81e-4 | -8.41e-4 |
| 162 | 3.00e-3 | 7 | 1.96e-3 | 3.17e-3 | 2.47e-3 | 2.69e-3 | 43 | -1.72e-2 | +4.66e-3 | -6.69e-4 | -5.58e-4 |
| 163 | 3.00e-3 | 6 | 2.28e-3 | 3.98e-3 | 2.69e-3 | 2.61e-3 | 44 | -1.31e-2 | +4.18e-3 | -1.17e-3 | -7.19e-4 |
| 164 | 3.00e-3 | 8 | 2.29e-3 | 3.52e-3 | 2.57e-3 | 2.52e-3 | 42 | -7.61e-3 | +3.62e-3 | -5.28e-4 | -5.09e-4 |
| 165 | 3.00e-3 | 5 | 2.08e-3 | 3.61e-3 | 2.49e-3 | 2.25e-3 | 37 | -1.62e-2 | +4.35e-3 | -1.91e-3 | -1.01e-3 |
| 166 | 3.00e-3 | 6 | 2.11e-3 | 3.38e-3 | 2.54e-3 | 2.31e-3 | 42 | -8.17e-3 | +5.52e-3 | -7.42e-4 | -8.92e-4 |
| 167 | 3.00e-3 | 7 | 2.00e-3 | 3.66e-3 | 2.46e-3 | 2.20e-3 | 31 | -7.87e-3 | +4.94e-3 | -1.09e-3 | -9.31e-4 |
| 168 | 3.00e-3 | 9 | 2.06e-3 | 3.64e-3 | 2.42e-3 | 2.29e-3 | 35 | -1.30e-2 | +6.55e-3 | -8.11e-4 | -7.31e-4 |
| 169 | 3.00e-3 | 4 | 2.47e-3 | 3.35e-3 | 2.77e-3 | 2.70e-3 | 49 | -8.73e-3 | +4.89e-3 | -4.71e-4 | -6.42e-4 |
| 170 | 3.00e-3 | 7 | 2.20e-3 | 3.61e-3 | 2.50e-3 | 2.23e-3 | 35 | -8.48e-3 | +3.29e-3 | -1.24e-3 | -8.82e-4 |
| 171 | 3.00e-3 | 7 | 2.30e-3 | 3.65e-3 | 2.58e-3 | 2.30e-3 | 35 | -8.52e-3 | +5.61e-3 | -8.04e-4 | -8.38e-4 |
| 172 | 3.00e-3 | 6 | 2.02e-3 | 3.56e-3 | 2.50e-3 | 2.02e-3 | 33 | -1.23e-2 | +4.52e-3 | -1.64e-3 | -1.33e-3 |
| 173 | 3.00e-3 | 7 | 2.17e-3 | 3.31e-3 | 2.48e-3 | 2.42e-3 | 38 | -1.29e-2 | +6.67e-3 | -4.85e-4 | -8.36e-4 |
| 174 | 3.00e-3 | 6 | 2.31e-3 | 3.68e-3 | 2.64e-3 | 2.38e-3 | 40 | -9.55e-3 | +4.74e-3 | -1.13e-3 | -9.48e-4 |
| 175 | 3.00e-3 | 7 | 2.01e-3 | 3.67e-3 | 2.52e-3 | 2.26e-3 | 35 | -8.32e-3 | +5.36e-3 | -1.10e-3 | -9.67e-4 |
| 176 | 3.00e-3 | 8 | 2.16e-3 | 3.47e-3 | 2.56e-3 | 2.67e-3 | 46 | -1.22e-2 | +5.56e-3 | -4.28e-4 | -5.63e-4 |
| 177 | 3.00e-3 | 4 | 2.63e-3 | 3.75e-3 | 3.02e-3 | 3.01e-3 | 49 | -7.61e-3 | +3.98e-3 | -3.80e-4 | -4.78e-4 |
| 178 | 3.00e-3 | 8 | 2.31e-3 | 3.83e-3 | 2.62e-3 | 2.58e-3 | 41 | -1.17e-2 | +2.77e-3 | -9.87e-4 | -5.77e-4 |
| 179 | 3.00e-3 | 5 | 2.29e-3 | 3.73e-3 | 2.76e-3 | 2.29e-3 | 35 | -8.85e-3 | +4.40e-3 | -1.63e-3 | -1.07e-3 |
| 180 | 3.00e-3 | 6 | 2.12e-3 | 3.57e-3 | 2.73e-3 | 2.63e-3 | 44 | -1.49e-2 | +6.26e-3 | -6.62e-4 | -8.50e-4 |
| 181 | 3.00e-3 | 7 | 2.73e-3 | 3.97e-3 | 2.97e-3 | 2.73e-3 | 44 | -8.48e-3 | +4.97e-3 | -5.13e-4 | -6.79e-4 |
| 182 | 3.00e-3 | 5 | 2.23e-3 | 3.80e-3 | 2.74e-3 | 2.34e-3 | 37 | -7.85e-3 | +3.72e-3 | -1.70e-3 | -1.09e-3 |
| 183 | 3.00e-3 | 6 | 2.23e-3 | 3.54e-3 | 2.70e-3 | 2.74e-3 | 42 | -1.25e-2 | +5.76e-3 | -3.08e-4 | -6.62e-4 |
| 184 | 3.00e-3 | 10 | 2.19e-3 | 3.73e-3 | 2.52e-3 | 2.23e-3 | 35 | -8.00e-3 | +4.11e-3 | -9.00e-4 | -7.39e-4 |
| 185 | 3.00e-3 | 4 | 2.21e-3 | 3.86e-3 | 2.78e-3 | 2.39e-3 | 36 | -9.40e-3 | +6.60e-3 | -1.49e-3 | -1.03e-3 |
| 186 | 3.00e-3 | 7 | 2.09e-3 | 3.65e-3 | 2.51e-3 | 2.44e-3 | 37 | -1.86e-2 | +5.95e-3 | -1.30e-3 | -9.90e-4 |
| 187 | 3.00e-3 | 7 | 2.34e-3 | 3.60e-3 | 2.63e-3 | 2.41e-3 | 35 | -1.02e-2 | +4.95e-3 | -8.65e-4 | -8.88e-4 |
| 188 | 3.00e-3 | 7 | 2.01e-3 | 3.57e-3 | 2.47e-3 | 2.37e-3 | 35 | -1.17e-2 | +4.47e-3 | -9.94e-4 | -7.96e-4 |
| 189 | 3.00e-3 | 6 | 2.36e-3 | 4.18e-3 | 2.93e-3 | 2.65e-3 | 37 | -1.05e-2 | +6.10e-3 | -8.12e-4 | -7.77e-4 |
| 190 | 3.00e-3 | 7 | 2.41e-3 | 3.48e-3 | 2.75e-3 | 2.41e-3 | 41 | -8.11e-3 | +3.42e-3 | -8.47e-4 | -8.46e-4 |
| 191 | 3.00e-3 | 5 | 2.55e-3 | 3.56e-3 | 2.82e-3 | 2.67e-3 | 41 | -8.12e-3 | +5.46e-3 | -3.06e-4 | -6.40e-4 |
| 192 | 3.00e-3 | 7 | 2.21e-3 | 3.85e-3 | 2.63e-3 | 2.64e-3 | 39 | -8.19e-3 | +4.51e-3 | -7.12e-4 | -5.11e-4 |
| 193 | 3.00e-3 | 7 | 2.15e-3 | 4.01e-3 | 2.59e-3 | 2.60e-3 | 36 | -1.95e-2 | +4.56e-3 | -1.35e-3 | -6.93e-4 |
| 194 | 3.00e-3 | 7 | 2.24e-3 | 3.78e-3 | 2.58e-3 | 2.42e-3 | 34 | -1.53e-2 | +4.98e-3 | -1.20e-3 | -8.31e-4 |
| 195 | 3.00e-3 | 8 | 2.18e-3 | 4.03e-3 | 2.54e-3 | 2.18e-3 | 32 | -1.63e-2 | +6.29e-3 | -1.49e-3 | -1.13e-3 |
| 196 | 3.00e-3 | 8 | 2.15e-3 | 3.70e-3 | 2.75e-3 | 3.38e-3 | 55 | -1.63e-2 | +7.55e-3 | -2.47e-4 | -4.32e-4 |
| 197 | 3.00e-3 | 3 | 3.23e-3 | 4.24e-3 | 3.60e-3 | 3.35e-3 | 58 | -4.97e-3 | +2.35e-3 | -6.65e-4 | -5.09e-4 |
| 198 | 3.00e-3 | 5 | 2.95e-3 | 4.67e-3 | 3.34e-3 | 3.03e-3 | 46 | -9.15e-3 | +3.50e-3 | -1.03e-3 | -6.97e-4 |
| 199 | 3.00e-3 | 5 | 2.64e-3 | 4.00e-3 | 3.13e-3 | 3.14e-3 | 46 | -5.99e-3 | +3.12e-3 | -4.26e-4 | -5.29e-4 |

